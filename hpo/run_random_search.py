import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import optuna
from optuna.samplers import RandomSampler
from optuna.pruners import HyperbandPruner

HPO_DIR = Path(__file__).resolve().parent.parent / "hpo"
sys.path.insert(0, str(HPO_DIR))

import run_hpo as R
from run_hpo import (
    REPO_ROOT, ONLAB_DIR, BASELINE_MODELS_DIR,
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    CLASS_NAME, BASELINE_ARCH_KWARGS, HAS_DROPOUT, TRIAL_BUDGET, ARCH_ORDER,
    SEARCH_MAX_EPOCHS, SEARCH_PATIENCE, SEARCH_BATCH_SIZE,
    FINAL_MAX_EPOCHS, FINAL_BATCH_SIZE, BASE_SEED, DEVICE,
    build_trial_hparams, train_for_search, get_datasets, baseline_val_loss,
    get_model, count_parameters, train_model, evaluate_model,
    DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

FINAL_PATIENCE = 30

N_PAIRED_STARTUP = 10

TPE_RESULTS_DIR = HPO_DIR / "results"
TPE_P30_RESULTS_DIR = HPO_DIR / "results" / "p30"
PROTECTED_DIRS = [ONLAB_DIR]


def rel(path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def cwd_rel(path) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return str(path)


class Paths:
    def __init__(self, out_root: Path):
        self.out_root = out_root.resolve()
        self.studies_dir = self.out_root / "optuna_studies" / "random"
        self.models_dir = self.out_root / "models" / "random"
        self.results_dir = self.out_root / "results" / "random"
        self.logs_dir = self.out_root / "logs" / "random"

    @property
    def all_dirs(self):
        return (self.studies_dir, self.models_dir, self.results_dir, self.logs_dir)

    def ensure(self):
        for d in self.all_dirs:
            guard_path(d)
            d.mkdir(parents=True, exist_ok=True)


P = Paths(HPO_DIR)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def guard_path(path) -> Path:
    path = Path(path).resolve()
    for d in PROTECTED_DIRS:
        if _is_within(path, d):
            raise PermissionError(f"[guard] Vedett mappa, nem irhato: {rel(path)}")
    if not any(_is_within(path, d) for d in P.all_dirs):
        raise PermissionError(f"[guard] Nem a random/ kimenetek kozott, nem irhato: {rel(path)}")
    return path


class _Tee:
    def __init__(self, stream, f):
        self.stream = stream
        self.f = f

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()


def log(arch: str, msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(guard_path(P.logs_dir / f"{arch}.log"), 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def write_json(path: Path, obj):
    with open(guard_path(path), 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


CURVE_COLS = ['trial_number', 'epoch', 'train_loss', 'val_loss', 'lr']


def _append_curve_rows(arch: str, rows: list):
    if not rows:
        return
    curve_path = guard_path(P.results_dir / f"search_curves_{arch}.csv")
    df = pd.DataFrame(rows, columns=CURVE_COLS)
    write_header = not curve_path.exists()
    df.to_csv(curve_path, mode='a', header=write_header, index=False)


def make_objective_random(arch: str, search_curves: list, search_max_epochs: int):
    train_ds, val_ds = get_datasets()

    def objective(trial: 'optuna.Trial'):
        torch.manual_seed(BASE_SEED + trial.number)
        lr, weight_decay, dropout, model_kwargs = build_trial_hparams(trial, arch)
        model = get_model(arch, **model_kwargs)
        start_idx = len(search_curves)
        try:
            best_val = train_for_search(
                model, train_ds, val_ds, lr, weight_decay, trial,
                max_epochs=search_max_epochs, patience=SEARCH_PATIENCE,
                batch_size=SEARCH_BATCH_SIZE, device=DEVICE,
                curve_sink=search_curves,
            )
            return best_val
        finally:
            _append_curve_rows(arch, search_curves[start_idx:])

    return objective


def _valid_trial_duration_sum_s(trials_df: pd.DataFrame):
    if 'duration' not in trials_df or 'state' not in trials_df:
        return None
    valid = trials_df[trials_df['state'].isin(['COMPLETE', 'PRUNED'])]
    return float(pd.to_timedelta(valid['duration']).dt.total_seconds().sum())


def _fast_forward_sampler(sampler: RandomSampler, arch: str, n_trials: int) -> int:
    if n_trials <= 0:
        return 0
    tmp = optuna.create_study(direction='minimize', sampler=sampler)

    def _draw_only(trial):
        build_trial_hparams(trial, arch)
        return 0.0

    tmp.optimize(_draw_only, n_trials=n_trials, show_progress_bar=False)
    return n_trials


def run_study_random(arch: str, n_trials_target: int, search_max_epochs: int):
    log(arch, f"=== {arch} ({CLASS_NAME[arch]}) VELETLEN KERESES indul -- device={DEVICE}, "
              f"trial-cel={n_trials_target}, fix meret={BASELINE_ARCH_KWARGS[arch]}, "
              f"dropout kereshetos={HAS_DROPOUT[arch]}, kereses max_epochs={search_max_epochs} ===")

    db_path = guard_path(P.studies_dir / f"{arch}.db")
    storage = f"sqlite:///{db_path.as_posix()}"
    sampler = RandomSampler(seed=BASE_SEED)
    pruner = HyperbandPruner(min_resource=3, max_resource=SEARCH_MAX_EPOCHS, reduction_factor=3)

    study = optuna.create_study(
        study_name=arch, storage=storage, direction='minimize',
        sampler=sampler, pruner=pruner, load_if_exists=True,
    )
    study.set_user_attr('sampler', type(study.sampler).__name__)
    study.set_user_attr('sampler_seed', BASE_SEED)
    log(arch, f"Sampler: {type(study.sampler).__name__}(seed={BASE_SEED}), "
              f"pruner: {type(study.pruner).__name__}(min_resource=3, "
              f"max_resource={SEARCH_MAX_EPOCHS}, reduction_factor=3), storage: {rel(db_path)}")

    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            log(arch, f"Elakadt (stale) RUNNING trial #{t.number} FAIL-re allitva "
                      f"(korabban megszakadt folyamatbol maradt vissza).")
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)

    n_valid = sum(
        1 for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    )
    n_remaining = max(0, n_trials_target - n_valid)
    if len(study.trials) > 0:
        log(arch, f"Folytatas: {n_valid} ervenyes trial mar letezik a storage-ban, "
                  f"{n_remaining} tovabbi trial hianyzik a celhoz.")
        n_ff = _fast_forward_sampler(sampler, arch, len(study.trials))
        log(arch, f"RandomSampler generatora {n_ff} trial mintaival eloreleptetve "
                  f"(a megszakitatlan futas mintasorozatanak folytatasa).")

    search_curves = []
    prev_search_s = float(study.user_attrs.get('search_time_s_cumulative', 0.0))
    t0 = time.perf_counter()

    def _persist_search_time(st, _trial):
        st.set_user_attr('search_time_s_cumulative', prev_search_s + time.perf_counter() - t0)

    if n_remaining > 0:
        study.optimize(make_objective_random(arch, search_curves, search_max_epochs),
                       n_trials=n_remaining, show_progress_bar=False,
                       callbacks=[_persist_search_time])
    dt = time.perf_counter() - t0
    search_time_s = prev_search_s + dt
    study.set_user_attr('search_time_s_cumulative', search_time_s)
    n_runs = int(study.user_attrs.get('n_search_invocations', 0)) + (1 if n_remaining > 0 else 0)
    study.set_user_attr('n_search_invocations', n_runs)

    log(arch, f"Study kesz: {len(study.trials)} trial osszesen ({dt:.1f}s ebben a futasban, "
              f"halmozott keresesi ido {search_time_s:.1f}s). "
              f"Legjobb val_loss={study.best_value:.8e}, trial #{study.best_trial.number}, "
              f"params={study.best_trial.params}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(guard_path(P.results_dir / f"trials_{arch}.csv"), index=False)

    curve_path = P.results_dir / f"search_curves_{arch}.csv"
    if curve_path.exists():
        curve_df = pd.read_csv(curve_path)
        curve_df = curve_df.drop_duplicates(subset=['trial_number', 'epoch'], keep='last')
        curve_df = curve_df.sort_values(['trial_number', 'epoch'])
        curve_df.to_csv(guard_path(curve_path), index=False)

    try:
        importances = optuna.importance.get_param_importances(study)
        importances = {k: float(v) for k, v in importances.items()}
    except Exception as e:
        importances = {"error": str(e)}
    write_json(P.results_dir / f"param_importance_{arch}.json", importances)

    timing = {
        'search_time_s': search_time_s,
        'search_time_s_this_run': dt,
        'n_search_invocations': n_runs,
        'search_trial_duration_sum_s': _valid_trial_duration_sum_s(trials_df),
    }
    return study, trials_df, importances, timing


def paired_startup_check(arch: str, rs_df: pd.DataFrame):
    tpe_path = TPE_RESULTS_DIR / f"trials_{arch}.csv"
    if not tpe_path.exists():
        return {'error': f"nincs {rel(tpe_path)}"}
    tpe = pd.read_csv(tpe_path, float_precision='round_trip')
    rs = rs_df.copy()
    pcols = [c for c in tpe.columns if c.startswith('params_')]
    n = min(N_PAIRED_STARTUP, len(tpe), len(rs))
    per_trial = []
    for i in range(n):
        a, b = tpe.iloc[i], rs.iloc[i]
        params_eq = all(float(a[c]) == float(b[c]) for c in pcols)
        state_eq = a['state'] == b['state']
        val_eq = (pd.isna(a['value']) and pd.isna(b['value'])) or (float(a['value']) == float(b['value']))
        per_trial.append({'trial': int(i), 'params_equal': bool(params_eq),
                          'state_equal': bool(state_eq), 'value_equal': bool(val_eq),
                          'tpe_state': a['state'], 'rs_state': b['state']})
    n_after = min(len(tpe), len(rs))
    n_params_equal_after = sum(
        1 for i in range(n, n_after)
        if all(float(tpe.iloc[i][c]) == float(rs.iloc[i][c]) for c in pcols))
    return {
        'n_checked': n,
        'n_params_equal': sum(t['params_equal'] for t in per_trial),
        'n_state_equal': sum(t['state_equal'] for t in per_trial),
        'n_value_equal': sum(t['value_equal'] for t in per_trial),
        'all_identical': all(t['params_equal'] and t['state_equal'] and t['value_equal']
                             for t in per_trial),
        'n_params_equal_after_startup': n_params_equal_after,
        'per_trial': per_trial,
    }


def final_retrain_p30(arch: str, hp: dict, max_epochs: int):
    model_kwargs = dict(BASELINE_ARCH_KWARGS[arch])
    dropout = hp.get('dropout')
    if dropout is not None:
        model_kwargs['dropout'] = dropout

    model_name = f"{arch}_random_p30"
    ckpt_path = guard_path(P.models_dir / f"{model_name}_best.pt")

    log(arch, f"Vegso tanitas indul (max_epochs={max_epochs}, patience={FINAL_PATIENCE}, "
              f"batch={FINAL_BATCH_SIZE}, seed={BASE_SEED}) lr={hp['lr']!r}, "
              f"weight_decay={hp['weight_decay']!r}, dropout={dropout!r}, "
              f"model_kwargs={model_kwargs}")

    torch.manual_seed(BASE_SEED)
    model = get_model(arch, **model_kwargs)

    t0 = time.perf_counter()
    history = train_model(
        model=model,
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        output_dir=cwd_rel(P.models_dir),
        model_name=model_name,
        model_class=arch,
        model_kwargs=model_kwargs,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        batch_size=FINAL_BATCH_SIZE,
        max_epochs=max_epochs,
        lr=hp['lr'],
        weight_decay=hp['weight_decay'],
        patience=FINAL_PATIENCE,
        device=DEVICE,
    )
    dt = time.perf_counter() - t0
    n_run = len(history['val_loss'])
    log(arch, f"Vegso tanitas kesz ({dt:.1f}s): best_epoch={history['best_epoch']}, "
              f"futott epoch={n_run}, best_val_loss={history['best_val_loss']:.8e}")

    hist_df = pd.DataFrame({
        'epoch': list(range(1, n_run + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'lr': history['lr'],
    })
    hist_df.to_csv(guard_path(P.results_dir / f"final_training_history_{arch}.csv"), index=False)
    return history, model_kwargs, dt, ckpt_path


def param_count_check_p30(arch: str, model_kwargs: dict):
    baseline_ckpt_path = BASELINE_MODELS_DIR / f"{arch}_best.pt"
    baseline_ckpt = torch.load(baseline_ckpt_path, map_location='cpu', weights_only=False)
    baseline_count = count_parameters(get_model(baseline_ckpt['model_class'],
                                                **baseline_ckpt['model_kwargs']))
    rs_count = count_parameters(get_model(arch, **model_kwargs))
    result = {
        'arch': arch,
        'baseline_checkpoint': rel(baseline_ckpt_path),
        'baseline_model_kwargs': baseline_ckpt['model_kwargs'],
        'random_model_kwargs': model_kwargs,
        'baseline_param_count': baseline_count,
        'random_param_count': rs_count,
        'match': baseline_count == rs_count,
    }
    write_json(P.results_dir / f"param_count_check_{arch}.json", result)
    log(arch, f"Parameterszam-ellenorzes: baseline={baseline_count:,} vs random_p30={rs_count:,} "
              f"-> match={result['match']}")
    if not result['match']:
        raise RuntimeError(f"[{arch}] PARAMETERSZAM ELTERES! baseline={baseline_count} "
                           f"!= random_p30={rs_count}.")
    return result


def safe_metrics(checkpoint_path: Path, data_path: str):
    try:
        res = evaluate_model(str(checkpoint_path), data_path, device=DEVICE, verbose=False)
        return res['metrics']
    except Exception as e:
        return {'error': str(e)}


def tpe_reference(arch: str):
    out = {}
    p10 = TPE_RESULTS_DIR / f"best_params_{arch}.json"
    p30 = TPE_P30_RESULTS_DIR / f"best_params_{arch}.json"
    if p10.exists():
        d = read_json(p10)
        out.update({
            'tpe_search_best_val_loss': d.get('search_best_val_loss'),
            'tpe_search_best_trial_number': d.get('search_best_trial_number'),
            'tpe_lr': d.get('lr'), 'tpe_weight_decay': d.get('weight_decay'),
            'tpe_dropout': d.get('dropout'),
        })
    if p30.exists():
        d = read_json(p30)
        out.update({
            'tpe_p30_best_val_loss': d.get('final_retrain_best_val_loss'),
            'tpe_p30_best_epoch': d.get('final_retrain_best_epoch'),
            'tpe_p30_n_epochs_run': d.get('final_retrain_n_epochs_run'),
            'tpe_p30_test_metrics': d.get('hpo_test_metrics'),
            'tpe_p30_retrain_seconds': d.get('retrain_seconds'),
        })
    tpe_trials = TPE_RESULTS_DIR / f"trials_{arch}.csv"
    if tpe_trials.exists():
        t = pd.read_csv(tpe_trials)
        out['tpe_n_complete'] = int((t['state'] == 'COMPLETE').sum())
        out['tpe_n_pruned'] = int((t['state'] == 'PRUNED').sum())
        out['tpe_search_trial_duration_sum_s'] = float(
            pd.to_timedelta(t['duration']).dt.total_seconds().sum())
    return out


def run_arch(arch: str, n_trials_target: int, search_max_epochs: int, final_max_epochs: int):
    t_start = time.perf_counter()
    study, trials_df, importances, timing = run_study_random(arch, n_trials_target, search_max_epochs)

    best_trial = study.best_trial
    hp = dict(best_trial.params)
    hp.setdefault('dropout', None)

    history, model_kwargs, final_train_s, ckpt = final_retrain_p30(arch, hp, final_max_epochs)
    pcc = param_count_check_p30(arch, model_kwargs)

    stored = torch.load(ckpt, map_location='cpu', weights_only=False)
    val_metrics = safe_metrics(ckpt, VAL_PATH)
    val_recheck = {
        'checkpoint_val_loss': stored['val_loss'],
        'checkpoint_best_epoch': stored['best_epoch'],
        'evaluate_model_val_mse': (val_metrics['RMSE'] ** 2) if 'RMSE' in val_metrics else None,
    }
    if val_recheck['evaluate_model_val_mse'] is not None:
        val_recheck['rel_diff'] = (abs(val_recheck['evaluate_model_val_mse'] - stored['val_loss'])
                                   / stored['val_loss'])
    test_metrics = safe_metrics(ckpt, TEST_PATH)

    states = trials_df['state'].value_counts().to_dict()
    psc = paired_startup_check(arch, trials_df)
    log(arch, f"Paros indito fazis (0-{N_PAIRED_STARTUP - 1}. trial vs TPE): "
              f"params={psc.get('n_params_equal')}/{psc.get('n_checked')}, "
              f"state={psc.get('n_state_equal')}, value={psc.get('n_value_equal')}, "
              f"mind azonos={psc.get('all_identical')}")

    ref = tpe_reference(arch)
    h_val = history['best_val_loss']
    b_val = baseline_val_loss(arch)
    pct = lambda r: (r - h_val) / r * 100.0 if r else None
    n_run = len(history['val_loss'])
    total_s = time.perf_counter() - t_start

    out = {
        'arch': arch,
        'model_class': CLASS_NAME[arch],
        'architecture_size_fixed_baseline': BASELINE_ARCH_KWARGS[arch],
        'sampler': type(study.sampler).__name__,
        'sampler_seed': BASE_SEED,
        'pruner': f"HyperbandPruner(min_resource=3, max_resource={SEARCH_MAX_EPOCHS}, reduction_factor=3)",
        'study_name': study.study_name,
        'lr': hp['lr'],
        'weight_decay': hp['weight_decay'],
        'dropout': hp['dropout'],
        'search_best_val_loss': study.best_value,
        'search_best_trial_number': best_trial.number,
        'search_max_epochs': search_max_epochs,
        'search_patience': SEARCH_PATIENCE,
        'n_trials_total': len(study.trials),
        'n_trials_target': n_trials_target,
        'n_trials_complete': int(states.get('COMPLETE', 0)),
        'n_trials_pruned': int(states.get('PRUNED', 0)),
        'n_trials_fail': int(states.get('FAIL', 0)),
        'param_importance_ped_anova': importances,
        'paired_startup_check': psc,
        'final_patience': FINAL_PATIENCE,
        'final_max_epochs': final_max_epochs,
        'final_batch_size': FINAL_BATCH_SIZE,
        'seed': BASE_SEED,
        'final_retrain_best_epoch': history['best_epoch'],
        'final_retrain_n_epochs_run': n_run,
        'hit_max_epochs': n_run >= final_max_epochs,
        'final_retrain_best_val_loss': h_val,
        'final_test_metrics': test_metrics,
        'val_mse_recheck': val_recheck,
        'param_count_match': pcc['match'],
        'param_count': pcc['random_param_count'],
        'baseline_val_loss': b_val,
        'improvement_pct_vs_baseline_val_loss': pct(b_val),
        **ref,
        'improvement_pct_vs_tpe_p30_val_loss': pct(ref.get('tpe_p30_best_val_loss')),
        **timing,
        'final_train_time_s': final_train_s,
        'total_time_s': timing['search_time_s'] + final_train_s,
        'pipeline_time_s_this_run': total_s,
    }
    write_json(P.results_dir / f"best_params_{arch}.json", out)
    log(arch, f"=== {arch} KESZ ({total_s:.1f}s ebben a futasban; kereses {timing['search_time_s']:.1f}s, "
              f"vegso tanitas {final_train_s:.1f}s). RS p30 val={h_val:.6e}, "
              f"TPE p30 val={ref.get('tpe_p30_best_val_loss')} ===")
    return out


def _load_result(arch: str):
    p = P.results_dir / f"best_params_{arch}.json"
    if not p.exists():
        return None
    d = read_json(p)
    t = P.results_dir / f"trials_{arch}.csv"
    if t.exists():
        d['search_trial_duration_sum_s'] = _valid_trial_duration_sum_s(pd.read_csv(t))
    return d


def update_campaign_meta(run_record: dict):
    path = P.results_dir / "campaign_meta.json"
    meta = read_json(path) if path.exists() else {'runs': []}
    runs = [r for r in meta.get('runs', []) if r.get('start') != run_record.get('start')]
    runs.append(run_record)
    per_arch = {}
    for arch in ARCH_ORDER:
        d = _load_result(arch)
        if d is not None:
            per_arch[arch] = {'search_time_s': d.get('search_time_s'),
                              'final_train_time_s': d.get('final_train_time_s'),
                              'search_trial_duration_sum_s': d.get('search_trial_duration_sum_s')}
    meta.update({
        'experiment': 'random search control',
        'sampler': f"RandomSampler(seed={BASE_SEED})",
        'pruner': f"HyperbandPruner(min_resource=3, max_resource={SEARCH_MAX_EPOCHS}, reduction_factor=3)",
        'final_patience': FINAL_PATIENCE,
        'optuna_version': optuna.__version__,
        'torch_version': torch.__version__,
        'python_version': platform.python_version(),
        'device': DEVICE,
        'gpu': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else None,
        'out_root': rel(P.out_root),
        'runs': runs,
        'campaign_start': runs[0].get('start') if runs else None,
        'campaign_end': runs[-1].get('end') if runs else None,
        'campaign_wall_time_s_sum_of_runs': sum(r.get('duration_s') or 0.0 for r in runs),
        'per_arch_time_s': per_arch,
        'search_time_s_sum': sum(v['search_time_s'] or 0.0 for v in per_arch.values()),
        'final_train_time_s_sum': sum(v['final_train_time_s'] or 0.0 for v in per_arch.values()),
        'search_trial_duration_sum_s_sum': sum(v['search_trial_duration_sum_s'] or 0.0
                                               for v in per_arch.values()),
    })
    write_json(path, meta)


def _sci(v):
    if not isinstance(v, (int, float)):
        return "–"
    if v == 0:
        return "0"
    mant, exp = f"{v:.2e}".split("e")
    return f"{mant}e{int(exp)}"


def _change(base, new):
    if not isinstance(base, (int, float)) or not isinstance(new, (int, float)) or base <= 0:
        return "–"
    return f"{(new - base) / base * 100:+.1f}%".replace("-", "−")


def _f3(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "–"


def _s0(v):
    return f"{v:.0f}" if isinstance(v, (int, float)) else "–"


def write_summary(out_path=None):
    out_path = Path(out_path) if out_path is not None else P.results_dir / "random_search_summary.md"
    rows = []
    for arch in ARCH_ORDER:
        d = _load_result(arch)
        if d is not None:
            rows.append(d)
    L = []

    L.append("## Keresés\n")
    L.append("| Architektúra | Trial | COMPLETE / PRUNED, RS | COMPLETE / PRUNED, TPE | "
             "Legjobb keresési val MSE, RS (trial) | Legjobb keresési val MSE, TPE (trial) | "
             "Azonos 0–9. trial |")
    L.append("|---|---:|---:|---:|---:|---:|:---:|")
    for d in rows:
        psc = d.get('paired_startup_check') or {}
        n_valid = d['n_trials_complete'] + d['n_trials_pruned']
        fail = f" (+{d['n_trials_fail']} FAIL)" if d.get('n_trials_fail') else ""
        L.append(f"| {d['model_class']} | {n_valid}{fail} | "
                 f"{d['n_trials_complete']} / {d['n_trials_pruned']} | "
                 f"{d.get('tpe_n_complete', '–')} / {d.get('tpe_n_pruned', '–')} | "
                 f"{_sci(d['search_best_val_loss'])} (#{d['search_best_trial_number']}) | "
                 f"{_sci(d.get('tpe_search_best_val_loss'))} (#{d.get('tpe_search_best_trial_number')}) | "
                 f"{'igen' if psc.get('all_identical') else 'nem'} |")

    L.append("\n## Legjobb hiperparaméterek\n")
    L.append("| Architektúra | lr, RS | weight_decay, RS | dropout, RS | "
             "lr, TPE | weight_decay, TPE | dropout, TPE |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in rows:
        L.append(f"| {d['model_class']} | {_sci(d['lr'])} | {_sci(d['weight_decay'])} | "
                 f"{_f3(d['dropout'])} | {_sci(d.get('tpe_lr'))} | "
                 f"{_sci(d.get('tpe_weight_decay'))} | {_f3(d.get('tpe_dropout'))} |")

    L.append("\n## Végső tanítás (p30)\n")
    L.append("| Architektúra | Paraméter | Teszt-RMSE, RS | Teszt-RMSE, TPE | Val MSE, RS | "
             "Val MSE, TPE | Val MSE változása, RS a TPE-hez | Legjobb / futott epoch, RS |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for d in rows:
        tm = d.get('final_test_metrics') or {}
        tt = d.get('tpe_p30_test_metrics') or {}
        params = f"{d['param_count']:,}".replace(",", " ")
        L.append(f"| {d['model_class']} | {params} | {_sci(tm.get('RMSE'))} | "
                 f"{_sci(tt.get('RMSE'))} | {_sci(d['final_retrain_best_val_loss'])} | "
                 f"{_sci(d.get('tpe_p30_best_val_loss'))} | "
                 f"{_change(d.get('tpe_p30_best_val_loss'), d['final_retrain_best_val_loss'])} | "
                 f"{d['final_retrain_best_epoch']} / {d['final_retrain_n_epochs_run']} |")

    L.append("\n## Futásidő (s)\n")
    L.append("| Architektúra | Keresés, RS | Trialidők összege, RS | Trialidők összege, TPE | "
             "Végső tanítás, RS | Végső tanítás, TPE | Összesen, RS |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in rows:
        L.append(f"| {d['model_class']} | {_s0(d['search_time_s'])} | "
                 f"{_s0(d['search_trial_duration_sum_s'])} | "
                 f"{_s0(d.get('tpe_search_trial_duration_sum_s'))} | {_s0(d['final_train_time_s'])} | "
                 f"{_s0(d.get('tpe_p30_retrain_seconds'))} | {_s0(d['total_time_s'])} |")
    if rows:
        L.append(f"| Összesen | {_s0(sum(d['search_time_s'] for d in rows))} | "
                 f"{_s0(sum(d['search_trial_duration_sum_s'] or 0 for d in rows))} | "
                 f"{_s0(sum(d.get('tpe_search_trial_duration_sum_s') or 0 for d in rows))} | "
                 f"{_s0(sum(d['final_train_time_s'] for d in rows))} | "
                 f"{_s0(sum(d.get('tpe_p30_retrain_seconds') or 0 for d in rows))} | "
                 f"{_s0(sum(d['total_time_s'] for d in rows))} |")

    with open(guard_path(out_path), 'w', encoding='utf-8') as f:
        f.write("\n".join(L) + "\n")
    print(f"[summary] {out_path.name} irva ({len(rows)}/{len(ARCH_ORDER)} architektura).")


def main():
    global P
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default=None)
    parser.add_argument('--archs', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--skip-done', action='store_true',
                        help="A mar kesz (best_params_{arch}.json-nal rendelkezo) architekturak kihagyasa.")
    parser.add_argument('--out-root', type=str, default=None,
                        help="Kimeneti gyoker (alapertelmezes: hpo/). Fustprobahoz scratch mappa.")
    parser.add_argument('--smoke', type=int, default=None,
                        help="Fustproba: N trial, N epoch a keresesben es a vegso tanitasban "
                             "(csak --out-root-tal, nem a hpo/ ala).")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve() if args.out_root else HPO_DIR
    if args.smoke is not None and _is_within(out_root, HPO_DIR):
        raise SystemExit("[smoke] A fustproba kimenete nem kerulhet a hpo/ ala -- adj meg --out-root-ot.")
    if out_root != HPO_DIR.resolve() and _is_within(out_root, REPO_ROOT):
        raise SystemExit(f"[guard] Az --out-root a repon belul csak a hpo/ lehet: {rel(out_root)}")
    for d in PROTECTED_DIRS:
        if _is_within(out_root, d):
            raise SystemExit(f"[guard] Az --out-root vedett mappaban van: {rel(out_root)}")
    P = Paths(out_root)
    P.ensure()

    stdout_file = open(guard_path(P.logs_dir / "campaign_stdout.log"), 'a', encoding='utf-8')
    sys.stdout = _Tee(sys.stdout, stdout_file)
    sys.stderr = _Tee(sys.stderr, stdout_file)

    if args.summary_only:
        write_summary()
        return

    if args.arch:
        archs = [args.arch]
    elif args.archs:
        archs = [a.strip() for a in args.archs.split(',') if a.strip()]
    else:
        archs = ARCH_ORDER
    for a in archs:
        if a not in ARCH_ORDER:
            raise SystemExit(f"Ismeretlen architektura: {a}")
    if args.skip_done:
        done = [a for a in archs if (P.results_dir / f"best_params_{a}.json").exists()]
        if done:
            print(f"[skip-done] Kihagyva (mar kesz): {done}")
        archs = [a for a in archs if a not in done]

    smoke = args.smoke
    search_max_epochs = smoke if smoke is not None else SEARCH_MAX_EPOCHS
    final_max_epochs = smoke if smoke is not None else FINAL_MAX_EPOCHS

    start_str = time.strftime('%Y-%m-%d %H:%M:%S')
    t0 = time.perf_counter()
    print(f"[{start_str}] Veletlen keresesi kontrollkiserlet indul. Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Optuna {optuna.__version__}, torch {torch.__version__}; sampler RandomSampler(seed={BASE_SEED})")
    print(f"Kimeneti gyoker: {rel(P.out_root)}; kereses max_epochs={search_max_epochs}, "
          f"vegso max_epochs={final_max_epochs}, patience={FINAL_PATIENCE}"
          + (f"; FUSTPROBA: {smoke} trial/architektura" if smoke is not None else ""))
    print(f"Futtatando architekturak: {archs}")

    run_record = {'start': start_str, 'end': None, 'duration_s': None, 'archs': archs,
                  'smoke': smoke, 'status': 'running', 'failed_archs': []}
    update_campaign_meta(run_record)

    for arch in archs:
        n_trials = smoke if smoke is not None else TRIAL_BUDGET[arch]
        try:
            run_arch(arch, n_trials, search_max_epochs, final_max_epochs)
        except Exception:
            err = traceback.format_exc()
            with open(guard_path(P.logs_dir / f"{arch}_ERROR.log"), 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n{err}\n")
            print(f"[HIBA] {arch}:\n{err}")
            run_record['failed_archs'].append(arch)

    write_summary()
    run_record.update({'end': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'duration_s': time.perf_counter() - t0, 'status': 'finished'})
    update_campaign_meta(run_record)
    print(f"[{run_record['end']}] Kampany kesz: {run_record['duration_s']:.1f}s; "
          f"hibas architekturak: {run_record['failed_archs'] or 'nincs'}")


if __name__ == '__main__':
    main()
