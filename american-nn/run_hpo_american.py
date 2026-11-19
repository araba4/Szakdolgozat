import sys
sys.dont_write_bytecode = True

import argparse
import os
import pickle
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

AMNN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AMNN_DIR))

import common as C
from common import (
    ARCH_ORDER, CLASS_NAME, BASELINE_ARCH_KWARGS, HAS_DROPOUT, DEVICE,
    FEATURE_COLS, TARGETS, get_model, count_parameters, guard_path, write_json, read_json, rel,
)
import train_uniform_baseline as TU
import train_hpo_transfer as TT

HPO_DIR = C.REPO_ROOT / "hpo"


def _snapshot(root: Path) -> dict:
    out = {}
    for dp, dn, fn in os.walk(root):
        for d in dn:
            out[str(Path(dp, d))] = ('dir',)
        for f in fn:
            st = os.stat(Path(dp, f))
            out[str(Path(dp, f))] = (st.st_size, st.st_mtime_ns)
    return out


_HPO_SNAP_BEFORE = _snapshot(HPO_DIR)
sys.path.insert(1, str(HPO_DIR))
import run_hpo as R
_HPO_SNAP_AFTER = _snapshot(HPO_DIR)
HPO_IMPORT_CHECK = {
    'n_entries_before': len(_HPO_SNAP_BEFORE), 'n_entries_after': len(_HPO_SNAP_AFTER),
    'hpo_tree_unchanged_by_import': _HPO_SNAP_BEFORE == _HPO_SNAP_AFTER,
}
del _HPO_SNAP_BEFORE, _HPO_SNAP_AFTER

optuna.logging.set_verbosity(optuna.logging.WARNING)

assert R.BASELINE_ARCH_KWARGS == BASELINE_ARCH_KWARGS
assert R.HAS_DROPOUT == HAS_DROPOUT
assert R.CLASS_NAME == CLASS_NAME
assert R.ARCH_ORDER == ARCH_ORDER
assert R.DEVICE == DEVICE
assert R.get_model is get_model

FINAL_PATIENCE = 30
FINAL_MAX_EPOCHS = R.FINAL_MAX_EPOCHS
FINAL_BATCH_SIZE = R.FINAL_BATCH_SIZE
SEED = R.BASE_SEED
SEARCH_TARGET = 'american'
TARGET_ORDER = ['american', 'european']
PRUNER_KW = dict(min_resource=3, max_resource=R.SEARCH_MAX_EPOCHS, reduction_factor=3)
CURVE_COLS = ['trial_number', 'epoch', 'train_loss', 'val_loss', 'lr']

HPO_P30_DIR = HPO_DIR / "results" / "p30"
UNIFORM_RESULTS_DIR = AMNN_DIR / "results" / "uniform"
TRANSFER_RESULTS_DIR = AMNN_DIR / "results" / "transfer"


def run_name(arch, target):
    return f"{arch}_{target}_retune_p{FINAL_PATIENCE}"


class Paths:
    def __init__(self, out_root: Path):
        self.out_root = out_root.resolve()
        self.studies_dir = self.out_root / "optuna_studies" / "retune"
        self.models_dir = self.out_root / "models" / "retune"
        self.results_dir = self.out_root / "results" / "retune"
        self.logs_dir = self.out_root / "logs" / "retune"

    def ensure(self):
        for d in (self.studies_dir, self.models_dir, self.results_dir, self.logs_dir):
            guard_path(d).mkdir(parents=True, exist_ok=True)

    def ckpt(self, run):
        return self.models_dir / f"{run}_best.pt"

    def db(self, arch):
        return self.studies_dir / f"{arch}.db"

    def sampler_state(self, arch):
        return self.studies_dir / f"{arch}_sampler_state.pkl"


P = Paths(AMNN_DIR)


def log(name: str, msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(guard_path(P.logs_dir / f"{name}.log"), 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def _append_curve_rows(arch: str, rows: list):
    if not rows:
        return
    curve_path = guard_path(P.results_dir / f"search_curves_{arch}.csv")
    df = pd.DataFrame(rows, columns=CURVE_COLS)
    df.to_csv(curve_path, mode='a', header=not curve_path.exists(), index=False)


def make_objective(arch, search_max_epochs, train_ds, val_ds, clock):
    sink = []

    def objective(trial: 'optuna.Trial'):
        torch.manual_seed(R.BASE_SEED + trial.number)
        lr, weight_decay, dropout, model_kwargs = R.build_trial_hparams(trial, arch)
        model = R.get_model(arch, **model_kwargs)
        sink.clear()
        try:
            return R.train_for_search(
                model, train_ds, val_ds, lr, weight_decay, trial,
                max_epochs=search_max_epochs, patience=R.SEARCH_PATIENCE,
                batch_size=R.SEARCH_BATCH_SIZE, device=R.DEVICE,
                curve_sink=sink,
            )
        finally:
            _append_curve_rows(arch, sink)
            trial.set_user_attr('n_epochs', len(sink))
            trial.set_user_attr('search_time_s_cumulative', clock())
            sink.clear()

    return objective


def _valid_trial_duration_sum_s(trials_df: pd.DataFrame):
    if 'duration' not in trials_df or 'state' not in trials_df:
        return None
    valid = trials_df[trials_df['state'].isin(['COMPLETE', 'PRUNED'])]
    return float(pd.to_timedelta(valid['duration']).dt.total_seconds().sum())


def _write_trials_csv(study, arch):
    df = study.trials_dataframe()
    df.to_csv(guard_path(P.results_dir / f"trials_{arch}.csv"), index=False)
    return df


def _n_valid(study):
    return sum(1 for t in study.trials
               if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED))


def open_study(arch, resume_sampler=True):
    db_path = guard_path(P.db(arch))
    storage = f"sqlite:///{db_path.as_posix()}"
    sampler = TPESampler(seed=R.BASE_SEED)
    restored = None
    sp = P.sampler_state(arch)
    if resume_sampler and sp.exists() and db_path.exists():
        with open(sp, 'rb') as f:
            st = pickle.load(f)
        sampler = st['sampler']
        restored = {k: v for k, v in st.items() if k != 'sampler'}
    pruner = HyperbandPruner(**PRUNER_KW)
    study = optuna.create_study(study_name=arch, storage=storage, direction='minimize',
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    return study, db_path, restored


def run_search(arch, n_trials_target, search_max_epochs):
    log(arch, f"=== {arch} ({CLASS_NAME[arch]}) UJRAHANGOLAS (amerikai put) indul -- device={DEVICE}, "
              f"trial-cel={n_trials_target}, fix meret={BASELINE_ARCH_KWARGS[arch]}, "
              f"dropout kereshetos={HAS_DROPOUT[arch]}, kereses max_epochs={search_max_epochs} ===")
    study, db_path, restored = open_study(arch)
    if not study.user_attrs.get('created_at'):
        study.set_user_attr('created_at', time.strftime('%Y-%m-%d %H:%M:%S'))
        study.set_user_attr('sampler', f"TPESampler(seed={R.BASE_SEED})")
        study.set_user_attr('sampler_seed', R.BASE_SEED)
        study.set_user_attr('pruner', {'class': type(study.pruner).__name__,
                                       'min_resource': study.pruner._min_resource,
                                       'max_resource': study.pruner._max_resource,
                                       'reduction_factor': study.pruner._reduction_factor})
        study.set_user_attr('search_target', SEARCH_TARGET)
        study.set_user_attr('target_definition', TARGETS[SEARCH_TARGET]['formula'])
        study.set_user_attr('search_max_epochs', search_max_epochs)
        study.set_user_attr('search_patience', R.SEARCH_PATIENCE)
        study.set_user_attr('search_batch_size', R.SEARCH_BATCH_SIZE)
        study.set_user_attr('n_trials_target', n_trials_target)
        study.set_user_attr('experiment', 'HPO retune on american put')
    log(arch, f"Sampler: {type(study.sampler).__name__}(seed={R.BASE_SEED}), pruner: "
              f"{type(study.pruner).__name__}({PRUNER_KW}), storage: {rel(db_path)}")

    stale = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            log(arch, f"Elakadt (stale) RUNNING trial #{t.number} FAIL-re allitva "
                      f"(korabban megszakadt folyamatbol maradt vissza).")
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)
            stale.append(t.number)

    n_valid = _n_valid(study)
    n_remaining = max(0, n_trials_target - n_valid)
    resume_info = None
    if len(study.trials) > 0:
        finished = [t.number for t in study.trials if t.number not in stale
                    and t.state != optuna.trial.TrialState.FAIL]
        last_finished = max(finished) if finished else None
        consistent = restored is not None and restored.get('after_trial') == last_finished
        resume_info = {'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'n_trials_in_storage': len(study.trials),
                       'n_valid': n_valid, 'n_remaining': n_remaining, 'stale_trials_set_fail': stale,
                       'sampler_state_restored': restored is not None,
                       'sampler_state_after_trial': (restored or {}).get('after_trial'),
                       'last_finished_trial': last_finished, 'sampler_state_consistent': consistent}
        log(arch, f"Folytatas: {n_valid} ervenyes trial mar letezik, {n_remaining} hianyzik. "
                  f"Sampler-allapot visszatoltve: {restored is not None} (#{(restored or {}).get('after_trial')} "
                  f"utani; utolso lezart trial #{last_finished}; konzisztens: {consistent}).")
        if n_remaining > 0 and not consistent:
            log(arch, "FIGYELEM: a sampler-allapot nem a legutolso lezart trial utani -- a folytatott "
                      "mintasorozat elterhet a megszakitatlanetol (dokumentalando).")
        hist = list(study.user_attrs.get('resume_events', []))
        hist.append(resume_info)
        study.set_user_attr('resume_events', hist)

    train_ds, val_ds = TU.get_ds('train', SEARCH_TARGET), TU.get_ds('val', SEARCH_TARGET)
    prev_search_s = float(study.user_attrs.get('search_time_s_cumulative', 0.0))
    t0 = time.perf_counter()

    def clock():
        return prev_search_s + time.perf_counter() - t0

    def _after_trial(st, trial):
        st.set_user_attr('search_time_s_cumulative', clock())
        tmp = guard_path(P.sampler_state(arch).with_suffix('.pkl.tmp'))
        with open(tmp, 'wb') as f:
            pickle.dump({'after_trial': trial.number, 'n_trials': len(st.trials),
                         'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'sampler': st.sampler}, f)
        os.replace(tmp, guard_path(P.sampler_state(arch)))
        _write_trials_csv(st, arch)
        log(arch, f"trial #{trial.number}: {trial.state.name}, value={trial.value}, "
                  f"epoch={trial.user_attrs.get('n_epochs')}, params={trial.params}, "
                  f"ervenyes={_n_valid(st)}/{n_trials_target}")

    if n_remaining > 0:
        study.optimize(make_objective(arch, search_max_epochs, train_ds, val_ds, clock),
                       n_trials=n_remaining, show_progress_bar=False, callbacks=[_after_trial])
    dt = time.perf_counter() - t0
    search_time_s = prev_search_s + dt
    study.set_user_attr('search_time_s_cumulative', search_time_s)
    n_runs = int(study.user_attrs.get('n_search_invocations', 0)) + (1 if n_remaining > 0 else 0)
    study.set_user_attr('n_search_invocations', n_runs)

    trials_df = _write_trials_csv(study, arch)
    curve_path = P.results_dir / f"search_curves_{arch}.csv"
    if curve_path.exists():
        cdf = pd.read_csv(curve_path)
        cdf = cdf.drop_duplicates(subset=['trial_number', 'epoch'], keep='last')
        cdf = cdf.sort_values(['trial_number', 'epoch'])
        cdf.to_csv(guard_path(curve_path), index=False)
        n_search_epochs = int(len(cdf))
    else:
        n_search_epochs = 0

    try:
        importances = {k: float(v) for k, v in optuna.importance.get_param_importances(study).items()}
    except Exception as e:
        importances = {"error": str(e)}
    write_json(P.results_dir / f"param_importance_{arch}.json", importances)

    log(arch, f"Kereses kesz: {len(study.trials)} trial ({_n_valid(study)} ervenyes), "
              f"{dt:.1f}s ebben a futasban, halmozott {search_time_s:.1f}s. Legjobb val_loss="
              f"{study.best_value:.8e}, trial #{study.best_trial.number}, params={study.best_trial.params}")
    timing = {'search_time_s': search_time_s, 'search_time_s_this_run': dt,
              'n_search_invocations': n_runs,
              'search_trial_duration_sum_s': _valid_trial_duration_sum_s(trials_df),
              'n_search_epochs': n_search_epochs}
    return study, trials_df, importances, timing


def best_hp(study, arch) -> dict:
    p = dict(study.best_trial.params)
    return {'lr': p['lr'], 'weight_decay': p['weight_decay'],
            'dropout': p['dropout'] if HAS_DROPOUT[arch] else None}


def model_kwargs_for(arch, hp):
    kw = dict(BASELINE_ARCH_KWARGS[arch])
    if HAS_DROPOUT[arch]:
        kw['dropout'] = hp['dropout']
    return kw


def param_count_check(arch, target, model_kwargs) -> dict:
    ours = count_parameters(get_model(arch, **model_kwargs))
    v1_path = C.ONLAB_MODELS_DIR / f"{arch}_best.pt"
    v1 = torch.load(v1_path, map_location='cpu', weights_only=False)
    v1_count = count_parameters(get_model(v1['model_class'], **v1['model_kwargs']))
    hpo_pcc_path = HPO_P30_DIR / f"param_count_check_{arch}.json"
    hpo_count = read_json(hpo_pcc_path)['hpo_param_count']
    uni_path = UNIFORM_RESULTS_DIR / f"metrics_{arch}_{target}_p30.json"
    tr_path = TRANSFER_RESULTS_DIR / f"metrics_{arch}_{target}_transfer_p30.json"
    uni_count = read_json(uni_path)['param_count']
    tr_count = read_json(tr_path)['param_count']
    size_kw = {k: v for k, v in model_kwargs.items() if k != 'dropout'}
    res = {'arch': arch, 'target': target, 'model_kwargs': model_kwargs, 'param_count': ours,
           'onlab_v1_checkpoint': rel(v1_path), 'onlab_v1_param_count': v1_count,
           'hpo_p30_param_count_check': rel(hpo_pcc_path), 'hpo_p30_param_count': hpo_count,
           'uniform_p30_metrics': rel(uni_path), 'uniform_p30_param_count': uni_count,
           'transfer_p30_metrics': rel(tr_path), 'transfer_p30_param_count': tr_count,
           'size_kwargs_equal_baseline_arch_kwargs': size_kw == BASELINE_ARCH_KWARGS[arch]}
    res['param_count_match'] = bool(ours == v1_count == hpo_count == uni_count == tr_count)
    res['all_ok'] = bool(res['param_count_match'] and res['size_kwargs_equal_baseline_arch_kwargs'])
    if not res['all_ok']:
        raise RuntimeError(f"[{arch}] Kapacitas-kontroll serul, leallas: {res}")
    return res


def run_final(arch, target, hp, study_info, max_epochs):
    run = run_name(arch, target)
    t_start = time.time()
    kw = model_kwargs_for(arch, hp)
    spec = TARGETS[target]
    log(run, f"=== {run} ({CLASS_NAME[arch]}) -- cel: {spec['formula']}, patience={FINAL_PATIENCE}, "
             f"max_epochs={max_epochs}, lr={hp['lr']!r}, wd={hp['weight_decay']!r}, "
             f"dropout={hp['dropout']!r}, batch={FINAL_BATCH_SIZE}, seed={SEED}, "
             f"forras={study_info['study_db']} (#{study_info['best_trial_number']}), model_kwargs={kw} ===")
    pcc = param_count_check(arch, target, kw)
    write_json(P.results_dir / f"param_count_check_{run}.json", pcc)

    train_ds, val_ds = TU.get_ds('train', target), TU.get_ds('val', target)
    ckpt = guard_path(P.ckpt(run))
    hyper = {'lr': hp['lr'], 'weight_decay': hp['weight_decay'], 'dropout': hp['dropout'],
             'batch_size': FINAL_BATCH_SIZE}
    extra = {'target': target, 'target_definition': spec['formula'],
             'data': rel(C.AM_DATA_DIR), 'patience': FINAL_PATIENCE, 'max_epochs': max_epochs,
             'seed': SEED, 'hyperparameters': hyper, 'search_target': SEARCH_TARGET,
             'hyperparameter_source_study': study_info['study_db'],
             'hyperparameter_source_trial': study_info['best_trial_number'],
             'experiment': 'HPO retune on american put'}

    torch.manual_seed(SEED)
    model = get_model(arch, **kw)
    t0 = time.time()
    hist = C.train_loop(model, train_ds, val_ds, output_dir=P.models_dir, model_name=run,
                        model_class=arch, model_kwargs=kw, feature_cols=FEATURE_COLS,
                        target_col=spec['target_col'], batch_size=FINAL_BATCH_SIZE,
                        max_epochs=max_epochs, lr=hp['lr'], weight_decay=hp['weight_decay'],
                        patience=FINAL_PATIENCE, device=DEVICE, extra_meta=extra,
                        checkpoint_path=ckpt)
    train_seconds = time.time() - t0
    n_run = len(hist['val_loss'])
    pd.DataFrame({'epoch': range(1, n_run + 1), 'train_loss': hist['train_loss'],
                  'val_loss': hist['val_loss'], 'lr': hist['lr'],
                  'epoch_seconds': hist['epoch_seconds']}).to_csv(
        guard_path(P.results_dir / f"training_history_{run}.csv"), index=False)
    log(run, f"Tanitas kesz ({train_seconds:.1f}s): best_epoch={hist['best_epoch']}, "
             f"futott={n_run}, best_val_loss={hist['best_val_loss']:.8e}")

    ev = TU.evaluate_run(run, target, ckpt)
    out = {
        'run': run, 'arch': arch, 'model_class': CLASS_NAME[arch], 'target': target,
        'target_col': spec['target_col'], 'target_definition': spec['formula'],
        'level': 'retune_p30', 'search_target': SEARCH_TARGET,
        'patience': FINAL_PATIENCE, 'max_epochs': max_epochs, 'seed': SEED,
        'hyperparameters': hyper,
        'hyperparameter_source_study': study_info['study_db'],
        'hyperparameter_source_trial': study_info['best_trial_number'],
        'search_best_val_loss': study_info['best_value'],
        'model_kwargs': kw, 'param_count': pcc['param_count'],
        'param_count_match': pcc['param_count_match'], 'param_count_check': pcc,
        'checkpoint': rel(ckpt),
        'data_files': {s: rel(C.am_split_path(s)) for s in C.SPLITS},
        'best_epoch': hist['best_epoch'], 'n_epochs_run': n_run,
        'hit_max_epochs': n_run >= max_epochs,
        'best_val_loss': hist['best_val_loss'],
        'train_seconds': train_seconds,
        'mean_epoch_seconds': float(np.mean(hist['epoch_seconds'])),
        **ev,
    }
    write_json(P.results_dir / f"metrics_{run}.json", out)
    tm = ev['test_metrics']
    log(run, f"=== {run} KESZ ({time.time() - t_start:.1f}s): val={hist['best_val_loss']:.4e}, "
             f"test RMSE={tm['RMSE']:.4e}, R2={tm['R2']:.6f}, val ujraszamolva rel. elteres="
             f"{ev['val_recheck']['rel_diff']:.2e} ===")
    return out


def run_arch(arch, n_trials_target, search_max_epochs, final_max_epochs, final_only, skip_done):
    t_start = time.perf_counter()
    if final_only:
        study, db_path, _ = open_study(arch, resume_sampler=False)
        if _n_valid(study) < n_trials_target:
            raise RuntimeError(f"[{arch}] --final-only, de a study nem teljes: "
                               f"{_n_valid(study)}/{n_trials_target} ervenyes trial.")
        trials_df = _write_trials_csv(study, arch)
        pi = P.results_dir / f"param_importance_{arch}.json"
        importances = read_json(pi) if pi.exists() else None
        cpath = P.results_dir / f"search_curves_{arch}.csv"
        timing = {'search_time_s': float(study.user_attrs.get('search_time_s_cumulative', 0.0)),
                  'search_time_s_this_run': 0.0,
                  'n_search_invocations': int(study.user_attrs.get('n_search_invocations', 0)),
                  'search_trial_duration_sum_s': _valid_trial_duration_sum_s(trials_df),
                  'n_search_epochs': int(len(pd.read_csv(cpath))) if cpath.exists() else 0}
    else:
        study, trials_df, importances, timing = run_search(arch, n_trials_target, search_max_epochs)

    hp = best_hp(study, arch)
    study_info = {'study_db': rel(P.db(arch)), 'study_name': study.study_name,
                  'best_trial_number': study.best_trial.number, 'best_value': study.best_value}
    finals = {}
    for target in TARGET_ORDER:
        run = run_name(arch, target)
        mpath = P.results_dir / f"metrics_{run}.json"
        if skip_done and mpath.exists() and P.ckpt(run).exists():
            d = read_json(mpath)
            if d['hyperparameters']['lr'] == hp['lr'] and d['hyperparameters']['weight_decay'] == hp['weight_decay'] \
                    and d['hyperparameters']['dropout'] == hp['dropout']:
                log(arch, f"[skip-done] {run} mar kesz (azonos hiperparameterekkel), kihagyva.")
                finals[target] = d
                continue
            log(arch, f"[skip-done] {run} metrika-JSON-ja mas hiperparameterekkel keszult -- ujratanitas.")
        finals[target] = run_final(arch, target, hp, study_info, final_max_epochs)

    am, eu = finals['american'], finals['european']
    states = trials_df['state'].value_counts().to_dict()
    fail_numbers = [int(n) for n in trials_df.loc[trials_df['state'] == 'FAIL', 'number']]
    tr_bp = read_json(HPO_P30_DIR / f"best_params_{arch}.json")
    eu_search_bp = read_json(HPO_DIR / "results" / f"best_params_{arch}.json")
    out = {
        'arch': arch, 'model_class': CLASS_NAME[arch],
        'architecture_size_fixed_baseline': BASELINE_ARCH_KWARGS[arch],
        'lr': hp['lr'], 'weight_decay': hp['weight_decay'], 'dropout': hp['dropout'],
        'search_source': rel(P.db(arch)),
        'search_best_val_loss': study.best_value,
        'search_best_trial_number': study.best_trial.number,
        'n_trials_completed': len(study.trials),
        'n_trials_target': n_trials_target,
        'final_patience': FINAL_PATIENCE, 'final_max_epochs': final_max_epochs,
        'final_batch_size': FINAL_BATCH_SIZE, 'seed': SEED,
        'final_retrain_best_epoch': am['best_epoch'],
        'final_retrain_n_epochs_run': am['n_epochs_run'],
        'hit_max_epochs': am['hit_max_epochs'],
        'final_retrain_best_val_loss': am['best_val_loss'],
        'param_count_match': bool(am['param_count_match'] and eu['param_count_match']),
        'param_count': am['param_count'],
        'hpo_test_metrics': am['test_metrics'],
        'retrain_seconds': am['train_seconds'],
        'val_mse_recheck': am['val_recheck'],
        'target': 'american', 'target_definition': TARGETS['american']['formula'],
        'search_target': SEARCH_TARGET,
        'study_name': study.study_name, 'study_db': rel(P.db(arch)),
        'sampler': f"TPESampler(seed={R.BASE_SEED})",
        'pruner': f"HyperbandPruner(min_resource={PRUNER_KW['min_resource']}, "
                  f"max_resource={PRUNER_KW['max_resource']}, reduction_factor={PRUNER_KW['reduction_factor']})",
        'search_max_epochs': search_max_epochs, 'search_patience': R.SEARCH_PATIENCE,
        'search_batch_size': R.SEARCH_BATCH_SIZE, 'cold_start': True,
        'n_trials_total': len(study.trials),
        'n_trials_valid': int(states.get('COMPLETE', 0) + states.get('PRUNED', 0)),
        'n_trials_complete': int(states.get('COMPLETE', 0)),
        'n_trials_pruned': int(states.get('PRUNED', 0)),
        'n_trials_fail': int(states.get('FAIL', 0)), 'fail_trial_numbers': fail_numbers,
        'resume_events': study.user_attrs.get('resume_events', []),
        'param_importance_ped_anova': importances,
        'hpo_test_metrics_usd': am['test_metrics_usd'],
        'metrics_american': rel(P.results_dir / f"metrics_{am['run']}.json"),
        'european_reference': {
            'run': eu['run'], 'metrics': rel(P.results_dir / f"metrics_{eu['run']}.json"),
            'target_definition': TARGETS['european']['formula'],
            'hyperparameters': eu['hyperparameters'],
            'hyperparameters_bitwise_equal_american': bool(
                eu['hyperparameters']['lr'] == am['hyperparameters']['lr']
                and eu['hyperparameters']['weight_decay'] == am['hyperparameters']['weight_decay']
                and eu['hyperparameters']['dropout'] == am['hyperparameters']['dropout']),
            'best_epoch': eu['best_epoch'], 'n_epochs_run': eu['n_epochs_run'],
            'hit_max_epochs': eu['hit_max_epochs'], 'best_val_loss': eu['best_val_loss'],
            'test_metrics': eu['test_metrics'], 'test_metrics_usd': eu['test_metrics_usd'],
            'train_seconds': eu['train_seconds'], 'val_recheck': eu['val_recheck'],
        },
        'european_call_tpe_reference': {
            'search_json': rel(HPO_DIR / "results" / f"best_params_{arch}.json"),
            'p30_json': rel(HPO_P30_DIR / f"best_params_{arch}.json"),
            'lr': tr_bp['lr'], 'weight_decay': tr_bp['weight_decay'], 'dropout': tr_bp['dropout'],
            'search_best_val_loss': eu_search_bp.get('search_best_val_loss'),
            'search_best_trial_number': eu_search_bp.get('search_best_trial_number'),
            'p30_best_val_loss': tr_bp.get('final_retrain_best_val_loss'),
        },
        **timing,
        'final_train_time_s_american': am['train_seconds'],
        'final_train_time_s_european': eu['train_seconds'],
        'total_time_s': timing['search_time_s'] + am['train_seconds'] + eu['train_seconds'],
        'pipeline_time_s_this_run': time.perf_counter() - t_start,
    }
    write_json(P.results_dir / f"best_params_{arch}.json", out)
    log(arch, f"=== {arch} KESZ: retune p30 am. val={am['best_val_loss']:.4e} (teszt RMSE "
              f"{am['test_metrics']['RMSE']:.4e}), eu. val={eu['best_val_loss']:.4e}; kereses "
              f"{timing['search_time_s']:.1f}s, vegso {am['train_seconds'] + eu['train_seconds']:.1f}s ===")
    return out


def main():
    global P
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', type=str, default=None)
    ap.add_argument('--archs', type=str, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--skip-done', action='store_true',
                    help="Kesz architekturak (best_params_{arch}.json) es kesz vegso tanitasok "
                         "(metrics_{run}.json + checkpoint) kihagyasa; a felbehagyott study folytatasa.")
    ap.add_argument('--final-only', action='store_true',
                    help="Csak a vegso tanitasok (a meglevo, teljes study legjobb trialjaval).")
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--smoke', type=int, default=None,
                    help="Fustproba: N trial, N epoch a keresesben es a vegso tanitasban "
                         "(csak az american-nn/-en kivuli --out-root-tal).")
    ap.add_argument('--smoke-trials', type=int, default=None,
                    help="Fustprobanal a trialszam felulirasa (a folytatas tesztelesehez).")
    ap.add_argument('--out-root', type=str, default=None)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve() if args.out_root else AMNN_DIR
    if args.smoke is not None and C._is_within(out_root, AMNN_DIR):
        raise SystemExit("[smoke] A fustproba kimenete nem kerulhet az american-nn/ ala -- adj meg --out-root-ot.")
    if args.smoke is None and args.smoke_trials is not None:
        raise SystemExit("--smoke-trials csak --smoke-kal.")
    C.set_allowed_roots(out_root)
    P = Paths(out_root)
    TU.P = P
    TT.P = P
    P.ensure()
    sys.stdout = TU._Tee(sys.stdout, P.logs_dir / "campaign_stdout.log")
    sys.stderr = TU._Tee(sys.stderr, P.logs_dir / "campaign_stdout.log")

    if args.summary_only:
        import analyze_retune as A
        A.main(P)
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

    pc = TU.pairing_check()
    if not pc['all_paired']:
        raise SystemExit("[parositas] A ket adathalmaz NEM paros -- a kampany leall.")
    edges = TT.use_uniform_premium_edges()
    print(f"[premium] uniform vodorhatarok, ujraszamolva bitre azonos: {edges['recomputed_bitwise_equal']}")

    smoke = args.smoke
    search_max_epochs = smoke if smoke is not None else R.SEARCH_MAX_EPOCHS
    final_max_epochs = smoke if smoke is not None else FINAL_MAX_EPOCHS

    meta_path = P.results_dir / "campaign_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {'invocations': []}
    meta.update({
        'experiment': 'HPO retune on american put',
        'env': C.env_meta(), 'optuna_version': optuna.__version__,
        'machine': {'processor': platform.processor()},
        'protocol_source': 'hpo/run_hpo.py (import run_hpo as R)',
        'search_space': "run_hpo.build_trial_hparams: lr [1e-4,1e-2] log; weight_decay [1e-6,1e-2] log; "
                        "dropout [0,0.5] (HAS_DROPOUT)",
        'trial_budget': R.TRIAL_BUDGET, 'search_max_epochs': R.SEARCH_MAX_EPOCHS,
        'search_patience': R.SEARCH_PATIENCE, 'search_batch_size': R.SEARCH_BATCH_SIZE,
        'sampler': f"TPESampler(seed={R.BASE_SEED})", 'pruner': f"HyperbandPruner({PRUNER_KW})",
        'trial_seed': 'torch.manual_seed(BASE_SEED + trial.number)', 'cold_start': True,
        'search_target': TARGETS[SEARCH_TARGET]['formula'],
        'final': {'patience': FINAL_PATIENCE, 'max_epochs': FINAL_MAX_EPOCHS,
                  'batch_size': FINAL_BATCH_SIZE, 'seed': SEED, 'loop': 'common.train_loop',
                  'targets': {t: TARGETS[t]['formula'] for t in TARGET_ORDER}},
        'hpo_import_check': HPO_IMPORT_CHECK,
    })
    meta['invocations'].append({'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'argv': sys.argv[1:],
                                'archs': archs, 'smoke': smoke, 'final_only': args.final_only,
                                'pid': os.getpid()})
    write_json(meta_path, meta)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Amerikai ujrahangolas indul. Device: {DEVICE} "
          f"({C.env_meta().get('gpu_name')}); Optuna {optuna.__version__}; kimenet: {P.out_root}")
    print(f"hpo/ import-ellenorzes: {HPO_IMPORT_CHECK}")
    print(f"Futtatando architekturak: {archs}; kereses max_epochs={search_max_epochs}, "
          f"vegso max_epochs={final_max_epochs}" + (f"; FUSTPROBA {smoke}" if smoke is not None else ""))
    t0 = time.time()
    failed = []
    for arch in archs:
        if smoke is not None:
            n_trials = args.smoke_trials if args.smoke_trials is not None else smoke
        else:
            n_trials = R.TRIAL_BUDGET[arch]
        try:
            run_arch(arch, n_trials, search_max_epochs, final_max_epochs, args.final_only, args.skip_done)
        except Exception:
            err = traceback.format_exc()
            with open(guard_path(P.logs_dir / f"{arch}_ERROR.log"), 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n{err}\n")
            print(f"[HIBA] {arch}:\n{err}")
            failed.append(arch)
        finally:
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

    meta = read_json(meta_path)
    meta['invocations'][-1].update({'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'wall_seconds': time.time() - t0, 'failed_archs': failed})
    write_json(meta_path, meta)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Kampany kesz: {time.time() - t0:.1f}s, "
          f"hibas: {failed or 'nincs'}")

    import analyze_retune as A
    A.main(P)


if __name__ == '__main__':
    main()
