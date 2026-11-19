import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HPO_DIR = Path(__file__).resolve().parent.parent / "hpo"
sys.path.insert(0, str(HPO_DIR))

import run_hpo as R
from run_hpo import (
    REPO_ROOT, ONLAB_DIR, BASELINE_MODELS_DIR, BASELINE_RESULTS_DIR,
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    CLASS_NAME, BASELINE_ARCH_KWARGS, ARCH_ORDER,
    FINAL_MAX_EPOCHS, FINAL_BATCH_SIZE, BASE_SEED, DEVICE,
    get_model, count_parameters, train_model, evaluate_model,
    DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL,
    baseline_val_loss,
)

FINAL_PATIENCE = 30
P10_PATIENCE = R.FINAL_PATIENCE

P10_RESULTS_DIR = HPO_DIR / "results"
P10_MODELS_DIR = HPO_DIR / "models"
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
        self.models_dir = self.out_root / "models"
        self.results_dir = self.out_root / "results" / "p30"
        self.logs_dir = self.out_root / "logs" / "p30"

    def ensure(self):
        for d in (self.models_dir, self.results_dir, self.logs_dir):
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
    if _is_within(path, HPO_DIR):
        ok = (
            _is_within(path, HPO_DIR / "results" / "p30")
            or _is_within(path, HPO_DIR / "logs" / "p30")
            or (path.parent == (HPO_DIR / "models").resolve()
                and path.name.endswith("_hpo_p30_best.pt"))
        )
        if not ok:
            raise PermissionError(
                f"[guard] Az elso HPO-kampany utvonala, nem irhato: {rel(path)}")
    return path


class _Tee:
    def __init__(self, stream, path: Path):
        self.stream = stream
        self.f = open(guard_path(path), 'a', encoding='utf-8')

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


def p10_info(arch: str):
    bp = read_json(P10_RESULTS_DIR / f"best_params_{arch}.json")
    hist = pd.read_csv(P10_RESULTS_DIR / f"final_training_history_{arch}.csv",
                       float_precision='round_trip')
    return bp, hist


def onlab_history(arch: str, version: str):
    suffix = "" if version == 'v1' else "_v2"
    base = BASELINE_RESULTS_DIR / f"training_history_{CLASS_NAME[arch]}{suffix}"
    if not (base.with_suffix('.json').exists() and base.with_suffix('.csv').exists()):
        return None
    return read_json(base.with_suffix('.json')), pd.read_csv(base.with_suffix('.csv'))


def onlab_ckpt(arch: str, version: str) -> Path:
    return BASELINE_MODELS_DIR / (f"{arch}_best.pt" if version == 'v1' else f"{arch}_v2_best.pt")


def onlab_best(arch: str):
    cands = []
    for v in ('v1', 'v2'):
        h = onlab_history(arch, v)
        if h is None or not onlab_ckpt(arch, v).exists():
            continue
        js, df = h
        cands.append(dict(version=v, best_val_loss=js['best_val_loss'],
                          best_epoch=js['best_epoch'], n_epochs_run=len(js['val_loss'])))
    return min(cands, key=lambda c: c['best_val_loss'])


def safe_metrics(checkpoint_path: Path, data_path: str):
    try:
        res = evaluate_model(str(checkpoint_path), data_path, device=DEVICE, verbose=False)
        return res['metrics']
    except Exception as e:
        return {'error': str(e)}


def final_retrain_p30(arch: str, hp: dict, max_epochs: int):
    model_kwargs = dict(BASELINE_ARCH_KWARGS[arch])
    dropout = hp.get('dropout')
    if dropout is not None:
        model_kwargs['dropout'] = dropout

    model_name = f"{arch}_hpo_p30"
    ckpt_path = guard_path(P.models_dir / f"{model_name}_best.pt")

    log(arch, f"Vegso ujratanitas indul (max_epochs={max_epochs}, patience={FINAL_PATIENCE}, "
              f"batch={FINAL_BATCH_SIZE}, seed={BASE_SEED}) lr={hp['lr']!r}, "
              f"weight_decay={hp['weight_decay']!r}, dropout={dropout!r}, "
              f"model_kwargs={model_kwargs}")

    torch.manual_seed(BASE_SEED)
    model = get_model(arch, **model_kwargs)

    t0 = time.time()
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
    dt = time.time() - t0
    n_run = len(history['val_loss'])
    log(arch, f"Vegso ujratanitas kesz ({dt:.1f}s): best_epoch={history['best_epoch']}, "
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
    p30_count = count_parameters(get_model(arch, **model_kwargs))
    result = {
        'arch': arch,
        'baseline_checkpoint': rel(baseline_ckpt_path),
        'baseline_model_kwargs': baseline_ckpt['model_kwargs'],
        'hpo_model_kwargs': model_kwargs,
        'baseline_param_count': baseline_count,
        'hpo_param_count': p30_count,
        'match': baseline_count == p30_count,
    }
    write_json(P.results_dir / f"param_count_check_{arch}.json", result)
    log(arch, f"Parameterszam-ellenorzes: baseline={baseline_count:,} vs hpo_p30={p30_count:,} "
              f"-> match={result['match']}")
    if not result['match']:
        raise RuntimeError(f"[{arch}] PARAMETERSZAM ELTERES! baseline={baseline_count} "
                           f"!= hpo_p30={p30_count}.")
    return result


def reproducibility_check(history: dict, p10_hist: pd.DataFrame):
    n_common = min(len(history['val_loss']), len(p10_hist))
    out = {'n_common_epochs': n_common, 'columns': {}}
    first_diff_any = None
    for col in ('train_loss', 'val_loss', 'lr'):
        a = np.asarray(history[col][:n_common], dtype=np.float64)
        b = p10_hist[col].to_numpy(dtype=np.float64)[:n_common]
        diff = np.abs(a - b)
        rel = diff / np.maximum(np.abs(b), 1e-300)
        neq = np.nonzero(a != b)[0]
        first = int(neq[0]) + 1 if len(neq) else None
        out['columns'][col] = {
            'max_abs_diff': float(diff.max()) if n_common else None,
            'max_rel_diff': float(rel.max()) if n_common else None,
            'first_differing_epoch': first,
            'n_differing_epochs': int(len(neq)),
        }
        if first is not None:
            first_diff_any = first if first_diff_any is None else min(first_diff_any, first)
    out['first_differing_epoch'] = first_diff_any
    out['bitwise_identical'] = first_diff_any is None
    out['max_abs_diff'] = max(c['max_abs_diff'] for c in out['columns'].values()) if n_common else None
    return out


def run_arch(arch: str, max_epochs: int):
    t_start = time.time()
    src_json = P10_RESULTS_DIR / f"best_params_{arch}.json"
    bp10, p10_hist = p10_info(arch)
    hp = {'lr': bp10['lr'], 'weight_decay': bp10['weight_decay'], 'dropout': bp10['dropout']}
    log(arch, f"=== {arch} ({CLASS_NAME[arch]}) p30 ujratanitas -- device={DEVICE}, "
              f"hiperparameterek forrasa: {rel(src_json)} ===")

    history, model_kwargs, retrain_seconds, ckpt = final_retrain_p30(arch, hp, max_epochs)
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
    log(arch, f"Val MSE ujraszamolva: {val_recheck}")

    p30_test = safe_metrics(ckpt, TEST_PATH)
    p10_test = safe_metrics(P10_MODELS_DIR / f"{arch}_hpo_best.pt", TEST_PATH)
    v1_test = safe_metrics(onlab_ckpt(arch, 'v1'), TEST_PATH)
    v2_test = safe_metrics(onlab_ckpt(arch, 'v2'), TEST_PATH) if onlab_ckpt(arch, 'v2').exists() else None
    ob = onlab_best(arch)
    ob_test = v1_test if ob['version'] == 'v1' else v2_test

    repro = reproducibility_check(history, p10_hist)
    log(arch, f"Reprodukalhatosag (p30 vs p10, {repro['n_common_epochs']} kozos epoch): "
              f"bitre azonos={repro['bitwise_identical']}, "
              f"elso elteres={repro['first_differing_epoch']}, max abs elteres={repro['max_abs_diff']}")

    b_val = baseline_val_loss(arch)
    h_val = history['best_val_loss']
    pct = lambda ref: (ref - h_val) / ref * 100.0 if ref else None
    n_run = len(history['val_loss'])

    out = {
        'arch': arch,
        'model_class': CLASS_NAME[arch],
        'architecture_size_fixed_baseline': BASELINE_ARCH_KWARGS[arch],
        'lr': hp['lr'],
        'weight_decay': hp['weight_decay'],
        'dropout': hp['dropout'],
        'search_source': rel(src_json),
        'search_best_val_loss': bp10['search_best_val_loss'],
        'search_best_trial_number': bp10['search_best_trial_number'],
        'n_trials_completed': bp10['n_trials_completed'],
        'n_trials_target': bp10['n_trials_target'],
        'final_patience': FINAL_PATIENCE,
        'final_max_epochs': max_epochs,
        'final_batch_size': FINAL_BATCH_SIZE,
        'seed': BASE_SEED,
        'final_retrain_best_epoch': history['best_epoch'],
        'final_retrain_n_epochs_run': n_run,
        'hit_max_epochs': n_run >= max_epochs,
        'final_retrain_best_val_loss': h_val,
        'baseline_val_loss': b_val,
        'improvement_pct_vs_baseline_val_loss': pct(b_val),
        'param_count_match': pcc['match'],
        'param_count': pcc['hpo_param_count'],
        'baseline_test_metrics': v1_test,
        'hpo_test_metrics': p30_test,
        'retrain_seconds': retrain_seconds,
        'p10_patience': P10_PATIENCE,
        'p10_best_epoch': bp10['final_retrain_best_epoch'],
        'p10_n_epochs_run': int(len(p10_hist)),
        'p10_best_val_loss': bp10['final_retrain_best_val_loss'],
        'p10_test_metrics': p10_test,
        'p10_retrain_seconds': bp10.get('retrain_seconds'),
        'improvement_pct_vs_p10_val_loss': pct(bp10['final_retrain_best_val_loss']),
        'onlab_best_ref': ob['version'],
        'onlab_best_val_loss': ob['best_val_loss'],
        'onlab_best_epoch': ob['best_epoch'],
        'onlab_best_n_epochs_run': ob['n_epochs_run'],
        'onlab_best_test_metrics': ob_test,
        'onlab_v2_test_metrics': v2_test,
        'improvement_pct_vs_onlab_best_val_loss': pct(ob['best_val_loss']),
        'val_mse_recheck': val_recheck,
        'reproducibility_check': repro,
    }
    write_json(P.results_dir / f"best_params_{arch}.json", out)
    log(arch, f"=== {arch} KESZ ({time.time() - t_start:.1f}s). p10_val={bp10['final_retrain_best_val_loss']:.6e}, "
              f"p30_val={h_val:.6e}, onlab_best({ob['version']})={ob['best_val_loss']:.6e} ===")
    return out


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


def _load_rows():
    rows = []
    for arch in ARCH_ORDER:
        p = P.results_dir / f"best_params_{arch}.json"
        if p.exists():
            rows.append(read_json(p))
    return rows


def write_summary(out_path=None):
    out_path = Path(out_path) if out_path is not None else P.results_dir / "hpo_p30_summary.md"
    rows = _load_rows()
    rmse = lambda m: (m or {}).get('RMSE')
    L = []

    L.append("## Teszt-RMSE és validációs MSE\n")
    L.append("| Architektúra | Paraméter | Teszt-RMSE, kiindulási | Teszt-RMSE, HPO p30 | "
             "Változás | Val MSE, kiindulási | Val MSE, HPO p30 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in rows:
        b, h = rmse(d.get('baseline_test_metrics')), rmse(d.get('hpo_test_metrics'))
        params = f"{d['param_count']:,}".replace(",", " ")
        L.append(f"| {d['model_class']} | {params} | {_sci(b)} | {_sci(h)} | {_change(b, h)} | "
                 f"{_sci(d['baseline_val_loss'])} | {_sci(d['final_retrain_best_val_loss'])} |")

    L.append("\n## Hiperparaméterek\n")
    L.append("| Architektúra | Trial | lr | weight_decay | dropout |")
    L.append("|---|---:|---:|---:|---:|")
    for d in rows:
        dr = f"{d['dropout']:.3f}" if d['dropout'] is not None else "–"
        L.append(f"| {d['model_class']} | {d['n_trials_completed']} | {_sci(d['lr'])} | "
                 f"{_sci(d['weight_decay'])} | {dr} |")

    L.append("\n## Kiindulási legjobb, HPO p10 és HPO p30\n")
    L.append("| Architektúra | Modell | Val MSE | Teszt-RMSE | Teszt max. hiba | Legjobb / futott epoch |")
    L.append("|---|---|---:|---:|---:|---:|")
    for d in rows:
        variants = [
            (f"kiindulási legjobb ({d['onlab_best_ref']})", d['onlab_best_val_loss'],
             d['onlab_best_test_metrics'], d['onlab_best_epoch'], d['onlab_best_n_epochs_run']),
            ("HPO p10", d['p10_best_val_loss'], d['p10_test_metrics'],
             d['p10_best_epoch'], d['p10_n_epochs_run']),
            ("HPO p30", d['final_retrain_best_val_loss'], d['hpo_test_metrics'],
             d['final_retrain_best_epoch'], d['final_retrain_n_epochs_run']),
        ]
        for name, val, tm, be, nr in variants:
            tm = tm or {}
            L.append(f"| {d['model_class']} | {name} | {_sci(val)} | {_sci(tm.get('RMSE'))} | "
                     f"{_sci(tm.get('max_error'))} | {be} / {nr} |")

    L.append("\n## A HPO p30 változása (val MSE) és a tanítási idő\n")
    L.append("| Architektúra | A HPO p10-hez képest | A kiindulási legjobbhoz képest | "
             "Tanítási idő p30 (s) | Tanítási idő p10 (s) |")
    L.append("|---|---:|---:|---:|---:|")
    for d in rows:
        h = d['final_retrain_best_val_loss']
        L.append(f"| {d['model_class']} | {_change(d['p10_best_val_loss'], h)} | "
                 f"{_change(d['onlab_best_val_loss'], h)} | {d['retrain_seconds']:.0f} | "
                 f"{d['p10_retrain_seconds']:.0f} |")

    L.append("\n## A p30 és a p10 tanítási görbéje a közös epochokon\n")
    L.append("| Architektúra | Közös epoch | Bitre azonos | Első eltérő epoch | "
             "Max. eltérés, train_loss | Max. eltérés, val_loss | Max. eltérés, lr |")
    L.append("|---|---:|:---:|---:|---:|---:|---:|")
    for d in rows:
        r = d['reproducibility_check']
        c = r['columns']
        first = r['first_differing_epoch']
        L.append(f"| {d['model_class']} | {r['n_common_epochs']} | "
                 f"{'igen' if r['bitwise_identical'] else 'nem'} | "
                 f"{first if first is not None else '–'} | "
                 f"{_sci(c['train_loss']['max_abs_diff'])} | {_sci(c['val_loss']['max_abs_diff'])} | "
                 f"{_sci(c['lr']['max_abs_diff'])} |")

    L.append("\n## Checkpoint-ellenőrzés: újraszámolt és tárolt val MSE\n")
    L.append("| Architektúra | Tárolt val MSE | Újraszámolt val MSE | Relatív eltérés |")
    L.append("|---|---:|---:|---:|")
    for d in rows:
        v = d['val_mse_recheck']
        L.append(f"| {d['model_class']} | {_sci(v['checkpoint_val_loss'])} | "
                 f"{_sci(v['evaluate_model_val_mse'])} | {_sci(v.get('rel_diff'))} |")

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
    parser.add_argument('--out-root', type=str, default=None,
                        help="Kimeneti gyoker (alapertelmezes: hpo/). Fustprobahoz scratch mappa.")
    parser.add_argument('--smoke', type=int, default=None,
                        help="Fustproba: max_epochs felulirasa (csak --out-root-tal, nem a hpo/ ala).")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve() if args.out_root else HPO_DIR
    if args.smoke is not None and _is_within(out_root, HPO_DIR):
        raise SystemExit("[smoke] A fustproba kimenete nem kerulhet a hpo/ ala -- adj meg --out-root-ot.")
    for d in PROTECTED_DIRS:
        if _is_within(out_root, d):
            raise SystemExit(f"[guard] Az --out-root vedett mappaban van: {rel(out_root)}")
    P = Paths(out_root)
    P.ensure()
    max_epochs = args.smoke if args.smoke is not None else FINAL_MAX_EPOCHS

    sys.stdout = _Tee(sys.stdout, P.logs_dir / "campaign_stdout.log")

    if args.summary_only:
        write_summary()
        return

    if args.arch:
        archs = [args.arch]
    elif args.archs:
        archs = [a.strip() for a in args.archs.split(',') if a.strip()]
    else:
        archs = ARCH_ORDER

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] p30 ujratanitas indul. Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kimeneti gyoker: {rel(P.out_root)}; max_epochs={max_epochs}, patience={FINAL_PATIENCE}")
    print(f"Futtatando architekturak: {archs}")

    t0 = time.time()
    for arch in archs:
        try:
            run_arch(arch, max_epochs)
        except Exception:
            err = traceback.format_exc()
            with open(guard_path(P.logs_dir / f"{arch}_ERROR.log"), 'a', encoding='utf-8') as f:
                f.write(err + "\n")
            print(f"[HIBA] {arch}:\n{err}")

    write_summary()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] p30 kampany kesz: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
