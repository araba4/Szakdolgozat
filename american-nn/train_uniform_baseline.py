import sys
sys.dont_write_bytecode = True

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AMNN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AMNN_DIR))

import common as C
from common import (
    ARCH_ORDER, CLASS_NAME, BASELINE_ARCH_KWARGS, HAS_DROPOUT, BASE_SEED, DEVICE,
    FEATURE_COLS, TARGETS, PricingDataset, get_model, count_parameters,
    load_model, compute_metrics, guard_path, write_json, read_json, rel,
)


UNIFORM_HP = dict(lr=1e-3, weight_decay=1e-4, dropout=0.1, batch_size=4096, max_epochs=200)
PATIENCES = [10, 30]
TARGET_ORDER = ['american', 'european']


def run_name(arch, target, patience):
    return f"{arch}_{target}_p{patience}"


class Paths:
    def __init__(self, out_root: Path):
        self.out_root = out_root.resolve()
        self.models_dir = self.out_root / "models" / "uniform"
        self.results_dir = self.out_root / "results" / "uniform"
        self.logs_dir = self.out_root / "logs" / "uniform"

    def ensure(self):
        for d in (self.models_dir, self.results_dir, self.logs_dir):
            guard_path(d).mkdir(parents=True, exist_ok=True)

    def ckpt(self, run):
        return self.models_dir / f"{run}_best.pt"


P = Paths(AMNN_DIR)


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


def log(run: str, msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(guard_path(P.logs_dir / f"{run}.log"), 'a', encoding='utf-8') as f:
        f.write(line + "\n")


PAIR_COLS = ['S', 'K', 'T', 'r', 'sigma', 'moneyness'] + FEATURE_COLS


def pairing_check() -> dict:
    out = {'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'american_files': {}, 'onlab_files': {}, 'splits': {}}
    all_ok = True
    for split in C.SPLITS:
        pa, pe = C.am_split_path(split), C.onlab_split_path(split)
        out['american_files'][split] = rel(pa)
        out['onlab_files'][split] = rel(pe)
        a = pd.read_parquet(pa, columns=PAIR_COLS + ['european_put_price'])
        e = pd.read_parquet(pe, columns=PAIR_COLS + ['put_price'])
        s = {'n_rows_american': len(a), 'n_rows_onlab': len(e), 'columns': {}}
        ok = len(a) == len(e)
        if ok:
            for col in PAIR_COLS:
                x, y = a[col].to_numpy(), e[col].to_numpy()
                s['columns'][col] = {
                    'bitwise_equal': bool(np.array_equal(x, y)),
                    'allclose_rtol0_atol1e-12': bool(np.allclose(x, y, rtol=0, atol=1e-12)),
                    'max_abs_diff': float(np.max(np.abs(x - y))),
                }
                ok &= s['columns'][col]['allclose_rtol0_atol1e-12']
            x, y = a['european_put_price'].to_numpy(), e['put_price'].to_numpy()
            s['european_put_price_vs_onlab_put_price'] = {
                'bitwise_equal': bool(np.array_equal(x, y)),
                'max_abs_diff': float(np.max(np.abs(x - y))),
                'max_abs_diff_norm_by_K': float(np.max(np.abs(x - y) / a['K'].to_numpy())),
                'allclose_rtol0_atol1e-10': bool(np.allclose(x, y, rtol=0, atol=1e-10)),
            }
            ok &= s['european_put_price_vs_onlab_put_price']['allclose_rtol0_atol1e-10']
            neg = x < 0
            s['european_put_price_negative'] = {
                'n_negative': int(neg.sum()),
                'min': float(x.min()),
                'note': 'lebegopontos zaj, nem klippeljuk',
            }
        s['paired'] = bool(ok)
        out['splits'][split] = s
        all_ok &= ok
    out['all_paired'] = bool(all_ok)
    write_json(P.results_dir / "pairing_check.json", out)
    print(f"[parositas] all_paired={all_ok} -> {rel(P.results_dir / 'pairing_check.json')}")
    return out


_DS_CACHE = {}


def get_ds(split, target):
    key = (split, target)
    if key not in _DS_CACHE:
        _DS_CACHE[key] = PricingDataset.from_split(split, target, device=DEVICE)
    return _DS_CACHE[key]


def model_kwargs_for(arch):
    kw = dict(BASELINE_ARCH_KWARGS[arch])
    if HAS_DROPOUT[arch]:
        kw['dropout'] = UNIFORM_HP['dropout']
    return kw


def param_count_check(arch, model_kwargs):
    ck_path = C.ONLAB_MODELS_DIR / f"{arch}_best.pt"
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    base = count_parameters(get_model(ck['model_class'], **ck['model_kwargs']))
    ours = count_parameters(get_model(arch, **model_kwargs))
    res = {'onlab_v1_checkpoint': rel(ck_path), 'onlab_v1_model_kwargs': ck['model_kwargs'],
           'model_kwargs': model_kwargs, 'onlab_v1_param_count': base, 'param_count': ours,
           'param_count_match': base == ours,
           'model_kwargs_match': ck['model_kwargs'] == model_kwargs}
    if not (res['param_count_match'] and res['model_kwargs_match']):
        raise RuntimeError(f"[{arch}] Parameterszam/model_kwargs elteres az Onlab v1-tol: {res}")
    return res


_EDGES = None


def premium_edges():
    global _EDGES
    if _EDGES is None:
        df = C.load_am_split('test')
        _EDGES = C.premium_bucket_edges((df['early_exercise_premium'] / df['K']).to_numpy())
    return _EDGES


def evaluate_run(run, target, ckpt_path):
    model, meta = load_model(str(ckpt_path), device=DEVICE)
    stored = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    val_ds = get_ds('val', target)
    val_pred = C.predict(model, val_ds)
    val_true = val_ds.y.cpu().numpy().ravel()
    val_mse = float(np.mean((val_true.astype(np.float64) - val_pred.astype(np.float64)) ** 2))
    val_recheck = {'checkpoint_val_loss': stored['val_loss'],
                   'checkpoint_best_epoch': stored['best_epoch'],
                   'reloaded_val_mse': val_mse,
                   'rel_diff': abs(val_mse - stored['val_loss']) / stored['val_loss'],
                   'loader': 'Önlab src/evaluate.py load_model + PricingDataset'}

    test_ds = get_ds('test', target)
    y_pred = C.predict(model, test_ds)
    y_true = test_ds.y.cpu().numpy().ravel()
    df = C.load_am_split('test')
    K = df['K'].to_numpy()

    m = compute_metrics(y_true, y_pred)
    m['MSE'] = m['RMSE'] ** 2
    usd = compute_metrics(y_true * K, y_pred * K)
    usd = {k: v for k, v in usd.items() if k != 'MAPE (%)'}

    masks = C.segment_masks(df, premium_edges())
    seg_rows = []
    for group in ('moneyness', 'lejarat'):
        for r in C.masked_metrics(y_true, y_pred, masks[group], scale=K):
            seg_rows.append({'run': run, 'group': group, **r})
    prem_rows = [{'run': run, 'group': 'premium', **r}
                 for r in C.masked_metrics(y_true, y_pred, masks['premium'], scale=K)]
    pd.DataFrame(seg_rows).to_csv(guard_path(P.results_dir / f"segmented_{run}.csv"), index=False)
    pd.DataFrame(prem_rows).to_csv(guard_path(P.results_dir / f"premium_buckets_{run}.csv"), index=False)
    return dict(val_recheck=val_recheck, test_metrics=m, test_metrics_usd=usd,
                checkpoint_target_col=meta['target_col'],
                checkpoint_model_kwargs=meta['model_kwargs'],
                checkpoint_feature_cols=meta['feature_cols'])


def run_one(arch, target, patience, max_epochs):
    run = run_name(arch, target, patience)
    t_start = time.time()
    kw = model_kwargs_for(arch)
    spec = TARGETS[target]
    log(run, f"=== {run} ({CLASS_NAME[arch]}) -- cel: {spec['formula']}, patience={patience}, "
             f"max_epochs={max_epochs}, lr={UNIFORM_HP['lr']}, wd={UNIFORM_HP['weight_decay']}, "
             f"batch={UNIFORM_HP['batch_size']}, seed={BASE_SEED}, model_kwargs={kw} ===")
    pcc = param_count_check(arch, kw)

    train_ds, val_ds = get_ds('train', target), get_ds('val', target)
    ckpt = guard_path(P.ckpt(run))
    extra = {'target': target, 'target_definition': spec['formula'],
             'data': rel(C.AM_DATA_DIR), 'patience': patience, 'max_epochs': max_epochs,
             'seed': BASE_SEED, 'lr': UNIFORM_HP['lr'], 'weight_decay': UNIFORM_HP['weight_decay'],
             'batch_size': UNIFORM_HP['batch_size'], 'experiment': 'uniform baseline'}

    torch.manual_seed(BASE_SEED)
    model = get_model(arch, **kw)
    t0 = time.time()
    hist = C.train_loop(model, train_ds, val_ds, output_dir=P.models_dir, model_name=run,
                        model_class=arch, model_kwargs=kw, feature_cols=FEATURE_COLS,
                        target_col=spec['target_col'], batch_size=UNIFORM_HP['batch_size'],
                        max_epochs=max_epochs, lr=UNIFORM_HP['lr'],
                        weight_decay=UNIFORM_HP['weight_decay'], patience=patience,
                        device=DEVICE, extra_meta=extra, checkpoint_path=ckpt)
    train_seconds = time.time() - t0
    n_run = len(hist['val_loss'])
    pd.DataFrame({'epoch': range(1, n_run + 1), 'train_loss': hist['train_loss'],
                  'val_loss': hist['val_loss'], 'lr': hist['lr'],
                  'epoch_seconds': hist['epoch_seconds']}).to_csv(
        guard_path(P.results_dir / f"training_history_{run}.csv"), index=False)
    log(run, f"Tanitas kesz ({train_seconds:.1f}s): best_epoch={hist['best_epoch']}, "
             f"futott={n_run}, best_val_loss={hist['best_val_loss']:.8e}")

    ev = evaluate_run(run, target, ckpt)
    out = {
        'run': run, 'arch': arch, 'model_class': CLASS_NAME[arch], 'target': target,
        'target_col': spec['target_col'], 'target_definition': spec['formula'],
        'patience': patience, 'max_epochs': max_epochs, 'seed': BASE_SEED,
        'hyperparameters': {k: UNIFORM_HP[k] for k in ('lr', 'weight_decay', 'batch_size')}
                           | {'dropout': kw.get('dropout')},
        'model_kwargs': kw, 'param_count': pcc['param_count'], 'param_count_check': pcc,
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
             f"{ev['val_recheck']['rel_diff']:.2e}, target_col={ev['checkpoint_target_col']} ===")
    return out


def loop_check(n_epochs: int, arch: str = 'dense_mlp'):
    from src.train import train_model, OptionDataset, DEFAULT_TARGET_COL
    work = guard_path(P.results_dir / "loop_check")
    work.mkdir(parents=True, exist_ok=True)
    kw = model_kwargs_for(arch)
    tr, va = str(C.onlab_split_path('train')), str(C.onlab_split_path('val'))

    torch.manual_seed(BASE_SEED)
    m1 = get_model(arch, **kw)
    h1 = train_model(m1, tr, va, output_dir=str(work), model_name=f"onlab_train_model_{arch}",
                     model_class=arch, model_kwargs=kw, max_epochs=n_epochs, patience=10,
                     device=DEVICE)

    dtr = pd.read_parquet(tr, columns=FEATURE_COLS + [DEFAULT_TARGET_COL])
    dva = pd.read_parquet(va, columns=FEATURE_COLS + [DEFAULT_TARGET_COL])
    ds_tr = PricingDataset(dtr[FEATURE_COLS].values, dtr[DEFAULT_TARGET_COL].values,
                           FEATURE_COLS, DEFAULT_TARGET_COL, device=DEVICE)
    ds_va = PricingDataset(dva[FEATURE_COLS].values, dva[DEFAULT_TARGET_COL].values,
                           FEATURE_COLS, DEFAULT_TARGET_COL, device=DEVICE)
    ref = OptionDataset(tr, FEATURE_COLS, DEFAULT_TARGET_COL, device=DEVICE)
    tensors_equal = bool(torch.equal(ref.X, ds_tr.X) and torch.equal(ref.y, ds_tr.y))
    del ref

    torch.manual_seed(BASE_SEED)
    m2 = get_model(arch, **kw)
    h2 = C.train_loop(m2, ds_tr, ds_va, output_dir=work, model_name=f"train_loop_{arch}",
                      model_class=arch, model_kwargs=kw, feature_cols=FEATURE_COLS,
                      target_col=DEFAULT_TARGET_COL, max_epochs=n_epochs, patience=10,
                      device=DEVICE)
    res = {'arch': arch, 'n_epochs': n_epochs, 'data': 'Önlab eredmények/data (call_price_norm)',
           'dataset_tensors_bitwise_equal': tensors_equal}
    for col in ('train_loss', 'val_loss', 'lr'):
        res[col] = {'onlab_train_model': h1[col], 'train_loop': h2[col],
                    'bitwise_equal': h1[col] == h2[col]}
    s1 = torch.load(work / f"onlab_train_model_{arch}_best.pt", map_location='cpu', weights_only=False)
    s2 = torch.load(work / f"train_loop_{arch}_best.pt", map_location='cpu', weights_only=False)
    res['state_dict_bitwise_equal'] = all(torch.equal(s1['state_dict'][k], s2['state_dict'][k])
                                          for k in s1['state_dict'])
    res['all_equal'] = bool(tensors_equal and res['state_dict_bitwise_equal']
                            and all(res[c]['bitwise_equal'] for c in ('train_loss', 'val_loss', 'lr')))
    for f in (f"onlab_train_model_{arch}_best.pt", f"train_loop_{arch}_best.pt"):
        (work / f).unlink()
    write_json(P.results_dir / "loop_check.json", res)
    print(f"[loop-check] {arch}, {n_epochs} epoch: all_equal={res['all_equal']}")
    return res


def main():
    global P
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', type=str, default=None)
    ap.add_argument('--archs', type=str, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--target', choices=['american', 'european', 'both'], default='both')
    ap.add_argument('--patience', choices=['10', '30', 'both'], default='both')
    ap.add_argument('--skip-done', action='store_true')
    ap.add_argument('--smoke', type=int, default=None,
                    help="Fustproba: max_epochs felulirasa (csak az american-nn/-en kivuli --out-root-tal).")
    ap.add_argument('--out-root', type=str, default=None)
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--loop-check', type=int, default=None,
                    help="Ciklus-ekvivalencia ellenorzes N epochra (Onlab train_model vs. train_loop).")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve() if args.out_root else AMNN_DIR
    if args.smoke is not None and C._is_within(out_root, AMNN_DIR):
        raise SystemExit("[smoke] A fustproba kimenete nem kerulhet az american-nn/ ala -- adj meg --out-root-ot.")
    C.set_allowed_roots(out_root)
    P = Paths(out_root)
    P.ensure()
    sys.stdout = _Tee(sys.stdout, P.logs_dir / "campaign_stdout.log")

    if args.summary_only:
        import summarize_uniform as S
        S.main(P)
        return

    if args.loop_check is not None:
        loop_check(args.loop_check)
        return

    pc = pairing_check()
    if not pc['all_paired']:
        raise SystemExit("[parositas] A ket adathalmaz NEM paros -- a kampany leall.")

    if args.arch:
        archs = [args.arch]
    elif args.archs:
        archs = [a.strip() for a in args.archs.split(',') if a.strip()]
    else:
        archs = ARCH_ORDER
    targets = TARGET_ORDER if args.target == 'both' else [args.target]
    pats = PATIENCES if args.patience == 'both' else [int(args.patience)]
    max_epochs = args.smoke if args.smoke is not None else UNIFORM_HP['max_epochs']

    write_json(P.results_dir / "premium_bucket_edges.json",
               {'source': rel(C.am_split_path('test')), 'quantity': 'early_exercise_premium / K',
                'zero_threshold': C.PREMIUM_ZERO_THRESHOLD, 'edges': premium_edges(),
                'labels': C.premium_bucket_labels(premium_edges())})
    meta_path = P.results_dir / "campaign_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {'invocations': []}
    meta.update({'env': C.env_meta(), 'uniform_hyperparameters': UNIFORM_HP,
                 'scheduler': "ReduceLROnPlateau(mode='min', factor=0.5, patience=5, min_lr=1e-6)",
                 'optimizer': 'Adam', 'loss': 'MSELoss', 'feature_cols': FEATURE_COLS,
                 'seed': BASE_SEED, 'targets': {t: TARGETS[t]['formula'] for t in TARGET_ORDER}})
    meta['invocations'].append({'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'argv': sys.argv[1:],
                                'max_epochs': max_epochs})
    write_json(meta_path, meta)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Egyseges baseline indul. Device: {DEVICE} "
          f"({C.env_meta().get('gpu_name')}); kimenet: {P.out_root}; max_epochs={max_epochs}")
    runs = [(a, t, p) for a in archs for t in targets for p in pats]
    print(f"Futtatando: {len(runs)} tanitas: {[run_name(*r) for r in runs]}")
    t0 = time.time()
    n_done = n_skip = n_err = 0
    for arch, target, pat in runs:
        run = run_name(arch, target, pat)
        if args.skip_done and (P.results_dir / f"metrics_{run}.json").exists() and P.ckpt(run).exists():
            print(f"[skip-done] {run} mar kesz, kihagyva.")
            n_skip += 1
            continue
        try:
            run_one(arch, target, pat, max_epochs)
            n_done += 1
        except Exception:
            err = traceback.format_exc()
            with open(guard_path(P.logs_dir / f"{run}_ERROR.log"), 'a', encoding='utf-8') as f:
                f.write(err + "\n")
            print(f"[HIBA] {run}:\n{err}")
            n_err += 1
    meta = read_json(meta_path)
    meta['invocations'][-1].update({'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'wall_seconds': time.time() - t0, 'n_trained': n_done,
                                    'n_skipped': n_skip, 'n_errors': n_err})
    write_json(meta_path, meta)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Kampany kesz: {time.time() - t0:.1f}s, "
          f"tanitva={n_done}, kihagyva={n_skip}, hiba={n_err}")

    import summarize_uniform as S
    S.main(P)


if __name__ == '__main__':
    main()
