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
    ARCH_ORDER, CLASS_NAME, BASELINE_ARCH_KWARGS, HAS_DROPOUT, DEVICE,
    FEATURE_COLS, TARGETS, get_model, count_parameters, load_model, compute_metrics,
    guard_path, write_json, read_json, rel,
)
import train_uniform_baseline as TU

RETUNE_RESULTS_DIR = AMNN_DIR / "results" / "retune"
RETUNE_MODELS_DIR = AMNN_DIR / "models" / "retune"
UNIFORM_RESULTS_DIR = AMNN_DIR / "results" / "uniform"
HPO_P30_DIR = C.REPO_ROOT / "hpo" / "results" / "p30"

TARGET = 'american_minus_bs'
PRICE_TARGET = 'american'
BASE_TARGET = 'european'
PATIENCE = 30
EXPECTED_FINAL = {'final_max_epochs': 200, 'final_batch_size': 4096, 'seed': 42, 'final_patience': 30}


def run_name(arch):
    return f"{arch}_residual_retune_p{PATIENCE}"


def direct_run_name(arch):
    return f"{arch}_american_retune_p{PATIENCE}"


class Paths:
    def __init__(self, out_root: Path):
        self.out_root = out_root.resolve()
        self.models_dir = self.out_root / "models" / "residual"
        self.results_dir = self.out_root / "results" / "residual"
        self.logs_dir = self.out_root / "logs" / "residual"

    def ensure(self):
        for d in (self.models_dir, self.results_dir, self.logs_dir):
            guard_path(d).mkdir(parents=True, exist_ok=True)

    def ckpt(self, run):
        return self.models_dir / f"{run}_best.pt"


P = Paths(AMNN_DIR)


def log(run: str, msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(guard_path(P.logs_dir / f"{run}.log"), 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def _q(x, qs=(0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)):
    return {f"q{int(round(q * 100)):02d}": float(np.quantile(x, q)) for q in qs}


def target_stats() -> dict:
    out = {'definition': TARGETS[TARGET]['formula'], 'splits': {}}
    for split in C.SPLITS:
        df = C.load_am_split(split)
        K = df['K'].to_numpy(np.float64)
        res = C.target_values(df, TARGET)
        am = C.target_values(df, PRICE_TARGET)
        bs = C.target_values(df, BASE_TARGET)
        prem = df['early_exercise_premium'].to_numpy(np.float64) / K
        res_usd = res * K
        recon = bs + res
        neg = res < 0
        s = {
            'n': int(len(df)),
            'residual_norm': {'min': float(res.min()), 'max': float(res.max()), 'mean': float(res.mean()),
                              'median': float(np.median(res)), 'std': float(res.std()), **_q(res)},
            'residual_usd': {'min': float(res_usd.min()), 'max': float(res_usd.max()),
                             'median': float(np.median(res_usd)), 'mean': float(res_usd.mean())},
            'negative': {
                'n_lt0': int(neg.sum()), 'share_lt0': float(neg.mean()),
                'n_lt_minus1e-6': int((res < -1e-6).sum()),
                'share_lt_minus1e-6': float((res < -1e-6).mean()),
                'min_norm': float(res.min()), 'min_usd': float(res_usd.min()),
                'median_of_negatives_norm': float(np.median(res[neg])) if neg.any() else None,
                'median_of_negatives_usd': float(np.median(res_usd[neg])) if neg.any() else None,
                'mean_of_negatives_usd': float(res_usd[neg].mean()) if neg.any() else None,
                'moneyness_median_of_negatives': float(np.median(df['moneyness'].to_numpy()[neg])) if neg.any() else None,
                'T_median_of_negatives': float(np.median(df['T'].to_numpy()[neg])) if neg.any() else None,
                'share_negatives_otm': float((df['moneyness'].to_numpy()[neg] >= 1.1).mean()) if neg.any() else None,
                'share_negatives_short_T': float((df['T'].to_numpy()[neg] < 0.25).mean()) if neg.any() else None,
            },
            'american_norm': {'min': float(am.min()), 'max': float(am.max()), 'mean': float(am.mean()),
                              'median': float(np.median(am)), 'std': float(am.std())},
            'scale_vs_american': {
                'std_ratio_residual_over_american': float(res.std() / am.std()),
                'mean_abs_ratio': float(np.abs(res).mean() / np.abs(am).mean()),
                'max_ratio': float(res.max() / am.max()),
                'mean_square_ratio': float(np.mean(res ** 2) / np.mean(am ** 2)),
                'note': 'A reziduum varianciája az ár/K-énak ennyiszerese; az MSE-veszteség és a gradiensek '
                        'skálája ennyivel kisebb (a hiperparaméterek az árra hangolódtak).',
            },
            'residual_minus_premium': {
                'definition': '(american - BS)/K - premium/K = (crr_european - BS)/K',
                'max_abs_norm': float(np.abs(res - prem).max()),
                'max_abs_usd': float(np.abs((res - prem) * K).max()),
                'median_abs_usd': float(np.median(np.abs((res - prem) * K))),
                'check_vs_crr_eu_minus_bs_max_abs_usd': float(np.abs(
                    (res - prem) * K - (df['crr_european_put_price'].to_numpy() - df['european_put_price'].to_numpy())).max()),
            },
            'premium_norm': {'min': float(prem.min()), 'max': float(prem.max()),
                             'median': float(np.median(prem))},
            'reconstruction': {
                'formula': 'european_put_price/K + target == american_put_price/K (float64)',
                'max_abs_diff_norm': float(np.abs(recon - am).max()),
                'n_not_bitwise_equal': int((recon != am).sum()),
                'allclose_rtol0_atol1e-15': bool(np.allclose(recon, am, rtol=0, atol=1e-15)),
                'float32_target_reconstruction_max_abs_diff': float(np.abs(
                    bs + res.astype(np.float32).astype(np.float64) - am).max()),
            },
        }
        out['splits'][split] = s
    out['all_reconstruction_ok'] = all(s['reconstruction']['allclose_rtol0_atol1e-15']
                                       for s in out['splits'].values())
    write_json(P.results_dir / "target_stats.json", out)
    tr = out['splits']['train']
    print(f"[target-stats] train: min={tr['residual_norm']['min']:.3e}, max={tr['residual_norm']['max']:.3e}, "
          f"median={tr['residual_norm']['median']:.3e}, negativ={tr['negative']['n_lt0']} "
          f"({tr['negative']['share_lt0']:.2%}), szoras-arany={tr['scale_vs_american']['std_ratio_residual_over_american']:.3f}, "
          f"rekonstrukcio ok={out['all_reconstruction_ok']}")
    return out


def target_check() -> dict:
    res = {'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'direct_formula': {}, 'retune_models': []}
    for split in C.SPLITS:
        df = C.load_am_split(split)
        K = df['K'].to_numpy(np.float64)
        for t, col in (('american', 'american_put_price'), ('european', 'european_put_price')):
            ref = df[col].to_numpy(np.float64) / K
            res['direct_formula'][f"{split}:{t}"] = bool(np.array_equal(C.target_values(df, t), ref))
    for arch in ARCH_ORDER:
        for t in ('american', 'european'):
            run = f"{arch}_{t}_retune_p30"
            ck = RETUNE_MODELS_DIR / f"{run}_best.pt"
            mj = read_json(RETUNE_RESULTS_DIR / f"metrics_{run}.json")
            model, meta = load_model(str(ck), device=DEVICE)
            vds, tds = TU.get_ds('val', t), TU.get_ds('test', t)
            vp = C.predict(model, vds)
            vt = vds.y.cpu().numpy().ravel()
            vmse = float(np.mean((vt.astype(np.float64) - vp.astype(np.float64)) ** 2))
            tp = C.predict(model, tds)
            tt = tds.y.cpu().numpy().ravel()
            trmse = compute_metrics(tt, tp)['RMSE']
            res['retune_models'].append({
                'run': run, 'reloaded_val_mse_now': vmse,
                'reloaded_val_mse_011': mj['val_recheck']['reloaded_val_mse'],
                'val_bitwise_equal': vmse == mj['val_recheck']['reloaded_val_mse'],
                'test_rmse_now': float(trmse), 'test_rmse_011': mj['test_metrics']['RMSE'],
                'test_rmse_bitwise_equal': float(trmse) == mj['test_metrics']['RMSE'],
                'target_col': meta['target_col'], 'target_col_expected': TARGETS[t]['target_col'],
            })
            del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    res['all_direct_formula_equal'] = all(res['direct_formula'].values())
    res['all_val_bitwise_equal'] = all(r['val_bitwise_equal'] for r in res['retune_models'])
    res['all_test_bitwise_equal'] = all(r['test_rmse_bitwise_equal'] for r in res['retune_models'])
    res['all_ok'] = bool(res['all_direct_formula_equal'] and res['all_val_bitwise_equal']
                         and res['all_test_bitwise_equal'])
    write_json(P.results_dir / "target_check.json", res)
    print(f"[target-check] keplet={res['all_direct_formula_equal']}, val bitre={res['all_val_bitwise_equal']}, "
          f"teszt bitre={res['all_test_bitwise_equal']} -> {rel(P.results_dir / 'target_check.json')}")
    return res


def best_params_path(arch) -> Path:
    return RETUNE_RESULTS_DIR / f"best_params_{arch}.json"


def load_retune_hp(arch) -> dict:
    src = best_params_path(arch)
    bp = read_json(src)
    if bp.get('arch') != arch or bp.get('target') != 'american':
        raise RuntimeError(f"[{arch}] Hibas best_params JSON: {rel(src)}")
    for k, v in EXPECTED_FINAL.items():
        if bp.get(k) != v:
            raise RuntimeError(f"[{arch}] {rel(src)}: {k}={bp.get(k)!r}, elvart {v!r} -- leallas.")
    if HAS_DROPOUT[arch] and bp.get('dropout') is None:
        raise RuntimeError(f"[{arch}] Hianyzo dropout a {rel(src)} fajlban.")
    hp = {'lr': bp['lr'], 'weight_decay': bp['weight_decay'],
          'dropout': bp['dropout'] if HAS_DROPOUT[arch] else None,
          'batch_size': bp['final_batch_size'], 'max_epochs': bp['final_max_epochs'],
          'seed': bp['seed'], 'patience': bp['final_patience'], 'source_json': rel(src),
          'source_study': bp.get('search_source'), 'source_trial': bp.get('search_best_trial_number')}
    mj = read_json(RETUNE_RESULTS_DIR / f"metrics_{direct_run_name(arch)}.json")
    ck = torch.load(RETUNE_MODELS_DIR / f"{direct_run_name(arch)}_best.pt", map_location='cpu', weights_only=False)
    ck_h = ck['extra_meta']['hyperparameters']
    hp['bitwise_equal_retune_metrics_json'] = all(mj['hyperparameters'][k] == hp[k]
                                                  for k in ('lr', 'weight_decay', 'dropout', 'batch_size'))
    hp['bitwise_equal_retune_checkpoint'] = all(ck_h[k] == hp[k] for k in ('lr', 'weight_decay', 'dropout', 'batch_size'))
    if not (hp['bitwise_equal_retune_metrics_json'] and hp['bitwise_equal_retune_checkpoint']):
        raise RuntimeError(f"[{arch}] A best_params nem egyezik a retune modellel: {hp} vs {mj['hyperparameters']} / {ck_h}")
    return hp


def model_kwargs_for(arch, hp):
    kw = dict(BASELINE_ARCH_KWARGS[arch])
    if HAS_DROPOUT[arch]:
        kw['dropout'] = hp['dropout']
    return kw


def param_count_check(arch, model_kwargs) -> dict:
    ours = count_parameters(get_model(arch, **model_kwargs))
    v1_path = C.ONLAB_MODELS_DIR / f"{arch}_best.pt"
    v1 = torch.load(v1_path, map_location='cpu', weights_only=False)
    v1_count = count_parameters(get_model(v1['model_class'], **v1['model_kwargs']))
    hpo_pcc_path = HPO_P30_DIR / f"param_count_check_{arch}.json"
    hpo_count = read_json(hpo_pcc_path)['hpo_param_count']
    rt_pcc_path = RETUNE_RESULTS_DIR / f"param_count_check_{direct_run_name(arch)}.json"
    rt_count = read_json(rt_pcc_path)['param_count']
    rt_ck_path = RETUNE_MODELS_DIR / f"{direct_run_name(arch)}_best.pt"
    rt_ck = torch.load(rt_ck_path, map_location='cpu', weights_only=False)
    size_kw = {k: v for k, v in model_kwargs.items() if k != 'dropout'}
    res = {'arch': arch, 'model_kwargs': model_kwargs, 'param_count': ours,
           'onlab_v1_checkpoint': rel(v1_path), 'onlab_v1_param_count': v1_count,
           'hpo_p30_param_count_check': rel(hpo_pcc_path), 'hpo_p30_param_count': hpo_count,
           'retune_param_count_check': rel(rt_pcc_path), 'retune_param_count': rt_count,
           'retune_checkpoint': rel(rt_ck_path), 'retune_model_kwargs': rt_ck['model_kwargs'],
           'size_kwargs_equal_baseline_arch_kwargs': size_kw == BASELINE_ARCH_KWARGS[arch],
           'model_kwargs_equal_retune_checkpoint': rt_ck['model_kwargs'] == model_kwargs}
    res['param_count_match'] = bool(ours == v1_count == hpo_count == rt_count)
    res['all_ok'] = bool(res['param_count_match'] and res['size_kwargs_equal_baseline_arch_kwargs']
                         and res['model_kwargs_equal_retune_checkpoint'])
    if not res['all_ok']:
        raise RuntimeError(f"[{arch}] Kapacitas-kontroll serul, leallas: {res}")
    return res


_EDGES = None


def use_uniform_premium_edges() -> dict:
    global _EDGES
    src = UNIFORM_RESULTS_DIR / "premium_bucket_edges.json"
    stored = read_json(src)
    df = C.load_am_split('test')
    recomputed = C.premium_bucket_edges((df['early_exercise_premium'] / df['K']).to_numpy())
    equal = [float(a) for a in stored['edges']] == [float(b) for b in recomputed]
    if not equal:
        raise RuntimeError(f"[premium] A uniform vodorhatarok elternek: {stored['edges']} vs {recomputed}")
    _EDGES = [float(a) for a in stored['edges']]
    out = dict(stored)
    out.update({'copied_from': rel(src), 'recomputed_bitwise_equal': equal})
    write_json(P.results_dir / "premium_bucket_edges.json", out)
    return out


def reconstruct(nn_out: np.ndarray, bs_norm: np.ndarray) -> np.ndarray:
    return (bs_norm.astype(np.float64) + nn_out.astype(np.float64)).astype(np.float32)


def evaluate_residual(run, ckpt_path):
    model, meta = load_model(str(ckpt_path), device=DEVICE)
    stored = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    out = {'checkpoint_target_col': meta['target_col'], 'checkpoint_model_kwargs': meta['model_kwargs'],
           'checkpoint_feature_cols': meta['feature_cols']}

    vres = TU.get_ds('val', TARGET)
    nn_val = C.predict(model, vres)
    vt_res = vres.y.cpu().numpy().ravel()
    val_mse_res = float(np.mean((vt_res.astype(np.float64) - nn_val.astype(np.float64)) ** 2))
    out['val_recheck'] = {'checkpoint_val_loss': stored['val_loss'], 'checkpoint_best_epoch': stored['best_epoch'],
                          'reloaded_val_mse': val_mse_res,
                          'rel_diff': abs(val_mse_res - stored['val_loss']) / stored['val_loss'],
                          'scale': 'reziduum (american − BS)/K',
                          'loader': 'Önlab src/evaluate.py load_model + PricingDataset'}
    dfv = C.load_am_split('val')
    bs_val = C.target_values(dfv, BASE_TARGET)
    vt_price = TU.get_ds('val', PRICE_TARGET).y.cpu().numpy().ravel()
    vp_price = reconstruct(nn_val, bs_val)
    out['val_mse_price'] = float(np.mean((vt_price.astype(np.float64) - vp_price.astype(np.float64)) ** 2))
    out['val_mse_price_note'] = ('a rekonstruált ár (BS/K + NN) MSE-je az american/K float32 célon, '
                                 'a direkt modellek val_recheck.reloaded_val_mse-jével azonos képlettel')

    tres = TU.get_ds('test', TARGET)
    nn_test = C.predict(model, tres)
    df = C.load_am_split('test')
    K = df['K'].to_numpy()
    bs_test = C.target_values(df, BASE_TARGET)
    y_true = TU.get_ds('test', PRICE_TARGET).y.cpu().numpy().ravel()
    y_pred = reconstruct(nn_test, bs_test)

    m = compute_metrics(y_true, y_pred)
    m['MSE'] = m['RMSE'] ** 2
    usd = compute_metrics(y_true * K, y_pred * K)
    out['test_metrics'] = m
    out['test_metrics_usd'] = {k: v for k, v in usd.items() if k != 'MAPE (%)'}
    rt = tres.y.cpu().numpy().ravel()
    mr = compute_metrics(rt, nn_test)
    mr['MSE'] = mr['RMSE'] ** 2
    out['test_metrics_residual_scale'] = {k: v for k, v in mr.items() if k != 'MAPE (%)'}
    tgt64 = C.target_values(df, TARGET)
    out['test_sign'] = {'share_nn_lt0': float((nn_test < 0).mean()), 'share_target_lt0': float((tgt64 < 0).mean()),
                        'nn_min': float(nn_test.min()), 'nn_max': float(nn_test.max())}

    masks = C.segment_masks(df, _EDGES)
    seg_rows = []
    for group in ('moneyness', 'lejarat'):
        for r in C.masked_metrics(y_true, y_pred, masks[group], scale=K):
            seg_rows.append({'run': run, 'group': group, **r})
    prem_rows = [{'run': run, 'group': 'premium', **r}
                 for r in C.masked_metrics(y_true, y_pred, masks['premium'], scale=K)]
    pd.DataFrame(seg_rows).to_csv(guard_path(P.results_dir / f"segmented_{run}.csv"), index=False)
    pd.DataFrame(prem_rows).to_csv(guard_path(P.results_dir / f"premium_buckets_{run}.csv"), index=False)
    del model
    return out


def run_one(arch, max_epochs_override=None):
    run = run_name(arch)
    t_start = time.time()
    hp = load_retune_hp(arch)
    max_epochs = max_epochs_override if max_epochs_override is not None else hp['max_epochs']
    kw = model_kwargs_for(arch, hp)
    spec = TARGETS[TARGET]
    log(run, f"=== {run} ({CLASS_NAME[arch]}) -- cel: {spec['formula']}, patience={hp['patience']}, "
             f"max_epochs={max_epochs}, lr={hp['lr']!r}, wd={hp['weight_decay']!r}, "
             f"dropout={hp['dropout']!r}, batch={hp['batch_size']}, seed={hp['seed']}, "
             f"forras={hp['source_json']}, model_kwargs={kw} ===")
    pcc = param_count_check(arch, kw)
    write_json(P.results_dir / f"param_count_check_{run}.json", pcc)

    train_ds, val_ds = TU.get_ds('train', TARGET), TU.get_ds('val', TARGET)
    ckpt = guard_path(P.ckpt(run))
    hyper = {'lr': hp['lr'], 'weight_decay': hp['weight_decay'], 'dropout': hp['dropout'],
             'batch_size': hp['batch_size']}
    extra = {'target': TARGET, 'target_definition': spec['formula'],
             'prediction': 'american_put_price/K = european_put_price/K (BS) + NN(x)',
             'base_target': BASE_TARGET, 'residual_scaling': None,
             'data': rel(C.AM_DATA_DIR), 'patience': hp['patience'], 'max_epochs': max_epochs,
             'seed': hp['seed'], 'hyperparameters': hyper,
             'hyperparameter_source_json': hp['source_json'],
             'experiment': 'BS-residual learning (R1) on american put'}

    torch.manual_seed(hp['seed'])
    model = get_model(arch, **kw)
    t0 = time.time()
    hist = C.train_loop(model, train_ds, val_ds, output_dir=P.models_dir, model_name=run,
                        model_class=arch, model_kwargs=kw, feature_cols=FEATURE_COLS,
                        target_col=spec['target_col'], batch_size=hp['batch_size'],
                        max_epochs=max_epochs, lr=hp['lr'], weight_decay=hp['weight_decay'],
                        patience=hp['patience'], device=DEVICE, extra_meta=extra,
                        checkpoint_path=ckpt)
    train_seconds = time.time() - t0
    n_run = len(hist['val_loss'])
    pd.DataFrame({'epoch': range(1, n_run + 1), 'train_loss': hist['train_loss'],
                  'val_loss': hist['val_loss'], 'lr': hist['lr'],
                  'epoch_seconds': hist['epoch_seconds']}).to_csv(
        guard_path(P.results_dir / f"training_history_{run}.csv"), index=False)
    log(run, f"Tanitas kesz ({train_seconds:.1f}s): best_epoch={hist['best_epoch']}, "
             f"futott={n_run}, best_val_loss(reziduum)={hist['best_val_loss']:.8e}")
    del model
    ev = evaluate_residual(run, ckpt)
    out = {
        'run': run, 'arch': arch, 'model_class': CLASS_NAME[arch], 'target': TARGET,
        'target_col': spec['target_col'], 'target_definition': spec['formula'],
        'prediction': extra['prediction'], 'residual_scaling': None,
        'level': 'residual_retune_p30', 'direct_pair': direct_run_name(arch),
        'patience': hp['patience'], 'max_epochs': max_epochs, 'seed': hp['seed'],
        'hyperparameters': hyper, 'hyperparameter_source_json': hp['source_json'],
        'hyperparameter_source_study': hp['source_study'], 'hyperparameter_source_trial': hp['source_trial'],
        'hyperparameters_bitwise_equal_retune': bool(hp['bitwise_equal_retune_metrics_json']
                                                     and hp['bitwise_equal_retune_checkpoint']),
        'model_kwargs': kw, 'param_count': pcc['param_count'],
        'param_count_match': pcc['param_count_match'], 'param_count_check': pcc,
        'checkpoint': rel(ckpt),
        'data_files': {s: rel(C.am_split_path(s)) for s in C.SPLITS},
        'best_epoch': hist['best_epoch'], 'n_epochs_run': n_run,
        'hit_max_epochs': n_run >= max_epochs,
        'best_val_loss': hist['best_val_loss'],
        'best_val_loss_scale': 'reziduum (american − BS)/K',
        'train_seconds': train_seconds,
        'mean_epoch_seconds': float(np.mean(hist['epoch_seconds'])),
        **ev,
    }
    write_json(P.results_dir / f"metrics_{run}.json", out)
    tm = ev['test_metrics']
    log(run, f"=== {run} KESZ ({time.time() - t_start:.1f}s): val(rez.)={hist['best_val_loss']:.4e}, "
             f"val(ar)={ev['val_mse_price']:.4e}, test RMSE(ar)={tm['RMSE']:.4e}, R2={tm['R2']:.6f}, "
             f"NN<0 a teszten: {ev['test_sign']['share_nn_lt0']:.2%}, val ujraszamolva rel. elteres="
             f"{ev['val_recheck']['rel_diff']:.2e} ===")
    return out


def main():
    global P
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', type=str, default=None)
    ap.add_argument('--archs', type=str, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--skip-done', action='store_true')
    ap.add_argument('--smoke', type=int, default=None,
                    help="Fustproba: max_epochs felulirasa (csak az american-nn/-en kivuli --out-root-tal).")
    ap.add_argument('--out-root', type=str, default=None)
    ap.add_argument('--target-stats', action='store_true')
    ap.add_argument('--target-check', action='store_true')
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve() if args.out_root else AMNN_DIR
    if args.smoke is not None and C._is_within(out_root, AMNN_DIR):
        raise SystemExit("[smoke] A fustproba kimenete nem kerulhet az american-nn/ ala -- adj meg --out-root-ot.")
    C.set_allowed_roots(out_root)
    P = Paths(out_root)
    TU.P = P
    P.ensure()
    sys.stdout = TU._Tee(sys.stdout, P.logs_dir / "campaign_stdout.log")

    if args.target_stats or args.target_check:
        if args.target_stats:
            target_stats()
        if args.target_check:
            target_check()
        return

    pc = TU.pairing_check()
    if not pc['all_paired']:
        raise SystemExit("[parositas] A ket adathalmaz NEM paros -- a kampany leall.")
    edges = use_uniform_premium_edges()
    print(f"[premium] uniform vodorhatarok, ujraszamolva bitre azonos: {edges['recomputed_bitwise_equal']}")

    if args.arch:
        archs = [args.arch]
    elif args.archs:
        archs = [a.strip() for a in args.archs.split(',') if a.strip()]
    else:
        archs = ARCH_ORDER
    hps = {a: load_retune_hp(a) for a in archs}

    meta_path = P.results_dir / "campaign_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {'invocations': []}
    meta.update({'env': C.env_meta(), 'experiment': 'BS-residual learning (R1)',
                 'hyperparameters': hps, 'target': TARGETS[TARGET]['formula'],
                 'prediction': 'american_put_price/K = european_put_price/K + NN(x)',
                 'residual_scaling': None,
                 'scheduler': "ReduceLROnPlateau(mode='min', factor=0.5, patience=5, min_lr=1e-6)",
                 'optimizer': 'Adam', 'loss': 'MSELoss (a reziduumon)', 'feature_cols': FEATURE_COLS,
                 'patience': PATIENCE,
                 'machine': {'processor': __import__('platform').processor()}})
    meta['invocations'].append({'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'argv': sys.argv[1:],
                                'max_epochs_override': args.smoke})
    write_json(meta_path, meta)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] R1 rezidualis tanitas indul. Device: {DEVICE} "
          f"({C.env_meta().get('gpu_name')}); kimenet: {P.out_root}; smoke={args.smoke}")
    print(f"Futtatando: {len(archs)} tanitas: {[run_name(a) for a in archs]}")
    t0 = time.time()
    n_done = n_skip = n_err = 0
    for arch in archs:
        run = run_name(arch)
        if args.skip_done and (P.results_dir / f"metrics_{run}.json").exists() and P.ckpt(run).exists():
            print(f"[skip-done] {run} mar kesz, kihagyva.")
            n_skip += 1
            continue
        try:
            run_one(arch, args.smoke)
            n_done += 1
        except Exception:
            err = traceback.format_exc()
            with open(guard_path(P.logs_dir / f"{run}_ERROR.log"), 'a', encoding='utf-8') as f:
                f.write(err + "\n")
            print(f"[HIBA] {run}:\n{err}")
            n_err += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    meta = read_json(meta_path)
    meta['invocations'][-1].update({'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'wall_seconds': time.time() - t0, 'n_trained': n_done,
                                    'n_skipped': n_skip, 'n_errors': n_err})
    write_json(meta_path, meta)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Kampany kesz: {time.time() - t0:.1f}s, "
          f"tanitva={n_done}, kihagyva={n_skip}, hiba={n_err}")


if __name__ == '__main__':
    main()
