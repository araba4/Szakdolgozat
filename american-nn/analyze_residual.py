import sys
sys.dont_write_bytecode = True

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, spearmanr

AMNN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AMNN_DIR))

import common as C
import eval_common as E
import check_arbitrage as CA
import analyze_error_map as AEM
import benchmark_speed as BSP
from common import ARCH_ORDER, guard_path, write_json, read_json, rel
from src.evaluate import compute_metrics

RETUNE_DIR = AMNN_DIR / "results" / "retune"
ARB_DIR = AMNN_DIR / "results" / "evaluation" / "arbitrage"
EMAP_DIR = AMNN_DIR / "results" / "evaluation" / "error_map"
SPEED_DIR = AMNN_DIR / "results" / "evaluation" / "speed"
DIRECT_LEVEL = 'retune_p30'
R1_LEVEL = 'residual_retune_p30'
ARCH_LABEL = {'mlp': 'MLP', 'deep_mlp': 'DeepMLP', 'dense_mlp': 'DenseMLP', 'resnet': 'ResNet',
              'gelu_resnet': 'GELU-ResNet', 'highway': 'Highway', 'finn': 'FINN'}
SPEED_BATCHES = [1, 1_000, 100_000, 1_000_000]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Paths:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.models = self.root / "models" / "residual"
        self.results = self.root / "results" / "residual"

    def r1_ckpt(self, arch):
        return self.models / f"{arch}_residual_retune_p30_best.pt"

    def r1_metrics(self, arch):
        return self.results / f"metrics_{arch}_residual_retune_p30.json"


P = Paths(AMNN_DIR)


def r1_run(arch):
    return f"{arch}_residual_retune_p30"


def direct_run(arch):
    return f"{arch}_american_retune_p30"


def direct_ckpt(arch):
    return AMNN_DIR / "models" / "retune" / f"{direct_run(arch)}_best.pt"


def bs_derivs_norm(m, T, r, s):
    m, T, r, s = (np.asarray(a, dtype=np.float64) for a in (m, T, r, s))
    sq = np.sqrt(T)
    d1 = (np.log(m) + (r + 0.5 * s ** 2) * T) / (s * sq)
    d2 = d1 - s * sq
    return norm.cdf(d1) - 1.0, m * norm.pdf(d1) * s / (2.0 * sq) - r * np.exp(-r * T) * norm.cdf(-d2)


def bs_deriv_check(tp, n=2000, h=1e-5):
    rng = np.random.default_rng(42)
    ok = np.flatnonzero((tp['T'] > 2 * h) & (tp['m'] > 0.5 + 2 * h))
    i = np.sort(rng.choice(ok, size=n, replace=False))
    m, T, r, s = tp['m'][i], tp['T'][i], tp['r'][i], tp['s'][i]
    dm, dT = bs_derivs_norm(m, T, r, s)
    fm = (E.bs_put_norm(m + h, T, r, s) - E.bs_put_norm(m - h, T, r, s)) / (2 * h)
    fT = (E.bs_put_norm(m, T + h, r, s) - E.bs_put_norm(m, T - h, r, s)) / (2 * h)
    return {'n': n, 'h': h, 'delta_max_abs_diff': float(np.abs(dm - fm).max()),
            'theta_max_abs_diff': float(np.abs(dT - fT).max())}


METRIC_COLS = [('best_val_mse_price', 'val MSE (ár/K)'), ('test_rmse', 'teszt RMSE'), ('test_mae', 'teszt MAE'),
               ('test_r2', 'teszt R²'), ('test_max_error', 'teszt max. hiba'),
               ('test_usd_rmse', 'teszt RMSE (USD)'), ('test_usd_mae', 'teszt MAE (USD)'),
               ('test_usd_max_error', 'teszt max. hiba (USD)')]


def step_metrics(archs):
    lt = pd.read_csv(RETUNE_DIR / "levels_table.csv", float_precision='round_trip')
    rows, checks = [], []
    for a in archs:
        dj = read_json(RETUNE_DIR / f"metrics_{direct_run(a)}.json")
        rj = read_json(P.r1_metrics(a))
        ltr = lt[(lt.arch == a) & (lt.target == 'american') & (lt.level == DIRECT_LEVEL)].iloc[0]
        d = {'best_val_mse_price': dj['val_recheck']['reloaded_val_mse'],
             'best_val_loss_checkpoint': dj['best_val_loss'],
             'test_rmse': dj['test_metrics']['RMSE'], 'test_mae': dj['test_metrics']['MAE'],
             'test_r2': dj['test_metrics']['R2'], 'test_max_error': dj['test_metrics']['max_error'],
             'test_usd_rmse': dj['test_metrics_usd']['RMSE'], 'test_usd_mae': dj['test_metrics_usd']['MAE'],
             'test_usd_max_error': dj['test_metrics_usd']['max_error'],
             'best_epoch': dj['best_epoch'], 'n_epochs_run': dj['n_epochs_run'], 'train_seconds': dj['train_seconds']}
        r = {'best_val_mse_price': rj['val_mse_price'],
             'best_val_loss_checkpoint': rj['best_val_loss'],
             'test_rmse': rj['test_metrics']['RMSE'], 'test_mae': rj['test_metrics']['MAE'],
             'test_r2': rj['test_metrics']['R2'], 'test_max_error': rj['test_metrics']['max_error'],
             'test_usd_rmse': rj['test_metrics_usd']['RMSE'], 'test_usd_mae': rj['test_metrics_usd']['MAE'],
             'test_usd_max_error': rj['test_metrics_usd']['max_error'],
             'best_epoch': rj['best_epoch'], 'n_epochs_run': rj['n_epochs_run'], 'train_seconds': rj['train_seconds']}
        row = {'arch': a, 'direct_run': direct_run(a), 'residual_run': r1_run(a),
               'param_count': rj['param_count'], 'param_count_match': rj['param_count_match'],
               'hyperparameters_bitwise_equal_retune': rj['hyperparameters_bitwise_equal_retune']}
        for k, v in d.items():
            row[f"direct_{k}"] = v
        for k, v in r.items():
            row[f"residual_{k}"] = v
        for k, _ in METRIC_COLS:
            if k == 'test_r2':
                row['r2_gap_ratio_residual_over_direct'] = (1 - r[k]) / (1 - d[k])
                continue
            row[f"improvement_pct_{k}"] = 100.0 * (1.0 - r[k] / d[k])
            row[f"ratio_{k}"] = r[k] / d[k]
        row['residual_val_mse_residual_scale'] = rj['best_val_loss']
        row['residual_test_rmse_residual_scale'] = rj['test_metrics_residual_scale']['RMSE']
        row['residual_share_nn_lt0_test'] = rj['test_sign']['share_nn_lt0']
        row['residual_hit_max_epochs'] = rj['hit_max_epochs']
        row['direct_source'] = rel(RETUNE_DIR / f"metrics_{direct_run(a)}.json")
        row['residual_source'] = rel(P.r1_metrics(a))
        rows.append(row)
        lt_pairs = {'test_rmse': 'test_rmse', 'best_val_loss_checkpoint': 'best_val_mse', 'test_mae': 'test_mae',
                    'test_r2': 'test_r2', 'test_max_error': 'test_max_error', 'test_usd_rmse': 'test_usd_rmse'}
        checks.append({'arch': a,
                       'direct_max_rel_diff_vs_levels_table': max(abs(d[k] - float(ltr[c])) / abs(d[k])
                                                                  for k, c in lt_pairs.items()),
                       'residual_param_count_match': rj['param_count_match'],
                       'residual_param_count_eq_direct': rj['param_count'] == dj['param_count'],
                       'residual_hparams_bitwise_equal': rj['hyperparameters_bitwise_equal_retune']
                                                         and rj['hyperparameters'] == dj['hyperparameters'],
                       'residual_model_kwargs_eq_direct': rj['model_kwargs'] == dj['model_kwargs'],
                       'residual_val_recheck_rel_diff': rj['val_recheck']['rel_diff']})
    tab = pd.DataFrame(rows)
    tab.to_csv(guard_path(P.results / "direct_vs_residual.csv"), index=False)
    back = pd.read_csv(P.results / "direct_vs_residual.csv", float_precision="round_trip")
    json_keys = {'direct_test_rmse': ('test_metrics', 'RMSE'), 'direct_test_mae': ('test_metrics', 'MAE'),
                 'direct_test_r2': ('test_metrics', 'R2'), 'direct_test_max_error': ('test_metrics', 'max_error'),
                 'direct_test_usd_rmse': ('test_metrics_usd', 'RMSE'), 'direct_test_usd_mae': ('test_metrics_usd', 'MAE'),
                 'direct_test_usd_max_error': ('test_metrics_usd', 'max_error'),
                 'direct_best_val_mse_price': ('val_recheck', 'reloaded_val_mse')}
    for c, b in zip(checks, back.itertuples()):
        dj = read_json(RETUNE_DIR / f"metrics_{direct_run(c['arch'])}.json")
        c['direct_bitwise_equal_011_metrics_json_csv_roundtrip'] = all(
            getattr(b, col) == dj[k1][k2] for col, (k1, k2) in json_keys.items()) and             b.direct_best_val_loss_checkpoint == dj['best_val_loss']

    seg, prem = [], []
    for a in archs:
        for kind, store in (('segmented', seg), ('premium_buckets', prem)):
            d = pd.read_csv(RETUNE_DIR / f"{kind}_{direct_run(a)}.csv")
            r = pd.read_csv(P.results / f"{kind}_{r1_run(a)}.csv")
            assert list(d.segment) == list(r.segment) and list(d.n) == list(r.n)
            x = d[['group', 'segment', 'n']].copy()
            x.insert(0, 'arch', a)
            for col in ('RMSE', 'MAE', 'max_error', 'usd_RMSE', 'usd_MAE'):
                x[f"direct_{col}"] = d[col].to_numpy()
                x[f"residual_{col}"] = r[col].to_numpy()
                x[f"ratio_{col}"] = r[col].to_numpy() / d[col].to_numpy()
            x['improvement_pct_RMSE'] = 100.0 * (1.0 - x['ratio_RMSE'])
            store.append(x)
    seg = pd.concat(seg, ignore_index=True)
    prem = pd.concat(prem, ignore_index=True)
    seg.to_csv(guard_path(P.results / "segment_comparison.csv"), index=False)
    prem.to_csv(guard_path(P.results / "premium_bucket_comparison.csv"), index=False)
    med = []
    for name, df_ in (('segment', seg), ('premium', prem)):
        for (g, s), sub in df_.groupby(['group', 'segment'], sort=False):
            med.append({'table': name, 'group': g, 'segment': s, 'n': int(sub.n.iloc[0]),
                        'median_ratio_RMSE': float(sub.ratio_RMSE.median()),
                        'n_arch_improved': int((sub.ratio_RMSE < 1).sum()), 'n_arch': len(sub),
                        'best_arch_ratio': float(sub.ratio_RMSE.min()), 'worst_arch_ratio': float(sub.ratio_RMSE.max())})
    pd.DataFrame(med).to_csv(guard_path(P.results / "segment_premium_median_ratio.csv"), index=False)
    write_json(P.results / "verification_metrics.json", {
        'rows': checks,
        'all_direct_bitwise_equal_011_metrics_json': all(c['direct_bitwise_equal_011_metrics_json_csv_roundtrip']
                                                         for c in checks),
        'max_direct_rel_diff_vs_levels_table': max(c['direct_max_rel_diff_vs_levels_table'] for c in checks),
        'all_direct_bitwise_equal_levels_table': max(c['direct_max_rel_diff_vs_levels_table'] for c in checks) == 0.0,
        'csv_parsing': "pd.read_csv(float_precision='round_trip') -- az alapértelmezett gyors parser nem bitre pontos",
        'all_residual_param_count_match': all(c['residual_param_count_match'] and c['residual_param_count_eq_direct']
                                              for c in checks),
        'all_residual_hparams_bitwise_equal': all(c['residual_hparams_bitwise_equal'] for c in checks),
        'all_residual_model_kwargs_eq_direct': all(c['residual_model_kwargs_eq_direct'] for c in checks),
        'max_residual_val_recheck_rel_diff': max(c['residual_val_recheck_rel_diff'] for c in checks),
        'levels_table': rel(RETUNE_DIR / "levels_table.csv")})
    log(f"metrics: {len(tab)} arch, direkt = retune JSON bitre: "
        f"{all(c['direct_bitwise_equal_011_metrics_json_csv_roundtrip'] for c in checks)}, levels_table max rel. "
        f"elteres {max(c['direct_max_rel_diff_vs_levels_table'] for c in checks):.1e}")
    return tab


def r1_eval_on_test(model, tp, f_bs_parq):
    m, T, r, s = tp['m'], tp['T'], tp['r'], tp['s']
    nn0, dm_nn, dT_nn = E.nn_price_and_grads(model, m, T, r, s)
    nn0b = E.nn_price_norm(model, m, T, r, s, batch_size=4096).astype(np.float64)
    f = lambda mm, TT: E.bs_put_norm(mm, TT, r, s) + E.nn_price_norm(model, mm, TT, r, s, batch_size=4096).astype(np.float64)
    f0_level = f_bs_parq + nn0b
    f0 = E.bs_put_norm(m, T, r, s) + nn0b
    fmp, fmm, fTp = f(m + tp['h_m'], T), f(m - tp['h_m'], T), f(m, T + tp['h_T'])
    dm_bs, dT_bs = bs_derivs_norm(m, T, r, s)
    return f0_level, f0, fmp, fmm, fTp, dm_bs + dm_nn, dT_bs + dT_nn, nn0b, float(np.abs(nn0 - nn0b).max())


def sign_decomposition(f, am, bs, intrinsic, tols):
    out = []
    for ck, ref in (('c2_am_ge_bs', bs), ('c1_intrinsic', intrinsic)):
        vm, vl = ref - f, ref - am
        for tn, tol in tols:
            mv, lv = vm > tol, vl > tol
            both, mo, lo = mv & lv, mv & ~lv, ~mv & lv
            n = len(f)
            out.append({'check': ck, 'tol_name': tn, 'tol': tol, 'n_points': n,
                        'n_model_viol': int(mv.sum()), 'n_label_viol': int(lv.sum()),
                        'n_both': int(both.sum()), 'n_model_only': int(mo.sum()), 'n_label_only': int(lo.sum()),
                        'rate_model_viol': mv.mean(), 'rate_model_only': mo.mean(), 'rate_both': both.mean(),
                        'share_model_viol_where_label_also': both.sum() / mv.sum() if mv.any() else np.nan,
                        'max_model_only_norm': float(vm[mo].max()) if mo.any() else np.nan,
                        'mean_model_only_norm': float(vm[mo].mean()) if mo.any() else np.nan,
                        'mean_abs_err_on_both_norm': float(np.abs(f - am)[both].mean()) if both.any() else np.nan})
    return out


def step_consistency(archs):
    df = C.load_am_split('test')
    tp = CA.test_points(df)
    gp = CA.grid_points()
    K = tp['K']
    f_bs_parq = df['european_put_price'].to_numpy() / K
    am = C.target_values(df, 'american')
    intrinsic = np.maximum(1.0 - tp['m'], 0.0)
    reg_idx = CA.region_index(tp['m'], tp['T'])
    bsd = bs_deriv_check(tp)
    log(f"BS-derivalt ellenorzes (zart alak vs. kozponti differencia): {bsd}")
    rows, sign_rows, reg_rows, pred_check = [], [], [], []
    for a in archs:
        t0 = time.time()
        rj = read_json(P.r1_metrics(a))
        rmse = rj['test_metrics']['RMSE']
        tols = [('0', 0.0), ('1e-6', 1e-6), ('RMSE', rmse)]
        meta = dict(run=r1_run(a), arch=a, target='american', level=R1_LEVEL)
        model, _ = E.load(P.r1_ckpt(a))
        f0l, f0, fmp, fmm, fTp, dm, dT, nn0, gdiff = r1_eval_on_test(model, tp, f_bs_parq)
        y32 = am.astype(np.float32)
        rm = compute_metrics(y32, f0l.astype(np.float32))['RMSE']
        pred_check.append({'run': r1_run(a), 'test_rmse_recomputed': rm, 'test_rmse_metrics_json': rmse,
                           'rel_diff': abs(rm - rmse) / rmse, 'grad_path_vs_predict_max_abs': gdiff})
        ch = {**CA.level_checks(f0l, tp, f_bs=f_bs_parq)}
        ch.update({k: v for k, v in CA.shape_checks_test(f0, fmp, fmm, fTp, tp).items()})
        ch = {k: v for k, v in ch.items() if k in CA.CHECKS_AM}
        ch.update(CA.autograd_checks(dm, dT, tp))
        rows += CA.aggregate(ch, tp, tols, meta)
        nn_g, dm_g, dT_g = E.nn_price_and_grads(model, gp['m'], gp['T'], gp['r'], gp['s'])
        fbs_g = E.bs_put_norm(gp['m'], gp['T'], gp['r'], gp['s'])
        fg = fbs_g + nn_g.astype(np.float64)
        dmb, dTb = bs_derivs_norm(gp['m'], gp['T'], gp['r'], gp['s'])
        chg = {**CA.level_checks(fg, gp, f_bs=fbs_g), **CA.shape_checks_grid(fg, gp)}
        chg = {k: v for k, v in chg.items() if k in CA.CHECKS_AM}
        chg.update(CA.autograd_checks(dm_g + dmb, dT_g + dTb, gp))
        rows += CA.aggregate(chg, gp, tols, meta)
        for rr in sign_decomposition(f0l, am, f_bs_parq, intrinsic, tols):
            if rr['check'] == 'c2_am_ge_bs':
                rr['n_nn_lt_minus_tol'] = int((nn0 < -rr['tol']).sum())
            sign_rows.append({'model': 'R1', 'arch': a, 'run': r1_run(a), **rr})
        del model
        dmodel, _ = E.load(direct_ckpt(a))
        fd = E.nn_price_norm(dmodel, tp['m'], tp['T'], tp['r'], tp['s'], batch_size=4096).astype(np.float64)
        del dmodel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        drmse = read_json(RETUNE_DIR / f"metrics_{direct_run(a)}.json")['test_metrics']['RMSE']
        for rr in sign_decomposition(fd, am, f_bs_parq, intrinsic, [('0', 0.0), ('1e-6', 1e-6), ('RMSE', drmse)]):
            sign_rows.append({'model': 'direkt', 'arch': a, 'run': direct_run(a), **rr})
        e_d = fd.astype(np.float32).astype(np.float64) - y32.astype(np.float64)
        e_r = f0l.astype(np.float32).astype(np.float64) - y32.astype(np.float64)
        for g, name in enumerate(CA.REGION_NAMES):
            msk = reg_idx == g
            prem = (df['early_exercise_premium'].to_numpy() / K)[msk]
            reg_rows.append({'arch': a, 'region': name, 'n': int(msk.sum()),
                             'premium_mean_norm': float(prem.mean()),
                             'direct_rmse': float(np.sqrt(np.mean(e_d[msk] ** 2))),
                             'residual_rmse': float(np.sqrt(np.mean(e_r[msk] ** 2))),
                             'direct_bias': float(e_d[msk].mean()), 'residual_bias': float(e_r[msk].mean())})
        log(f"[konzisztencia] {a}: {time.time() - t0:.1f} s")

    res = pd.DataFrame(rows)
    res['check_label'] = res['check'].map(CA.CHECK_LABEL)
    res['main_check'] = res['check'].isin(CA.MAIN_CHECKS['american'])
    res.to_csv(guard_path(P.results / "violations_by_region_residual.csv"), index=False)
    pd.DataFrame(pred_check).to_csv(guard_path(P.results / "prediction_check_residual.csv"), index=False)
    reg = pd.DataFrame(reg_rows)
    reg['ratio_rmse'] = reg.residual_rmse / reg.direct_rmse
    reg.to_csv(guard_path(P.results / "region_comparison.csv"), index=False)
    sg = pd.DataFrame(sign_rows)
    sg.to_csv(guard_path(P.results / "sign_decomposition.csv"), index=False)

    v012 = pd.read_csv(ARB_DIR / "violations_by_region.csv")
    vd = v012[(v012.level == DIRECT_LEVEL) & (v012.target == 'american') & v012.arch.isin(archs)].copy()
    both = pd.concat([vd, res], ignore_index=True)
    keep = ['arch', 'level', 'set', 'check', 'check_label', 'tol_name', 'region', 'n_points', 'n_viol', 'rate',
            'max_norm', 'mean_norm', 'max_usd', 'mean_usd']
    tot = both[both.region == 'összes'][keep]
    wide = tot.pivot_table(index=['arch', 'set', 'check', 'check_label', 'tol_name'], columns='level',
                           values=['rate', 'n_viol', 'max_usd', 'mean_usd'], aggfunc='first')
    wide.columns = [f"{v}_{'direkt' if l == DIRECT_LEVEL else 'R1'}" for v, l in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(guard_path(P.results / "consistency_direct_vs_residual.csv"), index=False)
    med = wide.groupby(['set', 'check', 'check_label', 'tol_name'])[['rate_direkt', 'rate_R1', 'max_usd_direkt',
                                                                     'max_usd_R1']].median().reset_index()
    med['n_arch_R1_lower_rate'] = wide.groupby(['set', 'check', 'check_label', 'tol_name']).apply(
        lambda g: int((g.rate_R1 < g.rate_direkt).sum()), include_groups=False).to_numpy()
    med.to_csv(guard_path(P.results / "consistency_median.csv"), index=False)
    ver = []
    for a in archs:
        for ck in ('c2_am_ge_bs', 'c1_intrinsic'):
            for tn in ('0', '1e-6'):
                n012 = int(vd[(vd.arch == a) & (vd.set == 'test') & (vd.check == ck) & (vd.tol_name == tn)
                              & (vd.region == 'összes')].n_viol.iloc[0])
                nown = int(sg[(sg.model == 'direkt') & (sg.arch == a) & (sg.check == ck)
                              & (sg.tol_name == tn)].n_model_viol.iloc[0])
                ver.append({'arch': a, 'check': ck, 'tol': tn, 'n_viol_012': n012, 'n_viol_recomputed': nown,
                            'equal': n012 == nown})
    r1_nn = sg[(sg.model == 'R1') & (sg.check == 'c2_am_ge_bs')]
    write_json(P.results / "verification_consistency.json", {
        'bs_derivative_check': bsd,
        'direct_counts_vs_012': ver, 'all_direct_counts_equal_012': all(v['equal'] for v in ver),
        'r1_c2_violation_equals_nn_negative': bool((r1_nn.n_model_viol == r1_nn.n_nn_lt_minus_tol).all()),
        'r1_c2_detail': r1_nn[['arch', 'tol_name', 'n_model_viol', 'n_nn_lt_minus_tol']].to_dict(orient='records'),
        'max_pred_rel_diff_vs_metrics_json': float(pd.DataFrame(pred_check).rel_diff.max()),
        'direct_source': rel(ARB_DIR / "violations_by_region.csv"),
        'level_checks_bs': 'parquet european_put_price/K (mint a kiertekelesben); 4-6. pont és rács: zárt alakú BS',
    })
    log(f"[konzisztencia] direkt = kiertekeles: {all(v['equal'] for v in ver)}")


def step_plateau(archs):
    ts = read_json(P.results / "target_stats.json")
    var_val = ts['splits']['val']['residual_norm']['std'] ** 2
    var_test = ts['splits']['test']['residual_norm']['std'] ** 2
    rows = []
    for a in archs:
        hr = pd.read_csv(P.results / f"training_history_{r1_run(a)}.csv")
        hd = pd.read_csv(RETUNE_DIR / f"training_history_{direct_run(a)}.csv")
        rj = read_json(P.r1_metrics(a))
        br, bd = int(hr.val_loss.idxmin()), int(hd.val_loss.idxmin())
        reach = np.flatnonzero(hd.val_loss.to_numpy() <= hr.val_loss.min())
        post = hr.val_loss.iloc[br + 1:]
        rm = rj['test_metrics_residual_scale']['RMSE']
        rows.append({
            'arch': a, 'lr': rj['hyperparameters']['lr'], 'weight_decay': rj['hyperparameters']['weight_decay'],
            'dropout': rj['hyperparameters']['dropout'],
            'r1_best_epoch': br + 1, 'r1_n_epochs': len(hr), 'r1_lr_halvings_until_best': int((hr.lr.iloc[:br + 1].diff() < 0).sum()),
            'r1_lr_halvings_total': int((hr.lr.diff() < 0).sum()), 'r1_lr_final': float(hr.lr.iloc[-1]),
            'direct_best_epoch': bd + 1, 'direct_n_epochs': len(hd),
            'direct_lr_halvings_until_best': int((hd.lr.iloc[:bd + 1].diff() < 0).sum()),
            'r1_best_val_mse': float(hr.val_loss.min()), 'direct_best_val_mse': float(hd.val_loss.min()),
            'r1_val_mse_over_residual_var': float(hr.val_loss.min() / var_val),
            'r1_test_r2_residual_scale': float(1.0 - rm ** 2 / var_test),
            'r1_first_epoch_val': float(hr.val_loss.iloc[0]), 'direct_first_epoch_val': float(hd.val_loss.iloc[0]),
            'direct_val_at_r1_best_epoch': float(hd.val_loss.iloc[br]) if br < len(hd) else np.nan,
            'direct_ever_reached_r1_best': bool(len(reach)),
            'r1_post_best_val_median_over_min': float(post.median() / hr.val_loss.min()) if len(post) else np.nan,
            'r1_train_loss_at_best': float(hr.train_loss.iloc[br]), 'r1_train_loss_end': float(hr.train_loss.iloc[-1]),
            'r1_train_seconds': rj['train_seconds'],
            'collapse_criterion_rmse_ge_half_bs_only': bool(rj['test_metrics']['RMSE']
                                                            >= 0.5 * np.sqrt(ts['splits']['test']['residual_norm']['std'] ** 2
                                                                             + ts['splits']['test']['residual_norm']['mean'] ** 2)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(guard_path(P.results / "plateau_check.csv"), index=False)
    write_json(P.results / "plateau_check_meta.json", {
        'residual_var_val': var_val, 'residual_var_test': var_test, 'source': rel(P.results / "target_stats.json"),
        'note': 'r1_val_mse_over_residual_var: 1 = konstans (atlag) tanulasa; '
                'collapse_criterion: a szetesesi kriterium (teszt-RMSE >= a csak-BS RMSE fele)'})
    log(f"plato: {rel(P.results / 'plateau_check.csv')}")


def step_errormap(archs):
    tab = pd.read_csv(P.results / "direct_vs_residual.csv")
    tab = tab[tab.arch.isin(archs)]
    best_r1 = tab.loc[tab.residual_best_val_mse_price.idxmin(), 'arch']
    lt = pd.read_csv(RETUNE_DIR / "levels_table.csv")
    sub = lt[lt.target == 'american']
    best_d = sub.loc[sub.best_val_mse.idxmin()]
    best_direct_run = E.run_id(best_d.arch, 'american', best_d.level)
    df = C.load_am_split('test')
    m, T, K = (df[c].to_numpy(np.float64) for c in ('moneyness', 'T', 'K'))
    S = df['S'].to_numpy(np.float64)
    prem = df['early_exercise_premium'].to_numpy(np.float64) / K
    amp = df['american_put_price'].to_numpy(np.float64)
    ex = (K > S) & ((amp - (K - S)) < AEM.EX_EPS * K)
    mi, ti = AEM.bin_index(m, AEM.M_EDGES), AEM.bin_index(T, AEM.T_EDGES)
    y32 = C.target_values(df, 'american').astype(np.float32).astype(np.float64)
    frames, errs = {}, {}
    for kind, run, ck in (('direkt', best_direct_run, E.checkpoint_path(best_d.arch, 'american', best_d.level)),
                          ('R1', r1_run(best_r1), P.r1_ckpt(best_r1))):
        model, _ = E.load(ck)
        nn = E.nn_price_norm(model, m, T, df['r'], df['sigma'], batch_size=4096)
        del model
        pred = nn if kind == 'direkt' else (C.target_values(df, 'european') + nn.astype(np.float64)).astype(np.float32)
        err = pred.astype(np.float64) - y32
        errs[kind] = err
        st = AEM.cell_stats(err, err * K, prem, prem * K, ex, mi, ti, AEM.N_M, AEM.N_T)
        frames[kind] = AEM.to_frame(st, run, 'összes σ', AEM.M_EDGES, AEM.T_EDGES).drop(columns=['sse_norm'])
    c012 = pd.read_csv(EMAP_DIR / "error_map_cells.csv")
    c012 = c012[(c012.run == best_direct_run) & (c012.sigma_slice == 'összes σ')].reset_index(drop=True)
    dcells = frames['direkt'].reset_index(drop=True)
    diff = float(np.nanmax(np.abs(c012.rmse_norm.to_numpy() - dcells.rmse_norm.to_numpy())
                           / c012.rmse_norm.to_numpy()))
    cells = pd.concat([frames['direkt'].assign(model='direkt'), frames['R1'].assign(model='R1')], ignore_index=True)
    a, b = frames['direkt'].reset_index(drop=True), frames['R1'].reset_index(drop=True)
    cells.to_csv(guard_path(P.results / "error_map_cells_residual.csv"), index=False)
    ratio = np.log2(b.rmse_norm / a.rmse_norm)
    rho, p = spearmanr(ratio, a.premium_mean_norm)
    hyp = {'best_residual_run': r1_run(best_r1), 'best_direct_run': best_direct_run,
           'selection': 'legkisebb val MSE az áron (R1: val_mse_price; direkt: levels_table best_val_mse)',
           'direct_cells_vs_012_max_rel_diff': diff,
           'share_cells_r1_better': float((ratio < 0).mean()),
           'median_cell_log2_ratio': float(np.median(ratio)),
           'spearman_log2ratio_vs_premium': {'rho': float(rho), 'p': float(p)},
           'pooled_rmse': {}}
    regions = {'OTM (m≥1,1)': m >= 1.1, 'ATM': (m >= 0.9) & (m < 1.1), 'ITM (m<0,9)': m < 0.9,
               'rövid T (<0,25)': T < 0.25, 'ITM × hosszú T': (m < 0.9) & (T >= 1.0),
               'lehívási sorok': ex, 'prémium < 1e-6': prem < C.PREMIUM_ZERO_THRESHOLD, 'összes': np.ones_like(ex)}
    for nm, msk in regions.items():
        hyp['pooled_rmse'][nm] = {'direkt': AEM.pooled_rmse(errs['direkt'], msk),
                                  'R1': AEM.pooled_rmse(errs['R1'], msk), 'n': int(msk.sum())}
        hyp['pooled_rmse'][nm]['ratio'] = hyp['pooled_rmse'][nm]['R1'] / hyp['pooled_rmse'][nm]['direkt']
    write_json(P.results / "error_map_comparison.json", hyp)
    log(f"[hibaterkep] {hyp['best_residual_run']} vs {best_direct_run}; direkt cellak vs kiertekeles max rel. elteres {diff:.2e}")


def bs_put_norm_torch(m, T, r, s):
    sq = torch.sqrt(T)
    d1 = (torch.log(m) + (r + 0.5 * s * s) * T) / (s * sq)
    d2 = d1 - s * sq
    return torch.exp(-r * T) * torch.special.ndtr(-d2) - m * torch.special.ndtr(-d1)


def step_speed(archs):
    cuda = torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    state_before = BSP.machine_state('előtte')
    log(f"Gep allapota: CPU {state_before.get('cpu_load_pct_3x1s')}, szabad RAM {state_before.get('ram_free_mb')} MB")
    df = pd.read_parquet(C.am_split_path('test'), columns=['moneyness', 'T', 'r', 'sigma'])
    raw = BSP.make_inputs(df, max(SPEED_BATCHES), np.random.default_rng(BSP.SEED))
    n_threads = torch.get_num_threads()
    rows = []

    def add(**kw):
        rows.append(kw)

    for b in SPEED_BATCHES:
        m_, T_, r_, s_ = (x[:b] for x in raw)
        st = BSP.timeit(lambda: E.bs_put_norm(m_, T_, r_, s_), BSP.REPS_BS[b], warmup=3)
        add(method='BS put (zárt alak)', arch='-', device='CPU', batch=b, mode='vektorizált NumPy', **st)
    bs_torch_check = None
    if cuda:
        m_, T_, r_, s_ = (torch.tensor(x, dtype=torch.float64, device='cuda') for x in raw)
        bt = bs_put_norm_torch(m_, T_, r_, s_).cpu().numpy()
        bs_torch_check = {'n': int(len(bt)), 'max_abs_diff_vs_numpy': float(np.abs(bt - E.bs_put_norm(*raw)).max())}
        for b in SPEED_BATCHES:
            mm, TT, rr, ss = m_[:b], T_[:b], r_[:b], s_[:b]
            st = BSP.timeit(lambda: bs_put_norm_torch(mm, TT, rr, ss), BSP.REPS_GPU[b], sync=sync)
            add(method='BS put (zárt alak)', arch='-', device='GPU', batch=b,
                mode='torch float64, bemenet a GPU-n', **st)
        del m_, T_, r_, s_
    for a in archs:
        if cuda:
            model, _ = E.load(P.r1_ckpt(a), device='cuda')
            for b in SPEED_BATCHES:
                m_, T_, r_, s_ = (x[:b] for x in raw)

                def e2e_gpu_bs():
                    xx = torch.tensor(E.normalize(m_, T_, r_, s_), dtype=torch.float32).to('cuda')
                    rw = torch.tensor(np.stack([m_, T_, r_, s_]), dtype=torch.float64).to('cuda')
                    with torch.inference_mode():
                        out = (bs_put_norm_torch(rw[0], rw[1], rw[2], rw[3]) + model(xx).ravel().double()).cpu().numpy()
                    return out
                st = BSP.timeit(e2e_gpu_bs, max(3, BSP.REPS_GPU[b] // 2), warmup=5, sync=sync)
                add(method='NN R1 (BS a GPU-n)', arch=a, device='GPU', batch=b,
                    mode='teljes út (nyers float64 → normalizálás → GPU: NN + BS torch float64 → CPU)', **st)
            del model
            torch.cuda.empty_cache()
        for kind, ck in (('direkt', direct_ckpt(a)), ('R1', P.r1_ckpt(a))):
            log(f"sebesseg: {a} {kind}")
            if cuda:
                model, _ = E.load(ck, device='cuda')
                for b in SPEED_BATCHES:
                    m_, T_, r_, s_ = (x[:b] for x in raw)

                    def e2e():
                        xx = torch.tensor(E.normalize(m_, T_, r_, s_), dtype=torch.float32).to('cuda')
                        with torch.inference_mode():
                            out = model(xx).cpu().numpy().ravel()
                        if kind == 'R1':
                            out = E.bs_put_norm(m_, T_, r_, s_) + out
                        return out
                    st = BSP.timeit(e2e, max(3, BSP.REPS_GPU[b] // 2), warmup=5, sync=sync)
                    add(method=f'NN {kind}', arch=a, device='GPU', batch=b,
                        mode='teljes út (nyers float64 → normalizálás → GPU → CPU' + (' + BS NumPy' if kind == 'R1' else '') + ')',
                        **st)
                del model
                torch.cuda.empty_cache()
            model, _ = E.load(ck, device='cpu')
            for b in (1, 1_000):
                m_, T_, r_, s_ = (x[:b] for x in raw)

                def e2e_cpu():
                    xx = torch.tensor(E.normalize(m_, T_, r_, s_), dtype=torch.float32)
                    with torch.inference_mode():
                        out = model(xx).numpy().ravel()
                    if kind == 'R1':
                        out = E.bs_put_norm(m_, T_, r_, s_) + out
                    return out
                st = BSP.timeit(e2e_cpu, BSP.REPS_CPU[b], warmup=BSP.WARMUP)
                add(method=f'NN {kind}', arch=a, device='CPU', threads=n_threads, batch=b,
                    mode='teljes út (nyers float64 → normalizálás → CPU' + (' + BS NumPy' if kind == 'R1' else '') + ')',
                    **st)
            del model
    state_after = BSP.machine_state('utána')
    res = pd.DataFrame(rows)
    res['per_row_s'] = res['median_s'] / res['batch']
    res.to_csv(guard_path(P.results / "speed_timings.csv"), index=False)
    ov = []
    for a in archs:
        for dev in ('GPU', 'CPU'):
            for b in SPEED_BATCHES:
                d = res[(res.method == 'NN direkt') & (res.arch == a) & (res.device == dev) & (res.batch == b)]
                r = res[(res.method == 'NN R1') & (res.arch == a) & (res.device == dev) & (res.batch == b)]
                rg = res[(res.method == 'NN R1 (BS a GPU-n)') & (res.arch == a) & (res.device == dev) & (res.batch == b)]
                if len(d) and len(r):
                    bs = res[(res.method == 'BS put (zárt alak)') & (res.device == 'CPU') & (res.batch == b)].median_s.iloc[0]
                    row = {'arch': a, 'device': dev, 'batch': b, 'direct_median_s': d.median_s.iloc[0],
                           'r1_median_s': r.median_s.iloc[0], 'bs_alone_numpy_median_s': bs,
                           'overhead_s': r.median_s.iloc[0] - d.median_s.iloc[0],
                           'overhead_per_row_s': (r.median_s.iloc[0] - d.median_s.iloc[0]) / b,
                           'ratio_r1_over_direct': r.median_s.iloc[0] / d.median_s.iloc[0]}
                    if len(rg):
                        row.update({'r1_bs_gpu_median_s': rg.median_s.iloc[0],
                                    'overhead_bs_gpu_per_row_s': (rg.median_s.iloc[0] - d.median_s.iloc[0]) / b,
                                    'ratio_r1_bs_gpu_over_direct': rg.median_s.iloc[0] / d.median_s.iloc[0]})
                    ov.append(row)
    ov = pd.DataFrame(ov)
    ov.to_csv(guard_path(P.results / "speed_overhead.csv"), index=False)
    t012 = pd.read_csv(SPEED_DIR / "timings.csv")
    crr = t012[(t012.method == 'CRR-amerikai (500 lépés)') & ~t012['mode'].str.contains('ismétlés')].median_s.iloc[0]
    crr_eu = t012[t012.method == 'CRR-európai batch (500 lépés)'][['batch', 'per_row_s']].to_dict(orient='records')
    write_json(P.results / "speed_meta.json", {
        'script': 'american-nn/analyze_residual.py --steps speed', 'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'env': E.env_extra(), 'torch_num_threads': n_threads, 'bs_torch_vs_numpy': bs_torch_check,
        'machine_state_before': state_before, 'machine_state_after': state_after,
        'measurement_code': 'american-nn/benchmark_speed.py: timeit, make_inputs, REPS_* (importálva)',
        'reference_012': {'crr_american_s_per_row': float(crr), 'crr_european_batch_s_per_row': crr_eu,
                          'source': rel(SPEED_DIR / "timings.csv"),
                          'note': 'R2 (CRR-európai + NN) becsült többletköltsége = a CRR-európai batch ideje/ár'}})
    log("sebesseg kesz")


def step_summary(archs):
    import md_tables as M
    CN = C.CLASS_NAME
    tab = pd.read_csv(P.results / "direct_vs_residual.csv")
    tab = tab[tab.arch.isin(archs)]
    L = []

    L += ["## Direkt és reziduális (R1) modell az amerikai áron", ""]
    L += M.table(["Architektúra", "Val MSE, direkt", "Val MSE, R1", "Teszt-RMSE, direkt", "Teszt-RMSE, R1",
                  "Teszt-RMSE változása", "R1 és direkt aránya", "Teszt-RMSE ($), direkt", "Teszt-RMSE ($), R1",
                  "Max. hiba ($), direkt", "Max. hiba ($), R1", "R1 legjobb epoch", "R1 futott epoch"],
                 [[CN[r.arch], M.sci(r.direct_best_val_mse_price), M.sci(r.residual_best_val_mse_price),
                   M.sci(r.direct_test_rmse), M.sci(r.residual_test_rmse), M.chg(-r.improvement_pct_test_rmse),
                   M.f(r.ratio_test_rmse, 3), M.f(r.direct_test_usd_rmse, 4), M.f(r.residual_test_usd_rmse, 4),
                   M.f(r.direct_test_usd_max_error), M.f(r.residual_test_usd_max_error),
                   int(r.residual_best_epoch), int(r.residual_n_epochs_run)] for _, r in tab.iterrows()])
    best = tab.loc[tab.residual_best_val_mse_price.idxmin()]
    L += ["", "## Összesítés", ""]
    L += M.table(["Mutató", "Érték"], [
        ["Kisebb R1 teszt-RMSE", f"{int((tab.ratio_test_rmse < 1).sum())}/{len(tab)}"],
        ["A teszt-RMSE változásának mediánja", M.chg(-tab.improvement_pct_test_rmse.median())],
        ["A legjobb R1 modell (val MSE az áron)", CN[best.arch]],
        ["A legjobb R1 modell teszt-RMSE-je", M.sci(best.residual_test_rmse)]], "lr")

    med = pd.read_csv(P.results / "segment_premium_median_ratio.csv")
    L += ["", "## Az R1 és a direkt teszt-RMSE aránya szegmensenként (medián a 7 architektúrán)", ""]
    L += M.table(["Csoport", "Szegmens", "n", "Medián arány", "Javult architektúra", "Legjobb arány",
                  "Legrosszabb arány"],
                 [[r.group, r.segment, M.p(r.n), M.f(r.median_ratio_RMSE, 3), f"{r.n_arch_improved}/{r.n_arch}",
                   M.f(r.best_arch_ratio, 3), M.f(r.worst_arch_ratio, 3)] for _, r in med.iterrows()], "llrrrrr")

    cm = pd.read_csv(P.results / "consistency_median.csv")
    L += ["", "## Konzisztencia-ellenőrzések (medián a 7 architektúrán)", ""]
    L += M.table(["Halmaz", "Ellenőrzés", "τ", "Megsértés, direkt", "Megsértés, R1", "Max. ($), direkt",
                  "Max. ($), R1", "Kisebb R1 arány"],
                 [[r.set, r.check_label, r.tol_name, M.pct(r.rate_direkt), M.pct(r.rate_R1), M.f(r.max_usd_direkt, 3),
                   M.f(r.max_usd_R1, 3), f"{int(r.n_arch_R1_lower_rate)}/{len(archs)}"]
                  for _, r in cm[cm.check.isin(CA.MAIN_CHECKS['american'])].iterrows()], "lll" + "r" * 5)

    sg = pd.read_csv(P.results / "sign_decomposition.csv")
    L += ["", "## A P_am ≥ P_BS megsértésének felbontása (teszthalmaz)", ""]
    L += M.table(["Architektúra", "Modell", "τ", "Megsértés", "A cél is sérti", "Csak a modell sérti",
                  "Csak a modell, max. (ár/K)"],
                 [[CN[r.arch], r.model, r.tol_name, M.pct(r.rate_model_viol), M.p(r.n_both), M.p(r.n_model_only),
                   M.sci(r.max_model_only_norm)]
                  for _, r in sg[(sg.check == 'c2_am_ge_bs') & sg.tol_name.isin(['0', '1e-6', 'RMSE'])].iterrows()],
                 "lll" + "r" * 4)

    if (P.results / "plateau_check.csv").exists():
        pc = pd.read_csv(P.results / "plateau_check.csv")
        pc = pc[pc.arch.isin(archs)]
        L += ["", "## Korai leállás és plató", ""]
        L += M.table(["Architektúra", "lr", "dropout", "R1 legjobb epoch", "R1 futott epoch",
                      "LR-felezés a legjobb epochig (R1)", "Direkt legjobb epoch", "Direkt futott epoch",
                      "R1 val MSE / a reziduum varianciája", "R1 teszt R² a reziduumon",
                      "A direkt elérte az R1 legjobb val MSE-jét"],
                     [[CN[r.arch], M.sci(r.lr), M.f(r.dropout, 3) if np.isfinite(r.dropout) else '–', r.r1_best_epoch,
                       r.r1_n_epochs, r.r1_lr_halvings_until_best, r.direct_best_epoch, r.direct_n_epochs,
                       M.f(r.r1_val_mse_over_residual_var, 4), M.f(r.r1_test_r2_residual_scale, 4),
                       M.yes(bool(r.direct_ever_reached_r1_best))] for _, r in pc.iterrows()])

    if (P.results / "error_map_comparison.json").exists():
        h = read_json(P.results / "error_map_comparison.json")
        L += ["", "## Hibatérkép: a legjobb R1 és a legjobb direkt modell", ""]
        L += M.table(["Mutató", "Érték"], [
            ["R1 modell", h['best_residual_run']], ["Direkt modell", h['best_direct_run']],
            ["Cellák, ahol az R1 jobb (400 cellából)", M.pct(h['share_cells_r1_better'], 1)],
            ["A cellánkénti log₂-arány mediánja", M.f(h['median_cell_log2_ratio'], 3)],
            ["Spearman ρ (log₂-arány és prémium)", M.f(h['spearman_log2ratio_vs_premium']['rho'], 3)]], "lr")
        L += ["", "## Hibatérkép: teszt-RMSE régiónként", ""]
        L += M.table(["Régió", "n", "Teszt-RMSE, direkt", "Teszt-RMSE, R1", "R1 és direkt aránya"],
                     [[k, M.p(v['n']), M.sci(v['direkt']), M.sci(v['R1']), M.f(v['ratio'], 3)]
                      for k, v in h['pooled_rmse'].items()])

    if (P.results / "speed_overhead.csv").exists():
        ov = pd.read_csv(P.results / "speed_overhead.csv")
        L += ["", "## Sebesség: a teljes út mediánja", ""]
        L += M.table(["Architektúra", "Eszköz", "Batch", "Direkt (s)", "R1, BS NumPy-jal (s)", "R1 és direkt aránya",
                      "R1, BS a GPU-n (s)", "R1 GPU-BS-sel és direkt aránya"],
                     [[CN[r.arch], r.device, M.p(r.batch), M.sci(r.direct_median_s), M.sci(r.r1_median_s),
                       M.f(r.ratio_r1_over_direct), M.sci(r.get('r1_bs_gpu_median_s', np.nan)),
                       M.f(r.get('ratio_r1_bs_gpu_over_direct', np.nan))] for _, r in ov.iterrows()], "ll" + "r" * 6)
    with open(guard_path(P.results / "residual_summary.md"), 'w', encoding='utf-8') as fh:
        fh.write("\n".join(L) + "\n")
    log(f"osszefoglalo: {rel(P.results / 'residual_summary.md')}")


def main():
    global P
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', nargs='*', default=['metrics', 'consistency', 'plateau', 'errormap', 'speed', 'summary'])
    ap.add_argument('--root', type=str, default=None, help='az R1 modellek/eredmenyek gyokere (fustproba)')
    ap.add_argument('--archs', type=str, default=None)
    ap.add_argument('--quick', action='store_true', help='sebesseg: kevesebb ismetles (csak fustprobahoz)')
    args = ap.parse_args()
    if args.quick:
        for d in (BSP.REPS_GPU, BSP.REPS_CPU, BSP.REPS_BS):
            for k in d:
                d[k] = max(2, d[k] // 50)
        BSP.WARMUP = 2
    root = Path(args.root).resolve() if args.root else AMNN_DIR
    C.set_allowed_roots(root)
    P = Paths(root)
    guard_path(P.results).mkdir(parents=True, exist_ok=True)
    archs = [a.strip() for a in args.archs.split(',')] if args.archs else ARCH_ORDER
    t0 = time.time()
    log(f"Kornyezet: {E.env_extra()}; gyoker: {P.root}; lepesek: {args.steps}")
    for st in args.steps:
        {'metrics': step_metrics, 'consistency': step_consistency, 'plateau': step_plateau, 'errormap': step_errormap,
         'speed': step_speed, 'summary': step_summary}[st](archs)
    write_json(P.results / f"analysis_meta_{'_'.join(args.steps)}.json",
               {'script': 'american-nn/analyze_residual.py', 'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'seconds': time.time() - t0, 'steps': args.steps, 'archs': archs, 'env': E.env_extra()})
    log(f"Kesz: {time.time() - t0:.1f} s")


if __name__ == '__main__':
    main()
