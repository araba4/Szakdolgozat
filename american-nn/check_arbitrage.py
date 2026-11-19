import sys
sys.dont_write_bytecode = True

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common as C
import eval_common as E
from common import guard_path, write_json, rel
from src.evaluate import compute_metrics

OUT_DIR = E.EVAL_DIR / "arbitrage"

H_M = 0.01
H_T = 0.01
TOL_FIXED = [0.0, 1e-6]
K_GRID = 100.0
FD_H = 1e-4
N_FD = 2000

GRID_M = np.linspace(0.5, 1.5, 101)
GRID_T = np.linspace(0.005, 2.0, 41)
GRID_R = np.array([0.0, 0.025, 0.05])
GRID_S = np.array([0.05, 0.2625, 0.475, 0.6875, 0.9])
CRR_SLICE = dict(r=0.05, sigma=0.2625)

CHECK_LABEL = {
    'c1_intrinsic':      '1. P ≥ max(K−S,0)',
    'c1e_eu_lower':      '1e. P ≥ max(Ke^{−rT}−S,0)',
    'c2_am_ge_bs':       '2. P_am ≥ P_BS',
    'c2c_am_ge_crr_eu':  "2'. P_am ≥ P_CRR-eu (azonos fa)",
    'c3_le_K':           '3. P ≤ K',
    'c4a_mono_S':        '4a. monoton csökken S-ben',
    'c4b_delta_ge_m1':   '4b. ∂P/∂S ≥ −1',
    'c5_mono_T':         '5. monoton nő T-ben',
    'c6_convex_S':       '6. konvex S-ben',
    'd4a_delta_le0':     '4a (autograd). Δ ≤ 0',
    'd4b_delta_gem1':    '4b (autograd). Δ ≥ −1',
    'd5_theta_ge0':      '5 (autograd). ∂P/∂T ≥ 0',
}
CHECKS_AM = ['c1_intrinsic', 'c2_am_ge_bs', 'c3_le_K', 'c4a_mono_S', 'c4b_delta_ge_m1',
             'c5_mono_T', 'c6_convex_S', 'c1e_eu_lower']
CHECKS_EU = ['c1e_eu_lower', 'c3_le_K', 'c4a_mono_S', 'c4b_delta_ge_m1', 'c6_convex_S',
             'c5_mono_T', 'c1_intrinsic']
MAIN_CHECKS = {'american': ['c1_intrinsic', 'c2_am_ge_bs', 'c3_le_K', 'c4a_mono_S',
                            'c4b_delta_ge_m1', 'c5_mono_T', 'c6_convex_S'],
               'european': ['c1e_eu_lower', 'c3_le_K', 'c4a_mono_S', 'c4b_delta_ge_m1',
                            'c6_convex_S']}
AUTOGRAD_CHECKS = ['d4a_delta_le0', 'd4b_delta_gem1', 'd5_theta_ge0']

REGION_NAMES = [f"{mk} × {tk}" for mk in C.PUT_MONEYNESS_SEGMENTS for tk in C.MATURITY_SEGMENTS]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def region_index(m, T):
    mi = np.full(len(m), -1)
    for i, (lo, hi) in enumerate(C.PUT_MONEYNESS_SEGMENTS.values()):
        mi[(m >= lo) & (m < hi)] = i
    ti = np.full(len(T), -1)
    for j, (lo, hi) in enumerate(C.MATURITY_SEGMENTS.values()):
        ti[(T >= lo) & (T < hi)] = j
    assert (mi >= 0).all() and (ti >= 0).all()
    return mi * len(C.MATURITY_SEGMENTS) + ti


def test_points(df):
    m = df['moneyness'].to_numpy(np.float64)
    T = df['T'].to_numpy(np.float64)
    return dict(name='test', m=m, T=T, r=df['r'].to_numpy(np.float64),
                s=df['sigma'].to_numpy(np.float64), K=df['K'].to_numpy(np.float64),
                valid_S=(m - H_M >= 0.5) & (m + H_M <= 1.5),
                valid_T=(T + H_T <= 2.0), h_m=H_M, h_T=H_T)


def grid_points():
    rr, ss, TT, mm = np.meshgrid(GRID_R, GRID_S, GRID_T, GRID_M, indexing='ij')
    shape = mm.shape
    return dict(name='grid', shape=shape, m=mm.ravel(), T=TT.ravel(), r=rr.ravel(), s=ss.ravel(),
                K=np.full(mm.size, K_GRID), h_m=float(GRID_M[1] - GRID_M[0]),
                h_T=float(GRID_T[1] - GRID_T[0]))


def crr_slice_points():
    TT, mm = np.meshgrid(GRID_T, GRID_M, indexing='ij')
    shape = (1, 1) + mm.shape
    return dict(name='crr_slice', shape=shape, m=mm.ravel(), T=TT.ravel(),
                r=np.full(mm.size, CRR_SLICE['r']), s=np.full(mm.size, CRR_SLICE['sigma']),
                K=np.full(mm.size, K_GRID), h_m=float(GRID_M[1] - GRID_M[0]),
                h_T=float(GRID_T[1] - GRID_T[0]))


def level_checks(f, pts, f_bs=None, f_crr_eu=None):
    m, T, r = pts['m'], pts['T'], pts['r']
    n = len(m)
    allv = np.ones(n, bool)
    idx = np.arange(n)
    out = {
        'c1_intrinsic': (np.maximum(1.0 - m, 0.0) - f, allv, idx),
        'c1e_eu_lower': (np.maximum(np.exp(-r * T) - m, 0.0) - f, allv, idx),
        'c3_le_K': (f - 1.0, allv, idx),
    }
    if f_bs is not None:
        out['c2_am_ge_bs'] = (f_bs - f, allv, idx)
    if f_crr_eu is not None:
        out['c2c_am_ge_crr_eu'] = (f_crr_eu - f, allv, idx)
    return out


def shape_checks_test(f0, fmp, fmm, fTp, pts):
    idx = np.arange(len(f0))
    vS, vT = pts['valid_S'], pts['valid_T']
    return {
        'c4a_mono_S': (fmp - f0, vS, idx),
        'c4b_delta_ge_m1': ((f0 - fmp) - pts['h_m'], vS, idx),
        'c5_mono_T': (f0 - fTp, vT, idx),
        'c6_convex_S': (-(fmp - 2.0 * f0 + fmm), vS, idx),
    }


def shape_checks_grid(f, pts):
    F = f.reshape(pts['shape'])
    I = np.arange(f.size).reshape(pts['shape'])
    dm = np.diff(GRID_M)[None, None, None, :]
    a = F[..., 1:] - F[..., :-1]
    b = (F[..., :-1] - F[..., 1:]) - dm
    c = F[..., :-1, :] - F[..., 1:, :]
    d = -(F[..., 2:] - 2.0 * F[..., 1:-1] + F[..., :-2])
    def pack(v, ii):
        return (v.ravel(), np.ones(v.size, bool), ii.ravel())
    return {
        'c4a_mono_S': pack(a, I[..., :-1]),
        'c4b_delta_ge_m1': pack(b, I[..., :-1]),
        'c5_mono_T': pack(c, I[..., :-1, :]),
        'c6_convex_S': pack(d, I[..., 1:-1]),
    }


def autograd_checks(dm, dT, pts):
    n = len(dm)
    idx = np.arange(n)
    vS = vT = np.ones(n, bool)
    return {
        'd4a_delta_le0': (dm * pts['h_m'], vS, idx),
        'd4b_delta_gem1': ((-dm - 1.0) * pts['h_m'], vS, idx),
        'd5_theta_ge0': (-dT * pts['h_T'], vT, idx),
    }


def aggregate(checks, pts, tols, meta):
    reg_all = region_index(pts['m'], pts['T'])
    nreg = len(REGION_NAMES)
    rows = []
    for ck, (v, valid, idx) in checks.items():
        reg = reg_all[idx]
        K = pts['K'][idx]
        vv, rr, KK = v[valid], reg[valid], K[valid]
        n_pts = np.bincount(rr, minlength=nreg)
        for tol_name, tol in tols:
            viol = vv > tol
            n_v = np.bincount(rr[viol], minlength=nreg)
            s_v = np.bincount(rr[viol], weights=vv[viol], minlength=nreg)
            s_u = np.bincount(rr[viol], weights=vv[viol] * KK[viol], minlength=nreg)
            mx_v = np.full(nreg, np.nan)
            mx_u = np.full(nreg, np.nan)
            if viol.any():
                mx_v_ = np.full(nreg, -np.inf); mx_u_ = np.full(nreg, -np.inf)
                np.maximum.at(mx_v_, rr[viol], vv[viol])
                np.maximum.at(mx_u_, rr[viol], vv[viol] * KK[viol])
                mx_v = np.where(np.isfinite(mx_v_), mx_v_, np.nan)
                mx_u = np.where(np.isfinite(mx_u_), mx_u_, np.nan)
            for g in list(range(nreg)) + ['all']:
                if g == 'all':
                    npt, nv = int(n_pts.sum()), int(n_v.sum())
                    sv, su = float(s_v.sum()), float(s_u.sum())
                    mv = float(np.nanmax(mx_v)) if nv else np.nan
                    mu = float(np.nanmax(mx_u)) if nv else np.nan
                    region = 'összes'
                else:
                    npt, nv = int(n_pts[g]), int(n_v[g])
                    sv, su, mv, mu = float(s_v[g]), float(s_u[g]), float(mx_v[g]), float(mx_u[g])
                    region = REGION_NAMES[g]
                rows.append({**meta, 'set': pts['name'], 'check': ck, 'tol_name': tol_name,
                             'tol': tol, 'region': region, 'n_points': npt, 'n_viol': nv,
                             'rate': nv / npt if npt else np.nan,
                             'max_norm': mv if nv else np.nan,
                             'mean_norm': sv / nv if nv else np.nan,
                             'max_usd': mu if nv else np.nan,
                             'mean_usd': su / nv if nv else np.nan})
    return rows


def eval_nn_on_test(model, pts):
    m, T, r, s = pts['m'], pts['T'], pts['r'], pts['s']
    f0, dm, dT = E.nn_price_and_grads(model, m, T, r, s)
    f0n = E.nn_price_norm(model, m, T, r, s, batch_size=4096)
    fmp = E.nn_price_norm(model, m + pts['h_m'], T, r, s, batch_size=4096)
    fmm = E.nn_price_norm(model, m - pts['h_m'], T, r, s, batch_size=4096)
    fTp = E.nn_price_norm(model, m, T + pts['h_T'], r, s, batch_size=4096)
    return (f0n.astype(np.float64), fmp.astype(np.float64), fmm.astype(np.float64),
            fTp.astype(np.float64), dm, dT, float(np.abs(f0 - f0n).max()))


def fd_crosscheck(model, pts, rng_idx):
    m, T, r, s = (pts[k][rng_idx] for k in ('m', 'T', 'r', 's'))
    _, dm32, dT32 = E.nn_price_and_grads(model, m, T, r, s)
    md = copy.deepcopy(model).double().eval()

    def f64(mm, TT):
        X = torch.tensor(E.normalize(mm, TT, r, s), dtype=torch.float64, device=C.DEVICE)
        with torch.no_grad():
            return md(X).cpu().numpy().ravel()

    X = torch.tensor(E.normalize(m, T, r, s), dtype=torch.float64, device=C.DEVICE).requires_grad_(True)
    y = md(X)
    g, = torch.autograd.grad(y.sum(), X)
    g = g.detach().cpu().numpy()
    dm64, dT64 = g[:, 0] / E.NORM_SPAN[0], g[:, 1] / E.NORM_SPAN[1]
    fd_m = (f64(m + FD_H, T) - f64(m - FD_H, T)) / (2 * FD_H)
    fd_T = (f64(m, T + FD_H) - f64(m, T - FD_H)) / (2 * FD_H)
    del md
    out = {}
    for name, ad, fd in (('delta_ad64', dm64, fd_m), ('theta_ad64', dT64, fd_T),
                         ('delta_ad32', dm32, fd_m), ('theta_ad32', dT32, fd_T)):
        a = np.abs(ad - fd)
        rr = a / np.maximum(np.abs(fd), 1e-6)
        out[f"{name}_abs_median"] = float(np.median(a))
        out[f"{name}_rel_median"] = float(np.median(rr))
        out[f"{name}_rel_p95"] = float(np.quantile(rr, 0.95))
        out[f"{name}_rel_max"] = float(rr.max())
        out[f"{name}_share_rel_gt_1e-3"] = float((rr > 1e-3).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=None, help='csak ezek a run-ok (teszteleshez)')
    ap.add_argument('--skip-crr', action='store_true', help='a CRR-szelet kihagyasa (teszteleshez)')
    args = ap.parse_args()

    guard_path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f"Kornyezet: {E.env_extra()}")

    norm_report = {'formula': '(x - low) / (high - low), DEFAULT_PARAMS (data_generator_american.py 307-308. sor)',
                   'ranges': {p: list(E.DEFAULT_PARAMS[p]) for p in E.RAW_FEATURES}, 'splits': {}}
    train_minmax = {}
    for split in C.SPLITS:
        d = pd.read_parquet(C.am_split_path(split), columns=E.RAW_FEATURES + C.FEATURE_COLS)
        Xn = E.normalize(d['moneyness'], d['T'], d['r'], d['sigma'])
        Xs = d[C.FEATURE_COLS].to_numpy(np.float64)
        norm_report['splits'][split] = {
            'n': len(d),
            'float64_max_abs_diff': {p: float(np.abs(Xn[:, j] - Xs[:, j]).max()) for j, p in enumerate(E.RAW_FEATURES)},
            'float64_n_not_bitwise_equal': {p: int((Xn[:, j] != Xs[:, j]).sum()) for j, p in enumerate(E.RAW_FEATURES)},
            'float32_n_not_bitwise_equal': {p: int((Xn[:, j].astype(np.float32) != Xs[:, j].astype(np.float32)).sum())
                                            for j, p in enumerate(E.RAW_FEATURES)},
        }
        if split == 'train':
            train_minmax = {p: [float(d[p].min()), float(d[p].max())] for p in E.RAW_FEATURES}
        del d, Xn, Xs
    norm_report['float32_all_bitwise_equal'] = all(
        v == 0 for s in norm_report['splits'].values() for v in s['float32_n_not_bitwise_equal'].values())
    norm_report['float64_all_bitwise_equal'] = all(
        v == 0 for s in norm_report['splits'].values() for v in s['float64_n_not_bitwise_equal'].values())
    log(f"Normalizalas: float32 bitre egyezik: {norm_report['float32_all_bitwise_equal']}, "
        f"float64: {norm_report['float64_all_bitwise_equal']}")
    if not norm_report['float32_all_bitwise_equal']:
        raise SystemExit("A halo float32 bemenete nem egyezik az adathalmazeval -- leallas.")

    grid_def = {'m': [float(GRID_M[0]), float(GRID_M[-1]), len(GRID_M)],
                'T': [float(GRID_T[0]), float(GRID_T[-1]), len(GRID_T)],
                'r': GRID_R.tolist(), 'sigma': GRID_S.tolist(), 'n_points': int(len(GRID_M) * len(GRID_T) * 15),
                'K_ref_usd': K_GRID, 'train_minmax': train_minmax,
                'extrapolation_beyond_train_minmax': {
                    p: {'below_min': max(0.0, train_minmax[p][0] - lo), 'above_max': max(0.0, hi - train_minmax[p][1])}
                    for p, (lo, hi) in zip(['moneyness', 'T', 'r', 'sigma'],
                                           [(GRID_M[0], GRID_M[-1]), (GRID_T[0], GRID_T[-1]),
                                            (GRID_R[0], GRID_R[-1]), (GRID_S[0], GRID_S[-1])])},
                'within_DEFAULT_PARAMS': True,
                'test_steps': {'h_m': H_M, 'h_T': H_T},
                'grid_steps': {'h_m': float(GRID_M[1] - GRID_M[0]), 'h_T': float(GRID_T[1] - GRID_T[0])}}

    df = C.load_am_split('test')
    tp = test_points(df)
    gp = grid_points()
    grid_def['test_excluded'] = {'shape_S_checks_excluded_rows': int((~tp['valid_S']).sum()),
                                 'mono_T_excluded_rows': int((~tp['valid_T']).sum())}
    write_json(OUT_DIR / 'norm_check.json', norm_report)
    write_json(OUT_DIR / 'grid_definition.json', grid_def)

    sanity = {}
    rows = []
    tols_ref = [('0', 0.0), ('1e-6', 1e-6)]
    K = tp['K']
    f_lab = df['american_put_price'].to_numpy() / K
    f_bs_test = E.bs_put_norm(tp['m'], tp['T'], tp['r'], tp['s'])
    f_bs_parq = df['european_put_price'].to_numpy() / K
    f_crr_eu_parq = df['crr_european_put_price'].to_numpy() / K
    sanity['bs_parquet_vs_bs_formula_max_abs_norm'] = float(np.abs(f_bs_test - f_bs_parq).max())
    ch = level_checks(f_lab, tp, f_bs=f_bs_parq, f_crr_eu=f_crr_eu_parq)
    rows += aggregate(ch, tp, tols_ref, dict(run='label_crr500', arch='címke', target='american', level='címke'))
    bs_fun = E.bs_put_norm
    f0 = bs_fun(tp['m'], tp['T'], tp['r'], tp['s'])
    fmp = bs_fun(tp['m'] + H_M, tp['T'], tp['r'], tp['s'])
    fmm = bs_fun(tp['m'] - H_M, tp['T'], tp['r'], tp['s'])
    fTp = bs_fun(tp['m'], tp['T'] + H_T, tp['r'], tp['s'])
    ch = {**level_checks(f0, tp, f_bs=f_bs_test), **shape_checks_test(f0, fmp, fmm, fTp, tp)}
    rows += aggregate(ch, tp, tols_ref, dict(run='bs_put_model', arch='BS', target='european', level='zárt alak'))
    fg = bs_fun(gp['m'], gp['T'], gp['r'], gp['s'])
    ch = {**level_checks(fg, gp, f_bs=fg), **shape_checks_grid(fg, gp)}
    rows += aggregate(ch, gp, tols_ref, dict(run='bs_put_model', arch='BS', target='european', level='zárt alak'))
    if not args.skip_crr:
        cp = crr_slice_points()
        t0 = time.perf_counter()
        f_crr = E.AP.crr_american_price_batch(cp['m'], np.ones_like(cp['m']), cp['T'], cp['r'], cp['s'],
                                              option_type='put', n_steps=500)
        t_crr = time.perf_counter() - t0
        f_crr_eu = E.AP.crr_european_price_batch(cp['m'], np.ones_like(cp['m']), cp['T'], cp['r'], cp['s'],
                                                 option_type='put', n_steps=500)
        f_bs_c = bs_fun(cp['m'], cp['T'], cp['r'], cp['s'])
        ch = {**level_checks(f_crr, cp, f_bs=f_bs_c, f_crr_eu=f_crr_eu), **shape_checks_grid(f_crr, cp)}
        rows += aggregate(ch, cp, tols_ref, dict(run='crr500_slice', arch='CRR', target='american', level='címke'))
        sanity['crr_slice'] = {**CRR_SLICE, 'n_points': int(cp['m'].size), 'crr_seconds': t_crr,
                               'note': 'K = 1 (homogenitás), 500 lépés'}
        log(f"CRR-szelet: {cp['m'].size} ar, {t_crr:.1f} s")
    sanity_df = pd.DataFrame(rows)
    sanity_df.to_csv(guard_path(OUT_DIR / 'sanity_checks.csv'), index=False)
    tot = sanity_df[sanity_df.region == 'összes']
    sanity['summary_total'] = tot[['run', 'set', 'check', 'tol_name', 'n_points', 'n_viol', 'rate', 'max_norm']] \
        .to_dict(orient='records')

    def nviol(run, set_, check, tol='1e-6'):
        r = tot[(tot.run == run) & (tot.set == set_) & (tot.check == check) & (tot.tol_name == tol)]
        return int(r.n_viol.iloc[0]) if len(r) else None

    sanity['bs_model_meaningful_violations_tol1e-6'] = {
        f"{s}:{c}": nviol('bs_put_model', s, c) for s in ('test', 'grid')
        for c in ('c2_am_ge_bs', 'c3_le_K', 'c4a_mono_S', 'c4b_delta_ge_m1', 'c6_convex_S')}
    sanity['bs_model_ok'] = all(v == 0 for v in sanity['bs_model_meaningful_violations_tol1e-6'].values())
    if not args.skip_crr:
        sanity['crr_slice_violations_tol1e-6'] = {
            c: nviol('crr500_slice', 'crr_slice', c) for c in
            ('c1_intrinsic', 'c2_am_ge_bs', 'c2c_am_ge_crr_eu', 'c3_le_K', 'c4a_mono_S',
             'c4b_delta_ge_m1', 'c5_mono_T', 'c6_convex_S')}
        sanity['crr_slice_ok_1_3_4_5_and_2prime'] = all(
            sanity['crr_slice_violations_tol1e-6'][c] == 0 for c in
            ('c1_intrinsic', 'c2c_am_ge_crr_eu', 'c3_le_K', 'c4a_mono_S', 'c4b_delta_ge_m1', 'c5_mono_T'))
    sanity['label_test_violations'] = {
        f"{c}@{t}": nviol('label_crr500', 'test', c, t) for c in
        ('c1_intrinsic', 'c2_am_ge_bs', 'c2c_am_ge_crr_eu', 'c3_le_K') for t in ('0', '1e-6')}
    write_json(OUT_DIR / 'sanity.json', sanity)
    log(f"Szanity: BS ok = {sanity['bs_model_ok']}, CRR = {sanity.get('crr_slice_ok_1_3_4_5_and_2prime')}")

    reg = E.model_registry()
    if args.only:
        reg = [r for r in reg if r['run'] in args.only]
    rng = np.random.default_rng(42)
    ok_fd = np.flatnonzero((tp['m'] - 2 * FD_H > 0.5) & (tp['m'] + 2 * FD_H < 1.5)
                           & (tp['T'] - 2 * FD_H > 0.005) & (tp['T'] + 2 * FD_H < 2.0))
    fd_idx = np.sort(rng.choice(ok_fd, size=N_FD, replace=False))
    all_rows, fd_rows, pred_check = [], [], []
    best_points = {}
    for i, r in enumerate(reg):
        t0 = time.time()
        model, _ = E.load(r['checkpoint'])
        tols = [('0', 0.0), ('1e-6', 1e-6), ('RMSE', r['test_rmse'])]
        meta = dict(run=r['run'], arch=r['arch'], target=r['target'], level=r['level'])
        checks_list = CHECKS_AM if r['target'] == 'american' else CHECKS_EU
        f0, fmp, fmm, fTp, dm, dT, gdiff = eval_nn_on_test(model, tp)
        y_true = C.target_values(df, r['target']).astype(np.float32)
        rmse = compute_metrics(y_true, f0.astype(np.float32))['RMSE']
        pred_check.append({'run': r['run'], 'test_rmse_recomputed': rmse, 'test_rmse_levels_table': r['test_rmse'],
                           'rel_diff': abs(rmse - r['test_rmse']) / r['test_rmse'],
                           'grad_path_vs_predict_max_abs': gdiff})
        ch = {**level_checks(f0, tp, f_bs=f_bs_parq), **shape_checks_test(f0, fmp, fmm, fTp, tp)}
        ch = {k: v for k, v in ch.items() if k in checks_list}
        ch.update(autograd_checks(dm, dT, tp))
        all_rows += aggregate(ch, tp, tols, meta)
        if r['run'] == 'mlp_american_retune_p30':
            best_points['test'] = {k: (v[0], v[1]) for k, v in ch.items()}
        fg, dmg, dTg = E.nn_price_and_grads(model, gp['m'], gp['T'], gp['r'], gp['s'])
        fg = fg.astype(np.float64)
        fbs_g = E.bs_put_norm(gp['m'], gp['T'], gp['r'], gp['s'])
        chg = {**level_checks(fg, gp, f_bs=fbs_g), **shape_checks_grid(fg, gp)}
        chg = {k: v for k, v in chg.items() if k in checks_list}
        chg.update(autograd_checks(dmg, dTg, gp))
        all_rows += aggregate(chg, gp, tols, meta)
        if r['run'] == 'mlp_american_retune_p30':
            best_points['grid'] = {k: v for k, v in chg.items()}
        fd_rows.append({**meta, **fd_crosscheck(model, tp, fd_idx)})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"[{i + 1}/{len(reg)}] {r['run']}: {time.time() - t0:.1f} s")

    res = pd.DataFrame(all_rows)
    res['check_label'] = res['check'].map(CHECK_LABEL)
    res['main_check'] = [c in MAIN_CHECKS[t] for c, t in zip(res['check'], res['target'])]
    res.to_csv(guard_path(OUT_DIR / 'violations_by_region.csv'), index=False)
    pd.DataFrame(fd_rows).to_csv(guard_path(OUT_DIR / 'autograd_fd_check.csv'), index=False)
    pd.DataFrame(pred_check).to_csv(guard_path(OUT_DIR / 'prediction_check.csv'), index=False)

    if 'test' in best_points:
        bp = {'m': tp['m'], 'T': tp['T']}
        for k, (v, valid) in best_points['test'].items():
            bp[f"v_{k}"] = np.where(valid, v, np.nan)
        pd.DataFrame(bp).astype(np.float32).to_parquet(guard_path(OUT_DIR / 'best_model_test_points.parquet'),
                                                       index=False)
    if 'grid' in best_points:
        gpd = {}
        for k, (v, valid, idx) in best_points['grid'].items():
            gpd[k] = pd.DataFrame({'m': gp['m'][idx], 'T': gp['T'][idx], 'r': gp['r'][idx],
                                   'sigma': gp['s'][idx], 'v': v}).astype(np.float32)
        pd.concat({k: d for k, d in gpd.items()}, names=['check']).reset_index(level=0) \
            .to_parquet(guard_path(OUT_DIR / 'best_model_grid_points.parquet'), index=False)

    write_json(OUT_DIR / 'run_meta.json', {
        'script': 'american-nn/check_arbitrage.py', 'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seconds': time.time() - t_start, 'env': E.env_extra(), 'n_models': len(reg),
        'tolerances': ['0', '1e-6', 'RMSE (a modell teszt-RMSE-je, levels_table.csv)'],
        'fd_check': {'n_rows': N_FD, 'h': FD_H, 'seed': 42},
        'levels_table': rel(E.LEVELS_TABLE),
        'max_pred_rel_diff_vs_levels_table': float(pd.DataFrame(pred_check)['rel_diff'].max()),
    })
    log(f"Kesz: {time.time() - t_start:.1f} s")


if __name__ == '__main__':
    main()
