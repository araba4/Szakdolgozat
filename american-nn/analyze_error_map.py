import sys
sys.dont_write_bytecode = True

import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import common as C
import eval_common as E
from common import guard_path, write_json, rel

OUT_DIR = E.EVAL_DIR / "error_map"

N_M, N_T = 20, 20
N_M_S, N_T_S = 10, 10
M_EDGES = np.linspace(0.5, 1.5, N_M + 1)
T_EDGES = np.linspace(0.005, 2.0, N_T + 1)
M_EDGES_S = np.linspace(0.5, 1.5, N_M_S + 1)
T_EDGES_S = np.linspace(0.005, 2.0, N_T_S + 1)
SIGMA_EDGES = np.linspace(0.05, 0.9, 4)
SIGMA_LABELS = ['σ∈[0,05; 0,333)', 'σ∈[0,333; 0,617)', 'σ∈[0,617; 0,9]']
EX_EPS = 1e-9
BOUNDARY = (0.05, 0.95)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_models(lt):
    sel = {}
    for tgt in E.TARGETS:
        sub = lt[lt.target == tgt]
        best = sub.loc[sub.best_val_mse.idxmin()]
        sel[f'best_{tgt}'] = E.run_id(best.arch, tgt, best.level)
    runs = {sel['best_american'], sel['best_european']}
    for a in ('resnet', 'mlp'):
        for tgt in E.TARGETS:
            runs.add(E.run_id(a, tgt, 'retune_p30'))
    return sel, sorted(runs)


def bin_index(x, edges):
    i = np.digitize(x, edges[1:-1], right=False)
    return np.clip(i, 0, len(edges) - 2)


def cell_stats(err, err_usd, prem, prem_usd, ex, mi, ti, nm, nt):
    cid = mi * nt + ti
    nc = nm * nt
    n = np.bincount(cid, minlength=nc).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = {
            'n': n,
            'rmse_norm': np.sqrt(np.bincount(cid, err ** 2, nc) / n),
            'bias_norm': np.bincount(cid, err, nc) / n,
            'sse_norm': np.bincount(cid, err ** 2, nc),
            'rmse_usd': np.sqrt(np.bincount(cid, err_usd ** 2, nc) / n),
            'bias_usd': np.bincount(cid, err_usd, nc) / n,
            'premium_mean_norm': np.bincount(cid, prem, nc) / n,
            'premium_mean_usd': np.bincount(cid, prem_usd, nc) / n,
            'exercise_frac': np.bincount(cid, ex.astype(float), nc) / n,
        }
    return out


def to_frame(st, run, sigma_slice, m_edges, t_edges):
    nm, nt = len(m_edges) - 1, len(t_edges) - 1
    mi, ti = np.divmod(np.arange(nm * nt), nt)
    d = pd.DataFrame({'run': run, 'sigma_slice': sigma_slice, 'm_bin': mi, 'T_bin': ti,
                      'm_lo': m_edges[mi], 'm_hi': m_edges[mi + 1],
                      'T_lo': t_edges[ti], 'T_hi': t_edges[ti + 1], **st})
    mc = 0.5 * (d.m_lo + d.m_hi)
    Tc = 0.5 * (d.T_lo + d.T_hi)
    d['region'] = [E_region(a, b) for a, b in zip(mc, Tc)]
    return d


def E_region(m, T):
    mk = next(k for k, (lo, hi) in C.PUT_MONEYNESS_SEGMENTS.items() if lo <= m < hi)
    tk = next(k for k, (lo, hi) in C.MATURITY_SEGMENTS.items() if lo <= T < hi)
    return f"{mk} × {tk}"


def pooled_rmse(err, mask):
    return float(np.sqrt(np.mean(err[mask] ** 2))) if mask.any() else float('nan')


def main():
    guard_path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lt = E.levels_table()
    sel, runs = select_models(lt)
    log(f"Kivalasztas: {sel}; modellek: {runs}")
    reg = {r['run']: r for r in E.model_registry()}

    df = C.load_am_split('test')
    m = df['moneyness'].to_numpy(np.float64)
    T = df['T'].to_numpy(np.float64)
    sig = df['sigma'].to_numpy(np.float64)
    K = df['K'].to_numpy(np.float64)
    S = df['S'].to_numpy(np.float64)
    prem = df['early_exercise_premium'].to_numpy(np.float64) / K
    am = df['american_put_price'].to_numpy(np.float64)
    ex = (K > S) & ((am - (K - S)) < EX_EPS * K)
    mi, ti = bin_index(m, M_EDGES), bin_index(T, T_EDGES)
    mis, tis = bin_index(m, M_EDGES_S), bin_index(T, T_EDGES_S)
    si = bin_index(sig, SIGMA_EDGES)

    frames, agg_check, errs = [], [], {}
    for run in runs:
        r = reg[run]
        model, _ = E.load(r['checkpoint'])
        pred = E.nn_price_norm(model, m, T, df['r'], sig, batch_size=4096)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        y32 = C.target_values(df, r['target']).astype(np.float32)
        err = pred.astype(np.float64) - y32.astype(np.float64)
        err_usd = err * K
        errs[run] = err
        st = cell_stats(err, err_usd, prem, prem * K, ex, mi, ti, N_M, N_T)
        frames.append(to_frame(st, run, 'összes σ', M_EDGES, T_EDGES))
        for k, lab in enumerate(SIGMA_LABELS):
            msk = si == k
            st_s = cell_stats(err[msk], err_usd[msk], prem[msk], prem[msk] * K[msk], ex[msk],
                              mis[msk], tis[msk], N_M_S, N_T_S)
            frames.append(to_frame(st_s, run, lab, M_EDGES_S, T_EDGES_S))
        rmse_agg = float(np.sqrt(st['sse_norm'].sum() / st['n'].sum()))
        agg_check.append({'run': run, 'n_rows': int(st['n'].sum()), 'test_rmse_from_cells': rmse_agg,
                          'test_rmse_levels_table': r['test_rmse'],
                          'rel_diff': abs(rmse_agg - r['test_rmse']) / r['test_rmse'],
                          'source': r['source_file']})
        log(f"{run}: RMSE cellakbol {rmse_agg:.6e} vs. levels_table {r['test_rmse']:.6e}")

    cells = pd.concat(frames, ignore_index=True)
    ratio_rows = []
    for run in runs:
        if '_american_' not in run:
            continue
        eu = run.replace('_american_', '_european_')
        if eu not in runs:
            continue
        a = cells[cells.run == run].reset_index(drop=True)
        b = cells[cells.run == eu].reset_index(drop=True)
        rr = a[['sigma_slice', 'm_bin', 'T_bin', 'm_lo', 'm_hi', 'T_lo', 'T_hi', 'region', 'n',
                'premium_mean_norm', 'premium_mean_usd', 'exercise_frac']].copy()
        rr['pair'] = run.replace('_american', '')
        rr['rmse_am'] = a.rmse_norm
        rr['rmse_eu'] = b.rmse_norm
        rr['ratio_am_eu'] = a.rmse_norm / b.rmse_norm
        ratio_rows.append(rr)
    ratios = pd.concat(ratio_rows, ignore_index=True)
    cells = cells.drop(columns=['sse_norm'])
    cells.to_csv(guard_path(OUT_DIR / 'error_map_cells.csv'), index=False)
    ratios.to_csv(guard_path(OUT_DIR / 'ratio_cells.csv'), index=False)
    write_json(OUT_DIR / 'aggregation_check.json', {'rows': agg_check,
                                                     'max_rel_diff': max(x['rel_diff'] for x in agg_check)})

    itm_long = (m < 0.9) & (T >= 1.0)
    hyp = {'selection': sel, 'runs': runs, 'definitions': {
        'exercise_row': f'K > S és american − (K−S) < {EX_EPS}·K', 'boundary_cell': f'{BOUNDARY[0]} < lehívási arány < {BOUNDARY[1]}',
        'itm_long': 'm < 0,9 és T ≥ 1', 'bins': f'{N_M}×{N_T} egyenletes (m, T); σ-szeletben {N_M_S}×{N_T_S}'},
        'test_rows_exercise_share': float(ex.mean()), 'models': {}, 'pairs': {}}
    full = cells[cells.sigma_slice == 'összes σ']
    for run in runs:
        c = full[full.run == run]
        err = errs[run]
        top = c.nlargest(int(round(0.1 * len(c))), 'rmse_norm')
        bnd = (c.exercise_frac > BOUNDARY[0]) & (c.exercise_frac < BOUNDARY[1])
        rho, p = spearmanr(c.rmse_norm, c.premium_mean_norm)
        rho_u, p_u = spearmanr(c.rmse_usd, c.premium_mean_usd)
        hyp['models'][run] = {
            'spearman_cell_rmse_vs_premium_norm': {'rho': float(rho), 'p': float(p)},
            'spearman_cell_rmse_vs_premium_usd': {'rho': float(rho_u), 'p': float(p_u)},
            'rmse_itm_long': pooled_rmse(err, itm_long), 'rmse_rest': pooled_rmse(err, ~itm_long),
            'rmse_exercise_rows': pooled_rmse(err, ex), 'rmse_continuation_rows': pooled_rmse(err, ~ex),
            'n_boundary_cells': int(bnd.sum()),
            'rmse_boundary_cells': float(np.sqrt((c.rmse_norm[bnd] ** 2 * c.n[bnd]).sum() / c.n[bnd].sum())),
            'rmse_other_cells': float(np.sqrt((c.rmse_norm[~bnd] ** 2 * c.n[~bnd]).sum() / c.n[~bnd].sum())),
            'top10pct_cells': {'n': len(top), 'share_itm_long_region': float((top.region == 'ITM (S/K<0,9) × hosszú (1≤T≤2)').mean()),
                               'share_boundary': float(((top.exercise_frac > BOUNDARY[0]) & (top.exercise_frac < BOUNDARY[1])).mean()),
                               'share_short_T': float((top.T_hi <= 0.25 + 1e-9).mean()),
                               'share_atm': float(((top.m_lo >= 0.9 - 1e-9) & (top.m_hi <= 1.1 + 1e-9)).mean()),
                               'mean_premium_norm_top': float(top.premium_mean_norm.mean()),
                               'mean_premium_norm_all': float(c.premium_mean_norm.mean()),
                               'cells': top[['m_lo', 'm_hi', 'T_lo', 'T_hi', 'rmse_norm', 'premium_mean_norm',
                                             'exercise_frac']].to_dict(orient='records')},
            'worst_cell': c.loc[c.rmse_norm.idxmax(), ['m_lo', 'm_hi', 'T_lo', 'T_hi', 'rmse_norm', 'rmse_usd',
                                                       'bias_norm', 'premium_mean_norm']].to_dict(),
        }
    rf = ratios[ratios.sigma_slice == 'összes σ']
    for pair, c in rf.groupby('pair'):
        bnd = (c.exercise_frac > BOUNDARY[0]) & (c.exercise_frac < BOUNDARY[1])
        rho, p = spearmanr(c.ratio_am_eu, c.premium_mean_norm)
        hi_prem = c.premium_mean_norm >= c.premium_mean_norm.quantile(0.9)
        am_run, eu_run = pair.replace('_retune', '_american_retune'), pair.replace('_retune', '_european_retune')
        ea, ee = errs[am_run], errs[eu_run]
        hyp['pairs'][pair] = {
            'spearman_ratio_vs_premium': {'rho': float(rho), 'p': float(p)},
            'ratio_median_all_cells': float(c.ratio_am_eu.median()),
            'ratio_median_top10pct_premium_cells': float(c.ratio_am_eu[hi_prem].median()),
            'ratio_median_boundary_cells': float(c.ratio_am_eu[bnd].median()),
            'ratio_median_other_cells': float(c.ratio_am_eu[~bnd].median()),
            'share_cells_ratio_gt_1': float((c.ratio_am_eu > 1).mean()),
            'pooled_ratio_itm_long': pooled_rmse(ea, itm_long) / pooled_rmse(ee, itm_long),
            'pooled_ratio_rest': pooled_rmse(ea, ~itm_long) / pooled_rmse(ee, ~itm_long),
            'pooled_ratio_exercise_rows': pooled_rmse(ea, ex) / pooled_rmse(ee, ex),
            'pooled_ratio_all': pooled_rmse(ea, np.ones_like(ex)) / pooled_rmse(ee, np.ones_like(ex)),
        }
    write_json(OUT_DIR / 'hypothesis.json', hyp)
    write_json(OUT_DIR / 'run_meta.json', {'script': 'american-nn/analyze_error_map.py',
                                           'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                                           'seconds': time.time() - t0, 'env': E.env_extra(),
                                           'levels_table': rel(E.LEVELS_TABLE)})
    log(f"Kesz: {time.time() - t0:.1f} s")


if __name__ == '__main__':
    main()
