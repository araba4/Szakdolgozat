import sys
sys.dont_write_bytecode = True

import argparse
import json
import time

import numpy as np
import pandas as pd

t_import0 = time.perf_counter()
import torch
T_IMPORT_TORCH = time.perf_counter() - t_import0

import common as C
import eval_common as E
from common import guard_path, write_json, read_json, rel
import benchmark_speed as B
import data_generator_american as G

N_FULL = 1_000_000
SEED = 42
ARCH = 'mlp'
CKPT = C.AMNN_DIR / "models" / "retune" / "mlp_american_retune_p30_best.pt"
REPS = {'cuda': 5, 'cpu': 3}
N_CRR_SAMPLE = 1000
CRR_REPS = 3
INFER_CHUNK_CPU = 100_000
QA_REL_TOL, QA_ABS_TOL = G.QA_REL_TOL, G.QA_ABS_TOL
SPLITS = ['train', 'val', 'test']
INPUT_COLS = ['S', 'K', 'moneyness', 'T', 'r', 'sigma']
NORM_COLS = ['S_norm', 'moneyness_norm', 'T_norm', 'r_norm', 'sigma_norm']


def log(msg):
    B.log(msg)


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def generate(model, device, n, write_dir=None):
    t = {}
    sync(device)
    T0 = time.perf_counter()

    t0 = time.perf_counter()
    samples = G._sample_lhs(n, SEED)
    S, m, T, r, sig = (samples[:, i] for i in range(len(G.PARAM_NAMES)))
    K = S / m
    t['1_lhs_mintavetel'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    X = E.normalize(m, T, r, sig)
    Xt = torch.tensor(X, dtype=torch.float32)
    t['2_normalizalas'] = time.perf_counter() - t0

    sync(device)
    t0 = time.perf_counter()
    Xd = Xt.to(device)
    sync(device)
    t['3_masolas_eszkozre'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.inference_mode():
        if device == 'cpu':
            y = torch.cat([model(Xd[i:i + INFER_CHUNK_CPU]) for i in range(0, n, INFER_CHUNK_CPU)])
        else:
            y = model(Xd)
    sync(device)
    t['4_inferencia'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    f = y.cpu().numpy().ravel()
    t['5_masolas_vissza'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    price = K * f.astype(np.float64)
    t['6_arszamitas'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    bs = G._bs_put(S, K, T, r, sig)
    t['7_bs_put'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = pd.DataFrame({'S': S, 'K': K, 'moneyness': m, 'T': T, 'r': r, 'sigma': sig,
                       'american_put_price_nn': price, 'american_put_price_nn_norm': f,
                       'european_put_price': bs})
    for p in ['S', 'moneyness', 'T', 'r', 'sigma']:
        lo, hi = G.DEFAULT_PARAMS[p]
        df[f"{p}_norm"] = (df[p] - lo) / (hi - lo)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    splits = {'train': df.iloc[:n_tr], 'val': df.iloc[n_tr:n_tr + n_va], 'test': df.iloc[n_tr + n_va:]}
    t['8_dataframe_felosztas'] = time.perf_counter() - t0

    t['mag_mintavetel_tol_arig'] = sum(t[k] for k in ('1_lhs_mintavetel', '2_normalizalas', '3_masolas_eszkozre',
                                                       '4_inferencia', '5_masolas_vissza', '6_arszamitas'))
    if write_dir is not None:
        t0 = time.perf_counter()
        write_dir.mkdir(parents=True, exist_ok=True)
        for name, sdf in splits.items():
            sdf.to_parquet(guard_path(write_dir / f"{name}.parquet"), index=False)
        t['9_parquet_iras'] = time.perf_counter() - t0
    t['osszesen'] = time.perf_counter() - T0
    del Xd, y
    return splits, t


def load_crr(n):
    parts = []
    for s in SPLITS:
        d = pd.read_parquet(C.am_split_path(s))
        d['split'] = s
        parts.append(d)
    crr = pd.concat(parts, ignore_index=True)
    if n != len(crr):
        crr = None
    return crr


def completeness(splits, n):
    out = {'n_rows': int(sum(len(v) for v in splits.values())), 'n_expected': n,
           'split_sizes': {k: int(len(v)) for k, v in splits.items()},
           'split_sizes_expected': {'train': int(0.70 * n), 'val': int(0.15 * n),
                                    'test': n - int(0.70 * n) - int(0.15 * n)},
           'columns': list(next(iter(splits.values())).columns)}
    allv = pd.concat(splits.values(), ignore_index=True)
    num = allv.select_dtypes(include=[np.number])
    out['n_nan'] = {c: int(v) for c, v in num.isna().sum().items() if v > 0}
    out['n_inf'] = {c: int(v) for c, v in np.isinf(num).sum().items() if v > 0}
    out['n_duplicate_input_rows'] = int(allv.duplicated(subset=INPUT_COLS).sum())
    out['ok'] = (out['n_rows'] == n and out['split_sizes'] == out['split_sizes_expected']
                 and not out['n_nan'] and not out['n_inf'])
    return out


def alignment(nn_all, crr):
    out = {}
    for c in INPUT_COLS + NORM_COLS:
        a, b = nn_all[c].to_numpy(), crr[c].to_numpy()
        out[c] = {'n_not_bitwise_equal': int((a != b).sum()), 'max_abs_diff': float(np.abs(a - b).max())}
    split_nn = np.repeat(SPLITS, [int((nn_all.split == s).sum()) for s in SPLITS])
    out['split_labels_equal'] = bool((split_nn == crr['split'].to_numpy()).all())
    out['all_bitwise_equal'] = all(v['n_not_bitwise_equal'] == 0 for k, v in out.items() if isinstance(v, dict))
    return out


def err_stats(y_true, y_pred, K):
    d = y_pred - y_true
    du = d * K
    ad = np.abs(du)
    q = np.quantile(ad, [0.5, 0.9, 0.99, 0.999])
    return {'n': int(len(d)), 'rmse_norm': float(np.sqrt(np.mean(d ** 2))), 'mae_norm': float(np.mean(np.abs(d))),
            'max_abs_norm': float(np.abs(d).max()), 'bias_norm': float(d.mean()),
            'rmse_usd': float(np.sqrt(np.mean(du ** 2))), 'mae_usd': float(ad.mean()),
            'abs_err_usd_p50': float(q[0]), 'abs_err_usd_p90': float(q[1]), 'abs_err_usd_p99': float(q[2]),
            'abs_err_usd_p999': float(q[3]), 'max_abs_usd': float(ad.max()), 'bias_usd': float(du.mean()),
            'pct_within_qa_tol': float(100 * np.mean((ad <= QA_ABS_TOL) |
                                                     (ad <= QA_REL_TOL * np.abs(y_true * K))))}


def accuracy(df):
    y = (df['american_put_price'] / df['K']).to_numpy()
    p = df['american_put_price_nn_norm'].to_numpy().astype(np.float64)
    K = df['K'].to_numpy()
    rows = []
    for s in SPLITS + ['összes']:
        msk = np.ones(len(df), bool) if s == 'összes' else (df.split == s).to_numpy()
        rows.append({'split': s, **err_stats(y[msk], p[msk], K[msk])})
    return pd.DataFrame(rows)


def segments(df_test):
    y = (df_test['american_put_price'] / df_test['K']).to_numpy()
    p = df_test['american_put_price_nn_norm'].to_numpy().astype(np.float64)
    K = df_test['K'].to_numpy()
    edges = C.premium_bucket_edges((df_test['early_exercise_premium'] / df_test['K']).to_numpy())
    rows = []
    for grp, masks in C.segment_masks(df_test, edges).items():
        for name, msk in masks.items():
            if msk.sum():
                rows.append({'csoport': grp, 'szegmens': name, **err_stats(y[msk], p[msk], K[msk])})
    return pd.DataFrame(rows)


def lsm_qa(df):
    q = df[df['is_qa_lsm']]
    lsm = q['lsm_put_price'].to_numpy()
    out = {}
    for name, price in (('nn', q['american_put_price_nn'].to_numpy()), ('crr', q['american_put_price'].to_numpy())):
        ad = np.abs(price - lsm)
        ok = (ad <= QA_ABS_TOL) | (ad / np.maximum(np.abs(price), 1e-8) <= QA_REL_TOL)
        out[name] = {'n': int(len(q)), 'n_within_tol': int(ok.sum()), 'pct_within_tol': float(100 * ok.mean()),
                     'abs_diff_median_usd': float(np.median(ad)), 'abs_diff_mean_usd': float(ad.mean()),
                     'abs_diff_max_usd': float(ad.max()), 'rmse_usd': float(np.sqrt(np.mean(ad ** 2)))}
        out[name]['by_split'] = {s: {'n': int((q.split == s).sum()), 'n_within_tol': int(ok[(q.split == s).to_numpy()].sum())}
                                 for s in SPLITS}
    pn = q['american_put_price_nn'].to_numpy()
    adn = np.abs(pn - lsm)
    fail = q[~((adn <= QA_ABS_TOL) | (adn / np.maximum(np.abs(pn), 1e-8) <= QA_REL_TOL))]
    out['nn_failures'] = fail[['split', 'S', 'K', 'moneyness', 'T', 'r', 'sigma', 'american_put_price',
                               'american_put_price_nn', 'lsm_put_price']].head(50).to_dict(orient='records')
    out['tolerance'] = {'rel': QA_REL_TOL, 'abs_usd': QA_ABS_TOL, 'source': 'data_generator_american'}
    return out


def arbitrage(df, taus):
    K = df['K'].to_numpy()
    S = df['S'].to_numpy()
    bs = df['european_put_price'].to_numpy() / K
    intr = np.maximum(K - S, 0.0) / K
    rows = []
    srcs = [('nn', df['american_put_price_nn_norm'].to_numpy().astype(np.float64))]
    if 'american_put_price' in df:
        srcs.append(('crr_cimke', (df['american_put_price'] / df['K']).to_numpy()))
    for src, P in srcs:
        for tau_name, tau in taus.items():
            checks = {'P ≥ 0': -P, 'P ≤ K': P - 1.0, 'P ≥ max(K−S,0)': intr - P, 'P_am ≥ P_BS': bs - P}
            for cname, viol in checks.items():
                bad = viol > tau
                rows.append({'forras': src, 'tau': tau_name, 'tau_norm': tau, 'feltetel': cname,
                             'n_serul': int(bad.sum()), 'pct_serul': float(100 * bad.mean()),
                             'max_serules_norm': float(max(viol.max(), 0.0)),
                             'max_serules_usd': float(max((viol * K).max(), 0.0))})
    return pd.DataFrame(rows)


def premium_view(df):
    t = df[df.split == 'test']
    nn_ex = (t['american_put_price_nn'] - t['european_put_price']).to_numpy()
    lab_ex = (t['american_put_price'] - t['european_put_price']).to_numpy()
    return {'n': int(len(t)), 'nn_minus_bs_negative_pct': float(100 * np.mean(nn_ex < 0)),
            'label_minus_bs_negative_pct': float(100 * np.mean(lab_ex < 0)),
            'nn_minus_bs_min_usd': float(nn_ex.min()), 'label_minus_bs_min_usd': float(lab_ex.min()),
            'nn_minus_bs_mean_usd': float(nn_ex.mean()), 'label_minus_bs_mean_usd': float(lab_ex.mean()),
            'rmse_of_excess_usd': float(np.sqrt(np.mean((nn_ex - lab_ex) ** 2)))}


def crr_sample_timing(n_rows):
    rng = np.random.default_rng(SEED)
    samples = G._sample_lhs(N_FULL, SEED)
    idx = np.sort(rng.choice(N_FULL, size=n_rows, replace=False))
    S, m, T, r, sig = (samples[idx, i] for i in range(5))
    K = S / m
    E.AP.crr_american_price(S[0], K[0], T[0], r[0], sig[0], option_type='put', n_steps=500)
    ts = []
    for _ in range(CRR_REPS):
        t0 = time.perf_counter()
        E.AP.crr_american_price_batch(S, K, T, r, sig, option_type='put', n_steps=500)
        ts.append((time.perf_counter() - t0) / n_rows)
    return {'n_rows': n_rows, 'reps': CRR_REPS, 's_per_row_each': ts, 's_per_row_median': float(np.median(ts))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    n = 10_000 if args.smoke else N_FULL
    sub = 'smoke' if args.smoke else ''
    out_dir = C.AMNN_DIR / "results" / "nn_dataset" / sub
    data_dir = C.AMNN_DIR / "data" / "nn_generated" / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    guard_path(out_dir / "x")
    T_RUN = time.perf_counter()
    meta = {'script': 'american-nn/generate_nn_dataset.py', 'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n': n, 'seed': SEED, 'checkpoint': rel(CKPT), 'env': E.env_extra(),
            'torch_import_s': T_IMPORT_TORCH}
    log(f"Gep allapota (elotte), n = {n:,}")
    meta['machine_state_before'] = B.machine_state('előtte')

    t0 = time.perf_counter()
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    meta['cuda_init_s'] = time.perf_counter() - t0
    models = {}
    for dev in ('cuda', 'cpu'):
        t0 = time.perf_counter()
        models[dev], _ = E.load(CKPT, device=dev)
        sync(dev)
        meta[f'model_load_{dev}_s'] = time.perf_counter() - t0

    timings = []
    splits_final = None
    for dev in ('cuda', 'cpu'):
        for rep in range(REPS[dev] + 1):
            wd = data_dir if (dev == 'cuda' and rep == REPS[dev]) else None
            log(f"Generalas: {dev}, ismetles {rep}{' (hideg)' if rep == 0 else ''}{' + iras' if wd else ''}")
            splits, t = generate(models[dev], dev, n, write_dir=wd)
            timings.append({'device': dev, 'rep': rep, 'cold': rep == 0, **t})
            if dev == 'cuda' and rep == REPS[dev]:
                splits_final = splits
            elif dev == 'cpu' and rep == REPS[dev]:
                splits_cpu = splits
    tim = pd.DataFrame(timings)
    tim.to_csv(guard_path(out_dir / "timings.csv"), index=False)

    log(f"CRR-500 minta ({N_CRR_SAMPLE} sor x {CRR_REPS})")
    crr_t = crr_sample_timing(min(N_CRR_SAMPLE, n))
    meta['crr_sample_timing'] = crr_t
    gen_meta = read_json(C.AM_DATA_DIR / "generation_meta.json")

    a = pd.concat(splits_final.values())['american_put_price_nn_norm'].to_numpy()
    b = pd.concat(splits_cpu.values())['american_put_price_nn_norm'].to_numpy()
    meta['gpu_vs_cpu_norm_max_abs_diff'] = float(np.abs(a.astype(np.float64) - b).max())
    meta['gpu_vs_cpu_n_bitwise_equal'] = int((a == b).sum())

    log("Ellenorzesek")
    res = {'completeness': completeness(splits_final, n)}
    back = {s: pd.read_parquet(data_dir / f"{s}.parquet") for s in SPLITS}
    res['readback_equal'] = all(back[s].equals(splits_final[s].reset_index(drop=True)) for s in SPLITS)
    res['parquet_bytes'] = {s: (data_dir / f"{s}.parquet").stat().st_size for s in SPLITS}
    nn_all = pd.concat([splits_final[s].assign(split=s) for s in SPLITS], ignore_index=True)

    crr = load_crr(n)
    lt = pd.read_csv(C.AMNN_DIR / "results" / "retune" / "levels_table.csv")
    lt = lt[(lt.arch == ARCH) & (lt.target == 'american') & (lt.level == 'retune_p30')].iloc[0]
    if crr is not None:
        res['alignment_vs_crr_dataset'] = alignment(nn_all, crr)
        both = nn_all.copy()
        for c in ['american_put_price', 'crr_european_put_price', 'early_exercise_premium', 'lsm_put_price',
                  'is_qa_lsm']:
            both[c] = crr[c].to_numpy()
        res['bs_column_equal_to_crr_dataset'] = {
            'n_not_bitwise_equal': int((both['european_put_price'].to_numpy() != crr['european_put_price'].to_numpy()).sum()),
            'max_abs_diff': float(np.abs(both['european_put_price'].to_numpy() - crr['european_put_price'].to_numpy()).max())}
        acc = accuracy(both)
        acc.to_csv(guard_path(out_dir / "accuracy_by_split.csv"), index=False)
        te = acc[acc.split == 'test'].iloc[0]
        res['reproduces_levels_table'] = {
            'test_rmse_now': float(te.rmse_norm), 'test_rmse_levels_table': float(lt.test_rmse),
            'rel_diff': float(te.rmse_norm / lt.test_rmse - 1),
            'test_usd_rmse_now': float(te.rmse_usd), 'test_usd_rmse_levels_table': float(lt.test_usd_rmse)}
        segments(both[both.split == 'test']).to_csv(guard_path(out_dir / "segments_test.csv"), index=False)
        res['lsm_qa'] = lsm_qa(both)
        tau_rmse = float(lt.test_rmse)
        arbitrage(both, {'0': 0.0, '1e-6': 1e-6, 'teszt-RMSE': tau_rmse}).to_csv(
            guard_path(out_dir / "arbitrage_levels.csv"), index=False)
        res['excess_over_bs_test'] = premium_view(both)

    warm = tim[~tim.cold]
    med = warm.groupby('device').median(numeric_only=True)
    crr_row = crr_t['s_per_row_median']
    summ = {}
    for dev in ('cuda', 'cpu'):
        tot = float(med.loc[dev, 'osszesen'])
        core = float(med.loc[dev, 'mag_mintavetel_tol_arig'])
        inf = float(med.loc[dev, '4_inferencia'])
        summ[dev] = {'osszesen_s_median': tot, 'mag_s_median': core, 'inferencia_s_median': inf,
                     'hideg_osszesen_s': float(tim[(tim.device == dev) & tim.cold].osszesen.iloc[0]),
                     'ns_per_row_osszesen': 1e9 * tot / n, 'ns_per_row_inferencia': 1e9 * inf / n,
                     'gyorsulas_vs_crr_most_osszesen': crr_row * n / tot,
                     'gyorsulas_vs_crr_most_inferencia': crr_row * n / inf,
                     'gyorsulas_vs_crr_generalaskor_osszesen': gen_meta['crr_elapsed_s'] * n / N_FULL / tot}
    gw = tim[(tim.device == 'cuda') & (tim.rep == REPS['cuda'])].iloc[0]
    summ['cuda_iras_s'] = float(gw['9_parquet_iras'])
    summ['egyszeri'] = {'torch_import_s': T_IMPORT_TORCH, 'cuda_init_s': meta['cuda_init_s'],
                        'model_load_cuda_s': meta['model_load_cuda_s'], 'model_load_cpu_s': meta['model_load_cpu_s']}
    summ['vegponttol_vegpontig_gpu_s'] = (T_IMPORT_TORCH + meta['cuda_init_s'] + meta['model_load_cuda_s']
                                          + float(gw['osszesen']))
    summ['crr_most_s_per_row'] = crr_row
    summ['crr_most_extrapolalt_1M_s'] = crr_row * N_FULL
    summ['crr_generalaskor'] = {'crr_elapsed_s': gen_meta['crr_elapsed_s'], 'crr_eu_elapsed_s': gen_meta['crr_eu_elapsed_s'],
                                'lsm_elapsed_s': gen_meta['lsm_elapsed_s'], 'total_elapsed_s': gen_meta['total_elapsed_s']}
    res['speed_summary'] = summ

    meta['machine_state_after'] = B.machine_state('utána')
    meta['seconds'] = time.perf_counter() - T_RUN
    meta['results'] = res
    write_json(out_dir / "nn_dataset_meta.json", meta)
    log(f"Kesz: {meta['seconds']:.1f} s")
    print(json.dumps(res['speed_summary'], indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
