import os
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

from american_pricers import (
    crr_american_price,
    crr_american_price_batch,
    crr_european_price_batch,
    lsm_american_price,
)


DEFAULT_PARAMS = {
    'S':         (10.0, 150.0),
    'moneyness': (0.5,    1.5),
    'T':         (0.005,  2.0),
    'r':         (0.00,  0.05),
    'sigma':     (0.05,  0.90),
}

PARAM_NAMES = ['S', 'moneyness', 'T', 'r', 'sigma']

QA_REL_TOL = 0.02
QA_ABS_TOL = 0.05


def _sample_lhs(n: int, seed: int) -> np.ndarray:
    bounds = np.array([DEFAULT_PARAMS[p] for p in PARAM_NAMES])
    low, high = bounds[:, 0], bounds[:, 1]
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit_sample = sampler.random(n)
    return qmc.scale(unit_sample, low, high)


def _bs_put(S, K, T, r, sigma):
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    near_expiry = T < 1e-8
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_intrinsic = np.maximum(S - K * np.exp(-r * T), 0.0)
    call_price = np.where(
        near_expiry,
        call_intrinsic,
        S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
    )
    put_price = call_price - S + K * np.exp(-r * T)
    return put_price


def generate_dataset_american(
    n: int,
    seed: int = 42,
    crr_n_steps: int = 500,
    crr_chunk_size: int = 10_000,
    n_qa_lsm: int = 2000,
    lsm_n_paths: int = 100_000,
    lsm_degree: int = 3,
    lsm_antithetic: bool = True,
    verbose: bool = True,
):
    rng = np.random.default_rng(seed)
    samples = _sample_lhs(n, seed)
    S, moneyness, T, r, sigma = (samples[:, i] for i in range(len(PARAM_NAMES)))
    K = S / moneyness

    failed_rows = []

    american_put_price = np.empty(n, dtype=float)
    t_start_crr = time.perf_counter()
    chunk_size = max(1, min(crr_chunk_size, n))
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        try:
            american_put_price[start:end] = crr_american_price_batch(
                S[start:end], K[start:end], T[start:end], r[start:end], sigma[start:end],
                option_type='put', n_steps=crr_n_steps,
            )
        except Exception:
            for idx in range(start, end):
                try:
                    american_put_price[idx] = crr_american_price(
                        S[idx], K[idx], T[idx], r[idx], sigma[idx],
                        option_type='put', n_steps=crr_n_steps,
                    )
                except Exception as exc2:
                    failed_rows.append((int(idx), str(exc2)))
                    american_put_price[idx] = np.nan

        if verbose:
            elapsed = time.perf_counter() - t_start_crr
            rate = end / elapsed if elapsed > 0 else float('inf')
            eta = (n - end) / rate if rate > 0 else float('nan')
            print(f"  CRR: {end:>9}/{n} kész ({elapsed:8.1f}s eltelt, "
                  f"becsült hátralévő idő: {eta:8.1f}s, {rate:7.1f} sor/s)")

    crr_elapsed = time.perf_counter() - t_start_crr

    european_put_price = _bs_put(S, K, T, r, sigma)

    t_start_eu = time.perf_counter()
    crr_european_put_price = crr_european_price_batch(
        S, K, T, r, sigma, option_type='put', n_steps=crr_n_steps,
    )
    crr_eu_elapsed = time.perf_counter() - t_start_eu
    early_exercise_premium = american_put_price - crr_european_put_price
    n_negative_premium = int(np.sum(early_exercise_premium < 0.0))
    if n_negative_premium > 0:
        raise RuntimeError(f"{n_negative_premium} sorban negatív a fa-alapú prémium")
    crr_eu_vs_bs = np.abs(crr_european_put_price - european_put_price)
    if verbose:
        print(f"  CRR európai (ugyanazon a fán): {crr_eu_elapsed:.1f}s, "
              f"|CRR_eu - BS| medián {np.median(crr_eu_vs_bs):.5f}, max {crr_eu_vs_bs.max():.5f}")

    n_qa = min(n_qa_lsm, n)
    qa_idx = np.sort(rng.choice(n, size=n_qa, replace=False)) if n_qa > 0 else np.array([], dtype=int)

    lsm_put_price = np.full(n, np.nan, dtype=float)
    lsm_std_error = np.full(n, np.nan, dtype=float)
    lsm_n_steps_used = np.full(n, np.nan, dtype=float)
    is_qa_lsm = np.zeros(n, dtype=bool)
    is_qa_lsm[qa_idx] = True

    t_start_lsm = time.perf_counter()
    for j, idx in enumerate(qa_idx):
        idx = int(idx)
        n_steps_i = max(1, round(50.0 * T[idx]))
        try:
            price, se = lsm_american_price(
                S[idx], K[idx], T[idx], r[idx], sigma[idx],
                option_type='put',
                n_paths=lsm_n_paths,
                n_steps=n_steps_i,
                degree=lsm_degree,
                seed=seed + idx,
                antithetic=lsm_antithetic,
            )
        except Exception as exc:
            failed_rows.append((f'qa_lsm_{idx}', str(exc)))
            price, se = np.nan, np.nan
        lsm_put_price[idx] = price
        lsm_std_error[idx] = se
        lsm_n_steps_used[idx] = n_steps_i

        if verbose and n_qa > 0 and (j + 1) % max(1, n_qa // 20) == 0:
            elapsed = time.perf_counter() - t_start_lsm
            rate = (j + 1) / elapsed if elapsed > 0 else float('inf')
            eta = (n_qa - (j + 1)) / rate if rate > 0 else float('nan')
            print(f"  LSM QA: {j + 1:>6}/{n_qa} kész ({elapsed:7.1f}s eltelt, "
                  f"becsült hátralévő idő: {eta:7.1f}s)")

    lsm_elapsed = time.perf_counter() - t_start_lsm

    if verbose:
        print(f"  CRR összesen: {crr_elapsed:.1f}s ({crr_elapsed / n:.5f}s/sor átlag, n={n})")
        print(f"  LSM QA-alminta ({n_qa} sor): {lsm_elapsed:.1f}s"
              + (f" ({lsm_elapsed / n_qa:.4f}s/sor átlag)" if n_qa > 0 else ""))

    qa_agreement = None
    if n_qa > 0:
        crr_qa = american_put_price[qa_idx]
        lsm_qa = lsm_put_price[qa_idx]
        valid = ~np.isnan(lsm_qa)
        crr_v, lsm_v = crr_qa[valid], lsm_qa[valid]
        abs_diff = np.abs(crr_v - lsm_v)
        denom = np.maximum(np.abs(crr_v), 1e-8)
        rel_diff = abs_diff / denom
        within_tol = (rel_diff <= QA_REL_TOL) | (abs_diff <= QA_ABS_TOL)
        qa_agreement = {
            'n_qa_valid': int(valid.sum()),
            'n_within_tolerance': int(within_tol.sum()),
            'pct_within_tolerance': float(100.0 * within_tol.sum() / max(1, valid.sum())),
            'abs_diff_mean': float(np.mean(abs_diff)) if abs_diff.size else float('nan'),
            'abs_diff_median': float(np.median(abs_diff)) if abs_diff.size else float('nan'),
            'abs_diff_max': float(np.max(abs_diff)) if abs_diff.size else float('nan'),
            'rel_diff_median': float(np.median(rel_diff)) if rel_diff.size else float('nan'),
            'rel_diff_p75': float(np.percentile(rel_diff, 75)) if rel_diff.size else float('nan'),
            'rel_diff_max': float(np.max(rel_diff)) if rel_diff.size else float('nan'),
            'rel_tol': QA_REL_TOL,
            'abs_tol': QA_ABS_TOL,
        }
        if verbose:
            print(f"  QA egyezés: {qa_agreement['n_within_tolerance']}/{qa_agreement['n_qa_valid']} "
                  f"({qa_agreement['pct_within_tolerance']:.1f}%) a {QA_REL_TOL*100:.0f}% rel. VAGY "
                  f"{QA_ABS_TOL} abs. tolerancián belül")

    data = {
        'S': S,
        'K': K,
        'moneyness': moneyness,
        'T': T,
        'r': r,
        'sigma': sigma,
        'american_put_price': american_put_price,
        'european_put_price': european_put_price,
        'crr_european_put_price': crr_european_put_price,
        'early_exercise_premium': early_exercise_premium,
        'crr_n_steps': np.full(n, crr_n_steps, dtype=int),
        'lsm_put_price': lsm_put_price,
        'lsm_std_error': lsm_std_error,
        'lsm_n_steps': lsm_n_steps_used,
        'is_qa_lsm': is_qa_lsm,
    }

    for param in PARAM_NAMES:
        low, high = DEFAULT_PARAMS[param]
        data[f'{param}_norm'] = (data[param] - low) / (high - low)

    df = pd.DataFrame(data)

    meta = {
        'n': n,
        'seed': seed,
        'crr_n_steps': crr_n_steps,
        'crr_elapsed_s': crr_elapsed,
        'crr_s_per_row': crr_elapsed / n,
        'premium_method': 'crr_american - crr_european (azonos fa, control variate)',
        'crr_eu_elapsed_s': crr_eu_elapsed,
        'n_negative_premium': n_negative_premium,
        'crr_eu_vs_bs_abs_diff_median': float(np.median(crr_eu_vs_bs)),
        'crr_eu_vs_bs_abs_diff_max': float(crr_eu_vs_bs.max()),
        'n_qa_lsm': n_qa,
        'qa_idx': qa_idx.tolist(),
        'lsm_elapsed_s': lsm_elapsed,
        'lsm_s_per_row_qa': (lsm_elapsed / n_qa) if n_qa > 0 else 0.0,
        'lsm_n_paths': lsm_n_paths,
        'lsm_degree': lsm_degree,
        'lsm_antithetic': lsm_antithetic,
        'qa_agreement': qa_agreement,
        'failed_rows': failed_rows,
    }
    return df, meta


def save_dataset_american(
    df: pd.DataFrame,
    output_path: str,
    format: str = 'csv',
    seed: int = 42,
) -> None:
    os.makedirs(output_path, exist_ok=True)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    splits = {
        'train': df.iloc[:n_train],
        'val': df.iloc[n_train:n_train + n_val],
        'test': df.iloc[n_train + n_val:],
    }

    for split_name, split_df in splits.items():
        fname = f"{split_name}.{format}"
        fpath = os.path.join(output_path, fname)
        if format == 'csv':
            split_df.to_csv(fpath, index=False)
        elif format == 'parquet':
            split_df.to_parquet(fpath, index=False)
        else:
            raise ValueError(f"Ismeretlen formátum: '{format}'. Válasszon: csv, parquet")
        print(f"  Mentve: {fpath}  ({len(split_df):,} sor)")


if __name__ == '__main__':
    N_TOTAL = int(os.environ.get('N_SAMPLES', 3000))
    N_QA_LSM = int(os.environ.get('N_QA_LSM', 2000))
    SEED = 42
    OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

    print(f"Amerikai put adathalmaz generálása: n={N_TOTAL}, seed={SEED}")
    print("CRR (elsődleges, bulk) paraméterek: n_steps=500")
    print(f"LSM (QA, fix alminta) paraméterek: n_paths=100_000, n_steps=round(50*T), "
          f"degree=3, antithetic=True, alminta méret={N_QA_LSM}")
    print()

    t0 = time.perf_counter()
    df, meta = generate_dataset_american(n=N_TOTAL, seed=SEED, n_qa_lsm=N_QA_LSM)
    total_elapsed = time.perf_counter() - t0

    print()
    print(f"Kész. Teljes futásidő: {total_elapsed:.1f}s")
    print(f"CRR összesen: {meta['crr_elapsed_s']:.1f}s ({meta['crr_s_per_row']:.5f}s/sor átlag)")
    print(f"LSM QA futásidő: {meta['lsm_elapsed_s']:.1f}s ({meta['n_qa_lsm']} sor)")
    print(f"Negatív fa-alapú prémium: {meta['n_negative_premium']} sor; "
          f"|CRR_eu - BS| max. {meta['crr_eu_vs_bs_abs_diff_max']:.6f}")
    print(f"Sikertelen sorok: {len(meta['failed_rows'])}")
    if meta['qa_agreement'] is not None:
        qa = meta['qa_agreement']
        print(f"QA egyezés (CRR vs LSM): {qa['n_within_tolerance']}/{qa['n_qa_valid']} "
              f"({qa['pct_within_tolerance']:.1f}%) tolerancián belül")

    save_dataset_american(df, OUT_DIR, format='csv', seed=SEED)
    save_dataset_american(df, OUT_DIR, format='parquet', seed=SEED)

    meta['total_elapsed_s'] = total_elapsed
    meta_path = os.path.join(OUT_DIR, 'generation_meta.json')
    import json
    meta_serializable = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in meta.items()
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_serializable, f, indent=2, default=str)
    print(f"  Mentve: {meta_path}")
