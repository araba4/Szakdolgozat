import sys

sys.dont_write_bytecode = True

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from american_pricers import crr_american_price_batch, crr_european_price_batch

N_REF = 5000
N_PER_GROUP = 200
OUT_JSON = ROOT / 'scripts' / 'premium_accuracy_output.json'


def error_stats(err):
    err = np.asarray(err)
    return {'abs_median': float(np.median(np.abs(err))), 'abs_p95': float(np.quantile(np.abs(err), 0.95)),
            'abs_max': float(np.abs(err).max()), 'mean': float(err.mean())}


def main():
    df = pd.read_parquet(ROOT / 'data' / 'test.parquet')
    if 'crr_european_put_price' not in df.columns:
        sys.exit("Előbb az add_crr_european_premium.py-t kell lefuttatni.")
    raw_v1 = df['american_put_price'] - df['european_put_price']
    rng = np.random.default_rng(0)
    groups = {
        'veletlen': df.iloc[rng.choice(len(df), N_PER_GROUP, replace=False)],
        'v1_klippelt': df[raw_v1 < 0].iloc[rng.choice(int((raw_v1 < 0).sum()), N_PER_GROUP, replace=False)],
    }

    results = {'n_ref': N_REF}
    for name, g in groups.items():
        args = [g[c].to_numpy() for c in ['S', 'K', 'T', 'r', 'sigma']]
        t0 = time.perf_counter()
        am_ref = crr_american_price_batch(*args, option_type='put', n_steps=N_REF)
        eu_ref = crr_european_price_batch(*args, option_type='put', n_steps=N_REF, chunk_size=50)
        print(f"  {name}: {N_REF} lépéses referencia, {time.perf_counter() - t0:.1f}s")

        ref_tree = am_ref - eu_ref
        ref_bs = am_ref - g['european_put_price'].to_numpy()
        v1 = np.maximum(g['american_put_price'] - g['european_put_price'], 0.0).to_numpy()
        v2 = g['early_exercise_premium'].to_numpy()
        raw_v1_g = (g['american_put_price'] - g['european_put_price']).to_numpy()

        results[name] = {
            'n': len(g),
            'premium_ref_median': float(np.median(ref_tree)),
            'ref_tree_minus_ref_bs': error_stats(ref_tree - ref_bs),
            'n_negative_ref_bs': int((ref_bs < 0).sum()),
            'n_negative_ref_tree': int((ref_tree < 0).sum()),
            'n_negative_v1_raw': int((raw_v1_g < 0).sum()),
            'n_negative_v2': int((v2 < 0).sum()),
            'v1_err_vs_ref_tree': error_stats(v1 - ref_tree),
            'v2_err_vs_ref_tree': error_stats(v2 - ref_tree),
            'v1_err_vs_ref_bs': error_stats(v1 - ref_bs),
            'v2_err_vs_ref_bs': error_stats(v2 - ref_bs),
            'v2_more_accurate_share_vs_ref_tree': float(np.mean(np.abs(v2 - ref_tree) < np.abs(v1 - ref_tree))),
        }

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  Mentve: {OUT_JSON}")


if __name__ == '__main__':
    main()
