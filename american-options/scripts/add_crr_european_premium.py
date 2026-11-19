import sys

sys.dont_write_bytecode = True

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from american_pricers import crr_european_price_batch

DATA_DIR = ROOT / 'data'
SPLITS = ['train', 'val', 'test']
OUT_JSON = ROOT / 'scripts' / 'premium_migration_output.json'


def premium_diagnostics(df, col):
    p = df[col]
    out = {'mean': p.mean(), 'median': p.median(), 'std': p.std(), 'max': p.max(),
           'n_zero': int((p == 0.0).sum()), 'corr_T': p.corr(df['T']), 'corr_sigma': p.corr(df['sigma'])}
    for name in ['T', 'moneyness']:
        terc = pd.qcut(df[name], 3, labels=['alacsony', 'kozepes', 'magas'])
        out[f'mean_by_{name}_tercile'] = p.groupby(terc, observed=True).mean().to_dict()
    terc_m = pd.qcut(df['moneyness'], 3, labels=['ITM', 'ATM', 'OTM'])
    out['corr_sigma_by_moneyness'] = {k: g[col].corr(g['sigma']) for k, g in df.groupby(terc_m, observed=True)}
    return out


def main():
    meta_path = DATA_DIR / 'generation_meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    n_steps = int(meta['crr_n_steps'])
    if 'crr_european_put_price' in pd.read_parquet(DATA_DIR / 'train.parquet', columns=None).columns:
        sys.exit("Az adathalmaz már tartalmazza a crr_european_put_price oszlopot - nincs teendő.")

    frames = {s: pd.read_parquet(DATA_DIR / f'{s}.parquet') for s in SPLITS}
    stats = {'n_steps': n_steps, 'splits': {}}

    t0 = time.perf_counter()
    for split, df in frames.items():
        raw_bs = df['american_put_price'] - df['european_put_price']
        assert np.allclose(df['early_exercise_premium'], np.maximum(raw_bs, 0.0), rtol=0, atol=1e-12)

        t_split = time.perf_counter()
        crr_eu = crr_european_price_batch(
            df['S'].to_numpy(), df['K'].to_numpy(), df['T'].to_numpy(),
            df['r'].to_numpy(), df['sigma'].to_numpy(), option_type='put', n_steps=n_steps,
        )
        print(f"  {split}: CRR európai {len(df):,} sor, {time.perf_counter() - t_split:.1f}s")

        new_premium = df['american_put_price'].to_numpy() - crr_eu
        n_neg = int((new_premium < 0).sum())
        assert n_neg == 0, f"{split}: {n_neg} negatív fa-alapú prémium"

        old = df.copy()
        old['premium_raw_bs'] = raw_bs
        df.insert(df.columns.get_loc('european_put_price') + 1, 'crr_european_put_price', crr_eu)
        df['early_exercise_premium'] = new_premium
        frames[split] = df

        diff = df['early_exercise_premium'] - old['early_exercise_premium']
        stats['splits'][split] = {
            'n': len(df),
            'n_clipped_v1': int((raw_bs < 0).sum()),
            'premium_v1': premium_diagnostics(old, 'early_exercise_premium'),
            'premium_v2': premium_diagnostics(df, 'early_exercise_premium'),
            'v2_minus_v1_abs_median': float(diff.abs().median()),
            'v2_minus_v1_abs_max': float(diff.abs().max()),
            'crr_eu_vs_bs_abs_median': float((df['crr_european_put_price'] - df['european_put_price']).abs().median()),
            'crr_eu_vs_bs_abs_p99': float((df['crr_european_put_price'] - df['european_put_price']).abs().quantile(0.99)),
            'crr_eu_vs_bs_abs_max': float((df['crr_european_put_price'] - df['european_put_price']).abs().max()),
            'crr_eu_minus_bs_mean': float((df['crr_european_put_price'] - df['european_put_price']).mean()),
        }
    crr_eu_elapsed = time.perf_counter() - t0

    full_old_raw = pd.concat([f['american_put_price'] - f['european_put_price'] for f in frames.values()])
    full = pd.concat(frames.values(), ignore_index=True)
    full_old = full.assign(early_exercise_premium=np.maximum(full_old_raw.to_numpy(), 0.0))
    stats['full'] = {
        'premium_v1': premium_diagnostics(full_old, 'early_exercise_premium'),
        'premium_v2': premium_diagnostics(full, 'early_exercise_premium'),
        'crr_eu_elapsed_s': crr_eu_elapsed,
    }
    eu_diff = (full['crr_european_put_price'] - full['european_put_price']).abs()
    was_clipped = full_old_raw.to_numpy() < 0
    stats['full'].update({
        'n_clipped_v1': int(was_clipped.sum()),
        'v2_premium_on_v1_clipped_rows_median': float(full.loc[was_clipped, 'early_exercise_premium'].median()),
        'v2_premium_on_v1_clipped_rows_max': float(full.loc[was_clipped, 'early_exercise_premium'].max()),
        'crr_eu_vs_bs_abs_median': float(eu_diff.median()),
        'crr_eu_vs_bs_abs_p99': float(eu_diff.quantile(0.99)),
        'crr_eu_vs_bs_abs_max': float(eu_diff.max()),
    })

    for split, df in frames.items():
        df.to_parquet(DATA_DIR / f'{split}.parquet', index=False)
        df.to_csv(DATA_DIR / f'{split}.csv', index=False)
        print(f"  Mentve: {split}.parquet/.csv ({len(df):,} sor)")

    meta['premium_v1'] = {
        'method': 'max(crr_american - bs_european, 0)',
        'n_clipped': meta.pop('n_clipped'),
        'max_clip_violation': meta.pop('max_clip_violation'),
        'note': 'Felváltotta a fa-alapú prémium.',
    }
    meta.update({
        'premium_method': 'crr_american - crr_european (azonos fa, control variate)',
        'crr_eu_elapsed_s': crr_eu_elapsed,
        'n_negative_premium': 0,
        'crr_eu_vs_bs_abs_diff_median': stats['full']['crr_eu_vs_bs_abs_median'],
        'crr_eu_vs_bs_abs_diff_max': stats['full']['crr_eu_vs_bs_abs_max'],
        'premium_migration': {
            'script': 'american-options/scripts/add_crr_european_premium.py',
            'date': date.today().isoformat(),
        },
    })
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    print(f"  Mentve: {meta_path}")

    OUT_JSON.write_text(json.dumps(stats, indent=2, default=float, ensure_ascii=False), encoding='utf-8')
    print(f"  Mentve: {OUT_JSON}")
    print(json.dumps(stats['full'], indent=2, default=float, ensure_ascii=False))


if __name__ == '__main__':
    main()
