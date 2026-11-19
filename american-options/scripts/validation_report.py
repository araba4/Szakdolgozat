import itertools
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from src.american_pricers import crr_american_price, lsm_american_price

TABLE1 = [
    (36, 0.20, 1, 4.478),
    (36, 0.20, 2, 4.840),
    (36, 0.40, 1, 7.101),
    (36, 0.40, 2, 8.508),
    (38, 0.20, 1, 3.250),
    (38, 0.20, 2, 3.745),
    (38, 0.40, 1, 6.148),
    (38, 0.40, 2, 7.670),
    (40, 0.20, 1, 2.314),
    (40, 0.20, 2, 2.885),
    (40, 0.40, 1, 5.312),
    (40, 0.40, 2, 6.920),
    (42, 0.20, 1, 1.617),
    (42, 0.20, 2, 2.212),
    (42, 0.40, 1, 4.582),
    (42, 0.40, 2, 6.248),
    (44, 0.20, 1, 1.110),
    (44, 0.20, 2, 1.690),
    (44, 0.40, 1, 3.948),
    (44, 0.40, 2, 5.647),
]
K_TABLE1 = 40.0
R_TABLE1 = 0.06


def section1_table1_reproduction():
    print("\n" + "=" * 78)
    print("1) Longstaff-Schwartz (2001) Table 1 reprodukcio (K=40, r=.06)")
    print("=" * 78)
    rows = []
    header = f"{'S':>4} {'sigma':>6} {'T':>3} {'published':>10} {'LSM (se)':>16} {'rel.err':>9}"
    print(header)
    for S, sigma, T, published in TABLE1:
        n_steps = int(round(50 * T))
        price, se = lsm_american_price(
            S, K_TABLE1, T, R_TABLE1, sigma, option_type="put",
            n_paths=100_000, n_steps=n_steps, degree=3, seed=42, antithetic=True,
        )
        rel_err = abs(price - published) / published
        print(f"{S:>4} {sigma:>6.2f} {T:>3} {published:>10.3f} "
              f"{price:>10.3f} ({se:.3f}) {rel_err * 100:>7.3f}%")
        rows.append({
            "S": S, "sigma": sigma, "T": T, "published": published,
            "lsm_price": price, "lsm_se": se, "rel_err": rel_err,
        })
    max_err = max(r["rel_err"] for r in rows)
    mean_err = float(np.mean([r["rel_err"] for r in rows]))
    n_within_2pct = sum(1 for r in rows if r["rel_err"] <= 0.02)
    print(f"\nmax rel.err = {max_err*100:.3f}%  mean rel.err = {mean_err*100:.3f}%  "
          f"within 2%: {n_within_2pct}/{len(rows)}")
    return rows


DEFAULT_PARAMS = {
    "S": (10.0, 150.0),
    "moneyness": (0.5, 1.5),
    "T": (0.005, 2.0),
    "r": (0.00, 0.05),
    "sigma": (0.05, 0.90),
}


def section2_cross_validation(n_grid=40, seed=7):
    print("\n" + "=" * 78)
    print("2) LSM vs CRR kereszt-validacio a teljes parameterteren "
          f"(n={n_grid} LHS-szeru mintat)")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    S = rng.uniform(*DEFAULT_PARAMS["S"], n_grid)
    moneyness = rng.uniform(*DEFAULT_PARAMS["moneyness"], n_grid)
    T = rng.uniform(*DEFAULT_PARAMS["T"], n_grid)
    r = rng.uniform(*DEFAULT_PARAMS["r"], n_grid)
    sigma = rng.uniform(*DEFAULT_PARAMS["sigma"], n_grid)
    K = S / moneyness

    diffs = []
    rows = []
    t0 = time.time()
    for i in range(n_grid):
        crr_price = crr_american_price(S[i], K[i], T[i], r[i], sigma[i],
                                        option_type="put", n_steps=1000)
        lsm_price, lsm_se = lsm_american_price(
            S[i], K[i], T[i], r[i], sigma[i], option_type="put",
            n_paths=100_000, n_steps=max(10, int(round(50 * T[i]))),
            degree=3, seed=100 + i, antithetic=True,
        )
        denom = max(crr_price, 1e-6)
        rel_diff = abs(lsm_price - crr_price) / denom
        rows.append({
            "S": S[i], "K": K[i], "T": T[i], "r": r[i], "sigma": sigma[i],
            "crr_price": crr_price, "lsm_price": lsm_price, "lsm_se": lsm_se,
            "rel_diff": rel_diff,
        })
        diffs.append(rel_diff)
    elapsed = time.time() - t0

    diffs = np.array(diffs)
    print(f"{'#':>3} {'S':>7} {'K':>7} {'T':>5} {'r':>6} {'sigma':>6} "
          f"{'CRR':>9} {'LSM':>9} {'rel.diff':>9}")
    for i, row in enumerate(rows):
        print(f"{i:>3} {row['S']:>7.2f} {row['K']:>7.2f} {row['T']:>5.3f} "
              f"{row['r']:>6.3f} {row['sigma']:>6.3f} {row['crr_price']:>9.4f} "
              f"{row['lsm_price']:>9.4f} {row['rel_diff']*100:>8.3f}%")

    print(f"\nmax rel.diff = {diffs.max()*100:.3f}%  mean rel.diff = {diffs.mean()*100:.3f}%  "
          f"median = {np.median(diffs)*100:.3f}%  (elapsed {elapsed:.1f}s for {n_grid} pairs)")
    return rows


REP_CASES = [
    ("ATM, sigma=.2, T=1", 40, 40, 0.06, 0.20, 1),
    ("ITM put, sigma=.4, T=1", 44, 40, 0.06, 0.40, 1),
    ("OTM put, sigma=.2, T=2", 36, 40, 0.06, 0.20, 2),
]


def section3_convergence():
    print("\n" + "=" * 78)
    print("3a) LSM konvergencia n_paths fuggvenyeben (n_steps=50*T, degree=3)")
    print("=" * 78)
    n_paths_grid = [1_000, 10_000, 50_000, 100_000, 500_000]
    lsm_conv = {}
    for label, S, K, r, sigma, T in REP_CASES:
        print(f"\n  {label}  (S={S}, K={K}, r={r}, sigma={sigma}, T={T})")
        rows = []
        for n_paths in n_paths_grid:
            t0 = time.time()
            price, se = lsm_american_price(
                S, K, T, r, sigma, option_type="put", n_paths=n_paths,
                n_steps=int(round(50 * T)), degree=3, seed=42, antithetic=True,
            )
            dt = time.time() - t0
            print(f"    n_paths={n_paths:>7}: price={price:.4f}  se={se:.4f}  "
                  f"time={dt:.2f}s")
            rows.append({"n_paths": n_paths, "price": price, "se": se, "time_s": dt})
        lsm_conv[label] = rows

    print("\n" + "=" * 78)
    print("3b) CRR konvergencia n_steps fuggvenyeben")
    print("=" * 78)
    n_steps_grid = [50, 100, 500, 1000]
    crr_conv = {}
    for label, S, K, r, sigma, T in REP_CASES:
        print(f"\n  {label}  (S={S}, K={K}, r={r}, sigma={sigma}, T={T})")
        rows = []
        for n_steps in n_steps_grid:
            t0 = time.time()
            price = crr_american_price(S, K, T, r, sigma, option_type="put", n_steps=n_steps)
            dt = time.time() - t0
            print(f"    n_steps={n_steps:>5}: price={price:.4f}  time={dt:.4f}s")
            rows.append({"n_steps": n_steps, "price": price, "time_s": dt})
        crr_conv[label] = rows

    return lsm_conv, crr_conv


if __name__ == "__main__":
    t1_rows = section1_table1_reproduction()
    t2_rows = section2_cross_validation()
    lsm_conv, crr_conv = section3_convergence()

    with open("scripts/validation_output.json", "w", encoding="utf-8") as f:
        json.dump({
            "table1_reproduction": t1_rows,
            "cross_validation": t2_rows,
            "lsm_convergence": lsm_conv,
            "crr_convergence": crr_conv,
        }, f, indent=2)
    print("\nSaved scripts/validation_output.json")
