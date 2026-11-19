import numpy as np


def crr_american_price(S, K, T, r, sigma, option_type="put", n_steps=500):
    if option_type not in ("call", "put"):
        raise ValueError(f"Ismeretlen option_type: '{option_type}'. Válasszon: call, put")

    S, K, T, r, sigma = (float(x) for x in (S, K, T, r, sigma))

    if T < 1e-8:
        return float(max(K - S, 0.0)) if option_type == "put" else float(max(S - K, 0.0))

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    j = np.arange(n_steps + 1)
    S_T = S * (u ** j) * (d ** (n_steps - j))

    if option_type == "put":
        values = np.maximum(K - S_T, 0.0)
    else:
        values = np.maximum(S_T - K, 0.0)

    for step in range(n_steps - 1, -1, -1):
        continuation = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        j = np.arange(step + 1)
        S_t = S * (u ** j) * (d ** (step - j))
        if option_type == "put":
            intrinsic = np.maximum(K - S_t, 0.0)
        else:
            intrinsic = np.maximum(S_t - K, 0.0)
        values = np.maximum(continuation, intrinsic)

    return float(values[0])


def crr_american_price_batch(S, K, T, r, sigma, option_type="put", n_steps=500):
    S, K, T, r, sigma = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (S, K, T, r, sigma))
    S, K, T, r, sigma = np.broadcast_arrays(S, K, T, r, sigma)
    out = np.empty(S.shape, dtype=float)
    flat_S, flat_K, flat_T, flat_r, flat_sigma = (a.ravel() for a in (S, K, T, r, sigma))
    flat_out = out.ravel()
    for i in range(flat_S.size):
        flat_out[i] = crr_american_price(
            flat_S[i], flat_K[i], flat_T[i], flat_r[i], flat_sigma[i],
            option_type=option_type, n_steps=n_steps,
        )
    return out.reshape(S.shape)


def crr_european_price_batch(S, K, T, r, sigma, option_type="put", n_steps=500, chunk_size=20_000):
    if option_type not in ("call", "put"):
        raise ValueError(f"Ismeretlen option_type: '{option_type}'. Válasszon: call, put")

    S, K, T, r, sigma = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (S, K, T, r, sigma))
    S, K, T, r, sigma = np.broadcast_arrays(S, K, T, r, sigma)
    shape = S.shape
    S, K, T, r, sigma = (a.ravel() for a in (S, K, T, r, sigma))
    out = np.empty(S.size, dtype=float)

    near_expiry = T < 1e-8
    if option_type == "put":
        out[near_expiry] = np.maximum(K[near_expiry] - S[near_expiry], 0.0)
    else:
        out[near_expiry] = np.maximum(S[near_expiry] - K[near_expiry], 0.0)

    idx_all = np.flatnonzero(~near_expiry)
    j = np.arange(n_steps + 1)
    for start in range(0, idx_all.size, chunk_size):
        idx = idx_all[start:start + chunk_size]
        s, k, t, rr, sig = S[idx, None], K[idx, None], T[idx, None], r[idx, None], sigma[idx, None]

        dt = t / n_steps
        u = np.exp(sig * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-rr * dt)
        p = (np.exp(rr * dt) - d) / (u - d)

        S_T = s * (u ** j) * (d ** (n_steps - j))
        if option_type == "put":
            values = np.maximum(k - S_T, 0.0)
        else:
            values = np.maximum(S_T - k, 0.0)

        for _ in range(n_steps):
            values = disc * (p * values[:, 1:] + (1.0 - p) * values[:, :-1])

        out[idx] = values[:, 0]

    return out.reshape(shape)


def lsm_american_price(
    S, K, T, r, sigma,
    option_type="put",
    n_paths=100_000,
    n_steps=50,
    degree=2,
    seed=42,
    antithetic=True,
):
    if option_type not in ("call", "put"):
        raise ValueError(f"Ismeretlen option_type: '{option_type}'. Válasszon: call, put")

    S, K, T, r, sigma = (float(x) for x in (S, K, T, r, sigma))

    if T < 1e-8:
        intrinsic = max(K - S, 0.0) if option_type == "put" else max(S - K, 0.0)
        return float(intrinsic), 0.0

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    disc = np.exp(-r * dt)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Z = np.vstack([Z, -Z])
    else:
        Z = rng.standard_normal((n_paths, n_steps))
    n_eff = Z.shape[0]

    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(drift + diffusion, axis=1)
    S_paths = S * np.exp(log_paths)

    def payoff(s):
        return np.maximum(K - s, 0.0) if option_type == "put" else np.maximum(s - K, 0.0)

    cashflow = payoff(S_paths[:, -1])
    exercise_step = np.full(n_eff, n_steps - 1, dtype=int)

    for step in range(n_steps - 2, -1, -1):
        St = S_paths[:, step]
        intrinsic = payoff(St)
        itm = intrinsic > 0.0
        n_itm = int(np.count_nonzero(itm))
        if n_itm <= degree + 1:
            continue

        X = St[itm]
        discount_power = (exercise_step[itm] - step).astype(float)
        Y = cashflow[itm] * disc ** discount_power

        basis = np.vander(X, degree + 1, increasing=True)
        coeffs, *_ = np.linalg.lstsq(basis, Y, rcond=None)
        continuation_value = basis @ coeffs

        exercise_now = intrinsic[itm] > continuation_value
        idx_itm = np.where(itm)[0]
        idx_exercise = idx_itm[exercise_now]
        cashflow[idx_exercise] = intrinsic[itm][exercise_now]
        exercise_step[idx_exercise] = step

    discounted_cf = cashflow * disc ** (exercise_step + 1).astype(float)
    price = float(np.mean(discounted_cf))

    if antithetic:
        half = n_eff // 2
        pair_means = 0.5 * (discounted_cf[:half] + discounted_cf[half:])
        se = float(np.std(pair_means, ddof=1) / np.sqrt(half)) if half > 1 else 0.0
    else:
        se = float(np.std(discounted_cf, ddof=1) / np.sqrt(n_eff)) if n_eff > 1 else 0.0

    return price, se


def lsm_american_price_batch(
    S, K, T, r, sigma,
    option_type="put",
    n_paths=100_000,
    n_steps=50,
    degree=2,
    seed=42,
    antithetic=True,
):
    S, K, T, r, sigma = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (S, K, T, r, sigma))
    S, K, T, r, sigma = np.broadcast_arrays(S, K, T, r, sigma)
    shape = S.shape
    flat_S, flat_K, flat_T, flat_r, flat_sigma = (a.ravel() for a in (S, K, T, r, sigma))
    n = flat_S.size
    prices = np.empty(n, dtype=float)
    std_errors = np.empty(n, dtype=float)
    for i in range(n):
        prices[i], std_errors[i] = lsm_american_price(
            flat_S[i], flat_K[i], flat_T[i], flat_r[i], flat_sigma[i],
            option_type=option_type, n_paths=n_paths, n_steps=n_steps,
            degree=degree, seed=seed + i, antithetic=antithetic,
        )
    return prices.reshape(shape), std_errors.reshape(shape)
