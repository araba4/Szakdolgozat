import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.american_pricers import (
    crr_american_price,
    crr_american_price_batch,
    crr_european_price_batch,
    lsm_american_price,
    lsm_american_price_batch,
)


class TestBoundaryBehaviour:
    def test_crr_put_zero_maturity_equals_intrinsic(self):
        price = crr_american_price(35, 40, 1e-10, 0.06, 0.2, option_type="put", n_steps=200)
        assert price == pytest.approx(5.0, abs=1e-6)

    def test_crr_call_zero_maturity_equals_intrinsic(self):
        price = crr_american_price(45, 40, 1e-10, 0.06, 0.2, option_type="call", n_steps=200)
        assert price == pytest.approx(5.0, abs=1e-6)

    def test_lsm_put_zero_maturity_equals_intrinsic(self):
        price, se = lsm_american_price(35, 40, 1e-10, 0.06, 0.2, option_type="put")
        assert price == pytest.approx(5.0, abs=1e-6)
        assert se == 0.0

    def test_crr_deep_itm_put_near_intrinsic(self):
        price = crr_american_price(10, 40, 0.5, 0.06, 0.1, option_type="put", n_steps=500)
        intrinsic = 40 - 10
        assert price >= intrinsic - 1e-8
        assert price == pytest.approx(intrinsic, rel=0.02)

    def test_crr_deep_otm_put_near_zero(self):
        price = crr_american_price(150, 40, 0.5, 0.06, 0.2, option_type="put", n_steps=500)
        assert 0.0 <= price < 0.01

    def test_crr_deep_otm_call_near_zero(self):
        price = crr_american_price(10, 100, 0.5, 0.06, 0.2, option_type="call", n_steps=500)
        assert 0.0 <= price < 0.01

    def test_american_put_never_below_intrinsic(self):
        for S in (20, 35, 40, 45, 60):
            price = crr_american_price(S, 40, 1.0, 0.06, 0.3, option_type="put", n_steps=500)
            assert price >= max(40 - S, 0.0) - 1e-8


class TestMonotonicity:
    def test_crr_put_price_increasing_in_sigma(self):
        sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
        prices = [crr_american_price(40, 40, 1.0, 0.06, s, option_type="put", n_steps=500)
                  for s in sigmas]
        assert all(p2 >= p1 - 1e-9 for p1, p2 in zip(prices, prices[1:]))

    def test_crr_call_price_increasing_in_sigma(self):
        sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
        prices = [crr_american_price(40, 40, 1.0, 0.06, s, option_type="call", n_steps=500)
                  for s in sigmas]
        assert all(p2 >= p1 - 1e-9 for p1, p2 in zip(prices, prices[1:]))

    def test_crr_put_price_decreasing_in_S(self):
        S_values = [20, 30, 40, 50, 60]
        prices = [crr_american_price(S, 40, 1.0, 0.06, 0.3, option_type="put", n_steps=500)
                  for S in S_values]
        assert all(p2 <= p1 + 1e-9 for p1, p2 in zip(prices, prices[1:]))

    def test_lsm_put_price_increasing_in_sigma(self):
        sigmas = [0.1, 0.3, 0.5]
        prices = [lsm_american_price(40, 40, 1.0, 0.06, s, option_type="put",
                                      n_paths=50_000, n_steps=50, degree=3, seed=1)[0]
                  for s in sigmas]
        assert prices[0] < prices[-1]
        assert prices[1] < prices[-1] + 0.05


class TestLsmCrrAgreement:
    @pytest.mark.parametrize("S,K,T,r,sigma", [
        (36, 40, 1.0, 0.06, 0.20),
        (40, 40, 1.0, 0.06, 0.30),
        (44, 40, 2.0, 0.06, 0.40),
        (30, 40, 0.5, 0.03, 0.25),
    ])
    def test_lsm_matches_crr_within_tolerance(self, S, K, T, r, sigma):
        crr_price = crr_american_price(S, K, T, r, sigma, option_type="put", n_steps=1000)
        lsm_price, lsm_se = lsm_american_price(
            S, K, T, r, sigma, option_type="put",
            n_paths=100_000, n_steps=int(round(50 * T)), degree=3, seed=42,
        )
        rel_diff = abs(lsm_price - crr_price) / max(crr_price, 1e-6)
        assert (rel_diff <= 0.02) or (abs(lsm_price - crr_price) <= 0.05)

    def test_batch_wrappers_match_scalar_versions(self):
        S = np.array([36.0, 40.0, 44.0])
        K = np.array([40.0, 40.0, 40.0])
        T = np.array([1.0, 1.0, 1.0])
        r = np.array([0.06, 0.06, 0.06])
        sigma = np.array([0.2, 0.2, 0.2])

        crr_batch = crr_american_price_batch(S, K, T, r, sigma, option_type="put", n_steps=500)
        crr_scalar = np.array([
            crr_american_price(s, k, t, rr, sig, option_type="put", n_steps=500)
            for s, k, t, rr, sig in zip(S, K, T, r, sigma)
        ])
        np.testing.assert_allclose(crr_batch, crr_scalar)

        lsm_batch_prices, lsm_batch_se = lsm_american_price_batch(
            S, K, T, r, sigma, option_type="put", n_paths=20_000, n_steps=50, seed=5,
        )
        assert lsm_batch_prices.shape == S.shape
        assert lsm_batch_se.shape == S.shape
        assert np.all(lsm_batch_se >= 0)


class TestLongstaffSchwartzTable1:
    @pytest.mark.parametrize("S,sigma,T,published", [
        (36, 0.20, 1, 4.478),
        (36, 0.40, 2, 8.508),
        (40, 0.20, 1, 2.314),
        (42, 0.40, 1, 4.582),
        (44, 0.20, 2, 1.690),
    ])
    def test_table1_case_within_tolerance(self, S, sigma, T, published):
        price, se = lsm_american_price(
            S, 40, T, 0.06, sigma, option_type="put",
            n_paths=100_000, n_steps=int(round(50 * T)), degree=3, seed=42,
        )
        rel_err = abs(price - published) / published
        assert rel_err <= 0.02, (
            f"S={S} sigma={sigma} T={T}: published={published} got={price:.4f} "
            f"rel_err={rel_err:.4%}"
        )


def _bs_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


class TestCrrEuropean:
    @pytest.mark.parametrize("S,K,T,r,sigma", [
        (36, 40, 1.0, 0.06, 0.20),
        (40, 40, 0.5, 0.03, 0.40),
        (120, 90, 0.05, 0.01, 0.15),
        (20, 35, 2.0, 0.05, 0.80),
    ])
    def test_converges_to_black_scholes(self, S, K, T, r, sigma):
        price = crr_european_price_batch(S, K, T, r, sigma, option_type="put", n_steps=1000)[0]
        assert price == pytest.approx(_bs_put(S, K, T, r, sigma), abs=0.01)

    def test_tree_premium_never_negative(self):
        rng = np.random.default_rng(0)
        n = 200
        S = rng.uniform(10, 150, n)
        moneyness = np.concatenate([rng.uniform(0.5, 1.5, n // 2), rng.uniform(1.2, 1.5, n // 2)])
        T = np.concatenate([rng.uniform(0.005, 2.0, n // 2), rng.uniform(0.005, 0.2, n // 2)])
        r = np.concatenate([rng.uniform(0.0, 0.05, n // 2), rng.uniform(0.0, 0.01, n // 2)])
        sigma = rng.uniform(0.05, 0.9, n)
        K = S / moneyness

        american = crr_american_price_batch(S, K, T, r, sigma, option_type="put", n_steps=200)
        european = crr_european_price_batch(S, K, T, r, sigma, option_type="put", n_steps=200)
        assert np.all(american - european >= 0.0)

    def test_call_has_no_premium_on_tree(self):
        S, K, T, r, sigma = 44.0, 40.0, 1.0, 0.06, 0.3
        american = crr_american_price(S, K, T, r, sigma, option_type="call", n_steps=300)
        european = crr_european_price_batch(S, K, T, r, sigma, option_type="call", n_steps=300)[0]
        assert american == pytest.approx(european, abs=1e-10)

    def test_zero_maturity_equals_intrinsic(self):
        price = crr_european_price_batch(35, 40, 1e-10, 0.06, 0.2, option_type="put")[0]
        assert price == pytest.approx(5.0, abs=1e-12)

    def test_shape_and_chunking_invariance(self):
        rng = np.random.default_rng(1)
        S = rng.uniform(10, 150, (3, 4))
        K = S / rng.uniform(0.5, 1.5, (3, 4))
        T = rng.uniform(0.005, 2.0, (3, 4))
        whole = crr_european_price_batch(S, K, T, 0.03, 0.3, n_steps=100)
        chunked = crr_european_price_batch(S, K, T, 0.03, 0.3, n_steps=100, chunk_size=5)
        assert whole.shape == (3, 4)
        np.testing.assert_array_equal(whole, chunked)
