import numpy as np
import pytest

from src.black_scholes import (
    bs_call, bs_put, bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho,
)


ATM = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
ITM = dict(S=110.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
OTM = dict(S=90.0,  K=100.0, T=1.0, r=0.05, sigma=0.2)


class TestCallPrice:
    def test_atm_ismert_ertek(self):
        assert bs_call(**ATM) == pytest.approx(10.4506, rel=1e-3)

    def test_itm_nagyobb_mint_otm(self):
        assert bs_call(**ITM) > bs_call(**ATM) > bs_call(**OTM)

    def test_also_korlat_intrinsic(self):
        c = bs_call(**ATM)
        intrinsic = max(ATM['S'] - ATM['K'] * np.exp(-ATM['r'] * ATM['T']), 0)
        assert c >= intrinsic

    def test_felso_korlat(self):
        assert bs_call(**ATM) <= ATM['S']

    def test_nem_negativ(self):
        assert bs_call(**ATM) >= 0
        assert bs_call(**OTM) >= 0

    def test_lejarat_koezeleben_intrinsic(self):
        assert bs_call(S=110, K=100, T=1e-10, r=0.05, sigma=0.2) == pytest.approx(10.0, abs=1e-4)
        assert bs_call(S=90,  K=100, T=1e-10, r=0.05, sigma=0.2) == pytest.approx(0.0,  abs=1e-4)

    def test_vektorizalt(self):
        S = np.array([90.0, 100.0, 110.0])
        prices = bs_call(S, K=100, T=1, r=0.05, sigma=0.2)
        assert prices.shape == (3,)
        assert np.all(np.diff(prices) > 0)


class TestPutPrice:
    def test_nem_negativ(self):
        assert bs_put(**ATM) >= 0
        assert bs_put(**ITM) >= 0

    def test_lejarat_koezeleben_intrinsic(self):
        assert bs_put(S=90,  K=100, T=1e-10, r=0.05, sigma=0.2) == pytest.approx(10.0, abs=1e-4)
        assert bs_put(S=110, K=100, T=1e-10, r=0.05, sigma=0.2) == pytest.approx(0.0,  abs=1e-4)


class TestPutCallParitas:

    def _ellenorzes(self, params):
        C = bs_call(**params)
        P = bs_put(**params)
        bal  = C - P
        jobb = params['S'] - params['K'] * np.exp(-params['r'] * params['T'])
        assert bal == pytest.approx(jobb, rel=1e-6)

    def test_atm(self): self._ellenorzes(ATM)
    def test_itm(self): self._ellenorzes(ITM)
    def test_otm(self): self._ellenorzes(OTM)

    def test_vektorizalt(self):
        S = np.linspace(70, 130, 50)
        C = bs_call(S, K=100, T=1, r=0.05, sigma=0.2)
        P = bs_put( S, K=100, T=1, r=0.05, sigma=0.2)
        jobb = S - 100 * np.exp(-0.05)
        np.testing.assert_allclose(C - P, jobb, rtol=1e-6)


class TestDelta:
    def test_tartomany(self):
        for p in (ATM, ITM, OTM):
            d = bs_delta(**p)
            assert 0.0 <= d <= 1.0

    def test_atm_koerul_fel(self):
        d = bs_delta(**ATM)
        assert 0.5 < d < 1.0

    def test_deep_itm_kozeliti_egyet(self):
        assert bs_delta(S=200, K=100, T=1, r=0.05, sigma=0.2) == pytest.approx(1.0, abs=0.01)

    def test_deep_otm_kozeliti_nullat(self):
        assert bs_delta(S=10, K=100, T=1, r=0.05, sigma=0.2) == pytest.approx(0.0, abs=0.01)

    def test_monoton_s_ban(self):
        S = np.linspace(60, 140, 30)
        deltas = bs_delta(S, K=100, T=1, r=0.05, sigma=0.2)
        assert np.all(np.diff(deltas) > 0)

    def test_lejarat_koezeleben(self):
        assert bs_delta(S=110, K=100, T=1e-10, r=0.05, sigma=0.2) == pytest.approx(1.0, abs=1e-4)
        assert bs_delta(S=90,  K=100, T=1e-10, r=0.05, sigma=0.2) == pytest.approx(0.0, abs=1e-4)


class TestGamma:
    def test_nem_negativ(self):
        for p in (ATM, ITM, OTM):
            assert bs_gamma(**p) >= 0

    def test_deep_itm_otm_kisebb(self):
        g_atm  = bs_gamma(**ATM)
        g_deep_itm = bs_gamma(S=200, K=100, T=1, r=0.05, sigma=0.2)
        g_deep_otm = bs_gamma(S=20,  K=100, T=1, r=0.05, sigma=0.2)
        assert g_atm > g_deep_itm
        assert g_atm > g_deep_otm

    def test_lejarat_koezeleben_nulla(self):
        assert bs_gamma(S=100, K=100, T=1e-10, r=0.05, sigma=0.2) == pytest.approx(0.0, abs=1e-4)


class TestVega:
    def test_nem_negativ(self):
        for p in (ATM, ITM, OTM):
            assert bs_vega(**p) >= 0

    def test_atm_maximalis(self):
        assert bs_vega(**ATM) > bs_vega(**ITM)
        assert bs_vega(**ATM) > bs_vega(**OTM)

    def test_lejarat_koezeleben_nulla(self):
        assert bs_vega(S=100, K=100, T=1e-10, r=0.05, sigma=0.2) == pytest.approx(0.0, abs=1e-6)

    def test_konzisztens_call_arral(self):
        eps = 1e-4
        p = ATM
        numerikus = (bs_call(**{**p, 'sigma': p['sigma'] + eps})
                   - bs_call(**{**p, 'sigma': p['sigma'] - eps})) / (2 * eps)
        assert bs_vega(**p) * 100 == pytest.approx(numerikus, rel=1e-3)


class TestTheta:
    def test_altalaban_negativ(self):
        for p in (ATM, ITM, OTM):
            assert bs_theta(**p) < 0

    def test_lejarat_koezeleben_nulla(self):
        assert bs_theta(S=100, K=100, T=1e-10, r=0.05, sigma=0.2) == pytest.approx(0.0, abs=1e-6)

    def test_konzisztens_call_arral(self):
        eps = 1e-4
        p = ATM
        dc_dt = (bs_call(**{**p, 'T': p['T'] + eps})
               - bs_call(**{**p, 'T': p['T'] - eps})) / (2 * eps)
        assert bs_theta(**p) * 365 == pytest.approx(-dc_dt, rel=1e-3)


class TestRho:
    def test_nem_negativ_call(self):
        for p in (ATM, ITM, OTM):
            assert bs_rho(**p) >= 0

    def test_lejarat_koezeleben_nulla(self):
        assert bs_rho(S=100, K=100, T=1e-10, r=0.05, sigma=0.2) == pytest.approx(0.0, abs=1e-6)

    def test_konzisztens_call_arral(self):
        eps = 1e-4
        p = ATM
        numerikus = (bs_call(**{**p, 'r': p['r'] + eps})
                   - bs_call(**{**p, 'r': p['r'] - eps})) / (2 * eps)
        assert bs_rho(**p) * 100 == pytest.approx(numerikus, rel=1e-3)


class TestLineárisHomogenitás:

    def test_skálázás(self):
        for lam in (0.5, 2.0, 10.0):
            c_orig   = bs_call(**ATM)
            c_scaled = bs_call(S=ATM['S']*lam, K=ATM['K']*lam,
                               T=ATM['T'], r=ATM['r'], sigma=ATM['sigma'])
            assert c_scaled == pytest.approx(lam * c_orig, rel=1e-6)
