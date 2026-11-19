import sys
sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C
import train_residual as R


@pytest.fixture(scope='module', params=list(C.SPLITS))
def df(request):
    return C.load_am_split(request.param)


def test_existing_targets_registry_unchanged():
    assert C.TARGETS['american'] == dict(price_cols=['american_put_price'], target_col='american_put_price_norm',
                                         label='amerikai put', formula='american_put_price / K')
    assert C.TARGETS['european'] == dict(price_cols=['european_put_price'], target_col='european_put_price_norm',
                                         label='európai put (BS)', formula='european_put_price / K')
    assert 'minus_col' not in C.TARGETS['american'] and 'minus_col' not in C.TARGETS['european']


def test_existing_targets_bitwise(df):
    K = df['K'].to_numpy(np.float64)
    for t, col in (('american', 'american_put_price'), ('european', 'european_put_price')):
        ref = df[col].to_numpy(np.float64) / K
        assert np.array_equal(C.target_values(df, t), ref)


def test_residual_definition(df):
    K = df['K'].to_numpy(np.float64)
    res = C.target_values(df, 'american_minus_bs')
    ref = (df['american_put_price'].to_numpy(np.float64) - df['european_put_price'].to_numpy(np.float64)) / K
    assert np.array_equal(res, ref)
    assert C.TARGETS['american_minus_bs']['target_col'] == 'american_minus_bs_norm'


def test_reconstruction_float64(df):
    am = C.target_values(df, 'american')
    bs = C.target_values(df, 'european')
    res = C.target_values(df, 'american_minus_bs')
    np.testing.assert_allclose(bs + res, am, rtol=0, atol=1e-15)


def test_reconstruction_float32_path(df):
    am = C.target_values(df, 'american')
    bs = C.target_values(df, 'european')
    res32 = C.target_values(df, 'american_minus_bs').astype(np.float32)
    rec = R.reconstruct(res32, bs)
    assert rec.dtype == np.float32
    np.testing.assert_allclose(rec.astype(np.float64), am.astype(np.float32).astype(np.float64),
                               rtol=0, atol=1e-7)


def test_residual_is_not_premium(df):
    K = df['K'].to_numpy(np.float64)
    res = C.target_values(df, 'american_minus_bs')
    prem = df['early_exercise_premium'].to_numpy(np.float64) / K
    tree_err = (df['crr_european_put_price'].to_numpy(np.float64)
                - df['european_put_price'].to_numpy(np.float64)) / K
    np.testing.assert_allclose(res - prem, tree_err, rtol=0, atol=1e-15)
    assert np.abs(res - prem).max() > 1e-5
    assert (res < 0).any() and (prem >= 0).all()
