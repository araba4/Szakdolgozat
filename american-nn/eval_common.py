import sys
sys.dont_write_bytecode = True

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import common as C
from common import AMNN_DIR, REPO_ROOT, guard_path, load_model

sys.path.insert(0, str(REPO_ROOT / "american-options" / "src"))
from data_generator_american import DEFAULT_PARAMS
import american_pricers as AP
from src.black_scholes import bs_put

EVAL_DIR = AMNN_DIR / "results" / "evaluation"
LOG_DIR = AMNN_DIR / "logs" / "evaluation"
LEVELS_TABLE = AMNN_DIR / "results" / "retune" / "levels_table.csv"

LEVELS = ['uniform_p30', 'transfer_p30', 'retune_p30']
LEVEL_LABEL = {'uniform_p30': 'uniform p30', 'transfer_p30': 'transfer p30',
               'retune_p30': 'retune p30'}
TARGETS = ['american', 'european']

RAW_FEATURES = ['moneyness', 'T', 'r', 'sigma']
assert C.FEATURE_COLS == [f"{p}_norm" for p in RAW_FEATURES]
NORM_LOW = np.array([DEFAULT_PARAMS[p][0] for p in RAW_FEATURES], dtype=np.float64)
NORM_HIGH = np.array([DEFAULT_PARAMS[p][1] for p in RAW_FEATURES], dtype=np.float64)
NORM_SPAN = NORM_HIGH - NORM_LOW


def checkpoint_path(arch: str, target: str, level: str) -> Path:
    if level == 'uniform_p30':
        return AMNN_DIR / "models" / "uniform" / f"{arch}_{target}_p30_best.pt"
    if level == 'transfer_p30':
        return AMNN_DIR / "models" / "transfer" / f"{arch}_{target}_transfer_p30_best.pt"
    if level == 'retune_p30':
        return AMNN_DIR / "models" / "retune" / f"{arch}_{target}_retune_p30_best.pt"
    raise ValueError(level)


def run_id(arch: str, target: str, level: str) -> str:
    return f"{arch}_{target}_{level}"


def levels_table() -> pd.DataFrame:
    return pd.read_csv(LEVELS_TABLE)


def model_registry() -> List[dict]:
    lt = levels_table()
    out = []
    for _, r in lt.iterrows():
        ck = checkpoint_path(r['arch'], r['target'], r['level'])
        if not ck.exists():
            raise FileNotFoundError(ck)
        out.append(dict(arch=r['arch'], target=r['target'], level=r['level'],
                        run=run_id(r['arch'], r['target'], r['level']),
                        checkpoint=ck, best_val_mse=float(r['best_val_mse']),
                        test_rmse=float(r['test_rmse']), test_usd_rmse=float(r['test_usd_rmse']),
                        source_file=r['source_file']))
    return out


def normalize(m, T, r, sigma) -> np.ndarray:
    raw = {'moneyness': np.asarray(m, dtype=np.float64), 'T': np.asarray(T, dtype=np.float64),
           'r': np.asarray(r, dtype=np.float64), 'sigma': np.asarray(sigma, dtype=np.float64)}
    cols = []
    for p in RAW_FEATURES:
        low, high = DEFAULT_PARAMS[p]
        cols.append((raw[p] - low) / (high - low))
    return np.stack(cols, axis=1)


def check_norm_against_parquet(df: pd.DataFrame) -> dict:
    Xn = normalize(df['moneyness'], df['T'], df['r'], df['sigma'])
    out = {}
    for j, p in enumerate(RAW_FEATURES):
        stored = df[f"{p}_norm"].to_numpy(dtype=np.float64)
        diff = np.abs(Xn[:, j] - stored)
        out[p] = {'max_abs_diff': float(diff.max()), 'n_not_bitwise_equal': int((Xn[:, j] != stored).sum())}
    out['all_bitwise_equal'] = all(v['n_not_bitwise_equal'] == 0 for v in out.values())
    return out


def load(ckpt: Path, device: str = C.DEVICE):
    model, meta = load_model(str(ckpt), device=device)
    model.eval()
    return model, meta


@torch.no_grad()
def nn_price_norm(model, m, T, r, sigma, device: str = C.DEVICE, batch_size: int = 65536) -> np.ndarray:
    X = normalize(m, T, r, sigma)
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    out = [model(Xt[i:i + batch_size]).cpu().numpy() for i in range(0, len(Xt), batch_size)]
    return np.concatenate(out, axis=0).ravel()


def nn_price_and_grads(model, m, T, r, sigma, device: str = C.DEVICE, batch_size: int = 65536):
    X = normalize(m, T, r, sigma)
    Xt_all = torch.tensor(X, dtype=torch.float32).to(device)
    fs, gm, gT = [], [], []
    for i in range(0, len(Xt_all), batch_size):
        x = Xt_all[i:i + batch_size].clone().requires_grad_(True)
        y = model(x)
        g, = torch.autograd.grad(y.sum(), x)
        fs.append(y.detach().cpu().numpy().ravel())
        g = g.detach().cpu().numpy().astype(np.float64)
        gm.append(g[:, 0] / NORM_SPAN[0])
        gT.append(g[:, 1] / NORM_SPAN[1])
    return np.concatenate(fs), np.concatenate(gm), np.concatenate(gT)


def bs_put_norm(m, T, r, sigma) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return bs_put(m, np.ones_like(m), T, r, sigma)


def env_extra() -> Dict:
    meta = C.env_meta()
    meta['torch_num_threads'] = torch.get_num_threads()
    return meta
