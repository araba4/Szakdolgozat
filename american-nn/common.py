import sys
sys.dont_write_bytecode = True

import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


AMNN_DIR = Path(__file__).resolve().parent
REPO_ROOT = AMNN_DIR.parent
ONLAB_DIR = REPO_ROOT / "Önlab eredmények"
ONLAB_DATA_DIR = ONLAB_DIR / "data"
ONLAB_MODELS_DIR = ONLAB_DIR / "models"
ONLAB_RESULTS_DIR = ONLAB_DIR / "results"
AM_DATA_DIR = REPO_ROOT / "american-options" / "data"

PROTECTED_DIRS = [ONLAB_DIR, AM_DATA_DIR, REPO_ROOT / "hpo"]

sys.path.insert(0, str(ONLAB_DIR))

from src.model import get_model, count_parameters
from src.train import DEFAULT_FEATURE_COLS
from src.evaluate import load_model, compute_metrics

FEATURE_COLS = list(DEFAULT_FEATURE_COLS)


CLASS_NAME = {
    'mlp':         'MLPPricer',
    'deep_mlp':    'DeepMLPPricer',
    'resnet':      'ResNetPricer',
    'gelu_resnet': 'GELUResNetPricer',
    'dense_mlp':   'DenseMLPPricer',
    'highway':     'HighwayPricer',
    'finn':        'FINNPricer',
}

BASELINE_ARCH_KWARGS = {
    'mlp':         dict(input_dim=4, hidden_dim=100, n_layers=4),
    'deep_mlp':    dict(input_dim=4, hidden_dim=256, n_layers=4),
    'resnet':      dict(input_dim=4, hidden_dim=256, n_blocks=3),
    'gelu_resnet': dict(input_dim=4, hidden_dim=256, n_blocks=3),
    'dense_mlp':   dict(input_dim=4, hidden_dim=128, n_layers=4),
    'highway':     dict(input_dim=4, hidden_dim=256, n_blocks=4),
    'finn':        dict(input_dim=4, approx_dim=64, resnet_dim=256, n_blocks=3),
}

HAS_DROPOUT = {
    'mlp': False, 'deep_mlp': True, 'resnet': True, 'gelu_resnet': True,
    'dense_mlp': True, 'highway': True, 'finn': True,
}

ARCH_ORDER = ['mlp', 'deep_mlp', 'dense_mlp', 'resnet', 'gelu_resnet', 'highway', 'finn']

BASE_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


_ALLOWED_ROOTS: List[Path] = [AMNN_DIR]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def set_allowed_roots(*roots: Path):
    global _ALLOWED_ROOTS
    for r in roots:
        for d in PROTECTED_DIRS:
            if _is_within(r, d):
                raise PermissionError(f"[guard] Vedett mappa nem lehet kimeneti gyoker: {r}")
    _ALLOWED_ROOTS = [Path(r).resolve() for r in roots]


def guard_path(path) -> Path:
    path = Path(path).resolve()
    for d in PROTECTED_DIRS:
        if _is_within(path, d):
            raise PermissionError(f"[guard] Vedett mappa, nem irhato: {path}")
    if not any(_is_within(path, r) for r in _ALLOWED_ROOTS):
        raise PermissionError(f"[guard] Nem engedelyezett kimeneti hely: {path}")
    return path


def write_json(path: Path, obj):
    with open(guard_path(path), 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def rel(path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


SPLITS = ('train', 'val', 'test')


def am_split_path(split: str) -> Path:
    return AM_DATA_DIR / f"{split}.parquet"


def onlab_split_path(split: str) -> Path:
    return ONLAB_DATA_DIR / f"{split}.parquet"


TARGETS = {
    'american': dict(price_cols=['american_put_price'],
                     target_col='american_put_price_norm',
                     label='amerikai put',
                     formula='american_put_price / K'),
    'european': dict(price_cols=['european_put_price'],
                     target_col='european_put_price_norm',
                     label='európai put (BS)',
                     formula='european_put_price / K'),
    'american_minus_bs': dict(price_cols=['american_put_price'],
                              minus_col='european_put_price',
                              target_col='american_minus_bs_norm',
                              label='amerikai put − BS put (reziduum)',
                              formula='(american_put_price - european_put_price) / K'),
}


def target_values(df: pd.DataFrame, target: str) -> np.ndarray:
    spec = TARGETS[target]
    price = df[spec['price_cols'][0]].to_numpy(dtype=np.float64)
    if 'minus_col' in spec:
        price = price - df[spec['minus_col']].to_numpy(dtype=np.float64)
    return price / df['K'].to_numpy(dtype=np.float64)


_DF_CACHE: Dict[str, pd.DataFrame] = {}

EVAL_COLS = ['S', 'K', 'moneyness', 'T', 'r', 'sigma', 'american_put_price',
             'european_put_price', 'crr_european_put_price', 'early_exercise_premium']


def load_am_split(split: str) -> pd.DataFrame:
    if split not in _DF_CACHE:
        _DF_CACHE[split] = pd.read_parquet(am_split_path(split), columns=FEATURE_COLS + EVAL_COLS)
    return _DF_CACHE[split]


class PricingDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str],
                 target_col: str, device: str = 'cpu'):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
        self.feature_cols = feature_cols
        self.target_col = target_col

    @classmethod
    def from_split(cls, split: str, target: str, device: str = 'cpu') -> 'PricingDataset':
        df = load_am_split(split)
        return cls(df[FEATURE_COLS].values, target_values(df, target),
                   FEATURE_COLS, TARGETS[target]['target_col'], device=device)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_loop(model: nn.Module, train_ds: Dataset, val_ds: Dataset,
               output_dir, model_name: str, model_class: str, model_kwargs: dict,
               feature_cols: List[str], target_col: str,
               batch_size: int = 4096, max_epochs: int = 200, lr: float = 1e-3,
               weight_decay: float = 1e-4, patience: int = 10, device: str = 'cuda',
               extra_meta: Optional[dict] = None, checkpoint_path=None) -> dict:
    device = torch.device(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    print(f"Modell parameterek: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    if checkpoint_path is None:
        checkpoint_path = os.path.join(str(output_dir), f"{model_name}_best.pt")
    checkpoint_path = str(guard_path(checkpoint_path))
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'epoch_seconds': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    print(f"\nTanitas ({max_epochs} epoch max, patience={patience})")
    print("-" * 60)

    for epoch in range(1, max_epochs + 1):
        t_epoch = time.time()

        model.train()
        train_loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(X_batch)
        train_loss = train_loss_sum / len(train_ds)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                val_loss_sum += criterion(pred, y_batch).item() * len(X_batch)
        val_loss = val_loss_sum / len(val_ds)

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'state_dict':   model.state_dict(),
                'model_class':  model_class,
                'model_kwargs': model_kwargs,
                'feature_cols': feature_cols,
                'target_col':   target_col,
                'history':      history,
                'best_epoch':   epoch,
                'val_loss':     val_loss,
                'extra_meta':   extra_meta or {},
            }, checkpoint_path)
        else:
            epochs_no_improve += 1

        epoch_time = time.time() - t_epoch
        history['epoch_seconds'].append(epoch_time)
        print(
            f"Epoch {epoch:4d}/{max_epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"{epoch_time:.1f}s"
            + (" *" if epochs_no_improve == 0 else ""),
            flush=True,
        )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping: {patience} epoch ota nem javult a val loss.")
            break

    print("-" * 60)
    print(f"Legjobb epoch: {best_epoch}, val loss: {best_val_loss:.6f}")
    print(f"Checkpoint mentve: {checkpoint_path}")

    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val_loss
    return history


@torch.no_grad()
def predict(model: nn.Module, ds: Dataset, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = [model(Xb).cpu().numpy() for Xb, _ in loader]
    return np.concatenate(out, axis=0).ravel()


PUT_MONEYNESS_SEGMENTS = {
    'ITM (S/K<0,9)':      (0.0, 0.9),
    'ATM (0,9≤S/K<1,1)':  (0.9, 1.1),
    'OTM (S/K≥1,1)':      (1.1, np.inf),
}

MATURITY_SEGMENTS = {
    'rövid (T<0,25)':       (0.0, 0.25),
    'közép (0,25≤T<1)':     (0.25, 1.0),
    'hosszú (1≤T≤2)':       (1.0, np.inf),
}

PREMIUM_ZERO_THRESHOLD = 1e-6
PREMIUM_N_QUANTILE_BUCKETS = 5


def premium_bucket_edges(prem_norm: np.ndarray) -> List[float]:
    pos = prem_norm[prem_norm >= PREMIUM_ZERO_THRESHOLD]
    qs = np.quantile(pos, np.linspace(0, 1, PREMIUM_N_QUANTILE_BUCKETS + 1)[1:-1])
    return [0.0, PREMIUM_ZERO_THRESHOLD] + [float(q) for q in qs] + [float(np.inf)]


def premium_bucket_labels(edges: List[float]) -> List[str]:
    labels = [f"~0 (<{edges[1]:.0e})"]
    for i in range(1, len(edges) - 1):
        hi = edges[i + 1]
        labels.append(f"Q{i} [{edges[i]:.2e}; {'max' if np.isinf(hi) else f'{hi:.2e}'})")
    return labels


def masked_metrics(y_true: np.ndarray, y_pred: np.ndarray, masks: Dict[str, np.ndarray],
                   scale: Optional[np.ndarray] = None) -> List[dict]:
    rows = []
    for name, m in masks.items():
        n = int(m.sum())
        row = {'segment': name, 'n': n}
        if n == 0:
            rows.append(row)
            continue
        met = compute_metrics(y_true[m], y_pred[m])
        met['MSE'] = met['RMSE'] ** 2
        row.update(met)
        if scale is not None:
            usd = compute_metrics(y_true[m] * scale[m], y_pred[m] * scale[m])
            row.update({f"usd_{k}": v for k, v in usd.items() if k != 'MAPE (%)'})
        rows.append(row)
    return rows


def segment_masks(df: pd.DataFrame, edges_prem: Optional[List[float]] = None) -> Dict[str, Dict[str, np.ndarray]]:
    m = df['moneyness'].to_numpy()
    T = df['T'].to_numpy()
    out = {
        'moneyness': {k: (m >= lo) & (m < hi) for k, (lo, hi) in PUT_MONEYNESS_SEGMENTS.items()},
        'lejarat': {k: (T >= lo) & (T < hi) for k, (lo, hi) in MATURITY_SEGMENTS.items()},
    }
    if edges_prem is not None:
        p = (df['early_exercise_premium'] / df['K']).to_numpy()
        labels = premium_bucket_labels(edges_prem)
        out['premium'] = {lab: (p >= edges_prem[i]) & (p < edges_prem[i + 1])
                          for i, lab in enumerate(labels)}
    return out


def env_meta() -> dict:
    meta = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'platform': platform.platform(),
        'device': DEVICE,
    }
    if torch.cuda.is_available():
        meta['gpu_name'] = torch.cuda.get_device_name(0)
    return meta
