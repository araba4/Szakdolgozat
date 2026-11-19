import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model import get_model, count_parameters

DEFAULT_FEATURE_COLS = ['moneyness_norm', 'T_norm', 'r_norm', 'sigma_norm']
DEFAULT_TARGET_COL = 'call_price_norm'

AUGMENTED_FEATURE_COLS = DEFAULT_FEATURE_COLS + ['is_put']


class OptionDataset(Dataset):

    def __init__(self, path: str,
                 feature_cols: List[str] = None,
                 target_col: str = None,
                 device: str = 'cpu'):
        if feature_cols is None:
            feature_cols = DEFAULT_FEATURE_COLS
        if target_col is None:
            target_col = DEFAULT_TARGET_COL

        df = pd.read_parquet(path, columns=feature_cols + [target_col])

        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(device)
        self.y = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1).to(device)

        self.feature_cols = feature_cols
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AugmentedOptionDataset(Dataset):

    def __init__(self, path: str,
                 feature_cols: List[str] = None,
                 target_col: str = None,
                 device: str = 'cpu'):
        if feature_cols is None:
            feature_cols = DEFAULT_FEATURE_COLS
        if target_col is None:
            target_col = DEFAULT_TARGET_COL

        base_feature_cols = [c for c in feature_cols if c != 'is_put']
        raw_cols = ['moneyness', 'T', 'r']
        df = pd.read_parquet(path, columns=base_feature_cols + [target_col] + raw_cols)

        X_base = df[base_feature_cols].values.astype(np.float32)
        y_call = df[target_col].values.astype(np.float32)

        m  = df['moneyness'].values
        T  = df['T'].values
        r  = df['r'].values
        y_put = (y_call - m + np.exp(-r * T)).astype(np.float32)

        n = len(df)
        X_call = np.hstack([X_base, np.zeros((n, 1), dtype=np.float32)])
        X_put  = np.hstack([X_base, np.ones( (n, 1), dtype=np.float32)])

        X_all = np.vstack([X_call, X_put])
        y_all = np.concatenate([y_call, y_put]).reshape(-1, 1)

        self.X = torch.tensor(X_all, dtype=torch.float32).to(device)
        self.y = torch.tensor(y_all, dtype=torch.float32).to(device)

        self.feature_cols = feature_cols + ['is_put']
        self.target_col   = target_col

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(
    model: nn.Module,
    train_path: str,
    val_path: str,
    output_dir: str,
    model_name: str,
    model_class: str,
    model_kwargs: dict,
    feature_cols: List[str] = None,
    target_col: str = None,
    batch_size: int = 4096,
    max_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: str = 'auto',
    augment_put: bool = False,
    physics_loss: bool = False,
    physics_lambda: float = 0.1,
) -> dict:
    if feature_cols is None:
        feature_cols = AUGMENTED_FEATURE_COLS if augment_put else DEFAULT_FEATURE_COLS
    if target_col is None:
        target_col = DEFAULT_TARGET_COL

    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    device = torch.device(device)
    print(f"Eszköz: {device}")
    if device.type == 'cuda':
        idx = device.index if device.index is not None else torch.cuda.current_device()
        print(f"  GPU neve:    {torch.cuda.get_device_name(idx)}")
        total_mem = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        print(f"  GPU memória: {total_mem:.1f} GB")

    DatasetCls = AugmentedOptionDataset if augment_put else OptionDataset
    print(f"Adatok betöltése {'(put-call augmentáció)' if augment_put else ''}...")
    t0 = time.time()
    train_ds = DatasetCls(train_path, feature_cols, target_col, device=str(device))
    val_ds   = DatasetCls(val_path,   feature_cols, target_col, device=str(device))
    print(f"  Tanítóhalmaz:    {len(train_ds):,} sor")
    print(f"  Validációs halmaz: {len(val_ds):,} sor")
    print(f"  Betöltési idő:   {time.time() - t0:.2f}s")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = model.to(device)
    print(f"Modell paraméterek: {count_parameters(model):,}")
    if device.type == 'cuda':
        actual_device = next(model.parameters()).device
        assert actual_device.type == 'cuda', f"Modell nem GPU-n van: {actual_device}"
        print(f"  Modell GPU-n: {actual_device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    moneyness_idx: Optional[int] = None
    if physics_loss:
        try:
            moneyness_idx = (feature_cols or DEFAULT_FEATURE_COLS).index('moneyness_norm')
        except ValueError:
            raise ValueError("physics_loss=True, de 'moneyness_norm' nincs a feature_cols-ban.")
        print(f"Physics-informed loss: bekapcsolva (lambda={physics_lambda}, "
              f"moneyness_idx={moneyness_idx})")

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{model_name}_best.pt")

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    print(f"\nTanítás ({max_epochs} epoch max, patience={patience})")
    print("-" * 60)

    for epoch in range(1, max_epochs + 1):
        t_epoch = time.time()

        model.train()
        train_loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            if physics_loss:
                X_phys = X_batch.detach().requires_grad_(True)
                pred = model(X_phys)
                mse = criterion(pred, y_batch)
                grads = torch.autograd.grad(
                    pred.sum(), X_phys, create_graph=True
                )[0]
                delta = grads[:, moneyness_idx]
                phys = physics_lambda * (
                    torch.relu(-delta) + torch.relu(delta - 1.0)
                ).mean()
                loss = mse + phys
            else:
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
            }, checkpoint_path)
        else:
            epochs_no_improve += 1

        epoch_time = time.time() - t_epoch
        print(
            f"Epoch {epoch:4d}/{max_epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"{epoch_time:.1f}s"
            + (" *" if epochs_no_improve == 0 else "")
        )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping: {patience} epoch óta nem javult a val loss.")
            break

    print("-" * 60)
    print(f"Legjobb epoch: {best_epoch}, val loss: {best_val_loss:.6f}")
    print(f"Checkpoint mentve: {checkpoint_path}")

    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val_loss
    return history
