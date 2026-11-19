import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)


REPO_ROOT = Path(__file__).resolve().parent.parent
ONLAB_DIR = REPO_ROOT / "Önlab eredmények"
sys.path.insert(0, str(ONLAB_DIR))

from src.model import get_model, count_parameters
from src.train import (
    OptionDataset, train_model,
    DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL,
)
from src.evaluate import evaluate_model

HPO_DIR = REPO_ROOT / "hpo"
DATA_DIR = ONLAB_DIR / "data"
BASELINE_MODELS_DIR = ONLAB_DIR / "models"
BASELINE_RESULTS_DIR = ONLAB_DIR / "results"

STUDIES_DIR = HPO_DIR / "optuna_studies"
MODELS_DIR = HPO_DIR / "models"
RESULTS_DIR = HPO_DIR / "results"
LOGS_DIR = HPO_DIR / "logs"
for _d in (STUDIES_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = str(DATA_DIR / "train.parquet")
VAL_PATH = str(DATA_DIR / "val.parquet")
TEST_PATH = str(DATA_DIR / "test.parquet")


def rel(path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def cwd_rel(path) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return str(path)


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

TRIAL_BUDGET = {
    'mlp': 25, 'deep_mlp': 30, 'dense_mlp': 30,
    'resnet': 40, 'gelu_resnet': 40, 'highway': 40, 'finn': 40,
}

ARCH_ORDER = ['mlp', 'deep_mlp', 'dense_mlp', 'resnet', 'gelu_resnet', 'highway', 'finn']

SEARCH_MAX_EPOCHS = 40
SEARCH_PATIENCE = 8
SEARCH_BATCH_SIZE = 4096

FINAL_MAX_EPOCHS = 200
FINAL_PATIENCE = 10
FINAL_BATCH_SIZE = 4096

BASE_SEED = 42

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def log(arch: str, msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGS_DIR / f"{arch}.log", 'a', encoding='utf-8') as f:
        f.write(line + "\n")


_DATA_CACHE = {}


def get_datasets():
    if 'train' not in _DATA_CACHE:
        print(f"[data] Betoltes: {rel(TRAIN_PATH)} / {rel(VAL_PATH)} (device={DEVICE})")
        t0 = time.time()
        _DATA_CACHE['train'] = OptionDataset(TRAIN_PATH, DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL, device=DEVICE)
        _DATA_CACHE['val'] = OptionDataset(VAL_PATH, DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL, device=DEVICE)
        print(f"[data] train={len(_DATA_CACHE['train']):,} sor, "
              f"val={len(_DATA_CACHE['val']):,} sor ({time.time() - t0:.1f}s)")
    return _DATA_CACHE['train'], _DATA_CACHE['val']


def train_for_search(model, train_ds, val_ds, lr, weight_decay, trial,
                      max_epochs, patience, batch_size, device, curve_sink):
    model = model.to(device)
    X_train, y_train = train_ds.X, train_ds.y
    X_val, y_val = val_ds.X, val_ds.y
    n_train, n_val = len(X_train), len(X_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        perm = torch.randperm(n_train, device=X_train.device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            X_batch, y_batch = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(idx)
        train_loss = train_loss_sum / n_train

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                X_batch = X_val[start:start + batch_size]
                y_batch = y_val[start:start + batch_size]
                pred = model(X_batch)
                val_loss_sum += criterion(pred, y_batch).item() * len(X_batch)
        val_loss = val_loss_sum / n_val

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        curve_sink.append({
            'trial_number': trial.number, 'epoch': epoch,
            'train_loss': train_loss, 'val_loss': val_loss, 'lr': current_lr,
        })

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if epochs_no_improve >= patience:
            break

    return best_val


def build_trial_hparams(trial: 'optuna.Trial', arch: str):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    model_kwargs = dict(BASELINE_ARCH_KWARGS[arch])
    dropout = None
    if HAS_DROPOUT[arch]:
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        model_kwargs['dropout'] = dropout
    return lr, weight_decay, dropout, model_kwargs


def _append_curve_rows(arch: str, rows: list):
    if not rows:
        return
    curve_path = RESULTS_DIR / f"search_curves_{arch}.csv"
    df = pd.DataFrame(rows, columns=['trial_number', 'epoch', 'train_loss', 'val_loss', 'lr'])
    write_header = not curve_path.exists()
    df.to_csv(curve_path, mode='a', header=write_header, index=False)


def make_objective(arch: str, search_curves: list):
    train_ds, val_ds = get_datasets()

    def objective(trial: 'optuna.Trial'):
        torch.manual_seed(BASE_SEED + trial.number)
        lr, weight_decay, dropout, model_kwargs = build_trial_hparams(trial, arch)
        model = get_model(arch, **model_kwargs)
        start_idx = len(search_curves)
        try:
            best_val = train_for_search(
                model, train_ds, val_ds, lr, weight_decay, trial,
                max_epochs=SEARCH_MAX_EPOCHS, patience=SEARCH_PATIENCE,
                batch_size=SEARCH_BATCH_SIZE, device=DEVICE,
                curve_sink=search_curves,
            )
            return best_val
        finally:
            _append_curve_rows(arch, search_curves[start_idx:])

    return objective


def run_study(arch: str):
    n_trials_target = TRIAL_BUDGET[arch]
    class_name = CLASS_NAME[arch]

    log(arch, f"=== {arch} ({class_name}) HPO study indul -- device={DEVICE}, "
              f"trial-cel={n_trials_target}, fix meret={BASELINE_ARCH_KWARGS[arch]}, "
              f"dropout kereshetos={HAS_DROPOUT[arch]} ===")

    storage = f"sqlite:///{(STUDIES_DIR / f'{arch}.db').as_posix()}"
    sampler = TPESampler(seed=BASE_SEED)
    pruner = HyperbandPruner(min_resource=3, max_resource=SEARCH_MAX_EPOCHS, reduction_factor=3)

    study = optuna.create_study(
        study_name=arch, storage=storage, direction='minimize',
        sampler=sampler, pruner=pruner, load_if_exists=True,
    )

    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            log(arch, f"Elakadt (stale) RUNNING trial #{t.number} FAIL-re allitva "
                      f"(korabban megszakadt folyamatbol maradt vissza).")
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)

    n_valid = sum(
        1 for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    )
    n_remaining = max(0, n_trials_target - n_valid)
    if len(study.trials) > 0:
        log(arch, f"Folytatas: {n_valid} ervenyes trial mar letezik a storage-ban, "
                  f"{n_remaining} tovabbi trial hianyzik a celhoz.")

    search_curves = []
    t0 = time.time()
    if n_remaining > 0:
        study.optimize(make_objective(arch, search_curves), n_trials=n_remaining,
                        show_progress_bar=False)
    dt = time.time() - t0
    log(arch, f"Study kesz: {len(study.trials)} trial osszesen ({dt:.1f}s ebben a futasban). "
              f"Legjobb val_loss={study.best_value:.8f}, trial #{study.best_trial.number}, "
              f"params={study.best_trial.params}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(RESULTS_DIR / f"trials_{arch}.csv", index=False)

    curve_path = RESULTS_DIR / f"search_curves_{arch}.csv"
    if curve_path.exists():
        curve_df = pd.read_csv(curve_path)
        curve_df = curve_df.drop_duplicates(subset=['trial_number', 'epoch'], keep='last')
        curve_df = curve_df.sort_values(['trial_number', 'epoch'])
        curve_df.to_csv(curve_path, index=False)

    try:
        importances = optuna.importance.get_param_importances(study)
        importances = {k: float(v) for k, v in importances.items()}
    except Exception as e:
        importances = {"error": str(e)}
    with open(RESULTS_DIR / f"param_importance_{arch}.json", 'w', encoding='utf-8') as f:
        json.dump(importances, f, indent=2, ensure_ascii=False)

    return study


def final_retrain(arch: str, best_params: dict):
    class_name = CLASS_NAME[arch]
    model_kwargs = dict(BASELINE_ARCH_KWARGS[arch])
    dropout = best_params.get('dropout')
    if dropout is not None:
        model_kwargs['dropout'] = dropout

    log(arch, f"Vegso ujratanitas indul (teljes koltsegvetes: max_epochs={FINAL_MAX_EPOCHS}, "
              f"patience={FINAL_PATIENCE}) lr={best_params['lr']:.6g}, "
              f"weight_decay={best_params['weight_decay']:.6g}, dropout={dropout}, "
              f"model_kwargs={model_kwargs}")

    torch.manual_seed(BASE_SEED)
    model = get_model(arch, **model_kwargs)

    t0 = time.time()
    history = train_model(
        model=model,
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        output_dir=cwd_rel(MODELS_DIR),
        model_name=f"{arch}_hpo",
        model_class=arch,
        model_kwargs=model_kwargs,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        batch_size=FINAL_BATCH_SIZE,
        max_epochs=FINAL_MAX_EPOCHS,
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
        patience=FINAL_PATIENCE,
        device=DEVICE,
    )
    dt = time.time() - t0
    log(arch, f"Vegso ujratanitas kesz ({dt:.1f}s): best_epoch={history['best_epoch']}, "
              f"best_val_loss={history['best_val_loss']:.8f}")

    hist_df = pd.DataFrame({
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'lr': history['lr'],
    })
    hist_df.to_csv(RESULTS_DIR / f"final_training_history_{arch}.csv", index=False)

    return history, model_kwargs, dt


def param_count_check(arch: str, hpo_model_kwargs: dict):
    baseline_ckpt_path = BASELINE_MODELS_DIR / f"{arch}_best.pt"
    baseline_ckpt = torch.load(baseline_ckpt_path, map_location='cpu', weights_only=False)
    baseline_model = get_model(baseline_ckpt['model_class'], **baseline_ckpt['model_kwargs'])
    baseline_count = count_parameters(baseline_model)

    hpo_model = get_model(arch, **hpo_model_kwargs)
    hpo_count = count_parameters(hpo_model)

    result = {
        'arch': arch,
        'baseline_checkpoint': rel(baseline_ckpt_path),
        'baseline_model_kwargs': baseline_ckpt['model_kwargs'],
        'hpo_model_kwargs': hpo_model_kwargs,
        'baseline_param_count': baseline_count,
        'hpo_param_count': hpo_count,
        'match': baseline_count == hpo_count,
    }
    with open(RESULTS_DIR / f"param_count_check_{arch}.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log(arch, f"Parameterszam-ellenorzes: baseline={baseline_count:,} vs hpo={hpo_count:,} "
              f"-> match={result['match']}")
    if not result['match']:
        raise RuntimeError(
            f"[{arch}] PARAMETERSZAM ELTERES! baseline={baseline_count} != hpo={hpo_count}. "
            f"Ez hibat jelez a fix-meretu architektura atadasaban -- javitando mielott tovabb megyunk."
        )
    return result


def safe_test_metrics(checkpoint_path: Path):
    try:
        res = evaluate_model(str(checkpoint_path), TEST_PATH, device=DEVICE, verbose=False)
        return res['metrics']
    except Exception as e:
        return {'error': str(e)}


def baseline_val_loss(arch: str):
    class_name = CLASS_NAME[arch]
    path = BASELINE_RESULTS_DIR / f"training_history_{class_name}.json"
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        h = json.load(f)
    return h.get('best_val_loss')


def run_full_pipeline(arch: str):
    t_start = time.time()
    study = run_study(arch)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    if 'dropout' not in best_params:
        best_params['dropout'] = None

    history, model_kwargs, retrain_seconds = final_retrain(arch, best_params)

    pcc = param_count_check(arch, model_kwargs)

    baseline_ckpt = BASELINE_MODELS_DIR / f"{arch}_best.pt"
    hpo_ckpt = MODELS_DIR / f"{arch}_hpo_best.pt"

    baseline_metrics = safe_test_metrics(baseline_ckpt)
    hpo_metrics = safe_test_metrics(hpo_ckpt)

    b_val = baseline_val_loss(arch)
    h_val = history['best_val_loss']
    improvement_pct = None
    if b_val is not None and b_val > 0:
        improvement_pct = (b_val - h_val) / b_val * 100.0

    best_params_out = {
        'arch': arch,
        'model_class': CLASS_NAME[arch],
        'architecture_size_fixed_baseline': BASELINE_ARCH_KWARGS[arch],
        'lr': best_params['lr'],
        'weight_decay': best_params['weight_decay'],
        'dropout': best_params['dropout'],
        'search_best_val_loss': study.best_value,
        'search_best_trial_number': best_trial.number,
        'n_trials_completed': len(study.trials),
        'n_trials_target': TRIAL_BUDGET[arch],
        'final_retrain_best_epoch': history['best_epoch'],
        'final_retrain_best_val_loss': h_val,
        'baseline_val_loss': b_val,
        'improvement_pct_vs_baseline_val_loss': improvement_pct,
        'param_count_match': pcc['match'],
        'param_count': pcc['hpo_param_count'],
        'baseline_test_metrics': baseline_metrics,
        'hpo_test_metrics': hpo_metrics,
        'retrain_seconds': retrain_seconds,
    }
    with open(RESULTS_DIR / f"best_params_{arch}.json", 'w', encoding='utf-8') as f:
        json.dump(best_params_out, f, indent=2, ensure_ascii=False)

    total_dt = time.time() - t_start
    log(arch, f"=== {arch} PIPELINE KESZ ({total_dt:.1f}s). "
              f"baseline_val={b_val}, hpo_val={h_val}, javulas={improvement_pct}% ===")
    return best_params_out


def _sci(v):
    if not isinstance(v, (int, float)):
        return "–"
    mant, exp = f"{v:.2e}".split("e")
    return f"{mant}e{int(exp)}"


def _change(base, new):
    if not isinstance(base, (int, float)) or not isinstance(new, (int, float)) or base <= 0:
        return "–"
    return f"{(new - base) / base * 100:+.1f}%".replace("-", "−")


def write_summary(out_path=None):
    out_path = Path(out_path) if out_path is not None else RESULTS_DIR / "hpo_summary.md"
    rows = []
    for arch in ARCH_ORDER:
        p = RESULTS_DIR / f"best_params_{arch}.json"
        if not p.exists():
            continue
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        rows.append(d)

    lines = []
    lines.append("## Teszt-RMSE és validációs MSE\n")
    lines.append(
        "| Architektúra | Paraméter | Teszt-RMSE, kiindulási | Teszt-RMSE, HPO | "
        "Változás | Val MSE, kiindulási | Val MSE, HPO |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in rows:
        b_rmse = (d.get('baseline_test_metrics') or {}).get('RMSE')
        h_rmse = (d.get('hpo_test_metrics') or {}).get('RMSE')
        params = f"{d['param_count']:,}".replace(",", " ")
        lines.append(
            f"| {d['model_class']} | {params} | {_sci(b_rmse)} | {_sci(h_rmse)} | "
            f"{_change(b_rmse, h_rmse)} | {_sci(d['baseline_val_loss'])} | "
            f"{_sci(d['final_retrain_best_val_loss'])} |"
        )

    lines.append("\n## Legjobb hiperparaméterek\n")
    lines.append("| Architektúra | Trial | lr | weight_decay | dropout |")
    lines.append("|---|---:|---:|---:|---:|")
    for d in rows:
        dropout = f"{d['dropout']:.3f}" if d['dropout'] is not None else "–"
        lines.append(
            f"| {d['model_class']} | {d['n_trials_completed']} | "
            f"{_sci(d['lr'])} | {_sci(d['weight_decay'])} | {dropout} |"
        )

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[summary] {out_path.name} irva ({len(rows)}/{len(ARCH_ORDER)} architektura).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default=None)
    parser.add_argument('--archs', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()

    if args.summary_only:
        write_summary()
        return

    if args.arch:
        archs = [args.arch]
    elif args.archs:
        archs = [a.strip() for a in args.archs.split(',') if a.strip()]
    else:
        archs = ARCH_ORDER

    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU memoria: {total_mem:.1f} GB")
    print(f"Futtatando architekturak: {archs}")

    for arch in archs:
        try:
            run_full_pipeline(arch)
        except Exception:
            err = traceback.format_exc()
            with open(LOGS_DIR / f"{arch}_ERROR.log", 'a', encoding='utf-8') as f:
                f.write(err + "\n")
            print(f"[HIBA] {arch}:\n{err}")

    write_summary()


if __name__ == '__main__':
    main()
