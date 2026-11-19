import sys
sys.dont_write_bytecode = True

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AMNN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AMNN_DIR))

import common as C
from common import (
    ARCH_ORDER, CLASS_NAME, BASELINE_ARCH_KWARGS, HAS_DROPOUT, DEVICE,
    FEATURE_COLS, TARGETS, get_model, count_parameters, guard_path, write_json, read_json, rel,
)
import train_uniform_baseline as TU

HPO_P30_DIR = C.REPO_ROOT / "hpo" / "results" / "p30"
HPO_MODELS_DIR = C.REPO_ROOT / "hpo" / "models"
UNIFORM_RESULTS_DIR = AMNN_DIR / "results" / "uniform"

TARGET_ORDER = ['american', 'european']
PATIENCE = 30
EXPECTED_FINAL = {'final_max_epochs': 200, 'final_batch_size': 4096, 'seed': 42, 'final_patience': 30}


def run_name(arch, target):
    return f"{arch}_{target}_transfer_p{PATIENCE}"


class Paths:
    def __init__(self, out_root: Path):
        self.out_root = out_root.resolve()
        self.models_dir = self.out_root / "models" / "transfer"
        self.results_dir = self.out_root / "results" / "transfer"
        self.logs_dir = self.out_root / "logs" / "transfer"

    def ensure(self):
        for d in (self.models_dir, self.results_dir, self.logs_dir):
            guard_path(d).mkdir(parents=True, exist_ok=True)

    def ckpt(self, run):
        return self.models_dir / f"{run}_best.pt"


P = Paths(AMNN_DIR)


def best_params_path(arch) -> Path:
    return HPO_P30_DIR / f"best_params_{arch}.json"


def load_transfer_hp(arch) -> dict:
    src = best_params_path(arch)
    bp = read_json(src)
    if bp.get('arch') != arch:
        raise RuntimeError(f"[{arch}] A best_params JSON mas architekturae: {bp.get('arch')}")
    for k, v in EXPECTED_FINAL.items():
        if bp.get(k) != v:
            raise RuntimeError(f"[{arch}] {rel(src)}: {k}={bp.get(k)!r}, elvart {v!r} -- leallas.")
    if HAS_DROPOUT[arch] and bp.get('dropout') is None:
        raise RuntimeError(f"[{arch}] Hianyzo dropout a {rel(src)} fajlban.")
    return {
        'lr': bp['lr'], 'weight_decay': bp['weight_decay'],
        'dropout': bp['dropout'] if HAS_DROPOUT[arch] else None,
        'batch_size': bp['final_batch_size'], 'max_epochs': bp['final_max_epochs'],
        'seed': bp['seed'], 'patience': bp['final_patience'],
        'source_json': rel(src),
        'source_search': bp.get('search_source'),
        'source_european_call_p30_val_loss': bp.get('final_retrain_best_val_loss'),
    }


def model_kwargs_for(arch, hp):
    kw = dict(BASELINE_ARCH_KWARGS[arch])
    if HAS_DROPOUT[arch]:
        kw['dropout'] = hp['dropout']
    return kw


def param_count_check(arch, target, model_kwargs) -> dict:
    ours = count_parameters(get_model(arch, **model_kwargs))
    v1_path = C.ONLAB_MODELS_DIR / f"{arch}_best.pt"
    v1 = torch.load(v1_path, map_location='cpu', weights_only=False)
    v1_count = count_parameters(get_model(v1['model_class'], **v1['model_kwargs']))
    hpo_pcc_path = HPO_P30_DIR / f"param_count_check_{arch}.json"
    hpo_pcc = read_json(hpo_pcc_path)
    hpo_ck_path = HPO_MODELS_DIR / f"{arch}_hpo_p30_best.pt"
    hpo_ck = torch.load(hpo_ck_path, map_location='cpu', weights_only=False)
    uni_path = UNIFORM_RESULTS_DIR / f"metrics_{arch}_{target}_p30.json"
    uni_count = read_json(uni_path)['param_count'] if uni_path.exists() else None
    size_kw = {k: v for k, v in model_kwargs.items() if k != 'dropout'}
    res = {
        'arch': arch, 'target': target, 'model_kwargs': model_kwargs, 'param_count': ours,
        'onlab_v1_checkpoint': rel(v1_path), 'onlab_v1_param_count': v1_count,
        'hpo_p30_param_count_check': rel(hpo_pcc_path), 'hpo_p30_param_count': hpo_pcc['hpo_param_count'],
        'hpo_p30_checkpoint': rel(hpo_ck_path), 'hpo_p30_model_kwargs': hpo_ck['model_kwargs'],
        'uniform_p30_metrics': rel(uni_path), 'uniform_p30_param_count': uni_count,
        'size_kwargs_equal_baseline_arch_kwargs': size_kw == BASELINE_ARCH_KWARGS[arch],
        'model_kwargs_equal_hpo_p30_checkpoint': hpo_ck['model_kwargs'] == model_kwargs,
    }
    res['param_count_match'] = bool(ours == v1_count == hpo_pcc['hpo_param_count']
                                    and (uni_count is None or ours == uni_count))
    res['all_ok'] = bool(res['param_count_match'] and res['size_kwargs_equal_baseline_arch_kwargs']
                         and res['model_kwargs_equal_hpo_p30_checkpoint'])
    if not res['all_ok']:
        raise RuntimeError(f"[{arch}] Kapacitas-kontroll serul, leallas: {res}")
    return res


def use_uniform_premium_edges() -> dict:
    src = UNIFORM_RESULTS_DIR / "premium_bucket_edges.json"
    stored = read_json(src)
    df = C.load_am_split('test')
    recomputed = C.premium_bucket_edges((df['early_exercise_premium'] / df['K']).to_numpy())
    equal = [float(a) for a in stored['edges']] == [float(b) for b in recomputed]
    if not equal:
        raise RuntimeError(f"[premium] A uniform vodorhatarok elternek az ujraszamolttol: "
                           f"{stored['edges']} vs {recomputed}")
    TU._EDGES = [float(a) for a in stored['edges']]
    out = dict(stored)
    out.update({'copied_from': rel(src), 'recomputed_bitwise_equal': equal})
    write_json(P.results_dir / "premium_bucket_edges.json", out)
    return out


def run_one(arch, target, max_epochs_override=None):
    run = run_name(arch, target)
    t_start = time.time()
    hp = load_transfer_hp(arch)
    max_epochs = max_epochs_override if max_epochs_override is not None else hp['max_epochs']
    kw = model_kwargs_for(arch, hp)
    spec = TARGETS[target]
    TU.log(run, f"=== {run} ({CLASS_NAME[arch]}) -- cel: {spec['formula']}, patience={hp['patience']}, "
                f"max_epochs={max_epochs}, lr={hp['lr']!r}, wd={hp['weight_decay']!r}, "
                f"dropout={hp['dropout']!r}, batch={hp['batch_size']}, seed={hp['seed']}, "
                f"forras={hp['source_json']}, model_kwargs={kw} ===")
    pcc = param_count_check(arch, target, kw)
    write_json(P.results_dir / f"param_count_check_{run}.json", pcc)

    train_ds, val_ds = TU.get_ds('train', target), TU.get_ds('val', target)
    ckpt = guard_path(P.ckpt(run))
    hyper = {'lr': hp['lr'], 'weight_decay': hp['weight_decay'], 'dropout': hp['dropout'],
             'batch_size': hp['batch_size']}
    extra = {'target': target, 'target_definition': spec['formula'],
             'data': rel(C.AM_DATA_DIR), 'patience': hp['patience'], 'max_epochs': max_epochs,
             'seed': hp['seed'], 'hyperparameters': hyper,
             'hyperparameter_source_json': hp['source_json'],
             'experiment': 'HPO transfer (european call TPE p30 -> put)'}

    torch.manual_seed(hp['seed'])
    model = get_model(arch, **kw)
    t0 = time.time()
    hist = C.train_loop(model, train_ds, val_ds, output_dir=P.models_dir, model_name=run,
                        model_class=arch, model_kwargs=kw, feature_cols=FEATURE_COLS,
                        target_col=spec['target_col'], batch_size=hp['batch_size'],
                        max_epochs=max_epochs, lr=hp['lr'], weight_decay=hp['weight_decay'],
                        patience=hp['patience'], device=DEVICE, extra_meta=extra,
                        checkpoint_path=ckpt)
    train_seconds = time.time() - t0
    n_run = len(hist['val_loss'])
    pd.DataFrame({'epoch': range(1, n_run + 1), 'train_loss': hist['train_loss'],
                  'val_loss': hist['val_loss'], 'lr': hist['lr'],
                  'epoch_seconds': hist['epoch_seconds']}).to_csv(
        guard_path(P.results_dir / f"training_history_{run}.csv"), index=False)
    TU.log(run, f"Tanitas kesz ({train_seconds:.1f}s): best_epoch={hist['best_epoch']}, "
                f"futott={n_run}, best_val_loss={hist['best_val_loss']:.8e}")

    ev = TU.evaluate_run(run, target, ckpt)
    out = {
        'run': run, 'arch': arch, 'model_class': CLASS_NAME[arch], 'target': target,
        'target_col': spec['target_col'], 'target_definition': spec['formula'],
        'level': 'transfer_p30', 'patience': hp['patience'], 'max_epochs': max_epochs,
        'seed': hp['seed'], 'hyperparameters': hyper,
        'hyperparameter_source_json': hp['source_json'],
        'hyperparameter_source_search': hp['source_search'],
        'source_european_call_p30_val_loss': hp['source_european_call_p30_val_loss'],
        'model_kwargs': kw, 'param_count': pcc['param_count'],
        'param_count_match': pcc['param_count_match'], 'param_count_check': pcc,
        'checkpoint': rel(ckpt),
        'data_files': {s: rel(C.am_split_path(s)) for s in C.SPLITS},
        'best_epoch': hist['best_epoch'], 'n_epochs_run': n_run,
        'hit_max_epochs': n_run >= max_epochs,
        'best_val_loss': hist['best_val_loss'],
        'train_seconds': train_seconds,
        'mean_epoch_seconds': float(np.mean(hist['epoch_seconds'])),
        **ev,
    }
    write_json(P.results_dir / f"metrics_{run}.json", out)
    tm = ev['test_metrics']
    TU.log(run, f"=== {run} KESZ ({time.time() - t_start:.1f}s): val={hist['best_val_loss']:.4e}, "
                f"test RMSE={tm['RMSE']:.4e}, R2={tm['R2']:.6f}, val ujraszamolva rel. elteres="
                f"{ev['val_recheck']['rel_diff']:.2e}, target_col={ev['checkpoint_target_col']} ===")
    return out


def main():
    global P
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', type=str, default=None)
    ap.add_argument('--archs', type=str, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--target', choices=['american', 'european', 'both'], default='both')
    ap.add_argument('--skip-done', action='store_true')
    ap.add_argument('--smoke', type=int, default=None,
                    help="Fustproba: max_epochs felulirasa (csak az american-nn/-en kivuli --out-root-tal).")
    ap.add_argument('--out-root', type=str, default=None)
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve() if args.out_root else AMNN_DIR
    if args.smoke is not None and C._is_within(out_root, AMNN_DIR):
        raise SystemExit("[smoke] A fustproba kimenete nem kerulhet az american-nn/ ala -- adj meg --out-root-ot.")
    C.set_allowed_roots(out_root)
    P = Paths(out_root)
    TU.P = P
    P.ensure()
    sys.stdout = TU._Tee(sys.stdout, P.logs_dir / "campaign_stdout.log")

    if args.summary_only:
        import summarize_transfer as S
        S.main(P)
        return

    pc = TU.pairing_check()
    if not pc['all_paired']:
        raise SystemExit("[parositas] A ket adathalmaz NEM paros -- a kampany leall.")
    edges = use_uniform_premium_edges()
    print(f"[premium] uniform vodorhatarok hasznalva, ujraszamolva bitre azonos: {edges['recomputed_bitwise_equal']}")

    if args.arch:
        archs = [args.arch]
    elif args.archs:
        archs = [a.strip() for a in args.archs.split(',') if a.strip()]
    else:
        archs = ARCH_ORDER
    targets = TARGET_ORDER if args.target == 'both' else [args.target]
    hps = {a: load_transfer_hp(a) for a in archs}

    meta_path = P.results_dir / "campaign_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {'invocations': []}
    meta.update({'env': C.env_meta(), 'experiment': 'HPO transfer',
                 'transferred_hyperparameters': hps,
                 'scheduler': "ReduceLROnPlateau(mode='min', factor=0.5, patience=5, min_lr=1e-6)",
                 'optimizer': 'Adam', 'loss': 'MSELoss', 'feature_cols': FEATURE_COLS,
                 'patience': PATIENCE, 'targets': {t: TARGETS[t]['formula'] for t in TARGET_ORDER},
                 'machine': {'processor': __import__('platform').processor()}})
    meta['invocations'].append({'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'argv': sys.argv[1:],
                                'max_epochs_override': args.smoke})
    write_json(meta_path, meta)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] HPO-atvitel indul. Device: {DEVICE} "
          f"({C.env_meta().get('gpu_name')}); kimenet: {P.out_root}; smoke={args.smoke}")
    runs = [(a, t) for a in archs for t in targets]
    print(f"Futtatando: {len(runs)} tanitas: {[run_name(*r) for r in runs]}")
    t0 = time.time()
    n_done = n_skip = n_err = 0
    for arch, target in runs:
        run = run_name(arch, target)
        if args.skip_done and (P.results_dir / f"metrics_{run}.json").exists() and P.ckpt(run).exists():
            print(f"[skip-done] {run} mar kesz, kihagyva.")
            n_skip += 1
            continue
        try:
            run_one(arch, target, args.smoke)
            n_done += 1
        except Exception:
            err = traceback.format_exc()
            with open(guard_path(P.logs_dir / f"{run}_ERROR.log"), 'a', encoding='utf-8') as f:
                f.write(err + "\n")
            print(f"[HIBA] {run}:\n{err}")
            n_err += 1
    meta = read_json(meta_path)
    meta['invocations'][-1].update({'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'wall_seconds': time.time() - t0, 'n_trained': n_done,
                                    'n_skipped': n_skip, 'n_errors': n_err})
    write_json(meta_path, meta)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Kampany kesz: {time.time() - t0:.1f}s, "
          f"tanitva={n_done}, kihagyva={n_skip}, hiba={n_err}")

    import summarize_transfer as S
    S.main(P)


if __name__ == '__main__':
    main()
