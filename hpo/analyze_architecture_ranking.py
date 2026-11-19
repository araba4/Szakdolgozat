import sys
sys.dont_write_bytecode = True

import argparse
import json
import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HPO_DIR = Path(__file__).resolve().parent.parent / "hpo"
REPO_ROOT = HPO_DIR.parent
ONLAB_DIR = REPO_ROOT / "Önlab eredmények"
ONLAB_RESULTS = ONLAB_DIR / "results"
ONLAB_MODELS = ONLAB_DIR / "models"
ONLAB_DATA = ONLAB_DIR / "data"
TPE_DIR = HPO_DIR / "results"
TPE_P30_DIR = TPE_DIR / "p30"
RS_DIR = TPE_DIR / "random"
HPO_MODELS = HPO_DIR / "models"
AM_DIR = REPO_ROOT / "american-nn"
AM_RESULTS = AM_DIR / "results"
AM_MODELS = AM_DIR / "models"
OUT_DIR = TPE_DIR / "ranking"

sys.path.insert(0, str(ONLAB_DIR))

ARCH_ORDER = ['mlp', 'deep_mlp', 'dense_mlp', 'resnet', 'gelu_resnet', 'highway', 'finn']
CLASS_NAME = {'mlp': 'MLPPricer', 'deep_mlp': 'DeepMLPPricer', 'dense_mlp': 'DenseMLPPricer',
              'resnet': 'ResNetPricer', 'gelu_resnet': 'GELUResNetPricer',
              'highway': 'HighwayPricer', 'finn': 'FINNPricer'}
SHORT = {a: CLASS_NAME[a].replace('Pricer', '') for a in ARCH_ORDER}
HAS_V2 = {a: a != 'mlp' for a in ARCH_ORDER}

EU_LEVELS = ['onlab_v1', 'onlab_best', 'tpe_p10', 'tpe_p30', 'rs_p30']
EU_LABEL = {'onlab_v1': 'Önlab v1', 'onlab_best': 'Önlab legjobb', 'tpe_p10': 'TPE p10',
            'tpe_p30': 'TPE p30', 'rs_p30': 'RS p30'}
AM_LEVELS = ['uniform_p30', 'transfer_p30', 'retune_p30', 'r1']
AM_LABEL = {'uniform_p30': 'uniform p30', 'transfer_p30': 'transfer p30',
            'retune_p30': 'retune p30', 'r1': 'R1 (BS + NN)'}

TOL_TEST = 1e-5
TOL_VAL = 1e-4


def guard_path(path) -> Path:
    p = Path(path).resolve()
    try:
        p.relative_to(OUT_DIR.resolve())
    except ValueError:
        raise RuntimeError(f"guard_path: tiltott irasi cel (csak {rel_path(OUT_DIR)} ala szabad irni): "
                           f"{rel_path(p)}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_json(p: Path):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def write_json(obj, p: Path):
    with open(guard_path(p), 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def rel(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / abs(b) if b != 0 else abs(a - b)


def rel_path(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def onlab_history(arch: str, v2: bool):
    suffix = '_v2' if v2 else ''
    p = ONLAB_RESULTS / f"training_history_{CLASS_NAME[arch]}{suffix}.json"
    h = read_json(p)
    return dict(best_val_loss=h['best_val_loss'], best_epoch=h['best_epoch'],
                n_epochs_run=len(h['val_loss']), source=rel_path(p))


def ckpt_paths(arch: str) -> dict:
    d = {'onlab_v1': ONLAB_MODELS / f"{arch}_best.pt",
         'tpe_p10': HPO_MODELS / f"{arch}_hpo_best.pt",
         'tpe_p30': HPO_MODELS / f"{arch}_hpo_p30_best.pt",
         'rs_p30': HPO_MODELS / "random" / f"{arch}_random_p30_best.pt"}
    if HAS_V2[arch]:
        d['onlab_v2'] = ONLAB_MODELS / f"{arch}_v2_best.pt"
    return d


def load_saved_eu(arch: str) -> dict:
    tpe = read_json(TPE_DIR / f"best_params_{arch}.json")
    p30 = read_json(TPE_P30_DIR / f"best_params_{arch}.json")
    rs = read_json(RS_DIR / f"best_params_{arch}.json")
    h1 = onlab_history(arch, v2=False)
    h2 = onlab_history(arch, v2=True) if HAS_V2[arch] else None

    best_ref = 'v2' if (h2 is not None and h2['best_val_loss'] < h1['best_val_loss']) else 'v1'

    S = {}
    S['onlab_v1'] = dict(
        test_rmse={'hpo/results/best_params.baseline_test_metrics': tpe['baseline_test_metrics']['RMSE'],
                   'hpo/results/p30/best_params.baseline_test_metrics': p30['baseline_test_metrics']['RMSE']},
        val_mse={'Önlab training_history.best_val_loss': h1['best_val_loss'],
                 'hpo/results/best_params.baseline_val_loss': tpe['baseline_val_loss']},
        best_epoch=h1['best_epoch'], n_epochs_run=h1['n_epochs_run'])
    if h2 is not None:
        S['onlab_v2'] = dict(
            test_rmse={'hpo/results/p30/best_params.onlab_v2_test_metrics': p30['onlab_v2_test_metrics']['RMSE']},
            val_mse={'Önlab training_history_v2.best_val_loss': h2['best_val_loss']},
            best_epoch=h2['best_epoch'], n_epochs_run=h2['n_epochs_run'])
    S['onlab_best'] = dict(
        ref=best_ref, ref_saved=p30['onlab_best_ref'],
        test_rmse={'hpo/results/p30/best_params.onlab_best_test_metrics': p30['onlab_best_test_metrics']['RMSE']},
        val_mse={'hpo/results/p30/best_params.onlab_best_val_loss': p30['onlab_best_val_loss']},
        best_epoch=p30['onlab_best_epoch'], n_epochs_run=p30['onlab_best_n_epochs_run'])
    S['tpe_p10'] = dict(
        test_rmse={'hpo/results/best_params.hpo_test_metrics': tpe['hpo_test_metrics']['RMSE'],
                   'hpo/results/p30/best_params.p10_test_metrics': p30['p10_test_metrics']['RMSE']},
        val_mse={'hpo/results/best_params.final_retrain_best_val_loss': tpe['final_retrain_best_val_loss'],
                 'hpo/results/p30/best_params.p10_best_val_loss': p30['p10_best_val_loss']},
        best_epoch=p30['p10_best_epoch'], n_epochs_run=p30['p10_n_epochs_run'])
    S['tpe_p30'] = dict(
        test_rmse={'hpo/results/p30/best_params.hpo_test_metrics': p30['hpo_test_metrics']['RMSE'],
                   'hpo/results/random/best_params.tpe_p30_test_metrics': rs['tpe_p30_test_metrics']['RMSE']},
        val_mse={'hpo/results/p30/best_params.final_retrain_best_val_loss': p30['final_retrain_best_val_loss'],
                 'hpo/results/random/best_params.tpe_p30_best_val_loss': rs['tpe_p30_best_val_loss']},
        best_epoch=p30['final_retrain_best_epoch'], n_epochs_run=p30['final_retrain_n_epochs_run'])
    S['rs_p30'] = dict(
        test_rmse={'hpo/results/random/best_params.final_test_metrics': rs['final_test_metrics']['RMSE']},
        val_mse={'hpo/results/random/best_params.final_retrain_best_val_loss': rs['final_retrain_best_val_loss']},
        best_epoch=rs['final_retrain_best_epoch'], n_epochs_run=rs['final_retrain_n_epochs_run'])

    pc = {}
    for name, p in [('tpe_p10', TPE_DIR / f"param_count_check_{arch}.json"),
                    ('tpe_p30', TPE_P30_DIR / f"param_count_check_{arch}.json"),
                    ('rs_p30', RS_DIR / f"param_count_check_{arch}.json")]:
        j = read_json(p)
        pc[f"{rel_path(p)}.baseline_param_count"] = j['baseline_param_count']
        key = 'random_param_count' if name == 'rs_p30' else 'hpo_param_count'
        pc[f"{rel_path(p)}.{key}"] = j[key]
        pc[f"{rel_path(p)}.match"] = j['match']
    pc['hpo/results/best_params.param_count'] = tpe['param_count']
    pc['hpo/results/p30/best_params.param_count'] = p30['param_count']
    pc['hpo/results/random/best_params.param_count'] = rs['param_count']
    return dict(S=S, pc=pc, best_ref=best_ref, tpe=tpe, p30=p30, rs=rs)


def recompute_eu(device: str) -> dict:
    import torch
    from src.evaluate import evaluate_model, load_model
    from src.model import count_parameters
    test_path = str(ONLAB_DATA / "test.parquet")
    val_path = str(ONLAB_DATA / "val.parquet")
    out = {}
    for arch in ARCH_ORDER:
        out[arch] = {}
        for lvl, ck in ckpt_paths(arch).items():
            t0 = time.time()
            rt = evaluate_model(str(ck), test_path, device=device, verbose=False)
            rv = evaluate_model(str(ck), val_path, device=device, verbose=False)
            model, meta = load_model(str(ck), device='cpu')
            out[arch][lvl] = dict(
                checkpoint=rel_path(ck),
                test_rmse=rt['metrics']['RMSE'], test_mae=rt['metrics']['MAE'],
                test_r2=rt['metrics']['R2'], test_max_error=rt['metrics']['max_error'],
                val_mse=rv['metrics']['RMSE'] ** 2,
                n_test=int(len(rt['y_true'])), n_val=int(len(rv['y_true'])),
                param_count=int(count_parameters(model)),
                model_class=meta['model_class'], model_kwargs=meta['model_kwargs'],
                target_col=meta['target_col'], ckpt_val_loss=meta['val_loss'],
                ckpt_best_epoch=meta['best_epoch'], seconds=time.time() - t0)
            print(f"[inf] {arch:12s} {lvl:9s} test RMSE {out[arch][lvl]['test_rmse']:.6e}  "
                  f"val MSE {out[arch][lvl]['val_mse']:.6e}  P={out[arch][lvl]['param_count']}")
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    return out


def primary(dct: dict) -> float:
    return next(iter(dct.values()))


def build_eu_table(L: dict, RC: dict) -> pd.DataFrame:
    rows = []
    for arch in ARCH_ORDER:
        S = L[arch]['S']
        r = dict(arch=arch, model_class=CLASS_NAME[arch],
                 param_count=RC[arch]['tpe_p30']['param_count'],
                 onlab_best_ref=L[arch]['best_ref'])
        for lvl in EU_LEVELS:
            r[f"{lvl}_test_rmse"] = primary(S[lvl]['test_rmse'])
            r[f"{lvl}_val_mse"] = primary(S[lvl]['val_mse'])
            r[f"{lvl}_best_epoch"] = S[lvl]['best_epoch']
            r[f"{lvl}_n_epochs_run"] = S[lvl]['n_epochs_run']
        rows.append(r)
    df = pd.DataFrame(rows)
    for lvl in EU_LEVELS:
        df[f"{lvl}_rank_test_rmse"] = df[f"{lvl}_test_rmse"].rank(method='min').astype(int)
        df[f"{lvl}_rank_val_mse"] = df[f"{lvl}_val_mse"].rank(method='min').astype(int)
    df['param_rank_small_to_large'] = df['param_count'].rank(method='min').astype(int)
    for lvl in ['tpe_p10', 'tpe_p30', 'rs_p30']:
        for base in ['onlab_v1', 'onlab_best']:
            for m in ['test_rmse', 'val_mse']:
                df[f"impr_pct_{m}_{lvl}_vs_{base}"] = 100.0 * (1 - df[f"{lvl}_{m}"] / df[f"{base}_{m}"])
    cols = (['arch', 'model_class', 'param_count', 'param_rank_small_to_large', 'onlab_best_ref']
            + [f"{lvl}_{c}" for lvl in EU_LEVELS for c in
               ('test_rmse', 'rank_test_rmse', 'val_mse', 'rank_val_mse', 'best_epoch', 'n_epochs_run')]
            + [c for c in df.columns if c.startswith('impr_pct_')])
    return df[cols]


def kendall(a, b) -> dict:
    t, p = stats.kendalltau(a, b)
    return dict(tau=float(t), p_value=float(p))


def order_of(df: pd.DataFrame, col: str) -> list:
    return list(df.sort_values(col)['arch'])


def tau_block(df: pd.DataFrame, levels: list, label: dict) -> dict:
    out = dict(n=int(len(df)), pairs=[], within_level_test_vs_val=[], param_count_vs_error=[], orders={})
    for lvl in levels:
        out['orders'][lvl] = dict(test_rmse=order_of(df, f"{lvl}_test_rmse"),
                                  val_mse=order_of(df, f"{lvl}_val_mse"))
    for m in ['test_rmse', 'val_mse']:
        for a, b in combinations(levels, 2):
            k = kendall(df[f"{a}_{m}"], df[f"{b}_{m}"])
            out['pairs'].append(dict(metric=m, a=a, b=b, a_label=label[a], b_label=label[b], **k))
    for lvl in levels:
        k = kendall(df[f"{lvl}_test_rmse"], df[f"{lvl}_val_mse"])
        out['within_level_test_vs_val'].append(dict(level=lvl, **k))
    for lvl in levels:
        k = kendall(df['param_count'], df[f"{lvl}_test_rmse"])
        rho, prho = stats.spearmanr(df['param_count'], df[f"{lvl}_test_rmse"])
        out['param_count_vs_error'].append(dict(level=lvl, metric='test_rmse', kendall_tau=k['tau'],
                                                kendall_p=k['p_value'], spearman_rho=float(rho),
                                                spearman_p=float(prho)))
    return out


def build_am_table() -> tuple:
    lv = pd.read_csv(AM_RESULTS / "retune" / "levels_table.csv", float_precision='round_trip')
    lv = lv[lv.target == 'american']
    dr = pd.read_csv(AM_RESULTS / "residual" / "direct_vs_residual.csv", float_precision='round_trip')
    rows = []
    for arch in ARCH_ORDER:
        r = dict(arch=arch, model_class=CLASS_NAME[arch])
        pcs = {}
        for lvl in ['uniform_p30', 'transfer_p30', 'retune_p30']:
            x = lv[(lv.arch == arch) & (lv.level == lvl)]
            assert len(x) == 1, (arch, lvl)
            x = x.iloc[0]
            r[f"{lvl}_test_rmse"] = x.test_rmse
            r[f"{lvl}_val_mse"] = x.best_val_mse
            r[f"{lvl}_test_usd_rmse"] = x.test_usd_rmse
            r[f"{lvl}_best_epoch"] = int(x.best_epoch)
            r[f"{lvl}_n_epochs_run"] = int(x.n_epochs_run)
            pcs[f"levels_table.{lvl}"] = int(x.param_count)
        y = dr[dr.arch == arch]
        assert len(y) == 1, arch
        y = y.iloc[0]
        r['r1_test_rmse'] = y.residual_test_rmse
        r['r1_val_mse'] = y.residual_best_val_mse_price
        r['r1_test_usd_rmse'] = y.residual_test_usd_rmse
        r['r1_best_epoch'] = int(y.residual_best_epoch)
        r['r1_n_epochs_run'] = int(y.residual_n_epochs_run)
        pcs['direct_vs_residual.param_count'] = int(y.param_count)
        rpc = read_json(AM_RESULTS / "residual" / f"param_count_check_{arch}_residual_retune_p30.json")
        pcs['residual/param_count_check.param_count'] = int(rpc['param_count'])
        r['param_count'] = pcs['levels_table.retune_p30']
        r['_pcs'] = pcs
        rows.append(r)
    df = pd.DataFrame(rows)
    for lvl in AM_LEVELS:
        df[f"{lvl}_rank_test_rmse"] = df[f"{lvl}_test_rmse"].rank(method='min').astype(int)
        df[f"{lvl}_rank_val_mse"] = df[f"{lvl}_val_mse"].rank(method='min').astype(int)
    df['param_rank_small_to_large'] = df['param_count'].rank(method='min').astype(int)
    pcs = dict(zip(df['arch'], df.pop('_pcs')))
    cols = (['arch', 'model_class', 'param_count', 'param_rank_small_to_large']
            + [f"{lvl}_{c}" for lvl in AM_LEVELS for c in
               ('test_rmse', 'rank_test_rmse', 'val_mse', 'rank_val_mse', 'test_usd_rmse',
                'best_epoch', 'n_epochs_run')])
    return df[cols], pcs


def am_checkpoint_param_counts() -> dict:
    from src.model import get_model, count_parameters
    import torch
    out = {}
    paths = {'uniform_p30': AM_MODELS / "uniform" / "{a}_american_p30_best.pt",
             'transfer_p30': AM_MODELS / "transfer" / "{a}_american_transfer_p30_best.pt",
             'retune_p30': AM_MODELS / "retune" / "{a}_american_retune_p30_best.pt",
             'r1': AM_MODELS / "residual" / "{a}_residual_retune_p30_best.pt"}
    for arch in ARCH_ORDER:
        out[arch] = {}
        for lvl, tmpl in paths.items():
            p = Path(str(tmpl).format(a=arch))
            ck = torch.load(p, map_location='cpu', weights_only=False)
            m = get_model(ck['model_class'], **ck['model_kwargs'])
            m.load_state_dict(ck['state_dict'])
            n_sd = int(sum(v.numel() for k, v in ck['state_dict'].items()
                           if not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))))
            out[arch][lvl] = dict(checkpoint=rel_path(p), count_parameters=int(count_parameters(m)),
                                  state_dict_numel_no_bn_buffers=n_sd,
                                  model_kwargs={k: v for k, v in ck['model_kwargs'].items() if k != 'dropout'})
    return out


def check_am_ranking_csv(am: pd.DataFrame) -> dict:
    rk = pd.read_csv(AM_RESULTS / "retune" / "ranking.csv", float_precision='round_trip')
    n = 0
    mism = []
    for lvl in ['uniform_p30', 'transfer_p30', 'retune_p30']:
        for m, col in [('test_rmse', 'rank_test_rmse'), ('best_val_mse', 'rank_val_mse')]:
            sub = rk[(rk.target == 'american') & (rk.level == lvl) & (rk.metric == m)]
            for _, s in sub.iterrows():
                mine = int(am.loc[am.arch == s.arch, f"{lvl}_{col}"].iloc[0])
                n += 1
                if mine != int(s['rank']):
                    mism.append(dict(level=lvl, metric=m, arch=s.arch, mine=mine, saved=int(s['rank'])))
    return dict(n_checked=n, n_mismatch=len(mism), mismatches=mism, ok=(len(mism) == 0))


def check_eu_call_ranking_csv(eu: pd.DataFrame) -> dict:
    rk = pd.read_csv(AM_RESULTS / "retune" / "ranking.csv", float_precision='round_trip')
    n = 0
    mism = []
    for m, col in [('test_rmse', 'tpe_p30_rank_test_rmse'), ('best_val_mse', 'tpe_p30_rank_val_mse')]:
        sub = rk[(rk.target == 'european_call') & (rk.level == 'hpo_p30') & (rk.metric == m)]
        for _, s in sub.iterrows():
            mine = int(eu.loc[eu.arch == s.arch, col].iloc[0])
            n += 1
            if mine != int(s['rank']):
                mism.append(dict(metric=m, arch=s.arch, mine=mine, saved=int(s['rank'])))
    return dict(n_checked=n, n_mismatch=len(mism), mismatches=mism, ok=(len(mism) == 0))


def verify_eu(L: dict, RC: dict) -> dict:
    rows = []
    ok = True
    max_rel_test = max_rel_val = 0.0
    for arch in ARCH_ORDER:
        S = L[arch]['S']
        lvls = ['onlab_v1', 'tpe_p10', 'tpe_p30', 'rs_p30'] + (['onlab_v2'] if HAS_V2[arch] else [])
        for lvl in lvls:
            rc = RC[arch][lvl]
            for src, v in S[lvl]['test_rmse'].items():
                d = rel(rc['test_rmse'], v)
                max_rel_test = max(max_rel_test, d)
                good = d <= TOL_TEST
                ok &= good
                rows.append(dict(arch=arch, level=lvl, quantity='test_rmse', source=src, saved=v,
                                 recomputed=rc['test_rmse'], rel_diff=d, ok=good))
            for src, v in S[lvl]['val_mse'].items():
                d = rel(rc['val_mse'], v)
                max_rel_val = max(max_rel_val, d)
                good = d <= TOL_VAL
                ok &= good
                rows.append(dict(arch=arch, level=lvl, quantity='val_mse', source=src, saved=v,
                                 recomputed=rc['val_mse'], rel_diff=d, ok=good))
            d = rel(rc['val_mse'], rc['ckpt_val_loss'])
            max_rel_val = max(max_rel_val, d)
            good = d <= TOL_VAL
            ok &= good
            rows.append(dict(arch=arch, level=lvl, quantity='val_mse', source=f"{rc['checkpoint']}:val_loss",
                             saved=rc['ckpt_val_loss'], recomputed=rc['val_mse'], rel_diff=d, ok=good))
        ref = L[arch]['best_ref']
        rc = RC[arch]['onlab_v2' if ref == 'v2' else 'onlab_v1']
        for q in ['test_rmse', 'val_mse']:
            for src, v in S['onlab_best'][q].items():
                d = rel(rc[q], v)
                tol = TOL_TEST if q == 'test_rmse' else TOL_VAL
                good = d <= tol
                ok &= good
                rows.append(dict(arch=arch, level=f"onlab_best ({ref})", quantity=q, source=src, saved=v,
                                 recomputed=rc[q], rel_diff=d, ok=good))
    ref_check = {a: dict(mine=L[a]['best_ref'], saved=L[a]['S']['onlab_best']['ref_saved'],
                         ok=L[a]['best_ref'] == L[a]['S']['onlab_best']['ref_saved']) for a in ARCH_ORDER}
    ok &= all(v['ok'] for v in ref_check.values())

    rvt = pd.read_csv(RS_DIR / "comparison" / "random_vs_tpe_table.csv", float_precision='round_trip')
    rvt_rows = []
    for _, s in rvt.iterrows():
        a = s.arch
        for col, lvl, q in [('onlab_test_rmse', 'onlab_best', 'test_rmse'), ('onlab_val_mse', 'onlab_best', 'val_mse'),
                            ('tpe_p30_test_rmse', 'tpe_p30', 'test_rmse'), ('tpe_p30_val_mse', 'tpe_p30', 'val_mse'),
                            ('rs_p30_test_rmse', 'rs_p30', 'test_rmse'), ('rs_p30_val_mse', 'rs_p30', 'val_mse')]:
            d = rel(primary(L[a]['S'][lvl][q]), s[col])
            rvt_rows.append(dict(arch=a, column=col, rel_diff=d, ok=d == 0.0))
    ok &= all(r['ok'] for r in rvt_rows)

    pc_rows = []
    for arch in ARCH_ORDER:
        counts = {lvl: RC[arch][lvl]['param_count'] for lvl in RC[arch]}
        saved = {k: v for k, v in L[arch]['pc'].items() if not k.endswith('.match')}
        matches = {k: v for k, v in L[arch]['pc'].items() if k.endswith('.match')}
        vals = set(counts.values()) | set(saved.values())
        good = len(vals) == 1 and all(matches.values())
        ok &= good
        pc_rows.append(dict(arch=arch, count_parameters_per_checkpoint=counts, saved=saved, match_flags=matches,
                            param_count=int(next(iter(vals))) if len(vals) == 1 else None, ok=good))

    return dict(
        ok=bool(ok), tol_test_rel=TOL_TEST, tol_val_rel=TOL_VAL,
        max_rel_diff_test_rmse=max_rel_test, max_rel_diff_val_mse=max_rel_val,
        n_value_checks=len(rows), n_failed=sum(not r['ok'] for r in rows),
        value_checks=rows, onlab_best_ref_check=ref_check,
        random_vs_tpe_table_check=dict(n=len(rvt_rows), n_failed=sum(not r['ok'] for r in rvt_rows),
                                       max_rel_diff=max(r['rel_diff'] for r in rvt_rows)),
        param_count_check=pc_rows)


def _sci(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if v == 0:
        return "0"
    mant, exp = f"{v:.2e}".split("e")
    return f"{mant}e{int(exp)}"


def _f(v, nd=2):
    return '–' if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _chg(impr):
    return '–' if impr is None else f"{-impr:+.1f}%".replace("-", "−")


def _p(n):
    return f"{int(n):,}".replace(',', ' ')


MD_EU_LABEL = {'onlab_v1': 'kiindulási v1', 'onlab_best': 'kiindulási legjobb', 'tpe_p10': 'TPE p10',
               'tpe_p30': 'TPE p30', 'rs_p30': 'RS p30'}
MD_AM_LABEL = AM_LABEL


def capacity_rows(eu: pd.DataFrame) -> list:
    out = []
    for arch in ['dense_mlp', 'highway']:
        r = eu[eu.arch == arch].iloc[0]
        out.append(dict(
            arch=arch, param_count=int(r.param_count),
            v1_test=r.onlab_v1_test_rmse, v1_val=r.onlab_v1_val_mse,
            v1_rank_test=int(r.onlab_v1_rank_test_rmse), v1_rank_val=int(r.onlab_v1_rank_val_mse),
            **{f"{lvl}_{k}": r[f"{lvl}_{c}"] for lvl in ['tpe_p10', 'tpe_p30', 'rs_p30']
               for k, c in [('test', 'test_rmse'), ('val', 'val_mse'), ('rank_test', 'rank_test_rmse'),
                            ('rank_val', 'rank_val_mse')]},
            **{f"impr_{m}_{lvl}": r[f"impr_pct_{m}_{lvl}_vs_onlab_v1"] for m in ['test_rmse', 'val_mse']
               for lvl in ['tpe_p10', 'tpe_p30', 'rs_p30']}))
    return out


def rank_table_md(df: pd.DataFrame, levels: list, label: dict, metric: str) -> list:
    head = "| Architektúra | Paraméter | " + " | ".join(label[l] for l in levels) + " |"
    sep = "|---|---:|" + "---:|" * len(levels)
    lines = [head, sep]
    rk = 'rank_test_rmse' if metric == 'test_rmse' else 'rank_val_mse'
    for _, r in df.iterrows():
        cells = [f"{_sci(r[f'{l}_{metric}'])} ({int(r[f'{l}_{rk}'])}.)" for l in levels]
        lines.append(f"| {CLASS_NAME[r.arch]} | {_p(r.param_count)} | " + " | ".join(cells) + " |")
    return lines


def tau_matrix_md(T: dict, levels: list, label: dict, metric: str) -> list:
    tau = {(p['a'], p['b']): p for p in T['pairs'] if p['metric'] == metric}
    lines = ["| | " + " | ".join(label[l] for l in levels) + " |", "|---|" + "---:|" * len(levels)]
    for a in levels:
        cells = []
        for b in levels:
            if a == b:
                cells.append('1')
            else:
                p = tau.get((a, b)) or tau.get((b, a))
                cells.append(f"{_f(p['tau'], 2)} (p = {_f(p['p_value'], 3)})")
        lines.append(f"| {label[a]} | " + " | ".join(cells) + " |")
    return lines


def write_summary(eu, Teu, am, Tam, V, Vam, cap):
    L = []
    E, M = MD_EU_LABEL, MD_AM_LABEL

    L += ["## Európai call: teszt-RMSE szintenként (rang)", ""]
    L += rank_table_md(eu, EU_LEVELS, E, 'test_rmse')
    L += ["", "## Európai call: val MSE szintenként (rang)", ""]
    L += rank_table_md(eu, EU_LEVELS, E, 'val_mse')
    L += ["", "## Európai call: Kendall τ a szintpárok között, teszt-RMSE", ""]
    L += tau_matrix_md(Teu, EU_LEVELS, E, 'test_rmse')
    L += ["", "## Európai call: Kendall τ a szintpárok között, val MSE", ""]
    L += tau_matrix_md(Teu, EU_LEVELS, E, 'val_mse')
    L += ["", "## Európai call: Kendall τ a teszt-RMSE és a val MSE rangsora között, szintenként", ""]
    L += ["| Szint | τ | p |", "|---|---:|---:|"]
    for w in Teu['within_level_test_vs_val']:
        L.append(f"| {E[w['level']]} | {_f(w['tau'], 2)} | {_f(w['p_value'], 3)} |")

    L += ["", "## DenseMLP és Highway: a kiindulási v1 és a hangolt szintek", ""]
    L.append("| Architektúra | Paraméter | Teszt-RMSE, kiindulási v1 (rang) | Teszt-RMSE, TPE p10 (rang) | "
             "Teszt-RMSE, TPE p30 (rang) | Teszt-RMSE, RS p30 (rang) | "
             "Teszt-RMSE változása, TPE p10 / TPE p30 / RS p30 | Val MSE változása, TPE p10 / TPE p30 / RS p30 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in cap:
        L.append(f"| {CLASS_NAME[c['arch']]} | {_p(c['param_count'])} | {_sci(c['v1_test'])} ({c['v1_rank_test']}.) | "
                 f"{_sci(c['tpe_p10_test'])} ({c['tpe_p10_rank_test']}.) | "
                 f"{_sci(c['tpe_p30_test'])} ({c['tpe_p30_rank_test']}.) | "
                 f"{_sci(c['rs_p30_test'])} ({c['rs_p30_rank_test']}.) | "
                 f"{_chg(c['impr_test_rmse_tpe_p10'])} / {_chg(c['impr_test_rmse_tpe_p30'])} / "
                 f"{_chg(c['impr_test_rmse_rs_p30'])} | "
                 f"{_chg(c['impr_val_mse_tpe_p10'])} / {_chg(c['impr_val_mse_tpe_p30'])} / "
                 f"{_chg(c['impr_val_mse_rs_p30'])} |")

    L += ["", "## Amerikai put: teszt-RMSE szintenként (rang)", ""]
    L += rank_table_md(am, AM_LEVELS, M, 'test_rmse')
    L += ["", "## Amerikai put: val MSE szintenként (rang)", ""]
    L += rank_table_md(am, AM_LEVELS, M, 'val_mse')
    L += ["", "## Amerikai put: Kendall τ a szintpárok között, teszt-RMSE", ""]
    L += tau_matrix_md(Tam, AM_LEVELS, M, 'test_rmse')

    L += ["", "## Paraméterszám és teszt-RMSE", ""]
    L += ["| Szint | Kendall τ | p | Spearman ρ | p |", "|---|---:|---:|---:|---:|"]
    for T, lab, pre in [(Teu, E, 'európai call'), (Tam, M, 'amerikai put')]:
        for r in T['param_count_vs_error']:
            L.append(f"| {pre}, {lab[r['level']]} | {_f(r['kendall_tau'], 2)} | {_f(r['kendall_p'], 3)} | "
                     f"{_f(r['spearman_rho'], 2)} | {_f(r['spearman_p'], 3)} |")

    L += ["", "## Verifikáció", ""]
    L += ["| Ellenőrzés | Összevetés | Eltérés | Legnagyobb relatív eltérés |", "|---|---:|---:|---:|"]
    L.append(f"| Európai checkpointok (34): újraszámolt és mentett teszt-RMSE és val MSE | "
             f"{V['n_value_checks']} | {V['n_failed']} | teszt-RMSE {_sci(V['max_rel_diff_test_rmse'])}, "
             f"val MSE {_sci(V['max_rel_diff_val_mse'])} |")
    rc = V['onlab_best_ref_check']
    L.append(f"| A kiindulási legjobb választása (v1 vagy v2) és az `onlab_best_ref` mező | {len(rc)} | "
             f"{sum(not v['ok'] for v in rc.values())} | – |")
    t = V['random_vs_tpe_table_check']
    L.append(f"| `hpo/results/random/comparison/random_vs_tpe_table.csv` | {t['n']} | {t['n_failed']} | "
             f"{_sci(t['max_rel_diff'])} |")
    n_pc_fail = (sum(not r['ok'] for r in V['param_count_check'])
                 + sum(not r['ok'] for r in Vam['param_count_check']))
    L.append(f"| Paraméterszám: `count_parameters` a 34 európai és a 28 amerikai checkpointon, a mentett "
             f"értékekkel (architektúránként) | {len(V['param_count_check'])} | {n_pc_fail} | – |")
    L.append(f"| Amerikai rangok és az `american-nn/results/retune/ranking.csv` | "
             f"{Vam['ranking_csv']['n_checked']} | {Vam['ranking_csv']['n_mismatch']} | – |")
    L.append(f"| TPE p30 rangok és a `ranking.csv` `european_call` sorai | "
             f"{V['call_ranking_csv']['n_checked']} | {V['call_ranking_csv']['n_mismatch']} | – |")
    L.append("")
    with open(guard_path(OUT_DIR / "ranking_summary.md"), 'w', encoding='utf-8') as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpu', action='store_true', help='inferencia CPU-n (alapertelmezes: CUDA, ha van)')
    args = ap.parse_args()
    import torch
    device = 'cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda'
    t0 = time.time()

    L = {a: load_saved_eu(a) for a in ARCH_ORDER}
    RC = recompute_eu(device)

    eu = build_eu_table(L, RC)
    Teu = tau_block(eu, EU_LEVELS, EU_LABEL)
    V = verify_eu(L, RC)
    V['call_ranking_csv'] = check_eu_call_ranking_csv(eu)

    am, am_pcs = build_am_table()
    Tam = tau_block(am, AM_LEVELS, AM_LABEL)
    am_ck = am_checkpoint_param_counts()
    am_pc_rows = []
    am_pc_ok = True
    eu_pc = dict(zip(eu.arch, eu.param_count))
    for arch in ARCH_ORDER:
        vals = set(am_pcs[arch].values()) | {v['count_parameters'] for v in am_ck[arch].values()} \
            | {v['state_dict_numel_no_bn_buffers'] for v in am_ck[arch].values()}
        good = (len(vals) == 1 and next(iter(vals)) == eu_pc[arch])
        am_pc_ok &= good
        am_pc_rows.append(dict(arch=arch, saved=am_pcs[arch], checkpoints=am_ck[arch],
                               equal_to_european_param_count=good, ok=good))
    Vam = dict(ranking_csv=check_am_ranking_csv(am), param_count_check=am_pc_rows, ok=bool(am_pc_ok))
    V['param_ok_all'] = bool(all(r['ok'] for r in V['param_count_check']) and am_pc_ok)

    cap = capacity_rows(eu)

    eu.to_csv(guard_path(OUT_DIR / "european_ranking_table.csv"), index=False)
    am.to_csv(guard_path(OUT_DIR / "american_ranking_table.csv"), index=False)
    write_json(dict(note="Kendall tau-b (scipy.stats.kendalltau) a 7 architektura rangsorai kozott; "
                         "egy seed, tajekoztato. Elsodleges metrika: test_rmse.",
                    levels=EU_LEVELS, level_labels=EU_LABEL,
                    source="hpo/results/ranking/european_ranking_table.csv",
                    capacity_limit_refutation=cap, **Teu),
               OUT_DIR / "european_kendall_tau.json")
    write_json(dict(note="Kendall tau-b az amerikai put rangsorai kozott (uniform/transfer/retune p30, R1); "
                         "egy seed, tajekoztato.",
                    levels=AM_LEVELS, level_labels=AM_LABEL,
                    source="hpo/results/ranking/american_ranking_table.csv", **Tam),
               OUT_DIR / "american_kendall_tau.json")
    all_ok = bool(V['ok'] and V['call_ranking_csv']['ok'] and Vam['ok'] and Vam['ranking_csv']['ok'])
    write_json(dict(all_ok=all_ok, device=device, generated_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                    runtime_s=time.time() - t0,
                    european=V, american=Vam,
                    recomputed_european={a: RC[a] for a in ARCH_ORDER}),
               OUT_DIR / "verification.json")
    write_summary(eu, Teu, am, Tam, V, Vam, cap)

    print(f"[{'ok' if all_ok else 'HIBA'}] verifikacio: europai {V['n_value_checks']} osszevetes, "
          f"{V['n_failed']} elteres; max rel. teszt {V['max_rel_diff_test_rmse']:.1e}, val {V['max_rel_diff_val_mse']:.1e}; "
          f"parameterszam {'ok' if V['param_ok_all'] else 'HIBA'}")
    print(eu[['arch', 'param_count'] + [f"{l}_rank_test_rmse" for l in EU_LEVELS]].to_string(index=False))
    print(am[['arch', 'param_count'] + [f"{l}_rank_test_rmse" for l in AM_LEVELS]].to_string(index=False))
    print(f"[ok] Kimenetek: {rel_path(OUT_DIR)}  ({time.time() - t0:.1f} s)")
    if not all_ok:
        raise SystemExit("Verifikacio: elteres, ld. verification.json")


if __name__ == '__main__':
    main()
