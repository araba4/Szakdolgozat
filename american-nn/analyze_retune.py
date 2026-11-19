import sys
sys.dont_write_bytecode = True

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

AMNN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AMNN_DIR))

import common as C
from common import (ARCH_ORDER, CLASS_NAME, DEVICE, HAS_DROPOUT, guard_path, write_json,
                    read_json, rel, load_model)

TARGET_ORDER = ['american', 'european']
LEVELS = ['uniform_p30', 'transfer_p30', 'retune_p30']
TLAB = {'american': 'amerikai', 'european': 'európai'}
LLAB = {'uniform_p30': 'uniform p30', 'transfer_p30': 'transfer p30', 'retune_p30': 'retune p30'}

UNIFORM_DIR = C.AMNN_DIR / "results" / "uniform"
TRANSFER_DIR = C.AMNN_DIR / "results" / "transfer"
TRANSFER_MODELS_DIR = C.AMNN_DIR / "models" / "transfer"
HPO_DIR = C.REPO_ROOT / "hpo"
HPO_RESULTS = HPO_DIR / "results"
HPO_P30 = HPO_RESULTS / "p30"

CMP_COLS = ['best_val_mse', 'test_mse', 'test_rmse', 'test_mae', 'test_r2', 'test_max_error',
            'test_mape_pct_nem_fo_metrika', 'test_usd_rmse', 'test_usd_mae', 'test_usd_max_error',
            'best_epoch', 'n_epochs_run', 'hit_max_epochs', 'train_seconds', 'mean_epoch_seconds',
            'val_recheck_rel_diff', 'param_count', 'checkpoint_target_col']
HP_COLS = ['lr', 'weight_decay', 'dropout']

WD_MIN, WD_MAX = 1e-6, 1e-2
WD_BOTTOM_TENTH = 10 ** (np.log10(WD_MIN) + 0.1 * (np.log10(WD_MAX) - np.log10(WD_MIN)))

CRR_REL_UNCERTAINTY = 4e-4
CRR_TREE_EU_ERR_MEDIAN_USD = 0.0016


def run_name(arch, target):
    return f"{arch}_{target}_retune_p30"


def short(arch):
    return CLASS_NAME[arch].replace('Pricer', '')


def _e(v, nd=3):
    return f"{v:.{nd}e}" if isinstance(v, (int, float, np.floating)) and v == v else "n/a"


def _f(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float, np.floating)) and v == v else "n/a"


def _eq(a, b):
    na = a is None or (isinstance(a, float) and a != a)
    nb = b is None or (isinstance(b, float) and b != b)
    if na or nb:
        return na and nb
    return a == b


def load_retune(P):
    rows = {}
    for a in ARCH_ORDER:
        for t in TARGET_ORDER:
            f = P.results_dir / f"metrics_{run_name(a, t)}.json"
            if f.exists():
                rows[(a, t)] = read_json(f)
    bps = {a: read_json(P.results_dir / f"best_params_{a}.json") for a in ARCH_ORDER
           if (P.results_dir / f"best_params_{a}.json").exists()}
    return rows, bps


def uniform_src():
    df = pd.read_csv(UNIFORM_DIR / "comparison_table.csv", float_precision='round_trip')
    return df[df.patience == 30].copy()


def transfer_src():
    df = pd.read_csv(TRANSFER_DIR / "comparison_table.csv", float_precision='round_trip')
    return df[df.level == 'transfer_p30'].copy()


def call_tpe():
    out = {}
    for a in ARCH_ORDER:
        bp = read_json(HPO_P30 / f"best_params_{a}.json")
        out[a] = {'file': rel(HPO_P30 / f"best_params_{a}.json"), 'val_mse': bp['final_retrain_best_val_loss'],
                  'test_rmse': bp['hpo_test_metrics']['RMSE'], 'lr': bp['lr'],
                  'weight_decay': bp['weight_decay'], 'dropout': bp['dropout'],
                  'search_best_val_loss': bp['search_best_val_loss'],
                  'search_best_trial_number': bp['search_best_trial_number']}
    return out


def levels_table(P, rows):
    uni, tr = uniform_src(), transfer_src()
    recs = []
    for a in ARCH_ORDER:
        for t in TARGET_ORDER:
            u = uni[(uni.arch == a) & (uni.target == t)]
            if len(u):
                u = u.iloc[0]
                rec = {'arch': a, 'model_class': CLASS_NAME[a], 'target': t, 'level': 'uniform_p30'}
                rec.update({c: u[c] for c in CMP_COLS})
                rec.update({'lr': 1e-3, 'weight_decay': 1e-4, 'dropout': 0.1 if HAS_DROPOUT[a] else None,
                            'source_file': u['metrics_file']})
                recs.append(rec)
            x = tr[(tr.arch == a) & (tr.target == t)]
            if len(x):
                x = x.iloc[0]
                rec = {'arch': a, 'model_class': CLASS_NAME[a], 'target': t, 'level': 'transfer_p30'}
                rec.update({c: x[c] for c in CMP_COLS + HP_COLS})
                rec['source_file'] = x['source_file']
                recs.append(rec)
            d = rows.get((a, t))
            if d:
                tm, usd, hp = d['test_metrics'], d['test_metrics_usd'], d['hyperparameters']
                recs.append({
                    'arch': a, 'model_class': CLASS_NAME[a], 'target': t, 'level': 'retune_p30',
                    'best_val_mse': d['best_val_loss'], 'test_mse': tm['MSE'], 'test_rmse': tm['RMSE'],
                    'test_mae': tm['MAE'], 'test_r2': tm['R2'], 'test_max_error': tm['max_error'],
                    'test_mape_pct_nem_fo_metrika': tm['MAPE (%)'], 'test_usd_rmse': usd['RMSE'],
                    'test_usd_mae': usd['MAE'], 'test_usd_max_error': usd['max_error'],
                    'best_epoch': d['best_epoch'], 'n_epochs_run': d['n_epochs_run'],
                    'hit_max_epochs': d['hit_max_epochs'], 'train_seconds': d['train_seconds'],
                    'mean_epoch_seconds': d['mean_epoch_seconds'],
                    'val_recheck_rel_diff': d['val_recheck']['rel_diff'], 'param_count': d['param_count'],
                    'checkpoint_target_col': d['checkpoint_target_col'],
                    'lr': hp['lr'], 'weight_decay': hp['weight_decay'], 'dropout': hp['dropout'],
                    'source_file': rel(P.results_dir / f"metrics_{d['run']}.json")})
    df = pd.DataFrame(recs)
    df.to_csv(guard_path(P.results_dir / "levels_table.csv"), index=False)
    return df


def improvement_table(P, df):
    recs = []
    for a in ARCH_ORDER:
        for t in TARGET_ORDER:
            s = df[(df.arch == a) & (df.target == t)].set_index('level')
            if not set(LEVELS) <= set(s.index):
                continue
            u, x, r = s.loc['uniform_p30'], s.loc['transfer_p30'], s.loc['retune_p30']
            recs.append({
                'arch': a, 'model_class': CLASS_NAME[a], 'target': t,
                'test_rmse_uniform_p30': u['test_rmse'], 'test_rmse_transfer_p30': x['test_rmse'],
                'test_rmse_retune_p30': r['test_rmse'],
                'test_rmse_impr_pct_vs_transfer': 100 * (1 - r['test_rmse'] / x['test_rmse']),
                'test_rmse_ratio_transfer_over_retune': x['test_rmse'] / r['test_rmse'],
                'test_rmse_impr_pct_vs_uniform': 100 * (1 - r['test_rmse'] / u['test_rmse']),
                'test_rmse_ratio_uniform_over_retune': u['test_rmse'] / r['test_rmse'],
                'val_mse_uniform_p30': u['best_val_mse'], 'val_mse_transfer_p30': x['best_val_mse'],
                'val_mse_retune_p30': r['best_val_mse'],
                'val_mse_impr_pct_vs_transfer': 100 * (1 - r['best_val_mse'] / x['best_val_mse']),
                'val_mse_impr_pct_vs_uniform': 100 * (1 - r['best_val_mse'] / u['best_val_mse']),
                'retune_worse_than_transfer_test_rmse': bool(r['test_rmse'] > x['test_rmse']),
                'retune_worse_than_transfer_val_mse': bool(r['best_val_mse'] > x['best_val_mse']),
                'retune_worse_than_uniform_test_rmse': bool(r['test_rmse'] > u['test_rmse']),
                'best_epoch_retune': int(r['best_epoch']), 'n_epochs_run_retune': int(r['n_epochs_run']),
                'hit_max_epochs_retune': bool(r['hit_max_epochs']),
                'hit_max_epochs_transfer': bool(x['hit_max_epochs']),
            })
    out = pd.DataFrame(recs)
    out.to_csv(guard_path(P.results_dir / "improvement_retune.csv"), index=False)
    return out


def _valid(df):
    return df[df['state'].isin(['COMPLETE', 'PRUNED'])]


def _top5_wd_logmedian(trials):
    v = _valid(trials).dropna(subset=['value']).sort_values('value').head(5)
    return float(10 ** np.median(np.log10(v['params_weight_decay']))) if len(v) else np.nan


def hparam_table(P, bps, call):
    recs = []
    for a in ARCH_ORDER:
        if a not in bps:
            continue
        am, eu = bps[a], call[a]
        t_am = pd.read_csv(P.results_dir / f"trials_{a}.csv")
        t_eu = pd.read_csv(HPO_RESULTS / f"trials_{a}.csv")
        rec = {'arch': a, 'model_class': CLASS_NAME[a],
               'eu_lr': eu['lr'], 'am_lr': am['lr'], 'lr_ratio_am_over_eu': am['lr'] / eu['lr'],
               'lr_log10_diff': float(np.log10(am['lr'] / eu['lr'])),
               'eu_weight_decay': eu['weight_decay'], 'am_weight_decay': am['weight_decay'],
               'wd_ratio_am_over_eu': am['weight_decay'] / eu['weight_decay'],
               'wd_log10_diff': float(np.log10(am['weight_decay'] / eu['weight_decay'])),
               'eu_wd_decades_above_min': float(np.log10(eu['weight_decay'] / WD_MIN)),
               'am_wd_decades_above_min': float(np.log10(am['weight_decay'] / WD_MIN)),
               'eu_wd_in_bottom_tenth': bool(eu['weight_decay'] < WD_BOTTOM_TENTH),
               'am_wd_in_bottom_tenth': bool(am['weight_decay'] < WD_BOTTOM_TENTH),
               'eu_dropout': eu['dropout'], 'am_dropout': am['dropout'],
               'dropout_diff_am_minus_eu': (am['dropout'] - eu['dropout']) if am['dropout'] is not None else None,
               'eu_top5_wd_logmedian': _top5_wd_logmedian(t_eu),
               'am_top5_wd_logmedian': _top5_wd_logmedian(t_am),
               'eu_frac_valid_trials_wd_lt_1e-5': float((_valid(t_eu)['params_weight_decay'] < 1e-5).mean()),
               'am_frac_valid_trials_wd_lt_1e-5': float((_valid(t_am)['params_weight_decay'] < 1e-5).mean()),
               'eu_best_trial': eu['search_best_trial_number'], 'am_best_trial': am['search_best_trial_number'],
               'eu_source': eu['file'], 'am_source': rel(P.results_dir / f"best_params_{a}.json")}
        recs.append(rec)
    out = pd.DataFrame(recs)
    out.to_csv(guard_path(P.results_dir / "hparams_eu_vs_am.csv"), index=False)
    return out


def _anytime(trials):
    t = trials.sort_values('number')
    best, out = np.inf, []
    for _, r in t.iterrows():
        if r['state'] == 'COMPLETE' and r['value'] == r['value']:
            best = min(best, r['value'])
        out.append(best if np.isfinite(best) else np.nan)
    return t['number'].to_numpy(), np.array(out)


def search_tables(P, bps):
    stat_recs, any_recs = [], []
    for a in ARCH_ORDER:
        if a not in bps:
            continue
        for camp, tdir, cdir in (('eu_call_tpe', HPO_RESULTS, HPO_RESULTS),
                                 ('am_put_retune', P.results_dir, P.results_dir)):
            t = pd.read_csv(tdir / f"trials_{a}.csv")
            c = pd.read_csv(cdir / f"search_curves_{a}.csv")
            v = _valid(t)
            ep = c.groupby('trial_number').size()
            dur = pd.to_timedelta(v['duration']).dt.total_seconds().sum()
            n_ep_valid = int(ep[ep.index.isin(v['number'])].sum())
            stat_recs.append({'arch': a, 'campaign': camp,
                              'n_trials': len(t), 'n_valid': len(v),
                              'n_complete': int((t.state == 'COMPLETE').sum()),
                              'n_pruned': int((t.state == 'PRUNED').sum()),
                              'n_fail': int((t.state == 'FAIL').sum()),
                              'fail_numbers': ','.join(str(int(n)) for n in t.loc[t.state == 'FAIL', 'number']),
                              'n_search_epochs': n_ep_valid,
                              'mean_epochs_per_valid_trial': n_ep_valid / max(len(v), 1),
                              'max_epoch_in_curves': int(c['epoch'].max()),
                              'trial_duration_sum_s_valid': float(dur),
                              's_per_epoch': float(dur) / max(n_ep_valid, 1),
                              'search_time_s': bps[a]['search_time_s'] if camp == 'am_put_retune' else None,
                              'best_value': float(t.loc[t.state == 'COMPLETE', 'value'].min()),
                              'best_trial': int(t.loc[t.loc[t.state == 'COMPLETE', 'value'].idxmin(), 'number'])})
            num, best = _anytime(t)
            for n, b in zip(num, best):
                any_recs.append({'arch': a, 'campaign': camp, 'trial_number': int(n), 'anytime_best': b})
    st, an = pd.DataFrame(stat_recs), pd.DataFrame(any_recs)
    st.to_csv(guard_path(P.results_dir / "search_stats.csv"), index=False)
    an.to_csv(guard_path(P.results_dir / "anytime_best.csv"), index=False)
    return st, an


def importance_table(P, bps):
    recs = []
    for a in ARCH_ORDER:
        if a not in bps:
            continue
        eu = read_json(HPO_RESULTS / f"param_importance_{a}.json")
        am = read_json(P.results_dir / f"param_importance_{a}.json")
        for p in ('lr', 'weight_decay', 'dropout'):
            if p in eu or p in am:
                recs.append({'arch': a, 'param': p, 'eu_call_tpe': eu.get(p), 'am_put_retune': am.get(p)})
        recs_a = [r for r in recs if r['arch'] == a]
        top_eu = max(recs_a, key=lambda r: r['eu_call_tpe'] or -1)['param'] if recs_a else None
        top_am = max(recs_a, key=lambda r: r['am_put_retune'] or -1)['param'] if recs_a else None
        for r in recs_a:
            r['top_eu'], r['top_am'] = top_eu, top_am
    out = pd.DataFrame(recs)
    out.to_csv(guard_path(P.results_dir / "param_importance_eu_vs_am.csv"), index=False)
    return out


def ratio_table(P, df):
    recs = []
    for lv in LEVELS:
        sub = []
        for a in ARCH_ORDER:
            s = df[(df.arch == a) & (df.level == lv)].set_index('target')
            if not set(TARGET_ORDER) <= set(s.index):
                continue
            r = {'level': lv, 'arch': a, 'model_class': CLASS_NAME[a],
                 'test_rmse_american': s.loc['american', 'test_rmse'],
                 'test_rmse_european': s.loc['european', 'test_rmse'],
                 'ratio_rmse_am_over_eu': s.loc['american', 'test_rmse'] / s.loc['european', 'test_rmse'],
                 'val_mse_american': s.loc['american', 'best_val_mse'],
                 'val_mse_european': s.loc['european', 'best_val_mse'],
                 'ratio_val_mse_am_over_eu': s.loc['american', 'best_val_mse'] / s.loc['european', 'best_val_mse']}
            recs.append(r)
            sub.append(r)
        if sub:
            recs.append({'level': lv, 'arch': 'MEDIAN', 'model_class': 'medián',
                         'ratio_rmse_am_over_eu': float(np.median([r['ratio_rmse_am_over_eu'] for r in sub])),
                         'ratio_val_mse_am_over_eu': float(np.median([r['ratio_val_mse_am_over_eu'] for r in sub]))})
    out = pd.DataFrame(recs)
    out.to_csv(guard_path(P.results_dir / "ratio_american_vs_european.csv"), index=False)
    return out


def ranking_tables(P, df, call):
    from scipy.stats import kendalltau
    recs, order = [], {}
    for t in TARGET_ORDER:
        for lv in LEVELS:
            s = df[(df.target == t) & (df.level == lv)]
            for metric in ('test_rmse', 'best_val_mse'):
                srt = s.sort_values(metric)
                order[(t, lv, metric)] = list(srt['arch'])
                for k, (_, row) in enumerate(srt.iterrows(), 1):
                    recs.append({'target': t, 'level': lv, 'metric': metric, 'rank': k,
                                 'arch': row['arch'], 'value': row[metric]})
    for metric, key in (('test_rmse', 'test_rmse'), ('best_val_mse', 'val_mse')):
        srt = sorted(call, key=lambda a: call[a][key])
        order[('european_call', 'hpo_p30', metric)] = srt
        for k, a in enumerate(srt, 1):
            recs.append({'target': 'european_call', 'level': 'hpo_p30', 'metric': metric, 'rank': k,
                         'arch': a, 'value': call[a][key]})
    rank = pd.DataFrame(recs)
    rank.to_csv(guard_path(P.results_dir / "ranking.csv"), index=False)

    def tau(o1, o2):
        common = [a for a in o1 if a in o2]
        if len(common) < 3:
            return {}
        res = kendalltau([o1.index(a) for a in common], [o2.index(a) for a in common])
        return {'tau': float(res.statistic), 'p_value': float(res.pvalue), 'n': len(common)}

    kt = {'note': 'Kendall tau a rangsorok kozott (7 architektura, egy seed: tajekoztato).',
          'call_source': 'hpo/results/p30/best_params_*.json (hpo_test_metrics.RMSE, final_retrain_best_val_loss)',
          'pairs': []}
    pairs = [('american', 'uniform_p30', 'european', 'uniform_p30'),
             ('american', 'transfer_p30', 'european', 'transfer_p30'),
             ('american', 'retune_p30', 'european', 'retune_p30'),
             ('american', 'retune_p30', 'european_call', 'hpo_p30'),
             ('european', 'retune_p30', 'european_call', 'hpo_p30'),
             ('american', 'retune_p30', 'american', 'transfer_p30'),
             ('american', 'retune_p30', 'american', 'uniform_p30')]
    for metric in ('test_rmse', 'best_val_mse'):
        for t1, l1, t2, l2 in pairs:
            o1, o2 = order.get((t1, l1, metric)), order.get((t2, l2, metric))
            if o1 and o2:
                kt['pairs'].append({'metric': metric, 'a': f"{t1} {l1}", 'b': f"{t2} {l2}",
                                    'order_a': o1, 'order_b': o2, **tau(o1, o2)})
    write_json(P.results_dir / "kendall_tau.json", kt)
    return rank, kt


def concat_runs(P, rows, prefix, out_name):
    parts = []
    for (a, t), d in rows.items():
        f = P.results_dir / f"{prefix}_{d['run']}.csv"
        if f.exists():
            x = pd.read_csv(f)
            x.insert(1, 'arch', a)
            x.insert(2, 'target', t)
            x.insert(3, 'level', 'retune_p30')
            parts.append(x)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_csv(guard_path(P.results_dir / out_name), index=False)
    return out


def segment_ratios(P, seg, prem):
    out = {}
    for tab, prev_name in ((seg, 'segment_ratio_am_eu.csv'), (prem, 'premium_ratio_am_eu.csv')):
        recs = []
        for a in ARCH_ORDER:
            if not len(tab):
                break
            for (g, sgm), grp in tab[tab.arch == a].groupby(['group', 'segment'], sort=False):
                s = grp.set_index('target')
                if not set(TARGET_ORDER) <= set(s.index):
                    continue
                recs.append({'level': 'retune_p30', 'arch': a, 'group': g, 'segment': sgm,
                             'n': int(s.loc['american', 'n']), 'rmse_american': s.loc['american', 'RMSE'],
                             'rmse_european': s.loc['european', 'RMSE'],
                             'ratio_am_over_eu': s.loc['american', 'RMSE'] / s.loc['european', 'RMSE']})
        prev = pd.read_csv(TRANSFER_DIR / prev_name)
        both = pd.concat([prev, pd.DataFrame(recs)], ignore_index=True)
        both.to_csv(guard_path(P.results_dir / prev_name), index=False)
        out[prev_name] = both
    return out['segment_ratio_am_eu.csv'], out['premium_ratio_am_eu.csv']


def sanity_am_vs_eu(P, rows):
    df = C.load_am_split('test')
    K = df['K'].to_numpy()
    d_true = ((df['american_put_price'] - df['european_put_price']) / df['K']).to_numpy()
    out = {'test_file': rel(C.am_split_path('test')), 'true_mean_am_minus_eu_norm': float(d_true.mean()),
           'models': {}}
    ds = C.PricingDataset.from_split('test', 'american', device=DEVICE)
    for a in ARCH_ORDER:
        if (a, 'american') not in rows or (a, 'european') not in rows:
            continue
        m_am, _ = load_model(str(P.ckpt(run_name(a, 'american'))), device=DEVICE)
        m_eu, _ = load_model(str(P.ckpt(run_name(a, 'european'))), device=DEVICE)
        pa, pe = C.predict(m_am, ds), C.predict(m_eu, ds)
        d = pa.astype(np.float64) - pe.astype(np.float64)
        out['models'][a] = {'mean_pred_am_minus_eu_norm': float(d.mean()),
                            'mean_am_ge_eu': bool(pa.mean() >= pe.mean()),
                            'frac_pred_am_lt_eu': float((d < 0).mean()),
                            'premium_pred_rmse_vs_true_am_minus_bs_norm': float(np.sqrt(np.mean((d - d_true) ** 2))),
                            'usd_mean_pred_am_minus_eu': float(np.mean(d * K))}
    del ds
    write_json(P.results_dir / "sanity_american_vs_european.json", out)
    return out


def label_uncertainty(P, df_levels):
    am = df_levels[df_levels.target == 'american'].sort_values('test_rmse')
    cands = []
    best = am.iloc[0]
    cands.append((best['arch'], best['level']))
    ret = am[am.level == 'retune_p30']
    if len(ret) and (ret.iloc[0]['arch'], 'retune_p30') not in cands:
        cands.append((ret.iloc[0]['arch'], 'retune_p30'))
    df = C.load_am_split('test')
    price = df['american_put_price'].to_numpy()
    K = df['K'].to_numpy()
    ds = C.PricingDataset.from_split('test', 'american', device=DEVICE)
    out = {'reference': {'crr_500_vs_1000_rel_max': CRR_REL_UNCERTAINTY,
                         'crr_500_vs_1000_source': 'CRR 500 vs. 1000 lepes (legfeljebb ~0,04%; ATM 2,3190 -> 2,3193)',
                         'crr_tree_european_error_median_usd': CRR_TREE_EU_ERR_MEDIAN_USD,
                         'crr_tree_european_error_source': 'CRR-europai vs. BS (medián 0,0016 $, max. 0,043 $)'},
           'test_file': rel(C.am_split_path('test')), 'models': []}
    for a, lv in cands:
        ck = (P.ckpt(run_name(a, 'american')) if lv == 'retune_p30'
              else TRANSFER_MODELS_DIR / f"{a}_american_transfer_p30_best.pt" if lv == 'transfer_p30'
              else C.AMNN_DIR / "models" / "uniform" / f"{a}_american_p30_best.pt")
        m, _ = load_model(str(ck), device=DEVICE)
        pred = C.predict(m, ds).astype(np.float64) * K
        err = np.abs(pred - price)
        mask = price >= 0.01 * K
        rel_err = err[mask] / price[mask]
        out['models'].append({
            'arch': a, 'level': lv, 'checkpoint': rel(ck),
            'usd_rmse': float(np.sqrt(np.mean(err ** 2))), 'usd_mae': float(err.mean()),
            'usd_median_abs_err': float(np.median(err)), 'usd_p95_abs_err': float(np.quantile(err, 0.95)),
            'n_rows_price_ge_1pct_K': int(mask.sum()),
            'rel_err_median_price_ge_1pct_K': float(np.median(rel_err)),
            'rel_err_p95_price_ge_1pct_K': float(np.quantile(rel_err, 0.95)),
            'median_rel_err_over_crr_uncertainty': float(np.median(rel_err) / CRR_REL_UNCERTAINTY),
            'median_abs_err_over_tree_eu_err_median': float(np.median(err) / CRR_TREE_EU_ERR_MEDIAN_USD),
            'frac_rows_rel_err_below_crr_uncertainty': float((rel_err < CRR_REL_UNCERTAINTY).mean()),
        })
    del ds
    write_json(P.results_dir / "label_uncertainty.json", out)
    return out


def _startup_equal(eu, am, n=10):
    te = sorted(eu.trials, key=lambda t: t.number)[:n]
    ta = sorted(am.trials, key=lambda t: t.number)[:n]
    k = min(len(te), len(ta))
    eq = [te[i].params == ta[i].params for i in range(k)]
    ve = [t for t in sorted(eu.trials, key=lambda t: t.number) if t.state.name != 'FAIL'][:n]
    va = [t for t in sorted(am.trials, key=lambda t: t.number) if t.state.name != 'FAIL'][:n]
    kv = min(len(ve), len(va))
    eqv = [ve[i].params == va[i].params for i in range(kv)]
    return {'n_checked': k, 'n_equal': int(sum(eq)), 'all_equal': bool(k and all(eq)),
            'by_number_all_equal': bool(k and all(eq)),
            'n_checked_excl_fail': kv, 'n_equal_excl_fail': int(sum(eqv)),
            'all_equal_excl_fail': bool(kv and all(eqv)),
            'american_trial_numbers_excl_fail': [t.number for t in va]}


def protocol_check(P, bps):
    import optuna
    import run_hpo_american as RA
    R = RA.R
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    res = {'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'run_hpo_constants': {'SEARCH_MAX_EPOCHS': R.SEARCH_MAX_EPOCHS, 'SEARCH_PATIENCE': R.SEARCH_PATIENCE,
                                 'SEARCH_BATCH_SIZE': R.SEARCH_BATCH_SIZE, 'BASE_SEED': R.BASE_SEED,
                                 'TRIAL_BUDGET': R.TRIAL_BUDGET},
           'script_constants': {'PRUNER_KW': RA.PRUNER_KW},
           'archs': {}}
    res['pruner_kw_equal_run_hpo'] = RA.PRUNER_KW == dict(min_resource=3, max_resource=R.SEARCH_MAX_EPOCHS,
                                                          reduction_factor=3)
    ok_all = res['pruner_kw_equal_run_hpo']
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        for a in ARCH_ORDER:
            if a not in bps:
                continue
            tmp = optuna.create_study()
            tr = tmp.ask()
            R.build_trial_hparams(tr, a)
            expected = dict(tr.distributions)
            src = HPO_DIR / "optuna_studies" / f"{a}.db"
            cp = Path(td) / f"eu_{a}.db"
            shutil.copy2(src, cp)
            eu_st = optuna.storages.RDBStorage(f"sqlite:///{cp.as_posix()}")
            am_st = optuna.storages.RDBStorage(f"sqlite:///{P.db(a).as_posix()}")
            eu = optuna.load_study(study_name=a, storage=eu_st)
            am = optuna.load_study(study_name=a, storage=am_st)
            eu_d = [dict(t.distributions) for t in eu.trials if t.state.name in ('COMPLETE', 'PRUNED')]
            am_d = [dict(t.distributions) for t in am.trials if t.state.name in ('COMPLETE', 'PRUNED')]
            ua = am.user_attrs
            n_valid = sum(t.state.name in ('COMPLETE', 'PRUNED') for t in am.trials)
            r = {
                'expected_distributions': {k: repr(v) for k, v in expected.items()},
                'american_distributions_all_equal_expected': all(d == expected for d in am_d),
                'european_distributions_all_equal_expected': all(d == expected for d in eu_d),
                'n_valid_american': n_valid, 'n_valid_european': len(eu_d),
                'budget': R.TRIAL_BUDGET[a],
                'budget_met_american': n_valid == R.TRIAL_BUDGET[a] or (bps[a]['n_trials_target'] != R.TRIAL_BUDGET[a]
                                                                         and n_valid == bps[a]['n_trials_target']),
                'fail_trials_american': [t.number for t in am.trials if t.state.name == 'FAIL'],
                'study_name_american': am.study_name, 'study_name_european': eu.study_name,
                'study_name_equal': am.study_name == eu.study_name == a,
                'user_attrs_sampler': ua.get('sampler'), 'user_attrs_sampler_seed': ua.get('sampler_seed'),
                'user_attrs_pruner': ua.get('pruner'),
                'sampler_ok': ua.get('sampler') == f"TPESampler(seed={R.BASE_SEED})"
                              and ua.get('sampler_seed') == R.BASE_SEED,
                'pruner_ok': ua.get('pruner') == {'class': 'HyperbandPruner', **dict(
                    min_resource=3, max_resource=R.SEARCH_MAX_EPOCHS, reduction_factor=3)},
                'search_patience_ok': ua.get('search_patience') == R.SEARCH_PATIENCE,
                'search_batch_ok': ua.get('search_batch_size') == R.SEARCH_BATCH_SIZE,
                'search_max_epochs_attr': ua.get('search_max_epochs'),
                'search_max_epochs_ok': ua.get('search_max_epochs') == R.SEARCH_MAX_EPOCHS,
                'startup_params_equal_european': _startup_equal(eu, am),
                'european_trial_numbers_valid':[t.number for t in eu.trials if t.state.name in ('COMPLETE', 'PRUNED')] == list(range(len(eu_d))),
            }
            r['all_ok'] = bool(r['american_distributions_all_equal_expected'] and r['european_distributions_all_equal_expected']
                               and r['budget_met_american'] and r['study_name_equal'] and r['sampler_ok']
                               and r['pruner_ok'] and r['search_patience_ok'] and r['search_batch_ok']
                               and r['search_max_epochs_ok'])
            ok_all &= r['all_ok']
            res['archs'][a] = r
            del eu, am
            eu_st.engine.dispose()
            am_st.engine.dispose()
    res['all_ok'] = bool(ok_all)
    write_json(P.results_dir / "protocol_check.json", res)
    return res


def verification(P, rows, bps, df, proto):
    import torch
    import optuna
    v = {'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'n_runs': len(rows)}
    v['param_count_match_all'] = bool(len(rows) and all(d['param_count_match'] for d in rows.values()))
    hp_rows = []
    for a in ARCH_ORDER:
        if (a, 'american') not in rows or (a, 'european') not in rows:
            continue
        am, eu = rows[(a, 'american')], rows[(a, 'european')]
        ck_am = torch.load(P.ckpt(am['run']), map_location='cpu', weights_only=False)
        ck_eu = torch.load(P.ckpt(eu['run']), map_location='cpu', weights_only=False)
        st_storage = optuna.storages.RDBStorage(f"sqlite:///{P.db(a).as_posix()}")
        st = optuna.load_study(study_name=a, storage=st_storage)
        bt = st.best_trial.params
        r = {'arch': a}
        for k in HP_COLS:
            exp = bt.get(k) if (k != 'dropout' or HAS_DROPOUT[a]) else None
            r[f'{k}_equal'] = bool(_eq(am['hyperparameters'][k], eu['hyperparameters'][k])
                                   and _eq(am['hyperparameters'][k], exp)
                                   and _eq(ck_am['extra_meta']['hyperparameters'][k], exp)
                                   and _eq(ck_eu['extra_meta']['hyperparameters'][k], exp)
                                   and _eq(bps[a][k], exp))
        r['model_kwargs_equal'] = ck_am['model_kwargs'] == ck_eu['model_kwargs']
        r['best_trial_equal'] = (am['hyperparameter_source_trial'] == eu['hyperparameter_source_trial']
                                 == st.best_trial.number == bps[a]['search_best_trial_number'])
        r['target_cols_ok'] = (ck_am['target_col'] == C.TARGETS['american']['target_col']
                               and ck_eu['target_col'] == C.TARGETS['european']['target_col'])
        r['final_settings_ok'] = all(d['patience'] == 30 and d['seed'] == 42
                                     and d['hyperparameters']['batch_size'] == 4096 for d in (am, eu))
        r['all_equal'] = bool(all(r[f'{k}_equal'] for k in HP_COLS) and r['model_kwargs_equal']
                              and r['best_trial_equal'] and r['target_cols_ok'] and r['final_settings_ok'])
        hp_rows.append(r)
        del st, ck_am, ck_eu
        st_storage.engine.dispose()
    v['hyperparameters'] = hp_rows
    v['european_hp_bitwise_equal_american_all'] = bool(hp_rows and all(r['all_equal'] for r in hp_rows))
    rd = [d['val_recheck']['rel_diff'] for d in rows.values()]
    v['val_recheck_max_rel_diff'] = float(max(rd)) if rd else None
    v['val_recheck_ok'] = bool(rd and max(rd) < 1e-6)
    v['max_epochs_all_200'] = bool(rows and all(d['max_epochs'] == 200 for d in rows.values()))
    ours = pd.read_csv(P.results_dir / "levels_table.csv", float_precision='round_trip')
    mism = []
    for lv, src, cols in (('uniform_p30', uniform_src(), CMP_COLS),
                          ('transfer_p30', transfer_src(), CMP_COLS + HP_COLS)):
        o = ours[ours.level == lv]
        for _, r in o.iterrows():
            s = src[(src.arch == r['arch']) & (src.target == r['target'])].iloc[0]
            for c in cols:
                a_, b_ = r[c], s[c]
                if not _eq(a_.item() if hasattr(a_, 'item') else a_, b_.item() if hasattr(b_, 'item') else b_):
                    mism.append({'level': lv, 'arch': r['arch'], 'target': r['target'], 'col': c,
                                 'ours': str(a_), 'src': str(b_)})
        v[f'{lv}_rows_n'] = int(len(o))
    v['level_rows_sources'] = {'uniform_p30': rel(UNIFORM_DIR / "comparison_table.csv"),
                               'transfer_p30': rel(TRANSFER_DIR / "comparison_table.csv")}
    v['level_rows_bitwise_equal'] = bool(v['uniform_p30_rows_n'] == 14 and v['transfer_p30_rows_n'] == 14
                                         and not mism)
    v['level_rows_mismatches'] = mism
    v['n_hit_max_epochs'] = int(sum(d['hit_max_epochs'] for d in rows.values()))
    v['hit_max_epochs_runs'] = [d['run'] for d in rows.values() if d['hit_max_epochs']]
    v['protocol_check_all_ok'] = proto['all_ok']
    meta_p = P.results_dir / "campaign_meta.json"
    v['hpo_import_check'] = read_json(meta_p).get('hpo_import_check') if meta_p.exists() else None
    v['n_checkpoints'] = int(sum(P.ckpt(d['run']).exists() for d in rows.values()))
    write_json(P.results_dir / "verification.json", v)
    return v


RANK_LABEL = {('american', 'uniform_p30'): 'amerikai put, uniform p30',
              ('american', 'transfer_p30'): 'amerikai put, transfer p30',
              ('american', 'retune_p30'): 'amerikai put, retune p30',
              ('european', 'uniform_p30'): 'európai put, uniform p30',
              ('european', 'transfer_p30'): 'európai put, transfer p30',
              ('european', 'retune_p30'): 'európai put, retune p30',
              ('european_call', 'hpo_p30'): 'európai call, TPE p30'}
METRIC_LABEL = {'test_rmse': 'teszt-RMSE', 'best_val_mse': 'val MSE'}


def _rank_label(s):
    t, lv = s.split(' ')
    return RANK_LABEL.get((t, lv), s)


def write_md(P, rows, bps, df, imp, hp, st, imps, ratio, rank, kt, prem_ratio, sanity, lab, proto, ver, call):
    import md_tables as M
    R_ = P.results_dir
    L = []

    L += ["## Teszt-RMSE három szinten", ""]
    body = []
    for a in ARCH_ORDER:
        cells = []
        for t in TARGET_ORDER:
            for lv in LEVELS:
                s = df[(df.arch == a) & (df.target == t) & (df.level == lv)]
                cells.append(M.sci(s['test_rmse'].iloc[0]) if len(s) else '–')
        body.append([CLASS_NAME[a]] + cells)
    L += M.table(["Architektúra"] + [f"{TLAB[t].capitalize()}, {LLAB[lv]}" for t in TARGET_ORDER for lv in LEVELS], body)

    L += ["", "## A retune p30 tanítása", ""]
    body = []
    for a in ARCH_ORDER:
        for t in TARGET_ORDER:
            d = rows.get((a, t))
            if d:
                body.append([CLASS_NAME[a], TLAB[t], d['best_epoch'], d['n_epochs_run'], M.yes(bool(d['hit_max_epochs']))])
    L += M.table(["Architektúra", "Cél", "Legjobb epoch", "Futott epoch", "200 epochos korlát"], body, "llrrr")

    L += ["", "## A legjobb amerikai modellek", ""]
    body = []
    am_all = df[df.target == 'american'].sort_values('test_rmse')
    ret = df[(df.target == 'american') & (df.level == 'retune_p30')].sort_values('test_rmse')
    for title, tab in (('A három szint közül', am_all), ('A retune p30 szinten', ret)):
        if len(tab):
            b = tab.iloc[0]
            body.append([title, f"{CLASS_NAME[b['arch']]}, {LLAB[b['level']]}", M.sci(b['test_rmse']),
                         M.sci(b['best_val_mse']), M.f(b['test_r2'], 6), M.f(b['test_usd_rmse'], 4)])
    L += M.table(["Kiválasztás", "Modell", "Teszt-RMSE", "Val MSE", "Teszt R²", "Teszt-RMSE ($)"], body, "llrrrr")

    L += ["", "## A retune p30 változása a transfer p30-hoz és a uniform p30-hoz képest", ""]
    body = [[CLASS_NAME[r['arch']], TLAB[r['target']], M.chg(-r['test_rmse_impr_pct_vs_transfer']),
             M.f(r['test_rmse_ratio_transfer_over_retune']), M.chg(-r['test_rmse_impr_pct_vs_uniform']),
             M.f(r['test_rmse_ratio_uniform_over_retune']), M.chg(-r['val_mse_impr_pct_vs_transfer']),
             M.chg(-r['val_mse_impr_pct_vs_uniform'])] for _, r in imp.iterrows()]
    for t in TARGET_ORDER:
        s = imp[imp.target == t]
        if len(s):
            body.append([f"Medián, {TLAB[t]}", '', M.chg(-s['test_rmse_impr_pct_vs_transfer'].median()),
                         M.f(s['test_rmse_ratio_transfer_over_retune'].median()),
                         M.chg(-s['test_rmse_impr_pct_vs_uniform'].median()), '', '', ''])
    L += M.table(["Architektúra", "Cél", "Teszt-RMSE, a transferhez", "Transfer és retune aránya",
                  "Teszt-RMSE, a uniformhoz", "Uniform és retune aránya", "Val MSE, a transferhez",
                  "Val MSE, a uniformhoz"], body, "ll" + "r" * 6)

    L += ["", "## Ahol a retune p30 rosszabb a transfer p30-nál", ""]
    L += M.table(["Cél", "Architektúra"],
                 [[TLAB[t], ', '.join(CLASS_NAME[a] for a in imp[(imp.target == t)
                                                                 & imp.retune_worse_than_transfer_test_rmse].arch)
                   or 'nincs'] for t in TARGET_ORDER if len(imp[imp.target == t])], "ll")

    worse = imp[imp.retune_worse_than_transfer_test_rmse]
    if len(worse):
        L += ["", "## A transfer és a retune konfiguráció a romlott futásoknál", ""]

        def _hp(s):
            d = s['dropout']
            return f"{M.sci(s['lr'])} / {M.sci(s['weight_decay'])} / {'–' if pd.isna(d) else M.f(d, 3)}"

        body = []
        for _, r in worse.iterrows():
            a, t = r['arch'], r['target']
            tr = df[(df.arch == a) & (df.target == t) & (df.level == 'transfer_p30')].iloc[0]
            rt = df[(df.arch == a) & (df.target == t) & (df.level == 'retune_p30')].iloc[0]
            am_tr = _valid(pd.read_csv(R_ / f"trials_{a}.csv")).sort_values('value')
            eu_tr = _valid(pd.read_csv(HPO_DIR / "results" / f"trials_{a}.csv")).sort_values('value')
            body.append([CLASS_NAME[a], TLAB[t], _hp(tr), _hp(rt),
                         f"{M.sci(am_tr['value'].iloc[0])} (#{int(am_tr['number'].iloc[0])})",
                         f"{M.sci(am_tr['value'].iloc[1])} (#{int(am_tr['number'].iloc[1])})",
                         f"{M.sci(eu_tr['value'].iloc[0])} (#{int(eu_tr['number'].iloc[0])})",
                         M.sci(tr['best_val_mse']), M.sci(rt['best_val_mse'])])
        L += M.table(["Architektúra", "Cél", "Transfer: lr / wd / dropout", "Retune: lr / wd / dropout",
                      "Amerikai keresés legjobbja (trial)", "Amerikai keresés 2. (trial)",
                      "Európai call keresés legjobbja (trial)", "p30 val MSE, transfer", "p30 val MSE, retune"],
                     body, "llll" + "r" * 5)

    L += ["", "## Hiperparaméterek: európai call TPE p30 és amerikai put retune", ""]
    dr = lambda v: M.f(v, 3) if v is not None and v == v else '–'
    body = [[CLASS_NAME[r['arch']], M.sci(r['eu_lr']), M.sci(r['am_lr']), M.f(r['lr_ratio_am_over_eu']),
             M.sci(r['eu_weight_decay']), M.f(r['eu_wd_decades_above_min']), M.sci(r['am_weight_decay']),
             M.f(r['am_wd_decades_above_min']), M.yes(bool(r['am_wd_in_bottom_tenth'])), dr(r['eu_dropout']),
             dr(r['am_dropout']), M.sci(r['eu_top5_wd_logmedian']), M.sci(r['am_top5_wd_logmedian'])]
            for _, r in hp.iterrows()]
    L += M.table(["Architektúra", "Európai lr", "Amerikai lr", "Amerikai és európai lr aránya", "Európai wd",
                  "Európai wd, dekád 1e-6 felett", "Amerikai wd", "Amerikai wd, dekád 1e-6 felett",
                  f"Amerikai wd < {M.sci(WD_BOTTOM_TENTH)}", "Európai dropout", "Amerikai dropout",
                  "Top-5 wd log-mediánja, európai", "Top-5 wd log-mediánja, amerikai"], body)
    if len(hp):
        L += ["", "## A hiperparaméterek összesítése", ""]
        L += M.table(["Mutató", "Érték"], [
            [f"Amerikai retune: wd < {M.sci(WD_BOTTOM_TENTH)}", f"{int(hp['am_wd_in_bottom_tenth'].sum())}/{len(hp)}"],
            [f"Európai call TPE: wd < {M.sci(WD_BOTTOM_TENTH)}", f"{int(hp['eu_wd_in_bottom_tenth'].sum())}/{len(hp)}"],
            ["Az amerikai és az európai érték log10-különbségének abszolút mediánja, lr", M.f(hp['lr_log10_diff'].abs().median())],
            ["Az amerikai és az európai érték log10-különbségének abszolút mediánja, wd", M.f(hp['wd_log10_diff'].abs().median())]], "lr")

    L += ["", "## A keresés lefolyása", ""]
    body = []
    for _, r in st.iterrows():
        camp = 'európai call TPE' if r['campaign'] == 'eu_call_tpe' else 'amerikai put retune'
        budget = (bps[r['arch']]['n_trials_target'] if r['campaign'] == 'am_put_retune'
                  else read_json(HPO_RESULTS / f"best_params_{r['arch']}.json")['n_trials_target'])
        body.append([CLASS_NAME[r['arch']], camp, f"{r['n_valid']}/{budget}", r['n_complete'], r['n_pruned'],
                     r['n_fail'], r['n_search_epochs'], M.f(r['mean_epochs_per_valid_trial'], 1),
                     M.f(r['trial_duration_sum_s_valid'], 1), M.f(r['s_per_epoch']),
                     f"{M.sci(r['best_value'])} (#{r['best_trial']})"])
    for camp, lab_ in (('eu_call_tpe', 'európai call TPE'), ('am_put_retune', 'amerikai put retune')):
        s = st[st.campaign == camp]
        if len(s):
            body.append(["Összesen", lab_, '', int(s.n_complete.sum()), int(s.n_pruned.sum()), int(s.n_fail.sum()),
                         int(s.n_search_epochs.sum()), '', M.f(s.trial_duration_sum_s_valid.sum(), 1), '', ''])
    L += M.table(["Architektúra", "Kampány", "Érvényes trial / cél", "COMPLETE", "PRUNED", "FAIL", "Keresési epoch",
                  "Epoch trialonként", "Trialidő (s)", "s/epoch", "Legjobb keresési val MSE (trial)"], body, "ll" + "r" * 9)
    fails = {a: d['fail_trial_numbers'] for a, d in bps.items() if d.get('fail_trial_numbers')}
    L += ["", "## FAIL trialok (amerikai put retune)", ""]
    L += M.table(["Architektúra", "Trial"], [[CLASS_NAME[a], ', '.join(f"#{n}" for n in v)] for a, v in fails.items()]
                 or [['nincs', '–']], "ll")

    L += ["", "## Hiperparaméter-fontosság (PED-ANOVA, tájékoztató)", ""]
    L += M.table(["Architektúra", "Paraméter", "Európai call TPE", "Amerikai put retune"],
                 [[CLASS_NAME[r['arch']], r['param'], M.f(r['eu_call_tpe']), M.f(r['am_put_retune'])]
                  for _, r in imps.iterrows()], "llrr")
    if len(imps):
        tops = imps.drop_duplicates('arch')
        L += ["", "## A legfontosabb paraméter a két kampányban", ""]
        L += M.table(["Architektúra", "Európai call TPE", "Amerikai put retune", "Azonos"],
                     [[CLASS_NAME[r['arch']], r['top_eu'], r['top_am'], M.yes(r['top_eu'] == r['top_am'])]
                      for _, r in tops.iterrows()] +
                     [["Összesen", '', '', f"{int((tops.top_eu == tops.top_am).sum())}/{len(tops)}"]], "lllr")

    L += ["", "## Amerikai és európai teszt-RMSE aránya három szinten", ""]
    rr = ratio[ratio.arch != 'MEDIAN']
    body = []
    for a in ARCH_ORDER:
        cells = []
        for lv in LEVELS:
            s = rr[(rr.arch == a) & (rr.level == lv)]
            cells.append(M.f(s['ratio_rmse_am_over_eu'].iloc[0]) if len(s) else '–')
        body.append([CLASS_NAME[a]] + cells)
    cells = []
    for lv in LEVELS:
        m = ratio[(ratio.arch == 'MEDIAN') & (ratio.level == lv)]
        cells.append(M.f(m['ratio_rmse_am_over_eu'].iloc[0]) if len(m) else '–')
    body.append(["Medián"] + cells)
    L += M.table(["Architektúra"] + [LLAB[lv] for lv in LEVELS], body)

    L += ["", "## Rangsor teszt-RMSE szerint", ""]
    cols_ = (('american', 'uniform_p30'), ('american', 'transfer_p30'), ('american', 'retune_p30'),
             ('european', 'uniform_p30'), ('european', 'transfer_p30'), ('european', 'retune_p30'),
             ('european_call', 'hpo_p30'))
    rt = rank[rank.metric == 'test_rmse']
    body = []
    for k in range(1, len(ARCH_ORDER) + 1):
        cells = []
        for t, lv in cols_:
            s = rt[(rt.target == t) & (rt.level == lv) & (rt['rank'] == k)]
            cells.append(f"{CLASS_NAME[s.iloc[0]['arch']]} ({M.sci(s.iloc[0]['value'])})" if len(s) else '–')
        body.append([f"{k}."] + cells)
    L += M.table(["Hely"] + [RANK_LABEL[c] for c in cols_], body, "r" + "l" * 7)

    L += ["", "## Kendall τ a rangsorok között", ""]
    L += M.table(["Metrika", "Egyik rangsor", "Másik rangsor", "Kendall τ", "p"],
                 [[METRIC_LABEL.get(p['metric'], p['metric']), _rank_label(p['a']), _rank_label(p['b']),
                   M.f(p.get('tau')), M.f(p.get('p_value'), 3)] for p in kt['pairs']], "lllrr")

    if len(prem_ratio):
        segs = list(dict.fromkeys(prem_ratio['segment']))
        L += ["", "## Amerikai és európai teszt-RMSE aránya a korai lehívási prémium szerint", ""]
        body = []
        for a in ARCH_ORDER:
            for lv in LEVELS:
                q = prem_ratio[(prem_ratio.arch == a) & (prem_ratio.level == lv)].set_index('segment')
                if len(q):
                    body.append([CLASS_NAME[a], LLAB[lv]] + [M.f(q.loc[s, 'ratio_am_over_eu']) if s in q.index
                                                            else '–' for s in segs])
        L += M.table(["Architektúra", "Szint"] + [s.split(' ')[0] for s in segs], body, "ll" + "r" * len(segs))

    L += ["", "## A legjobb amerikai modell hibája és a CRR-címke bizonytalansága", ""]
    L += M.table(["Modell", "RMSE ($)", "MAE ($)", "Abszolút hiba mediánja ($)",
                  "Relatív hiba mediánja (ár ≥ 1% · K)", "A medián és a 0.04% aránya", "Relatív hiba < 0.04%"],
                 [[f"{CLASS_NAME[m['arch']]}, {LLAB[m['level']]}", M.f(m['usd_rmse'], 4), M.f(m['usd_mae'], 4),
                   M.f(m['usd_median_abs_err'], 4), M.pct(m['rel_err_median_price_ge_1pct_K'], 3),
                   M.f(m['median_rel_err_over_crr_uncertainty'], 1),
                   M.pct(m['frac_rows_rel_err_below_crr_uncertainty'], 1)] for m in lab['models']])
    L += ["", "## A CRR-címke bizonytalansága", ""]
    L += M.table(["Forrás", "Érték"], [["CRR 500 és 1000 lépés relatív eltérése", "legfeljebb ~0.04%"],
                                       ["A CRR-fa európai árának eltérése a BS ártól, medián", "0.0016 $"]], "ll")

    L += ["", "## Az indító fázis (0–9. trial) egyezése az európai call TPE kampánnyal", ""]
    su = {a: r['startup_params_equal_european'] for a, r in proto['archs'].items()}
    body = [[CLASS_NAME[a], f"{v['n_equal']}/{v['n_checked']}", M.yes(v['all_equal']),
             f"{v.get('n_equal_excl_fail')}/{v.get('n_checked_excl_fail')}", M.yes(v.get('all_equal_excl_fail', False))]
            for a, v in su.items()]
    body.append(["Összesen", '', f"{sum(v['all_equal'] for v in su.values())}/{len(su)}", '',
                 f"{sum(v.get('all_equal_excl_fail', False) for v in su.values())}/{len(su)}"])
    L += M.table(["Architektúra", "Azonos trial, sorszám szerint", "Mind azonos",
                  "Azonos trial, FAIL nélkül", "Mind azonos, FAIL nélkül"], body)

    L += ["", "## Ellenőrzések", ""]
    hic = ver.get('hpo_import_check') or {}
    L += M.table(["Ellenőrzés", "Eredmény"], [
        ["Protokoll: keresési tér, sampler, pruner, keresési türelem, batch, epoch és költségvetés az európai "
         "kampánnyal azonos", M.yes(proto['all_ok'])],
        ["Végső checkpointok", ver['n_checkpoints']],
        [f"Paraméterszám = kiindulási v1 = HPO p30 = uniform p30 = transfer p30 ({ver['n_runs']} futás)",
         M.yes(ver['param_count_match_all'])],
        ["Az európai retune hiperparaméterei bitre az amerikai páréval, a study legjobb trialjáéval és a "
         "`best_params` értékével", M.yes(ver['european_hp_bitwise_equal_american_all'])],
        ["Checkpoint-visszatöltés: legnagyobb relatív eltérés a mentett val MSE-től", M.sci(ver['val_recheck_max_rel_diff'])],
        [f"A uniform és a transfer sorok bitre a forrásaikkal ({ver['uniform_p30_rows_n']} + "
         f"{ver['transfer_p30_rows_n']} sor)", M.yes(ver['level_rows_bitwise_equal'])],
        [f"A `run_hpo` importja nem változtatta a `hpo/` mappát ({hic.get('n_entries_before')} és "
         f"{hic.get('n_entries_after')} bejegyzés)", M.yes(hic.get('hpo_tree_unchanged_by_import'))],
        ["A 200 epochos korlátig futott retune tanítások", f"{ver['n_hit_max_epochs']} "
         f"({', '.join(ver['hit_max_epochs_runs']) or '–'})"]])

    if sanity.get('models'):
        L += ["", "## Amerikai és európai ár a teszthalmazon (retune p30)", ""]
        body = [[CLASS_NAME[a], M.sci(m['mean_pred_am_minus_eu_norm']), M.yes(m['mean_am_ge_eu']),
                 M.pct(m['frac_pred_am_lt_eu']), M.sci(m['premium_pred_rmse_vs_true_am_minus_bs_norm'])]
                for a, m in sanity['models'].items()]
        body.append(["Célváltozó", M.sci(sanity['true_mean_am_minus_eu_norm']), '–', '–', '–'])
        L += M.table(["Modell", "Átlagos különbség", "Átlagban amerikai ≥ európai", "Amerikai < európai",
                      "A különbség RMSE-je a célhoz"], body)

    if bps:
        L += ["", "## Futásidő", ""]
        body = [[CLASS_NAME[a], M.f(d['search_time_s'], 1), M.f(d['search_trial_duration_sum_s'], 1),
                 d['n_search_epochs'], M.f(d['final_train_time_s_american'], 1),
                 M.f(d['final_train_time_s_european'], 1)] for a, d in bps.items()]
        s_search = sum(d['search_time_s'] for d in bps.values())
        s_trial = sum(d['search_trial_duration_sum_s'] or 0 for d in bps.values())
        s_am = sum(d['final_train_time_s_american'] for d in bps.values())
        s_eu = sum(d['final_train_time_s_european'] for d in bps.values())
        body.append(["Összesen", M.f(s_search, 1), M.f(s_trial, 1), sum(d['n_search_epochs'] for d in bps.values()),
                     M.f(s_am, 1), M.f(s_eu, 1)])
        L += M.table(["Architektúra", "Keresés (s)", "Trialidő (s)", "Keresési epoch", "Végső tanítás, amerikai (s)",
                      "Végső tanítás, európai (s)"], body)
        n_ep = sum(d['n_epochs_run'] for d in rows.values())
        L += ["", "## Teljes futásidő", ""]
        L += M.table(["Végső tanítás epochja", "Keresés és végső tanítás (s)", "Keresés és végső tanítás (óra)"],
                     [[n_ep, M.f(s_search + s_am + s_eu, 1), M.f((s_search + s_am + s_eu) / 3600)]], "rrr")

    meta_p = R_ / "campaign_meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding='utf-8'))
        inv = meta.get('invocations', [])
        L += ["", "## Indítások", ""]
        L += M.table(["Indítás", "Kezdés", "Architektúrák"],
                     [[i + 1, x['started'], ', '.join(CLASS_NAME[a] for a in x['archs'])] for i, x in enumerate(inv)],
                     "rll")
        ev = [(a, e) for a, d in bps.items() for e in d.get('resume_events', [])]
        if ev:
            L += ["", "## Folytatás félbehagyott study-ból", ""]
            L += M.table(["Architektúra", "Időpont", "Érvényes trial", "FAIL-re állított trial",
                          "Sampler-állapot visszatöltve", "Utolsó trial a mentés előtt", "Konzisztens"],
                         [[CLASS_NAME[a], e['at'], e['n_valid'],
                           ', '.join(f"#{n}" for n in e['stale_trials_set_fail']) or '–',
                           M.yes(e['sampler_state_restored']), f"#{e['sampler_state_after_trial']}",
                           M.yes(e['sampler_state_consistent'])] for a, e in ev], "llrrlrl")
    L.append("")
    with open(guard_path(R_ / "retune_summary.md"), 'w', encoding='utf-8') as fh:
        fh.write("\n".join(L))
    print(f"[summary] {rel(R_ / 'retune_summary.md')} irva ({len(rows)} retune futas).")


def main(P=None):
    if P is None:
        import run_hpo_american as RA
        ap = argparse.ArgumentParser()
        ap.add_argument('--out-root', type=str, default=None)
        args = ap.parse_args()
        out_root = Path(args.out_root).resolve() if args.out_root else C.AMNN_DIR
        C.set_allowed_roots(out_root)
        P = RA.Paths(out_root)
        P.ensure()
    rows, bps = load_retune(P)
    print(f"[analyze] {len(rows)} retune futas, {len(bps)} kesz architektura.")
    call = call_tpe()
    df = levels_table(P, rows)
    imp = improvement_table(P, df)
    hp = hparam_table(P, bps, call)
    st, an = search_tables(P, bps)
    imps = importance_table(P, bps)
    ratio = ratio_table(P, df)
    rank, kt = ranking_tables(P, df, call)
    seg = concat_runs(P, rows, 'segmented', 'segmented_all.csv')
    prem = concat_runs(P, rows, 'premium_buckets', 'premium_buckets_all.csv')
    seg_ratio, prem_ratio = segment_ratios(P, seg, prem)
    sanity = sanity_am_vs_eu(P, rows)
    lab = label_uncertainty(P, df)
    proto = protocol_check(P, bps)
    ver = verification(P, rows, bps, df, proto)
    write_md(P, rows, bps, df, imp, hp, st, imps, ratio, rank, kt, prem_ratio, sanity, lab, proto, ver, call)
    print(f"[analyze] protokoll={proto['all_ok']}, param={ver['param_count_match_all']}, "
          f"hp_eu=am={ver['european_hp_bitwise_equal_american_all']}, val_recheck={ver['val_recheck_ok']}, "
          f"szint-sorok={ver['level_rows_bitwise_equal']}, checkpointok={ver['n_checkpoints']}")


if __name__ == '__main__':
    main()
