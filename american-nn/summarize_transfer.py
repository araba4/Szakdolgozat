import sys
sys.dont_write_bytecode = True

import time

import numpy as np
import pandas as pd

import common as C
from common import (ARCH_ORDER, CLASS_NAME, DEVICE, HAS_DROPOUT, guard_path, write_json, read_json,
                    rel, load_model)

TARGET_ORDER = ['american', 'european']
LEVELS = ['uniform_p30', 'transfer_p30']
TLAB = {'american': 'amerikai', 'european': 'európai'}
LLAB = {'uniform_p30': 'uniform p30', 'transfer_p30': 'transfer p30'}

UNIFORM_DIR = C.AMNN_DIR / "results" / "uniform"
HPO_P30_DIR = C.REPO_ROOT / "hpo" / "results" / "p30"

CMP_COLS = ['best_val_mse', 'test_mse', 'test_rmse', 'test_mae', 'test_r2', 'test_max_error',
            'test_mape_pct_nem_fo_metrika', 'test_usd_rmse', 'test_usd_mae', 'test_usd_max_error',
            'best_epoch', 'n_epochs_run', 'hit_max_epochs', 'train_seconds', 'mean_epoch_seconds',
            'val_recheck_rel_diff', 'param_count', 'checkpoint_target_col']


def run_name(arch, target):
    return f"{arch}_{target}_transfer_p30"


def _e(v, nd=3):
    return f"{v:.{nd}e}" if isinstance(v, (int, float, np.floating)) and v == v else "n/a"


def _f(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float, np.floating)) and v == v else "n/a"


def short(arch):
    return CLASS_NAME[arch].replace('Pricer', '')


def load_rows(P):
    rows = {}
    for a in ARCH_ORDER:
        for t in TARGET_ORDER:
            f = P.results_dir / f"metrics_{run_name(a, t)}.json"
            if f.exists():
                rows[(a, t)] = read_json(f)
    return rows


def uniform_table():
    df = pd.read_csv(UNIFORM_DIR / "comparison_table.csv", float_precision='round_trip')
    return df[df.patience == 30].copy()


def call_hpo_p30():
    out = {}
    for a in ARCH_ORDER:
        f = HPO_P30_DIR / f"best_params_{a}.json"
        bp = read_json(f)
        out[a] = {'file': rel(f), 'val_mse': bp['final_retrain_best_val_loss'],
                  'test_rmse': bp['hpo_test_metrics']['RMSE'], 'lr': bp['lr'],
                  'weight_decay': bp['weight_decay'], 'dropout': bp['dropout']}
    return out


def comparison_table(P, rows):
    uni = uniform_table()
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
            d = rows.get((a, t))
            if d:
                tm, usd = d['test_metrics'], d['test_metrics_usd']
                hp = d['hyperparameters']
                recs.append({
                    'arch': a, 'model_class': CLASS_NAME[a], 'target': t, 'level': 'transfer_p30',
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
                    'source_file': rel(P.results_dir / f"metrics_{d['run']}.json"),
                })
    df = pd.DataFrame(recs)
    df.to_csv(guard_path(P.results_dir / "comparison_table.csv"), index=False)
    return df


def improvement_table(P, df):
    recs = []
    for a in ARCH_ORDER:
        for t in TARGET_ORDER:
            s = df[(df.arch == a) & (df.target == t)].set_index('level')
            if not set(LEVELS) <= set(s.index):
                continue
            u, x = s.loc['uniform_p30'], s.loc['transfer_p30']
            recs.append({
                'arch': a, 'model_class': CLASS_NAME[a], 'target': t,
                'test_rmse_uniform_p30': u['test_rmse'], 'test_rmse_transfer_p30': x['test_rmse'],
                'test_rmse_improvement_pct': 100 * (1 - x['test_rmse'] / u['test_rmse']),
                'test_rmse_ratio_uniform_over_transfer': u['test_rmse'] / x['test_rmse'],
                'val_mse_uniform_p30': u['best_val_mse'], 'val_mse_transfer_p30': x['best_val_mse'],
                'val_mse_improvement_pct': 100 * (1 - x['best_val_mse'] / u['best_val_mse']),
                'val_mse_ratio_uniform_over_transfer': u['best_val_mse'] / x['best_val_mse'],
                'test_max_error_uniform_p30': u['test_max_error'], 'test_max_error_transfer_p30': x['test_max_error'],
                'transfer_worse_test_rmse': bool(x['test_rmse'] > u['test_rmse']),
                'transfer_worse_val_mse': bool(x['best_val_mse'] > u['best_val_mse']),
            })
    r = pd.DataFrame(recs)
    r.to_csv(guard_path(P.results_dir / "improvement_vs_uniform.csv"), index=False)
    return r


def ratio_table(P, df):
    recs = []
    for lv in LEVELS:
        for a in ARCH_ORDER:
            s = df[(df.arch == a) & (df.level == lv)].set_index('target')
            if not set(TARGET_ORDER) <= set(s.index):
                continue
            recs.append({'level': lv, 'arch': a, 'model_class': CLASS_NAME[a],
                         'test_rmse_american': s.loc['american', 'test_rmse'],
                         'test_rmse_european': s.loc['european', 'test_rmse'],
                         'ratio_rmse_am_over_eu': s.loc['american', 'test_rmse'] / s.loc['european', 'test_rmse'],
                         'val_mse_american': s.loc['american', 'best_val_mse'],
                         'val_mse_european': s.loc['european', 'best_val_mse'],
                         'ratio_val_mse_am_over_eu': s.loc['american', 'best_val_mse'] / s.loc['european', 'best_val_mse']})
        sub = [r for r in recs if r['level'] == lv]
        if sub:
            recs.append({'level': lv, 'arch': 'MEDIAN', 'model_class': 'medián',
                         'ratio_rmse_am_over_eu': float(np.median([r['ratio_rmse_am_over_eu'] for r in sub])),
                         'ratio_val_mse_am_over_eu': float(np.median([r['ratio_val_mse_am_over_eu'] for r in sub]))})
    r = pd.DataFrame(recs)
    r.to_csv(guard_path(P.results_dir / "ratio_american_vs_european.csv"), index=False)
    return r


def ranking_tables(P, df, call):
    from scipy.stats import kendalltau
    recs = []
    order = {}
    for t in TARGET_ORDER:
        for lv in LEVELS:
            s = df[(df.target == t) & (df.level == lv)]
            for metric in ('test_rmse', 'best_val_mse'):
                srt = s.sort_values(metric)
                order[(t, lv, metric)] = list(srt['arch'])
                for rank, (_, row) in enumerate(srt.iterrows(), 1):
                    recs.append({'target': t, 'level': lv, 'metric': metric, 'rank': rank,
                                 'arch': row['arch'], 'value': row[metric]})
    for metric, key in (('test_rmse', 'test_rmse'), ('best_val_mse', 'val_mse')):
        srt = sorted(call, key=lambda a: call[a][key])
        order[('european_call', 'hpo_p30', metric)] = srt
        for rank, a in enumerate(srt, 1):
            recs.append({'target': 'european_call', 'level': 'hpo_p30', 'metric': metric, 'rank': rank,
                         'arch': a, 'value': call[a][key]})
    rank = pd.DataFrame(recs)
    rank.to_csv(guard_path(P.results_dir / "ranking.csv"), index=False)

    def tau(o1, o2):
        common = [a for a in o1 if a in o2]
        if len(common) < 3:
            return None
        res = kendalltau([o1.index(a) for a in common], [o2.index(a) for a in common])
        return {'tau': float(res.statistic), 'p_value': float(res.pvalue), 'n': len(common)}

    kt = {'note': 'Kendall tau a rangsorok kozott (7 architektura, egy seed: tajekoztato).',
          'call_source': 'hpo/results/p30/best_params_*.json (hpo_test_metrics.RMSE, final_retrain_best_val_loss)',
          'pairs': []}
    for metric in ('test_rmse', 'best_val_mse'):
        pairs = [('american', 'uniform_p30', 'european', 'uniform_p30'),
                 ('american', 'transfer_p30', 'european', 'transfer_p30'),
                 ('american', 'transfer_p30', 'european_call', 'hpo_p30'),
                 ('european', 'transfer_p30', 'european_call', 'hpo_p30'),
                 ('american', 'uniform_p30', 'american', 'transfer_p30')]
        for t1, l1, t2, l2 in pairs:
            o1, o2 = order.get((t1, l1, metric)), order.get((t2, l2, metric))
            if o1 and o2:
                kt['pairs'].append({'metric': metric, 'a': f"{t1} {l1}", 'b': f"{t2} {l2}",
                                    'order_a': o1, 'order_b': o2, **(tau(o1, o2) or {})})
    write_json(P.results_dir / "kendall_tau.json", kt)
    return rank, kt


def concat_transfer(P, rows, prefix, out_name):
    parts = []
    for (a, t), d in rows.items():
        f = P.results_dir / f"{prefix}_{d['run']}.csv"
        if f.exists():
            x = pd.read_csv(f)
            x.insert(1, 'arch', a)
            x.insert(2, 'target', t)
            x.insert(3, 'level', 'transfer_p30')
            parts.append(x)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_csv(guard_path(P.results_dir / out_name), index=False)
    return out


def segment_ratios(P, seg_tr, prem_tr):
    def uni(name):
        d = pd.read_csv(UNIFORM_DIR / name)
        d = d[d.patience == 30].drop(columns=['patience']).copy()
        d.insert(3, 'level', 'uniform_p30')
        return d
    out = {}
    for name, tr, fname in (('segmented_all.csv', seg_tr, 'segment_ratio_am_eu.csv'),
                            ('premium_buckets_all.csv', prem_tr, 'premium_ratio_am_eu.csv')):
        both = pd.concat([uni(name), tr], ignore_index=True) if len(tr) else uni(name)
        recs = []
        for lv in LEVELS:
            for a in ARCH_ORDER:
                for (g, sgm), grp in both[(both.level == lv) & (both.arch == a)].groupby(['group', 'segment'], sort=False):
                    s = grp.set_index('target')
                    if not set(TARGET_ORDER) <= set(s.index):
                        continue
                    recs.append({'level': lv, 'arch': a, 'group': g, 'segment': sgm, 'n': int(s.loc['american', 'n']),
                                 'rmse_american': s.loc['american', 'RMSE'], 'rmse_european': s.loc['european', 'RMSE'],
                                 'ratio_am_over_eu': s.loc['american', 'RMSE'] / s.loc['european', 'RMSE']})
        r = pd.DataFrame(recs)
        r.to_csv(guard_path(P.results_dir / fname), index=False)
        out[fname] = r
    return out['segment_ratio_am_eu.csv'], out['premium_ratio_am_eu.csv']


def sanity_am_vs_eu(P, rows):
    df = C.load_am_split('test')
    K = df['K'].to_numpy()
    d_true = ((df['american_put_price'] - df['european_put_price']) / df['K']).to_numpy()
    out = {'test_file': rel(C.am_split_path('test')),
           'true_mean_am_minus_eu_norm': float(d_true.mean()), 'models': {}}
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
    write_json(P.results_dir / "sanity_american_vs_european.json", out)
    return out


def verification(P, rows, df):
    import torch
    v = {'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'n_runs': len(rows)}
    v['param_count_match_all'] = bool(all(d['param_count_match'] for d in rows.values()))
    hp_rows = []
    for (a, t), d in rows.items():
        bp = read_json(HPO_P30_DIR / f"best_params_{a}.json")
        ck = torch.load(P.ckpt(d['run']), map_location='cpu', weights_only=False)
        ck_hp = ck['extra_meta']['hyperparameters']
        exp_do = bp['dropout'] if HAS_DROPOUT[a] else None
        r = {'run': d['run'], 'source': rel(HPO_P30_DIR / f"best_params_{a}.json"),
             'lr_equal': d['hyperparameters']['lr'] == bp['lr'] == ck_hp['lr'],
             'weight_decay_equal': d['hyperparameters']['weight_decay'] == bp['weight_decay'] == ck_hp['weight_decay'],
             'dropout_equal': d['hyperparameters']['dropout'] == exp_do == ck_hp['dropout']
                              and ck['model_kwargs'].get('dropout') == exp_do,
             'batch_epochs_seed_patience_equal': (d['hyperparameters']['batch_size'] == bp['final_batch_size']
                                                  and d['max_epochs'] == bp['final_max_epochs']
                                                  and d['seed'] == bp['seed'] and d['patience'] == bp['final_patience']),
             'target_col_ok': ck['target_col'] == C.TARGETS[t]['target_col'],
             'checkpoint_source_json': ck['extra_meta'].get('hyperparameter_source_json')}
        r['all_equal'] = bool(r['lr_equal'] and r['weight_decay_equal'] and r['dropout_equal']
                              and r['batch_epochs_seed_patience_equal'] and r['target_col_ok'])
        hp_rows.append(r)
    v['hyperparameters'] = hp_rows
    v['hyperparameters_bitwise_equal_all'] = bool(all(r['all_equal'] for r in hp_rows))
    rd = [d['val_recheck']['rel_diff'] for d in rows.values()]
    v['val_recheck_max_rel_diff'] = float(max(rd)) if rd else None
    v['val_recheck_ok'] = bool(rd and max(rd) < 1e-6)
    ours = pd.read_csv(P.results_dir / "comparison_table.csv", float_precision='round_trip')
    ours = ours[ours.level == 'uniform_p30']
    src = uniform_table()
    cols = [c for c in CMP_COLS if c != 'checkpoint_target_col']
    mism = []
    for _, r in ours.iterrows():
        s = src[(src.arch == r['arch']) & (src.target == r['target'])].iloc[0]
        for c in cols + ['checkpoint_target_col']:
            if not (r[c] == s[c]):
                mism.append({'arch': r['arch'], 'target': r['target'], 'col': c, 'ours': str(r[c]), 'src': str(s[c])})
    v['uniform_rows_source'] = rel(UNIFORM_DIR / "comparison_table.csv")
    v['uniform_rows_n'] = int(len(ours))
    v['uniform_rows_bitwise_equal'] = bool(len(ours) == 14 and not mism)
    v['uniform_rows_mismatches'] = mism
    v['n_hit_max_epochs'] = int(sum(d['hit_max_epochs'] for d in rows.values()))
    write_json(P.results_dir / "verification.json", v)
    return v


RANK_LABEL = {('american', 'uniform_p30'): 'amerikai put, uniform p30',
              ('european', 'uniform_p30'): 'európai put, uniform p30',
              ('american', 'transfer_p30'): 'amerikai put, transfer p30',
              ('european', 'transfer_p30'): 'európai put, transfer p30',
              ('european_call', 'hpo_p30'): 'európai call, TPE p30'}
METRIC_LABEL = {'test_rmse': 'teszt-RMSE', 'best_val_mse': 'val MSE'}


def _rank_label(s):
    t, lv = s.split(' ')
    return RANK_LABEL.get((t, lv), s)


def write_md(P, rows, df, imp, ratio, rank, kt, seg_ratio, prem_ratio, sanity, ver, call):
    import md_tables as M
    R = P.results_dir
    L = []

    L += ["## Teszt-RMSE, val MSE és tanítás", ""]
    L += M.table(["Architektúra", "Cél", "Szint", "Val MSE", "Teszt-RMSE", "Teszt MAE", "Teszt R²", "Teszt max. hiba",
                  "Teszt-RMSE ($)", "Legjobb epoch", "Futott epoch", "200 epochos korlát"],
                 [[r['model_class'], TLAB[r['target']], LLAB[r['level']], M.sci(r['best_val_mse']),
                   M.sci(r['test_rmse']), M.sci(r['test_mae']), M.f(r['test_r2'], 6), M.sci(r['test_max_error']),
                   M.sci(r['test_usd_rmse']), int(r['best_epoch']), int(r['n_epochs_run']),
                   M.yes(bool(r['hit_max_epochs']))] for _, r in df.iterrows()],
                 "lll" + "r" * 9)

    L += ["", "## A transfer p30 változása a uniform p30-hoz képest", ""]
    body = [[CLASS_NAME[r['arch']], TLAB[r['target']], M.sci(r['test_rmse_uniform_p30']),
             M.sci(r['test_rmse_transfer_p30']), M.chg(-r['test_rmse_improvement_pct']),
             M.f(r['test_rmse_ratio_uniform_over_transfer']), M.chg(-r['val_mse_improvement_pct'])]
            for _, r in imp.iterrows()]
    for t in TARGET_ORDER:
        s = imp[imp.target == t]
        if len(s):
            body.append([f"Medián, {TLAB[t]}", '', '', '', M.chg(-s['test_rmse_improvement_pct'].median()),
                         M.f(s['test_rmse_ratio_uniform_over_transfer'].median()), ''])
    L += M.table(["Architektúra", "Cél", "Teszt-RMSE, uniform p30", "Teszt-RMSE, transfer p30",
                  "Teszt-RMSE változása", "Uniform és transfer aránya", "Val MSE változása"], body, "ll" + "r" * 5)

    L += ["", "## Rontott-e az átvitel", ""]
    L += M.table(["Cél", "Romlott teszt-RMSE-jű architektúra"],
                 [[TLAB[t], ', '.join(CLASS_NAME[a] for a in imp[(imp.target == t) & imp.transfer_worse_test_rmse].arch)
                   or 'nincs'] for t in TARGET_ORDER if len(imp[imp.target == t])], "ll")

    L += ["", "## Amerikai és európai teszt-RMSE aránya", ""]
    rr = ratio[ratio.arch != 'MEDIAN']
    body = []
    for a in ARCH_ORDER:
        cells = []
        for lv in LEVELS:
            s = rr[(rr.arch == a) & (rr.level == lv)]
            cells += ([M.sci(s['test_rmse_american'].iloc[0]), M.sci(s['test_rmse_european'].iloc[0]),
                       M.f(s['ratio_rmse_am_over_eu'].iloc[0])] if len(s) else ['–'] * 3)
        body.append([CLASS_NAME[a]] + cells)
    for lab, fn in (('Medián', 'median'), ('Minimum', 'min'), ('Maximum', 'max')):
        cells = []
        for lv in LEVELS:
            s = rr[rr.level == lv]['ratio_rmse_am_over_eu']
            cells += ['', '', M.f(getattr(s, fn)())]
        body.append([lab] + cells)
    L += M.table(["Architektúra", "Amerikai, uniform p30", "Európai, uniform p30", "Arány, uniform p30",
                  "Amerikai, transfer p30", "Európai, transfer p30", "Arány, transfer p30"], body)

    L += ["", "## Rangsor teszt-RMSE szerint", ""]
    rt = rank[rank.metric == 'test_rmse']
    cols = (('american', 'uniform_p30'), ('european', 'uniform_p30'), ('american', 'transfer_p30'),
            ('european', 'transfer_p30'), ('european_call', 'hpo_p30'))
    body = []
    for k in range(1, len(ARCH_ORDER) + 1):
        cells = []
        for t, lv in cols:
            s = rt[(rt.target == t) & (rt.level == lv) & (rt['rank'] == k)]
            cells.append(f"{CLASS_NAME[s.iloc[0]['arch']]} ({M.sci(s.iloc[0]['value'])})" if len(s) else '–')
        body.append([f"{k}."] + cells)
    L += M.table(["Hely"] + [RANK_LABEL[c] for c in cols], body, "rlllll")

    L += ["", "## Kendall τ a rangsorok között", ""]
    L += M.table(["Metrika", "Egyik rangsor", "Másik rangsor", "Kendall τ", "p"],
                 [[METRIC_LABEL.get(p['metric'], p['metric']), _rank_label(p['a']), _rank_label(p['b']),
                   M.f(p.get('tau')), M.f(p.get('p_value'), 3)] for p in kt['pairs']], "lllrr")

    for title, tab in (('Amerikai és európai teszt-RMSE aránya moneyness és lejárat szerint', seg_ratio),
                       ('Amerikai és európai teszt-RMSE aránya a korai lehívási prémium szerint', prem_ratio)):
        if not len(tab):
            continue
        segs = list(dict.fromkeys(tab['segment']))
        L += ["", f"## {title}", ""]
        body = []
        for a in ARCH_ORDER:
            for lv in LEVELS:
                q = tab[(tab.arch == a) & (tab.level == lv)].set_index('segment')
                if len(q):
                    body.append([CLASS_NAME[a], LLAB[lv]] + [M.f(q.loc[s, 'ratio_am_over_eu']) if s in q.index
                                                            else '–' for s in segs])
        L += M.table(["Architektúra", "Szint"] + segs, body, "ll" + "r" * len(segs))

    if sanity.get('models'):
        L += ["", "## Amerikai és európai ár a teszthalmazon (transfer p30)", ""]
        body = [[CLASS_NAME[a], M.sci(m['mean_pred_am_minus_eu_norm']), M.yes(m['mean_am_ge_eu']),
                 M.pct(m['frac_pred_am_lt_eu']), M.sci(m['premium_pred_rmse_vs_true_am_minus_bs_norm'])]
                for a, m in sanity['models'].items()]
        body.append(["Célváltozó", M.sci(sanity['true_mean_am_minus_eu_norm']), '–', '–', '–'])
        L += M.table(["Modell", "Átlagos különbség", "Átlagban amerikai ≥ európai", "Amerikai < európai",
                      "A különbség RMSE-je a célhoz"], body)

    L += ["", "## Ellenőrzések", ""]
    pc = R / "pairing_check.json"
    L += M.table(["Ellenőrzés", "Eredmény"], [
        ["Az amerikai és az európai adathalmaz páros", M.yes(read_json(pc)['all_paired']) if pc.exists() else '–'],
        [f"Paraméterszám = kiindulási v1 = HPO p30 = uniform p30 ({ver['n_runs']} futás)",
         M.yes(ver['param_count_match_all'])],
        ["`lr`, `weight_decay`, `dropout`, batch, epoch, seed és türelem bitre a `best_params_*.json` értéke",
         M.yes(ver['hyperparameters_bitwise_equal_all'])],
        ["Checkpoint-visszatöltés: legnagyobb relatív eltérés a mentett val MSE-től",
         M.sci(ver['val_recheck_max_rel_diff'])],
        [f"A uniform p30 sorok bitre a `{ver['uniform_rows_source']}` soraival ({ver['uniform_rows_n']} sor)",
         M.yes(ver['uniform_rows_bitwise_equal'])],
        ["A 200 epochos korlátig futott tanítások", ver['n_hit_max_epochs']]])

    L += ["", "## Átvitt hiperparaméterek", ""]
    L += M.table(["Architektúra", "lr", "weight_decay", "dropout", "Forrás", "Call TPE p30 val MSE",
                  "Call TPE p30 teszt-RMSE"],
                 [[CLASS_NAME[a], M.sci(call[a]['lr']), M.sci(call[a]['weight_decay']),
                   M.f(call[a]['dropout'], 4) if call[a]['dropout'] is not None else '–', f"`{call[a]['file']}`",
                   M.sci(call[a]['val_mse']), M.sci(call[a]['test_rmse'])] for a in ARCH_ORDER], "lrrrlrr")

    tr = df[df.level == 'transfer_p30']
    tot = tr['train_seconds'].sum() if len(tr) else 0
    L += ["", "## Futásidő (transfer p30)", ""]
    L += M.table(["Tanítás", "Epoch", "Összes idő (s)", "Összes idő (óra)", "Átlagos epochidő (s)"],
                 [[len(tr), int(tr['n_epochs_run'].sum()) if len(tr) else 0, f"{tot:.0f}", M.f(tot / 3600),
                   M.f(tr['mean_epoch_seconds'].mean())]], "rrrrr")
    L.append("")
    with open(guard_path(R / "transfer_summary.md"), 'w', encoding='utf-8') as fh:
        fh.write("\n".join(L))
    print(f"[summary] {rel(R / 'transfer_summary.md')} irva ({len(rows)} transfer futas).")


def main(P):
    rows = load_rows(P)
    print(f"[summary] {len(rows)} kesz transfer futas.")
    call = call_hpo_p30()
    df = comparison_table(P, rows)
    imp = improvement_table(P, df)
    ratio = ratio_table(P, df)
    rank, kt = ranking_tables(P, df, call)
    seg_tr = concat_transfer(P, rows, 'segmented', 'segmented_all.csv')
    prem_tr = concat_transfer(P, rows, 'premium_buckets', 'premium_buckets_all.csv')
    seg_ratio, prem_ratio = segment_ratios(P, seg_tr, prem_tr)
    sanity = sanity_am_vs_eu(P, rows)
    ver = verification(P, rows, df)
    write_md(P, rows, df, imp, ratio, rank, kt, seg_ratio, prem_ratio, sanity, ver, call)
    print(f"[verify] param={ver['param_count_match_all']}, hp={ver['hyperparameters_bitwise_equal_all']}, "
          f"val_recheck_ok={ver['val_recheck_ok']}, uniform_rows={ver['uniform_rows_bitwise_equal']}")
