import sys
sys.dont_write_bytecode = True

import time

import numpy as np
import pandas as pd
import torch

import common as C
from common import (ARCH_ORDER, CLASS_NAME, DEVICE, guard_path, write_json, read_json, rel,
                    load_model, compute_metrics)

TARGET_ORDER = ['american', 'european']
PATIENCES = [10, 30]
TLAB = {'american': 'amerikai', 'european': 'európai'}


def run_name(arch, target, patience):
    return f"{arch}_{target}_p{patience}"


def _e(v, nd=3):
    return f"{v:.{nd}e}" if isinstance(v, (int, float)) and v == v else "n/a"


def _f(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else "n/a"


def short(arch):
    return CLASS_NAME[arch].replace('Pricer', '')


def load_rows(P):
    rows = {}
    for a in ARCH_ORDER:
        for t in TARGET_ORDER:
            for p in PATIENCES:
                f = P.results_dir / f"metrics_{run_name(a, t, p)}.json"
                if f.exists():
                    rows[(a, t, p)] = read_json(f)
    return rows


def onlab_call_reference(P):
    from src.evaluate import evaluate_model
    out = {'note': 'Önlab v1 (*_best.pt), európai CALL, call_price_norm, Önlab test.parquet; '
                   'csak tájékoztató, a fő összevetés put–put.',
           'test_file': rel(C.onlab_split_path('test')), 'archs': {}}
    for a in ARCH_ORDER:
        ck = C.ONLAB_MODELS_DIR / f"{a}_best.pt"
        res = evaluate_model(str(ck), str(C.onlab_split_path('test')), device=DEVICE, verbose=False)
        st = torch.load(ck, map_location='cpu', weights_only=False)
        out['archs'][a] = {'checkpoint': rel(ck), 'val_loss': st['val_loss'],
                           'best_epoch': st['best_epoch'], 'test_metrics': res['metrics']}
    write_json(P.results_dir / "onlab_call_v1_reference.json", out)
    return out


def reproducibility(P):
    out = {}
    for a in ARCH_ORDER:
        for t in TARGET_ORDER:
            f10 = P.results_dir / f"training_history_{run_name(a, t, 10)}.csv"
            f30 = P.results_dir / f"training_history_{run_name(a, t, 30)}.csv"
            if not (f10.exists() and f30.exists()):
                continue
            h10 = pd.read_csv(f10, float_precision='round_trip')
            h30 = pd.read_csv(f30, float_precision='round_trip')
            n = min(len(h10), len(h30))
            r = {'n_epochs_p10': len(h10), 'n_epochs_p30': len(h30), 'n_common_epochs': n, 'columns': {}}
            first_any = None
            for col in ('train_loss', 'val_loss', 'lr'):
                x, y = h10[col].to_numpy()[:n], h30[col].to_numpy()[:n]
                neq = np.nonzero(x != y)[0]
                first = int(neq[0]) + 1 if len(neq) else None
                r['columns'][col] = {'n_differing': int(len(neq)), 'first_differing_epoch': first,
                                     'max_abs_diff': float(np.max(np.abs(x - y))) if n else None}
                if first is not None:
                    first_any = first if first_any is None else min(first_any, first)
            r['bitwise_identical'] = first_any is None
            r['first_differing_epoch'] = first_any
            out[f"{a}_{t}"] = r
    write_json(P.results_dir / "reproducibility_check.json", out)
    return out


def sanity_am_vs_eu(P, rows):
    df = C.load_am_split('test')
    K = df['K'].to_numpy()
    true_am = (df['american_put_price'] / df['K']).to_numpy()
    true_eu = (df['european_put_price'] / df['K']).to_numpy()
    d_true = true_am - true_eu
    out = {'test_file': rel(C.am_split_path('test')),
           'true_targets': {
               'mean_am_minus_eu_norm': float(d_true.mean()),
               'frac_am_lt_eu': float((d_true < 0).mean()),
               'n_am_lt_eu': int((d_true < 0).sum()),
               'min_am_minus_eu_norm': float(d_true.min()),
               'note': 'CRR-amerikai < BS-európai a fa diszkretizációs hibája miatt lehet.'},
           'models': {}}
    ds_am = C.PricingDataset.from_split('test', 'american', device=DEVICE)
    for a in ARCH_ORDER:
        for p in PATIENCES:
            if (a, 'american', p) not in rows or (a, 'european', p) not in rows:
                continue
            m_am, _ = load_model(str(P.ckpt(run_name(a, 'american', p))), device=DEVICE)
            m_eu, _ = load_model(str(P.ckpt(run_name(a, 'european', p))), device=DEVICE)
            pa, pe = C.predict(m_am, ds_am), C.predict(m_eu, ds_am)
            d = pa.astype(np.float64) - pe.astype(np.float64)
            out['models'][f"{a}_p{p}"] = {
                'arch': a, 'patience': p,
                'mean_pred_am_norm': float(pa.mean()), 'mean_pred_eu_norm': float(pe.mean()),
                'mean_pred_am_minus_eu_norm': float(d.mean()),
                'mean_true_am_minus_eu_norm': float(d_true.mean()),
                'frac_pred_am_lt_eu': float((d < 0).mean()),
                'n_pred_am_lt_eu': int((d < 0).sum()),
                'min_pred_am_minus_eu_norm': float(d.min()),
                'mean_am_ge_eu': bool(pa.mean() >= pe.mean()),
                'premium_pred_rmse_vs_true_am_minus_bs_norm': float(np.sqrt(np.mean((d - d_true) ** 2))),
                'usd_mean_pred_am_minus_eu': float(np.mean(d * K)),
            }
    write_json(P.results_dir / "sanity_american_vs_european.json", out)
    return out


def comparison_table(P, rows):
    recs = []
    for (a, t, p), d in rows.items():
        tm, usd = d['test_metrics'], d['test_metrics_usd']
        recs.append({
            'arch': a, 'model_class': d['model_class'], 'target': t, 'patience': p,
            'param_count': d['param_count'], 'param_count_match_onlab_v1': d['param_count_check']['param_count_match'],
            'best_val_mse': d['best_val_loss'],
            'test_mse': tm['MSE'], 'test_rmse': tm['RMSE'], 'test_mae': tm['MAE'], 'test_r2': tm['R2'],
            'test_max_error': tm['max_error'], 'test_mape_pct_nem_fo_metrika': tm['MAPE (%)'],
            'test_usd_rmse': usd['RMSE'], 'test_usd_mae': usd['MAE'], 'test_usd_max_error': usd['max_error'],
            'best_epoch': d['best_epoch'], 'n_epochs_run': d['n_epochs_run'],
            'hit_max_epochs': d['hit_max_epochs'], 'train_seconds': d['train_seconds'],
            'mean_epoch_seconds': d['mean_epoch_seconds'],
            'val_recheck_rel_diff': d['val_recheck']['rel_diff'],
            'checkpoint_target_col': d['checkpoint_target_col'],
            'metrics_file': rel(P.results_dir / f"metrics_{d['run']}.json"),
        })
    df = pd.DataFrame(recs)
    if len(df):
        df['_o'] = df['arch'].map(ARCH_ORDER.index)
        df = df.sort_values(['_o', 'target', 'patience']).drop(columns='_o')
    df.to_csv(guard_path(P.results_dir / "comparison_table.csv"), index=False)
    return df


def ratio_table(P, df):
    recs = []
    for a in ARCH_ORDER:
        for p in PATIENCES:
            s = df[(df.arch == a) & (df.patience == p)].set_index('target')
            if not {'american', 'european'} <= set(s.index):
                continue
            recs.append({'arch': a, 'model_class': CLASS_NAME[a], 'patience': p,
                         'test_rmse_american': s.loc['american', 'test_rmse'],
                         'test_rmse_european': s.loc['european', 'test_rmse'],
                         'ratio_rmse_am_over_eu': s.loc['american', 'test_rmse'] / s.loc['european', 'test_rmse'],
                         'val_mse_american': s.loc['american', 'best_val_mse'],
                         'val_mse_european': s.loc['european', 'best_val_mse'],
                         'ratio_val_mse_am_over_eu': s.loc['american', 'best_val_mse'] / s.loc['european', 'best_val_mse']})
    r = pd.DataFrame(recs)
    r.to_csv(guard_path(P.results_dir / "ratio_american_vs_european.csv"), index=False)
    return r


def ranking_table(P, df):
    recs = []
    for t in TARGET_ORDER:
        for p in PATIENCES:
            s = df[(df.target == t) & (df.patience == p)]
            for metric in ('test_rmse', 'best_val_mse'):
                for rank, (_, row) in enumerate(s.sort_values(metric).iterrows(), 1):
                    recs.append({'target': t, 'patience': p, 'metric': metric, 'rank': rank,
                                 'arch': row['arch'], 'value': row[metric]})
    r = pd.DataFrame(recs)
    r.to_csv(guard_path(P.results_dir / "ranking.csv"), index=False)
    return r


def concat_csvs(P, rows, prefix, out_name):
    parts = []
    for (a, t, p), d in rows.items():
        f = P.results_dir / f"{prefix}_{d['run']}.csv"
        if f.exists():
            x = pd.read_csv(f)
            x.insert(1, 'arch', a)
            x.insert(2, 'target', t)
            x.insert(3, 'patience', p)
            parts.append(x)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_csv(guard_path(P.results_dir / out_name), index=False)
    return out


def write_md(P, rows, df, ratio, rank, repro, sanity, onlab, seg, prem, extra):
    import md_tables as M
    R = P.results_dir
    L = []

    L += ["## Teszt-RMSE, val MSE és tanítás", ""]
    L += M.table(["Architektúra", "Cél", "Türelem", "Paraméter", "Val MSE", "Teszt-RMSE", "Teszt R²",
                  "Teszt max. hiba", "Teszt-RMSE ($)", "Legjobb epoch", "Futott epoch", "200 epochos korlát",
                  "Tanítási idő (s)"],
                 [[r['model_class'], TLAB[r['target']], r['patience'], M.p(r['param_count']),
                   M.sci(r['best_val_mse']), M.sci(r['test_rmse']), M.f(r['test_r2'], 6),
                   M.sci(r['test_max_error']), M.sci(r['test_usd_rmse']), r['best_epoch'], r['n_epochs_run'],
                   M.yes(bool(r['hit_max_epochs'])), f"{r['train_seconds']:.0f}"] for _, r in df.iterrows()],
                 "llr" + "r" * 10)

    L += ["", "## Amerikai és európai teszt-RMSE aránya", ""]
    rows_r = []
    for a in ARCH_ORDER:
        s = ratio[ratio.arch == a].set_index('patience') if len(ratio) else pd.DataFrame()
        if not len(s):
            continue
        cells = []
        for p in PATIENCES:
            cells += ([M.sci(s.loc[p, 'test_rmse_american']), M.sci(s.loc[p, 'test_rmse_european']),
                       M.f(s.loc[p, 'ratio_rmse_am_over_eu'])] if p in s.index else ['–'] * 3)
        rows_r.append([CLASS_NAME[a]] + cells)
    for lab, fn in (('Medián', 'median'), ('Minimum', 'min'), ('Maximum', 'max')):
        cells = []
        for p in PATIENCES:
            s = ratio[ratio.patience == p]['ratio_rmse_am_over_eu']
            cells += ['', '', M.f(getattr(s, fn)())]
        rows_r.append([lab] + cells)
    L += M.table(["Architektúra", "Amerikai, p10", "Európai, p10", "Arány, p10",
                  "Amerikai, p30", "Európai, p30", "Arány, p30"], rows_r)

    L += ["", "## Rangsor teszt-RMSE szerint", ""]
    rr = rank[rank.metric == 'test_rmse'] if len(rank) else rank
    cols = (('american', 10), ('european', 10), ('american', 30), ('european', 30))
    rows_k = []
    for k in range(1, len(ARCH_ORDER) + 1):
        cells = []
        for t, p in cols:
            s = rr[(rr.target == t) & (rr.patience == p) & (rr['rank'] == k)] if len(rr) else []
            cells.append(f"{CLASS_NAME[s.iloc[0]['arch']]} ({M.sci(s.iloc[0]['value'])})" if len(s) else '–')
        rows_k.append([f"{k}."] + cells)
    L += M.table(["Hely", "Amerikai, p10", "Európai, p10", "Amerikai, p30", "Európai, p30"], rows_k, "rllll")

    L += ["", "## Az amerikai és az európai rangsor Spearman-korrelációja", ""]
    from scipy.stats import spearmanr
    rows_s = []
    for p in PATIENCES:
        am = list(rr[(rr.target == 'american') & (rr.patience == p)].sort_values('rank')['arch'])
        eu = list(rr[(rr.target == 'european') & (rr.patience == p)].sort_values('rank')['arch'])
        common = [a for a in am if a in eu]
        rho = (spearmanr([am.index(a) for a in common], [eu.index(a) for a in common]).correlation
               if len(common) > 2 else float('nan'))
        rows_s.append([f"p{p}", CLASS_NAME[am[0]], CLASS_NAME[eu[0]], M.f(rho)])
    L += M.table(["Türelem", "Legjobb, amerikai", "Legjobb, európai", "Spearman ρ"], rows_s, "lllr")

    if len(seg):
        for group, title in (('moneyness', 'moneyness szerint'), ('lejarat', 'lejárat szerint')):
            s = seg[(seg.group == group) & (seg.patience == 30)]
            segs = list(dict.fromkeys(s['segment']))
            L += ["", f"## Teszt-RMSE {title} (p30)", ""]
            body = []
            for a in ARCH_ORDER:
                for t in TARGET_ORDER:
                    q = s[(s.arch == a) & (s.target == t)].set_index('segment')
                    if len(q):
                        body.append([CLASS_NAME[a], TLAB[t]] + [M.sci(q.loc[x, 'RMSE']) for x in segs])
            L += M.table(["Architektúra", "Cél"] + [f"{x}, n = {M.p(s[s.segment == x]['n'].iloc[0])}" for x in segs],
                         body, "ll" + "r" * len(segs))

    if len(prem):
        s = prem[prem.patience == 30]
        segs = list(dict.fromkeys(s['segment']))
        L += ["", "## Teszt-RMSE a korai lehívási prémium szerint (p30)", ""]
        body = []
        for a in ARCH_ORDER:
            for t in TARGET_ORDER:
                q = s[(s.arch == a) & (s.target == t)].set_index('segment')
                if len(q):
                    body.append([CLASS_NAME[a], TLAB[t]] + [M.sci(q.loc[x, 'RMSE']) for x in segs])
        L += M.table(["Architektúra", "Cél"] + [f"{x}, n = {M.p(s[s.segment == x]['n'].iloc[0])}" for x in segs],
                     body, "ll" + "r" * len(segs))
        L += ["", "## Amerikai és európai teszt-RMSE aránya a korai lehívási prémium szerint (p30)", ""]
        body = []
        for a in ARCH_ORDER:
            qa = s[(s.arch == a) & (s.target == 'american')].set_index('segment')
            qe = s[(s.arch == a) & (s.target == 'european')].set_index('segment')
            if len(qa) and len(qe):
                body.append([CLASS_NAME[a]] + [M.f(qa.loc[x, 'RMSE'] / qe.loc[x, 'RMSE']) for x in segs])
        L += M.table(["Architektúra"] + [x.split(' ')[0] for x in segs], body)

    L += ["", "## A p10 és a p30 tanítási görbéje a közös epochokon", ""]
    L += M.table(["Futás", "Közös epoch", "Bitre azonos", "Első eltérő epoch"],
                 [[k, r['n_common_epochs'], M.yes(r['bitwise_identical']), r['first_differing_epoch'] or '–']
                  for k, r in repro.items()])

    tt = sanity['true_targets']
    L += ["", "## Amerikai és európai ár a teszthalmazon", ""]
    body = [[f"{CLASS_NAME[m['arch']]}, p{m['patience']}", M.f(m['mean_pred_am_norm'], 6),
             M.f(m['mean_pred_eu_norm'], 6), M.sci(m['mean_pred_am_minus_eu_norm']),
             M.pct(m['frac_pred_am_lt_eu']), M.sci(m['premium_pred_rmse_vs_true_am_minus_bs_norm'])]
            for m in sanity['models'].values()]
    body.append(["Célváltozó", "–", "–", M.sci(tt['mean_am_minus_eu_norm']), M.pct(tt['frac_am_lt_eu'], 4), "–"])
    L += M.table(["Modell", "Átlagos amerikai ár/K", "Átlagos európai ár/K", "Átlagos különbség",
                  "Amerikai < európai", "A különbség RMSE-je a célhoz"], body)

    L += ["", "## Ellenőrzések", ""]
    chk = [["Az amerikai és az európai adathalmaz páros", M.yes(extra.get('all_paired'))],
           ["Checkpoint-visszatöltés: legnagyobb relatív eltérés a mentett val MSE-től",
            M.sci(df['val_recheck_rel_diff'].max())],
           ["A checkpointok célváltozója helyes", M.yes(extra.get('target_col_ok'))],
           [f"Paraméterszám = kiindulási v1 ({len(df)} futás)", M.yes(bool(df['param_count_match_onlab_v1'].all()))]]
    if extra.get('loop_check') is not None:
        lc = extra['loop_check']
        chk.append([f"A kiindulási `train_model` és a `train_loop` bitre azonos ({lc['arch']}, {lc['n_epochs']} epoch)",
                    M.yes(lc['all_equal'])])
    L += M.table(["Ellenőrzés", "Eredmény"], chk)

    L += ["", "## Európai call, kiindulási v1 modellek (tájékoztató)", ""]
    body = []
    for a in ARCH_ORDER:
        o = onlab['archs'].get(a)
        pe, pa = rows.get((a, 'european', 10)), rows.get((a, 'american', 10))
        body.append([CLASS_NAME[a], M.sci(o['val_loss']), M.sci(o['test_metrics']['RMSE']),
                     M.sci(pe['test_metrics']['RMSE']) if pe else '–', M.sci(pa['test_metrics']['RMSE']) if pa else '–'])
    L += M.table(["Architektúra", "Call v1 val MSE", "Call v1 teszt-RMSE", "Európai put p10 teszt-RMSE",
                  "Amerikai put p10 teszt-RMSE"], body)

    tot = df['train_seconds'].sum() if len(df) else 0
    L += ["", "## Futásidő", ""]
    L += M.table(["Tanítás", "Összes idő (s)", "Összes idő (óra)", "Átlagos epochidő (s)"],
                 [[len(df), f"{tot:.0f}", M.f(tot / 3600), M.f(df['mean_epoch_seconds'].mean())]], "rrrr")
    L.append("")
    with open(guard_path(R / "uniform_baseline_summary.md"), 'w', encoding='utf-8') as fh:
        fh.write("\n".join(L))
    print(f"[summary] {rel(R / 'uniform_baseline_summary.md')} irva ({len(df)} futas).")


def main(P):
    rows = load_rows(P)
    print(f"[summary] {len(rows)} kesz futas.")
    df = comparison_table(P, rows)
    ratio = ratio_table(P, df) if len(df) else pd.DataFrame()
    rank = ranking_table(P, df) if len(df) else pd.DataFrame()
    seg = concat_csvs(P, rows, 'segmented', 'segmented_all.csv')
    prem = concat_csvs(P, rows, 'premium_buckets', 'premium_buckets_all.csv')
    repro = reproducibility(P)
    sanity = sanity_am_vs_eu(P, rows)
    onlab = onlab_call_reference(P)
    pc = P.results_dir / "pairing_check.json"
    lc = P.results_dir / "loop_check.json"
    extra = {'all_paired': read_json(pc)['all_paired'] if pc.exists() else None,
             'loop_check': read_json(lc) if lc.exists() else None,
             'target_col_ok': all(d['checkpoint_target_col'] == C.TARGETS[t]['target_col']
                                  for (a, t, p), d in rows.items())}
    if len(df):
        write_md(P, rows, df, ratio, rank, repro, sanity, onlab, seg, prem, extra)
