import sys
sys.dont_write_bytecode = True

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HPO_DIR = Path(__file__).resolve().parent.parent / "hpo"
REPO_ROOT = HPO_DIR.parent
ONLAB_RESULTS = REPO_ROOT / "Önlab eredmények" / "results"
TPE_DIR = HPO_DIR / "results"
TPE_P30_DIR = TPE_DIR / "p30"
RS_DIR = TPE_DIR / "random"
RS_LOGS = HPO_DIR / "logs" / "random"
OUT_DIR = RS_DIR / "comparison"

ARCH_ORDER = ['mlp', 'deep_mlp', 'dense_mlp', 'resnet', 'gelu_resnet', 'highway', 'finn']
CLASS_NAME = {'mlp': 'MLPPricer', 'deep_mlp': 'DeepMLPPricer', 'dense_mlp': 'DenseMLPPricer',
              'resnet': 'ResNetPricer', 'gelu_resnet': 'GELUResNetPricer',
              'highway': 'HighwayPricer', 'finn': 'FINNPricer'}
N_STARTUP = 10
WD_LOW = 1e-6
VALID = ('COMPLETE', 'PRUNED')


def relp(path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def guard_path(path) -> Path:
    p = Path(path).resolve()
    try:
        p.relative_to(OUT_DIR.resolve())
    except ValueError:
        raise RuntimeError(f"guard_path: tiltott irasi cel (csak {relp(OUT_DIR)} ala szabad irni): {relp(p)}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_json(p: Path):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def trial_budget_from_source() -> dict:
    tree = ast.parse((HPO_DIR / "run_hpo.py").read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'TRIAL_BUDGET' for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("TRIAL_BUDGET nem talalhato a run_hpo.py-ban")


def load_trials(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, float_precision='round_trip')
    df['duration_s'] = pd.to_timedelta(df['duration']).dt.total_seconds()
    df['datetime_start'] = pd.to_datetime(df['datetime_start'])
    df['datetime_complete'] = pd.to_datetime(df['datetime_complete'])
    return df.sort_values('number').reset_index(drop=True)


def load_arch(arch: str) -> dict:
    d = dict(arch=arch, cls=CLASS_NAME[arch])
    d['tpe_all'] = load_trials(TPE_DIR / f"trials_{arch}.csv")
    d['rs_all'] = load_trials(RS_DIR / f"trials_{arch}.csv")
    d['tpe'] = d['tpe_all'][d['tpe_all'].state.isin(VALID)].reset_index(drop=True)
    d['rs'] = d['rs_all'][d['rs_all'].state.isin(VALID)].reset_index(drop=True)
    d['tpe_curves'] = pd.read_csv(TPE_DIR / f"search_curves_{arch}.csv", float_precision='round_trip')
    d['rs_curves'] = pd.read_csv(RS_DIR / f"search_curves_{arch}.csv", float_precision='round_trip')
    d['tpe_best'] = read_json(TPE_DIR / f"best_params_{arch}.json")
    d['tpe_p30'] = read_json(TPE_P30_DIR / f"best_params_{arch}.json")
    d['rs_best'] = read_json(RS_DIR / f"best_params_{arch}.json")
    d['tpe_imp'] = read_json(TPE_DIR / f"param_importance_{arch}.json")
    d['rs_imp'] = read_json(RS_DIR / f"param_importance_{arch}.json")
    d['rs_pc'] = read_json(RS_DIR / f"param_count_check_{arch}.json")
    return d


def _eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def completeness(D: dict, budget: dict) -> list:
    rows = []
    err_logs = sorted(p.name for p in RS_LOGS.glob("*_ERROR.log"))
    for arch in ARCH_ORDER:
        d = D[arch]
        rs_all, tpe_all = d['rs_all'], d['tpe_all']
        n_valid_rs = int(rs_all.state.isin(VALID).sum())
        n_valid_tpe = int(tpe_all.state.isin(VALID).sum())
        n_fail_rs = int((rs_all.state == 'FAIL').sum())
        pcols = [c for c in tpe_all.columns if c.startswith('params_')]
        n_par = n_state = n_val = 0
        for k in range(N_STARTUP):
            a = tpe_all[tpe_all.number == k].iloc[0]
            b = rs_all[rs_all.number == k].iloc[0]
            n_par += all(_eq(a[c], b[c]) for c in pcols)
            n_state += a.state == b.state
            n_val += _eq(a.value, b.value)
        after = 0
        for k in range(N_STARTUP, budget[arch]):
            a = tpe_all[tpe_all.number == k]
            b = rs_all[rs_all.number == k]
            if len(a) and len(b) and all(_eq(a.iloc[0][c], b.iloc[0][c]) for c in pcols):
                after += 1
        rows.append(dict(
            arch=arch, cls=d['cls'], budget=budget[arch],
            n_valid_rs=n_valid_rs, n_valid_tpe=n_valid_tpe, n_fail_rs=n_fail_rs,
            budget_ok=(n_valid_rs == budget[arch] and n_valid_tpe == budget[arch]),
            param_match=bool(d['rs_pc']['match']) and bool(d['rs_best']['param_count_match']),
            param_count=d['rs_best']['param_count'],
            startup_params=n_par, startup_state=n_state, startup_value=n_val,
            startup_ok=(n_par == n_state == n_val == N_STARTUP),
            after_startup_equal=after,
            final_patience_ok=(d['rs_best']['final_patience'] == 30 and d['tpe_p30']['final_patience'] == 30),
        ))
    return rows, err_logs


def anytime(trials: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    t = trials.copy()
    t['epochs'] = epochs_per_trial(curves, t).values
    t['cum_epochs'] = t['epochs'].cumsum()
    comp_val = t['value'].where(t.state == 'COMPLETE')
    t['best'] = comp_val.cummin().ffill()
    t['cum_time_s'] = t['duration_s'].cumsum()
    return t


def first_reach(at: pd.DataFrame, target: float):
    hit = at[at['best'] <= target]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return dict(trial=int(r.number), n_trials=int(at.index[at.number == r.number][0]) + 1,
                cum_time_s=float(r.cum_time_s), cum_epochs=int(r.cum_epochs))


def epochs_per_trial(curves: pd.DataFrame, trials: pd.DataFrame) -> pd.Series:
    n = curves.groupby('trial_number').size()
    return n.reindex(trials.number).fillna(0).astype(int)


def prune_stats(trials: pd.DataFrame, curves: pd.DataFrame, best_overall: float) -> dict:
    ep = epochs_per_trial(curves, trials)
    t = trials.assign(epochs=ep.values)
    post = t[t.number >= N_STARTUP]
    start_best = t[(t.number < N_STARTUP) & (t.state == 'COMPLETE')].value.min()
    return dict(
        n_complete=int((t.state == 'COMPLETE').sum()), n_pruned=int((t.state == 'PRUNED').sum()),
        complete_ratio=float((t.state == 'COMPLETE').mean()),
        mean_epochs=float(t.epochs.mean()), total_epochs=int(t.epochs.sum()),
        post_n=len(post), post_complete=int((post.state == 'COMPLETE').sum()),
        post_mean_epochs=float(post.epochs.mean()),
        post_full40=int((post.epochs >= 40).sum()),
        post_better_than_startup=int(((post.state == 'COMPLETE') & (post.value < start_best)).sum()),
        post_within2x=int(((post.state == 'COMPLETE') & (post.value <= 2 * best_overall)).sum()),
        startup_best=float(start_best),
    )


def rel(a, b):
    return (a - b) / b * 100.0


def compute(D: dict) -> dict:
    R = {}
    for arch in ARCH_ORDER:
        d = D[arch]
        r = dict(arch=arch, cls=d['cls'])
        at_t, at_r = anytime(d['tpe'], d['tpe_curves']), anytime(d['rs'], d['rs_curves'])
        st_t, st_r = at_t[at_t.number < N_STARTUP], at_r[at_r.number < N_STARTUP]
        assert st_t.epochs.sum() == st_r.epochs.sum()
        r['startup_epochs'] = int(st_t.epochs.sum())
        r['startup_time_tpe'], r['startup_time_rs'] = float(st_t.duration_s.sum()), float(st_r.duration_s.sum())
        r['sec_per_epoch_tpe'] = float(at_t.duration_s.sum() / at_t.epochs.sum())
        r['sec_per_epoch_rs'] = float(at_r.duration_s.sum() / at_r.epochs.sum())
        r['epochs_tpe'], r['epochs_rs'] = int(at_t.epochs.sum()), int(at_r.epochs.sum())
        r['at_tpe'], r['at_rs'] = at_t, at_r
        r['tpe_search_best'] = float(at_t.best.iloc[-1])
        r['rs_search_best'] = float(at_r.best.iloc[-1])
        r['tpe_min_value'] = float(d['tpe'].value.min())
        r['rs_min_value'] = float(d['rs'].value.min())
        r['tpe_best_trial'] = int(d['tpe'].loc[d['tpe'].value.idxmin(), 'number'])
        r['rs_best_trial'] = int(d['rs'].loc[d['rs'].value.idxmin(), 'number'])
        r['startup_best'] = float(at_t[at_t.number < N_STARTUP].best.iloc[-1])
        r['rs_reach_tpe'] = first_reach(at_r, r['tpe_search_best'])
        r['tpe_reach_rs'] = first_reach(at_t, r['rs_search_best'])
        r['tpe_total_time'] = float(at_t.cum_time_s.iloc[-1])
        r['rs_total_time'] = float(at_r.cum_time_s.iloc[-1])
        tp, rb = d['tpe_p30'], d['rs_best']
        r['onlab_ref'] = tp['onlab_best_ref']
        r['onlab_val'] = tp['onlab_best_val_loss']
        r['onlab_test'] = tp['onlab_best_test_metrics']
        r['onlab_epochs'] = (tp['onlab_best_epoch'], tp['onlab_best_n_epochs_run'])
        r['tpe_val'] = tp['final_retrain_best_val_loss']
        r['tpe_test'] = tp['hpo_test_metrics']
        r['tpe_epochs'] = (tp['final_retrain_best_epoch'], tp['final_retrain_n_epochs_run'])
        r['rs_val'] = rb['final_retrain_best_val_loss']
        r['rs_test'] = rb['final_test_metrics']
        r['rs_epochs'] = (rb['final_retrain_best_epoch'], rb['final_retrain_n_epochs_run'])
        assert rb['tpe_p30_best_val_loss'] == tp['final_retrain_best_val_loss']
        assert rb['tpe_p30_test_metrics'] == tp['hpo_test_metrics']
        sfx = '' if r['onlab_ref'] == 'v1' else '_v2'
        oj = read_json(ONLAB_RESULTS / f"training_history_{d['cls']}{sfx}.json")
        r['onlab_crosscheck'] = (oj['best_val_loss'] == r['onlab_val'])
        best_overall = min(r['tpe_search_best'], r['rs_search_best'])
        r['ps_tpe'] = prune_stats(d['tpe'], d['tpe_curves'], best_overall)
        r['ps_rs'] = prune_stats(d['rs'], d['rs_curves'], best_overall)
        has_do = 'params_dropout' in d['tpe'].columns
        r['has_dropout'] = has_do
        r['hp_tpe'] = dict(lr=d['tpe_best']['lr'], weight_decay=d['tpe_best']['weight_decay'],
                           dropout=d['tpe_best']['dropout'])
        r['hp_rs'] = dict(lr=rb['lr'], weight_decay=rb['weight_decay'], dropout=rb['dropout'])
        for key, tr in (('tpe', d['tpe']), ('rs', d['rs'])):
            top = tr.nsmallest(5, 'value')
            r[f'top5_wd_med_{key}'] = float(10 ** np.median(np.log10(top.params_weight_decay)))
            r[f'wd_frac_low_{key}'] = float((tr.params_weight_decay < 1e-5).mean())
        r['imp_tpe'], r['imp_rs'] = d['tpe_imp'], d['rs_imp']
        r['rs_search_time_json'] = rb['search_time_s']
        r['rs_final_time'] = rb['final_train_time_s']
        r['tpe_final_time'] = tp['retrain_seconds']
        r['tpe_span_s'] = (d['tpe_all'].datetime_complete.max() - d['tpe_all'].datetime_start.min()).total_seconds()
        r['rs_span_s'] = (d['rs_all'].datetime_complete.max() - d['rs_all'].datetime_start.min()).total_seconds()
        r['rs_search_time'], r['rs_search_time_note'] = r['rs_search_time_json'], ''
        fails = d['rs_all'][d['rs_all'].state == 'FAIL']
        if len(fails):
            fnum = int(fails.number.min())
            pre = d['rs_all'][d['rs_all'].number < fnum]
            lost = (pre.datetime_complete.max() - pre.datetime_start.min()).total_seconds()
            if r['rs_search_time_json'] < 0.9 * r['rs_total_time']:
                r['rs_search_time'] = r['rs_search_time_json'] + lost
                r['rs_search_time_note'] = (f"korrigált: {r['rs_search_time_json']:.1f} s (JSON) + "
                                            f"{lost:.1f} s (a #0–#{fnum - 1} trial a study időbélyegeiből)")
            r['rs_span_note'] = 'megszakítást tartalmaz'
        R[arch] = r
    return R


def rank_and_tests(R: dict) -> dict:
    out = {}
    archs = ARCH_ORDER
    for key, a, b in (('search', 'rs_search_best', 'tpe_search_best'), ('p30', 'rs_val', 'tpe_val')):
        rs = np.array([R[x][a] for x in archs])
        tp = np.array([R[x][b] for x in archs])
        lr = np.log10(rs / tp)
        n_tpe = int((lr > 0).sum())
        sign_p = stats.binomtest(n_tpe, len(archs), 0.5, alternative='two-sided').pvalue
        w = stats.wilcoxon(lr, alternative='two-sided')
        out[key] = dict(n_tpe_better=n_tpe, n=len(archs), sign_p=float(sign_p),
                        wilcoxon_stat=float(w.statistic), wilcoxon_p=float(w.pvalue),
                        median_ratio=float(10 ** np.median(lr)), geo_mean_ratio=float(10 ** lr.mean()))
    rs_rank = pd.Series({x: R[x]['rs_val'] for x in archs}).rank().astype(int)
    tp_rank = pd.Series({x: R[x]['tpe_val'] for x in archs}).rank().astype(int)
    tau = stats.kendalltau(rs_rank.values, tp_rank.values)
    out['rank_rs'], out['rank_tpe'] = rs_rank.to_dict(), tp_rank.to_dict()
    out['kendall_tau'], out['kendall_p'] = float(tau.statistic), float(tau.pvalue)
    return out


def _sci(v):
    if not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if v == 0:
        return "0"
    mant, exp = f"{v:.2e}".split("e")
    return f"{mant}e{int(exp)}"


def _pct(v):
    return "–" if v is None else f"{v:+.1f}%".replace("-", "−")


def write_csv(R: dict):
    rows = []
    for arch in ARCH_ORDER:
        r = R[arch]
        row = dict(arch=arch, model_class=r['cls'], onlab_best_ref=r['onlab_ref'])
        for pre, val, tm, ep in (('onlab', r['onlab_val'], r['onlab_test'], r['onlab_epochs']),
                                 ('rs_p30', r['rs_val'], r['rs_test'], r['rs_epochs']),
                                 ('tpe_p30', r['tpe_val'], r['tpe_test'], r['tpe_epochs'])):
            row[f'{pre}_val_mse'] = val
            row[f'{pre}_test_rmse'] = tm['RMSE']
            row[f'{pre}_test_mae'] = tm['MAE']
            row[f'{pre}_test_r2'] = tm['R2']
            row[f'{pre}_test_max_error'] = tm['max_error']
            row[f'{pre}_best_epoch'] = ep[0]
            row[f'{pre}_n_epochs_run'] = ep[1]
        row['rel_diff_val_mse_rs_vs_tpe_pct'] = rel(r['rs_val'], r['tpe_val'])
        row['rel_diff_test_rmse_rs_vs_tpe_pct'] = rel(r['rs_test']['RMSE'], r['tpe_test']['RMSE'])
        row['improvement_val_mse_rs_vs_onlab_pct'] = -rel(r['rs_val'], r['onlab_val'])
        row['improvement_val_mse_tpe_vs_onlab_pct'] = -rel(r['tpe_val'], r['onlab_val'])
        row['rs_search_best_val_loss'] = r['rs_search_best']
        row['tpe_search_best_val_loss'] = r['tpe_search_best']
        rows.append(row)
    pd.DataFrame(rows).to_csv(guard_path(OUT_DIR / "random_vs_tpe_table.csv"), index=False)


def write_summary(R: dict, T: dict, comp: list, err_logs: list, checks: list):
    L = []
    A = L.append

    A("## Teljességi ellenőrzés\n")
    A("| Architektúra | Trial-keret | Érvényes trial, RS / TPE | FAIL, RS | Paraméter | Paraméterszám egyezik | "
      "Azonos 0–9. trial: paraméter / állapot / érték | Azonos paraméterű trial a 10.-től | Türelem 30 |")
    A("|---|---:|---:|---:|---:|:---:|---:|---:|:---:|")
    for c in comp:
        A(f"| {c['cls']} | {c['budget']} | {c['n_valid_rs']} / {c['n_valid_tpe']} | {c['n_fail_rs']} | "
          f"{c['param_count']:,} | {'igen' if c['param_match'] else 'nem'} | ".replace(",", " ")
          + f"{c['startup_params']} / {c['startup_state']} / {c['startup_value']} | {c['after_startup_equal']} | "
          f"{'igen' if c['final_patience_ok'] else 'nem'} |")

    A("\n## Végső tanítás (p30)\n")
    A("| Architektúra | Modell | Val MSE | Teszt-RMSE | Teszt-MAE | Teszt-R² | Teszt max. hiba | Legjobb / futott epoch |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for arch in ARCH_ORDER:
        r = R[arch]
        for name, val, tm, ep in ((f"kiindulási legjobb ({r['onlab_ref']})", r['onlab_val'], r['onlab_test'], r['onlab_epochs']),
                                  ("RS p30", r['rs_val'], r['rs_test'], r['rs_epochs']),
                                  ("TPE p30", r['tpe_val'], r['tpe_test'], r['tpe_epochs'])):
            A(f"| {r['cls']} | {name} | {_sci(val)} | {_sci(tm['RMSE'])} | {_sci(tm['MAE'])} | "
              f"{tm['R2']:.6f} | {_sci(tm['max_error'])} | {ep[0]} / {ep[1]} |")

    A("\n## Relatív különbségek (p30)\n")
    A("| Architektúra | Val MSE, RS a TPE-hez | Teszt-RMSE, RS a TPE-hez | Val MSE-arány RS / TPE | "
      "Teszt-RMSE-arány RS / TPE | Val MSE, RS a kiindulási legjobbhoz | Val MSE, TPE a kiindulási legjobbhoz | "
      "Jobb (val MSE) |")
    A("|---|---:|---:|---:|---:|---:|---:|:---:|")
    q_val, q_rmse = [], []
    for arch in ARCH_ORDER:
        r = R[arch]
        qv = r['rs_val'] / r['tpe_val']
        qt = r['rs_test']['RMSE'] / r['tpe_test']['RMSE']
        q_val.append(qv)
        q_rmse.append(qt)
        A(f"| {r['cls']} | {_pct(rel(r['rs_val'], r['tpe_val']))} | "
          f"{_pct(rel(r['rs_test']['RMSE'], r['tpe_test']['RMSE']))} | {qv:.2f} | {qt:.2f} | "
          f"{_pct(rel(r['rs_val'], r['onlab_val']))} | {_pct(rel(r['tpe_val'], r['onlab_val']))} | "
          f"{'TPE' if r['tpe_val'] < r['rs_val'] else 'RS'} |")
    A(f"| Medián | – | – | {np.median(q_val):.2f} | {np.median(q_rmse):.2f} | – | – | – |")

    A("\n## Rangsor (p30 val MSE, 1 = legjobb)\n")
    A("| Architektúra | RS p30 | TPE p30 |")
    A("|---|---:|---:|")
    for arch in ARCH_ORDER:
        A(f"| {CLASS_NAME[arch]} | {T['rank_rs'][arch]} | {T['rank_tpe'][arch]} |")

    A("\n## Kendall τ a két rangsor között\n")
    A("| Rangsor | τ | p |")
    A("|---|---:|---:|")
    A(f"| p30 val MSE, RS vs. TPE | {T['kendall_tau']:.3f} | {T['kendall_p']:.3f} |")

    A("\n## Páros összevetés (7 architektúra)\n")
    A("| Mutató | A TPE jobb | Előjelpróba, p | Wilcoxon, W | Wilcoxon, p | RS / TPE medián | RS / TPE mértani átlag |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for key, lab in (('search', 'Legjobb keresési val MSE'), ('p30', 'Végső p30 val MSE')):
        t = T[key]
        A(f"| {lab} | {t['n_tpe_better']}/{t['n']} | {t['sign_p']:.3f} | {t['wilcoxon_stat']:.0f} | "
          f"{t['wilcoxon_p']:.3f} | {t['median_ratio']:.2f} | {t['geo_mean_ratio']:.2f} |")

    A("\n## Keresés: legjobb keresési val MSE\n")
    A("| Architektúra | Közös indítás legjobbja | Legjobb, RS (trial) | Legjobb, TPE (trial) | RS / TPE | "
      "RS eléri a TPE legjobbját | TPE eléri az RS legjobbját | Teljes keresés, RS / TPE (s) |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for arch in ARCH_ORDER:
        r = R[arch]

        def reach(x):
            return "nem éri el" if x is None else (f"#{x['trial']} ({x['n_trials']}. érvényes trial, "
                                                    f"{x['cum_time_s']:.0f} s, {x['cum_epochs']} epoch)")
        A(f"| {r['cls']} | {_sci(r['startup_best'])} | {_sci(r['rs_search_best'])} (#{r['rs_best_trial']}) | "
          f"{_sci(r['tpe_search_best'])} (#{r['tpe_best_trial']}) | {r['rs_search_best'] / r['tpe_search_best']:.2f} | "
          f"{reach(r['rs_reach_tpe'])} | {reach(r['tpe_reach_rs'])} | "
          f"{r['rs_total_time']:.0f} / {r['tpe_total_time']:.0f} |")

    A("\n## Pruning\n")
    A("| Architektúra | Módszer | COMPLETE / PRUNED | COMPLETE-arány | Átl. epoch / trial | Összes epoch | "
      "10. trialtól: COMPLETE / összes | 10. trialtól: átl. epoch | 10. trialtól: 40 epochig futott | "
      "Jobb a közös indításnál | ≤ 2× a legjobb |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    tot = {k: dict(c=0, p=0, ep=0, n=0, pc=0, pn=0, pep=0, imp=0, w2=0) for k in ('tpe', 'rs')}
    for arch in ARCH_ORDER:
        r = R[arch]
        for key, lab in (('rs', 'RS'), ('tpe', 'TPE')):
            p = r[f'ps_{key}']
            A(f"| {r['cls']} | {lab} | {p['n_complete']} / {p['n_pruned']} | "
              f"{p['complete_ratio'] * 100:.0f}% | {p['mean_epochs']:.1f} | {p['total_epochs']} | "
              f"{p['post_complete']} / {p['post_n']} | {p['post_mean_epochs']:.1f} | {p['post_full40']} | "
              f"{p['post_better_than_startup']} | {p['post_within2x']} |")
            t = tot[key]
            t['c'] += p['n_complete']; t['p'] += p['n_pruned']; t['ep'] += p['total_epochs']
            t['n'] += p['n_complete'] + p['n_pruned']; t['pc'] += p['post_complete']; t['pn'] += p['post_n']
            t['pep'] += p['post_mean_epochs'] * p['post_n']; t['imp'] += p['post_better_than_startup']
            t['w2'] += p['post_within2x']
    for key, lab in (('rs', 'RS'), ('tpe', 'TPE')):
        t = tot[key]
        A(f"| Összesen | {lab} | {t['c']} / {t['p']} | {t['c'] / t['n'] * 100:.0f}% | {t['ep'] / t['n']:.1f} | "
          f"{t['ep']} | {t['pc']} / {t['pn']} | {t['pep'] / t['pn']:.1f} | – | {t['imp']} | {t['w2']} |")

    A("\n## Legjobb hiperparaméterek\n")
    A("| Architektúra | Módszer | lr | weight_decay | Dekád az alsó határtól | dropout | "
      "Top-5 weight_decay (medián) | weight_decay < 1e-5 |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for arch in ARCH_ORDER:
        r = R[arch]
        for key, lab in (('rs', 'RS'), ('tpe', 'TPE')):
            h = r[f'hp_{key}']
            do = f"{h['dropout']:.3f}" if h['dropout'] is not None else "–"
            A(f"| {r['cls']} | {lab} | {_sci(h['lr'])} | {_sci(h['weight_decay'])} | "
              f"{math.log10(h['weight_decay'] / WD_LOW):.2f} | {do} | {_sci(r[f'top5_wd_med_{key}'])} | "
              f"{r[f'wd_frac_low_{key}'] * 100:.0f}% |")

    A("\n## Hiperparaméter-fontosság (PED-ANOVA)\n")
    A("| Architektúra | RS: lr / wd / dropout | RS sorrend | TPE: lr / wd / dropout | TPE sorrend | Azonos sorrend |")
    A("|---|---|---|---|---|:---:|")
    nm = {'lr': 'lr', 'weight_decay': 'wd', 'dropout': 'dropout'}
    for arch in ARCH_ORDER:
        r = R[arch]

        def fmt(imp):
            return " / ".join(f"{imp[k]:.2f}" if k in imp else "–" for k in ('lr', 'weight_decay', 'dropout'))

        def order(imp):
            return sorted(imp, key=lambda k: -imp[k])
        o_rs, o_tp = order(r['imp_rs']), order(r['imp_tpe'])
        A(f"| {r['cls']} | {fmt(r['imp_rs'])} | {' > '.join(nm[k] for k in o_rs)} | {fmt(r['imp_tpe'])} | "
          f"{' > '.join(nm[k] for k in o_tp)} | {'igen' if o_rs == o_tp else 'nem'} |")

    A("\n## Futásidő (s)\n")
    A("| Architektúra | Keresés, RS | Trialidő, RS | Első–utolsó trial, RS | Trialidő, TPE | Első–utolsó trial, TPE | "
      "Trialidő-arány RS / TPE | Végső tanítás, RS | Végső tanítás, TPE |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    S = dict(rss=0, rst=0, tpt=0, tps=0, rsf=0, tpf=0)
    for arch in ARCH_ORDER:
        r = R[arch]
        A(f"| {r['cls']} | {r['rs_search_time']:.0f} | {r['rs_total_time']:.0f} | {r['rs_span_s']:.0f} | "
          f"{r['tpe_total_time']:.0f} | {r['tpe_span_s']:.0f} | {r['rs_total_time'] / r['tpe_total_time']:.2f} | "
          f"{r['rs_final_time']:.0f} | {r['tpe_final_time']:.0f} |")
        S['rss'] += r['rs_search_time']; S['rst'] += r['rs_total_time']; S['tpt'] += r['tpe_total_time']
        S['tps'] += r['tpe_span_s']; S['rsf'] += r['rs_final_time']; S['tpf'] += r['tpe_final_time']
    A(f"| Összesen | {S['rss']:.0f} | {S['rst']:.0f} | – | {S['tpt']:.0f} | {S['tps']:.0f} | "
      f"{S['rst'] / S['tpt']:.2f} | {S['rsf']:.0f} | {S['tpf']:.0f} |")

    A("\n## Gépsebesség: a közös 0–9. trial ideje\n")
    A("| Architektúra | Epoch, 0–9. trial | Idő, 0–9. trial, TPE / RS (s) | Időarány RS / TPE | s / epoch, TPE | "
      "s / epoch, RS | Keresési epoch, TPE / RS | Epoch-arány RS / TPE |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    et = er = 0
    for arch in ARCH_ORDER:
        r = R[arch]
        et += r['epochs_tpe']; er += r['epochs_rs']
        A(f"| {r['cls']} | {r['startup_epochs']} | {r['startup_time_tpe']:.0f} / {r['startup_time_rs']:.0f} | "
          f"{r['startup_time_rs'] / r['startup_time_tpe']:.2f} | {r['sec_per_epoch_tpe']:.2f} | "
          f"{r['sec_per_epoch_rs']:.2f} | {r['epochs_tpe']} / {r['epochs_rs']} | "
          f"{r['epochs_rs'] / r['epochs_tpe']:.2f} |")
    stt = sum(R[a]['startup_time_tpe'] for a in ARCH_ORDER)
    srs = sum(R[a]['startup_time_rs'] for a in ARCH_ORDER)
    A(f"| Összesen | {sum(R[a]['startup_epochs'] for a in ARCH_ORDER)} | {stt:.0f} / {srs:.0f} | {srs / stt:.2f} | "
      f"{S['tpt'] / et:.2f} | {S['rst'] / er:.2f} | {et} / {er} | {er / et:.2f} |")

    A("\n## Ellenőrzések\n")
    A("| Ellenőrzés | Eredmény |")
    A("|---|---|")
    A(f"| `*_ERROR.log` a `hpo/logs/random/` alatt | {', '.join(err_logs) if err_logs else 'nincs'} |")
    for label, result in checks:
        A(f"| {label} | {result} |")

    with open(guard_path(OUT_DIR / "random_vs_tpe_summary.md"), 'w', encoding='utf-8') as f:
        f.write("\n".join(L) + "\n")


def main():
    budget = trial_budget_from_source()
    D = {a: load_arch(a) for a in ARCH_ORDER}
    comp, err_logs = completeness(D, budget)
    problems = []
    if err_logs:
        problems.append(f"ERROR log: {err_logs}")
    for c in comp:
        if not (c['budget_ok'] and c['param_match'] and c['startup_ok'] and c['after_startup_equal'] == 0
                and c['final_patience_ok']):
            problems.append(f"{c['arch']}: {c}")
        if c['n_fail_rs'] and c['arch'] != 'gelu_resnet':
            problems.append(f"{c['arch']}: varatlan FAIL trial")
    for p in problems:
        print("[FIGYELEM]", p)
    if problems:
        raise SystemExit("Teljessegi ellenorzes: eltérés, az elemzes leall.")
    print("[ok] Teljessegi ellenorzes: minden rendben (ismert: gelu_resnet RS #16 FAIL).")

    R = compute(D)
    T = rank_and_tests(R)
    checks = []
    ok_any = all(R[a]['tpe_search_best'] == R[a]['tpe_min_value'] and R[a]['rs_search_best'] == R[a]['rs_min_value']
                 for a in ARCH_ORDER)
    checks.append(("A legjobb keresési érték = a `trials_{arch}.csv` minimum `value`-ja (7 architektúra, TPE és RS)",
                   'igen' if ok_any else 'nem'))
    reld = max(max(abs(R[a]['tpe_search_best'] / D[a]['tpe_best']['search_best_val_loss'] - 1),
                   abs(R[a]['rs_search_best'] / D[a]['rs_best']['search_best_val_loss'] - 1)) for a in ARCH_ORDER)
    ok_json = reld < 1e-12
    checks.append(("A legjobb keresési érték = a `best_params_{arch}.json` `search_best_val_loss` mezője (TPE és RS)",
                   f"{'igen' if ok_json else 'nem'} (legnagyobb relatív eltérés {reld:.1e})"))
    ok_on = all(R[a]['onlab_crosscheck'] for a in ARCH_ORDER)
    checks.append(("A kiindulási legjobb val MSE = a `training_history_*.json` `best_val_loss` mezője",
                   'igen' if ok_on else 'nem'))
    checks.append(("Az RS `best_params_{arch}.json` `tpe_p30_*` mezői = a `hpo/results/p30/` értékei", 'igen'))
    assert ok_any and ok_json and ok_on
    write_csv(R)
    write_summary(R, T, comp, err_logs, checks)
    print(f"[ok] Kimenetek: {relp(OUT_DIR)}")
    for a in ARCH_ORDER:
        r = R[a]
        print(f"{a:12s} search RS {r['rs_search_best']:.3e} TPE {r['tpe_search_best']:.3e} | "
              f"p30 RS {r['rs_val']:.3e} TPE {r['tpe_val']:.3e} ONLAB {r['onlab_val']:.3e}")
    print(json.dumps({k: v for k, v in T.items()}, indent=1, default=str))


if __name__ == '__main__':
    main()
