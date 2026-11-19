import sys
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

import common as C
import eval_common as E
from common import guard_path, write_json, read_json, rel
import md_tables as M
from check_arbitrage import CHECK_LABEL, MAIN_CHECKS, REGION_NAMES

ARB = E.EVAL_DIR / "arbitrage"
EMAP = E.EVAL_DIR / "error_map"
SPEED = E.EVAL_DIR / "speed"
BEST_AM = 'mlp_american_retune_p30'
ARCH_LABEL = {'mlp': 'MLP', 'deep_mlp': 'DeepMLP', 'dense_mlp': 'DenseMLP', 'resnet': 'ResNet',
              'gelu_resnet': 'GELU-ResNet', 'highway': 'Highway', 'finn': 'FINN'}


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '–'
    if x == 0:
        return '0'
    a = abs(x)
    if a >= 1e-2 and a < 1e4:
        s = f"{x:.{nd}g}"
    else:
        s = f"{x:.{max(nd - 1, 1)}e}"
    return s.replace('.', ',')


def pct(x, nd=2):
    if x is None or not np.isfinite(x):
        return '–'
    return f"{100 * x:.{nd}f}%".replace('.', ',')


def arbitrage_tables():
    d = pd.read_csv(ARB / 'violations_by_region.csv')
    tot = d[d.region == 'összes'].copy()
    main = tot[tot.main_check]
    wide = main.pivot_table(index=['target', 'level', 'arch', 'run'], columns=['set', 'check', 'tol_name'],
                            values='rate')
    wide.columns = [f"{s}|{c}|{t}" for s, c, t in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(guard_path(ARB / 'summary_by_model.csv'), index=False)
    lvl = main.groupby(['target', 'level', 'set', 'check', 'tol_name']).agg(
        rate_median=('rate', 'median'), rate_mean=('rate', 'mean'), rate_min=('rate', 'min'),
        rate_max=('rate', 'max'), max_usd_median=('max_usd', 'median')).reset_index()
    lvl.to_csv(guard_path(ARB / 'level_comparison.csv'), index=False)
    best = d[(d.run == BEST_AM) & d.main_check & (d.region != 'összes')]
    best.to_csv(guard_path(ARB / 'regions_best_model.csv'), index=False)
    rows = []
    for run, g in main[(main.set == 'test') & (main.tol_name == '1e-6')].groupby('run'):
        top = g.loc[g.rate.idxmax()]
        reg = d[(d.run == run) & (d.set == 'test') & (d.tol_name == '1e-6') & (d.check == top.check)
                & (d.region != 'összes')]
        rmax = reg.loc[reg.n_viol.idxmax()]
        rows.append({'run': run, 'target': top.target, 'level': top.level, 'arch': top.arch,
                     'most_violated_check': top.check, 'rate': top.rate,
                     'region_with_most_violations': rmax.region,
                     'share_of_violations_in_region': rmax.n_viol / max(1, top.n_viol),
                     'region_rate': rmax.rate})
    mv = pd.DataFrame(rows)
    mv.to_csv(guard_path(ARB / 'most_violated.csv'), index=False)
    return d, tot, main, wide, lvl, best, mv


def speed_tables():
    t = pd.read_csv(SPEED / 'timings.csv')
    meta = read_json(SPEED / 'speed_meta.json')
    crr = t[(t.method == 'CRR-amerikai (500 lépés)') & ~t['mode'].str.contains('ismétlés')].iloc[0]
    crr_row = float(crr.median_s)
    t['speedup_vs_crr'] = crr_row / t['per_row_s']
    t.to_csv(guard_path(SPEED / 'timings_with_speedup.csv'), index=False)
    acc = meta['accuracy']
    nn = t[(t.method == 'NN') & t['mode'].str.startswith('inferencia')]
    rows = []
    for arch, usd in acc['nn_test_usd_rmse'].items():
        for dev, b, lab in (('GPU', 1_000_000, 'GPU, batch 1e6'), ('GPU', 1, 'GPU, batch 1'),
                            ('CPU', 1, 'CPU, batch 1')):
            s = nn[(nn.arch == arch) & (nn.device == dev) & (nn.batch == b) & (nn['mode'] != 'inferencia, 1 szál')]
            if len(s):
                rows.append({'method': f"NN {ARCH_LABEL[arch]}", 'setting': lab, 'per_row_s': float(s.per_row_s.iloc[0]),
                             'test_rmse_usd': usd, 'accuracy_source': 'levels_table.csv (retune p30, teszthalmaz)'})
    rows.append({'method': 'CRR-500', 'setting': 'CPU, soronként', 'per_row_s': crr_row,
                 'test_rmse_usd': acc['crr500_vs_crr1000_rmse_usd'],
                 'accuracy_source': f"CRR-500 vs. CRR-1000, {acc['crr500_vs_crr1000_n']} sor (a címke saját hibája)"})
    lsm = t[t.method == 'LSM (100 000 pálya)'].iloc[0]
    rows.append({'method': 'LSM', 'setting': 'CPU, soronként', 'per_row_s': float(lsm.median_s),
                 'test_rmse_usd': acc['lsm_qa_test_rmse_usd'],
                 'accuracy_source': f"LSM-QA vs. CRR-címke, {acc['lsm_qa_test_n']} tesztsor"})
    bs = t[(t.method == 'BS put (zárt alak)') & (t.batch == 1_000_000)].iloc[0]
    rows.append({'method': 'BS put (európai)', 'setting': 'CPU, batch 1e6', 'per_row_s': float(bs.per_row_s),
                 'test_rmse_usd': acc['bs_as_american_rmse_usd'],
                 'accuracy_source': 'BS put vs. amerikai címke, teljes teszthalmaz (≈ a prémium)'})
    par = pd.DataFrame(rows)
    par['speedup_vs_crr'] = crr_row / par.per_row_s
    par['pareto'] = [not (((par.per_row_s <= r.per_row_s) & (par.test_rmse_usd <= r.test_rmse_usd))
                          & ((par.per_row_s < r.per_row_s) | (par.test_rmse_usd < r.test_rmse_usd))).any()
                     for r in par.itertuples()]
    par.to_csv(guard_path(SPEED / 'pareto_table.csv'), index=False)
    return t, meta, par, crr_row


def num(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '–'
    if x == 0:
        return '0'
    if 1e-2 <= abs(x) < 1e4:
        return f"{x:#.3g}".rstrip('.').replace('-', '−')
    return M.sci(x)


def sci6(x):
    mant, exp = f"{x:.5e}".split('e')
    return f"{mant.replace('-', '−')}e{int(exp)}"


def _scheme(s):
    if not s:
        return '–'
    return s[s.find('(') + 1:s.rfind(')')] if '(' in s else s


def _battery(b):
    if not isinstance(b, dict):
        return b if b else '–'
    return f"töltöttség {b.get('EstimatedChargeRemaining')}%, BatteryStatus = {b.get('BatteryStatus')}"


def main():
    L = []
    PR = {a: C.CLASS_NAME[a] for a in C.ARCH_ORDER}
    SHORT2CLASS = {v: C.CLASS_NAME[k] for k, v in ARCH_LABEL.items()}

    hyp = read_json(EMAP / 'hypothesis.json')
    agg = read_json(EMAP / 'aggregation_check.json')
    L += ["## Hibatérkép: modellválasztás", ""]
    L += M.table(["Szerep", "Modell"], [
        ["Legjobb amerikai (legkisebb val MSE)", f"`{hyp['selection']['best_american']}`"],
        ["Legjobb európai put (legkisebb val MSE)", f"`{hyp['selection']['best_european']}`"],
        ["Kiegészítő pár", "retune p30 ResNetPricer, mindkét célon"],
        ["Cellák", hyp['definitions']['bins']]], "ll")
    L += ["", "## Hibatérkép: a cellákból visszaszámolt teszt-RMSE", ""]
    L += M.table(["Modell", "Cellákból", "levels_table.csv", "Relatív eltérés"],
                 [[f"`{r['run']}`", sci6(r['test_rmse_from_cells']), sci6(r['test_rmse_levels_table']),
                   num(r['rel_diff'])] for r in agg['rows']])
    L += ["", "## Hibatérkép: teszt-RMSE régiónként", ""]
    body = []
    for run, v in hyp['models'].items():
        wc = v['worst_cell']
        body.append([f"`{run}`", num(v['spearman_cell_rmse_vs_premium_norm']['rho']), num(v['rmse_itm_long']),
                     num(v['rmse_rest']), num(v['rmse_exercise_rows']), num(v['rmse_continuation_rows']),
                     f"[{num(wc['m_lo'])}; {num(wc['m_hi'])}) × [{num(wc['T_lo'])}; {num(wc['T_hi'])})",
                     num(wc['rmse_norm'])])
    L += M.table(["Modell", "Spearman ρ (cella-RMSE és prémium)", "ITM × hosszú T", "Máshol", "Lehívási sorok",
                  "Folytatási sorok", "Legrosszabb cella (m; T)", "A legrosszabb cella RMSE-je"], body, "lrrrrrlr")
    L += ["", "## Hibatérkép: amerikai és európai teszt-RMSE aránya", ""]
    L += M.table(["Pár", "Medián (400 cella)", "A 10% legnagyobb prémiumú cella", "Határ-közeli cellák",
                  "Többi cella", "ITM × hosszú T (sorok)", "Lehívási sorok", "Spearman ρ (arány és prémium)", "p"],
                 [[f"`{pair}`", num(v['ratio_median_all_cells']), num(v['ratio_median_top10pct_premium_cells']),
                   num(v['ratio_median_boundary_cells']), num(v['ratio_median_other_cells']),
                   num(v['pooled_ratio_itm_long']), num(v['pooled_ratio_exercise_rows']),
                   num(v['spearman_ratio_vs_premium']['rho']), num(v['spearman_ratio_vs_premium']['p'])]
                  for pair, v in hyp['pairs'].items()])

    d, tot, main_, wide, lvl, best, mv = arbitrage_tables()
    san = read_json(ARB / 'sanity.json')
    fdc = pd.read_csv(ARB / 'autograd_fd_check.csv')
    nc = read_json(ARB / 'norm_check.json')
    L += ["", "## Arbitrázsmentesség: technikai ellenőrzések", ""]
    L += M.table(["Ellenőrzés", "Eredmény"], [
        ["A háló float32 bemenete bitre egyezik a parquettel (mindhárom split)", M.yes(nc['float32_all_bitwise_equal'])],
        ["float64-ben a legnagyobb abszolút eltérés",
         num(max(v for s in nc['splits'].values() for v in s['float64_max_abs_diff'].values()))],
        ["Autograd és véges differencia: a delta relatív eltérésének legnagyobb mediánja (42 modell)",
         num(fdc.delta_ad64_rel_median.max())],
        ["Autograd és véges differencia: a théta relatív eltérésének legnagyobb mediánja (42 modell)",
         num(fdc.theta_ad64_rel_median.max())],
        ["1e-3 feletti eltérésű sorok legnagyobb aránya, delta", M.pct(fdc['delta_ad64_share_rel_gt_1e-3'].max())],
        ["1e-3 feletti eltérésű sorok legnagyobb aránya, théta", M.pct(fdc['theta_ad64_share_rel_gt_1e-3'].max())],
        ["BS put mint modell: 2., 3., 4. és 6. pont, τ = 1e-6, megsértés nélkül", M.yes(san['bs_model_ok'])],
        ["CRR-500 címke a rácsszeleten (r = 0.05, σ = 0.2625): 1., 2'., 3., 4., 5. pont megsértés nélkül",
         M.yes(san.get('crr_slice_ok_1_3_4_5_and_2prime'))],
        ["CRR-500 címke a rácsszeleten: a 2. pont megsértései", san['crr_slice_violations_tol1e-6']['c2_am_ge_bs']],
        ["Parquet-címkék a teszthalmazon (150 000 sor): a 2. pont megsértései, τ = 0",
         M.p(san['label_test_violations']['c2_am_ge_bs@0'])],
        ["Parquet-címkék a teszthalmazon (150 000 sor): a 2. pont megsértései, τ = 1e-6",
         M.p(san['label_test_violations']['c2_am_ge_bs@1e-6'])]])
    cks = MAIN_CHECKS['american']
    for set_ in ('test', 'grid'):
        L += ["", f"## Megsértési arány, amerikai put modellek, {'teszthalmaz' if set_ == 'test' else 'rács'} "
                  f"(τ = 1e-6 / τ = teszt-RMSE)", ""]
        body = []
        for l in E.LEVELS:
            for a in C.ARCH_ORDER:
                run = E.run_id(a, 'american', l)
                cells = []
                for c in cks:
                    r1 = tot[(tot.run == run) & (tot.set == set_) & (tot.check == c) & (tot.tol_name == '1e-6')].rate.iloc[0]
                    r2 = tot[(tot.run == run) & (tot.set == set_) & (tot.check == c) & (tot.tol_name == 'RMSE')].rate.iloc[0]
                    cells.append(f"{M.pct(r1)} / {M.pct(r2)}")
                body.append([f"{PR[a]}, {E.LEVEL_LABEL[l]}"] + cells)
        L += M.table(["Modell"] + [CHECK_LABEL[c] for c in cks], body)
    L += ["", "## Megsértési arány szintenként, amerikai put (medián a 7 architektúrán, teszthalmaz)", ""]
    body = []
    for tol in ('0', '1e-6', 'RMSE'):
        for l in E.LEVELS:
            s = lvl[(lvl.target == 'american') & (lvl.set == 'test') & (lvl.tol_name == tol) & (lvl.level == l)] \
                .set_index('check')
            body.append([E.LEVEL_LABEL[l], tol] + [M.pct(s.rate_median.get(c, np.nan)) for c in cks])
    L += M.table(["Szint", "τ"] + [CHECK_LABEL[c] for c in cks], body, "ll" + "r" * len(cks))
    L += ["", "## Megsértési arány szintenként, európai put (medián a 7 architektúrán, teszthalmaz, "
              "τ = 1e-6 / τ = teszt-RMSE)", ""]
    cke = MAIN_CHECKS['european']
    body = []
    for l in E.LEVELS:
        s1 = lvl[(lvl.target == 'european') & (lvl.set == 'test') & (lvl.tol_name == '1e-6') & (lvl.level == l)].set_index('check')
        s2 = lvl[(lvl.target == 'european') & (lvl.set == 'test') & (lvl.tol_name == 'RMSE') & (lvl.level == l)].set_index('check')
        body.append([E.LEVEL_LABEL[l]] + [f"{M.pct(s1.rate_median.get(c))} / {M.pct(s2.rate_median.get(c))}" for c in cke])
    L += M.table(["Szint"] + [CHECK_LABEL[c] for c in cke], body)
    L += ["", "## Megsértési arány régiónként, MLPPricer retune p30 (teszthalmaz, τ = 1e-6)", ""]
    b = best[(best.set == 'test') & (best.tol_name == '1e-6')]
    L += M.table(["Régió"] + [CHECK_LABEL[c] for c in cks],
                 [[reg] + [M.pct(b[(b.region == reg) & (b.check == c)].rate.iloc[0]) for c in cks] for reg in REGION_NAMES])
    L += ["", "## A leggyakoribb megsértés modellenként (teszthalmaz, τ = 1e-6)", ""]
    L += M.table(["Cél", "Leggyakoribb megsértés", "Modellek", "A legtöbb megsértés régiója (módusz)"],
                 [[{'american': 'amerikai put', 'european': 'európai put'}.get(tg, tg), CHECK_LABEL[ck], len(g), g.region_with_most_violations.mode().iloc[0]]
                  for (tg, ck), g in mv.groupby(['target', 'most_violated_check'])], "llrl")

    t, meta, par, crr_row = speed_tables()
    cmp_ = meta['comparison_with_earlier']
    env = meta['env']
    st0 = meta['machine_state_before']
    st1 = meta['machine_state_after']
    L += ["", "## Sebesség: mérési környezet", ""]
    L += M.table(["Jellemző", "Érték"], [
        ["CPU", f"{st0['cpu']['Name'].strip()} ({st0['cpu']['NumberOfCores']} mag, {st0['cpu']['NumberOfLogicalProcessors']} szál)"
         if st0.get('cpu') else '–'],
        ["GPU", env.get('gpu_name')],
        ["Szoftver", f"torch {env['torch']}, CUDA {env['cuda_version']}, Python {env['python']}, NumPy {env['numpy']}"],
        ["torch-szálszám", meta['torch_num_threads_used']],
        ["CPU-terhelés előtte / utána (%)", f"{st0.get('cpu_load_pct_3x1s')} / {st1.get('cpu_load_pct_3x1s')}"],
        ["Szabad RAM előtte / utána (MB)", f"{M.p(st0.get('ram_free_mb'))} / {M.p(st1.get('ram_free_mb'))}"],
        ["Energiaséma", _scheme(st0.get('power_scheme'))],
        ["Akkumulátor", _battery(st0.get('battery'))]], "ll")
    L += ["", "## Sebesség és pontosság (a CRR soronként hívott, egyszálú Python-implementáció)", ""]
    L += M.table(["Módszer", "Beállítás", "Idő áranként (s, medián)", "Gyorsulás a CRR-hez", "RMSE ($)",
                  "A pontosság forrása"],
                 [[f"NN, {SHORT2CLASS[r.method[3:]]}" if r.method.startswith('NN ') else r.method, r.setting, num(r.per_row_s),
                   num(r.speedup_vs_crr), num(r.test_rmse_usd), r.accuracy_source]
                  for r in par.sort_values('per_row_s').itertuples()], "llrrrl")
    L += ["", "## Összevetés a tanítóadat generálásakor mért idővel", ""]
    L += M.table(["Módszer", "Most", "A generáláskor", "Arány"], [
        ["CRR-amerikai, ms/ár", f"{num(cmp_['crr_ms_per_row_now'])} (a futás végén {num(cmp_['crr_ms_per_row_end_of_run'])})",
         num(cmp_['crr_ms_per_row_generation_meta']), num(cmp_['crr_ratio_now_vs_generation'])],
        ["CRR-európai batch, ms/ár (most 1000, a generáláskor 1M sor)", num(cmp_['crr_eu_ms_per_row_now_1000']),
         num(cmp_['crr_eu_ms_per_row_generation_meta']),
         num(cmp_['crr_eu_ms_per_row_now_1000'] / cmp_['crr_eu_ms_per_row_generation_meta'])],
        ["LSM, s/ár (20 sor, zárójelben a QA lépésszámára skálázva)",
         f"{num(cmp_['lsm_s_per_row_now_mean'])} ({num(cmp_['lsm_s_per_row_now_scaled_to_qa_steps'])})",
         num(cmp_['lsm_s_per_row_generation_meta']), num(cmp_['lsm_ratio_now_scaled_vs_generation'])]], "lrrr")
    nn = t[(t.method == 'NN') & t['mode'].isin(['inferencia (bemenet a GPU-n)', 'inferencia',
                                               'inferencia, 1e5-ös darabokban', 'inferencia, 1 szál'])]
    L += ["", "## NN-inferencia: idő áranként (s, medián)", ""]
    body = []
    for a in C.ARCH_ORDER:
        g = nn[nn.arch == a]
        if not len(g):
            continue

        def pr(dev, b_, one=False):
            s = g[(g.device == dev) & (g.batch == b_) & ((g['mode'] == 'inferencia, 1 szál') == one)]
            return num(float(s.per_row_s.iloc[0])) if len(s) else '–'
        body.append([PR[a]] + [pr('GPU', x) for x in E_BATCHES] + [pr('CPU', x) for x in E_BATCHES] + [pr('CPU', 1, True)])
    L += M.table(["Architektúra", "GPU, batch 1", "GPU, batch 1e3", "GPU, batch 1e5", "GPU, batch 1e6", "CPU, batch 1",
                  "CPU, batch 1e3", "CPU, batch 1e5", "CPU, batch 1e6", "CPU, 1 szál, batch 1"], body)
    txt = "\n".join(L) + "\n"
    with open(guard_path(E.EVAL_DIR / 'evaluation_summary.md'), 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print(f"[summary] {rel(E.EVAL_DIR / 'evaluation_summary.md')} irva.")


E_BATCHES = [1, 1_000, 100_000, 1_000_000]

if __name__ == '__main__':
    main()
