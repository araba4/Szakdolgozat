import sys
sys.dont_write_bytecode = True

import argparse
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch

import common as C
import eval_common as E
from common import guard_path, write_json, read_json, rel

OUT_DIR = E.EVAL_DIR / "speed"
GEN_META = C.AM_DATA_DIR / "generation_meta.json"

BATCHES = [1, 1_000, 100_000, 1_000_000]
REPS_GPU = {1: 300, 1_000: 200, 100_000: 30, 1_000_000: 10}
REPS_CPU = {1: 300, 1_000: 100, 100_000: 10, 1_000_000: 3}
REPS_BS = {1: 300, 1_000: 200, 100_000: 20, 1_000_000: 5}
WARMUP = 20
CPU_CHUNK = 100_000
CPU_MAX_ACT_BYTES = 2e9
N_CRR = 1000
N_CRR_EU_BIG = 10_000
N_LSM = 20
N_CRR_ACC = 200
CRR_REPS = 3
SEED = 42
LSM_SETTINGS = dict(n_paths=100_000, degree=3, antithetic=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ps(cmd):
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True,
                             text=True, timeout=60)
        return out.stdout.strip()
    except Exception as exc:
        return f"hiba: {exc}"


def machine_state(tag):
    st = {'tag': tag, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    try:
        st['cpu'] = json.loads(_ps("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,"
                                   "NumberOfLogicalProcessors,MaxClockSpeed,CurrentClockSpeed,LoadPercentage "
                                   "| ConvertTo-Json"))
    except Exception:
        st['cpu'] = None
    try:
        osd = json.loads(_ps("Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,"
                             "FreePhysicalMemory | ConvertTo-Json"))
        st['ram_total_mb'] = osd['TotalVisibleMemorySize'] / 1024
        st['ram_free_mb'] = osd['FreePhysicalMemory'] / 1024
    except Exception:
        st['ram_free_mb'] = None
    raw = _ps("1..3 | ForEach-Object { (Get-CimInstance Win32_Processor).LoadPercentage; Start-Sleep -Seconds 1 }")
    try:
        st['cpu_load_pct_3x1s'] = [float(x.replace(',', '.')) for x in raw.split()]
    except Exception:
        st['cpu_load_pct_3x1s'] = raw
    try:
        st['battery'] = json.loads(_ps("Get-CimInstance Win32_Battery | Select-Object BatteryStatus,"
                                       "EstimatedChargeRemaining | ConvertTo-Json"))
        st['battery_note'] = 'BatteryStatus 2 = hálózati tápról'
    except Exception:
        st['battery'] = None
    st['power_scheme'] = _ps("powercfg /getactivescheme")
    try:
        st['top_processes_by_memory'] = json.loads(_ps(
            "Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10 "
            "ProcessName,Id,@{n='WS_MB';e={[math]::Round($_.WorkingSet64/1MB)}},CPU | ConvertTo-Json"))
    except Exception:
        st['top_processes_by_memory'] = None
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,'
                            'utilization.gpu,pstate,clocks.sm,clocks.max.sm,power.draw,temperature.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
        st['nvidia_smi'] = q.stdout.strip()
    except Exception as exc:
        st['nvidia_smi'] = f"hiba: {exc}"
    return st


def stats(ts):
    a = np.asarray(ts, dtype=np.float64)
    q25, q50, q75 = np.quantile(a, [0.25, 0.5, 0.75])
    return dict(n=len(a), median_s=float(q50), q25_s=float(q25), q75_s=float(q75),
                iqr_s=float(q75 - q25), min_s=float(a.min()), mean_s=float(a.mean()))


def timeit(fn, reps, warmup=WARMUP, sync=lambda: None):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return stats(ts)


def make_inputs(df, n, rng):
    idx = rng.integers(0, len(df), size=n)
    return (df['moneyness'].to_numpy()[idx], df['T'].to_numpy()[idx], df['r'].to_numpy()[idx],
            df['sigma'].to_numpy()[idx])


def bench_nn(reg_runs, df, rows):
    cuda = torch.cuda.is_available()
    sync_gpu = torch.cuda.synchronize if cuda else (lambda: None)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)
    rng = np.random.default_rng(SEED)
    X64 = E.normalize(*make_inputs(df, max(BATCHES), rng))
    X_cpu_full = torch.tensor(X64, dtype=torch.float32)
    raw_full = make_inputs(df, max(BATCHES), np.random.default_rng(SEED))
    peak_bytes_per_row = {}
    for r in reg_runs:
        arch = r['arch']
        log(f"NN: {r['run']}")
        if cuda:
            model, _ = E.load(r['checkpoint'], device='cuda')
            for b in BATCHES:
                x = X_cpu_full[:b].to('cuda')
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()

                def f():
                    with torch.inference_mode():
                        model(x)
                try:
                    s = timeit(f, REPS_GPU[b], sync=sync_gpu)
                    peak = torch.cuda.max_memory_allocated() - base
                    if b == 100_000:
                        peak_bytes_per_row[arch] = peak / b
                    rows.append(dict(method='NN', arch=arch, run=r['run'], device='GPU', threads=None, batch=b,
                                     mode='inferencia (bemenet a GPU-n)', gpu_peak_act_mb=peak / 2 ** 20, **s))
                except torch.cuda.OutOfMemoryError:
                    rows.append(dict(method='NN', arch=arch, run=r['run'], device='GPU', batch=b,
                                     mode='inferencia (bemenet a GPU-n)', note='OOM'))
                    torch.cuda.empty_cache()
                del x
            for b in BATCHES:
                xc = X_cpu_full[:b]
                s = timeit(lambda: xc.to('cuda'), REPS_GPU[b], sync=sync_gpu)
                rows.append(dict(method='másolás', arch='-', run='-', device='GPU', batch=b,
                                 mode='host→device (float32, 4 oszlop)', **s))
                if arch == 'mlp':
                    xg = xc.to('cuda')
                    with torch.inference_mode():
                        yg = model(xg)
                    s = timeit(lambda: yg.cpu(), REPS_GPU[b], sync=sync_gpu)
                    rows.append(dict(method='másolás', arch='-', run='-', device='GPU', batch=b,
                                     mode='device→host (float32, 1 oszlop)', **s))
                    del xg, yg
                m_, T_, r_, s_ = (a[:b] for a in raw_full)

                def e2e():
                    xx = torch.tensor(E.normalize(m_, T_, r_, s_), dtype=torch.float32).to('cuda')
                    with torch.inference_mode():
                        model(xx).cpu().numpy()
                s = timeit(e2e, max(3, REPS_GPU[b] // 2), warmup=5, sync=sync_gpu)
                rows.append(dict(method='NN', arch=arch, run=r['run'], device='GPU', batch=b,
                                 mode='teljes út (nyers float64 → normalizálás → GPU → CPU)', **s))
            del model
            torch.cuda.empty_cache()
        model, _ = E.load(r['checkpoint'], device='cpu')
        for b in BATCHES:
            x = X_cpu_full[:b]
            est = peak_bytes_per_row.get(arch, 0.0) * b
            chunked = (b == 1_000_000 and est > CPU_MAX_ACT_BYTES)

            def f():
                with torch.inference_mode():
                    if chunked:
                        for i in range(0, b, CPU_CHUNK):
                            model(x[i:i + CPU_CHUNK])
                    else:
                        model(x)
            s = timeit(f, REPS_CPU[b], warmup=3 if b >= 100_000 else WARMUP)
            rows.append(dict(method='NN', arch=arch, run=r['run'], device='CPU', threads=n_threads, batch=b,
                             mode=('inferencia, 1e5-ös darabokban' if chunked else 'inferencia'),
                             est_act_mb=est / 2 ** 20, **s))
        torch.set_num_threads(1)
        for b in (1, 1_000):
            x = X_cpu_full[:b]

            def f1():
                with torch.inference_mode():
                    model(x)
            s = timeit(f1, REPS_CPU[b])
            rows.append(dict(method='NN', arch=arch, run=r['run'], device='CPU', threads=1, batch=b,
                             mode='inferencia, 1 szál', **s))
        torch.set_num_threads(n_threads)
        del model
    return n_threads


def bench_bs(df, rows):
    rng = np.random.default_rng(SEED)
    m, T, r, s = make_inputs(df, max(BATCHES), rng)
    for b in BATCHES:
        mm, TT, rr, ss = m[:b], T[:b], r[:b], s[:b]
        st = timeit(lambda: E.bs_put_norm(mm, TT, rr, ss), REPS_BS[b], warmup=3)
        rows.append(dict(method='BS put (zárt alak)', arch='-', run='-', device='CPU', threads=1, batch=b,
                         mode='vektorizált NumPy', **st))


def bench_trees(df, rows, acc):
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(df), size=N_CRR, replace=False))
    sub = df.iloc[idx]
    S, K, T, r, s = (sub[c].to_numpy() for c in ('S', 'K', 'T', 'r', 'sigma'))
    E.AP.crr_american_price(S[0], K[0], T[0], r[0], s[0], option_type='put', n_steps=500)
    ts = []
    for _ in range(CRR_REPS):
        t0 = time.perf_counter()
        p = E.AP.crr_american_price_batch(S, K, T, r, s, option_type='put', n_steps=500)
        ts.append((time.perf_counter() - t0) / N_CRR)
    rows.append(dict(method='CRR-amerikai (500 lépés)', arch='-', run='-', device='CPU', threads=1, batch=N_CRR,
                     mode='soronkénti Python-hívás', per_row=True, **stats(ts)))
    acc['crr_vs_label_max_abs_usd'] = float(np.abs(p - sub['american_put_price'].to_numpy()).max())
    j = idx[:N_CRR_ACC]
    sj = df.iloc[j]
    p500 = sj['american_put_price'].to_numpy()
    t0 = time.perf_counter()
    p1000 = E.AP.crr_american_price_batch(sj['S'].to_numpy(), sj['K'].to_numpy(), sj['T'].to_numpy(),
                                          sj['r'].to_numpy(), sj['sigma'].to_numpy(), option_type='put',
                                          n_steps=1000)
    acc['crr1000_seconds_per_row'] = (time.perf_counter() - t0) / N_CRR_ACC
    acc['crr500_vs_crr1000_rmse_usd'] = float(np.sqrt(np.mean((p500 - p1000) ** 2)))
    acc['crr500_vs_crr1000_max_abs_usd'] = float(np.abs(p500 - p1000).max())
    acc['crr500_vs_crr1000_n'] = N_CRR_ACC
    E.AP.crr_european_price_batch(S[:10], K[:10], T[:10], r[:10], s[:10], option_type='put', n_steps=500)
    ts = []
    for _ in range(CRR_REPS):
        t0 = time.perf_counter()
        E.AP.crr_european_price_batch(S, K, T, r, s, option_type='put', n_steps=500)
        ts.append((time.perf_counter() - t0) / N_CRR)
    rows.append(dict(method='CRR-európai batch (500 lépés)', arch='-', run='-', device='CPU', threads=1,
                     batch=N_CRR, mode='sorok felett vektorizált', per_row=True, **stats(ts)))
    idx2 = np.random.default_rng(SEED + 1).choice(len(df), size=N_CRR_EU_BIG, replace=False)
    sb = df.iloc[idx2]
    ts = []
    for _ in range(CRR_REPS):
        t0 = time.perf_counter()
        E.AP.crr_european_price_batch(sb['S'].to_numpy(), sb['K'].to_numpy(), sb['T'].to_numpy(),
                                      sb['r'].to_numpy(), sb['sigma'].to_numpy(), option_type='put', n_steps=500)
        ts.append((time.perf_counter() - t0) / N_CRR_EU_BIG)
    rows.append(dict(method='CRR-európai batch (500 lépés)', arch='-', run='-', device='CPU', threads=1,
                     batch=N_CRR_EU_BIG, mode='sorok felett vektorizált', per_row=True, **stats(ts)))
    return dict(crr_rows=idx.tolist()[:20], crr_mean_T=float(T.mean()))


def bench_crr_repeat(df):
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(df), size=N_CRR, replace=False))
    sub = df.iloc[idx]
    S, K, T, r, s = (sub[c].to_numpy() for c in ('S', 'K', 'T', 'r', 'sigma'))
    ts = []
    for _ in range(CRR_REPS):
        t0 = time.perf_counter()
        E.AP.crr_american_price_batch(S, K, T, r, s, option_type='put', n_steps=500)
        ts.append((time.perf_counter() - t0) / N_CRR)
    return dict(method='CRR-amerikai (500 lépés)', arch='-', run='-', device='CPU', threads=1, batch=N_CRR,
                mode='soronkénti Python-hívás (ismétlés a futás végén)', per_row=True, **stats(ts))


def bench_lsm(df, rows, acc):
    qa = df[df['is_qa_lsm']] if 'is_qa_lsm' in df else None
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(len(qa), size=N_LSM, replace=False))
    sub = qa.iloc[pick]
    ts, steps, prices = [], [], []
    for _, row in sub.iterrows():
        n_steps = max(1, round(50.0 * row['T']))
        t0 = time.perf_counter()
        pr, _ = E.AP.lsm_american_price(row['S'], row['K'], row['T'], row['r'], row['sigma'], option_type='put',
                                        n_steps=n_steps, seed=SEED, **LSM_SETTINGS)
        ts.append(time.perf_counter() - t0)
        steps.append(n_steps)
        prices.append(pr)
    ts, steps = np.array(ts), np.array(steps)
    rows.append(dict(method='LSM (100 000 pálya)', arch='-', run='-', device='CPU', threads=1, batch=N_LSM,
                     mode='soronkénti hívás, n_steps = round(50·T)', per_row=True, **stats(ts)))
    acc['lsm_timing'] = {'n': N_LSM, 'mean_s_per_row': float(ts.mean()), 'median_s_per_row': float(np.median(ts)),
                         'mean_n_steps': float(steps.mean()),
                         'us_per_1000_path_steps': float(1e6 * ts.sum() / (steps.sum() * LSM_SETTINGS['n_paths'] / 1000))}
    d = qa['lsm_put_price'].to_numpy() - qa['american_put_price'].to_numpy()
    acc['lsm_qa_test_rmse_usd'] = float(np.sqrt(np.mean(d ** 2)))
    acc['lsm_qa_test_n'] = int(len(qa))
    acc['lsm_rerun_vs_parquet_max_abs_usd'] = float(np.abs(np.array(prices) - sub['lsm_put_price'].to_numpy()).max())
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='rovid probafutas (kevesebb ismetles, 2 modell)')
    args = ap.parse_args()
    if args.quick:
        for d in (REPS_GPU, REPS_CPU, REPS_BS):
            for k in d:
                d[k] = max(2, d[k] // 50)
        global N_CRR, N_LSM, N_CRR_ACC, CRR_REPS, N_CRR_EU_BIG
        N_CRR, N_LSM, N_CRR_ACC, CRR_REPS, N_CRR_EU_BIG = 50, 2, 10, 1, 500
    out_dir = OUT_DIR / ('quick' if args.quick else '')
    guard_path(out_dir).mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    state_before = machine_state('előtte')
    log(f"Gep allapota (elotte): CPU-terheles {state_before.get('cpu_load_pct_3x1s')}, "
        f"szabad RAM {state_before.get('ram_free_mb')} MB, GPU {state_before.get('nvidia_smi')}")
    env = E.env_extra()
    env['processor'] = platform.processor()
    df = pd.read_parquet(C.am_split_path('test'),
                         columns=['S', 'K', 'moneyness', 'T', 'r', 'sigma', 'american_put_price',
                                  'european_put_price', 'lsm_put_price', 'is_qa_lsm'])
    lt = E.levels_table()
    reg = [r for r in E.model_registry() if r['target'] == 'american' and r['level'] == 'retune_p30']
    reg = sorted(reg, key=lambda r: C.ARCH_ORDER.index(r['arch']))
    if args.quick:
        reg = reg[:2]
    rows, acc = [], {}
    log("CRR / CRR-europai")
    bench_trees(df, rows, acc)
    log("LSM")
    bench_lsm(df, rows, acc)
    log("BS")
    bench_bs(df, rows)
    n_threads = bench_nn(reg, df, rows)
    rows.append(bench_crr_repeat(df))
    state_after = machine_state('utána')

    res = pd.DataFrame(rows)
    per_row = res['per_row'].astype('boolean').fillna(False).to_numpy(dtype=bool)
    res['per_row_s'] = np.where(per_row, res['median_s'], res['median_s'] / res['batch'])
    res.to_csv(guard_path(out_dir / 'timings.csv'), index=False)

    gm = read_json(GEN_META)
    crr_now = float(res[(res.method == 'CRR-amerikai (500 lépés)') & ~res['mode'].str.contains('ismétlés')]
                    ['median_s'].iloc[0])
    crr_end = float(res[(res.method == 'CRR-amerikai (500 lépés)') & res['mode'].str.contains('ismétlés')]
                    ['median_s'].iloc[0])
    lsm = acc['lsm_timing']
    gen_qa_idx = gm['qa_idx']
    comparison = {
        'crr_ms_per_row_now': 1e3 * crr_now, 'crr_ms_per_row_end_of_run': 1e3 * crr_end,
        'crr_ms_per_row_generation_meta': 1e3 * gm['crr_s_per_row'],
        'crr_ratio_now_vs_generation': crr_now / gm['crr_s_per_row'],
        'crr_eu_ms_per_row_now_1000': 1e3 * float(res[(res.method == 'CRR-európai batch (500 lépés)') &
                                                      (res.batch == N_CRR)]['median_s'].iloc[0]),
        'crr_eu_ms_per_row_generation_meta': 1e3 * gm['crr_eu_elapsed_s'] / gm['n'],
        'lsm_s_per_row_now_mean': lsm['mean_s_per_row'], 'lsm_s_per_row_generation_meta': gm['lsm_s_per_row_qa'],
        'lsm_mean_n_steps_now': lsm['mean_n_steps'],
        'lsm_note': 'A generálási átlag 2000 QA-soron készült (n_steps = round(50·T)); a 20 soros minta '
                    'átlagos lépésszáma eltérhet, ezért a pálya×lépésre vetített idő is szerepel.',
        'n_gen_qa': len(gen_qa_idx),
    }
    qa_T = df.loc[df['is_qa_lsm'], 'T'].to_numpy()
    mean_steps_qa = float(np.mean(np.maximum(1, np.round(50 * qa_T))))
    comparison['lsm_mean_n_steps_test_qa'] = mean_steps_qa
    comparison['lsm_s_per_row_now_scaled_to_qa_steps'] = lsm['us_per_1000_path_steps'] * 1e-6 * \
        mean_steps_qa * LSM_SETTINGS['n_paths'] / 1000
    comparison['lsm_ratio_now_scaled_vs_generation'] = comparison['lsm_s_per_row_now_scaled_to_qa_steps'] / \
        gm['lsm_s_per_row_qa']

    acc['bs_as_american_rmse_usd'] = float(np.sqrt(np.mean((df['european_put_price'] - df['american_put_price']) ** 2)))
    acc['nn_test_usd_rmse'] = {r['arch']: r['test_usd_rmse'] for r in reg}
    acc['source_nn'] = rel(E.LEVELS_TABLE)

    write_json(out_dir / 'speed_meta.json', {
        'script': 'american-nn/benchmark_speed.py', 'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seconds': time.time() - t_start, 'env': env, 'torch_num_threads_used': n_threads,
        'machine_state_before': state_before, 'machine_state_after': state_after,
        'settings': {'batches': BATCHES, 'reps_gpu': REPS_GPU, 'reps_cpu': REPS_CPU, 'reps_bs': REPS_BS,
                     'warmup': WARMUP, 'cpu_chunk': CPU_CHUNK, 'cpu_max_act_bytes': CPU_MAX_ACT_BYTES,
                     'n_crr': N_CRR, 'n_lsm': N_LSM, 'lsm': LSM_SETTINGS, 'seed': SEED,
                     'inference': 'model.eval() + torch.inference_mode(), float32'},
        'comparison_with_earlier': comparison, 'accuracy': acc,
        'models': [rel(r['checkpoint']) for r in reg],
        'fairness_note': 'A CRR nem optimalizált: Python-ciklus az időlépéseken, soronkénti hívás, egy szál. '
                         'A gyorsulás „ehhez az implementációhoz képest” érvényes.',
    })
    log(f"Kesz: {time.time() - t_start:.1f} s")


if __name__ == '__main__':
    main()
