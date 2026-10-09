"""Experiment 10: the three regimes at d=128 (docs/spec_exp10.md; docs/preregistration.md, 'Three regimes at a second d (exp 10, backlog N2)').

Single-neuron code of exps 6-7 (sigma_2, k=2) at d=128: m_0 = d^-1/2 exactly, eta = 1/d^2, B=64, early stop |m|>=0.5.
Cells: a pinned Gamma=1 (N 16/64/256, seeds 0-1, cap 1e6); b pinned Gamma=0.1 (N 16/64/256, seeds 0-2, cap 2.5e6);
c tied rho0=0.01 (N 16/64/256, seeds 0-2, cap 1e6); d free Gamma_0=0.01, eta_Gamma=eta (N=256, seeds 0-1, cap 1.2e7).  23 runs.
Launch order: c, b N=64/256, a, b N=16, d last (free seeds are the ones dropped by the global CPU cap).
Resumable CSV, per-run CPU time, global CPU cap (gates launching only), 1 BLAS thread/worker.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

from .train import train

OUT = "results/exp10"
D = 128
ETA = 1.0 / D ** 2
B = 64
NS = [16, 64, 256]
COLS = ["cell", "protocol", "N", "B", "d", "eta", "seed", "m0", "max_steps", "steps", "T05", "reached05", "final_abs_m",
        "max_abs_m", "readout_at_T05", "final_readout", "cpu_s", "wall_s"]
# cell -> (label, kwargs for train, model, seeds, max_steps, Ns)
CELLS = {
    "a": ("pinned g=1", dict(gamma=1.0), "A", (0, 1), 1_000_000, NS),
    "b": ("pinned g=0.1", dict(gamma=0.1), "A", (0, 1, 2), 2_500_000, NS),
    "c": ("tied rho0=0.01", dict(rho0=0.01), "At", (0, 1, 2), 1_000_000, NS),
    "d": ("free G0=0.01 etaG=eta", dict(gamma=0.01, train_gamma=True, eta_gamma_mult=1.0), "A", (0, 1), 12_000_000, [256]),
}
# launch order (cell, N list)
ORDER = [("c", [16, 64, 256]), ("b", [64, 256]), ("a", [16, 64, 256]), ("b", [16]), ("d", [256])]


def key_of(r):
    return (r["cell"], int(r["N"]), int(r["seed"]))


def run_one(job):
    c, traj_dir = job
    label, kw0, model, _, _, _ = CELLS[c["cell"]]
    kw = dict(kw0, N=c["N"], B=B, max_steps=c["max_steps"], log_every=100, stop_m=0.5, init="fixed", m0_scale=1.0,
              rng_tag=1000, fast=True)
    c0, w0 = time.process_time(), time.time()
    r = train(model, D, 2, ETA, c["seed"], **kw)
    cpu = time.process_time() - c0
    traj = r.pop("traj")
    np.savez_compressed(os.path.join(traj_dir, f"{c['cell']}_N{c['N']}_s{c['seed']}.npz"), **traj)
    free_or_tied = model == "At" or kw0.get("train_gamma", False)
    return dict(c, protocol=label, B=B, d=D, eta=ETA, m0=r["m0"], steps=r["steps"], T05=r["T05"], reached05=bool(r["reached05"]),
                final_abs_m=abs(r["final_m"]), max_abs_m=r["max_abs_m"],
                readout_at_T05=(r["gamma_at_T05"] if free_or_tied else float("nan")),
                final_readout=(r["final_gamma"] if free_or_tied else float("nan")), cpu_s=cpu, wall_s=time.time() - w0)


def jobs(max_steps=None):
    J = []
    for cell, Ns in ORDER:
        _, _, _, seeds, cap, _ = CELLS[cell]
        for N in Ns:
            for s in seeds:
                J.append(dict(cell=cell, N=N, seed=s, max_steps=cap if max_steps is None else max_steps))
    return J


def spent(path):
    if not os.path.exists(path):
        return 0.0
    with open(path) as f:
        return sum(float(r["cpu_s"]) for r in csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu-cap-h", type=float, default=5.0)
    ap.add_argument("--max-steps", type=int, default=None, help="override every cell's cap (smoke test)")
    ap.add_argument("--only", default=None, help="comma list of cells (smoke test)")
    ap.add_argument("--only-N", default=None, help="comma list of N (smoke test)")
    a = ap.parse_args(argv)
    traj_dir = os.path.join(a.out, "traj")
    os.makedirs(traj_dir, exist_ok=True)
    path = os.path.join(a.out, "runs.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {key_of(r) for r in csv.DictReader(f)}
    J = [c for c in jobs(a.max_steps) if key_of(c) not in done and (a.only is None or c["cell"] in a.only.split(","))
         and (a.only_N is None or str(c["N"]) in a.only_N.split(","))]
    tot = spent(path)
    print(f"{len(done)} done, {len(J)} to run, so far {tot/3600:.3f} h, cap {a.cpu_cap_h} h, workers {a.workers}", flush=True)
    new = not os.path.exists(path)
    t0 = time.time()
    with open(path, "a", newline="") as f, ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:
        wr = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new:
            wr.writeheader()
        it, pending, n, ex_done = iter(J), set(), 0, False
        while True:
            while not ex_done and len(pending) < a.workers:
                if tot >= a.cpu_cap_h * 3600:
                    print("CPU cap reached: no further runs launched", flush=True)
                    ex_done = True
                    break
                nx = next(it, None)
                if nx is None:
                    ex_done = True
                    break
                print(f"launch {nx['cell']} N={nx['N']} s={nx['seed']} cap={nx['max_steps']}", flush=True)
                pending.add(ex.submit(run_one, (nx, traj_dir)))
            if not pending:
                break
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fu in fin:
                r = fu.result()
                wr.writerow(r)
                f.flush()
                n += 1
                tot += r["cpu_s"]
                print(f"[{n}/{len(J)} {time.time()-t0:6.0f}s wall {tot/3600:.3f} h cpu] {r['cell']} N={r['N']} s={r['seed']} "
                      f"T05={r['T05']} steps={r['steps']} |m|={r['final_abs_m']:.3f} ro@T={r['readout_at_T05']:.3g} cpu={r['cpu_s']:.0f}s", flush=True)
    print(f"finished {n} runs, total CPU {spent(path)/3600:.3f} h, wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
