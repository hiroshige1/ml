"""Experiment 7: controls requested by review 2 (docs/preregistration.md, 'Controls requested by review 2 (exp 7)').

Same single-neuron Model A / tied A-tilde code, sigma_2, d=64, m_0 = d^-1/2 exactly, as sweep6.  Cells a-e: N in {16,64,256}, B=64, eta=1/d^2;
cell f: pinned gamma in {0.01 (no), 0.1}, N=64, B in {16,256}, eta=(B/64)/d^2.  3 seeds, early stop |m|>=0.5, cap 2.5e6 steps
(the (d) cap can be lowered with --cap-d).  Resumable CSV, per-run CPU time, global CPU cap, 1 BLAS thread/worker.
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

OUT = "results/exp7"
D = 64
COLS = ["cell", "protocol", "N", "B", "d", "eta", "seed", "m0", "max_steps", "steps", "T05", "reached05", "final_abs_m",
        "max_abs_m", "readout_at_T05", "final_readout", "cpu_s", "wall_s"]
NS = [16, 64, 256]
MAX_STEPS = 2_500_000
# cell -> (label, kwargs for train, model)
CELLS = {
    "a": ("free G0=0.01 etaG=10eta", dict(gamma=0.01, train_gamma=True, eta_gamma_mult=10.0), "A"),
    "b": ("free G0=0.1 etaG=eta", dict(gamma=0.1, train_gamma=True, eta_gamma_mult=1.0), "A"),
    "c": ("tied rho0=0.1", dict(rho0=0.1), "At"),
    "d": ("tied rho0=0.3", dict(rho0=0.3), "At"),
    "e": ("pinned g=0.01", dict(gamma=0.01), "A"),
    "f": ("pinned g=0.1 eta~B", dict(gamma=0.1), "A"),
}


def key_of(r):
    return (r["cell"], int(r["N"]), int(r["B"]), int(r["seed"]))


def cost_est(c):
    return c["max_steps"] * (60 + 0.04 * c["N"] * c["B"]) * 1e-6


def run_one(job):
    c, traj_dir = job
    label, kw0, model = CELLS[c["cell"]]
    eta = (c["B"] / 64.0) / D ** 2 if c["cell"] == "f" else 1.0 / D ** 2
    kw = dict(kw0, N=c["N"], B=c["B"], max_steps=c["max_steps"], log_every=100, stop_m=0.5, init="fixed", m0_scale=1.0,
              rng_tag=1000, fast=True)
    c0, w0 = time.process_time(), time.time()
    r = train(model, D, 2, eta, c["seed"], **kw)
    cpu = time.process_time() - c0
    traj = r.pop("traj")
    np.savez_compressed(os.path.join(traj_dir, f"{c['cell']}_N{c['N']}_B{c['B']}_s{c['seed']}.npz"), **traj)
    free_or_tied = model == "At" or kw0.get("train_gamma", False)
    return dict(c, protocol=label, d=D, eta=eta, m0=r["m0"], steps=r["steps"], T05=r["T05"], reached05=bool(r["reached05"]),
                final_abs_m=abs(r["final_m"]), max_abs_m=r["max_abs_m"],
                readout_at_T05=(r["gamma_at_T05"] if free_or_tied else float("nan")),
                final_readout=(r["final_gamma"] if free_or_tied else float("nan")), cpu_s=cpu, wall_s=time.time() - w0)


def jobs(cap_d, seeds=3):
    J = []
    for cell in "abcde":
        for N in NS:
            for s in range(seeds):
                J.append(dict(cell=cell, N=N, B=64, seed=s, max_steps=cap_d if cell == "d" else MAX_STEPS))
    for B in (16, 256):
        for s in range(seeds):
            J.append(dict(cell="f", N=64, B=B, seed=s, max_steps=MAX_STEPS))
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
    ap.add_argument("--cpu-cap-h", type=float, default=2.0)
    ap.add_argument("--cap-d", type=int, default=MAX_STEPS)
    ap.add_argument("--only", default=None, help="comma list of cells (pilot)")
    a = ap.parse_args(argv)
    traj_dir = os.path.join(a.out, "traj")
    os.makedirs(traj_dir, exist_ok=True)
    path = os.path.join(a.out, "runs.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {key_of(r) for r in csv.DictReader(f)}
    J = [c for c in jobs(a.cap_d) if key_of(c) not in done and (a.only is None or c["cell"] in a.only.split(","))]
    # cheap cells first so that the global CPU cap can only ever cut the most expensive tail (pinned g=0.01, N=256)
    J.sort(key=lambda c: ({'f': 0, 'c': 1, 'd': 2, 'b': 3, 'a': 4, 'e': 5}[c['cell']], c['N'], c['seed']))
    tot = spent(path)
    print(f"{len(done)} done, {len(J)} to run, so far {tot/3600:.3f} h, cap {a.cpu_cap_h} h", flush=True)
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
                    ex_done = True
                    break
                nx = next(it, None)
                if nx is None:
                    ex_done = True
                    break
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
                print(f"[{n}/{len(J)} {time.time()-t0:6.0f}s wall {tot/3600:.3f} h cpu] {r['cell']} N={r['N']} B={r['B']} s={r['seed']} "
                      f"T05={r['T05']} steps={r['steps']} |m|={r['final_abs_m']:.3f} ro@T={r['readout_at_T05']:.3g} cpu={r['cpu_s']:.0f}s", flush=True)
    print(f"finished {n} runs, total CPU {spent(path)/3600:.3f} h, wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
