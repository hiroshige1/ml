"""Experiment 9: k*=3 sanity run (docs/spec_exp9.md; docs/preregistration.md, 'k*=3 sanity run (exp 9, backlog N3)').

Single-neuron code of exps 6-7 with k=3 (sigma_3 = He_3/sqrt(6)), m_0 = d^-1/2 exactly, B=64, eta=2e-4 for every cell, early stop |m|>=0.5,
cap 1e6 steps, 3 seeds.  Cells: a tied rho0=0.01 N=128; b tied rho0=0.01 N=16; c pinned gamma=0.2 N=128; d in {8,16,32}.  27 runs.
Resumable CSV, per-run CPU time, global CPU cap (stops launching new runs), 1 BLAS thread/worker.
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

OUT = "results/exp9"
K = 3
ETA = 2e-4
B = 64
DS = (8, 16, 32)
MAX_STEPS = 1_000_000
COLS = ["cell", "protocol", "N", "B", "d", "eta", "seed", "m0", "max_steps", "steps", "T05", "reached05", "final_abs_m",
        "max_abs_m", "readout_at_T05", "final_readout", "cpu_s", "wall_s"]
# cell -> (label, kwargs for train, model, N)
CELLS = {
    "a": ("tied rho0=0.01 N=128", dict(rho0=0.01), "At", 128),
    "b": ("tied rho0=0.01 N=16", dict(rho0=0.01), "At", 16),
    "c": ("pinned gamma=0.2 N=128", dict(gamma=0.2, train_gamma=False), "A", 128),
}
# launch order (short first, capped last): (cell, d)
ORDER = [("a", 8), ("a", 16), ("b", 8), ("b", 16), ("c", 8), ("a", 32), ("b", 32), ("c", 16), ("c", 32)]


def key_of(r):
    return (r["cell"], int(r["d"]), int(r["seed"]))


def run_one(job):
    c, traj_dir = job
    label, kw0, model, N = CELLS[c["cell"]]
    kw = dict(kw0, N=N, B=B, max_steps=c["max_steps"], log_every=100, stop_m=0.5, init="fixed", m0_scale=1.0, rng_tag=1000, fast=True)
    c0, w0 = time.process_time(), time.time()
    r = train(model, c["d"], K, ETA, c["seed"], **kw)
    cpu = time.process_time() - c0
    traj = r.pop("traj")
    np.savez_compressed(os.path.join(traj_dir, f"{c['cell']}_d{c['d']}_s{c['seed']}.npz"), **traj)
    tied = model == "At"
    return dict(c, protocol=label, N=N, B=B, eta=ETA, m0=r["m0"], steps=r["steps"], T05=r["T05"], reached05=bool(r["reached05"]),
                final_abs_m=abs(r["final_m"]), max_abs_m=r["max_abs_m"],
                readout_at_T05=(r["gamma_at_T05"] if tied else float("nan")),
                final_readout=(r["final_gamma"] if tied else float("nan")), cpu_s=cpu, wall_s=time.time() - w0)


def jobs(max_steps=MAX_STEPS, seeds=3):
    return [dict(cell=cell, d=d, seed=s, max_steps=max_steps) for cell, d in ORDER for s in range(seeds)]


def spent(path):
    if not os.path.exists(path):
        return 0.0
    with open(path) as f:
        return sum(float(r["cpu_s"]) for r in csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpu-cap-h", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--only", default=None, help="comma list of cells (smoke test)")
    ap.add_argument("--only-d", default=None, help="comma list of d (smoke test)")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    traj_dir = os.path.join(a.out, "traj")
    os.makedirs(traj_dir, exist_ok=True)
    path = os.path.join(a.out, "runs.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {key_of(r) for r in csv.DictReader(f)}
    J = [c for c in jobs(a.max_steps, a.seeds) if key_of(c) not in done and (a.only is None or c["cell"] in a.only.split(","))
         and (a.only_d is None or str(c["d"]) in a.only_d.split(","))]
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
                print(f"[{n}/{len(J)} {time.time()-t0:6.0f}s wall {tot/3600:.3f} h cpu] {r['cell']} d={r['d']} s={r['seed']} "
                      f"T05={r['T05']} steps={r['steps']} |m|={r['final_abs_m']:.3f} ro@T={r['readout_at_T05']:.3g} cpu={r['cpu_s']:.0f}s", flush=True)
    print(f"finished {n} runs, total CPU {spent(path)/3600:.3f} h, wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
