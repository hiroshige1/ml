"""Experiment 6: N-T exchange by readout protocol (docs/preregistration.md, 'N-T exchange by readout protocol (exp 6)').

Model A single neuron, k=2, d=64, m_0 = d^-1/2 exactly, eta = 1/d^2, N in {16,64,256}.
Batch schemes: 'tok' (N*B = 4096: B = 256/64/16) and 'B64' (B = 64 for all N).
Protocols: fixed1 (Gamma=1), fixed01 (Gamma=0.1), free (Gamma_0=0.01, eta_Gamma=eta), tied (rho_0=0.01).
3 seeds, early stop at |m| >= 0.5.  Reuses train.train (fast sampler) and the sweep1b conventions (1 BLAS thread / worker,
resumable CSV, CPU time measured per run, global CPU cap).
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

OUT = "results/exp6"
D = 64
COLS = ["protocol", "scheme", "N", "B", "d", "eta", "seed", "m0", "max_steps", "steps", "T05", "reached05", "final_abs_m",
        "max_abs_m", "readout_at_T05", "final_readout", "cpu_s", "wall_s"]
PROTOS = ["fixed1", "fixed01", "free", "tied"]
NS = [16, 64, 256]
SCHEMES = {"tok": {16: 256, 64: 64, 256: 16}, "B64": {16: 64, 64: 64, 256: 64}}
MAX_STEPS = 1_000_000
CAP_FIXED1 = {"fixed1": MAX_STEPS}  # overridden by --fixed1-steps


def key_of(r):
    return (r["protocol"], r["scheme"], int(r["N"]), int(r["seed"]))


def cost_est(c):
    return c["max_steps"] * (60 + 0.04 * c["N"] * c["B"]) * 1e-6


def run_one(job):
    c, traj_dir = job
    p, d = c["protocol"], D
    eta = 1.0 / d ** 2
    kw = dict(N=c["N"], B=c["B"], max_steps=c["max_steps"], log_every=100, stop_m=0.5, init="fixed", m0_scale=1.0,
              rng_tag=1000, fast=True)
    if p == "fixed1":
        kw.update(gamma=1.0)
    elif p == "fixed01":
        kw.update(gamma=0.1)
    elif p == "free":
        kw.update(gamma=0.01, train_gamma=True, eta_gamma_mult=1.0)
    else:
        kw.update(rho0=0.01)
    model = "At" if p == "tied" else "A"
    c0, w0 = time.process_time(), time.time()
    r = train(model, d, 2, eta, c["seed"], **kw)
    cpu = time.process_time() - c0
    traj = r.pop("traj")
    np.savez_compressed(os.path.join(traj_dir, f"{p}_{c['scheme']}_N{c['N']}_s{c['seed']}.npz"), **traj)
    reached = bool(r["reached05"])
    row = dict(c, d=d, eta=eta, m0=r["m0"], steps=r["steps"], T05=r["T05"], reached05=reached,
               final_abs_m=abs(r["final_m"]), max_abs_m=r["max_abs_m"],
               readout_at_T05=(r["gamma_at_T05"] if p in ("free", "tied") else float("nan")),
               final_readout=(r["final_gamma"] if p in ("free", "tied") else float("nan")), cpu_s=cpu,
               wall_s=time.time() - w0)
    return row


def jobs(fixed1_steps, seeds=3):
    J = []
    for p in PROTOS:
        for sch, Bs in SCHEMES.items():
            for N in NS:
                for s in range(seeds):
                    J.append(dict(protocol=p, scheme=sch, N=N, B=Bs[N], seed=s,
                                  max_steps=fixed1_steps if p == "fixed1" else MAX_STEPS))
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
    ap.add_argument("--cpu-cap-h", type=float, default=3.0)
    ap.add_argument("--fixed1-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--only", default=None, help="comma list of protocols")
    a = ap.parse_args(argv)
    traj_dir = os.path.join(a.out, "traj")
    os.makedirs(traj_dir, exist_ok=True)
    path = os.path.join(a.out, "runs.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {key_of(r) for r in csv.DictReader(f)}
    J = [c for c in jobs(a.fixed1_steps) if key_of(c) not in done and (a.only is None or c["protocol"] in a.only.split(","))]
    J.sort(key=lambda c: -cost_est(c))
    tot = spent(path)
    print(f"{len(done)} done, {len(J)} to run, cpu so far {tot/3600:.3f} h, cap {a.cpu_cap_h} h", flush=True)
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
                print(f"[{n}/{len(J)} {time.time()-t0:6.0f}s wall {tot/3600:.3f} h cpu] {r['protocol']} {r['scheme']} N={r['N']} "
                      f"B={r['B']} s={r['seed']} T05={r['T05']} steps={r['steps']} |m|={r['final_abs_m']:.3f} "
                      f"ro@T={r['readout_at_T05']:.3g} cpu={r['cpu_s']:.0f}s", flush=True)
    print(f"finished {n} runs, total CPU {spent(path)/3600:.3f} h, wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
