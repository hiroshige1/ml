"""Experiment-1b sweeps (docs/spec_exp1b.md, 'Revision after exp 1').  Resumable CSVs under results/exp1b/.

Phases (priority order):
  thr    P2/P3: fixed gamma in {1.0, 0.1}, N in {32,128,512}, d in {16,32,64}, 3 seeds, eta = 1/d^2, 3e5 steps, m0 = d^-1/2
  p7     P7: gamma=1, N=128, d=64, B in {32,256,1024}, steps {3e5,1e5,3e4}
  tied1  P4' side check: tied readout with rho0 = 1 (N=128, d in {32,64}, 3e5 steps)
  kappa  P4/P4': d=32, N in {32,128,512,4096}, protocols free1 / free10 / tied, m0 in {0.5,0.7,1.0,1.4}*d^-1/2, 3 seeds
  dscan  priority 3: free-Gamma T_0.5 vs d in {16,24,32,48}, N in {128,2048}
  single one run with the individual flags (--model/--init/--m0-scale/--train-gamma/--gamma0/--eta-gamma-mult)

Every run is pure numpy / float64 with one BLAS thread; the CPU time of each run is measured inside the worker
(time.process_time) and stored as cpu_s.  The driver stops dispatching new jobs once the CPU time already spent
(all CSVs in the output directory) reaches --cpu-cap-h (global) or --phase-cap-h (this phase).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"  # one thread per worker

import argparse
import csv
import glob
import math
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

from .train import train

OUT = "results/exp1b"
COLS = ["phase", "protocol", "model", "k", "N", "B", "d", "eta0", "eta", "seed", "gamma", "gamma0", "eta_gamma_mult",
        "rho0", "m0_scale", "m0", "max_steps", "steps", "T05", "reached05", "final_abs_m", "max_abs_m", "exceeded05",
        "stuck", "final_gamma", "gamma_at_T05", "cpu_s", "wall_s"]
KEY = ["phase", "protocol", "N", "B", "d", "eta0", "seed", "gamma", "m0_scale", "rho0", "max_steps"]
M0_MULT = [0.5, 0.7, 1.0, 1.4]
# step caps for the kappa phase.  Spec: 1e6.  Tied runs are cheap (T ~ 1e4) and keep 1e6; the free-Gamma runs were capped
# lower to stay inside the 3 CPU-hour budget (pilot: free eta_Gamma=eta did not reach m=0.5 in 1e5 steps at any N).
KAPPA_STEPS = {("tied", 32): 1_000_000, ("tied", 128): 1_000_000, ("tied", 512): 1_000_000, ("tied", 4096): 1_000_000,
               ("free1", 32): 200_000, ("free1", 128): 200_000, ("free1", 512): 300_000, ("free1", 4096): 120_000,
               ("free10", 32): 200_000, ("free10", 128): 200_000, ("free10", 512): 300_000, ("free10", 4096): 200_000}


def _fmt(x):
    return f"{float(x):.6g}"


def key_of(row):
    return tuple(_fmt(row[c]) if c not in ("phase", "protocol") else str(row[c]) for c in KEY)


def cost_est(cfg):
    """Rough CPU seconds (measured us/step scaling on this machine), used only for ordering and planning."""
    N, B = cfg["N"], cfg["B"]
    us = 60 + 0.04 * N * B
    return cfg["max_steps"] * us * 1e-6


def make_cfg(phase, protocol, N, B, d, seed, max_steps, gamma=float("nan"), m0_scale=1.0, rho0=float("nan"),
             eta0=1.0, gamma0=0.01):
    return dict(phase=phase, protocol=protocol, N=N, B=B, d=d, seed=seed, max_steps=max_steps, gamma=gamma,
                m0_scale=m0_scale, rho0=rho0, eta0=eta0, gamma0=gamma0, k=2)


def run_one(job):
    cfg, traj_dir = job
    p = cfg["protocol"]
    d, eta = cfg["d"], cfg["eta0"] / cfg["d"] ** 2
    kw = dict(N=cfg["N"], B=cfg["B"], max_steps=cfg["max_steps"], log_every=500, stop_m=0.5, init="fixed",
              m0_scale=cfg["m0_scale"], rng_tag=int(round(1000 * cfg["m0_scale"])), fast=True)
    if p == "fixed":
        kw.update(gamma=cfg["gamma"])
    elif p in ("free1", "free10"):
        kw.update(gamma=cfg["gamma0"], train_gamma=True, eta_gamma_mult=1.0 if p == "free1" else 10.0)
    elif p == "tied":
        kw.update(rho0=cfg["rho0"])
    else:
        raise ValueError(p)
    model = "A" if p in ("fixed", "free1", "free10") else "At"
    c0, w0 = time.process_time(), time.time()
    r = train(model, d, cfg["k"], eta, cfg["seed"], **kw)
    cpu = time.process_time() - c0
    traj = r.pop("traj")
    name = (f"{cfg['phase']}_{p}_N{cfg['N']}_B{cfg['B']}_d{d}_g{_fmt(cfg['gamma'])}_r{_fmt(cfg['rho0'])}"
            f"_m{cfg['m0_scale']:g}_s{cfg['seed']}.npz")
    np.savez_compressed(os.path.join(traj_dir, name), **traj)
    row = dict(cfg)
    reached = bool(r["reached05"])
    row.update(model=model, eta=eta, m0=r["m0"], steps=r["steps"], T05=r["T05"], reached05=reached,
               final_abs_m=abs(r["final_m"]), max_abs_m=r["max_abs_m"], exceeded05=bool(r["max_abs_m"] > 0.5),
               stuck=bool((not reached) and abs(r["final_m"]) < 2.0 * abs(r["m0"])), final_gamma=r["final_gamma"],
               gamma_at_T05=r["gamma_at_T05"], gamma0=cfg["gamma0"] if p.startswith("free") else float("nan"),
               eta_gamma_mult={"free1": 1.0, "free10": 10.0}.get(p, float("nan")), cpu_s=cpu,
               wall_s=time.time() - w0)
    return row


def phase_jobs(phase, seeds=3):
    J = []
    if phase == "thr":
        for gamma in (1.0, 0.1):
            for N in (32, 128, 512):
                for d in (16, 32, 64):
                    for s in range(seeds):
                        J.append(make_cfg("thr", "fixed", N, 32, d, s, 300_000, gamma=gamma))
    elif phase == "p7":
        for B, steps in ((32, 300_000), (256, 100_000), (1024, 30_000)):
            for s in range(seeds):
                J.append(make_cfg("p7", "fixed", 128, B, 64, s, steps, gamma=1.0))
    elif phase == "tied1":
        for d in (32, 64):
            for s in range(seeds):
                J.append(make_cfg("tied1", "tied", 128, 32, d, s, 300_000, rho0=1.0))
    elif phase == "kappa":
        for N, B in ((32, 32), (128, 32), (512, 8), (4096, 8)):
            for p in ("free1", "free10", "tied"):
                for mm in M0_MULT:
                    for s in range(seeds):
                        J.append(make_cfg("kappa", p, N, B, 32, s, KAPPA_STEPS[(p, N)], m0_scale=mm,
                                          rho0=0.01 if p == "tied" else float("nan")))
    elif phase == "kappa2":
        # re-run (from scratch, same seeds => identical trajectory up to the old cap) the kappa runs that were censored
        # at the reduced caps, with the spec's 1e6 steps where affordable (N=4096 free1: 3e5)
        import pandas as pd
        k = pd.read_csv(os.path.join(OUT, "kappa.csv"))
        for _, r in k[~k.reached05].iterrows():
            ms = 300_000 if (r.protocol == "free1" and r.N == 4096) else 1_000_000
            if ms > r.max_steps:
                J.append(make_cfg("kappa2", r.protocol, int(r.N), int(r.B), int(r.d), int(r.seed), ms,
                                  m0_scale=float(r.m0_scale), rho0=float(r.rho0) if r.protocol == "tied" else float("nan")))
    elif phase == "bctl":
        # control for the B confound in the kappa table (B=8 was used at N>=512, B=32 at N<=128): free1, N=128, B=8
        for mm in M0_MULT:
            for s in range(seeds):
                J.append(make_cfg("bctl", "free1", 128, 8, 32, s, 1_000_000, m0_scale=mm))
    elif phase == "dscan":
        for N, B in ((128, 32), (2048, 8)):
            for d in (16, 24, 32, 48):
                for s in range(seeds):
                    J.append(make_cfg("dscan", "free1", N, B, d, s, 1_000_000))
    else:
        raise ValueError(phase)
    return J


def spent_cpu(out):
    tot = 0.0
    for f in glob.glob(os.path.join(out, "*.csv")):
        if os.path.basename(f)[:-4] not in ("thr", "p7", "tied1", "kappa", "kappa2", "bctl", "dscan"):
            continue  # skip derived files (summary.csv duplicates the run rows, tables carry no CPU time)
        with open(f) as fh:
            for r in csv.DictReader(fh):
                try:
                    tot += float(r["cpu_s"])
                except (KeyError, ValueError):
                    pass
    return tot


def run_phase(phase, out, workers, cpu_cap_h, phase_cap_h, seeds=3):
    traj_dir = os.path.join(out, "traj")
    os.makedirs(traj_dir, exist_ok=True)
    csv_path = os.path.join(out, f"{phase}.csv")
    done = set()
    phase_cpu = 0.0
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                done.add(key_of(r))
                phase_cpu += float(r["cpu_s"])
    jobs = [(c, traj_dir) for c in phase_jobs(phase, seeds) if key_of(c) not in done]
    jobs.sort(key=lambda j: -cost_est(j[0]))
    total_before = spent_cpu(out)
    print(f"[{phase}] {len(done)} done, {len(jobs)} to run; CPU spent so far {total_before / 3600:.3f} h "
          f"(global cap {cpu_cap_h} h, phase cap {phase_cap_h} h, phase so far {phase_cpu / 3600:.3f} h)", flush=True)
    new = not os.path.exists(csv_path)
    t0 = time.time()
    skipped = 0
    with open(csv_path, "a", newline="") as f, ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        wr = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new:
            wr.writeheader()
        pending, it, n_done = set(), iter(jobs), 0
        running_est = 0.0
        exhausted = False
        while True:
            while not exhausted and len(pending) < workers:
                over = (total_before + 0 >= cpu_cap_h * 3600) or (phase_cpu >= phase_cap_h * 3600)
                if over:
                    exhausted = True
                    break
                nxt = next(it, None)
                if nxt is None:
                    exhausted = True
                    break
                pending.add(ex.submit(run_one, nxt))
            if not pending:
                break
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fu in fin:
                r = fu.result()
                wr.writerow(r)
                f.flush()
                n_done += 1
                total_before += r["cpu_s"]
                phase_cpu += r["cpu_s"]
                print(f"[{phase} {n_done}/{len(jobs)} {time.time() - t0:6.0f}s wall, {total_before / 3600:.3f} h cpu] "
                      f"{r['protocol']} N={r['N']} B={r['B']} d={r['d']} g={r['gamma']} m0s={r['m0_scale']} "
                      f"s={r['seed']} T05={r['T05']} steps={r['steps']} final|m|={r['final_abs_m']:.3f} "
                      f"cpu={r['cpu_s']:.0f}s", flush=True)
        skipped = len(jobs) - n_done
    print(f"[{phase}] finished: ran {n_done}, NOT run (budget) {skipped}; wall {time.time() - t0:.0f}s; "
          f"total CPU {spent_cpu(out) / 3600:.3f} h", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=["thr", "p7", "tied1", "kappa", "kappa2", "bctl", "dscan", "single"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu-cap-h", type=float, default=3.0)
    ap.add_argument("--phase-cap-h", type=float, default=3.0)
    ap.add_argument("--seeds", type=int, default=3)
    # single-run flags
    ap.add_argument("--model", choices=["A", "At"], default="A")
    ap.add_argument("--init", choices=["uniform", "fixed"], default="fixed")
    ap.add_argument("--m0-scale", type=float, default=1.0, help="m_0 = m0_scale * d^-1/2 (init fixed)")
    ap.add_argument("--train-gamma", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.1, help="fixed Gamma (model A without --train-gamma)")
    ap.add_argument("--gamma0", type=float, default=0.01)
    ap.add_argument("--eta-gamma-mult", type=float, default=1.0)
    ap.add_argument("--rho0", type=float, default=0.01)
    ap.add_argument("-d", type=int, default=32)
    ap.add_argument("-N", type=int, default=128)
    ap.add_argument("-B", type=int, default=32)
    ap.add_argument("--eta0", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.phase == "single":
        c0 = time.process_time()
        g = a.gamma0 if a.train_gamma else a.gamma
        r = train(a.model, a.d, 2, a.eta0 / a.d ** 2, a.seed, N=a.N, B=a.B, gamma=g, train_gamma=a.train_gamma,
                  max_steps=a.max_steps, stop_m=0.5, init=a.init, m0_scale=a.m0_scale, eta_gamma_mult=a.eta_gamma_mult,
                  rho0=a.rho0, rng_tag=int(round(1000 * a.m0_scale)), fast=True)
        r.pop("traj")
        print(r, f"cpu_s={time.process_time() - c0:.1f}")
        return
    run_phase(a.phase, a.out, a.workers, a.cpu_cap_h, a.phase_cap_h, a.seeds)


if __name__ == "__main__":
    main()
