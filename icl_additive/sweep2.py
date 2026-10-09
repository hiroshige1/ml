"""Experiment-2 driver (docs/spec_exp2.md): Model A / Model B runs, `--workers` processes, 1 BLAS thread each.

Each job (model, alpha, rho, seed) writes results/exp2/runs/<tag>/seed<s>.npz and appends a row to results/exp2/runs.csv
(steps, stop reason, CPU seconds measured in the worker).  Jobs already in runs.csv are skipped (resumable).  New jobs are
dispatched only while the CPU time already spent (+ --cpu-already, e.g. pilot runs) is below --cpu-cap-h.
tag = A_a1.5 | B_a1.5_rho0.1 | ...
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

from .many import train_many

COLS = ["tag", "model", "alpha", "rho", "gamma", "seed", "M", "P", "d", "N", "B", "max_steps", "steps", "stop_reason", "cpu_s", "wall_s"]


def tag_of(model, alpha, rho):
    return f"A_a{alpha}" if model == "A" else f"B_a{alpha}_rho{rho}"


def run_one(job):
    model, alpha, rho, seed, max_steps, M, out = job
    tag = tag_of(model, alpha, rho)
    os.makedirs(os.path.join(out, "runs", tag), exist_ok=True)
    r = train_many(model, seed, alpha=alpha, rho=rho, M=M, max_steps=max_steps)
    traj, V = r.pop("traj"), r.pop("V")
    np.savez_compressed(os.path.join(out, "runs", tag, f"seed{seed}.npz"), V=V, **traj)
    r["tag"] = tag
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp2")
    ap.add_argument("--jobs", nargs="+", required=True,
                    help="entries model:alpha:rho:seed:max_steps, e.g. A:1.5:0:0:1000000 B:1.5:0.1:0:3000000")
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu-cap-h", type=float, default=4.0)
    ap.add_argument("--cpu-already", type=float, default=0.0, help="hours already spent outside runs.csv (pilots)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, "runs.csv")
    done, spent = set(), a.cpu_already * 3600
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path)):
            spent += float(r["cpu_s"])
            done.add((r["tag"], int(r["seed"])))
    jobs = []
    for j in a.jobs:
        m, al, rho, s, ms = j.split(":")
        al, rho = float(al), float(rho)
        if (tag_of(m, al, rho), int(s)) not in done:
            jobs.append((m, al, rho, int(s), int(ms), a.M, a.out))
    print(f"{len(done)} done, {len(jobs)} to run; CPU spent {spent / 3600:.3f} h (cap {a.cpu_cap_h} h)", flush=True)
    new = not os.path.exists(csv_path)
    t0 = time.time()
    with open(csv_path, "a", newline="") as f, ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:
        wr = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new:
            wr.writeheader()
        it, pending, exhausted = iter(jobs), set(), False
        while True:
            while not exhausted and len(pending) < a.workers:
                if spent >= a.cpu_cap_h * 3600:
                    print("CPU cap reached; not dispatching further jobs", flush=True)
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
                spent += r["cpu_s"]
                print(f"[{time.time() - t0:6.0f}s wall] {r['tag']} seed {r['seed']} steps={r['steps']} stop={r['stop_reason']} "
                      f"cpu={r['cpu_s']:.0f}s total {spent / 3600:.3f} h", flush=True)


if __name__ == "__main__":
    main()
