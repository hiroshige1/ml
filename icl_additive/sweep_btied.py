"""Driver for the B-tied baseline (exp 2 add-on): jobs eta_mult:seed (eta = eta_mult / d^2), 4 workers x 1 BLAS thread.
Writes results/exp2/btied/runs/<tag>/seed<s>.npz and results/exp2/btied/runs.csv (resumable).  tag = Btied_eta<mult>."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

from .btied import train_btied

COLS = ["tag", "seed", "eta_mult", "eta", "rho0", "M", "P", "d", "B", "steps", "cpu_s", "wall_s"]


def tag_of(mult):
    return f"Btied_eta{mult:g}"


def run_one(job):
    mult, seed, max_steps, out = job
    tag = tag_of(mult)
    os.makedirs(os.path.join(out, "runs", tag), exist_ok=True)
    r = train_btied(seed, eta=mult / 32 ** 2, max_steps=max_steps)
    traj, V = r.pop("traj"), r.pop("V")
    np.savez_compressed(os.path.join(out, "runs", tag, f"seed{seed}.npz"), V=V, **traj)
    return dict(tag=tag, seed=seed, eta_mult=mult, eta=r["eta"], rho0=0.01, M=64, P=16, d=32, B=32, steps=r["steps"], cpu_s=r["cpu_s"], wall_s=r["wall_s"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp2/btied")
    ap.add_argument("--jobs", nargs="+", required=True, help="eta_mult:seed, e.g. 1:0 4:0")
    ap.add_argument("--max-steps", type=int, default=3_000_000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "runs.csv")
    done = set()
    if os.path.exists(path):
        done = {(r["tag"], int(r["seed"])) for r in csv.DictReader(open(path))}
    jobs = []
    for j in a.jobs:
        m, s = j.split(":")
        if (tag_of(float(m)), int(s)) not in done:
            jobs.append((float(m), int(s), a.max_steps, a.out))
    new = not os.path.exists(path)
    t0 = time.time()
    with open(path, "a", newline="") as f, ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:
        wr = csv.DictWriter(f, fieldnames=COLS)
        if new:
            wr.writeheader()
        pend = {ex.submit(run_one, j) for j in jobs}
        while pend:
            fin, pend = wait(pend, return_when=FIRST_COMPLETED)
            for fu in fin:
                r = fu.result()
                wr.writerow(r); f.flush()
                print(f"[{time.time() - t0:6.0f}s] {r['tag']} seed {r['seed']} cpu={r['cpu_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
