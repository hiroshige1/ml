"""Experiment-3 driver (docs/spec_exp3.md): 5 seeds of the M=4, P=2 multi-neuron Model A, `--workers` processes.

Each seed writes results/exp3/<tag>/traj/seed<s>.npz (all logged arrays) and appends one row to results/exp3/<tag>/runs.csv
(steps, stop reason, CPU seconds measured inside the worker).  The driver stops dispatching new seeds once the CPU time
already spent in <tag> reaches --cpu-cap-h.  Resumable: seeds already in runs.csv are skipped.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import json
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

from .multi import train_multi

COLS = ["seed", "gamma", "M", "P", "N", "B", "d", "max_steps", "steps", "stop_reason", "stop_align", "cpu_s", "wall_s"]


def run_one(job):
    seed, kw, traj_dir = job
    r = train_multi(seed, **kw)
    traj = r.pop("traj")
    np.savez_compressed(os.path.join(traj_dir, f"seed{seed}.npz"), **traj)
    row = dict(r)
    row["stop_align"] = kw.get("stop_align")
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default="results/exp3")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=1_500_000)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--stop-mse", type=float, default=0.1)
    ap.add_argument("--stop-align", type=float, default=None)
    ap.add_argument("--align-hold", type=int, default=0)
    ap.add_argument("--cpu-cap-h", type=float, default=3.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    out = os.path.join(a.out, a.tag)
    traj_dir = os.path.join(out, "traj")
    os.makedirs(traj_dir, exist_ok=True)
    csv_path = os.path.join(out, "runs.csv")
    done, spent = set(), 0.0
    for f in os.listdir(a.out):  # CPU already spent in every tag under --out counts against the cap
        p = os.path.join(a.out, f, "runs.csv")
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                spent += float(r["cpu_s"])
                if f == a.tag:
                    done.add(int(r["seed"]))
    kw = dict(gamma=a.gamma, max_steps=a.max_steps, stop_mse=a.stop_mse, stop_align=a.stop_align,
              align_hold=a.align_hold, verbose=a.verbose)
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"))
    jobs = [(s, kw, traj_dir) for s in a.seeds if s not in done]
    print(f"[{a.tag}] {len(done)} done, {len(jobs)} to run; CPU spent so far {spent / 3600:.3f} h (cap {a.cpu_cap_h} h)", flush=True)
    new = not os.path.exists(csv_path)
    t0 = time.time()
    with open(csv_path, "a", newline="") as f, ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:
        wr = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new:
            wr.writeheader()
        it, pending = iter(jobs), set()
        exhausted = False
        while True:
            while not exhausted and len(pending) < a.workers:
                if spent >= a.cpu_cap_h * 3600:
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
                print(f"[{a.tag} {time.time() - t0:6.0f}s wall] seed {r['seed']} steps={r['steps']} stop={r['stop_reason']} "
                      f"cpu={r['cpu_s']:.0f}s total {spent / 3600:.3f} h", flush=True)


if __name__ == "__main__":
    main()
