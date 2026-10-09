"""Experiment 15: does Adam make prompts per step (B) matter when the drift is positive?  (P28a/b)

Reuses the exp-12 linear single-feature model (pinned Gamma=1, sigma_2, d=64, m_0 = 1/8 exactly) by import: `ablation_exp12.run`
(Adam: full d=64 data; SGD: distribution-exact projected sampler).  Only the grid, step caps and logging interval are set here.
N=1024, B in {16,64,256,1024}, seeds {0,1,2}; (a) Adam lr 1e-3, cap 2e4 steps, m logged every 10 steps;
(b) SGD eta = 1/4096, cap 3e5 steps, m logged every 100 steps.  Stop at |m| >= 0.5.

Usage:
  python -m icl_additive.ablation_exp15 --worker K [--drop-sgd1024-seed2]
  python -m icl_additive.ablation_exp15 --timing
"""
import argparse
import json
import os
import time

from . import ablation_exp12 as e12

N = 1024
BS = (16, 64, 256, 1024)
SEEDS = (0, 1, 2)
CAPS = {"sgd": 300_000, "adam": 20_000}
LOG = {"sgd": 100, "adam": 10}
LR_ADAM = 1e-3
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp15")


def jobs():
    """(opt, B, seed); most expensive first."""
    order = [("sgd", 1024), ("sgd", 256), ("adam", 1024), ("adam", 256), ("adam", 64), ("adam", 16), ("sgd", 64), ("sgd", 16)]
    return [(opt, B, s) for opt, B in order for s in SEEDS]


def run_one(opt, B, seed, trace_dir=None, max_steps=None):
    e12.LOG_EVERY = LOG[opt]  # run() reads this module global at call time
    return e12.run("linear", opt, N, B, seed, lr_adam=LR_ADAM, trace_dir=trace_dir, max_steps=max_steps or CAPS[opt])


def worker(k, drop=()):
    rows = os.path.join(OUT, "rows")
    claims = os.path.join(OUT, "claims")
    os.makedirs(rows, exist_ok=True)
    os.makedirs(claims, exist_ok=True)
    for (opt, B, seed) in jobs():
        if (opt, B, seed) in drop:
            continue
        name = f"linear_{opt}_N{N}_B{B}_s{seed}"
        if os.path.exists(os.path.join(rows, name + ".json")):  # resumable: completed rows are never re-run
            continue
        try:
            fd = os.open(os.path.join(claims, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue
        print(f"[w{k}] start {name}", flush=True)
        r = run_one(opt, B, seed, trace_dir=os.path.join(OUT, "traces"))
        r["lr_adam"] = LR_ADAM if opt == "adam" else float("nan")
        r["log_every"] = LOG[opt]
        with open(os.path.join(rows, name + ".json"), "w") as f:
            json.dump(r, f)
        print(f"[w{k}] done {name} steps={r['steps']} T05={r['T05']} max|m|={r['max_abs_m']:.3f} cpu={r['cpu_s']:.0f}s", flush=True)


def timing():
    for opt in ("sgd", "adam"):
        for B in BS:
            n = 30 if opt == "adam" else 200
            t = time.process_time()
            run_one(opt, B, 0, max_steps=n)
            print(opt, B, f"{(time.process_time() - t) / n * 1e3:.2f} ms/step", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--drop-sgd1024-seed2", action="store_true")
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    if a.timing:
        timing()
    elif a.worker is not None:
        worker(a.worker, drop={("sgd", 1024, 2)} if a.drop_sgd1024_seed2 else ())
