"""Experiment-1 sweep CLI.  Default: full sweep from docs/spec_exp1.md.  `--quick`: tiny smoke version.

Writes <out>/summary.csv (one row per run, appended as runs finish) and <out>/traj/*.npz.
Resumable: (model, k, d, eta0, seed, N, B, gamma) combinations already in summary.csv are skipped.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per worker (the numpy analogue of torch.set_num_threads(1))

import argparse
import csv
import multiprocessing as mp
import time

import numpy as np

from .train import train

COLS = ["model", "k", "d", "N", "B", "gamma", "eta0", "eta", "seed", "m0", "T05", "T09", "reached", "reached05",
        "steps", "final_m", "max_steps", "sampler", "wall_s", "sym_stop"]
KEY = ["model", "k", "d", "eta0", "seed", "N", "B", "gamma"]

# default grids (spec_exp1.md; Model A k=2 extended beyond d=48 because it is cheap, see README)
GRID_B = {1: [8, 16, 32, 64, 128], 2: [8, 16, 32, 64, 128], 3: [8, 16, 32, 64, 128]}
GRID_A = {1: [8, 16, 32, 64, 128], 2: [8, 16, 24, 32, 48, 64, 96, 128], 3: [8, 16, 32]}
KAPPA = {("B", 1): 1, ("B", 2): 2, ("B", 3): 3, ("A", 1): 2, ("A", 2): 4, ("A", 3): 6}


def _key(row):
    return (str(row["model"]), str(int(row["k"]) if str(row["k"]).isdigit() else row["k"]), int(row["d"]),
            float(row["eta0"]), int(row["seed"]), int(row["N"]), int(row["B"]), float(row["gamma"]))


def _run(job):
    cfg, traj_dir = job
    r = train(cfg["model"], cfg["d"], cfg["k"], cfg["eta0"] / cfg["d"] ** 2, cfg["seed"], N=cfg["N"], B=cfg["B"],
              gamma=cfg["gamma"], max_steps=cfg["max_steps"], log_every=cfg["log_every"], sampler=cfg["sampler"],
              init=cfg.get("init", "uniform"), m0_scale=cfg.get("m0_scale", 1.0))
    r["eta0"] = cfg["eta0"]
    name = f"{r['model']}_k{r['k']}_d{r['d']}_eta{cfg['eta0']:g}_s{r['seed']}.npz"
    np.savez_compressed(os.path.join(traj_dir, name), **r.pop("traj"))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smoke test: d in {8,16}, 1 seed, max_steps 20000")
    ap.add_argument("--out", default=None, help="output dir (default results/exp1, or results/exp1_quick)")
    ap.add_argument("--models", nargs="+", default=["A", "B"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--ds", nargs="+", type=int, default=None, help="override the d grid for all configs")
    ap.add_argument("--eta0", nargs="+", type=float, default=[2.0, 1.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=3_000_000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--sampler", choices=["proj", "full"], default="proj")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", choices=["uniform", "fixed"], default="uniform",
                    help="uniform: w uniform on the sphere (exp 1); fixed: <w,v> = m0_scale*d^-1/2 exactly (exp 1b)")
    ap.add_argument("--m0-scale", type=float, default=1.0)
    args = ap.parse_args(argv)
    if args.quick:
        args.ds, args.seeds, args.max_steps = [8, 16], 1, 20000
    out = args.out or ("results/exp1_quick" if args.quick else "results/exp1")
    traj_dir = os.path.join(out, "traj")
    os.makedirs(traj_dir, exist_ok=True)
    summ = os.path.join(out, "summary.csv")

    done = set()
    if os.path.exists(summ):
        with open(summ) as f:
            done = {_key(r) for r in csv.DictReader(f)}
    jobs = []
    for model in args.models:
        for k in args.ks:
            for d in args.ds or (GRID_A if model == "A" else GRID_B)[k]:
                for eta0 in args.eta0:
                    for seed in range(args.seeds):
                        cfg = dict(model=model, k=k, d=d, eta0=eta0, seed=seed, N=args.N, B=args.B,
                                   gamma=args.gamma, max_steps=args.max_steps, log_every=args.log_every,
                                   sampler=args.sampler, init=args.init, m0_scale=args.m0_scale)
                        if _key(cfg) not in done:
                            jobs.append((cfg, traj_dir))
    jobs.sort(key=lambda j: -(j[0]["d"] ** (KAPPA[(j[0]["model"], j[0]["k"])] / 2 + 1) / j[0]["eta0"]))
    print(f"{len(done)} runs already done, {len(jobs)} to run -> {summ}", flush=True)
    new_file = not os.path.exists(summ)
    t0 = time.time()
    with open(summ, "a", newline="") as f, mp.Pool(args.workers) as pool:
        wr = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new_file:
            wr.writeheader()
        for i, r in enumerate(pool.imap_unordered(_run, jobs), 1):
            wr.writerow(r)
            f.flush()
            print(f"[{i}/{len(jobs)} {time.time() - t0:7.0f}s] {r['model']} k={r['k']} d={r['d']} eta0={r['eta0']:g} "
                  f"seed={r['seed']} T05={r['T05']} T09={r['T09']} reached={r['reached']} "
                  f"steps={r['steps']} run={r['wall_s']:.1f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s wall", flush=True)


if __name__ == "__main__":
    main()
