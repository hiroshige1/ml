"""Experiment 14: B-dependence of the rbf-softmax + Adam escape at N=16 (pre-registration P27).

Reuses exp 13's model and run function unchanged (icl_additive.ablation_exp13.run): rbf softmax, beta=0.3, Gamma=1, d=64, m_0=1/8,
Adam lr 1e-3, cap 1e5 steps, w renormalised each step, m logged every 100 steps, stop at |m| >= 0.5.
Cells: N=16, B in {16, 64, 256, 1024}, seeds {0, 1, 2}.  (B, seed) pairs already run in exp 13 (B=64, 1024; seeds 0, 1) are copied from
results/exp13/runs.csv (source=exp13) and not re-run.  8 fresh runs.

Usage:
  python -m icl_additive.ablation_exp14 --worker K     # claim-file job queue (longest jobs first)
  python -m icl_additive.ablation_exp14 --collect      # merge rows/*.json + reused exp 13 rows into results/exp14/runs.csv
Budget stop: touching results/exp14/STOP (all) or results/exp14/STOP_<run name> ends a run at the next 100-step boundary.
"""
import argparse
import csv
import json
import os
import sys

from . import ablation_exp13 as e13

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "exp14")
E13_CSV = os.path.join(ROOT, "results", "exp13", "runs.csv")
N = 16
BS = (16, 64, 256, 1024)
SEEDS = (0, 1, 2)
REUSED = {(64, 0), (64, 1), (1024, 0), (1024, 1)}
e13.OUT = OUT  # stop-file location used by e13.stop_requested -> exp14 folder (exp 13's folder is not touched)

COLS = ["model", "opt", "N", "B", "seed", "steps", "T05", "reached05", "final_abs_m", "max_abs_m", "max_dm_step", "cpu_s",
        "censored_by", "lr_adam", "source"]


def name_of(B, seed):
    return f"rbf_adam_N{N}_B{B}_s{seed}"


def jobs():
    """Fresh runs only, longest first (large B costs most per step)."""
    js = [(B, s) for B in sorted(BS, reverse=True) for s in SEEDS if (B, s) not in REUSED]
    assert len(js) == 8
    return js


def worker(k):
    rows = os.path.join(OUT, "rows")
    claims = os.path.join(OUT, "claims")
    os.makedirs(rows, exist_ok=True)
    os.makedirs(claims, exist_ok=True)
    for (B, seed) in jobs():
        name = name_of(B, seed)
        try:
            os.close(os.open(os.path.join(claims, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        print(f"[w{k}] start {name}", flush=True)
        r = e13.run("adam", N, B, seed, trace_dir=os.path.join(OUT, "traces"), log=sys.stdout)
        with open(os.path.join(rows, name + ".json"), "w") as f:
            json.dump(r, f)
        print(f"[w{k}] done {name} steps={r['steps']} T05={r['T05']} max|m|={r['max_abs_m']:.3f} cpu={r['cpu_s']:.0f}s "
              f"censored_by={r['censored_by']}", flush=True)


def collect():
    out = []
    with open(E13_CSV) as f:
        for r in csv.DictReader(f):
            if r["model"] == "rbf" and r["opt"] == "adam" and int(r["N"]) == N and (int(r["B"]), int(r["seed"])) in REUSED:
                r["source"] = "exp13"
                out.append(r)
    assert len(out) == 4, len(out)
    for (B, seed) in jobs():
        p = os.path.join(OUT, "rows", name_of(B, seed) + ".json")
        r = json.load(open(p))
        r["source"] = "exp14"
        out.append(r)
    out.sort(key=lambda r: (int(r["B"]), int(r["seed"])))
    with open(os.path.join(OUT, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.worker is not None:
        worker(a.worker)
    elif a.collect:
        collect()
