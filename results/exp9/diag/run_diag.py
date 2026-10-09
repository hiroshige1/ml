"""Post-hoc diagnostics P22d/P22e for exp 9 (declared in docs/preregistration.md before running).
P22d: cell a (tied rho0=0.01, N=128), d=8,  eta=5e-5, B=64   -> ODE 24632 steps (tau=1.232)
P22e: cell a (tied rho0=0.01, N=128), d=32, eta=2e-4, B=256  -> ODE 132880 steps (tau=26.58)
Same seeds / rng_tag / init as sweep9. Run from /home/user/ml:  OMP_NUM_THREADS=1 python3 results/exp9/diag/run_diag.py
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import csv, time, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.getcwd())
from icl_additive.train import train
JOBS = [("P22d", 8, 5e-5, 64, s) for s in range(3)] + [("P22e", 32, 2e-4, 256, s) for s in range(3)]
def run(j):
    tag, d, eta, B, seed = j
    c0 = time.process_time()
    r = train("At", d, 3, eta, seed, N=128, B=B, rho0=0.01, max_steps=1_000_000, log_every=100, stop_m=0.5, init="fixed", m0_scale=1.0, rng_tag=1000, fast=True)
    return dict(diag=tag, d=d, eta=eta, B=B, seed=seed, steps=r["steps"], T05=r["T05"], reached05=bool(r["reached05"]), m0=r["m0"], rho_at_T05=r.get("gamma_at_T05"), final_abs_m=abs(r["final_m"]), cpu_s=time.process_time() - c0)
if __name__ == "__main__":
    out = "results/exp9/diag/runs.csv"
    with ProcessPoolExecutor(2) as ex, open(out, "w", newline="") as f:
        w = None
        for row in ex.map(run, JOBS):
            if w is None: w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader()
            w.writerow(row); f.flush(); print(row, flush=True)
