# Spec: experiment 10 — the three regimes at d=128 (backlog N2)

Pre-registration: `docs/preregistration.md`, section "Three regimes at a second d (exp 10)" (P23a–d). ODE predictions: `scripts/ode_exp10.py` →
`results/exp10/ode_predictions.txt` (fixed before the run).

## Code
- `icl_additive/sweep10.py` modelled on `icl_additive/sweep6.py` / `sweep7.py` (resumable CSV `results/exp10/runs.csv`, per-run `cpu_s`, global CPU cap,
  `OMP_NUM_THREADS=1`, trajectories `results/exp10/traj/*.npz`, stdout `results/exp10/stdout.txt`). `train.train(model, 128, 2, eta=1/128**2, seed, ...)`
  with `init="fixed"`, `m0_scale=1.0`, `rng_tag=1000`, `stop_m=0.5`, `log_every=100`, `fast=True`, `B=64`.
- Cells: `a`: model `"A"`, `gamma=1.0`, N∈{16,64,256}, seeds {0,1}, `max_steps=1_000_000`; `b`: `"A"`, `gamma=0.1`, N∈{16,64,256}, seeds {0,1,2},
  `max_steps=2_500_000`; `c`: `"At"`, `rho0=0.01`, N∈{16,64,256}, seeds {0,1,2}, `max_steps=1_000_000`; `d`: `"A"`, `gamma=0.01, train_gamma=True,
  eta_gamma_mult=1.0`, N=256, seeds {0,1}, `max_steps=12_000_000`. 23 runs. Launch order: c (short), b N=64/256, a, b N=16, d last.
- `icl_additive/analyze10.py`: per cell/N median steps (censored = +inf), ODE steps, ratio; tied N-ratio; free secant exponent over d against exp 6
  (`results/exp6`: free N=256 median at d=64, and the ODE 50.1 flow); `results/exp10/summary.csv`, `results/exp10/figs/regimes_d128.png`.
- `results/exp10/README.md` in the exp 7/9 format (header block: Pre-registration, Code, CPU time, Deviations), tables, verdict lines for P23a–d
  quoting the criteria, CPU time, deviations.

## Budget and scheduling
Cap 5 CPU-h; the sweep stops launching at the cap (the free seeds are the ones to drop). Start with 2 workers while exp 8 is running
(`pgrep -f transformer_exp8` non-empty); when exp 8 has finished, further launches may use up to 4 workers.
