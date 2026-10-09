# Spec: experiment 9 — k*=3 sanity run (backlog N3)

Pre-registration: `docs/preregistration.md`, section "k*=3 sanity run (exp 9)" (P22a–c). ODE predictions: `scripts/ode_k3.py` →
`results/exp9/ode_predictions.csv` (fixed before the run).

## Code
- `icl_additive/sweep9.py`, modelled on `icl_additive/sweep7.py` (resumable CSV `results/exp9/runs.csv`, per-run `cpu_s`, global CPU cap,
  `OMP_NUM_THREADS=1`, trajectories in `results/exp9/traj/*.npz`, stdout in `results/exp9/stdout.txt`). Calls `train.train(model, d, 3, eta, seed, ...)`
  with `init="fixed"`, `m0_scale=1.0`, `rng_tag=1000`, `stop_m=0.5`, `log_every=100`, `fast=True`, `N`, `B=64`, `max_steps=1_000_000`.
- Cells: `a`: model `"At"`, `rho0=0.01`, `N=128`; `b`: `"At"`, `rho0=0.01`, `N=16`; `c`: model `"A"`, `gamma=0.2`, `train_gamma=False`, `N=128`.
  `d ∈ {8, 16, 32}`, `eta = 2e-4` everywhere, seeds `{0,1,2}`. 27 runs. Order: all short cells first (a, b d≤16, c d=8), then the capped ones.
- `icl_additive/analyze9.py`: per cell/d median steps (censored = +inf), ODE steps, ratio, secant `κ_eff` between d-pairs (medians), N=16/N=128
  ratios, `ρ` or `Γ` at `T_0.5`, censoring table; writes `results/exp9/summary.csv`, `results/exp9/ratios.csv`, `results/exp9/figs/k3.png`
  (log T vs log m0 per cell with ODE lines).
- `results/exp9/README.md` in the exp 7 format: setup, **Deviations (read first)**, tables, P22a–c verdicts, CPU time.

## Budget
Cap 2 CPU-h total (the sweep stops launching new runs at the cap; report any overrun). Expected ≈ 1.2 CPU-h (9 capped runs × ≤ 400 s + short runs).
Run with 2 workers while exp 8 occupies the other two cores.
