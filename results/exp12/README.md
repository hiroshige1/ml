# Experiment 12: which ingredient breaks the mechanism, optimiser or softmax? (single-feature model, sigma_2)

Pre-registration: `docs/preregistration.md`, section "Which ingredient breaks the mechanism: optimiser or softmax? (exp 12)" (P25a-d; committed in 7a48255 before any run).
Spec: `docs/spec_exp12.md`. Code: `icl_additive/ablation_exp12.py` (torch fp32, CPU, 1 thread per process, 4 workers with a claim-file job queue), analysis `icl_additive/analyze_exp12.py`.
Raw rows: `runs.csv` (24 runs; extra columns `max_dm_step` = largest single-step |Delta m|, `lr_adam`), per-run traces `traces/*.npz` (m every 100 steps; gitignored), logs `logs/worker*.log`,
`sanity.json`, `sampler_check.json`, `table.md`, `verdicts.json`.

## Setup (as pre-registered)

d = 64, sigma_2, `m_0 = d^-1/2 = 0.125` exactly (`init_fixed` semantics), pinned readout Gamma = 1, `x ~ N(0,I)`, `c ~ N(0,1)` per prompt, seeds {0, 1}.
`linear`: `yhat = sigma(w.x_q) mean_i y_i sigma(w.x_i)`; `softmax`: `yhat = sum_i softmax_i(sigma(w.x_i) sigma(w.x_q)) y_i` (temperature 1). Loss = batch mean of `(yhat - y_q)^2`; `w <- w/|w|` after every step.
`sgd`: eta = 1/d^2 = 1/4096, no momentum, cap 3e5 steps. `adam`: `torch.optim.Adam(lr=1e-3)`, default betas, cap 1e5 steps. Stop at `|m| >= 0.5`; `m` logged every 100 steps (T_0.5 is checked every step).
Cells: {linear, softmax} x {sgd, adam} x (N,B) in {(16,64), (16,1024), (1024,64)} x 2 seeds = 24 runs, N=16 cells launched first.

## Deviations (read first)

- **SGD cells use the distribution-exact projected sampler, not full d=64 data (not in the spec; done for budget).** With full data the stuck SGD cells (N=16,B=1024; 4 of them for 3e5 steps) would have cost
  about 3 CPU-h alone (7 ms/step measured). For SGD the update is rotation-equivariant, so `x = p v + u e2 + xi` (as in `icl_additive/data.py` / `train.py`; `<v,x> = p`, `<w,x> = m p + sqrt(1-m^2) u`)
  gives the same law of the gradient: `dL/dw = sum_j C_j x_j` with `C_j = dL/dz_j`, whose complement part is `N(0, sum C_j^2 I)` exactly (`proj_sgd_grad`). This also holds for softmax (the loss depends on `x` only through `z`, `p`, `c`).
  **Adam cells use full d=64 data** (Adam is not rotation-invariant, so the coordinate system matters). Check (`sampler_check.json`, one SGD step from `m=0.3`, 20000 reps at N=16 and 6666 at N=1024, full vs projected):
  means agree (|z| <= 1.7 in all four cells), IQR ratio proj/full 0.999 / 1.019 / 1.029 / 1.005 (linear N=16, linear N=1024, softmax N=16, softmax N=1024), 99th percentile of |Delta m| ratio 1.05 / 1.09 / 1.04 / 1.06.
  (Sample std ratios 1.28 / 1.12 / 1.00 / 1.00; the linear ones are heavy-tailed, so the IQR is the stable statistic.)
- **CPU 3.10 h vs the 3 h cap: overran by 0.10 h (3%).** The two softmax+Adam N=1024,B=64 runs alone took 2927 s and 2466 s (full data, 25-30 ms/step, stuck for 1e5 steps). The sanity and sampler checks (about 0.1 CPU-h) are not counted. Per group (CPU-h): Adam N=16,B=64 0.10; Adam N=16,B=1024 0.81; Adam N=1024,B=64 1.51; SGD 0.12 / 0.24 / 0.32.
- **Declared extra (Adam lr 1e-4 at N=16, B=1024) was not run.** The spec's trigger is "Adam at lr 1e-3 unstable (|m| oscillating with large steps)". It is not: the largest single-step |Delta m| is 0.005-0.011 (SGD at B=64 reaches 0.03-0.07 from gradient noise), and `m` fluctuates around 0 with std 0.01-0.03 (mean |m| over the second half of the trace 0.010-0.051 at N=16 for linear+Adam, 0.014-0.028 for softmax+Adam). So lr 1e-3 behaves as a noisy-but-stable optimiser here.
- Float32 for training (the sanity check below is done in both fp64 and fp32). Data streams: `torch.Generator` seeded from `SeedSequence([seed, 12, model, opt, N, B])`; teacher `v` and `w_0` from numpy `init_fixed` seeded by `[seed, 64, 12]` (same `v`, `w_0` for all cells of a seed). `runs.csv` has no `lr_adam` for SGD (NaN).
- The first 300-step smoke test and the sanity/sampler checks are not part of the 24 runs.

## Sanity check (before the sweep)

Torch `linear` model (Gamma = 1, sigma_2) vs `icl_additive.models.ModelA(w, 2, gamma=1.0).loss` on the same batches (5 seeds x (N,B) in {(16,64), (1024,64)}, `w = w_0`):

| precision | max relative difference, batch loss | max relative difference, gradient (`loss_grad`, 2-norm) |
|---|---|---|
| fp64 | 3.5e-16 | 1.5e-15 |
| **fp32 (the precision used in the sweep)** | **2.0e-7** | 2.9e-6 |

Required: relative loss difference < 1e-5. Met.

## Table: per cell (2 seeds each)

CENS = no `|m| >= 0.5` within the step cap. `max |m|` includes the initial value 0.125. `T_0.5` and `steps` are optimiser steps.

| model | opt | N | B | steps (seed0 / seed1) | T_0.5 (seed0 / seed1) | reached | final \|m\| | max \|m\| | max step \|dm\| | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | adam | 16 | 64 | 100000 / 100000 | CENS / CENS | N/N | 0.083 / 0.042 | 0.127 / 0.125 | 0.0062 / 0.0070 | 87 / 85 |
| linear | adam | 16 | 1024 | 100000 / 100000 | CENS / CENS | N/N | 0.034 / 0.009 | 0.126 / 0.125 | 0.0093 / 0.0072 | 726 / 720 |
| linear | adam | 1024 | 64 | 379 / 1421 | 379 / 1421 | Y/Y | 0.500 / 0.505 | 0.500 / 0.505 | 0.0043 / 0.0081 | 11 / 41 |
| softmax | adam | 16 | 64 | 100000 / 100000 | CENS / CENS | N/N | 0.018 / 0.040 | 0.125 / 0.126 | 0.0075 / 0.0066 | 89 / 91 |
| softmax | adam | 16 | 1024 | 100000 / 100000 | CENS / CENS | N/N | 0.014 / 0.013 | 0.125 / 0.125 | 0.0062 / 0.0054 | 731 / 740 |
| softmax | adam | 1024 | 64 | 100000 / 100000 | CENS / CENS | N/N | 0.021 / 0.012 | 0.125 / 0.125 | 0.0110 / 0.0079 | 2927 / 2466 |
| linear | sgd | 16 | 64 | 300000 / 300000 | CENS / CENS | N/N | 0.008 / 0.022 | 0.125 / 0.125 | 0.0373 / 0.0440 | 104 / 105 |
| linear | sgd | 16 | 1024 | 300000 / 300000 | CENS / CENS | N/N | 0.002 / 0.009 | 0.125 / 0.125 | 0.0147 / 0.0115 | 203 / 208 |
| linear | sgd | 1024 | 64 | 34517 / 37356 | 34517 / 37356 | Y/Y | 0.500 / 0.500 | 0.500 / 0.500 | 0.0031 / 0.0032 | 55 / 58 |
| softmax | sgd | 16 | 64 | 300000 / 300000 | CENS / CENS | N/N | 0.007 / 0.014 | 0.126 / 0.125 | 0.0247 / 0.0289 | 111 / 113 |
| softmax | sgd | 16 | 1024 | 300000 / 300000 | CENS / CENS | N/N | 0.001 / 0.000 | 0.125 / 0.125 | 0.0094 / 0.0062 | 226 / 226 |
| softmax | sgd | 1024 | 64 | 300000 / 300000 | CENS / CENS | N/N | 0.016 / 0.008 | 0.125 / 0.125 | 0.0684 / 0.0630 | 520 / 513 |

Where the trap holds, `|m|` falls from `m_0 = 0.125` towards 0 within the first ~10% of the run and stays there (final |m| 0.000-0.083); the only runs whose `|m|` ever exceeds 0.125 are the four escaping ones.

Escape pattern: **escapes** (>= 1 of 2 seeds, the pre-registered reading rule) in linear+SGD N=1024 (T_0.5 = 34517 / 37356 steps, flow-like) and linear+Adam N=1024 (379 / 1421 steps). **Nowhere else**: neither seed escapes in any of the other 10 cells, including all four N=16 cells of each of linear+Adam, softmax+SGD and softmax+Adam, and softmax at N=1024 under both optimisers.

## Verdicts

Reading rule (fixed in the pre-registration): a cell "escapes" if >= 1 of 2 seeds reaches `|m| >= 0.5` within the step cap.

**P25a (linear + SGD reference). HELD.** Criterion (quoted): "`N=16` stuck at both `B` (0/4), `N=1024` escapes 2/2 (E2 / P20f replicated)". Measured: N=16 B=64 0/2, B=1024 0/2 (0/4 total); N=1024 B=64 2/2 (T_0.5 = 34517, 37356). The reference reproduces E2/P20f in the torch implementation.

**P25b (linear + Adam). HELD.** Criterion (quoted): "**same pattern**: Adam rescales coordinates but cannot reverse the sign of the mean drift below `m*`; `N=16` stuck at `B=64` and `B=1024`, `N=1024` escapes"; default "`B=1024` rescues `N=16` under Adam => the optimiser is the ingredient". Measured: N=16 B=64 0/2, B=1024 0/2; N=1024 2/2 (T_0.5 = 379, 1421 steps). `B=1024` does not rescue N=16 under Adam; the default is not triggered. (Adam is ~25-90x faster in steps at N=1024, as expected from its per-step size of ~lr, but this is not part of the criterion.)

**P25c (softmax + SGD). FAILED as pre-registered, on the N=1024 leg; the N=16 leg held.** Criterion (quoted): "same pattern (the finite-`N` variance of a softmax-weighted statistic still grows with alignment)"; default "`N=16` escapes at `B=64` or `B=1024` => softmax is the ingredient". Measured: N=16 B=64 0/2, B=1024 0/2 (default not triggered: softmax does not rescue N=16); but **N=1024 B=64 escapes 0/2** (final |m| 0.016 / 0.008, both below m_0, no excursion above 0.117 in the trace), whereas "same pattern" requires 2/2. So the pre-registered statement that softmax leaves the pattern unchanged is wrong at N=1024: softmax + SGD is trapped at both context lengths in this model.

**P25d (softmax + Adam). FAILED as pre-registered, on the N=1024 leg; the N=16 leg held.** Criterion (quoted): "same pattern". Measured: N=16 B=64 0/2, B=1024 0/2; N=1024 B=64 0/2 (final |m| 0.021 / 0.012, max in trace 0.108 / 0.092). Same failure as P25c.

Reading-rule consequence (pre-registered): P25b holds, so the transformer's `B`-rescue is **not** attributed to Adam; P25c's decisive default (softmax rescues N=16) is not triggered, so it is **not** attributed to softmax either. Neither Adam nor softmax breaks the mechanism for N=16 in the single-feature model: no combination of the two ingredients ({linear, softmax} x {SGD, Adam}) escapes at N=16, at either B. Per the pre-registration, the difference then lies in depth / LayerNorm / the residual stream (untested here). No claim about the transformer itself follows. In addition (unpredicted): softmax removes the N=1024 escape that the linear model has; the N-dependence of the escape is therefore not preserved by softmax at temperature 1.

Hypothesis for the softmax N=1024 trap (post hoc, not tested here): `sigma_2` logits `sigma(w.x_i) sigma(w.x_q)` are heavy-tailed (chi-square-like), so for `sigma(w.x_q)` of order 1 the softmax concentrates on a few context positions and the effective context length stays small however large `N` is. This is not checked (no effective-N measurement was made) and is not used in any verdict.

## Summary

Held: P25a, P25b, and the N=16 legs of P25c and P25d. Failed as pre-registered: P25c and P25d (softmax is also trapped at N=1024 under both optimisers). 24 runs, CPU 3.099 h (cap 3 h, overran by 3%). 4 of 24 runs escaped (all linear, all N=1024, B=64).
