# Experiment 1 results: Model A (in-context) vs Model B (in-weight)

Everything here was produced by the code in `icl_additive/` (pure numpy, float64, analytic gradients; see
"Deviations from the spec"). All numbers below are generated from the CSVs in this directory
(`drift.csv`, `drift_slopes.csv`, `summary.csv`, `T_by_config.csv`, `slopes.csv`, `slopes_dge16.csv`,
`per_seed_regression.csv`). Figures: `figs/T_vs_d.png`, `figs/drift_vs_m.png`, `figs/traj_examples.png`.

Setup actually run: N=128, B=32 prompts/step (drift check: B=64), gamma=0.1 fixed, a=1 fixed, eta = eta_0/d^2,
eta_0 in {2, 1} for everything, plus eta_0=0.5 for B (k=1,2), A k=1 and A k=2 with d<=48. 5 seeds per config,
max_steps = 3e6 (spec said 2e6 in one place; 3e6 was requested), early stop at |m|>=0.9 (see below).
Loss is the plain mean square error (no factor 1/2).

## 1. Drift check (Monte Carlo vs closed-form population loss)

Model-A drift and Model-B drift at overlap m, d=16, N=128, gamma=0.1, a=1, `drift_mc` = -E[grad L].t / sqrt(1-m^2)
(t = unit tangent toward v), averaged over 50000 fresh batches of B=64 (a first gate run with 2000 batches gave
ratios MC/formula of 0.999-1.02 at m=0.2 and 0.4 for A k=1,2 and B k=1,2, so the sweep was started; the table
below is the later, higher-precision run). `drift_formula` = -dL/dm from the closed forms in `drift.py`
(central finite difference). z = (MC - formula)/SE.

| model | k | m | drift_mc | SE | drift_formula | MC/formula | z |
|---|---|---|---|---|---|---|---|
| A | 1 | 0.05 | 0.01888 | 0.0002 | 0.01899 | 0.994 | -0.57 |
| A | 1 | 0.1 | 0.03797 | 0.0002 | 0.03798 | 1 | -0.06 |
| A | 1 | 0.2 | 0.07595 | 0.00022 | 0.07597 | 1 | -0.10 |
| A | 1 | 0.4 | 0.1519 | 0.00029 | 0.1519 | 1 | -0.23 |
| A | 2 | 0.05 | 6.769e-05 | 8.2e-05 | 3.231e-05 | 2.1 | 0.43 |
| A | 2 | 0.1 | 0.0007304 | 9.6e-05 | 0.0006334 | 1.15 | 1.01 |
| A | 2 | 0.2 | 0.005658 | 0.00014 | 0.005818 | 0.973 | -1.13 |
| A | 2 | 0.4 | 0.04787 | 0.00036 | 0.04804 | 0.997 | -0.46 |
| A | 3 | 0.05 | -1.639e-05 | 0.00012 | -0.0001424 | 0.115 | 1.06 |
| A | 3 | 0.1 | -3.304e-05 | 0.00011 | -0.0002868 | 0.115 | 2.25 |
| A | 3 | 0.2 | -0.0003021 | 0.00026 | -0.0003355 | 0.9 | 0.13 |
| A | 3 | 0.4 | 0.009587 | 0.00064 | 0.009377 | 1.02 | 0.33 |
| B | 1 | 0.05 | 2 | 0.0019 | 2 | 1 | 0.18 |
| B | 1 | 0.1 | 2.001 | 0.0019 | 2 | 1 | 0.67 |
| B | 1 | 0.2 | 1.996 | 0.0018 | 2 | 0.998 | -2.27 |
| B | 1 | 0.4 | 2.002 | 0.0017 | 2 | 1 | 0.97 |
| B | 2 | 0.05 | 0.2004 | 0.0039 | 0.2 | 1 | 0.10 |
| B | 2 | 0.1 | 0.3998 | 0.0039 | 0.4 | 0.999 | -0.06 |
| B | 2 | 0.2 | 0.8001 | 0.004 | 0.8 | 1 | 0.02 |
| B | 2 | 0.4 | 1.597 | 0.0043 | 1.6 | 0.998 | -0.64 |
| B | 3 | 0.05 | 0.0175 | 0.012 | 0.015 | 1.17 | 0.22 |
| B | 3 | 0.1 | 0.04859 | 0.011 | 0.06 | 0.81 | -1.00 |
| B | 3 | 0.2 | 0.2387 | 0.011 | 0.24 | 0.994 | -0.12 |
| B | 3 | 0.4 | 0.9605 | 0.012 | 0.96 | 1 | 0.04 |

All 24 points are within 2.3 SE of the closed form (largest |z| = 2.27, B k=1 m=0.2; two points beyond 2 SE
of 24 is not unusual). Points where the formula drift is below the MC noise (A k=2 at m=0.05, A k=3 at m<=0.2) are not
resolved; there the ratio column is meaningless and only z is informative.

Log-log slope of drift vs m over m in {0.05, 0.1, 0.2, 0.4}:

| model | k | slope MC (+-SE) | slope of closed form | predicted (k-1 for B, 2k-1 for A) |
|---|---|---|---|---|
| A | 1 | 1 (+-0.0014) | 1 | 1 |
| A | 2 | 3.14 (+-0.07) | 3.48 | 3 |
| A | 3 |  (+-) |  | 5 |
| B | 1 | -0.000101 (+-0.0011) | -8.01e-14 | 0 |
| B | 2 | 0.999 (+-0.00059) | 1 | 1 |
| B | 3 | 1.96 (+-0.12) | 2 | 2 |

Comments (factual):
- Model B matches the predicted slopes (0, 1, 2 for k=1,2,3).
- Model A k=1 matches (slope 1.00). Model A k=2: MC slope 3.14+-0.07 but the closed form itself has slope 3.48
  over this m-range, because the finite-context repulsive term gamma^2 V'(m)/N (about 1.25e-3 m for k=2,
  gamma=0.1, N=128) is not negligible against the attractive 8 gamma m^3 at m<=0.1 (they are equal at
  m ~ 0.04). The asymptotic cubic drift is only recovered for m >~ 0.2.
- Model A k=3: the closed-form drift is **negative** (repulsive) for m below about 0.25 with gamma=0.1, N=128, and
  its MC estimate is consistent with that (3 of 4 MC means <= 0, so the slope fit is undefined). This was not
  swept.

## 2. Emergence time T_0.5 vs d: fitted slopes

Fit of log(median T_0.5 over 5 seeds) on log d, one fit per (model, k, eta_0), `scipy.stats.linregress`
standard error (this SE ignores the sampling scatter of the 5-seed median and the points are not independent
noise draws, so treat it as optimistic). Unreached runs (not reaching m=0.5 in 3e6 steps) count as T=+inf; if
the median is censored the point is dropped from the fit (see "what did not run"). "Predicted" is the spec's
`kappa/2 + 1` exponent.

### All d in the grid

| model | k | eta_0 | d range (points) | slope +- SE | predicted |
|---|---|---|---|---|---|
| B | 1 | 2 | 8-128 (5) | 2.38 +- 0.23 | d^1.5 (1.5) |
| B | 1 | 1 | 8-128 (5) | 2.46 +- 0.28 | d^1.5 (1.5) |
| B | 1 | 0.5 | 8-128 (5) | 2.46 +- 0.28 | d^1.5 (1.5) |
| B | 2 | 2 | 8-128 (5) | 2.40 +- 0.13 | d^2 polylog (2.1-2.3) |
| B | 2 | 1 | 8-128 (5) | 2.49 +- 0.15 | d^2 polylog (2.1-2.3) |
| B | 2 | 0.5 | 8-128 (5) | 2.40 +- 0.12 | d^2 polylog (2.1-2.3) |
| A | 1 | 2 | 8-128 (5) | 2.82 +- 0.26 | d^2 polylog (2) |
| A | 1 | 1 | 8-128 (5) | 2.77 +- 0.24 | d^2 polylog (2) |
| A | 1 | 0.5 | 8-128 (5) | 2.77 +- 0.24 | d^2 polylog (2) |
| A | 2 | 2 | 8-96 (7) | 3.44 +- 0.51 | d^3 (3) |
| A | 2 | 1 | 8-96 (7) | 3.35 +- 0.44 | d^3 (3) |
| A | 2 | 0.5 | 8-48 (5) | 3.11 +- 0.05 | d^3 (3) |

### d >= 16 only (drops d=8, where T_0.5 is 1-40 steps and integer resolution matters)

| model | k | eta_0 | d range (points) | slope +- SE | predicted |
|---|---|---|---|---|---|
| B | 1 | 2 | 16-128 (4) | 1.98 +- 0.06 | d^1.5 (1.5) |
| B | 1 | 1 | 16-128 (4) | 1.97 +- 0.06 | d^1.5 (1.5) |
| B | 1 | 0.5 | 16-128 (4) | 1.97 +- 0.06 | d^1.5 (1.5) |
| B | 2 | 2 | 16-128 (4) | 2.33 +- 0.22 | d^2 polylog (2.1-2.3) |
| B | 2 | 1 | 16-128 (4) | 2.30 +- 0.18 | d^2 polylog (2.1-2.3) |
| B | 2 | 0.5 | 16-128 (4) | 2.31 +- 0.18 | d^2 polylog (2.1-2.3) |
| A | 1 | 2 | 16-128 (4) | 2.71 +- 0.43 | d^2 polylog (2) |
| A | 1 | 1 | 16-128 (4) | 2.69 +- 0.41 | d^2 polylog (2) |
| A | 1 | 0.5 | 16-128 (4) | 2.68 +- 0.41 | d^2 polylog (2) |
| A | 2 | 2 | 16-96 (6) | 3.56 +- 0.80 | d^3 (3) |
| A | 2 | 1 | 16-96 (6) | 3.47 +- 0.70 | d^3 (3) |
| A | 2 | 0.5 | 16-48 (4) | 3.15 +- 0.11 | d^3 (3) |

### Measured vs predicted (plain statement)

- **B, k=1 (predicted 1.5): not met.** Measured 1.97+-0.06 (d>=16; 2.4+-0.3 if d=8 is included). The data are
  consistent with d^2, i.e. T*eta = const. Checking the population ODE for this case, dm/dt = eta*(-dL/dm)(1-m^2)
  with -dL/dm = 2 (constant, see section 1) gives T_0.5 = atanh(0.5)/(2 eta) ~ 0.27 d^2/eta_0 which matches the measured
  T_0.5*eta ~ 0.22-0.25 (see `T_by_config.csv`). So the exponent 1.5 in the spec table is not what these runs show.
- **B, k=2 (predicted 2.1-2.3): 2.30-2.33+-0.2 for d>=16 (within), 2.40-2.49+-0.15 over all d (slightly above).**
- **A, k=1 (predicted 2 polylog): 2.7+-0.25 (all d), 2.7+-0.4 (d>=16): above prediction by ~3 SE (all d) / ~1.7 SE (d>=16).** Medians
  at d=32 and d=64 are not monotone-smooth (d=32: 3.2e3, d=64: 7.3e3, d=128: 5.8e4 for eta_0=2), which is m_0
  scatter with 5 seeds (m_0 = |N(0,1/d)| differs per seed; see below).
- **A, k=2 (predicted 3): 3.35+-0.44 / 3.44+-0.51 (eta_0=1/2, all d up to 96), and 3.15+-0.11 (eta_0=0.5, d=8-48,
  no censored runs).** Consistent with 3 within errors. These fits **exclude d=128** (median censored at 3e6 steps:
  only 1 of 5 (eta_0=1) / 2 of 5 (eta_0=2) seeds escaped). The fitted lines predict ~3.2e6 (eta_0=1) and
  ~1.9e6 (eta_0=2) at d=128, while the observed medians are >3e6, so including d=128 as a lower bound would
  make the slope steeper, not shallower. At d=64 and d=96, 2 of 5 seeds are censored, so the plotted median is
  the largest uncensored time. The uncertainty is dominated by the m_0 scatter (next section).
- **Headline A vs B at k=2** (same eta_0=1, all d): 3.35+-0.44 vs 2.49+-0.15, difference ~0.9 +- 0.5 (predicted
  ~0.8-1.0); with d in [16,48] only, A = 3.15-3.22 (SE 0.11-0.17, all three eta_0) against B 2.30+-0.18 (d>=16 fit
  incl. d=64,128). The direction agrees with the prediction, but the 5-seed medians do not separate the exponents
  by much more than 2 sigma individually; the absolute times are clearly separated: median T_A/T_B is about 50 at d=16 and 65 at d=32 (same eta_0).
  Model B k=2 and Model A k=2 also differ in constants, not only in exponent.

### Supplementary (not in the spec): dependence on the initial overlap m_0

T for Model A depends strongly on the individual m_0 = |<w_0, v>| (a random O(d^-1/2) number), so medians of
5 seeds are noisy. OLS of log(T_0.5 eta_0) on [1, log d, log|m_0|] over the individual runs that reached m=0.5
(d>=16; censored runs dropped, which biases toward fast runs):

| model | k | eta_0 | runs | coef log d (SE) | coef log|m_0| (SE) |
|---|---|---|---|---|---|
| B | 1 | 2 | 20 | 2.11 (0.05) | -0.15 (0.03) |
| B | 1 | 1 | 20 | 2.11 (0.05) | -0.15 (0.03) |
| B | 1 | 0.5 | 20 | 2.12 (0.06) | -0.16 (0.03) |
| B | 2 | 2 | 20 | 2.32 (0.13) | -0.48 (0.08) |
| B | 2 | 1 | 20 | 2.30 (0.12) | -0.50 (0.08) |
| B | 2 | 0.5 | 20 | 2.27 (0.11) | -0.54 (0.07) |
| A | 1 | 2 | 20 | 2.42 (0.20) | -0.42 (0.09) |
| A | 1 | 1 | 20 | 2.42 (0.20) | -0.43 (0.09) |
| A | 1 | 0.5 | 20 | 2.43 (0.19) | -0.44 (0.08) |
| A | 2 | 2 | 27 | 2.38 (0.13) | -2.01 (0.10) |
| A | 2 | 1 | 25 | 2.13 (0.10) | -2.47 (0.09) |
| A | 2 | 0.5 | 18 | 2.02 (0.17) | -2.64 (0.09) |

For A k=2 the fitted coefficient of log|m_0| is -2.0 to -2.6 (drift-only estimate: T ~ 1/(eta m_0^2)), and the
coefficient of log d at fixed m_0 is ~2.0-2.4; with m_0 ~ d^-1/2 this reproduces d^3 (2 + 1). For B k=1 the m_0
dependence is weak (-0.15) and the d coefficient is 2.1.

## 3. Invariance of T_0.5 * eta under eta_0 (spec: >30 percent difference => also run eta_0=0.5)

Ratio of median(T_0.5 eta) at eta_0=1 to eta_0=2 (and eta_0=0.5 to 2 where run), per config:

| model | k | d | eta_0=1 / eta_0=2 | eta_0=0.5 / eta_0=2 |
|---|---|---|---|---|
| A | 1 | 8 | 1.2 | 1.15 |
| A | 1 | 16 | 1.04 | 1.06 |
| A | 1 | 32 | 0.989 | 0.996 |
| A | 1 | 64 | 1 | 0.999 |
| A | 1 | 128 | 0.997 | 1 |
| A | 2 | 8 | 1.05 | 0.977 |
| A | 2 | 16 | 1.01 | 1.08 |
| A | 2 | 24 | 0.988 | 0.985 |
| A | 2 | 32 | 0.963 | 0.935 |
| A | 2 | 48 | 1.03 | 1.02 |
| A | 2 | 64 | 0.681 |  |
| A | 2 | 96 | 0.98 |  |
| B | 1 | 8 | 0.75 | 0.75 |
| B | 1 | 16 | 1.02 | 1.04 |
| B | 1 | 32 | 1 | 0.991 |
| B | 1 | 64 | 1 | 1.01 |
| B | 1 | 128 | 1 | 1 |
| B | 2 | 8 | 0.7 | 0.95 |
| B | 2 | 16 | 1.06 | 1.03 |
| B | 2 | 32 | 0.979 | 0.973 |
| B | 2 | 64 | 0.944 | 0.921 |
| B | 2 | 128 | 1.02 | 1.02 |

Result: for d>=16 the ratios are within ~8 percent (0.92-1.08) except A k=2 d=64 (0.68; medians with 2 of 5
seeds censored) and d=128 (undefined). At d=8, where T_0.5 is only 1-50 steps (integer resolution), the ratios are
0.70-1.2 (B: 0.70-0.75, A k=1: 1.2). Per-seed, the
median over (model,k,d,seed) of T_0.5(eta_0=1) eta/(T_0.5(eta_0=2) eta) is 0.99-1.00 for all four (model,k)
(identical seeds => identical v, w_0; runs differ only in the noise realisation scaled by eta). So we are in the
drift-dominated regime at the tested sizes. The eta_0=0.5 runs were nonetheless done (cheap) for all configs
except A k=2 with d>=64 (too expensive).

## 4. Timing

- Sum of per-run wall times in `summary.csv` (each run uses one thread): **6.02 CPU-hours** over 330 runs
  (roughly 2.1 h of 4-core wall time over four sweep invocations: the first two were interrupted/restarted and
  their exact wall time was not recorded), plus ~0.35 CPU-h for the 50000-batch drift check (5.3 min wall), plus
  the CPU time of runs that were in flight when I interrupted the sweep (not recorded).
  **This is well over the ~2 CPU-h budget in the task.** Essentially all of it (4.9 of 6.0 h) is Model A k=2 with
  d>=48, dominated by 20 runs that hit the 3e6-step cap (~830 s each).
- Speed: Model A ~ 0.27 ms/step (projected sampler), Model B ~ 0.05 ms/step, independent of d. A full-input
  sampler (`--sampler full`) is ~4x slower for A at d=16 and does not give statistically different gradients
  (tested in `tests/test_basic.py::test_projected_sampler_matches_full`).

## 5. What did not run / censored / deviations

Censored runs (did not reach |m|=0.9 within 3e6 steps; `reached=False` in `summary.csv`):

| model | k | d | eta_0 | runs censored of 5 |
|---|---|---|---|---|
| A | 2 | 24 | 0.5 | 1 |
| A | 2 | 24 | 1 | 1 |
| A | 2 | 48 | 0.5 | 1 |
| A | 2 | 48 | 1 | 1 |
| A | 2 | 48 | 2 | 1 |
| A | 2 | 64 | 1 | 2 |
| A | 2 | 64 | 2 | 2 |
| A | 2 | 96 | 1 | 2 |
| A | 2 | 96 | 2 | 2 |
| A | 2 | 128 | 1 | 4 |
| A | 2 | 128 | 2 | 3 |

All censored runs are Model A k=2 and sit near m ~ 0 (final |m| < 0.1) with a small initial m_0 (< ~0.05); the
expected escape time 1/(1.6 eta m_0^2) exceeds 3e6 there. They are right-censored, not diverged.

- k=3 was **not swept** (spec: only if runtime allows; the budget was exceeded by k=2 Model A). Model B k=3 would
  be cheap, Model A k=3 has a repulsive drift at small m (section 1). Drift check for k=3 was done.
- Model A k=2 grid extended to d in {8,16,24,32,48,64,96,128} (spec: up to 48). I estimated d=128 at a
  few minutes per run from the d=16 timings (T ~ d^3), which ignored the 1/m_0^2 tail; the d=96 and d=128 jobs
  (scheduled first) completed before I could judge the cost, and the budget was exceeded. They are in
  `summary.csv`; the d=128 medians are censored and excluded from the fits, so effectively the fits use d<=96.
- Model A k=1 and Model B ran for d in {8,16,32,64,128} (no 24, 48).
- Stopping rule deviation: **the stopping criterion uses |m| when the model is invariant under w -> -w**
  (Model A for all k, Model B for even k), i.e. `m >= 0.9` is replaced by `|m| >= 0.9` and T_0.5 is the first time
  |m| >= 0.5 (column `sym_stop`). Reason: for these cases the student can escape to -v, which is the same
  solution (the first sweep pass had 4 such runs, e.g. A k=1 d=128 seed 1 and B k=2 d=32 seed 4, that went to
  m=-1 and were wrongly counted as unreached). The 4 affected rows were deleted and rerun with the corrected rule;
  all other rows were verified to be bit-identical under the corrected code (re-run of 6 random rows, and none of the
  reached runs ever visited m < -0.4). For Model B k=1 (not symmetric) m>=0.9 is used. Initialisation still has m_0>0.
- Not numpy-vs-torch: the code is pure numpy float64 with analytic gradients (torch not installable), checked by
  finite differences in `tests/test_basic.py`.
- Sampling: the sweep uses a distribution-exact projected sampler (data.py docstring): since the gradient is
  sum_j C_j x_j with C_j a function of (<v,x_j>, <w,x_j>), only these two projections are drawn per point and the
  d-2 orthogonal components of the gradient are drawn as one Gaussian with variance sum_j C_j^2. This is the
  same distribution as fresh full-dimensional data each step (tested), and the drift check uses full inputs.
- Spec says to log ||grad||: stored (running mean per log interval) in `traj/*.npz` (`gnorm`); `traj/*.npz`
  files are git-ignored.
- Seeds: `np.random.SeedSequence([seed, d, k, model])`; the same seed gives the same teacher, w_0 and data stream for the different eta_0.

## 6. Anomalies

- m_0 scatter: T for Model A is dominated by the random m_0 (T ~ m_0^-2 for k=2); at d=64 the median T_0.5
  (eta_0=2: 1.3e6, eta_0=1: 1.8e6 steps) is *larger* than at d=96 (2.4e5 / 4.8e5) because the five d=64 seeds
  happen to have smaller m_0. A k=2 medians are therefore not monotone in d.
- Sign symmetry: m(t) of Model A crosses zero and can end at -1 (seen in `figs/traj_examples.png`, A d=16 seed 3).
- Runs of Model A k=2 with m_0 < 0.03 sit at m ~ 0 for 10^6+ steps, diffusing; m(t) is non-monotone there.
- Model B k=1 d=8 has T_0.5 of 1-6 steps (single steps cross 0.5) so d=8 is excluded in the d>=16 fits.
- Sensitivity to eta_0: none beyond the T ~ 1/eta scaling, see section 3 (but eta_0=2 at d=8 means eta=0.03,
  which is quite large; no divergences observed).
