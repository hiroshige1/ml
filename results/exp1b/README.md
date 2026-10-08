# Experiment 1b results: context length, fixed vs trainable vs tied readout (Model A, k = 2)

Spec: `docs/spec_exp1b.md` (final section "Revision after exp 1" is binding). Pre-registered expectations: `docs/preregistration.md`
P2, P3, P4, P4', P7. Code: `icl_additive/{models,train,sweep1b,analyze1b}.py`, tests in `tests/test_basic.py`
(`python tests/test_basic.py`, all pass). Raw per-run rows: `results/exp1b/{thr,p7,tied1,kappa,kappa2,bctl,dscan}.csv`
(merged: `summary.csv`; column `superseded` marks censored `kappa` rows that were re-run in `kappa2`);
derived tables: `threshold_table.csv, p7_table.csv, kappa_fits.csv, gamma_at_T05.csv, dscan_slopes.csv`;
trajectories (m, Gamma/rho, loss, |grad| every 500 steps): `results/exp1b/traj/*.npz`; stdout logs `run_*_stdout.txt`,
`analysis_stdout.txt`. Figures: `figs/threshold_grid.png`, `figs/kappa_vs_N.png`, `figs/P7_tradeoff.png`.
All numbers below are generated from those CSVs by `python -m icl_additive.analyze1b`.

Setup actually run: k = 2, M = P = 1, projected sampler, float64, online spherical SGD, `eta = 1/d^2` (eta_0 = 1 only),
init `--init fixed` with `m_0 = m0_scale * d^(-1/2)` exactly (`m0_scale = 1` except in the m_0 regression), 3 seeds per cell,
early stop when |m| >= 0.5 (not 0.9, see deviations). Model A-tied: unnormalised `w`, `Gamma = ||w||^2`, no renormalisation;
`rho_0 = ||w_0||^2 = 0.01` unless stated. Free Gamma: `Gamma_0 = 0.01`, plain SGD with lr `eta_Gamma = eta` or `10 eta`.
"Stuck" = not reaching |m| = 0.5 and final |m| < 2 m_0 after the step budget.

## 1. Priority 1: threshold tests (P2, P3), fixed Gamma, 3e5 steps, B = 32

Predicted escape iff `d < d* = N(4-2g)/(16g)` (equivalently `m_0^2 > 16g/(N(4-2g))`).

| gamma | N | d | d* = N(4-2g)/(16g) | predicted | escaped /3 | stuck /3 (final abs m < 2 m0) | median final abs m | median T_0.5 of escapers |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 32 | 16 | 76 | escape | 3 | 0 | 0.5004 | 2562 |
| 0.1 | 32 | 32 | 76 | escape | 3 | 0 | 0.5 | 2.356e+04 |
| 0.1 | 32 | 64 | 76 | escape | 3 | 0 | 0.5 | 2.367e+05 |
| 0.1 | 128 | 16 | 304 | escape | 3 | 0 | 0.5002 | 2451 |
| 0.1 | 128 | 32 | 304 | escape | 3 | 0 | 0.5 | 2.115e+04 |
| 0.1 | 128 | 64 | 304 | escape | 3 | 0 | 0.5 | 1.808e+05 |
| 0.1 | 512 | 16 | 1216 | escape | 3 | 0 | 0.5003 | 2292 |
| 0.1 | 512 | 32 | 1216 | escape | 3 | 0 | 0.5 | 2.073e+04 |
| 0.1 | 512 | 64 | 1216 | escape | 3 | 0 | 0.5 | 1.725e+05 |
| 1 | 32 | 16 | 4 | stuck | 2 | 1 | 0.5016 | 1.992e+05 |
| 1 | 32 | 32 | 4 | stuck | 0 | 3 | 0.03323 | nan |
| 1 | 32 | 64 | 4 | stuck | 0 | 3 | 0.002112 | nan |
| 1 | 128 | 16 | 16 | stuck | 3 | 0 | 0.5021 | 992 |
| 1 | 128 | 32 | 16 | stuck | 1 | 2 | 0.00931 | 1.652e+04 |
| 1 | 128 | 64 | 16 | stuck | 0 | 3 | 0.01017 | nan |
| 1 | 512 | 16 | 64 | escape | 3 | 0 | 0.501 | 396 |
| 1 | 512 | 32 | 64 | escape | 3 | 0 | 0.5001 | 5223 |
| 1 | 512 | 64 | 64 | stuck | 3 | 0 | 0.5001 | 5.048e+04 |

Pre-registered vs measured:

- **P2 (gamma = 1, N = 128): "escape for d <= 16, no escape for d >= 32": partly failed.** d = 16: 3/3 escape (met). d = 64: 0/3 (met; final
  |m| 0.007-0.015, i.e. pushed below m_0 = 0.125 toward 0). **d = 32: 1/3 escaped (T_0.5 = 1.65e4; the other two finished at |m| = 0.002 and 0.009),
  contrary to "no escape for d >= 32".** K2 (escape at d = 64) is not triggered. The d = 32 escape is the one case where SGD noise
  apparently beat the drift bias; it is a single seed, I did not investigate it further.
- **P3 (gamma = 0.1, N = 128): "escape for all d <= 256 in the sweep": met**, 27/27 escapes over all N and d for gamma = 0.1 (predicted
  d* = 76 / 304 / 1216 for N = 32 / 128 / 512). Note the d = 64, N = 32 cell (d just below d* = 76) is slower: median
  T_0.5 = 2.4e5 vs 1.7-1.8e5 at N = 128, 512 (budget 3e5).
- **Other cells of the same grid, not in the pre-registration, but the formula makes a prediction:** gamma = 1, N = 512: 9/9 escape, but d = 64 sits exactly on
  d* = 64 (3/3 escaped, T_0.5 = 5.0e4). **gamma = 1, N = 32 (d* = 4, predicted stuck at every d): d = 32, 64: 0/3 (met); d = 16: 2/3 escaped (not met)**, at
  T_0.5 = 1.05e5 and 2.9e5 (close to the 3e5 budget); the third seed stayed at |m| = 0.055 (max 0.37).
  The sharp "threshold" is therefore not sharp at the SGD level for N = 32: late escapes occur at d = 16, where the drift is repulsive by formula.
- Anomaly: one N = 32, d = 16, gamma = 1 run ends with final |m| = 0.685 although the run stops at the first step with |m| >= 0.5, i.e. a single SGD
  step moved |m| by >= 0.18. At N = 32 with gamma = 1 the per-step gradient is heavy-tailed; I did not examine it further.
- The threshold is therefore located roughly where predicted for N = 128 (d = 16 escapes, 64 does not, 32 is mixed), in the right direction
  for gamma and N throughout, but not an exact "hard" threshold. Seeds are few (3 per cell).

## 2. P7: N-T tradeoff (gamma = 1, N = 128, d = 64, m_0 = 0.125)

| B | steps | total_samples | escaped | stuck | exceeded_0p5 | final_abs_m_seeds | max_abs_m_median | cpu_s |
|---|---|---|---|---|---|---|---|---|
| 32 | 300000 | 9600000 | 0 | 3 | 0 | 0.015 / 0.010 / 0.007 | 0.1261 | 186.8 |
| 256 | 100000 | 25600000 | 0 | 3 | 0 | 0.005 / 0.005 / 0.007 | 0.1251 | 405.8 |
| 1024 | 30000 | 30720000 | 0 | 3 | 0 | 0.067 / 0.066 / 0.063 | 0.125 | 564.7 |

- **0/9 escapes at every B, final |m| (0.005-0.015 at B = 32 and 256; 0.063-0.067 at B = 1024) below m_0 = 0.125, max |m| over each run <= 0.1265 (m_0 = 0.125).**
  Matches the pre-registered "no escape at any B". B = 1024 ends higher (0.065 vs 0.01), which looks like less noise-driven diffusion around
  m = 0 (and 3e4 steps rather than 3e5-1e5 for the drift to push it down); I did not test this.
- **Deviation:** the spec's step counts (3e5, 1e5, 3e4 at B = 32, 256, 1024) give total prompts B*T = 9.6e6, 2.56e7, 3.07e7, i.e. ×1, ×2.7, ×3.2, **not**
  the "×1, ×3, ×10" the spec text states. I used the explicit step counts. The ×10 leg (B = 1024, ~9.4e4 steps, ~0.16 CPU-h per seed) was not run.
  So P7 is tested up to ×3.2 total samples, not ×10. (`figs/P7_tradeoff.png`)

## 3. Side check on P4': tied readout with rho_0 = 1 (N = 128, B = 32, 3e5 steps)

| d | escaped | stuck | median_final_abs_m | median_T05 | median_final_rho |
|---|---|---|---|---|---|
| 32 | 3 | 0 | 0.5008 | 7320 | 0.6357 |
| 64 | 0 | 3 | 0.01104 | nan | 0.1768 |

Pre-registered: "with rho_0 = 1, N = 128 still stuck for d >= 32 (like fixed gamma = 1)". Measured: **d = 32: 3/3 escaped (T_0.5 ~ 7e3, faster than
fixed gamma = 1, where 1/3 escaped at T_0.5 = 1.65e4, but slower than the rho_0 = 0.01 tied runs at N = 128, d = 32, m_0 = d^(-1/2), T_0.5 ~ 1.8e3); d = 64: 0/3 (stuck).**
In the stuck d = 64 runs rho decays from 1 to ~0.18; i.e. the tied norm moves quickly, unlike a fixed gamma. So the tied d = 32 prediction failed, the d = 64 one held.

## 4. Priority 2: kappa_eff from the m_0 regression (d = 32, eta = 1/1024)

Fit `log(T_0.5 * eta) = a + b log m_0` per (protocol, N) over the 4 multipliers {0.5, 0.7, 1.0, 1.4} x d^(-1/2) x 3 seeds (OLS on individual runs; SE
is the OLS standard error and ignores model misspecification, e.g. log corrections, so treat it as optimistic); `kappa_eff = 2 - b`.
Censored runs (not reaching |m| = 0.5 within the cap) are **excluded** from the fit; they are the slowest runs and are all at the smallest m_0
= 0.5 d^(-1/2) (small-m_0 / large-T end), so **excluding them flattens the fitted slope and biases kappa_eff downward**. The column
`kappa_if_censored_set_to_cap` fills censored runs with T = cap (a lower bound for each such T) and refits; it is not a bound on kappa
but it is only an indication: it is larger than the excluded-run fit in 2 of the 3 censored main cells (free1 N = 32: 5.92 vs 5.45; free10 N = 128: 6.54 vs 5.78) and not larger at free10 N = 32 (6.50 vs 6.61).

| protocol | N | B | max_steps | n_reached | n_censored | censored_by_m0 | b | b_se | kappa | kappa_if_censored_set_to_cap | med T0.5*eta by m0 multiplier (reached/3) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| free1 | 32 | 32 | 1000000 | 9 | 3 | 0.5:3 | -3.449 | 0.1202 | 5.449 | 5.916 | 0.5:nan(0/3); 0.7:175(3/3); 1:52.2(3/3); 1.4:16.4(3/3) |
| free1 | 128 | 32 | 1000000 | 12 | 0 | - | -3.591 | 0.1966 | 5.591 | nan | 0.5:565(3/3); 0.7:113(3/3); 1:36(3/3); 1.4:13.2(3/3) |
| free1 | 512 | 8 | 300000 | 12 | 0 | - | -2.81 | 0.1075 | 4.81 | nan | 0.5:254(3/3); 0.7:95.4(3/3); 1:33.1(3/3); 1.4:13.3(3/3) |
| free1 | 4096 | 8 | 300000 | 12 | 0 | - | -2.92 | 0.1032 | 4.92 | nan | 0.5:248(3/3); 0.7:93.8(3/3); 1:38.5(3/3); 1.4:11(3/3) |
| free10 | 32 | 32 | 1000000 | 9 | 3 | 0.5:3 | -4.612 | 0.2561 | 6.612 | 6.503 | 0.5:nan(0/3); 0.7:200(3/3); 1:39.9(3/3); 1.4:9.85(3/3) |
| free10 | 128 | 32 | 1000000 | 10 | 2 | 0.5:2 | -3.783 | 0.1141 | 5.783 | 6.535 | 0.5:nan(1/3); 0.7:81.6(3/3); 1:17.2(3/3); 1.4:5.58(3/3) |
| free10 | 512 | 8 | 300000 | 12 | 0 | - | -2.922 | 0.1558 | 4.922 | nan | 0.5:97.3(3/3); 0.7:48.7(3/3); 1:15.3(3/3); 1.4:5.5(3/3) |
| free10 | 4096 | 8 | 200000 | 12 | 0 | - | -3.173 | 0.2068 | 5.173 | nan | 0.5:128(3/3); 0.7:37.8(3/3); 1:15.8(3/3); 1.4:4.52(3/3) |
| free1_B8ctl | 128 | 8 | 1000000 | 11 | 1 | 0.5:1 | -3.378 | 0.2566 | 5.378 | 5.625 | 0.5:nan(2/3); 0.7:103(3/3); 1:35.9(3/3); 1.4:12.9(3/3) |
| tied | 32 | 32 | 1000000 | 12 | 0 | - | -2.101 | 0.3018 | 4.101 | nan | 0.5:11.3(3/3); 0.7:2.69(3/3); 1:1.43(3/3); 1.4:1.03(3/3) |
| tied | 128 | 32 | 1000000 | 12 | 0 | - | -2.259 | 0.1921 | 4.259 | nan | 0.5:9.24(3/3); 0.7:3.58(3/3); 1:1.73(3/3); 1.4:0.917(3/3) |
| tied | 512 | 8 | 1000000 | 12 | 0 | - | -2.3 | 0.2085 | 4.3 | nan | 0.5:10.2(3/3); 0.7:2.93(3/3); 1:2.1(3/3); 1.4:0.805(3/3) |
| tied | 4096 | 8 | 1000000 | 12 | 0 | - | -2.442 | 0.1755 | 4.442 | nan | 0.5:11(3/3); 0.7:4.07(3/3); 1:2.1(3/3); 1.4:0.848(3/3) |

B = 32 for N <= 128 and B = 8 for N >= 512 (cost; see deviations). Row `free1_B8ctl` is a control: same protocol as `free1` at N = 128 but B = 8.

Summary against the pre-registration (free kappa_eff rises from ~4 at N = 4096 toward ~8 at N = 32; tied ~4 at every N):

| protocol | N=4096 | N=512 | N=128 | N=32 | pre-registered |
|---|---|---|---|---|---|
| free Gamma, eta_Gamma = eta | 4.92 +- 0.10 | 4.81 +- 0.11 | 5.59 +- 0.20 | 5.45 +- 0.12 (3/12 censored; 5.92 if set to cap) | ~4 -> ~8 |
| free Gamma, eta_Gamma = 10 eta | 5.17 +- 0.21 | 4.92 +- 0.16 | 5.78 +- 0.11 (2/12 censored) | 6.61 +- 0.26 (3/12 censored; 6.50 if set to cap) | ~4 -> ~8 |
| tied, rho_0 = 0.01 | 4.44 +- 0.18 | 4.30 +- 0.21 | 4.26 +- 0.19 | 4.10 +- 0.30 | ~4 at every N |

- **Tied: met to within ~0.4.** kappa_eff = 4.1-4.4 at every N (three of four cells within 1.5 SE of 4; N = 4096: 4.44 +- 0.18 is 2.5 SE above 4), no trend in N (4.10 -> 4.44, differences < 1.2 SE).
- **Free Gamma: the direction of the predicted trend is present, the magnitude is not.** kappa_eff is larger at small N (N <= 128: 5.4-6.6) than at large N
  (N >= 512: 4.8-5.2), but the large-N end is ~5, not 4, and the small-N end never reaches 8 (the largest value, 6.6 +- 0.26, comes from a cell with 3/12 runs censored). 
  The pre-registered "N = 4096 -> ~4" anchor was not met for the free readout (4.92 +- 0.10 and 5.17 +- 0.21).
  Free kappa_eff exceeds tied kappa_eff at the same N by 0.5-0.7 (N >= 512), 1.3-1.5 (N = 128) and 1.4-2.5 (N = 32).
  The step from N = 512 to N = 4096 is flat (4.81 -> 4.92; 4.92 -> 5.17); the change happens between N = 128 and 512 (free1: 5.59 -> 4.81). 
  Unlike the 4 -> 8 "crossover" prediction, I see a modest, gradual rise of ~0.5-1.5 as N decreases.
  The two free protocols differ by < 1 SE except at N = 32 (6.61 vs 5.45, but both N = 32 cells lack the m_0 = 0.5 multiplier, fit on 0.7/1.0/1.4 only, 3/12 censored).
- The B control: free1, N = 128, B = 8 gives 5.38 +- 0.26 (1/12 censored) vs 5.59 +- 0.20 at B = 32, so B alone does not explain the N = 128 -> 512 drop (they differ by 0.2, < 1 SE).
- The fit is OLS over 3-4 points in log m_0 per cell. The median T_0.5*eta values are not exactly power laws in m_0 (e.g. free1, N = 4096: local slopes 2.9, 2.5, 3.7 between consecutive multipliers). kappa_eff therefore has
  systematic uncertainty beyond the quoted SE.
- Note T_0.5*eta for the tied runs is 0.8-11, vs 4.5-565 for free Gamma: tied escapes much sooner than free-Gamma at the same m_0 and eta.
  This is by itself a large effect of the parameterisation which is not about the exponent.

### Learned Gamma at T_0.5 (free-Gamma runs that reached 0.5, d = 32; `Gamma*(0.5;N)` = Wiener value from the spec)

| protocol | N | n | Gamma_at_T05_median | Gamma_at_T05_min | Gamma_at_T05_max | Gamma_star_0p5 |
|---|---|---|---|---|---|---|
| free1 | 32 | 9 | 0.1844 | 0.1751 | 0.1896 | 0.3765 |
| free1 | 128 | 24 | 0.2232 | 0.2078 | 0.2366 | 0.7072 |
| free1 | 512 | 12 | 0.2417 | 0.2182 | 0.2496 | 0.9062 |
| free1 | 2048 | 12 | 0.2423 | 0.2239 | 0.2541 | 0.9748 |
| free1 | 4096 | 12 | 0.2467 | 0.2286 | 0.2556 | 0.9872 |
| free10 | 32 | 9 | 0.3243 | 0.1996 | 0.3891 | 0.3765 |
| free10 | 128 | 10 | 0.5159 | 0.486 | 0.5588 | 0.7072 |
| free10 | 512 | 12 | 0.632 | 0.5164 | 0.7284 | 0.9062 |
| free10 | 4096 | 12 | 0.6985 | 0.5882 | 0.8808 | 0.9872 |
| free1_B8ctl | 128 | 11 | 0.2288 | 0.2095 | 0.2647 | 0.7072 |

Pre-registered P5 ("Gamma at T_0.5 ~ Gamma*(0.5;N), independent of d, increasing in N"): **not met in magnitude.** With eta_Gamma = eta, Gamma(T_0.5) = 0.18-0.25, a weak increase in N (0.18 -> 0.25) but far below Gamma* (0.38-0.99);
with eta_Gamma = 10 eta it is closer and increases more clearly with N (0.32 -> 0.70 vs Gamma* 0.38 -> 0.99), still below it for N >= 128.
(Gamma is logged every 500 steps; the value is taken at the log point nearest to T_0.5.) Independence of d is supported by the d-scan below (0.22-0.25 over d = 16-48).

## 5. Priority 3: T_0.5 vs d, free Gamma (eta_Gamma = eta), m_0 = d^(-1/2), eta = 1/d^2, d in {16, 24, 32, 48}

| N | d | T05_median | Gamma_at_T05 | eta | T05_x_eta |
|---|---|---|---|---|---|
| 128 | 16 | 3277 | 0.2214 | 0.003906 | 12.8 |
| 128 | 24 | 1.517e+04 | 0.2206 | 0.001736 | 26.34 |
| 128 | 32 | 3.688e+04 | 0.2232 | 0.0009766 | 36.02 |
| 128 | 48 | 1.561e+05 | 0.2236 | 0.000434 | 67.76 |
| 2048 | 16 | 2372 | 0.2261 | 0.003906 | 9.266 |
| 2048 | 24 | 1.448e+04 | 0.2339 | 0.001736 | 25.14 |
| 2048 | 32 | 3.224e+04 | 0.2467 | 0.0009766 | 31.48 |
| 2048 | 48 | 1.488e+05 | 0.2428 | 0.000434 | 64.6 |

| N | slope | se | Gamma_star_0p5 |
|---|---|---|---|
| 128 | 3.489 | 0.08409 | 0.7072 |
| 2048 | 3.704 | 0.2105 | 0.9748 |

Slope of log median T_0.5 on log d: **3.49 +- 0.08 (N = 128), 3.70 +- 0.21 (N = 2048)**. P4 predicted ~4 for d >> sqrt(N) (N = 128: sqrt(N) = 11) and ~2 for d << sqrt(N)
(N = 2048: sqrt(N) = 45, all d <= 48). The two slopes are indistinguishable (difference 0.2 +- 0.2) and T_0.5 itself is nearly the same at N = 128 and 2048
(T_0.5*eta within 13% at d >= 24, 28% at d = 16), i.e. no N-dependence of the d-scaling is seen, and **the pre-registered bend/N-shift (x16 in the crossover d for N = 128 -> 2048) is not observed in this range.**
Only 4 d values, 3 seeds, medians; and d >= 64, the region where N = 128 and 2048 should differ more, was not run. (K3 in the pre-registration says "a single slope with no N-dependence across N in {32,...,2048}" kills the shrinkage story; this experiment gives
one pair of N and a limited d range, so I only report that the observation is of that form, not that K3 is formally decided.)
eta_0-invariance (the admissibility check) was not run: only eta_0 = 1 was used, as in the Revision section.

## 6. CPU time

Per-run CPU time is `time.process_time()` inside the worker (column `cpu_s`). 4 worker processes, one BLAS/OMP thread each.

| phase | runs | cpu_h | max_run_cpu_s |
|---|---|---|---|
| bctl | 12 | 0.0749 | 106.1 |
| dscan | 24 | 0.1364 | 92.83 |
| kappa | 144 | 0.8838 | 287.4 |
| kappa2 | 19 | 0.7153 | 350.2 |
| p7 | 9 | 0.3215 | 188.4 |
| thr | 54 | 0.3454 | 102.7 |
| tied1 | 6 | 0.05701 | 66.96 |

- Total CPU time of all runs recorded in the CSVs: **2.534 h** (9124 s), of which 0.278 h belongs to `kappa` rows that were censored at the first (reduced) cap and then re-run from scratch in `kappa2`
  (identical trajectories up to the old cap; unavoidable as I did not checkpoint).
- Not recorded in the CSVs: pilot runs (12 runs of 1e5 steps, then an 8-run pilot whose output parser crashed after the runs finished) and tests, estimated at ~0.2-0.3 h CPU from the per-step costs.
  **Total therefore ~2.8 h CPU, below the 3 h cap** (the exact pilot share is an estimate).
- Wall-clock: priority 1 (thr, p7, tied1) ~ 11 min; kappa 13 min; kappa2 11 min; dscan 2 min; bctl 2 min (4 workers).
- Cost per step (us, 1 thread, B x N prompts): ~110 (N = 32, B = 32), ~200 (N = 128, B = 32), ~700 (N = 512, B = 32 or N = 2048, B = 8), ~1500 (N = 4096, B = 8).

## 7. Deviations from the spec / things to know

1. **Early stop at |m| >= 0.5** (spec exp 1 used 0.9). T_0.5 and Gamma at T_0.5 are therefore exact, T_0.9 is not recorded in any exp-1b CSV. Final |m| of escaped runs is ~0.5 by construction.
2. **Faster sampler for exp 1b only** (`train(..., fast=True)`, `ctx_stats_fast`): the same projected-sampler distribution, but one `standard_normal((2, B, N+1))` draw, fused reductions, and
   an `SFC64` generator seeded from `(seed, d, k, model, round(1000*m0_scale))` (exp 1 streams, `PCG64`, are unchanged because `tag=None` there). `ctx_stats_fast` is tested against `ModelA.coefs` (equal to 1e-12); the
   projected sampler itself is tested against full sampling in `test_projected_sampler_matches_full` / `test_init_fixed_and_tied_proj_matches_full` (tied model).
   Different m_0 multipliers use different random streams (so v, u, data differ across the 4 m_0 values for the same seed index); the same seed index shares streams across free1/free10/fixed protocols (and across N).
3. **B in the kappa phase: 32 for N <= 128, 8 for N >= 512** (4096 x 32 would have cost ~6 ms/step). The B-control at N = 128 shows no B effect on kappa_eff within 1 SE (5.38 vs 5.59), but it was done for N = 128, not at N >= 512.
4. **Step caps in the kappa phase** (spec: 1e6): tied 1e6 (all reached by ~2e4); first pass free: 2e5 (N = 32, 128), 3e5 (N = 512), 1.2e5 (free1, N = 4096), 2e5 (free10, N = 4096). Censored runs were re-run: N = 32, 128 at 1e6, free1 N = 4096 at 3e5.
   Final censoring: free1 N = 32 (3/12, all at m_0 multiplier 0.5, 1e6 steps), free10 N = 32 (3/12), free10 N = 128 (2/12), none elsewhere (free1 N = 4096: all reached, the slowest at 2.96e5 of 3e5 steps, close to the cap; N = 512 and free10 N = 4096 reached well before their caps).
5. **P7 total-sample ratios** are x1, x2.7, x3.2 (see section 2), not x1, x3, x10.
6. eta_0 invariance and `eta = eta_0/d^3` runs (spec exp 1b "Predictions" section) were not run; the Revision section only specifies eta = 1/d^2. d = 128 was not run.
7. Priority 3 was run only for N in {128, 2048} and d in {16, 24, 32, 48} with B = 32 (N = 128) and B = 8 (N = 2048).
8. An extra tied rho_0 = 1 check (section 3) and the B control were added beyond the stated runs.
9. The kappa SEs are plain OLS SEs over 12 (or fewer) runs and treat the log-linear form as exact; the free-Gamma T_0.5 includes the Gamma-growth phase, which need not follow `T ~ m_0^(-(kappa-2))`. I did not test curvature.

## 8. Bottom line (facts only)

- Fixed Gamma: the context-length threshold is qualitatively present (escape fraction decreases with gamma, d and with decreasing N) and quantitatively right within ~1 grid step at N = 128, 512; but 4 of the 7 cells where the formula says stuck (d >= d*) had at least one escaping seed (gamma = 1: N = 32 d = 16 2/3, N = 128 d = 32 1/3, and the two boundary cells d = d*: N = 128 d = 16 3/3, N = 512 d = 64 3/3), so it is not a hard threshold. P3 and P7 are met; P2 is met at d = 16 and 64 and failed at d = 32 (1/3 escaped).
- Tied readout: kappa_eff ~ 4.1-4.4 at all N (P4' met); tied with rho_0 = 1 does not behave like fixed gamma = 1 at d = 32 (escapes).
- Free readout: kappa_eff = 4.8-5.2 for N >= 512 and 5.4-6.6 for N <= 128; the pre-registered ~4 -> ~8 span is not reproduced; the d-scan at N = 128 vs 2048 shows no N-dependence of slope (3.5 vs 3.7).
