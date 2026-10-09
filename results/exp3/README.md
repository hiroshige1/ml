# Experiment 3 results: additive skill composition in a multi-neuron linear-in-label ICL model (C4)

- **Pre-registration:** `docs/preregistration.md`, section "Secondary question (C4)" (P12-P15)
- **Code:** `icl_additive/{multi,sweep3,analyze3}.py` (tests: `tests/test_basic.py`)
- **CPU time:** ~2.3 CPU-h (2.19 sweeps + ~0.1 pilot/tests) (cap 3)
- **Deviations from pre-registration:** added "settled" stop rule (spec stop and T_12 threshold unreachable at Gamma=0.1); absolute E12 threshold 0.5; extra seeds 5-9 and Gamma=0.3 runs added after seeing seed 0 (see "Deviations from the spec")

Spec: `docs/spec_exp3.md`. Pre-registered: P12-P15 in `docs/preregistration.md` ("Secondary question (C4)"). Code: `icl_additive/multi.py`
(model, exact projected sampler, evaluation sets, training loop), `icl_additive/sweep3.py` (driver, 2 workers),
`icl_additive/analyze3.py` (tables + figures); tests in `tests/test_basic.py` (`python tests/test_basic.py`, all pass).
Everything below is generated from `results/exp3/{main,extra_seeds5to9,gamma0.3}/` (`runs.csv`, `times.csv`, `additivity_residual.csv`,
`acc_fits.csv`, `assignment.csv`, `neurons.csv`, `analysis_stdout.txt`, `traj/seed*.npz` = all logged arrays). Figures: `figs/` (main, seeds 0-4),
`figs_extra/` (seeds 5-9), `figs_gamma0.3/`.

**Measured, not confirmed: the headline is mixed.** The additive structure itself is exact and visible (R_pair ~ 0, E12 curve = E1 + E2 curve),
but (i) the pre-registered estimator of R fails P12's 0.02 bound by sampling noise, (ii) the pre-registered T_12 (E12 MSE < 0.5) is unreachable at
Gamma = 0.1 so P13 is untested as written, (iii) skill 2 is not learned at all in 2 of 10 seeds, and (iv) the Gaussian-error curve of P14 is wrong in level.

## Setup actually run
d = 32, P = 2 orthonormal teachers (QR of a Gaussian matrix), k* = 2 (sigma_2) for both, N = 128, B = 32, eta = 1/d^2, M = 4 neurons uniform on the sphere,
fixed readout Gamma_j = 0.1, per-neuron renormalisation after each step, float64, analytic gradients. Single-skill prompts, skill 2 w.p. 0.25.
Seeds 0-4 are the pre-registered 5 seeds (`main`); seeds 5-9 (`extra_seeds5to9`) were added after seeing that skill 2 is missed in 1/5 seeds; `gamma0.3` (seeds 0-2, Gamma = 0.3)
is a supplementary run that makes the spec thresholds reachable. Logging every 500 steps (all m_jp, four eval MSEs/accuracies, running train loss).
Early stop: spec rule (E1 and E2 MSE < 0.1) plus "settled" (every neuron has |m_jp| >= 0.95 for some p, for 50,000 steps) or max_steps = 1.5e6 (see deviations).

**Exact projected sampler** (test `test_multi_projected_matches_full`): any orthonormal Q spanning [v_1, v_2, w_1..w_4] (6 dims) gives x = Qy + x_perp; the span part is
sampled in 6 coordinates, the orthogonal gradient part is sampled as jointly Gaussian across neurons with covariance S_jk = sum_n C^j_n C^k_n (exact, not an approximation).
Verified against full d = 32-style sampling for mean, per-neuron second moments and cross-neuron products (5 SE), and for a rank-deficient basis (neurons on v). 
**Tests:** finite-difference gradient (M=4, P=2, all four activations, relative error < 1e-4), M=1/P=1 reproduces `ModelA` loss and gradient to 1e-13, exact linearity in
context labels (error of the pair = e_1 + e_2 prompt by prompt), projected vs full sampling, rank-deficient QR.

**Evaluation sets** (fixed eval seed 20261008, same x for every training seed, 4096 contexts of 128 points + query): set A (inputs shared by E1, E2, E_x and the paired pair E12A:
independent c_1, c_2, c_x per prompt; E12A uses E1's c_1 and E2's c_2) and set B (own inputs and own c_1, c_2) = **E12 of the spec** (independent additive-pair set).
E_x: y = c sigma_1(v_1 x) sigma_1(v_2 x), E[y^2] = 1. MSE is absolute (E[y^2] = 1, 1, 2, 1 for E1, E2, E12, E_x). Each statistic comes in two versions:
`mse_*`/`acc*_*` = spec (one query per prompt, 4096 prompts) and `hmse_*`/`hacc*_*` = 17 queries on the same 4096 contexts (A_j computed once). **This is an addition to the spec**:
with one query per prompt the MSE has sampling SE ~ 0.1 (heavy-tailed y^2), which dominates every R(t) comparison (section 2).

## 0. The floor problem (read first)
With fixed Gamma = 0.1 and M = 4, n_p neurons on skill p can reduce MSE_p only to (1 - n_p Gamma)^2: 0.81 (n=1), 0.64 (2), 0.49 (3), 0.36 (4); E12 floor = sum of the two floors
(1.28 for a 2+2 split). Observed plateaus match (hMSE E1/E2 0.62-0.66 for n=2, 0.51 for n=3, 0.39 for n=4, 0.81-0.84 for n=1; the unlearned skill stays at 1.05-1.06). **The spec's stop rule (E1, E2 < 0.1), the
spec's T_12 (E12 MSE < 0.5) and the accuracy-saturation regime are therefore unreachable at Gamma = 0.1 by construction**: E12 reached 0.5 in 0 of 10 runs (min over t and seeds of hMSE_E12 = 1.28). 
At Gamma = 0.3 it was reached in 1 of 3 runs (seed 1, a 2+2 split; T_12 = 145,500, E12 plateau 0.398).

## 1. Times (steps; 500-step logging)
T_p = first log with max_j |m_jp| >= 0.5. T_12 (spec) = first log with MSE_E12 < 0.5. "never" = not reached when the run ended.

| config | seed | stop (steps) | T_1 | T_2 | T_2/T_1 | T_12 (spec, MSE<0.5) |
|---|---|---|---|---|---|---|
| main G=0.1 | 0 | settled (123,000) | 2,000 | **never** (4 neurons on v_1) | - | never |
| main | 1 | settled (281,500) | 10,000 | 48,500 | 4.9 | never |
| main | 2 | settled (716,000) | 14,000 | 283,500 | 20 | never |
| main | 3 | **max_steps** (1.5e6) | 10,500 | 303,000 | 29 | never |
| main | 4 | settled (243,500) | 13,500 | 46,000 | 3.4 | never |
| extra | 5 | settled (861,500) | 10,000 | 794,000 | 79 | never |
| extra | 6 | **max_steps** | 102,000 | 30,000 | 0.29 | never |
| extra | 7 | settled (697,000) | 12,500 | **never** (4 on v_1) | - | never |
| extra | 8 | settled (132,500) | 15,500 | 9,000 | 0.58 | never |
| extra | 9 | settled (470,000) | 21,000 | 7,000 | 0.33 | never |
| G=0.3 | 0 | settled (131,500) | 1,000 | 75,500 | 76 | never |
| G=0.3 | 1 | settled (196,500) | 4,000 | 18,500 | 4.6 | **145,500** |
| G=0.3 | 2 | **max_steps** | 6,000 | 105,000 | 18 | never |

- **T_2 > T_1 (pre-registered expectation) holds in 5 of the 8 G=0.1 seeds where skill 2 is learned (seeds 1-5) and fails in 3 (6, 8, 9)**, plus 2 seeds where skill 2 is never learned. The
  expectation "because pi_2 < pi_1" is not reliable with M = 4 random neurons: T_p is the *minimum over neurons* of an arrival time that depends on that neuron's initial alignment (below).
- **Why:** neurons arrive independently of one another with T_half ~ K / (eta pi_p m_0^2), K = T_half * eta * pi_p * m_0^2 (m_0 = initial |m_jp| with the teacher the neuron ends on; eta = 1/1024).
  First arrivals on a teacher (G=0.1, 18 first-arrival neurons over the 10 seeds; the 15 with m_0 < 0.3): K median 0.60, range 0.54-0.78; medians by teacher (all 18) 0.60 (T1, n=10) vs 0.58 (T2, n=8); the exp 1b single-neuron value at m_0 = d^-1/2, pi = 1 is
  T = 2.1e4 -> K = 0.64. The 3 first arrivals with large initial alignment (m_0 = 0.39-0.43, seeds 0, 8, 9) are faster than the cubic law (K = 0.26-0.34); the 20 later arrivals on an already-learned teacher are
  slower (K median 0.73, range 0.60-1.36; the 2nd arrivals alone 0.71). So **the skill-frequency delay is a clean factor pi_1/pi_2 = 3 in the neuron time, but the realised T_2/T_1 ratio ranges 0.29-79 because of the
  initial alignments (m_0^2 varies over a factor 100)**. Per-neuron table: `*/analysis_stdout.txt`, `*/neurons.csv`.

**T_12 versus max(T_1, T_2) (P13).** The pre-registered T_12 is undefined at Gamma = 0.1 (section 0). Substitutes (17-query MSE):
- E12 half-way / 90%-way to its own plateau (`Tm12_50`, `Tm12_90`) vs max of the single-skill half-way / 90% times (main, seeds with both skills learned): 90%: 227.5k vs 230.5k (s1), 664.5k vs 665.5k (s2), 318k vs 321k (s3),
  189k vs 192.5k (s4); extra: 809.5k vs 814k (s5), 107k vs 108.5k (s6), 80.5k vs 81.5k (s8), 416k vs 419.5k (s9). So T_12^90 <= max(T_1^90, T_2^90) always and lags it by only 1,000-4,500 steps (2-9 log intervals, **not within 500**), as it must,
  because E12 is the *sum* of the two curves and the earlier skill is already done.
- Fixed thresholds (E_p < 0.9, E12 < 1.8 = 0.9 + 0.9): T_12(sum) is typically far *earlier* than max(T_1, T_2) (e.g. s3: 23.5k vs 315k)
  since E12 < 1.8 can be met by one skill alone; "both single MSEs < 0.9" = max(T_1, T_2) by construction.
- **No extra delay for the pair is observed anywhere** (no run in which E12 lags the later single skill by more than 9 log intervals); but **no pair-specific threshold test of P13 as written was possible**, and the
  prediction "within one logging interval" is not met by the substitute definitions. Also for G=0.3 seed 1 the spec T_12 (145.5k) is 7x later than max(T_1, T_2) = 18.5k: reaching 0.5 needs a *second* neuron on each
  skill (T_12 follows the n-th neuron, not the skill emergence), i.e. a threshold on MSE at fixed Gamma measures neuron counts, not skill emergence.

## 2. Additivity residual R(t) = MSE_E12 - MSE_E1 - MSE_E2 (P12: |R| <~ 0.02 at all t)
Four estimators over the full trajectory (seed table: `*/additivity_residual.csv`; max |R| over logged t, mean R over the trajectory):

| main G=0.1 | spec: indep. sets, 1 query | 17 queries, indep. set B | paired (E12A), 1 query | paired, 17 queries |
|---|---|---|---|---|
| seed 0 | max 0.355, mean -0.259 | 0.107, -0.074 | 0.014, -0.010 | **0.0010**, -0.0004 |
| seed 1 | 0.265, +0.168 | 0.028, +0.014 | 0.039, -0.025 | **0.0018**, -0.0012 |
| seed 2 | 0.216, +0.177 | 0.041, +0.027 | 0.062, -0.048 | **0.0018**, -0.0004 |
| seed 3 | 0.037, +0.021 | 0.044, +0.032 | 0.022, -0.019 | **0.0056**, -0.0037 |
| seed 4 | 0.236, +0.164 | 0.074, +0.065 | 0.009, +0.008 | **0.0032**, +0.0024 |
| extra 5-9 (max over seeds) | 0.38 | 0.096 | 0.049 | **0.0086** |
| G=0.3 seeds 0-2 | 0.355 | 0.107 | 0.061 | **0.0036** |

- **P12 as pre-registered (spec estimator, independent 4096-prompt sets, one query): FAILED.** |R| > 0.02 in all 5 main seeds (max 0.04-0.35; 0 of 5 seeds had |R| < 0.02 at every log); the same in all 13 runs (max |R| 0.037-0.38).
- **Diagnosis (recorded as post-hoc, but checkable):** in the model R has *expectation exactly zero* for mean-zero independent c_1, c_2 (not "O(1/N)": the theory-notes cross term is zero in expectation), because the prediction is exactly linear in the
  context labels and e_12 = e_1 + e_2; so R = 2 mean(c_1 c_2 u_1 u_2) over the eval set, i.e. pure finite-eval-set noise (+ the difference in sample target energy between the two independent sets, which gives the large *constant*
  offsets: the offset is already present at t = 0 and R barely moves). Evidence: the paired estimator on the same x and c with 17 queries has max |R| <= 0.0086 in all 13 runs, mean R within +-0.004, sd over time 2-5e-4. With one query the paired
  estimator still exceeds 0.02 in 4 of 10 G=0.1 seeds (max 0.062). The time-dependence of the spec estimator is small relative to its offset (R - R(0): max 0.03-0.15 spec, 0.015-0.045 with 17 queries).
- So: **the additive structure holds to the precision of the paired measurement (|R| < 0.01, independent of trajectory stage, including during neuron transitions), and the pre-registered estimator cannot resolve 0.02.** The kill criterion "P12 fails => §4 wrong" is
  *literally* met by the pre-registered estimator; my reading is that it is the estimator, not §4, but that reading rests on the paired result and the linearity test, which the lead should judge.

## 3. Accuracy multiplicativity (P14)
acc = P(|err|^2 < 0.5) for E1, E2 **and E12 (absolute 0.5, as in the Gaussian model of the task)**; the spec's literal "0.5 E[y^2]" would give 1.0 for E12 and is reported as `rel_*`. Fits of acc_E12 on acc_E1 * acc_E2 over all logs
(`*/acc_fits.csv`; figure `figs/acc_multiplicative.png`):

| set (all logs) | variant | slope | intercept | RMS dev. from y = x | RMS about fit | mean dev. from y = x |
|---|---|---|---|---|---|---|
| main (n=5733) | 0.5 abs, 17 queries | 0.851 | +0.090 | **0.0084** | 0.0018 | -0.0075 |
| main | 0.5 abs, spec (1 query) | 0.795 | +0.127 | 0.0102 | 0.0064 | -0.0065 |
| main | log-spaced subsample, 17 q | 0.813 | +0.114 | 0.0072 | 0.0027 | |
| main | spec "0.5 E[y^2]" (1.0 for E12), 17 q | 0.723 | +0.302 | 0.121 | 0.0015 | +0.120 |
| extra 5-9 | 0.5 abs, 17 queries | 0.807 | +0.116 | 0.0115 | 0.0015 | -0.0108 |
| G=0.3 | 0.5 abs, 17 queries | 0.972 | +0.006 | 0.0165 | 0.0028 | -0.0162 |

- The data sit in a small corner (acc_1 acc_2 in 0.59-0.70 for G=0.1: acc_1 = acc_2 = 0.77 at t=0, 0.76-0.90 at the end, since the floor MSE >= 0.36 never lets the 0.5 threshold saturate). Over that range acc_12 = acc_1 acc_2 to RMS 0.008-0.012, slope < 1
  but the range is too short to constrain slope and intercept separately (slope and intercept are strongly anti-correlated; intercept 0.09 at slope 0.85 means the fit crosses y = x at ~0.6). For G=0.3 the range is larger (accuracy up to 0.97) and the fit is slope 0.97, intercept 0.006.
  Approximate multiplicativity: supported for the absolute-0.5 definition; with the spec's literal threshold for E12 (1.0) acc_12 exceeds the product by 0.12 (it is a different, more lenient event).
- **Gaussian-error model: the curve is wrong, not just approximate.** With s_p^2 = MSE_p, acc_p = P(chi^2_1 s_p^2 < 0.5) = erf(0.5/s_p): at t = 0 (MSE = 1) it predicts 0.52 but the observed single-skill accuracy is 0.77 (the error is the label c sigma_2(.) itself, heavy-tailed with most mass near 0). RMS
  error of the Gaussian prediction vs observed: acc_1 0.22, acc_2 0.22, acc_12 0.21 (main, 17 q; G=0.3: 0.17/0.17/0.20); the model's pair curve is also not on y = x (RMS deviation 0.09 from the line, curve slope 0.71, intercept +0.19) whereas the measured points deviate by 0.008.
  The theory points plotted on the same axes (black crosses, Gaussian curve for s_1 = s_2) fall at acc_1 acc_2 = 0.27-0.39, far from the data (0.59-0.70). **So the deviation from multiplicativity is not predicted by the Gaussian-error model; the observed approximate multiplicativity is real but its explanation here is not the Gaussian one.** I did not test an alternative error model.

## 4. E_x (non-additive product pair) (P15)
hMSE_Ex (E[y^2] = 1) per seed (init / min over t / final), G=0.1: s0 1.02/1.01/1.03, s1 1.03/1.03/1.03, s2 0.99/0.96/0.99, s3 1.02/1.01/1.02, s4 1.01/1.01/1.01; extra 5-9: finals 1.02, 1.03, 1.03, 1.00, 1.01.
**Stays at ~1.0 throughout (max dip 0.03 below its t=0 value in G=0.1, 0.06-0.08 in two G=0.3 runs, transient and reverting)**: P15 met; no product-skill learning. (The small transient dip occurs while a neuron is partly aligned with both teachers: sigma_2(w.x) of w = (v_1+v_2)/sqrt 2 contains the cross term, so partial overlap with the product exists.
That reading is from the formula, not from a dedicated check.)

## 5. Decoupling: neuron-teacher assignment
Assignment = argmax_p |m_jp| once max >= 0.5; final counts (n on T1, n on T2, n unassigned). `figs/alignments.png` (seed 1, a 2+2 split), `figs/alignments_all_seeds.png`.

| G=0.1 seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| (n1, n2, unassigned) | 4,0,0 | 2,2,0 | 2,2,0 | 2,1,1 | 2,2,0 | 3,1,0 | 1,2,1 | 4,0,0 | 3,1,0 | 2,2,0 |

- Skill 2 is learned in 8/10 seeds (G=0.3: 3/3); **2/10 seeds have all four neurons on teacher 1 and skill 2 never emerges** (state stable: unchanged for >= 50,000 steps by the settled rule; seed 0's pilot ran 300k with no change). A 2+2 split occurs in 4/10 seeds. The assignment is
  winner-take-all by (pi_p m_0^2): a neuron goes to teacher 2 only if its initial alignment is large enough to beat the factor 3 in frequency (a rough rule |m_j2| > 1.44 |m_j1|; not fitted).
- **Switching: none.** No neuron that reached |m| >= 0.5 on one teacher was later assigned to the other (13 runs x 4 neurons, 500-step resolution). Neurons that start with partial alignment to the "wrong" teacher (up to 0.35-0.48) lose it when the right-teacher transition completes (column `max_other_after_assigned` in `assignment.csv` is the maximum over the whole trajectory, so it includes the initial alignments).
- **One neuron trapped at the origin in 3 of 13 runs** (G=0.1 seed 3 neuron 1, seed 6 neuron 2; G=0.3 seed 2 neuron 1): initial |m| = (0.03, 0.03), (0.007, 0.009), (0.053, 0.009), still < 0.06 on both teachers after 1.5e6 steps. These are exactly the three runs that hit max_steps (so max_steps did not stop a run that was still moving
  toward a solution; the remaining neuron was stuck, consistent with the exp 1b drift-stable origin for m_0 below ~2 sqrt(Gamma/N) = 0.056 at Gamma=0.1; at Gamma = 0.3 the threshold is ~0.1). This is the single-neuron repulsion mechanism carried over, not a new effect.
- Neurons beyond the first on a teacher are slightly slower (K median 0.73 vs 0.60 for first arrivals), i.e. coupling through the residual is weak (Gamma = 0.1) but visible; first arrivals on the two teachers have the same K within noise (median 0.60 vs 0.58), so the two skills' learning times are consistent with independent single-neuron escapes with rate scaled by pi_p (decoupled across skills; mildly coupled within a skill).

## 6. Pre-registered predictions vs measurements
| ID | prediction (ours) | measured | verdict |
|---|---|---|---|
| P12 | abs(R) <~ 0.02 at all t | spec estimator: max abs(R) 0.04-0.35 per seed (fails, sampling noise/offset of fixed independent eval sets); paired 17-query estimator: max 0.0086 over 13 runs | **failed as pre-registered; additivity itself holds (see section 2)** |
| P13 | T_12 = max(T_1, T_2) within 500 steps | spec T_12 unreachable at Gamma=0.1 (0/10); substitute definitions: T_12^90 lags max by 1,000-4,500 steps, never exceeds it | **not testable as written; no extra pair delay seen; "within 500" not met by substitutes** |
| P14 | acc_12 ~ acc_1 acc_2, deviation from the Gaussian-error model | abs-0.5: RMS 0.008-0.012 from y=x over a narrow range; Gaussian model wrong in level (0.52 vs 0.77) and in curvature | **approximate multiplicativity yes; Gaussian-error explanation no** |
| P15 | E_x MSE ~ 1 throughout | 0.91-1.05 for all 13 runs (transient dip <= 0.08 below t=0 value) | **met** |
| (C4 expectation, spec) | T_2 > T_1 since pi_2 < pi_1 | 5 of 8 seeds with both learned; 2/10 seeds never learn skill 2 | partly |

## Deviations from the spec / things done differently
1. **Floor problem and thresholds** (section 0): spec stop rule and T_12 cannot be met at Gamma = 0.1, M = 4. Added the "settled" stop rule (50,000 steps with every neuron |m| >= 0.95 on some teacher; or max_steps 1.5e6). The spec stop never fired (0/13).
   Seeds 3 (main), 6 (extra), and G=0.3 seed 2 ran to the full 1.5e6 steps; the rest stopped as "settled" at 0.12-0.86 of that.
2. **E12 accuracy threshold:** absolute 0.5 used as primary (the task's Gaussian model); the spec's 0.5 E[y^2] (= 1.0 for E12) reported as `rel_*`.
3. **Evaluation:** single query per prompt gives MSE SE ~ 0.1; added 16 extra queries per context (`hmse_*`) and a paired pair set E12A (same x and c as E1, E2) next to the independent spec set; spec statistics (`mse_*`, `acc*`) are all logged and reported.
   E1, E2, E_x share inputs (independent c) in set A. Time-to-threshold columns use the 17-query version (low noise); alignment times need no eval.
4. **Extra seeds 5-9 and Gamma = 0.3 (seeds 0-2)** were not in the spec (added after seeing seed 0 lose skill 2 and the floor problem). The main tables use seeds 0-4 as pre-registered; seeds 5-9 are reported separately and not used to alter conclusions about the 5 pre-registered seeds, only to give counts (2/10, 4/10).
5. **Initial alignment drives T_p:** teachers and neurons are drawn per seed with no control of m_0 (as spec: uniform init), hence the wide T_2/T_1 spread; no new init protocol was tried.
6. T_12 under the "E12 sum threshold" definitions is my choice (spec threshold is unreachable); other choices are possible and change the numbers (section 1).
7. A work-in-progress commit (`37b5613`) by the lead was taken while runs were in progress; the runs use the final working-tree code (`multi.py` after the multi-query evaluation and `stop_align` edits).

## Compute
CPU time measured inside workers (`runs.csv`): main 0.758 h (119/271/684/224/1430 s for seeds 0-4), extra seeds 0.954 h, Gamma = 0.3 0.481 h; **sweeps total 2.19 h**. Plus the pilot (seed 0, 300k steps, 284 s) and test/timing runs
(~0.1 h): about 2.3 CPU-hours of the 3-hour cap; 2 worker processes. Throughput 0.95 ms per training step (B=32, N=128, D=6 coordinates) + 0.11 s per evaluation (4096x2 contexts x 17 queries); the projected sampler gives
~5x fewer random numbers than full sampling. Pilot: 40,000 steps -> 51 s before optimisation (1.3 ms/step incl. eval) -> extrapolated 1.5e6 steps = 0.55 CPU-h per seed, 2.7 h for 5 seeds; with the "settled" rule the 5 main seeds cost 0.76 h (only seed 3 hit max_steps). No reduction of max_steps was needed.

## Files
`figs/curves.png` (E1, E2, E12, E_x vs t, seeds overlaid, 17-query MSE), `figs/additivity_residual.png` (R spec; R paired 17-query), `figs/acc_multiplicative.png` (acc_12 vs acc_1 acc_2, OLS fit, Gaussian-error curve and points, zoom),
`figs/alignments.png` (|m_jp|, seed 1), `figs/alignments_all_seeds.png`; `run_*_stdout.txt`, `pilot_stdout.txt`; per-config analysis in `*/analysis_stdout.txt`.
