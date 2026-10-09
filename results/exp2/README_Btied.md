# Experiment 2 add-on: B-tied, a decoupled (2-homogeneous, unconstrained-norm) in-weight baseline

Code: `icl_additive/btied.py` (model, exact gradient, exact population MSE, training loop), `icl_additive/sweep_btied.py` (driver, 4 workers x 1 BLAS thread), `icl_additive/analyze_btied.py`
(statistics = `analyze2`'s `skill_stats`/`window_fit`/`reg`, imported unchanged; figures), `tests/test_btied.py`. Outputs: `results/exp2/btied/` (`runs.csv`, `runs/Btied_eta{0.25,1,4}/seed{0,1,2}.npz`, `summary.csv`, `per_skill.csv`,
`analysis_stdout.txt` with every number quoted here), figures `results/exp2/figs/Tp_vs_p_with_Btied.png` (A | B | B-tied, T_p(0.5) and T_p,0.5 rows) and `results/exp2/figs/Btied_internals_seed0.png`.
No existing exp-2 file was modified. **Measured, not confirmed.** Headline: the per-skill timescale law T_p,0.5 ~ 1/(eta a_p) holds (slope 0.75 on the specified 1000-step log, 0.87-0.89 once the early transitions are resolved; pre-registered for B: 0.75 +- 0.2),
the pre-registered exponent (0.67 +- 0.15) is **not** met (0.24-0.31), and **B-tied is collective, not decoupled**: no neuron ever exceeds |m| = 0.89, 49-51 of 64 neurons sit at 0.5-0.9, every skill 1-8 is shared by >= 2 neurons (skill 1 by ~19), and the skills are reproduced only through the pooled quantity sum_j |u_j|^2 m_jp^2 = a_p.

## Model and what was run
f(x) = sum_j |u_j|^2 sigma_2(uhat_j . x), uhat_j = u_j/|u_j|; target y = sum_p a_p sigma_2(v_p . x), a_p = sqrt(pi_p), pi_p ~ p^-1.5 (a_p ~ p^-0.75, sum a_p^2 = 1; sum a_p = 3.173); d = 32, P = 16, M = 64, |u_j|^2 = rho_0 = 0.01 at init (sum 0.64),
uniform directions, **plain online SGD on the unconstrained u (no projection, no renormalisation)**, batch B = 32 fresh Gaussian inputs per step (same B as exp 2), float64, max_steps 3e6, seeds 0, 1, 2. V and the initial directions are **identical to exp 2's A and B at the same seed** (same `init_multi` seed sequence).
eta = 1/d^2 is the primary setting; also eta = 4/d^2 (as requested) and, **added by me**, eta = 0.25/d^2 (see "eta" below). Logged every 1000 steps: all m_jp = uhat_j . v_p (T,64,16), |u_j|^2, exact population MSE, alignment loss; **additionally** (supplementary, my addition) a 100-step log for t <= 30000 and the initial/final directions.

Gradient (n_j = |u_j|^2, z = uhat . x, r = f - y, B-batch mean): dL/dn_j = sum_b (2 r_b/B) sigma(z_bj); dL/duhat_j = n_j sum_b (2 r_b/B) sigma'(z_bj) x_b; **dL/du_j = (I - uhat uhat^T) dL/duhat_j / |u_j| + 2 u_j dL/dn_j** (chain rule as in `models.ModelATied`).
**Checks** (`python tests/test_btied.py`): central finite differences at three norm scales (|u|^2 ~ 0.01, 0.25, 1, random per-neuron spread): max relative error over 200 random coordinates **8.7e-8, 1.3e-7, 3.4e-6** (< 1e-4), relative error along a random direction <= 1.5e-8;
exact population MSE ||sum_j n_j uhat_j uhat_j^T - sum_p a_p v_p v_p^T||_F^2 agrees with Monte Carlo (400k points) within 4 SE at all three scales (e.g. 0.8644 vs 0.8632 +- 0.0032). `tests/test_many.py` still passes.

## Definitions (identical to exp 2, `README.md`)
mx_p = max_j |m_jp|; l_p = 1 - mx_p^2; L_align(t) = sum_p pi_p l_p. T_p(0.5) = first t with mx_p >= 0.5 (0 if already true at t = 0, excluded from log fits); T_p,0.5 = midpoint of the drop of l_p (fraction 0.5 of its own total drop); sharpness = (T_p,0.9 - T_p,0.1)/T_p,0.5 (drop-based);
slope = OLS of log T on log p over learned skills, pooled over the 3 seeds (SE = OLS SE; n = skill-seed pairs); exponent = -slope of log L_align vs log t between the 3rd and 8th emergence (order statistics of T_p(0.5), as specified). **"Learned" = alignment loss dropped by >= 0.1** (the exp-2 Model-B criterion; the A criterion final mx >= 0.9 is met by no neuron here).
Skills learned this way: 7-11 per run (all of 1-8 in 7 of 9 runs, 7 of 8 in the other two; only 4 skill-seed pairs with p >= 9 pass, p = 9, 13, 14, at every eta).

## Predictions vs measurements (alpha = 1.5, seeds 0-2), next to exp 2's A and B rows
| statistic | pre-registered | A (exp 2) | B (exp 2, rho = 0.0496) | **B-tied eta = 1/d^2 (primary)** | B-tied eta = 0.25/d^2 | B-tied eta = 4/d^2 |
|---|---|---|---|---|---|---|
| slope, T_p(0.5) (spec), 1000-step log | B: 0.75 +- 0.2 | 1.53 +- 0.22 (n=23) | 1.15 +- 0.24 (n=22) | **1.30 +- 0.19** (n=25); p<=8 1.19 +- 0.23 | 1.50 +- 0.23 (n=24) | 0.69 +- 0.15 (n=24) (resolution-limited) |
| slope, T_p,0.5 (drop midpoint), 1000-step log | B: 0.75 +- 0.2 | 1.44 +- 0.08 (n=28) | 0.75 +- 0.08 (n=25) | **0.75 +- 0.09** (n=28; per seed 0.71, 0.82, 0.62); p<=8 0.59 +- 0.10 | 0.87 +- 0.09 (n=27) | 0.22 +- 0.05 (n=27): **artifact**, T_p,0.5 < 1000 steps |
| slope, T_p,0.5 with the 100-step early log (supplementary) | | n/a | n/a | **0.89 +- 0.09** (n=28); p<=8 0.77 +- 0.10 | 0.88 +- 0.09 | 0.89 +- 0.09 |
| slope, T_p(0.5) with the 100-step early log | | | | 1.50 +- 0.21 | 1.52 +- 0.23 | 1.43 +- 0.22 |
| median sharpness (drop-based) | A < B/2 | **1.08** | **4.09** | **2.07** (per seed 2.05, 2.15, 2.08); with T_p(0.5) as T_mid 3.86 | 2.00 | 1.60 on the 1000-step log (unresolved); 2.00 with the 100-step log |
| exponent of L_align, 3rd-8th emergence (spec window) | B: 0.67 +- 0.15 | 0.40 +- 0.035 | 0.19 (0.25, 0.16, 0.15) | **not computable** on the 1000-step log (window 100-1900 steps holds 0-1 logged points); 100-step log: **0.27 +- 0.02** (0.30, 0.27, 0.23) | 1000-step log 0.31 (0.36, n/a [2 points], 0.26); 100-step log **0.24 +- 0.04** (0.30, 0.24, 0.18) | 100-step log 0.31 +- 0.10 (0.34, 0.47, 0.13) |
| same, drop-window (T_p,0.5 order statistics) | | 0.76 | 0.22, 0.18, n/a | 0.15 (0.21, 0.18, 0.05) | 0.14 | 0.22 |
| exponent of the exact MSE, same window (secondary) | | 0.27 | 0.92 (+-0.26) | 0.58 (0.63, 0.33, 0.79) | 0.44 (0.63, 0.27, 0.42) | 1.27 (1.20, 0.95, 1.67) |
| L_align at the end of the run | | 0.03-0.07 | 0.37-0.40 | 0.37-0.42 | 0.36-0.42 | ~0.40 |
| population MSE at 3e6 steps (seed 0) | | (floor 0.25-0.30, not comparable) | ~1e-5 | 2.9e-8 (7e-7 at 6e5 steps) | 4.5e-7 | 1.9e-9 |

Verdicts against the pre-registered B expectations: **slope 0.75 +- 0.2: met** by the drop-midpoint (0.75 +- 0.09 on the specified log, 0.87-0.89 +- 0.09 resolved; both inside 0.55-0.95), **not met by the spec's T_p(0.5)** (1.3-1.5 +- 0.2; same pattern as exp 2's B, polluted by the initial alignments: 1-2 skills per seed are at >= 0.5 at t = 0, best-of-64 initial |m| is 0.36-0.55).
**Exponent 0.67 +- 0.15: not met** (0.24-0.31 for the alignment-based loss, outside 0.52-0.82; the MSE-based exponent 0.4-0.6 is nearer but is a curved, 3x-scattered transient). B-tied's alignment exponent (0.24-0.31) lies between exp 2's B (0.19) and A (0.40), nearer B; A/B-tied = 1.5, i.e. the ordering of the P18 ratio (A should decay slower than B) is still reversed, as for exp 2's B.

### eta
Pilot (seed 0, 6e5 steps, both eta = 1/d^2 and 4/d^2): the fit is essentially complete after the norm growth, MSE 0.9 -> 2e-2 within ~3000 steps at eta = 1/d^2 and -> 8e-3 within 1000 steps at 4/d^2, so **1/d^2 is not too slow; it is already faster than the requested 1000-step log resolves** (the emergence times of skills 1-8 are 300-2500 steps, 0.3-2.5 log intervals), and 4/d^2 is worse (T_p,0.5 < 1000 steps, which makes its 1000-step slope 0.22 and sharpness 1.6 pure interpolation artifacts).
I therefore (a) report both as requested, (b) **added eta = 0.25/d^2** (T 4x longer: 1000-step log resolves it) and (c) added the 100-step early log. T_p,0.5 scales as 1/eta: T eta a_p is 0.23-0.26 (below) at 0.25, 1 and 4. The eta-robust slope is 0.87-0.89 +- 0.09, the 0.75 of the primary 1000-step log is lowered by the interpolation (skill 1: T_p,0.5 = 500 in all three seeds = midpoint of the first log interval; 100-step log: 282-348).

## T_p tables (eta = 1/d^2, steps; "(x)" = not learned (drop < 0.1), "0" = aligned at t = 0, not used in fits)
| p | a_p | T_p(0.5) s0 | s1 | s2 | T_p,0.5 (1000-step log) s0 | s1 | s2 | T_p,0.5 (100-step early log) s0 | s1 | s2 | final best |m| s0/s1/s2 | neurons ending on p (argmax) s0/s1/s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.687 | 174 | 156 | 62 | 500 | 500 | 500 | 348 | 315 | 282 | 0.85/0.86/0.87 | 18/20/18 |
| 2 | 0.408 | 90 | 276 | 283 | 561 | 639 | 678 | 490 | 650 | 540 | 0.80/0.89/0.69 | 12/8/12 |
| 3 | 0.301 | 410 | 268 | 454 | 684 | 706 | 823 | 683 | 665 | 835 | 0.71/0.78/0.62 | 10/8/9 |
| 4 | 0.243 | 1351 | 1430 | 497 | 1252 | 1495 | 998 | 1199 | 1450 | 1001 | 0.59/0.58/0.73 | 7/6/6 |
| 5 | 0.205 | 1267 | 210 | 0 | 1290 | 1120 | 1270 | 1375 | 1099 | 1215 | 0.61/0.68/0.84 | 3/3/2 |
| 6 | 0.179 | 714 | 0 | 570 | 1170 | 743 | 912 | 1149 | 670 | 931 | 0.66/0.75/0.57 | 3/3/3 |
| 7 | 0.160 | 0 | 2805 | 1948 | 1711 | 2355 | 1409 | 1684 | 2250 | 1197 | 0.72/0.53/0.54 | 2/3/3 |
| 8 | 0.144 | 321 | 839 | 3066 | 696 | 2522 | 2535 | 645 | 2492 | 2520 | 0.59/0.63/0.54 | 4/5/2 |
| 9 | 0.132 | (never) | 837 | (0) | (3629) | 2048 | (0) | (3594) | 2043 | (98) | 0.48/0.64/0.51 | 0/3/1 |
| 10 | 0.122 | (never) | (0) | (5747) | (5879) | (967) | (5618) | (5817) | (544) | (5570) | 0.48/0.58/0.51 | 0/1/2 |
| 11 | 0.114 | (never) | (never) | (never) | (0) | (54286) | (0) | (0) | (54286) | (0) | 0.43/0.45/0.46 | 0/0/1 |
| 12 | 0.107 | (never) | (never) | (never) | (0) | (4914) | (4466) | (0) | (4909) | (4372) | 0.45/0.44/0.43 | 0/1/0 |
| 13 | 0.100 | (4847) | 4731 | (5437) | (3602) | 4078 | (4169) | (3569) | 4068 | (4130) | 0.53/0.55/0.53 | 1/2/2 |
| 14 | 0.095 | 8337 | 2309 | (never) | 4778 | 3622 | (4114) | 4764 | 3592 | (4165) | 0.53/0.60/0.50 | 2/1/1 |
| 15 | 0.090 | (3774) | (never) | (never) | (4953) | (0) | (19673) | (4939) | (0) | (19700) | 0.56/0.38/0.50 | 1/0/1 |
| 16 | 0.086 | (never) | (never) | (never) | (5937) | (0) | (6674) | (5917) | (0) | (6632) | 0.49/0.39/0.43 | 1/0/1 |

## The T_p,0.5 eta a_p constant (in-weight k* = 2 drift ~ a_p m)
Neuron j_end(p) = the neuron with the largest final |m_jp|; m0 = its initial |m|. Learned skills, pooled over seeds (`analysis_stdout.txt`, INVARIANT CHECK):
| K | eta | grid | median (IQR) | CV | slope of log K vs log p |
|---|---|---|---|---|---|
| T_p,0.5 eta a_p | 1/d^2 | 1000-step | 0.262 (0.223-0.335) | 0.29 | 0.00 +- 0.09 (p<=8: -0.16 +- 0.10) |
| | 1/d^2 | 100-step | 0.241 (0.200-0.296) | 0.32 | 0.14 +- 0.09 (p<=8: 0.02 +- 0.10) |
| | 0.25/d^2 | 1000-step | 0.226 (0.204-0.277) | 0.33 | 0.12 +- 0.09 (p<=8: -0.03 +- 0.10) |
| | 4/d^2 | 100-step | 0.243 (0.206-0.331) | 0.32 | 0.14 +- 0.09 (p<=8: 0.03 +- 0.11) |
| T_p,0.5 eta a_p m0 (j_end) | 1/d^2 | 1000-step | 0.110 | 0.28 | 0.02 +- 0.08 |
| T_p,0.5 eta a_p m0^2 (j_end) | 1/d^2 | 1000-step | 0.049 | 0.31 | 0.04 +- 0.08 |
| T_p,0.5 eta a_p m0^2 (best neuron at init) | 1/d^2 | 1000-step | 0.051 | 0.29 | 0.02 +- 0.08 |
| T_p,0.5 eta a_p / ln(1/m0) (j_end) | 1/d^2 | 1000-step | 0.305 | 0.27 | 0.03 +- 0.08 |

So T_p,0.5 ~ 1/(eta a_p) holds across a 16x range of eta and skills 1-16 (K ~ 0.24 +- 30%, no trend in p), consistent with a slope of 0.75 (a_p^-1 ~ p^0.75), measured 0.75-0.89. **Multiplying by the initial alignment of the neuron that ends on p does not reduce the scatter** (CV 0.27-0.34 for every variant, 1000- and 100-step logs, vs 0.29-0.32 without it; contrast A, where K = T eta pi m0^2 had IQR 0.70-0.77): the initial alignment of the "winning" neuron is not what sets T here, as expected if skills are not carried by single neurons.
In exp 2's A the scatter of T_p was explained by m0; here it is not (and not by the best-at-init neuron either).

## Neurons per skill, sharing, and decoupled vs collective
Seeds 0-2, final state (identical picture at all three eta; per-seed lines in `analysis_stdout.txt`):
* **Neurons ending on skill p** (argmax_p |m_jp|; eta = 1/d^2, mean over seeds), p = 1..16: 18.7, 10.7, 9.0, 6.3, 2.7, 3.0, 2.7, 3.7, 1.3, 1.0, 0.3, 0.3, 1.7, 1.3, 0.7, 0.7 (55-59 of the 64 neurons end on skills 1-8). **Skills share neurons: every skill 1-8 has >= 2 neurons in every run** (9 of 9 runs; skill 1 has 17-20); neurons with |m_jp| >= 0.3: 33 (p=1) ... 5-7 (p=16).
* **No neuron is aligned**: best final |m| per skill 0.86, 0.79, 0.71, 0.63, 0.71, 0.66, 0.60, 0.59 (p = 1-8) and 0.43-0.54 for p >= 9 (close to the initial best-of-64 draws, 0.36-0.55, though a few move up to 0.64); max over all neurons and skills 0.85-0.89; 0 neurons >= 0.9, 48-53 with max |m| >= 0.5 (0.5-0.9). For p >= 9 the best alignment barely moves (and for some it dips to 0.32 transiently, `Btied_internals_seed0.png`): these skills **never emerge** individually.
* **The fit is pooled**: q_p = sum_j |u_j|^2 m_jp^2 / a_p = 1.000 for all 16 skills (it is 0.02-0.27 at t = 0), MSE ~ 1e-7, while the single best neuron supplies only a share max_j n_j m_jp^2/a_p = 0.06-0.18 (mean 0.10) of a_p. About ten neurons are needed per skill, and 64 neurons carry 16 skills.
* Directions do move (mean |cos(uhat_init, uhat_end)| = 0.65, min 0.45), but towards a collective arrangement, not onto the teachers (skill 1 overshoots to |m| ~ 0.90 at ~1200 steps, then relaxes to 0.85, `Btied_internals_seed0.png`).
* Pooled-alignment midpoint statistic (supplementary, q_p >= 0.5): slope 0.44 +- 0.03 (eta = 1/d^2), 0.41 +- 0.03 (0.25/d^2), mostly the norm-growth phase.

**Verdict: collective, not decoupled.** The timescale law per skill (T ~ 1/(eta a_p)) looks decoupled, but the mechanism is pooled: the total norm grows from 0.64 to the target's sum a_p = 3.173 within ~3000 steps (sum |u|^2 = 3.174 at the end) while the 64 > d = 32 neurons re-weight and rotate until sum_j n_j uhat uhat^T = sum_p a_p v_p v_p^T, with no neuron isolating a skill.
**The pooled alignment is needed**: individual-neuron alignment gives L_align stuck at 0.40 while the population MSE falls to 1e-7, and 8 of 16 skills (p >= 9) are fit exactly (q = 1) with no individual alignment event. This is the same qualitative regime as exp 2's projected B (rho = 0.0496), reached here without a hand-chosen readout and without projection, so the exp-2 caveat ("B at M > d is a collective lazy-like fit") is **not an artifact of the fixed-norm/rho choice**. I did **not** test the likely cause (M = 64 > d = 32 neurons spanning R^d, so a PSD target is reachable by norm re-weighting alone), e.g. with M < d; this would be the natural next check.
A comparison to A: A ends with 34-44 neurons at |m| >= 0.9, T_p carries m0 (K IQR 0.70-0.77), sharpness 1.08, and L_align falls to 0.03-0.07; B-tied has none of these.

## Anomalies and caveats
* **Time resolution.** All emergence times of skills 1-8 are 0.3-3 log intervals long at eta = 1/d^2 (and < 1 at 4/d^2). Linear interpolation inside the first interval puts T_p,0.5 of skill 1 at exactly 500 in all seeds. The primary 1000-step slope (0.75) is therefore biased low relative to the 100-step one (0.89); I report both and the primary eta/log is as requested. The spec exponent window (T_p(0.5) order stats 3-8, ~100-1900 steps) contains 0-1 logged points at eta = 1/d^2 and 0 at 4/d^2, so the exponent is computed from the 100-step early log (log-spaced resampling, 40 points) - a deviation; at eta = 0.25/d^2 the 1000-step version works for 2 of 3 seeds (0.36, 0.26; the third has 2 points) and agrees roughly (0.31).
* T_p(0.5) is contaminated by the initial alignment (1-2 skills per seed already >= 0.5 at t = 0); T_p(0.5) vs the drop-midpoint differ by +0.55 in slope (1.30 vs 0.75), as in exp 2's B (1.15 vs 0.75).
* "Learned" by the weak drop >= 0.1 criterion includes 4 skill-seed pairs with p >= 9 (p = 9, 13, 14) whose T_p,0.5 (2-5k steps) is set by weak alignment changes; fits restricted to p <= 8 give 0.59 +- 0.10 (eta = 1/d^2, 1000-step log), 0.77 +- 0.10 (100-step log) and 0.72 +- 0.10 (eta = 0.25/d^2), lower than the all-learned values but within ~1.5 SE. The un-filtered (all 16 skills) slope is 1.05 +- 0.13, because p >= 9 "emergence" times are noise around 4-6k steps.
* Alignment is non-monotone for p >= 9 (best-neuron |m| dips by up to 0.1 and recovers) and skill 1 overshoots; l_p = 1 - mx^2 can therefore be non-monotone, which is why the drop is measured against the minimum over the run.
* L_align plateaus at ~0.40, so a power-law exponent for it describes a transient in front of a plateau (1-2 decades in t at most), not a scaling law; the MSE-based exponent is curved and 3-fold scattered between seeds.
* The M < d question above, and a 3rd and finer eta grid, are not run. Seeds are 0, 1, 2 only (as specified); SEs are OLS SEs over skill-seed pairs, per-seed slopes are listed in `analysis_stdout.txt`.
* Repository note: during this run the shared working tree was committed by another process (commits `ebf79e3` "WIP: B-tied baseline partial results" and `fe59223`), which swept my partial files (`btied.py`, `analyze_btied.py`, four `.npz` of the first runs) into history; I did not commit or push anything. Each `.npz` is ~13 MB (float32 m, 3001 x 64 x 16, incompressible), 9 files = 117 MB under `results/exp2/btied/runs/`.

## Compute
Nine runs x 3e6 steps, 0.082 ms/step: 246-248 s CPU each = **0.62 CPU-h** (`runs.csv`; 3 x 0.205 h per eta; 4 workers x 1 thread, 765 s wall for the batch) + pilots (2 x 50 s) + tests and analysis (~0.05 h) = **about 0.7 CPU-h** of the 1.5 h budget.
