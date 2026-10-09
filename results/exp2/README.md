# Experiment 2 results: emergence times and scaling law with many skills (P = 16, M = 64)

- **Pre-registration:** `docs/preregistration.md`, section "Many-skill scaling (exp 2)" (P16-P18)
- **Code:** `icl_additive/{many,sweep2,analyze2}.py` (tests: `tests/test_many.py`); B-tied add-on: `icl_additive/{btied,sweep_btied,analyze_btied}.py`
- **CPU time:** ~3.2 CPU-h (2.92 sweeps + 0.3 pilots/tests) (cap 4)
- **Deviations from pre-registration:** Model B readout rho = sum a_p/M not 1 (spec-literal is degenerate); A max_steps 1e6 not 3e6; extra seeds 3-4 and alpha=1.0 runs; eval set and T_p definitions changed (see "Deviations from the spec")

Spec: `docs/spec_exp2.md`. Pre-registered: P16-P18 in `docs/preregistration.md` ("Many-skill scaling (exp 2)"). Code: `icl_additive/many.py` (Model A fast exact
gradient, Model B, evaluation, training loop), `icl_additive/sweep2.py` (driver, 4 workers x 1 thread), `icl_additive/analyze2.py` (tables + figures);
tests `tests/test_many.py` (`python tests/test_many.py`, all pass). **Measured, not confirmed. Headline: P16 (in-context slope ~ alpha) and P17 (A sharper than B) hold
numerically, but P18 does not (alignment-based exponent of A is not below B's), and Model B at M = 64 > d = 32 is a collective (lazy-like) fit in which most of its "skills"
never emerge as individual events, so every A-vs-B comparison below carries that caveat.**

Layout: `summary.csv`, `per_skill.csv`, `analysis_stdout.txt` = **seeds 0-2** (the 3 pre-registered seeds; used for all tables below unless stated); `*_seeds0-4.*` and `figs_seeds0-4/` = the
same analysis with the two extra seeds 3, 4 at alpha = 1.5 (added after seeing the 3-seed result, as a robustness check); `runs.csv` (steps, stop reason, CPU s per run), `runs/<tag>/seed*.npz`
(all logged arrays: `m` (T,64,16) every 1000 steps, `mse_spec`/`mse_h` per-skill eval MSE (A), `mse_total`, `train_loss`), `run_*_stdout.txt`. Figures `figs/`: `Tp_vs_p.png`, `curves_per_skill.png`,
`scaling_law.png` (alpha = 1.5) and `*_alpha1.0.png`.

## Setup actually run
d = 32, sigma_2, P = 16 orthonormal teachers (QR of a Gaussian), pi_p ~ p^-alpha normalised (alpha = 1.5: pi_1 = 0.472, pi_8 = 0.0208, pi_16 = 0.0074), N = 128, B = 32, eta = 1/d^2, M = 64 neurons uniform on the sphere (per-neuron
renormalisation), float64. **Model A**: `multi.MultiModelA`, fixed readout Gamma = 0.1, single-skill prompts, x sampled fully in d = 32 (no projected sampler). **Model B**: y = sum_p a_p sigma_2(v_p.x), a_p = sqrt(pi_p)
(so a_p ~ p^(-alpha/2), sum a_p^2 = 1), yhat = rho sum_j sigma_2(w_j.x), B = 32 fresh inputs per step, same V and same initial W0 as Model A at the same (seed, alpha). Logging every 1000 steps. A stops when skills 1-8 all have max_j|m_jp| >= 0.95
(never hit the cap), B always runs 3e6 steps. **Gradient checks** (`tests/test_many.py`): Model A M = 64, P = 16, d = 32: central-difference relative error 3e-6 (40 random coordinates), and the fast gradient (`grad_A_fast`, an exact algebraic
rewrite via G_b = (1/N) sum_n y_n x_n x_n^T that never forms the (B, N, M) activations; 2.8 ms/step) equals `MultiModelA.loss_grad` to 9e-17; Model B: FD error 1e-7, and the exact population MSE ||rho W^T W - sum a_p v_p v_p^T||_F^2 matches
Monte Carlo (185.6 vs 185.1 +- 0.5 at rho = 1; 1.709 vs 1.703 +- 0.005 at rho = 0.1).

## Definitions (what is primary)
mx_p(t) = max_j |m_jp(t)|; alignment-based per-skill loss l_p = 1 - mx_p^2; weighted L_align(t) = sum_p pi_p l_p(t) (primary loss; MSE-based L_mse = sum pi_p MSE_p for A, exact population MSE for B is secondary).
* **T_p(0.5)** (spec): first t with mx_p >= 0.5 (linear interpolation between logs; 0 if aligned >= 0.5 at init, excluded from log fits). **T_p(0.9)**: same with 0.9. **T_p,x**: first t at which l_p has lost a fraction x (0.1, 0.5, 0.9) of its own total drop D_p = l_p(0) - min_t l_p. Sharpness = (T_p,0.9 - T_p,0.1)/T_mid with **T_mid = T_p,0.5 (drop midpoint) as the primary version** (defined for every skill, same definition for both models); the version with T_mid = T_p(0.5) of the spec is also given.
* **Learned**: A: final mx_p >= 0.9. B: max_j|m| plateaus well below 1 because several neurons share a skill, so "learned" = l_p dropped by >= 0.1 (a weak criterion; see Model B section).
* Fits: OLS of log T_p on log p over learned skills with T > 0 (SE = OLS SE; pooled over seeds, n = skill-seed pairs). Exponent: OLS of log L_align on log t over logged t in the window between the 3rd and 8th skill emergence (order statistics of T_p(0.5) over all 16 skills, **as specified**; "drop-window" = the same with the order statistics of T_p,0.5 over learned skills); SEs from the OLS are far too small (autocorrelated points), so the across-seed SE is used for the headline.

**A deviation that matters:** with 64 uniform neurons the best neuron per teacher already has max_j|m_jp(0)| = 0.36-0.55 (mean 0.44, 5 of 48 skill-seed pairs >= 0.5 at t = 0). T_p(0.5) is therefore a statistic of the initial draw and of noisy crossings
(e.g. skill 15 crosses 0.5 at 17k in seed 0 and is never learned), not of the transition; I report it because it is the spec's definition, but the drop-midpoint T_p,0.5 and T_p(0.9) are the clean measures of the transition. Spec estimate T_p ~ 2e4/pi_p assumed m_0 = d^(-1/2); the realised m_0 of the best neuron is ~0.44, so T is ~6x shorter: T_p,0.5 * eta * pi_p * m_0p^2 = **0.73 (IQR 0.70-0.77, n = 24 skills p <= 8)**, i.e. the exp 3 constant K ~ 0.6-0.7 (exp 3: K median 0.60 first arrivals, 0.73 later arrivals).

## Predictions vs measurements (alpha = 1.5, seeds 0-2; seeds 0-4 in brackets)
| ID | prediction | measured | verdict |
|---|---|---|---|
| P16 A | slope of log T_p vs log p = 1.5 +- 0.2 | T_p(0.5) (spec): **1.53 +- 0.22** (n=23) [1.66 +- 0.20, n=36]; T_p(0.9): 1.45 +- 0.07 (n=28); T_p,0.5 drop: 1.44 +- 0.08 (n=28) [1.52 +- 0.07, n=45]; same restricted to p <= 8: 1.66 +- 0.23 / 1.51 +- 0.08 / 1.50 +- 0.09; per-seed T_p,0.5 slopes 1.46, 1.26, 1.56 (SE 0.13-0.18). Init-corrected (log(T_p,0.5 m_0^2) vs log p): 1.52, 1.46, 1.60. | **met** (all within 1.5 +- 0.2 of the three estimates) |
| P16 B | slope 0.75 +- 0.2 (B rho = 0.0496, the capacity-matched readout) | T_p(0.5) (spec): 1.15 +- 0.24 (n=22) [1.39 +- 0.21]; **T_p,0.5 drop: 0.75 +- 0.08** (n=25) [0.82 +- 0.08, n=42]; pooled-alignment midpoint (supplementary, p <= 8-ish): 0.93 +- 0.06 (n=27). B rho = 0.1: 0.34 +- 0.08 (T_p,0.5, n=16); B rho = 1: no learning (below). | **met by the drop-based T_p,0.5; not by the spec's T_p(0.5)** (polluted by init alignments, see above); sensitive to rho |
| P16 A vs B | A steeper by ~0.75 | drop-midpoint: 1.44 - 0.75 = **0.69 +- 0.11**; spec T_p(0.5): 1.53 - 1.15 = 0.38 +- 0.33 (not significant) | **A steeper, clearly only with the drop-based T**. Kill (A slope < 1.0): **not triggered** |
| P17 | median sharpness of A < half of B's | A (T_mid = T_p,0.5): **1.08** (per seed 1.05, 1.10, 1.05; n = 28); B rho = 0.0496: **4.09** [5 seeds: 4.16]; ratio B/A = 3.8. With T_mid = T_p(0.5): A 3.08, B 11.0 (3.6x). B pooled-alignment version: 3.49 (3.2x). A, per-skill MSE curves instead of alignment: 1.09, 1.03, 0.88 (per seed). B rho = 0.1: 2.6 (2.4x). | **met** (ratio 3-4 >= 2), with the caveat that B's drops are small/partial and that A's absolute sharpness ~1 is just the cubic-escape shape (T_0.1 is already early because m_0 ~ 0.4) |
| P18 A | exponent of L_align in the 3rd-8th emergence window ~ 0.33 +- 0.15 | window as specified (T_p(0.5) order stats): **0.46, 0.42, 0.34, mean 0.40 +- 0.035 (SE over seeds)** [5 seeds 0.37 +- 0.03]; drop-window: 0.73, 0.85, 0.71 (**mean 0.76**); MSE-based (secondary): 0.33, 0.28, 0.20 (mean 0.27 +- 0.04) | within tolerance for the spec window and the MSE-based loss, **outside it (0.76) when emergence is defined by the drop midpoint**: the exponent over such a short window (a factor 3-8 in time, on a staircase) depends on the window definition |
| P18 B | ~ 0.67 +- 0.15 | L_align: **0.25, 0.16, 0.15, mean 0.19** [5 seeds 0.18 +- 0.02] (drop-window 0.22, 0.18, n/a); MSE-based: 0.98, 0.45, 1.35 (mean 0.92 +- 0.26: MSE falls to 1e-5, not a clean power law) | **not met** for the primary (alignment-based) loss, which saturates at 0.37-0.40 because neurons share skills; MSE-based exponent is of the predicted size but with a 3x seed scatter |
| P18 ratio | A/B ~ 1/2 (A decays slower) | alignment-based: A/B = 0.40/0.19 = **2.1 (reversed)**; MSE-based: 0.27/0.92 = 0.3 | **not met** (primary); met in direction for the secondary |

alpha = 1.0 (3 seeds each, not pre-registered; theory: A slope 1.0, B slope 0.5, exponent 0): A slope 0.88 +- 0.09 (T_p,0.5, n = 37; p <= 8: 1.16 +- 0.13; spec T_p(0.5): 0.68 +- 0.30), B (rho = 0.0566) 0.61 +- 0.08 (pooled-alignment 0.77 +- 0.06); sharpness A 1.03 vs B 4.84;
L_align exponents A 0.14 (0.13, 0.19, 0.11), MSE-based 0.04 (consistent with ~0 for alpha = 1 up to the window ambiguity), B 0.06; drop-window exponents 0.57, 0.57, 0.88 (A). A learns 10-14 of 16 skills before skills 1-8 are done (alpha = 1 spreads the times less).

## T_p table (steps; alpha = 1.5, seeds 0, 1, 2; "(x)" = skill not learned at the end of the run (value not used); "0" = aligned >= 0.5 at t = 0)
Model A (stops at 244k / 249k / 506k steps when skills 1-8 are all >= 0.95):

| p | pi_p | T_p(0.5) s0 | s1 | s2 | T_p,0.5 (drop midpoint) s0 | s1 | s2 | T_p(0.9) s0 | s1 | s2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.4717 | 3.2k | 2.7k | 0.9k | 8.9k | 8.1k | 6.9k | 11.5k | 10.7k | 9.0k |
| 2 | 0.1668 | 3.0k | 15.2k | 7.9k | 18.2k | 30.8k | 25.3k | 24.2k | 38.1k | 31.8k |
| 3 | 0.0908 | 24.5k | 17.6k | 13.7k | 53.7k | 45.8k | 45.9k | 66.5k | 57.8k | 58.1k |
| 4 | 0.0590 | 62.1k | 36.9k | 31.6k | 109.7k | 82.9k | 73.8k | 130.0k | 100.8k | 93.4k |
| 5 | 0.0422 | 79.5k | 14.6k | 0 | 139.2k | 79.1k | 68.2k | 171.4k | 101.9k | 90.9k |
| 6 | 0.0321 | 58.2k | 0 | 19.8k | 140.1k | 72.1k | 114.4k | 176.8k | 97.5k | 148.5k |
| 7 | 0.0255 | 0 | 58.8k | 60.8k | 111.0k | 159.9k | 178.0k | 147.2k | 204.5k | 223.1k |
| 8 | 0.0208 | 24.4k | 28.9k | 280.5k | 152.5k | 151.3k | 397.1k | 213.5k | 215.0k | 472.3k |
| 9 | 0.0175 | (109k) | (59k) | 0 | (171k) | (173k) | 152.4k | never | never | 199.7k |
| 10 | 0.0149 | (89k) | 0 | (23k) | (145k) | 144.0k | 251.0k | never | 211.9k | 312.9k |
| 11 | 0.0129 | never | never | (128k) | (0) | (3k) | 307.9k | never | never | 395.9k |
| 12-16 | | mostly never / spurious | | | (not learned) | | | never | never | never |

Skills learned (final mx >= 0.9) when the run stopped: seed 0: 8 (p = 1-8), seed 1: 9 (1-8, 10), seed 2: 11 (1-11); alpha = 1.0: 14, 10, 13. T_p increases with p overall, but not strictly: ordering inversions are set by the initial alignments
(T_p(0.9): seed 1 skill 6 at 97k before skills 4 and 5 at 101k and 102k; seed 0 skill 7 at 147k before skill 6 at 177k).

Model B, rho = 0.0496 (runs 3e6 steps; B reaches its plateau by ~1e5-4e5):

| p | T_p(0.5) s0 | s1 | s2 | T_p,0.5 (drop midpoint) s0 | s1 | s2 |
|---|---|---|---|---|---|---|
| 1 | 1.9k | 1.9k | 0.6k | 6.7k | 7.2k | 6.1k |
| 2 | 1.3k | 4.6k | 3.4k | 9.9k | 13.5k | 13.8k |
| 3 | 7.5k | 5.5k | 8.6k | 13.8k | 15.5k | 15.5k |
| 4 | 25.7k | 20.7k | 6.8k | 28.8k | 38.7k | 15.9k |
| 5 | 30.5k | 4.8k | 0 | 26.4k | 29.8k | 22.7k |
| 6 | 9.8k | 0 | 10.6k | 24.9k | 19.4k | 25.7k |
| 7 | 0 | 44.2k | 698k | 17.2k | 43.3k | (61k) |
| 8 | 11.4k | 8.8k | 84.9k | 26.4k | 39.3k | 48.9k |
| 9-16 | never/spurious | | | (not learned: the alignment of these skills hardly moves, see below) | | |

## Model B: why it is only weakly comparable (read before using the B numbers)
* **Spec-literal B (rho = 1, M = 64) does not learn at all.** Initial MSE 176, final 142 (3e6 steps); max_j|m_jp| stays 0.35-0.57 for every skill (`B_a1.5_rho1.0`: 57-62 of 64 neurons < 0.5 at the end; its "learned" counts in `summary.csv` are noise crossings and must not be used). Reason (exact): the population MSE
  is ||rho W^T W - T||_F^2 with T = sum a_p v_p v_p^T; with M = 64 > d = 32 unit-norm neurons, tr(rho W^T W) = rho M = 64 vs tr T = sum a_p = 3.2, so the trace constraint makes the target unreachable and gradient descent just keeps W^T W near isotropic.
* **rho = 0.1 (= Gamma of A)**: reaches MSE 0.326 (floor from the same trace mismatch: rho M = 6.4 vs 3.2) with skills beyond ~7 not emerging (4-7 skills pass the D >= 0.1 criterion); the isotropic baseline gives q_p(0) = rho sum_j m_jp^2 / a_p ~ 1 already for p >= 9 and the model then over-shoots (q_p(end) = 1.15-2.2).
* **rho = 0.0496 = sum_p a_p / M (capacity-matched; my choice, used as "B" in the tables)**: the target is exactly reachable and the MSE goes to ~1e-5. The solution is collective: the pooled alignment q_p = rho sum_j m_jp^2 / a_p goes from 0.14 (p = 1) ... 1.1 (p = 16) at t = 0 to **0.999 for every p** at the end,
  but individual neurons end with max_j|m_jp| = 0.5-0.9 (51 of 64 neurons in 0.5-0.9, none >= 0.9; skill 1 is shared by 23 neurons with |m| >= 0.5, 37-38 with |m| >= 0.3), so the alignment-based loss saturates at 0.37-0.40 and only skills with a_p well above the isotropic baseline (p <= 8) show a visible alignment drop (D >= 0.1: p = 1-8). Skills with p >= 9 are
  essentially already fit at init (q_p(0) ~ 1) so there is no emergence event for them. B's T_p for p <= 8 (and sharpness) are measured on drops of 0.1-0.6 in l_p, not on a 0 -> 1 transition. The pooled-alignment version (supplementary, `slope_Tq05`, `sharp_q_median` in `summary.csv`) uses q_p from its t = 0 value to 1 and is the cleaner B statistic: slope 0.93 +- 0.06, sharpness 3.49.
* **Choice of rho was made after I had looked at seed-0 pilots of B (rho = 0.1, 1, 0.03, 0.045: MSE and max alignments), but before computing any T_p slope, sharpness or exponent for B.** rho = 0.03 / 0.045 pilots (not in the tables) gave the same collective picture (MSE 0.098 / 0.005).

## Anomalies and observations
* **Skills never learned:** none among p = 1-8 in any A run (the stop rule needs them); beyond that A learned 0, 1, 3 of the skills 9-16 by the stop (alpha = 1.5), 6, 2, 5 (alpha = 1.0). Skills 9-16 sit at mx 0.23-1.0 (mostly 0.4-0.8) at the stop of the alpha = 1.5 runs: mid-transition when the run ends (I did not run on to check that they all complete). All B skills p >= 9 never "emerge" (above).
* **Several neurons per skill (A):** at the stop, neurons with |m| >= 0.5: skill 1 has 9-12, skills 2-3 have 6-10, skills 4-6 have 2-6, skills 7-8 have 2-5 (`runs/*/seed*.npz`, table in `analysis_stdout.txt`). Of 64 neurons at the stop: 34-44 are at >= 0.9, 13-18 in 0.5-0.9, 7-16 below 0.5 (still moving; the early stop leaves them mid-transition).
  The MSE floor (1 - Gamma n_p)^2 with n_p ~ 10 makes A's MSE-based L(t) end at 0.25-0.30 while L_align falls to 0.03-0.07: the MSE-based curves are floor-dominated, as in exp 3.
* **No neuron trapped at the origin:** min over neurons of max_p|m_jp| at the end is 0.29-0.31 (A, alpha = 1.5; 0.22-0.25 at init); with 16 teachers (half the dimensions) every neuron has substantial overlap with some teacher, unlike exp 3 (P = 2, M = 4).
* **Transition times are set by the initial alignment as much as by pi_p** (exp 3 lesson, again): K = T_p,0.5 eta pi_p m_0^2 is flat at 0.73 (0.65-0.85 for p <= 4; one outlier 1.32 for skill 8 in seed 2), and T_p vs p at fixed init would give the pi_p^-1 law exactly; the scatter of the raw T-vs-p fits is the scatter of m_0^2 (0.13-0.30).
* Hand-picked check not done: whether the first neuron to arrive on skill p is also the one that ends highest; I did not track neuron identity.
* **Staircases in L_align(t):** the weighted loss is a sum of steps ordered by p, visible as shoulders in `scaling_law.png`; the 3rd-8th window covers only a factor ~3-20 in time, so the exponents (0.34-0.46 for A in the spec window) are an average over ~5 steps, not a clean power law.

## Deviations from the spec
1. **Model B readout:** spec said 1; spec-literal is degenerate (above), so the primary B uses rho = sum a_p / M = 0.0496 (alpha = 1.0: 0.0566); rho = 1 and 0.1 are reported/stored too (tags `B_a1.5_rho1.0`, `B_a1.5_rho0.1`, `B_a1.0_rho0.1`).
2. **Eval set:** per-skill MSE of A on a *fixed* set of 512 contexts (same inputs for every skill, independent task scalar per skill and prompt) with 17 queries per context (`mse_h`; the single-query spec statistic is `mse_spec`), not 512 fresh prompts per skill at every log. B: exact population MSE (no per-skill MSE).
3. **Model A gradient** computed with the exact rewrite `grad_A_fast` (not `MultiModelA.loss_grad` directly); identical to 9e-17, ~3x faster. x is sampled fully in d = 32 as instructed.
4. **max_steps:** A capped at 1e6 (not 3e6) as a budget guard (a stalled run at 2.8 ms/step would cost 2.3 CPU-h); never reached (largest run 506k). B ran the full 3e6 (it plateaus after ~4e5).
5. **Extra runs not in the spec:** seeds 3, 4 (alpha = 1.5, both models, all three B readouts), alpha = 1.0 (3 seeds, A, B rho = 0.0566 and 0.1). Pre-registered tables = seeds 0-2 only.
6. **Windows / definitions:** the primary T_p(0.5) is the spec's but is dominated by initial alignments (above), so I report drop-midpoint and 0.9 versions beside it; "learned" for B is a drop criterion (>= 0.1); sharpness primary denominator is T_p,0.5 (drop) instead of T_p(0.5). I did not choose between definitions by outcome: all versions are shown in the table above.
7. Logging every 1000 steps; times between logs are linearly interpolated (A's T_1 ~ 1e3-3e3 is therefore resolved to about one log interval; its sharpness is not affected as it is computed from T_p,0.1/0.5/0.9 of the drop).

## Compute
CPU seconds measured inside the workers (`runs.csv`): A alpha = 1.5: 653, 671, 1364 s (seeds 0-2; 0.75 h) + 755, 1120 s (seeds 3, 4), total 1.27 h; A alpha = 1.0: 844, 527, 703 s (0.58 h); B: ~183 s per run, 21 runs = 1.07 h. **Sweeps total 2.92 h**; plus pilots (A 2e5 steps 553 s + 2e4 steps 56 s; four B pilots 402 s) and tests: 0.3 h;
**about 3.2 CPU-hours of the 4-hour cap.** 4 worker processes, 1 BLAS thread each (3 workers in the second batch). Throughput: A 2.8 ms/step (1.3 ms of it is generating the 132k Gaussians per step), B 0.06 ms/step. The pilot (A seed 0, 2e5 steps) showed skills 1-8 nearly finished by 2e5 steps (skill 8 at 0.87),
so extrapolating to ~2.5e5-5e5 steps per A run (0.2-0.4 CPU-h) needed no reduction of M (64 kept), of max_steps or of the seed count; the spec's 3e6-step estimate assumed m_0 = d^(-1/2) instead of the best-of-64 alignment.
