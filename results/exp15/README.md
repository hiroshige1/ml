# Experiment 15: does Adam make prompts per step matter when the drift is positive? (single-feature linear model, sigma_2)

Pre-registration: `docs/preregistration.md`, section "Does Adam make prompts per step matter when the drift is positive? (exp 15)" (P28a/b; committed in 727a72b before any run; not edited).
Code: `icl_additive/ablation_exp15.py` (grid and caps; imports the exp-12 `linear` model and runner `icl_additive/ablation_exp12.py`, torch fp32, CPU, 1 thread per process, 4 workers with a claim-file queue), analysis `icl_additive/analyze_exp15.py`.
Raw rows: `runs.csv` (24 runs; `max_dm_step` = largest single-step |Delta m|, `lr_adam`, `log_every`), per-run traces `traces/*.npz` (m every 10 steps for Adam, every 100 for SGD; gitignored), logs `logs/worker*.log`, `table.md`, `verdicts.json`.

## Setup (as pre-registered)

Linear single-feature model `yhat = sigma(w.x_q) mean_i y_i sigma(w.x_i)`, pinned readout Gamma = 1, sigma_2, d = 64, `m_0 = d^-1/2 = 0.125` exactly (`init_fixed`), N = 1024, `w <- w/|w|` after every step,
stop at `|m| >= 0.5` (checked every step). B in {16, 64, 256, 1024}, seeds {0, 1, 2}; 24 runs.
(a) Adam `lr = 1e-3` (default betas), full d = 64 data, cap 2e4 steps, m logged every 10 steps.
(b) SGD `eta = 1/d^2 = 1/4096`, no momentum, distribution-exact projected sampler (exp 12, `proj_sgd_grad`), cap 3e5 steps, m logged every 100 steps.
All 24 runs reached `|m| >= 0.5` (no censoring). Largest single-step |Delta m| 0.0088 (Adam) / 0.0084 (SGD): no instability.

## Table (T_0.5 in optimiser steps; flow time = T_0.5 x eta, SGD only)

| opt | B | T_0.5 steps (seed 0 / 1 / 2) | median T_0.5 | median flow time T eta | reached | CPU s (sum) |
|---|---|---|---|---|---|---|
| adam | 16 | 1019 / 5281 / 2537 | 2537 | n/a | 3/3 | 53 |
| adam | 64 | 379 / 1421 / 3313 | 1421 | n/a | 3/3 | 129 |
| adam | 256 | 215 / 216 / 268 | 216 | n/a | 3/3 | 109 |
| adam | 1024 | 204 / 130 / 146 | 146 | n/a | 3/3 | 305 |
| sgd | 16 | 41392 / 48188 / 32582 | 41392 | 10.11 | 3/3 | 74 |
| sgd | 64 | 34517 / 37356 / 34281 | 34517 | 8.43 | 3/3 | 159 |
| sgd | 256 | 35853 / 37054 / 37692 | 37054 | 9.05 | 3/3 | 799 |
| sgd | 1024 | 38539 / 37344 / 37789 | 37789 | 9.23 | 3/3 | 3378 |

## Scaling statistics

OLS of `log(median T_0.5)` on `log B` over the four B (n = 4, SE with 2 dof):

| opt | slope | SE | pooled-runs slope (12 points, supplementary) | T(16)/T(1024) | max/min of the four medians |
|---|---|---|---|---|---|
| Adam | **-0.754** | 0.144 | -0.709 +- 0.128 | **17.4** | 17.4 |
| SGD | -0.015 | 0.028 | -0.010 +- 0.019 | 1.095 | 1.199 |

SGD flow-time spread: medians 10.11 / 8.43 / 9.05 / 9.23 (B = 16 / 64 / 256 / 1024), max/min = 1.199; the within-cell seed range at B = 16 is 32582-48188 steps (a factor 1.48), at B = 64..1024 at most 1.09.
Adam per-seed scatter is large at small B (B = 16: 1019-5281, B = 64: 379-3313, a factor 5-9) and small at large B (B = 256: 215-268; B = 1024: 130-204).

## Verdicts

**P28a (Adam, B-dependence). FAILED as literally worded; the failure is by 0.004 in slope, and the qualitative hypothesis is strongly supported.**
Criterion (quoted): "median `T_0.5` decreases with `B`; OLS slope of `log T` vs `log B` over the four `B` in `[-0.75, -0.25]` (`sqrt(B)` SNR scaling gives `-0.5`) | slope in the band and `T(16)/T(1024) >= 2`"; default: "slope in `(-0.25, 0.25)`: Adam is `B`-invariant like SGD".
Measured: slope -0.7538 +- 0.144 (outside the band `[-0.75, -0.25]` by 0.0038, i.e. 0.03 SE), `T(16)/T(1024) = 17.4` (>= 2 met). The pass condition requires both, so P28a does not pass. The default (B-invariance) is rejected by a wide margin (3.5 SE below -0.25). The medians fall monotonically with B (2537, 1421, 216, 146). The measured slope is steeper than the sqrt(B) value -0.5 (about 1.8 SE from it), not shallower: per-step Adam progress grows faster than sqrt(B) here, and the log-log curve is not a straight line (local slopes -0.42 for B 16-64, -1.36 for 64-256, -0.28 for 256-1024; four points, three seeds, so these are not individually resolved).

**P28b (SGD, B-invariance of the flow time). FAILED on the second clause.**
Criterion (quoted): "median flow time `T eta` is `B`-invariant: slope in `(-0.15, 0.15)` and all four medians within 15% of each other (E5 control at `N=64` gave 0.5%)".
Measured: slope -0.015 +- 0.028 (inside), but max/min of the four medians = 1.199 (> 1.15). The excess is the B = 16 cell (median 10.11 vs 8.43-9.23 for the other three; the other three are within 1.095 of each other, and the B = 16 seeds span 32582-48188 steps). With three seeds per cell I cannot say whether the 20% is a small-B effect (SGD noise at B = 16 is largest) or seed noise; the pre-registered 15% band is exceeded either way. The slope itself shows no B trend.

Reading: the contrast the hypothesis predicted is present and large (Adam T falls 17x over B = 16..1024 while SGD flow time moves by at most 20% with no trend), but the two pre-registered pass criteria are both missed at the margin (P28a slope by 0.004, P28b spread by 4.9 percentage points). Per the pre-registration's consequence clause, "if P28a holds (and P28b)" the paper adds the Adam-SNR sentence and exp 16 is pre-registered; since P28a is formally failed, that wording is not triggered by the letter of the rule. This is for the decision-maker; nothing is claimed here beyond the measured numbers.

## CPU time

Fresh runs: 1.24 CPU-h (22 runs); 2 runs reused from before a container restart (SGD B = 256 seeds 0 and 1, 263 s and 273 s = 0.15 CPU-h); total of the 24 rows 1.39 CPU-h (Adam 0.17, SGD 1.22), cap 2 CPU-h. SGD B = 1024 alone: 3378 s (0.94 CPU-h), so seed 2 was kept in every cell. A 30/200-step `--timing` check before the relaunch (about 0.01 CPU-h) is not counted. Wall time of the relaunch about 30 min on 4 cores.

## Deviations

- **Container restart.** The first runner was killed after completing 2 of 24 runs (`linear_sgd_N1024_B256_s0`, `_s1`); their rows and traces are reused unchanged (the runs are deterministic per `SeedSequence([seed, 12, cell_code])`, same code path). Stale claim files and the logs of the killed workers were deleted and the other 22 runs were relaunched; `logs/worker*.log` therefore only cover the 22 fresh runs. To make the sweep resumable by rows rather than by claim files, `ablation_exp15.py` was edited after the restart to skip any job whose `rows/*.json` exists (no change to the runs).
- Float32 training; SGD uses the projected sampler (exp 12, validated there against full data), Adam uses full d = 64 data, as in exp 12. `runs.csv` has `lr_adam` NaN for SGD.
- OLS slopes use the medians of three seeds at four B values (2 dof): the SE is itself imprecise; the pooled-runs slope is given as a check and agrees.
- Everything else as pre-registered. No run was dropped or rerun.
