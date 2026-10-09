# Experiment 14: is the softmax + Adam escape noise-driven? B-dependence at N=16

Pre-registration: `docs/preregistration.md`, section "Is the softmax+Adam escape noise-driven? B-dependence at N=16 (exp 14)" (P27; committed in d984c95 before any run).
Code: `icl_additive/ablation_exp14.py` (imports and calls exp 13's `run` unchanged; torch fp32, CPU, 1 thread per process, 4 workers with a claim-file job queue), analysis `icl_additive/analyze_exp14.py`.
Raw rows: `runs.csv` (12 rows; `source=exp13` for the 4 reused runs, `source=exp14` for the 8 fresh ones; `censored_by` empty if |m| >= 0.5 was reached, else `cap`), `rows/*.json`, logs `logs/worker*.log`, `verdicts.json`, `table.md`; traces `traces/*.npz` (m every 100 steps; gitignored).
**CPU time:** 0.174 h for the 8 fresh runs (cap 1.5 h; sum of per-run `process_time`; the 4 reused exp 13 runs, 0.182 h, are not counted).

## Setup (as pre-registered)

Exactly exp 13's rbf + Adam cell: d = 64, sigma_2, `m_0 = d^-1/2 = 0.125`, pinned readout Gamma = 1, `beta = 0.3`, Adam lr 1e-3 (default betas), cap 1e5 steps, `w <- w/|w|` every step, stop at `|m| >= 0.5`, full d=64 data. N = 16, B in {16, 64, 256, 1024}, seeds {0, 1, 2}. Seeds 0-1 at B = 64 and B = 1024 are exp 13's runs (reused, not re-run); the other 8 are fresh.

## Table

CENS = no `|m| >= 0.5` within 1e5 steps (counted as infinity in the median). `T_0.5` is in optimiser steps, seeds 0 / 1 / 2.

| B | T_0.5 (s0 / s1 / s2) | escapes | escape fraction | median T_0.5 | final \|m\| (s0 / s1 / s2) | max \|m\| | source (s0 / s1 / s2) |
|---|---|---|---|---|---|---|---|
| 16 | 5464 / 35024 / 8010 | 3/3 | 1.00 | 8010 | 0.500 / 0.501 / 0.500 | 0.500 / 0.501 / 0.500 | exp14 / exp14 / exp14 |
| 64 | 26348 / 24709 / 10046 | 3/3 | 1.00 | 24709 | 0.500 / 0.500 / 0.500 | 0.500 / 0.500 / 0.500 | exp13 / exp13 / exp14 |
| 256 | 39189 / 18095 / 62631 | 3/3 | 1.00 | 39189 | 0.500 / 0.500 / 0.500 | 0.500 / 0.500 / 0.500 | exp14 / exp14 / exp14 |
| 1024 | 3561 / CENS / 60240 | 2/3 | 0.67 | 60240 | 0.501 / 0.022 / 0.500 | 0.501 / 0.131 / 0.500 | exp13 / exp13 / exp14 |

## Monotonicity checks and verdict

Criterion (quoted from the pre-registration): "the escape fraction within `10^5` steps is non-increasing in `B` and the median `T_0.5` (censored = infinity) is non-decreasing in `B`; in particular `B=16` escapes >= 2/3 and `B=1024` <= 2/3"; pass: "both monotonicities hold up to one tie"; default: "escape fraction increases with `B` (drift-driven, transformer-like)".

Reading of "up to one tie" (mine, fixed before looking at the numbers): a monotonicity holds if no adjacent pair (16-64, 64-256, 256-1024) reverses strictly and at most one adjacent pair is tied. The literal reading (no strict reversal, any number of ties) is reported alongside.

- Escape fraction by B (16, 64, 256, 1024): 1.00 -> 1.00 -> 1.00 -> 0.67. Strict increases at none; ties at [(16, 64), (64, 256)]. Non-increasing up to one tie: **no** (literal reading: yes).
- Median T_0.5 by B: 8010 -> 24709 -> 39189 -> 60240. Strict decreases at none; ties at none. Non-decreasing up to one tie: **yes** (literal reading: yes).
- B = 16 escapes >= 2/3: 3/3 -> **yes**; B = 1024 escapes <= 2/3: 2/3 -> **yes**.

**P27 FAILED on the declared (strict) reading of "up to one tie", but only through saturation; it HOLDS on the literal reading.** No pair reverses and both end-point conditions hold, but the escape fraction is tied in more than one adjacent pair (it is 3/3 at B = 16, 64, 256 and drops only at B = 1024), so the pre-registered "up to one tie" allowance is exceeded. The default (escape fraction increases with B) is NOT supported: the escape fraction never increases with B and the median T_0.5 rises monotonically with B. The data therefore lean towards the noise-driven reading (larger B is not helpful and is slower), but the only escape-fraction evidence is a single stuck seed at B = 1024 (2/3).

## Deviations

- None from the pre-registered design: the model, optimiser, cap, seeds and cells are exp 13's; `run` is imported, not copied. The only code-level difference is that the STOP-file location read by `stop_requested` is redirected to `results/exp14/` (no STOP file was used).
- Data streams: `torch.Generator` seeded from `SeedSequence([seed, 13, opt, N, B])` and the teacher / initial `w` from `[seed, 64, 13]`, exactly as in exp 13, so the reused runs are the same runs this script would produce for those (B, seed).
- The interpretation of "up to one tie" is mine (stated above); the literal reading is also shown.
