# Exp 16: transformer B-scan at N=16 (free readout), prediction P29 in `docs/preregistration.md`

- **Pre-registration:** `docs/preregistration.md`, block "Transformer B-scan at N=16: is the rescue graded like Adam's in the toy? (exp 16)" (commit 89391b3), prediction P29.
- **Code:** `icl_additive/transformer_exp8.py --run free 16 B SEED --out results/exp16/` (unchanged; exp-5 model/data/eval), analysis `icl_additive/analyze_exp16.py`. Logs in `logs/`, curves in `curves/*.npz` (not tracked).
- **CPU time:** 11704 s = 3.25 CPU-h for the 5 fresh runs (cap 4.5 CPU-h); 1 thread per process, at most 3 processes at once.
- **Deviations:** see the section at the end.

Setting as exp 5/8 free: d=256, d_model=64, MLP width 32, 2 pre-LN blocks, 4-head causal softmax attention, trainable linear readout, sigma_2 link, Adam 1e-3, eval every 200 steps on 2048 held-out prompts,
T_e = first eval MSE/E[y^2] < 0.5 (and < 0.25), early stop at < 0.25 or 30000 steps. `N=16`, seeds 0 and 1 (same teacher/init as exps 5/8 for the same seed). Rows with `source` exp5/exp8 are re-used, not re-run.

## Per-run results (`n/r` = not reached; censored runs enter medians at the cap 30000)

| B | seed | source | steps run | T_e(0.5) | T_e(0.25) | final MSE | max align | censored | CPU s |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 0 | exp5 | 30000 | n/r | n/r | 0.831 | 0.37 | max_steps | 414 |
| 64 | 0 | exp8 | 30000 | n/r | n/r | 0.831 | 0.37 | max_steps | 380 |
| 64 | 1 | exp5 | 30000 | n/r | n/r | 1.058 | 0.42 | max_steps | 408 |
| 64 | 1 | exp8 | 30000 | n/r | n/r | 1.058 | 0.42 | max_steps | 372 |
| 256 | 0 | exp16 | 30000 | n/r | n/r | 0.832 | 0.28 | max_steps | 1412 |
| 256 | 1 | exp16 | 30000 | n/r | n/r | 1.057 | 0.28 | max_steps | 1443 |
| 512 | 0 | exp16 | 13600 | 13400 | 13600 | 0.119 | 0.99 | - | 1291 |
| 512 | 1 | exp16 | 30000 | n/r | n/r | 1.056 | 0.33 | max_steps | 2700 |
| 1024 | 0 | exp5 | 15000 | 14800 | 14800 | 0.053 | 0.96 | - | 2971 |
| 1024 | 1 | exp5 | 10800 | 10600 | 10800 | 0.109 | 0.99 | - | 2170 |
| 2048 | 0 | exp16 | 11000 | 11000 | 11000 | 0.247 | 0.95 | - | 4858 |

## Medians over seeds (B=64: all 4 runs; `+` = the median is a censored lower bound)

| B | runs | reached 0.5 / 0.25 | median T_e(0.5) | median T_e(0.25) |
|---|---|---|---|---|
| 64 | 4 | 0 / 0 | 30000+ | 30000+ |
| 256 | 2 | 0 / 0 | 30000+ | 30000+ |
| 512 | 2 | 1 / 1 | 21700+ | 21800+ |
| 1024 | 2 | 2 / 2 | 12700 | 12800 |
| 2048 | 1 | 1 / 1 | 11000 | 11000 |

OLS of log(median T_e) on log B over B in {256, 512, 1024, 2048} (4 points, 2 dof):
- T_e(0.5): slope **-0.512 +- 0.074** (SE)
- T_e(0.25): slope -0.511 +- 0.072 (secondary; P29 is stated for T_e(0.5))

Per-seed T_e(0.5): B=256: s0 n/r, s1 n/r; B=512: s0 13400, s1 n/r; B=1024: s0 14800, s1 10600; B=2048: s0 11000.

## P29 verdict

Criteria (quoted): "`T_e` decreases with `B` gradually: `B=256` and `B=512` emerge (>=1/2 each) at times between `B=64`'s (inf) and `B=1024`'s, `B=2048` earlier than `B=1024`; OLS slope of `log T_e` (medians, censored = cap) vs `log B` over `B in {256,...,2048}` in `[-1.2, -0.3]`" - pass = all three clauses; default = "`B=256` and `B=512` both stuck 2/2 (threshold-like) or slope in `(-0.3, 0)`".

Operationalisation (mine): clause 1 = at each of B=256, 512 at least 1 of 2 seeds reaches T_e(0.5) < 30000 AND the median T_e(0.5) is above the B=1024 median (12700) and not above the cap; clause 2 = median T_e(0.5) at B=2048 below the B=1024 median; clause 3 = slope of median T_e(0.5) in [-1.2, -0.3].

| clause | observed | holds? |
|---|---|---|
| 1. B=256, B=512 emerge (>=1/2) with median T_e between B=1024's (12700) and infinity | B=256: 0/2 emerged, median 30000; B=512: 1/2 emerged, median 21700 | **NO** |
| 2. B=2048 earlier than B=1024 | median 11000 vs 12700 | yes |
| 3. slope in [-1.2, -0.3] | -0.512 +- 0.074 | yes |

**P29 FAILS: clause(s) 1.**
Clause 1 detail: B=256 emerges in 0/2 seeds, B=512 in 1/2. Clauses 2 and 3 alone would hold, but the graded-rescue claim needs all three. 
Default condition check: B=256 and B=512 both stuck 2/2: False; slope in (-0.3, 0): False.
n=2 seeds per cell, 4 points, 2 degrees of freedom: the slope SE is a rough guide only.

## CPU time

Fresh runs: 11704 s (3.25 CPU-h) summed over `cpu_s` of the exp16 rows (cap 4.5 CPU-h = 16200 s). Re-used rows cost nothing new.

## Deviations / what was not done

1. **B=2048 seed 1 was not run (dropped for budget, as pre-registered).** Timing from the logs: 0.046 s/step (B=256), 0.093 s/step (B=512), 0.45 s/step (B=2048), measured at the first logged point (step 1000; logs print every 1000 steps, not 500). Projection: B=256 s1 + B=512 s1 + B=2048 s0 were already running, and after B=256 (2x ~1.4k s), B=512 (1.3k + 2.7k s) and the B=2048 s0 run (4.9k s, emerged at 11000 steps) the cumulative CPU was 11.7k s of the 16.2k s cap; a second B=2048 run would need ~4.9k s if it emerged as early as seed 0 (13.5k s if stuck), more than the 4.5k s left, so it was dropped before launch. No run was killed or censored by the budget; the B=2048 cell therefore has 1 seed (median = that seed).
2. Per-step times were read from the 1000-step log lines (the script logs every 1000 steps), not from the first 500 steps.
3. Reused rows: B=64 = exp 5 cell c seeds 0,1 and exp 8 free re-runs seeds 0,1 (4 runs, all stuck at 30000 steps; the regression does not use B=64); B=1024 = exp 5 cell d seeds 0,1 (14800, 10600).
4. Clause 1 is operationalised (see verdict): "between B=64's and B=1024's" is read as median T_e(0.5) above the B=1024 median and finite-or-cap.
