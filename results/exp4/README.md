# Exp 4: finite-context effect in a small transformer (spec `docs/spec_exp4.md`, predictions P8-P10 in `docs/preregistration.md`)

Code: `icl_additive/transformer_exp4.py` (model/data/train/CLI, resumable via `summary.csv`), `icl_additive/analyze_exp4.py` (tables and figures).
Data: `summary.csv` (31 runs), `curves/*.npz` (full eval curve + alignment per run), `fine/` (supplementary k*=1 / in-weight runs, eval every 20 steps),
`fine_align/` (supplementary seed-0 reruns with alignment every 200 steps), `tables.md` (these tables), logs `run_*.log`. Figures in `figs/`.
Setup as in the spec: 2-layer pre-LN transformer (d_model 64, 4 heads, MLP 256, ~100k params), Adam lr 1e-3, no warm-up/clipping, fresh data every step,
N*B = 8192 tokens/step, eval on 2048 fixed held-out prompts every 200 steps (MSE, normalised by E[y^2]=1), early stop at MSE < 0.1 or 30000 steps.
T_e = first eval step with MSE < 0.5 (also < 0.25, < 0.1 recorded). `n/r` = not reached within the run.

## Results: T_e per cell

| setting | k* | d | N | B | seeds | T_e(0.5) median | T_e(0.5) per seed | T_e(0.25) median | T_e(0.25) per seed | T_e(0.1) per seed | reached 0.5 | alignment proxy at last logged step (per seed; step 0 only if run < 1000 steps) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| softmax | 2 | 32 | 16 | 512 | 3 | 1000 | [800, 1000, 1200] | 1200 | [1000, 1400, 1200] | [1200, 1600, 1400] | 3/3 | 0.99, 0.95, 0.94 |
| softmax | 2 | 32 | 64 | 128 | 3 | 1400 | [1400, 3200, 1000] | 1800 | [1800, 3400, 1200] | [2400, 4000, 1600] | 3/3 | 0.99, 0.96, 0.94 |
| softmax | 2 | 32 | 256 | 32 | 3 | 2800 | [2600, 3200, 2800] | 3600 | [4000, 3600, 3400] | [6200, 4600, 5400] | 3/3 | 0.97, 0.99, 0.99 |
| softmax | 2 | 16 | 16 | 512 | 2 | 400 | [400, 400] | 400 | [400, 400] | [600, 600] | 2/2 | 0.83, 0.59 |
| softmax | 2 | 16 | 64 | 128 | 2 | 500 | [600, 400] | 700 | [800, 600] | [1200, 1000] | 2/2 | 0.99, 0.99 |
| softmax | 2 | 16 | 256 | 32 | 2 | 800 | [1000, 600] | 1300 | [1400, 1200] | [3400, 2800] | 2/2 | 0.99, 0.98 |
| softmax | 1 | 32 | 16 | 512 | 3 | 200 | [200, 200, 200] | 200 | [200, 200, 200] | [200, 200, 200] | 3/3 | 0.43, 0.55, 0.45 |
| softmax | 1 | 32 | 64 | 128 | 3 | 200 | [200, 200, 200] | 200 | [200, 200, 200] | [200, 200, 200] | 3/3 | 0.43, 0.55, 0.45 |
| softmax | 1 | 32 | 256 | 32 | 3 | 200 | [200, 200, 200] | 200 | [200, 200, 200] | [200, 200, 200] | 3/3 | 0.43, 0.55, 0.45 |
| linear | 2 | 32 | 16 | 512 | 2 | 2600 | [2200, 3000] | 2600 | [2200, 3000] | [2200, 3200] | 2/2 | 0.95, 0.99 |
| linear | 2 | 32 | 256 | 32 | 2 | inf | [n/r, n/r] | inf | [n/r, n/r] | [n/r, n/r] | 0/2 | 0.44, 0.53 |
| in-weight control softmax | 2 | 32 | 64 | 128 | 3 | 200 | [200, 200, 200] | 200 | [200, 200, 200] | [200, 200, 200] | 3/3 | 0.43, 0.55, 0.45 |

Seeds s share the teacher direction `v` and the initial weights across cells with the same d (the seed fixes both), so seed s is paired across N, k and attention type.

- P8 (d=16): median T_e N=16/64/256 = 400 / 500 / 800; ratio T_e(16)/T_e(256) = 0.50 (P8 requires >= 3); ratio T_e(256)/T_e(16) = 2.00
- P8 (d=32): median T_e N=16/64/256 = 1000 / 1400 / 2800; ratio T_e(16)/T_e(256) = 0.36 (P8 requires >= 3); ratio T_e(256)/T_e(16) = 2.80
- P9a (k*=1, d=32): median T_e N=16/64/256 = 200 / 200 / 200; max/min = 1.00 (P9 requires < 1.5)
- P9b (in-weight control k*=2, d=32, B=128, N=64 prompts of zeroed labels): T_e per seed [200, 200, 200] (median 200); in-context medians N=16: 1000, N=64: 1400, N=256: 2800; faster than every in-context cell (median-vs-median): True; faster than every in-context *run* (max iw seed < min over all in-context seeds): True
- P10 (d=32, N=16): softmax median 1000, linear median 2600 (per-seed linear [2200, 3000])
- P10 (d=32, N=256): softmax median 2800, linear median inf (per-seed linear [n/r, n/r])
- P10 ratio T_e(256)/T_e(16): linear inf, softmax 2.80
- Total: 31 runs, CPU time 15804 s = 4.39 CPU-h (process time summed over 4 threads), wall 1.12 h; diverged runs: 0


## Fine-resolution supplementary runs

| setting | N | B | T_e(0.5) per seed (eval every 20) | median |
|---|---|---|---|---|
| k*=1 softmax | 16 | 512 | [60, 60, 60] | 60 |
| k*=1 softmax | 64 | 128 | [60, 60, 80] | 60 |
| k*=1 softmax | 256 | 32 | [120, 100, 120] | 120 |
| in-weight control k*=2 | 64 | 128 | [40, 40, 60] | 40 |

- P9a at fine resolution: median T_e(k*=1) N=16/64/256 = 60 / 60 / 120; max/min = 2.00

## Zero-predictor MSE on the fixed eval sets

| k | d | N | seed | zero-predictor MSE on eval set |
|---|---|---|---|---|
| 2 | 32 | 16 | 0 | 0.975 |
| 2 | 32 | 16 | 1 | 1.152 |
| 2 | 32 | 16 | 2 | 1.006 |
| 2 | 32 | 256 | 0 | 1.145 |
| 2 | 32 | 256 | 1 | 0.969 |
| 2 | 32 | 256 | 2 | 1.120 |
| 2 | 32 | 64 | 0 | 1.120 |
| 2 | 32 | 64 | 1 | 0.842 |
| 2 | 32 | 64 | 2 | 0.932 |


## Pre-registered predictions: what held

| ID | Prediction | Outcome |
|---|---|---|
| P8 | T_e (k*=2, softmax, d=32) falls by >= x3 from N=16 to N=256 at fixed N*B; some N=16 seeds may not emerge | **Did not hold; the direction is reversed.** Median T_e = 1000 / 1400 / 2800 for N = 16 / 64 / 256 (T_e(16)/T_e(256) = 0.36). All 3 N=16 seeds emerged (800-1200); N=256 seeds 2600-3200, no overlap with N=16. Same direction at d=16 (400 / 500 / 800, 2 seeds). T_e(0.25) and T_e(0.1) give the same ordering. This is the pre-registered rival ("T_e increasing with N"), and the pre-registered **kill condition (N=256 slower than N=16) is met**. |
| P9a | k*=1: N-dependence < x1.5 | **Ambiguous / not supported at fine resolution.** At the specified eval every 200 steps all k*=1 runs have T_e = 200 (the first eval; ratio 1.0, but this is the resolution floor). Supplementary runs with eval every 20 steps: median T_e = 60 / 60 / 120 for N = 16 / 64 / 256 (ratio 2.0 > 1.5, same direction as k*=2: larger N slower), 3 seeds, values of 60-120 steps so only a few eval points. |
| P9b | In-weight control (k*=2) emerges faster than every in-context cell | **Held.** T_e = 200 for all 3 seeds at 200-step resolution (40, 40, 60 at 20-step resolution), versus 800 for the fastest in-context k*=2 run (d=32). |
| P10 | Linear attention: N-dependence present in both variants; record which is stronger | **Present in both, same direction (larger N slower); linear is slower than softmax at both N.** N=16: linear 2200, 3000 vs softmax 800-1200. N=256: linear did **not** emerge in 6000 steps in either seed (eval MSE 1.145 and 0.969 = the zero-predictor level of their eval sets), whereas all 3 softmax seeds emerged by 3200. With censoring the linear ratio T_e(256)/T_e(16) is only bounded below (> ~2.0-2.7 from 6000/3000 and 6000/2200), compared with 2.8 for softmax, so "stronger" cannot be decided from the ratio; the failure to emerge at N=256 is the evidence that linear attention is at least as N-dependent. |

Summary: of P8, P9 and P10, only the in-weight part of P9 and the "N-dependence in both variants" part of P10 held. The central P8 prediction failed, so by the
pre-registered rule the toy finite-context phenomenon (T_e decreasing with N at fixed N*B) does not transfer to this transformer at this scale; what is observed is
the opposite trend (T_e increases with N at fixed tokens/step). No explanation is tested here. One fact visible in the design: at fixed N*B, larger N means fewer
prompts per step (B = 512 -> 32), and only the query token of each prompt is supervised, so the number of supervised targets per step is B.

## Alignment proxy

Computed exactly as: row j of block 1's MLP first layer `fc1.weight[j]` (64-vector), multiplied elementwise by block-1 `ln2.weight` (LN gain), then composed with the
embedding restricted to the x-coordinates, `W_j = (fc1.weight[j] * ln2.weight) @ emb.weight[:, :d]` (a d-vector); reported value `max_j |cos(W_j, v)|` over the 256 rows,
logged every 1000 steps (and at step 0). Ignored: LN centring/variance scaling, the attention-branch contribution to the MLP input, and the label coordinate of the embedding.
At initialisation the proxy is 0.43-0.55 at d=32 (max over 256 random rows) and 0.59-0.83 at d=16, i.e. the noise floor is not small.

- Every emerged k*=2 softmax run ends with alignment 0.94-0.99 (d=32). The logged values are at or near the floor (0.43-0.57) until the eval MSE leaves the plateau, and are 0.80-0.99 at the first logged step after the drop (figure `figs/alignment.png`, left/middle).
- Runs that stop before step 1000 (all k*=1, in-weight, d=16 N=16 and N=64 cells) have only the step-0 value in their curves, so the table's "last logged step" value for those is the initialisation value and says nothing.
- The supplementary seed-0 reruns with alignment every 200 steps (`fine_align/`, d=32; they reproduce the main run exactly: same T_e and final MSE): N=16 alignment 0.43 / 0.43 / 0.51 / 0.57 / 0.97 at steps 0-800 together with MSE 1.35 -> 0.94 (step 600) -> 0.29 (step 800); N=64 alignment stays 0.42-0.48 through step 1200, then 0.91 at 1400 when the MSE falls from 1.12 to 0.42. So in these two runs the proxy stays at its floor during the plateau and jumps within one 200-step interval of the MSE drop. The N=256 seed-0 run was not repeated (budget).

## CPU time and budget

- Logged runs: summary.csv 31 runs = 4.39 CPU-h (4.39 h process time summed over threads; wall 1.12 h), `fine/` 0.15 CPU-h, `fine_align/` 0.16 CPU-h: 4.70 CPU-h.
- Not in the logs: a killed linear-attention N=256 seed-0 run (see deviations; about 16-20 min wall, roughly 1.1-1.3 CPU-h) and the 400-step pilots (about 0.1 CPU-h). Total about **6.0 CPU-h, i.e. at the 6 CPU-h cap** (my estimate; the unlogged parts are estimated from wall time). Process time summed over the 4 intra-op threads overstates the compute needed (1 thread ran at 0.7x the CPU-seconds per step in the benchmark).
- Pilot timing (4 threads): softmax N=256 about 60 ms/step, N=16 about 47 ms/step, linear N=256 about 137 ms/step (the `(QK^T)/N` path is not fused).

## Deviations from the spec

1. **Linear-attention cells (P10) capped at 6000 steps and 2 seeds** (not 30000 and 3). A first seed-0 linear N=256 run was started with the 30000-step cap, showed a flat eval MSE of 1.145 from step 400 to step 8000 (log `run_full_part1.log`), and was killed at about 8000-10000 steps before writing its curve/summary row (full 30000-step runs would have used >4 CPU-h each). It was re-run with a 6000-step cap (2.1x the slowest softmax N=256 seed); its result (not reached) is in `summary.csv`. The linear N=256 "no emergence" is therefore right-censored at 6000 steps, not 30000.
2. **d=16 (priority 4) run with 2 seeds**, not 3. Priorities 1-3 otherwise follow the spec (3 seeds, softmax d=32 k*=2 / k*=1 / in-weight). k*=1 and in-weight controls were run with 3 seeds.
3. **In-weight control** uses N=64, B=128 (N*B = 8192, same tokens/step), c = 1, context labels zeroed, query label sigma_2(<v,x_q>), 3 seeds.
4. **Eval resolution.** T_e for k*=1 and the in-weight control equals the first eval point (200) in every run, so at the specified resolution these cells carry no N information. I added supplementary runs with eval every 20 steps (separate `fine/summary.csv`; the main `summary.csv` is untouched). These use the same seeds, so their curves coincide with the main runs.
5. **Eval-set normalisation.** The MSE is divided by the population E[y^2] = 1 as specified, but c*sigma_2 is heavy-tailed, so the zero-predictor MSE on a fixed 2048-prompt eval set varies between 0.84 and 1.15 across seeds (table above); the pre-drop plateau values (0.84-1.15) are exactly that level. The thresholds 0.5 / 0.25 / 0.1 are therefore not at identical fractions of the null level across runs; the orderings in P8 are the same for all three thresholds.
6. Invocation is `python3 -I icl_additive/transformer_exp4.py ...` (path to the script) rather than `-m`, because `-I` removes the cwd from `sys.path`. The `--eval-every` and `--align-every` options exist only for the supplementary runs.
7. Linear attention applies `(QK^T)/N` (N = number of context tokens, so T-1) under the causal mask with no softmax; for context tokens early in the sequence this means fewer than N terms are summed.

## Anomalies / things to know

- No divergence (0 of 31 runs non-finite). No lr tuning was done: lr 1e-3 for all cells.
- Every k*=2 run shows a long plateau at the null level followed by a sharp drop; the plateau (read off the curves) lasts roughly 600-800 steps (N=16, d=32), 800-3000 (N=64; seed 1 is the 3000 one), 2200-3000 (N=256). The N=64 seed spread is large (T_e 1000 / 1400 / 3200), so the N=16 vs N=64 difference is within seed noise; the N=16 vs N=256 difference is not (3 seeds each).
- N=256 (B=32) post-emergence curves are noisy and non-monotone (e.g. seed 0 MSE rises from 0.13 to 0.35 around step 5600 before falling to 0.09); T_e is defined at the first crossing so this does not affect T_e(0.5), but it does affect the time to reach 0.1.
- Linear N=256 stays exactly at the null level for the whole run in both seeds (no partial progress, alignment 0.44-0.53 ~ initial).
- Only 3 (2 for d=16 and linear) seeds per cell; medians of 3 are coarse and T_e granularity is 200 steps.
- The k*=1 and in-weight tasks are learned within 40-120 steps, i.e. before the first 200-step eval, so they act as controls that "everything is learned fast when no context-dependent quadratic-feature composition is needed", nothing more.

## Figures

- `figs/Te_vs_N.png`: T_e vs N, per k and attention type, d=32 and d=16; small markers = seeds, in-weight control as the dashed horizontal line (shaded: seed range). Unreached linear N=256 seeds are drawn at 6000 (the cap), not 30000.
- `figs/curves_k2.png`: eval MSE curves for k*=2 softmax d=32, one panel per N, seeds overlaid (runs end at MSE < 0.1).
- `figs/alignment.png`: alignment proxy as logged (d=32, d=16) and the supplementary 200-step seed-0 curves overlaid with eval MSE.
