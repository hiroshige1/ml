# Exp 5: regime-matched transformer test (d=256, MLP width 32), spec `docs/spec_exp5.md`, predictions P11a-c in `docs/preregistration.md`

- **Pre-registration:** `docs/preregistration.md`, section "Regime-matched transformer test (exp 5)" (P11a-c)
- **Code:** `icl_additive/{transformer_exp5,analyze_exp5}.py`
- **CPU time:** 5.98 CPU-h (5.89 in `summary.csv` + pilots/benchmark) (cap 6)
- **Deviations from pre-registration:** cell (b) has 1 seed not 3; cell (a) seed 2 censored at 17000 steps (both for the CPU cap); 1 thread x 2 processes; early stop at MSE<0.25 for later runs

Code: `icl_additive/transformer_exp5.py` (standalone copy of the exp-4 data/model/loop with d=256, MLP width 32; resumable via `summary.csv`; invoked as
`python3 -I icl_additive/transformer_exp5.py --cell NAME SEED` or `--grid --cells a,d,c,b,e,f`), `icl_additive/analyze_exp5.py` (tables, figures).
Exp 4 files/results untouched. Data: `summary.csv`, `curves/*.npz` (eval MSE and alignment proxy every 200 steps; block-2 fc2 / attn-out / head weight norms every 1000), `logs/`, `tables.md`, `figs/{Te_cells,curves,alignment}.png`.

Setting as in the spec: d=256, k*=2 (sigma_2), softmax attention, embedding Linear(257->64), 2 pre-LN blocks, 4 heads, MLP width 32, Adam lr 1e-3, fresh data every step, fp32,
eval every 200 steps on 2048 fixed held-out prompts (MSE, E[y^2]=1), T_e = first eval MSE < 0.5 (also < 0.25), max 30000 steps. `n/r` = not reached.
Seeds fix both the teacher direction v and the initial weights (so seed s is paired across cells); all cells use B as in the spec (including (d): B=1024, no reduction).

## Results

| cell | k* | N | B | seeds | T_e(0.5) per seed | T_e(0.25) per seed | reached 0.5 | steps run per seed | final eval MSE per seed |
|---|---|---|---|---|---|---|---|---|---|
| (a) | 2 | 256 | 64 | 3 | [4000, 2800, n/r] | [5000, 3000, n/r] | 2/3 | [12800, 3000, 17000*] | [0.089, 0.226, 0.788*] |
| (b) | 2 | 64 | 64 | **1** | [n/r] | [n/r] | 0/1 | [30000] | [0.985] |
| (c) | 2 | 16 | 64 | 3 | [n/r, n/r, n/r] | [n/r, n/r, n/r] | 0/3 | [30000, 30000, 30000] | [0.831, 1.058, 0.851] |
| (d) | 2 | 16 | 1024 | 3 | [14800, 10600, 11600] | [14800, 10800, 11800] | 3/3 | [15000, 10800, 11800] | [0.053, 0.109, 0.170] |
| (e) | 1 | 16 | 64 | 2 | [1400, 1200] | [1600, 1200] | 2/2 | [1600, 1200] | [0.093, 0.092] |
| (f) in-weight | 2 | 64 | 64 | 2 | [400, 200] | [400, 400] | 2/2 | [400, 400] | [0.132, 0.081] |

\* (a) seed 2 was **killed by me at the CPU cap** (step 17000-17999, last logged step 17000), still on the null plateau (eval MSE 0.787-0.788 = the zero-predictor level of its eval set for the whole run). It is *censored at 17000*, not a 30000-step failure. Its `summary.csv` row and `curves/a_s2.npz` were reconstructed from the log (1000-step resolution, no norm records); `wall_s`/`cpu_s` there are the last logged time (5449 s), not a measured end time. Early stop: (a) seed 0 stopped at MSE < 0.1; every run started after it (all but (a) seed 0 and (d) seed 0) stopped at MSE < 0.25 to save CPU, so `T_e(0.1)` in `summary.csv` is only meaningful for those two, and post-emergence curves of the other runs end at ~0.1-0.25.

Initial alignment proxy `max_j |cos(W_j, v)|` (block 1, 32 rows; identical across cells for a given seed because the seed fixes v and the init): **seed 0: 0.1261, seed 1: 0.1515, seed 2: 0.1759** (spec expected ~0.15-0.2; seed 0 is slightly below).
Compare exp 4 (d=32, 256 rows): 0.43-0.55. The init noise floor is therefore in the intended regime.

Per-seed T_e side by side with the pre-registered predictions:

| ID | prediction (rival = ours) | observed | held? |
|---|---|---|---|
| P11a | (a),(b) emerge within 30k for all seeds; (c) fails to emerge for >= 2/3 seeds | (a): 2/3 emerged (4000, 2800), seed 2 not by 17000 (censored, run stopped for budget). (b): the one seed run did **not** emerge in 30000 steps. (c): 0/3 emerged in 30000 steps | **No.** The (c) half held (3/3 stuck), but (b) failed to emerge (n=1 seed only) and (a) is not shown to emerge for all seeds (2/3 + 1 censored) |
| P11b (decisive) | (d) fails to emerge for >= 2/3 seeds while (a) emerges | (d) emerged in **3/3** seeds (14800, 10600, 11600), none stuck at 30000 (max run 15000 steps) | **No (rival rejected).** |
| P11c | k*=1 at N=16 emerges < 2000 steps; in-weight control emerges quickly | (e) 1400, 1200; (f) 400, 200 | **Yes** |

The default's own P11b statement ("(d) emerges at about the same step count as (a), or sooner") also did not hold in the seeds where (a) emerged: (d)/(a) = 14800/4000 = 3.7 (seed 0) and 10600/2800 = 3.8 (seed 1); medians 11600 vs 4000 (ratio 2.9; the (a) median is exact because seed 2 is censored above 4000). For seed 2, (d) emerged at 11600 while (a) had not by 17000 (the reverse order; (a)'s seed spread is wide, 2800 to > 17000). The scope rule "(d) emerges at <~ 2x the steps of (a) => single-neuron-only" was **not triggered by the numbers** (ratios 3.7, 3.8, median 2.9), and the rival branch ("(d) stuck") is rejected too; the outcome is intermediate: finite-N still delays emergence strongly at fixed B and fixed tokens/step, but number of prompts per step substitutes for N to a large degree, since (d) with 16x the prompts of (c) emerges 3/3 while (c) never does within 30000 steps. I do not draw a scope conclusion beyond this table.

## Observations

- Plateaus: every k*=2 run sits at the null level (the zero-predictor MSE of its eval set: 0.83 / 1.06 / 0.85 for seeds 0/1/2 at N=16, 0.79-1.14 elsewhere) and then drops sharply (see `figs/curves.png`). Plateau ends: (d) 10400-14600; (a) ~2500 and ~3500.
- Alignment (`figs/alignment.png`): in emerged runs the proxy rises from 0.13-0.18 to 0.9-0.99 around the drop (a_s1 jumps within the 200-step interval of the drop; a_s0 is already 0.5 at step 3000 while MSE is still 0.95). In stuck runs it wanders (0.1-0.4): maximum over the run: (c) 0.35-0.42, (b) 0.33, a_s2 0.15-0.39, and (d) seeds 0.15-0.4 during the plateau before the jump. For (d) the proxy stays 0.1-0.4 during the plateau (seed 1 reaches ~0.38 near step 7500) and jumps to 0.96-0.99 within one 200-step interval of the MSE drop.
- Readout magnitude (norm of the final head weight, recorded every 1000 steps; `norms` columns: block-2 fc2, block-2 attention out-projection, head): at init 0.56-0.58; it collapses to about 0.03-0.16 in the stuck runs (c_s0 0.05 at 15000 and 0.03 at 30000, b_s0 0.05, d_s0 0.07 at 8000, d_s1 0.08 at 10000) and grows again with emergence (d_s0 1.72 at 15000; a_s0 1.27 at 12000). The block-2 fc2/attn-out norms grow slowly in all runs (4.6 -> 7-11).
- The k*=1 control (e) takes 1200-1400 steps here (it showed T_e 60-200 in exp 4's d=32 wide net), i.e. also slower than in exp 4 but < 2000 as pre-registered; the in-weight control (f) emerged at 200-400.

## CPU time

Single-thread processes, two at a time (never more than 2 threads in total). Process CPU seconds summed from `summary.csv`: **21,192 s = 5.89 CPU-h** (wall sum 5.89 h; one thread each, so CPU ~= wall), plus pilots (2 x 400 steps at 2 threads, about 256 CPU-s) and a micro-benchmark (about 60 s): total about **5.98 CPU-h, at the 6 CPU-h cap**, which is why runs were stopped (below).
Per-cell CPU: (a) 10,766 s (incl. censored seed 2), (b) 1,581 s, (c) 1,262 s, (d) 7,504 s, (e) 37 s, (f) 41 s.
Measured speed (1 thread): (a) ~0.32 s/step, (d) B=1024 ~0.20 s/step, (b) ~0.05, (c)/(e) ~0.013.

## Deviations from the spec

1. **(b) has 1 seed (not 3)** and **(a) seed 2 is censored at 17000 steps (not 30000)**: both because of the 6 CPU-h cap (priority order a, d, c, b, e, f; (a) at 0.32 s/step made each long run very expensive, and a seed-2 plateau of > 17000 steps would have cost ~7000 more CPU-s to resolve). (b) seeds 1 and 2 were not run. The conclusions about (b) therefore rest on one seed, and about (a) on 2 emerged + 1 censored seed. I started the cheap cells f, e, b before c; this cost some budget that c/b seeds 1-2 would have needed.
2. **Threads: 1 per process, two processes** instead of `set_num_threads(4)` (spec) / 2 (instruction): same total of 2 threads; a pilot showed 2 threads were only ~1.5x faster than 1.
3. **Early stopping** at MSE < 0.1 for (a) seed 0 and (d) seed 0, at MSE < 0.25 for all later runs (stop value is a CLI flag `--stop-at`); non-emerging runs ran the full 30000 steps (b, c).
4. Only `T_e(0.5)` and `T_e(0.25)` are comparable across all emerged runs (see 3).
5. No reduction of B for (d): B=1024 was used for (d) (no reduction to 512 was needed: ~0.2 s/step, and it emerged at 10-15k steps).
6. The alignment proxy and norms follow the spec; the proxy ignores LN centring/scaling, the attention-branch contribution and the label coordinate (as in exp 4). It is logged every 200 steps with the eval (same cadence for all cells).

## Anomalies

- (a) seed 2 stays on its plateau for > 17000 steps (alignment drifts 0.15 -> 0.39 without emergence), whereas seeds 0/1 emerge at 4000/2800: the seed variance of (a) is more than a factor 6.
- (a) seed 0 post-emergence MSE is non-monotone (0.1-0.43 between 5000 and 12000) and only crosses 0.1 at 12800.
- The null plateau levels differ by seed because of the heavy-tailed `c*sigma_2` on a fixed 2048-prompt eval set (0.79 to 1.14), so the 0.5 / 0.25 thresholds are not the same fraction of the null across seeds (same as exp 4).
- Seeds share v and init across cells, so seed-paired comparisons across cells are valid but only 2-3 pairs exist.
