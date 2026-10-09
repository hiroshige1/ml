# Exp 8: transformer trap-vs-tradeoff test (pinned vs free readout), predictions P21a-c in `docs/preregistration.md`

Code: `icl_additive/transformer_exp8.py` (imports the exp-5 model/data/eval/alignment code unchanged; `transformer_exp5.py` is NOT modified; adds `--readout/proto` in {free, pinned1, pinned_small}),
`icl_additive/analyze_exp8.py` (summary.csv, tables.md, figs). Data: `summary.csv` (all 18 cells x seeds), `summary_raw.csv` (rows written by exp-8 runs only), `curves/*.npz` (eval MSE / alignment proxy every 200 steps; block-2 norms every 1000;
`head_norm` = readout weight norm at step 0, every 1000 steps, and at the end), `logs/`, `tables.md`, `figs/{Te_cells,curves}.png`.

Setting as exp 5: d=256, d_model=64, MLP width 32, 2 pre-LN blocks, 4-head causal softmax attention, linear readout, sigma_2 link (k*=2), c~N(0,1), Adam 1e-3, fp32, eval every 200 steps on 2048 held-out prompts, T_e = first eval MSE/E[y^2] < 0.5 (also < 0.25), early stop at < 0.25 or 30000 steps.
`pinned1`: readout weight and bias frozen at init (requires_grad=False, not in the optimizer; asserted bit-identical at the end of each run). `pinned_small`: init readout weight x0.1 (norm 0.056-0.058 vs 0.555-0.576), then frozen (bias frozen at its init value).
Seeds fix teacher direction and init exactly as in exp 5, so init alignment (0.1261 / 0.1515) and the init readout norm (0.555 / 0.576) are identical to exp 5 for the same seed; the free N=16,B=64 re-runs reproduce exp 5 (final MSE 0.831 / 1.058).

## Results (per seed; `n/r` = not reached; MSE normalised by E[y^2]=1)

| protocol | N | B | seed | src | steps run | T_e(0.5) | T_e(0.25) | reached 0.5 / 0.25 | final MSE | head norm init -> final | max align | censored | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pinned1 | 16 | 64 | 0 | exp8 | 30000 | n/r | n/r | 0 / 0 | 0.831 | 0.555 -> 0.555 | 0.32 | max_steps | 329 |
| pinned1 | 16 | 64 | 1 | exp8 | 30000 | n/r | n/r | 0 / 0 | 1.060 | 0.576 -> 0.576 | 0.45 | max_steps | 330 |
| pinned1 | 16 | 1024 | 0 | exp8 | 8600 | 8400 | 8600 | 1 / 1 | 0.052 | 0.555 -> 0.555 | 1.00 | - | 1512 |
| pinned1 | 16 | 1024 | 1 | exp8 | 5800 | 5800 | 5800 | 1 / 1 | 0.233 | 0.576 -> 0.576 | 0.96 | - | 1022 |
| pinned1 | 256 | 64 | 0 | exp8 | 7000 | 6200 | 7000 | 1 / 1 | 0.243 | 0.555 -> 0.555 | 0.94 | - | 1988 |
| pinned1 | 256 | 64 | 1 | exp8 | 2800 | 2400 | 2800 | 1 / 1 | 0.235 | 0.576 -> 0.576 | 0.72 | - | 822 |
| pinned_small | 16 | 64 | 0 | exp8 | 30000 | n/r | n/r | 0 / 0 | 0.831 | 0.056 -> 0.056 | 0.44 | max_steps | 339 |
| pinned_small | 16 | 64 | 1 | exp8 | 30000 | n/r | n/r | 0 / 0 | 1.057 | 0.058 -> 0.058 | 0.33 | max_steps | 342 |
| pinned_small | 16 | 1024 | 0 | exp8 | 7600 | 6600 | 7600 | 1 / 1 | 0.244 | 0.056 -> 0.056 | 0.99 | - | 1456 |
| pinned_small | 16 | 1024 | 1 | exp8 | 18400 | 14600 | 18400 | 1 / 1 | 0.246 | 0.058 -> 0.058 | 1.00 | - | 3589 |
| pinned_small | 256 | 64 | 0 | exp8 | 17000 | n/r | n/r | 0 / 0 | 1.137 | 0.056 -> 0.056 | 0.31 | killed_17000_cpu_budget | 5170 |
| pinned_small | 256 | 64 | 1 | exp8 | 7000 | 4600 | 7000 | 1 / 1 | 0.248 | 0.058 -> 0.058 | 0.97 | - | 1957 |
| free | 16 | 64 | 0 | exp8 | 30000 | n/r | n/r | 0 / 0 | 0.831 | 0.555 -> 0.028 | 0.37 | max_steps | 380 |
| free | 16 | 64 | 1 | exp8 | 30000 | n/r | n/r | 0 / 0 | 1.058 | 0.576 -> 0.051 | 0.42 | max_steps | 372 |
| free | 16 | 1024 | 0 | exp5 | 15000 | 14800 | 14800 | 1 / 1 | 0.053 | 0.555 -> 1.724 | 0.96 | - | 2971 |
| free | 16 | 1024 | 1 | exp5 | 10800 | 10600 | 10800 | 1 / 1 | 0.109 | 0.576 -> 0.079 | 0.99 | - | 2170 |
| free | 256 | 64 | 0 | exp5 | 12800 | 4000 | 5000 | 1 / 1 | 0.089 | 0.555 -> 1.270 | 0.95 | - | 4356 |
| free | 256 | 64 | 1 | exp5 | 3000 | 2800 | 3000 | 1 / 1 | 0.226 | 0.576 -> 0.828 | 0.95 | - | 961 |


Free rows with src=exp5 are the exp-5 results re-used (cells c, d, a, seeds 0-1; stop rule identical except exp-5 seed a_0 ran to MSE<0.1; head norm from exp-5 `norms[:,2]`, logged every 1000 steps). The two free N=16,B=64 rows are fresh exp-8 re-runs (code-path check; they match exp 5).
`killed_17000_cpu_budget`: pinned_small N=256,B=64 seed 0 was **killed by me** (SIGTERM) at step ~17000 for budget; it was still on the null plateau (eval MSE 1.137-1.138 throughout, alignment proxy <= 0.31). It is **censored at 17000, not a 30000-step failure**. Its `curves/pinned_small_N256_B64_s0.npz` and summary row were reconstructed from the log (1000-step resolution, no 200-step eval, no block norms); its `cpu_s` (5170) is an estimate from the last log time.

Compact view, T_e(0.5) per seed (T_e(0.25) is within 0-3800 steps of it, in `summary.csv`):

| protocol | N=16, B=64 | N=16, B=1024 | N=256, B=64 |
|---|---|---|---|
| pinned1 (readout norm 0.56 fixed) | n/r, n/r (30000) | 8400, 5800 | 6200, 2400 |
| pinned_small (0.1x, fixed) | n/r, n/r (30000) | 6600, 14600 | 4600, n/r (censored at 17000) |
| free (= exp 5) | n/r, n/r (30000) | 14800, 10600 | 4000, 2800 |

## Pre-registered predictions

| ID | prediction | observed | held? |
|---|---|---|---|
| P21a (pinned1) | N=16 stuck at both B=64 and B=1024 (the trap); N=256 emerges | B=64: stuck 2/2 (30000 steps). **B=1024: emerged 2/2 (T_e 8400, 5800)**. N=256: emerged 2/2 (6200, 2400) | **No.** The N=256 half and the B=64 half held; the central claim (no B-compensation under a pinned O(1) readout) failed: B=1024 rescues it, and faster than the free readout (8400/5800 vs 14800/10600, n=2) |
| P21b (pinned_small) | N=16,B=64 emerges (rescued), within x2 of N=256 | N=16,B=64: **stuck 2/2** (30000 steps, MSE 0.831 / 1.057 = null level; alignment proxy max 0.44). N=256,B=64: 1 emerged (4600), 1 censored on the plateau at 17000 | **No.** A 10x smaller pinned readout does not rescue N=16,B=64 |
| P21c (free) | replicates exp 5: N=16,B=64 stuck; N=16,B=1024 emerges ~3x later than N=256 | N=16,B=64 stuck 2/2 (re-run, reproduces exp 5). N=16,B=1024 / N=256: 14800, 10600 vs 4000, 2800 (exp 5) = 3.7x, 3.8x per seed, median 12700 vs 3400 = 3.7x | **Yes** (ratio 3.7, a bit above the nominal 3; exp 5 data re-used, not re-run) |

Kill criterion (P21a fails **and** P21b fails): **met.** Both failed, so by the pre-registered rule the readout-context mechanism does not transfer to this transformer even qualitatively, and the paper keeps Sec. 5 as observations only. I draw no further conclusion.
Statements limited by n=2 seeds per cell: the within-protocol timing differences (e.g. pinned1 vs free at N=16,B=1024: 8400/5800 vs 14800/10600 - seed-paired, same direction both seeds, but two pairs only; pinned_small N=16,B=1024 has seed spread 6600 vs 14600, as large as the protocol difference).

## Observations (descriptive)

- Every run, whatever the protocol, sits on the null plateau (MSE = zero-predictor level of the eval set: 0.83 / 1.06 seeds 0/1 at N=16; 1.14 / 0.85 at N=256 seeds 0/1) and then drops sharply; plateau lengths are what T_e measures (`figs/curves.png`).
- Frozen readouts do not shrink the plateau-length dependence on B at N=16: B=64 stuck, B=1024 emerges, under pinned1, pinned_small and free alike. N=16,B=64 was never rescued by the readout scale (0/6 runs across the three protocols, 30000 steps each).
- Free-readout norm (`head_norm`) collapses from 0.555/0.576 to 0.03-0.05 in the stuck N=16,B=64 re-runs (as in exp 5) and grows after emergence (1.72, 1.27, 0.83, 0.08 in the exp-5 emerged runs). Pinned runs stay at their init norm by construction (0.555/0.576; 0.0555/0.0576).
- Alignment proxy: as in exp 5, stuck runs wander at 0.1-0.45; emerged runs reach 0.7-1.0 around the drop (max over run in the table).
- pinned_small emerged runs end the post-drop descent more slowly (MSE crosses 0.5 at 4600/6600/14600 but then needs 1000-3800 further steps to reach 0.25), consistent with an effective small output scale, but I did not test this.

## CPU time

All runs: 1 thread per process (`torch.set_num_threads(1)`), two processes at a time (one extra short single-thread run at the end). Sum of `cpu_s` over the 14 exp-8 runs: **19,609 s** (of which 5,170 s is the estimated cost of the killed pinned_small N=256 seed 0), plus the pilot (300 steps of pinned1 N=16,B=1024 and 300 steps of pinned_small N=16,B=64: 73 s). Total **about 19,680 s = 5.47 CPU-h** (cap 6). Exp-5 results re-used for 4 free cells cost nothing new (they cost 10.4k s in exp 5).
Measured speed (1 thread): N=16,B=1024 ~0.17-0.20 s/step (pilot 0.227 s/step incl. startup); N=256,B=64 ~0.28-0.30 s/step; N=16,B=64 ~0.011-0.013 s/step.

## Deviations / what was not done

1. **Free cells N=16,B=1024 (2 seeds) and N=256,B=64 (2 seeds) are exp-5 results (cells d and a, seeds 0-1), not re-run** (the budget allowed it only for the cheap N=16,B=64 cells, which were re-run and reproduce exp 5). Exp-5's a_s0 and d_s0 ran to MSE<0.1 instead of <0.25; T_e(0.5)/(0.25) are unaffected. So 14 of the 18 cells x seeds are fresh runs (12 pinned + 2 free).
2. **pinned_small N=256,B=64 seed 0 is censored at 17000 steps** (killed, reconstructed from log; see above) because stuck 30000-step N=256 runs cost ~8400 CPU-s and the lower-priority pinned_small N=16,B=1024 cell had to be afforded. No seed cuts were needed otherwise: all cells have 2 seeds.
3. Pilot (300 steps of the B=1024 cell) showed ~0.23 s/step (about 6800 CPU-s if stuck for 30000 steps); the pinned1 B=1024 runs emerged at 5800-8600 steps, which is why all cells fit.
4. Early stop at MSE < 0.25 for every run (so T_e(0.1) is not recorded); non-emerging runs ran the full 30000 steps.
5. Readout norm is logged at init and every 1000 steps for all protocols (not only free); for the exp-5 free rows it comes from exp 5's `norms` column, same cadence.
6. Per-run stop-file (`STOP_<name>`) was added to the script mid-experiment; it only affected runs started after the edit and was not used. Runs were driven by a claim-file queue with 2 worker processes (a third worker was started after one was killed).
7. The readout in this architecture multiplies the final-LN residual stream, not the context statistic, as in the pre-registration caveat; nothing here tests the toy's Gamma directly.
