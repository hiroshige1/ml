# Exp 11: attention output scale as the Gamma-analogue (frozen / x0.1-frozen `attn.out`), predictions P24a-c in `docs/preregistration.md`

Pre-registration: `docs/preregistration.md`, section "Transformer: attention output scale as the Gamma-analogue (exp 11, backlog N11)" (committed ff2cc7d before any run); spec `docs/spec_exp11.md`.
Code: `icl_additive/transformer_exp11.py` (imports the exp-5 model/data/eval/alignment code unchanged; `transformer_exp5.py` and `transformer_exp8.py` are NOT modified; `--proto` in {O1, Os}), `icl_additive/analyze_exp11.py` (summary.csv, tables.md, figs). Data: `summary.csv` (exp-11 rows + the exp-8 `free` rows, reused not re-run), `summary_raw.csv` (exp-11 runs only), `curves/*.npz` (gitignored; eval MSE / alignment every 200 steps, `ao_norm` = Frobenius norm of both `attn.out` weights at step 0, every 1000 steps, and at the end), `logs/`, `tables.md`, `figs/{Te_cells,curves}.png`.
CPU time: **21,664 s = 6.02 CPU-h** (11 runs 21,593 s + 71 s pilot; cap 6.0, over by 64 s, see Deviations). Deviations: (1) Os N=16,B=1024 seed 1 not run; (2) Os N=16,B=1024 seed 0 and Os N=256,B=64 seed 1 killed for CPU budget, censored on the plateau (not failures); see below.

Setting as exp 5/8: d=256, d_model=64, MLP width 32, 2 pre-LN blocks, 4-head causal softmax attention, trainable linear readout, sigma_2 link (k*=2), c~N(0,1), Adam 1e-3, fp32, eval every 200 steps on 2048 held-out prompts, T_e = first eval MSE/E[y^2] < 0.5 (also < 0.25), early stop at < 0.25 or 30000 steps. Seeds 0, 1 fix teacher direction and init exactly as in exps 5/8 (init alignment 0.1261 / 0.1515 identical).
`O1`: weight and bias of `blocks[0].attn.out` and `blocks[1].attn.out` frozen at init (`requires_grad=False`, not in the optimizer; asserted bit-identical at the end of every run, assertion passed for all 11 runs). `Os`: both `attn.out` weights x0.1 at init (norm 4.6 -> 0.46 per block), bias unchanged, then weight and bias frozen. Final readout trainable in both. `free` rows are the exp-8 `free` rows (`results/exp8/summary.csv`; N=16,B=1024 and N=256,B=64 are exp-5 results re-used by exp 8); they did not log the block-0 `attn.out` norm (`n/l`), the block-1 value is exp-5's `norms[:,1]` and shows the free norm growing from 4.6 to 5.6-10.8.

## Results (per seed; `n/r` = not reached; MSE normalised by E[y^2]=1)

| protocol | N | B | seed | src | steps run | T_e(0.5) | T_e(0.25) | reached 0.5 / 0.25 | final MSE | attn.out norm block 0 init -> final | attn.out norm block 1 init -> final | max align | censored | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O1 | 16 | 64 | 0 | exp11 | 30000 | n/r | n/r | 0 / 0 | 0.831 | 4.643 -> 4.643 | 4.617 -> 4.617 | 0.41 | max_steps | 335 |
| O1 | 16 | 64 | 1 | exp11 | 30000 | n/r | n/r | 0 / 0 | 1.058 | 4.596 -> 4.596 | 4.615 -> 4.615 | 0.38 | max_steps | 341 |
| O1 | 16 | 1024 | 0 | exp11 | 15200 | 15000 | 15200 | 1 / 1 | 0.087 | 4.643 -> 4.643 | 4.617 -> 4.617 | 0.98 | - | 2798 |
| O1 | 16 | 1024 | 1 | exp11 | 26600 | 26400 | 26600 | 1 / 1 | 0.161 | 4.596 -> 4.596 | 4.615 -> 4.615 | 0.99 | - | 4864 |
| O1 | 256 | 64 | 0 | exp11 | 13800 | 13600 | 13800 | 1 / 1 | 0.181 | 4.643 -> 4.643 | 4.617 -> 4.617 | 0.97 | - | 4024 |
| O1 | 256 | 64 | 1 | exp11 | 7400 | 7200 | 7400 | 1 / 1 | 0.211 | 4.596 -> 4.596 | 4.615 -> 4.615 | 0.95 | - | 2198 |
| Os | 16 | 64 | 0 | exp11 | 30000 | n/r | n/r | 0 / 0 | 0.831 | 0.464 -> 0.464 | 0.462 -> 0.462 | 0.43 | max_steps | 343 |
| Os | 16 | 64 | 1 | exp11 | 30000 | n/r | n/r | 0 / 0 | 1.058 | 0.460 -> 0.460 | 0.461 -> 0.461 | 0.32 | max_steps | 340 |
| Os | 16 | 1024 | 0 | exp11 | 8000 | n/r | n/r | 0 / 0 | 0.831 | 0.464 -> 0.464 | 0.462 -> 0.462 | 0.30 | killed_8000_cpu_budget | 1541 |
| Os | 256 | 64 | 0 | exp11 | 6000 | 5400 | 6000 | 1 / 1 | 0.214 | 0.464 -> 0.464 | 0.462 -> 0.462 | 0.93 | - | 1761 |
| Os | 256 | 64 | 1 | exp11 | 10800 | n/r | n/r | 0 / 0 | 0.844 | 0.460 -> 0.460 | 0.461 -> 0.461 | 0.30 | killed_10800_cpu_budget | 3046 |
| free | 16 | 64 | 0 | exp8 | 30000 | n/r | n/r | 0 / 0 | 0.831 | n/l | 4.617 -> 8.487 | 0.37 | max_steps | 380 |
| free | 16 | 64 | 1 | exp8 | 30000 | n/r | n/r | 0 / 0 | 1.058 | n/l | 4.615 -> 10.840 | 0.42 | max_steps | 372 |
| free | 16 | 1024 | 0 | exp8 (exp5 row) | 15000 | 14800 | 14800 | 1 / 1 | 0.053 | n/l | 4.617 -> 7.602 | 0.96 | - | 2971 |
| free | 16 | 1024 | 1 | exp8 (exp5 row) | 10800 | 10600 | 10800 | 1 / 1 | 0.109 | n/l | 4.615 -> 6.268 | 0.99 | - | 2170 |
| free | 256 | 64 | 0 | exp8 (exp5 row) | 12800 | 4000 | 5000 | 1 / 1 | 0.089 | n/l | 4.617 -> 10.037 | 0.95 | - | 4356 |
| free | 256 | 64 | 1 | exp8 (exp5 row) | 3000 | 2800 | 3000 | 1 / 1 | 0.226 | n/l | 4.615 -> 5.589 | 0.95 | - | 961 |

`killed_<step>_cpu_budget`: killed by me (graceful stop file, at the next eval) because the 6 CPU-h cap was reached; both runs were still on the null plateau (eval MSE 0.831 / 0.844 = zero-predictor level, alignment proxy <= 0.30). They are **censored at that step, not 30000-step failures**; `Os N=16,B=1024 seed 0` was killed at 8000 and `Os N=256,B=64 seed 1` at 10800. `Os N=16,B=1024 seed 1` was never run (dropped for budget, see Deviations). The frozen `attn.out` norms are constant by construction (`init -> final` equal).

T_e(0.5) per seed (T_e(0.25) is within 0-200 steps of it in all emerged exp-11 runs):

| protocol | N=16, B=64 | N=16, B=1024 | N=256, B=64 |
|---|---|---|---|
| O1 | n/r (30000), n/r (30000) | 15000, 26400 | 13600, 7200 |
| Os | n/r (30000), n/r (30000) | n/r (8000) | 5400, n/r (10800) |
| free | n/r (30000), n/r (30000) | 14800, 10600 | 4000, 2800 |


## Pre-registered predictions

| ID | pre-registered criterion | observed | held? |
|---|---|---|---|
| P24a (O1) | "`N=16` stuck at **both** `B=64` and `B=1024` (trap: no `N`-`B` tradeoff); `N=256` emerges" (default: "`B=1024` rescues as in exps 5, 8") | N=16,B=64: stuck 2/2 (30000 steps, MSE 0.831 / 1.058 = null level, max align 0.41 / 0.38). **N=16,B=1024: emerged 2/2 (T_e 15000, 26400)**. N=256,B=64: emerged 2/2 (13600, 7200) | **No.** The B=64 and N=256 halves held; the central claim (no B-compensation under a frozen O(1) attention output) failed: B=1024 rescues N=16, as the default said. |
| P24b (Os) | "`N=16, B=64` **emerges** (rescued; free/pinned readouts were stuck 2/2-3/3) within x2 of `N=256`" (default: "still stuck") | N=16,B=64: **stuck 2/2** (30000 steps, MSE 0.831 / 1.058, max align 0.43 / 0.32). N=256,B=64: emerged 5400 (seed 0), seed 1 censored on the plateau at 10800 | **No.** Scaling `attn.out` down x10 does not rescue N=16,B=64 (0/2); the default held. |
| P24c (O1 vs F at N=256) | "emergence time within x1.5 of free (an O(1) frozen scale does not hurt at long context)" | O1 13600, 7200 vs free 4000, 2800: per-seed ratios **3.4, 2.6**; medians 10400 vs 3400 = 3.1 | **No.** A frozen `attn.out` is 2.6-3.4x slower than free at N=256 (same direction in both seeds, n=2). |

Kill criterion (P24a fails **and** P24b fails): **met.** Both failed, so by the pre-registered rule neither the readout (exp 8) nor the attention-output scale carries the mechanism in this transformer; the paper's Sec. 5 states both negative results and names the embedding/feature norm (the tied analogue) as untested. I draw no further conclusion; the LayerNorm caveat (the next block renormalises the sum) means a null for a scale of `attn.out` is not a test of the toy's exact Gamma multiplier.

## Observations (descriptive, n=2 per cell)

- N=16,B=64 was never rescued: 0/4 runs across O1 and Os (plus 0/2 free, 0/6 pinned readouts in exps 5/8), every one at the null-MSE plateau with alignment proxy <= 0.43.
- B=1024 rescues N=16 under O1 with a spread between seeds (15000 vs 26400) larger than the difference to free (14800 / 10600): O1/free ratio 1.0 and 2.5 per seed. Os N=16,B=1024 seed 0 was still on the plateau at 8000 (censored; the exp-8 pinned_small seeds emerged at 6600 and 14600, so this is not informative about failure).
- At N=256 both frozen protocols emerge but later than free (O1 7200 / 13600 vs 2800 / 4000; Os 5400 and one seed censored on the plateau at 10800 vs 2800 / 4000). Freezing `attn.out` slows emergence there; whether this is a scale effect or just a loss of trainable parameters is not separated (O1 vs Os at N=256 are within seed spread: 7200-13600 vs 5400 / >10800).
- Free-run block-1 `attn.out` norm grows from 4.6 to 5.6-10.8 over training (including in stuck N=16,B=64 runs: 8.5, 10.8), i.e. the free model does move this scale; O1/Os hold it at 4.6 / 0.46.

## CPU time

All runs: 1 thread per process (`torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`), 4 worker processes (claim-file queue, priority order of the spec). Sum of `cpu_s` over the 11 exp-11 runs: **21,593 s**, plus the pilot (300 steps of O1 N=16,B=1024, ran 400 steps: 71 s, 0.178 s/step; run in a scratch directory, not included in the results) = **21,664 s = 6.02 CPU-h** (cap 6.0). Per-step cost with 4 concurrent processes: N=16,B=1024 ~0.18 s/step, N=256,B=64 ~0.29 s/step, N=16,B=64 ~0.011 s/step. Free rows reused from exp 8 cost nothing new.

## Deviations / what was not done

1. **Os N=16,B=1024 seed 1 was not run** (budget; lowest-priority expensive cell). I decided this after the first four runs, when O1 was still on the plateau at N=256 and N=16,B=1024 and the worst-case projection (stuck 30000-step runs at 5.4k s for B=1024 and 8.7k s for N=256) was far above 6 CPU-h. The pilot alone projected roughly the exp-8 level (5.5 CPU-h) if runs emerged; it did not bound the stuck case. Decision logged in `logs/decisions.log`.
2. **Two runs censored by budget** (not failures): Os N=16,B=1024 seed 0 (killed at 8000, 1541 s) and Os N=256,B=64 seed 1 (killed at 10800, 3046 s), `censored_by=killed_<step>_cpu_budget`. I used the logged times (which lag by up to 1000 steps) to trigger the kill; the total ended 64 s (0.3%) above the 6 CPU-h cap.
3. Consequently the cells Os N=16,B=1024 (1 seed, censored) and Os N=256,B=64 (1 emerged, 1 censored) are incomplete; P24b's N=256 comparator therefore rests on one emerged seed (5400). The P24b verdict does not depend on it (N=16,B=64 is stuck 2/2 regardless).
4. Early stop at MSE < 0.25 for every run; non-emerging runs ran the full 30000 steps except the two censored ones.
5. Free-row `attn.out` norms for block 0 were not logged in exps 5/8 (`n/l`); block-1 values come from exp-5 `norms[:,1]` (every 1000 steps).
6. A stop-file kill writes the same summary row as a normal finish; `steps_run` is the step at which the run stopped.
