# Exp 19: transformer B-scan at N=16 with 3 seeds (threshold or steep power law?), prediction P32 in `docs/preregistration.md`

- **Pre-registration:** `docs/preregistration.md`, block "Transformer B-scan at N=16 with more seeds: threshold or steep power law? (exp 19, N26 part 2)" (commit d6b622c), prediction P32.
- **Code:** `icl_additive/transformer_exp8.py --run free 16 B SEED --out results/exp19/ --max-steps 40000` (unchanged; exp-5 model/data/eval), launcher `results/exp19/run_queue.sh`, analysis `icl_additive/analyze_exp19.py`. Logs in `logs/`, curves in `curves/*.npz` (not tracked).
- **CPU time:** 20201 s = 5.61 CPU-h for the 9 fresh runs (cap 10 CPU-h = 36000 s); 1 thread per process (OMP_NUM_THREADS=1), at most 3 processes at once.
- **Deviations:** see the section at the end.

Setting as exps 5/16/17 free: d=256, d_model=64, MLP width 32, 2 pre-LN blocks, 4-head causal softmax attention, trainable linear readout, sigma_2 link, Adam 1e-3, eval every 200 steps on 2048 held-out prompts,
T_e = first eval MSE/E[y^2] < 0.5 (and < 0.25), early stop at < 0.25 or 40000 steps. `N=16`, seeds 0,1,2 (same teacher/init per seed as exps 5/16/17). Rows with `source` exp5/exp16/exp17 are re-used, not re-run.

## Per-run results (`n/r` = not reached within the steps run; censored runs enter medians at the cap 40000)

| B | seed | source | cap | steps run | T_e(0.5) | T_e(0.25) | final MSE | max align | censored | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 0 | exp17 (not in median) | 80000 | 80000 | n/r | n/r | 0.831 | 0.28 | max_steps | 3884 |
| 256 | 1 | exp17 (not in median) | 80000 | 77000 | n/r | n/r | 1.058 | 0.28 | killed_77000_cpu_budget | 5091 |
| 384 | 0 | exp19 | 40000 | 18200 | 17800 | 18200 | 0.187 | 0.96 | - | 1231 |
| 384 | 1 | exp19 | 40000 | 40000 | n/r | n/r | 1.056 | 0.30 | max_steps | 2706 |
| 384 | 2 | exp19 | 40000 | 40000 | n/r | n/r | 0.851 | 0.40 | max_steps | 2623 |
| 512 | 0 | exp16 (not in median) | 30000 | 13600 | 13400 | 13600 | 0.119 | 0.99 | - | 1291 |
| 512 | 0 | exp19 | 40000 | 13600 | 13400 | 13600 | 0.119 | 0.99 | - | 1258 |
| 512 | 1 | exp16 (not in median) | 30000 | 30000 | n/r | n/r | 1.056 | 0.33 | max_steps | 2700 |
| 512 | 1 | exp19 | 40000 | 40000 | n/r | n/r | 1.058 | 0.33 | max_steps | 3700 |
| 512 | 2 | exp19 | 40000 | 12800 | 12400 | 12800 | 0.222 | 0.98 | - | 1152 |
| 768 | 0 | exp19 | 40000 | 11200 | 11000 | 11200 | 0.130 | 0.94 | - | 1492 |
| 768 | 1 | exp19 | 40000 | 40000 | n/r | n/r | 1.057 | 0.31 | max_steps | 5475 |
| 768 | 2 | exp19 | 40000 | 4400 | 4400 | 4400 | 0.226 | 0.98 | - | 564 |
| 1024 | 0 | exp5 | 30000 | 15000 | 14800 | 14800 | 0.053 | 0.96 | - | 2971 |
| 1024 | 1 | exp5 | 30000 | 10800 | 10600 | 10800 | 0.109 | 0.99 | - | 2170 |
| 1024 | 2 | exp5 | 30000 | 11800 | 11600 | 11800 | 0.170 | 0.96 | - | 2364 |
| 2048 | 0 | exp16 (not in median) | 30000 | 11000 | 11000 | 11000 | 0.247 | 0.95 | - | 4858 |

## Escape fractions and medians (scan B in {384, 512, 768, 1024}; `+` = median is a censored lower bound)

| B | runs | escape T_e(0.5) | escape T_e(0.25) | median T_e(0.5) | median T_e(0.25) | exp16 B=512 (comparison) |
|---|---|---|---|---|---|---|
| 384 | 3 | 1/3 | 1/3 | 40000+ | 40000+ |  |
| 512 | 3 | 2/3 | 2/3 | 13400 | 13600 | s0 13400; s1 n/r (cap 30000) |
| 768 | 3 | 2/3 | 2/3 | 11000 | 11200 |  |
| 1024 | 3 | 3/3 | 3/3 | 11600 | 11800 |  |

Per-seed T_e(0.5): B=384: s0 17800, s1 n/r, s2 n/r; B=512: s0 13400, s1 n/r, s2 12400; B=768: s0 11000, s1 n/r, s2 4400; B=1024: s0 14800, s1 10600, s2 11600. B=256 (exp 17): stuck 2/2 to 80k/77k steps.

OLS of log(median T_e) on log B over B in {384, 512, 768, 1024} (4 points, 2 dof):
- T_e(0.5): slope **-1.149 +- 0.574** (SE)
- T_e(0.25): slope -1.133 +- 0.566 (secondary; P32 is stated for T_e(0.5), as P29)

## P32 verdict

Criteria (quoted): "P32-threshold: `B=384` escapes <= 1/3 and `B=512` >= 2/3 and `B=768` 3/3, with `T_e(512)` and `T_e(768)` medians within x1.5 of `T_e(1024)`'s (12.7k)"; "P32-graded: medians (censored = cap) strictly decrease from `384` to `768` **and** the OLS slope of `log T_e` vs `log B` over `{384, 512, 768, 1024}` is <= -1.3 with `B=384` escaping >= 2/3"; otherwise "neither pattern; n=3 insufficient".

Operationalisation (mine): T_e is T_e(0.5); the 1024 reference in the threshold criterion is the quoted 12.7k (band [8467, 19050]); with the 3-seed B=1024 median 11600 (exp 5 seeds 0-2: 14800, 10600, 11600) the band would be [7733, 17400] - the within-x1.5 clause is unchanged under this alternative (holds). Escape = T_e(0.5) reached within the run (killed/capped = not escaped).

| clause | observed | holds? |
|---|---|---|
| T1. B=384 escapes <= 1/3 | 1/3 | yes |
| T2. B=512 escapes >= 2/3 | 2/3 | yes |
| T3. B=768 escapes 3/3 | 2/3 | **NO** |
| T4. medians T_e(512), T_e(768) within x1.5 of 12.7k | 13400, 11000 (alt. ref 11600: holds) | yes |
| G1. medians strictly decrease 384 > 512 > 768 | 40000, 13400, 11000 | yes |
| G2. OLS slope <= -1.3 | -1.149 +- 0.574 | **NO** |
| G3. B=384 escapes >= 2/3 | 1/3 | **NO** |

P32-threshold: **fails**; P32-graded: **fails**.

**Verdict: neither pattern; n=3 insufficient.**
n=3 seeds per cell, 4 points, 2 degrees of freedom: the slope SE is a rough guide only.

## Observation (not a pre-registered criterion)

Seed 1 never leaves the plateau at B = 384, 512 (exp 16: 30k; here 40k) or 768 (all at MSE ~1.06, max align 0.31-0.33), but escapes at B = 1024 (10600) and 2048 (exp 16 seed 0 only); seed 2 escapes at B = 512 (12400) and 768 (4400) but not 384; seed 0 escapes at B = 384 (17800), 512, 768. Escape fractions are 1/3, 2/3, 2/3, 3/3 for B = 384, 512, 768, 1024, rising with B but not reaching 3/3 at 768 (T3 fails by one seed), and the escape times among escapers do not decrease strictly with B (17800, 12400-13400, 4400-11000, 10600-14800). Seed-to-seed variation at fixed B is as large as the B effect, which is why n=3 does not separate the two readings.

## CPU time

Fresh runs: 20201 s (5.61 CPU-h) summed over `cpu_s` of the exp19 rows (cap 10 CPU-h = 36000 s). Re-used rows cost nothing new.

## Deviations / what was not done

1. **No run dropped or killed for budget.** Per-step costs measured at step 1000 of the first run of each B (3 processes concurrent): 0.10 s (B=512), 0.13 s (B=768), 0.069 s (B=384). Worst-case projection (every run to the 40000-step cap, plus the already-observed costs) was about 22k CPU-s (observed costs of finished runs plus cap-length extrapolation of the rest) at the time B=384 was launched, below the 10 CPU-h cap (36000 s), so neither seed 2 of B=384 nor of B=768 was dropped; no row has `killed_*_cpu_budget`.
2. Launch order was B=512 s0-2, B=768 s0-2, B=384 s0-2 via `results/exp19/run_queue.sh` (xargs -P3, so a later run started as soon as any slot freed; B=384 s0 started while B=512 s1 and B=768 s1 were still running).
3. The pre-registered "T_e(1024) (12.7k)" is the median of the two seeds then available (14800, 10600); with exp 5 seed 2 (11600) the 3-seed median is 11600. The within-x1.5 clause is evaluated with the quoted 12.7k (primary) and with 11.6k (reported as sensitivity); see the verdict section.
4. Medians use T_e(0.5); censored runs enter at the 40000 cap, also for the reused B=1024 rows (all of which escaped, so this does not matter). B=512 rows from exp 16 (seeds 0,1; 30k cap) are shown as a comparison column and are not used in the medians. The fresh B=512 seeds 0 and 1 reproduce exp 16's results exactly (13400 and stuck, now stuck to 40000), as expected from identical seeds.
5. Per-run CPU times are higher than the single-process exp-16 timings because three processes ran at once (about 0.10 s/step at B=512 vs 0.093 s/step).
