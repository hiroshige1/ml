# Exp 17: transformer B=256 at N=16 with a longer cap (80k steps), prediction P30 in `docs/preregistration.md`

- **Pre-registration:** `docs/preregistration.md`, block "Transformer B=256 at N=16 with a longer cap (exp 17)" (commit eed5c94), prediction P30.
- **Code:** `icl_additive/transformer_exp8.py --run free 16 256 SEED --out results/exp17/ --max-steps 80000` (unchanged; exp-5 model/data/eval), analysis `icl_additive/analyze_exp17.py`. Logs in `logs/`, curves in `curves/*.npz` (not tracked).
- **CPU time:** 8974 s = 2.49 CPU-h for the 2 runs (budget cap 2.5 CPU-h = 9000 s); 1 thread per process, 2 processes in parallel.
- **Deviations:** see the section at the end.

Setting as exps 5/8/16 free: d=256, d_model=64, MLP width 32, 2 pre-LN blocks, 4-head causal softmax attention, trainable linear readout, sigma_2 link, Adam 1e-3, eval every 200 steps on 2048 held-out prompts,
T_e = first eval MSE/E[y^2] < 0.5 (and < 0.25), early stop at < 0.25 or 80000 steps. `N=16`, `B=256`, seeds 0 and 1 (same teacher/init as exps 5/8/16 for the same seed; the first 30000 steps reproduce exp 16's B=256 runs).

## Per-seed results (`n/r` = not reached within the steps run)

| seed | steps run | T_e(0.5) | T_e(0.25) | final MSE | max align | censored | CPU s | T_e(0.5) / 36k | T_e(0.5) vs 30k cap |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 80000 | n/r | n/r | 0.831 | 0.28 | max_steps | 3884 | - | not reached (> 80000) |
| 1 | 77000 | n/r | n/r | 1.058 | 0.28 | killed_77000_cpu_budget | 5091 | - | not reached (> 77000) |

Median T_e(0.5) over the 2 seeds (censored at 80000): 80000+.
Reference points: exp-16 prediction T_e(256) = 12700 x (256/1024)^(-0.75) = **35921** (toy Adam law, slope -0.75, from B=1024's median 12.7k); exp-16 cap = 30000 (both B=256 seeds were stuck there, final MSE 0.83 / 1.06).

## P30 verdict

Criterion (quoted): "the graded law holds: >=1/2 seeds emerge with `T_e(0.5) in [30k, 80k]`"; default (quoted): "still stuck 2/2 at 80k => the rescue is threshold-like between `B=256` and `B=512` and the Adam mechanism's graded law is not supported in the transformer".

Seeds emerging with T_e(0.5) in [30000, 80000]: 0/2 (any emergence: 0/2; stuck at 80k: 2/2).

**P30 FAILS; the default holds** (still stuck 2/2 at the end of the runs: steps run 80000/77000, cap 80000).

Reading: the exp-16 graded-law prediction T_e(256) ~ 36k (inside the 30k-80k band) was not realised: neither seed left the plateau (MSE stays at 0.83 / 1.06, max alignment 0.28, readout norm still decaying toward 0), i.e. 2.2x the predicted time without any sign of escape. Per the pre-registered default, the B-dependence of the transformer's rescue looks threshold-like between B=256 (stuck to >=77k) and B=512 (1/2 emerged by 30k), and the toy's graded Adam law (slope -0.75) is not supported there. Caveats: n=2 seeds; B=512 itself emerged in only 1/2 seeds at 30k, so the 'threshold' is soft; seed 1 was observed to 77000 steps, not 80000 (budget).

## CPU time

8974 s (2.49 CPU-h) summed over `cpu_s` of the 2 runs (budget 2.5 CPU-h).

## Deviations / what was not done

1. **Seed 1 was stopped at step 77000 of 80000 (budget exhausted).** Seed 0 ran the full 80000 steps (3884 CPU s). Seed 1 ran slower than the 0.046 s/step measured in exp 16 (about 0.08 s/step from step 50000 on, probably CPU contention between the two processes), so the 2.5 CPU-h budget (9000 s) would have been exceeded by ~200 s; when its process CPU time reached 5090 s I touched `results/exp17/STOP_free_N16_B256_s1` (the script's own graceful-stop file, checked at each eval), the run ended at the next eval (step 77000) and wrote its row normally. The row's `censored_by` was edited from the script's `STOP_file` to `killed_77000_cpu_budget` in `summary_raw.csv`. Total 8974 CPU s = 2.49 CPU-h. Seed 1 was at MSE 1.057 (max align 0.28) at step 77000, with no sign of departing from the plateau, so the missing 3000 steps are unlikely to change the P30 verdict, but strictly it was observed to 77000, not 80000.
2. The results of both seeds are identical to exp 16 up to 30000 steps (same seeds/data), as expected; no other deviations. The script was invoked with `python3 -m` (as instructed) rather than the `python3 -I icl_additive/...` form in its docstring; `transformer_exp5.py` / `transformer_exp8.py` were not modified.
