# Spec: experiment 11 — attention output scale as the Γ-analogue (backlog N11)

Pre-registration: `docs/preregistration.md`, section "Transformer: attention output scale as the Γ-analogue (exp 11)" (P24a–c).

## Code
- `icl_additive/transformer_exp11.py`: copy of `icl_additive/transformer_exp8.py` (which imports the exp-5 model/data/eval code unchanged) with the
  readout protocols replaced by `--proto {O1, Os}` acting on `model.blocks[b].attn.out` for b = 0, 1: `O1` freezes weight and bias at init;
  `Os` multiplies the weight by 0.1 at init (bias unchanged) then freezes both. Final readout stays trainable (as free). Assert at the end of each run
  that the frozen tensors are bit-identical to their stored copies. Log, as exp 8 did for the head: the Frobenius norm of each `attn.out` weight at
  step 0, every 1000 steps and at the end, plus the eval MSE / alignment proxy every 200 steps. Same seeds (0, 1), same `--run PROTO N B SEED`
  interface, resumable `results/exp11/summary_raw.csv`, curves in `results/exp11/curves/*.npz`, logs in `results/exp11/logs/`.
- Cells and priority: O1 N=16 B=1024 → Os N=16 B=64 → O1 N=256 B=64 → Os N=256 B=64 → Os N=16 B=1024 → O1 N=16 B=64; seeds 0 then 1 within each.
  Free rows for comparison come from `results/exp8/summary.csv` (proto free) — do not re-run them.
- `icl_additive/analyze_exp11.py`: `summary.csv` (exp-11 rows + the free rows from exp 8), `tables.md`, `figs/Te_cells.png`, `figs/curves.png`.
- `results/exp11/README.md` in the exp-8 format: header block (Pre-registration, Code, CPU time, Deviations), per-seed table, verdict lines for
  P24a, P24b, P24c quoting the criteria, kill-criterion statement, CPU time, deviations.

## Budget
Cap 6 CPU-h (exp 8 used 5.5 for 14 runs with 2 single-thread workers). Use 4 single-thread workers. A pilot of 300 steps on O1 N=16 B=1024 fixes the
per-step cost before the queue starts; if the projection exceeds 6 CPU-h, drop seed 1 of the lowest-priority cells and say so in Deviations.
A run may be stopped early only when it is already censored by budget; mark it `censored_by=killed_<step>_cpu_budget`, never as a failure.
