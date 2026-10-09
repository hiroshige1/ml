# Spec: experiment 13 — distance-based softmax attention in the single-feature model (backlog N19)

Pre-registration: `docs/preregistration.md`, section "Softmax attention that learns: distance-based logits (exp 13)" (P26a–b). Drift table used
for the design: `results/exp13/drift_mc.txt` (`scripts/softmax_drift.py`).

## Code
- `icl_additive/ablation_exp13.py`: copy of `icl_additive/ablation_exp12.py` with the softmax model replaced by the rbf form
  `ŷ = Γ Σ_i softmax_i(−β (s_i − s_q)²) y_i`, `s = σ_2(w·x)`, `β = 0.3`, `Γ = 1`; everything else identical (data as exp 12 with **full d=64 data for
  both optimisers**, `init_fixed` semantics, seeds 0/1, `w` renormalised every step, `m` logged every 100 steps, stop at `|m| ≥ 0.5`).
  Caps: SGD `5e5` steps, Adam `1e5`. Cells: opt ∈ {sgd, adam} × (N,B) ∈ {(16,64), (16,1024), (1024,64)} × seed ∈ {0,1} = 12 runs.
- Sanity before the sweep: at `m = 0.5`, `N = 1024`, `B = 4096` the Monte-Carlo drift of the torch model (gradient projected as in
  `scripts/softmax_drift.py`) must be positive and within 2 s.e. of the table's `+3.25e-2 ± 4e-3` (β=0.3, N=1024, m=0.25 is `+3.25e-2`; use that
  point: m = 0.25). Record the number.
- Output: `results/exp13/runs.csv`, traces `results/exp13/traces/*.npz` (add to .gitignore), `icl_additive/analyze_exp13.py`, `results/exp13/README.md`
  in the exp-12 format with verdict lines for P26a and P26b quoting the criteria and the reading rule, CPU time, deviations.

## Budget
Cap 3 CPU-h, 4 workers. The N=1024 SGD runs at 5e5 steps with B=64 full data are the expensive ones (~0.5 CPU-h each); launch them first.
