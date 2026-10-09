# Spec: experiment 12 — ingredient ablation (optimiser, softmax) in the single-feature model

Pre-registration: `docs/preregistration.md`, section "Which ingredient breaks the mechanism: optimiser or softmax? (exp 12)" (P25a–d).

## Code
- `icl_additive/ablation_exp12.py` (torch, CPU, 1 thread per process). Data exactly as `icl_additive/data.py` / E2: teacher `v` uniform on the sphere,
  `x ~ N(0, I_d)`, `c ~ N(0,1)` per prompt, `y = c σ_2(⟨v,x⟩)`, `σ_2(z) = (z²−1)/√2`, prompts of `N` context pairs plus one query; `w_0` with
  `⟨w_0, v⟩ = d^{-1/2}` exactly (use `icl_additive.train.init_fixed` semantics: `v`, then `w_0 = m_0 v + sqrt(1−m_0²) u`, `u ⊥ v` random); seeds 0, 1.
- Model: single feature `w ∈ S^{d−1}`, pinned readout `Γ = 1`. `linear`: `ŷ = Γ σ(w·x_q)·mean_i(y_i σ(w·x_i))`. `softmax`:
  `ŷ = Γ Σ_i softmax_i(σ(w·x_i) σ(w·x_q)) y_i` (softmax over the N context positions, temperature 1). Loss: mean over the batch of `(ŷ − y_q)²`.
- Optimisers: `sgd` with `η = 1/d² = 1/4096` (plain, no momentum); `adam` with `torch.optim.Adam(lr=1e-3)` default betas. After every step
  `w ← w/‖w‖` (both). Log `m = ⟨w, v⟩` every 100 steps; stop at `|m| ≥ 0.5`; caps `3e5` (sgd) and `1e5` (adam) steps.
- Cells: model ∈ {linear, softmax} × opt ∈ {sgd, adam} × (N,B) ∈ {(16,64), (16,1024), (1024,64)} × seed ∈ {0,1} = 24 runs. Launch order: all
  N=16 cells first (they decide P25b/c), then N=1024.
- Output: `results/exp12/runs.csv` (model, opt, N, B, seed, steps, T05, reached05, final_abs_m, max_abs_m, cpu_s), per-run `m` traces
  `results/exp12/traces/*.npz` (gitignored), `results/exp12/README.md` in the exp-9 format with a per-cell table and verdict lines for P25a–d
  quoting the criteria, CPU time, deviations. Also `icl_additive/analyze_exp12.py` for the table.
- Sanity before the sweep: with `linear`+`sgd` the loss and gradient must match `icl_additive` numerics: compare the batch loss of the torch
  model with `ModelA(w, 2, gamma=1.0).loss(...)` from `icl_additive/models.py` on one batch (relative difference < 1e-5); report the number.

## Budget
Cap 3 CPU-h, 4 workers. Expected far less (single feature, d=64). If Adam at lr 1e-3 is unstable (|m| oscillating with large steps), record it and
also run lr 1e-4 for the N=16,B=1024 cells as a declared extra (not a replacement).
