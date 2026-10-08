# Experiment 4 spec: does the finite-context effect survive in a small softmax transformer?

Gate 3 of the methodology report ("verify outside the toy"). Theory: `docs/theory_notes.md` §2.2–2.4; toy results:
`results/exp1b/README.md`. Pre-registered predictions are added to `docs/preregistration.md` as P8–P10 **before** running.

## Data (same as exp 1/1b, in-context regression)
- `x ~ N(0, I_d)`, one teacher direction `v` (P=1) fixed per run; task `c ~ N(0,1)` per prompt; `y = c·σ_k(⟨v,x⟩)`,
  `σ_2 = (z²−1)/√2` (k*=2, the interesting case) and `σ_1 = z` (k*=1 control). Prompt = `N` context pairs + query.
- Tokens: `[x_i ; y_i]` for context (dimension `d+1`), `[x_q ; 0]` for the query; prediction read from the query position.
- `d ∈ {16, 32}`. Context length `N ∈ {16, 64, 256}`. Batch `B` chosen so that **`N·B = 8192` tokens per step is constant**
  (`B = 512, 128, 32`): this is the Oko et al. Remark 3 tradeoff (`N_1 T_1` fixed) made into a controlled variable.

## Model
2-layer transformer: embedding `Linear(d+1 → 64)`, 2 blocks of [softmax self-attention (4 heads, causal mask so the query
attends to all context tokens) + MLP(64→256→64, ReLU)], LayerNorm pre-norm, linear readout. ~100k params. Adam, lr `1e-3`,
fresh data every step (online), float32, CPU. Also a **linear-attention variant** (replace softmax by identity/`1/N`
averaging) to separate "label nonlinearity" from "finite context".

## In-weight control
Same network, same tokens, but the task is **fixed** (`c = 1` always) and the context labels are **zeroed** (so the only
way to predict is in-weight). This is Model B's analogue.

## Measurements
- Query MSE on fresh prompts every 200 steps (held-out eval of 2048 prompts), normalised by `E[y²] = 1`.
- **Emergence time** `T_e`: first step at which eval MSE < 0.5 (also record < 0.25).
- Per run also log, every 1000 steps, the **first-layer MLP alignment** `max_j |⟨ŵ_j, v⟩|` over the MLP's first-layer rows in
  block 1 (after undoing the embedding: use `W_in·E` where `E` is the embedding weight restricted to the `x` coordinates) — a
  rough proxy for `m`.
- 3 seeds per cell; budget: `max_steps = 30k`; stop early at MSE < 0.1.

## Grid (priority order; hard cap 6 CPU-hours total, use all 4 cores via torch intra-op threads, run cells sequentially)
1. **P8 (context length is not interchangeable with batch):** softmax, `k*=2`, `d=32`, `N ∈ {16, 64, 256}` with `N·B` fixed, 3 seeds.
2. **P9 (control):** same with `k*=1`; and the in-weight control for `k*=2` (any `N`, labels zeroed; use `B=128`).
3. **P10 (linear vs softmax):** linear-attention variant, `k*=2`, `d=32`, `N ∈ {16, 256}`.
4. If budget remains: `d=16` for P8.

## Pre-registered predictions (copy into preregistration.md as P8–P10 before running)
- **P8:** at fixed tokens/step, `T_e` for `k*=2` **decreases strongly with `N`** (at least ×3 from `N=16` to `N=256`), and the
  `N=16` cell may fail to emerge within 30k steps for some seeds. Rival (Oko Remark 3 / default): `T_e` roughly independent of `N`
  at fixed `N·B`, or *increasing* with `N` (fewer prompts per step).
- **P9:** for `k*=1` the `N`-dependence is weak (< ×1.5); the in-weight control for `k*=2` emerges faster than every in-context
  cell at `d=32`.
- **P10:** the linear-attention variant shows the same or stronger `N`-dependence than softmax (if softmax's label nonlinearity
  helps pretraining, as a reviewer might expect from Nishikawa et al., the softmax cells will depend *less* on `N`; our theory
  says the finite-context repulsion lives in the linear-in-label part and survives either way). Either outcome is informative;
  record which.
- **Kill:** if P8 fails (no `N`-dependence, or `N=256` slower than `N=16`), the toy phenomenon does not transfer at this scale;
  report it plainly as the main result of this experiment.

## Deliverables
`icl_additive/transformer_exp4.py` (model, data, training loop, CLI with `--cell` arguments, resumable CSV),
`results/exp4/summary.csv`, `results/exp4/curves/*.npz`, `results/exp4/figs/{Te_vs_N.png, curves_k2.png, alignment.png}`,
`results/exp4/README.md` with the P8–P10 table filled in, CPU time, deviations. Use `python3 -I`-safe code; set
`torch.set_num_threads(4)`; seed everything.
