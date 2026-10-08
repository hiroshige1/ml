# Experiment 5 spec: regime-matched transformer test of the finite-context effect (P11)

Follows the exp 4 post-hoc diagnosis (`docs/preregistration.md`, "Exp 4 outcome"): the toy mechanism needs a *small initial
alignment* (`m_0 ≲ 2√(γ/N)`), which a wide MLP in low `d` never has. Here we put the transformer in the toy's regime and run the
P7-type test (number of prompts cannot substitute for context length). Code: extend `icl_additive/transformer_exp4.py`
(same data, same training loop; add width/d_model flags). Budget: hard cap 6 CPU-hours; pilot first.

## Setting
- `d = 256`, `k* = 2` (link `σ_2`), one teacher direction. Embedding `Linear(d+1 → 64)`, 2 blocks, 4 heads, **MLP width 32**
  (so the composed first-layer rows are 32 random directions in 256-d: max initial |cos| ≈ 0.15–0.2 — log it at init).
  Softmax attention. Adam lr `1e-3`, online data, fp32, `torch.set_num_threads(4)`.
- Cells (3 seeds each unless stated), `max_steps = 30000`, eval every 200 steps on 2048 held-out prompts, `T_e` = first MSE < 0.5:
  - (a) `N = 256, B = 64`  — reference (expected to emerge).
  - (b) `N = 64,  B = 64`.
  - (c) `N = 16,  B = 64`  — short context, same number of prompts per step as (a).
  - (d) `N = 16,  B = 1024` — short context, **same tokens per step as (a)** (the Oko N·T tradeoff made explicit).
  - (e) `k* = 1` control at `N = 16, B = 64` (2 seeds).
  - (f) in-weight control (`c ≡ 1`, labels zeroed) at `N = 64, B = 64` (2 seeds).
- Log the alignment proxy (max |cos| of composed first-layer rows with `v`) every 200 steps, and the norm of the block-2 output
  projection / final readout every 1000 steps (as a crude "readout magnitude" record).

## Pre-registered predictions (copy as P11a–c into preregistration.md before running)
- **P11a (threshold):** (a) and (b) emerge within 30k steps for all seeds; (c) fails to emerge for ≥ 2/3 seeds.
  Default: (c) emerges, merely later (if tokens matter, at ≈ 16× the steps of (a), i.e. possibly beyond 30k — so P11a alone is
  *not* decisive; P11b is).
- **P11b (N–T non-interchangeability, decisive):** (d) — same tokens/step as (a) — still fails to emerge for ≥ 2/3 seeds, while
  (a) emerges. Default (Oko Remark 3): (d) emerges at roughly the same step count as (a), or sooner (more independent prompts).
- **P11c:** `k*=1` at `N=16` emerges quickly (< 2000 steps); in-weight control emerges quickly.
- **Kill / scope:** if (d) emerges at ≲ 2× the steps of (a), the finite-context repulsion does not control emergence in this
  transformer even in the small-alignment regime; the paper then claims C2/C3 for the single-neuron / narrow model only and
  reports both transformer experiments as the boundary. If (d) is stuck while (a) emerges, the mechanism transfers and
  the `d ≫ width` regime is the condition to state.

## Deliverables
`results/exp5/summary.csv`, curves, `figs/{Te_cells.png, curves.png, alignment.png}`, `results/exp5/README.md` with the P11 table,
CPU time, deviations. If the pilot shows cell (d) is too expensive (16k tokens/step at d=256), reduce `B` to 512 and say so.
