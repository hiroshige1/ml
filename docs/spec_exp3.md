# Experiment 3 spec: is additive skill composition free in a linear-in-label in-context learner? (C4)

Theory: `docs/theory_notes.md` §4. Pre-registered as P12–P14 in `docs/preregistration.md` (written before the run).
Code base: extend `icl_additive/` (numpy, analytic gradients, float64). Budget: hard cap 3 CPU-hours.

## Setting
- `d = 32`, `P = 2` orthonormal teacher directions `v_1, v_2`, link `σ_2` (k*=2) for both; `N = 128`; `B = 32`; `η = 1/d²`.
- Student: **`M = 4` neurons** `w_j ∈ S^{d−1}` (uniform init, sign-symmetric so no flipping needed), **fixed readout
  `Γ_j = γ = 0.1`** for all `j` (so the dynamics is the exp 1 dynamics per neuron and the decoupling question is visible).
  Prediction `ŷ_q = Σ_j Γ_j σ_2(w_j·x_q) A_j`, `A_j = (1/N)Σ_i y_i σ_2(w_j·x_i)`.
- **Pretraining prompts are single-skill only:** each prompt picks skill `p` with probability `π_1 = 0.75, π_2 = 0.25`, then
  `c_p ~ N(0,1)`, `y = c_p σ_2(⟨v_p, x⟩)`. The model never sees a prompt with both skills.
- Evaluation prompts (fresh, 4096 each, every 500 steps): (E1) skill-1 only, (E2) skill-2 only, (E12) **additive pair**
  `y = c_1 σ_2(v_1·x) + c_2 σ_2(v_2·x)`, `c_1, c_2 ~ N(0,1)` iid, (E×) **non-additive pair** `y = c·σ_1(v_1·x)·σ_1(v_2·x)`
  (product of the two linear features; `E[y²]=1`). Record normalised MSE for each, and the thresholded accuracy
  `acc = P(|ŷ_q − y_q|² < 0.5·E[y²])` for E1, E2, E12.
- Also log per-neuron alignments `m_{jp} = ⟨w_j, v_p⟩` every 500 steps; define `T_p` = first time `max_j |m_{jp}| ≥ 0.5`.
- 5 seeds; `max_steps = 1.5·10⁶`; stop when both E1 and E2 MSE < 0.1.

## What to report (tables next to the pre-registered predictions)
1. `T_1`, `T_2` (expect `T_2 > T_1` because `π_2 < π_1`), and `T_12` := first time E12 MSE < 0.5.
2. **Additivity residual** `R(t) = MSE_E12(t) − MSE_E1(t) − MSE_E2(t)` over the whole trajectory (should be `O(1/N)` ≈ 0.01 in
   absolute value — report max |R| and its mean).
3. **Multiplicativity of accuracy:** scatter `acc_E12(t)` against `acc_E1(t)·acc_E2(t)` over all logged `t` and seeds; report the
   slope/intercept of the best fit and the RMS deviation. (This is approximate — Gaussian errors do not multiply exactly — so also
   report the analogous curve computed from the *theoretical* Gaussian-error model for comparison.)
4. E× MSE over time (expect it to stay ≈ 1, i.e. no learning of the product skill).
5. Decoupling: how many neurons end up on each teacher; any neuron that switches teacher mid-run.

## Figures
`figs/curves.png` (E1, E2, E12, E× MSE vs t, seeds overlaid), `figs/additivity_residual.png`, `figs/acc_multiplicative.png`,
`figs/alignments.png` (m_{jp} vs t for one seed). Write `results/exp3/README.md`.
