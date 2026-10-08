# Findings so far (living document; evidence levels are explicit)

Evidence levels: **T** = derived and checked numerically against the closed form; **S** = measured in online SGD on the
single-neuron / few-neuron model; **X** = measured in a small softmax transformer; **L** = full text of the relevant prior work
read. Pre-registration status refers to `docs/preregistration.md`.

## F1. In-context pretraining squares the link; the effective information exponent of a skill is `2k*` (fixed readout)
- Population loss `L_A(m) = s²[1 − 2Γg² + Γ²g(1)((1−1/N)g² + V/N)]` = `L_B[g→g²]` as `N→∞`. **T** (drift check: 24/24 points within
  2.3 SE, ≤3% at `m ≥ 0.2`). **S**: `κ_eff` from the `m_0`-regression = 4.19 ± 0.08 (A, `k*=2`) vs log-type (B, `k*=2`).
- **L**: the `m^{2Q−1}` product formula is Oko et al. 2024 Lemma 21 (and Nishikawa et al. 2025 Lemmas 20–21); the in-weight
  "fast readout doubles the exponent" is BBPV 2023 §5. **Our part:** the interpretation, the online-SGD timescale comparison, and
  the `k*=2` regime (excluded by Ren et al. Assumption 2.1). Status: *mechanism to cite, not a claim of novelty.*

## F2. Finite context makes the unaligned state drift-stable for `k* ≥ 2` (C2)
- Repulsive term `Γ²V'(m)/N` with `V(m) = E[σ(v·x)²σ(w·x)²]` increasing in `m`; origin stable iff `m² < 16Γ/(N(4−2Γ))` (`σ_2`).
  **T** (drift negative for `m<0.25` at `k*=3`, `Γ=0.1`, `N=128`). **S**: `Γ=1, N=128`: `d=16` escapes 3/3, `d=32` 1/3, `d=64` 0/3;
  `Γ=0.1`: 27/27 escape. The threshold is **soft under SGD** (noise-activated escapes near the boundary) — hard-threshold wording withdrawn.
- **P7 (N–T non-interchangeability, vs Oko et al. Remark 3):** `Γ=1, N=128, d=64`: 0/9 escapes for `B ∈ {32, 256, 1024}`, final `|m|`
  below `m_0`. **S**. This is the discriminating result against a specific published statement.
- **L**: not in Oko (tiny readout init switches the term off, Lemma 18), Nishikawa (same), Gu–Xu–Zdeborová (drop the aligned
  fourth moment; `k*=1`), Ren (in-weight). Qualitative precursor: Kim–Suzuki 2024 App. C.4 (finite-`n` landscape may have a
  spurious minimum at 0) — must be cited.
- **X**: in a 2-layer softmax transformer at `d=32`, width 256, the effect is **not** visible (exp 4: `T_e` increases with `N` at
  fixed tokens/step) — the initial alignment of the best of 256 neurons (0.43–0.55) is far above the barrier. Exp 5 tests the
  `d ≫ width` regime. **Scope is restricted until exp 5 reports.**

## F3. The readout's parameterisation selects the exponent (C3)
- Free scalar readout relaxes fast to the Wiener value `Γ* = g²/((1−1/N)g² + V/N)` and multiplies the feature gradient ⇒
  effective exponent `4k*` where the context statistic is noise-dominated (`d ≫ N^{1/k*}`), `2k*` otherwise. Tied readout
  (`Γ = ‖w‖²`, Ren et al.'s 2-homogeneous form) has norm-independent directional dynamics ⇒ `2k*` at every `N` (but the F2
  repulsion survives). **T** (two ODEs). **S**: at `d=32`, `κ_eff` free `η_Γ=η`: 5.45/5.59/4.81/4.92 at `N=32/128/512/4096`;
  free `10η`: 6.61/5.78/4.92/5.17; **tied 4.10/4.26/4.30/4.44**. The population ODE evaluated at the exact parameters gives
  6.47/5.46/4.99/4.85, 7.02/6.18/5.32/4.92, 4.18/4.15/4.15/4.15 — every cell within ≈0.2–0.5 (the ODE was committed before the
  run; its evaluation at these parameters was post hoc, declared).
- **L**: the static shrinkage formula is Gu–Xu–Zdeborová eq. (298) (cite); its dynamical effect, the `N`-crossover and the
  parameterisation dependence are not in print; BBPV §5 argues the opposite direction for in-weight (regularised link estimation
  is sample-cost-free). Gu et al. list "jointly trained readouts" and "beyond `µ_1≠0`" as open — exactly F2/F3's regime.
- **X**: untested (exp 4 cannot separate it); exp 5 logs readout norms only crudely.

## F4. Additive composition is free in a linear-in-label learner; multiplicative accuracy is derived (C4)
- **T** (exact additivity of the loss for orthogonal skills, `N→∞`); **S**: exp 3 running (P12–P15).
- **L**: Okawa 2023's product accuracy is a *definition*; Arora–Goyal Cor. 13 predicts an *extra delay* for tuples (discriminating
  rival); Wang et al. 2026 supports the non-additive side (`d^{k/2}`); Kobayashi et al. 2024 and He et al. 2024 are about
  compositions that are not output-additive or must be discovered against a memoriser — cite as boundaries.

## F5. In-weight ≫ in-context for the same `k*=2` skill, in the transformer too
- **X**: in-weight control emerges in 40–60 steps vs ≥ 800 (exp 4) — qualitatively consistent with F1 but confounded by the
  zeroed-label design; not a quantitative exponent test.

## What would make this a paper (current best framing, subject to exp 3/5)
*"The exponent of in-context skill emergence is set by the readout and the context, not by the skill alone"*: F1 as the known
mechanism, F3 (parameterisation × context length, ODE-exact) as the main theorem-level claim in the solvable model, F2+P7 as the
sharp qualitative prediction against Oko Remark 3, F4 as the composition corollary, and the transformer experiments reported
honestly as the boundary (exp 4 negative; exp 5 decides whether `d ≫ width` restores the effect). If exp 5's cell (d) emerges,
the paper is a solvable-model paper with an explicit "does not transfer at small width/d ratios" section; if (d) is stuck, the
transfer claim is added with the regime condition.

## Open theory items
1. Noise-activated escape rate for the soft threshold (Kramers-type estimate for the SGD diffusion over the barrier at
   `m* = 2√(Γ/N)`): predicts the escape *probability* vs `d`, `N`, `Γ`, `η` seen in exp 1b.
2. Multi-neuron decoupling in the in-context case (exp 3 will show whether the greedy assignment holds with effective exponent 4).
3. Transformer-level statement of F3: which of (tied first-layer norm, free value/output weights) dominates under Adam.
