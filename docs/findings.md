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
- **X**: exp 4 (`d=32`, width 256, init alignment 0.43–0.55): `T_e` *increases* with `N` at fixed tokens/step — mechanism invisible.
  Exp 5 (`d=256`, width 32, init alignment 0.13–0.18): `N=16,B=64` stuck 3/3; `N=16,B=1024` (same tokens as `N=256,B=64`) emerges 3/3
  but ≈3× later than `N=256,B=64`; readout norm collapses 0.57→0.03–0.16 while stuck and regrows at emergence. **Verdict: in the
  transformer, short context is a ≈3× per-token handicap with a shrinking readout, not a trap.** Both pre-registered extremes failed.

## F3. The readout's parameterisation selects what the finite-context term does (C3, reframed after the adversarial review)
- Same loss, three `N`-dependences of the escape time (population ODE from `m_0 = d^{−1/2}`): **pinned** readout (`Γ=0.1`): `T` saturates
  in `N` and diverges (trap) for `d > d*(N)` (`d=512`: `∞, 771, 386, 352` at `N=32…2048`); **free** readout: noise-dominated regime
  `T·N ≈ const` (`2.9, 2.8, 2.75·10⁶` at `d=512`) — i.e. *Oko et al.'s multiplicative N–T trade-off reappears as a consequence of
  shrinkage* — crossing over to `N`-independence where signal-dominated (`d=32`: `53, 37, 34, 33`); **tied** readout: `T` independent
  of `N` at every `d`. So "prompts cannot substitute for context" (P7) is a statement about the pinned readout only; with a free
  readout the trade-off holds and the exponent crossover is the visible effect. **This is the paper's central statement now.**
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
- **X**: exp 5's readout-norm collapse in stuck runs (0.57→0.03–0.16, regrowth at emergence) is the qualitative signature of the
  shrinkage mechanism in a transformer; not a quantitative test of the exponent.

## F4. Additive composition is free in a linear-in-label learner; multiplicative accuracy is derived (C4)
- **T** (additivity is an identity of label-linear models; unit-tested to 1e-13). **S** (exp 3): paired residual `≤ 0.009`; pair
  emerges with the slower skill, never after it (vs Arora–Goyal's extra delay); `acc_12 ≈ 0.85·acc_1acc_2 + 0.09` (approximately
  multiplicative; the Gaussian-error closed form is wrong in level — heavy tails); product composition stays unlearned (MSE ≈ 1).
  Exploratory: neuron arrival times `T ≈ 0.60/(η π_p m_0²)` match the single-neuron constant (0.64) — the multi-skill dynamics
  decouple with rate `∝ π_p`; the rare skill is missed entirely in 2/10 seeds (assignment failure).
- **L**: Okawa 2023's product accuracy is a *definition*; Arora–Goyal Cor. 13 predicts an *extra delay* for tuples (discriminating
  rival); Wang et al. 2026 supports the non-additive side (`d^{k/2}`); Kobayashi et al. 2024 and He et al. 2024 are about
  compositions that are not output-additive or must be discovered against a memoriser — cite as boundaries.

## F6. Many skills decouple with frequency-proportional rates; in-context orders skills ≈2× more steeply than in-weight
- **S** (exp 2, `P=16`, `π_p ∝ p^{−1.5}`, `M=64`): `T_p ∝ p^{1.44–1.53}` (predicted `1.5`); `T_{p,0.5}·η·π_p·m_0² = 0.73` constant across
  skills/seeds (= single-neuron constant). In-weight baseline (capacity-matched readout): slope `0.75` (predicted `0.75`), but it is a
  collective fit, so the comparison is qualified. Per-skill drops ≈4× sharper in-context. Alignment-loss power law for A: `0.40`
  (predicted `(α−1)/α = 0.33`); no A/B exponent-ratio claim (B saturates).
- Mechanism: the in-context signal for skill `p` carries the *frequency* `π_p` (and `E[c_p²]`), the in-weight one carries the
  *coefficient* `a_p ∝ √π_p` — so rare skills are penalised quadratically more in-context.

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
