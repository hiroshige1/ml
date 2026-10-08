# Full-text check: two-timescale / optimal-readout papers vs claims C3-inweight and C3

Date: 2026-10-08. PDFs taken from arXiv and converted with `pdftotext -layout`; the copies and text dumps are in
`/tmp/claude-0/-home-user-ml/5f7f99a8-9492-5376-82b9-185ad2dc461e/scratchpad/papers_twotime/`
(2310.19793 = BBPV, 2303.00055v4 = BMZ, 2210.15651 = BBSS). Page numbers are PDF page numbers.
The quotes are verbatim. pdftotext breaks up the math, so I rebuilt the inline formulas by hand
(sub/superscripts written as `^`, `_`).

Claims under test
- **C3-inweight**: when the readout sits at its optimum a* = g(m)/g(1), the effective landscape in the overlap is 1 − g(m)²/g(1)
  with g(m) = Σ α_k² m^k. So the effective information exponent is 2k*: an optimal or fast readout doubles the exponent, and the
  gradient-flow escape time goes from d^{(k*−2)/2} to d^{k*−1}.
- **C3**: the in-context version with a finite context: Γ* = g²/((1−1/N) g² + V/N), giving exponent 4k* when noise dominates.

---

## (1) Bietti, Bruna, Pillaud-Vivien 2023, "On learning Gaussian multi-index models with gradient flow" (arXiv 2310.19793v2)

**Setting.** Two timescales. The link f is non-parametric and is fitted infinitely fast, so it sits at its optimum for the
current W. W follows Grassmann gradient flow on the population loss. For q = 1, M = m (the overlap) and G = m².

**Definition of the effective loss.** §3.1.1, p.19, Eqs. (30)–(31):
> "fW := argmin_{f∈L²γr} L(f, W) = argmin ½‖f‖² − ⟨f, A_{M⊤} f*⟩ = A_{M⊤} f*. (30)
> Hence, plugging fW back into the loss, if we denote G = M M⊤ = V Λ² V⊤ ∈ C_q, ..., the resulting loss writes
> L(W) := L(fW, W) = ½‖f*‖² − ½⟨A_G f*, f*⟩ (31)."

Corollary 3.5, p.23, Eq. (42): "Let α_β(V) = ⟨f, H_β(V)⟩ be the decomposition of f, then L(W) = Σ_β α_β²(V) λ_β^2."
(Remark 3.4 switches to the correlation, to be maximised.) The intro gives the same result, p.5:
"L(W) = ‖f*‖² − Σ_β λ_β^2 ⟨f*, H_β(V)⟩²".
For q = 1 this reads L(m) = ½ Σ_k α_k² (1 − m^{2k}). The loss depends on the squared overlap (G = MM⊤, λ² = m²). Each term
carries one squared target Hermite coefficient α_k². There is no α_k²·α_k² (α⁴) product.

**Which coefficients govern the escape.** Eq. (58), p.28, defines the information exponent: s(f) := inf{|β| : ⟨f, H_β⟩ ≠ 0}.
Theorem 1.4, p.5 (informal version of Thm 3.25): "the gradient flow escapes each saddle after time t_k ≃ d^{s_k − 1}, where
(s_k) are the successive relative information exponents". Theorem 3.25, p.31: "Each saddle is escaped within a time-scale
τ_k ≤ C_{f,δ,s}(d^{s̃_k −1} ∨ log(d))". Remark 3.13, p.27: "λ ∼ d^{−1/2}, ... we have ℓ ∼ d^{−s/2}, thus ∇_λ ℓ ∼ d^{−s+1/2}".

**Planted (fixed-link) comparison.** Theorem 4.1, p.35, radial planted case: convergence "after time
t ≥ C_{f,δ,s}[(d^{s/2−1} ∨ log d) + log(1/η)]". Intro, p.3, on the single-index planted case: "t ∼ d^{s/2−1}".
So the paper's own theorems give planted d^{s/2−1} against fast-link d^{s−1}.

**Explicit doubling statement.** §5, "Fast Kernel Learning", p.42:
> "Recall that while the planted correlation loss is given by W ↦ ⟨A_M f, f⟩, with M = W∗⊤W, the fast learning objective
> becomes W ↦ ⟨A_G f, f⟩, where G = MM⊤. At first glance, the fast learning method appears to incur in a 'doubling' of the
> information exponent, since the order of the saddles is doubled when going from M to G, resulting in an effective
> attenuation of the signal gradient 'strength' around initialisation, from Θ(d^{−(s−1)/2}) to Θ(d^{−s+1/2}), where s is the
> information exponent relative to the first subspace"

It then continues: "However, let us derive here a non-rigorous argument that such doubling is not necessarily incurring in
worse sample complexity guarantees, e.g. in the offline setting." The heuristic concludes that n ≫ d^s suffices (Eq. (90),
p.43). p.44: "the natural device to compensate for the 'doubling' exponent would be to consider larger step-sizes than the
prescribed η ∼ d^{−s/2}. We note that this procedure could potentially improve the rate to n = O(d^{s−1})".

**Harder or easier than a fixed link.** p.6: "The planted setting corresponds to the setting where the link function f = f*
is known in advance. Quite surprisingly, this set up turns out to be less well conditioned with respect to the subspace
learning via gradient flow." p.34: "At first glance, the planted model appears to be an easier estimation problem than the
semi-parametric problem ... As we show next, the situation for q > 1 is fundamentally different ... This reveals an inherent
advantage of performing hierarchical learning in the semi-parametric problem." Abstract: the planted problem "has a rough
optimization landscape". In short:
- In time, learning the link simultaneously is slower, because the exponent doubles (§5).
- In landscape, it is better: Grassmann flow converges globally, while the planted problem can trap Stiefel flow for q > 1.
- In samples, they argue heuristically that the doubling need not cost anything (§5).

**Readout shrinking from noise or finite samples.**
- Prop. 3.3, p.21, Eq. (39): ridge regularisation in the RKHS replaces f* by f*^µ := [Q+µI]^{−1/2} Q^{1/2} f*, "a spectral
  shrinkage of the initial target f*". p.22: "Crucially, this approximation error is independent of W, so it does not affect
  the geometry of the landscape nor the gradient flow dynamics". This shrinkage does not depend on m or on the SNR.
- §5, p.42–44, finite-n heuristic. Eq. (84): f̂^µ_W = (Q̂+µI)^{−1}[Q̂ A_{M⊤} f + (1/n)Σ ε_i φ(z_i)], where the "noise"
  ε_i = f(W∗⊤x_i) − A_{M⊤} f(z_i) has E[ε_i² | z_i] = σ² ≤ ‖f‖². Eq. (88) is the bias–variance bound
  σ²/(µn) Tr(...) + µ⟨...⟩. p.44: "the kernel ridge regression is aggressively regularized in the mediocrity phase with
  µ = Θ(1), as the signal-to-noise ratio d^{−s}‖f‖^{−2} is vanishingly small."
- This is the nearest thing to a Γ*-type statement, and it differs in an important way. The authors do not derive a readout
  of the form signal/(signal + noise/N). They do not let the shrinkage feed back into the effective exponent: no 4s appears.
  They conclude that n ≫ d^s keeps the gradient error small.

**In-context learning / context length:** none. The only hits for "context" are phrases like "when the context is clear".

**Verdicts (BBPV)**
- **C3-inweight: PARTIALLY contained, and the central qualitative claim is CONTAINED.**
  - Stated explicitly: an optimal or fast link makes the overlap loss depend on G = MM⊤ (m²), and this doubles the information
    exponent and the saddle order (§5 p.42, quoted above). Their theorems give d^{s−1} for the fast link against d^{s/2−1}
    planted (Thm 3.25 and Thm 4.1). This matches the claimed d^{(k−2)/2} → d^{k−1}.
  - Not stated: the exact scalar-readout formula a* = g(m)/g(1) and L_eff = 1 − g(m)²/g(1) with g(m) = Σα_k² m^k.
  - Their link is non-parametric, so their effective loss is Σ α_k²(1 − m^{2k}). Its leading coefficient is α_{k*}², not
    α_{k*}⁴/g(1). The leading exponent 2k* is the same.
  - Our C3-inweight should be presented as a known mechanism (citing BBPV §5) and a variant for a parametric readout, not as new.
- **C3: NOT CONTAINED.**
  - There is no in-context or finite-context analysis and no Γ*.
  - There is no claim that noise-driven shrinkage of the readout raises the exponent to 4k*.
  - Their finite-sample heuristic argues the opposite way: with suitable regularisation the doubling costs nothing.
  - Cite them as the closest neighbour, together with their µ-shrinkage remark.

---

## (2) Berthier, Montanari, Zhou, "Learning time-scales in two-layers neural networks" (arXiv 2303.00055v4)

**Setting.** f(x) = (1/m)Σ a_i σ(⟨u_i,x⟩). The relative step size is ε: ∂_t(ε a_i) = −m ∂_{a_i}R (Eqs. (5)–(6), p.5).
Second layer fast means ε → 0. Order of limits: d → ∞ and m → ∞ first, then ε → 0. So the overlaps start at s_i(0) = 0
(p.11–12), and the timescales are powers of ε, not of d.

**Effective loss.** There is no effective loss in the overlap with a eliminated. The objects they use:
- V(s) = E{φ(G)σ(G_s)} = Σ φ_k σ_k s^k and U(s) = Σ σ_k² s^k (Eqs. (12)–(13), p.9).
- Rmf(a,s) = ½‖φ‖² − (1/m)Σ a_i V(s_i) + (1/2m²)Σ a_i a_j U(s_i s_j) (Eq. (23), p.11).
- Prop. 4, Eq. (38), p.13: R_{mf,*}(ρ) = ½ Σ_k (φ_k − σ_k ∫ a(ω) s(ω)^k dρ(ω))².

Minimising Eq. (23) over a for one neuron would give ½‖φ‖² − V(s)²/(2U(1)), which is our form with σ = φ. The paper never
writes this down or uses it.

**On putting the readout exactly at its optimum.** §8, p.30:
> "We also note that, while the limit of small ε is interesting, setting directly ε = 0 leads to a singular behavior.
> Formally, setting ε = 0 corresponds to keeping second layer weights equal to their optimal values: a correct analysis of
> this case requires to account for the role of stepsize and not just use the gradient flow approximation."
> (footnote: "No matter how we rescale time, in this case learning takes place instantly, up to a certain critical degree.")

**Escape from the uninformative point.** They need φ_1 ≠ 0, i.e. information exponent 1 in the usual sense. Appendix E.2,
p.63–64, "Case 2: φ0 = ··· = φk = 0": "Note that if φ0 = φ1 = 0, then V′(s) = s·v(s) ... which immediately implies that
s(t) ≡ 0 for t ∈ [0,T]. Therefore, we won't be able to learn any component of φ with degree ≥ 1." There is no d^{...} escape
time, and no statement about a doubled exponent.

Their degree-l phase (Theorem 1, p.26; §6.5, Eqs. (92)–(97)) is controlled by the products σ_l φ_l (e.g.
"σ_l φ_l a(ω,0)s(ω,0)^l ≍ ε^{1/2l}"). The component is learnt at t ≍ ε^{1/2l}, with a = Θ(ε^{−ω_l}) and s = Θ(ε^{β_l}).
§8, p.30: "the alignment is small, and second layer weights are large". So in their regime the readout is not small: it grows.

**Readout shrinkage from noise or finite samples:** none. §7 compares SGD with gradient flow; with O(d) samples the
dynamics follow gradient flow.

**In-context learning / context length:** none.

**Verdicts (BMZ)**
- **C3-inweight: NOT CONTAINED.** They study a fast but not optimal readout (ε > 0) with d → ∞ taken first. They call the
  optimal-readout limit ε = 0 singular and do not analyse it. There is no effective overlap landscape and no doubling
  statement. Their related remark is cited above.
- **C3: NOT CONTAINED.**

---

## (3) Bietti, Bruna, Sanford, Song, "Learning single-index models with shallow neural networks" (NeurIPS 2022, arXiv 2210.15651), main text only

**Setting.** Shared direction θ. Frozen random biases. Second-layer weights c are fitted by ridge (λ). Labels have Gaussian
noise σ² (§3, p.5).

**Effective loss.** p.9: "the projected population loss L̄(θ) := min_c L(c, θ)". Claim 5.4, Eqs. (11)–(12), p.10:
g_m(z) := Σ_{j≥s} α_j m^j h_j(z), and L̄(θ) = −⟨P̂_λ g_m, g_m⟩ + (1+σ²). In the ideal case (N = ∞, λ = 0), p.10:
> "L̄(θ) = min_{g∈L²(γ)} ‖f*(⟨θ*,·⟩) − g(⟨θ,·⟩)‖² + σ² = −Σ_{j≥s} α_j² m^{2j} + (1+σ²), where m = ⟨θ*, θ⟩."

This is the same structure as BBPV: m^{2j} with one squared coefficient α_j². The critical-point equation (14) involves
Σ α_j² j m^{2j−1}.

**Doubling.** Not stated. Their algorithm deliberately avoids the projected landscape during escape. p.12 (Eq. (16) and
Procedure 1): "ζ(t) = 1(t > T0) ... During the first phase, up until time T0, we only optimize the first-layer parameter θ".
The intuition on p.12 is "The empirical optimization landscape (when regarded as a function only of the direction θ) near
the equator (|m| ≪ 1) is of the form L(θ) ≈ m^s." Theorem 6.1, p.13: "(vi) T0 = Θ̃(d^{s/2−1})", which is the fixed-readout
timescale. Remark on p.15:
> "The time-scale separation schedule for ζ in Theorem 6.1 is sufficient but possibly not necessary. The analysis of vanilla
> dynamics (ζ(t) ≡ 1) is challenging, since during the initial phase of training there may be adverse interaction effects
> between c and θ, which under naive analysis lead to sub-optimal sample complexity of n ≥ O(d^{2s})."

This hints at the difficulty, but the paper does not attribute it to an m^{2s} landscape. p.18 lists "Gradient dynamics
without warm-start" as open.

**Shrinkage from noise or finite samples.** The ridge projection P̂_λ and the label noise σ² appear. Label noise enters only
as an additive constant in L̄. There is no Γ*-type readout and no exponent change caused by noise.

**In-context learning / context length:** none.

**Verdicts (BBSS)**
- **C3-inweight: PARTIALLY contained.**
  - Present: the optimal-readout projected loss −Σ α_j² m^{2j} (the m^{2s} structure), p.10.
  - Absent: any statement that this doubles the exponent or the escape time. Their algorithm instead freezes c to get the
    m^s landscape and T0 = d^{s/2−1}.
  - Absent: the scalar-readout formula 1 − g²/g(1).
- **C3: NOT CONTAINED.**

---

## Summary table

| Paper | C3-inweight (optimal readout → 2k*) | C3 (in-context Γ*, 4k*) |
|---|---|---|
| BBPV 2023 (2310.19793) | PARTIALLY: doubling stated explicitly (§5 p.42; Thm 3.25 d^{s−1} vs Thm 4.1 d^{s/2−1}); exact 1−g²/g(1) form absent | NOT CONTAINED (closest: µ-ridge shrinkage Prop 3.3 and SNR heuristic §5, no exponent change) |
| BMZ 2023 (2303.00055) | NOT CONTAINED (ε=0 optimal readout called "singular", not analysed; needs φ1≠0) | NOT CONTAINED |
| BBSS 2022 (2210.15651) | PARTIALLY: projected loss −Σα_j² m^{2j} (p.10); no doubling remark; uses frozen-c warm start with T0=d^{s/2−1} | NOT CONTAINED |

**Implication for our write-up.**
- C3-inweight is known in substance. Cite BBPV §5 ("'doubling' of the information exponent ... from M to G = MM⊤") and
  BBSS Eq. (12) / p.10.
- What remains ours is narrow:
  1. The exact scalar-readout formula 1 − g(m)²/g(1). Its leading coefficient is α_{k*}⁴/g(1), not α_{k*}².
  2. The in-weight framing.
- C3 (Γ* with finite context N, noise-dominated exponent 4k*) does not appear in any of the three papers.
  - BBPV's §5 points the other way: suitably regularised link estimation makes the doubling cost-free in samples.
  - We should cite that contrast and position C3 against it.
