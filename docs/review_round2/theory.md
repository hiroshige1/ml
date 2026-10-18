# Theory review (round 2): readout parameterisation and finite-context traps in in-context single-index learning

Line references are to `paper/body.tex` (B) and `paper/theorems.tex` (T). I ran my own sympy, Monte-Carlo and LSODA checks. I did not use the authors' verification scripts.

## 1. What is new, concretely

The single-neuron model is exactly reducible: the population loss depends on w only through m (B 89–95), so every theorem is a statement about a 1-D or 2-D polynomial ODE. Against prior work:

- **Exponent 2k\* ("link squared", B 104–110, 35–40).** The m^{2k−1} gradient is Oko et al. Lemma 21, and the 2Q exponent sits in their N₁T₁ count. Making it "dynamical" means integrating ṁ ∝ m^{2k−1}, which is one line. Small nit: B 37/104 says "in-weight loss with g↦g²", but L_A(∞)=1−Γg²(2−Γg(1)) has an m-dependent quadratic term. So Γ\*(N=∞)=1/g(1), and a fast readout does *not* double the exponent at N=∞, unlike the in-weight case.
- **Pinned trap, Thm A(i)–(ii) (T 25–38).** The finite-N origin minimum is qualitatively in Kim–Suzuki App. C.4. What is new is the explicit threshold m\*, the bound γ₁=2N/(N+9), and the general-k asymptotics. That is elementary algebra (Descartes' rule plus a partial-fraction integral) but correct.
- **Tied readout, A(iii)(a) (T 40–42).** This is Ren et al. Lemma B.1 restated. A(iii)(c) (N-flatness for ρ₀≤ρ\*(m₀)/4) is a perturbative consequence: once ρ≪ρ\*, the Γ² noise term is negligible and (a) leaves the in-weight-like flow. The bootstrap adds explicit constants, not a new phenomenon. Note that the hypothesis ε=kρ₀d^{k−1}/N≪1 means the readout must start *below the Wiener scale*. That is the γ∝d^{−k} regime of one-step analyses (Oko Lemma 18), so "tying removes N-dependence" really means "a small readout removes N-dependence". The paper's own pinned γ=0.01 control (B 215–216) is equally N-flat.
- **Tied trap, A(iii)(b) (T 43–48, Remark T 152–161).** This is genuinely new and the most non-trivial argument: a comparison ODE in z=m^{−(2k−2)}, the rates ρ∝t^{−1} and m∝t^{−k²/2}, and the sharp constant c_k. It is still a planar-ODE result.
- **Free readout, Thm B (T 173–213).** The proof uses adiabatic elimination of Γ onto the Wiener value Γ\*≈Nm^{2k}, giving exponent 4k and τ∝d^{2k−1}/N. Exponent doubling under a fast readout is Bietti et al. 2023 §5 and Berthier–Montanari–Zhou. The static Γ\* is Gu–Xu–Zdeborová eqs. (297)–(298), and the 1/N trade-off is Oko Remark 3. Once L(m,Γ\*)=1−G²/D is written down, the exponent and the 1/N follow in one line (B App. 33–34). The new pieces are the transient factor exp((4k−2)k²Γ₀²/r) and a rigorous four-phase proof.

**In 2–3 sentences:** The paper proves rigorous population-gradient-flow statements for a one-neuron, label-linear ICL model with k\*≥2:
- An O(1) pinned readout has a drift-stable unaligned state below an explicit m\*(γ,N).
- A tied (2-homogeneous) readout with a large initial norm traps permanently, while with a small norm it escapes in N-independent time.
- A free readout escapes in time ∝ e^{(4k−2)k²Γ₀²/r} d^{2k−1}/N.

The exponents (2k, 4k), the static shrinkage, the 1/N trade-off and the norm-independence are each one line from Oko et al., Gu–Xu–Zdeborová, Bietti et al. and Ren et al. What is genuinely new is the explicit trap thresholds (pinned and tied, including c_k), the transient prefactor, and the observation that one loss produces all three behaviours depending on how Γ is parameterised.

## 2. Mathematics: checks performed

1. **Loss and Hermite algebra (Prop. B 86–96, T 12–15).** I derived β_{2i} symbolically for k=2,3,4, and the closed form for β²_{2i} and V_k match. A Monte-Carlo test of L_A at d=5, N=8, Γ=0.7, m=0.6 gave 1.173±0.018 against 1.159 from the formula. **Correct.**
2. **Thm A(i)–(ii), k=2.** −∂_mL factors as −4γm[(N(γ−2)+5γ)m²+4γ]/N, which gives m\*²=4γ/(2N−(N+5)γ) and C>B⇔γ<2N/(N+9). The equation u̇=2u(1−u)(Cu−B), the partial fractions and dF/du are all verified symbolically. **Correct.**
3. **Thm A(iii)(b).** I re-derived the comparison step. The trapping exponent a=k²(k−1)(1−m̄²)/(ND̄) and K are right, and so is ρ_trap(1/8; N=16)=0.178. The exact LSODA flow escapes at ρ₀=0.15 and traps at ρ₀≥0.16, consistent with the boundary 1.3735ρ\*=0.159 in Remark T 157. Asymptotically ρt→2.000=N/8 and mt² converges. **Correct.**
4. **Thm A(iii)(c).** The constants v̄₂=11/8, v̄₃=65/24 and C₂=C₃=4 check out. The bound holds at every cell I tried (d∈{64,256}, N∈{16,…,4096}), but it is loose: at d=256, N=16, |Xτ−1|=0.15 against a bound of 1.06, which is vacuous.
5. **Thm B.** The phase-by-phase argument is sound: Γ decays at rate 2r/N, ∫Γ²=NΓ₀²/4r gives ln(m₁/m₀)=−k²Γ₀²/r, the e′-invariance condition holds, and the LaSalle step uses Lemma `lem:mono` (ρ\*>Γ\*). Numerically τ/τ_B = 0.735, 0.970, 0.9969, 0.99969 at d = 10², 10³, 10⁴, 10⁵ (k=2, N=4, r=1, Γ₀=0.5), and 0.48→0.9993 for N=16. **Correct**, but the convergence is slow (25–50% off at d=100).
6. **Prop. joint (T 215–240).** The rescaled limit system and the corner (c) constant check out (k=2: ∫dx/(x³√(x²−1))=π/4). The proof is a sketch, and the interchange of limits with ∂/∂ln μ₀ is admittedly unproved (T 238).

**Gaps.** None of these are errors, but they limit what the theorems say.
- (i) No theorem covers SGD. The transfer is Conjecture T 253–258, and Remark T 249 shows the tracking condition *fails* as d→∞ at the step sizes used. So the headline 1/N law (Thm B) concerns a regime the experiments cannot reach (B 376).
- (ii) The pinned "trap" is a population statement. Under SGD it is a Kramers delay (B 127–131), and the tied trap is likewise escaped under SGD (B 216–219). "Hard requirement" (B 165) therefore overstates what is proved for the stochastic algorithm.
- (iii) The trap needs N≲d^{k−1}, which is an unusual pretraining regime.

## 3. Is the model too contrived?

Partly. The mechanism is generic and simple. The context statistic has variance Γ²V(m)/N, which grows with m because V has nonnegative Hermite coefficients. The attraction is linear in Γ and the noise quadratic, so the ratio Γ/(Nm^{2k−2}) decides everything. Pinned, tied and free are three ways of setting that ratio. This is a clean lesson: finite-context noise penalises alignment, so the readout must shrink, and shrinking slows learning. A workshop audience will find it digestible. It is not specific to attention; it is the Wiener-filter trade-off plus 2-homogeneity.

The paper's own transformer tests remove the bridge to real architectures. Freezing the final head or the attention output reproduces neither the trap nor the rescue (B 336–341). The authors note that pre-LN leaves no scalar playing ρ's role, and the transformer's N–B interchange runs opposite to the toy's prediction (B 343–349). As it stands, "readout parameterisation decides trap/slowdown/N-independence" is a fact about a one-neuron, one-scalar model. Nothing shows it survives width M>1 with shared readouts, softmax, or a learned key/query.

The many-skill result is a corollary given decoupling (B 273–277), and composition is forced by linearity (B 295–296). The pre-registration bookkeeping in the abstract (B 9–13), at B 151–152 and in E1 (B 172–195) is honest but buries the theory.

## 4. Verdict and scores

- **(a) Top workshop (HiLD/M3L): 5/10**, borderline accept. The theorems are correct and cleanly stated, the experiments track the ODE quantitatively, and the related-work positioning is honest. The novelty over Oko, Ren, Gu–Xu–Zdeborová and Bietti is thin, and the transformer section is negative.
  - **Single change that would raise it most:** rewrite as a focused four-page paper. Keep the loss, Thm A(i)/(iii)(b) and Thm B, with one regime figure. Lead with the mechanism (alignment-dependent context noise forces shrinkage), move the pre-registration ledger to the appendix, and state plainly which parts are one-line consequences of prior work.
- **(b) ICML/NeurIPS/COLT main track: 2/10.** All results are population-flow analyses of a 1–2-dimensional ODE, with no high-dimensional SGD guarantee and no architecture beyond one neuron.
  - **Single change that would raise it most:** prove Conjecture T 253–258. That means online-SGD sample complexity Θ̃(d^{4k−1}/N) for the free readout and a w.h.p. trap for the pinned one, à la Ben Arous–Gheissari–Jagannath. Ideally extend it to width M with a shared or learned attention readout, so the mechanism is shown beyond one scalar.
