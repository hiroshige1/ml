# Adversarial review: "Context length and readout parameterisation select the emergence exponent of in-context skills in an additive model" (Draft v0.2)

Reviewer stance: hostile but honest. Every attack cites `main.tex` / `appendix.tex` line numbers (from `cat -n`) and the
result or full-text note it rests on. Where I recomputed something, the script is in the session scratchpad. I re-ran
`scripts/ode_kappa_exp1b.py` and extended it (`ode_check.py`, `ntrade.py`). I also refit the exp-1 regressions from
`results/exp1/summary.csv` (`m0reg.py`).

---

## 1. Correctness attacks

### 1.1 Algebra, redone from scratch

| Item | Paper | My derivation | Verdict |
|---|---|---|---|
| Hermite expansion of σ₂² (app. l.6) | σ₂² = √6σ₄ + 2√2σ₂ + 1 | z⁴ = He₄ + 6He₂ + 3, so z⁴ − 2z² + 1 = He₄ + 4He₂ + 2. Halving and normalising gives √6σ₄ + 2√2σ₂ + 1 | **correct** |
| V(m) for σ₂ (l.100, app. l.7) | 6m⁴ + 8m² + 1 | Σβ_j²m^j = 6m⁴ + 8m² + 1 | **correct** |
| V for σ₁ (app. l.7) | 1 + 2m² | σ₁² = √2σ₂ + 1 | **correct** |
| L_A, Prop. 1 (l.109) | s²[1 − 2Γg² + Γ²g(1)((1−1/N)g² + V/N)] | E[y_qŷ_q] = Γs²g²; E[ŷ²] = Γ²g(1)s²[(1−1/N)g² + V/N] | **correct when E[σ²] = g(1) = 1**. The constant "1" in both L_A and L_B should be g(1) (and L_B lacks s²). This is harmless for normalised Hermite links, but it is inconsistent with keeping g(1) explicit in the Γ² term. |
| Drift, Prop. 2 (l.130, app. l.15) | (8γ − 4γ²(1−1/N) − 24γ²/N)m³ − 16γ²m/N | ∂_m of γ²(1+8m²+6m⁴)/N is γ²(16m + 24m³)/N | **correct** |
| m*² (l.131) | 16γ/(N(8 − 4γ(1−1/N) − 24γ/N)) ≈ 2γ/(N(1−γ/2)) | Same expression. At γ=1, N=128: m*² = 0.03252 and d* = 30.75 | **correct** |
| d* values (app. l.44) | 6.8, 30.8, 126.8 | 6.75, 30.75, 126.75 | **correct** |
| Abstract threshold (l.30) | m* ≈ √(2Γ/N) | Drops the (1−Γ/2)⁻¹ factor. At the paper's own experimental Γ=1, N=128 it gives m* = 0.125, but the true value is 0.180 (44% off; a factor 2 in d*). | **error in the abstract.** Corrected: m* ≈ √(2Γ/(N(1−Γ/2))) |
| Small-m drift (l.121) | 4Γs²k*α⁴m^{2k*−1} vs 2ak*α²m^{k*−1} | Correct for N → ∞. At finite N the leading small-m term is the repulsion −Γ²V′(m)/N ∝ m, which dominates (the exp-1 MC slope of the closed form is 3.48, not 3). | correct as an N→∞ statement; ambiguous as placed |
| k*=1 repulsion fraction (l.136) | γ/N of the attraction | V′/N term = 4γ²m/N vs 4γm | **correct**. The (1−1/N)g² self-term adds a further γ/2 fraction. |
| Γ* (Prop. 3a, l.150) | g²/(g(1)[(1−1/N)g² + V/N]) | ∂_Γ L = 0 | **correct**. **But contribution (3), l.69, writes Γ* without g(1).** This is an internal inconsistency (harmless for σ₂). |
| Profile loss, Prop. 3a | ≈ s²[1 − Ng⁴/(g(1)V)] for g² ≪ V/N | Same; the crossover is m ≪ N^{−1/(2k*)}, i.e. d ≫ N^{1/k*} | **correct** |
| Relaxation rate (l.149) | ≈ 2V/N | Curvature 2g(1)[(1−1/N)g² + V/N], times r = η_Γ/η | correct up to the factor r, which is omitted |
| Tied ODE (Prop. 3b, app. l.21) | ṁ = (1−m²)[4gg′ − ρ(2(1−1/N)gg′ + V′/N)], ρ̇ = −4ρ∂_ρL | ŵ̇ = −(1/ρ)P∇_ŵL; ρ̇ = 2u·u̇ = −4ρ∂_ρL | **ODE correct** |
| Tied ODE explanation (app. l.21–22) | "a factor 1/‖u‖ that cancels the ρ = ‖u‖²" | The factor is **1/‖u‖²** = 1/ρ: one 1/‖u‖ from the gradient and one from converting velocity to angle. As written, 1/‖u‖ cannot cancel ‖u‖². | **wording error**; the equations are right |
| Kramers remark (l.141–144) | ΔL/D_eff ≈ 0.02 (d=32) and ≈ 20 (d=64) | ΔL(d=32) = B²/(4A) − L(m₀) ≈ 2·10⁻⁶ and ΔL(d=64) ≈ 2.7·10⁻⁴ (A = 3.844, B = 16/N). D_eff is not reported anywhere in `results/`, so **neither ratio can be verified**. The 1-D estimate also omits the spherical entropic term ((d−3)/2)log(1−m²). At d=32 that term contributes ≈ 16(m*² − m₀²) ≈ 0.02, the same size as the quoted barrier. | **unverified heuristic, incomplete** |

**Bottom line on correctness.** The core algebra (Props 1–3, the appendix drift and both ODEs) is correct. The errors are
peripheral: the abstract threshold, the missing g(1) in l.69, the 1/‖u‖ wording, and the unverifiable Kramers numbers.
Contribution (2) (l.65–66) states a σ₂-specific m*² formula under "for k* ≥ 2". The general statement needs
β₂² = 2k*² (for σ_k), and m*^{2(k*−1)} then scales as Γk*²/N.

### 1.2 Numbers checked against `results/*/README.md`

| # | Paper claim (line) | Source | Status |
|---|---|---|---|
| 1 | "all 24 points agree within 2.3 s.e. and within 3% for m ≥ 0.2" (l.119) | exp1 §1 | **Partly wrong.** Within 2.3 SE: yes (max \|z\| = 2.27). "Within 3% for m ≥ 0.2" fails for A k*=3 at m=0.2 (ratio 0.900; the appendix l.28 prints 0.88). It is inside noise, but the sentence is false as written. |
| 2 | "12/12 cells within 0.5" (abstract l.34); "every cell within ≈0.2–0.5" (l.167) | exp1b §4 and my rerun of the ODE | **Wrong.** Free η_Γ=η, N=32: SGD 5.45 vs ODE 6.47, Δ = **1.02**. Setting censored runs to the cap gives 5.92 (Δ = 0.55). The like-for-like grid (the SGD fit drops c=0.5, so I refit the ODE on c ∈ {0.7, 1, 1.4}) gives ODE 6.05 (Δ = 0.60). So it is **11/12**. The lower bound "0.2" is also wrong: Δ = 0.07 (free1, N=4096), 0.08 (tied, N=32), 0.13 (free1, N=128). |
| 3 | tied 4.10–4.44, ODE 4.15–4.18 (l.167) | exp1b; my ODE rerun | correct |
| 4 | free 4.8–6.6, ODE 4.85–7.0 (l.167) | same | correct in range. But "rising as N falls" is not monotone: free1 is 5.45 (N=32) < 5.59 (N=128), and free10 is 4.92 (N=512) < 5.17 (N=4096). |
| 5 | escapes 3/3, 1/3, 0/3 (l.143) | exp1b §1 | correct |
| 6 | P7: B ∈ {32, 256, 1024}, total ×1/×2.7/×3.2, 0/9, final \|m\| below m₀ (Fig. 2, app. l.45–46) | exp1b §2 | correct |
| 7 | slopes 1.44 ± 0.08, 1.53 ± 0.22, 0.75 ± 0.08; B-tied 0.75–0.89 (l.197–199) | exp2 table; prereg addendum | Numbers correct, but **selectively paired** (see §3.3): the B value for the 1.53 statistic is 1.15 ± 0.24. |
| 8 | T_pηπ_pm₀² = 0.73 (IQR 0.70–0.77) "equals" 0.64 (l.204) | exp2 | Numbers correct. "Equals" overstates a 14% difference, measured with different time definitions and at different m₀ (0.44 vs 0.18). |
| 9 | relative width 1.08 vs 4.09; t^{−0.40±0.04} (l.206) | exp2 | correct |
| 10 | acc₁₂ ≈ 0.85acc₁acc₂ + 0.09, r.m.s. 0.008 (l.214) | exp3 §3 | Correct for seeds 0–4. Extra seeds give 0.807/0.116/0.0115 and Γ=0.3 gives 0.972/0.006/0.0165; not mentioned. The fitted range of acc₁acc₂ is only 0.59–0.70. |
| 11 | transformer 1000/1400/2800; \|cos\| 0.43–0.55; 0.13–0.18 (l.244–246) | exp4, exp5 | correct |
| 12 | (d) 10.6–14.8k, "≈3× later than N=256 (2.8–4.0k)" (l.248) | exp5 | Numbers correct, but **omits that (a) seed 2 was still on the plateau at 17k while (d) seed 2 emerged at 11.6k.** The ordering reverses in one of three paired seeds. |
| 13 | readout norm 0.57 → 0.03–0.16, back at 1.7 (l.248–249) | exp5 | correct |
| 14 | appendix m₀-regressions −2.19 ± 0.08, −0.57 ± 0.06, −0.50 ± 0.06, −0.13 ± 0.02 (app. l.40–41) | exp1 README / `summary.csv` | **Three of four reproduce** (pooled, d ≥ 16, no d covariate): −0.58 ± 0.05, −0.50 ± 0.05, −0.13 ± 0.02. **A k*=2 does not**: I get −2.29 ± 0.07 (d ≥ 16) or −2.64 ± 0.12 (all d). The README's per-η₀ fits range from −2.01 to −2.64. I could not find −2.19 in any results file. |
| 15 | 20 of 105 runs censored (app. l.39–40) | exp1 §5 | correct (20; 8·5·2 + 5·5 = 105) |

---

## 2. Novelty attacks (contribution by contribution)

### (1) Closed form and "the link is squared" (l.60–64; abstract l.27–29)
- **Closest prior.** Oko et al. 2024 eq. (B.4) and **Lemma 21** have the population gradient as a task-averaged product of two
  correlations, with main term ∝ ‖w_{1:r}‖^{2Q−2}w_{1:r} ∝ m^{2Q−1}. Their Thm 1 has N₁T₁ = Ω̃(d^{2Q+1}r). Nishikawa et al.
  2025 **Lemmas 20–21** do the same computation. Kim–Suzuki 2024 eqs. (2)–(3) give a loss quadratic in the cross-covariance.
  For the in-weight "fast readout doubles the exponent", BBPV 2023 **§5** says so explicitly ("'doubling' of the information
  exponent"). Gu et al. **eq. (297)** is the same closed-form quadratic risk with isotropic V
  (R = v − 2μ²/d·TrW + ((L−1)/L·μ²/d + v/L)‖W‖²).
- **Overclaim 1.** The abstract states "the effective information exponent is 2k*" without qualification. The full-text note
  on Nishikawa (§1.6) shows that an identity-initialised merged K·Q gives d^{ie+2}, not d^{2ie}. So 2k* is a statement about
  a *factored vector-overlap parameterisation from random init*. The novelty protocol required this caveat ("the
  parameterisation caveat made explicit"). **The paper never states it** and cites Nishikawa only for softmax (l.259–261).
- **Overclaim 2.** Calling a two-line conditional-variance computation "Closed form" as contribution #1 inflates it. The only
  genuinely new ingredient is the *alignment dependence* of V(m), which Gu et al. explicitly discard (their eq. 66 and eq. 128).
  Gu eq. (297) should be cited for the structure.
- **Overclaim 3.** "including the k*=2 case excluded from Ren" (l.64). Ren et al. are purely in-weight. No in-context case can
  be "excluded" by them, and in-context k*=2 has effective exponent 4, which is *inside* Ren's even-k*≥4 regime. The
  in-weight IE=2 case is covered by Ren's companion paper (Ben Arous–Erdogdu–Vural–Wu 2025, quadratic networks). That paper
  is neither cited nor full-text checked, although `ren2025.md` §7c says it must be.

### (2) Finite context makes the unaligned state drift-stable (l.65–67; abstract l.29–32; related work l.256–259)
- **Closest prior.** Kim–Suzuki App. **C.4**: at finite n the landscape "may no longer be benign" and a saddle can become a
  local minimum. They argue such basins have W₂ radius ≤ n^{−1/4} and that trapping is implausible. The paper cites this
  fairly (l.263–264). Oko **Lemma 18** suppresses the f·∇f term with tiny γ (cited, l.137). Gu et al. drop B(a)
  (eq. 66/128). Gu's feature-learner transition at **L = Θ(d)** (Thm 1, eq. 22) and w_{d,L} → 0 for L ≪ d have the **same
  linear-in-N scaling** as d* ≈ N(2−γ)/(4γ). `gu2026.md` §6 says the paper "must explain why ours is distinct". It does not.
- **Overclaim (strong).** "the number of prompts cannot substitute for context length, in contrast to the N·T trade-off of
  one-step analyses" (l.31) and "Oko Remark 3 … that Prop. 2 and Fig. 2 contradict beyond the one-step regime" (l.258–259).
  Three problems:
  1. Oko's Remark 3 lives in a tiny-readout regime (γ ∝ d^{−Q}). The trap exists only with the readout **held at O(1)**
     (γ=1). At γ=0.1, 27/27 runs escape. P7 tests a regime Oko never claimed.
  2. **The paper's own Prop. 3(a) reproduces Oko's multiplicative trade-off** in the free-readout regime. With Γ at Γ* ≈ Nm⁴/V,
     ṁ ≈ 8Nm⁷, so T ≈ 1/(48Nm₀⁶) and T·N is constant. I checked this with the paper's own ODE (`ntrade.py`): at d=512,
     T·N = 2.89·10⁶, 2.82·10⁶, 2.75·10⁶ for N = 32, 128, 512 at η_Γ = η. So with a trainable readout, halving N doubles the
     number of prompts needed: an N×T trade-off of Oko's form. With η = 1/d² and m₀ = d^{−1/2}, the required N·(#prompts)
     ∝ Bd⁵ = d^{2Q+1}, the same d-scaling as Oko's N₁T₁. Non-substitutability holds only when the readout is prevented from
     shrinking.
  3. In the only realistic architecture tested (exp 5), the README's own words are "number of prompts per step substitutes
     for N to a large degree". (c) N=16, B=64 never emerges; (d) N=16, B=1024 emerges in 3/3.
- **What remains novel.** The explicit Γ²V(m)/N mechanism, its alignment dependence, and the stability condition for a
  readout at O(1). That is real but narrower than claimed.

### (3) Readout parameterisation selects the exponent (l.68–72; Prop. 3; l.155–160)
- **Closest prior.**
  - **Gu eq. (298)**: w_{d,L} = Lμ²/(dv + (L−1)μ²) is literally Γ*. The paper cites this correctly (l.156).
  - **BBPV §5**: a fast or optimal readout doubles the exponent in-weight. Cited (l.158–159). BBPV also argue the doubling is
    sample-cost-free under regularised link estimation, the opposite direction; the paper does not mention this.
  - **Ren et al. Lemma B.1**: under the 2-homogeneous parameterisation the tangent dynamics of v̄ do not depend on ‖v‖. This
    is exactly Prop. 3(b)'s "the attraction does not carry ρ". The paper says "our tied protocol is theirs" (l.265) but
    presents the norm-independence as its own finding (l.71, l.151–153) without citing Lemma B.1.
  - Berthier–Montanari–Zhou 2023 §8 call the ε=0 optimal-readout limit "singular" (relevant to "pins it at Γ*"). Not cited.
- **Overclaim.**
  - The tied half is Ren's Lemma B.1 plus the observation that the repulsion term keeps a factor ρ. The free half is BBPV's
    doubling with an N-dependent Γ*. The only in-context-specific content is that Γ* depends on N. "Selects the exponent" is
    then mostly a statement about plain-GD geometry: tied plain GD effectively rescales the angular learning rate by 1/ρ.
  - The README itself notes that tied escapes 20–50× sooner than free at the same η and m₀ (T·η 0.8–11 vs 4.5–565). It calls
    this "a large effect of the parameterisation which is not about the exponent".
  - The mechanism sentence "the finite-context noise makes the readout fast, pins it at Γ*" (l.159–160) is **contradicted by
    P5**: Γ at T₀.₅ is 0.18–0.25 against Γ* = 0.38–0.99, i.e. the readout is not pinned.
  - At N=4096 the relaxation rate 2V/N ≈ 5·10⁻⁴ is slower than the overlap dynamics.

### (4) Many skills and composition (l.73–75, §4)
- **Closest prior for T_p ∝ 1/π_p.**
  - Michaud et al. and Nam et al. quanta models: each sample uses one quantum with frequency π_p, and the rate is ∝ π_p.
    The in-context Σπ_p ℓ(m_p) is a mixture loss of exactly this form.
  - **Ren Prop. 2.2**: in-weight T_p ∝ 1/a_p. **Ren Cor. 2.3** ("unstable" schedule, η ∝ a_p): T_p ∝ a_p^{−2} = p^{2β},
    i.e. in-weight learning already gives the *doubled* ordering under a different learning-rate protocol (`ren2025.md`
    §7b flags exactly this).
- **Overclaim.** "in-context learning orders skills by frequency about twice as steeply" (l.195; abstract l.35–36). The factor
  2 comes from two conventions. The first is the choice a_p² ∝ π_p for the in-weight baseline. The second is that the
  in-context drift carries E[c_p²] while the in-weight drift carries a_p, which is contribution (1) restated for
  coefficients. Under either alternative below, the in-context and in-weight slopes coincide:
  - an in-weight mixture baseline (each sample from one skill with probability π_p, so the Bayes target has coefficient π_p);
  - Ren's Cor. 2.3 schedule.
  This is a corollary of (1) plus a convention, not a separate finding.
- **Composition.**
  - "additive compositions of learned skills cost nothing" (l.36–37, l.74–75) is an algebraic identity of a label-linear
    predictor (the paper admits it, l.210–211; prereg exp-3: "it should not have been pre-registered as an empirical test").
  - Arora–Goyal Cor. 13 (tuple competence needs a k′-fold smaller loss) concerns AND-accounted errors in a no-network
    framework. Their "extra delay" maps onto the paper's **thresholded pair accuracy**, not onto the MSE sum. The paper's own
    finding that acc₁₂ ≈ acc₁acc₂ (l.214) means the pair's accuracy reaches any threshold *later* than either skill. On
    accuracy, the data agree with Arora–Goyal's extra delay. "In contrast to the extra delay … predicted by Arora–Goyal"
    (l.213) compares a metric where delay is impossible by construction (E₁₂ = E₁ + E₂) with one where it is not.
  - Okawa: the product metric is a *definition* there. The paper's approximate multiplicativity is over acc₁acc₂ ∈ [0.59, 0.70]
    with slope 0.85 and intercept 0.09. Over an interval 0.11 wide, nearly any smooth monotone relation fits to RMS 0.01.
  - **He et al. 2024 is mis-cited** (l.269–270) as a "not output-additive" composition. In He et al. the OOD algorithm
    (Modular Regression) is **linear in the context labels**, and it emerges only with ≥ 256 tasks, structured batches and
    sufficient depth. That is direct counter-evidence to "label-linear composition is free" once the architecture must
    *discover* label-linearity. `icl_composition_mean.md` asked for exactly this sentence.
- **"Product compositions are not learned"** (l.75, l.216). The model was trained **only on single-skill prompts** and never
  saw a product task. The product σ₁(v₁·x)σ₁(v₂·x) = [σ₂(u₊·x) − σ₂(u₋·x)]/√2 with u± = (v₁ ± v₂)/√2 **is representable** by
  σ₂ features of the model. So P15 is an out-of-distribution non-test, not a learnability result.

### (5) Transformers (l.76–79, §5)
- **Closest prior.**
  - Nishikawa et al.: softmax's label transform helps *inference-time* context length, not one-step pretraining.
  - Zhang–Wu–Bartlett 2024: the task mean is learned in-weight by the MLP, and LSA alone cannot (Thm 4.1, eq. C.2).
- l.123–124 claims "A bias added to A_j restores an in-weight path of exponent k* for the mean task only" with **no derivation
  and no experiment** anywhere in `results/`. Either prove it or cut it.
- The transformer section is honest in its body (l.251: "consistent with—not a confirmation of"). The abstract
  (l.36–38: "the effect appears as …") is not; see §3.

---

## 3. Overclaiming and framing attacks

1. **Abstract l.34, "12/12 cells within 0.5".** False (11/12; see §1.2 #2). The main text (l.166–169) also omits that the ODE
   *evaluation* at the experimental parameters was post hoc. `preregistration.md` declares this ("post-hoc but
   parameter-free"); the paper does not.
2. **Abstract l.30–32 and l.258–259, "prompts cannot substitute for context length … in contrast to … one-step analyses".**
   This holds only for a readout pinned at O(1). The paper's own free-readout ODE gives T·N ≈ const (Oko-type trade-off).
   The transformer README says prompts substitute "to a large degree". P7 increased total prompts only ×3.2 (the ×10 leg was
   not run). It also *cut flow time by ×10* at fixed η (B=1024 for 3·10⁴ steps vs B=32 for 3·10⁵), so it does not test
   "more prompts" fairly.
3. **Abstract l.32–33 and contribution (3), "raises the effective exponent towards 4k* as N decreases".**
   - At η_Γ = η the maximum measured is 5.59, and it is non-monotone in N.
   - The pre-registered d-scaling test of the same claim (exp1b Priority 3: N=128 vs 2048, d = 16–48) found **no
     N-dependence**: slopes 3.49 vs 3.70, and T₀.₅η within 13%. K3 says such an observation "kills the shrinkage story";
     the README says the observation is "of that form". **This null result appears nowhere in the paper**, not even in
     Limitations.
   - At m₀ = d^{−1/2} the free1 escape times at N=128 and N=4096 are 36 vs 38.5 (×η). The N=128 → 4096 κ shift is carried by
     the single smallest-m₀ point (565 vs 248).
4. **Abstract l.35–36, "1.6–2× steeper ordering".**
   - The range mixes statistics. 1.53 is the pre-registered T_p(0.5) statistic, whose in-weight counterpart is **1.15 ± 0.24**
     (ratio 1.33; A − B = 0.38 ± 0.33, n.s., exp2 README).
   - With the pooled-alignment B statistic (0.93 ± 0.06) the ratio is 1.55. With B at ρ = 0.1 it is 1.44/0.34 = 4.2.
   - Both in-weight baselines are collective fits at M=64 > d (acknowledged, l.200–203). The convention dependence (§2(4))
     is not acknowledged at all.
5. **Abstract l.36–37, "additive compositions of learned skills cost nothing".** This is an identity of the architecture,
   presented as a dynamical finding. "Product compositions are not learned" (l.75) is a non-test (never trained on products).
6. **Abstract l.36–38, "In small softmax transformers the effect appears as a ≈3× per-token handicap … accompanied by a
   collapsing readout".**
   - The binding scope rule in `preregistration.md` (exp 5) allows only "an observation consistent with, not a confirmation
     of, the mechanism". "The effect appears as" asserts identity.
   - The 3× rests on **2 paired seeds**. In the third, (d) emerged at 11.6k while (a) was still on the plateau at 17k (ratio
     < 0.68). Fig. 4a shows the censored (a) seed but the text (l.248) does not say the order reverses.
   - Fig. 4b shows the readout norms of the stuck run (c) and the *later-emerging* run (d) **collapsing identically** until
     (d) emerges. So the collapse does not distinguish "trapped" from "about to emerge". It is what any model with
     mean-zero targets does on a null plateau: the optimal scalar multiplier of an uninformative predictor is 0. That is
     Γ* → 0 in a trivial sense, not evidence for the finite-context shrinkage mechanism. There are no norm traces for (a)
     during its plateau or for the in-weight control.
7. **"invisible when the initial feature alignment is large" (l.38, l.243–245).**
   - Exp 4 shows the **opposite ordering** (short context 2.8× faster), not invisibility.
   - The "initial alignment" explanation is a post-hoc diagnosis (prereg exp-4 outcome). The regime-matched test designed to
     confirm it (P11b) **failed**.
   - Exp 4 → exp 5 changed d (32 → 256), width (256 → 32) and B simultaneously, so attributing the reversal to initial
     alignment is confounded.
8. **Intro l.54–57 / title.** "Context length … select[s] the emergence exponent". In the accessible regime the N effect on
   κ_eff is about 0.5–1.5 units, from 4 m₀ values over a factor 2–2.8, and the absolute escape time barely moves with N at the
   canonical m₀.
9. **Limitations l.272–278** omit:
   - the d-scan null;
   - the failed P4′ leg (tied ρ₀=1 at d=32 escaped 3/3, contrary to the pre-registered "stuck like fixed γ=1");
   - P5 not met (only in the appendix table);
   - the (a)-seed-2 reversal;
   - the B change (32 → 8) between N=128 and N=512 in the κ table.
10. **Appendix pre-registration table (app. l.84).** P4′ is listed as "met (4.10–4.44)", silently dropping the failed ρ₀=1 leg
    of the same pre-registered prediction (exp1b README §3: "So the tied d = 32 prediction failed").

---

## 4. Experimental-design attacks

1. **Single-neuron theory vs M=64 experiments (Exp. 2).**
   - With 64 uniform neurons the best neuron per skill starts at m₀ = 0.36–0.55 (5/48 pairs already ≥ 0.5), i.e. in the
     *large-initial-alignment* regime where the paper says the mechanism is invisible (l.245).
   - Skills carry 2–12 neurons each, so per-skill effective readouts are n_pΓ ≈ 0.2–1.2, not Γ = 0.1.
   - The "decoupling" constant K = 0.73 is measured with drop midpoints at m₀ ≈ 0.44. Exp 3 found first arrivals from
     m₀ ≈ 0.4 to be *faster* than the cubic law (K = 0.26–0.34). The "equals 0.64" agreement mixes definitions.
2. **Collective-fit baselines.** Both in-weight baselines are collective at M=64 > d. The decoupled in-weight regime (M < d)
   "was not run" (app. l.55). The A-vs-B slope comparison is therefore between a decoupled and a collective learner.
3. **Censoring.**
   - Censored runs are *excluded* from the κ fits (biasing κ down at exactly the cells that carry the N effect).
   - At N=32 the whole c=0.5 column is censored, so κ there is fit on 3 m₀ values spanning a factor 2 (9 runs). The ODE is
     fit on 4, so the comparison is not like-for-like. I refit the ODE on the 3-point grid: free1 6.05 and free10 6.74, vs SGD
     5.45 and 6.61.
   - No Tobit or survival model is used.
   - Exp 1's d-slopes drop d=128 entirely (censored medians).
4. **Soft threshold.**
   - 4/7 "stuck" cells had escapes. N=32, d=16, γ=1 escaped 2/3 although m₀² is 2.4× below m*², deep in the trap; the
     README notes heavy-tailed single-step jumps of ≥ 0.18 in |m|. The Kramers remark (l.140–144) discusses only N=128 and
     does not address this.
   - With 3 seeds per cell, escape *fractions* are resolved only to thirds.
5. **Learning-rate choices.**
   - Exp 1b uses only η₀=1. The pre-declared η₀-invariance admissibility check (prereg "Pre-declared analysis choices";
     exp1b README §7.6) **was not run**, so drift-dominance is assumed exactly where noise-activated escape is claimed.
   - Free-readout κ depends on η_Γ/η and Γ₀ (ODE 6.47 vs 7.02 at N=32 for r = 1 vs 10). At finite d the "exponent" is a
     function of optimiser hyperparameters, not only of N and the parameterisation.
   - Transformers: Adam lr 1e-3 fixed across B = 64 and 1024, with no lr tuning or √B / linear scaling. The 3× "per-token
     handicap" of (d) vs (a) could shrink or vanish with a B-appropriate learning rate.
6. **Is κ_eff from 4 m₀ × 3 seeds a sound estimator?**
   - It is a secant slope over log m₀ ranges of 1.03 (factor 2.8) or 0.69 (factor 2). Local slopes inside one cell vary
     2.5–3.7 (README §4: free1 N=4096), so it is not a power-law exponent.
   - The OLS SEs ignore misspecification, heteroscedasticity and censoring.
   - B differs (32 vs 8) exactly where κ drops (N=128 → 512). The B-control was done only at N=128.
   - It is sound only as a like-for-like *comparison* statistic against the ODE.
   - **The stronger test is sitting unused**: the ODE matches absolute escape times. Free1 N=128: ODE 473/120/37/13 vs SGD
     565/113/36/13 (×η). Tied: 8.2/4.1/1.9/0.9 vs 9.2/3.6/1.7/0.9. This should be shown instead of, or beside, κ.
7. **P7 design.**
   - Raising B while cutting steps at fixed η reduces integrated drift ×10 at B=1024. The final |m| = 0.065 at B=1024 (vs 0.01)
     reflects less elapsed flow time, not "more prompts".
   - The ×10 total-sample leg was not run.
   - Since the population drift is negative in the trap, no B could help at any η. That makes P7 a confirmation of Prop. 2's
     sign, not an independent test of Oko Remark 3.
8. **Transformer evaluation.**
   - The thresholds (MSE < 0.5) are normalised by the population E[y²], while the zero-predictor level on the fixed eval sets
     ranges 0.79–1.14. So "emergence" is declared at different fractions of the null across seeds.
   - Cell (b) N=64 (1 seed, never emerged) is not mentioned in the paper.
9. **What a careful reviewer would ask to see:**
   - absolute T(m₀) curves, SGD vs ODE;
   - the d-scan extended to d = 64–128 where the ODE predicts N separation;
   - the free-readout N-vs-prompts exchange at equal tokens;
   - η₀ invariance for exp 1b;
   - norm traces for all transformer cells, including the in-weight control;
   - a B-scaled lr control for exp 5;
   - the in-weight baseline at M < d.

---

## 5. What would make it acceptable (ranked)

1. **Rewrite the abstract, intro and conclusion to the evidence.**
   - Replace "12/12" with the truthful statement (11/12 within 0.5; better, absolute-time agreement).
   - Qualify (ii) to "a readout held at O(1)", and say plainly that a free readout yields an Oko-type multiplicative N×T
     trade-off (T·N ≈ const) up to the crossover d ~ N^{1/k*}. This turns a false contrast into a sharp, correct statement:
     *the trade-off holds iff the readout can shrink*.
   - Turn the transformer sentence into "consistent with, not a confirmation of", and give the 2/3-seed caveat.
   - Recast "twice as steep" as the E[c²]-vs-a corollary of (1), with the convention dependence and Ren Cor. 2.3 stated.
   - Demote additive composition to a remark (it is an identity) and drop the product-composition claim.
2. **A sharper theorem for Prop. 3**, stated with explicit regime conditions:
   - **(a) Free readout.** For r = η_Γ/η and Γ₀, give the escape-time asymptotics T ≍ 1/(N·m₀^{4k*−2}) for
     m₀ ≪ N^{−1/(2k*)}, under the adiabaticity condition r·V/N ≫ Nm₀^{4k*−2}, together with the non-adiabatic
     (Γ-growth-limited) regime that P5 shows the experiments are actually in.
   - **(b) Tied readout.** T ≍ m₀^{−(2k*−2)}, citing Ren Lemma B.1.
   - **(c) Trap.** A general-k* trap condition using β₂² = 2k*², stated as a function of the readout scale relative to Γ*.
     This also explains why the trap needs an O(1) readout and why tied with ρ₀=1 gets stuck at d=64.
3. **One extra experiment (at most).** The free-readout N-vs-prompts exchange in the solvable model.
   - Setup: d ∈ {48, 64}, equal tokens per step, η scaled with B, prompts counted as steps × B.
   - It directly decides whether the headline (ii) survives outside the artificial fixed-Γ=1 regime. The ODE predicts T·N ≈
     const there.
   - If budget allows only one cheaper alternative: extend the Priority-3 d-scan to d = 64–128, where the ODE says N=128 and
     N=2048 separate.
4. **Report the unused and adverse evidence.**
   - Absolute ODE-vs-SGD times.
   - The d-scan null, P4′'s failed ρ₀=1 leg, P5, and the (a)-seed-2 reversal.
   - The B=32/8 switch in the κ table, and the fact that the ODE evaluation was post hoc.
   - A survival-model fit for censored runs.
   - Show the tied ρ₀=1, d=64 trap (0/3, ρ decays to 0.18) as the *natural-readout* evidence for (2). It is better evidence
     than γ=1 fixed, and it is currently absent.
5. **Fix the related work.**
   - Cite Ren Lemma B.1 for Prop. 3(b) and Ren Cor. 2.3 for the doubled ordering.
   - Cite Gu eq. (297) for the closed form, and Gu's L = Θ(d) threshold, with a sentence on why d* ∝ N is a different object.
   - Cite Nishikawa's merged-KQ caveat for 2k*.
   - Re-cite He et al. correctly (label-linear but *not* free when it must be discovered).
   - Full-text check and cite Ben Arous–Erdogdu–Vural–Wu 2025 for in-weight IE=2. Cite BMZ §8.
   - Either derive or delete the "bias restores exponent k*" claim (l.123–124).

---

## 6. Strongest case FOR acceptance (stated as fairly as I can)

- The solvable model is clean and the algebra is right. A Monte-Carlo drift check verifies the closed form at 24 points, and
  the drift sign at k*=3 (negative below m ≈ 0.25) is confirmed by MC.
- The parameter-free population ODE predicts SGD remarkably well, better than the paper claims: absolute escape times agree
  to ~10–20% across protocols (see §4.6), and the κ ordering free10 > free1 > tied is reproduced at every N.
- The alignment-dependent finite-context variance Γ²V(m)/N is, per the full-text checks, genuinely absent from Oko, Nishikawa,
  Gu and Ren, and only qualitatively foreshadowed by Kim–Suzuki C.4. It explains why one-step analyses must initialise the
  readout tiny.
- The tied-vs-free contrast makes a useful conceptual point for ICL theory: whether a readout can shrink independently of the
  feature norm changes the dynamics. Even if the pieces are known in-weight, the combination with an N-dependent Γ* is new.
- Methodological honesty is far above workshop norms. Every prediction was pre-registered with kill criteria, failures are
  recorded (P8 reversed, P11b both branches failed, P12 estimator failure, P18-B failed), and the transformer section body
  avoids claiming confirmation.
- For HiLD, a solvable high-dimensional model with SGD-vs-ODE agreement and explicitly reported boundaries is squarely in
  scope.

## 7. Verdict

**Score: 5/10 (borderline; weak accept only if the framing is cut back as in §5.1).**

The mathematics is correct where it matters. The ODE-vs-SGD agreement is real and under-sold. The finite-context variance
mechanism is a genuine, if modest, addition to ICL feature-learning theory. But the paper's headline claims run ahead of its
evidence in ways a careful reviewer will find quickly:
- "12/12 within 0.5" is false.
- "Prompts cannot substitute for context length" is true only for a readout pinned at O(1). The paper's own free-readout ODE
  yields the very multiplicative N×T trade-off it claims to contradict.
- The pre-registered d-scaling test of the N-dependent exponent came back null and is omitted.
- "Twice as steep" is a corollary of the squared link plus a convention.
- "Composition costs nothing" is an identity, and "products are not learned" was never trained.
- The transformer evidence (2 seeds, a reversal in the third, a readout collapse that also occurs in the run that emerges) is
  described with stronger verbs than the binding pre-registration scope allows.

Several novelty sentences omit the most direct prior: Ren Lemma B.1, Ren Cor. 2.3, Gu eq. (297) and its L = Θ(d) threshold,
and Nishikawa's parameterisation caveat. With an honest rewrite the paper is a solid workshop contribution. As written, I
would expect at least one reviewer to recommend rejection for overclaiming.

**Fair one-sentence summary of contribution:** In a solvable label-linear in-context model with learned single-index features,
finite context adds an alignment-dependent variance term Γ²V(m)/N. That term makes the unaligned state drift-stable when the
readout is held at O(1), and makes the effective exponent N-dependent between 2k* and 4k* when a free scalar readout shrinks
towards its Wiener value, while a norm-tied readout stays at 2k*. A parameter-free population ODE reproduces online SGD at
d=32. The many-skill, composition and transformer results are consistent with but not tests of this mechanism.
