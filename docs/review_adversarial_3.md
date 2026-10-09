# Third adversarial review (theory rigour): "Context length and readout parameterisation select the emergence exponent of in-context skills in an additive model" (Draft v0.7)

Reviewer stance: a COLT/HiLD-style theory reviewer. I care about four things: what is proved, what is asserted, what the
limits are, and whether "exponent" means anything. I do not repeat the first two reviews (`review_adversarial.md`, 5/10;
`review_adversarial_2.md`, 6/10) except to report the status of their top-5 items.

Line numbers refer to `paper/main.tex` and `paper/appendix.tex` at commit `fb669b4` (v0.7, 15-page PDF). My scripts are in
the session scratchpad under `review3/`:
- `checks.py`: sympy redo of the drift and V(m) for σ_1–σ_4, and long integrations of the tied and free flows;
- `closed.py`: closed-form pinned escape time, and the general-k* threshold;
- `integ.py`: the paper's Euler integrator compared with LSODA (rtol 1e-10);
- `thm.py`, `limits.py`: numerical checks of the two theorem statements proposed in §5, and of the order of limits.

---

## 0. Summary for the area chair

The algebra in Props. 1–2 is right, and Prop. 1 is a complete proof. Everything else labelled a proposition is a scaling
heuristic with no stated limit. Three findings are new:

1. **Prop. 3(c) is false for the population flow it is stated for.** With ρ₀ above threshold the flow is a *permanent* trap:
   m ∝ t^{-2} while ρ ∝ t^{-1}, so the release condition is never met. The "delay, not a permanent trap" seen in E5 is an
   SGD-noise effect. The paper's own E5 paragraph (l.228–229) and App. Exp. 7 ("ODE: trap") say so, and contradict the
   proposition.
2. **The "exponent" depends on the order of limits.**
   - At fixed N, η_Γ/η and Γ₀, with d → ∞, the free-readout exponent is 4k* for *every* finite N. N then enters only as a
     1/N prefactor.
   - With N ∝ d^{k*} and η_Γ/η fixed, it tends to 2k*.
   - Only under the joint scaling N = λd^{k*}, η_Γ/η = r̃d does a d-independent function κ(λ, r̃, Γ₀) exist. That function
     is exactly what the d=32 experiments measure: the ODE values at d=32 and at d=2048 under this scaling agree to 0.02.
   - The title's claim is therefore true in a precise sense, but not the sense the paper states. Without that sense it
     cannot be defended against a theorist.
3. **Two quantitative statements in the theory text are wrong.**
   - The "link is squared" identity L_A = L_B[g ↦ g²] (l.127) is false, and the small-m drift coefficient there is missing
     a factor (1 − Γg(1)/2). That is a factor 2 at Γ=1.
   - The tied/pinned threshold "ρ ≪ Nm²/4" (l.169–170, l.228) is a leftover of the corrected factor-2 slip. The correct
     value is ≈ Nm²/2, and the exact value at the E5 cell is 0.115, not 0.0625.

Earlier reviews: review 1's top-5 is mostly addressed. Review 2's Prop.-3 demand (its #1) is only partly addressed, and the
part that was added (the adiabatic/growth-limited split) uses an undefined criterion that is false for one of the E2 cells.

**Score: 5/10 as a theory paper** (7/10 is reachable with the two theorems in §5, which are true; I checked both
numerically).

---

## 1. Status of the earlier reviews' top-5 items

### Review 1 (§5, ranked)

| # | Demand | Status | Evidence in v0.7 |
|---|---|---|---|
| 1 | Rewrite abstract/intro to the evidence | **Mostly fixed** | "11 of 12 … censored within 1.0" (l.33–34): true, but now *undersells*, since the Tobit fit puts all 12 within 0.49 (l.195). "Pinned at order one" qualifier: in. Transformer: "consistent with, but do not test" (l.37–38). Composition "follow by linearity" (l.36). **Open:** the convention dependence of "twice as steep" (a mixture-type in-weight baseline gives equal slopes) is still unstated (l.287–289). |
| 2 | Sharper Prop. 3 with regime conditions, general-k* trap | **Partly** | Three regimes, σ₂ stated (l.155). But: no limit; (b)'s adiabatic criterion is undefined and misapplied (§2.6); (c)'s last sentence is false (§2.7); no general-k* trap (§4). |
| 3 | Free-readout N-vs-prompts exchange, η ∝ B | **Partly** | η ∝ B was run, but only for pinned γ=0.1 (E5, l.232–234). §2.9 shows why this generalises to every protocol at leading order. That argument is missing from the paper. |
| 4 | Report adverse evidence | **Fixed** | d-scan null, P5, P4′ ρ₀=1, post-hoc ODE, Tobit, ratio table, B = 32/8 switch (l.197–198) are all in. |
| 5 | Related-work fixes | **Mostly fixed** | Ren B.1, Ren Cor. 2.3, Gu (297)–(298), BAEVW25 and BMZ are cited. The "bias restores exponent k*" claim is hedged ("would … we do not analyse", l.129), but it is still an unproved assertion inside a theory paragraph. Cut it, or prove it: it is a two-line computation. |

### Review 2 (§3, top-5)

| # | Demand | Status | Evidence in v0.7 |
|---|---|---|---|
| 3.1 | Make Prop. 3 checkable: state the limit and units; (a) exact; split (b); fix the (c) wording; "case (ii)"; the "pins it at Γ*" sentence | **Partly** | Units (flow time) in. (b) split in. "case (ii)" gone. **Not done:** (i) no asymptotic limit anywhere ("Θ", "∝", "saturates", "≈" undefined); (ii) the (a) closed form is not given, although it exists (§2.5); (iii) "pins it at Γ*" is **still in l.184**, contradicted by P5 and by the ODE (Γ/Γ* = 0.24–0.48 at T₀.₅); (iv) the (c) constant is stale and its conclusion false (§2.7). |
| 3.2 | "Never the number of prompts" is a fixed-η artefact | **Fixed** | l.218–220 and the E5 η ∝ B control. |
| 3.3 | Free exponent depends on (N, Γ₀, η_Γ/η), not N alone | **Partly** | E5 now says so (l.226–227). The title, abstract (l.31–32), contribution (2) (l.69) and the Fig. 1 caption still attribute the exponent change to N alone. §3 below gives the formulation that makes the dependence a theorem rather than a confound. |
| 3.4 | Statistics: Tobit, absolute ratios, intervals, disclosure | **Mostly fixed** | Tobit and the ratio table are in (app. l.65–72). Seed ranges are given for E2. The appendix P19 row still says "within 1–3%" (app. l.133). |
| 3.5 | Structure: E-numbering, protocols in §2, lead with E2 | **Partly** | §2 now defines B, η_Γ, Γ₀, ρ₀, T₀.₅ and flow time. Main text uses E1–E5, but the appendix still uses Exp. 1/1b/2/3/6/7 with no mapping line. E2 is still not in the abstract. The page count (15 pp. with appendix; main text ~9) is unchanged. |

**New editorial defect:** l.174–176 and l.177–179 are the same paragraph twice, with the same numbers ("Integrating the flow
… illustrates" / "Integrating the population flow confirms …").

---

## 2. Proposition-by-proposition audit

Notation: s² = E c²; g(1) = E σ²; V(0) = g(1)²; σ² = Σ_j β_j σ_j. For σ = σ_k, sympy gives V = 1 + 2k²m² + … + C(2k,k)m^{2k}:
- k=3: V = 1 + 18m² + 54m⁴ + 20m⁶;
- k=4: V = 1 + 32m² + 216m⁴ + 320m⁶ + 70m⁸.

So β₂² = 2k² and β_{2k}² = C(2k,k).

### 2.1 Prop. 1 (population loss): **proved**

Conditioning on c is exact. Only second moments of c enter. Independence of x_q from the context and Mehler's formula
E[f(z_u)h(z_w)] = Σ f_j h_j m^j give E[A|c] = cg(m) and Var[A|c] = c²(V − g²)/N exactly. The loss depends on w only through m
because E σ(⟨w,x⟩)² = g(1) on the sphere. Nothing is missing.

Cosmetic issues, still open from review 1:
- The leading "1" should be g(1).
- L_B is written for target coefficient 1 and student readout a, but §2 says Model B is fit "on the fixed task c ≡ a".
  Write the target coefficient a and the student readout b: L_B = a²g(1) − 2abg + b²g(1).

This is a theorem. Call it one.

### 2.2 "The link is squared" (l.127–130): **the identity is false; the exponent claim is true**

As N → ∞, L_A = s²[g(1) − 2Γg² + Γ²g(1)g²]. L_B[g ↦ g²] = g(1) − 2ag² + a²g(1) instead has an m-independent quadratic term.
The correct statement is that, as N → ∞, Model A is Model B with the readout replaced by the alignment-dependent readout
a ↦ Γg(m).

The small-m drift is then

  −∂_m L_A = s²(4Γ − 2Γ²g(1)) k* α⁴_{k*} m^{2k*−1} + O(m^{2k*+1}),

not 4Γs²k*α⁴m^{2k*−1}. The bracket is 2 instead of 4 at Γ=1 and 0.38 instead of 0.4 at Γ=0.1. The appendix (l.20) uses
the right coefficient (c_γ = γ(8−4γ) for k*=2), so main text and appendix disagree.

**Safe statement.** For 0 < Γ < 2/g(1), L_A^{(N=∞)}(m) − L_A^{(N=∞)}(0) = −s²Γ(2 − Γg(1))α⁴_{k*}m^{2k*} + O(m^{2k*+1}).
The information exponent of the N=∞ in-context population loss is therefore 2k*, in the sense of BGJ (order of the first
non-vanishing Taylor coefficient of the population loss in m).

### 2.3 The BGJ timescale paragraph (l.131–133): **heuristic, and missing a hypothesis**

"d^{κ−1} with κ = 2k*; in-context needs d³" is the BGJ sample complexity at the *optimal* step size η ≍ d^{−κ/2} (up to
logs), not at the paper's η = η₀/d². It also silently assumes the finite-N repulsion is negligible at m₀ = d^{−1/2}. That
holds iff N ≫ Γk*d^{k*−1}/((2−Γ)α⁴) (from §2.4). At fixed N the pinned model is trapped as d → ∞, so "in-context d³" is
false at fixed N.

**To make it a theorem about online SGD**, apply BGJ (2021, Thm 1.3/1.4) to the loss φ(m) = L_A(m) with N = N(d). It needs:
- (H1) φ′(m) ≤ −c m^{2k*−1} on (d^{−1/2}/C, ε). This holds iff N/(Γd^{k*−1}) → ∞.
- (H2) The per-prompt gradient has bounded 4+ι moments in the v-direction and O(d^{(4+ι)/2}) in norm, uniformly in N.
  This holds: the gradient is a polynomial in Gaussians, and A has variance O(1/N).
- (H3) E c⁸ < ∞.

None of (H1)–(H3) is stated.

### 2.4 Prop. 2 (drift-stable origin): **algebra correct; hypotheses missing; a general-k* version is easy and rigorous**

I redid the drift with sympy:

  −∂_mL_A = s²[γ(8 − 4γ − 20γ/N)m³ − (16γ²/N)m]

This is the same as the paper's form, since 4γ²(1−1/N) + 24γ²/N = 4γ² + 20γ²/N. Hence

  m*² = 4γ / (2N − (N+5)γ).

That is simpler than the paper's l.137 expression and identical to it (0.03252 at γ=1, N=128).

Missing hypotheses:
- The cubic coefficient must be positive, i.e. **γ < 2N/(N+5)**. Otherwise there is no escape from any m₀ < 1.
- "Repulsive for m < m*" should read "points towards m = 0 on 0 < m < m*". "Repulsive" is ambiguous.
- The full flow is ṁ = (1−m²)m(Cm² − B) (σ₂ is even), so (0, m*) is in the basin of 0 and (m*, 1) escapes. This is a
  complete phase portrait. State it.

General-k* version (σ = σ_k): see §4.

### 2.5 Prop. 3(a) (pinned): **"Θ(m₀^{−2})" and "saturates" are imprecise; an exact formula exists**

With u = m², C = γ(8 − 4γ − 20γ/N), B = 16γ²/N and u_e = 1/4, for u₀ > B/C:

  τ_esc = ½ [ F(u_e) − F(u₀) ],  F(u) = −(ln u)/B − ln(1−u)/(C−B) + C ln(Cu − B)/(B(C−B)).

I checked this against quadrature: 744.72 vs 744.72 at d=512, N=128, γ=0.1.

Consequences:
- τ_esc = Θ_δ(m₀^{−2}) holds only for m₀² ≥ (1+δ)m*². Near the threshold τ diverges like (C/(2B(C−B)))·ln(1/(Cu₀ − B)).
- The N-correction is relative O(γd/N), which is the precise meaning of "saturates for N ≫ γd".

**Integrator bias.** The paper's Euler stepper (h = 1% of m/ṁ) overestimates every ODE time by 1–4%:
- pinned d=512: 771/386 vs exact 744.7/379.3;
- E2 pinned-0.1: 414k/197k/176k vs exact 400k/193k/175k;
- E2 free: 1.84M/627k/414k vs 1.82M/623k/409k;
- tied: 17.2k vs 17.0k.

κ_eff is unaffected (to ±0.01). But the "9 of 12 medians below the ODE" pattern (review 2) is partly integrator bias. The
exact pinned predictions are *closer* to SGD (396k/192k/179k).

### 2.6 Prop. 3(b) (free readout): **the adiabatic reduction is a correct heuristic; the criterion is undefined and misapplied**

What is right:
- Γ* is the minimiser.
- By the envelope theorem, the drift on the slow manifold is −d/dm L(m, Γ*(m)).
- The reduced loss −s²g⁴/(g(1)[(1−1/N)g² + V/N]) has Taylor order 4k* at m = 0 for every finite N, with coefficient
  Nα⁸/g(1)³. It has order 2k* at N = ∞.
- The appendix formula for the drift, s²N·4k*α⁸m^{4k*−1}/g(1)³, is correct (V(0) = g(1)²).

What is wrong or missing:

1. **"2r/N ≫ the overlap rate"** has no defined "overlap rate". The relevant ratio is
   ε := (relaxation time)/(escape time) = N/(2rs²g(1)³τ_esc). At the start of the noise-dominated phase,
   ε ≈ 2k*α⁸N²/(r g(1)⁶ d^{2k*−1}). For k* = 2 this is 4N²/(rd³).
2. **"which holds for all parameters used in our SGD runs" (l.165) is false.**
   - At E2, d=64, N=16, the full ODE gives τ = 444.2 and the ODE forced onto Γ = Γ*(m) gives 440.5: adiabatic to 1%.
   - Only N=64 (152 vs 115) and N=256 (99.9 vs 33.6) are growth-limited.
   - So the 4.4 ratio is "N=16 adiabatic, N ≥ 64 growth-limited". It is not a "growth-limited value".
3. **The transient is not discussed.** With Γ₀ ≫ Γ*(m₀) (always true as d → ∞), Γ first decays at rate 2rs²g(1)³/N.
   During that time the repulsion −Γ²g(1)V′/N shrinks m by the exact factor exp(−β₂²Γ₀²/(2r g(1)²)), i.e. e^{−4Γ₀²/r} for σ₂.
   - This is why "free Γ₀ = 1" looks like a trap in review 2's table: the flow does escape, at t ≈ 8.6·10¹⁰ (d=32, N=32),
     after m has fallen to 0.0044.
   - The paper's `T05` returns ∞ as soon as ṁ ≤ 0 at the first step. That stop rule is not a trap test.
4. **"at d=32 the free readout is signal-dominated" (l.176, l.179) is false by the paper's own criterion.**
   m₀^{2k*} = 1/1024 ≪ V/N = 1/32 at N=32, and the same holds for N ≤ 512. The weak N-dependence of τ = 53/37/34/33 there
   is growth limitation (N/2r = 16–1024 against τ ≈ 35–50), not signal dominance.
5. **The adiabatic manifold is valid only in the noise-dominated phase.** In the signal-dominated phase the readout's
   relaxation rate 2rg² is *slower* than the overlap rate 4k*Γm^{2k*−2} whenever r m² < 2k*. This is the N=∞
   "joint slow dynamics" of theory notes §2.3. The escape-time asymptotics survive only because the post-crossover phase
   costs O(N^{(k*−1)/k*}) ≪ d^{2k*−1}/N.

**What makes (b) a theorem** (population flow; Theorem B in §5): fixed N, r, Γ₀ ∈ (0, 2/g(1)), s, link σ_{k*}, with d → ∞.
The lemmas are: linear-ODE Gronwall for Γ − Γ*; the transient factor; monotonicity of the reduced loss; and the
negligibility of the post-crossover phase.

The resulting formula is

  τ_esc = (1+o(1)) · exp((4k*−2)β₂²Γ₀²/(2r g(1)²)) · g(1)³ d^{2k*−1} / (4k*(4k*−2)α⁸_{k*} s² N).

Numerically (σ₂), τ_num/τ_formula at d = 256/1024/4096/16384 is:
- N=32, r=1, Γ₀=0.01: 1.045/1.009/1.002/1.001;
- Γ₀=0.3: 0.566/0.872/0.967/0.992;
- N=128, Γ₀=0.3: 0.047/0.564/0.874/0.967;
- N=32, r=10, Γ₀=0.5: 0.930/0.981/0.995/0.999.

**For online SGD**, there is a new obstruction the paper does not mention. The readout's own SGD noise has stationary
variance Var(Γ) ≈ η_Γ/(4B), independent of N. Because L is quadratic in Γ, it adds an extra repulsion Var(Γ)·g(1)V′(m)/N.
This is negligible iff Var(Γ) ≪ Γ*², i.e. **η_Γ ≪ 4BN²d^{−2k*}**.
- With η_Γ = η₀/d² and k* = 2 the condition is d² ≪ 4BN²/(rη₀). That holds in E2 (4096 ≪ 65536 at N=16).
- It **fails at fixed N as d → ∞**: at the paper's step-size scaling, SGD does not follow the free-readout population flow
  asymptotically.

The w-noise is not the bottleneck here, because every w-gradient carries a factor Γ.

### 2.7 Prop. 3(c) (tied readout): **an exact reduction is available; the last sentence is false; the constant is stale**

1. **Exact identity (not in the paper).** The tied angular velocity equals the pinned velocity at γ = ρ, divided by ρ:
   ṁ_tied(m, ρ) = ṁ_pinned(m; γ=ρ)/ρ. This is immediate from l.31–32 of the appendix.
   - So, with ρ frozen, the tied flow *is* the pinned flow at γ = ρ₀ run 1/ρ₀ times faster.
   - That explains the data exactly: pinned γ=0.01 at 1.72M/1.67M/1.65M steps = 100 × tied 17.2k/16.7k/16.5k.
   - "Tying removes the N-dependence" is therefore "a small readout removes it". Tying buys speed, not N-independence.
     Review 2 made this point empirically; the identity makes it a one-line proof. Ren Lemma B.1 is the reason the
     identity holds.
2. **Stale constant.** From §2.4 with γ ↦ ρ, m² = 4ρ/(2N − (N+5)ρ), so the exact threshold is
   **ρ*(m) = 2Nm²/(4 + (N+5)m²) ≈ Nm²/2, not Nm²/4**. The /4 is the pre-correction formula
   (theory notes §2.2 before 2026-10-08).
   - At the E5 cell (d=64, N=16) the exact threshold is ρ* = 0.1155, against Nm₀²/4 = 0.0625.
   - So "ρ₀ = 0.3 > Nm₀²/4" (l.228) is true, but for the wrong reason. Under the stale constant ρ₀ = 0.1 would also be
     above threshold; it is in fact below it (0.1 < 0.115), which is why it escaped with a 1.9× delay.
   - The appendix writes "ρ₀ ≪ Nm₀²" (no constant).
   - Fix all three places.
3. **"a delay, not a permanent trap" is false for the flow.** For small m with ρ above threshold:
   - ṁ/m ≈ −ρ·2β₂²g(1)/N and ρ̇ ≈ −8ρ²g(1)V(0)s²/N. Hence ρ ≈ N/(8t), m ∝ t^{−k*²/2} (t^{−2} for σ₂).
   - Release would need ρ(t) < ρ*(m(t)) ∝ m^{2k*−2}, i.e. t^{−1} < t^{−k*²(k*−1)}. That never happens.
   - LSODA confirms this at d=64, N=16, ρ₀=0.3: at t = 10²/10⁴/10⁶/10⁸, m = 7.4·10⁻⁴/8.5·10⁻⁸/8.5·10⁻¹²/8.5·10⁻¹⁶ and
     ρ = 1.9·10⁻²/2·10⁻⁴/2·10⁻⁶/2·10⁻⁸; ρ/ρ* grows without bound.
   - The SGD escape after a 30× delay happens because spherical noise keeps m at the scale d^{−1/2} (§2.9), so m cannot
     collapse.
   - Correct statement: *trap in the population flow; under SGD, a delay of order N/(8ρ*(m_floor)) flow time, the time for ρ to decay to the threshold at the noise floor m_floor ≍ d^{−1/2}*.
   - The paper's own l.228–229 and App. Exp. 7 ("ODE: trap") already say the flow traps.

### 2.8 Remark 1 (Kramers): **heuristic; d=32 still quoted against the appendix's own caveat**

The remark still states "d=32 … ΔL/D_eff ≈ 0.02" (l.149), although app. l.37–39 says only d=64 is meaningful. A theorist
will object to the 1-D potential L(m), which omits the spherical term. The clean object is the rescaled SDE of §2.9, in
which the spherical drag appears automatically. There, at N = νd, the Arrhenius exponent scales as B/(η₀γ) and is
independent of d.

Two regimes need to be distinguished:
- At fixed N with d → ∞, the trap is *hard*. Escape needs a large deviation of m to m* = Θ(1), whose probability is
  ≲ exp(−c d m*²) even without the population barrier.
- Only on the ridge N ≍ γd is it Kramers-soft.

State both.

### 2.9 The SGD ↔ population-ODE link (used throughout, never stated)

For k* = 2 and pinned/tied readouts, η = η₀/d² is exactly BGJ's critical step for κ = 4.

In μ = √d·m and s = t/d (t = η·steps), as d → ∞ with N = νd, the drift (Cμ³ − 16γ²μ/ν) is O(1). The per-unit-s martingale
variance is η₀σ_v²/B, and the renormalisation drag is −(η₀σ_⊥²/2B)μ, where σ² is the per-prompt gradient variance (∝ γ²).
Heuristically, the limit is

  dμ = [Cμ³ − (16γ²/ν)μ − θμ] ds + √(2θ) dW,  θ = η₀σ²/(2B)

(using σ_⊥ ≈ σ_v by isotropy at m ≈ 0). The θ-part is the Ornstein–Uhlenbeck process whose stationary law is the uniform
sphere marginal (μ ~ N(0,1)). Three things follow:
- (i) The population ODE is the θ → 0 limit, which explains the 10–20% agreement at η₀/B = 1/32–1/64.
- (ii) The law depends on η and B only through η/B. So the E5 result "η ∝ B leaves flow time invariant to 0.5%" is a
  corollary, and it extends to every protocol. That closes review 1's #3.
- (iii) The trap is metastable with exponent ∝ B/η₀. Larger B at fixed η deepens it, as Remark 1 says.

**For k* ≥ 3 the paper's scaling η = η₀/d² is supercritical** for the pinned/tied 2k*-problem. The critical step is
η ≍ d^{−k*}. Any k*-general SGD statement must use η = η₀d^{−k*}.

Making this rigorous is the BGJ (2022, "effective dynamics and critical scaling") programme applied to the rescaled
statistic.

### 2.10 Appendix derivations

| Item | Verdict |
|---|---|
| Hermite algebra, V for σ₁, σ₂ (app. l.5–7) | correct (sympy) |
| Population loss (l.9–11) | correct; "1" vs g(1) as in §2.1 |
| Drift and m* (l.13–17) | correct; simplify to m*² = 4γ/(2N−(N+5)γ); add γ < 2N/(N+5) |
| Escape (a) (l.19–21) | coefficient right (unlike main l.127); "up to the N-dependent correction … which saturates" → give the closed form |
| Escape (b) (l.21–25) | drift formula right; adiabatic-validity and transient factor missing; "d^{k*} ≫ N" is the noise-dominance condition, **not** the adiabaticity condition. Adiabaticity is N² ≪ r d^{2k*−1} at r = O(1), which is strictly stronger. |
| Escape (c) (l.25–27) | "ρ₀ ≪ Nm₀²": constant missing (≈ 1/2); see §2.7 |
| Numbers at d=512 (l.27–28) | reproduce with the Euler integrator; exact values 3.5% lower for pinned N=128 |
| Population ODEs (l.30–35) | correct, including the 1/‖u‖² chain rule |
| Kramers (l.37–39) | see §2.8 |
| MC drift check (l.41–43) | consistent: the k*=3 sign change is predicted at m* = 0.2406 (exact root, §4); MC is negative at m ≤ 0.2 and positive at 0.4 |

---

## 3. Is "exponent" well-defined?

The paper uses one word for three objects.

1. **Information exponent of a population loss** (BGJ): the order of the first non-zero Taylor coefficient of φ(m) − φ(0).
   - In-weight: k*. In-context, N = ∞: 2k* (§2.2).
   - Pinned or tied at finite N: **2, with the wrong sign** (Γ²g(1)β₂²m²/N > 0). There is then no information exponent in
     the escape sense, only a trap.
   - Free, after minimising out Γ: **4k* for every finite N**.
2. **Effective exponent of a trajectory**: κ_eff := 2 − ∂log τ_esc/∂log m₀, estimated as a secant over m₀ ∈ [0.5, 1.4]d^{−1/2}.
   It depends on d, N, Γ₀, r and on the m₀ window. ODE local slopes in the free N=32 cell are 7.35/6.43/5.62.
3. **The asymptote quoted in the text** ("4 → 8 are the d → ∞ limits", l.202–203). This is wrong as stated, because the
   d → ∞ limit depends on how N and r scale with d. From the ODE (free, Γ₀ = 0.01), κ_eff at d = 32/128/512/2048:

| scaling | d=32 | 128 | 512 | 2048 | limit |
|---|---|---|---|---|---|
| N = 32 fixed, r = 1 | 6.46 | 7.70 | 7.96 | 7.99 | 4k* = 8 |
| N = 4096 fixed, r = 1 | 4.85 | 4.86 | 5.81 | 7.88 | 8 |
| N = d²/32, r = 1 | 6.46 | 5.87 | 5.20 | 4.61 | → 4 (readout freezes) |
| N = 4d², r = 1 | 4.85 | 4.71 | 4.56 | 4.38 | → 4 |
| **N = d²/32, r = d/32** | 6.46 | 6.45 | 6.45 | 6.45 | **6.45** |
| **N = 4d², r = d/32** | 4.85 | 4.82 | 4.82 | 4.83 | **4.83** |

So:
- At fixed N, context length does **not** select the exponent: it is 4k* for all N, and N sets the prefactor (τ ∝ 1/N).
- The N-dependence of an exponent exists only in the joint scaling **λ = N/d^{k*}, r̃ = (η_Γ/η)/d, Γ₀ fixed, m₀ = μ₀d^{−1/2}**.
  There τ_esc·d^{1−k*} → T(μ₀; λ, r̃, Γ₀), and κ(μ₀; λ, r̃, Γ₀) := 2 − ∂log T/∂log μ₀ is a well-defined, d-free function.
  The rescaled limit system (Theorem B(ii)), with dropped terms O(1/d), is:
  - dμ/ds = 4k*α⁴μ^{2k*−1}Γ(1 − Γg(1)/2)·s²
  - dΓ/ds = r̃ s²[2α⁴μ^{2k*} − 2Γg(1)(α⁴μ^{2k*} + V(0)/λ)]

  The finite-N repulsion is O(1/d) in this scaling.
- The d=32 experiments sit at λ = N/1024 and r̃ = 1/32 or 10/32. **The measured κ_eff(N) is an estimate of κ(λ = N/d², r̃, Γ₀)
  at μ₀ ∈ [0.5, 1.4], and the d=32 ODE values equal the d → ∞ values to ≤ 0.06** (6.47/5.46/4.99/4.85 vs limit
  6.45/5.40/4.95/4.83).
- Corners, checked at a d=4096 proxy:
  - λ → 0 (with λ/r̃ → 0): κ → 4k* (7.98);
  - λ → ∞ with Γ₀ ≫ √(r̃/2k*)μ₀: κ → 2k* (4.00);
  - **λ → ∞ with Γ₀ ≪ √(r̃/2k*)μ₀: κ → 2k* + 1** (N = ∞, Γ₀ = 10⁻⁶: 4.97/4.99/5.00 at d = 64/1024/16384). The reason:
    Γ² = (r/2k*)(m² − m₀²), so Γ ∝ m and ṁ ∝ m^{2k*}.
- The paper's lower endpoint "2k*" is therefore not the large-N limit of its own protocol (Γ₀ = 0.01 ≪ 0.18μ₀ at d=32). The
  N=4096 value 4.85 is close to 2k*+1 = 5, not to 4.

**Wording a theorist cannot object to** (for the abstract and Prop. 3(b)):

> "For each fixed context length N the readout-minimised loss has information exponent 4k* (2k* at N = ∞); context length
> enters the escape time as a 1/N prefactor. In the joint limit N = λd^{k*}, η_Γ/η = r̃d, the escape time has a d-free limit
> whose effective exponent κ(λ, r̃, Γ₀) interpolates between 4k* (λ → 0) and 2k* or 2k*+1 (λ → ∞, depending on the readout's
> initial scale); our d=32 measurements estimate this function."

Rename κ_eff to "secant exponent" everywhere. Change "rises from 2k* towards 4k*" to "rises from ≈2k*+1 towards 4k* at
fixed (r̃, Γ₀)".

---

## 4. Are the general-k* claims supported?

Every derivation beyond Prop. 1 is σ₂. Every SGD run is k* = 2. The only k* = 3 evidence is the MC drift check. Here is
what is and is not safe.

**Safe and provable now (σ = σ_k, k ≥ 2; I checked k = 2, 3, 4):**

- **(G1) Prop. 1 for any σ ∈ L⁴(γ).** Already proved.
- **(G2) Pinned trap, exact.** Write x = m². The drift satisfies −∂_mL_A/m = s²[a x^{k−1} − Σ_{i=0}^{k−1} b_i x^i], with
  a = k(4Γ − 2Γ²(1−1/N)) and b_i = (Γ²/N)(2i+2)β²_{2i+2} ≥ 0, b₀ = 4k²Γ²/N > 0.
  - If **Γ < 2N/(N + C(2k,k) − 1)**, the leading coefficient a − b_{k−1} is positive.
  - Descartes' rule then gives **exactly one** positive root x*. The origin is drift-stable on (0, m*) and escape occurs
    on (m*, 1).
  - Asymptotically, m*^{2(k−1)} = 2kΓ/((2−Γ)N)·(1 + O(N^{−1/(k−1)})). Note k, not k² as review 1 wrote.
  - At practical N the correction is not small. For k=3, Γ=0.1, N=128: exact m* = 0.2406 vs leading order 0.2229. For k=4:
    0.543 vs 0.386.
  - Threshold in d: d* = m*^{−2} ≍ (N(2−Γ)/(2kΓ))^{1/(k−1)}. For k* = 3, Γ = 0.1, N = 128 that is d* ≈ 17, against 607 at
    k* = 2. **The "harmless" small-pinned-readout regime of k* = 2 is trapped already at d ≳ 17 for k* = 3.** Say this; it
    strengthens the paper.
- **(G3) Tied ≡ pinned under a time change (§2.7.1), for any k.** With ρ ≪ ρ*(m₀) ≍ N m₀^{2k−2}, τ_tied = ρ₀τ_pinned(ρ₀) =
  Θ(m₀^{−(2k−2)}). The N-dependence has relative size O(ρ₀d^{k−1}/N). With ρ₀ > ρ*(m₀) the flow is permanently trapped,
  with m ∝ t^{−k²/2}.
- **(G4) Free readout, fixed N, d → ∞ (Theorem B(i))**, with exponent 4k* and τ ∝ d^{2k*−1}/N. This needs Lemma B3
  (monotonicity of the reduced loss). For σ_k it holds because mV′ ≤ 2kV < 4kV = 4(mg′/g)V. The adiabatic free flow
  therefore has **no** spurious minimum for any k. That is a clean contrast with (G2) worth stating: minimising out the
  readout removes the trap.
- **(G5) For a general link with information exponent k* and σ² even**, statements (G2)–(G4) hold locally, with β₂ in place
  of √2k*, provided β₂ ≠ 0 (generic). If σ has mixed parity then β₁ may be ≠ 0, V′(0) ≠ 0, and the origin is not even a
  critical point. Exclude this case explicitly.

**Not supported:**
- Any SGD statement for k* ≥ 3 at η = η₀/d², which is supercritical (§2.9).
- "Exponent 4k* … τ ∝ m₀^{−(4k*−2)}/N" (l.163–164) as a statement about SGD for k* ≥ 3.
- The pinned "N-dependence saturates for N ≫ γd" for k* ≥ 3. The correct scale is N ≫ γd^{k*−1}.
- The many-skill T_p ∝ 1/π_p for general k*. It needs the decoupling lemma, and that is plausible precisely *because* the
  in-context effective exponent 2k* ≥ 4 puts the problem in Ren et al.'s k > 2 regime. The theory notes flag this; the
  paper does not say it.

**Recommended scoping sentence:** "Props. 1 and 2′ (the σ_k version, (G2)) and Theorem B hold for every k* ≥ 2; the
numerical and SGD results are for k* = 2, where η = η₀/d² is the critical step size."

---

## 5. The two results that deserve formal theorems

### Theorem A (finite-context trap and its dependence on the readout parameterisation; population flow, σ = σ_k, k ≥ 2)

**Statement.** Let ṁ = −(1−m²)∂_mL_A with pinned Γ ∈ (0, Γ_max(N,k)), Γ_max = 2N/(N + C(2k,k) − 1).

- (i) There is a unique m* ∈ (0, 1) such that m(t) → 0 if m₀ < m*, and m(t) reaches 1/2 in finite time if m₀ > m*.
  m* is the unique positive root of the polynomial in (G2), and m*^{2(k−1)} = 2kΓ/((2−Γ)N)(1 + o(1)) as N → ∞.
- (ii) For k = 2 the escape time is the closed form of §2.5.
- (iii) Tied readout with ρ(0) = ρ₀: if ρ₀ < ρ*(m₀)(1 − δ), then τ_tied = ρ₀τ_pinned(Γ=ρ₀)(1 + O(ρ₀d^{k−1}/N)). If
  ρ₀ > ρ*(m₀), then m(t) → 0 with m(t) ≍ t^{−k²/2} and ρ(t) ≍ N/(8t).
- (iv) Corollary for online SGD at fixed N, d → ∞, from uniform initialisation: for any η ≤ η₀d^{−k} and horizon
  exp(c d m*²), P(max_{n ≤ horizon} m_n ≥ m*/2) → 0. More prompts per step do not help at any η/B.

**Lemma list.**
1. Closed-form L_A (Prop. 1) and the Hermite coefficients β₂² = 2k², β_{2k}² = C(2k,k).
2. Sign structure of drift/m as a polynomial in m²; Descartes ⇒ unique root; a 1-D phase portrait on [0, 1].
3. Partial-fraction integral for k = 2 (§2.5).
4. Time-change identity ṁ_tied(m,ρ) = ṁ_pinned(m;ρ)/ρ (Ren Lemma B.1 + chain rule).
5. Comparison lemma for (m, ρ): ρ is decreasing while m ≤ m*(ρ), ρ ≤ ρ₀/(1 + 8ρ₀ s²g(1)V(0)t/N), and the Gronwall bound
   m(t) ≤ m₀ exp(−(2β₂²g(1)/N)∫ρ). Hence ρ/ρ*(m) is non-decreasing (permanent trap).
6. (SGD) A Lyapunov/supermartingale bound for f(m) = exp(a d m²) on {m ≤ m*}, using a negative population drift plus the
   spherical drag (BGJ 2021, §5-type bounded-increment martingale estimates).

Every piece is elementary. Items 1–5 fit in one appendix page. This makes the paper's central qualitative claim, that "a
readout held at O(1) makes context length a hard requirement", a theorem for all k* ≥ 2, together with its exact
threshold.

### Theorem B (free readout: exponent selection by λ = N/d^{k*}; population flow)

**Statement.** Free readout Γ̇ = −r∂_ΓL_A, Γ(0) = Γ₀ ∈ (0, 2/g(1)), σ = σ_k, k ≥ 2, m₀ = μ₀d^{−1/2}.

- (i) *Fixed N, r, Γ₀; d → ∞:* τ_esc = (1+o(1)) · exp((4k−2)β₂²Γ₀²/(2r g(1)²)) · g(1)³ μ₀^{−(4k−2)} d^{2k−1}/(4k(4k−2)α⁸s²N).
  In particular the information exponent is 4k for every finite N, and N is a pure 1/N prefactor (the Oko-type trade-off).
  Numerically verified in §2.6.
- (ii) *Joint scaling N = λd^k, r = r̃d, Γ₀ fixed:* τ_esc·d^{1−k} → T(μ₀; λ, r̃, Γ₀), the hitting time of μ = ∞ for the
  2-D system in §3. Verified: τ/d = 1.636/1.505/1.441/1.410/1.394 at d = 32…512 for λ = r̃ = 1/32, and d-flat κ in §3.
- (iii) *Corners:* κ → 4k as λ → 0 with λ/r̃ → 0; κ → 2k as λ → ∞ with Γ₀/(√r̃ μ₀) → ∞; κ → 2k+1 as λ → ∞ with
  Γ₀/(√r̃ μ₀) → 0.

**Lemma list.**
1. The reduced loss L(m, Γ*(m)) and its expansion. Envelope theorem: ∂_mL(m, Γ*(m)) = d/dm L(m, Γ*(m)).
2. Uniform fast-variable contraction: Γ̇ = −λ(m)(Γ − Γ*(m)) is linear in Γ, with λ(m) ≥ λ₀ = 2rs²g(1)³/N. Gronwall gives
   |Γ − Γ*| ≤ e^{−λ₀t}|Γ₀ − Γ*(m₀)| + sup|dΓ*/dt|/λ₀, with relative error O(ε), ε = N(ṁ/m)/λ₀ (Tikhonov without Fenichel
   machinery).
3. Transient lemma: during t ≲ (N/2r)·log(Γ₀d^k/N), the attraction/repulsion ratio is O(N/(Γd^{k−1})) → 0, so
   log(m_end/m₀) = −β₂²Γ₀²/(2rg(1)²) + o(1).
4. Monotonicity: d/dm[g⁴/((1−1/N)g² + V/N)] > 0 on (0, 1) for σ_k (mV′ ≤ 2kV). Hence no spurious minimum.
5. Phase-splitting: the time spent above m = δ is O_δ(1)·poly(N) = o(d^{2k−1}/N). The 4k-power integral is dominated by
   its lower limit.
6. (ii) Rescaling μ = √d m, s = t d^{1−k}; uniform-on-compacts convergence of the vector field with an O(1/d) remainder.
   Blow-up of dμ/ds ∝ μ^{2k−1} in finite s (k ≥ 2) gives a finite T. Continuity of the hitting time in parameters gives
   continuity of κ.
7. (iii) Explicit integration in the corner limits. As λ → ∞ and Γ ≪ 1, dΓ/dμ ≈ r̃μ/(2kΓ), so Γ² − (r̃/2k)μ² is an
   approximate first integral; hence Γ ∝ μ, dμ/ds ∝ μ^{2k} and κ = 2k+1.

**SGD transfer (separate, harder; state it as a conjecture or under hypotheses).** Γ-noise control needs η_Γ ≪ 4BN²d^{−2k}
(§2.6). w-noise control needs η ≪ B/N in the noise-dominated phase, because the gradient carries a factor Γ. These are
satisfied in E2 but not uniformly as d → ∞ at η = η₀/d². Write: "the population flow describes online SGD whenever
η_Γ ≪ BN²d^{−2k*} and η ≪ min(B/N, d^{−k*}); our runs satisfy both by factors ≥ 4".

Theorem B turns the title claim, the E5 "(N, Γ₀, η_Γ/η)" finding and the d=32 κ table into one statement. Theorem A turns
the trap into a statement for every k* ≥ 2.

---

## 6. My top-5 issues

1. **Prop. 3(c) is false for the population flow** (§2.7.3). The tied trap above threshold is permanent in the flow; the
   delay is SGD-specific. The constant Nm²/4 is stale (should be ≈ Nm²/2; exact 0.115 at the E5 cell). Fix the
   proposition, l.228 and app. l.27, and add the time-change identity ṁ_tied = ṁ_pinned(γ=ρ)/ρ.
2. **"Exponent" is order-of-limits dependent** (§3). At fixed N the free-readout exponent is 4k* for all N. The paper's
   N-dependence exists only in the scaling N ∝ d^{k*}, η_Γ/η ∝ d, which the d=32 data estimate to ≤ 0.06. "4 → 8 are the
   d → ∞ limits" (l.202–203) is wrong as stated. The protocol's large-N endpoint is 2k*+1, not 2k*.
3. **No proposition beyond Prop. 1 has a stated limit or hypotheses.** Prop. 2 lacks γ < 2N/(N+5). Prop. 3(a) has "Θ" and
   "saturates" undefined, although an exact formula exists. Prop. 3(b)'s adiabatic criterion "2r/N ≫ overlap rate" is
   undefined, and "holds for all our SGD runs" is false (E2 N=16 is adiabatic to 1%). "Signal-dominated at d=32" is false.
   The transient factor exp(4Γ₀²/r) is absent.
4. **The "link is squared" identity is false, and its drift coefficient is off by (1 − Γg(1)/2)** (§2.2), a factor 2 at
   Γ=1. The main text disagrees with the appendix. The BGJ timescale paragraph omits N ≫ Γd^{k*−1} and the step-size
   assumption (§2.3).
5. **The k*-general claims rest on σ₂ algebra, and the SGD step size is critical only for k* = 2** (§4, §2.9). The exact
   σ_k trap (unique root, Γ_max, d* ≈ 17 at k*=3, Γ=0.1) is cheap and strengthens the paper. Any SGD claim for k* ≥ 3 needs
   η ≍ d^{−k*}. The free-readout SGD transfer has an unmentioned Γ-noise condition, η_Γ ≪ BN²d^{−2k*}.

Smaller items:
- The paragraph at l.174–179 is duplicated.
- l.184 "pins it at Γ*" survives (review 2).
- Remark 1 still quotes the d=32 ratio.
- The Euler integrator is biased by 1–4% upward; switch to LSODA, which makes pinned and free predictions closer to SGD.
- L_B notation.
- "1" vs g(1).
- The T05 "dm ≤ 0 ⇒ ∞" stop rule is not a trap test (free Γ₀ = 1 escapes at t ≈ 10¹¹).

---

## 7. Score and the three highest-value edits

**Score: 5/10 from a theory reviewer.** For the paper:
- Prop. 1 is a correct closed form, and the alignment-dependent variance term is a real, well-identified mechanism.
- The population ODE is unusually well validated against SGD.
- The two theorems in §5 are true (checked numerically) and within reach.

Against:
- Apart from Prop. 1, the "propositions" are scaling heuristics without limits.
- One of them (3(c)) is false for its stated object.
- The headline notion, "exponent", is not defined in a way that survives a change in the order of limits.
- The k*-general phrasing outruns the σ₂ derivations.

With the edits below I would score it 7.

1. **Replace Prop. 3 by Theorems A and B (§5), with the limits stated.**
   - Theorem A: exact σ_k threshold, k=2 closed form, tied = pinned time change, permanent flow trap above threshold.
   - Theorem B: fixed-N formula with transient factor; joint-scaling limit and corners.
   - Put proofs (lemma lists 1–5 and 1–7) in the appendix. Keep the SGD statements as clearly labelled corollaries or
     conjectures with their step-size and Γ-noise hypotheses.
2. **Define the exponent as in §3 and rewrite title, abstract, contribution (2), Fig. 1 caption and l.202–203 accordingly.**
   - Context length enters through λ = N/d^{k*}: for fixed N it is a 1/N prefactor on a 4k* law.
   - κ_eff is a secant estimate of κ(λ, r̃, Γ₀).
   - The protocol's large-N endpoint is 2k*+1.
   - Show the d-flat column of §3 (one line in the appendix) as evidence that d=32 measures the limit object.
3. **Fix the false and stale statements.**
   - Prop. 3(c)'s "delay, not trap", the Nm²/4 constant (three places), "signal-dominated at d=32", "growth-limited holds
     for all runs".
   - The L_A = L_B[g ↦ g²] identity and drift coefficient.
   - γ < 2N/(N+5) in Prop. 2.
   - The duplicate paragraph and l.184.
   - Re-run the ODE predictions with an accurate integrator, which moves them 1–4% toward the SGD medians.
