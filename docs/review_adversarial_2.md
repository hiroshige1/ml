# Second adversarial review: "Context length and readout parameterisation select the emergence exponent of in-context skills in an additive model" (Draft v0.5)

Reviewer stance: an independent second reviewer. I focus on statistics and methods, on whether Prop. 3 can be checked, on
structure and on writing. I do not repeat the first review (`docs/review_adversarial.md`, 5/10) except where its points are
still open.

Line numbers refer to `paper/main.tex` and `paper/appendix.tex` at commit `c8ba6d9` (07:58, with the `fig:regimes` figure
added). Main-text line numbers after l.205 moved by +15 when that figure was inserted.

I recomputed everything with my own scripts, in the session scratchpad under `review2/`:
- `kfit.py`: OLS, Tobit and bootstrap fits of κ_eff from `results/exp1b/summary.csv`;
- `ode.py`, `ode2.py`, `ode3.py`: the authors' ODE right-hand sides (`scripts/ode_kappa_exp1b.py`), run per cell and under
  different Γ₀, η_Γ/η and ρ₀, plus an adiabatic Γ = Γ*(m) variant;
- inline scripts for `exp6/summary.csv`, `exp2/per_skill.csv` and `exp1b/dscan.csv`.

---

## 0. Summary for the area chair

v0.5 is much more honest than v0.2. Exp. 2 (repo `exp6`) is the paper's best evidence, and it was genuinely pre-registered:
the P19 commit `89e8869` precedes the `exp6` runs, whose stdout files are timestamped 03:42 and later. In it, the population
ODE predicts SGD escape steps in 15 of 18 cells to within the 3-seed resolution.

The remaining problems are of a different kind from the first review's:
1. **Prop. 3 is not stated precisely enough to check.** Two of its sentences contradict the paper's own ODE and data.
2. **The "prompts do not substitute" conclusion is built into the design.** The step size η is held fixed while B varies.
3. **The free-readout exponent is not identified as a function of N alone.** At the experimental d it moves as much with Γ₀
   and η_Γ/η as with N.
4. **Statistics.** Precision claims ("1–3%") exceed what 3 seeds resolve, censoring is handled ad hoc, and some design switches
   are undisclosed.
5. **Structure.** Two different experiments are both called "Experiment 2". The order "prediction, then run" is not visible to
   the reader.

**Score: 6/10** (weak accept for a workshop, conditional on the Prop. 3 rewrite and the fixed-η caveat).

---

## 1. Status of the first review's five attacks (its §5, ranked)

| # | First-review demand | Status in v0.5 | Evidence |
|---|---|---|---|
| 1 | **Rewrite abstract / intro to the evidence.** (a) fix "12/12"; (b) restrict non-substitutability to an O(1) readout and say the free readout restores Oko's N×T trade-off; (c) transformer as "consistent with, not a confirmation"; (d) "twice as steep" as a corollary with its convention dependence; (e) demote additive composition and drop the product claim | **Mostly fixed** | (a) abstract l.33–34 "11 of 12 cells within 0.6, the censored cell within 1.0", true (I get Δ ≤ 0.41 in 11 cells, 1.02 in the 12th). (b) Fixed in the abstract (l.30–32), contribution (2) (l.65–71) and Prop. 3(b). **But Related work l.311–312 still says Oko's Remark 3 is what "Prop. 2 and Fig. 3 contradict beyond the one-step regime"**, which contradicts the new contribution (2) (l.69–71: "the multiplicative trade-off … reappears"). (c) Fixed (l.37, l.302–305); the 2-seed caveat is in. (d) Partly: called "a corollary of the squared link" (l.244–245). The convention dependence (choice a_p² ∝ π_p; Ren Cor. 2.3) is still unstated. (e) Partly: the abstract still says "additive compositions of learned skills cost nothing" (l.35), an identity presented as a finding. The product claim survives with a caveat (l.269). |
| 2 | **Sharper Prop. 3** with explicit regime conditions, an adiabatic condition and a non-adiabatic regime for (a) free; (b) tied citing Ren B.1; (c) a general-k* trap condition | **Partly** | Prop. 3 is now three regimes (l.148–165) and cites Ren Lemma B.1 (l.162). **Not done:** (i) the adiabaticity condition. Prop. 3(b) instead asserts relaxation is "fast … as soon as N < ∞" (l.155), which is false at the experimental parameters (§3.1 below). (ii) The non-adiabatic regime that P5 shows the experiments are in. (iii) A general-k* trap condition: (a) points to m* of Prop. 2, which is σ₂-only, although Prop. 3 opens with "Let k* ≥ 2". |
| 3 | **One extra experiment:** free-readout N-vs-prompts exchange at equal tokens, *η scaled with B*, d ∈ {48, 64} | **Partly** | Exp. 2 / `exp6` (l.195–203) runs equal tokens at d=64 with four protocols, which is good. **But η = 1/d² is fixed for every B**, so the exchange the first review asked for (prompts counted as steps × B, η ∝ B) was not run. The P19 text itself says "the population drift does not depend on B; only the noise does". See §3.2. |
| 4 | **Report unused and adverse evidence** | **Mostly fixed** | In now: the d-scan null (l.190–192), P5 (l.189), P4′ ρ₀=1 at d=32 and the d=64 trap (l.192–193), post-hoc ODE evaluation (l.182–183), one absolute-time comparison (l.186). **Still missing:** (i) the B = 32 → 8 switch at N ≥ 512 and the different step caps in the κ table (app. l.61–63 says nothing; app. l.48 says "B = 32 unless stated"); (ii) a survival/Tobit fit; (iii) absolute times for all cells, not only the best one (free, N=128); (iv) η₀-invariance, the pre-declared admissibility check, never run. |
| 5 | **Fix related work** | **Partly** | Done: Ren Lemma B.1, Gu eqs. (297)–(298) and N = Θ(d), the Nishikawa caveat (l.63), He et al. recast (l.324–326), BAEVW25 (l.320–321). **Not done:** Ren Cor. 2.3 for the doubled ordering; Berthier–Montanari–Zhou §8. **The underived claim "A bias added to A_j restores an in-weight path of exponent k*" (l.124–125) is still there.** |

Smaller first-review items: the 1/‖u‖² wording is fixed (app. l.33). The Kramers d=32 ratio is caveated in the appendix
(app. l.37–39) but still stated as meaningful in Remark 1 (l.144). "Equals the single-feature value" for 0.73 vs 0.64 (l.254),
a 14% gap, is unchanged.

---

## 2. Numbers I recomputed from the CSVs

| # | Paper claim (line) | Source | My recomputation | Verdict |
|---|---|---|---|---|
| R1 | κ_eff free 4.8–6.6, tied 4.10–4.44 (l.183–184) | `exp1b/summary.csv`, OLS on reached runs | free1 5.45/5.59/4.81/4.92; free10 6.61/5.78/4.92/5.17; tied 4.10/4.26/4.30/4.44 (N = 32/128/512/4096); SEs 0.10–0.30 | **reproduced exactly** |
| R2 | ODE 4.85–7.0; "eleven of twelve within 0.6"; censored cell 1.0 below, 0.55 if set to cap (l.184–185) | `scripts/ode_kappa_exp1b.py` | ODE free1 6.47/5.46/4.99/4.85, free10 7.02/6.18/5.32/4.92, tied 4.18/4.15/4.15/4.15. \|Δ\| in 11 cells ≤ 0.41; free1 N=32 Δ = 1.02; cap-filled 5.92 (Δ = 0.55) | **reproduced**. But see §3.4: with a censored-data (Tobit) fit, **all 12 cells are within 0.49**. The paper undersells this with an ad hoc estimator. |
| R3 | absolute times free N=128: 565/113/36/13 vs ODE 473/120/37/13 (l.186) | same | reproduced. Over all 45 uncensored (protocol, N, m₀) medians, SGD/ODE is **0.64–1.34**; about half (24/45) are within 10%; tied cells are the noisiest (0.64–1.34) | reproduced; **the quoted cell is the best one**. Report the full range. |
| R4 | Exp. 2 medians and ODE: pinned-0.1 396k/192k/179k vs 414k/197k/176k; free 1.78M/614k/402k vs 1.84M/627k/414k; T(16)/T(256) = 4.4 (ODE 4.44); tied 17.0k/15.1k/14.0k vs 17.2k/16.7k/16.5k (l.199–201) | `exp6/summary.csv`; my ODE run | All reproduced (ODE flow times 101.1/47.8/43.4, 448.1/153.4/100.7, 4.21/4.07/4.03, × 4096). These are the **tokens-matched** medians; B=64 gives 1.88M/614k/411k and 18.9k/15.1k/16.3k. Seed-resampling range of T(16)/T(256): free 3.6–5.0 (tok), 4.0–5.3 (B64); tied 1.01–1.30 (tok), 0.91–1.45 (B64); pinned-0.1 1.94–3.57 (tok), 2.08–2.53 (B64) | reproduced. **Intervals not reported** (§3.4). |
| R5 | "free within 1–3% of ODE" (app. Table P19) | `exp6/summary.csv` | median/ODE = 0.97, 0.98, 0.97, 1.02, 0.98, 0.99. Within-cell sd(log T) = 0.03–0.14, pooled 0.075, so the s.e. of a 3-seed median is ≈ 5%. 9 of the 12 pinned-0.1/free medians lie below the ODE | numbers right; **the precision implied is not**. Say "within one s.e. (≈5%)". |
| R6 | d=512 ODE: pinned-0.1 ∞/771/386; free 9.0·10⁴/2.2·10⁴/5.4·10³, τN = 2.9/2.8/2.75·10⁶; tied 39/34/33; d=32 free 53/37/34/33 (l.166–168) | my ODE run | ∞/770.5/385.5; 90 278/22 022/5 371, τN = 2.89/2.82/2.75·10⁶; 39.3/34.1/33.1; 52.8/37.1/34.0/33.2 | **reproduced** |
| R7 | many-skill slopes 1.44 ± 0.08, 1.53 ± 0.22, 0.75 ± 0.08; K = 0.73 (IQR 0.70–0.77) (l.246–254) | `exp2/per_skill.csv` | 1.435 ± 0.082 (n=28; per-seed 1.46/1.26/1.56, seed-level s.e. 0.09); 1.534 ± 0.221 (n=23); 0.750 ± 0.076 (n=25); K median 0.734, IQR **0.70–0.78** | reproduced (IQR upper end 0.78, not 0.77) |
| R8 | learned Γ at T_0.5 = 0.18–0.25 vs Γ*(0.5;N) = 0.38–0.99 (l.189) | `exp1b` README; ODE | The **ODE gives the same Γ at T_0.5**: 0.18 (N=32), 0.23 (128), 0.24 (512), 0.25 (4096). So the failure of P5 is predicted by the paper's own ODE, not an SGD artefact. Along the ODE path Γ/Γ*(m) at m = 0.5 is 0.48 (N=32), 0.32 (N=128), 0.24 (N=4096) | numbers right; **the interpretation in Prop. 3(b) is wrong** (§3.1) |
| R9 | d-scan null "is what the ODE predicts for d ≤ 48" (l.191) | `exp1b/dscan.csv`; ODE | T(128)/T(2048), data: 1.38/1.05/1.14/1.05 (d = 16/24/32/48). ODE: 1.05/1.08/1.12/1.20; the ODE ratio reaches 2 only at d=128 | correct |

---

## 3. My top-5 issues

### 3.1 Prop. 3 cannot be checked as stated, and two of its sentences contradict the paper's own ODE (§3, l.148–175; app. l.19–28)

What is wrong:

1. **No asymptotic regime.** "≍", "τ_esc ∝ 1/N" and "saturates in N" have no stated limit. Is it d → ∞ at fixed N, or N =
   N(d)? Is τ_esc in flow time (steps × η) or in steps? Is η = η₀/d² assumed? Is s² fixed? The appendix gives the regime
   condition in terms of d (d^{k*} ≫ N, app. l.24); the proposition gives it only in m (l.157–158).
2. **(a) is σ₂-specific but claimed for k* ≥ 2.** m*(γ,N) "of Prop. 2" exists only for σ₂. "A prefactor that saturates in N"
   is also vague. At d=64 the pinned-0.1 escape time changes ×2.2 between N=16 and 256 (both the ODE and SGD show this), and
   ×2 between N=128 and 512 at d=512. For σ₂ the 1-D ODE ṁ ≈ A m³ − B m (A, B from Prop. 2) integrates in closed form, so
   (a) can be stated exactly.
3. **(b) "Γ relaxes … fast relative to the overlap dynamics as soon as N < ∞" (l.155) is false at every SGD parameter in
   the paper.** The relaxation time in flow units is ≈ N/(2r g(1)V), with r = η_Γ/η. With r = 1 this is 128 at N=256
   against τ_esc = 101 (Exp. 2), and 2048 at N=4096 against τ_esc ≤ 234 (Exp. 1). The ODE confirms that Γ is never near Γ*
   (R8).
   - The condition that is actually needed is adiabaticity, N ≪ 2r g(1)V τ_esc. It is missing.
   - Consequence: **Exp. 2's free-readout ratio 4.4 does not test the 1/N law of Prop. 3(b).** The cells satisfy (b)'s stated
     condition (d^{k*} = 4096 ≫ N ≤ 256), so (b) as written predicts T(16)/T(256) = 16. Forcing Γ = Γ*(m) in the ODE gives
     13.2. The ODE with the experiment's Γ₀ = 0.01, r = 1 gives 4.44, i.e. T ∝ N^{−0.54}.
   - So Exp. 2 confirms the ODE, not the proposition. "ratio T(16)/T(256) = 4.4 as predicted (4.44)" (l.200) reads as if
     Prop. 3 had predicted it.
   - The 1/N law is shown only by the ODE at d=512 (l.167), where τ ≫ N/2 and the readout is adiabatic.
4. **(c) "ρ̇ = −4ρ∂_ρL_A keeps ρ at Γ* ≪ 1 from a small initialisation" (l.162) is contradicted by the data.**
   - ρ hardly moves: it is 0.0110–0.0116 at T_0.5 in every `exp6` tied cell and in the ODE (ρ₀ = 0.01), whereas Γ*(0.5; N=16)
     = 0.23.
   - What makes (c) N-independent is ρ ≈ ρ₀ ≪ N m₀². The paper states this condition only in the appendix (app. l.27) as
     "ρ₀ ≪ N m₀²".
   - In the ODE at d=64, ρ₀ = 0.1 already gives escape times 8.1/4.8/4.3 (ratio 1.9), and ρ₀ = 0.3 traps at N=16.
   - A **pinned γ = 0.01** readout is equally N-flat in the ODE (421/407/403, ratio 1.04).
   - The tied protocol at ρ₀ = 0.01 is therefore not distinguished from "a tiny readout". What tying adds is speed (the
     angular step is η/ρ ≈ 100η), not N-independence.
   - Yet the abstract says "a readout tied to the feature norm removes the N-dependence altogether" (l.32–33), and
     l.192–193 shows that tied with ρ₀ = 1 is trapped at d=64.
5. **Dangling reference.** "One-step analyses sit in case (ii) by construction" (l.174) points to an item that does not exist
   (the items are (a)–(c)). The sentence before it (l.172–173, "the finite-context noise makes the readout fast and pins it
   at Γ*") is contradicted by R8.
6. **No proof.** "Proposition" with no proof. App. l.19–28 is a scaling heuristic.

**Fix.**
- Recast as "Claim 3 (informal)", or add proof sketches, and state the limit: d → ∞; N fixed or N = d^ν; η = η₀/d²; τ in flow
  time.
- (a): σ₂ only, with the closed-form τ(m₀; γ, N) from ṁ = A m³ − B m. Also give the general-k* threshold
  m*^{2(k*−1)} ∝ γ k*²/N (first review §1.1).
- (b): split into (b1) adiabatic, N ≪ r τ_esc ⇒ τ ≍ m₀^{−(4k*−2)}/N when d^{k*} ≫ N; and (b2) readout-growth-limited,
  N ≳ r τ_esc ⇒ τ is set by Γ growing from Γ₀ (Γ̇ ≈ 2r g²). (b2) is where both SGD experiments are. Say plainly that Exp. 2
  sits between (b1) and (b2) and tests the ODE, not the 1/N law.
- (c): replace "keeps ρ at Γ*" with "ρ stays ≈ ρ₀ while ρ₀ ≪ N m₀², and the repulsion is then negligible". Move that
  condition into the statement.
- Fix "case (ii)".

### 3.2 "Never the number of prompts per step" follows from holding η fixed (l.201–202; abstract l.30–31; Fig. 3 caption l.235; contribution l.67)

- In Exp. 2, η = 1/d² for every B. In a drift-dominated regime the step count is then set by the population ODE, which does
  not contain B. The pre-registration says exactly this (P19 setting: "the population drift does not depend on B; only the
  noise does").
- So "In every protocol the two batch schemes give the same step counts: what trades off against context length is training
  time, never the number of prompts per step" is a property of the design, not a finding. It holds trivially for pinned-0.1,
  free and tied.
- At N=64 the two schemes are the same three runs (app. l.67), so only N=16 and N=256 carry any comparison.
- The B-invariance is informative in exactly one place, the pinned-γ=1 trap. There a larger B only reduces noise, and
  Remark 1 itself says that deepens the trap.
- Oko's Remark 3 concerns total sample count. Whether more prompts substitute is the question whether a larger B allows a
  proportionally larger η (linear scaling) without breaking the drift-dominated regime. That was not tested.
- P7 (Fig. 3 right) has the same confound plus a ×10 cut in flow time (first review §4.7). It is still captioned "more prompts
  do not substitute for context length".
- The transformer section notes the analogous issue ("the Adam step was not rescaled with B", l.303). The toy model, where it
  is cheap to fix, does not.

**Fix (cheap).**
- Free readout, d=64, N=16: B ∈ {64, 256} with η ∈ {1, 4}/d². If the flow-time ODE still holds at η = 4/d², the B=256 run
  escapes in ≈ 1.84M/4 ≈ 460k steps, i.e. prompts *do* substitute through η. If it does not hold, the paper has a real
  noise-limited statement. About 0.3 CPU-h.
- Otherwise rewrite l.201–202 to: "At fixed step size, the step count depends on N and not on B, as the population drift
  predicts; whether a larger B buys a larger step was not tested." Qualify the abstract and the Fig. 3 caption the same way.

### 3.3 The free-readout "exponent" is not identified as a function of N: Γ₀ and η_Γ/η move it as much (title; abstract l.31–32; l.69; Fig. 1)

The title says context length "selects the emergence exponent". The only evidence that N moves an exponent is κ_eff(N) for
the free readout at d=32; the paper itself concedes "established only through κ_eff(N) at fixed d" (l.192). The paper's own
ODE, evaluated at the experimental d=32 m₀-grid, gives:

| ODE κ_eff | N=32 | 128 | 512 | 4096 |
|---|---|---|---|---|
| Γ₀=0.01, r=1 (Exp. 1) | 6.47 | 5.46 | 4.99 | 4.85 |
| Γ₀=0.1, r=1 | 6.73 | 4.97 | 4.46 | 4.35 |
| Γ₀=0.001, r=1 | 6.44 | 5.51 | 5.10 | 4.98 |
| Γ₀=0.01, r=10 (Exp. 1) | 7.02 | 6.18 | 5.32 | 4.92 |
| Γ₀=0.01, r=100 | 7.17 | 6.63 | 5.69 | 4.81 |
| Γ₀=1, r=1 | trap | trap | 4.58 | 4.20 |

- At N=128, changing Γ₀ from 0.01 to 0.1 lowers κ_eff by 0.5. That is comparable to the whole N=128 → 512 shift (ODE 0.47, SGD 0.78).
  Changing r from 1 to 100 raises it by 1.2.
- At d=64 (Exp. 2) the predicted T(16)/T(256) is 4.0, 4.4 or 9.1 for Γ₀ = 0.001, 0.01 or 0.1, and 4.4, 8.9 or 12.3 for
  r = 1, 10 or 100.
- So at accessible d, "the exponent" of the free protocol is a joint function of (N, Γ₀, η_Γ/η, m₀-window). Only the
  d → ∞ asymptote 2k* → 4k* is a function of N alone, and no SGD run reaches it. The local ODE κ at N=32 runs 7.35 → 5.63
  across the m₀-grid.
- The ODE tracks all of this, which is a strength. But the title's causal verb is not supported by an experiment that varies
  only N.

**Missing controls a careful reviewer will ask for** (ODE predictions above make them sharp; all are cheap):
1. **η_Γ/η and Γ₀ at d=64.** Free readout, N ∈ {16, 256}, r = 10 (ODE ratio 8.9; 1.83M / 0.21M steps) and Γ₀ = 0.1, r = 1
   (ratio 9.1). This is the single best extra run. It tests whether the ODE follows the readout hyperparameters, and it moves
   the system toward the 1/N regime. About 0.5 CPU-h.
2. **Tied with ρ₀ ∈ {0.1, 0.3} and pinned γ = 0.01, across N ∈ {16, 64, 256} at d=64.** ODE: ratio 1.9 at ρ₀ = 0.1; a trap at
   N=16 for ρ₀ = 0.3; pinned-0.01 flat (1.04). This decides whether "tied removes the N-dependence" is a property of tying or
   of a small readout. Tied runs take about 10 s each.
3. **k* = 3 in at least one regime.** Prop. 3 is claimed for k* ≥ 2, but every SGD run is k* = 2. At k* = 3, Model A already
   has negative drift below m ≈ 0.25 at γ = 0.1, N = 128 (MC, app. l.43), so even the "harmless" pinned-0.1 regime changes.
4. **A second d for the three-regime test.** d=128 is infeasible for the free readout at r = 1 (51M steps at N=16). At r = 10,
   d = 64 (control 1) does the same job. Tied and pinned-0.1 at d = 32 and 128 are cheap and test the N-flatness and the
   pinned N-ratio at a second d.
5. **η₀-invariance.** Pre-declared as the admissibility criterion ("Pre-declared analysis choices") and never run for
   Exp. 1 or 2. One extra η₀ ∈ {0.5, 2} for the free and tied N=16 cells of Exp. 2 suffices.

### 3.4 Statistics: estimator, censoring, precision and disclosure (l.178–186, l.199–201; Fig. 1; app. l.58–67)

1. **κ_eff is a secant, not an exponent.**
   - It is a 4-point OLS over a factor 2.8 in m₀, on runs that include a Γ-growth phase.
   - SGD local slopes inside one cell vary widely. Tied N=32 local κ is 6.3/3.8/3.0 between consecutive m₀ while the ODE's is
     4.1/4.1/4.3. Free N=32 ODE local κ is 7.35/6.44/5.63.
   - The OLS s.e. (0.10–0.30) ignores this misspecification, as the README notes. Fig. 1's error bars are therefore optimistic
     and the "exponent" label overstates.
   - **Fix:** show T(m₀) curves, SGD vs ODE, for all 12 cells, at least in the appendix. Call κ_eff a "secant exponent over
     m₀ ∈ [0.5, 1.4]·d^{−1/2}".
2. **Censoring.** Dropping censored runs (as in the main fit) and setting them to the cap (the triangles) are both ad hoc. A
   right-censored log-normal regression (Tobit), which is standard for censored escape times, gives:
   - free1 N=32: **6.05 ± 0.18** (ODE 6.47);
   - free10 N=32: 6.73 ± 0.18 (ODE 7.02);
   - free10 N=128: **6.67 ± 0.37** (ODE 6.18).

   **All 12 cells are then within 0.49 of the ODE.** In s.e. units the residuals are −2.3, 0.7, −1.6, 0.7 / −1.6, 1.3, −2.6,
   1.2 / −0.3, 0.6, 0.7, 1.6, so χ² ≈ 25 on 12 d.f. With optimistic s.e. that is acceptable, but it is the honest summary.
   **Fix:** report the Tobit κ, or the SGD/ODE ratio of absolute times, as the primary statistic, and declare it post hoc.
   This is a better result than "11 of 12 within 0.6, the censored cell within 1.0", and easier to read.
3. **Undisclosed design switches in the κ table** (still open from the first review):
   - B = 32 for N ≤ 128 and B = 8 for N ≥ 512, exactly where κ drops. The B control was done only at N=128.
   - Step caps are 10⁶ (N ≤ 128), 3·10⁵ (N=512) and 2–3·10⁵ (N=4096).
   - Each m₀ uses a different random stream, but a seed index shares streams across N and protocols (exp1b README dev. 2).
     The cells are therefore correlated (common random numbers), so the N-comparisons are paired while the s.e. treat them as
     independent.

   **Fix:** one sentence in the Exp. 1 paragraph and the B/cap columns in an appendix table.
4. **Medians of 3 seeds and "1–3%".** See R4 and R5. The 3-seed resolution is ≈ 5% (free) to ≈ 9% (tied, pinned-0.1 tok
   N=16, where one seed is +39%). **Fix:** report each median with its seed range or a 3-seed bootstrap interval, and replace
   "within 1–3%" with "within one s.e. (≈ 5%) in every free and pinned-0.1 cell".
   - The tied claim "flat within ×1.25" (l.201) is a goalpost moved from the pre-registered 5%. The appendix table says so;
     the main text should too.
   - Note that 9 of the 12 non-tied medians (7 of 10 distinct run sets) fall below the ODE. That is a small, plausible noise-acceleration bias, worth a
     clause.
5. **Escape fractions at 3 seeds** (Fig. 3 left; pinned-1 N=256: 2/3 vs 1/3). These resolve only "none / some / all". Say so
   in the caption, or pool the boundary cells.

### 3.5 Structure: a reader cannot reconstruct the protocols or the order of prediction and test from §§2–3 (l.82–203; app. l.45–67)

1. **The name "Experiment 2" is used twice.**
   - It is the three-regime test (l.195, repo `exp6`, "Exp. 6" in app. l.65) and the many-skill test (l.246, repo `exp2`).
   - l.241 says "Fig. 4 summarises Exps. 2–3", meaning the second pair.
   - Main "Experiment 1" is appendix "Exp. 1b". Appendix "Exp. 1" (uniform initialisation, d-slopes) is never mentioned in
     the main text except through P6.
   - Fig. 4(b)'s caption says "Exp. 3".
   - **Fix:** number the experiments E1–E5 in paper order, use those names everywhere, and add a one-line repo mapping in
     App. B.
2. **§2 does not define what the experiments vary.** Missing from §2 or the start of §3:
   - B (first used at l.67 and l.197);
   - η_Γ and how Γ is trained (plain SGD, lr η_Γ);
   - Γ₀ and ρ₀ (0.01; only in app. l.62 and l.66);
   - the tied update (plain SGD on u, no renormalisation, so the angular step is η/ρ);
   - the spherical update (projection plus renormalisation?);
   - T_0.5 (first defined in a figure caption, l.219);
   - the time unit "steps·η" (l.186) versus τ_esc (flow) versus steps (l.199).

   **Fix:** a 6-line "Protocols" paragraph in §2 listing B, η = η₀/d², the three readout updates with their initial values,
   T_0.5, and flow time t = η·steps.
3. **Which prediction came before which run is not visible.**
   - The ODE equations were fixed before Exp. 1. Their evaluation at Exp. 1's parameters was post hoc.
   - Prop. 3 in its three-regime form was written after Exp. 1 and the first review (commit `e2ddfb5`). It was committed one
     minute after the P19 pre-registration, after the `exp6` code commit and before the `exp6` runs.
   - Exp. 2 is therefore **the only prospective test of Prop. 3**, and it is the paper's strongest evidence.
   - The text presents Exp. 1 (post hoc, d=32) first and at greater length. The abstract mentions only Exp. 1 ("reproduces
     the exponents … at d=32", l.33) and **does not mention Exp. 2 at all**.

   **Fix:** a small "prediction ledger" table in §3. Columns: claim, fixed at (before or after which run), tested by, outcome.
   Rows: P1, P2/P7, P4/P4′, the κ-ODE (post hoc), P5, the d-scan, P19a–d. Lead the experimental part of §3 with Exp. 2 and put
   it in the abstract.
4. **Length.** The main text runs to p. 8 of the PDF, with Limitations on p. 8 and references from p. 9. Check the venue
   limit; ICML workshop limits are usually far shorter. §§4–5 are the obvious cuts (a short paragraph each plus the
   appendix), since neither varies N or the readout in a way that tests Prop. 3. §4 is at fixed Γ = 0.1 and N = 128.

---

## 4. Title and abstract fit

- **Title.** "Context length and readout parameterisation select the emergence exponent of in-context skills" promises a
  causal, general statement.
  - §3 delivers it only for one model, one link (σ₂), one d per experiment (32 for exponents, 64 for regimes), and with the
    caveat of §3.3.
  - §4 (many skills) and §5 (transformers) vary neither N-in-the-regime-of-Prop.-3 nor the readout. They do not support the
    title.
  - "Select the exponent" is also inaccurate for two of the three regimes. Pinned gives 2k* or a trap; tied gives 2k*
    (from small ρ₀). The exponent moves only for the free readout, and there it moves with Γ₀ and η_Γ/η too.
  - Suggested title: **"Finite context in in-context feature learning: the readout decides between a trap, a trade-off and
    no effect"**. Or, closer to the current one: "How the readout parameterisation decides whether pretraining context
    length matters for learning in-context features".
- **Abstract.**
  1. l.31–32, "restores a multiplicative trade-off between context length and training time". The SGD evidence is
     T ∝ N^{−0.54} at d=64; 1/N appears only in the ODE at d=512. Say "a trade-off, multiplicative (∝ 1/N) when the readout
     equilibrates".
  2. l.30–31, "more prompts cannot substitute for context length". Add "at fixed step size" (§3.2).
  3. l.32–33, "tied … removes the N-dependence altogether". Add "from a small initial norm" (§3.1, item 4).
  4. l.33–34. Replace the Exp. 1 count with Exp. 2: "Escape times of online SGD at d=64, predicted before the runs by a
     parameter-free population ODE, agree within the 3-seed resolution (≈5%) for pinned and free readouts at three context
     lengths; the ODE also reproduces the exponents measured at d=32."
  5. l.35, "additive compositions of learned skills cost nothing". This is an identity of a label-linear readout. Either cut
     it or say "cost nothing, by linearity of the readout in the labels".

---

## 5. Writing: concrete sentences, with rewrites

1. **Abstract l.28–33** (one 75-word sentence with a colon and two semicolons). *Rewrite:* "At finite N the loss gains a
   variance term Γ²V(m)/N that grows with alignment. What this term does depends on the readout Γ. Held at order one, it makes
   the unaligned state drift-stable below an overlap m*. Trained freely, Γ shrinks towards a Wiener factor and the effective
   exponent rises from 2k* towards 4k*. Tied to the feature norm and started small, Γ stays small and the N-dependence
   vanishes."
2. **Abstract l.33–34**, "(11 of 12 cells within 0.6, the censored cell within 1.0)". Parenthetical bookkeeping in an
   abstract. Use item 4 of §4, or "within the statistical resolution in all cells".
3. **l.62–64**, "The m^{2k*−1} gradient is inside the one-step analysis of Oko … we make the statement dynamical and verify it
   by online SGD, including the k*=2 case excluded from Ren." "Is inside" and "make the statement dynamical" are jargon, and
   "excluded from Ren" is misleading (Ren is in-weight). *Rewrite:* "The m^{2k*−1} gradient already appears in the one-step
   analysis of Oko et al. (Lemma 21); we show that it governs the full online-SGD dynamics, including k*=2, which in-weight
   analyses of the additive model (Ren et al.) do not cover."
4. **l.134–135**, "an aligned feature has heavy tails that coincide with the teacher's". Unclear. *Rewrite:* "When w aligns
   with v, large values of σ(⟨w,x⟩) occur together with large labels, so the context statistic A has a larger variance:
   V(m) grows with m."
5. **l.155–156**, "Γ relaxes at rate ≈2η_Γg(1)V(m)/N—fast relative to the overlap dynamics as soon as N<∞—to Γ*". False
   (§3.1) and mixes step and flow units. *Rewrite:* "Γ relaxes towards Γ*(m) on the flow-time scale N/(2r g(1)V(m)),
   r = η_Γ/η. If this is short compared with the escape time (adiabatic case), then …; otherwise the escape is limited by the
   growth of Γ from Γ₀."
6. **l.171–175** (a 70-word sentence with a nested "whether … is irrelevant, or is a hard requirement is decided by"). Its
   premise, "pins it at Γ*", is contradicted by P5. *Rewrite:* "What is specific to the in-context problem is that the optimal
   readout Γ* depends on N. Whether context length trades off against training time, does not matter, or is a hard
   requirement therefore depends on how the readout is parameterised."
7. **l.174**, "One-step analyses sit in case (ii) by construction". There is no case (ii). Name the item, and say why: a tiny
   readout makes Γ²V/N negligible, which is case (a) with γ → 0.
8. **l.178–180**. Setup, estimator, its derivation and a caveat in one sentence. *Rewrite* as two sentences: "Setup: …. We
   estimate κ_eff as 2 minus the slope of log T_0.5 against log m₀ (escape from m₀ takes ∝ m₀^{−(κ−2)}); with 4 values of m₀
   this is a coarse secant exponent."
9. **l.188–193**. About 90 words with three nested parentheses. Split into "Failed predictions: (i) …; (ii) …" as a short
   list, and give the tied ρ₀=1 result its own sentence.
10. **l.201–202**, "what trades off against context length is training time, never the number of prompts per step". Overclaim
    (§3.2). *Rewrite:* "At fixed step size the two batch schemes give the same step counts, as the population drift (which
    does not contain B) predicts."
11. **l.263–268, composition** (about 100 words: a dash, a semicolon, a nested citation and a second dash). *Rewrite:* "Trained
    on single-skill prompts only, the pair loss reaches 90% of its drop no later than the slower skill in every run. This is
    forced by linearity, so it checks the identity under SGD rather than testing anything. On thresholded accuracy, where
    Arora and Goyal predict an extra delay for tuples, the pair accuracy is approximately the product of the single-skill
    accuracies (fit …), the multiplicative form reported by Okawa et al. It therefore does reach any threshold later than
    either skill."
12. **l.301–302** (the transformer sentence mixing emergence times, norm values and logging). Split off the readout-norm
    observation into its own sentence.
13. **Inconsistent terms.**
    - "fixed" (l.95, l.217) vs "pinned" (l.30, l.152);
    - "escape time" as T_0.5 (steps) vs τ_esc (flow time) vs "steps·η";
    - κ vs κ_eff;
    - Γ vs γ vs ρ for the same readout scale.

    Pick one term per object and define it in §2.

---

## 6. Score

**6/10, weak accept for a HiLD-style workshop.**

For, compared with v0.2:
- Every first-review headline overclaim that was factually false has been corrected.
- The adverse evidence (P5, d-scan null, P4′ ρ₀=1, post-hoc ODE) is now in the main text.
- A genuinely pre-registered experiment (Exp. 2) was added. There the population ODE predicts SGD escape steps at d=64 across
  four protocols and three context lengths, to within what 3 seeds resolve.
- Every number I recomputed (§2) reproduces. A solvable model plus a parameter-free ODE that tracks SGD is squarely in scope.

Against:
- The formal centrepiece, Prop. 3, has no stated limit, and two of its sentences are contradicted by the paper's own
  ODE: "fast relaxation" and "ρ kept at Γ*". Its 1/N law is therefore not what Exp. 2 tests.
- The headline "prompts cannot substitute for context length" is a consequence of holding η fixed.
- The title's "context length selects the exponent" is confounded with Γ₀ and η_Γ/η at every accessible d.
- The experiment numbering and the missing protocol definitions make §§2–3 hard to reconstruct.

These are fixable in a revision without new theory and with under 2 CPU-h of runs. With the three edits below I would move to
7.

---

## 7. The three highest-value edits

1. **Rewrite Prop. 3 so it is checkable and true at the experimental parameters** (§3.1).
   - Give the limit and the time unit.
   - (a) σ₂-exact via ṁ = A m³ − B m.
   - (b) split into an adiabatic case (N ≪ r τ_esc ⇒ τ ∝ m₀^{−(4k*−2)}/N when d^{k*} ≫ N) and a readout-growth-limited case.
   - (c) "ρ stays ≈ ρ₀, N-independent while ρ₀ ≪ N m₀²".
   - Then say plainly that Exp. 2's 4.4 is an ODE prediction in the crossover (adiabatic value ≈ 13, (b)-as-written 16), and
     that the 1/N law is shown only by the ODE at d=512.
   - Fix "case (ii)" and l.172–173.
2. **Make Exp. 2 the lead result, and put the fixed-η caveat and the hyperparameter controls next to it** (§§3.2, 3.3, 3.5).
   - Put Exp. 2 in the abstract.
   - Rename the experiments E1–E5 and add the prediction-ledger table.
   - Replace "never the number of prompts per step" with the fixed-η statement, or run the η ∝ B control.
   - Run the two cheapest decisive controls, both with ODE predictions fixed in advance:
     - free readout at r = 10 (or Γ₀ = 0.1), N ∈ {16, 256}, d = 64: predicted ratio ≈ 9 vs 4.4;
     - tied ρ₀ ∈ {0.1, 0.3} and pinned γ = 0.01 across N: predicted N-ratio 1.9 / trap / 1.04.
3. **Replace the ad hoc κ statistics with censored-data fits and absolute-time comparisons, with intervals** (§3.4).
   - Use Tobit (or AFT) κ: all 12 cells within 0.49 of the ODE.
   - Add an appendix table of SGD/ODE escape-time ratios for every cell (0.64–1.34, about half within 10%).
   - Give seed ranges or bootstrap intervals for the Exp. 2 medians and ratios.
   - Disclose the B = 32/8 and step-cap switches.
   - Change "1–3%" to "within one s.e. (≈ 5%)".
   - Retitle the paper to match what §3 shows (§4).
