# Fourth adversarial review: area chair, HiLD-type workshop (Draft v0.15)

Line numbers refer to `paper/body.tex` (B), `paper/appendix.tex` (A) and `paper/theorems.tex` (T). Review 3's points are not repeated.

## 1. Score and verdict

**6/10. Weak accept for a workshop, conditional on the abstract and Sec. 5 edits below.**

**What the paper contributes.** Most of it sits in a one-neuron solvable model.
- The finite-context term Γ²V(m)/N grows with alignment. The readout's parameterisation then decides the outcome: a pinned O(1) readout traps; a free readout slows escape by a factor set by (N, Γ₀, η_Γ/η); a tied or small readout removes the N-dependence.
- Theorems A(i)–(iii)(b) and B prove this for the population flow. I checked them, and they hold.
- A parameter-free ODE predicts SGD escape times at d=64 and d=128 to within 1–12%, including a predicted trap flip.

**What it does not contribute.**
- Anything about transformers.
- A free-readout 1/N law ever observed under SGD.
- A k*=3 exponent confirmed at the pre-registered precision.

The abstract still claims a bit of each.

## 2. Top 5 problems, ranked by severity

### P1. The transformer results are stated too strongly, and "feature norm untested" is a dodge

**Passages.**
- Abstract l.15: "these observations are consistent with the mechanism".
- B l.353: "the remaining untested analogue is the feature norm".

**Why this is a problem.**
- **"Consistent" is false in a checkable sense.**
  - The model's central SGD fact (E2, l.206–208) is that prompts per step B do not enter the drift.
  - In the transformer, N=16 at B=64 never emerges (0/12 runs across exps 5, 8 and 11). At B=1024 it emerges in 6–26k steps under every protocol.
  - So the transformer behaves like the Oko-style *default* (more prompts help), not like the mechanism.
  - The large-alignment reversal is "consistent" only because m₀ ≫ m* makes any outcome consistent.
- **Two pre-registered kills.** P21 (final readout) and P24 (attention output) both met their kill criteria. The abstract mentions only the first.
- **The feature-norm clause.** It was pre-registered as the kill-branch wording, so it is not post-hoc. In substance it is still a dodge:
  - In a pre-LN transformer, every block input and the readout input are LayerNorm-normalised. No free scalar plays the role of ρ, so the quantity is undefined there, not merely "untested".
  - The toy's tied regime predicts N-flatness, and the transformer is strongly N-dependent.
- **Overstatements in Sec. 5.**
  - l.346 "N=256 emerged under every protocol": one pinned-small seed is censored.
  - l.351–352 "slowed emergence at every context length (≈3× at N=256)": this holds only for O1. For Os, N=16 with B=1024 is one seed censored at 8k (the README calls it uninformative), and N=256 is one seed at 1.35× free.
  - l.337 "back at 1.7": this is one run. Seed 1 ends at 0.079 after emerging.
  - l.383 says "the readout-protocol test" in the singular.

**Fix.**
- Replace the abstract's last sentence with: "In small softmax transformers no parameter whose scale we froze reproduces the trap or the rescue (two pre-registered tests, both killed), and, unlike in the solvable model, prompts per step partly substitute for context length."
- Delete the feature-norm clause.
- Drop Fig. 5(b).

### P2. Prop. 3 attributes to theorems what the theorems do not prove

I checked the following steps, and they are correct:
- A(i): Descartes, and γ₁ = 2N/(N+9) recomputed from V = 1+8m²+6m⁴.
- A(ii): the partial fractions and F(u).
- A(iii)(b): the comparison z ≥ (z₀−K)E^a + KE, with K and a; the 1/ρ = 8s²t/N + c₁ + o(1) step.
- B: the Phase 1 integral NΓ₀²/4r; |e(t₂)| = o(d^{−k}); the |e'| ≤ δ invariant; LaSalle with Lemma 1.

The gaps are in the main-text summary:
- **(c), l.153–154: "independent of N provided ρ₀ ≪ ρ_trap (Theorem A(iii))".**
  - A(iii)(a) is a frozen-ρ identity, and A(iii)(b) proves only the trap.
  - Nothing bounds ρ(t) − ρ₀, so the N-flat third of the trichotomy is heuristic (A l.36–38).
- **(c), l.155: "ρ₀ > c₂ρ* … traps permanently (Theorem A(iii))".**
  - The theorem needs ρ₀ > ρ_trap(m₀), and that equals c₂ρ* only as m₀ → 0.
  - At the E5 cell, ρ_trap = 0.178, c₂ρ* = 0.154, and the numerical boundary is 0.1586 (T l.100).
  - So the stated claim is false for ρ₀ ∈ (0.154, 0.159) at the paper's own cell.
  - The "first falls, then escapes" half is proved only for the small-m system (T l.102).
- **(a), l.143: "otherwise τ_esc = Θ(m₀^{−2})".** This is false as m₀ ↓ m*, where τ diverges logarithmically (A(ii)).
- **Prop. joint is a sketch.**
  - The interchange of limits needed for κ is unproved (T l.171).
  - Hypothesis Γ₀ ∈ (0,2) excludes corner (c), where Γ₀ = 0.
  - Yet B l.189–191 states the "d→∞ values" as facts.

**Fix.** Prove the tied escape (ρ₀/E ≤ ρ ≤ ρ₀ plus Gronwall suffices for ρ₀ ≪ ρ_trap), or label it numerical. Write "ρ_trap(m₀) (→ c₂ρ*)". Mark the joint-limit numbers "(numerical)".

### P3. The free-readout headline claims a law SGD never showed

**Passages.**
- Abstract l.9–10: "raises the effective exponent from 2k* towards 4k* and restores a multiplicative trade-off".
- The Fig. 1 caption.

**Why this is a problem.**
- The 1/N law is Theorem B (fixed N, d → ∞). The paper's own Remark (T l.180–183) shows that SGD at η_Γ ∝ d^{−2} cannot track that flow.
- What was observed:
  - an N-ratio of 4.4, against 16 for 1/N;
  - a d-scan with no N-dependence (l.194);
  - a maximum κ_eff of 6.6.
- Prop. joint(c) puts the large-λ end at Γ₀ = 0.01 at 2k*+1 = 5, not 2k*. Yet E7 (l.239) calls 5.4 "the crossover value between the signal-dominated 4 and the adiabatic 8", although 5.4 is next to the lower end.
- Any value in (4, 8) would qualify as "crossover". The real evidence is ODE 5.40 vs SGD 5.37.

**Fix.** Use: "for the population flow a free readout has exponent 4k* at every fixed N with τ ∝ 1/N (Theorem B); at accessible d, SGD reproduces the flow's intermediate, (N, Γ₀, η_Γ/η)-dependent slowdown (N-ratios 4.4–10.7)."

### P4. E6 (k*=3) is reported more favourably than its pre-registration allows

**Passages.**
- Abstract l.12: "reproduces … the k*=3 exponent and threshold".
- B l.226–228: "medians within 15% … once the two declared post-hoc controls are applied".

**Why this is a problem.**
- P22a failed:
  - medians were 1.18 / 0.89 / 0.80 of the flow;
  - the 8→16 secant is 5.70 against 6.51 ± 0.5.
- The main text never says "P22a failed" and omits the secant miss.
- The rescue control P22e missed *its own* declared 10% band (1.14), and is then judged against the original 15%. That moves the goalposts.
- "The 2k* rule at k*=3" rests on two-point secants at m₀ = 0.35–0.18, d ≤ 32, with η fixed across d. Even the ODE gives 6.5.
- The ρ₀ ≪ ρ_trap qualifier was added after P22b failed.

**Fix.**
- Open the paragraph with "P22a and P22b failed at their pre-registered precision." Give 5.70, and give P22e as 1.14 against a declared 10%.
- Abstract: "the k*=3 trap threshold, and an exponent near 6".
- Flag the ρ_trap qualifier as post-hoc.

### P5. Numbers are inconsistent across main text, appendix and READMEs

I spot-checked 23 numbers. Fifteen match:
- E7: 0.989, 1.006, 1.22, 1.118/0.969/1.017, 1.195, 0.985, d* = 76 and 63.
- E6: m* = 0.30, 1.02, 47k/500k, ρ_trap = 0.0059.
- Transformer: 10.6–14.8k and 2.8–4.0k.

Eight do not:
- **(i)** B l.162 and A l.39 give Euler "771, 386"; LSODA gives 745/379. Yet A l.94 says "All ODE values in this appendix are LSODA".
- **(ii)** E2 l.204 and the Fig. 2 caption give Euler "1.84M/627k/414k"; LSODA gives ≈1.82M/623k/409k.
- **(iii)** T l.183 says "1.83M", while A l.97 says 1.81M.
- **(iv)** B l.289 says 0.73 (IQR 0.70–0.77) "equals" 0.64. It is outside the IQR.
- **(v)** B l.105 says "within 3% for m ≥ 0.2", but A l.54 gives k*=3 at m=0.2 as 0.88 ± 0.76.
- **(vi)** The exp10 README baseline is 101.0 (Euler); prereg gives 99.9.
- **(vii)** The P1 claims "back at 1.7" and "every protocol".
- **(viii)** l.155 cites c₂ to Theorem A(iii); it is in Remark `rem:ck`.

**Fix.** Regenerate every quoted ODE number from `ode_lsoda.csv` by script.

### Novelty

The comparisons below are fair and correctly scoped:
- Oko Lemma 21 / Remark 3 (credited);
- Gu–Xu–Zdeborová eq. 298 (static Γ* credited; the dynamics is the novelty);
- Ren Lemma B.1;
- Kim–Suzuki C.4 ("precursor").

Three corrections:
1. Contribution (1) is not a contribution. The closed form is a three-line Gaussian computation, and 2k* is in Oko. Make it the setup.
2. "Fails for a readout pinned at order one" (l.360) lies outside Oko's γ ∝ d^{−Q} regime, so it does not contradict Remark 3. Say so.
3. "The same scaling as our d*(N)" (l.364) is a k*=2 coincidence. Drop it.

## 3. A 4-page cut plan

| Section | Pages | Keep | Move to appendix |
|---|---|---|---|
| Abstract | 0.2 | §5's supported sentence plus the transformer null | — |
| 1. Intro | 0.5 | Question; 3 contributions (trichotomy + theorems; SGD match; transformer null) | Contribution (1) becomes setup |
| 2. Setting | 0.3 | Model A, protocols, T₀.₅, flow time | Hermite details |
| 3. Theory | 1.0 | Prop. 1; Theorem A(i) as Prop. 2; trimmed Prop. 3 with the P2 fixes | Kramers remark; d=512 numbers (l.161–163); l.164–170 to related work; Prop. joint |
| 4. Experiments | 1.2 | Fig. 2 as `figure*` (~0.4 pp) with d=128 points overlaid; E2+E7 merged; E5 in 3 sentences; E6 in 3 honest sentences; E1 in 2 sentences | Figs. 1 and 3; E1 detail; Tobit |
| 5. Transformers | 0.3 | One paragraph: the 3× observation, the reversal, two kills, B substitutes | Fig. 5 (drop panel b) |
| Many skills | 0.05 | One sentence (slope 1.44 vs 0.75–0.89) | Sec. 4 and Fig. 4 |
| Related work | 0.35 | Oko, Gu, Kim–Suzuki, Ren, Bietti, Nishikawa | Emergence and composition citations |
| Limitations | 0.1 | k* ∈ {2,3}; flow vs SGD; no transfer | — |
| **Total** | **4.0** | | |

## 4. Minor issues

1. `\todo{repo}` is still in l.386.
2. The E- and Exp-numbers interleave (E2=Exp.6 … E7=Exp.10), and the Fig. 4(b) caption says "Exp. 3" for E4. Add a mapping.
3. l.163 "τ=52,37,34,33 for N=32–2048" does not name the protocol or the N-grid.
4. Prop. joint's hypothesis Γ₀ ∈ (0,2) contradicts corner (c).
5. The Fig. 1 caption "tied stays at 4": the measured range is 4.10–4.44 (ODE 4.15–4.18).
6. Intro l.44 uses the unsimplified m*², and Prop. 2 uses 4γ/(2N−(N+5)γ). Use one form.
7. In Theorem A(ii), the O(·) constant blows up as γ → γ₁. Restrict to γ ≤ (1−δ)γ₁.
8. The abstract quotes OLS ("11 of 12 within 0.6") and A l.85 quotes Tobit ("all 12 within 0.49"). Pick one.

## 5. What a strong version would claim

**Supported:** "In a solvable in-context model, the alignment-dependent variance of the context statistic makes the readout's parameterisation decide whether short pretraining contexts trap the feature, slow it by a factor set jointly by context length and readout initialisation and learning rate, or do not matter; we prove this for the population flow, and online SGD matches the flow's escape times to within 1–12% at d = 64 and 128, including a predicted flip from escape to trap."

**Not supported:** "Pretraining context length selects the exponent with which transformers acquire in-context skills, and more prompts cannot substitute for it." The transformer data contradict the second half, and both knob tests were killed.
