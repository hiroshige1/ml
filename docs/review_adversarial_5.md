# Fifth adversarial review: senior reviewer, ICML learning-dynamics workshop (Draft v0.22, focus: Sec. 5)

B = `paper/body.tex`, A = `paper/appendix.tex`. Secs. 1–4 are in good shape after review 4. Sec. 5 is not: it has grown from v0.16's clean null result into ~920 words (B l.325–348, one ~400-word paragraph) carrying eight pre-registrations (P21, P24–P30).

## 1. What Sec. 5 establishes, in my words

**Established:** in one small pre-LN softmax transformer trained with Adam, a short context (N=16) is a handicap that more prompts per step largely remove (stuck at B≤256, emerges at B≥1024). Freezing the scale of the final readout or the attention output reproduces neither the toy's trap nor its rescue.

**Open:** why B rescues N. The single-feature ablations show that B *can* matter through Adam, and that the population loss has no B in it (trivially true, Theorem A(iv)). They do not identify the transformer's mechanism.

**Does the text match?** It does until the last third of B l.348. There the ladder becomes a chain of exclusions ("what is excluded is … Adam's graded B-dependence alone"), stated more strongly than the evidence allows.

## 2. Strength of each claim

**(a) Exp 12/13. Overstated, and n is not the main problem.** The text claims "softmax and Adam together remove the trap … while neither does alone". The 2×2 behind this is confounded.
- The "Adam alone" cell uses the linear statistic. At N=16 its barrier is deep (m*=0.60 against m₀=0.125).
- The "softmax alone" cell uses the rbf statistic. Its pre-registered drift is "−5.7·10⁻³ ± 6·10⁻³ (N=16) … The signs at m₀ are 1σ".
- So there may be no population trap in the rbf model at N=16. The body quotes "drift −6·10⁻³" without the error bar.
- Supported reading: Adam crosses a barrier indistinguishable from zero but not the deep linear one. That is about barrier depth, not an interaction between softmax and Adam.
- Two seeds per cell (0/4 vs 3/4) would suffice if the cells were comparable.

**Exp 14 supports "consistent with noise-driven", not "this escape is noise-driven".**
- The seed ranges overlap heavily. B=16 spans 5.5k–35k, and B=1024 contains a 3.6k run, the second-fastest of all 12.
- The README's reading, fixed before looking at the data, says "P27 FAILED on the declared (strict) reading". The appendix row says only "held".

**(b) Exp 15. The effect is real; "prompts per step enter through the optimiser" is fair but incomplete.**
- The contrast is large: 17.4× for Adam against ≤1.20× for SGD. The P28a miss of 0.004 is honestly reported.
- But the body says "the SGD flow time is B-invariant" and never states that **P28b failed** (spread 1.20 against 15%; A l.158).
- The effect is measured in steps. In prompts consumed, B=1024 uses 3.7× *more* data than B=16.
- The toy's own local slopes (−0.42, −1.36, −0.28) are not a power law, so "graded law B^−0.75" is a fit the toy itself does not obey. P30 then extrapolated that fit to the transformer.
- Adam's SNR-driven batch-size dependence is known (square-root scaling rules for adaptive methods); cite it.

**(c) Exps 16–17, "threshold-like". Not fair as worded.**
- P29 pre-registered its *own* threshold default: "B=256 and B=512 both stuck 2/2". That default failed, because B=512 emerged in 1 of 2 seeds.
- P30 rejects only slope −0.75 anchored at B=1024. Any slope ≤ −1.33 fits B=256 > 80k.
- With n=2, one B=2048 seed, and a B=512 seed censored at 30k, "threshold" and "steep graded" cannot be told apart.
- B l.348 says "B=256 stuck through 80k steps in both seeds". Seed 1 ran only to 77k.
- **Fair wording:** "much steeper than the toy's Adam scaling: B=256 stays stuck through 77–80k steps (n=2) where that scaling predicts ≈36k."

**(d) Abstract.** The clause "unlike in the solvable model, prompts per step partly substitute … threshold-like … neither the loss nor Adam's graded batch-size effect in the model accounts for" has three problems:
- "Unlike in the solvable model" contradicts Exp 15, where the solvable model under Adam shows a 17× B effect.
- "Threshold-like" fails for the reasons in (c).
- It makes a negative explanatory claim from an extrapolated slope.

**Proposed wording:** "…and prompts per step partly substitute for context length, which the population loss cannot produce; the mechanism is open."

## 3. Cut to ≤0.6 page

**What stays (~330 words + Fig. 5(a), shortened):**
1. **The observation (~130 words).**
   - Large alignment: the ordering reverses.
   - Small alignment: N=16, B=64 is stuck 3/3. N=16, B=1024 emerges at 10.6–14.8k, against 2.8–4.0k for N=256 in *two* paired seeds.
   - The third seed reversed this (11.6k against >17k for N=256; exp 5 README). The body omits it.
2. **Two null knob tests, P21 and P24 (~80 words).**
3. **What B does (~120 words).**
   - The population drift has no B in it.
   - Under Adam, B matters in the toy (17×; P28 both clauses narrowly missed).
   - The transformer's dependence is steeper than the toy's (n=2).
   - The mechanism is open.

**What moves to an appendix "Ablation ladder" table (one row per experiment: cells, n, outcome):**
- The readout-norm collapse and Fig. 5(b) (identical in stuck and emerging runs, so not evidence).
- Exp 12's product-softmax arm.
- Exps 13–14.
- The "generic speed knob" remark, the LayerNorm/ρ sentence, and the exclusion list.

**What is deleted:** the claim "softmax and Adam together … neither alone".

## 4. Contradictions and number checks

**Matched (17 of 20):** Exp 8 5.8k/8.4k; Exp 11 15.0k/26.4k, 3.4×/2.6×, 1.35×; Exp 12 379/1421, 0/4; Exp 13 0/4 at 5·10⁵, 3/4, 26.3k/24.7k, 63.9k/65.8k; Exp 14 8.0/25/39/60k; Exp 15 2537→146, −0.754±0.144, −0.015±0.028, 17.4; Exps 16–17 35,921≈36k, B=512 1/2, slope −0.51; Exp 5 10.6–14.8k, readout norm 0.57→0.03–0.16.

**Contradictions:**
1. **"Rescues N=16 under every protocol"** (B l.336, l.348). Under Os (Exp 11) the B=1024 cell has one seed censored at 8k and the other not run (A l.122). There is no evidence either way.
2. **"Through 80k in both seeds"** against 80k/77k (A l.159).
3. **"SGD flow time is B-invariant"** against the P28b row "failed on spread".
4. **P27 "held"** (A l.157) against the README's "FAILED on the declared (strict) reading". The pre-registration discloses this; the paper does not.
5. **"Threshold"** (abstract, B l.348, F8) against P29's failed threshold default and the exp 16 outcome text.
6. **Exps 12–17 are missing from two sections.** Intro (3) (B l.54–56) and the Limitations (l.376) mention only the two knob tests, while the abstract leads with the B-threshold.
7. **"≈3× later"** is stated without the seed-2 reversal.
8. **"Unlike in the solvable model"** (abstract) against Exp 15.
9. **The section title** promises an effect that never shows.

## 5. Is anything publishable on its own?

**As written, no.** This is boundary reporting, honestly scored, where each rung answers a question the previous rung created. The exclusions are weak:
- "Not the loss" follows from the minibatch gradient being unbiased.
- "Not the readout" and "not the attention scale" are the two knob kills.
- "Not Adam" rests on a mismatch with an extrapolated fit.

**One thread is interesting, and the paper closes it prematurely.**
- **The argument.** In the noise-dominated regime, Adam's step *norm* is roughly B-independent (≈ lr·√d·const). Its component along the signal, however, grows with SNR ∝ √B.
- **What it predicts.** On the sphere, or for scale-invariant pre-LN weights, renormalisation then exerts a B-*independent* drag (≈ m|u|²/2) against a B-*dependent* signal. That gives **a B-threshold wherever the drift is weak**, and graded behaviour where it is strong.
- **Where the two experiments sit.** Exp 15 is strong-drift: drag ~10⁻⁷ per step against ~10⁻⁴ per step of progress. The transformer, on a plateau at |cos|₀ = 0.13–0.18, plausibly is weak-drift. Exp 15's sigmoidal local slopes hint at the transition.
- **Why it matters.** The toy and the transformer may be the *same* optimiser effect in different regimes, the opposite of B l.348's conclusion. One pre-registered toy B-scan at weak drift (larger d, or m₀ near m*) would settle it.
- **Until then:** "the strong-drift Adam law does not match; the weak-drift regime is untested".

## 6. Score, verdict, top 3 fixes

**6/10: weak accept, carried by Secs. 3–4.** On its own, Sec. 5 is ~4/10: its recent growth added claims, not evidence.

1. **Cut Sec. 5 to ≤0.6 page as in §3.** Move Exps 12–14, Fig. 5(b) and the exclusion list into an appendix ladder table. Retitle the section, e.g. "Transformers: one observation, two null tests". Align Intro (3) and the Limitations with what remains.
2. **Restate strengths.**
   - Drop "neither alone", or report the rbf drift with its ±1σ.
   - Say "consistent with noise-driven".
   - Report P28b's failure in the body.
   - Replace "threshold-like" with "steeper than the toy's Adam scaling (n=2)" and note that P29's threshold default failed.
   - Fix "every protocol", "both seeds through 80k", and the seed-2 reversal behind "≈3×".
   - Replace the abstract clause as in §2(d).
3. **Withdraw the Adam exclusion, or test it.** The weak-drift drag argument predicts a threshold. Either run one pre-registered toy B-scan or name it as the leading open candidate.
