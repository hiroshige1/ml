# Full-text novelty check: claim C4 (composition in a label-linear in-context model)

Checked 2026-10-08. PDFs downloaded from arxiv.org with curl into
`/tmp/claude-0/-home-user-ml/5f7f99a8-9492-5376-82b9-185ad2dc461e/scratchpad/papers_comp/`
and text extracted with pypdf. Page numbers below are PDF page indices, which match the printed
page numbers for (1)-(4) main text.

Coverage: (1) and (2) read in full, including appendices. (3) and (4) main text read
((3): pp. 1-9, (4): pp. 1-12).

**C4 restated.** Take an in-context predictor that is linear in the context labels,
ŷ_q = Σ_j Γ_j σ(w_j·x_q)(1/N)Σ_i y_i σ(w_j·x_i), with additive targets y = Σ_p c_p σ(⟨v_p,x⟩),
pretrained on single-skill prompts only. Then:
(a) the two-skill-prompt loss is exactly the sum of the two single-skill losses;
(b) so additive composition emerges for free at max(T_p, T_q), with no compositional data;
(c) a thresholded accuracy is the product of the per-skill accuracies, i.e. "multiplicative
emergence" is derived from additive structure plus a thresholded metric;
(d) non-additive compositions (products or interactions) are not free, which gives a sharp boundary.

---

## (1) arXiv 2609.33708, Rydel, "A spectral theory of compositional learning" (ICLR 2027 submission, 22 pp.)

- **Model class.** These are relational embedding models (TransE-style), not in-context models.
  Entities are points x_a and relations are displacements r, with fact error
  "ϵa,r,b = xa + r − xb. (1)" (p3). The paper studies a shallow lookup-table model and deep
  linear factorisations E = W_N…W_1 trained by SGD on the loss "L(E) = 1/(2|D|) ‖AE − C‖²_F (2)" (p4).
- **What "additive" means here.** It is a relational law between displacements, not an additive
  target over skills: "one z step must cover the same displacement as the two-step path:
  rz = rx + ry. This compositional law holds in the world by construction but is not built into
  the model, which must recover it from the facts" (p3).
- **Derived.**
  - Prop 4.1, identifiability, is the classical estimability criterion
    "ℓj = Q+Q⊤+ℓj. (4)" (p5).
  - Prop 4.2, law-error trajectory: "δj(t) = ℓ⊤j E(t) = Σ_{k:λk>0} ajk ζk(0)e^{−λkt/τ} . (8)" (p5).
  - Deep-network dynamics are integrated numerically (Eq. 10, p6).
- **Observed.**
  - Emergence time t* = "the ﬁrst evaluation at which held-out compositional accuracy reached
    100% and geometric error fell below 0.5" (p6).
  - Predicted vs observed log t* gives R² = 1.0000, 0.9979, 0.9793 (p7).
  - A single linking fact unlocks inferences (pp7-8).
- **Emergence-time law.** No closed form beyond Eq. (8). Timing is set by the spectral modes
  that support the law. There is no max(T_p, T_q) rule and no product rule.
- **"Composition is free for additive structure"?** No. The training data contain z-facts
  (composite facts): "Some valid z-facts were included in training and distinct valid z-facts
  were withheld" (p15). Composition has to be identified from evidence, and it can fail even when
  the premises are known (Exp. 2). This is the opposite of "free from single-skill data".
- **"Multiplicative accuracy from thresholding"?** No. Accuracy is nearest-neighbour retrieval (p4).
- **In-context learning?** No.
- **Non-additive boundary?** Only as a limitation: "left open whether the same principles apply
  to heterogeneous or non-additive relations" (p9).
- **Verdict: NOT CONTAINED.** The overlap is limited to vocabulary ("additive", "spectral",
  "emergence time") and the general idea of deriving composition timing from learning dynamics
  in linear models.

## (2) arXiv 2310.09336, Okawa, Lubana, Dick, Tanaka, "Compositional abilities emerge multiplicatively" (NeurIPS 2023, 23 pp.)

- **Model class.** A conditional diffusion model (U-Net with attention; Appendix A.3, p18)
  trained on synthetic shapes with tuple conditioning (shape, color, size), plus CelebA.
  It is not in-context and there is no theory of the network.
- **What is meant by "multiplicative".**
  - The headline effect is empirical, and the metric is a product by definition:
    "We define a model's accuracy for generating images of a given concept class as the product
    of the probabilities outputted by the three probes" (p6).
  - Eq. (2), p19: "Accuracy = 1/Nt Σ_n 1(f0(x0),v0) · 1(f1(x0),v1) · 1(f2(x0),v2)".
  - "Multiplicative influence of capabilities drives the sudden emergence of compositional
    generalization (Fig. 6)" (p7).
  - The paper contrasts an "additive measure (dashed line)", meaning the average of per-concept
    accuracies, with a "multiplicative measure" (p7). This "additive" is about how the metric is
    aggregated. It is not about the target or task structure in C4's sense.
- **Hypothesis (p7).** "compositional generalization is hindered by the multiplicative, not
  additive, impact of learning progress on each concept". Also: "compositional tasks require the
  concurrent acquisition of all involved 'atomic' capabilities, akin to an AND logical condition" (p7).
- **The only "derivation" is a toy Bernoulli model (pp7-8).**
  - Probability that one atomic ability is learned: "1−(1−p)^t".
  - Probability for an n-composition: "P(n) = (1−(1−p)^t)^n".
  - Critical time: "t∗ = ⌈ log(1−(P∗)^{1/n}) / log(1−p) ⌉".
  - Its assumption is stated outright: "the learning dynamics of atomic abilities are independent
    (hence the name atomic) and that once an atomic ability is learned, it will not be forgotten" (p8).
  - There is no network or loss model behind this.
- **Training data.** The training data are a minimal set of concept classes
  ({000, 100, 010, 001}; p6) that are 1 concept away from each other. Tests are at concept
  distance 1–2 (111). This is analogous to "single-skill" training with compositional test data,
  but it is observed, not derived.
  - Caveat: the p6 text lists 001 as "(circle, blue, small)", which would be distance 2 from 000.
    The encoding in the text looks inconsistent.
- **Emergence order.** Observed, not derived: "first learns the concept classes in the training
  dataset ... then generalizes to concept classes with a concept distance of 1 ... Thereafter, the
  model suddenly acquires the capability ... distance of 2" (p6). The bottleneck is the slowest
  concept: "the model fails to acquire strong color transformation capabilities until the final
  stage of training, effectively bottlenecking compositional generalization of 111 class" (p7).
  This is qualitatively a max-of-constituent-times picture, but it is observed and never stated
  as max(T_p, T_q).
- **"Composition is free for additive structure"?** No. There is no additive target, no loss
  decomposition and no "free" statement. Compositional generalization is shown to be delayed and
  sometimes to fail (Fig. 9, adversarial training set; fine-tuning does not fix it, p9).
- **"Multiplicative accuracy from thresholding"?** Partly, but in the reverse logical direction.
  - The product form is imposed as the metric definition (Eq. 2), and the toy model simply
    assumes independent atomic events with AND.
  - The paper does not derive that a thresholded accuracy on a composed output factorises into
    per-skill accuracies because of additive structure in the model or target.
  - It also does not prove factorisation; independence is assumed.
- **In-context learning?** No.
- **Non-additive boundary?** No.
- **Verdict: PARTIALLY CONTAINED.**
  - The phenomenon and its explanatory slogan (multiplicative emergence of compositional
    accuracy, bottlenecked by the slowest constituent, an AND/product of per-concept success,
    and t* growing with the number of composed atoms) are already in Okawa et al.
  - C4's novelty has to rest on three things:
    1. a model-level derivation (exact loss additivity in a label-linear ICL model, so the
       compositional loss is a sum of single-skill losses);
    2. obtaining the product of accuracies from additive structure plus a threshold, rather than
       by definition or an independence assumption;
    3. the additive vs non-additive boundary.
  - C4 must not claim to have discovered "multiplicative emergence". It should cite Okawa as the
    empirical origin and say it supplies a mechanism or derivation for it.

## (3) arXiv 2604.22951, Wang, Dang, Lee, Lyu, "The power of power law: asymmetry enables compositional reasoning" (ICML 2026, main text pp. 1-9)

- **Model class.**
  - Theory: a scalar "k-multiplicative composition" model,
    "y := Π_{i=1}^k (x⊤_i w∗)" with learner "fw(X) = Π_{i=1}^k (x⊤_i w)" (pp3-4), trained by
    online/minibatch SGD on squared loss.
  - Experiments: transformers trained from scratch on S5 state tracking, multi-step arithmetic,
    multi-hop QA and iGSM.
  - The training data are the composed k-hop tasks themselves. Multi-hop QA training is "a
    mixture of the profile fact data (1-hop) and the query questions (k-hop)" (p8).
- **Derived.**
  - Thm 1 (CSQ lower bound, uniform skills): "requires n ≳ d^{k/2} samples when q ≲ d^{k/2}" (p4).
  - Thm 2 (power law p_j ∝ j^{−α}, α>1): learns "with Õ(d^{2α}/(ηε)) samples in
    t ≤ Õ(d^α/η log 1/ε) time" (p4).
  - Stage-wise dynamics from "wj(t+1)−wj(t) = ηkpj(A(t)^{k−1}w∗j − B(t)^{k−1}wj(t))" (pp5-6).
- **Observed.** Power-law skill sampling beats uniform sampling on all of the tasks above
  (Figs 1, 3-5).
- **Emergence-time or accuracy law for compositions.** Only sample and time complexity bounds.
  There is no formula combining per-skill emergence times and no product-of-accuracies law.
  Exact-match accuracy is measured but not decomposed.
- **"Composition is free for additive structure"?** No. The composition operation is
  multiplication. The paper is about why multiplicative composition is hard (d^{Ω(k)}) under
  uniform sampling.
- **"Multiplicative accuracy from thresholding"?** No.
- **In-context learning?** No.
- **Relevance to C4(d).** The paper is consistent with, and could be cited as support for, the
  claim that product or interaction compositions are not free. Product-of-skills targets are
  provably hard to learn even with compositional training data under symmetric distributions.
  It does not state a contrast with additive compositions being free.
- **Verdict: NOT CONTAINED.** This is supporting related work for the non-additive side of the
  boundary.

## (4) arXiv 2307.15936, Arora & Goyal, "A theory for emergence of complex skills in language models" (main text pp. 1-12)

- **Model class.** There is no network model. It is a statistical framework that assumes scaling
  laws, a random bipartite skill graph (each text piece needs k i.i.d. skills, Def. 5, p7), and
  cloze-question competence. Training dynamics are explicitly sidestepped: "the theory will
  assume scaling laws such as (2) and thus not need to reason about training and generalization" (p5).
- **Training data.** These are not single-skill. Every text piece already mixes k skills:
  "normal language — down to short paragraph level— already utilizes multiple skills, mixed up
  randomly" (p12). The novelty argument is that most specific k′-tuples are unseen (poverty of
  stimulus, pp2, 12).
- **Derived.**
  - Thm 8 / Thm 14, random-graph bounds (pp8-11): "H(θ) + kθ(H(βα) − βα log 1/α − (1−βα) log(1/(1−α))) < 0".
  - Corollary 13 (p10): "When the model M1 with loss δ is scaled up ... so that the new model M2
    has loss δ/k′, then the performance curve inferred by our method for k′-tuples of skills using
    M2 is identical to the curve inferred for individual skills on model M1."
  - Takeaway 1 (p11): "reducing θ by a factor 2 (... roughly one order of scaling up ...) has the
    effect of raising competence on 2k′-tuples to at least the same level as what it was on
    k′-tuples before scaling."
- **Emergence law for compositions.** Tuple competence needs a k′-fold smaller loss. It is
  "essentially at very similar scaling" (abstract p1) but delayed: about 10x scale per doubling
  of tuple size. This is not max(T_p, T_q), which would mean no extra delay.
- **"Free composition"?** Qualitatively similar in spirit: "proficiency in combinations of skills
  arises just as naturally as proficiency in the individual skills themselves, and need not
  require seeing examples of all (or even most) of these combinations in the training set" (p12).
  But it is not tied to additive structure and is not exact.
- **Multiplicative or thresholded accuracy?**
  - There is an AND-style "conservative accounting": "if the model incorrectly answered this cloze
    question ... then that loss was incurred in all these statistical tasks" and "error now means
    an erroneous answer on any cloze question in the entire k′-piece" (pp8, 10).
  - A text piece counts as a mistake if its loss is "at least 1/2" (p8), which is a threshold.
  - No product-of-accuracies formula is given.
  - Notably, the paper rejects linearity-of-expectation decompositions in general:
    "k·Et[loss(t)] = Es[failure rate of statistical task τs]. (Incorrect!) (7)" (p8).
    C4's exact loss additivity is a model-specific property that this framework explicitly does
    not assume.
- **In-context learning?** Mentioned only as motivation (pp1-2, "flexibly combining its various
  capabilities during in-context learning"). It is not modelled.
- **Non-additive boundary?** No.
- **Verdict: PARTIALLY CONTAINED, qualitative message only.** "Skill combinations emerge without
  seeing those combinations" is A&G's headline, at a scaling-law level. C4's mechanism (an exact
  loss-sum identity in a label-linear ICL model, single-skill-only pretraining, max(T_p, T_q)
  timing, product accuracy from thresholding, and the additive vs interaction boundary) is absent.

---

## Overall assessment for C4

| Paper | Verdict | What overlaps | What is absent |
|---|---|---|---|
| (1) Rydel 2026 | NOT CONTAINED | Additive relational laws in linear nets; spectral emergence times | ICL, single-skill-only training, loss additivity, product accuracy, boundary |
| (2) Okawa+ 2023 | PARTIALLY | Multiplicative emergence; product metric (by definition); toy t* for n atoms; slowest-concept bottleneck (observed) | Any model-level derivation; additive-target structure; ICL; "free" composition; non-additive boundary |
| (3) Wang+ 2026 | NOT CONTAINED | Product-of-skills composition is hard, which supports C4(d) | ICL, additive free composition, accuracy laws |
| (4) Arora & Goyal 2023 | PARTIALLY (qualitative) | Combinations emerge without seeing them; AND-style error accounting | Exact mechanism, additive structure, max(T_p, T_q) (A&G predict extra delay), ICL, boundary |

**Required framing.**
- Cite Okawa et al. as the origin of "multiplicative emergence" and present C4(c) as a
  derivation of it. Okawa defines the product metric and, in the toy model, assumes independent
  Bernoulli atoms.
- Cite Arora & Goyal for the "combinations for free" idea, and contrast it: A&G need a k′-fold
  smaller loss for k′-tuples, while C4 gives no extra delay for additive composition.
- Cite Wang et al. as consistent with the non-additive side of the boundary.

**Remaining risk.** None of these four papers treat in-context learning, so the ICL-specific
novelty must still be checked against ICL-composition theory papers not covered here.
