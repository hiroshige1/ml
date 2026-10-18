# Novelty scout, round 2 (search date 2026-10-18)

Scope: is claim (1) exponent doubling, (2) finite-N variance term + readout-parameterisation trichotomy (trap / Wiener shrinkage
with N–T trade-off / tied readout N-independence), or (3) many-skill emergence T_p ∝ 1/π_p already in the literature?
Papers the authors already discuss (Oko+24, Ren+25, Nishikawa+25, Gu–Xu–Zdeborová 26, Kim–Suzuki 24, Bietti, Berthier,
Arora–Goyal, Okawa) are not re-reviewed except where a new aspect showed up.

Method: ~25 queries (WebSearch, standard and extended; arXiv API full-text-field queries) on: in-context learning dynamics +
context length / plateau / emergence / induction head / batch size / number of prompts / task diversity / linear attention /
gradient flow / single-index / multi-index / information exponent / readout / value scale / spurious stationary point / Zipfian
task frequency. Abstracts were read for every paper listed; the relevant sections of Zucchet+25 and Bordelon+25 were read in full
text (arXiv HTML). arxiv.org is not reachable by WebFetch here, only by curl; openreview not tried.

## Overlap table

| # | Paper | arXiv / venue | Year | Claim | Overlap | Overlapping result (paraphrase / quote) |
|---|---|---|---|---|---|---|
| 1 | Bordelon, Letey, Pehlevan, *Theory of scaling laws for in-context regression: depth, width, context and time* | 2510.01098 | 2025 | (2) shrinkage piece | partially | Linear attention, linear tasks: under mean gradients the ICL matrix relaxes to the ridge-like fixed point "Γ⋆=(1+α⁻¹+σ²α⁻¹)⁻¹ I" (α = context/dimension), and SGD noise moves it further; they track batch size and data per context. This is the linear analogue of our Wiener value Γ*(m). There is no feature learning, so there is no trap and no effect on escape time. |
| 2 | Wu, Zou, Chen, Braverman, Gu, *How many pretraining tasks are needed for ICL of linear regression?* | 2310.08391, ICLR 2024 | 2023 | (2) shrinkage piece | related only | The pretrained linear-attention step size equals optimally tuned ridge at fixed context length. Finite-N shrinkage of the in-context readout is classical in linear ICL. |
| 3 | Zucchet, D'Angelo, Lampinen, Chan, *The emergence of sparse attention: impact of data distribution and benefits of repetition* | 2505.17863, NeurIPS 2025 (oral) | 2025 | (2) context length vs emergence time | partially, opposite mechanism | Toy gradient-flow model in which plateau length is a power law in sequence length T and dimension d. The escape time is "proportional to T√d/B" (B = in-context repetition), so a *longer* context slows emergence (attention dilution over distractors). No label-noise variance term, no trap, no readout dependence. |
| 4 | Musat, Pimentel, Noci, Stolfo, Sachan, Hofmann, *On the emergence of induction heads for in-context learning* | 2511.01033 | 2025 | (2) context length vs emergence time | related only | "The time until the emergence of an induction head follows a tight asymptotic bound that is quadratic in the input context length". Again longer context means slower, through dilution, not finite-N estimation noise. |
| 5 | Baherwani, Chen, Qiu, Wilson, Izmailov, *Emergent capabilities arise randomly from learning sparse attention patterns* | 2606.25010 | 2026 | (2) stochastic escape, context length | related only | Empirical: emergence is stochastic across seeds, and "the difficulty of learning attention patterns depends on context length and pattern sparsity". This matches the authors' transformer observation (escape probability within budget varies with B more than escape time does) but has no theory of a finite-N variance term. |
| 6 | Kim, Suzuki, *Transformers learn nonlinear features in context* | 2402.01258, ICML 2024 | 2024 | (2) trap | partially (already cited) | App. C.4: with finite prompt length, a spurious minimum may appear near the origin. A qualitative remark only. The authors already cite it as a precursor. |
| 7 | Oko, Song, Suzuki, Wu | 2411.02544, NeurIPS 2024 | 2024 | (1), (2) N–T trade-off | partially (already cited) | Lemma 21 contains the m^{2k*−1} gradient. Remark 3 states the prompt-length vs number-of-prompts trade-off at γ ∝ d^{-Q}. Both are acknowledged in the paper. |
| 8 | Xu, Sagitova, Zdeborová, Krzakala, *High-dimensional learning dynamics of attention-indexed models* | 2609.03858 | 2026 | (2) "parameterisation decides the dynamics, trap" | related only | In-weight attention-indexed models: "Direct optimization of an attention matrix S … can remain trapped in an uninformative state. Tied attention (S=WWᵀ) induces an automatic symmetry-breaking mechanism". Same moral (parameterisation decides whether online SGD is trapped) but a different object: the attention matrix, not the ICL readout, and with no context-length mechanism. |
| 9 | Tsiolis, Mousavi-Hosseini, Erdogdu, *From information to generative exponent: learning rate induces phase transitions in SGD* | 2510.21020 | 2025 | (2) readout timescale changes the exponent | related only | In-weight single-index: layer-wise two-timescale learning rates change the governing exponent. Same flavour as "readout protocol changes the exponent", but there is no ICL and no finite-N term. |
| 10 | Arnaboldi, Loureiro, Stephan, Krzakala, Zdeborová, *Asymptotics of SGD in sequence single-index models and single-layer attention networks* | 2506.02651 | 2025 | (2) sequence length vs SGD escape | related only | A closed-form population loss in sufficient statistics, with SGD escape and alignment phases. They show "how the sequence length … influence[s] convergence speed". There the target is a fixed function of the sequence, not an in-context task, so there is no task-coefficient variance term. |
| 11 | Gu, Xu, Zdeborová | 2610.01712 | 2026 | (2) variance term in loss | partially (already cited) | The replica analysis includes the finite training- and inference-context terms in the loss of trained models. It is static, not dynamical, and k*=1. Already cited at their eqs. (297)–(298). |
| 12 | Letey, Rysmakhanov, Lu, Pehlevan, Zavatone-Veth, *What can linear attention learn from nonlinear teachers in-context?* | 2610.10761 | 2026 | (1) | related only | Linear attention extracts only the linear Hermite component of a single-index link, and the rest acts as effective noise. This covers kernel-type learners. It says nothing about feature-learning exponents during pretraining. |
| 13 | Kim, Kwon, Choi, Park, Cho, Lee, *Task diversity shortens the ICL plateau* | 2410.05448, TMLR 2025 | 2024 | (3) | related only | Training on several ICL task families at once shortens each plateau. In our model, by contrast, skills decouple with T_p ∝ 1/π_p. This is worth one sentence: the two settings differ because their tasks share structure and our orthogonal skills do not. |
| 14 | Kwon, Xu, Yaras, Song, Balzano, Qu, *The effect of training task diversity on ICL through the lens of low-dimensional subspaces* | 2606.06814 | 2026 | (3) | related only | A linear-attention theory of why task diversity shortens the plateau. It has no emergence ordering by frequency. |
| 15 | Huang, Cheng, Liang, *In-context convergence of transformers* | 2310.05249, ICML 2024 | 2023 | (3) | related only | Softmax ICL with imbalanced features: dominant features converge in one phase and under-represented features later, in four phases. This is frequency-dependent ordering, but it is not a 1/π_p law. |
| 16 | Zhang, Singh, Latham, Saxe, *Training dynamics of ICL in linear attention* | 2501.16265, ICML 2025 | 2025 | (2) parameterisation decides the dynamics, (3) ordering | related only | Merged vs separate K/Q parameterisations give one abrupt drop vs saddle-to-saddle. Plateau time ∝ 1/eigenvalue², and finite-context corrections are given only for the fixed points, not the escape time. Another instance of "parameterisation decides the dynamics", in linear ICL. |
| 17 | Mainali, Teixeira, *Exact learning dynamics of ICL in linear transformers* | 2504.12916 | 2025 | (2) | related only | Closed-form SGD dynamics in which the converged loss is non-zero because of the finite context window. There is no trap and no N-dependent escape. |
| 18 | Varre, Yüce, Flammarion, *Sub-n-grams are near-stationary points* | 2508.12837 | 2025 | (2) trap | related only | Plateaus are explained by stationary points that become exact only as sequence length → ∞. The direction is the reverse of ours: their trap is an infinite-length artefact. |
| 19 | Kuwataka, Suzuki (TTT + ICL, 2509.25741); Oh, Huang, Suzuki (Mamba ICL, 2510.12026) | 2025 | 2025 | (1) | related only | Single-index ICL follow-ups to Oko+24. Their exponents concern test-time complexity (ie vs ge), not a doubled pretraining exponent. |
| 20 | Mehta, Gupta, *Scaling laws and ICL: a unified framework* | 2511.06232 | 2025 | (2) | related only | Phenomenological power laws in context length, with "sharp phase transitions at critical scales". There is no mechanism comparable to ours. |

No paper was found that contains any of the following:
- a finite-context variance term that *grows with alignment* and makes the origin drift-stable for an order-one readout;
- the statement that more prompts per step do not remove such a threshold;
- an exponent crossover 2k* → 4k* caused by Wiener shrinkage of a trainable readout;
- the tied-readout N-independence in ICL;
- an exponent-doubling statement for in-context single- or multi-index pretraining under online SGD, beyond what is implicit in Oko+24 Lemma 21;
- T_p ∝ 1/π_p (vs ≈ π_p^{-1/2}-type slopes in-weight) for an in-context additive dictionary.

## Verdict per claim

- **(1) Exponent doubling as N→∞.** Not new as a mathematical fact: it is inside Oko+24 Lemma 21, and "a fast readout doubles the exponent" is in Bietti+23 §5. The paper already concedes both. What remains is the dynamical verification under online SGD, including k*=2. That is a modest but legitimate addition, not a headline contribution.
- **(2) Finite-N term and readout trichotomy.** Novel in its main pieces, as far as this search can tell. Pieces are known separately:
  - the variance term appears in static loss formulas (Gu–Xu–Zdeborová; trivially in linear ICL);
  - finite-N ridge shrinkage of the readout is classical in linear ICL (Wu+23; Bordelon+25's Γ⋆);
  - the multiplicative N–T trade-off is Oko+24 Remark 3;
  - a spurious finite-N minimum near the origin is a one-line remark in Kim–Suzuki App. C.4;
  - "parameterisation decides whether SGD is trapped" has 2025–26 analogues for attention matrices (Xu+26) and K/Q factorisation (Zhang+25).

  Nobody connects these into one result: the same loss gives a trap at fixed Γ, shrinkage-driven slowdown with an exponent crossover at trainable Γ, and N-independence at tied Γ. The empirical context-length literature (Zucchet+25, Musat+25, Baherwani+26) finds *longer* contexts slowing emergence via attention dilution, which is the opposite sign and a different mechanism. That contrast should be cited and discussed, because a reviewer will raise it against the transformer section.
- **(3) T_p ∝ 1/π_p.** No direct precedent for the in-context additive dictionary. However, the paper itself calls it a corollary of (1)–(2), and the analogous in-weight laws (Ren+25, Nam+24) are known. Novel but incremental. Frequency-dependent ordering in ICL exists qualitatively (Huang+23), and task-diversity plateau shortening (Kim+24, Kwon+26) should be contrasted.

## Assessment for a workshop paper

The defensible novelty is claim (2): a clean, solvable mechanism by which the finite-context variance term Γ²V(m)/N, which grows with alignment, interacts with the readout parameterisation to give three qualitatively different pretraining dynamics. These are a trap that more prompts do not fix, a 1/N trade-off with an exponent crossover, and N-independence. The mechanism also explains when Oko+24's prompt-length/number-of-prompts interchangeability holds and when it fails. I did not find this result in the 2024–2026 literature. It is a natural question, so a concurrent paper could appear at any time, and the individual ingredients are each known. That is enough for a workshop paper (e.g., a NeurIPS/ICML theory-of-ICL or HiLD-type workshop), provided the paper:
- leads with (2) rather than (1) or (3);
- states plainly that (1) is a dynamical restatement of Oko+24 Lemma 21 and Bietti+23;
- adds the missing related work: Bordelon+25's ridge fixed point, Wu+23, Zucchet+25, Musat+25, Baherwani+26, Xu+26, Zhang+25, Kim+24 task diversity;
- keeps the transformer section explicitly observational.

For a main-track venue the result would likely be judged too toy-scale, because the trap does not transfer to transformers.
