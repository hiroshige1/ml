# Area-chair read (round 2)

## 1. Takeaway from the abstract and page 1

My one sentence after a fast read: *"In a one-neuron model of in-context learning with an attention readout, the readout's parameterisation decides whether context length during pretraining is a hard requirement (pinned, so a trap), a time trade-off (free, exponent toward 4k*), or irrelevant (tied)."*

Is it clear? **No.** I had to piece that sentence together. The abstract tries to cover everything: three regimes, Theorem B, the growth-limited caveat, a bound in symbols, Tobit fits, missed bands, d=128 flips, k*=3, composition, and two killed transformer tests. The three-regime result is the paper, but it is buried in one 90-word sentence. Is it interesting? Moderately, to ICL feature-learning theorists. It treats context length as a *training-time* resource, which is a fair new angle. It is not interesting to the broader science-of-DL audience, because the paper itself says the mechanism does not transfer to transformers.

## 2. Who cites this, and for what sentence

- ICL feature-learning theorists (follow-ups to Oko et al. 2024 and Gu et al. 2026): *"the N–T trade-off of one-step analyses depends on readout scaling; with an O(1) readout, finite context makes the unaligned state stable [X]."* That is a caveat citation, not a building-block citation.
- Skill/emergence/scaling-law people (Michaud, Nam, Ren et al.): *"in-context learning squares the link, so skills are ordered by frequency twice as steeply as in-weight (T_p ∝ 1/π_p vs π_p^{-1/2}) [X]."* I think this is the **most citable sentence in the paper**, and right now it gets three lines at the end of the 4-page version.

Expected citations: single digits as it stands.

## 3. Honesty: help or hurt?

The honesty itself helps. The way it is presented hurts. Pre-registration with kill criteria is rare and respected. But every result in the body comes with an inline post-mortem ("missed by 0.004", "the qualifier was added after E6", "withdrawn in favour of"). A reader cannot separate what holds from what failed, and the paper reads like a lab notebook. Roughly 40% of the 4 pages goes to caveats and to a negative transformer section.

Fix: (a) state the surviving claims cleanly in the body; (b) put **one table** in the appendix, "Pre-registered predictions: P1–P30, outcome (held / missed band / killed), what was revised", and point to it once from the intro ("k of n pre-registered predictions held; the failures are in Table X"); (c) in the workshop version, cut the transformer material to one paragraph: "the mechanism does not transfer to small softmax transformers; freezing the head or attention-output scale reproduces neither trap nor rescue (App. Y)." The negative result is useful, but it does not deserve a section.

## 4. Is the toy model acceptable?

For a workshop, yes. Single-index models are standard, and a parameter-free ODE that matches SGD to a few percent (E2/E7) is real work. For a main track the toy model has to earn its keep, and the only bridge to a real model is negative. What would make it worth it:
- **A phase diagram** in (readout scale γ or ρ₀) × (N/d), with the trap/escape boundary drawn analytically and SGD overlaid. The paper has the ingredients (d* ≈ N/(2γ), ρ_trap ≈ (4/3)ρ*) but no headline figure. Make this Figure 1.
- **The frequency-ordering law as a usable prediction.** The exponent ratio 2 (in-context vs in-weight emergence order) is something that can be checked in a real model, for example a linear-attention transformer or a small LM on synthetic tasks with known frequencies. A positive check there would be worth more than all the k*=3 precision bands.
- **An intermediate architecture**, for example linear attention with a trainable output scale, to show *where* transfer breaks.

## 5. Venue recommendation (as of Oct 2026)

| Venue | Fit | P(accept) |
|---|---|---|
| ICML 2027 workshops (deadline ~late May; e.g. a high-dimensional learning dynamics / HiLD-type workshop, or one on ICL theory) | Best fit: the audience is exactly single-index/ODE people | **65–75%** after a reframe; ~50% as is |
| ICLR 2027 workshops (deadline ~early Feb 2027; science-of-DL, ICL, or theory-of-foundation-models workshops) | Good, and the earliest option | **55–65%** |
| NeurIPS 2027 workshops (deadline ~late Aug/Sep; M3L / math-of-modern-ML type) | Good fallback | 55–65% |
| TMLR (rolling) | Fine for the full version; TMLR rewards correctness and candour over novelty | **50–60%** with the restructuring in §3, ~35% as is |
| ICML 2027 main (deadline ~late Jan) | Theorems are for a population flow of one neuron, and the transfer is negative | 10–15% |
| COLT 2027 (deadline ~early Feb) | Not a fit: no SGD-level sample-complexity theorem | <10% |

Workshop names change every year, so check the calls. **Plan:** a reframed 4-page version to an ICLR 2027 workshop, then the full version to TMLR. Aim for a main track only if the phase diagram and a real-model check of the frequency law work out.

## 6. Title and abstract I would accept

**Title:** *Readout Parameterisation Decides Whether Context Length Matters for Learning In-Context Features: A Solvable Model*

**Abstract (≈148 words):**
Does context length change how fast a model learns the features it uses in context? We study a solvable model in which a single-index skill with information exponent k* is learned by a feature on the sphere and a linear-in-label attention readout, trained by online SGD. We derive the population loss in closed form. At infinite context the link is effectively squared (exponent 2k*). Finite context N adds an alignment-dependent variance term, and its effect depends on the readout. A readout pinned at order one makes the unaligned state stable below a threshold, so long contexts are required. A free readout shrinks, raising the exponent towards 4k* and trading context length against training time. A readout tied to the feature norm makes escape N-independent below an explicit initial-norm threshold. Parameter-free ODE predictions match online SGD for k*=2,3 and d up to 128. With many skills, in-context learning orders skills by frequency twice as steeply as in-weight learning. These mechanisms do not reproduce in small softmax transformers.
