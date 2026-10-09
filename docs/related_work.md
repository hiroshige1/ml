# Related work and novelty check

Status legend: **[read]** full text read; **[snippets]** only abstract/search snippets seen (arXiv is
blocked from this environment; add `arxiv.org`, `export.arxiv.org`, `openreview.net`,
`proceedings.neurips.cc` to the environment's allowed domains to upgrade these).

## Directly overlapping

| paper | what it does | overlap with us | status |
|---|---|---|---|
| Ren, Nichani, Wu, Lee 2025, *Emergence and scaling laws in SGD learning of shallow NNs* (arXiv 2504.19983, NeurIPS 2025) | Additive target `Σ_p a_p σ(⟨v_p,x⟩)`, power-law `a_p ≍ p^{−β}`, online SGD on 2-layer student; **even activation with `k* > 2`** (abstract); extensive width `P ≫ 1`; per-neuron abrupt emergence sums to smooth scaling law; greedy max-selection decoupling. Note: our most interesting regime is `k* = 2` (even, *outside* their assumption), where in-weight is easy (`d log d`) but in-context is not. | Our Model B is their setting. We change the *learning problem* to in-context (Model A). | [snippets] |
| Oko, Song, Suzuki, Wu 2024, *Pretrained transformer efficiently learns low-dimensional target functions in-context* (arXiv 2411.02544, NeurIPS 2024) | Single-index tasks with `β` in an `r`-dim subspace; MLP + linear attention; pretraining complexity `d^{Θ(Q)}` (`Q` = info exponent), ICL prompt length `r^{Θ(P)}` | Same architecture as our Model A. **Must check whether their pretraining analysis already contains the `g ↦ g²` (exponent `2Q`) effect** (the `Θ(Q)` hides it). Their task prior is a continuous subspace, ours is a discrete power-law dictionary; they do one/few GD steps, we do online SGD dynamics and emergence times. | [snippets] — **blocking for novelty claim §2** |
| Nishikawa, Song, Oko, Wu, Suzuki 2025, *Nonlinear transformers can perform inference-time feature learning* (ICML 2025) | Softmax attention + ReLU MLP learns `β` from the test prompt alone; beats CSQ lower bound via nonlinear label transformations induced by softmax | Relevant to §4 (architectures that can use `y²`-type statistics) and to the kernel-vs-feature distinction. Does not study emergence timescales or skill dictionaries. | [snippets] |
| Gu, Xu, Zdeborová 2026, *In-context learning of single-index targets: comparing kernel and feature learners* (arXiv 2610.01712, Oct 2026) | Kernel learner needs context `d^k` per Hermite component (stage-wise); feature learner single transition at `L = Θ(d)` | Context-length thresholds for in-context *estimation*. Our §2.2 threshold is for *feature emergence during pretraining* — different object, must be contrasted explicitly. | [snippets] |
| Nam, Fonseca, Lee, Mingard, Louis 2024, *An exactly solvable model for emergence and scaling laws* (NeurIPS 2024) | Skills as basis functions, multilinear model; sigmoidal emergence + scaling laws in time/data/size | Phenomenology we also predict (§3); no ICL. | [snippets] |
| Michaud et al. 2023 (quanta), Arora & Goyal 2023 (skill composition, slingshot), Okawa et al. 2023 (multiplicative emergence of compositional abilities) | Empirical/heuristic skill–scaling pictures | §4 derives a multiplicative law from the additive structure; cite as the empirical target. | [snippets] |
| Kunin et al. 2025, *Alternating gradient flows* (NeurIPS 2025) | Saddle-to-saddle feature learning from small init in 2-layer nets, staircase loss | Relevant to the two-timescale readout/feature dynamics (§2.2 TODO). | [snippets] |
| Kim & Suzuki 2024, *Transformers learn nonlinear features in context* (ICML 2024) | Two-timescale: attention factored out, mean-field MLP | Same two-timescale trick; different question (landscape, not emergence times). | [snippets] |
| Lu et al. 2024, *Asymptotic theory of ICL by linear attention* | Linear attention, memorization/generalisation transition with task diversity | Task-diversity thresholds; could matter for our finite-`P` dictionary. | [snippets] |
| *The power of power law: asymmetry enables compositional reasoning* (arXiv 2604.22951) | Power-law skill frequency helps compositional learning (empirical + minimal theory) | Related to §4; different mechanism (curriculum). | [snippets] |
| Mehta & Gupta 2025 (arXiv 2511.06232) | Unified ICL scaling framework, power laws in depth/width/context | Phenomenological; check whether its "critical scales" overlap §2.2. | [snippets] |

| Oko et al. 2024 — conclusion | They state their pretraining complexity rests on a one-gradient-step analysis and **conjecture it can be improved by modifying the pretraining procedure or objective** | Our C3/C5 (shrinkage, trichotomy) is a mechanistic answer to that conjecture *if* their exponent is the squared one. | [snippets] |
| *From information to generative exponent: learning rate induces phase transitions in SGD* (arXiv 2510.21020) | Large learning rates / second-layer or layer-wise training change the effective exponent from information to generative | Closest in spirit to C3's "readout protocol changes the effective exponent"; must be read and contrasted (they: in-weight, large-LR; we: in-context, finite-N shrinkage). | [snippets] |

## Background we rely on
- Ben Arous, Gheissari, Jagannath 2021 — online SGD for single-index: `d^{k−1}` (k≥3), `d log d` (k=2), `d` (k=1).
- Damian, Pillaud-Vivien, Lee, Bruna 2024 — generative exponent; label transformations lower the exponent. Relevant to why softmax attention may beat linear-in-label readouts (§4).
- Ba, Erdogdu, Suzuki, Wang, Wu, Yang 2022; Damian, Lee, Soltanolkotabi 2022 — one-step gradient feature learning.

## Open novelty questions (to resolve before writing)
1. Does Oko et al.'s Theorem 1 pretraining bound scale as `d^{Q}` or `d^{2Q}`-ish, and does their proof note the squaring? → decides how §2 is framed.
2. Has anyone written the finite-context repulsion (§2.2)? Search terms to try once arXiv is reachable: "in-context" "variance of the context statistic" alignment; "context length" "feature learning" pretraining threshold.
3. Is the multiplicative-emergence derivation for additive skills (§4) in Arora–Goyal or Okawa et al. already in this quantitative form?
