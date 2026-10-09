# Paper outline (v0, 2026-10-09) — target: ICML 2027 workshop (HiLD-type), 4–8 pages + appendix

## One ping
**The exponent of in-context skill emergence is set by the readout and the context length, not by the skill alone.**

## Working title candidates
- "Context length and readout parameterisation select the emergence exponent of in-context skills"
- "In-context feature learning of an additive model: when the context, not the skill, sets the plateau"
(Nam et al.-style rule: the title names the solvable setting; no claim about LLMs.)

## Abstract (draft, ~180 words)
We study how a feature-learning model acquires skills when the skills are presented *in context*: targets are additive models
`Σ_p c_p σ(⟨v_p,x⟩)` with mean-zero task coefficients, and a student with MLP features and a linear-in-label readout must learn the
directions `v_p` by online SGD from prompts. For a single skill of information exponent `k*` we derive the population loss in closed
form and show that (i) the in-context loss is the in-weight loss with the link squared — the effective exponent is `2k*`, a fact
implicit in one-step analyses (Oko et al. 2024) that we make dynamical; (ii) a finite context `N` adds an alignment-dependent
variance term that makes the unaligned state drift-stable below `m* ≈ √(2Γ/N)`, so that increasing the number of prompts cannot
substitute for context length — contrary to the `N·T` tradeoff of one-step analyses; (iii) a trainable scalar readout shrinks to a
Wiener factor that starves feature learning, raising the effective exponent toward `4k*` as `N` decreases, while a readout tied to the
feature norm does not. A parameter-free population ODE predicts the SGD-measured exponents across readout protocols and context
lengths (12/12 cells within 0.5). Many skills decouple with emergence times `∝ 1/π_p`; additive compositions of learned skills are
free. In small softmax transformers the effect appears as a ≈3× per-token handicap of short contexts with a collapsing readout, not as
a trap; we report the regime where it is invisible.

## Sections and figures

1. **Introduction** (1 p.) — the question (two hypotheses: context length as inference-time resource vs. pretraining order parameter);
   contributions (i)–(iii) + composition + transformer boundary; what is known (Oko Lemma 21; BBPV §5; Gu eq. 298; Kim–Suzuki C.4).
2. **Setup** (0.5 p.) — data, Model A (free / tied readout), Model B; Hermite notation; `g(m)`, `V(m)`.
3. **Single skill: closed form and effective exponents** (1.5 p.)
   - Prop. 1 (population loss, eq. 2.1) → `L_A = L_B[g→g²]` as `N→∞`.
   - Prop. 2 (finite-`N` repulsion; stability threshold `m*`); Remark: soft under SGD (Kramers).
   - Prop. 3 (trainable readout: `Γ*`, `4k*` for `d ≫ N^{1/k*}`; tied readout: `2k*`).
   - **Fig. 1**: drift vs `m` (MC vs closed form; `results/exp1/figs/drift_vs_m.png`) + `κ_eff` table/figure (`results/exp1b`,
     measured vs ODE; `figs/kappa_vs_N.png` regenerated with ODE overlay).
   - **Fig. 2**: escape-fraction grid with corrected `d*(N)` (`figs/threshold_grid.png`) + P7 panel (`figs/P7_tradeoff.png`).
4. **Many skills and composition** (1 p.)
   - Decoupling and `T_p ∝ 1/π_p` (exp 2, pending); frequency ordering twice as steep as in-weight (P16); scaling exponent (P18).
   - Additive composition free; no extra delay; product not free (exp 3). **Fig. 3**: per-skill curves + composition curves.
5. **Transformers: where the effect shows and where it does not** (0.75 p.)
   - Exp 4 (`d=32`, width 256): opposite `N`-dependence at fixed tokens — mechanism invisible when initial alignment is large.
   - Exp 5 (`d=256`, width 32): `N=16` stuck at `B=64`, emerges at `B=1024` ≈3× later than `N=256`; readout norm collapse while stuck.
   - **Fig. 4**: `T_e` per cell + readout-norm trace.
6. **Related work** (0.5 p.) — exact placement per `docs/novelty_protocol.md` (Oko; Nishikawa; Gu–Xu–Zdeborová; Ren; BBPV/BBSS;
   Kim–Suzuki; Zhang–Wu–Bartlett; Okawa; Arora–Goyal; Wang et al.; Kobayashi; He).
7. **Limitations** (0.25 p.) — single-neuron theory; `k*=2` focus; transformer results observational; SGD threshold soft; the Gaussian
   accuracy model fails; pre-registration deviations listed (κ=1 slip, threshold factor-2 slip, exp 3 estimator).
8. **Appendix** — derivations (Hermite algebra for `V(m)`), ODEs, all pre-registered predictions with outcomes (verbatim from
   `docs/preregistration.md`), full tables, compute.

## Claims → evidence map (must stay consistent with `docs/findings.md`)
| claim in paper | evidence | status |
|---|---|---|
| `L_A = L_B[g→g²]` | Prop. 1 + drift check | T, S |
| drift-stable origin for `k*≥2`, soft threshold | Prop. 2 + exp 1b grid (corrected `d*`) | T, S |
| `N` not substitutable by prompts (toy) | P7 | S |
| readout parameterisation selects exponent; ODE predicts SGD | Prop. 3 + exp 1b κ table + ODE | T, S |
| `T_p ∝ 1/π_p`, decoupling | exp 3 exploratory; exp 2 (pending) | S |
| additive composition free; product not | exp 3 | T (identity), S |
| transformer: ≈3× per-token handicap, readout collapse | exp 5 | X (observational) |
| transformer: invisible at large init alignment | exp 4 | X |

## Reviewer attack list (write the answers before submission)
1. "Squared link is Oko Lemma 21." — Yes; cited as mechanism; our claims are (ii)–(iii) and dynamics.
2. "Shrinkage formula is Gu et al. eq. 298." — Yes; static; ours is its effect on feature dynamics + parameterisation dependence.
3. "Tied vs free is just homogeneity (Ren Lemma B.1)." — The directional independence is known; its consequence for the finite-`N`
   exponent crossover is not; and the repulsion survives tying.
4. "Transformers don't show the trap." — Agreed and reported; the quantitative handicap and readout collapse are what transfers.
5. "Why `k*=2`?" — Excluded by Ren et al.; it is where in-weight is easy and in-context is not; `k*=1` gives only a log.
6. "SGD threshold is soft." — Reported; Kramers picture; barrier heights.
7. "LLM relevance?" — None claimed. Section 5 is the boundary.

## Disclosure
AI assistance (Claude) used for literature search, code, derivation checks, and drafting; all derivations re-verified by the
numerical drift check; all references verified against full texts by hand; venue policy to be checked when the CFP appears.
