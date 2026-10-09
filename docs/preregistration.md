# Pre-registration (v0, 2026-10-08)

Format follows `reports/研究の問いの立て方とAI支援科学.md` §7: default hypothesis / rival / discriminating
prediction / best-case headline / who changes behaviour / kill criteria. Numbers are fixed **before**
the corresponding experiment is run; any later change is recorded in the changelog at the bottom, never
edited in place. Novelty status of every claim: see `docs/novelty_protocol.md` (all UNVERIFIED until
full texts are read).

## Core question (C2/C3)

**Question.** When a feature-learning model is pretrained on in-context regression over a dictionary of
skills, does the *pretraining-time context length* `N` select which exponent governs the emergence
time of a skill — and does a trained readout mediate that selection?

**Default hypothesis (what the field would say).** Context length is an inference-time resource. The
pretraining sample complexity of acquiring a skill with information exponent `k*` is governed by `k*`
(or by a squared version of it, as in one-step analyses), and `N` only enters through the usual
`1/N` estimation error of the in-context statistic.

**Rival hypothesis (ours).** The finite-`N` noise of the context statistic has an alignment-dependent
variance `V(m)/N`. With a fixed readout `Γ` this makes `m = 0` a *stable* fixed point unless
`m_0² ≳ 16Γ/(N(4−2Γ))`; with a trained readout, the Wiener-filter shrinkage `Γ*(m) = g²/((1−1/N)g² + V/N)`
multiplies the feature gradient and changes the effective exponent from `2k*` to `4k*` exactly when
`d ≫ N^{1/k*}`. So `N` is a *pretraining* resource that sets a crossover in the emergence exponent.

**Discriminating predictions** (model A, `k* = 2`, `η = η_0/d²`, online SGD; see `docs/theory_notes.md`).

| ID | setting | default predicts | rival predicts |
|---|---|---|---|
| P1 | drift of `m` at small `m`, `N → ∞`, fixed `Γ` | `∝ m^{k*−1} = m` (if the field expects in-weight-like) or `m^{2k*−1} = m³` (one-step intuition) | `∝ m³`, with the full closed form `−∂_m L_A` of eq. (2.1) incl. the `V'/N` term matching Monte-Carlo within 20% at `m ∈ {0.05, 0.1, 0.2, 0.4}` |
| P2 | fixed `Γ = 1`, `N = 128`: fraction of seeds escaping `m_0` within `2·10⁶` steps vs `d` | escape at every `d` (just slower) | escape for `d ≲ 16`, **no escape** for `d ≥ 32` (threshold `m² < 16Γ/(N(4−2Γ)) = 0.0625`) |
| P3 | fixed `Γ = 0.1`, `N = 128`: same | escape at every `d` | escape for all `d ≤ 256` in the sweep; stuck only beyond `d ≈ N/(4Γ) = 320` |
| P4 | trained `Γ` (`Γ_0 = 0.01`), slope `s` of `log T_0.5` vs `log d` | a single slope (≈2 under `η ∝ d^{−2}` if exponent is `2k*`, ≈1.5 if `k*`) | `s ≈ 2` for `d ≪ √N`, bending to `s ≈ 4` for `d ≫ √N`; `N = 128 → 2048` moves the bend by `×4` in `d` |
| P5 | learned `Γ` at `T_0.5` | grows with training, no `N`-law | `≈ Γ*(0.5; N)`, independent of `d`, increasing in `N` |
| P6 | A vs B (in-weight), fixed readouts, `k* = 2`, `N → ∞`-like (`N = 128, Γ = 0.1`) | both ≈ `d log d` | slope difference ≈ 1 (3 vs ≈2) |
| P4′ | same as P4 but with the readout **tied** to the feature norm (`Γ_j = ‖w_j‖²`, 2-homogeneous as in Ren et al.) | same as P4 (parameterisation irrelevant) | **no bend**: slope ≈ 2 at every `N`; and with `ρ_0 = 1`, `N = 128` still stuck for `d ≥ 32` (C2 survives, C3 disappears) |
| P7 | **N–T tradeoff test** (vs Oko et al. 2024, Remark 3): fixed `Γ = 1`, `N = 128`, `d = 64` (above the predicted threshold); increase the number of tasks per step `B` from 32 to 1024 and the step budget ×10 | escape time decreases with total samples `B·T` (context length and task count are interchangeable) | **no escape at any `B`**: the repulsion is a drift bias, not averaging noise; the only knobs that restore escape are larger `N` (≥ `4Γd`) or smaller `Γ` |

**Best-case headline.** "Short pretraining context makes even skills unlearnable: the context length
sets the exponent of in-context feature emergence through readout shrinkage, with a sharp crossover at
`d ~ N^{1/k*}`." — a quantitative, falsifiable statement about a pretraining resource that current ICL
theory treats as inference-time only.

**Who changes behaviour.** Theorists of ICL (context length must enter pretraining analyses as an
order parameter, not just the `1/N` error); anyone designing synthetic-ICL pretraining curricula (context
length vs. dimension of the skill); people interpreting emergence curves of ICL abilities (the
`4k*` regime predicts far longer plateaus than in-weight theory).

**What would kill it.**
- K1: the drift check (P1) disagrees with eq. (2.1) by >20% at `m = 0.2, 0.4` and the discrepancy is not
  a code bug → the algebra is wrong; stop and recompute before any further run.
- K2: P2 fails (fixed `Γ = 1`, `N = 128` escapes at `d = 64`) → the repulsion is not real at SGD level
  (noise-induced escape dominates); the "threshold" story dies, keep only the exponent story if P4 holds.
- K3: P4 shows a single slope with no `N`-dependence across `N ∈ {32, …, 2048}` → the shrinkage story
  dies. Then the remaining content is C1 (doubling), which is not a paper.
- K4: a full-text read of Oko et al. 2024 / Gu–Xu–Zdeborová 2026 / 2510.21020 shows the same
  `V(m)/N`-driven mechanism → stop, do not reframe.
- Time box: K1–K3 are decidable within 2 weeks of CPU time; if undecided by 2026-10-22, pivot.

**Pre-declared analysis choices.** 5 seeds per config (exp 1), 3 (exp 1b); medians and IQR; slopes by
least squares on medians over the declared `d` grid; `T_0.5` as the emergence time; `η_0`-invariance of
`T_0.5·η` within 30% as the admissibility criterion for a config (configs failing it are reported, not
dropped). No metric, grid, or seed count will be changed after seeing results without a changelog entry.

## Transfer test (exp 4, small softmax transformer) — pre-registered 2026-10-08 before any run; spec in `docs/spec_exp4.md`

| ID | setting | default predicts | rival (ours) predicts |
|---|---|---|---|
| P8 | 2-layer softmax transformer, in-context regression with a `k*=2` skill, `d=32`, tokens/step fixed (`N·B = 8192`), `N ∈ {16, 64, 256}` | emergence step `T_e` ≈ independent of `N` (Oko Remark 3: only `N_1T_1` matters) or increasing with `N` | `T_e` decreases ≥ ×3 from `N=16` to `N=256`; some `N=16` seeds never emerge in 30k steps |
| P9 | same, `k*=1`; and in-weight control (`c≡1`, labels zeroed) for `k*=2` | — | `k*=1`: `N`-dependence < ×1.5; in-weight control emerges faster than every in-context cell |
| P10 | linear-attention variant, `k*=2`, `N ∈ {16, 256}` | softmax label nonlinearity removes the `N`-dependence (Nishikawa-style reading) | `N`-dependence present in both; recorded either way |

Kill: P8 fails ⇒ the toy phenomenon does not transfer at this scale; that becomes the headline of exp 4.

### Exp 4 outcome (recorded 2026-10-08 after the run; `results/exp4/README.md`)

- **P8 FAILED, direction reversed; the pre-registered kill condition is met.** Softmax, `k*=2`, `d=32`, tokens/step fixed: median
  `T_e` = 1000 / 1400 / 2800 steps at `N` = 16 / 64 / 256 (seed ranges at `N=16` and `N=256` do not overlap); `d=16`: 400 / 500 / 800.
  Larger context at fixed tokens/step is *slower*, matching the default ("fewer prompts per step") not our prediction.
- **P9 partly held:** in-weight control emerges in 40–60 steps vs ≥ 800 for every in-context `k*=2` cell (in-weight ≫ in-context, as
  predicted); `k*=1` `N`-dependence is ×2.0 at 20-step resolution (predicted < ×1.5), same direction as `k*=2`.
- **P10:** `N`-dependence present in both softmax and linear attention, same direction; linear is slower (and never emerged at `N=256`
  within 6000 steps).
- **Post-hoc diagnosis (not a reinterpretation of P8; recorded as a hypothesis for a new pre-registration):** the alignment proxy
  `max_j |cos(W_j, v)|` over 256 MLP neurons is already **0.43–0.55 at initialisation** for `d=32`. The toy phenomenon needs
  `m_0 ≲ 2√(γ/N)` (a drift-stable origin), i.e. a small *initial* alignment, which a wide MLP in low `d` never has: the best of
  256 random directions in 32 dimensions is far above the repulsion barrier. At this scale the pretraining bottleneck is gradient
  noise across prompts (`B`), not the context statistic's noise. Whether the mechanism matters at all in transformers is therefore
  **untested, not refuted**, and testable only in the regime `d ≫ width` (e.g. `d=256`, width 32–64, where the max initial
  alignment is ≈ 0.15–0.2) with `B` held fixed and `N` varied, plus a readout that is not initialised tiny. That experiment, if run,
  needs its own pre-registration (P11) and an explicit statement that both outcomes change the paper's scope.
- Budget 6.0 CPU-h (cap 6).

**Scope consequence (binding until P11 is run):** C2/C3 may be claimed only for the single-neuron / narrow regime where the
initial alignment is `Θ(d^{−1/2})`; any statement about transformers must say that a 2-layer softmax transformer at `d=32`,
width 256 showed the opposite `N`-dependence at fixed tokens/step.

## Regime-matched transformer test (exp 5) — pre-registered 2026-10-08 before any run; spec in `docs/spec_exp5.md`

| ID | setting (`d=256`, MLP width 32, `k*=2`, softmax, 30k steps) | default predicts | rival (ours) predicts |
|---|---|---|---|
| P11a | (a) `N=256,B=64`; (b) `N=64,B=64`; (c) `N=16,B=64` | (c) emerges, later (tokens) | (a),(b) emerge all seeds; (c) stuck ≥2/3 seeds |
| P11b (decisive) | (d) `N=16,B=1024` — same tokens/step as (a) | (d) emerges at ≈ the step count of (a) or sooner | (d) stuck ≥2/3 seeds while (a) emerges |
| P11c | `k*=1` at `N=16,B=64`; in-weight control | — | both emerge < 2000 steps |

Kill/scope: (d) emerges at ≲ 2× the steps of (a) ⇒ C2/C3 claimed for the single-neuron/narrow model only; both transformer
experiments reported as the boundary.

### Exp 5 outcome (recorded 2026-10-09 after the run; `results/exp5/README.md`)

- Init alignment in the `d=256`, width-32 regime: 0.126 / 0.152 / 0.176 (the small-alignment regime was reached).
- **P11a: mixed.** (c) `N=16, B=64` stuck 3/3 at 30k steps ✓; (a) `N=256, B=64` emerged 2/3 (third censored at 17k by budget, still on
  the plateau); (b) `N=64` 0/1 (one seed only — under-powered, no conclusion).
- **P11b (decisive): the rival prediction FAILED, and so did the default.** (d) `N=16, B=1024` (same tokens/step as (a)) emerged 3/3 at
  10.6k–14.8k steps — not stuck — but **≈3× later than (a)** (per-seed ratios 3.7, 3.8; median 2.9; in seed 2 (d) emerged while (a)
  had not by 17k). The default ("(d) at about the same step count as (a), or sooner") failed in both seeds where (a) emerged. The
  kill rule ("(d) ≲ 2× (a)") is not met; the transfer claim as pre-registered ("stuck") is not met either.
- **P11c held:** `k*=1` at `N=16`: 1200–1400 steps; in-weight: 200–400.
- **Unplanned observations (exploratory):** in stuck runs the final-head weight norm **collapses from 0.57 to 0.03–0.16 and regrows
  after emergence** — the readout-shrinkage signature of §2.3 in a transformer; alignment drifts to 0.3–0.4 in stuck runs without
  emergence and jumps to 0.9–0.99 at the MSE drop.
- Budget 5.98 CPU-h (cap 6).

**Scope consequence (binding):** in a softmax transformer in the small-alignment regime, short context is a *quantitative*
handicap per token (≈3× more steps at equal tokens/step; `N=16` with 16× the prompts emerges while `N=16` with 1× never does within
budget) and comes with a shrinking readout, but it is **not** a trap. The paper may claim: (i) the solvable-model results in full;
(ii) for transformers, only "context tokens are worth more than prompt tokens for acquiring an even skill (≈3× at
`d=256`, width 32, `N` 16→256), with the readout shrinking while the skill is not yet acquired" — stated as an observation
consistent with, not a confirmation of, the mechanism; (iii) exp 4's opposite result at `d=32`, width 256 as the boundary.

## Secondary question (C4) — pre-registered 2026-10-08 before any run; spec in `docs/spec_exp3.md`

Setting: `d=32`, `P=2` orthogonal `σ_2` skills with frequencies `0.75/0.25`, `M=4` neurons, fixed `Γ=0.1`, `N=128`, single-skill
prompts only.

| ID | quantity | default predicts (Arora–Goyal Cor. 13 / Okawa) | rival (ours, §4) predicts |
|---|---|---|---|
| P12 | additivity residual `R(t) = MSE_12 − MSE_1 − MSE_2` | not constrained | `|R| ≲ 0.02` (=O(1/N)) at all `t` |
| P13 | `T_12` vs `max(T_1,T_2)` | extra delay for the pair (≈ ×10 scale per doubling in A–G) | `T_12 = max(T_1,T_2)` within one logging interval (500 steps) |
| P14 | `acc_12(t)` vs `acc_1(t)·acc_2(t)` | multiplicative by definition/independence (Okawa) | multiplicative *approximately*, with the deviation predicted by the Gaussian-error model (report both) |
| P15 | non-additive pair `σ_1(v_1x)σ_1(v_2x)` | — | MSE stays ≈ 1 throughout (not free) |

Kill: P12 or P13 fails ⇒ §4 is wrong as stated (the exact additivity argument has an error); stop and recompute.

### Exp 3 outcome (recorded 2026-10-08 after the run; `results/exp3/README.md`)

- **Two design errors of mine made the pre-registered estimators unusable, and I record them as such:** (i) with fixed `Γ=0.1` and
  `M=4` a skill's MSE floor is `(1−0.1n)²` (0.36 with all four neurons), so the stop rule (`MSE<0.1`) and `T_12` (`E12<0.5`) can never
  fire; (ii) the additivity residual `R` was defined on *independent* eval sets, whose sampling noise (0.04–0.35) dwarfs the 0.02 bound.
  **P12 as pre-registered therefore fails** (13/13 runs exceed 0.02). The kill clause says "§4 wrong; recompute": recomputed — the
  additivity `ŷ(c_1,c_2) = ŷ(c_1,0) + ŷ(0,c_2)` is an *identity* of a label-linear model (unit-tested to 1e-13), and the loss
  additivity needs only `E[c_1c_2]=0`. A **post-hoc paired estimator** (same inputs and `c`, 17 queries/context) gives
  `max|R| ≤ 0.0086` over all 13 runs. Verdict: estimator failure, not theory failure — but P12 is not a confirmation either, since
  the statement is an identity; it should not have been pre-registered as an empirical test.
- **P13 (no extra compositional delay): supported with substitute definitions, untested as written.** E12 reaches 90% of its total
  drop 1,000–4,500 steps *before or at* the later skill's own 90% point, never after. Against Arora–Goyal's extra delay for tuples.
- **P14 (multiplicative accuracy): partly.** Observed `acc_12 ≈ 0.85·acc_1·acc_2 + 0.09`, RMS from `y=x` 0.008 — approximately
  multiplicative — but the **Gaussian-error derivation is wrong in level** (predicts 0.52 at `t=0`, observed 0.77; errors are
  heavy-tailed). The "derived from additive structure" wording must be weakened to "additivity of the loss + any threshold metric
  gives approximate multiplicativity; the Gaussian closed form does not fit".
- **P15 (product composition not free): met** (E× MSE 0.91–1.05 throughout).
- **Unplanned (exploratory) findings:** neurons arrive independently with `T ≈ K/(η π_p m_0²)`, `K = 0.60` (0.54–0.78) vs the
  single-neuron exp 1b value 0.64 — multi-neuron dynamics decouple into the single-neuron one with rate `∝ π_p` (supports §3's
  per-skill timescale picture; the factor-3 frequency delay is clean but hidden behind random `m_0`). Decoupling failures: skill 2
  never acquired in 2/10 seeds (all neurons → skill 1), a neuron trapped near the origin in 3/13 runs (the F2 repulsion at `Γ=0.1`,
  `N=128` is weak but real for unlucky `m_0`).
- Budget 2.3 CPU-h (cap 3).

**Scope consequence:** C4's defensible content is "label-linear ICL ⇒ additive composition costs nothing (identity), arrives with
the slower skill (observed), is approximately multiplicative in thresholded accuracy (observed), and non-additive composition is
not free (observed)"; the quantitative multiplicative law is *not* derived.

## Many-skill scaling (exp 2) — pre-registered 2026-10-09 before any run; spec in `docs/spec_exp2.md`

Setting: `d=32`, `σ_2`, `P=16` skills, `π_p ∝ p^{−1.5}`, single-skill prompts, `M=64`, fixed `Γ=0.1`; in-weight baseline with
`a_p² ∝ π_p`.

| ID | quantity | prediction |
|---|---|---|
| P16 | slope of `log T_p` vs `log p` | A: `1.5 ± 0.2` (`T_p ∝ 1/π_p`); B: `0.75 ± 0.2` (`T_p ∝ 1/a_p`) — in-context orders skills by frequency twice as steeply |
| P17 | relative step sharpness `(T_{0.9}−T_{0.1})/T_{0.5}` | A's median < B's median / 2 |
| P18 | power-law exponent of the alignment-based frequency-weighted loss | A ≈ 0.33, B ≈ 0.67 (ratio ≈ 1/2), ±0.15 |

Kill: P16 slope for A < 1.0 ⇒ the "`π_p` vs `√π_p` ordering" claim is wrong.

### Exp 2 outcome (recorded 2026-10-09 after the run; `results/exp2/README.md`)

- **Deviation (declared):** the spec's in-weight baseline with readout 1 and `M=64 > d` cannot learn (trace mismatch); the agent
  switched to a capacity-matched readout `ρ = Σa_p/M` after looking at seed-0 pilots but before computing any B statistic. B is then a
  *collective* fit (pooled alignment → 1, individual neurons at 0.5–0.9), so every A-vs-B comparison below carries that caveat.
  Also: `T_p(0.5)` is contaminated by initial alignments (best of 64 neurons already at 0.36–0.55); the drop-midpoint `T_{p,0.5}` is
  the cleaner statistic (reported alongside the pre-registered one).
- **P16 held.** A: slope `1.53 ± 0.22` (spec `T_p(0.5)`), `1.44 ± 0.08` (drop midpoint), `1.45 ± 0.07` (`T_p(0.9)`) vs predicted
  `1.5 ± 0.2`; B: `0.75 ± 0.08` (drop midpoint) vs predicted `0.75 ± 0.2` (spec statistic `1.15 ± 0.24`). A − B = `0.69 ± 0.11`
  (midpoint), `0.38 ± 0.33` (spec statistic, n.s.). Kill not triggered. Five-seed A slopes `1.52–1.66`. **In-context ordering of
  skills by frequency is ≈ twice as steep as in-weight** — with the B-readout caveat (B at `ρ=0.1` gives `0.34`).
- **P17 held** (median sharpness A 1.08 vs B 4.09, ratio 3.8), weakened by B's partial drops.
- **P18: A within tolerance (0.40 ± 0.035 vs 0.33 ± 0.15 on the primary alignment loss; 0.27 on MSE), B failed (0.19 vs 0.67):** B's
  alignment loss saturates at 0.37–0.40 because neurons share skills — the decoupled-neuron prediction does not apply to a
  collective fit. The predicted A/B exponent ratio is therefore **unsupported**, and the exponent depends on the emergence
  definition (0.40 vs 0.73 with midpoints) because the window spans only a factor 3–8 in time on a staircase.
- **Consistency:** `T_{p,0.5}·η·π_p·m_0² = 0.73` (IQR 0.70–0.77) across skills and seeds — the same constant as exp 3 (0.60–0.78) and
  exp 1b (0.64): the many-skill in-context dynamics is the single-neuron dynamics with rate `∝ π_p`.
- Budget 3.2 CPU-h (cap 4).

- **Addendum (B-tied, 2026-10-09):** Ren et al.'s 2-homogeneous in-weight student (no hand-chosen readout, `η ∈ {0.25,1,4}/d²`) gives
  slope `0.75 ± 0.09` (1000-step log) / `0.87–0.89` (100-step log), `T_p·η·a_p = 0.23–0.26` constant; it is *also* a collective fit at
  `M=64>d` (pooled alignment 1.000, no feature above 0.9). So the in-weight ordering by `a_p ∝ √π_p` is robust to the readout
  protocol, and the A-vs-B contrast (1.44–1.53 vs 0.75–0.89, i.e. ×1.6–2) stands with the collective-fit caveat on B. Exponent: 0.24–0.31
  (pre-registered 0.67 — not met; transient in front of a plateau at 0.40).

**Scope consequence:** claim `T_p ∝ 1/π_p` (decoupling with frequency-proportional rate) for the in-context model, and the ≈2×
steeper ordering than in-weight *at matched readout capacity*; do not claim a scaling-exponent ratio.

## N–T exchange by readout protocol (exp 6) — pre-registered 2026-10-09 before any run (reviewer change #3)

Setting: Model A, `σ_2`, `d=64`, `m_0 = d^{−1/2}` fixed, `N ∈ {16, 64, 256}`, prompts per step `B` chosen so that **tokens per step `N·B = 4096`**
(`B = 256, 64, 16`), `η = 1/d²` fixed (the population drift does not depend on `B`; only the noise does); *also* run the `B`-matched control `B=64` for all `N`. Three protocols: pinned `Γ=1`; pinned `Γ=0.1`; free
`Γ_0=0.01, η_Γ=η`; tied `ρ_0=0.01`. 3 seeds, `max_steps 10⁶`.

| ID | protocol | prediction (population ODE at these parameters, computed before the run) |
|---|---|---|
| P19a | pinned `Γ=1` | ODE: trapped at **all** three `N` (`d*(256) ≈ 63 < 64`, marginal) — SGD noise may let `N=256` escape; no `B` compensates at `N ≤ 64` |
| P19b | free | ODE flow times `448 / 153 / 101` at `N = 16/64/256` (steps ×4096): `T(16)/T(256) ≈ 4.4` at fixed `B` — a strong but sub-linear trade-off (partially noise-dominated at `d=64`); with `N·B = 4096` fixed, the step counts should still differ by ≈4 (the population drift does not see `B`), i.e. **not** interchangeable one-for-one at this `d`, but far from a trap |
| P19c | tied | ODE `4.21 / 4.07 / 4.03`: `T` independent of `N` within 5% |
| P19d | pinned `Γ=0.1` | ODE `101 / 48 / 43`: escapes at all `N`, `T(16)/T(256) ≈ 2.3` |

Kill: P19b fails (free readout `T` independent of `N` at fixed `B`, or a trap appears) or P19c fails (tied readout shows the free
readout's `N`-dependence) ⇒ the reframed central claim is wrong. ODE numbers computed with `scripts/ode_kappa_exp1b.py` functions
before any SGD run (see git log).

Kill: P19b fails (free readout `T` independent of `N` at fixed `B`, or the trap appears) ⇒ the reframed central claim is wrong.

### Exp 6 outcome (recorded 2026-10-09 after the run; `results/exp6/README.md`)

- **P19a held:** pinned `Γ=1`: 0/9 escapes at `N ≤ 64` (incl. `B=256`), 3/6 at `N=256` (marginal `d* ≈ 63`); stuck runs end at `|m| ≤ 0.021`.
- **P19b held, better than written:** free readout medians `1.78M / 614k / 402k` steps (ODE `1.835M / 627k / 414k`), ratio `T(16)/T(256) = 4.4` (ODE 4.44);
  **tokens-matched and `B=64` schemes give the same step counts** (`1.88M/614k/411k`) — prompts per step do not substitute for context in
  any protocol; what trades off against `N` is training *time*. The `N=16` cells needed a rerun to `2.5·10⁶` steps (pre-registered cap `10⁶`
  was below the ODE time; declared).
- **P19c partly:** tied `17.0k / 15.1k / 14.0k` (ODE `17.2k / 16.7k / 16.5k`) — flat within ×1.25 over 16× in `N`, not within the literal 5%
  (seed scatter 4–20%); kill not triggered.
- **P19d held:** pinned `Γ=0.1`: `396k / 192k / 179k` (ODE `414k / 197k / 176k`).
- Budget 2.4 CPU-h (cap 3). 12 of 72 runs were duplicates (`N=64` has `B=64` in both schemes).

**Scope consequence:** the three-regime statement (Prop. 3) is quantitatively confirmed at `d=64`: the population ODE predicts the SGD escape
steps to 1–4% for pinned-0.1 and free readouts across `N`, the trap for pinned-1, and `N`-independence (within scatter) for tied.

## Controls requested by review 2 (exp 7) — pre-registered 2026-10-09 before any run

Setting as exp 6 (`σ_2`, `d=64`, `m_0=d^{−1/2}`, `η=1/d²`, `N ∈ {16,64,256}`, `B=64`, 3 seeds, cap `2.5·10⁶` steps). ODE flow times (×4096 = steps)
computed before the run:

| ID | protocol | ODE `T` at `N=16/64/256` (flow) | ratio `T(16)/T(256)` | what it tests |
|---|---|---|---|---|
| P20a | free `Γ_0=0.01, η_Γ=10η` | 447 / 122 / 50.1 | 8.9 | the exponent/ratio depends on `η_Γ/η` (review-2 issue 3) |
| P20b | free `Γ_0=0.1, η_Γ=η` | 363 / 66.9 / 40.1 | 9.1 | dependence on `Γ_0` |
| P20c | tied `ρ_0=0.1` | 8.13 / 4.76 / 4.34 | 1.9 | `N`-independence needs `ρ_0 ≪ N m_0²` (= 0.25 at N=16) |
| P20d | tied `ρ_0=0.3` | ∞ / 7.67 / 5.2 | trap at `N=16` | tied readout with `ρ_0 > N m_0²` traps like a pinned one |
| P20e | pinned `γ=0.01` | 421 / 407 / 403 | 1.04 | a small pinned readout is as `N`-flat as tied |
| P20f | pinned `γ=0.1`, **`η ∝ B`**: `B ∈ {16, 256}` with `η = (B/64)·(1/d²)` at `N=64` | not an ODE quantity | — | whether "prompts per step never substitute" is an artefact of fixed `η` (review-2 issue 2): prediction — steps scale as `1/η` (i.e. `×4` fewer steps at `B=256`, `×4` more at `B=16`) *if* the larger step size stays in the drift-dominated regime; escape **flow time** `T·η` unchanged |

Kill: none of these changes the three-regime statement; they fix its stated conditions. If P20d does not trap, the claim "the repulsion survives
tying" is wrong.

### Exp 7 outcome (recorded 2026-10-09 after the run; `results/exp7/README.md`; 2.2 CPU-h, cap 2 overrun by 10%)

- **P20a (free, `η_Γ=10η`): held qualitatively, 24% slow at `N=16`** (`2.26M` vs ODE `1.83M`; `N=64/256` within 4%); ratio 10.7 (ODE 8.9) —
  the `η_Γ/η` dependence is real and in the predicted direction (exp 6 at `η_Γ=η`: 4.4).
- **P20b (free, `Γ_0=0.1`): held** (all medians within 2%; ratio 9.16 vs 9.05).
- **P20c (tied, `ρ_0=0.1`): held on medians** (ratio 2.15 vs 1.87); one `N=16` seed took 15–22× longer than the other two (a tail the ODE
  does not have).
- **P20d (tied, `ρ_0=0.3`): FAILED as worded.** `N=16` escaped in 3/3 seeds at `6.1–8.5·10⁵` steps (≈30× slower than `N=64/256`), after `ρ`
  had decayed from 0.3 to ≈0.012. The deterministic flow predicts a trap because `m` reaches 0 before `ρ` decays; under SGD noise keeps `m`
  away from 0 and the algebraically decaying `ρ` eventually releases the feature. **By the pre-registered kill rule, "the repulsion
  survives tying" is wrong as worded: tying converts the trap into a delay** (`ρ ∝ 1/t` release), whereas a pinned readout cannot
  decay and traps for good. Exp 1b's tied `ρ_0=1, d=64` "stuck at 10⁶ steps" is now read as the same delay, not a trap.
- **P20e (pinned `γ=0.01`): held** (within 2%; ratio 1.056 vs 1.045).
- **P20f (`η ∝ B`): held** — steps scale exactly as `1/η` (×4.02, ×0.250), flow time `T·η` invariant to 0.5%, total prompts to escape
  unchanged (`1.23·10⁷`). So at matched *flow time* the number of prompts per step is irrelevant; "prompts do not substitute for context" is
  a statement about flow time, and the step-size scaling is a separate choice.

**Scope consequence:** Prop. 3(c) must say "delayed, not trapped" for `ρ_0 ≳ N m_0²/4`; the pinned-readout trap is the only permanent one.

## Transformer trap-vs-tradeoff test (exp 8, backlog N1) — pre-registered 2026-10-09 before any run

Setting: exp 5's transformer (`d=256`, `d_model=64`, MLP width 32, 2 blocks, softmax, Adam `1e-3`, `k*=2`, 30k steps, eval every 200). The
knob standing in for the toy's readout `Γ` is the **final linear readout** (the closest available analogue; it multiplies the residual stream,
not the context statistic directly — caveat recorded). Protocols: (F) free/trainable (= exp 5); (P1) frozen at its initial norm (≈0.57, "pinned
O(1)"); (Ps) frozen at `0.1×` its initial norm ("pinned small"). Cells: `N=16, B=64`; `N=16, B=1024`; `N=256, B=64`; 2 seeds each; cap 6 CPU-h.

| ID | protocol | prediction (from Prop. 3 by analogy) | default |
|---|---|---|---|
| P21a | P1 (pinned O(1)) | `N=16` stuck at **both** `B=64` and `B=1024` (no tradeoff: the trap); `N=256` emerges | B compensates as in exp 5 |
| P21b | Ps (pinned small) | `N=16, B=64` **emerges** (rescued relative to F, which was stuck 3/3 in exp 5), at a time within ×2 of `N=256` | still stuck |
| P21c | F (free) | replicates exp 5: `N=16,B=64` stuck, `N=16,B=1024` emerges ≈3× later than `N=256` | — |

Kill: P21a fails (B=1024 rescues the pinned-O(1) readout) **and** P21b fails (small pinned readout does not rescue) ⇒ the readout–context
mechanism does not transfer to this transformer even qualitatively; the paper keeps Sec. 5 as observations only. Either one holding is
reported as partial transfer.

## `k*=3` sanity run (exp 9, backlog N3) — pre-registered 2026-10-09 before any run; ODE numbers from `scripts/ode_k3.py` (`results/exp9/ode_predictions.csv`)

Setting: single-neuron Model A / tied Ã code of exps 6–7 unchanged except `k=3` (`σ_3 = He_3/√6`, `g=m³`, `V = 1+18m²+54m⁴+20m⁶`),
`m_0 = d^{-1/2}` exactly (`init="fixed"`), `B=64`, **`η = 2·10⁻⁴` for every cell** (so cross-`d` comparisons are not confounded by `η`),
early stop at `|m| ≥ 0.5`, cap `10⁶` steps, 3 seeds, 1 BLAS thread per worker, total cap 2 CPU-h. Flow time = steps·η. Cells:
(a) tied `ρ_0 = 0.01`, `N=128`, `d ∈ {8,16,32}`; (b) tied `ρ_0 = 0.01`, `N=16`, same `d`; (c) pinned `γ = 0.2`, `N=128`, same `d`.

| ID | cell | ODE prediction (flow time → steps) | pass criterion | default |
|---|---|---|---|---|
| P22a | (a) | τ = 1.232 / 5.88 / 26.58 → 6.2e3 / 2.94e4 / 1.33e5 steps; secant `κ_eff` (2 − slope of log T vs log m₀) = **6.51** (d 8→16), **6.35** (16→32), i.e. the `k=3` analogue of the `2k*` rule (`2k*=6`; `k=2` gave ≈4) | every median within 15% of ODE steps; both secants within ±0.5 of ODE | secant ≈ 4 (no `k`-dependence) or ≈ 2 (in-weight-like) |
| P22b | (b) vs (a) | tied readout is **not** `N`-flat at `k=3`: ratio `T(N=16)/T(N=128)` = 1.16 (d=8), 1.63 (d=16); at `d=32` the `N=16` flow **traps** (`ρ_0=0.01 > ρ_trap = 0.0059`), so under SGD the `d=32, N=16` median is ≥ 3× the `N=128` median (soft trap → delay, as in E5) | ratios within ±0.25 at d=8,16; d=32 ratio ≥ 3 (censored counts as ∞) | `N`-flat at all `d` (the k=2 picture carried over naïvely) |
| P22c | (c) | `m* = 0.3006`: `d=8` (`m₀/m* = 1.18`) escapes, τ = 11.83 → 5.9e4 steps; `d=16` (`0.83`) and `d=32` (`0.59`) trapped in the flow | d=8 median within 15%; d=16 and d=32 each censored in ≥ 2/3 seeds at 10⁶ steps | d=16 or 32 escapes 3/3 |

Honest caveats fixed now: the SGD trap is soft (Kramers), so `d=16` in (c) at `m₀/m*=0.83` may escape in a minority of seeds; one escape of three is
consistent, three of three is a failure. At `d=8` the flow lasts only ~6·10³ steps at `η=2·10⁻⁴`, so discretisation error of a few % is expected and
is covered by the 15% band. `ρ` is predicted to stay at `ρ_0` (`ρ(T) = 0.0105`); it is recorded. Kill: P22a fails on the secants (both < 5 or both > 8)
⇒ the `2k*` statement is specific to `k=2` and the paper must say so.

## Exp 9 outcome (recorded 2026-10-09 after the run; `results/exp9/README.md`, commit 3de17b5)

27/27 runs, 0.89 CPU-h. **P22c held** (pinned γ=0.2: d=8 median 60212 vs ODE 59140, ratio 1.02; d=16 and d=32 censored 3/3 at 10⁶ steps as
predicted from m*=0.30). **P22a failed on the 15% band, kill not triggered**: tied N=128 medians/ODE = 1.18 (d=8), 0.89 (d=16), 0.80 (d=32);
secant κ_eff = 5.70 (8→16) and 6.02 (16→32) vs ODE 6.51 / 6.35 — the k=3 exponent is ≈6, clearly above the k=2 value ≈4, but the SGD runs
are faster than the flow at d=32 and slower at d=8 (both outside ±15%). **P22b failed at d=16**: T(16)/T(128) = 1.14 at d=8 (predicted 1.16)
and ∞ at d=32 (3/3 censored; predicted trap), but at d=16 the N=16 seeds gave 47171 / 499629 / censored (max |m| = 0.27, ρ drifted down to
0.0073–0.0090) against a flow prediction of 1.63× — with ρ₀ = 0.01 only a factor 2.6 below ρ_trap = 0.026, the k=3 drift ∝ m⁵ is so weak near
m₀ that SGD noise in (m, ρ) decides between escape and a long delay. ρ stayed at ≈ ρ₀ in every escaping run (0.0103–0.0110 at N=128).

Reading: the `2k*` exponent rule carries over to k=3 (exponent ≈6), the pinned trap threshold is quantitatively right, and the tied readout's
"N-flatness" is **not** a k-independent statement: its condition is ρ₀ ≪ ρ_trap(m₀, N) ∝ m₀^{2k−2}·(…), which for k=3 is violated already at
N=16, d≥16. The paper's Prop. 3(c) must say "N-independent for ρ₀ ≪ ρ_trap" with ρ_trap given, not "N-independent".

Post-hoc diagnostics, declared here **before** running them (not pre-registered; flow predictions unchanged):
- P22d: cell (a) d=8 at η = 5·10⁻⁵ (4× smaller). If the 1.18 excess is discretisation, the median moves toward the ODE's 24632 steps (within 10%).
- P22e: cell (a) d=32 at B=256 (4× less gradient noise, η unchanged). If the 0.80 deficit is noise-assisted escape, the median moves toward the
  ODE's 132880 steps (within 10%); if it stays at ≈0.80 the flow itself is off at this m₀ (e.g. an O(m₀²) correction to the m₀=d^{-1/2} picture).

Diagnostics outcome (`results/exp9/diag/runs.csv`, 6 runs, 0.23 CPU-h): **P22d held** — at η=5·10⁻⁵ the d=8 medians are 23277 / 24815 / 30249
steps, median/ODE = 1.007 (was 1.18 at η=2·10⁻⁴): the excess was discretisation. **P22e: sign flipped** — at B=256 the d=32 medians are 132061 /
151406 / 164371, median/ODE = 1.14 (was 0.80 at B=64; per-seed 0.99 / 1.14 / 1.24). The deficit at B=64 is therefore noise-assisted escape, as
hypothesised; the 14% overshoot at B=256 is inside the 3-seed spread (±12%) but outside the declared 10% band, so the flow is confirmed only to
≈15% at this m₀. Net: with discretisation and noise controlled, the k=3 tied flow reproduces SGD within 15% at all three d.

## Three regimes at a second `d` (exp 10, backlog N2) — pre-registered 2026-10-09 before any run; ODE numbers from `scripts/ode_exp10.py` (`results/exp10/ode_predictions.txt`)

Setting: exp 6's protocols unchanged (`σ_2`, `m_0 = d^{-1/2}` exactly, `η = 1/d²`, `B = 64`, early stop `|m| ≥ 0.5`), now at **`d = 128`**
(`m_0 = 0.0884`, `η = 6.1·10⁻⁵`). Cells: (a) pinned `Γ=1`, `N ∈ {16,64,256}`, 2 seeds, cap `10⁶` steps; (b) pinned `Γ=0.1`, `N ∈ {16,64,256}`,
3 seeds, cap `2.5·10⁶`; (c) tied `ρ_0 = 0.01`, `N ∈ {16,64,256}`, 3 seeds, cap `10⁶`; (d) free `Γ_0 = 0.01`, `η_Γ = η`, `N = 256` only (budget),
2 seeds, cap `1.2·10⁷`. Total cap 5 CPU-h. The point of the run: the `d`-dependence of the regime boundaries. At `d = 64` (exp 6) pinned
`Γ=0.1` escaped at every `N`; the threshold `d*(N) = N(8−4γ(1−1/N)−24γ/N)/(16γ)` gives `d*(16) = 76`, `d*(64) = 304`, `d*(256) = 1216`, so at
`d = 128` the `N=16` cell must flip to trapped while `N=64, 256` still escape.

| ID | cell | ODE prediction | pass criterion | default |
|---|---|---|---|---|
| P23a | (a) pinned Γ=1 | trapped at all three `N` (`m* = 0.60, 0.26, 0.126 > m_0`; `d*(256) = 63 < 128`) | censored 2/2 at every `N` | `N=256` escapes |
| P23b | (b) pinned Γ=0.1 | **`N=16` trapped** (`m* = 0.116 > m_0`), `N=64`: τ = 110.1 → 1.80·10⁶ steps, `N=256`: τ = 89.0 → 1.46·10⁶ (ratio 1.24: `N`-saturating, not `1/N`) | `N=16` censored ≥ 2/3 at 2.5·10⁶; `N=64, 256` medians within 15% | `N=16` escapes 3/3 (no `d`-dependence of the trap) |
| P23c | (c) tied | τ = 8.76 / 8.20 / 8.07 → 1.44·10⁵ / 1.34·10⁵ / 1.32·10⁵ steps; `N`-flat (ratio 16/256 = 1.09); `ρ(T) ≈ 0.011` | medians within 15%; ratio within ±0.15 of 1.09 | ratio > 2 |
| P23d | (d) free N=256 | τ = 327.9 → 5.37·10⁶ steps; against exp 6's `d=64` value (τ = 50.1) the secant exponent over `d` is `κ = 2 − ln(327.9/50.1)/ln(2^{-1/2}) = 7.4` (crossover regime, between 4 and 8) | median within 15%; secant within ±0.5 of 7.4 | secant ≤ 4.5 or ≥ 8.5 |

Honest caveats fixed now: the SGD trap is soft, so P23b's `N=16` (`m_0/m* = 0.76`) may escape in one seed; P23a's `N=256` has `m_0/m* = 0.70`
and the same caveat. The free cell is long (`5·10⁶` steps at `η = 6·10⁻⁵`); if the 5 CPU-h cap is hit the second free seed is dropped and
reported as such. Kill for the "trap for `d > d*(N)`" statement: P23b `N=16` escapes 3/3 **and** P23a `N=256` escapes 2/2.

## Exp 8 outcome (recorded 2026-10-09 after the run; `results/exp8/README.md`)

18 cells × seeds (14 fresh runs, 4 free cells reused from exp 5 with identical seeds and code), 5.47 CPU-h (cap 6). `T_e(0.5)` per seed:
pinned O(1) (`‖readout‖ = 0.56–0.58`, frozen): `N=16,B=64` not reached 2/2 (30k); `N=16,B=1024` **8400, 5800**; `N=256,B=64` 6200, 2400.
Pinned small (`0.056–0.058`, frozen): `N=16,B=64` **not reached 2/2**; `N=16,B=1024` 6600, 14600; `N=256,B=64` 4600 and one seed censored at 17k
(killed for budget while still on the plateau). Free: `N=16,B=64` not reached 2/2 (re-run; reproduces exp 5); `N=16,B=1024` 14800, 10600;
`N=256,B=64` 4000, 2800 (exp 5).

**P21a failed**: a readout frozen at O(1) norm is rescued by `B=1024` exactly as the free readout is (and slightly earlier). **P21b failed**: a
readout frozen at 0.1× its initial norm does not rescue `N=16, B=64`. **P21c held**: the free protocol reproduces exp 5 (`3.7×`, `3.8×` later
than `N=256` for the two paired seeds). **Kill criterion met** (P21a and P21b both fail): the readout–context mechanism of Prop. 3 does not
transfer to this transformer even qualitatively through the final linear readout. Reading: in the transformer the final readout multiplies the
residual stream, not the context statistic; the quantity playing Γ's role (if any) sits inside the attention value/output path, which these
protocols did not control. As pre-registered, the paper keeps Sec. 5 as observations only and states the negative result.

Deviations (recorded by the runner): free cells reused from exp 5 rather than re-run (except `N=16,B=64`, re-run); one `pinned_small N=256` seed
killed at 17k steps to afford a lower-priority cell (censored, not a failure); early stop at MSE < 0.25, so `T_e(0.1)` is not recorded.

## (superseded) Secondary question (C4) — to be pre-registered after exp 1 passes K1

Additive composition in a linear-in-label ICL model: emergence at `max(T_p, T_q)` with no compositional
data; thresholded accuracy multiplies. Default: compositional skills need compositional data or extra
depth (Okawa; He et al.). Rival: for additive targets the composition is free in any linear-in-label
readout, and the "multiplicative emergence" observed empirically is the additive structure seen through
a thresholded metric. Predictions and kill criteria to be written once the single-skill dynamics (exp 1)
are confirmed.

## Exp 1 outcome (recorded 2026-10-08 after the run; full tables in `results/exp1/README.md`)

- **K1 passed.** Monte-Carlo drift vs closed form (2.1) incl. the `V'/N` term: all 24 points within 2.3 SE, within ~3% at
  `m = 0.2, 0.4` for A/B, `k = 1,2,3`. For A `k=3`, `γ=0.1`, `N=128` the drift is *negative* below `m ≈ 0.25`
  (population-level confirmation of the repulsion, C2).
- **P1 met.** A `k=2` MC drift slope `3.14 ± 0.07` (pred. 3; formula's own slope 3.48 because of the repulsion at small `m`).
- **P6 met, with a sharper statistic than pre-registered.** Regressing `log(T_0.5·η)` on `log|m_0|` (escape from
  `m_0` scales as `m_0^{−(κ−2)}` for `κ>2`, `log(1/m_0)` for `κ=2`): A `k=2`: slope `−2.19 ± 0.08` ⇒ `κ_eff ≈ 4.2`
  (lower bound: 20 censored runs bias it toward 0); B `k=2`: `−0.57 ± 0.06` (log-like, `κ=2`); A `k=1`: `−0.50 ± 0.06`
  (log-like, `κ=2`); B `k=1`: `−0.13 ± 0.02` (`κ=1`). The pre-registered `d`-slopes: A `k=2` `3.1–3.5` vs B `k=2` `2.3–2.5`
  (difference `0.9 ± 0.5` on medians). Almost all `d`-dependence is through `m_0 = d^{−1/2}` (conditional `d` slope `0.27 ± 0.09`).
- **One pre-registered prediction was wrong, by my error:** B `k=1` was predicted `d^{1.5}`; measured `d^2` for `d ≥ 16`.
  The formula `T ~ m_0^{−(κ−2)}/η` does not apply for `κ < 2` (drift is `O(1)`, so `T ~ 1/η = d²/η_0`). Corrected table:
  κ=1 → `d²`, κ=2 → `d² log d`, κ=4 → `d³`. The A-vs-B comparison is unaffected.
- **Budget:** 6.4 CPU-h vs ~2 planned (censored A `k=2` runs at `d ≥ 48`). Exp 1b uses a fixed `m_0 = d^{−1/2}` initialisation
  to remove the `m_0` scatter and caps `d` at 48 for trained-readout runs.

## Exp 1b outcome (recorded 2026-10-08 after the run; `results/exp1b/README.md`)

- **P2 (fixed `Γ=1, N=128`): partially met.** `d=16` escapes 3/3, `d=64` stuck 0/3 (K2 not triggered), but `d=32` escaped 1/3
  (pre-registered: none). Across the grid, 4 of the 7 cells the population formula calls "stuck" had ≥1 late escape, two of them on
  the boundary `d = d*`. **Correction:** the drift-stable origin is a *soft* threshold under SGD — escape by noise activation over the
  unstable point `m* ≈ 2√(γ/N)`, with escape probability falling with `d`. The hard-threshold wording is withdrawn.
- **P3 (`γ=0.1`): met**, 27/27.
- **P7 (N–T tradeoff vs Oko et al. Remark 3): met.** `γ=1, N=128, d=64`: 0/9 escapes at `B ∈ {32, 256, 1024}` (total samples ×1, ×2.7,
  ×3.2; the ×10 leg not run), and the final `|m|` (0.005–0.015 at `B ≤ 256`) is *below* `m_0 = 0.125`: more tasks per step do not
  substitute for context length; the model is driven toward `m=0`.
- **P4 / P4′ (`κ_eff` from the `m_0` regression at `d=32`): direction met, magnitudes match the population ODE, not the asymptotic
  4→8.** Measured: free `η_Γ=η` 5.45 / 5.59 / 4.81 / 4.92 at `N = 32/128/512/4096` (N=32 biased low by censoring, ~5.9 if censored runs
  set to cap); free `10η` 6.61 / 5.78 / 4.92 / 5.17; **tied 4.10 / 4.26 / 4.30 / 4.44** (no `N` trend, as pre-registered).
  **Post-hoc but parameter-free check** (`scripts/ode_kappa_exp1b.py`, same equations committed before the run): the population ODE at
  the *exact* experimental `d`, `m_0` grid, `Γ_0`, `η_Γ/η` gives free `η` 6.47 / 5.46 / 4.99 / 4.85, free `10η` 7.02 / 6.18 / 5.32 / 4.92,
  tied 4.18 / 4.15 / 4.15 / 4.15 — within ≈0.2–0.5 of every measured cell (N=32 cells within the censoring bias). The pre-registered
  "4 → 8" is the `d → ∞` asymptote; at `d=32` the same theory predicts 4.85 → 6.5–7.0, which is what SGD shows.
- **P5 (learned `Γ ≈ Γ*(0.5;N)`): not met in magnitude** — `Γ` at `T_0.5` is 0.18–0.25 (`η_Γ=η`), below `Γ*` = 0.38–0.99; the
  fast-equilibration assumption behind P5 does not hold at these learning rates. The exponent predictions do not depend on it.
- **Priority 3 (`d`-slope bend between `N=128` and `N=2048`): not seen at `d ≤ 48`** (3.49 vs 3.70). The population ODE predicts the two
  `N` to differ by <15% in `T` for `d ≤ 48` and to separate only for `d ≳ 64–128`; the test was under-powered in the accessible range.
  Honest status: the `N`-dependence of the exponent is established via `κ_eff(N)` at fixed `d`, not via the `d`-slope.
- Budget 2.8 CPU-h (cap 3).

## Changelog
- 2026-10-08 v0: written before any exp 1 result was inspected (drift check and sweep running).
- 2026-10-09 v0.21: Theorems A/B, the joint-scaling proposition and the SGD-transfer conjecture (backlog N9) drafted in `paper/theorems.tex`
  with proofs; every closed form checked numerically in `scripts/verify_theorems.py` (61/61: Q_γ polynomial and γ₁ bound, k=2 closed forms,
  exact partial-fraction escape time vs LSODA, tied time change, ρ_trap, B-independence, Theorem B prefactor vs ODE at d=64–512, joint-limit
  corners). Two statements in the main text sharpened as a consequence: Prop. 2 is stated for γ < γ₁ = 2N/(N+9) (k=2) and Prop. 3(c) traps for
  ρ₀ > c_k ρ*(m₀) with c₂ = 4/3 (previously ρ₀ > ρ*). No prediction or outcome changes.
- 2026-10-09 v0.20: all ODE predictions recomputed with LSODA (`scripts/ode_lsoda.py`, `results/README_ode.md`): Euler overestimated times by
  0.6–3.6%; 46 of 101 quoted numbers change by >1% (none of the exponents, ratios or trap verdicts); paper updated. Also corrected: the exp-1b
  'd-scan within 15% for d ≤ 48' claim holds only to d=32 (20% at d=48).
- 2026-10-09 v0.19: review 3 (`docs/review_adversarial_3.md`, 5/10 theory persona) acted on: Prop. 3(c) corrected (flow traps permanently above
  `ρ* = 2Nm²/(4+(N+5)m²) ≈ Nm²/2`; the E5 "delay" is an SGD-noise effect) — all `Nm²/4` constants were leftovers of the factor-2 slip; tied =
  pinned/ρ identity added; `L_A = L_B[g→g²]` replaced by the correct statement (dependence through `g²`; drift coefficient `2Γ(2−Γg(1))`);
  Prop. 2 condition `γ < 2N/(N+5)` and general-`k` threshold (`β₂² = 2k²`, verified by quadrature); exponent defined as a secant with the
  limit structure (`4k*` at fixed `N`, `d→∞`; joint scaling `λ=N/d^{k*}`); "4→8 asymptotes" withdrawn; duplicate paragraph removed.
- 2026-10-09 v0.18: P21 (exp 8, transformer trap-vs-tradeoff) pre-registered.
- 2026-10-09 v0.17: exp 7 outcome recorded — P20d failed as worded (tied: delay, not trap); P20a–c,e,f held (a: 24% slow at N=16).
- 2026-10-09 v0.16: review 2 (`docs/review_adversarial_2.md`, 6/10) received; P20a–f pre-registered with ODE values before the run.
- 2026-10-09 v0.15: exp 6 outcome recorded — P19a/b/d held (ODE within 1–4%), P19c flat but outside the literal 5% band.
- 2026-10-09 v0.14: P19 (exp 6: N–T exchange by readout protocol) pre-registered.
- 2026-10-09 v0.13: adversarial review (`docs/review_adversarial.md`, score 5/10) acted on: (a) "12/12 within 0.5" corrected to 11/12 within
  0.6 and one censored cell at 1.0; post-hoc ODE evaluation now disclosed in the paper; (b) the N–T non-substitutability claim restricted
  to pinned readouts — with a free readout the ODE gives `T·N ≈ const` in the noise-dominated regime (Oko Remark 3 recovered); (c) omitted
  adverse evidence (P5 fail, d-scan no N-dependence, P4′ `ρ_0=1, d=32` escape) put in the paper; (d) many-skill/composition results
  relabelled as corollaries/identities; (e) transformer readout-collapse "signature" withdrawn (identical in stuck and emerging runs);
  (f) abstract `m*` formula and `Γ*` `g(1)` factor fixed; (g) related work: Ren Lemma B.1, Gu eq. 297 and `L=Θ(d)`, Nishikawa caveat, He
  et al. recast, BAEVW25 (check pending).
- 2026-10-09 v0.12: exp 2 outcome recorded — P16, P17 held; P18 held for A, failed for B (collective-fit baseline); B-readout deviation declared.
- 2026-10-09 v0.11: P16–P18 (exp 2, many-skill scaling) pre-registered before the run.
- 2026-10-09 v0.10: exp 5 outcome recorded — P11b: both rival and default failed; intermediate (≈3× handicap, not a trap); scope fixed.
- 2026-10-08 v0.9: exp 3 outcome recorded; two pre-registration design errors acknowledged (floor, estimator); C4 scope narrowed.
- 2026-10-08 v0.8: **algebra correction** — the stability threshold used in P2/P3 (`m² < 16γ/(N(4−2γ))`, `d* = N(4−2γ)/(16γ)`) had a
  factor-2 slip in the hand derivative of (2.1); correct: `m*² = 16γ/(N(8−4γ(1−1/N)−24γ/N))`, `d* ≈ 31` for `γ=1, N=128` (was 16),
  `d* ≈ 7` for `γ=1, N=32` (was 4), `d* ≈ 150–2400` for `γ=0.1`. The exp 1b data were not changed; the corrected threshold places the
  `d=32` cell on the boundary (1/3 escapes) and `d=16` below it (3/3), i.e. the data fit the corrected formula *better*. P2's
  pre-registered wording ("stuck for `d ≥ 32`") was derived from the wrong formula and is withdrawn on those grounds as well; the
  `threshold_grid.png` curve in `results/exp1b/figs` is drawn with the old formula and must be regenerated before any write-up.
- 2026-10-08 v0.7: P11a–c (exp 5, regime-matched transformer) and P12–P15 (exp 3, composition) pre-registered before the runs.
- 2026-10-08 v0.6: exp 4 outcome recorded — P8 failed (kill condition met); scope of C2/C3 restricted; P11 regime identified but not yet pre-registered.
- 2026-10-08 v0.5: P8–P10 (transformer transfer test) pre-registered before exp 4 runs.
- 2026-10-08 v0.4: exp 1b outcome recorded; hard-threshold wording withdrawn (soft, noise-activated); post-hoc parameter-free ODE check declared as such.
- 2026-10-08 v0.3: exp 1 outcome recorded; κ=1 prediction corrected (error acknowledged, not reinterpreted).
- 2026-10-08 v0.2: added P4′ after the Ren et al. full-text check revealed their 2-homogeneous parameterisation; the
  tied-readout population ODE (`scripts/ode_tied.py`) was run *before* writing P4′ and gave slope 1.03–1.05 (flow time)
  at all `N` — so P4′ is a population-level prediction awaiting the SGD test, not a blind one. Scope of C3 narrowed to
  free readouts.
- 2026-10-08 v0.1: added P7 after reading Oko et al. 2024 Thm 1 / Remark 3 in full (their `N_1 T_1` multiplicative
  tradeoff is now the named default hypothesis). No exp 1 result inspected beyond 14/230 partial rows of
  `summary.csv` (T_0.5 values only).
