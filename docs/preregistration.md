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
