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

## Secondary question (C4) — to be pre-registered after exp 1 passes K1

Additive composition in a linear-in-label ICL model: emergence at `max(T_p, T_q)` with no compositional
data; thresholded accuracy multiplies. Default: compositional skills need compositional data or extra
depth (Okawa; He et al.). Rival: for additive targets the composition is free in any linear-in-label
readout, and the "multiplicative emergence" observed empirically is the additive structure seen through
a thresholded metric. Predictions and kill criteria to be written once the single-skill dynamics (exp 1)
are confirmed.

## Changelog
- 2026-10-08 v0: written before any exp 1 result was inspected (drift check and sweep running).
- 2026-10-08 v0.1: added P7 after reading Oko et al. 2024 Thm 1 / Remark 3 in full (their `N_1 T_1` multiplicative
  tradeoff is now the named default hypothesis). No exp 1 result inspected beyond 14/230 partial rows of
  `summary.csv` (T_0.5 values only).
