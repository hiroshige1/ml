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

## Secondary question (C4) — to be pre-registered after exp 1 passes K1

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
- 2026-10-08 v0.4: exp 1b outcome recorded; hard-threshold wording withdrawn (soft, noise-activated); post-hoc parameter-free ODE check declared as such.
- 2026-10-08 v0.3: exp 1 outcome recorded; κ=1 prediction corrected (error acknowledged, not reinterpreted).
- 2026-10-08 v0.2: added P4′ after the Ren et al. full-text check revealed their 2-homogeneous parameterisation; the
  tied-readout population ODE (`scripts/ode_tied.py`) was run *before* writing P4′ and gave slope 1.03–1.05 (flow time)
  at all `N` — so P4′ is a population-level prediction awaiting the SGD test, not a blind one. Scope of C3 narrowed to
  free readouts.
- 2026-10-08 v0.1: added P7 after reading Oko et al. 2024 Thm 1 / Remark 3 in full (their `N_1 T_1` multiplicative
  tradeoff is now the named default hypothesis). No exp 1 result inspected beyond 14/230 partial rows of
  `summary.csv` (T_0.5 values only).
