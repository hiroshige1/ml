# Experiment 1b spec: context length controls the emergence exponent (trainable readout)

Depends on the code from `spec_exp1.md` (Model A, `k = 2`, `M = P = 1`). Theory: `docs/theory_notes.md` §2.2–2.3.

## Predictions to test

1. **Fixed readout `Γ = γ`: a hard context-length threshold.** The aligned direction is attractive at
   initialisation iff `m_0² ≳ 16γ/(N(4−2γ))`, i.e. roughly `d ≲ N/(4γ)`. Below the threshold runs
   escape; above it they stay at `m ≈ m_0` for the whole budget (SGD noise may still let them escape
   eventually — record that too).
2. **Trainable readout: a crossover in the `T_0.5 ~ d^s` slope.** With `η = η_0/d²` the predicted
   step-count slope is `≈ 2` (gradient-flow slope 1 from `κ_eff = 4`, plus the `1/η` factor) for
   `d ≪ √N`, bending towards **`4`** (gradient-flow slope 3 from `κ_eff = 8`) for `d ≫ √N`.
   The learned `Γ` at `T_0.5` should be `≈ Γ*(0.5)`, independent of `d`.
   Note the SGD-noise condition is now `η ≲ d^{−4}` in the far regime; with `η_0/d²` the largest `d`
   may become noise-dominated — detect this by the `η_0` invariance check and, if needed, add
   `η = η_0/d³` runs for the two largest `d`.

## Runs

- `k = 2`, `N ∈ {32, 128, 512, 2048}`, `d ∈ {8, 16, 32, 64}` (add `128` where the budget allows).
- (a) fixed `γ ∈ {0.1, 1.0}`; (b) trainable `Γ`, `Γ_0 = 0.01`, readout learning rate `η_Γ = η`
  (same as features) and also `η_Γ = 10η`.
- `η_0 ∈ {2, 1}`, `B = 32`, 3 seeds, `max_steps = 2e6`. Context `N = 2048` is expensive: use `B = 8`
  there and note it.
- Log `Γ(t)` alongside `m(t)`.

## Outputs

- `results/exp1b/summary.csv`, `results/exp1b/README.md`.
- `figs/threshold_fixed_gamma.png`: for each `(N, γ)` the fraction of seeds that escaped vs `d`, with the
  predicted threshold `d* = N(4−2γ)/(16γ)` marked.
- `figs/slope_vs_N_trainable.png`: fitted slope of `log T_0.5` vs `log d` as a function of `N`
  (one line per `η_Γ`), with the two predicted asymptotes (2 and 4) as horizontal lines.
- `figs/gamma_at_T.png`: learned `Γ` at `T_0.5` vs `d`, per `N`, with `Γ*(0.5; N)` overlaid.

## Revision after exp 1 (2026-10-08) — binding for the run

- **Initialisation:** set `m_0 = +d^{−1/2}` *exactly* (w = m_0 v + sqrt(1−m_0²) u, u ⟂ v random) in every run; exp 1 showed
  that the `m_0` scatter of a uniform-sphere init dominates the `d`-scatter and causes censoring. Keep a `--init uniform`
  flag for comparability but do not use it in the sweep.
- **Primary statistic:** besides `T_0.5` vs `d`, also regress `log(T_0.5·η)` on `log m_0` across a small set of
  deliberately chosen `m_0 ∈ {0.5, 0.7, 1.0, 1.4} × d^{−1/2}` at fixed `d = 32` (4 values × 3 seeds) for each protocol:
  the slope is `−(κ_eff − 2)`. This is cheaper and sharper than the `d`-sweep.
- **Models:** (i) Model A, fixed `Γ = γ`; (ii) Model A, free trainable scalar `Γ` (init `Γ_0`, lr `η_Γ`);
  (iii) **Model A-tied**: `Γ = ‖w‖²` with `w` *unnormalised* (no projection), features `σ(ŵ·x)` with `ŵ = w/‖w‖`,
  prediction `ŷ_q = ‖w‖² σ(ŵ·x_q) A(ŵ)`; init `‖w_0‖² = ρ_0`. Gradient: write it analytically (chain rule through `ŵ`),
  add it to the finite-difference test.
- **Priority order and budget (hard cap 3 CPU-hours total, 4 workers):**
  1. **P2/P3/P7 threshold tests** (cheapest: a run is "stuck" if `|m| < 2 m_0` after `3·10⁵` steps):
     fixed `γ ∈ {1.0, 0.1}`, `N ∈ {32, 128, 512}`, `d ∈ {16, 32, 64}`, 3 seeds, `η = 1/d²`; **P7:** `γ = 1, N = 128, d = 64`
     with `B ∈ {32, 256, 1024}` (steps `3·10⁵`, `10⁵`, `3·10⁴` so total samples differ ×1, ×3, ×10).
  2. **P4 vs P4′ (`m_0`-regression form)** at `d = 32`, `N ∈ {32, 128, 512, ∞→4096}`: free `Γ` (`Γ_0 = 0.01`, `η_Γ ∈ {η, 10η}`)
     and tied (`ρ_0 = 0.01`), 4 `m_0` values × 3 seeds, `η = 1/d²`, `max_steps 10⁶`. Report `κ_eff` per (protocol, N) with SE.
  3. Only if budget remains: `T_0.5` vs `d ∈ {16, 24, 32, 48}` for free-`Γ` at `N ∈ {128, 2048}`.
- **Pre-registered expectations** (`docs/preregistration.md` P2, P3, P4, P4′, P7): fixed `γ=1, N=128` stuck for `d ≥ 32`
  (and `d=16` escapes); `γ=0.1` escapes at all three `d`; P7 stuck at every `B`; free-`Γ` `κ_eff` rises from ≈4 (`N=4096`)
  toward ≈8 as `N` decreases (`N=32`); tied `κ_eff ≈ 4` at every `N`; with `ρ_0 = 1` tied is stuck like fixed `γ=1`.
- Write `results/exp1b/README.md` with the tables next to these expectations; state plainly what failed.
