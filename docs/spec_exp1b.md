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
