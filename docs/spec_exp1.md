# Experiment 1 spec: does in-context learning double the information exponent?

## Setting

- Inputs `x ~ N(0, I_d)`.
- One teacher direction `v ∈ S^{d-1}` (unit vector, fixed per run; `P = 1` for this experiment).
- Activation `σ_k = He_k / sqrt(k!)`, the normalized probabilists' Hermite polynomial, so
  `E[σ_k(z)^2] = 1` for `z ~ N(0,1)` and the information exponent is exactly `k`.
  Support `k ∈ {1, 2, 3}` and also `relu` (centered: `relu(z) - 1/sqrt(2π)`).
- A **task** is a scalar `c ~ N(0, 1)` (mean zero). Given a task, labels are `y = c · σ_k(⟨v, x⟩)`.
- A **prompt** = `N` context pairs `(x_i, y_i)` from one task plus a query `(x_q, y_q)` from the same task.

## Two models, one student neuron (`M = 1`), weight `w ∈ S^{d-1}`

Both are trained by **online spherical SGD**: fresh data at every step, batch size `B`,
after each step re-normalize `w ← w / ||w||`. Square loss on the query.

**Model A — in-context (linear-in-label readout).**
```
A(w; prompt) = (1/N) Σ_i y_i σ_k(⟨w, x_i⟩)          # context statistic
ŷ_q         = Γ · σ_k(⟨w, x_q⟩) · A(w; prompt)
```
`Γ` is a scalar readout. Default: **fixed `Γ = γ = 0.1`** (flag `--gamma`, flag `--train-gamma` makes it
learnable). Why small: with finite context `N` the statistic `A` is noisy and its variance *grows* with
alignment, which adds a repulsive force of relative size `~ 2γ/(N m²)` near `m ≈ 0` for `k = 2`;
we need `γ ≪ N m_0² = N/d` for the aligned direction to be attractive at initialisation.
With `N = 128`, `γ = 0.1` this holds for every `d ≤ 256` in the sweep.

Population-loss formulas the code should reproduce (write them in `drift.py` docstring):
```
g(m)  = Σ_k α_k² m^k                       # for σ = σ_k: g(m) = m^k
V(m)  = E[σ(⟨v,x⟩)² σ(⟨w,x⟩)²]             # for σ_2: V = 1 + 8m² + 6m⁴ ; for σ_1: V = 1 + 2m²
L_B(m) = 1 - 2 a g(m) + a² g(1)
L_A(m) = s² [ 1 - 2γ g(m)² + γ² g(1) ( (1 - 1/N) g(m)² + V(m)/N ) ]      (s² = E[c²] = 1)
```
`drift.py` must compare the Monte-Carlo drift with `-dL/dm` from these formulas (not just the slope).

**Model B — in-weight baseline (Ren et al. setting, no context).**
Data: single fixed task `c = 1`, so `y = σ_k(⟨v, x⟩)`; model `ŷ = a · σ_k(⟨w, x⟩)` with fixed `a = 1`
(flag `--train-a`). Trained on single `(x, y)` samples with batch size `B`.

Both models are initialised with `w ~ Unif(S^{d-1})` **conditioned on `⟨w, v⟩ > 0`** (flip sign if
negative) so the alignment starts at `m_0 ≈ +d^{-1/2}`.

## What to log

Every `log_every` steps: `t`, `m = ⟨w, v⟩`, running train loss. Stop at `m ≥ 0.9` or at `max_steps`.
Record the **emergence time** `T_0.5 = first t with m ≥ 0.5` (and `T_0.9`).

Also implement the **population gradient check** (used to validate the theory, cheap):
`drift(m) = -E[∂L/∂w] · v` restricted to the tangent space, estimated by Monte Carlo at
`m ∈ {0.05, 0.1, 0.2, 0.4}` for each model and `k`. Output a CSV and a log-log plot of `drift(m)` vs `m`;
fit the slope. (Prediction: slope `k-1` for Model B, `2k-1` for Model A.)

## Sweep

- `k ∈ {1, 2}` (and `3` only if runtime allows).
- `d ∈ {8, 16, 32, 64, 128}` for Model B; for Model A `k=2` use `d ∈ {8, 16, 24, 32, 48}` (it's expected
  to scale like `d^3`); cap `max_steps` at `2e6`.
- `N = 128` context length, `B = 32` prompts per step.
- Learning rate **`η = η_0 / d²`** for *all* models (uniform, conservative; the noise analysis says
  escape from `m_0 ~ d^{-1/2}` needs `η ≲ d^{-κ/2}` where `κ` is the effective exponent, and the
  largest `κ` in the sweep is 4). Use `η_0 ∈ {2, 1}` (two values so we can check that `T_0.5 · η` is
  roughly invariant, i.e. we are in the drift-dominated regime). If `T_0.5·η` differs by more than 30%
  between the two `η_0`, note it and also run `η_0 = 0.5`.
- Under `η ∝ d^{-2}` the predicted step counts are `T ~ d^{(κ-2)/2} / η ~ d^{κ/2+1}`
  (κ=1: d^1.5, κ=2: d^2·polylog, κ=4: d^3). Put these in the README table instead of the earlier ones.
- 5 seeds per config. Report median and inter-quartile range of `T_0.5`.

Fit `log T_0.5` vs `log d` (median across seeds) per (model, k, η_0); report the slope with standard error.

## Deliverables (keep it simple, pure PyTorch, CPU, no frameworks)

```
icl_additive/
  __init__.py
  hermite.py         # normalized He_k, relu, vectorized; unit test: E[σ_k^2]=1, E[σ_j σ_k]=0 (MC)
  data.py            # sample_prompts(d, N, B, v, k, task_std) -> x_ctx, y_ctx, x_q, y_q ; sample_iw(...)
  models.py          # ModelA (ICL), ModelB (in-weight); both expose w, forward, loss
  train.py           # online spherical SGD loop; returns a dict/record of trajectory + T_0.5, T_0.9
  drift.py           # Monte Carlo population drift vs m for both models
  sweep.py           # CLI: runs the sweep, writes results/exp1/raw.csv (one row per seed/config/log step
                     # is too big -> write results/exp1/summary.csv with T_0.5,T_0.9 per run and
                     # results/exp1/traj/*.npz for trajectories)
  plot.py            # (1) T_0.5 vs d log-log per model/k with fitted slopes; (2) drift vs m log-log;
                     # (3) example m(t) trajectories. PNGs in results/exp1/figs/
tests/test_basic.py  # hermite normalization; at m=1 Model A has ~zero loss for large N; shapes
```

Rules:
- Use `torch.manual_seed`/`np.random.default_rng` with explicit seeds; store the seed in every record.
- `torch.set_num_threads(4)`; everything float32 except the Hermite normalisation checks (float64).
- No autograd tricks needed: autograd on `w` is fine. Project the gradient onto the tangent space is
  **not** required (renormalisation after the step is enough), but do log `||grad||`.
- Make `sweep.py` resumable (skip configs whose summary rows already exist).
- Keep the total CPU budget of the first full sweep under ~2 hours. If `k=2` Model A at `d=48` won't finish,
  drop it and say so in the summary.
- Write a short `results/exp1/README.md` with the fitted slopes table and what was/wasn't run.

## Predictions to compare against (write them into the README next to the measured slopes)

| model | k | predicted effective exponent κ | predicted T_0.5 scaling under η = η_0/d² |
|---|---|---|---|
| B (in-weight) | 1 | 1 | d^1.5 |
| B | 2 | 2 | d^2 · polylog (slope ≈2.1–2.3 over this range) |
| A (in-context) | 1 | 2 | d^2 · polylog |
| A | 2 | 4 | d^3 (slope ≈3) |

The headline comparison is **A vs B at k = 2**: same teacher, same data distribution, the only
difference is whether the label enters through the context; predicted slopes 3 vs ≈2.
