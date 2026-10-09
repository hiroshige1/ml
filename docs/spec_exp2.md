# Experiment 2 spec: emergence and scaling law of the in-context additive model (many skills)

Theory: `docs/theory_notes.md` §3. Pre-registered as P16–P18 in `docs/preregistration.md` before the run. Code: extend
`icl_additive/multi.py` (M neurons, P teachers, fixed Γ) — exp 3 showed it decouples into single-neuron dynamics with rate ∝ π_p.
Budget: hard cap 4 CPU-hours, 4 workers.

## Setting
- `d = 32`, `σ_2` link (k*=2), `P = 16` orthonormal teachers; skill frequencies `π_p ∝ p^{−α}` with **`α = 1.5`** (and `α = 1.0` if
  budget allows); single-skill prompts (`c_p ~ N(0,1)`), `N = 128`, `B = 32`, `η = 1/d²`.
- **Model A (in-context):** `M = 64` neurons, fixed `Γ_j = 1/M·… ` — no: use `Γ_j = γ = 0.1` as in exp 3 (floor is then
  `(1 − 0.1 n_p)²` per skill; report the *alignment-based* emergence times, not loss thresholds, exactly because of exp 3's lesson).
  **Initialise each neuron at exactly `m_0 = d^{−1/2}` with its target chosen by a greedy max-selection over a random Gaussian draw**
  is NOT allowed (it would bake in the assignment) — use plain uniform init on the sphere, `M = 64 ≥ P log P`.
- **Model B (in-weight baseline):** same `M`, same teachers, fixed task `y = Σ_p a_p σ_2(v_p·x)` with `a_p ∝ p^{−α/2}` (so that
  `a_p² ∝ π_p`, matching the in-context signal weights), readout `a = 1` per neuron. (This is Ren et al.'s setting at `k*=2`,
  which their Assumption 2.1 excludes — we report it as an empirical baseline only.)
- Log every 1000 steps: `m_{jp}` for all `j, p`; per-skill "learned" indicator `max_j |m_{jp}| ≥ 0.5`; **excess in-context loss
  proxy** `ℓ(t) = Σ_p π_p · [1 − max_j m_{jp}²]²`-type is model-specific — instead compute the actual eval MSE on 2048 fresh
  single-skill prompts per skill (16 × 2048 is too many: use 512 per skill) and the frequency-weighted total `L(t) = Σ_p π_p MSE_p(t)`.
- 3 seeds per (model, α); `max_steps = 3·10⁶` (≈ enough for `p ≤ 8` at α = 1.5 given exp 3's `T ≈ 0.6/(η π_p m_0²)`:
  `T_p ≈ 0.6·d²·d/π_p = 0.6·32³/π_p ≈ 2·10⁴/π_p`; with `π_8 ≈ 0.011` → `1.8·10⁶` ✓; skills `p > 8` will not be learned within budget
  and that is fine — report which were).

## Pre-registered predictions (P16–P18)
- **P16 (per-skill emergence times scale with frequency):** `T_p ∝ 1/π_p` for Model A, i.e. `T_p ∝ p^{α}`; regression of `log T_p`
  on `log p` over learned skills gives slope `α ± 0.2` (α=1.5 → 1.5). For Model B the analogous slope is governed by `a_p` with the
  `k*=2` in-weight drift `∝ a_p m` ⇒ `T_p ∝ 1/a_p ∝ p^{α/2}`, slope `0.75 ± 0.2`. **The in-context model orders skills by frequency
  twice as steeply as the in-weight model** (because the in-context signal carries `s_p² = π_p`, the in-weight one `a_p`).
- **P17 (sharper steps):** the per-skill MSE curves of Model A have a longer plateau and a sharper drop than Model B's: define the
  sharpness as `(T_{p,0.9} − T_{p,0.1}) / T_{p,0.5}` (time between 10% and 90% of the drop, relative to the midpoint); predict Model A's
  median sharpness is **smaller** than Model B's by at least a factor 2.
- **P18 (scaling-law exponent):** the frequency-weighted total `L(t)` decays as a power law over the window where 3–8 skills have
  been learned; fitted exponent for Model A ≈ `(α−1)/α` (α=1.5 → 0.33) with the `d`-time-unit absorbed, and for Model B ≈
  `(α−1)/(α/2)`·… — no: for B with `T_p ∝ p^{α/2}` the exponent is `(α−1)/(α/2) = 0.67`. So **A decays slower in log-time than B**
  with exponent ratio ≈ 1/2. Tolerance ±0.15 on each exponent. Note: the loss floor `(1−0.1 n_p)²` biases these; also report the
  *alignment-based* loss `Σ_p π_p (1 − max_j m_{jp}²)` which has no floor, and fit the exponent on that too (declare which is primary:
  **alignment-based is primary**).
- Kill: if P16's slope for Model A is < 1.0 (not steeper than B), the "ICL orders skills by `π_p` not `√π_p`" claim is wrong.

## Deliverables
`icl_additive/sweep2.py`, `results/exp2/{summary.csv, per_skill.csv, README.md}`, `figs/{Tp_vs_p.png, curves_per_skill.png,
scaling_law.png}`. README with tables next to P16–P18, CPU time, deviations.
