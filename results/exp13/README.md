# Experiment 13: softmax attention that learns (distance-based / rbf logits) in the single-feature model, sigma_2

Pre-registration: `docs/preregistration.md`, section "Softmax attention that learns: distance-based logits (exp 13, backlog N19)" (P26a-b; committed in cd61744 before any run).
Spec: `docs/spec_exp13.md`. Drift table used for the design: `drift_mc.txt` (`scripts/softmax_drift.py`).
Code: `icl_additive/ablation_exp13.py` (torch fp32, CPU, 1 thread per process, 4 workers with a claim-file job queue, N=1024 SGD claimed first), analysis `icl_additive/analyze_exp13.py`.
Raw rows: `runs.csv` (12 runs; `max_dm_step` = largest single-step |Delta m|, `censored_by` = `cap` or `killed_<step>_cpu_budget`, empty if the run reached |m| >= 0.5), per-run traces `traces/*.npz`
(m every 100 steps; gitignored), logs `logs/worker*.log` and `logs/sanity.log`, `rows/*.json`, `sanity.json`, `table.md`, `verdicts.json`.
**CPU time:** 2.738 h for the 12 runs (cap 3 h; not exceeded; no run was killed for budget). The sanity check (116 s) and a 200-step smoke test are not counted.

## Setup (as pre-registered)

d = 64, sigma_2, `m_0 = d^-1/2 = 0.125` exactly (`init_fixed` semantics), pinned readout Gamma = 1, `x ~ N(0,I)`, `c ~ N(0,1)` per prompt, seeds {0, 1}.
Model: `yhat = Gamma sum_i softmax_i(-beta (s_i - s_q)^2) y_i`, `s = sigma_2(w.x)`, `beta = 0.3`, `Gamma = 1`. Loss = batch mean of `(yhat - y_q)^2`; `w <- w/|w|` after every step.
`sgd`: eta = 1/d^2 = 1/4096, no momentum, cap 5e5 steps. `adam`: `torch.optim.Adam(lr=1e-3)`, default betas, cap 1e5 steps. **Full d=64 data for both optimisers** (no projected sampler).
Stop at `|m| >= 0.5` (checked every step); `m` logged every 100 steps. Cells: {sgd, adam} x (N,B) in {(16,64), (16,1024), (1024,64)} x 2 seeds = 12 runs.

## Deviations

- None from the spec in the model, data, caps, cells or seeds. The SGD gradient is computed by the chain rule through `z = X w` (`C = dL/dz` by autograd, then `g = C^T X`) instead of calling autograd through the matvec; the two agree to 7e-7 / 3e-6 (relative, N=16 / N=1024, fp32; `sanity.json`).
- Data streams: `torch.Generator` seeded from `SeedSequence([seed, 13, opt, N, B])`; teacher `v` and `w_0` from numpy `init_fixed` seeded by `[seed, 64, 13]` (same `v`, `w_0` for all cells of a seed).
- Float32 for training. `runs.csv`: `lr_adam` is NaN for SGD.
- The reading-rule counting of the pre-registration ("`N=1024` escapes 2/2"; "`N=16` stuck ... (0/4)") is applied cell by cell as written; the table also gives per-seed outcomes.

## Sanity check (before the sweep)

Monte-Carlo population drift `-dL/dm` of the torch model of `ablation_exp13.py` at `m = 0.25`, `N = 1024`, `B = 4096` (chunks of 1024, 48 independent repeats; `v = e_0`, `w = m e_0 + sqrt(1-m^2) e_1`,
gradient projected on `t = (v - m w)/sqrt(1-m^2)` and divided by `sqrt(1-m^2)`, exactly as `scripts/softmax_drift.py`):

**drift = +3.43e-2 +- 3.7e-3 (s.e.)**, reference table `+3.25e-2 +- 4e-3`; difference +1.8e-3 = 0.33 combined s.e. Required: positive and within 2 s.e. of the table. **Met.**

## Table: per cell (2 seeds each)

CENS = no `|m| >= 0.5` within the step cap. `reached` = Y/N per seed (seed 0 / seed 1). `max |m|` includes the initial value 0.125. `T_0.5` and `steps` are optimiser steps.

| opt | N | B | steps (seed0 / seed1) | T_0.5 (seed0 / seed1) | reached | final \|m\| | max \|m\| | max step \|dm\| | CPU s | censored_by |
|---|---|---|---|---|---|---|---|---|---|---|
| sgd | 16 | 64 | 500000 / 500000 | CENS / CENS | N/N | 0.012 / 0.013 | 0.146 / 0.126 | 0.0182 / 0.0242 | 383 / 378 | cap / cap |
| sgd | 16 | 1024 | 500000 / 500000 | CENS / CENS | N/N | 0.004 / 0.002 | 0.125 / 0.125 | 0.0021 / 0.0023 | 2819 / 2800 | cap / cap |
| sgd | 1024 | 64 | 63865 / 65759 | 63865 / 65759 | Y/Y | 0.500 / 0.500 | 0.500 / 0.500 | 0.0033 / 0.0031 | 1487 / 1299 | - / - |
| adam | 16 | 64 | 26348 / 24709 | 26348 / 24709 | Y/Y | 0.500 / 0.500 | 0.500 / 0.500 | 0.0038 / 0.0049 | 26 / 24 | - / - |
| adam | 16 | 1024 | 3561 / 100000 | 3561 / CENS | Y/N | 0.501 / 0.022 | 0.501 / 0.131 | 0.0035 / 0.0052 | 22 / 583 | - / cap |
| adam | 1024 | 64 | 741 / 697 | 741 / 697 | Y/Y | 0.500 / 0.503 | 0.500 / 0.503 | 0.0037 / 0.0044 | 19 / 19 | - / - |

SGD, N=1024: `m` rises monotonically from 0.125 (m at 10k / 20k / 30k / 40k / 50k steps: 0.135 / 0.143 / 0.160 / 0.186 / 0.231 for seed 0, 0.128 / 0.139 / 0.155 / 0.175 / 0.219 for seed 1) with an accelerating,
flow-like profile, i.e. the positive drift of `drift_mc.txt` at `m >= 0.125` is realised. SGD, N=16 (both B): `|m|` decays from 0.125 towards 0 (mean |m| over the second half of the trace 0.003-0.011) and never exceeds 0.146.
Adam, N=16, B=1024 seed 1 is a noisy-stable trap (mean |m| over the second half 0.029, max 0.131); no instability anywhere (max single-step |dm| <= 0.0052 for Adam).

## Verdicts

Reading rule (pre-registered, "as exp 12"): a cell "escapes" if >= 1 of 2 seeds reaches `|m| >= 0.5` within the step cap. The prediction rows require `N=1024` 2/2 and `N=16` 0/4.

**P26a (rbf + SGD). HELD.** Criterion (quoted): "`N=1024` escapes 2/2 within `5*10^5`; `N=16` stuck at `B=64` **and** `B=1024` (0/4)"; default "`N=1024` stuck (variant still uninformative) or `N=16` rescued by `B=1024` (softmax is the ingredient)".
Measured: N=1024 B=64 escapes 2/2 (T_0.5 = 63865, 65759 steps, 8x below the cap); N=16 B=64 0/2 and B=1024 0/2 (0/4 total, both run to the full 5e5 cap). Neither default is triggered. The rbf softmax therefore reproduces the linear model's SGD pattern (exp 12 P25a), with the
escape about 1.8x later than the linear model's (34517 / 37356 steps), consistent with the smaller drift. The variant is informative (it learns at N=1024).

**P26b (rbf + Adam). FAILED on the N=16 leg; the N=1024 leg held.** Criterion (quoted): "same pattern within `10^5`" (N=1024 escapes 2/2; N=16 stuck at both B, 0/4); default "same" (i.e. N=1024 stuck or N=16 rescued).
Measured: N=1024 B=64 escapes 2/2 (T_0.5 = 741, 697). But **N=16 escapes**: B=64 2/2 (T_0.5 = 26348, 24709) and B=1024 1/2 (T_0.5 = 3561; the other seed stuck at 1e5, final |m| 0.022). That is 3/4 of the N=16 runs, not 0/4, and in both N=16 cells at least one seed escapes.
Under the reading rule ("if `N=16` is rescued by `B=1024` with the softmax but not with the linear statistic (exp 12), softmax is implicated"): the N=16, B=1024 cell escapes (1/2) here while linear+Adam did not (exp 12: 0/2 at B=1024, 0/2 at B=64), so **the rule's condition is met: with Adam, the distance-based softmax is implicated** in escaping at N=16.
Two qualifications that I state rather than interpret away: (i) the escape under Adam does not need the larger batch: N=16 escapes at B=64 (2/2) at least as reliably as at B=1024 (1/2), so what is observed is a softmax+Adam effect at N=16, not a B-rescue specifically; (ii) with 2 seeds and one seed stuck at B=1024, the B=1024 evidence is weak on its own.
The pattern "SGD pins N=16, Adam does not" is also an optimiser-by-softmax interaction: the same Adam with the linear statistic stays trapped at N=16 (exp 12), and the same softmax with SGD stays trapped at N=16 (P26a).

Combined reading (per the pre-registered reading rule): the N=1024 variant is informative under both optimisers (it escapes), so the softmax question is not left open. With SGD, softmax is excluded as an ingredient for the N=16 trap-escape (P26a held).
With Adam, softmax is implicated (P26b failed). No claim about the transformer itself follows from this single-feature model.

## Summary

Sanity drift +3.43e-2 +- 3.7e-3 (table +3.25e-2 +- 4e-3). P26a held (SGD: N=1024 escapes 2/2 at T_0.5 = 63865 / 65759; N=16 stuck 0/4 at the 5e5 cap). P26b failed on the N=16 leg (Adam: N=1024 2/2 at 741 / 697; N=16 B=64 2/2 at 26348 / 24709; N=16 B=1024 1/2 at 3561),
so under the reading rule softmax is implicated with Adam. 12 runs, CPU 2.738 h (cap 3 h), no run killed for budget; 7 of 12 runs escaped.
