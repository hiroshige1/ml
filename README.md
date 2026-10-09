# In-context skill emergence in an additive model

Code, pre-registration documents, results and paper draft for the project *"Context length and readout parameterisation select
the emergence exponent of in-context skills in an additive model"*.

## Layout

| path | content |
|---|---|
| `paper/` | `main.tex`, `appendix.tex`, `refs.bib`, `figs/` (paper figures; `make_figs.py` regenerates them from `results/`) |
| `docs/theory_notes.md` | derivations (population loss, finite-context repulsion, readout regimes, Kramers remark) with corrections logged |
| `docs/preregistration.md` | every prediction (P1–P19) with the date it was written, the outcome, and a changelog of deviations |
| `docs/novelty_protocol.md`, `docs/fulltext_checks/` | what each cited paper actually contains (16 full-text reads) and the resulting claim-by-claim verdicts |
| `docs/findings.md` | the claims with evidence levels (T derived/checked, S SGD toy, X transformer, L literature) |
| `docs/review_adversarial*.md` | adversarial reviews commissioned on the draft and the responses |
| `docs/spec_exp*.md` | experiment specifications written before each run |
| `icl_additive/` | numpy implementation (analytic gradients, float64) of the single- and multi-neuron in-context model, the in-weight baselines, and the PyTorch transformer experiments |
| `scripts/` | population ODEs (`ode_two_timescale.py`, `ode_tied.py`, `ode_kappa_exp1b.py`) |
| `results/exp*/` | per-experiment `README.md`, `summary.csv`, figures; large `.npz` run files are not tracked |
| `reports/`, `research_notes/` | the methodology report on research-question formulation and AI-assisted science that shaped the protocol |

## Reproduce

```
pip install numpy scipy pandas matplotlib            # toy experiments
pip install torch                                     # transformer experiments (CPU is enough)
python3 -I scripts/ode_kappa_exp1b.py                 # population-ODE exponents at the exp-1b parameters
python3 -m icl_additive.sweep   --quick               # exp 1 smoke test
python3 -m icl_additive.sweep1b                       # exp 1b (threshold grid, P7, kappa_eff)
python3 -m icl_additive.sweep6                        # exp 6 (three regimes)
python3 -m icl_additive.sweep2 ; python3 -m icl_additive.sweep_btied   # exp 2 + fair in-weight baseline
python3 -m icl_additive.sweep3                        # exp 3 (composition)
python3 -I icl_additive/transformer_exp4.py --grid ; python3 -I icl_additive/transformer_exp5.py --grid
python3 -I paper/figs/make_figs.py
```

### Tests

Run from the repository root (about 15 s on one core; each script prints `ok <test name>` per check and exits non-zero on failure):

```
OMP_NUM_THREADS=1 python3 tests/test_basic.py && OMP_NUM_THREADS=1 python3 tests/test_many.py && OMP_NUM_THREADS=1 python3 tests/test_btied.py
```

`test_basic.py` checks the single- and multi-neuron model (finite-difference gradients, projected vs full sampler, closed-form population
loss, additivity in context labels); `test_many.py` the many-skill model of exp 2 (gradients, exact population MSE of the in-weight baseline);
`test_btied.py` the B-tied baseline. They need only `numpy`; no torch and no GPU.

Each sweep is resumable and writes `results/expN/summary.csv`; CPU budgets per experiment are recorded in the READMEs
(total ≈ 30 CPU-hours on 4 cores).

## Process

Predictions were written to `docs/preregistration.md` before each run; failures and the two algebra slips found afterwards are
recorded there rather than edited away. AI assistance (Claude) was used for literature search, implementation, derivation
checks, figure production and drafting; every derivation was checked against a Monte-Carlo drift computation and every citation
against the full text.
