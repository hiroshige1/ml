# Empirical / methodology review (round 2)

I read body.tex, appendix.tex, results/exp*/, README_stats.md, README_ode.md, preregistration.md and the SGD/ODE code. I re-ran the tests and the figure script.

## 1. Is the ODE–SGD agreement a finding or a sanity check?

Mostly a sanity check. The SGD runs use a projected sampler (`icl_additive/train.py`). It is distribution-exact, but it shows that the dynamics are a Markov chain in about two dimensions: m, plus an isotropic perpendicular kick of size sqrt(S2). With η = 1/d² and B = 32–64, the omitted Itô/renormalisation drag (≈ η·d·E[S2]·m/2 in flow time) is a small, d-independent fraction of the drift. There, gradient flow *must* match SGD. The paper's own controls confirm it: step counts do not depend on B (E2); flow time is unchanged under η ∝ B (E5); η/4 moves the k*=3, d=8 cell from 1.18 to 1.007. So the 1–4% agreement in E2/E7 shows that Prop. 1 is right (already checked against Monte-Carlo drift) and that η is small. It is not independent support for the theory.

The ODE (`scripts/ode_kappa_exp1b.py`) has no noise term. The residuals are therefore this omitted, computable correction, and they grow where the drift is weak:
- E1 smallest m0: ratios 0.64–1.34, plus one censored cell ≥2.35× slow.
- k*=3: 0.80–1.18 (P22a and P22b failed).
- Tied ρ0=0.3: the flow traps, but SGD escapes after a 30× delay.

The tied readout's N-flatness is also weaker under SGD than in the flow every time it is measured: T(16)/T(256) = 1.21 vs 1.04 at d=64, 1.195 vs 1.09 at d=128, and it fails at k*=3.

**Which agreements are informative.**
- *The d=128 flip (pinned γ=0.1, N=16): prospective but weak.* m0/m* = 0.76, so the drift is negative at initialisation and censoring follows from the already-verified drift formula.
- *The free ratio 4.4: informative.* The ODE prospectively picks the growth-limited value rather than the adiabatic 16×.
- *Cells near the threshold: the most informative.* Examples are pinned γ=1, N=256, d=64 (3/6 escape) and the Kramers remark. Here noise decides, and the flow *fails*. This is the more interesting empirical result, and it is under-sold.
- *Asymptotic theory: untested.* The 4k* exponent and the 1/N law are out of reach. κ_eff is a secant of the fixed-d ODE, so it tests the ODE, not the exponent theory.

## 2. Statistics

**Re-computed (own code, from the CSVs):**

| quantity | paper | recomputed | verdict |
|---|---|---|---|
| E1 κ_eff, OLS and Tobit, 12 cells (own Tobit fit) | free1 N=32: 5.45 / 6.05; free10 N=128 Tobit: 6.67 | identical in all 12 cells | match |
| E1 ratios, 45 fully reached cells | [0.64, 1.34]; 24 within 10%, 35 within 20% | same | match |
| E2 free T(16)/T(256) | 4.4 | 4.42 (token-matched); 4.57 (B=64 scheme, not reported) | match |
| E7 median/ODE: pinned 0.1; tied | 0.99, 1.01; 1.12 / 0.97 / 1.02 | 0.989, 1.006; 1.118 / 0.969 / 1.017 | match |
| E6 tied median/ODE | 1.18 / 0.89 / 0.80 | same | match |

**Text errors.**
- Body E7 says "three seeds"; pinned γ=1 and free N=256 have **2 seeds** (the appendix is correct).
- E2 tied medians are quoted from the token-matched scheme only; the B=64 scheme gives 18.9k / 15.1k / 16.3k.
- E2 calls the ODE times "computed before the run", but 400k/193k/175k are post-hoc LSODA values. The pre-registered values were 414k/197k/176k. That does not change any verdict, but "within 1–2%" is then at integrator-error level.
- "The two batch schemes give the same step counts" is trivially true at N=64, where they are the same runs.

**Seeds.** Medians of 3 seeds have seed spreads of ±10–25%, so agreements "within 1%" are partly luck. Give min–max intervals in the body.

**Censoring and estimator choice.**
- The Tobit fit was adopted after OLS missed by 1.0.
- The 0.5 band was not pre-registered.
- The worst cell (free10, N=128) passes at 0.49, with a bimodal seed pattern and bootstrap SE 0.49.
- Fig. 1 plots OLS (censored runs dropped), while the abstract quotes Tobit.

**Forking paths in E1.** The primary pre-registered test P4 found no N-dependence (d-slopes 3.49 ± 0.08 vs 3.70 ± 0.21), the pattern kill criterion K3 describes. The N-dependence was then shown with a different estimator (κ_eff vs m0 at fixed d), with the ODE evaluated post hoc. All of this is disclosed, which is commendable, but E1 is a post-hoc check and should not lead the abstract.

**Confound.** E1 uses B=32 for N≤128 and B=8 for N≥512. Only N=128 has a B control.

**Failed predictions.** They are reported, which is a real strength. But several were re-worded into softer passes:
- P19c: "flat within 5%" became "×1.25".
- P4′: "ρ0=1 stuck for d≥32" became "d=32 escaped".
- P20d: "trap" became "delay".
- P22b: rescued by a theorem hypothesis "added after E6".

Each change is defensible. Together they tune the theory's stated conditions to exactly the cells where the noise-free flow fails.

## 3. Transformer section (Exps 4, 5, 8, 11, 16–19)

Every pre-registered prediction failed (P8, P9, P21a/b, P24a–c, P29, P30, P32). Cells have 2–3 seeds, some runs were censored at the CPU cap, and Adam was not rescaled with B. The mechanism is "open".

"What is excluded is … the final readout, or the attention-output scale" is too strong for 2-seed nulls; "did not reproduce in two seeds" is what the data support. The section costs about a quarter of the body plus a figure and blurs the paper's focus.

**Cut it** to one paragraph:
- B-irrelevance does not carry over to a softmax transformer trained with Adam;
- two readout-scale interventions did not reproduce the trap.

Keep Exp. 15 (in the solvable model, B matters under Adam), which is clean. Everything else goes to the appendix.

## 4. Reproducibility

- `OMP_NUM_THREADS=1 python3 tests/test_basic.py`: 18/18 pass in 6 s, including finite-difference gradient checks and a check that the projected sampler matches full sampling.
- `make_figs.py`: runs in 9 s. All five PDFs differ from the committed ones in 8 bytes (timestamps only), and the PNGs are byte-identical. I restored them with `git checkout -- paper/figs`.

So the figures regenerate exactly, but **from cached CSVs**, and the ODE curves are hard-coded in `make_figs.py`. Re-running SGD looks feasible (a few CPU-hours per experiment) but was not tested. The repo link is still `\todo{repo}`.

## 5. Verdict

**5/10 for a top workshop.** The study is transparent, pre-registered, and its numbers reproduce exactly. But its empirical headline is the expected deterministic limit, and a large share of the body is a negative transformer study.

**Top 3 weaknesses, with the cheapest fix for each:**

1. **The ODE agreement is presented as a finding but is a consistency check.** The runs sit in the small η·noise regime where the flow must match. Deviations appear exactly where noise matters: soft threshold, trap→delay, k*=3, tied N-flatness.
   *Fix (no new runs):* reframe as validation. Report a noise-to-drift ratio η·d·E[S2]/drift(m0) per cell, or add the drag term to the ODE. Add a short "where the flow fails" paragraph. Optionally, run one η-sweep at a boundary cell.

2. **The E1 headline is post hoc and depends on the estimator.** P4 found no N-dependence; κ_eff, the ODE evaluation and the Tobit fit were chosen afterwards; the band is passed at 0.49; Fig. 1 shows OLS; B changes with N.
   *Fix:* lead with the prospective E2/E7. Label E1 post hoc. Plot Tobit estimates with bootstrap SEs in Fig. 1.

3. **Seed counts are thin and partly misreported, and the transformer claims over-reach.** E7 has 2 seeds in two cells; "excluded" rests on 2-seed nulls.
   *Fix:* correct the seed counts, and show min–max intervals in Fig. 2 and the text. Cut Sec. 5 to one paragraph with "did not reproduce (2 seeds)" wording.
