# Paper map: from every figure, table and number family to the script and results file

Scope: `paper/body.tex`, `paper/appendix.tex`, `paper/theorems.tex` (the 4-page and ICML variants reuse the same figures and numbers).
All paths are relative to the repository root. Labels E1-E7 are the paragraph labels of `paper/body.tex`; "Exp. n" is the experiment number
used in `results/exp<n>/`, `docs/preregistration.md` and the appendix. "Pre-reg section" is the section heading in `docs/preregistration.md`
(each section has a matching "Exp n outcome" section further down). Nothing here was re-run to write this map; it was built from the code
headers, the READMEs and the quoted-number CSVs.

Conventions used below:

- "SGD data" = raw per-run rows written by a `sweep*.py` / `transformer_exp*.py` driver; "analysis" = the `analyze*.py` script that turns them into the CSV/README tables the paper quotes.
- `results/exp<n>/README.md` holds the prose tables from which most appendix numbers were copied; the CSVs next to it hold the same numbers in machine-readable form.
- `results/ode_lsoda_quoted.csv` has one row per ODE number quoted in the paper, pre-registration or notes (columns `source`, `quantity`, `quoted`, `euler_value`, `lsoda_value`). The `source` strings used for the paper are `paper Prop.3 discussion (after Prop. 3)`, `prereg P4 / paper E1 + App. (kappa_eff ODE)`, `prereg exp6 outcome / paper E2 / App. Exp.6`, `paper App. Exp.7 / E5`, `prereg P19 (exp6 table)` and `prereg P20 (exp7 table)`.

## 1. Figures (5 rows)

All five are made by `python3 -I paper/figs/make_figs.py [kappa|threshold|many|transformer|regimes]` (no argument = all five). The script reads `results/` and writes only `paper/figs/`. Per-figure caveats are in `paper/figs/README.md`.

| Figure label | File in `paper/figs/` | Function in `make_figs.py` | Input results files | Experiment (E-label) |
|---|---|---|---|---|
| `fig:kappa` | `fig_kappa.pdf`, `fig_kappa.png` | `fig_kappa()` | `results/exp1b/kappa_fits.csv` (columns `protocol`, `N`, `kappa`, `kappa_se`, `kappa_cap`, `n_censored`). The dashed ODE curves are hard-coded in `make_figs.py` (`ODE_N`, `ODE`) from `scripts/ode_kappa_exp1b.py`; `results/ode_lsoda_quoted.csv` (source `prereg P4 / paper E1 + App. (kappa_eff ODE)`) shows LSODA moves them by at most 0.004. | Exp. 1b = E1 |
| `fig:regimes` | `fig_regimes.pdf`, `fig_regimes.png` | `fig_regimes()` | `results/exp6/summary.csv` (columns `protocol`, `scheme`, `N`, `T05_seeds`, `T05_median`, `max_steps`, `n_reached`, `ode_steps`, `variant`; rows with `variant=ext` are the 2.5e6-step reruns of free N=16, which come from `results/exp6/runs_ext.csv`). The dashed ODE lines are hard-coded (`ODE6_N`, `ODE6`, LSODA values 400k/193k/175k, 1.819M/623k/409k, 17.0k/16.4k/16.3k); they are the pre-registered flow times x d^2 recomputed by `scripts/ode_lsoda.py` (`results/ode_lsoda_quoted.csv`, source `prereg exp6 outcome / paper E2 / App. Exp.6`). | Exp. 6 = E2 |
| `fig:threshold` | `fig_threshold.pdf`, `fig_threshold.png` | `fig_threshold()` | `results/exp1b/threshold_table.csv` (left: `gamma`, `N`, `d`, `escaped`, `n`); `results/exp1b/p7.csv` (right: `B`, `seed`, `final_abs_m`). The curve d*(N) is the closed form `dstar()` in `make_figs.py` (equal to column `pred_threshold_d_corrected`). | Exp. 1b (P2, P3, P7); cited in Remark "Soft under SGD" and Prop. 2 |
| `fig:many` | `fig_many.pdf`, `fig_many.png` | `fig_many()` | (a) `results/exp2/per_skill.csv` (tags `A_a1.5`, `B_a1.5_rho0.0496`, column `T05d`, rows with `learned` and `T05d>0`) and `results/exp2/btied/per_skill.csv` (tag `Btied_eta1`, drawn automatically when the file exists; it does now). (b) `results/exp3/main/traj/seed1.npz` (gitignored; keys `t`, `hmse_E1`, `hmse_E2`, `hmse_E12`, `hmse_Ex`) and `results/exp3/main/times.csv` (`T1`, `T2`). | (a) Exp. 2 = E3 (B-tied add-on); (b) Exp. 3 = E4 |
| `fig:transformer` | `fig_transformer.pdf`, `fig_transformer.png` | `fig_transformer()` | `results/exp5/curves/{a,c,d}_s{0,1,2}.npz` (gitignored; keys `steps`, `mse`, `norm_steps`, `norms`, optional `partial`) and `results/exp5/summary.csv` (`Te05` for the emergence marker). | Exp. 5 (Sec. 5, no E-label) |

## 2. Numbers by paragraph (31 rows)

"Producer" is the script that computes the number; "Where the quoted numbers live" is the file to open to audit it. "Pre-reg section / P-ids" is blank-dashed when the number is a derivation and was never pre-registered.

| # | Paper location and number family | Producer script(s) | Where the quoted numbers live | Pre-reg section / P-ids |
|---|---|---|---|---|
| 1 | Prop. 1 (population loss): Monte-Carlo drift check, 24 points within 2.3 s.e., within 3% for m >= 0.2 at k*<=2 (Sec. 3 and App. A "Monte-Carlo drift check") | `icl_additive/drift.py` (closed forms in `icl_additive/models.py`, `icl_additive/hermite.py`) | `results/exp1/drift.csv`, `results/exp1/drift_slopes.csv`, `results/exp1/README.md` section 1 | Core question (C2/C3): K1, P1 |
| 2 | Prop. 2 and Remark "Soft under SGD": m*, gamma_1, d* (about 31 at gamma=1, N=128; d*>150 at gamma=0.1), escape counts 3/3, 1/3, 0/3 | Closed forms checked by `scripts/verify_theorems.py` (Theorem A(i)); counts from `icl_additive/sweep1b.py --phase thr` then `icl_additive/analyze1b.py` | `results/exp1b/threshold_table.csv` (columns `pred_threshold_d_corrected`, `escaped`, `n`), `results/exp1b/thr.csv`, `results/exp1b/README.md` section 1 | Core question (C2/C3): P2, P3 |
| 3 | Remark "Soft under SGD": Kramers ratios Delta L/D_eff (0.02 at d=32, about 20 at d=64) and App. A "Kramers estimate" (sd 0.33-0.37) | No script in the repo (computed ad hoc; documented only in prose) | `docs/theory_notes.md` section 2.5 | none (post-hoc) |
| 4 | Fig. `fig:threshold` right panel / P7: 0/9 escapes at B in {32,256,1024}, final |m| 0.005-0.015 and 0.063-0.067, totals x1/x2.7/x3.2 | `icl_additive/sweep1b.py --phase p7`, `icl_additive/analyze1b.py` | `results/exp1b/p7.csv`, `results/exp1b/p7_table.csv`, `results/exp1b/README.md` section 2 | Core question (C2/C3): P7 |
| 5 | Prop. 3 discussion after the proposition: d=512 flows (pinned 0.1: inf/745/379; free: 8.9e4/2.2e4/5.3e3 with tau N = 2.9/2.8/2.7e6; tied: 39/34/33 for N=32/128/512) and d=32 free flows 52/37/34/33 for N=32/128/512/2048 | `scripts/ode_lsoda.py` (function `part4`; LSODA, rtol 1e-10). It calls the Euler functions of `scripts/ode_two_timescale.py` and `scripts/ode_tied.py` for the comparison column. | `results/ode_lsoda_quoted.csv` (15 rows, source `paper Prop.3 discussion (after Prop. 3)`), `results/ode_lsoda.csv` (quantity `tau_prop3`, `tau_times_N_prop3`), `results/README_ode.md` | none (theory; first written in `docs/theory_notes.md` sections 2.3-2.4) |
| 6 | Prop. 3(b) "tau(16)/tau(256) = 4.4 at d=64, Gamma_0=0.01, r=1" (growth-limited) versus 16 adiabatic | `scripts/ode_lsoda.py` (`part23`, quantity `ratio_T16_over_T256`) | `results/ode_lsoda_quoted.csv` (rows `free ratio T(16)/T(256) (4.4)` and `... ODE (4.44)`), SGD side in `results/exp6/ratios.csv` | Exp 6: P19b |
| 7 | Prop. 3(c) and Theorem A(iii): rho_trap = 0.178 against c_2 rho* = 0.154 at d=64, N=16; rho* = 0.1155; boundary ratios 1.3333, 1.0589, 1.3735 | `scripts/verify_theorems.py` (E5-cell check and trap-boundary bisection; output goes to stdout only, no saved file) | stdout of `python3 -I scripts/verify_theorems.py`; Remark `rem:ck` in `paper/theorems.tex` | Controls requested by review 2 (exp 7): P20d (rho_0=0.3 trap) |
| 8 | Theorem A(iii)(c) N-flatness: "10 of the 19 tied cells satisfy the hypothesis, 80/80 checks", N-ratios 1.043 / 1.859 | `scripts/verify_tied_flat.py` | `results/verify_tied_flat.txt` | none (checks E5, E6, E7 cells) |
| 9 | Theorem A, Theorem B, Prop. `prop:joint` closed forms ("61 checks", Theorem B at d up to 6.5e4, corner exponents 4k, 2k, 2k+1) | `scripts/verify_theorems.py` | stdout of the script only (no saved output) | none |
| 10 | E1: kappa_eff by SGD at d=32 (free eta_G=eta, free 10 eta, tied; N in {32,128,512,4096}); OLS values 4.8-6.6 (free), 4.10-4.44 (tied); censored counts; B=8 control 5.38 +- 0.26 vs 5.59 +- 0.20 | `icl_additive/sweep1b.py --phase kappa`, `--phase kappa2` (reruns of censored cells), `--phase bctl`; analysis `icl_additive/analyze1b.py` | `results/exp1b/kappa.csv`, `kappa2.csv`, `bctl.csv`, `summary.csv` (column `superseded`), `kappa_fits.csv`, `results/exp1b/README.md` section 4 | Core question (C2/C3): P4, P4' (outcome in Exp 1b outcome) |
| 11 | E1: population-ODE kappa_eff (free 4.85-7.0, tied 4.15-4.18) evaluated at the exact d, m_0 grid, Gamma_0, eta_G/eta; LSODA vs Euler (0.6-3.6% on times, <= 0.004 on kappa) | `scripts/ode_kappa_exp1b.py` (Euler, written before the run); `scripts/ode_lsoda.py` (`part1`, LSODA) | `results/ode_lsoda_quoted.csv` (12 rows, source `prereg P4 / paper E1 + App. (kappa_eff ODE)`), `results/ode_lsoda.csv`, `results/README_ode.md` | P4 / P4' (ODE evaluation itself was post-hoc, stated in the paper) |
| 12 | E1 and abstract: Tobit (censored) kappa_eff, "all twelve cells within 0.5" (6.05, 6.73, 6.67 in the censored cells), bootstrap s.e. 0.21-0.49, OLS 11 within 0.6 | `scripts/stats_tobit_ratios.py` | `results/exp1b/kappa_tobit.csv`, `results/README_stats.md` section 1 | P4 (post-hoc statistic) |
| 13 | E1 and App. "Censored fits and absolute-time ratios": median SGD/ODE ratios (45 E1 cells in [0.64, 1.34], 24 within 10%, 35 within 20%; E2: 18 cells, 17/18/18, range 0.85-1.10; seeds 0.50-2.08; one censored cell at least 2.35x slower) | `scripts/stats_tobit_ratios.py` | `results/ratios_sgd_vs_ode.csv`, `results/README_stats.md` section 2 (note: ratios use the Euler ODE times; `results/README_ode.md` says they rise by 0.6-3.6% with LSODA and were not regenerated) | P4, P19 (post-hoc statistic) |
| 14 | E1: learned Gamma at T_0.5 = 0.18-0.25 vs Gamma*(0.5;N) = 0.38-0.99 (P5 failed); d-scan slopes 3.49 +- 0.08 (N=128) vs 3.70 +- 0.21 (N=2048); tied rho_0=1 at N=128: d=32 escaped 3/3, d=64 trapped 0/3 with rho to 0.18 | `icl_additive/sweep1b.py --phase dscan` / `--phase tied1`; `icl_additive/analyze1b.py` | `results/exp1b/gamma_at_T05.csv`, `results/exp1b/dscan.csv`, `dscan_slopes.csv`, `results/exp1b/tied1.csv`; ODE side of the d-scan (1.05/1.08/1.12/1.20, "<15%" claim not reproduced) in `results/README_ode.md` | Core question (C2/C3): P5, P4', Priority 3 (all in Exp 1b outcome) |
| 15 | E1: "d to infinity" kappa values 6.45/5.40/4.95/4.83 (joint scaling lambda = N/d^2, r~ = r/d) and Theorem B / Prop. `prop:joint` corner statements | **No script in `scripts/`** produces the four numbers (`results/README_ode.md` says so). The corner exponents themselves are checked by `scripts/verify_theorems.py` (sections "Proposition B(ii)" and "B(iii)"). | Table in `docs/review_adversarial_3.md` section 3 (rows `N = d^2/32, r = d/32` and `N = 4d^2, r = d/32`) | none (the pre-registered "4 to 8" was withdrawn) |
| 16 | E2: SGD medians at d=64 (pinned 0.1: 396k/192k/179k; free: 1.78M/614k/402k; tied: 17.0k/15.1k/14.0k), pinned gamma=1 escape counts (0/9 at N<=64, 3/6 at N=256), free N=16 rerun to 2.5e6 steps | `icl_additive/sweep6.py` (and `sweep6.py --ext-free16-steps 2500000`), analysis `icl_additive/analyze6.py` | `results/exp6/runs.csv`, `runs_ext.csv`, `summary.csv`, `results/exp6/README.md` | N-T exchange by readout protocol (exp 6): P19a-d |
| 17 | E2: ODE escape times (400k/193k/175k, 1.82M/623k/409k, 17.0k/16.4k/16.3k) and ratio T(16)/T(256) = 4.4 (3.6-5.0 over seed pairings) vs ODE 4.44; tied seed pairings 0.9-1.45 | ODE fixed before the run: flow times in the pre-registration and in `ODE` of `icl_additive/analyze6.py`; recomputed in LSODA by `scripts/ode_lsoda.py` (`part23`) | `results/exp6/summary.csv` (`ode_steps`, Euler), `results/exp6/ratios.csv`, `results/ode_lsoda_quoted.csv` (source `prereg exp6 outcome / paper E2 / App. Exp.6`) | P19a-d |
| 18 | E5: controls at d=64 (free Gamma_0=0.1 and eta_G=10 eta: ratios 9.2, 10.7 vs ODE 9.1, 8.9; pinned 0.01 ratio 1.06 vs 1.04; tied rho_0=0.1 ratio 2.15; tied rho_0=0.3 escapes after 30x delay at rho = 0.012; eta proportional to B leaves flow time invariant to 0.5%) | `icl_additive/sweep7.py`, analysis `icl_additive/analyze7.py` | `results/exp7/runs.csv`, `summary.csv`, `ratios.csv`, `f_invariance.csv`, `results/exp7/README.md`; ODE column in `results/ode_lsoda_quoted.csv` (sources `paper App. Exp.7 / E5`, `prereg P20 (exp7 table)`) | Controls requested by review 2 (exp 7): P20a-f |
| 19 | E6 (k*=3): medians/flow 1.18/0.89/0.80, secants 5.7 and 6.0 (ODE 6.5, 6.4), controls 1.007 (eta/4) and 1.14 (B=256), pinned gamma=0.2 (m*=0.30), tied N=16 failure (47k, 500k, censored; d=32 trapped) | `scripts/ode_k3.py` (predictions fixed before the run), `icl_additive/sweep9.py`, analysis `icl_additive/analyze9.py` | `results/exp9/ode_predictions.csv`, `runs.csv`, `summary.csv`, `ratios.csv`, `secants.csv`, `verdicts.csv`, `results/exp9/README.md`; rho_trap for k=3 from `scripts/verify_tied_flat.py` (`results/verify_tied_flat.txt`) | k*=3 sanity run (exp 9): P22a-c |
| 20 | E7 (d=128): pinned 0.1 N=16 censored 3/3 at 2.5e6 steps, N=64/256 at 0.99 and 1.01 of flow (1.78e6, 1.47e6 steps), pinned 1 censored at every N, tied medians/flow 1.12/0.97/1.02 and ratio 1.20 (flow 1.09), free N=256 at 5.29e6 steps (0.985 of flow), secant 5.4 vs pre-registered 7.4 (wrong baseline) | `scripts/ode_exp10.py` (writes `results/exp10/ode_predictions.txt`), `icl_additive/sweep10.py`, analysis `icl_additive/analyze10.py` | `results/exp10/ode_predictions.txt`, `runs.csv`, `summary.csv`, `secants.csv`, `verdicts.csv`, `tables.md`, `results/exp10/README.md` | Three regimes at a second d (exp 10): P23a-d |
| 21 | Sec. 4 (E3): slopes of log T_p vs log p (in-context 1.44 +- 0.08 drop midpoint, 1.53 +- 0.22 at the 0.5 crossing; capacity-matched in-weight 0.75 +- 0.08; 2-homogeneous in-weight 0.75-0.89), T_p eta pi_p m_0^2 = 0.73 (IQR 0.70-0.77) vs single-feature 0.64, T_p eta a_p = 0.23-0.26, relative width 1.08 vs 4.09, loss exponent 0.40 +- 0.04 vs 0.33 | `icl_additive/sweep2.py` (driver; model in `icl_additive/many.py`), analysis `icl_additive/analyze2.py`; B-tied baseline: `icl_additive/btied.py`, `icl_additive/sweep_btied.py`, `icl_additive/analyze_btied.py` | `results/exp2/per_skill.csv`, `summary.csv`, `analysis_stdout.txt`, `README.md`; `results/exp2/btied/per_skill.csv`, `summary.csv`, `analysis_stdout.txt`, `results/exp2/README_Btied.md`; single-feature 0.64 in `results/exp3/README.md` section 1 | Many-skill scaling (exp 2): P16-P18 |
| 22 | Sec. 4 (E3 text): "2/10 seeds" where a rare skill is not decoupled; alpha=1 slopes 0.88 +- 0.09 (A) vs 0.61 +- 0.08 (B) | `icl_additive/sweep3.py` (2/10 seeds: `extra_seeds5to9`), `icl_additive/analyze3.py`; alpha=1 from `sweep2.py` / `analyze2.py` | `results/exp3/README.md` section 5 and `results/exp3/main/assignment.csv`, `results/exp3/extra_seeds5to9/assignment.csv`; alpha=1 in `results/exp2/README.md` | P16 (alpha=1 not pre-registered) |
| 23 | E4: additivity residual (max |R| <= 0.0086 over 13 runs, paired 17-query estimator), "90% of its drop before or together with the slower skill", accuracy fit acc12 = 0.85 acc1 acc2 + 0.09 (r.m.s. 0.008), product pair E_x not learned, neuron assignments, K=0.60 | `icl_additive/sweep3.py` (tags `main`, `extra_seeds5to9`, `gamma0.3`), analysis `icl_additive/analyze3.py` (model/sampler in `icl_additive/multi.py`) | `results/exp3/main/{additivity_residual,times,acc_fits,assignment,neurons}.csv` and `analysis_stdout.txt` (same files in `extra_seeds5to9/` and `gamma0.3/`), `results/exp3/README.md` sections 1-5 | Secondary question (C4): P12-P15 |
| 24 | Sec. 5, "Large initial alignment" (d=32, width 256): emergence steps 1000/1400/2800 for N=16/64/256 at 8192 tokens/step, best random direction |cos| = 0.43-0.55, k*=1 and in-weight controls | `icl_additive/transformer_exp4.py --grid`, analysis `icl_additive/analyze_exp4.py` | `results/exp4/summary.csv`, `results/exp4/tables.md`, `results/exp4/README.md` (alignment section); `curves/*.npz` gitignored | Transfer test (exp 4): P8-P10 |
| 25 | Sec. 5, "Small initial alignment" (d=256, width 32) and Fig. `fig:transformer`: (c) never emerges in 30k steps 3/3, (d) emerges at 10.6-14.8k, (a) at 2.8-4.0k with one seed censored at 17k, init |cos| 0.13-0.18, readout norm 0.57 to 0.03-0.16, 1.7 at one logged point after emergence, k*=1 at 1.4k, in-weight at 0.4k | `icl_additive/transformer_exp5.py --grid`, analysis `icl_additive/analyze_exp5.py` | `results/exp5/summary.csv`, `results/exp5/tables.md`, `results/exp5/README.md`; `results/exp5/curves/*.npz` (gitignored) | Regime-matched transformer test (exp 5): P11a-c |
| 26 | Sec. 5, Exp. 8 ("The readout is not the transformer's knob"): pinned O(1) readout emerges at 5.8k and 8.4k with B=1024; pinned small stays on the plateau 2/2; free reproduces Exp. 5 (3.7x, 3.8x) | `icl_additive/transformer_exp8.py --queue`, analysis `icl_additive/analyze_exp8.py` (also reads `results/exp5/curves/` for the `free` rows) | `results/exp8/summary_raw.csv`, `summary.csv`, `tables.md`, `results/exp8/README.md`; `curves/*.npz` gitignored | Transformer trap-vs-tradeoff test (exp 8): P21a-c |
| 27 | Sec. 5, Exp. 11 (attention output scale): frozen O(1) rescued by B=1024 at 15.0k and 26.4k, frozen 0.1x stuck 2/2, 3.4x and 2.6x slower at N=256, censored runs at 8k and 10.8k | `icl_additive/transformer_exp11.py --queue`, analysis `icl_additive/analyze_exp11.py` (reuses the `free` rows of `results/exp8/`) | `results/exp11/summary_raw.csv`, `summary.csv`, `tables.md`, `results/exp11/README.md`; `curves/*.npz` gitignored | Transformer: attention output scale as the Gamma-analogue (exp 11): P24a-c |
| 28 | App. C "Exp. 1" paragraph: d-slopes (B k*=1 1.98 +- 0.06, B k*=2 2.33 +- 0.22, A k*=1 2.71 +- 0.43, A k*=2 3.1-3.5), m_0 regressions (-2.19 +- 0.08 etc.), 20 of 105 runs censored | `icl_additive/sweep.py` (driver; `icl_additive/train.py`, `icl_additive/models.py`), analysis `icl_additive/plot.py` | `results/exp1/summary.csv`, `slopes.csv`, `slopes_dge16.csv`, `per_seed_regression.csv`, `T_by_config.csv`, `results/exp1/README.md` | Core question (C2/C3): K1, P1, P6 (Exp 1 outcome); the pre-registered B k*=1 slope 1.5 was an error |
| 29 | App. C "Exp. 1b" paragraph: grid of escapes at gamma=1 (2/3,0/3,0/3; 3/3,1/3,0/3; 3/3,3/3,3/3), gamma=0.1 27/27, corrected d* = 6.8, 30.8, 126.8, censored-run counts 3/12, 3/12, 2/12 | see rows 2, 4 and 10 | `results/exp1b/threshold_table.csv`, `kappa_fits.csv`, `results/exp1b/README.md` sections 1 and 4 | P2, P3, P7 |
| 30 | App. C compute figures (CPU-hours per experiment, e.g. Exp. 1 6.4, Exp. 1b 2.8, Exp. 5 6.0, Exp. 8 5.5, Exp. 11 6.0, Exp. 9 0.89, Exp. 10 3.8) | Per-run `cpu_s` is recorded by every driver | `cpu_s` columns of each `results/exp*/runs.csv` / `summary.csv`, and the "CPU time" line at the top of each `results/exp*/README.md` | each experiment's pre-reg section (budget caps) |
| 31 | App. D pre-registration summary table (P1-P24 predictions and outcomes) | none (hand-written summary) | `docs/preregistration.md` (prediction tables and the "Exp n outcome" sections; changelog at the bottom); machine-readable verdicts exist only for `results/exp9/verdicts.csv` (P22) and `results/exp10/verdicts.csv` (P23) | all |

Numbers that I could not trace to a script (also listed in the closing "Gaps" section): rows 3 and 15, and three statements of Sec. 6 "Limitations"
and Prop. 3(b): "E2 at N=16 is adiabatic to 1%", "the flow caps tau(16)/tau(256) at 12.8 as r goes to infinity (at d=64)", and "at d=256 the law would need about 1e9 steps".

## 3. E-label, experiment, results folder, scripts and pre-registration (12 rows)

Rows E1-E7 are the paragraph labels of `paper/body.tex`; the last five rows are the experiments that have no E-label. Specs are in `docs/spec_exp<n>.md` where one exists (Exps. 1, 1b, 2, 3, 4, 5, 9, 10, 11); Exps. 6, 7 and 8 are specified only inside `docs/preregistration.md`. All sweep/driver and analysis scripts are in `icl_additive/` unless a path is given.

| E-label | Exp. | Results folder | Sweep / driver script | Analysis script | ODE / prediction / statistics scripts | Pre-reg section | P-ids |
|---|---|---|---|---|---|---|---|
| E1 | 1b | `results/exp1b/` | `sweep1b.py` (phases `thr`, `p7`, `tied1`, `kappa`, `kappa2`, `bctl`, `dscan`) | `analyze1b.py` | `scripts/ode_kappa_exp1b.py`, `scripts/ode_lsoda.py`, `scripts/stats_tobit_ratios.py` | Core question (C2/C3) (outcome: Exp 1b outcome) | P2-P7, P4', Priority 3 |
| E2 | 6 | `results/exp6/` | `sweep6.py` (`--ext-free16-steps` for the rerun) | `analyze6.py` | flow times in `analyze6.py` (`ODE`); `scripts/ode_lsoda.py`; `scripts/stats_tobit_ratios.py` (ratio table) | N-T exchange by readout protocol (exp 6) | P19a-d |
| E3 | 2 | `results/exp2/` (add-on `results/exp2/btied/`) | `sweep2.py`, `many.py`; `sweep_btied.py`, `btied.py` | `analyze2.py`; `analyze_btied.py` | none | Many-skill scaling (exp 2) | P16-P18 |
| E4 | 3 | `results/exp3/` (`main/`, `extra_seeds5to9/`, `gamma0.3/`) | `sweep3.py`, `multi.py` | `analyze3.py` | none | Secondary question (C4) | P12-P15 |
| E5 | 7 | `results/exp7/` | `sweep7.py` | `analyze7.py` | `scripts/ode_lsoda.py` (`part23`), `scripts/verify_theorems.py`, `scripts/verify_tied_flat.py` | Controls requested by review 2 (exp 7) | P20a-f |
| E6 | 9 | `results/exp9/` | `sweep9.py` | `analyze9.py` | `scripts/ode_k3.py` (writes `results/exp9/ode_predictions.csv`), `scripts/verify_tied_flat.py` | k*=3 sanity run (exp 9, backlog N3) | P22a-c |
| E7 | 10 | `results/exp10/` | `sweep10.py` | `analyze10.py` | `scripts/ode_exp10.py` (writes `results/exp10/ode_predictions.txt`), `scripts/verify_tied_flat.py` | Three regimes at a second d (exp 10, backlog N2) | P23a-d |
| none | 1 | `results/exp1/` | `sweep.py` (also `drift.py` for the MC drift check) | `plot.py` | none | Core question (C2/C3) (outcome: Exp 1 outcome) | K1, P1, P6 |
| none | 4 | `results/exp4/` | `transformer_exp4.py` | `analyze_exp4.py` | none | Transfer test (exp 4, small softmax transformer) | P8-P10 |
| none | 5 | `results/exp5/` | `transformer_exp5.py` | `analyze_exp5.py` | none | Regime-matched transformer test (exp 5) | P11a-c |
| none | 8 | `results/exp8/` | `transformer_exp8.py` | `analyze_exp8.py` | none | Transformer trap-vs-tradeoff test (exp 8, backlog N1) | P21a-c |
| none | 11 | `results/exp11/` | `transformer_exp11.py` | `analyze_exp11.py` | none | Transformer: attention output scale as the Gamma-analogue (exp 11, backlog N11) | P24a-c |

Theory-only scripts (no experiment folder): `scripts/verify_theorems.py` (Theorems A and B, Prop. `prop:joint`; stdout only), `scripts/verify_tied_flat.py` (Theorem A(iii)(c); writes `results/verify_tied_flat.txt`), `scripts/ode_lsoda.py` (writes `results/ode_lsoda.csv`, `results/ode_lsoda_quoted.csv`, `results/README_ode.md`), `scripts/ode_two_timescale.py` and `scripts/ode_tied.py` (Euler flows; they run their experiments at import time, so `ode_lsoda.py` loads only their function definitions). `scripts/stats_tobit_ratios.py` writes `results/exp1b/kappa_tobit.csv` and `results/ratios_sgd_vs_ode.csv` and documents them in `results/README_stats.md`.

## 4. Regeneration

All commands run from the repository root (`/home/user/ml`). Requirements: `numpy scipy pandas matplotlib` for the toy experiments and `torch` (CPU) for Exps. 4, 5, 8, 11. Analysis scripts overwrite the committed CSV/markdown files in their results folder.

### 4.1 Figures

```
python3 -I paper/figs/make_figs.py                 # all five
python3 -I paper/figs/make_figs.py kappa           # or: threshold many transformer regimes
```

Needs only tracked CSVs, except `many` (panel b needs `results/exp3/main/traj/seed1.npz`) and `transformer` (needs `results/exp5/curves/{a,c,d}_s{0,1,2}.npz`). Both inputs are gitignored.

### 4.2 Analysis scripts (re-derive the CSV tables from the raw run files)

```
python3 -m icl_additive.drift                         # Exp 1 Monte-Carlo drift check -> results/exp1/drift*.csv
python3 -m icl_additive.plot                          # Exp 1 slopes -> results/exp1/{slopes,slopes_dge16,per_seed_regression,T_by_config}.csv
python3 -m icl_additive.analyze1b                     # Exp 1b -> summary, threshold_table, p7_table, kappa_fits, gamma_at_T05, dscan_slopes
python3 -m icl_additive.analyze2                      # Exp 2 -> per_skill.csv, summary.csv, analysis_stdout.txt (--max-seed 4 for the seeds0-4 files)
python3 -m icl_additive.analyze_btied                 # Exp 2 add-on -> results/exp2/btied/{summary,per_skill}.csv
python3 -m icl_additive.analyze3 --dir results/exp3/main                 # also --dir results/exp3/extra_seeds5to9 and results/exp3/gamma0.3 (with --figs)
python3 -m icl_additive.analyze6                      # Exp 6 -> summary.csv, ratios.csv, tables.md
python3 -m icl_additive.analyze7                      # Exp 7 -> summary.csv, ratios.csv, f_invariance.csv
python3 -m icl_additive.analyze9                      # Exp 9 -> summary, ratios, secants, verdicts
python3 -m icl_additive.analyze10                     # Exp 10 -> summary, verdicts, secants, tables.md
python3 -I icl_additive/analyze_exp4.py               # Exp 4 -> tables.md, figs
python3 -I icl_additive/analyze_exp5.py               # Exp 5
python3 -I icl_additive/analyze_exp8.py               # Exp 8 (reads results/exp5/curves too)
python3 -I icl_additive/analyze_exp11.py              # Exp 11 (reads results/exp8/curves/free_*.npz)
python3 scripts/stats_tobit_ratios.py                 # Tobit + ratio tables -> kappa_tobit.csv, ratios_sgd_vs_ode.csv
python3 scripts/ode_lsoda.py                          # ODE tables -> results/ode_lsoda.csv, ode_lsoda_quoted.csv, README_ode.md
python3 -I scripts/ode_kappa_exp1b.py                 # ODE kappa_eff, printed (the numbers hard-coded in fig_kappa)
python3 scripts/ode_k3.py                             # -> results/exp9/ode_predictions.csv
python3 scripts/ode_exp10.py > results/exp10/ode_predictions.txt
python3 -I scripts/verify_theorems.py                 # prints PASS/FAIL per check; no output file
python3 -I scripts/verify_tied_flat.py                # -> results/verify_tied_flat.txt
```

The analyses of Exps. 1b, 6, 7, 9, 10 and 2 (and the Tobit/ODE scripts) read only tracked files and can be run from a fresh clone. The analyses of Exps. 3, 4, 5, 8, 11 and `fig_many` (b) / `fig_transformer` need gitignored `.npz` files.

### 4.3 Inputs that are gitignored (need the sweep to be re-run)

`.gitignore` excludes `results/**/traj/*.npz`, `results/**/curves/*.npz`, `results/**/runs/*.npz` and `results/exp3/runs/*.npz`. Present on disk but not in a clone:

| Directory | Used by | Sweep that recreates it | Cost (CPU-h, from the READMEs) |
|---|---|---|---|
| `results/exp1/traj/` (330 files) | `icl_additive/plot.py` trajectory figure only (tables use `summary.csv`) | `icl_additive/sweep.py` | 6.4 |
| `results/exp1b/traj/` (268) | not read by `analyze1b.py` or `make_figs.py` | `icl_additive/sweep1b.py --phase ...` | 2.8 |
| `results/exp3/{main,extra_seeds5to9,gamma0.3}/traj/` (13) | `analyze3.py`, `fig_many` panel (b) (`main/traj/seed1.npz`) | `icl_additive/sweep3.py` | 2.3 |
| `results/exp4/curves/` (31), `results/exp4/fine_align/curves/` (2) | `analyze_exp4.py` figures | `icl_additive/transformer_exp4.py --grid` | 6.0 |
| `results/exp5/curves/` (14) | `analyze_exp5.py`, `analyze_exp8.py`, `fig_transformer` | `icl_additive/transformer_exp5.py --grid` | 6.0 |
| `results/exp8/curves/` (18), `results/exp11/curves/` (11) | `analyze_exp8.py`, `analyze_exp11.py` figures | `transformer_exp8.py --queue`, `transformer_exp11.py --queue` | 5.5, 6.0 |
| `results/exp6/traj/` (78), `exp7/traj/` (51), `exp9/traj/` (27), `exp10/traj/` (26) | not read by `analyze6/7/9/10.py` | `sweep6.py`, `sweep7.py`, `sweep9.py`, `sweep10.py` | 2.4, 2.2, 0.9, 3.8 |

The `.npz` files under `results/exp2/runs/` and `results/exp2/btied/runs/` (38 files) are tracked, so Exp. 2 and `fig_many` (a) regenerate from a clone.

### 4.4 Re-running the sweeps

Each driver is resumable through its CSV and skips rows already present, so move the existing `runs.csv`/`summary.csv` (and the `.npz` directory) aside before re-running or nothing is executed. Drivers use 1 BLAS thread per worker. The exact job lists used for the paper are in the `run_*_stdout.txt` / `stdout.txt` / `stage*_stdout.txt` logs of each folder and in `args.json` (Exp. 3).

```
python3 -m icl_additive.sweep                          # Exp 1 (full spec grid; --quick = smoke test)
python3 -m icl_additive.sweep1b --phase thr            # Exp 1b, one call per phase: thr p7 tied1 kappa kappa2 bctl dscan
python3 -m icl_additive.sweep2 --jobs A:1.5:0:0:1000000 B:1.5:0.1:0:3000000 ...      # Exp 2 (entries model:alpha:rho:seed:max_steps)
python3 -m icl_additive.sweep_btied --jobs 1:0 1:1 1:2 4:0 0.25:0 ...                # Exp 2 add-on (eta_mult:seed)
python3 -m icl_additive.sweep3 --tag main --seeds 0 1 2 3 4 --stop-align 0.95 --align-hold 50000
python3 -m icl_additive.sweep3 --tag extra_seeds5to9 --seeds 5 6 7 8 9 --stop-align 0.95 --align-hold 50000
python3 -m icl_additive.sweep3 --tag gamma0.3 --gamma 0.3 --seeds 0 1 2 --stop-align 0.95 --align-hold 50000
python3 -I icl_additive/transformer_exp4.py --grid     # Exp 4 (resumable via results/exp4/summary.csv)
python3 -I icl_additive/transformer_exp5.py --grid --cells a,d,c,b,e,f
python3 -I icl_additive/transformer_exp8.py --queue    # Exp 8
python3 -I icl_additive/transformer_exp11.py --queue   # Exp 11
python3 -m icl_additive.sweep6                         # Exp 6
python3 -m icl_additive.sweep6 --ext-free16-steps 2500000   # Exp 6 free N=16 rerun -> runs_ext.csv
python3 -m icl_additive.sweep7                         # Exp 7
python3 -m icl_additive.sweep9                         # Exp 9
python3 -m icl_additive.sweep10                        # Exp 10
```

The toy drivers are numpy float64 with fixed seeds; the transformer drivers use torch on CPU, so a re-run may differ at the level of floating-point nondeterminism (not checked here).

### 4.5 Caveats found while building this map

- `paper/figs/README.md` is partly stale. It says the B-tied series is not in `fig_many` and that the `fig_regimes` ODE lines are asserted to match `ode_steps` within 1%. Today `results/exp2/btied/per_skill.csv` exists, so `fig_many` draws the third (violet, tag `Btied_eta1`) series. `make_figs.py` contains no such assertion (a comment says the Euler `ode_steps` in `summary.csv` are up to 3.5% higher than the hard-coded LSODA lines).
- The ODE curves in `fig_kappa` and `fig_regimes` are typed into `make_figs.py`, not read from a file. The audit trail is `scripts/ode_kappa_exp1b.py` / `scripts/ode_lsoda.py` / `results/ode_lsoda_quoted.csv`.
- `results/ratios_sgd_vs_ode.csv` and `results/README_stats.md` use the Euler ODE times; `results/README_ode.md` records that the LSODA times are 0.7-3.6% shorter and that the ratio tables were not regenerated.
- `README.md` lists `python3 -m icl_additive.sweep1b` without `--phase`, but `--phase` is a required argument.

### 4.6 Gaps: numbers with no producing script in the repository

1. d to infinity kappa values 6.45/5.40/4.95/4.83 (E1): only a table in `docs/review_adversarial_3.md`; `results/README_ode.md` also states no script exists.
2. Kramers ratios Delta L/D_eff (about 0.02 at d=32, about 20 at d=64) and the sd 0.33-0.37 of the per-step gradient (Remark "Soft under SGD", App. A): prose in `docs/theory_notes.md` section 2.5 only.
3. Prop. 3(b) "E2 at N=16 is adiabatic to 1%".
4. Limitations: "flow caps tau(16)/tau(256) at 12.8 as r goes to infinity (at d=64)" and "at d=256 the law would need about 1e9 steps". `scripts/verify_theorems.py` has the machinery (`tau_free`, `tau_B`) but no line prints these values.
5. Outputs of `scripts/verify_theorems.py` (the "61 checks", the E5-cell rho_trap = 0.178 and the trap-boundary ratios) are printed to stdout and not saved to a file.


## Added after the map was written (scripts created to close the gaps listed above)

| number family | script | output |
|---|---|---|
| d→∞ κ values 6.45/5.40/4.95/4.83 (E1 paragraph) | `scripts/kappa_limits.py` | `results/kappa_limits.txt` |
| Limitations: flow caps τ(16)/τ(256) at 12.8 as r→∞ at d=64; "E2 at N=16 adiabatic to 1%"; ~10⁹ steps at d=256 | `scripts/ode_fast_readout.py` | `results/ode_fast_readout.txt` |
| Theorem A/B checks (61/61) and ρ_trap values | `scripts/verify_theorems.py` | `results/verify_theorems.txt` (saved stdout) |
| Kramers ratios and gradient s.d. | prose in `docs/theory_notes.md` §2.5 (no script; derived from `icl_additive/drift.py` output in `results/exp1/README.md`) | — |
