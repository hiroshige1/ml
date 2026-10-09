# Statistics addendum: Tobit kappa_eff (exp 1b) and SGD/ODE absolute-time ratios (exp 1b, exp 6)

Code: `scripts/stats_tobit_ratios.py` (reads `results/exp1b/summary.csv`, `results/exp6/summary.csv`, ODE from `scripts/ode_kappa_exp1b.py`; seeded, deterministic).
Outputs: `results/exp1b/kappa_tobit.csv`, `results/ratios_sgd_vs_ode.csv`. Nothing else in the repo was modified or committed.

## 1. Tobit (right-censored) regression for kappa_eff

Model per (protocol, N) cell, 12 runs (4 m_0 multipliers x 3 seeds): `y = log(T_0.5 * eta) = a + b log m_0 + sigma * eps`, eps ~ N(0,1), eta = 1/1024.
A run that did not reach |m| = 0.5 within its step cap contributes `P(y > log(cap * eta)) = 1 - Phi((log(cap*eta) - a - b log m_0)/sigma)`
(`scipy.stats.norm.logsf`); reached runs contribute the Gaussian density. Parameters (a, b, log sigma) by BFGS + Nelder-Mead polish from 3 starts (`scipy.optimize`);
statsmodels is not used. `kappa_eff = 2 - b`. SE(b) = sqrt of the (b,b) element of the inverse observed information (central finite-difference Hessian of the
negative log-likelihood at the MLE; this is the primary SE) and, as a check, the SD of b over 500 non-parametric bootstrap resamples of the 12 runs (warm-started Tobit refits).
OLS-on-reached = ordinary least squares on the uncensored runs only (reproduces `kappa_fits.csv`; OLS SE with n-2 dof).
ODE kappa values are the hard-coded ones from the task (recomputing with `ode_kappa_exp1b.py` reproduces them to 2 decimals).

**Censoring handling.** Runs are `summary.csv` rows with phase in {kappa, kappa2} and `superseded == False` (first-pass censored rows that were re-run at a higher cap are dropped, as in
the README). Censoring point = the `max_steps` of that run (1e6 for all censored runs; the censored runs are exactly the 8 with `reached05 == False`: free1 N=32 m_0 x0.5 (3/3),
free10 N=32 x0.5 (3/3), free10 N=128 x0.5 (2/3)). For uncensored runs the cap is irrelevant to the likelihood. Assumptions: homoscedastic Gaussian errors in log T and that a
censored run would eventually cross 0.5 (its final |m| is 0.04-0.11, i.e. near or below m_0 = 0.088, so for those runs this is an assumption, not an observation).

| protocol | N | censored/12 | b (Tobit) | SE obs-info | SE boot | kappa_Tobit | kappa_ODE | Tobit - ODE | kappa_OLS(reached) +- SE | OLS - ODE |
|---|---|---|---|---|---|---|---|---|---|---|
| free1 | 32 | 3 | -4.048 | 0.180 | 0.228 | 6.05 | 6.47 | -0.42 | 5.45 +- 0.12 | -1.02 |
| free1 | 128 | 0 | -3.591 | 0.179 | 0.241 | 5.59 | 5.46 | +0.13 | 5.59 +- 0.20 | +0.13 |
| free1 | 512 | 0 | -2.810 | 0.098 | 0.128 | 4.81 | 4.99 | -0.18 | 4.81 +- 0.11 | -0.18 |
| free1 | 4096 | 0 | -2.920 | 0.094 | 0.112 | 4.92 | 4.85 | +0.07 | 4.92 +- 0.10 | +0.07 |
| free10 | 32 | 3 | -4.731 | 0.182 | 0.213 | 6.73 | 7.02 | -0.29 | 6.61 +- 0.26 | -0.41 |
| free10 | 128 | 2 | -4.666 | 0.368 | 0.493 | 6.67 | 6.18 | +0.49 | 5.78 +- 0.11 | -0.40 |
| free10 | 512 | 0 | -2.921 | 0.142 | 0.187 | 4.92 | 5.32 | -0.40 | 4.92 +- 0.16 | -0.40 |
| free10 | 4096 | 0 | -3.173 | 0.189 | 0.218 | 5.17 | 4.92 | +0.25 | 5.17 +- 0.21 | +0.25 |
| tied | 32 | 0 | -2.100 | 0.276 | 0.300 | 4.10 | 4.18 | -0.08 | 4.10 +- 0.30 | -0.08 |
| tied | 128 | 0 | -2.259 | 0.175 | 0.247 | 4.26 | 4.15 | +0.11 | 4.26 +- 0.19 | +0.11 |
| tied | 512 | 0 | -2.300 | 0.190 | 0.168 | 4.30 | 4.15 | +0.15 | 4.30 +- 0.21 | +0.15 |
| tied | 4096 | 0 | -2.442 | 0.160 | 0.161 | 4.44 | 4.15 | +0.29 | 4.44 +- 0.18 | +0.29 |

- max |kappa_Tobit - ODE| = 0.49 (free10, N=128; 1.3 SE); max |kappa_OLS - ODE| = 1.02 (free1, N=32).
- In the 9 cells with no censoring Tobit and OLS coincide exactly (same b; SEs differ slightly because ML uses sigma-hat with n, OLS with n-2 dof; Tobit SE is the smaller).
- Censoring correction moves kappa up by +0.60 (free1 N=32), +0.12 (free10 N=32), +0.88 (free10 N=128), as the README anticipated for exclusion bias; the Tobit
  values lie closer to the ODE in 2 of 3 censored cells (free1 N=32: 6.05 vs ODE 6.47; free10 N=32: 6.73 vs 7.02), and overshoots by 0.49 (1.3 SE) at free10 N=128 where one of the three m_0 x0.5 seeds did reach (T=2.6e5, ratio to ODE 0.61)
  while two did not (ODE 4.3e5 steps, cap 1e6): that cell is not well described by one Gaussian log-linear law (sigma-hat 0.47).
- Tobit SEs for the censored cells (0.18-0.37) are an extrapolation under the Gaussian assumption; the bootstrap SEs (0.21-0.49) are larger, so treat the SE as optimistic.
- All 12 Tobit kappa values are within 0.5 of the ODE value; 10 of 12 are within 2 SE.

## 2. Absolute-time ratio table: median(SGD T_0.5) / ODE

Full table (all cells, with per-seed T, ODE steps, bootstrap and min-max): `results/ratios_sgd_vs_ode.csv` (columns `ratio_median`, `ratio_boot68_lo/hi`, `ratio_min/max`, `in_summary`).

- **exp 1b** ODE steps = `T05(rhs, m0, G0=0.01, N, r)/eta` at each run's exact m_0 (`rhs_free` with r=1 (free1) or r=10 (free10), `rhs_tied`), eta = 1/1024; the ratio is formed per seed
  (T_seed / ODE(m_0 of that run)); the cell value is the median of the 3 seed ratios. 16 cells per protocol (N x 4 m_0 multipliers).
- **exp 6** ODE steps = the `ode_steps` column of `exp6/summary.csv` (pre-registered flow time x d^2). Pinned Gamma=1 is excluded (ODE: "trapped", no prediction). For free N=16 the 2.5e6-step reruns
  (`variant=ext`) are used; the censored 1e6 `main` rows for those two cells are dropped (they only give a lower bound). N=64 in the tok and B64 schemes are the *same* runs, so those 6 cells are duplicates (kept in the CSV, flagged `note`).
- **Bootstrap 68% interval**: resample the 3 seed ratios with replacement 5000 times, median each, take the 16th-84th percentiles. With only 3 seeds this is a discrete, coarse interval (it can only
  land on seed values), so min-max over the seeds is also given.
- **Censoring in the ratio table**: only cells where all 3 seeds reached |m| = 0.5 are "uncensored" and enter the summary (45 exp1b + 18 exp6). 3 exp1b cells are censored and are listed in the CSV
  with `in_summary=False`, no median: free1 N=32 x0.5 (0/3 reached; cap 1e6 = 0.63 x ODE 1.59e6, i.e. no information against the ODE), free10 N=32 x0.5 (0/3; cap = 0.62 x ODE 1.62e6, no information),
  free10 N=128 x0.5 (1/3 reached, ratio 0.61; the other two have ratio >= 2.35, so the median ratio is >= 2.35 against the ODE; this is the one censored cell that clearly disagrees with the ODE). The partial exp 1b control `free1_B8ctl` (B=8) is not included.


Summary (cell = median-of-3 ratio; "within x%" = |ratio - 1| <= x):

| set | n cells | within 10% | within 20% | within 50% | min | max | seed-level min | seed-level max |
|---|---|---|---|---|---|---|---|---|
| all (exp1b + exp6, duplicates counted) | 63 | 41 | 53 | 63 | 0.641 | 1.345 | 0.503 | 2.080 |
| all, exp6 N=64 counted once | 57 | 35 | 47 | 57 | 0.641 | 1.345 | 0.503 | 2.080 |
| exp1b | 45 | 24 | 35 | 45 | 0.641 | 1.345 | 0.503 | 2.080 |
| exp1b free1 | 15 | 11 | 14 | 15 | 0.679 | 1.194 | 0.601 | 1.800 |
| exp1b free10 | 14 | 7 | 11 | 14 | 0.645 | 1.343 | 0.503 | 2.036 |
| exp1b tied | 16 | 6 | 10 | 16 | 0.641 | 1.345 | 0.511 | 2.080 |
| exp6 (duplicates counted) | 18 | 17 | 18 | 18 | 0.851 | 1.095 | 0.824 | 1.393 |
| exp6 (N=64 once) | 12 | 11 | 12 | 12 | 0.851 | 1.095 | 0.840 | 1.393 |
| exp6 pinned0.1 | 6 | 6 | 6 | 6 | 0.957 | 1.016 | 0.840 | 1.393 |
| exp6 free | 6 | 6 | 6 | 6 | 0.970 | 1.024 | 0.858 | 1.140 |
| exp6 tied | 6 | 5 | 6 | 6 | 0.851 | 1.095 | 0.824 | 1.349 |

Reading: exp 6 (N-T exchange at m_0 = d^-1/2, d = 64) is within 15% in every cell. In exp 1b the cells at the smallest m_0 (x0.5, longest T and strongest dependence on the Gamma-growth phase and on noise)
carry the largest deviations (up to 1.34 for tied/free10 N=4096 x0.5; 0.64-0.71 at tied N=32/512 x0.7 and free10 N=512 x0.5); the larger-m_0 cells (x1.0, x1.4) are mostly within 10-15%.
Individual seeds scatter by 0.50-2.08x around the ODE, and neither the sign nor the size of the deviation is systematic across N or protocols (cell medians centre on 1.0).

### exp 1b cells (T in steps)

| protocol | N | m_0 x d^-1/2 | n reached | T_SGD seeds | ODE steps | median ratio | boot 68% | min-max |
|---|---|---|---|---|---|---|---|---|
| free1 | 32 | 0.5 | 0/3 | cens>=1000000 / cens>=1000000 / cens>=1000000 | 1591100 | censored (cap/ODE >= 0.63) | - | - |
| free1 | 32 | 0.7 | 3/3 | 158107 / 178694 / 216571 | 263130 | 0.679 | 0.60-0.82 | 0.60-0.82 |
| free1 | 32 | 1.0 | 3/3 | 59204 / 47882 / 53477 | 54054 | 0.989 | 0.89-1.10 | 0.89-1.10 |
| free1 | 32 | 1.4 | 3/3 | 16764 / 16721 / 16765 | 15956 | 1.051 | 1.05-1.05 | 1.05-1.05 |
| free1 | 128 | 0.5 | 3/3 | 578587 / 361530 / 872621 | 484750 | 1.194 | 0.75-1.80 | 0.75-1.80 |
| free1 | 128 | 0.7 | 3/3 | 107634 / 130017 / 115305 | 123290 | 0.935 | 0.87-1.05 | 0.87-1.05 |
| free1 | 128 | 1.0 | 3/3 | 36882 / 40733 / 33986 | 37960 | 0.972 | 0.90-1.07 | 0.90-1.07 |
| free1 | 128 | 1.4 | 3/3 | 13124 / 13516 / 13682 | 13434 | 1.006 | 0.98-1.02 | 0.98-1.02 |
| free1 | 512 | 0.5 | 3/3 | 175302 / 259763 / 279346 | 281100 | 0.924 | 0.62-0.99 | 0.62-0.99 |
| free1 | 512 | 0.7 | 3/3 | 110838 / 97678 / 95602 | 99131 | 0.985 | 0.96-1.12 | 0.96-1.12 |
| free1 | 512 | 1.0 | 3/3 | 40558 / 33915 / 33588 | 34770 | 0.975 | 0.97-1.17 | 0.97-1.17 |
| free1 | 512 | 1.4 | 3/3 | 11664 / 13615 / 14129 | 12867 | 1.058 | 0.91-1.10 | 0.91-1.10 |
| free1 | 4096 | 0.5 | 3/3 | 198772 / 253735 / 296213 | 239640 | 1.059 | 0.83-1.24 | 0.83-1.24 |
| free1 | 4096 | 0.7 | 3/3 | 96032 / 93000 / 97450 | 93168 | 1.031 | 1.00-1.05 | 1.00-1.05 |
| free1 | 4096 | 1.0 | 3/3 | 29735 / 39473 / 40165 | 33883 | 1.165 | 0.88-1.19 | 0.88-1.19 |
| free1 | 4096 | 1.4 | 3/3 | 11110 / 11269 / 13455 | 12703 | 0.887 | 0.87-1.06 | 0.87-1.06 |
| free10 | 32 | 0.5 | 0/3 | cens>=1000000 / cens>=1000000 / cens>=1000000 | 1615700 | censored (cap/ODE >= 0.62) | - | - |
| free10 | 32 | 0.7 | 3/3 | 201581 / 204798 / 375827 | 246950 | 0.829 | 0.82-1.52 | 0.82-1.52 |
| free10 | 32 | 1.0 | 3/3 | 50519 / 38265 / 40820 | 40198 | 1.016 | 0.95-1.26 | 0.95-1.26 |
| free10 | 32 | 1.4 | 3/3 | 9636 / 11007 / 10082 | 9234 | 1.092 | 1.04-1.19 | 1.04-1.19 |
| free10 | 128 | 0.5 | 1/3 | 260308 / cens>=1000000 / cens>=1000000 | 425850 | censored (cap/ODE >= 2.35) | - | - |
| free10 | 128 | 0.7 | 3/3 | 69730 / 88285 / 83596 | 77278 | 1.082 | 0.90-1.14 | 0.90-1.14 |
| free10 | 128 | 1.0 | 3/3 | 17619 / 16305 / 23037 | 18000 | 0.979 | 0.91-1.28 | 0.91-1.28 |
| free10 | 128 | 1.4 | 3/3 | 5385 / 5716 / 5879 | 5701 | 1.003 | 0.94-1.03 | 0.94-1.03 |
| free10 | 512 | 0.5 | 3/3 | 77688 / 99623 / 106513 | 154370 | 0.645 | 0.50-0.69 | 0.50-0.69 |
| free10 | 512 | 0.7 | 3/3 | 46805 / 52627 / 49873 | 43326 | 1.151 | 1.08-1.21 | 1.08-1.21 |
| free10 | 512 | 1.0 | 3/3 | 11183 / 15902 / 15664 | 13890 | 1.128 | 0.81-1.14 | 0.81-1.14 |
| free10 | 512 | 1.4 | 3/3 | 3916 / 5662 / 5630 | 5002 | 1.126 | 0.78-1.13 | 0.78-1.13 |
| free10 | 4096 | 0.5 | 3/3 | 75209 / 131174 / 198906 | 97689 | 1.343 | 0.77-2.04 | 0.77-2.04 |
| free10 | 4096 | 0.7 | 3/3 | 45212 / 38666 / 36700 | 36103 | 1.071 | 1.02-1.25 | 1.02-1.25 |
| free10 | 4096 | 1.0 | 3/3 | 10036 / 18337 / 16131 | 12880 | 1.252 | 0.78-1.42 | 0.78-1.42 |
| free10 | 4096 | 1.4 | 3/3 | 4625 / 5290 / 3972 | 4821 | 0.959 | 0.82-1.10 | 0.82-1.10 |
| tied | 32 | 0.5 | 3/3 | 5009 / 11595 / 12188 | 8713 | 1.331 | 0.57-1.40 | 0.57-1.40 |
| tied | 32 | 0.7 | 3/3 | 2356 / 6076 / 2757 | 4302 | 0.641 | 0.55-1.41 | 0.55-1.41 |
| tied | 32 | 1.0 | 3/3 | 1465 / 2306 / 1026 | 2007 | 0.730 | 0.51-1.15 | 0.51-1.15 |
| tied | 32 | 1.4 | 3/3 | 982 / 1104 / 1059 | 919 | 1.152 | 1.07-1.20 | 1.07-1.20 |
| tied | 128 | 0.5 | 3/3 | 9464 / 6961 / 17541 | 8433 | 1.122 | 0.83-2.08 | 0.83-2.08 |
| tied | 128 | 0.7 | 3/3 | 3670 / 3534 / 3929 | 4229 | 0.868 | 0.84-0.93 | 0.84-0.93 |
| tied | 128 | 1.0 | 3/3 | 1773 / 2091 / 1749 | 1989 | 0.892 | 0.88-1.05 | 0.88-1.05 |
| tied | 128 | 1.4 | 3/3 | 939 / 873 / 1163 | 914 | 1.027 | 0.96-1.27 | 0.96-1.27 |
| tied | 512 | 0.5 | 3/3 | 7418 / 10491 / 12164 | 8366 | 1.254 | 0.89-1.45 | 0.89-1.45 |
| tied | 512 | 0.7 | 3/3 | 6432 / 3003 / 2769 | 4211 | 0.713 | 0.66-1.53 | 0.66-1.53 |
| tied | 512 | 1.0 | 3/3 | 1462 / 2152 / 2292 | 1984 | 1.085 | 0.74-1.16 | 0.74-1.16 |
| tied | 512 | 1.4 | 3/3 | 1016 / 792 / 824 | 913 | 0.903 | 0.87-1.11 | 0.87-1.11 |
| tied | 4096 | 0.5 | 3/3 | 10324 / 11225 / 15879 | 8347 | 1.345 | 1.24-1.90 | 1.24-1.90 |
| tied | 4096 | 0.7 | 3/3 | 4164 / 2895 / 4714 | 4206 | 0.990 | 0.69-1.12 | 0.69-1.12 |
| tied | 4096 | 1.0 | 3/3 | 2562 / 1879 / 2155 | 1983 | 1.087 | 0.95-1.29 | 0.95-1.29 |
| tied | 4096 | 1.4 | 3/3 | 868 / 763 / 1073 | 912 | 0.951 | 0.84-1.18 | 0.84-1.18 |

### exp 6 cells (T in steps; ODE steps from `exp6/summary.csv`)

| protocol | scheme | N | T_SGD seeds | ODE steps | median ratio | boot 68% | min-max | note |
|---|---|---|---|---|---|---|---|---|
| pinned0.1 | B64 | 16 | 396360 / 375718 / 435745 | 413700 | 0.958 | 0.91-1.05 | 0.91-1.05 |  |
| pinned0.1 | B64 | 64 | 192356 / 184478 / 204102 | 196610 | 0.978 | 0.94-1.04 | 0.94-1.04 | duplicate of the other scheme (identical runs at N=64) |
| pinned0.1 | B64 | 256 | 178220 / 172147 / 180436 | 176130 | 1.012 | 0.98-1.02 | 0.98-1.02 |  |
| pinned0.1 | tok | 16 | 395743 / 347440 / 576183 | 413700 | 0.957 | 0.84-1.39 | 0.84-1.39 |  |
| pinned0.1 | tok | 64 | 192356 / 184478 / 204102 | 196610 | 0.978 | 0.94-1.04 | 0.94-1.04 | duplicate of the other scheme (identical runs at N=64) |
| pinned0.1 | tok | 256 | 178969 / 161622 / 179476 | 176130 | 1.016 | 0.92-1.02 | 0.92-1.02 |  |
| free | B64 | 16 | 1878749 / 1729934 / 2037708 | 1835000 | 1.024 | 0.94-1.11 | 0.94-1.11 | ; 2.5e6 rerun |
| free | B64 | 64 | 632151 / 613975 / 592577 | 626690 | 0.980 | 0.95-1.01 | 0.95-1.01 | duplicate of the other scheme (identical runs at N=64) |
| free | B64 | 256 | 411137 / 381685 / 430747 | 413700 | 0.994 | 0.92-1.04 | 0.92-1.04 |  |
| free | tok | 16 | 1695529 / 1779437 / 1789265 | 1835000 | 0.970 | 0.92-0.98 | 0.92-0.98 | ; 2.5e6 rerun |
| free | tok | 64 | 632151 / 613975 / 592577 | 626690 | 0.980 | 0.95-1.01 | 0.95-1.01 | duplicate of the other scheme (identical runs at N=64) |
| free | tok | 256 | 402442 / 354938 / 471462 | 413700 | 0.973 | 0.86-1.14 | 0.86-1.14 |  |
| tied | B64 | 16 | 15559 / 23268 / 18883 | 17244 | 1.095 | 0.90-1.35 | 0.90-1.35 |  |
| tied | B64 | 64 | 18536 / 15137 / 13736 | 16671 | 0.908 | 0.82-1.11 | 0.82-1.11 | duplicate of the other scheme (identical runs at N=64) |
| tied | B64 | 256 | 16066 / 16327 / 17080 | 16507 | 0.989 | 0.97-1.03 | 0.97-1.03 |  |
| tied | tok | 16 | 18142 / 16975 / 17037 | 17244 | 0.988 | 0.98-1.05 | 0.98-1.05 |  |
| tied | tok | 64 | 18536 / 15137 / 13736 | 16671 | 0.908 | 0.82-1.11 | 0.82-1.11 | duplicate of the other scheme (identical runs at N=64) |
| tied | tok | 256 | 14048 / 14003 / 16748 | 16507 | 0.851 | 0.85-1.01 | 0.85-1.01 |  |
