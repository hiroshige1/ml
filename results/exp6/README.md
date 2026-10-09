# Experiment 6: N-T exchange by readout protocol (Model A, k=2, d=64)

Pre-registration: `docs/preregistration.md`, section "N-T exchange by readout protocol (exp 6)" (P19a-d). Code: `icl_additive/sweep6.py`
(reuses `train.train`, fast projected sampler, `init=fixed`), analysis `icl_additive/analyze6.py`. Raw rows: `runs.csv` (72 runs),
`runs_ext.csv` (6 rerun, see deviations), per-run trajectories `traj/`, stdout `stage{1,2,3_ext}_stdout.txt`. Derived: `summary.csv`
(per protocol/scheme/N, with per-seed T_0.5, medians, ODE steps, readout at T_0.5), `ratios.csv`, `figs/exchange.png`.

Setup as pre-registered: sigma_2, d=64, m_0 = d^-1/2 = 0.125 exactly, eta = 1/d^2 = 1/4096, N in {16,64,256}; schemes "tok" (N*B = 4096:
B = 256/64/16) and "B64" (B = 64); protocols pinned Gamma=1, pinned Gamma=0.1, free (Gamma_0 = 0.01, eta_Gamma = eta), tied (rho_0 = 0.01);
3 seeds; early stop at |m| >= 0.5; max_steps 1e6. ODE steps = pre-registered flow time x d^2 (= x 4096).
"Median" of T_0.5 counts censored runs as +infinity; "-" = median censored. Readout at T_0.5 is read from the log (every 100 steps).

## Table: T_0.5 per seed (steps) vs ODE

| protocol | scheme | N | B | T_0.5 per seed (steps) | reached | median | ODE steps | median/ODE | readout at T_0.5 (seeds) |
|---|---|---|---|---|---|---|---|---|---|
| pinned G=1 | tok | 16 | 256 | NaN/NaN/NaN | 0/3 | - | trapped | - |  |
| pinned G=1 | tok | 64 | 64 | NaN/NaN/NaN | 0/3 | - | trapped | - |  |
| pinned G=1 | tok | 256 | 16 | 49614/43524/NaN | 2/3 | 49614 | trapped | - |  |
| pinned G=1 | B64 | 16 | 64 | NaN/NaN/NaN | 0/3 | - | trapped | - |  |
| pinned G=1 | B64 | 64 | 64 | NaN/NaN/NaN | 0/3 | - | trapped | - |  |
| pinned G=1 | B64 | 256 | 64 | NaN/113319/NaN | 1/3 | - | trapped | - |  |
| pinned G=0.1 | tok | 16 | 256 | 395743/347440/576183 | 3/3 | 395743 | 413696 | 0.96 |  |
| pinned G=0.1 | tok | 64 | 64 | 192356/184478/204102 | 3/3 | 192356 | 196608 | 0.98 |  |
| pinned G=0.1 | tok | 256 | 16 | 178969/161622/179476 | 3/3 | 178969 | 176128 | 1.02 |  |
| pinned G=0.1 | B64 | 16 | 64 | 396360/375718/435745 | 3/3 | 396360 | 413696 | 0.96 |  |
| pinned G=0.1 | B64 | 64 | 64 | 192356/184478/204102 | 3/3 | 192356 | 196608 | 0.98 |  |
| pinned G=0.1 | B64 | 256 | 64 | 178220/172147/180436 | 3/3 | 178220 | 176128 | 1.01 |  |
| free (rerun 2.5e6) | tok | 16 | 256 | 1695529/1779437/1789265 | 3/3 | 1779437 | 1835008 | 0.97 | 0.1456/0.147/0.1442 |
| free | tok | 16 | 256 | NaN/NaN/NaN | 0/3 | - | 1835008 | - | nan/nan/nan |
| free | tok | 64 | 64 | 632151/613975/592577 | 3/3 | 613975 | 626688 | 0.98 | 0.2048/0.2089/0.2096 |
| free | tok | 256 | 16 | 402442/354938/471462 | 3/3 | 402442 | 413696 | 0.97 | 0.2369/0.2335/0.2277 |
| free (rerun 2.5e6) | B64 | 16 | 64 | 1878749/1729934/2037708 | 3/3 | 1878749 | 1835008 | 1.02 | 0.1552/0.152/0.1506 |
| free | B64 | 16 | 64 | NaN/NaN/NaN | 0/3 | - | 1835008 | - | nan/nan/nan |
| free | B64 | 64 | 64 | 632151/613975/592577 | 3/3 | 613975 | 626688 | 0.98 | 0.2048/0.2089/0.2096 |
| free | B64 | 256 | 64 | 411137/381685/430747 | 3/3 | 411137 | 413696 | 0.99 | 0.2347/0.2354/0.2353 |
| tied | tok | 16 | 256 | 18142/16975/17037 | 3/3 | 17037 | 17244 | 0.99 | 0.01101/0.01104/0.01112 |
| tied | tok | 64 | 64 | 18536/15137/13736 | 3/3 | 15137 | 16671 | 0.91 | 0.01138/0.01143/0.01159 |
| tied | tok | 256 | 16 | 14048/14003/16748 | 3/3 | 14048 | 16507 | 0.85 | 0.0116/0.01139/0.01142 |
| tied | B64 | 16 | 64 | 15559/23268/18883 | 3/3 | 18883 | 17244 | 1.10 | 0.01135/0.01098/0.01123 |
| tied | B64 | 64 | 64 | 18536/15137/13736 | 3/3 | 15137 | 16671 | 0.91 | 0.01138/0.01143/0.01159 |
| tied | B64 | 256 | 64 | 16066/16327/17080 | 3/3 | 16327 | 16507 | 0.99 | 0.01139/0.01144/0.01139 |

Pinned Gamma=1 row details (final |m|, max |m| over the run): the 12 stuck runs end at |m| = 0.000-0.021 and never exceed |m| = 0.146
(m_0 = 0.125): pushed toward 0, as in exp 1b P7.

## Ratios T(16)/T(256) (median of 3 seeds; free uses the rerun to 2.5e6 steps for N=16)

| protocol | scheme | T(16)/T(256) measured | ODE | T(16)/T(64) | T(64)/T(256) |
|---|---|---|---|---|---|
| pinned G=1 | tok, B64 | undefined (stuck / censored) | trapped | | |
| pinned G=0.1 | tok | 2.21 | 2.35 | 2.06 | 1.07 |
| pinned G=0.1 | B64 | 2.22 | 2.35 | 2.06 | 1.08 |
| free | tok | 4.42 | 4.44 | 2.90 | 1.53 |
| free | B64 | 4.57 | 4.44 | 3.06 | 1.49 |
| tied | tok | 1.21 | 1.04 | 1.13 | 1.08 |
| tied | B64 | 1.16 | 1.04 | 1.25 | 0.93 |

Escape counts, pinned Gamma=1 (reached |m| = 0.5 within 1e6 steps): N=16: 0/3 (tok, B=256) and 0/3 (B64); N=64: 0/3 (tok and B64 are
the *same* three runs, B=64); N=256: 2/3 (tok, B=16; T = 49.6k, 43.5k steps) and 1/3 (B64; T = 113k). So 0/9 distinct runs escape at N <= 64,
and 3/6 escape at N=256.

## Did P19a-d hold?

- **P19a (pinned Gamma=1: trapped at all N; N=256 marginal, may escape by noise; no B compensates at N <= 64): held.** N=16 and N=64 never
  escaped for any B tested (B up to 256 at N=16, i.e. 4x the samples of B64; 0/9 distinct runs), N=256 escaped in 3/6 runs (marginal case, as the
  pre-registration allowed). Caveat: the escapes at N=256 are *faster* than the free/pinned-0.1 runs (4-11e4 steps) and happen at the *smaller* batch
  (B=16: 2/3 vs B=64: 1/3), which is consistent with noise-activated escape but with 3 seeds I cannot say that smaller B helps; 2/3 vs 1/3 is not a
  resolvable difference. The "no B compensates" statement is tested only for B in {64, 256} at N=16 and B=64 at N=64.
- **P19b (free readout: flow times 448/153/101, T(16)/T(256) ~ 4.4 at fixed B, tokens-matched step counts also differ by ~4, not a trap): held,
  but only after exceeding the pre-registered step cap.** The pre-registered N=16 flow time 448 is 1.835e6 steps, which is *above* the
  pre-registered max_steps = 1e6, so the N=16 free cells were censored at 1e6 (0/6 reached; final |m| ~ 0.14-0.15, no trap: m still rising, readout
  0.15) and the pre-registered protocol could not measure the ratio (lower bound only: > 2.4). I reran those 6 runs with max_steps 2.5e6 (same
  seeds, hence the same trajectory up to 1e6; deviation, 0.46 CPU-h). They reached at 1.70-2.04e6 steps (median ratio to ODE 0.97 tok, 1.02 B64), giving
  T(16)/T(256) = 4.42 (tok) / 4.57 (B64) against ODE 4.44. T(64) and T(256) are within 1-3% of the ODE. At N*B = 4096 the step counts differ by the
  same factor ~4.4 as at B=64, i.e. the tokens-matched and fixed-B schemes are indistinguishable (T at N=64 is the same run in both; N=16 and N=256 differ
  by <=10% between schemes, within seed scatter): B does not substitute for N at d = 64 with a free readout. No kill condition triggered.
  (Contrary to the pre-registration's wording "partially noise-dominated at d=64": the measured T matches the *population* ODE to within 3%,
  so there is no sign of a noise-driven contribution to T in these cells.)
- **P19c (tied: T independent of N within 5%): not met as stated; the kill condition is not triggered.** Median T_0.5 over N=16/64/256:
  17.0k/15.1k/14.0k (tok) and 18.9k/15.1k/16.3k (B64); ODE 17.2k/16.7k/16.5k. The medians of the two schemes differ across N by 21% (tok) and
  25% (B64, non-monotone), more than 5%. Seed scatter is comparable (std/mean 4-20%; e.g. N=16 B64: 15.6k, 18.9k, 23.3k), so with 3 seeds the 5%
  claim is not resolvable either way, and T(16)/T(256) = 1.21 / 1.16 is borderline (ODE 1.04). What can be said: tied T is within a factor 1.25 of
  the ODE at every cell and varies by a factor <= 1.25 across a 16x change in N, versus 4.4 for free and 2.2 for pinned 0.1; no monotone N-dependence like
  the free readout's is visible (B64: non-monotone). The readout stays at rho ~ 0.0114 at T_0.5 (hardly moves).
- **P19d (pinned Gamma=0.1: escapes at all N, ODE 101/48/43, T(16)/T(256) ~ 2.3): held.** 18/18 escape; median T within 0.96-1.02 of ODE;
  T(16)/T(256) = 2.21 / 2.22 (ODE 2.35, 6% below). Tokens-matched and B64 agree (N=16: 396k vs 396k; N=256: 179k vs 178k; one tok N=16 seed
  was slow, 576k).

## Deviations and notes (what did not go to plan)

- **Duplicate runs.** For N=64 the two schemes have the same (N, B, seed) config (B = 64), so tok/N=64 and B64/N=64 are identical runs (byte-identical T). The
  72 listed runs are 60 distinct; the duplicates cost ~0.16 CPU-h. Medians per scheme are therefore not independent between schemes at N=64.
- **Free N=16 rerun to 2.5e6 steps** (above). The primary table keeps the 1e6 results (censored) as run under the pre-registered cap.
- Pinned Gamma=1 stayed at 1e6 steps (pilot: N=16, B=256 costs ~213 us/step, ~0.06 CPU-h per run; total budget held, no cut to 5e5 needed).
- Pilot/main protocol for the early-stopped, 3-seed design: scatter is large for a few cells (tied, pinned-1 N=256), so ratios near 1 or counts
  such as 2/3 vs 1/3 are not statistically separated.
- CPU: main 72 runs 1.97 CPU-h, rerun 0.46 CPU-h, total 2.43 CPU-h (cap 3), 4 single-thread workers, wall ~0.5 h.
