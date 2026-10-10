# Exp 18: depth and LayerNorm ablations of the transformer's B-rescue (D1 = one block, NL = no LayerNorm)

- **Pre-registration:** `docs/preregistration.md`, section "Locating the ingredient of the transformer's B-rescue: depth and LayerNorm (exp 18, backlog N26 part 1)", P31a/b with the four pre-interpreted patterns (commit 7fa36cb; not edited).
- **Code:** `icl_additive/transformer_exp18.py` (reuses exp 8's `run()` with the `free` protocol; the exp-5 model `E5.TF(d)` is patched in place right after construction; exp 5 / exp 8 are not modified), `icl_additive/analyze_exp18.py` (summary.csv, tables.md, figs/curves.png, pattern classification). Runs: `python3 -I icl_additive/transformer_exp18.py --run {D1,NL} N B SEED`, driven by a 3-worker claim-queue shell loop (order: N=256,B=64 controls, N=16,B=1024, N=16,B=64; seed, then variant D1, NL), one log per run in `logs/`. Data: `summary_raw.csv` (11 rows), `summary.csv`, `tables.md`, `figs/curves.png`, `{D1,NL}/curves/*.npz` (not committed).
- **CPU time:** 21,537 s = **5.98 CPU-h** (cap 6), 3 single-thread processes (`torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`).
- **Deviations:** 11 of the 12 pre-registered runs were done; 3 of the 11 are not full 30k-step runs for budget reasons (see Deviations below).

Setting as exp 5/8 (d=256, d_model=64, MLP width 32, 4-head causal softmax attention, linear readout, sigma_2 link, Adam 1e-3, 30000 steps max, eval every 200 steps on 2048 held-out prompts, T_e(0.5) = first eval MSE/E[y^2] < 0.5, early stop at MSE < 0.25, seeds 0 and 1). Seeds fix teacher direction and the initialisation exactly as in exp 5 (init alignment 0.1261 / 0.1515 and init readout norm 0.555 / 0.576 reproduce; block 0 of D1 keeps its seed-identical init). Variants: **D1** `model.blocks = ModuleList([blocks[0]])` (37,793 parameters vs 58,881 for the full model); **NL** `ln1, ln2` of every block and `lnf` replaced by `nn.Identity()` (58,241 parameters). The exp-5 alignment proxy (uses `ln2.weight`; set to 1 for NL) and the block-2 norms logging (last block) were redefined for the patched models only; neither enters T_e.

## Results (per seed; `n/r` = not reached; MSE normalised by E[y^2] = 1)

| variant | N | B | seed | steps run | T_e(0.5) | T_e(0.25) | reached 0.5 / 0.25 | final MSE | max align | censored | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 256 | 64 | 0 | 18600 | n/r | n/r | 0 / 0 | 1.138 | 0.27 | killed_18600_cpu_budget | 3009 |
| D1 | 256 | 64 | 1 | 3200 | 2400 | 3200 | 1 / 1 | 0.241 | 0.94 | - | 520 |
| D1 | 16 | 1024 | 0 | 25000 | 24600 | 25000 | 1 / 1 | 0.202 | 1.00 | - | 2684 |
| D1 | 16 | 1024 | 1 | 30000 | n/r | n/r | 0 / 0 | 1.058 | 0.33 | max_steps | 2872 |
| D1 | 16 | 64 | 0 | 30000 | n/r | n/r | 0 / 0 | 0.831 | 0.28 | max_steps | 302 |
| D1 | 16 | 64 | 1 | 30000 | n/r | n/r | 0 / 0 | 1.058 | 0.26 | max_steps | 233 |
| NL | 256 | 64 | 0 | 10800 | n/r | n/r | 0 / 0 | 1.143 | 0.32 | killed_10800_cpu_budget | 2992 |
| NL | 256 | 64 | 1 | 2000 | 1600 | 2000 | 1 / 1 | 0.242 | 0.96 | - | 555 |
| NL | 16 | 1024 | 0 | 30000 | n/r | n/r | 0 / 0 | 0.833 | 0.41 | max_steps | 5146 |
| NL | 16 | 1024 | 1 | 16000 | 13200 | 16000 | 1 / 1 | 0.227 | 0.97 | - | 2761 |
| NL | 16 | 64 | 0 | 30000 | n/r | n/r | 0 / 0 | 0.848 | 0.38 | max_steps | 462 |
| NL | 16 | 64 | 1 | (not run, budget) | | | | | | | |

`killed_<step>_cpu_budget`: the two N=256,B=64 seed-0 runs were stopped at the next eval after a stop-file (`STOP_free_N256_B64_s0`, which only these two runs shared) was touched; both were still on the null plateau (eval MSE 1.137-1.143 throughout, alignment proxy <= 0.32). They are **censored at 18600 (D1) / 10800 (NL) steps, not 30000-step failures**. `censored_by` was edited from the code's generic `STOP_file` to `killed_<step>_cpu_budget` in `summary_raw.csv`; npz and rows are otherwise as the code wrote them.

Compact view, T_e(0.5) per seed (seed 0, seed 1):

| variant | N=256, B=64 (control) | N=16, B=1024 | N=16, B=64 | escapes (256 / 16,1024 / 16,64) | pattern |
|---|---|---|---|---|---|
| D1 (one block) | n/r (censored 18600), 2400 | 24600, n/r | n/r, n/r | yes / yes / no | **A** |
| NL (no LayerNorm) | n/r (censored 10800), 1600 | n/r, 13200 | n/r, not run | yes / yes / no | **A** |
| full model (exp 5 / exp 8, for reference) | 4000, 2800 | 14800, 10600 | n/r, n/r | yes / yes / no | A |

## Pattern per variant and its pre-stated reading

- **D1 -> Pattern A** (N=256 escapes; N=16,B=64 stuck; N=16,B=1024 escapes): the full model's pattern. Pre-stated reading: **the removed ingredient (the second block / depth beyond one block) is not required for the B-rescue.**
- **NL -> Pattern A**: same pattern. Pre-stated reading: **LayerNorm is not required for the B-rescue.**

Both readings rest on "escapes >= 1 of 2 seeds" with n=2 per cell and are exactly as strong as that: in each variant the B=1024 rescue is one seed out of two (D1: seed 0 at 24600, seed 1 stuck at 30000; NL: seed 1 at 13200, seed 0 stuck at 30000), and in both variants one of the two N=256 control seeds was stuck on the plateau when censored for budget (the full model escaped 2/2 at N=256 within 4000 steps). The N=16,B=64 cell is stuck in every run that was run (D1 2/2, NL 1/1; the NL seed 1 was dropped, see below). I draw no transfer claim; no variant beats the full model's pattern or removes the handicap (no Pattern C).

## Whether P31a/b's weak priors matched

(The priors carried no pass criterion; all four patterns were pre-interpreted.)
- **P31a (D1: Pattern B or 0): not matched.** D1 shows Pattern A. One attention block suffices for the B-rescue (D1 B=1024 seed 0 escaped at 24600, i.e. slower than the full model's 14800 / 10600, with seed 1 stuck at 30000; the rescue is weaker and less reliable than in the two-block model but present).
- **P31b (NL: Pattern A): matched.**

## Observations (descriptive, not tested)

- Speeds (single thread): D1 about 0.105 s/step at N=16,B=1024 and 0.16-0.17 s/step at N=256,B=64 (cheaper than the full model, as expected); NL about 0.16-0.17 s/step and 0.28 s/step (same as the full model).
- Plateau lengths are long and variable: all stuck runs sit at the zero-predictor MSE (0.83 / 1.06 for N=16 seeds 0 / 1; 1.14 at N=256 seed 0) with alignment proxy <= 0.41, and escape is a sudden drop (alignment proxy to 0.94-1.00 at escape). The late escapes (D1 24600, NL 13200) and the stuck controls mean that with n=2 and a 30k-step cap the "stuck" label is a statement about the cap.

## CPU time

Sum of `cpu_s` over the 11 runs: **21,537 s = 5.98 CPU-h** (cap 6). Includes the two censored control runs (3009 s + 2992 s). The 50-step smoke tests (a few seconds) are not counted.

## Deviations / what was not done

1. **NL, N=16, B=64, seed 1 was not run** (dropped for the cap, as pre-announced for the lowest-priority cell: projected total after the other runs was 21.5k s, 350 s more would have exceeded 21.6k s). The NL N=16,B=64 cell therefore has one seed (stuck).
2. **Two N=256,B=64 seed-0 control runs were stopped for budget** (D1 at 18600, NL at 10800) while on the plateau, after the same cell's seed 1 had escaped (so the control criterion "N=256 escapes" was already met and the extra steps could not change the pattern; they could only have added a second escape time). A full-length stuck N=256 NL run costs about 8,400 CPU-s, which would not have fit together with the N=16 cells. At the time of the stop the projection (every other run stuck to 30000) was about 24k CPU-s.
3. The runs were driven by a bash claim-queue (3 workers) instead of a python `--queue`; per-run logs are in `logs/`; `--out results/exp18` writes curves to `results/exp18/{D1,NL}/curves/` (file names `free_N{N}_B{B}_s{seed}.npz`, because exp 8's `run()` names files by protocol).
4. Early stop at MSE < 0.25 for every run (T_e(0.1) not recorded); non-escaping runs ran the full 30000 steps except as listed in 1-2.
5. The exp-5 `alignment` and `norms` helpers do not apply to the patched models and were replaced in memory (see Setting); they are logging only.
