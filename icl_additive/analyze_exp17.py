"""Exp 17 analysis (P30): transformer B=256 at N=16, free readout, cap 80000 steps.
Reads results/exp17/summary_raw.csv, writes results/exp17/summary.csv and results/exp17/README.md.
Compares with the exp-16 prediction T_e(256) ~ 36k (toy Adam law, slope -0.75, applied to B=1024's median 12.7k from exp 5)
and with the exp-16 cap 30k.  Header text is read from results/exp17/deviations.txt if present."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "exp17")
CAP = 80000
OLDCAP = 30000
BASE_B, BASE_T, SLOPE = 1024, 12700.0, -0.75
PRED = BASE_T * (256 / BASE_B) ** SLOPE  # = 12700 * 4^0.75 = 35.9k
BUDGET_S = 2.5 * 3600


def fmt(v):
    return "n/r" if pd.isna(v) else str(int(v))


def main():
    df = pd.read_csv(os.path.join(OUT, "summary_raw.csv"))
    df = df[(df.proto == "free") & (df.N == 16) & (df.B == 256)].sort_values("seed").reset_index(drop=True)
    df["censored_by"] = df.censored_by.fillna("")
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    cpu = float(df.cpu_s.sum())
    n = len(df)
    e05 = df[df.reached05 == 1]
    in_band = e05[(e05.Te05 >= OLDCAP) & (e05.Te05 <= CAP)]
    n_band = len(in_band)
    ok = n_band * 2 >= n and n_band >= 1
    stuck = int((df.reached05 == 0).sum())
    L = ["| seed | steps run | T_e(0.5) | T_e(0.25) | final MSE | max align | censored | CPU s | T_e(0.5) / 36k | T_e(0.5) vs 30k cap |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        ratio = "-" if pd.isna(r.Te05) else f"{r.Te05 / PRED:.2f}"
        vs = "not reached (> %d)" % r.steps_run if pd.isna(r.Te05) else ("after cap (+%d)" % (r.Te05 - OLDCAP) if r.Te05 >= OLDCAP else "before cap")
        L.append(f"| {int(r.seed)} | {int(r.steps_run)} | {fmt(r.Te05)} | {fmt(r.Te025)} | {r.final_mse:.3f} | {r.max_align:.2f} | {r.censored_by or '-'} | {r.cpu_s:.0f} | {ratio} | {vs} |")
    t = df.Te05.fillna(CAP)
    med = float(np.median(t))
    medtxt = f"{med:.0f}" + ("+" if stuck * 2 >= n and stuck > 0 else "")
    dev_path = os.path.join(OUT, "deviations.txt")
    dev = open(dev_path).read().strip() if os.path.exists(dev_path) else "None."
    if ok:
        verdict = "**P30 HOLDS**"
    elif stuck == n:
        verdict = "**P30 FAILS; the default holds** (still stuck %d/%d at the end of the runs: steps run %s, cap 80000)" % (stuck, n, "/".join(str(int(x)) for x in df.steps_run))
    else:
        verdict = "**P30 FAILS** (%d/%d seeds emerged, %d with T_e(0.5) in [30k, 80k]; the default 'stuck 2/2' also does not hold)" % (len(e05), n, n_band)
    if stuck == n:
        interp = (f"Reading: the exp-16 graded-law prediction T_e(256) ~ {PRED / 1000:.0f}k (inside the 30k-80k band) was not realised: neither seed left the plateau "
                  "(MSE stays at 0.83 / 1.06, max alignment 0.28, readout norm still decaying toward 0), i.e. 2.2x the predicted time without any sign of escape. "
                  "Per the pre-registered default, the B-dependence of the transformer's rescue looks threshold-like between B=256 (stuck to >=77k) and B=512 (1/2 emerged by 30k), "
                  "and the toy's graded Adam law (slope -0.75) is not supported there. Caveats: n=2 seeds; B=512 itself emerged in only 1/2 seeds at 30k, so the 'threshold' is soft; "
                  "seed 1 was observed to 77000 steps, not 80000 (budget).")
    else:
        interp = ""
    text = f"""# Exp 17: transformer B=256 at N=16 with a longer cap (80k steps), prediction P30 in `docs/preregistration.md`

- **Pre-registration:** `docs/preregistration.md`, block "Transformer B=256 at N=16 with a longer cap (exp 17)" (commit eed5c94), prediction P30.
- **Code:** `icl_additive/transformer_exp8.py --run free 16 256 SEED --out results/exp17/ --max-steps 80000` (unchanged; exp-5 model/data/eval), analysis `icl_additive/analyze_exp17.py`. Logs in `logs/`, curves in `curves/*.npz` (not tracked).
- **CPU time:** {cpu:.0f} s = {cpu / 3600:.2f} CPU-h for the {n} runs (budget cap 2.5 CPU-h = {BUDGET_S:.0f} s); 1 thread per process, 2 processes in parallel.
- **Deviations:** see the section at the end.

Setting as exps 5/8/16 free: d=256, d_model=64, MLP width 32, 2 pre-LN blocks, 4-head causal softmax attention, trainable linear readout, sigma_2 link, Adam 1e-3, eval every 200 steps on 2048 held-out prompts,
T_e = first eval MSE/E[y^2] < 0.5 (and < 0.25), early stop at < 0.25 or 80000 steps. `N=16`, `B=256`, seeds 0 and 1 (same teacher/init as exps 5/8/16 for the same seed; the first 30000 steps reproduce exp 16's B=256 runs).

## Per-seed results (`n/r` = not reached within the steps run)

{chr(10).join(L)}

Median T_e(0.5) over the {n} seeds (censored at 80000): {medtxt}.
Reference points: exp-16 prediction T_e(256) = 12700 x (256/1024)^(-0.75) = **{PRED:.0f}** (toy Adam law, slope -0.75, from B=1024's median 12.7k); exp-16 cap = 30000 (both B=256 seeds were stuck there, final MSE 0.83 / 1.06).

## P30 verdict

Criterion (quoted): "the graded law holds: >=1/2 seeds emerge with `T_e(0.5) in [30k, 80k]`"; default (quoted): "still stuck 2/2 at 80k => the rescue is threshold-like between `B=256` and `B=512` and the Adam mechanism's graded law is not supported in the transformer".

Seeds emerging with T_e(0.5) in [30000, 80000]: {n_band}/{n} (any emergence: {len(e05)}/{n}; stuck at 80k: {stuck}/{n}).

{verdict}.

{interp}

## CPU time

{cpu:.0f} s ({cpu / 3600:.2f} CPU-h) summed over `cpu_s` of the {n} runs (budget 2.5 CPU-h).

## Deviations / what was not done

{dev}
"""
    open(os.path.join(OUT, "README.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
