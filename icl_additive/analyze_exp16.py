"""Exp 16 analysis (P29): transformer B-scan at N=16, free readout.
Reads results/exp16/summary_raw.csv (fresh runs: B=256,512,2048) and adds the re-used rows
(B=64 stuck: exp 5 cell c seeds 0,1 and exp 8 free re-runs seeds 0,1; B=1024: exp 5 cell d seeds 0,1).
Writes results/exp16/summary.csv and results/exp16/README.md.  Censored runs enter medians at the 30000-step cap."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "exp16")
CAP = 30000
SCAN = [256, 512, 1024, 2048]
COLS = ["B", "seed", "source", "steps_run", "Te05", "Te025", "reached05", "reached025", "final_mse", "max_align", "censored_by", "cpu_s"]


def ols(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    n = len(x)
    sxx = ((x - x.mean()) ** 2).sum()
    b = ((x - x.mean()) * (y - y.mean())).sum() / sxx
    a = y.mean() - b * x.mean()
    se = np.sqrt(((y - a - b * x) ** 2).sum() / (n - 2) / sxx)
    return float(b), float(se), float(a)


def load():
    fresh = pd.read_csv(os.path.join(OUT, "summary_raw.csv"))
    fresh = fresh[(fresh.proto == "free") & (fresh.N == 16)].copy()
    fresh["source"] = "exp16"
    e5 = pd.read_csv(os.path.join(ROOT, "results", "exp5", "summary.csv"))
    e5 = e5[(e5.N == 16) & (e5.k == 2) & (e5.inweight == 0) & (e5.B.isin([64, 1024])) & (e5.seed.isin([0, 1]))].copy()
    e5["source"] = "exp5"
    e5["censored_by"] = np.where(e5.reached025 == 1, "", "max_steps")
    e8 = pd.read_csv(os.path.join(ROOT, "results", "exp8", "summary_raw.csv"))
    e8 = e8[(e8.proto == "free") & (e8.N == 16) & (e8.B == 64)].copy()
    e8["source"] = "exp8"
    df = pd.concat([d.reindex(columns=COLS) for d in (e5, e8, fresh)], ignore_index=True)
    df["censored_by"] = df.censored_by.fillna("")
    return df.sort_values(["B", "seed", "source"]).reset_index(drop=True)


def fmt(v):
    return "n/r" if pd.isna(v) else str(int(v))


def main():
    df = load()
    df[COLS].to_csv(os.path.join(OUT, "summary.csv"), index=False)
    for c in ("Te05", "Te025"):
        df[c + "_c"] = df[c].fillna(CAP)  # censored (n/r or killed) at the cap
    med = {}
    for B in sorted(df.B.unique()):
        g = df[(df.B == B)]
        if B == 64:  # 4 runs
            pass
        med[B] = dict(n=len(g), r05=int(g.reached05.sum()), r025=int(g.reached025.sum()),
                      m05=float(np.median(g.Te05_c)), m025=float(np.median(g.Te025_c)))
    # OLS over the scan
    out = {}
    for key in ("m05", "m025"):
        out[key] = ols(SCAN, [med[B][key] for B in SCAN])
    # clauses (T_e(0.5), the P29 statistic)
    m1024 = med[1024]["m05"]
    c1 = all(med[B]["r05"] >= 1 and med[B]["m05"] > m1024 and med[B]["m05"] <= CAP for B in (256, 512))
    c2 = med[2048]["m05"] < m1024
    s, se, _ = out["m05"]
    c3 = -1.2 <= s <= -0.3
    ok = c1 and c2 and c3
    # table
    L = []
    L.append("| B | seed | source | steps run | T_e(0.5) | T_e(0.25) | final MSE | max align | censored | CPU s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        L.append(f"| {r.B} | {r.seed} | {r.source} | {int(r.steps_run)} | {fmt(r.Te05)} | {fmt(r.Te025)} | {r.final_mse:.3f} | {r.max_align:.2f} | {r.censored_by or '-'} | {r.cpu_s:.0f} |")
    M = ["| B | runs | reached 0.5 / 0.25 | median T_e(0.5) | median T_e(0.25) |", "|---|---|---|---|---|"]
    for B, m in med.items():
        M.append(f"| {B} | {m['n']} | {m['r05']} / {m['r025']} | {m['m05']:.0f}{'+' if m['r05'] * 2 <= m['n'] else ''} | {m['m025']:.0f}{'+' if m['r025'] * 2 <= m['n'] else ''} |")
    fresh_cpu = df[df.source == "exp16"].cpu_s.sum()
    dev_path = os.path.join(OUT, "deviations.txt")
    dev = open(dev_path).read().strip() if os.path.exists(dev_path) else "None recorded."
    per = {B: df[df.B == B].sort_values("seed") for B in (256, 512, 1024, 2048)}
    perseed = "; ".join(f"B={B}: " + ", ".join(f"s{int(r.seed)} {fmt(r.Te05)}" for _, r in per[B].iterrows()) for B in per)
    text = f"""# Exp 16: transformer B-scan at N=16 (free readout), prediction P29 in `docs/preregistration.md`

- **Pre-registration:** `docs/preregistration.md`, block "Transformer B-scan at N=16: is the rescue graded like Adam's in the toy? (exp 16)" (commit 89391b3), prediction P29.
- **Code:** `icl_additive/transformer_exp8.py --run free 16 B SEED --out results/exp16/` (unchanged; exp-5 model/data/eval), analysis `icl_additive/analyze_exp16.py`. Logs in `logs/`, curves in `curves/*.npz` (not tracked).
- **CPU time:** {fresh_cpu:.0f} s = {fresh_cpu / 3600:.2f} CPU-h for the {int((df.source == 'exp16').sum())} fresh runs (cap 4.5 CPU-h); 1 thread per process, at most 3 processes at once.
- **Deviations:** see the section at the end.

Setting as exp 5/8 free: d=256, d_model=64, MLP width 32, 2 pre-LN blocks, 4-head causal softmax attention, trainable linear readout, sigma_2 link, Adam 1e-3, eval every 200 steps on 2048 held-out prompts,
T_e = first eval MSE/E[y^2] < 0.5 (and < 0.25), early stop at < 0.25 or 30000 steps. `N=16`, seeds 0 and 1 (same teacher/init as exps 5/8 for the same seed). Rows with `source` exp5/exp8 are re-used, not re-run.

## Per-run results (`n/r` = not reached; censored runs enter medians at the cap {CAP})

{chr(10).join(L)}

## Medians over seeds (B=64: all 4 runs; `+` = the median is a censored lower bound)

{chr(10).join(M)}

OLS of log(median T_e) on log B over B in {{256, 512, 1024, 2048}} (4 points, 2 dof):
- T_e(0.5): slope **{out['m05'][0]:.3f} +- {out['m05'][1]:.3f}** (SE)
- T_e(0.25): slope {out['m025'][0]:.3f} +- {out['m025'][1]:.3f} (secondary; P29 is stated for T_e(0.5))

Per-seed T_e(0.5): {perseed}.

## P29 verdict

Criteria (quoted): "`T_e` decreases with `B` gradually: `B=256` and `B=512` emerge (>=1/2 each) at times between `B=64`'s (inf) and `B=1024`'s, `B=2048` earlier than `B=1024`; OLS slope of `log T_e` (medians, censored = cap) vs `log B` over `B in {{256,...,2048}}` in `[-1.2, -0.3]`" - pass = all three clauses; default = "`B=256` and `B=512` both stuck 2/2 (threshold-like) or slope in `(-0.3, 0)`".

Operationalisation (mine): clause 1 = at each of B=256, 512 at least 1 of 2 seeds reaches T_e(0.5) < 30000 AND the median T_e(0.5) is above the B=1024 median ({m1024:.0f}) and not above the cap; clause 2 = median T_e(0.5) at B=2048 below the B=1024 median; clause 3 = slope of median T_e(0.5) in [-1.2, -0.3].

| clause | observed | holds? |
|---|---|---|
| 1. B=256, B=512 emerge (>=1/2) with median T_e between B=1024's ({m1024:.0f}) and infinity | B=256: {med[256]['r05']}/2 emerged, median {med[256]['m05']:.0f}; B=512: {med[512]['r05']}/2 emerged, median {med[512]['m05']:.0f} | {'yes' if c1 else '**NO**'} |
| 2. B=2048 earlier than B=1024 | median {med[2048]['m05']:.0f} vs {m1024:.0f} | {'yes' if c2 else '**NO**'} |
| 3. slope in [-1.2, -0.3] | {s:.3f} +- {se:.3f} | {'yes' if c3 else '**NO**'} |

**P29 {'HOLDS (all three clauses)' if ok else 'FAILS: clause(s) ' + ', '.join(str(i) for i, c in ((1, c1), (2, c2), (3, c3)) if not c)}.**
{'' if ok else 'Clause 1 detail: B=256 emerges in ' + str(med[256]['r05']) + '/2 seeds, B=512 in ' + str(med[512]['r05']) + '/2. Clauses 2 and 3 alone would hold, but the graded-rescue claim needs all three. '}
{'' if ok else 'Default condition check: B=256 and B=512 both stuck 2/2: ' + str(med[256]['r05'] == 0 and med[512]['r05'] == 0) + '; slope in (-0.3, 0): ' + str(-0.3 < s < 0) + '.'}
n=2 seeds per cell, 4 points, 2 degrees of freedom: the slope SE is a rough guide only.

## CPU time

Fresh runs: {fresh_cpu:.0f} s ({fresh_cpu / 3600:.2f} CPU-h) summed over `cpu_s` of the exp16 rows (cap 4.5 CPU-h = 16200 s). Re-used rows cost nothing new.

## Deviations / what was not done

{dev}
"""
    open(os.path.join(OUT, "README.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
