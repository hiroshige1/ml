"""Experiment 15 analysis: merge rows into results/exp15/runs.csv; per-(opt,B) table, log-log OLS slopes of median T_0.5 vs B, ratios, verdicts P28a/b."""
import glob
import json
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp15")
COLS = ["model", "opt", "N", "B", "seed", "steps", "T05", "reached05", "final_abs_m", "max_abs_m", "cpu_s", "max_dm_step", "lr_adam", "log_every"]
ETA = 1.0 / 64 ** 2


def load():
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "rows", "*.json")))]
    df = pd.DataFrame(rows).sort_values(["opt", "B", "seed"]).reset_index(drop=True)
    df[COLS].to_csv(os.path.join(OUT, "runs.csv"), index=False)
    return df


def ols(x, y):
    """slope, SE of the slope (n-2 dof), intercept."""
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    n = len(x)
    xb, yb = x.mean(), y.mean()
    sxx = ((x - xb) ** 2).sum()
    b = ((x - xb) * (y - yb)).sum() / sxx
    a = yb - b * xb
    res = y - a - b * x
    se = np.sqrt((res ** 2).sum() / (n - 2) / sxx) if n > 2 else float("nan")
    return float(b), float(se), float(a)


def cell_stats(df):
    """T per run = T05 if reached else the step cap (a lower bound; flagged CENS)."""
    out = {}
    for opt, go in df.groupby("opt"):
        for B, g in go.groupby("B"):
            g = g.sort_values("seed")
            T = np.where(g.reached05 == 1, g.T05, g.steps).astype(float)
            out[(opt, int(B))] = dict(seeds=list(g.seed), T=T, reached=list(g.reached05), median=float(np.median(T)),
                                      cens=int((g.reached05 == 0).sum()), cpu=float(g.cpu_s.sum()))
    return out


def table(cs):
    lines = ["| opt | B | T_0.5 steps (per seed) | median T_0.5 (steps) | median T_0.5 x eta (flow time) | reached | CPU s (sum) |",
             "|---|---|---|---|---|---|---|"]
    for (opt, B), c in sorted(cs.items()):
        ts = " / ".join(f"{int(t)}" + ("" if r else " (CENS)") for t, r in zip(c["T"], c["reached"]))
        flow = f"{c['median'] * ETA:.2f}" if opt == "sgd" else "n/a"
        lines.append(f"| {opt} | {B} | {ts} | {c['median']:.0f} | {flow} | {sum(c['reached'])}/{len(c['reached'])} | {c['cpu']:.0f} |")
    return "\n".join(lines)


def summarise(cs, df):
    res = {}
    for opt in ("adam", "sgd"):
        Bs = sorted(B for (o, B) in cs if o == opt)
        med = [cs[(opt, B)]["median"] for B in Bs]
        b, se, _ = ols(Bs, med)
        allB = [B for B in Bs for _ in cs[(opt, B)]["T"]]
        allT = [t for B in Bs for t in cs[(opt, B)]["T"]]
        bp, sep, _ = ols(allB, allT)  # supplementary: all runs pooled
        r = dict(Bs=Bs, medians=med, slope=b, slope_se=se, slope_pooled_runs=bp, slope_pooled_se=sep,
                 ratio_16_over_1024=(cs[(opt, 16)]["median"] / cs[(opt, 1024)]["median"]) if (opt, 16) in cs and (opt, 1024) in cs else None,
                 spread_max_over_min=max(med) / min(med), censored=sum(cs[(opt, B)]["cens"] for B in Bs))
        res[opt] = r
    a, s = res["adam"], res["sgd"]
    a["P28a_pass"] = bool(-0.75 <= a["slope"] <= -0.25 and a["ratio_16_over_1024"] >= 2)
    a["P28a_default"] = bool(-0.25 < a["slope"] < 0.25)
    s["P28b_pass"] = bool(-0.15 < s["slope"] < 0.15 and s["spread_max_over_min"] <= 1.15)
    return res


if __name__ == "__main__":
    df = load()
    cs = cell_stats(df)
    tab = table(cs)
    res = summarise(cs, df)
    print(tab)
    print(json.dumps(res, indent=1))
    print("runs:", len(df), "total CPU h:", df.cpu_s.sum() / 3600)
    open(os.path.join(OUT, "table.md"), "w").write(tab + "\n")
    json.dump(res, open(os.path.join(OUT, "verdicts.json"), "w"), indent=1)
