"""Experiment 12 analysis: merge per-run rows into results/exp12/runs.csv, print the per-cell table and the P25a-d verdicts."""
import glob
import json
import os

import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp12")
COLS = ["model", "opt", "N", "B", "seed", "steps", "T05", "reached05", "final_abs_m", "max_abs_m", "cpu_s", "max_dm_step", "lr_adam"]


def load():
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "rows", "*.json")))]
    df = pd.DataFrame(rows)
    df = df.sort_values(["opt", "model", "N", "B", "seed"]).reset_index(drop=True)
    df[COLS].to_csv(os.path.join(OUT, "runs.csv"), index=False)
    return df


def escapes(g):
    return int(g.reached05.sum())


def cell_table(df):
    lines = ["| model | opt | N | B | steps (seed0 / seed1) | T_0.5 (seed0 / seed1) | reached | final \\|m\\| | max \\|m\\| | max step \\|dm\\| | CPU s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (opt, model, N, B), g in df.groupby(["opt", "model", "N", "B"], sort=False):
        g = g.sort_values("seed")
        f = lambda c, fmt="{}": " / ".join(fmt.format(x) for x in g[c])
        t05 = " / ".join(str(int(x)) if x > 0 else "CENS" for x in g.T05)
        lines.append(f"| {model} | {opt} | {N} | {B} | {f('steps')} | {t05} | {'/'.join('Y' if r else 'N' for r in g.reached05)} | "
                     f"{f('final_abs_m', '{:.3f}')} | {f('max_abs_m', '{:.3f}')} | {f('max_dm_step', '{:.4f}')} | {f('cpu_s', '{:.0f}')} |")
    return "\n".join(lines)


def verdicts(df):
    main = df[~df.opt.str.contains("lr1e-4")]
    def esc(model, opt, N, B):
        g = main[(main.model == model) & (main.opt == opt) & (main.N == N) & (main.B == B)]
        return escapes(g), len(g)
    out = {}
    for pid, model, opt in (("P25a", "linear", "sgd"), ("P25b", "linear", "adam"), ("P25c", "softmax", "sgd"), ("P25d", "softmax", "adam")):
        e = {f'N{N}_B{B}': esc(model, opt, N, B) for N, B in ((16, 64), (16, 1024), (1024, 64))}
        # a cell "escapes" if >=1 of 2 seeds reaches |m|>=0.5
        stuck16 = all(e[k][0] == 0 for k in ('N16_B64', 'N16_B1024'))
        esc1024 = e['N1024_B64'][0] == 2
        out[pid] = dict(cells=e, stuck16=stuck16, esc1024_2of2=esc1024, held=stuck16 and esc1024)
    return out


if __name__ == "__main__":
    df = load()
    tab, ver = cell_table(df), verdicts(df)
    print(tab)
    print(json.dumps(ver, indent=1))
    print("total CPU h:", df.cpu_s.sum() / 3600)
    open(os.path.join(OUT, "table.md"), "w").write(tab + "\n")
    json.dump(ver, open(os.path.join(OUT, "verdicts.json"), "w"), indent=1)
