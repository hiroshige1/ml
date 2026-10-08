"""Tables + figures for exp 5.  python3 -I icl_additive/analyze_exp5.py  (reads results/exp5/summary.csv, curves/*.npz)"""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/exp5"
LAB = {"a": "(a) N=256,B=64", "b": "(b) N=64,B=64", "c": "(c) N=16,B=64", "d": "(d) N=16,B=1024", "e": "(e) k*=1 N=16,B=64", "f": "(f) in-weight N=64,B=64"}
ORDER = "abcdef"
rows = list(csv.DictReader(open(f"{R}/summary.csv")))
by = {c: sorted([r for r in rows if r["cell"] == c], key=lambda r: int(r["seed"])) for c in ORDER}
COL = ["#2a6fbb", "#d0702a", "#3a9a5b"]


def fmt(r, key):
    return r[key] if r[key] not in ("", "None") else "n/r"


def table():
    out = ["| cell | B | seeds | T_e(0.5) per seed | T_e(0.25) per seed | reached 0.5 | steps run | final MSE | init align | CPU s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in ORDER:
        rs = by[c]
        if not rs:
            continue
        out.append(f"| {LAB[c]} | {rs[0]['B']} | {len(rs)} | [{', '.join(fmt(r, 'Te05') for r in rs)}] | [{', '.join(fmt(r, 'Te025') for r in rs)}] | "
                   f"{sum(int(r['reached05']) for r in rs)}/{len(rs)} | [{', '.join(r['steps_run'] for r in rs)}] | [{', '.join('%.3f' % float(r['final_mse']) for r in rs)}] | "
                   f"[{', '.join('%.3f' % float(r['init_align']) for r in rs)}] | {sum(float(r['cpu_s']) for r in rs):.0f} |")
    return "\n".join(out)


def figs():
    os.makedirs(f"{R}/figs", exist_ok=True)
    cells = [c for c in ORDER if by[c]]
    # T_e
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cap = max(int(r["steps_run"]) for c in cells for r in by[c])
    for i, c in enumerate(cells):
        for j, r in enumerate(by[c]):
            ok = r["Te05"] not in ("", "None")
            y = int(r["Te05"]) if ok else int(r["steps_run"])
            ax.scatter(i + (j - 1) * 0.12, y, marker="o" if ok else "^", s=50, color=COL[j % 3], facecolor=COL[j % 3] if ok else "none", zorder=3)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels([LAB[c].replace(",", ",\n", 1) for c in cells], fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("T_e (first eval MSE < 0.5), steps"); ax.grid(alpha=.3, axis="y")
    ax.set_title("Exp 5: emergence step per cell (filled = reached; open triangle = not reached, plotted at steps run)\ncolour = seed", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{R}/figs/Te_cells.png", dpi=150); plt.close(fig)
    # curves
    n = len(cells); fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    for ax, c in zip(axs.ravel(), cells):
        for j, r in enumerate(by[c]):
            z = np.load(f"{R}/curves/{c}_s{r['seed']}.npz")
            ax.plot(z["steps"], z["mse"], color=COL[j % 3], label=f"seed {r['seed']}")
        ax.axhline(0.5, color="k", ls=":", lw=.8); ax.set_title(LAB[c], fontsize=9); ax.set_xlabel("step"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    for ax in axs[:, 0]: ax.set_ylabel("eval MSE")
    fig.tight_layout(); fig.savefig(f"{R}/figs/curves.png", dpi=150); plt.close(fig)
    # alignment
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    for ax, c in zip(axs.ravel(), cells):
        for j, r in enumerate(by[c]):
            z = np.load(f"{R}/curves/{c}_s{r['seed']}.npz")
            ax.plot(z["steps"], z["align"], color=COL[j % 3], label=f"seed {r['seed']}")
            ax.plot(z["steps"], np.clip(z["mse"], 0, 1.5) / 1.5, color=COL[j % 3], ls=":", lw=.8)
        ax.set_title(LAB[c], fontsize=9); ax.set_xlabel("step"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    for ax in axs[:, 0]: ax.set_ylabel("max_j |cos(W_j, v)|   (dotted: eval MSE / 1.5)")
    fig.tight_layout(); fig.savefig(f"{R}/figs/alignment.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    t = table(); open(f"{R}/tables.md", "w").write(t + "\n"); print(t)
    tot = sum(float(r["cpu_s"]) for r in rows); print("total cpu_s", tot, "=", tot / 3600, "CPU-h (process time), wall sum", sum(float(r["wall_s"]) for r in rows) / 3600)
    figs()
