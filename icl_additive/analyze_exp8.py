"""Tables + figures for exp 8.  python3 -I icl_additive/analyze_exp8.py
Reads results/exp8/summary_raw.csv (+ curves/*.npz written by transformer_exp8.py) and, for the free protocol cells not re-run, the exp-5 results
(results/exp5/summary.csv: cells c=N16B64, d=N16B1024, a=N256B64, seeds 0-1), which are copied into results/exp8/curves/free_*.npz (source=exp5)."""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/exp8"
PROTOS = ["pinned1", "pinned_small", "free"]
CELLS = [(16, 64), (16, 1024), (256, 64)]
E5CELL = {(16, 64): "c", (16, 1024): "d", (256, 64): "a"}
COL = {(16, 64): "#d0702a", (16, 1024): "#2a6fbb", (256, 64): "#3a9a5b"}
FIELDS = ["proto", "N", "B", "seed", "source", "steps_run", "Te05", "Te025", "reached05", "reached025", "final_mse", "min_mse", "init_mse", "init_align",
          "final_align", "max_align", "head_norm_init", "head_norm_final", "censored_by", "cpu_s"]

rows = []
if os.path.exists(f"{R}/summary_raw.csv"):
    for r in csv.DictReader(open(f"{R}/summary_raw.csv")):
        r["source"] = "exp8"; rows.append(r)
have = {(r["proto"], int(r["N"]), int(r["B"]), int(r["seed"])) for r in rows}
for r in csv.DictReader(open("results/exp5/summary.csv")):
    key = [(nb, c) for nb, c in E5CELL.items() if c == r["cell"]]
    if not key or int(r["seed"]) > 1:
        continue
    N, B = key[0][0]
    if ("free", N, B, int(r["seed"])) in have:
        continue
    z = np.load(f"results/exp5/curves/{r['cell']}_s{r['seed']}.npz")
    name = f"free_N{N}_B{B}_s{r['seed']}"
    if not os.path.exists(f"{R}/curves/{name}.npz"):
        np.savez(f"{R}/curves/{name}.npz", steps=z["steps"], mse=z["mse"], train_mse=z["train_mse"], align=z["align"], norm_steps=z["norm_steps"],
                 norms=z["norms"], head_steps=z["norm_steps"], head_norm=z["norms"][:, 2], proto="free", N=N, B=B, seed=int(r["seed"]), source="exp5")
    hn = z["norms"][:, 2]
    # exp 5 steps_run/censoring: all exp5 runs here stopped at MSE<0.25 or ran the full 30000 steps
    cens = "max_steps" if (r["Te025"] == "" and int(r["steps_run"]) >= 30000) else ""
    rows.append(dict(proto="free", N=N, B=B, seed=r["seed"], source="exp5", steps_run=r["steps_run"], Te05=r["Te05"], Te025=r["Te025"],
                     reached05=r["reached05"], reached025=r["reached025"], final_mse=r["final_mse"], min_mse=r["min_mse"], init_mse=float(z["mse"][0]),
                     init_align=r["init_align"], final_align=r["final_align"], max_align=r["max_align"], head_norm_init=hn[0], head_norm_final=hn[-1],
                     censored_by=cens, cpu_s=r["cpu_s"]))
order = {p: i for i, p in enumerate(PROTOS)}
rows.sort(key=lambda r: (order[r["proto"]], CELLS.index((int(r["N"]), int(r["B"]))), int(r["seed"])))
with open(f"{R}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def te(r, k):
    return r[k] if r[k] not in ("", "None") else "n/r"


def table():
    out = ["| protocol | N | B | seed | src | steps run | T_e(0.5) | T_e(0.25) | reached 0.5 / 0.25 | final MSE | head norm init -> final | max align | censored | CPU s |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['proto']} | {r['N']} | {r['B']} | {r['seed']} | {r['source']} | {r['steps_run']} | {te(r, 'Te05')} | {te(r, 'Te025')} | "
                   f"{r['reached05']} / {r['reached025']} | {float(r['final_mse']):.3f} | {float(r['head_norm_init']):.3f} -> {float(r['head_norm_final']):.3f} | "
                   f"{float(r['max_align']):.2f} | {r['censored_by'] or '-'} | {float(r['cpu_s']):.0f} |")
    return "\n".join(out)


def figs():
    os.makedirs(f"{R}/figs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    pos = 0; ticks = []; labs = []
    for p in PROTOS:
        for (N, B) in CELLS:
            rs = [r for r in rows if r["proto"] == p and int(r["N"]) == N and int(r["B"]) == B]
            for j, r in enumerate(rs):
                x = pos + (j - 0.5) * 0.25
                if r["Te05"] not in ("", "None"):
                    ax.plot(x, int(r["Te05"]), "o", color=COL[(N, B)], ms=8)
                else:
                    y = int(r["steps_run"])
                    ax.plot(x, y, "^", mfc="none", color=COL[(N, B)], ms=9)
                    ax.annotate("", xy=(x, y * 1.3), xytext=(x, y), arrowprops=dict(arrowstyle="->", color=COL[(N, B)]))
            ticks.append(pos); labs.append(f"N={N}\nB={B}"); pos += 1
        pos += 0.6
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=8); ax.set_yscale("log"); ax.set_ylim(150, 80000)
    ax.set_ylabel("steps"); ax.axhline(30000, color="grey", ls=":", lw=1)
    for i, p in enumerate(PROTOS):
        ax.text(i * 3.6 + 1, 55000, p, ha="center", fontsize=10, weight="bold")
    ax.set_title("T_e(0.5) per seed (filled = reached; open triangle = not reached, at steps run; free N=16 B=64 / N=16 B=1024 / N=256 B=64 are exp-5 runs)", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{R}/figs/Te_cells.png", dpi=140); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, p in zip(axs, PROTOS):
        for (N, B) in CELLS:
            for r in [r for r in rows if r["proto"] == p and int(r["N"]) == N and int(r["B"]) == B]:
                z = np.load(f"{R}/curves/{p}_N{N}_B{B}_s{r['seed']}.npz")
                ax.plot(z["steps"], z["mse"], color=COL[(N, B)], ls="-" if r["seed"] == "0" else "--", lw=1.3,
                        label=f"N={N},B={B}" if r["seed"] == "0" else None)
        ax.axhline(0.5, color="grey", ls=":", lw=0.8); ax.axhline(0.25, color="grey", ls=":", lw=0.8)
        ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title(f"{p} (solid seed 0, dashed seed 1)"); ax.legend(fontsize=8)
    axs[0].set_ylabel("eval MSE (E[y^2]=1)")
    fig.tight_layout(); fig.savefig(f"{R}/figs/curves.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    open(f"{R}/tables.md", "w").write(table() + "\n"); print(table()); figs()
