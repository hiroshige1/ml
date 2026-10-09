"""Tables + figures for exp 11.  python3 -I icl_additive/analyze_exp11.py
Reads results/exp11/summary_raw.csv (+ curves/*.npz from transformer_exp11.py) and the free rows of results/exp8/summary.csv (not re-run;
curves from results/exp8/curves/free_*.npz).  Free runs did not log attn.out norms for block 0 (block 1 = E5 norms[:,1], logged every 1000)."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, R8 = "results/exp11", "results/exp8"
PROTOS = ["O1", "Os", "free"]
CELLS = [(16, 64), (16, 1024), (256, 64)]
COL = {(16, 64): "#d0702a", (16, 1024): "#2a6fbb", (256, 64): "#3a9a5b"}
FIELDS = ["proto", "N", "B", "seed", "source", "steps_run", "Te05", "Te025", "reached05", "reached025", "final_mse", "min_mse", "init_mse", "init_align",
          "final_align", "max_align", "ao0_norm_init", "ao0_norm_final", "ao1_norm_init", "ao1_norm_final", "censored_by", "cpu_s"]

rows = []
if os.path.exists(f"{R}/summary_raw.csv"):
    for r in csv.DictReader(open(f"{R}/summary_raw.csv")):
        r["source"] = "exp11"; rows.append(r)
for r in csv.DictReader(open(f"{R8}/summary.csv")):
    if r["proto"] != "free":
        continue
    z = np.load(f"{R8}/curves/free_N{r['N']}_B{r['B']}_s{r['seed']}.npz")
    nm = z["norms"][:, 1]   # block-1 (second block) attn.out weight norm, every 1000 steps
    r = dict(r); r["source"] = "exp8" if r["source"] == "exp8" else "exp8 (exp5 row)"
    r.update(ao0_norm_init="", ao0_norm_final="", ao1_norm_init=nm[0], ao1_norm_final=nm[-1])
    rows.append(r)
order = {p: i for i, p in enumerate(PROTOS)}
rows.sort(key=lambda r: (order[r["proto"]], CELLS.index((int(r["N"]), int(r["B"]))), int(r["seed"])))
with open(f"{R}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def te(r, k):
    return r[k] if r[k] not in ("", "None") else "n/r"


def nm(r, b):
    a, z = r[f"ao{b}_norm_init"], r[f"ao{b}_norm_final"]
    return "n/l" if a == "" else f"{float(a):.3f} -> {float(z):.3f}"


def table():
    out = ["| protocol | N | B | seed | src | steps run | T_e(0.5) | T_e(0.25) | reached 0.5 / 0.25 | final MSE | attn.out norm block 0 init -> final | attn.out norm block 1 init -> final | max align | censored | CPU s |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['proto']} | {r['N']} | {r['B']} | {r['seed']} | {r['source']} | {r['steps_run']} | {te(r, 'Te05')} | {te(r, 'Te025')} | "
                   f"{r['reached05']} / {r['reached025']} | {float(r['final_mse']):.3f} | {nm(r, 0)} | {nm(r, 1)} | "
                   f"{float(r['max_align']):.2f} | {r['censored_by'] or '-'} | {float(r['cpu_s']):.0f} |")
    return "\n".join(out)


def compact():
    out = ["| protocol | N=16, B=64 | N=16, B=1024 | N=256, B=64 |", "|---|---|---|---|"]
    for p in PROTOS:
        cells = []
        for (N, B) in CELLS:
            rs = [r for r in rows if r["proto"] == p and int(r["N"]) == N and int(r["B"]) == B]
            if not rs:
                cells.append("not run"); continue
            cells.append(", ".join(r["Te05"] if r["Te05"] not in ("", "None") else f"n/r ({r['steps_run']})" for r in rs))
        out.append(f"| {p} | " + " | ".join(cells) + " |")
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
    ax.set_title("T_e(0.5) per seed (filled = reached; open triangle = not reached, at steps run; free = exp-8/exp-5 rows, not re-run)", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{R}/figs/Te_cells.png", dpi=140); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, p in zip(axs, PROTOS):
        for (N, B) in CELLS:
            for r in [r for r in rows if r["proto"] == p and int(r["N"]) == N and int(r["B"]) == B]:
                d = R if p != "free" else R8
                f = f"{d}/curves/{p}_N{N}_B{B}_s{r['seed']}.npz"
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                ax.plot(z["steps"], z["mse"], color=COL[(N, B)], ls="-" if r["seed"] == "0" else "--", lw=1.3,
                        label=f"N={N},B={B}" if r["seed"] == "0" else None)
        ax.axhline(0.5, color="grey", ls=":", lw=0.8); ax.axhline(0.25, color="grey", ls=":", lw=0.8)
        ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title(f"{p} (solid seed 0, dashed seed 1)"); ax.legend(fontsize=8)
    axs[0].set_ylabel("eval MSE (E[y^2]=1)")
    fig.tight_layout(); fig.savefig(f"{R}/figs/curves.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    open(f"{R}/tables.md", "w").write(table() + "\n\nT_e(0.5) per seed:\n\n" + compact() + "\n"); print(table()); print(compact()); figs()
