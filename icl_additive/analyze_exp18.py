"""Tables + pattern classification for exp 18.  python3 -I icl_additive/analyze_exp18.py
Reads results/exp18/summary_raw.csv (written by transformer_exp18.py).  Writes results/exp18/summary.csv, tables.md, figs/curves.png.
Pattern per variant (pre-registered; escapes = >=1 of the seeds reaches T_e(0.5) within 30000 steps):
  0: N=256,B=64 stuck (no escape)                                  -> variant does not learn the task; uninformative
  A: N=256 escapes, N=16,B=64 stuck, N=16,B=1024 escapes          -> removed ingredient NOT required for the B-rescue
  B: N=256 escapes, N=16 stuck at both B                           -> removed ingredient REQUIRED for the rescue
  C: N=16 escapes at both B                                        -> removed ingredient CAUSES the short-context handicap
  other: none of the above (e.g. N=16,B=64 stuck, N=16,B=1024 stuck is B; N=16,B=64 escapes but B=1024 stuck is reported as 'other')
Cells that are missing (budget) are reported as 'n/a' and a pattern needing them is labelled 'provisional'."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/exp18"
VARIANTS = ["D1", "NL"]
CELLS = [(256, 64), (16, 1024), (16, 64)]
COL = {(16, 64): "#d0702a", (16, 1024): "#2a6fbb", (256, 64): "#3a9a5b"}
rows = list(csv.DictReader(open(f"{R}/summary_raw.csv")))
order = {v: i for i, v in enumerate(VARIANTS)}
rows.sort(key=lambda r: (order[r["variant"]], CELLS.index((int(r["N"]), int(r["B"]))), int(r["seed"])))
with open(f"{R}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def te(r, k):
    return r[k] if r[k] not in ("", "None") else "n/r"


def esc(v, N, B):
    rs = [r for r in rows if r["variant"] == v and int(r["N"]) == N and int(r["B"]) == B]
    if not rs:
        return None
    return any(int(r["reached05"]) for r in rs)


def pattern(v):
    c256, c1024, c64 = esc(v, 256, 64), esc(v, 16, 1024), esc(v, 16, 64)
    miss = [n for n, c in (("N256B64", c256), ("N16B1024", c1024), ("N16B64", c64)) if c is None]
    if c256 is False:
        p = "0"
    elif c256 and c1024 and c64:
        p = "C"
    elif c256 and c1024 is False and c64 is False:
        p = "B"
    elif c256 and c1024 and c64 is False:
        p = "A"
    elif c256 and c1024 is None and c64 is None:
        p = "?"
    else:
        p = "other"
    note = f" (provisional; missing {','.join(miss)})" if miss else ""
    return p + note


lines = ["| variant | N | B | seed | steps run | T_e(0.5) | T_e(0.25) | reached 0.5 / 0.25 | final MSE | max align | censored | CPU s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['variant']} | {r['N']} | {r['B']} | {r['seed']} | {r['steps_run']} | {te(r, 'Te05')} | {te(r, 'Te025')} | {r['reached05']} / {r['reached025']} | "
                 f"{float(r['final_mse']):.3f} | {float(r['max_align']):.2f} | {r['censored_by'] or '-'} | {float(r['cpu_s']):.0f} |")
lines += ["", "Compact view, T_e(0.5) per seed (0, 1); `n/r` = not reached; `-` = run not done:", "",
          "| variant | N=256, B=64 | N=16, B=1024 | N=16, B=64 | escapes (256 / 16,1024 / 16,64) | pattern |", "|---|---|---|---|---|---|"]
for v in VARIANTS:
    cs = []
    for (N, B) in CELLS:
        d = {int(r["seed"]): te(r, "Te05") for r in rows if r["variant"] == v and int(r["N"]) == N and int(r["B"]) == B}
        cs.append(", ".join(d.get(s, "-") for s in (0, 1)))
    ee = ["-" if esc(v, N, B) is None else ("yes" if esc(v, N, B) else "no") for (N, B) in CELLS]
    lines.append(f"| {v} | " + " | ".join(cs) + f" | {' / '.join(ee)} | {pattern(v)} |")
tot = sum(float(r["cpu_s"]) for r in rows)
lines += ["", f"Sum of cpu_s over {len(rows)} runs: {tot:.0f} s = {tot / 3600:.2f} CPU-h."]
open(f"{R}/tables.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

os.makedirs(f"{R}/figs", exist_ok=True)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, v in zip(axs, VARIANTS):
    for (N, B) in CELLS:
        for s, ls in ((0, "-"), (1, "--")):
            p = f"{R}/{v}/curves/free_N{N}_B{B}_s{s}.npz"
            if os.path.exists(p):
                z = np.load(p)
                ax.plot(z["steps"], z["mse"], ls, color=COL[(N, B)], label=f"N={N} B={B} s{s}")
    ax.axhline(0.5, color="gray", lw=0.6); ax.axhline(0.25, color="gray", lw=0.6, ls=":")
    ax.set_title(v); ax.set_xlabel("step"); ax.set_ylim(0, 1.8)
axs[0].set_ylabel("eval MSE / E[y^2]"); axs[1].legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(f"{R}/figs/curves.png", dpi=130)
