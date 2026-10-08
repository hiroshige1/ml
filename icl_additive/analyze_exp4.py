"""Exp 4 analysis: tables (markdown to results/exp4/tables.md) and figures."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = "results/exp4"
df = pd.read_csv(f"{R}/summary.csv")
CAP = 30000
Ns = [16, 64, 256]


def te_vals(sub, col="Te05"):
    """T_e per seed; unreached -> inf (right-censored at steps_run)."""
    return [r[col] if not pd.isna(r[col]) else np.inf for _, r in sub.sort_values("seed").iterrows()]


def fmt(vals):
    return "[" + ", ".join("n/r" if not np.isfinite(v) else str(int(v)) for v in vals) + "]"


def med(vals):
    return np.median(vals) if len(vals) else np.nan


lines = []
cells = df.groupby(["k", "d", "attn", "inweight", "N"]).size().reset_index()[["k", "d", "attn", "inweight", "N"]]
lines.append("| setting | k* | d | N | B | seeds | T_e(0.5) median | T_e(0.5) per seed | T_e(0.25) median | T_e(0.25) per seed | T_e(0.1) per seed | reached 0.5 | alignment proxy at last logged step (per seed; step 0 only if run < 1000 steps) |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
order = cells.sort_values(["inweight", "attn", "k", "d", "N"], ascending=[True, False, False, False, True])
for _, c in order.iterrows():
    sub = df[(df.k == c.k) & (df.d == c.d) & (df.attn == c.attn) & (df.inweight == c.inweight) & (df.N == c.N)]
    t5, t25, t1 = te_vals(sub, "Te05"), te_vals(sub, "Te025"), te_vals(sub, "Te01")
    name = ("in-weight control " if c.inweight else "") + c.attn
    lines.append(f"| {name} | {c.k} | {c.d} | {c.N} | {int(sub.B.iloc[0])} | {len(sub)} | {med(t5):.0f} | {fmt(t5)} | {med(t25):.0f} | {fmt(t25)} | {fmt(t1)} | "
                 f"{int(sub.reached05.sum())}/{len(sub)} | {', '.join(f'{a:.2f}' for a in sub.sort_values('seed').final_align)} |")
tab = "\n".join(lines)


def M(k, d, attn, N, iw=0):
    sub = df[(df.k == k) & (df.d == d) & (df.attn == attn) & (df.inweight == iw) & (df.N == N)]
    return te_vals(sub)


checks = []
for d in sorted(df.d.unique()):
    a, b, c = M(2, d, "softmax", 16), M(2, d, "softmax", 64), M(2, d, "softmax", 256)
    if len(a) and len(c):
        checks.append(f"- P8 (d={d}): median T_e N=16/64/256 = {med(a):.0f} / {med(b):.0f} / {med(c):.0f}; ratio T_e(16)/T_e(256) = {med(a) / med(c):.2f} "
                      f"(P8 requires >= 3); ratio T_e(256)/T_e(16) = {med(c) / med(a):.2f}")
k1 = [med(M(1, 32, "softmax", N)) for N in (16, 64, 256)]
if all(np.isfinite(k1)):
    checks.append(f"- P9a (k*=1, d=32): median T_e N=16/64/256 = {k1[0]:.0f} / {k1[1]:.0f} / {k1[2]:.0f}; max/min = {max(k1) / min(k1):.2f} (P9 requires < 1.5)")
iw = M(2, 32, "softmax", 64, 1)
if len(iw):
    inctx = {N: M(2, 32, "softmax", N) for N in (16, 64, 256)}
    checks.append(f"- P9b (in-weight control k*=2, d=32, B=128, N=64 prompts of zeroed labels): T_e per seed {fmt(iw)} (median {med(iw):.0f}); in-context medians "
                  + ", ".join(f"N={N}: {med(v):.0f}" for N, v in inctx.items())
                  + f"; faster than every in-context cell (median-vs-median): {all(med(iw) < med(v) for v in inctx.values())}; "
                  f"faster than every in-context *run* (max iw seed < min over all in-context seeds): {max(iw) < min(min(v) for v in inctx.values())}")
for N in (16, 256):
    s, l = M(2, 32, "softmax", N), M(2, 32, "linear", N)
    if len(l):
        checks.append(f"- P10 (d=32, N={N}): softmax median {med(s):.0f}, linear median {med(l):.0f} (per-seed linear {fmt(l)})")
l16, l256 = M(2, 32, "linear", 16), M(2, 32, "linear", 256)
if len(l16) and len(l256):
    checks.append(f"- P10 ratio T_e(256)/T_e(16): linear {med(l256) / med(l16):.2f}, softmax {med(M(2, 32, 'softmax', 256)) / med(M(2, 32, 'softmax', 16)):.2f}")
tot_cpu = df.cpu_s.sum()
tot_wall = df.wall_s.sum()
checks.append(f"- Total: {len(df)} runs, CPU time {tot_cpu:.0f} s = {tot_cpu / 3600:.2f} CPU-h (process time summed over 4 threads), wall {tot_wall / 3600:.2f} h; "
              f"diverged runs: {int(df.diverged.sum())}")
open(f"{R}/tables.md", "w").write(tab + "\n\n" + "\n".join(checks) + "\n")
print(tab); print(); print("\n".join(checks))


# ---- supplementary: fine-resolution (eval every 20 steps) k*=1 and in-weight runs ------------------------
fdf = pd.read_csv(f"{R}/fine/summary.csv")
fl = ["| setting | N | B | T_e(0.5) per seed (eval every 20) | median |", "|---|---|---|---|---|"]
fmed = {}
for (iw, k, N), sub in fdf.groupby(["inweight", "k", "N"]):
    t = te_vals(sub)
    fmed[(iw, k, N)] = med(t)
    fl.append(f"| {'in-weight control k*=2' if iw else f'k*={k} softmax'} | {N} | {int(sub.B.iloc[0])} | {fmt(t)} | {med(t):.0f} |")
f16, f256 = fmed.get((0, 1, 16)), fmed.get((0, 1, 256))
fine_txt = "\n".join(fl) + f"\n\n- P9a at fine resolution: median T_e(k*=1) N=16/64/256 = {fmed[(0,1,16)]:.0f} / {fmed[(0,1,64)]:.0f} / {fmed[(0,1,256)]:.0f}; max/min = {max(fmed[(0,1,N)] for N in Ns) / min(fmed[(0,1,N)] for N in Ns):.2f}"
# zero-predictor MSE on each fixed eval set (empirical mean y^2), to calibrate the 0.5/0.25/0.1 thresholds
import sys
sys.path.insert(0, "icl_additive")
import torch
import transformer_exp4 as T
zl = ["| k | d | N | seed | zero-predictor MSE on eval set |", "|---|---|---|---|---|"]
for (k, d, N, seed) in [(2, 32, 16, 0), (2, 32, 16, 1), (2, 32, 16, 2), (2, 32, 256, 0), (2, 32, 256, 1), (2, 32, 256, 2), (2, 32, 64, 0), (2, 32, 64, 1), (2, 32, 64, 2)]:
    gv = torch.Generator().manual_seed(1000 + seed); v = torch.randn(d, generator=gv); v = v / v.norm()
    gen = torch.Generator().manual_seed(T.EVAL_SEED); tot = 0.0
    for _ in range(4):
        _, y = T.make_batch(d, N, 512, v, k, gen, False); tot += (y ** 2).sum().item()
    zl.append(f"| {k} | {d} | {N} | {seed} | {tot / 2048:.3f} |")
open(f"{R}/tables.md", "a").write("\n## Fine-resolution supplementary runs\n\n" + fine_txt + "\n\n## Zero-predictor MSE on the fixed eval sets\n\n" + "\n".join(zl) + "\n")
print(fine_txt); print("\n".join(zl))

# ---- figures ------------------------------------------------------------------------------------------
os.makedirs(f"{R}/figs", exist_ok=True)
Ns = [16, 64, 256]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
styles = {(2, "softmax"): ("C0", "o"), (1, "softmax"): ("C1", "s"), (2, "linear"): ("C3", "^")}
for ax, d in zip(axes, [32, 16]):
    for (k, attn), (col, mk) in styles.items():
        xs, ys = [], []
        for N in Ns:
            v = M(k, d, attn, N)
            if not len(v):
                continue
            for j, t in enumerate(v):
                ax.plot(N * (1 + 0.03 * (j - 1)), t if np.isfinite(t) else (6000 if attn == "linear" else CAP), mk, color=col, alpha=0.35, ms=4)
            xs.append(N); ys.append(med(v))
        if xs:
            ax.plot(xs, ys, "-" + mk, color=col, label=f"k*={k}, {attn} (median)")
    iwv = M(2, d, "softmax", 64, 1)
    if len(iwv):
        ax.axhline(med(iwv), color="k", ls="--", label="in-weight control, k*=2 (median)")
        ax.axhspan(min(iwv), max(iwv), color="k", alpha=0.08)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
    ax.set_xlabel("context length N (N*B = 8192 fixed)"); ax.set_title(f"d = {d}")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("T_e (first eval step with MSE < 0.5)")
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{R}/figs/Te_vs_N.png", dpi=140); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, N in zip(axes, Ns):
    for f in sorted(glob.glob(f"{R}/curves/k2_d32_N{N}_softmax_s?.npz")):
        z = np.load(f)
        ax.plot(z["steps"], z["mse"], label=f"seed {int(z['seed'])}")
    ax.axhline(0.5, color="gray", ls=":"); ax.axhline(0.1, color="gray", ls=":")
    ax.set_yscale("log"); ax.set_title(f"k*=2 softmax, d=32, N={N}, B={8192 // N}"); ax.set_xlabel("step"); ax.grid(alpha=0.3)
axes[0].set_ylabel("eval MSE / E[y^2]"); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{R}/figs/curves_k2.png", dpi=140); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
col = {16: "C0", 64: "C2", 256: "C3"}
for ax, d in zip(axes[:2], [32, 16]):
    for f in sorted(glob.glob(f"{R}/curves/k2_d{d}_N*_softmax_s?.npz")):
        z = np.load(f); N = int(z["N"])
        ax.plot(z["align_steps"], z["align"], "-o", ms=3, color=col[N], alpha=0.7, label=f"N={N}" if int(z["seed"]) == 0 else None)
    ax.set_title(f"as logged (every 1000 steps), k*=2 softmax, d={d}", fontsize=9); ax.set_xlabel("step"); ax.grid(alpha=0.3)
ax = axes[2]
for N in (16, 64):
    z = np.load(f"{R}/fine_align/curves/k2_d32_N{N}_softmax_s0.npz")
    ax.plot(z["align_steps"], z["align"], "-o", ms=3, color=col[N], label=f"N={N}: alignment")
    ax.plot(z["steps"], np.clip(z["mse"], 0, 1.6) / 1.6, "--", color=col[N], alpha=0.6, label=f"N={N}: eval MSE / 1.6")
ax.set_title("seed 0, d=32, alignment every 200 steps (supplementary)", fontsize=9); ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
axes[0].set_ylabel("max_j |cos(W_j, v)|  (block-1 MLP rows)"); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{R}/figs/alignment.png", dpi=140); plt.close(fig)
