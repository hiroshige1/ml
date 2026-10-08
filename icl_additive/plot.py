"""Figures + fitted-slope tables for experiment 1.  Usage: python -m icl_additive.plot [--out results/exp1]"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# predicted T_0.5 slope in d under eta = eta_0/d^2:  kappa/2 + 1
KAPPA = {("B", 1): 1, ("B", 2): 2, ("B", 3): 3, ("A", 1): 2, ("A", 2): 4, ("A", 3): 6}
PRED = {key: kap / 2 + 1 for key, kap in KAPPA.items()}
PRED_TXT = {("B", 1): "1.5", ("B", 2): "2 (2.1-2.3 incl. polylog)", ("B", 3): "2.5", ("A", 1): "2 (+polylog)",
            ("A", 2): "3", ("A", 3): "4"}
COL = {"A": "#d95f02", "B": "#1b9e77"}
LS = {2.0: "-", 1.0: "--", 0.5: ":"}
MK = {2.0: "o", 1.0: "s", 0.5: "^"}


def load(out):
    df = pd.read_csv(os.path.join(out, "summary.csv"))
    df["eta"] = df["eta0"] / df["d"] ** 2
    df["T05_eta"] = df["T05"] * df["eta"]
    return df


def agg(df):
    rows = []
    for (mo, k, e0, d), g in df.groupby(["model", "k", "eta0", "d"]):
        T = g["T05"].dropna().values
        rows.append(dict(model=mo, k=k, eta0=e0, d=d, n_seeds=len(g), n_reached05=len(T),
                         n_reached09=int(g["reached"].sum()),
                         T_med=np.median(T) if len(T) > len(g) / 2 else np.nan,
                         T_q25=np.percentile(T, 25) if len(T) else np.nan,
                         T_q75=np.percentile(T, 75) if len(T) else np.nan,
                         Teta_med=np.median(g["T05_eta"].dropna()) if len(T) > len(g) / 2 else np.nan,
                         m0_med=g["m0"].median()))
    return pd.DataFrame(rows)


def fit(a):
    rows = []
    for (mo, k, e0), g in a.groupby(["model", "k", "eta0"]):
        g = g.dropna(subset=["T_med"]).sort_values("d")
        if len(g) < 3:
            continue
        r = stats.linregress(np.log(g.d), np.log(g.T_med))
        rows.append(dict(model=mo, k=k, eta0=e0, n_d=len(g), d_min=g.d.min(), d_max=g.d.max(), slope=r.slope,
                         slope_se=r.stderr, intercept=r.intercept, r2=r.rvalue ** 2,
                         predicted=PRED.get((mo, k), np.nan)))
    return pd.DataFrame(rows)


def fig_T_vs_d(a, f, out):
    ks = sorted(a.k.unique())
    fig, axs = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 4.4), squeeze=False)
    for ax, k in zip(axs[0], ks):
        for mo in "BA":
            for e0 in sorted(a.eta0.unique(), reverse=True):
                g = a[(a.model == mo) & (a.k == k) & (a.eta0 == e0)].sort_values("d")
                if g.empty:
                    continue
                gg = g.dropna(subset=["T_med"])
                ax.errorbar(gg.d, gg.T_med, yerr=[gg.T_med - gg.T_q25, gg.T_q75 - gg.T_med], color=COL[mo],
                            marker=MK.get(e0, "x"), ls="none", capsize=3, ms=5, alpha=0.9)
                fr = f[(f.model == mo) & (f.k == k) & (f.eta0 == e0)]
                lab = f"{mo} $\\eta_0$={e0:g}"
                if len(fr):
                    r = fr.iloc[0]
                    dd = np.array([r.d_min, r.d_max], float)
                    ax.plot(dd, np.exp(r.intercept) * dd ** r.slope, color=COL[mo], ls=LS.get(e0, "-"), lw=1.3,
                             label=f"{lab}: slope {r.slope:.2f}$\\pm${r.slope_se:.2f} (pred {PRED[(mo, k)]:g})")
                else:
                    ax.plot([], [], color=COL[mo], marker=MK.get(e0, "x"), label=lab)
        ax.set(xscale="log", yscale="log", xlabel="d", ylabel="$T_{0.5}$ (steps, median, IQR)",
               title=f"k = {k}   ($\\eta=\\eta_0/d^2$)")
        ax.set_xticks(sorted(a.d.unique())); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figs", "T_vs_d.png"), dpi=140)
    plt.close(fig)


def fig_drift(out):
    p = os.path.join(out, "drift.csv")
    if not os.path.exists(p):
        return
    dr = pd.read_csv(p)
    ks = sorted(dr.k.unique())
    fig, axs = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 4.2), squeeze=False)
    mm = np.linspace(0.04, 0.5, 100)
    from .drift import drift_formula
    for ax, k in zip(axs[0], ks):
        for mo in "BA":
            g = dr[(dr.model == mo) & (dr.k == k)].sort_values("m")
            kw = dict(N=int(g.N.iloc[0]), gamma=float(g.gamma.iloc[0]), a=float(g.a.iloc[0]))
            ff = np.array([drift_formula(mo, k, m, **kw) for m in mm])
            ok = ff > 0
            ax.plot(mm[ok], ff[ok], color=COL[mo], lw=1.2, label=f"{mo} closed form")
            pos = g.drift_mc > 0
            ax.errorbar(g.m[pos], g.drift_mc[pos], yerr=g.se[pos], color=COL[mo], marker="o", ls="none", capsize=3,
                        ms=5, label=f"{mo} Monte Carlo ({int(g.n_batches.iloc[0])} batches)")
            if (~pos).any():
                ax.plot([], [], ls="none", label=f"{mo}: {int((~pos).sum())} MC point(s) <= 0 not shown")
        ax.set(xscale="log", yscale="log", xlabel="m", ylabel="drift  $-dL/dm$", title=f"k = {k}")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figs", "drift_vs_m.png"), dpi=140)
    plt.close(fig)


def fig_traj(df, out, k=2, e0=2.0, seed=0):
    ds = sorted(df[(df.k == k) & (df.eta0 == e0)].d.unique())
    ds = [d for d in ds if d in (8, 16, 32, 64, 128)]
    cm = plt.cm.viridis(np.linspace(0.05, 0.9, max(len(ds), 1)))
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, mo in zip(axs, "BA"):
        for c, d in zip(cm, ds):
            f = os.path.join(out, "traj", f"{mo}_k{k}_d{d}_eta{e0:g}_s{seed}.npz")
            if os.path.exists(f):
                z = np.load(f)
                ax.plot(np.maximum(z["t"], 1), z["m"], color=c, label=f"d={d}")
        ax.set(xscale="log", xlabel="step t", title=f"Model {mo}, k={k}, $\\eta_0$={e0:g}, seed {seed}")
        ax.axhline(0.5, color="gray", lw=0.6, ls=":"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    axs[0].set_ylabel("m = <w, v>")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figs", "traj_examples.png"), dpi=140)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp1")
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(args.out, "figs"), exist_ok=True)
    df = load(args.out)
    a = agg(df)
    f = fit(a)
    a.to_csv(os.path.join(args.out, "T_by_config.csv"), index=False)
    f.to_csv(os.path.join(args.out, "slopes.csv"), index=False)
    fig_T_vs_d(a, f, args.out)
    fig_drift(args.out)
    fig_traj(df, args.out)
    pd.set_option("display.width", 200)
    print(f.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
