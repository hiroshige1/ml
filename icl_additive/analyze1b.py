"""Analysis + figures for experiment 1b.  Reads results/exp1b/{thr,p7,tied1,kappa,dscan}.csv.

Writes results/exp1b/{summary.csv, threshold_table.csv, p7_table.csv, kappa_fits.csv, gamma_at_T05.csv} and
figs/{threshold_grid,kappa_vs_N,P7_tradeoff}.png; prints all tables as markdown.
"""
import glob
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/exp1b"
FIG = "figs"
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"


def load():
    dfs = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "*.csv"))) if not f.endswith("summary.csv")
           and not any(f.endswith(x) for x in ("_table.csv", "_fits.csv", "gamma_at_T05.csv", "dscan_slopes.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    # kappa2 re-runs (longer step cap, same seed => same trajectory up to the old cap) supersede the censored kappa rows
    k2 = df[df.phase == "kappa2"]
    key = lambda r: (r.protocol, r.N, round(r.m0_scale, 6), r.seed)
    sup = {key(r) for r in k2.itertuples()}
    df["superseded"] = [(r.phase == "kappa" and key(r) in sup) for r in df.itertuples()]
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    return df


def md(df, floatfmt="{:.4g}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(x) if isinstance(x, (float, np.floating)) else str(x)
                                      for x in r.values) + " |")
    return "\n".join(lines)


def threshold_table(df):
    t = df[df.phase == "thr"]
    rows = []
    for (g, N, d), s in t.groupby(["gamma", "N", "d"]):
        thr = N * (4 - 2 * g) / (16 * g)
        esc = int(s.reached05.sum())
        rows.append(dict(gamma=g, N=N, d=d, m0sq=1 / d, pred_threshold_d=thr,
                         predicted="escape" if d < thr else "stuck", n=len(s), escaped=esc,
                         stuck=int(s.stuck.sum()), exceeded_0p5=int(s.exceeded05.sum()),
                         median_final_abs_m=s.final_abs_m.median(), min_final_abs_m=s.final_abs_m.min(),
                         max_final_abs_m=s.final_abs_m.max(),
                         T05_median_escaped=s[s.reached05].T05.median() if esc else float("nan"),
                         cpu_s=s.cpu_s.sum()))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "threshold_table.csv"), index=False)
    return r


def p7_table(df):
    t = df[df.phase == "p7"]
    rows = []
    for B, s in t.groupby("B"):
        T = s.max_steps.iloc[0]
        rows.append(dict(B=B, steps=T, total_samples=B * T, n=len(s), escaped=int(s.reached05.sum()),
                         stuck=int(s.stuck.sum()), exceeded_0p5=int(s.exceeded05.sum()),
                         final_abs_m_seeds=" / ".join(f"{x:.3f}" for x in s.sort_values("seed").final_abs_m),
                         max_abs_m_median=s.max_abs_m.median(), m0=s.m0.iloc[0], cpu_s=s.cpu_s.sum()))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "p7_table.csv"), index=False)
    return r


def kappa_fits(df):
    t = df[df.phase == "kappa"]
    rows = []
    for (p, N), s in t.groupby(["protocol", "N"]):
        cens = s[~s.reached05]
        ok = s[s.reached05]
        row = dict(protocol=p, N=N, B=s.B.iloc[0], max_steps=int(s.max_steps.max()), n_runs=len(s), n_reached=len(ok),
                   n_censored=len(cens), censored_by_m0=",".join(
                       f"{k:g}:{int(v)}" for k, v in cens.groupby("m0_scale").size().items()) or "-",
                   n_distinct_m0_used=ok.m0_scale.nunique())
        row.update(b=np.nan, b_se=np.nan, kappa=np.nan, kappa_se=np.nan, b_cap=np.nan, kappa_cap=np.nan)
        if len(ok) >= 4 and ok.m0_scale.nunique() >= 2:
            x, y = np.log(ok.m0.values), np.log(ok.T05.values * ok.eta.values)
            fit = stats.linregress(x, y)
            row.update(b=fit.slope, b_se=fit.stderr, kappa=2 - fit.slope, kappa_se=fit.stderr)
        # censored runs set to the cap (T >= cap, so this is a LOWER BOUND on each censored T; slope is still biased)
        if len(cens) and len(s) >= 4:
            Tfill = np.where(s.reached05, s.T05, s.max_steps)
            fit = stats.linregress(np.log(s.m0.values), np.log(Tfill * s.eta.values))
            row.update(b_cap=fit.slope, kappa_cap=2 - fit.slope)
        row["median_logTeta_by_m0"] = "; ".join(
            f"{k:g}:{np.median(v.T05 * v.eta):.3g}({int(v.reached05.sum())}/{len(v)})"
            for k, v in s.groupby("m0_scale"))
        rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "kappa_fits.csv"), index=False)
    return r


def gamma_star(m, N):
    g = m ** 2
    V = 1 + 8 * m ** 2 + 6 * m ** 4
    return g * g / ((1 - 1 / N) * g * g + V / N)


def gamma_table(df):
    t = df[(df.phase.isin(["kappa", "dscan"])) & df.reached05 & df.protocol.str.startswith("free")]
    rows = []
    for (p, N), s in t.groupby(["protocol", "N"]):
        rows.append(dict(protocol=p, N=N, n=len(s), Gamma_at_T05_median=s.gamma_at_T05.median(),
                         Gamma_at_T05_min=s.gamma_at_T05.min(), Gamma_at_T05_max=s.gamma_at_T05.max(),
                         Gamma_star_0p5=gamma_star(0.5, N)))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "gamma_at_T05.csv"), index=False)
    return r


# ----------------------------------------------------------------------------- figures
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def fig_threshold(tt):
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    Ns, ds = [32, 128, 512], [16, 32, 64]
    for ax, g in zip(axs, (1.0, 0.1)):
        Z = np.full((3, 3), np.nan)
        for i, d in enumerate(ds):
            for j, N in enumerate(Ns):
                r = tt[(tt.gamma == g) & (tt.N == N) & (tt.d == d)]
                if len(r):
                    Z[i, j] = r.escaped.iloc[0] / r.n.iloc[0]
        xe = np.log2(Ns)
        xe = np.r_[xe - 1, xe[-1] + 1]
        ye = np.log2(ds)
        ye = np.r_[ye - 0.5, ye[-1] + 0.5]
        im = ax.pcolormesh(xe, ye, Z, cmap="Blues", vmin=0, vmax=1, edgecolors="#fcfcfb", linewidth=2)
        for i, d in enumerate(ds):
            for j, N in enumerate(Ns):
                r = tt[(tt.gamma == g) & (tt.N == N) & (tt.d == d)]
                if len(r):
                    z = Z[i, j]
                    ax.text(np.log2(N), np.log2(d), f"{int(r.escaped.iloc[0])}/{int(r.n.iloc[0])}", ha="center",
                            va="center", fontsize=11, color="white" if z > 0.55 else INK)
        xs = np.linspace(xe[0], xe[-1], 100)
        ystar = xs + math.log2((4 - 2 * g) / (16 * g))
        ax.plot(xs, ystar, color=ORANGE, lw=2, label=r"predicted threshold $d^*=N(4-2\gamma)/(16\gamma)$")
        ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ye[0], ye[-1])
        ax.set_xticks(np.log2(Ns)); ax.set_xticklabels(Ns)
        ax.set_yticks(np.log2(ds)); ax.set_yticklabels(ds)
        ax.set_xlabel("context length N"); ax.set_ylabel("dimension d")
        ax.set_title(f"fixed $\\gamma$ = {g:g}: seeds reaching |m| = 0.5 (3e5 steps)", fontsize=10, loc="left")
        style(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8, frameon=False, labelcolor=INK)
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02)
    cb.set_label("escape fraction", color=MUTED)
    fig.savefig(os.path.join(FIG, "threshold_grid.png"), dpi=160)
    plt.close(fig)


def fig_kappa(kf):
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    spec = [("free1", BLUE, "o", r"free $\Gamma$, $\eta_\Gamma=\eta$"), ("free10", ORANGE, "s", r"free $\Gamma$, $\eta_\Gamma=10\eta$"),
            ("tied", AQUA, "^", r"tied $\Gamma=\|w\|^2$")]
    Ns = sorted(kf.N.unique())
    for p, col, mk, lab in spec:
        r = kf[kf.protocol == p].sort_values("N")
        good = r[np.isfinite(r.kappa)]
        ax.errorbar(good.N, good.kappa, yerr=good.kappa_se, color=col, marker=mk, ms=7, lw=1.6, capsize=3, label=lab)
        bad = r[~np.isfinite(r.kappa)]
        for _, q in bad.iterrows():
            ax.annotate("no fit\n(%d/%d reached)" % (q.n_reached, q.n_runs), (q.N, 2.2 if p == "free1" else (2.9 if p == "free10" else 3.6)),
                        fontsize=7, color=col, ha="center")
        # kappa from fits with censored runs set to the cap: lower-bound style indicator, open markers
        cap = r[np.isfinite(r.kappa_cap)]
        ax.plot(cap.N, cap.kappa_cap, ls="none", marker=mk, mfc="none", color=col, ms=7, alpha=0.7)
    for yv, lab in ((4, "4: $\\kappa=2k^*$"), (8, "8: $\\kappa=4k^*$")):
        ax.axhline(yv, color=MUTED, lw=1, ls="--")
        ax.text(Ns[0] * 0.8, yv + 0.12, lab, fontsize=8, color=MUTED)
    ax.set_xscale("log", base=2); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
    ax.set_xlabel("context length N (d = 32)"); ax.set_ylabel(r"$\kappa_{\rm eff}=2-b$")
    ax.set_title("filled: fit on reached runs (±SE); open: censored runs set to cap", fontsize=9, loc="left")
    style(ax)
    ax.set_ylim(3.6, 8.6)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right", bbox_to_anchor=(1.0, 0.72))
    fig.savefig(os.path.join(FIG, "kappa_vs_N.png"), dpi=160)
    plt.close(fig)


def fig_p7(df):
    t = df[df.phase == "p7"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    cols = {32: BLUE, 256: ORANGE, 1024: AQUA}
    for B, s in t.groupby("B"):
        tot = s.B * s.steps
        ax.plot(tot, s.final_abs_m, "o", color=cols[B], ms=7, alpha=0.8)
        ax.plot(tot.iloc[0], s.final_abs_m.median(), "_", color=cols[B], ms=22, mew=2)
        ax.annotate(f"B={B}", (tot.iloc[0], s.final_abs_m.max()), textcoords="offset points", xytext=(8, 6), fontsize=8,
                    color=INK)
    m0 = t.m0.iloc[0]
    ax.axhline(m0, color=MUTED, lw=1, ls=":"); ax.text(0.99, m0 * 1.05, "$m_0$", transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color=MUTED)
    ax.axhline(2 * m0, color=MUTED, lw=1, ls="--"); ax.text(0.99, 2 * m0 * 1.05, "$2m_0$ (stuck threshold)", transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color=MUTED)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--"); ax.text(0.99, 0.52, "0.5 (escape)", transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color=MUTED)
    ax.set_xscale("log"); ax.set_ylim(0, 0.6); ax.set_xlim(6e6, 5e7)
    ax.set_xticks([1e7, 2e7, 3e7]); ax.set_xticklabels(["1e7", "2e7", "3e7"]); ax.minorticks_off()
    ax.set_xlabel("total samples  B·T (prompts)"); ax.set_ylabel("final |m| (3 seeds, bar = median)")
    ax.set_title(r"P7: $\gamma=1$, N=128, d=64: no escape at any B (total samples ×1, ×2.7, ×3.2)", fontsize=9, loc="left")
    style(ax)
    fig.savefig(os.path.join(FIG, "P7_tradeoff.png"), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    full = load()
    df = full[~full.superseded].copy()
    df.loc[df.phase == "kappa2", "phase"] = "kappa"
    df.loc[df.phase == "bctl", "protocol"] = "free1_B8ctl"  # B-confound control, fitted like a kappa cell
    df.loc[df.phase == "bctl", "phase"] = "kappa"
    pd.set_option("display.width", 250)
    tt = threshold_table(df)
    print("## threshold (P2/P3)\n", md(tt))
    p7 = p7_table(df)
    print("\n## P7\n", md(p7))
    t1 = df[df.phase == "tied1"]
    if len(t1):
        print("\n## tied rho0=1\n", md(t1.groupby("d").agg(n=("seed", "count"), escaped=("reached05", "sum"),
                                                          stuck=("stuck", "sum"), median_final=("final_abs_m", "median"),
                                                          T05_med=("T05", "median"), final_rho=("final_gamma", "median")).reset_index()))
    if (df.phase == "kappa").any():
        kf = kappa_fits(df)
        print("\n## kappa\n", md(kf))
        print("\n## Gamma at T05\n", md(gamma_table(df)))
        fig_kappa(kf)
    ds = df[df.phase == "dscan"]
    if len(ds):
        rows = []
        for N, g in ds.groupby("N"):
            med = g.groupby("d").agg(T05_med=("T05", "median"), n_reached=("reached05", "sum"), n=("seed", "count"),
                                     Gamma_T05=("gamma_at_T05", "median"), eta=("eta", "first"))
            med["T05_eta"] = med.T05_med * med.eta
            ok = med[med.n_reached == med.n]
            fit = stats.linregress(np.log(ok.index.values), np.log(ok.T05_med.values))
            print(f"\n## dscan N={N}: slope of log median T05 vs log d = {fit.slope:.3f} +- {fit.stderr:.3f} "
                  f"(points {list(ok.index)}; all runs reached: {len(ok) == len(med)})")
            print(md(med.reset_index()))
            rows.append(dict(N=N, slope=fit.slope, se=fit.stderr, Gamma_star_0p5=gamma_star(0.5, N)))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "dscan_slopes.csv"), index=False)
    fig_threshold(tt)
    fig_p7(df)
    tot = full.cpu_s.sum()
    print(f"\nTOTAL CPU (all rows incl. superseded censored runs, excl. ~0.1 h of untracked pilots): {tot:.0f} s = "
          f"{tot / 3600:.3f} h over {len(full)} runs; by phase:")
    print(full.groupby("phase").agg(runs=("cpu_s", "count"), cpu_h=("cpu_s", lambda x: x.sum() / 3600)))


if __name__ == "__main__":
    main()
