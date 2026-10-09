"""Analysis for exp 6 (N-T exchange by readout protocol): writes results/exp6/summary.csv, tables.md, figs/exchange.png."""
import os

import numpy as np
import pandas as pd

OUT = "results/exp6"
D2 = 64 ** 2
ODE = {  # flow-time predictions, docs/preregistration.md exp 6 (steps = flow x d^2)
    "free": {16: 448, 64: 153, 256: 101}, "tied": {16: 4.21, 64: 4.07, 256: 4.03},
    "fixed01": {16: 101, 64: 48, 256: 43}, "fixed1": {16: np.inf, 64: np.inf, 256: np.inf}}
LABEL = {"fixed1": "pinned Gamma=1", "fixed01": "pinned Gamma=0.1", "free": "free (Gamma_0=0.01)", "tied": "tied (rho_0=0.01)"}
PROTOS = ["fixed1", "fixed01", "free", "tied"]


def main():
    r = pd.read_csv(os.path.join(OUT, "runs.csv"))
    ext = pd.read_csv(os.path.join(OUT, "runs_ext.csv")) if os.path.exists(os.path.join(OUT, "runs_ext.csv")) and os.path.getsize(os.path.join(OUT, "runs_ext.csv")) > 0 else None
    rows = []
    allr = r if ext is None else pd.concat([r.assign(variant="main"), ext.assign(variant="ext")])
    if ext is None:
        allr = r.assign(variant="main")
    for (p, sch, N, var), g in allr.groupby(["protocol", "scheme", "N", "variant"]):
        g = g.sort_values("seed")
        T = g["T05"].values
        Tc = np.where(g.reached05.values, T, np.nan)
        rows.append(dict(protocol=p, variant=var, scheme=sch, N=N, B=int(g.B.iloc[0]), max_steps=int(g.max_steps.iloc[0]), n=len(g),
                         n_reached=int(g.reached05.sum()), T05_seeds="/".join("%.0f" % t if t == t else "NaN" for t in T),
                         T05_median=(lambda m: m if m < g.max_steps.iloc[0] else np.nan)(np.median(np.where(g.reached05.values, T, g.max_steps.values))),
                         T05_median_censored_as_cap=np.median(np.where(g.reached05.values, T, g.max_steps.values)),
                         final_abs_m_seeds="/".join("%.3f" % x for x in g.final_abs_m),
                         readout_at_T05_seeds="/".join("%.4g" % x for x in g.readout_at_T05) if p in ("free", "tied") else "",
                         ode_steps=ODE[p][N] * D2, cpu_s=g.cpu_s.sum()))
    s = pd.DataFrame(rows)
    s["escaped"] = s.n_reached
    s["T05_ratio_to_N256_same_proto_scheme"] = np.nan
    s["ratio_med_over_ode"] = s.T05_median / s.ode_steps
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    # ratios T(16)/T(256)
    rat = []
    for p in PROTOS:
        for sch in ("tok", "B64"):
            a = s[(s.protocol == p) & (s.scheme == sch) & (s.variant == "main")].set_index("N")
            if len(a) < 3:
                continue
            if p == "free" and ext is not None:
                e = s[(s.protocol == p) & (s.scheme == sch) & (s.variant == "ext")].set_index("N")
                if len(e):
                    a = a.copy()
                    a.loc[16, "T05_median_censored_as_cap"] = e.loc[16, "T05_median_censored_as_cap"]
                    a.loc[16, "n_reached"] = e.loc[16, "n_reached"]
            if p == "fixed1":
                continue  # trapped: ratios are not defined (censored at the cap)
            rr = a.loc[16, "T05_median_censored_as_cap"] / a.loc[256, "T05_median_censored_as_cap"]
            cens = a.loc[16, "n_reached"] < a.loc[16, "n"] or a.loc[256, "n_reached"] < a.loc[256, "n"]
            ode = ODE[p][16] / ODE[p][256]
            rat.append(dict(protocol=p, scheme=sch, ratio_T16_over_T256=rr, censored_involved=cens, ode_ratio=ode,
                            ratio_T16_T64=a.loc[16, "T05_median_censored_as_cap"] / a.loc[64, "T05_median_censored_as_cap"],
                            ratio_T64_T256=a.loc[64, "T05_median_censored_as_cap"] / a.loc[256, "T05_median_censored_as_cap"]))
    pd.DataFrame(rat).to_csv(os.path.join(OUT, "ratios.csv"), index=False)
    print(s.to_string())
    print(pd.DataFrame(rat).to_string())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    fig, axs = plt.subplots(1, 4, figsize=(17, 4), sharey=False)
    col = {"tok": "#1f77b4", "B64": "#d95f02"}
    for ax, p in zip(axs, PROTOS):
        for sch, mk in (("tok", "o"), ("B64", "s")):
            g = r[(r.protocol == p) & (r.scheme == sch)]
            if g.empty:
                continue
            reached = g[g.reached05]
            ax.plot(reached.N * (1.06 if sch == "B64" else 0.94), reached.T05, mk, color=col[sch], alpha=0.5, ms=5)
            cens = g[~g.reached05]
            ax.plot(cens.N * (1.06 if sch == "B64" else 0.94), cens.max_steps, mk, mfc="none", color=col[sch], ms=7,
                    label=None)
            ss = s[(s.protocol == p) & (s.scheme == sch) & (s.variant == "main")].sort_values("N")
            ys = ss.set_index("N").T05_median_censored_as_cap.copy()
            if p == "free" and ext is not None:
                ys[16] = np.median(ext[ext.scheme == sch].T05)
            if p == "fixed1":
                ys = ys.iloc[0:0]  # trapped / censored: no line through medians
            ax.plot(ys.index, ys.values, "-", color=col[sch],
                    label=("tokens-matched N*B=4096" if sch == "tok" else "B=64") + " (median)")
        if np.isfinite(ODE[p][16]):
            Ns = np.array([16, 64, 256])
            ax.plot(Ns, [ODE[p][n] * D2 for n in Ns], "k--", label="population ODE")
        else:
            ax.text(0.5, 0.95, "ODE: trapped at all N", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        if p == "tied":
            ax.set_ylim(0.5 * 4.1 * D2, 2.0 * 4.1 * D2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([16, 64, 256])
        ax.set_xticklabels(["16", "64", "256"])
        ax.set_title(LABEL[p])
        ax.set_xlabel("context length N")
        ax.grid(alpha=0.3, which="both")
    if ext is not None:
        ax = axs[PROTOS.index("free")]
        for sch in ("tok", "B64"):
            g = ext[ext.scheme == sch]
            ax.plot(g.N * (1.06 if sch == "B64" else 0.94), np.where(g.reached05, g.T05, g.max_steps), "*", color=col[sch], ms=11,
                    mec="k", label="N=16 rerun to 2.5e6 steps (%s)" % sch)
        ax.legend(fontsize=6, loc="lower left")
    axs[0].set_ylabel("T_0.5 (steps); open = censored at cap")
    axs[1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Exp 6: N-T exchange, d=64, eta=1/d^2, m_0=d^-1/2")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "exchange.png"), dpi=130)


if __name__ == "__main__":
    main()
