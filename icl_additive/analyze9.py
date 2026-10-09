"""Analysis for exp 9 (k*=3 sanity run): writes results/exp9/summary.csv, ratios.csv, secants.csv, verdicts.csv, figs/k3.png."""
import os

import numpy as np
import pandas as pd

from .sweep9 import CELLS, DS, ETA

OUT = "results/exp9"
CELL_ODE = {"a": "tied rho0=0.01,128", "b": "tied rho0=0.01,16", "c": "pinned gamma=0.2,128"}


def fmt(x, f="%.0f"):
    return f % x if np.isfinite(x) else "NaN"


def secant(m0a, Ta, m0b, Tb):
    """kappa_eff = 2 - slope of log T vs log m0 between two points."""
    if not (np.isfinite(Ta) and np.isfinite(Tb)):
        return np.nan
    return 2.0 - (np.log(Tb) - np.log(Ta)) / (np.log(m0b) - np.log(m0a))


def main():
    r = pd.read_csv(os.path.join(OUT, "runs.csv")).sort_values(["cell", "d", "seed"])
    ode = pd.read_csv(os.path.join(OUT, "ode_predictions.csv"))
    rows, meds = [], {}
    for (cell, d), g in r.groupby(["cell", "d"]):
        cap = int(g.max_steps.iloc[0])
        reached = g.reached05.values.astype(bool)
        Tc = np.where(reached, g.T05.values, np.inf)
        m = float(np.median(Tc))
        o = ode[(ode.cell == CELL_ODE[cell].split(",")[0]) & (ode.N == int(CELL_ODE[cell].split(",")[1])) & (ode.d == d)
                & ~ode.cell.str.startswith("kappa")].iloc[0]
        ode_flow, ode_steps = float(o.tau_flow), float(o.steps_eta2e4)
        ro = g.readout_at_T05.values if cell != "c" else np.full(len(g), 0.2)
        rows.append(dict(cell=cell, protocol=CELLS[cell][0], N=int(g.N.iloc[0]), d=d, m0=float(g.m0.iloc[0]), n=len(g), n_reached=int(reached.sum()),
                         cap=cap, T05_seeds="/".join(fmt(t) for t in np.where(reached, g.T05.values, np.nan)),
                         reached_flags="/".join("Y" if x else "N" for x in reached),
                         T05_median=m if np.isfinite(m) else np.nan, flow_median=m * ETA if np.isfinite(m) else np.nan,
                         ode_flow=ode_flow, ode_steps=ode_steps, median_over_ode=m / ode_steps if np.isfinite(ode_steps) else np.nan,
                         readout_at_T05_seeds="/".join("%.4g" % x for x in ro), readout_at_T05_median=float(np.nanmedian(ro)) if reached.any() else np.nan,
                         final_abs_m_seeds="/".join("%.3f" % x for x in g.final_abs_m), cpu_s=g.cpu_s.sum()))
        meds[(cell, d)] = dict(m0=float(g.m0.iloc[0]), Tc=Tc, med=m, ode=ode_steps, cap=cap, g=g)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    # secants (medians) vs ODE
    sec = []
    for cell in "abc":
        for d1, d2 in ((8, 16), (16, 32)):
            if (cell, d1) in meds and (cell, d2) in meds:
                a, b = meds[(cell, d1)], meds[(cell, d2)]
                sec.append(dict(cell=cell, protocol=CELLS[cell][0], pair=f"{d1}-{d2}", kappa_sgd=secant(a["m0"], a["med"], b["m0"], b["med"]),
                                kappa_ode=secant(a["m0"], a["ode"], b["m0"], b["ode"])))
    sec = pd.DataFrame(sec)
    sec.to_csv(os.path.join(OUT, "secants.csv"), index=False)
    # N=16 / N=128 ratios (tied)
    rat = []
    for d in DS:
        if ("a", d) in meds and ("b", d) in meds:
            a, b = meds[("a", d)], meds[("b", d)]
            ode_r = b["ode"] / a["ode"] if np.isfinite(b["ode"]) and np.isfinite(a["ode"]) else np.inf
            pairs = [x / y for x in np.where(np.isinf(b["Tc"]), b["cap"], b["Tc"]) for y in np.where(np.isinf(a["Tc"]), a["cap"], a["Tc"])]
            rat.append(dict(d=d, ratio_median=b["med"] / a["med"] if np.isfinite(a["med"]) else np.nan, ratio_min_pair=min(pairs), ratio_max_pair=max(pairs),
                            n16_censored=int(np.isinf(b["Tc"]).sum()), n128_censored=int(np.isinf(a["Tc"]).sum()), ode_ratio=ode_r))
    rat = pd.DataFrame(rat)
    rat.to_csv(os.path.join(OUT, "ratios.csv"), index=False)
    # verdicts against the pre-registered criteria
    V = []
    ok = True
    detail = []
    for d in DS:
        x = meds[("a", d)]
        w = abs(x["med"] / x["ode"] - 1) <= 0.15
        ok &= bool(w)
        detail.append("d=%d med/ODE=%.3f %s" % (d, x["med"] / x["ode"], "ok" if w else "OUT"))
    ks = sec[sec.cell == "a"]
    for _, q in ks.iterrows():
        tgt = q.kappa_ode
        w = bool(abs(q.kappa_sgd - 6.5109 if q.pair == "8-16" else q.kappa_sgd - 6.3523) <= 0.5)
        ok &= w
        detail.append("kappa %s = %.2f (pre-reg %.2f; ODE-from-csv %.2f) %s" % (q.pair, q.kappa_sgd, 6.5109 if q.pair == "8-16" else 6.3523, tgt, "ok" if w else "OUT"))
    both_lo = bool((ks.kappa_sgd < 5).all())
    both_hi = bool((ks.kappa_sgd > 8).all())
    V.append(dict(id="P22a", passed=ok, kill_triggered=both_lo or both_hi, detail="; ".join(detail)))
    ok, detail = True, []
    tgt = {8: (1.16, 0.25), 16: (1.63, 0.25)}
    for d, (t0, tol) in tgt.items():
        q = rat[rat.d == d].iloc[0]
        w = abs(q.ratio_median - t0) <= tol
        ok &= bool(w)
        detail.append("d=%d ratio=%.2f (pre-reg %.2f +-%.2f) %s" % (d, q.ratio_median, t0, tol, "ok" if w else "OUT"))
    q = rat[rat.d == 32].iloc[0]
    w = (q.ratio_median >= 3) if np.isfinite(q.ratio_median) else True
    ok &= bool(w)
    detail.append("d=32 ratio=%s (lower bound if censored; n16 censored %d/3, n128 censored %d/3; need >=3) %s" % (
        "inf" if not np.isfinite(q.ratio_median) else "%.2f" % q.ratio_median, q.n16_censored, q.n128_censored, "ok" if w else "OUT"))
    V.append(dict(id="P22b", passed=ok, kill_triggered=False, detail="; ".join(detail)))
    ok, detail = True, []
    x = meds[("c", 8)]
    w = abs(x["med"] / x["ode"] - 1) <= 0.15
    ok &= bool(w)
    detail.append("d=8 med/ODE=%.3f %s" % (x["med"] / x["ode"], "ok" if w else "OUT"))
    for d in (16, 32):
        nc = int(np.isinf(meds[("c", d)]["Tc"]).sum())
        w = nc >= 2
        ok &= bool(w)
        detail.append("d=%d censored %d/3 (need >=2) %s" % (d, nc, "ok" if w else "OUT"))
    V.append(dict(id="P22c", passed=ok, kill_triggered=False, detail="; ".join(detail)))
    V = pd.DataFrame(V)
    V.to_csv(os.path.join(OUT, "verdicts.csv"), index=False)
    print(s.drop(columns=["protocol"]).to_string())
    print(sec.to_string())
    print(rat.to_string())
    print(V.to_string())
    print("total CPU h: %.3f" % (r.cpu_s.sum() / 3600))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, cell in zip(axs, "abc"):
        g = r[r.cell == cell]
        ok_ = g.reached05.values.astype(bool)
        ax.plot(g.m0[ok_], g.T05[ok_], "o", color="#1f77b4", alpha=0.7, label="SGD, reached")
        ax.plot(g.m0[~ok_], g.max_steps[~ok_], "o", mfc="none", color="#d95f02", ms=8, label="censored (at cap)")
        sub = s[s.cell == cell].sort_values("m0")
        ax.plot(sub.m0, np.where(sub.n_reached > sub.n / 2, sub.T05_median, np.nan), "-", color="#1f77b4", label="SGD median")
        ax.plot(sub.m0, np.where(np.isfinite(sub.ode_steps), sub.ode_steps, np.nan), "k--", label="population ODE")
        for _, q in sub[~np.isfinite(sub.ode_steps)].iterrows():
            ax.axvline(q.m0, color="gray", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sub.m0)
        ax.set_xticklabels(["%.3f\n(d=%d)" % (m, d) for m, d in zip(sub.m0, sub.d)], fontsize=8)
        ax.set_xlabel("m_0 = d^-1/2")
        ax.set_ylabel("T_0.5 (steps), eta=2e-4")
        ax.set_title("(%s) %s" % (cell, CELLS[cell][0]))
        ax.grid(alpha=0.3, which="both")
    axs[0].legend(fontsize=8)
    fig.suptitle("Exp 9: k*=3 sanity run (dotted vertical: ODE-trapped)")
    fig.tight_layout()
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    fig.savefig(os.path.join(OUT, "figs", "k3.png"), dpi=130)


if __name__ == "__main__":
    main()
