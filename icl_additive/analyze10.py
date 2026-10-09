"""Analysis for exp 10 (three regimes at d=128): writes results/exp10/summary.csv, verdicts.csv, tables.md, figs/regimes_d128.png."""
import os
import re

import numpy as np
import pandas as pd

from .sweep10 import CELLS, D, ETA

OUT = "results/exp10"
EXP6 = "results/exp6"
PREREG_D64_FLOW = 50.1  # value quoted in the pre-registration (this is exp 7 cell a, eta_Gamma = 10 eta)
PREREG_KAPPA = 7.4


def parse_ode(path=os.path.join(OUT, "ode_predictions.txt")):
    """-> {(cell, N): (tau_flow, rho_T or nan)}; trapped -> inf."""
    ode = {}
    for line in open(path):
        m = re.match(r"pinned g=([\d.]+) N=(\d+): (TRAPPED|tau=([\d.e+-]+))", line)
        if m:
            cell = "a" if float(m.group(1)) == 1.0 else "b"
            ode[(cell, int(m.group(2)))] = (float("inf") if m.group(3) == "TRAPPED" else float(m.group(4)), float("nan"))
            continue
        m = re.match(r"tied rho0=0.01 N=(\d+): tau=([\d.e+-]+) .*rho_T=([\d.e+-]+)", line)
        if m:
            ode[("c", int(m.group(1)))] = (float(m.group(2)), float(m.group(3)))
            continue
        m = re.match(r"free G0=0.01 r=1 N=(\d+): tau=([\d.e+-]+)", line)
        if m:
            ode[("d", int(m.group(1)))] = (float(m.group(2)), float("nan"))
    return ode


def secant(tau_hi, tau_lo, ratio_m0=2 ** -0.5):
    """kappa_eff = 2 - d ln tau / d ln m0 between d=64 (tau_lo) and d=128 (tau_hi); m0(128)/m0(64) = 2^-1/2."""
    return 2.0 - np.log(tau_hi / tau_lo) / np.log(ratio_m0)


def fnum(x, f="%.0f"):
    return f % x if np.isfinite(x) else "inf"


def main():
    r = pd.read_csv(os.path.join(OUT, "runs.csv")).sort_values(["cell", "N", "seed"])
    ode = parse_ode()
    rows, M = [], {}
    for (cell, N), g in r.groupby(["cell", "N"]):
        reached = g.reached05.values.astype(bool)
        Tc = np.where(reached, g.T05.values, np.inf)
        med = float(np.median(Tc))
        otau, orho = ode[(cell, N)]
        osteps = otau / ETA
        ro = g.readout_at_T05.values
        rows.append(dict(cell=cell, protocol=CELLS[cell][0], N=N, n=len(g), n_expected=len(CELLS[cell][3]), cap=int(g.max_steps.iloc[0]),
                         n_reached=int(reached.sum()), T05_seeds="/".join(fnum(t) if ok else "CENS" for t, ok in zip(g.T05.values, reached)),
                         T05_median=med, flow_median=med * ETA, ode_flow=otau, ode_steps=osteps,
                         median_over_ode=med / osteps if np.isfinite(med) and np.isfinite(osteps) else np.nan,
                         readout_at_T05_median=float(np.nanmedian(ro)) if reached.any() and np.isfinite(ro).any() else np.nan,
                         readout_seeds="/".join("%.4g" % x if ok else "-" for x, ok in zip(ro, reached)),
                         ode_rho_T=orho, final_abs_m_seeds="/".join("%.3f" % x for x in g.final_abs_m),
                         max_abs_m_seeds="/".join("%.3f" % x for x in g.max_abs_m), cpu_s=g.cpu_s.sum()))
        M[(cell, N)] = dict(Tc=Tc, med=med, osteps=osteps, otau=otau, n=len(g), g=g)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    def within(cell, N, tol=0.15):
        x = M.get((cell, N))
        if x is None or not np.isfinite(x["med"]) or not np.isfinite(x["osteps"]):
            return False, np.nan
        q = x["med"] / x["osteps"]
        return abs(q - 1) <= tol, q

    def ncens(cell, N):
        x = M.get((cell, N))
        return (int(np.isinf(x["Tc"]).sum()), x["n"]) if x is not None else (None, 0)

    V, lines = [], []
    # ---- P23a
    det, ok = [], True
    for N in (16, 64, 256):
        c, n = ncens("a", N)
        ok &= (n == 2 and c == 2)
        det.append("N=%d censored %s/%s" % (N, c, n))
    c256, n256 = ncens("a", 256)
    V.append(dict(id="P23a", passed=bool(ok), default_triggered=bool(n256 and c256 == 0), detail="; ".join(det)))
    # ---- P23b
    c16, n16 = ncens("b", 16)
    okb = bool(n16 == 3 and c16 >= 2)
    det = ["N=16 censored %s/%s (need >=2/3)" % (c16, n16)]
    for N in (64, 256):
        w, q = within("b", N)
        okb &= bool(w)
        det.append("N=%d median/ODE=%s %s" % (N, "%.3f" % q if np.isfinite(q) else "nan(censored)", "ok" if w else "OUT"))
    V.append(dict(id="P23b", passed=okb, default_triggered=bool(n16 and c16 == 0), detail="; ".join(det)))
    # ---- P23c
    okc, det = True, []
    for N in (16, 64, 256):
        w, q = within("c", N)
        okc &= bool(w)
        det.append("N=%d median/ODE=%s %s" % (N, "%.3f" % q if np.isfinite(q) else "nan", "ok" if w else "OUT"))
    tied_ratio = M[("c", 16)]["med"] / M[("c", 256)]["med"] if ("c", 16) in M and ("c", 256) in M else np.nan
    ode_ratio = ode[("c", 16)][0] / ode[("c", 256)][0]
    wr = bool(np.isfinite(tied_ratio) and abs(tied_ratio - 1.09) <= 0.15)
    okc &= wr
    det.append("ratio T(16)/T(256)=%.3f (pre-reg 1.09 +-0.15; ODE %.3f) %s" % (tied_ratio, ode_ratio, "ok" if wr else "OUT"))
    V.append(dict(id="P23c", passed=bool(okc), default_triggered=bool(not np.isfinite(tied_ratio) or tied_ratio > 2), detail="; ".join(det)))
    # ---- P23d
    sec = {}
    e6 = pd.read_csv(os.path.join(EXP6, "runs.csv"))
    e6 = e6[(e6.protocol == "free") & (e6.scheme == "B64") & (e6.N == 256) & (e6.d == 64)]
    e6_med = float(np.median(e6.T05.values))
    e6_flow = e6_med / 4096.0
    e6s = pd.read_csv(os.path.join(EXP6, "summary.csv"))
    e6_ode_steps = float(e6s[(e6s.protocol == "free") & (e6s.scheme == "B64") & (e6s.N == 256)].ode_steps.iloc[0])
    e6_ode_flow = e6_ode_steps / 4096.0
    x = M.get(("d", 256))
    w, q = within("d", 256)
    nc, nd = ncens("d", 256)
    sec["ode_prereg_baseline"] = secant(ode[("d", 256)][0], PREREG_D64_FLOW)
    sec["ode_exp6_baseline"] = secant(ode[("d", 256)][0], e6_ode_flow)
    sec["sgd_exp6_baseline"] = secant(x["med"] * ETA, e6_flow) if x is not None and np.isfinite(x["med"]) else np.nan
    sec["sgd_prereg_baseline"] = secant(x["med"] * ETA, PREREG_D64_FLOW) if x is not None and np.isfinite(x["med"]) else np.nan
    oks = bool(np.isfinite(sec["sgd_exp6_baseline"]) and abs(sec["sgd_exp6_baseline"] - PREREG_KAPPA) <= 0.5)
    default = bool(np.isfinite(sec["sgd_exp6_baseline"]) and (sec["sgd_exp6_baseline"] <= 4.5 or sec["sgd_exp6_baseline"] >= 8.5))
    det = ["n seeds run %d/2, censored %s" % (nd, nc), "median/ODE=%s %s" % ("%.3f" % q if np.isfinite(q) else "nan", "ok" if w else "OUT"),
           "SGD secant (exp 6 d=64 median %.1f flow) = %s (need 7.4 +-0.5) %s" % (e6_flow, "%.2f" % sec["sgd_exp6_baseline"], "ok" if oks else "OUT")]
    V.append(dict(id="P23d", passed=bool(w and oks), default_triggered=default, detail="; ".join(det)))
    V = pd.DataFrame(V)
    V.to_csv(os.path.join(OUT, "verdicts.csv"), index=False)
    pd.DataFrame([dict(k=k, kappa=v) for k, v in sec.items()] + [dict(k="exp6_sgd_median_steps_d64", kappa=e6_med),
                 dict(k="exp6_sgd_median_flow_d64", kappa=e6_flow), dict(k="exp6_ode_flow_d64", kappa=e6_ode_flow)]).to_csv(
        os.path.join(OUT, "secants.csv"), index=False)

    # ---- markdown tables
    L = ["| cell | N | T_0.5 per seed (steps) | reached | median steps | median flow T*eta | ODE flow | ODE steps | median/ODE | readout at T_0.5 (seeds) | final \\|m\\| |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, q in s.iterrows():
        L.append("| %s | %d | %s | %d/%d | %s | %s | %s | %s | %s | %s | %s |" % (
            q.cell + " " + q.protocol, q.N, q.T05_seeds, q.n_reached, q.n, fnum(q.T05_median), fnum(q.flow_median, "%.1f"),
            fnum(q.ode_flow, "%.1f") if np.isfinite(q.ode_flow) else "trapped", fnum(q.ode_steps, "%.3g") if np.isfinite(q.ode_steps) else "trapped",
            "%.3f" % q.median_over_ode if np.isfinite(q.median_over_ode) else "-", q.readout_seeds if q.cell in "cd" else "-", q.final_abs_m_seeds))
    L += ["", "Censoring (no |m|=0.5 within cap):", "", "| cell | N=16 | N=64 | N=256 | cap |", "|---|---|---|---|---|"]
    for cell in "abcd":
        ent = []
        for N in (16, 64, 256):
            c, n = ncens(cell, N)
            ent.append("-" if not n else "%d/%d" % (c, n))
        L.append("| %s | %s | %s |" % (cell + " " + CELLS[cell][0], " | ".join(ent), CELLS[cell][4]))
    L += ["", "Tied N-ratio T(16)/T(256) = %.3f (ODE %.3f, pre-reg 1.09); T(16)/T(64) = %.3f, T(64)/T(256) = %.3f" % (
        tied_ratio, ode_ratio, M[("c", 16)]["med"] / M[("c", 64)]["med"], M[("c", 64)]["med"] / M[("c", 256)]["med"]), ""]
    L += ["Free secant kappa = 2 - ln(tau128/tau64)/ln(2^-1/2):", "",
          "| numerator (d=128) | denominator (d=64) | kappa |", "|---|---|---|",
          "| ODE 327.9 | ODE 50.1 (pre-registered baseline) | %.2f |" % sec["ode_prereg_baseline"],
          "| ODE 327.9 | ODE %.1f (exp 6 free eta_Gamma=eta, N=256, B=64) | %.2f |" % (e6_ode_flow, sec["ode_exp6_baseline"]),
          "| SGD median %s flow | SGD exp 6 median %.1f flow | %s |" % (fnum(x["med"] * ETA, "%.1f") if x else "-", e6_flow, "%.2f" % sec["sgd_exp6_baseline"]),
          "| SGD median | 50.1 (pre-registered baseline) | %s |" % ("%.2f" % sec["sgd_prereg_baseline"]), ""]
    L += ["Verdict detail:", ""] + ["- %s: passed=%s default_triggered=%s; %s" % (v.id, v.passed, v.default_triggered, v.detail) for v in V.itertuples()]
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("total CPU h: %.3f" % (r.cpu_s.sum() / 3600))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, cell in zip(axs, "abcd"):
        sub = s[s.cell == cell]
        g = r[r.cell == cell]
        ok_ = g.reached05.values.astype(bool)
        ax.plot(g.N[ok_], g.T05[ok_], "o", color="#1f77b4", alpha=0.7, label="SGD, reached")
        ax.plot(g.N[~ok_], g.max_steps[~ok_], "o", mfc="none", color="#d95f02", ms=8, label="censored (at cap)")
        fin = sub[np.isfinite(sub.ode_steps)]
        ax.plot(fin.N, fin.ode_steps, "k--", marker="x", label="population ODE")
        for _, q in sub[~np.isfinite(sub.ode_steps)].iterrows():
            ax.axvline(q.N, color="gray", ls=":", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sorted(sub.N))
        ax.set_xticklabels([str(n) for n in sorted(sub.N)])
        ax.set_xlabel("N")
        ax.set_ylabel("T_0.5 (steps), eta=1/128^2")
        ax.set_title("(%s) %s" % (cell, CELLS[cell][0]))
        ax.grid(alpha=0.3, which="both")
    axs[0].legend(fontsize=8)
    fig.suptitle("Exp 10: three regimes at d=128 (dotted vertical: ODE-trapped)")
    fig.tight_layout()
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    fig.savefig(os.path.join(OUT, "figs", "regimes_d128.png"), dpi=130)


if __name__ == "__main__":
    main()
