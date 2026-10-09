"""Analysis for exp 7 (controls requested by review 2): writes results/exp7/summary.csv, tables.md, figs/controls.png."""
import itertools
import os

import numpy as np
import pandas as pd

from .sweep7 import CELLS

OUT = "results/exp7"
D2 = 64 ** 2
ODE = {  # flow times (steps = flow x 4096), docs/preregistration.md exp 7
    "a": {16: 447, 64: 122, 256: 50.1}, "b": {16: 363, 64: 66.9, 256: 40.1}, "c": {16: 8.13, 64: 4.76, 256: 4.34},
    "d": {16: np.inf, 64: 7.67, 256: 5.2}, "e": {16: 421, 64: 407, 256: 403}}
F_ODE_FLOW_N64 = 48.0  # exp 6 pinned gamma=0.1, N=64, B=64, eta=1/d^2: ODE 48 flow (196608 steps)


def fmt(x, f="%.0f"):
    return f % x if np.isfinite(x) else "NaN"


def main():
    r = pd.read_csv(os.path.join(OUT, "runs.csv")).sort_values(["cell", "N", "B", "seed"])
    r["flow"] = r.steps * r.eta * D2  # flow time in ODE units (= steps/4096 at eta=1/d^2)
    rows = []
    med = {}
    for (cell, N, B), g in r.groupby(["cell", "N", "B"]):
        cap = int(g.max_steps.iloc[0])
        reached = g.reached05.values.astype(bool)
        Tc = np.where(reached, g.T05.values, np.inf)
        Tcap = np.where(reached, g.T05.values, cap)  # censored counted at the cap (lower bound)
        m = np.median(Tc)
        eta = g.eta.iloc[0]
        ode_flow = ODE[cell][N] if cell != "f" else (F_ODE_FLOW_N64 if N == 64 else np.nan)
        rows.append(dict(cell=cell, protocol=CELLS[cell][0], N=N, B=B, eta=eta, cap=cap, n=len(g), n_reached=int(reached.sum()),
                         T05_seeds="/".join(fmt(t) for t in np.where(reached, g.T05.values, np.nan)),
                         T05_median=m if np.isfinite(m) else np.nan,
                         flow_seeds="/".join(fmt(t * eta * D2, "%.1f") for t in np.where(reached, g.T05.values, np.nan)),
                         flow_median=m * eta * D2 if np.isfinite(m) else np.nan,
                         T_eta_median=m * eta if np.isfinite(m) else np.nan,
                         reached_flags="/".join("Y" if x else "N" for x in reached),
                         final_abs_m_seeds="/".join("%.3f" % x for x in g.final_abs_m),
                         readout_at_T05_seeds="/".join("%.4g" % x for x in g.readout_at_T05) if cell != "e" and cell != "f" else "",
                         final_readout_seeds="/".join("%.4g" % x for x in g.final_readout) if cell != "e" and cell != "f" else "",
                         ode_flow=ode_flow, ode_steps=ode_flow * D2 if cell != "f" else ode_flow * D2 * 64 / B,
                         cpu_s=g.cpu_s.sum()))
        med[(cell, N, B)] = (Tc, Tcap, cap)
    s = pd.DataFrame(rows)
    s["median_over_ode"] = s.T05_median / s.ode_steps
    # ratios T(16)/T(256), min-max over the 3x3 seed pairings
    rat = []
    for cell in "abcde":
        a, b = med.get((cell, 16, 64)), med.get((cell, 256, 64))
        if a is None or b is None:
            continue
        pairs = [x / y for x in a[1] for y in b[1]]  # censored at cap -> lower bound for the numerator
        cens16 = bool(np.isinf(a[0]).any())
        ode = ODE[cell][16] / ODE[cell][256]
        rat.append(dict(cell=cell, protocol=CELLS[cell][0], ratio_median_of_medians=np.median(a[1]) / np.median(b[1]),
                        ratio_min=min(pairs), ratio_max=max(pairs), n16_censored=int(np.isinf(a[0]).sum()),
                        n256_censored=int(np.isinf(b[0]).sum()), lower_bound_only=cens16 or bool(np.isinf(b[0]).any()), ode_ratio=ode))
    rat = pd.DataFrame(rat)
    # (f): B invariance, reference = exp 6 pinned 0.1 N=64 B=64 (same seeds, same code, eta=1/d^2)
    f6 = pd.read_csv("results/exp6/runs.csv")
    f6 = f6[(f6.protocol == "fixed01") & (f6.N == 64) & (f6.scheme == "B64")].sort_values("seed")
    base_steps = np.median(f6.T05)
    fr = [dict(B=64, eta=1 / D2, source="exp6 (same seeds)", T05_seeds="/".join(fmt(t) for t in f6.T05), T05_median=base_steps,
               flow_median=base_steps / D2, T_eta_median=base_steps / D2, steps_ratio_to_B64=1.0, flow_ratio_to_B64=1.0, reached="3/3")]
    for B in (16, 256):
        row = s[(s.cell == "f") & (s.B == B)]
        if row.empty:
            continue
        row = row.iloc[0]
        fr.append(dict(B=B, eta=row.eta, source="exp7", T05_seeds=row.T05_seeds, T05_median=row.T05_median, flow_median=row.flow_median,
                       T_eta_median=row.T_eta_median, steps_ratio_to_B64=row.T05_median / base_steps,
                       flow_ratio_to_B64=row.flow_median / (base_steps / D2), reached="%d/3" % row.n_reached))
    fr = pd.DataFrame(fr).sort_values("B")
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    rat.to_csv(os.path.join(OUT, "ratios.csv"), index=False)
    fr.to_csv(os.path.join(OUT, "f_invariance.csv"), index=False)
    print(s.drop(columns=["protocol"]).to_string())
    print(rat.to_string())
    print(fr.to_string())
    print("total CPU h: %.3f" % (r.cpu_s.sum() / 3600))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    Ns = [16, 64, 256]
    for ax, cell in zip(axs.flat[:5], "abcde"):
        g = r[r.cell == cell]
        for N in Ns:
            gg = g[g.N == N]
            x = N * np.array([0.93, 1.0, 1.07])[:len(gg)]
            ok = gg.reached05.values.astype(bool)
            ax.plot(x[ok], gg.T05.values[ok], "o", color="#1f77b4", alpha=0.7)
            ax.plot(x[~ok], gg.max_steps.values[~ok], "o", mfc="none", color="#d95f02", ms=8)
        ms = s[s.cell == cell].sort_values("N")
        ax.plot(ms.N, np.where(ms.n_reached == ms.n, ms.T05_median, np.nan), "-", color="#1f77b4", label="measured median")
        o = np.array([ODE[cell][n] for n in Ns]) * D2
        ax.plot(Ns, np.where(np.isfinite(o), o, np.nan), "k--", label="population ODE")
        if cell == "d":
            ax.text(0.5, 0.05, "ODE: trapped at N=16", transform=ax.transAxes, ha="center", fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(Ns)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_title("(%s) %s" % (cell, CELLS[cell][0]))
        ax.set_xlabel("context length N")
        ax.set_ylabel("T_0.5 (steps); open = censored at cap")
        ax.grid(alpha=0.3, which="both")
    axs.flat[0].legend(fontsize=8)
    ax = axs.flat[5]
    gf = r[r.cell == "f"]
    ax.plot(f6.shape[0] * [64], f6.T05 / D2, "s", color="gray", label="exp 6, B=64 (eta=1/d^2)")
    for B in (16, 256):
        gg = gf[gf.B == B]
        ax.plot([B] * len(gg), gg.flow, "o", color="#1f77b4", label="exp 7 (eta=(B/64)/d^2)" if B == 16 else None)
    ax.axhline(F_ODE_FLOW_N64, color="k", ls="--", label="ODE flow (48)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 64, 256])
    ax.set_xticklabels(["16", "64", "256"])
    ax.set_xlabel("batch size B (N=64, pinned gamma=0.1)")
    ax.set_ylabel("flow time T*eta*d^2 (= steps/4096 at B=64)")
    ax.set_title("(f) flow-time invariance across B")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle("Exp 7: controls from review 2 (d=64, m_0=d^-1/2, sigma_2)")
    fig.tight_layout()
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    fig.savefig(os.path.join(OUT, "figs", "controls.png"), dpi=130)


if __name__ == "__main__":
    main()
