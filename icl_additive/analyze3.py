"""Analysis + figures for experiment 3.  Usage: python -m icl_additive.analyze3 [--dir results/exp3/main] [--figs results/exp3/figs]

Prints markdown tables (also written to <dir>/analysis_stdout.txt) and writes per-seed CSVs and the four figures.
"""
import argparse
import glob
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erf

COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]  # categorical slots 1,2,3,4,7 (reference palette)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
                     "figure.dpi": 110})


def load(d):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "traj", "seed*.npz"))):
        s = int(os.path.basename(f)[4:-4])
        z = np.load(f)
        runs[s] = {k: z[k] for k in z.files}
    return runs


def first_cross(t, x, thr, below=True):
    idx = np.where(x < thr)[0] if below else np.where(x >= thr)[0]
    return float(t[idx[0]]) if len(idx) else float("nan")


def progress_time(t, f, phi, tail=10):
    """First t with f <= f(0) - phi (f(0) - F), F = mean of the last `tail` logged values (plateau)."""
    f0, F = f[0], float(np.mean(f[-tail:]))
    if f0 - F < 0.1:  # nothing was learned: progress time undefined
        return float("nan"), F
    return first_cross(t, f, f0 - phi * (f0 - F)), F


def acc_theory(s2, theta):
    """P(chi2_1 s2 < theta) for Gaussian prediction error of variance s2."""
    return erf(np.sqrt(theta / 2.0) / np.sqrt(s2))


def assignments(m, thr=0.5):
    """m (T,M,P) -> assigned teacher per (t,j): argmax_p |m| if max >= thr else -1."""
    a = np.abs(m)
    return np.where(a.max(axis=2) >= thr, a.argmax(axis=2), -1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/exp3/main")
    ap.add_argument("--figs", default=None)
    a = ap.parse_args(argv)
    figs = a.figs or os.path.join(os.path.dirname(a.dir.rstrip("/")), "figs")
    os.makedirs(figs, exist_ok=True)
    runs = load(a.dir)
    seeds = sorted(runs)
    meta = pd.read_csv(os.path.join(a.dir, "runs.csv")).set_index("seed")
    gamma = float(meta["gamma"].iloc[0])
    M = int(meta["M"].iloc[0])
    lines = []

    def P(*x):
        s = " ".join(str(y) for y in x)
        print(s)
        lines.append(s)

    P(f"# analysis of {a.dir}  (gamma={gamma}, M={M}, seeds {seeds})")
    # ---------------------------------------------------------------- 1. times
    rows = []
    for s in seeds:
        r = runs[s]
        t = r["t"]
        al = np.abs(r["m"]).max(axis=1)  # (T,P)
        T1, T2 = first_cross(t, al[:, 0], 0.5, below=False), first_cross(t, al[:, 1], 0.5, below=False)
        T12_spec = first_cross(t, r["mse_E12"], 0.5)
        T12_spec_h = first_cross(t, r["hmse_E12"], 0.5)
        row = dict(seed=s, steps=int(meta.loc[s, "steps"]), stop=meta.loc[s, "stop_reason"], T1=T1, T2=T2,
                   T12_spec=T12_spec, T12_spec_h=T12_spec_h,
                   max_T1T2=np.nanmax([T1, T2]) if not (math.isnan(T1) or math.isnan(T2)) else float("nan"))
        for phi in (0.5, 0.9):
            t1, F1 = progress_time(t, r["hmse_E1"], phi)
            t2, F2 = progress_time(t, r["hmse_E2"], phi)
            t12, F12 = progress_time(t, r["hmse_E12"], phi)
            tag = f"{int(phi * 100)}"
            row.update({f"Tm1_{tag}": t1, f"Tm2_{tag}": t2, f"Tm12_{tag}": t12,
                        f"max_Tm_{tag}": max(t1, t2) if not (math.isnan(t1) or math.isnan(t2)) else float("nan")})
            if tag == "90":
                row.update(F1=F1, F2=F2, F12=F12)
        # fixed reachable thresholds (one neuron on a skill gives MSE 0.81 at Gamma=0.1): E_p < 0.9, E12 < 1.8 (= sum)
        a1, a2 = first_cross(t, r["hmse_E1"], 0.9), first_cross(t, r["hmse_E2"], 0.9)
        row.update(T1_mse09=a1, T2_mse09=a2, T12_sum18=first_cross(t, r["hmse_E12"], 1.8),
                   T12_both=max(a1, a2) if not (math.isnan(a1) or math.isnan(a2)) else float("nan"))
        row.update(min_E1=float(r["hmse_E1"].min()), min_E2=float(r["hmse_E2"].min()), min_E12=float(r["hmse_E12"].min()))
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.dir, "times.csv"), index=False)
    P("\n## Times (steps). T_p: first log with max_j |m_jp| >= 0.5.  T12_spec: first log with MSE_E12 < 0.5 (spec; 1-query set / 17-query).")
    P(T[["seed", "steps", "stop", "T1", "T2", "T12_spec", "T12_spec_h"]].to_string(index=False))
    P("\n## MSE-progress times: Tm_p^phi = first log where hMSE_p has covered fraction phi of the way from its t=0 value to its final plateau"
      " (mean of last 10 logs); the pair uses the same rule on E12 (independent set). NaN = nothing learned (plateau within 0.1 of the t=0 value).")
    P(T[["seed", "Tm1_50", "Tm2_50", "Tm12_50", "max_Tm_50", "Tm1_90", "Tm2_90", "Tm12_90", "max_Tm_90"]].to_string(index=False))
    P("\n## Fixed reachable thresholds on the 17-query MSE: T_p = first log with MSE_p < 0.9; T12_sum18 = first log with MSE_E12 < 1.8 (= 0.9+0.9);"
      " T12_both = max(T_1, T_2) (both single-skill MSEs < 0.9)")
    P(T[["seed", "T1_mse09", "T2_mse09", "T12_sum18", "T12_both"]].to_string(index=False))
    P("\nplateau hMSE (mean last 10 logs):")
    P(T[["seed", "F1", "F2", "F12", "min_E1", "min_E2", "min_E12"]].round(4).to_string(index=False))
    # ---------------------------------------------------------------- 2. additivity residual
    P("\n## Additivity residual R(t) = MSE_E12 - MSE_E1 - MSE_E2 over all logs (absolute MSE; E[y^2]=2 for E12)")
    P("variants: spec = 4096 prompts, 1 query, E12 on an independent set;  h = 17 queries/context;  pair = E12A on the same inputs/c as E1,E2 (error = e1+e2 prompt by prompt)")
    rr = []
    for s in seeds:
        r = runs[s]
        Rs = {"spec": r["mse_E12"] - r["mse_E1"] - r["mse_E2"], "h": r["hmse_E12"] - r["hmse_E1"] - r["hmse_E2"],
              "pair": r["mse_E12A"] - r["mse_E1"] - r["mse_E2"], "pair_h": r["hmse_E12A"] - r["hmse_E1"] - r["hmse_E2"]}
        row = dict(seed=s)
        for k, v in Rs.items():
            row[f"max|R|_{k}"] = float(np.abs(v).max())
            row[f"mean_R_{k}"] = float(v.mean())
            row[f"sd_R_{k}"] = float(v.std())
            row[f"frac|R|<.02_{k}"] = float((np.abs(v) < 0.02).mean())
            row[f"max|R-R0|_{k}"] = float(np.abs(v - v[0]).max())
        rr.append(row)
        runs[s]["R"] = Rs
    RR = pd.DataFrame(rr)
    RR.to_csv(os.path.join(a.dir, "additivity_residual.csv"), index=False)
    P(RR.round(4).T.to_string())
    allR = {k: np.concatenate([runs[s]["R"][k] for s in seeds]) for k in ("spec", "h", "pair", "pair_h")}
    P("pooled over seeds and logs: " + "; ".join(f"{k}: mean {v.mean():+.4f} sd {v.std():.4f} max|R| {np.abs(v).max():.4f}" for k, v in allR.items()))
    # ---------------------------------------------------------------- 3. accuracy multiplicativity
    P("\n## Accuracy multiplicativity: acc_E12 vs acc_E1 * acc_E2 over all logged t and seeds")
    fits = []
    store = {}
    for tag, (pre, thr_txt, theta12, theta1) in {
            "abs05_spec": ("acc05", "|err|^2<0.5 for all three (lead's definition), 1 query", 0.5, 0.5),
            "abs05_h": ("hacc05", "|err|^2<0.5 for all three, 17 queries/context", 0.5, 0.5),
            "rel_spec": ("accrel", "|err|^2<0.5 E[y^2] (spec: 1.0 for E12), 1 query", 1.0, 0.5),
            "rel_h": ("haccrel", "|err|^2<0.5 E[y^2] (1.0 for E12), 17 queries/context", 1.0, 0.5)}.items():
        x = np.concatenate([runs[s][f"{pre}_E1"] * runs[s][f"{pre}_E2"] for s in seeds])
        y = np.concatenate([runs[s][f"{pre}_E12"] for s in seeds])
        yA = np.concatenate([runs[s][f"{pre}_E12A"] for s in seeds])
        mp_ = "h" if tag.endswith("_h") else ""
        s1 = np.concatenate([runs[s][f"{mp_}mse_E1"] for s in seeds])
        s2 = np.concatenate([runs[s][f"{mp_}mse_E2"] for s in seeds])
        th = acc_theory(s1 + s2, theta12)
        thx = acc_theory(s1, theta1) * acc_theory(s2, theta1)
        lr = stats.linregress(x, y)
        lrA = stats.linregress(x, yA)
        lrt = stats.linregress(thx, th)
        e1 = np.concatenate([runs[s][f"{pre}_E1"] for s in seeds])
        e2 = np.concatenate([runs[s][f"{pre}_E2"] for s in seeds])
        # log-spaced subsample (<= 60 logs per seed) so that long plateaus do not dominate the fit
        sub, off = [], 0
        for s_ in seeds:
            n_ = len(runs[s_]["t"])
            sub.append(off + np.unique(np.round(np.geomspace(1, n_, 60)).astype(int)) - 1)
            off += n_
        sub = np.concatenate(sub)
        lrs = stats.linregress(x[sub], y[sub])
        fits.append(dict(variant=tag, definition=thr_txt, n=len(x), n_sub=len(sub), slope_sub=lrs.slope, intercept_sub=lrs.intercept,
                         rms_identity_sub=float(np.sqrt(np.mean((y[sub] - x[sub]) ** 2))),
                         rms_fit_sub=float(np.sqrt(np.mean((y[sub] - lrs.intercept - lrs.slope * x[sub]) ** 2))), slope=lr.slope, intercept=lr.intercept, r2=lr.rvalue ** 2,
                         rms_identity=float(np.sqrt(np.mean((y - x) ** 2))), rms_fit=float(np.sqrt(np.mean((y - lr.intercept - lr.slope * x) ** 2))),
                         max_dev_identity=float(np.abs(y - x).max()), mean_dev_identity=float(np.mean(y - x)),
                         pairedE12A_slope=lrA.slope, pairedE12A_intercept=lrA.intercept,
                         rms_identity_E12A=float(np.sqrt(np.mean((yA - x) ** 2))),
                         theory_slope=lrt.slope, theory_intercept=lrt.intercept,
                         rms_theory_identity=float(np.sqrt(np.mean((th - thx) ** 2))),
                         rms_obs_vs_theory_acc12=float(np.sqrt(np.mean((y - th) ** 2))),
                         rms_obs_vs_theory_acc1=float(np.sqrt(np.mean((e1 - acc_theory(s1, theta1)) ** 2))),
                         rms_obs_vs_theory_acc2=float(np.sqrt(np.mean((e2 - acc_theory(s2, theta1)) ** 2)))))
        store[tag] = dict(x=x, y=y, yA=yA, th=th, thx=thx, theta12=theta12, theta1=theta1,
                          seed=np.concatenate([[s] * len(runs[s]["t"]) for s in seeds]))
    F = pd.DataFrame(fits)
    F.to_csv(os.path.join(a.dir, "acc_fits.csv"), index=False)
    P(F.round(4).T.to_string())
    # ---------------------------------------------------------------- 4. E_x
    P("\n## E_x (non-additive product pair) MSE (E[y^2]=1): initial / min over t / final / max over t")
    ex = pd.DataFrame([dict(seed=s, init=runs[s]["hmse_Ex"][0], min=runs[s]["hmse_Ex"].min(), final=runs[s]["hmse_Ex"][-1],
                            max=runs[s]["hmse_Ex"].max(), spec_min=runs[s]["mse_Ex"].min(), spec_final=runs[s]["mse_Ex"][-1]) for s in seeds])
    P(ex.round(4).to_string(index=False))
    # ---------------------------------------------------------------- 5. assignment
    P("\n## Neuron-teacher assignment (assigned = argmax_p |m_jp| once max >= 0.5)")
    arows = []
    for s in seeds:
        r = runs[s]
        asg = assignments(r["m"])
        fin = np.abs(r["m"][-1])
        row = dict(seed=s, final_assign=" ".join(("T%d" % (asg[-1, j] + 1)) if asg[-1, j] >= 0 else "--" for j in range(M)),
                   n_on_T1=int((asg[-1] == 0).sum()), n_on_T2=int((asg[-1] == 1).sum()), n_unassigned=int((asg[-1] < 0).sum()),
                   final_absm=" ".join(f"{fin[j].max():.2f}" for j in range(M)))
        sw = 0
        first = []
        for j in range(M):
            col = asg[:, j]
            nz = col[col >= 0]
            sw += int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0
            idx = np.where(col >= 0)[0]
            first.append(int(r["t"][idx[0]]) if len(idx) else -1)
        row.update(switches=sw, first_assigned_t=first, max_other_after_assigned=float(max((np.abs(r["m"][:, j, 1 - asg[-1, j]]).max()
                                                                                            for j in range(M) if asg[-1, j] >= 0), default=float("nan"))))
        arows.append(row)
        runs[s]["asg"] = asg
    AR = pd.DataFrame(arows)
    AR.to_csv(os.path.join(a.dir, "assignment.csv"), index=False)
    P(AR.to_string(index=False))
    # per-neuron timing: decoupling check.  If neurons evolved independently, T_jp * eta * pi_p * m0^2 would be ~constant
    # (cubic drift ~ pi_p m^3 => T ~ 1/(eta pi_p m0^2)); a second neuron on a teacher sees a smaller residual and should be slower.
    pi = [0.75, 0.25]
    eta = 1.0 / 32 ** 2
    nrows = []
    for s in seeds:
        r = runs[s]
        asg, al = r["asg"], np.abs(r["m"])
        for p in range(2):
            js = [j for j in range(M) if asg[-1, j] == p]
            ts_ = sorted([(first_cross(r["t"], al[:, j, p], 0.5, below=False), j) for j in js])
            for rank, (tj, j) in enumerate(ts_, 1):
                m0 = al[0, j, p]
                nrows.append(dict(seed=s, teacher=p + 1, neuron=j + 1, arrival_rank_on_teacher=rank, m0=m0, T_half=tj,
                                  K=tj * eta * pi[p] * m0 ** 2, K_nopi=tj * eta * m0 ** 2))
    NR = pd.DataFrame(nrows)
    NR.to_csv(os.path.join(a.dir, "neurons.csv"), index=False)
    P("\n## Per-neuron arrival: T_half = first log with |m_jp| >= 0.5 on its final teacher; K = T_half * eta * pi_p * m0^2 (constant if neurons were independent single-neuron cubic escapes)")
    P(NR.round(4).to_string(index=False))
    for rk in (1, 2):
        g = NR[NR.arrival_rank_on_teacher == rk]
        if len(g):
            P(f"rank {rk}: n={len(g)}  K median {g.K.median():.3g}  (min {g.K.min():.3g}, max {g.K.max():.3g});  K_nopi median {g.K_nopi.median():.3g}")
    for p in (1, 2):
        g = NR[(NR.teacher == p) & (NR.arrival_rank_on_teacher == 1)]
        if len(g):
            P(f"teacher {p}, first arrivals: n={len(g)} K median {g.K.median():.3g} range [{g.K.min():.3g}, {g.K.max():.3g}]")
    P("\nCPU seconds per seed: " + ", ".join(f"{s}: {meta.loc[s, 'cpu_s']:.0f}" for s in seeds) + f"; total {meta['cpu_s'].sum() / 3600:.3f} h")
    # ---------------------------------------------------------------- figures
    tmax = max(runs[s]["t"][-1] for s in seeds)
    xs = lambda ax: (ax.set_xscale("symlog", linthresh=2000), ax.set_xlim(0, tmax * 1.02))  # noqa: E731
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True)
    for ax, (key, ttl, ref) in zip(axs.ravel(), [("E1", "E1  skill 1 (pi=0.75)", 1.0), ("E2", "E2  skill 2 (pi=0.25)", 1.0),
                                                 ("E12", "E12  additive pair (E[y^2]=2)", 2.0), ("Ex", "Ex  product pair (E[y^2]=1)", 1.0)]):
        for i, s in enumerate(seeds):
            ax.plot(runs[s]["t"], runs[s]["hmse_" + key], color=COL[i % 5], lw=1.4, label=f"seed {s}")
        ax.axhline(ref, color=INK2, lw=0.8, ls=":")
        if key == "E12":
            ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
            ax.text(tmax, 0.52, "spec threshold 0.5", ha="right", fontsize=8, color=INK2)
        ax.set_title(ttl, loc="left"); ax.set_ylabel("MSE (absolute)"); xs(ax)
        ax.set_ylim(0, None)
    for ax in axs[1]:
        ax.set_xlabel("step t")
    axs[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle(f"Evaluation MSE (17 queries x 4096 contexts); Gamma={gamma}, M={M}", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "curves.png")); plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i, s in enumerate(seeds):
        axs[0].plot(runs[s]["t"], runs[s]["R"]["spec"], color=COL[i % 5], lw=0.9, alpha=0.9, label=f"seed {s}")
        axs[1].plot(runs[s]["t"], runs[s]["R"]["pair_h"], color=COL[i % 5], lw=0.9, alpha=0.9)
    for ax, ttl in zip(axs, ["R = MSE_E12 - MSE_E1 - MSE_E2  (spec: independent sets, 1 query, 4096 prompts)",
                             "R_pair = MSE_E12A - MSE_E1 - MSE_E2  (same inputs and c, 17 queries/context)"]):
        ax.axhspan(-0.02, 0.02, color="#1baf7a", alpha=0.15, lw=0); ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(ttl, loc="left"); ax.set_ylabel("R"); xs(ax)
    axs[1].set_xlabel("step t"); axs[0].legend(ncol=5, fontsize=8)
    axs[0].text(tmax, 0.025, "|R| <= 0.02 band (P12)", ha="right", fontsize=8, color=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "additivity_residual.png")); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
    ss = np.linspace(0.05, 3.0, 400)
    for ax, (tag, zoom), ttl in zip(axs, [("abs05_h", False), ("abs05_h", True)], ["17 queries/context, full range", "same data, zoomed on the observed range"]):
        st = store[tag]
        for i, s in enumerate(seeds):
            mk = st["seed"] == s
            ax.scatter(st["x"][mk], st["y"][mk], s=7, color=COL[i % 5], alpha=0.7, label=f"seed {s}", lw=0)
        ax.scatter(st["thx"], st["th"], s=5, color=INK, alpha=0.25, marker="x", lw=0.6, label="Gaussian model at each logged (MSE_1, MSE_2)")
        a1 = acc_theory(ss ** 2, 0.5)  # theoretical curve with s1 = s2 = s
        ax.plot(a1 * a1, acc_theory(2 * ss ** 2, 0.5), color=INK, lw=1.4, label="Gaussian curve, s1 = s2")
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":", label="y = x")
        row = F[F.variant == tag].iloc[0]
        xx = np.linspace(0, 1, 2)
        ax.plot(xx, row.intercept + row.slope * xx, color="#eb6834", lw=1.2, ls="--", label=f"OLS fit: {row.slope:.2f} x {row.intercept:+.3f}")
        ax.set_xlabel("acc_E1 * acc_E2"); ax.set_ylabel("acc_E12"); ax.set_title(ttl + "  [|err|^2 < 0.5]", loc="left")
        if zoom:
            lo, hi = min(st["x"].min(), st["y"].min()) - 0.01, max(st["x"].max(), st["y"].max()) + 0.01
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        else:
            ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)
        ax.set_aspect("equal", adjustable="box")
    axs[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "acc_multiplicative.png")); plt.close(fig)

    s0 = 1 if 1 in seeds else seeds[0]  # seed 1: a 2+2 split, both teachers learned
    r = runs[s0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for p in range(2):
        for j in range(M):
            axs[p].plot(r["t"], np.abs(r["m"][:, j, p]), color=COL[j], lw=1.4, label=f"neuron {j + 1}")
        axs[p].axhline(0.5, color=INK2, lw=0.8, ls="--")
        axs[p].set_title(f"|m_j{p + 1}| = |<w_j, v_{p + 1}>|  (seed {s0})", loc="left"); axs[p].set_xlabel("step t"); xs(axs[p])
    axs[0].set_ylabel("alignment"); axs[0].legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "alignments.png")); plt.close(fig)
    # all-seed alignments (supplementary)
    fig, axs = plt.subplots(len(seeds), 2, figsize=(10, 2.2 * len(seeds)), sharex=True, sharey=True, squeeze=False)
    for i, s in enumerate(seeds):
        for p in range(2):
            for j in range(M):
                axs[i, p].plot(runs[s]["t"], np.abs(runs[s]["m"][:, j, p]), color=COL[j], lw=1.1)
            axs[i, p].axhline(0.5, color=INK2, lw=0.6, ls="--"); xs(axs[i, p])
            axs[i, p].set_title(f"seed {s}: |m_j{p + 1}|", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "alignments_all_seeds.png")); plt.close(fig)
    open(os.path.join(a.dir, "analysis_stdout.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
