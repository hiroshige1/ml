"""Paper figures.  Run:  python3 -I paper/figs/make_figs.py   (from /home/user/ml)
Reads results/ (never writes there) and writes fig_*.pdf / fig_*.png next to this script.
Palette: reference categorical slots 1,2,3,7 (blue, orange, aqua, violet), validated all-pairs (see README.md).
Figures are drawn at their final print size (1-column 11pt article, 6.5in text width): include with
width=<native width> (e.g. \\includegraphics[width=\\linewidth]{figs/fig_threshold.pdf}) so fonts are 1:1.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = "/home/user/ml"
R = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "figs")

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FS = 8.5          # base font size (pt), identical in every figure
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS - 0.5,
    "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 1, "legend.frameon": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print("wrote", name)


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=FS + 1, ha="right", va="bottom")


# ----------------------------------------------------------------------------------------------------------------
# Fig 1: kappa_eff vs N
# ODE values: `python3 -I scripts/ode_kappa_exp1b.py` (d=32, m0 = c/sqrt(d), c in {.5,.7,1,1.4}, no free parameters)
ODE_N = [32, 128, 512, 4096]
ODE = {"free1": [6.47, 5.46, 4.99, 4.85], "free10": [7.02, 6.18, 5.32, 4.92], "tied": [4.18, 4.15, 4.15, 4.15]}


def fig_kappa():
    f = pd.read_csv(os.path.join(R, "exp1b", "kappa_fits.csv"))
    spec = [("free1", BLUE, r"free readout, $\eta_\Gamma=\eta$", "o"),
            ("free10", ORANGE, r"free readout, $\eta_\Gamma=10\eta$", "s"),
            ("tied", AQUA, r"tied readout, $\Gamma=\|w\|^2$", "D")]
    jit = {"free1": 0.92, "free10": 1.0, "tied": 1.08}  # multiplicative x-offset so the +-s.e. bars do not overlap
    fig, ax = plt.subplots(figsize=(4.85, 3.0))
    ax.axhline(4, color=INK2, lw=0.7, ls=(0, (1, 2)), zorder=1)
    ax.axhline(8, color=INK2, lw=0.7, ls=(0, (1, 2)), zorder=1)
    ax.text(230, 3.95, r"$2k^{\!*}\!=4$", color=INK2, fontsize=FS - 1, va="top", ha="center")
    ax.text(26, 8.07, r"$4k^{\!*}\!=8$", color=INK2, fontsize=FS - 1, va="bottom", ha="left")
    for key, col, lab, mk in spec:
        ax.plot(ODE_N, ODE[key], ls=(0, (4, 2)), color=col, lw=1.2, zorder=2)
        g = f[f.protocol == key].sort_values("N")
        x = g.N.values * jit[key]
        ax.errorbar(x, g.kappa, yerr=g.kappa_se, fmt=mk, ms=4.5, color=col, mec="white", mew=0.5, ecolor=col,
                    elinewidth=1.0, capsize=2, capthick=1.0, zorder=4)
        c = g[g.n_censored > 0]
        ax.scatter(c.N.values * jit[key], c.kappa_cap, marker="^", s=34, facecolor="none", edgecolor=col, lw=1.1,
                   zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(25, 6000)
    ax.set_ylim(3.5, 9.2)
    ax.set_xlabel(r"context length $N$")
    ax.set_ylabel(r"effective exponent $\kappa_{\rm eff}$")
    ax.set_xticks([32, 128, 512, 4096])
    ax.set_xticklabels(["32", "128", "512", "4096"])
    ax.minorticks_off()
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(axis="x", visible=False)
    h = [(Line2D([], [], marker=mk, ms=4.5, color=col, mec="white", mew=0.5, ls=(0, (4, 2)), lw=1.2)) for _, col, _, mk in spec]
    leg = ax.legend(h, [x[2] for x in spec], loc="upper right", ncol=1, handlelength=2.6, labelspacing=0.3,
                    borderaxespad=0.2)
    ax.add_artist(leg)
    key = [Line2D([], [], marker="o", ms=4.5, color=INK2, ls="none", mec="white", mew=0.5),
           Line2D([], [], color=INK2, ls=(0, (4, 2)), lw=1.2),
           Line2D([], [], marker="^", ms=5.5, mfc="none", mec=INK2, ls="none", mew=1.1)]
    ax.legend(key, [r"SGD $\pm$ s.e.", "population ODE", "censored, set to cap"], loc="center right",
              bbox_to_anchor=(1.0, 0.52), handlelength=2.0, labelspacing=0.3, borderaxespad=0.2)
    save(fig, "fig_kappa")


# ----------------------------------------------------------------------------------------------------------------
# Fig 2: threshold heatmaps + P7
def dstar(N, g):
    N = np.asarray(N, float)
    return N * (8 - 4 * g * (1 - 1 / N) - 24 * g / N) / (16 * g)


def fig_threshold():
    t = pd.read_csv(os.path.join(R, "exp1b", "threshold_table.csv"))
    p7 = pd.read_csv(os.path.join(R, "exp1b", "p7.csv"))
    Ns, ds = [32, 128, 512], [16, 32, 64]
    cmap = LinearSegmentedColormap.from_list("esc", ["#e9f0fa", "#9dc0ec", "#2a78d6", "#14407a"])
    fig = plt.figure(figsize=(7.6, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.35], wspace=0.42)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    for ax, g in zip(axes, [1.0, 0.1]):
        s = t[np.isclose(t.gamma, g)]
        for N in Ns:
            for d in ds:
                r = s[(s.N == N) & (s.d == d)].iloc[0]
                frac = r.escaped / r.n
                x0, y0 = np.log2(d) - 0.5, np.log2(N) - 1.0
                ax.add_patch(plt.Rectangle((x0, y0), 1.0, 2.0, facecolor=cmap(frac), edgecolor="white", lw=1.5, zorder=2))
                ax.text(np.log2(d), np.log2(N) - 0.55, f"{int(r.escaped)}/{int(r.n)}", ha="center", va="center", zorder=5,
                        fontsize=FS - 0.5, color="white" if frac > 0.45 else INK)
        ax.set_xlim(3.5, 6.5)
        ax.set_ylim(4, 10)
        ax.set_xticks(np.log2(ds))
        ax.set_xticklabels([str(d) for d in ds])
        ax.set_yticks(np.log2(Ns))
        ax.set_yticklabels([str(n) for n in Ns])
        ax.set_xlabel(r"dimension $d$")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(rf"$\gamma={g:g}$" + (r": runs escaping (of 3)" if g == 1 else r" ($d^*>150$)"), fontsize=FS, pad=3)
    axes[0].set_ylabel(r"context length $N$")
    axes[1].set_yticklabels([])
    # corrected threshold curve, gamma = 1 (continuous in N; clipped to the grid)
    Nc = np.geomspace(16, 1024, 400)
    dc = dstar(Nc, 1.0)
    X, Y = np.log2(dc), np.log2(Nc)
    m = (X >= 3.5) & (X <= 6.5)
    import matplotlib.patheffects as pe
    axes[0].plot(X[m], Y[m], color=INK, lw=1.5, zorder=6, solid_capstyle="round",
                 path_effects=[pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()])
    axes[0].text(6.28, 7.4, r"$d^*(N)$", fontsize=FS, color=INK, ha="center", va="center", zorder=7, rotation=0,
                 path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    # P7
    ax = fig.add_subplot(gs[0, 2])
    Bs = [32, 256, 1024]
    rng = np.random.default_rng(0)
    for B in Bs:
        v = p7[p7.B == B].sort_values("seed").final_abs_m.values
        x = B * np.array([0.88, 1.0, 1.14])
        ax.scatter(x, v, s=22, color=BLUE, edgecolor="white", lw=0.5, zorder=4)
    ax.axhline(0.125, color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.axhline(0.5, color=ORANGE, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.text(26, 0.125 * 1.08, r"$m_0=0.125$", fontsize=FS - 1, color=INK2, va="bottom", ha="left")
    ax.text(26, 0.5 * 1.08, r"escape threshold $0.5$", fontsize=FS - 1, color=INK2, va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(24, 1500)
    ax.set_ylim(0.003, 1.2)
    tot = {32: "9.6", 256: "25.6", 1024: "30.7"}
    ax.set_xticks(Bs)
    ax.set_xticklabels([f"{B}\n({tot[B]}M)" for B in Bs])
    ax.minorticks_off()
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("prompts per step $B$  (total prompts)")
    ax.set_ylabel(r"final $|m|$")
    ax.text(0.97, 0.30, r"0/9 escaped", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS, color=INK)
    ax.set_title(r"$\gamma=1,\ N=128,\ d=64$", fontsize=FS, pad=3)
    save(fig, "fig_threshold")


# ----------------------------------------------------------------------------------------------------------------
# Fig 3: many skills (exp 2) + composition (exp 3)
def loglog_fit(p, T):
    x, y = np.log(p), np.log(T)
    n = len(x)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    se = np.sqrt(res @ res / (n - 2) / ((x - x.mean()) ** 2).sum())
    return b, se, a, n


def fig_many():
    d = pd.read_csv(os.path.join(R, "exp2", "per_skill.csv"))
    series = [("A_a1.5", BLUE, "o", "in-context A"),
              ("B_a1.5_rho0.0496", ORANGE, "s", "in-weight B")]
    bt = os.path.join(R, "exp2", "btied", "per_skill.csv")
    btied_used = None
    if os.path.exists(bt):
        db = pd.read_csv(bt)
        tags = sorted(db.tag.unique())
        pref = [t for t in tags if t.endswith("eta1")] or tags
        btied_used = pref[0]
        d = pd.concat([d, db[db.tag == btied_used]], ignore_index=True)
        series.append((btied_used, VIOLET, "^", "in-weight (B-tied)"))
        print("B-tied series included:", btied_used)

    fig = plt.figure(figsize=(7.4, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    pp = np.array([1, 16])
    handles, labels = [], []
    for tag, col, mk, lab in series:
        s = d[(d.tag == tag) & d.learned & (d.T05d > 0)]
        jit = {0: 0.95, 1: 1.0, 2: 1.05}
        ax.scatter(s.p * s.seed.map(jit) if False else s.p, s.T05d, s=14, marker=mk, color=col, alpha=0.75, edgecolor="white",
                   lw=0.4, zorder=3)
        b, se, a, n = loglog_fit(s.p.values.astype(float), s.T05d.values)
        ax.plot(pp, np.exp(a) * pp ** b, color=col, lw=1.3, zorder=2)
        handles.append(Line2D([], [], color=col, marker=mk, ms=4, lw=1.3, mec="white", mew=0.4))
        labels.append(f"{lab}: slope ${b:.2f}\\pm{se:.2f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.minorticks_off()
    ax.set_xlabel(r"skill index $p$   ($\pi_p\propto p^{-1.5}$)")
    ax.set_ylabel(r"emergence time $T_p$ (steps)")
    ax.set_xlim(0.85, 18)
    ax.legend(handles, labels, loc="upper left", handlelength=1.8, labelspacing=0.3, borderaxespad=0.2)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 4)
    ax.set_title("many skills: $T_p$ vs $p$ (3 seeds)", fontsize=FS, pad=3)
    panel_label(ax, "(a)")

    # exp 3 composition, seed 1
    z = np.load(os.path.join(R, "exp3", "main", "traj", "seed1.npz"))
    tm = pd.read_csv(os.path.join(R, "exp3", "main", "times.csv"))
    r = tm[tm.seed == 1].iloc[0]
    T1, T2 = r.T1, r.T2
    t = z["t"].astype(float)
    k = t > 0
    e1, e2, e12, ex = z["hmse_E1"], z["hmse_E2"], z["hmse_E12"] / 2.0, z["hmse_Ex"]  # normalise by E[y^2] = 1, 1, 2, 1
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t[k], ex[k], color=VIOLET, lw=1.3, zorder=3)
    ax.plot(t[k], e1[k], color=BLUE, lw=1.3, zorder=3)
    ax.plot(t[k], e2[k], color=ORANGE, lw=1.3, zorder=3)
    ax.plot(t[k], e12[k], color=AQUA, lw=2.4, zorder=2, solid_capstyle="butt")
    ax.plot(t[k], (e1[k] + e2[k]) / 2.0, color=INK, lw=0.8, ls=(0, (2, 2)), zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(500, 3e5)
    ax.set_ylim(0.2, 1.2)
    ax.minorticks_off()
    ax.set_xlabel("training step")
    ax.set_ylabel(r"normalised MSE ($\mathrm{MSE}/\mathbb{E}\,y^2$)")
    for T, c, lab in [(T1, BLUE, r"$T_1$"), (T2, ORANGE, r"$T_2$")]:
        ax.plot([T, T], [1.2, 1.14], color=c, lw=1.8, zorder=5)
        ax.text(T * 1.1, 1.17, lab, ha="left", va="center", fontsize=FS - 0.5, color=INK)

    h = [Line2D([], [], color=BLUE, lw=1.3), Line2D([], [], color=ORANGE, lw=1.3),
         Line2D([], [], color=AQUA, lw=2.4), Line2D([], [], color=INK, lw=0.8, ls=(0, (2, 2))),
         Line2D([], [], color=VIOLET, lw=1.3)]
    ax.legend(h, [r"$E_1$", r"$E_2$", r"$E_{12}/2$ (additive pair)", r"$(E_1+E_2)/2$", r"$E_\times$ (product)"],
              loc="lower left", bbox_to_anchor=(0.0, 0.0), handlelength=1.8, labelspacing=0.25, borderaxespad=0.2,
              ncol=1)
    ax.set_title("composition (seed 1)", fontsize=FS, pad=3)
    panel_label(ax, "(b)")
    save(fig, "fig_many")
    return btied_used


# ----------------------------------------------------------------------------------------------------------------
# Fig 4: transformers (exp 5)
def fig_transformer():
    C = os.path.join(R, "exp5", "curves")
    S = pd.read_csv(os.path.join(R, "exp5", "summary.csv"))
    cells = [("a", BLUE, r"(a) $N{=}256,\ B{=}64$"), ("c", ORANGE, r"(c) $N{=}16,\ B{=}64$"),
             ("d", AQUA, r"(d) $N{=}16,\ B{=}1024$")]
    fig = plt.figure(figsize=(7.5, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.text(30000, 0.52, "0.5", color=INK2, fontsize=FS - 1, ha="right", va="bottom")
    for cell, col, lab in cells:
        for s in range(3):
            z = np.load(os.path.join(C, f"{cell}_s{s}.npz"))
            st, mse = z["steps"], z["mse"]
            ax.plot(st, mse, color=col, lw=1.0, alpha=0.9, zorder=3)
            if bool(z["partial"]) if "partial" in z.files else False:  # a_s2: censored at 17000 by the CPU cap
                ax.plot(st[-1], mse[-1], marker=">", ms=4.5, color=col, mec="white", mew=0.4, zorder=4, clip_on=False)
    ax.set_xlim(0, 30000)
    ax.set_ylim(0, 1.3)
    ax.set_xticks([0, 10000, 20000, 30000])
    ax.set_xticklabels(["0", "10k", "20k", "30k"])
    ax.set_xlabel("training step")
    ax.set_ylabel("eval MSE")
    h = [Line2D([], [], color=c, lw=1.4) for _, c, _ in cells]
    ax.legend(h, [l for _, _, l in cells], loc="lower right", handlelength=1.6, labelspacing=0.3, borderaxespad=0.2)
    ymin, ymax = ax.get_ylim()
    ax.set_title("eval MSE, 3 seeds per cell", fontsize=FS, pad=3)
    panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    zc = np.load(os.path.join(C, "c_s0.npz"))
    zd = np.load(os.path.join(C, "d_s0.npz"))
    Te = float(S[(S.cell == "d") & (S.seed == 0)].Te05.iloc[0])
    for z, col, lab, mk in [(zc, ORANGE, r"(c) stuck", "o"), (zd, AQUA, r"(d) emerging", "s")]:
        ax.plot(z["norm_steps"], z["norms"][:, 2], color=col, lw=1.3, marker=mk, ms=3.2, mec="white", mew=0.3, label=lab,
                zorder=3)
    ax.axvline(Te, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.text(Te - 400, 0.02 * 1.0, "(d) emerges\n(eval MSE $< 0.5$)", fontsize=FS - 1.5, color=INK2, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 3.5)
    ax.set_xlim(0, 30000)
    ax.set_xticks([0, 10000, 20000, 30000])
    ax.set_xticklabels(["0", "10k", "20k", "30k"])
    ax.set_xlabel("training step")
    ax.set_ylabel("final readout norm $\\|W_{\\rm head}\\|$")
    ax.legend(loc="upper right", handlelength=1.8, labelspacing=0.3, borderaxespad=0.2)
    ax.set_title("readout norm, seed 0", fontsize=FS, pad=3)
    panel_label(ax, "(b)")
    save(fig, "fig_transformer")


# ----------------------------------------------------------------------------------------------------------------
# Fig 5: three regimes by SGD (exp 6): escape steps T_0.5 vs N, one panel per readout protocol
# ODE steps = pre-registered flow time x d^2 (= x 4096), fixed before the run; cross-checked against summary.csv below.
ODE6_N = [16, 64, 256]
ODE6 = {"fixed01": [400e3, 193e3, 175e3], "free": [1.835e6, 627e3, 414e3], "tied": [17.0e3, 16.4e3, 16.3e3]}  # LSODA (rtol 1e-10)


def fig_regimes():
    S = pd.read_csv(os.path.join(R, "exp6", "summary.csv"))
    # free N=16 was rerun with max_steps 2.5e6 (variant "ext"): use the rerun rows where present
    S = S.sort_values("variant", key=lambda v: v.map({"ext": 0}).fillna(1)).drop_duplicates(["protocol", "scheme", "N"])
    panels = [("fixed1", r"pinned $\gamma=1$"), ("fixed01", r"pinned $\gamma=0.1$"), ("free", "free"), ("tied", "tied")]
    schemes = [("tok", BLUE, "o", r"tokens matched, $NB=4096$", 0.84), ("B64", ORANGE, "s", r"$B=64$", 1.19)]
    seed_off = np.array([0.95, 1.0, 1.05])
    fig, axes = plt.subplots(1, 4, figsize=(6.4, 2.45), sharey=True, gridspec_kw={"wspace": 0.10})
    ymax = 4e6
    for ax, (prot, title) in zip(axes, panels):
        g = S[S.protocol == prot]
        if prot in ODE6:  # check the hard-coded pre-registered values against the table
            od = g[g.scheme == "tok"].sort_values("N").ode_steps.values
            assert np.allclose(od, ODE6[prot], rtol=0.01), (prot, od)
            ax.plot(ODE6_N, ODE6[prot], ls=(0, (4, 2)), color=INK2, lw=1.2, zorder=2)
        else:  # trapped: no finite ODE escape time; shade everything beyond the step cap
            assert np.isinf(g.ode_steps).all()
            ax.axhspan(1e6, ymax, facecolor="#ecebe7", edgecolor="none", zorder=0)
            ax.axhline(1e6, color=INK2, lw=0.7, ls=(0, (1, 2)), zorder=1)
            ax.text(0.5, 0.965, "trapped (ODE)", transform=ax.transAxes, ha="center", va="top", fontsize=FS - 1,
                    color=INK2)
        for sch, col, mk, _, so in schemes:
            med_x, med_y = [], []
            for N in ODE6_N:
                r = g[(g.scheme == sch) & (g.N == N)].iloc[0]
                T = np.array([float(v) for v in r.T05_seeds.split("/")])
                x = N * so * seed_off
                ok = np.isfinite(T)
                ax.scatter(x[ok], T[ok], s=11, marker=mk, color=col, alpha=0.55, edgecolor="none", zorder=3)
                if (~ok).any():  # censored: upward open triangle at the cap of this run
                    ax.scatter(x[~ok], np.full((~ok).sum(), float(r.max_steps)), marker="^", s=26, facecolor="none",
                               edgecolor=col, lw=1.0, zorder=4)
                if np.isfinite(r.T05_median):
                    med_x.append(N * so)
                    med_y.append(r.T05_median)
            ax.plot(med_x, med_y, color=col, lw=0.9, zorder=3.5)
            ax.plot(med_x, med_y, ls="none", marker=mk, ms=4.8, color=col, mec="white", mew=0.6, zorder=5)
        if prot == "fixed1":  # escape counts
            for sch, col, _, _, so in schemes:
                for N in ODE6_N:
                    r = g[(g.scheme == sch) & (g.N == N)].iloc[0]
                    ax.text(N * so, 1.05e4 if sch == "tok" else 0.8e4, f"{int(r.n_reached)}/{int(r.n)}", color=col,
                            ha="center", va="center", fontsize=FS - 1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10.5, 390)
        ax.set_ylim(6.5e3, ymax)
        ax.set_xticks(ODE6_N)
        ax.set_xticklabels([str(n) for n in ODE6_N])
        ax.minorticks_off()
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=FS, pad=3)
    axes[0].set_ylabel(r"escape steps $T_{0.5}$")
    fig.subplots_adjust(left=0.10, right=0.995, top=0.90, bottom=0.34)
    fig.text(0.5, 0.215, r"context length $N$", fontsize=FS, ha="center", va="top")
    h = [Line2D([], [], marker=mk, ms=4.8, color=col, mec="white", mew=0.6, lw=0.9) for _, col, mk, _, _ in schemes]
    h += [Line2D([], [], marker="o", ms=2.8, color=INK2, alpha=0.6, ls="none"),
          Line2D([], [], marker="^", ms=5, mfc="none", mec=INK2, ls="none", mew=1.0),
          Line2D([], [], color=INK2, ls=(0, (4, 2)), lw=1.2)]
    lab = [schemes[0][3], schemes[1][3], "single seed", "censored at cap", "pre-registered ODE"]
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), handlelength=2.0, labelspacing=0.3,
               columnspacing=1.6)
    save(fig, "fig_regimes")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or ["kappa", "threshold", "many", "transformer", "regimes"]
    for n in only:
        {"kappa": fig_kappa, "threshold": fig_threshold, "many": fig_many, "transformer": fig_transformer,
     "regimes": fig_regimes}[n]()
