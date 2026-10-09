"""Analysis + figures for experiment 2 (many skills).  Usage: python -m icl_additive.analyze2 [--dir results/exp2]

Reads results/exp2/runs/<tag>/seed*.npz and runs.csv; writes summary.csv, per_skill.csv, analysis_stdout.txt, figs/*.png.
Primary quantities are alignment-based: mx_p(t) = max_j |m_jp(t)|, per-skill loss l_p = 1 - mx_p^2, weighted loss
L(t) = sum_p pi_p l_p(t).
"""
import argparse
import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .many import skill_freqs

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CA, CB = "#2a78d6", "#eb6834"
plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
                     "figure.dpi": 110})
warnings.filterwarnings("ignore")
RHO_PRIMARY = {1.5: 0.0496, 1.0: 0.0566}  # capacity-matched readout rho = sum_p a_p / M for Model B
LEARNED_B_DROP = 0.1  # Model B: max_j|m| plateaus well below 1 (several neurons share a skill), so "learned" = alignment loss fell by >= 0.1
LEARNED = 0.9  # a skill counts as learned (for fits / sharpness) if its final max_j |m| >= this


SEED_MAX = None


def load(d, tag):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "runs", tag, "seed*.npz"))):
        z = np.load(f)
        sd = int(os.path.basename(f)[4:-4])
        if SEED_MAX is None or sd <= SEED_MAX:
            runs[sd] = {k: z[k] for k in z.files}
    return runs


def first_cross_up(t, x, thr):
    """First time x >= thr, linearly interpolated between logs (0 if already true at t=0, nan if never)."""
    idx = np.where(x >= thr)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return 0.0
    return float(t[i - 1] + (thr - x[i - 1]) / (x[i] - x[i - 1]) * (t[i] - t[i - 1]))


def first_cross_down(t, l, thr):
    return first_cross_up(t, -l, -thr)


def reg(x, y):
    """OLS slope, SE, n of y on x (finite pairs)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    r = stats.linregress(x[ok], y[ok])
    return float(r.slope), float(r.stderr), int(ok.sum())


def skill_stats(run, pi, model="A", rho=np.nan):
    """Per-skill times and alignments for one run."""
    t, m = run["t"].astype(float), run["m"].astype(float)  # (T,), (T,M,P)
    mx = np.abs(m).max(axis=1)  # (T,P)
    P = mx.shape[1]
    l = 1.0 - mx ** 2
    rows = []
    a_end = np.abs(m[-1])
    for p in range(P):
        D = l[0, p] - l[:, p].min()
        T01d = first_cross_down(t, l[:, p], l[0, p] - 0.1 * D)
        T05d = first_cross_down(t, l[:, p], l[0, p] - 0.5 * D)
        T09d = first_cross_down(t, l[:, p], l[0, p] - 0.9 * D)
        rows.append(dict(p=p + 1, pi=pi[p], m0=mx[0, p], T05=first_cross_up(t, mx[:, p], 0.5),
                         T09=first_cross_up(t, mx[:, p], 0.9), T01d=T01d, T05d=T05d, T09d=T09d,
                         drop=D, final_mx=mx[-1, p], learned=bool(mx[-1, p] >= LEARNED if model == "A" else D >= LEARNED_B_DROP),
                         n_neur_05=int((a_end[:, p] >= 0.5).sum()), n_neur_09=int((a_end[:, p] >= 0.9).sum())))
    df = pd.DataFrame(rows)
    if model == "B":  # pooled alignment rho * sum_j m_jp^2 / a_p (1 = the skill's weight in the target is reproduced)
        df["pooled0"] = rho * (m[0] ** 2).sum(axis=0) / np.sqrt(pi)
        df["pooled_end"] = rho * (m[-1] ** 2).sum(axis=0) / np.sqrt(pi)
        q = rho * (m ** 2).sum(axis=1) / np.sqrt(pi)[None, :]  # (T,P) pooled alignment, supplementary (not alignment-max based)
        for nm, x in (("Tq01", 0.1), ("Tq05", 0.5), ("Tq09", 0.9)):
            df[nm] = [first_cross_up(t, q[:, p], q[0, p] + x * (1.0 - q[0, p])) if q[0, p] <= 0.8 else np.nan for p in range(P)]
        df["sharp_q"] = (df.Tq09 - df.Tq01) / df.Tq05
    df["sharp_d"] = (df.T09d - df.T01d) / df.T05d  # relative to the drop midpoint (defined for every skill)
    df["sharp_m"] = (df.T09d - df.T01d) / df.T05  # relative to T_p(0.5) of the spec (m >= 0.5)
    return df


def smooth(x, w=5):
    k = np.ones(w) / w
    return np.convolve(np.pad(x, (w // 2, w // 2), mode="edge"), k, mode="valid")


def mse_sharpness(run, learned_p):
    """Model A only: (T_0.9 - T_0.1)/T_0.5 of the drop of the per-skill eval MSE (17-query, 5-point running mean) for learned skills."""
    t = run["t"].astype(float)
    out = {}
    for p in learned_p:
        e = smooth(run["mse_h"][:, p - 1].astype(float))
        e0, eF = e[:3].mean(), e[-5:].mean()
        D = e0 - eF
        if D < 0.1:
            continue
        ts = [first_cross_down(t, e, e0 - x * D) for x in (0.1, 0.5, 0.9)]
        out[p] = (ts[2] - ts[0]) / ts[1] if ts[1] > 0 else np.nan
    return out


def window_fit(t, L, t_lo, t_hi):
    """OLS of log L on log t over logged t in [t_lo, t_hi], t > 0.  Returns exponent (= -slope), SE, n."""
    ok = (t >= t_lo) & (t <= t_hi) & (t > 0) & (L > 0)
    if ok.sum() < 4:
        return np.nan, np.nan, int(ok.sum())
    r = stats.linregress(np.log(t[ok]), np.log(L[ok]))
    return -float(r.slope), float(r.stderr), int(ok.sum())


def analyse(d, tag, alpha, lines):
    runs = load(d, tag)
    pi = skill_freqs(16, alpha)
    out_sum, out_ps = [], []
    for s, run in sorted(runs.items()):
        t = run["t"].astype(float)
        mx = np.abs(run["m"].astype(float)).max(axis=1)
        sk = skill_stats(run, pi, tag[0], float(tag.split("rho")[1]) if "rho" in tag else np.nan)
        sk.insert(0, "seed", s); sk.insert(0, "tag", tag)
        out_ps.append(sk)
        Lal = (pi[None, :] * (1 - mx ** 2)).sum(axis=1)
        Lmse = run["mse_total"].astype(float)
        # emergence order statistics (T_p(0.5) of the spec; 0 if aligned >= 0.5 at init)
        Te = np.sort(sk.T05.values[np.isfinite(sk.T05.values)])
        t3, t8 = (Te[2] if len(Te) >= 3 else np.nan), (Te[7] if len(Te) >= 8 else np.nan)
        # window between the 3rd and 8th emergence (by the sharper definition T_p,0.5 of the alignment drop, secondary)
        Td = np.sort(sk.T05d.values[sk.learned.values & np.isfinite(sk.T05d.values)])
        d3, d8 = (Td[2] if len(Td) >= 3 else np.nan), (Td[7] if len(Td) >= 8 else np.nan)
        lo = max(t3, t[1]) if np.isfinite(t3) else np.nan
        e_al, se_al, n_al = window_fit(t, Lal, lo, t8)
        e_ms, se_ms, n_ms = window_fit(t, Lmse, lo, t8)
        lo_d = max(d3, t[1]) if np.isfinite(d3) else np.nan
        e_al_d, se_al_d, n_al_d = window_fit(t, Lal, lo_d, d8)
        L = sk[sk.learned]
        row = dict(tag=tag, seed=s, alpha=alpha, steps=int(t[-1]), n_learned=int(len(L)),
                   n_learned_first8=int(L.p.le(8).sum()), n_at_init_ge05=int((sk.m0 >= 0.5).sum()),
                   t3=t3, t8=t8, win_n=n_al, expo_align=e_al, expo_align_se=se_al, expo_mse=e_ms, expo_mse_se=se_ms,
                   expo_align_dropwin=e_al_d, expo_align_dropwin_se=se_al_d, win_drop=(lo_d, d8),
                   L_align_end=float(Lal[-1]), L_mse_end=float(Lmse[-1]))
        for name, col in (("T05", "T05"), ("T09", "T09"), ("T05d", "T05d")):
            ok = L[col].values > 0
            sl, se, n = reg(np.log(L.p.values[ok].astype(float)), np.log(L[col].values[ok]))
            row[f"slope_{name}"], row[f"slope_{name}_se"], row[f"slope_{name}_n"] = sl, se, n
        L8 = L[L.p <= 8]
        for name, col in (("T05", "T05"), ("T09", "T09"), ("T05d", "T05d")):
            ok = L8[col].values > 0
            sl, se, n = reg(np.log(L8.p.values[ok].astype(float)), np.log(L8[col].values[ok]))
            row[f"slope8_{name}"], row[f"slope8_{name}_se"], row[f"slope8_{name}_n"] = sl, se, n
        # init-corrected slope: log(T05d m0^2) vs log p
        ok = L.T05d.values > 0
        sl, se, n = reg(np.log(L.p.values[ok].astype(float)), np.log(L.T05d.values[ok] * L.m0.values[ok] ** 2))
        row["slope_T05d_m0corr"], row["slope_T05d_m0corr_se"] = sl, se
        if tag[0] == "A":
            ms_ = mse_sharpness(run, list(L.p.values))
            row["sharp_mse_median"] = float(np.nanmedian(list(ms_.values()))) if ms_ else np.nan
            row["sharp_mse_n"] = len(ms_)
        am = np.abs(run["m"][-1].astype(float))
        row["min_neuron_maxm_end"] = float(am.max(axis=1).min())
        if tag[0] == "B":
            Lq = sk[np.isfinite(sk.Tq05) & (sk.Tq05 > 0)]
            row["slope_Tq05"], row["slope_Tq05_se"], row["slope_Tq05_n"] = reg(np.log(Lq.p.values.astype(float)), np.log(Lq.Tq05.values))
            row["sharp_q_median"] = float(np.nanmedian(Lq.sharp_q)) if len(Lq) else np.nan
        row["sharp_d_median"] = float(np.nanmedian(L.sharp_d)) if len(L) else np.nan
        row["sharp_m_median"] = float(np.nanmedian(L.sharp_m[L.T05 > 0])) if (L.T05 > 0).any() else np.nan
        row["sharp_d_median_p_le8"] = float(np.nanmedian(L.sharp_d[L.p <= 8])) if (L.p <= 8).any() else np.nan
        a_end = np.abs(run["m"][-1].astype(float))
        mxj = a_end.max(axis=1)
        row["neurons_ge09"], row["neurons_05_09"], row["neurons_lt05"], row["neurons_lt01"] = (
            int((mxj >= 0.9).sum()), int(((mxj >= 0.5) & (mxj < 0.9)).sum()), int((mxj < 0.5).sum()), int((mxj < 0.1).sum()))
        row["cpu_s"] = np.nan
        out_sum.append(row)
    return runs, pd.DataFrame(out_sum), (pd.concat(out_ps, ignore_index=True) if out_ps else pd.DataFrame())


def pooled(ps, col, pmax=16):
    """Pooled OLS of log T on log p over learned (skill, seed) pairs with T > 0 (and p <= pmax)."""
    L = ps[ps.learned & (ps[col] > 0) & (ps.p <= pmax)]
    return reg(np.log(L.p.values.astype(float)), np.log(L[col].values))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/exp2")
    ap.add_argument("--max-seed", type=int, default=None, help="use only seeds <= this (outputs get a suffix _seeds0-N)")
    a = ap.parse_args(argv)
    global SEED_MAX
    SEED_MAX = a.max_seed
    sfx0 = "" if a.max_seed is None else f"_seeds0-{a.max_seed}"
    figs = os.path.join(a.dir, "figs" + sfx0)
    os.makedirs(figs, exist_ok=True)
    lines = []

    def P(*x):
        s = " ".join(str(y) for y in x)
        print(s)
        lines.append(s)

    meta = pd.read_csv(os.path.join(a.dir, "runs.csv"))
    configs = []
    for tag in sorted(meta.tag.unique()):
        r = meta[meta.tag == tag].iloc[0]
        configs.append((tag, r.model, float(r.alpha), r.rho))
    sums, pss, runs_all = [], [], {}
    for tag, model, alpha, rho in configs:
        runs, sm, ps = analyse(a.dir, tag, alpha, lines)
        sm.insert(1, "model", model); ps.insert(1, "model", model)
        cp = meta[meta.tag == tag].set_index("seed").cpu_s
        if not runs:
            continue
        sm["cpu_s"] = sm.seed.map(cp)
        sm["stop_reason"] = sm.seed.map(meta[meta.tag == tag].set_index("seed").stop_reason)
        sums.append(sm); pss.append(ps); runs_all[tag] = runs
    S, PS = pd.concat(sums, ignore_index=True), pd.concat(pss, ignore_index=True)
    S.to_csv(os.path.join(a.dir, f"summary{sfx0}.csv"), index=False)
    PS.to_csv(os.path.join(a.dir, f"per_skill{sfx0}.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_columns", 50, "display.float_format", lambda x: f"{x:.3f}")
    for tag, model, alpha, rho in configs:
        sm, ps = S[S.tag == tag], PS[PS.tag == tag]
        P(f"\n## {tag}  (model {model}, alpha {alpha}, rho {rho}; seeds {sorted(sm.seed)})")
        P(sm[["seed", "steps", "stop_reason", "n_learned", "n_learned_first8", "n_at_init_ge05", "neurons_ge09", "neurons_05_09",
              "neurons_lt05", "neurons_lt01", "cpu_s"]].to_string(index=False))
        for col, nm in (("T05", "T_p(m>=0.5)"), ("T09", "T_p(m>=0.9)"), ("T05d", "T_p,0.5 (align-loss drop midpoint)")):
            P(f"\n{nm}: per-seed slope of log T vs log p (± SE, n learned skills with T>0):")
            for _, r in sm.iterrows():
                P(f"  seed {int(r.seed)}: {r[f'slope_{col}']:.3f} ± {r[f'slope_{col}_se']:.3f} (n={int(r[f'slope_{col}_n'])})")
            sl, se, n = pooled(ps, col)
            ms = sm[f"slope_{col}"].values
            P(f"  pooled over seeds: {sl:.3f} ± {se:.3f} (n={n});  mean of per-seed slopes {np.nanmean(ms):.3f} ± {np.nanstd(ms, ddof=1) / np.sqrt(np.isfinite(ms).sum()):.3f} (SE over seeds)")
            sl8, se8, n8 = pooled(ps, col, 8)
            P(f"  restricted to p <= 8, pooled: {sl8:.3f} ± {se8:.3f} (n={n8})")
        P(f"\ninit-corrected slope of log(T_p,0.5 * m0^2) vs log p: {np.round(sm.slope_T05d_m0corr.values, 3).tolist()}")
        P("\nT_p table (steps; nan = never; 0 = aligned >= 0.5 at init), T_p(m>=0.5) | T_p(m>=0.9) | T_p,0.5(drop) | m0 per seed")
        for col in ("T05", "T09", "T05d", "m0"):
            w = ps.pivot(index="p", columns="seed", values=col)
            w.columns = [f"s{c}" for c in w.columns]
            P(f"-- {col}"); P(w.round(0 if col != 'm0' else 3).to_string())
        P("\nsharpness (T_0.9 - T_0.1)/T_mid of the alignment-based per-skill loss (learned skills):")
        for _, r in sm.iterrows():
            P(f"  seed {int(r.seed)}: median over learned skills [mid=drop midpoint] {r.sharp_d_median:.3f}, [mid=T_p(0.5)] {r.sharp_m_median:.3f}, p<=8 [drop mid] {r.sharp_d_median_p_le8:.3f}")
        Ls = ps[ps.learned]
        P(f"  pooled medians: drop-mid {np.nanmedian(Ls.sharp_d):.3f} (n={Ls.sharp_d.notna().sum()}), T_p(0.5)-mid {np.nanmedian(Ls.sharp_m[Ls.T05 > 0]):.3f}")
        P("\nscaling exponents over the window between the 3rd and 8th skill emergence (T_p(0.5) order statistics):")
        P(sm[["seed", "t3", "t8", "win_n", "expo_align", "expo_align_se", "expo_mse", "expo_mse_se", "expo_align_dropwin", "expo_align_dropwin_se", "L_align_end", "L_mse_end"]].to_string(index=False))
        P(f"  mean exponent over seeds: align {sm.expo_align.mean():.3f} ± {sm.expo_align.std(ddof=1) / np.sqrt(len(sm)):.3f} (SE over seeds), mse {sm.expo_mse.mean():.3f} ± {sm.expo_mse.std(ddof=1) / np.sqrt(len(sm)):.3f}")
        if model == "B":
            P("\nSUPPLEMENTARY (Model B pooled alignment q_p = rho sum_j m_jp^2 / a_p; skills with q_p(0) <= 0.8 only; midpoint = q0 + 0.5 (1 - q0)):")
            for _, r in sm.iterrows():
                P(f"  seed {int(r.seed)}: slope log T_q,0.5 vs log p = {r.slope_Tq05:.3f} ± {r.slope_Tq05_se:.3f} (n={int(r.slope_Tq05_n)}); median sharpness {r.sharp_q_median:.3f}")
            Lq = ps[np.isfinite(ps.Tq05) & (ps.Tq05 > 0)]
            sl, se, n = reg(np.log(Lq.p.values.astype(float)), np.log(Lq.Tq05.values))
            P(f"  pooled: slope {sl:.3f} ± {se:.3f} (n={n}); pooled median sharpness {np.nanmedian(Lq.sharp_q):.3f}")
            P("  pooled alignment q_p at t=0 (mean over seeds): " + str(ps.groupby('p').pooled0.mean().round(2).tolist()))
            P("  pooled alignment q_p at the end (mean over seeds): " + str(ps.groupby('p').pooled_end.mean().round(3).tolist()))
        if model == "A":
            P(f"\nMODEL A per-skill MSE-based sharpness medians (learned skills; n skills): {[(round(x, 3), int(n)) for x, n in zip(sm.sharp_mse_median, sm.sharp_mse_n)]}")
        P(f"min over neurons of max_p|m| at the end, per seed: {np.round(sm.min_neuron_maxm_end.values, 3).tolist()}")
        P("\nneurons per skill at the end (|m|>=0.5) per seed:")
        P(ps.pivot(index="p", columns="seed", values="n_neur_05").T.to_string())
    open(os.path.join(a.dir, f"analysis_stdout{sfx0}.txt"), "w").write("\n".join(lines))

    # ------------------------------------------------------------------------------------------ figures
    def cfg(model, alpha, rho=None):
        for tag, m, al, r in configs:
            if m == model and abs(al - alpha) < 1e-9 and (model == "A" or rho is None or abs(r - rho) < 1e-9):
                return tag
        return None

    for alpha in sorted({c[2] for c in configs}):
        sfx = "" if alpha == 1.5 else f"_alpha{alpha}"
        tags = [("A", cfg("A", alpha), CA, "Model A (in-context)"), ("B", cfg("B", alpha, RHO_PRIMARY.get(alpha)), CB, "Model B (in-weight, capacity-matched readout)")]
        tags = [x for x in tags if x[1]]
        if not tags:
            continue
        # Tp vs p
        fig, ax = plt.subplots(2, len(tags), figsize=(5.2 * len(tags), 7.2), squeeze=False)
        for j, (mod, tag, col, nm) in enumerate(tags):
            ps = PS[PS.tag == tag]
            for i, (c, ttl) in enumerate((("T05", "T_p: first t with max_j|m_jp| >= 0.5"), ("T05d", "T_p,0.5: midpoint of alignment-loss drop"))):
                a_ = ax[i, j]
                for s, mk in zip(sorted(ps.seed.unique()), "os^Dv<>"):
                    q = ps[(ps.seed == s) & ps.learned & (ps[c] > 0)]
                    a_.plot(q.p, q[c], mk, color=col, alpha=0.7, ms=5, label=f"seed {s}")
                    q2 = ps[(ps.seed == s) & ~ps.learned & (ps[c] > 0)]
                    a_.plot(q2.p, q2[c], mk, mfc="none", color=INK2, alpha=0.6, ms=5)
                sl, se, n = pooled(ps, c)
                L = ps[ps.learned & (ps[c] > 0)]
                if n >= 3:
                    r = stats.linregress(np.log(L.p.values.astype(float)), np.log(L[c].values))
                    xs = np.array([1, 16.0])
                    a_.plot(xs, np.exp(r.intercept) * xs ** r.slope, "-", color=INK, lw=1.2, label=f"fit slope {sl:.2f} ± {se:.2f} (n={n})")
                    ref = 1.5 if mod == "A" else 0.75
                    ref = ref * alpha / 1.5
                    a_.plot(xs, np.exp(r.intercept) * xs ** ref, "--", color=INK2, lw=1, label=f"predicted slope {ref:.2f}")
                a_.set_xscale("log"); a_.set_yscale("log"); a_.set_xlabel("skill index p"); a_.set_ylabel("steps")
                a_.set_title(f"{nm}\n{ttl}", fontsize=9)
                a_.legend(fontsize=7, loc="upper left")
        fig.suptitle(f"Emergence time vs skill index, alpha = {alpha} (open grey markers: not learned, final max|m| < {LEARNED})", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(figs, f"Tp_vs_p{sfx}.png")); plt.close(fig)
        # curves per skill
        seeds = sorted(runs_all[tags[0][1]])
        fig, ax = plt.subplots(len(seeds), len(tags), figsize=(5.4 * len(tags), 3.3 * len(seeds)), squeeze=False, sharey=True)
        cmap = plt.get_cmap("viridis")
        for j, (mod, tag, col, nm) in enumerate(tags):
            for i, s in enumerate(seeds):
                run = runs_all[tag].get(s)
                a_ = ax[i, j]
                if run is None:
                    continue
                t = run["t"].astype(float)
                mx = np.abs(run["m"].astype(float)).max(axis=1)
                for p in range(16):
                    a_.plot(np.maximum(t, 500), 1 - mx[:, p] ** 2, color=cmap(p / 15), lw=1.1)
                a_.set_xscale("log"); a_.set_xlabel("step t"); a_.set_ylabel("1 - max_j m_jp^2")
                a_.set_title(f"{nm}, seed {s}", fontsize=9)
        sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 16))
        fig.colorbar(sm_, ax=ax.ravel().tolist(), label="skill p", shrink=0.6)
        fig.suptitle(f"Alignment-based per-skill loss, alpha = {alpha}", fontsize=10)
        fig.subplots_adjust(hspace=0.45); fig.savefig(os.path.join(figs, f"curves_per_skill{sfx}.png"), bbox_inches="tight"); plt.close(fig)
        # scaling law
        fig, ax = plt.subplots(2, len(tags), figsize=(5.4 * len(tags), 7.2), squeeze=False)
        pi = skill_freqs(16, alpha)
        for j, (mod, tag, col, nm) in enumerate(tags):
            sm = S[S.tag == tag]
            for i, (key, ttl, ek) in enumerate((("align", "alignment-based L(t) = sum pi_p (1 - max_j m^2)", "expo_align"), ("mse", "MSE-based L(t)", "expo_mse"))):
                a_ = ax[i, j]
                for _, r in sm.iterrows():
                    run = runs_all[tag][int(r.seed)]
                    t = run["t"].astype(float)
                    mx = np.abs(run["m"].astype(float)).max(axis=1)
                    L = (pi[None] * (1 - mx ** 2)).sum(axis=1) if key == "align" else run["mse_total"].astype(float)
                    a_.plot(t[1:], L[1:], color=col, alpha=0.6, lw=1)
                    if np.isfinite(r.t3) and np.isfinite(r.t8):
                        lo = max(r.t3, t[1])
                        ok = (t >= lo) & (t <= r.t8) & (t > 0)
                        a_.plot(t[ok], L[ok], color=INK, lw=2, alpha=0.8)
                        e, se, n = window_fit(t, L, lo, r.t8)
                        if np.isfinite(e):
                            sl = stats.linregress(np.log(t[ok]), np.log(L[ok]))
                            tt = np.array([lo, r.t8])
                            a_.plot(tt, np.exp(sl.intercept) * tt ** sl.slope, "--", color=INK2, lw=1)
                ea = sm[ek]
                a_.set_title(f"{nm}\n{ttl}\nexponent {ea.mean():.2f} (seeds: {', '.join(f'{x:.2f}' for x in ea)})", fontsize=8.5)
                a_.set_xscale("log"); a_.set_yscale("log"); a_.set_xlabel("step t"); a_.set_ylabel("L(t)")
        fig.suptitle(f"Frequency-weighted loss; bold = window between 3rd and 8th skill emergence, dashed = power-law fit (alpha = {alpha}; predicted A {(alpha - 1) / alpha:.2f}, B {(alpha - 1) / (alpha / 2):.2f})", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(figs, f"scaling_law{sfx}.png")); plt.close(fig)


if __name__ == "__main__":
    main()
