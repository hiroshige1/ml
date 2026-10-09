"""Analysis of the B-tied baseline (exp 2 add-on).  python -m icl_additive.analyze_btied
Reads results/exp2/btied/runs/<tag>/seed*.npz (+ exp 2's A and B runs for the comparison figure), writes
results/exp2/btied/{summary.csv,per_skill.csv,analysis_stdout.txt} and results/exp2/figs/Tp_vs_p_with_Btied.png.
Statistics are exactly those of analyze2 (its skill_stats / window_fit / reg are imported unchanged); "learned" = drop of the
alignment loss >= 0.1 (the exp-2 Model-B criterion), since no neuron ever reaches |m| = 0.9.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import analyze2 as A2
from .analyze2 import CA, CB, GRID, INK, INK2, reg, skill_stats, window_fit
from .many import skill_freqs

CT = "#2f9e6b"
ALPHA, D, P_ = 1.5, 32, 16
PI = skill_freqs(P_, ALPHA)
AP = np.sqrt(PI)


def load_runs(d, tag):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "runs", tag, "seed*.npz"))):
        z = np.load(f)
        out[int(os.path.basename(f)[4:-4])] = {k: z[k] for k in z.files}
    return out


def merged(run):
    """1000-step log merged with the dense (100-step) early log -> finer time resolution for T_p (supplementary)."""
    t = np.concatenate([run["fine_t"], run["t"]]).astype(float)
    m = np.concatenate([run["fine_m"], run["m"]]).astype(float)
    _, idx = np.unique(t, return_index=True)
    return {"t": t[idx], "m": m[idx]}


def pooled_q(run):
    return (run["norm2"][:, :, None] * run["m"].astype(float) ** 2).sum(axis=1) / AP[None, :]  # (T,P) sum_j n_j m_jp^2 / a_p


def analyse_run(run, eta, seed):
    t = run["t"].astype(float)
    m = run["m"].astype(float)
    n2 = run["norm2"].astype(float)
    mx = np.abs(m).max(axis=1)
    sk = skill_stats(run, PI, "B", rho=1.0)
    q = pooled_q(run)
    sk["q0"], sk["q_end"] = q[0], q[-1]
    sk["top_share_end"] = (n2[-1][:, None] * m[-1] ** 2 / AP[None, :]).max(axis=0)  # best single neuron's share of a_p
    skm = skill_stats(merged(run), PI, "B", rho=1.0)  # fine-grid version
    for c in ("T05", "T05d", "T09d", "T01d", "sharp_d"):
        sk[c + "_fine"] = skm[c].values
    # neuron that ends on skill p, its initial alignment, own norm
    jbest = np.abs(m[-1]).argmax(axis=0)
    sk["j_end"] = jbest
    sk["m0_jend"] = np.abs(m[0][jbest, np.arange(P_)])
    sk["m_end_jend"] = np.abs(m[-1][jbest, np.arange(P_)])
    sk["n_end_jend"] = n2[-1][jbest]
    sk["n0_jend"] = n2[0][jbest]
    sk["a_p"] = AP
    assign = np.abs(m[-1]).argmax(axis=1)  # skill each neuron ends on
    sk["n_assigned"] = np.bincount(assign, minlength=P_)
    sk["n_m_ge03"] = (np.abs(m[-1]) >= 0.3).sum(axis=0)
    # same-time invariants (T_p,0.5 drop midpoint on the primary 1000-step grid)
    for T, nm in ((sk.T05d, "d"), (sk.T05d_fine, "dfine")):
        sk[f"K1_{nm}"] = T * eta * AP                                           # T eta a_p                    (drift ~ a_p m n)
        sk[f"K2_{nm}"] = T * eta * AP * sk.m0_jend                              # T eta a_p m0
        sk[f"K3_{nm}"] = T * eta * AP * sk.m0_jend ** 2                         # T eta a_p m0^2 (the exp-2 A/B form)
        sk[f"K4_{nm}"] = T * eta * AP / np.log(1.0 / sk.m0_jend)                # T eta a_p / ln(1/m0)         (exponential escape from m0)
    sk.insert(0, "seed", seed)
    # window exponents (as analyze2.analyse)
    Lal = (PI[None, :] * (1 - mx ** 2)).sum(axis=1)
    Lmse = run["mse_total"].astype(float)
    Te = np.sort(sk.T05.values[np.isfinite(sk.T05.values)])
    t3, t8 = (Te[2] if len(Te) >= 3 else np.nan), (Te[7] if len(Te) >= 8 else np.nan)
    Td = np.sort(sk.T05d.values[sk.learned.values & np.isfinite(sk.T05d.values)])
    d3, d8 = (Td[2] if len(Td) >= 3 else np.nan), (Td[7] if len(Td) >= 8 else np.nan)
    lo = max(t3, t[1]) if np.isfinite(t3) else np.nan
    lo_d = max(d3, t[1]) if np.isfinite(d3) else np.nan
    e_al, _, n_al = window_fit(t, Lal, lo, t8)
    e_ms, _, _ = window_fit(t, Lmse, lo, t8)
    e_ald, _, n_ald = window_fit(t, Lal, lo_d, d8)
    e_msd, _, _ = window_fit(t, Lmse, lo_d, d8)
    L = sk[sk.learned]
    row = dict(seed=seed, steps=int(t[-1]), n_learned=len(L), n_learned_first8=int(L.p.le(8).sum()), n_at_init_ge05=int((sk.m0 >= .5).sum()),
               t3=t3, t8=t8, win_n=n_al, expo_align=e_al, expo_mse=e_ms, d3=d3, d8=d8, win_n_drop=n_ald, expo_align_dropwin=e_ald, expo_mse_dropwin=e_msd,
               L_align_end=float(Lal[-1]), L_mse_end=float(Lmse[-1]), mse_t1000=float(Lmse[1]), sum_n_end=float(n2[-1].sum()),
               n_neurons_mx_ge09=int((np.abs(m[-1]).max(axis=1) >= .9).sum()), n_neurons_mx_ge05=int((np.abs(m[-1]).max(axis=1) >= .5).sum()),
               max_mx_end=float(np.abs(m[-1]).max()), max_n_end=float(n2[-1].max()), min_n_end=float(n2[-1].min()))
    for name, col in (("T05", "T05"), ("T05d", "T05d"), ("T05d_fine", "T05d_fine"), ("T05_fine", "T05_fine")):
        for sub, Lx in (("", L), ("8", L[L.p <= 8])):
            ok = Lx[col].values > 0
            sl, se, nn = reg(np.log(Lx.p.values[ok].astype(float)), np.log(Lx[col].values[ok]))
            row[f"slope{sub}_{name}"], row[f"slope{sub}_{name}_se"], row[f"slope{sub}_{name}_n"] = sl, se, nn
    row["sharp_d_median"] = float(np.nanmedian(L.sharp_d)) if len(L) else np.nan
    row["sharp_m_median"] = float(np.nanmedian(L.sharp_m[L.T05 > 0])) if (L.T05 > 0).any() else np.nan
    row["sharp_d_fine_median"] = float(np.nanmedian(L.sharp_d_fine)) if len(L) else np.nan
    Uh0, Uh1 = run["Uhat_init"], run["Uhat_end"]
    cos = np.abs((Uh0 * Uh1).sum(axis=1))
    row["cos_init_end_mean"], row["cos_init_end_min"] = float(cos.mean()), float(cos.min())
    row["frac_neurons_shared"] = float((np.bincount(assign, minlength=P_)[:8] >= 2).mean())  # fraction of skills 1-8 with >= 2 neurons
    return sk, row


def pooled(ps, col, pmax=16, learned=True):
    L = ps[(ps.learned if learned else True) & (ps[col] > 0) & (ps.p <= pmax)]
    return reg(np.log(L.p.values.astype(float)), np.log(L[col].values))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/exp2/btied")
    ap.add_argument("--exp2", default="results/exp2")
    a = ap.parse_args(argv)
    lines = []

    def P(*x):
        s = " ".join(str(y) for y in x)
        print(s); lines.append(s)

    meta = pd.read_csv(os.path.join(a.dir, "runs.csv"))
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", lambda x: f"{x:.3f}")
    S, PS, RUNS = [], [], {}
    for tag in sorted(meta.tag.unique()):
        eta = float(meta[meta.tag == tag].eta.iloc[0])
        runs = load_runs(a.dir, tag)
        RUNS[tag] = runs
        rows, pss = [], []
        for s, run in sorted(runs.items()):
            sk, row = analyse_run(run, eta, s)
            sk.insert(0, "tag", tag); row["tag"] = tag; row["cpu_s"] = float(meta[(meta.tag == tag) & (meta.seed == s)].cpu_s.iloc[0])
            pss.append(sk); rows.append(row)
        sm, ps = pd.DataFrame(rows), pd.concat(pss, ignore_index=True)
        S.append(sm); PS.append(ps)
        P(f"\n## {tag}   (eta = {eta:.6g} = {eta * D * D:g}/d^2; seeds {sorted(runs)})")
        P(sm[["seed", "steps", "n_learned", "n_learned_first8", "n_at_init_ge05", "mse_t1000", "sum_n_end", "L_align_end", "L_mse_end", "n_neurons_mx_ge09", "n_neurons_mx_ge05",
              "max_mx_end", "cos_init_end_mean", "cos_init_end_min", "cpu_s"]].to_string(index=False))
        for col, nm in (("T05", "T_p(m>=0.5) [spec]"), ("T05d", "T_p,0.5 (drop midpoint) [1000-step grid]"), ("T05d_fine", "T_p,0.5 on merged 100-step early grid (supplementary)"),
                        ("T05_fine", "T_p(m>=0.5) on merged 100-step early grid (supplementary)")):
            P(f"\n{nm}: slope of log T vs log p over learned skills (T>0)")
            for _, r in sm.iterrows():
                P(f"  seed {int(r.seed)}: {r[f'slope_{col}']:.3f} ± {r[f'slope_{col}_se']:.3f} (n={int(r[f'slope_{col}_n'])}); p<=8: {r[f'slope8_{col}']:.3f} ± {r[f'slope8_{col}_se']:.3f}")
            sl, se, n = pooled(ps, col)
            ms = sm[f"slope_{col}"].values
            sl8, se8, n8 = pooled(ps, col, 8)
            sla, sea, na = pooled(ps, col, learned=False)
            P(f"  pooled over seeds: {sl:.3f} ± {se:.3f} (n={n});  mean of per-seed slopes {np.nanmean(ms):.3f} ± {np.nanstd(ms, ddof=1) / np.sqrt(np.isfinite(ms).sum()):.3f};  p<=8: {sl8:.3f} ± {se8:.3f} (n={n8});  all 16 skills (no learned filter): {sla:.3f} ± {sea:.3f} (n={na})")
        Ls = ps[ps.learned]
        P(f"\nsharpness (T_0.9-T_0.1)/T_mid, learned skills: per seed [mid=drop midpoint] {np.round(sm.sharp_d_median.values, 3).tolist()}, pooled median {np.nanmedian(Ls.sharp_d):.3f} (n={Ls.sharp_d.notna().sum()}); "
          f"[mid=T_p(0.5)] pooled {np.nanmedian(Ls.sharp_m[Ls.T05 > 0]):.3f}; fine-grid [mid=drop] pooled {np.nanmedian(Ls.sharp_d_fine):.3f}")
        P("\nscaling exponent of L_align = sum pi_p (1 - max_j m^2) between 3rd and 8th emergence (T_p(0.5) order statistics, as specified); drop-window = same with T_p,0.5 order statistics")
        P(sm[["seed", "t3", "t8", "win_n", "expo_align", "expo_mse", "d3", "d8", "win_n_drop", "expo_align_dropwin", "expo_mse_dropwin"]].to_string(index=False))
        ex = sm.expo_align.values
        P(f"  mean align exponent {np.nanmean(ex):.3f} ± {np.nanstd(ex, ddof=1) / np.sqrt(np.isfinite(ex).sum()):.3f} (SE over seeds); drop-window mean {np.nanmean(sm.expo_align_dropwin):.3f}; mse-based mean {np.nanmean(sm.expo_mse):.3f}")
        P("\nT_p tables (steps; nan never; 0 = aligned >= 0.5 at init): T05 | T05d | T05d_fine | m0 (best neuron at init)")
        for col in ("T05", "T05d", "T05d_fine", "m0"):
            w = ps.pivot(index="p", columns="seed", values=col)
            w.columns = [f"s{c}" for c in w.columns]
            P(f"-- {col}"); P(w.round(3 if col == "m0" else 0).to_string())
        P("\nINVARIANT CHECK (neuron j_end that ends on skill p; m0 = its initial |alignment| with v_p), learned skills pooled over seeds:")
        for nm, kk, desc in (("K1", "K1", "T eta a_p"), ("K2", "K2", "T eta a_p m0"), ("K3", "K3", "T eta a_p m0^2"), ("K4", "K4", "T eta a_p / ln(1/m0)")):
            for g in ("d", "dfine"):
                col = f"{kk}_{g}"
                for lab, Lx in (("learned", Ls), ("learned p<=8", Ls[Ls.p <= 8])):
                    v = Lx[col].values
                    v = v[np.isfinite(v) & (v > 0)]
                    pp = Lx.p.values[np.isfinite(Lx[col].values) & (Lx[col].values > 0)]
                    sl, se, n = reg(np.log(pp.astype(float)), np.log(v))
                    P(f"  {desc:24s} [{'1000-grid' if g == 'd' else 'fine grid'}] {lab:13s}: median {np.median(v):.4g}, IQR {np.percentile(v, 25):.4g}-{np.percentile(v, 75):.4g}, CV {v.std() / v.mean():.2f}, slope of log K vs log p {sl:.2f} ± {se:.2f} (n={n})")
        P("\nNEURONS PER SKILL (end): # neurons whose argmax_p |m| is p | # neurons with |m_jp| >= 0.3 | top-neuron share max_j n_j m_jp^2/a_p | pooled q_p = sum_j n_j m_jp^2 / a_p at end (init)")
        for s in sorted(ps.seed.unique()):
            q = ps[ps.seed == s]
            P(f"  seed {s}: assigned {q.n_assigned.tolist()}")
            P(f"          |m|>=0.3 {q.n_m_ge03.tolist()}")
            P(f"          top share {np.round(q.top_share_end.values, 2).tolist()}")
            P(f"          q_end {np.round(q.q_end.values, 3).tolist()}")
            P(f"          q_0 {np.round(q.q0.values, 3).tolist()}")
            P(f"          best-neuron final |m| {np.round(q.final_mx.values, 2).tolist()}")
    SS, PSS = pd.concat(S, ignore_index=True), pd.concat(PS, ignore_index=True)
    SS.to_csv(os.path.join(a.dir, "summary.csv"), index=False)
    PSS.to_csv(os.path.join(a.dir, "per_skill.csv"), index=False)
    open(os.path.join(a.dir, "analysis_stdout.txt"), "w").write("\n".join(lines))

    # ---------------------------------------------------------------- comparison figure: A, B (rho=0.0496), B-tied
    A2.SEED_MAX = 2
    ref = {}
    for key, tag in (("A", "A_a1.5"), ("B", "B_a1.5_rho0.0496")):
        runs = A2.load(a.exp2, tag)
        _, sm, ps = A2.analyse(a.exp2, tag, 1.5, [])
        ref[key] = ps
    primary = "Btied_eta1"
    cols = [("A", ref["A"], CA, "A (in-context)"), ("B", ref["B"], CB, "B (in-weight, projected, rho=0.0496)"),
            ("Bt", PSS[PSS.tag == primary], CT, "B-tied (2-homogeneous, eta=1/d^2)")]
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 7.6))
    for j, (key, ps, col, nm) in enumerate(cols):
        for i, (c, ttl) in enumerate((("T05", "T_p: first t with max_j|m_jp| >= 0.5"), ("T05d", "T_p,0.5: midpoint of alignment-loss drop"))):
            a_ = ax[i, j]
            for s, mk in zip(sorted(ps.seed.unique()), "os^"):
                q = ps[(ps.seed == s) & ps.learned & (ps[c] > 0)]
                a_.plot(q.p, q[c], mk, color=col, alpha=0.75, ms=5, label=f"seed {s}")
                q2 = ps[(ps.seed == s) & ~ps.learned & (ps[c] > 0)]
                a_.plot(q2.p, q2[c], mk, mfc="none", color=INK2, alpha=0.6, ms=5)
            sl, se, n = pooled(ps, c)
            L = ps[ps.learned & (ps[c] > 0)]
            if n >= 3:
                r = stats.linregress(np.log(L.p.values.astype(float)), np.log(L[c].values))
                xs = np.array([1, 16.0])
                a_.plot(xs, np.exp(r.intercept) * xs ** r.slope, "-", color=INK, lw=1.2, label=f"fit slope {sl:.2f} ± {se:.2f} (n={n})")
                ref_s = 1.5 if key == "A" else 0.75
                a_.plot(xs, np.exp(r.intercept) * xs ** ref_s, "--", color=INK2, lw=1, label=f"pre-registered slope {ref_s:.2f}")
            a_.set_xscale("log"); a_.set_yscale("log"); a_.set_xlabel("skill index p"); a_.set_ylabel("steps")
            a_.set_title(f"{nm}\n{ttl}", fontsize=9)
            a_.legend(fontsize=7, loc="upper left")
    fig.suptitle("Emergence time vs skill index, alpha = 1.5 (open grey: not learned). B-tied: decoupled in-weight baseline (note: its transitions happen within the first ~1-3k steps; logs every 1000 steps)", fontsize=9.5)
    fig.tight_layout()
    os.makedirs(os.path.join(a.exp2, "figs"), exist_ok=True)
    fig.savefig(os.path.join(a.exp2, "figs", "Tp_vs_p_with_Btied.png")); plt.close(fig)

    # supplementary figure: B-tied internals (seed 0, eta=1)
    run = RUNS[primary][0]
    t = run["t"].astype(float); m = np.abs(run["m"].astype(float)); mx = m.max(axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cmap = plt.get_cmap("viridis")
    tf, mf = run["fine_t"].astype(float), np.abs(run["fine_m"].astype(float)).max(axis=1)
    for p in range(P_):
        ax[0].plot(np.maximum(tf, 50), mf[:, p], color=cmap(p / 15), lw=1)
    ax[0].set_xscale("log"); ax[0].set_xlabel("step (100-step early log)"); ax[0].set_ylabel("max_j |m_jp|"); ax[0].set_title("B-tied seed 0: best alignment per skill")
    for j in range(64):
        ax[1].plot(np.maximum(run["fine_t"], 50), run["fine_norm2"][:, j], color=INK2, lw=0.5, alpha=0.5)
    ax[1].set_xscale("log"); ax[1].set_xlabel("step"); ax[1].set_ylabel("||u_j||^2"); ax[1].set_title("neuron norms^2 (64 neurons)")
    ax[2].plot(np.maximum(run["fine_t"], 50), run["fine_mse"], color=CT, label="MSE (exact)")
    ax[2].plot(t[1:], run["mse_total"][1:], color=CT, ls=":")
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("step"); ax[2].set_title("population MSE")
    fig.tight_layout(); fig.savefig(os.path.join(a.exp2, "figs", "Btied_internals_seed0.png")); plt.close(fig)


if __name__ == "__main__":
    main()
