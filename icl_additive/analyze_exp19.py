"""Exp 19 analysis (P32): transformer B-scan at N=16 with 3 seeds at B in {384, 512, 768}.
Fresh rows: results/exp19/summary_raw.csv (cap 40000 steps).  Re-used rows: B=256 (exp 17, 80k cap, stuck 2/2), B=512 seeds 0,1 (exp 16, 30k cap;
comparison column only, NOT used in the medians), B=1024 seeds 0,1,2 (exp 5 cell d), B=2048 seed 0 (exp 16).
Medians use T_e(0.5) with censored (not reached / killed) runs at the cap CAP=40000 (the pre-registered cap of this experiment).
Writes results/exp19/summary.csv and results/exp19/README.md."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "exp19")
CAP = 40000
SCAN = [384, 512, 768, 1024]
REF1024_PREREG = 12700.0  # the value quoted in the pre-registration (median of seeds 0,1 of exp 5)
COLS = ["B", "seed", "source", "used_in_median", "cap", "steps_run", "Te05", "Te025", "reached05", "reached025", "final_mse", "max_align", "censored_by", "cpu_s"]


def ols(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    n = len(x)
    sxx = ((x - x.mean()) ** 2).sum()
    b = ((x - x.mean()) * (y - y.mean())).sum() / sxx
    a = y.mean() - b * x.mean()
    se = np.sqrt(((y - a - b * x) ** 2).sum() / (n - 2) / sxx)
    return float(b), float(se)


def load():
    fresh = pd.read_csv(os.path.join(OUT, "summary_raw.csv"))
    fresh = fresh[(fresh.proto == "free") & (fresh.N == 16)].copy()
    fresh["source"], fresh["used_in_median"], fresh["cap"] = "exp19", 1, CAP
    e17 = pd.read_csv(os.path.join(ROOT, "results", "exp17", "summary_raw.csv"))
    e17 = e17[(e17.proto == "free") & (e17.N == 16) & (e17.B == 256)].copy()
    e17["source"], e17["used_in_median"], e17["cap"] = "exp17", 0, 80000
    e16 = pd.read_csv(os.path.join(ROOT, "results", "exp16", "summary_raw.csv"))
    e16 = e16[(e16.proto == "free") & (e16.N == 16) & (e16.B.isin([512, 2048]))].copy()
    e16["source"], e16["cap"] = "exp16", 30000
    e16["used_in_median"] = 0  # B=512 comparison column; B=2048 not in the P32 scan
    e5 = pd.read_csv(os.path.join(ROOT, "results", "exp5", "summary.csv"))
    e5 = e5[(e5.N == 16) & (e5.k == 2) & (e5.inweight == 0) & (e5.B == 1024) & (e5.seed.isin([0, 1, 2]))].copy()
    e5["source"], e5["used_in_median"], e5["cap"] = "exp5", 1, 30000
    e5["censored_by"] = np.where(e5.reached025 == 1, "", "max_steps")
    df = pd.concat([d.reindex(columns=COLS) for d in (e17, e16, e5, fresh)], ignore_index=True)
    df["censored_by"] = df.censored_by.fillna("")
    df["B"], df["seed"] = df.B.astype(int), df.seed.astype(int)
    return df.sort_values(["B", "seed", "source"]).reset_index(drop=True)


def fmt(v):
    return "n/r" if pd.isna(v) else str(int(v))


def main():
    df = load()
    df[COLS].to_csv(os.path.join(OUT, "summary.csv"), index=False)
    u = df[df.used_in_median == 1].copy()
    u["Te05_c"], u["Te025_c"] = u.Te05.fillna(CAP), u.Te025.fillna(CAP)
    med = {}
    for B in SCAN:
        g = u[u.B == B]
        med[B] = dict(n=len(g), r05=int(g.reached05.sum()), r025=int(g.reached025.sum()),
                      m05=float(np.median(g.Te05_c)), m025=float(np.median(g.Te025_c)))
    slope, se = ols(SCAN, [med[B]["m05"] for B in SCAN])
    slope25, se25 = ols(SCAN, [med[B]["m025"] for B in SCAN])
    # sensitivity: 384 excluded etc. is not pre-registered; only the 1024 reference value is varied
    ref_alt = med[1024]["m05"]
    frac = {B: med[B]["r05"] / med[B]["n"] for B in SCAN}
    def within(m, ref):
        return ref / 1.5 <= m <= ref * 1.5
    # P32-threshold
    t1 = frac[384] <= 1 / 3 + 1e-9
    t2 = frac[512] >= 2 / 3 - 1e-9
    t3 = med[768]["r05"] == med[768]["n"] == 3
    t4 = within(med[512]["m05"], REF1024_PREREG) and within(med[768]["m05"], REF1024_PREREG)
    t4alt = within(med[512]["m05"], ref_alt) and within(med[768]["m05"], ref_alt)
    thr = t1 and t2 and t3 and t4
    # P32-graded
    g1 = med[384]["m05"] > med[512]["m05"] > med[768]["m05"]
    g2 = slope <= -1.3
    g3 = frac[384] >= 2 / 3 - 1e-9
    gra = g1 and g2 and g3
    verdict = "P32-threshold" if thr and not gra else "P32-graded" if gra and not thr else "neither" if not thr and not gra else "both (inconsistent)"
    yn = lambda c: "yes" if c else "**NO**"
    L = ["| B | seed | source | cap | steps run | T_e(0.5) | T_e(0.25) | final MSE | max align | censored | CPU s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.B} | {r.seed} | {r.source}{'' if r.used_in_median else ' (not in median)'} | {int(r.cap)} | {int(r.steps_run)} | {fmt(r.Te05)} | {fmt(r.Te025)} | {r.final_mse:.3f} | {r.max_align:.2f} | {r.censored_by or '-'} | {r.cpu_s:.0f} |")
    M = ["| B | runs | escape T_e(0.5) | escape T_e(0.25) | median T_e(0.5) | median T_e(0.25) | exp16 B=512 (comparison) |", "|---|---|---|---|---|---|---|"]
    for B, m in med.items():
        cmpc = ""
        if B == 512:
            c = df[(df.B == 512) & (df.source == "exp16")].sort_values("seed")
            cmpc = "; ".join(f"s{r.seed} {fmt(r.Te05)}" for _, r in c.iterrows()) + " (cap 30000)"
        plus = lambda k, r: "+" if r * 2 <= m["n"] else ""
        M.append(f"| {B} | {m['n']} | {m['r05']}/{m['n']} | {m['r025']}/{m['n']} | {m['m05']:.0f}{plus('m05', m['r05'])} | {m['m025']:.0f}{plus('m025', m['r025'])} | {cmpc} |")
    fresh = df[df.source == "exp19"]
    fresh_cpu = fresh.cpu_s.sum()
    perseed = "; ".join(f"B={B}: " + ", ".join(f"s{r.seed} {fmt(r.Te05)}" for _, r in u[u.B == B].sort_values("seed").iterrows()) for B in SCAN)
    dev_path = os.path.join(OUT, "deviations.txt")
    dev = open(dev_path).read().strip() if os.path.exists(dev_path) else "None recorded."
    text = f"""# Exp 19: transformer B-scan at N=16 with 3 seeds (threshold or steep power law?), prediction P32 in `docs/preregistration.md`

- **Pre-registration:** `docs/preregistration.md`, block "Transformer B-scan at N=16 with more seeds: threshold or steep power law? (exp 19, N26 part 2)" (commit d6b622c), prediction P32.
- **Code:** `icl_additive/transformer_exp8.py --run free 16 B SEED --out results/exp19/ --max-steps 40000` (unchanged; exp-5 model/data/eval), launcher `results/exp19/run_queue.sh`, analysis `icl_additive/analyze_exp19.py`. Logs in `logs/`, curves in `curves/*.npz` (not tracked).
- **CPU time:** {fresh_cpu:.0f} s = {fresh_cpu / 3600:.2f} CPU-h for the {len(fresh)} fresh runs (cap 10 CPU-h = 36000 s); 1 thread per process (OMP_NUM_THREADS=1), at most 3 processes at once.
- **Deviations:** see the section at the end.

Setting as exps 5/16/17 free: d=256, d_model=64, MLP width 32, 2 pre-LN blocks, 4-head causal softmax attention, trainable linear readout, sigma_2 link, Adam 1e-3, eval every 200 steps on 2048 held-out prompts,
T_e = first eval MSE/E[y^2] < 0.5 (and < 0.25), early stop at < 0.25 or 40000 steps. `N=16`, seeds 0,1,2 (same teacher/init per seed as exps 5/16/17). Rows with `source` exp5/exp16/exp17 are re-used, not re-run.

## Per-run results (`n/r` = not reached within the steps run; censored runs enter medians at the cap {CAP})

{chr(10).join(L)}

## Escape fractions and medians (scan B in {{384, 512, 768, 1024}}; `+` = median is a censored lower bound)

{chr(10).join(M)}

Per-seed T_e(0.5): {perseed}. B=256 (exp 17): stuck 2/2 to 80k/77k steps.

OLS of log(median T_e) on log B over B in {{384, 512, 768, 1024}} (4 points, 2 dof):
- T_e(0.5): slope **{slope:.3f} +- {se:.3f}** (SE)
- T_e(0.25): slope {slope25:.3f} +- {se25:.3f} (secondary; P32 is stated for T_e(0.5), as P29)

## P32 verdict

Criteria (quoted): "P32-threshold: `B=384` escapes <= 1/3 and `B=512` >= 2/3 and `B=768` 3/3, with `T_e(512)` and `T_e(768)` medians within x1.5 of `T_e(1024)`'s (12.7k)"; "P32-graded: medians (censored = cap) strictly decrease from `384` to `768` **and** the OLS slope of `log T_e` vs `log B` over `{{384, 512, 768, 1024}}` is <= -1.3 with `B=384` escaping >= 2/3"; otherwise "neither pattern; n=3 insufficient".

Operationalisation (mine): T_e is T_e(0.5); the 1024 reference in the threshold criterion is the quoted 12.7k (band [{REF1024_PREREG / 1.5:.0f}, {REF1024_PREREG * 1.5:.0f}]); with the 3-seed B=1024 median {ref_alt:.0f} (exp 5 seeds 0-2: 14800, 10600, 11600) the band would be [{ref_alt / 1.5:.0f}, {ref_alt * 1.5:.0f}] - the within-x1.5 clause is {'unchanged' if t4 == t4alt else 'DIFFERENT'} under this alternative ({'holds' if t4alt else 'fails'}). Escape = T_e(0.5) reached within the run (killed/capped = not escaped).

| clause | observed | holds? |
|---|---|---|
| T1. B=384 escapes <= 1/3 | {med[384]['r05']}/{med[384]['n']} | {yn(t1)} |
| T2. B=512 escapes >= 2/3 | {med[512]['r05']}/{med[512]['n']} | {yn(t2)} |
| T3. B=768 escapes 3/3 | {med[768]['r05']}/{med[768]['n']} | {yn(t3)} |
| T4. medians T_e(512), T_e(768) within x1.5 of 12.7k | {med[512]['m05']:.0f}, {med[768]['m05']:.0f} (alt. ref {ref_alt:.0f}: {'holds' if t4alt else 'fails'}) | {yn(t4)} |
| G1. medians strictly decrease 384 > 512 > 768 | {med[384]['m05']:.0f}, {med[512]['m05']:.0f}, {med[768]['m05']:.0f} | {yn(g1)} |
| G2. OLS slope <= -1.3 | {slope:.3f} +- {se:.3f} | {yn(g2)} |
| G3. B=384 escapes >= 2/3 | {med[384]['r05']}/{med[384]['n']} | {yn(g3)} |

P32-threshold: **{'holds' if thr else 'fails'}**; P32-graded: **{'holds' if gra else 'fails'}**.

**Verdict: {verdict if verdict != 'neither' else 'neither pattern; n=3 insufficient'}.**
n=3 seeds per cell, 4 points, 2 degrees of freedom: the slope SE is a rough guide only.

## Observation (not a pre-registered criterion)

Seed 1 never leaves the plateau at B = 384, 512 (exp 16: 30k; here 40k) or 768 (all at MSE ~1.06, max align 0.31-0.33), but escapes at B = 1024 (10600) and 2048 (exp 16 seed 0 only); seed 2 escapes at B = 512 (12400) and 768 (4400) but not 384; seed 0 escapes at B = 384 (17800), 512, 768. Escape fractions are 1/3, 2/3, 2/3, 3/3 for B = 384, 512, 768, 1024, rising with B but not reaching 3/3 at 768 (T3 fails by one seed), and the escape times among escapers do not decrease strictly with B (17800, 12400-13400, 4400-11000, 10600-14800). Seed-to-seed variation at fixed B is as large as the B effect, which is why n=3 does not separate the two readings.

## CPU time

Fresh runs: {fresh_cpu:.0f} s ({fresh_cpu / 3600:.2f} CPU-h) summed over `cpu_s` of the exp19 rows (cap 10 CPU-h = 36000 s). Re-used rows cost nothing new.

## Deviations / what was not done

{dev}
"""
    open(os.path.join(OUT, "README.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
