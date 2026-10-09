"""Analysis of exp 14 (P27): per-B table, monotonicity checks, verdict, results/exp14/README.md.  Run after ablation_exp14 --collect."""
import csv
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "exp14")
BS = (16, 64, 256, 1024)
INF = float("inf")


def load():
    rows = list(csv.DictReader(open(os.path.join(OUT, "runs.csv"))))
    by = {B: {} for B in BS}
    for r in rows:
        by[int(r["B"])][int(r["seed"])] = r
    return rows, by


def med(xs):
    return statistics.median(xs)  # inf-aware: median of 3 values, censored = inf


def fmt_t(t):
    return "CENS" if t == INF else f"{int(t)}"


def analyse():
    rows, by = load()
    tab = {}
    for B in BS:
        ts = []
        for s in sorted(by[B]):
            r = by[B][s]
            ts.append(float(r["T05"]) if int(r["reached05"]) else INF)
        tab[B] = dict(T=ts, n=len(ts), esc=sum(t < INF for t in ts), frac=sum(t < INF for t in ts) / len(ts), med=med(ts),
                      seeds=sorted(by[B]), src=[by[B][s]["source"] for s in sorted(by[B])],
                      cpu=[float(by[B][s]["cpu_s"]) for s in sorted(by[B])],
                      final=[float(by[B][s]["final_abs_m"]) for s in sorted(by[B])],
                      maxm=[float(by[B][s]["max_abs_m"]) for s in sorted(by[B])])
    fr = [tab[B]["frac"] for B in BS]
    md = [tab[B]["med"] for B in BS]
    pairs = list(zip(BS[:-1], BS[1:]))
    frac_up = [(a, b) for a, b in pairs if tab[b]["frac"] > tab[a]["frac"] + 1e-12]
    frac_tie = [(a, b) for a, b in pairs if abs(tab[b]["frac"] - tab[a]["frac"]) < 1e-12]
    med_down = [(a, b) for a, b in pairs if tab[b]["med"] < tab[a]["med"]]
    med_tie = [(a, b) for a, b in pairs if tab[b]["med"] == tab[a]["med"]]
    # "up to one tie": monotone (no strict reversal), with at most one adjacent pair tied
    m1 = len(frac_up) == 0 and len(frac_tie) <= 1
    m2 = len(med_down) == 0 and len(med_tie) <= 1
    m1_lit = len(frac_up) == 0  # literal non-increasing, any number of ties
    m2_lit = len(med_down) == 0
    c16 = tab[16]["frac"] >= 2 / 3 - 1e-12
    c1024 = tab[1024]["frac"] <= 2 / 3 + 1e-12
    p27 = m1 and m2 and c16 and c1024
    increases = tab[1024]["frac"] > tab[16]["frac"] and len(frac_up) > 0 and len(frac_up) >= len([1 for a, b in pairs if tab[b]["frac"] < tab[a]["frac"]])
    res = dict(table={str(B): dict(T05=[None if t == INF else t for t in tab[B]["T"]], escapes=tab[B]["esc"], n=tab[B]["n"],
                                    escape_fraction=tab[B]["frac"], median_T05=None if tab[B]["med"] == INF else tab[B]["med"])
                      for B in BS},
               frac_increases_pairs=frac_up, frac_tie_pairs=frac_tie, med_decreases_pairs=med_down, med_tie_pairs=med_tie,
               mono_escape_fraction_nonincreasing_up_to_one_tie=m1, mono_median_T05_nondecreasing_up_to_one_tie=m2,
               literal_escape_fraction_nonincreasing=m1_lit, literal_median_nondecreasing=m2_lit,
               B16_escapes_ge_2of3=c16, B1024_escapes_le_2of3=c1024, P27_held=p27, default_escape_fraction_increases_with_B=increases)
    return rows, tab, res


def readme(rows, tab, res):
    cpu = sum(float(r["cpu_s"]) for r in rows if r["source"] == "exp14")
    cpu_e13 = sum(float(r["cpu_s"]) for r in rows if r["source"] == "exp13")
    L = []
    L.append("# Experiment 14: is the softmax + Adam escape noise-driven? B-dependence at N=16\n")
    L.append("Pre-registration: `docs/preregistration.md`, section \"Is the softmax+Adam escape noise-driven? B-dependence at N=16 (exp 14)\" (P27; committed in d984c95 before any run).")
    L.append("Code: `icl_additive/ablation_exp14.py` (imports and calls exp 13's `run` unchanged; torch fp32, CPU, 1 thread per process, 4 workers with a claim-file job queue), analysis `icl_additive/analyze_exp14.py`.")
    L.append("Raw rows: `runs.csv` (12 rows; `source=exp13` for the 4 reused runs, `source=exp14` for the 8 fresh ones; `censored_by` empty if |m| >= 0.5 was reached, else `cap`), `rows/*.json`, logs `logs/worker*.log`, `verdicts.json`, `table.md`; traces `traces/*.npz` (m every 100 steps; gitignored).")
    L.append(f"**CPU time:** {cpu / 3600:.3f} h for the 8 fresh runs (cap 1.5 h; sum of per-run `process_time`; the 4 reused exp 13 runs, {cpu_e13 / 3600:.3f} h, are not counted).\n")
    L.append("## Setup (as pre-registered)\n")
    L.append("Exactly exp 13's rbf + Adam cell: d = 64, sigma_2, `m_0 = d^-1/2 = 0.125`, pinned readout Gamma = 1, `beta = 0.3`, Adam lr 1e-3 (default betas), cap 1e5 steps, `w <- w/|w|` every step, stop at `|m| >= 0.5`, full d=64 data. N = 16, B in {16, 64, 256, 1024}, seeds {0, 1, 2}. Seeds 0-1 at B = 64 and B = 1024 are exp 13's runs (reused, not re-run); the other 8 are fresh.\n")
    L.append("## Table\n")
    L.append("CENS = no `|m| >= 0.5` within 1e5 steps (counted as infinity in the median). `T_0.5` is in optimiser steps, seeds 0 / 1 / 2.\n")
    hdr = "| B | T_0.5 (s0 / s1 / s2) | escapes | escape fraction | median T_0.5 | final \\|m\\| (s0 / s1 / s2) | max \\|m\\| | source (s0 / s1 / s2) |"
    T = [hdr, "|---|---|---|---|---|---|---|---|"]
    for B in BS:
        t = tab[B]
        T.append(f"| {B} | {' / '.join(fmt_t(x) for x in t['T'])} | {t['esc']}/{t['n']} | {t['frac']:.2f} | {fmt_t(t['med'])} | "
                 f"{' / '.join(f'{x:.3f}' for x in t['final'])} | {' / '.join(f'{x:.3f}' for x in t['maxm'])} | {' / '.join(t['src'])} |")
    L += T
    open(os.path.join(OUT, "table.md"), "w").write("\n".join(T) + "\n")
    L.append("")
    L.append("## Monotonicity checks and verdict\n")
    L.append("Criterion (quoted from the pre-registration): \"the escape fraction within `10^5` steps is non-increasing in `B` and the median `T_0.5` (censored = infinity) is non-decreasing in `B`; in particular `B=16` escapes >= 2/3 and `B=1024` <= 2/3\"; pass: \"both monotonicities hold up to one tie\"; default: \"escape fraction increases with `B` (drift-driven, transformer-like)\".\n")
    L.append("Reading of \"up to one tie\" (mine, fixed before looking at the numbers): a monotonicity holds if no adjacent pair (16-64, 64-256, 256-1024) reverses strictly and at most one adjacent pair is tied. The literal reading (no strict reversal, any number of ties) is reported alongside.\n")
    fs = " -> ".join(f"{tab[B]['frac']:.2f}" for B in BS)
    ms = " -> ".join(fmt_t(tab[B]["med"]) for B in BS)
    L.append(f"- Escape fraction by B (16, 64, 256, 1024): {fs}. Strict increases at {res['frac_increases_pairs'] or 'none'}; ties at {res['frac_tie_pairs'] or 'none'}. Non-increasing up to one tie: **{'yes' if res['mono_escape_fraction_nonincreasing_up_to_one_tie'] else 'no'}** (literal reading: {'yes' if res['literal_escape_fraction_nonincreasing'] else 'no'}).")
    L.append(f"- Median T_0.5 by B: {ms}. Strict decreases at {res['med_decreases_pairs'] or 'none'}; ties at {res['med_tie_pairs'] or 'none'}. Non-decreasing up to one tie: **{'yes' if res['mono_median_T05_nondecreasing_up_to_one_tie'] else 'no'}** (literal reading: {'yes' if res['literal_median_nondecreasing'] else 'no'}).")
    L.append(f"- B = 16 escapes >= 2/3: {tab[16]['esc']}/{tab[16]['n']} -> **{'yes' if res['B16_escapes_ge_2of3'] else 'no'}**; B = 1024 escapes <= 2/3: {tab[1024]['esc']}/{tab[1024]['n']} -> **{'yes' if res['B1024_escapes_le_2of3'] else 'no'}**.\n")
    if res["P27_held"]:
        v = "**P27 HELD.** Both monotonicities and both end-point conditions hold: the softmax + Adam escape at N=16 gets harder with more prompts per step, i.e. it is noise-driven, the opposite of the transformer's B-rescue."
    elif res["default_escape_fraction_increases_with_B"]:
        v = ("**P27 FAILED; the default holds.** The escape fraction increases with B: the single-feature softmax + Adam model does reproduce the transformer-like, drift-driven "
             "behaviour (larger batches help), so the escape is not noise-driven in the sense hypothesised. The \"not a B-rescue\" sentence of paper v0.19 is withdrawn.")
    elif res["literal_escape_fraction_nonincreasing"] and res["literal_median_nondecreasing"] and res["B16_escapes_ge_2of3"] and res["B1024_escapes_le_2of3"]:
        v = ("**P27 FAILED on the declared (strict) reading of \"up to one tie\", but only through saturation; it HOLDS on the literal reading.** "
             "No pair reverses and both end-point conditions hold, but the escape fraction is tied in more than one adjacent pair (it is 3/3 at B = 16, 64, 256 and drops only at B = 1024), so the pre-registered "
             "\"up to one tie\" allowance is exceeded. The default (escape fraction increases with B) is NOT supported: the escape fraction never increases with B and the median T_0.5 rises monotonically with B. "
             "The data therefore lean towards the noise-driven reading (larger B is not helpful and is slower), but the only escape-fraction evidence is a single stuck seed at B = 1024 (2/3).")
    else:
        v = ("**P27 FAILED; the default (escape fraction increases with B) is not cleanly met either.** The pre-registered criterion is not satisfied (see the checks above); "
             "the data are described as they are: the escape fraction does not increase with B and does not decrease monotonically.")
    L.append(v + "\n")
    L.append("## Deviations\n")
    L.append("- None from the pre-registered design: the model, optimiser, cap, seeds and cells are exp 13's; `run` is imported, not copied. The only code-level difference is that the STOP-file location read by `stop_requested` is redirected to `results/exp14/` (no STOP file was used).")
    L.append("- Data streams: `torch.Generator` seeded from `SeedSequence([seed, 13, opt, N, B])` and the teacher / initial `w` from `[seed, 64, 13]`, exactly as in exp 13, so the reused runs are the same runs this script would produce for those (B, seed).")
    L.append("- The interpretation of \"up to one tie\" is mine (stated above); the literal reading is also shown.")
    open(os.path.join(OUT, "README.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    rows, tab, res = analyse()
    json.dump(res, open(os.path.join(OUT, "verdicts.json"), "w"), indent=1)
    readme(rows, tab, res)
    print(open(os.path.join(OUT, "table.md")).read())
    print(json.dumps({k: v for k, v in res.items() if k != "table"}, indent=1))
