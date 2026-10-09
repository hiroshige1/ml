# Standing backlog (worked autonomously; each item: pre-register if it is an experiment → run → record → commit → push)

Order is priority. Mark items `[x]` when done with the commit hash. Never delete items; add new ones at the bottom with a date.

- [ ] **N1. Transformer trap-vs-tradeoff test (exp 8).** The three-regime statement predicts, for a transformer: readout *frozen at an O(1)
  norm* (pinned) ⇒ `N=16` stays stuck even with `B=1024` (no tradeoff); trainable readout (free) ⇒ emerges with a tradeoff (= exp 5's result);
  readout frozen *small* ⇒ `N`-flat. Design: `d=256`, width 32 as exp 5, cells `N∈{16,256}`, `B∈{64,1024}` for `N=16`, three readout
  protocols (frozen at init-norm×10, trainable, frozen at init-norm/10), 2–3 seeds, 30k steps, cap 6 CPU-h. Pre-register as P21 first.
- [ ] **N2. Second `d` for the three-regime test.** Pinned `γ∈{1,0.1}` and tied at `d=128`, `N∈{16,64,256}` (free only at `N=256`, budget
  permitting); ODE predictions first. Checks that the `d`-dependence of the regimes is as stated (trap for `d>d*(N)`; tied `N`-flat).
- [ ] **N3. `k*=3` sanity run for one regime.** Tied readout at `d=16`, `N=128` (exponent `2k*=6` predicted; ODE first); pinned `γ=0.1` trap
  check (drift negative below `m≈0.25` per the drift check). Small.
- [ ] **N4. Third adversarial review** on v0.7 with a "theory-rigour" persona (what would need to be proved for Prop. 1–3 to be theorems;
  which statements are heuristic), then respond.
- [ ] **N5. Venue formatting.** Obtain the ICML 2026 LaTeX style (icml2026.sty from the ICML site or CTAN mirror) as a proxy for 2027,
  reflow the paper to two columns, measure main-text length against a 4–8 page workshop limit, move material to the appendix as needed.
- [ ] **N6. Repository hygiene.** Deduplicate logs in `results/`, make every `results/exp*/README.md` state CPU time and deviations in the
  same format, add `LICENSE` (MIT) and `CITATION.cff`, confirm `python3 tests/*.py` pass from a fresh clone.
- [ ] **N7. Japanese research memo** (`docs/memo_ja.md`): the question, the three regimes, what is new vs. known (table), what failed,
  what the transformer results do and do not show — for the human author to decide the framing.
- [ ] **N8. Monitor ICML 2027 / HiLD CFP.** Check `icml.cc` and the HiLD site once per wake; when a CFP appears, record dates, page limit,
  LLM policy, archival status in `docs/venue.md`.
