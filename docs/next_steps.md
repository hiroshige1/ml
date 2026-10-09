# Standing backlog (worked autonomously; each item: pre-register if it is an experiment → run → record → commit → push)

Order is priority. Mark items `[x]` when done with the commit hash. Never delete items; add new ones at the bottom with a date.

- [x] (P21 pre-registered 7a9a670; P21a/b failed, P21c held, kill met; results/exp8; paper v0.12) **N1. Transformer trap-vs-tradeoff test (exp 8).** The three-regime statement predicts, for a transformer: readout *frozen at an O(1)
  norm* (pinned) ⇒ `N=16` stays stuck even with `B=1024` (no tradeoff); trainable readout (free) ⇒ emerges with a tradeoff (= exp 5's result);
  readout frozen *small* ⇒ `N`-flat. Design: `d=256`, width 32 as exp 5, cells `N∈{16,256}`, `B∈{64,1024}` for `N=16`, three readout
  protocols (frozen at init-norm×10, trainable, frozen at init-norm/10), 2–3 seeds, 30k steps, cap 6 CPU-h. Pre-register as P21 first.
- [x] (exp 10: 692081e prereg, 10213c4 results; P23a-c held, P23d secant baseline error; paper v0.13) **N2. Second `d` for the three-regime test.** Pinned `γ∈{1,0.1}` and tied at `d=128`, `N∈{16,64,256}` (free only at `N=256`, budget
  permitting); ODE predictions first. Checks that the `d`-dependence of the regimes is as stated (trap for `d>d*(N)`; tied `N`-flat).
- [x] (exp 9: 3de17b5, diagnostics 9899684; P22c held, P22a/b failed as worded; paper v0.11) **N3. `k*=3` sanity run for one regime.** Tied readout at `d=16`, `N=128` (exponent `2k*=6` predicted; ODE first); pinned `γ=0.1` trap
  check (drift negative below `m≈0.25` per the drift check). Small.
- [x] (review in docs/review_adversarial_3.md; fixes in v0.8; theorem drafting delegated) **N4. Third adversarial review** on v0.7 with a "theory-rigour" persona (what would need to be proved for Prop. 1–3 to be theorems;
  which statements are heuristic), then respond.
- [x] (paper/main_icml.tex with icml2026.sty: main text 6.2 pp two-column; shared body.tex; docs/venue.md) **N5. Venue formatting.** Obtain the ICML 2026 LaTeX style (icml2026.sty from the ICML site or CTAN mirror) as a proxy for 2027,
  reflow the paper to two columns, measure main-text length against a 4–8 page workshop limit, move material to the appendix as needed.
- [x] (LICENSE, CITATION.cff; dedup + README headers + Tests section in b9c8586) **N6. Repository hygiene.** Deduplicate logs in `results/`, make every `results/exp*/README.md` state CPU time and deviations in the
  same format, add `LICENSE` (MIT) and `CITATION.cff`, confirm `python3 tests/*.py` pass from a fresh clone.
- [x] (docs/memo_ja.md) **N7. Japanese research memo** (`docs/memo_ja.md`): the question, the three regimes, what is new vs. known (table), what failed,
  what the transformer results do and do not show — for the human author to decide the framing.
- [~] (checked 2026-10-09 ×2: icml.cc/Conferences/2027 is 404, HiLD site still on 2026; docs/venue.md) **N8. Monitor ICML 2027 / HiLD CFP.** Check `icml.cc` and the HiLD site once per wake; when a CFP appears, record dates, page limit,
  LLM policy, archival status in `docs/venue.md`.

- [x] (paper/theorems.tex, scripts/verify_theorems.py 61/61; paper v0.10; prereg v0.21) **N9 (2026-10-09, from review 3). Theorems A/B.** Replace Prop. 3 by (A) finite-context trap for σ_k population flow incl. tied=pinned time
  change and SGD corollary; (B) free-readout asymptotics τ = (1+o(1))·exp((4k−2)β₂²Γ₀²/(2r))·d^{2k−1}/(4k(4k−2)α⁸N) at fixed N, joint-scaling
  limit, corner exponents 4k/2k/2k+1. Proof drafts with sympy checks → appendix.
- [x] (scripts/ode_lsoda.py, results/README_ode.md; paper v0.9) **N10 (2026-10-09). Accurate ODE integration.** Recompute all quoted ODE predictions with LSODA (rtol 1e-10); the Euler integrator may
  overestimate by 1–4%; update figures/tables where the change exceeds 1%.

- [x] (exp 11: ff2cc7d prereg, 9eb8f7d results; P24a-c failed, kill met; paper v0.15) **N11 (2026-10-09, from exp 8). Transformer: control the value/output projection instead of the final readout.** Freeze the norm of the
  attention output projection (and separately the MLP output) at O(1) and at 0.1×, same cells as exp 8; pre-register as P24 with the same
  trap/rescue predictions. Only if budget allows (≈6 CPU-h); otherwise leave as a stated open test in the paper.

- [ ] **N12 (2026-10-09, from exp 11). Fourth adversarial review** of v0.15 (persona: workshop area chair; 4-page cut plan; is the negative transformer
  result stated at the right strength; are E6/E7 and the pre-registration errors presented honestly), then respond in the paper.
