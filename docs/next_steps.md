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

- [x] (docs/review_adversarial_4.md 6/10; response docs/review_adversarial_4_response.md; paper v0.16) **N12 (2026-10-09, from exp 11). Fourth adversarial review** of v0.15 (persona: workshop area chair; 4-page cut plan; is the negative transformer
  result stated at the right strength; are E6/E7 and the pre-registration errors presented honestly), then respond in the paper.

- [x] (paper/theorems.tex Thm A(iii)(c) + rem:tiedflat; scripts/verify_tied_flat.py 80/80; paper v0.17) **N13 (2026-10-09, from review 4). Prove the tied-readout N-flatness (Theorem A(iii)(c)).** Sketch: along the tied flow with m ≤ 1/2,
  dρ/dm = 8ρ²(g² − ρD)/ṁ with ṁ = (1−m²)[4gg' − ρ(2(1−1/N)gg' + V'/N)]/ρ ≥ (1−m²)4gg'(1−ρ/ρ*(m))/ρ, so 1/ρ moves by at most
  ∫ m/((1−m²)(1−ρ/ρ*)) dm (upward in ρ) and at most ∫ 2Dρ/((1−m²)2m^{2k−1}(1−ρ/ρ*)) dm ≈ ρ_max/((2k−2)N m₀^{2k−2}) (downward);
  hence ρ ∈ [ρ₀/(1+Cρ₀/(N m₀^{2k−2})), ρ₀/(1−C'ρ₀)] on [m₀,1/2] whenever ρ₀ ≤ ρ*(m₀)/4. Then τ_tied = ∫ dm/(4k m^{2k−1}(1 − ρ/ρ*(m))(1+O(ρ)))
  is sandwiched: |τ_tied·(2k−2)4k m₀^{2k−2} − 1| ≤ C(ρ₀ k/(N m₀^{2k−2}) + ρ₀ + m₀²). Verify numerically against LSODA over the exp 6/7/9/10
  tied cells, add to paper/theorems.tex, cite from Prop. 3(c), update the review-4 response.
- [x] (paper/main_icml_4p.tex, body_4p.tex, appendix_figs_4p.tex; 4 pp main text; docs/venue.md) **N14 (2026-10-09, from review 4). Four-page variant** `paper/main_icml_4p.tex` following the review's cut plan (for the author to choose; the
  full version stays the main one).

- [x] (docs/proof_check_A3c.md: true and proved, 600 random cells pass; two wording slips fixed) **N15 (2026-10-09). Independent proof check of Theorem A(iii)(c)** by a second agent that has not seen the proof's derivation: verify every inequality
  (ρ comparison, the (1−m²) and O(ρ) terms, the sandwich constants C_k, λ_k), report any gap; fix or weaken the statement accordingly.
- [x] (fresh clone of 2713aef: 3 test scripts, verify_theorems 61/61, ode_lsoda, make_figs and all three LaTeX builds succeed with no gitignored input needed) **N16 (2026-10-09). Fresh-clone reproducibility**: clone the pushed branch into a scratch directory, run the three test scripts, `scripts/ode_lsoda.py`,
  `scripts/verify_theorems.py`, `paper/figs/make_figs.py` and both LaTeX builds; record what fails because of gitignored inputs (npz) and fix by committing
  small derived CSVs or documenting the regeneration command.
- [x] (docs/paper_map.md, linked from README; gap-closing scripts kappa_limits.py, ode_fast_readout.py; verify_theorems output saved) **N17 (2026-10-09). Paper map in the root README**: a table from every figure/table/number family in the paper to the script and results file that
  produces it, and the experiment ↔ E-label mapping.

- [x] (7a48255 prereg; fe4013d results; P25a/b held, P25c/d failed as worded (softmax arm inconclusive); paper v0.18) **N18 (2026-10-09, from exps 8/11). Ingredient ablation (exp 12).** Add Adam and softmax attention one at a time to
  the single-feature pinned-readout model at `d=64`, `Γ=1`, `N∈{16,1024}`, `B∈{64,1024}`; which ingredient (if any) lets `B` rescue `N=16`? P25a–d.

- [x] (drift table; exp 13 cd61744 prereg, 38c8acd results; P26a held, P26b failed: softmax+Adam escape N=16 at B=64; paper v0.19) **N19 (2026-10-09, from exp 12). A softmax variant that learns.** The exp-12 softmax form (logits σ(w·x_i)σ(w·x_q), temperature 1) stays stuck even at
  N=1024. Before any ablation can use it: find by ODE/MC drift (`icl_additive/drift.py` style) a temperature or logit normalisation at which the
  population drift at m_0=1/8, N=1024 is positive; pre-register and re-run P25c/d with it. Only then can softmax be excluded or implicated.

- [x] (d984c95 prereg; d957200 results; P27 held: noise-driven; paper v0.20) **N20 (2026-10-09, from exp 13). B-dependence of the softmax+Adam escape at N=16 (exp 14, P27)**: noise-driven
  (escape fraction non-increasing in B) vs drift-driven (increasing, transformer-like).

- [x] (727a72b prereg; 7505ffb results; Adam slope −0.75, SGD flat; P28 failed as worded; paper v0.21) **N21 (2026-10-09, from exps 12-14). Adam makes B matter when the drift is positive? (exp 15, P28a/b)**: linear
  pinned Γ=1, N=1024, B∈{16,…,1024}: Adam slope of log T vs log B in [−0.75,−0.25] vs SGD flat. If it holds, exp 16 = transformer B-scan at N=16.

- [x] (89391b3 prereg; 9543b7c results; P29 failed on clause 1, B=256 censored at 30k) **N22 (2026-10-09, from exp 15). Transformer B-scan at N=16 (exp 16, P29)**: graded power-law B-dependence (Adam
  mechanism) vs threshold.

- [x] (eed5c94 prereg; a2413bb results; P30 failed: threshold-like; paper v0.22) **N23 (2026-10-09, from exp 16). Transformer B=256, N=16, 80k-step cap (exp 17, P30)**: graded law predicts T_e ≈ 36k.

- [ ] **N24 (2026-10-10). Refresh the four-page variant** (`paper/body_4p.tex` is based on v0.16; Sec. 5 has since gained Exps. 12–17) once the main text
  is frozen; cuts and moves only.
- [x] (docs/review_adversarial_5.md 6/10; response docs/review_adversarial_5_response.md; paper v0.23) **N25 (2026-10-10). Fifth adversarial review of v0.22** focused on Sec. 5 (is the ablation ladder Exps. 8–17 coherent, honestly framed and worth its
  length; what should move to the appendix), then respond.
