# Novelty protocol

Rule: **no claim enters the paper as "new" until the full text of every paper in its "could already
contain it" column has been read and the specific theorem/proof/section is cited as *not* containing it.**
Search snippets do not count. Until then the claim is tagged `UNVERIFIED` in all docs.

| # | claim | could already contain it | what to check | status |
|---|---|---|---|---|
| C1 | In-context pretraining squares the link: `L_A = L_B[g ↦ g²]`, effective info exponent `2k*` (fixed readout) | Oko–Song–Suzuki–Wu 2024 (Thm 1 + proof of pretraining step; their `d^{Θ(Q)}` and their *conjecture that a modified pretraining objective improves the complexity*); Kim–Suzuki 2024 (mean-field MLP + attention); Nishikawa et al. 2025 | whether the proof shows alignment gradient `∝ m^{2Q−1}`; whether they state the exponent is `2Q` and why | **PARTIALLY CONTAINED (full text read 2026-10-08).** Oko et al. Thm 1 (p.7): `T_1 = Ω̃(d^{Q+1} r^Q)`, **`N_1 T_1 = Ω̃(d^{2Q+1} r)`**, `η_1 ≍ m^{3/2} r d^{2Q−1/2}`, readout init `γ ≍ 1/(m^{3/2} r^{1/2} d^Q)`. The `2Q` is explicit in the total pretraining sample count and the step size, i.e. the squared-link signal `m_0^{2Q−1}` is *inside* their one-step analysis, though they describe it only as "makes use of the correlational information" and conjecture improvement via modified objectives (p.14). So C1 as a fact is theirs; what is ours is the interpretation (`L_A = L_B[g→g²]`), the online-SGD timescale `d^{2k*−1}` vs `d^{k*−1}` comparison, and the `k*=2` regime. **Frame C1 as mechanism, cite Thm 1.** |
| C2 | Finite context makes the origin a stable fixed point for `k* ≥ 2` (repulsion `∝ Γ²V'(m)/N`); threshold `d ≲ N/(4γ)` | Oko et al. 2024 (their pretraining prompt length `N_1 = Ω̃(d^{Θ(Q)})` — do they *need* long context for this reason?); Lu et al. 2024/2025 (linear attention asymptotics, non-monotone context dependence); 2610.01712 | any statement that short context prevents feature alignment during pretraining; any `V(m)`-type variance term | UNVERIFIED for the mechanism, but a **named default hypothesis now exists**: Oko et al. Remark 3 (p.7–8): "The multiplicative scaling between N_1 and T_1 ... suggests that one can tradeoff between the two quantities, that is, pretraining on more diverse tasks (larger T_1) can reduce the required pretraining context length N_1." Our C2/C3 predict the opposite beyond the one-step regime: the alignment-dependent variance `V(m)/N` is a *bias* in the drift, not averaging noise, so no number of tasks/steps removes the threshold in `N` (fixed readout) or the `d ~ N^{1/k*}` crossover (trained readout). Note their readout init `γ ∝ d^{−Q}` is tiny, which makes the `Γ²V/N` repulsion negligible in their regime — consistent with our condition `γ ≪ N m_0²`; the phenomenon lives exactly where `Γ` is O(1) or trained. |
| C3 | Trainable readout → Wiener shrinkage `Γ* = g²/(g²(1−1/N)+V/N)` → effective exponent `4k*` for `d ≫ N^{1/k*}`; crossover controlled by `N` | same as C2; Kunin et al. 2025 (saddle-to-saddle with readout); small-init two-layer literature for the in-weight analogue (`a* = g` ⇒ `2k*`) | whether the in-weight analogue is a known remark (likely yes, phrase accordingly); whether any ICL paper has the `N`-crossover | UNVERIFIED |
| C4 | Additive skill composition is free in linear-in-label ICL: emerges at `max(T_p,T_q)`; thresholded accuracy multiplies ⇒ Okawa et al.'s multiplicative law derived | Arora–Goyal 2023; Okawa et al. 2023; "When can transformers compositionally generalize in-context?" (2407.12275); "A spectral theory of compositional learning" (2609.33708, additive relational worlds, emergence times) | whether 2609.33708 already derives emergence times for additive compositions (different model class: relational embeddings, not ICL — but check the law's form) | UNVERIFIED — 2609.33708 is the closest and must be read. |
| C5 | Trichotomy for how real transformers escape C3: long context / nonzero task mean / label nonlinearity (softmax) | Nishikawa et al. 2025 (softmax label transformation beats CSQ — this is the "label nonlinearity" leg, at inference time rather than pretraining) | whether they discuss *pretraining* feature learning benefiting from the label nonlinearity | UNVERIFIED |
| C6 | Same scaling exponent `(β−1)/β` for ICL loss as in-weight loss, different `d`-time-unit; sharper per-skill steps | Ren et al. 2025 (their exponents, their second-layer protocol); Nam et al. 2024 | their theorem's dependence on `k*`; whether the student second layer is trained | UNVERIFIED (and low novelty on its own) |

## Positioning rule
The paper's claimed contribution is the *set* {C2, C3, C4} with C1 as the mechanism and C5 as the
bridge to real architectures. If C1 turns out to be explicit in Oko et al., we cite it as the mechanism
and keep the rest. If C2/C3 turn out to be in print somewhere, the paper is not worth writing in this
form — stop and rethink rather than reframe.

## Verification log
- 2026-10-08 (morning): arXiv/OpenReview/NeurIPS proceedings unreachable from this environment (network policy).
  Six targeted web searches on C1–C4 phrasing returned no paper stating C2/C3/C4; C1 undetermined.
- 2026-10-08 (afternoon): network opened. Downloaded 2504.19983, 2411.02544, 2610.01712, 2510.21020,
  2609.33708, 2604.22951, 2402.01258, 2404.17563 to `papers/` (git-ignored). Read Oko et al. Alg. 1, Thm 1,
  Remark 3, proof sketch §3.2.1, conclusion myself → C1 PARTIALLY CONTAINED (see table). Full-text checks of all
  eight papers delegated; results in `docs/fulltext_checks/`.
