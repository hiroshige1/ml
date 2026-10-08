# Novelty protocol

Rule: **no claim enters the paper as "new" until the full text of every paper in its "could already
contain it" column has been read and the specific theorem/proof/section is cited as *not* containing it.**
Search snippets do not count. Until then the claim is tagged `UNVERIFIED` in all docs.

| # | claim | could already contain it | what to check | status |
|---|---|---|---|---|
| C1 | In-context pretraining squares the link: `L_A = L_B[g ↦ g²]`, effective info exponent `2k*` (fixed readout) | Oko–Song–Suzuki–Wu 2024 (Thm 1 + proof of pretraining step; their `d^{Θ(Q)}` and their *conjecture that a modified pretraining objective improves the complexity*); Kim–Suzuki 2024 (mean-field MLP + attention); Nishikawa et al. 2025 | whether the proof shows alignment gradient `∝ m^{2Q−1}`; whether they state the exponent is `2Q` and why | UNVERIFIED — snippets suggest their bound comes from a one-gradient-step argument and they *conjecture* it can be improved, which is consistent with (but not proof of) `2Q` appearing there. Even if C1 is implicit in their proof, the *dynamical/emergence-time* statement and the in-weight comparison are ours. |
| C2 | Finite context makes the origin a stable fixed point for `k* ≥ 2` (repulsion `∝ Γ²V'(m)/N`); threshold `d ≲ N/(4γ)` | Oko et al. 2024 (their pretraining prompt length `N_1 = Ω̃(d^{Θ(Q)})` — do they *need* long context for this reason?); Lu et al. 2024/2025 (linear attention asymptotics, non-monotone context dependence); 2610.01712 | any statement that short context prevents feature alignment during pretraining; any `V(m)`-type variance term | UNVERIFIED — no snippet found mentioning it; the 2 searches with "repulsive/shrinkage/variance" terms returned nothing relevant. |
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
- 2026-10-08: arXiv/OpenReview/NeurIPS proceedings unreachable from this environment (network policy).
  Six targeted web searches on C1–C4 phrasing returned no paper stating C2/C3/C4; C1 undetermined.
