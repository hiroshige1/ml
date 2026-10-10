# Response to review 5 (6/10; Sec. 5 ≈ 4/10) — changes in Draft v0.23

| Point | Action |
|---|---|
| Cut Sec. 5 to ≤ 0.6 page, retitle, move Exps. 12–14 and the readout collapse to an appendix | Done: Sec. 5 is now three paragraphs (observation incl. the third-seed reversal; two null knob tests; what prompts per step do). Exps. 12–14, the collapse and Fig. 5(b) discussion are in Appendix "Ablation ladder" (app:ladder). Title: "an observation and what it excludes". |
| Drop "softmax + Adam, neither alone" | Dropped; replaced by the barrier-depth reading (rbf drift at m₀, N=16 is −6e-3 ± 6e-3, indistinguishable from zero) in the appendix, findings and memo. |
| State that P28b failed | Sec. 5 now says SGD is B-invariant "within 20% at N=1024 (Exp. 15, where the pre-registered 15% clause P28b was missed)". |
| Replace "threshold-like" | Replaced everywhere in the paper by "steeper still … with two seeds per cell a threshold cannot be told from a steep power law"; abstract clause replaced by the reviewer's wording ("which the population loss cannot produce; the mechanism is open"). |
| "every protocol", "80k in both seeds", third-seed reversal behind "≈3×" | "every frozen protocol that ran"; "77–80k steps in two seeds"; third seed (11.6k vs censored 17k) now stated. |
| Exps. 12–17 missing from intro (3) and Limitations | Added to both. |
| P27 appendix row vs README's strict reading | Row now states both readings. |
| Cite the square-root scaling rule for Adam | Malladi, Lyu, Panigrahi, Arora (NeurIPS 2022, arXiv:2205.10287) added after verifying the abstract (square-root scaling rule for RMSprop/Adam). |
| "Adam excluded" — withdraw or test | Withdrawn: the paper now says the mechanism is open and lists only the population loss, the final readout and the attention-output scale as excluded; Adam's role in the transformer is neither excluded nor established. |
| Local slopes of the toy are not a power law | Acknowledged implicitly by quoting the OLS slope only as a secant; the 36k extrapolation is labelled as "that slope, anchored at B=1024". |
