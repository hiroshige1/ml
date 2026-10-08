# Full-text check: softmax-ICL single-index (Nishikawa et al., ICML 2025) and Kim & Suzuki (ICML 2024)

Sources (downloaded 2026-10-08 to `scratchpad/papers_softmax/`):

- **[NSOWS25]** Nishikawa, Song, Oko, Wu, Suzuki, "Nonlinear transformers can perform inference-time feature learning", ICML 2025, PMLR 267. The paper is **not on arXiv**. I checked arXiv title search, the arXiv API with title and author queries, and the author pages; none found it. The OpenReview PDF for forum `xQTSvP57C3` returned 403. I read the camera-ready PMLR PDF instead, `mlresearch/v267/assets/nishikawa25a/nishikawa25a.pdf`, 32 pp. Main text is pp. 1-9, appendices A-G pp. 13-32. I read all of it; the proofs of Prop. 11 (pp. 17-20) I only skimmed.
- **[KS24]** Kim & Suzuki, "Transformers learn nonlinear features in context: nonconvex mean-field dynamics on the attention landscape", ICML 2024, arXiv:2402.01258v2 (2 Jun 2024), 35 pp. I read the main text pp. 1-9 in full, plus App. B (Lemma 2.2 proof), App. C.2-C.4 (landscape proof and the finite prompt/task length discussion), and the statements in App. D-E.

Page numbers are PDF pages. Quotes come from the pypdf extraction, and I re-typed the math where the extraction garbled it.

Claims under test are as defined in `oko2024.md`:

- C1: the ICL readout is linear in the label, so the pretraining gradient is a product of two correlations, giving effective IE 2Q ("link squared").
- C2: a finite pretraining context N gives an alignment-dependent noise term of order 1/N, which makes m=0 stable. This implies a context-length threshold for feature emergence.
- C3: the trained readout acts as a shrinkage factor, which moves the effective exponent to 4Q.
- C5: there are three escape routes: label nonlinearity (softmax), nonzero task mean, or long context.

## Verdicts (summary)

| Claim | [NSOWS25] Nishikawa et al. 2025 | [KS24] Kim & Suzuki 2024 |
|---|---|---|
| **C1** linear-in-label readout squares the link (eff. IE 2k*) | **PARTIALLY.** The product-of-two-correlations calculation is on the page (Lemmas 20-21). The leading pretraining signal is the *first-order-in-label* (Θ(ρ⁻¹)) part of the context factor times the query correlation, so it is ∝ c_ie². The paper never names a squared link or an exponent 2·ie. It states pretraining complexity as d^{Θ(ie)}, concretely T₁, N_pt = Ω̃(r²d^{ie+2}), because Γ is a merged d×d KQ matrix initialized at I_d/√d (see §1.6). | **PARTIALLY.** Eq. (2) is explicitly quadratic in the cross-covariance Σ_{µ°,µ} = E[f°h_µᵀ]. The fast-W profile (3) equals ½trΣ_{µ°µ°} − ½tr(Σ_{µ°µ}Σ_{µµ}⁻¹Σ_{µµ°}). There are no Hermite coefficients, information exponent, high-dimensional overlap or d-scaling analysis, and the squaring is never interpreted. |
| **C2** finite-N alignment-dependent 1/N noise makes m=0 stable; context-length threshold | **NOT CONTAINED.** Finite N enters only as concentration error Õ(√(d/N)), which is bounded in norm and not alignment-dependent. N_pt = Ω̃(r²d^{ie+2}) is a sufficient condition that keeps that error below the signal. The f·∇f term, where a 1/N variance term would live, is made negligible by the tiny readout init α (Lemma 21, eq. C.3). There is no landscape or fixed-point analysis. | **PARTIALLY (qualitative only).** App. C.4 explicitly warns that at finite n "the landscape Lₙ may no longer be benign" and gives a toy example where a saddle "becomes a local minimum for all ϵ>0". It then argues that such basins have W₂ radius ≤ n^{-1/4} and that the dynamics "mostly manages to avoid being trapped". It has no explicit 1/n alignment-dependent term, no threshold and no high-dimensional analysis. Its perturbation bound is √(log k/n), not 1/n. |
| **C3** trained readout shrinkage → 4k* | **NOT CONTAINED.** Training is staged. The readout a stays at a tiny init α1_m during the single Γ step, then is fit by ridge regression after Γ is frozen (Alg. 1). The paper never discusses shrinkage feeding back into the feature exponent. | **NOT CONTAINED.** The fast attention W sits at the exact least-squares optimum W_µ (Lemma 3.1, n=∞), which is a Σ⁻¹ whitening and not a noise-dependent shrinkage. Shrinkage is never discussed. |
| **C5** escape routes: softmax label nonlinearity / nonzero task mean / long context | **PARTIALLY, and adverse on the pretraining side.** The softmax's exp(y/ρ) is a label transformation that reduces the *inference-time* context length from r^{Θ(ie)} to r^{Θ(ge)}. But the *pretraining* complexity in their analysis stays d^{Θ(ie)}, because the query-side target is the raw label, i.e. a CSQ factor. Improving it to d^{Θ(ge)} is listed as future work requiring "going beyond one gradient step". The task mean is zero (β uniform on the sphere of S_r). Nonzero mean appears only as a pointer to Zhang, Wu, Bartlett (2024). Long context appears only as the sufficient condition N_pt, N_test large. | **NOT CONTAINED.** The model uses linear attention with no label nonlinearity. Only Σ_v = E[vvᵀ] enters the loss, so a task mean is absorbed into Σ_v and cannot act as an escape route; Σ_v is normalized to I_k. n→∞ is assumed for convenience, not as an escape route. Zhang et al. (2024) "mean signal vector" appears only in Related Work. |

Bottom line: neither paper states C1-C3 or C5 as we frame them. The closest overlaps, in decreasing order:

1. [NSOWS25] Lemmas 20-21. The product-of-correlations computation, the same math that appears in Oko et al. 2024.
2. [KS24] App. C.4. The qualitative remark that a finite prompt length can turn saddles into small local minima.
3. [KS24] eqs. (2)-(3). A loss that is quadratic in the feature-target cross-covariance.
4. [NSOWS25] Conclusion. The conjecture that pretraining could improve from ie to ge with multi-step training.

Three adversarial cautions for our paper, details in §3:

- (a) [NSOWS25]'s identity-initialized merged-KQ Γ attains d^{ie+2} rather than d^{2ie}. So C1's "2k*" must be stated in terms of a *vector/factored* feature overlap m, and is parametrization- and initialization-dependent.
- (b) [NSOWS25] shows that softmax label nonlinearity does **not** improve the pretraining exponent in a one-step analysis, because the query factor stays raw. C5's "softmax escape route" must say which factor is transformed and in which regime it helps.
- (c) [KS24] C.4 qualitatively anticipates the "finite n creates spurious minima near saddles" phenomenon, so C2 should cite it and stress what is new: the explicit alignment-dependent 1/N term, the threshold, and the high-dimensional regime where the init lies inside the basin.

---

## 1. Nishikawa, Song, Oko, Wu, Suzuki (ICML 2025)

### 1.1 Task distribution (Assumption 4, pp. 4-5; verbatim)
> "Let τ = O_d(1) be the noise level. A prompt (x₁, y₁, ..., x_N, y_N, x) is generated as: x₁, ..., x_N, x ∼ D_x = N(0, I_d), y_i = f*(x_i) + ζ_i, ζ_i ∼ Unif({−τ, τ}), f*(x_i) = σ*(⟨β, x_i⟩)."
> "σ* is a polynomial that remains fixed across tasks. We assume the normalization conditions E_{z∼N(0,1)}[σ*(z)] = 0 and E_{z∼N(0,1)}[σ*²(z)] = 1."
> "S_r denotes an r-dimensional linear subspace of R^d for r ≤ d. For each prompt, β is drawn uniformly from the unit sphere Supp(β) := {β | β ∈ S_r, ‖β‖ = 1} in S_r."

- **The task mean is zero**: β is uniform on the sphere of S_r, so E[β] = 0. The pretraining gradient is ∝ E_β[ββᵀ] (Lemma 21).
- The link is fixed across tasks and mean-zero. Remark 7 (p. 6): "For simplicity of analysis, we assume that the link function σ* is fixed across tasks."

### 1.2 Architecture (Sec. 3.2, p. 5; App. A.2, p. 13)
The embedding is E = [x₁ ⋯ x_N x; y₁ ⋯ y_N 1] ∈ R^{(d+1)×(N+1)}. Note the **1 in the label slot of the query**. The model is one softmax self-attention layer followed by a ReLU MLP.
> "Attn(E) = W^V E · Softmax(Mask(ρ⁻¹ · (W^K E)ᵀ W^Q E)), where ρ > 0 is the temperature"

The merged and zeroed parameterization is W^{KQ} = [[Γ, 0],[0, 1]] and W^{FV} = [O_{m×d}, v]. This gives eq. (3.2):

  f_TF(X, y, x; Γ, v, b, a) = Σ_{j=1}^m a_j σ( v_j · [Σ_i y_i e^{y_i/ρ} e^{x_iᵀΓx/ρ}] / [Σ_i e^{y_i/ρ} e^{x_iᵀΓx/ρ}] + b_j ).

- **Where the label transformation comes from.** The fixed (d+1,d+1) entry 1 of W^{KQ} multiplies the key's y_i by the query's 1, so y_i/ρ is added to every attention score. This is *not* trained. The value is linear in y_i: W^{FV}'s last column v multiplies y_i. So each context token contributes y_i e^{y_i/ρ} (numerator) and e^{y_i/ρ} (denominator).
- The masking of the last row (Remark 6) prevents self-attention to the query.
- The temperature is ρ = Θ(log^{Cρ} d), large (Prop. 11, p. 14). So e^{y/ρ} ≈ 1 + y/ρ + y²/(2ρ²) + ⋯, and the j-th label power σ*^j is suppressed only by polylog factors ρ^{−j}.

### 1.3 Training procedure (Alg. 1, p. 6; Sec. 3.2, p. 5; Prop. 22, p. 25)
> "Initialize Γ^{(0)} ∼ I_d/√d, v^{(0)} ∼ Unif({±1}^m), b^{(0)} = 0_m, a^{(0)} = α1_m"
> "Stage I: Gradient descent on Attention Matrix. Γ* ← Γ^{(0)} − η₁∇_Γ(L₁(Γ^{(0)}, v^{(0)}, b^{(0)}, a^{(0)}) + λ₁/2‖Γ‖²_F)"
> "Stage II: Optimization of MLP Layer. Initialize b*_j ∼ Unif([−1, 1]), v* = v^{(0)}. a* ← argmin_a L₂(Γ*, v*, b*, a) + λ₂/2‖a‖²."

p. 5: "In Stage I, we perform a single gradient descent step on the attention matrix Γ using the ℓ2-regularized squared loss ... computed over T₁ independent pretraining tasks." Also: "we randomize the weight v and bias b, and perform ridge regression on the output weights a. Note that the only role of the MLP layer is to fit the polynomial link function σ*."

- **Trained parameters.** Γ, a d×d merged KQ matrix acting on raw x, gets **one GD step**. Only a is trained afterwards (ridge, convex). v stays at its init and b is re-randomized. W^V/W^F are not separately trained, and the label row/column of W^{KQ} is fixed.
- **Hyperparameters (Prop. 22, p. 25):** "αm = O(r⁻¹d^{−(ie(σ*)+1)/2} log^{−Cα} d), N_pt, T₁ = Ω̃(r²d^{ie(σ*)+2}), η₁ = Θ(α⁻¹m⁻¹d^{(ie(σ*)−1)/2} log^{−Cη1} d) r^{1/2} and λ₁ = η₁⁻¹". With λ₁ = 1/η₁, the weight decay cancels Γ^{(0)}, so Γ* = −η₁∇L₁. This is the Ba et al./Damian et al. trick. Result: "Γ* = 1/(r^{1/2} log^{Cκ} d) (r E_β[ββᵀ] + N) ... ‖N‖_F = O_d(1/√d)".

### 1.4 Main theorem (Theorem 1, p. 6; verbatim, cleaned)
> "Let f_TF(X, y, x; Γ*, v*, b*, a*) be a transformer pretrained via Algorithm 1 with MLP width m = Ω̃(r^{2ge(σ*)}) and initialization scale α = O(m⁻¹r⁻¹d^{−(ie(σ*)+1)/2} log^{−Cα} d) for constant Cα. ...
> 1. (Optimization) Suppose that the pretraining task size and the pretraining context length satisfy T₁, N_pt = Ω̃(r²d^{ie(σ*)+2}) and T₂ = Ω̃(r^{3ge(σ*)/2}) (i.e., T_pt = Ω̃(r²d^{ie(σ*)+2} ∨ r^{3ge(σ*)/2})). Then, Algorithm 1 yields parameters ... such that the empirical loss ... = o_d(1). Moreover, we have ‖a*‖ = Õ(r^{3ge(σ*)/4} m^{−1/2}).
> 2. (Inference-time sample complexity) At inference time, if the (in-context) test prompt length satisfies N_test = Ω̃(r^{3ge(σ*)/2}), then the ICL error (2.1) R^{ICL}_{N_test}(Γ*, v*, b*, a*) = o_d(1)."

- **d-dependence:** both the number of tasks T₁ **and** the per-prompt context length N_pt must be at least r²d^{ie+2}, so the total token count is T₁·N_pt ≳ r⁴d^{2ie+4}. The paper never comments on this product.
- The proof of Part 2 (p. 30) also needs N_pt ≥ Ω̃(r^{3ge/2}): "From Proposition 11, if N_pt, N_test = Ω̃(r^{3ge(σ*)/2}), then |g(X_{1:N_test}, ...) − g(X_{1:N_pt}, ...)| ≤ o_d(r^{−3ge(σ*)/4} log^{−C} d)".

### 1.5 Which exponent governs what (verbatim)
- **(a) Pretraining: the information exponent.** p. 6, (iii): "our result provides an optimization guarantee for pretraining along with the pretraining task/sample complexity of d^{Θ(ie(σ*))} for T_pt and N_pt". Table 4.1 (p. 7) lists "This work — Pretraining: d^{Θ(ie(σ*))}; Inference: r^{Θ(ge(σ*))}". Sec. 5.1 (p. 7): "we can approximate the population (expected) gradient by the population correlational gradient E[y∇_Γ f_TF]. We show that the magnitude of this correlation is O(d^{−(ie(σ*)−1)/2}). A standard matrix concentration argument then yields the required pretraining task and sample complexity of d^{Θ(ie(σ*))}".
- **(b) Inference context length: the generative exponent.** p. 3: "the nonlinearity in the attention reduces the information exponent of the link function σ*, which is shown through a careful analysis of the nonlinear correlation computed by the attention scores. Mechanism (i) enables inference-time feature learning, whereas mechanism (ii) leads to an information exponent-free sample complexity." Sec. 5 (p. 7): "the factor e^{y_i/ρ} reduces the information exponent of σ* to its generative exponent ge(σ*) through the exponential transformation e^{·/ρ}". p. 8: "Crucially, since the correlational signal strength depends only on the generative exponent ge(σ*), the required context length N does not scale with ie(σ*)."
- **Lemma 9 (p. 15):** "For p ≥ 1, if e_p(σ*) < ∞, then H(exp(σ̄*), p) = Θ(1/(log d)^{Cρ e_p(σ*)}) and H(σ̄* exp(σ̄*), p) = Θ(1/(log d)^{Cρ e_p(σ*)}) ... This immediately implies that the information exponent of exp(σ̄*) and σ̄* exp(σ̄*) is equal to ge(σ*)." Here e_p(σ*) is the minimal j with H(σ*^j, p) ≠ 0 (Def. 14). So the softmax uses the monomial σ*^j at a cost of ρ^{−j}, which is polylog.
- **Prop. 8/11:** g ≈ P₁ + P₂(⟨Gx, β⟩/√r)^{ge(σ*)}. For even σ* (ge = 2), the attention output encodes ⟨β, x⟩², not ⟨β, x⟩. The MLP then fits ς* with σ* = ς*(z²) (Lemma 23).

### 1.6 Pretraining gradient mechanism: the relation to C1 "squared link" (Lemmas 20-21, pp. 21-25)
Lemma 20: "∇_Γ f_TF(...)|_{Γ=I_d/√d} = αL_m(P₀ + P₁z + ⋯ + P_{ie(σ*)−1} z^{ie(σ*)−1} + ⋯)βxᵀ + αL_m n xᵀ where ... P_{ie(σ*)−1} = Θ(ρ⁻¹) and ‖n‖ ≤ O(r⁻¹d^{−(ie(σ*)+1)/2} log^{−Cn} d)", with z := ⟨β, x⟩/(ρ√d).

Proof, p. 23: "to calculate the expected correlational gradient E[y∇_Γ f_TF], it suffices to calculate E[y · z^k βxᵀ] = E[σ*(⟨β,x⟩)(βᵀx/ρ√d)^k βxᵀ] = E[σ*'(⟨β,x⟩)(βᵀx/ρ√d)^k ββᵀ] + ⋯ Now, if k < ie(σ*) − 1, the first term ... is zero ... If k = ie(σ*) − 1, then the expectation is nonzero and proportional to E[ββᵀ]".

p. 23: "From Lemma 9, A_i, B_i = Θ(1/ρ^{e_i(σ*)}) ... we can see that e_{ie(σ*)} = 1 and then A_{ie(σ*)}, B_{ie(σ*)} = O(ρ⁻¹) holds."

Lemma 21 (p. 24): "∇_Γ (1/2T₁)Σ_t(f_TF − y^t)²|_{Γ=Γ^{(0)}} = Θ(αm d^{−(ie(σ*)−1)/2} ρ^{−ie(σ*)}) E_β[ββᵀ] + O(αm r⁻¹d^{−ie(σ*)/2} log^{−Cn} d) + Õ(α²m²√d)".

**Reading.** The leading coefficient is a product of two factors:
1. A context-side Hermite coefficient at degree ie of the transformed label. Its leading part is the **linear** term σ*/ρ (e_ie = 1, Θ(ρ⁻¹)).
2. A query-side correlation E[σ*'(⟨β,x⟩)⟨β,x⟩^{ie−1}] ∝ c_ie(σ*) of the **raw** query label.

So the signal is ∝ c_ie(σ*)²/ρ. This is the "product of two correlations / link squared" structure, as in Oko et al. 2024, and it is visible on the page. The paper never names it.

**Why the d-exponent is ie, not 2ie.** Γ is a merged d×d bilinear form initialized at I_d/√d. The relevant overlap is the quadratic form M = βᵀΓβ = d^{−1/2}, not a vector overlap m = ⟨w, β⟩ ≈ d^{−1/2}. The gradient scales as M^{ie−1} = d^{−(ie−1)/2}. In our C1 language, a factored Γ = wwᵀ would have M = m², so the doubled exponent in m is the single exponent in M. The identity init gives M₀ = d^{−1/2}, the same as m₀ = d^{−1/4}, which halves the squared-link penalty. The paper does not discuss any of this. **It is a caution for C1:** the "2k*" statement depends on the parametrization and initialization.

**The softmax does not help the pretraining step.** The query-side factor is the raw target y (squared loss on y), so it vanishes for degrees < ie whatever the context-side label transformation. In the one-step analysis, the softmax therefore leaves the pretraining exponent at ie.

### 1.7 Finite context N, f·∇f term, alignment-dependent noise (C2)
- Finite-N effects appear only as concentration errors. Lemma 18 (p. 16): "N⁻¹Σ_i exp(y_i/ρ)exp(x_iᵀGx/(ρκ√r)) − E_{x₁,ζ₁}[⋯] = Õ(N^{−1/2})". Lemma 19/20 vector versions: "+ Õ(d^{1/2}N^{−1/2})". Proof of Lemma 20 (p. 23): "Set N = Ω̃(r²d^{ie(σ*)+2}) to ensure that all the noise terms in π₁, π₂, ξ₁, ξ₂ are of O(r⁻¹d^{−(ie(σ*)+1)/2} log^{−C} d)".
- The f·∇f term is killed by small α. Lemma 21 proof (p. 24): "|f_TF(X, y, x; Γ, v^{(0)}, b^{(0)}, a^{(0)})| ≤ ... = Õ(αm) ... ‖(1/T₁)Σ_t f_t G_t‖_F = Õ(α²m²√d) (C.3)". The prediction-squared term, where a 1/N variance term ∝ alignment would sit, is therefore negligible by construction.
- There is no alignment-dependent noise, no 1/N bias term, no fixed-point or stability analysis of Γ ≈ 0 or m = 0, and no context-length threshold beyond the sufficient conditions.
- Searches for "shrink", "align", "1/N" bias and "squared link" find nothing relevant. "align" appears only as "aligns with the CSQ lower bound" (p. 2) and "aligns with the findings of Oko et al." (p. 3).

### 1.8 Readout and shrinkage (C3)
- The readout a stays fixed at α1_m with a tiny α during the Γ step. Ridge λ₂ on a is applied only after Γ* is frozen. Sec. 5.3 (p. 8): "thanks to the equivalence between the ℓ2-regularized convex optimization and its norm-constrained counterpart, we can show that there exists a ridge parameter λ₂ such that ℓ(a*) ≤ ℓ(a′) and ‖a*‖ ≤ ‖a′‖". The ridge serves generalization through a Rademacher norm bound, not feature learning.
- Remark 7 (p. 6) mentions test-time training of a: "An alternative could be to fit the MLP parameter a using each test sample: such test-time training constitutes a relatively lightweight optimization". This is not related to shrinkage.

### 1.9 Conclusion and limitations (Sec. 7, p. 9; verbatim)
> "We studied in-context learning of single-index polynomials and demonstrated that pretrained transformers achieve inference-time sample complexity that surpasses both kernel methods and CSQ lower bounds. Our analysis highlights that the Softmax attention can extract feature information by leveraging nonlinear transformations applied to labels.
> We outline several future directions. First, the pretraining procedure and its current sample complexity of d^{Θ(ie(σ*))} may potentially be improved to d^{Θ(ge(σ*))} by going beyond one gradient step (Dandi et al., 2024; Arnaboldi et al., 2024; Lee et al., 2024). Second, extending our analysis to broader function classes – such as non-polynomial σ*, multi-index targets, or anisotropic covariates (Ghorbani et al., 2020; Refinetti et al., 2021; Mousavi-Hosseini et al., 2023b) – is another interesting direction. Finally, while our study demonstrates both "memorization" of the r-dimensional support of the target function class and inference-time feature learning, it remains an open question whether a tradeoff or certain interaction exists between these two mechanisms."

### 1.10 Other relevant remarks
- Related work (p. 3): "(Zhang et al., 2024) showed that linear transformer can adapt to the prior mean of the coefficient vector". This is the only mention of a nonzero task mean. It refers to Zhang, Wu, Bartlett, arXiv:2402.14951. **We should check that paper for C5's "nonzero task mean" route.**
- Experiments (Sec. 6, App. G): a 6-layer GPT-2 with curriculum and y = He₃. The empirical N_test ∝ d^{1.1-1.16}, against the CSQ bound d^{1.5} and the kernel bound d³. Pretraining scaling was not measured.

---

## 2. Kim & Suzuki (ICML 2024), arXiv:2402.01258v2

### 2.1 Model and task (Sec. 2.1-2.2, p. 3; verbatim, cleaned)
- The MLP feature is h_µ(x) = ∫ aσ(wᵀx)µ(dθ) ∈ R^k (mean-field). Its output feeds linear self-attention: "We further absorb v into W and fix v = 1 in order to focus on the more complex dynamics of the MLP layer." The prediction is
  ŷ_qr = (1/n)Σ_{i=1}^n y_i h_µ(x_i)ᵀ W h_µ(x_qr) (linear in the labels; the query label slot is 0).
- The task is "ICL of linear regression tasks over a common nonlinear transformation or feature map f° ∈ C(X, R^k), that is T = {vᵀf° | v ∈ R^k} with covariance Σ_v = E_v[vvᵀ]. By replacing f° by Σ_v^{1/2}f°, we may assume Σ_v = I_k."
- **Task mean.** It is not stated as zero. But the loss is quadratic in v and depends on v only through E[vvᵀ], so a mean μ is absorbed into Σ_v = Cov + μμᵀ. The model has no task-independent (in-weight) prediction path: ŷ is linear in the y_i, and y_qr enters as 0.
- **Context length.** "We also take the n → ∞ (infinite prompt length) limit to disregard sampling error and let ŷ_qr = E_x[f(x)h_µ(x)ᵀ]W h_µ(x_qr) for any task f ∈ T; but see Appendix C.4 for a discussion on how to incorporate finite samples."
- Eq. (2): L_TF(µ, W) = ½E_{x_qr}[‖f°(x_qr) − E_x[f°(x)h_µ(x)ᵀ]W h_µ(x_qr)‖²].

### 2.2 Two-timescale structure (Sec. 3.1, p. 5; verbatim)
> "we first notice that minimizing L_TF over W is a least-squares regression problem. In particular, L_TF is convex with respect to W (strongly convex unless Σ_{µ°,µ} or Σ_{µ,µ} are singular) and thus is optimized potentially much more quickly."
> "Lemma 3.1. For any fixed µ ∈ P₂⁺(Θ) and any initialization W₀ ∈ R^{k×k}, the flow d/dt W_t = −∇_W L_TF(µ, W_t) converges linearly to some W_µ ∈ argmin_W L_TF(µ, W) which satisfies Σ_{µ°,µ}W_µ = Σ_{µ°,µ}Σ_{µ,µ}⁻¹."
> "Therefore it is reasonable to suppose that W is updated sufficiently quickly and has already converged to W_µ for each µ – formally by modeling as two-timescale dynamics (Berglund & Gentz, 2006) – leading us to study the objective L(µ) := inf_W L_TF(µ, W) = ½E_x[‖ζ_{µ°,µ}(x)‖²] (3) where we denote ζ_{µ°,µ}(x) := h_µ°(x) − Σ_{µ°,µ}Σ_{µ,µ}⁻¹h_µ(x)."

- **The fast layer is the attention matrix W; the slow layer is the MLP measure µ.** p. 7 gives an equivalent three-layer network "taking the two-timescale limit where the last layer updates infinitely quickly". p. 2 cites Bietti et al. (2023) for k-index models "after factoring out the link function via a similar two-timescale limit".
- **Squared-link structure.** Expanding (2) gives ½trΣ_{µ°µ°} − tr(Σ_{µ°µ}WΣ_{µµ°}) + ½tr(WᵀΣ_{µµ°}Σ_{µ°µ}WΣ_{µµ}), as in the expansion in the Lemma 2.2 proof (p. 14). This is quadratic in the feature-target cross-covariance Σ_{µ°,µ} = E[f°h_µᵀ] for *any* W, including fixed W. With a single-index teacher/student neuron, Σ_{µ°,µ} ∝ Σ_j c_j(σ°)c_j(σ)m^j, so the loss is ∝ (Σ_j c_j c'_j m^j)². This is exactly C1's square, but **the paper never writes it in Hermite/overlap form or draws any exponent conclusion**. The analysis uses no Gaussian single-index structure, and the d-dependence is only polynomial in the rates: "The rate is polynomial in the number of features k but only linear in d" (p. 8).
- **Adversarial note.** The benign-landscape argument moves along the homotopy (1−s)µ + sR♯µ° (Thm 3.3), i.e., it puts mass directly on the true features. The quantitative rates need Assumption 3, ‖dµ°/dπ‖_∞ ≤ R₄, with a birth-death process. That constant hides the high-dimensional search (information-exponent) difficulty entirely. Non-global critical points are *strict* saddles with L ≥ r̲/2 and d²/ds² L ≤ −(4/(kR₁²))L(µ)². That is Θ(1) curvature in their (non-high-dimensional) mean-field geometry, in contrast with the flat m=0 points of the d→∞ overlap picture.

### 2.3 Finite context length (App. C.4, pp. 22-23; verbatim). This is the closest overlap with C2.
> "we consider the n-sample in-context prediction loss L^n_TF(µ, W) = ½E_{x₁,⋯,x_n,x_qr}[‖f°(x_qr) − (1/n)Σ_i f°(x_i)h_µ(x_i)ᵀW h_µ(x_qr)‖²]. By treating the sampling process as stochastic noise, we can bound the perturbation magnitude using concentration ... ≲ √(log k / n) for large enough n ... |L^n(µ) − L(µ)| ≲ √(log k / n) (10)"
> "The main issue is that even with this guarantee, the landscape L^n may no longer be benign so that our results on mean-field dynamics in the subsequent sections do not directly apply. For an illustration, consider the ϵ-perturbed quadratic function −x² − ϵexp(−(x/ϵ)²) on R with a bump at the origin; the point x = 0 becomes a local minimum for all ϵ > 0. Nevertheless, it is easily shown that such attraction basins must still be small with radius at most O(√ϵ). Similarly, combining the strict curvature bound Theorem 3.3(ii) and (10) with regularity estimates for L^n as in Lemma E.4 yields an n^{−1/4} upper bound for the W₂ radius of potential local attraction basins. Hence it is plausible that the dynamics still mostly manages to avoid being trapped in local minima."
> "Finally, we note that complexity bounds for finite task and prompt lengths have been established for the single LSA layer model in Wu et al. (2024). We could also consider noisy data y_i = f(x_i) + ε_i, ε_i ∼ N(0, σ²_ε), which only leads to L_TF being shifted by a constant ½σ²_ε."

**Assessment.** This qualitatively raises "finite n can make a saddle a local minimum". It does **not**:
- compute the 1/n variance term. The n-sample loss contains (1/n)E[‖f°(x)h(x)ᵀWh(x_qr)‖²]-type terms, which grow with the feature magnitude and alignment. The paper never computes them.
- identify the term's sign or dependence on alignment, or a threshold in n.
- connect the basin radius to the high-dimensional init overlap.

It concludes the opposite of C2: that trapping is implausible. A W₂ radius of n^{−1/4} would contain an m ≈ d^{−1/2} initialization unless n ≳ d², but the paper does not say this. Note also that their noise model is said to only shift L by a constant, in the n=∞ setting.

### 2.4 Conclusion (Sec. 7, p. 9; verbatim)
> "In this paper, we explored the training dynamics of a Transformer with one MLP and one attention layer, enabling in-context feature learning of regression tasks on a rich class of representations. We showed that the loss landscape becomes benign in the two-timescale and mean-field limit and developed instability and improvement guarantees for the Wasserstein gradient flow. To our knowledge, this represents both the first work to theoretically study how features are learned in context, and the first analysis of nonconvex mean-field dynamics for strict saddle objectives. We hope our insights may be extended to more complex in-context learning behavior in deeper MLP-attention models."

### 2.5 Other
- Concurrent works (p. 2): "Zhang et al. (2024) consider a linear attention layer followed by a linear layer which learns to encode a mean signal vector, but their model does not include any nonlinearities." This is the same Zhang-Wu-Bartlett 2024 pointer as in [NSOWS25], and is relevant to the C5 nonzero-mean route.
- Experiments (Sec. 6): d=20, k=5, 500 sigmoid neurons, 10K prompts × 1K token pairs. The attention W is trained jointly, and "the attention and static Transformers exhibit similar dynamics ... justifying the two-timescale approach". No context-length sweep.

---

## 3. Per-claim notes for our write-up

- **C1.** Cite [NSOWS25] Lemmas 20-21 and [KS24] eq. (2)/(3) as containing the product/square structure implicitly. Our novelty is the explicit identification as a squared link with effective IE 2k* in the vector overlap, and its consequences. We must address [NSOWS25]'s d^{ie+2} rate: it comes from a merged KQ matrix with *identity* init, where βᵀΓ₀β = d^{−1/2}, the same as m₀² at m₀ = d^{−1/4}. So "2k*" is a statement about factored/learned-feature parametrizations from random init. An identity-like KQ init acts as a warm start. Also note that [NSOWS25] needs T₁·N_pt ≳ d^{2ie+4} tokens, though they never interpret this.
- **C2.** Not in [NSOWS25]: the f∇f term is suppressed by tiny α, and N enters only through concentration. [KS24] C.4 is the qualitative precursor and must be cited. Our explicit alignment-dependent 1/N term, the stability of m=0 and the context-length threshold are not there. [KS24] in fact argues trapping is unlikely.
- **C3.** Neither paper contains it. [NSOWS25] stages readout training after the single feature step. [KS24]'s fast W is the n=∞ least-squares optimum with no noise-dependent shrinkage.
- **C5.**
  - Softmax: [NSOWS25] contains the label-nonlinearity-via-softmax mechanism, but **only for inference-time context length** (ie → ge). Its pretraining stays d^{Θ(ie)}, and the ge improvement is an open "future direction" requiring multi-step training. Our C5 must make clear that the softmax helps pretraining only if the transformed label enters a factor that is not paired with the raw query label. Otherwise [NSOWS25]'s one-step analysis is a counterexample to an unqualified "softmax escapes".
  - Nonzero task mean: absent from both. Both point to Zhang-Wu-Bartlett 2024 (arXiv:2402.14951), which must be checked next.
  - Long context: in both papers it appears only as a sufficient condition ([NSOWS25]) or an idealization ([KS24]), not as an escape route.
