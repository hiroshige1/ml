# Referee check: Theorem A(iii)(c) and Remark rem:tiedflat

Scope: paper/theorems.tex, statement of (iii)(c), its proof, and Remark rem:tiedflat. I derived every step myself with s^2=1 and compared it with the text. I also wrote my own integrator of the tied flow, with V and rho* coded from the formulas rather than taken from scripts/verify_theorems.py. With it I checked 600 random cells: k=2..5, N from 1 to 10^4, m0 from 3e-3 (k=2) or 5e-2 (k>=3) up to 0.499, and rho0/rho*(m0) drawn from (1e-4, 1/4], with one third of the cells exactly at 1/4. **All 8 bounds held in every cell (0 failures).** No repository file was modified apart from this report.

## Step-by-step

**(1) Flow form and dρ/dm: correct.** dL/dm = ρ(ρD' − 2G'), so ṁ = (1−m²)·2G'(1−q) = 4k(1−m²)m^{2k−1}(1−q) with q = ρ/ρ*. Also ρ̇ = 8ρ(G−ρD). For q<1, d ln ρ/dm = 2(G−ρD)/(k(1−m²)m^{2k−1}(1−q)). This can have either sign, and the proof only uses the upper bound G−ρD ≤ G. The expression 1/ρ* = (1−1/N)/2 + V'/(4kNm^{2k−1}) is correct. The (q-bounds) follow from 4k² ≤ V'(m)/m ≤ 2V'(1/2) for m ≤ 1/2, which holds because V'(m)/m is a polynomial in m² with nonnegative coefficients.

**(2) q < 1/3 throughout: correct.** Bootstrap: while q<1/3, ρ ≤ (4/3)^{3/(2k)}ρ0 ≤ 1.2408ρ0 (step 4). ρ* is increasing in m, because 1/ρ* = (1−1/N)/2 + N^{-1}Σ_i 2iβ²_{2i} m^{2i−2k}/(4k) and every exponent is ≤ 0. Since m ≥ m0, q ≤ 1.2408/4 = 0.310 < 1/3, and continuity closes the argument. So q stays below 1/3 as m grows because ρ can grow by at most a factor of 1.241 while ρ* only increases. Numerically, max q never went above max(q0, 0.264).
*Slip (wording):* T is defined as a sup over t ≤ τ_tied before τ_tied is known to be finite. Fix: allow τ_tied = ∞ in the definition. Then ṁ ≥ 2k m0^{2k−1} on [0,T) forces τ_tied ≤ (1/2−m0)/(2k m0^{2k−1}). The text essentially says this already.

**(3) Lower bound, c1: correct.** d(1/ρ)/dm = 2(D−G/ρ)/(k(1−m²)m^{2k−1}(1−q)), with D − G/ρ = V/N − m^{2k}(1/ρ − 1 + 1/N) ≤ V/N because ρ ≤ 0.32ρ*(m0) < 1/(1−1/N). Using 1/(1−q) ≤ 3/2 and V/(1−m²) ≤ (4/3)V(1/2), d(1/ρ)/dm ≤ 4V(1/2)/(kNm^{2k−1}). Integrating gives 1/ρ ≤ 1/ρ0 + 2V(1/2)/(k(k−1)Nm0^{2k−2}). I get the same c1. Explicit values: c1 = 27/8 (k=2) and 49/16 (k=3), with V = 1+8m²+6m⁴ and 1+18m²+54m⁴+20m⁶.
*Slip (wording):* "(both increasing)" only applies to V/(1−m²). The bound 1/(1−q) ≤ 3/2 comes from q < 1/3, not from monotonicity.

**(4) Upper bound on ρ: correct.** d ln ρ/dm ≤ 2m/(k(1−m²)(1−q)) ≤ 3m/(k(1−m²)). Integrating gives ρ ≤ ρ0((1−m0²)/(1−m²))^{3/(2k)} ≤ (4/3)^{3/(2k)}ρ0. The parenthetical claim that this limit becomes ((4/3)(1−m0²))^{1/k} as ρ0→0 is also correct.

**(5) τ sandwich: correct.** τ = ∫dm/ṁ, which gives Xτ = (2k−2)m0^{2k−2}∫ m^{1−2k}/((1−m²)(1−q)) dm. The two integral identities, 1−(2m0)^{2k−2} and (1−(2m0)^{4k−4})/(2m0^{2k−2}), are correct.
- Lower bound: the (1−m²) factor is dropped (it is ≥ 1), and 1/(1−q) ≥ 1+q.
- Upper bound: for m ≤ 1/2, 1/(1−m²) ≤ 1+(4/3)m², and for q ≤ 1/3, 1/(1−q) ≤ 1+(3/2)q. The cross term is absorbed via 1+(4/3)m² ≤ 4/3, giving the 2q term.
- The O(ρ) term 2k(1−1/N)m^{2k−1}ρ in ρD' is exactly the (1−1/N)ρ/2 part of q. It contributes ≤ ρ_max on top and ≥ (1−(2m0)^{2k−2})(1−1/N)ρ_min/2 below. The noise part gives the ε-type terms.
- ℓ_2 = 2 ln(1/(2m0)) and ℓ_k ≤ (k−1)/(k−2) are correct.

Smallest numerical slacks: T1 1.0002, T2 1.013.

**(6) Constants: correct.** The lower side is ≥ 1−(2m0)^{2k−2} ≥ 1−4m0² ≥ 1−4λ_k m0². The upper side is ≤ 1 + (8/3)λ_k m0² + 1.241ρ0 + (4/3)^{3/(2k)} v̄_k ε. This uses (4/3)ℓ_2 ≤ (8/3)λ_2 and (4/3)(k−1)/(k−2) ≤ 8/3 for k ≥ 3. So C_k = max{4, (4/3)^{3/(2k)} v̄_k} suffices.
- v̄_2 = 11/8 and v̄_3 = 65/24, so C_2 = C_3 = 4 (the second entries are 1.71 and 3.13).
- C_k grows with k: C_4 = 7.11, C_5 = 18.30. The statement does not claim otherwise.
- The largest observed |Xτ−1|/bound was 0.71.

**(7) Hypotheses: correct.** ρ0 ≤ ρ*(m0)/4 is used twice: for q(0) ≤ 1/4 < 1/3, and for the closing estimate 1.241/4 < 1/3. The argument works for any fraction below (1/3)/(4/3)^{3/4} = 0.2686. Every bound uses m ≤ 1/2, and m0 < 1/2 is exactly what makes the interval [m0, 1/2] nonempty. Nothing else is needed (N=1 is also fine). One consequence is worth stating: by the left q-bound, the hypothesis already forces ε ≤ 1/4.

## Remark rem:tiedflat

The logic is correct. I reproduced its numbers independently:
- E5: ρ0/ρ*(m0) = 0.866, and ρ falls to 0.725ρ0.
- τ(16)/τ(256) = 1.859.
- 10 of the 19 tied cells satisfy the hypothesis.

Clarification: "N enters only through ε and 1−1/N" is true of the bounds. N also enters through the hypothesis ρ0 ≤ ρ*(m0)/4, since ρ* depends on N. "N-flat" means τ = (1/X)(1+O(ε+ρ0+λ_k m0²)), and 1/X does not depend on N. It does not mean τ is exactly independent of N.

## Verdict

**The theorem as stated is true, and the proof is complete.** It has two wording slips (in steps 2 and 3), and neither affects validity. No weakening is needed. The certified version is the one stated, with C_2 = C_3 = 4 and C_k = (4/3)^{3/(2k)}V'(1/2)/(2k²) when that value exceeds 4 (k ≥ 4). The hypothesis can be relaxed to ρ0 < 0.2686ρ*(m0) at no cost.
