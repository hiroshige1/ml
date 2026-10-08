# Theory notes: emergence of skills under in-context pretraining of an additive model

Working document. Everything here is a derivation to be checked numerically (`results/exp1`) and
against the literature (`docs/related_work.md`). Claims marked **[new?]** are the ones we have not
found in the literature *yet*; the literature check is incomplete because arXiv is unreachable from this
environment (only search snippets available).

## 0. Notation

- `x ~ N(0, I_d)`, teacher directions `v_1,…,v_P` orthonormal, student neurons `w_1,…,w_M ∈ S^{d-1}`.
- Normalised Hermite basis `σ_k = He_k/√k!`, `E[σ_j(⟨u,x⟩)σ_k(⟨w,x⟩)] = δ_jk ⟨u,w⟩^k` for unit `u,w`.
- Link `σ = Σ_k α_k σ_k`; `g(m) := E[σ(⟨v,x⟩)σ(⟨w,x⟩)] = Σ_k α_k² m^k`, `m = ⟨v,w⟩`.
  Information exponent `k* = min{k ≥ 1 : α_k ≠ 0}`; `g(m) ≍ α_{k*}² m^{k*}` as `m → 0`.
- `V(m) := E[σ(⟨v,x⟩)² σ(⟨w,x⟩)²] = Σ_j β_j² m^j` where `σ² = Σ_j β_j σ_j`.
  For `σ = σ_2`: `σ_2² = √6 σ_4 + 2√2 σ_2 + 1`, so `V = 1 + 8m² + 6m⁴`. For `σ = σ_1`: `V = 1 + 2m²`.

## 1. The in-context additive model

A *task* is `c ∈ R^P`; a prompt is `N` pairs `(x_i, y_i)` with `y = Σ_p c_p σ(⟨v_p, x⟩)` plus a query.
Task prior: `c_p` independent, mean zero, `E[c_p²] = s_p²`, `s_p² ∝ p^{-β}` (power-law skill frequency).
(Sparse variants: each prompt activates a random subset `S` of skills; mean-zero marginals keep all
formulas below with `s_p² → π_p s_p²`, `π_p` = activation probability.)

**Model A (linear-in-label ICL readout).**
```
A_j(prompt) = (1/N) Σ_i y_i σ(⟨w_j, x_i⟩)
ŷ_q         = Σ_j Γ_j σ(⟨w_j, x_q⟩) A_j
```
This is the "MLP features + linear attention" architecture of Oko et al. (2024) / the one-step-GD
estimator in feature space. **Model B** is ordinary regression `ŷ = Σ_j a_j σ(⟨w_j,x⟩)` on a fixed task
(Ren–Nichani–Wu–Lee 2025 setting).

## 2. Single skill, single neuron: ICL squares the link **[new? — check Oko et al. Thm 1 proof]**

`P = M = 1`, `E[c²] = s²`. Conditional on the task, `A = c g(m) + ξ/√N` with `E ξ = 0`,
`E ξ² = c² (V(m) − g(m)²)`. Since `x_q ⟂ context`,
```
L_A(m) = E (y_q − ŷ_q)²
       = s² [ 1 − 2Γ g(m)² + Γ² g(1) ( (1 − 1/N) g(m)² + V(m)/N ) ].          (2.1)
```
Compare `L_B(m) = 1 − 2a g(m) + a² g(1)`.

**Observation 2.1.** As `N → ∞`, `L_A` is `L_B` with `g ↦ g²`. The *effective information exponent of
in-context pretraining is `2k*`*: the drift of the alignment near `m = 0` is
`−∂_m L_A ≈ 4Γ s² k* α_{k*}⁴ m^{2k*−1}` versus `−∂_m L_B ≈ 2a k* α_{k*}² m^{k*−1}`.

Why: the label enters the prediction *twice*, once through the context statistic `A` and once through
the comparison with `y_q`; the mean-zero task prior kills the `E[c]²` cross terms that would otherwise
provide a degree-`k*` path. (A learnable *bias* `b_j` added to `A_j`, i.e. `ŷ_q = Σ Γ_j σ(w_j x_q)(A_j + b_j)`,
restores an in-weight path with exponent `k*`, but only for the *mean* task `E[c]`; with mean-zero tasks
there is nothing for it to learn. This gives a clean in-weight vs in-context dichotomy.)

**Consequences for online SGD timescales** (Ben Arous–Gheissari–Jagannath 2021 heuristics; `η` step size,
`m_0 ≍ d^{-1/2}`): effective exponent `κ`, drift `η C m^{κ−1}`, renormalisation drag `−η² d m/2`;
escape needs `η ≲ d^{−κ/2}` and then takes `T ≍ d^{(κ−2)/2}/η`, i.e. sample complexity `d^{κ−1}`
(`d log d` for `κ = 2`, `d` for `κ = 1`).

| teacher `k*` | Model B (in-weight) | Model A (in-context) |
|---|---|---|
| 1 | `d` | `d log d` |
| 2 | `d log d` | `d³` |
| 3 | `d²` | `d⁵` |

The `k* = 2` row is the experimental target (exp 1).

### 2.2 Finite context creates a repulsive force at the origin **[new?]**

From (2.1) with `N < ∞`, the term `Γ² g(1) V(m)/N` *increases* with `m` (aligned features have
correlated heavy tails with the teacher, so the context statistic is noisier when it is useful).
Near `m = 0`, for `σ = σ_2`, `Γ = γ`, `g(1) = 1`:
```
−∂_m L_A = s² [ (4γ − 2γ²(1−1/N)) m³ − γ² (16 m + 24 m³)/N ].
```
The origin is **stable** (no escape by drift) when `m² < 4γ/(N(1 − γ/2))·…≈ 4γ/N`... more precisely
when `m² < 16γ/(N(4 − 2γ))`, i.e. `m ≲ 2√(γ/N)` for small `γ`. With `γ = 1, N = 64` this is `m < 0.25`:
a single-neuron ICL model with unit readout and short context *cannot* learn a `k* = 2` skill from a random
initialisation in `d ≳ 16`. Escape requires either a long context (`N ≫ γ d`) or a small readout
(`γ ≪ N/d`). In a trained model `Γ` is itself learned and starts small, so the real dynamics are
two-timescale: `Γ` must stay below `N m²/4` while `m` grows. For `k* = 1` the same computation gives a
repulsion that is a *constant fraction* `γ/N` of the attraction — harmless.

This is a genuinely in-context phenomenon (there is no analogue in `L_B`), and it predicts a
**context-length threshold for feature emergence** that is distinct from the usual "context length
needed for in-context estimation" threshold. To be tested in exp 1b (vary `N` at fixed `d`, `γ`).

### 2.3 Trainable readout: shrinkage starves feature learning (population ODE, verified numerically)

Let `Γ` be trained. At fixed `m` the readout relaxes to the Wiener-filter value
```
Γ*(m) = g² / ( g(1) ( (1−1/N) g² + V/N ) )            (signal / (signal + context noise))
```
at rate `2 g(1)((1−1/N)g² + V/N) ≈ 2V/N` — which is **fast** compared with the alignment dynamics
(rate `O(m^{2k*−1})`) as soon as `N < ∞`. Plugging `Γ*` in:
```
L_A(m, Γ*) = s² [ 1 − g⁴ / ( g(1)((1−1/N) g² + V/N) ) ]
           ≈ s² [ 1 − N g⁴ / (g(1) V) ]        for  g² ≪ V/N  (i.e. m ≪ N^{−1/(2k*)})
           ≈ s² [ 1 − g² / g(1) ]              for  g² ≫ V/N.
```
So with a trainable readout the effective exponent is **`4k*`** while the context statistic is
noise-dominated, and `2k*` once it is signal-dominated. From `m_0 = d^{−1/2}` the initial regime is
noise-dominated iff `d ≫ N^{1/k*}` (for `k* = 2`: `d ≫ √N`). Interpretation: the optimal readout
*shrinks* an untrustworthy context statistic, and the shrunken readout multiplies the feature gradient,
so **feature learning is starved exactly when the context is too short to make the feature useful**.

Numerical check of the population gradient flow (`scripts/ode_two_timescale.py`, `k* = 2`, flow time
to `m = 0.5`, slope of `log T` vs `log d` over `d ∈ [16, 1024]`):

| protocol | N = ∞ | N = 1024 | N = 128 | prediction |
|---|---|---|---|---|
| fixed `Γ = 0.1` | 1.03 | 1.06 | 1.24 (stuck at d=1024) | `(2k*−2)/2 = 1`; stuck when `m_0² < 4Γ/N` |
| fixed `Γ = 1` | 1.03 | stuck for d ≥ 256 | stuck for d ≥ 32 | same |
| trainable `Γ`, `Γ_0 = 0.01` | 1.37 | 1.72 | 2.30 (→ 3.0 for d ≥ 256) | `(4k*−2)/2 = 3` for `d ≫ √N` |

At `N = 128` the last three points (`d = 256, 512, 1024`) have successive ratios 7.6, 8.0 ⇒ slope 3.0 ✓,
and the learned readout saturates at `Γ ≈ 0.23` independent of `d` (it tracks `Γ*`). The `N = ∞`
trainable case (1.37) is a genuinely joint slow dynamics (`Γ̇ = 2g²(1−Γ)` is *slow* when `N = ∞`),
not a clean power law.

**Caveat (also applies to Model B):** training the second layer `a` from small init in the in-weight
model does the same thing: `a* = g/g(1)` ⇒ `L_B(m, a*) = 1 − g²/g(1)`, exponent `2k*`. So the
*ratio* A/B of effective exponents is 2 under either protocol (fixed: `2k*/k*`; trainable: `4k*/2k*`),
but the absolute numbers depend on the protocol, and the finite-`N` crossover is specific to A.
Exp 1 uses fixed readouts for both (clean comparison); exp 1b varies `N` with trainable `Γ`.

### 2.4 The parameterisation of the readout decides whether shrinkage starves features (ODE-verified)

Ren et al. (2025) use the 2-homogeneous student `Σ_k ‖v_k‖² σ(v̄_k·x)`: the readout is *tied* to the feature
norm, and their Lemma B.1 makes the directional dynamics independent of the norm. Redo §2.3 with the tied
parameterisation `Γ = ρ = ‖w‖²` for Model A (`scripts/ode_tied.py`):
```
dm/dt  = (1−m²) [ 4 g g' − ρ ( 2(1−1/N) g g' + V'/N ) ]      ← attraction does NOT carry ρ
dρ/dt  = −4ρ [ −2g² + 2ρ((1−1/N) g² + V/N) ]                  ← ρ relaxes to Γ*(m) (tiny), multiplicatively
```
Result (`k* = 2`, flow time to `m = 0.5`, slope of `log T` vs `log d`): **1.03–1.05 for every `N ∈ {128, 1024, ∞}`
when `ρ_0 = 0.01`** — the `2k*` exponent, no starvation, no `N`-crossover; and **stuck for `d ≥ 32` when `ρ_0 = 1`,
`N = 128`** — the repulsion threshold of §2.2 is still there.

So: **C2 (finite-context repulsion, threshold `ρ ≲ N m_0²/4`) is robust to how the readout is parameterised;
C3 (shrinkage → `4k*`) is specific to a readout that can shrink independently of the feature norm** (a free
attention/output weight). In a transformer both exist: the first-layer norm is a tied (homogeneous) scale for
ReLU-type MLPs, and the value/output matrices are free readouts; with balanced initialisation gradient flow
conserves `‖w‖² − ‖Γ‖²`-type quantities, so which regime wins is a quantitative question for the real-model
experiment — and a cleaner, more surprising claim than "ICL is harder": *the exponent of in-context feature
emergence is selected by the parameterisation of the readout and by the context length, not by the target alone.*
This connects to Nam et al. 2025's "dynamical feedback principle" (which factor's magnitude gates which
factor's speed).

**Consequence in SGD sample-complexity terms** (heuristic, `κ_eff = 4k*`, free readout only): escape needs
`η ≲ d^{−κ/2}` and `n ≍ d^{κ−1} = d^{4k*−1}` — for `k* = 2` that is `d^7`: a linear-attention ICL model
with a trainable readout and context `N ≪ d²` essentially **cannot acquire an even (k*=2) skill from
scratch in high dimension**. Whatever real transformers do instead must be one of: long context
(`N ≳ d^{k*}`), a nonzero task mean (in-weight path, exponent `k*`), or a label nonlinearity in the
architecture (softmax attention; cf. Nishikawa et al. 2025's "beats CSQ via label transformations").
That trichotomy is a testable prediction for the small-transformer experiment.

## 3. Many skills: decoupling and the ICL scaling law

With `P` orthogonal teachers and `M ≥ P` neurons, `N → ∞`, the population loss is
```
L_A = Σ_p s_p² [ 1 − 2 Σ_j Γ_j g(m_pj)² ] + Σ_{j,j'} Γ_j Γ_j' g(⟨w_j,w_j'⟩) Σ_p s_p² g(m_pj) g(m_pj')  + O(1/N).
```
When the neuron–teacher assignment is one-to-one and the `w_j` are nearly orthogonal (which the Ren et al.
greedy-max-selection argument gives for high information exponent — here the *effective* exponent is
`2k* ≥ 2` even for `k* = 1`, which may make the decoupling easier to justify than in the in-weight case
**[check]**), the loss splits as `Σ_p s_p² ℓ(m_p)` with `ℓ(m) = 1 − 2Γ g(m)² + Γ² g(1) g(m)²`, and each
skill's alignment follows the single-skill ODE with drift coefficient `∝ Γ_p s_p²`.

Emergence time of skill `p` (time to reach `m = 1/2`):  `T_p ≍ d^{(2k*−2)/2} / (η Γ_p s_p²)` ⇒ with
`s_p² ∝ p^{−β}`: **`T_p ∝ p^{β}`**. Loss contribution of skill `p` before emergence is `s_p²`.
Number of skills learned by time `t`: `p(t) ∝ t^{1/β}`; residual loss
`L(t) ≈ Σ_{p > p(t)} s_p² ≍ p(t)^{1−β} ∝ t^{−(β−1)/β}`  (for `β > 1`).

So the **ICL loss obeys the same power law in training time as the in-weight loss**, with the same
exponent `(β−1)/β`; what changes is the *prefactor* / the `d`-dependence of the time unit
(`d^{k*−1}` → `d^{2k*−1}`), **and** the per-skill learning curves are sharper (higher effective exponent ⇒
longer plateau, faster escape). Prediction to test: the same `β` gives the same scaling exponent for
Model A and Model B, but Model A's individual skill curves are more step-like.

Finite-`N` corrections: the repulsion of §2.2 is per skill, `∝ Γ_p² s_p² m_p/N`, so **rare skills are
not disproportionately affected** (both forces scale with `s_p²`), but the *readout* `Γ_p` must still be
small; with a shared readout this couples skills. **[to think through]**

## 4. Composition

Train on single-skill prompts (`c = c_p e_p`), test on prompts with `c = c_p e_p + c_q e_q`.

In Model A (linear-in-label readout, `N → ∞`), the prediction is linear in the context labels, so
`ŷ_q(c_p e_p + c_q e_q) = ŷ_q(c_p e_p) + ŷ_q(c_q e_q)` exactly, and the compositional loss is
```
L_pair = s_p² ℓ(m_p) + s_q² ℓ(m_q)     (orthogonal teachers)
```
**Additive composition is free**: it emerges at `max(T_p, T_q)` with no compositional training data.
In terms of a thresholded accuracy, `P(correct) ≈ P_p · P_q` — a multiplicative emergence law of the
kind observed empirically by Okawa et al. (2023), here *derived* from the additive structure.
The finite-`N` cross term is `Γ² g(1) · 2 c_p c_q (E[σ_p σ_q σ_w²] − g_p g_q)/N`, zero in expectation over
mean-zero tasks.

Non-additive composition (`σ(⟨v_p,x⟩)σ(⟨v_q,x⟩)` or `h(⟨v_p,x⟩ + ⟨v_q,x⟩)`) needs features outside
`span{σ(⟨v_p,·⟩)}`: it is **not** free in Model A. Question for exp 3: does having learned `v_p, v_q`
lower the *leap* for the interaction term (staircase effect: the interaction `σ_1(v_p x)σ_1(v_q x)` has
information exponent 2 w.r.t. a fresh direction, but exponent 1 w.r.t. each of the known directions)?
A softmax transformer could in principle build the product in context (attention is multiplicative in
the labels after one nonlinearity), so this is where architectures should separate.

## 5. What would make this a real contribution (self-critique)

1. §2 (squaring) may already be implicit in the one-step-gradient analysis of Oko et al. 2024 (their
   pretraining complexity is `d^{Θ(Q)}` with `Q` the information exponent — the `Θ` hides whether it is `Q`
   or `2Q`). We must read their Theorem 1 and proof. If they have `2Q`, our §2 is a *dynamical* restatement,
   not a new fact; the contribution then rests on §2.2, §3, §4.
2. §2.2 (finite-context repulsion / readout–feature two-timescale) looks new and is testable; it is also the
   kind of statement that can be wrong because of an algebra error — exp 1's drift check is designed to
   catch that.
3. §3's "same exponent, different time unit" is a negative-ish result. It becomes interesting only with the
   sharper-steps prediction and with finite-`N` effects; otherwise it is "Ren et al. with g²".
4. §4 is the most interesting for a workshop audience (emergence of composition), and the cheapest to test.
   The multiplicative law is a concrete, falsifiable link to existing empirics.
5. All of this is for the toy architecture. One real-transformer experiment (2-layer softmax, same data)
   is mandatory to claim relevance, in particular for §2.2 and the non-additive composition.
