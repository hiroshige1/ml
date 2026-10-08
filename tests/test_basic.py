"""Basic tests: python -m pytest tests/ -q   (or: python tests/test_basic.py)"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from icl_additive.data import sample_iw, sample_iw_proj, sample_prompts, sample_prompts_proj  # noqa: E402
from icl_additive.drift import V_fn, g_fn, loss_formula  # noqa: E402
from icl_additive.hermite import dsigma, hermite_norm, sigma  # noqa: E402
from icl_additive.models import ModelA, ModelATied, ModelB  # noqa: E402
from icl_additive.train import ctx_stats_fast, init_fixed, init_sphere, train  # noqa: E402

KS = [1, 2, 3, "relu"]


def test_hermite_orthonormal_gauss_hermite():
    x, wts = np.polynomial.hermite_e.hermegauss(60)
    wts = wts / math.sqrt(2 * math.pi)
    for j in (1, 2, 3):
        for k in (1, 2, 3):
            val = np.sum(wts * sigma(j, x) * sigma(k, x))
            assert abs(val - (1.0 if j == k else 0.0)) < 1e-12
        assert np.allclose(sigma(j, x), hermite_norm(j, x))
    assert abs(np.sum(wts * sigma("relu", x))) < 5e-3  # centered (relu is non-smooth: quadrature is only ~3e-3 accurate)


def test_hermite_orthonormal_mc():
    z = np.random.default_rng(0).standard_normal(2_000_000)
    for j in (1, 2, 3):
        for k in (1, 2, 3):
            val = np.mean(sigma(j, z) * sigma(k, z))
            assert abs(val - (1.0 if j == k else 0.0)) < 0.02, (j, k, val)
    assert abs(np.mean(sigma("relu", z))) < 0.002


def test_dsigma_finite_difference():
    z = np.linspace(-2, 2, 41) + 0.013
    h = 1e-6
    for k in KS:
        fd = (sigma(k, z + h) - sigma(k, z - h)) / (2 * h)
        assert np.allclose(fd, dsigma(k, z), atol=1e-6)


def test_shapes():
    rng = np.random.default_rng(0)
    d, N, B = 8, 16, 5
    v, w = init_sphere(d, rng)
    assert w @ v > 0 and abs(np.linalg.norm(w) - 1) < 1e-12
    x_ctx, y_ctx, x_q, y_q = sample_prompts(d, N, B, v, 2, 1.0, rng)
    assert x_ctx.shape == (B, N, d) and y_ctx.shape == (B, N) and x_q.shape == (B, d) and y_q.shape == (B,)
    mA = ModelA(w, 2)
    assert mA.forward(x_ctx, y_ctx, x_q).shape == (B,)
    loss, gw, gG = mA.loss_grad(x_ctx, y_ctx, x_q, y_q)
    assert gw.shape == (d,) and np.isscalar(gG)
    x, y = sample_iw(d, B, v, 2, rng)
    assert x.shape == (B, d) and y.shape == (B,)
    mB = ModelB(w, 2)
    assert mB.forward(x).shape == (B,) and mB.loss_grad(x, y)[1].shape == (d,)
    p, u, pq, uq, c = sample_prompts_proj(N, B, 2, 1.0, rng)
    assert p.shape == (B, N) and pq.shape == (B,) and c.shape == (B,)
    assert sample_iw_proj(B, 2, rng)[0].shape == (B,)


def test_model_a_zero_loss_at_m1_large_N():
    """At w = v, Gamma = 1/g(1) ... : yhat_q = Gamma sigma(p_q) * c * mean sigma(p_i)^2 -> c sigma(p_q) for large N."""
    rng = np.random.default_rng(1)
    d, N, B = 8, 20000, 4
    v, _ = init_sphere(d, rng)
    for k in (1, 2):
        model = ModelA(v, k, gamma=1.0)
        batch = sample_prompts(d, N, B, v, k, 1.0, rng)
        assert model.loss(*batch) < 5e-3, (k, model.loss(*batch))
    # and Model B at w = v with a = 1 has exactly zero loss
    x, y = sample_iw(d, 64, v, 2, rng)
    assert ModelB(v, 2).loss(x, y) < 1e-24


def _fd_check(loss_fn, params, grad, eps=1e-6):
    num = np.zeros_like(params)
    for i in range(params.size):
        e = np.zeros_like(params); e[i] = eps
        num[i] = (loss_fn(params + e) - loss_fn(params - e)) / (2 * eps)
    return np.linalg.norm(num - grad) / np.linalg.norm(num)


def test_gradients_finite_difference():
    """Analytic gradients vs central differences (float64, relative error < 1e-4)."""
    rng = np.random.default_rng(2)
    d, N, B = 6, 10, 7
    v, w = init_sphere(d, rng)
    for k in KS:
        batch = sample_prompts(d, N, B, v, k, 1.0, rng)
        mA = ModelA(w, k, gamma=0.3)
        _, gw, gG = mA.loss_grad(*batch)
        err = _fd_check(lambda ww: ModelA(ww, k, 0.3).loss(*batch), w.copy(), gw)
        assert err < 1e-4, ("A", k, err)
        eps = 1e-6  # d/dGamma
        num = (ModelA(w, k, 0.3 + eps).loss(*batch) - ModelA(w, k, 0.3 - eps).loss(*batch)) / (2 * eps)
        assert abs(num - gG) / abs(num) < 1e-4
        x, y = sample_iw(d, B, v, k, rng)
        mB = ModelB(w, k, a=0.7)
        _, gw, ga = mB.loss_grad(x, y)
        assert _fd_check(lambda ww: ModelB(ww, k, 0.7).loss(x, y), w.copy(), gw) < 1e-4
        num = (ModelB(w, k, 0.7 + eps).loss(x, y) - ModelB(w, k, 0.7 - eps).loss(x, y)) / (2 * eps)
        assert abs(num - ga) / abs(num) < 1e-4


def test_tied_gradient_finite_difference():
    """Model A-tied: analytic gradient wrt the UNNORMALISED w (directional projection + 2 w dL/dGamma) vs central
    differences, relative error < 1e-4, for every activation and for |w| != 1."""
    rng = np.random.default_rng(5)
    d, N, B = 6, 10, 7
    v, w = init_sphere(d, rng)
    for k in KS:
        batch = sample_prompts(d, N, B, v, k, 1.0, rng)
        for rho in (0.01, 0.7, 2.5):
            wu = math.sqrt(rho) * w + 0.0
            mdl = ModelATied(wu, k)
            assert abs(mdl.rho - rho) < 1e-12
            _, gw, _ = mdl.loss_grad(*batch)
            err = _fd_check(lambda ww: ModelATied(ww, k).loss(*batch), wu.copy(), gw)
            assert err < 1e-4, ("At", k, rho, err)
            # consistency with the free-Gamma model at (w_hat, Gamma=rho): same loss
            assert abs(mdl.loss(*batch) - ModelA(w, k, rho).loss(*batch)) < 1e-12


def test_init_fixed_and_tied_proj_matches_full():
    rng = np.random.default_rng(6)
    d = 20
    v, w = init_fixed(d, rng, 0.3)
    assert abs(v @ w - 0.3) < 1e-12 and abs(np.linalg.norm(w) - 1) < 1e-12
    # projected sampler + tied chain rule has the same mean/spread of dL/dw (unnormalised) as full sampling
    d, N, B, nb, m, rho = 6, 8, 4, 6000, 0.35, 0.8
    v, _ = init_sphere(d, rng)
    u = rng.standard_normal(d); u -= (u @ v) * v; u /= np.linalg.norm(u)
    s = math.sqrt(1 - m * m)
    wu = math.sqrt(rho) * (m * v + s * u)
    mdl = ModelATied(wu, 2)
    full = np.empty((nb, 3)); proj = np.empty((nb, 3))
    r2 = np.random.default_rng(7)
    for i in range(nb):
        _, gw, _ = mdl.loss_grad(*sample_prompts(d, N, B, v, 2, 1.0, rng))
        p, uu, pq, uq, c = sample_prompts_proj(N, B, 2, 1.0, r2)
        _, Cc, Cq, gG = mdl.coefs(m * p + s * uu, c[:, None] * sigma(2, p), m * pq + s * uq, c * sigma(2, pq))
        Gv, Ge, S2 = (Cc * p).sum() + Cq @ pq, (Cc * uu).sum() + Cq @ uq, (Cc ** 2).sum() + Cq @ Cq
        xi = r2.standard_normal(d); xi -= (xi @ v) * v; xi -= (xi @ u) * u
        gp = mdl.w_grad(Gv * v + Ge * u + math.sqrt(S2) * xi, gG)
        full[i] = [gw @ v, gw @ u, gw @ gw]
        proj[i] = [gp @ v, gp @ u, gp @ gp]
    for j in range(3):
        se = math.sqrt(full[:, j].var() / nb + proj[:, j].var() / nb)
        assert abs(full[:, j].mean() - proj[:, j].mean()) < 5 * se, j


def test_fast_ctx_stats_matches_coefs():
    rng = np.random.default_rng(8)
    B, N = 5, 20
    for k in KS:
        pc, uc = rng.standard_normal((B, N)), rng.standard_normal((B, N))
        pq, uq, c = rng.standard_normal((3, B))
        m, s, G = 0.3, math.sqrt(1 - 0.09), 0.7
        loss, Cc, Cq, gG = ModelA(np.ones(3), k, G).coefs(m * pc + s * uc, c[:, None] * sigma(k, pc), m * pq + s * uq, c * sigma(k, pq))
        ref = (loss, (Cc * pc).sum() + Cq @ pq, (Cc * uc).sum() + Cq @ uq, (Cc ** 2).sum() + Cq @ Cq, gG)
        assert np.allclose(ref, ctx_stats_fast(k, G, m, s, pc, uc, pq, uq, c), rtol=1e-12), k


def test_closed_form_g_V():
    for m in (0.0, 0.3, 0.7, 1.0):
        assert abs(g_fn(1, m) - m) < 1e-10 and abs(g_fn(2, m) - m ** 2) < 1e-10 and abs(g_fn(3, m) - m ** 3) < 1e-10
        assert abs(V_fn(1, m) - (1 + 2 * m * m)) < 1e-10
        assert abs(V_fn(2, m) - (1 + 8 * m * m + 6 * m ** 4)) < 1e-10
    # L_A(m=1) at Gamma = 1: 1 - 2 + 1*(1-1/N + V(1)/N) = (V(1)-1)/N  -> 0 as N -> infinity
    assert abs(loss_formula("A", 2, 1.0, N=10 ** 6, gamma=1.0)) < 1e-4
    assert abs(loss_formula("B", 2, 1.0)) < 1e-12


def test_projected_sampler_matches_full():
    """The fast projected sampler must give the same gradient distribution as full sampling (mean and spread)."""
    d, N, B, nb, m = 6, 8, 4, 6000, 0.35
    rng = np.random.default_rng(3)
    v, _ = init_sphere(d, rng)
    u = rng.standard_normal(d); u -= (u @ v) * v; u /= np.linalg.norm(u)
    s = math.sqrt(1 - m * m)
    w = m * v + s * u
    e2 = u
    for model, k in [("A", 2), ("B", 2), ("A", 1)]:
        mdl = ModelA(w, k, 0.5) if model == "A" else ModelB(w, k)
        full = np.empty((nb, 3)); proj = np.empty((nb, 3))
        r2 = np.random.default_rng(4)
        for i in range(nb):
            if model == "A":
                _, gw, _ = mdl.loss_grad(*sample_prompts(d, N, B, v, k, 1.0, rng))
                p, uu, pq, uq, c = sample_prompts_proj(N, B, k, 1.0, r2)
                _, Cc, Cq, _ = mdl.coefs(m * p + s * uu, c[:, None] * sigma(k, p), m * pq + s * uq, c * sigma(k, pq))
                Gv, Ge, S2 = (Cc * p).sum() + Cq @ pq, (Cc * uu).sum() + Cq @ uq, (Cc ** 2).sum() + Cq @ Cq
            else:
                _, gw, _ = mdl.loss_grad(*sample_iw(d, B, v, k, rng))
                p, uu = sample_iw_proj(B, k, r2)
                _, C, _ = mdl.coefs(m * p + s * uu, sigma(k, p))
                Gv, Ge, S2 = C @ p, C @ uu, C @ C
            full[i] = [gw @ v, gw @ e2, gw @ gw]
            proj[i] = [Gv, Ge, Gv ** 2 + Ge ** 2 + S2 * (d - 2)]  # E|G_perp|^2 = S2 (d-2)
        for j in range(3):
            se = math.sqrt(full[:, j].var() / nb + proj[:, j].var() / nb)
            assert abs(full[:, j].mean() - proj[:, j].mean()) < 5 * se, (model, k, j)


def test_train_smoke_escape():
    r = train("B", 8, 2, 2.0 / 64, seed=0, max_steps=20000)
    assert r["reached"] and r["T05"] <= r["T09"] and 0 < r["m0"] < 1
    assert len(r["traj"]["t"]) == len(r["traj"]["m"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
