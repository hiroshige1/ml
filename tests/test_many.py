"""Exp-2 tests: python tests/test_many.py  (or pytest)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from icl_additive.hermite import sigma  # noqa: E402
from icl_additive.many import (EvalSkills, MultiModelB, grad_A_fast, sample_prompts_full, skill_freqs)  # noqa: E402
from icl_additive.multi import MultiModelA, init_multi  # noqa: E402


def _fd(f, x, g, eps=1e-6, n=12, seed=0):
    rng = np.random.default_rng(seed)
    errs = []
    for i in rng.choice(x.size, size=n, replace=False):
        e = np.zeros_like(x); e[i] = eps
        errs.append(abs((f(x + e) - f(x - e)) / (2 * eps) - g[i]) / (abs(g[i]) + 1e-8))
    return max(errs)


def test_A_gradient_M64_P16_finite_difference_and_fast_path():
    rng = np.random.default_rng(3)
    d, N, B, M, P = 32, 128, 32, 64, 16
    V, W = init_multi(d, P, M, rng)
    pi = skill_freqs(P, 1.5)
    X, y, _ = sample_prompts_full(d, N, B, V, pi, rng)
    mdl = MultiModelA(W, 2, 0.1)
    args = (X[:, :N], y[:, :N], X[:, N], y[:, N])
    loss, gW = mdl.loss_grad(*args)
    err = _fd(lambda wf: MultiModelA(wf.reshape(M, d), 2, 0.1).loss(*args), W.ravel().copy(), gW.ravel(), n=40)
    assert err < 1e-4, err
    l2, g2 = grad_A_fast(W, mdl.gamma, X, y)
    assert abs(l2 - loss) < 1e-12 and np.max(np.abs(g2 - gW)) < 1e-12 * max(1, np.abs(gW).max()), np.max(np.abs(g2 - gW))
    print("A: FD rel err", err, " fast-vs-reference max diff", np.max(np.abs(g2 - gW)))


def test_B_gradient_finite_difference_and_pop_mse():
    rng = np.random.default_rng(5)
    d, B, M, P = 32, 64, 64, 16
    V, W = init_multi(d, P, M, rng)
    a = np.sqrt(skill_freqs(P, 1.5))
    for rho in (1.0, 0.1):
        x = rng.standard_normal((B, d))
        mdl = MultiModelB(W, V, a, 2, rho)
        y = mdl.target(x)
        loss, gW = mdl.loss_grad(x, y)
        err = _fd(lambda wf: MultiModelB(wf.reshape(M, d), V, a, 2, rho).loss(x, y), W.ravel().copy(), gW.ravel(), n=40)
        assert err < 1e-5, err
        # exact population MSE vs Monte Carlo
        xs = rng.standard_normal((400000, d))
        mc = np.mean((mdl.forward(xs) - mdl.target(xs)) ** 2)
        se = np.std((mdl.forward(xs) - mdl.target(xs)) ** 2) / np.sqrt(len(xs))
        assert abs(mc - mdl.pop_mse()) < 4 * se, (mc, mdl.pop_mse(), se)
        print(f"B rho={rho}: FD rel err {err:.2e}; pop MSE {mdl.pop_mse():.4f} vs MC {mc:.4f} +- {se:.4f}")


def test_eval_skills_matches_multimodelA():
    rng = np.random.default_rng(9)
    d, N, M, P = 8, 16, 6, 3
    V, W = init_multi(d, P, M, rng)
    ev = EvalSkills(V, d, N, n=64, H=2)
    ms, mh = ev.per_skill_mse(W, np.full(M, 0.1))
    mdl = MultiModelA(W, 2, 0.1)
    p = 1
    yc = ev.c[:, p, None] * ev.Sc[:, :, p]
    errs = []
    for b in range(64):
        for h in range(3):
            errs.append(mdl.forward(ev.X[b:b + 1], yc[b:b + 1], ev.Q[b:b + 1, h])[0] - ev.c[b, p] * ev.Sq[b, h, p])
    errs = np.array(errs).reshape(64, 3) ** 2
    assert abs(errs[:, 0].mean() - ms[p]) < 1e-10 and abs(errs.mean() - mh[p]) < 1e-10


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
