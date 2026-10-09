"""B-tied tests: python tests/test_btied.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icl_additive.btied import BTied  # noqa: E402
from icl_additive.many import skill_freqs  # noqa: E402
from icl_additive.multi import init_multi  # noqa: E402


def test_btied_gradient_fd_and_pop_mse():
    rng = np.random.default_rng(11)
    d, B, M, P = 32, 64, 64, 16
    V, W = init_multi(d, P, M, rng)
    a = np.sqrt(skill_freqs(P, 1.5))
    for scale in (0.1, 1.0, 0.5):  # |u|^2 = 0.01 (init), 1, 0.25 with random per-neuron norm spread
        U = W * (scale * np.exp(0.5 * rng.standard_normal((M, 1))))
        x = rng.standard_normal((B, d))
        mdl = BTied(U, V, a)
        y = mdl.target(x)
        loss, g = mdl.loss_grad(x, y)
        assert abs(loss - mdl.loss(x, y)) < 1e-12
        # all coordinates of 60 random neurons' worth would be slow: use 200 random coordinates, central differences
        errs = []
        for i in rng.choice(U.size, size=200, replace=False):
            e = np.zeros(U.size); e[i] = 1e-6 * max(1.0, abs(U.ravel()[i]))
            h = e[i]
            lp = BTied((U.ravel() + e).reshape(M, d), V, a).loss(x, y)
            lm = BTied((U.ravel() - e).reshape(M, d), V, a).loss(x, y)
            errs.append(abs((lp - lm) / (2 * h) - g.ravel()[i]) / (abs(g.ravel()[i]) + 1e-8 * max(1.0, np.abs(g).max())))
        err = max(errs)
        # full-vector relative error along a random direction
        dirn = rng.standard_normal(U.shape); dirn /= np.linalg.norm(dirn)
        h = 1e-6
        dd = (BTied(U + h * dirn, V, a).loss(x, y) - BTied(U - h * dirn, V, a).loss(x, y)) / (2 * h)
        rel_dir = abs(dd - np.sum(g * dirn)) / abs(dd)
        assert err < 1e-4 and rel_dir < 1e-6, (err, rel_dir)
        xs = rng.standard_normal((400000, d))
        r2 = (mdl.forward(xs) - mdl.target(xs)) ** 2
        se = r2.std() / np.sqrt(len(xs))
        assert abs(r2.mean() - mdl.pop_mse()) < 4 * se, (r2.mean(), mdl.pop_mse(), se)
        print(f"B-tied scale {scale}: max coord FD rel err {err:.2e}, directional rel err {rel_dir:.2e}; pop MSE {mdl.pop_mse():.4f} vs MC {r2.mean():.4f} +- {se:.4f}")


if __name__ == "__main__":
    test_btied_gradient_fd_and_pop_mse()
    print("ok")
