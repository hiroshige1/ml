"""Monte-Carlo population drift vs the closed-form population losses.

Closed forms (alpha_k = Hermite coefficients of sigma, s^2 = E[c^2] = task_std^2, P = 1 teacher):

    g(m)  = sum_k alpha_k^2 m^k = E[sigma(<v,x>) sigma(<w,x>)]           (sigma = sigma_k: g(m) = m^k)
    V(m)  = E[sigma(<v,x>)^2 sigma(<w,x>)^2]       (sigma_1: 1 + 2 m^2;  sigma_2: 1 + 8 m^2 + 6 m^4)
    L_B(m) = 1 - 2 a g(m) + a^2 g(1)
    L_A(m) = s^2 [ 1 - 2 gamma g(m)^2 + gamma^2 g(1) ( (1 - 1/N) g(m)^2 + V(m)/N ) ]

Loss is the plain square loss E[(yhat - y)^2] (no factor 1/2).  g and V are evaluated numerically by 2D
Gauss-Hermite quadrature (exact for polynomial sigma; tests/test_basic.py checks them against the closed
forms above), and the formula drift is  -dL/dm  by a central finite difference.

Monte-Carlo drift at overlap m: take w = m v + sqrt(1-m^2) u with u uniform on the unit sphere of v^perp,
average the analytic gradient of the batch loss over many fresh batches, and project onto the unit tangent
direction toward v,  t = (v - m w)/sqrt(1-m^2).  Moving w along t by angle theta changes m by
sqrt(1-m^2) theta, hence  -dL/dm = -(grad L . t)/sqrt(1-m^2).  That is the reported `drift_mc`.
(The projection only involves the (v, w)-plane, so it does not depend on d; d only sets the cost.)
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from .data import sample_iw, sample_prompts
from .hermite import sigma
from .models import ModelA, ModelB
from .train import K_CODE, M_CODE

_X, _W = np.polynomial.hermite_e.hermegauss(120)
_W = _W / math.sqrt(2.0 * math.pi)


def _expect2(f, m):
    """E[f(p, z)] with p ~ N(0,1), z = m p + sqrt(1-m^2) y, y ~ N(0,1) independent."""
    s = math.sqrt(max(1.0 - m * m, 0.0))
    p = _X[:, None]
    z = m * p + s * _X[None, :]
    return float(np.sum(_W[:, None] * _W[None, :] * f(p, z)))


def g_fn(k, m):
    return _expect2(lambda p, z: sigma(k, p) * sigma(k, z), m)


def V_fn(k, m):
    return _expect2(lambda p, z: sigma(k, p) ** 2 * sigma(k, z) ** 2, m)


def loss_formula(model, k, m, N=128, gamma=0.1, a=1.0, task_std=1.0):
    g, g1 = g_fn(k, m), g_fn(k, 1.0)
    if model == "B":
        return 1.0 - 2.0 * a * g + a * a * g1
    return task_std ** 2 * (1.0 - 2.0 * gamma * g * g
                            + gamma ** 2 * g1 * ((1.0 - 1.0 / N) * g * g + V_fn(k, m) / N))


def drift_formula(model, k, m, h=1e-4, **kw):
    return -(loss_formula(model, k, m + h, **kw) - loss_formula(model, k, m - h, **kw)) / (2 * h)


def drift_mc(model, k, m, d=16, N=128, B=64, n_batches=2000, gamma=0.1, a=1.0, task_std=1.0, seed=0):
    """Return (mean, standard error) of the MC drift over n_batches fresh batches."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, d, K_CODE[k], M_CODE[model], int(round(m * 1000))]))
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    u = rng.standard_normal(d); u -= (u @ v) * v; u /= np.linalg.norm(u)
    s = math.sqrt(1 - m * m)
    w = m * v + s * u
    t = (v - m * w) / s
    mdl = ModelA(w, k, gamma) if model == "A" else ModelB(w, k, a)
    vals = np.empty(n_batches)
    for i in range(n_batches):
        if model == "A":
            _, gw, _ = mdl.loss_grad(*sample_prompts(d, N, B, v, k, task_std, rng))
        else:
            _, gw, _ = mdl.loss_grad(*sample_iw(d, B, v, k, rng))
        vals[i] = -(gw @ t) / s
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(n_batches))


def _job(args):
    model, k, m, kw = args
    t0 = time.time()
    mc, se = drift_mc(model, k, m, **kw)
    return dict(model=model, k=k, m=m, drift_mc=mc, se=se,
                drift_formula=drift_formula(model, k, m, N=kw["N"], gamma=kw["gamma"], a=kw["a"],
                                            task_std=kw["task_std"]),
                n_batches=kw["n_batches"], B=kw["B"], d=kw["d"], N=kw["N"], gamma=kw["gamma"], a=kw["a"],
                wall_s=time.time() - t0)


def loglog_slope(x, y):
    r = stats.linregress(np.log(x), np.log(y))
    return r.slope, r.stderr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/exp1")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--ms", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--n-batches", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--task-std", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    kw = dict(d=args.d, N=args.N, B=args.B, n_batches=args.n_batches, gamma=args.gamma, a=args.a,
              task_std=args.task_std, seed=args.seed)
    jobs = [(mo, k, m, kw) for mo in "AB" for k in args.ks for m in args.ms]
    with mp.Pool(args.workers, initializer=lambda: __import__("os").environ.update(OMP_NUM_THREADS="1")) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df["ratio"] = df["drift_mc"] / df["drift_formula"]
    df["z_score"] = (df["drift_mc"] - df["drift_formula"]) / df["se"]
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "drift.csv"), index=False)
    srows = []
    for (mo, k), g in df.groupby(["model", "k"]):
        s_mc, e_mc = loglog_slope(g.m, np.abs(g.drift_mc)) if (g.drift_mc > 0).all() else (float("nan"),) * 2
        s_f, e_f = loglog_slope(g.m, g.drift_formula) if (g.drift_formula > 0).all() else (float("nan"),) * 2
        pred = k - 1 if mo == "B" else 2 * k - 1
        srows.append(dict(model=mo, k=k, slope_mc=s_mc, slope_mc_se=e_mc, slope_formula=s_f, slope_formula_se=e_f,
                          predicted=pred))
    sl = pd.DataFrame(srows)
    sl.to_csv(os.path.join(args.out, "drift_slopes.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.drop(columns=["wall_s"]).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(sl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
