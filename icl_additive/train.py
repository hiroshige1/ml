"""Online spherical SGD for Model A / Model B (fresh data every step, renormalise w after each step)."""
import math
import time

import numpy as np

from .data import (sample_iw, sample_iw_proj, sample_prompts, sample_prompts_proj)
from .hermite import sigma
from .models import ModelA, ModelB

K_CODE = {1: 1, 2: 2, 3: 3, "relu": 9}
M_CODE = {"A": 1, "B": 2}


def make_rng(seed, d, k, model):
    """Explicit, config-dependent stream; the same seed gives the same v, w_0 and data for different eta_0."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), d, K_CODE[k], M_CODE[model]]))


def sign_symmetric(model, k):
    """True if the population loss is invariant under w -> -w (then only |<w,v>| is identifiable).

    Model A: yhat is a product of two sigma_k(<w,.>) factors, so it is even in w for every k.
    Model B: yhat = a sigma_k(<w,x>) is even in w only for even k (k = 2).  relu is never symmetric.
    """
    if k == "relu":
        return False
    return model == "A" or k % 2 == 0


def init_sphere(d, rng):
    """Teacher v uniform on S^{d-1}; w uniform on S^{d-1}, sign-flipped so <w,v> > 0."""
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    if w @ v < 0:
        w = -w
    return v, w


def train(model, d, k, eta, seed, N=128, B=32, gamma=0.1, train_gamma=False, a=1.0, train_a=False,
          task_std=1.0, max_steps=2_000_000, log_every=500, stop_m=0.9, sampler="proj"):
    """Run one training run. sampler: 'proj' (distribution-exact projected sampler, fast) or 'full'.

    Returns a record dict with scalars (seed, m0, T05, T09, reached, ...) and a 'traj' sub-dict of arrays
    (t, m, loss, gnorm) logged every `log_every` steps (loss/gnorm are running means over the interval).
    """
    t_start = time.time()
    rng = make_rng(seed, d, k, model)
    v, w0 = init_sphere(d, rng)
    mdl = ModelA(w0, k, gamma, train_gamma) if model == "A" else ModelB(w0, k, a, train_a)
    m = float(mdl.w @ v)
    m0 = m
    sym = sign_symmetric(model, k)  # thresholds are applied to |m| if w -> -w is a symmetry (see docstring)
    T05 = T09 = float("nan")
    ts, ms, ls, gs = [0], [m], [float("nan")], [float("nan")]
    loss_acc = g_acc = 0.0
    n_acc = 0
    t = 0
    while t < max_steps:
        t += 1
        w = mdl.w
        if sampler == "full":
            if model == "A":
                batch = sample_prompts(d, N, B, v, k, task_std, rng)
                loss, gw, gp = mdl.loss_grad(*batch)
            else:
                batch = sample_iw(d, B, v, k, rng)
                loss, gw, gp = mdl.loss_grad(*batch)
        else:
            s = math.sqrt(max(1.0 - m * m, 1e-12))
            e2 = (w - m * v) / s
            if model == "A":
                p_c, u_c, p_q, u_q, c = sample_prompts_proj(N, B, k, task_std, rng)
                y_c = c[:, None] * sigma(k, p_c)
                y_q = c * sigma(k, p_q)
                loss, C_c, C_q, gp = mdl.coefs(m * p_c + s * u_c, y_c, m * p_q + s * u_q, y_q)
                Gv = float((C_c * p_c).sum() + C_q @ p_q)
                Ge = float((C_c * u_c).sum() + C_q @ u_q)
                S2 = float((C_c * C_c).sum() + C_q @ C_q)
            else:
                p, u = sample_iw_proj(B, k, rng)
                loss, C, gp = mdl.coefs(m * p + s * u, sigma(k, p))
                Gv, Ge, S2 = float(C @ p), float(C @ u), float(C @ C)
            xi = rng.standard_normal(d)
            xi -= (xi @ v) * v
            xi -= (xi @ e2) * e2
            gw = Gv * v + Ge * e2 + math.sqrt(S2) * xi
        mdl.update(gw, gp, eta)
        m = float(mdl.w @ v)
        loss_acc += loss
        g_acc += float(np.linalg.norm(gw))
        n_acc += 1
        mc = abs(m) if sym else m
        if math.isnan(T05) and mc >= 0.5:
            T05 = float(t)
        if mc >= stop_m:
            T09 = float(t)
        if t % log_every == 0 or T09 == T09 or t == max_steps:
            ts.append(t); ms.append(m); ls.append(loss_acc / n_acc); gs.append(g_acc / n_acc)
            loss_acc = g_acc = 0.0
            n_acc = 0
        if T09 == T09:
            break
        if not np.isfinite(m):
            break
    return dict(model=model, k=k, d=d, N=N, B=B, gamma=gamma, eta=eta, seed=seed, m0=m0, T05=T05, T09=T09,
                reached=bool(T09 == T09), reached05=bool(T05 == T05), steps=t, final_m=m,
                max_steps=max_steps, sampler=sampler, sym_stop=sym, wall_s=time.time() - t_start,
                traj=dict(t=np.array(ts), m=np.array(ms), loss=np.array(ls), gnorm=np.array(gs)))
