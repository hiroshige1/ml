"""Online spherical SGD for Model A / Model B (fresh data every step, renormalise w after each step)."""
import math
import time

import numpy as np

from .data import (sample_iw, sample_iw_proj, sample_prompts, sample_prompts_proj)
from .hermite import dsigma, sigma
from .models import ModelA, ModelATied, ModelB

K_CODE = {1: 1, 2: 2, 3: 3, "relu": 9}
M_CODE = {"A": 1, "B": 2, "At": 3}


def make_rng(seed, d, k, model, tag=None):
    """Explicit, config-dependent stream; the same seed gives the same v, w_0 and data for different eta_0.
    `tag` (exp 1b: integer code of the m_0 multiplier) decorrelates streams across m_0 values; None = exp-1 stream."""
    ent = [int(seed), d, K_CODE[k], M_CODE[model]] + ([] if tag is None else [int(tag)])
    if tag is None:
        return np.random.default_rng(np.random.SeedSequence(ent))
    return np.random.Generator(np.random.SFC64(np.random.SeedSequence(ent)))  # exp 1b: ~20% faster normals


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


def ctx_stats_fast(k, G, m, s, p_c, u_c, p_q, u_q, c):
    """Fused version of ModelA.coefs + the (Gv, Ge, S2) reductions of the projected sampler (identical maths, fewer
    temporaries).  G is Gamma (|w|^2 for the tied model).  Returns loss, Gv, Ge, S2, dL/dGamma.
    Uses C_ctx[b,i] = kap_b * q[b,i] with q = sigma_k(p) sigma_k'(z) and kap_b = f_b G sigma(z_q) c_b / N."""
    B, N = p_c.shape
    z_c = m * p_c + s * u_c
    sp = sigma(k, p_c)
    A = c * np.einsum("bn,bn->b", sp, sigma(k, z_c)) / N
    z_q = m * p_q + s * u_q
    sq = sigma(k, z_q)
    r = G * sq * A - c * sigma(k, p_q)
    f = 2.0 * r / B
    C_q = f * G * dsigma(k, z_q) * A
    kap = f * G * sq * c / N
    q = sp * dsigma(k, z_c)
    Gv = float(kap @ np.einsum("bn,bn->b", q, p_c) + C_q @ p_q)
    Ge = float(kap @ np.einsum("bn,bn->b", q, u_c) + C_q @ u_q)
    S2 = float((kap * kap) @ np.einsum("bn,bn->b", q, q) + C_q @ C_q)
    return float(r @ r / B), Gv, Ge, S2, float(np.sum(f * sq * A))


def init_fixed(d, rng, m0):
    """Teacher v uniform on S^{d-1}; w = m0 v + sqrt(1-m0^2) u with u uniform on the sphere orthogonal to v."""
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    u = rng.standard_normal(d)
    u -= (u @ v) * v
    u /= np.linalg.norm(u)
    return v, m0 * v + math.sqrt(1.0 - m0 * m0) * u


def train(model, d, k, eta, seed, N=128, B=32, gamma=0.1, train_gamma=False, a=1.0, train_a=False,
          task_std=1.0, max_steps=2_000_000, log_every=500, stop_m=0.9, sampler="proj",
          init="uniform", m0=None, m0_scale=1.0, eta_gamma_mult=1.0, rho0=0.01, rng_tag=None, fast=False):
    """Run one training run. sampler: 'proj' (distribution-exact projected sampler, fast) or 'full'.

    init: 'uniform' (exp 1: w uniform on the sphere, sign-flipped) or 'fixed' (exp 1b: <w,v> = m0 exactly, with
    m0 = m0_scale * d^{-1/2} unless m0 is given).  model: 'A' (fixed gamma, or free trainable gamma with learning
    rate eta*eta_gamma_mult if train_gamma), 'At' (A-tied, unnormalised w, Gamma = |w|^2 = rho, |w_0|^2 = rho0), 'B'.

    Returns a record dict with scalars (seed, m0, T05, T09, reached, ...) and a 'traj' sub-dict of arrays
    (t, m, loss, gnorm) logged every `log_every` steps (loss/gnorm are running means over the interval).
    """
    t_start = time.time()
    rng = make_rng(seed, d, k, model, rng_tag)
    if init == "fixed":
        v, w0 = init_fixed(d, rng, m0 if m0 is not None else m0_scale * d ** -0.5)
    else:
        v, w0 = init_sphere(d, rng)
    if model == "A":
        mdl = ModelA(w0, k, gamma, train_gamma)
    elif model == "At":
        mdl = ModelATied(math.sqrt(rho0) * w0, k)
    else:
        mdl = ModelB(w0, k, a, train_a)
    tied = model == "At"
    ctx_model = model in ("A", "At")
    what = (lambda: mdl.what) if tied else (lambda: mdl.w)
    gam = (lambda: mdl.rho) if tied else ((lambda: mdl.gamma) if model == "A" else (lambda: float("nan")))
    eta_g = eta * eta_gamma_mult
    m = float(what() @ v)
    m0 = m
    max_abs_m = abs(m)
    sym = sign_symmetric(model, k)  # thresholds are applied to |m| if w -> -w is a symmetry (see docstring)
    T05 = T09 = float("nan")
    ts, ms, ls, gs, gams = [0], [m], [float("nan")], [float("nan")], [gam()]
    loss_acc = g_acc = 0.0
    n_acc = 0
    t = 0
    while t < max_steps:
        t += 1
        w = what()
        if sampler == "full":
            if ctx_model:
                batch = sample_prompts(d, N, B, v, k, task_std, rng)
                loss, gw, gp = mdl.loss_grad(*batch)
            else:
                batch = sample_iw(d, B, v, k, rng)
                loss, gw, gp = mdl.loss_grad(*batch)
        else:
            s = math.sqrt(max(1.0 - m * m, 1e-12))
            e2 = (w - m * v) / s
            if ctx_model and fast:
                pu = rng.standard_normal((2, B, N + 1))
                c = task_std * rng.standard_normal(B)
                loss, Gv, Ge, S2, gp = ctx_stats_fast(k, mdl.gamma, m, s, pu[0, :, :N], pu[1, :, :N], pu[0, :, N],
                                                      pu[1, :, N], c)
            elif ctx_model:
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
            if tied:
                gw = mdl.w_grad(gw, gp)  # gw was g_hat = dL/dw_hat; chain rule to the unnormalised w
        mdl.update(gw, gp, eta, eta_g)
        m = float(what() @ v)
        if abs(m) > max_abs_m:
            max_abs_m = abs(m)
        loss_acc += loss
        g_acc += float(np.linalg.norm(gw))
        n_acc += 1
        mc = abs(m) if sym else m
        if math.isnan(T05) and mc >= 0.5:
            T05 = float(t)
        if mc >= stop_m:
            T09 = float(t)
        if t % log_every == 0 or T09 == T09 or t == max_steps:
            ts.append(t); ms.append(m); ls.append(loss_acc / n_acc); gs.append(g_acc / n_acc); gams.append(gam())
            loss_acc = g_acc = 0.0
            n_acc = 0
        if T09 == T09:
            break
        if not np.isfinite(m):
            break
    G05 = float("nan")
    if T05 == T05:
        G05 = float(gams[min(range(len(ts)), key=lambda i: abs(ts[i] - T05))])  # Gamma at the log point nearest T05
    return dict(model=model, k=k, d=d, N=N, B=B, gamma=gamma, eta=eta, seed=seed, m0=m0, T05=T05, T09=T09,
                reached=bool(T09 == T09), reached05=bool(T05 == T05), steps=t, final_m=m,
                max_steps=max_steps, sampler=sampler, sym_stop=sym, wall_s=time.time() - t_start, max_abs_m=max_abs_m,
                final_gamma=gam(), gamma_at_T05=G05, init=init, rho0=rho0 if tied else float("nan"),
                eta_gamma_mult=eta_gamma_mult if (train_gamma and model == "A") else float("nan"),
                traj=dict(t=np.array(ts), m=np.array(ms), loss=np.array(ls), gnorm=np.array(gs),
                          gamma=np.array(gams)))
