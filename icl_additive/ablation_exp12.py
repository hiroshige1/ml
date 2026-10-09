"""Experiment 12: ingredient ablation (optimiser, softmax) in the single-feature model (torch, CPU, 1 thread).

Spec: docs/spec_exp12.md; pre-registration: docs/preregistration.md (P25a-d).
Model: w on S^{d-1}, pinned readout Gamma=1, sigma_2.  linear: yhat = sigma(w.x_q) * mean_i y_i sigma(w.x_i);
softmax: yhat = sum_i softmax_i(sigma(w.x_i) sigma(w.x_q)) y_i.  Loss = mean_b (yhat - y_q)^2.  w <- w/|w| after each step.

Usage:
  python -m icl_additive.ablation_exp12 --sanity
  python -m icl_additive.ablation_exp12 --worker K            # dynamic job queue (claim files), N=16 cells first
  python -m icl_additive.ablation_exp12 --extra --worker K    # declared extra: Adam lr 1e-4, N=16,B=1024
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)

from .train import init_fixed

D = 64
M0 = D ** -0.5
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp12")
SQRT2 = math.sqrt(2.0)
CAPS = {"sgd": 300_000, "adam": 100_000}
LOG_EVERY = 100
STOP_M = 0.5


def sig2(z):
    return (z * z - 1.0) / SQRT2


def yhat_fn(model, w, xc, yc, xq):
    """xc (B,N,d), yc (B,N), xq (B,d) -> yhat (B,).  Pinned Gamma = 1."""
    zc = xc @ w
    zq = xq @ w
    sc, sq = sig2(zc), sig2(zq)
    if model == "linear":
        return sq * (yc * sc).mean(dim=1)
    if model == "softmax":
        a = torch.softmax(sc * sq[:, None], dim=1)
        return (a * yc).sum(dim=1)
    raise ValueError(model)


def loss_fn(model, w, xc, yc, xq, yq):
    r = yhat_fn(model, w, xc, yc, xq) - yq
    return (r * r).mean()


def sample(gen, v, N, B, dtype):
    X = torch.randn(B, N + 1, D, generator=gen, dtype=dtype)
    c = torch.randn(B, generator=gen, dtype=dtype)
    y = c[:, None] * sig2(X @ v)
    return X[:, :N], y[:, :N], X[:, N], y[:, N]


def yhat_z(model, zc, yc, zq):
    """Same as yhat_fn but from the projections z = w.x (what the loss depends on)."""
    sc, sq = sig2(zc), sig2(zq)
    if model == "linear":
        return sq * (yc * sc).mean(dim=1)
    a = torch.softmax(sc * sq[:, None], dim=1)
    return (a * yc).sum(dim=1)


def proj_sgd_grad(model, w, v, N, B, gen):
    """Distribution-exact gradient of the batch loss wrt w for the projected sampler (rotation-equivariant => SGD only).

    x = p v + u e2 + xi, e2 = unit vector in span(v,w) orthogonal to v, p,u ~ N(0,1), xi ~ N(0,I) on the complement of span(v,e2).
    <v,x> = p, <w,x> = m p + sqrt(1-m^2) u; loss depends on x only via (p,u,c).  dL/dw = sum_j C_j x_j with C_j = dL/dz_j,
    whose complement part is N(0, sum C_j^2 I) exactly.  Returns (grad, loss)."""
    m = float(w @ v)
    s = math.sqrt(max(1.0 - m * m, 1e-12))
    e2 = (w - m * v) / s
    pu = torch.randn(B, N + 1, 2, generator=gen, dtype=w.dtype)
    c = torch.randn(B, generator=gen, dtype=w.dtype)
    p, u = pu[..., 0], pu[..., 1]
    y = c[:, None] * sig2(p)
    z = (m * p + s * u).requires_grad_(True)
    r = yhat_z(model, z[:, :N], y[:, :N], z[:, N]) - y[:, N]
    loss = (r * r).mean()
    C, = torch.autograd.grad(loss, z)
    zeta = torch.randn(D, generator=gen, dtype=w.dtype)
    zeta = zeta - (zeta @ v) * v
    zeta = zeta - (zeta @ e2) * e2
    g = (C * p).sum() * v + (C * u).sum() * e2 + torch.sqrt((C * C).sum()) * zeta
    return g, loss.detach()


def make_init(seed):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), D, 12]))
    v, w0 = init_fixed(D, rng, M0)
    return v, w0


def run(model, opt, N, B, seed, lr_adam=1e-3, trace_dir=None, max_steps=None, tag=""):
    t0 = time.process_time()
    v_np, w0_np = make_init(seed)
    v = torch.tensor(v_np, dtype=torch.float32)
    w = torch.tensor(w0_np, dtype=torch.float32, requires_grad=True)
    cell_code = (0 if model == "linear" else 1, 0 if opt == "sgd" else 1, N, B)
    gen = torch.Generator().manual_seed(int(np.random.SeedSequence([seed, 12, *cell_code]).generate_state(1)[0]))
    eta = 1.0 / D ** 2
    optim = torch.optim.Adam([w], lr=lr_adam) if opt == "adam" else None
    cap = max_steps or CAPS[opt]
    ts, ms = [0], [float((w.detach() @ v))]
    T05, reached, max_abs, step = -1, False, abs(ms[0]), 0
    dm_max = 0.0  # largest single-step |delta m| (instability diagnostic)
    m_prev = ms[0]
    while step < cap:
        if optim is None:  # SGD: projected sampler (distribution-exact; see proj_sgd_grad)
            g, _ = proj_sgd_grad(model, w.detach(), v, N, B, gen)
            with torch.no_grad():
                w -= eta * g
        else:
            xc, yc, xq, yq = sample(gen, v, N, B, torch.float32)
            loss = loss_fn(model, w, xc, yc, xq, yq)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        with torch.no_grad():
            w /= w.norm()
            m = float(w @ v)
        step += 1
        dm_max = max(dm_max, abs(m - m_prev))
        m_prev = m
        am = abs(m)
        if am > max_abs:
            max_abs = am
        if step % LOG_EVERY == 0:
            ts.append(step)
            ms.append(m)
        if am >= STOP_M:
            T05, reached = step, True
            if ts[-1] != step:
                ts.append(step)
                ms.append(m)
            break
    cpu = time.process_time() - t0
    name = f"{model}_{opt}{tag}_N{N}_B{B}_s{seed}"
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        np.savez_compressed(os.path.join(trace_dir, name + ".npz"), t=np.array(ts), m=np.array(ms, dtype=np.float32))
    return dict(model=model, opt=opt + tag, N=N, B=B, seed=seed, steps=step, T05=T05, reached05=int(reached),
                final_abs_m=abs(ms[-1]), max_abs_m=max_abs, max_dm_step=dm_max, cpu_s=cpu, name=name)


def sanity():
    """Torch linear+SGD batch loss and gradient vs ModelA(w, 2, gamma=1).loss on the same batch."""
    from .models import ModelA
    out = {}
    for dtype, label in ((torch.float64, "fp64"), (torch.float32, "fp32")):
        rels, grels = [], []
        for seed in range(5):
            v_np, w0_np = make_init(seed)
            gen = torch.Generator().manual_seed(1000 + seed)
            v = torch.tensor(v_np, dtype=dtype)
            w = torch.tensor(w0_np, dtype=dtype, requires_grad=True)
            for (N, B) in ((16, 64), (1024, 64)):
                xc, yc, xq, yq = sample(gen, v, N, B, dtype)
                lt = loss_fn("linear", w, xc, yc, xq, yq)
                gt, = torch.autograd.grad(lt, w)
                a = ModelA(w0_np, 2, gamma=1.0)
                xcn, ycn, xqn, yqn = (t.double().numpy() for t in (xc, yc, xq, yq))
                la = a.loss(xcn, ycn, xqn, yqn)
                _, ga, _ = a.loss_grad(xcn, ycn, xqn, yqn)
                rels.append(abs(float(lt.detach()) - la) / abs(la))
                grels.append(float(np.linalg.norm(gt.double().numpy() - ga) / np.linalg.norm(ga)))
        out[label] = dict(max_rel_loss=max(rels), max_rel_grad=max(grels))
    print(json.dumps(out, indent=1))
    return out


def sampler_check(reps=20000):
    """One SGD step from a fixed w (m = 0.3, so the drift is clearly non-zero): distribution of the new m under the
    full-data torch step vs the projected sampler.  Reports mean and std of Delta m for both (z-score of the mean difference)."""
    out = []
    v_np, w0_np = make_init(0)
    v = torch.tensor(v_np, dtype=torch.float32)
    u = torch.tensor(w0_np, dtype=torch.float32)
    u = u - (u @ v) * v
    u = u / u.norm()
    w = 0.3 * v + math.sqrt(1 - 0.09) * u
    eta = 1.0 / D ** 2
    for model in ("linear", "softmax"):
        for (N, B) in ((16, 64), (1024, 64)):
            n = reps if N == 16 else reps // 3
            res = {}
            for kind in ("full", "proj"):
                gen = torch.Generator().manual_seed(7 + (kind == "proj"))
                dms = []
                for _ in range(n):
                    if kind == "full":
                        wl = w.clone().requires_grad_(True)
                        xc, yc, xq, yq = sample(gen, v, N, B, torch.float32)
                        g, = torch.autograd.grad(loss_fn(model, wl, xc, yc, xq, yq), wl)
                    else:
                        g, _ = proj_sgd_grad(model, w, v, N, B, gen)
                    w2 = w - eta * g
                    w2 = w2 / w2.norm()
                    dms.append(float(w2 @ v) - 0.3)
                res[kind] = np.array(dms)
            a, b = res["full"], res["proj"]
            se = math.sqrt(a.var() / n + b.var() / n)
            out.append(dict(model=model, N=N, B=B, n=n, mean_full=a.mean(), mean_proj=b.mean(), std_full=a.std(), std_proj=b.std(),
                            z_mean=(a.mean() - b.mean()) / se, std_ratio=b.std() / a.std(),
                            iqr_ratio=float(np.subtract(*np.percentile(b, [75, 25])) / np.subtract(*np.percentile(a, [75, 25]))),
                            q99_ratio=float(np.percentile(np.abs(b), 99) / np.percentile(np.abs(a), 99))))
    print(json.dumps(out, indent=1))
    return out


def jobs(extra=False):
    js = []
    if extra:
        for model in ("linear", "softmax"):
            for seed in (0, 1):
                js.append((model, "adam", 16, 1024, seed, 1e-4, "_lr1e-4"))
        return js
    for N, B in ((16, 64), (16, 1024), (1024, 64)):
        for model in ("linear", "softmax"):
            for opt in ("sgd", "adam"):
                for seed in (0, 1):
                    js.append((model, opt, N, B, seed, 1e-3, ""))
    return js


def worker(k, extra=False):
    rows = os.path.join(OUT, "rows")
    claims = os.path.join(OUT, "claims")
    os.makedirs(rows, exist_ok=True)
    os.makedirs(claims, exist_ok=True)
    for (model, opt, N, B, seed, lr, tag) in jobs(extra):
        name = f"{model}_{opt}{tag}_N{N}_B{B}_s{seed}"
        try:
            fd = os.open(os.path.join(claims, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue
        print(f"[w{k}] start {name}", flush=True)
        r = run(model, opt, N, B, seed, lr_adam=lr, trace_dir=os.path.join(OUT, "traces"), tag=tag)
        r["lr_adam"] = lr if opt == "adam" else float("nan")
        with open(os.path.join(rows, name + ".json"), "w") as f:
            json.dump(r, f)
        print(f"[w{k}] done {name} steps={r['steps']} T05={r['T05']} max|m|={r['max_abs_m']:.3f} "
              f"max_dm={r['max_dm_step']:.3f} cpu={r['cpu_s']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--sampler-check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.sanity:
        sanity()
    elif a.sampler_check:
        sampler_check()
    elif a.smoke:
        for cfg in (("linear", "sgd", 16, 64), ("softmax", "adam", 16, 64), ("linear", "sgd", 1024, 64)):
            r = run(*cfg, 0, max_steps=300)
            print(r)
    elif a.worker is not None:
        worker(a.worker, a.extra)
