"""Experiment 13: distance-based (rbf) softmax attention in the single-feature model (torch, CPU, 1 thread).

Spec: docs/spec_exp13.md; pre-registration: docs/preregistration.md (P26a-b).
Model: w on S^{d-1}, pinned readout Gamma=1, sigma_2, s = sigma_2(w.x):
    yhat = Gamma * sum_i softmax_i(-beta (s_i - s_q)^2) y_i ,  beta = 0.3 ,  y_i = c sigma_2(v.x_i).
Loss = mean_b (yhat - y_q)^2.  w <- w/|w| after each step.  FULL d=64 data for both SGD and Adam.

Usage:
  python -m icl_additive.ablation_exp13 --sanity                 # Monte-Carlo drift check (m = 0.25, N = 1024, B = 4096)
  python -m icl_additive.ablation_exp13 --worker K               # dynamic job queue (claim files), N=1024 SGD first
Budget stop: touching results/exp13/STOP (all) or results/exp13/STOP_<run name> ends a run at the next 100-step boundary and
marks it censored_by=killed_<step>_cpu_budget.
"""
import argparse
import json
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)

from .train import init_fixed

D = 64
M0 = D ** -0.5
BETA = 0.3
GAMMA = 1.0
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp13")
SQRT2 = math.sqrt(2.0)
CAPS = {"sgd": 500_000, "adam": 100_000}
LOG_EVERY = 100
STOP_M = 0.5


def sig2(z):
    return (z * z - 1.0) / SQRT2


def yhat_z(zc, yc, zq):
    """rbf softmax from the projections z = w.x: Gamma * sum_i softmax_i(-beta (s_i - s_q)^2) y_i."""
    sc, sq = sig2(zc), sig2(zq)
    a = torch.softmax(-BETA * (sc - sq[:, None]) ** 2, dim=1)
    return GAMMA * (a * yc).sum(dim=1)


def yhat_fn(w, xc, yc, xq):
    """xc (B,N,d), yc (B,N), xq (B,d) -> yhat (B,)."""
    return yhat_z(xc @ w, yc, xq @ w)


def loss_fn(w, xc, yc, xq, yq):
    r = yhat_fn(w, xc, yc, xq) - yq
    return (r * r).mean()


def sample(gen, v, N, B, dtype):
    X = torch.randn(B, N + 1, D, generator=gen, dtype=dtype)
    c = torch.randn(B, generator=gen, dtype=dtype)
    y = c[:, None] * sig2(X @ v)
    return X[:, :N], y[:, :N], X[:, N], y[:, N]


def full_grad(w, v, N, B, gen):
    """Gradient of the batch loss wrt w on full d=64 data via the chain rule through z = X w (one matvec forward, one for the
    gradient; identical to autograd on loss_fn, checked in --sanity).  Returns (grad, loss)."""
    X = torch.randn(B, N + 1, D, generator=gen, dtype=w.dtype)
    c = torch.randn(B, generator=gen, dtype=w.dtype)
    Xf = X.reshape(-1, D)
    zp = (Xf @ torch.stack([w, v], dim=1)).reshape(B, N + 1, 2)
    y = c[:, None] * sig2(zp[..., 1])
    z = zp[..., 0].clone().requires_grad_(True)
    r = yhat_z(z[:, :N], y[:, :N], z[:, N]) - y[:, N]
    loss = (r * r).mean()
    C, = torch.autograd.grad(loss, z)
    g = C.reshape(1, -1) @ Xf
    return g.reshape(D), loss.detach()


def make_init(seed):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), D, 13]))
    v, w0 = init_fixed(D, rng, M0)
    return v, w0


def stop_requested(name):
    return os.path.exists(os.path.join(OUT, "STOP")) or os.path.exists(os.path.join(OUT, "STOP_" + name))


def run(opt, N, B, seed, lr_adam=1e-3, trace_dir=None, max_steps=None, log=None):
    t0 = time.process_time()
    v_np, w0_np = make_init(seed)
    v = torch.tensor(v_np, dtype=torch.float32)
    w = torch.tensor(w0_np, dtype=torch.float32)
    name = f"rbf_{opt}_N{N}_B{B}_s{seed}"
    gen = torch.Generator().manual_seed(int(np.random.SeedSequence([seed, 13, 0 if opt == "sgd" else 1, N, B]).generate_state(1)[0]))
    eta = 1.0 / D ** 2
    wp = torch.nn.Parameter(w.clone())
    optim = torch.optim.Adam([wp], lr=lr_adam) if opt == "adam" else None
    cap = max_steps or CAPS[opt]
    ts, ms = [0], [float(w @ v)]
    T05, reached, max_abs, step = -1, False, abs(ms[0]), 0
    dm_max, m_prev, censored_by = 0.0, ms[0], ""
    while step < cap:
        g, _ = full_grad(wp.detach(), v, N, B, gen)
        with torch.no_grad():
            if optim is None:
                wp -= eta * g
            else:
                wp.grad = g
                optim.step()
            wp /= wp.norm()
            m = float(wp @ v)
        step += 1
        dm_max = max(dm_max, abs(m - m_prev))
        m_prev = m
        am = abs(m)
        max_abs = max(max_abs, am)
        if step % LOG_EVERY == 0:
            ts.append(step)
            ms.append(m)
            if log and step % 10000 == 0:
                print(f"    {name} step={step} m={m:+.3f} max|m|={max_abs:.3f} cpu={time.process_time() - t0:.0f}s", file=log, flush=True)
            if stop_requested(name):
                censored_by = f"killed_{step}_cpu_budget"
                if ts[-1] != step:
                    ts.append(step)
                    ms.append(m)
                break
        if am >= STOP_M:
            T05, reached = step, True
            if ts[-1] != step:
                ts.append(step)
                ms.append(m)
            break
    if not reached and not censored_by:
        censored_by = "cap"
    cpu = time.process_time() - t0
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        np.savez_compressed(os.path.join(trace_dir, name + ".npz"), t=np.array(ts), m=np.array(ms, dtype=np.float32))
    return dict(model="rbf", opt=opt, N=N, B=B, seed=seed, steps=step, T05=T05, reached05=int(reached),
                final_abs_m=abs(ms[-1]), max_abs_m=max_abs, max_dm_step=dm_max, cpu_s=cpu, censored_by=censored_by,
                lr_adam=lr_adam if opt == "adam" else float("nan"), name=name)


def mc_drift(m, N, B, reps, seed, chunk=1024):
    """Population drift -dL/dm of the torch model (v = e0, w = m e0 + sqrt(1-m^2) e1), gradient projected as in
    scripts/softmax_drift.py: t = (v - m w)/sqrt(1-m^2), drift = -(g.t)/sqrt(1-m^2).  B is split in chunks (memory); the batch
    gradient is the average of chunk gradients.  Returns (mean, se, per-rep values)."""
    gen = torch.Generator().manual_seed(seed)
    v = torch.zeros(D); v[0] = 1.0
    u = torch.zeros(D); u[1] = 1.0
    s = math.sqrt(1 - m * m)
    w = (m * v + s * u).requires_grad_(True)
    t = (v - m * w.detach()) / s
    vals = []
    for _ in range(reps):
        gsum = torch.zeros(D)
        for _c in range(B // chunk):
            xc, yc, xq, yq = sample(gen, v, N, chunk, torch.float32)
            gc, = torch.autograd.grad(loss_fn(w, xc, yc, xq, yq), w)
            gsum += gc * (chunk / B)
        vals.append(-float(gsum @ t) / s)
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(reps)), vals


def sanity(reps=48, seed=0):
    out = {}
    # (1) hand-written chain-rule gradient == autograd on the same full-data batch
    v_np, w0_np = make_init(0)
    v = torch.tensor(v_np, dtype=torch.float32)
    w = torch.tensor(w0_np, dtype=torch.float32, requires_grad=True)
    rel = []
    for N, B in ((16, 64), (1024, 64)):
        g1 = torch.Generator().manual_seed(5)
        g2 = torch.Generator().manual_seed(5)
        g_hand, l_hand = full_grad(w.detach(), v, N, B, g1)
        X = torch.randn(B, N + 1, D, generator=g2)
        c = torch.randn(B, generator=g2)
        y = c[:, None] * sig2(X @ v)
        l_auto = loss_fn(w, X[:, :N], y[:, :N], X[:, N], y[:, N])
        g_auto, = torch.autograd.grad(l_auto, w)
        rel.append(float((g_hand - g_auto).norm() / g_auto.norm()))
    out["grad_hand_vs_autograd_rel"] = rel
    # (2) Monte-Carlo drift at m = 0.25, N = 1024, B = 4096
    t0 = time.process_time()
    mean, se, vals = mc_drift(0.25, 1024, 4096, reps, seed)
    ref, ref_se = 3.25e-2, 4e-3
    comb = math.sqrt(se ** 2 + ref_se ** 2)
    out["drift_m0.25_N1024_B4096"] = dict(mean=mean, se=se, reps=reps, ref=ref, ref_se=ref_se, diff=mean - ref, diff_over_combined_se=(mean - ref) / comb,
                                           positive=mean > 0, within_2se=abs(mean - ref) <= 2 * comb, cpu_s=time.process_time() - t0)
    out["pass"] = bool(mean > 0 and abs(mean - ref) <= 2 * comb)
    print(json.dumps(out, indent=1))
    return out


def jobs():
    js = []
    for opt, N, B in (("sgd", 1024, 64), ("adam", 1024, 64), ("sgd", 16, 1024), ("adam", 16, 1024), ("sgd", 16, 64), ("adam", 16, 64)):
        for seed in (0, 1):
            js.append((opt, N, B, seed))
    return js


def worker(k):
    rows = os.path.join(OUT, "rows")
    claims = os.path.join(OUT, "claims")
    os.makedirs(rows, exist_ok=True)
    os.makedirs(claims, exist_ok=True)
    import sys
    for (opt, N, B, seed) in jobs():
        name = f"rbf_{opt}_N{N}_B{B}_s{seed}"
        try:
            fd = os.open(os.path.join(claims, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue
        print(f"[w{k}] start {name}", flush=True)
        r = run(opt, N, B, seed, trace_dir=os.path.join(OUT, "traces"), log=sys.stdout)
        with open(os.path.join(rows, name + ".json"), "w") as f:
            json.dump(r, f)
        print(f"[w{k}] done {name} steps={r['steps']} T05={r['T05']} max|m|={r['max_abs_m']:.3f} "
              f"max_dm={r['max_dm_step']:.3f} cpu={r['cpu_s']:.0f}s censored_by={r['censored_by']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reps", type=int, default=48)
    a = ap.parse_args()
    if a.sanity:
        res = sanity(a.reps)
        os.makedirs(OUT, exist_ok=True)
        json.dump(res, open(os.path.join(OUT, "sanity.json"), "w"), indent=1)
    elif a.smoke:
        for cfg in (("sgd", 16, 64), ("adam", 16, 64), ("sgd", 1024, 64), ("adam", 1024, 64)):
            print(run(*cfg[:3], 0, max_steps=200))
    elif a.worker is not None:
        worker(a.worker)
