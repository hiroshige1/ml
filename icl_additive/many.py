"""Experiment 2: many skills (P = 16 teachers, M = 64 neurons, pi_p ~ p^-alpha), Model A (in-context) and Model B (in-weight).

Model A is `multi.MultiModelA` (fixed readout gamma).  Inputs are sampled fully in d dimensions (the projected sampler of
exp 3 has no advantage here: its span 16 + 64 = 80 >= d = 32).  `grad_A_fast` is an algebraically *exact* rewrite of
`MultiModelA.loss_grad` for k = 2 that never forms the (B, N, M) activations: with
    G_b = (1/N) sum_n y_bn x_bn x_bn^T,   s_b = (1/N) sum_n y_bn,   sigma_2(z) = (z^2 - 1)/sqrt2
the context statistic of neuron m is A_bm = (w_m^T G_b w_m - s_b)/sqrt2 and
    dL/dw_m = sum_b (2 r_b / B) gamma [ sigma'(zq_bm) A_bm xq_b + sigma(zq_bm) sqrt2 G_b w_m ]         (test_many.py checks it).

Model B (in-weight): fixed task y = sum_p a_p sigma_2(<v_p,x>), yhat = rho sum_j sigma_2(<w_j,x>), loss = mean (yhat - y)^2 over a
batch of B fresh inputs, each w_j on the sphere.  Population MSE is exact:  E(yhat - y)^2 = || rho W^T W - sum_p a_p v_p v_p^T ||_F^2
(for symmetric A, E[(x^T A x - tr A)^2] = 2||A||_F^2 and sigma_2 carries the 1/sqrt2).
"""
import time

import numpy as np

from .hermite import dsigma, sigma
from .multi import MultiModelA, init_multi

SQ2 = np.sqrt(2.0)


def skill_freqs(P, alpha):
    pi = np.arange(1, P + 1, dtype=np.float64) ** (-alpha)
    return pi / pi.sum()


# ------------------------------------------------------------------------------------------------ Model A
def sample_prompts_full(d, N, B, V, pi, rng, k=2):
    """B single-skill prompts, x fully sampled.  Returns X (B,N+1,d) (last point = query), y (B,N+1), skill (B,)."""
    P = V.shape[0]
    X = rng.standard_normal((B, N + 1, d))
    skill = rng.choice(P, size=B, p=pi)
    c = rng.standard_normal(B)
    y = c[:, None] * sigma(k, np.einsum("bnd,bd->bn", X, V[skill]))
    return X, y, skill


def grad_A_fast(W, gamma, X, y):
    """Exact (loss, dL/dW) of MultiModelA (k=2) on prompts X (B,N+1,d), y (B,N+1); W (M,d), gamma (M,)."""
    B, N1, d = X.shape
    N = N1 - 1
    Xc, xq, yc, yq = X[:, :N], X[:, N], y[:, :N], y[:, N]
    G = np.matmul((Xc * yc[:, :, None]).transpose(0, 2, 1), Xc) / N  # (B,d,d)
    s = yc.mean(axis=1)
    GW = np.matmul(G, W.T)  # (B,d,M): G_b w_m
    A = ((GW * W.T[None]).sum(axis=1) - s[:, None]) / SQ2  # (B,M)
    zq = xq @ W.T  # (B,M)
    sq = (zq * zq - 1.0) / SQ2
    r = (gamma * sq * A).sum(axis=1) - yq
    f = 2.0 * r / B
    Cq = f[:, None] * gamma * (SQ2 * zq) * A  # (B,M)
    kap = f[:, None] * gamma * sq  # (B,M)
    gW = Cq.T @ xq + SQ2 * np.einsum("bm,bdm->md", kap, GW)
    return float(np.mean(r * r)), gW


# ------------------------------------------------------------------------------------------------ Model B
class MultiModelB:
    """W (M,d) on the unit sphere; yhat(x) = rho sum_j sigma_k(<w_j,x>); target y = sum_p a_p sigma_k(<v_p,x>)."""

    def __init__(self, W, V, a, k=2, rho=1.0):
        self.W = np.array(W, dtype=np.float64)
        self.V, self.a, self.k, self.rho = np.asarray(V), np.asarray(a), k, float(rho)
        self.M = self.W.shape[0]

    def target(self, x):
        return sigma(self.k, x @ self.V.T) @ self.a

    def forward(self, x):
        return self.rho * sigma(self.k, x @ self.W.T).sum(axis=1)

    def loss(self, x, y):
        r = self.forward(x) - y
        return float(np.mean(r * r))

    def loss_grad(self, x, y):
        z = x @ self.W.T
        r = self.rho * sigma(self.k, z).sum(axis=1) - y
        f = 2.0 * r / x.shape[0]
        return float(np.mean(r * r)), (f[:, None] * self.rho * dsigma(self.k, z)).T @ x

    def update(self, gW, eta):
        W = self.W - eta * gW
        self.W = W / np.linalg.norm(W, axis=1, keepdims=True)

    def pop_mse(self):
        """Exact population MSE (k = 2): || rho W^T W - V^T diag(a) V ||_F^2."""
        Dm = self.rho * (self.W.T @ self.W) - (self.V.T * self.a) @ self.V
        return float(np.sum(Dm * Dm))


# ------------------------------------------------------------------------------------------------ evaluation of A
class EvalSkills:
    """512 fixed contexts x (1 + H) queries; for every skill p an independent task scalar c_bp (same inputs for all skills).
    mse_spec = query 0 only (512 prompts per skill); mse_h = all 1+H queries (same contexts; lower variance)."""

    def __init__(self, V, d, N, n=512, H=16, eval_seed=20261009, k=2):
        self.V, self.N, self.n, self.k = V, N, n, k
        rng = np.random.default_rng(eval_seed)
        self.X = rng.standard_normal((n, N, d))
        self.Q = rng.standard_normal((n, 1 + H, d))
        self.c = rng.standard_normal((n, V.shape[0]))
        self.Sc = sigma(k, self.X @ V.T)  # (n,N,P)
        self.Sq = sigma(k, self.Q @ V.T)  # (n,1+H,P)

    def per_skill_mse(self, W, gamma):
        Sw = sigma(self.k, self.X @ W.T)  # (n,N,M)
        A = np.einsum("bnp,bnm->bpm", self.Sc, Sw) / self.N * self.c[:, :, None]
        Zq = sigma(self.k, self.Q @ W.T)  # (n,1+H,M)
        pred = np.einsum("bhm,bpm->bhp", Zq * gamma, A)
        err = pred - self.c[:, None, :] * self.Sq
        e2 = err * err
        return e2[:, 0].mean(axis=0), e2.mean(axis=(0, 1))


# ------------------------------------------------------------------------------------------------ training
MODEL_CODE = {"A": 1, "B": 2}


def train_many(model, seed, alpha=1.5, d=32, P=16, M=64, N=128, B=32, gamma=0.1, rho=1.0, k=2, eta=None,
               max_steps=3_000_000, log_every=1000, stop_skills=8, stop_m=0.95, verbose=False, n_eval=512):
    """One run of model 'A' or 'B'.  Same teachers V and initial neurons W0 for both models at a given (seed, alpha).
    Early stop: skills 1..stop_skills all have max_j |m_jp| >= stop_m (checked at log steps), else max_steps."""
    t0c, t0w = time.process_time(), time.time()
    eta = 1.0 / d ** 2 if eta is None else eta
    pi = skill_freqs(P, alpha)
    a = np.sqrt(pi)  # a_p ~ p^{-alpha/2}, sum a_p^2 = 1
    rng0 = np.random.Generator(np.random.SFC64(np.random.SeedSequence([int(seed), 2, int(round(alpha * 10)), d, P, M])))
    V, W0 = init_multi(d, P, M, rng0)
    rng = np.random.Generator(np.random.SFC64(np.random.SeedSequence([int(seed), 2, int(round(alpha * 10)), MODEL_CODE[model], 7])))
    if model == "A":
        mdl = MultiModelA(W0, k, gamma)
        ev = EvalSkills(V, d, N, n_eval, k=k)
        gam_vec = mdl.gamma
    else:
        mdl = MultiModelB(W0, V, a, k, rho)
    log = {"t": [], "m": [], "train_loss": [], "mse_spec": [], "mse_h": [], "mse_total": []}
    acc, nacc = 0.0, 0

    def do_log(t):
        m = mdl.W @ V.T
        log["t"].append(t)
        log["m"].append(m.astype(np.float32))
        log["train_loss"].append(acc / nacc if nacc else np.nan)
        if model == "A":
            ms, mh = ev.per_skill_mse(mdl.W, gam_vec)
            log["mse_spec"].append(ms)
            log["mse_h"].append(mh)
            log["mse_total"].append(float(pi @ mh))
        else:
            log["mse_spec"].append(np.full(P, np.nan))
            log["mse_h"].append(np.full(P, np.nan))
            log["mse_total"].append(mdl.pop_mse())
        return np.abs(m).max(axis=0)

    mx = do_log(0)
    t = 0
    stop_reason = "max_steps"
    while t < max_steps:
        t += 1
        if model == "A":
            X, y, _ = sample_prompts_full(d, N, B, V, pi, rng, k)
            loss, gW = grad_A_fast(mdl.W, gam_vec, X, y)
        else:
            x = rng.standard_normal((B, d))
            loss, gW = mdl.loss_grad(x, mdl.target(x))
        mdl.update(gW, eta)
        acc += loss
        nacc += 1
        if t % log_every == 0:
            mx = do_log(t)
            acc, nacc = 0.0, 0
            if verbose and (t // log_every) % 20 == 0:
                print(f"{model} seed {seed} t={t} learned(>=.5)={int((mx >= 0.5).sum())} first8 max|m|="
                      f"{np.round(mx[:8], 2).tolist()} cpu={time.process_time() - t0c:.0f}s", flush=True)
            if np.all(mx[:stop_skills] >= stop_m):
                stop_reason = "skills_learned"
                break
    traj = {key: np.array(v) for key, v in log.items()}
    return dict(model=model, seed=seed, alpha=alpha, d=d, P=P, M=M, N=N, B=B, gamma=gamma if model == "A" else float("nan"),
                rho=rho if model == "B" else float("nan"), eta=eta, steps=t, stop_reason=stop_reason, max_steps=max_steps,
                cpu_s=time.process_time() - t0c, wall_s=time.time() - t0w, traj=traj, V=V)
