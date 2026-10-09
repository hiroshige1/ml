"""Exp 2 add-on: "B-tied", the fair decoupled in-weight baseline (Ren et al. 2025 style 2-homogeneous student).

    f(x) = sum_j |u_j|^2 sigma_2(uhat_j . x),   uhat_j = u_j / |u_j|,   target y = sum_p a_p sigma_2(v_p . x),  a_p = sqrt(pi_p).

Plain online SGD on the unconstrained u (no projection / renormalisation), small init |u_j|^2 = rho0.  Gradient (n_j = |u_j|^2):
    dL/dn_j   = sum_b (2 r_b / B) sigma(z_bj)                    (z = uhat . x)
    dL/duhat_j = n_j sum_b (2 r_b / B) sigma'(z_bj) x_b
    dL/du_j   = (I - uhat uhat^T) dL/duhat_j / |u_j|  +  2 u_j dL/dn_j
Population MSE is exact: || sum_j n_j uhat_j uhat_j^T - sum_p a_p v_p v_p^T ||_F^2 (same derivation as MultiModelB.pop_mse).
Same V and same initial directions as exp 2 at the same seed (init_multi with the exp-2 rng0 seed sequence), alpha = 1.5.
"""
import time

import numpy as np

from .hermite import dsigma, sigma
from .many import skill_freqs
from .multi import init_multi


class BTied:
    def __init__(self, U, V, a, k=2):
        self.U = np.array(U, dtype=np.float64)
        self.V, self.a, self.k = np.asarray(V), np.asarray(a), k

    @property
    def n(self):
        return np.einsum("jd,jd->j", self.U, self.U)

    @property
    def Uhat(self):
        return self.U / np.sqrt(self.n)[:, None]

    def target(self, x):
        return sigma(self.k, x @ self.V.T) @ self.a

    def forward(self, x):
        n = self.n
        return sigma(self.k, x @ self.Uhat.T) @ n

    def loss(self, x, y):
        r = self.forward(x) - y
        return float(np.mean(r * r))

    def loss_grad(self, x, y):
        n = self.n
        nrm = np.sqrt(n)
        Uh = self.U / nrm[:, None]
        z = x @ Uh.T  # (B,M)
        r = sigma(self.k, z) @ n - y
        f = 2.0 * r / x.shape[0]
        g_n = f @ sigma(self.k, z)  # (M,) dL/dn_j
        g_hat = n[:, None] * ((f[:, None] * dsigma(self.k, z)).T @ x)  # (M,d) dL/duhat_j
        radial = np.einsum("jd,jd->j", g_hat, Uh)
        g_perp = g_hat - radial[:, None] * Uh
        return float(np.mean(r * r)), g_perp / nrm[:, None] + 2.0 * self.U * g_n[:, None]

    def pop_mse(self):
        Uh = self.Uhat
        D = (Uh.T * self.n) @ Uh - (self.V.T * self.a) @ self.V
        return float(np.sum(D * D))


def train_btied(seed, eta=None, alpha=1.5, d=32, P=16, M=64, B=32, rho0=0.01, k=2, max_steps=3_000_000, log_every=1000, fine_every=100, fine_until=30_000, verbose=False):
    t0c, t0w = time.process_time(), time.time()
    eta = 1.0 / d ** 2 if eta is None else eta
    pi = skill_freqs(P, alpha)
    a = np.sqrt(pi)
    rng0 = np.random.Generator(np.random.SFC64(np.random.SeedSequence([int(seed), 2, int(round(alpha * 10)), d, P, M])))
    V, W0 = init_multi(d, P, M, rng0)  # identical V, W0 to exp-2 Models A / B at this (seed, alpha)
    rng = np.random.Generator(np.random.SFC64(np.random.SeedSequence([int(seed), 2, int(round(alpha * 10)), 3, 7])))
    mdl = BTied(np.sqrt(rho0) * W0, V, a, k)
    log = {"t": [], "m": [], "norm2": [], "mse_total": [], "train_loss": [], "L_align": []}
    acc, nacc = 0.0, 0
    fine = {"fine_t": [], "fine_m": [], "fine_norm2": [], "fine_mse": []}  # supplementary dense early log (the 1000-step log is primary)

    def do_fine(t):
        m = mdl.Uhat @ V.T
        fine["fine_t"].append(t); fine["fine_m"].append(m.astype(np.float32)); fine["fine_norm2"].append(mdl.n.copy()); fine["fine_mse"].append(mdl.pop_mse())

    def do_log(t):
        m = mdl.Uhat @ V.T
        log["t"].append(t); log["m"].append(m.astype(np.float32)); log["norm2"].append(mdl.n.copy())
        log["mse_total"].append(mdl.pop_mse()); log["train_loss"].append(acc / nacc if nacc else np.nan)
        log["L_align"].append(float(pi @ (1.0 - (m ** 2).max(axis=0))))
        return np.abs(m).max(axis=0)

    do_log(0)
    do_fine(0)
    for t in range(1, max_steps + 1):
        x = rng.standard_normal((B, d))
        loss, g = mdl.loss_grad(x, mdl.target(x))
        mdl.U -= eta * g
        acc += loss; nacc += 1
        if t <= fine_until and t % fine_every == 0:
            do_fine(t)
        if t % log_every == 0:
            mx = do_log(t)
            acc, nacc = 0.0, 0
            if verbose and (t // log_every) % 100 == 0:
                print(f"seed {seed} eta*d^2={eta * d * d:g} t={t} mse={log['mse_total'][-1]:.4g} sum n={mdl.n.sum():.3f} "
                      f"n>=.5: {int((mx >= .5).sum())} first8 mx={np.round(mx[:8], 2).tolist()} cpu={time.process_time() - t0c:.0f}s", flush=True)
    traj = {key: np.array(v) for key, v in {**log, **fine}.items()}
    traj["Uhat_init"], traj["Uhat_end"] = W0.copy(), mdl.Uhat.copy()
    return dict(seed=seed, eta=eta, alpha=alpha, steps=max_steps, cpu_s=time.process_time() - t0c, wall_s=time.time() - t0w, traj=traj, V=V)
