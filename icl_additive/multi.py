"""Experiment 3: multi-neuron, multi-teacher Model A (M neurons, P teachers, fixed readout Gamma_j), float64.

yhat_q = sum_j Gamma_j sigma_k(<w_j,x_q>) A_j,   A_j = (1/N) sum_i y_i sigma_k(<w_j,x_i>),   each w_j on the sphere.
Loss = mean over the batch of (yhat - y)^2.  Pretraining prompts are single-skill: skill s ~ pi, c ~ N(0,1),
y = c sigma_k(<v_s,x>).

Projected sampler (exact).  Let Q (d x D) be ANY orthonormal matrix whose columns span [v_1..v_P, w_1..w_M]
(D = P+M; Householder QR is orthonormal even when the columns are nearly dependent) and R = Q^T [V W] (D x (P+M)).
Then x = Q y + x_perp with y ~ N(0, I_D) and x_perp ~ N(0, I - QQ^T) independent, labels and activations depend on
y only (<v,x> = y.R[:,v], <w,x> = y.R[:,w]) and the gradient of neuron j is
    g_j = Q (sum_n C^j_n y_n) + P_perp (sum_n C^j_n xi_n)
with C^j_n the per-point coefficients (functions of y only) and xi_n iid N(0, I_d).  Given the C's the second term is
jointly Gaussian across neurons with covariance S_jk = sum_n C^j_n C^k_n times P_perp, so it is sampled with
an (M x M) factor of S: exact in distribution, D instead of d coordinates per point.  Tested against full sampling.
"""
import time

import numpy as np

from .hermite import dsigma, sigma


class MultiModelA:
    """W (M,d) rows on the unit sphere, fixed readout gamma (scalar or (M,)), activation sigma_k."""

    def __init__(self, W, k=2, gamma=0.1):
        self.W = np.array(W, dtype=np.float64)
        self.k = k
        self.M = self.W.shape[0]
        self.gamma = np.broadcast_to(np.asarray(gamma, dtype=np.float64), (self.M,)).copy()

    def forward(self, x_ctx, y_ctx, x_q):
        return self._yhat(x_ctx @ self.W.T, y_ctx, x_q @ self.W.T)[0]

    def _yhat(self, Z_ctx, y_ctx, Z_q):
        """Z_ctx (B,N,M), y_ctx (B,N), Z_q (B,M) -> yhat (B,), A (B,M)."""
        A = np.matmul(y_ctx[:, None, :], sigma(self.k, Z_ctx))[:, 0, :] / Z_ctx.shape[1]
        return (sigma(self.k, Z_q) * A) @ self.gamma, A

    def coefs(self, Z_ctx, y_ctx, Z_q, y_q):
        """loss, C_ctx (B,N,M), C_q (B,M) with dL/dw_j = sum_{b,n} C_ctx[b,n,j] x_ctx[b,n] + sum_b C_q[b,j] x_q[b]."""
        B, N, M = Z_ctx.shape
        k = self.k
        yhat, A = self._yhat(Z_ctx, y_ctx, Z_q)
        r = yhat - y_q
        f = 2.0 * r / B
        sq = sigma(k, Z_q)
        C_q = f[:, None] * self.gamma * dsigma(k, Z_q) * A
        C_ctx = ((f[:, None] * self.gamma * sq)[:, None, :] * y_ctx[:, :, None] * dsigma(k, Z_ctx)) / N
        return float(np.mean(r * r)), C_ctx, C_q

    def loss(self, x_ctx, y_ctx, x_q, y_q):
        r = self.forward(x_ctx, y_ctx, x_q) - y_q
        return float(np.mean(r * r))

    def loss_grad(self, x_ctx, y_ctx, x_q, y_q):
        loss, C_ctx, C_q = self.coefs(x_ctx @ self.W.T, y_ctx, x_q @ self.W.T, y_q)
        gW = np.einsum("bnm,bnd->md", C_ctx, x_ctx) + C_q.T @ x_q
        return loss, gW

    def update(self, gW, eta):
        W = self.W - eta * gW
        self.W = W / np.linalg.norm(W, axis=1, keepdims=True)  # per-neuron spherical renormalisation


def teacher_labels(k, x, V, c):
    """y = sum_p c_p sigma_k(<v_p,x>);  x (...,d), V (P,d), c (...,P) -> (...)."""
    return np.sum(c * sigma(k, x @ V.T), axis=-1)


def sample_prompts_multi(d, N, B, V, pi, rng, k=2, task_std=1.0):
    """B single-skill prompts. Returns x_ctx (B,N,d), y_ctx (B,N), x_q (B,d), y_q (B,), skill (B,)."""
    P = V.shape[0]
    x_ctx = rng.standard_normal((B, N, d))
    x_q = rng.standard_normal((B, d))
    skill = rng.choice(P, size=B, p=pi)
    c = np.zeros((B, P))
    c[np.arange(B), skill] = task_std * rng.standard_normal(B)
    return x_ctx, teacher_labels(k, x_ctx, V, c[:, None, :]), x_q, teacher_labels(k, x_q, V, c), skill


def init_multi(d, P, M, rng):
    """Orthonormal teachers V (P,d) (QR of a Gaussian matrix) and M independent uniform unit neurons W (M,d)."""
    Q, _ = np.linalg.qr(rng.standard_normal((d, P)))
    W = rng.standard_normal((M, d))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return Q.T.copy(), W


def proj_basis(V, W):
    """Q (d,D), R (D,P+M) with [V;W]^T = Q R and Q orthonormal (D = P+M)."""
    return np.linalg.qr(np.concatenate([V, W], axis=0).T)


def proj_step_stats(model, R, P, Y, skill, c, k=2):
    """Gradient statistics from span coordinates Y (B,N+1,D) of a batch (last point of each prompt = query).
    Returns loss, Gy (M,D) = span-part coefficients of the gradient (g_j = Q Gy_j + noise), S (M,M) = noise covariance."""
    B, N1, D = Y.shape
    N = N1 - 1
    T = Y @ R  # (B,N+1,P+M): <v_p,x>, <w_j,x>
    pt = sigma(k, T[:, :, :P])
    lab = np.take_along_axis(pt, skill[:, None, None], axis=2)[:, :, 0] * c[:, None]
    Z = T[:, :, P:]
    loss, C_ctx, C_q = model.coefs(Z[:, :N], lab[:, :N], Z[:, N], lab[:, N])
    C = np.concatenate([C_ctx, C_q[:, None, :]], axis=1).reshape(B * N1, -1)  # all N+1 points, (B(N+1), M)
    Gy = C.T @ Y.reshape(B * N1, D)
    return loss, Gy, C.T @ C


def sample_noise(S, Q, rng, d):
    """(M,d) Gaussian rows with covariance S_jk * (I - QQ^T)."""
    lam, U = np.linalg.eigh(S)
    L = U * np.sqrt(np.maximum(lam, 0.0))
    Xi = rng.standard_normal((S.shape[0], d))
    Xi -= (Xi @ Q) @ Q.T
    return L @ Xi


# ----------------------------------------------------------------------------- evaluation
class EvalSets:
    """Fixed evaluation prompts (n contexts of N points each; query 0 of each context is THE spec query, so the `mse_*`,
    `acc*_*` statistics are exactly 'n prompts').  Queries 1..H are extra independent queries on the same contexts,
    used only for the lower-variance `hmse_*`/`hacc05_*` statistics (a prompt has one query, so the spec statistics have
    sampling SE ~ 0.1 on a MSE; the extra queries are cheap because A_j is computed once per context).

    Set A holds the inputs shared by E1, E2, Ex and the paired E12A (independent task scalars c_1, c_2, c_x; E12A uses
    E1's c_1 and E2's c_2, so its error is e_1 + e_2 prompt by prompt); set B (own inputs, own c_1, c_2) is the
    independent additive-pair set E12 of the spec.  Input draws use a fixed evaluation seed, identical for every
    training seed; only the teachers differ."""

    def __init__(self, V, d, N, n=4096, eval_seed=20261008, k=2, H=16):
        self.V, self.N, self.n, self.k, self.H = V, N, n, k, H
        rng = np.random.default_rng(eval_seed)
        self.XA = rng.standard_normal((n, N, d))
        self.QA = rng.standard_normal((n, 1 + H, d))
        self.XB = rng.standard_normal((n, N, d))
        self.QB = rng.standard_normal((n, 1 + H, d))
        cA = rng.standard_normal((n, 3))
        cB = rng.standard_normal((n, 2))
        zA, zQA = self.XA @ V.T, self.QA @ V.T
        SA, SQA = sigma(k, zA), sigma(k, zQA)
        SB, SQB = sigma(k, self.XB @ V.T), sigma(k, self.QB @ V.T)
        c1, c2, cx = cA[:, 0, None], cA[:, 1, None], cA[:, 2, None]
        self.A = {  # name -> (ctx labels (n,N), query labels (n,1+H))
            "E1": (c1 * SA[:, :, 0], c1 * SQA[:, :, 0]),
            "E2": (c2 * SA[:, :, 1], c2 * SQA[:, :, 1]),
            "E12A": (c1 * SA[:, :, 0] + c2 * SA[:, :, 1], c1 * SQA[:, :, 0] + c2 * SQA[:, :, 1]),
            "Ex": (cx * zA[:, :, 0] * zA[:, :, 1], cx * zQA[:, :, 0] * zQA[:, :, 1]),  # c s1(v1.x) s1(v2.x), E[y^2]=1
        }
        b1, b2 = cB[:, 0, None], cB[:, 1, None]
        self.B = {"E12": (b1 * SB[:, :, 0] + b2 * SB[:, :, 1], b1 * SQB[:, :, 0] + b2 * SQB[:, :, 1])}
        self.Ey2 = {"E1": 1.0, "E2": 1.0, "E12": 2.0, "E12A": 2.0, "Ex": 1.0}

    def _resid(self, Xc, Xq, labs, model):
        Zc, Zq = sigma(self.k, Xc @ model.W.T), sigma(self.k, Xq @ model.W.T)
        out = {}
        for key, (lc, lq) in labs.items():
            A = np.matmul(lc[:, None, :], Zc)[:, 0, :] / self.N  # (n,M)
            out[key] = (Zq * A[:, None, :]) @ model.gamma - lq  # (n,1+H)
        return out

    def evaluate(self, model):
        """dict: mse_<set> / acc05_<set> (|err|^2 < 0.5) / accrel_<set> (< 0.5 E[y^2]) from the spec query (n prompts);
        hmse_<set>, hacc05_<set>, haccrel_<set> from all 1+H queries per context."""
        res = self._resid(self.XA, self.QA, self.A, model)
        res.update(self._resid(self.XB, self.QB, self.B, model))
        out = {}
        for key, r in res.items():
            r2 = r * r
            out["mse_" + key] = float(r2[:, 0].mean())
            out["acc05_" + key] = float((r2[:, 0] < 0.5).mean())
            out["accrel_" + key] = float((r2[:, 0] < 0.5 * self.Ey2[key]).mean())
            out["hmse_" + key] = float(r2.mean())
            out["hacc05_" + key] = float((r2 < 0.5).mean())
            out["haccrel_" + key] = float((r2 < 0.5 * self.Ey2[key]).mean())
        return out


EVAL_KEYS = ["E1", "E2", "E12", "E12A", "Ex"]
STATS = ["mse", "acc05", "accrel", "hmse", "hacc05", "haccrel"]


def train_multi(seed, d=32, P=2, M=4, N=128, B=32, gamma=0.1, pi=(0.75, 0.25), k=2, eta=None, max_steps=1_500_000,
                log_every=500, stop_mse=0.1, stop_align=None, align_hold=0, n_eval=4096, verbose=False):
    """One run.  Stops early when MSE_E1 and MSE_E2 < stop_mse (spec rule; unreachable when M*gamma is small, see README)
    or, if stop_align is given, once every neuron has |m_jp| >= stop_align for some p, continuously for `align_hold` steps
    (the state is then settled: sign-symmetric spherical dynamics with all neurons at a teacher).
    Returns dict with scalars and 'traj' arrays."""
    t0c, t0w = time.process_time(), time.time()
    eta = 1.0 / d ** 2 if eta is None else eta
    pi = np.asarray(pi, dtype=np.float64)
    rng = np.random.Generator(np.random.SFC64(np.random.SeedSequence([int(seed), 3, d, P, M])))
    V, W0 = init_multi(d, P, M, rng)
    model = MultiModelA(W0, k, gamma)
    ev = EvalSets(V, d, N, n_eval, k=k)
    D = P + M
    log = {key: [] for key in ["t", "m", "train_loss"] + [f"{a}_{e}" for e in EVAL_KEYS for a in STATS]}
    acc_loss, n_acc = 0.0, 0

    def do_log(t):
        r = ev.evaluate(model)
        log["t"].append(t)
        log["m"].append(model.W @ V.T)
        log["train_loss"].append(acc_loss / n_acc if n_acc else float("nan"))
        for e in EVAL_KEYS:
            for a in STATS:
                log[f"{a}_{e}"].append(r[f"{a}_{e}"])
        return r

    do_log(0)
    t = 0
    stop_reason, align_reached_at = "max_steps", None
    while t < max_steps:
        t += 1
        Q, R = proj_basis(V, model.W)
        skill = (rng.random(B) < pi[1]).astype(np.int64) if P == 2 else rng.choice(P, size=B, p=pi)
        c = rng.standard_normal(B)
        Y = rng.standard_normal((B, N + 1, D))
        loss, Gy, S = proj_step_stats(model, R, P, Y, skill, c, k)
        gW = Gy @ Q.T + sample_noise(S, Q, rng, d)
        model.update(gW, eta)
        acc_loss += loss
        n_acc += 1
        if t % log_every == 0:
            r = do_log(t)
            acc_loss, n_acc = 0.0, 0
            if verbose and (t // log_every) % 20 == 0:
                mm = np.abs(model.W @ V.T)
                print(f"seed {seed} t={t} E1={r['mse_E1']:.3f} E2={r['mse_E2']:.3f} E12={r['mse_E12']:.3f} "
                      f"Ex={r['mse_Ex']:.3f} |m|={np.round(mm, 2).tolist()} cpu={time.process_time() - t0c:.0f}s", flush=True)
            if r["mse_E1"] < stop_mse and r["mse_E2"] < stop_mse:
                stop_reason = "mse"
                break
            if stop_align is not None:  # "settled": every neuron has some teacher with |m| >= stop_align, for align_hold steps
                if np.all(np.abs(model.W @ V.T).max(axis=1) >= stop_align):
                    if align_reached_at is None:
                        align_reached_at = t
                    if t - align_reached_at >= align_hold:
                        stop_reason = "settled"
                        break
                else:
                    align_reached_at = None
    traj = {key: np.array(val) for key, val in log.items()}
    return dict(seed=seed, d=d, P=P, M=M, N=N, B=B, gamma=gamma, eta=eta, pi=list(pi), steps=t, stop_reason=stop_reason,
                max_steps=max_steps, cpu_s=time.process_time() - t0c, wall_s=time.time() - t0w, traj=traj)
