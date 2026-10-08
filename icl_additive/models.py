"""Models with M = 1 student neuron, weight w on the unit sphere, analytic gradients, float64.

Loss is the mean over the batch of (yhat - y)^2.  Every model exposes
  w, forward(...), loss(...), loss_grad(...)           (full d-dim inputs)
  coefs(...)  -> loss and the per-point gradient coefficients (grad_w = sum_j C_j x_j), used by the
                 projected sampler in train.py and by the full-input path alike.
"""
import numpy as np

from .hermite import dsigma, sigma


class ModelA:
    """In-context model: yhat_q = Gamma * sigma(<w,x_q>) * (1/N) sum_i y_i sigma(<w,x_i>)."""

    name = "A"

    def __init__(self, w, k, gamma=0.1, train_gamma=False):
        self.w = np.array(w, dtype=np.float64)
        self.k = k
        self.gamma = float(gamma)
        self.train_gamma = train_gamma

    def forward(self, x_ctx, y_ctx, x_q):
        return self._yhat(x_ctx @ self.w, y_ctx, x_q @ self.w)[0]

    def _yhat(self, z_ctx, y_ctx, z_q):
        A = (y_ctx * sigma(self.k, z_ctx)).mean(axis=1)
        return self.gamma * sigma(self.k, z_q) * A, A

    def coefs(self, z_ctx, y_ctx, z_q, y_q):
        """Return loss, C_ctx (B,N), C_q (B,), dL/dGamma with dL/dw = sum_b,j C_bj x_bj."""
        B, N = z_ctx.shape
        k, G = self.k, self.gamma
        sq = sigma(k, z_q)
        yhat, A = self._yhat(z_ctx, y_ctx, z_q)
        r = yhat - y_q
        f = 2.0 * r / B
        C_q = f * G * dsigma(k, z_q) * A
        C_ctx = (f * G * sq)[:, None] * y_ctx * dsigma(k, z_ctx) / N
        return float(np.mean(r * r)), C_ctx, C_q, float(np.sum(f * sq * A))

    def loss(self, x_ctx, y_ctx, x_q, y_q):
        r = self.forward(x_ctx, y_ctx, x_q) - y_q
        return float(np.mean(r * r))

    def loss_grad(self, x_ctx, y_ctx, x_q, y_q):
        loss, C_ctx, C_q, gG = self.coefs(x_ctx @ self.w, y_ctx, x_q @ self.w, y_q)
        gw = np.einsum("bn,bnd->d", C_ctx, x_ctx) + C_q @ x_q
        return loss, gw, gG

    def update(self, gw, gG, eta):
        w = self.w - eta * gw
        self.w = w / np.linalg.norm(w)
        if self.train_gamma:
            self.gamma -= eta * gG


class ModelB:
    """In-weight baseline: yhat = a * sigma(<w,x>) with a single fixed task c = 1."""

    name = "B"

    def __init__(self, w, k, a=1.0, train_a=False):
        self.w = np.array(w, dtype=np.float64)
        self.k = k
        self.a = float(a)
        self.train_a = train_a

    def forward(self, x):
        return self.a * sigma(self.k, x @ self.w)

    def coefs(self, z, y):
        B = z.shape[0]
        s = sigma(self.k, z)
        r = self.a * s - y
        f = 2.0 * r / B
        return float(np.mean(r * r)), f * self.a * dsigma(self.k, z), float(np.sum(f * s))

    def loss(self, x, y):
        r = self.forward(x) - y
        return float(np.mean(r * r))

    def loss_grad(self, x, y):
        loss, C, ga = self.coefs(x @ self.w, y)
        return loss, C @ x, ga

    def update(self, gw, ga, eta):
        w = self.w - eta * gw
        self.w = w / np.linalg.norm(w)
        if self.train_a:
            self.a -= eta * ga
