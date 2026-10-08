"""Data samplers.

Full samplers (`sample_prompts`, `sample_iw`) draw complete d-dimensional inputs.

The projected sampler (`sample_prompts_proj`, `sample_iw_proj`) is a distribution-exact shortcut used by the
long sweeps.  For a student w with m = <w,v> and unit e2 in span(v,w) orthogonal to v, any input decomposes as
x = p v + u e2 + xi with p, u ~ N(0,1) and xi ~ N(0, I_{d-2}) on the complement, all independent.  Labels and
activations depend only on (p, u): <v,x> = p, <w,x> = m p + sqrt(1-m^2) u.  The loss gradient is
sum_j C_j x_j with coefficients C_j that depend only on (p_j, u_j) (and the task scalar c), so its component
in the complement is exactly N(0, (sum_j C_j^2) I_{d-2}).  See train.py for how this is used.
"""
import numpy as np

from .hermite import sigma


def sample_prompts(d, N, B, v, k, task_std, rng):
    """B prompts. Returns x_ctx (B,N,d), y_ctx (B,N), x_q (B,d), y_q (B,); task c ~ N(0, task_std^2)."""
    x_ctx = rng.standard_normal((B, N, d))
    x_q = rng.standard_normal((B, d))
    c = task_std * rng.standard_normal(B)
    y_ctx = c[:, None] * sigma(k, x_ctx @ v)
    y_q = c * sigma(k, x_q @ v)
    return x_ctx, y_ctx, x_q, y_q


def sample_iw(d, B, v, k, rng, c=1.0):
    """B in-weight samples with the single fixed task c: x (B,d), y (B,)."""
    x = rng.standard_normal((B, d))
    return x, c * sigma(k, x @ v)


def sample_prompts_proj(N, B, k, task_std, rng):
    """Projected prompts: p_ctx,u_ctx (B,N), p_q,u_q (B,), c (B,)  (labels are c*sigma(p))."""
    pu = rng.standard_normal((B, N + 1, 2))
    c = task_std * rng.standard_normal(B)
    return pu[:, :N, 0], pu[:, :N, 1], pu[:, N, 0], pu[:, N, 1], c


def sample_iw_proj(B, k, rng, c=1.0):
    pu = rng.standard_normal((B, 2))
    return pu[:, 0], pu[:, 1]
