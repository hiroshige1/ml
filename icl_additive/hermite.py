"""Normalized probabilists' Hermite polynomials sigma_k = He_k / sqrt(k!) and the centered relu.

All functions are vectorized numpy and keep the dtype of the input (float64 by default).
`k` is an int in {1, 2, 3} or the string "relu".
"""
import math

import numpy as np

_SQRT2 = math.sqrt(2.0)
_SQRT6 = math.sqrt(6.0)
_RELU_MEAN = 1.0 / math.sqrt(2.0 * math.pi)


def sigma(k, z):
    """Activation sigma_k(z); E[sigma_k(z)^2] = 1 for z ~ N(0,1) (relu: centered, not unit variance)."""
    if k == 1:
        return z
    if k == 2:
        return (z * z - 1.0) / _SQRT2
    if k == 3:
        return (z * z * z - 3.0 * z) / _SQRT6
    if k == "relu":
        return np.maximum(z, 0.0) - _RELU_MEAN
    raise ValueError(f"unsupported activation {k!r}")


def dsigma(k, z):
    """Derivative of sigma_k."""
    if k == 1:
        return np.ones_like(z)
    if k == 2:
        return _SQRT2 * z
    if k == 3:
        return (3.0 * z * z - 3.0) / _SQRT6
    if k == "relu":
        return (z > 0).astype(z.dtype)
    raise ValueError(f"unsupported activation {k!r}")


def hermite_norm(n, z):
    """Generic normalized He_n(z)/sqrt(n!) via the three-term recurrence (used for testing)."""
    z = np.asarray(z, dtype=np.float64)
    h_prev, h = np.ones_like(z), z.copy()
    if n == 0:
        return h_prev
    for j in range(1, n):
        h_prev, h = h, z * h - j * h_prev
    return h / math.sqrt(math.factorial(n))
