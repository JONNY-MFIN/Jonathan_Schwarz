"""Black-Scholes prices, delta and the lognormal terminal density.

All functions accept scalars or NumPy arrays and broadcast. Time is in years and
rates are continuously compounded. Volatility is floored at 1e-6 so a fitted
smile that touches zero still prices.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

SIGMA_FLOOR = 1e-6

FloatArray = NDArray[np.float64]


def _d1_d2(
    spot: ArrayLike, strike: ArrayLike, t: float, rate: float, sigma: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    vol = np.maximum(np.asarray(sigma, dtype=float), SIGMA_FLOOR)
    s, k = np.asarray(spot, dtype=float), np.asarray(strike, dtype=float)
    d1 = (np.log(s / k) + (rate + 0.5 * vol**2) * t) / (vol * np.sqrt(t))
    return d1, d1 - vol * np.sqrt(t), vol


def call_price(
    spot: ArrayLike, strike: ArrayLike, t: float, rate: float, sigma: ArrayLike
) -> FloatArray:
    d1, d2, _ = _d1_d2(spot, strike, t, rate, sigma)
    s, k = np.asarray(spot, dtype=float), np.asarray(strike, dtype=float)
    return np.asarray(s * norm.cdf(d1) - k * np.exp(-rate * t) * norm.cdf(d2), dtype=float)


def put_price(
    spot: ArrayLike, strike: ArrayLike, t: float, rate: float, sigma: ArrayLike
) -> FloatArray:
    d1, d2, _ = _d1_d2(spot, strike, t, rate, sigma)
    s, k = np.asarray(spot, dtype=float), np.asarray(strike, dtype=float)
    return np.asarray(k * np.exp(-rate * t) * norm.cdf(-d2) - s * norm.cdf(-d1), dtype=float)


def call_delta(
    spot: ArrayLike, strike: ArrayLike, t: float, rate: float, sigma: ArrayLike
) -> FloatArray:
    d1, _, _ = _d1_d2(spot, strike, t, rate, sigma)
    return np.asarray(norm.cdf(d1), dtype=float)


def lognormal_density(
    strike: ArrayLike, spot: float, t: float, rate: float, sigma: float
) -> FloatArray:
    """Risk-neutral density of S_T under Black-Scholes, evaluated at ``strike``.

    ln S_T ~ N(ln S + (r - sigma^2 / 2) T, sigma^2 T).
    """
    vol = max(float(sigma), SIGMA_FLOOR)
    k = np.asarray(strike, dtype=float)
    mu = np.log(spot) + (rate - 0.5 * vol**2) * t
    return np.asarray(
        np.exp(-((np.log(k) - mu) ** 2) / (2 * vol**2 * t)) / (k * vol * np.sqrt(2 * np.pi * t)),
        dtype=float,
    )
