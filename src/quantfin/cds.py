"""CDS pricing and bootstrapping with a piecewise-constant default intensity.

The intensity is constant between curve knots: ``intensities[i]`` applies on
``(knots[i-1], knots[i]]`` and the last one extends beyond the final knot.
Premiums are paid at the end of each period, and premium accrued since the
last payment date is paid on default, approximated as half a period.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class CDSMarket:
    """Flat risk-free rate (continuously compounded), recovery and premium period."""

    rate: float
    recovery: float
    period: float = 0.25

    def discount(self, t: float) -> float:
        return float(np.exp(-self.rate * t))

    def payment_times(self, maturity: float) -> list[float]:
        n = round(maturity / self.period)
        return [self.period * i for i in range(1, n + 1)]


@dataclass(frozen=True)
class HazardCurve:
    """Piecewise-constant default intensity."""

    knots: tuple[float, ...]
    intensities: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.knots) != len(self.intensities) or not self.knots:
            raise ValueError("knots and intensities must be non-empty and of equal length")
        if any(b <= a for a, b in pairwise(self.knots)):
            raise ValueError("knots must be strictly increasing")

    def survival(self, t: float) -> float:
        """P_S(t) = exp(-integral_0^t lambda(s) ds)."""
        integral, start = 0.0, 0.0
        for i, lam in enumerate(self.intensities):
            end = self.knots[i] if i < len(self.knots) - 1 else np.inf
            if t <= start:
                break
            integral += lam * (min(t, end) - start)
            start = end
        return float(np.exp(-integral))


def rpv01(curve: HazardCurve, market: CDSMarket, maturity: float) -> float:
    """Risky PV01: discounted premium per unit spread, including accrual on default."""
    total = 0.0
    for t in market.payment_times(maturity):
        ps_t, ps_prev = curve.survival(t), curve.survival(t - market.period)
        total += market.discount(t) * market.period * (ps_t + 0.5 * (ps_prev - ps_t))
    return total


def default_leg(curve: HazardCurve, market: CDSMarket, maturity: float) -> float:
    """PV of the protection leg: (1 - R) * sum DF(t_i) * [P_S(t_{i-1}) - P_S(t_i)]."""
    total = 0.0
    for t in market.payment_times(maturity):
        total += market.discount(t) * (curve.survival(t - market.period) - curve.survival(t))
    return (1.0 - market.recovery) * total


def fair_spread(curve: HazardCurve, market: CDSMarket, maturity: float) -> float:
    """Spread that sets the premium and protection legs equal."""
    return default_leg(curve, market, maturity) / rpv01(curve, market, maturity)


def bootstrap(
    maturities: Sequence[float],
    spreads: Sequence[float],
    market: CDSMarket,
    *,
    upper: float = 5.0,
) -> HazardCurve:
    """Solve the intensities one segment at a time so each quoted spread is repriced.

    The intensity for segment i is found with all earlier segments fixed, so the
    i-th quote depends only on intensities up to i.
    """
    if len(maturities) != len(spreads):
        raise ValueError("maturities and spreads must have equal length")
    knots = tuple(float(m) for m in maturities)
    solved: list[float] = []
    for i, (maturity, spread) in enumerate(zip(knots, spreads, strict=True)):

        def pricing_error(
            lam: float, i: int = i, maturity: float = maturity, spread: float = spread
        ) -> float:
            curve = HazardCurve(knots[: i + 1], (*solved, lam))
            return fair_spread(curve, market, maturity) - spread

        solved.append(float(brentq(pricing_error, 1e-10, upper, xtol=1e-15)))
    return HazardCurve(knots, tuple(solved))


def mark_to_market(
    spread_now: float, spread_traded: float, rpv01_now: float, notional: float
) -> float:
    """MtM for the protection buyer: (s_t - s_0) * RPV01_t * N."""
    return (spread_now - spread_traded) * rpv01_now * notional
