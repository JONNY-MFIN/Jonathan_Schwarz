from itertools import pairwise

import numpy as np
import pytest

from quantfin.cds import (
    CDSMarket,
    HazardCurve,
    bootstrap,
    fair_spread,
    mark_to_market,
    rpv01,
)

MARKET = CDSMarket(rate=0.035, recovery=0.40, period=0.25)
MATURITIES = (1.0, 3.0, 5.0)
SPREADS = (0.0070, 0.0095, 0.0120)


def test_survival_is_one_at_zero_and_decreasing() -> None:
    curve = HazardCurve((1.0, 3.0), (0.02, 0.05))
    times = np.linspace(0.0, 6.0, 61)
    values = [curve.survival(float(t)) for t in times]
    assert values[0] == 1.0
    assert all(b < a for a, b in pairwise(values))


def test_survival_integrates_piecewise() -> None:
    curve = HazardCurve((1.0, 3.0), (0.02, 0.05))
    assert curve.survival(2.0) == pytest.approx(np.exp(-(0.02 * 1.0 + 0.05 * 1.0)))
    # the last intensity extends beyond the final knot
    assert curve.survival(4.0) == pytest.approx(np.exp(-(0.02 + 0.05 * 3.0)))


def test_bootstrap_reprices_every_quote() -> None:
    curve = bootstrap(MATURITIES, SPREADS, MARKET)
    for maturity, spread in zip(MATURITIES, SPREADS, strict=True):
        assert fair_spread(curve, MARKET, maturity) == pytest.approx(spread, abs=1e-12)


def test_upward_sloping_spreads_give_rising_intensities() -> None:
    curve = bootstrap(MATURITIES, SPREADS, MARKET)
    assert curve.intensities[0] < curve.intensities[1] < curve.intensities[2]


def test_flat_curve_matches_credit_triangle() -> None:
    # with a flat intensity, s is close to lambda * (1 - R)
    curve = HazardCurve((5.0,), (0.02,))
    assert fair_spread(curve, MARKET, 5.0) == pytest.approx(0.02 * 0.6, rel=0.01)


def test_mark_to_market_sign_for_protection_buyer() -> None:
    assert mark_to_market(0.011, 0.006, 3.6, 1e6) > 0
    assert mark_to_market(0.004, 0.006, 3.6, 1e6) < 0


def test_course_example_regression() -> None:
    curve = bootstrap(MATURITIES, SPREADS, MARKET)
    assert np.array(curve.intensities) * 1e4 == pytest.approx(
        [116.6667, 180.8128, 271.9720], abs=1e-3
    )
    s4 = fair_spread(curve, MARKET, 4.0)
    annuity = rpv01(curve, MARKET, 4.0)
    assert s4 * 1e4 == pytest.approx(110.6726, abs=1e-3)
    assert annuity == pytest.approx(3.6043, abs=1e-4)
    assert mark_to_market(s4, 0.0060, annuity, 5e6) == pytest.approx(91_319.95, abs=0.05)
