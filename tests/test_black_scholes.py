import numpy as np
import pytest

from quantfin.black_scholes import call_delta, call_price, lognormal_density, put_price

S, T, R, SIGMA = 100.0, 0.5, 0.03, 0.25


@pytest.mark.parametrize("strike", [70.0, 100.0, 130.0])
def test_put_call_parity(strike: float) -> None:
    lhs = call_price(S, strike, T, R, SIGMA) - put_price(S, strike, T, R, SIGMA)
    assert float(lhs) == pytest.approx(S - strike * np.exp(-R * T), abs=1e-10)


def test_known_value() -> None:
    # textbook example (Hull): S=42, K=40, r=10%, sigma=20%, T=0.5 -> c = 4.76
    assert float(call_price(42.0, 40.0, 0.5, 0.10, 0.20)) == pytest.approx(4.76, abs=0.005)


def test_delta_matches_finite_difference() -> None:
    h = 1e-4
    bump = (call_price(S + h, 105.0, T, R, SIGMA) - call_price(S - h, 105.0, T, R, SIGMA)) / (2 * h)
    assert float(call_delta(S, 105.0, T, R, SIGMA)) == pytest.approx(float(bump), abs=1e-6)


def test_call_price_falls_with_strike_and_rises_with_volatility() -> None:
    strikes = np.linspace(60, 140, 41)
    assert np.all(np.diff(call_price(S, strikes, T, R, SIGMA)) < 0)
    assert call_price(S, 100.0, T, R, 0.4) > call_price(S, 100.0, T, R, 0.2)


def test_vectorised_matches_scalar() -> None:
    strikes = np.array([80.0, 100.0, 120.0])
    vols = np.array([0.3, 0.25, 0.22])
    expected = [float(call_price(S, k, T, R, v)) for k, v in zip(strikes, vols, strict=True)]
    assert call_price(S, strikes, T, R, vols) == pytest.approx(expected, rel=1e-14)


def test_lognormal_density_integrates_to_one_with_forward_mean() -> None:
    grid = np.linspace(1.0, 400.0, 20_001)
    density = lognormal_density(grid, S, T, R, SIGMA)
    assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=1e-6)
    assert np.trapezoid(grid * density, grid) == pytest.approx(S * np.exp(R * T), rel=1e-6)
