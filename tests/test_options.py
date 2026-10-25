import numpy as np
import pandas as pd
import pytest

from quantfin.black_scholes import call_delta, call_price, lognormal_density
from quantfin.options import (
    density_stats,
    extract_rnd,
    iv_level_and_skew,
    option_payoff,
    quotes_on,
    select_hedge_contract,
    simulate_delta_hedge,
    unhedged_error,
)

S, R, SIGMA = 100.0, 0.03, 0.30


def bs_path(seed: int, n_days: int = 60, strike: float = 100.0) -> pd.DataFrame:
    """A GBM stock path with Black-Scholes option prices and deltas, one row per day."""
    rng = np.random.default_rng(seed)
    dt = 1 / 365
    shocks = rng.standard_normal(n_days - 1)
    log_s = np.log(S) + np.concatenate(
        [[0.0], np.cumsum((R - 0.5 * SIGMA**2) * dt + SIGMA * np.sqrt(dt) * shocks)]
    )
    spot = np.exp(log_s)
    ttm = (n_days - np.arange(n_days)) / 365  # expiry the day after the last observation
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n_days, freq="D"),
            "stock_price": spot,
            "option_price": [
                float(call_price(s, strike, t, R, SIGMA)) for s, t in zip(spot, ttm, strict=True)
            ],
            "delta": [
                float(call_delta(s, strike, t, R, SIGMA)) for s, t in zip(spot, ttm, strict=True)
            ],
            "rf": R,
        }
    )


def test_option_payoff() -> None:
    assert option_payoff(110.0, 100.0, "C") == 10.0
    assert option_payoff(90.0, 100.0, "C") == 0.0
    assert option_payoff(90.0, 100.0, "P") == 10.0


def test_hedge_starts_with_zero_error() -> None:
    out, _ = simulate_delta_hedge(bs_path(0), 100.0, "C", rebalance_step=1)
    assert out["hedging_error"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_daily_rebalancing_beats_weekly_on_black_scholes_paths() -> None:
    daily, weekly = [], []
    for seed in range(40):
        path = bs_path(seed)
        daily.append(simulate_delta_hedge(path, 100.0, "C", 1)[1]["terminal_abs_error"])
        weekly.append(simulate_delta_hedge(path, 100.0, "C", 5)[1]["terminal_abs_error"])
    premium = float(call_price(S, 100.0, 60 / 365, R, SIGMA))
    assert np.mean(daily) < np.mean(weekly)
    assert np.mean(daily) < 0.1 * premium


def test_rebalance_counts() -> None:
    path = bs_path(1, n_days=21)
    _, daily = simulate_delta_hedge(path, 100.0, "C", 1)
    _, weekly = simulate_delta_hedge(path, 100.0, "C", 5)
    assert daily["n_rebalances"] == 20  # inception plus every day except the last
    assert weekly["n_rebalances"] == 4  # inception, then observations 5, 10, 15


def test_unhedged_error_when_option_expires_worthless() -> None:
    path = bs_path(2, n_days=30)
    strike = float(path["stock_price"].max()) + 50  # deep out of the money throughout
    growth = np.exp(R * 29 / 365)
    expected = float(path["option_price"].iloc[0]) * growth
    assert unhedged_error(path, strike, "C") == pytest.approx(expected, rel=1e-12)


def test_rnd_from_a_flat_smile_is_lognormal() -> None:
    strikes = np.linspace(30.0, 250.0, 60)
    t = 0.25
    rnd = extract_rnd(strikes, np.full_like(strikes, SIGMA), S, R, t)
    stats = density_stats(rnd.strikes, rnd.density, S, t)
    assert np.trapezoid(rnd.density, rnd.strikes) == pytest.approx(1.0, abs=1e-9)
    assert stats["mean"] == pytest.approx(S * np.exp(R * t), rel=0.01)
    assert stats["sigma_rnd"] == pytest.approx(SIGMA, rel=0.05)
    interior = (rnd.strikes > 60) & (rnd.strikes < 160)
    reference = lognormal_density(rnd.strikes, S, t, R, SIGMA)
    assert np.max(np.abs(rnd.density - reference)[interior]) < 0.02 * np.max(reference)


def test_density_stats_of_a_normal() -> None:
    grid = np.linspace(40.0, 160.0, 4001)
    density = np.exp(-0.5 * ((grid - 100) / 10) ** 2) / (10 * np.sqrt(2 * np.pi))
    stats = density_stats(grid, density, spot=100.0, t=1.0)
    assert stats["mean"] == pytest.approx(100.0, abs=1e-6)
    assert stats["std"] == pytest.approx(10.0, rel=1e-4)
    assert stats["skew"] == pytest.approx(0.0, abs=1e-6)
    assert stats["kurt"] == pytest.approx(3.0, abs=1e-3)
    assert stats["median"] == pytest.approx(100.0, abs=0.05)
    # integrates strictly above the spot, so half a grid cell at K = S is left out
    assert stats["otm_prob"] == pytest.approx(0.5, abs=2e-3)


def quote_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["moneyness"] = frame["stock_price"] / frame["strike_price"]
    return frame


def test_quotes_on_keeps_the_most_traded_quote_per_strike() -> None:
    day, expiry = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-02-01")
    base = {"date": day, "exdate": expiry, "stock_price": 100.0, "impl_volatility": 0.3}
    quotes = quote_frame(
        [
            {**base, "cp_flag": "C", "strike_price": 100.0, "volume": 5},
            {**base, "cp_flag": "P", "strike_price": 100.0, "volume": 9},
            {**base, "cp_flag": "C", "strike_price": 105.0, "volume": 0},
        ]
    )
    both = quotes_on(quotes, day, expiry)
    assert both["strike_price"].tolist() == [100.0]
    assert both["cp_flag"].tolist() == ["P"]
    assert quotes_on(quotes, day, expiry, cp_flag="C")["volume"].tolist() == [5]


def test_skew_is_put_side_minus_call_side() -> None:
    day = pd.Timestamp("2025-01-02")
    quotes = quote_frame(
        [
            {
                "date": day,
                "stock_price": 100.0,
                "strike_price": 80.0,
                "impl_volatility": 0.40,
                "volume": 1,
            },
            {
                "date": day,
                "stock_price": 100.0,
                "strike_price": 120.0,
                "impl_volatility": 0.25,
                "volume": 1,
            },
        ]
    )
    result = iv_level_and_skew(quotes)
    assert result.loc[day, "skew"] == pytest.approx(0.15)
    assert result.loc[day, "level"] == pytest.approx(0.325)


def test_selection_prefers_the_path_centred_on_the_event() -> None:
    event = pd.Timestamp("2025-01-21")
    dates = pd.bdate_range("2024-12-02", "2025-02-28")
    rows = []
    for optionid, expiry in ((1, pd.Timestamp("2025-01-24")), (2, pd.Timestamp("2025-02-14"))):
        for d in dates[dates <= expiry]:
            rows.append(
                {
                    "optionid": optionid,
                    "date": d,
                    "exdate": expiry,
                    "cp_flag": "C",
                    "strike_price": 100.0,
                    "time2exp": (expiry - d).days,
                    "volume": 1,
                }
            )
    selection = select_hedge_contract(pd.DataFrame(rows), event, ttm_window=(20, 40))
    path_dates = selection.path["date"]
    position = int(np.flatnonzero(path_dates == event)[0])
    best = selection.candidates.iloc[0]
    # the short contract expires three days after the event, so the longer one is better centred
    assert selection.contract["optionid"] == 2
    assert best["centering_error"] == abs(position - len(path_dates) / 2)
    assert best["centering_error"] == selection.candidates["centering_error"].min()
    assert path_dates.iloc[0] == selection.start_date
