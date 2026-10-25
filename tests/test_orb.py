import datetime as dt

import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike

from quantfin.orb import (
    compute_opening_range,
    daily_returns_from_log,
    long_short_decomposition,
    run_orb_backtest,
    run_sweep,
)

SPREAD = 0.01  # half-spread: bid = price - SPREAD, ask = price + SPREAD


def session(day: str, prices: ArrayLike) -> pd.DataFrame:
    """One session of minute bars from 09:00 with flat bars at ``prices``."""
    p = np.asarray(prices, dtype=float)
    times = pd.date_range(pd.Timestamp(f"{day} 09:00"), periods=len(p), freq="min")
    return pd.DataFrame(
        {
            "datetime": times,
            "date": times.date,
            "time": times.time,
            "Open": p,
            "High": p,
            "Low": p,
            "Close": p,
            "Bid": p - SPREAD,
            "Ask": p + SPREAD,
            "Volume": 1.0,
        }
    )


def bars(*sessions: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(sessions, ignore_index=True)


def opening(n: int = 30, low: float = 100.0, high: float = 101.0) -> list[float]:
    """An opening range that alternates between ``low`` and ``high``."""
    return [low if i % 2 == 0 else high for i in range(n)]


def test_opening_range_uses_only_the_first_minutes() -> None:
    day = session("2025-01-02", opening() + [105.0] * 30)
    levels = compute_opening_range(day, or_minutes=30)
    row = levels.iloc[0]
    assert (row["or_high"], row["or_low"], row["or_range"]) == (101.0, 100.0, 1.0)


def test_long_breakout_exits_at_the_close() -> None:
    prices = opening() + [100.5] * 5 + [101.5] + [102.0] * 20
    log = run_orb_backtest(session("2025-01-02", prices), or_minutes=30, stop_k=1.0, tc_rt=1e-4)
    trade = log.iloc[0]
    assert trade["direction"] == 1
    assert trade["exit_reason"] == "MOC"
    assert trade["entry_price"] == pytest.approx(101.5 + SPREAD)  # bought at the ask
    assert trade["exit_price"] == pytest.approx(102.0 - SPREAD)  # sold at the bid
    gross = (102.0 - SPREAD - 101.5 - SPREAD) / (101.5 + SPREAD)
    assert trade["net_return"] == pytest.approx(gross - 1e-4)


def test_short_breakout_is_stopped_out() -> None:
    prices = opening() + [99.5] + [100.0] * 3 + [101.0] + [100.0] * 10
    log = run_orb_backtest(session("2025-01-02", prices), or_minutes=30, stop_k=1.0, tc_rt=0.0)
    trade = log.iloc[0]
    entry = 99.5 - SPREAD  # sold at the bid
    stop = entry + 1.0  # one range width
    assert trade["direction"] == -1
    assert trade["exit_reason"] == "STOP"
    assert trade["exit_price"] == pytest.approx(max(101.0 + SPREAD, stop))  # bought back at the ask
    assert trade["net_return"] < 0


def test_same_bar_break_on_both_sides_is_skipped() -> None:
    day = session("2025-01-02", opening() + [100.5] * 10)
    breakout = day.index[35]
    day.loc[breakout, ["High", "Low"]] = [102.0, 99.0]
    log = run_orb_backtest(day, or_minutes=30)
    assert np.isnan(log.iloc[0]["direction"])


def test_entry_cutoff_blocks_late_breakouts() -> None:
    prices = opening() + [100.5] * 120 + [102.0] * 10  # breakout two hours after the range
    day = session("2025-01-02", prices)
    assert np.isnan(run_orb_backtest(day, or_minutes=30, time_cutoff_hours=1).iloc[0]["direction"])
    assert run_orb_backtest(day, or_minutes=30, time_cutoff_hours=None).iloc[0]["direction"] == 1


def random_sessions(n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=n_days)
    out = []
    for d in days:
        walk = 100 + np.cumsum(rng.normal(0, 0.05 * rng.uniform(0.5, 2.0), 240))
        out.append(session(d.strftime("%Y-%m-%d"), walk))
    return bars(*out)


def test_no_look_ahead_results_do_not_depend_on_later_days() -> None:
    data = random_sessions(30, seed=3)
    full = run_orb_backtest(data, or_minutes=30, range_pctile=40, time_cutoff_hours=1)
    cutoff = pd.Timestamp("2025-01-24").date()
    truncated = run_orb_backtest(
        data[data["date"] <= cutoff], or_minutes=30, range_pctile=40, time_cutoff_hours=1
    )
    head = full[full["date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(head, truncated)


def test_range_filter_skips_narrow_days() -> None:
    wide = session("2025-01-02", opening(high=102.0) + [103.0] * 10)
    narrow = session("2025-01-03", opening(high=100.2) + [101.0] * 10)
    log = run_orb_backtest(bars(wide, narrow), or_minutes=30, range_pctile=50)
    assert log["direction"].tolist()[0] == 1
    assert np.isnan(log["direction"].tolist()[1])


def test_daily_returns_are_zero_on_flat_days() -> None:
    trade_day = session("2025-01-02", opening() + [101.5] + [102.0] * 10)
    flat_day = session("2025-01-03", opening() + [100.5] * 10)
    log = run_orb_backtest(bars(trade_day, flat_day), or_minutes=30)
    returns = daily_returns_from_log(log)
    assert isinstance(returns.index, pd.DatetimeIndex)
    assert returns.iloc[1] == 0.0
    assert returns.iloc[0] == pytest.approx(log.iloc[0]["net_return"])
    sides = long_short_decomposition(log)
    assert sides["Long"].sum() == pytest.approx(returns.sum())
    assert sides["Short"].sum() == 0.0


def test_sweep_covers_the_grid_and_costs_lower_sharpe() -> None:
    data = random_sessions(40, seed=5)
    rows = run_sweep(
        data,
        {"or_minutes": [15, 30], "stop_k": [1.0], "tc_rt": [0.0, 1e-3]},
        is_end=pd.Timestamp("2025-02-10"),
        oos_start=pd.Timestamp("2025-02-11"),
    )
    assert len(rows) == 4
    by_cost = {(r["or_minutes"], r["tc_rt"]): r["full_sharpe"] for r in rows}
    assert by_cost[(30, 1e-3)] < by_cost[(30, 0.0)]


def test_session_helper_matches_cleaned_data_layout() -> None:
    day = session("2025-01-02", [100.0, 101.0])
    assert isinstance(day.loc[0, "date"], dt.date)
    assert isinstance(day.loc[0, "time"], dt.time)
