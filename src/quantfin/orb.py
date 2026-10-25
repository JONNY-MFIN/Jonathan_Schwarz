"""Intraday opening-range breakout (ORB) backtest on bid/ask minute bars.

Each day: the opening range is the high and low of the first ``or_minutes`` of
regular trading hours. The first bar after the range closes whose high breaks
the range high is a long entry, whose low breaks the range low a short entry;
if both happen on the same bar the day is skipped. Longs buy at the ask and
sell at the bid, shorts the reverse. The stop sits ``stop_k`` range widths from
the entry price and is checked on the bars after the entry bar; otherwise the
position is closed at the last bar of the session. Costs are a flat round-trip
fraction on top of the bid/ask spread.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from quantfin.performance import drawdown_series, nav_from_returns

TRADING_DAYS = 252


def compute_opening_range(rth: pd.DataFrame, or_minutes: int) -> pd.DataFrame:
    """Per-day high, low, volume and width of the first ``or_minutes`` of the session."""
    session_open = rth.groupby("date")["datetime"].transform("min")
    in_range = rth["datetime"] < session_open + pd.Timedelta(minutes=or_minutes)
    levels = (
        rth.loc[in_range]
        .groupby("date")
        .agg(or_high=("High", "max"), or_low=("Low", "min"), or_vol=("Volume", "sum"))
    )
    levels["or_range"] = levels["or_high"] - levels["or_low"]
    levels["or_minutes"] = or_minutes
    return levels


def _entry_signals(
    rth: pd.DataFrame,
    levels: pd.DataFrame,
    or_minutes: int,
    time_cutoff_hours: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First long and short breakout bar per day; returns (signals, post-range bars)."""
    bars = rth.merge(levels, left_on="date", right_index=True, how="left")
    range_close = bars.groupby("date")["datetime"].transform("min") + pd.Timedelta(
        minutes=or_minutes
    )
    after = bars["datetime"] >= range_close
    if time_cutoff_hours is not None:
        after &= bars["datetime"] <= range_close + pd.Timedelta(hours=time_cutoff_hours)
    post = bars.loc[after].copy()

    long_first = post.loc[post["High"] > post["or_high"]].groupby("date")["datetime"].idxmin()
    short_first = post.loc[post["Low"] < post["or_low"]].groupby("date")["datetime"].idxmin()

    sig = pd.DataFrame(index=levels.index)
    sig["long_idx"] = long_first.reindex(levels.index)
    sig["short_idx"] = short_first.reindex(levels.index)
    times = post["datetime"]
    long_ts = sig["long_idx"].dropna().astype(int).map(times).reindex(sig.index)
    short_ts = sig["short_idx"].dropna().astype(int).map(times).reindex(sig.index)

    both = long_ts.notna() & short_ts.notna()
    sig["direction"] = np.nan
    sig.loc[long_ts.notna() & short_ts.isna(), "direction"] = 1
    sig.loc[short_ts.notna() & long_ts.isna(), "direction"] = -1
    sig.loc[both & (long_ts < short_ts), "direction"] = 1
    sig.loc[both & (short_ts < long_ts), "direction"] = -1
    sig.loc[both & (long_ts == short_ts), "direction"] = np.nan  # same bar: ambiguous, skip
    sig["entry_idx"] = np.where(
        sig["direction"] == 1,
        sig["long_idx"],
        np.where(sig["direction"] == -1, sig["short_idx"], np.nan),
    )
    return sig, post


def run_orb_backtest(
    rth: pd.DataFrame,
    or_minutes: int = 15,
    stop_k: float = 1.0,
    range_pctile: int = 0,
    time_cutoff_hours: float | None = None,
    tc_rt: float = 1e-4,
) -> pd.DataFrame:
    """One backtest pass; a trade log with one row per day (``direction`` NaN = flat).

    ``range_pctile`` skips days whose opening range is below that percentile of
    all ranges up to and including the day itself (expanding window; the day's
    own range is known when the range closes, before any entry). 0 disables it.
    ``time_cutoff_hours`` limits entries to that many hours after the range closes.
    """
    levels = compute_opening_range(rth, or_minutes)
    if range_pctile > 0:
        threshold = levels["or_range"].expanding().quantile(range_pctile / 100)
        range_ok = levels["or_range"] >= threshold
    else:
        range_ok = pd.Series(True, index=levels.index)

    sig, post = _entry_signals(rth, levels, or_minutes, time_cutoff_hours)
    by_time = rth.set_index("datetime").sort_index()

    trades: list[dict[str, Any]] = []
    for day in levels.index:
        direction = sig.loc[day, "direction"]
        if not range_ok.loc[day] or np.isnan(direction):
            trades.append({"date": day, "direction": np.nan})
            continue

        entry_bar = post.loc[int(sig.loc[day, "entry_idx"])]
        entry_price = entry_bar["Ask"] if direction == 1 else entry_bar["Bid"]
        if np.isnan(entry_price) or entry_price <= 0:
            trades.append({"date": day, "direction": np.nan})
            continue

        entry_time = entry_bar["datetime"]
        or_range = levels.loc[day, "or_range"]
        stop = (
            entry_price - stop_k * or_range if direction == 1 else entry_price + stop_k * or_range
        )
        later = by_time.loc[(by_time.index > entry_time) & (by_time["date"] == day)]

        exit_reason, exit_time = "MOC", entry_time
        if later.empty:
            exit_price = entry_bar["Bid"] if direction == 1 else entry_bar["Ask"]
        else:
            hit = later["Low"] <= stop if direction == 1 else later["High"] >= stop
            if hit.any():
                exit_time = hit.idxmax()
                exit_reason = "STOP"
                exit_price = (
                    min(later.loc[exit_time, "Bid"], stop)
                    if direction == 1
                    else max(later.loc[exit_time, "Ask"], stop)
                )
            else:
                exit_time = later.index[-1]
                exit_price = later["Bid"].iloc[-1] if direction == 1 else later["Ask"].iloc[-1]

        if np.isnan(exit_price) or exit_price <= 0:
            trades.append({"date": day, "direction": np.nan})
            continue

        gross = direction * (exit_price - entry_price) / entry_price
        trades.append(
            {
                "date": day,
                "direction": int(direction),
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "or_range": or_range,
                "or_high": levels.loc[day, "or_high"],
                "or_low": levels.loc[day, "or_low"],
                "gross_return": gross,
                "tc": tc_rt,
                "net_return": gross - tc_rt,
            }
        )

    log = pd.DataFrame(trades)
    log["or_minutes"] = or_minutes
    log["stop_k"] = stop_k
    log["range_pctile"] = range_pctile
    log["time_cutoff"] = time_cutoff_hours
    return log


def daily_returns_from_log(log: pd.DataFrame) -> pd.Series:
    """Daily net returns; 0 on days without a trade."""
    traded = log.dropna(subset=["direction"]).set_index("date")["net_return"]
    returns = traded.reindex(log["date"]).fillna(0.0)
    returns.index = pd.DatetimeIndex(returns.index, name="date")
    return returns


def long_short_decomposition(log: pd.DataFrame) -> dict[str, pd.Series]:
    """Daily returns of the long trades and of the short trades, each 0 elsewhere."""
    dates = pd.DatetimeIndex(log["date"].values).normalize().unique()
    traded = log.dropna(subset=["direction"])
    out = {}
    for name, side in (("Long", 1), ("Short", -1)):
        series = traded[traded["direction"] == side].set_index("date")["net_return"]
        series.index = pd.DatetimeIndex(series.index)
        series = series.reindex(dates).fillna(0)
        series.index.name = "date"
        out[name] = series
    return out


def sharpe_from_log(
    log: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> float:
    """Annualised Sharpe of the daily returns between ``start`` and ``end`` (inclusive)."""
    returns = daily_returns_from_log(log)
    if start is not None:
        returns = returns[returns.index >= start]
    if end is not None:
        returns = returns[returns.index <= end]
    if len(returns) < 10 or returns.std() == 0:
        return float("nan")
    return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS))


def run_sweep(
    rth: pd.DataFrame,
    grid: Mapping[str, Sequence[Any]],
    is_end: pd.Timestamp,
    oos_start: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Backtest every combination in ``grid`` (keys are ``run_orb_backtest`` arguments).

    Parameters not in ``grid`` keep the defaults of ``run_orb_backtest``.
    """
    rows = []
    for combo in product(*grid.values()):
        kwargs = dict(zip(grid.keys(), combo, strict=True))
        log = run_orb_backtest(rth, **kwargs)
        nav = nav_from_returns(daily_returns_from_log(log))
        sharpes = {
            "is_sharpe": sharpe_from_log(log, end=is_end),
            "oos_sharpe": sharpe_from_log(log, start=oos_start),
            "full_sharpe": sharpe_from_log(log),
        }
        rows.append(
            {
                **kwargs,
                **{k: round(v, 3) if not np.isnan(v) else np.nan for k, v in sharpes.items()},
                "cum_return": nav.iloc[-1] / 100 - 1,
                "max_dd": drawdown_series(nav).min(),
                "n_trades": int(log["direction"].notna().sum()),
            }
        )
    return rows
