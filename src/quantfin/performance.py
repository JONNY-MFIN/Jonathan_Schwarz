"""Performance statistics for daily return series."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

TRADING_DAYS = 252


def nav_from_returns(returns: pd.Series, base: float = 100.0) -> pd.Series:
    return (1 + returns).cumprod() * base


def drawdown_series(nav: pd.Series) -> pd.Series:
    """Drawdown from the running peak, as a negative fraction (-0.05 = -5%)."""
    return nav / nav.cummax() - 1


def longest_drawdown(drawdown: pd.Series) -> int:
    """Length of the longest run of observations below the running peak."""
    under = (drawdown < 0).astype(int)
    if not under.any():
        return 0
    runs = (under != under.shift()).cumsum()
    return int(under.groupby(runs).sum().max())


def sharpe_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    sigma = returns.std(ddof=1)
    return float(returns.mean() / sigma * np.sqrt(periods)) if sigma > 0 else 0.0


def stats_table(returns_by_name: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Return, risk, drawdown and win statistics, one column per return series.

    Values are formatted for display (percentages as strings). Sortino uses the
    standard deviation of negative days only; VaR and CVaR are the empirical 5%
    quantile of daily returns and the mean below it; kurtosis is excess kurtosis.
    """
    records: dict[str, dict[str, Any]] = {}
    for name, series in returns_by_name.items():
        ret = series.dropna()
        n = len(ret)
        if n < 2:
            records[name] = {}
            continue

        nav = nav_from_returns(ret)
        dd = drawdown_series(nav)
        cum_ret = nav.iloc[-1] / 100 - 1
        years = n / TRADING_DAYS
        cagr = (1 + cum_ret) ** (1 / years) - 1 if years > 0 else 0

        mu = ret.mean()
        ann_vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
        down = ret[ret < 0]
        down_std = down.std(ddof=1) if len(down) > 1 else np.nan
        sortino = (
            mu * TRADING_DAYS / (down_std * np.sqrt(TRADING_DAYS))
            if (down_std and down_std > 0)
            else np.nan
        )

        max_dd = dd.min()
        max_dd_date = pd.Timestamp(dd.index[int(np.argmin(dd.to_numpy()))])  # first trough
        calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

        wins, losses = ret[ret > 0], ret[ret < 0]
        wl_ratio = (
            wins.mean() / abs(losses.mean()) if len(losses) > 0 and losses.mean() != 0 else np.nan
        )
        profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

        var5 = np.percentile(ret, 5)
        cvar = ret[ret <= var5].mean() if (ret <= var5).any() else var5
        p95 = np.percentile(ret, 95)
        tail_ratio = abs(p95 / var5) if var5 != 0 else np.nan

        monthly = (1 + ret).resample("ME").prod() - 1
        weekly = (1 + ret).resample("W").prod() - 1
        win_months = (monthly > 0).mean() if len(monthly) > 0 else np.nan
        win_weeks = (weekly > 0).mean() if len(weekly) > 0 else np.nan
        dd_periods = dd[dd < 0]
        avg_dd = dd_periods.mean() if len(dd_periods) > 0 else 0
        traded = ret[ret != 0]

        def opt(value: float, digits: int = 3) -> float | str:
            return round(value, digits) if not np.isnan(value) else "N/A"

        records[name] = {
            "Start": ret.index[0].strftime("%Y-%m-%d"),
            "End": ret.index[-1].strftime("%Y-%m-%d"),
            "Cumulative Return": f"{cum_ret:.2%}",
            "CAGR": f"{cagr:.2%}",
            "Sharpe": round(sharpe_ratio(ret), 3),
            "Sortino": opt(sortino),
            "Max Drawdown": f"{max_dd:.2%}",
            "Max DD Date": max_dd_date.strftime("%Y-%m-%d"),
            "Longest DD (Days)": longest_drawdown(dd),
            "Volatility (ann.)": f"{ann_vol:.2%}",
            "Calmar": opt(calmar),
            "Skew": round(float(sp_stats.skew(ret)), 3),
            "Kurtosis": round(float(sp_stats.kurtosis(ret)), 3),
            "Win/Loss Ratio": opt(wl_ratio),
            "Profit Factor": opt(profit_factor),
            "Daily VaR (5%)": f"{var5:.2%}",
            "CVaR": f"{cvar:.2%}",
            "Tail Ratio": opt(tail_ratio),
            "Best Day": f"{ret.max():.2%}",
            "Worst Day": f"{ret.min():.2%}",
            "Best Month": f"{monthly.max():.2%}" if len(monthly) else "N/A",
            "Worst Month": f"{monthly.min():.2%}" if len(monthly) else "N/A",
            "Avg. Drawdown": f"{avg_dd:.2%}",
            "Win Days (%)": f"{len(wins) / n:.1%}",
            "Win Weeks (%)": f"{win_weeks:.1%}" if not np.isnan(win_weeks) else "N/A",
            "Win Months (%)": f"{win_months:.1%}" if not np.isnan(win_months) else "N/A",
            "# Trades": len(traded),
            "Avg Trade Return": f"{traded.mean():.4%}" if len(traded) > 0 else "N/A",
        }
    return pd.DataFrame(records)
