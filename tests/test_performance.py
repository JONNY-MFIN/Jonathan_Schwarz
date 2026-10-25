import numpy as np
import pandas as pd
import pytest

from quantfin.performance import (
    drawdown_series,
    longest_drawdown,
    nav_from_returns,
    sharpe_ratio,
    stats_table,
)


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2025-01-02", periods=len(values)))


def test_nav_compounds_from_base() -> None:
    nav = nav_from_returns(series([0.10, -0.10]))
    assert nav.tolist() == pytest.approx([110.0, 99.0])


def test_drawdown_and_longest_drawdown() -> None:
    nav = series([100, 110, 99, 105, 110, 120, 108])
    dd = drawdown_series(nav)
    assert dd.min() == pytest.approx(99 / 110 - 1)
    assert dd.iloc[-1] == pytest.approx(108 / 120 - 1)
    assert longest_drawdown(dd) == 2  # 99 and 105 sit below the 110 peak
    assert longest_drawdown(drawdown_series(series([1, 2, 3]))) == 0


def test_sharpe_ratio() -> None:
    rng = np.random.default_rng(0)
    returns = series(list(rng.normal(0.001, 0.01, 1000)))
    expected = returns.mean() / returns.std() * np.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)
    assert sharpe_ratio(series([0.0, 0.0, 0.0])) == 0.0


def test_stats_table_columns_and_values() -> None:
    returns = series([0.01, -0.02, 0.0, 0.03, -0.01, 0.0, 0.02])
    table = stats_table({"strategy": returns, "too short": series([0.01])})
    col = table["strategy"]
    assert col["# Trades"] == 5  # non-zero days
    assert col["Cumulative Return"] == f"{nav_from_returns(returns).iloc[-1] / 100 - 1:.2%}"
    assert col["Win Days (%)"] == f"{3 / 7:.1%}"
    assert table["too short"].isna().all()
