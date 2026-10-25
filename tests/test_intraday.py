import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantfin.intraday import (
    close_to_close_returns,
    parse_treasury_32nds,
    prepare_all_symbols,
    regular_hours,
)


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("112-05", 112 + 5 / 32),
        ("112-05+", 112 + 5 / 32 + 1 / 64),
        ("112 5/32", 112 + 5 / 32),
        ("112", 112.0),
        ("1,112.5", 1112.5),
    ],
)
def test_treasury_quotes(quote: str, expected: float) -> None:
    assert parse_treasury_32nds(quote) == pytest.approx(expected)


@pytest.mark.parametrize("quote", ["112-40", "abc", "", None])
def test_invalid_treasury_quotes_are_nan(quote: str | None) -> None:
    assert np.isnan(parse_treasury_32nds(quote))


def write_export(path: Path) -> None:
    rows = [
        # date, time, open, high, low, close, volume, bid, ask
        ("02-Jan-25", "08:59", 100, 100, 100, 100, 5, 99.9, 100.1),  # before the session
        ("02-Jan-25", "09:00", 100, 101, 99, 100.5, 10, 100.4, 100.6),
        ("02-Jan-25", "09:00", 999, 999, 999, 999, 10, 999, 999),  # duplicate timestamp
        ("02-Jan-25", "09:01", 100, 99, 101, 100, 10, 99.9, 100.1),  # low above high
        ("02-Jan-25", "09:02", 100, 101, 99, 100, 0, 99.9, 100.1),  # zero volume
        ("02-Jan-25", "09:03", "n/a", 101, 99, 101, 30, 100.9, 101.1),  # unparseable open
        ("02-Jan-25", "14:30", 101, 102, 100, 102, 20, 101.9, 102.1),
        ("03-Jan-25", "09:00", 102, 103, 101, 103, 10, 102.9, 103.1),
        ("03-Jan-25", "14:30", 103, 104, 102, 104, 10, 103.9, 104.1),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "Exchange Date",
            "Exchange Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Bid",
            "Ask",
        ],
    )
    frame.to_csv(path, index=False)


def test_cleaning_drops_bad_rows_and_reports_them(tmp_path: Path) -> None:
    export = tmp_path / "clc1.csv"
    write_export(export)
    cleaned, audit = prepare_all_symbols(paths={"CLc1": export})
    bars = cleaned["CLC1"]
    assert list(audit.loc[0, ["rows_total", "rows_clean"]]) == [9, 5]
    assert bars["datetime"].is_monotonic_increasing
    assert not bars["datetime"].duplicated().any()
    assert bars.loc[bars["datetime"] == pd.Timestamp("2025-01-02 09:00"), "Close"].item() == 100.5
    assert (bars["Volume"] > 0).all()


def test_regular_hours_and_vwap(tmp_path: Path) -> None:
    export = tmp_path / "clc1.csv"
    write_export(export)
    bars = prepare_all_symbols(paths={"CLc1": export})[0]["CLC1"]
    rth = regular_hours(bars, dt.time(9, 0), dt.time(14, 30))
    assert rth["time"].min() == dt.time(9, 0)
    first_day = rth[rth["date"] == dt.date(2025, 1, 2)]
    # VWAP after two bars: (100.5 * 10 + 102 * 20) / 30
    assert first_day["VWAP"].iloc[-1] == pytest.approx((100.5 * 10 + 102 * 20) / 30)
    returns = close_to_close_returns(rth)
    assert returns.iloc[0] == pytest.approx(104 / 102 - 1)


def test_both_path_arguments_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        prepare_all_symbols(paths={"A": tmp_path}, csv_paths={"A": tmp_path})
