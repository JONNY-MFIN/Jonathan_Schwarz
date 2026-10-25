"""Loading and cleaning intraday futures bars (one CSV export per symbol).

Rows without a timestamp, with a missing or zero volume, or with inconsistent
OHLC values (low above high, open or close outside the range) are dropped;
duplicate timestamps keep the first row. Treasury futures (symbols starting
with ``TY``) are quoted in 32nds and parsed separately.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_SOURCE_COLS = [
    "Exchange Date",
    "Exchange Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]

CANONICAL_SOURCE_COLS = {
    "exchange date": "Exchange Date",
    "date": "Exchange Date",
    "trade date": "Exchange Date",
    "exchange time": "Exchange Time",
    "time": "Exchange Time",
    "trade time": "Exchange Time",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def parse_treasury_32nds(value: Any) -> float:
    """Convert common UST quote formats into decimal price points."""
    if pd.isna(value):
        return np.nan

    s = str(value).strip()
    if not s:
        return np.nan

    s_no_commas = s.replace(",", "")
    try:
        return float(s_no_commas)
    except ValueError:
        pass

    s_norm = re.sub(r"\s+", " ", s)

    m_space = re.match(r"^(?P<whole>\d{2,3})\s+(?P<num>\d{1,2})/32(?P<plus>\+)?$", s_norm)
    if m_space:
        whole = int(m_space.group("whole"))
        num = int(m_space.group("num"))
        if num <= 31:
            plus_tick = 1.0 / 64.0 if m_space.group("plus") else 0.0
            return whole + (num / 32.0) + plus_tick
        return np.nan

    m_dash = re.match(r"^(?P<whole>\d{2,3})-(?P<num>\d{1,2})(?P<plus>\+)?$", s_norm)
    if m_dash:
        whole = int(m_dash.group("whole"))
        num = int(m_dash.group("num"))
        if num <= 31:
            plus_tick = 1.0 / 64.0 if m_dash.group("plus") else 0.0
            return whole + (num / 32.0) + plus_tick
        return np.nan

    if re.match(r"^\d{2,3}$", s_norm):
        return float(int(s_norm))

    return np.nan


def _numeric_from_string(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    return out


def _resolve_required_columns(df: pd.DataFrame) -> dict[str, str]:
    canonical_sources: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        target = CANONICAL_SOURCE_COLS.get(key)
        if target and target not in canonical_sources:
            canonical_sources[target] = col

    missing = [c for c in REQUIRED_SOURCE_COLS if c not in canonical_sources]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return canonical_sources


def _parse_timestamp(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    ts_str = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()

    # Current files use day-month-abbrev-year plus minute timestamps.
    ts = pd.to_datetime(ts_str, format="%d-%b-%y %H:%M", errors="coerce")
    if ts.isna().any():
        ts.loc[ts.isna()] = pd.to_datetime(
            ts_str.loc[ts.isna()],
            format="%d-%b-%Y %H:%M",
            errors="coerce",
        )
    return ts


def _convert_ohlc(
    df: pd.DataFrame,
    symbol: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    volume_col: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    symbol_upper = symbol.upper()

    if symbol_upper.startswith("TY"):
        open_s = df[open_col].apply(parse_treasury_32nds)
        high_s = df[high_col].apply(parse_treasury_32nds)
        low_s = df[low_col].apply(parse_treasury_32nds)
        close_s = df[close_col].apply(parse_treasury_32nds)
    else:
        open_s = _numeric_from_string(df[open_col])
        high_s = _numeric_from_string(df[high_col])
        low_s = _numeric_from_string(df[low_col])
        close_s = _numeric_from_string(df[close_col])

    volume_s = _numeric_from_string(df[volume_col])
    return open_s, high_s, low_s, close_s, volume_s


def _prepare_symbol_with_stats(
    path: str | Path, symbol: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path, dtype=str)
    raw = _normalize_column_names(raw)
    source_cols = _resolve_required_columns(raw)

    rows_total = len(raw)

    timestamp = _parse_timestamp(
        raw,
        date_col=source_cols["Exchange Date"],
        time_col=source_cols["Exchange Time"],
    )
    open_s, high_s, low_s, close_s, volume_s = _convert_ohlc(
        raw,
        symbol=symbol,
        open_col=source_cols["Open"],
        high_col=source_cols["High"],
        low_col=source_cols["Low"],
        close_col=source_cols["Close"],
        volume_col=source_cols["Volume"],
    )

    base_valid = (
        timestamp.notna()
        & open_s.notna()
        & high_s.notna()
        & low_s.notna()
        & close_s.notna()
        & volume_s.notna()
        & (volume_s > 0)
    )

    is_consistent = (
        (low_s <= high_s)
        & (open_s >= low_s)
        & (open_s <= high_s)
        & (close_s >= low_s)
        & (close_s <= high_s)
    )
    keep_mask = base_valid & is_consistent

    out = raw.loc[keep_mask].copy()
    timestamp_kept = timestamp.loc[keep_mask]

    out = out.assign(datetime=timestamp_kept.values)
    out = out.sort_values("datetime")
    out = out.loc[~out["datetime"].duplicated(keep="first")].copy()
    out = out.reset_index(drop=True)

    # ------ convert OHLCV to numeric on the final frame ------
    out["Open"] = _numeric_from_string(out[source_cols["Open"]])
    out["High"] = _numeric_from_string(out[source_cols["High"]])
    out["Low"] = _numeric_from_string(out[source_cols["Low"]])
    out["Close"] = _numeric_from_string(out[source_cols["Close"]])
    out["Volume"] = _numeric_from_string(out[source_cols["Volume"]])

    # ------ convert optional market-data columns to numeric ------
    for col in ("Bid", "Ask", "Trade Price", "Mid Price"):
        if col in out.columns:
            out[col] = _numeric_from_string(out[col])

    # ------ derived helper columns ------
    out["date"] = out["datetime"].dt.date
    out["time"] = out["datetime"].dt.time

    rows_clean = len(out)
    rows_dropped = rows_total - rows_clean
    pct_retained = round((rows_clean / rows_total) * 100.0, 2) if rows_total else 0.0

    stats = {
        "symbol": symbol.upper(),
        "rows_total": rows_total,
        "rows_clean": rows_clean,
        "rows_dropped": rows_dropped,
        "pct_retained": pct_retained,
    }
    return out, stats


def prepare_symbol_csv(path: str | Path, symbol: str) -> pd.DataFrame:
    cleaned, _ = _prepare_symbol_with_stats(path=path, symbol=symbol)
    return cleaned


def prepare_all_symbols(
    paths: Mapping[str, str | Path] | None = None,
    *,
    csv_paths: Mapping[str, str | Path] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if paths is None:
        paths = csv_paths
    elif csv_paths is not None:
        raise TypeError("Pass only one of 'paths' or 'csv_paths'.")

    if paths is None:
        raise TypeError("Missing required argument: provide 'paths' or 'csv_paths'.")

    prepared: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []

    for symbol, path in paths.items():
        cleaned, stats = _prepare_symbol_with_stats(path=path, symbol=symbol)
        symbol_upper = symbol.upper()
        prepared[symbol_upper] = cleaned
        audit_rows.append(stats)

    audit = pd.DataFrame(audit_rows).sort_values("symbol").reset_index(drop=True)
    return prepared, audit


def regular_hours(
    bars: pd.DataFrame, session_open: dt.time, session_close: dt.time
) -> pd.DataFrame:
    """Bars within regular trading hours, with the cumulative session VWAP."""
    mask = (bars["time"] >= session_open) & (bars["time"] <= session_close)
    rth = bars.loc[mask].sort_values("datetime").reset_index(drop=True)
    value = rth["Close"] * rth["Volume"]
    rth["VWAP"] = value.groupby(rth["date"]).cumsum() / rth.groupby("date")["Volume"].cumsum()
    return rth


def close_to_close_returns(rth: pd.DataFrame) -> pd.Series:
    """Daily buy-and-hold returns from the last regular-hours close of each day."""
    closes = rth.groupby("date")["Close"].last()
    returns = closes.pct_change().dropna()
    returns.index = pd.DatetimeIndex(returns.index, name="date")
    return returns
