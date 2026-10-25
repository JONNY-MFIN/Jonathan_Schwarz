"""Option-quote analysis: smiles, risk-neutral densities and delta hedging.

Quotes are OptionMetrics-style rows with at least ``date``, ``exdate``,
``cp_flag``, ``strike_price``, ``impl_volatility``, ``volume``, ``stock_price``
and ``rf``; hedging also needs ``optionid``, ``delta`` and ``option_price``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quantfin.black_scholes import call_price, lognormal_density

FloatArray = NDArray[np.float64]


# ── Quotes ────────────────────────────────────────────────────────────────────


def prepare_quotes(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and add days to expiry and moneyness S/K."""
    quotes = raw.copy()
    quotes["date"] = pd.to_datetime(quotes["date"], format="mixed", dayfirst=True)
    quotes["exdate"] = pd.to_datetime(quotes["exdate"], format="mixed", dayfirst=True)
    quotes["time2exp"] = (quotes["exdate"] - quotes["date"]).dt.days
    quotes["moneyness"] = quotes["stock_price"] / quotes["strike_price"]
    return quotes


def quotes_on(
    quotes: pd.DataFrame, date: pd.Timestamp, exdate: pd.Timestamp, cp_flag: str | None = None
) -> pd.DataFrame:
    """Traded quotes for one date and expiry, one row per strike (the most traded)."""
    mask = (quotes["date"] == date) & (quotes["exdate"] == exdate) & (quotes["volume"] > 0)
    if cp_flag is not None:
        mask &= quotes["cp_flag"] == cp_flag
    return (
        quotes[mask]
        .sort_values(["strike_price", "volume"], ascending=[True, False])
        .drop_duplicates(subset="strike_price", keep="first")
        .reset_index(drop=True)
    )


def traded_dates(quotes: pd.DataFrame, exdate: pd.Timestamp) -> pd.DatetimeIndex:
    """Dates with at least one traded quote for ``exdate``."""
    traded = quotes[(quotes["exdate"] == exdate) & (quotes["volume"] > 0)]
    return pd.DatetimeIndex(sorted(traded["date"].unique()))


def last_date_before(
    quotes: pd.DataFrame, exdate: pd.Timestamp, when: pd.Timestamp
) -> pd.Timestamp:
    dates = traded_dates(quotes, exdate)
    return pd.Timestamp(dates[dates < when][-1])


def first_date_after(
    quotes: pd.DataFrame, exdate: pd.Timestamp, when: pd.Timestamp
) -> pd.Timestamp:
    dates = traded_dates(quotes, exdate)
    return pd.Timestamp(dates[dates > when][0])


# ── Implied volatility ────────────────────────────────────────────────────────


def fit_smile(strikes: FloatArray, ivs: FloatArray, degree: int = 2) -> np.poly1d:
    """Polynomial fit of implied volatility in strike."""
    return np.poly1d(np.polyfit(strikes, ivs, deg=degree))


def atm_term_structure(day_quotes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """ATM implied volatility per expiry: the call and the put closest to S/K = 1."""
    rows: dict[str, list[dict[str, object]]] = {"C": [], "P": []}
    for exdate, group in day_quotes[day_quotes["volume"] > 0].groupby("exdate"):
        for flag in ("C", "P"):
            side = group[group["cp_flag"] == flag]
            if side.empty:
                continue
            atm = side.iloc[int((side["moneyness"] - 1).abs().argmin())]
            rows[flag].append(
                {"exdate": exdate, "tte": atm["time2exp"], "iv_atm": atm["impl_volatility"]}
            )
    return {
        flag: pd.DataFrame(r, columns=["exdate", "tte", "iv_atm"]).sort_values("tte")
        for flag, r in rows.items()
    }


def iv_level_and_skew(quotes: pd.DataFrame) -> pd.DataFrame:
    """Daily IV level (mean of traded quotes) and skew.

    Skew is the mean IV of OTM puts (S/K > 1, so K < S) minus the mean IV of
    OTM calls (S/K < 1).
    """
    traded = quotes[quotes["volume"] > 0]
    level = traded.groupby("date")["impl_volatility"].mean().rename("level")
    skew = (
        traded.groupby("date")
        .apply(
            lambda g: (
                g.loc[g["moneyness"] > 1, "impl_volatility"].mean()
                - g.loc[g["moneyness"] < 1, "impl_volatility"].mean()
            ),
            include_groups=False,
        )
        .rename("skew")
    )
    return pd.concat([level, skew], axis=1)


# ── Risk-neutral density ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class RiskNeutralDensity:
    strikes: FloatArray
    density: FloatArray  # Breeden-Litzenberger, smoothed, tapered, normalised
    lognormal: FloatArray  # Black-Scholes density at the ATM volatility
    sigma_atm: float


def extract_rnd(
    strikes: FloatArray, ivs: FloatArray, spot: float, rate: float, t: float, n_grid: int = 500
) -> RiskNeutralDensity:
    """Breeden-Litzenberger density f(K) = e^{rT} d^2C/dK^2 from a quadratic smile.

    Calls are priced on a fine strike grid from the fitted smile, smoothed with a
    short kernel (edge-padded), differentiated twice, clipped at zero, tapered
    over the outer 5% of each tail and renormalised to integrate to one.
    """
    order = np.argsort(strikes)
    k_obs, iv_obs = strikes[order], ivs[order]

    grid = np.linspace(np.min(k_obs), np.max(k_obs), n_grid)
    iv_grid = np.clip(fit_smile(k_obs, iv_obs)(grid), 1e-4, None)
    calls = call_price(spot, grid, t, rate, iv_grid)

    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    smooth = np.convolve(np.pad(calls, pad, mode="edge"), kernel, mode="valid")

    dk = grid[1] - grid[0]
    density = np.exp(rate * t) * np.gradient(
        np.gradient(smooth, dk, edge_order=2), dk, edge_order=2
    )
    density = np.clip(density, 0.0, None)

    n_taper = max(int(0.05 * len(density)), 3)
    taper = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_taper)))
    density[:n_taper] *= taper
    density[-n_taper:] *= taper[::-1]

    mass = np.trapezoid(density, grid)
    if mass > 0:
        density = density / mass

    sigma_atm = max(float(iv_obs[np.argmin(np.abs(k_obs - spot))]), 1e-4)
    return RiskNeutralDensity(
        grid, density, lognormal_density(grid, spot, t, rate, sigma_atm), sigma_atm
    )


def density_stats(
    strikes: FloatArray, density: FloatArray, spot: float, t: float
) -> dict[str, float]:
    """Moments of a density on a strike grid, its median and P(S_T > S_0)."""
    dk = strikes[1] - strikes[0]
    mean = float(np.trapezoid(strikes * density, strikes))
    var = float(np.trapezoid((strikes - mean) ** 2 * density, strikes))
    std = float(np.sqrt(max(var, 1e-8)))
    skew = float(np.trapezoid(((strikes - mean) / std) ** 3 * density, strikes))
    kurt = float(np.trapezoid(((strikes - mean) / std) ** 4 * density, strikes))
    cdf = np.cumsum(density) * dk
    median = float(strikes[min(int(np.searchsorted(np.clip(cdf, 0, 1), 0.5)), len(strikes) - 1)])
    above = strikes > spot
    p_up = float(np.trapezoid(density[above], strikes[above])) if above.any() else 0.0
    return {
        "mean": mean,
        "std": std,
        "sigma_rnd": float((std / mean) / np.sqrt(t)),  # annualised
        "median": median,
        "skew": skew,
        "kurt": kurt,
        "kurt_ex": kurt - 3,
        "otm_prob": p_up,
    }


# ── Delta hedging ─────────────────────────────────────────────────────────────


def option_payoff(spot: float, strike: float, cp_flag: str) -> float:
    return max(spot - strike, 0.0) if cp_flag == "C" else max(strike - spot, 0.0)


def contract_path(
    quotes: pd.DataFrame, optionid: int, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """One contract's daily observations from ``start`` to ``end``, one row per date."""
    rows = quotes[
        (quotes["optionid"] == optionid) & (quotes["date"] >= start) & (quotes["date"] <= end)
    ]
    return rows.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _position_of(dates: pd.Series, when: pd.Timestamp) -> int | None:
    hits = np.flatnonzero(dates.to_numpy() == np.datetime64(when))
    return int(hits[0]) if len(hits) else None


@dataclass(frozen=True)
class HedgeSelection:
    contract: pd.Series
    path: pd.DataFrame
    start_date: pd.Timestamp
    candidates: pd.DataFrame


def select_hedge_contract(
    quotes: pd.DataFrame,
    event_date: pd.Timestamp,
    *,
    cp_flag: str = "C",
    ttm_window: tuple[int, int] = (25, 35),
    start_offsets: Iterable[int] = range(15, 45, 5),
    min_obs: int = 5,
) -> HedgeSelection:
    """Contract whose observed path has ``event_date`` closest to its midpoint.

    For each start date (``start_offsets`` calendar days before the event,
    snapped to the nearest trading day), candidates are contracts of type
    ``cp_flag`` with days to expiry in ``ttm_window`` that expire after the
    event, with fallbacks if none qualify. Candidates are ranked by whether the
    path spans the event and has an observation on it, then by distance of the
    event from the path midpoint, then by traded volume.
    """
    trade_dates = np.sort(quotes["date"].unique())
    low, high = ttm_window
    rank_cols = ["spans_earnings", "has_obs_on_earnings", "centering_error", "vol_sum"]
    rank_asc = [False, False, True, False]

    best: tuple[pd.Series, pd.DataFrame, pd.Timestamp] | None = None
    best_error = 9999.0
    for offset in start_offsets:
        target = event_date - pd.Timedelta(days=offset)
        start = pd.Timestamp(trade_dates[np.abs(trade_dates - np.datetime64(target)).argmin()])

        universe = quotes[
            (quotes["date"] == start) & (quotes["cp_flag"] == cp_flag) & (quotes["time2exp"] > 0)
        ]
        if universe.empty:
            continue
        in_window = (universe["time2exp"] >= low) & (universe["time2exp"] <= high)
        after_event = universe["exdate"] >= event_date
        for mask in (in_window & after_event, after_event, in_window):
            candidates = universe[mask]
            if not candidates.empty:
                break
        else:
            candidates = universe

        meta = candidates[
            ["optionid", "cp_flag", "strike_price", "exdate", "time2exp"]
        ].drop_duplicates()
        scores = []
        for _, row in meta.iterrows():
            path = contract_path(quotes, int(row["optionid"]), start, row["exdate"])
            spans = bool(path["date"].min() <= event_date <= path["date"].max())
            position = _position_of(path["date"], event_date) if spans else None
            scores.append(
                {
                    "optionid": int(row["optionid"]),
                    "n_obs": int(path["date"].nunique()),
                    "vol_sum": float(path["volume"].fillna(0).sum()),
                    "has_obs_on_earnings": bool((path["date"] == event_date).any()),
                    "spans_earnings": spans,
                    "centering_error": abs(position - len(path) / 2.0)
                    if position is not None
                    else 9999.0,
                    "start_date": start,
                }
            )
        meta = meta.merge(pd.DataFrame(scores), on="optionid", how="left")
        top = meta.sort_values(rank_cols, ascending=rank_asc).iloc[0]
        if top["centering_error"] < best_error:
            best_error = float(top["centering_error"])
            best = (candidates[candidates["optionid"] == top["optionid"]].iloc[0], meta, start)

    if best is None:
        raise ValueError("no contract found for the given event and filters")
    contract, meta, start = best
    path = contract_path(quotes, int(contract["optionid"]), start, contract["exdate"])
    if len(path) < min_obs:
        raise ValueError("selected contract has too few observations to compare hedges")
    return HedgeSelection(contract, path, start, meta.sort_values(rank_cols, ascending=rank_asc))


def simulate_delta_hedge(
    path: pd.DataFrame, strike: float, cp_flag: str, rebalance_step: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replicate a sold option with delta shares and a cash account.

    At inception the hedge holds ``delta`` shares and the option premium net of
    the share purchase in cash. Every ``rebalance_step`` observations the share
    position is reset to the current delta, with the trade settled in cash; cash
    accrues at the risk-free rate between observations. The hedging error is
    the hedge portfolio's value minus the option's value (at expiry: its payoff).
    """
    df = path.sort_values("date").reset_index(drop=True)
    n = len(df)
    rebalance_at = set(range(0, n - 1, rebalance_step))

    s0, c0 = float(df.loc[0, "stock_price"]), float(df.loc[0, "option_price"])
    shares = float(df.loc[0, "delta"])
    cash = c0 - shares * s0

    records = [
        {
            "date": df.loc[0, "date"],
            "stock_price": s0,
            "option_price": c0,
            "delta_used": shares,
            "shares": shares,
            "cash": cash,
            "portfolio_value": shares * s0 + cash,
            "hedging_error": shares * s0 + cash - c0,
            "rebalanced": True,
        }
    ]
    for k in range(n - 1):
        dt = max((df.loc[k + 1, "date"] - df.loc[k, "date"]).days / 365.0, 0.0)
        cash *= np.exp(float(df.loc[k, "rf"]) * dt)

        s_next, c_next = float(df.loc[k + 1, "stock_price"]), float(df.loc[k + 1, "option_price"])
        rebalanced = (k + 1) in rebalance_at and (k + 1) < n - 1
        if rebalanced:
            new_delta = float(df.loc[k + 1, "delta"])
            cash -= (new_delta - shares) * s_next
            shares = new_delta

        value = shares * s_next + cash
        records.append(
            {
                "date": df.loc[k + 1, "date"],
                "stock_price": s_next,
                "option_price": c_next,
                "delta_used": shares,
                "shares": shares,
                "cash": cash,
                "portfolio_value": value,
                "hedging_error": value - c_next,
                "rebalanced": rebalanced,
            }
        )

    out = pd.DataFrame(records)
    payoff = option_payoff(float(df.loc[n - 1, "stock_price"]), strike, cp_flag)
    terminal = float(out["portfolio_value"].iloc[-1])
    error = terminal - payoff
    stats = {
        "n_obs": float(n),
        "n_rebalances": float(out["rebalanced"].sum()),
        "terminal_portfolio": terminal,
        "terminal_payoff": payoff,
        "terminal_error": error,
        "terminal_abs_error": abs(error),
        "terminal_error_pct_premium": error / c0,
        "rmse_path_error": float(np.sqrt(np.mean(out["hedging_error"] ** 2))),
    }
    return out, stats


def unhedged_error(path: pd.DataFrame, strike: float, cp_flag: str) -> float:
    """Terminal error of selling the option and holding the premium in cash."""
    df = path.sort_values("date").reset_index(drop=True)
    cash = float(df.loc[0, "option_price"])
    for k in range(len(df) - 1):
        dt = (df.loc[k + 1, "date"] - df.loc[k, "date"]).days / 365.0
        cash *= np.exp(float(df.loc[k, "rf"]) * dt)
    return cash - option_payoff(float(df["stock_price"].iloc[-1]), strike, cp_flag)
