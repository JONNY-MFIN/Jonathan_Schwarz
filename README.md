# Jonathan Schwarz: Quantitative Finance Projects

[![checks](https://github.com/JONNY-MFIN/Jonathan_Schwarz/actions/workflows/checks.yml/badge.svg)](https://github.com/JONNY-MFIN/Jonathan_Schwarz/actions/workflows/checks.yml)

Code for selected projects from my MSc in Finance (Financial Markets) at Nova SBE. Each folder has its own README with the question, data, method and results. The notebooks hold the narrative and the results; the logic lives in `quantfin`, a small typed and tested Python package.

| Project | Course | Topics |
|---|---|---|
| [Netflix options around earnings](nflx-options-earnings/) | Derivatives | implied-volatility smile and term structure, risk-neutral densities (Breeden–Litzenberger), daily vs weekly delta hedging |
| [WTI opening-range breakout](wti-opening-range-breakout/) | Hedge Funds | intraday backtest on 1-minute bid/ask bars, look-ahead-free filter, in-/out-of-sample split, cost sensitivity |
| [Credit VaR (CreditMetrics)](credit-var-creditmetrics/) | Credit Risk | rating-migration revaluation, joint migration probabilities, factor-model Monte Carlo, Beta-distributed recovery |
| [CDS pricing and hazard-rate bootstrapping](cds-hazard-rate-bootstrapping/) | Credit Risk | piecewise-constant default intensities, CDS fair spread, mark-to-market |

The projects started as Nova SBE coursework and have since been reworked into the package below. Licensed data (OptionMetrics via WRDS, intraday futures bars) and course data files are not included; each README says which file to place where.

## Code

```
src/quantfin/
    black_scholes.py   prices, delta and the lognormal density
    options.py         smile fits, risk-neutral densities, contract selection, delta hedging
    orb.py             opening-range breakout engine and parameter sweeps
    performance.py     NAV, drawdowns and the statistics table
    intraday.py        loading and cleaning intraday futures bars
    creditmetrics.py   Bond, Portfolio, rating thresholds, factor-model simulation
    cds.py             hazard curve, CDS pricing and bootstrapping
tests/                 pytest on synthetic data
```

Type hints throughout, checked with `mypy --strict` (one relaxation for pandas return types). The tests check the finance rather than only the plumbing, for example:

- the bootstrapped hazard curve reprices every quoted CDS spread;
- put–call parity holds, and a Breeden–Litzenberger density from a flat smile is the Black–Scholes lognormal;
- on simulated Black–Scholes paths, daily delta hedging beats weekly and its error is small relative to the premium;
- joint migration probabilities sum to one and their marginals are the single-bond transition rows;
- the breakout backtest has no look-ahead: results up to any day are unchanged when later data is removed.

Regression tests against the course workbook run when the file is present and are skipped otherwise, as in CI.

## Setup

Python 3.10 or later.

```bash
pip install -e ".[dev]"
pytest                                       # tests
mypy                                         # type check
ruff check . && ruff format --check .        # lint and formatting
```

---

## UnderQ: portfolio analytics engine

[underq.app](https://underq.app) ([live demo](https://underq.app/demo), no sign-up) is a portfolio analytics platform I am building on my own. It turns raw broker exports into FX-adjusted, multi-currency performance, risk and allocation analysis. The source is private; this section describes how the engine works.

### Architecture

```
web client             TypeScript, deployed on Vercel
   │  HTTP
api/                   FastAPI on Railway: translation only, no business logic
   │
engine/                typed, framework-free Python: all of the finance
   │
Supabase Postgres      per-user data under row-level security; shared price cache
```

The engine imports no web framework, so the same code runs behind the API, in a script, in a notebook or in a test. 480+ tests run offline in CI on every push to main.

### Domain objects

Every object that crosses a module boundary is a dataclass; no bare dicts.

| Object | What it holds |
|---|---|
| `Transaction` | one booked trade or holdings event: buy, sell, transfer, split (with ratio), corporate action; shares, price, fee, tax, currency, ISIN, contract multiplier and expiry for options |
| `CashEvent` | one cash-ledger event: deposit, withdrawal, dividend, interest, fee, tax, transfer; signed amount and currency |
| `Portfolio` | name, base currency and trading calendar; transactions and cash events attach to it by id |
| `Holding` | open-position snapshot: shares, price, value, weight, cost basis, unrealised P&L, asset class, sector |
| `NavResult` | daily value series in base currency, external flows, per-position components, unpriced tickers and FX warnings |
| `Stats` | returns, risk, risk-adjusted ratios, drawdown episodes and trade statistics for one window |
| `PriceStore` | a `Protocol` for prices, FX and instrument resolution, implemented by a disk cache, an in-memory test double and shared Postgres |

### Pipeline

```
Transaction / CashEvent
   → ledger      share series per ticker, cash balance, external flows
   → prices, fx  raw closes and FX rates, behind PriceStore
   → nav         shares × price × FX → daily value in base currency
   → returns     flow-neutral daily returns, performance index, XIRR
   → metrics     risk, risk-adjusted ratios, drawdowns, benchmark statistics

basis            the same replay, tracking cost instead of shares → cost basis, unrealised P&L
panel, frontier  holdings' returns on one calendar, converted first → covariance, frontier, risk contributions
```

**The ledger** turns the discrete event list into daily series. It only tracks quantities: shares per ticker (splits as share-ratio events, expired options closed out the day after expiry) and the cash balance. It keeps two cash views apart:

- the **cash balance** counts every cash event, because dividends, interest, fees and tax are part of the portfolio's economics;
- the **external cash flows** count only deposits, withdrawals and transfers across the portfolio boundary.

That split is what makes performance measurement work: the NAV uses the first, the return calculation neutralises the second.

### Accounting conventions

- **Time-weighted returns neutralise external flows.** A deposit is not a gain. Each day's external flow is removed before the return is computed, so performance measures the investment decisions, not the funding schedule. XIRR gives the money-weighted counterpart.
- **Raw prices, explicit dividends.** Prices are unadjusted and dividends enter as dated cash events. Adjusted closes already embed reinvested distributions, so combining them with a dividend ledger would count income twice.
- **FX at the trade date.** Each trade and cash flow converts at its own date's rate, and the daily NAV converts every position at that day's rate.
- **Convert before the covariance.** The return panel is converted to the base currency before the covariance matrix is estimated, so the risk model describes the portfolio actually held. An instrument with too little history is dropped and named, never forward-filled.
- **Trading-day returns.** The ledger is calendar-daily, but returns are restricted to trading days, so weekends do not dilute volatility and annualisation matches the benchmarks.
- **Bad data fails loudly.** A row without a usable date is dropped and reported, never defaulted. A position that cannot be priced is carried at cost and labelled, never valued at zero.

### Analytics

- **Performance:** flow-neutral TWR, XIRR, CAGR, period returns, average cost basis and unrealised P&L.
- **Risk:** volatility, drawdown episodes, historical VaR and CVaR (empirical 5th percentile of daily returns), Sharpe, Sortino and Calmar ratios.
- **Benchmarks:** FX-converted total-return benchmarks; beta, alpha and correlation; rolling Sharpe, volatility and beta.
- **Portfolio construction:** Euler risk contributions on the full covariance matrix; long-only efficient frontier by SLSQP with the minimum-variance and maximum-Sharpe portfolios; drift against target weights with rebalancing bands.

---

## Master's thesis (Sep 2026 – Jan 2027, in progress)

**Risk-managed momentum.** I rebuilt the momentum factor from the full CRSP universe (it matches Ken French's factor); the thesis scales it by HAR forecasts of variance and semivariance and tests whether real-time market-state signals improve momentum's risk or return forecast, and whether scaling by semivariance beats variance, out of sample and net of costs. The code is private.

## License

All rights reserved. The code is published for viewing; see [LICENSE](LICENSE).
