# Intraday Opening-Range Breakout on WTI Crude Oil Futures

Nova SBE, Hedge Funds (2026).

## Question
Does an opening-range breakout (ORB) earn risk-adjusted returns on front-month WTI crude futures after realistic execution costs, and does it hold up out of sample?

## Data
Front-month WTI futures (CLc1), 1-minute bid/ask bars, 25 Mar 2025 – 18 Mar 2026: 345,917 rows, 320,328 after cleaning, 254 trading days. Regular trading hours 09:00–14:30 ET. The data is licensed and **not included**. To run the notebook, place the export as `Clc1.csv` next to it.

## Strategy
- **Opening range:** the high and low of the first 30 minutes. Go long on a break above the high, short on a break below the low, entering at most 1 hour after the range closes.
- **Stop:** 1× the opening-range width; otherwise exit at the close.
- **Range filter:** skip days whose opening range is below the 40th percentile of past ranges (expanding window, so each day uses only data up to that point).
- **Execution:** buy at the ask, sell at the bid, plus 1bp round-trip for fees.
- **In-/out-of-sample:** parameters were chosen on in-sample Sharpe (25 Mar – 31 Oct 2025) from a 3 × 4 grid of range length × stop multiple. Out of sample: 1 Nov 2025 – 18 Mar 2026.

## Results
| | In-sample | Out-of-sample | Full period |
|---|---|---|---|
| Sharpe | 2.57 | 1.15 | 1.84 |
| Max drawdown | −2.98% | −7.76% | −7.76% |
| Buy-and-hold Sharpe | −0.43 | 2.42 | 0.92 |

146 trades (70 long, 76 short). Sensitivity sweeps cover range length, stop multiple, range filter, entry cutoff and costs.

**Cost sensitivity** (extra round-trip costs on top of bid/ask execution):

| Extra costs | 0bp | 1bp (base) | 5bp | 10bp |
|---|---|---|---|---|
| In-sample Sharpe | 2.68 | 2.57 | 2.11 | 1.53 |
| Out-of-sample Sharpe | 1.23 | 1.15 | 0.81 | 0.39 |

**Caveats:**
- About one year of data. The out-of-sample period is under five months.
- Out of sample, buy-and-hold had the higher Sharpe in a strongly rising market. The strategy had lower volatility and drawdowns but a much lower return (+7.7% vs +56.1%).

## Files
- `orb_strategy.ipynb`: parameters, backtest runs, statistics, figures and sensitivity analysis.
- [`src/quantfin/orb.py`](../src/quantfin/orb.py): opening range, breakout signals, the backtest engine and parameter sweeps; tested in [`tests/test_orb.py`](../tests/test_orb.py), including a check that results never depend on later data.
- [`src/quantfin/performance.py`](../src/quantfin/performance.py): NAV, drawdowns and the statistics table; tested in [`tests/test_performance.py`](../tests/test_performance.py).
- [`src/quantfin/intraday.py`](../src/quantfin/intraday.py): loading, cleaning and audit of the raw bar export, regular-hours filter; tested in [`tests/test_intraday.py`](../tests/test_intraday.py).
