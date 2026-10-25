# Netflix Options around the Q4 2024 Earnings Announcement

Nova SBE, Derivatives (2026).

## Question
How do Netflix option prices react to the earnings announcement of 21 Jan 2025, and how well does delta hedging replicate a call across the announcement?

## Data
OptionMetrics NFLX option quotes, 29 Nov 2024 – 27 Feb 2025 (264,817 observations after cleaning). The data is licensed (WRDS) and **not included**. To run the notebook, place the export as `NFLX_Q2_2425.csv` next to it.

## What the notebook does
1. **Implied volatility:** IV smile for a single expiry (quadratic fit), ATM term structure, daily IV level and skew. Level and skew are positively correlated (Pearson r = 0.58, p < 0.001, 60 trading days).
2. **Risk-neutral densities:** Breeden–Litzenberger densities from call prices implied by the fitted IV smile, before (17 Jan) and after (22 Jan) earnings for the 24 Jan expiry.
3. **Delta hedging:** a K = 700 call (expiry 7 Feb 2025) whose path spans the announcement, hedged daily and weekly with OptionMetrics deltas.

## Results (delta hedging)
| | Daily | Weekly | Unhedged |
|---|---|---|---|
| Terminal error, % of premium | +0.05% | −2.85% | −47.7% |

Unhedged means selling the call and holding the premium in cash at the risk-free rate.

Limitation: this is one contract on one path (18 observations, thinly traded), so it illustrates the effect and does not estimate it.

## Files
- `nflx_options.ipynb`: data loading, results and figures (written to `figs/`).
- [`src/quantfin/options.py`](../src/quantfin/options.py): quote selection, smile fit, ATM term structure, IV level and skew, Breeden–Litzenberger density, contract selection and the delta-hedging engine; tested in [`tests/test_options.py`](../tests/test_options.py).
- [`src/quantfin/black_scholes.py`](../src/quantfin/black_scholes.py): prices, delta and the lognormal density; tested in [`tests/test_black_scholes.py`](../tests/test_black_scholes.py).
