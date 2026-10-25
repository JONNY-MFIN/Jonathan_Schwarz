# Credit VaR for a Bond Portfolio (CreditMetrics)

Nova SBE, Credit Risk (2026).

## Question
What is the one-year 99% credit VaR of a bond portfolio when losses come from rating migrations and defaults, and how much do correlation, diversification and recovery uncertainty change it?

## Inputs (course data, not included)
- One-year forward zero curves per rating and the one-year rating transition matrix (CreditMetrics tables)
- 17 bonds: coupon, maturity, rating, recovery mean and standard deviation, loadings on three correlated factors (T, L, C)
- Each bond priced at par today, $100 invested in each

To run the notebook, place `CreditVaR_data.xlsx` next to it.

## Method
- **`Bond`:** values the bond at the one-year horizon in each rating state by discounting the remaining cash flows on that rating's forward curve; recovery in default. Expected value, percentile value and stand-alone credit VaR from the issuer's transition row.
- **`Portfolio` (two bonds):** each rating state maps to an interval of a standard normal asset return. Joint migration probabilities are bivariate normal rectangle probabilities, which give the full distribution of portfolio values.
- **Simulation:** asset returns $r_i = \beta_i' f + \varepsilon_i$ with correlated factors $f$ and $\sigma^2_{\varepsilon_i} = 1 - \beta_i'\Sigma\beta_i$; 100,000 scenarios, each return mapped to a rating state and revalued.
- **Random recovery:** in default, recovery is drawn from a Beta distribution matched to the recovery mean and standard deviation.

## Results (99%, $100 per bond)
| | Mean value | 1st percentile | Credit VaR |
|---|---|---|---|
| Two bonds, analytical, ρ = 0.3 | 213.27 | 203.73 | 9.54 |
| Two bonds, analytical, ρ = 0.072 | 213.27 | 203.73 | 9.54 |
| Two bonds, factor-model simulation | 213.28 | 204.39 | 8.89 |
| 17 bonds, constant recovery | 1,744.41 | 1,483.43 | 260.98 |
| 17 bonds, Beta recovery | 1,744.35 | 1,452.75 | 291.60 |

- The analytical two-bond case reproduces the course reference (credit VaR 9.54 against 9.23 in the CreditMetrics documentation, which differs slightly in the joint probabilities). The percentile is read conservatively off a discrete distribution, which is why the two correlations give the same number.
- Random recovery leaves the mean almost unchanged but adds about 31 to credit VaR through a fatter left tail.

## Files
- `credit_var.ipynb`: inputs, intermediate tables and results.
- [`src/quantfin/creditmetrics.py`](../src/quantfin/creditmetrics.py): `Bond`, `Portfolio`, rating thresholds and the factor-model simulation; tested in [`tests/test_creditmetrics.py`](../tests/test_creditmetrics.py).
