# CDS Pricing and Hazard-Rate Bootstrapping

Nova SBE, Credit Risk (2026).

## Question
Given a CDS spread curve for a reference entity, what is the fair spread of a 4-year CDS, and what is a protection position bought one year ago at 60bp worth today?

## Inputs
- CDS spreads: 70bp (1y), 95bp (3y), 120bp (5y); quarterly premiums
- Recovery rate 40%, risk-free rate 3.5% (continuously compounded)
- Position: $5M notional of protection bought at 60bp

## Method
1. **Bootstrapping:** piecewise-constant default intensities on 0–1y, 1–3y and 3–5y, solved sequentially with a root finder (`scipy.optimize.brentq`) so that the model spread matches each quoted spread.
2. **Pricing:** fair 4-year spread = protection leg ÷ risky PV01 under the calibrated survival curve.
3. **Mark to market:** MtM = (s_t − s_0) × RPV01 × notional.

## Results
| | Value |
|---|---|
| Default intensities (0–1y / 1–3y / 3–5y) | 117 / 181 / 272 bp |
| Fair 4-year CDS spread | 110.7bp |
| Risky PV01 (4y) | 3.60 |
| MtM of the 60bp protection position | $91,320 |

## Files
- `cds_pricing.ipynb`: inputs and results, no external data needed.
- [`src/quantfin/cds.py`](../src/quantfin/cds.py): survival curve, risky PV01, default leg, bootstrapping and MtM; tested in [`tests/test_cds.py`](../tests/test_cds.py).
