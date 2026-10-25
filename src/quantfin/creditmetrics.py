"""One-year credit VaR in the CreditMetrics framework.

A bond is revalued in every rating state it can migrate to over one year.
Migration is driven by a standard normal asset return: each rating state is an
interval of that return, with thresholds set by the issuer's transition
probabilities. Two bonds are combined analytically through a bivariate normal;
larger portfolios are simulated with a factor model for the asset returns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.stats import norm

RATINGS: tuple[str, ...] = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default")
FACTORS: tuple[str, ...] = ("T", "L", "C")

Bounds = dict[str, tuple[float, float]]


# ── Inputs ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CreditData:
    """Forward curves per rating, transition matrix, bond list and factor correlations."""

    forward_curves: pd.DataFrame  # rating x year, decimals
    transitions: pd.DataFrame  # initial rating x RATINGS, decimals, rows sum to 1
    bonds: pd.DataFrame  # one row per bond
    factor_corr: NDArray[np.float64]  # correlation of the factors T, L, C


def load_workbook(path: str | Path, factor_corr: NDArray[np.float64]) -> CreditData:
    """Read the course workbook: sheets ``interest``, ``transition`` and ``bonds``.

    Percentages are converted to decimals and the numeric rating (1 = AAA ...
    8 = Default) to its label.
    """
    sheets = pd.read_excel(path, sheet_name=None)
    raw_rates, raw_trans, raw_bonds = (sheets[name].copy() for name in list(sheets)[:3])

    def matrix(raw: pd.DataFrame) -> pd.DataFrame:
        raw.index = raw.iloc[:, 0]
        raw.columns = raw.iloc[0, :]
        table: pd.DataFrame = raw.iloc[1:, 1:].astype(float) / 100
        return table

    bonds = raw_bonds.set_index(raw_bonds.columns[0]).astype(float)
    bonds = bonds.rename(columns={"Rating_(1=AAA,8=Def)": "Rating"})
    bonds["Rating"] = bonds["Rating"].astype(int).map(dict(enumerate(RATINGS, start=1)))
    for col in ("CR", "R_mean", "R_std"):
        bonds[col] = bonds[col] / 100

    return CreditData(matrix(raw_rates), matrix(raw_trans), bonds, factor_corr)


# ── Bonds ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Bond:
    """An annual-coupon bond, priced at par today, with its migration inputs."""

    bond_id: int
    coupon: float
    maturity: int
    rating: str
    recovery: float
    recovery_std: float
    loadings: tuple[float, ...]  # on the factors T, L, C
    forward_curves: pd.DataFrame = field(repr=False)
    transition_probs: pd.Series = field(repr=False)

    @classmethod
    def from_data(cls, data: CreditData, bond_id: int) -> Bond:
        row = data.bonds.loc[bond_id]
        return cls(
            bond_id=bond_id,
            coupon=float(row["CR"]),
            maturity=int(row["Maturity"]),
            rating=str(row["Rating"]),
            recovery=float(row["R_mean"]),
            recovery_std=float(row["R_std"]),
            loadings=tuple(float(row[f"Beta_{f}"]) for f in FACTORS),
            forward_curves=data.forward_curves,
            transition_probs=data.transitions.loc[row["Rating"]],
        )

    def cash_flows(self) -> list[float]:
        """Coupons plus principal, per unit of par, one per year."""
        flows = [self.coupon] * self.maturity
        flows[-1] += 1.0
        return flows

    def state_values(self) -> pd.Series:
        """Value at the one-year horizon in each rating state, per unit of par.

        The first coupon is received at the horizon; the remaining cash flows are
        discounted on the forward zero curve of the new rating. In default the
        bond is worth its recovery rate.
        """
        first, *remaining = self.cash_flows()
        years = list(self.forward_curves.columns)
        values: dict[str, float] = {}
        for state, curve in self.forward_curves.iterrows():
            pv = sum(cf / (1 + curve[years[t]]) ** (t + 1) for t, cf in enumerate(remaining))
            values[str(state)] = first + float(pv)
        values["Default"] = self.recovery
        series: pd.Series = pd.Series(values, name="Value at t+1")
        return series

    def expected_value(self) -> float:
        return float((self.transition_probs * self.state_values()).sum())

    def critical_value(self, percentile: float = 0.01) -> float:
        return critical_value(self.state_values(), self.transition_probs, percentile)

    def credit_var(self, percentile: float = 0.01) -> float:
        return self.expected_value() - self.critical_value(percentile)


def critical_value(
    values: pd.Series | NDArray[np.float64],
    probs: pd.Series | NDArray[np.float64],
    percentile: float = 0.01,
) -> float:
    """Last value before the cumulative probability reaches ``percentile`` (conservative)."""
    dist = pd.DataFrame({"Value": np.asarray(values), "Probability": np.asarray(probs)})
    dist = dist.sort_values("Value").reset_index(drop=True)
    tail = dist[dist["Probability"].cumsum() < percentile]
    return float(tail["Value"].iloc[-1]) if len(tail) else float("nan")


# ── Rating thresholds ─────────────────────────────────────────────────────────


def state_bounds(probs: pd.Series) -> Bounds:
    """Interval (lower, upper] of the standard normal asset return for each state.

    Thresholds come from the transition probabilities accumulated from default
    upwards: Default is (-inf, z_D], CCC is (z_D, z_CCC], ..., AAA is (z_AA, inf).
    """
    reversed_states = RATINGS[::-1]
    cumulative = probs.reindex(list(RATINGS)).to_numpy()[::-1].cumsum()
    z = norm.ppf(cumulative[:-1])
    upper = {reversed_states[i]: float(z[i]) for i in range(len(z))}

    bounds: Bounds = {}
    for i, state in enumerate(RATINGS):
        if state == "AAA":
            bounds[state] = (upper["AA"], np.inf)
        elif state == "Default":
            bounds[state] = (-np.inf, upper["Default"])
        else:
            bounds[state] = (upper[RATINGS[i + 1]], upper[state])
    return bounds


def map_to_states(returns: NDArray[np.float64], bounds: Bounds) -> NDArray[np.object_]:
    """Rating state for each simulated asset return."""
    states = np.empty(len(returns), dtype=object)
    for state, (lower, upper) in bounds.items():
        if np.isneginf(lower):
            mask = returns <= upper
        elif np.isposinf(upper):
            mask = returns > lower
        else:
            mask = (returns > lower) & (returns <= upper)
        states[mask] = state
    return states


# ── Two-bond portfolio, analytical ────────────────────────────────────────────


def rectangle_probability(
    bounds_x: tuple[float, float], bounds_y: tuple[float, float], rho: float
) -> float:
    """P(l_x < X <= u_x, l_y < Y <= u_y) for standard normals with correlation rho.

    Conditioning on X, Y is normal with mean rho*x and variance 1 - rho^2, so
    the probability is the integral over x of phi(x) times the conditional
    probability of Y's interval. One-dimensional quadrature keeps this
    deterministic and accurate to about 1e-12, including infinite bounds.
    """
    if not -1.0 < rho < 1.0:
        raise ValueError("correlation must lie strictly between -1 and 1")
    (lx, ux), (ly, uy) = bounds_x, bounds_y
    scale = np.sqrt(1.0 - rho**2)

    def integrand(x: float) -> float:
        conditional = norm.cdf((uy - rho * x) / scale) - norm.cdf((ly - rho * x) / scale)
        return float(norm.pdf(x) * conditional)

    value, _ = quad(integrand, lx, ux, epsabs=1e-14, epsrel=1e-12, limit=200)
    return float(value)


@dataclass(frozen=True)
class Portfolio:
    """Two bonds with correlated asset returns; ``investment`` in each bond."""

    bonds: tuple[Bond, Bond]
    correlation: float = 0.0
    investment: float = 100.0

    def value_matrix(self) -> pd.DataFrame:
        first, second = (b.state_values() * self.investment for b in self.bonds)
        values: NDArray[np.float64] = np.add.outer(first.to_numpy(), second.to_numpy())
        frame: pd.DataFrame = pd.DataFrame(values, index=list(RATINGS), columns=list(RATINGS))
        return frame

    def joint_prob_matrix(self) -> pd.DataFrame:
        """P(l1 < r1 <= u1, l2 < r2 <= u2) for every pair of states (bivariate normal)."""
        b1, b2 = (state_bounds(b.transition_probs) for b in self.bonds)
        joint = np.zeros((len(RATINGS), len(RATINGS)))
        for i, s1 in enumerate(RATINGS):
            for j, s2 in enumerate(RATINGS):
                joint[i, j] = rectangle_probability(b1[s1], b2[s2], self.correlation)
        frame: pd.DataFrame = pd.DataFrame(joint, index=list(RATINGS), columns=list(RATINGS))
        return frame

    def distribution(self) -> pd.DataFrame:
        """All 64 joint states with portfolio value and probability, sorted by value."""
        values = self.value_matrix().stack().rename("Value")
        probs = self.joint_prob_matrix().stack().rename("Probability")
        dist: pd.DataFrame = (
            pd.concat([values, probs], axis=1).rename_axis(["State1", "State2"]).reset_index()
        )
        dist = dist.sort_values("Value").reset_index(drop=True)
        dist["Cum_Prob"] = dist["Probability"].cumsum()
        return dist

    def mean_value(self) -> float:
        dist = self.distribution()
        return float((dist["Probability"] * dist["Value"]).sum())

    def critical_value(self, percentile: float = 0.01) -> float:
        dist = self.distribution()
        return critical_value(dist["Value"], dist["Probability"], percentile)

    def credit_var(self, percentile: float = 0.01) -> float:
        return self.mean_value() - self.critical_value(percentile)


# ── Simulation ────────────────────────────────────────────────────────────────


def idiosyncratic_std(
    loadings: NDArray[np.float64], factor_corr: NDArray[np.float64]
) -> NDArray[np.float64]:
    """sigma_eps = sqrt(1 - beta' Sigma beta) per bond, since asset returns have unit variance."""
    systematic = np.einsum("ij,jk,ik->i", loadings, factor_corr, loadings)
    if np.any(systematic >= 1.0):
        raise ValueError("factor loadings imply a systematic variance of 1 or more")
    return np.asarray(np.sqrt(1.0 - systematic), dtype=np.float64)


def beta_parameters(mean: float, std: float) -> tuple[float, float]:
    """Beta(alpha, beta) with the given mean and standard deviation.

    Var = m(1 - m) / (nu + 1), so nu = m(1 - m) / s^2 - 1, alpha = m nu, beta = (1 - m) nu.
    """
    nu = mean * (1 - mean) / std**2 - 1
    if nu <= 0:
        raise ValueError("standard deviation too large for a Beta distribution with this mean")
    return mean * nu, (1 - mean) * nu


def simulate_values(
    bonds: Sequence[Bond],
    factor_corr: NDArray[np.float64],
    n_scenarios: int,
    seed: int,
    *,
    notional: float = 100.0,
    random_recovery: bool = False,
) -> NDArray[np.float64]:
    """Simulated one-year portfolio values.

    Asset returns are r_i = beta_i' f + eps_i with f ~ N(0, factor_corr) and
    independent eps_i scaled so that Var(r_i) = 1. Each return is mapped to a
    rating state and the bond revalued. With ``random_recovery`` a defaulted
    bond recovers a Beta-distributed fraction of par instead of its mean.
    """
    rng = np.random.default_rng(seed)
    loadings = np.array([b.loadings for b in bonds])
    factors = rng.multivariate_normal(np.zeros(len(FACTORS)), factor_corr, size=n_scenarios)
    eps = rng.standard_normal((n_scenarios, len(bonds))) * idiosyncratic_std(loadings, factor_corr)
    returns = factors @ loadings.T + eps

    total = np.zeros(n_scenarios)
    for j, bond in enumerate(bonds):
        states = map_to_states(returns[:, j], state_bounds(bond.transition_probs))
        value_of: Mapping[str, float] = {
            str(k): float(v) for k, v in (bond.state_values() * notional).items()
        }
        values = np.array([value_of[s] for s in states], dtype=float)
        if random_recovery:
            defaulted = states == "Default"
            alpha, beta = beta_parameters(bond.recovery, bond.recovery_std)
            values[defaulted] = rng.beta(alpha, beta, size=int(defaulted.sum())) * notional
        total += values
    return total


def simulated_credit_var(
    values: NDArray[np.float64], percentile: float = 0.01
) -> tuple[float, float, float]:
    """(mean, critical value, credit VaR) from simulated values, equally weighted."""
    ordered = np.sort(values)
    cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
    tail = ordered[cumulative < percentile]
    v_star = float(tail[-1]) if len(tail) else float("nan")
    mean = float(ordered.mean())
    return mean, v_star, mean - v_star
