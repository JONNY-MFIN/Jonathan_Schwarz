from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from quantfin.creditmetrics import (
    RATINGS,
    Bond,
    Portfolio,
    beta_parameters,
    critical_value,
    idiosyncratic_std,
    load_workbook,
    map_to_states,
    rectangle_probability,
    simulate_values,
    simulated_credit_var,
    state_bounds,
)

FACTOR_CORR = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
WORKBOOK = Path(__file__).parents[1] / "credit-var-creditmetrics" / "CreditVaR_data.xlsx"

# Synthetic inputs: forward rates rise as ratings fall, transition rows sum to 1.
CURVES = pd.DataFrame(
    [[0.03 + 0.01 * i + 0.002 * t for t in range(4)] for i in range(7)],
    index=list(RATINGS[:-1]),
    columns=["Year 1", "Year 2", "Year 3", "Year 4"],
)
PROBS_BBB = pd.Series([0.001, 0.004, 0.06, 0.86, 0.05, 0.015, 0.005, 0.005], index=list(RATINGS))
PROBS_A = pd.Series([0.002, 0.02, 0.90, 0.06, 0.01, 0.005, 0.002, 0.001], index=list(RATINGS))


def make_bond(probs: pd.Series, loadings: tuple[float, ...] = (0.4, 0.2, 0.0)) -> Bond:
    return Bond(
        bond_id=1,
        coupon=0.05,
        maturity=4,
        rating="BBB",
        recovery=0.5,
        recovery_std=0.25,
        loadings=loadings,
        forward_curves=CURVES,
        transition_probs=probs,
    )


def test_state_bounds_reproduce_transition_probabilities() -> None:
    bounds = state_bounds(PROBS_BBB)
    implied = [norm.cdf(upper) - norm.cdf(lower) for lower, upper in bounds.values()]
    assert implied == pytest.approx(PROBS_BBB.to_list(), abs=1e-12)


def test_map_to_states_hits_every_interval() -> None:
    bounds = state_bounds(PROBS_BBB)
    midpoints = np.array(
        [
            u - 1 if np.isneginf(lo) else lo + 1 if np.isposinf(u) else (lo + u) / 2
            for lo, u in bounds.values()
        ]
    )
    assert list(map_to_states(midpoints, bounds)) == list(RATINGS)


def test_bond_values_fall_with_rating_and_default_pays_recovery() -> None:
    values = make_bond(PROBS_BBB).state_values()
    assert values["Default"] == 0.5
    assert values.iloc[:-1].is_monotonic_decreasing


def test_one_year_bond_is_worth_its_final_cash_flow_in_every_state() -> None:
    bond = Bond(1, 0.05, 1, "BBB", 0.5, 0.2, (0.0, 0.0, 0.0), CURVES, PROBS_BBB)
    assert bond.state_values().iloc[:-1].to_list() == pytest.approx([1.05] * 7)


def test_critical_value_is_conservative() -> None:
    values = np.array([50.0, 80.0, 90.0, 100.0])
    probs = np.array([0.004, 0.005, 0.2, 0.791])
    # cumulative: 0.004, 0.009, 0.209 -> last value with cumulative < 1% is 80
    assert critical_value(values, probs, 0.01) == 80.0


def test_joint_matrix_sums_to_one_and_has_the_right_marginals() -> None:
    portfolio = Portfolio((make_bond(PROBS_BBB), make_bond(PROBS_A)), correlation=0.3)
    joint = portfolio.joint_prob_matrix()
    assert joint.to_numpy().sum() == pytest.approx(1.0, abs=1e-6)
    assert joint.sum(axis=1).to_list() == pytest.approx(PROBS_BBB.to_list(), abs=1e-6)
    assert joint.sum(axis=0).to_list() == pytest.approx(PROBS_A.to_list(), abs=1e-6)


def test_rectangle_probability_against_closed_forms() -> None:
    # quadrant probability: P(X <= 0, Y <= 0) = 1/4 + arcsin(rho) / (2 pi)
    for rho in (-0.5, 0.0, 0.3, 0.9):
        quadrant = rectangle_probability((-np.inf, 0.0), (-np.inf, 0.0), rho)
        assert quadrant == pytest.approx(0.25 + np.arcsin(rho) / (2 * np.pi), abs=1e-10)
    # independence: the rectangle factorises
    box = rectangle_probability((-1.0, 0.5), (0.2, 2.0), 0.0)
    expected = (norm.cdf(0.5) - norm.cdf(-1.0)) * (norm.cdf(2.0) - norm.cdf(0.2))
    assert box == pytest.approx(expected, abs=1e-12)
    with pytest.raises(ValueError):
        rectangle_probability((-1.0, 1.0), (-1.0, 1.0), 1.0)


def test_zero_correlation_gives_independent_migrations() -> None:
    portfolio = Portfolio((make_bond(PROBS_BBB), make_bond(PROBS_A)), correlation=0.0)
    expected = np.outer(PROBS_BBB.to_numpy(), PROBS_A.to_numpy())
    assert portfolio.joint_prob_matrix().to_numpy() == pytest.approx(expected, abs=1e-6)


def test_mean_value_does_not_depend_on_correlation() -> None:
    bonds = (make_bond(PROBS_BBB), make_bond(PROBS_A))
    means = [Portfolio(bonds, correlation=rho).mean_value() for rho in (0.0, 0.3, 0.8)]
    assert means == pytest.approx([means[0]] * 3, rel=1e-6)


def test_idiosyncratic_std_gives_unit_variance() -> None:
    loadings = np.array([[0.4, 0.2, 0.0], [0.0, 0.0, 0.9]])
    eps = idiosyncratic_std(loadings, FACTOR_CORR)
    systematic = np.einsum("ij,jk,ik->i", loadings, FACTOR_CORR, loadings)
    assert systematic + eps**2 == pytest.approx([1.0, 1.0])
    assert eps[0] == pytest.approx(0.8672, abs=1e-4)  # value given in the assignment


def test_idiosyncratic_std_rejects_impossible_loadings() -> None:
    with pytest.raises(ValueError):
        idiosyncratic_std(np.array([[1.0, 1.0, 0.0]]), FACTOR_CORR)


def test_beta_parameters_match_mean_and_std() -> None:
    alpha, beta = beta_parameters(0.45, 0.25)
    mean = alpha / (alpha + beta)
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    assert (mean, np.sqrt(var)) == pytest.approx((0.45, 0.25))


def test_simulated_default_frequency_matches_pd() -> None:
    bond = make_bond(PROBS_BBB)
    values = simulate_values([bond], FACTOR_CORR, 200_000, seed=1)
    default_value = bond.state_values()["Default"] * 100
    assert np.mean(values == default_value) == pytest.approx(PROBS_BBB["Default"], abs=5e-4)


def test_simulation_is_reproducible_with_a_seed() -> None:
    bonds = [make_bond(PROBS_BBB), make_bond(PROBS_A, (0.0, 0.0, 0.9))]
    first = simulate_values(bonds, FACTOR_CORR, 1_000, seed=7, random_recovery=True)
    second = simulate_values(bonds, FACTOR_CORR, 1_000, seed=7, random_recovery=True)
    assert np.array_equal(first, second)


def test_simulated_credit_var_reads_the_first_percentile() -> None:
    values = np.arange(1.0, 1001.0)  # 1..1000, equally likely
    mean, v_star, cvar = simulated_credit_var(values, 0.01)
    assert (mean, v_star) == (500.5, 9.0)
    assert cvar == mean - v_star


# ── Regression on the course workbook (skipped when the data is not present) ──

needs_workbook = pytest.mark.skipif(not WORKBOOK.exists(), reason="course workbook not included")


@needs_workbook
def test_textbook_two_bond_example() -> None:
    data = load_workbook(WORKBOOK, FACTOR_CORR)
    portfolio = Portfolio((Bond.from_data(data, 1), Bond.from_data(data, 2)), correlation=0.3)
    assert portfolio.mean_value() == pytest.approx(213.27, abs=0.005)
    assert portfolio.critical_value() == pytest.approx(203.73, abs=0.005)
    assert portfolio.credit_var() == pytest.approx(9.54, abs=0.005)


@needs_workbook
def test_full_portfolio_results() -> None:
    data = load_workbook(WORKBOOK, FACTOR_CORR)
    bonds = [Bond.from_data(data, int(i)) for i in data.bonds.index]
    fixed = simulated_credit_var(simulate_values(bonds, FACTOR_CORR, 100_000, seed=42))
    beta = simulated_credit_var(
        simulate_values(bonds, FACTOR_CORR, 100_000, seed=42, random_recovery=True)
    )
    assert fixed[2] == pytest.approx(260.979, abs=1e-3)
    assert beta[2] == pytest.approx(291.6037, abs=1e-3)
