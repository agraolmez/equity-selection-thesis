from __future__ import annotations

import numpy as np

from portfolio_bot.research_v6_1.portfolio_step_04 import solve_optional_short_portfolio


def _solve(
    mu: np.ndarray,
    *,
    long: np.ndarray,
    short: np.ndarray,
    alpha_short: np.ndarray | None = None,
    betas: np.ndarray | None = None,
    risk_aversion: float = 3.0,
    market_beta_limit: float = 1.0,
    spy_allowed: bool = False,
):
    n = len(mu)
    if alpha_short is None:
        alpha_short = np.zeros(n, dtype=bool)
    if betas is None:
        betas = np.zeros((n, 6), dtype=float)
    covariance_root = np.zeros((6, 6), dtype=float)
    covariance_root[0, 0] = np.sqrt(0.001)
    return solve_optional_short_portfolio(
        expected_stock_excess_return=np.asarray(mu, dtype=float),
        long_candidate=np.asarray(long, dtype=bool),
        short_candidate=np.asarray(short, dtype=bool),
        alpha_short_candidate=np.asarray(alpha_short, dtype=bool),
        reduction_only=np.zeros(n, dtype=bool),
        previous_stock_weights=np.zeros(n),
        factor_betas=betas,
        factor_covariance_root=covariance_root,
        specific_variance_monthly=np.full(n, 1e-6),
        full_spread=np.zeros(n),
        adv20_usd=np.full(n, 1e10),
        impact_volatility_daily=np.zeros(n),
        borrow_rate_annual=np.zeros(n),
        sic2_industry=np.array(["10"] * n),
        nav_dollars=100_000_000.0,
        risk_aversion=risk_aversion,
        forecast_volatility_limit=0.50,
        maximum_gross=1.50,
        minimum_net=-1.0,
        maximum_net=1.0,
        maximum_position_weight=0.20,
        maximum_one_way_turnover=1.0,
        maximum_one_way_adv_participation=1.0,
        market_beta_absolute_limit=market_beta_limit,
        other_factor_absolute_limit=1.0,
        sic2_industry_net_absolute_limit=1.0,
        sic2_industry_gross_limit=1.5,
        spread_multiplier=1.0,
        impact_coefficient=0.0,
        financing_rate_annual=0.0,
        spy_allowed=spy_allowed,
        spy_expected_excess_return=0.005,
        spy_factor_exposures=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        spy_specific_variance_monthly=1e-6,
        maximum_spy_short_gross=0.75,
        spy_transaction_cost_rate=0.0,
        spy_borrow_rate_annual=0.0,
        require_nonnegative_expected_alpha_short_net=bool(alpha_short.any()),
    )


def test_zero_is_feasible_without_long_or_short_floor() -> None:
    solution = _solve(
        np.array([0.0, 0.0]),
        long=np.array([True, False]),
        short=np.array([False, True]),
        risk_aversion=10.0,
    )
    assert np.max(np.abs(solution.stock_weights)) < 1e-5
    assert abs(solution.spy_weight) < 1e-8


def test_utility_can_accept_standalone_negative_short_for_covariance_benefit() -> None:
    betas = np.zeros((2, 6), dtype=float)
    betas[:, 0] = 1.0
    solution = _solve(
        np.array([0.020, 0.005]),
        long=np.array([True, False]),
        short=np.array([False, True]),
        betas=betas,
        risk_aversion=10.0,
    )
    assert solution.stock_weights[0] > 0.05
    # The second stock has a positive expected stock return, so shorting it has
    # negative standalone expected return.  It is nevertheless useful here as
    # a covariance hedge for the stronger long opportunity.
    assert solution.stock_weights[1] < -0.01


def test_spy_supplies_residual_beta_hedge_separately() -> None:
    betas = np.zeros((1, 6), dtype=float)
    betas[0, 0] = 1.0
    solution = _solve(
        np.array([0.030]),
        long=np.array([True]),
        short=np.array([False]),
        betas=betas,
        risk_aversion=1.0,
        market_beta_limit=0.001,
        spy_allowed=True,
    )
    assert solution.stock_weights[0] > 0.05
    assert solution.spy_weight < -0.05
    assert abs(solution.factor_exposures[0]) <= 0.00101
