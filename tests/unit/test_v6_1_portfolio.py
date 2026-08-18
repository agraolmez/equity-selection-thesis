from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_bot.research_v6_1.portfolio import (
    causal_monotonic_calibration,
    rank_strength,
    solve_expected_net_return_portfolio,
)


def test_rank_strength_methods_preserve_order_and_emphasize_tail() -> None:
    ranks = np.asarray([0.79, 0.80, 0.90, 0.99])
    linear = rank_strength(ranks, "linear_tail")
    gaussian = rank_strength(ranks, "gaussian_tail")
    squared = rank_strength(ranks, "squared_tail")
    assert linear[0] == 0.0
    assert gaussian[0] == 0.0
    assert squared[0] == 0.0
    assert np.all(np.diff(linear[1:]) > 0.0)
    assert np.all(np.diff(gaussian[1:]) > 0.0)
    assert np.all(np.diff(squared[1:]) > 0.0)
    assert squared[-1] / squared[2] > linear[-1] / linear[2]


def test_causal_calibration_excludes_unavailable_future_and_is_monotonic() -> None:
    rows = []
    for month_index, month in enumerate(pd.period_range("2020-01", "2021-12", freq="M").astype(str)):
        for score in (0.05, 0.25, 0.50, 0.75, 0.95):
            rows.append(
                {
                    "holding_month": month,
                    "score": score,
                    "outcome": score - 0.5,
                    "available": pd.Timestamp("2022-01-15"),
                }
            )
    rows.append(
        {
            "holding_month": "2022-01",
            "score": 0.95,
            "outcome": -1000.0,
            "available": pd.Timestamp("2023-01-01"),
        }
    )
    history = pd.DataFrame(rows)
    result = causal_monotonic_calibration(
        history,
        np.asarray([0.05, 0.50, 0.95]),
        score_column="score",
        outcome_column="outcome",
        month_column="holding_month",
        availability_column="available",
        signal_date=pd.Timestamp("2022-02-01"),
        bin_edges=[0.0, 0.10, 0.40, 0.60, 0.90, 1.0],
        minimum_history_months=24,
        shrinkage_prior_months=12,
    )
    assert result.history_months == 24
    assert np.all(np.diff(result.values) >= 0.0)
    assert result.values[-1] > 0.0


def test_optimizer_maximizes_net_edge_with_risk_and_side_constraints() -> None:
    n = 8
    mu = np.asarray([0.03, 0.025, 0.02, 0.015, -0.03, -0.025, -0.02, -0.015])
    long_candidate = np.asarray([True] * 4 + [False] * 4)
    short_candidate = ~long_candidate
    result = solve_expected_net_return_portfolio(
        expected_stock_excess_return=mu,
        long_candidate=long_candidate,
        short_candidate=short_candidate,
        reduction_only=np.zeros(n, dtype=bool),
        previous_weights=np.zeros(n),
        factor_betas=np.zeros((n, 6)),
        factor_covariance_root=np.zeros((6, 6)),
        specific_variance_monthly=np.full(n, 0.0001),
        full_spread=np.zeros(n),
        adv20_usd=np.full(n, 1e9),
        impact_volatility_daily=np.zeros(n),
        borrow_rate_annual=np.zeros(n),
        sic2_industry=np.asarray([f"I{i}" for i in range(n)]),
        nav_dollars=100_000_000.0,
        forecast_volatility_target=0.50,
        maximum_gross=0.60,
        minimum_long_gross=0.20,
        minimum_short_gross=0.20,
        minimum_net=-0.10,
        maximum_net=0.10,
        maximum_position_weight=0.10,
        maximum_one_way_turnover=0.50,
        maximum_one_way_adv_participation=0.10,
        market_beta_absolute_limit=0.10,
        other_factor_absolute_limit=0.10,
        sic2_industry_net_absolute_limit=0.10,
        sic2_industry_gross_limit=0.30,
        spread_multiplier=1.0,
        impact_coefficient=0.10,
        financing_rate_annual=0.0,
    )
    assert result.status in {"optimal", "optimal_inaccurate"}
    assert result.long_gross >= 0.20 - 1e-7
    assert result.short_gross >= 0.20 - 1e-7
    assert result.gross <= 0.60 + 1e-7
    assert np.all(result.weights[long_candidate] >= -1e-9)
    assert np.all(result.weights[short_candidate] <= 1e-9)
    assert result.objective_value > 0.0


def test_optimizer_caps_zero_alpha_hedge_shorts_without_forcing_them() -> None:
    result = solve_expected_net_return_portfolio(
        expected_stock_excess_return=np.asarray([0.03, 0.02, 0.0, 0.0]),
        long_candidate=np.asarray([True, True, False, False]),
        short_candidate=np.asarray([False, False, True, True]),
        hedge_short_candidate=np.asarray([False, False, True, True]),
        maximum_hedge_short_gross=0.05,
        reduction_only=np.zeros(4, dtype=bool),
        previous_weights=np.zeros(4),
        factor_betas=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), (4, 1)),
        factor_covariance_root=np.zeros((6, 6)),
        specific_variance_monthly=np.full(4, 0.0001),
        full_spread=np.zeros(4),
        adv20_usd=np.full(4, 1e9),
        impact_volatility_daily=np.zeros(4),
        borrow_rate_annual=np.zeros(4),
        sic2_industry=np.asarray(["A", "B", "C", "D"]),
        nav_dollars=100_000_000.0,
        forecast_volatility_target=0.50,
        maximum_gross=0.50,
        minimum_long_gross=0.0,
        minimum_short_gross=0.0,
        minimum_net=-0.10,
        maximum_net=1.0,
        maximum_position_weight=0.10,
        maximum_one_way_turnover=0.50,
        maximum_one_way_adv_participation=0.10,
        market_beta_absolute_limit=0.10,
        other_factor_absolute_limit=0.10,
        sic2_industry_net_absolute_limit=0.10,
        sic2_industry_gross_limit=0.30,
        spread_multiplier=1.0,
        impact_coefficient=0.10,
        financing_rate_annual=0.0,
    )
    assert result.short_gross <= 0.05 + 1e-7
    assert result.long_gross > 0.0


def test_optimizer_lexicographically_unwinds_stale_holdings() -> None:
    previous = np.asarray([0.08, 0.0, 0.0, 0.0, 0.0])
    result = solve_expected_net_return_portfolio(
        expected_stock_excess_return=np.asarray([0.0, 0.03, 0.02, -0.03, -0.02]),
        long_candidate=np.asarray([False, True, True, False, False]),
        short_candidate=np.asarray([False, False, False, True, True]),
        reduction_only=np.asarray([True, False, False, False, False]),
        previous_weights=previous,
        factor_betas=np.zeros((5, 6)),
        factor_covariance_root=np.zeros((6, 6)),
        specific_variance_monthly=np.full(5, 0.0001),
        full_spread=np.full(5, 0.02),
        adv20_usd=np.full(5, 1e9),
        impact_volatility_daily=np.zeros(5),
        borrow_rate_annual=np.zeros(5),
        sic2_industry=np.asarray([f"I{i}" for i in range(5)]),
        nav_dollars=100_000_000.0,
        forecast_volatility_target=0.50,
        maximum_gross=0.50,
        minimum_long_gross=0.10,
        minimum_short_gross=0.10,
        minimum_net=-0.10,
        maximum_net=0.10,
        maximum_position_weight=0.10,
        maximum_one_way_turnover=0.50,
        maximum_one_way_adv_participation=0.10,
        market_beta_absolute_limit=0.10,
        other_factor_absolute_limit=0.10,
        sic2_industry_net_absolute_limit=0.10,
        sic2_industry_gross_limit=0.30,
        spread_multiplier=1.0,
        impact_coefficient=0.10,
        financing_rate_annual=0.0,
    )
    assert abs(result.weights[0]) <= 2e-6
    assert result.long_gross >= 0.10 - 1e-7
    assert result.short_gross >= 0.10 - 1e-7
