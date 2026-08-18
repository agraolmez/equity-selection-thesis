"""V6.1 Step 04 optional-short and explicit-hedge optimizer primitives.

The frozen V6.1 selectors are inputs, not retrained here.  This module only
changes the portfolio decision: stock shorts have no minimum allocation and an
optional SPY hedge is represented separately from alpha-seeking stock shorts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cvxpy as cp
import numpy as np

from portfolio_bot.allocation.costs import cvxpy_total_cost, numpy_total_cost


@dataclass(frozen=True)
class Step04Solution:
    stock_weights: np.ndarray
    spy_weight: float
    status: str
    objective_value: float
    forecast_volatility: float
    factor_exposures: np.ndarray
    stock_transaction_cost: float
    stock_borrow_cost: float
    spy_transaction_cost: float
    spy_borrow_cost: float
    financing_cost: float
    total_cost: float
    turnover: float


def _numeric_risk(
    stock_weights: np.ndarray,
    spy_weight: float,
    factor_betas: np.ndarray,
    covariance_root: np.ndarray,
    specific_variance: np.ndarray,
    spy_factor_exposures: np.ndarray,
    spy_specific_variance: float,
) -> tuple[float, np.ndarray]:
    exposures = factor_betas.T @ stock_weights + spy_factor_exposures * spy_weight
    factor_component = covariance_root @ exposures
    stock_specific = np.sqrt(np.maximum(specific_variance, 0.0)) * stock_weights
    monthly_variance = float(
        factor_component @ factor_component
        + stock_specific @ stock_specific
        + max(float(spy_specific_variance), 0.0) * spy_weight**2
    )
    return float(np.sqrt(max(12.0 * monthly_variance, 0.0))), exposures


def solve_optional_short_portfolio(
    *,
    expected_stock_excess_return: np.ndarray,
    long_candidate: np.ndarray,
    short_candidate: np.ndarray,
    alpha_short_candidate: np.ndarray,
    reduction_only: np.ndarray,
    previous_stock_weights: np.ndarray,
    factor_betas: np.ndarray,
    factor_covariance_root: np.ndarray,
    specific_variance_monthly: np.ndarray,
    full_spread: np.ndarray,
    adv20_usd: np.ndarray,
    impact_volatility_daily: np.ndarray,
    borrow_rate_annual: np.ndarray,
    sic2_industry: np.ndarray,
    nav_dollars: float,
    risk_aversion: float,
    forecast_volatility_limit: float,
    maximum_gross: float,
    minimum_net: float,
    maximum_net: float,
    maximum_position_weight: float,
    maximum_one_way_turnover: float,
    maximum_one_way_adv_participation: float,
    market_beta_absolute_limit: float,
    other_factor_absolute_limit: float,
    sic2_industry_net_absolute_limit: float,
    sic2_industry_gross_limit: float,
    spread_multiplier: float,
    impact_coefficient: float,
    financing_rate_annual: float,
    previous_spy_weight: float = 0.0,
    spy_allowed: bool = False,
    spy_expected_excess_return: float = 0.0,
    spy_factor_exposures: np.ndarray | None = None,
    spy_specific_variance_monthly: float = 0.0,
    maximum_spy_short_gross: float = 0.0,
    spy_transaction_cost_rate: float = 0.0,
    spy_borrow_rate_annual: float = 0.0,
    require_nonnegative_expected_alpha_short_net: bool = False,
    solver: str = "CLARABEL",
) -> Step04Solution:
    """Maximize expected net utility with optional stocks and an explicit SPY hedge.

    The risk penalty is ``risk_aversion / 2 * annualized variance``.  A stock
    short sleeve can therefore be chosen for covariance benefit even when its
    standalone expected contribution is negative.  When requested, the
    alpha-short subset must have nonnegative *expected* net contribution; this
    never conditions on its future realized return.
    """

    mu = np.asarray(expected_stock_excess_return, dtype=np.float64)
    previous = np.asarray(previous_stock_weights, dtype=np.float64)
    long_mask = np.asarray(long_candidate, dtype=bool)
    short_mask = np.asarray(short_candidate, dtype=bool)
    alpha_short = np.asarray(alpha_short_candidate, dtype=bool)
    reduction = np.asarray(reduction_only, dtype=bool)
    betas = np.asarray(factor_betas, dtype=np.float64)
    specific = np.asarray(specific_variance_monthly, dtype=np.float64)
    spy_betas = np.asarray(
        np.r_[1.0, np.zeros(betas.shape[1] - 1)]
        if spy_factor_exposures is None
        else spy_factor_exposures,
        dtype=np.float64,
    )
    n = len(mu)
    arrays = (
        previous,
        long_mask,
        short_mask,
        alpha_short,
        reduction,
        specific,
        full_spread,
        adv20_usd,
        impact_volatility_daily,
        borrow_rate_annual,
        sic2_industry,
    )
    if any(len(value) != n for value in arrays) or betas.shape[0] != n:
        raise ValueError("Step 04 optimizer input lengths differ")
    if betas.ndim != 2 or spy_betas.shape != (betas.shape[1],):
        raise ValueError("factor exposure dimensions differ")
    if np.any(long_mask & short_mask) or np.any(alpha_short & ~short_mask):
        raise ValueError("candidate-side masks are inconsistent")
    if not np.isfinite(mu).all() or not np.isfinite(previous).all():
        raise ValueError("expected returns and previous weights must be finite")
    if risk_aversion < 0.0 or maximum_spy_short_gross < 0.0:
        raise ValueError("risk aversion and SPY limit must be nonnegative")

    w = cp.Variable(n)
    h = cp.Variable()
    constraints: list[cp.Constraint] = []
    long_index = np.flatnonzero(long_mask)
    short_index = np.flatnonzero(short_mask)
    unused = ~(long_mask | short_mask | reduction)
    if unused.any():
        constraints.append(w[np.flatnonzero(unused)] == 0.0)
    if long_index.size:
        constraints.extend([w[long_index] >= 0.0, w[long_index] <= maximum_position_weight])
    if short_index.size:
        constraints.extend([w[short_index] <= 0.0, w[short_index] >= -maximum_position_weight])
    for index in np.flatnonzero(reduction):
        if previous[index] > 0.0:
            constraints.extend([w[index] >= 0.0, w[index] <= previous[index]])
        elif previous[index] < 0.0:
            constraints.extend([w[index] <= 0.0, w[index] >= previous[index]])
        else:
            constraints.append(w[index] == 0.0)

    if spy_allowed:
        constraints.extend([h <= 0.0, h >= -float(maximum_spy_short_gross)])
    else:
        constraints.append(h == 0.0)

    delta = w - previous
    spy_delta = h - float(previous_spy_weight)
    valid_adv = np.isfinite(adv20_usd) & (np.asarray(adv20_usd) > 0.0)
    trade_cap = np.where(
        valid_adv,
        maximum_one_way_adv_participation * np.asarray(adv20_usd, dtype=float) / float(nav_dollars),
        0.0,
    )
    constraints.extend(
        [
            cp.abs(delta) <= trade_cap,
            0.5 * (cp.norm1(delta) + cp.abs(spy_delta)) <= maximum_one_way_turnover,
        ]
    )

    long_side = long_mask | (reduction & (previous > 0.0))
    short_side = short_mask | (reduction & (previous < 0.0))
    long_gross = cp.sum(w[np.flatnonzero(long_side)]) if long_side.any() else cp.Constant(0.0)
    short_gross = -cp.sum(w[np.flatnonzero(short_side)]) if short_side.any() else cp.Constant(0.0)
    total_gross = long_gross + short_gross - h
    total_net = cp.sum(w) + h
    constraints.extend(
        [
            total_gross <= maximum_gross,
            total_net >= minimum_net,
            total_net <= maximum_net,
        ]
    )

    exposures = betas.T @ w + spy_betas * h
    constraints.extend(
        [
            cp.abs(exposures[0]) <= market_beta_absolute_limit,
            cp.abs(exposures[1:]) <= other_factor_absolute_limit,
        ]
    )
    factor_component = factor_covariance_root @ exposures
    stock_specific = cp.multiply(np.sqrt(np.maximum(specific, 0.0)), w)
    monthly_variance = (
        cp.sum_squares(factor_component)
        + cp.sum_squares(stock_specific)
        + max(float(spy_specific_variance_monthly), 0.0) * cp.square(h)
    )
    annual_variance = 12.0 * monthly_variance
    constraints.append(annual_variance <= float(forecast_volatility_limit) ** 2)

    industries = np.asarray(sic2_industry, dtype=str)
    for industry in sorted(set(industries.tolist())):
        index = np.flatnonzero(industries == industry)
        if index.size:
            constraints.extend(
                [
                    cp.abs(cp.sum(w[index])) <= sic2_industry_net_absolute_limit,
                    cp.sum(cp.abs(w[index])) <= sic2_industry_gross_limit,
                ]
            )

    stock_cost = cvxpy_total_cost(
        w,
        previous,
        full_spread,
        adv20_usd,
        impact_volatility_daily,
        borrow_rate_annual,
        nav_dollars,
        impact_coefficient=impact_coefficient,
        spread_multiplier=spread_multiplier,
        financing_rate_annual=0.0,
    )
    constraints.extend(stock_cost.constraints)
    spy_transaction = float(spy_transaction_cost_rate) * cp.abs(spy_delta)
    spy_borrow = float(spy_borrow_rate_annual) / 12.0 * (-h)
    financing = float(financing_rate_annual) / 12.0 * cp.pos(total_gross - 1.0)
    total_cost = stock_cost.transaction + stock_cost.borrow + spy_transaction + spy_borrow + financing

    alpha_index = np.flatnonzero(alpha_short)
    if require_nonnegative_expected_alpha_short_net and alpha_index.size:
        alpha_cost = cvxpy_total_cost(
            w[alpha_index],
            previous[alpha_index],
            np.asarray(full_spread)[alpha_index],
            np.asarray(adv20_usd)[alpha_index],
            np.asarray(impact_volatility_daily)[alpha_index],
            np.asarray(borrow_rate_annual)[alpha_index],
            nav_dollars,
            impact_coefficient=impact_coefficient,
            spread_multiplier=spread_multiplier,
            financing_rate_annual=0.0,
        )
        constraints.extend(alpha_cost.constraints)
        constraints.append(mu[alpha_index] @ w[alpha_index] >= alpha_cost.total)

    settings: Mapping[str, float | int | bool] = {
        "max_iter": 1000,
        "tol_gap_abs": 1e-8,
        "tol_gap_rel": 1e-8,
        "tol_feas": 1e-8,
        "equilibrate_enable": True,
    }
    stale_long = np.flatnonzero(reduction & (previous > 0.0))
    stale_short = np.flatnonzero(reduction & (previous < 0.0))
    if stale_long.size or stale_short.size:
        stale_gross = cp.Constant(0.0)
        if stale_long.size:
            stale_gross += cp.sum(w[stale_long])
        if stale_short.size:
            stale_gross -= cp.sum(w[stale_short])
        repair = cp.Problem(cp.Minimize(stale_gross), constraints)
        repair.solve(solver=solver, **settings)
        if repair.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or w.value is None:
            raise RuntimeError(f"Step 04 stale-position repair failed with status {repair.status}")
        repaired = np.asarray(w.value, dtype=float).reshape(-1)
        minimum = float(repaired[stale_long].sum() - repaired[stale_short].sum())
        constraints.append(stale_gross <= max(minimum, 0.0) + 1e-6)

    expected_net = mu @ w + float(spy_expected_excess_return) * h - total_cost
    problem = cp.Problem(
        cp.Maximize(10_000.0 * (expected_net - 0.5 * float(risk_aversion) * annual_variance)),
        constraints,
    )
    problem.solve(solver=solver, **settings)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or w.value is None or h.value is None:
        raise RuntimeError(f"Step 04 optimizer failed with status {problem.status}")

    weights = np.asarray(w.value, dtype=np.float64).reshape(-1)
    weights[np.abs(weights) < 1e-10] = 0.0
    spy_weight = float(h.value)
    if abs(spy_weight) < 1e-10:
        spy_weight = 0.0
    stock_numeric = numpy_total_cost(
        weights,
        previous,
        full_spread,
        adv20_usd,
        impact_volatility_daily,
        borrow_rate_annual,
        nav_dollars,
        impact_coefficient=impact_coefficient,
        spread_multiplier=spread_multiplier,
        financing_rate_annual=0.0,
    )
    spy_transaction_numeric = float(spy_transaction_cost_rate) * abs(spy_weight - previous_spy_weight)
    spy_borrow_numeric = float(spy_borrow_rate_annual) / 12.0 * max(-spy_weight, 0.0)
    long_numeric = float(np.maximum(weights, 0.0).sum())
    short_numeric = float(np.maximum(-weights, 0.0).sum())
    gross_numeric = long_numeric + short_numeric + abs(spy_weight)
    financing_numeric = max(gross_numeric - 1.0, 0.0) * float(financing_rate_annual) / 12.0
    total_numeric = (
        stock_numeric.transaction
        + stock_numeric.borrow
        + spy_transaction_numeric
        + spy_borrow_numeric
        + financing_numeric
    )
    risk, numeric_exposures = _numeric_risk(
        weights,
        spy_weight,
        betas,
        factor_covariance_root,
        specific,
        spy_betas,
        spy_specific_variance_monthly,
    )
    return Step04Solution(
        stock_weights=weights,
        spy_weight=spy_weight,
        status=str(problem.status),
        objective_value=float(mu @ weights + spy_expected_excess_return * spy_weight - total_numeric),
        forecast_volatility=risk,
        factor_exposures=numeric_exposures,
        stock_transaction_cost=stock_numeric.transaction,
        stock_borrow_cost=stock_numeric.borrow,
        spy_transaction_cost=spy_transaction_numeric,
        spy_borrow_cost=spy_borrow_numeric,
        financing_cost=financing_numeric,
        total_cost=total_numeric,
        turnover=0.5 * (float(np.abs(weights - previous).sum()) + abs(spy_weight - previous_spy_weight)),
    )


__all__ = ["Step04Solution", "solve_optional_short_portfolio"]
