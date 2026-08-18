"""Deterministic fictitious inputs used only to verify Stage 10 machinery."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from portfolio_bot.allocation.domain import (
    PRIMARY_BORROW_SCENARIO,
    AllocationProblemData,
)


def balanced_synthetic_problem(
    *,
    zero_alpha: bool = False,
    inherited_violation: bool = False,
) -> AllocationProblemData:
    """Return a small non-empirical long/short fixture with all risk families."""

    n = 8
    side = np.asarray([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.int8)
    betas = np.asarray(
        [
            [1.00, 0.20, 0.10, 0.00, 0.00, 0.10],
            [0.90, -0.10, 0.00, 0.10, 0.00, 0.00],
            [1.10, 0.00, -0.10, 0.00, 0.10, -0.10],
            [0.80, 0.10, 0.00, -0.10, 0.00, 0.00],
            [1.00, 0.20, 0.10, 0.00, 0.00, 0.10],
            [0.90, -0.10, 0.00, 0.10, 0.00, 0.00],
            [1.10, 0.00, -0.10, 0.00, 0.10, -0.10],
            [0.80, 0.10, 0.00, -0.10, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    factor_covariance = np.diag(
        np.asarray([0.0020, 0.0004, 0.0003, 0.0002, 0.0002, 0.0005])
    )
    specific = np.full(n, 0.004, dtype=np.float64)
    forecast_volatility = np.sqrt(
        np.einsum("ij,jk,ik->i", betas, factor_covariance, betas) + specific
    )
    previous = np.zeros(n, dtype=np.float64)
    reduction = np.zeros(n, dtype=np.bool_)
    if inherited_violation:
        previous[[0, 4]] = [0.0200, -0.0200]
        reduction[[0, 4]] = True
    expected = np.zeros(n, dtype=np.float64)
    if not zero_alpha:
        expected = np.asarray(
            [0.030, 0.028, 0.026, 0.024, -0.030, -0.028, -0.026, -0.024],
            dtype=np.float64,
        )
    capacity = np.full(n, 0.0200, dtype=np.float64)
    if inherited_violation:
        capacity[[0, 4]] = 0.0010
    return AllocationProblemData(
        permno=np.arange(900001, 900001 + n, dtype=np.int64),
        # Two share classes on each side exercise issuer caps.
        permco=np.asarray(
            [810001, 810001, 810003, 810004, 810005, 810005, 810007, 810008]
        ),
        side=side,
        previous_weight=previous,
        reduction_only=reduction,
        required_exit=np.zeros(n, dtype=np.bool_),
        expected_return=expected,
        score=np.asarray([0.45, 0.32, 0.25, 0.20, -0.45, -0.32, -0.25, -0.20]),
        factor_betas=betas,
        factor_covariance_monthly=factor_covariance,
        specific_variance_monthly=specific,
        forecast_volatility=forecast_volatility,
        full_spread=np.full(n, 0.0020, dtype=np.float64),
        adv20_usd=np.full(n, 250_000_000.0, dtype=np.float64),
        impact_volatility_daily=np.full(n, 0.020, dtype=np.float64),
        borrow_liquidity_quintile=np.asarray([0, 0, 0, 0, 1, 2, 3, 4]),
        borrow_scenario=PRIMARY_BORROW_SCENARIO,
        trade_capacity_weight=capacity,
        sector=np.asarray(
            [
                "TECH",
                "TECH",
                "HEALTH",
                "INDUSTRIAL",
                "TECH",
                "TECH",
                "HEALTH",
                "INDUSTRIAL",
            ]
        ),
        country=np.asarray(
            [
                "USA",
                "USA",
                "NLD",
                "UNKNOWN_COUNTRY",
                "USA",
                "USA",
                "NLD",
                "UNKNOWN_FOREIGN",
            ]
        ),
        nav_dollars=100_000_000.0,
        market_variance_monthly=float(factor_covariance[0, 0]),
    )


def with_expected_return(
    data: AllocationProblemData,
    expected_return: np.ndarray,
) -> AllocationProblemData:
    """Dataclass-safe convenience for synthetic tests."""

    return replace(data, expected_return=expected_return)


__all__ = ["balanced_synthetic_problem", "with_expected_return"]
