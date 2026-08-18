"""Frozen rank/volatility challenger for Stage 10."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from portfolio_bot.allocation.domain import AllocationProblemData
from portfolio_bot.allocation.solver import AllocationResult, solve_projection


@dataclass(frozen=True)
class RankRawPortfolio:
    weights: np.ndarray
    long_strength_sum: float
    short_strength_sum: float
    unscaled_augmented_annual_volatility: float
    scale: float

    def __post_init__(self) -> None:
        vector = np.asarray(self.weights, dtype=np.float64).copy()
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("rank raw weights must be a finite vector")
        vector.setflags(write=False)
        object.__setattr__(self, "weights", vector)


def form_rank_raw_portfolio(
    data: AllocationProblemData,
    *,
    long_entry_boundary: float | None,
    short_entry_boundary: float | None,
) -> RankRawPortfolio:
    """Build equal-notional raw legs and scale them toward the 10% risk budget."""

    # An insufficient-candidate month is represented by a genuine zero-row
    # problem.  Such a month has no empirical tail boundaries, and the common
    # solver's required result is deterministic all-cash.  Do not manufacture
    # arbitrary boundary values merely to reach that explicit path.
    if data.n_assets == 0:
        return RankRawPortfolio(
            weights=np.empty(0, dtype=np.float64),
            long_strength_sum=0.0,
            short_strength_sum=0.0,
            unscaled_augmented_annual_volatility=0.0,
            scale=0.0,
        )
    if long_entry_boundary is None or short_entry_boundary is None:
        raise ValueError("nonempty rank allocation requires both entry boundaries")
    long_boundary = float(long_entry_boundary)
    short_boundary = float(short_entry_boundary)
    if not np.isfinite(long_boundary) or not np.isfinite(short_boundary):
        raise ValueError("rank entry boundaries must be finite")
    if short_boundary > long_boundary:
        raise ValueError("short boundary cannot exceed long boundary")

    active = ~data.reduction_only
    long_strength = np.where(
        data.long_mask & active,
        np.maximum(data.score - long_boundary, 0.0) / data.forecast_volatility,
        0.0,
    )
    short_strength = np.where(
        data.short_mask & active,
        np.maximum(short_boundary - data.score, 0.0) / data.forecast_volatility,
        0.0,
    )
    long_sum = float(np.sum(long_strength))
    short_sum = float(np.sum(short_strength))
    raw = np.zeros(data.n_assets, dtype=np.float64)
    if long_sum <= 0.0 or short_sum <= 0.0:
        return RankRawPortfolio(
            weights=raw,
            long_strength_sum=long_sum,
            short_strength_sum=short_sum,
            unscaled_augmented_annual_volatility=0.0,
            scale=0.0,
        )
    raw += long_strength / long_sum
    raw -= short_strength / short_sum
    unscaled = float(np.sqrt(12.0) * np.linalg.norm(data.risk_root @ raw))
    if not np.isfinite(unscaled):
        raise ValueError("rank raw risk is nonfinite")
    scale = 1.0 if unscaled <= 0.0 else min(1.0, 0.10 / unscaled)
    raw *= scale
    return RankRawPortfolio(
        weights=raw,
        long_strength_sum=long_sum,
        short_strength_sum=short_sum,
        unscaled_augmented_annual_volatility=unscaled,
        scale=scale,
    )


def allocate_rank(
    data: AllocationProblemData,
    *,
    long_entry_boundary: float | None,
    short_entry_boundary: float | None,
) -> tuple[RankRawPortfolio, AllocationResult]:
    raw = form_rank_raw_portfolio(
        data,
        long_entry_boundary=long_entry_boundary,
        short_entry_boundary=short_entry_boundary,
    )
    result = solve_projection(
        data,
        raw.weights,
        architecture="rank_volatility_projected",
    )
    return raw, result


__all__ = ["RankRawPortfolio", "allocate_rank", "form_rank_raw_portfolio"]
