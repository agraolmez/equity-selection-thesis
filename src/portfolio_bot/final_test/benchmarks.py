"""Frozen Stage 13 benchmark signals and calendar-alignment plumbing.

This module contains no data access and no optimizer.  It produces the two
traditional expected-return signals, passive/cash monthly series aligned to an
exact final-period calendar, and deterministic 100%-gross equal-weight targets
for the common downstream execution and cost engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from portfolio_bot.final_test.calendar import (
    FinalCalendarError,
    validate_final_month_sequence,
)
from portfolio_bot.selectors.metrics import centered_average_rank


TRAILING_RESIDUAL_WINDOW_MONTHS = 60
TRAILING_RESIDUAL_MINIMUM_MONTHS = 36
TRAILING_RESIDUAL_PRIOR_STRENGTH = 60.0

VALUE_FEATURES = (
    "book_to_market",
    "earnings_to_price",
    "operating_cash_flow_to_price",
)
QUALITY_POSITIVE_FEATURES = (
    "gross_profitability",
    "return_on_assets",
)
QUALITY_NEGATIVE_FEATURES = (
    "accruals_to_average_assets",
    "total_debt_to_assets",
)
MOMENTUM_FEATURES = (
    "momentum_6_1",
    "momentum_12_1",
)
VQM_FEATURES = frozenset(
    VALUE_FEATURES
    + QUALITY_POSITIVE_FEATURES
    + QUALITY_NEGATIVE_FEATURES
    + MOMENTUM_FEATURES
)


class BenchmarkError(ValueError):
    """Raised when benchmark inputs violate a frozen definition."""


class BenchmarkId(str, Enum):
    CONDITIONAL_ELASTIC_NET_IDENTICAL_DOWNSTREAM_SYSTEM = (
        "conditional_elastic_net_identical_downstream_system"
    )
    TRAILING_RESIDUAL_MEAN_MARKOWITZ = "trailing_residual_mean_markowitz"
    FIXED_VALUE_QUALITY_MOMENTUM = "fixed_value_quality_momentum"
    RANK_ALLOCATOR = "rank_allocator"
    DECISION_FOCUSED_ALLOCATOR = "decision_focused_allocator"
    CASH_OR_ONE_MONTH_TBILL = "cash_or_one_month_tbill"
    SPY_TOTAL_RETURN = "spy_total_return"
    QQQ_TOTAL_RETURN = "qqq_total_return"
    EQUAL_WEIGHT_LONG_ONLY_SCORING_UNIVERSE = "equal_weight_long_only_scoring_universe"


CONFIRMATORY_BENCHMARK_IDS = frozenset(
    {BenchmarkId.CONDITIONAL_ELASTIC_NET_IDENTICAL_DOWNSTREAM_SYSTEM}
)
SECONDARY_STRATEGY_COMPARATOR_IDS = frozenset(
    {
        BenchmarkId.TRAILING_RESIDUAL_MEAN_MARKOWITZ,
        BenchmarkId.FIXED_VALUE_QUALITY_MOMENTUM,
        BenchmarkId.RANK_ALLOCATOR,
        BenchmarkId.DECISION_FOCUSED_ALLOCATOR,
    }
)
DESCRIPTIVE_BENCHMARK_IDS = frozenset(
    {
        BenchmarkId.CASH_OR_ONE_MONTH_TBILL,
        BenchmarkId.SPY_TOTAL_RETURN,
        BenchmarkId.QQQ_TOTAL_RETURN,
        BenchmarkId.EQUAL_WEIGHT_LONG_ONLY_SCORING_UNIVERSE,
    }
)
ACTIVE_COST_MODEL_BENCHMARK_IDS = frozenset(
    CONFIRMATORY_BENCHMARK_IDS
    | SECONDARY_STRATEGY_COMPARATOR_IDS
    | {BenchmarkId.EQUAL_WEIGHT_LONG_ONLY_SCORING_UNIVERSE}
)
NO_ACTIVE_COST_MODEL_BENCHMARK_IDS = frozenset(
    {
        BenchmarkId.CASH_OR_ONE_MONTH_TBILL,
        BenchmarkId.SPY_TOTAL_RETURN,
        BenchmarkId.QQQ_TOTAL_RETURN,
    }
)

if (
    ACTIVE_COST_MODEL_BENCHMARK_IDS | NO_ACTIVE_COST_MODEL_BENCHMARK_IDS
    != frozenset(BenchmarkId)
    or ACTIVE_COST_MODEL_BENCHMARK_IDS & NO_ACTIVE_COST_MODEL_BENCHMARK_IDS
):
    raise RuntimeError("internal benchmark cost-treatment partition differs")


def _numeric_with_missing(
    values: Sequence[object],
    *,
    field_name: str,
) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise BenchmarkError(f"{field_name} must be a numeric sequence")
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BenchmarkError(f"{field_name} must be numeric") from exc
    if vector.ndim != 1:
        raise BenchmarkError(f"{field_name} must be one-dimensional")
    if np.isinf(vector).any():
        raise BenchmarkError(f"{field_name} cannot contain infinite values")
    return vector


def _finite_vector(
    values: Sequence[object],
    *,
    field_name: str,
) -> np.ndarray:
    vector = _numeric_with_missing(values, field_name=field_name)
    if not np.isfinite(vector).all():
        raise BenchmarkError(f"{field_name} must contain only finite values")
    return vector


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _validate_benchmark_months(values: Sequence[str]) -> tuple[str, ...]:
    try:
        return validate_final_month_sequence(values)
    except FinalCalendarError as exc:
        raise BenchmarkError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class TrailingResidualMeanEstimate:
    """Auditable one-security output of the frozen shrinkage rule."""

    observations_used: int
    sample_mean: float | None
    shrinkage_weight: float
    expected_return: float | None

    @property
    def available(self) -> bool:
        return self.expected_return is not None


def trailing_residual_mean_estimate(
    completed_monthly_residual_returns: Sequence[object],
) -> TrailingResidualMeanEstimate:
    """Use the latest 60 monthly slots, require 36, and shrink by ``n/(n+60)``."""

    vector = _numeric_with_missing(
        completed_monthly_residual_returns,
        field_name="completed_monthly_residual_returns",
    )
    trailing = vector[-TRAILING_RESIDUAL_WINDOW_MONTHS:]
    finite = trailing[np.isfinite(trailing)]
    observations = int(finite.size)
    shrinkage_weight = observations / (observations + TRAILING_RESIDUAL_PRIOR_STRENGTH)
    sample_mean = (
        None if observations == 0 else float(np.mean(finite, dtype=np.float64))
    )
    expected_return = (
        None
        if observations < TRAILING_RESIDUAL_MINIMUM_MONTHS
        else float(shrinkage_weight * sample_mean)  # type: ignore[operator]
    )
    return TrailingResidualMeanEstimate(
        observations_used=observations,
        sample_mean=sample_mean,
        shrinkage_weight=float(shrinkage_weight),
        expected_return=expected_return,
    )


def trailing_residual_mean_cross_section(
    completed_returns_by_security: Mapping[object, Sequence[object]],
) -> Mapping[object, TrailingResidualMeanEstimate]:
    """Apply the same frozen rule independently to each registered security."""

    if not isinstance(completed_returns_by_security, Mapping):
        raise BenchmarkError("completed_returns_by_security must be a mapping")
    return MappingProxyType(
        {
            security: trailing_residual_mean_estimate(values)
            for security, values in completed_returns_by_security.items()
        }
    )


@dataclass(frozen=True, slots=True)
class ValueQualityMomentumScores:
    """Sleeve receipts, untuned raw composite, and one centered rank transform."""

    value_sleeve: np.ndarray
    quality_sleeve: np.ndarray
    momentum_sleeve: np.ndarray
    raw_composite: np.ndarray
    centered_score: np.ndarray


def value_quality_momentum_scores(
    centered_rank_features: Mapping[str, Sequence[object]],
) -> ValueQualityMomentumScores:
    """Compute the fixed 3/4/2-component VQM definition.

    Inputs are the already preprocessed centered ranks.  Missing values are
    replaced by the common neutral zero before the fixed equal-weight sleeve
    averages.  The three sleeves are then averaged equally and ranked once.
    """

    if not isinstance(centered_rank_features, Mapping):
        raise BenchmarkError("centered_rank_features must be a mapping")
    actual = frozenset(centered_rank_features)
    if actual != VQM_FEATURES:
        missing = sorted(VQM_FEATURES - actual)
        extra = sorted(actual - VQM_FEATURES)
        raise BenchmarkError(
            f"VQM feature set differs; missing={missing}, extra={extra}"
        )
    normalized: dict[str, np.ndarray] = {}
    row_count: int | None = None
    for feature_name in sorted(VQM_FEATURES):
        values = _numeric_with_missing(
            centered_rank_features[feature_name],
            field_name=feature_name,
        )
        if row_count is None:
            row_count = int(values.size)
            if row_count == 0:
                raise BenchmarkError("VQM cross-section cannot be empty")
        elif values.size != row_count:
            raise BenchmarkError("all VQM feature vectors must have equal length")
        finite = np.isfinite(values)
        if np.any(np.abs(values[finite]) > 0.5):
            raise BenchmarkError(f"{feature_name} is not a centered-rank input")
        normalized[feature_name] = np.where(finite, values, 0.0)

    value = sum(normalized[name] for name in VALUE_FEATURES) / 3.0
    quality = (
        sum(normalized[name] for name in QUALITY_POSITIVE_FEATURES)
        - sum(normalized[name] for name in QUALITY_NEGATIVE_FEATURES)
    ) / 4.0
    momentum = sum(normalized[name] for name in MOMENTUM_FEATURES) / 2.0
    raw = (value + quality + momentum) / 3.0
    centered = centered_average_rank(raw)
    return ValueQualityMomentumScores(
        value_sleeve=_readonly(value),
        quality_sleeve=_readonly(quality),
        momentum_sleeve=_readonly(momentum),
        raw_composite=_readonly(raw),
        centered_score=_readonly(centered),
    )


@dataclass(frozen=True, slots=True)
class AlignedMonthlyBenchmark:
    """One finite benchmark return vector on an exact contiguous month domain."""

    benchmark_id: BenchmarkId
    holding_months: tuple[str, ...]
    total_returns: np.ndarray
    charged_active_cost_model: bool

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_id, BenchmarkId):
            raise BenchmarkError("benchmark_id must be a BenchmarkId")
        months = _validate_benchmark_months(self.holding_months)
        returns = _finite_vector(
            self.total_returns,
            field_name="total_returns",
        )
        if returns.size != len(months):
            raise BenchmarkError("benchmark return count differs from month count")
        if type(self.charged_active_cost_model) is not bool:
            raise BenchmarkError("charged_active_cost_model must be boolean")
        if (
            self.benchmark_id in NO_ACTIVE_COST_MODEL_BENCHMARK_IDS
            and self.charged_active_cost_model
        ):
            raise BenchmarkError(
                "cash and passive ETF series cannot receive active turnover costs"
            )
        if (
            self.benchmark_id in ACTIVE_COST_MODEL_BENCHMARK_IDS
            and not self.charged_active_cost_model
        ):
            raise BenchmarkError(
                "strategy comparators and equal-weight returns require the active "
                "cost model"
            )
        object.__setattr__(self, "holding_months", months)
        object.__setattr__(self, "total_returns", _readonly(returns))


def align_benchmark_returns(
    *,
    benchmark_id: BenchmarkId,
    holding_months: Sequence[str],
    returns_by_month: Mapping[str, object],
    charged_active_cost_model: bool,
) -> AlignedMonthlyBenchmark:
    """Align by exact month keys; missing, extra, or nonfinite values fail."""

    if not isinstance(benchmark_id, BenchmarkId):
        raise BenchmarkError("benchmark_id must be a BenchmarkId")
    months = _validate_benchmark_months(holding_months)
    if not isinstance(returns_by_month, Mapping):
        raise BenchmarkError("returns_by_month must be a mapping")
    expected = frozenset(months)
    actual = frozenset(returns_by_month)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BenchmarkError(
            f"benchmark month keys differ; missing={missing}, extra={extra}"
        )
    returns = _finite_vector(
        [returns_by_month[month] for month in months],
        field_name="returns_by_month",
    )
    return AlignedMonthlyBenchmark(
        benchmark_id=benchmark_id,
        holding_months=months,
        total_returns=returns,
        charged_active_cost_model=charged_active_cost_model,
    )


def cash_benchmark(
    *,
    holding_months: Sequence[str],
    one_month_tbill_returns: Mapping[str, object],
) -> AlignedMonthlyBenchmark:
    """Use the aligned one-month Treasury-bill/risk-free return without costs."""

    return align_benchmark_returns(
        benchmark_id=BenchmarkId.CASH_OR_ONE_MONTH_TBILL,
        holding_months=holding_months,
        returns_by_month=one_month_tbill_returns,
        charged_active_cost_model=False,
    )


def passive_etf_benchmark(
    *,
    benchmark_id: BenchmarkId,
    holding_months: Sequence[str],
    total_returns_by_month: Mapping[str, object],
) -> AlignedMonthlyBenchmark:
    """Align SPY or QQQ total returns without active-strategy turnover costs."""

    if benchmark_id not in {
        BenchmarkId.SPY_TOTAL_RETURN,
        BenchmarkId.QQQ_TOTAL_RETURN,
    }:
        raise BenchmarkError("passive ETF benchmark must be SPY or QQQ")
    return align_benchmark_returns(
        benchmark_id=benchmark_id,
        holding_months=holding_months,
        returns_by_month=total_returns_by_month,
        charged_active_cost_model=False,
    )


def equal_weight_long_only_targets(
    security_ids: Sequence[object],
    *,
    eligible: Sequence[bool] | None = None,
) -> np.ndarray:
    """Return deterministic 100%-gross targets for the common cost engine."""

    if isinstance(security_ids, (str, bytes)):
        raise BenchmarkError("security_ids must be a sequence")
    identifiers = tuple(security_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise BenchmarkError("security_ids must be nonempty and unique")
    if eligible is None:
        mask = np.ones(len(identifiers), dtype=bool)
    else:
        mask = np.asarray(eligible)
        if mask.ndim != 1 or mask.size != len(identifiers):
            raise BenchmarkError("eligible must match security_ids")
        if mask.dtype != np.bool_:
            if any(type(value) is not bool for value in eligible):
                raise BenchmarkError("eligible must contain booleans")
            mask = mask.astype(bool)
    count = int(np.sum(mask))
    if count == 0:
        raise BenchmarkError("equal-weight universe has no eligible securities")
    weights = np.zeros(len(identifiers), dtype=np.float64)
    weights[mask] = 1.0 / count
    return _readonly(weights)


@dataclass(frozen=True, slots=True)
class AlignedBenchmarkPanel:
    """Joint monthly benchmark rows suitable for paired inference plumbing."""

    holding_months: tuple[str, ...]
    series: Mapping[BenchmarkId, np.ndarray]


def aligned_benchmark_panel(
    series: Sequence[AlignedMonthlyBenchmark],
) -> AlignedBenchmarkPanel:
    """Require all benchmark series to share one exact ordered month axis."""

    if not series:
        raise BenchmarkError("benchmark panel cannot be empty")
    months = series[0].holding_months
    by_id: dict[BenchmarkId, np.ndarray] = {}
    for item in series:
        if not isinstance(item, AlignedMonthlyBenchmark):
            raise BenchmarkError(
                "benchmark panel members must be AlignedMonthlyBenchmark objects"
            )
        if item.holding_months != months:
            raise BenchmarkError("benchmark panel month axes differ")
        if item.benchmark_id in by_id:
            raise BenchmarkError("benchmark panel contains a duplicate identity")
        by_id[item.benchmark_id] = item.total_returns
    return AlignedBenchmarkPanel(
        holding_months=months,
        series=MappingProxyType(by_id),
    )


__all__ = [
    "AlignedBenchmarkPanel",
    "AlignedMonthlyBenchmark",
    "ACTIVE_COST_MODEL_BENCHMARK_IDS",
    "BenchmarkError",
    "BenchmarkId",
    "CONFIRMATORY_BENCHMARK_IDS",
    "DESCRIPTIVE_BENCHMARK_IDS",
    "MOMENTUM_FEATURES",
    "NO_ACTIVE_COST_MODEL_BENCHMARK_IDS",
    "QUALITY_NEGATIVE_FEATURES",
    "QUALITY_POSITIVE_FEATURES",
    "SECONDARY_STRATEGY_COMPARATOR_IDS",
    "TRAILING_RESIDUAL_MINIMUM_MONTHS",
    "TRAILING_RESIDUAL_PRIOR_STRENGTH",
    "TRAILING_RESIDUAL_WINDOW_MONTHS",
    "TrailingResidualMeanEstimate",
    "VALUE_FEATURES",
    "VQM_FEATURES",
    "ValueQualityMomentumScores",
    "align_benchmark_returns",
    "aligned_benchmark_panel",
    "cash_benchmark",
    "equal_weight_long_only_targets",
    "passive_etf_benchmark",
    "trailing_residual_mean_cross_section",
    "trailing_residual_mean_estimate",
    "value_quality_momentum_scores",
]
