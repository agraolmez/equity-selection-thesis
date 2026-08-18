"""Closed, data-independent economic kernel for the Stage 13 final test.

This module freezes the economic choices that Stage 13 is allowed to execute
and supplies pure in-memory adapters into the already certified Stage 10/11
portfolio mathematics.  It performs no filesystem or database access and
accepts no paths, readers, callbacks, dates outside the frozen calendar, or
model/configuration overrides.

The primary full-feature book and the continuously maintained reduced-core
book are distinct.  The latter is shadow-run through August 2025, becomes the
genesis of the separately reported September--November core extension, and is
carried into descriptive partial December.  Return-chain identities change at
both reporting boundaries, so primary, core, and partial P&L cannot be spliced.
"""

from __future__ import annotations

import calendar as _calendar
import datetime as dt
import multiprocessing as mp
import hashlib
import json
import math
import re
import time
import traceback
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from queue import Empty
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import HuberRegressor

from portfolio_bot.allocation.rank import allocate_rank
from portfolio_bot.allocation.solver import solve_hybrid
from portfolio_bot.allocation.costs import (
    COMMISSION_RATE,
    PRIMARY_FINANCING_ANNUAL_RATE,
)
from portfolio_bot.final_test.benchmarks import (
    VQM_FEATURES,
    equal_weight_long_only_targets,
    value_quality_momentum_scores,
)
from portfolio_bot.final_test.calendar import (
    CORE_COMPLETE_HOLDING_MONTHS,
    FROZEN_HOLDING_INTERVAL_ANCHORS,
    PARTIAL_DECEMBER_HOLDING_MONTH,
    PRIMARY_FULL_HOLDING_MONTHS,
    FinalPeriodSegment,
    final_holding_month_descriptor,
    shift_month,
)
from portfolio_bot.final_test.input_materialization import (
    FinalDailyAccountingProjection,
)
from portfolio_bot.final_test.terminal_event_sensitivity import (
    TerminalEventSource,
    TerminalEventStrategyObservation,
)
from portfolio_bot.numerical_materiality import (
    AGGREGATE_DOLLAR_MATERIALITY_LIMIT,
    NUMERICAL_MATERIALITY_POLICY_ID,
    STRICT_DOLLAR_MATERIALITY,
    NumericalMaterialityAudit,
    audit_strategy_month_residue_dollars,
)
from portfolio_bot.selectors.metrics import centered_average_rank
from portfolio_bot.validation.accounting import (
    PortfolioState,
    apply_daily_accounting,
    rebalance_portfolio,
)
from portfolio_bot.validation.corporate_actions import (
    CASH_SETTLEMENT,
    CorporateActionLedger,
)
from portfolio_bot.validation.daily_engine import (
    DatedHoldingPeriodAccounting,
    MarkContinuityLedger,
    run_dated_holding_period,
)
from portfolio_bot.validation.problem_builder import (
    AuthenticatedMonthlyIdentity,
    InheritedHoldings,
    MarketSnapshot,
    MonthlyRiskInputs,
    ScoredSnapshot,
    build_monthly_problem,
)
from portfolio_bot.validation.session_authority import (
    HoldingPeriodSessionAuthority,
)


FINAL_ECONOMIC_KERNEL_ID = "stage13-final-economic-kernel-v2"
FINAL_ECONOMIC_KERNEL_SCHEMA_VERSION = 1

NEURAL_NETWORK_CONFIG_ID = "nn_001"
XGBOOST_CONFIG_ID = "xgb_core_038_g10"
ELASTIC_NET_CONFIG_ID = "enet_r004_l005"
FINAL_STOCHASTIC_SEEDS = (11, 23, 47, 71, 101)

PRIMARY_STRATEGY_ID = "ensemble_hybrid_primary"
ELASTIC_NET_STRATEGY_ID = "elastic_net_hybrid_challenger"
RANK_STRATEGY_ID = "ensemble_rank_challenger"
TRAILING_RESIDUAL_STRATEGY_ID = "trailing_residual_mean_markowitz"
VQM_STRATEGY_ID = "fixed_value_quality_momentum"
EQUAL_WEIGHT_STRATEGY_ID = "equal_weight_long_only_scoring_universe"
DECISION_FOCUSED_STRATEGY_ID = "decision_focused_neural_projected"

ACTIVE_STRATEGY_IDS = (
    PRIMARY_STRATEGY_ID,
    ELASTIC_NET_STRATEGY_ID,
    RANK_STRATEGY_ID,
    TRAILING_RESIDUAL_STRATEGY_ID,
    VQM_STRATEGY_ID,
    EQUAL_WEIGHT_STRATEGY_ID,
)
ALL_STRATEGY_IDS = ACTIVE_STRATEGY_IDS + (DECISION_FOCUSED_STRATEGY_ID,)
CALIBRATED_STRATEGY_IDS = (
    PRIMARY_STRATEGY_ID,
    ELASTIC_NET_STRATEGY_ID,
    VQM_STRATEGY_ID,
)

PRIMARY_REFIT_SIGNAL_MONTHS = frozenset({3, 6, 9, 12})
CALIBRATION_HALF_LIFE_MONTHS = 120.0
CALIBRATION_HUBER_DELTA = 1.345
REFERENCE_NAV_DOLLARS = 100_000_000.0
DECISION_FOCUSED_INACTIVE_REASON = "production_training_runtime_not_certified"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MONTH = re.compile(r"^[0-9]{4}-(?:0[1-9]|1[0-2])$")


def _closed_month_span(first: str, last: str) -> tuple[str, ...]:
    values: list[str] = []
    current = first
    while current <= last:
        values.append(current)
        current = shift_month(current, 1)
    return tuple(values)


PREFINAL_DEVELOPMENT_OOF_HOLDING_MONTHS = _closed_month_span("2007-01", "2014-11")
PREFINAL_VALIDATION_HOLDING_MONTHS = _closed_month_span("2015-01", "2019-11")
PREFINAL_TRAINING_HOLDING_MONTHS = (
    PREFINAL_DEVELOPMENT_OOF_HOLDING_MONTHS + PREFINAL_VALIDATION_HOLDING_MONTHS
)
PREFINAL_FINAL_REFIT_SIGNAL_DATE = dt.date(2019, 12, 31)


class FinalEconomicKernelError(ValueError):
    """Raised when a proposed final-test calculation escapes the freeze."""


class FeatureScope(str, Enum):
    FULL_FEATURE_PRIMARY = "full_feature_primary"
    REDUCED_AUXILIARY_CORE = "reduced_auxiliary_core"


SHARED_CALIBRATION_SOURCE_SCOPE = FeatureScope.FULL_FEATURE_PRIMARY
SHARED_CALIBRATION_POLICY_ID = (
    "full-feature-prefinal-and-prior-live-shared-by-strategy-v1"
)


def _require_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise FinalEconomicKernelError(f"{field} must be an exact SHA-256")
    return value


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _semantic_sha256(value: object) -> str:
    return _hash_bytes(_canonical_json_bytes(value))


def _float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalEconomicKernelError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise FinalEconomicKernelError(f"{field} must be finite")
    return result


def _finite_vector(value: object, *, field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalEconomicKernelError(f"{field} must be numeric") from exc
    if result.ndim != 1 or not np.isfinite(result).all():
        raise FinalEconomicKernelError(f"{field} must be a finite vector")
    result.setflags(write=False)
    return result


def _finite_matrix(value: object, *, field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalEconomicKernelError(f"{field} must be numeric") from exc
    if result.ndim != 2 or not np.isfinite(result).all():
        raise FinalEconomicKernelError(f"{field} must be a finite matrix")
    result.setflags(write=False)
    return result


def _integer_vector(value: object, *, field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.int64).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalEconomicKernelError(f"{field} must contain integers") from exc
    if result.ndim != 1 or np.any(result <= 0) or len(np.unique(result)) != len(result):
        raise FinalEconomicKernelError(f"{field} must contain unique positive integers")
    result.setflags(write=False)
    return result


def _array_payload(value: np.ndarray) -> dict[str, object]:
    vector = np.asarray(value)
    flat: list[object]
    if vector.dtype.kind == "f":
        if not np.isfinite(vector).all():
            raise FinalEconomicKernelError("semantic arrays must be finite")
        flat = [float(item).hex() for item in vector.reshape(-1)]
    elif vector.dtype.kind in {"i", "u"}:
        flat = [int(item) for item in vector.reshape(-1)]
    elif vector.dtype.kind == "b":
        flat = [bool(item) for item in vector.reshape(-1)]
    else:
        flat = [str(item) for item in vector.reshape(-1)]
    return {
        "dtype": vector.dtype.str,
        "shape": list(vector.shape),
        "values": flat,
    }


def _month_ordinal(value: str) -> int:
    if not isinstance(value, str) or _MONTH.fullmatch(value) is None:
        raise FinalEconomicKernelError("holding_month must use exact YYYY-MM")
    year, month = (int(part) for part in value.split("-"))
    return year * 12 + month - 1


def _previous_month(value: str) -> str:
    return shift_month(value, -1)


def _validate_month_dates(
    *,
    holding_month: str,
    signal_date: dt.date,
    entry_date: dt.date,
    exit_date: dt.date,
) -> None:
    descriptor = final_holding_month_descriptor(holding_month)
    if any(
        type(value) is not dt.date for value in (signal_date, entry_date, exit_date)
    ):
        raise FinalEconomicKernelError("signal, entry, and exit must be dates")
    if not signal_date < entry_date <= exit_date:
        raise FinalEconomicKernelError(
            "month dates must satisfy signal < entry <= exit"
        )
    if signal_date.strftime("%Y-%m") != _previous_month(holding_month):
        raise FinalEconomicKernelError("signal date is outside the prior month")
    if entry_date.strftime("%Y-%m") != holding_month:
        raise FinalEconomicKernelError("entry date is outside the holding month")
    expected_exit_month = (
        shift_month(holding_month, 1)
        if descriptor.complete_holding_interval
        else holding_month
    )
    if exit_date.strftime("%Y-%m") != expected_exit_month:
        raise FinalEconomicKernelError("exit date is outside the frozen interval")
    anchor = FROZEN_HOLDING_INTERVAL_ANCHORS.get(holding_month)
    if anchor is not None and (
        signal_date != anchor.signal_date
        or entry_date != anchor.entry_date
        or exit_date != anchor.exit_date
    ):
        raise FinalEconomicKernelError("methodology-anchored interval dates differ")


def is_quarterly_refit(
    *,
    holding_month: str,
    signal_date: dt.date,
) -> bool:
    """Return the frozen refit decision after validating final chronology."""

    descriptor = final_holding_month_descriptor(holding_month)
    del descriptor
    if type(signal_date) is not dt.date:
        raise FinalEconomicKernelError("signal_date must be a date")
    if signal_date.strftime("%Y-%m") != _previous_month(holding_month):
        raise FinalEconomicKernelError("signal date is outside the prior month")
    return signal_date.month in PRIMARY_REFIT_SIGNAL_MONTHS


@dataclass(frozen=True, slots=True)
class FrozenModelDefinition:
    family: str
    config_id: str
    seeds: tuple[int | None, ...]

    def __post_init__(self) -> None:
        expected = {
            "neural_network": (NEURAL_NETWORK_CONFIG_ID, FINAL_STOCHASTIC_SEEDS),
            "xgboost": (XGBOOST_CONFIG_ID, FINAL_STOCHASTIC_SEEDS),
            "elastic_net": (ELASTIC_NET_CONFIG_ID, (None,)),
        }
        if (
            self.family not in expected
            or (self.config_id, self.seeds) != expected[self.family]
        ):
            raise FinalEconomicKernelError("model definition differs from freeze")

    def to_mapping(self) -> dict[str, object]:
        return {
            "family": self.family,
            "config_id": self.config_id,
            "seeds": list(self.seeds),
        }


@dataclass(frozen=True, slots=True)
class FrozenStrategyDefinition:
    strategy_id: str
    signal: str
    allocator: str
    active: bool
    confirmatory: bool
    inactive_reason: str | None = None
    romano_wolf_p_value: float | None = None

    def __post_init__(self) -> None:
        if self.strategy_id not in ALL_STRATEGY_IDS:
            raise FinalEconomicKernelError("strategy escaped the frozen set")
        if type(self.active) is not bool or type(self.confirmatory) is not bool:
            raise FinalEconomicKernelError("strategy activity flags must be boolean")
        if self.strategy_id == DECISION_FOCUSED_STRATEGY_ID:
            if (
                self.active
                or self.inactive_reason != DECISION_FOCUSED_INACTIVE_REASON
                or self.romano_wolf_p_value != 1.0
            ):
                raise FinalEconomicKernelError(
                    "decision-focused strategy must remain explicitly inactive"
                )
        elif not self.active or self.inactive_reason is not None:
            raise FinalEconomicKernelError("an active frozen strategy is disabled")

    def to_mapping(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "signal": self.signal,
            "allocator": self.allocator,
            "active": self.active,
            "confirmatory": self.confirmatory,
            "inactive_reason": self.inactive_reason,
            "romano_wolf_p_value": self.romano_wolf_p_value,
        }


@dataclass(frozen=True, slots=True)
class FrozenEconomicSegmentPlan:
    segment: FinalPeriodSegment
    holding_months: tuple[str, ...]
    feature_scope: FeatureScope
    confirmatory_inference: bool
    genesis_rule: str

    def __post_init__(self) -> None:
        expected = {
            FinalPeriodSegment.PRIMARY_FULL: (
                PRIMARY_FULL_HOLDING_MONTHS,
                FeatureScope.FULL_FEATURE_PRIMARY,
                True,
                "reference_nav_all_cash_then_monthly_primary_chain",
            ),
            FinalPeriodSegment.CORE_COMPLETE: (
                CORE_COMPLETE_HOLDING_MONTHS,
                FeatureScope.REDUCED_AUXILIARY_CORE,
                False,
                "bind_august_2025_authenticated_shadow_core_head",
            ),
            FinalPeriodSegment.PARTIAL_DECEMBER_DESCRIPTIVE: (
                (PARTIAL_DECEMBER_HOLDING_MONTH,),
                FeatureScope.REDUCED_AUXILIARY_CORE,
                False,
                "bind_november_2025_authenticated_core_head",
            ),
        }[self.segment]
        observed = (
            self.holding_months,
            self.feature_scope,
            self.confirmatory_inference,
            self.genesis_rule,
        )
        if observed != expected:
            raise FinalEconomicKernelError("economic segment plan differs from freeze")

    def to_mapping(self) -> dict[str, object]:
        return {
            "segment": self.segment.value,
            "holding_months": list(self.holding_months),
            "feature_scope": self.feature_scope.value,
            "confirmatory_inference": self.confirmatory_inference,
            "genesis_rule": self.genesis_rule,
        }


FROZEN_MODEL_DEFINITIONS = (
    FrozenModelDefinition(
        "neural_network", NEURAL_NETWORK_CONFIG_ID, FINAL_STOCHASTIC_SEEDS
    ),
    FrozenModelDefinition("xgboost", XGBOOST_CONFIG_ID, FINAL_STOCHASTIC_SEEDS),
    FrozenModelDefinition("elastic_net", ELASTIC_NET_CONFIG_ID, (None,)),
)

FROZEN_STRATEGY_DEFINITIONS = (
    FrozenStrategyDefinition(
        PRIMARY_STRATEGY_ID,
        "0.5_nn_centered_rank_plus_0.5_xgb_centered_rank_no_rerank",
        "hybrid_alpha_risk_cost_convex",
        True,
        True,
    ),
    FrozenStrategyDefinition(
        ELASTIC_NET_STRATEGY_ID,
        "conditional_elastic_net_centered_rank",
        "hybrid_alpha_risk_cost_convex",
        True,
        True,
    ),
    FrozenStrategyDefinition(
        RANK_STRATEGY_ID,
        "primary_ensemble_score",
        "rank_volatility_projected",
        True,
        False,
    ),
    FrozenStrategyDefinition(
        TRAILING_RESIDUAL_STRATEGY_ID,
        "trailing_60_completed_residual_mean_n_over_n_plus_60",
        "hybrid_alpha_risk_cost_convex",
        True,
        False,
    ),
    FrozenStrategyDefinition(
        VQM_STRATEGY_ID,
        "fixed_value_quality_momentum_once_ranked_and_huber_calibrated",
        "hybrid_alpha_risk_cost_convex",
        True,
        False,
    ),
    FrozenStrategyDefinition(
        EQUAL_WEIGHT_STRATEGY_ID,
        "scoring_universe_equal_weight",
        "descriptive_100_percent_gross_long_only",
        True,
        False,
    ),
    FrozenStrategyDefinition(
        DECISION_FOCUSED_STRATEGY_ID,
        "frozen_inactive",
        "decision_focused_projected",
        False,
        False,
        DECISION_FOCUSED_INACTIVE_REASON,
        1.0,
    ),
)

FROZEN_ECONOMIC_SEGMENT_PLANS = (
    FrozenEconomicSegmentPlan(
        FinalPeriodSegment.PRIMARY_FULL,
        PRIMARY_FULL_HOLDING_MONTHS,
        FeatureScope.FULL_FEATURE_PRIMARY,
        True,
        "reference_nav_all_cash_then_monthly_primary_chain",
    ),
    FrozenEconomicSegmentPlan(
        FinalPeriodSegment.CORE_COMPLETE,
        CORE_COMPLETE_HOLDING_MONTHS,
        FeatureScope.REDUCED_AUXILIARY_CORE,
        False,
        "bind_august_2025_authenticated_shadow_core_head",
    ),
    FrozenEconomicSegmentPlan(
        FinalPeriodSegment.PARTIAL_DECEMBER_DESCRIPTIVE,
        (PARTIAL_DECEMBER_HOLDING_MONTH,),
        FeatureScope.REDUCED_AUXILIARY_CORE,
        False,
        "bind_november_2025_authenticated_core_head",
    ),
)


@dataclass(frozen=True, slots=True)
class FinalEconomicKernelContract:
    schema_version: int
    kernel_id: str
    models: tuple[FrozenModelDefinition, ...]
    strategies: tuple[FrozenStrategyDefinition, ...]
    segment_plans: tuple[FrozenEconomicSegmentPlan, ...]
    materiality_policy_id: str
    per_position_limit_dollars_strict: float
    aggregate_limit_dollars: float
    contract_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != FINAL_ECONOMIC_KERNEL_SCHEMA_VERSION
            or self.kernel_id != FINAL_ECONOMIC_KERNEL_ID
            or self.models != FROZEN_MODEL_DEFINITIONS
            or self.strategies != FROZEN_STRATEGY_DEFINITIONS
            or self.segment_plans != FROZEN_ECONOMIC_SEGMENT_PLANS
            or self.materiality_policy_id != NUMERICAL_MATERIALITY_POLICY_ID
            or self.per_position_limit_dollars_strict != STRICT_DOLLAR_MATERIALITY
            or self.aggregate_limit_dollars != AGGREGATE_DOLLAR_MATERIALITY_LIMIT
        ):
            raise FinalEconomicKernelError("economic kernel contract differs")
        _require_hash(self.contract_sha256, field="contract_sha256")
        if self.contract_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("economic kernel contract digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kernel_id": self.kernel_id,
            "models": [item.to_mapping() for item in self.models],
            "strategies": [item.to_mapping() for item in self.strategies],
            "segment_plans": [item.to_mapping() for item in self.segment_plans],
            "seed_aggregation": "mean_raw_forecasts_per_security",
            "component_transform": "centered_average_percentile_rank",
            "ensemble": {
                "neural_network_weight": 0.5,
                "xgboost_weight": 0.5,
                "reranked": False,
            },
            "refit": {
                "signal_months": [3, 6, 9, 12],
                "completed_labels_only": True,
                "memory_half_life_months": 120,
                "calibration": "zero_intercept_huber_1.345_nonnegative",
                "calibration_scope_policy_id": (SHARED_CALIBRATION_POLICY_ID),
                "calibration_source_scope": (SHARED_CALIBRATION_SOURCE_SCOPE.value),
                "same_strategy_slope_shared_across_feature_scopes": True,
                "reduced_core_scores_relabelled_as_prefinal_oof": False,
            },
            "materiality": {
                "policy_id": self.materiality_policy_id,
                "per_position_limit_dollars_strict": (
                    self.per_position_limit_dollars_strict
                ),
                "exactly_threshold_is_material": True,
                "aggregate_limit_dollars": self.aggregate_limit_dollars,
            },
            "pnl_splicing_permitted": False,
        }
        if include_digest:
            value["contract_sha256"] = self.contract_sha256
        return value


def _build_contract() -> FinalEconomicKernelContract:
    provisional = FinalEconomicKernelContract.__new__(FinalEconomicKernelContract)
    object.__setattr__(
        provisional, "schema_version", FINAL_ECONOMIC_KERNEL_SCHEMA_VERSION
    )
    object.__setattr__(provisional, "kernel_id", FINAL_ECONOMIC_KERNEL_ID)
    object.__setattr__(provisional, "models", FROZEN_MODEL_DEFINITIONS)
    object.__setattr__(provisional, "strategies", FROZEN_STRATEGY_DEFINITIONS)
    object.__setattr__(provisional, "segment_plans", FROZEN_ECONOMIC_SEGMENT_PLANS)
    object.__setattr__(
        provisional, "materiality_policy_id", NUMERICAL_MATERIALITY_POLICY_ID
    )
    object.__setattr__(
        provisional,
        "per_position_limit_dollars_strict",
        STRICT_DOLLAR_MATERIALITY,
    )
    object.__setattr__(
        provisional,
        "aggregate_limit_dollars",
        AGGREGATE_DOLLAR_MATERIALITY_LIMIT,
    )
    object.__setattr__(provisional, "contract_sha256", "")
    digest = _semantic_sha256(provisional.payload(False))
    return FinalEconomicKernelContract(
        FINAL_ECONOMIC_KERNEL_SCHEMA_VERSION,
        FINAL_ECONOMIC_KERNEL_ID,
        FROZEN_MODEL_DEFINITIONS,
        FROZEN_STRATEGY_DEFINITIONS,
        FROZEN_ECONOMIC_SEGMENT_PLANS,
        NUMERICAL_MATERIALITY_POLICY_ID,
        STRICT_DOLLAR_MATERIALITY,
        AGGREGATE_DOLLAR_MATERIALITY_LIMIT,
        digest,
    )


FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT = _build_contract()


def _expected_config(family: str) -> tuple[str, tuple[int | None, ...]]:
    for definition in FROZEN_MODEL_DEFINITIONS:
        if definition.family == family:
            return definition.config_id, definition.seeds
    raise FinalEconomicKernelError("unknown frozen model family")


@dataclass(frozen=True, slots=True, eq=False)
class SeedForecastBlock:
    """Raw selected-model forecasts in the exact final seed order."""

    model_family: str
    config_id: str
    seeds: tuple[int | None, ...]
    raw_forecasts: np.ndarray
    fitted_state_sha256_by_seed: tuple[str, ...]
    block_sha256: str

    def __post_init__(self) -> None:
        expected_config, expected_seeds = _expected_config(self.model_family)
        if self.config_id != expected_config or self.seeds != expected_seeds:
            raise FinalEconomicKernelError("forecast model or seed order differs")
        matrix = _finite_matrix(self.raw_forecasts, field="raw_forecasts")
        if matrix.shape[0] != len(expected_seeds) or matrix.shape[1] == 0:
            raise FinalEconomicKernelError("forecast matrix shape differs")
        states = tuple(self.fitted_state_sha256_by_seed)
        if len(states) != len(expected_seeds):
            raise FinalEconomicKernelError("forecast fitted-state count differs")
        for digest in states:
            _require_hash(digest, field="fitted_state_sha256")
        object.__setattr__(self, "raw_forecasts", matrix)
        object.__setattr__(self, "fitted_state_sha256_by_seed", states)
        _require_hash(self.block_sha256, field="block_sha256")
        if self.block_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("forecast block digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "model_family": self.model_family,
            "config_id": self.config_id,
            "seeds": list(self.seeds),
            "raw_forecasts": _array_payload(self.raw_forecasts),
            "fitted_state_sha256_by_seed": list(self.fitted_state_sha256_by_seed),
        }
        if include_digest:
            value["block_sha256"] = self.block_sha256
        return value

    @classmethod
    def create(
        cls,
        *,
        model_family: str,
        config_id: str,
        seeds: Sequence[int | None],
        raw_forecasts: object,
        fitted_state_sha256_by_seed: Sequence[str],
    ) -> "SeedForecastBlock":
        provisional = cls.__new__(cls)
        object.__setattr__(provisional, "model_family", model_family)
        object.__setattr__(provisional, "config_id", config_id)
        object.__setattr__(provisional, "seeds", tuple(seeds))
        object.__setattr__(
            provisional,
            "raw_forecasts",
            _finite_matrix(raw_forecasts, field="raw_forecasts"),
        )
        object.__setattr__(
            provisional,
            "fitted_state_sha256_by_seed",
            tuple(fitted_state_sha256_by_seed),
        )
        object.__setattr__(provisional, "block_sha256", "")
        digest = _semantic_sha256(provisional.payload(False))
        return cls(
            model_family=model_family,
            config_id=config_id,
            seeds=tuple(seeds),
            raw_forecasts=raw_forecasts,  # type: ignore[arg-type]
            fitted_state_sha256_by_seed=tuple(fitted_state_sha256_by_seed),
            block_sha256=digest,
        )


@dataclass(frozen=True, slots=True, eq=False)
class FinalForecastReceipt:
    feature_scope: FeatureScope
    holding_month: str
    permno: np.ndarray
    neural_network_raw_mean: np.ndarray
    xgboost_raw_mean: np.ndarray
    elastic_net_raw: np.ndarray
    neural_network_score: np.ndarray
    xgboost_score: np.ndarray
    elastic_net_score: np.ndarray
    primary_ensemble_score: np.ndarray
    ensemble_reranked: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        final_holding_month_descriptor(self.holding_month)
        if not isinstance(self.feature_scope, FeatureScope):
            raise FinalEconomicKernelError("feature_scope must be frozen")
        permno = _integer_vector(self.permno, field="permno")
        arrays: dict[str, np.ndarray] = {}
        for name in (
            "neural_network_raw_mean",
            "xgboost_raw_mean",
            "elastic_net_raw",
            "neural_network_score",
            "xgboost_score",
            "elastic_net_score",
            "primary_ensemble_score",
        ):
            value = _finite_vector(getattr(self, name), field=name)
            if len(value) != len(permno):
                raise FinalEconomicKernelError("forecast receipt lengths differ")
            arrays[name] = value
        if self.ensemble_reranked is not False:
            raise FinalEconomicKernelError("final ensemble may not be reranked")
        expected = 0.5 * arrays["neural_network_score"] + 0.5 * arrays["xgboost_score"]
        if not np.array_equal(arrays["primary_ensemble_score"], expected):
            raise FinalEconomicKernelError("final ensemble is not the direct 50/50 sum")
        for name in (
            "neural_network_score",
            "xgboost_score",
            "elastic_net_score",
        ):
            if abs(float(np.mean(arrays[name], dtype=np.float64))) > 1e-12:
                raise FinalEconomicKernelError("component score is not centered")
        object.__setattr__(self, "permno", permno)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        _require_hash(self.receipt_sha256, field="receipt_sha256")
        if self.receipt_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("forecast receipt digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "feature_scope": self.feature_scope.value,
            "holding_month": self.holding_month,
            "permno": _array_payload(self.permno),
            "neural_network_raw_mean": _array_payload(self.neural_network_raw_mean),
            "xgboost_raw_mean": _array_payload(self.xgboost_raw_mean),
            "elastic_net_raw": _array_payload(self.elastic_net_raw),
            "neural_network_score": _array_payload(self.neural_network_score),
            "xgboost_score": _array_payload(self.xgboost_score),
            "elastic_net_score": _array_payload(self.elastic_net_score),
            "primary_ensemble_score": _array_payload(self.primary_ensemble_score),
            "seed_aggregation": "mean_raw_forecasts_per_security",
            "component_transform": "centered_average_percentile_rank",
            "ensemble_weights": [0.5, 0.5],
            "ensemble_reranked": self.ensemble_reranked,
        }
        if include_digest:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def aggregate_selected_forecasts(
    *,
    feature_scope: FeatureScope,
    holding_month: str,
    permno: object,
    neural_network: SeedForecastBlock,
    xgboost: SeedForecastBlock,
    elastic_net: SeedForecastBlock,
) -> FinalForecastReceipt:
    """Apply the frozen seed mean, component ranks, and direct ensemble sum."""

    securities = _integer_vector(permno, field="permno")
    blocks = {
        "neural_network": neural_network,
        "xgboost": xgboost,
        "elastic_net": elastic_net,
    }
    if any(
        not isinstance(block, SeedForecastBlock)
        or block.model_family != family
        or block.raw_forecasts.shape[1] != len(securities)
        for family, block in blocks.items()
    ):
        raise FinalEconomicKernelError("selected forecast block set differs")
    means = {
        family: np.mean(block.raw_forecasts, axis=0, dtype=np.float64)
        for family, block in blocks.items()
    }
    scores = {family: centered_average_rank(value) for family, value in means.items()}
    primary = 0.5 * scores["neural_network"] + 0.5 * scores["xgboost"]
    provisional = FinalForecastReceipt.__new__(FinalForecastReceipt)
    for name, value in (
        ("feature_scope", feature_scope),
        ("holding_month", holding_month),
        ("permno", securities),
        ("neural_network_raw_mean", means["neural_network"]),
        ("xgboost_raw_mean", means["xgboost"]),
        ("elastic_net_raw", means["elastic_net"]),
        ("neural_network_score", scores["neural_network"]),
        ("xgboost_score", scores["xgboost"]),
        ("elastic_net_score", scores["elastic_net"]),
        ("primary_ensemble_score", primary),
        ("ensemble_reranked", False),
        ("receipt_sha256", ""),
    ):
        object.__setattr__(provisional, name, value)
    digest = _semantic_sha256(provisional.payload(False))
    return FinalForecastReceipt(
        feature_scope=feature_scope,
        holding_month=holding_month,
        permno=securities,
        neural_network_raw_mean=means["neural_network"],
        xgboost_raw_mean=means["xgboost"],
        elastic_net_raw=means["elastic_net"],
        neural_network_score=scores["neural_network"],
        xgboost_score=scores["xgboost"],
        elastic_net_score=scores["elastic_net"],
        primary_ensemble_score=primary,
        ensemble_reranked=False,
        receipt_sha256=digest,
    )


@dataclass(frozen=True, slots=True)
class FinalCalibrationObservation:
    """One pre-final OOF or previously committed final live observation."""

    feature_scope: FeatureScope
    strategy_id: str
    forecast_id: str
    holding_month: str
    permno: int
    score: float
    residual_return: float
    signal_date: dt.date
    entry_date: dt.date
    exit_date: dt.date
    outcome_available_date: dt.date
    forecast_payload_sha256: str
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature_scope, FeatureScope):
            raise FinalEconomicKernelError("calibration feature scope differs")
        if self.strategy_id not in CALIBRATED_STRATEGY_IDS:
            raise FinalEconomicKernelError("strategy does not use Huber calibration")
        if not isinstance(self.forecast_id, str) or not self.forecast_id:
            raise FinalEconomicKernelError("forecast_id must be nonempty")
        _month_ordinal(self.holding_month)
        if type(self.permno) is not int or self.permno <= 0:
            raise FinalEconomicKernelError("calibration permno must be positive")
        score = _float(self.score, field="score")
        outcome = _float(self.residual_return, field="residual_return")
        if not -0.5 <= score <= 0.5:
            raise FinalEconomicKernelError("calibration score must be in [-0.5, 0.5]")
        if any(
            type(value) is not dt.date
            for value in (
                self.signal_date,
                self.entry_date,
                self.exit_date,
                self.outcome_available_date,
            )
        ):
            raise FinalEconomicKernelError("calibration chronology must use dates")
        if not self.signal_date < self.entry_date < self.exit_date:
            raise FinalEconomicKernelError("calibration interval chronology differs")
        if self.entry_date.strftime("%Y-%m") != self.holding_month:
            raise FinalEconomicKernelError("calibration holding month differs")
        if self.outcome_available_date < self.exit_date:
            raise FinalEconomicKernelError("outcome predates completed holding period")
        _require_hash(self.forecast_payload_sha256, field="forecast_payload_sha256")
        if self.source not in {"pre_final_pseudo_oof", "final_prior_live"}:
            raise FinalEconomicKernelError("calibration source is not OOF/prior-live")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "residual_return", outcome)

    def to_mapping(self) -> dict[str, object]:
        return {
            "feature_scope": self.feature_scope.value,
            "strategy_id": self.strategy_id,
            "forecast_id": self.forecast_id,
            "holding_month": self.holding_month,
            "permno": self.permno,
            "score": self.score.hex(),
            "residual_return": self.residual_return.hex(),
            "signal_date": self.signal_date.isoformat(),
            "entry_date": self.entry_date.isoformat(),
            "exit_date": self.exit_date.isoformat(),
            "outcome_available_date": self.outcome_available_date.isoformat(),
            "forecast_payload_sha256": self.forecast_payload_sha256,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class FinalCalibrationReceipt:
    feature_scope: FeatureScope
    strategy_id: str
    fit_signal_date: dt.date
    kappa: float
    unconstrained_slope: float
    observation_count: int
    distinct_month_count: int
    first_holding_month: str
    last_holding_month: str
    pre_final_observation_count: int
    final_prior_live_observation_count: int
    completed_labels_only: bool
    calibration_sha256: str

    def __post_init__(self) -> None:
        if self.strategy_id not in CALIBRATED_STRATEGY_IDS:
            raise FinalEconomicKernelError("calibration receipt strategy differs")
        if type(self.fit_signal_date) is not dt.date:
            raise FinalEconomicKernelError("fit_signal_date must be a date")
        kappa = _float(self.kappa, field="kappa")
        slope = _float(self.unconstrained_slope, field="unconstrained_slope")
        if kappa != max(0.0, slope):
            raise FinalEconomicKernelError("calibration nonnegative truncation differs")
        if (
            type(self.observation_count) is not int
            or self.observation_count <= 0
            or type(self.distinct_month_count) is not int
            or self.distinct_month_count <= 0
            or self.pre_final_observation_count
            + self.final_prior_live_observation_count
            != self.observation_count
            or self.completed_labels_only is not True
        ):
            raise FinalEconomicKernelError("calibration receipt counts differ")
        if _month_ordinal(self.first_holding_month) > _month_ordinal(
            self.last_holding_month
        ):
            raise FinalEconomicKernelError("calibration month bounds are reversed")
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "unconstrained_slope", slope)
        _require_hash(self.calibration_sha256, field="calibration_sha256")
        if self.calibration_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("calibration digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "feature_scope": self.feature_scope.value,
            "strategy_id": self.strategy_id,
            "fit_signal_date": self.fit_signal_date.isoformat(),
            "kappa": self.kappa.hex(),
            "unconstrained_slope": self.unconstrained_slope.hex(),
            "observation_count": self.observation_count,
            "distinct_month_count": self.distinct_month_count,
            "first_holding_month": self.first_holding_month,
            "last_holding_month": self.last_holding_month,
            "pre_final_observation_count": self.pre_final_observation_count,
            "final_prior_live_observation_count": (
                self.final_prior_live_observation_count
            ),
            "equal_calendar_month_mass": True,
            "memory_half_life_months": CALIBRATION_HALF_LIFE_MONTHS,
            "huber_delta": CALIBRATION_HUBER_DELTA,
            "zero_intercept": True,
            "completed_labels_only": self.completed_labels_only,
        }
        if include_digest:
            value["calibration_sha256"] = self.calibration_sha256
        return value


def fit_final_calibration(
    observations: Sequence[FinalCalibrationObservation],
    *,
    feature_scope: FeatureScope,
    strategy_id: str,
    fit_holding_month: str,
    fit_signal_date: dt.date,
) -> FinalCalibrationReceipt:
    """Fit the fixed final-period Huber slope from completed labels only."""

    if not is_quarterly_refit(
        holding_month=fit_holding_month,
        signal_date=fit_signal_date,
    ):
        raise FinalEconomicKernelError("calibration fit is not quarterly")
    rows = tuple(observations)
    if not rows:
        raise FinalEconomicKernelError("calibration observation set is empty")
    keys: set[tuple[str, int]] = set()
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, FinalCalibrationObservation):
            raise FinalEconomicKernelError("calibration rows have the wrong type")
        if row.feature_scope is not feature_scope or row.strategy_id != strategy_id:
            raise FinalEconomicKernelError("calibration row identity differs")
        if (
            row.exit_date > fit_signal_date
            or row.outcome_available_date > fit_signal_date
            or _month_ordinal(row.holding_month) >= _month_ordinal(fit_holding_month)
        ):
            raise FinalEconomicKernelError(
                "current, overlapping, unavailable, or future label entered refit"
            )
        key = (row.holding_month, row.permno)
        if key in keys:
            raise FinalEconomicKernelError("duplicate calibration security-month")
        keys.add(key)
        counts[row.holding_month] = counts.get(row.holding_month, 0) + 1

    fit_ordinal = _month_ordinal(fit_holding_month)
    raw_weights = np.asarray(
        [
            2.0
            ** (
                -float(fit_ordinal - _month_ordinal(row.holding_month))
                / CALIBRATION_HALF_LIFE_MONTHS
            )
            / counts[row.holding_month]
            for row in rows
        ],
        dtype=np.float64,
    )
    weights = raw_weights * (len(rows) / float(np.sum(raw_weights)))
    score = np.asarray([row.score for row in rows], dtype=np.float64)
    raw_outcome = np.asarray([row.residual_return for row in rows], dtype=np.float64)
    outcome = raw_outcome.copy()
    positions_by_month: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        positions_by_month.setdefault(row.holding_month, []).append(index)
    for positions in positions_by_month.values():
        selected = np.asarray(positions, dtype=np.int64)
        outcome[selected] -= float(np.mean(raw_outcome[selected], dtype=np.float64))
    if np.all(score == 0.0):
        slope = 0.0
    else:
        model = HuberRegressor(
            epsilon=CALIBRATION_HUBER_DELTA,
            alpha=0.0,
            fit_intercept=False,
            max_iter=1000,
            tol=1e-10,
        )
        try:
            model.fit(score.reshape(-1, 1), outcome, sample_weight=weights)
        except (ValueError, FloatingPointError) as exc:
            raise FinalEconomicKernelError("final Huber calibration failed") from exc
        slope = float(model.coef_[0])
        if not math.isfinite(slope):
            raise FinalEconomicKernelError("final Huber calibration is nonfinite")
    kappa = max(0.0, slope)
    ordered_months = sorted(counts, key=_month_ordinal)
    values = {
        "feature_scope": feature_scope,
        "strategy_id": strategy_id,
        "fit_signal_date": fit_signal_date,
        "kappa": kappa,
        "unconstrained_slope": slope,
        "observation_count": len(rows),
        "distinct_month_count": len(counts),
        "first_holding_month": ordered_months[0],
        "last_holding_month": ordered_months[-1],
        "pre_final_observation_count": sum(
            row.source == "pre_final_pseudo_oof" for row in rows
        ),
        "final_prior_live_observation_count": sum(
            row.source == "final_prior_live" for row in rows
        ),
        "completed_labels_only": True,
    }
    provisional = FinalCalibrationReceipt.__new__(FinalCalibrationReceipt)
    for name, value in (*values.items(), ("calibration_sha256", "")):
        object.__setattr__(provisional, name, value)
    digest = _semantic_sha256(provisional.payload(False))
    return FinalCalibrationReceipt(**values, calibration_sha256=digest)


@dataclass(frozen=True, slots=True)
class FinalModelFitCheckpoint:
    feature_scope: FeatureScope
    model_family: str
    config_id: str
    seed: int | None
    fitted_state_sha256: str

    def __post_init__(self) -> None:
        config, seeds = _expected_config(self.model_family)
        if self.config_id != config or self.seed not in seeds:
            raise FinalEconomicKernelError("refit checkpoint identity differs")
        _require_hash(self.fitted_state_sha256, field="fitted_state_sha256")

    def to_mapping(self) -> dict[str, object]:
        return {
            "feature_scope": self.feature_scope.value,
            "model_family": self.model_family,
            "config_id": self.config_id,
            "seed": self.seed,
            "fitted_state_sha256": self.fitted_state_sha256,
        }


@dataclass(frozen=True, slots=True)
class FinalRefitEvidence:
    feature_scope: FeatureScope
    holding_month: str
    fit_signal_date: dt.date
    completed_label_set_sha256: str
    completed_label_months: tuple[str, ...]
    maximum_outcome_available_date: dt.date
    checkpoints: tuple[FinalModelFitCheckpoint, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not is_quarterly_refit(
            holding_month=self.holding_month,
            signal_date=self.fit_signal_date,
        ):
            raise FinalEconomicKernelError("model refit evidence is not quarterly")
        _require_hash(
            self.completed_label_set_sha256,
            field="completed_label_set_sha256",
        )
        months = tuple(self.completed_label_months)
        if not months or months != tuple(sorted(set(months), key=_month_ordinal)):
            raise FinalEconomicKernelError("completed label months are not canonical")
        if any(
            _month_ordinal(month) >= _month_ordinal(self.holding_month)
            for month in months
        ):
            raise FinalEconomicKernelError("current/future label month entered refit")
        if (
            type(self.maximum_outcome_available_date) is not dt.date
            or self.maximum_outcome_available_date > self.fit_signal_date
        ):
            raise FinalEconomicKernelError("unavailable label entered refit evidence")
        checkpoints = tuple(self.checkpoints)
        expected = tuple(
            (definition.family, definition.config_id, seed)
            for definition in FROZEN_MODEL_DEFINITIONS
            for seed in definition.seeds
        )
        observed = tuple(
            (item.model_family, item.config_id, item.seed) for item in checkpoints
        )
        if observed != expected or any(
            item.feature_scope is not self.feature_scope for item in checkpoints
        ):
            raise FinalEconomicKernelError("refit checkpoint set/order differs")
        object.__setattr__(self, "completed_label_months", months)
        object.__setattr__(self, "checkpoints", checkpoints)
        _require_hash(self.evidence_sha256, field="evidence_sha256")
        if self.evidence_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("refit evidence digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "feature_scope": self.feature_scope.value,
            "holding_month": self.holding_month,
            "fit_signal_date": self.fit_signal_date.isoformat(),
            "completed_label_set_sha256": self.completed_label_set_sha256,
            "completed_label_months": list(self.completed_label_months),
            "maximum_outcome_available_date": (
                self.maximum_outcome_available_date.isoformat()
            ),
            "completed_labels_only": True,
            "checkpoints": [item.to_mapping() for item in self.checkpoints],
        }
        if include_digest:
            value["evidence_sha256"] = self.evidence_sha256
        return value

    @classmethod
    def create(
        cls,
        *,
        feature_scope: FeatureScope,
        holding_month: str,
        fit_signal_date: dt.date,
        completed_label_set_sha256: str,
        completed_label_months: Sequence[str],
        maximum_outcome_available_date: dt.date,
        checkpoints: Sequence[FinalModelFitCheckpoint],
    ) -> "FinalRefitEvidence":
        values = {
            "feature_scope": feature_scope,
            "holding_month": holding_month,
            "fit_signal_date": fit_signal_date,
            "completed_label_set_sha256": completed_label_set_sha256,
            "completed_label_months": tuple(completed_label_months),
            "maximum_outcome_available_date": maximum_outcome_available_date,
            "checkpoints": tuple(checkpoints),
        }
        provisional = cls.__new__(cls)
        for name, value in (*values.items(), ("evidence_sha256", "")):
            object.__setattr__(provisional, name, value)
        digest = _semantic_sha256(provisional.payload(False))
        return cls(**values, evidence_sha256=digest)


@dataclass(frozen=True, slots=True, eq=False)
class DailyReturnPanel:
    """In-memory total-return rows aligned by security identity."""

    permno: np.ndarray
    trading_dates: tuple[dt.date, ...]
    asset_total_returns: np.ndarray
    daily_risk_free_rates: np.ndarray
    monthly_source_bundle_sha256: str
    total_return_source_sha256: str
    risk_free_source_sha256: str
    corporate_action_ledger_sha256: str
    mark_continuity_ledger_sha256: str
    corporate_actions_integrated_in_total_returns: bool
    panel_sha256: str

    def __post_init__(self) -> None:
        permno = _integer_vector(self.permno, field="daily_return.permno")
        returns = _finite_matrix(self.asset_total_returns, field="asset_total_returns")
        risk_free = _finite_vector(
            self.daily_risk_free_rates, field="daily_risk_free_rates"
        )
        dates = tuple(self.trading_dates)
        if (
            returns.shape[0] == 0
            or returns.shape[1] != len(permno)
            or len(risk_free) != returns.shape[0]
            or len(dates) != returns.shape[0]
            or any(type(value) is not dt.date for value in dates)
            or dates != tuple(sorted(set(dates)))
            or np.any(returns < -1.0)
            or np.any(risk_free <= -1.0)
        ):
            raise FinalEconomicKernelError("daily return panel dimensions differ")
        for field in (
            "monthly_source_bundle_sha256",
            "total_return_source_sha256",
            "risk_free_source_sha256",
            "corporate_action_ledger_sha256",
            "mark_continuity_ledger_sha256",
        ):
            _require_hash(getattr(self, field), field=field)
        if self.corporate_actions_integrated_in_total_returns is not True:
            raise FinalEconomicKernelError(
                "daily total returns must integrate mandatory corporate actions"
            )
        object.__setattr__(self, "permno", permno)
        object.__setattr__(self, "trading_dates", dates)
        object.__setattr__(self, "asset_total_returns", returns)
        object.__setattr__(self, "daily_risk_free_rates", risk_free)
        _require_hash(self.panel_sha256, field="panel_sha256")
        if self.panel_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("daily return panel digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "permno": _array_payload(self.permno),
            "trading_dates": [value.isoformat() for value in self.trading_dates],
            "asset_total_returns": _array_payload(self.asset_total_returns),
            "daily_risk_free_rates": _array_payload(self.daily_risk_free_rates),
            "monthly_source_bundle_sha256": self.monthly_source_bundle_sha256,
            "total_return_source_sha256": self.total_return_source_sha256,
            "risk_free_source_sha256": self.risk_free_source_sha256,
            "corporate_action_ledger_sha256": (self.corporate_action_ledger_sha256),
            "mark_continuity_ledger_sha256": (self.mark_continuity_ledger_sha256),
            "return_semantics": (
                "authoritative_total_return_including_delisting_and_"
                "mandatory_corporate_actions"
            ),
            "corporate_actions_integrated_in_total_returns": (
                self.corporate_actions_integrated_in_total_returns
            ),
        }
        if include_digest:
            value["panel_sha256"] = self.panel_sha256
        return value

    @classmethod
    def create(
        cls,
        *,
        permno: object,
        trading_dates: Sequence[dt.date],
        asset_total_returns: object,
        daily_risk_free_rates: object,
        monthly_source_bundle_sha256: str,
        total_return_source_sha256: str,
        risk_free_source_sha256: str,
        corporate_action_ledger_sha256: str,
        mark_continuity_ledger_sha256: str,
        corporate_actions_integrated_in_total_returns: bool,
    ) -> "DailyReturnPanel":
        values = {
            "permno": _integer_vector(permno, field="daily_return.permno"),
            "trading_dates": tuple(trading_dates),
            "asset_total_returns": _finite_matrix(
                asset_total_returns, field="asset_total_returns"
            ),
            "daily_risk_free_rates": _finite_vector(
                daily_risk_free_rates, field="daily_risk_free_rates"
            ),
            "monthly_source_bundle_sha256": monthly_source_bundle_sha256,
            "total_return_source_sha256": total_return_source_sha256,
            "risk_free_source_sha256": risk_free_source_sha256,
            "corporate_action_ledger_sha256": corporate_action_ledger_sha256,
            "mark_continuity_ledger_sha256": mark_continuity_ledger_sha256,
            "corporate_actions_integrated_in_total_returns": (
                corporate_actions_integrated_in_total_returns
            ),
        }
        provisional = cls.__new__(cls)
        for name, value in (*values.items(), ("panel_sha256", "")):
            object.__setattr__(provisional, name, value)
        digest = _semantic_sha256(provisional.payload(False))
        return cls(**values, panel_sha256=digest)

    def aligned_returns(self, target_permno: np.ndarray) -> np.ndarray:
        index = {int(value): position for position, value in enumerate(self.permno)}
        missing = sorted(
            int(value) for value in target_permno if int(value) not in index
        )
        if missing:
            raise FinalEconomicKernelError(
                "daily return panel omits target securities: "
                + ", ".join(str(value) for value in missing)
            )
        columns = np.asarray([index[int(value)] for value in target_permno])
        return self.asset_total_returns[:, columns]


@dataclass(frozen=True, slots=True)
class StrategyMonthMaterialityEvidence:
    strategy_id: str
    holding_month: str
    residue_count: int
    aggregate_absolute_dollars: float
    maximum_absolute_dollars: float
    policy_id: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.strategy_id not in ACTIVE_STRATEGY_IDS:
            raise FinalEconomicKernelError("materiality strategy differs")
        final_holding_month_descriptor(self.holding_month)
        if (
            type(self.residue_count) is not int
            or self.residue_count < 0
            or not 0.0 <= self.maximum_absolute_dollars < STRICT_DOLLAR_MATERIALITY
            or not 0.0
            <= self.aggregate_absolute_dollars
            <= AGGREGATE_DOLLAR_MATERIALITY_LIMIT
            or self.policy_id != NUMERICAL_MATERIALITY_POLICY_ID
        ):
            raise FinalEconomicKernelError("materiality evidence differs")
        _require_hash(self.evidence_sha256, field="evidence_sha256")
        if self.evidence_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("materiality evidence digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "strategy_id": self.strategy_id,
            "holding_month": self.holding_month,
            "residue_count": self.residue_count,
            "aggregate_absolute_dollars": self.aggregate_absolute_dollars.hex(),
            "maximum_absolute_dollars": self.maximum_absolute_dollars.hex(),
            "policy_id": self.policy_id,
            "per_position_limit_dollars_strict": STRICT_DOLLAR_MATERIALITY,
            "aggregate_limit_dollars": AGGREGATE_DOLLAR_MATERIALITY_LIMIT,
        }
        if include_digest:
            value["evidence_sha256"] = self.evidence_sha256
        return value


def audit_final_strategy_month_materiality(
    *,
    strategy_id: str,
    holding_month: str,
    residue_dollars: object,
) -> StrategyMonthMaterialityEvidence:
    audit: NumericalMaterialityAudit = audit_strategy_month_residue_dollars(
        residue_dollars  # type: ignore[arg-type]
    )
    values = {
        "strategy_id": strategy_id,
        "holding_month": holding_month,
        "residue_count": audit.residue_count,
        "aggregate_absolute_dollars": audit.aggregate_absolute_dollars,
        "maximum_absolute_dollars": audit.maximum_absolute_dollars,
        "policy_id": audit.policy_id,
    }
    provisional = StrategyMonthMaterialityEvidence.__new__(
        StrategyMonthMaterialityEvidence
    )
    for name, value in (*values.items(), ("evidence_sha256", "")):
        object.__setattr__(provisional, name, value)
    digest = _semantic_sha256(provisional.payload(False))
    return StrategyMonthMaterialityEvidence(**values, evidence_sha256=digest)


def _corporate_action_ledger_sha256(ledger: CorporateActionLedger) -> str:
    if not isinstance(ledger, CorporateActionLedger):
        raise FinalEconomicKernelError("corporate-action ledger differs")
    return _semantic_sha256(
        {
            "applied_action_ids": list(ledger.applied_action_ids),
            "last_assigned_return_date": (
                None
                if ledger.last_assigned_return_date is None
                else ledger.last_assigned_return_date.isoformat()
            ),
            "last_action_sequence": ledger.last_action_sequence,
        }
    )


def _scope_is_permitted(
    *,
    segment: FinalPeriodSegment,
    feature_scope: FeatureScope,
) -> bool:
    if segment is FinalPeriodSegment.PRIMARY_FULL:
        return feature_scope in {
            FeatureScope.FULL_FEATURE_PRIMARY,
            FeatureScope.REDUCED_AUXILIARY_CORE,
        }
    return feature_scope is FeatureScope.REDUCED_AUXILIARY_CORE


def _pnl_series_id(
    *,
    segment: FinalPeriodSegment,
    feature_scope: FeatureScope,
    strategy_id: str,
) -> str:
    if segment is FinalPeriodSegment.PRIMARY_FULL:
        prefix = (
            "stage13-primary-full"
            if feature_scope is FeatureScope.FULL_FEATURE_PRIMARY
            else "stage13-shadow-core-unreported"
        )
    elif segment is FinalPeriodSegment.CORE_COMPLETE:
        prefix = "stage13-core-complete"
    else:
        prefix = "stage13-partial-december-descriptive"
    return f"{prefix}::{strategy_id}"


def _verify_opening_inherited(
    opening_state: PortfolioState,
    inherited: InheritedHoldings,
) -> None:
    nonzero = {
        int(permno): float(value) / opening_state.nav_dollars
        for permno, value in zip(
            opening_state.permno,
            opening_state.position_value_dollars,
            strict=True,
        )
        if value != 0.0
    }
    supplied = {
        int(permno): float(weight)
        for permno, weight in zip(
            inherited.permno,
            inherited.previous_weight,
            strict=True,
        )
    }
    if set(nonzero) != set(supplied) or any(
        not np.isclose(nonzero[key], supplied[key], rtol=1e-13, atol=1e-14)
        for key in nonzero
    ):
        raise FinalEconomicKernelError(
            "inherited weights do not match the authenticated opening book"
        )


@dataclass(frozen=True, slots=True, eq=False)
class StrategyMonthInput:
    """Closed typed input to one fixed strategy/month transition."""

    strategy_id: str
    feature_scope: FeatureScope
    holding_month: str
    signal_date: dt.date
    entry_date: dt.date
    exit_date: dt.date
    scored: ScoredSnapshot
    market: MarketSnapshot
    inherited: InheritedHoldings
    risk: MonthlyRiskInputs
    opening_state: PortfolioState
    daily_returns: DailyReturnPanel | FinalDailyAccountingProjection
    expected_return_by_permno: Mapping[int, float] | None = None
    access_receipt_sha256: str | None = None
    session_authority: HoldingPeriodSessionAuthority | None = None
    initial_action_ledger: CorporateActionLedger = CorporateActionLedger()
    initial_mark_continuity_ledger: MarkContinuityLedger | None = None

    def __post_init__(self) -> None:
        definition = next(
            (
                item
                for item in FROZEN_STRATEGY_DEFINITIONS
                if item.strategy_id == self.strategy_id
            ),
            None,
        )
        if definition is None or not definition.active:
            raise FinalEconomicKernelError(
                "strategy month is not an active frozen book"
            )
        _validate_month_dates(
            holding_month=self.holding_month,
            signal_date=self.signal_date,
            entry_date=self.entry_date,
            exit_date=self.exit_date,
        )
        descriptor = final_holding_month_descriptor(self.holding_month)
        if not _scope_is_permitted(
            segment=descriptor.segment, feature_scope=self.feature_scope
        ):
            raise FinalEconomicKernelError("feature scope escaped its final segment")
        if not all(
            isinstance(value, expected)
            for value, expected in (
                (self.scored, ScoredSnapshot),
                (self.market, MarketSnapshot),
                (self.inherited, InheritedHoldings),
                (self.risk, MonthlyRiskInputs),
                (self.opening_state, PortfolioState),
            )
        ):
            raise FinalEconomicKernelError("strategy month typed input differs")
        if not isinstance(
            self.daily_returns,
            (DailyReturnPanel, FinalDailyAccountingProjection),
        ):
            raise FinalEconomicKernelError("daily accounting projection differs")
        identities = (
            self.scored.identity,
            self.market.identity,
            self.inherited.identity,
            self.risk.identity,
        )
        if any(value != identities[0] for value in identities[1:]):
            raise FinalEconomicKernelError("strategy month identities differ")
        if identities[0].signal_date != self.signal_date:
            raise FinalEconomicKernelError("strategy identity signal date differs")
        if isinstance(self.daily_returns, DailyReturnPanel):
            dates = self.daily_returns.trading_dates
            if (
                self.daily_returns.monthly_source_bundle_sha256
                != identities[0].source_bundle_sha256
            ):
                raise FinalEconomicKernelError(
                    "daily accounting source bundle differs from monthly identity"
                )
            if self.session_authority is not None:
                raise FinalEconomicKernelError(
                    "matrix daily accounting cannot carry a dated session authority"
                )
            if self.initial_mark_continuity_ledger is not None:
                raise FinalEconomicKernelError(
                    "matrix daily accounting cannot carry a mark-continuity ledger"
                )
        else:
            projection = self.daily_returns
            dates = tuple(value.trading_date for value in projection.daily_inputs)
            if (
                projection.holding_month != self.holding_month
                or projection.signal_date != self.signal_date
                or projection.entry_date != self.entry_date
                or projection.exit_date != self.exit_date
            ):
                raise FinalEconomicKernelError(
                    "dated daily accounting interval differs from strategy month"
                )
            if not isinstance(self.session_authority, HoldingPeriodSessionAuthority):
                raise FinalEconomicKernelError(
                    "dated daily accounting requires authenticated session authority"
                )
            row = self.session_authority.calendar_row
            if (
                row.holding_month != self.holding_month
                or row.signal_date != self.signal_date
                or row.entry_date != self.entry_date
                or row.exit_date != self.exit_date
            ):
                raise FinalEconomicKernelError(
                    "dated session authority interval differs from strategy month"
                )
            try:
                self.session_authority.require_exact_return_dates(dates)
            except Exception as exc:
                raise FinalEconomicKernelError(
                    "dated projection does not contain every authenticated session"
                ) from exc
            if self.access_receipt_sha256 is None:
                raise FinalEconomicKernelError(
                    "dated production accounting requires an access receipt"
                )
        if (
            not dates
            or dates[0] <= self.entry_date
            or dates[-1] != self.exit_date
            or any(
                value <= self.entry_date or value > self.exit_date for value in dates
            )
        ):
            raise FinalEconomicKernelError(
                "daily accounting dates do not span entry-close through exit-close"
            )
        if not isinstance(self.initial_action_ledger, CorporateActionLedger):
            raise FinalEconomicKernelError("initial corporate-action ledger differs")
        if self.initial_mark_continuity_ledger is not None and not isinstance(
            self.initial_mark_continuity_ledger, MarkContinuityLedger
        ):
            raise FinalEconomicKernelError("initial mark-continuity ledger differs")
        if self.access_receipt_sha256 is not None:
            _require_hash(self.access_receipt_sha256, field="access_receipt_sha256")
        _verify_opening_inherited(self.opening_state, self.inherited)
        if self.strategy_id == TRAILING_RESIDUAL_STRATEGY_ID:
            if not isinstance(self.expected_return_by_permno, Mapping):
                raise FinalEconomicKernelError(
                    "trailing-residual strategy requires direct expected returns"
                )
            normalized: dict[int, float] = {}
            for key, value in self.expected_return_by_permno.items():
                if type(key) is not int or key <= 0:
                    raise FinalEconomicKernelError(
                        "expected-return keys must be positive PERMNOs"
                    )
                normalized[key] = _float(value, field="expected_return")
            object.__setattr__(
                self,
                "expected_return_by_permno",
                MappingProxyType(normalized),
            )
        elif self.expected_return_by_permno is not None:
            raise FinalEconomicKernelError(
                "only trailing-residual strategy accepts direct expected returns"
            )


@dataclass(frozen=True, slots=True)
class StrategyMonthDecisionInput:
    """Pre-return inputs allowed before the final accounting payload is opened."""

    strategy_id: str
    feature_scope: FeatureScope
    holding_month: str
    signal_date: dt.date
    entry_date: dt.date
    exit_date: dt.date
    scored: ScoredSnapshot
    market: MarketSnapshot
    inherited: InheritedHoldings
    risk: MonthlyRiskInputs
    opening_state: PortfolioState
    decision_access_receipt_sha256: str
    expected_return_by_permno: Mapping[int, float] | None = None

    def __post_init__(self) -> None:
        _require_hash(
            self.decision_access_receipt_sha256,
            field="decision_access_receipt_sha256",
        )
        # Reuse the closed month validator with a data-independent zero-return
        # panel.  This opens no final row and cannot affect the committed target.
        dummy = _decision_validation_panel(self)
        StrategyMonthInput(
            strategy_id=self.strategy_id,
            feature_scope=self.feature_scope,
            holding_month=self.holding_month,
            signal_date=self.signal_date,
            entry_date=self.entry_date,
            exit_date=self.exit_date,
            scored=self.scored,
            market=self.market,
            inherited=self.inherited,
            risk=self.risk,
            opening_state=self.opening_state,
            daily_returns=dummy,
            expected_return_by_permno=self.expected_return_by_permno,
            access_receipt_sha256=self.decision_access_receipt_sha256,
        )


def _decision_validation_panel(
    value: StrategyMonthDecisionInput,
) -> DailyReturnPanel:
    identity_sha = value.scored.identity.source_bundle_sha256
    digest = _semantic_sha256(
        {
            "mode": "pre_accounting_decision_commitment",
            "holding_month": value.holding_month,
            "decision_access_receipt_sha256": (value.decision_access_receipt_sha256),
        }
    )
    return DailyReturnPanel.create(
        permno=value.market.permno,
        trading_dates=(value.exit_date,),
        asset_total_returns=np.zeros((1, len(value.market.permno))),
        daily_risk_free_rates=np.zeros(1),
        monthly_source_bundle_sha256=identity_sha,
        total_return_source_sha256=digest,
        risk_free_source_sha256=digest,
        corporate_action_ledger_sha256=digest,
        mark_continuity_ledger_sha256=digest,
        corporate_actions_integrated_in_total_returns=True,
    )


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    permno: tuple[int, ...]
    position_value_dollars_hex: tuple[str, ...]
    cash_dollars_hex: str
    snapshot_sha256: str

    def __post_init__(self) -> None:
        if (
            len(self.permno) != len(self.position_value_dollars_hex)
            or any(type(value) is not int or value <= 0 for value in self.permno)
            or len(set(self.permno)) != len(self.permno)
        ):
            raise FinalEconomicKernelError("portfolio snapshot identities differ")
        try:
            values = tuple(
                float.fromhex(value) for value in self.position_value_dollars_hex
            )
            cash = float.fromhex(self.cash_dollars_hex)
        except (TypeError, ValueError) as exc:
            raise FinalEconomicKernelError("portfolio snapshot floats differ") from exc
        state = PortfolioState(
            permno=np.asarray(self.permno, dtype=np.int64),
            position_value_dollars=np.asarray(values, dtype=np.float64),
            cash_dollars=cash,
        )
        del state
        _require_hash(self.snapshot_sha256, field="snapshot_sha256")
        if self.snapshot_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("portfolio snapshot digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "permno": list(self.permno),
            "position_value_dollars_hex": list(self.position_value_dollars_hex),
            "cash_dollars_hex": self.cash_dollars_hex,
        }
        if include_digest:
            value["snapshot_sha256"] = self.snapshot_sha256
        return value

    @classmethod
    def from_state(cls, state: PortfolioState) -> "PortfolioSnapshot":
        if not isinstance(state, PortfolioState):
            raise FinalEconomicKernelError("portfolio state type differs")
        values = {
            "permno": tuple(int(value) for value in state.permno),
            "position_value_dollars_hex": tuple(
                float(value).hex() for value in state.position_value_dollars
            ),
            "cash_dollars_hex": float(state.cash_dollars).hex(),
        }
        provisional = cls.__new__(cls)
        for name, value in (*values.items(), ("snapshot_sha256", "")):
            object.__setattr__(provisional, name, value)
        digest = _semantic_sha256(provisional.payload(False))
        return cls(**values, snapshot_sha256=digest)

    def to_state(self) -> PortfolioState:
        return PortfolioState(
            permno=np.asarray(self.permno, dtype=np.int64),
            position_value_dollars=np.asarray(
                [float.fromhex(value) for value in self.position_value_dollars_hex],
                dtype=np.float64,
            ),
            cash_dollars=float.fromhex(self.cash_dollars_hex),
        )


@dataclass(frozen=True, slots=True)
class KernelReceipt:
    """Immutable canonical-JSON receipt for one kernel decision."""

    receipt_kind: str
    canonical_payload_json: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.receipt_kind, str) or not self.receipt_kind:
            raise FinalEconomicKernelError("kernel receipt kind differs")
        try:
            payload = json.loads(self.canonical_payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise FinalEconomicKernelError("kernel receipt JSON differs") from exc
        if not isinstance(payload, Mapping):
            raise FinalEconomicKernelError("kernel receipt payload must be an object")
        canonical = _canonical_json_bytes(dict(payload)).decode("utf-8")
        if canonical != self.canonical_payload_json:
            raise FinalEconomicKernelError("kernel receipt JSON is not canonical")
        _require_hash(self.receipt_sha256, field="receipt_sha256")
        if self.receipt_sha256 != _hash_bytes(canonical.encode("utf-8")):
            raise FinalEconomicKernelError("kernel receipt digest differs")

    @classmethod
    def create(
        cls, receipt_kind: str, payload: Mapping[str, object]
    ) -> "KernelReceipt":
        canonical = _canonical_json_bytes(dict(payload)).decode("utf-8")
        return cls(
            receipt_kind=receipt_kind,
            canonical_payload_json=canonical,
            receipt_sha256=_hash_bytes(canonical.encode("utf-8")),
        )

    def payload(self) -> dict[str, object]:
        return {
            "receipt_kind": self.receipt_kind,
            "canonical_payload": json.loads(self.canonical_payload_json),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class StrategyAllocationEvidence:
    """The ordered position-level decision observed at the rebalance close."""

    permno: tuple[int, ...]
    score: tuple[float, ...]
    side: tuple[int, ...]
    candidate_side: tuple[str, ...]
    long_candidate: tuple[bool, ...]
    short_candidate: tuple[bool, ...]
    reduction_only: tuple[bool, ...]
    mandatory_action: tuple[bool, ...]
    current_weight: tuple[float, ...]
    target_weight: tuple[float, ...]
    trade_weight: tuple[float, ...]
    adv_trade_fraction: tuple[float, ...]
    residue_dollars: tuple[float, ...]
    row_certificate_sha256: tuple[str, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        fields = (
            self.score,
            self.side,
            self.candidate_side,
            self.long_candidate,
            self.short_candidate,
            self.reduction_only,
            self.mandatory_action,
            self.current_weight,
            self.target_weight,
            self.trade_weight,
            self.adv_trade_fraction,
            self.residue_dollars,
            self.row_certificate_sha256,
        )
        if (
            not self.permno
            or any(len(value) != len(self.permno) for value in fields)
            or any(type(value) is not int or value <= 0 for value in self.permno)
            or len(set(self.permno)) != len(self.permno)
        ):
            raise FinalEconomicKernelError("allocation evidence dimensions differ")
        if any(value not in {-1, 0, 1} for value in self.side):
            raise FinalEconomicKernelError("allocation side differs")
        allowed_classes = {
            "long_candidate",
            "short_candidate",
            "reduction_only",
            "descriptive_long",
            "descriptive_blocked",
        }
        if any(value not in allowed_classes for value in self.candidate_side):
            raise FinalEconomicKernelError("candidate classification differs")
        numeric = (
            *self.score,
            *self.current_weight,
            *self.target_weight,
            *self.trade_weight,
            *self.adv_trade_fraction,
            *self.residue_dollars,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise FinalEconomicKernelError("allocation evidence is nonfinite")
        if any(value < 0.0 for value in self.adv_trade_fraction):
            raise FinalEconomicKernelError("ADV trade fraction is negative")
        if any(
            not np.isclose(trade, target - current, rtol=1e-13, atol=1e-14)
            for current, target, trade in zip(
                self.current_weight,
                self.target_weight,
                self.trade_weight,
                strict=True,
            )
        ):
            raise FinalEconomicKernelError("allocation trade identity differs")
        for digest in self.row_certificate_sha256:
            _require_hash(digest, field="row_certificate_sha256")
        _require_hash(self.evidence_sha256, field="evidence_sha256")
        if self.evidence_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("allocation evidence digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "permno": list(self.permno),
            "score_hex": [float(item).hex() for item in self.score],
            "side": list(self.side),
            "candidate_side": list(self.candidate_side),
            "long_candidate": list(self.long_candidate),
            "short_candidate": list(self.short_candidate),
            "reduction_only": list(self.reduction_only),
            "mandatory_action": list(self.mandatory_action),
            "current_weight_hex": [float(item).hex() for item in self.current_weight],
            "target_weight_hex": [float(item).hex() for item in self.target_weight],
            "trade_weight_hex": [float(item).hex() for item in self.trade_weight],
            "adv_trade_fraction_hex": [
                float(item).hex() for item in self.adv_trade_fraction
            ],
            "residue_dollars_hex": [float(item).hex() for item in self.residue_dollars],
            "row_certificate_sha256": list(self.row_certificate_sha256),
        }
        if include_digest:
            value["evidence_sha256"] = self.evidence_sha256
        return value


@dataclass(frozen=True, slots=True)
class FinalDailyAccountingEvidence:
    """One observed dated NAV transition, including every charged cost."""

    trading_date: dt.date
    start_nav_dollars: float
    end_nav_dollars: float
    gross_pnl_dollars: float
    commission_dollars: float
    spread_dollars: float
    impact_dollars: float
    borrow_cost_dollars: float
    financing_cost_dollars: float
    total_cost_dollars: float
    net_pnl_dollars: float
    net_total_return: float
    daily_risk_free_return: float
    source_payload_sha256: str
    opening_state_sha256: str
    closing_state_sha256: str
    opening_mark_continuity_sha256: str
    closing_mark_continuity_sha256: str
    no_mark_permnos: tuple[int, ...]
    corporate_action_ids: tuple[str, ...]
    access_receipt_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if type(self.trading_date) is not dt.date:
            raise FinalEconomicKernelError("daily evidence date differs")
        for name in (
            "start_nav_dollars",
            "end_nav_dollars",
            "gross_pnl_dollars",
            "commission_dollars",
            "spread_dollars",
            "impact_dollars",
            "borrow_cost_dollars",
            "financing_cost_dollars",
            "total_cost_dollars",
            "net_pnl_dollars",
            "net_total_return",
            "daily_risk_free_return",
        ):
            object.__setattr__(self, name, _float(getattr(self, name), field=name))
        costs = (
            self.commission_dollars,
            self.spread_dollars,
            self.impact_dollars,
            self.borrow_cost_dollars,
            self.financing_cost_dollars,
        )
        if self.start_nav_dollars <= 0.0 or self.end_nav_dollars <= 0.0:
            raise FinalEconomicKernelError("daily evidence NAV is nonpositive")
        if any(value < 0.0 for value in costs):
            raise FinalEconomicKernelError("daily evidence cost is negative")
        tolerance = max(1e-7, self.start_nav_dollars * 1e-13)
        if not math.isclose(self.total_cost_dollars, sum(costs), abs_tol=tolerance):
            raise FinalEconomicKernelError("daily cost components do not reconcile")
        if not math.isclose(
            self.net_pnl_dollars,
            self.gross_pnl_dollars - self.total_cost_dollars,
            abs_tol=tolerance,
        ) or not math.isclose(
            self.end_nav_dollars,
            self.start_nav_dollars + self.net_pnl_dollars,
            abs_tol=tolerance,
        ):
            raise FinalEconomicKernelError("daily NAV/P&L does not reconcile")
        if not math.isclose(
            self.net_total_return,
            self.net_pnl_dollars / self.start_nav_dollars,
            abs_tol=1e-14,
        ):
            raise FinalEconomicKernelError("daily return does not reconcile")
        for field in (
            "source_payload_sha256",
            "opening_state_sha256",
            "closing_state_sha256",
            "opening_mark_continuity_sha256",
            "closing_mark_continuity_sha256",
            "access_receipt_sha256",
            "evidence_sha256",
        ):
            _require_hash(getattr(self, field), field=field)
        if any(type(value) is not int or value <= 0 for value in self.no_mark_permnos):
            raise FinalEconomicKernelError("daily no-mark identities differ")
        if len(set(self.no_mark_permnos)) != len(self.no_mark_permnos):
            raise FinalEconomicKernelError("daily no-mark identities repeat")
        if any(
            not isinstance(value, str) or not value
            for value in self.corporate_action_ids
        ):
            raise FinalEconomicKernelError("daily corporate-action identities differ")
        if self.evidence_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("daily evidence digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "trading_date": self.trading_date.isoformat(),
            "start_nav_dollars": self.start_nav_dollars.hex(),
            "end_nav_dollars": self.end_nav_dollars.hex(),
            "gross_pnl_dollars": self.gross_pnl_dollars.hex(),
            "commission_dollars": self.commission_dollars.hex(),
            "spread_dollars": self.spread_dollars.hex(),
            "impact_dollars": self.impact_dollars.hex(),
            "borrow_cost_dollars": self.borrow_cost_dollars.hex(),
            "financing_cost_dollars": self.financing_cost_dollars.hex(),
            "total_cost_dollars": self.total_cost_dollars.hex(),
            "net_pnl_dollars": self.net_pnl_dollars.hex(),
            "net_total_return": self.net_total_return.hex(),
            "daily_risk_free_return": self.daily_risk_free_return.hex(),
            "source_payload_sha256": self.source_payload_sha256,
            "opening_state_sha256": self.opening_state_sha256,
            "closing_state_sha256": self.closing_state_sha256,
            "opening_mark_continuity_sha256": (self.opening_mark_continuity_sha256),
            "closing_mark_continuity_sha256": (self.closing_mark_continuity_sha256),
            "no_mark_permnos": list(self.no_mark_permnos),
            "corporate_action_ids": list(self.corporate_action_ids),
            "access_receipt_sha256": self.access_receipt_sha256,
        }
        if include_digest:
            value["evidence_sha256"] = self.evidence_sha256
        return value


@dataclass(frozen=True, slots=True)
class StrategyMonthResult:
    strategy_id: str
    feature_scope: FeatureScope
    holding_month: str
    signal_date: dt.date
    entry_date: dt.date
    exit_date: dt.date
    pnl_series_id: str
    opening_portfolio_sha256: str
    closing_portfolio: PortfolioSnapshot
    problem_sha256: str
    weights_sha256: str
    allocation_architecture: str
    solver_mode: str
    solver_status: str
    net_total_return: float
    net_excess_return: float
    risk_free_total_return: float
    start_nav_dollars: float
    end_nav_dollars: float
    gross_pnl_dollars: float
    net_pnl_dollars: float
    commission_dollars: float
    spread_dollars: float
    impact_dollars: float
    transaction_cost_dollars: float
    borrow_cost_dollars: float
    financing_cost_dollars: float
    one_way_turnover: float
    gross_exposure: float
    net_exposure: float
    market_beta: float
    allocation: StrategyAllocationEvidence
    daily_accounting: tuple[FinalDailyAccountingEvidence, ...]
    terminal_event_sources: tuple[TerminalEventSource, ...]
    terminal_event_strategy_observations: tuple[TerminalEventStrategyObservation, ...]
    problem_receipt: KernelReceipt
    solver_receipt: KernelReceipt
    constraint_certificate_receipt: KernelReceipt
    interval_sha256: str
    accounting_sha256: str
    corporate_action_ledger_sha256: str
    mark_continuity_ledger_sha256: str
    closing_action_ledger: CorporateActionLedger | None
    closing_mark_continuity_ledger: MarkContinuityLedger | None
    access_receipt_sha256: str
    materiality: StrategyMonthMaterialityEvidence
    result_sha256: str

    def __post_init__(self) -> None:
        if self.strategy_id not in ACTIVE_STRATEGY_IDS:
            raise FinalEconomicKernelError("strategy result identity differs")
        _validate_month_dates(
            holding_month=self.holding_month,
            signal_date=self.signal_date,
            entry_date=self.entry_date,
            exit_date=self.exit_date,
        )
        _require_hash(self.opening_portfolio_sha256, field="opening_portfolio_sha256")
        for field in (
            "problem_sha256",
            "weights_sha256",
            "interval_sha256",
            "accounting_sha256",
            "corporate_action_ledger_sha256",
            "mark_continuity_ledger_sha256",
            "access_receipt_sha256",
        ):
            _require_hash(getattr(self, field), field=field)
        if (self.closing_action_ledger is None) != (
            self.closing_mark_continuity_ledger is None
        ):
            raise FinalEconomicKernelError(
                "closing accounting ledgers must be retained together"
            )
        if self.closing_action_ledger is not None:
            if not isinstance(self.closing_action_ledger, CorporateActionLedger):
                raise FinalEconomicKernelError(
                    "closing corporate-action ledger differs"
                )
            if not isinstance(
                self.closing_mark_continuity_ledger, MarkContinuityLedger
            ):
                raise FinalEconomicKernelError("closing mark-continuity ledger differs")
            if (
                self.corporate_action_ledger_sha256
                != _corporate_action_ledger_sha256(self.closing_action_ledger)
                or self.mark_continuity_ledger_sha256
                != self.closing_mark_continuity_ledger.ledger_sha256
            ):
                raise FinalEconomicKernelError(
                    "retained closing accounting ledger digest differs"
                )
        if not isinstance(self.closing_portfolio, PortfolioSnapshot):
            raise FinalEconomicKernelError("closing portfolio snapshot differs")
        for name in (
            "net_total_return",
            "net_excess_return",
            "risk_free_total_return",
            "start_nav_dollars",
            "end_nav_dollars",
            "gross_pnl_dollars",
            "net_pnl_dollars",
            "commission_dollars",
            "spread_dollars",
            "impact_dollars",
            "transaction_cost_dollars",
            "borrow_cost_dollars",
            "financing_cost_dollars",
            "one_way_turnover",
            "gross_exposure",
            "net_exposure",
            "market_beta",
        ):
            object.__setattr__(self, name, _float(getattr(self, name), field=name))
        if not isinstance(self.allocation, StrategyAllocationEvidence):
            raise FinalEconomicKernelError("strategy allocation evidence differs")
        if (
            not self.daily_accounting
            or any(
                not isinstance(value, FinalDailyAccountingEvidence)
                for value in self.daily_accounting
            )
            or tuple(value.trading_date for value in self.daily_accounting)
            != tuple(sorted({value.trading_date for value in self.daily_accounting}))
            or self.daily_accounting[0].trading_date <= self.entry_date
            or self.daily_accounting[-1].trading_date != self.exit_date
        ):
            raise FinalEconomicKernelError("strategy daily accounting path differs")
        for left, right in zip(self.daily_accounting, self.daily_accounting[1:]):
            if not math.isclose(
                left.end_nav_dollars,
                right.start_nav_dollars,
                abs_tol=max(1e-7, left.end_nav_dollars * 1e-13),
            ):
                raise FinalEconomicKernelError("daily NAV continuity differs")
        if (
            type(self.terminal_event_sources) is not tuple
            or type(self.terminal_event_strategy_observations) is not tuple
            or any(
                not isinstance(value, TerminalEventSource)
                for value in self.terminal_event_sources
            )
            or any(
                not isinstance(value, TerminalEventStrategyObservation)
                for value in self.terminal_event_strategy_observations
            )
        ):
            raise FinalEconomicKernelError(
                "terminal-event sensitivity evidence type differs"
            )
        source_action_ids = tuple(
            value.terminal_action_id for value in self.terminal_event_sources
        )
        observation_action_ids = tuple(
            value.terminal_action_id
            for value in self.terminal_event_strategy_observations
        )
        if (
            len(set(source_action_ids)) != len(source_action_ids)
            or source_action_ids != observation_action_ids
            or tuple(
                sorted(
                    self.terminal_event_sources,
                    key=lambda value: (
                        value.event_date,
                        value.permno,
                        value.terminal_action_id,
                    ),
                )
            )
            != self.terminal_event_sources
        ):
            raise FinalEconomicKernelError(
                "terminal-event sensitivity action identities differ"
            )
        daily_by_date = {value.trading_date: value for value in self.daily_accounting}
        target_weight_by_permno = dict(
            zip(
                self.allocation.permno,
                self.allocation.target_weight,
                strict=True,
            )
        )
        for source, observation in zip(
            self.terminal_event_sources,
            self.terminal_event_strategy_observations,
            strict=True,
        ):
            daily_row = daily_by_date.get(source.event_date)
            expected_target_weight = float(
                target_weight_by_permno.get(source.permno, 0.0)
            )
            if (
                source.holding_month != self.holding_month
                or observation.strategy_id != self.strategy_id
                or daily_row is None
                or not math.isclose(
                    observation.month_start_nav_dollars,
                    self.start_nav_dollars,
                    rel_tol=0.0,
                    abs_tol=max(1e-7, self.start_nav_dollars * 1e-13),
                )
                or not math.isclose(
                    observation.event_nav_dollars,
                    daily_row.end_nav_dollars,
                    rel_tol=0.0,
                    abs_tol=max(1e-7, daily_row.end_nav_dollars * 1e-13),
                )
                or observation.target_weight != expected_target_weight
            ):
                raise FinalEconomicKernelError(
                    "terminal-event sensitivity observation differs"
                )
        if any(
            not isinstance(value, KernelReceipt)
            for value in (
                self.problem_receipt,
                self.solver_receipt,
                self.constraint_certificate_receipt,
            )
        ):
            raise FinalEconomicKernelError("strategy receipt type differs")
        tolerance = max(1e-7, self.start_nav_dollars * 1e-13)
        sums = {
            "gross_pnl_dollars": sum(
                value.gross_pnl_dollars for value in self.daily_accounting
            ),
            "net_pnl_dollars": sum(
                value.net_pnl_dollars for value in self.daily_accounting
            ),
            "commission_dollars": sum(
                value.commission_dollars for value in self.daily_accounting
            ),
            "spread_dollars": sum(
                value.spread_dollars for value in self.daily_accounting
            ),
            "impact_dollars": sum(
                value.impact_dollars for value in self.daily_accounting
            ),
            "borrow_cost_dollars": sum(
                value.borrow_cost_dollars for value in self.daily_accounting
            ),
            "financing_cost_dollars": sum(
                value.financing_cost_dollars for value in self.daily_accounting
            ),
        }
        if any(
            not math.isclose(getattr(self, name), value, abs_tol=tolerance)
            for name, value in sums.items()
        ):
            raise FinalEconomicKernelError("daily/monthly accounting differs")
        if (
            not math.isclose(
                self.start_nav_dollars,
                self.daily_accounting[0].start_nav_dollars,
                abs_tol=tolerance,
            )
            or not math.isclose(
                self.end_nav_dollars,
                self.daily_accounting[-1].end_nav_dollars,
                abs_tol=tolerance,
            )
            or not math.isclose(
                self.closing_portfolio.to_state().nav_dollars,
                self.end_nav_dollars,
                abs_tol=tolerance,
            )
            or not math.isclose(
                self.transaction_cost_dollars,
                self.commission_dollars + self.spread_dollars + self.impact_dollars,
                abs_tol=tolerance,
            )
            or not math.isclose(
                self.net_pnl_dollars,
                self.gross_pnl_dollars
                - self.transaction_cost_dollars
                - self.borrow_cost_dollars
                - self.financing_cost_dollars,
                abs_tol=tolerance,
            )
            or not math.isclose(
                self.net_total_return,
                self.net_pnl_dollars / self.start_nav_dollars,
                abs_tol=1e-14,
            )
            or not math.isclose(
                self.net_excess_return,
                self.net_total_return - self.risk_free_total_return,
                abs_tol=1e-14,
            )
        ):
            raise FinalEconomicKernelError("strategy monthly reconciliation differs")
        if self.accounting_sha256 != _semantic_sha256(
            [value.payload() for value in self.daily_accounting]
        ):
            raise FinalEconomicKernelError("strategy accounting digest differs")
        if (
            self.materiality.strategy_id != self.strategy_id
            or self.materiality.holding_month != self.holding_month
            or not math.isclose(
                self.materiality.aggregate_absolute_dollars,
                sum(abs(value) for value in self.allocation.residue_dollars),
                abs_tol=1e-14,
            )
        ):
            raise FinalEconomicKernelError("strategy materiality evidence differs")
        _require_hash(self.result_sha256, field="result_sha256")
        if self.result_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("strategy result digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "strategy_id": self.strategy_id,
            "feature_scope": self.feature_scope.value,
            "holding_month": self.holding_month,
            "signal_date": self.signal_date.isoformat(),
            "entry_date": self.entry_date.isoformat(),
            "exit_date": self.exit_date.isoformat(),
            "pnl_series_id": self.pnl_series_id,
            "opening_portfolio_sha256": self.opening_portfolio_sha256,
            "closing_portfolio": self.closing_portfolio.payload(),
            "problem_sha256": self.problem_sha256,
            "weights_sha256": self.weights_sha256,
            "allocation_architecture": self.allocation_architecture,
            "solver_mode": self.solver_mode,
            "solver_status": self.solver_status,
            "net_total_return": self.net_total_return.hex(),
            "net_excess_return": self.net_excess_return.hex(),
            "risk_free_total_return": self.risk_free_total_return.hex(),
            "start_nav_dollars": self.start_nav_dollars.hex(),
            "end_nav_dollars": self.end_nav_dollars.hex(),
            "gross_pnl_dollars": self.gross_pnl_dollars.hex(),
            "net_pnl_dollars": self.net_pnl_dollars.hex(),
            "commission_dollars": self.commission_dollars.hex(),
            "spread_dollars": self.spread_dollars.hex(),
            "impact_dollars": self.impact_dollars.hex(),
            "transaction_cost_dollars": self.transaction_cost_dollars.hex(),
            "borrow_cost_dollars": self.borrow_cost_dollars.hex(),
            "financing_cost_dollars": self.financing_cost_dollars.hex(),
            "one_way_turnover": self.one_way_turnover.hex(),
            "gross_exposure": self.gross_exposure.hex(),
            "net_exposure": self.net_exposure.hex(),
            "market_beta": self.market_beta.hex(),
            "allocation": self.allocation.payload(),
            "daily_accounting": [value.payload() for value in self.daily_accounting],
            "terminal_event_sources": [
                value.payload() for value in self.terminal_event_sources
            ],
            "terminal_event_strategy_observations": [
                value.payload() for value in self.terminal_event_strategy_observations
            ],
            "problem_receipt": self.problem_receipt.payload(),
            "solver_receipt": self.solver_receipt.payload(),
            "constraint_certificate_receipt": (
                self.constraint_certificate_receipt.payload()
            ),
            "interval_sha256": self.interval_sha256,
            "accounting_sha256": self.accounting_sha256,
            "corporate_action_ledger_sha256": (self.corporate_action_ledger_sha256),
            "mark_continuity_ledger_sha256": (self.mark_continuity_ledger_sha256),
            "access_receipt_sha256": self.access_receipt_sha256,
            "materiality": self.materiality.payload(),
        }
        if include_digest:
            value["result_sha256"] = self.result_sha256
        return value


@dataclass(frozen=True, slots=True)
class StrategyMonthDecisionCommitment:
    """Hash-bound target fixed before any holding-period return is opened."""

    decision_input: StrategyMonthDecisionInput
    opening_portfolio_sha256: str
    problem_sha256: str
    weights_sha256: str
    allocation_architecture: str
    solver_mode: str
    solver_status: str
    allocation: StrategyAllocationEvidence
    problem_receipt: KernelReceipt
    solver_receipt: KernelReceipt
    constraint_certificate_receipt: KernelReceipt
    decision_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.decision_input, StrategyMonthDecisionInput):
            raise FinalEconomicKernelError("decision commitment input differs")
        for field in (
            "opening_portfolio_sha256",
            "problem_sha256",
            "weights_sha256",
            "decision_sha256",
        ):
            _require_hash(getattr(self, field), field=field)
        if not isinstance(self.allocation, StrategyAllocationEvidence) or any(
            not isinstance(value, KernelReceipt)
            for value in (
                self.problem_receipt,
                self.solver_receipt,
                self.constraint_certificate_receipt,
            )
        ):
            raise FinalEconomicKernelError("decision commitment evidence differs")
        if self.decision_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("decision commitment digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        source = self.decision_input
        value: dict[str, object] = {
            "strategy_id": source.strategy_id,
            "feature_scope": source.feature_scope.value,
            "holding_month": source.holding_month,
            "signal_date": source.signal_date.isoformat(),
            "entry_date": source.entry_date.isoformat(),
            "exit_date": source.exit_date.isoformat(),
            "monthly_source_bundle_sha256": (
                source.scored.identity.source_bundle_sha256
            ),
            "decision_access_receipt_sha256": (source.decision_access_receipt_sha256),
            "opening_portfolio_sha256": self.opening_portfolio_sha256,
            "problem_sha256": self.problem_sha256,
            "weights_sha256": self.weights_sha256,
            "allocation_architecture": self.allocation_architecture,
            "solver_mode": self.solver_mode,
            "solver_status": self.solver_status,
            "allocation": self.allocation.payload(),
            "problem_receipt": self.problem_receipt.payload(),
            "solver_receipt": self.solver_receipt.payload(),
            "constraint_certificate_receipt": (
                self.constraint_certificate_receipt.payload()
            ),
            "accounting_payload_opened": False,
        }
        if include_digest:
            value["decision_sha256"] = self.decision_sha256
        return value


def _aligned_market_vector(
    market: MarketSnapshot,
    target_permno: np.ndarray,
    field_name: str,
) -> np.ndarray:
    index = {int(value): position for position, value in enumerate(market.permno)}
    missing = sorted(int(value) for value in target_permno if int(value) not in index)
    if missing:
        raise FinalEconomicKernelError(
            f"market input omits target securities for {field_name}: "
            + ", ".join(str(value) for value in missing)
        )
    positions = np.asarray([index[int(value)] for value in target_permno])
    return np.asarray(getattr(market, field_name)[positions])


def _problem_semantic_sha(problem: object) -> str:
    names = (
        "permno",
        "permco",
        "side",
        "previous_weight",
        "expected_return",
        "score",
        "factor_betas",
        "factor_covariance_monthly",
        "specific_variance_monthly",
        "forecast_volatility",
        "full_spread",
        "adv20_usd",
        "impact_volatility_daily",
        "trade_capacity_weight",
        "sector",
        "country",
    )
    payload: dict[str, object] = {}
    for name in names:
        payload[name] = _array_payload(np.asarray(getattr(problem, name)))
    payload["nav_dollars"] = float(getattr(problem, "nav_dollars")).hex()
    payload["market_variance_monthly"] = float(
        getattr(problem, "market_variance_monthly")
    ).hex()
    return _semantic_sha256(payload)


def _aligned_state_weights(
    state: PortfolioState,
    target_permno: np.ndarray,
) -> np.ndarray:
    index = {int(value): position for position, value in enumerate(state.permno)}
    result = np.zeros(len(target_permno), dtype=np.float64)
    for target_index, raw_permno in enumerate(target_permno):
        position = index.get(int(raw_permno))
        if position is not None:
            result[target_index] = (
                float(state.position_value_dollars[position]) / state.nav_dollars
            )
    return result


def _aligned_projection_vector(
    projection: FinalDailyAccountingProjection,
    target_permno: np.ndarray,
    field_name: str,
    *,
    missing_permitted: np.ndarray | None = None,
) -> np.ndarray:
    index = {
        int(value): position
        for position, value in enumerate(projection.execution_permno)
    }
    present = np.asarray(
        [int(value) in index for value in target_permno], dtype=np.bool_
    )
    if missing_permitted is None:
        permitted = np.zeros(len(target_permno), dtype=np.bool_)
    else:
        permitted = np.asarray(missing_permitted, dtype=np.bool_)
        if permitted.shape != (len(target_permno),):
            raise FinalEconomicKernelError(
                "dated accounting missing-row permission differs"
            )
    prohibited_missing = ~present & ~permitted
    if np.any(prohibited_missing):
        missing = sorted(int(value) for value in target_permno[prohibited_missing])
        raise FinalEconomicKernelError(
            "dated accounting projection omits target securities: "
            + ", ".join(str(value) for value in missing)
        )
    result = np.full(len(target_permno), np.nan, dtype=np.float64)
    positions = np.asarray(
        [index[int(value)] for value in target_permno[present]], dtype=np.int64
    )
    result[present] = np.asarray(
        getattr(projection, field_name)[positions], dtype=np.float64
    )
    return result


def _transaction_cost_components(
    *,
    nav_dollars: float,
    trade_weight: np.ndarray,
    full_spread: np.ndarray,
    adv_dollars: np.ndarray,
    impact_volatility_daily: np.ndarray,
) -> tuple[float, float, float]:
    absolute = np.abs(np.asarray(trade_weight, dtype=np.float64))
    active = absolute != 0.0
    commission = nav_dollars * COMMISSION_RATE * float(np.sum(absolute))
    spread = nav_dollars * float(
        np.sum(np.where(active, full_spread / 2.0 * absolute, 0.0))
    )
    impact = nav_dollars * float(
        np.sum(
            np.where(
                active,
                0.10
                * impact_volatility_daily
                * np.sqrt(nav_dollars / adv_dollars)
                * np.power(absolute, 1.5),
                0.0,
            )
        )
    )
    return commission, spread, impact


def _position_residue(
    residue_dollars: np.ndarray,
    position_count: int,
) -> tuple[float, ...]:
    if len(residue_dollars) > position_count:
        raise FinalEconomicKernelError(
            "strategy residue count exceeds retained allocation positions"
        )
    return tuple(float(value) for value in residue_dollars) + (0.0,) * (
        position_count - len(residue_dollars)
    )


def _make_allocation_evidence(
    *,
    problem_sha256: str,
    target_permno: np.ndarray,
    score: np.ndarray,
    side: np.ndarray,
    candidate_side: Sequence[str],
    long_candidate: np.ndarray,
    short_candidate: np.ndarray,
    reduction_only: np.ndarray,
    mandatory_action: np.ndarray,
    current_weight: np.ndarray,
    target_weight: np.ndarray,
    adv_dollars: np.ndarray,
    nav_dollars: float,
    residue_dollars: np.ndarray,
) -> StrategyAllocationEvidence:
    trade = np.asarray(target_weight, dtype=np.float64) - np.asarray(
        current_weight, dtype=np.float64
    )
    adv_fraction = np.divide(
        np.abs(trade) * nav_dollars,
        adv_dollars,
        out=np.zeros_like(trade),
        where=np.asarray(adv_dollars) > 0.0,
    )
    residues = _position_residue(residue_dollars, len(target_permno))
    certificates = tuple(
        _semantic_sha256(
            {
                "allocation_problem_sha256": problem_sha256,
                "permno": int(permno),
                "current_weight": float(current).hex(),
                "target_weight": float(target).hex(),
                "trade_weight": float(delta).hex(),
            }
        )
        for permno, current, target, delta in zip(
            target_permno,
            current_weight,
            target_weight,
            trade,
            strict=True,
        )
    )
    values = {
        "permno": tuple(int(value) for value in target_permno),
        "score": tuple(float(value) for value in score),
        "side": tuple(int(value) for value in side),
        "candidate_side": tuple(str(value) for value in candidate_side),
        "long_candidate": tuple(bool(value) for value in long_candidate),
        "short_candidate": tuple(bool(value) for value in short_candidate),
        "reduction_only": tuple(bool(value) for value in reduction_only),
        "mandatory_action": tuple(bool(value) for value in mandatory_action),
        "current_weight": tuple(float(value) for value in current_weight),
        "target_weight": tuple(float(value) for value in target_weight),
        "trade_weight": tuple(float(value) for value in trade),
        "adv_trade_fraction": tuple(float(value) for value in adv_fraction),
        "residue_dollars": residues,
        "row_certificate_sha256": certificates,
    }
    provisional = StrategyAllocationEvidence.__new__(StrategyAllocationEvidence)
    for name, value in (*values.items(), ("evidence_sha256", "")):
        object.__setattr__(provisional, name, value)
    return StrategyAllocationEvidence(
        **values,
        evidence_sha256=_semantic_sha256(provisional.payload(False)),
    )


def _make_daily_evidence(**values: object) -> FinalDailyAccountingEvidence:
    provisional = FinalDailyAccountingEvidence.__new__(FinalDailyAccountingEvidence)
    for name, value in (*values.items(), ("evidence_sha256", "")):
        object.__setattr__(provisional, name, value)
    return FinalDailyAccountingEvidence(
        **values,  # type: ignore[arg-type]
        evidence_sha256=_semantic_sha256(provisional.payload(False)),
    )


def _capture_terminal_event_sensitivity(
    *,
    month: StrategyMonthInput,
    projection: FinalDailyAccountingProjection,
    dated: DatedHoldingPeriodAccounting,
    allocation: StrategyAllocationEvidence,
) -> tuple[
    tuple[TerminalEventSource, ...],
    tuple[TerminalEventStrategyObservation, ...],
]:
    """Retain reporting-only terminal-event inputs from primary accounting.

    Held actions have one exact post-return/pre-settlement receipt.  Unheld
    actions intentionally have no action receipt, but still produce an
    explicit zero strategy observation.  Both classifications must appear
    exactly once in the dated session receipts.
    """

    if not isinstance(dated, DatedHoldingPeriodAccounting):
        raise FinalEconomicKernelError("dated terminal-event accounting differs")
    action_ids = tuple(action.action_id for action in projection.corporate_actions)
    if len(set(action_ids)) != len(action_ids):
        raise FinalEconomicKernelError("corporate-action identities repeat")
    terminal_actions = tuple(
        action
        for action in projection.corporate_actions
        if any(
            value is not None
            for value in (
                action.terminal_sensitivity_policy_sha256,
                action.terminal_sensitivity_evidence_sha256,
                action.event_day_quoted_full_spread,
            )
        )
    )
    if not terminal_actions:
        return (), ()
    terminal_actions = tuple(
        sorted(
            terminal_actions,
            key=lambda action: (
                action.assigned_return_date,
                action.source_permno,
                action.action_id,
            ),
        )
    )

    if len(projection.daily_inputs) != len(dated.session_receipts) or len(
        projection.daily_inputs
    ) != len(dated.daily_nav_dollars):
        raise FinalEconomicKernelError("terminal-event daily path differs")
    date_to_index: dict[dt.date, int] = {}
    for index, (day, session) in enumerate(
        zip(projection.daily_inputs, dated.session_receipts, strict=True)
    ):
        if (
            day.trading_date in date_to_index
            or session.trading_date != day.trading_date
        ):
            raise FinalEconomicKernelError("terminal-event session identity differs")
        date_to_index[day.trading_date] = index

    execution_permno = np.asarray(projection.execution_permno, dtype=np.int64)
    execution_spread = np.asarray(projection.full_spread, dtype=np.float64)
    if execution_permno.ndim != 1 or execution_spread.shape != execution_permno.shape:
        raise FinalEconomicKernelError("terminal-event spread projection differs")
    target_weight_by_permno = dict(
        zip(allocation.permno, allocation.target_weight, strict=True)
    )
    start_nav = float(month.opening_state.nav_dollars)
    if not math.isclose(
        float(dated.pre_trade_nav_dollars),
        start_nav,
        rel_tol=0.0,
        abs_tol=max(1e-7, start_nav * 1e-13),
    ):
        raise FinalEconomicKernelError("terminal-event month-start NAV differs")

    sources: list[TerminalEventSource] = []
    observations: list[TerminalEventStrategyObservation] = []
    for action in terminal_actions:
        if (
            action.terminal_sensitivity_policy_sha256 is None
            or action.terminal_sensitivity_evidence_sha256 is None
            or action.event_day_quoted_full_spread is None
            or action.settlement_kind != CASH_SETTLEMENT
            or action.successor_permno is not None
            or action.settlement_value_multiplier != 1.0
            or action.terminal_total_return_already_applied is not True
        ):
            raise FinalEconomicKernelError(
                "terminal-event action metadata or multiplier differs"
            )
        event_index = date_to_index.get(action.assigned_return_date)
        if event_index is None:
            raise FinalEconomicKernelError("terminal-event session is missing")

        held_hits: list[int] = []
        unheld_hits: list[int] = []
        held_occurrences = 0
        unheld_occurrences = 0
        for index, session in enumerate(dated.session_receipts):
            held_count = session.corporate_action_ids.count(action.action_id)
            unheld_count = session.unheld_corporate_action_ids.count(action.action_id)
            held_occurrences += held_count
            unheld_occurrences += unheld_count
            if held_count:
                held_hits.append(index)
            if unheld_count:
                unheld_hits.append(index)
        if (
            held_occurrences + unheld_occurrences != 1
            or bool(held_occurrences) == bool(unheld_occurrences)
            or (held_hits or unheld_hits) != [event_index]
        ):
            raise FinalEconomicKernelError(
                "terminal-event held/unheld classification differs"
            )
        position_held = held_occurrences == 1
        matching_receipts = tuple(
            receipt
            for receipt in dated.corporate_action_receipts
            if receipt.action_id == action.action_id
        )
        if position_held:
            if len(matching_receipts) != 1:
                raise FinalEconomicKernelError(
                    "held terminal event lacks one exact action receipt"
                )
            receipt = matching_receipts[0]
            if (
                receipt.economic_event_date != action.economic_event_date
                or receipt.assigned_return_date != action.assigned_return_date
                or receipt.action_sequence != action.action_sequence
                or receipt.source_record_sha256 != action.source_record_sha256
                or receipt.source_permno != action.source_permno
                or receipt.settlement_kind != action.settlement_kind
                or receipt.successor_permno != action.successor_permno
                or receipt.terminal_sensitivity_policy_sha256
                != action.terminal_sensitivity_policy_sha256
                or receipt.terminal_sensitivity_evidence_sha256
                != action.terminal_sensitivity_evidence_sha256
                or receipt.event_day_quoted_full_spread
                != action.event_day_quoted_full_spread
                or receipt.settled_value_dollars != receipt.opening_source_value_dollars
            ):
                raise FinalEconomicKernelError(
                    "terminal-event action receipt differs from its action"
                )
            signed_event_value = float(receipt.opening_source_value_dollars)
            if not math.isfinite(signed_event_value):
                raise FinalEconomicKernelError(
                    "terminal-event signed position value is nonfinite"
                )
        else:
            if matching_receipts:
                raise FinalEconomicKernelError(
                    "unheld terminal event unexpectedly has an action receipt"
                )
            signed_event_value = 0.0

        spread_rows = np.flatnonzero(execution_permno == action.source_permno)
        if spread_rows.size != 1:
            raise FinalEconomicKernelError(
                "terminal-event frozen spread identity is missing or duplicated"
            )
        frozen_spread = float(execution_spread[int(spread_rows[0])])
        quoted_spread = float(action.event_day_quoted_full_spread)
        if (
            not math.isfinite(frozen_spread)
            or frozen_spread < 0.0
            or not math.isfinite(quoted_spread)
            or quoted_spread < 0.0
        ):
            raise FinalEconomicKernelError("terminal-event spread is nonfinite")
        remaining_growth = math.prod(
            1.0 + float(day.daily_risk_free_rate)
            for day in projection.daily_inputs[event_index + 1 :]
        )
        if not math.isfinite(remaining_growth) or remaining_growth <= 0.0:
            raise FinalEconomicKernelError(
                "terminal-event remaining cash growth differs"
            )
        target_weight = float(target_weight_by_permno.get(action.source_permno, 0.0))
        if not position_held and target_weight != 0.0:
            raise FinalEconomicKernelError(
                "unheld terminal event has a nonzero target weight"
            )
        event_nav = float(dated.daily_nav_dollars[event_index])
        if not math.isfinite(event_nav) or event_nav <= 0.0:
            raise FinalEconomicKernelError("terminal-event NAV differs")
        sources.append(
            TerminalEventSource(
                terminal_action_id=action.action_id,
                permno=action.source_permno,
                event_date=action.assigned_return_date,
                holding_month=month.holding_month,
                source_record_sha256=action.source_record_sha256,
                terminal_policy_sha256=(action.terminal_sensitivity_policy_sha256),
                terminal_evidence_sha256=(action.terminal_sensitivity_evidence_sha256),
                event_day_quoted_full_spread=quoted_spread,
                frozen_full_spread_20d=frozen_spread,
                remaining_cash_growth=remaining_growth,
            )
        )
        observations.append(
            TerminalEventStrategyObservation(
                terminal_action_id=action.action_id,
                strategy_id=month.strategy_id,
                position_held=position_held,
                target_weight=target_weight,
                month_start_nav_dollars=start_nav,
                event_nav_dollars=event_nav,
                signed_event_value_dollars=signed_event_value,
            )
        )
    return tuple(sources), tuple(observations)


def execute_certified_strategy_month(
    month: StrategyMonthInput,
) -> StrategyMonthResult:
    """Execute and retain one complete, dated strategy-month ledger."""

    if not isinstance(month, StrategyMonthInput):
        raise FinalEconomicKernelError("month must be StrategyMonthInput")
    descriptor = final_holding_month_descriptor(month.holding_month)
    opening_snapshot = PortfolioSnapshot.from_state(month.opening_state)
    access_sha = month.access_receipt_sha256 or _semantic_sha256(
        {
            "mode": "closed_in_memory_synthetic_or_preflight",
            "monthly_source_bundle_sha256": month.scored.identity.source_bundle_sha256,
        }
    )
    interval_sha = _semantic_sha256(
        {
            "holding_month": month.holding_month,
            "signal_date": month.signal_date.isoformat(),
            "entry_date": month.entry_date.isoformat(),
            "exit_date": month.exit_date.isoformat(),
            "return_dates": [
                value.isoformat()
                for value in (
                    month.daily_returns.trading_dates
                    if isinstance(month.daily_returns, DailyReturnPanel)
                    else tuple(
                        item.trading_date for item in month.daily_returns.daily_inputs
                    )
                )
            ],
        }
    )

    built = None
    allocation_result = None
    if month.strategy_id == EQUAL_WEIGHT_STRATEGY_ID:
        # The descriptive book is rebalanced over the current scoring
        # universe, but its execution domain must retain every inherited
        # nonzero holding.  A name that disappears from the next score
        # cross-section is therefore represented as an explicit zero-target
        # liquidation instead of being silently omitted from accounting.
        scored_index = {
            int(value): index for index, value in enumerate(month.scored.permno)
        }
        current_eligible = {
            int(value)
            for value, eligible in zip(
                month.scored.permno,
                month.scored.long_eligible,
                strict=True,
            )
            if bool(eligible)
        }
        inherited_index = {
            int(value): index for index, value in enumerate(month.inherited.permno)
        }
        target_permno = np.asarray(
            sorted(current_eligible.union(inherited_index)),
            dtype=np.int64,
        )
        scored_row = np.asarray(
            [scored_index.get(int(value), -1) for value in target_permno],
            dtype=np.int64,
        )
        inherited_row = np.asarray(
            [inherited_index.get(int(value), -1) for value in target_permno],
            dtype=np.int64,
        )
        has_score = scored_row >= 0
        has_inherited = inherited_row >= 0
        aligned_long_eligible = np.zeros(len(target_permno), dtype=np.bool_)
        aligned_score = np.zeros(len(target_permno), dtype=np.float64)
        aligned_long_eligible[has_score] = np.asarray(
            month.scored.long_eligible[scored_row[has_score]], dtype=np.bool_
        )
        scored_and_eligible = has_score & aligned_long_eligible
        aligned_score[scored_and_eligible] = np.asarray(
            month.scored.score[scored_row[scored_and_eligible]],
            dtype=np.float64,
        )
        current_weight = _aligned_state_weights(month.opening_state, target_permno)
        aligned_new_long_trade_allowed = np.asarray(
            _aligned_market_vector(
                month.market,
                target_permno,
                "new_long_trade_allowed",
            ),
            dtype=np.bool_,
        )
        aligned_execution_price = np.asarray(
            _aligned_market_vector(month.market, target_permno, "execution_price"),
            dtype=np.float64,
        )
        aligned_full_spread = np.asarray(
            _aligned_market_vector(month.market, target_permno, "full_spread"),
            dtype=np.float64,
        )
        aligned_adv = np.asarray(
            _aligned_market_vector(month.market, target_permno, "adv20_usd"),
            dtype=np.float64,
        )
        aligned_impact = np.asarray(
            _aligned_market_vector(
                month.market,
                target_permno,
                "impact_volatility_daily",
            ),
            dtype=np.float64,
        )
        valid_execution_cost = (
            np.isfinite(aligned_execution_price)
            & (aligned_execution_price > 0.0)
            & np.isfinite(aligned_full_spread)
            & (aligned_full_spread >= 0.0)
            & np.isfinite(aligned_adv)
            & (aligned_adv > 0.0)
            & np.isfinite(aligned_impact)
            & (aligned_impact >= 0.0)
        )
        aligned_factor_betas = np.asarray(
            _aligned_market_vector(month.market, target_permno, "factor_betas"),
            dtype=np.float64,
        )
        aligned_specific_variance = np.asarray(
            _aligned_market_vector(
                month.market,
                target_permno,
                "specific_variance_monthly",
            ),
            dtype=np.float64,
        )
        valid_current_risk = (
            np.isfinite(aligned_factor_betas).all(axis=1)
            & np.isfinite(aligned_specific_variance)
            & (aligned_specific_variance > 0.0)
        )
        aligned_required_exit = np.zeros(len(target_permno), dtype=np.bool_)
        aligned_required_exit[has_inherited] = np.asarray(
            month.inherited.required_exit[inherited_row[has_inherited]],
            dtype=np.bool_,
        )
        executable_long = (
            aligned_long_eligible
            & aligned_new_long_trade_allowed
            & valid_execution_cost
            & valid_current_risk
            & ~aligned_required_exit
        )
        frozen_exact_hold = (
            has_inherited & (current_weight != 0.0) & ~valid_execution_cost
        )
        if np.any(current_weight[frozen_exact_hold] < 0.0):
            raise FinalEconomicKernelError(
                "equal-weight missing-cost hold is not long-only"
            )
        frozen_gross = float(np.sum(current_weight[frozen_exact_hold]))
        if not math.isfinite(frozen_gross) or frozen_gross > 1.0:
            raise FinalEconomicKernelError(
                "equal-weight frozen holdings exceed 100 percent gross"
            )
        residual_gross = max(0.0, 1.0 - frozen_gross)
        executable_count = int(np.sum(executable_long))
        if executable_count == 0 and residual_gross != 0.0:
            raise FinalEconomicKernelError(
                "equal-weight residual gross lacks an executable long universe"
            )
        weights = np.zeros(len(target_permno), dtype=np.float64)
        weights[frozen_exact_hold] = current_weight[frozen_exact_hold]
        if executable_count:
            weights += residual_gross * equal_weight_long_only_targets(
                tuple(int(value) for value in target_permno),
                eligible=tuple(bool(value) for value in executable_long),
            )
        score = aligned_score
        side = np.where(aligned_long_eligible, 1, 0).astype(np.int8)
        long_candidate = np.asarray(executable_long, dtype=np.bool_)
        short_candidate = np.zeros(len(target_permno), dtype=np.bool_)
        reduction_only = has_inherited & ~executable_long
        mandatory_action = aligned_required_exit
        candidate_side = tuple(
            (
                "reduction_only"
                if bool(reduction)
                else "descriptive_long" if bool(executable) else "descriptive_blocked"
            )
            for reduction, executable in zip(
                reduction_only,
                executable_long,
                strict=True,
            )
        )
        raw_market_beta = np.asarray(aligned_factor_betas[:, 0], dtype=np.float64)
        inherited_market_beta = np.full(len(target_permno), np.nan, dtype=np.float64)
        inherited_market_beta[has_inherited] = np.asarray(
            month.inherited.last_factor_betas[
                inherited_row[has_inherited],
                0,
            ],
            dtype=np.float64,
        )
        inherited_market_beta_fallback = (
            (weights != 0.0)
            & frozen_exact_hold
            & ~np.isfinite(raw_market_beta)
            & np.isfinite(inherited_market_beta)
        )
        resolved_market_beta = raw_market_beta.copy()
        resolved_market_beta[inherited_market_beta_fallback] = inherited_market_beta[
            inherited_market_beta_fallback
        ]
        if np.any((weights != 0.0) & ~np.isfinite(resolved_market_beta)):
            raise FinalEconomicKernelError(
                "equal-weight target lacks a finite authenticated market beta"
            )
        market_beta_vector = np.where(weights == 0.0, 0.0, resolved_market_beta)
        problem_sha = _semantic_sha256(
            {
                "architecture": "descriptive_100_percent_gross_long_only",
                "permno": _array_payload(target_permno),
                "current_weight": _array_payload(current_weight),
                "weights": _array_payload(weights),
                "reduction_only": _array_payload(reduction_only),
                "new_long_trade_allowed": _array_payload(
                    aligned_new_long_trade_allowed
                ),
                "valid_execution_cost": _array_payload(valid_execution_cost),
                "valid_current_risk": _array_payload(valid_current_risk),
                "executable_long": _array_payload(executable_long),
                "frozen_exact_hold": _array_payload(frozen_exact_hold),
                "required_exit": _array_payload(aligned_required_exit),
                "raw_market_beta_finite": _array_payload(np.isfinite(raw_market_beta)),
                "inherited_market_beta_fallback": _array_payload(
                    inherited_market_beta_fallback
                ),
                "market_beta_vector": _array_payload(market_beta_vector),
            }
        )
        architecture = "descriptive_100_percent_gross_long_only"
        solver_mode = "deterministic_closed_form"
        solver_status = "not_applicable"
        borrow_rate = np.zeros(len(target_permno), dtype=np.float64)
        eligible_count = executable_count
        candidate_payload: dict[str, object] = {
            "eligible_universe_count": len(target_permno),
            "candidate_long_count": eligible_count,
            "candidate_short_count": 0,
            "reduction_only_count": int(np.sum(reduction_only)),
            "minimum_new_names_each_side": 50,
            "minimum_each_side_applies": False,
            "tail_extension_used": False,
            "insufficient_side_fallback_used": False,
            "new_risk_permitted": executable_count > 0 and residual_gross > 0.0,
            "candidate_sets_disjoint": True,
        }
        solver_payload: dict[str, object] = {
            "architecture": architecture,
            "mode": solver_mode,
            "status": solver_status,
            "status_history": ["deterministic_closed_form"],
        }
        certificate_payload: dict[str, object] = {
            "hard_constraint_violation_count": 0,
            "maximum_constraint_violation": 0.0,
            "normalized_slacks": {},
            "binding_trade_limits": [],
        }
        residue_dollars = np.zeros(len(target_permno), dtype=np.float64)
    else:
        built = build_monthly_problem(
            month.scored,
            month.market,
            month.inherited,
            month.risk,
            nav_dollars=month.opening_state.nav_dollars,
        )
        problem = built.problem
        if month.strategy_id == TRAILING_RESIDUAL_STRATEGY_ID:
            assert month.expected_return_by_permno is not None
            problem = replace(
                problem,
                expected_return=np.asarray(
                    [
                        month.expected_return_by_permno.get(int(permno), 0.0)
                        for permno in problem.permno
                    ],
                    dtype=np.float64,
                ),
            )
        if month.strategy_id == RANK_STRATEGY_ID:
            _, allocation_result = allocate_rank(
                problem,
                long_entry_boundary=built.selection.long_entry_boundary,
                short_entry_boundary=built.selection.short_entry_boundary,
            )
        else:
            allocation_result = solve_hybrid(problem)
        target_permno = np.asarray(problem.permno, dtype=np.int64)
        weights = np.asarray(allocation_result.weights, dtype=np.float64)
        score = np.asarray(problem.score, dtype=np.float64)
        side = np.asarray(problem.side, dtype=np.int8)
        indices = built.selection.solver_indices
        long_candidate = np.asarray(
            built.selection.long_candidate[indices], dtype=np.bool_
        )
        short_candidate = np.asarray(
            built.selection.short_candidate[indices], dtype=np.bool_
        )
        reduction_only = np.asarray(problem.reduction_only, dtype=np.bool_)
        mandatory_action = np.asarray(
            problem.mandatory_corporate_action, dtype=np.bool_
        )
        candidate_side = tuple(
            (
                "reduction_only"
                if bool(reduction)
                else "long_candidate" if int(raw_side) > 0 else "short_candidate"
            )
            for raw_side, reduction in zip(side, reduction_only, strict=True)
        )
        problem_sha = _problem_semantic_sha(problem)
        architecture = allocation_result.architecture
        solver_mode = allocation_result.mode
        solver_status = allocation_result.solver_status
        borrow_rate = np.asarray(problem.borrow_rate_annual, dtype=np.float64)
        market_beta_vector = np.asarray(problem.factor_betas[:, 0], dtype=np.float64)
        candidate_payload = {
            "eligible_universe_count": built.universe.n_assets,
            "candidate_long_count": sum(
                value == "long_candidate" for value in candidate_side
            ),
            "candidate_short_count": sum(
                value == "short_candidate" for value in candidate_side
            ),
            "reduction_only_count": sum(
                value == "reduction_only" for value in candidate_side
            ),
            "minimum_new_names_each_side": built.selection.minimum_candidates,
            "minimum_each_side_applies": True,
            "tail_extension_used": (
                built.selection.tail_size < built.selection.available_long_candidates
                or built.selection.tail_size
                < built.selection.available_short_candidates
            ),
            "insufficient_side_fallback_used": not built.selection.new_risk_allowed,
            "new_risk_permitted": built.selection.new_risk_allowed,
            "candidate_sets_disjoint": built.selection.overlap_count == 0,
            "selection_status": built.selection.status,
        }
        diagnostics = allocation_result.diagnostics
        solver_payload = {
            "architecture": architecture,
            "mode": solver_mode,
            "status": solver_status,
            "status_history": list(allocation_result.solver_status_history),
            "solver_attempt": allocation_result.solver_attempt,
            "objective_value": allocation_result.objective_value,
            "transaction_cost_return": allocation_result.transaction_cost,
            "borrow_cost_return": allocation_result.borrow_cost,
            "financing_cost_return": allocation_result.financing_cost,
            "total_cost_return": allocation_result.total_cost,
            "cash_weight": allocation_result.cash_weight,
        }
        certificate_payload = {
            "long_gross": diagnostics.long_gross,
            "short_gross": diagnostics.short_gross,
            "gross": diagnostics.gross,
            "net": diagnostics.net,
            "one_way_turnover": diagnostics.one_way_turnover,
            "augmented_annual_variance": diagnostics.augmented_annual_variance,
            "augmented_annual_volatility": diagnostics.augmented_annual_volatility,
            "factor_exposures": list(diagnostics.factor_exposures),
            "violations": dict(diagnostics.violations),
            "normalized_violations": dict(diagnostics.normalized_violations),
            "normalized_slacks": dict(allocation_result.normalized_slacks),
            "binding_trade_limits": list(allocation_result.binding_trade_limits),
            "estimated_months_to_compliance": (
                allocation_result.estimated_months_to_compliance
            ),
            "hard_constraint_violation_count": 0,
            "maximum_constraint_violation": diagnostics.maximum_violation,
        }
        residue_dollars = np.asarray(
            allocation_result.residue_dollars, dtype=np.float64
        )

    current_weight = _aligned_state_weights(month.opening_state, target_permno)
    if isinstance(month.daily_returns, DailyReturnPanel):
        execution_price = _aligned_market_vector(
            month.market, target_permno, "execution_price"
        )
        full_spread = _aligned_market_vector(month.market, target_permno, "full_spread")
        adv = _aligned_market_vector(month.market, target_permno, "adv20_usd")
        impact = _aligned_market_vector(
            month.market, target_permno, "impact_volatility_daily"
        )
    else:
        flat_missing_accounting_permitted = (current_weight == 0.0) & (weights == 0.0)
        execution_price = _aligned_projection_vector(
            month.daily_returns,
            target_permno,
            "execution_price",
            missing_permitted=flat_missing_accounting_permitted,
        )
        full_spread = _aligned_projection_vector(
            month.daily_returns,
            target_permno,
            "full_spread",
            missing_permitted=flat_missing_accounting_permitted,
        )
        adv = _aligned_projection_vector(
            month.daily_returns,
            target_permno,
            "adv_dollars",
            missing_permitted=flat_missing_accounting_permitted,
        )
        impact = _aligned_projection_vector(
            month.daily_returns,
            target_permno,
            "impact_volatility_daily",
            missing_permitted=flat_missing_accounting_permitted,
        )
    allocation_evidence = _make_allocation_evidence(
        problem_sha256=problem_sha,
        target_permno=target_permno,
        score=score,
        side=side,
        candidate_side=candidate_side,
        long_candidate=long_candidate,
        short_candidate=short_candidate,
        reduction_only=reduction_only,
        mandatory_action=mandatory_action,
        current_weight=current_weight,
        target_weight=weights,
        adv_dollars=adv,
        nav_dollars=month.opening_state.nav_dollars,
        residue_dollars=residue_dollars,
    )
    problem_receipt = KernelReceipt.create(
        "monthly_problem",
        {
            "strategy_id": month.strategy_id,
            "holding_month": month.holding_month,
            "problem_sha256": problem_sha,
            "architecture": architecture,
            "ordered_permno": list(allocation_evidence.permno),
            "candidate_audit": candidate_payload,
        },
    )
    solver_receipt = KernelReceipt.create("solver", solver_payload)
    constraint_receipt = KernelReceipt.create(
        "constraint_certificate", certificate_payload
    )

    commission, spread, impact_cost = _transaction_cost_components(
        nav_dollars=month.opening_state.nav_dollars,
        trade_weight=np.asarray(allocation_evidence.trade_weight),
        full_spread=full_spread,
        adv_dollars=adv,
        impact_volatility_daily=impact,
    )
    daily_evidence: list[FinalDailyAccountingEvidence] = []
    terminal_event_sources: tuple[TerminalEventSource, ...] = ()
    terminal_event_strategy_observations: tuple[
        TerminalEventStrategyObservation, ...
    ] = ()
    if isinstance(month.daily_returns, DailyReturnPanel):
        rebalance = rebalance_portfolio(
            month.opening_state,
            target_permno=target_permno,
            target_weights=weights,
            execution_price=execution_price,
            full_spread=full_spread,
            adv_dollars=adv,
            impact_volatility_daily=impact,
        )
        if not math.isclose(
            rebalance.transaction_cost_dollars,
            commission + spread + impact_cost,
            abs_tol=max(1e-7, month.opening_state.nav_dollars * 1e-13),
        ):
            raise FinalEconomicKernelError("transaction-cost decomposition differs")
        current = rebalance.state
        returns = month.daily_returns.aligned_returns(target_permno)
        for index, (date_value, day_returns, risk_free) in enumerate(
            zip(
                month.daily_returns.trading_dates,
                returns,
                month.daily_returns.daily_risk_free_rates,
                strict=True,
            )
        ):
            start_nav = (
                rebalance.pre_trade_nav_dollars if index == 0 else current.nav_dollars
            )
            opening_sha = (
                opening_snapshot.snapshot_sha256
                if index == 0
                else PortfolioSnapshot.from_state(current).snapshot_sha256
            )
            gross_pnl = float(
                np.dot(current.position_value_dollars, day_returns)
                + current.cash_dollars * float(risk_free)
            )
            accounted = apply_daily_accounting(
                current,
                asset_total_returns=day_returns,
                daily_risk_free_rate=float(risk_free),
                borrow_rate_annual=borrow_rate,
                financing_rate_annual=(
                    0.0
                    if month.strategy_id == EQUAL_WEIGHT_STRATEGY_ID
                    else PRIMARY_FINANCING_ANNUAL_RATE
                ),
            )
            current = accounted.state
            day_commission = commission if index == 0 else 0.0
            day_spread = spread if index == 0 else 0.0
            day_impact = impact_cost if index == 0 else 0.0
            total_cost = (
                day_commission
                + day_spread
                + day_impact
                + accounted.borrow_cost_dollars
                + accounted.financing_cost_dollars
            )
            net_pnl = gross_pnl - total_cost
            source_sha = _semantic_sha256(
                {
                    "panel_sha256": month.daily_returns.panel_sha256,
                    "trading_date": date_value.isoformat(),
                    "ordered_permno": list(allocation_evidence.permno),
                }
            )
            daily_evidence.append(
                _make_daily_evidence(
                    trading_date=date_value,
                    start_nav_dollars=start_nav,
                    end_nav_dollars=current.nav_dollars,
                    gross_pnl_dollars=gross_pnl,
                    commission_dollars=day_commission,
                    spread_dollars=day_spread,
                    impact_dollars=day_impact,
                    borrow_cost_dollars=accounted.borrow_cost_dollars,
                    financing_cost_dollars=accounted.financing_cost_dollars,
                    total_cost_dollars=total_cost,
                    net_pnl_dollars=net_pnl,
                    net_total_return=net_pnl / start_nav,
                    daily_risk_free_return=float(risk_free),
                    source_payload_sha256=source_sha,
                    opening_state_sha256=opening_sha,
                    closing_state_sha256=PortfolioSnapshot.from_state(
                        current
                    ).snapshot_sha256,
                    opening_mark_continuity_sha256=(
                        month.daily_returns.mark_continuity_ledger_sha256
                    ),
                    closing_mark_continuity_sha256=(
                        month.daily_returns.mark_continuity_ledger_sha256
                    ),
                    no_mark_permnos=(),
                    corporate_action_ids=(),
                    access_receipt_sha256=access_sha,
                )
            )
        closing_state = current
        transaction_cost = rebalance.transaction_cost_dollars
        corporate_action_sha = month.daily_returns.corporate_action_ledger_sha256
        mark_sha = month.daily_returns.mark_continuity_ledger_sha256
        closing_action_ledger = None
        closing_mark_continuity_ledger = None
        risk_free_total = float(
            np.prod(1.0 + month.daily_returns.daily_risk_free_rates) - 1.0
        )
    else:
        projection = month.daily_returns
        assert month.session_authority is not None
        dated = run_dated_holding_period(
            month.opening_state,
            session_authority=month.session_authority,
            target_permno=target_permno,
            target_weights=weights,
            execution_price=execution_price,
            full_spread=full_spread,
            adv_dollars=adv,
            impact_volatility_daily=impact,
            daily_inputs=projection.daily_inputs,
            corporate_actions=projection.corporate_actions,
            initial_action_ledger=month.initial_action_ledger,
            initial_mark_continuity_ledger=month.initial_mark_continuity_ledger,
            financing_rate_annual=(
                0.0
                if month.strategy_id == EQUAL_WEIGHT_STRATEGY_ID
                else projection.financing_rate_annual
            ),
        )
        if not math.isclose(
            dated.transaction_cost_dollars,
            commission + spread + impact_cost,
            abs_tol=max(1e-7, month.opening_state.nav_dollars * 1e-13),
        ):
            raise FinalEconomicKernelError("dated transaction costs differ")
        (
            terminal_event_sources,
            terminal_event_strategy_observations,
        ) = _capture_terminal_event_sensitivity(
            month=month,
            projection=projection,
            dated=dated,
            allocation=allocation_evidence,
        )
        prior_end = dated.pre_trade_nav_dollars
        for index, (day, receipt, end_nav) in enumerate(
            zip(
                projection.daily_inputs,
                dated.session_receipts,
                dated.daily_nav_dollars,
                strict=True,
            )
        ):
            start_nav = dated.pre_trade_nav_dollars if index == 0 else prior_end
            day_commission = commission if index == 0 else 0.0
            day_spread = spread if index == 0 else 0.0
            day_impact = impact_cost if index == 0 else 0.0
            total_cost = (
                day_commission
                + day_spread
                + day_impact
                + receipt.borrow_cost_dollars
                + receipt.financing_cost_dollars
            )
            net_pnl = float(end_nav) - start_nav
            gross_pnl = net_pnl + total_cost
            daily_evidence.append(
                _make_daily_evidence(
                    trading_date=day.trading_date,
                    start_nav_dollars=start_nav,
                    end_nav_dollars=float(end_nav),
                    gross_pnl_dollars=gross_pnl,
                    commission_dollars=day_commission,
                    spread_dollars=day_spread,
                    impact_dollars=day_impact,
                    borrow_cost_dollars=receipt.borrow_cost_dollars,
                    financing_cost_dollars=receipt.financing_cost_dollars,
                    total_cost_dollars=total_cost,
                    net_pnl_dollars=net_pnl,
                    net_total_return=net_pnl / start_nav,
                    daily_risk_free_return=day.daily_risk_free_rate,
                    source_payload_sha256=receipt.projected_payload_sha256,
                    opening_state_sha256=receipt.opening_state_sha256,
                    closing_state_sha256=receipt.closing_state_sha256,
                    opening_mark_continuity_sha256=(
                        receipt.opening_mark_continuity_sha256
                    ),
                    closing_mark_continuity_sha256=(
                        receipt.closing_mark_continuity_sha256
                    ),
                    no_mark_permnos=receipt.no_mark_permnos,
                    corporate_action_ids=receipt.corporate_action_ids,
                    access_receipt_sha256=access_sha,
                )
            )
            prior_end = float(end_nav)
        closing_state = dated.state
        transaction_cost = dated.transaction_cost_dollars
        risk_free_total = dated.risk_free_total_return
        corporate_action_sha = _corporate_action_ledger_sha256(dated.action_ledger)
        mark_sha = dated.mark_continuity_ledger.ledger_sha256
        closing_action_ledger = dated.action_ledger
        closing_mark_continuity_ledger = dated.mark_continuity_ledger

    materiality = audit_final_strategy_month_materiality(
        strategy_id=month.strategy_id,
        holding_month=month.holding_month,
        residue_dollars=np.asarray(allocation_evidence.residue_dollars),
    )
    closing = PortfolioSnapshot.from_state(closing_state)
    daily_tuple = tuple(daily_evidence)
    start_nav = month.opening_state.nav_dollars
    end_nav = closing_state.nav_dollars
    gross_pnl = sum(value.gross_pnl_dollars for value in daily_tuple)
    net_pnl = end_nav - start_nav
    borrow_cost = sum(value.borrow_cost_dollars for value in daily_tuple)
    financing_cost = sum(value.financing_cost_dollars for value in daily_tuple)
    gross_exposure = float(np.sum(np.abs(weights)))
    net_exposure = float(np.sum(weights))
    values = {
        "strategy_id": month.strategy_id,
        "feature_scope": month.feature_scope,
        "holding_month": month.holding_month,
        "signal_date": month.signal_date,
        "entry_date": month.entry_date,
        "exit_date": month.exit_date,
        "pnl_series_id": _pnl_series_id(
            segment=descriptor.segment,
            feature_scope=month.feature_scope,
            strategy_id=month.strategy_id,
        ),
        "opening_portfolio_sha256": opening_snapshot.snapshot_sha256,
        "closing_portfolio": closing,
        "problem_sha256": problem_sha,
        "weights_sha256": _semantic_sha256(_array_payload(np.asarray(weights))),
        "allocation_architecture": architecture,
        "solver_mode": solver_mode,
        "solver_status": solver_status,
        "net_total_return": net_pnl / start_nav,
        "net_excess_return": net_pnl / start_nav - risk_free_total,
        "risk_free_total_return": risk_free_total,
        "start_nav_dollars": start_nav,
        "end_nav_dollars": end_nav,
        "gross_pnl_dollars": gross_pnl,
        "net_pnl_dollars": net_pnl,
        "commission_dollars": commission,
        "spread_dollars": spread,
        "impact_dollars": impact_cost,
        "transaction_cost_dollars": transaction_cost,
        "borrow_cost_dollars": borrow_cost,
        "financing_cost_dollars": financing_cost,
        "one_way_turnover": 0.5
        * float(np.sum(np.abs(np.asarray(allocation_evidence.trade_weight)))),
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "market_beta": float(np.dot(weights, market_beta_vector)),
        "allocation": allocation_evidence,
        "daily_accounting": daily_tuple,
        "terminal_event_sources": terminal_event_sources,
        "terminal_event_strategy_observations": (terminal_event_strategy_observations),
        "problem_receipt": problem_receipt,
        "solver_receipt": solver_receipt,
        "constraint_certificate_receipt": constraint_receipt,
        "interval_sha256": interval_sha,
        "accounting_sha256": _semantic_sha256(
            [value.payload() for value in daily_tuple]
        ),
        "corporate_action_ledger_sha256": corporate_action_sha,
        "mark_continuity_ledger_sha256": mark_sha,
        "closing_action_ledger": closing_action_ledger,
        "closing_mark_continuity_ledger": closing_mark_continuity_ledger,
        "access_receipt_sha256": access_sha,
        "materiality": materiality,
    }
    provisional = StrategyMonthResult.__new__(StrategyMonthResult)
    for name, value in (*values.items(), ("result_sha256", "")):
        object.__setattr__(provisional, name, value)
    return StrategyMonthResult(
        **values,
        result_sha256=_semantic_sha256(provisional.payload(False)),
    )


def prepare_certified_strategy_month_decision(
    month: StrategyMonthDecisionInput,
) -> StrategyMonthDecisionCommitment:
    """Commit target weights using no holding-period return or factor row."""

    if not isinstance(month, StrategyMonthDecisionInput):
        raise FinalEconomicKernelError("month must be StrategyMonthDecisionInput")
    result = execute_certified_strategy_month(
        StrategyMonthInput(
            strategy_id=month.strategy_id,
            feature_scope=month.feature_scope,
            holding_month=month.holding_month,
            signal_date=month.signal_date,
            entry_date=month.entry_date,
            exit_date=month.exit_date,
            scored=month.scored,
            market=month.market,
            inherited=month.inherited,
            risk=month.risk,
            opening_state=month.opening_state,
            daily_returns=_decision_validation_panel(month),
            expected_return_by_permno=month.expected_return_by_permno,
            access_receipt_sha256=month.decision_access_receipt_sha256,
        )
    )
    values = {
        "decision_input": month,
        "opening_portfolio_sha256": result.opening_portfolio_sha256,
        "problem_sha256": result.problem_sha256,
        "weights_sha256": result.weights_sha256,
        "allocation_architecture": result.allocation_architecture,
        "solver_mode": result.solver_mode,
        "solver_status": result.solver_status,
        "allocation": result.allocation,
        "problem_receipt": result.problem_receipt,
        "solver_receipt": result.solver_receipt,
        "constraint_certificate_receipt": (result.constraint_certificate_receipt),
    }
    provisional = StrategyMonthDecisionCommitment.__new__(
        StrategyMonthDecisionCommitment
    )
    for name, value in (*values.items(), ("decision_sha256", "")):
        object.__setattr__(provisional, name, value)
    return StrategyMonthDecisionCommitment(
        **values,
        decision_sha256=_semantic_sha256(provisional.payload(False)),
    )


def account_certified_strategy_month_decision(
    decision: StrategyMonthDecisionCommitment,
    *,
    daily_accounting: DailyReturnPanel | FinalDailyAccountingProjection,
    accounting_access_receipt_sha256: str,
    session_authority: HoldingPeriodSessionAuthority | None = None,
    initial_action_ledger: CorporateActionLedger = CorporateActionLedger(),
    initial_mark_continuity_ledger: MarkContinuityLedger | None = None,
) -> StrategyMonthResult:
    """Open/account returns only after an immutable decision already exists."""

    if not isinstance(decision, StrategyMonthDecisionCommitment):
        raise FinalEconomicKernelError(
            "decision must be StrategyMonthDecisionCommitment"
        )
    accounting_access = _require_hash(
        accounting_access_receipt_sha256,
        field="accounting_access_receipt_sha256",
    )
    source = decision.decision_input
    result = execute_certified_strategy_month(
        StrategyMonthInput(
            strategy_id=source.strategy_id,
            feature_scope=source.feature_scope,
            holding_month=source.holding_month,
            signal_date=source.signal_date,
            entry_date=source.entry_date,
            exit_date=source.exit_date,
            scored=source.scored,
            market=source.market,
            inherited=source.inherited,
            risk=source.risk,
            opening_state=source.opening_state,
            daily_returns=daily_accounting,
            expected_return_by_permno=source.expected_return_by_permno,
            access_receipt_sha256=accounting_access,
            session_authority=session_authority,
            initial_action_ledger=initial_action_ledger,
            initial_mark_continuity_ledger=initial_mark_continuity_ledger,
        )
    )
    observed = (
        result.opening_portfolio_sha256,
        result.problem_sha256,
        result.weights_sha256,
        result.allocation_architecture,
        result.solver_mode,
        result.solver_status,
        result.allocation.evidence_sha256,
        result.problem_receipt.receipt_sha256,
        result.solver_receipt.receipt_sha256,
        result.constraint_certificate_receipt.receipt_sha256,
    )
    committed = (
        decision.opening_portfolio_sha256,
        decision.problem_sha256,
        decision.weights_sha256,
        decision.allocation_architecture,
        decision.solver_mode,
        decision.solver_status,
        decision.allocation.evidence_sha256,
        decision.problem_receipt.receipt_sha256,
        decision.solver_receipt.receipt_sha256,
        decision.constraint_certificate_receipt.receipt_sha256,
    )
    if observed != committed:
        raise FinalEconomicKernelError(
            "post-open accounting attempted to alter the committed decision"
        )
    return result


@dataclass(frozen=True, slots=True)
class KernelStage13DocumentRows:
    """Kernel-owned rows ready to merge with the eight external Stage 13 docs."""

    allocation_records: tuple[Mapping[str, object], ...]
    candidate_audits: tuple[Mapping[str, object], ...]
    daily_records: tuple[Mapping[str, object], ...]
    monthly_records: tuple[Mapping[str, object], ...]
    audit_records: tuple[Mapping[str, object], ...]
    terminal_event_sources: tuple[Mapping[str, object], ...]
    terminal_event_strategy_observations: tuple[Mapping[str, object], ...]
    synthetic: bool
    rows_sha256: str

    def __post_init__(self) -> None:
        if type(self.synthetic) is not bool:
            raise FinalEconomicKernelError("kernel document synthetic flag differs")
        required_collections = (
            self.allocation_records,
            self.candidate_audits,
            self.daily_records,
            self.monthly_records,
            self.audit_records,
        )
        if any(
            not values or any(not isinstance(row, Mapping) for row in values)
            for values in required_collections
        ):
            raise FinalEconomicKernelError("kernel result document rows differ")
        if (
            type(self.terminal_event_sources) is not tuple
            or type(self.terminal_event_strategy_observations) is not tuple
            or any(not isinstance(row, Mapping) for row in self.terminal_event_sources)
            or any(
                not isinstance(row, Mapping)
                for row in self.terminal_event_strategy_observations
            )
        ):
            raise FinalEconomicKernelError("kernel terminal-event document rows differ")
        _require_hash(self.rows_sha256, field="rows_sha256")
        if self.rows_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("kernel document-row digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "allocation_records": [dict(row) for row in self.allocation_records],
            "candidate_audits": [dict(row) for row in self.candidate_audits],
            "daily_records": [dict(row) for row in self.daily_records],
            "monthly_records": [dict(row) for row in self.monthly_records],
            "audit_records": [dict(row) for row in self.audit_records],
            "terminal_event_sources": [
                dict(row) for row in self.terminal_event_sources
            ],
            "terminal_event_strategy_observations": [
                dict(row) for row in self.terminal_event_strategy_observations
            ],
            "synthetic": self.synthetic,
        }
        if include_digest:
            value["rows_sha256"] = self.rows_sha256
        return value

    def to_documents(self) -> Mapping[str, Mapping[str, object]]:
        common: dict[str, object] = {
            "schema_version": 1,
            "stage": 13,
            "synthetic": self.synthetic,
        }
        return MappingProxyType(
            {
                "allocation_weights_trades.json": {
                    **common,
                    "weight_unit": "fraction_of_start_nav",
                    "problem_contract": (
                        "frozen_stage12_problem_constraints_costs_and_inherited_state"
                    ),
                    "candidate_policy": {
                        "minimum_new_names_each_side": 50,
                        "tail_extension_rule": ("scan_extreme_scores_until_minimum"),
                        "insufficient_side_rule": (
                            "take_no_new_risk_and_reduce_existing_risk"
                        ),
                        "candidate_sets_must_be_disjoint": True,
                        "equal_weight_class": (
                            "descriptive_costed_long_only_not_long_short"
                        ),
                    },
                    "candidate_audits": [dict(row) for row in self.candidate_audits],
                    "records": [dict(row) for row in self.allocation_records],
                },
                "daily_accounting.json": {
                    **common,
                    "records": [dict(row) for row in self.daily_records],
                },
                "monthly_accounting.json": {
                    **common,
                    "segments_are_separate": True,
                    "records": [dict(row) for row in self.monthly_records],
                },
                "costs_constraints_materiality.json": {
                    **common,
                    "per_position_residue_rule": ("strictly_less_than_0.05_usd"),
                    "aggregate_residue_rule": (
                        "less_than_or_equal_to_1.00_usd_per_strategy_month"
                    ),
                    "records": [dict(row) for row in self.audit_records],
                    "terminal_event_sources": [
                        dict(row) for row in self.terminal_event_sources
                    ],
                    "terminal_event_strategy_observations": [
                        dict(row) for row in self.terminal_event_strategy_observations
                    ],
                },
            }
        )


def build_stage13_kernel_document_rows(
    results: Sequence[StrategyMonthResult],
    *,
    checkpoint_sha256_by_segment_month: Mapping[tuple[str, str], str],
    synthetic: bool = False,
) -> KernelStage13DocumentRows:
    """Project retained kernel observations into the five owned result members."""

    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise FinalEconomicKernelError("strategy results must be a sequence")
    ordered = tuple(results)
    if not ordered or any(
        not isinstance(value, StrategyMonthResult) for value in ordered
    ):
        raise FinalEconomicKernelError("strategy results differ")
    if type(synthetic) is not bool:
        raise FinalEconomicKernelError("synthetic flag differs")
    expected_keys = {
        (
            final_holding_month_descriptor(value.holding_month).segment.value,
            value.holding_month,
        )
        for value in ordered
    }
    if not isinstance(
        checkpoint_sha256_by_segment_month, Mapping
    ) or not expected_keys.issubset(checkpoint_sha256_by_segment_month):
        raise FinalEconomicKernelError("checkpoint mapping omits kernel months")
    if len(
        {
            (value.pnl_series_id, value.holding_month, value.strategy_id)
            for value in ordered
        }
    ) != len(ordered):
        raise FinalEconomicKernelError("strategy result keys repeat")

    allocations: list[Mapping[str, object]] = []
    candidates: list[Mapping[str, object]] = []
    daily: list[Mapping[str, object]] = []
    monthly: list[Mapping[str, object]] = []
    audits: list[Mapping[str, object]] = []
    terminal_sources_by_action_id: dict[str, Mapping[str, object]] = {}
    terminal_observations: list[Mapping[str, object]] = []
    for result in ordered:
        segment = final_holding_month_descriptor(result.holding_month).segment.value
        checkpoint = _require_hash(
            checkpoint_sha256_by_segment_month[(segment, result.holding_month)],
            field="checkpoint_sha256",
        )
        for source in result.terminal_event_sources:
            source_row = MappingProxyType(
                {
                    **source.as_dict(),
                    "segment": segment,
                    "checkpoint_sha256": checkpoint,
                }
            )
            prior_source = terminal_sources_by_action_id.get(source.terminal_action_id)
            if prior_source is not None and dict(prior_source) != dict(source_row):
                raise FinalEconomicKernelError(
                    "terminal-event source differs across strategies"
                )
            terminal_sources_by_action_id.setdefault(
                source.terminal_action_id,
                source_row,
            )
        for observation in result.terminal_event_strategy_observations:
            terminal_observations.append(
                MappingProxyType(
                    {
                        **observation.as_dict(),
                        "holding_month": result.holding_month,
                        "segment": segment,
                        "checkpoint_sha256": checkpoint,
                    }
                )
            )
        publication_problem_sha = _semantic_sha256(
            {
                "segment": segment,
                "holding_month": result.holding_month,
                "strategy_id": result.strategy_id,
                "checkpoint_sha256": checkpoint,
                "policy_id": "frozen-stage12-allocation-policy-v1",
            }
        )
        for index, permno in enumerate(result.allocation.permno):
            current = result.allocation.current_weight[index]
            target = result.allocation.target_weight[index]
            trade = result.allocation.trade_weight[index]
            certificate = _semantic_sha256(
                {
                    "allocation_problem_sha256": publication_problem_sha,
                    "permno": permno,
                    "current_weight": current,
                    "target_weight": target,
                    "trade_weight": trade,
                }
            )
            allocations.append(
                MappingProxyType(
                    {
                        "holding_month": result.holding_month,
                        "segment": segment,
                        "strategy_id": result.strategy_id,
                        "permno": permno,
                        "score": result.allocation.score[index],
                        "current_weight": current,
                        "target_weight": target,
                        "trade_weight": trade,
                        "adv_trade_fraction": (
                            result.allocation.adv_trade_fraction[index]
                        ),
                        "mandatory_action": (result.allocation.mandatory_action[index]),
                        "candidate_side": result.allocation.candidate_side[index],
                        "residue_usd": result.allocation.residue_dollars[index],
                        "allocation_problem_sha256": publication_problem_sha,
                        "solver_status": (
                            "DETERMINISTIC_EQUAL_WEIGHT"
                            if result.strategy_id == EQUAL_WEIGHT_STRATEGY_ID
                            else "OPTIMAL_OR_CERTIFIED_FALLBACK"
                        ),
                        "constraint_certificate_sha256": certificate,
                        "checkpoint_sha256": checkpoint,
                    }
                )
            )
        problem_payload = json.loads(result.problem_receipt.canonical_payload_json)
        raw_candidate = problem_payload["candidate_audit"]
        candidate_fields = (
            "eligible_universe_count",
            "candidate_long_count",
            "candidate_short_count",
            "reduction_only_count",
            "minimum_new_names_each_side",
            "minimum_each_side_applies",
            "tail_extension_used",
            "insufficient_side_fallback_used",
            "new_risk_permitted",
            "candidate_sets_disjoint",
        )
        candidate_payload = {
            "holding_month": result.holding_month,
            "segment": segment,
            "strategy_id": result.strategy_id,
            **{name: raw_candidate[name] for name in candidate_fields},
            "checkpoint_sha256": checkpoint,
        }
        candidates.append(
            MappingProxyType(
                {
                    **candidate_payload,
                    "candidate_evidence_sha256": _semantic_sha256(candidate_payload),
                }
            )
        )
        for row in result.daily_accounting:
            daily.append(
                MappingProxyType(
                    {
                        "holding_month": result.holding_month,
                        "segment": segment,
                        "strategy_id": result.strategy_id,
                        "trading_date": row.trading_date.isoformat(),
                        "start_nav_usd": row.start_nav_dollars,
                        "gross_pnl_usd": row.gross_pnl_dollars,
                        "total_cost_usd": row.total_cost_dollars,
                        "net_pnl_usd": row.net_pnl_dollars,
                        "end_nav_usd": row.end_nav_dollars,
                        "checkpoint_sha256": checkpoint,
                    }
                )
            )
        weights = np.asarray(result.allocation.target_weight, dtype=np.float64)
        concentration = float(np.dot(weights, weights))
        nonzero = int(np.count_nonzero(weights))
        effective = (
            result.gross_exposure * result.gross_exposure / concentration
            if concentration > 0.0
            else 0.0
        )
        monthly.append(
            MappingProxyType(
                {
                    "holding_month": result.holding_month,
                    "segment": segment,
                    "strategy_id": result.strategy_id,
                    "start_nav_usd": result.start_nav_dollars,
                    "end_nav_usd": result.end_nav_dollars,
                    "gross_pnl_usd": result.gross_pnl_dollars,
                    "net_pnl_usd": result.net_pnl_dollars,
                    "total_cost_usd": (
                        result.transaction_cost_dollars
                        + result.borrow_cost_dollars
                        + result.financing_cost_dollars
                    ),
                    "gross_return": result.gross_pnl_dollars / result.start_nav_dollars,
                    "net_return": result.net_total_return,
                    "risk_free_return": result.risk_free_total_return,
                    "excess_return": result.net_excess_return,
                    "turnover": result.one_way_turnover,
                    "gross_exposure": result.gross_exposure,
                    "net_exposure": result.net_exposure,
                    "market_beta": result.market_beta,
                    "cash_weight": 1.0 - result.net_exposure,
                    "position_count": nonzero,
                    "effective_position_count": effective,
                    "maximum_abs_weight": float(np.max(np.abs(weights))),
                    "concentration": concentration,
                    "checkpoint_sha256": checkpoint,
                }
            )
        )
        certificate_payload = json.loads(
            result.constraint_certificate_receipt.canonical_payload_json
        )
        audits.append(
            MappingProxyType(
                {
                    "holding_month": result.holding_month,
                    "segment": segment,
                    "strategy_id": result.strategy_id,
                    "commission_usd": result.commission_dollars,
                    "spread_usd": result.spread_dollars,
                    "impact_usd": result.impact_dollars,
                    "borrow_usd": result.borrow_cost_dollars,
                    "financing_usd": result.financing_cost_dollars,
                    "total_cost_usd": (
                        result.transaction_cost_dollars
                        + result.borrow_cost_dollars
                        + result.financing_cost_dollars
                    ),
                    "one_way_turnover": result.one_way_turnover,
                    "gross_exposure": result.gross_exposure,
                    "net_exposure": result.net_exposure,
                    "market_beta": result.market_beta,
                    "maximum_adv_trade_fraction": max(
                        result.allocation.adv_trade_fraction
                    ),
                    "maximum_single_position_residue_usd": max(
                        abs(value) for value in result.allocation.residue_dollars
                    ),
                    "aggregate_absolute_residue_usd": sum(
                        abs(value) for value in result.allocation.residue_dollars
                    ),
                    "hard_constraint_violation_count": certificate_payload[
                        "hard_constraint_violation_count"
                    ],
                    "checkpoint_sha256": checkpoint,
                }
            )
        )
    values = {
        "allocation_records": tuple(allocations),
        "candidate_audits": tuple(candidates),
        "daily_records": tuple(daily),
        "monthly_records": tuple(monthly),
        "audit_records": tuple(audits),
        "terminal_event_sources": tuple(
            sorted(
                terminal_sources_by_action_id.values(),
                key=lambda row: (
                    str(row["holding_month"]),
                    str(row["event_date"]),
                    int(row["permno"]),
                    str(row["terminal_action_id"]),
                ),
            )
        ),
        "terminal_event_strategy_observations": tuple(
            sorted(
                terminal_observations,
                key=lambda row: (
                    str(row["holding_month"]),
                    str(row["terminal_action_id"]),
                    str(row["strategy_id"]),
                ),
            )
        ),
        "synthetic": synthetic,
    }
    provisional = KernelStage13DocumentRows.__new__(KernelStage13DocumentRows)
    for name, value in (*values.items(), ("rows_sha256", "")):
        object.__setattr__(provisional, name, value)
    return KernelStage13DocumentRows(
        **values,
        rows_sha256=_semantic_sha256(provisional.payload(False)),
    )


@dataclass(frozen=True, slots=True)
class PreFinalTrainingIdentity:
    """Fixed Stage 9 plus Stage 11 history allowed in the first final refit."""

    stage9_development_oof_receipt_sha256: str
    stage11_validation_forecast_receipt_sha256: str
    stage11_completed_outcome_receipt_sha256: str
    feature_history_receipt_sha256: str
    development_oof_row_count: int
    validation_forecast_row_count: int
    completed_outcome_row_count: int
    completed_label_holding_months: tuple[str, ...]
    maximum_outcome_available_date: dt.date
    identity_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "stage9_development_oof_receipt_sha256",
            "stage11_validation_forecast_receipt_sha256",
            "stage11_completed_outcome_receipt_sha256",
            "feature_history_receipt_sha256",
            "identity_sha256",
        ):
            _require_hash(getattr(self, field), field=field)
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.development_oof_row_count,
                self.validation_forecast_row_count,
                self.completed_outcome_row_count,
            )
        ):
            raise FinalEconomicKernelError("pre-final history row counts differ")
        if (
            len(PREFINAL_DEVELOPMENT_OOF_HOLDING_MONTHS) != 95
            or len(PREFINAL_VALIDATION_HOLDING_MONTHS) != 59
            or self.completed_label_holding_months != PREFINAL_TRAINING_HOLDING_MONTHS
            or "2014-12" in self.completed_label_holding_months
            or "2019-12" in self.completed_label_holding_months
            or self.maximum_outcome_available_date != dt.date(2019, 12, 2)
        ):
            raise FinalEconomicKernelError("pre-final history chronology differs")
        if self.identity_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("pre-final history identity differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "contract_id": "stage13-pre-final-training-history-v1",
            "stage9_development_oof_receipt_sha256": (
                self.stage9_development_oof_receipt_sha256
            ),
            "stage11_validation_forecast_receipt_sha256": (
                self.stage11_validation_forecast_receipt_sha256
            ),
            "stage11_completed_outcome_receipt_sha256": (
                self.stage11_completed_outcome_receipt_sha256
            ),
            "feature_history_receipt_sha256": self.feature_history_receipt_sha256,
            "development_oof_row_count": self.development_oof_row_count,
            "validation_forecast_row_count": self.validation_forecast_row_count,
            "completed_outcome_row_count": self.completed_outcome_row_count,
            "development_oof_holding_months": list(
                PREFINAL_DEVELOPMENT_OOF_HOLDING_MONTHS
            ),
            "validation_holding_months": list(PREFINAL_VALIDATION_HOLDING_MONTHS),
            "completed_label_holding_months": list(self.completed_label_holding_months),
            "maximum_outcome_available_date": (
                self.maximum_outcome_available_date.isoformat()
            ),
            "first_final_refit_signal_date": (
                PREFINAL_FINAL_REFIT_SIGNAL_DATE.isoformat()
            ),
            "selected_model_config_ids": [
                NEURAL_NETWORK_CONFIG_ID,
                XGBOOST_CONFIG_ID,
                ELASTIC_NET_CONFIG_ID,
            ],
            "stochastic_seeds": list(FINAL_STOCHASTIC_SEEDS),
            "completed_labels_only": True,
            "development_validation_or_final_splicing": False,
        }
        if include_digest:
            value["identity_sha256"] = self.identity_sha256
        return value


def create_pre_final_training_identity(
    *,
    stage9_development_oof_receipt_sha256: str,
    stage11_validation_forecast_receipt_sha256: str,
    stage11_completed_outcome_receipt_sha256: str,
    feature_history_receipt_sha256: str,
    development_oof_row_count: int,
    validation_forecast_row_count: int,
    completed_outcome_row_count: int,
) -> PreFinalTrainingIdentity:
    """Bind authenticated empirical history; this function performs no I/O."""

    values = {
        "stage9_development_oof_receipt_sha256": (
            stage9_development_oof_receipt_sha256
        ),
        "stage11_validation_forecast_receipt_sha256": (
            stage11_validation_forecast_receipt_sha256
        ),
        "stage11_completed_outcome_receipt_sha256": (
            stage11_completed_outcome_receipt_sha256
        ),
        "feature_history_receipt_sha256": feature_history_receipt_sha256,
        "development_oof_row_count": development_oof_row_count,
        "validation_forecast_row_count": validation_forecast_row_count,
        "completed_outcome_row_count": completed_outcome_row_count,
        "completed_label_holding_months": PREFINAL_TRAINING_HOLDING_MONTHS,
        "maximum_outcome_available_date": dt.date(2019, 12, 2),
    }
    provisional = PreFinalTrainingIdentity.__new__(PreFinalTrainingIdentity)
    for name, value in (*values.items(), ("identity_sha256", "")):
        object.__setattr__(provisional, name, value)
    return PreFinalTrainingIdentity(
        **values,
        identity_sha256=_semantic_sha256(provisional.payload(False)),
    )


def _strict_hash_mapping(
    value: Mapping[str, str],
    *,
    expected_keys: Sequence[str],
    field: str,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        raise FinalEconomicKernelError(f"{field} keys differ")
    result = {
        key: _require_hash(value[key], field=f"{field}.{key}") for key in expected_keys
    }
    return MappingProxyType(result)


MODEL_STATE_KEYS = tuple(
    f"{scope.value}:{definition.family}"
    for scope in FeatureScope
    for definition in FROZEN_MODEL_DEFINITIONS
)
CALIBRATION_STATE_KEYS = tuple(
    f"{scope.value}:{strategy_id}"
    for scope in FeatureScope
    for strategy_id in CALIBRATED_STRATEGY_IDS
)


@dataclass(frozen=True, slots=True)
class FinalKernelState:
    """Semantic monthly checkpoint state used by the memoized runner."""

    holding_month: str
    segment: FinalPeriodSegment
    contract_sha256: str
    previous_state_sha256: str
    segment_genesis_sha256: str
    opening_book_sha256_by_strategy: Mapping[str, str]
    closing_book_sha256_by_strategy: Mapping[str, str]
    pnl_series_id_by_strategy: Mapping[str, str]
    pnl_chain_head_sha256_by_strategy: Mapping[str, str]
    model_state_sha256: Mapping[str, str]
    calibration_state_sha256: Mapping[str, str]
    shadow_core_opening_book_sha256_by_strategy: Mapping[str, str]
    shadow_core_closing_book_sha256_by_strategy: Mapping[str, str]
    shadow_core_head_sha256: str | None
    state_sha256: str

    def __post_init__(self) -> None:
        descriptor = final_holding_month_descriptor(self.holding_month)
        if self.segment is not descriptor.segment:
            raise FinalEconomicKernelError("kernel state segment differs")
        if (
            self.contract_sha256
            != FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT.contract_sha256
        ):
            raise FinalEconomicKernelError("kernel state contract differs")
        _require_hash(self.previous_state_sha256, field="previous_state_sha256")
        _require_hash(self.segment_genesis_sha256, field="segment_genesis_sha256")
        opening = _strict_hash_mapping(
            self.opening_book_sha256_by_strategy,
            expected_keys=ACTIVE_STRATEGY_IDS,
            field="opening_book",
        )
        closing = _strict_hash_mapping(
            self.closing_book_sha256_by_strategy,
            expected_keys=ACTIVE_STRATEGY_IDS,
            field="closing_book",
        )
        pnl_head = _strict_hash_mapping(
            self.pnl_chain_head_sha256_by_strategy,
            expected_keys=ACTIVE_STRATEGY_IDS,
            field="pnl_chain_head",
        )
        expected_series = {
            strategy_id: _pnl_series_id(
                segment=self.segment,
                feature_scope=(
                    FeatureScope.FULL_FEATURE_PRIMARY
                    if self.segment is FinalPeriodSegment.PRIMARY_FULL
                    else FeatureScope.REDUCED_AUXILIARY_CORE
                ),
                strategy_id=strategy_id,
            )
            for strategy_id in ACTIVE_STRATEGY_IDS
        }
        if dict(self.pnl_series_id_by_strategy) != expected_series:
            raise FinalEconomicKernelError("P&L series identity differs")
        model = _strict_hash_mapping(
            self.model_state_sha256,
            expected_keys=MODEL_STATE_KEYS,
            field="model_state",
        )
        calibration = _strict_hash_mapping(
            self.calibration_state_sha256,
            expected_keys=CALIBRATION_STATE_KEYS,
            field="calibration_state",
        )
        if self.segment is FinalPeriodSegment.PRIMARY_FULL:
            shadow_open = _strict_hash_mapping(
                self.shadow_core_opening_book_sha256_by_strategy,
                expected_keys=ACTIVE_STRATEGY_IDS,
                field="shadow_core_opening_book",
            )
            shadow_close = _strict_hash_mapping(
                self.shadow_core_closing_book_sha256_by_strategy,
                expected_keys=ACTIVE_STRATEGY_IDS,
                field="shadow_core_closing_book",
            )
            _require_hash(self.shadow_core_head_sha256, field="shadow_core_head_sha256")
        else:
            if (
                self.shadow_core_opening_book_sha256_by_strategy
                or self.shadow_core_closing_book_sha256_by_strategy
                or self.shadow_core_head_sha256 is not None
            ):
                raise FinalEconomicKernelError(
                    "reported core/partial state cannot contain a shadow book"
                )
            shadow_open = MappingProxyType({})
            shadow_close = MappingProxyType({})
        for name, value in (
            ("opening_book_sha256_by_strategy", opening),
            ("closing_book_sha256_by_strategy", closing),
            ("pnl_series_id_by_strategy", MappingProxyType(expected_series)),
            ("pnl_chain_head_sha256_by_strategy", pnl_head),
            ("model_state_sha256", model),
            ("calibration_state_sha256", calibration),
            ("shadow_core_opening_book_sha256_by_strategy", shadow_open),
            ("shadow_core_closing_book_sha256_by_strategy", shadow_close),
        ):
            object.__setattr__(self, name, value)
        _require_hash(self.state_sha256, field="state_sha256")
        if self.state_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("kernel state digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": FINAL_ECONOMIC_KERNEL_SCHEMA_VERSION,
            "holding_month": self.holding_month,
            "segment": self.segment.value,
            "contract_sha256": self.contract_sha256,
            "previous_state_sha256": self.previous_state_sha256,
            "segment_genesis_sha256": self.segment_genesis_sha256,
            "opening_book_sha256_by_strategy": dict(
                self.opening_book_sha256_by_strategy
            ),
            "closing_book_sha256_by_strategy": dict(
                self.closing_book_sha256_by_strategy
            ),
            "pnl_series_id_by_strategy": dict(self.pnl_series_id_by_strategy),
            "pnl_chain_head_sha256_by_strategy": dict(
                self.pnl_chain_head_sha256_by_strategy
            ),
            "model_state_sha256": dict(self.model_state_sha256),
            "calibration_state_sha256": dict(self.calibration_state_sha256),
            "shadow_core_opening_book_sha256_by_strategy": dict(
                self.shadow_core_opening_book_sha256_by_strategy
            ),
            "shadow_core_closing_book_sha256_by_strategy": dict(
                self.shadow_core_closing_book_sha256_by_strategy
            ),
            "shadow_core_head_sha256": self.shadow_core_head_sha256,
        }
        if include_digest:
            value["state_sha256"] = self.state_sha256
        return value

    @classmethod
    def create(
        cls,
        *,
        holding_month: str,
        previous_state_sha256: str,
        segment_genesis_sha256: str,
        opening_book_sha256_by_strategy: Mapping[str, str],
        closing_book_sha256_by_strategy: Mapping[str, str],
        prior_pnl_chain_head_sha256_by_strategy: Mapping[str, str] | None,
        result_sha256_by_strategy: Mapping[str, str],
        model_state_sha256: Mapping[str, str],
        calibration_state_sha256: Mapping[str, str],
        shadow_core_opening_book_sha256_by_strategy: Mapping[str, str] | None = None,
        shadow_core_closing_book_sha256_by_strategy: Mapping[str, str] | None = None,
        prior_shadow_core_head_sha256: str | None = None,
        shadow_core_result_sha256_by_strategy: Mapping[str, str] | None = None,
    ) -> "FinalKernelState":
        descriptor = final_holding_month_descriptor(holding_month)
        if set(result_sha256_by_strategy) != set(ACTIVE_STRATEGY_IDS):
            raise FinalEconomicKernelError("monthly result digest set differs")
        for digest in result_sha256_by_strategy.values():
            _require_hash(digest, field="result_sha256")
        series = {
            strategy_id: _pnl_series_id(
                segment=descriptor.segment,
                feature_scope=(
                    FeatureScope.FULL_FEATURE_PRIMARY
                    if descriptor.segment is FinalPeriodSegment.PRIMARY_FULL
                    else FeatureScope.REDUCED_AUXILIARY_CORE
                ),
                strategy_id=strategy_id,
            )
            for strategy_id in ACTIVE_STRATEGY_IDS
        }
        if prior_pnl_chain_head_sha256_by_strategy is None:
            prior_heads = {
                strategy_id: _semantic_sha256(
                    {
                        "new_series": series[strategy_id],
                        "segment_genesis_sha256": segment_genesis_sha256,
                    }
                )
                for strategy_id in ACTIVE_STRATEGY_IDS
            }
        else:
            prior_heads = dict(
                _strict_hash_mapping(
                    prior_pnl_chain_head_sha256_by_strategy,
                    expected_keys=ACTIVE_STRATEGY_IDS,
                    field="prior_pnl_chain_head",
                )
            )
        pnl_heads = {
            strategy_id: _semantic_sha256(
                {
                    "series_id": series[strategy_id],
                    "prior_head_sha256": prior_heads[strategy_id],
                    "month_result_sha256": result_sha256_by_strategy[strategy_id],
                }
            )
            for strategy_id in ACTIVE_STRATEGY_IDS
        }
        shadow_open = dict(shadow_core_opening_book_sha256_by_strategy or {})
        shadow_close = dict(shadow_core_closing_book_sha256_by_strategy or {})
        shadow_head: str | None = None
        if descriptor.segment is FinalPeriodSegment.PRIMARY_FULL:
            if (
                set(shadow_open) != set(ACTIVE_STRATEGY_IDS)
                or set(shadow_close) != set(ACTIVE_STRATEGY_IDS)
                or shadow_core_result_sha256_by_strategy is None
                or set(shadow_core_result_sha256_by_strategy)
                != set(ACTIVE_STRATEGY_IDS)
            ):
                raise FinalEconomicKernelError(
                    "primary shadow-core state is incomplete"
                )
            shadow_prior = (
                prior_shadow_core_head_sha256
                if prior_shadow_core_head_sha256 is not None
                else segment_genesis_sha256
            )
            _require_hash(shadow_prior, field="prior_shadow_core_head_sha256")
            shadow_head = _semantic_sha256(
                {
                    "scope": FeatureScope.REDUCED_AUXILIARY_CORE.value,
                    "holding_month": holding_month,
                    "prior_head_sha256": shadow_prior,
                    "opening_books": shadow_open,
                    "closing_books": shadow_close,
                    "result_sha256_by_strategy": dict(
                        shadow_core_result_sha256_by_strategy
                    ),
                    "core_model_state_sha256": {
                        key: model_state_sha256[key]
                        for key in MODEL_STATE_KEYS
                        if key.startswith(
                            FeatureScope.REDUCED_AUXILIARY_CORE.value + ":"
                        )
                    },
                    "core_calibration_state_sha256": {
                        key: calibration_state_sha256[key]
                        for key in CALIBRATION_STATE_KEYS
                        if key.startswith(
                            FeatureScope.REDUCED_AUXILIARY_CORE.value + ":"
                        )
                    },
                }
            )
        values = {
            "holding_month": holding_month,
            "segment": descriptor.segment,
            "contract_sha256": FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT.contract_sha256,
            "previous_state_sha256": previous_state_sha256,
            "segment_genesis_sha256": segment_genesis_sha256,
            "opening_book_sha256_by_strategy": dict(opening_book_sha256_by_strategy),
            "closing_book_sha256_by_strategy": dict(closing_book_sha256_by_strategy),
            "pnl_series_id_by_strategy": series,
            "pnl_chain_head_sha256_by_strategy": pnl_heads,
            "model_state_sha256": dict(model_state_sha256),
            "calibration_state_sha256": dict(calibration_state_sha256),
            "shadow_core_opening_book_sha256_by_strategy": shadow_open,
            "shadow_core_closing_book_sha256_by_strategy": shadow_close,
            "shadow_core_head_sha256": shadow_head,
        }
        provisional = cls.__new__(cls)
        for name, value in (*values.items(), ("state_sha256", "")):
            object.__setattr__(provisional, name, value)
        digest = _semantic_sha256(provisional.payload(False))
        return cls(**values, state_sha256=digest)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FinalKernelState":
        expected = {
            "schema_version",
            "holding_month",
            "segment",
            "contract_sha256",
            "previous_state_sha256",
            "segment_genesis_sha256",
            "opening_book_sha256_by_strategy",
            "closing_book_sha256_by_strategy",
            "pnl_series_id_by_strategy",
            "pnl_chain_head_sha256_by_strategy",
            "model_state_sha256",
            "calibration_state_sha256",
            "shadow_core_opening_book_sha256_by_strategy",
            "shadow_core_closing_book_sha256_by_strategy",
            "shadow_core_head_sha256",
            "state_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise FinalEconomicKernelError("kernel state mapping fields differ")
        if value["schema_version"] != FINAL_ECONOMIC_KERNEL_SCHEMA_VERSION:
            raise FinalEconomicKernelError("kernel state schema differs")
        try:
            segment = FinalPeriodSegment(str(value["segment"]))
        except ValueError as exc:
            raise FinalEconomicKernelError("kernel state segment is invalid") from exc
        mapping_fields = (
            "opening_book_sha256_by_strategy",
            "closing_book_sha256_by_strategy",
            "pnl_series_id_by_strategy",
            "pnl_chain_head_sha256_by_strategy",
            "model_state_sha256",
            "calibration_state_sha256",
            "shadow_core_opening_book_sha256_by_strategy",
            "shadow_core_closing_book_sha256_by_strategy",
        )
        if any(not isinstance(value[name], Mapping) for name in mapping_fields):
            raise FinalEconomicKernelError("kernel state nested mapping differs")
        return cls(
            holding_month=value["holding_month"],  # type: ignore[arg-type]
            segment=segment,
            contract_sha256=value["contract_sha256"],  # type: ignore[arg-type]
            previous_state_sha256=value["previous_state_sha256"],  # type: ignore[arg-type]
            segment_genesis_sha256=value["segment_genesis_sha256"],  # type: ignore[arg-type]
            opening_book_sha256_by_strategy=value[
                "opening_book_sha256_by_strategy"
            ],  # type: ignore[arg-type]
            closing_book_sha256_by_strategy=value[
                "closing_book_sha256_by_strategy"
            ],  # type: ignore[arg-type]
            pnl_series_id_by_strategy=value[
                "pnl_series_id_by_strategy"
            ],  # type: ignore[arg-type]
            pnl_chain_head_sha256_by_strategy=value[
                "pnl_chain_head_sha256_by_strategy"
            ],  # type: ignore[arg-type]
            model_state_sha256=value["model_state_sha256"],  # type: ignore[arg-type]
            calibration_state_sha256=value[
                "calibration_state_sha256"
            ],  # type: ignore[arg-type]
            shadow_core_opening_book_sha256_by_strategy=value[
                "shadow_core_opening_book_sha256_by_strategy"
            ],  # type: ignore[arg-type]
            shadow_core_closing_book_sha256_by_strategy=value[
                "shadow_core_closing_book_sha256_by_strategy"
            ],  # type: ignore[arg-type]
            shadow_core_head_sha256=value["shadow_core_head_sha256"],  # type: ignore[arg-type]
            state_sha256=value["state_sha256"],  # type: ignore[arg-type]
        )


def validate_kernel_state_successor(
    previous: FinalKernelState,
    current: FinalKernelState,
) -> None:
    """Authenticate one adjacent month, including both reporting boundaries."""

    if not isinstance(previous, FinalKernelState) or not isinstance(
        current, FinalKernelState
    ):
        raise FinalEconomicKernelError("continuity requires kernel states")
    if current.previous_state_sha256 != previous.state_sha256:
        raise FinalEconomicKernelError("kernel predecessor digest differs")
    if shift_month(previous.holding_month, 1) != current.holding_month:
        raise FinalEconomicKernelError("kernel states are not adjacent months")
    if previous.segment is current.segment:
        if current.segment_genesis_sha256 != previous.segment_genesis_sha256 or dict(
            current.opening_book_sha256_by_strategy
        ) != dict(previous.closing_book_sha256_by_strategy):
            raise FinalEconomicKernelError("within-segment state continuity differs")
        return
    if (
        previous.holding_month == "2025-08"
        and current.holding_month == "2025-09"
        and previous.segment is FinalPeriodSegment.PRIMARY_FULL
        and current.segment is FinalPeriodSegment.CORE_COMPLETE
    ):
        if (
            current.segment_genesis_sha256 != previous.shadow_core_head_sha256
            or dict(current.opening_book_sha256_by_strategy)
            != dict(previous.shadow_core_closing_book_sha256_by_strategy)
            or set(current.pnl_series_id_by_strategy.values())
            & set(previous.pnl_series_id_by_strategy.values())
        ):
            raise FinalEconomicKernelError("core extension genesis differs")
        return
    if (
        previous.holding_month == "2025-11"
        and current.holding_month == "2025-12"
        and previous.segment is FinalPeriodSegment.CORE_COMPLETE
        and current.segment is FinalPeriodSegment.PARTIAL_DECEMBER_DESCRIPTIVE
    ):
        if (
            current.segment_genesis_sha256 != previous.state_sha256
            or dict(current.opening_book_sha256_by_strategy)
            != dict(previous.closing_book_sha256_by_strategy)
            or set(current.pnl_series_id_by_strategy.values())
            & set(previous.pnl_series_id_by_strategy.values())
        ):
            raise FinalEconomicKernelError("partial-December genesis differs")
        return
    raise FinalEconomicKernelError("prohibited final-segment transition")


@dataclass(frozen=True, slots=True)
class NativeModelProbeReceipt:
    """Synthetic proof that one selected native fit/predict path executed."""

    model_family: str
    config_id: str
    seed: int | None
    runtime_fit_status: str
    model_payload_sha256: str
    prediction_sha256: str
    prediction_rows: int
    portfolio_or_final_data_read: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        expected_config, expected_seeds = _expected_config(self.model_family)
        expected_seed = None if self.model_family == "elastic_net" else 11
        if (
            self.config_id != expected_config
            or self.seed != expected_seed
            or self.seed not in expected_seeds
            or self.runtime_fit_status != "NONAUTHORITATIVE_SYNTHETIC_TEST_ONLY"
            or type(self.prediction_rows) is not int
            or self.prediction_rows != 11
            or self.portfolio_or_final_data_read is not False
        ):
            raise FinalEconomicKernelError("selected native model probe differs")
        _require_hash(self.model_payload_sha256, field="model_payload_sha256")
        _require_hash(self.prediction_sha256, field="prediction_sha256")
        _require_hash(self.receipt_sha256, field="receipt_sha256")
        if self.receipt_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("native model probe digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "model_family": self.model_family,
            "config_id": self.config_id,
            "seed": self.seed,
            "runtime_fit_status": self.runtime_fit_status,
            "model_payload_sha256": self.model_payload_sha256,
            "prediction_sha256": self.prediction_sha256,
            "prediction_rows": self.prediction_rows,
            "portfolio_or_final_data_read": self.portfolio_or_final_data_read,
            "runtime_binding": (
                "portfolio_bot.validation.selector_native_runtime:"
                "_fit_locked_selector_model_nonauthoritative_for_test+"
                "predict_locked_selector_model"
            ),
        }
        if include_digest:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class NativeSelectedModelsCertification:
    receipts: tuple[NativeModelProbeReceipt, ...]
    empirical_final_rows_opened: int
    certification_sha256: str

    def __post_init__(self) -> None:
        expected = (
            ("neural_network", NEURAL_NETWORK_CONFIG_ID, 11),
            ("xgboost", XGBOOST_CONFIG_ID, 11),
            ("elastic_net", ELASTIC_NET_CONFIG_ID, None),
        )
        observed = tuple(
            (item.model_family, item.config_id, item.seed) for item in self.receipts
        )
        if observed != expected or self.empirical_final_rows_opened != 0:
            raise FinalEconomicKernelError("selected native certification differs")
        _require_hash(self.certification_sha256, field="certification_sha256")
        if self.certification_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError(
                "selected native certification digest differs"
            )

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "receipts": [item.payload() for item in self.receipts],
            "selected_stochastic_seed_domain": list(FINAL_STOCHASTIC_SEEDS),
            "representative_executed_stochastic_seed": 11,
            "all_selected_configs_bound": True,
            "empirical_final_rows_opened": self.empirical_final_rows_opened,
        }
        if include_digest:
            value["certification_sha256"] = self.certification_sha256
        return value


def _native_probe_preprocessor(
    *,
    scope: str,
    matrix_kind: str,
):
    from portfolio_bot.validation.selector_runtime import PersistedPreprocessorState

    return PersistedPreprocessorState(
        scope=scope,
        matrix_kind=matrix_kind,
        refit_signal_date=dt.date(2019, 12, 31),
        fitted_signal_date_max=dt.date(2019, 11, 29),
        predictor_availability_date_max=dt.date(2019, 11, 29),
        frozen_column_names=("synthetic_a", "synthetic_b", "synthetic_c", "inactive"),
        active_column_indices=(0, 1, 2),
        feature_registry_sha256=_synthetic_hash("native:feature-registry"),
        category_vocabulary_sha256=_synthetic_hash("native:categories"),
        aggregate_history_sha256=_synthetic_hash("native:aggregates"),
        interaction_registry_sha256=(
            _synthetic_hash("native:interactions") if matrix_kind == "elastic" else None
        ),
    )


def _native_probe_fixture(
    family: str,
    config_id: str,
    seed: int | None,
):
    from portfolio_bot.validation.selector_runtime import (
        array_sha256,
        build_refit_identity,
        prepare_validation_refit_arrays,
    )

    months: list[str] = []
    exits: list[dt.date] = []
    for year, month in (
        (year, month)
        for year in (2018, 2019)
        for month in range(1, 13)
        if (year, month) <= (2019, 11)
    ):
        holding_month = f"{year:04d}-{month:02d}"
        exit_year, exit_month = (year + 1, 1) if month == 12 else (year, month + 1)
        months.extend((holding_month,) * 4)
        exits.extend((dt.date(exit_year, exit_month, 5),) * 4)
    rng = np.random.default_rng(20260712)
    raw = rng.normal(size=(len(months), 4)).astype(np.float32)
    target = np.clip(
        0.15 * raw[:, 0]
        - 0.08 * raw[:, 1]
        + 0.04 * raw[:, 2]
        + rng.normal(0.0, 0.02, len(months)),
        -0.49,
        0.49,
    ).astype(np.float32)
    base_full = _native_probe_preprocessor(scope="full", matrix_kind="base")
    base_inner = _native_probe_preprocessor(scope="inner", matrix_kind="base")
    arrays = prepare_validation_refit_arrays(
        refit_signal_date=dt.date(2019, 12, 31),
        full_base_matrix=raw,
        inner_base_matrix=raw,
        targets=target,
        holding_months=tuple(months),
        exit_dates=tuple(exits),
        outcome_availability_dates=tuple(exits),
        row_keys_sha256=array_sha256(np.arange(len(months), dtype=np.int64)),
        full_preprocessor=base_full,
        inner_preprocessor=base_inner,
    )
    full = (
        _native_probe_preprocessor(scope="full", matrix_kind="elastic")
        if family == "elastic_net"
        else base_full
    )
    inner = None if family == "elastic_net" else base_inner
    identity = build_refit_identity(
        family=family,
        config_id=config_id,
        seed=seed,
        refit_signal_date=dt.date(2019, 12, 31),
        arrays=arrays,
        full_preprocessor=full,
        inner_preprocessor=inner,
        validation_calendar_sha256=_synthetic_hash("native:calendar"),
        model_code_sha256=_synthetic_hash(f"native:model-code:{family}"),
        runtime_versions_sha256=_synthetic_hash("native:runtime-versions"),
        stage9_shortlist_sha256=_synthetic_hash("native:shortlist"),
    )
    return identity, arrays, full, inner, raw


def _selected_native_probe_worker(
    family: str,
    config_id: str,
    seed: int | None,
    result_queue: object,
) -> None:
    """Spawn-only worker that executes the real low-level model functions."""

    try:
        from portfolio_bot.validation.selector_native_runtime import (
            _fit_locked_selector_model_nonauthoritative_for_test,
            predict_locked_selector_model,
        )
        from portfolio_bot.validation.selector_runtime import LoadedSelectorCheckpoint

        identity, arrays, full, inner, raw = _native_probe_fixture(
            family, config_id, seed
        )
        artifact = _fit_locked_selector_model_nonauthoritative_for_test(
            identity=identity,
            arrays=arrays,
        )
        checkpoint = LoadedSelectorCheckpoint(
            root=Path("."),
            manifest={"checkpoint_id": identity.checkpoint_id},
            identity=identity.payload(),
            model_payload=artifact.model_payload,
            inner_preprocessor=inner,
            full_preprocessor=full,
        )
        predictions = predict_locked_selector_model(
            checkpoint=checkpoint,
            full_frozen_matrix=raw[:11],
        )
        message = {
            "ok": True,
            "model_family": family,
            "config_id": config_id,
            "seed": seed,
            "runtime_fit_status": artifact.fit_receipt["status"],
            "model_payload_sha256": _hash_bytes(artifact.model_payload),
            "prediction_sha256": _semantic_sha256(_array_payload(predictions)),
            "prediction_rows": int(len(predictions)),
            "portfolio_or_final_data_read": bool(
                artifact.fit_receipt["portfolio_or_final_data_read"]
            ),
        }
        result_queue.put(message)  # type: ignore[attr-defined]
    except BaseException:
        result_queue.put(  # type: ignore[attr-defined]
            {"ok": False, "traceback": traceback.format_exc()}
        )


@lru_cache(maxsize=1)
def certify_selected_native_fit_predict() -> NativeSelectedModelsCertification:
    """Run one fixed synthetic fit/predict probe for every selected family.

    Stochastic seed 11 is executed as the representative numerical probe; the
    closed five-seed domain is separately enforced by every forecast block and
    recorded in this receipt.  The function has no parameters, data paths, or
    callbacks and each native family runs in its own ``spawn`` process.
    """

    jobs = (
        ("neural_network", NEURAL_NETWORK_CONFIG_ID, 11),
        ("xgboost", XGBOOST_CONFIG_ID, 11),
        ("elastic_net", ELASTIC_NET_CONFIG_ID, None),
    )
    context = mp.get_context("spawn")
    queues: list[object] = []
    processes: list[mp.Process] = []
    messages: list[Mapping[str, object] | None] = [None] * len(jobs)
    started_at: list[float] = []
    try:
        for family, config_id, seed in jobs:
            queue = context.Queue()
            process = context.Process(
                target=_selected_native_probe_worker,
                args=(family, config_id, seed, queue),
                name=f"stage13-synthetic-{family}",
            )
            queues.append(queue)
            processes.append(process)
            process.start()
            started_at.append(time.monotonic())
        while any(message is None for message in messages):
            for index, (process, queue, job) in enumerate(
                zip(processes, queues, jobs, strict=True)
            ):
                if messages[index] is not None:
                    continue
                if time.monotonic() - started_at[index] > 120.0:
                    raise FinalEconomicKernelError(
                        f"selected native probe timed out for {job[0]}"
                    )
                try:
                    # Read before join so a large probe receipt cannot fill
                    # the Queue feeder pipe and deadlock child shutdown.
                    message = queue.get(timeout=0.05)  # type: ignore[attr-defined]
                except Empty:
                    if process.is_alive():
                        continue
                    process.join()
                    try:
                        message = queue.get(timeout=10.0)  # type: ignore[attr-defined]
                    except Empty as exc:
                        raise FinalEconomicKernelError(
                            f"selected native probe emitted no receipt for {job[0]}"
                        ) from exc
                process.join(timeout=10.0)
                if process.is_alive():
                    raise FinalEconomicKernelError(
                        f"selected native probe did not exit after receipt drain for {job[0]}"
                    )
                if (
                    process.exitcode != 0
                    or not isinstance(message, Mapping)
                    or message.get("ok") is not True
                ):
                    detail = (
                        message.get("traceback", "no traceback")
                        if isinstance(message, Mapping)
                        else "invalid receipt"
                    )
                    raise FinalEconomicKernelError(
                        f"selected native probe failed for {job[0]}: {detail}"
                    )
                messages[index] = message
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10.0)
            if process.is_alive():
                process.kill()
                process.join()
        for queue in queues:
            queue.close()  # type: ignore[attr-defined]

    receipts: list[NativeModelProbeReceipt] = []
    for message in messages:
        assert message is not None
        values = {
            name: message[name]
            for name in (
                "model_family",
                "config_id",
                "seed",
                "runtime_fit_status",
                "model_payload_sha256",
                "prediction_sha256",
                "prediction_rows",
                "portfolio_or_final_data_read",
            )
        }
        provisional = NativeModelProbeReceipt.__new__(NativeModelProbeReceipt)
        for name, value in (*values.items(), ("receipt_sha256", "")):
            object.__setattr__(provisional, name, value)
        digest = _semantic_sha256(provisional.payload(False))
        receipts.append(
            NativeModelProbeReceipt(**values, receipt_sha256=digest)  # type: ignore[arg-type]
        )
    values = {
        "receipts": tuple(receipts),
        "empirical_final_rows_opened": 0,
    }
    provisional = NativeSelectedModelsCertification.__new__(
        NativeSelectedModelsCertification
    )
    for name, value in (*values.items(), ("certification_sha256", "")):
        object.__setattr__(provisional, name, value)
    digest = _semantic_sha256(provisional.payload(False))
    return NativeSelectedModelsCertification(**values, certification_sha256=digest)


@dataclass(frozen=True, slots=True)
class SyntheticKernelCertification:
    contract_sha256: str
    native_model_certification_sha256: str
    forecast_receipt_sha256: str
    calibration_sha256: str
    hybrid_result_sha256: str
    elastic_net_result_sha256: str
    rank_result_sha256: str
    trailing_residual_result_sha256: str
    vqm_result_sha256: str
    equal_weight_result_sha256: str
    detailed_evidence_sha256: str
    kernel_document_rows_sha256: str
    allocation_position_count: int
    daily_session_count: int
    exact_accounting_reconciliation: bool
    exercised_components: tuple[str, ...]
    empirical_final_rows_opened: int
    certification_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "contract_sha256",
            "native_model_certification_sha256",
            "forecast_receipt_sha256",
            "calibration_sha256",
            "hybrid_result_sha256",
            "elastic_net_result_sha256",
            "rank_result_sha256",
            "trailing_residual_result_sha256",
            "vqm_result_sha256",
            "equal_weight_result_sha256",
            "detailed_evidence_sha256",
            "kernel_document_rows_sha256",
        ):
            _require_hash(getattr(self, name), field=name)
        if (
            self.contract_sha256
            != FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT.contract_sha256
        ):
            raise FinalEconomicKernelError("synthetic certification contract differs")
        if self.exercised_components != (
            "forecast_seed_mean",
            "selected_nn_native_fit_predict",
            "selected_xgboost_native_fit_predict",
            "selected_elastic_net_native_fit_predict",
            "centered_component_ranks",
            "direct_unreranked_ensemble",
            "completed_label_huber_calibration",
            "monthly_problem_builder",
            "pre_accounting_decision_commitment",
            "hybrid_solver",
            "rank_projection_solver",
            "trailing_residual_direct_expected_return",
            "fixed_vqm_signal_and_huber_calibration",
            "fixed_holding_accounting",
            "equal_weight_costed_accounting",
            "ordered_position_trade_evidence",
            "observed_daily_nav_cost_reconciliation",
            "stage13_kernel_document_handoff",
            "numerical_materiality_v2",
        ):
            raise FinalEconomicKernelError("synthetic component coverage differs")
        if (
            self.allocation_position_count != 120
            or self.daily_session_count != 2
            or self.exact_accounting_reconciliation is not True
            or self.empirical_final_rows_opened != 0
        ):
            raise FinalEconomicKernelError("synthetic certification opened final rows")
        _require_hash(self.certification_sha256, field="certification_sha256")
        if self.certification_sha256 != _semantic_sha256(self.payload(False)):
            raise FinalEconomicKernelError("synthetic certification digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "contract_sha256": self.contract_sha256,
            "native_model_certification_sha256": (
                self.native_model_certification_sha256
            ),
            "forecast_receipt_sha256": self.forecast_receipt_sha256,
            "calibration_sha256": self.calibration_sha256,
            "hybrid_result_sha256": self.hybrid_result_sha256,
            "elastic_net_result_sha256": self.elastic_net_result_sha256,
            "rank_result_sha256": self.rank_result_sha256,
            "trailing_residual_result_sha256": (self.trailing_residual_result_sha256),
            "vqm_result_sha256": self.vqm_result_sha256,
            "equal_weight_result_sha256": self.equal_weight_result_sha256,
            "detailed_evidence_sha256": self.detailed_evidence_sha256,
            "kernel_document_rows_sha256": self.kernel_document_rows_sha256,
            "allocation_position_count": self.allocation_position_count,
            "daily_session_count": self.daily_session_count,
            "exact_accounting_reconciliation": (self.exact_accounting_reconciliation),
            "exercised_components": list(self.exercised_components),
            "empirical_final_rows_opened": self.empirical_final_rows_opened,
        }
        if include_digest:
            value["certification_sha256"] = self.certification_sha256
        return value


def _synthetic_hash(label: str) -> str:
    return _semantic_sha256({"synthetic": True, "label": label})


def _synthetic_forecasts(
    *,
    permno: np.ndarray,
) -> FinalForecastReceipt:
    n = len(permno)
    base = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    pattern = np.sin(np.arange(n, dtype=np.float64) * 0.17)
    nn = np.vstack([base + (index - 2) * 0.002 * pattern for index in range(5)])
    xgb = np.vstack([0.85 * base + (index - 2) * 0.001 * pattern for index in range(5)])
    enet = (0.70 * base).reshape(1, -1)
    blocks = {
        "neural_network": SeedForecastBlock.create(
            model_family="neural_network",
            config_id=NEURAL_NETWORK_CONFIG_ID,
            seeds=FINAL_STOCHASTIC_SEEDS,
            raw_forecasts=nn,
            fitted_state_sha256_by_seed=tuple(
                _synthetic_hash(f"nn:{seed}") for seed in FINAL_STOCHASTIC_SEEDS
            ),
        ),
        "xgboost": SeedForecastBlock.create(
            model_family="xgboost",
            config_id=XGBOOST_CONFIG_ID,
            seeds=FINAL_STOCHASTIC_SEEDS,
            raw_forecasts=xgb,
            fitted_state_sha256_by_seed=tuple(
                _synthetic_hash(f"xgb:{seed}") for seed in FINAL_STOCHASTIC_SEEDS
            ),
        ),
        "elastic_net": SeedForecastBlock.create(
            model_family="elastic_net",
            config_id=ELASTIC_NET_CONFIG_ID,
            seeds=(None,),
            raw_forecasts=enet,
            fitted_state_sha256_by_seed=(_synthetic_hash("enet:deterministic"),),
        ),
    }
    return aggregate_selected_forecasts(
        feature_scope=FeatureScope.FULL_FEATURE_PRIMARY,
        holding_month="2020-01",
        permno=permno,
        neural_network=blocks["neural_network"],
        xgboost=blocks["xgboost"],
        elastic_net=blocks["elastic_net"],
    )


def _synthetic_calibration_observations(
    *,
    strategy_id: str = PRIMARY_STRATEGY_ID,
) -> tuple[FinalCalibrationObservation, ...]:
    rows: list[FinalCalibrationObservation] = []
    for month_index, holding_month in enumerate(
        tuple(f"2018-{month:02d}" for month in range(1, 13))
        + tuple(f"2019-{month:02d}" for month in range(1, 12))
    ):
        year, month = (int(part) for part in holding_month.split("-"))
        signal_year = year if month > 1 else year - 1
        signal_month = month - 1 if month > 1 else 12
        signal_date = dt.date(
            signal_year,
            signal_month,
            _calendar.monthrange(signal_year, signal_month)[1],
        )
        entry_date = dt.date(year, month, 2)
        exit_month = month + 1 if month < 12 else 1
        exit_year = year if month < 12 else year + 1
        exit_date = dt.date(exit_year, exit_month, 2)
        for offset, score in enumerate((-0.4, -0.2, 0.2, 0.4), start=1):
            rows.append(
                FinalCalibrationObservation(
                    feature_scope=FeatureScope.FULL_FEATURE_PRIMARY,
                    strategy_id=strategy_id,
                    forecast_id=f"synthetic:{strategy_id}:{holding_month}:{offset}",
                    holding_month=holding_month,
                    permno=700000 + offset,
                    score=score,
                    residual_return=(0.04 * score) + month_index * 1e-5,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    outcome_available_date=exit_date,
                    forecast_payload_sha256=_synthetic_hash(
                        f"forecast:{strategy_id}:{holding_month}:{offset}"
                    ),
                    source="pre_final_pseudo_oof",
                )
            )
    return tuple(rows)


def _synthetic_month_inputs(
    forecasts: FinalForecastReceipt,
    calibration: FinalCalibrationReceipt,
) -> tuple[
    AuthenticatedMonthlyIdentity,
    MarketSnapshot,
    InheritedHoldings,
    MonthlyRiskInputs,
    DailyReturnPanel,
]:
    permno = forecasts.permno
    n = len(permno)
    identity = AuthenticatedMonthlyIdentity(
        signal_date=dt.date(2019, 12, 31),
        source_bundle_sha256=_synthetic_hash("month-source-bundle"),
    )
    beta = np.zeros((n, 6), dtype=np.float64)
    beta[:, 0] = 1.0
    beta[:, 1] = np.tile(np.asarray([0.10, -0.10]), n // 2)
    covariance = np.diag(np.asarray([0.0020, 0.0004, 0.0003, 0.0002, 0.0002, 0.0005]))
    market = MarketSnapshot(
        identity=identity,
        permno=permno,
        permco=permno + 1_000_000,
        execution_price=np.full(n, 50.0),
        full_spread=np.full(n, 0.0010),
        adv20_usd=np.full(n, 1_000_000_000.0),
        impact_volatility_daily=np.full(n, 0.015),
        factor_betas=beta,
        specific_variance_monthly=np.full(n, 0.004),
        sector=np.asarray(
            ["TECH" if index % 2 == 0 else "HEALTH" for index in range(n)]
        ),
        country=np.full(n, "USA"),
        new_long_trade_allowed=np.ones(n, dtype=bool),
        new_short_trade_allowed=np.ones(n, dtype=bool),
    )
    inherited = InheritedHoldings.empty(identity)
    risk = MonthlyRiskInputs(
        identity=identity,
        factor_covariance_monthly=covariance,
        specific_variance_p95_monthly=0.006,
        required_exit_impact_volatility_p95_daily=0.030,
    )
    direction = np.sign(forecasts.primary_ensemble_score)
    daily = DailyReturnPanel.create(
        permno=permno,
        trading_dates=(dt.date(2020, 1, 3), dt.date(2020, 2, 3)),
        asset_total_returns=np.vstack((0.0005 * direction, 0.0003 * direction)),
        daily_risk_free_rates=np.asarray([0.00005, 0.00005]),
        monthly_source_bundle_sha256=identity.source_bundle_sha256,
        total_return_source_sha256=_synthetic_hash("daily:total-return"),
        risk_free_source_sha256=_synthetic_hash("daily:risk-free"),
        corporate_action_ledger_sha256=_synthetic_hash("daily:corporate-actions"),
        mark_continuity_ledger_sha256=_synthetic_hash("daily:mark-continuity"),
        corporate_actions_integrated_in_total_returns=True,
    )
    del calibration
    return identity, market, inherited, risk, daily


def certify_synthetic_economic_kernel() -> SyntheticKernelCertification:
    """Exercise the frozen calculation stack without opening any empirical row."""

    native_models = certify_selected_native_fit_predict()
    permno = np.arange(910001, 910121, dtype=np.int64)
    forecasts = _synthetic_forecasts(permno=permno)
    calibration = fit_final_calibration(
        _synthetic_calibration_observations(),
        feature_scope=FeatureScope.FULL_FEATURE_PRIMARY,
        strategy_id=PRIMARY_STRATEGY_ID,
        fit_holding_month="2020-01",
        fit_signal_date=dt.date(2019, 12, 31),
    )
    enet_calibration = fit_final_calibration(
        _synthetic_calibration_observations(strategy_id=ELASTIC_NET_STRATEGY_ID),
        feature_scope=FeatureScope.FULL_FEATURE_PRIMARY,
        strategy_id=ELASTIC_NET_STRATEGY_ID,
        fit_holding_month="2020-01",
        fit_signal_date=dt.date(2019, 12, 31),
    )
    vqm_calibration = fit_final_calibration(
        _synthetic_calibration_observations(strategy_id=VQM_STRATEGY_ID),
        feature_scope=FeatureScope.FULL_FEATURE_PRIMARY,
        strategy_id=VQM_STRATEGY_ID,
        fit_holding_month="2020-01",
        fit_signal_date=dt.date(2019, 12, 31),
    )
    identity, market, inherited, risk, daily = _synthetic_month_inputs(
        forecasts, calibration
    )
    primary_scored = ScoredSnapshot(
        identity=identity,
        permno=permno,
        score=forecasts.primary_ensemble_score,
        calibration_kappa_monthly=calibration.kappa,
        long_eligible=np.ones(len(permno), dtype=bool),
        short_eligible=np.ones(len(permno), dtype=bool),
    )
    enet_scored = ScoredSnapshot(
        identity=identity,
        permno=permno,
        score=forecasts.elastic_net_score,
        calibration_kappa_monthly=enet_calibration.kappa,
        long_eligible=np.ones(len(permno), dtype=bool),
        short_eligible=np.ones(len(permno), dtype=bool),
    )
    position = np.arange(len(permno), dtype=np.float64)
    vqm_inputs = {
        feature_name: centered_average_rank(
            np.sin(position * (0.017 + index * 0.001) + index * 0.31)
        )
        for index, feature_name in enumerate(sorted(VQM_FEATURES), start=1)
    }
    vqm_score = value_quality_momentum_scores(vqm_inputs).centered_score
    vqm_scored = ScoredSnapshot(
        identity=identity,
        permno=permno,
        score=vqm_score,
        calibration_kappa_monthly=vqm_calibration.kappa,
        long_eligible=np.ones(len(permno), dtype=bool),
        short_eligible=np.ones(len(permno), dtype=bool),
    )
    trailing_expected = 0.012 * np.tanh(
        np.linspace(-2.0, 2.0, len(permno), dtype=np.float64)
    )
    trailing_scored = ScoredSnapshot(
        identity=identity,
        permno=permno,
        score=centered_average_rank(trailing_expected),
        calibration_kappa_monthly=1.0,
        long_eligible=np.ones(len(permno), dtype=bool),
        short_eligible=np.ones(len(permno), dtype=bool),
    )
    opening = PortfolioState.all_cash(REFERENCE_NAV_DOLLARS)
    common = {
        "feature_scope": FeatureScope.FULL_FEATURE_PRIMARY,
        "holding_month": "2020-01",
        "signal_date": dt.date(2019, 12, 31),
        "entry_date": dt.date(2020, 1, 2),
        "exit_date": dt.date(2020, 2, 3),
        "market": market,
        "inherited": inherited,
        "risk": risk,
        "opening_state": opening,
        "daily_returns": daily,
    }
    primary_decision = prepare_certified_strategy_month_decision(
        StrategyMonthDecisionInput(
            strategy_id=PRIMARY_STRATEGY_ID,
            scored=primary_scored,
            decision_access_receipt_sha256=_synthetic_hash("decision-access:2020-01"),
            **{
                key: value
                for key, value in common.items()
                if key not in {"daily_returns"}
            },
        )
    )
    hybrid = account_certified_strategy_month_decision(
        primary_decision,
        daily_accounting=daily,
        accounting_access_receipt_sha256=_synthetic_hash("accounting-access:2020-01"),
    )
    elastic_net = execute_certified_strategy_month(
        StrategyMonthInput(
            strategy_id=ELASTIC_NET_STRATEGY_ID,
            scored=enet_scored,
            **common,
        )
    )
    rank = execute_certified_strategy_month(
        StrategyMonthInput(
            strategy_id=RANK_STRATEGY_ID,
            scored=primary_scored,
            **common,
        )
    )
    trailing = execute_certified_strategy_month(
        StrategyMonthInput(
            strategy_id=TRAILING_RESIDUAL_STRATEGY_ID,
            scored=trailing_scored,
            expected_return_by_permno={
                int(security): float(expected)
                for security, expected in zip(permno, trailing_expected, strict=True)
            },
            **common,
        )
    )
    vqm = execute_certified_strategy_month(
        StrategyMonthInput(
            strategy_id=VQM_STRATEGY_ID,
            scored=vqm_scored,
            **common,
        )
    )
    equal_weight = execute_certified_strategy_month(
        StrategyMonthInput(
            strategy_id=EQUAL_WEIGHT_STRATEGY_ID,
            scored=primary_scored,
            **common,
        )
    )
    strategy_results = (
        hybrid,
        elastic_net,
        rank,
        trailing,
        vqm,
        equal_weight,
    )
    synthetic_checkpoint = _synthetic_hash("stage13-checkpoint:2020-01")
    document_rows = build_stage13_kernel_document_rows(
        strategy_results,
        checkpoint_sha256_by_segment_month={
            (FinalPeriodSegment.PRIMARY_FULL.value, "2020-01"): (synthetic_checkpoint)
        },
        synthetic=True,
    )
    detailed_evidence_sha = _semantic_sha256(
        {
            value.strategy_id: {
                "allocation": value.allocation.evidence_sha256,
                "accounting": value.accounting_sha256,
                "problem": value.problem_receipt.receipt_sha256,
                "solver": value.solver_receipt.receipt_sha256,
                "certificate": value.constraint_certificate_receipt.receipt_sha256,
            }
            for value in strategy_results
        }
    )
    values = {
        "contract_sha256": FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT.contract_sha256,
        "native_model_certification_sha256": (native_models.certification_sha256),
        "forecast_receipt_sha256": forecasts.receipt_sha256,
        "calibration_sha256": _semantic_sha256(
            {
                PRIMARY_STRATEGY_ID: calibration.calibration_sha256,
                ELASTIC_NET_STRATEGY_ID: enet_calibration.calibration_sha256,
                VQM_STRATEGY_ID: vqm_calibration.calibration_sha256,
            }
        ),
        "hybrid_result_sha256": hybrid.result_sha256,
        "elastic_net_result_sha256": elastic_net.result_sha256,
        "rank_result_sha256": rank.result_sha256,
        "trailing_residual_result_sha256": trailing.result_sha256,
        "vqm_result_sha256": vqm.result_sha256,
        "equal_weight_result_sha256": equal_weight.result_sha256,
        "detailed_evidence_sha256": detailed_evidence_sha,
        "kernel_document_rows_sha256": document_rows.rows_sha256,
        "allocation_position_count": max(
            len(value.allocation.permno) for value in strategy_results
        ),
        "daily_session_count": len(hybrid.daily_accounting),
        "exact_accounting_reconciliation": all(
            math.isclose(
                value.end_nav_dollars,
                value.start_nav_dollars
                + value.gross_pnl_dollars
                - value.transaction_cost_dollars
                - value.borrow_cost_dollars
                - value.financing_cost_dollars,
                abs_tol=max(1e-7, value.start_nav_dollars * 1e-13),
            )
            for value in strategy_results
        ),
        "exercised_components": (
            "forecast_seed_mean",
            "selected_nn_native_fit_predict",
            "selected_xgboost_native_fit_predict",
            "selected_elastic_net_native_fit_predict",
            "centered_component_ranks",
            "direct_unreranked_ensemble",
            "completed_label_huber_calibration",
            "monthly_problem_builder",
            "pre_accounting_decision_commitment",
            "hybrid_solver",
            "rank_projection_solver",
            "trailing_residual_direct_expected_return",
            "fixed_vqm_signal_and_huber_calibration",
            "fixed_holding_accounting",
            "equal_weight_costed_accounting",
            "ordered_position_trade_evidence",
            "observed_daily_nav_cost_reconciliation",
            "stage13_kernel_document_handoff",
            "numerical_materiality_v2",
        ),
        "empirical_final_rows_opened": 0,
    }
    provisional = SyntheticKernelCertification.__new__(SyntheticKernelCertification)
    for name, value in (*values.items(), ("certification_sha256", "")):
        object.__setattr__(provisional, name, value)
    digest = _semantic_sha256(provisional.payload(False))
    return SyntheticKernelCertification(**values, certification_sha256=digest)


__all__ = [
    "ACTIVE_STRATEGY_IDS",
    "ALL_STRATEGY_IDS",
    "CALIBRATED_STRATEGY_IDS",
    "CALIBRATION_HALF_LIFE_MONTHS",
    "CALIBRATION_HUBER_DELTA",
    "CALIBRATION_STATE_KEYS",
    "DECISION_FOCUSED_INACTIVE_REASON",
    "DECISION_FOCUSED_STRATEGY_ID",
    "ELASTIC_NET_CONFIG_ID",
    "ELASTIC_NET_STRATEGY_ID",
    "EQUAL_WEIGHT_STRATEGY_ID",
    "FINAL_ECONOMIC_KERNEL_ID",
    "FINAL_STOCHASTIC_SEEDS",
    "FROZEN_ECONOMIC_SEGMENT_PLANS",
    "FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT",
    "FROZEN_MODEL_DEFINITIONS",
    "FROZEN_STRATEGY_DEFINITIONS",
    "FeatureScope",
    "FinalCalibrationObservation",
    "FinalCalibrationReceipt",
    "FinalEconomicKernelContract",
    "FinalEconomicKernelError",
    "FinalDailyAccountingEvidence",
    "FinalForecastReceipt",
    "FinalKernelState",
    "FinalModelFitCheckpoint",
    "FinalRefitEvidence",
    "MODEL_STATE_KEYS",
    "NEURAL_NETWORK_CONFIG_ID",
    "NativeModelProbeReceipt",
    "NativeSelectedModelsCertification",
    "PREFINAL_DEVELOPMENT_OOF_HOLDING_MONTHS",
    "PREFINAL_FINAL_REFIT_SIGNAL_DATE",
    "PREFINAL_TRAINING_HOLDING_MONTHS",
    "PREFINAL_VALIDATION_HOLDING_MONTHS",
    "PRIMARY_REFIT_SIGNAL_MONTHS",
    "PRIMARY_STRATEGY_ID",
    "PortfolioSnapshot",
    "PreFinalTrainingIdentity",
    "RANK_STRATEGY_ID",
    "REFERENCE_NAV_DOLLARS",
    "SHARED_CALIBRATION_POLICY_ID",
    "SHARED_CALIBRATION_SOURCE_SCOPE",
    "SeedForecastBlock",
    "KernelReceipt",
    "KernelStage13DocumentRows",
    "StrategyAllocationEvidence",
    "StrategyMonthDecisionCommitment",
    "StrategyMonthDecisionInput",
    "StrategyMonthInput",
    "StrategyMonthMaterialityEvidence",
    "StrategyMonthResult",
    "SyntheticKernelCertification",
    "TRAILING_RESIDUAL_STRATEGY_ID",
    "VQM_STRATEGY_ID",
    "XGBOOST_CONFIG_ID",
    "aggregate_selected_forecasts",
    "account_certified_strategy_month_decision",
    "audit_final_strategy_month_materiality",
    "certify_synthetic_economic_kernel",
    "certify_selected_native_fit_predict",
    "build_stage13_kernel_document_rows",
    "create_pre_final_training_identity",
    "execute_certified_strategy_month",
    "fit_final_calibration",
    "is_quarterly_refit",
    "prepare_certified_strategy_month_decision",
    "validate_kernel_state_successor",
]
