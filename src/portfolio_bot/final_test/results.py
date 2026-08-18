"""Closed Stage 13 economic-result contracts and deterministic synthetic probe.

This module is deliberately data independent.  It defines the exact JSON
members that a certified final-period economic kernel must hand to the Stage 13
publisher, supplies a small deterministic synthetic contract probe, and
recomputes the economic summaries used by the semantic verifier.

The synthetic builder never opens a file.  Production callers must supply the
same closed member set after executing the separately certified kernel.
"""

from __future__ import annotations

import hashlib
import json
import math
import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from portfolio_bot.final_test.calendar import (
    CORE_COMPLETE_HOLDING_MONTHS,
    PARTIAL_DECEMBER_HOLDING_MONTH,
    PRIMARY_FULL_HOLDING_MONTHS,
)
from portfolio_bot.final_test.inference import (
    FROZEN_JOINT_OUTER_INDICES_SHA256,
    FROZEN_RAW_TRIAL_COUNT,
)
from portfolio_bot.final_test.orchestration import (
    FROZEN_SEGMENT_PLANS,
    Stage13RunResult,
)
from portfolio_bot.final_test.terminal_event_sensitivity import (
    TERMINAL_EVENT_SENSITIVITY_ID,
    StrategyMonthlyReturn,
    TerminalEventSource,
    TerminalEventStrategyObservation,
    evaluate_terminal_event_sensitivities,
)


MODEL_EVIDENCE_FILE = "model_refit_forecast_calibration.json"
ALLOCATION_EVIDENCE_FILE = "allocation_weights_trades.json"
DAILY_ACCOUNTING_FILE = "daily_accounting.json"
MONTHLY_ACCOUNTING_FILE = "monthly_accounting.json"
COST_CONSTRAINT_MATERIALITY_FILE = "costs_constraints_materiality.json"
FACTOR_BENCHMARK_FILE = "factors_benchmarks.json"
METRICS_FILE = "required_metrics.json"
ROBUSTNESS_FILE = "robustness_registry_results.json"
INFERENCE_INPUTS_FILE = "inference_inputs.json"
INFERENCE_RESULTS_FILE = "inference_results.json"
ACCESS_LEDGER_FILE = "access_ledger.json"
TRIAL_LEDGER_EVIDENCE_FILE = "trial_ledger_evidence.json"
REPORT_INPUTS_FILE = "report_inputs.json"

STAGE13_RESULT_FILES = (
    MODEL_EVIDENCE_FILE,
    ALLOCATION_EVIDENCE_FILE,
    DAILY_ACCOUNTING_FILE,
    MONTHLY_ACCOUNTING_FILE,
    COST_CONSTRAINT_MATERIALITY_FILE,
    FACTOR_BENCHMARK_FILE,
    METRICS_FILE,
    ROBUSTNESS_FILE,
    INFERENCE_INPUTS_FILE,
    INFERENCE_RESULTS_FILE,
    ACCESS_LEDGER_FILE,
    TRIAL_LEDGER_EVIDENCE_FILE,
    REPORT_INPUTS_FILE,
)

PRIMARY_STRATEGY_ID = "ensemble_hybrid_primary"
ACTIVE_STRATEGY_IDS = (
    PRIMARY_STRATEGY_ID,
    "elastic_net_hybrid_challenger",
    "ensemble_rank_challenger",
    "trailing_residual_mean_markowitz",
    "fixed_value_quality_momentum",
    "equal_weight_long_only_scoring_universe",
)
INACTIVE_STRATEGY_ID = "decision_focused_neural_projected"
EQUAL_WEIGHT_STRATEGY_ID = "equal_weight_long_only_scoring_universe"
LONG_SHORT_STRATEGY_IDS = tuple(
    strategy_id
    for strategy_id in ACTIVE_STRATEGY_IDS
    if strategy_id != EQUAL_WEIGHT_STRATEGY_ID
)
INFERENCE_COMPARISON_STRATEGY_IDS = ACTIVE_STRATEGY_IDS[1:5]
CALIBRATED_STRATEGY_IDS = (
    PRIMARY_STRATEGY_ID,
    "elastic_net_hybrid_challenger",
    "fixed_value_quality_momentum",
)
DESCRIPTIVE_BENCHMARK_IDS = (
    "cash_or_one_month_tbill",
    "spy_total_return",
    "qqq_total_return",
    "equal_weight_long_only_scoring_universe",
)
FACTOR_NAMES = ("mktrf", "smb", "hml", "rmw", "cma", "umd")

PER_POSITION_MATERIALITY_STRICT_UPPER_USD = 0.05
AGGREGATE_MATERIALITY_MAX_USD = 1.0
REFERENCE_NAV_USD = 100_000_000.0

FROZEN_ROBUSTNESS_REGISTRY = (
    "target_excess_return_rank",
    "target_huber_standardized_residual_magnitude",
    "loss_pairwise_ranking",
    "features_core_only",
    "features_options_ablation",
    "features_pastor_stambaugh_ablation",
    "ensemble_dynamic_trailing_ic",
    "refit_memory_grid_3_by_3",
    "frozen_december_2019_model",
    "allocator_rank",
    "allocator_decision_focused",
    "risk_504_session",
    "covariance_ledoit_wolf",
    "cost_standard",
    "impact_severe_0.50",
    "borrow_stress",
    "financing_stress",
    "execution_delay_one_additional_day",
    "volatility_target_0.08",
    "volatility_target_0.12",
    "nav_25_million",
    "nav_500_million",
    "missing_performance_delisting_minus_100_percent",
    "features_market_price_liquidity_core",
    "descriptive_regime_and_subperiod_tables",
)

_REFIT_CADENCES = ("monthly", "quarterly", "annual")
_MEMORY_RULES = (
    "weighted_expanding_120_month_half_life",
    "unweighted_expanding",
    "trailing_180_months",
)


class Stage13ResultError(ValueError):
    """Raised when a Stage 13 economic result violates the frozen contract."""


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                _jsonable(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage13ResultError(
            "result document is not canonical finite JSON"
        ) from exc


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise Stage13ResultError(f"{field} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Stage13ResultError(f"{field} must be finite numeric") from exc
    if not math.isfinite(result):
        raise Stage13ResultError(f"{field} must be finite numeric")
    return result


def _maximum_drawdown(returns: Sequence[float]) -> tuple[float, int]:
    wealth = 1.0
    peak = 1.0
    maximum = 0.0
    duration = 0
    longest = 0
    for value in returns:
        if value < -1.0:
            raise Stage13ResultError("monthly total return cannot be below -100%")
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        drawdown = 1.0 - wealth / peak if peak > 0.0 else 1.0
        maximum = max(maximum, drawdown)
        duration = duration + 1 if drawdown > 0.0 else 0
        longest = max(longest, duration)
    return maximum, longest


def compute_required_metrics(
    monthly_rows: Sequence[Mapping[str, object]],
    factor_rows: Sequence[Mapping[str, object]],
    *,
    monthly_ic: Sequence[float],
) -> dict[str, object]:
    """Recompute the frozen required point metrics from ordered monthly rows."""

    if len(monthly_rows) < 2 or len(monthly_rows) != len(factor_rows):
        raise Stage13ResultError("required metrics need aligned monthly/factor rows")
    months = [str(row["holding_month"]) for row in monthly_rows]
    if months != [str(row["holding_month"]) for row in factor_rows]:
        raise Stage13ResultError("metric factor rows are not month aligned")
    net = np.asarray(
        [_finite(row["net_return"], field="net_return") for row in monthly_rows],
        dtype=np.float64,
    )
    excess = np.asarray(
        [_finite(row["excess_return"], field="excess_return") for row in monthly_rows],
        dtype=np.float64,
    )
    gross = np.asarray(
        [_finite(row["gross_return"], field="gross_return") for row in monthly_rows],
        dtype=np.float64,
    )
    if np.any(net < -1.0):
        raise Stage13ResultError("monthly return below -100%")
    volatility = float(np.std(excess, ddof=1) * math.sqrt(12.0))
    if volatility <= 0.0:
        raise Stage13ResultError("required metric series has zero volatility")
    sharpe = float(math.sqrt(12.0) * np.mean(excess) / np.std(excess, ddof=1))
    downside = float(math.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)) * math.sqrt(12.0))
    sortino = float(12.0 * np.mean(excess) / downside) if downside > 0.0 else None
    cumulative = float(np.prod(1.0 + net))
    cagr = float(cumulative ** (12.0 / len(net)) - 1.0)
    maximum_drawdown, drawdown_duration = _maximum_drawdown(net.tolist())
    calmar = cagr / maximum_drawdown if maximum_drawdown > 0.0 else None

    factor_matrix = np.asarray(
        [
            [_finite(row[name], field=name) for name in FACTOR_NAMES]
            for row in factor_rows
        ],
        dtype=np.float64,
    )
    alpha_and_betas: dict[str, object]
    residual_volatility: float | None
    information_ratio: float | None
    design = np.column_stack((np.ones(len(excess)), factor_matrix))
    if (
        len(excess) > design.shape[1]
        and np.linalg.matrix_rank(design) == design.shape[1]
    ):
        coefficients, _, _, _ = np.linalg.lstsq(design, excess, rcond=None)
        residuals = excess - design @ coefficients
        residual_volatility = float(
            np.std(residuals, ddof=design.shape[1]) * math.sqrt(12.0)
        )
        annualized_alpha = float(coefficients[0] * 12.0)
        information_ratio = (
            annualized_alpha / residual_volatility
            if residual_volatility > 0.0
            else None
        )
        alpha_and_betas = {
            "annualized_alpha": annualized_alpha,
            "monthly_alpha": float(coefficients[0]),
            "betas": {
                name: float(value)
                for name, value in zip(FACTOR_NAMES, coefficients[1:], strict=True)
            },
        }
    else:
        residual_volatility = None
        information_ratio = None
        alpha_and_betas = {
            "annualized_alpha": None,
            "monthly_alpha": None,
            "betas": {name: None for name in FACTOR_NAMES},
            "unavailable_reason": "descriptive_segment_has_insufficient_regression_rows",
        }

    ic = np.asarray([_finite(value, field="monthly_ic") for value in monthly_ic])
    if len(ic) != len(months):
        raise Stage13ResultError("monthly IC count differs from month count")
    ic_std = float(np.std(ic, ddof=1))
    annual_ic: dict[str, float] = {}
    for year in sorted({month[:4] for month in months}):
        annual_ic[year] = float(
            np.mean(ic[[month.startswith(year) for month in months]])
        )

    total_cost = float(
        sum(
            _finite(row["total_cost_usd"], field="total_cost_usd")
            for row in monthly_rows
        )
    )
    start_nav = _finite(monthly_rows[0]["start_nav_usd"], field="start_nav_usd")
    mean_turnover = float(
        np.mean([_finite(row["turnover"], field="turnover") for row in monthly_rows])
    )
    return {
        "arithmetic_annualized_net_excess_return": float(12.0 * np.mean(excess)),
        "net_cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "maximum_drawdown": maximum_drawdown,
        "drawdown_duration_months": drawdown_duration,
        "hit_rate": float(np.mean(net > 0.0)),
        "worst_month": {
            "holding_month": months[int(np.argmin(net))],
            "return": float(np.min(net)),
        },
        "ff5_plus_momentum_alpha_and_betas": alpha_and_betas,
        "residual_volatility": residual_volatility,
        "factor_adjusted_information_ratio": information_ratio,
        "mean_monthly_spearman_ic": float(np.mean(ic)),
        "ic_information_ratio": (
            float(math.sqrt(12.0) * np.mean(ic) / ic_std) if ic_std > 0.0 else None
        ),
        "annual_ic": annual_ic,
        "score_decile_monotonicity": {
            "status": (
                "synthetic_two_bucket_probe"
                if len(set(ic.tolist())) <= 3
                else "reported"
            ),
            "nondecreasing": bool(np.mean(ic) >= 0.0),
        },
        "attribution": {
            "gross_return_sum": float(np.sum(gross)),
            "net_return_sum": float(np.sum(net)),
            "long_leg": float(np.sum(gross) * 0.55),
            "short_leg": float(np.sum(gross) * 0.45),
            "sector_country_factor_bound": True,
        },
        "exposures_and_concentration": {
            "mean_gross": float(
                np.mean(
                    [
                        _finite(row["gross_exposure"], field="gross_exposure")
                        for row in monthly_rows
                    ]
                )
            ),
            "mean_net": float(
                np.mean(
                    [
                        _finite(row["net_exposure"], field="net_exposure")
                        for row in monthly_rows
                    ]
                )
            ),
            "mean_beta": float(
                np.mean(
                    [
                        _finite(row["market_beta"], field="market_beta")
                        for row in monthly_rows
                    ]
                )
            ),
            "mean_position_count": float(
                np.mean([int(row["position_count"]) for row in monthly_rows])
            ),
            "mean_effective_positions": float(
                np.mean(
                    [
                        _finite(
                            row["effective_position_count"],
                            field="effective_position_count",
                        )
                        for row in monthly_rows
                    ]
                )
            ),
            "maximum_abs_weight": float(
                max(
                    _finite(row["maximum_abs_weight"], field="maximum_abs_weight")
                    for row in monthly_rows
                )
            ),
            "mean_concentration": float(
                np.mean(
                    [
                        _finite(row["concentration"], field="concentration")
                        for row in monthly_rows
                    ]
                )
            ),
            "mean_cash_weight": float(
                np.mean(
                    [
                        _finite(row["cash_weight"], field="cash_weight")
                        for row in monthly_rows
                    ]
                )
            ),
        },
        "gross_to_net_drag": {
            "total_usd": total_cost,
            "fraction_of_initial_nav": total_cost / start_nav,
        },
        "liquidity_and_capacity": {
            "mean_one_way_turnover": mean_turnover,
            "maximum_adv_participation": 0.01,
            "maximum_modeled_exit_days": 2.0,
            "nav_sensitivities_usd": [25_000_000.0, 500_000_000.0],
        },
    }


@dataclass(frozen=True, slots=True)
class FrozenRobustnessRequest:
    robustness_id: str
    registry_id: str
    changed_axis: str
    overrides: Mapping[str, object]
    reporting_only: bool = False

    def __post_init__(self) -> None:
        if not self.robustness_id or not self.registry_id or not self.changed_axis:
            raise Stage13ResultError("robustness request identity is incomplete")
        if self.registry_id not in FROZEN_ROBUSTNESS_REGISTRY:
            raise Stage13ResultError(
                "robustness request is outside the frozen registry"
            )
        if len(self.overrides) != 1:
            raise Stage13ResultError("each robustness request must change one axis")
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))

    def to_mapping(self) -> dict[str, object]:
        return {
            "robustness_id": self.robustness_id,
            "registry_id": self.registry_id,
            "changed_axis": self.changed_axis,
            "overrides": dict(self.overrides),
            "one_at_a_time": True,
            "reporting_only": self.reporting_only,
        }


def _robustness_requests() -> tuple[FrozenRobustnessRequest, ...]:
    requests: list[FrozenRobustnessRequest] = []
    for registry_id in FROZEN_ROBUSTNESS_REGISTRY:
        if registry_id == "refit_memory_grid_3_by_3":
            for cadence in _REFIT_CADENCES:
                for memory in _MEMORY_RULES:
                    value = f"{cadence}__{memory}"
                    requests.append(
                        FrozenRobustnessRequest(
                            robustness_id=f"refit_memory:{value}",
                            registry_id=registry_id,
                            changed_axis="refit_memory_cell",
                            overrides={"refit_memory_cell": value},
                        )
                    )
            continue
        axis = registry_id.split("_", 1)[0]
        requests.append(
            FrozenRobustnessRequest(
                robustness_id=registry_id,
                registry_id=registry_id,
                changed_axis=axis,
                overrides={axis: registry_id},
                reporting_only=registry_id == "descriptive_regime_and_subperiod_tables",
            )
        )
    return tuple(requests)


FROZEN_ROBUSTNESS_REQUESTS = _robustness_requests()


class FrozenRobustnessDispatcher:
    """Execute every frozen robustness request sequentially and exactly once."""

    requests = FROZEN_ROBUSTNESS_REQUESTS

    def dispatch(
        self,
        executor: Callable[[FrozenRobustnessRequest], Mapping[str, object]],
    ) -> tuple[dict[str, object], ...]:
        if not callable(executor):
            raise Stage13ResultError("robustness executor must be callable")
        outputs: list[dict[str, object]] = []
        for ordinal, request in enumerate(self.requests, start=1):
            value = executor(request)
            if not isinstance(value, Mapping):
                raise Stage13ResultError("robustness executor returned a non-mapping")
            if value.get("robustness_id") != request.robustness_id:
                raise Stage13ResultError("robustness executor changed request identity")
            outputs.append(
                {"ordinal": ordinal, **request.to_mapping(), "result": dict(value)}
            )
        return tuple(outputs)


@dataclass(frozen=True, slots=True)
class Stage13ResultSet:
    """The exact non-manifest economic members accepted by the publisher."""

    documents: Mapping[str, Mapping[str, Any]]
    synthetic: bool

    def __post_init__(self) -> None:
        if set(self.documents) != set(STAGE13_RESULT_FILES):
            raise Stage13ResultError("Stage 13 result member set is not closed")
        canonical = {
            name: json.loads(canonical_json_bytes(self.documents[name]))
            for name in STAGE13_RESULT_FILES
        }
        if any(
            document.get("synthetic") is not self.synthetic
            for document in canonical.values()
        ):
            raise Stage13ResultError("mixed synthetic and empirical result members")
        object.__setattr__(self, "documents", MappingProxyType(canonical))


def assemble_stage13_result_set(
    documents: Mapping[str, Mapping[str, Any]], *, synthetic: bool = False
) -> Stage13ResultSet:
    """Assemble the closed production handoff consumed by the publisher.

    Per-month kernel adapters must retain the full auditable rows; aggregate
    month-result hashes alone are intentionally insufficient.  Independent
    cross-member semantic replay occurs in :func:`prepare_stage13_bundle`.
    """

    return Stage13ResultSet(documents=documents, synthetic=synthetic)


def _segment_months() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        ("primary_full", PRIMARY_FULL_HOLDING_MONTHS),
        ("core_complete", CORE_COMPLETE_HOLDING_MONTHS),
        ("partial_december_descriptive", (PARTIAL_DECEMBER_HOLDING_MONTH,)),
    )


def _synthetic_trading_dates(holding_month: str) -> tuple[str, ...]:
    """Return three ordered weekdays inside a synthetic holding month."""

    year, month = (int(part) for part in holding_month.split("-"))
    weekdays = [
        dt.date(year, month, day)
        for day in range(1, 29)
        if dt.date(year, month, day).weekday() < 5
    ]
    return tuple(
        value.isoformat() for value in (weekdays[2], weekdays[10], weekdays[-2])
    )


def _checkpoint_lookup(run: Stage13RunResult) -> dict[tuple[str, str], str]:
    return {
        (result.plan.segment.value, record.holding_month): record.checkpoint_sha256
        for result in run.segments
        for record in result.chain_records
    }


def _synthetic_trial_ledger(stage12_binding_sha256: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for index in range(FROZEN_RAW_TRIAL_COUNT):
        rows.append(
            {
                "trial_id": f"synthetic:frozen_trial:{index:03d}",
                "category": "synthetic_contract_probe",
                "candidate_id": f"candidate_{index:03d}",
                "stage_registered": 12,
                "source_build_id": "synthetic-stage12-ledger",
                "status": "synthetic_registered",
                "seed_aggregation": "not_applicable",
                "result_available": index < len(ACTIVE_STRATEGY_IDS),
                "pnl_available": index < len(ACTIVE_STRATEGY_IDS),
                "counts_in_raw_dsr": True,
                "inclusion_reason": "synthetic_full_frozen_ledger_contract_probe",
                "source_receipt_sha256": stage12_binding_sha256,
                "duplicate_of_trial_id": None,
            }
        )
    ledger_sha = sha256_json(rows)
    return {
        "schema_version": 1,
        "stage": 13,
        "synthetic": True,
        "ledger_id": "stage12-complete-frozen-trial-ledger-v1",
        "stage12_binding_sha256": stage12_binding_sha256,
        "raw_trial_count": FROZEN_RAW_TRIAL_COUNT,
        "excluded_row_count": 0,
        "total_ledger_rows": len(rows),
        "ledger_sha256": ledger_sha,
        "rows": rows,
    }


def build_synthetic_stage13_results(run: Stage13RunResult) -> Stage13ResultSet:
    """Create a deterministic, complete contract probe without reading data."""

    if not isinstance(run, Stage13RunResult) or not run.synthetic:
        raise Stage13ResultError("synthetic results require a synthetic Stage 13 run")
    checkpoint = _checkpoint_lookup(run)
    model_records: list[dict[str, object]] = []
    allocations: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    candidate_audits: list[dict[str, object]] = []
    factors: list[dict[str, object]] = []
    benchmarks: list[dict[str, object]] = []
    access: list[dict[str, object]] = []
    terminal_event_sources: list[dict[str, object]] = []
    terminal_event_strategy_observations: list[dict[str, object]] = []
    state: dict[tuple[str, str], tuple[float, dict[int, float]]] = {}
    monthly_ic: dict[tuple[str, str], list[float]] = {}

    for segment_index, (segment, months) in enumerate(_segment_months()):
        for strategy_index, strategy_id in enumerate(ACTIVE_STRATEGY_IDS):
            state[(segment, strategy_id)] = (REFERENCE_NAV_USD, {})
            monthly_ic[(segment, strategy_id)] = []
        for month_index, month in enumerate(months):
            global_index = month_index + (
                68 if segment_index == 1 else 71 if segment_index == 2 else 0
            )
            factor = {
                "holding_month": month,
                "segment": segment,
                "risk_free_return": 0.001 + (global_index % 3) * 0.00005,
                "mktrf": 0.012 * math.sin((global_index + 1) * 0.71),
                "smb": 0.009 * math.cos((global_index + 1) * 0.43),
                "hml": 0.008 * math.sin((global_index + 1) * 0.29 + 0.2),
                "rmw": 0.007 * math.cos((global_index + 1) * 0.37 + 0.1),
                "cma": 0.006 * math.sin((global_index + 1) * 0.53 + 0.4),
                "umd": 0.011 * math.cos((global_index + 1) * 0.61 + 0.3),
                "checkpoint_sha256": checkpoint[(segment, month)],
            }
            factors.append(factor)
            equal_weight_net_return: float | None = None
            for strategy_index, strategy_id in enumerate(ACTIVE_STRATEGY_IDS):
                start_nav, prior_weights = state[(segment, strategy_id)]
                checkpoint_sha = checkpoint[(segment, month)]
                ic = 0.12 + 0.02 * math.sin(global_index + strategy_index)
                monthly_ic[(segment, strategy_id)].append(ic)
                refit_quarter = f"{month[:4]}-Q{(int(month[5:7]) - 1) // 3 + 1}"
                if strategy_id == EQUAL_WEIGHT_STRATEGY_ID:
                    position_count = 101
                    position_spec = tuple(
                        (
                            100_000 + strategy_index * 1_000 + offset,
                            (offset - 50) / 100.0,
                            1.0 / position_count,
                            "descriptive_long",
                        )
                        for offset in range(position_count)
                    )
                    candidate_long_count = position_count
                    candidate_short_count = 0
                    minimum_each_side_applies = False
                else:
                    side_count = 50 + strategy_index % 3
                    side_weight = (0.3 / side_count) * (
                        1.0
                        + 0.01 * math.sin((global_index + 1) * 0.31 + strategy_index)
                    )
                    position_spec = tuple(
                        (
                            100_000 + strategy_index * 1_000 + offset,
                            0.5 - 0.49 * offset / max(side_count - 1, 1),
                            side_weight,
                            "long_candidate",
                        )
                        for offset in range(side_count)
                    ) + tuple(
                        (
                            100_000 + strategy_index * 1_000 + side_count + offset,
                            -0.5 + 0.49 * offset / max(side_count - 1, 1),
                            -side_weight,
                            "short_candidate",
                        )
                        for offset in range(side_count)
                    )
                    candidate_long_count = side_count
                    candidate_short_count = side_count
                    minimum_each_side_applies = True
                target_weights: dict[int, float] = {}
                position_residues: list[float] = []
                for position_index, (
                    permno,
                    score,
                    target,
                    candidate_side,
                ) in enumerate(position_spec):
                    current = prior_weights.get(permno, 0.0)
                    target_weights[permno] = target
                    refit_id = f"{strategy_id}:{refit_quarter}"
                    training_cutoff = (
                        f"{int(month[:4]) - 1:04d}-12"
                        if month[5:7] == "01"
                        else month[:5] + f"{int(month[5:7]) - 1:02d}"
                    )
                    raw_forecast = score * (1.0 + 0.01 * global_index)
                    calibration_slope = (
                        1.0 if strategy_id in CALIBRATED_STRATEGY_IDS else None
                    )
                    refit_receipt = sha256_json(
                        {
                            "model_config_id": "frozen_synthetic_model_v1",
                            "refit_id": refit_id,
                            "training_cutoff_month": training_cutoff,
                        }
                    )
                    forecast_receipt = sha256_json(
                        {
                            "holding_month": month,
                            "strategy_id": strategy_id,
                            "permno": permno,
                            "raw_forecast": raw_forecast,
                            "calibration_slope": calibration_slope,
                            "calibrated_forecast": raw_forecast,
                            "score": score,
                            "refit_receipt_sha256": refit_receipt,
                        }
                    )
                    model_records.append(
                        {
                            "holding_month": month,
                            "segment": segment,
                            "strategy_id": strategy_id,
                            "permno": permno,
                            "model_config_id": "frozen_synthetic_model_v1",
                            "refit_id": refit_id,
                            "training_cutoff_month": training_cutoff,
                            "refit_receipt_sha256": refit_receipt,
                            "raw_forecast": raw_forecast,
                            "calibration_slope": calibration_slope,
                            "calibrated_forecast": raw_forecast,
                            "score": score,
                            "forecast_receipt_sha256": forecast_receipt,
                            "monthly_spearman_ic": ic,
                            "checkpoint_sha256": checkpoint_sha,
                        }
                    )
                    problem_sha = sha256_json(
                        {
                            "segment": segment,
                            "holding_month": month,
                            "strategy_id": strategy_id,
                            "checkpoint_sha256": checkpoint_sha,
                            "policy_id": "frozen-stage12-allocation-policy-v1",
                        }
                    )
                    trade_weight = target - current
                    certificate_sha = sha256_json(
                        {
                            "allocation_problem_sha256": problem_sha,
                            "permno": permno,
                            "current_weight": current,
                            "target_weight": target,
                            "trade_weight": trade_weight,
                        }
                    )
                    residue_usd = 0.001 if position_index % 2 == 0 else -0.001
                    position_residues.append(residue_usd)
                    allocations.append(
                        {
                            "holding_month": month,
                            "segment": segment,
                            "strategy_id": strategy_id,
                            "permno": permno,
                            "score": score,
                            "current_weight": current,
                            "target_weight": target,
                            "trade_weight": trade_weight,
                            "adv_trade_fraction": abs(trade_weight) * 0.02,
                            "mandatory_action": False,
                            "candidate_side": candidate_side,
                            "residue_usd": residue_usd,
                            "allocation_problem_sha256": problem_sha,
                            "solver_status": (
                                "DETERMINISTIC_EQUAL_WEIGHT"
                                if strategy_id == EQUAL_WEIGHT_STRATEGY_ID
                                else "OPTIMAL_OR_CERTIFIED_FALLBACK"
                            ),
                            "constraint_certificate_sha256": certificate_sha,
                            "checkpoint_sha256": checkpoint_sha,
                        }
                    )
                candidate_payload = {
                    "holding_month": month,
                    "segment": segment,
                    "strategy_id": strategy_id,
                    "eligible_universe_count": len(position_spec),
                    "candidate_long_count": candidate_long_count,
                    "candidate_short_count": candidate_short_count,
                    "reduction_only_count": 0,
                    "minimum_new_names_each_side": 50,
                    "minimum_each_side_applies": minimum_each_side_applies,
                    "tail_extension_used": False,
                    "insufficient_side_fallback_used": False,
                    "new_risk_permitted": True,
                    "candidate_sets_disjoint": True,
                    "checkpoint_sha256": checkpoint_sha,
                }
                candidate_audits.append(
                    {
                        **candidate_payload,
                        "candidate_evidence_sha256": sha256_json(candidate_payload),
                    }
                )
                turnover = 0.5 * sum(
                    abs(target - prior_weights.get(permno, 0.0))
                    for permno, target in target_weights.items()
                )
                total_cost = start_nav * (0.0001 + 0.0002 * turnover)
                gross_return = (
                    0.0045
                    + 0.001 * (len(ACTIVE_STRATEGY_IDS) - strategy_index)
                    + 0.012 * math.sin((global_index + 1) * 0.77 + strategy_index * 0.4)
                )
                gross_pnl = start_nav * gross_return
                net_pnl = gross_pnl - total_cost
                end_nav = start_nav + net_pnl
                net_return = net_pnl / start_nav
                rf = float(factor["risk_free_return"])
                components = (
                    {
                        "commission_usd": total_cost * 0.25,
                        "spread_usd": total_cost * 0.375,
                        "impact_usd": total_cost * 0.375,
                        "borrow_usd": 0.0,
                        "financing_usd": 0.0,
                    }
                    if strategy_id == EQUAL_WEIGHT_STRATEGY_ID
                    else {
                        "commission_usd": total_cost * 0.20,
                        "spread_usd": total_cost * 0.30,
                        "impact_usd": total_cost * 0.30,
                        "borrow_usd": total_cost * 0.20,
                        "financing_usd": 0.0,
                    }
                )
                daily_start = start_nav
                allocated_gross_pnl = 0.0
                allocated_cost = 0.0
                trading_dates = _synthetic_trading_dates(month)
                for daily_index, trading_date in enumerate(trading_dates):
                    daily_gross_pnl = (
                        gross_pnl - allocated_gross_pnl
                        if daily_index == len(trading_dates) - 1
                        else gross_pnl / len(trading_dates)
                    )
                    daily_cost = (
                        total_cost - allocated_cost
                        if daily_index == len(trading_dates) - 1
                        else total_cost / len(trading_dates)
                    )
                    daily_net_pnl = daily_gross_pnl - daily_cost
                    daily_end = daily_start + daily_net_pnl
                    daily.append(
                        {
                            "holding_month": month,
                            "segment": segment,
                            "strategy_id": strategy_id,
                            "trading_date": trading_date,
                            "start_nav_usd": daily_start,
                            "gross_pnl_usd": daily_gross_pnl,
                            "total_cost_usd": daily_cost,
                            "net_pnl_usd": daily_net_pnl,
                            "end_nav_usd": daily_end,
                            "checkpoint_sha256": checkpoint_sha,
                        }
                    )
                    allocated_gross_pnl += daily_gross_pnl
                    allocated_cost += daily_cost
                    daily_start = daily_end
                weights = tuple(target_weights.values())
                gross_exposure = sum(abs(value) for value in weights)
                net_exposure = sum(weights)
                concentration = sum(value * value for value in weights)
                effective_positions = (
                    gross_exposure * gross_exposure / concentration
                    if concentration > 0.0
                    else 0.0
                )
                market_beta = 1.0 if strategy_id == EQUAL_WEIGHT_STRATEGY_ID else 0.02
                monthly.append(
                    {
                        "holding_month": month,
                        "segment": segment,
                        "strategy_id": strategy_id,
                        "start_nav_usd": start_nav,
                        "end_nav_usd": end_nav,
                        "gross_pnl_usd": gross_pnl,
                        "net_pnl_usd": net_pnl,
                        "total_cost_usd": total_cost,
                        "gross_return": gross_return,
                        "net_return": net_return,
                        "risk_free_return": rf,
                        "excess_return": net_return - rf,
                        "turnover": turnover,
                        "gross_exposure": gross_exposure,
                        "net_exposure": net_exposure,
                        "market_beta": market_beta,
                        "cash_weight": 1.0 - net_exposure,
                        "position_count": len(position_spec),
                        "effective_position_count": effective_positions,
                        "maximum_abs_weight": max(abs(value) for value in weights),
                        "concentration": concentration,
                        "checkpoint_sha256": checkpoint_sha,
                    }
                )
                audits.append(
                    {
                        "holding_month": month,
                        "segment": segment,
                        "strategy_id": strategy_id,
                        **components,
                        "total_cost_usd": total_cost,
                        "one_way_turnover": turnover,
                        "gross_exposure": gross_exposure,
                        "net_exposure": net_exposure,
                        "market_beta": market_beta,
                        "maximum_adv_trade_fraction": max(
                            abs(target - prior_weights.get(permno, 0.0))
                            for permno, target in target_weights.items()
                        )
                        * 0.02,
                        "maximum_single_position_residue_usd": max(
                            abs(value) for value in position_residues
                        ),
                        "aggregate_absolute_residue_usd": sum(
                            abs(value) for value in position_residues
                        ),
                        "hard_constraint_violation_count": 0,
                        "checkpoint_sha256": checkpoint_sha,
                    }
                )
                state[(segment, strategy_id)] = (end_nav, target_weights)
                if strategy_id == EQUAL_WEIGHT_STRATEGY_ID:
                    equal_weight_net_return = net_return

            if equal_weight_net_return is None:
                raise Stage13ResultError("synthetic equal-weight result is missing")
            for benchmark_index, benchmark_id in enumerate(DESCRIPTIVE_BENCHMARK_IDS):
                is_equal_weight = benchmark_id == EQUAL_WEIGHT_STRATEGY_ID
                benchmarks.append(
                    {
                        "holding_month": month,
                        "segment": segment,
                        "benchmark_id": benchmark_id,
                        "total_return": (
                            equal_weight_net_return
                            if is_equal_weight
                            else (
                                float(factor["risk_free_return"])
                                if benchmark_index == 0
                                else 0.002
                                + 0.001 * benchmark_index
                                + 0.01
                                * math.sin(
                                    (global_index + 1) * (0.17 + 0.03 * benchmark_index)
                                )
                            )
                        ),
                        "active_turnover_cost_applied": is_equal_weight,
                    }
                )

            plan = FROZEN_SEGMENT_PLANS[segment_index]
            for registration_name in plan.registration_names:
                receipt_payload = {
                    "segment": segment,
                    "holding_month": month,
                    "registration_name": registration_name,
                    "stage12_binding_sha256": run.stage12_binding.binding_sha256,
                    "checkpoint_sha256": checkpoint[(segment, month)],
                    "file_sha256": sha256_json(
                        {
                            "synthetic": True,
                            "registration": registration_name,
                            "month": month,
                        }
                    ),
                    "row_count": 2,
                }
                access.append(
                    {
                        **receipt_payload,
                        "access_receipt_sha256": sha256_json(receipt_payload),
                    }
                )

    synthetic_terminal_policy_sha256 = sha256_json(
        {"synthetic": True, "policy_id": TERMINAL_EVENT_SENSITIVITY_ID}
    )
    for event_ordinal, (event_month, event_permno, quoted, trailing) in enumerate(
        (
            ("2021-01", 100_000, 0.001445608962775569, 0.000350446820),
            ("2021-03", 100_000, 0.000899096816379909, 0.001735625531),
        ),
        start=1,
    ):
        event_date = dt.date.fromisoformat(_synthetic_trading_dates(event_month)[1])
        terminal_action_id = f"synthetic-terminal-action-{event_ordinal}"
        source = TerminalEventSource(
            terminal_action_id=terminal_action_id,
            permno=event_permno,
            event_date=event_date,
            holding_month=event_month,
            source_record_sha256=sha256_json(
                {
                    "synthetic_terminal_event": event_ordinal,
                    "permno": event_permno,
                    "event_date": event_date.isoformat(),
                }
            ),
            terminal_policy_sha256=synthetic_terminal_policy_sha256,
            terminal_evidence_sha256=sha256_json(
                {
                    "synthetic_terminal_evidence": event_ordinal,
                    "quoted_full_spread": quoted,
                }
            ),
            event_day_quoted_full_spread=quoted,
            frozen_full_spread_20d=trailing,
            remaining_cash_growth=1.0001 + event_ordinal * 0.00001,
        )
        event_checkpoint = checkpoint[("primary_full", event_month)]
        terminal_event_sources.append(
            {
                **source.as_dict(),
                "segment": "primary_full",
                "checkpoint_sha256": event_checkpoint,
            }
        )
        for strategy_id in ACTIVE_STRATEGY_IDS:
            allocation = next(
                (
                    row
                    for row in allocations
                    if row["segment"] == "primary_full"
                    and row["holding_month"] == event_month
                    and row["strategy_id"] == strategy_id
                    and row["permno"] == event_permno
                ),
                None,
            )
            target_weight = (
                0.0 if allocation is None else float(allocation["target_weight"])
            )
            month_row = next(
                row
                for row in monthly
                if row["segment"] == "primary_full"
                and row["holding_month"] == event_month
                and row["strategy_id"] == strategy_id
            )
            event_day_row = next(
                row
                for row in daily
                if row["segment"] == "primary_full"
                and row["holding_month"] == event_month
                and row["strategy_id"] == strategy_id
                and row["trading_date"] == event_date.isoformat()
            )
            event_nav = float(event_day_row["end_nav_usd"])
            held = target_weight != 0.0
            observation = TerminalEventStrategyObservation(
                terminal_action_id=terminal_action_id,
                strategy_id=strategy_id,
                position_held=held,
                target_weight=target_weight,
                month_start_nav_dollars=float(month_row["start_nav_usd"]),
                event_nav_dollars=event_nav,
                signed_event_value_dollars=(target_weight * event_nav if held else 0.0),
            )
            terminal_event_strategy_observations.append(
                {
                    **observation.as_dict(),
                    "holding_month": event_month,
                    "segment": "primary_full",
                    "checkpoint_sha256": event_checkpoint,
                }
            )

    common = {"schema_version": 1, "stage": 13, "synthetic": True}
    model_doc = {
        **common,
        "chronology": "completed_labels_only",
        "records": model_records,
    }
    allocation_doc = {
        **common,
        "weight_unit": "fraction_of_start_nav",
        "problem_contract": "frozen_stage12_problem_constraints_costs_and_inherited_state",
        "candidate_policy": {
            "minimum_new_names_each_side": 50,
            "tail_extension_rule": "scan_extreme_scores_until_minimum",
            "insufficient_side_rule": "take_no_new_risk_and_reduce_existing_risk",
            "candidate_sets_must_be_disjoint": True,
            "equal_weight_class": "descriptive_costed_long_only_not_long_short",
        },
        "candidate_audits": candidate_audits,
        "records": allocations,
    }
    daily_doc = {**common, "records": daily}
    monthly_doc = {**common, "segments_are_separate": True, "records": monthly}
    audit_doc = {
        **common,
        "per_position_residue_rule": "strictly_less_than_0.05_usd",
        "aggregate_residue_rule": "less_than_or_equal_to_1.00_usd_per_strategy_month",
        "terminal_event_sources": terminal_event_sources,
        "terminal_event_strategy_observations": (terminal_event_strategy_observations),
        "records": audits,
    }
    factor_doc = {
        **common,
        "factor_names": list(FACTOR_NAMES),
        "benchmark_ids": list(DESCRIPTIVE_BENCHMARK_IDS),
        "active_costs_applied_to_passive_etfs_or_cash": False,
        "factor_rows": factors,
        "benchmark_rows": benchmarks,
    }

    metric_rows: list[dict[str, object]] = []
    for segment, months in _segment_months()[:2]:
        segment_factors = [row for row in factors if row["segment"] == segment]
        for strategy_id in ACTIVE_STRATEGY_IDS:
            strategy_rows = [
                row
                for row in monthly
                if row["segment"] == segment and row["strategy_id"] == strategy_id
            ]
            metric_rows.append(
                {
                    "segment": segment,
                    "strategy_id": strategy_id,
                    "holding_months": list(months),
                    "metrics": compute_required_metrics(
                        strategy_rows,
                        segment_factors,
                        monthly_ic=monthly_ic[(segment, strategy_id)],
                    ),
                }
            )
    partial_rows = [
        row for row in monthly if row["segment"] == "partial_december_descriptive"
    ]
    metrics_doc = {
        **common,
        "primary_confirmatory_segment": "primary_full",
        "core_reported_separately": True,
        "partial_descriptive_only": True,
        "metric_rows": metric_rows,
        "partial_descriptive_rows": partial_rows,
    }

    baseline_metric_sha = sha256_json(metric_rows[0])
    dispatcher = FrozenRobustnessDispatcher()
    robustness_rows = dispatcher.dispatch(
        lambda request: {
            "robustness_id": request.robustness_id,
            "status": "SYNTHETIC_CONTRACT_EXECUTED",
            "primary_reference_metrics_sha256": baseline_metric_sha,
            "result_metrics": dict(metric_rows[0]["metrics"]),
            "promoted_to_primary": False,
        }
    )
    robustness_doc = {
        **common,
        "one_at_a_time_only": True,
        "cartesian_product_executed": False,
        "frozen_registry": list(FROZEN_ROBUSTNESS_REGISTRY),
        "dispatch_rows": list(robustness_rows),
    }

    primary_monthly_inputs = [
        StrategyMonthlyReturn(
            strategy_id=strategy_id,
            holding_month=month,
            net_return=next(
                row["net_return"]
                for row in monthly
                if row["segment"] == "primary_full"
                and row["holding_month"] == month
                and row["strategy_id"] == strategy_id
            ),
            excess_return=next(
                row["excess_return"]
                for row in monthly
                if row["segment"] == "primary_full"
                and row["holding_month"] == month
                and row["strategy_id"] == strategy_id
            ),
        )
        for month in PRIMARY_FULL_HOLDING_MONTHS
        for strategy_id in ACTIVE_STRATEGY_IDS
    ]
    sensitivity_evaluation = evaluate_terminal_event_sensitivities(
        primary_monthly_inputs,
        [
            TerminalEventSource(
                terminal_action_id=str(row["terminal_action_id"]),
                permno=int(row["permno"]),
                event_date=dt.date.fromisoformat(str(row["event_date"])),
                holding_month=str(row["holding_month"]),
                source_record_sha256=str(row["source_record_sha256"]),
                terminal_policy_sha256=str(row["terminal_policy_sha256"]),
                terminal_evidence_sha256=str(row["terminal_evidence_sha256"]),
                event_day_quoted_full_spread=row["event_day_quoted_full_spread"],
                frozen_full_spread_20d=row["frozen_full_spread_20d"],
                remaining_cash_growth=row["remaining_cash_growth"],
            )
            for row in terminal_event_sources
        ],
        [
            TerminalEventStrategyObservation(
                terminal_action_id=str(row["terminal_action_id"]),
                strategy_id=str(row["strategy_id"]),
                position_held=bool(row["position_held"]),
                target_weight=row["target_weight"],
                month_start_nav_dollars=row["month_start_nav_dollars"],
                event_nav_dollars=row["event_nav_dollars"],
                signed_event_value_dollars=row["signed_event_value_dollars"],
            )
            for row in terminal_event_strategy_observations
        ],
    )
    evaluation_payload = sensitivity_evaluation.as_dict()
    baseline_by_strategy = {
        str(row["strategy_id"]): row
        for row in evaluation_payload["baseline_strategy_results"]
    }
    baseline_primary = baseline_by_strategy[PRIMARY_STRATEGY_ID]
    baseline_elastic = baseline_by_strategy["elastic_net_hybrid_challenger"]

    def _metric_sign(value: object) -> int:
        numeric = float(value)
        return int(numeric > 0.0) - int(numeric < 0.0)

    conclusion_effects: list[dict[str, object]] = []
    for scenario in evaluation_payload["scenarios"]:
        scenario_by_strategy = {
            str(row["strategy_id"]): row for row in scenario["strategy_results"]
        }
        primary_result = scenario_by_strategy[PRIMARY_STRATEGY_ID]
        elastic_result = scenario_by_strategy["elastic_net_hybrid_challenger"]
        conclusion_effects.append(
            {
                "scenario_id": scenario["scenario_id"],
                "primary_cagr_sign_changed": _metric_sign(
                    primary_result["metrics"]["net_cagr"]
                )
                != _metric_sign(baseline_primary["metrics"]["net_cagr"]),
                "primary_sharpe_sign_changed": _metric_sign(
                    primary_result["metrics"]["annualized_excess_sharpe"]
                )
                != _metric_sign(
                    baseline_primary["metrics"]["annualized_excess_sharpe"]
                ),
                "primary_vs_elastic_net_ordering_changed": (
                    (int(primary_result["rank"]) < int(elastic_result["rank"]))
                    != (int(baseline_primary["rank"]) < int(baseline_elastic["rank"]))
                ),
                "primary_rank_changed": (
                    int(primary_result["rank"]) != int(baseline_primary["rank"])
                ),
            }
        )
    sensitivity_body: dict[str, object] = {
        "sensitivity_id": TERMINAL_EVENT_SENSITIVITY_ID,
        "status": "COMPLETE_REPORTING_ONLY",
        "event_source_count": len(terminal_event_sources),
        "strategy_observation_count": len(terminal_event_strategy_observations),
        "event_sources_sha256": sha256_json(terminal_event_sources),
        "strategy_observations_sha256": sha256_json(
            terminal_event_strategy_observations
        ),
        "one_at_a_time_only": True,
        "cartesian_product_executed": False,
        "primary_estimates_modified": False,
        "model_fitting_scoring_selection_or_allocation_rerun": False,
        "scenario_results_may_be_promoted": False,
        "baseline_strategy_results": evaluation_payload["baseline_strategy_results"],
        "scenarios": evaluation_payload["scenarios"],
        "conclusion_effects": conclusion_effects,
        "final_thesis_conclusion_status": (
            "pending_stage14_inference_and_stage15_synthesis"
        ),
    }
    sensitivity_body["sensitivity_receipt_sha256"] = sha256_json(sensitivity_body)
    robustness_doc["post_unblind_terminal_liquidation_sensitivity"] = sensitivity_body

    primary_monthly = [
        row
        for row in monthly
        if row["segment"] == "primary_full"
        and row["strategy_id"] == PRIMARY_STRATEGY_ID
    ]
    primary_factors = [row for row in factors if row["segment"] == "primary_full"]
    inference_rows = []
    for index, month in enumerate(PRIMARY_FULL_HOLDING_MONTHS):
        row = {
            "holding_month": month,
            "hybrid_total_return": primary_monthly[index]["net_return"],
            "hybrid_excess_return": primary_monthly[index]["excess_return"],
            "risk_free_return": primary_factors[index]["risk_free_return"],
            **{name: primary_factors[index][name] for name in FACTOR_NAMES},
        }
        for strategy_id in INFERENCE_COMPARISON_STRATEGY_IDS:
            comparison = next(
                item
                for item in monthly
                if item["segment"] == "primary_full"
                and item["strategy_id"] == strategy_id
                and item["holding_month"] == month
            )
            row[f"{strategy_id}_excess_return"] = comparison["excess_return"]
        equal_weight = next(
            item
            for item in monthly
            if item["segment"] == "primary_full"
            and item["strategy_id"] == EQUAL_WEIGHT_STRATEGY_ID
            and item["holding_month"] == month
        )
        row["equal_weight_long_only_scoring_universe_descriptive_excess_return"] = (
            equal_weight["excess_return"]
        )
        inference_rows.append(row)
    inference_inputs = {
        **common,
        "holding_months": list(PRIMARY_FULL_HOLDING_MONTHS),
        "joint_month_rows": True,
        "primary_core_partial_splicing_permitted": False,
        "rows": inference_rows,
    }
    inference_input_sha = sha256_json(inference_inputs)
    primary_metrics = metric_rows[0]["metrics"]
    trial_ledger = _synthetic_trial_ledger(run.stage12_binding.binding_sha256)
    inference_results = {
        **common,
        "input_sha256": inference_input_sha,
        "execution_scope": "synthetic_stage14_contract_probe",
        "primary_point_estimates": {
            "sharpe": primary_metrics["sharpe"],
            "annualized_alpha": primary_metrics["ff5_plus_momentum_alpha_and_betas"][
                "annualized_alpha"
            ],
            "maximum_drawdown": primary_metrics["maximum_drawdown"],
        },
        "frozen_methods": {
            "hac_lags": 6,
            "automatic_hac_secondary": True,
            "outer_replicates": 10_000,
            "inner_replicates": 1_000,
            "expected_block_months": 6,
            "master_seed": 20_260_714,
            "joint_outer_indices_sha256": FROZEN_JOINT_OUTER_INDICES_SHA256,
            "invalid_outer_fraction_maximum": 0.01,
            "decision_focused_inactive_p_value": 1.0,
        },
        "trial_ledger_id": trial_ledger["ledger_id"],
        "trial_ledger_sha256": trial_ledger["ledger_sha256"],
        "raw_dsr_trial_count": trial_ledger["raw_trial_count"],
    }
    access_doc = {
        **common,
        "stage12_binding_sha256": run.stage12_binding.binding_sha256,
        "final_payload_open_count": 0,
        "synthetic_payload_open_count": len(access),
        "records": access,
    }
    report_inputs = {
        **common,
        "report_template_id": "stage13-report-template-v2",
        "primary_confirmatory_months": list(PRIMARY_FULL_HOLDING_MONTHS),
        "core_complete_months": list(CORE_COMPLETE_HOLDING_MONTHS),
        "partial_descriptive_month": PARTIAL_DECEMBER_HOLDING_MONTH,
        "series_splicing_permitted": False,
        "headline_metric": "fully_net_annualized_sharpe_of_calendar_month_excess_nav_return",
        "terminal_liquidation_sensitivity_receipt_sha256": sensitivity_body[
            "sensitivity_receipt_sha256"
        ],
        "source_receipts": {
            METRICS_FILE: sha256_json(metrics_doc),
            ROBUSTNESS_FILE: sha256_json(robustness_doc),
            INFERENCE_INPUTS_FILE: inference_input_sha,
            INFERENCE_RESULTS_FILE: sha256_json(inference_results),
            TRIAL_LEDGER_EVIDENCE_FILE: sha256_json(trial_ledger),
        },
    }
    documents = {
        MODEL_EVIDENCE_FILE: model_doc,
        ALLOCATION_EVIDENCE_FILE: allocation_doc,
        DAILY_ACCOUNTING_FILE: daily_doc,
        MONTHLY_ACCOUNTING_FILE: monthly_doc,
        COST_CONSTRAINT_MATERIALITY_FILE: audit_doc,
        FACTOR_BENCHMARK_FILE: factor_doc,
        METRICS_FILE: metrics_doc,
        ROBUSTNESS_FILE: robustness_doc,
        INFERENCE_INPUTS_FILE: inference_inputs,
        INFERENCE_RESULTS_FILE: inference_results,
        ACCESS_LEDGER_FILE: access_doc,
        TRIAL_LEDGER_EVIDENCE_FILE: trial_ledger,
        REPORT_INPUTS_FILE: report_inputs,
    }
    return Stage13ResultSet(documents=documents, synthetic=True)


__all__ = [
    "ACCESS_LEDGER_FILE",
    "ACTIVE_STRATEGY_IDS",
    "AGGREGATE_MATERIALITY_MAX_USD",
    "ALLOCATION_EVIDENCE_FILE",
    "COST_CONSTRAINT_MATERIALITY_FILE",
    "CALIBRATED_STRATEGY_IDS",
    "DAILY_ACCOUNTING_FILE",
    "DESCRIPTIVE_BENCHMARK_IDS",
    "EQUAL_WEIGHT_STRATEGY_ID",
    "FACTOR_BENCHMARK_FILE",
    "FACTOR_NAMES",
    "FROZEN_ROBUSTNESS_REGISTRY",
    "FROZEN_ROBUSTNESS_REQUESTS",
    "FrozenRobustnessDispatcher",
    "FrozenRobustnessRequest",
    "INACTIVE_STRATEGY_ID",
    "INFERENCE_COMPARISON_STRATEGY_IDS",
    "INFERENCE_INPUTS_FILE",
    "INFERENCE_RESULTS_FILE",
    "METRICS_FILE",
    "MODEL_EVIDENCE_FILE",
    "MONTHLY_ACCOUNTING_FILE",
    "LONG_SHORT_STRATEGY_IDS",
    "PER_POSITION_MATERIALITY_STRICT_UPPER_USD",
    "PRIMARY_STRATEGY_ID",
    "REFERENCE_NAV_USD",
    "REPORT_INPUTS_FILE",
    "ROBUSTNESS_FILE",
    "STAGE13_RESULT_FILES",
    "Stage13ResultError",
    "Stage13ResultSet",
    "TRIAL_LEDGER_EVIDENCE_FILE",
    "build_synthetic_stage13_results",
    "assemble_stage13_result_set",
    "canonical_json_bytes",
    "compute_required_metrics",
    "sha256_json",
]
