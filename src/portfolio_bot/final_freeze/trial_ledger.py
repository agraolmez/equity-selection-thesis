"""Construct the complete frozen trial ledger used by the Deflated Sharpe Ratio.

The 346 development selector configurations are read from the authenticated
Stage 9 configuration table.  Every additional row is fixed below from
Methodology v1.  The implementation deliberately records descriptive
benchmarks and duplicate robustness mentions as excluded rows rather than
silently dropping them.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq
import pyarrow as pa

from portfolio_bot.final_freeze.inventory import stable_file


STAGE9_SELECTOR_BUILD_ID = "phase04-selector-results-09a59d61ed58e77d80448a97"
STAGE9_CONFIGURATION_SHA256 = (
    "28bd5def81d7ef738a00e55f085d242668bceb004238a63cd8850ff3ccf3dda4"
)
METHODOLOGY_SHA256 = (
    "77ac9a4a79b0f46188b56f9a9a3d4affccdbfb13feafdc0f5facc79f2bbc6e81"
)
EXPECTED_STAGE9_CONFIGURATION_ROWS = 346
EXPECTED_ADDITIONAL_COUNTED_TRIALS = 40
EXPECTED_RAW_TRIAL_COUNT = 386

_REQUIRED_STAGE9_COLUMNS = frozenset(
    {
        "model_family",
        "config_id",
        "gate_pass",
        "trial_phase",
        "configuration_json",
    }
)


class TrialLedgerError(RuntimeError):
    """Raised when the frozen trial population is incomplete or ambiguous."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrialLedgerError(message)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _row(
    *,
    trial_id: str,
    category: str,
    candidate_id: str,
    stage_registered: int,
    status: str,
    result_available: bool,
    pnl_available: bool,
    counts_in_raw_dsr: bool,
    inclusion_reason: str,
    source_receipt_sha256: str = METHODOLOGY_SHA256,
    source_build_id: str = "methodology-v1",
    seed_aggregation: str = "not_applicable",
    duplicate_of_trial_id: str | None = None,
) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "category": category,
        "candidate_id": candidate_id,
        "stage_registered": stage_registered,
        "source_build_id": source_build_id,
        "status": status,
        "seed_aggregation": seed_aggregation,
        "result_available": result_available,
        "pnl_available": pnl_available,
        "counts_in_raw_dsr": counts_in_raw_dsr,
        "inclusion_reason": inclusion_reason,
        "source_receipt_sha256": source_receipt_sha256,
        "duplicate_of_trial_id": duplicate_of_trial_id,
    }


def _fixed_counted_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        label = format(weight, ".2f")
        rows.append(
            _row(
                trial_id=f"ensemble:nn_weight:{label}",
                category="ensemble_weight",
                candidate_id=label,
                stage_registered=11,
                status="selected" if weight == 0.5 else "evaluated_not_selected",
                result_available=True,
                pnl_available=False,
                counts_in_raw_dsr=True,
                inclusion_reason="prespecified_validation_ensemble_weight",
            )
        )
    for candidate_id, status in (
        ("hybrid_alpha_risk_cost_convex", "primary"),
        ("rank_volatility_projected", "challenger"),
        ("decision_focused_neural_projected", "inactive_runtime_not_certified"),
    ):
        rows.append(
            _row(
                trial_id=f"allocator:{candidate_id}",
                category="allocation_architecture",
                candidate_id=candidate_id,
                stage_registered=10,
                status=status,
                result_available=candidate_id != "decision_focused_neural_projected",
                pnl_available=candidate_id != "decision_focused_neural_projected",
                counts_in_raw_dsr=True,
                inclusion_reason="prespecified_allocation_architecture",
            )
        )
    for candidate_id in (
        "trailing_residual_mean_markowitz",
        "fixed_value_quality_momentum",
    ):
        rows.append(
            _row(
                trial_id=f"comparator:{candidate_id}",
                category="strategy_comparator",
                candidate_id=candidate_id,
                stage_registered=12,
                status="registered_before_final_unblind",
                result_available=False,
                pnl_available=False,
                counts_in_raw_dsr=True,
                inclusion_reason="economically_distinct_prespecified_strategy_comparator",
            )
        )
    for category, candidates in (
        (
            "target_or_loss",
            (
                "target_excess_return_rank",
                "target_huber_standardized_residual_magnitude",
                "loss_pairwise_ranking",
            ),
        ),
        (
            "feature_specification",
            (
                "features_core_only",
                "features_options_ablation",
                "features_pastor_stambaugh_ablation",
                "features_market_price_liquidity_core",
            ),
        ),
        ("ensemble_rule", ("ensemble_dynamic_trailing_ic",)),
        ("frozen_model_control", ("frozen_december_2019_model",)),
        (
            "risk_or_covariance",
            ("risk_504_session", "covariance_ledoit_wolf"),
        ),
        (
            "cost_or_financing",
            (
                "cost_standard",
                "impact_severe_0.50",
                "borrow_stress",
                "financing_stress",
            ),
        ),
        ("execution_timing", ("execution_delay_one_additional_day",)),
        (
            "volatility_target",
            ("volatility_target_0.08", "volatility_target_0.12"),
        ),
        ("reference_nav", ("nav_25_million", "nav_500_million")),
        (
            "delisting_assumption",
            ("missing_performance_delisting_minus_100_percent",),
        ),
    ):
        for candidate_id in candidates:
            rows.append(
                _row(
                    trial_id=f"robustness:{candidate_id}",
                    category=category,
                    candidate_id=candidate_id,
                    stage_registered=12,
                    status="registered_before_final_unblind",
                    result_available=False,
                    pnl_available=False,
                    counts_in_raw_dsr=True,
                    inclusion_reason="prespecified_one_at_a_time_robustness",
                )
            )
    for cadence in ("monthly", "quarterly", "annual"):
        for memory in (
            "weighted_expanding_120_month_half_life",
            "unweighted_expanding",
            "trailing_180_months",
        ):
            candidate_id = f"{cadence}__{memory}"
            rows.append(
                _row(
                    trial_id=f"refit_memory:{candidate_id}",
                    category="refit_memory_cell",
                    candidate_id=candidate_id,
                    stage_registered=12,
                    status=(
                        "primary"
                        if candidate_id
                        == "quarterly__weighted_expanding_120_month_half_life"
                        else "registered_before_final_unblind"
                    ),
                    result_available=False,
                    pnl_available=False,
                    counts_in_raw_dsr=True,
                    inclusion_reason="prespecified_three_by_three_refit_memory_grid",
                )
            )
    _require(
        len(rows) == EXPECTED_ADDITIONAL_COUNTED_TRIALS,
        "fixed counted trial population differs",
    )
    return rows


def _excluded_rows() -> list[dict[str, Any]]:
    return [
        _row(
            trial_id=f"descriptive:{candidate}",
            category="descriptive_benchmark",
            candidate_id=candidate,
            stage_registered=12,
            status="registered_descriptive_only",
            result_available=False,
            pnl_available=False,
            counts_in_raw_dsr=False,
            inclusion_reason=reason,
        )
        for candidate, reason in (
            ("cash_or_one_month_tbill", "risk_free_series_excluded_by_methodology"),
            ("spy_total_return", "descriptive_index_excluded_by_methodology"),
            ("qqq_total_return", "descriptive_index_excluded_by_methodology"),
            (
                "equal_weight_long_only_scoring_universe",
                "descriptive_long_only_benchmark_not_a_research_choice",
            ),
            (
                "regime_and_subperiod_tables",
                "reporting_only_slices_excluded_by_methodology",
            ),
        )
    ] + [
        _row(
            trial_id="duplicate:robustness:allocator_rank",
            category="duplicate_registry_reference",
            candidate_id="allocator_rank",
            stage_registered=12,
            status="duplicate_not_counted",
            result_available=False,
            pnl_available=False,
            counts_in_raw_dsr=False,
            inclusion_reason="already_counted_as_allocation_architecture",
            duplicate_of_trial_id="allocator:rank_volatility_projected",
        ),
        _row(
            trial_id="duplicate:robustness:allocator_decision_focused",
            category="duplicate_registry_reference",
            candidate_id="allocator_decision_focused",
            stage_registered=12,
            status="duplicate_not_counted",
            result_available=False,
            pnl_available=False,
            counts_in_raw_dsr=False,
            inclusion_reason="already_counted_as_allocation_architecture",
            duplicate_of_trial_id="allocator:decision_focused_neural_projected",
        ),
    ]


@dataclass(frozen=True, slots=True)
class FrozenTrialLedger:
    rows: tuple[Mapping[str, Any], ...]
    raw_trial_count: int
    excluded_row_count: int
    ledger_sha256: str
    source_table_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "ledger_id": "stage12-complete-frozen-trial-ledger-v1",
            "raw_trial_count": self.raw_trial_count,
            "excluded_row_count": self.excluded_row_count,
            "total_ledger_rows": len(self.rows),
            "source_table_sha256": self.source_table_sha256,
            "ledger_sha256": self.ledger_sha256,
            "rows": [dict(row) for row in self.rows],
        }


def build_frozen_trial_ledger(
    configuration_table: Path,
    *,
    expected_source_sha256: str = STAGE9_CONFIGURATION_SHA256,
    expected_source_rows: int = EXPECTED_STAGE9_CONFIGURATION_ROWS,
) -> FrozenTrialLedger:
    """Authenticate Stage 9 trials and append the closed pre-final registry."""

    path = Path(configuration_table)
    _require(path.is_file() and not path.is_symlink(), "unsafe trial source")
    source_file = stable_file(path, keep_payload=True)
    source_sha = source_file.sha256
    _require(
        source_sha == expected_source_sha256,
        "Stage 9 configuration table SHA-256 differs",
    )
    assert source_file.payload is not None
    table = pq.read_table(pa.BufferReader(source_file.payload))
    _require(
        _REQUIRED_STAGE9_COLUMNS.issubset(table.column_names),
        "Stage 9 configuration columns are incomplete",
    )
    source = table.select(sorted(_REQUIRED_STAGE9_COLUMNS)).to_pylist()
    _require(len(source) == expected_source_rows, "Stage 9 trial row count differs")
    stage9_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in source:
        family = str(item["model_family"])
        config_id = str(item["config_id"])
        key = (family, config_id)
        _require(key not in seen, "duplicate Stage 9 family/configuration trial")
        seen.add(key)
        configuration = str(item["configuration_json"])
        try:
            decoded = json.loads(configuration)
        except json.JSONDecodeError as exc:
            raise TrialLedgerError("invalid Stage 9 configuration JSON") from exc
        _require(
            isinstance(decoded, dict) and decoded.get("config_id") == config_id,
            "Stage 9 configuration identity differs",
        )
        stage9_rows.append(
            _row(
                trial_id=f"selector:{family}:{config_id}",
                category="selector_configuration",
                candidate_id=f"{family}:{config_id}",
                stage_registered=9,
                source_build_id=STAGE9_SELECTOR_BUILD_ID,
                status=(
                    "development_gate_passed"
                    if bool(item["gate_pass"])
                    else "development_gate_failed"
                ),
                seed_aggregation=(
                    "mean_raw_forecasts_seeds_11_23_47"
                    if family in {"neural_network", "xgboost"}
                    else "deterministic"
                ),
                result_available=True,
                pnl_available=False,
                counts_in_raw_dsr=True,
                inclusion_reason=(
                    "economically_distinct_development_configuration_after_required_seed_aggregation"
                ),
                source_receipt_sha256=source_sha,
            )
        )
    all_rows = stage9_rows + _fixed_counted_rows() + _excluded_rows()
    all_rows.sort(key=lambda value: str(value["trial_id"]))
    trial_ids = [str(row["trial_id"]) for row in all_rows]
    _require(len(trial_ids) == len(set(trial_ids)), "trial IDs are not unique")
    ids = set(trial_ids)
    for row in all_rows:
        duplicate = row["duplicate_of_trial_id"]
        _require(
            duplicate is None or duplicate in ids,
            "duplicate trial reference is unresolved",
        )
    raw_count = sum(bool(row["counts_in_raw_dsr"]) for row in all_rows)
    _require(
        raw_count == expected_source_rows + EXPECTED_ADDITIONAL_COUNTED_TRIALS,
        "raw DSR trial count does not reconcile",
    )
    if expected_source_rows == EXPECTED_STAGE9_CONFIGURATION_ROWS:
        _require(
            raw_count == EXPECTED_RAW_TRIAL_COUNT,
            "production raw DSR trial count differs",
        )
    excluded = len(all_rows) - raw_count
    ledger_sha = hashlib.sha256(_canonical_json(all_rows)).hexdigest()
    return FrozenTrialLedger(
        rows=tuple(all_rows),
        raw_trial_count=raw_count,
        excluded_row_count=excluded,
        ledger_sha256=ledger_sha,
        source_table_sha256=source_sha,
    )


def canonical_trial_ledger_bytes(ledger: FrozenTrialLedger) -> bytes:
    return _canonical_json(ledger.to_payload())


__all__ = [
    "EXPECTED_ADDITIONAL_COUNTED_TRIALS",
    "EXPECTED_RAW_TRIAL_COUNT",
    "EXPECTED_STAGE9_CONFIGURATION_ROWS",
    "FrozenTrialLedger",
    "TrialLedgerError",
    "build_frozen_trial_ledger",
    "canonical_trial_ledger_bytes",
]
