"""Deterministic empirical Stage 13 result-document assembly.

The Stage 13 production runner retains one immutable fragment per completed
month.  This module joins those fragments into the exact thirteen economic
documents accepted by the frozen publisher, recomputes every derived metric,
binds the frozen trial ledger and Stage 14 handoff, and performs the
publisher's full semantic replay before returning.

Nothing in this module opens a payload or accepts a path, reader, schema, or
execution callback.  Final-period access remains the responsibility of the
authorization-first production runner.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from portfolio_bot.final_test.calendar import (
    CORE_COMPLETE_HOLDING_MONTHS,
    PARTIAL_DECEMBER_HOLDING_MONTH,
    PRIMARY_FULL_HOLDING_MONTHS,
)
from portfolio_bot.final_test.inference import (
    FROZEN_JOINT_OUTER_INDICES_SHA256,
)
from portfolio_bot.final_test.orchestration import (
    FROZEN_SEGMENT_PLANS,
    Stage13RunResult,
)
from portfolio_bot.final_test.results import (
    ACCESS_LEDGER_FILE,
    ACTIVE_STRATEGY_IDS,
    ALLOCATION_EVIDENCE_FILE,
    COST_CONSTRAINT_MATERIALITY_FILE,
    DAILY_ACCOUNTING_FILE,
    DESCRIPTIVE_BENCHMARK_IDS,
    EQUAL_WEIGHT_STRATEGY_ID,
    FACTOR_BENCHMARK_FILE,
    FACTOR_NAMES,
    FROZEN_ROBUSTNESS_REGISTRY,
    FROZEN_ROBUSTNESS_REQUESTS,
    INFERENCE_COMPARISON_STRATEGY_IDS,
    INFERENCE_INPUTS_FILE,
    INFERENCE_RESULTS_FILE,
    METRICS_FILE,
    MODEL_EVIDENCE_FILE,
    MONTHLY_ACCOUNTING_FILE,
    PRIMARY_STRATEGY_ID,
    REPORT_INPUTS_FILE,
    ROBUSTNESS_FILE,
    TRIAL_LEDGER_EVIDENCE_FILE,
    Stage13ResultError,
    Stage13ResultSet,
    assemble_stage13_result_set,
    canonical_json_bytes,
    compute_required_metrics,
    sha256_json,
)
from portfolio_bot.final_test.terminal_event_sensitivity import (
    StrategyMonthlyReturn,
    TerminalEventSensitivityError,
    TerminalEventSource,
    TerminalEventStrategyObservation,
    evaluate_terminal_event_sensitivities,
)


PRODUCTION_RESULT_ASSEMBLER_ID = "stage13-empirical-result-assembler-v3"
STAGE14_PENDING_STATUS = "PENDING_STAGE14_EXECUTION"
STAGE14_HANDOFF_SCOPE = "verified_stage13_result_required_before_stage14_execution"
MONTH_FRAGMENT_FILES = (
    MODEL_EVIDENCE_FILE,
    ALLOCATION_EVIDENCE_FILE,
    DAILY_ACCOUNTING_FILE,
    MONTHLY_ACCOUNTING_FILE,
    COST_CONSTRAINT_MATERIALITY_FILE,
    FACTOR_BENCHMARK_FILE,
    ACCESS_LEDGER_FILE,
)

_HEX64 = frozenset("0123456789abcdef")
_FRAGMENT_FIELDS = MappingProxyType(
    {
        MODEL_EVIDENCE_FILE: frozenset({"records"}),
        ALLOCATION_EVIDENCE_FILE: frozenset({"candidate_audits", "records"}),
        DAILY_ACCOUNTING_FILE: frozenset({"records"}),
        MONTHLY_ACCOUNTING_FILE: frozenset({"records"}),
        COST_CONSTRAINT_MATERIALITY_FILE: frozenset(
            {
                "records",
                "terminal_event_sources",
                "terminal_event_strategy_observations",
            }
        ),
        FACTOR_BENCHMARK_FILE: frozenset({"factor_rows", "benchmark_rows"}),
        ACCESS_LEDGER_FILE: frozenset({"records"}),
    }
)
_TERMINAL_EVENT_SOURCE_FIELDS = frozenset(
    {
        "terminal_action_id",
        "permno",
        "event_date",
        "holding_month",
        "source_record_sha256",
        "terminal_policy_sha256",
        "terminal_evidence_sha256",
        "event_day_quoted_full_spread",
        "frozen_full_spread_20d",
        "selected_full_spread_stress_rate",
        "remaining_cash_growth",
        "segment",
        "checkpoint_sha256",
    }
)
_TERMINAL_EVENT_OBSERVATION_FIELDS = frozenset(
    {
        "terminal_action_id",
        "strategy_id",
        "position_held",
        "target_weight",
        "month_start_nav_dollars",
        "event_nav_dollars",
        "signed_event_value_dollars",
        "event_day_portfolio_weight",
        "holding_month",
        "segment",
        "checkpoint_sha256",
    }
)
_ELASTIC_NET_STRATEGY_ID = "elastic_net_hybrid_challenger"
_SENSITIVITY_FINAL_CONCLUSION_STATUS = "pending_stage14_inference_and_stage15_synthesis"


class ProductionResultAssemblyError(Stage13ResultError):
    """Raised when retained empirical fragments are incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionResultAssemblyError(message)


def _require_sha256(value: object, *, field: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and set(value).issubset(_HEX64),
        f"{field} must be an exact lowercase SHA-256",
    )
    return value


def _canonical_copy(value: object, *, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (Stage13ResultError, json.JSONDecodeError) as exc:
        raise ProductionResultAssemblyError(
            f"{label} is not finite canonical JSON"
        ) from exc


def _rows(value: object, *, label: str) -> tuple[dict[str, Any], ...]:
    _require(isinstance(value, list) and bool(value), f"{label} rows are empty")
    result: list[dict[str, Any]] = []
    for row in value:
        _require(isinstance(row, Mapping), f"{label} row is not an object")
        result.append(_canonical_copy(dict(row), label=f"{label} row"))
    return tuple(result)


def _optional_rows(value: object, *, label: str) -> tuple[dict[str, Any], ...]:
    _require(isinstance(value, list), f"{label} rows are not a list")
    result: list[dict[str, Any]] = []
    for row in value:
        _require(isinstance(row, Mapping), f"{label} row is not an object")
        result.append(_canonical_copy(dict(row), label=f"{label} row"))
    return tuple(result)


def _row_month_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    segment: str,
    holding_month: str,
    checkpoint_sha256: str,
    label: str,
    checkpoint_required: bool = True,
) -> None:
    for row in rows:
        _require(
            row.get("segment") == segment and row.get("holding_month") == holding_month,
            f"{label} escaped its retained strategy-month",
        )
        if checkpoint_required:
            _require(
                row.get("checkpoint_sha256") == checkpoint_sha256,
                f"{label} checkpoint binding differs",
            )


@dataclass(frozen=True, slots=True)
class EmpiricalMonthResultFragment:
    """One content-addressed month of publication-grade empirical evidence."""

    segment: str
    holding_month: str
    checkpoint_sha256: str
    documents: Mapping[str, Mapping[str, Any]]
    fragment_sha256: str

    def __post_init__(self) -> None:
        expected_segments = {
            plan.segment.value: plan.holding_months for plan in FROZEN_SEGMENT_PLANS
        }
        _require(
            self.segment in expected_segments
            and self.holding_month in expected_segments[self.segment],
            "month fragment is outside the frozen final chronology",
        )
        _require_sha256(
            self.checkpoint_sha256, field="month fragment checkpoint_sha256"
        )
        _require(
            isinstance(self.documents, Mapping)
            and set(self.documents) == set(MONTH_FRAGMENT_FILES),
            "month fragment member set differs",
        )
        normalized: dict[str, Mapping[str, Any]] = {}
        for name in MONTH_FRAGMENT_FILES:
            raw = self.documents[name]
            _require(isinstance(raw, Mapping), f"{name} fragment is not an object")
            _require(
                set(raw) == set(_FRAGMENT_FIELDS[name]),
                f"{name} fragment fields differ",
            )
            document = _canonical_copy(dict(raw), label=f"{name} fragment")
            normalized[name] = MappingProxyType(document)

        model = _rows(
            normalized[MODEL_EVIDENCE_FILE]["records"],
            label="model fragment",
        )
        allocation = _rows(
            normalized[ALLOCATION_EVIDENCE_FILE]["records"],
            label="allocation fragment",
        )
        candidates = _rows(
            normalized[ALLOCATION_EVIDENCE_FILE]["candidate_audits"],
            label="candidate fragment",
        )
        daily = _rows(
            normalized[DAILY_ACCOUNTING_FILE]["records"],
            label="daily fragment",
        )
        monthly = _rows(
            normalized[MONTHLY_ACCOUNTING_FILE]["records"],
            label="monthly fragment",
        )
        audit = _rows(
            normalized[COST_CONSTRAINT_MATERIALITY_FILE]["records"],
            label="cost fragment",
        )
        terminal_sources = _optional_rows(
            normalized[COST_CONSTRAINT_MATERIALITY_FILE]["terminal_event_sources"],
            label="terminal event source fragment",
        )
        terminal_observations = _optional_rows(
            normalized[COST_CONSTRAINT_MATERIALITY_FILE][
                "terminal_event_strategy_observations"
            ],
            label="terminal event observation fragment",
        )
        factors = _rows(
            normalized[FACTOR_BENCHMARK_FILE]["factor_rows"],
            label="factor fragment",
        )
        benchmarks = _rows(
            normalized[FACTOR_BENCHMARK_FILE]["benchmark_rows"],
            label="benchmark fragment",
        )
        access = _rows(
            normalized[ACCESS_LEDGER_FILE]["records"],
            label="access fragment",
        )
        for label, rows, checkpoint_required in (
            ("model fragment", model, True),
            ("allocation fragment", allocation, True),
            ("candidate fragment", candidates, True),
            ("daily fragment", daily, True),
            ("monthly fragment", monthly, True),
            ("cost fragment", audit, True),
            ("terminal event source fragment", terminal_sources, True),
            (
                "terminal event observation fragment",
                terminal_observations,
                True,
            ),
            ("factor fragment", factors, True),
            ("benchmark fragment", benchmarks, False),
            ("access fragment", access, True),
        ):
            _row_month_identity(
                rows,
                segment=self.segment,
                holding_month=self.holding_month,
                checkpoint_sha256=self.checkpoint_sha256,
                label=label,
                checkpoint_required=checkpoint_required,
            )
        _require(
            len(factors) == 1,
            "month fragment must contain exactly one factor row",
        )
        _require_sha256(self.fragment_sha256, field="fragment_sha256")
        object.__setattr__(self, "documents", MappingProxyType(normalized))
        _require(
            self.fragment_sha256 == sha256_json(self.payload(False)),
            "month fragment digest differs",
        )

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "assembler_id": PRODUCTION_RESULT_ASSEMBLER_ID,
            "segment": self.segment,
            "holding_month": self.holding_month,
            "checkpoint_sha256": self.checkpoint_sha256,
            "documents": {
                name: dict(self.documents[name]) for name in MONTH_FRAGMENT_FILES
            },
        }
        if include_digest:
            result["fragment_sha256"] = self.fragment_sha256
        return result

    @classmethod
    def create(
        cls,
        *,
        segment: str,
        holding_month: str,
        checkpoint_sha256: str,
        documents: Mapping[str, Mapping[str, Any]],
    ) -> "EmpiricalMonthResultFragment":
        provisional = cls.__new__(cls)
        for name, value in (
            ("segment", segment),
            ("holding_month", holding_month),
            ("checkpoint_sha256", checkpoint_sha256),
            ("documents", documents),
            ("fragment_sha256", ""),
        ):
            object.__setattr__(provisional, name, value)
        digest = sha256_json(provisional.payload(False))
        return cls(
            segment=segment,
            holding_month=holding_month,
            checkpoint_sha256=checkpoint_sha256,
            documents=documents,
            fragment_sha256=digest,
        )


@dataclass(frozen=True, slots=True)
class EmpiricalAuxiliaryEvidence:
    """Frozen Stage 12 trial ledger needed by the Stage 13 handoff."""

    stage12_trial_ledger: Mapping[str, Any]
    evidence_sha256: str

    def __post_init__(self) -> None:
        ledger = _canonical_copy(
            dict(self.stage12_trial_ledger), label="Stage 12 trial ledger"
        )
        expected_ledger_fields = {
            "schema_version",
            "ledger_id",
            "raw_trial_count",
            "excluded_row_count",
            "total_ledger_rows",
            "source_table_sha256",
            "ledger_sha256",
            "rows",
        }
        _require(
            set(ledger) == expected_ledger_fields
            and ledger["schema_version"] == 1
            and ledger["ledger_id"] == "stage12-complete-frozen-trial-ledger-v1"
            and ledger["raw_trial_count"] == 386
            and isinstance(ledger["rows"], list)
            and ledger["total_ledger_rows"] == len(ledger["rows"]),
            "Stage 12 trial ledger contract differs",
        )
        _require_sha256(
            ledger["source_table_sha256"], field="trial source_table_sha256"
        )
        _require_sha256(ledger["ledger_sha256"], field="trial ledger_sha256")
        _require(
            ledger["ledger_sha256"] == sha256_json(ledger["rows"])
            and ledger["excluded_row_count"]
            == len(ledger["rows"]) - ledger["raw_trial_count"],
            "Stage 12 trial ledger does not reconcile",
        )

        _require_sha256(self.evidence_sha256, field="auxiliary evidence_sha256")
        object.__setattr__(self, "stage12_trial_ledger", MappingProxyType(ledger))
        _require(
            self.evidence_sha256 == sha256_json(self.payload(False)),
            "auxiliary evidence digest differs",
        )

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "assembler_id": PRODUCTION_RESULT_ASSEMBLER_ID,
            "stage12_trial_ledger": dict(self.stage12_trial_ledger),
        }
        if include_digest:
            result["evidence_sha256"] = self.evidence_sha256
        return result

    @classmethod
    def create(
        cls,
        *,
        stage12_trial_ledger: Mapping[str, Any],
    ) -> "EmpiricalAuxiliaryEvidence":
        provisional = cls.__new__(cls)
        for name, value in (
            ("stage12_trial_ledger", stage12_trial_ledger),
            ("evidence_sha256", ""),
        ):
            object.__setattr__(provisional, name, value)
        digest = sha256_json(provisional.payload(False))
        return cls(
            stage12_trial_ledger=stage12_trial_ledger,
            evidence_sha256=digest,
        )


def _checkpoint_lookup(run: Stage13RunResult) -> dict[tuple[str, str], str]:
    return {
        (result.plan.segment.value, record.holding_month): record.checkpoint_sha256
        for result in run.segments
        for record in result.chain_records
    }


def _expected_month_keys() -> tuple[tuple[str, str], ...]:
    return tuple(
        (plan.segment.value, month)
        for plan in FROZEN_SEGMENT_PLANS
        for month in plan.holding_months
    )


def _extend(
    destination: list[dict[str, Any]],
    fragment: EmpiricalMonthResultFragment,
    member: str,
    field: str,
) -> None:
    destination.extend(
        _canonical_copy(
            fragment.documents[member][field],
            label=f"{member}.{field}",
        )
    )


def _parse_terminal_event_source(
    row: Mapping[str, Any],
) -> TerminalEventSource:
    _require(
        set(row) == set(_TERMINAL_EVENT_SOURCE_FIELDS),
        "terminal event source fields differ",
    )
    _require(
        row["segment"] == "primary_full",
        "terminal event source escaped the primary segment",
    )
    try:
        raw_event_date = row["event_date"]
        _require(
            isinstance(raw_event_date, str),
            "terminal event source event_date must be ISO text",
        )
        event_date = dt.date.fromisoformat(raw_event_date)
        _require(
            event_date.isoformat() == raw_event_date,
            "terminal event source event_date is not canonical ISO text",
        )
        source = TerminalEventSource(
            terminal_action_id=row["terminal_action_id"],
            permno=row["permno"],
            event_date=event_date,
            holding_month=row["holding_month"],
            source_record_sha256=row["source_record_sha256"],
            terminal_policy_sha256=row["terminal_policy_sha256"],
            terminal_evidence_sha256=row["terminal_evidence_sha256"],
            event_day_quoted_full_spread=row["event_day_quoted_full_spread"],
            frozen_full_spread_20d=row["frozen_full_spread_20d"],
            remaining_cash_growth=row["remaining_cash_growth"],
        )
    except (KeyError, TypeError, ValueError, TerminalEventSensitivityError) as exc:
        if isinstance(exc, ProductionResultAssemblyError):
            raise
        raise ProductionResultAssemblyError(
            "terminal event source payload is invalid"
        ) from exc
    expected = {
        **source.as_dict(),
        "segment": row["segment"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    _require(
        dict(row) == expected,
        "terminal event source derived evidence differs",
    )
    return source


def _parse_terminal_event_observation(
    row: Mapping[str, Any],
) -> TerminalEventStrategyObservation:
    _require(
        set(row) == set(_TERMINAL_EVENT_OBSERVATION_FIELDS),
        "terminal event observation fields differ",
    )
    _require(
        row["segment"] == "primary_full",
        "terminal event observation escaped the primary segment",
    )
    try:
        observation = TerminalEventStrategyObservation(
            terminal_action_id=row["terminal_action_id"],
            strategy_id=row["strategy_id"],
            position_held=row["position_held"],
            target_weight=row["target_weight"],
            month_start_nav_dollars=row["month_start_nav_dollars"],
            event_nav_dollars=row["event_nav_dollars"],
            signed_event_value_dollars=row["signed_event_value_dollars"],
        )
    except (KeyError, TypeError, ValueError, TerminalEventSensitivityError) as exc:
        raise ProductionResultAssemblyError(
            "terminal event observation payload is invalid"
        ) from exc
    expected = {
        **observation.as_dict(),
        "holding_month": row["holding_month"],
        "segment": row["segment"],
        "checkpoint_sha256": row["checkpoint_sha256"],
    }
    _require(
        dict(row) == expected,
        "terminal event observation derived evidence differs",
    )
    return observation


def _terminal_event_evidence(
    source_rows: Sequence[Mapping[str, Any]],
    observation_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    tuple[TerminalEventSource, ...],
    tuple[TerminalEventStrategyObservation, ...],
]:
    source_by_action: dict[str, tuple[dict[str, Any], TerminalEventSource]] = {}
    for raw in source_rows:
        row = _canonical_copy(dict(raw), label="terminal event source")
        parsed = _parse_terminal_event_source(row)
        existing = source_by_action.get(parsed.terminal_action_id)
        if existing is not None:
            _require(
                existing[0] == row,
                "duplicate terminal event source evidence differs",
            )
            continue
        source_by_action[parsed.terminal_action_id] = (row, parsed)

    ordered_source_pairs = sorted(
        source_by_action.values(),
        key=lambda value: (
            value[1].event_date,
            value[1].terminal_action_id,
        ),
    )
    _require(
        len(ordered_source_pairs) == 2,
        "terminal event source set must contain exactly two events",
    )
    ordered_source_rows = [value[0] for value in ordered_source_pairs]
    ordered_sources = tuple(value[1] for value in ordered_source_pairs)

    observation_by_key: dict[
        tuple[str, str], tuple[dict[str, Any], TerminalEventStrategyObservation]
    ] = {}
    source_month_by_action = {
        source.terminal_action_id: source.holding_month for source in ordered_sources
    }
    for raw in observation_rows:
        row = _canonical_copy(dict(raw), label="terminal event observation")
        parsed = _parse_terminal_event_observation(row)
        key = (parsed.terminal_action_id, parsed.strategy_id)
        _require(
            key not in observation_by_key,
            "terminal event observations contain a duplicate action-strategy",
        )
        _require(
            parsed.terminal_action_id in source_month_by_action
            and row["holding_month"]
            == source_month_by_action[parsed.terminal_action_id],
            "terminal event observation source-month identity differs",
        )
        observation_by_key[key] = (row, parsed)

    expected_observation_keys = {
        (source.terminal_action_id, strategy_id)
        for source in ordered_sources
        for strategy_id in ACTIVE_STRATEGY_IDS
    }
    _require(
        len(observation_by_key) == 12
        and set(observation_by_key) == expected_observation_keys,
        "terminal event observations must contain the exact two-by-six grid",
    )
    ordered_observation_pairs = tuple(
        observation_by_key[(source.terminal_action_id, strategy_id)]
        for source in ordered_sources
        for strategy_id in ACTIVE_STRATEGY_IDS
    )
    return (
        ordered_source_rows,
        [value[0] for value in ordered_observation_pairs],
        ordered_sources,
        tuple(value[1] for value in ordered_observation_pairs),
    )


def _metric_sign(value: object) -> int:
    numeric = float(value)
    return 1 if numeric > 0.0 else -1 if numeric < 0.0 else 0


def _terminal_sensitivity_document(
    monthly_records: Sequence[Mapping[str, Any]],
    terminal_event_source_rows: Sequence[Mapping[str, Any]],
    terminal_event_observation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, object], list[dict[str, Any]], list[dict[str, Any]]]:
    (
        sources_for_publication,
        observations_for_publication,
        event_sources,
        observations,
    ) = _terminal_event_evidence(
        terminal_event_source_rows,
        terminal_event_observation_rows,
    )

    expected_primary_keys = {
        (month, strategy_id)
        for month in PRIMARY_FULL_HOLDING_MONTHS
        for strategy_id in ACTIVE_STRATEGY_IDS
    }
    primary_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in monthly_records:
        if row.get("segment") != "primary_full":
            continue
        key = (str(row.get("holding_month")), str(row.get("strategy_id")))
        _require(
            key not in primary_rows,
            "primary sensitivity input contains a duplicate strategy-month",
        )
        primary_rows[key] = row
    _require(
        len(primary_rows) == len(PRIMARY_FULL_HOLDING_MONTHS) * len(ACTIVE_STRATEGY_IDS)
        and set(primary_rows) == expected_primary_keys,
        "primary sensitivity input is not the exact 68-by-six panel",
    )
    monthly_inputs = tuple(
        StrategyMonthlyReturn(
            strategy_id=strategy_id,
            holding_month=month,
            net_return=primary_rows[(month, strategy_id)]["net_return"],
            excess_return=primary_rows[(month, strategy_id)]["excess_return"],
        )
        for month in PRIMARY_FULL_HOLDING_MONTHS
        for strategy_id in ACTIVE_STRATEGY_IDS
    )
    try:
        evaluation = evaluate_terminal_event_sensitivities(
            monthly_inputs,
            event_sources,
            observations,
        ).as_dict()
    except (KeyError, TypeError, ValueError, TerminalEventSensitivityError) as exc:
        raise ProductionResultAssemblyError(
            "terminal event sensitivity evaluation failed"
        ) from exc

    baseline_by_strategy = {
        str(row["strategy_id"]): row for row in evaluation["baseline_strategy_results"]
    }
    _require(
        set(baseline_by_strategy) == set(ACTIVE_STRATEGY_IDS),
        "terminal sensitivity baseline strategy set differs",
    )
    baseline_primary = baseline_by_strategy[PRIMARY_STRATEGY_ID]
    baseline_elastic = baseline_by_strategy[_ELASTIC_NET_STRATEGY_ID]
    conclusion_effects: list[dict[str, object]] = []
    scenarios = evaluation["scenarios"]
    _require(
        isinstance(scenarios, list) and len(scenarios) == 4,
        "terminal sensitivity scenario set differs",
    )
    for scenario in scenarios:
        scenario_by_strategy = {
            str(row["strategy_id"]): row for row in scenario["strategy_results"]
        }
        _require(
            set(scenario_by_strategy) == set(ACTIVE_STRATEGY_IDS),
            "terminal sensitivity scenario strategy set differs",
        )
        scenario_primary = scenario_by_strategy[PRIMARY_STRATEGY_ID]
        scenario_elastic = scenario_by_strategy[_ELASTIC_NET_STRATEGY_ID]
        conclusion_effects.append(
            {
                "scenario_id": scenario["scenario_id"],
                "primary_cagr_sign_changed": _metric_sign(
                    scenario_primary["metrics"]["net_cagr"]
                )
                != _metric_sign(baseline_primary["metrics"]["net_cagr"]),
                "primary_sharpe_sign_changed": _metric_sign(
                    scenario_primary["metrics"]["annualized_excess_sharpe"]
                )
                != _metric_sign(
                    baseline_primary["metrics"]["annualized_excess_sharpe"]
                ),
                "primary_vs_elastic_net_ordering_changed": (
                    int(scenario_primary["rank"]) < int(scenario_elastic["rank"])
                )
                != (int(baseline_primary["rank"]) < int(baseline_elastic["rank"])),
                "primary_rank_changed": int(scenario_primary["rank"])
                != int(baseline_primary["rank"]),
            }
        )

    sensitivity_without_receipt: dict[str, object] = {
        "sensitivity_id": evaluation["sensitivity_id"],
        "status": "COMPLETE_REPORTING_ONLY",
        "event_source_count": len(sources_for_publication),
        "strategy_observation_count": len(observations_for_publication),
        "event_sources_sha256": sha256_json(sources_for_publication),
        "strategy_observations_sha256": sha256_json(observations_for_publication),
        "one_at_a_time_only": evaluation["one_at_a_time_only"],
        "cartesian_product_executed": evaluation["cartesian_product_executed"],
        "primary_estimates_modified": evaluation["primary_estimates_modified"],
        "model_fitting_scoring_selection_or_allocation_rerun": False,
        "scenario_results_may_be_promoted": False,
        "baseline_strategy_results": evaluation["baseline_strategy_results"],
        "scenarios": scenarios,
        "conclusion_effects": conclusion_effects,
        "final_thesis_conclusion_status": (_SENSITIVITY_FINAL_CONCLUSION_STATUS),
    }
    sensitivity = {
        **sensitivity_without_receipt,
        "sensitivity_receipt_sha256": sha256_json(sensitivity_without_receipt),
    }
    return sensitivity, sources_for_publication, observations_for_publication


def _trial_ledger_document(
    auxiliary: EmpiricalAuxiliaryEvidence,
    *,
    stage12_binding_sha256: str,
) -> dict[str, object]:
    source = auxiliary.stage12_trial_ledger
    return {
        "schema_version": 1,
        "stage": 13,
        "synthetic": False,
        "ledger_id": source["ledger_id"],
        "stage12_binding_sha256": stage12_binding_sha256,
        "raw_trial_count": source["raw_trial_count"],
        "excluded_row_count": source["excluded_row_count"],
        "total_ledger_rows": source["total_ledger_rows"],
        "ledger_sha256": source["ledger_sha256"],
        "rows": source["rows"],
    }


def assemble_empirical_stage13_results(
    run: Stage13RunResult,
    fragments: Sequence[EmpiricalMonthResultFragment],
    auxiliary: EmpiricalAuxiliaryEvidence,
) -> Stage13ResultSet:
    """Assemble and fully replay all thirteen empirical result documents."""

    _require(
        isinstance(run, Stage13RunResult) and run.synthetic is False,
        "empirical result assembly requires an empirical Stage 13 run",
    )
    _require(
        isinstance(fragments, Sequence) and not isinstance(fragments, (str, bytes)),
        "empirical month fragments must be a sequence",
    )
    values = tuple(fragments)
    _require(
        all(isinstance(value, EmpiricalMonthResultFragment) for value in values),
        "empirical month fragment type differs",
    )
    checkpoint = _checkpoint_lookup(run)
    expected_keys = _expected_month_keys()
    by_key = {(value.segment, value.holding_month): value for value in values}
    _require(
        len(by_key) == len(values)
        and tuple(by_key) == expected_keys
        and set(checkpoint) == set(expected_keys),
        "empirical month fragment chronology differs",
    )
    for key, fragment in by_key.items():
        _require(
            fragment.checkpoint_sha256 == checkpoint[key],
            "empirical month fragment checkpoint differs",
        )

    model_records: list[dict[str, Any]] = []
    allocation_records: list[dict[str, Any]] = []
    candidate_audits: list[dict[str, Any]] = []
    daily_records: list[dict[str, Any]] = []
    monthly_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    terminal_event_source_rows: list[dict[str, Any]] = []
    terminal_event_observation_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    access_records: list[dict[str, Any]] = []
    for key in expected_keys:
        fragment = by_key[key]
        _extend(model_records, fragment, MODEL_EVIDENCE_FILE, "records")
        _extend(allocation_records, fragment, ALLOCATION_EVIDENCE_FILE, "records")
        _extend(
            candidate_audits,
            fragment,
            ALLOCATION_EVIDENCE_FILE,
            "candidate_audits",
        )
        _extend(daily_records, fragment, DAILY_ACCOUNTING_FILE, "records")
        _extend(monthly_records, fragment, MONTHLY_ACCOUNTING_FILE, "records")
        _extend(
            audit_records,
            fragment,
            COST_CONSTRAINT_MATERIALITY_FILE,
            "records",
        )
        _extend(
            terminal_event_source_rows,
            fragment,
            COST_CONSTRAINT_MATERIALITY_FILE,
            "terminal_event_sources",
        )
        _extend(
            terminal_event_observation_rows,
            fragment,
            COST_CONSTRAINT_MATERIALITY_FILE,
            "terminal_event_strategy_observations",
        )
        _extend(factor_rows, fragment, FACTOR_BENCHMARK_FILE, "factor_rows")
        _extend(benchmark_rows, fragment, FACTOR_BENCHMARK_FILE, "benchmark_rows")
        _extend(access_records, fragment, ACCESS_LEDGER_FILE, "records")

    (
        terminal_sensitivity,
        terminal_event_sources,
        terminal_event_strategy_observations,
    ) = _terminal_sensitivity_document(
        monthly_records,
        terminal_event_source_rows,
        terminal_event_observation_rows,
    )

    common = {"schema_version": 1, "stage": 13, "synthetic": False}
    model_doc = {
        **common,
        "chronology": "completed_labels_only",
        "records": model_records,
    }
    allocation_doc = {
        **common,
        "weight_unit": "fraction_of_start_nav",
        "problem_contract": (
            "frozen_stage12_problem_constraints_costs_and_inherited_state"
        ),
        "candidate_policy": {
            "minimum_new_names_each_side": 50,
            "tail_extension_rule": "scan_extreme_scores_until_minimum",
            "insufficient_side_rule": ("take_no_new_risk_and_reduce_existing_risk"),
            "candidate_sets_must_be_disjoint": True,
            "equal_weight_class": ("descriptive_costed_long_only_not_long_short"),
        },
        "candidate_audits": candidate_audits,
        "records": allocation_records,
    }
    daily_doc = {**common, "records": daily_records}
    monthly_doc = {
        **common,
        "segments_are_separate": True,
        "records": monthly_records,
    }
    audit_doc = {
        **common,
        "per_position_residue_rule": "strictly_less_than_0.05_usd",
        "aggregate_residue_rule": ("less_than_or_equal_to_1.00_usd_per_strategy_month"),
        "terminal_event_sources": terminal_event_sources,
        "terminal_event_strategy_observations": (terminal_event_strategy_observations),
        "records": audit_records,
    }
    factor_doc = {
        **common,
        "factor_names": list(FACTOR_NAMES),
        "benchmark_ids": list(DESCRIPTIVE_BENCHMARK_IDS),
        "active_costs_applied_to_passive_etfs_or_cash": False,
        "factor_rows": factor_rows,
        "benchmark_rows": benchmark_rows,
    }

    monthly_by_key = {
        (
            str(row["segment"]),
            str(row["holding_month"]),
            str(row["strategy_id"]),
        ): row
        for row in monthly_records
    }
    factor_by_key = {
        (str(row["segment"]), str(row["holding_month"])): row for row in factor_rows
    }
    model_ic: dict[tuple[str, str, str], float] = {}
    for row in model_records:
        key = (
            str(row["segment"]),
            str(row["holding_month"]),
            str(row["strategy_id"]),
        )
        value = float(row["monthly_spearman_ic"])
        if key in model_ic:
            _require(
                model_ic[key] == value,
                "monthly IC differs within an empirical strategy-month",
            )
        else:
            model_ic[key] = value

    segment_months = (
        ("primary_full", PRIMARY_FULL_HOLDING_MONTHS),
        ("core_complete", CORE_COMPLETE_HOLDING_MONTHS),
    )
    metric_rows: list[dict[str, object]] = []
    for segment, months in segment_months:
        source_factors = [factor_by_key[(segment, month)] for month in months]
        for strategy_id in ACTIVE_STRATEGY_IDS:
            source_monthly = [
                monthly_by_key[(segment, month, strategy_id)] for month in months
            ]
            metric_rows.append(
                {
                    "segment": segment,
                    "strategy_id": strategy_id,
                    "holding_months": list(months),
                    "metrics": compute_required_metrics(
                        source_monthly,
                        source_factors,
                        monthly_ic=[
                            model_ic[(segment, month, strategy_id)] for month in months
                        ],
                    ),
                }
            )
    partial_rows = [
        monthly_by_key[
            (
                "partial_december_descriptive",
                PARTIAL_DECEMBER_HOLDING_MONTH,
                strategy_id,
            )
        ]
        for strategy_id in ACTIVE_STRATEGY_IDS
    ]
    metrics_doc = {
        **common,
        "primary_confirmatory_segment": "primary_full",
        "core_reported_separately": True,
        "partial_descriptive_only": True,
        "metric_rows": metric_rows,
        "partial_descriptive_rows": partial_rows,
    }

    primary_metric = next(
        row
        for row in metric_rows
        if row["segment"] == "primary_full"
        and row["strategy_id"] == PRIMARY_STRATEGY_ID
    )
    primary_metric_sha256 = sha256_json(primary_metric)
    dispatch_rows: list[dict[str, object]] = []
    for ordinal, request in enumerate(FROZEN_ROBUSTNESS_REQUESTS, start=1):
        dispatch_rows.append(
            {
                "ordinal": ordinal,
                **request.to_mapping(),
                "result": {
                    "robustness_id": request.robustness_id,
                    "status": STAGE14_PENDING_STATUS,
                    "primary_reference_metrics_sha256": (primary_metric_sha256),
                    "result_metrics": {},
                    "promoted_to_primary": False,
                },
            }
        )
    robustness_doc = {
        **common,
        "one_at_a_time_only": True,
        "cartesian_product_executed": False,
        "execution_status": STAGE14_PENDING_STATUS,
        "frozen_registry": list(FROZEN_ROBUSTNESS_REGISTRY),
        "dispatch_rows": dispatch_rows,
        "post_unblind_terminal_liquidation_sensitivity": (terminal_sensitivity),
    }

    inference_rows: list[dict[str, object]] = []
    for month in PRIMARY_FULL_HOLDING_MONTHS:
        primary = monthly_by_key[("primary_full", month, PRIMARY_STRATEGY_ID)]
        factor = factor_by_key[("primary_full", month)]
        row: dict[str, object] = {
            "holding_month": month,
            "hybrid_total_return": primary["net_return"],
            "hybrid_excess_return": primary["excess_return"],
            "risk_free_return": factor["risk_free_return"],
            **{name: factor[name] for name in FACTOR_NAMES},
        }
        for strategy_id in INFERENCE_COMPARISON_STRATEGY_IDS:
            row[f"{strategy_id}_excess_return"] = monthly_by_key[
                ("primary_full", month, strategy_id)
            ]["excess_return"]
        row["equal_weight_long_only_scoring_universe_descriptive_excess_return"] = (
            monthly_by_key[("primary_full", month, EQUAL_WEIGHT_STRATEGY_ID)][
                "excess_return"
            ]
        )
        inference_rows.append(row)
    inference_inputs = {
        **common,
        "holding_months": list(PRIMARY_FULL_HOLDING_MONTHS),
        "joint_month_rows": True,
        "primary_core_partial_splicing_permitted": False,
        "rows": inference_rows,
    }
    trial_ledger = _trial_ledger_document(
        auxiliary,
        stage12_binding_sha256=run.stage12_binding.binding_sha256,
    )
    primary_metrics = primary_metric["metrics"]
    _require(
        isinstance(primary_metrics, Mapping),
        "primary metric payload is not an object",
    )
    inference_results = {
        **common,
        "input_sha256": sha256_json(inference_inputs),
        "execution_status": STAGE14_PENDING_STATUS,
        "execution_scope": STAGE14_HANDOFF_SCOPE,
        "verified_stage13_binding_required": True,
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
            "joint_outer_indices_sha256": (FROZEN_JOINT_OUTER_INDICES_SHA256),
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
        "final_payload_open_count": len(access_records),
        "synthetic_payload_open_count": 0,
        "records": access_records,
    }
    report_inputs = {
        **common,
        "report_template_id": "stage13-report-template-v2",
        "primary_confirmatory_months": list(PRIMARY_FULL_HOLDING_MONTHS),
        "core_complete_months": list(CORE_COMPLETE_HOLDING_MONTHS),
        "partial_descriptive_month": PARTIAL_DECEMBER_HOLDING_MONTH,
        "series_splicing_permitted": False,
        "headline_metric": (
            "fully_net_annualized_sharpe_of_calendar_month_excess_nav_return"
        ),
        "stage14_execution_status": STAGE14_PENDING_STATUS,
        "terminal_liquidation_sensitivity_receipt_sha256": (
            terminal_sensitivity["sensitivity_receipt_sha256"]
        ),
        "source_receipts": {
            METRICS_FILE: sha256_json(metrics_doc),
            ROBUSTNESS_FILE: sha256_json(robustness_doc),
            INFERENCE_INPUTS_FILE: sha256_json(inference_inputs),
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
    results = assemble_stage13_result_set(documents, synthetic=False)

    # Import lazily so publication can import the result constants without a
    # module-initialization cycle.  A return is impossible unless the complete
    # frozen semantic replay succeeds.
    from portfolio_bot.final_test.publication import (
        Stage13PublicationError,
        prepare_stage13_bundle,
    )

    try:
        prepare_stage13_bundle(run, results)
    except Stage13PublicationError as exc:
        raise ProductionResultAssemblyError(
            "empirical Stage 13 result semantic replay failed"
        ) from exc
    return results


def production_result_assembler_sha256() -> str:
    """Stable identity of the closed assembler contract, independent of data."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "assembler_id": PRODUCTION_RESULT_ASSEMBLER_ID,
                "month_fragment_files": list(MONTH_FRAGMENT_FILES),
                "result_files": [
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
                ],
                "active_strategies": list(ACTIVE_STRATEGY_IDS),
                "cost_constraint_fragment_fields": sorted(
                    _FRAGMENT_FIELDS[COST_CONSTRAINT_MATERIALITY_FILE]
                ),
                "terminal_event_source_fields": sorted(_TERMINAL_EVENT_SOURCE_FIELDS),
                "terminal_event_observation_fields": sorted(
                    _TERMINAL_EVENT_OBSERVATION_FIELDS
                ),
                "terminal_liquidation_sensitivity": {
                    "event_source_count": 2,
                    "strategy_observation_count": 12,
                    "scenario_count": 4,
                    "status": "COMPLETE_REPORTING_ONLY",
                    "final_thesis_conclusion_status": (
                        _SENSITIVITY_FINAL_CONCLUSION_STATUS
                    ),
                    "report_template_id": "stage13-report-template-v2",
                },
                "robustness_request_ids": [
                    value.robustness_id for value in FROZEN_ROBUSTNESS_REQUESTS
                ],
                "raw_dsr_trial_count": 386,
                "stage14_pending_status": STAGE14_PENDING_STATUS,
                "stage14_handoff_scope": STAGE14_HANDOFF_SCOPE,
                "synthetic_builder_permitted": False,
            }
        )
    ).hexdigest()


__all__ = [
    "EmpiricalAuxiliaryEvidence",
    "EmpiricalMonthResultFragment",
    "MONTH_FRAGMENT_FILES",
    "PRODUCTION_RESULT_ASSEMBLER_ID",
    "STAGE14_HANDOFF_SCOPE",
    "STAGE14_PENDING_STATUS",
    "ProductionResultAssemblyError",
    "assemble_empirical_stage13_results",
    "production_result_assembler_sha256",
]
