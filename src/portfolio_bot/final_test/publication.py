"""Immutable Stage 13 result publication and independent semantic verification."""

from __future__ import annotations

import hashlib
import json
import math
import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from portfolio_bot.data.secure_publication import (
    AuthenticatedBundleGuard,
    AuthenticatedPointerGuard,
    PublishedBundle,
    SecurePublicationError,
    SecureStagingDirectory,
    replace_pointer_last,
)
from portfolio_bot.final_test.calendar import (
    PARTIAL_DECEMBER_HOLDING_MONTH,
    shift_month,
)
from portfolio_bot.final_test.checkpoint import CheckpointError, MonthlyCheckpointRecord
from portfolio_bot.final_test.orchestration import (
    FROZEN_SEGMENT_PLANS,
    PROJECT_ROOT,
    Stage13RunResult,
)
from portfolio_bot.final_test.production_results import (
    STAGE14_HANDOFF_SCOPE,
    STAGE14_PENDING_STATUS,
)
from portfolio_bot.final_test.results import (
    ACCESS_LEDGER_FILE,
    ACTIVE_STRATEGY_IDS,
    AGGREGATE_MATERIALITY_MAX_USD,
    ALLOCATION_EVIDENCE_FILE,
    CALIBRATED_STRATEGY_IDS,
    COST_CONSTRAINT_MATERIALITY_FILE,
    DAILY_ACCOUNTING_FILE,
    DESCRIPTIVE_BENCHMARK_IDS,
    EQUAL_WEIGHT_STRATEGY_ID,
    FACTOR_BENCHMARK_FILE,
    FACTOR_NAMES,
    FROZEN_ROBUSTNESS_REGISTRY,
    FROZEN_ROBUSTNESS_REQUESTS,
    INFERENCE_INPUTS_FILE,
    INFERENCE_COMPARISON_STRATEGY_IDS,
    INFERENCE_RESULTS_FILE,
    METRICS_FILE,
    MODEL_EVIDENCE_FILE,
    MONTHLY_ACCOUNTING_FILE,
    LONG_SHORT_STRATEGY_IDS,
    PER_POSITION_MATERIALITY_STRICT_UPPER_USD,
    PRIMARY_STRATEGY_ID,
    REPORT_INPUTS_FILE,
    ROBUSTNESS_FILE,
    STAGE13_RESULT_FILES,
    TRIAL_LEDGER_EVIDENCE_FILE,
    Stage13ResultError,
    Stage13ResultSet,
    build_synthetic_stage13_results,
    canonical_json_bytes,
    compute_required_metrics,
    sha256_json,
)
from portfolio_bot.final_test.terminal_event_sensitivity import (
    TERMINAL_EVENT_SENSITIVITY_ID,
    StrategyMonthlyReturn,
    TerminalEventSource,
    TerminalEventStrategyObservation,
    evaluate_terminal_event_sensitivities,
)


SEGMENT_CHAINS_FILE = "segment_chains.json"
STAGE13_PRIMARY_FILES = (
    "_SUCCESS.json",
    "manifest.json",
    SEGMENT_CHAINS_FILE,
    *STAGE13_RESULT_FILES,
)
STAGE13_VERIFICATION_FILES = ("_SUCCESS.json", "verification_report.json")
STAGE13_PRIMARY_POINTER = "artifacts/phase_06_final_test/LATEST_STAGE_13_PRIMARY.json"
STAGE13_VERIFIED_POINTER = "artifacts/phase_06_final_test/LATEST_STAGE_13_VERIFIED.json"
STAGE13_PRIMARY_NAMESPACE = "artifacts/phase_06_final_test/stage_13_results"
STAGE13_VERIFICATION_NAMESPACE = "artifacts/phase_06_final_test/stage_13_verifications"


class Stage13PublicationError(RuntimeError):
    """Raised when a Stage 13 result cannot be immutably authenticated."""


@dataclass(frozen=True, slots=True)
class PreparedStage13Bundle:
    bundle_id: str
    documents: Mapping[str, Mapping[str, Any]]
    payloads: Mapping[str, bytes]
    hashes: Mapping[str, str]
    synthetic: bool


@dataclass(frozen=True, slots=True)
class Stage13PrimaryPublication:
    bundle_id: str
    bundle: PublishedBundle
    pointer_path: Path
    pointer_sha256: str
    manifest_sha256: str
    synthetic: bool


@dataclass(frozen=True, slots=True)
class Stage13VerificationPublication:
    verification_id: str
    bundle: PublishedBundle
    pointer_path: Path
    pointer_sha256: str
    report_sha256: str
    synthetic: bool


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return canonical_json_bytes(value)
    except Stage13ResultError as exc:
        raise Stage13PublicationError(str(exc)) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _receipt(value: bytes) -> dict[str, object]:
    return {"sha256": _sha256_bytes(value), "size_bytes": len(value)}


def _write_member(stage: SecureStagingDirectory, name: str, payload: bytes) -> None:
    with stage.open_exclusive_binary(name) as stream:
        written = stream.write(payload)
        if written != len(payload):
            raise Stage13PublicationError(f"short Stage 13 write: {name}")


def _chain_document(run: Stage13RunResult) -> dict[str, object]:
    segments: list[dict[str, object]] = []
    try:
        pairs = zip(FROZEN_SEGMENT_PLANS, run.segments, strict=True)
        for expected, result in pairs:
            if result.plan != expected or result.head.ordinal != len(
                expected.holding_months
            ):
                raise Stage13PublicationError("Stage 13 segment is not complete")
            segments.append(
                {
                    "segment": expected.segment.value,
                    "chain_id": expected.chain_id,
                    "checkpoint_plan_sha256": result.checkpoint_plan_sha256,
                    "holding_months": list(expected.holding_months),
                    "registration_names": list(expected.registration_names),
                    "head_checkpoint_sha256": result.head.checkpoint_sha256,
                    "head_state_sha256": result.head.state_sha256,
                    "chain_records": [
                        record.to_mapping() for record in result.chain_records
                    ],
                }
            )
    except ValueError as exc:
        raise Stage13PublicationError("Stage 13 segment result count differs") from exc
    return {
        "schema_version": 1,
        "stage": 13,
        "synthetic": run.synthetic,
        "segments_are_separate": True,
        "primary_core_partial_splicing_permitted": False,
        "segments": segments,
    }


def prepare_stage13_bundle(
    run: Stage13RunResult,
    results: Stage13ResultSet | None = None,
) -> PreparedStage13Bundle:
    """Build the closed bundle; synthetic calls may request the built-in probe."""

    if not isinstance(run, Stage13RunResult):
        raise Stage13PublicationError("Stage 13 run result type differs")
    if results is None:
        if not run.synthetic:
            raise Stage13PublicationError(
                "empirical publication requires complete economic results"
            )
        results = build_synthetic_stage13_results(run)
    if (
        not isinstance(results, Stage13ResultSet)
        or results.synthetic is not run.synthetic
    ):
        raise Stage13PublicationError("Stage 13 economic result identity differs")

    chains = _chain_document(run)
    body_documents: dict[str, Mapping[str, Any]] = {
        SEGMENT_CHAINS_FILE: chains,
        **{name: results.documents[name] for name in STAGE13_RESULT_FILES},
    }
    body_payloads = {
        name: _canonical_json_bytes(document)
        for name, document in body_documents.items()
    }
    body_receipts = {name: _receipt(payload) for name, payload in body_payloads.items()}
    content_fingerprint = _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema_version": 2,
                "stage12_binding": run.stage12_binding.to_mapping(),
                "stage12_binding_sha256": run.stage12_binding.binding_sha256,
                "execution_config_sha256": run.execution_config_sha256,
                "members": body_receipts,
                "synthetic": run.synthetic,
            }
        )
    )
    bundle_id = f"stage13-result-{content_fingerprint[:24]}"
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "stage": 13,
        "bundle_id": bundle_id,
        "status": (
            "STAGE13_SYNTHETIC_COMPLETE_CONTRACT_VERIFIED"
            if run.synthetic
            else "STAGE13_FINAL_TEST_COMPLETE"
        ),
        "synthetic": run.synthetic,
        "content_fingerprint": content_fingerprint,
        "stage12_binding": run.stage12_binding.to_mapping(),
        "stage12_binding_sha256": run.stage12_binding.binding_sha256,
        "execution_config_sha256": run.execution_config_sha256,
        "members": body_receipts,
        "inference_boundaries": {
            "primary_confirmatory_month_count": 68,
            "core_reported_separately": True,
            "partial_december_descriptive_only": True,
            "series_splicing_permitted": False,
        },
        "materiality_boundaries": {
            "per_position_residue_strict_upper_usd": PER_POSITION_MATERIALITY_STRICT_UPPER_USD,
            "aggregate_absolute_residue_max_usd_per_strategy_month": AGGREGATE_MATERIALITY_MAX_USD,
        },
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    success: dict[str, Any] = {
        "schema_version": 2,
        "stage": 13,
        "bundle_id": bundle_id,
        "status": manifest["status"],
        "synthetic": run.synthetic,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "members": {
            "manifest.json": _receipt(manifest_bytes),
            **body_receipts,
        },
    }
    documents: dict[str, Mapping[str, Any]] = {
        "_SUCCESS.json": success,
        "manifest.json": manifest,
        **body_documents,
    }
    payloads = {
        name: _canonical_json_bytes(documents[name]) for name in STAGE13_PRIMARY_FILES
    }
    prepared = PreparedStage13Bundle(
        bundle_id=bundle_id,
        documents=documents,
        payloads=payloads,
        hashes={name: _sha256_bytes(payload) for name, payload in payloads.items()},
        synthetic=run.synthetic,
    )
    # This uses the same independent semantic replay as post-publication verification.
    _validate_document_set(documents, expected_synthetic=run.synthetic)
    return prepared


def _publish_primary(
    prepared: PreparedStage13Bundle, *, project_root: Path
) -> Stage13PrimaryPublication:
    root = project_root.absolute()
    destination = root / STAGE13_PRIMARY_NAMESPACE / prepared.bundle_id
    with SecureStagingDirectory.create(
        trusted_root=root,
        destination=destination,
        expected_files=STAGE13_PRIMARY_FILES,
        label="Stage 13 result bundle",
    ) as stage:
        for name in sorted(STAGE13_PRIMARY_FILES):
            _write_member(stage, name, prepared.payloads[name])
        snapshot = stage.snapshot(expected_hashes=prepared.hashes)
        published = stage.publish(expected_hashes=snapshot.hashes)
    pointer = {
        "schema_version": 2,
        "stage": 13,
        "bundle_id": prepared.bundle_id,
        "status": prepared.documents["manifest.json"]["status"],
        "synthetic": prepared.synthetic,
        "bundle_path": destination.relative_to(root).as_posix(),
        "closed_member_count": len(STAGE13_PRIMARY_FILES),
        "member_hashes": dict(prepared.hashes),
        "manifest_sha256": prepared.hashes["manifest.json"],
        "success_sha256": prepared.hashes["_SUCCESS.json"],
    }
    pointer_bytes = _canonical_json_bytes(pointer)
    pointer_path = root / STAGE13_PRIMARY_POINTER

    def gate_primary_bundle() -> None:
        with AuthenticatedBundleGuard.open(
            trusted_root=root,
            path=published.path,
            expected_files=STAGE13_PRIMARY_FILES,
            expected_hashes=prepared.hashes,
            label="Stage 13 primary result bundle",
        ) as guard:
            guard.gate()

    replace_pointer_last(
        pointer_path,
        pointer_bytes,
        trusted_root=root,
        pre_replace_gate=gate_primary_bundle,
        label="Stage 13 primary pointer",
    )
    return Stage13PrimaryPublication(
        bundle_id=prepared.bundle_id,
        bundle=published,
        pointer_path=pointer_path,
        pointer_sha256=_sha256_bytes(pointer_bytes),
        manifest_sha256=prepared.hashes["manifest.json"],
        synthetic=prepared.synthetic,
    )


def _parse_object(payload: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage13PublicationError(f"{name} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise Stage13PublicationError(f"{name} is not an object")
    return value


def _safe_bundle_path(root: Path, relative_value: object, *, bundle_id: str) -> Path:
    if not isinstance(relative_value, str) or "\\" in relative_value:
        raise Stage13PublicationError("Stage 13 bundle path is invalid")
    relative = PurePosixPath(relative_value)
    expected = PurePosixPath(STAGE13_PRIMARY_NAMESPACE) / bundle_id
    if relative.is_absolute() or ".." in relative.parts or relative != expected:
        raise Stage13PublicationError("Stage 13 bundle path escaped its namespace")
    return root / Path(*relative.parts)


def _authenticate_primary_pointer(
    *, project_root: Path, pointer_path: Path, expected_synthetic: bool
) -> tuple[AuthenticatedPointerGuard, AuthenticatedBundleGuard, Mapping[str, Any]]:
    root = project_root.absolute()
    pointer_guard = AuthenticatedPointerGuard.open(
        trusted_root=root, path=pointer_path, label="Stage 13 primary pointer"
    )
    try:
        pointer = _parse_object(pointer_guard.payload, name="Stage 13 primary pointer")
        bundle_id = str(pointer.get("bundle_id"))
        hashes = pointer.get("member_hashes")
        if (
            pointer.get("schema_version") != 2
            or pointer.get("stage") != 13
            or pointer.get("synthetic") is not expected_synthetic
            or not isinstance(hashes, Mapping)
            or set(hashes) != set(STAGE13_PRIMARY_FILES)
            or pointer.get("closed_member_count") != len(STAGE13_PRIMARY_FILES)
        ):
            raise Stage13PublicationError("Stage 13 primary pointer semantics differ")
        bundle_path = _safe_bundle_path(
            root, pointer.get("bundle_path"), bundle_id=bundle_id
        )
        bundle_guard = AuthenticatedBundleGuard.open(
            trusted_root=root,
            path=bundle_path,
            expected_files=STAGE13_PRIMARY_FILES,
            expected_hashes={name: str(value) for name, value in hashes.items()},
            label="Stage 13 primary result bundle",
        )
        return pointer_guard, bundle_guard, pointer
    except BaseException:
        pointer_guard.close()
        raise


def _strict(value: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    if set(value) != fields:
        raise Stage13PublicationError(f"{label} field set differs")


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise Stage13PublicationError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Stage13PublicationError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise Stage13PublicationError(f"{label} must be finite")
    return result


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage13PublicationError(f"{label} must be an exact SHA-256")
    return value


def _close(left: object, right: object, *, label: str, atol: float = 1e-10) -> None:
    if not math.isclose(
        _finite(left, label=label),
        _finite(right, label=label),
        rel_tol=1e-12,
        abs_tol=atol,
    ):
        raise Stage13PublicationError(f"{label} does not reconcile")


def _records(document: Mapping[str, Any], *, label: str) -> list[Mapping[str, Any]]:
    value = document.get("records")
    if not isinstance(value, list) or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise Stage13PublicationError(f"{label} records are invalid")
    return value


def _validate_chains(
    chains: Mapping[str, Any], *, expected_synthetic: bool
) -> dict[tuple[str, str], str]:
    _strict(
        chains,
        {
            "schema_version",
            "stage",
            "synthetic",
            "segments_are_separate",
            "primary_core_partial_splicing_permitted",
            "segments",
        },
        label="segment chains",
    )
    if (
        chains["schema_version"] != 1
        or chains["stage"] != 13
        or chains["synthetic"] is not expected_synthetic
        or chains["segments_are_separate"] is not True
        or chains["primary_core_partial_splicing_permitted"] is not False
    ):
        raise Stage13PublicationError("Stage 13 segment chain boundaries differ")
    segments = chains["segments"]
    if not isinstance(segments, list) or len(segments) != len(FROZEN_SEGMENT_PLANS):
        raise Stage13PublicationError("Stage 13 segment set differs")
    lookup: dict[tuple[str, str], str] = {}
    for expected, segment in zip(FROZEN_SEGMENT_PLANS, segments, strict=True):
        if not isinstance(segment, Mapping):
            raise Stage13PublicationError("Stage 13 segment receipt is invalid")
        _strict(
            segment,
            {
                "segment",
                "chain_id",
                "checkpoint_plan_sha256",
                "holding_months",
                "registration_names",
                "head_checkpoint_sha256",
                "head_state_sha256",
                "chain_records",
            },
            label="segment receipt",
        )
        records = segment["chain_records"]
        if not isinstance(records, list) or len(records) != len(
            expected.holding_months
        ):
            raise Stage13PublicationError("Stage 13 chain length differs")
        previous: str | None = None
        for ordinal, (month, raw) in enumerate(
            zip(expected.holding_months, records, strict=True), start=1
        ):
            try:
                record = MonthlyCheckpointRecord.from_mapping(raw)
            except (CheckpointError, TypeError) as exc:
                raise Stage13PublicationError(
                    "Stage 13 checkpoint receipt is invalid"
                ) from exc
            if (
                record.holding_month != month
                or record.ordinal != ordinal
                or record.previous_checkpoint_sha256 != previous
                or record.chain_id != expected.chain_id
                or record.plan_sha256 != segment["checkpoint_plan_sha256"]
            ):
                raise Stage13PublicationError("Stage 13 chain is reordered or spliced")
            lookup[(expected.segment.value, month)] = record.checkpoint_sha256
            previous = record.checkpoint_sha256
        if (
            segment["segment"] != expected.segment.value
            or segment["chain_id"] != expected.chain_id
            or segment["holding_months"] != list(expected.holding_months)
            or segment["registration_names"] != list(expected.registration_names)
            or segment["head_checkpoint_sha256"] != previous
            or segment["head_state_sha256"] != records[-1]["state_sha256"]
        ):
            raise Stage13PublicationError("Stage 13 segment boundary differs")
    return lookup


def _expected_months() -> dict[str, tuple[str, ...]]:
    return {plan.segment.value: plan.holding_months for plan in FROZEN_SEGMENT_PLANS}


def _common(document: Mapping[str, Any], *, expected_synthetic: bool) -> None:
    if (
        document.get("schema_version") != 1
        or document.get("stage") != 13
        or document.get("synthetic") is not expected_synthetic
    ):
        raise Stage13PublicationError("economic member identity differs")


def _validate_economic_documents(
    documents: Mapping[str, Mapping[str, Any]],
    *,
    checkpoint: Mapping[tuple[str, str], str],
    stage12_binding_sha256: str,
    expected_synthetic: bool,
) -> dict[str, int]:
    for name in STAGE13_RESULT_FILES:
        _common(documents[name], expected_synthetic=expected_synthetic)
    months_by_segment = _expected_months()
    expected_strategy_months = {
        (segment, month, strategy)
        for segment, months in months_by_segment.items()
        for month in months
        for strategy in ACTIVE_STRATEGY_IDS
    }

    model_doc = documents[MODEL_EVIDENCE_FILE]
    _strict(
        model_doc,
        {"schema_version", "stage", "synthetic", "chronology", "records"},
        label="model evidence",
    )
    if model_doc["chronology"] != "completed_labels_only":
        raise Stage13PublicationError("model chronology differs")
    model_rows = _records(model_doc, label="model evidence")
    model_by_key: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    grouped_models: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in model_rows:
        _strict(
            row,
            {
                "holding_month",
                "segment",
                "strategy_id",
                "permno",
                "model_config_id",
                "refit_id",
                "training_cutoff_month",
                "refit_receipt_sha256",
                "raw_forecast",
                "calibration_slope",
                "calibrated_forecast",
                "score",
                "forecast_receipt_sha256",
                "monthly_spearman_ic",
                "checkpoint_sha256",
            },
            label="model row",
        )
        segment, month, strategy = (
            str(row["segment"]),
            str(row["holding_month"]),
            str(row["strategy_id"]),
        )
        if (segment, month, strategy) not in expected_strategy_months or str(
            row["training_cutoff_month"]
        ) >= month:
            raise Stage13PublicationError(
                "model row crossed a frozen chronology boundary"
            )
        permno = row["permno"]
        if type(permno) is not int or permno <= 0:
            raise Stage13PublicationError("model permno is invalid")
        key = (segment, month, strategy, permno)
        if (
            key in model_by_key
            or row["checkpoint_sha256"] != checkpoint[(segment, month)]
        ):
            raise Stage13PublicationError("model row key/checkpoint differs")
        score = _finite(row["score"], label="score")
        slope = row["calibration_slope"]
        if not -0.5 <= score <= 0.5:
            raise Stage13PublicationError("model calibration/score differs")
        if strategy in CALIBRATED_STRATEGY_IDS:
            if _finite(slope, label="calibration_slope") < 0.0:
                raise Stage13PublicationError("model calibration/score differs")
        elif slope is not None:
            raise Stage13PublicationError(
                "uncalibrated strategy carries a calibration slope"
            )
        _finite(row["raw_forecast"], label="raw_forecast")
        _finite(row["calibrated_forecast"], label="calibrated_forecast")
        _finite(row["monthly_spearman_ic"], label="monthly_spearman_ic")
        expected_refit_receipt = sha256_json(
            {
                "model_config_id": row["model_config_id"],
                "refit_id": row["refit_id"],
                "training_cutoff_month": row["training_cutoff_month"],
            }
        )
        expected_forecast_receipt = sha256_json(
            {
                "holding_month": month,
                "strategy_id": strategy,
                "permno": permno,
                "raw_forecast": row["raw_forecast"],
                "calibration_slope": row["calibration_slope"],
                "calibrated_forecast": row["calibrated_forecast"],
                "score": row["score"],
                "refit_receipt_sha256": row["refit_receipt_sha256"],
            }
        )
        if (
            row["refit_receipt_sha256"] != expected_refit_receipt
            or row["forecast_receipt_sha256"] != expected_forecast_receipt
        ):
            raise Stage13PublicationError("model refit/forecast receipt differs")
        model_by_key[key] = row
        grouped_models.setdefault((segment, month, strategy), []).append(row)
    if set(grouped_models) != expected_strategy_months or any(
        not rows for rows in grouped_models.values()
    ):
        raise Stage13PublicationError("model evidence key domain differs")

    allocation_doc = documents[ALLOCATION_EVIDENCE_FILE]
    _strict(
        allocation_doc,
        {
            "schema_version",
            "stage",
            "synthetic",
            "weight_unit",
            "problem_contract",
            "candidate_policy",
            "candidate_audits",
            "records",
        },
        label="allocation evidence",
    )
    if (
        allocation_doc["weight_unit"] != "fraction_of_start_nav"
        or allocation_doc["problem_contract"]
        != "frozen_stage12_problem_constraints_costs_and_inherited_state"
    ):
        raise Stage13PublicationError("allocation weight unit differs")
    if allocation_doc["candidate_policy"] != {
        "minimum_new_names_each_side": 50,
        "tail_extension_rule": "scan_extreme_scores_until_minimum",
        "insufficient_side_rule": "take_no_new_risk_and_reduce_existing_risk",
        "candidate_sets_must_be_disjoint": True,
        "equal_weight_class": "descriptive_costed_long_only_not_long_short",
    }:
        raise Stage13PublicationError("candidate policy differs")
    allocation_rows = _records(allocation_doc, label="allocation evidence")
    allocation_by_key: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    grouped_allocations: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in allocation_rows:
        _strict(
            row,
            {
                "holding_month",
                "segment",
                "strategy_id",
                "permno",
                "score",
                "current_weight",
                "target_weight",
                "trade_weight",
                "adv_trade_fraction",
                "mandatory_action",
                "candidate_side",
                "residue_usd",
                "allocation_problem_sha256",
                "solver_status",
                "constraint_certificate_sha256",
                "checkpoint_sha256",
            },
            label="allocation row",
        )
        segment, month, strategy = (
            str(row["segment"]),
            str(row["holding_month"]),
            str(row["strategy_id"]),
        )
        permno = row["permno"]
        key = (segment, month, strategy, permno)
        if (
            key in allocation_by_key
            or row["checkpoint_sha256"] != checkpoint[(segment, month)]
        ):
            raise Stage13PublicationError("allocation key/checkpoint differs")
        score = _finite(row["score"], label="allocation score")
        target = _finite(row["target_weight"], label="target_weight")
        current = _finite(row["current_weight"], label="current_weight")
        trade = _finite(row["trade_weight"], label="trade_weight")
        if type(row["mandatory_action"]) is not bool:
            raise Stage13PublicationError("mandatory-action flag is invalid")
        _close(trade, target - current, label="trade delta")
        candidate_side = row["candidate_side"]
        if strategy == EQUAL_WEIGHT_STRATEGY_ID:
            if (
                candidate_side
                not in {
                    "descriptive_long",
                    "descriptive_blocked",
                    "reduction_only",
                }
                or target < 0.0
                or (
                    candidate_side == "reduction_only"
                    and target != 0.0
                    and (target != current or trade != 0.0)
                )
                or (
                    candidate_side == "descriptive_blocked"
                    and (current != 0.0 or target != 0.0 or trade != 0.0)
                )
            ):
                raise Stage13PublicationError("equal-weight position class differs")
        elif (
            candidate_side
            not in {"long_candidate", "short_candidate", "reduction_only"}
            or (candidate_side == "long_candidate" and (target < 0.0 or score < 0.0))
            or (candidate_side == "short_candidate" and (target > 0.0 or score > 0.0))
        ):
            raise Stage13PublicationError("candidate sign/class differs")
        model = model_by_key.get(key)
        if model is None:
            if candidate_side != "reduction_only":
                raise Stage13PublicationError(
                    "risk-adding allocation lacks model evidence"
                )
        else:
            _close(score, model["score"], label="model/allocation score")
        if candidate_side == "reduction_only" and abs(target) > abs(current) + 1e-12:
            raise Stage13PublicationError("reduction-only allocation added risk")
        if (
            strategy != EQUAL_WEIGHT_STRATEGY_ID
            and candidate_side != "reduction_only"
            and target != 0.0
            and math.copysign(1.0, target) != math.copysign(1.0, score)
        ):
            raise Stage13PublicationError("target sign opposes score")
        residue = _finite(row["residue_usd"], label="position residue")
        if abs(residue) >= PER_POSITION_MATERIALITY_STRICT_UPPER_USD:
            raise Stage13PublicationError("per-position numerical residue is material")
        adv_fraction = _finite(row["adv_trade_fraction"], label="adv_trade_fraction")
        if not 0.0 <= adv_fraction <= 0.1:
            raise Stage13PublicationError("ADV trade constraint is violated")
        problem_sha = sha256_json(
            {
                "segment": segment,
                "holding_month": month,
                "strategy_id": strategy,
                "checkpoint_sha256": row["checkpoint_sha256"],
                "policy_id": "frozen-stage12-allocation-policy-v1",
            }
        )
        certificate_sha = sha256_json(
            {
                "allocation_problem_sha256": problem_sha,
                "permno": permno,
                "current_weight": row["current_weight"],
                "target_weight": row["target_weight"],
                "trade_weight": row["trade_weight"],
            }
        )
        expected_solver_status = (
            "DETERMINISTIC_EQUAL_WEIGHT"
            if strategy == EQUAL_WEIGHT_STRATEGY_ID
            else "OPTIMAL_OR_CERTIFIED_FALLBACK"
        )
        if (
            row["allocation_problem_sha256"] != problem_sha
            or row["solver_status"] != expected_solver_status
            or row["constraint_certificate_sha256"] != certificate_sha
        ):
            raise Stage13PublicationError("allocation problem/certificate differs")
        # Current weights are measured from the authenticated opening book
        # after the prior holding-period return path.  They generally differ
        # from prior entry targets.  Portfolio continuity is already bound by
        # the ordered Stage 13 checkpoint/state chain and the kernel result
        # committed to this checkpoint; comparing against a prior target here
        # would reject economically correct return drift.
        allocation_by_key[key] = row
        grouped_allocations.setdefault((segment, month, strategy), []).append(row)
    if set(grouped_allocations) != expected_strategy_months or any(
        not rows for rows in grouped_allocations.values()
    ):
        raise Stage13PublicationError("allocation key domain differs")

    candidate_audits = allocation_doc["candidate_audits"]
    if not isinstance(candidate_audits, list) or not all(
        isinstance(row, Mapping) for row in candidate_audits
    ):
        raise Stage13PublicationError("candidate audit rows are invalid")
    candidate_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    candidate_fields = {
        "holding_month",
        "segment",
        "strategy_id",
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
        "checkpoint_sha256",
    }
    for row in candidate_audits:
        _strict(
            row,
            {*candidate_fields, "candidate_evidence_sha256"},
            label="candidate audit row",
        )
        key = (
            str(row["segment"]),
            str(row["holding_month"]),
            str(row["strategy_id"]),
        )
        if (
            key not in expected_strategy_months
            or key in candidate_by_key
            or row["checkpoint_sha256"] != checkpoint[key[:2]]
            or row["minimum_new_names_each_side"] != 50
            or row["candidate_sets_disjoint"] is not True
        ):
            raise Stage13PublicationError("candidate audit key/policy differs")
        for name in (
            "eligible_universe_count",
            "candidate_long_count",
            "candidate_short_count",
            "reduction_only_count",
        ):
            if type(row[name]) is not int or row[name] < 0:
                raise Stage13PublicationError("candidate count is invalid")
        for name in (
            "minimum_each_side_applies",
            "tail_extension_used",
            "insufficient_side_fallback_used",
            "new_risk_permitted",
            "candidate_sets_disjoint",
        ):
            if type(row[name]) is not bool:
                raise Stage13PublicationError("candidate policy flag is invalid")
        payload = {name: row[name] for name in candidate_fields}
        if row["candidate_evidence_sha256"] != sha256_json(payload):
            raise Stage13PublicationError("candidate evidence digest differs")
        positions = grouped_allocations[key]
        long_count = sum(
            item["candidate_side"] == "long_candidate" for item in positions
        )
        short_count = sum(
            item["candidate_side"] == "short_candidate" for item in positions
        )
        descriptive_count = sum(
            item["candidate_side"] == "descriptive_long" for item in positions
        )
        blocked_descriptive_count = sum(
            item["candidate_side"] == "descriptive_blocked" for item in positions
        )
        reduction_count = sum(
            item["candidate_side"] == "reduction_only" for item in positions
        )
        if row["reduction_only_count"] != reduction_count:
            raise Stage13PublicationError("reduction-only candidate count differs")
        if row["eligible_universe_count"] < len(positions):
            raise Stage13PublicationError("candidate universe count is too small")
        if key[2] == EQUAL_WEIGHT_STRATEGY_ID:
            expected_new_risk_permitted = any(
                item["candidate_side"] == "descriptive_long"
                and _finite(item["target_weight"], label="target weight") > 0.0
                for item in positions
            )
            if (
                row["minimum_each_side_applies"] is not False
                or row["candidate_long_count"] != descriptive_count
                or row["candidate_short_count"] != 0
                or row["insufficient_side_fallback_used"] is not False
                or row["new_risk_permitted"] is not expected_new_risk_permitted
                or descriptive_count + blocked_descriptive_count + reduction_count
                != len(positions)
            ):
                raise Stage13PublicationError("equal-weight candidate evidence differs")
        else:
            fallback = row["insufficient_side_fallback_used"]
            enough = long_count >= 50 and short_count >= 50
            if (
                row["minimum_each_side_applies"] is not True
                or row["candidate_long_count"] != long_count
                or row["candidate_short_count"] != short_count
                or fallback is enough
                or row["new_risk_permitted"] is not enough
            ):
                raise Stage13PublicationError("long/short candidate minimum differs")
            if fallback and any(
                abs(_finite(item["target_weight"], label="fallback target"))
                > abs(_finite(item["current_weight"], label="fallback current")) + 1e-12
                for item in positions
            ):
                raise Stage13PublicationError("insufficient-side fallback added risk")
        candidate_by_key[key] = row
    if set(candidate_by_key) != expected_strategy_months:
        raise Stage13PublicationError("candidate audit key domain differs")

    daily_doc = documents[DAILY_ACCOUNTING_FILE]
    _strict(
        daily_doc,
        {"schema_version", "stage", "synthetic", "records"},
        label="daily accounting",
    )
    daily_rows = _records(daily_doc, label="daily accounting")
    daily_by_key: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    daily_keys: set[tuple[str, str, str, str]] = set()
    for row in daily_rows:
        _strict(
            row,
            {
                "holding_month",
                "segment",
                "strategy_id",
                "trading_date",
                "start_nav_usd",
                "gross_pnl_usd",
                "total_cost_usd",
                "net_pnl_usd",
                "end_nav_usd",
                "checkpoint_sha256",
            },
            label="daily row",
        )
        key = (str(row["segment"]), str(row["holding_month"]), str(row["strategy_id"]))
        trading_date_text = str(row["trading_date"])
        try:
            trading_date = dt.date.fromisoformat(trading_date_text)
        except ValueError as exc:
            raise Stage13PublicationError("daily accounting date is invalid") from exc
        allowed_months = {key[1]}
        if key[1] != PARTIAL_DECEMBER_HOLDING_MONTH:
            allowed_months.add(shift_month(key[1], 1))
        daily_key = (*key, trading_date_text)
        if (
            key not in expected_strategy_months
            or daily_key in daily_keys
            or trading_date.strftime("%Y-%m") not in allowed_months
            or trading_date.weekday() >= 5
            or row["checkpoint_sha256"] != checkpoint[key[:2]]
        ):
            raise Stage13PublicationError(
                "daily accounting key/date/checkpoint differs"
            )
        start = _finite(row["start_nav_usd"], label="daily start NAV")
        gross_pnl = _finite(row["gross_pnl_usd"], label="daily gross P&L")
        cost = _finite(row["total_cost_usd"], label="daily cost")
        net_pnl = _finite(row["net_pnl_usd"], label="daily net P&L")
        end = _finite(row["end_nav_usd"], label="daily end NAV")
        if start <= 0.0 or end <= 0.0 or cost < 0.0:
            raise Stage13PublicationError("daily accounting has an invalid sign")
        _close(net_pnl, gross_pnl - cost, label="daily net P&L")
        _close(end, start + net_pnl, label="daily end NAV", atol=1e-6)
        daily_by_key.setdefault(key, []).append(row)
        daily_keys.add(daily_key)
    if set(daily_by_key) != expected_strategy_months or any(
        len(rows) < 2 for rows in daily_by_key.values()
    ):
        raise Stage13PublicationError("daily accounting key domain differs")
    for rows in daily_by_key.values():
        dates = [str(row["trading_date"]) for row in rows]
        if dates != sorted(dates) or not dates[0].startswith(
            str(rows[0]["holding_month"]) + "-"
        ):
            raise Stage13PublicationError("daily accounting dates are reordered")
        for previous, current in zip(rows, rows[1:], strict=False):
            _close(
                current["start_nav_usd"],
                previous["end_nav_usd"],
                label="daily NAV continuity",
                atol=1e-6,
            )

    monthly_doc = documents[MONTHLY_ACCOUNTING_FILE]
    _strict(
        monthly_doc,
        {"schema_version", "stage", "synthetic", "segments_are_separate", "records"},
        label="monthly accounting",
    )
    if monthly_doc["segments_are_separate"] is not True:
        raise Stage13PublicationError("monthly accounting permits series splicing")
    monthly_rows = _records(monthly_doc, label="monthly accounting")
    monthly_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    prior_nav: dict[tuple[str, str], float] = {}
    monthly_fields = {
        "holding_month",
        "segment",
        "strategy_id",
        "start_nav_usd",
        "end_nav_usd",
        "gross_pnl_usd",
        "net_pnl_usd",
        "total_cost_usd",
        "gross_return",
        "net_return",
        "risk_free_return",
        "excess_return",
        "turnover",
        "gross_exposure",
        "net_exposure",
        "market_beta",
        "cash_weight",
        "position_count",
        "effective_position_count",
        "maximum_abs_weight",
        "concentration",
        "checkpoint_sha256",
    }
    for row in monthly_rows:
        _strict(row, monthly_fields, label="monthly row")
        key = (str(row["segment"]), str(row["holding_month"]), str(row["strategy_id"]))
        if (
            key not in expected_strategy_months
            or key in monthly_by_key
            or row["checkpoint_sha256"] != checkpoint[key[:2]]
        ):
            raise Stage13PublicationError("monthly accounting key/checkpoint differs")
        daily = daily_by_key[key]
        _close(
            row["start_nav_usd"],
            daily[0]["start_nav_usd"],
            label="daily/monthly start NAV",
            atol=1e-6,
        )
        _close(
            row["end_nav_usd"],
            daily[-1]["end_nav_usd"],
            label="daily/monthly end NAV",
            atol=1e-6,
        )
        for name in ("gross_pnl_usd", "net_pnl_usd", "total_cost_usd"):
            _close(
                row[name],
                sum(_finite(item[name], label=f"daily {name}") for item in daily),
                label=f"daily/monthly {name}",
                atol=1e-6,
            )
        start = _finite(row["start_nav_usd"], label="monthly start NAV")
        end = _finite(row["end_nav_usd"], label="monthly end NAV")
        _close(
            row["gross_return"],
            _finite(row["gross_pnl_usd"], label="gross P&L") / start,
            label="gross return",
        )
        _close(
            row["net_return"],
            _finite(row["net_pnl_usd"], label="net P&L") / start,
            label="net return",
        )
        _close(
            row["excess_return"],
            _finite(row["net_return"], label="net return")
            - _finite(row["risk_free_return"], label="risk free"),
            label="excess return",
        )
        sequence_key = (key[0], key[2])
        if sequence_key in prior_nav:
            _close(
                start,
                prior_nav[sequence_key],
                label="within-segment NAV continuity",
                atol=1e-6,
            )
        prior_nav[sequence_key] = end
        if type(row["position_count"]) is not int or row["position_count"] < 0:
            raise Stage13PublicationError("monthly position count differs")
        monthly_by_key[key] = row
    if set(monthly_by_key) != expected_strategy_months:
        raise Stage13PublicationError("monthly accounting key domain differs")

    audit_doc = documents[COST_CONSTRAINT_MATERIALITY_FILE]
    _strict(
        audit_doc,
        {
            "schema_version",
            "stage",
            "synthetic",
            "per_position_residue_rule",
            "aggregate_residue_rule",
            "terminal_event_sources",
            "terminal_event_strategy_observations",
            "records",
        },
        label="cost/constraint/materiality evidence",
    )
    if (
        audit_doc["per_position_residue_rule"] != "strictly_less_than_0.05_usd"
        or audit_doc["aggregate_residue_rule"]
        != "less_than_or_equal_to_1.00_usd_per_strategy_month"
    ):
        raise Stage13PublicationError("materiality rule label differs")
    audit_rows = _records(audit_doc, label="cost/constraint/materiality evidence")
    audit_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    cost_parts = (
        "commission_usd",
        "spread_usd",
        "impact_usd",
        "borrow_usd",
        "financing_usd",
    )
    audit_fields = {
        "holding_month",
        "segment",
        "strategy_id",
        *cost_parts,
        "total_cost_usd",
        "one_way_turnover",
        "gross_exposure",
        "net_exposure",
        "market_beta",
        "maximum_adv_trade_fraction",
        "maximum_single_position_residue_usd",
        "aggregate_absolute_residue_usd",
        "hard_constraint_violation_count",
        "checkpoint_sha256",
    }
    for row in audit_rows:
        _strict(row, audit_fields, label="cost/constraint/materiality row")
        key = (str(row["segment"]), str(row["holding_month"]), str(row["strategy_id"]))
        if (
            key not in expected_strategy_months
            or key in audit_by_key
            or row["checkpoint_sha256"] != checkpoint[key[:2]]
        ):
            raise Stage13PublicationError("cost audit key/checkpoint differs")
        components = [_finite(row[name], label=name) for name in cost_parts]
        if any(value < 0.0 for value in components):
            raise Stage13PublicationError("cost component has an invalid sign")
        total = _finite(row["total_cost_usd"], label="total cost")
        _close(total, sum(components), label="cost component sum", atol=1e-6)
        monthly = monthly_by_key[key]
        _close(
            total, monthly["total_cost_usd"], label="cost/accounting total", atol=1e-6
        )
        allocation = grouped_allocations[key]
        turnover = 0.5 * sum(
            abs(_finite(item["trade_weight"], label="trade weight"))
            for item in allocation
        )
        gross_exposure = sum(
            abs(_finite(item["target_weight"], label="target weight"))
            for item in allocation
        )
        net_exposure = sum(
            _finite(item["target_weight"], label="target weight") for item in allocation
        )
        target_weights = [
            _finite(item["target_weight"], label="target weight") for item in allocation
        ]
        nonzero_weights = [value for value in target_weights if value != 0.0]
        concentration = sum(value * value for value in target_weights)
        effective_positions = (
            gross_exposure * gross_exposure / concentration
            if concentration > 0.0
            else 0.0
        )
        maximum_abs_weight = max(abs(value) for value in target_weights)
        _close(row["one_way_turnover"], turnover, label="turnover")
        _close(row["gross_exposure"], gross_exposure, label="gross exposure")
        _close(row["net_exposure"], net_exposure, label="net exposure")
        for name in (
            "one_way_turnover",
            "gross_exposure",
            "net_exposure",
            "market_beta",
        ):
            monthly_name = "turnover" if name == "one_way_turnover" else name
            _close(row[name], monthly[monthly_name], label=f"audit/monthly {name}")
        if monthly["position_count"] != len(nonzero_weights):
            raise Stage13PublicationError("position count does not match weights")
        _close(
            monthly["effective_position_count"],
            effective_positions,
            label="effective position count",
        )
        _close(
            monthly["maximum_abs_weight"],
            maximum_abs_weight,
            label="maximum absolute weight",
        )
        _close(monthly["concentration"], concentration, label="concentration")
        _close(monthly["cash_weight"], 1.0 - net_exposure, label="cash weight")
        observed_maximum_adv = max(
            _finite(item["adv_trade_fraction"], label="ADV trade fraction")
            for item in allocation
        )
        _close(
            row["maximum_adv_trade_fraction"],
            observed_maximum_adv,
            label="maximum ADV participation",
        )
        maximum_residue = _finite(
            row["maximum_single_position_residue_usd"], label="maximum residue"
        )
        aggregate_residue = _finite(
            row["aggregate_absolute_residue_usd"], label="aggregate residue"
        )
        position_residues = [
            abs(_finite(item["residue_usd"], label="position residue"))
            for item in allocation
        ]
        _close(
            maximum_residue,
            max(position_residues),
            label="maximum position residue",
        )
        _close(
            aggregate_residue,
            sum(position_residues),
            label="aggregate position residue",
        )
        if not 0.0 <= maximum_residue < PER_POSITION_MATERIALITY_STRICT_UPPER_USD:
            raise Stage13PublicationError("per-position numerical residue is material")
        if not 0.0 <= aggregate_residue <= AGGREGATE_MATERIALITY_MAX_USD:
            raise Stage13PublicationError("aggregate numerical residue is material")
        if row["hard_constraint_violation_count"] != 0:
            raise Stage13PublicationError("hard constraint violation was reported")
        if any(
            value > 0.01 + 1e-12 or value < -0.0075 - 1e-12 for value in target_weights
        ):
            raise Stage13PublicationError("issuer position constraint is violated")
        common_constraint_failure = (
            gross_exposure > 2.0 + 1e-12
            or turnover > 0.5 + 1e-12
            or observed_maximum_adv > 0.1 + 1e-12
        )
        if key[2] == EQUAL_WEIGHT_STRATEGY_ID:
            positive_descriptive_weights = [
                _finite(item["target_weight"], label="target weight")
                for item in allocation
                if item["candidate_side"] == "descriptive_long"
                and _finite(item["target_weight"], label="target weight") > 0.0
            ]
            new_risk_permitted = candidate_by_key[key]["new_risk_permitted"]
            strategy_constraint_failure = (
                any(value < 0.0 for value in target_weights)
                or abs(gross_exposure - 1.0) > 1e-12
                or abs(net_exposure - gross_exposure) > 1e-12
                or (
                    new_risk_permitted is True
                    and (
                        not positive_descriptive_weights
                        or any(
                            not math.isclose(
                                value,
                                positive_descriptive_weights[0],
                                rel_tol=1e-12,
                                abs_tol=1e-15,
                            )
                            for value in positive_descriptive_weights[1:]
                        )
                    )
                )
                or (new_risk_permitted is False and bool(positive_descriptive_weights))
                or components[cost_parts.index("borrow_usd")] != 0.0
                or components[cost_parts.index("financing_usd")] != 0.0
            )
        else:
            strategy_constraint_failure = (
                abs(net_exposure) > min(0.1, 0.05 * gross_exposure) + 1e-12
                or abs(_finite(row["market_beta"], label="market beta")) > 0.1
            )
        if common_constraint_failure or strategy_constraint_failure:
            raise Stage13PublicationError("frozen portfolio constraint is violated")
        audit_by_key[key] = row
    if set(audit_by_key) != expected_strategy_months:
        raise Stage13PublicationError("cost audit key domain differs")

    terminal_source_rows = audit_doc["terminal_event_sources"]
    terminal_observation_rows = audit_doc["terminal_event_strategy_observations"]
    if (
        not isinstance(terminal_source_rows, list)
        or len(terminal_source_rows) != 2
        or not isinstance(terminal_observation_rows, list)
        or len(terminal_observation_rows) != 2 * len(ACTIVE_STRATEGY_IDS)
    ):
        raise Stage13PublicationError("terminal event evidence count differs")
    source_fields = {
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
    terminal_sources: list[TerminalEventSource] = []
    source_by_action: dict[str, TerminalEventSource] = {}
    for row in terminal_source_rows:
        if not isinstance(row, Mapping):
            raise Stage13PublicationError("terminal event source row is invalid")
        _strict(row, source_fields, label="terminal event source row")
        try:
            source = TerminalEventSource(
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
        except (TypeError, ValueError) as exc:
            raise Stage13PublicationError(
                "terminal event source semantics differ"
            ) from exc
        key = ("primary_full", source.holding_month)
        if (
            row["segment"] != "primary_full"
            or key not in checkpoint
            or row["checkpoint_sha256"] != checkpoint[key]
            or source.terminal_action_id in source_by_action
        ):
            raise Stage13PublicationError(
                "terminal event source key/checkpoint differs"
            )
        _close(
            row["selected_full_spread_stress_rate"],
            source.selected_full_spread_stress_rate,
            label="terminal selected spread",
            atol=0.0,
        )
        source_by_action[source.terminal_action_id] = source
        terminal_sources.append(source)
    if not expected_synthetic:
        observed_terminal_identities = {
            (source.permno, source.event_date) for source in terminal_sources
        }
        if (
            observed_terminal_identities
            != {
                (85_421, dt.date(2021, 1, 8)),
                (88_888, dt.date(2021, 3, 8)),
            }
            or len({source.terminal_policy_sha256 for source in terminal_sources}) != 1
        ):
            raise Stage13PublicationError("empirical terminal event population differs")

    observation_fields = {
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
    terminal_observations: list[TerminalEventStrategyObservation] = []
    observation_keys: set[tuple[str, str]] = set()
    for row in terminal_observation_rows:
        if not isinstance(row, Mapping):
            raise Stage13PublicationError(
                "terminal event strategy observation is invalid"
            )
        _strict(
            row,
            observation_fields,
            label="terminal event strategy observation",
        )
        action_id = str(row["terminal_action_id"])
        source = source_by_action.get(action_id)
        strategy_id = str(row["strategy_id"])
        if source is None:
            raise Stage13PublicationError(
                "terminal observation names an unknown source"
            )
        key = ("primary_full", source.holding_month, strategy_id)
        observation_key = (action_id, strategy_id)
        if (
            strategy_id not in ACTIVE_STRATEGY_IDS
            or observation_key in observation_keys
            or row["segment"] != "primary_full"
            or row["holding_month"] != source.holding_month
            or row["checkpoint_sha256"]
            != checkpoint[("primary_full", source.holding_month)]
        ):
            raise Stage13PublicationError("terminal observation key/checkpoint differs")
        try:
            observation = TerminalEventStrategyObservation(
                terminal_action_id=action_id,
                strategy_id=strategy_id,
                position_held=row["position_held"],
                target_weight=row["target_weight"],
                month_start_nav_dollars=row["month_start_nav_dollars"],
                event_nav_dollars=row["event_nav_dollars"],
                signed_event_value_dollars=row["signed_event_value_dollars"],
            )
        except (TypeError, ValueError) as exc:
            raise Stage13PublicationError(
                "terminal observation semantics differ"
            ) from exc
        allocation_matches = [
            allocation
            for allocation in grouped_allocations[key]
            if int(allocation["permno"]) == source.permno
        ]
        expected_target = (
            0.0
            if not allocation_matches
            else _finite(
                allocation_matches[0]["target_weight"],
                label="terminal target weight",
            )
        )
        if len(allocation_matches) > 1:
            raise Stage13PublicationError("terminal target allocation repeats")
        _close(
            observation.target_weight,
            expected_target,
            label="terminal observation target weight",
            atol=0.0,
        )
        _close(
            observation.month_start_nav_dollars,
            monthly_by_key[key]["start_nav_usd"],
            label="terminal observation month start NAV",
            atol=1e-6,
        )
        event_daily = [
            daily
            for daily in daily_by_key[key]
            if daily["trading_date"] == source.event_date.isoformat()
        ]
        if len(event_daily) != 1:
            raise Stage13PublicationError("terminal event daily NAV row differs")
        _close(
            observation.event_nav_dollars,
            event_daily[0]["end_nav_usd"],
            label="terminal observation event NAV",
            atol=1e-6,
        )
        _close(
            row["event_day_portfolio_weight"],
            observation.event_day_portfolio_weight,
            label="terminal event portfolio weight",
            atol=1e-15,
        )
        observation_keys.add(observation_key)
        terminal_observations.append(observation)
    if observation_keys != {
        (source.terminal_action_id, strategy_id)
        for source in terminal_sources
        for strategy_id in ACTIVE_STRATEGY_IDS
    }:
        raise Stage13PublicationError(
            "terminal observations do not form the complete strategy grid"
        )

    factor_doc = documents[FACTOR_BENCHMARK_FILE]
    _strict(
        factor_doc,
        {
            "schema_version",
            "stage",
            "synthetic",
            "factor_names",
            "benchmark_ids",
            "active_costs_applied_to_passive_etfs_or_cash",
            "factor_rows",
            "benchmark_rows",
        },
        label="factor/benchmark evidence",
    )
    if (
        factor_doc["factor_names"] != list(FACTOR_NAMES)
        or factor_doc["benchmark_ids"] != list(DESCRIPTIVE_BENCHMARK_IDS)
        or factor_doc["active_costs_applied_to_passive_etfs_or_cash"] is not False
    ):
        raise Stage13PublicationError("factor/benchmark registry differs")
    factor_rows = factor_doc["factor_rows"]
    benchmark_rows = factor_doc["benchmark_rows"]
    if not isinstance(factor_rows, list) or not isinstance(benchmark_rows, list):
        raise Stage13PublicationError("factor/benchmark rows are invalid")
    factor_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in factor_rows:
        if not isinstance(row, Mapping):
            raise Stage13PublicationError("factor row is invalid")
        _strict(
            row,
            {
                "holding_month",
                "segment",
                "risk_free_return",
                *FACTOR_NAMES,
                "checkpoint_sha256",
            },
            label="factor row",
        )
        key = (str(row["segment"]), str(row["holding_month"]))
        if (
            key not in checkpoint
            or key in factor_by_key
            or row["checkpoint_sha256"] != checkpoint[key]
        ):
            raise Stage13PublicationError("factor row key/checkpoint differs")
        for name in ("risk_free_return", *FACTOR_NAMES):
            _finite(row[name], label=name)
        factor_by_key[key] = row
    if set(factor_by_key) != set(checkpoint):
        raise Stage13PublicationError("factor month domain differs")
    benchmark_keys: set[tuple[str, str, str]] = set()
    for row in benchmark_rows:
        if not isinstance(row, Mapping):
            raise Stage13PublicationError("benchmark row is invalid")
        _strict(
            row,
            {
                "holding_month",
                "segment",
                "benchmark_id",
                "total_return",
                "active_turnover_cost_applied",
            },
            label="benchmark row",
        )
        key = (str(row["segment"]), str(row["holding_month"]), str(row["benchmark_id"]))
        expected_active_cost = key[2] == EQUAL_WEIGHT_STRATEGY_ID
        if (
            key[:2] not in checkpoint
            or key[2] not in DESCRIPTIVE_BENCHMARK_IDS
            or key in benchmark_keys
            or row["active_turnover_cost_applied"] is not expected_active_cost
        ):
            raise Stage13PublicationError("benchmark key/cost treatment differs")
        _finite(row["total_return"], label="benchmark return")
        if key[2] == EQUAL_WEIGHT_STRATEGY_ID:
            _close(
                row["total_return"],
                monthly_by_key[(key[0], key[1], EQUAL_WEIGHT_STRATEGY_ID)][
                    "net_return"
                ],
                label="equal-weight costed benchmark return",
            )
        elif key[2] == "cash_or_one_month_tbill":
            _close(
                row["total_return"],
                factor_by_key[key[:2]]["risk_free_return"],
                label="cash benchmark return",
            )
        benchmark_keys.add(key)
    expected_benchmarks = {
        (segment, month, benchmark)
        for segment, months in months_by_segment.items()
        for month in months
        for benchmark in DESCRIPTIVE_BENCHMARK_IDS
    }
    if benchmark_keys != expected_benchmarks:
        raise Stage13PublicationError("benchmark month domain differs")

    metrics_doc = documents[METRICS_FILE]
    _strict(
        metrics_doc,
        {
            "schema_version",
            "stage",
            "synthetic",
            "primary_confirmatory_segment",
            "core_reported_separately",
            "partial_descriptive_only",
            "metric_rows",
            "partial_descriptive_rows",
        },
        label="required metrics",
    )
    if (
        metrics_doc["primary_confirmatory_segment"] != "primary_full"
        or metrics_doc["core_reported_separately"] is not True
        or metrics_doc["partial_descriptive_only"] is not True
    ):
        raise Stage13PublicationError("metric segment boundary differs")
    metric_rows = metrics_doc["metric_rows"]
    if not isinstance(metric_rows, list):
        raise Stage13PublicationError("metric rows are invalid")
    expected_metric_keys = {
        (segment, strategy)
        for segment in ("primary_full", "core_complete")
        for strategy in ACTIVE_STRATEGY_IDS
    }
    metric_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in metric_rows:
        if not isinstance(row, Mapping):
            raise Stage13PublicationError("metric row is invalid")
        _strict(
            row,
            {"segment", "strategy_id", "holding_months", "metrics"},
            label="metric row",
        )
        key = (str(row["segment"]), str(row["strategy_id"]))
        if (
            key not in expected_metric_keys
            or key in metric_by_key
            or row["holding_months"] != list(months_by_segment[key[0]])
        ):
            raise Stage13PublicationError("metric row key/month domain differs")
        source_monthly = [
            monthly_by_key[(key[0], month, key[1])]
            for month in months_by_segment[key[0]]
        ]
        source_factors = [
            factor_by_key[(key[0], month)] for month in months_by_segment[key[0]]
        ]
        ic = []
        for month in months_by_segment[key[0]]:
            permnos = sorted(
                item[3] for item in model_by_key if item[:3] == (key[0], month, key[1])
            )
            values = [
                _finite(
                    model_by_key[(key[0], month, key[1], permno)][
                        "monthly_spearman_ic"
                    ],
                    label="monthly IC",
                )
                for permno in permnos
            ]
            if any(
                not math.isclose(value, values[0], rel_tol=0.0, abs_tol=1e-15)
                for value in values[1:]
            ):
                raise Stage13PublicationError(
                    "monthly IC evidence differs within a strategy-month"
                )
            ic.append(values[0])
        expected_metrics = compute_required_metrics(
            source_monthly, source_factors, monthly_ic=ic
        )
        if _canonical_json_bytes(row["metrics"]) != _canonical_json_bytes(
            expected_metrics
        ):
            raise Stage13PublicationError(
                "required metrics were not recomputed from source rows"
            )
        metric_by_key[key] = row
    if set(metric_by_key) != expected_metric_keys:
        raise Stage13PublicationError("required metric key domain differs")
    expected_partial = [
        monthly_by_key[
            ("partial_december_descriptive", PARTIAL_DECEMBER_HOLDING_MONTH, strategy)
        ]
        for strategy in ACTIVE_STRATEGY_IDS
    ]
    if _canonical_json_bytes(
        metrics_doc["partial_descriptive_rows"]
    ) != _canonical_json_bytes(expected_partial):
        raise Stage13PublicationError("partial December was altered or promoted")

    ledger = documents[TRIAL_LEDGER_EVIDENCE_FILE]
    _strict(
        ledger,
        {
            "schema_version",
            "stage",
            "synthetic",
            "ledger_id",
            "stage12_binding_sha256",
            "raw_trial_count",
            "excluded_row_count",
            "total_ledger_rows",
            "ledger_sha256",
            "rows",
        },
        label="trial ledger evidence",
    )
    trial_rows = ledger["rows"]
    if not isinstance(trial_rows, list) or not all(
        isinstance(row, Mapping) for row in trial_rows
    ):
        raise Stage13PublicationError("trial ledger rows are invalid")
    trial_fields = {
        "trial_id",
        "category",
        "candidate_id",
        "stage_registered",
        "source_build_id",
        "status",
        "seed_aggregation",
        "result_available",
        "pnl_available",
        "counts_in_raw_dsr",
        "inclusion_reason",
        "source_receipt_sha256",
        "duplicate_of_trial_id",
    }
    for row in trial_rows:
        _strict(row, trial_fields, label="trial ledger row")
        _require_sha256(row["source_receipt_sha256"], label="trial source receipt")
    trial_ids = [row["trial_id"] for row in trial_rows]
    raw_count = sum(row["counts_in_raw_dsr"] is True for row in trial_rows)
    if (
        ledger["ledger_id"] != "stage12-complete-frozen-trial-ledger-v1"
        or ledger["stage12_binding_sha256"] != stage12_binding_sha256
        or ledger["raw_trial_count"] != 386
        or raw_count != 386
        or ledger["total_ledger_rows"] != len(trial_rows)
        or ledger["excluded_row_count"] != len(trial_rows) - raw_count
        or len(trial_ids) != len(set(trial_ids))
        or ledger["ledger_sha256"] != sha256_json(trial_rows)
    ):
        raise Stage13PublicationError("trial ledger evidence does not reconcile")

    robustness = documents[ROBUSTNESS_FILE]
    robustness_fields = {
        "schema_version",
        "stage",
        "synthetic",
        "one_at_a_time_only",
        "cartesian_product_executed",
        "frozen_registry",
        "dispatch_rows",
        "post_unblind_terminal_liquidation_sensitivity",
    }
    if not expected_synthetic:
        robustness_fields.add("execution_status")
    _strict(
        robustness,
        robustness_fields,
        label="robustness results",
    )
    if (
        robustness["one_at_a_time_only"] is not True
        or robustness["cartesian_product_executed"] is not False
        or robustness["frozen_registry"] != list(FROZEN_ROBUSTNESS_REGISTRY)
        or (
            not expected_synthetic
            and robustness["execution_status"] != STAGE14_PENDING_STATUS
        )
    ):
        raise Stage13PublicationError("robustness boundary differs")
    dispatch_rows = robustness["dispatch_rows"]
    if not isinstance(dispatch_rows, list) or len(dispatch_rows) != len(
        FROZEN_ROBUSTNESS_REQUESTS
    ):
        raise Stage13PublicationError("robustness dispatch count differs")
    baseline = metric_by_key[("primary_full", PRIMARY_STRATEGY_ID)]
    baseline_hash = sha256_json(baseline)
    for ordinal, (row, request) in enumerate(
        zip(dispatch_rows, FROZEN_ROBUSTNESS_REQUESTS, strict=True), start=1
    ):
        if not isinstance(row, Mapping):
            raise Stage13PublicationError("robustness dispatch row is invalid")
        _strict(
            row,
            {
                "ordinal",
                "robustness_id",
                "registry_id",
                "changed_axis",
                "overrides",
                "one_at_a_time",
                "reporting_only",
                "result",
            },
            label="robustness dispatch row",
        )
        expected_request = {"ordinal": ordinal, **request.to_mapping()}
        if any(row[name] != value for name, value in expected_request.items()):
            raise Stage13PublicationError(
                "robustness request was reordered or combined"
            )
        result = row["result"]
        if not isinstance(result, Mapping):
            raise Stage13PublicationError("robustness result is invalid")
        _strict(
            result,
            {
                "robustness_id",
                "status",
                "primary_reference_metrics_sha256",
                "result_metrics",
                "promoted_to_primary",
            },
            label="robustness result",
        )
        if (
            result["robustness_id"] != request.robustness_id
            or result["primary_reference_metrics_sha256"] != baseline_hash
            or result["promoted_to_primary"] is not False
            or (
                not expected_synthetic
                and (
                    result["status"] != STAGE14_PENDING_STATUS
                    or result["result_metrics"] != {}
                )
            )
        ):
            raise Stage13PublicationError("robustness result lost its primary binding")

    sensitivity = robustness["post_unblind_terminal_liquidation_sensitivity"]
    sensitivity_fields = {
        "sensitivity_id",
        "status",
        "event_source_count",
        "strategy_observation_count",
        "event_sources_sha256",
        "strategy_observations_sha256",
        "one_at_a_time_only",
        "cartesian_product_executed",
        "primary_estimates_modified",
        "model_fitting_scoring_selection_or_allocation_rerun",
        "scenario_results_may_be_promoted",
        "baseline_strategy_results",
        "scenarios",
        "conclusion_effects",
        "final_thesis_conclusion_status",
        "sensitivity_receipt_sha256",
    }
    if not isinstance(sensitivity, Mapping):
        raise Stage13PublicationError("terminal sensitivity result is invalid")
    _strict(
        sensitivity,
        sensitivity_fields,
        label="terminal liquidation sensitivity",
    )
    primary_sensitivity_monthly = [
        StrategyMonthlyReturn(
            strategy_id=strategy_id,
            holding_month=month,
            net_return=monthly_by_key[("primary_full", month, strategy_id)][
                "net_return"
            ],
            excess_return=monthly_by_key[("primary_full", month, strategy_id)][
                "excess_return"
            ],
        )
        for month in months_by_segment["primary_full"]
        for strategy_id in ACTIVE_STRATEGY_IDS
    ]
    try:
        independently_evaluated = evaluate_terminal_event_sensitivities(
            primary_sensitivity_monthly,
            terminal_sources,
            terminal_observations,
        ).as_dict()
    except (TypeError, ValueError) as exc:
        raise Stage13PublicationError(
            "terminal sensitivity independent replay failed"
        ) from exc
    baseline_by_strategy = {
        str(row["strategy_id"]): row
        for row in independently_evaluated["baseline_strategy_results"]
    }
    baseline_primary_sensitivity = baseline_by_strategy[PRIMARY_STRATEGY_ID]
    baseline_elastic_sensitivity = baseline_by_strategy["elastic_net_hybrid_challenger"]

    def _metric_sign(value: object) -> int:
        numeric = _finite(value, label="terminal sensitivity metric")
        return int(numeric > 0.0) - int(numeric < 0.0)

    expected_conclusion_effects: list[dict[str, object]] = []
    for scenario in independently_evaluated["scenarios"]:
        scenario_by_strategy = {
            str(row["strategy_id"]): row for row in scenario["strategy_results"]
        }
        primary_result = scenario_by_strategy[PRIMARY_STRATEGY_ID]
        elastic_result = scenario_by_strategy["elastic_net_hybrid_challenger"]
        expected_conclusion_effects.append(
            {
                "scenario_id": scenario["scenario_id"],
                "primary_cagr_sign_changed": _metric_sign(
                    primary_result["metrics"]["net_cagr"]
                )
                != _metric_sign(baseline_primary_sensitivity["metrics"]["net_cagr"]),
                "primary_sharpe_sign_changed": _metric_sign(
                    primary_result["metrics"]["annualized_excess_sharpe"]
                )
                != _metric_sign(
                    baseline_primary_sensitivity["metrics"]["annualized_excess_sharpe"]
                ),
                "primary_vs_elastic_net_ordering_changed": (
                    (int(primary_result["rank"]) < int(elastic_result["rank"]))
                    != (
                        int(baseline_primary_sensitivity["rank"])
                        < int(baseline_elastic_sensitivity["rank"])
                    )
                ),
                "primary_rank_changed": (
                    int(primary_result["rank"])
                    != int(baseline_primary_sensitivity["rank"])
                ),
            }
        )
    expected_sensitivity: dict[str, object] = {
        "sensitivity_id": TERMINAL_EVENT_SENSITIVITY_ID,
        "status": "COMPLETE_REPORTING_ONLY",
        "event_source_count": 2,
        "strategy_observation_count": 2 * len(ACTIVE_STRATEGY_IDS),
        "event_sources_sha256": sha256_json(terminal_source_rows),
        "strategy_observations_sha256": sha256_json(terminal_observation_rows),
        "one_at_a_time_only": True,
        "cartesian_product_executed": False,
        "primary_estimates_modified": False,
        "model_fitting_scoring_selection_or_allocation_rerun": False,
        "scenario_results_may_be_promoted": False,
        "baseline_strategy_results": independently_evaluated[
            "baseline_strategy_results"
        ],
        "scenarios": independently_evaluated["scenarios"],
        "conclusion_effects": expected_conclusion_effects,
        "final_thesis_conclusion_status": (
            "pending_stage14_inference_and_stage15_synthesis"
        ),
    }
    expected_sensitivity["sensitivity_receipt_sha256"] = sha256_json(
        expected_sensitivity
    )
    if _canonical_json_bytes(sensitivity) != _canonical_json_bytes(
        expected_sensitivity
    ):
        raise Stage13PublicationError(
            "terminal liquidation sensitivity does not independently reconcile"
        )

    inference_inputs = documents[INFERENCE_INPUTS_FILE]
    _strict(
        inference_inputs,
        {
            "schema_version",
            "stage",
            "synthetic",
            "holding_months",
            "joint_month_rows",
            "primary_core_partial_splicing_permitted",
            "rows",
        },
        label="inference inputs",
    )
    inference_rows = inference_inputs["rows"]
    if (
        inference_inputs["holding_months"] != list(months_by_segment["primary_full"])
        or inference_inputs["joint_month_rows"] is not True
        or inference_inputs["primary_core_partial_splicing_permitted"] is not False
        or not isinstance(inference_rows, list)
        or len(inference_rows) != 68
    ):
        raise Stage13PublicationError("inference input calendar differs")
    comparison_fields = {
        f"{strategy}_excess_return" for strategy in INFERENCE_COMPARISON_STRATEGY_IDS
    }
    equal_weight_field = (
        "equal_weight_long_only_scoring_universe_descriptive_excess_return"
    )
    inference_fields = {
        "holding_month",
        "hybrid_total_return",
        "hybrid_excess_return",
        "risk_free_return",
        *FACTOR_NAMES,
        *comparison_fields,
        equal_weight_field,
    }
    for month, row in zip(
        months_by_segment["primary_full"], inference_rows, strict=True
    ):
        if not isinstance(row, Mapping):
            raise Stage13PublicationError("inference row is invalid")
        _strict(row, inference_fields, label="inference row")
        if row["holding_month"] != month:
            raise Stage13PublicationError("inference rows were reordered")
        source = monthly_by_key[("primary_full", month, PRIMARY_STRATEGY_ID)]
        factor = factor_by_key[("primary_full", month)]
        _close(
            row["hybrid_total_return"],
            source["net_return"],
            label="inference total return",
        )
        _close(
            row["hybrid_excess_return"],
            source["excess_return"],
            label="inference excess return",
        )
        for name in ("risk_free_return", *FACTOR_NAMES):
            _close(row[name], factor[name], label=f"inference {name}")
        for strategy in INFERENCE_COMPARISON_STRATEGY_IDS:
            _close(
                row[f"{strategy}_excess_return"],
                monthly_by_key[("primary_full", month, strategy)]["excess_return"],
                label="paired inference return",
            )
        _close(
            row[equal_weight_field],
            monthly_by_key[("primary_full", month, EQUAL_WEIGHT_STRATEGY_ID)][
                "excess_return"
            ],
            label="descriptive equal-weight inference return",
        )

    inference_results = documents[INFERENCE_RESULTS_FILE]
    inference_result_fields = {
        "schema_version",
        "stage",
        "synthetic",
        "input_sha256",
        "execution_scope",
        "primary_point_estimates",
        "frozen_methods",
        "trial_ledger_id",
        "trial_ledger_sha256",
        "raw_dsr_trial_count",
    }
    if not expected_synthetic:
        inference_result_fields.update(
            {"execution_status", "verified_stage13_binding_required"}
        )
    _strict(
        inference_results,
        inference_result_fields,
        label="inference results",
    )
    expected_points = {
        "sharpe": baseline["metrics"]["sharpe"],
        "annualized_alpha": baseline["metrics"]["ff5_plus_momentum_alpha_and_betas"][
            "annualized_alpha"
        ],
        "maximum_drawdown": baseline["metrics"]["maximum_drawdown"],
    }
    expected_methods = {
        "hac_lags": 6,
        "automatic_hac_secondary": True,
        "outer_replicates": 10_000,
        "inner_replicates": 1_000,
        "expected_block_months": 6,
        "master_seed": 20_260_714,
        "joint_outer_indices_sha256": "5c925d9acce36c6452f23239115adfa7a873a3594537daf53d5fa31f5c8fe41d",
        "invalid_outer_fraction_maximum": 0.01,
        "decision_focused_inactive_p_value": 1.0,
    }
    if (
        inference_results["input_sha256"] != sha256_json(inference_inputs)
        or _canonical_json_bytes(inference_results["primary_point_estimates"])
        != _canonical_json_bytes(expected_points)
        or inference_results["frozen_methods"] != expected_methods
        or inference_results["trial_ledger_id"] != ledger["ledger_id"]
        or inference_results["trial_ledger_sha256"] != ledger["ledger_sha256"]
        or inference_results["raw_dsr_trial_count"] != 386
        or (
            not expected_synthetic
            and (
                inference_results["execution_status"] != STAGE14_PENDING_STATUS
                or inference_results["execution_scope"] != STAGE14_HANDOFF_SCOPE
                or inference_results["verified_stage13_binding_required"] is not True
            )
        )
    ):
        raise Stage13PublicationError("inference result binding differs")

    access_doc = documents[ACCESS_LEDGER_FILE]
    _strict(
        access_doc,
        {
            "schema_version",
            "stage",
            "synthetic",
            "stage12_binding_sha256",
            "final_payload_open_count",
            "synthetic_payload_open_count",
            "records",
        },
        label="access ledger",
    )
    access_rows = _records(access_doc, label="access ledger")
    expected_access = {
        (plan.segment.value, month, registration)
        for plan in FROZEN_SEGMENT_PLANS
        for month in plan.holding_months
        for registration in plan.registration_names
    }
    access_keys: set[tuple[str, str, str]] = set()
    receipt_fields = {
        "segment",
        "holding_month",
        "registration_name",
        "stage12_binding_sha256",
        "checkpoint_sha256",
        "file_sha256",
        "row_count",
    }
    for row in access_rows:
        _strict(row, {*receipt_fields, "access_receipt_sha256"}, label="access receipt")
        key = (
            str(row["segment"]),
            str(row["holding_month"]),
            str(row["registration_name"]),
        )
        if (
            key not in expected_access
            or key in access_keys
            or row["stage12_binding_sha256"] != stage12_binding_sha256
            or row["checkpoint_sha256"] != checkpoint[key[:2]]
            or type(row["row_count"]) is not int
            or row["row_count"] <= 0
        ):
            raise Stage13PublicationError("access receipt key/binding differs")
        payload = {name: row[name] for name in receipt_fields}
        _require_sha256(row["file_sha256"], label="accessed file digest")
        if row["access_receipt_sha256"] != sha256_json(payload):
            raise Stage13PublicationError("access receipt digest differs")
        access_keys.add(key)
    if (
        access_keys != expected_access
        or access_doc["stage12_binding_sha256"] != stage12_binding_sha256
    ):
        raise Stage13PublicationError("access ledger domain differs")
    if expected_synthetic:
        if access_doc["final_payload_open_count"] != 0 or access_doc[
            "synthetic_payload_open_count"
        ] != len(access_rows):
            raise Stage13PublicationError("synthetic access counters differ")
    elif access_doc["synthetic_payload_open_count"] != 0 or access_doc[
        "final_payload_open_count"
    ] != len(access_rows):
        raise Stage13PublicationError("empirical access counters differ")

    report = documents[REPORT_INPUTS_FILE]
    report_fields = {
        "schema_version",
        "stage",
        "synthetic",
        "report_template_id",
        "primary_confirmatory_months",
        "core_complete_months",
        "partial_descriptive_month",
        "series_splicing_permitted",
        "headline_metric",
        "terminal_liquidation_sensitivity_receipt_sha256",
        "source_receipts",
    }
    if not expected_synthetic:
        report_fields.add("stage14_execution_status")
    _strict(
        report,
        report_fields,
        label="report inputs",
    )
    expected_receipts = {
        name: sha256_json(documents[name])
        for name in (
            METRICS_FILE,
            ROBUSTNESS_FILE,
            INFERENCE_INPUTS_FILE,
            INFERENCE_RESULTS_FILE,
            TRIAL_LEDGER_EVIDENCE_FILE,
        )
    }
    if (
        report["report_template_id"] != "stage13-report-template-v2"
        or report["primary_confirmatory_months"]
        != list(months_by_segment["primary_full"])
        or report["core_complete_months"] != list(months_by_segment["core_complete"])
        or report["partial_descriptive_month"] != "2025-12"
        or report["series_splicing_permitted"] is not False
        or report["terminal_liquidation_sensitivity_receipt_sha256"]
        != sensitivity["sensitivity_receipt_sha256"]
        or report["source_receipts"] != expected_receipts
        or (
            not expected_synthetic
            and report["stage14_execution_status"] != STAGE14_PENDING_STATUS
        )
    ):
        raise Stage13PublicationError("report input binding differs")
    return {
        "economic_member_count": len(STAGE13_RESULT_FILES),
        "strategy_month_count": len(monthly_by_key),
        "position_month_count": len(allocation_by_key),
        "access_receipt_count": len(access_rows),
        "trial_ledger_row_count": len(trial_rows),
        "robustness_dispatch_count": len(dispatch_rows),
    }


def _validate_document_set(
    documents: Mapping[str, Mapping[str, Any]], *, expected_synthetic: bool
) -> dict[str, int]:
    if set(documents) != set(STAGE13_PRIMARY_FILES):
        raise Stage13PublicationError("Stage 13 document set is not closed")
    manifest = documents["manifest.json"]
    success = documents["_SUCCESS.json"]
    chains = documents[SEGMENT_CHAINS_FILE]
    _strict(
        manifest,
        {
            "schema_version",
            "stage",
            "bundle_id",
            "status",
            "synthetic",
            "content_fingerprint",
            "stage12_binding",
            "stage12_binding_sha256",
            "execution_config_sha256",
            "members",
            "inference_boundaries",
            "materiality_boundaries",
        },
        label="manifest",
    )
    _strict(
        success,
        {
            "schema_version",
            "stage",
            "bundle_id",
            "status",
            "synthetic",
            "manifest_sha256",
            "members",
        },
        label="success",
    )
    body_names = (SEGMENT_CHAINS_FILE, *STAGE13_RESULT_FILES)
    body_receipts = {
        name: _receipt(_canonical_json_bytes(documents[name])) for name in body_names
    }
    expected_success_members = {
        "manifest.json": _receipt(_canonical_json_bytes(manifest)),
        **body_receipts,
    }
    fingerprint = _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema_version": 2,
                "stage12_binding": manifest["stage12_binding"],
                "stage12_binding_sha256": manifest["stage12_binding_sha256"],
                "execution_config_sha256": manifest["execution_config_sha256"],
                "members": body_receipts,
                "synthetic": expected_synthetic,
            }
        )
    )
    expected_bundle_id = f"stage13-result-{fingerprint[:24]}"
    binding = manifest["stage12_binding"]
    if not isinstance(binding, Mapping):
        raise Stage13PublicationError("Stage 12 binding is invalid")
    _strict(
        binding,
        {
            "stage12_build_id",
            "stage12_verification_id",
            "stage12_manifest_sha256",
            "stage12_verified_pointer_path",
            "stage12_verified_pointer_sha256",
        },
        label="Stage 12 binding",
    )
    for name in (
        "stage12_manifest_sha256",
        "stage12_verified_pointer_sha256",
    ):
        _require_sha256(binding[name], label=name)
    _require_sha256(manifest["execution_config_sha256"], label="execution config")
    if (
        manifest["schema_version"] != 2
        or manifest["stage"] != 13
        or manifest["synthetic"] is not expected_synthetic
        or manifest["bundle_id"] != expected_bundle_id
        or manifest["content_fingerprint"] != fingerprint
        or manifest["stage12_binding_sha256"] != sha256_json(binding)
        or manifest["members"] != body_receipts
        or success["schema_version"] != 2
        or success["stage"] != 13
        or success["synthetic"] is not expected_synthetic
        or success["bundle_id"] != expected_bundle_id
        or success["status"] != manifest["status"]
        or success["manifest_sha256"] != _sha256_bytes(_canonical_json_bytes(manifest))
        or success["members"] != expected_success_members
        or manifest["inference_boundaries"]
        != {
            "primary_confirmatory_month_count": 68,
            "core_reported_separately": True,
            "partial_december_descriptive_only": True,
            "series_splicing_permitted": False,
        }
        or manifest["materiality_boundaries"]
        != {
            "per_position_residue_strict_upper_usd": 0.05,
            "aggregate_absolute_residue_max_usd_per_strategy_month": 1.0,
        }
    ):
        raise Stage13PublicationError("Stage 13 manifest/success semantics differ")
    checkpoint = _validate_chains(chains, expected_synthetic=expected_synthetic)
    return _validate_economic_documents(
        {name: documents[name] for name in STAGE13_RESULT_FILES},
        checkpoint=checkpoint,
        stage12_binding_sha256=str(manifest["stage12_binding_sha256"]),
        expected_synthetic=expected_synthetic,
    )


def _validate_primary_documents(
    *,
    bundle_guard: AuthenticatedBundleGuard,
    pointer: Mapping[str, Any],
    expected_synthetic: bool,
) -> tuple[Mapping[str, Any], dict[str, int]]:
    payloads = {
        name: (bundle_guard.path / name).read_bytes() for name in STAGE13_PRIMARY_FILES
    }
    bundle_guard.gate()
    documents = {
        name: _parse_object(payload, name=f"Stage 13 {name}")
        for name, payload in payloads.items()
    }
    counters = _validate_document_set(documents, expected_synthetic=expected_synthetic)
    manifest = documents["manifest.json"]
    success = documents["_SUCCESS.json"]
    if (
        manifest["bundle_id"] != pointer.get("bundle_id")
        or pointer.get("manifest_sha256") != _sha256_bytes(payloads["manifest.json"])
        or pointer.get("success_sha256") != _sha256_bytes(payloads["_SUCCESS.json"])
        or success["manifest_sha256"] != _sha256_bytes(payloads["manifest.json"])
    ):
        raise Stage13PublicationError("Stage 13 pointer/bundle semantics differ")
    return manifest, counters


def _verify_and_publish(
    *, project_root: Path, expected_synthetic: bool
) -> Stage13VerificationPublication:
    root = project_root.absolute()
    pointer_path = root / STAGE13_PRIMARY_POINTER
    pointer_guard: AuthenticatedPointerGuard | None = None
    bundle_guard: AuthenticatedBundleGuard | None = None
    try:
        pointer_guard, bundle_guard, pointer = _authenticate_primary_pointer(
            project_root=root,
            pointer_path=pointer_path,
            expected_synthetic=expected_synthetic,
        )
        manifest, counters = _validate_primary_documents(
            bundle_guard=bundle_guard,
            pointer=pointer,
            expected_synthetic=expected_synthetic,
        )
        pointer_guard.gate()
        report = {
            "schema_version": 2,
            "stage": 13,
            "verification_status": "VERIFIED",
            "synthetic": expected_synthetic,
            "difference_count": 0,
            "primary_bundle_id": pointer["bundle_id"],
            "primary_pointer_sha256": pointer_guard.payload_sha256,
            "primary_manifest_sha256": pointer["manifest_sha256"],
            "stage12_binding_sha256": manifest["stage12_binding_sha256"],
            "segment_chain_count": 3,
            "authenticated_month_count": 72,
            "segment_splicing_detected": False,
            "semantic_replay": counters,
        }
        report_bytes = _canonical_json_bytes(report)
        verification_id = f"stage13-verify-{_sha256_bytes(report_bytes)[:24]}"
        success = {
            "schema_version": 2,
            "stage": 13,
            "verification_id": verification_id,
            "status": "STAGE13_VERIFIED",
            "synthetic": expected_synthetic,
            "verification_report_sha256": _sha256_bytes(report_bytes),
        }
        success_bytes = _canonical_json_bytes(success)
        payloads = {
            "_SUCCESS.json": success_bytes,
            "verification_report.json": report_bytes,
        }
        hashes = {name: _sha256_bytes(payload) for name, payload in payloads.items()}
        destination = root / STAGE13_VERIFICATION_NAMESPACE / verification_id
        with SecureStagingDirectory.create(
            trusted_root=root,
            destination=destination,
            expected_files=STAGE13_VERIFICATION_FILES,
            label="Stage 13 verification bundle",
        ) as stage:
            for name in sorted(STAGE13_VERIFICATION_FILES):
                _write_member(stage, name, payloads[name])
            snapshot = stage.snapshot(expected_hashes=hashes)
            published = stage.publish(
                expected_hashes=snapshot.hashes,
                pre_publish_gate=lambda: (pointer_guard.gate(), bundle_guard.gate()),
                post_publish_gate=lambda: (pointer_guard.gate(), bundle_guard.gate()),
            )
        verified_pointer = {
            "schema_version": 2,
            "stage": 13,
            "status": "STAGE13_VERIFIED",
            "synthetic": expected_synthetic,
            "verification_id": verification_id,
            "verification_bundle_path": destination.relative_to(root).as_posix(),
            "verification_member_hashes": hashes,
            "verification_report_sha256": hashes["verification_report.json"],
            "primary_bundle_id": pointer["bundle_id"],
            "primary_pointer_path": STAGE13_PRIMARY_POINTER,
            "primary_pointer_sha256": pointer_guard.payload_sha256,
            "primary_manifest_sha256": pointer["manifest_sha256"],
            "difference_count": 0,
        }
        verified_bytes = _canonical_json_bytes(verified_pointer)
        verified_path = root / STAGE13_VERIFIED_POINTER
        replace_pointer_last(
            verified_path,
            verified_bytes,
            trusted_root=root,
            pre_replace_gate=lambda: (pointer_guard.gate(), bundle_guard.gate()),
            label="Stage 13 verified pointer",
        )
        return Stage13VerificationPublication(
            verification_id=verification_id,
            bundle=published,
            pointer_path=verified_path,
            pointer_sha256=_sha256_bytes(verified_bytes),
            report_sha256=hashes["verification_report.json"],
            synthetic=expected_synthetic,
        )
    except SecurePublicationError as exc:
        raise Stage13PublicationError(str(exc)) from exc
    finally:
        if bundle_guard is not None:
            bundle_guard.close()
        if pointer_guard is not None:
            pointer_guard.close()


def publish_stage13_final_test(
    run: Stage13RunResult, results: Stage13ResultSet
) -> Stage13PrimaryPublication:
    """Publish only a certified empirical run with its complete result set."""

    if run.synthetic or results.synthetic:
        raise Stage13PublicationError("synthetic Stage 13 cannot enter production")
    return _publish_primary(
        prepare_stage13_bundle(run, results), project_root=PROJECT_ROOT
    )


def verify_stage13_final_test() -> Stage13VerificationPublication:
    """Verify only the fixed production pointer; no caller path is accepted."""

    return _verify_and_publish(project_root=PROJECT_ROOT, expected_synthetic=False)


def _publish_synthetic_stage13_for_test(
    run: Stage13RunResult, *, project_root: Path
) -> tuple[Stage13PrimaryPublication, Stage13VerificationPublication]:
    if not run.synthetic:
        raise Stage13PublicationError("synthetic publisher requires synthetic run")
    primary = _publish_primary(prepare_stage13_bundle(run), project_root=project_root)
    verified = _verify_and_publish(project_root=project_root, expected_synthetic=True)
    return primary, verified


__all__ = [
    "STAGE13_PRIMARY_FILES",
    "STAGE13_VERIFICATION_FILES",
    "PreparedStage13Bundle",
    "Stage13PrimaryPublication",
    "Stage13PublicationError",
    "Stage13VerificationPublication",
    "prepare_stage13_bundle",
    "publish_stage13_final_test",
    "verify_stage13_final_test",
]
