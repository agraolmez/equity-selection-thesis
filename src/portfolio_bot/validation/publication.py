"""Secure, content-addressed publication for the Stage 11 preflight.

The publisher is deliberately input-data agnostic.  It accepts only in-memory
JSON evidence and check rows; it has no database-root argument and contains no
reader for predictor, outcome, return, weight, trade, or P&L data.  Both the
primary and independent-verification bundles are immutable closed sets whose
mutable pointers are replaced only after the complete bundle is durable.

This module publishes *preflight evidence only*.  A verified preflight pointer
does not authorize validation outcomes, performance calculation, real
portfolio construction, final-period access, or any later stage.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    AuthenticatedBundleGuard,
    AuthenticatedPointerGuard,
    PublishedBundle,
    SecureStagingDirectory,
    replace_pointer_last,
)
from portfolio_bot.validation.access import AccessLedger, PreflightAccessBroker
from portfolio_bot.validation.contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
    REQUIRED_PREFLIGHT_CHECK_NAMES,
    Stage11PreflightContractReceipt,
    Stage11PreflightCounters,
    authenticate_stage11_preflight_contract,
    canonical_json_bytes,
    enforce_stage11_preflight_boundary,
    load_policy,
    sha256_file,
)


MANIFEST_NAME = "manifest.json"
PREFLIGHT_REPORT_NAME = "preflight_report.json"
PREFLIGHT_CHECKS_NAME = "preflight_checks.parquet"
VERIFICATION_REPORT_NAME = "verification_report.json"
VERIFICATION_CHECKS_NAME = "verification_checks.parquet"
DIFFERENCES_NAME = "differences.json"
SUCCESS_NAME = "_SUCCESS.json"

PRIMARY_FILES = (
    MANIFEST_NAME,
    PREFLIGHT_REPORT_NAME,
    PREFLIGHT_CHECKS_NAME,
    SUCCESS_NAME,
)
VERIFICATION_FILES = (
    VERIFICATION_REPORT_NAME,
    VERIFICATION_CHECKS_NAME,
    DIFFERENCES_NAME,
    SUCCESS_NAME,
)

PRIMARY_STATUS = "PREFLIGHT_PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
VERIFIED_STATUS = "PREFLIGHT_VERIFIED_PASSED"

_EXPLICIT_IMPLEMENTATION_CODE_PATHS = (
    "pyproject.toml",
    "configs/phase_04_models/allocation_build_rules_v1.json",
    "configs/phase_04_models/allocation_policy_v1.toml",
    "configs/phase_05_validation/stage_11_corporate_action_mapping_v1.json",
    "configs/phase_05_validation/stage_11_preflight_approval_v1.json",
    "configs/phase_05_validation/stage_11_preflight_policy_v1.toml",
    "configs/phase_05_validation/stage_11_preflight_rules_v1.json",
    "configs/phase_05_validation/stage_11_source_materialization_approval_v1.schema.json",
    "configs/phase_05_validation/stage_11_source_materialization_manifest_v1.schema.json",
    "configs/phase_05_validation/stage_11_source_materialization_plan_v1.json",
    "outputs/methodology_v1_freeze.json",
    "outputs/methodology_v1_source.md",
    "scripts/build_stage_11_preflight.py",
    "scripts/materialize_stage_11_validation_sources.py",
    "scripts/run_stage_11_validation.py",
    "scripts/verify_stage_11_preflight.py",
    "scripts/verify_stage_11_validation_sources.py",
    "src/portfolio_bot/allocation/candidates.py",
    "src/portfolio_bot/allocation/contract.py",
    "src/portfolio_bot/allocation/costs.py",
    "src/portfolio_bot/allocation/domain.py",
    "src/portfolio_bot/allocation/evidence.py",
    "src/portfolio_bot/allocation/fixtures.py",
    "src/portfolio_bot/allocation/constraints.py",
    "src/portfolio_bot/allocation/solver.py",
    "src/portfolio_bot/allocation/decision_focused.py",
    "src/portfolio_bot/allocation/risk.py",
    "src/portfolio_bot/allocation/rank.py",
    "src/portfolio_bot/data/secure_publication.py",
    "src/portfolio_bot/data/phase_02_auxiliary_core.py",
    "src/portfolio_bot/data/phase_02_crsp_return_reconciliation.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_discovery.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_transforms.py",
    "src/portfolio_bot/features/feature_store_contract.py",
    "src/portfolio_bot/features/label_math.py",
    "src/portfolio_bot/features/label_store_contract.py",
    "src/portfolio_bot/features/label_store_materialization.py",
    "src/portfolio_bot/features/phase_03_contract.py",
    "src/portfolio_bot/risk_cost/cost_math.py",
    "src/portfolio_bot/risk_cost/contract.py",
    "src/portfolio_bot/risk_cost/risk_math.py",
    "src/portfolio_bot/selectors/elastic_net.py",
    "src/portfolio_bot/selectors/contract.py",
    "src/portfolio_bot/selectors/inputs.py",
    "src/portfolio_bot/selectors/metrics.py",
    "src/portfolio_bot/selectors/model_specs.py",
    "src/portfolio_bot/selectors/native_process.py",
    "src/portfolio_bot/selectors/native_worker.py",
    "src/portfolio_bot/selectors/neural_model.py",
    "src/portfolio_bot/selectors/preprocessing.py",
    "src/portfolio_bot/selectors/weights.py",
    "src/portfolio_bot/selectors/xgboost_model.py",
    "src/portfolio_bot/validation/__init__.py",
    "src/portfolio_bot/validation/access.py",
    "src/portfolio_bot/validation/accounting.py",
    "src/portfolio_bot/validation/authorization.py",
    "src/portfolio_bot/validation/calendar.py",
    "src/portfolio_bot/validation/calibration.py",
    "src/portfolio_bot/validation/candidate_adapter.py",
    "src/portfolio_bot/validation/contract.py",
    "src/portfolio_bot/validation/corporate_action_adapter.py",
    "src/portfolio_bot/validation/corporate_actions.py",
    "src/portfolio_bot/validation/daily_engine.py",
    "src/portfolio_bot/validation/development_calibration.py",
    "src/portfolio_bot/validation/empirical_execution.py",
    "src/portfolio_bot/validation/empirical_portfolio_backend.py",
    "src/portfolio_bot/validation/empirical_selector_backend.py",
    "src/portfolio_bot/validation/execution_access.py",
    "src/portfolio_bot/validation/execution_input_manifest.py",
    "src/portfolio_bot/validation/execution_orchestrator.py",
    "src/portfolio_bot/validation/execution_publication.py",
    "src/portfolio_bot/validation/execution_risk_inputs.py",
    "src/portfolio_bot/validation/locked_selection.py",
    "src/portfolio_bot/validation/outcome_materialization.py",
    "src/portfolio_bot/validation/predictor_adapter.py",
    "src/portfolio_bot/validation/preflight.py",
    "src/portfolio_bot/validation/problem_builder.py",
    "src/portfolio_bot/validation/publication.py",
    "src/portfolio_bot/validation/selector_native_runtime.py",
    "src/portfolio_bot/validation/selector_runtime.py",
    "src/portfolio_bot/validation/session_authority.py",
    "src/portfolio_bot/validation/validation_build_code_identity.py",
    "src/portfolio_bot/validation/validation_data_contract.py",
    "src/portfolio_bot/validation/validation_data_independent_verification.py",
    "src/portfolio_bot/validation/validation_feature_input_adapter.py",
    "src/portfolio_bot/validation/validation_market_extension.py",
    "src/portfolio_bot/validation/validation_return_adapter.py",
    "src/portfolio_bot/validation/validation_source_extraction.py",
    "src/portfolio_bot/validation/validation_source_materialization.py",
    "src/portfolio_bot/validation/validation_source_publication.py",
    "src/portfolio_bot/validation/validation_source_verification.py",
    "src/portfolio_bot/validation/walk_forward.py",
    "src/portfolio_bot/validation/validation_runner.py",
)
_ALL_PORTFOLIO_BOT_PYTHON_PATHS = tuple(
    sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "src/portfolio_bot").rglob("*.py")
    )
)
_ALL_REVIEWED_STAGE_SCRIPT_PATHS = tuple(
    sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "scripts").glob("*.py")
    )
)
_ALL_REVIEWED_STAGE_CONFIG_PATHS = tuple(
    sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for root in (
            PROJECT_ROOT / "configs/phase_04_models",
            PROJECT_ROOT / "configs/phase_05_validation",
        )
        for path in root.iterdir()
        if path.is_file() and path.suffix in {".json", ".toml"}
    )
)
# The preflight binds the conservative complete local package surface so no
# recursive runtime dependency can be omitted from the independently verified
# code identity.  Non-Python configs, methodology files, and entry scripts
# remain explicit above.
DEFAULT_IMPLEMENTATION_CODE_PATHS = tuple(
    dict.fromkeys(
        (
            *_EXPLICIT_IMPLEMENTATION_CODE_PATHS,
            *_ALL_PORTFOLIO_BOT_PYTHON_PATHS,
            *_ALL_REVIEWED_STAGE_SCRIPT_PATHS,
            *_ALL_REVIEWED_STAGE_CONFIG_PATHS,
        )
    )
)

CHECK_SCHEMA = pa.schema(
    [
        pa.field("check_name", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("observed_value", pa.string(), nullable=False),
        pa.field("expected_value", pa.string(), nullable=False),
        pa.field("detail", pa.string(), nullable=False),
    ]
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,159}\Z")
_DATA_SUFFIXES = (
    ".arrow",
    ".csv",
    ".feather",
    ".jsonl",
    ".npy",
    ".parquet",
    ".pickle",
    ".pkl",
    ".tsv",
)

# A value beneath one of these keys is prohibited unless it is an explicit
# denial/zero.  This permits audit fields such as ``outcome_rows_read: 0`` and
# ``performance_authorized: false`` while rejecting the corresponding payload.
_ALWAYS_FORBIDDEN_KEY_PARTS = (
    "outcome",
    "return",
    "returns",
    "forward_return",
    "future_return",
    "target_return",
    "performance",
    "sharpe",
    "sortino",
    "rank_ic",
    "linear_ic",
    "spearman_ic",
    "pearson_ic",
    "information_coefficient",
    "profit_and_loss",
    "profit_loss",
    "p_and_l",
    "pnl",
    "holding",
    "holdings",
    "drawdown",
    "volatility",
    "alpha",
    "beta",
    "net_asset_value",
    "nav",
)
_REAL_POSITION_WORDS = ("weight", "weights", "trade", "trades", "position")
_REAL_POSITION_QUALIFIERS = ("real", "validation", "empirical", "portfolio")

_SUMMARY_MODES = frozenset(
    {
        "synthetic_no_outcome_preflight",
        "synthetic_and_development_parity_no_outcome",
    }
)


class Stage11PublicationError(RuntimeError):
    """Raised when a Stage 11 preflight bundle cannot be safely published."""


@dataclass(frozen=True, slots=True)
class PreflightRunSummary:
    """Closed, non-empirical description of one preflight execution."""

    runner_id: str
    implementation_mode: str
    required_check_count: int


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One fixed-schema preflight assertion; no tabular payload is accepted."""

    check_name: str
    status: str
    observed_value: bool | int | str | None
    expected_value: bool | int | str | None
    detail: str


@dataclass(frozen=True, slots=True)
class PreflightEvidenceDigest:
    """Content digest for the evidence behind one required check."""

    check_name: str
    evidence_sha256: str
    observation_count: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11PublicationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11PublicationError(f"Invalid {label} JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _safe_relative(value: object, *, label: str) -> Path:
    path = Path(str(value))
    _require(
        bool(path.parts) and not path.is_absolute() and ".." not in path.parts,
        f"Unsafe {label} path",
    )
    return path


def _safe_json_mapping(value: object, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    try:
        payload = canonical_json_bytes(dict(value))
        normalized = json.loads(payload)
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
        raise Stage11PublicationError(f"{label} is not finite canonical JSON") from exc
    _require(isinstance(normalized, dict), f"{label} must be a JSON object")
    return normalized


def _key_is_forbidden(key: str) -> bool:
    normalized = key.lower().replace("-", "_").replace(" ", "_")
    tokens = normalized.split("_")
    if (
        "ic" in tokens
        or "sr" in tokens
        or any(part in normalized for part in _ALWAYS_FORBIDDEN_KEY_PARTS)
    ):
        return True
    return any(word in normalized for word in _REAL_POSITION_WORDS) and any(
        qualifier in normalized for qualifier in _REAL_POSITION_QUALIFIERS
    )


def _is_explicit_denial(value: object) -> bool:
    if value is None or value is False:
        return True
    return type(value) in {int, float} and value == 0


def _looks_like_forbidden_data_path(value: str) -> bool:
    lowered = value.lower().replace("-", "_")
    path_like = "/" in lowered or lowered.endswith(_DATA_SUFFIXES)
    return path_like


def reject_forbidden_payload(
    value: object, *, label: str = "preflight evidence"
) -> None:
    """Reject outcome/performance/IC/Sharpe or real portfolio payloads.

    Explicit zero counters and false authorization flags are allowed.  The
    traversal accepts canonical JSON values only and therefore cannot hide a
    file handle, Arrow table, NumPy array, path object, or custom reader.
    """

    def visit(item: object, trail: tuple[str, ...]) -> None:
        location = ".".join(trail) or label
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                _require(
                    isinstance(raw_key, str) and bool(raw_key),
                    f"{label} contains an invalid key at {location}",
                )
                if _key_is_forbidden(raw_key) and not _is_explicit_denial(child):
                    raise Stage11PublicationError(
                        f"Forbidden outcome/performance/portfolio payload at "
                        f"{location}.{raw_key}"
                    )
                visit(child, (*trail, raw_key))
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, (*trail, str(index)))
            return
        if isinstance(item, str):
            if _looks_like_forbidden_data_path(item):
                raise Stage11PublicationError(
                    f"Forbidden data-payload path at {location}"
                )
            return
        _require(
            item is None or type(item) in {bool, int, float},
            f"{label} contains a non-JSON payload at {location}",
        )

    visit(value, ())
    # Also proves finite floats and deterministic JSON encodability.
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Stage11PublicationError(f"{label} is not finite canonical JSON") from exc


def _safe_bounded_text(value: object, *, label: str, maximum: int) -> str:
    _require(
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and len(value) <= maximum
        and not any(ord(character) < 32 for character in value),
        f"{label} must be bounded normalized text",
    )
    _require(
        not _looks_like_forbidden_data_path(value),
        f"{label} contains a forbidden data-payload path",
    )
    return value


def _check_scalar(value: object, *, label: str) -> bool | int | str | None:
    _require(
        value is None or type(value) in {bool, int, str},
        f"{label} must be a scalar bool, integer, string, or null",
    )
    if isinstance(value, str):
        return _safe_bounded_text(value, label=label, maximum=256)
    return value  # type: ignore[return-value]


def _display(value: bool | int | str | None) -> str:
    if isinstance(value, str):
        return value
    return canonical_json_bytes(value).decode("utf-8").rstrip("\n")


def _normalize_summary(summary: PreflightRunSummary) -> dict[str, object]:
    _require(
        type(summary) is PreflightRunSummary,
        "preflight summary must use the closed PreflightRunSummary type",
    )
    _require(
        isinstance(summary.runner_id, str)
        and _SAFE_ID.fullmatch(summary.runner_id) is not None,
        "preflight runner_id is malformed",
    )
    _require(
        isinstance(summary.implementation_mode, str)
        and summary.implementation_mode in _SUMMARY_MODES,
        "preflight implementation mode is not permitted",
    )
    _require(
        type(summary.required_check_count) is int
        and summary.required_check_count == len(REQUIRED_PREFLIGHT_CHECK_NAMES),
        "preflight summary required-check count differs",
    )
    return {
        "runner_id": summary.runner_id,
        "implementation_mode": summary.implementation_mode,
        "required_check_count": summary.required_check_count,
    }


def _normalize_evidence(
    evidence: Sequence[PreflightEvidenceDigest],
    *,
    label: str,
) -> tuple[dict[str, object], ...]:
    _require(
        not isinstance(evidence, (str, bytes)) and isinstance(evidence, Sequence),
        f"{label} must be a sequence",
    )
    rows: list[dict[str, object]] = []
    names: set[str] = set()
    for index, row in enumerate(evidence):
        _require(
            type(row) is PreflightEvidenceDigest,
            f"{label} row {index} must use PreflightEvidenceDigest",
        )
        _require(
            isinstance(row.check_name, str)
            and row.check_name in REQUIRED_PREFLIGHT_CHECK_NAMES,
            f"{label} contains an unknown check name",
        )
        _require(row.check_name not in names, f"{label} check name is duplicated")
        _require(
            isinstance(row.evidence_sha256, str)
            and _SHA256.fullmatch(row.evidence_sha256) is not None,
            f"{label} evidence digest is malformed",
        )
        _require(
            type(row.observation_count) is int and row.observation_count >= 0,
            f"{label} observation count must be nonnegative",
        )
        names.add(row.check_name)
        rows.append(
            {
                "check_name": row.check_name,
                "evidence_sha256": row.evidence_sha256,
                "observation_count": row.observation_count,
            }
        )
    _require(
        names == set(REQUIRED_PREFLIGHT_CHECK_NAMES),
        f"{label} membership differs from the configured required check set",
    )
    return tuple(sorted(rows, key=lambda row: str(row["check_name"])))


def _normalize_checks(
    checks: Sequence[PreflightCheck],
    *,
    label: str,
) -> tuple[dict[str, str], ...]:
    _require(
        not isinstance(checks, (str, bytes)) and isinstance(checks, Sequence),
        f"{label} checks must be a sequence",
    )
    normalized: list[dict[str, str]] = []
    names: set[str] = set()
    for index, row in enumerate(checks):
        _require(
            type(row) is PreflightCheck,
            f"{label} check {index} must use the closed PreflightCheck type",
        )
        name = row.check_name
        _require(
            isinstance(name, str) and name in REQUIRED_PREFLIGHT_CHECK_NAMES,
            f"{label} contains an unknown or omitted required check: {name}",
        )
        _require(name not in names, f"{label} check name is duplicated: {name}")
        _require(row.status == "PASS", f"{label} check did not pass: {name}")
        observed = _check_scalar(
            row.observed_value, label=f"{label} check {name} observed value"
        )
        expected = _check_scalar(
            row.expected_value, label=f"{label} check {name} expected value"
        )
        detail = _safe_bounded_text(
            row.detail, label=f"{label} check {name} detail", maximum=512
        )
        names.add(name)
        normalized.append(
            {
                "check_name": name,
                "status": row.status,
                "observed_value": _display(observed),
                "expected_value": _display(expected),
                "detail": detail,
            }
        )
    _require(
        names == set(REQUIRED_PREFLIGHT_CHECK_NAMES),
        f"{label} membership differs from the configured required check set",
    )
    return tuple(sorted(normalized, key=lambda row: row["check_name"]))


def _checks_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    table = pa.Table.from_pylist(list(rows), schema=CHECK_SCHEMA)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=False,
        version="2.6",
        data_page_version="1.0",
        write_statistics=True,
        row_group_size=max(1, table.num_rows),
    )
    return sink.getvalue().to_pybytes()


def _validate_check_payload(
    payload: bytes, *, label: str
) -> tuple[dict[str, str], ...]:
    try:
        table = pq.read_table(pa.BufferReader(payload))
    except (pa.ArrowInvalid, pa.ArrowTypeError, OSError) as exc:
        raise Stage11PublicationError(f"Invalid {label} check table") from exc
    _require(
        table.schema.equals(CHECK_SCHEMA, check_metadata=True),
        f"{label} check schema differs",
    )
    _require(
        table.num_rows == len(REQUIRED_PREFLIGHT_CHECK_NAMES),
        f"{label} check table row count differs",
    )
    _require(
        set(table["status"].to_pylist()) == {"PASS"},
        f"{label} contains a failing check",
    )
    rows = tuple(table.to_pylist())
    names = tuple(str(row["check_name"]) for row in rows)
    _require(
        set(names) == set(REQUIRED_PREFLIGHT_CHECK_NAMES)
        and names == tuple(sorted(names)),
        f"{label} check membership/order differs",
    )
    return rows  # type: ignore[return-value]


def _hash_code_paths(
    paths: Sequence[str],
    *,
    project_root: Path,
) -> dict[str, str]:
    _require(bool(paths), "Stage 11 publication code set cannot be empty")
    result: dict[str, str] = {}
    for raw in paths:
        path = Path(str(raw))
        _require(
            bool(path.parts) and not path.is_absolute() and ".." not in path.parts,
            f"Unsafe Stage 11 code path: {raw}",
        )
        name = path.as_posix()
        _require(name not in result, f"Duplicated Stage 11 code path: {name}")
        selected = project_root / path
        _require(
            selected.is_file() and not selected.is_symlink(),
            f"Missing or unsafe Stage 11 code dependency: {name}",
        )
        result[name] = sha256_file(selected)
    return dict(sorted(result.items()))


def _authority_hashes(receipt: Stage11PreflightContractReceipt) -> dict[str, str]:
    names = [
        "policy_sha256",
        "rules_sha256",
        "methodology_source_sha256",
        "methodology_freeze_sha256",
        "stage10_verified_pointer_sha256",
        "stage10_primary_manifest_sha256",
        "stage10_policy_sha256",
        "stage10_rules_sha256",
    ]
    # The contract deliberately distinguishes the immutable approval file from
    # its self-authenticating logical payload.  Retain compatibility with an
    # older synthetic receipt name only for isolated publication tests.
    if hasattr(receipt, "approval_file_sha256"):
        names.extend(("approval_file_sha256", "approval_payload_sha256"))
    else:
        names.append("approval_sha256")
    result = {name: str(getattr(receipt, name)) for name in names}
    _require(
        all(_SHA256.fullmatch(value) is not None for value in result.values()),
        "Stage 11 contract receipt contains a malformed authority hash",
    )
    return result


def _contract_counters_from_broker(
    access_broker: PreflightAccessBroker,
) -> tuple[Stage11PreflightCounters, AccessLedger]:
    """Derive publication counters from the live fail-closed broker only."""

    _require(
        type(access_broker) is PreflightAccessBroker,
        "preflight counters must be derived from PreflightAccessBroker",
    )
    ledger = access_broker.ledger()
    _require(
        type(ledger) is AccessLedger,
        "preflight access ledger has an unexpected type",
    )
    counters = Stage11PreflightCounters(
        validation_predictor_rows_read=ledger.validation_predictor_rows_read,
        validation_predictor_maximum_signal_date=(
            ledger.validation_predictor_maximum_signal_date
        ),
        validation_predictor_forward_return_or_label_columns_read=(
            ledger.validation_predictor_forward_return_or_label_columns_read
        ),
        validation_outcome_rows_read=ledger.validation_outcome_rows_read,
        final_period_rows_read=ledger.final_period_rows_read,
        real_validation_weights_created=ledger.real_validation_weights_created,
        real_validation_trades_created=ledger.real_validation_trades_created,
        real_validation_pnl_or_performance_rows_created=(
            ledger.real_validation_pnl_or_performance_rows_created
        ),
    )
    return counters, ledger


def _validate_counters(
    counters: Stage11PreflightCounters,
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        type(counters) is Stage11PreflightCounters,
        "preflight counters must use the immutable contract type",
    )
    normalized = counters.to_dict()
    required_zero = policy.get("counters_required_zero")
    _require(
        isinstance(required_zero, Mapping) and bool(required_zero),
        "Stage 11 policy omitted forbidden counters",
    )
    for name, expected in required_zero.items():
        _require(expected == 0, f"Policy forbidden counter is not zero: {name}")
        _require(name in normalized, f"Preflight counter is missing: {name}")
        value = normalized[name]
        _require(
            type(value) is int and value == 0,
            f"Stage 11 preflight prohibits nonzero {name}",
        )

    audited = policy.get("counters_required_audited", {})
    _require(isinstance(audited, Mapping), "Stage 11 audited counters are malformed")
    _require(
        set(normalized) == set(required_zero) | set(audited),
        "Preflight counter membership differs from the current policy",
    )
    if audited:
        if audited.get("validation_predictor_rows_read") is True:
            value = normalized.get("validation_predictor_rows_read")
            _require(
                type(value) is int and value >= 0,
                "validation_predictor_rows_read must be a nonnegative integer",
            )
        maximum = audited.get("validation_predictor_maximum_signal_date")
        if maximum is not None:
            observed = normalized.get("validation_predictor_maximum_signal_date")
            predictor_rows = normalized.get("validation_predictor_rows_read", 0)
            _require(
                (predictor_rows == 0 and observed is None)
                or (
                    type(predictor_rows) is int
                    and predictor_rows > 0
                    and isinstance(observed, str)
                    and observed <= str(maximum)
                ),
                "Validation predictor maximum signal date crossed its boundary",
            )
        required_label_count = audited.get(
            "validation_predictor_forward_return_or_label_columns_read"
        )
        if required_label_count is not None:
            observed = normalized.get(
                "validation_predictor_forward_return_or_label_columns_read"
            )
            _require(
                type(observed) is int and observed == required_label_count == 0,
                "Validation predictor payload exposed a return or label column",
            )
    return normalized


@dataclass(frozen=True)
class PrimaryPreflightPublication:
    build_id: str
    bundle_path: Path
    manifest_path: Path
    pointer_path: Path
    manifest: Mapping[str, Any]
    pointer: Mapping[str, Any]
    bundle: PublishedBundle
    reused_existing: bool


@dataclass(frozen=True)
class VerifiedPreflightPublication:
    build_id: str
    verification_id: str
    bundle_path: Path
    report_path: Path
    pointer_path: Path
    report: Mapping[str, Any]
    pointer: Mapping[str, Any]
    bundle: PublishedBundle
    reused_existing: bool


def _configured_members(
    policy: Mapping[str, Any],
    *,
    kind: str,
) -> tuple[str, ...]:
    publication = policy.get("publication")
    _require(isinstance(publication, Mapping), "Stage 11 publication policy is missing")
    key = (
        "primary_bundle_members" if kind == "primary" else "verification_bundle_members"
    )
    observed = tuple(publication.get(key, ()))
    expected = PRIMARY_FILES if kind == "primary" else VERIFICATION_FILES
    _require(observed == expected, f"Configured Stage 11 {kind} closed set differs")
    _require(len(observed) == len(set(observed)), f"Stage 11 {kind} members repeat")
    return observed


def _select_paths(
    policy: Mapping[str, Any],
    *,
    kind: str,
    identifier: str,
    project_root: Path,
    artifact_root: Path | None,
    pointer_path: Path | None,
    trusted_root: Path | None,
) -> tuple[Path, Path, Path, Path]:
    publication = policy["publication"]
    selected_root = (
        project_root / str(publication["artifact_root"])
        if artifact_root is None
        else Path(artifact_root)
    ).absolute()
    directory_key = "build_directory" if kind == "primary" else "verification_directory"
    destination = selected_root / str(publication[directory_key]) / identifier
    pointer_key = "primary_pointer" if kind == "primary" else "verified_pointer"
    selected_pointer = (
        selected_root / Path(str(publication[pointer_key])).name
        if pointer_path is None
        else Path(pointer_path).absolute()
    )
    secure_root = (
        Path(trusted_root).absolute()
        if trusted_root is not None
        else (project_root.absolute() if artifact_root is None else selected_root)
    )
    try:
        destination.relative_to(secure_root)
        selected_pointer.relative_to(secure_root)
    except ValueError as exc:
        raise Stage11PublicationError(
            "Stage 11 publication path escaped its trusted root"
        ) from exc
    return selected_root, destination, selected_pointer, secure_root


def _write_bundle(
    *,
    trusted_root: Path,
    destination: Path,
    members: Sequence[str],
    payloads: Mapping[str, bytes],
    label: str,
    gate: Any,
) -> PublishedBundle:
    _require(set(payloads) == set(members), f"{label} payload membership differs")
    hashes = {name: _sha256_bytes(payloads[name]) for name in members}
    with SecureStagingDirectory.create(
        trusted_root=trusted_root,
        destination=destination,
        expected_files=members,
        label=label,
    ) as stage:
        for name in members:
            payload = payloads[name]
            with stage.open_exclusive_binary(name) as stream:
                written = stream.write(payload)
                _require(written == len(payload), f"Short Stage 11 write: {name}")
        snapshot = stage.snapshot(expected_hashes=hashes)
        return stage.publish(
            expected_hashes=snapshot.hashes,
            pre_publish_gate=gate,
            post_publish_gate=gate,
        )


def _primary_authorizations() -> dict[str, bool]:
    return {
        "independent_verification_authorized": True,
        "independent_verification_completed": False,
        "validation_outcome_or_forward_return_access_authorized": False,
        "real_validation_weights_or_trades_authorized": False,
        "real_validation_pnl_or_performance_authorized": False,
        "final_2020_2025_access_authorized": False,
        "stage12_or_later_authorized": False,
    }


def _verified_authorizations() -> dict[str, bool]:
    return {
        "independent_verification_completed": True,
        "no_outcome_preflight_technical_gate_passed": True,
        "validation_unblind_requires_separate_approval": True,
        "validation_outcome_or_forward_return_access_authorized_now": False,
        "real_validation_weights_or_trades_authorized_now": False,
        "real_validation_pnl_or_performance_authorized_now": False,
        "final_2020_2025_access_authorized": False,
        "stage12_or_later_authorized": False,
    }


def publish_primary_preflight(
    *,
    summary: PreflightRunSummary,
    checks: Sequence[PreflightCheck],
    evidence: Sequence[PreflightEvidenceDigest],
    access_broker: PreflightAccessBroker,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
    artifact_root: Path | None = None,
    pointer_path: Path | None = None,
    trusted_root: Path | None = None,
) -> PrimaryPreflightPublication:
    """Publish the deterministic Stage 11 no-outcome primary preflight bundle."""

    policy = load_policy(policy_path, project_root=project_root)
    members = _configured_members(policy, kind="primary")
    publication_policy = policy["publication"]
    _require(
        publication_policy.get("pointer_replaced_last") is True
        and publication_policy.get("outcome_or_performance_payload_permitted") is False,
        "Stage 11 publication boundary differs",
    )
    safe_report = _normalize_summary(summary)
    safe_evidence = _normalize_evidence(evidence, label="preflight evidence")
    authenticated_counters, initial_access_ledger = _contract_counters_from_broker(
        access_broker
    )
    safe_counters = _validate_counters(authenticated_counters, policy=policy)
    receipt = authenticate_stage11_preflight_contract(
        policy_path,
        project_root=project_root,
        counters=authenticated_counters,
    )
    check_rows = _normalize_checks(checks, label="primary preflight")
    checks_payload = _checks_bytes(check_rows)
    _validate_check_payload(checks_payload, label="primary preflight")
    evidence_payload = canonical_json_bytes(list(safe_evidence))
    evidence_sha = _sha256_bytes(evidence_payload)
    authority_hashes = _authority_hashes(receipt)
    code_hashes = _hash_code_paths(
        DEFAULT_IMPLEMENTATION_CODE_PATHS, project_root=project_root
    )
    report_value = {
        "schema_version": 1,
        "stage": 11,
        "status": "NO_OUTCOME_PREFLIGHT_PASSED",
        "scope": "stage11_implementation_no_outcome_preflight",
        "evidence": list(safe_evidence),
        "evidence_sha256": evidence_sha,
        "counters": safe_counters,
        "summary": safe_report,
        "outcome_or_performance_payload_members": 0,
    }
    report_payload = canonical_json_bytes(report_value)
    report_sha = _sha256_bytes(report_payload)
    checks_sha = _sha256_bytes(checks_payload)
    fingerprint_value = {
        "schema_version": 1,
        "purpose": "stage11_no_outcome_preflight_primary",
        "contract_receipt_sha256": receipt.sha256,
        "authority_hashes": authority_hashes,
        "code_hashes": code_hashes,
        "evidence_sha256": evidence_sha,
        "preflight_report_sha256": report_sha,
        "preflight_checks_sha256": checks_sha,
        "counters_sha256": _sha256_bytes(canonical_json_bytes(safe_counters)),
        "closed_members": list(members),
    }
    fingerprint = _sha256_bytes(canonical_json_bytes(fingerprint_value))
    prefix = str(publication_policy["build_prefix"])
    build_id = f"{prefix}{fingerprint[:24]}"
    _require(
        re.fullmatch(re.escape(prefix) + r"[0-9a-f]{24}", build_id) is not None,
        "Malformed Stage 11 preflight build ID",
    )
    authorizations = _primary_authorizations()
    manifest = {
        "schema_version": 1,
        "stage": 11,
        "contract_id": receipt.contract_id,
        "policy_id": receipt.policy_id,
        "rules_id": receipt.rules_id,
        "build_id": build_id,
        "status": PRIMARY_STATUS,
        "scope": "NO_OUTCOME_PREFLIGHT_ONLY",
        "authorizations": authorizations,
        "bindings": {
            "content_fingerprint": fingerprint,
            "contract_receipt_sha256": receipt.sha256,
            "authority_hashes": authority_hashes,
            "code_hashes": code_hashes,
            "evidence_sha256": evidence_sha,
            "preflight_report_sha256": report_sha,
            "preflight_checks_sha256": checks_sha,
            "counters_sha256": fingerprint_value["counters_sha256"],
        },
        "upstream": {
            "stage10_build_id": receipt.stage10_build_id,
            "stage10_verification_id": receipt.stage10_verification_id,
            "stage10_verified_pointer_sha256": receipt.stage10_verified_pointer_sha256,
            "stage10_primary_manifest_sha256": receipt.stage10_primary_manifest_sha256,
        },
        "contract_receipt": receipt.to_dict(),
        "counters": safe_counters,
        "files": {
            PREFLIGHT_REPORT_NAME: {
                "sha256": report_sha,
                "media_type": "application/json",
            },
            PREFLIGHT_CHECKS_NAME: {
                "sha256": checks_sha,
                "media_type": "application/vnd.apache.parquet",
                "rows": len(check_rows),
            },
        },
        "outcome_or_performance_payload_members": 0,
    }
    manifest_payload = canonical_json_bytes(manifest)
    manifest_sha = _sha256_bytes(manifest_payload)
    success_value = {
        "schema_version": 1,
        "stage": 11,
        "build_id": build_id,
        "status": "PREFLIGHT_PRIMARY_BUNDLE_COMPLETE",
        "manifest_sha256": manifest_sha,
        "preflight_report_sha256": report_sha,
        "preflight_checks_sha256": checks_sha,
        "closed_files": list(members),
        "outcome_or_performance_payload_members": 0,
    }
    success_payload = canonical_json_bytes(success_value)
    payloads = {
        MANIFEST_NAME: manifest_payload,
        PREFLIGHT_REPORT_NAME: report_payload,
        PREFLIGHT_CHECKS_NAME: checks_payload,
        SUCCESS_NAME: success_payload,
    }
    _, destination, selected_pointer, secure_root = _select_paths(
        policy,
        kind="primary",
        identifier=build_id,
        project_root=project_root,
        artifact_root=artifact_root,
        pointer_path=pointer_path,
        trusted_root=trusted_root,
    )

    def reauthenticate() -> None:
        current = authenticate_stage11_preflight_contract(
            policy_path,
            project_root=project_root,
            counters=authenticated_counters,
        )
        _require(current.sha256 == receipt.sha256, "Stage 11 contract changed")
        _require(
            _hash_code_paths(
                DEFAULT_IMPLEMENTATION_CODE_PATHS, project_root=project_root
            )
            == code_hashes,
            "Stage 11 implementation code changed",
        )
        _require(
            access_broker.ledger() == initial_access_ledger,
            "Stage 11 access ledger changed during publication",
        )

    bundle = _write_bundle(
        trusted_root=secure_root,
        destination=destination,
        members=members,
        payloads=payloads,
        label="Stage 11 no-outcome primary preflight",
        gate=reauthenticate,
    )
    member_hashes = {name: _sha256_bytes(payloads[name]) for name in members}
    pointer = {
        "schema_version": 1,
        "stage": 11,
        "contract_id": receipt.contract_id,
        "policy_id": receipt.policy_id,
        "rules_id": receipt.rules_id,
        "build_id": build_id,
        "status": PRIMARY_STATUS,
        "bundle_path": destination.relative_to(secure_root).as_posix(),
        "bundle_files": member_hashes,
        "primary_manifest_sha256": manifest_sha,
        "contract_receipt_sha256": receipt.sha256,
        "authorizations": authorizations,
    }
    pointer_payload = canonical_json_bytes(pointer)
    pointer_replaced = replace_pointer_last(
        selected_pointer,
        pointer_payload,
        trusted_root=secure_root,
        pre_replace_gate=reauthenticate,
        post_replace_gate=reauthenticate,
        label="Stage 11 latest primary preflight pointer",
    )
    return PrimaryPreflightPublication(
        build_id=build_id,
        bundle_path=destination,
        manifest_path=destination / MANIFEST_NAME,
        pointer_path=selected_pointer,
        manifest=manifest,
        pointer=pointer,
        bundle=bundle,
        reused_existing=bundle.reused_existing and not pointer_replaced,
    )


@dataclass
class _AuthenticatedPrimary:
    pointer_guard: AuthenticatedPointerGuard
    bundle_guard: AuthenticatedBundleGuard
    pointer: Mapping[str, Any]
    manifest: Mapping[str, Any]
    pointer_sha256: str
    manifest_sha256: str
    build_id: str
    checks: tuple[dict[str, str], ...]
    evidence: tuple[dict[str, object], ...]
    counters: dict[str, Any]
    summary: dict[str, object]

    def gate(self) -> None:
        self.pointer_guard.gate()
        self.bundle_guard.gate()


@contextlib.contextmanager
def _authenticate_primary(
    *,
    pointer_path: Path,
    trusted_root: Path,
    members: Sequence[str],
    receipt: Stage11PreflightContractReceipt,
    expected_code_hashes: Mapping[str, str],
) -> Iterator[_AuthenticatedPrimary]:
    with contextlib.ExitStack() as stack:
        pointer_guard = stack.enter_context(
            AuthenticatedPointerGuard.open(
                trusted_root=trusted_root,
                path=pointer_path,
                label="Stage 11 primary preflight pointer",
            )
        )
        pointer = _json(pointer_guard.payload, label="primary preflight pointer")
        _require(
            set(pointer)
            == {
                "schema_version",
                "stage",
                "contract_id",
                "policy_id",
                "rules_id",
                "build_id",
                "status",
                "bundle_path",
                "bundle_files",
                "primary_manifest_sha256",
                "contract_receipt_sha256",
                "authorizations",
            },
            "Primary pointer fields differ from the closed schema",
        )
        _require(
            pointer.get("schema_version") == 1
            and pointer.get("stage") == 11
            and pointer.get("contract_id") == receipt.contract_id
            and pointer.get("policy_id") == receipt.policy_id
            and pointer.get("rules_id") == receipt.rules_id
            and pointer.get("status") == PRIMARY_STATUS,
            "Primary pointer identity/status differs",
        )
        primary_receipt_sha = pointer.get("contract_receipt_sha256")
        _require(
            isinstance(primary_receipt_sha, str)
            and _SHA256.fullmatch(primary_receipt_sha) is not None,
            "Primary contract receipt hash is malformed",
        )
        _require(
            pointer.get("authorizations") == _primary_authorizations(),
            "Primary authorizations differ",
        )
        hashes = pointer.get("bundle_files")
        _require(
            isinstance(hashes, Mapping) and set(hashes) == set(members),
            "Primary pointer closed-set hashes differ",
        )
        _require(
            all(
                isinstance(value, str) and _SHA256.fullmatch(value) is not None
                for value in hashes.values()
            ),
            "Primary pointer contains a malformed member hash",
        )
        bundle_path = trusted_root / _safe_relative(
            pointer.get("bundle_path"), label="primary bundle"
        )
        bundle_guard = stack.enter_context(
            AuthenticatedBundleGuard.open(
                trusted_root=trusted_root,
                path=bundle_path,
                expected_files=members,
                expected_hashes={str(key): str(value) for key, value in hashes.items()},
                label="Stage 11 primary preflight bundle",
                _permit_authenticated_directory_ctime_refresh=True,
            )
        )
        with bundle_guard.open_binary(MANIFEST_NAME) as stream:
            manifest_payload = stream.read()
        manifest_sha = _sha256_bytes(manifest_payload)
        _require(
            manifest_sha == pointer.get("primary_manifest_sha256"),
            "Primary manifest hash differs",
        )
        manifest = _json(manifest_payload, label="primary preflight manifest")
        _require(
            set(manifest)
            == {
                "schema_version",
                "stage",
                "contract_id",
                "policy_id",
                "rules_id",
                "build_id",
                "status",
                "scope",
                "authorizations",
                "bindings",
                "upstream",
                "contract_receipt",
                "counters",
                "files",
                "outcome_or_performance_payload_members",
            },
            "Primary manifest fields differ from the closed schema",
        )
        _require(
            manifest.get("schema_version") == 1
            and manifest.get("stage") == 11
            and manifest.get("contract_id") == receipt.contract_id
            and manifest.get("policy_id") == receipt.policy_id
            and manifest.get("rules_id") == receipt.rules_id
            and manifest.get("build_id") == pointer.get("build_id")
            and manifest.get("status") == PRIMARY_STATUS
            and manifest.get("scope") == "NO_OUTCOME_PREFLIGHT_ONLY",
            "Primary manifest identity differs",
        )
        _require(
            manifest.get("authorizations") == _primary_authorizations(),
            "Primary manifest authorizations differ",
        )
        _require(
            manifest.get("outcome_or_performance_payload_members") == 0,
            "Primary manifest reports prohibited payload members",
        )
        embedded_receipt = manifest.get("contract_receipt")
        expected_receipt = receipt.to_dict()
        _require(
            isinstance(embedded_receipt, Mapping)
            and set(embedded_receipt) == set(expected_receipt)
            and _sha256_bytes(canonical_json_bytes(embedded_receipt))
            == primary_receipt_sha,
            "Primary embedded contract receipt is malformed",
        )
        embedded_without_counters = dict(embedded_receipt)
        expected_without_counters = dict(expected_receipt)
        embedded_counter_value = embedded_without_counters.pop("counters", None)
        expected_without_counters.pop("counters", None)
        _require(
            embedded_without_counters == expected_without_counters,
            "Primary contract authority differs",
        )
        _require(
            isinstance(embedded_counter_value, Mapping)
            and set(embedded_counter_value)
            == set(Stage11PreflightCounters.__dataclass_fields__),
            "Primary embedded contract counters are malformed",
        )
        try:
            primary_contract_counters = Stage11PreflightCounters(
                **dict(embedded_counter_value)
            )
            enforce_stage11_preflight_boundary(primary_contract_counters)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise Stage11PublicationError(
                "Primary embedded contract counters violate the boundary"
            ) from exc
        counters = manifest.get("counters")
        _require(
            isinstance(counters, dict)
            and counters == primary_contract_counters.to_dict(),
            "Primary counters differ from the authenticated broker receipt",
        )
        bindings = manifest.get("bindings")
        _require(
            isinstance(bindings, Mapping)
            and set(bindings)
            == {
                "content_fingerprint",
                "contract_receipt_sha256",
                "authority_hashes",
                "code_hashes",
                "evidence_sha256",
                "preflight_report_sha256",
                "preflight_checks_sha256",
                "counters_sha256",
            }
            and bindings.get("contract_receipt_sha256") == primary_receipt_sha
            and bindings.get("authority_hashes") == _authority_hashes(receipt)
            and bindings.get("code_hashes") == dict(expected_code_hashes)
            and bindings.get("counters_sha256")
            == _sha256_bytes(canonical_json_bytes(counters)),
            "Primary manifest bindings differ",
        )
        files = manifest.get("files")
        _require(
            isinstance(files, Mapping)
            and set(files) == {PREFLIGHT_REPORT_NAME, PREFLIGHT_CHECKS_NAME},
            "Primary manifest file declarations differ",
        )
        report_declaration = files.get(PREFLIGHT_REPORT_NAME)
        checks_declaration = files.get(PREFLIGHT_CHECKS_NAME)
        _require(
            isinstance(report_declaration, Mapping)
            and set(report_declaration) == {"sha256", "media_type"}
            and report_declaration.get("media_type") == "application/json"
            and isinstance(checks_declaration, Mapping)
            and set(checks_declaration) == {"sha256", "media_type", "rows"}
            and checks_declaration.get("media_type") == "application/vnd.apache.parquet"
            and checks_declaration.get("rows") == len(REQUIRED_PREFLIGHT_CHECK_NAMES),
            "Primary manifest file declaration schema differs",
        )
        with bundle_guard.open_binary(PREFLIGHT_REPORT_NAME) as stream:
            primary_report_payload = stream.read()
        primary_report = _json(primary_report_payload, label="primary preflight report")
        _require(
            set(primary_report)
            == {
                "schema_version",
                "stage",
                "status",
                "scope",
                "evidence",
                "evidence_sha256",
                "counters",
                "summary",
                "outcome_or_performance_payload_members",
            },
            "Primary report fields differ from the closed schema",
        )
        _require(
            _sha256_bytes(primary_report_payload)
            == bindings.get("preflight_report_sha256")
            == report_declaration.get("sha256"),
            "Primary report hash binding differs",
        )
        summary_value = primary_report.get("summary")
        _require(
            isinstance(summary_value, Mapping)
            and set(summary_value)
            == {"runner_id", "implementation_mode", "required_check_count"},
            "Primary summary fields differ from the closed schema",
        )
        normalized_summary = _normalize_summary(
            PreflightRunSummary(
                runner_id=summary_value["runner_id"],  # type: ignore[arg-type]
                implementation_mode=summary_value["implementation_mode"],  # type: ignore[arg-type]
                required_check_count=summary_value["required_check_count"],  # type: ignore[arg-type]
            )
        )
        evidence_value = primary_report.get("evidence")
        _require(isinstance(evidence_value, list), "Primary evidence is not a list")
        evidence_rows: list[PreflightEvidenceDigest] = []
        for row in evidence_value:
            _require(
                isinstance(row, Mapping)
                and set(row) == {"check_name", "evidence_sha256", "observation_count"},
                "Primary evidence fields differ from the closed schema",
            )
            evidence_rows.append(
                PreflightEvidenceDigest(
                    check_name=row["check_name"],  # type: ignore[arg-type]
                    evidence_sha256=row["evidence_sha256"],  # type: ignore[arg-type]
                    observation_count=row["observation_count"],  # type: ignore[arg-type]
                )
            )
        normalized_evidence = _normalize_evidence(
            evidence_rows, label="authenticated primary evidence"
        )
        _require(
            primary_report.get("evidence_sha256")
            == _sha256_bytes(canonical_json_bytes(list(normalized_evidence)))
            == bindings.get("evidence_sha256"),
            "Primary evidence hash binding differs",
        )
        _require(
            primary_report.get("counters") == counters
            and primary_report.get("outcome_or_performance_payload_members") == 0,
            "Primary report counters or payload boundary differs",
        )
        with bundle_guard.open_binary(PREFLIGHT_CHECKS_NAME) as stream:
            primary_checks_payload = stream.read()
        primary_checks = _validate_check_payload(
            primary_checks_payload, label="primary preflight"
        )
        _require(
            _sha256_bytes(primary_checks_payload)
            == bindings.get("preflight_checks_sha256")
            == checks_declaration.get("sha256"),
            "Primary check hash binding differs",
        )
        authenticated = _AuthenticatedPrimary(
            pointer_guard=pointer_guard,
            bundle_guard=bundle_guard,
            pointer=pointer,
            manifest=manifest,
            pointer_sha256=pointer_guard.payload_sha256,
            manifest_sha256=manifest_sha,
            build_id=str(pointer["build_id"]),
            checks=primary_checks,
            evidence=normalized_evidence,
            counters=dict(counters),
            summary=normalized_summary,
        )
        authenticated.gate()
        yield authenticated
        authenticated.gate()


def publish_independent_verification(
    *,
    primary_pointer_path: Path,
    summary: PreflightRunSummary,
    checks: Sequence[PreflightCheck],
    evidence: Sequence[PreflightEvidenceDigest],
    access_broker: PreflightAccessBroker,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
    artifact_root: Path | None = None,
    verified_pointer_path: Path | None = None,
    trusted_root: Path | None = None,
) -> VerifiedPreflightPublication:
    """Compare an independent rerun to primary evidence, then publish it."""

    policy = load_policy(policy_path, project_root=project_root)
    primary_members = _configured_members(policy, kind="primary")
    members = _configured_members(policy, kind="verification")
    safe_report = _normalize_summary(summary)
    safe_evidence = _normalize_evidence(evidence, label="verification evidence")
    authenticated_counters, initial_access_ledger = _contract_counters_from_broker(
        access_broker
    )
    safe_counters = _validate_counters(authenticated_counters, policy=policy)
    receipt = authenticate_stage11_preflight_contract(
        policy_path,
        project_root=project_root,
        counters=authenticated_counters,
    )
    check_rows = _normalize_checks(checks, label="independent verification")
    checks_payload = _checks_bytes(check_rows)
    _validate_check_payload(checks_payload, label="independent verification")
    evidence_payload = canonical_json_bytes(list(safe_evidence))
    evidence_sha = _sha256_bytes(evidence_payload)
    authority_hashes = _authority_hashes(receipt)
    code_hashes = _hash_code_paths(
        DEFAULT_IMPLEMENTATION_CODE_PATHS, project_root=project_root
    )
    secure_root = (
        Path(trusted_root).absolute()
        if trusted_root is not None
        else (
            project_root.absolute()
            if artifact_root is None
            else Path(artifact_root).absolute()
        )
    )
    authorizations = _verified_authorizations()
    with _authenticate_primary(
        pointer_path=Path(primary_pointer_path).absolute(),
        trusted_root=secure_root,
        members=primary_members,
        receipt=receipt,
        expected_code_hashes=code_hashes,
    ) as primary:
        computed_differences: list[str] = []
        if check_rows != primary.checks:
            computed_differences.append("required_checks")
        if safe_evidence != primary.evidence:
            computed_differences.append("evidence_digests")
        if safe_counters != primary.counters:
            computed_differences.append("access_counters")
        _require(
            not computed_differences,
            "Independent verification found recomputed differences: "
            + ", ".join(computed_differences),
        )
        report_value = {
            "schema_version": 1,
            "stage": 11,
            "build_id": primary.build_id,
            "status": VERIFIED_STATUS,
            "scope": "NO_OUTCOME_PREFLIGHT_ONLY",
            "difference_count": 0,
            "authorizations": authorizations,
            "primary_pointer_sha256": primary.pointer_sha256,
            "primary_manifest_sha256": primary.manifest_sha256,
            "bindings": {
                "contract_receipt_sha256": receipt.sha256,
                "authority_hashes": authority_hashes,
                "code_hashes": code_hashes,
            },
            "evidence": list(safe_evidence),
            "evidence_sha256": evidence_sha,
            "counters": safe_counters,
            "summary": safe_report,
            "outcome_or_performance_payload_members": 0,
        }
        report_payload = canonical_json_bytes(report_value)
        checks_sha = _sha256_bytes(checks_payload)
        report_sha = _sha256_bytes(report_payload)
        differences_value = {
            "schema_version": 1,
            "difference_count": 0,
            "differences": [],
        }
        differences_payload = canonical_json_bytes(differences_value)
        differences_sha = _sha256_bytes(differences_payload)
        fingerprint_value = {
            "schema_version": 1,
            "purpose": "stage11_no_outcome_preflight_independent_verification",
            "build_id": primary.build_id,
            "primary_pointer_sha256": primary.pointer_sha256,
            "primary_manifest_sha256": primary.manifest_sha256,
            "contract_receipt_sha256": receipt.sha256,
            "authority_hashes": authority_hashes,
            "code_hashes": code_hashes,
            "evidence_sha256": evidence_sha,
            "verification_report_sha256": report_sha,
            "verification_checks_sha256": checks_sha,
            "differences_sha256": differences_sha,
            "counters_sha256": _sha256_bytes(canonical_json_bytes(safe_counters)),
            "closed_members": list(members),
        }
        fingerprint = _sha256_bytes(canonical_json_bytes(fingerprint_value))
        prefix = str(policy["publication"]["verification_prefix"])
        verification_id = f"{prefix}{fingerprint[:16]}"
        _require(
            re.fullmatch(re.escape(prefix) + r"[0-9a-f]{16}", verification_id)
            is not None,
            "Malformed Stage 11 verification ID",
        )
        success_value = {
            "schema_version": 1,
            "stage": 11,
            "build_id": primary.build_id,
            "verification_id": verification_id,
            "status": "PREFLIGHT_VERIFICATION_BUNDLE_COMPLETE",
            "verification_report_sha256": report_sha,
            "verification_checks_sha256": checks_sha,
            "differences_sha256": differences_sha,
            "closed_files": list(members),
            "outcome_or_performance_payload_members": 0,
        }
        success_payload = canonical_json_bytes(success_value)
        payloads = {
            VERIFICATION_REPORT_NAME: report_payload,
            VERIFICATION_CHECKS_NAME: checks_payload,
            DIFFERENCES_NAME: differences_payload,
            SUCCESS_NAME: success_payload,
        }
        _, destination, selected_pointer, selected_secure_root = _select_paths(
            policy,
            kind="verification",
            identifier=verification_id,
            project_root=project_root,
            artifact_root=artifact_root,
            pointer_path=verified_pointer_path,
            trusted_root=trusted_root,
        )
        _require(
            selected_secure_root == secure_root,
            "Primary and verification trusted roots differ",
        )

        def reauthenticate() -> None:
            primary.gate()
            current = authenticate_stage11_preflight_contract(
                policy_path,
                project_root=project_root,
                counters=authenticated_counters,
            )
            _require(current.sha256 == receipt.sha256, "Stage 11 contract changed")
            _require(
                _hash_code_paths(
                    DEFAULT_IMPLEMENTATION_CODE_PATHS, project_root=project_root
                )
                == code_hashes,
                "Stage 11 verification code changed",
            )
            _require(
                access_broker.ledger() == initial_access_ledger,
                "Stage 11 independent access ledger changed during publication",
            )

        bundle = _write_bundle(
            trusted_root=secure_root,
            destination=destination,
            members=members,
            payloads=payloads,
            label="Stage 11 independent no-outcome preflight verification",
            gate=reauthenticate,
        )
        member_hashes = {name: _sha256_bytes(payloads[name]) for name in members}
        pointer = {
            "schema_version": 1,
            "stage": 11,
            "contract_id": receipt.contract_id,
            "policy_id": receipt.policy_id,
            "rules_id": receipt.rules_id,
            "build_id": primary.build_id,
            "verification_id": verification_id,
            "status": VERIFIED_STATUS,
            "content_fingerprint": fingerprint,
            "bundle_path": destination.relative_to(secure_root).as_posix(),
            "bundle_files": member_hashes,
            "verification_report_sha256": report_sha,
            "primary_pointer_sha256": primary.pointer_sha256,
            "primary_manifest_sha256": primary.manifest_sha256,
            "contract_receipt_sha256": receipt.sha256,
            "difference_count": 0,
            "authorizations": authorizations,
        }
        pointer_payload = canonical_json_bytes(pointer)
        pointer_replaced = replace_pointer_last(
            selected_pointer,
            pointer_payload,
            trusted_root=secure_root,
            pre_replace_gate=reauthenticate,
            post_replace_gate=reauthenticate,
            label="Stage 11 latest verified preflight pointer",
        )
        return VerifiedPreflightPublication(
            build_id=primary.build_id,
            verification_id=verification_id,
            bundle_path=destination,
            report_path=destination / VERIFICATION_REPORT_NAME,
            pointer_path=selected_pointer,
            report=report_value,
            pointer=pointer,
            bundle=bundle,
            reused_existing=bundle.reused_existing and not pointer_replaced,
        )


# Descriptive aliases for callers that prefer the fully qualified operation.
publish_primary_preflight_evidence = publish_primary_preflight
publish_independent_preflight_verification = publish_independent_verification


__all__ = [
    "CHECK_SCHEMA",
    "DEFAULT_IMPLEMENTATION_CODE_PATHS",
    "DIFFERENCES_NAME",
    "MANIFEST_NAME",
    "PREFLIGHT_CHECKS_NAME",
    "PREFLIGHT_REPORT_NAME",
    "PRIMARY_FILES",
    "PRIMARY_STATUS",
    "PreflightCheck",
    "PreflightEvidenceDigest",
    "PreflightRunSummary",
    "PrimaryPreflightPublication",
    "Stage11PublicationError",
    "SUCCESS_NAME",
    "VERIFICATION_CHECKS_NAME",
    "VERIFICATION_REPORT_NAME",
    "VERIFICATION_FILES",
    "VERIFIED_STATUS",
    "VerifiedPreflightPublication",
    "publish_independent_preflight_verification",
    "publish_independent_verification",
    "publish_primary_preflight",
    "publish_primary_preflight_evidence",
    "reject_forbidden_payload",
]
