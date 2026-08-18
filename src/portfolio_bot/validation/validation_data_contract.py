"""Fail-closed contract for the Stage 11 validation data extension.

This module authenticates only small frozen authority files and the verified
calendar.  It deliberately does not open raw security data, labels, forward
returns, validation outcomes, portfolios, trades, or P&L.  The historical
market payload materializer is a separate, still-sealed operation.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.features.feature_store_contract import (
    LINEAGE_SCHEMA,
    expanded_features,
    raw_feature_schema,
    validate_feature_store_contract,
)
from portfolio_bot.risk_cost.contract import (
    FACTOR_NAMES,
    published_schemas as stage8_published_schemas,
    validate_contract as validate_stage8_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "configs/phase_05_validation/stage_11_validation_data_policy_v1.toml"
)

POLICY_ID = "phase05-stage11-validation-data-policy-v1"
CONTRACT_ID = "phase05-stage11-validation-data-v1"
RULES_ID = "phase05-stage11-validation-data-rules-v1"

INTERNAL_CONTEXT_SIGNAL = dt.date(2014, 11, 28)
FIRST_SIGNAL = dt.date(2014, 12, 31)
LAST_SIGNAL = dt.date(2019, 10, 31)
FIRST_PROHIBITED_SIGNAL = dt.date(2019, 11, 29)
FIRST_FINAL_ENTRY = dt.date(2020, 1, 2)
REQUIRED_SIGNAL_COUNT = 59
REQUIRED_FEATURE_COUNT = 163
REQUIRED_PHYSICAL_COLUMNS = 170
REQUIRED_FACTOR_COVARIANCE_ROWS = 59 * len(FACTOR_NAMES) ** 2

AUDIT_COLUMN_NAMES = (
    "signal_date",
    "permno",
    "permco",
    "long_eligible",
    "short_eligible",
    "primary_exchange",
    "share_type",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_EXACT = frozenset(
    {
        "forward_return",
        "future_return",
        "residual_return",
        "realized_return",
        "outcome",
        "target",
        "label",
        "portfolio_weight",
        "trade",
        "pnl",
        "performance",
    }
)
_FORBIDDEN_TOKENS = (
    "forward_return",
    "future_return",
    "realized_return",
    "residual_return",
    "outcome",
    "target",
    "label",
    "pnl",
    "performance",
)


class ValidationDataContractError(RuntimeError):
    """Raised when the Stage 11 validation-data contract fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationDataContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _parquet_physical_schema(schema: pa.Schema) -> pa.Schema:
    """Return the exact schema Arrow reconstructs from a Parquet footer."""

    fields: list[pa.Field] = []
    for field in schema:
        field_type = field.type
        if pa.types.is_list(field_type):
            field_type = pa.list_(
                pa.field(
                    "element",
                    field_type.value_type,
                    nullable=field_type.value_field.nullable,
                )
            )
        fields.append(
            pa.field(
                field.name,
                field_type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        )
    return pa.schema(fields, metadata=schema.metadata)


def canonical_json_bytes(value: Any) -> bytes:
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


def _project_file(project_root: Path, value: object, expected_hash: object, label: str) -> Path:
    relative = PurePosixPath(str(value))
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"Unsafe {label} path",
    )
    path = project_root.joinpath(*relative.parts)
    require(path.is_file() and not path.is_symlink(), f"Missing or unsafe {label}")
    expected = str(expected_hash)
    require(_SHA256.fullmatch(expected) is not None, f"Invalid {label} SHA-256")
    require(sha256_file(path) == expected, f"{label} SHA-256 differs")
    return path


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationDataContractError(f"Invalid {label} JSON") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _date(value: object, label: str) -> dt.date:
    try:
        result = dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValidationDataContractError(f"Invalid {label}") from exc
    return result


@dataclass(frozen=True)
class ValidationSignal:
    holding_month: str
    signal_date: dt.date
    entry_date: dt.date
    exit_date: dt.date


@dataclass(frozen=True)
class ValidationPayloadSpec:
    role: str
    filename: str
    schema: pa.Schema
    cardinality_rule: str

    @property
    def schema_sha256(self) -> str:
        return schema_sha256(self.schema)


@dataclass(frozen=True)
class ValidationDataContract:
    policy_path: Path
    policy_sha256: str
    policy: Mapping[str, Any]
    rules: Mapping[str, Any]
    validation_signals: tuple[ValidationSignal, ...]
    raw_feature_schema: pa.Schema
    lineage_schema: pa.Schema
    feature_names: tuple[str, ...]
    family_feature_names: Mapping[str, tuple[str, ...]]
    payload_specs: Mapping[str, ValidationPayloadSpec]
    frozen_bindings: Mapping[str, str]

    @property
    def scored_signal_dates(self) -> tuple[dt.date, ...]:
        return tuple(value.signal_date for value in self.validation_signals)


def _calendar_path(
    project_root: Path,
    policy: Mapping[str, Any],
    rules: Mapping[str, Any],
) -> Path:
    root = Path(str(policy["database"]["root"])).expanduser().resolve()
    require(root.is_dir() and not root.is_symlink(), "Unsafe validation database root")
    relative = PurePosixPath(str(rules["calendar"]["rebalance_calendar_relative_path"]))
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Unsafe rebalance-calendar path",
    )
    path = root.joinpath(*relative.parts)
    require(path.is_file() and not path.is_symlink(), "Missing verified rebalance calendar")
    require(
        sha256_file(path) == rules["calendar"]["rebalance_calendar_sha256"],
        "Rebalance-calendar SHA-256 differs",
    )
    return path


def _validation_calendar(path: Path, rules: Mapping[str, Any]) -> tuple[ValidationSignal, ...]:
    required_columns = (
        "holding_month",
        "stage",
        "signal_date",
        "entry_date",
        "exit_date",
        "is_embargo",
        "is_partial",
        "full_feature_time_eligible",
    )
    table = pq.read_table(path, columns=list(required_columns), use_threads=False)
    stage = str(rules["calendar"]["validation_stage_value"])
    selected = table.filter(pc.equal(table["stage"], pa.scalar(stage)))
    values = selected.to_pylist()
    require(len(values) == REQUIRED_SIGNAL_COUNT, "Validation calendar is not exactly 59 rows")
    result: list[ValidationSignal] = []
    previous: dt.date | None = None
    for row in values:
        signal = row["signal_date"]
        entry = row["entry_date"]
        exit_date = row["exit_date"]
        require(
            isinstance(signal, dt.date)
            and isinstance(entry, dt.date)
            and isinstance(exit_date, dt.date),
            "Validation calendar contains an invalid date",
        )
        require(previous is None or signal > previous, "Validation signals are not sorted unique")
        require(signal < entry < exit_date, "Validation forecast does not precede entry and exit")
        require(
            row["is_embargo"] is False
            and row["is_partial"] is False
            and row["full_feature_time_eligible"] is True,
            "Validation calendar contains an ineligible scored interval",
        )
        result.append(ValidationSignal(str(row["holding_month"]), signal, entry, exit_date))
        previous = signal
    require(result[0].holding_month == "2015-01", "Wrong first validation holding month")
    require(result[-1].holding_month == "2019-11", "Wrong last validation holding month")
    require(result[0].signal_date == FIRST_SIGNAL, "Wrong first validation signal")
    require(result[-1].signal_date == LAST_SIGNAL, "Wrong last validation signal")
    require(result[-1].exit_date == dt.date(2019, 12, 2), "Wrong final validation exit")
    require(
        all(value.signal_date < FIRST_PROHIBITED_SIGNAL for value in result),
        "Validation calendar includes the embargo or final period",
    )
    return tuple(result)


def _risk_schemas() -> dict[str, pa.Schema]:
    source = stage8_published_schemas()
    return {
        "validation_risk_cost_audit": source["development_risk_cost_audit.parquet"],
        "validation_risk_cost_inputs": source["development_risk_cost_inputs.parquet"],
        "validation_factor_covariance": source["development_factor_covariance.parquet"],
        "validation_risk_cost_month_summary": source[
            "development_risk_cost_month_summary.parquet"
        ],
    }


def _payload_specs(
    rules: Mapping[str, Any],
    raw_schema: pa.Schema,
    family_names: Mapping[str, tuple[str, ...]],
) -> Mapping[str, ValidationPayloadSpec]:
    role_files = rules["published_roles"]
    require(isinstance(role_files, dict), "Published role map is missing")
    specs: dict[str, ValidationPayloadSpec] = {
        "validation_predictor_payload": ValidationPayloadSpec(
            "validation_predictor_payload",
            str(role_files["validation_predictor_payload"]),
            raw_schema,
            "empirical_nonempty_rows_with_exact_59_signal_dates",
        )
    }
    for family, names in family_names.items():
        role = f"validation_feature_lineage_{family}"
        specs[role] = ValidationPayloadSpec(
            role,
            str(role_files[role]),
            LINEAGE_SCHEMA,
            f"feature_rows_times_{len(names)}",
        )
    for role, schema in _risk_schemas().items():
        cardinality = {
            "validation_risk_cost_audit": "equals_feature_rows",
            "validation_risk_cost_inputs": "equals_long_eligible_feature_rows",
            "validation_factor_covariance": "exactly_2124_rows",
            "validation_risk_cost_month_summary": "exactly_59_rows",
        }[role]
        specs[role] = ValidationPayloadSpec(
            role,
            str(role_files[role]),
            schema,
            cardinality,
        )
    require(len({value.filename for value in specs.values()}) == len(specs), "Duplicate payload filename")
    return MappingProxyType(specs)


def _authenticate_pointer(
    payload: Mapping[str, Any],
    *,
    build_id: str,
    verification_id: str,
    label: str,
) -> None:
    require(payload.get("build_id") == build_id, f"Wrong {label} build ID")
    require(payload.get("verification_id") == verification_id, f"Wrong {label} verification ID")
    require(str(payload.get("status", "")).lower() == "verified_passed", f"{label} is not verified")


def unexpected_forbidden_columns(
    columns: tuple[str, ...] | list[str],
    *,
    allowed_columns: tuple[str, ...] | list[str] = (),
) -> tuple[str, ...]:
    """Return forbidden names outside a frozen allowlist.

    The allowlist matters because the frozen feature ``beta_label_*`` names are
    legitimate predictors; validation cannot add any *new* label-like field.
    """

    allowed = frozenset(allowed_columns)
    found: list[str] = []
    for column in columns:
        if column in allowed:
            continue
        normalized = column.strip().lower()
        if normalized in _FORBIDDEN_EXACT or any(token in normalized for token in _FORBIDDEN_TOKENS):
            found.append(column)
    return tuple(found)


def validate_validation_data_contract(
    policy_path: Path = DEFAULT_POLICY,
    *,
    project_root: Path = PROJECT_ROOT,
) -> ValidationDataContract:
    """Authenticate the no-outcome validation data-extension contract."""

    root = project_root.resolve()
    resolved_policy = policy_path.resolve()
    require(resolved_policy.is_file() and not resolved_policy.is_symlink(), "Unsafe validation-data policy")
    with resolved_policy.open("rb") as handle:
        policy = tomllib.load(handle)
    require(
        policy.get("schema_version") == 1
        and policy.get("stage") == 11
        and policy.get("policy_id") == POLICY_ID
        and policy.get("contract_id") == CONTRACT_ID,
        "Wrong validation-data policy identity",
    )
    require(
        policy.get("status") == "frozen_scaffold_before_validation_data_materialization",
        "Validation-data policy is not frozen at the scaffold boundary",
    )

    rules_path = _project_file(
        root,
        policy["physical_rules"]["path"],
        policy["physical_rules"]["sha256"],
        "validation-data rules",
    )
    rules = _json(rules_path, "validation-data rules")
    require(rules.get("rules_id") == RULES_ID, "Wrong validation-data rules identity")
    require(
        rules.get("status") == "frozen_scaffold_before_validation_data_materialization",
        "Validation-data rules are not frozen at the scaffold boundary",
    )

    bindings: dict[str, str] = {}
    for section, entries in (
        ("scientific_authority", (("methodology_source", "methodology_source_sha256"), ("methodology_freeze", "methodology_freeze_sha256"))),
        ("preflight_authority", (("policy", "policy_sha256"), ("approval", "approval_sha256"))),
        ("frozen_stage06", (("policy", "policy_sha256"), ("registry", "registry_sha256"), ("verified_pointer", "verified_pointer_sha256"))),
        ("frozen_stage08", (("policy", "policy_sha256"), ("rules", "rules_sha256"), ("verified_pointer", "verified_pointer_sha256"))),
        ("verified_calendar", (("verified_pointer", "verified_pointer_sha256"),)),
        ("database", (("paths_config", "paths_config_sha256"), ("relocation_receipt", "relocation_receipt_sha256"))),
    ):
        for path_key, hash_key in entries:
            path = _project_file(root, policy[section][path_key], policy[section][hash_key], f"{section} {path_key}")
            bindings[path.relative_to(root).as_posix()] = str(policy[section][hash_key])

    approval = _json(root / policy["preflight_authority"]["approval"], "preflight approval")
    require(approval.get("approved") is True, "Validation predictor preflight approval is closed")
    require(
        approval.get("scope") == policy["preflight_authority"]["required_scope"],
        "Preflight approval scope differs",
    )
    require(
        policy["preflight_authority"]["validation_outcome_access_authorized"] is False,
        "Validation outcome access was unexpectedly authorized",
    )

    stage6 = policy["frozen_stage06"]
    stage6_pointer = _json(root / stage6["verified_pointer"], "Stage 6 pointer")
    _authenticate_pointer(
        stage6_pointer,
        build_id=str(stage6["required_build_id"]),
        verification_id=str(stage6["required_verification_id"]),
        label="Stage 6 feature store",
    )
    stage6_contract = validate_feature_store_contract(root / stage6["policy"])

    stage8 = policy["frozen_stage08"]
    stage8_pointer = _json(root / stage8["verified_pointer"], "Stage 8 pointer")
    _authenticate_pointer(
        stage8_pointer,
        build_id=str(stage8["required_build_id"]),
        verification_id=str(stage8["required_verification_id"]),
        label="Stage 8 risk/cost store",
    )
    validate_stage8_contract(root / stage8["policy"])

    calendar_binding = policy["verified_calendar"]
    calendar_pointer = _json(root / calendar_binding["verified_pointer"], "calendar pointer")
    _authenticate_pointer(
        calendar_pointer,
        build_id=str(calendar_binding["required_build_id"]),
        verification_id=str(calendar_binding["required_verification_id"]),
        label="calendar",
    )

    scope = policy["scope"]
    require(_date(scope["internal_context_signal_date"], "context signal") == INTERNAL_CONTEXT_SIGNAL, "Context signal changed")
    require(_date(scope["first_signal_date"], "first signal") == FIRST_SIGNAL, "First signal changed")
    require(_date(scope["last_signal_date"], "last signal") == LAST_SIGNAL, "Last signal changed")
    require(scope["required_signal_dates"] == REQUIRED_SIGNAL_COUNT, "Signal count changed")
    require(scope["required_economic_features"] == REQUIRED_FEATURE_COUNT, "Feature count changed")
    require(scope["required_physical_feature_columns"] == REQUIRED_PHYSICAL_COLUMNS, "Physical feature count changed")
    require(scope["required_factor_covariance_rows"] == REQUIRED_FACTOR_COVARIANCE_ROWS, "Covariance row count changed")
    require(scope["validation_predictor_access_permitted"] is True, "Predictor access is closed")
    for key in (
        "validation_outcome_or_forward_return_access_permitted",
        "validation_labels_permitted",
        "model_fitting_permitted",
        "portfolio_weights_trades_or_pnl_permitted",
        "final_2020_2025_access_permitted",
    ):
        require(scope[key] is False, f"Prohibited validation-data scope reopened: {key}")

    registry = _json(root / stage6["registry"], "frozen feature registry")
    features = tuple(expanded_features(registry))
    feature_names = tuple(str(value["name"]) for value in features)
    require(len(feature_names) == REQUIRED_FEATURE_COUNT, "Frozen feature count differs")
    logical_raw_schema = raw_feature_schema(list(features))
    require(tuple(logical_raw_schema.names[:7]) == AUDIT_COLUMN_NAMES, "Frozen audit column order differs")
    require(tuple(logical_raw_schema.names[7:]) == feature_names, "Frozen economic feature order differs")
    require(logical_raw_schema.equals(stage6_contract["raw_schema"], check_metadata=True), "Stage 6 raw schema differs")
    raw_schema = _parquet_physical_schema(logical_raw_schema)
    require(
        schema_sha256(raw_schema)
        == "e67436adbe6e2adb2f445a799f07002a272a5cdf780662aca639f0eb3c0969c3",
        "Frozen Stage 6 physical Parquet schema differs",
    )
    require(len(raw_schema) == REQUIRED_PHYSICAL_COLUMNS, "Raw feature schema is not 170 columns")

    family_order = tuple(rules["features"]["family_order"])
    require(family_order == tuple(stage6_contract["family_order"]), "Feature family order differs")
    family_feature_names = MappingProxyType(
        {
            family: tuple(str(value["name"]) for value in features if value["family"] == family)
            for family in family_order
        }
    )
    require(sum(map(len, family_feature_names.values())) == REQUIRED_FEATURE_COUNT, "Family feature partition differs")

    calendar_path = _calendar_path(root, policy, rules)
    signals = _validation_calendar(calendar_path, rules)
    require(INTERNAL_CONTEXT_SIGNAL not in {value.signal_date for value in signals}, "Internal context signal was published")

    risk = rules["risk_cost"]
    require(tuple(risk["factor_order"]) == FACTOR_NAMES, "Risk factor order differs")
    require(risk["required_factor_covariance_rows"] == REQUIRED_FACTOR_COVARIANCE_ROWS, "Risk covariance shape differs")
    for key in ("post_cutoff_return_rows_used", "post_cutoff_factor_rows_used", "label_payload_rows_read"):
        require(risk[key] == 0, f"Risk/cost timing gate is nonzero: {key}")

    manifest = rules["manifest"]
    require(manifest["must_report_context_rows_published_as_zero"] is True, "Context publication gate missing")
    require(manifest["must_report_label_outcome_forward_and_final_rows_read_as_zero"] is True, "Outcome read gate missing")
    require(manifest["primary_manifest_authorizes_validation_outcome_access"] is False, "Primary manifest outcome gate reopened")
    require(manifest["independent_verification_required"] is True, "Independent verification is not required")

    materialization = rules["materialization"]
    require(materialization["frozen_stage02_through_stage08_source_files_may_be_modified"] is False, "Frozen-source mutation reopened")
    require(materialization["raw_source_rows_may_not_be_read_by_contract_or_verifier"] is True, "Contract raw-read gate changed")
    require(rules["development_parity"]["required_before_validation_materialization"] is True, "Development parity gate missing")

    specs = _payload_specs(rules, raw_schema, family_feature_names)
    return ValidationDataContract(
        policy_path=resolved_policy,
        policy_sha256=sha256_file(resolved_policy),
        policy=MappingProxyType(policy),
        rules=MappingProxyType(rules),
        validation_signals=signals,
        raw_feature_schema=raw_schema,
        lineage_schema=LINEAGE_SCHEMA,
        feature_names=feature_names,
        family_feature_names=family_feature_names,
        payload_specs=specs,
        frozen_bindings=MappingProxyType(dict(sorted(bindings.items()))),
    )


__all__ = [
    "CONTRACT_ID",
    "DEFAULT_POLICY",
    "FIRST_FINAL_ENTRY",
    "FIRST_PROHIBITED_SIGNAL",
    "FIRST_SIGNAL",
    "INTERNAL_CONTEXT_SIGNAL",
    "LAST_SIGNAL",
    "REQUIRED_FACTOR_COVARIANCE_ROWS",
    "REQUIRED_FEATURE_COUNT",
    "REQUIRED_PHYSICAL_COLUMNS",
    "REQUIRED_SIGNAL_COUNT",
    "ValidationDataContract",
    "ValidationDataContractError",
    "ValidationPayloadSpec",
    "ValidationSignal",
    "canonical_json_bytes",
    "require",
    "schema_sha256",
    "sha256_file",
    "unexpected_forbidden_columns",
    "validate_validation_data_contract",
]
