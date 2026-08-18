"""Schema-bound Stage 13 final-input registration manifests.

Importing this module is metadata-only.  The production preparer has no path,
schema, column, record, or reader arguments: after a separately authenticated
Stage 13 approval it inspects the 32 fixed Parquet paths registered at Stage
12.  An independent verifier reopens every file.  The broker loader accepts
only the resulting fixed verified pointer and never caller-created records.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.features.feature_store_contract import (
    expanded_features,
    raw_feature_schema,
    resolve_feature_sets,
)
from portfolio_bot.final_test.access import (
    FinalAccessBroker,
    FinalAccessError,
    FinalPayloadRole,
    RegisteredFinalPayload,
    _MANIFEST_REGISTRATION_TOKEN,
    _load_authenticated_registry,
    _parse_exact_arrow_schema,
    _schema_sha256,
)
from portfolio_bot.final_test.authorization import (
    Stage12VerifiedBinding,
    Stage13ApprovalReceipt,
    Stage13AuthorizationError,
    load_authenticated_stage12_binding,
    require_stage13_unblind_authorization,
)
from portfolio_bot.final_test.calendar import (
    CORE_COMPLETE_HOLDING_MONTHS,
    FINAL_HOLDING_MONTHS,
    PARTIAL_DECEMBER_HOLDING_MONTH,
    PRIMARY_FULL_HOLDING_MONTHS,
)
from portfolio_bot.risk_cost.contract import published_schemas
from portfolio_bot.validation.corporate_actions import CorporateAction
from portfolio_bot.validation.daily_engine import DatedMarketCloseInput
from portfolio_bot.validation.validation_source_extraction import (
    FACTOR_SCHEMA as STAGE11_FACTOR_SCHEMA,
    RETURN_SCHEMA as STAGE11_RETURN_SCHEMA,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = Path("configs/phase_06_final_test/stage_13_input_registry_v1.json")
PATHS_CONFIG = Path("configs/paths.toml")
FINAL_INPUT_NAMESPACE = Path("processed/phase_06_final_test/stage13-final-inputs-v2")
VERIFIED_REGISTRATION_POINTER = Path(
    "artifacts/phase_06_final_test/" "LATEST_STAGE_13_REGISTRATION_VERIFIED.json"
)
PRIMARY_MANIFEST_NAMESPACE = Path(
    "artifacts/phase_06_final_test/stage_13_input_registrations"
)
VERIFICATION_NAMESPACE = Path(
    "artifacts/phase_06_final_test/" "stage_13_input_registration_verifications"
)
MATERIALIZER_ID = "stage13-final-input-registration-materializer-v1"
VERIFIER_ID = "stage13-final-input-registration-independent-verifier-v1"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_ID = re.compile(r"stage13-inputs-[0-9a-f]{24}\Z")
_VERIFICATION_ID = re.compile(r"verify-stage13-inputs-[0-9a-f]{20}\Z")
_SEGMENT_MONTHS = MappingProxyType(
    {
        "all_metadata_only": FINAL_HOLDING_MONTHS,
        "primary_full": PRIMARY_FULL_HOLDING_MONTHS,
        "core_complete": CORE_COMPLETE_HOLDING_MONTHS,
        "partial_december_descriptive": (PARTIAL_DECEMBER_HOLDING_MONTH,),
    }
)
_CODE_MEMBERS = (
    Path("src/portfolio_bot/final_test/access.py"),
    Path("src/portfolio_bot/final_test/final_input_construction.py"),
    Path("src/portfolio_bot/final_test/input_materialization.py"),
    Path("src/portfolio_bot/features/feature_store_contract.py"),
    Path("src/portfolio_bot/risk_cost/contract.py"),
    Path("src/portfolio_bot/validation/corporate_actions.py"),
    Path("src/portfolio_bot/validation/daily_engine.py"),
    Path("src/portfolio_bot/validation/validation_source_extraction.py"),
)
_CONFIG_MEMBERS = (
    REGISTRY_PATH,
    PATHS_CONFIG,
    Path("configs/phase_03_features/feature_registry_v1.json"),
    Path("configs/phase_03_features/feature_build_rules_v1.json"),
    Path("configs/phase_03_features/label_build_rules_v1.json"),
    Path("configs/phase_04_models/risk_cost_build_rules_v1.json"),
    Path("configs/phase_02_canonicalization/crsp_typed_backbone_schema_v1.json"),
    Path("configs/phase_02_canonicalization/crsp_return_reconciliation_schema_v1.json"),
    Path("configs/phase_06_final_test/stage_12_final_system_v1.json"),
)


class FinalInputMaterializationError(RuntimeError):
    """Raised when final-input registration cannot be authenticated exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalInputMaterializationError(message)


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(value: object, *, label: str) -> str:
    _require(
        isinstance(value, str) and _HEX64.fullmatch(value) is not None,
        f"{label} must be an exact lowercase SHA-256",
    )
    return value


def _stable_regular_bytes(
    path: Path,
    *,
    trusted_root: Path,
    label: str,
    maximum_bytes: int = 32 * 1024 * 1024,
) -> bytes:
    """Read a bounded ordinary file without following any path symlink."""

    root = Path(os.path.abspath(trusted_root.expanduser()))
    selected = Path(os.path.abspath(path.expanduser()))
    try:
        relative = selected.relative_to(root)
    except ValueError as exc:
        raise FinalInputMaterializationError(f"{label} escaped its root") from exc
    _require(bool(relative.parts), f"{label} cannot equal its root")
    current = Path(selected.anchor)
    all_parts = selected.parts[1:]
    for index, part in enumerate(all_parts):
        current /= part
        try:
            observed = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise FinalInputMaterializationError(f"{label} is absent") from exc
        final = index == len(all_parts) - 1
        _require(not stat.S_ISLNK(observed.st_mode), f"{label} path is symlinked")
        _require(
            stat.S_ISREG(observed.st_mode) if final else stat.S_ISDIR(observed.st_mode),
            f"{label} path type differs",
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(selected, flags)
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and before.st_size <= maximum_bytes,
            f"{label} is not a bounded single-link file",
        )
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    _require(identity(before) == identity(after), f"{label} changed while read")
    return b"".join(chunks)


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalInputMaterializationError(f"{label} is not valid JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _member_hashes(project_root: Path, paths: tuple[Path, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        payload = _stable_regular_bytes(
            project_root / relative,
            trusted_root=project_root,
            label=f"bound member {relative.as_posix()}",
        )
        result[relative.as_posix()] = _sha256_bytes(payload)
    return result


def _configured_database_root(project_root: Path) -> Path:
    payload = _stable_regular_bytes(
        project_root / PATHS_CONFIG,
        trusted_root=project_root,
        label="paths config",
    )
    try:
        value = tomllib.loads(payload.decode("utf-8"))
        configured = value["database"]["root"]
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        raise FinalInputMaterializationError("database root config is invalid") from exc
    _require(isinstance(configured, str) and configured, "database root is missing")
    root = Path(configured).expanduser()
    _require(
        root.is_absolute() and root != Path(root.anchor), "database root is unsafe"
    )
    return Path(os.path.abspath(root))


def _physical_schema(schema: pa.Schema) -> pa.Schema:
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
        fields.append(pa.field(field.name, field_type, nullable=field.nullable))
    return pa.schema(fields)


def _prepend_holding_month(schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [pa.field("holding_month", pa.string(), nullable=False), *list(schema)]
    )


def _simple_schema(fields: tuple[tuple[str, pa.DataType], ...]) -> pa.Schema:
    return pa.schema([pa.field(name, kind, nullable=False) for name, kind in fields])


_SOURCE_TYPE_PATTERN = re.compile(r"decimal128\((\d+),(\d+)\)\Z")
_SOURCE_TYPES: Mapping[str, pa.DataType] = MappingProxyType(
    {
        "bool": pa.bool_(),
        "date32": pa.date32(),
        "float64": pa.float64(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "string": pa.string(),
        "utf8": pa.string(),
    }
)
_INTERVAL_FIELDS = (
    pa.field("holding_month", pa.string(), nullable=False),
    pa.field("signal_date", pa.date32(), nullable=False),
    pa.field("entry_date", pa.date32(), nullable=False),
    pa.field("exit_date", pa.date32(), nullable=False),
)
_MARKET_EXECUTION_FIELDS = (
    "permco",
    "long_eligible",
    "short_eligible",
    "information_cutoff_date",
    "full_spread_20d",
    "adv20_usd",
    "impact_volatility_daily",
    "impact_volatility_source",
    "impact_volatility_observations",
    "primary_borrow_rate_annual",
    "stress_borrow_rate_annual",
    "new_long_trade_allowed",
    "new_short_trade_allowed",
    "cost_input_status",
)


def _source_type(value: object) -> pa.DataType:
    _require(isinstance(value, str), "source Arrow type is missing")
    if value in _SOURCE_TYPES:
        return _SOURCE_TYPES[value]
    match = _SOURCE_TYPE_PATTERN.fullmatch(value)
    _require(match is not None, f"unsupported source Arrow type: {value}")
    return pa.decimal128(int(match.group(1)), int(match.group(2)))


def _source_table_schema(
    project_root: Path,
    relative_path: Path,
    table_name: str,
) -> pa.Schema:
    """Rebuild one frozen source schema without opening an empirical payload."""

    payload = _stable_regular_bytes(
        project_root / relative_path,
        trusted_root=project_root,
        label=f"source schema registry {relative_path.as_posix()}",
    )
    registry = _json_object(payload, label=f"source schema {relative_path.as_posix()}")
    tables = registry.get("tables")
    _require(isinstance(tables, list), "source schema table list is missing")
    matches = tuple(
        item
        for item in tables
        if isinstance(item, Mapping) and item.get("table") == table_name
    )
    _require(len(matches) == 1, f"source schema table differs: {table_name}")
    columns = matches[0].get("columns")
    _require(
        isinstance(columns, list) and bool(columns),
        f"source schema is empty: {table_name}",
    )
    fields: list[pa.Field] = []
    for raw in columns:
        _require(
            isinstance(raw, Mapping)
            and isinstance(raw.get("name"), str)
            and type(raw.get("nullable")) is bool,
            f"source schema field is invalid: {table_name}",
        )
        fields.append(
            pa.field(
                str(raw["name"]),
                _source_type(raw.get("type")),
                nullable=bool(raw["nullable"]),
            )
        )
    schema = pa.schema(fields)
    _require(
        len(schema.names) == len(set(schema.names)),
        f"source schema has duplicate fields: {table_name}",
    )
    return schema


def _field(schema: pa.Schema, name: str, *, renamed: str | None = None) -> pa.Field:
    try:
        source = schema.field(name)
    except KeyError as exc:
        raise FinalInputMaterializationError(
            f"frozen source field is absent: {name}"
        ) from exc
    return pa.field(renamed or source.name, source.type, nullable=source.nullable)


def _final_market_input_schema(project_root: Path) -> pa.Schema:
    """Daily/entry market spine with exact Stage 2 and Stage 8 authorities."""

    backbone = Path(
        "configs/phase_02_canonicalization/crsp_typed_backbone_schema_v1.json"
    )
    market = _source_table_schema(project_root, backbone, "crsp_security_day")
    shares = _source_table_schema(project_root, backbone, "crsp_shares_interval")
    risk = published_schemas()["development_risk_cost_inputs.parquet"]
    share_fields = [
        _field(shares, "start_date", renamed="shares_interval_start_date"),
        _field(shares, "end_date", renamed="shares_interval_end_date"),
        *[
            _field(shares, name)
            for name in (
                "shares_outstanding_thousand",
                "shares_outstanding",
                "shares_source",
                "shares_factor_type",
                "shares_adr_flag",
            )
        ],
    ]
    schema = pa.schema(
        [
            *_INTERVAL_FIELDS,
            *list(market),
            *share_fields,
            *[_field(risk, name) for name in _MARKET_EXECUTION_FIELDS],
            pa.field("entry_close_status", pa.string(), nullable=False),
        ]
    )
    _require(
        len(schema.names) == len(set(schema.names)) and len(schema) == 60,
        "final daily market schema width or field identity differs",
    )
    return schema


def _final_market_accounting_schema() -> pa.Schema:
    """Executable wide join of the Stage 11 five-shard accounting contract.

    The first four fields retain the authenticated interval.  The exact Stage
    11 return fields follow unchanged.  Entry/execution and factor fields are
    repeated on security-day rows so the table can be projected directly into
    ``DatedMarketCloseInput``.  Distribution audit fields remain non-economic;
    normalized corporate-action fields are sparse and are applied only after
    that day's authoritative total return.
    """

    factor_fields = [
        _field(
            STAGE11_FACTOR_SCHEMA,
            "observation_date",
            renamed="factor_observation_date",
        ),
        _field(
            STAGE11_FACTOR_SCHEMA,
            "public_availability_date",
            renamed="factor_public_availability_date",
        ),
        *[
            _field(STAGE11_FACTOR_SCHEMA, name)
            for name in ("mktrf", "smb", "hml", "rmw", "cma", "rf", "umd")
        ],
    ]
    nullable_action_fields = (
        pa.field("corporate_action_id", pa.string(), nullable=True),
        pa.field("corporate_economic_event_date", pa.date32(), nullable=True),
        pa.field("corporate_assigned_return_date", pa.date32(), nullable=True),
        pa.field("corporate_public_availability_date", pa.date32(), nullable=True),
        pa.field("corporate_action_sequence", pa.int32(), nullable=True),
        pa.field("corporate_source_record_sha256", pa.string(), nullable=True),
        pa.field("corporate_source_permno", pa.int64(), nullable=True),
        pa.field("corporate_settlement_kind", pa.string(), nullable=True),
        pa.field("corporate_successor_permno", pa.int64(), nullable=True),
        pa.field("corporate_settlement_value_multiplier", pa.float64(), nullable=True),
        pa.field(
            "corporate_terminal_total_return_already_applied",
            pa.bool_(),
            nullable=True,
        ),
        pa.field(
            "corporate_terminal_sensitivity_policy_sha256",
            pa.string(),
            nullable=True,
        ),
        pa.field(
            "corporate_terminal_sensitivity_evidence_sha256",
            pa.string(),
            nullable=True,
        ),
        pa.field(
            "corporate_event_day_quoted_full_spread",
            pa.float64(),
            nullable=True,
        ),
    )
    schema = pa.schema(
        [
            *_INTERVAL_FIELDS,
            *list(STAGE11_RETURN_SCHEMA),
            pa.field("authenticated_trading_session", pa.bool_(), nullable=False),
            pa.field(
                "close_usd_per_share",
                pa.decimal128(38, 6),
                nullable=True,
            ),
            pa.field("entry_close_usd_per_share", pa.float64(), nullable=False),
            pa.field("entry_close_status", pa.string(), nullable=False),
            pa.field("full_spread_20d", pa.float64(), nullable=False),
            pa.field("adv20_usd", pa.float64(), nullable=False),
            pa.field("impact_volatility_daily", pa.float64(), nullable=False),
            pa.field("borrow_rate_annual", pa.float64(), nullable=False),
            pa.field("financing_rate_annual", pa.float64(), nullable=False),
            pa.field("source_payload_sha256", pa.string(), nullable=False),
            pa.field("no_mark_source_record_sha256", pa.string(), nullable=True),
            *factor_fields,
            pa.field("distribution_event_count", pa.int32(), nullable=False),
            pa.field("distribution_event_set_sha256", pa.string(), nullable=False),
            pa.field(
                "distribution_events_assigned_exactly_once",
                pa.bool_(),
                nullable=False,
            ),
            *nullable_action_fields,
        ]
    )
    _require(
        len(schema.names) == len(set(schema.names)) and len(schema) == 57,
        "final daily accounting schema width or field identity differs",
    )
    return schema


@dataclass(frozen=True, slots=True)
class FinalDailyAccountingProjection:
    """Typed bridge from one registered month into the generic daily engine."""

    holding_month: str
    signal_date: dt.date
    entry_date: dt.date
    exit_date: dt.date
    execution_permno: np.ndarray
    execution_price: np.ndarray
    full_spread: np.ndarray
    adv_dollars: np.ndarray
    impact_volatility_daily: np.ndarray
    financing_rate_annual: float
    daily_inputs: tuple[DatedMarketCloseInput, ...]
    corporate_actions: tuple[CorporateAction, ...]


def _finite_float(value: object, *, label: str, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalInputMaterializationError(f"{label} is not numeric") from exc
    _require(math.isfinite(result), f"{label} is nonfinite")
    if minimum is not None:
        _require(result >= minimum, f"{label} is below its allowed minimum")
    return result


def _finite_or_nan_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    """Accept the v1 non-nullable NaN encoding, but never infinity."""

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalInputMaterializationError(f"{label} is not numeric") from exc
    _require(not math.isinf(result), f"{label} is infinite")
    if math.isnan(result):
        return result
    if minimum is not None:
        _require(result >= minimum, f"{label} is below its allowed minimum")
    return result


def project_final_daily_accounting_inputs(
    table: pa.Table,
    *,
    holding_month: str,
) -> FinalDailyAccountingProjection:
    """Validate and project one broker-returned wide table without I/O.

    Numeric authoritative returns are never inferred from prices.  Exact
    Stage 11 missing-return rows become the explicit no-mark channel, while
    other nullable states remain absent and therefore make an exposed holding
    fail closed in ``run_dated_holding_period``.  Sparse corporate actions are
    returned in post-return application order.
    """

    _require(
        isinstance(table, pa.Table), "daily accounting input must be an Arrow table"
    )
    source = table.combine_chunks()
    expected = _final_market_accounting_schema()
    _require(
        source.schema.equals(expected, check_metadata=True),
        "daily accounting Arrow schema differs",
    )
    _require(
        isinstance(holding_month, str) and holding_month in FINAL_HOLDING_MONTHS,
        "daily accounting holding month is outside the frozen final domain",
    )
    rows = source.to_pylist()
    _require(bool(rows), "daily accounting month is empty")
    _require(
        all(row["holding_month"] == holding_month for row in rows),
        "daily accounting table splices holding months",
    )
    for name in ("signal_date", "entry_date", "exit_date"):
        values = {row[name] for row in rows}
        _require(len(values) == 1, f"daily accounting {name} is not unique")
    signal_date = rows[0]["signal_date"]
    entry_date = rows[0]["entry_date"]
    exit_date = rows[0]["exit_date"]
    _require(
        type(signal_date) is dt.date
        and type(entry_date) is dt.date
        and type(exit_date) is dt.date
        and signal_date < entry_date < exit_date,
        "daily accounting interval chronology differs",
    )
    keys = tuple((row["calendar_date"], row["permno"]) for row in rows)
    _require(
        len(keys) == len(set(keys))
        and all(
            type(date_value) is dt.date
            and entry_date < date_value <= exit_date
            and type(permno) is int
            and permno > 0
            for date_value, permno in keys
        ),
        "daily accounting security-day identity differs",
    )
    _require(
        all(row["authenticated_trading_session"] is True for row in rows),
        "daily accounting row lacks trading-session authority",
    )

    financing = {
        _finite_float(
            row["financing_rate_annual"],
            label="daily accounting financing rate",
            minimum=0.0,
        )
        for row in rows
    }
    _require(len(financing) == 1, "daily accounting financing rate is not unique")

    execution_by_permno: dict[int, tuple[float, float, float, float]] = {}
    for row in rows:
        permno = int(row["permno"])
        entry_close = _finite_or_nan_float(
            row["entry_close_usd_per_share"],
            label=f"entry close {permno}",
            minimum=0.0,
        )
        entry_status = row["entry_close_status"]
        _require(
            (
                entry_status == "PRESENT"
                and math.isfinite(entry_close)
                and entry_close > 0.0
            )
            or (
                entry_status == "MISSING_VALID_ENTRY_CLOSE" and math.isnan(entry_close)
            ),
            f"daily accounting entry-close value/status differs: {permno}",
        )
        full_spread = _finite_or_nan_float(
            row["full_spread_20d"],
            label=f"full spread {permno}",
            minimum=0.0,
        )
        adv = _finite_or_nan_float(
            row["adv20_usd"],
            label=f"ADV {permno}",
            minimum=0.0,
        )
        impact = _finite_or_nan_float(
            row["impact_volatility_daily"],
            label=f"impact volatility {permno}",
            minimum=0.0,
        )
        _require(math.isnan(adv) or adv > 0.0, f"ADV is not positive: {permno}")
        values = (entry_close, full_spread, adv, impact)
        prior = execution_by_permno.get(permno)
        if prior is None:
            execution_by_permno[permno] = values
        else:
            _require(
                np.array_equal(
                    np.asarray(prior, dtype=np.float64),
                    np.asarray(values, dtype=np.float64),
                    equal_nan=True,
                ),
                f"entry execution changes within month: {permno}",
            )

    actions: list[CorporateAction] = []
    seen_action_ids: set[str] = set()
    action_fields = (
        "corporate_economic_event_date",
        "corporate_assigned_return_date",
        "corporate_public_availability_date",
        "corporate_action_sequence",
        "corporate_source_record_sha256",
        "corporate_source_permno",
        "corporate_settlement_kind",
        "corporate_settlement_value_multiplier",
        "corporate_terminal_total_return_already_applied",
    )
    terminal_sensitivity_fields = (
        "corporate_terminal_sensitivity_policy_sha256",
        "corporate_terminal_sensitivity_evidence_sha256",
        "corporate_event_day_quoted_full_spread",
    )
    for row in rows:
        action_id = row["corporate_action_id"]
        if action_id is None:
            _require(
                all(
                    row[name] is None
                    for name in (
                        *action_fields,
                        *terminal_sensitivity_fields,
                        "corporate_successor_permno",
                    )
                ),
                "daily accounting has an incomplete sparse corporate action",
            )
            continue
        _require(
            isinstance(action_id, str)
            and bool(action_id.strip())
            and action_id not in seen_action_ids
            and all(row[name] is not None for name in action_fields)
            and (
                all(row[name] is None for name in terminal_sensitivity_fields)
                or all(row[name] is not None for name in terminal_sensitivity_fields)
            )
            and row["corporate_assigned_return_date"] == row["calendar_date"]
            and row["corporate_source_permno"] == row["permno"],
            "daily accounting corporate action identity differs",
        )
        seen_action_ids.add(action_id)
        actions.append(
            CorporateAction(
                action_id=action_id,
                economic_event_date=row["corporate_economic_event_date"],
                assigned_return_date=row["corporate_assigned_return_date"],
                public_availability_date=row["corporate_public_availability_date"],
                action_sequence=int(row["corporate_action_sequence"]),
                source_record_sha256=str(row["corporate_source_record_sha256"]),
                source_permno=int(row["corporate_source_permno"]),
                settlement_kind=str(row["corporate_settlement_kind"]),  # type: ignore[arg-type]
                successor_permno=row["corporate_successor_permno"],
                settlement_value_multiplier=float(
                    row["corporate_settlement_value_multiplier"]
                ),
                terminal_total_return_already_applied=bool(
                    row["corporate_terminal_total_return_already_applied"]
                ),
                terminal_sensitivity_policy_sha256=(
                    row["corporate_terminal_sensitivity_policy_sha256"]
                ),
                terminal_sensitivity_evidence_sha256=(
                    row["corporate_terminal_sensitivity_evidence_sha256"]
                ),
                event_day_quoted_full_spread=(
                    row["corporate_event_day_quoted_full_spread"]
                ),
            )
        )

    by_date: dict[dt.date, list[dict[str, Any]]] = {}
    for row in rows:
        date_value = row["calendar_date"]
        _require(
            row["factor_observation_date"] == date_value
            and type(row["factor_public_availability_date"]) is dt.date
            and row["factor_public_availability_date"] <= date_value,
            "daily factor date or availability differs",
        )
        for name in (
            "source_payload_sha256",
            "distribution_event_set_sha256",
        ):
            _hash(row[name], label=f"daily accounting {name}")
        _require(
            type(row["distribution_event_count"]) is int
            and row["distribution_event_count"] >= 0
            and row["distribution_events_assigned_exactly_once"] is True,
            "daily distribution audit differs",
        )
        by_date.setdefault(date_value, []).append(row)

    days: list[DatedMarketCloseInput] = []
    for date_value in sorted(by_date):
        day_rows = sorted(by_date[date_value], key=lambda item: int(item["permno"]))
        rf_values = {
            _finite_float(row["rf"], label=f"daily RF {date_value}") for row in day_rows
        }
        _require(len(rf_values) == 1, f"daily RF is not unique: {date_value}")
        observed: list[tuple[int, float, float, dt.date]] = []
        no_mark: list[tuple[int, float, str]] = []
        for row in day_rows:
            permno = int(row["permno"])
            borrow = _finite_float(
                row["borrow_rate_annual"],
                label=f"borrow rate {permno} {date_value}",
                minimum=0.0,
            )
            value = row["authoritative_total_return"]
            if value is not None:
                total_return = _finite_float(
                    value,
                    label=f"authoritative return {permno} {date_value}",
                    minimum=-1.0,
                )
                span_start = row["return_span_start_date"]
                _require(
                    row["return_history_usable"] is True
                    and row["return_history_authorized"] is True
                    and type(span_start) is dt.date
                    and span_start < date_value
                    and row["return_span_end_date"] == date_value
                    and row["no_mark_source_record_sha256"] is None,
                    "numeric authoritative-return metadata differs",
                )
                observed.append((permno, total_return, borrow, span_start))
                continue
            exact_no_mark = (
                row["authoritative_return_state"] == "missing"
                and row["return_leg_class"] == "missing_return"
                and row["return_span_start_date"] is None
                and row["return_span_end_date"] == date_value
                and row["covered_trading_sessions"] == 0
                and row["span_alignment_status"] == "no_return_span"
                and row["delisting_event_count"] == 0
                and row["delisting_treatment_status"] == "none"
                and row["delisting_adjustment_return"] is None
                and row["return_history_usable"] is False
                and row["return_history_authorized"] is False
            )
            if exact_no_mark:
                evidence = _hash(
                    row["no_mark_source_record_sha256"],
                    label=f"no-mark evidence {permno} {date_value}",
                )
                no_mark.append((permno, borrow, evidence))
            else:
                _require(
                    row["no_mark_source_record_sha256"] is None,
                    "non-no-mark nullable row carries no-mark evidence",
                )
        _require(
            set(value[0] for value in observed).isdisjoint(
                value[0] for value in no_mark
            ),
            "observed and no-mark daily identities overlap",
        )
        payload = {
            "holding_month": holding_month,
            "calendar_date": date_value.isoformat(),
            "source_payload_sha256": sorted(
                str(row["source_payload_sha256"]) for row in day_rows
            ),
            "observed": [
                [permno, float(value).hex(), float(borrow).hex(), start.isoformat()]
                for permno, value, borrow, start in observed
            ],
            "no_mark": [list(value) for value in no_mark],
        }
        days.append(
            DatedMarketCloseInput(
                trading_date=date_value,
                permno=np.asarray([value[0] for value in observed], dtype=np.int64),
                total_return=np.asarray(
                    [value[1] for value in observed], dtype=np.float64
                ),
                borrow_rate_annual=np.asarray(
                    [value[2] for value in observed], dtype=np.float64
                ),
                daily_risk_free_rate=next(iter(rf_values)),
                source_payload_sha256=_sha256_bytes(_canonical_json_bytes(payload)),
                observed_return_span_start_date=tuple(value[3] for value in observed),
                no_mark_permno=np.asarray(
                    [value[0] for value in no_mark], dtype=np.int64
                ),
                no_mark_borrow_rate_annual=np.asarray(
                    [value[1] for value in no_mark], dtype=np.float64
                ),
                no_mark_source_record_sha256=tuple(value[2] for value in no_mark),
            )
        )

    permnos = np.asarray(sorted(execution_by_permno), dtype=np.int64)
    execution = [execution_by_permno[int(value)] for value in permnos]
    return FinalDailyAccountingProjection(
        holding_month=holding_month,
        signal_date=signal_date,
        entry_date=entry_date,
        exit_date=exit_date,
        execution_permno=permnos,
        execution_price=np.asarray([value[0] for value in execution], dtype=np.float64),
        full_spread=np.asarray([value[1] for value in execution], dtype=np.float64),
        adv_dollars=np.asarray([value[2] for value in execution], dtype=np.float64),
        impact_volatility_daily=np.asarray(
            [value[3] for value in execution], dtype=np.float64
        ),
        financing_rate_annual=next(iter(financing)),
        daily_inputs=tuple(days),
        corporate_actions=tuple(
            sorted(
                actions,
                key=lambda item: (
                    item.assigned_return_date,
                    item.action_sequence,
                    item.action_id,
                ),
            )
        ),
    )


def _independently_derived_schemas(project_root: Path) -> Mapping[str, pa.Schema]:
    """Regenerate all nontrivial schemas from frozen pre-final authorities."""

    registry_payload = _stable_regular_bytes(
        project_root / "configs/phase_03_features/feature_registry_v1.json",
        trusted_root=project_root,
        label="feature registry",
    )
    rules_payload = _stable_regular_bytes(
        project_root / "configs/phase_03_features/feature_build_rules_v1.json",
        trusted_root=project_root,
        label="feature build rules",
    )
    registry = _json_object(registry_payload, label="feature registry")
    rules = _json_object(rules_payload, label="feature build rules")
    features = expanded_features(registry)
    raw = _physical_schema(raw_feature_schema(features))
    _require(len(raw) == 170, "frozen raw feature schema is not 170 fields")
    feature_sets = resolve_feature_sets(features, rules)
    by_name = {field.name: field for field in raw}
    audit = list(raw)[:7]
    primary_predictor = _prepend_holding_month(raw)
    core_predictor = pa.schema(
        [
            pa.field("holding_month", pa.string(), nullable=False),
            *audit,
            *[by_name[name] for name in feature_sets["reduced_auxiliary_core"]],
        ]
    )
    _require(
        len(primary_predictor) == 171 and len(core_predictor) == 119,
        "frozen predictor schema widths differ",
    )
    stage8_schemas = published_schemas()
    risk_cost = _prepend_holding_month(
        stage8_schemas["development_risk_cost_inputs.parquet"]
    )
    factor_covariance = _prepend_holding_month(
        stage8_schemas["development_factor_covariance.parquet"]
    )
    risk_cost_month_summary = _prepend_holding_month(
        stage8_schemas["development_risk_cost_month_summary.parquet"]
    )
    calendar = _simple_schema(
        (
            ("holding_month", pa.string()),
            ("signal_date", pa.date32()),
            ("entry_date", pa.date32()),
            ("exit_date", pa.date32()),
            ("outcome_availability_date", pa.date32()),
            ("segment", pa.string()),
            ("complete_holding_interval", pa.bool_()),
            ("confirmatory_inference_eligible", pa.bool_()),
        )
    )
    market = _final_market_input_schema(project_root)
    accounting = _final_market_accounting_schema()
    factors = _simple_schema(
        (
            ("holding_month", pa.string()),
            ("mktrf", pa.float64()),
            ("smb", pa.float64()),
            ("hml", pa.float64()),
            ("rmw", pa.float64()),
            ("cma", pa.float64()),
            ("umd", pa.float64()),
            ("rf", pa.float64()),
        )
    )
    benchmarks = _simple_schema(
        (
            ("holding_month", pa.string()),
            ("benchmark_id", pa.string()),
            ("total_return", pa.float64()),
        )
    )
    labels = _simple_schema(
        (
            ("holding_month", pa.string()),
            ("signal_date", pa.date32()),
            ("entry_date", pa.date32()),
            ("exit_date", pa.date32()),
            ("outcome_availability_date", pa.date32()),
            ("permno", pa.int64()),
            ("unranked_residual_return", pa.float64()),
            ("residual_return_rank", pa.float64()),
            ("primary_label_status", pa.string()),
        )
    )
    outcomes = _simple_schema(
        (
            ("holding_month", pa.string()),
            ("entry_date", pa.date32()),
            ("exit_date", pa.date32()),
            ("permno", pa.int64()),
            ("total_return", pa.float64()),
        )
    )
    partial_outcome = _simple_schema(
        (
            ("holding_month", pa.string()),
            ("entry_date", pa.date32()),
            ("administrative_close_date", pa.date32()),
            ("permno", pa.int64()),
            ("partial_total_return", pa.float64()),
        )
    )
    result: dict[str, pa.Schema] = {"final_calendar": calendar}
    for prefix, predictor in (
        ("primary", primary_predictor),
        ("core", core_predictor),
        ("partial", core_predictor),
    ):
        result[f"{prefix}_predictors"] = predictor
        result[f"{prefix}_risk_cost"] = risk_cost
        result[f"{prefix}_cost_inputs"] = risk_cost
        result[f"{prefix}_factor_covariance"] = factor_covariance
        result[f"{prefix}_risk_cost_month_summary"] = risk_cost_month_summary
        result[f"{prefix}_market_inputs"] = market
        result[f"{prefix}_market_accounting"] = accounting
        result[f"{prefix}_benchmarks"] = benchmarks
    for prefix in ("primary", "core"):
        result[f"{prefix}_factors"] = factors
        result[f"{prefix}_completed_labels"] = labels
        result[f"{prefix}_outcomes"] = outcomes
    result["partial_descriptive_outcome"] = partial_outcome
    _require(len(result) == 32, "derived final-input schema set differs")
    return MappingProxyType(result)


def _validated_templates(
    project_root: Path,
    *,
    binding: Stage12VerifiedBinding,
) -> tuple[Mapping[str, Mapping[str, object]], Mapping[str, pa.Schema], str]:
    templates = _load_authenticated_registry(project_root, binding=binding)
    derived = _independently_derived_schemas(project_root)
    _require(set(templates) == set(derived), "frozen final-input name set differs")
    canonical_templates: list[dict[str, object]] = []
    for name, raw in templates.items():
        try:
            columns = tuple(raw["required_columns"])  # type: ignore[arg-type]
            schema = _parse_exact_arrow_schema(raw["arrow_schema"])
            months = _SEGMENT_MONTHS[str(raw["segment"])]
            FinalPayloadRole(str(raw["role"]))
        except (KeyError, TypeError, ValueError, FinalAccessError) as exc:
            raise FinalInputMaterializationError(
                f"frozen final-input template is invalid: {name}"
            ) from exc
        _require(
            columns == tuple(schema.names)
            and schema.equals(derived[name], check_metadata=True)
            and bool(months),
            f"frozen final-input schema differs: {name}",
        )
        canonical_templates.append(dict(raw))
    contract_sha = _sha256_bytes(
        _canonical_json_bytes(
            {
                "registry_id": "stage13-final-input-registry-v1",
                "templates": canonical_templates,
            }
        )
    )
    return templates, derived, contract_sha


def _binding_mapping(binding: Stage12VerifiedBinding) -> dict[str, object]:
    return {**binding.to_mapping(), "stage12_binding_sha256": binding.binding_sha256}


def _implementation_binding(project_root: Path) -> dict[str, object]:
    code = _member_hashes(project_root, _CODE_MEMBERS)
    config = _member_hashes(project_root, _CONFIG_MEMBERS)
    return {
        "materializer_id": MATERIALIZER_ID,
        "verifier_id": VERIFIER_ID,
        "code_hashes": code,
        "code_hashes_sha256": _sha256_bytes(_canonical_json_bytes(code)),
        "config_hashes": config,
        "config_hashes_sha256": _sha256_bytes(_canonical_json_bytes(config)),
    }


def _entry_path(database_root: Path, name: str) -> tuple[str, Path]:
    relative = (FINAL_INPUT_NAMESPACE / f"{name}.parquet").as_posix()
    return relative, database_root / relative


def _inspect_parquet(
    path: Path,
    *,
    database_root: Path,
    expected_schema: pa.Schema,
    expected_months: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    """Independently hash and inspect one exact Parquet file."""

    try:
        relative = path.relative_to(database_root)
    except ValueError as exc:
        raise FinalInputMaterializationError(f"{label} escaped database root") from exc
    _require(bool(relative.parts), f"{label} path is empty")
    current = Path(path.anchor)
    all_parts = path.parts[1:]
    for index, part in enumerate(all_parts):
        current /= part
        try:
            observed = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise FinalInputMaterializationError(f"{label} is absent") from exc
        final = index == len(all_parts) - 1
        _require(not stat.S_ISLNK(observed.st_mode), f"{label} path is symlinked")
        _require(
            stat.S_ISREG(observed.st_mode) if final else stat.S_ISDIR(observed.st_mode),
            f"{label} path type differs",
        )
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and before.st_size > 0,
            f"{label} file identity is unsafe",
        )
        digest = hashlib.sha256()
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
            parquet = pq.ParquetFile(stream)
            schema = parquet.schema_arrow
            rows = parquet.metadata.num_rows
            row_groups = parquet.metadata.num_row_groups
        _require(
            schema.equals(expected_schema, check_metadata=True),
            f"{label} Arrow schema differs",
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
            month_table = pq.read_table(stream, columns=["holding_month"])
        after = os.fstat(descriptor)
    except FinalInputMaterializationError:
        raise
    except (OSError, pa.ArrowException) as exc:
        raise FinalInputMaterializationError(f"{label} could not be inspected") from exc
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    _require(identity(before) == identity(after), f"{label} changed while inspected")
    try:
        normalized = pc.cast(month_table["holding_month"], pa.string(), safe=True)
    except pa.ArrowException as exc:
        raise FinalInputMaterializationError(
            f"{label} holding month is invalid"
        ) from exc
    values = normalized.to_pylist()
    observed_months = tuple(sorted({value for value in values if value is not None}))
    _require(
        None not in values and observed_months == expected_months,
        f"{label} month domain differs",
    )
    _require(rows > 0 and row_groups > 0, f"{label} is empty")
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": before.st_size,
        "rows": rows,
        "row_groups": row_groups,
        "arrow_schema_sha256": _schema_sha256(schema),
    }


def _prepare_manifest(
    *,
    project_root: Path,
    database_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
) -> dict[str, object]:
    templates, schemas, registry_sha = _validated_templates(
        project_root, binding=binding
    )
    implementation = _implementation_binding(project_root)
    entries: list[dict[str, object]] = []
    for name, template in templates.items():
        segment = str(template["segment"])
        months = _SEGMENT_MONTHS[segment]
        relative, absolute = _entry_path(database_root, name)
        receipt = _inspect_parquet(
            absolute,
            database_root=database_root,
            expected_schema=schemas[name],
            expected_months=months,
            label=f"final input {name}",
        )
        entries.append(
            {
                "registration_name": name,
                "role": str(template["role"]),
                "segment": segment,
                "month_domain": str(template["month_domain"]),
                "holding_months": list(months),
                "relative_path": relative,
                "absolute_path": str(absolute),
                "sha256": receipt["sha256"],
                "size_bytes": receipt["size_bytes"],
                "rows": receipt["rows"],
                "row_groups": receipt["row_groups"],
                "exact_columns": list(schemas[name].names),
                "arrow_schema_sha256": receipt["arrow_schema_sha256"],
                "materializer_code_sha256": implementation["code_hashes_sha256"],
                "materializer_config_sha256": implementation["config_hashes_sha256"],
                "stage12_binding_sha256": binding.binding_sha256,
            }
        )
    body: dict[str, object] = {
        "schema_version": 1,
        "status": "PRIMARY_COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
        "registry_id": "stage13-final-input-registry-v1",
        "registry_contract_sha256": registry_sha,
        "stage12_binding": _binding_mapping(binding),
        "approval_payload_sha256": approval.payload_sha256,
        "database_root": str(database_root),
        "fixed_input_namespace": FINAL_INPUT_NAMESPACE.as_posix(),
        "implementation": implementation,
        "entries": entries,
    }
    fingerprint = _sha256_bytes(_canonical_json_bytes(body))
    return {
        **body,
        "manifest_id": f"stage13-inputs-{fingerprint[:24]}",
        "content_fingerprint": fingerprint,
    }


def prepare_final_input_registration_manifest(
    approval_receipt: Stage13ApprovalReceipt | Mapping[str, object] | None,
) -> dict[str, object]:
    """Inspect the closed 32-file production namespace after Stage 13 approval."""

    root = PROJECT_ROOT.absolute()
    try:
        binding = load_authenticated_stage12_binding(root)
        approval = require_stage13_unblind_authorization(
            approval_receipt, expected_stage12_binding=binding
        )
    except Stage13AuthorizationError as exc:
        raise FinalInputMaterializationError(str(exc)) from exc
    return _prepare_manifest(
        project_root=root,
        database_root=_configured_database_root(root),
        binding=binding,
        approval=approval,
    )


_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "registry_id",
    "registry_contract_sha256",
    "stage12_binding",
    "approval_payload_sha256",
    "database_root",
    "fixed_input_namespace",
    "implementation",
    "entries",
    "manifest_id",
    "content_fingerprint",
}
_ENTRY_KEYS = {
    "registration_name",
    "role",
    "segment",
    "month_domain",
    "holding_months",
    "relative_path",
    "absolute_path",
    "sha256",
    "size_bytes",
    "rows",
    "row_groups",
    "exact_columns",
    "arrow_schema_sha256",
    "materializer_code_sha256",
    "materializer_config_sha256",
    "stage12_binding_sha256",
}


def _validate_manifest_metadata(
    manifest: Mapping[str, object],
    *,
    project_root: Path,
    database_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
) -> tuple[Mapping[str, Mapping[str, object]], Mapping[str, pa.Schema], str]:
    templates, schemas, registry_sha = _validated_templates(
        project_root, binding=binding
    )
    implementation = _implementation_binding(project_root)
    _require(set(manifest) == _MANIFEST_KEYS, "registration manifest fields differ")
    body = {
        key: manifest[key]
        for key in manifest
        if key not in {"manifest_id", "content_fingerprint"}
    }
    fingerprint = _sha256_bytes(_canonical_json_bytes(body))
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("status")
        == "PRIMARY_COMPLETE_PENDING_INDEPENDENT_VERIFICATION"
        and manifest.get("registry_id") == "stage13-final-input-registry-v1"
        and manifest.get("registry_contract_sha256") == registry_sha
        and manifest.get("stage12_binding") == _binding_mapping(binding)
        and manifest.get("approval_payload_sha256") == approval.payload_sha256
        and manifest.get("database_root") == str(database_root)
        and manifest.get("fixed_input_namespace") == FINAL_INPUT_NAMESPACE.as_posix()
        and manifest.get("implementation") == implementation
        and manifest.get("content_fingerprint") == fingerprint
        and manifest.get("manifest_id") == f"stage13-inputs-{fingerprint[:24]}",
        "registration manifest identity differs",
    )
    entries = manifest.get("entries")
    _require(
        isinstance(entries, list) and len(entries) == 32,
        "registration entry count differs",
    )
    names: list[str] = []
    for raw_entry in entries:
        _require(isinstance(raw_entry, Mapping), "registration entry is invalid")
        entry = raw_entry
        _require(set(entry) == _ENTRY_KEYS, "registration entry fields differ")
        name = str(entry.get("registration_name"))
        _require(name in templates and name not in names, "registration name differs")
        names.append(name)
        template = templates[name]
        segment = str(template["segment"])
        months = _SEGMENT_MONTHS[segment]
        relative, absolute = _entry_path(database_root, name)
        _require(
            entry.get("role") == template["role"]
            and entry.get("segment") == segment
            and entry.get("month_domain") == template["month_domain"]
            and entry.get("holding_months") == list(months)
            and entry.get("relative_path") == relative
            and entry.get("absolute_path") == str(absolute)
            and entry.get("exact_columns") == list(schemas[name].names)
            and entry.get("arrow_schema_sha256") == _schema_sha256(schemas[name])
            and entry.get("materializer_code_sha256")
            == implementation["code_hashes_sha256"]
            and entry.get("materializer_config_sha256")
            == implementation["config_hashes_sha256"]
            and entry.get("stage12_binding_sha256") == binding.binding_sha256,
            f"registration entry contract differs: {name}",
        )
        _hash(entry.get("sha256"), label=f"{name} file SHA-256")
        _require(
            type(entry.get("size_bytes")) is int
            and int(entry["size_bytes"]) > 0
            and type(entry.get("rows")) is int
            and int(entry["rows"]) > 0
            and type(entry.get("row_groups")) is int
            and int(entry["row_groups"]) > 0,
            f"registration entry counts differ: {name}",
        )
    _require(names == list(templates), "registration entry order differs")
    return templates, schemas, registry_sha


def _independently_verify(
    manifest: Mapping[str, object],
    *,
    project_root: Path,
    database_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
) -> dict[str, object]:
    templates, schemas, registry_sha = _validate_manifest_metadata(
        manifest,
        project_root=project_root,
        database_root=database_root,
        binding=binding,
        approval=approval,
    )
    entries_by_name = {
        str(item["registration_name"]): item for item in manifest["entries"]  # type: ignore[index]
    }
    receipts: list[dict[str, object]] = []
    for name, template in templates.items():
        entry = entries_by_name[name]
        _, absolute = _entry_path(database_root, name)
        observed = _inspect_parquet(
            absolute,
            database_root=database_root,
            expected_schema=schemas[name],
            expected_months=_SEGMENT_MONTHS[str(template["segment"])],
            label=f"independent final input {name}",
        )
        _require(
            all(observed[key] == entry[key] for key in observed),
            f"independent final-input difference: {name}",
        )
        receipts.append({"registration_name": name, **observed})
    manifest_payload = _canonical_json_bytes(dict(manifest))
    body: dict[str, object] = {
        "schema_version": 1,
        "status": "VERIFIED_PASSED",
        "verifier_id": VERIFIER_ID,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": _sha256_bytes(manifest_payload),
        "manifest_size_bytes": len(manifest_payload),
        "registry_contract_sha256": registry_sha,
        "stage12_binding_sha256": binding.binding_sha256,
        "approval_payload_sha256": approval.payload_sha256,
        "difference_count": 0,
        "entry_verifications": receipts,
    }
    fingerprint = _sha256_bytes(_canonical_json_bytes(body))
    return {
        **body,
        "verification_id": f"verify-stage13-inputs-{fingerprint[:20]}",
        "content_fingerprint": fingerprint,
    }


def independently_verify_final_input_registration_manifest(
    manifest: Mapping[str, object],
    approval_receipt: Stage13ApprovalReceipt | Mapping[str, object] | None,
) -> dict[str, object]:
    """Reopen all fixed inputs and return an independently derived receipt."""

    root = PROJECT_ROOT.absolute()
    try:
        binding = load_authenticated_stage12_binding(root)
        approval = require_stage13_unblind_authorization(
            approval_receipt, expected_stage12_binding=binding
        )
    except Stage13AuthorizationError as exc:
        raise FinalInputMaterializationError(str(exc)) from exc
    return _independently_verify(
        manifest,
        project_root=root,
        database_root=_configured_database_root(root),
        binding=binding,
        approval=approval,
    )


def _reauthenticate_publication_gate(
    *,
    project_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
) -> None:
    """Recheck both immutable authorities immediately before each publish."""

    try:
        observed_binding = load_authenticated_stage12_binding(project_root)
        observed_approval = require_stage13_unblind_authorization(
            approval,
            expected_stage12_binding=observed_binding,
        )
    except Stage13AuthorizationError as exc:
        raise FinalInputMaterializationError(str(exc)) from exc
    _require(
        observed_binding == binding
        and observed_binding.binding_sha256 == binding.binding_sha256
        and observed_approval.payload_sha256 == approval.payload_sha256,
        "final-input publication authority changed",
    )


def _ensure_publication_namespace(
    publication_root: Path,
    relative_namespace: Path,
) -> Path:
    _require(
        publication_root.is_absolute()
        and publication_root != Path(publication_root.anchor)
        and not relative_namespace.is_absolute()
        and ".." not in relative_namespace.parts,
        "final-input publication namespace is unsafe",
    )
    try:
        root_metadata = publication_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise FinalInputMaterializationError("publication root is absent") from exc
    _require(
        stat.S_ISDIR(root_metadata.st_mode) and not stat.S_ISLNK(root_metadata.st_mode),
        "publication root is unsafe",
    )
    current = publication_root
    for part in relative_namespace.parts:
        current /= part
        try:
            metadata = current.stat(follow_symlinks=False)
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o755)
            except FileExistsError:
                pass
            metadata = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise FinalInputMaterializationError(
                "publication namespace could not be inspected"
            ) from exc
        _require(
            stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
            "publication namespace is symlinked or not a directory",
        )
    return current


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_bytes(
    publication_root: Path,
    relative_path: Path,
    payload: bytes,
    *,
    replace: bool,
    label: str,
) -> None:
    """Publish immutable content or atomically replace the one fixed pointer."""

    _require(
        not relative_path.is_absolute()
        and ".." not in relative_path.parts
        and relative_path.suffix == ".json",
        f"{label} path is unsafe",
    )
    parent = _ensure_publication_namespace(publication_root, relative_path.parent)
    target = publication_root / relative_path
    if target.exists():
        current = _stable_regular_bytes(
            target,
            trusted_root=publication_root,
            label=label,
        )
        if current == payload:
            return
        _require(replace, f"content-addressed {label} differs")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if replace:
            if target.exists():
                metadata = target.stat(follow_symlinks=False)
                _require(
                    stat.S_ISREG(metadata.st_mode)
                    and not stat.S_ISLNK(metadata.st_mode)
                    and metadata.st_nlink == 1,
                    f"existing {label} is unsafe",
                )
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError:
                current = _stable_regular_bytes(
                    target,
                    trusted_root=publication_root,
                    label=label,
                )
                _require(current == payload, f"content-addressed {label} differs")
            finally:
                temporary.unlink(missing_ok=True)
        _fsync_directory(parent)
    except (OSError, FinalInputMaterializationError):
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    observed = _stable_regular_bytes(
        target,
        trusted_root=publication_root,
        label=label,
    )
    _require(observed == payload, f"published {label} differs")


def _publish_verified_final_input_registration(
    *,
    project_root: Path,
    database_root: Path,
    publication_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
) -> dict[str, object]:
    """Private injectable publisher used by the fixed production entrypoint."""

    manifest = _prepare_manifest(
        project_root=project_root,
        database_root=database_root,
        binding=binding,
        approval=approval,
    )
    manifest_payload = _canonical_json_bytes(manifest)
    manifest_path = PRIMARY_MANIFEST_NAMESPACE / f"{manifest['manifest_id']}.json"
    _reauthenticate_publication_gate(
        project_root=project_root,
        binding=binding,
        approval=approval,
    )
    _atomic_publish_bytes(
        publication_root,
        manifest_path,
        manifest_payload,
        replace=False,
        label="final-input registration manifest",
    )

    stored_manifest_payload = _stable_regular_bytes(
        publication_root / manifest_path,
        trusted_root=publication_root,
        label="final-input registration manifest",
    )
    _require(
        stored_manifest_payload == manifest_payload,
        "stored registration manifest differs",
    )
    stored_manifest = _json_object(
        stored_manifest_payload,
        label="stored registration manifest",
    )
    _validate_manifest_metadata(
        stored_manifest,
        project_root=project_root,
        database_root=database_root,
        binding=binding,
        approval=approval,
    )
    _reauthenticate_publication_gate(
        project_root=project_root,
        binding=binding,
        approval=approval,
    )
    verification = _independently_verify(
        stored_manifest,
        project_root=project_root,
        database_root=database_root,
        binding=binding,
        approval=approval,
    )
    verification_payload = _canonical_json_bytes(verification)
    verification_path = (
        VERIFICATION_NAMESPACE / f"{verification['verification_id']}.json"
    )
    _, _, registry_sha = _validated_templates(project_root, binding=binding)
    _validate_verification(
        verification,
        manifest=stored_manifest,
        manifest_payload=stored_manifest_payload,
        registry_sha=registry_sha,
        binding=binding,
        approval=approval,
    )
    _reauthenticate_publication_gate(
        project_root=project_root,
        binding=binding,
        approval=approval,
    )
    _atomic_publish_bytes(
        publication_root,
        verification_path,
        verification_payload,
        replace=False,
        label="final-input registration verification",
    )

    stored_verification_payload = _stable_regular_bytes(
        publication_root / verification_path,
        trusted_root=publication_root,
        label="final-input registration verification",
    )
    _require(
        stored_verification_payload == verification_payload,
        "stored registration verification differs",
    )
    stored_verification = _json_object(
        stored_verification_payload,
        label="stored registration verification",
    )
    _validate_verification(
        stored_verification,
        manifest=stored_manifest,
        manifest_payload=stored_manifest_payload,
        registry_sha=registry_sha,
        binding=binding,
        approval=approval,
    )
    pointer: dict[str, object] = {
        "schema_version": 1,
        "status": "VERIFIED_PASSED",
        "manifest_id": stored_manifest["manifest_id"],
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _sha256_bytes(stored_manifest_payload),
        "manifest_size_bytes": len(stored_manifest_payload),
        "verification_id": stored_verification["verification_id"],
        "verification_path": verification_path.as_posix(),
        "verification_sha256": _sha256_bytes(stored_verification_payload),
        "verification_size_bytes": len(stored_verification_payload),
        "registry_contract_sha256": registry_sha,
        "stage12_binding_sha256": binding.binding_sha256,
        "approval_payload_sha256": approval.payload_sha256,
    }
    pointer_payload = _canonical_json_bytes(pointer)
    _reauthenticate_publication_gate(
        project_root=project_root,
        binding=binding,
        approval=approval,
    )
    _atomic_publish_bytes(
        publication_root,
        VERIFIED_REGISTRATION_POINTER,
        pointer_payload,
        replace=True,
        label="verified final-input registration pointer",
    )
    return pointer


def publish_verified_final_input_registration(
    approval_receipt: Stage13ApprovalReceipt | Mapping[str, object] | None,
) -> dict[str, object]:
    """Atomically publish the fixed manifest, replay, and pointer—pointer last."""

    root = PROJECT_ROOT.absolute()
    try:
        binding = load_authenticated_stage12_binding(root)
        approval = require_stage13_unblind_authorization(
            approval_receipt,
            expected_stage12_binding=binding,
        )
    except Stage13AuthorizationError as exc:
        raise FinalInputMaterializationError(str(exc)) from exc
    return _publish_verified_final_input_registration(
        project_root=root,
        database_root=_configured_database_root(root),
        publication_root=root,
        binding=binding,
        approval=approval,
    )


def _validate_verification(
    verification: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    manifest_payload: bytes,
    registry_sha: str,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "verifier_id",
        "manifest_id",
        "manifest_sha256",
        "manifest_size_bytes",
        "registry_contract_sha256",
        "stage12_binding_sha256",
        "approval_payload_sha256",
        "difference_count",
        "entry_verifications",
        "verification_id",
        "content_fingerprint",
    }
    _require(
        set(verification) == expected_keys, "registration verification fields differ"
    )
    body = {
        key: verification[key]
        for key in verification
        if key not in {"verification_id", "content_fingerprint"}
    }
    fingerprint = _sha256_bytes(_canonical_json_bytes(body))
    _require(
        verification.get("schema_version") == 1
        and verification.get("status") == "VERIFIED_PASSED"
        and verification.get("verifier_id") == VERIFIER_ID
        and verification.get("manifest_id") == manifest["manifest_id"]
        and verification.get("manifest_sha256") == _sha256_bytes(manifest_payload)
        and verification.get("manifest_size_bytes") == len(manifest_payload)
        and verification.get("registry_contract_sha256") == registry_sha
        and verification.get("stage12_binding_sha256") == binding.binding_sha256
        and verification.get("approval_payload_sha256") == approval.payload_sha256
        and verification.get("difference_count") == 0
        and verification.get("content_fingerprint") == fingerprint
        and verification.get("verification_id")
        == f"verify-stage13-inputs-{fingerprint[:20]}",
        "registration verification identity differs",
    )
    receipts = verification.get("entry_verifications")
    entries = manifest["entries"]
    _require(
        isinstance(receipts, list)
        and len(receipts) == len(entries)  # type: ignore[arg-type]
        and [
            item.get("registration_name")
            for item in receipts
            if isinstance(item, Mapping)
        ]
        == [item["registration_name"] for item in entries],  # type: ignore[index]
        "registration verification entry set differs",
    )
    entry_by_name = {str(item["registration_name"]): item for item in entries}  # type: ignore[index]
    for receipt in receipts:
        _require(
            isinstance(receipt, Mapping)
            and set(receipt)
            == {
                "registration_name",
                "sha256",
                "size_bytes",
                "rows",
                "row_groups",
                "arrow_schema_sha256",
            },
            "registration verification entry is invalid",
        )
        entry = entry_by_name[str(receipt["registration_name"])]
        _require(
            all(
                receipt[key] == entry[key]
                for key in receipt
                if key != "registration_name"
            ),
            "registration verification entry differs",
        )


def _safe_metadata_path(
    project_root: Path,
    value: object,
    *,
    namespace: Path,
    label: str,
) -> Path:
    _require(isinstance(value, str), f"{label} path is missing")
    relative = Path(value)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.suffix == ".json",
        f"{label} path is unsafe",
    )
    try:
        relative.relative_to(namespace)
    except ValueError as exc:
        raise FinalInputMaterializationError(f"{label} path namespace differs") from exc
    return project_root / relative


def load_verified_final_access_broker(
    project_root: Path,
    approval_receipt: Stage13ApprovalReceipt | Mapping[str, object] | None,
) -> FinalAccessBroker:
    """Construct all broker registrations solely from the fixed verified pointer."""

    _require(project_root.is_absolute(), "project_root must be absolute")
    root = project_root.absolute()
    try:
        binding = load_authenticated_stage12_binding(root)
        approval = require_stage13_unblind_authorization(
            approval_receipt, expected_stage12_binding=binding
        )
    except Stage13AuthorizationError as exc:
        raise FinalInputMaterializationError(str(exc)) from exc
    database_root = _configured_database_root(root)
    pointer_payload = _stable_regular_bytes(
        root / VERIFIED_REGISTRATION_POINTER,
        trusted_root=root,
        label="verified final-input registration pointer",
    )
    pointer = _json_object(pointer_payload, label="verified registration pointer")
    pointer_keys = {
        "schema_version",
        "status",
        "manifest_id",
        "manifest_path",
        "manifest_sha256",
        "manifest_size_bytes",
        "verification_id",
        "verification_path",
        "verification_sha256",
        "verification_size_bytes",
        "registry_contract_sha256",
        "stage12_binding_sha256",
        "approval_payload_sha256",
    }
    _require(
        set(pointer) == pointer_keys
        and pointer.get("schema_version") == 1
        and pointer.get("status") == "VERIFIED_PASSED"
        and isinstance(pointer.get("manifest_id"), str)
        and _MANIFEST_ID.fullmatch(str(pointer["manifest_id"])) is not None
        and isinstance(pointer.get("verification_id"), str)
        and _VERIFICATION_ID.fullmatch(str(pointer["verification_id"])) is not None
        and pointer.get("stage12_binding_sha256") == binding.binding_sha256
        and pointer.get("approval_payload_sha256") == approval.payload_sha256,
        "verified registration pointer identity differs",
    )
    manifest_path = _safe_metadata_path(
        root,
        pointer["manifest_path"],
        namespace=PRIMARY_MANIFEST_NAMESPACE,
        label="registration manifest",
    )
    verification_path = _safe_metadata_path(
        root,
        pointer["verification_path"],
        namespace=VERIFICATION_NAMESPACE,
        label="registration verification",
    )
    manifest_payload = _stable_regular_bytes(
        manifest_path, trusted_root=root, label="registration manifest"
    )
    verification_payload = _stable_regular_bytes(
        verification_path, trusted_root=root, label="registration verification"
    )
    _require(
        pointer.get("manifest_sha256") == _sha256_bytes(manifest_payload)
        and pointer.get("manifest_size_bytes") == len(manifest_payload)
        and pointer.get("verification_sha256") == _sha256_bytes(verification_payload)
        and pointer.get("verification_size_bytes") == len(verification_payload),
        "verified registration pointer content identity differs",
    )
    manifest = _json_object(manifest_payload, label="registration manifest")
    templates, schemas, registry_sha = _validate_manifest_metadata(
        manifest,
        project_root=root,
        database_root=database_root,
        binding=binding,
        approval=approval,
    )
    _require(
        pointer.get("manifest_id") == manifest["manifest_id"]
        and pointer.get("registry_contract_sha256") == registry_sha,
        "verified registration pointer manifest binding differs",
    )
    verification = _json_object(verification_payload, label="registration verification")
    _validate_verification(
        verification,
        manifest=manifest,
        manifest_payload=manifest_payload,
        registry_sha=registry_sha,
        binding=binding,
        approval=approval,
    )
    _require(
        pointer.get("verification_id") == verification["verification_id"],
        "verified registration pointer verification binding differs",
    )
    records: dict[str, RegisteredFinalPayload] = {}
    for entry in manifest["entries"]:  # type: ignore[index]
        name = str(entry["registration_name"])
        records[name] = RegisteredFinalPayload(
            role=FinalPayloadRole(str(entry["role"])),
            path=Path(str(entry["absolute_path"])),
            trusted_root=database_root,
            sha256=str(entry["sha256"]),
            size_bytes=int(entry["size_bytes"]),
            allowed_columns=tuple(str(value) for value in entry["exact_columns"]),
            arrow_schema=schemas[name],
            arrow_schema_sha256=str(entry["arrow_schema_sha256"]),
            holding_month_column="holding_month",
            holding_months=tuple(str(value) for value in entry["holding_months"]),
        )
    return FinalAccessBroker(
        project_root=root,
        approval_receipt=approval,
        _authenticated_registrations=records,
        _authenticated_registration_identity={
            "verified_pointer_path": VERIFIED_REGISTRATION_POINTER.as_posix(),
            "verified_pointer_sha256": _sha256_bytes(pointer_payload),
            "manifest_id": str(manifest["manifest_id"]),
            "manifest_sha256": _sha256_bytes(manifest_payload),
            "verification_id": str(verification["verification_id"]),
            "verification_sha256": _sha256_bytes(verification_payload),
            "registry_contract_sha256": registry_sha,
        },
        _registration_token=_MANIFEST_REGISTRATION_TOKEN,
    )


__all__ = [
    "FINAL_INPUT_NAMESPACE",
    "FinalDailyAccountingProjection",
    "FinalInputMaterializationError",
    "VERIFIED_REGISTRATION_POINTER",
    "independently_verify_final_input_registration_manifest",
    "load_verified_final_access_broker",
    "prepare_final_input_registration_manifest",
    "project_final_daily_accounting_inputs",
    "publish_verified_final_input_registration",
]
