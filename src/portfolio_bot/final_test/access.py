"""Fail-closed, pre-open access guard for Stage 13 final-period payloads.

Registrations bind an exact absolute path, content identity, role, projection,
and month domain. Authorization, role, month, and column checks all complete
before the broker-controlled Parquet reader can open the path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

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
    FinalCalendarError,
    FinalPeriodSegment,
    final_holding_month_descriptor,
    validate_final_month_sequence,
)


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REGISTRATION_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{2,127}$")
_COLUMN_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.:-]{0,127}$")
_REQUEST_ISSUANCE_TOKEN = object()
_MANIFEST_REGISTRATION_TOKEN = object()
_REGISTRATION_IDENTITY_KEYS = frozenset(
    {
        "verified_pointer_path",
        "verified_pointer_sha256",
        "manifest_id",
        "manifest_sha256",
        "verification_id",
        "verification_sha256",
        "registry_contract_sha256",
    }
)

_ARROW_TYPES: Mapping[str, pa.DataType] = MappingProxyType(
    {
        "bool": pa.bool_(),
        "date32": pa.date32(),
        "decimal128(38,1)": pa.decimal128(38, 1),
        "decimal128(38,2)": pa.decimal128(38, 2),
        "decimal128(38,6)": pa.decimal128(38, 6),
        "decimal128(38,12)": pa.decimal128(38, 12),
        "float64": pa.float64(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "list<string>": pa.list_(pa.field("element", pa.string())),
        "string": pa.string(),
    }
)
_FORWARD_TOKENS = frozenset({"forward", "future", "fwd", "lead", "next"})
_FROZEN_BETA_LABEL_COLUMNS = frozenset(
    {
        "beta_label_mktrf",
        "beta_label_smb",
        "beta_label_hml",
        "beta_label_rmw",
        "beta_label_cma",
        "beta_label_umd",
        "beta_label_idiosyncratic_volatility",
    }
)


class FinalAccessError(RuntimeError):
    """Raised before opening when a Stage 13 access request is not exact."""


class FinalPayloadRole(str, Enum):
    """Closed semantic roles for final-period empirical inputs."""

    CALENDAR_METADATA = "final_calendar_metadata"
    PREDICTOR = "final_predictor"
    RISK_INPUT = "final_risk_input"
    COST_INPUT = "final_cost_input"
    MARKET_INPUT = "final_market_input"
    ACCOUNTING_INPUT = "final_accounting_input"
    FACTOR_RETURN = "final_factor_return"
    BENCHMARK_RETURN = "final_benchmark_return"
    LABEL = "final_label"
    OUTCOME = "final_outcome"
    FORWARD_RETURN = "final_forward_return"
    PARTIAL_DECEMBER_DESCRIPTIVE_RETURN = "partial_december_descriptive_return"


_ALL_SEGMENTS = frozenset(FinalPeriodSegment)
_COMPLETE_SEGMENTS = frozenset(
    {
        FinalPeriodSegment.PRIMARY_FULL,
        FinalPeriodSegment.CORE_COMPLETE,
    }
)
_PARTIAL_ONLY = frozenset({FinalPeriodSegment.PARTIAL_DECEMBER_DESCRIPTIVE})
_ROLE_SEGMENTS: Mapping[FinalPayloadRole, frozenset[FinalPeriodSegment]] = (
    MappingProxyType(
        {
            FinalPayloadRole.CALENDAR_METADATA: _ALL_SEGMENTS,
            FinalPayloadRole.PREDICTOR: _ALL_SEGMENTS,
            FinalPayloadRole.RISK_INPUT: _ALL_SEGMENTS,
            FinalPayloadRole.COST_INPUT: _ALL_SEGMENTS,
            FinalPayloadRole.MARKET_INPUT: _ALL_SEGMENTS,
            FinalPayloadRole.ACCOUNTING_INPUT: _ALL_SEGMENTS,
            FinalPayloadRole.FACTOR_RETURN: _ALL_SEGMENTS,
            FinalPayloadRole.BENCHMARK_RETURN: _ALL_SEGMENTS,
            FinalPayloadRole.LABEL: _COMPLETE_SEGMENTS,
            FinalPayloadRole.OUTCOME: _COMPLETE_SEGMENTS,
            FinalPayloadRole.FORWARD_RETURN: _COMPLETE_SEGMENTS,
            FinalPayloadRole.PARTIAL_DECEMBER_DESCRIPTIVE_RETURN: _PARTIAL_ONLY,
        }
    )
)


def _require_hash(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise FinalAccessError(
            f"{field_name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _normalize_role(value: FinalPayloadRole | str) -> FinalPayloadRole:
    try:
        return value if isinstance(value, FinalPayloadRole) else FinalPayloadRole(value)
    except (TypeError, ValueError) as exc:
        raise FinalAccessError("unregistered final-period payload role") from exc


def _normalize_columns(values: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise FinalAccessError(f"{field_name} must be a column sequence")
    try:
        columns = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise FinalAccessError(f"{field_name} must be iterable") from exc
    if (
        not columns
        or len(columns) != len(set(columns))
        or any(
            not isinstance(value, str) or _COLUMN_NAME.fullmatch(value) is None
            for value in columns
        )
    ):
        raise FinalAccessError(
            f"{field_name} must be a unique, nonempty sequence of column names"
        )
    return columns


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _parse_exact_arrow_schema(value: object) -> pa.Schema:
    """Parse the deliberately small, metadata-free Stage 13 type grammar."""

    if not isinstance(value, list) or not value:
        raise FinalAccessError("registered Arrow schema must be a nonempty list")
    fields: list[pa.Field] = []
    for raw_field in value:
        if not isinstance(raw_field, Mapping) or set(raw_field) != {
            "name",
            "type",
            "nullable",
        }:
            raise FinalAccessError("registered Arrow field is not exact")
        name = raw_field.get("name")
        type_name = raw_field.get("type")
        nullable = raw_field.get("nullable")
        if (
            not isinstance(name, str)
            or _COLUMN_NAME.fullmatch(name) is None
            or not isinstance(type_name, str)
            or type_name not in _ARROW_TYPES
            or type(nullable) is not bool
        ):
            raise FinalAccessError("registered Arrow field is invalid")
        fields.append(pa.field(name, _ARROW_TYPES[type_name], nullable=nullable))
    schema = pa.schema(fields)
    if len(schema.names) != len(set(schema.names)):
        raise FinalAccessError("registered Arrow schema has duplicate fields")
    return schema


def _column_tokens(value: str) -> tuple[str, ...]:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return tuple(token for token in re.split(r"[^a-z0-9]+", snake.lower()) if token)


def _is_prohibited_predictor_column(value: str) -> bool:
    """Reject outcome semantics independently of a manifest allowlist.

    The frozen feature registry contains six legitimate ``beta_label_*``
    point-in-time beta estimates.  That exact prefix is the sole label-token
    exception; generic label, target, outcome, and forward-looking names fail.
    """

    ordered = _column_tokens(value)
    tokens = set(ordered)
    if "outcome" in tokens or "target" in tokens or bool(tokens & _FORWARD_TOKENS):
        return True
    if "label" in tokens and value not in _FROZEN_BETA_LABEL_COLUMNS:
        return True
    if "return" in tokens and (
        bool(tokens & _FORWARD_TOKENS) or "residual" in tokens or "realized" in tokens
    ):
        return True
    return False


def _normalize_path(value: object) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise FinalAccessError("registered final-period path must be absolute")
    if ".." in value.parts:
        raise FinalAccessError("registered final-period path cannot contain '..'")
    return value


def _normalize_trusted_root(value: object) -> Path:
    root = _normalize_path(value)
    if root == Path(root.anchor):
        raise FinalAccessError("registered trusted root is too broad")
    return root


def _require_confined_payload_path(path: Path, *, trusted_root: Path) -> None:
    try:
        relative = path.relative_to(trusted_root)
    except ValueError as exc:
        raise FinalAccessError("registered payload escaped its trusted root") from exc
    if not relative.parts:
        raise FinalAccessError("registered payload cannot equal its trusted root")


def _require_no_symlink_components(path: Path, *, trusted_root: Path) -> None:
    """Authenticate every extant component immediately before payload open."""

    _require_confined_payload_path(path, trusted_root=trusted_root)
    current = Path(trusted_root.anchor)
    root_parts = trusted_root.parts[1:]
    relative = path.relative_to(trusted_root)
    all_parts = (*root_parts, *relative.parts)
    for index, component in enumerate(all_parts):
        current /= component
        try:
            observed = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise FinalAccessError("registered payload path is absent") from exc
        final = index == len(all_parts) - 1
        if (
            stat.S_ISLNK(observed.st_mode)
            or (final and not stat.S_ISREG(observed.st_mode))
            or (not final and not stat.S_ISDIR(observed.st_mode))
        ):
            raise FinalAccessError("registered payload path is unsafe")


def _validate_role_months(
    role: FinalPayloadRole,
    holding_months: object,
) -> tuple[str, ...]:
    try:
        months = validate_final_month_sequence(
            holding_months,  # type: ignore[arg-type]
            allowed_segments=_ROLE_SEGMENTS[role],
            allow_cross_segment=(role is FinalPayloadRole.CALENDAR_METADATA),
        )
    except FinalCalendarError as exc:
        raise FinalAccessError(str(exc)) from exc
    if role is FinalPayloadRole.PARTIAL_DECEMBER_DESCRIPTIVE_RETURN and months != (
        PARTIAL_DECEMBER_HOLDING_MONTH,
    ):
        raise FinalAccessError(
            "partial-December return role must be exactly December 2025"
        )
    return months


@dataclass(frozen=True, slots=True)
class RegisteredFinalPayload:
    """Manifest-authenticated description of one final-period table."""

    role: FinalPayloadRole
    path: Path
    trusted_root: Path
    sha256: str
    size_bytes: int
    allowed_columns: tuple[str, ...]
    arrow_schema: pa.Schema
    arrow_schema_sha256: str
    holding_month_column: str
    holding_months: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, FinalPayloadRole):
            raise FinalAccessError("role must be a FinalPayloadRole")
        path = _normalize_path(self.path)
        trusted_root = _normalize_trusted_root(self.trusted_root)
        _require_confined_payload_path(path, trusted_root=trusted_root)
        _require_hash(self.sha256, field_name="sha256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise FinalAccessError("size_bytes must be a nonnegative integer")
        columns = _normalize_columns(
            self.allowed_columns,
            field_name="allowed_columns",
        )
        if not isinstance(self.arrow_schema, pa.Schema):
            raise FinalAccessError("arrow_schema must be an exact Arrow Schema")
        if self.arrow_schema.metadata is not None:
            raise FinalAccessError("registered Arrow schema metadata is prohibited")
        if tuple(self.arrow_schema.names) != columns:
            raise FinalAccessError(
                "registered Arrow schema/order differs from allowed_columns"
            )
        _require_hash(
            self.arrow_schema_sha256,
            field_name="arrow_schema_sha256",
        )
        if _schema_sha256(self.arrow_schema) != self.arrow_schema_sha256:
            raise FinalAccessError("registered Arrow schema digest differs")
        if self.role is FinalPayloadRole.PREDICTOR and any(
            _is_prohibited_predictor_column(column) for column in columns
        ):
            raise FinalAccessError(
                "predictor schema contains prohibited outcome/target/future/label semantics"
            )
        if (
            not isinstance(self.holding_month_column, str)
            or self.holding_month_column not in columns
        ):
            raise FinalAccessError(
                "holding_month_column must be in the exact column allowlist"
            )
        months = _validate_role_months(self.role, self.holding_months)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "trusted_root", trusted_root)
        object.__setattr__(self, "allowed_columns", columns)
        object.__setattr__(self, "holding_months", months)


@dataclass(frozen=True, slots=True, init=False)
class AuthorizedFinalTableRequest:
    """Privately issued request that is usable only by its issuing broker."""

    registration_name: str
    record: RegisteredFinalPayload
    columns: tuple[str, ...]
    holding_months: tuple[str, ...]
    approval_payload_sha256: str
    stage12_binding_sha256: str
    _broker_token: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        registration_name: str,
        record: RegisteredFinalPayload,
        columns: tuple[str, ...],
        holding_months: tuple[str, ...],
        approval_payload_sha256: str,
        stage12_binding_sha256: str,
        _broker_token: object,
        _issuance_token: object,
    ) -> None:
        if _issuance_token is not _REQUEST_ISSUANCE_TOKEN:
            raise FinalAccessError(
                "final table requests must be issued by a live access broker"
            )
        object.__setattr__(self, "registration_name", registration_name)
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "holding_months", holding_months)
        object.__setattr__(
            self,
            "approval_payload_sha256",
            approval_payload_sha256,
        )
        object.__setattr__(
            self,
            "stage12_binding_sha256",
            stage12_binding_sha256,
        )
        object.__setattr__(self, "_broker_token", _broker_token)


@dataclass(frozen=True, slots=True)
class FinalAccessEvent:
    registration_name: str
    role: str
    holding_months: tuple[str, ...]
    columns: tuple[str, ...]
    sha256: str
    size_bytes: int
    row_count: int


@dataclass(frozen=True, slots=True)
class FinalAccessLedger:
    """Immutable successful-read ledger plus pre-open rejection count."""

    stage12_binding_sha256: str
    approval_payload_sha256: str
    successful_reads: tuple[FinalAccessEvent, ...]
    rejected_before_open: int

    @property
    def rows_read(self) -> int:
        return sum(event.row_count for event in self.successful_reads)


_SEGMENT_MONTHS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "all_metadata_only": FINAL_HOLDING_MONTHS,
        "primary_full": PRIMARY_FULL_HOLDING_MONTHS,
        "core_complete": CORE_COMPLETE_HOLDING_MONTHS,
        "partial_december_descriptive": (PARTIAL_DECEMBER_HOLDING_MONTH,),
    }
)


def _load_authenticated_registry(
    project_root: Path,
    *,
    binding: Stage12VerifiedBinding,
) -> Mapping[str, Mapping[str, object]]:
    def read_json(
        path: Path,
        *,
        object_name: str,
    ) -> tuple[dict[str, object], bytes, str]:
        try:
            relative = path.absolute().relative_to(project_root.absolute())
        except ValueError as exc:
            raise FinalAccessError(f"{object_name} escaped the project root") from exc
        current = project_root.absolute()
        for index, component in enumerate(relative.parts):
            current = current / component
            try:
                metadata = current.stat(follow_symlinks=False)
            except OSError as exc:
                raise FinalAccessError(f"{object_name} is absent") from exc
            final = index == len(relative.parts) - 1
            if (
                stat.S_ISLNK(metadata.st_mode)
                or (
                    final
                    and (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1)
                )
                or (not final and not stat.S_ISDIR(metadata.st_mode))
            ):
                raise FinalAccessError(f"{object_name} has an unsafe path")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise FinalAccessError(f"{object_name} could not be opened safely") from exc
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size > 32 * 1024 * 1024
            ):
                raise FinalAccessError(f"{object_name} is not a bounded file")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field) for field in identity_fields
        ):
            raise FinalAccessError(f"{object_name} changed while being read")
        payload = b"".join(chunks)
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FinalAccessError(f"{object_name} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise FinalAccessError(f"{object_name} must be a JSON object")
        return value, payload, hashlib.sha256(payload).hexdigest()

    verified_pointer_path = (
        project_root / "artifacts/phase_06_final_test/LATEST_STAGE_12_VERIFIED.json"
    )
    verified_pointer, _, verified_pointer_sha = read_json(
        verified_pointer_path,
        object_name="Stage 12 verified pointer",
    )
    if (
        verified_pointer_sha != binding.verified_pointer_sha256
        or verified_pointer.get("primary_freeze_id") != binding.build_id
        or verified_pointer.get("primary_manifest_sha256") != binding.manifest_sha256
        or verified_pointer.get("primary_pointer_path")
        != "artifacts/phase_06_final_test/LATEST_STAGE_12_PRIMARY.json"
    ):
        raise FinalAccessError("Stage 12 verified pointer differs from binding")
    primary_pointer_path = (
        project_root / "artifacts/phase_06_final_test/LATEST_STAGE_12_PRIMARY.json"
    )
    pointer, _, pointer_sha = read_json(
        primary_pointer_path,
        object_name="Stage 12 primary pointer",
    )
    if (
        pointer_sha != verified_pointer.get("primary_pointer_sha256")
        or pointer.get("freeze_id") != binding.build_id
        or pointer.get("manifest_sha256") != binding.manifest_sha256
    ):
        raise FinalAccessError("Stage 12 primary pointer differs from binding")
    bundle = project_root / str(pointer.get("bundle_path"))
    manifest, manifest_payload, observed_manifest_sha = read_json(
        bundle / "pre_final_manifest.json",
        object_name="Stage 12 primary manifest",
    )
    wrapper, registry_payload, observed_registry_sha = read_json(
        bundle / "final_input_registry.json",
        object_name="Stage 12 final-input registry",
    )
    if (
        observed_manifest_sha != binding.manifest_sha256
        or manifest.get("members", {})
        .get("final_input_registry.json", {})
        .get("sha256")
        != observed_registry_sha
    ):
        raise FinalAccessError("frozen final-input registry hash differs")
    registry = wrapper.get("registry")
    if not isinstance(registry, dict):
        raise FinalAccessError("frozen final-input registry is missing")
    rules = registry.get("global_rules")
    templates = registry.get("templates")
    if (
        not isinstance(rules, dict)
        or rules.get("caller_created_template_permitted") is not False
        or rules.get("caller_supplied_reader_permitted") is not False
        or rules.get("primary_core_partial_splicing_permitted") is not False
        or not isinstance(templates, list)
    ):
        raise FinalAccessError("frozen final-input registry rules differ")
    result: dict[str, Mapping[str, object]] = {}
    for item in templates:
        if not isinstance(item, dict):
            raise FinalAccessError("frozen final-input template is invalid")
        name = str(item.get("registration_name"))
        if name in result or name == "None":
            raise FinalAccessError("frozen final-input template identity differs")
        result[name] = MappingProxyType(dict(item))
    return MappingProxyType(result)


class FinalAccessBroker:
    """Live broker that authenticates metadata but opens no final payload."""

    def __init__(
        self,
        *,
        project_root: Path,
        approval_receipt: Stage13ApprovalReceipt | Mapping[str, object] | None,
        _authenticated_registrations: (
            Mapping[str, RegisteredFinalPayload] | None
        ) = None,
        _authenticated_registration_identity: Mapping[str, str] | None = None,
        _registration_token: object | None = None,
    ) -> None:
        if not isinstance(project_root, Path) or not project_root.is_absolute():
            raise FinalAccessError("project_root must be an absolute Path")
        try:
            stage12_binding = load_authenticated_stage12_binding(project_root)
            approval = require_stage13_unblind_authorization(
                approval_receipt,
                expected_stage12_binding=stage12_binding,
            )
        except Stage13AuthorizationError as exc:
            raise FinalAccessError(str(exc)) from exc
        self._stage12_binding = stage12_binding
        self._approval = approval
        self._templates = _load_authenticated_registry(
            project_root,
            binding=stage12_binding,
        )
        if _authenticated_registrations is None:
            registrations: dict[str, RegisteredFinalPayload] = {}
        else:
            if _registration_token is not _MANIFEST_REGISTRATION_TOKEN:
                raise FinalAccessError(
                    "registrations require a verified registration manifest"
                )
            registrations = dict(_authenticated_registrations)
            if set(registrations) != set(self._templates):
                raise FinalAccessError(
                    "verified registration manifest does not contain the closed registry"
                )
            for name, record in registrations.items():
                self._validate_manifest_registration(name, record)
        if _authenticated_registration_identity is None:
            registration_identity: dict[str, str] = {}
        else:
            if _registration_token is not _MANIFEST_REGISTRATION_TOKEN:
                raise FinalAccessError(
                    "registration identity requires a verified registration manifest"
                )
            registration_identity = dict(_authenticated_registration_identity)
            if (
                set(registration_identity) != _REGISTRATION_IDENTITY_KEYS
                or registration_identity.get("verified_pointer_path")
                != "artifacts/phase_06_final_test/"
                "LATEST_STAGE_13_REGISTRATION_VERIFIED.json"
                or any(
                    _HEX64.fullmatch(registration_identity.get(name, "")) is None
                    for name in (
                        "verified_pointer_sha256",
                        "manifest_sha256",
                        "verification_sha256",
                        "registry_contract_sha256",
                    )
                )
                or re.fullmatch(
                    r"stage13-inputs-[0-9a-f]{24}",
                    registration_identity.get("manifest_id", ""),
                )
                is None
                or re.fullmatch(
                    r"verify-stage13-inputs-[0-9a-f]{20}",
                    registration_identity.get("verification_id", ""),
                )
                is None
            ):
                raise FinalAccessError("verified registration identity differs")
        self._registrations = registrations
        self._registration_identity = MappingProxyType(registration_identity)
        self._broker_token = object()
        self._successful_reads: list[FinalAccessEvent] = []
        self._rejected_before_open = 0

    @property
    def registration_identity(self) -> Mapping[str, str]:
        """Read-only identities of the verified pointer, manifest, and replay."""

        return self._registration_identity

    def register(
        self,
        registration_name: str,
        record: RegisteredFinalPayload,
    ) -> None:
        """Reject caller-created registrations.

        Production registrations are installed atomically by the verified
        manifest loader; no public record-by-record registration path exists.
        """

        del registration_name, record
        raise FinalAccessError(
            "caller-created registrations are prohibited; load a verified manifest"
        )

    def _validate_manifest_registration(
        self,
        registration_name: str,
        record: RegisteredFinalPayload,
    ) -> None:
        """Validate one loader-issued record against the frozen template."""

        if (
            not isinstance(registration_name, str)
            or _REGISTRATION_NAME.fullmatch(registration_name) is None
        ):
            raise FinalAccessError("registration_name is not a bounded identifier")
        if not isinstance(record, RegisteredFinalPayload):
            raise FinalAccessError("record must be a RegisteredFinalPayload")
        template = self._templates.get(registration_name)
        if template is None:
            raise FinalAccessError(
                "registration is absent from the frozen Stage 12 registry"
            )
        try:
            template_role = FinalPayloadRole(str(template["role"]))
            template_months = _SEGMENT_MONTHS[str(template["segment"])]
            required_columns = tuple(
                str(value) for value in template["required_columns"]
            )
            template_schema = _parse_exact_arrow_schema(template["arrow_schema"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FinalAccessError("frozen registration template is invalid") from exc
        if (
            record.role is not template_role
            or record.holding_months != template_months
            or record.allowed_columns != required_columns
            or not record.arrow_schema.equals(template_schema, check_metadata=True)
            or record.arrow_schema_sha256 != _schema_sha256(template_schema)
        ):
            raise FinalAccessError(
                "registration differs from the frozen Stage 12 template"
            )

    def _reject(self, message: str) -> None:
        self._rejected_before_open += 1
        raise FinalAccessError(message)

    def request_table(
        self,
        registration_name: str,
        *,
        role: FinalPayloadRole | str,
        holding_months: object,
        columns: object,
    ) -> AuthorizedFinalTableRequest:
        """Validate exact semantics and issue a request without opening a file."""

        try:
            record = self._registrations[registration_name]
        except (KeyError, TypeError):
            self._reject("unregistered final-period payload")
        requested_role = _normalize_role(role)
        if requested_role is not record.role:
            self._reject("requested role differs from registered role")
        try:
            requested_months = _validate_role_months(
                requested_role,
                holding_months,
            )
            requested_columns = _normalize_columns(
                columns,
                field_name="requested columns",
            )
        except FinalAccessError as exc:
            self._reject(str(exc))
        if any(month not in record.holding_months for month in requested_months):
            self._reject("requested months exceed the registered month domain")
        if any(column not in record.allowed_columns for column in requested_columns):
            self._reject("requested columns exceed the registered projection")
        if record.holding_month_column not in requested_columns:
            self._reject(
                "requested projection must retain the registered holding-month column"
            )
        return AuthorizedFinalTableRequest(
            registration_name=registration_name,
            record=record,
            columns=requested_columns,
            holding_months=requested_months,
            approval_payload_sha256=self._approval.payload_sha256,
            stage12_binding_sha256=self._stage12_binding.binding_sha256,
            _broker_token=self._broker_token,
            _issuance_token=_REQUEST_ISSUANCE_TOKEN,
        )

    def _validate_live_request(
        self,
        request: AuthorizedFinalTableRequest,
    ) -> RegisteredFinalPayload:
        try:
            require_stage13_unblind_authorization(
                self._approval,
                expected_stage12_binding=self._stage12_binding,
            )
        except Stage13AuthorizationError as exc:
            self._reject(str(exc))
        if (
            not isinstance(request, AuthorizedFinalTableRequest)
            or request._broker_token is not self._broker_token
        ):
            self._reject("request was not issued by this live access broker")
        if request.approval_payload_sha256 != self._approval.payload_sha256:
            self._reject("request approval identity differs")
        if request.stage12_binding_sha256 != self._stage12_binding.binding_sha256:
            self._reject("request Stage 12 binding differs")
        current = self._registrations.get(request.registration_name)
        if current is None or current != request.record:
            self._reject("request registration is absent or changed")
        # Revalidate all semantics before any callback can run.
        _validate_role_months(current.role, request.holding_months)
        if any(month not in current.holding_months for month in request.holding_months):
            self._reject("request months no longer match registration")
        if (
            any(column not in current.allowed_columns for column in request.columns)
            or current.holding_month_column not in request.columns
        ):
            self._reject("request projection no longer matches registration")
        return current

    def _record_success(
        self,
        request: AuthorizedFinalTableRequest,
        *,
        record: RegisteredFinalPayload,
        row_count: int,
    ) -> None:
        self._successful_reads.append(
            FinalAccessEvent(
                registration_name=request.registration_name,
                role=record.role.value,
                holding_months=request.holding_months,
                columns=request.columns,
                sha256=record.sha256,
                size_bytes=record.size_bytes,
                row_count=row_count,
            )
        )

    def read_guarded_parquet(
        self,
        request: AuthorizedFinalTableRequest,
    ) -> pa.Table:
        """Read one exact projected/month-filtered Parquet registration.

        The broker itself opens, hashes, projects, filters, and validates the
        table.  No caller-supplied reader or self-reported digest participates.
        """

        record = self._validate_live_request(request)
        _require_no_symlink_components(
            record.path,
            trusted_root=record.trusted_root,
        )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(record.path, flags)
        except OSError as exc:
            raise FinalAccessError(
                "authorized payload could not be opened safely"
            ) from exc
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != record.size_bytes
            ):
                raise FinalAccessError(
                    "authorized payload has an unsafe or changed identity"
                )
            digest = hashlib.sha256()
            while block := os.read(descriptor, 8 * 1024 * 1024):
                digest.update(block)
            if digest.hexdigest() != record.sha256:
                raise FinalAccessError(
                    "opened payload differs from its registered content identity"
                )
            os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
                observed_schema = pq.ParquetFile(stream).schema_arrow
            if not observed_schema.equals(
                record.arrow_schema,
                check_metadata=True,
            ):
                raise FinalAccessError(
                    "authorized Parquet full Arrow schema differs from registration"
                )
            if _schema_sha256(observed_schema) != record.arrow_schema_sha256:
                raise FinalAccessError("authorized Parquet Arrow schema digest differs")
            os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
                month_table = pq.read_table(
                    stream,
                    columns=[record.holding_month_column],
                )
            try:
                registered_month_values = pc.cast(
                    month_table[record.holding_month_column],
                    pa.string(),
                    safe=True,
                )
            except pa.ArrowException as exc:
                raise FinalAccessError(
                    "holding-month column cannot be normalized"
                ) from exc
            observed_registered_months = tuple(
                sorted(
                    {
                        value
                        for value in registered_month_values.to_pylist()
                        if value is not None
                    }
                )
            )
            if observed_registered_months != record.holding_months:
                raise FinalAccessError(
                    "authorized Parquet month domain differs from registration"
                )
            os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
                table = pq.read_table(
                    stream,
                    columns=list(request.columns),
                    filters=[
                        (
                            record.holding_month_column,
                            "in",
                            list(request.holding_months),
                        )
                    ],
                )
            after = os.fstat(descriptor)
        except FinalAccessError:
            raise
        except (OSError, pa.ArrowException) as exc:
            raise FinalAccessError(
                "authorized Parquet payload could not be read"
            ) from exc
        finally:
            os.close(descriptor)
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field) for field in identity_fields
        ):
            raise FinalAccessError("authorized payload changed while being read")
        if tuple(table.column_names) != request.columns:
            raise FinalAccessError("authorized Parquet projection schema/order differs")
        month_column = table[record.holding_month_column]
        try:
            normalized_months = pc.cast(month_column, pa.string(), safe=True)
        except pa.ArrowException as exc:
            raise FinalAccessError("holding-month column cannot be normalized") from exc
        requested_set = set(request.holding_months)
        selected = table
        observed = tuple(
            sorted(
                set(
                    value
                    for value in normalized_months.to_pylist()
                    if value is not None
                )
            )
        )
        if (
            selected.num_rows <= 0
            or observed != request.holding_months
            or any(value not in requested_set for value in observed)
        ):
            raise FinalAccessError(
                "authorized Parquet rows do not exactly cover requested months"
            )
        self._record_success(
            request,
            record=record,
            row_count=selected.num_rows,
        )
        return selected

    def ledger(self) -> FinalAccessLedger:
        return FinalAccessLedger(
            stage12_binding_sha256=self._stage12_binding.binding_sha256,
            approval_payload_sha256=self._approval.payload_sha256,
            successful_reads=tuple(self._successful_reads),
            rejected_before_open=self._rejected_before_open,
        )


__all__ = [
    "AuthorizedFinalTableRequest",
    "FinalAccessBroker",
    "FinalAccessError",
    "FinalAccessEvent",
    "FinalAccessLedger",
    "FinalPayloadRole",
    "RegisteredFinalPayload",
]
