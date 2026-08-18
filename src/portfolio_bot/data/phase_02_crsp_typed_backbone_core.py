"""Exact Arrow typing and logical hashing for the Phase 2 CRSP backbone.

This module contains no raw-source scanning and no publication logic.  It is a
small primary-build utility: the independent verifier deliberately reimplements
the wire format instead of importing this file.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa


LOGICAL_HASH_DOMAIN = b"phase02-crsp-typed-logical-v1\x00"
_DECIMAL_RE = re.compile(r"decimal128\(([1-9][0-9]*),([0-9]+)\)\Z")


class TypedBackboneCoreError(ValueError):
    """Raised when a schema or typed value violates the locked wire contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the project's deterministic UTF-8 JSON representation."""

    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypedBackboneCoreError(f"Value is not canonical JSON: {exc}") from exc
    return (rendered + "\n").encode("utf-8")


def parse_arrow_type(logical_type: str) -> pa.DataType:
    """Parse the deliberately small logical-type vocabulary used by the build."""

    fixed: dict[str, pa.DataType] = {
        "bool": pa.bool_(),
        "date32": pa.date32(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "utf8": pa.string(),
    }
    if logical_type in fixed:
        return fixed[logical_type]
    match = _DECIMAL_RE.fullmatch(logical_type)
    if match is None:
        raise TypedBackboneCoreError(
            f"Unsupported logical type {logical_type!r}"
        )
    precision, scale = (int(value) for value in match.groups())
    if precision > 38:
        raise TypedBackboneCoreError("decimal128 precision cannot exceed 38")
    if scale > precision:
        raise TypedBackboneCoreError("decimal128 scale cannot exceed precision")
    return pa.decimal128(precision, scale)


def _metadata_bytes(metadata: Mapping[str, Any]) -> dict[bytes, bytes]:
    result: dict[bytes, bytes] = {}
    for key, value in sorted(metadata.items()):
        if not isinstance(key, str) or not key:
            raise TypedBackboneCoreError("Field metadata keys must be nonempty strings")
        if isinstance(value, (dict, list, tuple, bool, int)) or value is None:
            payload = canonical_json_bytes(value)[:-1]
        elif isinstance(value, str):
            payload = value.encode("utf-8")
        else:
            raise TypedBackboneCoreError(
                f"Unsupported metadata value for {key!r}: {type(value).__name__}"
            )
        result[key.encode("utf-8")] = payload
    return result


@dataclass(frozen=True)
class FieldSpec:
    """One locked field in schema order."""

    name: str
    logical_type: str
    nullable: bool
    metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FieldSpec":
        if "logical_type" in value and "type" in value:
            raise TypedBackboneCoreError(
                "Field specification cannot contain both type and logical_type"
            )
        type_key = "logical_type" if "logical_type" in value else "type"
        required = {"name", type_key, "nullable"}
        missing = required - set(value)
        if missing:
            raise TypedBackboneCoreError(
                f"Field specification is missing {sorted(missing)}"
            )
        name = value["name"]
        logical_type = value[type_key]
        nullable = value["nullable"]
        if not isinstance(name, str) or not name:
            raise TypedBackboneCoreError("Field name must be a nonempty string")
        if not isinstance(logical_type, str):
            raise TypedBackboneCoreError("logical_type must be a string")
        if type(nullable) is not bool:
            raise TypedBackboneCoreError("nullable must be a boolean")
        parse_arrow_type(logical_type)
        metadata = {
            key: item
            for key, item in value.items()
            if key not in required
        }
        return cls(name, logical_type, nullable, metadata)

    def arrow_field(self) -> pa.Field:
        return pa.field(
            self.name,
            parse_arrow_type(self.logical_type),
            nullable=self.nullable,
            metadata=_metadata_bytes(self.metadata),
        )

    def semantic_mapping(self) -> dict[str, Any]:
        return {
            "logical_type": self.logical_type,
            "name": self.name,
            "nullable": self.nullable,
            **dict(self.metadata),
        }


@dataclass(frozen=True)
class TableSpec:
    """Validated schema-registry contract for one physical typed table."""

    name: str
    role: str
    primary_source: str
    grain: str
    primary_key: tuple[str, ...]
    sort_key: tuple[str, ...]
    expected_rows: int
    fields: tuple[FieldSpec, ...]
    metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TableSpec":
        required = {
            "table",
            "role",
            "primary_source",
            "grain",
            "primary_key",
            "sort_key",
            "expected_rows",
            "columns",
        }
        missing = required - set(value)
        if missing:
            raise TypedBackboneCoreError(
                f"Table specification is missing {sorted(missing)}"
            )
        scalar_names = ("table", "role", "primary_source", "grain")
        if any(not isinstance(value[name], str) or not value[name] for name in scalar_names):
            raise TypedBackboneCoreError("Table identity fields must be nonempty strings")
        if type(value["expected_rows"]) is not int or value["expected_rows"] < 0:
            raise TypedBackboneCoreError("expected_rows must be a nonnegative integer")
        columns = value["columns"]
        if not isinstance(columns, list):
            raise TypedBackboneCoreError("columns must be a list")
        fields = field_specs(columns)
        field_names = {field.name for field in fields}

        def keys(label: str) -> tuple[str, ...]:
            raw = value[label]
            if (
                not isinstance(raw, list)
                or not raw
                or any(not isinstance(item, str) or not item for item in raw)
                or len(raw) != len(set(raw))
            ):
                raise TypedBackboneCoreError(
                    f"{label} must be a nonempty unique string list"
                )
            unknown = set(raw) - field_names
            if unknown:
                raise TypedBackboneCoreError(
                    f"{label} references unknown fields {sorted(unknown)}"
                )
            return tuple(raw)

        primary_key = keys("primary_key")
        sort_key = keys("sort_key")
        if sort_key[: len(primary_key)] != primary_key:
            raise TypedBackboneCoreError(
                "sort_key must begin with the complete primary_key"
            )
        field_by_name = {field.name: field for field in fields}
        if any(field_by_name[name].nullable for name in primary_key):
            raise TypedBackboneCoreError("Primary-key fields must be nonnullable")
        metadata = {
            key: item
            for key, item in value.items()
            if key not in required
        }
        return cls(
            name=value["table"],
            role=value["role"],
            primary_source=value["primary_source"],
            grain=value["grain"],
            primary_key=primary_key,
            sort_key=sort_key,
            expected_rows=value["expected_rows"],
            fields=fields,
            metadata=metadata,
        )

    def table_metadata(self) -> dict[str, Any]:
        return {
            "expected_rows": self.expected_rows,
            "grain": self.grain,
            "primary_key": list(self.primary_key),
            "primary_source": self.primary_source,
            "role": self.role,
            "sort_key": list(self.sort_key),
            "table": self.name,
            **dict(self.metadata),
        }

    def arrow_schema(self) -> pa.Schema:
        return arrow_schema(self.fields, table_metadata=self.table_metadata())

    def fingerprint(self) -> str:
        return schema_fingerprint(
            self.fields, table_metadata=self.table_metadata()
        )


@dataclass(frozen=True)
class SchemaRegistry:
    schema_version: int
    schema_id: str
    status: str
    tables: Mapping[str, TableSpec]
    metadata: Mapping[str, Any]
    source_sha256: str | None = None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, source_sha256: str | None = None
    ) -> "SchemaRegistry":
        required = {"schema_version", "schema_id", "status", "tables"}
        missing = required - set(value)
        if missing:
            raise TypedBackboneCoreError(
                f"Schema registry is missing {sorted(missing)}"
            )
        if type(value["schema_version"]) is not int or value["schema_version"] <= 0:
            raise TypedBackboneCoreError("schema_version must be a positive integer")
        if any(
            not isinstance(value[name], str) or not value[name]
            for name in ("schema_id", "status")
        ):
            raise TypedBackboneCoreError("schema_id and status must be nonempty strings")
        raw_tables = value["tables"]
        if not isinstance(raw_tables, list) or not raw_tables:
            raise TypedBackboneCoreError("tables must be a nonempty list")
        parsed: dict[str, TableSpec] = {}
        for raw in raw_tables:
            if not isinstance(raw, Mapping):
                raise TypedBackboneCoreError("Every table must be an object")
            table = TableSpec.from_mapping(raw)
            if table.name in parsed:
                raise TypedBackboneCoreError(f"Duplicate table {table.name!r}")
            parsed[table.name] = table
        metadata = {
            key: item for key, item in value.items() if key not in required
        }
        vocabulary = metadata.get("feature_use_vocabulary")
        if isinstance(vocabulary, Mapping):
            allowed = set(vocabulary)
            for table in parsed.values():
                for field in table.fields:
                    feature_use = field.metadata.get("feature_use")
                    if feature_use not in allowed:
                        raise TypedBackboneCoreError(
                            f"Unknown feature_use {feature_use!r} for "
                            f"{table.name}.{field.name}"
                        )
        return cls(
            schema_version=value["schema_version"],
            schema_id=value["schema_id"],
            status=value["status"],
            tables=parsed,
            metadata=metadata,
            source_sha256=source_sha256,
        )


def load_schema_registry(path: Path) -> SchemaRegistry:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise TypedBackboneCoreError(f"Cannot read schema registry {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypedBackboneCoreError("Schema registry must be a JSON object")
    return SchemaRegistry.from_mapping(
        value, source_sha256=hashlib.sha256(payload).hexdigest()
    )


def field_specs(values: Sequence[Mapping[str, Any]]) -> tuple[FieldSpec, ...]:
    parsed = tuple(FieldSpec.from_mapping(value) for value in values)
    names = [field.name for field in parsed]
    if len(names) != len(set(names)):
        raise TypedBackboneCoreError("Field names must be unique")
    if not names:
        raise TypedBackboneCoreError("A typed table must contain at least one field")
    return parsed


def arrow_schema(
    fields: Sequence[FieldSpec], *, table_metadata: Mapping[str, Any] | None = None
) -> pa.Schema:
    if not fields:
        raise TypedBackboneCoreError("A typed table must contain at least one field")
    return pa.schema(
        [field.arrow_field() for field in fields],
        metadata=_metadata_bytes(table_metadata or {}),
    )


def schema_fingerprint(
    fields: Sequence[FieldSpec], *, table_metadata: Mapping[str, Any] | None = None
) -> str:
    """Hash logical schema semantics independently of Arrow's binary metadata."""

    payload = {
        "fields": [field.semantic_mapping() for field in fields],
        "table_metadata": dict(table_metadata or {}),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _decimal_unscaled(value: Decimal, data_type: pa.Decimal128Type) -> int:
    if not isinstance(value, Decimal):
        raise TypedBackboneCoreError("decimal128 values must be Decimal instances")
    if not value.is_finite():
        raise TypedBackboneCoreError("Nonfinite decimal values are prohibited")
    parts = value.as_tuple()
    coefficient = 0
    for digit in parts.digits:
        coefficient = coefficient * 10 + digit
    if parts.sign:
        coefficient = -coefficient
    shift = parts.exponent + data_type.scale
    if shift >= 0:
        integer = coefficient * (10**shift)
    else:
        divisor = 10 ** (-shift)
        if coefficient % divisor:
            raise TypedBackboneCoreError(
                f"Decimal {value!s} exceeds locked scale {data_type.scale}"
            )
        integer = coefficient // divisor
    if abs(integer) >= 10 ** data_type.precision:
        raise TypedBackboneCoreError(
            f"Decimal {value!s} exceeds locked precision {data_type.precision}"
        )
    return integer


def _date32_days(value: dt.date) -> int:
    if not isinstance(value, dt.date) or isinstance(value, dt.datetime):
        raise TypedBackboneCoreError("date32 values must be datetime.date instances")
    return (value - dt.date(1970, 1, 1)).days


def encode_scalar(value: Any, data_type: pa.DataType, *, nullable: bool) -> bytes:
    """Encode one scalar in the independent, domain-separated logical format."""

    if value is None:
        if not nullable:
            raise TypedBackboneCoreError("Null supplied for a nonnullable field")
        return b"\x00"
    prefix = b"\x01"
    if pa.types.is_boolean(data_type):
        if type(value) is not bool:
            raise TypedBackboneCoreError("Boolean field requires bool, not integer")
        return prefix + (b"\x01" if value else b"\x00")
    if pa.types.is_int8(data_type):
        if type(value) is not int or not -(1 << 7) <= value < (1 << 7):
            raise TypedBackboneCoreError("Value is outside signed int8")
        return prefix + struct.pack(">b", value)
    if pa.types.is_int16(data_type):
        if type(value) is not int or not -(1 << 15) <= value < (1 << 15):
            raise TypedBackboneCoreError("Value is outside signed int16")
        return prefix + struct.pack(">h", value)
    if pa.types.is_int32(data_type):
        if type(value) is not int or not -(1 << 31) <= value < (1 << 31):
            raise TypedBackboneCoreError("Value is outside signed int32")
        return prefix + struct.pack(">i", value)
    if pa.types.is_int64(data_type):
        if type(value) is not int or not -(1 << 63) <= value < (1 << 63):
            raise TypedBackboneCoreError("Value is outside signed int64")
        return prefix + struct.pack(">q", value)
    if pa.types.is_date32(data_type):
        days = _date32_days(value)
        if not -(1 << 31) <= days < (1 << 31):
            raise TypedBackboneCoreError("Date is outside date32 range")
        return prefix + struct.pack(">i", days)
    if pa.types.is_string(data_type):
        if not isinstance(value, str):
            raise TypedBackboneCoreError("UTF-8 field requires str")
        payload = value.encode("utf-8")
        return prefix + struct.pack(">Q", len(payload)) + payload
    if pa.types.is_decimal128(data_type):
        integer = _decimal_unscaled(value, data_type)
        try:
            payload = integer.to_bytes(16, "big", signed=True)
        except OverflowError as exc:
            raise TypedBackboneCoreError("Decimal exceeds signed 128-bit range") from exc
        return prefix + payload
    raise TypedBackboneCoreError(f"Unsupported Arrow type {data_type}")


class LogicalRowDigest:
    """Streaming logical digest over one table shard in exact row order."""

    def __init__(
        self,
        *,
        table_name: str,
        partition_name: str,
        fields: Sequence[FieldSpec],
        table_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not table_name or not partition_name:
            raise TypedBackboneCoreError("Table and partition names must be nonempty")
        self._fields = tuple(fields)
        if not self._fields:
            raise TypedBackboneCoreError("Logical digest requires a schema")
        self._types = tuple(parse_arrow_type(field.logical_type) for field in fields)
        self._hash = hashlib.sha256()
        self._hash.update(LOGICAL_HASH_DOMAIN)
        for label in (table_name, partition_name):
            payload = label.encode("utf-8")
            self._hash.update(struct.pack(">Q", len(payload)))
            self._hash.update(payload)
        fingerprint = schema_fingerprint(fields, table_metadata=table_metadata)
        self._hash.update(bytes.fromhex(fingerprint))
        self._row_count = 0

    @property
    def row_count(self) -> int:
        return self._row_count

    def update_row(self, values: Sequence[Any] | Mapping[str, Any]) -> None:
        if isinstance(values, Mapping):
            if set(values) != {field.name for field in self._fields}:
                raise TypedBackboneCoreError("Row mapping fields differ from schema")
            ordered = [values[field.name] for field in self._fields]
        else:
            ordered = list(values)
        if len(ordered) != len(self._fields):
            raise TypedBackboneCoreError("Row width differs from schema")
        self._hash.update(b"\x52")
        for value, field, data_type in zip(
            ordered, self._fields, self._types, strict=True
        ):
            self._hash.update(
                encode_scalar(value, data_type, nullable=field.nullable)
            )
        self._row_count += 1

    def update_rows(
        self, rows: Iterable[Sequence[Any] | Mapping[str, Any]]
    ) -> None:
        for row in rows:
            self.update_row(row)

    def hexdigest(self) -> str:
        final = self._hash.copy()
        final.update(b"\x45")
        final.update(struct.pack(">Q", self._row_count))
        return final.hexdigest()
