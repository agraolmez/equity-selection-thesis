"""Shared exact, schema-driven primitives for Stage 2 auxiliary staging."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa


class AuxiliaryBuildError(RuntimeError):
    """Raised when an auxiliary-source contract cannot be honored exactly."""


_DECIMAL_RE = re.compile(r"decimal128\((\d+),(\d+)\)\Z")
_COLUMN_RE = re.compile(r"([^:]+):([^!?]+(?:\([^)]*\))?)([!?])\Z")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def arrow_type(name: str) -> pa.DataType:
    fixed = {
        "utf8": pa.string(),
        "int64": pa.int64(),
        "int32": pa.int32(),
        "int16": pa.int16(),
        "bool": pa.bool_(),
        "date32": pa.date32(),
        "timestamp_us_utc": pa.timestamp("us", tz="UTC"),
    }
    if name in fixed:
        return fixed[name]
    match = _DECIMAL_RE.fullmatch(name)
    if match:
        return pa.decimal128(int(match.group(1)), int(match.group(2)))
    raise AuxiliaryBuildError(f"Unsupported auxiliary Arrow type {name!r}")


def load_schema_document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise AuxiliaryBuildError("Auxiliary schema must be a JSON object")
    return value


def table_contract(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    matches = [item for item in document["tables"] if item["table"] == name]
    if len(matches) != 1:
        raise AuxiliaryBuildError(f"Auxiliary table contract is not unique: {name}")
    return matches[0]


def table_schema(document: Mapping[str, Any], name: str) -> pa.Schema:
    contract = table_contract(document, name)
    fields: list[pa.Field] = []
    for encoded in contract["columns"]:
        match = _COLUMN_RE.fullmatch(encoded)
        if not match:
            raise AuxiliaryBuildError(f"Malformed compact column contract {encoded!r}")
        column_name, logical_type, suffix = match.groups()
        fields.append(pa.field(column_name, arrow_type(logical_type), nullable=suffix == "?"))
    for payload in contract.get("source_payload_columns", []):
        if name == "compustat_annual_observation" and payload == "fyear":
            logical = pa.int16()
        elif name == "compustat_quarterly_observation" and payload in {"fyearq", "fqtr", "fyr"}:
            logical = pa.int16()
        elif name == "compustat_quarterly_observation" and payload == "datafqtr":
            logical = pa.string()
        else:
            logical = pa.decimal128(38, 12)
        fields.append(pa.field(payload, logical, nullable=True))
    metadata = {
        key.encode(): (
            json.dumps(contract[key], sort_keys=True, separators=(",", ":"))
            if isinstance(contract[key], (list, dict))
            else str(contract[key])
        ).encode()
        for key in ("table", "source", "grain", "primary_key", "sort_key")
        if key in contract
    }
    return pa.schema(fields, metadata=metadata)


def normalize_cusip8(value: object) -> str | None:
    text = "" if value is None else str(value).upper()
    normalized = "".join(character for character in text if character.isascii() and character.isalnum())
    return normalized[:8] if len(normalized) >= 8 else None


def optional_text(value: object) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def required_text(value: object, label: str) -> str:
    text = optional_text(value)
    if text is None:
        raise AuxiliaryBuildError(f"Blank required {label}")
    return text


def optional_int(value: object, label: str) -> int | None:
    text = optional_text(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise AuxiliaryBuildError(f"Invalid integer {label}: {text!r}") from exc


def required_int(value: object, label: str) -> int:
    result = optional_int(value, label)
    if result is None:
        raise AuxiliaryBuildError(f"Blank required integer {label}")
    return result


def optional_date(value: object, label: str) -> dt.date | None:
    text = optional_text(value)
    if text is None or text == "E":
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError as exc:
        raise AuxiliaryBuildError(f"Invalid date {label}: {text!r}") from exc


def required_date(value: object, label: str) -> dt.date:
    result = optional_date(value, label)
    if result is None:
        raise AuxiliaryBuildError(f"Blank required date {label}")
    return result


def optional_decimal(value: object, *, scale: int, label: str) -> Decimal | None:
    text = optional_text(value)
    if text is None:
        return None
    try:
        result = Decimal(text)
    except Exception as exc:
        raise AuxiliaryBuildError(f"Invalid decimal {label}: {text!r}") from exc
    if not result.is_finite():
        raise AuxiliaryBuildError(f"Non-finite decimal {label}: {text!r}")
    exponent = -result.as_tuple().exponent
    if exponent > scale:
        raise AuxiliaryBuildError(
            f"Decimal {label} has {exponent} places but contract permits {scale}"
        )
    return result.quantize(Decimal(1).scaleb(-scale))


def source_record_id(filename: str, row_ordinal: int, values: Iterable[object]) -> str:
    body = [filename, row_ordinal, *("" if value is None else str(value) for value in values)]
    return hashlib.sha256(canonical_json(body)).hexdigest()


class TradingCalendar:
    """Exact date advancement on the verified observed CRSP calendar."""

    def __init__(self, dates: Iterable[dt.date]) -> None:
        values = np.array(
            [(value - dt.date(1970, 1, 1)).days for value in dates], dtype=np.int64
        )
        if len(values) == 0 or np.any(values[1:] <= values[:-1]):
            raise AuxiliaryBuildError("Verified calendar must be nonempty and strictly sorted")
        self.ordinals = values

    @staticmethod
    def _ordinal(value: dt.date) -> int:
        return (value - dt.date(1970, 1, 1)).days

    @staticmethod
    def _date(value: int) -> dt.date:
        return dt.date(1970, 1, 1) + dt.timedelta(days=int(value))

    def on_or_after(self, value: dt.date) -> dt.date:
        position = int(np.searchsorted(self.ordinals, self._ordinal(value), side="left"))
        if position >= len(self.ordinals):
            raise AuxiliaryBuildError(f"Calendar lacks on-or-after context for {value}")
        return self._date(int(self.ordinals[position]))

    def strictly_after(self, value: dt.date) -> dt.date:
        position = int(np.searchsorted(self.ordinals, self._ordinal(value), side="right"))
        if position >= len(self.ordinals):
            raise AuxiliaryBuildError(f"Calendar lacks strictly-after context for {value}")
        return self._date(int(self.ordinals[position]))


def table_from_rows(
    document: Mapping[str, Any], name: str, rows: list[Mapping[str, Any]]
) -> pa.Table:
    schema = table_schema(document, name)
    return pa.Table.from_pylist(rows, schema=schema)
