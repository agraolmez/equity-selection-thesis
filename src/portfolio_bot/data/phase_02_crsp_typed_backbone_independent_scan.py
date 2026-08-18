"""Independent streaming scanner for the Phase 2 typed CRSP backbone.

The module intentionally does not import the primary typed-backbone builder,
transform library, schema utility, or publisher.  It reconstructs the frozen
Arrow wire schemas from JSON, rereads the four authenticated raw CSV sources,
replays canonicalization and typing, and compares every resulting value with
the six Parquet tables.  Only aggregate evidence leaves the scanner.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import re
import stat
import tomllib
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/crsp_typed_backbone_policy_v1.toml"
SCHEMA_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/crsp_typed_backbone_schema_v1.json"
IDENTITY_LOCK_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/crsp_typed_backbone_identity_empirical_lock_v1.json"
DEVELOPMENT_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/crsp_development_v1.toml"
SUPPORTING_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/crsp_supporting_sources_development_v1.toml"
PATHS_PATH = PROJECT_ROOT / "configs/paths.toml"

LOCKED_POLICY_SHA256 = "23ec145f587c95b735f675c5ee665d43d506ad2d91d0ae6a46a3942ed5674a89"
LOCKED_SCHEMA_SHA256 = "62ad603c88798212f60ea4ec9796bffbf65008c6cfd87a08239cddf057e31391"
LOCKED_IDENTITY_SHA256 = "f369b9eccf3dbfb9741fcbaa0e0a6a0376d3585c0acb04c2443c94d090a33406"
LOCKED_DEVELOPMENT_SHA256 = "645b6ff51480d4e3cbcad313e3488c5a8d45df272ca4af6e7010e870296b4c35"
LOCKED_SUPPORTING_SHA256 = "6117495f76106a9a89922c1d2a01d23c824e9e94f783ffd8a8290a99bec11cf1"
LOCKED_PATHS_SHA256 = "a41135d4b12e3138b65027b5d522e5cb6066ce50e742f878c474ac120a075878"

ROW_GROUP_SIZE = 262_144
DEFAULT_BLOCK_SIZE = 16 << 20
TABLE_FILES = {
    "crsp_identity_interval": "crsp_identity_interval.parquet",
    "crsp_security_day": "crsp_security_day.parquet",
    "crsp_security_day_return_staging": "crsp_security_day_return_staging.parquet",
    "crsp_security_month_crosscheck": "crsp_security_month_crosscheck.parquet",
    "crsp_security_month_return_staging": "crsp_security_month_return_staging.parquet",
    "crsp_shares_interval": "crsp_shares_interval.parquet",
}
IDENTITY_PAYLOAD_FIELDS = (
    "PERMCO", "CUSIP", "PrimaryExch", "TradingStatusFlg", "USIncFlg",
    "IssuerType", "SecurityType", "SecuritySubType", "ShareType", "Ticker",
    "SICCD", "NAICS", "IssuerNm",
)
_DURATION = {
    "D1": 1, "D2": 1, "D3": 1, "D4": 1, "DU": 1,
    "P1": 2, "P2": 3, "P3": 4, "P4": 5, "P5": 6,
    "P6": 7, "P7": 8, "P8": 9, "P9": 10, "MR": 0, "DD": -1,
}
_DECIMAL_RE = re.compile(r"decimal128\(([1-9][0-9]*),([0-9]+)\)\Z")
_EPOCH = np.datetime64("1970-01-01", "D")


class IndependentTypedScanError(RuntimeError):
    """Raised when independently observed evidence violates the frozen lock."""


@dataclasses.dataclass(frozen=True)
class TableContract:
    name: str
    expected_rows: int
    primary_key: tuple[str, ...]
    sort_key: tuple[str, ...]
    fields: tuple[Mapping[str, Any], ...]
    raw: Mapping[str, Any]

    def arrow_schema(self) -> pa.Schema:
        fields = []
        for item in self.fields:
            metadata = {
                str(key).encode(): _metadata_value(value)
                for key, value in sorted(item.items())
                if key not in {"name", "type", "nullable"}
            }
            fields.append(
                pa.field(
                    str(item["name"]),
                    _arrow_type(str(item["type"])),
                    nullable=bool(item["nullable"]),
                    metadata=metadata,
                )
            )
        required = {
            "table", "role", "primary_source", "grain", "primary_key",
            "sort_key", "expected_rows", "columns",
        }
        table_metadata = {
            "table": self.name,
            "expected_rows": self.expected_rows,
            "primary_key": list(self.primary_key),
            "sort_key": list(self.sort_key),
            **{key: value for key, value in self.raw.items() if key not in required},
            "role": self.raw["role"],
            "primary_source": self.raw["primary_source"],
            "grain": self.raw["grain"],
        }
        return pa.schema(
            fields,
            metadata={
                str(key).encode(): _metadata_value(value)
                for key, value in sorted(table_metadata.items())
            },
        )

    def fingerprint(self) -> str:
        required = {
            "table", "role", "primary_source", "grain", "primary_key",
            "sort_key", "expected_rows", "columns",
        }
        table_metadata = {
            "expected_rows": self.expected_rows,
            "grain": self.raw["grain"],
            "primary_key": list(self.primary_key),
            "primary_source": self.raw["primary_source"],
            "role": self.raw["role"],
            "sort_key": list(self.sort_key),
            "table": self.name,
            **{key: value for key, value in self.raw.items() if key not in required},
        }
        fields = []
        for item in self.fields:
            fields.append(
                {
                    "logical_type": item["type"],
                    "name": item["name"],
                    "nullable": item["nullable"],
                    **{
                        key: value
                        for key, value in item.items()
                        if key not in {"name", "type", "nullable"}
                    },
                }
            )
        return hashlib.sha256(
            _canonical_json({"fields": fields, "table_metadata": table_metadata})
        ).hexdigest()


@dataclasses.dataclass(frozen=True)
class IndependentSettings:
    project_root: Path
    database_root: Path
    raw_root: Path
    artifact_root: Path
    policy: Mapping[str, Any]
    schema_document: Mapping[str, Any]
    identity_lock: Mapping[str, Any]
    contracts: Mapping[str, TableContract]
    source_headers: Mapping[str, tuple[str, ...]]
    block_size: int


@dataclasses.dataclass
class ScanEvidence:
    raw_sources: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    canonicalization: dict[str, dict[str, int]] = dataclasses.field(default_factory=dict)
    table_rows: Counter[str] = dataclasses.field(default_factory=Counter)
    table_nulls: dict[str, Counter[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(Counter)
    )
    key_sha256: dict[str, str] = dataclasses.field(default_factory=dict)
    parquet: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    joins: dict[str, dict[str, int]] = dataclasses.field(default_factory=dict)
    compared_values: int = 0
    key_quality: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    raw_expected_nulls: dict[str, Counter[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(Counter)
    )
    decimal_roundtrip: dict[str, dict[str, Counter[str]]] = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Counter))
    )
    formula_quality: dict[str, Counter[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(Counter)
    )
    return_split: dict[str, Counter[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(Counter)
    )
    quarantine_guards: dict[str, Counter[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(Counter)
    )
    consumer_return_absence: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    date_scope: dict[str, Counter[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(Counter)
    )


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _metadata_value(value: Any) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    return _canonical_json(value)[:-1]


def _arrow_type(value: str) -> pa.DataType:
    fixed = {
        "bool": pa.bool_(), "date32": pa.date32(), "int16": pa.int16(),
        "int32": pa.int32(), "int64": pa.int64(), "utf8": pa.string(),
    }
    if value in fixed:
        return fixed[value]
    match = _DECIMAL_RE.fullmatch(value)
    if match is None:
        raise IndependentTypedScanError(f"Unsupported locked Arrow type {value!r}")
    precision, scale = (int(part) for part in match.groups())
    if precision > 38 or scale > precision:
        raise IndependentTypedScanError(f"Invalid decimal128 contract {value!r}")
    return pa.decimal128(precision, scale)


def _read_locked(path: Path, digest: str, kind: str) -> Any:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise IndependentTypedScanError(f"Cannot read locked {kind}") from exc
    if hashlib.sha256(payload).hexdigest() != digest:
        raise IndependentTypedScanError(f"Locked {kind} SHA-256 differs")
    try:
        if path.suffix == ".toml":
            return tomllib.loads(payload.decode("utf-8"))
        return json.loads(payload)
    except (UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise IndependentTypedScanError(f"Cannot parse locked {kind}") from exc


def load_independent_settings(*, block_size: int = DEFAULT_BLOCK_SIZE) -> IndependentSettings:
    """Authenticate all verifier authorities without importing primary code."""

    if block_size < 4096:
        raise IndependentTypedScanError("Arrow CSV block size is below 4096 bytes")
    policy = _read_locked(POLICY_PATH, LOCKED_POLICY_SHA256, "typed policy")
    schema = _read_locked(SCHEMA_PATH, LOCKED_SCHEMA_SHA256, "typed schema")
    identity = _read_locked(IDENTITY_LOCK_PATH, LOCKED_IDENTITY_SHA256, "identity lock")
    development = _read_locked(DEVELOPMENT_PATH, LOCKED_DEVELOPMENT_SHA256, "development profile")
    supporting = _read_locked(SUPPORTING_PATH, LOCKED_SUPPORTING_SHA256, "supporting profile")
    paths = _read_locked(PATHS_PATH, LOCKED_PATHS_SHA256, "paths config")
    if policy["schema"]["sha256"] != LOCKED_SCHEMA_SHA256:
        raise IndependentTypedScanError("Policy/schema binding differs")
    if policy["upstream"]["identity_empirical_lock"]["sha256"] != LOCKED_IDENTITY_SHA256:
        raise IndependentTypedScanError("Policy/identity-lock binding differs")
    if identity.get("status") != "locked_after_aggregate_only_development_discovery":
        raise IndependentTypedScanError("Identity empirical lock status differs")
    contracts: dict[str, TableContract] = {}
    for raw in schema["tables"]:
        name = str(raw["table"])
        contracts[name] = TableContract(
            name=name,
            expected_rows=int(raw["expected_rows"]),
            primary_key=tuple(raw["primary_key"]),
            sort_key=tuple(raw["sort_key"]),
            fields=tuple(raw["columns"]),
            raw=raw,
        )
    if set(contracts) != set(TABLE_FILES):
        raise IndependentTypedScanError("Typed schema table set differs")
    supporting_sources = {item["filename"]: item for item in supporting["sources"]}
    headers = {
        "daily": tuple(development["sources"]["daily"]["header"]),
        "monthly": tuple(development["sources"]["monthly"]["header"]),
        "identity": tuple(supporting_sources["crsp_names_ciz.csv"]["header"]),
        "shares": tuple(supporting_sources["crsp_shares_outstanding_ciz.csv"]["header"]),
    }
    database_root = Path(paths["database"]["root"])
    return IndependentSettings(
        project_root=PROJECT_ROOT,
        database_root=database_root,
        raw_root=database_root / paths["database"]["raw"],
        artifact_root=PROJECT_ROOT / paths["project"]["artifacts"] / "phase_02_canonical_panel",
        policy=policy,
        schema_document=schema,
        identity_lock=identity,
        contracts=contracts,
        source_headers=headers,
        block_size=block_size,
    )


def _assert_no_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise IndependentTypedScanError(
                f"Cannot authenticate raw path component for {path.name}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise IndependentTypedScanError(
                f"Symlink is prohibited in raw path for {path.name}"
            )


def _stable_hash(
    path: Path, *, expected_size: int, expected_sha256: str
) -> tuple[dict[str, Any], int]:
    """Hash and retain one no-follow raw descriptor for the subsequent scan."""

    _assert_no_symlink_components(path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise IndependentTypedScanError(f"Cannot stat raw source {path.name}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise IndependentTypedScanError(f"Raw source is not a single-link regular file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise IndependentTypedScanError(f"Raw source identity changed: {path.name}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 << 20):
            digest.update(chunk)
        after_fd = os.fstat(descriptor)
    except BaseException as exc:
        if descriptor is not None:
            os.close(descriptor)
        if isinstance(exc, IndependentTypedScanError):
            raise
        raise IndependentTypedScanError(f"Cannot hash raw source {path.name}") from exc
    try:
        assert descriptor is not None
        _assert_no_symlink_components(path)
        after = path.lstat()
        signature = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_mode)
        if signature(before) != signature(opened) or signature(opened) != signature(after_fd) or signature(after_fd) != signature(after):
            raise IndependentTypedScanError(f"Raw source changed while hashing: {path.name}")
        actual = digest.hexdigest()
        if before.st_size != expected_size or actual != expected_sha256:
            raise IndependentTypedScanError(f"Raw source authentication differs: {path.name}")
        os.lseek(descriptor, 0, os.SEEK_SET)
    except BaseException as exc:
        os.close(descriptor)
        if isinstance(exc, IndependentTypedScanError):
            raise
        raise IndependentTypedScanError(
            f"Cannot rewind authenticated raw source {path.name}"
        ) from exc
    return {
        "sha256": actual,
        "size_bytes": int(before.st_size),
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "mtime_ns": int(before.st_mtime_ns),
        "mode": int(before.st_mode),
    }, descriptor


class _DigestingRawReader(io.RawIOBase):
    """Sequential reader whose digest covers the exact bytes supplied to Arrow."""

    def __init__(self, stream: io.RawIOBase) -> None:
        super().__init__()
        self._stream = stream
        self._digest = hashlib.sha256()
        self.bytes_read = 0

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        payload = self._stream.read(size)
        if payload:
            self._digest.update(payload)
            self.bytes_read += len(payload)
        return payload

    def readinto(self, buffer: Any) -> int:
        count = self._stream.readinto(buffer)
        if count:
            self._digest.update(memoryview(buffer)[:count])
            self.bytes_read += count
        return count

    def close(self) -> None:
        if not self.closed:
            self._stream.close()
        super().close()


def _recheck_stat(path: Path, receipt: Mapping[str, Any], descriptor: int) -> None:
    _assert_no_symlink_components(path)
    current = path.lstat()
    held = os.fstat(descriptor)
    observed = {
        "size_bytes": current.st_size, "device": current.st_dev,
        "inode": current.st_ino, "mtime_ns": current.st_mtime_ns, "mode": current.st_mode,
    }
    held_values = {
        "size_bytes": held.st_size, "device": held.st_dev,
        "inode": held.st_ino, "mtime_ns": held.st_mtime_ns, "mode": held.st_mode,
    }
    for key, actual in observed.items():
        if int(receipt[key]) != int(actual):
            raise IndependentTypedScanError(f"Raw source changed after scan: {path.name}")
        if int(receipt[key]) != int(held_values[key]):
            raise IndependentTypedScanError(
                f"Held raw source changed during scan: {path.name}"
            )


def _open_csv(
    source: Any, label: str, header: Sequence[str], block_size: int
) -> pacsv.CSVStreamingReader:
    try:
        reader = pacsv.open_csv(
            source,
            read_options=pacsv.ReadOptions(block_size=block_size, use_threads=True, encoding="utf8"),
            parse_options=pacsv.ParseOptions(
                delimiter=",", quote_char='"', double_quote=True, escape_char=False,
                newlines_in_values=True, ignore_empty_lines=False,
            ),
            convert_options=pacsv.ConvertOptions(
                column_types={name: pa.string() for name in header},
                strings_can_be_null=False, quoted_strings_can_be_null=False,
                null_values=[], check_utf8=True,
            ),
        )
    except (OSError, pa.ArrowException) as exc:
        raise IndependentTypedScanError(f"Cannot open raw CSV {label}") from exc
    if tuple(reader.schema.names) != tuple(header) or any(not pa.types.is_string(field.type) for field in reader.schema):
        raise IndependentTypedScanError(f"Raw CSV header/schema differs: {label}")
    return reader


def _batches(reader: pacsv.CSVStreamingReader, label: str) -> Iterator[pa.RecordBatch]:
    while True:
        try:
            yield reader.read_next_batch()
        except StopIteration:
            return
        except (OSError, pa.ArrowException) as exc:
            raise IndependentTypedScanError(f"Raw CSV scan failed: {label}") from exc


def _column(batch: pa.RecordBatch | pa.Table, name: str) -> pa.Array | pa.ChunkedArray:
    return batch[name] if isinstance(batch, pa.Table) else batch.column(batch.schema.get_field_index(name))


def _require(mask: pa.Array | pa.ChunkedArray, message: str) -> None:
    value = pc.all(pc.fill_null(mask, False), skip_nulls=False)
    if not value.is_valid or value.as_py() is not True:
        raise IndependentTypedScanError(message)


def _date(raw: pa.Array | pa.ChunkedArray, label: str, *, optional: bool = False) -> pa.Array | pa.ChunkedArray:
    blank = pc.equal(raw, "")
    accepted = pc.match_substring_regex(raw, r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    if optional:
        accepted = pc.or_(blank, accepted)
    _require(accepted, f"{label} is not strict YYYY-MM-DD")
    prepared = pc.if_else(blank, None, raw) if optional else raw
    try:
        parsed = pc.cast(prepared, pa.date32(), safe=True)
    except pa.ArrowException as exc:
        raise IndependentTypedScanError(f"{label} is not Gregorian") from exc
    roundtrip = pc.strftime(pc.cast(parsed, pa.timestamp("s")), format="%Y-%m-%d")
    roundtrip_equal = pc.equal(roundtrip, raw)
    _require(
        pc.if_else(blank, True, roundtrip_equal)
        if optional
        else roundtrip_equal,
        f"{label} date roundtrip differs",
    )
    return parsed


def _integer(raw: pa.Array | pa.ChunkedArray, label: str, *, optional: bool = False, positive: bool = True, target: pa.DataType = pa.int64()) -> pa.Array | pa.ChunkedArray:
    blank = pc.equal(raw, "")
    grammar = pc.match_substring_regex(
        raw, r"^[1-9][0-9]*$" if positive else r"^[0-9]+$"
    )
    _require(pc.or_(blank, grammar) if optional else grammar, f"{label} integer lexeme differs")
    try:
        return pc.cast(pc.if_else(blank, None, raw) if optional else raw, target, safe=True)
    except pa.ArrowException as exc:
        raise IndependentTypedScanError(f"{label} integer range differs") from exc


def _decimal(raw: pa.Array | pa.ChunkedArray, target: pa.Decimal128Type, label: str) -> pa.Array | pa.ChunkedArray:
    blank = pc.equal(raw, "")
    grammar = pc.match_substring_regex(raw, r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?$")
    _require(pc.or_(blank, grammar), f"{label} decimal lexeme differs")
    coefficient = pc.replace_substring_regex(
        pc.replace_substring_regex(
            pc.replace_substring_regex(raw, pattern=r"[Ee].*$", replacement=""),
            pattern=r"^[+-]", replacement="",
        ),
        pattern=r"\.", replacement="",
    )
    significant = pc.replace_substring_regex(
        coefficient, pattern=r"^0+", replacement=""
    )
    unsafe_arrow_coefficient = pc.and_(
        pc.invert(blank),
        pc.or_(
            pc.match_substring_regex(raw, r"[Ee]"),
            pc.greater(
                pc.add_checked(
                    pc.utf8_length(significant),
                    pa.scalar(target.scale, pa.int32()),
                ),
                pa.scalar(target.precision, pa.int32()),
            ),
        ),
    )
    prepared = pc.if_else(
        blank, None, pc.if_else(unsafe_arrow_coefficient, "0", raw)
    )
    try:
        ordinary = pc.cast(
            prepared,
            options=pc.CastOptions(target_type=target, allow_decimal_truncate=False),
        )
    except pa.ArrowException as exc:
        raise IndependentTypedScanError(f"{label} is not exactly representable as {target}") from exc
    positions = pc.indices_nonzero(unsafe_arrow_coefficient).to_pylist()
    if not positions:
        return ordinary
    replacements: list[Decimal | None] = [None] * len(raw)
    for position in positions:
        replacements[position] = _arbitrary_precision_decimal(
            raw[position].as_py(), target, label
        )
    return pc.if_else(
        unsafe_arrow_coefficient, pa.array(replacements, type=target), ordinary
    )


def _arbitrary_precision_decimal(
    lexeme: str, target: pa.Decimal128Type, label: str
) -> Decimal:
    """Prove representability without feeding a wide coefficient to Arrow."""

    try:
        value = Decimal(lexeme)
    except (InvalidOperation, ValueError) as exc:
        raise IndependentTypedScanError(f"{label} decimal lexeme differs") from exc
    if not value.is_finite():
        raise IndependentTypedScanError(
            f"{label} is not exactly representable as {target}"
        )
    sign, digits_tuple, exponent = value.as_tuple()
    digits = "".join(str(digit) for digit in digits_tuple).lstrip("0")
    if not digits:
        return Decimal((sign, (0,), -target.scale))
    shift = int(exponent) + target.scale
    if shift >= 0:
        if len(digits) + shift > target.precision:
            raise IndependentTypedScanError(
                f"{label} is not exactly representable as {target}"
            )
        unscaled = digits + ("0" * shift)
    else:
        removed = -shift
        if removed > len(digits) or any(
            character != "0" for character in digits[-removed:]
        ):
            raise IndependentTypedScanError(
                f"{label} is not exactly representable as {target}"
            )
        unscaled = digits[:-removed] or "0"
    significant_unscaled = unscaled.lstrip("0") or "0"
    if len(significant_unscaled) > target.precision:
        raise IndependentTypedScanError(
            f"{label} is not exactly representable as {target}"
        )
    return Decimal(
        (sign, tuple(int(character) for character in significant_unscaled), -target.scale)
    )


def _combine(batches: Sequence[pa.RecordBatch]) -> pa.RecordBatch:
    table = pa.Table.from_batches(list(batches)).combine_chunks()
    return table.to_batches(max_chunksize=max(1, table.num_rows))[0]


def _key_arrays(batch: pa.RecordBatch, columns: Sequence[str]) -> tuple[np.ndarray, ...]:
    values = []
    for name in columns:
        raw = _column(batch, name)
        if name == "PERMNO":
            array = _integer(raw, "PERMNO")
            values.append(np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.int64))
        else:
            array = _date(raw, name)
            values.append(np.asarray(array.to_numpy(zero_copy_only=False)).astype("datetime64[D]").astype(np.int32))
    return tuple(values)


def _assert_sorted(arrays: Sequence[np.ndarray], label: str) -> None:
    if not arrays or len(arrays[0]) < 2:
        return
    undecided = np.ones(len(arrays[0]) - 1, dtype=bool)
    violation = np.zeros(len(arrays[0]) - 1, dtype=bool)
    for array in arrays:
        violation |= undecided & (array[1:] < array[:-1])
        undecided &= array[1:] == array[:-1]
    if bool(violation.any()):
        raise IndependentTypedScanError(f"{label} is not sorted by the locked typed key")


def _complete_groups(batches: Iterable[pa.RecordBatch], columns: Sequence[str], label: str) -> Iterator[pa.RecordBatch]:
    carry: pa.RecordBatch | None = None
    for incoming in batches:
        if incoming.num_rows == 0:
            continue
        combined = incoming if carry is None else _combine((carry, incoming))
        arrays = _key_arrays(combined, columns)
        _assert_sorted(arrays, label)
        same = np.ones(combined.num_rows - 1, dtype=bool)
        for array in arrays:
            same &= array[1:] == array[:-1]
        group_starts = np.r_[0, np.flatnonzero(~same) + 1]
        final_start = int(group_starts[-1])
        if final_start:
            yield combined.slice(0, final_start)
        carry = combined.slice(final_start)
    if carry is not None and carry.num_rows:
        _assert_sorted(_key_arrays(carry, columns), label)
        yield carry


def _collapse_exact(batch: pa.RecordBatch, key: Sequence[str], metrics: Counter[str], label: str) -> pa.RecordBatch:
    arrays = _key_arrays(batch, key)
    duplicate = np.zeros(batch.num_rows, dtype=bool)
    if batch.num_rows > 1:
        adjacent = np.ones(batch.num_rows - 1, dtype=bool)
        for array in arrays:
            adjacent &= array[1:] == array[:-1]
        duplicate[1:] = adjacent
    positions = np.flatnonzero(duplicate)
    if positions.size:
        metrics["duplicate_groups"] += int(np.count_nonzero(duplicate & ~np.r_[False, duplicate[:-1]]))
        metrics["duplicate_surplus_rows"] += int(positions.size)
        for position in positions.tolist():
            if any(column[position].as_py() != column[position - 1].as_py() for column in batch.columns):
                metrics["conflicting_groups"] += 1
                raise IndependentTypedScanError(f"{label} contains conflicting duplicate rows")
    return batch.filter(pa.array(~duplicate))


def _scope_batches(settings: IndependentSettings, source: str, evidence: ScanEvidence) -> Iterator[pa.RecordBatch]:
    policy_source = settings.policy["source_access"][source]
    path = settings.raw_root / policy_source["filename"]
    receipt, descriptor = _stable_hash(
        path,
        expected_size=int(policy_source["size_bytes"]),
        expected_sha256=str(policy_source["sha256"]),
    )
    evidence.raw_sources[source] = dict(receipt)
    full_rows = 0
    retained_rows = 0
    scan_stream = _DigestingRawReader(
        os.fdopen(os.dup(descriptor), "rb", buffering=0)
    )
    try:
        reader = _open_csv(
            scan_stream, path.name, settings.source_headers[source], settings.block_size
        )
    except BaseException:
        scan_stream.close()
        os.close(descriptor)
        raise
    start = settings.policy["scope"]["start_date"]
    end = settings.policy["scope"]["quality_end_date"]
    try:
        for batch in _batches(reader, source):
            full_rows += batch.num_rows
            if source == "daily":
                dates = _date(_column(batch, "DlyCalDt"), "daily scope date")
                mask = pc.and_(pc.greater_equal(dates, dt.date.fromisoformat(start)), pc.less_equal(dates, dt.date.fromisoformat(end)))
            elif source == "monthly":
                dates = _date(_column(batch, "MthCalDt"), "monthly scope date")
                mask = pc.and_(pc.greater_equal(dates, dt.date.fromisoformat(start)), pc.less_equal(dates, dt.date.fromisoformat(end)))
            else:
                start_column, end_column = (("SecInfoStartDt", "SecInfoEndDt") if source == "identity" else ("ShrStartDt", "ShrEndDt"))
                starts = _date(_column(batch, start_column), f"{source} interval start")
                ends = _date(_column(batch, end_column), f"{source} interval end")
                _require(pc.less_equal(starts, ends), f"{source} interval is inverted")
                mask = pc.and_(pc.greater_equal(ends, dt.date.fromisoformat(start)), pc.less_equal(starts, dt.date.fromisoformat(end)))
            retained = batch.filter(mask)
            retained_rows += retained.num_rows
            if retained.num_rows:
                yield retained
    finally:
        try:
            reader.close()
        finally:
            scan_stream.close()
        if (
            scan_stream.bytes_read != int(receipt["size_bytes"])
            or scan_stream.sha256 != receipt["sha256"]
        ):
            os.close(descriptor)
            raise IndependentTypedScanError(
                f"Raw bytes supplied to parser differ: {path.name}"
            )
        try:
            _recheck_stat(path, receipt, descriptor)
        finally:
            os.close(descriptor)
    locked_full = int(policy_source["full_file_rows"])
    if full_rows != locked_full:
        raise IndependentTypedScanError(f"{source} full row count differs: {full_rows} != {locked_full}")
    evidence.raw_sources[source]["full_file_rows"] = full_rows
    evidence.raw_sources[source]["retained_rows"] = retained_rows


def _sorted_interval_source(
    settings: IndependentSettings, source: str, evidence: ScanEvidence
) -> Iterator[pa.RecordBatch]:
    """Materialize and independently sort retained interval source rows.

    The raw identity and shares exports are not contractually ordered.  Their
    typed tables are, so the verifier derives typed sort keys itself before it
    compares rows with the published Parquet stream.
    """

    batches = list(_scope_batches(settings, source, evidence))
    if not batches:
        return
    raw = pa.Table.from_batches(batches)
    if source == "identity":
        start_name, end_name = "SecInfoStartDt", "SecInfoEndDt"
    elif source == "shares":
        start_name, end_name = "ShrStartDt", "ShrEndDt"
    else:
        raise IndependentTypedScanError(
            f"Interval sorting is unsupported for {source}"
        )
    keys = pa.table(
        {
            "permno": _integer(_column(raw, "PERMNO"), f"{source} sort PERMNO"),
            "start_date": _date(
                _column(raw, start_name), f"{source} sort interval start"
            ),
            "end_date": _date(
                _column(raw, end_name), f"{source} sort interval end"
            ),
        }
    )
    order = pc.sort_indices(
        keys,
        sort_keys=[
            ("permno", "ascending"),
            ("start_date", "ascending"),
            ("end_date", "ascending"),
        ],
    )
    sorted_raw = raw.take(order)
    yield from sorted_raw.to_batches(max_chunksize=ROW_GROUP_SIZE)


def _daily_canonical(settings: IndependentSettings, evidence: ScanEvidence) -> Iterator[pa.RecordBatch]:
    metrics: Counter[str] = Counter()
    for complete in _complete_groups(_scope_batches(settings, "daily", evidence), ("PERMNO", "DlyCalDt"), "daily raw stream"):
        canonical = _collapse_exact(complete, ("PERMNO", "DlyCalDt"), metrics, "daily raw stream")
        metrics["canonical_rows"] += canonical.num_rows
        if canonical.num_rows:
            yield canonical
    evidence.canonicalization["daily"] = dict(metrics)


def _monthly_first_stage(settings: IndependentSettings, evidence: ScanEvidence, metrics: Counter[str]) -> Iterator[pa.RecordBatch]:
    raw_key = ("PERMNO", "MthCalDt", "SecInfoStartDt", "SecInfoEndDt")
    for complete in _complete_groups(_scope_batches(settings, "monthly", evidence), raw_key, "monthly raw stream"):
        collapsed = _collapse_exact(complete, raw_key, metrics, "monthly raw stream")
        if collapsed.num_rows:
            yield collapsed


def _select_monthly(batch: pa.RecordBatch, metrics: Counter[str]) -> pa.RecordBatch:
    arrays = _key_arrays(batch, ("PERMNO", "MthCalDt"))
    same = np.ones(batch.num_rows - 1, dtype=bool)
    for array in arrays:
        same &= array[1:] == array[:-1]
    starts = np.r_[0, np.flatnonzero(~same) + 1, batch.num_rows]
    selected: list[int] = []
    for first, stop in zip(starts[:-1], starts[1:], strict=True):
        if stop - first == 1:
            selected.append(int(first))
            continue
        metrics["candidate_groups"] += 1
        metrics[f"candidate_size_{int(stop - first)}"] += 1
        price_matches: list[int] = []
        calendar_matches: list[int] = []
        for position in range(int(first), int(stop)):
            interval_start = dt.date.fromisoformat(_column(batch, "SecInfoStartDt")[position].as_py())
            interval_end = dt.date.fromisoformat(_column(batch, "SecInfoEndDt")[position].as_py())
            price_raw = _column(batch, "MthPrcDt")[position].as_py()
            calendar = dt.date.fromisoformat(_column(batch, "MthCalDt")[position].as_py())
            if price_raw and interval_start <= dt.date.fromisoformat(price_raw) <= interval_end:
                price_matches.append(position)
            if interval_start <= calendar <= interval_end:
                calendar_matches.append(position)
        if len(price_matches) != 1 or len(calendar_matches) != 1 or price_matches[0] != calendar_matches[0]:
            raise IndependentTypedScanError("Monthly candidate selection is not unique and consistent")
        selected.append(price_matches[0])
    return batch.take(pa.array(selected, pa.int64()))


def _monthly_canonical(settings: IndependentSettings, evidence: ScanEvidence) -> Iterator[pa.RecordBatch]:
    metrics: Counter[str] = Counter()
    first = _monthly_first_stage(settings, evidence, metrics)
    for complete in _complete_groups(first, ("PERMNO", "MthCalDt"), "monthly canonical stream"):
        chosen = _select_monthly(complete, metrics)
        metrics["canonical_rows"] += chosen.num_rows
        if chosen.num_rows:
            yield chosen
    evidence.canonicalization["monthly"] = dict(metrics)


def _table(contract: TableContract, columns: Mapping[str, pa.Array | pa.ChunkedArray], rows: int) -> pa.Table:
    arrays = [columns[str(field["name"])] for field in contract.fields]
    table = pa.Table.from_arrays(arrays, schema=contract.arrow_schema())
    if table.num_rows != rows:
        raise IndependentTypedScanError(f"Independent transform row count differs for {contract.name}")
    table.validate(full=True)
    return table


def _scaled_1000(values: pa.Array | pa.ChunkedArray, target: pa.Decimal128Type) -> pa.Array | pa.ChunkedArray:
    wider = pc.cast(values, pa.decimal256(38, target.scale), safe=True)
    result = pc.multiply_checked(wider, pa.scalar(Decimal("1000"), pa.decimal256(4, 0)))
    return pc.cast(result, options=pc.CastOptions(target_type=target, allow_decimal_truncate=False))


def _daily_tables(settings: IndependentSettings, raw_batch: pa.RecordBatch) -> tuple[pa.Table, pa.Table]:
    r = {name: _column(raw_batch, name) for name in raw_batch.schema.names}
    d6, d2, d1 = pa.decimal128(38, 6), pa.decimal128(38, 2), pa.decimal128(38, 1)
    permno = _integer(r["PERMNO"], "daily PERMNO")
    date = _date(r["DlyCalDt"], "daily calendar date")
    interval_start = _date(r["SecInfoStartDt"], "daily interval start")
    interval_end = _date(r["SecInfoEndDt"], "daily interval end")
    _require(pc.less_equal(interval_start, interval_end), "daily embedded interval is inverted")
    close = _decimal(r["DlyClose"], d6, "DlyClose")
    cap = _decimal(r["DlyCap"], d2, "DlyCap")
    volume = _integer(r["DlyVol"], "DlyVol", optional=True, positive=False)
    price_valid = pc.fill_null(pc.greater(close, Decimal("0")), False)
    cap_valid = pc.fill_null(pc.greater(cap, Decimal("0")), False)
    liquidity = pc.and_(price_valid, pc.is_valid(volume))
    dollar = pc.cast(
        pc.multiply_checked(
            pc.cast(pc.if_else(liquidity, close, None), pa.decimal256(38, 6), safe=True),
            pc.cast(pc.if_else(liquidity, volume, None), pa.decimal256(19, 0), safe=True),
        ), options=pc.CastOptions(target_type=d6, allow_decimal_truncate=False),
    )
    consumer = {
        "permno": permno,
        "source_permco": _integer(r["PERMCO"], "daily PERMCO", optional=True),
        "source_security_info_start_date": interval_start,
        "source_security_info_end_date": interval_end,
        "source_cusip": r["CUSIP"], "source_primary_exchange": r["PrimaryExch"],
        "conditional_type": r["ConditionalType"], "source_trading_status_flag": r["TradingStatusFlg"],
        "source_us_incorporation_flag": r["USIncFlg"], "source_issuer_type": r["IssuerType"],
        "source_security_type": r["SecurityType"], "source_security_subtype": r["SecuritySubType"],
        "source_share_type": r["ShareType"], "source_ticker": r["Ticker"],
        "source_sic_code": r["SICCD"], "source_naics_code": r["NAICS"],
        "source_icb_industry": r["ICBIndustry"], "calendar_date": date,
        "diagnostic_price_usd_per_share": _decimal(r["DlyPrc"], d6, "DlyPrc"),
        "price_flag": r["DlyPrcFlg"], "market_cap_thousand_usd": cap,
        "market_cap_usd": _scaled_1000(cap, d2), "volume_shares": volume,
        "close_usd_per_share": close, "low_usd_per_share": _decimal(r["DlyLow"], d6, "DlyLow"),
        "high_usd_per_share": _decimal(r["DlyHigh"], d6, "DlyHigh"),
        "bid_usd_per_share": _decimal(r["DlyBid"], d6, "DlyBid"),
        "ask_usd_per_share": _decimal(r["DlyAsk"], d6, "DlyAsk"),
        "open_usd_per_share": _decimal(r["DlyOpen"], d6, "DlyOpen"),
        "vendor_price_volume_usd": _decimal(r["DlyPrcVol"], d1, "DlyPrcVol"),
        "screening_dollar_volume_usd": dollar,
        "screening_price_valid": price_valid, "market_cap_valid": cap_valid,
        "liquidity_session_valid": liquidity,
    }
    duration_keys = pa.array(tuple(_DURATION), pa.string())
    duration_values = pa.array(tuple(_DURATION.values()), pa.int16())
    indexes = pc.index_in(r["DlyRetDurFlg"], value_set=duration_keys)
    known = pc.greater_equal(indexes, 0)
    mapped = pc.if_else(known, pc.take(duration_values, pc.if_else(known, indexes, 0)), None)
    returns = {
        "permno": permno, "calendar_date": date,
        "raw_total_return": _decimal(r["DlyRet"], d6, "DlyRet"),
        "raw_return_ex_distributions": _decimal(r["DlyRetx"], d6, "DlyRetx"),
        "raw_return_missing_flag": r["DlyRetMissFlg"],
        "raw_return_duration_flag": r["DlyRetDurFlg"], "mapped_trading_periods": mapped,
        "reconciliation_status": pa.repeat("unreconciled", raw_batch.num_rows),
        "return_history_authorized": pa.repeat(False, raw_batch.num_rows),
    }
    return _table(settings.contracts["crsp_security_day"], consumer, raw_batch.num_rows), _table(settings.contracts["crsp_security_day_return_staging"], returns, raw_batch.num_rows)


def _monthly_tables(settings: IndependentSettings, raw_batch: pa.RecordBatch) -> tuple[pa.Table, pa.Table]:
    r = {name: _column(raw_batch, name) for name in raw_batch.schema.names}
    d6, d2, d1 = pa.decimal128(38, 6), pa.decimal128(38, 2), pa.decimal128(38, 1)
    permno = _integer(r["PERMNO"], "monthly PERMNO")
    month = _date(r["MthCalDt"], "monthly calendar month")
    observation = _date(r["MthPrcDt"], "monthly observation date", optional=True)
    interval_start = _date(r["SecInfoStartDt"], "monthly interval start")
    interval_end = _date(r["SecInfoEndDt"], "monthly interval end")
    _require(pc.less_equal(interval_start, interval_end), "monthly embedded interval is inverted")
    yyyymm = _integer(r["YYYYMM"], "YYYYMM", positive=True, target=pa.int32())
    calendar_period = pc.binary_join_element_wise(pc.utf8_slice_codeunits(r["MthCalDt"], 0, 4), pc.utf8_slice_codeunits(r["MthCalDt"], 5, 7), "")
    observation_period = pc.binary_join_element_wise(pc.utf8_slice_codeunits(r["MthPrcDt"], 0, 4), pc.utf8_slice_codeunits(r["MthPrcDt"], 5, 7), "")
    _require(pc.equal(r["YYYYMM"], calendar_period), "YYYYMM differs from calendar month")
    _require(pc.or_(pc.equal(r["MthPrcDt"], ""), pc.equal(r["YYYYMM"], observation_period)), "Observation date period differs from YYYYMM")
    cap = _decimal(r["MthCap"], d2, "MthCap")
    consumer = {
        "permno": permno, "source_permco": _integer(r["PERMCO"], "monthly PERMCO", optional=True),
        "source_security_info_start_date": interval_start,
        "source_security_info_end_date": interval_end,
        "source_cusip": r["CUSIP"], "source_primary_exchange": r["PrimaryExch"],
        "source_conditional_type": r["ConditionalType"], "source_trading_status_flag": r["TradingStatusFlg"],
        "source_us_incorporation_flag": r["USIncFlg"], "source_issuer_type": r["IssuerType"],
        "source_security_type": r["SecurityType"], "source_security_subtype": r["SecuritySubType"],
        "source_share_type": r["ShareType"], "source_sic_code": r["SICCD"], "yyyymm": yyyymm,
        "calendar_month": month, "price_usd_per_share": _decimal(r["MthPrc"], d6, "MthPrc"),
        "price_flag": r["MthPrcFlg"], "observation_date": observation,
        "market_cap_thousand_usd": cap, "market_cap_usd": _scaled_1000(cap, d2),
        "distribution_count": _integer(r["MthDisCnt"], "MthDisCnt", optional=True, positive=False),
        "volume_shares": _integer(r["MthVol"], "MthVol", optional=True, positive=False),
        "volume_flag": r["MthVolFlg"], "source_share_factor_flag": r["MthFacShrFlg"],
        "price_volume_usd": _decimal(r["MthPrcVol"], d1, "MthPrcVol"),
        "price_volume_missing_count": _integer(r["MthPrcVolMissCnt"], "MthPrcVolMissCnt", optional=True, positive=False),
        "embedded_shares_outstanding_thousand": _integer(r["ShrOut"], "monthly ShrOut", optional=True),
        "observation_date_available": pc.is_valid(observation),
        "market_cap_valid": pc.fill_null(pc.greater(cap, Decimal("0")), False),
    }
    returns = {
        "permno": permno, "calendar_month": month,
        "raw_total_return": _decimal(r["MthRet"], d6, "MthRet"),
        "raw_return_ex_distributions": _decimal(r["MthRetx"], d6, "MthRetx"),
        "raw_return_flag": r["MthRetFlg"],
        "reconciliation_status": pa.repeat("unreconciled", raw_batch.num_rows),
        "return_history_authorized": pa.repeat(False, raw_batch.num_rows),
    }
    return _table(settings.contracts["crsp_security_month_crosscheck"], consumer, raw_batch.num_rows), _table(settings.contracts["crsp_security_month_return_staging"], returns, raw_batch.num_rows)


def _identity_table(settings: IndependentSettings, raw_batch: pa.RecordBatch) -> pa.Table:
    r = {name: _column(raw_batch, name) for name in raw_batch.schema.names}
    start = _date(r["SecInfoStartDt"], "identity start")
    end = _date(r["SecInfoEndDt"], "identity end")
    security_begin = _date(r["SecurityBegDt"], "security begin")
    security_end = _date(r["SecurityEndDt"], "security end")
    _require(pc.less_equal(start, end), "identity information interval is inverted")
    _require(pc.less_equal(security_begin, security_end), "identity security lifetime is inverted")
    values = {
        "permno": _integer(r["PERMNO"], "identity PERMNO"), "permco": _integer(r["PERMCO"], "identity PERMCO"),
        "start_date": start, "end_date": end,
        "security_begin_date": security_begin, "security_end_date": security_end,
        "cusip": r["CUSIP"], "ticker": r["Ticker"], "issuer_name": r["IssuerNm"],
        "us_incorporation_flag": r["USIncFlg"], "issuer_type": r["IssuerType"],
        "security_type": r["SecurityType"], "security_subtype": r["SecuritySubType"],
        "share_type": r["ShareType"], "sic_code": r["SICCD"], "primary_exchange": r["PrimaryExch"],
        "naics_code": r["NAICS"], "trading_status_flag": r["TradingStatusFlg"],
    }
    return _table(settings.contracts["crsp_identity_interval"], values, raw_batch.num_rows)


def _shares_table(settings: IndependentSettings, raw_batch: pa.RecordBatch) -> pa.Table:
    r = {name: _column(raw_batch, name) for name in raw_batch.schema.names}
    thousand = _integer(r["ShrOut"], "shares outstanding")
    try:
        shares = pc.multiply_checked(thousand, 1000)
    except pa.ArrowException as exc:
        raise IndependentTypedScanError("Shares outstanding x1000 overflows int64") from exc
    values = {
        "permno": _integer(r["PERMNO"], "shares PERMNO"), "start_date": _date(r["ShrStartDt"], "shares start"),
        "end_date": _date(r["ShrEndDt"], "shares end"), "shares_outstanding_thousand": thousand,
        "shares_outstanding": shares, "shares_source": r["ShrSource"],
        "shares_factor_type": r["ShrFacType"], "shares_adr_flag": r["ShrAdrFlg"],
    }
    return _table(settings.contracts["crsp_shares_interval"], values, raw_batch.num_rows)


def _true_count(mask: pa.Array | pa.ChunkedArray) -> int:
    return int(
        pc.sum(pc.cast(pc.fill_null(mask, False), pa.int64())).as_py() or 0
    )


def _update_decimal_evidence(
    evidence: ScanEvidence,
    raw: pa.RecordBatch,
    table: pa.Table,
    contract: TableContract,
) -> None:
    for field in contract.fields:
        if (
            field["type"].startswith("decimal128")
            and field.get("transform") == "exact_decimal"
            and isinstance(field.get("source_column"), str)
        ):
            source = _column(raw, field["source_column"])
            nonblank = pc.not_equal(source, "")
            mantissa = pc.replace_substring_regex(
                source, pattern=r"[Ee].*$", replacement=""
            )
            coefficient = pc.replace_substring_regex(
                pc.replace_substring_regex(
                    mantissa, pattern=r"^[+-]", replacement=""
                ),
                pattern=r"\.",
                replacement="",
            )
            significant = pc.replace_substring_regex(
                coefficient, pattern=r"^0+", replacement=""
            )
            fallback = pc.and_(
                nonblank,
                pc.greater(pc.utf8_length(significant), pa.scalar(38, pa.int32())),
            )
            typed = table[field["name"]].combine_chunks()
            prepared = pc.if_else(
                pc.equal(source, ""), None,
                pc.if_else(fallback, "0", source),
            )
            try:
                ordinary_expected = pc.cast(
                    prepared,
                    options=pc.CastOptions(
                        target_type=typed.type, allow_decimal_truncate=False
                    ),
                )
            except pa.ArrowException as exc:
                raise IndependentTypedScanError(
                    f"{contract.name}.{field['name']} decimal evidence cannot be parsed"
                ) from exc
            ordinary = pc.and_(nonblank, pc.invert(fallback))
            ordinary_mismatch = _true_count(
                pc.and_(
                    ordinary,
                    pc.invert(
                        pc.fill_null(pc.equal(ordinary_expected, typed), False)
                    ),
                )
            )
            fallback_rows = 0
            fallback_mismatch = 0
            for position in pc.indices_nonzero(fallback).to_pylist():
                fallback_rows += 1
                try:
                    expected_value = Decimal(source[position].as_py())
                except (InvalidOperation, ValueError):
                    fallback_mismatch += 1
                    continue
                if typed[position].as_py() != expected_value:
                    fallback_mismatch += 1
            count = _true_count(nonblank)
            mismatch = ordinary_mismatch + fallback_mismatch
            evidence.decimal_roundtrip[contract.name][field["name"]].update(
                {
                    "arbitrary_precision_fallback_rows": fallback_rows,
                    "exact_match_rows": count - mismatch,
                    "grammar_failure_rows": 0,
                    "mismatch_rows": mismatch,
                    "source_nonblank_rows": count,
                }
            )


def _record_formula_evidence(
    target: Counter[str],
    expected: pa.Array | pa.ChunkedArray,
    actual: pa.Array | pa.ChunkedArray,
) -> None:
    matches = pc.or_(
        pc.and_(pc.is_null(expected), pc.is_null(actual)),
        pc.fill_null(pc.equal(expected, actual), False),
    )
    rows = len(expected)
    mismatch = rows - _true_count(matches)
    target.update(
        {
            "actual_null_rows": int(actual.null_count),
            "expected_null_rows": int(expected.null_count),
            "match_rows": rows - mismatch,
            "mismatch_rows": mismatch,
            "rows": rows,
        }
    )


def _update_formula_evidence(
    evidence: ScanEvidence, table: pa.Table, contract: TableContract
) -> None:
    if contract.name in {
        "crsp_security_day",
        "crsp_security_month_crosscheck",
    }:
        source = table["market_cap_thousand_usd"].combine_chunks()
        actual = table["market_cap_usd"].combine_chunks()
        expected = pc.multiply_checked(
            pc.cast(
                source, pa.decimal256(source.type.precision, source.type.scale), safe=True
            ),
            pa.scalar(Decimal("1000"), pa.decimal256(4, 0)),
        )
        _record_formula_evidence(
            evidence.formula_quality[f"{contract.name}.market_cap_x1000"],
            expected,
            pc.cast(actual, expected.type, safe=True),
        )
    elif contract.name == "crsp_shares_interval":
        source = table["shares_outstanding_thousand"].combine_chunks()
        actual = table["shares_outstanding"].combine_chunks()
        expected = pc.multiply_checked(source, pa.scalar(1000, pa.int64()))
        _record_formula_evidence(
            evidence.formula_quality[f"{contract.name}.shares_x1000"],
            expected,
            actual,
        )


def _update_date_scope_evidence(
    evidence: ScanEvidence,
    table: pa.Table,
    contract: TableContract,
    settings: IndependentSettings,
) -> None:
    date_name = next(
        (
            name
            for name in ("calendar_date", "calendar_month")
            if name in table.schema.names
        ),
        None,
    )
    if date_name is None:
        return
    values = table[date_name].combine_chunks()
    start = dt.date.fromisoformat(settings.policy["scope"]["start_date"])
    end = dt.date.fromisoformat(settings.policy["scope"]["quality_end_date"])
    target = evidence.date_scope[contract.name]
    target["rows"] += table.num_rows
    target["before_scope_rows"] += _true_count(pc.less(values, start))
    target["after_scope_rows"] += _true_count(pc.greater(values, end))
    target["december_2014_rows"] += _true_count(
        pc.and_(
            pc.greater_equal(values, dt.date(2014, 12, 1)),
            pc.less_equal(values, end),
        )
    )
    target["boundary_2014_12_01_rows"] += _true_count(
        pc.equal(values, dt.date(2014, 12, 1))
    )


def _update_table_evidence(
    evidence: ScanEvidence,
    raw: pa.RecordBatch,
    table: pa.Table,
    contract: TableContract,
    settings: IndependentSettings,
) -> None:
    _update_decimal_evidence(evidence, raw, table, contract)
    _update_formula_evidence(evidence, table, contract)
    _update_date_scope_evidence(evidence, table, contract, settings)


def _update_quarantine_evidence(
    evidence: ScanEvidence, table: pa.Table, table_name: str
) -> None:
    status = table["reconciliation_status"].combine_chunks()
    authorized = table["return_history_authorized"].combine_chunks()
    target = evidence.quarantine_guards[table_name]
    target["rows"] += table.num_rows
    target["status_not_unreconciled_rows"] += _true_count(
        pc.not_equal(status, "unreconciled")
    )
    target["authorized_true_rows"] += _true_count(authorized)
    target["status_null_rows"] += status.null_count
    target["authorized_null_rows"] += authorized.null_count


class _ParquetCursor:
    def __init__(self, parquet: pq.ParquetFile, contract: TableContract, evidence: ScanEvidence) -> None:
        self.contract = contract
        self.evidence = evidence
        self.iterator = iter(parquet.iter_batches(batch_size=ROW_GROUP_SIZE, use_threads=True))
        self.batch: pa.RecordBatch | None = None
        self.offset = 0
        self.rows = 0
        self.exhausted = False
        self.key_null_rows = 0
        self.key_duplicate_rows = 0
        self.key_sorted_violations = 0
        self.last_key: tuple[int, ...] | None = None
        self.key_digest = hashlib.sha256()
        self.key_digest.update(b"phase02-crsp-typed-key-v1\x00")
        digest_label = {
            "crsp_identity_interval": "identity",
            "crsp_shares_interval": "shares",
            "crsp_security_day": "daily",
            "crsp_security_day_return_staging": "daily",
            "crsp_security_month_crosscheck": "monthly",
            "crsp_security_month_return_staging": "monthly",
        }[contract.name]
        encoded = (
            digest_label + "\x00" + "\x00".join(contract.primary_key)
        ).encode("utf-8")
        self.key_digest.update(len(encoded).to_bytes(8, "big"))
        self.key_digest.update(encoded)

    def _next(self) -> None:
        try:
            self.batch = next(self.iterator)
            self.offset = 0
            self.rows += self.batch.num_rows
            for name, column in zip(self.batch.schema.names, self.batch.columns, strict=True):
                self.evidence.table_nulls[self.contract.name][name] += column.null_count
        except StopIteration:
            self.batch = None
            self.exhausted = True

    def consume(self, expected: pa.Table) -> None:
        packed_fields: list[tuple[str, str]] = []
        packed_values: list[np.ndarray] = []
        null_row = np.zeros(expected.num_rows, dtype=bool)
        for index, name in enumerate(self.contract.primary_key):
            column = expected[name].combine_chunks()
            if column.null_count:
                null_row |= np.asarray(
                    pc.is_null(column).to_numpy(zero_copy_only=False), dtype=bool
                )
            if pa.types.is_int64(column.type):
                dtype = ">i8"
                values = np.asarray(
                    column.to_numpy(zero_copy_only=False), dtype=dtype
                )
            elif pa.types.is_date32(column.type):
                dtype = ">i4"
                values = (
                    np.asarray(column.to_numpy(zero_copy_only=False))
                    .astype("datetime64[D]")
                    .astype(dtype)
                )
            else:
                raise IndependentTypedScanError(
                    f"Unsupported primary-key type: {self.contract.name}.{name}"
                )
            packed_fields.append((f"f{index}", dtype))
            packed_values.append(values)
        self.key_null_rows += int(np.count_nonzero(null_row))
        if packed_values and expected.num_rows:
            adjacent_equal = np.ones(expected.num_rows - 1, dtype=bool)
            undecided = np.ones(expected.num_rows - 1, dtype=bool)
            violations = np.zeros(expected.num_rows - 1, dtype=bool)
            for values in packed_values:
                adjacent_equal &= values[1:] == values[:-1]
                violations |= undecided & (values[1:] < values[:-1])
                undecided &= values[1:] == values[:-1]
            self.key_duplicate_rows += int(np.count_nonzero(adjacent_equal))
            self.key_sorted_violations += int(np.count_nonzero(violations))
            first_key = tuple(int(values[0]) for values in packed_values)
            if self.last_key is not None:
                if first_key == self.last_key:
                    self.key_duplicate_rows += 1
                elif first_key < self.last_key:
                    self.key_sorted_violations += 1
            self.last_key = tuple(int(values[-1]) for values in packed_values)
        packed = np.empty(expected.num_rows, dtype=np.dtype(packed_fields, align=False))
        for index, values in enumerate(packed_values):
            packed[f"f{index}"] = values
        self.key_digest.update(packed.tobytes(order="C"))
        for name, column in zip(expected.schema.names, expected.columns, strict=True):
            self.evidence.raw_expected_nulls[self.contract.name][name] += (
                column.null_count
            )
        position = 0
        while position < expected.num_rows:
            if self.batch is None or self.offset == self.batch.num_rows:
                self._next()
            if self.batch is None:
                raise IndependentTypedScanError(f"Parquet ended early: {self.contract.name}")
            count = min(expected.num_rows - position, self.batch.num_rows - self.offset)
            actual = pa.Table.from_batches([self.batch.slice(self.offset, count)])
            wanted = expected.slice(position, count)
            if not actual.equals(wanted, check_metadata=False):
                for name in wanted.schema.names:
                    if not actual[name].equals(wanted[name]):
                        raise IndependentTypedScanError(f"Raw-derived typed values differ: {self.contract.name}.{name}")
                raise IndependentTypedScanError(f"Raw-derived typed rows differ: {self.contract.name}")
            self.evidence.compared_values += count * len(wanted.schema)
            self.offset += count
            position += count

    def finish(self) -> None:
        if self.batch is not None and self.offset < self.batch.num_rows:
            raise IndependentTypedScanError(f"Parquet has unmatched trailing rows: {self.contract.name}")
        if not self.exhausted:
            self._next()
        if self.batch is not None:
            raise IndependentTypedScanError(f"Parquet has unmatched trailing rows: {self.contract.name}")
        if self.rows != self.contract.expected_rows:
            raise IndependentTypedScanError(f"Parquet scan count differs: {self.contract.name}")
        self.evidence.table_rows[self.contract.name] = self.rows
        self.evidence.key_sha256[self.contract.name] = self.key_digest.hexdigest()
        self.evidence.key_quality[self.contract.name] = {
            "duplicate_rows": self.key_duplicate_rows,
            "null_rows": self.key_null_rows,
            "primary_key": list(self.contract.primary_key),
            "rows": self.rows,
            "sorted_violations": self.key_sorted_violations,
        }


def inspect_parquet(parquet: pq.ParquetFile, contract: TableContract) -> dict[str, Any]:
    if not parquet.schema_arrow.equals(contract.arrow_schema(), check_metadata=True):
        raise IndependentTypedScanError(f"Exact Arrow schema differs: {contract.name}")
    metadata = parquet.metadata
    if metadata.num_rows != contract.expected_rows:
        raise IndependentTypedScanError(f"Parquet metadata row count differs: {contract.name}")
    group_rows = [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)]
    if any(value != ROW_GROUP_SIZE for value in group_rows[:-1]) or (group_rows and not 0 < group_rows[-1] <= ROW_GROUP_SIZE):
        raise IndependentTypedScanError(f"Parquet row-group sizing differs: {contract.name}")
    string_indexes = {index for index, field in enumerate(contract.arrow_schema()) if pa.types.is_string(field.type)}
    compression_codecs: set[str] = set()
    statistics_complete = True
    dictionary_complete = True
    for group_index in range(metadata.num_row_groups):
        group = metadata.row_group(group_index)
        for column_index in range(group.num_columns):
            column = group.column(column_index)
            compression_codecs.add(column.compression)
            statistics_complete &= column.statistics is not None
            if column.compression != "ZSTD" or column.statistics is None:
                raise IndependentTypedScanError(f"Parquet compression/statistics differ: {contract.name}")
            if column_index in string_indexes:
                encoded = bool(
                    {"RLE_DICTIONARY", "PLAIN_DICTIONARY"}
                    & set(column.encodings)
                )
                dictionary_complete &= encoded
                if not encoded:
                    raise IndependentTypedScanError(f"UTF-8 dictionary encoding absent: {contract.name}")
    if str(metadata.format_version) != "2.6":
        raise IndependentTypedScanError(f"Parquet format version differs: {contract.name}")
    key_value_metadata = metadata.metadata or {}
    store_schema = b"ARROW:schema" in key_value_metadata
    if not store_schema:
        raise IndependentTypedScanError(
            f"Stored Arrow schema is absent: {contract.name}"
        )
    return {
        "compression_codecs": sorted(compression_codecs),
        "data_page_version_proof": (
            "writer_configuration_bound_by_implementation_sha256"
        ),
        "dictionary_encode_utf8": dictionary_complete,
        "row_count": metadata.num_rows,
        "row_group_count": metadata.num_row_groups,
        "row_group_rows": group_rows,
        "format_version": str(metadata.format_version),
        "key_value_metadata_keys": sorted(
            key.decode("ascii", errors="strict") for key in key_value_metadata
        ),
        "schema_fingerprint": contract.fingerprint(),
        "statistics_complete": statistics_complete,
        "store_schema": store_schema,
    }


@dataclasses.dataclass(frozen=True)
class IntervalIndex:
    permnos: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    slices: Mapping[int, tuple[int, int]]
    payload: Mapping[str, pa.Array]
    unique_permnos: int
    duplicate_key_groups: int
    contiguous_edges: int
    gap_edges: int
    overlap_edges: int

    def match(self, permnos: np.ndarray, dates: np.ndarray, valid: np.ndarray) -> np.ndarray:
        result = np.full(len(permnos), -1, dtype=np.int64)
        for permno in np.unique(permnos[valid]):
            rows = np.flatnonzero(valid & (permnos == permno))
            bounds = self.slices.get(int(permno))
            if bounds is None:
                continue
            first, stop = bounds
            positions = np.searchsorted(self.starts[first:stop], dates[rows], side="right") - 1
            usable = positions >= 0
            safe = np.maximum(positions, 0)
            usable &= dates[rows] <= self.ends[first:stop][safe]
            result[rows[usable]] = first + positions[usable]
        return result


def _interval_index(tables: Sequence[pa.Table], payload_mapping: Mapping[str, str]) -> IntervalIndex:
    combined = pa.concat_tables(tables).combine_chunks()
    permnos = np.asarray(combined["permno"].to_numpy(), dtype=np.int64)
    starts = np.asarray(combined["start_date"].to_numpy()).astype("datetime64[D]").astype(np.int32)
    ends = np.asarray(combined["end_date"].to_numpy()).astype("datetime64[D]").astype(np.int32)
    _assert_sorted((permnos, starts, ends), "published interval table")
    same_permno = permnos[1:] == permnos[:-1]
    duplicate_key = same_permno & (starts[1:] == starts[:-1]) & (ends[1:] == ends[:-1])
    overlap = same_permno & (starts[1:] <= ends[:-1])
    contiguous = same_permno & (starts[1:] == ends[:-1] + 1)
    gap = same_permno & (starts[1:] > ends[:-1] + 1)
    if bool(np.any(overlap)):
        raise IndependentTypedScanError("Published intervals overlap")
    group_starts = np.r_[0, np.flatnonzero(~same_permno) + 1]
    stops = np.r_[group_starts[1:], len(permnos)]
    slices = {int(permnos[first]): (int(first), int(stop)) for first, stop in zip(group_starts, stops, strict=True)}
    payload = {raw: combined[typed].combine_chunks() for raw, typed in payload_mapping.items()}
    return IntervalIndex(
        permnos, starts, ends, slices, payload, len(slices),
        int(np.count_nonzero(duplicate_key)), int(np.count_nonzero(contiguous)),
        int(np.count_nonzero(gap)), int(np.count_nonzero(overlap)),
    )


def _days(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return np.asarray(values.to_numpy(zero_copy_only=False)).astype("datetime64[D]").astype(np.int32)


def _join_diagnostics(raw: pa.RecordBatch, identity: IntervalIndex, shares: IntervalIndex, *, monthly: bool) -> tuple[Counter[str], Counter[str]]:
    permnos = np.asarray(_integer(_column(raw, "PERMNO"), "join PERMNO").to_numpy(), dtype=np.int64)
    asof_raw = _column(raw, "MthPrcDt" if monthly else "DlyCalDt")
    if monthly:
        valid = np.asarray(
            pc.not_equal(asof_raw, "").to_numpy(zero_copy_only=False), dtype=bool
        )
        parsed = _date(asof_raw, "monthly join date", optional=True)
        dates = np.zeros(raw.num_rows, dtype=np.int32)
        if bool(valid.any()):
            dates[valid] = _days(pc.filter(parsed, pa.array(valid)))
    else:
        valid = np.ones(raw.num_rows, dtype=bool)
        dates = _days(_date(asof_raw, "daily join date"))
    identity_matches = identity.match(permnos, dates, valid)
    shares_matches = shares.match(permnos, dates, valid)
    identity_metrics: Counter[str] = Counter(
        population_rows=raw.num_rows,
        blank_as_of_date_rows=int(np.count_nonzero(~valid)),
        one_match_rows=int(np.count_nonzero(identity_matches >= 0)),
        zero_match_rows=int(np.count_nonzero(identity_matches < 0)),
        multiple_match_rows=0,
    )
    if monthly:
        identity_metrics["nonblank_as_of_date_rows"] = int(
            np.count_nonzero(valid)
        )
        identity_metrics["zero_match_rows_with_nonblank_as_of_date"] = int(
            np.count_nonzero(valid & (identity_matches < 0))
        )
    embedded_starts = _days(_date(_column(raw, "SecInfoStartDt"), "embedded interval start"))
    embedded_ends = _days(_date(_column(raw, "SecInfoEndDt"), "embedded interval end"))
    embedded_matches = identity.match(permnos, embedded_starts, np.ones(raw.num_rows, dtype=bool))
    safe_embedded = np.maximum(embedded_matches, 0)
    embedded_exact = (embedded_matches >= 0) & (identity.starts[safe_embedded] == embedded_starts) & (identity.ends[safe_embedded] == embedded_ends)
    identity_metrics["embedded_interval_exact_key_match_rows"] = int(np.count_nonzero(embedded_exact))
    identity_metrics["embedded_interval_exact_key_missing_rows"] = int(np.count_nonzero(~embedded_exact))
    safe = np.maximum(identity_matches, 0)
    interval_equal = (identity_matches >= 0) & (identity.starts[safe] == embedded_starts) & (identity.ends[safe] == embedded_ends)
    identity_metrics["as_of_interval_key_equal_rows"] = int(np.count_nonzero(interval_equal))
    identity_metrics["as_of_interval_key_different_rows"] = int(np.count_nonzero((identity_matches >= 0) & ~interval_equal))
    payload_equal = identity_matches >= 0
    take = pa.array(safe, pa.int64())
    for field in IDENTITY_PAYLOAD_FIELDS:
        reference = pc.take(identity.payload[field], take)
        observed = _column(raw, field)
        if pa.types.is_integer(reference.type):
            observed = _integer(observed, f"embedded identity {field}")
        payload_equal &= np.asarray(
            pc.equal(observed, reference).to_numpy(zero_copy_only=False),
            dtype=bool,
        )
    identity_metrics["as_of_payload_all_fields_equal_rows"] = int(np.count_nonzero(payload_equal))
    identity_metrics["as_of_payload_one_or_more_fields_different_rows"] = int(np.count_nonzero((identity_matches >= 0) & ~payload_equal))
    shares_metrics: Counter[str] = Counter(
        population_rows=raw.num_rows,
        blank_observation_date_rows=int(np.count_nonzero(~valid)),
        nonblank_observation_date_rows=int(np.count_nonzero(valid)),
        one_match_rows=int(np.count_nonzero(shares_matches >= 0)),
        zero_match_rows=int(np.count_nonzero(shares_matches < 0)),
        zero_match_rows_with_nonblank_observation_date=int(np.count_nonzero(valid & (shares_matches < 0))),
        total_rows_without_usable_share_match=int(np.count_nonzero(shares_matches < 0)),
        multiple_match_rows=0,
    )
    return identity_metrics, shares_metrics


def _add(target: Counter[str], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[key] += int(value)


def scan_and_compare(
    settings: IndependentSettings,
    parquet_openers: Mapping[str, Callable[[], Any]],
    *,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> ScanEvidence:
    """Reread raw inputs and every Parquet value through independent logic.

    ``parquet_openers`` must return context managers yielding binary streams.
    This permits the orchestrator to keep immutable-bundle descriptor guards in
    force for the complete scan.
    """

    import contextlib

    progress = progress or (lambda _stage, _details: None)
    evidence = ScanEvidence()
    return_sources = {
        "DlyRet", "DlyRetx", "DlyRetMissFlg", "DlyRetDurFlg",
        "MthRet", "MthRetx", "MthRetFlg",
    }
    for table_name in (
        "crsp_security_day",
        "crsp_security_month_crosscheck",
        "crsp_identity_interval",
        "crsp_shares_interval",
    ):
        prohibited = sorted(
            field["name"]
            for field in settings.contracts[table_name].fields
            if (
                isinstance(field.get("source_column"), str)
                and field["source_column"] in return_sources
            )
            or "return" in field["name"].lower()
        )
        evidence.consumer_return_absence[table_name] = {
            "prohibited_column_count": len(prohibited),
            "prohibited_columns": prohibited,
        }
    with contextlib.ExitStack() as stack:
        parquets: dict[str, pq.ParquetFile] = {}
        cursors: dict[str, _ParquetCursor] = {}
        for name, filename in TABLE_FILES.items():
            stream = stack.enter_context(parquet_openers[filename]())
            parquet = pq.ParquetFile(pa.PythonFile(stream, mode="r"))
            parquets[name] = parquet
            metadata = inspect_parquet(parquet, settings.contracts[name])
            metadata["file"] = filename
            evidence.parquet[filename] = metadata
            cursors[name] = _ParquetCursor(parquet, settings.contracts[name], evidence)

        identity_tables: list[pa.Table] = []
        for raw in _sorted_interval_source(settings, "identity", evidence):
            typed = _identity_table(settings, raw)
            _update_table_evidence(
                evidence, raw, typed,
                settings.contracts["crsp_identity_interval"], settings,
            )
            cursors["crsp_identity_interval"].consume(typed)
            identity_tables.append(typed)
        cursors["crsp_identity_interval"].finish()
        identity_map = {
            "PERMCO": "permco", "CUSIP": "cusip", "PrimaryExch": "primary_exchange",
            "TradingStatusFlg": "trading_status_flag", "USIncFlg": "us_incorporation_flag",
            "IssuerType": "issuer_type", "SecurityType": "security_type",
            "SecuritySubType": "security_subtype", "ShareType": "share_type",
            "Ticker": "ticker", "SICCD": "sic_code", "NAICS": "naics_code", "IssuerNm": "issuer_name",
        }
        identity_index = _interval_index(identity_tables, identity_map)
        evidence.canonicalization["identity"] = {
            "canonical_rows": len(identity_index.permnos),
            "unique_permnos": identity_index.unique_permnos,
            "duplicate_key_groups": identity_index.duplicate_key_groups,
            "contiguous_edges": identity_index.contiguous_edges,
            "gap_edges": identity_index.gap_edges,
            "overlap_edges": identity_index.overlap_edges,
        }
        progress("identity_verified", {"rows": evidence.table_rows["crsp_identity_interval"]})

        share_tables: list[pa.Table] = []
        for raw in _sorted_interval_source(settings, "shares", evidence):
            typed = _shares_table(settings, raw)
            _update_table_evidence(
                evidence, raw, typed,
                settings.contracts["crsp_shares_interval"], settings,
            )
            cursors["crsp_shares_interval"].consume(typed)
            share_tables.append(typed.select(("permno", "start_date", "end_date")))
        cursors["crsp_shares_interval"].finish()
        shares_index = _interval_index(share_tables, {})
        evidence.canonicalization["shares"] = {
            "canonical_rows": len(shares_index.permnos),
            "unique_permnos": shares_index.unique_permnos,
            "duplicate_key_groups": shares_index.duplicate_key_groups,
            "contiguous_edges": shares_index.contiguous_edges,
            "gap_edges": shares_index.gap_edges,
            "overlap_edges": shares_index.overlap_edges,
        }
        progress("shares_verified", {"rows": evidence.table_rows["crsp_shares_interval"]})

        daily_identity: Counter[str] = Counter()
        daily_shares: Counter[str] = Counter()
        daily_rows = 0
        for raw in _daily_canonical(settings, evidence):
            consumer, returns = _daily_tables(settings, raw)
            _update_table_evidence(
                evidence, raw, consumer,
                settings.contracts["crsp_security_day"], settings,
            )
            _update_table_evidence(
                evidence, raw, returns,
                settings.contracts["crsp_security_day_return_staging"], settings,
            )
            _update_quarantine_evidence(
                evidence, returns, "crsp_security_day_return_staging"
            )
            evidence.return_split["daily"].update(
                {
                    "raw_rows": raw.num_rows,
                    "consumer_rows": consumer.num_rows,
                    "return_staging_rows": returns.num_rows,
                }
            )
            cursors["crsp_security_day"].consume(consumer)
            cursors["crsp_security_day_return_staging"].consume(returns)
            identity_metrics, share_metrics = _join_diagnostics(raw, identity_index, shares_index, monthly=False)
            _add(daily_identity, identity_metrics)
            _add(daily_shares, share_metrics)
            daily_rows += raw.num_rows
            if daily_rows // 2_000_000 != (daily_rows - raw.num_rows) // 2_000_000:
                progress("daily_scan_progress", {"rows": daily_rows})
        cursors["crsp_security_day"].finish()
        cursors["crsp_security_day_return_staging"].finish()
        evidence.joins["daily_to_identity"] = dict(daily_identity)
        evidence.joins["daily_to_shares"] = dict(daily_shares)
        progress("daily_verified", {"rows": daily_rows})

        monthly_identity: Counter[str] = Counter()
        monthly_shares: Counter[str] = Counter()
        monthly_rows = 0
        for raw in _monthly_canonical(settings, evidence):
            consumer, returns = _monthly_tables(settings, raw)
            _update_table_evidence(
                evidence, raw, consumer,
                settings.contracts["crsp_security_month_crosscheck"], settings,
            )
            _update_table_evidence(
                evidence, raw, returns,
                settings.contracts["crsp_security_month_return_staging"], settings,
            )
            _update_quarantine_evidence(
                evidence, returns, "crsp_security_month_return_staging"
            )
            evidence.return_split["monthly"].update(
                {
                    "raw_rows": raw.num_rows,
                    "consumer_rows": consumer.num_rows,
                    "return_staging_rows": returns.num_rows,
                }
            )
            cursors["crsp_security_month_crosscheck"].consume(consumer)
            cursors["crsp_security_month_return_staging"].consume(returns)
            identity_metrics, share_metrics = _join_diagnostics(raw, identity_index, shares_index, monthly=True)
            _add(monthly_identity, identity_metrics)
            _add(monthly_shares, share_metrics)
            monthly_rows += raw.num_rows
        cursors["crsp_security_month_crosscheck"].finish()
        cursors["crsp_security_month_return_staging"].finish()
        evidence.joins["monthly_to_identity"] = dict(monthly_identity)
        evidence.joins["monthly_to_shares"] = dict(monthly_shares)
        progress("monthly_verified", {"rows": monthly_rows})

    _validate_locks(settings, evidence)
    return evidence


def _validate_locks(settings: IndependentSettings, evidence: ScanEvidence) -> None:
    policy = settings.policy
    for name, contract in settings.contracts.items():
        if evidence.table_rows[name] != contract.expected_rows:
            raise IndependentTypedScanError(f"Table row lock differs: {name}")
    for source in ("daily", "monthly"):
        actual = evidence.canonicalization[source]
        locked = policy["canonicalization"][source]
        expected = {
            "daily": {
                "duplicate_groups": "exact_duplicate_groups",
                "duplicate_surplus_rows": "duplicate_surplus_rows",
                "conflicting_groups": "conflicting_duplicate_groups",
                "canonical_rows": "output_rows",
            },
            "monthly": {
                "duplicate_groups": "exact_raw_key_duplicate_groups",
                "duplicate_surplus_rows": "exact_raw_key_duplicate_surplus_rows",
                "conflicting_groups": "conflicting_raw_key_groups",
                "candidate_groups": "effective_interval_candidate_groups",
                "canonical_rows": "output_rows",
            },
        }[source]
        for actual_name, locked_name in expected.items():
            locked_value = int(locked.get(locked_name, 0))
            if int(actual.get(actual_name, 0)) != locked_value:
                raise IndependentTypedScanError(f"{source} canonicalization differs for {actual_name}")
        if int(evidence.raw_sources[source]["retained_rows"]) != int(locked["retained_rows"]):
            raise IndependentTypedScanError(f"{source} retained-row lock differs")
    monthly = policy["canonicalization"]["monthly"]
    rows_after_collapse = (
        int(evidence.raw_sources["monthly"]["retained_rows"])
        - int(evidence.canonicalization["monthly"].get("duplicate_surplus_rows", 0))
    )
    if rows_after_collapse != int(monthly["rows_after_exact_raw_key_collapse"]):
        raise IndependentTypedScanError("monthly first-stage row conservation differs")
    expected_size = int(monthly["candidate_group_size"])
    observed_histogram = {
        int(key.removeprefix("candidate_size_")): value
        for key, value in evidence.canonicalization["monthly"].items()
        if key.startswith("candidate_size_")
    }
    if observed_histogram != {
        expected_size: int(monthly["effective_interval_candidate_groups"])
    }:
        raise IndependentTypedScanError("monthly candidate-size histogram differs")
    for source in ("identity", "shares"):
        actual = evidence.canonicalization[source]
        locked = policy["canonicalization"][source]
        mapping = {
            "canonical_rows": "output_rows",
            "unique_permnos": "unique_permnos",
            "duplicate_key_groups": "duplicate_key_groups",
            "contiguous_edges": "contiguous_edges",
            "gap_edges": "gap_edges",
            "overlap_edges": "overlap_edges",
        }
        for actual_name, locked_name in mapping.items():
            if locked_name in locked and int(actual[actual_name]) != int(locked[locked_name]):
                raise IndependentTypedScanError(
                    f"{source} interval lock differs for {actual_name}"
                )
        if int(evidence.raw_sources[source]["retained_rows"]) != int(locked["output_rows"]):
            raise IndependentTypedScanError(f"{source} retained-row lock differs")
    for join_name, actual in evidence.joins.items():
        locked = policy["joins"][join_name]
        for field in (
            "population_rows", "one_match_rows", "multiple_match_rows",
            "blank_as_of_date_rows", "zero_match_rows",
            "embedded_interval_exact_key_match_rows", "embedded_interval_exact_key_missing_rows",
            "as_of_interval_key_equal_rows", "as_of_interval_key_different_rows",
            "as_of_payload_all_fields_equal_rows", "as_of_payload_one_or_more_fields_different_rows",
            "blank_observation_date_rows", "zero_match_rows_with_nonblank_observation_date",
            "nonblank_observation_date_rows", "total_rows_without_usable_share_match",
        ):
            if field in locked and int(actual.get(field, 0)) != int(locked[field]):
                raise IndependentTypedScanError(f"Join lock differs: {join_name}.{field}")
        if join_name == "monthly_to_identity":
            nonblank = actual["population_rows"] - actual["blank_as_of_date_rows"]
            zero_nonblank = actual["zero_match_rows"] - actual["blank_as_of_date_rows"]
            if (
                nonblank != int(locked["nonblank_as_of_date_rows"])
                or zero_nonblank
                != int(locked["zero_match_rows_with_nonblank_as_of_date"])
            ):
                raise IndependentTypedScanError(
                    "monthly identity nonblank join conservation differs"
                )
    for return_table in ("crsp_security_day_return_staging", "crsp_security_month_return_staging"):
        if evidence.table_nulls[return_table]["reconciliation_status"] or evidence.table_nulls[return_table]["return_history_authorized"]:
            raise IndependentTypedScanError(f"Return quarantine guard contains nulls: {return_table}")
    for table_name, contract in settings.contracts.items():
        key = evidence.key_quality[table_name]
        if (
            key["rows"] != evidence.table_rows[table_name]
            or key["null_rows"]
            or key["duplicate_rows"]
            or key["sorted_violations"]
        ):
            raise IndependentTypedScanError(
                f"Primary-key quality differs: {table_name}"
            )
        for field in contract.fields:
            name = field["name"]
            if (
                evidence.raw_expected_nulls[table_name][name]
                != evidence.table_nulls[table_name][name]
            ):
                raise IndependentTypedScanError(
                    f"Raw-to-typed null conservation differs: {table_name}.{name}"
                )
    for table_name, columns in evidence.decimal_roundtrip.items():
        for column, counts in columns.items():
            if (
                counts["mismatch_rows"]
                or counts["grammar_failure_rows"]
                or counts["exact_match_rows"] != counts["source_nonblank_rows"]
            ):
                raise IndependentTypedScanError(
                    f"Exact decimal evidence differs: {table_name}.{column}"
                )
    for formula, counts in evidence.formula_quality.items():
        if (
            counts["mismatch_rows"]
            or counts["expected_null_rows"] != counts["actual_null_rows"]
            or counts["match_rows"] != counts["rows"]
        ):
            raise IndependentTypedScanError(
                f"Nominal unit formula differs: {formula}"
            )
    split_names = {
        "daily": (
            "crsp_security_day", "crsp_security_day_return_staging"
        ),
        "monthly": (
            "crsp_security_month_crosscheck",
            "crsp_security_month_return_staging",
        ),
    }
    split_lock = policy["canonicalization"]["return_split"]
    for frequency, (consumer, staging) in split_names.items():
        expected_rows = int(split_lock[f"{frequency}_consumer_rows"])
        if any(
            evidence.return_split[frequency][name] != expected_rows
            for name in ("raw_rows", "consumer_rows", "return_staging_rows")
        ) or evidence.key_sha256[consumer] != evidence.key_sha256[staging]:
            raise IndependentTypedScanError(
                f"{frequency} return split conservation differs"
            )
    for table_name in (
        "crsp_security_day_return_staging",
        "crsp_security_month_return_staging",
    ):
        guards = evidence.quarantine_guards[table_name]
        if guards["rows"] != evidence.table_rows[table_name] or any(
            guards[name]
            for name in (
                "status_not_unreconciled_rows", "authorized_true_rows",
                "status_null_rows", "authorized_null_rows",
            )
        ):
            raise IndependentTypedScanError(
                f"Return quarantine guard differs: {table_name}"
            )
    if any(
        value["prohibited_column_count"]
        for value in evidence.consumer_return_absence.values()
    ):
        raise IndependentTypedScanError("A consumer table contains return-family columns")
    for table_name in (
        "crsp_security_day", "crsp_security_day_return_staging",
        "crsp_security_month_crosscheck",
        "crsp_security_month_return_staging",
    ):
        scope = evidence.date_scope[table_name]
        if (
            scope["rows"] != evidence.table_rows[table_name]
            or scope["before_scope_rows"]
            or scope["after_scope_rows"]
        ):
            raise IndependentTypedScanError(
                f"Date-scope evidence differs: {table_name}"
            )


__all__ = [
    "IndependentSettings", "IndependentTypedScanError", "ScanEvidence",
    "TABLE_FILES", "inspect_parquet", "load_independent_settings", "scan_and_compare",
]
