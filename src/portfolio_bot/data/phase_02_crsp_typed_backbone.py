"""Primary compiler for the locked Phase 2 typed CRSP backbone.

The compiler is deliberately limited to the development/December-2014 quality
slice.  It authenticates the frozen sources, gates rows by date before Python
payload inspection, canonicalizes the already-sorted CRSP streams, writes the
six locked Arrow tables, and returns aggregate-only build evidence.  Publishing
and independent verification live in separate modules.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import hashlib
import tomllib
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data import phase_02_crsp_typed_backbone_core as core
from portfolio_bot.data import phase_02_crsp_typed_backbone_discovery as discovery


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_typed_backbone_policy_v1.toml"
)
DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_typed_backbone_schema_v1.json"
)
DEFAULT_SUPPORTING_PATH = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_supporting_sources_development_v1.toml"
)
LOCKED_POLICY_SHA256 = (
    "23ec145f587c95b735f675c5ee665d43d506ad2d91d0ae6a46a3942ed5674a89"
)
LOCKED_SCHEMA_SHA256 = (
    "62ad603c88798212f60ea4ec9796bffbf65008c6cfd87a08239cddf057e31391"
)
LOCKED_SUPPORTING_SHA256 = (
    "6117495f76106a9a89922c1d2a01d23c824e9e94f783ffd8a8290a99bec11cf1"
)
SOURCE_NAMES = ("identity", "shares", "daily", "monthly")
TABLE_NAMES = (
    "crsp_identity_interval",
    "crsp_security_day",
    "crsp_security_day_return_staging",
    "crsp_security_month_crosscheck",
    "crsp_security_month_return_staging",
    "crsp_shares_interval",
)
PARQUET_FILENAMES = {name: f"{name}.parquet" for name in TABLE_NAMES}
ROW_GROUP_SIZE = 262_144
DEFAULT_BLOCK_SIZE_BYTES = 16 << 20
_EPOCH = dt.date(1970, 1, 1)


class TypedBackboneBuildError(RuntimeError):
    """Raised when a typed build cannot prove the frozen contract."""


@dataclasses.dataclass(frozen=True)
class TypedBackboneSettings:
    project_root: Path
    database_root: Path
    raw_root: Path
    interim_root: Path
    quality_root: Path
    artifact_root: Path
    policy_path: Path
    schema_path: Path
    policy: Mapping[str, Any]
    registry: core.SchemaRegistry
    sources: Mapping[str, discovery.SourceSpec]
    start_date: dt.date
    end_date: dt.date
    block_size_bytes: int = DEFAULT_BLOCK_SIZE_BYTES


@dataclasses.dataclass
class CanonicalizationMetrics:
    full_rows: int = 0
    retained_rows: int = 0
    canonical_rows: int = 0
    duplicate_groups: int = 0
    duplicate_surplus_rows: int = 0
    conflicting_groups: int = 0
    candidate_groups: int = 0
    candidate_size_histogram: Counter[int] = dataclasses.field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_groups": self.candidate_groups,
            "candidate_size_histogram": {
                str(key): self.candidate_size_histogram[key]
                for key in sorted(self.candidate_size_histogram)
            },
            "canonical_rows": self.canonical_rows,
            "conflicting_groups": self.conflicting_groups,
            "duplicate_groups": self.duplicate_groups,
            "duplicate_surplus_rows": self.duplicate_surplus_rows,
            "full_rows": self.full_rows,
            "retained_rows": self.retained_rows,
        }


@dataclasses.dataclass
class JoinMetrics:
    population_rows: int = 0
    blank_as_of_date_rows: int = 0
    one_match_rows: int = 0
    zero_match_rows: int = 0
    multiple_match_rows: int = 0
    embedded_interval_exact_key_match_rows: int = 0
    embedded_interval_exact_key_missing_rows: int = 0
    as_of_interval_key_equal_rows: int = 0
    as_of_interval_key_different_rows: int = 0
    as_of_payload_all_fields_equal_rows: int = 0
    as_of_payload_one_or_more_fields_different_rows: int = 0

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class BuildEvidence:
    canonicalization: dict[str, CanonicalizationMetrics]
    table_rows: Counter[str] = dataclasses.field(default_factory=Counter)
    table_nulls: dict[str, Counter[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(Counter)
    )
    key_sha256: dict[str, str] = dataclasses.field(default_factory=dict)
    joins: dict[str, Mapping[str, int]] = dataclasses.field(default_factory=dict)
    parquet: dict[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)
    interval_structure: dict[str, Mapping[str, int]] = dataclasses.field(
        default_factory=dict
    )
    key_quality: dict[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)
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
    consumer_return_absence: dict[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    date_scope: dict[str, Counter[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(Counter)
    )


@dataclasses.dataclass
class KeyQualityTracker:
    """Independently prove typed primary-key order, uniqueness, and scope."""

    spec: core.TableSpec
    start_date: dt.date
    end_date: dt.date
    rows: int = 0
    null_rows: int = 0
    duplicate_rows: int = 0
    sorted_violations: int = 0
    _last_key: tuple[int, ...] | None = None

    def add(self, table: pa.Table) -> None:
        arrays: list[np.ndarray] = []
        null_row = np.zeros(table.num_rows, dtype=bool)
        for name in self.spec.primary_key:
            column = table[name].combine_chunks()
            if column.null_count:
                null_row |= np.asarray(
                    pc.is_null(column).to_numpy(zero_copy_only=False), dtype=bool
                )
                fill = (
                    pa.scalar(0, column.type)
                    if pa.types.is_integer(column.type)
                    else pa.scalar(_EPOCH, column.type)
                )
                column = pc.fill_null(column, fill)
            if pa.types.is_integer(column.type):
                arrays.append(
                    np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64)
                )
            elif pa.types.is_date32(column.type):
                arrays.append(
                    np.asarray(column.to_numpy(zero_copy_only=False))
                    .astype("datetime64[D]")
                    .astype(np.int64)
                )
            else:
                raise TypedBackboneBuildError(
                    f"Unsupported primary-key type in {self.spec.name}.{name}"
                )
        self.rows += table.num_rows
        self.null_rows += int(np.count_nonzero(null_row))
        if not arrays or table.num_rows == 0:
            return
        equal_previous = _equal_previous(arrays)
        self.duplicate_rows += int(np.count_nonzero(equal_previous))
        self.sorted_violations += _lexicographic_violation_count(arrays)
        first_key = tuple(int(array[0]) for array in arrays)
        if self._last_key is not None:
            if first_key == self._last_key:
                self.duplicate_rows += 1
            elif first_key < self._last_key:
                self.sorted_violations += 1
        self._last_key = tuple(int(array[-1]) for array in arrays)

    def as_dict(self) -> dict[str, Any]:
        return {
            "duplicate_rows": self.duplicate_rows,
            "null_rows": self.null_rows,
            "primary_key": list(self.spec.primary_key),
            "rows": self.rows,
            "sorted_violations": self.sorted_violations,
        }


@dataclasses.dataclass(frozen=True)
class IntervalIndex:
    """Compact nonoverlapping interval index used only for aggregate joins."""

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

    def match(
        self, permnos: np.ndarray, dates: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        if not (len(permnos) == len(dates) == len(valid)):
            raise TypedBackboneBuildError("Interval-match arrays have unequal lengths")
        result = np.full(len(permnos), -1, dtype=np.int64)
        if not len(permnos):
            return result
        if bool(np.all(permnos[1:] >= permnos[:-1])):
            order = np.arange(len(permnos), dtype=np.int64)
            ordered_permnos = permnos
        else:
            order = np.argsort(permnos, kind="stable")
            ordered_permnos = permnos[order]
        group_starts = np.r_[
            0, np.flatnonzero(ordered_permnos[1:] != ordered_permnos[:-1]) + 1
        ]
        group_stops = np.r_[group_starts[1:], len(ordered_permnos)]
        for first, stop in zip(group_starts, group_stops, strict=True):
            target = order[first:stop]
            target = target[valid[target]]
            if not target.size:
                continue
            permno = int(ordered_permnos[first])
            bounds = self.slices.get(permno)
            if bounds is None:
                continue
            interval_first, interval_stop = bounds
            local_starts = self.starts[interval_first:interval_stop]
            local_ends = self.ends[interval_first:interval_stop]
            positions = np.searchsorted(local_starts, dates[target], side="right") - 1
            usable = positions >= 0
            safe = np.maximum(positions, 0)
            usable &= dates[target] <= local_ends[safe]
            result[target[usable]] = interval_first + positions[usable]
        return result


class IntervalIndexBuilder:
    def __init__(self, payload_fields: Sequence[str] = ()) -> None:
        self._permnos: list[np.ndarray] = []
        self._starts: list[np.ndarray] = []
        self._ends: list[np.ndarray] = []
        self._payload: dict[str, list[pa.Array]] = {
            field: [] for field in payload_fields
        }

    def add(
        self,
        raw: pa.RecordBatch,
        *,
        start_column: str,
        end_column: str,
    ) -> None:
        permnos, starts, ends = _typed_key_arrays(
            raw, ("PERMNO", start_column, end_column)
        )
        self._permnos.append(permnos)
        self._starts.append(starts)
        self._ends.append(ends)
        for field in self._payload:
            self._payload[field].append(raw.column(raw.schema.get_field_index(field)))

    def finish(self, label: str) -> IntervalIndex:
        if not self._permnos:
            raise TypedBackboneBuildError(f"{label} interval index is empty")
        permnos = np.concatenate(self._permnos)
        starts = np.concatenate(self._starts)
        ends = np.concatenate(self._ends)
        _assert_sorted((permnos, starts, ends), f"{label} interval stream")
        same_permno = permnos[1:] == permnos[:-1]
        same_key = same_permno & (starts[1:] == starts[:-1]) & (ends[1:] == ends[:-1])
        duplicate_groups = int(np.count_nonzero(same_key))
        overlap = same_permno & (starts[1:] <= ends[:-1])
        contiguous = same_permno & (starts[1:] == ends[:-1] + 1)
        gap = same_permno & (starts[1:] > ends[:-1] + 1)
        slices: dict[int, tuple[int, int]] = {}
        group_starts = np.r_[0, np.flatnonzero(~same_permno) + 1]
        group_stops = np.r_[group_starts[1:], len(permnos)]
        for first, stop in zip(group_starts, group_stops, strict=True):
            slices[int(permnos[first])] = (int(first), int(stop))
        payload = {
            field: pa.chunked_array(chunks, type=pa.string()).combine_chunks()
            for field, chunks in self._payload.items()
        }
        return IntervalIndex(
            permnos=permnos,
            starts=starts,
            ends=ends,
            slices=slices,
            payload=payload,
            unique_permnos=len(slices),
            duplicate_key_groups=duplicate_groups,
            contiguous_edges=int(np.count_nonzero(contiguous)),
            gap_edges=int(np.count_nonzero(gap)),
            overlap_edges=int(np.count_nonzero(overlap)),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _authenticated_toml(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TypedBackboneBuildError(f"Cannot read {label}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise TypedBackboneBuildError(f"{label} SHA-256 differs")
    try:
        return tomllib.loads(payload.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise TypedBackboneBuildError(f"Cannot parse {label}") from exc


def load_production_settings(
    *, block_size_bytes: int = DEFAULT_BLOCK_SIZE_BYTES
) -> TypedBackboneSettings:
    """Authenticate the Stage 1 lock and reconstruct production build settings."""

    if block_size_bytes < discovery._MIN_ARROW_BLOCK_SIZE:
        raise TypedBackboneBuildError("Arrow block size is below the safe minimum")
    locked = discovery.load_production_settings(block_size_bytes=block_size_bytes)
    policy = _authenticated_toml(
        DEFAULT_POLICY_PATH, LOCKED_POLICY_SHA256, "typed-backbone policy"
    )
    registry = core.load_schema_registry(DEFAULT_SCHEMA_PATH)
    if registry.source_sha256 != LOCKED_SCHEMA_SHA256:
        raise TypedBackboneBuildError("Typed-backbone schema SHA-256 differs")
    supporting = _authenticated_toml(
        DEFAULT_SUPPORTING_PATH,
        LOCKED_SUPPORTING_SHA256,
        "supporting-source profile",
    )
    supporting_by_name = {
        str(source["filename"]): source for source in supporting["sources"]
    }
    shares_config = supporting_by_name["crsp_shares_outstanding_ciz.csv"]
    shares_expected = supporting["expected"]["crsp_shares_outstanding_ciz_csv"]
    shares = discovery.SourceSpec(
        name="shares",
        filename=str(shares_config["filename"]),
        header=tuple(str(value) for value in shares_config["header"]),
        sha256=str(shares_config["sha256"]),
        size_bytes=int(shares_config["size_bytes"]),
        full_file_rows=int(shares_config["full_file_rows"]),
        retained_rows=int(shares_expected["retained_rows"]),
        canonical_rows=int(shares_expected["retained_rows"]),
        duplicate_groups=int(shares_expected["duplicate_key_groups"]),
        duplicate_surplus_rows=int(shares_expected["duplicate_surplus_rows"]),
    )
    sources = {
        "daily": locked.sources["daily"],
        "monthly": locked.sources["monthly"],
        "identity": locked.sources["names"],
        "shares": shares,
    }
    database_root = locked.raw_root.parent
    return TypedBackboneSettings(
        project_root=PROJECT_ROOT,
        database_root=database_root,
        raw_root=locked.raw_root,
        interim_root=database_root / "interim",
        quality_root=database_root / "quality",
        artifact_root=PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel",
        policy_path=DEFAULT_POLICY_PATH,
        schema_path=DEFAULT_SCHEMA_PATH,
        policy=policy,
        registry=registry,
        sources=sources,
        start_date=locked.start_date,
        end_date=locked.end_date,
        block_size_bytes=block_size_bytes,
    )


def _combine_batches(batches: Sequence[pa.RecordBatch]) -> pa.RecordBatch:
    if not batches:
        raise TypedBackboneBuildError("Cannot combine an empty batch sequence")
    if len(batches) == 1:
        return batches[0]
    table = pa.Table.from_batches(list(batches)).combine_chunks()
    result = table.to_batches(max_chunksize=table.num_rows)
    if len(result) != 1:
        raise TypedBackboneBuildError("Cannot combine canonical raw batches")
    return result[0]


def _typed_key_arrays(
    batch: pa.RecordBatch, key_columns: Sequence[str]
) -> tuple[np.ndarray, ...]:
    arrays: list[np.ndarray] = []
    for name in key_columns:
        column = batch.column(batch.schema.get_field_index(name))
        if name == "PERMNO":
            arrays.append(discovery._positive_permnos(column, "typed build PERMNO"))
        else:
            parsed = discovery._strict_date32(column, f"typed build {name}")
            arrays.append(discovery._date_days(parsed))
    return tuple(arrays)


def _equal_previous(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise TypedBackboneBuildError("Canonical key cannot be empty")
    length = len(arrays[0])
    if any(len(array) != length for array in arrays):
        raise TypedBackboneBuildError("Canonical key arrays have unequal lengths")
    equal = np.zeros(length, dtype=bool)
    if length > 1:
        adjacent = np.ones(length - 1, dtype=bool)
        for array in arrays:
            adjacent &= array[1:] == array[:-1]
        equal[1:] = adjacent
    return equal


def _assert_sorted(arrays: Sequence[np.ndarray], label: str) -> None:
    if not arrays or len(arrays[0]) < 2:
        return
    undecided = np.ones(len(arrays[0]) - 1, dtype=bool)
    violation = np.zeros(len(arrays[0]) - 1, dtype=bool)
    for array in arrays:
        left = array[:-1]
        right = array[1:]
        violation |= undecided & (right < left)
        undecided &= right == left
    if bool(violation.any()):
        raise TypedBackboneBuildError(f"{label} is not sorted by its locked key")


def _lexicographic_violation_count(arrays: Sequence[np.ndarray]) -> int:
    if not arrays or len(arrays[0]) < 2:
        return 0
    undecided = np.ones(len(arrays[0]) - 1, dtype=bool)
    violation = np.zeros(len(arrays[0]) - 1, dtype=bool)
    for array in arrays:
        left = array[:-1]
        right = array[1:]
        violation |= undecided & (right < left)
        undecided &= right == left
    return int(np.count_nonzero(violation))


def _split_final_group(
    batch: pa.RecordBatch, key_columns: Sequence[str], label: str
) -> tuple[pa.RecordBatch, pa.RecordBatch]:
    if batch.num_rows == 0:
        return batch, batch
    arrays = _typed_key_arrays(batch, key_columns)
    _assert_sorted(arrays, label)
    equal_previous = _equal_previous(arrays)
    starts = np.flatnonzero(~equal_previous)
    final_start = int(starts[-1])
    return batch.slice(0, final_start), batch.slice(final_start)


def _rows_equal_at(batch: pa.RecordBatch, left: int, right: int) -> bool:
    for column in batch.columns:
        if column[left].as_py() != column[right].as_py():
            return False
    return True


def _collapse_exact_groups(
    batch: pa.RecordBatch,
    *,
    key_columns: Sequence[str],
    metrics: CanonicalizationMetrics,
    label: str,
) -> pa.RecordBatch:
    if batch.num_rows == 0:
        return batch
    arrays = _typed_key_arrays(batch, key_columns)
    _assert_sorted(arrays, label)
    duplicate = _equal_previous(arrays)
    duplicate_positions = np.flatnonzero(duplicate)
    if duplicate_positions.size:
        group_starts = int(np.count_nonzero(duplicate & ~np.r_[False, duplicate[:-1]]))
        metrics.duplicate_groups += group_starts
        metrics.duplicate_surplus_rows += int(duplicate_positions.size)
        for position in duplicate_positions.tolist():
            if not _rows_equal_at(batch, position - 1, position):
                metrics.conflicting_groups += 1
                raise TypedBackboneBuildError(
                    f"{label} contains a conflicting duplicate key group"
                )
    keep = pa.array(~duplicate, type=pa.bool_())
    return batch.filter(keep)


def _iter_complete_groups(
    batches: Iterable[pa.RecordBatch],
    *,
    key_columns: Sequence[str],
    label: str,
) -> Iterator[pa.RecordBatch]:
    carry: pa.RecordBatch | None = None
    for incoming in batches:
        if incoming.num_rows == 0:
            continue
        combined = incoming if carry is None else _combine_batches((carry, incoming))
        complete, carry = _split_final_group(combined, key_columns, label)
        if complete.num_rows:
            yield complete
    if carry is not None and carry.num_rows:
        _typed = _typed_key_arrays(carry, key_columns)
        _assert_sorted(_typed, label)
        yield carry


def canonicalize_daily_batches(
    batches: Iterable[pa.RecordBatch], metrics: CanonicalizationMetrics
) -> Iterator[pa.RecordBatch]:
    """Collapse only all-field-exact adjacent daily key duplicates."""

    for complete in _iter_complete_groups(
        batches,
        key_columns=("PERMNO", "DlyCalDt"),
        label="daily retained stream",
    ):
        canonical = _collapse_exact_groups(
            complete,
            key_columns=("PERMNO", "DlyCalDt"),
            metrics=metrics,
            label="daily retained stream",
        )
        metrics.canonical_rows += canonical.num_rows
        if canonical.num_rows:
            yield canonical


def _optional_date(value: str, label: str) -> dt.date | None:
    if value == "":
        return None
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise TypedBackboneBuildError(f"{label} contains an invalid date") from exc
    if parsed.isoformat() != value:
        raise TypedBackboneBuildError(f"{label} contains a noncanonical date")
    return parsed


def _select_monthly_candidates(
    batch: pa.RecordBatch, metrics: CanonicalizationMetrics
) -> pa.RecordBatch:
    if batch.num_rows == 0:
        return batch
    arrays = _typed_key_arrays(batch, ("PERMNO", "MthCalDt"))
    _assert_sorted(arrays, "monthly canonical stream")
    duplicate = _equal_previous(arrays)
    starts = np.flatnonzero(~duplicate).tolist()
    starts.append(batch.num_rows)
    selected: list[int] = []
    names = {name: batch.schema.get_field_index(name) for name in batch.schema.names}
    for group_number in range(len(starts) - 1):
        start = starts[group_number]
        stop = starts[group_number + 1]
        size = stop - start
        if size == 1:
            selected.append(start)
            continue
        metrics.candidate_groups += 1
        metrics.candidate_size_histogram[size] += 1
        price_matches: list[int] = []
        calendar_matches: list[int] = []
        for position in range(start, stop):
            interval_start = dt.date.fromisoformat(
                batch.column(names["SecInfoStartDt"])[position].as_py()
            )
            interval_end = dt.date.fromisoformat(
                batch.column(names["SecInfoEndDt"])[position].as_py()
            )
            if interval_start > interval_end:
                raise TypedBackboneBuildError("Monthly candidate interval is inverted")
            price_date = _optional_date(
                batch.column(names["MthPrcDt"])[position].as_py(),
                "monthly price date",
            )
            calendar_date = dt.date.fromisoformat(
                batch.column(names["MthCalDt"])[position].as_py()
            )
            if price_date is not None and interval_start <= price_date <= interval_end:
                price_matches.append(position)
            if interval_start <= calendar_date <= interval_end:
                calendar_matches.append(position)
        if len(price_matches) != 1 or len(calendar_matches) != 1:
            raise TypedBackboneBuildError(
                "Monthly candidate group lacks unique interval selection"
            )
        if price_matches[0] != calendar_matches[0]:
            raise TypedBackboneBuildError(
                "Monthly price and calendar dates select different intervals"
            )
        selected.append(price_matches[0])
    return batch.take(pa.array(selected, type=pa.int64()))


def canonicalize_monthly_batches(
    batches: Iterable[pa.RecordBatch], metrics: CanonicalizationMetrics
) -> Iterator[pa.RecordBatch]:
    """Apply the exact raw-key collapse and effective-interval selection."""

    for complete in _iter_complete_groups(
        batches,
        key_columns=("PERMNO", "MthCalDt"),
        label="monthly retained stream",
    ):
        raw_distinct = _collapse_exact_groups(
            complete,
            key_columns=(
                "PERMNO",
                "MthCalDt",
                "SecInfoStartDt",
                "SecInfoEndDt",
            ),
            metrics=metrics,
            label="monthly raw-key stream",
        )
        canonical = _select_monthly_candidates(raw_distinct, metrics)
        metrics.canonical_rows += canonical.num_rows
        if canonical.num_rows:
            yield canonical


def _scoped_market_batches(
    source: discovery._AuthenticatedSource,
    *,
    date_column: str,
    settings: TypedBackboneSettings,
    metrics: CanonicalizationMetrics,
) -> Iterator[pa.RecordBatch]:
    with source.batches(source.spec.header, settings.block_size_bytes) as batches:
        for batch in batches:
            metrics.full_rows += batch.num_rows
            retained, _ = discovery._scope_by_date(
                batch,
                date_column,
                settings.start_date,
                settings.end_date,
                source.spec.name,
            )
            metrics.retained_rows += retained.num_rows
            if retained.num_rows:
                # Payload decoding and transformations occur only after this gate.
                yield retained


def _scoped_interval_batches(
    source: discovery._AuthenticatedSource,
    *,
    start_column: str,
    end_column: str,
    settings: TypedBackboneSettings,
    metrics: CanonicalizationMetrics,
) -> Iterator[pa.RecordBatch]:
    with source.batches(source.spec.header, settings.block_size_bytes) as batches:
        for batch in batches:
            metrics.full_rows += batch.num_rows
            starts = discovery._strict_date32(
                batch.column(batch.schema.get_field_index(start_column)),
                f"{source.spec.name} interval start",
            )
            ends = discovery._strict_date32(
                batch.column(batch.schema.get_field_index(end_column)),
                f"{source.spec.name} interval end",
            )
            if bool(pc.any(pc.greater(starts, ends)).as_py()):
                raise TypedBackboneBuildError(
                    f"{source.spec.name} contains an inverted interval"
                )
            mask = pc.and_(
                pc.greater_equal(ends, pa.scalar(settings.start_date, pa.date32())),
                pc.less_equal(starts, pa.scalar(settings.end_date, pa.date32())),
            )
            retained = batch.filter(mask)
            metrics.retained_rows += retained.num_rows
            if retained.num_rows:
                yield retained


def _sorted_interval_batches(
    batches: Iterable[pa.RecordBatch],
    *,
    start_column: str,
    end_column: str,
    label: str,
) -> Iterator[pa.RecordBatch]:
    """Materialize and deterministically sort a bounded interval dimension.

    The supporting-source contract locks the output order but does not claim
    that the downloaded shares file is already globally ordered.  Identity and
    shares are small enough to sort in memory after the date gate; market fact
    streams remain bounded and streaming.
    """

    retained = list(batches)
    if not retained:
        return
    table = pa.Table.from_batches(retained)
    permnos = discovery._positive_permnos(
        table["PERMNO"].combine_chunks(), f"{label} PERMNO sort key"
    )
    starts = discovery._date_days(
        discovery._strict_date32(
            table[start_column].combine_chunks(), f"{label} start sort key"
        )
    )
    ends = discovery._date_days(
        discovery._strict_date32(
            table[end_column].combine_chunks(), f"{label} end sort key"
        )
    )
    order = np.lexsort((ends, starts, permnos))
    sorted_table = table.take(pa.array(order, type=pa.int64()))
    sorted_arrays = (
        permnos[order],
        starts[order],
        ends[order],
    )
    _assert_sorted(sorted_arrays, f"{label} sorted interval stream")
    yield from sorted_table.to_batches(max_chunksize=ROW_GROUP_SIZE)


class ParquetSink:
    """Bounded deterministic Parquet writer with exact locked row groups."""

    def __init__(self, stream: Any, spec: core.TableSpec) -> None:
        self.spec = spec
        self._writer = pq.ParquetWriter(
            stream,
            spec.arrow_schema(),
            version="2.6",
            compression="zstd",
            compression_level=9,
            use_dictionary=True,
            write_statistics=True,
            data_page_version="2.0",
            store_schema=True,
        )
        self._pending: list[pa.Table] = []
        self._pending_rows = 0
        self.row_count = 0
        self.null_counts: Counter[str] = Counter()
        self.closed = False

    def add(self, table: pa.Table) -> None:
        if self.closed:
            raise TypedBackboneBuildError("Cannot add to a closed Parquet sink")
        if not table.schema.equals(self.spec.arrow_schema(), check_metadata=True):
            raise TypedBackboneBuildError(f"Typed schema differs for {self.spec.name}")
        for name, column in zip(table.schema.names, table.columns, strict=True):
            self.null_counts[name] += column.null_count
        self.row_count += table.num_rows
        offset = 0
        while offset < table.num_rows:
            needed = ROW_GROUP_SIZE - self._pending_rows
            piece = table.slice(offset, min(needed, table.num_rows - offset))
            self._pending.append(piece)
            self._pending_rows += piece.num_rows
            offset += piece.num_rows
            if self._pending_rows == ROW_GROUP_SIZE:
                self._flush()

    def _flush(self) -> None:
        if not self._pending_rows:
            return
        table = pa.concat_tables(self._pending).combine_chunks()
        self._writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
        self._pending.clear()
        self._pending_rows = 0

    def close(self) -> None:
        if self.closed:
            return
        self._flush()
        self._writer.close()
        self.closed = True

    def __enter__(self) -> "ParquetSink":
        return self

    def __exit__(self, *_: Any) -> bool:
        self.close()
        return False


def _update_key_digest(digest: Any, table: pa.Table, key: Sequence[str]) -> None:
    """Hash fixed-width typed keys efficiently in declared row order."""

    dtypes: list[tuple[str, str]] = []
    values_by_name: list[np.ndarray] = []
    for index, name in enumerate(key):
        column = table[name].combine_chunks()
        if pa.types.is_int64(column.type):
            dtype = ">i8"
            values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype)
        elif pa.types.is_date32(column.type):
            dtype = ">i4"
            values = (
                np.asarray(column.to_numpy(zero_copy_only=False))
                .astype("datetime64[D]")
                .astype(dtype)
            )
        else:
            raise TypedBackboneBuildError(
                f"Unsupported primary-key type in {name}: {column.type}"
            )
        dtypes.append((f"f{index}", dtype))
        values_by_name.append(values)
    packed = np.empty(table.num_rows, dtype=np.dtype(dtypes, align=False))
    for index, values in enumerate(values_by_name):
        packed[f"f{index}"] = values
    digest.update(packed.tobytes(order="C"))


_DECIMAL_LEXEME = r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?$"
_DURATION_CODES = pa.array(
    (
        "D1",
        "D2",
        "D3",
        "D4",
        "DU",
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "P9",
        "MR",
        "DD",
    ),
    type=pa.string(),
)


def _true_count(mask: pa.Array | pa.ChunkedArray) -> int:
    return int(pc.sum(pc.cast(pc.fill_null(mask, False), pa.int64())).as_py() or 0)


def _raw_column(raw: pa.RecordBatch, name: str) -> pa.Array:
    index = raw.schema.get_field_index(name)
    if index < 0:
        raise TypedBackboneBuildError(f"Raw evidence column {name!r} is absent")
    return raw.column(index)


def _expected_null_mask(raw: pa.RecordBatch, field: core.FieldSpec) -> pa.Array:
    if not field.nullable:
        return pa.array(np.zeros(raw.num_rows, dtype=bool), type=pa.bool_())
    source = field.metadata.get("source_column")
    if field.name == "mapped_trading_periods":
        return pc.is_null(
            pc.index_in(
                _raw_column(raw, "DlyRetDurFlg"), value_set=_DURATION_CODES
            )
        )
    if field.name == "screening_dollar_volume_usd":
        close = _raw_column(raw, "DlyClose")
        volume = _raw_column(raw, "DlyVol")
        mantissa = pc.replace_substring_regex(close, pattern=r"[Ee].*$", replacement="")
        digits = pc.replace_substring_regex(
            pc.replace_substring_regex(mantissa, pattern=r"^[+-]", replacement=""),
            pattern=r"\.",
            replacement="",
        )
        significant = pc.replace_substring_regex(digits, pattern=r"^0+", replacement="")
        close_not_positive = pc.or_(
            pc.starts_with(mantissa, pattern="-"),
            pc.equal(pc.utf8_length(significant), 0),
        )
        return pc.or_(
            pc.or_(pc.equal(close, ""), pc.equal(volume, "")),
            close_not_positive,
        )
    if isinstance(source, str):
        return pc.equal(_raw_column(raw, source), "")
    if isinstance(source, list) and source:
        masks = [pc.equal(_raw_column(raw, name), "") for name in source]
        result = masks[0]
        for mask in masks[1:]:
            result = pc.or_(result, mask)
        return result
    return pa.array(np.zeros(raw.num_rows, dtype=bool), type=pa.bool_())


def _update_null_conservation(
    evidence: BuildEvidence,
    raw: pa.RecordBatch,
    table: pa.Table,
    spec: core.TableSpec,
) -> None:
    for field in spec.fields:
        expected = _true_count(_expected_null_mask(raw, field))
        evidence.raw_expected_nulls[spec.name][field.name] += expected


def _decimal_roundtrip_counts(
    raw_values: pa.Array,
    typed_values: pa.ChunkedArray,
) -> Counter[str]:
    """Compare source lexemes with typed decimals using an independent parser.

    Arrow performs the common <=38-coefficient-digit path with truncation
    disabled.  Rare longer coefficients use Python's arbitrary-precision
    Decimal solely for equality, avoiding Arrow's documented decimal128 wrap
    edge while preserving exact numerical semantics.
    """

    typed = typed_values.combine_chunks()
    blank = pc.equal(raw_values, "")
    nonblank = pc.invert(blank)
    grammar = pc.match_substring_regex(raw_values, _DECIMAL_LEXEME)
    grammar_failures = _true_count(pc.and_(nonblank, pc.invert(grammar)))
    coefficient = pc.replace_substring_regex(
        pc.replace_substring_regex(
            pc.replace_substring_regex(raw_values, pattern=r"[Ee].*$", replacement=""),
            pattern=r"^[+-]",
            replacement="",
        ),
        pattern=r"\.",
        replacement="",
    )
    significant = pc.replace_substring_regex(
        coefficient, pattern=r"^0+", replacement=""
    )
    long_coefficient = pc.and_(
        nonblank,
        pc.greater(pc.utf8_length(significant), pa.scalar(38, pa.int32())),
    )
    ordinary = pc.and_(nonblank, pc.invert(long_coefficient))
    prepared = pc.if_else(
        blank,
        pa.scalar(None, pa.string()),
        pc.if_else(long_coefficient, pa.scalar("0", pa.string()), raw_values),
    )
    try:
        expected = pc.cast(
            prepared,
            options=pc.CastOptions(
                target_type=typed.type,
                allow_decimal_truncate=False,
            ),
        )
    except pa.ArrowException as exc:
        raise TypedBackboneBuildError(
            "Independent exact-decimal roundtrip parsing failed"
        ) from exc
    ordinary_mismatch = _true_count(
        pc.and_(ordinary, pc.invert(pc.fill_null(pc.equal(expected, typed), False)))
    )
    fallback_rows = 0
    fallback_mismatches = 0
    for position in pc.indices_nonzero(long_coefficient).to_pylist():
        fallback_rows += 1
        try:
            source_value = Decimal(raw_values[position].as_py())
        except (InvalidOperation, ValueError):
            fallback_mismatches += 1
            continue
        typed_value = typed[position].as_py()
        if typed_value is None or source_value != typed_value:
            fallback_mismatches += 1
    nonblank_rows = _true_count(nonblank)
    mismatch_rows = ordinary_mismatch + fallback_mismatches + grammar_failures
    return Counter(
        {
            "arbitrary_precision_fallback_rows": fallback_rows,
            "exact_match_rows": nonblank_rows - mismatch_rows,
            "grammar_failure_rows": grammar_failures,
            "mismatch_rows": mismatch_rows,
            "source_nonblank_rows": nonblank_rows,
        }
    )


def _update_decimal_roundtrip(
    evidence: BuildEvidence,
    raw: pa.RecordBatch,
    table: pa.Table,
    spec: core.TableSpec,
) -> None:
    for field in spec.fields:
        if not field.logical_type.startswith("decimal128"):
            continue
        if field.metadata.get("transform") != "exact_decimal":
            continue
        source = field.metadata.get("source_column")
        if not isinstance(source, str):
            continue
        counts = _decimal_roundtrip_counts(_raw_column(raw, source), table[field.name])
        evidence.decimal_roundtrip[spec.name][field.name].update(counts)


def _record_formula(
    target: Counter[str],
    expected: pa.Array | pa.ChunkedArray,
    actual: pa.Array | pa.ChunkedArray,
) -> None:
    both_null = pc.and_(pc.is_null(expected), pc.is_null(actual))
    equal = pc.fill_null(pc.equal(expected, actual), False)
    matches = pc.or_(both_null, equal)
    rows = len(expected)
    mismatch_rows = rows - _true_count(matches)
    target.update(
        {
            "actual_null_rows": int(actual.null_count),
            "expected_null_rows": int(expected.null_count),
            "match_rows": rows - mismatch_rows,
            "mismatch_rows": mismatch_rows,
            "rows": rows,
        }
    )


def _update_formula_quality(
    evidence: BuildEvidence, table: pa.Table, spec: core.TableSpec
) -> None:
    if spec.name in {"crsp_security_day", "crsp_security_month_crosscheck"}:
        source = table["market_cap_thousand_usd"].combine_chunks()
        actual = table["market_cap_usd"].combine_chunks()
        expected = pc.multiply_checked(
            pc.cast(
                source,
                pa.decimal256(source.type.precision, source.type.scale),
                safe=True,
            ),
            pa.scalar(Decimal("1000"), pa.decimal256(4, 0)),
        )
        _record_formula(
            evidence.formula_quality[f"{spec.name}.market_cap_x1000"],
            expected,
            pc.cast(actual, expected.type, safe=True),
        )
    elif spec.name == "crsp_shares_interval":
        source = table["shares_outstanding_thousand"].combine_chunks()
        actual = table["shares_outstanding"].combine_chunks()
        expected = pc.multiply_checked(source, pa.scalar(1000, pa.int64()))
        _record_formula(
            evidence.formula_quality[f"{spec.name}.shares_x1000"],
            expected,
            actual,
        )


def _update_date_scope(
    evidence: BuildEvidence,
    table: pa.Table,
    spec: core.TableSpec,
    start_date: dt.date,
    end_date: dt.date,
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
    start = pa.scalar(start_date, pa.date32())
    end = pa.scalar(end_date, pa.date32())
    december_start = pa.scalar(dt.date(2014, 12, 1), pa.date32())
    boundary = december_start
    counts = evidence.date_scope[spec.name]
    counts["rows"] += table.num_rows
    counts["before_scope_rows"] += _true_count(pc.less(values, start))
    counts["after_scope_rows"] += _true_count(pc.greater(values, end))
    counts["december_2014_rows"] += _true_count(
        pc.and_(pc.greater_equal(values, december_start), pc.less_equal(values, end))
    )
    counts["boundary_2014_12_01_rows"] += _true_count(pc.equal(values, boundary))


def _update_table_quality(
    evidence: BuildEvidence,
    raw: pa.RecordBatch,
    table: pa.Table,
    spec: core.TableSpec,
    settings: TypedBackboneSettings,
) -> None:
    _update_null_conservation(evidence, raw, table, spec)
    _update_decimal_roundtrip(evidence, raw, table, spec)
    _update_formula_quality(evidence, table, spec)
    _update_date_scope(evidence, table, spec, settings.start_date, settings.end_date)


def _update_quarantine_guards(
    evidence: BuildEvidence, table: pa.Table, table_name: str
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


def _optional_date_days_array(
    raw: pa.Array, label: str
) -> tuple[np.ndarray, np.ndarray]:
    blank = np.asarray(pc.equal(raw, "").to_numpy(zero_copy_only=False), dtype=bool)
    routed = pc.if_else(
        pa.array(blank, type=pa.bool_()),
        pa.scalar("1970-01-01", pa.string()),
        raw,
    )
    parsed = discovery._strict_date32(routed, label)
    days = discovery._date_days(parsed)
    return days, ~blank


def _identity_diagnostics(
    raw: pa.RecordBatch,
    *,
    index: IntervalIndex,
    as_of_column: str,
    metrics: JoinMetrics,
) -> None:
    permnos = discovery._positive_permnos(
        raw.column(raw.schema.get_field_index("PERMNO")),
        "identity diagnostic PERMNO",
    )
    as_of_raw = raw.column(raw.schema.get_field_index(as_of_column))
    if as_of_column == "MthPrcDt":
        as_of_days, valid = _optional_date_days_array(
            as_of_raw, "monthly identity observation date"
        )
    else:
        parsed = discovery._strict_date32(as_of_raw, "daily identity observation date")
        as_of_days = discovery._date_days(parsed)
        valid = np.ones(raw.num_rows, dtype=bool)
    matched = index.match(permnos, as_of_days, valid)
    one = matched >= 0
    metrics.population_rows += raw.num_rows
    metrics.blank_as_of_date_rows += int(np.count_nonzero(~valid))
    metrics.one_match_rows += int(np.count_nonzero(one))
    metrics.zero_match_rows += int(np.count_nonzero(~one))

    embedded_starts = discovery._date_days(
        discovery._strict_date32(
            raw.column(raw.schema.get_field_index("SecInfoStartDt")),
            "embedded identity interval start",
        )
    )
    embedded_ends = discovery._date_days(
        discovery._strict_date32(
            raw.column(raw.schema.get_field_index("SecInfoEndDt")),
            "embedded identity interval end",
        )
    )
    embedded_ids = index.match(
        permnos,
        embedded_starts,
        np.ones(raw.num_rows, dtype=bool),
    )
    embedded_found = embedded_ids >= 0
    safe_embedded = np.maximum(embedded_ids, 0)
    embedded_exact = (
        embedded_found
        & (index.starts[safe_embedded] == embedded_starts)
        & (index.ends[safe_embedded] == embedded_ends)
    )
    metrics.embedded_interval_exact_key_match_rows += int(
        np.count_nonzero(embedded_exact)
    )
    metrics.embedded_interval_exact_key_missing_rows += int(
        np.count_nonzero(~embedded_exact)
    )

    safe = np.maximum(matched, 0)
    interval_equal = (
        one
        & (index.starts[safe] == embedded_starts)
        & (index.ends[safe] == embedded_ends)
    )
    metrics.as_of_interval_key_equal_rows += int(np.count_nonzero(interval_equal))
    metrics.as_of_interval_key_different_rows += int(
        np.count_nonzero(one & ~interval_equal)
    )

    payload_equal = one.copy()
    take_indices = pa.array(safe, type=pa.int64())
    for field in discovery.IDENTITY_PAYLOAD_FIELDS:
        reference = pc.take(index.payload[field], take_indices)
        observed = raw.column(raw.schema.get_field_index(field))
        equal = np.asarray(
            pc.equal(observed, reference).to_numpy(zero_copy_only=False),
            dtype=bool,
        )
        payload_equal &= equal
    metrics.as_of_payload_all_fields_equal_rows += int(np.count_nonzero(payload_equal))
    metrics.as_of_payload_one_or_more_fields_different_rows += int(
        np.count_nonzero(one & ~payload_equal)
    )


def _share_join_diagnostics(
    raw: pa.RecordBatch,
    *,
    index: IntervalIndex,
    as_of_column: str,
) -> dict[str, int]:
    permnos = discovery._positive_permnos(
        raw.column(raw.schema.get_field_index("PERMNO")),
        "shares diagnostic PERMNO",
    )
    as_of_raw = raw.column(raw.schema.get_field_index(as_of_column))
    if as_of_column == "MthPrcDt":
        dates, valid = _optional_date_days_array(
            as_of_raw, "monthly shares observation date"
        )
    else:
        dates = discovery._date_days(
            discovery._strict_date32(as_of_raw, "daily shares observation date")
        )
        valid = np.ones(raw.num_rows, dtype=bool)
    matched = index.match(permnos, dates, valid)
    blank = int(np.count_nonzero(~valid))
    one = int(np.count_nonzero(matched >= 0))
    zero_nonblank = int(np.count_nonzero(valid & (matched < 0)))
    return {
        "population_rows": raw.num_rows,
        "blank_observation_date_rows": blank,
        "nonblank_observation_date_rows": int(np.count_nonzero(valid)),
        "one_match_rows": one,
        "zero_match_rows": blank + zero_nonblank,
        "zero_match_rows_with_nonblank_observation_date": zero_nonblank,
        "total_rows_without_usable_share_match": blank + zero_nonblank,
        "multiple_match_rows": 0,
    }


def _add_counts(target: Counter[str], values: Mapping[str, int]) -> None:
    for key, value in values.items():
        target[key] += int(value)


def _key_hasher(label: str, key: Sequence[str]) -> Any:
    digest = hashlib.sha256()
    digest.update(b"phase02-crsp-typed-key-v1\x00")
    encoded = (label + "\x00" + "\x00".join(key)).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest


def _validate_interval_index(
    index: IntervalIndex,
    expected: Mapping[str, Any],
    *,
    expected_rows: int,
    label: str,
) -> None:
    pairs = {
        "retained_rows": (len(index.permnos), expected_rows),
        "unique_permnos": (index.unique_permnos, int(expected["unique_permnos"])),
        "duplicate_key_groups": (
            index.duplicate_key_groups,
            int(expected["duplicate_key_groups"]),
        ),
        "contiguous_edges": (
            index.contiguous_edges,
            int(expected.get("contiguous_edges", index.contiguous_edges)),
        ),
        "gap_edges": (index.gap_edges, int(expected["gap_edges"])),
        "overlap_edges": (index.overlap_edges, int(expected["overlap_edges"])),
    }
    differences = {
        field: {"actual": actual, "expected": locked}
        for field, (actual, locked) in pairs.items()
        if actual != locked
    }
    if differences:
        raise TypedBackboneBuildError(
            f"{label} interval structure differs from the lock: {differences}"
        )


def _assert_join_locks(
    settings: TypedBackboneSettings,
    identity: Mapping[str, JoinMetrics],
    shares: Mapping[str, Counter[str]],
) -> None:
    joins = settings.policy["joins"]
    identity_fields = {
        "daily": (
            "population_rows",
            "blank_as_of_date_rows",
            "one_match_rows",
            "zero_match_rows",
            "multiple_match_rows",
            "embedded_interval_exact_key_match_rows",
            "embedded_interval_exact_key_missing_rows",
            "as_of_interval_key_equal_rows",
            "as_of_interval_key_different_rows",
            "as_of_payload_all_fields_equal_rows",
            "as_of_payload_one_or_more_fields_different_rows",
        ),
        "monthly": (
            "population_rows",
            "blank_as_of_date_rows",
            "one_match_rows",
            "zero_match_rows",
            "multiple_match_rows",
            "embedded_interval_exact_key_match_rows",
            "embedded_interval_exact_key_missing_rows",
            "as_of_interval_key_equal_rows",
            "as_of_interval_key_different_rows",
            "as_of_payload_all_fields_equal_rows",
            "as_of_payload_one_or_more_fields_different_rows",
        ),
    }
    for frequency, fields in identity_fields.items():
        actual = identity[frequency].as_dict()
        expected = joins[f"{frequency}_to_identity"]
        for field in fields:
            if actual[field] != expected[field]:
                raise TypedBackboneBuildError(
                    f"{frequency} identity join differs for {field}: "
                    f"expected {expected[field]}, actual {actual[field]}"
                )
        if frequency == "monthly":
            nonblank = actual["population_rows"] - actual["blank_as_of_date_rows"]
            zero_nonblank = actual["zero_match_rows"] - actual["blank_as_of_date_rows"]
            if nonblank != expected["nonblank_as_of_date_rows"]:
                raise TypedBackboneBuildError(
                    "Monthly identity nonblank observation count differs"
                )
            if zero_nonblank != expected["zero_match_rows_with_nonblank_as_of_date"]:
                raise TypedBackboneBuildError(
                    "Monthly identity nonblank zero-match count differs"
                )
    for frequency in ("daily", "monthly"):
        actual = shares[frequency]
        expected = joins[f"{frequency}_to_shares"]
        fields = ("population_rows", "one_match_rows", "multiple_match_rows")
        if frequency == "monthly":
            fields += (
                "blank_observation_date_rows",
                "nonblank_observation_date_rows",
                "zero_match_rows_with_nonblank_observation_date",
                "total_rows_without_usable_share_match",
            )
        else:
            fields += ("zero_match_rows",)
        for field in fields:
            if actual[field] != expected[field]:
                raise TypedBackboneBuildError(
                    f"{frequency} shares join differs for {field}"
                )


def compile_typed_tables(
    settings: TypedBackboneSettings,
    streams: Mapping[str, Any],
    *,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> BuildEvidence:
    """Compile all six Parquet tables into caller-owned binary streams."""

    from portfolio_bot.data import phase_02_crsp_typed_backbone_transforms as transforms

    if set(streams) != set(TABLE_NAMES):
        raise TypedBackboneBuildError("Typed output stream set is not exact")
    progress = progress or (lambda _stage, _details: None)
    evidence = BuildEvidence(
        canonicalization={name: CanonicalizationMetrics() for name in SOURCE_NAMES}
    )
    identity_index_builder = IntervalIndexBuilder(discovery.IDENTITY_PAYLOAD_FIELDS)
    shares_index_builder = IntervalIndexBuilder()
    daily_identity = JoinMetrics()
    monthly_identity = JoinMetrics()
    share_counts = {"daily": Counter(), "monthly": Counter()}
    key_domains = {
        "crsp_identity_interval": "identity",
        "crsp_shares_interval": "shares",
        "crsp_security_day": "daily",
        "crsp_security_day_return_staging": "daily",
        "crsp_security_month_crosscheck": "monthly",
        "crsp_security_month_return_staging": "monthly",
    }
    # Every table receives a distinct hasher object.  Paired consumer/staging
    # tables deliberately share only the hash domain so equal digests prove
    # equal independently observed key byte streams.
    hashers = {
        table_name: _key_hasher(key_domains[table_name], spec.primary_key)
        for table_name, spec in settings.registry.tables.items()
    }
    key_trackers = {
        table_name: KeyQualityTracker(
            spec=spec,
            start_date=settings.start_date,
            end_date=settings.end_date,
        )
        for table_name, spec in settings.registry.tables.items()
    }
    return_sources = {
        "DlyRet",
        "DlyRetx",
        "DlyRetMissFlg",
        "DlyRetDurFlg",
        "MthRet",
        "MthRetx",
        "MthRetFlg",
    }
    for table_name in (
        "crsp_security_day",
        "crsp_security_month_crosscheck",
        "crsp_identity_interval",
        "crsp_shares_interval",
    ):
        spec = settings.registry.tables[table_name]
        prohibited = sorted(
            field.name
            for field in spec.fields
            if (
                isinstance(field.metadata.get("source_column"), str)
                and field.metadata["source_column"] in return_sources
            )
            or "return" in field.name.lower()
        )
        evidence.consumer_return_absence[table_name] = {
            "prohibited_column_count": len(prohibited),
            "prohibited_columns": prohibited,
        }
    sinks: dict[str, ParquetSink] = {}
    with contextlib.ExitStack() as stack:
        for table_name in TABLE_NAMES:
            sinks[table_name] = stack.enter_context(
                ParquetSink(streams[table_name], settings.registry.tables[table_name])
            )

        identity_spec = settings.sources["identity"]
        with discovery._AuthenticatedSource.open(
            settings.raw_root, identity_spec
        ) as source:
            progress(
                "identity_source_authenticated",
                {"size_bytes": identity_spec.size_bytes},
            )
            metrics = evidence.canonicalization["identity"]
            for raw in _sorted_interval_batches(
                _scoped_interval_batches(
                    source,
                    start_column="SecInfoStartDt",
                    end_column="SecInfoEndDt",
                    settings=settings,
                    metrics=metrics,
                ),
                start_column="SecInfoStartDt",
                end_column="SecInfoEndDt",
                label="identity",
            ):
                identity_index_builder.add(
                    raw,
                    start_column="SecInfoStartDt",
                    end_column="SecInfoEndDt",
                )
                table = transforms.transform_identity(
                    raw, settings.registry.tables["crsp_identity_interval"]
                )
                sinks["crsp_identity_interval"].add(table)
                key_trackers["crsp_identity_interval"].add(table)
                _update_table_quality(
                    evidence,
                    raw,
                    table,
                    settings.registry.tables["crsp_identity_interval"],
                    settings,
                )
                _update_key_digest(
                    hashers["crsp_identity_interval"],
                    table,
                    settings.registry.tables["crsp_identity_interval"].primary_key,
                )
                metrics.canonical_rows += table.num_rows
            identity_index = identity_index_builder.finish("identity")
            _validate_interval_index(
                identity_index,
                settings.policy["canonicalization"]["identity"],
                expected_rows=metrics.canonical_rows,
                label="identity",
            )
            evidence.interval_structure["identity"] = {
                "contiguous_edges": identity_index.contiguous_edges,
                "duplicate_key_groups": identity_index.duplicate_key_groups,
                "gap_edges": identity_index.gap_edges,
                "overlap_edges": identity_index.overlap_edges,
                "retained_rows": len(identity_index.permnos),
                "unique_permnos": identity_index.unique_permnos,
            }
            progress("identity_table_complete", metrics.as_dict())

        shares_spec = settings.sources["shares"]
        with discovery._AuthenticatedSource.open(
            settings.raw_root, shares_spec
        ) as source:
            progress(
                "shares_source_authenticated", {"size_bytes": shares_spec.size_bytes}
            )
            metrics = evidence.canonicalization["shares"]
            for raw in _sorted_interval_batches(
                _scoped_interval_batches(
                    source,
                    start_column="ShrStartDt",
                    end_column="ShrEndDt",
                    settings=settings,
                    metrics=metrics,
                ),
                start_column="ShrStartDt",
                end_column="ShrEndDt",
                label="shares",
            ):
                shares_index_builder.add(
                    raw, start_column="ShrStartDt", end_column="ShrEndDt"
                )
                table = transforms.transform_shares(
                    raw, settings.registry.tables["crsp_shares_interval"]
                )
                sinks["crsp_shares_interval"].add(table)
                key_trackers["crsp_shares_interval"].add(table)
                _update_table_quality(
                    evidence,
                    raw,
                    table,
                    settings.registry.tables["crsp_shares_interval"],
                    settings,
                )
                _update_key_digest(
                    hashers["crsp_shares_interval"],
                    table,
                    settings.registry.tables["crsp_shares_interval"].primary_key,
                )
                metrics.canonical_rows += table.num_rows
            shares_index = shares_index_builder.finish("shares")
            _validate_interval_index(
                shares_index,
                settings.policy["canonicalization"]["shares"],
                expected_rows=metrics.canonical_rows,
                label="shares",
            )
            evidence.interval_structure["shares"] = {
                "contiguous_edges": shares_index.contiguous_edges,
                "duplicate_key_groups": shares_index.duplicate_key_groups,
                "gap_edges": shares_index.gap_edges,
                "overlap_edges": shares_index.overlap_edges,
                "retained_rows": len(shares_index.permnos),
                "unique_permnos": shares_index.unique_permnos,
            }
            progress("shares_table_complete", metrics.as_dict())

        daily_spec = settings.sources["daily"]
        with discovery._AuthenticatedSource.open(
            settings.raw_root, daily_spec
        ) as source:
            progress(
                "daily_source_authenticated", {"size_bytes": daily_spec.size_bytes}
            )
            metrics = evidence.canonicalization["daily"]
            scoped = _scoped_market_batches(
                source,
                date_column="DlyCalDt",
                settings=settings,
                metrics=metrics,
            )
            last_report = 0
            for raw in canonicalize_daily_batches(scoped, metrics):
                _identity_diagnostics(
                    raw,
                    index=identity_index,
                    as_of_column="DlyCalDt",
                    metrics=daily_identity,
                )
                _add_counts(
                    share_counts["daily"],
                    _share_join_diagnostics(
                        raw,
                        index=shares_index,
                        as_of_column="DlyCalDt",
                    ),
                )
                consumer, returns = transforms.transform_daily(
                    raw,
                    settings.registry.tables["crsp_security_day"],
                    settings.registry.tables["crsp_security_day_return_staging"],
                )
                sinks["crsp_security_day"].add(consumer)
                sinks["crsp_security_day_return_staging"].add(returns)
                key_trackers["crsp_security_day"].add(consumer)
                key_trackers["crsp_security_day_return_staging"].add(returns)
                _update_table_quality(
                    evidence,
                    raw,
                    consumer,
                    settings.registry.tables["crsp_security_day"],
                    settings,
                )
                _update_table_quality(
                    evidence,
                    raw,
                    returns,
                    settings.registry.tables["crsp_security_day_return_staging"],
                    settings,
                )
                _update_quarantine_guards(
                    evidence, returns, "crsp_security_day_return_staging"
                )
                evidence.return_split["daily"].update(
                    {
                        "raw_rows": raw.num_rows,
                        "consumer_rows": consumer.num_rows,
                        "return_staging_rows": returns.num_rows,
                    }
                )
                _update_key_digest(
                    hashers["crsp_security_day"],
                    consumer,
                    settings.registry.tables["crsp_security_day"].primary_key,
                )
                _update_key_digest(
                    hashers["crsp_security_day_return_staging"],
                    returns,
                    settings.registry.tables[
                        "crsp_security_day_return_staging"
                    ].primary_key,
                )
                if metrics.canonical_rows - last_report >= 2_000_000:
                    last_report = metrics.canonical_rows
                    progress(
                        "daily_rows_compiled",
                        {"canonical_rows": metrics.canonical_rows},
                    )
            progress("daily_tables_complete", metrics.as_dict())

        monthly_spec = settings.sources["monthly"]
        with discovery._AuthenticatedSource.open(
            settings.raw_root, monthly_spec
        ) as source:
            progress(
                "monthly_source_authenticated", {"size_bytes": monthly_spec.size_bytes}
            )
            metrics = evidence.canonicalization["monthly"]
            scoped = _scoped_market_batches(
                source,
                date_column="MthCalDt",
                settings=settings,
                metrics=metrics,
            )
            for raw in canonicalize_monthly_batches(scoped, metrics):
                _identity_diagnostics(
                    raw,
                    index=identity_index,
                    as_of_column="MthPrcDt",
                    metrics=monthly_identity,
                )
                _add_counts(
                    share_counts["monthly"],
                    _share_join_diagnostics(
                        raw,
                        index=shares_index,
                        as_of_column="MthPrcDt",
                    ),
                )
                consumer, returns = transforms.transform_monthly(
                    raw,
                    settings.registry.tables["crsp_security_month_crosscheck"],
                    settings.registry.tables["crsp_security_month_return_staging"],
                )
                sinks["crsp_security_month_crosscheck"].add(consumer)
                sinks["crsp_security_month_return_staging"].add(returns)
                key_trackers["crsp_security_month_crosscheck"].add(consumer)
                key_trackers["crsp_security_month_return_staging"].add(returns)
                _update_table_quality(
                    evidence,
                    raw,
                    consumer,
                    settings.registry.tables["crsp_security_month_crosscheck"],
                    settings,
                )
                _update_table_quality(
                    evidence,
                    raw,
                    returns,
                    settings.registry.tables["crsp_security_month_return_staging"],
                    settings,
                )
                _update_quarantine_guards(
                    evidence, returns, "crsp_security_month_return_staging"
                )
                evidence.return_split["monthly"].update(
                    {
                        "raw_rows": raw.num_rows,
                        "consumer_rows": consumer.num_rows,
                        "return_staging_rows": returns.num_rows,
                    }
                )
                _update_key_digest(
                    hashers["crsp_security_month_crosscheck"],
                    consumer,
                    settings.registry.tables[
                        "crsp_security_month_crosscheck"
                    ].primary_key,
                )
                _update_key_digest(
                    hashers["crsp_security_month_return_staging"],
                    returns,
                    settings.registry.tables[
                        "crsp_security_month_return_staging"
                    ].primary_key,
                )
            progress("monthly_tables_complete", metrics.as_dict())

    for table_name, sink in sinks.items():
        evidence.table_rows[table_name] = sink.row_count
        evidence.table_nulls[table_name].update(sink.null_counts)
    evidence.key_sha256 = {
        table_name: digest.hexdigest() for table_name, digest in hashers.items()
    }
    evidence.key_quality = {
        table_name: tracker.as_dict() for table_name, tracker in key_trackers.items()
    }
    _assert_join_locks(
        settings,
        {"daily": daily_identity, "monthly": monthly_identity},
        share_counts,
    )
    monthly_identity_counts = monthly_identity.as_dict()
    monthly_identity_counts["nonblank_as_of_date_rows"] = (
        monthly_identity.population_rows - monthly_identity.blank_as_of_date_rows
    )
    monthly_identity_counts["zero_match_rows_with_nonblank_as_of_date"] = (
        monthly_identity.zero_match_rows - monthly_identity.blank_as_of_date_rows
    )
    evidence.joins = {
        "daily_to_identity": daily_identity.as_dict(),
        "monthly_to_identity": monthly_identity_counts,
        "daily_to_shares": dict(share_counts["daily"]),
        "monthly_to_shares": dict(share_counts["monthly"]),
    }
    validate_evidence(settings, evidence)
    progress("typed_compile_complete", {"table_rows": dict(evidence.table_rows)})
    return evidence


def validate_evidence(settings: TypedBackboneSettings, evidence: BuildEvidence) -> None:
    """Apply the exact hard row and canonicalization locks before publication."""

    for source_name, spec in settings.sources.items():
        metrics = evidence.canonicalization[source_name]
        source_pairs = (
            ("full rows", metrics.full_rows, spec.full_file_rows),
            ("retained rows", metrics.retained_rows, spec.retained_rows),
            ("canonical rows", metrics.canonical_rows, spec.canonical_rows),
        )
        for field, actual, expected in source_pairs:
            if expected is not None and actual != expected:
                raise TypedBackboneBuildError(
                    f"{source_name} {field} differs: expected {expected}, actual {actual}"
                )
    for table_name, spec in settings.registry.tables.items():
        actual = evidence.table_rows[table_name]
        if actual != spec.expected_rows:
            raise TypedBackboneBuildError(
                f"{table_name} row count differs: expected {spec.expected_rows}, actual {actual}"
            )
    daily_expected = settings.policy["canonicalization"]["daily"]
    monthly_expected = settings.policy["canonicalization"]["monthly"]
    daily = evidence.canonicalization["daily"]
    monthly = evidence.canonicalization["monthly"]
    pairs = (
        (daily.retained_rows, daily_expected["retained_rows"], "daily retained"),
        (daily.canonical_rows, daily_expected["output_rows"], "daily output"),
        (
            daily.duplicate_surplus_rows,
            daily_expected["duplicate_surplus_rows"],
            "daily duplicate surplus",
        ),
        (
            daily.duplicate_groups,
            daily_expected["exact_duplicate_groups"],
            "daily duplicate groups",
        ),
        (monthly.retained_rows, monthly_expected["retained_rows"], "monthly retained"),
        (monthly.canonical_rows, monthly_expected["output_rows"], "monthly output"),
        (
            monthly.duplicate_surplus_rows,
            monthly_expected["exact_raw_key_duplicate_surplus_rows"],
            "monthly duplicate surplus",
        ),
        (
            monthly.duplicate_groups,
            monthly_expected["exact_raw_key_duplicate_groups"],
            "monthly duplicate groups",
        ),
        (
            monthly.candidate_groups,
            monthly_expected["effective_interval_candidate_groups"],
            "monthly candidates",
        ),
    )
    for actual, expected, label in pairs:
        if actual != expected:
            raise TypedBackboneBuildError(
                f"{label} count differs: expected {expected}, actual {actual}"
            )
    if daily.conflicting_groups or monthly.conflicting_groups:
        raise TypedBackboneBuildError("Conflicting canonical duplicate groups exist")

    expected_candidate_size = int(monthly_expected["candidate_group_size"])
    unexpected_candidate_rows = sum(
        count
        for size, count in monthly.candidate_size_histogram.items()
        if int(size) != expected_candidate_size
    )
    if unexpected_candidate_rows:
        raise TypedBackboneBuildError(
            "Monthly effective-interval candidate group size differs"
        )

    for table_name, spec in settings.registry.tables.items():
        key = evidence.key_quality.get(table_name, {})
        if key.get("rows") != evidence.table_rows[table_name]:
            raise TypedBackboneBuildError(
                f"{table_name} key-quality row conservation differs"
            )
        for metric in ("null_rows", "duplicate_rows", "sorted_violations"):
            if key.get(metric) != 0:
                raise TypedBackboneBuildError(
                    f"{table_name} primary-key {metric} is nonzero"
                )
        for field in spec.fields:
            expected_nulls = evidence.raw_expected_nulls[table_name][field.name]
            actual_nulls = evidence.table_nulls[table_name][field.name]
            if expected_nulls != actual_nulls:
                raise TypedBackboneBuildError(
                    f"{table_name}.{field.name} raw-to-typed null conservation differs: "
                    f"expected {expected_nulls}, actual {actual_nulls}"
                )

    for table_name, columns in evidence.decimal_roundtrip.items():
        for column_name, counts in columns.items():
            if counts["mismatch_rows"] or counts["grammar_failure_rows"]:
                raise TypedBackboneBuildError(
                    f"{table_name}.{column_name} exact decimal roundtrip differs"
                )
            if counts["exact_match_rows"] != counts["source_nonblank_rows"]:
                raise TypedBackboneBuildError(
                    f"{table_name}.{column_name} decimal row conservation differs"
                )

    for formula_id, counts in evidence.formula_quality.items():
        if counts["mismatch_rows"]:
            raise TypedBackboneBuildError(f"{formula_id} exact identity differs")
        if counts["expected_null_rows"] != counts["actual_null_rows"]:
            raise TypedBackboneBuildError(
                f"{formula_id} derived-null conservation differs"
            )
        if counts["match_rows"] != counts["rows"]:
            raise TypedBackboneBuildError(f"{formula_id} row conservation differs")

    return_split = settings.policy["canonicalization"]["return_split"]
    split_tables = {
        "daily": ("crsp_security_day", "crsp_security_day_return_staging"),
        "monthly": (
            "crsp_security_month_crosscheck",
            "crsp_security_month_return_staging",
        ),
    }
    for frequency, (consumer_name, staging_name) in split_tables.items():
        counts = evidence.return_split[frequency]
        expected_rows = int(return_split[f"{frequency}_consumer_rows"])
        for metric in ("raw_rows", "consumer_rows", "return_staging_rows"):
            if counts[metric] != expected_rows:
                raise TypedBackboneBuildError(
                    f"{frequency} return split {metric} differs"
                )
        if evidence.key_sha256[consumer_name] != evidence.key_sha256[staging_name]:
            raise TypedBackboneBuildError(
                f"{frequency} independently hashed return-split keys differ"
            )

    for table_name in (
        "crsp_security_day_return_staging",
        "crsp_security_month_return_staging",
    ):
        guards = evidence.quarantine_guards[table_name]
        if guards["rows"] != evidence.table_rows[table_name]:
            raise TypedBackboneBuildError(
                f"{table_name} quarantine guard row conservation differs"
            )
        for metric in (
            "status_not_unreconciled_rows",
            "authorized_true_rows",
            "status_null_rows",
            "authorized_null_rows",
        ):
            if guards[metric]:
                raise TypedBackboneBuildError(
                    f"{table_name} quarantine guard {metric} is nonzero"
                )

    for table_name, counts in evidence.consumer_return_absence.items():
        if counts["prohibited_column_count"]:
            raise TypedBackboneBuildError(
                f"{table_name} contains a prohibited return-family column"
            )

    for table_name in (
        "crsp_security_day",
        "crsp_security_day_return_staging",
        "crsp_security_month_crosscheck",
        "crsp_security_month_return_staging",
    ):
        counts = evidence.date_scope[table_name]
        if counts["rows"] != evidence.table_rows[table_name]:
            raise TypedBackboneBuildError(
                f"{table_name} typed date-scope row conservation differs"
            )
        if counts["before_scope_rows"] or counts["after_scope_rows"]:
            raise TypedBackboneBuildError(
                f"{table_name} contains a typed key outside the permitted scope"
            )
