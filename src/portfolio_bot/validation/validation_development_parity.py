"""Executable full-development equivalence authority for Stage 11 predictors.

The proof rehashes the complete frozen Stage 6 input allowlist and verified
outputs, authenticates every one of the 459 immutable family-cache batches,
and then performs a bounded full streaming replay.  All seven cached family
value streams are aligned against every field of the published 170-column raw
store, and every cached lineage row (including status and timing dates) is
compared with the corresponding published lineage row.  It also proves that
the generalized Stage 11 wrapper uses the exact hash-frozen Stage 6 formula
callables and that its development scope adapter is an exact no-op.

No label, validation outcome, final-period row, portfolio, trade, or P&L
payload is opened.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import portfolio_bot.features.materialize_aggregate_state as aggregate_adapter
import portfolio_bot.features.materialize_crsp as crsp_adapter
import portfolio_bot.features.materialize_crsp_routing as crsp_routing
import portfolio_bot.features.materialize_crsp_streaming as crsp_streaming
import portfolio_bot.features.materialize_fundamentals as fundamental_adapter
import portfolio_bot.features.materialize_optionmetrics as optionmetrics_adapter
import portfolio_bot.features.materialize_ownership_events as ownership_adapter
from portfolio_bot.features.feature_store_inputs import resolve_input_catalog
from portfolio_bot.features.feature_lineage import FeatureFamilyResult, FeatureLineage
from portfolio_bot.features.feature_store_materialization import (
    FeatureMaterializationRow,
    FeatureStoreBuildIdentity,
    compute_feature_store_build_identity,
)
from portfolio_bot.features.feature_store_family_orchestration import (
    FamilyCompletionReceipt,
    authenticate_family_completion,
)
from portfolio_bot.features.materialize_crsp import CRSP_FAMILIES
from portfolio_bot.features.materialize_crsp_routing import CrspFamilyRowBatch
from portfolio_bot.validation.validation_data_contract import (
    PROJECT_ROOT,
    ValidationDataContract,
    canonical_json_bytes,
    schema_sha256,
    sha256_file,
    validate_validation_data_contract,
)


PARITY_STATUS = "PASS_FULL_DEVELOPMENT_STREAMING_REPLAY_EXACT"
PARITY_METHOD = (
    "authenticated_stage6_family_cache_full_streaming_replay_against_verified_outputs"
)
PARITY_RECEIPT_SCHEMA_VERSION = 1
STAGE6_POINTER = PROJECT_ROOT / "artifacts/phase_03_features/LATEST_FEATURE_STORE_VERIFIED.json"
STAGE6_MANIFEST_PARENT = PROJECT_ROOT / "artifacts/phase_03_features/feature_store_builds"


class ValidationDevelopmentParityError(RuntimeError):
    """Raised when development equivalence cannot be proved from bytes."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationDevelopmentParityError(message)


def _json(path: Path, label: str) -> dict[str, Any]:
    _check(path.is_file() and not path.is_symlink(), f"Missing or unsafe {label}")
    try:
        value = json.loads(path.read_text("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationDevelopmentParityError(f"Invalid {label} JSON") from exc
    _check(isinstance(value, dict), f"{label} is not an object")
    return value


def _ordinary(path: Path, root: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    trusted = Path(os.path.abspath(root.expanduser()))
    try:
        lexical.relative_to(trusted)
    except ValueError as exc:
        raise ValidationDevelopmentParityError(f"{label} escaped its root") from exc
    _check(lexical.resolve(strict=False) == lexical, f"{label} uses a symlinked path")
    metadata = lexical.stat(follow_symlinks=False)
    _check(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1, f"Unsafe {label}")
    return lexical


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _date_text(value: object) -> str:
    _check(type(value) is dt.date, "Development scope contains an invalid date")
    return value.isoformat()


def _development_scope_values() -> dict[str, Any]:
    """Read every mutable Stage 6 scope constant before any patch is active."""

    return {
        "aggregate_state.CONTEXT_END_DATE": _date_text(aggregate_adapter.CONTEXT_END_DATE),
        "aggregate_state.PREDICTOR_END_DATE": _date_text(aggregate_adapter.PREDICTOR_END_DATE),
        "crsp.DEVELOPMENT_LAST_SIGNAL": _date_text(crsp_adapter.DEVELOPMENT_LAST_SIGNAL),
        "crsp.MAXIMUM_CONTEXT_DATE": _date_text(crsp_adapter.MAXIMUM_CONTEXT_DATE),
        "crsp_streaming.DEVELOPMENT_LAST_SIGNAL": _date_text(
            crsp_streaming.DEVELOPMENT_LAST_SIGNAL
        ),
        "crsp_streaming.MAXIMUM_CONTEXT_DATE": _date_text(
            crsp_streaming.MAXIMUM_CONTEXT_DATE
        ),
        "fundamentals.EXPECTED_SIGNAL_DATES": fundamental_adapter.EXPECTED_SIGNAL_DATES,
        "fundamentals.EXPECTED_SPINE_ROWS": fundamental_adapter.EXPECTED_SPINE_ROWS,
        "fundamentals.FIRST_SIGNAL_DATE": _date_text(fundamental_adapter.FIRST_SIGNAL_DATE),
        "fundamentals.LAST_SIGNAL_DATE": _date_text(fundamental_adapter.LAST_SIGNAL_DATE),
        "fundamentals.MAXIMUM_CONTEXT_DATE": _date_text(
            fundamental_adapter.MAXIMUM_CONTEXT_DATE
        ),
        "optionmetrics.EXPECTED_SIGNAL_DATES": optionmetrics_adapter.EXPECTED_SIGNAL_DATES,
        "optionmetrics.EXPECTED_SPINE_ROWS": optionmetrics_adapter.EXPECTED_SPINE_ROWS,
        "optionmetrics.FIRST_SIGNAL_DATE": _date_text(optionmetrics_adapter.FIRST_SIGNAL_DATE),
        "optionmetrics.LAST_SIGNAL_DATE": _date_text(optionmetrics_adapter.LAST_SIGNAL_DATE),
        "ownership.FIRST_SIGNAL_DATE": _date_text(ownership_adapter.FIRST_SIGNAL_DATE),
        "ownership.LAST_SIGNAL_DATE": _date_text(ownership_adapter.LAST_SIGNAL_DATE),
    }


_EXPECTED_DEVELOPMENT_SCOPE = {
    "aggregate_state.CONTEXT_END_DATE": "2014-12-01",
    "aggregate_state.PREDICTOR_END_DATE": "2014-10-31",
    "crsp.DEVELOPMENT_LAST_SIGNAL": "2014-10-31",
    "crsp.MAXIMUM_CONTEXT_DATE": "2014-12-01",
    "crsp_streaming.DEVELOPMENT_LAST_SIGNAL": "2014-10-31",
    "crsp_streaming.MAXIMUM_CONTEXT_DATE": "2014-12-01",
    "fundamentals.EXPECTED_SIGNAL_DATES": 178,
    "fundamentals.EXPECTED_SPINE_ROWS": 1_263_048,
    "fundamentals.FIRST_SIGNAL_DATE": "2000-01-31",
    "fundamentals.LAST_SIGNAL_DATE": "2014-10-31",
    "fundamentals.MAXIMUM_CONTEXT_DATE": "2014-12-01",
    "optionmetrics.EXPECTED_SIGNAL_DATES": 178,
    "optionmetrics.EXPECTED_SPINE_ROWS": 1_263_048,
    "optionmetrics.FIRST_SIGNAL_DATE": "2000-01-31",
    "optionmetrics.LAST_SIGNAL_DATE": "2014-10-31",
    "ownership.FIRST_SIGNAL_DATE": "2000-01-31",
    "ownership.LAST_SIGNAL_DATE": "2014-10-31",
}


@dataclass(frozen=True, slots=True)
class DevelopmentParityReceipt:
    schema_version: int
    status: str
    method: str
    feature_rows: int
    lineage_rows: int
    empirical_replay_feature_rows: int
    empirical_replay_lineage_rows: int
    identity_mismatch_count: int
    empirical_difference_count: int | None
    reference_build_id: str
    reference_verification_id: str
    evidence_sha256: str
    evidence: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        # Round-trip to detach nested mutable containers and guarantee JSON
        # compatibility before the receipt enters another fingerprint.
        evidence = json.loads(json.dumps(dict(self.evidence), sort_keys=True))
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "method": self.method,
            "feature_rows": self.feature_rows,
            "lineage_rows": self.lineage_rows,
            "empirical_replay_feature_rows": self.empirical_replay_feature_rows,
            "empirical_replay_lineage_rows": self.empirical_replay_lineage_rows,
            "identity_mismatch_count": self.identity_mismatch_count,
            "empirical_difference_count": self.empirical_difference_count,
            "reference_build_id": self.reference_build_id,
            "reference_verification_id": self.reference_verification_id,
            "evidence_sha256": self.evidence_sha256,
            "evidence": evidence,
        }


_AUTHORITY_SEAL = object()


@dataclass(frozen=True, slots=True)
class ValidatedDevelopmentParity:
    """Opaque in-process authority created only after byte reauthentication."""

    receipt: DevelopmentParityReceipt
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _check(self._seal is _AUTHORITY_SEAL, "Development parity authority is not authenticated")


def _function_binding_evidence() -> dict[str, Any]:
    # Imports are local to avoid making the no-outcome empirical wrapper part
    # of Stage 6 module initialization.
    import portfolio_bot.validation.validation_data_candidate_builder as candidate
    import portfolio_bot.validation.validation_empirical_feature_stream as wrapper

    bindings = {
        "aggregate_materializer": (
            wrapper.materialize_aggregate_state,
            aggregate_adapter.materialize_aggregate_state,
        ),
        "crsp_router": (
            wrapper.spool_crsp_signal_router,
            crsp_routing.spool_crsp_signal_router,
        ),
        "crsp_stream": (
            wrapper.iter_unfinalized_crsp_rows,
            crsp_streaming.iter_unfinalized_crsp_rows,
        ),
        "fundamental_stream": (
            wrapper.iter_fundamental_feature_batches,
            fundamental_adapter.iter_fundamental_feature_batches,
        ),
        "optionmetrics_stream": (
            wrapper.iter_optionmetrics_feature_batches,
            optionmetrics_adapter.iter_optionmetrics_feature_batches,
        ),
        "ownership_stream": (
            wrapper.iter_ownership_event_feature_batches,
            ownership_adapter.iter_ownership_event_feature_batches,
        ),
    }
    _check(all(left is right for left, right in bindings.values()), "Wrapper formula callable differs")
    _check(
        fundamental_adapter.materialize_fundamentals
        is fundamental_adapter.iter_fundamental_feature_batches
        and ownership_adapter.materialize_ownership_events
        is ownership_adapter.iter_ownership_event_feature_batches
        and optionmetrics_adapter.materialize_optionmetrics
        is optionmetrics_adapter.iter_optionmetrics_feature_batches,
        "Stage 6 family aliases differ from the wrapper iterators",
    )
    expected_order = tuple(wrapper._FAMILY_ORDER)
    _check(
        expected_order
        == (
            "price_trends_path",
            "risk_return_distribution",
            "size_liquidity_trading_frictions",
            "valuation_fundamentals",
            "ownership_filings_events_segments",
            "optionmetrics",
            "aggregate_state",
        ),
        "Generalized wrapper family order differs",
    )
    return {
        "all_formula_callables_are_exact_stage6_objects": True,
        "bindings": {
            name: {
                "module": right.__module__,
                "qualname": right.__qualname__,
                "object_identity_equal": left is right,
            }
            for name, (left, right) in sorted(bindings.items())
        },
        "family_order": list(expected_order),
        "wrapper_sha256": sha256_file(Path(wrapper.__file__)),
        "candidate_assembly_sha256": sha256_file(Path(candidate.__file__)),
    }


def _signal_bounds(path: Path) -> tuple[str, str]:
    parquet = pq.ParquetFile(path)
    index = parquet.schema_arrow.get_field_index("signal_date")
    _check(index >= 0, f"Reference payload lacks signal_date: {path.name}")
    minima: list[dt.date] = []
    maxima: list[dt.date] = []
    for ordinal in range(parquet.num_row_groups):
        statistics = parquet.metadata.row_group(ordinal).column(index).statistics
        _check(statistics is not None and statistics.has_min_max, "Missing signal statistics")
        _check(type(statistics.min) is dt.date and type(statistics.max) is dt.date, "Bad signal statistics")
        minima.append(statistics.min)
        maxima.append(statistics.max)
    _check(bool(minima), f"Reference payload is empty: {path.name}")
    return min(minima).isoformat(), max(maxima).isoformat()


def _parquet_batches(
    paths: Sequence[Path],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 65_536,
) -> Iterator[pa.RecordBatch]:
    """Yield nonempty batches from an exact ordered path sequence."""

    _check(batch_size > 0 and bool(paths), "Development replay path set is empty")
    for path in paths:
        _check(path.is_file() and not path.is_symlink(), f"Unsafe replay input: {path}")
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=batch_size,
            columns=None if columns is None else list(columns),
            use_threads=False,
        ):
            _check(batch.num_rows > 0, f"Replay source emitted an empty batch: {path.name}")
            yield batch


def _aligned_record_batch_streams(
    streams: Mapping[str, Iterable[pa.RecordBatch]],
) -> Iterator[Mapping[str, pa.RecordBatch]]:
    """Align differently chunked ordered streams without buffering tables."""

    _check(bool(streams), "Development replay streams are empty")
    iterators = {name: iter(values) for name, values in streams.items()}
    active: dict[str, pa.RecordBatch | None] = {name: None for name in streams}
    offsets = {name: 0 for name in streams}
    while True:
        ended: set[str] = set()
        for name, iterator in iterators.items():
            if active[name] is None:
                try:
                    active[name] = next(iterator)
                    offsets[name] = 0
                except StopIteration:
                    ended.add(name)
        if ended:
            _check(
                len(ended) == len(streams)
                and all(active[name] is None for name in streams),
                "Development replay stream cardinalities differ",
            )
            return
        remaining = {
            name: batch.num_rows - offsets[name]
            for name, batch in active.items()
            if batch is not None
        }
        _check(
            set(remaining) == set(streams) and all(value > 0 for value in remaining.values()),
            "Development replay stream cursor is invalid",
        )
        width = min(remaining.values())
        aligned: dict[str, pa.RecordBatch] = {}
        for name, batch in active.items():
            assert batch is not None
            aligned[name] = batch.slice(offsets[name], width)
            offsets[name] += width
            if offsets[name] == batch.num_rows:
                active[name] = None
        yield MappingProxyType(aligned)


def _compare_columns_exact(
    left: pa.RecordBatch,
    right: pa.RecordBatch,
    *,
    names: Sequence[str],
    label: str,
    row_offset: int,
) -> None:
    """Require exact type/null/value equality for one aligned projection."""

    _check(left.num_rows == right.num_rows, f"Replay batch sizes differ: {label}")
    _check(
        tuple(left.schema.names) == tuple(names)
        and tuple(right.schema.names) == tuple(names),
        f"Replay column order differs: {label}",
    )
    for ordinal, name in enumerate(names):
        left_field = left.schema.field(ordinal)
        right_field = right.schema.field(ordinal)
        _check(
            left_field.type.equals(right_field.type)
            and left_field.nullable == right_field.nullable,
            f"Replay field type/nullability differs: {label}/{name}",
        )
        _check(
            left.column(ordinal).equals(right.column(ordinal)),
            f"Replay value differs at or after row {row_offset}: {label}/{name}",
        )


def _completion_value_paths(
    receipt: FamilyCompletionReceipt,
) -> tuple[Path, ...]:
    return tuple(
        receipt.path.parent
        / summary.batch_id
        / f"feature_values_{receipt.family}.parquet"
        for summary in receipt.batches
    )


def _completion_lineage_paths(
    receipt: FamilyCompletionReceipt,
) -> tuple[Path, ...]:
    return tuple(
        receipt.path.parent
        / summary.batch_id
        / f"feature_lineage_{receipt.family}.parquet"
        for summary in receipt.batches
    )


def _full_development_streaming_replay(
    *,
    completions: Mapping[str, FamilyCompletionReceipt],
    contract: ValidationDataContract,
    output_root: Path,
) -> dict[str, Any]:
    """Compare all cached formula outputs to all verified published outputs."""

    _check(
        tuple(completions) == tuple(contract.family_feature_names)
        and len(completions) == 7,
        "Development replay cache family role set/order differs",
    )
    _check(
        sum(item.batch_count for item in completions.values()) == 452,
        "Development replay authenticated cache batch count differs",
    )

    raw_path = output_root / "feature_store_raw.parquet"
    raw_parquet = pq.ParquetFile(raw_path)
    _check(
        raw_parquet.schema_arrow.equals(contract.raw_feature_schema, check_metadata=True),
        "Development replay raw schema/order differs",
    )
    audit_names = tuple(contract.raw_feature_schema.names[:7])
    streams: dict[str, Iterable[pa.RecordBatch]] = {
        "published_raw": _parquet_batches((raw_path,)),
    }
    for family, names in contract.family_feature_names.items():
        expected_names = (*audit_names, *names)
        streams[family] = _parquet_batches(
            _completion_value_paths(completions[family]),
            columns=expected_names,
        )

    raw_rows = 0
    for aligned in _aligned_record_batch_streams(streams):
        published = aligned["published_raw"]
        _check(
            published.schema.equals(contract.raw_feature_schema, check_metadata=True),
            "Development replay raw batch schema/order differs",
        )
        for family, names in contract.family_feature_names.items():
            projected_names = (*audit_names, *names)
            projected = published.select(projected_names)
            cached = aligned[family]
            _compare_columns_exact(
                projected,
                cached,
                names=projected_names,
                label=f"raw/{family}",
                row_offset=raw_rows,
            )
        raw_rows += published.num_rows

    expected_raw_rows = int(contract.rules["development_parity"]["reference_feature_rows"])
    _check(raw_rows == expected_raw_rows, "Full development raw replay row count differs")

    lineage_rows = 0
    lineage_by_family: dict[str, int] = {}
    for family, names in contract.family_feature_names.items():
        published_path = output_root / f"feature_lineage_{family}.parquet"
        published_parquet = pq.ParquetFile(published_path)
        _check(
            published_parquet.schema_arrow.equals(contract.lineage_schema, check_metadata=True),
            f"Development replay lineage schema differs: {family}",
        )
        family_rows = 0
        lineage_streams = {
            "published": _parquet_batches((published_path,)),
            "cached": _parquet_batches(_completion_lineage_paths(completions[family])),
        }
        for aligned in _aligned_record_batch_streams(lineage_streams):
            _compare_columns_exact(
                aligned["published"],
                aligned["cached"],
                names=contract.lineage_schema.names,
                label=f"lineage/{family}",
                row_offset=family_rows,
            )
            family_rows += aligned["published"].num_rows
        expected_family_rows = raw_rows * len(names)
        _check(
            family_rows == expected_family_rows,
            f"Full development lineage replay row count differs: {family}",
        )
        lineage_by_family[family] = family_rows
        lineage_rows += family_rows

    expected_lineage_rows = int(
        contract.rules["development_parity"]["reference_lineage_rows"]
    )
    _check(
        lineage_rows == expected_lineage_rows == raw_rows * len(contract.feature_names),
        "Full development lineage replay coverage differs",
    )
    completion_evidence = {
        family: {
            "manifest_sha256": receipt.manifest_sha256,
            "manifest_size_bytes": receipt.manifest_size_bytes,
            "batch_count": receipt.batch_count,
            "row_count": receipt.row_count,
            "lineage_row_count": receipt.lineage_row_count,
            "first_key": {
                "signal_date": receipt.first_key[0].isoformat(),
                "permno": receipt.first_key[1],
            },
            "last_key": {
                "signal_date": receipt.last_key[0].isoformat(),
                "permno": receipt.last_key[1],
            },
            "batch_receipts_sha256": _digest(
                [summary.as_dict() for summary in receipt.batches]
            ),
        }
        for family, receipt in completions.items()
    }
    return {
        "method": "bounded_full_cache_to_published_logical_value_replay",
        "cache_family_count": len(completions),
        "cache_batch_count": sum(item.batch_count for item in completions.values()),
        "cache_receipts": completion_evidence,
        "cache_receipts_sha256": _digest(completion_evidence),
        "raw_schema_order_exact": True,
        "raw_fields_compared": len(contract.raw_feature_schema),
        "raw_rows_compared": raw_rows,
        "raw_cells_compared": raw_rows * len(contract.raw_feature_schema),
        "lineage_schema_order_exact": True,
        "lineage_rows_compared": lineage_rows,
        "lineage_cells_compared": lineage_rows * len(contract.lineage_schema),
        "lineage_rows_by_family": lineage_by_family,
        "ordered_keys_values_status_and_dates_exact": True,
        "difference_count": 0,
        "memory_mode": "bounded_record_batches",
        "maximum_alignment_batch_rows": 65_536,
    }


def _key_in_ranges(
    key: tuple[dt.date, int],
    ranges: Sequence[tuple[tuple[dt.date, int], tuple[dt.date, int]]],
) -> bool:
    return any(first <= key <= last for first, last in ranges)


def _range_overlaps(
    first: tuple[dt.date, int],
    last: tuple[dt.date, int],
    ranges: Sequence[tuple[tuple[dt.date, int], tuple[dt.date, int]]],
) -> bool:
    return any(first <= selected_last and selected_first <= last for selected_first, selected_last in ranges)


def _cached_family_rows(
    receipt: FamilyCompletionReceipt,
    *,
    feature_names: Sequence[str],
    lineage_schema: pa.Schema,
    ranges: Sequence[tuple[tuple[dt.date, int], tuple[dt.date, int]]],
) -> Iterator[FeatureMaterializationRow]:
    """Rehydrate selected authenticated cache rows for the real wrapper path."""

    expected_value_names = (
        "signal_date",
        "permno",
        "permco",
        "long_eligible",
        "short_eligible",
        "primary_exchange",
        "share_type",
        *feature_names,
    )
    prior: tuple[dt.date, int] | None = None
    for summary in receipt.batches:
        if not _range_overlaps(summary.first_key, summary.last_key, ranges):
            continue
        batch_root = receipt.path.parent / summary.batch_id
        values = pq.read_table(
            batch_root / f"feature_values_{receipt.family}.parquet",
            use_threads=False,
        ).combine_chunks()
        lineages = pq.read_table(
            batch_root / f"feature_lineage_{receipt.family}.parquet",
            use_threads=False,
        ).combine_chunks()
        _check(
            tuple(values.schema.names) == expected_value_names
            and lineages.schema.equals(lineage_schema, check_metadata=True),
            f"Canary cache schema/order differs: {receipt.family}/{summary.batch_id}",
        )
        feature_count = len(feature_names)
        _check(
            values.num_rows == summary.row_count
            and lineages.num_rows == values.num_rows * feature_count,
            f"Canary cache cardinality differs: {receipt.family}/{summary.batch_id}",
        )
        value_columns = {name: values[name].chunk(0) for name in values.schema.names}
        lineage_columns = {name: lineages[name].chunk(0) for name in lineages.schema.names}
        for index in range(values.num_rows):
            signal = value_columns["signal_date"][index].as_py()
            permno = value_columns["permno"][index].as_py()
            _check(type(signal) is dt.date and type(permno) is int, "Canary key is invalid")
            key = (signal, permno)
            if not _key_in_ranges(key, ranges):
                continue
            _check(prior is None or prior < key, f"Canary cache order differs: {receipt.family}")
            prior = key
            base = index * feature_count
            values_map: dict[str, Any] = {}
            lineage_map: dict[str, FeatureLineage] = {}
            for offset, name in enumerate(feature_names):
                lineage_index = base + offset
                _check(
                    lineage_columns["signal_date"][lineage_index].as_py() == signal
                    and lineage_columns["permno"][lineage_index].as_py() == permno
                    and lineage_columns["feature_name"][lineage_index].as_py() == name,
                    f"Canary cache lineage order/key differs: {receipt.family}/{name}",
                )
                values_map[name] = value_columns[name][index].as_py()
                lineage_map[name] = FeatureLineage(
                    lineage_columns["max_observation_date"][lineage_index].as_py(),
                    lineage_columns["max_public_availability_date"][lineage_index].as_py(),
                    lineage_columns["stale_after_date"][lineage_index].as_py(),
                    lineage_columns["value_status"][lineage_index].as_py(),
                )
            yield FeatureMaterializationRow(
                signal_date=signal,
                permno=permno,
                permco=value_columns["permco"][index].as_py(),
                long_eligible=value_columns["long_eligible"][index].as_py(),
                short_eligible=value_columns["short_eligible"][index].as_py(),
                primary_exchange=value_columns["primary_exchange"][index].as_py(),
                share_type=value_columns["share_type"][index].as_py(),
                result=FeatureFamilyResult(
                    values=MappingProxyType(values_map),
                    lineage=MappingProxyType(lineage_map),
                ),
            )


def _signal_groups(
    rows: Iterable[FeatureMaterializationRow],
) -> Iterator[tuple[dt.date, tuple[FeatureMaterializationRow, ...]]]:
    active: dt.date | None = None
    buffered: list[FeatureMaterializationRow] = []
    for row in rows:
        if active is None:
            active = row.signal_date
        elif row.signal_date != active:
            _check(row.signal_date > active, "Canary signal order regressed")
            yield active, tuple(buffered)
            active = row.signal_date
            buffered = []
        buffered.append(row)
    if active is not None:
        yield active, tuple(buffered)


class _DevelopmentCacheCanaryRouter:
    """Cache-backed router implementing the exact interface used by Stage 11."""

    def __init__(
        self,
        *,
        factories: Mapping[str, Any],
        row_count: int,
    ) -> None:
        self._factories = factories
        self.row_count = row_count

    def iter_family_row_batches(
        self,
        *,
        batch_size: int,
    ) -> Iterator[CrspFamilyRowBatch]:
        grouped = {
            family: iter(_signal_groups(factory()))
            for family, factory in self._factories.items()
        }
        while True:
            current: dict[str, tuple[dt.date, tuple[FeatureMaterializationRow, ...]]] = {}
            ended: set[str] = set()
            for family, stream in grouped.items():
                try:
                    current[family] = next(stream)
                except StopIteration:
                    ended.add(family)
            if ended:
                _check(
                    len(ended) == len(grouped),
                    "Canary CRSP family streams ended at different points",
                )
                return
            signals = {value[0] for value in current.values()}
            _check(len(signals) == 1, "Canary CRSP routed signal dates differ")
            signal = next(iter(signals))
            for family in CRSP_FAMILIES:
                rows = current[family][1]
                _check(bool(rows), f"Canary CRSP signal is empty: {family}/{signal}")
                for ordinal, offset in enumerate(range(0, len(rows), batch_size)):
                    yield CrspFamilyRowBatch(
                        signal_date=signal,
                        family=family,
                        batch_ordinal=ordinal,
                        rows=rows[offset : offset + batch_size],
                    )


def _selected_mask(
    batch: pa.RecordBatch,
    ranges: Sequence[tuple[tuple[dt.date, int], tuple[dt.date, int]]],
) -> pa.Array:
    signal = batch.column(batch.schema.get_field_index("signal_date"))
    permno = batch.column(batch.schema.get_field_index("permno"))
    selected: pa.Array | None = None
    for first, last in ranges:
        lower = pc.or_(
            pc.greater(signal, pa.scalar(first[0])),
            pc.and_(pc.equal(signal, pa.scalar(first[0])), pc.greater_equal(permno, first[1])),
        )
        upper = pc.or_(
            pc.less(signal, pa.scalar(last[0])),
            pc.and_(pc.equal(signal, pa.scalar(last[0])), pc.less_equal(permno, last[1])),
        )
        within = pc.and_(lower, upper)
        selected = within if selected is None else pc.or_(selected, within)
    assert selected is not None
    return selected


def _selected_published_batches(
    path: Path,
    ranges: Sequence[tuple[tuple[dt.date, int], tuple[dt.date, int]]],
) -> Iterator[pa.RecordBatch]:
    for batch in _parquet_batches((path,)):
        selected = batch.filter(_selected_mask(batch, ranges))
        if selected.num_rows:
            yield selected


class _ExactBatchCursor:
    def __init__(self, batches: Iterable[pa.RecordBatch], label: str) -> None:
        self._batches = iter(batches)
        self._active: pa.RecordBatch | None = None
        self._offset = 0
        self._label = label

    def take(self, rows: int) -> pa.RecordBatch:
        _check(rows > 0, f"Canary requested an empty reference batch: {self._label}")
        parts: list[pa.RecordBatch] = []
        remaining = rows
        while remaining:
            if self._active is None:
                try:
                    self._active = next(self._batches)
                    self._offset = 0
                except StopIteration as exc:
                    raise ValidationDevelopmentParityError(
                        f"Canary reference ended early: {self._label}"
                    ) from exc
            available = self._active.num_rows - self._offset
            width = min(available, remaining)
            parts.append(self._active.slice(self._offset, width))
            self._offset += width
            remaining -= width
            if self._offset == self._active.num_rows:
                self._active = None
        table = pa.Table.from_batches(parts).combine_chunks()
        batches = table.to_batches(max_chunksize=rows)
        _check(len(batches) == 1 and batches[0].num_rows == rows, "Canary cursor assembly differs")
        return batches[0]

    def require_exhausted(self) -> None:
        if self._active is not None and self._offset < self._active.num_rows:
            raise ValidationDevelopmentParityError(f"Canary reference has extra rows: {self._label}")
        try:
            extra = next(self._batches)
        except StopIteration:
            return
        _check(extra.num_rows == 0, f"Canary reference has extra rows: {self._label}")


def _actual_stage11_orchestration_canary(
    *,
    completions: Mapping[str, FamilyCompletionReceipt],
    contract: ValidationDataContract,
    output_root: Path,
) -> dict[str, Any]:
    """Run selected full cache ranges through actual Stage 11 Python assembly."""

    import portfolio_bot.validation.validation_data_candidate_builder as candidate
    import portfolio_bot.validation.validation_empirical_feature_stream as wrapper

    canonical = completions["price_trends_path"]
    _check(canonical.batch_count == 64, "Canary canonical family batch count differs")
    selected_ordinals = (0, 38, canonical.batch_count - 1)
    selected_summaries = tuple(canonical.batches[index] for index in selected_ordinals)
    ranges = tuple((summary.first_key, summary.last_key) for summary in selected_summaries)
    _check(
        selected_summaries[0].first_key[0] == dt.date(2000, 1, 31)
        and selected_summaries[1].first_key[0] <= dt.date(2008, 9, 30)
        <= selected_summaries[1].last_key[0]
        and selected_summaries[-1].last_key[0] == dt.date(2014, 10, 31),
        "Canary regimes do not cover first/crisis/last boundaries",
    )
    expected_rows = sum(summary.row_count for summary in selected_summaries)

    def factory(family: str):
        return lambda: _cached_family_rows(
            completions[family],
            feature_names=contract.family_feature_names[family],
            lineage_schema=contract.lineage_schema,
            ranges=ranges,
        )

    router = _DevelopmentCacheCanaryRouter(
        factories={family: factory(family) for family in CRSP_FAMILIES},
        row_count=expected_rows,
    )
    raw_reference = _ExactBatchCursor(
        _selected_published_batches(output_root / "feature_store_raw.parquet", ranges),
        "published raw",
    )
    lineage_references = {
        family: _ExactBatchCursor(
            _selected_published_batches(
                output_root / f"feature_lineage_{family}.parquet",
                ranges,
            ),
            f"published lineage {family}",
        )
        for family in contract.family_feature_names
    }
    permitted_signals: set[dt.date] = set()
    rows_compared = 0
    lineage_compared = 0
    batches_compared = 0
    aligned = wrapper._aligned_scored_batches(
        router=router,
        fundamental_rows=factory("valuation_fundamentals")(),
        ownership_rows=factory("ownership_filings_events_segments")(),
        optionmetrics_rows=factory("optionmetrics")(),
        aggregate_rows=factory("aggregate_state")(),
        batch_size=2_048,
    )
    for rows in aligned:
        permitted_signals.update(item.signal_date for item in rows)
        feature, lineages, _dates, _long_rows, _segments = candidate._materialize_feature_chunk(
            rows,
            contract,
            allowed_signal_dates=frozenset(permitted_signals),
        )
        observed_raw = feature.combine_chunks().to_batches(max_chunksize=feature.num_rows)[0]
        expected_raw = raw_reference.take(feature.num_rows)
        _compare_columns_exact(
            observed_raw,
            expected_raw,
            names=contract.raw_feature_schema.names,
            label="actual_stage11_canary/raw",
            row_offset=rows_compared,
        )
        for family in contract.family_feature_names:
            observed_table = lineages[family].combine_chunks()
            observed = observed_table.to_batches(max_chunksize=observed_table.num_rows)[0]
            expected = lineage_references[family].take(observed.num_rows)
            _compare_columns_exact(
                observed,
                expected,
                names=contract.lineage_schema.names,
                label=f"actual_stage11_canary/lineage/{family}",
                row_offset=lineage_compared,
            )
            lineage_compared += observed.num_rows
        rows_compared += feature.num_rows
        batches_compared += 1
    raw_reference.require_exhausted()
    for cursor in lineage_references.values():
        cursor.require_exhausted()
    _check(
        rows_compared == expected_rows
        and lineage_compared == expected_rows * len(contract.feature_names),
        "Actual Stage 11 canary coverage differs",
    )
    return {
        "method": "actual_wrapper_aligned_scored_batches_plus_candidate_materialize_feature_chunk",
        "canonical_whole_cache_batch_ordinals": list(selected_ordinals),
        "regimes": [
            {
                "first_signal_date": summary.first_key[0].isoformat(),
                "first_permno": summary.first_key[1],
                "last_signal_date": summary.last_key[0].isoformat(),
                "last_permno": summary.last_key[1],
                "rows": summary.row_count,
            }
            for summary in selected_summaries
        ],
        "first_crisis_last_boundary_coverage": True,
        "raw_rows_compared": rows_compared,
        "raw_fields_compared": len(contract.raw_feature_schema),
        "lineage_rows_compared": lineage_compared,
        "assembly_batches_compared": batches_compared,
        "family_order": list(wrapper._FAMILY_ORDER),
        "difference_count": 0,
        "memory_mode": "bounded_2048_row_candidate_chunks",
    }


def prove_development_code_identity_parity(
    contract: ValidationDataContract | None = None,
) -> ValidatedDevelopmentParity:
    """Rehash every relevant byte and issue one opaque parity authority."""

    current = validate_validation_data_contract()
    if contract is not None:
        _check(type(contract) is ValidationDataContract, "Wrong validation-data contract type")
        _check(contract.policy_sha256 == current.policy_sha256, "Stale validation-data contract")
    contract = current
    rules = contract.rules["development_parity"]
    build_id = str(rules["reference_build_id"])
    verification_id = str(rules["reference_verification_id"])

    pointer_expected = contract.frozen_bindings[
        "artifacts/phase_03_features/LATEST_FEATURE_STORE_VERIFIED.json"
    ]
    pointer_sha = sha256_file(STAGE6_POINTER)
    _check(pointer_sha == pointer_expected, "Stage 6 verified pointer hash differs")
    pointer = _json(STAGE6_POINTER, "Stage 6 verified pointer")
    _check(
        pointer.get("status") == "VERIFIED_PASSED"
        and pointer.get("build_id") == build_id
        and pointer.get("verification_id") == verification_id,
        "Stage 6 independent authority differs",
    )

    manifest_path = STAGE6_MANIFEST_PARENT / build_id / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    _check(
        manifest_sha == pointer.get("primary_manifest_sha256"),
        "Stage 6 pointer does not bind the primary manifest",
    )
    manifest = _json(manifest_path, "Stage 6 primary manifest")
    _check(
        manifest.get("build_id") == build_id
        and manifest.get("status") == "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION",
        "Stage 6 primary identity differs",
    )

    # The resolver performs one complete byte-hash pass.  The identity factory
    # deliberately performs a second independent pass before calculating the
    # content fingerprint.  This redundancy makes the receipt wording
    # unambiguous and also catches mutation between the two passes.
    inputs = resolve_input_catalog(authenticate_hashes=True)
    identity = compute_feature_store_build_identity(inputs)
    _check(identity.build_id == build_id, "Current Stage 6 input/code identity differs")
    bindings = manifest.get("bindings", {})
    _check(
        identity.content_fingerprint == bindings.get("content_fingerprint")
        and identity.configuration_hash_map == bindings.get("configuration_hashes")
        and identity.code_hash_map == bindings.get("code_hashes")
        and identity.input_receipts_sha256 == bindings.get("input_receipts_sha256"),
        "Recomputed Stage 6 identity differs from the verified manifest",
    )
    _check(
        [item.as_dict() for item in identity.input_receipts]
        == manifest.get("inputs", {}).get("table_receipts"),
        "Stage 6 input receipts differ from the verified manifest",
    )

    output = manifest.get("outputs", {}).get("interim", {})
    relative_root = output.get("path")
    _check(isinstance(relative_root, str), "Stage 6 interim output path is absent")
    output_root = Path(os.path.abspath(inputs.database_root / relative_root))
    _check(output_root.is_dir() and not output_root.is_symlink(), "Unsafe Stage 6 output root")
    declared = output.get("files")
    _check(isinstance(declared, dict), "Stage 6 output receipts are absent")
    expected_data = {"feature_store_raw.parquet"} | {
        f"feature_lineage_{family}.parquet" for family in contract.family_feature_names
    }
    _check(expected_data.issubset(declared), "Stage 6 predictor/lineage receipt set differs")
    observed_files = {path.name for path in output_root.iterdir() if path.is_file()}
    _check(observed_files == set(declared), "Stage 6 interim directory is not closed")

    authenticated_outputs: dict[str, Any] = {}
    lineage_rows = 0
    output_bytes = 0
    for name in sorted(declared):
        receipt = declared[name]
        _check(isinstance(receipt, dict) and receipt.get("path") == name, f"Bad Stage 6 receipt: {name}")
        path = _ordinary(output_root / name, output_root, f"Stage 6 output {name}")
        digest = sha256_file(path)
        size = path.stat(follow_symlinks=False).st_size
        _check(digest == receipt.get("sha256") and size == receipt.get("size_bytes"), f"Stage 6 output changed: {name}")
        entry: dict[str, Any] = {"sha256": digest, "size_bytes": size}
        output_bytes += size
        if name.endswith(".parquet"):
            parquet = pq.ParquetFile(path)
            _check(
                parquet.metadata.num_rows == receipt.get("rows")
                and parquet.num_row_groups == receipt.get("row_groups")
                and schema_sha256(parquet.schema_arrow) == receipt.get("schema_sha256"),
                f"Stage 6 Parquet receipt differs: {name}",
            )
            entry.update(
                {
                    "rows": parquet.metadata.num_rows,
                    "row_groups": parquet.num_row_groups,
                    "schema_sha256": schema_sha256(parquet.schema_arrow),
                }
            )
            if name in expected_data:
                first, last = _signal_bounds(path)
                entry["first_signal_date"] = first
                entry["last_signal_date"] = last
            if name.startswith("feature_lineage_"):
                lineage_rows += parquet.metadata.num_rows
        authenticated_outputs[name] = entry

    raw_rows = int(authenticated_outputs["feature_store_raw.parquet"]["rows"])
    _check(
        raw_rows == int(rules["reference_feature_rows"])
        and lineage_rows == int(rules["reference_lineage_rows"])
        and lineage_rows == raw_rows * len(contract.feature_names),
        "Authenticated Stage 6 reference coverage differs",
    )
    _check(
        authenticated_outputs["feature_store_raw.parquet"]["schema_sha256"]
        == schema_sha256(contract.raw_feature_schema),
        "Stage 6 and Stage 11 raw schemas differ",
    )

    completions = {
        family: authenticate_family_completion(
            inputs=inputs,
            identity=identity,
            family=family,
        )
        for family in contract.family_feature_names
    }
    full_replay = _full_development_streaming_replay(
        completions=MappingProxyType(completions),
        contract=contract,
        output_root=output_root,
    )
    actual_stage11_canary = _actual_stage11_orchestration_canary(
        completions=MappingProxyType(completions),
        contract=contract,
        output_root=output_root,
    )
    _check(
        full_replay["raw_rows_compared"] == raw_rows
        and full_replay["lineage_rows_compared"] == lineage_rows
        and full_replay["difference_count"] == 0
        and actual_stage11_canary["difference_count"] == 0,
        "Development replay evidence is incomplete",
    )

    scope_before = _development_scope_values()
    _check(scope_before == _EXPECTED_DEVELOPMENT_SCOPE, "Development formula scope differs")
    import portfolio_bot.validation.validation_empirical_feature_stream as wrapper

    with wrapper._development_formula_scope_identity():
        scope_inside = _development_scope_values()
        _check(scope_inside == scope_before, "Development scope adapter is not a no-op")
    _check(_development_scope_values() == scope_before, "Development scope was not restored")
    function_bindings = _function_binding_evidence()
    _check(
        tuple(function_bindings["family_order"]) == tuple(contract.family_feature_names),
        "Stage 11 contract family order differs from the generalized wrapper",
    )

    input_bytes = sum(item.size_bytes for item in identity.input_receipts)
    evidence = {
        "access_counters": {
            "validation_label_rows_read": 0,
            "validation_outcome_rows_read": 0,
            "final_period_rows_read": 0,
            "portfolio_trade_or_pnl_rows_created": 0,
        },
        "development_scope": {
            "adapter_executed": True,
            "exact_noop": True,
            "scope_values": scope_before,
            "scope_values_sha256": _digest(scope_before),
        },
        "generalized_wrapper": function_bindings,
        "full_development_logical_replay": full_replay,
        "actual_stage11_orchestration_canary": actual_stage11_canary,
        "reference_outputs": {
            "all_declared_bytes_reauthenticated": True,
            "files": authenticated_outputs,
            "output_bytes_hashed": output_bytes,
            "raw_schema_equals_stage11_schema": True,
        },
        "stage6_identity": {
            "all_input_bytes_reauthenticated": True,
            "full_input_byte_hash_passes": 2,
            "code_hashes_sha256": _digest(identity.code_hash_map),
            "configuration_hashes_sha256": _digest(identity.configuration_hash_map),
            "content_fingerprint": identity.content_fingerprint,
            "input_bytes_per_hash_pass": input_bytes,
            "total_input_bytes_hashed": input_bytes * 2,
            "input_receipts_sha256": identity.input_receipts_sha256,
        },
        "stage6_primary_manifest": {
            "path": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": manifest_sha,
        },
        "stage6_verified_pointer": {
            "path": STAGE6_POINTER.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": pointer_sha,
            "status": pointer["status"],
        },
    }
    evidence_sha = _digest(evidence)
    receipt = DevelopmentParityReceipt(
        schema_version=PARITY_RECEIPT_SCHEMA_VERSION,
        status=PARITY_STATUS,
        method=PARITY_METHOD,
        feature_rows=raw_rows,
        lineage_rows=lineage_rows,
        empirical_replay_feature_rows=raw_rows,
        empirical_replay_lineage_rows=lineage_rows,
        identity_mismatch_count=0,
        empirical_difference_count=0,
        reference_build_id=build_id,
        reference_verification_id=verification_id,
        evidence_sha256=evidence_sha,
        evidence=MappingProxyType(evidence),
    )
    return ValidatedDevelopmentParity(receipt, _AUTHORITY_SEAL)


def write_development_parity_receipt(
    authority: ValidatedDevelopmentParity,
    destination: Path,
) -> Path:
    _check(type(authority) is ValidatedDevelopmentParity, "Wrong parity authority type")
    _check(authority._seal is _AUTHORITY_SEAL, "Unsealed parity authority")
    path = Path(os.path.abspath(destination.expanduser()))
    _check(path.parent.is_dir() and not path.parent.is_symlink(), "Unsafe parity receipt directory")
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(authority.receipt.as_dict()))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValidationDevelopmentParityError("Parity receipt destination already exists") from exc
    return path


def load_and_reauthenticate_development_parity(
    path: Path,
    contract: ValidationDataContract | None = None,
) -> ValidatedDevelopmentParity:
    """Load canonical evidence, then independently reproduce it from bytes."""

    supplied = _json(path, "development parity receipt")
    _check(path.read_bytes() == canonical_json_bytes(supplied), "Parity receipt is not canonical JSON")
    observed = prove_development_code_identity_parity(contract)
    _check(supplied == observed.receipt.as_dict(), "Development parity evidence differs")
    return observed


__all__ = [
    "DevelopmentParityReceipt",
    "PARITY_METHOD",
    "PARITY_STATUS",
    "ValidatedDevelopmentParity",
    "ValidationDevelopmentParityError",
    "load_and_reauthenticate_development_parity",
    "prove_development_code_identity_parity",
    "write_development_parity_receipt",
]
