"""Deterministic Arrow materialization for the aggregate-state feature family.

This module performs no filesystem writes.  It reads only paths registered by
``FeatureStoreInputs``, computes one aggregate-state result per distinct signal,
and exposes repeatable Arrow batch streams for the future Stage 6 writer.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.features.aggregate_state_features import (
    AGGREGATE_STATE_FEATURE_NAMES,
    AggregateStateResult,
    compute_aggregate_state_features,
)
from portfolio_bot.features.feature_store_inputs import FeatureStoreInputs
from portfolio_bot.features.feature_store_materialization import (
    FeatureMaterializationRow,
)


PREDICTOR_END_DATE = dt.date(2014, 10, 31)
CONTEXT_END_DATE = dt.date(2014, 12, 1)

SPINE_TABLE = "crsp_signal_date_universe.parquet"
MONTHLY_FACTOR_TABLE = "ff_factor_monthly_observation.parquet"
DAILY_FACTOR_TABLE = "ff_factor_daily_observation.parquet"
PASTOR_STAMBAUGH_TABLE = "ps_liquidity_observation.parquet"

SPINE_SCHEMA = pa.schema(
    [
        pa.field("signal_date", pa.date32(), nullable=False),
        pa.field("permno", pa.int64(), nullable=False),
        pa.field("permco", pa.int64(), nullable=False),
        pa.field("long_eligible", pa.bool_(), nullable=False),
        pa.field("short_eligible", pa.bool_(), nullable=False),
        pa.field("primary_exchange", pa.string(), nullable=False),
        pa.field("share_type", pa.string(), nullable=False),
    ]
)
FAMILY_SCHEMA = pa.schema(
    [
        *SPINE_SCHEMA,
        *(
            pa.field(name, pa.float64(), nullable=True)
            for name in AGGREGATE_STATE_FEATURE_NAMES
        ),
    ]
)
LINEAGE_SCHEMA = pa.schema(
    [
        pa.field("signal_date", pa.date32(), nullable=False),
        pa.field("permno", pa.int64(), nullable=False),
        pa.field("feature_name", pa.string(), nullable=False),
        pa.field("max_observation_date", pa.date32(), nullable=True),
        pa.field("max_public_availability_date", pa.date32(), nullable=True),
        pa.field("stale_after_date", pa.date32(), nullable=True),
        pa.field("value_status", pa.string(), nullable=False),
    ]
)


class AggregateStateMaterializationError(RuntimeError):
    """Raised when authenticated inputs cannot satisfy the family contract."""


@dataclass(frozen=True)
class AggregateStateMaterialization:
    """Repeatable Arrow batch streams backed by compact in-memory spine keys."""

    spine_keys: pa.Table
    signal_results: Mapping[dt.date, AggregateStateResult]
    source_context_rows: Mapping[str, int]
    batch_size: int

    @property
    def spine_row_count(self) -> int:
        return self.spine_keys.num_rows

    @property
    def signal_result_count(self) -> int:
        return len(self.signal_results)

    @property
    def lineage_row_count(self) -> int:
        return self.spine_row_count * len(AGGREGATE_STATE_FEATURE_NAMES)

    def iter_family_batches(self) -> Iterator[pa.RecordBatch]:
        """Yield wide family batches in exact authenticated spine order."""

        for spine_batch in self.spine_keys.to_batches(
            max_chunksize=self.batch_size
        ):
            signals = spine_batch.column(0).to_pylist()
            arrays: list[pa.Array] = [
                *spine_batch.columns,
            ]
            for name in AGGREGATE_STATE_FEATURE_NAMES:
                arrays.append(
                    pa.array(
                        [
                            self.signal_results[signal].values[name]
                            for signal in signals
                        ],
                        type=pa.float64(),
                    )
                )
            yield pa.RecordBatch.from_arrays(arrays, schema=FAMILY_SCHEMA)

    def iter_lineage_batches(self) -> Iterator[pa.RecordBatch]:
        """Yield long lineage batches in spine-major, registry-minor order."""

        for spine_batch in self.spine_keys.to_batches(
            max_chunksize=self.batch_size
        ):
            signals = spine_batch.column(0).to_pylist()
            permnos = spine_batch.column(1).to_pylist()
            signal_values: list[dt.date] = []
            permno_values: list[int] = []
            feature_names: list[str] = []
            observation_dates: list[dt.date | None] = []
            availability_dates: list[dt.date | None] = []
            stale_dates: list[dt.date | None] = []
            statuses: list[str] = []
            for signal, permno in zip(signals, permnos, strict=True):
                result = self.signal_results[signal]
                for name in AGGREGATE_STATE_FEATURE_NAMES:
                    item = result.lineage[name]
                    signal_values.append(signal)
                    permno_values.append(permno)
                    feature_names.append(name)
                    observation_dates.append(item.max_observation_date)
                    availability_dates.append(
                        item.max_public_availability_date
                    )
                    stale_dates.append(item.stale_after_date)
                    statuses.append(item.value_status)
            arrays = [
                pa.array(signal_values, type=pa.date32()),
                pa.array(permno_values, type=pa.int64()),
                pa.array(feature_names, type=pa.string()),
                pa.array(observation_dates, type=pa.date32()),
                pa.array(availability_dates, type=pa.date32()),
                pa.array(stale_dates, type=pa.date32()),
                pa.array(statuses, type=pa.string()),
            ]
            yield pa.RecordBatch.from_arrays(arrays, schema=LINEAGE_SCHEMA)

    def iter_materialization_row_batches(
        self,
    ) -> Iterator[tuple[FeatureMaterializationRow, ...]]:
        """Yield common-writer rows without collecting the wide family table."""

        for spine_batch in self.spine_keys.to_batches(
            max_chunksize=self.batch_size
        ):
            columns = [
                spine_batch.column(index).to_pylist()
                for index in range(len(SPINE_SCHEMA))
            ]
            rows: list[FeatureMaterializationRow] = []
            for values in zip(*columns, strict=True):
                (
                    signal,
                    permno,
                    permco,
                    long_eligible,
                    short_eligible,
                    primary_exchange,
                    share_type,
                ) = values
                rows.append(
                    FeatureMaterializationRow(
                        signal_date=signal,
                        permno=permno,
                        permco=permco,
                        long_eligible=long_eligible,
                        short_eligible=short_eligible,
                        primary_exchange=primary_exchange,
                        share_type=share_type,
                        result=self.signal_results[signal],
                    )
                )
            yield tuple(rows)

    def collect_family_table(self) -> pa.Table:
        """Collect the wide stream; intended for bounded tests and adapters."""

        return pa.Table.from_batches(
            list(self.iter_family_batches()), schema=FAMILY_SCHEMA
        )

    def collect_lineage_table(self) -> pa.Table:
        """Collect the lineage stream; intended for bounded tests and adapters."""

        return pa.Table.from_batches(
            list(self.iter_lineage_batches()), schema=LINEAGE_SCHEMA
        )


def aggregate_state_arrow_batches(
    materialization: AggregateStateMaterialization,
) -> tuple[Iterator[pa.RecordBatch], Iterator[pa.RecordBatch]]:
    """Adapter consumed by the forthcoming common family writer."""

    return (
        materialization.iter_family_batches(),
        materialization.iter_lineage_batches(),
    )


def _verified_parquet(
    inputs: FeatureStoreInputs, name: str
) -> tuple[Path, pq.ParquetFile]:
    try:
        receipt = inputs.tables[name]
    except KeyError as exc:
        raise AggregateStateMaterializationError(
            f"Required authenticated input is absent: {name}"
        ) from exc
    path = inputs.path(name)
    if path != receipt.path or not path.is_file() or path.is_symlink():
        raise AggregateStateMaterializationError(
            f"Authenticated input path is unavailable: {name}"
        )
    if path.stat().st_size != receipt.size_bytes:
        raise AggregateStateMaterializationError(
            f"Authenticated input size changed: {name}"
        )
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != receipt.rows:
        raise AggregateStateMaterializationError(
            f"Authenticated input row count changed: {name}"
        )
    return path, parquet


def _load_spine(inputs: FeatureStoreInputs) -> tuple[pa.Table, tuple[dt.date, ...]]:
    path, _ = _verified_parquet(inputs, SPINE_TABLE)
    try:
        spine = pq.read_table(path, columns=SPINE_SCHEMA.names).cast(
            SPINE_SCHEMA
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, KeyError) as exc:
        raise AggregateStateMaterializationError(
            "Signal spine does not satisfy the required key schema"
        ) from exc
    if spine.num_rows == 0:
        raise AggregateStateMaterializationError("Signal spine is empty")
    if any(column.null_count for column in spine.columns):
        raise AggregateStateMaterializationError("Signal spine key is null")

    signals: list[dt.date] = []
    previous: tuple[dt.date, int] | None = None
    for batch in spine.to_batches(max_chunksize=65_536):
        dates = batch.column(0).to_pylist()
        permnos = batch.column(1).to_pylist()
        for signal, permno in zip(dates, permnos, strict=True):
            if signal > PREDICTOR_END_DATE:
                raise AggregateStateMaterializationError(
                    "Predictor signal exceeds 2014-10-31"
                )
            key = (signal, permno)
            if previous is not None and key <= previous:
                raise AggregateStateMaterializationError(
                    "Signal spine is not strictly sorted and unique"
                )
            if not signals or signal != signals[-1]:
                signals.append(signal)
            previous = key
    return spine, tuple(signals)


def _load_context_rows(
    inputs: FeatureStoreInputs,
    name: str,
    columns: list[str],
) -> list[dict[str, object]]:
    path, _ = _verified_parquet(inputs, name)
    try:
        table = pq.read_table(
            path,
            columns=columns,
            filters=[
                ("observation_date", "<=", CONTEXT_END_DATE),
                ("public_availability_date", "<=", CONTEXT_END_DATE),
            ],
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, KeyError) as exc:
        raise AggregateStateMaterializationError(
            f"Aggregate context table has an invalid schema: {name}"
        ) from exc
    rows = table.to_pylist()
    if any(
        row["observation_date"] > CONTEXT_END_DATE
        or row["public_availability_date"] > CONTEXT_END_DATE
        for row in rows
    ):
        raise AggregateStateMaterializationError(
            f"Context cutoff was not enforced: {name}"
        )
    return rows


def materialize_aggregate_state(
    inputs: FeatureStoreInputs,
    *,
    batch_size: int = 8_192,
) -> AggregateStateMaterialization:
    """Build one result per signal and return repeatable Arrow batch streams."""

    if not isinstance(inputs, FeatureStoreInputs):
        raise AggregateStateMaterializationError(
            "Aggregate materialization requires FeatureStoreInputs"
        )
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise AggregateStateMaterializationError(
            "batch_size must be a positive integer"
        )
    spine, signals = _load_spine(inputs)
    monthly = _load_context_rows(
        inputs,
        MONTHLY_FACTOR_TABLE,
        [
            "factor_month",
            "observation_date",
            "public_availability_date",
            "mktrf",
            "smb",
            "hml",
            "rmw",
            "cma",
            "umd",
            "rf",
        ],
    )
    daily = _load_context_rows(
        inputs,
        DAILY_FACTOR_TABLE,
        ["observation_date", "public_availability_date", "mktrf"],
    )
    pastor_stambaugh = _load_context_rows(
        inputs,
        PASTOR_STAMBAUGH_TABLE,
        [
            "observation_month",
            "observation_date",
            "public_availability_date",
            "ps_level",
        ],
    )
    results: dict[dt.date, AggregateStateResult] = {}
    for signal in signals:
        computed = compute_aggregate_state_features(
            monthly,
            daily,
            pastor_stambaugh,
            signal,
        )
        for name in AGGREGATE_STATE_FEATURE_NAMES:
            item = computed.lineage[name]
            if (
                item.max_observation_date is not None
                and item.max_observation_date > signal
            ) or (
                item.max_public_availability_date is not None
                and item.max_public_availability_date > signal
            ):
                raise AggregateStateMaterializationError(
                    f"Future aggregate predictor escaped for {signal}: {name}"
                )
        results[signal] = AggregateStateResult(
            values=MappingProxyType(dict(computed.values)),
            lineage=MappingProxyType(dict(computed.lineage)),
        )
    return AggregateStateMaterialization(
        spine_keys=spine,
        signal_results=MappingProxyType(results),
        source_context_rows=MappingProxyType(
            {
                MONTHLY_FACTOR_TABLE: len(monthly),
                DAILY_FACTOR_TABLE: len(daily),
                PASTOR_STAMBAUGH_TABLE: len(pastor_stambaugh),
            }
        ),
        batch_size=batch_size,
    )
