"""Single-scan CRSP history router for Stage 6.3.

The router emits unfinalized security-signal rows in ``(permno, signal_date)``
order. Cross-sectional beta clipping and spread imputation are deliberately
performed in a later signal-sorted pass.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.features.feature_store_inputs import FeatureStoreInputs
from portfolio_bot.features.materialize_crsp import (
    DEVELOPMENT_LAST_SIGNAL,
    MAXIMUM_CONTEXT_DATE,
    CrspMaterializedRow,
    compute_unfinalized_crsp_security_rows,
)


SPINE_TABLE = "crsp_signal_date_universe.parquet"
MARKET_TABLE = "crsp_security_day.parquet"
RETURN_TABLE = "crsp_security_day_return.parquet"
SHARES_TABLE = "crsp_shares_interval.parquet"
CALENDAR_TABLE = "trading_calendar.parquet"
FACTOR_TABLE = "ff_factor_daily_observation.parquet"

SPINE_COLUMNS = (
    "signal_date",
    "permno",
    "permco",
    "long_eligible",
    "short_eligible",
    "primary_exchange",
    "share_type",
)
MARKET_COLUMNS = (
    "permno",
    "calendar_date",
    "source_primary_exchange",
    "conditional_type",
    "source_trading_status_flag",
    "source_security_type",
    "source_security_subtype",
    "source_share_type",
    "close_usd_per_share",
    "market_cap_usd",
    "volume_shares",
    "low_usd_per_share",
    "high_usd_per_share",
    "bid_usd_per_share",
    "ask_usd_per_share",
    "screening_dollar_volume_usd",
)
RETURN_COLUMNS = (
    "permno",
    "calendar_date",
    "return_span_start_date",
    "return_span_end_date",
    "mapped_trading_periods",
    "covered_trading_sessions",
    "return_leg_class",
    "raw_return_ex_distributions",
    "authoritative_total_return",
    "outcome_availability_date",
    "authoritative_return_state",
    "return_history_usable",
)
SHARES_COLUMNS = (
    "permno",
    "start_date",
    "end_date",
    "shares_outstanding",
)
FACTOR_COLUMNS = (
    "observation_date",
    "public_availability_date",
    "mktrf",
    "smb",
    "hml",
    "rmw",
    "cma",
    "rf",
    "umd",
)


class CrspStreamingMaterializationError(RuntimeError):
    """Raised when authenticated CRSP inputs cannot be streamed canonically."""


def _path(inputs: FeatureStoreInputs, name: str) -> Path:
    try:
        receipt = inputs.tables[name]
    except KeyError as exc:
        raise CrspStreamingMaterializationError(
            f"Required CRSP materialization input is absent: {name}"
        ) from exc
    path = inputs.path(name)
    if (
        path != receipt.path
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != receipt.size_bytes
    ):
        raise CrspStreamingMaterializationError(
            f"Authenticated CRSP input path changed: {name}"
        )
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != receipt.rows:
        raise CrspStreamingMaterializationError(
            f"Authenticated CRSP input row count changed: {name}"
        )
    return path


def _sorted_spine(inputs: FeatureStoreInputs) -> pa.Table:
    table = pq.read_table(
        _path(inputs, SPINE_TABLE), columns=list(SPINE_COLUMNS)
    )
    if table.num_rows == 0 or any(column.null_count for column in table.columns):
        raise CrspStreamingMaterializationError("Signal spine is empty or null")
    signals = table["signal_date"]
    if pc.max(signals).as_py() > DEVELOPMENT_LAST_SIGNAL:
        raise CrspStreamingMaterializationError(
            "Signal spine escaped development cutoff"
        )
    order = pc.sort_indices(
        table,
        sort_keys=[("permno", "ascending"), ("signal_date", "ascending")],
    )
    return table.take(order)


def _table_groups(
    table: pa.Table,
    key: str,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    current: int | None = None
    rows: list[dict[str, Any]] = []
    for batch in table.to_batches(max_chunksize=65_536):
        for row in batch.to_pylist():
            value = int(row[key])
            if current is None:
                current = value
            elif value != current:
                if value < current:
                    raise CrspStreamingMaterializationError(
                        f"Table is not sorted by {key}"
                    )
                yield current, rows
                current = value
                rows = []
            rows.append(row)
    if current is not None:
        yield current, rows


def _parquet_groups(
    path: Path,
    *,
    columns: Sequence[str],
    key: str,
    batch_size: int = 65_536,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    current: int | None = None
    rows: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=batch_size,
        columns=columns,
        use_threads=False,
    ):
        for row in batch.to_pylist():
            value = int(row[key])
            if current is None:
                current = value
            elif value != current:
                if value < current:
                    raise CrspStreamingMaterializationError(
                        f"Parquet is not sorted by {key}: {path.name}"
                    )
                yield current, rows
                current = value
                rows = []
            rows.append(row)
    if current is not None:
        yield current, rows


def _aligned_market_return_groups(
    market_path: Path,
    return_path: Path,
    *,
    batch_size: int = 65_536,
) -> Iterator[
    tuple[int, list[dict[str, Any]], list[dict[str, Any]]]
]:
    market_batches = pq.ParquetFile(market_path).iter_batches(
        batch_size=batch_size,
        columns=list(MARKET_COLUMNS),
        use_threads=False,
    )
    return_batches = pq.ParquetFile(return_path).iter_batches(
        batch_size=batch_size,
        columns=list(RETURN_COLUMNS),
        use_threads=False,
    )
    current: int | None = None
    market_rows: list[dict[str, Any]] = []
    return_rows: list[dict[str, Any]] = []
    market_count = return_count = 0
    while True:
        try:
            market_batch = next(market_batches)
        except StopIteration:
            market_batch = None
        try:
            return_batch = next(return_batches)
        except StopIteration:
            return_batch = None
        if market_batch is None or return_batch is None:
            if market_batch is not None or return_batch is not None:
                raise CrspStreamingMaterializationError(
                    "Market and return streams have different batch counts"
                )
            break
        if market_batch.num_rows != return_batch.num_rows:
            raise CrspStreamingMaterializationError(
                "Market and return streams have different batch sizes"
            )
        market_count += market_batch.num_rows
        return_count += return_batch.num_rows
        for market, returns in zip(
            market_batch.to_pylist(),
            return_batch.to_pylist(),
            strict=True,
        ):
            market_key = (int(market["permno"]), market["calendar_date"])
            return_key = (int(returns["permno"]), returns["calendar_date"])
            if market_key != return_key:
                raise CrspStreamingMaterializationError(
                    "Market and return keys are not aligned"
                )
            permno, day = market_key
            if current is None:
                current = permno
            elif permno != current:
                if permno < current:
                    raise CrspStreamingMaterializationError(
                        "Market/return stream is not sorted by PERMNO"
                    )
                yield current, market_rows, return_rows
                current = permno
                market_rows = []
                return_rows = []
            if day <= DEVELOPMENT_LAST_SIGNAL:
                market_rows.append(market)
                return_rows.append(returns)
    if market_count != return_count:
        raise CrspStreamingMaterializationError(
            "Market and return streams differ in row count"
        )
    if current is not None:
        yield current, market_rows, return_rows


def _context(
    inputs: FeatureStoreInputs,
) -> tuple[tuple[dt.date, ...], list[dict[str, Any]]]:
    calendar = pq.read_table(
        _path(inputs, CALENDAR_TABLE), columns=["calendar_date"]
    )
    sessions = tuple(
        day
        for day in calendar["calendar_date"].to_pylist()
        if day <= MAXIMUM_CONTEXT_DATE
    )
    if not sessions or sessions != tuple(sorted(set(sessions))):
        raise CrspStreamingMaterializationError(
            "Trading calendar is not unique and sorted"
        )
    factors = pq.read_table(
        _path(inputs, FACTOR_TABLE),
        columns=list(FACTOR_COLUMNS),
        filters=[
            ("observation_date", "<=", MAXIMUM_CONTEXT_DATE),
            ("public_availability_date", "<=", MAXIMUM_CONTEXT_DATE),
        ],
    ).to_pylist()
    return sessions, factors


def iter_unfinalized_crsp_rows(
    inputs: FeatureStoreInputs,
    *,
    start_after_key: tuple[int, dt.date] | None = None,
    progress: Callable[[Mapping[str, int]], None] | None = None,
) -> Iterator[CrspMaterializedRow]:
    """Scan aligned CRSP payloads once and emit every development signal row."""

    if start_after_key is not None:
        resume_permno, resume_signal = start_after_key
        if (
            not isinstance(resume_permno, int)
            or isinstance(resume_permno, bool)
            or not isinstance(resume_signal, dt.date)
            or isinstance(resume_signal, dt.datetime)
        ):
            raise CrspStreamingMaterializationError(
                "CRSP resume key is invalid"
            )
    spine = _sorted_spine(inputs)
    signal_groups = _table_groups(spine, "permno")
    share_groups = _parquet_groups(
        _path(inputs, SHARES_TABLE),
        columns=list(SHARES_COLUMNS),
        key="permno",
    )
    sessions, factors = _context(inputs)
    try:
        signal_permno, signals = next(signal_groups)
    except StopIteration as exc:
        raise CrspStreamingMaterializationError("Signal spine has no groups") from exc
    try:
        share_permno, shares = next(share_groups)
    except StopIteration:
        share_permno, shares = None, []
    emitted = securities = 0
    resume_found = start_after_key is None
    for permno, market, returns in _aligned_market_return_groups(
        _path(inputs, MARKET_TABLE),
        _path(inputs, RETURN_TABLE),
    ):
        while signal_permno < permno:
            raise CrspStreamingMaterializationError(
                f"Signal PERMNO lacks CRSP history: {signal_permno}"
            )
        if signal_permno != permno:
            continue
        materialization_signals = signals
        if start_after_key is not None and permno <= resume_permno:
            if permno < resume_permno:
                emitted += len(signals)
                materialization_signals = []
            else:
                if not any(
                    row["signal_date"] == resume_signal for row in signals
                ):
                    raise CrspStreamingMaterializationError(
                        "CRSP resume key is absent from the signal spine"
                    )
                resume_found = True
                emitted += sum(
                    row["signal_date"] <= resume_signal for row in signals
                )
                materialization_signals = [
                    row
                    for row in signals
                    if row["signal_date"] > resume_signal
                ]
        while share_permno is not None and share_permno < permno:
            try:
                share_permno, shares = next(share_groups)
            except StopIteration:
                share_permno, shares = None, []
                break
        security_shares = shares if share_permno == permno else []
        materialized = (
            []
            if not materialization_signals
            else compute_unfinalized_crsp_security_rows(
                signal_rows=materialization_signals,
                sessions=sessions,
                market_rows=market,
                return_rows=returns,
                factor_rows=factors,
                shares_rows=security_shares,
            )
        )
        yield from materialized
        emitted += len(materialized)
        securities += 1
        if progress is not None and securities % 100 == 0:
            progress(
                {
                    "securities": securities,
                    "rows": emitted,
                    "last_permno": permno,
                }
            )
        try:
            signal_permno, signals = next(signal_groups)
        except StopIteration:
            signal_permno = None
            break
    if signal_permno is not None:
        raise CrspStreamingMaterializationError(
            f"Unmaterialized signal PERMNO remains: {signal_permno}"
        )
    if not resume_found:
        raise CrspStreamingMaterializationError(
            "CRSP resume key is beyond the signal spine"
        )
    if emitted != spine.num_rows:
        raise CrspStreamingMaterializationError(
            f"CRSP emitted {emitted} rows; expected {spine.num_rows}"
        )
