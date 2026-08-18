"""Bounded, resumable signal routing for materialized CRSP feature rows.

The upstream CRSP scanner is naturally ordered by ``(permno, signal_date)``,
while beta clipping and spread imputation require complete signal-date cross
sections.  This module checkpoints the unfinalized stream into a caller-owned
DuckDB spool and later loads exactly one signal cross section at a time.  It
does not publish cache artifacts or write outside the supplied spool path.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import duckdb
import pyarrow as pa

from portfolio_bot.features.crsp_features import CRSP_FEATURE_NAMES
from portfolio_bot.features.feature_lineage import (
    FeatureFamilyResult,
    FeatureLineage,
    VALUE_STATUS_SET,
)
from portfolio_bot.features.feature_store_materialization import (
    FeatureMaterializationRow,
)
from portfolio_bot.features.materialize_crsp import (
    CRSP_FAMILIES,
    CrspMaterializedRow,
    finalize_crsp_materialized_signal,
)


_DATABASE_NAME = "crsp_signal_router.duckdb"
_SCHEMA_VERSION = 1
_STATUS_SPOOLING = "SPOOLING"
_STATUS_COMPLETE = "COMPLETE"
_AUDIT_FIELDS = (
    pa.field("signal_date", pa.date32(), nullable=False),
    pa.field("permno", pa.int64(), nullable=False),
    pa.field("permco", pa.int64(), nullable=False),
    pa.field("long_eligible", pa.bool_(), nullable=False),
    pa.field("short_eligible", pa.bool_(), nullable=False),
    pa.field("primary_exchange", pa.string(), nullable=False),
    pa.field("share_type", pa.string(), nullable=False),
)


class CrspSignalRoutingError(RuntimeError):
    """Raised when a CRSP signal spool cannot be resumed or finalized."""


@dataclass(frozen=True)
class CrspFamilyRowBatch:
    """One bounded common-writer batch from a finalized signal cross section."""

    signal_date: dt.date
    family: str
    batch_ordinal: int
    rows: tuple[FeatureMaterializationRow, ...]


@dataclass(frozen=True)
class CrspSignalRouter:
    """Authenticated handle to a complete caller-owned CRSP signal spool."""

    spool_path: Path
    source_id: str
    row_count: int
    signal_dates: tuple[dt.date, ...]
    reused_existing: bool

    @property
    def database_path(self) -> Path:
        return self.spool_path / _DATABASE_NAME

    def iter_family_row_batches(
        self,
        *,
        batch_size: int = 2_048,
    ) -> Iterator[CrspFamilyRowBatch]:
        """Finalize one signal at a time and emit all three CRSP families."""

        _positive_integer(batch_size, "batch_size")
        connection = _open_complete_router(self)
        try:
            for signal_date in self.signal_dates:
                table = connection.execute(
                    """
                    SELECT *
                    FROM unfinalized_crsp
                    WHERE signal_date = ?
                    ORDER BY permno
                    """,
                    [signal_date],
                ).fetch_arrow_table()
                raw_rows = _spool_table_to_rows(table)
                finalized = finalize_crsp_materialized_signal(raw_rows)
                if not finalized:
                    raise CrspSignalRoutingError(
                        f"Signal cross section is empty: {signal_date}"
                    )
                for family, names in CRSP_FAMILIES.items():
                    common = tuple(
                        _family_row(row, names) for row in finalized
                    )
                    for ordinal, start in enumerate(
                        range(0, len(common), batch_size)
                    ):
                        yield CrspFamilyRowBatch(
                            signal_date=signal_date,
                            family=family,
                            batch_ordinal=ordinal,
                            rows=common[start : start + batch_size],
                        )
        finally:
            connection.close()


def _positive_integer(value: int, label: str) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise CrspSignalRoutingError(f"{label} must be a positive integer")


def _spool_schema() -> pa.Schema:
    fields = list(_AUDIT_FIELDS)
    for name in CRSP_FEATURE_NAMES:
        fields.extend(
            (
                pa.field(name, pa.float64(), nullable=True),
                pa.field(f"{name}__observation", pa.date32(), nullable=True),
                pa.field(f"{name}__availability", pa.date32(), nullable=True),
                pa.field(f"{name}__stale_after", pa.date32(), nullable=True),
                pa.field(f"{name}__value_status", pa.string(), nullable=False),
            )
        )
    return pa.schema(fields)


def _safe_spool_path(value: Path) -> Path:
    path = Path(os.path.abspath(Path(value).expanduser()))
    if path.exists():
        observed = os.lstat(path)
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(
            observed.st_mode
        ):
            raise CrspSignalRoutingError(
                "CRSP spool path must be a real directory"
            )
    else:
        path.mkdir(parents=True)
    database = path / _DATABASE_NAME
    if database.is_symlink():
        raise CrspSignalRoutingError("CRSP spool database cannot be a symlink")
    return path


def _validate_source_id(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise CrspSignalRoutingError(
            "source_id must be a nonempty string of at most 256 characters"
        )
    return value


def _create_router_database(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_id: str,
    expected_rows: int,
) -> None:
    empty = pa.Table.from_batches([], schema=_spool_schema())
    connection.register("_empty_crsp_spool", empty)
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            """
            CREATE TABLE unfinalized_crsp AS
            SELECT * FROM _empty_crsp_spool
            """
        )
        connection.execute(
            """
            CREATE TABLE routing_metadata (
                singleton BOOLEAN PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                source_id VARCHAR NOT NULL,
                expected_rows BIGINT NOT NULL,
                row_count BIGINT NOT NULL,
                last_permno BIGINT,
                last_signal_date DATE,
                status VARCHAR NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO routing_metadata
            VALUES (TRUE, ?, ?, ?, 0, NULL, NULL, ?)
            """,
            [
                _SCHEMA_VERSION,
                source_id,
                expected_rows,
                _STATUS_SPOOLING,
            ],
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.unregister("_empty_crsp_spool")


def _metadata(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[int, str, int, int, int | None, dt.date | None, str]:
    try:
        rows = connection.execute(
            """
            SELECT
                schema_version,
                source_id,
                expected_rows,
                row_count,
                last_permno,
                last_signal_date,
                status
            FROM routing_metadata
            WHERE singleton = TRUE
            """
        ).fetchall()
    except duckdb.Error as exc:
        raise CrspSignalRoutingError(
            "CRSP spool metadata is unreadable"
        ) from exc
    if len(rows) != 1:
        raise CrspSignalRoutingError(
            "CRSP spool metadata cardinality differs"
        )
    return rows[0]


def _validate_spool_schema(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    try:
        observed = connection.execute(
            "SELECT * FROM unfinalized_crsp LIMIT 0"
        ).fetch_arrow_table().schema
    except duckdb.Error as exc:
        raise CrspSignalRoutingError(
            "CRSP spool table is unreadable"
        ) from exc
    expected = _spool_schema()
    if observed.names != expected.names or tuple(
        field.type for field in observed
    ) != tuple(field.type for field in expected):
        raise CrspSignalRoutingError("CRSP spool schema differs")


def _validate_checkpoint(
    connection: duckdb.DuckDBPyConnection,
    *,
    expected_row_count: int,
    expected_last_key: tuple[int, dt.date] | None,
) -> None:
    observed_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM unfinalized_crsp"
        ).fetchone()[0]
    )
    if observed_count != expected_row_count:
        raise CrspSignalRoutingError("CRSP spool row checkpoint differs")
    observed_last = connection.execute(
        """
        SELECT permno, signal_date
        FROM unfinalized_crsp
        ORDER BY permno DESC, signal_date DESC
        LIMIT 1
        """
    ).fetchone()
    if observed_last is not None:
        observed_last = (int(observed_last[0]), observed_last[1])
    if observed_last != expected_last_key:
        raise CrspSignalRoutingError("CRSP spool key checkpoint differs")


def _validate_row(row: CrspMaterializedRow) -> tuple[int, dt.date]:
    if not isinstance(row, CrspMaterializedRow):
        raise CrspSignalRoutingError(
            "CRSP stream contains an invalid row type"
        )
    if (
        not isinstance(row.signal_date, dt.date)
        or isinstance(row.signal_date, dt.datetime)
        or not isinstance(row.permno, int)
        or isinstance(row.permno, bool)
        or not isinstance(row.permco, int)
        or isinstance(row.permco, bool)
        or not isinstance(row.long_eligible, bool)
        or not isinstance(row.short_eligible, bool)
        or not isinstance(row.primary_exchange, str)
        or not isinstance(row.share_type, str)
    ):
        raise CrspSignalRoutingError("CRSP stream audit fields are invalid")
    if not isinstance(row.result, FeatureFamilyResult):
        raise CrspSignalRoutingError("CRSP stream result type is invalid")
    if (
        tuple(row.result.values) != CRSP_FEATURE_NAMES
        or tuple(row.result.lineage) != CRSP_FEATURE_NAMES
    ):
        raise CrspSignalRoutingError(
            "CRSP stream feature order differs"
        )
    for name in CRSP_FEATURE_NAMES:
        value = row.result.values[name]
        lineage = row.result.lineage[name]
        if (
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            )
        ):
            raise CrspSignalRoutingError(
                f"CRSP stream feature is nonfinite: {name}"
            )
        if (
            not isinstance(lineage, FeatureLineage)
            or lineage.value_status not in VALUE_STATUS_SET
            or (value is None) == (lineage.value_status == "PRESENT")
        ):
            raise CrspSignalRoutingError(
                f"CRSP stream lineage differs: {name}"
            )
        if lineage.value_status == "PRESENT" and (
            lineage.max_observation_date > row.signal_date
            or lineage.max_public_availability_date > row.signal_date
            or (
                lineage.stale_after_date is not None
                and lineage.stale_after_date < row.signal_date
            )
        ):
            raise CrspSignalRoutingError(
                f"CRSP stream lineage timing differs: {name}"
            )
    return row.permno, row.signal_date


def _rows_to_spool_table(
    rows: Sequence[CrspMaterializedRow],
) -> pa.Table:
    columns: dict[str, pa.Array] = {
        "signal_date": pa.array(
            [row.signal_date for row in rows], type=pa.date32()
        ),
        "permno": pa.array(
            [row.permno for row in rows], type=pa.int64()
        ),
        "permco": pa.array(
            [row.permco for row in rows], type=pa.int64()
        ),
        "long_eligible": pa.array(
            [row.long_eligible for row in rows], type=pa.bool_()
        ),
        "short_eligible": pa.array(
            [row.short_eligible for row in rows], type=pa.bool_()
        ),
        "primary_exchange": pa.array(
            [row.primary_exchange for row in rows], type=pa.string()
        ),
        "share_type": pa.array(
            [row.share_type for row in rows], type=pa.string()
        ),
    }
    for name in CRSP_FEATURE_NAMES:
        lineages = [row.result.lineage[name] for row in rows]
        columns[name] = pa.array(
            [row.result.values[name] for row in rows],
            type=pa.float64(),
            from_pandas=True,
        )
        columns[f"{name}__observation"] = pa.array(
            [item.max_observation_date for item in lineages],
            type=pa.date32(),
        )
        columns[f"{name}__availability"] = pa.array(
            [item.max_public_availability_date for item in lineages],
            type=pa.date32(),
        )
        columns[f"{name}__stale_after"] = pa.array(
            [item.stale_after_date for item in lineages],
            type=pa.date32(),
        )
        columns[f"{name}__value_status"] = pa.array(
            [item.value_status for item in lineages],
            type=pa.string(),
        )
    return pa.table(columns, schema=_spool_schema())


def _append_checkpoint(
    connection: duckdb.DuckDBPyConnection,
    rows: Sequence[CrspMaterializedRow],
    *,
    prior_count: int,
) -> int:
    table = _rows_to_spool_table(rows)
    connection.register("_crsp_spool_batch", table)
    last = rows[-1]
    new_count = prior_count + len(rows)
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            """
            INSERT INTO unfinalized_crsp
            SELECT * FROM _crsp_spool_batch
            """
        )
        connection.execute(
            """
            UPDATE routing_metadata
            SET
                row_count = ?,
                last_permno = ?,
                last_signal_date = ?
            WHERE singleton = TRUE
            """,
            [new_count, last.permno, last.signal_date],
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.unregister("_crsp_spool_batch")
    return new_count


def _signal_dates(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[dt.date, ...]:
    return tuple(
        row[0]
        for row in connection.execute(
            """
            SELECT DISTINCT signal_date
            FROM unfinalized_crsp
            ORDER BY signal_date
            """
        ).fetchall()
    )


def spool_crsp_signal_router(
    rows: (
        Iterable[CrspMaterializedRow]
        | Callable[
            [tuple[int, dt.date] | None],
            Iterable[CrspMaterializedRow],
        ]
    ),
    *,
    spool_path: Path,
    source_id: str,
    expected_rows: int,
    checkpoint_rows: int = 4_096,
) -> CrspSignalRouter:
    """Checkpoint a canonical PERMNO-sorted stream and return a complete router.

    A partial spool remains in ``SPOOLING`` state if iteration raises or ends
    before ``expected_rows``.  Calling this function again with the same
    cryptographic ``source_id`` safely skips the committed prefix and resumes.
    """

    source_id = _validate_source_id(source_id)
    _positive_integer(expected_rows, "expected_rows")
    _positive_integer(checkpoint_rows, "checkpoint_rows")
    root = _safe_spool_path(spool_path)
    database = root / _DATABASE_NAME
    existed = database.exists()
    connection = duckdb.connect(database.as_posix())
    try:
        if not existed:
            _create_router_database(
                connection,
                source_id=source_id,
                expected_rows=expected_rows,
            )
        _validate_spool_schema(connection)
        (
            version,
            stored_source,
            stored_expected,
            row_count,
            last_permno,
            last_signal,
            status,
        ) = _metadata(connection)
        if (
            version != _SCHEMA_VERSION
            or stored_source != source_id
            or stored_expected != expected_rows
            or status not in {_STATUS_SPOOLING, _STATUS_COMPLETE}
        ):
            raise CrspSignalRoutingError(
                "CRSP spool identity or state differs"
            )
        last_key = (
            None
            if last_permno is None or last_signal is None
            else (int(last_permno), last_signal)
        )
        _validate_checkpoint(
            connection,
            expected_row_count=int(row_count),
            expected_last_key=last_key,
        )
        if status == _STATUS_COMPLETE:
            if row_count != expected_rows:
                raise CrspSignalRoutingError(
                    "Complete CRSP spool cardinality differs"
                )
            return CrspSignalRouter(
                spool_path=root,
                source_id=source_id,
                row_count=int(row_count),
                signal_dates=_signal_dates(connection),
                reused_existing=True,
            )

        stream = rows(last_key) if callable(rows) else rows
        buffer: list[CrspMaterializedRow] = []
        input_last_key: tuple[int, dt.date] | None = None
        try:
            for row in stream:
                key = _validate_row(row)
                if input_last_key is not None and key <= input_last_key:
                    raise CrspSignalRoutingError(
                        "CRSP input stream is not strictly PERMNO/signal sorted"
                    )
                input_last_key = key
                if last_key is not None and key <= last_key:
                    continue
                if int(row_count) + len(buffer) >= expected_rows:
                    raise CrspSignalRoutingError(
                        "CRSP input stream exceeds expected_rows"
                    )
                buffer.append(row)
                if len(buffer) >= checkpoint_rows:
                    row_count = _append_checkpoint(
                        connection,
                        buffer,
                        prior_count=int(row_count),
                    )
                    last_key = (
                        buffer[-1].permno,
                        buffer[-1].signal_date,
                    )
                    buffer.clear()
        except Exception:
            if buffer:
                _append_checkpoint(
                    connection,
                    buffer,
                    prior_count=int(row_count),
                )
            raise
        if buffer:
            row_count = _append_checkpoint(
                connection,
                buffer,
                prior_count=int(row_count),
            )
            last_key = (buffer[-1].permno, buffer[-1].signal_date)
        if row_count != expected_rows:
            raise CrspSignalRoutingError(
                f"CRSP stream ended at {row_count} rows; "
                f"expected {expected_rows}"
            )
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                """
                UPDATE routing_metadata
                SET status = ?
                WHERE singleton = TRUE
                """,
                [_STATUS_COMPLETE],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        return CrspSignalRouter(
            spool_path=root,
            source_id=source_id,
            row_count=int(row_count),
            signal_dates=_signal_dates(connection),
            reused_existing=bool(existed),
        )
    except duckdb.Error as exc:
        raise CrspSignalRoutingError(
            "DuckDB could not maintain the CRSP signal spool"
        ) from exc
    finally:
        connection.close()


def _spool_table_to_rows(table: pa.Table) -> list[CrspMaterializedRow]:
    columns = {
        name: table[name].to_pylist() for name in table.schema.names
    }
    rows: list[CrspMaterializedRow] = []
    for index in range(table.num_rows):
        values = {
            name: columns[name][index] for name in CRSP_FEATURE_NAMES
        }
        lineage = {
            name: FeatureLineage(
                columns[f"{name}__observation"][index],
                columns[f"{name}__availability"][index],
                columns[f"{name}__stale_after"][index],
                columns[f"{name}__value_status"][index],
            )
            for name in CRSP_FEATURE_NAMES
        }
        rows.append(
            CrspMaterializedRow(
                signal_date=columns["signal_date"][index],
                permno=int(columns["permno"][index]),
                permco=int(columns["permco"][index]),
                long_eligible=bool(columns["long_eligible"][index]),
                short_eligible=bool(columns["short_eligible"][index]),
                primary_exchange=str(
                    columns["primary_exchange"][index]
                ),
                share_type=str(columns["share_type"][index]),
                result=FeatureFamilyResult(values, lineage),
            )
        )
    return rows


def _family_row(
    row: CrspMaterializedRow,
    names: Sequence[str],
) -> FeatureMaterializationRow:
    result = FeatureFamilyResult(
        values={name: row.result.values[name] for name in names},
        lineage={name: row.result.lineage[name] for name in names},
    )
    return FeatureMaterializationRow(
        signal_date=row.signal_date,
        permno=row.permno,
        permco=row.permco,
        long_eligible=row.long_eligible,
        short_eligible=row.short_eligible,
        primary_exchange=row.primary_exchange,
        share_type=row.share_type,
        result=result,
    )


def _open_complete_router(
    router: CrspSignalRouter,
) -> duckdb.DuckDBPyConnection:
    root = _safe_spool_path(router.spool_path)
    if root != router.spool_path or not router.database_path.is_file():
        raise CrspSignalRoutingError("CRSP router path differs")
    connection = duckdb.connect(
        router.database_path.as_posix(),
        read_only=True,
    )
    try:
        _validate_spool_schema(connection)
        (
            version,
            source_id,
            expected_rows,
            row_count,
            last_permno,
            last_signal,
            status,
        ) = _metadata(connection)
        if (
            version != _SCHEMA_VERSION
            or source_id != router.source_id
            or expected_rows != router.row_count
            or row_count != router.row_count
            or status != _STATUS_COMPLETE
        ):
            raise CrspSignalRoutingError(
                "CRSP router completion binding differs"
            )
        last_key = (
            None
            if last_permno is None or last_signal is None
            else (int(last_permno), last_signal)
        )
        _validate_checkpoint(
            connection,
            expected_row_count=router.row_count,
            expected_last_key=last_key,
        )
        if _signal_dates(connection) != router.signal_dates:
            raise CrspSignalRoutingError("CRSP router signal set differs")
        return connection
    except Exception:
        connection.close()
        raise
