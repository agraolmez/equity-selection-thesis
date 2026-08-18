"""Authenticated, point-in-time fundamentals adapter for Stage 6.

The adapter reads only the verified signal spine, auxiliary linkage, and
Compustat observation tables.  It emits common ``FeatureMaterializationRow``
batches and never writes a feature store, label, model, or portfolio artifact.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.features.feature_store_contract import DEFAULT_POLICY
from portfolio_bot.features.feature_store_inputs import FeatureStoreInputs
from portfolio_bot.features.feature_store_materialization import (
    FeatureMaterializationRow,
    FeatureStoreBuildIdentity,
    authenticate_feature_store_inputs,
    validate_precomputed_feature_store_build_identity,
)
from portfolio_bot.features.fundamental_features import (
    ANNUAL_FIELDS,
    QUARTERLY_BALANCE,
    QUARTERLY_DISCRETE,
    QUARTERLY_YTD,
    compute_fundamental_features,
    select_fundamental_bundle,
)


SPINE_TABLE = "crsp_signal_date_universe.parquet"
LINKAGE_TABLE = "auxiliary_signal_linkage.parquet"
QUARTERLY_TABLE = "compustat_quarterly_observation.parquet"
ANNUAL_TABLE = "compustat_annual_observation.parquet"

FIRST_SIGNAL_DATE = dt.date(2000, 1, 31)
LAST_SIGNAL_DATE = dt.date(2014, 10, 31)
MAXIMUM_CONTEXT_DATE = dt.date(2014, 12, 1)
EXPECTED_SPINE_ROWS = 1_263_048
EXPECTED_SIGNAL_DATES = 178

MATERIALIZATION_SPINE_COLUMNS = (
    "signal_date",
    "permno",
    "permco",
    "long_eligible",
    "short_eligible",
    "primary_exchange",
    "share_type",
)

_SPINE_COLUMNS = (
    "signal_date",
    "permno",
    "permco",
    "observation_date",
    "public_availability_date",
    "market_cap_usd",
    "long_eligible",
    "short_eligible",
    "primary_exchange",
    "share_type",
)
_LINKAGE_COLUMNS = (
    "signal_date",
    "permno",
    "ccm_match_status",
    "ccm_gvkey",
    "universe_row_preserved",
)
_QUARTERLY_COLUMNS = (
    "source_record_id",
    "gvkey",
    "permno",
    "fyearq",
    "fqtr",
    "fyr",
    "datadate",
    "observation_date",
    "public_availability_date",
    "stale_after_date",
    "standard_format_eligible",
    "currency_code",
    "fic",
    "loc",
    *QUARTERLY_DISCRETE.values(),
    *QUARTERLY_YTD.values(),
    *QUARTERLY_BALANCE.values(),
)
_ANNUAL_COLUMNS = (
    "source_record_id",
    "gvkey",
    "permno",
    "fyear",
    "datadate",
    "observation_date",
    "public_availability_date",
    "stale_after_date",
    "standard_format_eligible",
    "currency_code",
    "fic",
    "loc",
    *(column for aliases in ANNUAL_FIELDS.values() for column in aliases),
)


class FundamentalMaterializationError(RuntimeError):
    """Raised when verified fundamentals cannot be adapted without ambiguity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FundamentalMaterializationError(message)


def _date(value: Any, label: str) -> dt.date:
    _require(
        isinstance(value, dt.date) and not isinstance(value, dt.datetime),
        f"{label} must be a date",
    )
    return value


def _int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return int(value)


def _required_columns(path: Path, columns: Sequence[str], table: str) -> None:
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(columns) - schema_names
    _require(not missing, f"{table} is missing columns: {sorted(missing)}")


def _create_views(
    connection: duckdb.DuckDBPyConnection,
    inputs: FeatureStoreInputs,
) -> None:
    for view, table_name in (
        ("spine", SPINE_TABLE),
        ("linkage", LINKAGE_TABLE),
        ("quarterly", QUARTERLY_TABLE),
        ("annual", ANNUAL_TABLE),
    ):
        connection.read_parquet(str(inputs.path(table_name))).create_view(view)


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    value = connection.execute(query).fetchone()
    _require(value is not None and len(value) == 1, "DuckDB scalar gate failed")
    return int(value[0])


def _validate_source_gates(
    connection: duckdb.DuckDBPyConnection,
    inputs: FeatureStoreInputs,
) -> None:
    _require(
        inputs.tables[SPINE_TABLE].rows
        == inputs.tables[LINKAGE_TABLE].rows
        == EXPECTED_SPINE_ROWS,
        "Spine/linkage receipts do not equal the frozen 1,263,048-row scope",
    )
    spine_scope = connection.execute(
        """
        SELECT
            count(*) AS rows,
            count(DISTINCT signal_date) AS signal_dates,
            min(signal_date) AS first_signal_date,
            max(signal_date) AS last_signal_date
        FROM spine
        """
    ).fetchone()
    _require(
        spine_scope
        == (
            EXPECTED_SPINE_ROWS,
            EXPECTED_SIGNAL_DATES,
            FIRST_SIGNAL_DATE,
            LAST_SIGNAL_DATE,
        ),
        "Signal spine differs from the frozen row/date scope",
    )
    bad_spine = _scalar(
        connection,
        f"""
        SELECT count(*)
        FROM spine
        WHERE signal_date IS NULL
           OR permno IS NULL
           OR permco IS NULL
           OR long_eligible IS NULL
           OR short_eligible IS NULL
           OR primary_exchange IS NULL
           OR share_type IS NULL
           OR signal_date < DATE '{FIRST_SIGNAL_DATE.isoformat()}'
           OR signal_date > DATE '{LAST_SIGNAL_DATE.isoformat()}'
           OR observation_date <> signal_date
           OR public_availability_date <> signal_date
           OR (short_eligible IS TRUE AND long_eligible IS NOT TRUE)
        """,
    )
    _require(bad_spine == 0, "Signal spine escaped development/context cutoffs")
    bad_linkage = _scalar(
        connection,
        f"""
        SELECT count(*)
        FROM linkage
        WHERE signal_date IS NULL
           OR permno IS NULL
           OR signal_date < DATE '{FIRST_SIGNAL_DATE.isoformat()}'
           OR signal_date > DATE '{LAST_SIGNAL_DATE.isoformat()}'
           OR universe_row_preserved IS DISTINCT FROM TRUE
           OR ccm_match_status NOT IN ('MATCHED', 'UNMATCHED', 'AMBIGUOUS')
           OR (
                ccm_match_status = 'MATCHED'
                AND (ccm_gvkey IS NULL OR trim(ccm_gvkey) = '')
              )
           OR (ccm_match_status <> 'MATCHED' AND ccm_gvkey IS NOT NULL)
        """,
    )
    _require(bad_linkage == 0, "Auxiliary linkage violates the frozen signal contract")
    duplicate_keys = _scalar(
        connection,
        """
        SELECT count(*)
        FROM (
            SELECT 'spine' AS source, signal_date, permno
            FROM spine
            GROUP BY signal_date, permno
            HAVING count(*) > 1
            UNION ALL
            SELECT 'linkage' AS source, signal_date, permno
            FROM linkage
            GROUP BY signal_date, permno
            HAVING count(*) > 1
        )
        """,
    )
    _require(duplicate_keys == 0, "Spine or linkage has duplicate signal keys")
    unmatched_keys = _scalar(
        connection,
        """
        SELECT count(*)
        FROM spine
        FULL OUTER JOIN linkage
          USING (signal_date, permno)
        WHERE spine.permno IS NULL OR linkage.permno IS NULL
        """,
    )
    _require(
        unmatched_keys == 0,
        "Auxiliary linkage does not preserve the exact signal-spine key set",
    )

    for view in ("quarterly", "annual"):
        bad_dates = _scalar(
            connection,
            f"""
            SELECT count(*)
            FROM {view}
            WHERE datadate IS NULL
               OR observation_date IS NULL
               OR public_availability_date IS NULL
               OR stale_after_date IS NULL
               OR datadate > DATE '{MAXIMUM_CONTEXT_DATE.isoformat()}'
               OR observation_date > DATE '{MAXIMUM_CONTEXT_DATE.isoformat()}'
               OR public_availability_date > DATE '{MAXIMUM_CONTEXT_DATE.isoformat()}'
               OR stale_after_date < public_availability_date
            """,
        )
        _require(bad_dates == 0, f"{view} rows escaped development/context cutoffs")
        duplicate_sources = _scalar(
            connection,
            f"""
            SELECT count(*)
            FROM (
                SELECT source_record_id
                FROM {view}
                GROUP BY source_record_id
                HAVING count(*) > 1
            )
            """,
        )
        _require(duplicate_sources == 0, f"{view} has duplicate source_record_id")

    missing_quarter_key = _scalar(
        connection,
        """
        SELECT count(*)
        FROM quarterly
        WHERE standard_format_eligible IS TRUE
          AND (gvkey IS NULL OR permno IS NULL OR fyearq IS NULL OR fqtr IS NULL)
        """,
    )
    _require(missing_quarter_key == 0, "Eligible quarterly fiscal key is incomplete")
    duplicate_quarters = _scalar(
        connection,
        """
        SELECT count(*)
        FROM (
            SELECT gvkey, permno, fyearq, fqtr
            FROM quarterly
            WHERE standard_format_eligible IS TRUE
            GROUP BY gvkey, permno, fyearq, fqtr
            HAVING count(*) > 1
        )
        """,
    )
    _require(duplicate_quarters == 0, "Duplicate quarterly gvkey+permno fiscal key")

    missing_annual_key = _scalar(
        connection,
        """
        SELECT count(*)
        FROM annual
        WHERE standard_format_eligible IS TRUE
          AND (gvkey IS NULL OR permno IS NULL OR fyear IS NULL)
        """,
    )
    _require(missing_annual_key == 0, "Eligible annual fiscal key is incomplete")
    duplicate_annual = _scalar(
        connection,
        """
        SELECT count(*)
        FROM (
            SELECT gvkey, permno, fyear
            FROM annual
            WHERE standard_format_eligible IS TRUE
            GROUP BY gvkey, permno, fyear
            HAVING count(*) > 1
        )
        """,
    )
    _require(duplicate_annual == 0, "Duplicate annual gvkey+permno fiscal key")


def _row_iterator(
    path: Path,
    columns: Sequence[str],
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    source = pq.ParquetFile(path)
    for batch in source.iter_batches(batch_size=batch_size, columns=list(columns)):
        yield from batch.to_pylist()


def _next_spine_chunk(
    spine: Iterator[dict[str, Any]],
    linkage: Iterator[dict[str, Any]],
    batch_size: int,
    previous: tuple[dt.date, int] | None,
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    tuple[dt.date, int] | None,
    bool,
]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    exhausted = False
    for _ in range(batch_size):
        try:
            spine_row = next(spine)
        except StopIteration:
            exhausted = True
            try:
                next(linkage)
            except StopIteration:
                pass
            else:
                raise FundamentalMaterializationError(
                    "Linkage contains rows after the signal spine"
                )
            break
        try:
            linkage_row = next(linkage)
        except StopIteration as exc:
            raise FundamentalMaterializationError(
                "Linkage ended before the signal spine"
            ) from exc
        signal = _date(spine_row["signal_date"], "spine signal_date")
        permno = _int(spine_row["permno"], "spine permno")
        key = (signal, permno)
        _require(previous is None or key > previous, "Signal spine is not sorted and unique")
        linkage_key = (
            _date(linkage_row["signal_date"], "linkage signal_date"),
            _int(linkage_row["permno"], "linkage permno"),
        )
        _require(linkage_key == key, "Linkage is not row-for-row aligned to the spine")
        previous = key
        rows.append((spine_row, linkage_row))
    return rows, previous, exhausted


def _quoted(columns: Sequence[str], prefix: str = "") -> str:
    return ", ".join(
        f'{prefix}"{column}"' for column in dict.fromkeys(columns)
    )


@dataclass
class _FundamentalHistoryCache:
    """Load each matched Compustat history at most once per materialization."""

    connection: duckdb.DuckDBPyConnection
    quarterly: dict[tuple[str, int], list[dict[str, Any]]] = field(
        default_factory=dict
    )
    annual: dict[tuple[str, int], list[dict[str, Any]]] = field(
        default_factory=dict
    )
    source_queries: int = 0

    def _load(
        self,
        matched: Sequence[tuple[str, int]],
        *,
        view: str,
        columns: Sequence[str],
        fiscal_order: str,
        target: dict[tuple[str, int], list[dict[str, Any]]],
    ) -> None:
        missing = [key for key in matched if key not in target]
        if not missing:
            return
        for key in missing:
            target[key] = []
        keys = pa.table(
            {
                "gvkey": pa.array(
                    [key[0] for key in missing], type=pa.string()
                ),
                "permno": pa.array(
                    [key[1] for key in missing], type=pa.int64()
                ),
            }
        )
        self.connection.register("_fundamental_batch_keys", keys)
        try:
            reader = self.connection.execute(
                f"""
                SELECT {_quoted(columns, "source.")}
                FROM {view} AS source
                INNER JOIN _fundamental_batch_keys AS keys
                  ON source.gvkey = keys.gvkey
                 AND source.permno = keys.permno
                WHERE source.standard_format_eligible IS TRUE
                ORDER BY source.gvkey, source.permno, {fiscal_order}
                """
            ).fetch_record_batch(rows_per_batch=65_536)
            self.source_queries += 1
            for batch in reader:
                for item in batch.to_pylist():
                    key = (str(item["gvkey"]), int(item["permno"]))
                    _require(
                        key in target,
                        f"{view} query escaped its exact gvkey+permno key set",
                    )
                    target[key].append(item)
        finally:
            self.connection.unregister("_fundamental_batch_keys")

    def groups(
        self,
        rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    ) -> tuple[
        Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
        Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    ]:
        matched = sorted(
            {
                (str(link["ccm_gvkey"]), int(spine["permno"]))
                for spine, link in rows
                if link["ccm_match_status"] == "MATCHED"
            }
        )
        if matched:
            self._load(
                matched,
                view="quarterly",
                columns=_QUARTERLY_COLUMNS,
                fiscal_order='"fyearq", "fqtr", "datadate", "source_record_id"',
                target=self.quarterly,
            )
            self._load(
                matched,
                view="annual",
                columns=_ANNUAL_COLUMNS,
                fiscal_order='"fyear", "datadate", "source_record_id"',
                target=self.annual,
            )
        return self.quarterly, self.annual


def _missing_status(
    status: str,
    signal_date: dt.date,
    quarterly: Sequence[Mapping[str, Any]],
    annual: Sequence[Mapping[str, Any]],
) -> str:
    if status == "AMBIGUOUS":
        return "MISSING_AMBIGUOUS_LINK"
    if status != "MATCHED":
        return "MISSING_NOT_COVERED"
    available = [
        row
        for row in (*quarterly, *annual)
        if row.get("standard_format_eligible") is True
        and isinstance(row.get("observation_date"), dt.date)
        and isinstance(row.get("public_availability_date"), dt.date)
        and row["observation_date"] <= signal_date
        and row["public_availability_date"] <= signal_date
    ]
    if not available:
        return "MISSING_NOT_COVERED" if not quarterly and not annual else "MISSING_INSUFFICIENT_HISTORY"
    if all(
        not isinstance(row.get("stale_after_date"), dt.date)
        or signal_date > row["stale_after_date"]
        for row in available
    ):
        return "MISSING_STALE"
    return "MISSING_INSUFFICIENT_HISTORY"


def _feature_row(
    spine: Mapping[str, Any],
    linkage: Mapping[str, Any],
    quarterly: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    annual: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
) -> FeatureMaterializationRow:
    signal = _date(spine["signal_date"], "signal_date")
    permno = _int(spine["permno"], "permno")
    status = str(linkage["ccm_match_status"])
    key = (
        str(linkage["ccm_gvkey"]),
        permno,
    )
    quarter_rows = tuple(quarterly.get(key, ())) if status == "MATCHED" else ()
    annual_rows = tuple(annual.get(key, ())) if status == "MATCHED" else ()
    bundle = (
        select_fundamental_bundle(quarter_rows, annual_rows, signal)
        if status == "MATCHED"
        else None
    )
    market_cap = spine.get("market_cap_usd")
    if market_cap is not None:
        try:
            market_cap = float(market_cap)
        except (TypeError, ValueError, OverflowError):
            market_cap = None
        if market_cap is not None and not math.isfinite(market_cap):
            market_cap = None
    result = compute_fundamental_features(
        bundle,
        market_cap,
        market_observation_date=_date(
            spine["observation_date"], "market observation_date"
        ),
        market_public_availability_date=_date(
            spine["public_availability_date"], "market public_availability_date"
        ),
        bundle_missing_status=_missing_status(
            status, signal, quarter_rows, annual_rows
        ),
    )
    _require(
        isinstance(spine["long_eligible"], bool)
        and isinstance(spine["short_eligible"], bool),
        "Spine eligibility flags must be Boolean",
    )
    return FeatureMaterializationRow(
        signal_date=signal,
        permno=permno,
        permco=_int(spine["permco"], "permco"),
        long_eligible=spine["long_eligible"],
        short_eligible=spine["short_eligible"],
        primary_exchange=(
            None
            if spine.get("primary_exchange") is None
            else str(spine["primary_exchange"])
        ),
        share_type=None if spine.get("share_type") is None else str(spine["share_type"]),
        result=result,
    )


def iter_fundamental_feature_batches(
    inputs: FeatureStoreInputs,
    *,
    batch_size: int = 20_000,
    policy_path: Path = DEFAULT_POLICY,
    build_identity: FeatureStoreBuildIdentity | None = None,
) -> Iterator[tuple[FeatureMaterializationRow, ...]]:
    """Return a repeatable, spine-ordered iterator over all fundamental rows."""

    _require(
        isinstance(batch_size, int)
        and not isinstance(batch_size, bool)
        and batch_size > 0,
        "batch_size must be a positive integer",
    )
    if build_identity is None:
        authenticate_feature_store_inputs(inputs, policy_path)
    else:
        validate_precomputed_feature_store_build_identity(
            inputs,
            build_identity,
            policy_path,
        )
    for table_name, columns in (
        (SPINE_TABLE, _SPINE_COLUMNS),
        (LINKAGE_TABLE, _LINKAGE_COLUMNS),
        (QUARTERLY_TABLE, _QUARTERLY_COLUMNS),
        (ANNUAL_TABLE, _ANNUAL_COLUMNS),
    ):
        _required_columns(inputs.path(table_name), columns, table_name)

    def generate() -> Iterator[tuple[FeatureMaterializationRow, ...]]:
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute("PRAGMA threads=1")
            _create_views(connection, inputs)
            _validate_source_gates(connection, inputs)
            spine = _row_iterator(inputs.path(SPINE_TABLE), _SPINE_COLUMNS, batch_size)
            linkage = _row_iterator(
                inputs.path(LINKAGE_TABLE), _LINKAGE_COLUMNS, batch_size
            )
            history = _FundamentalHistoryCache(connection)
            previous: tuple[dt.date, int] | None = None
            produced = 0
            while True:
                source_rows, previous, exhausted = _next_spine_chunk(
                    spine, linkage, batch_size, previous
                )
                if source_rows:
                    quarterly, annual = history.groups(source_rows)
                    batch = tuple(
                        _feature_row(
                            spine_row,
                            linkage_row,
                            quarterly,
                            annual,
                        )
                        for spine_row, linkage_row in source_rows
                    )
                    produced += len(batch)
                    yield batch
                if exhausted:
                    break
            _require(
                produced == inputs.tables[SPINE_TABLE].rows,
                "Fundamental adapter did not preserve every signal-spine row",
            )
        finally:
            connection.close()

    return generate()


# Concise alias for orchestration code.
materialize_fundamentals = iter_fundamental_feature_batches
