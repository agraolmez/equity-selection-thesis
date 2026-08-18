"""Authenticated point-in-time ownership and event materialization.

The adapter consumes the complete verified signal spine, preserves its exact
order, and yields common :class:`FeatureMaterializationRow` batches.  Large SEC
sources are advanced once in public-availability order.  The 13F holding source
is linked once, set-wise, to report-date CRSP identity intervals instead of
being rescanned for every security.

This module performs no feature-store writes and constructs no labels, models,
portfolios, or performance outcomes.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import tomllib
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.features.feature_lineage import (
    FeatureFamilyResult,
    make_lineage,
)
from portfolio_bot.features.feature_store_contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
)
from portfolio_bot.features.feature_store_inputs import FeatureStoreInputs
from portfolio_bot.features.feature_store_materialization import (
    FeatureMaterializationRow,
    FeatureStoreBuildIdentity,
    authenticate_feature_store_inputs,
    validate_precomputed_feature_store_build_identity,
)
from portfolio_bot.features.fundamental_features import (
    FundamentalBundle,
    select_fundamental_bundle,
)
from portfolio_bot.features.ownership_event_features import (
    ANNUAL_FORMS,
    FEATURE_NAMES,
    INTERIM_FORMS,
    build_ownership_event_feature_result,
    normalize_text,
    _select_13f_snapshots,
)


SPINE_TABLE = "crsp_signal_date_universe.parquet"
LINKAGE_TABLE = "auxiliary_signal_linkage.parquet"
QUARTERLY_TABLE = "compustat_quarterly_observation.parquet"
ANNUAL_TABLE = "compustat_annual_observation.parquet"
SEGMENT_TABLE = "compustat_segment_observation.parquet"
IDENTITY_TABLE = "crsp_identity_interval.parquet"
SHARES_TABLE = "crsp_shares_interval.parquet"
SEC_FILING_TABLE = "sec_filing_observation.parquet"
SEC_ITEM_TABLE = "sec_8k_item_observation.parquet"
FILING_13F_TABLE = "sec_13f_filing_observation.parquet"
HOLDING_13F_TABLE = "sec_13f_holding_observation.parquet"
RESTATEMENT_LINK_TABLE = "restatement_crsp_link.parquet"
RESTATEMENT_EVENT_TABLE = "restatement_event_observation.parquet"

FIRST_SIGNAL_DATE = dt.date(2000, 1, 31)
LAST_SIGNAL_DATE = dt.date(2014, 10, 31)

_SPINE_COLUMNS = (
    "signal_date",
    "permno",
    "permco",
    "observation_date",
    "public_availability_date",
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
    "sec_match_status",
    "sec_cik",
    "universe_row_preserved",
)
_QUARTERLY_METADATA_COLUMNS = (
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
)
_ANNUAL_METADATA_COLUMNS = (
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
)
_IDENTITY_COLUMNS = (
    "permno",
    "permco",
    "start_date",
    "end_date",
    "cusip",
    "us_incorporation_flag",
    "share_type",
    "sic_code",
    "primary_exchange",
)
_SHARES_COLUMNS = (
    "permno",
    "start_date",
    "end_date",
    "shares_outstanding",
)
_SEC_FILING_COLUMNS = (
    "source_record_id",
    "cik",
    "filing_date",
    "form_family",
    "observation_date",
    "public_availability_date",
)
_SEC_ITEM_COLUMNS = (
    "source_record_id",
    "cik",
    "filing_date",
    "form_family",
    "item_number",
    "observation_date",
    "public_availability_date",
)
_FILING_13F_COLUMNS = (
    "filing_id",
    "manager_cik",
    "filing_date",
    "report_date",
    "is_amendment",
    "observation_date",
    "public_availability_date",
    "stale_after_date",
)
_HOLDING_13F_COLUMNS = (
    "filing_id",
    "source_row_ordinal",
    "report_date",
    "normalized_cusip8",
    "reported_shares",
    "put_call",
    "observation_date",
    "public_availability_date",
)
_RESTATEMENT_EVENT_COLUMNS = (
    "restatement_notification_key",
    "observation_date",
    "public_availability_date",
)
_RESTATEMENT_LINK_COLUMNS = (
    "restatement_notification_key",
    "permno",
    "link_status",
)

_MISSING = object()


class OwnershipEventMaterializationError(RuntimeError):
    """Raised when verified ownership/event inputs cannot be joined safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OwnershipEventMaterializationError(message)


def _date(value: Any, label: str) -> dt.date:
    _require(
        isinstance(value, dt.date) and not isinstance(value, dt.datetime),
        f"{label} must be a date",
    )
    return value


def _integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return int(value)


def _required_columns(path: Path, columns: Sequence[str], table: str) -> None:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(columns) - available
    _require(not missing, f"{table} is missing columns: {sorted(missing)}")


def _row_iterator(
    path: Path,
    columns: Sequence[str],
    *,
    batch_size: int = 65_536,
) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=list(columns),
    ):
        yield from batch.to_pylist()


def _quoted(columns: Sequence[str], prefix: str = "") -> str:
    return ", ".join(
        f'{prefix}"{column}"' for column in dict.fromkeys(columns)
    )


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    row = connection.execute(query).fetchone()
    _require(row is not None and len(row) == 1, "DuckDB scalar gate failed")
    return int(row[0])


def _create_views(
    connection: duckdb.DuckDBPyConnection,
    inputs: FeatureStoreInputs,
) -> None:
    names = (
        ("spine", SPINE_TABLE),
        ("linkage", LINKAGE_TABLE),
        ("quarterly", QUARTERLY_TABLE),
        ("annual", ANNUAL_TABLE),
        ("segments", SEGMENT_TABLE),
        ("identity", IDENTITY_TABLE),
        ("shares", SHARES_TABLE),
        ("sec_filings", SEC_FILING_TABLE),
        ("sec_items", SEC_ITEM_TABLE),
        ("filings_13f", FILING_13F_TABLE),
        ("holdings_13f", HOLDING_13F_TABLE),
        ("restatement_links", RESTATEMENT_LINK_TABLE),
        ("restatement_events", RESTATEMENT_EVENT_TABLE),
    )
    for view, table in names:
        connection.read_parquet(str(inputs.path(table))).create_view(view)


def _validate_source_gates(
    connection: duckdb.DuckDBPyConnection,
    inputs: FeatureStoreInputs,
) -> None:
    _require(
        inputs.tables[SPINE_TABLE].rows == inputs.tables[LINKAGE_TABLE].rows,
        "Auxiliary linkage receipt differs from the full signal spine",
    )
    bad_spine = _scalar(
        connection,
        f"""
        SELECT count(*)
        FROM spine
        WHERE signal_date < DATE '{FIRST_SIGNAL_DATE.isoformat()}'
           OR signal_date > DATE '{LAST_SIGNAL_DATE.isoformat()}'
           OR observation_date <> signal_date
           OR public_availability_date <> signal_date
        """,
    )
    _require(bad_spine == 0, "Signal spine escaped the development cutoff")
    bad_linkage = _scalar(
        connection,
        f"""
        SELECT count(*)
        FROM linkage
        WHERE signal_date < DATE '{FIRST_SIGNAL_DATE.isoformat()}'
           OR signal_date > DATE '{LAST_SIGNAL_DATE.isoformat()}'
           OR universe_row_preserved IS DISTINCT FROM TRUE
           OR ccm_match_status NOT IN ('MATCHED', 'UNMATCHED', 'AMBIGUOUS')
           OR sec_match_status NOT IN ('MATCHED', 'UNMATCHED', 'AMBIGUOUS')
           OR (ccm_match_status = 'MATCHED' AND ccm_gvkey IS NULL)
           OR (ccm_match_status <> 'MATCHED' AND ccm_gvkey IS NOT NULL)
           OR (sec_match_status = 'MATCHED' AND sec_cik IS NULL)
           OR (sec_match_status <> 'MATCHED' AND sec_cik IS NOT NULL)
        """,
    )
    _require(bad_linkage == 0, "Auxiliary linkage violates its frozen contract")

    for view in ("quarterly", "annual"):
        bad_dates = _scalar(
            connection,
            f"""
            SELECT count(*)
            FROM {view}
            WHERE observation_date > DATE '{LAST_SIGNAL_DATE.isoformat()}'
               OR public_availability_date > DATE '{LAST_SIGNAL_DATE.isoformat()}'
            """,
        )
        _require(bad_dates == 0, f"{view} escaped the predictor cutoff")
    for view in (
        "sec_filings",
        "sec_items",
        "filings_13f",
        "holdings_13f",
        "restatement_events",
    ):
        bad_dates = _scalar(
            connection,
            f"""
            SELECT count(*)
            FROM {view}
            WHERE observation_date > public_availability_date
               OR observation_date > DATE '{LAST_SIGNAL_DATE.isoformat()}'
               OR public_availability_date > DATE '{LAST_SIGNAL_DATE.isoformat()}'
            """,
        )
        _require(bad_dates == 0, f"{view} escaped the predictor cutoff")

    _require(
        _scalar(connection, "SELECT count(*) FROM segments") == 0,
        "Segment features are frozen as unavailable but the segment table is nonempty",
    )
    duplicate_specs = (
        ("quarterly", "source_record_id"),
        ("annual", "source_record_id"),
        ("sec_filings", "source_record_id"),
        ("sec_items", "source_record_id"),
        ("filings_13f", "filing_id"),
        ("holdings_13f", "filing_id, source_row_ordinal"),
        ("restatement_events", "restatement_notification_key"),
        ("restatement_links", "restatement_notification_key"),
    )
    for view, keys in duplicate_specs:
        duplicates = _scalar(
            connection,
            f"""
            SELECT count(*)
            FROM (
                SELECT {keys}
                FROM {view}
                GROUP BY {keys}
                HAVING count(*) > 1
            )
            """,
        )
        _require(duplicates == 0, f"{view} has duplicate authoritative keys")

    orphan_restatements = _scalar(
        connection,
        """
        SELECT count(*)
        FROM restatement_events AS events
        FULL OUTER JOIN restatement_links AS links
          USING (restatement_notification_key)
        WHERE events.restatement_notification_key IS NULL
           OR links.restatement_notification_key IS NULL
        """,
    )
    _require(orphan_restatements == 0, "Restatement event/link grains differ")


def _aligned_signal_groups(
    inputs: FeatureStoreInputs,
) -> Iterator[list[tuple[dict[str, Any], dict[str, Any]]]]:
    spine = _row_iterator(inputs.path(SPINE_TABLE), _SPINE_COLUMNS)
    linkage = _row_iterator(inputs.path(LINKAGE_TABLE), _LINKAGE_COLUMNS)
    previous: tuple[dt.date, int] | None = None
    current_signal: dt.date | None = None
    current: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for spine_row, linkage_row in zip_longest(
        spine,
        linkage,
        fillvalue=_MISSING,
    ):
        _require(
            spine_row is not _MISSING and linkage_row is not _MISSING,
            "Spine and auxiliary linkage lengths differ",
        )
        signal = _date(spine_row["signal_date"], "spine signal_date")
        permno = _integer(spine_row["permno"], "spine permno")
        key = (signal, permno)
        _require(
            previous is None or key > previous,
            "Signal spine is not strictly sorted and unique",
        )
        link_key = (
            _date(linkage_row["signal_date"], "linkage signal_date"),
            _integer(linkage_row["permno"], "linkage permno"),
        )
        _require(link_key == key, "Auxiliary linkage is not row-aligned")
        if current_signal is not None and signal != current_signal:
            yield current
            current = []
        current_signal = signal
        current.append((spine_row, linkage_row))
        previous = key
    if current:
        yield current


class _AvailabilityStream:
    def __init__(
        self,
        path: Path,
        columns: Sequence[str],
        table: str,
    ) -> None:
        self._rows = _sorted_availability_iterator(path, columns)
        self._table = table
        self._pending: dict[str, Any] | None = None
        self._previous: tuple[dt.date, str] | None = None
        self._advance_pending()

    def _advance_pending(self) -> None:
        try:
            row = next(self._rows)
        except StopIteration:
            self._pending = None
            return
        availability = _date(
            row["public_availability_date"],
            f"{self._table} availability",
        )
        source_id = str(row["source_record_id"])
        key = (availability, source_id)
        _require(
            self._previous is None or key >= self._previous,
            f"{self._table} is not sorted by public availability",
        )
        self._previous = key
        self._pending = row

    def pop_through(self, signal_date: dt.date) -> Iterator[dict[str, Any]]:
        while (
            self._pending is not None
            and self._pending["public_availability_date"] <= signal_date
        ):
            row = self._pending
            self._advance_pending()
            yield row

    def close(self) -> None:
        close = getattr(self._rows, "close", None)
        if close is not None:
            close()


def _sorted_availability_iterator(
    path: Path,
    columns: Sequence[str],
    *,
    batch_size: int = 65_536,
) -> Iterator[dict[str, Any]]:
    """Externally sort one CIK-clustered SEC source once, then stream it."""

    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("PRAGMA threads=1")
        relation = (
            connection.read_parquet(str(path))
            .project(_quoted(columns))
            .order("public_availability_date, source_record_id")
        )
        # DuckDB 1.5 renamed the relation batch-streaming method.  Accept the
        # current API while retaining compatibility with the older runtime
        # under which the frozen feature implementation was first published.
        reader = (
            relation.fetch_record_batch(batch_size)
            if hasattr(relation, "fetch_record_batch")
            else relation.record_batch(batch_size)
        )
        for batch in reader:
            yield from batch.to_pylist()
    finally:
        connection.close()


class _SECWindowIndex:
    """One-pass latest-filing and trailing-event index."""

    def __init__(self, inputs: FeatureStoreInputs) -> None:
        self._filings = _AvailabilityStream(
            inputs.path(SEC_FILING_TABLE),
            _SEC_FILING_COLUMNS,
            SEC_FILING_TABLE,
        )
        self._items = _AvailabilityStream(
            inputs.path(SEC_ITEM_TABLE),
            _SEC_ITEM_COLUMNS,
            SEC_ITEM_TABLE,
        )
        self._latest_annual: dict[str, dict[str, Any]] = {}
        self._latest_interim: dict[str, dict[str, Any]] = {}
        self._events: dict[
            tuple[str, str], deque[dict[str, Any]]
        ] = defaultdict(deque)
        self._item_events: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self._last_signal: dt.date | None = None

    @staticmethod
    def _rank(row: Mapping[str, Any]) -> tuple[dt.date, dt.date, str]:
        return (
            row["public_availability_date"],
            row.get("filing_date") or dt.date.min,
            str(row["source_record_id"]),
        )

    def advance(self, signal_date: dt.date) -> None:
        _require(
            self._last_signal is None or signal_date > self._last_signal,
            "SEC context signals must advance strictly",
        )
        for row in self._filings.pop_through(signal_date):
            cik = normalize_text(row.get("cik"))
            form = normalize_text(row.get("form_family"))
            if cik is None or form is None:
                continue
            if form in ANNUAL_FORMS:
                incumbent = self._latest_annual.get(cik)
                if incumbent is None or self._rank(row) > self._rank(incumbent):
                    self._latest_annual[cik] = row
            elif form in INTERIM_FORMS:
                incumbent = self._latest_interim.get(cik)
                if incumbent is None or self._rank(row) > self._rank(incumbent):
                    self._latest_interim[cik] = row
            if form in {"8-K", "6-K"}:
                self._events[(cik, form)].append(row)
        for row in self._items.pop_through(signal_date):
            cik = normalize_text(row.get("cik"))
            form = normalize_text(row.get("form_family"))
            if cik is not None and form == "8-K":
                self._item_events[cik].append(row)

        lower = signal_date - dt.timedelta(days=90)
        for mapping in (self._events, self._item_events):
            for key in list(mapping):
                values = mapping[key]
                while values and values[0]["public_availability_date"] <= lower:
                    values.popleft()
                if not values:
                    del mapping[key]
        self._last_signal = signal_date

    def filing_rows(self, cik: str | None) -> tuple[Mapping[str, Any], ...]:
        normalized = normalize_text(cik)
        if normalized is None:
            return ()
        rows: list[Mapping[str, Any]] = []
        for latest in (
            self._latest_annual.get(normalized),
            self._latest_interim.get(normalized),
        ):
            if latest is not None:
                rows.append(latest)
        rows.extend(self._events.get((normalized, "8-K"), ()))
        rows.extend(self._events.get((normalized, "6-K"), ()))
        return tuple(rows)

    def item_rows(self, cik: str | None) -> tuple[Mapping[str, Any], ...]:
        normalized = normalize_text(cik)
        return (
            ()
            if normalized is None
            else tuple(self._item_events.get(normalized, ()))
        )

    def close(self) -> None:
        self._filings.close()
        self._items.close()


def _fundamental_groups(
    connection: duckdb.DuckDBPyConnection,
    signal_date: dt.date,
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[
    dict[tuple[str, int], list[dict[str, Any]]],
    dict[tuple[str, int], list[dict[str, Any]]],
]:
    matched = sorted(
        {
            (str(link["ccm_gvkey"]), int(spine["permno"]))
            for spine, link in rows
            if link["ccm_match_status"] == "MATCHED"
        }
    )
    if not matched:
        return {}, {}
    connection.register(
        "_ownership_fundamental_keys",
        pa.table(
            {
                "gvkey": pa.array(
                    [key[0] for key in matched],
                    type=pa.string(),
                ),
                "permno": pa.array(
                    [key[1] for key in matched],
                    type=pa.int64(),
                ),
            }
        ),
    )
    targets: tuple[
        tuple[str, Sequence[str], str, int],
        ...,
    ] = (
        (
            "quarterly",
            _QUARTERLY_METADATA_COLUMNS,
            '"fyearq" DESC, "fqtr" DESC, "datadate" DESC, "source_record_id" DESC',
            12,
        ),
        (
            "annual",
            _ANNUAL_METADATA_COLUMNS,
            '"fyear" DESC, "datadate" DESC, "source_record_id" DESC',
            3,
        ),
    )
    grouped: list[
        dict[tuple[str, int], list[dict[str, Any]]]
    ] = []
    try:
        for view, columns, newest_order, limit in targets:
            table = connection.execute(
                f"""
                WITH ranked AS (
                    SELECT {_quoted(columns, "source.")},
                           row_number() OVER (
                               PARTITION BY source.gvkey, source.permno
                               ORDER BY {newest_order}
                           ) AS recent_rank
                    FROM {view} AS source
                    INNER JOIN _ownership_fundamental_keys AS keys
                      ON source.gvkey = keys.gvkey
                     AND source.permno = keys.permno
                    WHERE source.standard_format_eligible IS TRUE
                      AND source.observation_date
                          <= DATE '{signal_date.isoformat()}'
                      AND source.public_availability_date
                          <= DATE '{signal_date.isoformat()}'
                )
                SELECT {_quoted(columns)}
                FROM ranked
                WHERE recent_rank <= {limit}
                ORDER BY gvkey, permno, datadate, source_record_id
                """
            ).fetch_arrow_table()
            by_key: dict[
                tuple[str, int], list[dict[str, Any]]
            ] = defaultdict(list)
            for item in table.to_pylist():
                by_key[(str(item["gvkey"]), int(item["permno"]))].append(item)
            grouped.append(dict(by_key))
    finally:
        connection.unregister("_ownership_fundamental_keys")
    return grouped[0], grouped[1]


def _fundamental_metadata(
    link: Mapping[str, Any],
    permno: int,
    signal_date: dt.date,
    quarterly: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    annual: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
) -> tuple[
    str | None,
    str | None,
    str | None,
    dt.date | None,
    dt.date | None,
    dt.date | None,
]:
    if link["ccm_match_status"] != "MATCHED":
        return None, None, None, None, None, None
    key = (str(link["ccm_gvkey"]), permno)
    bundle: FundamentalBundle | None = select_fundamental_bundle(
        quarterly.get(key, ()),
        annual.get(key, ()),
        signal_date,
    )
    if bundle is None:
        return None, None, None, None, None, None
    current = bundle.current_rows
    currencies = {
        normalize_text(row.get("currency_code"))
        for row in current
    }
    currency = (
        next(iter(currencies))
        if len(currencies) == 1 and None not in currencies
        else None
    )
    anchor = bundle.anchor
    stale = anchor.get("stale_after_date")
    return (
        normalize_text(anchor.get("fic")),
        normalize_text(anchor.get("loc")),
        currency,
        max(row["observation_date"] for row in current),
        max(row["public_availability_date"] for row in current),
        stale if isinstance(stale, dt.date) else None,
    )


def _load_identity(
    inputs: FeatureStoreInputs,
) -> dict[int, tuple[dict[str, Any], ...]]:
    by_permno: dict[int, list[dict[str, Any]]] = defaultdict(list)
    previous: tuple[int, dt.date, dt.date] | None = None
    for row in _row_iterator(inputs.path(IDENTITY_TABLE), _IDENTITY_COLUMNS):
        permno = _integer(row["permno"], "identity permno")
        start = _date(row["start_date"], "identity start_date")
        end = _date(row["end_date"], "identity end_date")
        _require(start <= end, "CRSP identity interval is reversed")
        key = (permno, start, end)
        _require(
            previous is None or key >= previous,
            "CRSP identity intervals are not sorted",
        )
        by_permno[permno].append(row)
        previous = key
    return {
        key: tuple(value)
        for key, value in by_permno.items()
    }


def _active_identity(
    identity: Mapping[int, Sequence[Mapping[str, Any]]],
    permno: int,
    day: dt.date,
) -> Mapping[str, Any] | None:
    active = [
        row
        for row in identity.get(permno, ())
        if row["start_date"] <= day <= row["end_date"]
    ]
    _require(
        len(active) <= 1,
        f"Ambiguous CRSP identity at {day} for permno {permno}",
    )
    return None if not active else active[0]


def _normalized_cusip8(value: Any) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    compact = "".join(character for character in text if character.isalnum())
    return compact[:8] if len(compact) >= 8 else None


@dataclass(frozen=True)
class _RestatementIndex:
    first_availability: dt.date | None
    rows: Mapping[int, tuple[Mapping[str, Any], ...]]
    dates: Mapping[int, tuple[dt.date, ...]]

    def trailing(
        self,
        permno: int,
        signal_date: dt.date,
    ) -> tuple[Mapping[str, Any], ...]:
        rows = self.rows.get(permno, ())
        dates = self.dates.get(permno, ())
        lower = signal_date - dt.timedelta(days=365)
        start = bisect_right(dates, lower)
        end = bisect_right(dates, signal_date)
        return tuple(rows[start:end])


def _load_restatements(
    connection: duckdb.DuckDBPyConnection,
) -> _RestatementIndex:
    first = connection.execute(
        "SELECT min(public_availability_date) FROM restatement_events"
    ).fetchone()
    first_availability = (
        None if first is None or first[0] is None else first[0]
    )
    table = connection.execute(
        f"""
        SELECT links.permno,
               {_quoted(_RESTATEMENT_EVENT_COLUMNS, "events.")}
        FROM restatement_events AS events
        INNER JOIN restatement_links AS links
          USING (restatement_notification_key)
        WHERE links.link_status = 'MATCHED'
          AND links.permno IS NOT NULL
        ORDER BY links.permno,
                 events.public_availability_date,
                 events.restatement_notification_key
        """
    ).fetch_arrow_table()
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in table.to_pylist():
        grouped[int(row["permno"])].append(row)
    frozen = {key: tuple(value) for key, value in grouped.items()}
    return _RestatementIndex(
        first_availability=first_availability,
        rows=frozen,
        dates={
            key: tuple(row["public_availability_date"] for row in value)
            for key, value in frozen.items()
        },
    )


def _load_13f_filings(
    inputs: FeatureStoreInputs,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _row_iterator(inputs.path(FILING_13F_TABLE), _FILING_13F_COLUMNS)
    )


def _prepare_mapped_13f_positions(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Map and aggregate equity lines once at each filing's report date."""

    connection.execute(
        """
        CREATE TEMP TABLE mapped_13f_position AS
        WITH normalized_identity AS (
            SELECT permno,
                   start_date,
                   end_date,
                   substr(
                       upper(
                           regexp_replace(
                               trim(cusip),
                               '[^A-Za-z0-9]',
                               '',
                               'g'
                           )
                       ),
                       1,
                       8
                   ) AS normalized_cusip8
            FROM identity
            WHERE cusip IS NOT NULL
              AND length(
                    regexp_replace(
                        trim(cusip),
                        '[^A-Za-z0-9]',
                        '',
                        'g'
                    )
                  ) >= 8
        ),
        candidate AS (
            SELECT holdings.filing_id,
                   holdings.source_row_ordinal,
                   holdings.normalized_cusip8,
                   holdings.reported_shares,
                   holdings.observation_date,
                   holdings.public_availability_date,
                   identity.permno AS target_permno,
                   count(DISTINCT identity.permno) OVER (
                       PARTITION BY
                           holdings.filing_id,
                           holdings.source_row_ordinal
                   ) AS candidate_count,
                   row_number() OVER (
                       PARTITION BY
                           holdings.filing_id,
                           holdings.source_row_ordinal
                       ORDER BY
                           identity.permno,
                           identity.start_date,
                           identity.end_date
                   ) AS identity_rank
            FROM holdings_13f AS holdings
            INNER JOIN normalized_identity AS identity
              ON holdings.normalized_cusip8 = identity.normalized_cusip8
             AND holdings.report_date
                 BETWEEN identity.start_date AND identity.end_date
            WHERE holdings.normalized_cusip8 IS NOT NULL
              AND (
                    holdings.put_call IS NULL
                    OR trim(holdings.put_call) = ''
                  )
        ),
        unambiguous AS (
            SELECT *
            FROM candidate
            WHERE candidate_count = 1
              AND identity_rank = 1
        )
        SELECT filing_id,
               target_permno,
               normalized_cusip8,
               CASE
                   WHEN bool_or(
                       reported_shares IS NULL
                       OR reported_shares < 0
                   )
                   THEN NULL
                   ELSE sum(CAST(reported_shares AS DOUBLE))
               END AS reported_shares,
               max(observation_date) AS observation_date,
               max(public_availability_date) AS public_availability_date
        FROM unambiguous
        GROUP BY filing_id, target_permno, normalized_cusip8
        """
    )
    connection.execute(
        """
        CREATE INDEX mapped_13f_filing_index
        ON mapped_13f_position(filing_id)
        """
    )


@dataclass(frozen=True)
class _ThirteenFContext:
    filing_rows: Mapping[int, tuple[Mapping[str, Any], ...]]
    holding_rows: Mapping[int, tuple[Mapping[str, Any], ...]]
    shares_rows: Mapping[int, tuple[Mapping[str, Any], ...]]
    identity_rows: Mapping[int, tuple[Mapping[str, Any], ...]]
    global_current_rows: tuple[Mapping[str, Any], ...]
    global_comparison_rows: tuple[Mapping[str, Any], ...]
    fallback_rows: tuple[Mapping[str, Any], ...]

    def filings_for(self, permno: int) -> tuple[Mapping[str, Any], ...]:
        return self.filing_rows.get(permno, self.fallback_rows)


def _thirteen_f_context(
    connection: duckdb.DuckDBPyConnection,
    *,
    signal_date: dt.date,
    signal_permnos: Sequence[int],
    filings: Sequence[Mapping[str, Any]],
    identity: Mapping[int, Sequence[Mapping[str, Any]]],
) -> _ThirteenFContext:
    _all, snapshots = _select_13f_snapshots(filings, signal_date)
    current_rows = tuple(
        current
        for manager, (current, _prior) in sorted(snapshots.items())
    )
    comparison_rows = tuple(
        row
        for manager, (current, prior) in sorted(snapshots.items())
        for row in (current, prior)
        if row is not None
    )
    eligible = [
        row
        for row in filings
        if row["observation_date"] <= signal_date
        and row["public_availability_date"] <= signal_date
    ]
    if snapshots:
        first_manager = sorted(snapshots)[0]
        current, prior = snapshots[first_manager]
        fallback = (current,) if prior is None else (prior, current)
    elif eligible:
        fallback = (
            max(
                eligible,
                key=lambda row: (
                    row["public_availability_date"],
                    row["filing_date"],
                    str(row["filing_id"]),
                ),
            ),
        )
    else:
        fallback = ()
    if not snapshots:
        return _ThirteenFContext(
            filing_rows={},
            holding_rows={},
            shares_rows={},
            identity_rows={},
            global_current_rows=current_rows,
            global_comparison_rows=comparison_rows,
            fallback_rows=fallback,
        )

    selected_by_id = {
        str(row["filing_id"]): row for row in comparison_rows
    }
    selected_ids = sorted(selected_by_id)
    connection.register(
        "_ownership_selected_filings",
        pa.table(
            {"filing_id": pa.array(selected_ids, type=pa.string())}
        ),
    )
    signal_set = sorted(set(signal_permnos))
    connection.register(
        "_ownership_signal_permnos",
        pa.table(
            {"permno": pa.array(signal_set, type=pa.int64())}
        ),
    )
    try:
        positions = connection.execute(
            """
            SELECT positions.target_permno,
                   positions.filing_id,
                   positions.normalized_cusip8,
                   positions.reported_shares,
                   positions.observation_date,
                   positions.public_availability_date
            FROM mapped_13f_position AS positions
            INNER JOIN _ownership_selected_filings AS filings
              USING (filing_id)
            INNER JOIN _ownership_signal_permnos AS signal
              ON positions.target_permno = signal.permno
            ORDER BY
                positions.target_permno,
                positions.filing_id,
                positions.normalized_cusip8
            """
        ).fetch_arrow_table()
    finally:
        connection.unregister("_ownership_selected_filings")
        connection.unregister("_ownership_signal_permnos")

    holdings_by_target: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    managers_by_target: dict[int, set[str]] = defaultdict(set)
    filing_manager = {
        str(current["filing_id"]): manager
        for manager, pair in snapshots.items()
        for current in pair
        if current is not None
    }
    for row in positions.to_pylist():
        target = int(row["target_permno"])
        filing_id = str(row["filing_id"])
        manager = filing_manager[filing_id]
        managers_by_target[target].add(manager)
        holdings_by_target[target].append(
            {
                "filing_id": filing_id,
                "normalized_cusip8": row["normalized_cusip8"],
                "reported_shares": row["reported_shares"],
                "put_call": None,
                "observation_date": row["observation_date"],
                "public_availability_date": row[
                    "public_availability_date"
                ],
            }
        )

    filings_by_target: dict[int, tuple[Mapping[str, Any], ...]] = {}
    for target, managers in managers_by_target.items():
        rows: list[Mapping[str, Any]] = []
        for manager in sorted(managers):
            current, prior = snapshots[manager]
            if prior is not None:
                rows.append(prior)
            rows.append(current)
        filings_by_target[target] = tuple(rows)

    targets = sorted(holdings_by_target)
    shares_by_target: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    if targets:
        connection.register(
            "_ownership_share_permnos",
            pa.table({"permno": pa.array(targets, type=pa.int64())}),
        )
        try:
            shares_table = connection.execute(
                f"""
                SELECT {_quoted(_SHARES_COLUMNS, "shares.")}
                FROM shares
                INNER JOIN _ownership_share_permnos AS targets
                  USING (permno)
                ORDER BY shares.permno, shares.start_date, shares.end_date
                """
            ).fetch_arrow_table()
        finally:
            connection.unregister("_ownership_share_permnos")
        for row in shares_table.to_pylist():
            shares_by_target[int(row["permno"])].append(row)

    return _ThirteenFContext(
        filing_rows=filings_by_target,
        holding_rows={
            key: tuple(value) for key, value in holdings_by_target.items()
        },
        shares_rows={
            key: tuple(value) for key, value in shares_by_target.items()
        },
        identity_rows={
            key: tuple(identity.get(key, ()))
            for key in holdings_by_target
        },
        global_current_rows=current_rows,
        global_comparison_rows=comparison_rows,
        fallback_rows=fallback,
    )


def _max_date(
    rows: Iterable[Mapping[str, Any]],
    field: str,
) -> dt.date | None:
    values = [
        row.get(field)
        for row in rows
        if isinstance(row.get(field), dt.date)
    ]
    return max(values) if values else None


def _min_stale(
    rows: Iterable[Mapping[str, Any]],
) -> dt.date | None:
    values = [
        row.get("stale_after_date")
        for row in rows
        if isinstance(row.get("stale_after_date"), dt.date)
    ]
    return min(values) if values else None


def _globalize_13f_lineage(
    result: FeatureFamilyResult,
    context: _ThirteenFContext,
    signal_date: dt.date,
    first_source_availability: dt.date | None,
) -> FeatureFamilyResult:
    values = result.values
    current = context.global_current_rows
    comparison = context.global_comparison_rows
    before_source = (
        first_source_availability is None
        or signal_date < first_source_availability
    )
    lineage = dict(result.lineage)
    for name in FEATURE_NAMES[:4]:
        contributors = (
            comparison
            if name == "institutional_ownership_change_1q"
            else current
        )
        if before_source:
            missing_status = "MISSING_SOURCE_UNAVAILABLE"
        elif not current:
            missing_status = "MISSING_STALE"
        elif (
            name == "institutional_ownership_change_1q"
            and values[name] is None
        ):
            missing_status = "MISSING_INSUFFICIENT_HISTORY"
        else:
            missing_status = "MISSING_INVALID_INPUT"
        lineage[name] = make_lineage(
            values[name],
            _max_date(contributors, "observation_date"),
            _max_date(contributors, "public_availability_date"),
            _min_stale(current),
            missing_status,
        )
    return FeatureFamilyResult(
        values=dict(values),
        lineage={name: lineage[name] for name in FEATURE_NAMES},
    )


def _load_ff12_mapping(policy_path: Path) -> Mapping[str, Any]:
    with policy_path.open("rb") as handle:
        policy = tomllib.load(handle)
    raw_path = Path(policy["ff12_mapping"]["path"]).expanduser()
    path = (
        raw_path.resolve()
        if raw_path.is_absolute()
        else (PROJECT_ROOT / raw_path).resolve()
    )
    value = json.loads(path.read_text("utf-8"))
    _require(isinstance(value, dict), "FF12 mapping is not a JSON object")
    return value


def _build_row(
    *,
    spine: Mapping[str, Any],
    link: Mapping[str, Any],
    signal_date: dt.date,
    quarterly: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    annual: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    identity: Mapping[int, Sequence[Mapping[str, Any]]],
    sec: _SECWindowIndex,
    restatements: _RestatementIndex,
    thirteen_f: _ThirteenFContext,
    first_13f_availability: dt.date | None,
    ff12_mapping: Mapping[str, Any],
) -> FeatureMaterializationRow:
    permno = _integer(spine["permno"], "spine permno")
    identity_row = _active_identity(identity, permno, signal_date)
    _require(
        identity_row is not None,
        f"Signal-spine permno {permno} has no date-valid CRSP identity",
    )
    _require(
        normalize_text(identity_row.get("primary_exchange"))
        == normalize_text(spine.get("primary_exchange"))
        and normalize_text(identity_row.get("share_type"))
        == normalize_text(spine.get("share_type")),
        f"Signal-spine identity differs from CRSP interval for permno {permno}",
    )
    (
        fundamental_fic,
        fundamental_loc,
        reporting_currency,
        fundamental_observation,
        fundamental_availability,
        fundamental_stale,
    ) = _fundamental_metadata(
        link,
        permno,
        signal_date,
        quarterly,
        annual,
    )
    current_cusip = _normalized_cusip8(identity_row.get("cusip"))
    positively_covered = (
        current_cusip is not None
        and restatements.first_availability is not None
        and signal_date >= restatements.first_availability
    )
    result = build_ownership_event_feature_result(
        target_permno=permno,
        signal_date=signal_date,
        filing_13f_rows=thirteen_f.filings_for(permno),
        holding_13f_rows=thirteen_f.holding_rows.get(permno, ()),
        identity_intervals=thirteen_f.identity_rows.get(
            permno,
            identity.get(permno, ()),
        ),
        shares_intervals=thirteen_f.shares_rows.get(permno, ()),
        sec_match_status=str(link["sec_match_status"]),
        cik=None if link.get("sec_cik") is None else str(link["sec_cik"]),
        sec_filing_rows=sec.filing_rows(link.get("sec_cik")),
        sec_item_rows=sec.item_rows(link.get("sec_cik")),
        restatement_positively_covered=positively_covered,
        restatement_event_rows=restatements.trailing(permno, signal_date),
        identity_row=identity_row,
        fundamental_fic=fundamental_fic,
        fundamental_loc=fundamental_loc,
        reporting_currency=reporting_currency,
        fundamental_observation_date=fundamental_observation,
        fundamental_public_availability_date=fundamental_availability,
        fundamental_stale_after_date=fundamental_stale,
        ff12_mapping=ff12_mapping,
    )
    result = _globalize_13f_lineage(
        result,
        thirteen_f,
        signal_date,
        first_13f_availability,
    )
    _require(
        tuple(result.values) == FEATURE_NAMES
        and tuple(result.lineage) == FEATURE_NAMES,
        "Ownership/event result is not the exact 32-feature family",
    )
    for name, lineage in result.lineage.items():
        if lineage.value_status == "PRESENT":
            _require(
                lineage.max_observation_date <= signal_date
                and lineage.max_public_availability_date <= signal_date,
                f"Future ownership/event lineage escaped: {name}",
            )
            _require(
                lineage.stale_after_date is None
                or signal_date <= lineage.stale_after_date,
                f"Stale ownership/event lineage escaped: {name}",
            )
        value = result.values[name]
        if isinstance(value, float):
            _require(math.isfinite(value), f"Nonfinite feature escaped: {name}")
    for name in FEATURE_NAMES[18:22]:
        _require(
            result.values[name] is None
            and result.lineage[name].value_status
            == "MISSING_SOURCE_UNAVAILABLE",
            "Structurally unavailable segment feature was populated",
        )
    _require(
        result.values["latest_interim_financial_filing_form"]
        in {"10-Q", "UNKNOWN"},
        "Generic 6-K was reclassified as an interim financial filing",
    )
    _require(
        isinstance(spine["long_eligible"], bool)
        and isinstance(spine["short_eligible"], bool),
        "Spine eligibility flags must be Boolean",
    )
    return FeatureMaterializationRow(
        signal_date=signal_date,
        permno=permno,
        permco=_integer(spine["permco"], "spine permco"),
        long_eligible=spine["long_eligible"],
        short_eligible=spine["short_eligible"],
        primary_exchange=(
            None
            if spine.get("primary_exchange") is None
            else str(spine["primary_exchange"])
        ),
        share_type=(
            None if spine.get("share_type") is None else str(spine["share_type"])
        ),
        result=result,
    )


def iter_ownership_event_feature_batches(
    inputs: FeatureStoreInputs,
    *,
    batch_size: int = 20_000,
    policy_path: Path = DEFAULT_POLICY,
    build_identity: FeatureStoreBuildIdentity | None = None,
) -> Iterator[tuple[FeatureMaterializationRow, ...]]:
    """Yield deterministic, full-spine, point-in-time 32-feature batches."""

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
    required = (
        (SPINE_TABLE, _SPINE_COLUMNS),
        (LINKAGE_TABLE, _LINKAGE_COLUMNS),
        (QUARTERLY_TABLE, _QUARTERLY_METADATA_COLUMNS),
        (ANNUAL_TABLE, _ANNUAL_METADATA_COLUMNS),
        (IDENTITY_TABLE, _IDENTITY_COLUMNS),
        (SHARES_TABLE, _SHARES_COLUMNS),
        (SEC_FILING_TABLE, _SEC_FILING_COLUMNS),
        (SEC_ITEM_TABLE, _SEC_ITEM_COLUMNS),
        (FILING_13F_TABLE, _FILING_13F_COLUMNS),
        (HOLDING_13F_TABLE, _HOLDING_13F_COLUMNS),
        (RESTATEMENT_EVENT_TABLE, _RESTATEMENT_EVENT_COLUMNS),
        (RESTATEMENT_LINK_TABLE, _RESTATEMENT_LINK_COLUMNS),
    )
    for table, columns in required:
        _required_columns(inputs.path(table), columns, table)
    _required_columns(inputs.path(SEGMENT_TABLE), (), SEGMENT_TABLE)
    ff12_mapping = _load_ff12_mapping(policy_path)

    def generate() -> Iterator[tuple[FeatureMaterializationRow, ...]]:
        connection = duckdb.connect(database=":memory:")
        sec: _SECWindowIndex | None = None
        try:
            connection.execute("PRAGMA threads=1")
            connection.execute("SET preserve_insertion_order=false")
            _create_views(connection, inputs)
            _validate_source_gates(connection, inputs)
            identity = _load_identity(inputs)
            restatements = _load_restatements(connection)
            filings_13f = _load_13f_filings(inputs)
            first_13f_availability = min(
                (
                    row["public_availability_date"]
                    for row in filings_13f
                ),
                default=None,
            )
            sec = _SECWindowIndex(inputs)
            mapped_13f_ready = False
            pending: list[FeatureMaterializationRow] = []
            produced = 0
            for source_rows in _aligned_signal_groups(inputs):
                signal_date = _date(
                    source_rows[0][0]["signal_date"],
                    "signal date",
                )
                sec.advance(signal_date)
                quarterly, annual = _fundamental_groups(
                    connection,
                    signal_date,
                    source_rows,
                )
                if (
                    first_13f_availability is not None
                    and signal_date >= first_13f_availability
                    and not mapped_13f_ready
                ):
                    _prepare_mapped_13f_positions(connection)
                    mapped_13f_ready = True
                if mapped_13f_ready:
                    thirteen_f = _thirteen_f_context(
                        connection,
                        signal_date=signal_date,
                        signal_permnos=[
                            int(spine["permno"])
                            for spine, _link in source_rows
                        ],
                        filings=filings_13f,
                        identity=identity,
                    )
                else:
                    thirteen_f = _ThirteenFContext(
                        filing_rows={},
                        holding_rows={},
                        shares_rows={},
                        identity_rows={},
                        global_current_rows=(),
                        global_comparison_rows=(),
                        fallback_rows=(),
                    )
                for spine_row, link_row in source_rows:
                    pending.append(
                        _build_row(
                            spine=spine_row,
                            link=link_row,
                            signal_date=signal_date,
                            quarterly=quarterly,
                            annual=annual,
                            identity=identity,
                            sec=sec,
                            restatements=restatements,
                            thirteen_f=thirteen_f,
                            first_13f_availability=first_13f_availability,
                            ff12_mapping=ff12_mapping,
                        )
                    )
                    if len(pending) == batch_size:
                        produced += len(pending)
                        yield tuple(pending)
                        pending = []
            if pending:
                produced += len(pending)
                yield tuple(pending)
            _require(
                produced == inputs.tables[SPINE_TABLE].rows,
                "Ownership/event materializer did not preserve the full spine",
            )
        finally:
            if sec is not None:
                sec.close()
            connection.close()

    return generate()


# Concise orchestration alias.
materialize_ownership_events = iter_ownership_event_feature_batches
