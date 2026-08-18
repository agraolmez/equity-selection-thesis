"""Point-in-time CRSP market context for the Stage 11 validation window.

The verified Stage 2 CRSP tables intentionally end in 2014.  Stage 11 must not
change those tables (or their hashes), so this module builds a separate
extension from the same frozen raw snapshot.  It reuses the hash-authenticated
CSV router, exact duplicate handling, and locked Arrow field transforms from
Stage 2, while replacing only *table-level* development-scope metadata.

This module does not construct labels, select a model, or calculate P&L.  Its
predictor-context payload stops at the final permitted signal date,
2019-10-31.  The final holding-period observations from 2019-11-01 through
2019-12-02 are deliberately reserved for the separately authority-gated
outcome path and cannot be materialized by this module.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import threading
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data import phase_02_crsp_typed_backbone as typed
from portfolio_bot.data import phase_02_crsp_typed_backbone_discovery as discovery
from portfolio_bot.data import phase_02_crsp_typed_backbone_transforms as transforms
from portfolio_bot.data import phase_02_crsp_return_reconciliation as reconciliation
from portfolio_bot.data.phase_02_crsp_typed_backbone_core import (
    SchemaRegistry,
    TableSpec,
)
from portfolio_bot.validation import validation_return_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATABASE_ROOT = Path("/Users/agraemirolmez/Desktop/Thesis")

CONTRACT_ID = "phase05-stage11-validation-market-extension-v1"
ENTRY_MARKET_PROJECTION_CONTRACT_ID = "phase06-stage13-exact-entry-market-projection-v1"
DEVELOPMENT_CONTEXT_START = dt.date(2000, 1, 1)
EXTENSION_START = dt.date(2015, 1, 1)
LAST_VALIDATION_SIGNAL = dt.date(2019, 10, 31)
LAST_VALIDATION_EXIT = dt.date(2019, 12, 2)
OUTCOME_RESERVE_START = dt.date(2019, 11, 1)
FIRST_CONTEXT_SIGNAL = dt.date(2014, 11, 28)

ROW_GROUP_SIZE = typed.ROW_GROUP_SIZE
DEFAULT_BLOCK_SIZE_BYTES = 64 << 20

_MARKET_TABLES = (
    "crsp_security_day",
    "crsp_security_day_return_staging",
    "crsp_security_month_crosscheck",
    "crsp_security_month_return_staging",
)
_INTERVAL_TABLES = (
    "crsp_identity_interval",
    "crsp_shares_interval",
)
_ALL_TABLES = (*_INTERVAL_TABLES, *_MARKET_TABLES)

# This is deliberately a closed, return-free projection of the frozen daily
# consumer schema.  The four raw return fields are not requested from the CSV
# reader, decoded, transformed, staged, or written by the entry-market path.
_ENTRY_MARKET_RAW_COLUMNS = frozenset(
    {
        "PERMNO",
        "PERMCO",
        "SecInfoStartDt",
        "SecInfoEndDt",
        "CUSIP",
        "PrimaryExch",
        "ConditionalType",
        "TradingStatusFlg",
        "USIncFlg",
        "IssuerType",
        "SecurityType",
        "SecuritySubType",
        "ShareType",
        "Ticker",
        "SICCD",
        "NAICS",
        "ICBIndustry",
        "DlyCalDt",
        "DlyPrc",
        "DlyPrcFlg",
        "DlyCap",
        "DlyVol",
        "DlyClose",
        "DlyLow",
        "DlyHigh",
        "DlyBid",
        "DlyAsk",
        "DlyOpen",
        "DlyPrcVol",
    }
)
_RAW_RETURN_COLUMNS = frozenset({"DlyRet", "DlyRetx", "DlyRetMissFlg", "DlyRetDurFlg"})

# The frozen 2025-12-01 source has three daily rows that are closed delisting
# tombstones rather than usable entry-market observations.  They deliberately
# remain in the return-free daily projection, matching the Stage 2 left-row
# policy, while the unchanged final_market_inputs inner join excludes them
# because no active shares interval exists.  Bind the closed predicate to the
# aggregate key-set digest without disclosing the underlying security ids.
_ENTRY_MARKET_TOMBSTONE_PREDICATE_ID = (
    "stage13-entry-market-closed-delisting-tombstone-v1"
)
_ENTRY_MARKET_TOMBSTONE_EXPECTED_ROWS = 3
_ENTRY_MARKET_TOMBSTONE_EXPECTED_KEY_SET_SHA256 = (
    "5af68213b30cf617bf7b96eac18f1d62a12ca0cda2f3e19a27c132996d7a4b66"
)
# Exact return-free consumer canonicalization facts for the same pinned raw
# snapshot and exact 2025-12-01 scope.  Stable reuse authenticates these
# values without reopening the 12.7 GB raw source.
_ENTRY_MARKET_EXPECTED_DAILY_PROJECTION_RETAINED_ROWS = 10_605
_ENTRY_MARKET_EXPECTED_DAILY_CANONICAL_ROWS = 10_596
_ENTRY_MARKET_EXPECTED_DAILY_CONSUMER_EXACT_DUPLICATE_KEY_GROUPS = 9
_ENTRY_MARKET_EXPECTED_DAILY_CONSUMER_EXACT_DUPLICATE_SURPLUS_ROWS = 9
_ENTRY_MARKET_EXPECTED_DAILY_CONSUMER_CONFLICTING_DUPLICATE_GROUPS = 0
_ENTRY_MARKET_EXPECTED_DAILY_OUTPUT_DUPLICATE_KEYS = 0
_ENTRY_MARKET_TOMBSTONE_KEY_DOMAIN = (
    b"stage13-entry-market-zero-share-tombstone-keys-v1\x00"
)
_ENTRY_MARKET_COVERAGE_JOIN_RULE = (
    "equal permno and inclusive start_date <= entry_date <= end_date"
)
_ENTRY_MARKET_ZERO_MATCH_POLICY = (
    "retain only exact closed delisting tombstones upstream; unchanged "
    "final_market_inputs inner join excludes them"
)
_ENTRY_MARKET_FINAL_JOIN_SEMANTICS = (
    "unchanged_inner_join_requires_one_active_shares_interval"
)
_ENTRY_MARKET_TOMBSTONE_NULL_FIELDS = (
    "market_cap_thousand_usd",
    "market_cap_usd",
    "volume_shares",
    "close_usd_per_share",
    "low_usd_per_share",
    "high_usd_per_share",
    "bid_usd_per_share",
    "ask_usd_per_share",
    "open_usd_per_share",
    "vendor_price_volume_usd",
    "screening_dollar_volume_usd",
)
_ENTRY_MARKET_TOMBSTONE_FALSE_FIELDS = (
    "screening_price_valid",
    "market_cap_valid",
    "liquidity_session_valid",
)

Progress = Callable[[str, Mapping[str, Any]], None]
_RECONCILIATION_SCOPE_LOCK = threading.Lock()


class ValidationMarketExtensionError(RuntimeError):
    """Raised when the isolated Stage 11 market extension is not auditable."""


@dataclasses.dataclass(frozen=True, slots=True)
class ValidationMarketScope:
    """Closed date limits for predictor context and validation accounting."""

    extension_start: dt.date = EXTENSION_START
    last_signal: dt.date = LAST_VALIDATION_SIGNAL
    last_exit: dt.date = LAST_VALIDATION_EXIT
    first_context_signal: dt.date = FIRST_CONTEXT_SIGNAL

    def __post_init__(self) -> None:
        values = (
            self.extension_start,
            self.first_context_signal,
            self.last_signal,
            self.last_exit,
        )
        if any(type(value) is not dt.date for value in values):
            raise ValidationMarketExtensionError(
                "market scope dates must be date values"
            )
        if not (
            self.first_context_signal
            < self.extension_start
            <= self.last_signal
            < self.last_exit
        ):
            raise ValidationMarketExtensionError("Stage 11 market scope is not ordered")
        if self.extension_start != EXTENSION_START:
            raise ValidationMarketExtensionError("extension must begin on 2015-01-01")
        if self.last_signal != LAST_VALIDATION_SIGNAL:
            raise ValidationMarketExtensionError("last signal must remain 2019-10-31")
        if self.last_exit != LAST_VALIDATION_EXIT:
            raise ValidationMarketExtensionError("last exit must remain 2019-12-02")
        if OUTCOME_RESERVE_START != self.last_signal + dt.timedelta(days=1):
            raise ValidationMarketExtensionError(
                "outcome reserve must follow the last signal"
            )


DEFAULT_SCOPE = ValidationMarketScope()


@dataclasses.dataclass(frozen=True, slots=True)
class ValidationTableReceipt:
    name: str
    path: str
    rows: int
    row_groups: int
    size_bytes: int
    sha256: str
    schema_sha256: str
    minimum_date: str | None
    maximum_date: str | None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationMarketExtensionError(message)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Publish the authority receipt last or leave no apparent completion."""

    temporary = path.with_name(
        f".{path.name}.partial-{os.getpid()}-{hashlib.sha256(payload).hexdigest()[:12]}"
    )
    _require(
        not temporary.exists() and not temporary.is_symlink(),
        "entry-market receipt temporary path already exists",
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def validation_table_schema(
    spec: TableSpec,
    *,
    minimum_date: dt.date,
    maximum_date: dt.date,
) -> pa.Schema:
    """Keep the frozen fields but replace development-only table metadata."""

    _require(type(spec) is TableSpec, "validation schema requires an exact TableSpec")
    _require(type(minimum_date) is dt.date, "minimum_date must be a date")
    _require(type(maximum_date) is dt.date, "maximum_date must be a date")
    _require(minimum_date <= maximum_date, "validation schema dates are reversed")
    frozen = spec.arrow_schema()
    # Preserve every frozen semantic guard (for example
    # ``consumer_access_permitted=false``) while replacing only the
    # development-period row-count/scope identity.
    metadata = dict(frozen.metadata or {})
    metadata.pop(b"expected_rows", None)
    metadata.update(
        {
            b"contract_id": CONTRACT_ID.encode("utf-8"),
            b"grain": spec.grain.encode("utf-8"),
            b"primary_key": _canonical_json(list(spec.primary_key))[:-1],
            b"primary_source": spec.primary_source.encode("utf-8"),
            b"role": ("stage11_validation_extension_of_" + spec.role).encode("utf-8"),
            b"scope_maximum_date": maximum_date.isoformat().encode("ascii"),
            b"scope_minimum_date": minimum_date.isoformat().encode("ascii"),
            b"scope_semantics": (
                b"retain_intervals_overlapping_closed_scope; endpoint values may extend outside"
                if spec.name in _INTERVAL_TABLES
                else b"retain_observation_dates_inside_closed_scope"
            ),
            b"sort_key": _canonical_json(list(spec.sort_key))[:-1],
            b"table": spec.name.encode("utf-8"),
        }
    )
    return pa.schema(list(frozen), metadata=metadata)


def entry_market_projection_schema(
    spec: TableSpec,
    *,
    minimum_date: dt.date,
    maximum_date: dt.date,
    exact_entry_dates_sha256: str,
) -> pa.Schema:
    """Retain frozen fields while identifying the exact-entry-only role."""

    schema = validation_table_schema(
        spec,
        minimum_date=minimum_date,
        maximum_date=maximum_date,
    )
    _require(
        len(exact_entry_dates_sha256) == 64
        and set(exact_entry_dates_sha256) <= set("0123456789abcdef"),
        "exact entry-date digest is invalid",
    )
    metadata = dict(schema.metadata or {})
    interval = spec.name == "crsp_shares_interval"
    metadata.update(
        {
            b"contract_id": ENTRY_MARKET_PROJECTION_CONTRACT_ID.encode("ascii"),
            b"role": (
                b"stage13_prebarrier_entry_market_shares_consumer_only"
                if interval
                else b"stage13_prebarrier_exact_entry_market_consumer_only"
            ),
            b"scope_semantics": (
                b"retain_intervals_overlapping_the_exact_entry_date_domain"
                if interval
                else b"retain_only_the_exact_frozen_entry_dates"
            ),
            b"exact_entry_dates_sha256": exact_entry_dates_sha256.encode("ascii"),
            b"return_columns_permitted": b"false",
            b"labels_outcomes_accounting_permitted": b"false",
        }
    )
    return schema.with_metadata(metadata)


def _transform_entry_market_consumer(
    raw_batch: pa.RecordBatch | pa.Table,
    consumer_spec: TableSpec,
) -> pa.Table:
    """Apply the frozen Stage 2 consumer transform without return columns."""

    transforms._require_locked_spec(consumer_spec, "crsp_security_day")
    raw, rows = transforms._raw_strings(raw_batch, _ENTRY_MARKET_RAW_COLUMNS)
    decimal_6 = pa.decimal128(38, 6)
    decimal_2 = pa.decimal128(38, 2)
    decimal_1 = pa.decimal128(38, 1)
    permno = transforms._positive_int64(raw["PERMNO"], "daily PERMNO")
    permco = transforms._positive_int64(raw["PERMCO"], "daily PERMCO", optional=True)
    interval_start = transforms._date32(
        raw["SecInfoStartDt"], "daily security-information start"
    )
    interval_end = transforms._date32(
        raw["SecInfoEndDt"], "daily security-information end"
    )
    transforms._assert_ordered(
        interval_start,
        interval_end,
        "Daily security-information interval",
    )
    calendar_date = transforms._date32(raw["DlyCalDt"], "daily calendar date")
    diagnostic_price = transforms._decimal128_exact(raw["DlyPrc"], decimal_6, "DlyPrc")
    market_cap = transforms._decimal128_exact(raw["DlyCap"], decimal_2, "DlyCap")
    volume = transforms._nonnegative_int64(raw["DlyVol"], "DlyVol")
    close = transforms._decimal128_exact(raw["DlyClose"], decimal_6, "DlyClose")
    low = transforms._decimal128_exact(raw["DlyLow"], decimal_6, "DlyLow")
    high = transforms._decimal128_exact(raw["DlyHigh"], decimal_6, "DlyHigh")
    bid = transforms._decimal128_exact(raw["DlyBid"], decimal_6, "DlyBid")
    ask = transforms._decimal128_exact(raw["DlyAsk"], decimal_6, "DlyAsk")
    opened = transforms._decimal128_exact(raw["DlyOpen"], decimal_6, "DlyOpen")
    vendor_price_volume = transforms._decimal128_exact(
        raw["DlyPrcVol"], decimal_1, "DlyPrcVol"
    )
    normalized_cap = transforms._scale_decimal_1000(
        market_cap, decimal_2, "market_cap_usd"
    )
    price_valid = transforms._positive_decimal(close)
    market_cap_valid = transforms._positive_decimal(market_cap)
    liquidity_valid = pc.and_(price_valid, pc.is_valid(volume))
    dollar_volume = transforms._dollar_volume(
        close,
        volume,
        liquidity_valid,
        decimal_6,
    )
    consumer = {
        "permno": permno,
        "source_permco": permco,
        "source_security_info_start_date": interval_start,
        "source_security_info_end_date": interval_end,
        "source_cusip": raw["CUSIP"],
        "source_primary_exchange": raw["PrimaryExch"],
        "conditional_type": raw["ConditionalType"],
        "source_trading_status_flag": raw["TradingStatusFlg"],
        "source_us_incorporation_flag": raw["USIncFlg"],
        "source_issuer_type": raw["IssuerType"],
        "source_security_type": raw["SecurityType"],
        "source_security_subtype": raw["SecuritySubType"],
        "source_share_type": raw["ShareType"],
        "source_ticker": raw["Ticker"],
        "source_sic_code": raw["SICCD"],
        "source_naics_code": raw["NAICS"],
        "source_icb_industry": raw["ICBIndustry"],
        "calendar_date": calendar_date,
        "diagnostic_price_usd_per_share": diagnostic_price,
        "price_flag": raw["DlyPrcFlg"],
        "market_cap_thousand_usd": market_cap,
        "market_cap_usd": normalized_cap,
        "volume_shares": volume,
        "close_usd_per_share": close,
        "low_usd_per_share": low,
        "high_usd_per_share": high,
        "bid_usd_per_share": bid,
        "ask_usd_per_share": ask,
        "open_usd_per_share": opened,
        "vendor_price_volume_usd": vendor_price_volume,
        "screening_dollar_volume_usd": dollar_volume,
        "screening_price_valid": price_valid,
        "market_cap_valid": market_cap_valid,
        "liquidity_session_valid": liquidity_valid,
    }
    return transforms._build_table(consumer_spec, consumer, rows)


def _entry_market_tombstone_key_set_sha256(
    keys: Iterable[tuple[int, dt.date]],
) -> str:
    """Hash the sorted closed tombstone keys without serializing identifiers."""

    normalized = tuple(sorted(keys))
    _require(
        len(normalized) == len(set(normalized))
        and all(
            type(permno) is int and type(calendar_date) is dt.date
            for permno, calendar_date in normalized
        ),
        "entry-market tombstone keys are invalid or duplicate",
    )
    digest = hashlib.sha256()
    digest.update(_ENTRY_MARKET_TOMBSTONE_KEY_DOMAIN)
    for permno, calendar_date in normalized:
        digest.update(permno.to_bytes(8, byteorder="big", signed=True))
        digest.update(calendar_date.isoformat().encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _is_closed_entry_market_tombstone(
    row: Mapping[str, object],
    *,
    entry_date: dt.date,
) -> bool:
    """Recognize only the frozen, fact-free December delisting row shape."""

    return bool(
        row.get("calendar_date") == entry_date
        and row.get("source_security_info_start_date") == dt.date(2025, 11, 29)
        and row.get("source_security_info_end_date") == entry_date
        and row.get("source_trading_status_flag") == "D"
        and row.get("source_primary_exchange") == "X"
        and row.get("conditional_type") == "N/A"
        and row.get("source_security_type") == "N/A"
        and row.get("source_security_subtype") == "UNK"
        and row.get("source_share_type") == "N/A"
        and row.get("price_flag") in {"DA", "DP"}
        and all(row.get(name) is None for name in _ENTRY_MARKET_TOMBSTONE_NULL_FIELDS)
        and all(row.get(name) is False for name in _ENTRY_MARKET_TOMBSTONE_FALSE_FIELDS)
    )


def _entry_market_share_coverage(
    daily: pa.Table,
    shares: pa.Table,
    *,
    entry_date: dt.date,
) -> dict[str, object]:
    """Recompute the closed daily-to-shares coverage contract.

    The function consumes only the two return-free projection tables.  It is
    shared by the builder and stable cache reauthentication so receipt values
    can never substitute for observed coverage.
    """

    _require(type(entry_date) is dt.date, "entry-market coverage date is invalid")
    required_daily = {
        "permno",
        "calendar_date",
        "source_security_info_start_date",
        "source_security_info_end_date",
        "source_primary_exchange",
        "conditional_type",
        "source_trading_status_flag",
        "source_security_type",
        "source_security_subtype",
        "source_share_type",
        "price_flag",
        *_ENTRY_MARKET_TOMBSTONE_NULL_FIELDS,
        *_ENTRY_MARKET_TOMBSTONE_FALSE_FIELDS,
    }
    required_shares = {"permno", "start_date", "end_date"}
    _require(
        required_daily <= set(daily.column_names)
        and required_shares <= set(shares.column_names),
        "entry-market coverage columns differ",
    )
    daily_rows = daily.select(sorted(required_daily)).to_pylist()
    share_rows = shares.select(sorted(required_shares)).to_pylist()
    daily_keys = tuple((row["permno"], row["calendar_date"]) for row in daily_rows)
    _require(
        all(
            type(permno) is int and type(calendar_date) is dt.date
            for permno, calendar_date in daily_keys
        )
        and len(daily_keys) == len(set(daily_keys)),
        "entry-market daily coverage keys are invalid or duplicate",
    )

    intervals: dict[int, list[tuple[dt.date, dt.date]]] = {}
    for row in share_rows:
        permno = row["permno"]
        start = row["start_date"]
        end = row["end_date"]
        _require(
            type(permno) is int
            and type(start) is dt.date
            and type(end) is dt.date
            and start <= entry_date <= end,
            "entry-market shares contain an invalid active interval",
        )
        intervals.setdefault(permno, []).append((start, end))

    match_counts = tuple(
        sum(start <= calendar_date <= end for start, end in intervals.get(permno, ()))
        for permno, calendar_date in daily_keys
    )
    one_match_rows = sum(value == 1 for value in match_counts)
    zero_match_keys = {
        key for key, value in zip(daily_keys, match_counts, strict=True) if value == 0
    }
    multiple_match_rows = sum(value > 1 for value in match_counts)
    daily_permnos = {permno for permno, _ in daily_keys}
    active_share_rows_without_daily = sum(
        row["permno"] not in daily_permnos for row in share_rows
    )
    predicate_keys = {
        (row["permno"], row["calendar_date"])
        for row in daily_rows
        if _is_closed_entry_market_tombstone(row, entry_date=entry_date)
    }
    tombstone_digest = _entry_market_tombstone_key_set_sha256(predicate_keys)

    _require(
        multiple_match_rows == 0,
        "entry-market daily row has multiple active shares intervals",
    )
    _require(
        active_share_rows_without_daily == 0,
        "entry-market active shares row lacks a daily consumer row",
    )
    _require(
        predicate_keys == zero_match_keys,
        "entry-market zero-share rows differ from the closed tombstone predicate",
    )
    _require(
        len(predicate_keys) == _ENTRY_MARKET_TOMBSTONE_EXPECTED_ROWS
        and tombstone_digest == _ENTRY_MARKET_TOMBSTONE_EXPECTED_KEY_SET_SHA256,
        "entry-market closed tombstone count or key digest differs",
    )
    _require(
        one_match_rows + len(zero_match_keys) == len(daily_rows)
        and one_match_rows + active_share_rows_without_daily == len(share_rows),
        "entry-market share coverage does not conserve rows",
    )
    return {
        "join_rule": _ENTRY_MARKET_COVERAGE_JOIN_RULE,
        "population_rows": len(daily_rows),
        "one_match_rows": one_match_rows,
        "zero_match_rows": len(zero_match_keys),
        "multiple_match_rows": multiple_match_rows,
        "active_share_rows": len(share_rows),
        "active_share_rows_without_daily": active_share_rows_without_daily,
        "zero_match_policy": _ENTRY_MARKET_ZERO_MATCH_POLICY,
        "tombstone_predicate_id": _ENTRY_MARKET_TOMBSTONE_PREDICATE_ID,
        "tombstone_rows": len(predicate_keys),
        "tombstone_key_set_sha256": tombstone_digest,
        "final_market_inputs_join_semantics": _ENTRY_MARKET_FINAL_JOIN_SEMANTICS,
    }


class _ValidationParquetSink:
    """Deterministic writer that preserves frozen field metadata and types."""

    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.path = path
        self.schema = schema
        self.rows = 0
        self.nulls: Counter[str] = Counter()
        self._pending: list[pa.Table] = []
        self._pending_rows = 0
        self._writer = pq.ParquetWriter(
            path,
            schema,
            version="2.6",
            compression="zstd",
            compression_level=9,
            use_dictionary=True,
            write_statistics=True,
            data_page_version="2.0",
            store_schema=True,
        )

    def add(self, table: pa.Table) -> None:
        expected_fields = self.schema.remove_metadata()
        observed_fields = table.schema.remove_metadata()
        _require(
            observed_fields.equals(expected_fields, check_metadata=True),
            f"field schema differs for {self.path.name}",
        )
        table = table.replace_schema_metadata(self.schema.metadata)
        self.rows += table.num_rows
        for name, column in zip(table.column_names, table.columns, strict=True):
            self.nulls[name] += column.null_count
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
        if self._pending_rows:
            table = pa.concat_tables(self._pending).combine_chunks()
            self._writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
            self._pending.clear()
            self._pending_rows = 0

    def close(self) -> None:
        self._flush()
        self._writer.close()

    def __enter__(self) -> "_ValidationParquetSink":
        return self

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False


def _receipt(
    path: Path,
    *,
    name: str,
    date_column: str,
) -> ValidationTableReceipt:
    file = pq.ParquetFile(path)
    minimum: dt.date | None = None
    maximum: dt.date | None = None
    for group in range(file.num_row_groups):
        values = file.read_row_group(group, columns=[date_column])[date_column]
        if values.null_count:
            raise ValidationMarketExtensionError(
                f"date column contains nulls: {name}.{date_column}"
            )
        if len(values):
            extrema = pc.min_max(values).as_py()
            group_minimum = extrema["min"]
            group_maximum = extrema["max"]
            minimum = group_minimum if minimum is None else min(minimum, group_minimum)
            maximum = group_maximum if maximum is None else max(maximum, group_maximum)
    return ValidationTableReceipt(
        name=name,
        path=path.name,
        rows=file.metadata.num_rows,
        row_groups=file.metadata.num_row_groups,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        schema_sha256=_schema_sha256(file.schema_arrow),
        minimum_date=None if minimum is None else minimum.isoformat(),
        maximum_date=None if maximum is None else maximum.isoformat(),
    )


def _database_confined_path(
    path: Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    """Reject lexical escapes and symlinks in every existing path component."""

    root = Path(os.path.abspath(DATABASE_ROOT.expanduser()))
    candidate = Path(os.path.abspath(path.expanduser()))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValidationMarketExtensionError(
            f"{label} escaped the database root"
        ) from exc
    _require(root.is_dir(), "database root is absent")
    current = root
    for part in relative.parts:
        current /= part
        _require(not current.is_symlink(), f"symlinked {label} path is prohibited")
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ValidationMarketExtensionError(
            f"{label} does not resolve inside the database root"
        ) from exc
    return candidate


def _safe_output_directory(path: Path) -> Path:
    candidate = _database_confined_path(
        path,
        label="validation market output",
        must_exist=False,
    )
    if candidate.exists():
        _require(candidate.is_dir(), "validation market output is not a directory")
        _require(
            not any(candidate.iterdir()), "refusing a nonempty market output directory"
        )
    else:
        candidate.mkdir(parents=True)
    return candidate


def _emit(progress: Progress | None, event: str, **details: Any) -> None:
    if progress is not None:
        progress(event, MappingProxyType(details))


@contextmanager
def _return_reconciliation_scope(scope: ValidationMarketScope) -> Iterable[None]:
    """Temporarily scope the frozen pure reconciliation algorithm.

    The old module exposes its date window as constants rather than settings.
    A process-local lock prevents concurrent mutation, and the values are
    restored even after failure.  Stage 2 files and source code remain
    unchanged; the Stage 11 receipt binds this adapter and the reused source.
    """

    acquired = _RECONCILIATION_SCOPE_LOCK.acquire(blocking=False)
    _require(acquired, "a return-reconciliation scope is already active")
    previous = (
        reconciliation.SCOPE_START,
        reconciliation.SCOPE_END,
        reconciliation._read_events,
        reconciliation._distribution_mapping,
        reconciliation._delisting_treatments,
        reconciliation._build_monthly,
    )
    reconciliation.SCOPE_START = scope.extension_start
    # This adapter is predictor-only.  Outcome-period return bytes require a
    # distinct authority-gated path and can never be reached through here.
    reconciliation.SCOPE_END = scope.last_signal
    reconciliation._read_events = validation_return_adapter.validation_read_events
    reconciliation._distribution_mapping = (
        validation_return_adapter.validation_distribution_mapping
    )
    reconciliation._delisting_treatments = (
        validation_return_adapter.validation_delisting_treatments
    )
    validation_return_adapter.reset_monthly_transport_audit()
    reconciliation._build_monthly = validation_return_adapter.validation_build_monthly
    try:
        yield
    finally:
        (
            reconciliation.SCOPE_START,
            reconciliation.SCOPE_END,
            reconciliation._read_events,
            reconciliation._distribution_mapping,
            reconciliation._delisting_treatments,
            reconciliation._build_monthly,
        ) = previous
        _RECONCILIATION_SCOPE_LOCK.release()


def _registry_with_expected_rows(
    registry: SchemaRegistry,
    expected_rows: Mapping[str, int],
) -> SchemaRegistry:
    _require(set(expected_rows) == set(registry.tables), "return row-count map differs")
    tables = {
        name: dataclasses.replace(specification, expected_rows=int(expected_rows[name]))
        for name, specification in registry.tables.items()
    }
    _require(
        all(value >= 0 for value in expected_rows.values()), "negative return row count"
    )
    return dataclasses.replace(registry, tables=MappingProxyType(tables))


def build_entry_market_projection(
    output_directory: Path,
    *,
    entry_dates: Sequence[dt.date],
    scope: ValidationMarketScope = DEFAULT_SCOPE,
    block_size_bytes: int = DEFAULT_BLOCK_SIZE_BYTES,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Build an exact-date consumer-only market projection.

    This path authenticates the same frozen daily and shares sources used by
    Stage 2, but the CSV reader requests no raw return fields and no return,
    monthly, label, accounting, or outcome artifact is materialized.
    """

    _require(scope == DEFAULT_SCOPE, "only the frozen market scope is supported")
    dates = tuple(entry_dates)
    _require(
        dates
        and all(type(value) is dt.date for value in dates)
        and dates == tuple(sorted(set(dates))),
        "entry-market dates must be nonempty, sorted, and unique",
    )
    _require(
        len(dates) == 1,
        "entry-market projection v1 requires exactly one frozen entry date",
    )
    _require(
        dates[0] >= scope.extension_start and dates[-1] <= scope.last_exit,
        "entry-market dates escape the closed market scope",
    )
    _require(
        type(block_size_bytes) is int
        and block_size_bytes >= discovery._MIN_ARROW_BLOCK_SIZE,
        "CSV block size is below the safe minimum",
    )
    date_strings = tuple(value.isoformat() for value in dates)
    exact_dates_sha256 = hashlib.sha256(_canonical_json(list(date_strings))).hexdigest()
    output = _safe_output_directory(output_directory)
    locked = typed.load_production_settings(block_size_bytes=block_size_bytes)
    _require(
        locked.raw_root.resolve() == (DATABASE_ROOT / "raw").resolve(),
        "raw root differs",
    )

    daily_spec = locked.sources["daily"]
    daily_columns = tuple(
        name for name in daily_spec.header if name in _ENTRY_MARKET_RAW_COLUMNS
    )
    _require(
        set(daily_columns) == set(_ENTRY_MARKET_RAW_COLUMNS)
        and not (set(daily_columns) & set(_RAW_RETURN_COLUMNS)),
        "entry-market raw consumer column set differs",
    )
    daily_table_spec = locked.registry.tables["crsp_security_day"]
    daily_schema = entry_market_projection_schema(
        daily_table_spec,
        minimum_date=dates[0],
        maximum_date=dates[-1],
        exact_entry_dates_sha256=exact_dates_sha256,
    )
    daily_path = output / "crsp_security_day.parquet"
    daily_metrics = typed.CanonicalizationMetrics()
    wanted_dates = pa.array(dates, type=pa.date32())

    def scoped_daily_batches() -> Iterable[pa.RecordBatch]:
        with discovery._AuthenticatedSource.open(locked.raw_root, daily_spec) as source:
            with source.batches(daily_columns, block_size_bytes) as batches:
                for batch in batches:
                    daily_metrics.full_rows += batch.num_rows
                    observed_dates = discovery._strict_date32(
                        batch.column(batch.schema.get_field_index("DlyCalDt")),
                        "entry-market daily date",
                    )
                    retained = batch.filter(
                        pc.is_in(observed_dates, value_set=wanted_dates)
                    )
                    daily_metrics.retained_rows += retained.num_rows
                    if retained.num_rows:
                        yield retained

    _emit(progress, "entry_market_daily_started", entry_dates=date_strings)
    with _ValidationParquetSink(daily_path, daily_schema) as sink:
        for raw in typed.canonicalize_daily_batches(
            scoped_daily_batches(), daily_metrics
        ):
            sink.add(_transform_entry_market_consumer(raw, daily_table_spec))
    _require(
        daily_metrics.conflicting_groups == 0
        and daily_metrics.retained_rows
        == daily_metrics.canonical_rows + daily_metrics.duplicate_surplus_rows
        and daily_metrics.duplicate_groups <= daily_metrics.duplicate_surplus_rows
        and (daily_metrics.duplicate_groups == 0)
        == (daily_metrics.duplicate_surplus_rows == 0),
        "entry-market daily canonicalization metrics differ",
    )
    _require(
        daily_metrics.retained_rows
        == _ENTRY_MARKET_EXPECTED_DAILY_PROJECTION_RETAINED_ROWS
        and daily_metrics.canonical_rows
        == _ENTRY_MARKET_EXPECTED_DAILY_CANONICAL_ROWS
        and daily_metrics.duplicate_groups
        == _ENTRY_MARKET_EXPECTED_DAILY_CONSUMER_EXACT_DUPLICATE_KEY_GROUPS
        and daily_metrics.duplicate_surplus_rows
        == _ENTRY_MARKET_EXPECTED_DAILY_CONSUMER_EXACT_DUPLICATE_SURPLUS_ROWS
        and daily_metrics.conflicting_groups
        == _ENTRY_MARKET_EXPECTED_DAILY_CONSUMER_CONFLICTING_DUPLICATE_GROUPS,
        "entry-market daily canonicalization facts differ from the pinned source",
    )
    daily_receipt = _receipt(
        daily_path,
        name="crsp_security_day",
        date_column="calendar_date",
    )
    daily_dates = pq.read_table(daily_path, columns=["calendar_date"])[
        "calendar_date"
    ].combine_chunks()
    observed_date_strings = tuple(
        value.isoformat() for value in sorted(pc.unique(daily_dates).to_pylist())
    )
    _require(
        daily_receipt.rows > 0 and observed_date_strings == date_strings,
        "entry-market daily projection does not cover the exact entry dates",
    )
    rows_by_date = Counter(value.isoformat() for value in daily_dates.to_pylist())
    _require(
        set(rows_by_date) == set(date_strings)
        and all(rows_by_date[value] > 0 for value in date_strings),
        "an exact entry date has no market rows",
    )
    daily_coverage = pq.read_table(daily_path, use_threads=False).combine_chunks()
    daily_output_keys = (
        daily_coverage.select(["permno", "calendar_date"])
        .group_by(["permno", "calendar_date"])
        .aggregate([])
    )
    daily_output_duplicate_keys = daily_coverage.num_rows - daily_output_keys.num_rows
    _require(
        daily_metrics.canonical_rows == daily_receipt.rows
        and daily_metrics.canonical_rows == sum(rows_by_date.values())
        and daily_output_duplicate_keys == 0,
        "entry-market canonical daily output differs",
    )
    _emit(progress, "entry_market_daily_complete", rows=daily_receipt.rows)

    shares_spec = locked.sources["shares"]
    shares_table_spec = locked.registry.tables["crsp_shares_interval"]
    shares_schema = entry_market_projection_schema(
        shares_table_spec,
        minimum_date=dates[0],
        maximum_date=dates[-1],
        exact_entry_dates_sha256=exact_dates_sha256,
    )
    shares_path = output / "crsp_shares_interval.parquet"
    shares_metrics = typed.CanonicalizationMetrics()
    shares_index_builder = typed.IntervalIndexBuilder()
    shares_settings = dataclasses.replace(
        locked,
        start_date=dates[0],
        end_date=dates[-1],
    )
    _emit(progress, "entry_market_shares_started", entry_dates=date_strings)
    with discovery._AuthenticatedSource.open(locked.raw_root, shares_spec) as source:
        batches = typed._sorted_interval_batches(
            typed._scoped_interval_batches(
                source,
                start_column="ShrStartDt",
                end_column="ShrEndDt",
                settings=shares_settings,
                metrics=shares_metrics,
            ),
            start_column="ShrStartDt",
            end_column="ShrEndDt",
            label="entry-market shares",
        )
        with _ValidationParquetSink(shares_path, shares_schema) as sink:
            for raw in batches:
                shares_index_builder.add(
                    raw,
                    start_column="ShrStartDt",
                    end_column="ShrEndDt",
                )
                sink.add(transforms.transform_shares(raw, shares_table_spec))
    shares_index = shares_index_builder.finish("entry-market shares")
    _require(
        shares_index.duplicate_key_groups == 0
        and shares_index.overlap_edges == 0
        and shares_index.gap_edges == 0,
        "entry-market shares interval structure differs",
    )
    shares_receipt = _receipt(
        shares_path,
        name="crsp_shares_interval",
        date_column="start_date",
    )
    _require(shares_receipt.rows > 0, "entry-market shares projection is empty")
    shares_keys = pq.read_table(
        shares_path,
        columns=["permno", "start_date", "end_date"],
        use_threads=False,
    ).combine_chunks()
    entry_date = dates[0]
    _require(
        shares_keys["start_date"].null_count == 0
        and shares_keys["end_date"].null_count == 0
        and all(
            start <= entry_date <= end
            for start, end in zip(
                shares_keys["start_date"].to_pylist(),
                shares_keys["end_date"].to_pylist(),
                strict=True,
            )
        ),
        "entry-market shares projection retained a non-entry interval",
    )
    share_coverage = _entry_market_share_coverage(
        daily_coverage,
        shares_keys,
        entry_date=entry_date,
    )
    _emit(progress, "entry_market_shares_complete", rows=shares_receipt.rows)

    expected_files = {
        "crsp_security_day.parquet",
        "crsp_shares_interval.parquet",
    }
    _require(
        {path.name for path in output.iterdir()} == expected_files,
        "entry-market projection emitted an unauthorized artifact",
    )
    result = {
        "schema_version": 1,
        "contract_id": ENTRY_MARKET_PROJECTION_CONTRACT_ID,
        "status": "EXACT_ENTRY_MARKET_CONSUMER_PROJECTION_COMPLETE",
        "materialization_role": "prebarrier_exact_entry_market_consumer_only",
        "scope": {
            "exact_entry_dates": list(date_strings),
            "exact_entry_dates_sha256": exact_dates_sha256,
            "minimum_entry_date": date_strings[0],
            "maximum_entry_date": date_strings[-1],
            "decision_return_cutoff_unchanged": scope.last_signal.isoformat(),
            "reserved_outcome_end": scope.last_exit.isoformat(),
        },
        "raw_access": {
            "daily_requested_columns": list(daily_columns),
            "raw_return_columns_requested": [],
            "monthly_sources_opened": False,
            "return_reconciliation_invoked": False,
            "labels_outcomes_or_accounting_opened": False,
        },
        "source_bindings": {
            name: {
                "filename": locked.sources[name].filename,
                "sha256": locked.sources[name].sha256,
                "size_bytes": locked.sources[name].size_bytes,
                "full_file_rows": locked.sources[name].full_file_rows,
            }
            for name in ("daily", "shares")
        },
        "tables": {
            "crsp_security_day": daily_receipt.as_dict(),
            "crsp_shares_interval": shares_receipt.as_dict(),
        },
        "entry_rows_by_date": dict(sorted(rows_by_date.items())),
        "canonicalization": {
            "daily_projection_retained_rows": daily_metrics.retained_rows,
            "daily_canonical_rows": daily_metrics.canonical_rows,
            "daily_consumer_exact_duplicate_key_groups": (
                daily_metrics.duplicate_groups
            ),
            "daily_consumer_exact_duplicate_surplus_rows": (
                daily_metrics.duplicate_surplus_rows
            ),
            "daily_consumer_conflicting_duplicate_groups": (
                daily_metrics.conflicting_groups
            ),
            "daily_output_duplicate_keys": daily_output_duplicate_keys,
            "shares_duplicate_key_groups": shares_index.duplicate_key_groups,
            "shares_overlap_edges": shares_index.overlap_edges,
            "shares_gap_edges": shares_index.gap_edges,
            "shares_non_entry_interval_rows": 0,
        },
        "share_coverage": share_coverage,
        "authorizations": {
            "exact_entry_market_consumer_facts_only": True,
            "returns_labels_outcomes_accounting_authorized": False,
            "portfolio_weights_trades_pnl_or_performance_authorized": False,
        },
    }
    receipt_path = output / "entry_market_projection_receipt.json"
    _atomic_bytes(receipt_path, _canonical_json(result))
    _require(
        {path.name for path in output.iterdir()}
        == expected_files | {receipt_path.name},
        "entry-market projection final artifact set differs",
    )
    return result


def build_typed_market_extension(
    output_directory: Path,
    *,
    scope: ValidationMarketScope = DEFAULT_SCOPE,
    block_size_bytes: int = DEFAULT_BLOCK_SIZE_BYTES,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Build six isolated typed tables without changing Stage 2 artifacts.

    Daily/monthly facts retain only the post-development predictor extension
    through 2019-10-31.  Identity and shares are rebuilt over the full
    predictor context because new listings and new effective-dated records
    after 2014 do not exist in the Stage 2 payload.  The outcome reserve from
    2019-11-01 through 2019-12-02 is never read by this builder.
    """

    _require(scope == DEFAULT_SCOPE, "only the frozen Stage 11 scope is supported")
    _require(
        type(block_size_bytes) is int
        and block_size_bytes >= discovery._MIN_ARROW_BLOCK_SIZE,
        "CSV block size is below the safe minimum",
    )
    output = _safe_output_directory(output_directory)
    locked = typed.load_production_settings(block_size_bytes=block_size_bytes)
    _require(
        locked.raw_root.resolve() == (DATABASE_ROOT / "raw").resolve(),
        "raw root differs",
    )

    fact_settings = dataclasses.replace(
        locked,
        start_date=scope.extension_start,
        end_date=scope.last_signal,
    )
    context_settings = dataclasses.replace(
        locked,
        start_date=DEVELOPMENT_CONTEXT_START,
        end_date=scope.last_signal,
    )

    receipts: dict[str, ValidationTableReceipt] = {}
    interval_structures: dict[str, dict[str, int]] = {}

    # Interval dimensions are small and are sorted in-memory by the reused
    # Stage 2 router.  Future interval endpoints are retained exactly as in the
    # frozen Stage 2 contract; only starts through the validation exit can
    # enter the effective-date join.
    for source_name, table_name, start_column, end_column, transform in (
        (
            "identity",
            "crsp_identity_interval",
            "SecInfoStartDt",
            "SecInfoEndDt",
            transforms.transform_identity,
        ),
        (
            "shares",
            "crsp_shares_interval",
            "ShrStartDt",
            "ShrEndDt",
            transforms.transform_shares,
        ),
    ):
        _emit(progress, f"{source_name}_started")
        metrics = typed.CanonicalizationMetrics()
        source_spec = locked.sources[source_name]
        schema = validation_table_schema(
            locked.registry.tables[table_name],
            minimum_date=DEVELOPMENT_CONTEXT_START,
            maximum_date=scope.last_signal,
        )
        path = output / f"{table_name}.parquet"
        interval_builder = typed.IntervalIndexBuilder(
            discovery.IDENTITY_PAYLOAD_FIELDS if source_name == "identity" else ()
        )
        with discovery._AuthenticatedSource.open(
            locked.raw_root, source_spec
        ) as source:
            batches = typed._sorted_interval_batches(
                typed._scoped_interval_batches(
                    source,
                    start_column=start_column,
                    end_column=end_column,
                    settings=context_settings,
                    metrics=metrics,
                ),
                start_column=start_column,
                end_column=end_column,
                label=source_name,
            )
            with _ValidationParquetSink(path, schema) as sink:
                for raw in batches:
                    interval_builder.add(
                        raw,
                        start_column=start_column,
                        end_column=end_column,
                    )
                    sink.add(transform(raw, locked.registry.tables[table_name]))
        interval_index = interval_builder.finish(source_name)
        _require(
            interval_index.duplicate_key_groups == 0,
            f"{source_name} contains duplicate interval keys",
        )
        _require(
            interval_index.overlap_edges == 0,
            f"{source_name} contains overlapping intervals",
        )
        _require(
            interval_index.gap_edges == 0,
            f"{source_name} contains interval gaps inconsistent with the frozen contract",
        )
        interval_structures[source_name] = {
            "rows": len(interval_index.permnos),
            "unique_permnos": interval_index.unique_permnos,
            "duplicate_key_groups": interval_index.duplicate_key_groups,
            "contiguous_edges": interval_index.contiguous_edges,
            "gap_edges": interval_index.gap_edges,
            "overlap_edges": interval_index.overlap_edges,
        }
        receipts[table_name] = _receipt(
            path,
            name=table_name,
            date_column=("start_date"),
        )
        _emit(progress, f"{source_name}_complete", rows=receipts[table_name].rows)

    # The market streams remain date-first and streaming.  The old locked
    # transform authenticates every retained payload field before this module
    # replaces the table-level development metadata.
    for source_name, date_column, canonicalizer, pairs in (
        (
            "daily",
            "DlyCalDt",
            typed.canonicalize_daily_batches,
            (
                ("crsp_security_day", transforms.transform_daily, 0, "calendar_date"),
                (
                    "crsp_security_day_return_staging",
                    transforms.transform_daily,
                    1,
                    "calendar_date",
                ),
            ),
        ),
        (
            "monthly",
            "MthCalDt",
            typed.canonicalize_monthly_batches,
            (
                (
                    "crsp_security_month_crosscheck",
                    transforms.transform_monthly,
                    0,
                    "calendar_month",
                ),
                (
                    "crsp_security_month_return_staging",
                    transforms.transform_monthly,
                    1,
                    "calendar_month",
                ),
            ),
        ),
    ):
        _emit(progress, f"{source_name}_started")
        metrics = typed.CanonicalizationMetrics()
        source_spec = locked.sources[source_name]
        schemas = {
            name: validation_table_schema(
                locked.registry.tables[name],
                minimum_date=scope.extension_start,
                maximum_date=scope.last_signal,
            )
            for name, _function, _position, _date in pairs
        }
        paths = {name: output / f"{name}.parquet" for name in schemas}
        sinks = {
            name: _ValidationParquetSink(paths[name], schemas[name]) for name in schemas
        }
        try:
            with discovery._AuthenticatedSource.open(
                locked.raw_root, source_spec
            ) as source:
                scoped = typed._scoped_market_batches(
                    source,
                    date_column=date_column,
                    settings=fact_settings,
                    metrics=metrics,
                )
                for raw in canonicalizer(scoped, metrics):
                    if source_name == "daily":
                        converted = transforms.transform_daily(
                            raw,
                            locked.registry.tables["crsp_security_day"],
                            locked.registry.tables["crsp_security_day_return_staging"],
                        )
                    else:
                        converted = transforms.transform_monthly(
                            raw,
                            locked.registry.tables["crsp_security_month_crosscheck"],
                            locked.registry.tables[
                                "crsp_security_month_return_staging"
                            ],
                        )
                    for name, _function, position, _date in pairs:
                        sinks[name].add(converted[position])
                    if (
                        metrics.canonical_rows
                        and metrics.canonical_rows % 2_000_000 < raw.num_rows
                    ):
                        _emit(
                            progress,
                            f"{source_name}_progress",
                            canonical_rows=metrics.canonical_rows,
                        )
        finally:
            for sink in sinks.values():
                sink.close()
        for name, _function, _position, output_date_column in pairs:
            receipts[name] = _receipt(
                paths[name], name=name, date_column=output_date_column
            )
            observed = receipts[name]
            _require(observed.minimum_date is not None, f"{name} is unexpectedly empty")
            _require(
                observed.minimum_date >= scope.extension_start.isoformat(),
                f"{name} begins before the predictor extension",
            )
            _require(
                observed.maximum_date is not None
                and observed.maximum_date <= scope.last_signal.isoformat(),
                f"{name} reaches the reserved outcome interval",
            )
        _require(
            metrics.conflicting_groups == 0, f"{source_name} has conflicting duplicates"
        )
        _emit(
            progress,
            f"{source_name}_complete",
            retained_rows=metrics.retained_rows,
            canonical_rows=metrics.canonical_rows,
        )

    _require(set(receipts) == set(_ALL_TABLES), "typed extension receipt set differs")
    _require(
        receipts["crsp_security_day"].rows
        == receipts["crsp_security_day_return_staging"].rows,
        "daily consumer and return extension rows differ",
    )
    _require(
        receipts["crsp_security_month_crosscheck"].rows
        == receipts["crsp_security_month_return_staging"].rows,
        "monthly consumer and return extension rows differ",
    )

    result = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "TYPED_VALIDATION_PREDICTOR_MARKET_EXTENSION_BUILT_PENDING_VERIFICATION",
        "materialization_role": "point_in_time_predictor_context_only",
        "scope": {
            "extension_start": scope.extension_start.isoformat(),
            "last_signal": scope.last_signal.isoformat(),
            "last_exit": scope.last_exit.isoformat(),
            "first_context_signal": scope.first_context_signal.isoformat(),
            "maximum_payload_date": scope.last_signal.isoformat(),
            "post_2019_10_31_market_observation_or_interval_start_rows_retained": 0,
            "future_interval_endpoints_may_be_retained_for_asof_join_audit": True,
            "reserved_outcome_interval": {
                "start": OUTCOME_RESERVE_START.isoformat(),
                "end": scope.last_exit.isoformat(),
                "materialized": False,
                "separate_unblind_authority_required": True,
            },
        },
        "source_bindings": {
            name: {
                "filename": specification.filename,
                "sha256": specification.sha256,
                "size_bytes": specification.size_bytes,
                "full_file_rows": specification.full_file_rows,
            }
            for name, specification in sorted(locked.sources.items())
        },
        "tables": {
            name: receipt.as_dict() for name, receipt in sorted(receipts.items())
        },
        "interval_structure": interval_structures,
        "authorizations": {
            "predictor_context_materialization_only": True,
            "validation_labels_or_selection_authorized": False,
            "portfolio_weights_trades_pnl_or_performance_authorized": False,
            "final_2020_2025_access_authorized": False,
        },
    }
    receipt_path = output / "typed_extension_receipt.json"
    _atomic_bytes(receipt_path, _canonical_json(result))
    return result


def build_return_extension(
    typed_extension_directory: Path,
    output_base_directory: Path,
    *,
    scope: ValidationMarketScope = DEFAULT_SCOPE,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Apply the frozen exactly-once return engine to predictor context only.

    ``output_base_directory`` is a parent containing distinct ``interim``,
    ``quality``, and ``artifact`` children.  The Stage 2 publisher is invoked
    with pointer publication disabled, so no earlier authority pointer can
    move.  The returned receipt is still pending Stage 11 independent
    verification.  This function is structurally unable to reach the reserved
    2019-11-01 through 2019-12-02 outcome interval.
    """

    _require(scope == DEFAULT_SCOPE, "only the frozen Stage 11 scope is supported")
    typed_root = _database_confined_path(
        typed_extension_directory,
        label="typed extension",
        must_exist=True,
    )
    output_base = _database_confined_path(
        output_base_directory,
        label="return output",
        must_exist=False,
    )
    database_root = Path(os.path.abspath(DATABASE_ROOT.expanduser()))
    _require(
        typed_root.is_dir() and not typed_root.is_symlink(), "typed extension is absent"
    )
    required = {
        "crsp_security_day_return_staging.parquet",
        "crsp_security_month_return_staging.parquet",
    }
    _require(
        required.issubset({path.name for path in typed_root.iterdir()}),
        "typed return-extension inputs are incomplete",
    )
    _require(not output_base.exists(), "refusing an existing return output base")
    _require(
        output_base != typed_root
        and output_base not in typed_root.parents
        and typed_root not in output_base.parents,
        "typed input and return output paths may not overlap",
    )

    # Load and authenticate the frozen raw/calendar/schema authorities.  The
    # rehearsal output root is intentionally outside production publication.
    base = reconciliation.load_settings(
        output_database_root=database_root / "cache/phase_05_validation/settings_probe"
    )
    daily_rows = pq.ParquetFile(
        typed_root / "crsp_security_day_return_staging.parquet"
    ).metadata.num_rows
    monthly_rows = pq.ParquetFile(
        typed_root / "crsp_security_month_return_staging.parquet"
    ).metadata.num_rows

    probe = dataclasses.replace(base, typed_root=typed_root)
    with _return_reconciliation_scope(scope):
        distributions, delistings, _context = reconciliation._read_events(probe)
    expected_rows = {
        "crsp_distribution_event": len(distributions),
        "crsp_delisting_event": len(delistings),
        "crsp_security_day_return": int(daily_rows),
        "crsp_security_month_return_reconciliation": int(monthly_rows),
    }
    registry = _registry_with_expected_rows(base.schema, expected_rows)
    production_return_pointer = reconciliation.PROJECT_ROOT / (
        "artifacts/phase_02_canonical_panel/"
        "LATEST_CRSP_RETURN_RECONCILIATION_PRIMARY.json"
    )
    production_return_pointer_before = (
        _sha256(production_return_pointer)
        if production_return_pointer.is_file()
        else None
    )
    binding = {
        "contract_id": CONTRACT_ID,
        "materialization_role": "point_in_time_predictor_context_only",
        "scope_start": scope.extension_start.isoformat(),
        "scope_end": scope.last_signal.isoformat(),
        "reserved_outcome_start": OUTCOME_RESERVE_START.isoformat(),
        "reserved_outcome_end": scope.last_exit.isoformat(),
        "typed_daily_staging_sha256": _sha256(
            typed_root / "crsp_security_day_return_staging.parquet"
        ),
        "typed_monthly_staging_sha256": _sha256(
            typed_root / "crsp_security_month_return_staging.parquet"
        ),
        "reused_reconciliation_source_sha256": _sha256(
            Path(reconciliation.__file__).resolve()
        ),
        "stage11_return_adapter_source_sha256": _sha256(
            Path(validation_return_adapter.__file__).resolve()
        ),
        "stage11_market_extension_source_sha256": _sha256(Path(__file__).resolve()),
        "expected_rows": expected_rows,
        "production_return_pointer_sha256_before": production_return_pointer_before,
    }
    build_id = (
        "stage11-return-extension-"
        + hashlib.sha256(_canonical_json(binding)).hexdigest()[:24]
    )
    settings = reconciliation.Settings(
        database_root=database_root,
        output_database_root=database_root,
        is_production=False,
        raw_root=base.raw_root,
        typed_root=typed_root,
        calendar_root=base.calendar_root,
        schema=registry,
        empirical_lock=base.empirical_lock,
        build_id=build_id,
        interim_dir=output_base / "interim" / build_id,
        quality_dir=output_base / "quality" / build_id,
        artifact_dir=output_base / "artifact" / build_id,
        pointer_path=PROJECT_ROOT
        / "artifacts/phase_05_validation/PROHIBITED_UNUSED_RETURN_POINTER.json",
    )
    _emit(progress, "return_reconciliation_started", build_id=build_id)
    with _return_reconciliation_scope(scope):
        pointer = reconciliation.build(settings, publish_pointer=False)
        monthly_transport_audit = validation_return_adapter.monthly_transport_audit()
    _require(
        not settings.pointer_path.exists(),
        "return extension unexpectedly published an authority pointer",
    )
    production_return_pointer_after = (
        _sha256(production_return_pointer)
        if production_return_pointer.is_file()
        else None
    )
    _require(
        production_return_pointer_after == production_return_pointer_before,
        "production return pointer changed during predictor reconciliation",
    )
    authoritative = settings.interim_dir / "crsp_security_day_return.parquet"
    _require(authoritative.is_file(), "authorized return extension is absent")
    receipt = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "VALIDATION_PREDICTOR_RETURN_EXTENSION_BUILT_PENDING_VERIFICATION",
        "build_id": build_id,
        "materialization_role": "point_in_time_predictor_context_only",
        "scope": {
            "start": scope.extension_start.isoformat(),
            "end": scope.last_signal.isoformat(),
            "reserved_outcome_interval": {
                "start": OUTCOME_RESERVE_START.isoformat(),
                "end": scope.last_exit.isoformat(),
                "materialized": False,
                "separate_unblind_authority_required": True,
            },
            "first_prohibited_final_entry": "2020-01-02",
        },
        "bindings": binding,
        "pointer_publication_attempted": False,
        "production_return_pointer_sha256_after": production_return_pointer_after,
        "rehearsal_pointer": pointer,
        "monthly_transport_audit": monthly_transport_audit,
        "authoritative_daily_return": _receipt(
            authoritative,
            name="crsp_security_day_return",
            date_column="calendar_date",
        ).as_dict(),
        "authorizations": {
            "point_in_time_predictor_context_only": True,
            "validation_label_or_model_selection_access_authorized": False,
            "portfolio_or_performance_authorized": False,
            "final_2020_2025_access_authorized": False,
        },
    }
    receipt_path = output_base / "return_extension_receipt.json"
    _atomic_bytes(receipt_path, _canonical_json(receipt))
    _emit(progress, "return_reconciliation_complete", build_id=build_id)
    return receipt


__all__ = [
    "CONTRACT_ID",
    "DEFAULT_SCOPE",
    "DEVELOPMENT_CONTEXT_START",
    "ENTRY_MARKET_PROJECTION_CONTRACT_ID",
    "EXTENSION_START",
    "FIRST_CONTEXT_SIGNAL",
    "LAST_VALIDATION_EXIT",
    "LAST_VALIDATION_SIGNAL",
    "OUTCOME_RESERVE_START",
    "ValidationMarketExtensionError",
    "ValidationMarketScope",
    "ValidationTableReceipt",
    "build_entry_market_projection",
    "build_return_extension",
    "build_typed_market_extension",
    "entry_market_projection_schema",
    "validation_table_schema",
]
