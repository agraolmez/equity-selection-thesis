"""Build the development CRSP return/distribution/delisting gate.

Inputs are the independently verified typed CRSP Parquets plus the small raw
event ledgers.  The 12 GB raw daily CSV is deliberately not decoded again.
The build is development-only and never reads validation/final-period payloads.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import tomllib
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data.database_location import resolve_recorded_database_path
from portfolio_bot.data.phase_02_crsp_return_policy import (
    KNOWN_MONTHLY_FLAGS,
    MONTHLY_ABSOLUTE_TOLERANCE,
    RETURN_FLOOR,
    RETURN_QUANTUM,
    combine_nonoverlapping_returns,
    delisting_imputation,
    is_performance_related_delisting,
    observed_total_is_usable,
    trading_periods,
    within_monthly_tolerance,
)
from portfolio_bot.data.phase_02_crsp_typed_backbone_core import (
    SchemaRegistry,
    load_schema_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/crsp_return_reconciliation_policy_v1.toml"
SCHEMA_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/crsp_return_reconciliation_schema_v1.json"
EMPIRICAL_LOCK_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/crsp_return_reconciliation_empirical_lock_v1.json"
PATHS_PATH = PROJECT_ROOT / "configs/paths.toml"
TYPED_POINTER_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CRSP_TYPED_BACKBONE_VERIFIED.json"
TYPED_PRIMARY_POINTER_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CRSP_TYPED_BACKBONE_PRIMARY.json"
SUPPORTING_POINTER_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CRSP_SUPPORTING_PROFILE_VERIFIED.json"
CALENDAR_POINTER_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CALENDAR_VERIFIED.json"
DISCOVERY_RESULT_PATH = PROJECT_ROOT / "work/return_reconciliation_discovery.json"
DISCOVERY_IMPLEMENTATION_PATH = PROJECT_ROOT / "work/discover_return_reconciliation.py"
CONTRACT_ID = "phase02-crsp-b03263a970e52fbd"
EPOCH = dt.date(1970, 1, 1)
SCOPE_START = dt.date(2000, 1, 1)
SCOPE_END = dt.date(2014, 12, 31)
SOURCE_RETURN_SCALE = 1_000_000


class ReturnReconciliationError(RuntimeError):
    """Raised when a hard return-reconciliation gate fails."""


@dataclass(frozen=True)
class Settings:
    database_root: Path
    output_database_root: Path
    is_production: bool
    raw_root: Path
    typed_root: Path
    calendar_root: Path
    schema: SchemaRegistry
    empirical_lock: dict[str, Any]
    build_id: str
    interim_dir: Path
    quality_dir: Path
    artifact_dir: Path
    pointer_path: Path


@dataclass(frozen=True)
class DelistingTreatment:
    event_index: int
    target_key: int
    posting_key: int | None
    economic_month_key: int
    outcome_availability_date: dt.date
    status: str
    adjustment: Decimal | None
    authoritative_mode: str


@dataclass
class MonthAggregate:
    raw_gross: Decimal = Decimal(1)
    authorized_gross: Decimal = Decimal(1)
    raw_count: int = 0
    authorized_count: int = 0
    observation_count: int = 0
    multi_period_count: int = 0
    imputed_count: int = 0
    mv_count: int = 0
    event_adjustment_count: int = 0
    unresolved_delisting_count: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _date_ordinal(value: dt.date) -> int:
    return (value - EPOCH).days


def _month_ordinal_from_date(value: dt.date) -> int:
    return (value.year - 1970) * 12 + value.month - 1


def _event_key(permno: int, day_ordinal: int) -> int:
    return int(permno) * 100_000 + int(day_ordinal)


def _month_key(permno: int, month_ordinal: int) -> int:
    return int(permno) * 10_000 + int(month_ordinal)


def _delisting_outcome_availability(
    posting_date: dt.date,
    amount_date: dt.date | None,
    next_date: dt.date | None,
) -> dt.date:
    return max(
        value for value in (posting_date, amount_date, next_date) if value is not None
    )


def _parse_date(value: object) -> dt.date | None:
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="raise")
    return parsed.date()


def _date_series(values: pd.Series) -> list[dt.date | None]:
    """Parse a whole source column without the costly scalar parser."""

    parsed = pd.to_datetime(values.mask(values.str.strip() == ""), errors="coerce", format="mixed")
    return [None if pd.isna(value) else value.date() for value in parsed]


def _parse_decimal(value: object, scale: int = 6) -> Decimal | None:
    text = str(value).strip()
    if not text:
        return None
    quantum = Decimal(1).scaleb(-scale)
    return Decimal(text).quantize(quantum, rounding=ROUND_HALF_EVEN)


def _parse_positive_int(value: object) -> int | None:
    """Return a link identifier, mapping CRSP nonpositive sentinels to null."""

    text = str(value).strip()
    if not text:
        return None
    parsed = int(text)
    if parsed <= 0:
        return None
    return parsed


def _selected_csv_fields(raw_line: bytes, wanted: set[int]) -> dict[int, str]:
    """Decode only selected fields from a physical CSV record.

    CRSP event ledgers contain no free-text fields or embedded record
    newlines.  This scanner still honors quoted commas and escaped quotes,
    while leaving every non-selected payload byte undecoded until the date
    gate passes.
    """

    output: dict[int, str] = {}
    field_index = 0
    field_start = 0
    in_quotes = False
    position = 0
    while position < len(raw_line):
        value = raw_line[position]
        if value == 34:  # double quote
            if in_quotes and position + 1 < len(raw_line) and raw_line[position + 1] == 34:
                position += 2
                continue
            in_quotes = not in_quotes
        if (value == 44 and not in_quotes) or (value in {10, 13} and not in_quotes):
            if field_index in wanted:
                encoded = raw_line[field_start:position].decode("utf-8")
                decoded = next(csv.reader([encoded]))
                if len(decoded) != 1:
                    raise ReturnReconciliationError("Selected CSV field did not decode uniquely")
                output[field_index] = decoded[0].strip()
            field_index += 1
            field_start = position + 1
            if value in {10, 13}:
                break
        position += 1
    if in_quotes:
        raise ReturnReconciliationError("Embedded CSV record newlines are not permitted")
    if output.keys() != wanted:
        raise ReturnReconciliationError("Date-first CSV scan did not find every selected field")
    return output


def _read_date_first_csv(
    path: Path,
    *,
    date_columns: tuple[str, ...],
    include: Callable[[dict[str, dt.date | None]], bool],
) -> pd.DataFrame:
    """Decode full payload only after selected dates pass ``include``."""

    rows: list[list[str]] = []
    with path.open("rb") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ReturnReconciliationError(f"Empty event ledger: {path.name}")
        header = next(csv.reader([header_line.decode("utf-8-sig").rstrip("\r\n")]))
        positions = {name: header.index(name) for name in date_columns}
        wanted = set(positions.values())
        for raw_line in handle:
            selected_raw = _selected_csv_fields(raw_line, wanted)
            selected_dates = {
                name: (
                    None
                    if not selected_raw[index]
                    else dt.date.fromisoformat(selected_raw[index])
                )
                for name, index in positions.items()
            }
            if not include(selected_dates):
                continue
            decoded = next(csv.reader([raw_line.decode("utf-8").rstrip("\r\n")]))
            if len(decoded) != len(header):
                raise ReturnReconciliationError(
                    f"Event ledger row width differs from header in {path.name}"
                )
            rows.append(decoded)
    return pd.DataFrame.from_records(rows, columns=header)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ReturnReconciliationError(f"Expected JSON object at {path}")
    return value


def _verified_manifest_output(
    manifest: dict[str, Any],
    *,
    suffix: str,
    database_root: Path,
) -> Path:
    matches = [item for item in manifest.get("outputs", []) if item["path"].endswith(suffix)]
    if len(matches) != 1:
        raise ReturnReconciliationError(f"Expected one verified manifest output ending {suffix!r}")
    item = matches[0]
    recorded = Path(item["path"])
    path = (
        database_root / recorded
        if not recorded.is_absolute()
        else resolve_recorded_database_path(recorded, current_database_root=database_root)
    )
    if not path.is_file() or path.stat().st_size != item["size_bytes"] or _sha256(path) != item["sha256"]:
        raise ReturnReconciliationError(f"Verified upstream output binding differs for {suffix}")
    return path


def _validate_bindings(lock: dict[str, Any], *, raw_root: Path) -> None:
    if lock.get("status") != "locked_after_aggregate_only_development_discovery":
        raise ReturnReconciliationError("The empirical lock is absent or not frozen")
    if not lock.get("authorizations", {}).get("production_return_reconciliation_build_authorized"):
        raise ReturnReconciliationError("The empirical lock does not authorize a production build")
    bindings = lock["bindings"]
    local = {
        "policy_sha256": POLICY_PATH,
        "schema_sha256": SCHEMA_PATH,
        "verified_typed_backbone_pointer_sha256": TYPED_POINTER_PATH,
        "verified_supporting_profile_pointer_sha256": SUPPORTING_POINTER_PATH,
        "verified_calendar_pointer_sha256": CALENDAR_POINTER_PATH,
        "raw_distribution_sha256": raw_root / "crsp_distribution_ciz.csv",
        "raw_delisting_sha256": raw_root / "crsp_delisting_ciz.csv",
        "aggregate_discovery_result_sha256": DISCOVERY_RESULT_PATH,
        "aggregate_discovery_implementation_sha256": DISCOVERY_IMPLEMENTATION_PATH,
    }
    for name, path in local.items():
        actual = _sha256(path)
        if bindings.get(name) != actual:
            raise ReturnReconciliationError(
                f"Empirical lock binding mismatch for {name}: {actual}"
            )


def load_settings(*, output_database_root: Path | None = None) -> Settings:
    with PATHS_PATH.open("rb") as handle:
        paths = tomllib.load(handle)
    database_root = Path(os.environ.get("PORTFOLIO_DATABASE_ROOT", paths["database"]["root"]))
    output_root = Path(output_database_root) if output_database_root else database_root
    is_production = output_root.resolve() == database_root.resolve()
    typed_pointer = _load_json(TYPED_POINTER_PATH)
    if typed_pointer.get("status") != "verified_passed":
        raise ReturnReconciliationError("Typed backbone is not independently verified")
    primary_pointer = _load_json(TYPED_PRIMARY_POINTER_PATH)
    if (
        primary_pointer.get("build_id") != typed_pointer.get("build_id")
        or primary_pointer.get("manifest_sha256")
        != typed_pointer.get("primary_manifest_sha256")
    ):
        raise ReturnReconciliationError(
            "Typed primary pointer does not match the independently verified build"
        )
    typed_manifest_path = PROJECT_ROOT / primary_pointer["artifact_path"] / "manifest.json"
    if _sha256(typed_manifest_path) != primary_pointer.get("manifest_sha256"):
        raise ReturnReconciliationError("Typed primary manifest differs from its verified hash")
    typed_manifest = _load_json(typed_manifest_path)
    daily_input = _verified_manifest_output(
        typed_manifest,
        suffix="/crsp_security_day_return_staging.parquet",
        database_root=database_root,
    )
    monthly_input = _verified_manifest_output(
        typed_manifest,
        suffix="/crsp_security_month_return_staging.parquet",
        database_root=database_root,
    )
    if daily_input.parent != monthly_input.parent:
        raise ReturnReconciliationError("Verified typed return inputs do not share one build root")
    typed_root = daily_input.parent

    calendar_pointer = _load_json(CALENDAR_POINTER_PATH)
    if calendar_pointer.get("status") != "verified_passed":
        raise ReturnReconciliationError("Trading calendar is not independently verified")
    calendar_manifest = _load_json(PROJECT_ROOT / calendar_pointer["primary_artifact_path"] / "manifest.json")
    if _sha256(PROJECT_ROOT / calendar_pointer["primary_artifact_path"] / "manifest.json") != calendar_pointer.get(
        "primary_manifest_sha256"
    ):
        raise ReturnReconciliationError("Verified calendar manifest binding differs")
    calendar_path = _verified_manifest_output(
        calendar_manifest,
        suffix="/trading_calendar.parquet",
        database_root=database_root,
    )
    calendar_root = calendar_path.parent

    lock = _load_json(EMPIRICAL_LOCK_PATH)
    raw_root = database_root / paths["database"]["raw"]
    _validate_bindings(lock, raw_root=raw_root)
    schema = load_schema_registry(SCHEMA_PATH)
    build_binding = {
        "calendar_build_id": calendar_pointer["build_id"],
        "empirical_lock_sha256": _sha256(EMPIRICAL_LOCK_PATH),
        "implementation_sha256": _sha256(Path(__file__)),
        "policy_sha256": _sha256(POLICY_PATH),
        "publication_mode": "production" if is_production else "rehearsal",
        "schema_sha256": _sha256(SCHEMA_PATH),
        "typed_build_id": typed_pointer["build_id"],
    }
    build_prefix = "crsp-return-" if is_production else "crsp-return-rehearsal-"
    build_id = build_prefix + hashlib.sha256(_canonical_json(build_binding)).hexdigest()[:24]
    base = Path("phase_02_canonical_panel") / CONTRACT_ID / "crsp_return_reconciliation" / build_id
    artifact_dir = (
        PROJECT_ROOT / "artifacts/phase_02_canonical_panel/crsp_return_reconciliations" / build_id
        if is_production
        else output_root / "rehearsal_artifacts/phase_02_canonical_panel" / build_id
    )
    return Settings(
        database_root=database_root,
        output_database_root=output_root,
        is_production=is_production,
        raw_root=raw_root,
        typed_root=typed_root,
        calendar_root=calendar_root,
        schema=schema,
        empirical_lock=lock,
        build_id=build_id,
        interim_dir=output_root / paths["database"]["interim"] / base,
        quality_dir=output_root / paths["database"]["quality"] / base,
        artifact_dir=artifact_dir,
        pointer_path=PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CRSP_RETURN_RECONCILIATION_PRIMARY.json",
    )


def _read_events(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    distributions = _read_date_first_csv(
        settings.raw_root / "crsp_distribution_ciz.csv",
        date_columns=("DisExDt",),
        include=lambda dates: (
            dates["DisExDt"] is not None
            and SCOPE_START <= dates["DisExDt"] <= SCOPE_END
        ),
    )
    distributions["ex_date"] = _date_series(distributions["DisExDt"])
    distributions["permno"] = distributions["PERMNO"].map(lambda value: int(value))
    distributions.sort_values(["permno", "ex_date", "DisSeqNbr"], inplace=True, kind="mergesort")
    distributions.reset_index(drop=True, inplace=True)

    delisting_context = _read_date_first_csv(
        settings.raw_root / "crsp_delisting_ciz.csv",
        date_columns=("DelistingDt", "DelDlyDt"),
        include=lambda dates: (
            dates["DelistingDt"] is not None
            and (
                SCOPE_START <= dates["DelistingDt"] <= SCOPE_END
                or (
                    dates["DelistingDt"] < SCOPE_START
                    and dates["DelDlyDt"] is not None
                    and SCOPE_START <= dates["DelDlyDt"] <= SCOPE_END
                )
            )
        ),
    )
    delisting_context["delisting_date"] = _date_series(delisting_context["DelistingDt"])
    delisting_context["posting_date"] = _date_series(delisting_context["DelDlyDt"])
    delisting_context["amount_date"] = _date_series(delisting_context["DelAmtDt"])
    delisting_context["next_date"] = _date_series(delisting_context["DelNextDt"])
    delisting_context["permno"] = delisting_context["PERMNO"].map(lambda value: int(value))
    delisting_context["delret"] = delisting_context["DelRet"].map(_parse_decimal)
    delisting_context.sort_values(
        ["permno", "delisting_date"], inplace=True, kind="mergesort"
    )
    delisting_context.reset_index(drop=True, inplace=True)
    delistings = delisting_context.loc[
        delisting_context["delisting_date"].map(
            lambda value: SCOPE_START <= value <= SCOPE_END
        )
    ].copy()
    delistings["permno"] = delistings["PERMNO"].map(lambda value: int(value))
    delistings["delret"] = delistings["DelRet"].map(_parse_decimal)
    delistings.sort_values(["permno", "delisting_date"], inplace=True, kind="mergesort")
    delistings.reset_index(drop=True, inplace=True)
    return distributions, delistings, delisting_context


def _load_daily_arrays(settings: Settings) -> dict[str, np.ndarray]:
    table = pq.read_table(
        settings.typed_root / "crsp_security_day_return_staging.parquet",
        columns=[
            "permno",
            "calendar_date",
            "raw_total_return",
            "raw_return_ex_distributions",
            "raw_return_missing_flag",
            "raw_return_duration_flag",
        ],
    )
    def numpy(name: str, cast: pa.DataType | None = None) -> np.ndarray:
        array = table[name].combine_chunks()
        if cast is not None:
            array = pc.cast(array, cast)
        return array.to_numpy(zero_copy_only=False)

    def scaled_decimal(name: str) -> tuple[np.ndarray, np.ndarray]:
        array = table[name].combine_chunks()
        valid = pc.is_valid(array).to_numpy(zero_copy_only=False).astype(bool)
        narrowed = pc.cast(array, pa.decimal128(20, 6))
        multiplier = pa.scalar(
            Decimal(SOURCE_RETURN_SCALE), type=pa.decimal128(7, 0)
        )
        scaled = pc.cast(pc.multiply(narrowed, multiplier), pa.int64())
        values = pc.fill_null(scaled, 0).to_numpy(zero_copy_only=False).astype(np.int64)
        return values, valid

    total_scaled, total_valid = scaled_decimal("raw_total_return")
    retx_scaled, retx_valid = scaled_decimal("raw_return_ex_distributions")

    return {
        "permno": numpy("permno").astype(np.int64),
        "date": numpy("calendar_date").astype("datetime64[D]"),
        "total_scaled": total_scaled,
        "total_valid": total_valid,
        "retx_scaled": retx_scaled,
        "retx_valid": retx_valid,
        "missing": numpy("raw_return_missing_flag").astype(str),
        "duration": numpy("raw_return_duration_flag").astype(str),
    }


def _calendar_ordinals(settings: Settings) -> np.ndarray:
    table = pq.read_table(settings.calendar_root / "trading_calendar.parquet")
    date_column = next(name for name in table.column_names if "date" in name.lower())
    dates = table[date_column].combine_chunks().to_numpy(zero_copy_only=False).astype("datetime64[D]")
    ordinals = dates.astype(np.int64)
    scoped = ordinals[
        (ordinals >= _date_ordinal(SCOPE_START))
        & (ordinals <= _date_ordinal(SCOPE_END))
    ]
    if len(scoped) == 0 or np.any(scoped[1:] <= scoped[:-1]):
        raise ReturnReconciliationError("Verified calendar is empty or not strictly increasing")
    return scoped


def _daily_structure(
    arrays: dict[str, np.ndarray],
    calendar: np.ndarray,
    delistings: pd.DataFrame,
) -> dict[str, Any]:
    permno = arrays["permno"]
    day = arrays["date"].astype(np.int64)
    keys = permno * np.int64(100_000) + day
    if np.any(keys[1:] <= keys[:-1]):
        raise ReturnReconciliationError("Daily return staging is not strictly key-sorted")
    calendar_position = np.searchsorted(calendar, day)
    if np.any(calendar_position >= len(calendar)) or np.any(calendar[calendar_position] != day):
        raise ReturnReconciliationError("A daily row date is absent from the verified calendar")

    event_index_by_posting_key: dict[int, int] = {}
    event_index_by_delisting_key: dict[int, int] = {}
    for index, row in delistings.iterrows():
        posting = row["posting_date"]
        if posting is not None:
            post_key = _event_key(row["permno"], _date_ordinal(posting))
            if post_key in event_index_by_posting_key:
                raise ReturnReconciliationError("Duplicate delisting posting key")
            event_index_by_posting_key[post_key] = int(index)
        del_key = _event_key(row["permno"], _date_ordinal(row["delisting_date"]))
        if del_key in event_index_by_delisting_key:
            raise ReturnReconciliationError("Duplicate delisting economic key")
        event_index_by_delisting_key[del_key] = int(index)

    is_delisting_leg = np.zeros(len(keys), dtype=bool)
    posting_event_index = np.full(len(keys), -1, dtype=np.int32)
    for post_key, event_index in event_index_by_posting_key.items():
        position = int(np.searchsorted(keys, post_key))
        if position < len(keys) and int(keys[position]) == post_key:
            is_delisting_leg[position] = True
            posting_event_index[position] = event_index

    periods = np.zeros(len(keys), dtype=np.int16)
    span_start = np.full(len(keys), -2_147_483_648, dtype=np.int32)
    for code in sorted(set(arrays["duration"].tolist())):
        mask = arrays["duration"] == code
        mapped = trading_periods(code)
        if mapped is None:
            periods[mask] = -1
        else:
            periods[mask] = mapped
            valid = mask & (mapped > 0) & (calendar_position >= mapped)
            span_start[valid] = calendar[calendar_position[valid] - mapped]
            if code in {"D1", "D2", "D3", "D4"}:
                boundary = mask & (mapped > 0) & (calendar_position < mapped)
                span_start[boundary] = day[boundary] - int(code[1:])
    event_rows = np.flatnonzero(is_delisting_leg)
    periods[event_rows] = -1
    for position in event_rows:
        event = delistings.iloc[int(posting_event_index[position])]
        span_start[position] = _date_ordinal(event["delisting_date"])

    unique_permno, starts, counts = np.unique(permno, return_index=True, return_counts=True)
    bounds = {int(key): (int(start), int(start + count)) for key, start, count in zip(unique_permno, starts, counts, strict=True)}
    return {
        "keys": keys,
        "calendar_position": calendar_position,
        "is_delisting_leg": is_delisting_leg,
        "posting_event_index": posting_event_index,
        "periods": periods,
        "span_start": span_start,
        "bounds": bounds,
        "event_index_by_posting_key": event_index_by_posting_key,
        "event_index_by_delisting_key": event_index_by_delisting_key,
    }


def _owns_distribution_interval(
    *,
    row_date: int,
    effective_ex_date: int,
    span_start: int,
    mapped_periods: int,
    is_delisting_leg: bool,
) -> bool:
    """Return whether one row proves ownership of a distribution event.

    A generic same-day row is not sufficient evidence: the frozen policy owns
    distributions through a proven ``(span_start, row_date]`` interval.  The
    sole same-day exception is an explicitly linked terminal delisting leg,
    whose economic/event mapping is established independently.
    """

    span_is_known = span_start > -1_000_000_000
    if is_delisting_leg:
        return row_date == effective_ex_date or (
            span_is_known and span_start < effective_ex_date <= row_date
        )
    return (
        mapped_periods > 0
        and span_is_known
        and span_start < effective_ex_date <= row_date
    )


def _ordinary_multi_span_status(*, span_known: bool) -> str:
    """Label a multi-period row without overstating scope-boundary evidence."""

    return (
        "verified_multi_session_span"
        if span_known
        else "pre_scope_multi_session_span_unresolved"
    )


def _distribution_mapping(
    distributions: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    structure: dict[str, Any],
    calendar: np.ndarray,
) -> tuple[pd.DataFrame, dict[int, int], dict[str, int]]:
    result = distributions.copy()
    counts_by_daily_key: Counter[int] = Counter()
    group_status: Counter[str] = Counter()
    total_scaled = arrays["total_scaled"]
    total_valid = arrays["total_valid"]
    retx_scaled = arrays["retx_scaled"]
    retx_valid = arrays["retx_valid"]
    dates = arrays["date"].astype(np.int64)
    keys = structure["keys"]
    periods = structure["periods"]
    starts = structure["span_start"]
    is_delisting_leg = structure["is_delisting_leg"]

    ex_ordinals = np.fromiter(
        (_date_ordinal(value) for value in result["ex_date"]),
        dtype=np.int64,
        count=len(result),
    )
    calendar_positions = np.searchsorted(calendar, ex_ordinals, side="left")
    effective = ex_ordinals.copy()
    inside = calendar_positions < len(calendar)
    effective[inside] = calendar[calendar_positions[inside]]
    off_calendar = effective != ex_ordinals
    event_permno = result["permno"].to_numpy(np.int64)
    group_starts = np.r_[
        0,
        np.flatnonzero(
            (event_permno[1:] != event_permno[:-1])
            | (effective[1:] != effective[:-1])
        )
        + 1,
    ]
    group_ends = np.r_[group_starts[1:], len(result)]
    mapped_start_ord = np.full(len(result), -2_147_483_648, dtype=np.int32)
    mapped_date_ord = np.full(len(result), -2_147_483_648, dtype=np.int32)
    statuses = np.empty(len(result), dtype=object)
    assigned = np.zeros(len(result), dtype=bool)

    for group_start, group_end in zip(group_starts, group_ends, strict=True):
        permno = int(event_permno[group_start])
        effective_ordinal = int(effective[group_start])
        bound = structure["bounds"].get(int(permno))
        candidates: list[int] = []
        if bound is not None:
            left, right = bound
            local_dates = dates[left:right]
            first = int(np.searchsorted(local_dates, int(effective_ordinal), side="left")) + left
            for position in range(first, min(right, first + 12)):
                if (
                    _owns_distribution_interval(
                        row_date=int(dates[position]),
                        effective_ex_date=int(effective_ordinal),
                        span_start=int(starts[position]),
                        mapped_periods=int(periods[position]),
                        is_delisting_leg=bool(is_delisting_leg[position]),
                    )
                    and total_valid[position]
                    and retx_valid[position]
                ):
                    candidates.append(position)
        if len(candidates) > 1:
            same_day = [position for position in candidates if dates[position] == int(effective_ordinal)]
            if len(same_day) == 1:
                candidates = same_day
            else:
                changed = [
                    position
                    for position in candidates
                    if total_scaled[position] != retx_scaled[position]
                ]
                if len(changed) != 1:
                    raise ReturnReconciliationError(
                        "Distribution interval ownership remains ambiguous after event-aware spans"
                    )
                candidates = changed
        off = bool(np.any(off_calendar[group_start:group_end]))
        prefix = "off_calendar_" if off else ""
        if len(candidates) == 1:
            position = candidates[0]
            status = prefix + "assigned_unique_containing_return_span"
            mapped_start_ord[group_start:group_end] = int(starts[position])
            mapped_date_ord[group_start:group_end] = int(dates[position])
            counts_by_daily_key[int(keys[position])] += int(group_end - group_start)
            assigned[group_start:group_end] = True
        else:
            bound = structure["bounds"].get(int(permno))
            if bound is None:
                status = "structural_no_scoped_security_day"
            elif int(effective_ordinal) > int(dates[bound[1] - 1]):
                status = "structural_after_last_security_day"
            else:
                left, right = bound
                local_dates = dates[left:right]
                position = int(np.searchsorted(local_dates, int(effective_ordinal))) + left
                same_day = position < right and int(dates[position]) == int(effective_ordinal)
                status = (
                    "structural_missing_return_internal_gap"
                    if same_day or int(effective_ordinal) <= int(dates[right - 1])
                    else "structural_after_last_security_day"
                )
        statuses[group_start:group_end] = status
        group_status[status] += 1

    expected = {"zero": 509, "one": 258_595}
    assigned_groups = sum(value for key, value in group_status.items() if "assigned_unique" in key)
    unassigned_groups = sum(value for key, value in group_status.items() if key.startswith("structural_"))
    if {"zero": unassigned_groups, "one": assigned_groups} != expected:
        raise ReturnReconciliationError(
            "Distribution mapping differs from lock: "
            f"zero={unassigned_groups}, one={assigned_groups}, statuses={dict(group_status)}"
        )
    structural = {
        "structural_after_last_security_day": 412,
        "structural_missing_return_internal_gap": 59,
        "structural_no_scoped_security_day": 38,
    }
    actual_structural = {key: group_status.get(key, 0) for key in structural}
    if actual_structural != structural:
        raise ReturnReconciliationError(
            f"Distribution structural classes differ from lock: {actual_structural}"
        )
    result["effective_ordinal"] = effective
    result["off_calendar"] = off_calendar
    result["mapped_start"] = [
        None if value < -1_000_000_000 else EPOCH + dt.timedelta(days=int(value))
        for value in mapped_start_ord
    ]
    result["mapped_date"] = [
        None if value < -1_000_000_000 else EPOCH + dt.timedelta(days=int(value))
        for value in mapped_date_ord
    ]
    result["mapping_status"] = statuses.tolist()
    result["assigned_exactly_once"] = assigned
    return result, dict(counts_by_daily_key), dict(group_status)


def _delisting_treatments(
    delistings: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    structure: dict[str, Any],
) -> tuple[pd.DataFrame, dict[int, DelistingTreatment], dict[int, int]]:
    result = delistings.copy()
    keys = structure["keys"]
    total_scaled = arrays["total_scaled"]
    total_valid = arrays["total_valid"]
    missing = arrays["missing"]
    treatments: dict[int, DelistingTreatment] = {}
    counts_by_target: Counter[int] = Counter()
    output_columns = {
        "performance_related": [],
        "embedding_status": [],
        "treatment_status": [],
        "effective_return": [],
        "imputation_applied": [],
        "assigned_date": [],
        "assigned_exactly_once": [],
    }
    for event_index, row in result.iterrows():
        performance = is_performance_related_delisting(
            action_type=row["DelActionType"],
            payment_type=row["DelPaymentType"],
            status_type=row["DelStatusType"],
            reason_type=row["DelReasonType"],
        )
        posting = row["posting_date"]
        if posting is None:
            raise ReturnReconciliationError("Every scoped delisting must have DelDlyDt")
        posting_key = _event_key(row["permno"], _date_ordinal(posting))
        posting_position = int(np.searchsorted(keys, posting_key))
        posting_match = posting_position < len(keys) and int(keys[posting_position]) == posting_key
        target_key = _event_key(row["permno"], _date_ordinal(row["delisting_date"]))
        target_position = int(np.searchsorted(keys, target_key))
        if target_position >= len(keys) or int(keys[target_position]) != target_key:
            raise ReturnReconciliationError("Every delisting must have a DelistingDt daily row")
        if missing[target_position] != "NA" or not total_valid[target_position]:
            raise ReturnReconciliationError("DelistingDt ordinary return must be numeric NA")
        observed = row["delret"]
        adjustment: Decimal | None = None
        effective_return: Decimal | None = None
        imputed = False
        if observed is not None:
            if posting_match:
                daily_flag = missing[posting_position]
                observed_scaled = int(observed * SOURCE_RETURN_SCALE)
                if (
                    not total_valid[posting_position]
                    or observed_scaled != int(total_scaled[posting_position])
                ):
                    raise ReturnReconciliationError(
                        "Observed DelRet is not exactly embedded in posting-day DlyRet"
                    )
                if daily_flag not in {"NA", "MV"}:
                    raise ReturnReconciliationError(
                        f"Unexpected numeric observed delisting flag {daily_flag!r}"
                    )
                embedding = "embedded_in_posting_daily_total_used_as_identity_evidence"
            else:
                embedding = "posting_daily_row_absent_observed_event_return_available"
            adjustment = observed.quantize(RETURN_QUANTUM)
            effective_return = adjustment
            treatment_status = "observed_event_return_attached_to_delisting_date_once"
            mode = "compound"
        else:
            embedding = "posting_daily_total_missing"
            if not posting_match:
                raise ReturnReconciliationError("Missing DelRet also lacks its posting audit row")
            if total_valid[posting_position]:
                raise ReturnReconciliationError("Missing DelRet unexpectedly has numeric posting return")
            if performance:
                adjustment = delisting_imputation(row["PrimaryExch"])
                effective_return = adjustment
                treatment_status = "missing_performance_imputed_on_delisting_date_once"
                imputed = True
                mode = "compound"
            else:
                treatment_status = "missing_nonperformance_terminal_leg_unresolved"
                mode = "unresolved"

        economic_month = _month_key(row["permno"], _month_ordinal_from_date(row["delisting_date"]))
        outcome_availability_date = _delisting_outcome_availability(
            posting, row["amount_date"], row["next_date"]
        )
        treatment = DelistingTreatment(
            event_index=int(event_index),
            target_key=int(target_key),
            posting_key=int(posting_key),
            economic_month_key=economic_month,
            outcome_availability_date=outcome_availability_date,
            status=treatment_status,
            adjustment=adjustment,
            authoritative_mode=mode,
        )
        if treatment.target_key in treatments:
            raise ReturnReconciliationError("Multiple delisting treatments target one daily key")
        treatments[treatment.target_key] = treatment
        counts_by_target[treatment.target_key] += 1
        output_columns["performance_related"].append(performance)
        output_columns["embedding_status"].append(embedding)
        output_columns["treatment_status"].append(treatment_status)
        output_columns["effective_return"].append(effective_return)
        output_columns["imputation_applied"].append(imputed)
        output_columns["assigned_date"].append(row["delisting_date"])
        output_columns.setdefault("outcome_availability_date", []).append(
            outcome_availability_date
        )
        output_columns["assigned_exactly_once"].append(True)

    for name, values in output_columns.items():
        result[name] = values
    if int(result["imputation_applied"].sum()) != 374:
        raise ReturnReconciliationError("Missing-performance imputation count differs from empirical lock")
    if int((result["treatment_status"] == "observed_event_return_attached_to_delisting_date_once").sum()) != 7_847:
        raise ReturnReconciliationError("Observed delisting attachment count differs from empirical lock")
    if int((result["embedding_status"] == "posting_daily_row_absent_observed_event_return_available").sum()) != 8:
        raise ReturnReconciliationError("Absent posting-row count differs from empirical lock")
    if int((result["treatment_status"] == "missing_nonperformance_terminal_leg_unresolved").sum()) != 170:
        raise ReturnReconciliationError("Nonperformance unresolved count differs from empirical lock")
    return result, treatments, dict(counts_by_target)


def _event_table(
    frame: pd.DataFrame,
    table_name: str,
    schema: pa.Schema,
) -> pa.Table:
    if table_name == "crsp_distribution_event":
        payload = {
            "permno": frame["permno"].tolist(),
            "ex_date": frame["ex_date"].tolist(),
            "sequence_number": [int(value) for value in frame["DisSeqNbr"]],
            "ordinary_flag": frame["DisOrdinaryFlg"].tolist(),
            "distribution_type": frame["DisType"].tolist(),
            "frequency_type": frame["DisFreqType"].tolist(),
            "payment_type": frame["DisPaymentType"].tolist(),
            "detail_type": frame["DisDetailType"].tolist(),
            "tax_type": frame["DisTaxType"].tolist(),
            "original_currency_type": frame["DisOrigCurType"].tolist(),
            "dividend_amount": [_parse_decimal(value) for value in frame["DisDivAmt"]],
            "price_adjustment_factor": [_parse_decimal(value) for value in frame["DisFacPr"]],
            "share_adjustment_factor": [_parse_decimal(value) for value in frame["DisFacShr"]],
            "declare_date": _date_series(frame["DisDeclareDt"]),
            "record_date": _date_series(frame["DisRecordDt"]),
            "pay_date": _date_series(frame["DisPayDt"]),
            "source_distribution_permno": [_parse_positive_int(value) for value in frame["DisPERMNO"]],
            "source_distribution_permco": [_parse_positive_int(value) for value in frame["DisPERMCO"]],
            "amount_source_type": frame["DisAmountSourceType"].tolist(),
            "primary_exchange": frame["PrimaryExch"].tolist(),
            "mapped_return_span_start_date": frame["mapped_start"].tolist(),
            "mapped_return_date": frame["mapped_date"].tolist(),
            "mapping_status": frame["mapping_status"].tolist(),
            "assigned_exactly_once": frame["assigned_exactly_once"].tolist(),
        }
    else:
        payload = {
            "permno": frame["permno"].tolist(),
            "delisting_date": frame["delisting_date"].tolist(),
            "posting_date": frame["posting_date"].tolist(),
            "delisting_price": [_parse_decimal(value) for value in frame["DelDtPrc"]],
            "delisting_price_flag": frame["DelDtPrcFlg"].tolist(),
            "action_type": frame["DelActionType"].tolist(),
            "status_type": frame["DelStatusType"].tolist(),
            "reason_type": frame["DelReasonType"].tolist(),
            "payment_type": frame["DelPaymentType"].tolist(),
            "successor_permno": [_parse_positive_int(value) for value in frame["DelPERMNO"]],
            "successor_permco": [_parse_positive_int(value) for value in frame["DelPERMCO"]],
            "observed_delisting_return": frame["delret"].tolist(),
            "return_missing_type": frame["DelRetMissType"].tolist(),
            "next_date": _date_series(frame["DelNextDt"]),
            "next_price": [_parse_decimal(value) for value in frame["DelNextPrc"]],
            "next_price_flag": frame["DelNextPrcFlg"].tolist(),
            "amount_date": _date_series(frame["DelAmtDt"]),
            "delisting_dividend_amount": [_parse_decimal(value) for value in frame["DelDivAmt"]],
            "distribution_type": frame["DelDisType"].tolist(),
            "primary_exchange": frame["PrimaryExch"].tolist(),
            "performance_related": frame["performance_related"].tolist(),
            "embedding_status": frame["embedding_status"].tolist(),
            "treatment_status": frame["treatment_status"].tolist(),
            "effective_delisting_return": frame["effective_return"].tolist(),
            "imputation_applied": frame["imputation_applied"].tolist(),
            "assigned_return_date": frame["assigned_date"].tolist(),
            "outcome_availability_date": frame["outcome_availability_date"].tolist(),
            "assigned_exactly_once": frame["assigned_exactly_once"].tolist(),
        }
    return pa.Table.from_pydict(payload, schema=schema)


def _monthly_flag_class(flag: str) -> str:
    mapping = {
        "CR": "complete_numeric_transport",
        "DE": "delisting_conditioned",
        "GP": "gap_diagnostic",
        "IP": "incomplete_diagnostic",
        "MP": "missing_price_diagnostic",
        "NS": "new_security_partial_start",
        "NT": "not_tracked_missing",
    }
    if flag not in KNOWN_MONTHLY_FLAGS:
        raise ReturnReconciliationError(f"Unknown monthly flag {flag!r}")
    return mapping[flag]


def _quantized_return(gross: Decimal, count: int) -> Decimal | None:
    if count == 0:
        return None
    return max(
        RETURN_FLOOR,
        (gross - Decimal(1)).quantize(RETURN_QUANTUM, rounding=ROUND_HALF_EVEN),
    )


def _update_month(
    aggregate: MonthAggregate,
    *,
    raw: Decimal | None,
    authorized: Decimal | None,
    return_leg_class: str,
    state: str,
    count_observation: bool = True,
) -> None:
    if count_observation:
        aggregate.observation_count += 1
    if raw is not None and raw >= RETURN_FLOOR:
        aggregate.raw_gross *= Decimal(1) + raw
        aggregate.raw_count += 1
    if authorized is not None:
        aggregate.authorized_gross *= Decimal(1) + authorized
        aggregate.authorized_count += 1
    if return_leg_class == "ordinary_multi_session":
        aggregate.multi_period_count += 1
    if state == "imputed_delisting":
        aggregate.imputed_count += 1
    if state == "quarantined_MV":
        aggregate.mv_count += 1
    if state == "unresolved_nonperformance_delisting":
        aggregate.unresolved_delisting_count += 1
    if return_leg_class == "ordinary_plus_terminal_delisting_adjustment":
        aggregate.event_adjustment_count += 1


def _build_daily_and_month_aggregates(
    settings: Settings,
    arrays: dict[str, np.ndarray],
    structure: dict[str, Any],
    distribution_counts: dict[int, int],
    treatments: dict[int, DelistingTreatment],
    delisting_counts: dict[int, int],
    destination: Path,
    *,
    row_group_limit: int | None = None,
) -> tuple[dict[int, MonthAggregate], dict[str, int]]:
    input_file = pq.ParquetFile(settings.typed_root / "crsp_security_day_return_staging.parquet")
    schema = settings.schema.tables["crsp_security_day_return"].arrow_schema()
    writer = pq.ParquetWriter(destination, schema, compression="zstd", use_dictionary=True)
    month_aggregates: dict[int, MonthAggregate] = {}
    states: Counter[str] = Counter()
    offset = 0
    unexplained_total_retx = 0
    posting_treatments = {
        treatment.posting_key: treatment
        for treatment in treatments.values()
        if treatment.posting_key is not None
    }
    try:
        group_count = (
            input_file.num_row_groups
            if row_group_limit is None
            else min(input_file.num_row_groups, row_group_limit)
        )
        for group_index in range(group_count):
            source = input_file.read_row_group(group_index)
            rows = source.num_rows
            sl = slice(offset, offset + rows)
            permno = arrays["permno"][sl]
            day_ord = arrays["date"][sl].astype(np.int64)
            keys = structure["keys"][sl]
            raw_total = source["raw_total_return"].combine_chunks().to_pylist()
            raw_retx = source["raw_return_ex_distributions"].combine_chunks().to_pylist()
            missing_flags = arrays["missing"][sl]
            duration_flags = arrays["duration"][sl]
            periods = structure["periods"][sl]
            span_start = structure["span_start"][sl]
            is_event_leg = structure["is_delisting_leg"][sl]

            mapped_periods: list[int | None] = []
            leg_classes: list[str] = []
            start_dates: list[dt.date | None] = []
            covered_sessions: list[int | None] = []
            span_statuses: list[str] = []
            dist_counts: list[int] = []
            del_counts: list[int] = []
            del_statuses: list[str] = []
            adjustments: list[Decimal | None] = []
            authoritative: list[Decimal | None] = []
            authority_states: list[str] = []
            reconciliation: list[str] = []
            usable: list[bool] = []
            outcome_availability_dates: list[dt.date] = []

            for local in range(rows):
                key = int(keys[local])
                raw = raw_total[local]
                retx = raw_retx[local]
                treatment = treatments.get(key)
                posting_treatment = posting_treatments.get(key)
                posting_leg = bool(is_event_leg[local])
                dist_count = int(distribution_counts.get(key, 0))
                del_count = int(delisting_counts.get(key, 0))
                adjustment = None if treatment is None else treatment.adjustment
                policy_usable = observed_total_is_usable(
                    raw, missing_flags[local], duration_flags[local]
                )
                span_known = span_start[local] > -1_000_000_000
                raw_usable = policy_usable and span_known
                value: Decimal | None = raw.quantize(RETURN_QUANTUM) if raw_usable else None
                state = (
                    "observed"
                    if value is not None
                    else (
                        "quarantined_unresolved_span"
                        if policy_usable and not span_known
                        else (
                            "quarantined_MV"
                            if missing_flags[local] == "MV" and raw is not None
                            else "missing"
                        )
                    )
                )
                if posting_leg:
                    value = None
                    state = (
                        "suppressed_delisting_posting_payload"
                        if posting_treatment is not None
                        else "pre_scope_delisting_posting_context_suppressed"
                    )
                elif treatment is not None:
                    if treatment.authoritative_mode == "compound":
                        if value is None or adjustment is None:
                            raise ReturnReconciliationError("Delisting event-day return cannot be compounded")
                        value = combine_nonoverlapping_returns(value, adjustment)
                        state = (
                            "imputed_delisting"
                            if treatment.status.startswith("missing_performance_imputed")
                            else "observed_plus_delisting_adjustment"
                        )
                    elif treatment.authoritative_mode == "replace":
                        if adjustment is None:
                            raise ReturnReconciliationError(
                                "Observed replacement delisting return is absent"
                            )
                        value = adjustment
                        state = "observed_delisting_adjustment_only"
                    elif treatment.authoritative_mode == "unresolved":
                        value = None
                        state = "unresolved_nonperformance_delisting"

                if posting_leg:
                    leg_class = (
                        "terminal_delisting_posting_audit_suppressed"
                        if posting_treatment is not None
                        else "pre_scope_terminal_delisting_context_suppressed"
                    )
                    mapped = None
                    start = EPOCH + dt.timedelta(days=int(span_start[local]))
                    covered = 0
                    span_status = (
                        "event_anchored_posting_leg_suppressed_no_factor_window"
                        if posting_treatment is not None
                        else "pre_scope_event_context_suppressed_no_factor_window"
                    )
                    if posting_treatment is not None:
                        effective_month_key = posting_treatment.economic_month_key
                        outcome_availability_date = posting_treatment.outcome_availability_date
                    else:
                        effective_month_key = _month_key(
                            int(permno[local]),
                            int(
                                arrays["date"][offset + local]
                                .astype("datetime64[M]")
                                .astype(int)
                            ),
                        )
                        outcome_availability_date = EPOCH + dt.timedelta(
                            days=int(day_ord[local])
                        )
                elif treatment is not None:
                    leg_class = (
                        "ordinary_plus_terminal_delisting_adjustment"
                        if treatment.adjustment is not None
                        else "ordinary_with_unresolved_terminal_delisting"
                    )
                    mapped = int(periods[local])
                    start = None if span_start[local] < -1_000_000_000 else EPOCH + dt.timedelta(days=int(span_start[local]))
                    covered = mapped
                    span_status = (
                        "ordinary_span_plus_separate_terminal_adjustment"
                        if treatment.adjustment is not None
                        else "ordinary_span_terminal_outcome_unresolved"
                    )
                    effective_month_key = treatment.economic_month_key
                    outcome_availability_date = treatment.outcome_availability_date
                elif periods[local] > 1:
                    leg_class = "ordinary_multi_session"
                    mapped = int(periods[local])
                    start = None if span_start[local] < -1_000_000_000 else EPOCH + dt.timedelta(days=int(span_start[local]))
                    covered = mapped
                    span_status = _ordinary_multi_span_status(span_known=span_known)
                    effective_month_key = _month_key(int(permno[local]), int(arrays["date"][offset + local].astype("datetime64[M]").astype(int)))
                    outcome_availability_date = EPOCH + dt.timedelta(days=int(day_ord[local]))
                elif periods[local] == 1:
                    leg_class = "ordinary_one_session"
                    mapped = 1
                    start = None if span_start[local] < -1_000_000_000 else EPOCH + dt.timedelta(days=int(span_start[local]))
                    covered = 1
                    span_status = "verified_one_session_span"
                    effective_month_key = _month_key(int(permno[local]), int(arrays["date"][offset + local].astype("datetime64[M]").astype(int)))
                    outcome_availability_date = EPOCH + dt.timedelta(days=int(day_ord[local]))
                else:
                    leg_class = "missing_return"
                    mapped = 0
                    start = None
                    covered = 0
                    span_status = "no_return_span"
                    effective_month_key = _month_key(int(permno[local]), int(arrays["date"][offset + local].astype("datetime64[M]").astype(int)))
                    outcome_availability_date = EPOCH + dt.timedelta(days=int(day_ord[local]))

                differs = raw is not None and retx is not None and raw != retx
                if differs:
                    if dist_count:
                        rec = "total_minus_retx_explained_by_distribution"
                    elif posting_leg:
                        rec = "total_minus_retx_explained_by_delisting"
                    else:
                        rec = "unexplained_total_minus_retx"
                        unexplained_total_retx += 1
                elif dist_count:
                    rec = "mapped_distribution_without_total_retx_difference"
                elif treatment is not None or posting_leg:
                    rec = "delisting_treatment_accounted_once"
                else:
                    rec = "no_event_return_difference"

                mapped_periods.append(mapped)
                leg_classes.append(leg_class)
                start_dates.append(start)
                covered_sessions.append(covered)
                span_statuses.append(span_status)
                dist_counts.append(dist_count)
                del_counts.append(del_count)
                del_statuses.append(
                    posting_treatment.status + "_posting_payload_suppressed"
                    if posting_leg and posting_treatment is not None
                    else (
                        "pre_scope_delisting_context_suppressed"
                        if posting_leg
                        else ("none" if treatment is None else treatment.status)
                    )
                )
                adjustments.append(adjustment)
                authoritative.append(value)
                outcome_availability_dates.append(outcome_availability_date)
                authority_states.append(state)
                reconciliation.append(rec)
                usable.append(value is not None)
                states[state] += 1
                aggregate = month_aggregates.setdefault(effective_month_key, MonthAggregate())
                _update_month(
                    aggregate,
                    raw=raw,
                    authorized=value,
                    return_leg_class=leg_class,
                    state=state,
                    count_observation=not posting_leg,
                )

            payload = {
                "permno": source["permno"].combine_chunks(),
                "calendar_date": source["calendar_date"].combine_chunks(),
                "raw_total_return": source["raw_total_return"].combine_chunks(),
                "raw_return_ex_distributions": source["raw_return_ex_distributions"].combine_chunks(),
                "raw_return_missing_flag": source["raw_return_missing_flag"].combine_chunks(),
                "raw_return_duration_flag": source["raw_return_duration_flag"].combine_chunks(),
                "mapped_trading_periods": mapped_periods,
                "return_leg_class": leg_classes,
                "return_span_start_date": start_dates,
                "return_span_end_date": source["calendar_date"].combine_chunks(),
                "covered_trading_sessions": covered_sessions,
                "span_alignment_status": span_statuses,
                "distribution_event_count": dist_counts,
                "delisting_event_count": del_counts,
                "delisting_treatment_status": del_statuses,
                "delisting_adjustment_return": adjustments,
                "authoritative_total_return": authoritative,
                "outcome_availability_date": outcome_availability_dates,
                "authoritative_return_state": authority_states,
                "reconciliation_status": reconciliation,
                "return_history_usable": usable,
                "return_history_authorized": [False] * rows,
            }
            output = pa.Table.from_pydict(payload, schema=schema)
            writer.write_table(output, row_group_size=rows)
            offset += rows
            print(
                f"daily build: {offset:,}/{input_file.metadata.num_rows:,} rows",
                flush=True,
            )
    finally:
        writer.close()
    if unexplained_total_retx:
        raise ReturnReconciliationError(
            f"Found {unexplained_total_retx} unexplained DlyRet/DlyRetx differences"
        )
    return month_aggregates, dict(states)


def _build_monthly(
    settings: Settings,
    month_aggregates: dict[int, MonthAggregate],
    distributions: pd.DataFrame,
    delistings: pd.DataFrame,
    destination: Path,
) -> dict[str, int]:
    staging = pq.read_table(
        settings.typed_root / "crsp_security_month_return_staging.parquet"
    )
    schema = settings.schema.tables["crsp_security_month_return_reconciliation"].arrow_schema()
    permno = staging["permno"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    month_date = staging["calendar_month"].combine_chunks().to_numpy(zero_copy_only=False).astype("datetime64[D]")
    month_ord = month_date.astype("datetime64[M]").astype(np.int64)
    keys = permno * np.int64(10_000) + month_ord
    raw_monthly = staging["raw_total_return"].combine_chunks().to_pylist()
    flags = staging["raw_return_flag"].combine_chunks().to_numpy(zero_copy_only=False).astype(str)
    distribution_month_counts: Counter[int] = Counter(
        _month_key(int(row.permno), _month_ordinal_from_date(row.ex_date))
        for row in distributions.itertuples()
    )
    delisting_month_counts: Counter[int] = Counter(
        _month_key(int(row.permno), _month_ordinal_from_date(row.delisting_date))
        for row in delistings.itertuples()
    )

    columns: dict[str, list[Any]] = {name: [] for name in (
        "monthly_flag_class",
        "raw_compound",
        "authorized_compound",
        "observation_count",
        "usable_count",
        "missing_count",
        "multi_count",
        "distribution_count",
        "delisting_count",
        "coverage_class",
        "raw_signed",
        "raw_absolute",
        "authorized_signed",
        "authorized_absolute",
        "raw_status",
        "authorized_status",
        "reconciliation_status",
    )}
    status_counts: Counter[str] = Counter()
    hard_failures = 0
    for index, key_value in enumerate(keys):
        key = int(key_value)
        aggregate = month_aggregates.get(key, MonthAggregate())
        raw_compound = _quantized_return(aggregate.raw_gross, aggregate.raw_count)
        authorized_compound = _quantized_return(
            aggregate.authorized_gross, aggregate.authorized_count
        )
        vendor = raw_monthly[index]
        raw_signed = None if vendor is None or raw_compound is None else (
            raw_compound - vendor
        ).quantize(RETURN_QUANTUM)
        auth_signed = None if vendor is None or authorized_compound is None else (
            authorized_compound - vendor
        ).quantize(RETURN_QUANTUM)
        raw_absolute = None if raw_signed is None else abs(raw_signed)
        auth_absolute = None if auth_signed is None else abs(auth_signed)
        flag = flags[index]
        if vendor is None:
            raw_status = "vendor_monthly_missing"
        elif raw_compound is None:
            raw_status = "no_numeric_daily_return"
        elif within_monthly_tolerance(raw_compound, vendor):
            raw_status = "match_within_locked_tolerance"
        elif flag == "DE":
            raw_status = "delisting_conditioned_difference_missing_MthDelFlg"
        elif flag in {"GP", "IP", "MP"}:
            raw_status = "incomplete_month_diagnostic_difference"
        else:
            raw_status = "hard_unexplained_numeric_transport_difference"
            hard_failures += 1

        if vendor is None:
            auth_status = "vendor_monthly_missing"
        elif authorized_compound is None:
            auth_status = "no_authorized_daily_return"
        elif aggregate.mv_count:
            auth_status = "expected_difference_numeric_MV_quarantined"
        elif aggregate.imputed_count:
            auth_status = "expected_difference_missing_performance_imputation"
        elif aggregate.unresolved_delisting_count:
            auth_status = "nonperformance_terminal_outcome_unresolved"
        elif aggregate.event_adjustment_count:
            auth_status = "observed_delisting_adjustment_attached_once"
        elif within_monthly_tolerance(authorized_compound, vendor):
            auth_status = "match_within_locked_tolerance"
        elif flag in {"DE", "GP", "IP", "MP"}:
            auth_status = "diagnostic_vendor_difference"
        else:
            auth_status = "authorized_difference_due_missing_daily_return"

        if aggregate.imputed_count:
            coverage = "missing_performance_delisting_imputed"
        elif aggregate.unresolved_delisting_count:
            coverage = "nonperformance_delisting_outcome_unresolved"
        elif aggregate.mv_count:
            coverage = "numeric_MV_quarantined"
        elif aggregate.authorized_count == aggregate.observation_count:
            coverage = "complete_authorized_daily_coverage"
        elif aggregate.authorized_count == 0:
            coverage = "no_authorized_daily_returns"
        else:
            coverage = "partial_authorized_daily_coverage"
        overall = "hard_failure" if raw_status.startswith("hard_") else "accounted"
        status_counts[overall] += 1
        values = {
            "monthly_flag_class": _monthly_flag_class(flag),
            "raw_compound": raw_compound,
            "authorized_compound": authorized_compound,
            "observation_count": aggregate.observation_count,
            "usable_count": aggregate.authorized_count,
            "missing_count": aggregate.observation_count - aggregate.authorized_count,
            "multi_count": aggregate.multi_period_count,
            "distribution_count": int(distribution_month_counts[key]),
            "delisting_count": int(delisting_month_counts[key]),
            "coverage_class": coverage,
            "raw_signed": raw_signed,
            "raw_absolute": raw_absolute,
            "authorized_signed": auth_signed,
            "authorized_absolute": auth_absolute,
            "raw_status": raw_status,
            "authorized_status": auth_status,
            "reconciliation_status": overall,
        }
        for name, value in values.items():
            columns[name].append(value)

    if hard_failures:
        raise ReturnReconciliationError(
            f"Monthly numeric-transport hard comparison failed for {hard_failures} rows"
        )
    payload = {
        "permno": staging["permno"].combine_chunks(),
        "calendar_month": staging["calendar_month"].combine_chunks(),
        "raw_monthly_total_return": staging["raw_total_return"].combine_chunks(),
        "raw_monthly_return_ex_distributions": staging["raw_return_ex_distributions"].combine_chunks(),
        "raw_monthly_return_flag": staging["raw_return_flag"].combine_chunks(),
        "monthly_flag_class": columns["monthly_flag_class"],
        "compounded_raw_numeric_daily_total_return": columns["raw_compound"],
        "compounded_authorized_daily_total_return": columns["authorized_compound"],
        "daily_observation_count": columns["observation_count"],
        "usable_daily_return_count": columns["usable_count"],
        "missing_daily_return_count": columns["missing_count"],
        "multi_period_return_count": columns["multi_count"],
        "distribution_event_count": columns["distribution_count"],
        "delisting_event_count": columns["delisting_count"],
        "daily_coverage_class": columns["coverage_class"],
        "numeric_transport_signed_difference": columns["raw_signed"],
        "numeric_transport_absolute_difference": columns["raw_absolute"],
        "authorized_signed_difference": columns["authorized_signed"],
        "authorized_absolute_difference": columns["authorized_absolute"],
        "tolerance_used": [MONTHLY_ABSOLUTE_TOLERANCE] * len(keys),
        "numeric_transport_status": columns["raw_status"],
        "authorized_return_diagnostic_status": columns["authorized_status"],
        "reconciliation_status": columns["reconciliation_status"],
    }
    table = pa.Table.from_pydict(payload, schema=schema)
    pq.write_table(table, destination, compression="zstd", use_dictionary=True, row_group_size=262_144)
    return dict(status_counts)


def _validate_table(path: Path, spec: Any) -> dict[str, Any]:
    file = pq.ParquetFile(path)
    if file.metadata.num_rows != spec.expected_rows:
        raise ReturnReconciliationError(
            f"{path.name} row count {file.metadata.num_rows} != {spec.expected_rows}"
        )
    expected_schema = spec.arrow_schema()
    if not file.schema_arrow.equals(expected_schema, check_metadata=True):
        raise ReturnReconciliationError(f"Exact schema differs in {path.name}")
    required_columns = [field.name for field in expected_schema if not field.nullable]
    scan_columns = list(dict.fromkeys([*required_columns, *spec.primary_key]))
    previous_key: tuple[Any, ...] | None = None
    scanned_rows = 0
    for group_index in range(file.num_row_groups):
        table = file.read_row_group(group_index, columns=scan_columns)
        scanned_rows += table.num_rows
        for name in required_columns:
            if table[name].null_count:
                raise ReturnReconciliationError(
                    f"Null in required column {name!r} of {path.name}"
                )
        arrays = [
            table[name].combine_chunks().to_numpy(zero_copy_only=False)
            for name in spec.primary_key
        ]
        if table.num_rows == 0:
            continue
        first_key = tuple(array[0] for array in arrays)
        last_key = tuple(array[-1] for array in arrays)
        if previous_key is not None and first_key <= previous_key:
            raise ReturnReconciliationError(
                f"Primary key is not strictly sorted across row groups in {path.name}"
            )
        equal_prefix = np.ones(table.num_rows - 1, dtype=bool)
        descending = np.zeros(table.num_rows - 1, dtype=bool)
        for array in arrays:
            descending |= equal_prefix & (array[1:] < array[:-1])
            equal_prefix &= array[1:] == array[:-1]
        if np.any(descending):
            raise ReturnReconciliationError(f"Primary key is not sorted in {path.name}")
        if np.any(equal_prefix):
            raise ReturnReconciliationError(f"Duplicate primary key in {path.name}")
        previous_key = last_key
    if scanned_rows != spec.expected_rows:
        raise ReturnReconciliationError(f"Streaming QA row count differs in {path.name}")
    return {
        "path": path.name,
        "rows": spec.expected_rows,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(_canonical_json(value))
    os.replace(temporary, path)


def build(settings: Settings, *, publish_pointer: bool = True) -> dict[str, Any]:
    if publish_pointer and not settings.is_production:
        raise ReturnReconciliationError("A rehearsal build may never publish a production pointer")
    if (
        settings.interim_dir.exists()
        or settings.quality_dir.exists()
        or settings.artifact_dir.exists()
    ):
        raise ReturnReconciliationError(
            f"Build destination already exists for {settings.build_id}; refusing overwrite"
        )
    interim_stage = settings.interim_dir.with_name(".staging-" + settings.build_id)
    quality_stage = settings.quality_dir.with_name(".staging-" + settings.build_id)
    artifact_stage = settings.artifact_dir.with_name(".staging-" + settings.build_id)
    if interim_stage.exists() or quality_stage.exists() or artifact_stage.exists():
        raise ReturnReconciliationError("A prior staging directory exists; inspect it before retrying")
    interim_stage.mkdir(parents=True)
    quality_stage.mkdir(parents=True)
    artifact_stage.mkdir(parents=True)
    try:
        print("return gate 1/5: loading verified typed arrays and event ledgers", flush=True)
        distributions, delistings, delisting_context = _read_events(settings)
        arrays = _load_daily_arrays(settings)
        calendar = _calendar_ordinals(settings)
        structure = _daily_structure(arrays, calendar, delisting_context)
        print("return gate 2/5: resolving distribution and delisting ownership", flush=True)
        distribution_ledger, distribution_counts, distribution_status = _distribution_mapping(
            distributions, arrays, structure, calendar
        )
        delisting_ledger, treatments, delisting_counts = _delisting_treatments(
            delistings, arrays, structure
        )

        distribution_path = interim_stage / "crsp_distribution_event.parquet"
        delisting_path = interim_stage / "crsp_delisting_event.parquet"
        pq.write_table(
            _event_table(
                distribution_ledger,
                "crsp_distribution_event",
                settings.schema.tables["crsp_distribution_event"].arrow_schema(),
            ),
            distribution_path,
            compression="zstd",
        )
        pq.write_table(
            _event_table(
                delisting_ledger,
                "crsp_delisting_event",
                settings.schema.tables["crsp_delisting_event"].arrow_schema(),
            ),
            delisting_path,
            compression="zstd",
        )

        print("return gate 3/5: writing authorized daily return history", flush=True)
        daily_path = interim_stage / "crsp_security_day_return.parquet"
        month_aggregates, daily_states = _build_daily_and_month_aggregates(
            settings,
            arrays,
            structure,
            distribution_counts,
            treatments,
            delisting_counts,
            daily_path,
        )
        print("return gate 4/5: writing daily-to-monthly reconciliation", flush=True)
        monthly_path = interim_stage / "crsp_security_month_return_reconciliation.parquet"
        monthly_status = _build_monthly(
            settings,
            month_aggregates,
            distribution_ledger,
            delisting_ledger,
            monthly_path,
        )

        outputs = []
        for name, spec in settings.schema.tables.items():
            outputs.append(
                _validate_table(interim_stage / f"{name}.parquet", spec)
            )
        summary = {
            "schema_version": 1,
            "build_id": settings.build_id,
            "contract_id": CONTRACT_ID,
            "status": "return_reconciliation_primary_passed_pending_independent_verification",
            "scope": {"start": str(SCOPE_START), "end": str(SCOPE_END)},
            "counts": {
                "distribution_rows": len(distribution_ledger),
                "delisting_rows": len(delisting_ledger),
                "daily_rows": len(arrays["permno"]),
                "monthly_rows": settings.schema.tables[
                    "crsp_security_month_return_reconciliation"
                ].expected_rows,
                "imputed_missing_performance_delistings": int(
                    delisting_ledger["imputation_applied"].sum()
                ),
            },
            "distribution_group_status": distribution_status,
            "daily_authority_states": daily_states,
            "monthly_reconciliation_status": monthly_status,
            "outputs": outputs,
            "bindings": {
                "empirical_lock_sha256": _sha256(EMPIRICAL_LOCK_PATH),
                "policy_sha256": _sha256(POLICY_PATH),
                "schema_sha256": _sha256(SCHEMA_PATH),
                "typed_pointer_sha256": _sha256(TYPED_POINTER_PATH),
                "calendar_pointer_sha256": _sha256(CALENDAR_POINTER_PATH),
                "implementation_sha256": _sha256(Path(__file__)),
            },
            "authorizations": {
                "return_history_authorized": False,
                "adjusted_price_path_feature_authorized": False,
                "independent_verification_required": True,
                "beta_estimation_authorized": False,
                "features_labels_models_portfolios_pnl_performance_or_backtests_authorized": False,
                "validation_or_final_test_access_authorized": False,
            },
        }
        (interim_stage / "_SUCCESS.json").write_bytes(_canonical_json(summary))
        (quality_stage / "summary.json").write_bytes(_canonical_json(summary))
        with (quality_stage / "checks.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"])
            writer.writeheader()
            for check, detail in (
                ("distribution_event_conservation", len(distribution_ledger)),
                ("delisting_event_conservation", len(delisting_ledger)),
                ("daily_key_conservation", len(arrays["permno"])),
                ("missing_performance_imputation", int(delisting_ledger["imputation_applied"].sum())),
                ("unexplained_total_minus_retx", 0),
                ("monthly_hard_failures", monthly_status.get("hard_failure", 0)),
            ):
                writer.writerow({"check": check, "status": "pass", "detail": detail})

        manifest = {
            **summary,
            "interim_path": str(settings.interim_dir.relative_to(settings.output_database_root)),
            "quality_path": str(settings.quality_dir.relative_to(settings.output_database_root)),
            "primary_manifest_sha256": "pending",
        }
        manifest_bytes = _canonical_json({key: value for key, value in manifest.items() if key != "primary_manifest_sha256"})
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        manifest["primary_manifest_sha256"] = manifest_sha
        (artifact_stage / "manifest.json").write_bytes(_canonical_json(manifest))
        artifact_record = (
            str(settings.artifact_dir.relative_to(PROJECT_ROOT))
            if settings.is_production
            else str(settings.artifact_dir.relative_to(settings.output_database_root))
        )
        pointer = {
            "schema_version": 1,
            "status": "return_reconciliation_primary_passed_pending_independent_verification",
            "build_id": settings.build_id,
            "contract_id": CONTRACT_ID,
            "artifact_path": artifact_record,
            "interim_path": manifest["interim_path"],
            "quality_path": manifest["quality_path"],
            "manifest_sha256": _sha256(artifact_stage / "manifest.json"),
            "return_history_authorized": False,
            "adjusted_price_path_feature_authorized": False,
        }
        moved: list[tuple[Path, Path]] = []
        try:
            for stage, final in (
                (interim_stage, settings.interim_dir),
                (quality_stage, settings.quality_dir),
                (artifact_stage, settings.artifact_dir),
            ):
                os.replace(stage, final)
                moved.append((stage, final))
            if publish_pointer:
                _atomic_json(settings.pointer_path, pointer)
        except Exception:
            for stage, final in reversed(moved):
                if final.exists() and not stage.exists():
                    os.replace(final, stage)
            raise
        print("return gate 5/5: primary build complete; independent verification still required", flush=True)
        return pointer
    except Exception:
        # Failed staging is kept for diagnosis; no authority pointer advances.
        raise
