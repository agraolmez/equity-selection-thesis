"""Build the development-only signal-date equity universe.

The transform is intentionally limited to eligibility inputs.  It creates no
features, labels, scores, portfolios, P&L, or performance statistics.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tomllib
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data.database_location import resolve_recorded_database_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE_CONFIG_PATH = PROJECT_ROOT / "configs/phase_02_canonical_panel.toml"
POLICY_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/signal_date_universe_policy_v1.toml"
SCHEMA_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/signal_date_universe_schema_v1.json"
LOCK_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/signal_date_universe_empirical_lock_v1.json"
PATHS_PATH = PROJECT_ROOT / "configs/paths.toml"
TYPED_VERIFIED_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CRSP_TYPED_BACKBONE_VERIFIED.json"
TYPED_PRIMARY_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CRSP_TYPED_BACKBONE_PRIMARY.json"
RETURN_VERIFIED_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CRSP_RETURN_RECONCILIATION_VERIFIED.json"
RETURN_PRIMARY_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CRSP_RETURN_RECONCILIATION_PRIMARY.json"
CALENDAR_VERIFIED_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CALENDAR_VERIFIED.json"
SCOPE_END_SIGNAL = dt.date(2014, 10, 31)


class SignalUniverseError(RuntimeError):
    """Raised when the signal universe cannot pass a hard contract gate."""


@dataclass(frozen=True)
class Settings:
    database_root: Path
    typed_root: Path
    return_root: Path
    calendar_path: Path
    phase_config: dict[str, Any]


@dataclass
class UniverseResult:
    columns: dict[str, Any]
    summary: dict[str, Any]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise SignalUniverseError(f"Expected JSON object at {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _database_path(recorded: str, database_root: Path) -> Path:
    path = Path(recorded)
    if not path.is_absolute():
        return database_root / path
    return resolve_recorded_database_path(path, current_database_root=database_root)


def _verified_build_root(
    verified_path: Path,
    primary_path: Path,
    *,
    database_root: Path,
    expected_status: str,
    required_authorization: str | None = None,
) -> Path:
    verified = _json(verified_path)
    primary = _json(primary_path)
    if verified.get("status") != "verified_passed":
        raise SignalUniverseError(f"Upstream verified pointer is not passed: {verified_path.name}")
    if primary.get("status") != expected_status:
        raise SignalUniverseError(f"Upstream primary pointer status differs: {primary_path.name}")
    if required_authorization is not None and not verified.get("authorizations", {}).get(
        required_authorization, False
    ):
        raise SignalUniverseError(
            f"Upstream authorization {required_authorization!r} is not granted"
        )
    verified_manifest = verified.get("primary_manifest_sha256")
    primary_manifest = primary.get("manifest_sha256")
    if (
        verified.get("build_id") != primary.get("build_id")
        or verified_manifest != primary_manifest
    ):
        raise SignalUniverseError("Verified and primary upstream identities differ")
    manifest_path = PROJECT_ROOT / primary["artifact_path"] / "manifest.json"
    if _sha256(manifest_path) != primary_manifest:
        raise SignalUniverseError("Upstream primary manifest hash differs")
    return database_root / primary["interim_path"]


def load_settings() -> Settings:
    with PATHS_PATH.open("rb") as handle:
        paths = tomllib.load(handle)
    with PHASE_CONFIG_PATH.open("rb") as handle:
        phase = tomllib.load(handle)
    database_root = Path(os.environ.get("PORTFOLIO_DATABASE_ROOT", paths["database"]["root"]))
    typed_root = _verified_build_root(
        TYPED_VERIFIED_PATH,
        TYPED_PRIMARY_PATH,
        database_root=database_root,
        expected_status="typed_backbone_primary_passed_pending_independent_verification",
        required_authorization="universe_unit_inputs_authorized",
    )
    return_root = _verified_build_root(
        RETURN_VERIFIED_PATH,
        RETURN_PRIMARY_PATH,
        database_root=database_root,
        expected_status="return_reconciliation_primary_passed_pending_independent_verification",
        required_authorization="return_history_authorized",
    )
    calendar_pointer = _json(CALENDAR_VERIFIED_PATH)
    if calendar_pointer.get("status") != "verified_passed":
        raise SignalUniverseError("Verified calendar pointer is not passed")
    calendar_manifest_path = (
        PROJECT_ROOT / calendar_pointer["primary_artifact_path"] / "manifest.json"
    )
    if _sha256(calendar_manifest_path) != calendar_pointer["primary_manifest_sha256"]:
        raise SignalUniverseError("Calendar manifest hash differs")
    calendar_manifest = _json(calendar_manifest_path)
    matches = [
        item for item in calendar_manifest["outputs"]
        if item["path"].endswith("/trading_calendar.parquet")
    ]
    if len(matches) != 1:
        raise SignalUniverseError("Verified trading calendar output is not unique")
    calendar_path = _database_path(matches[0]["path"], database_root)
    if (
        calendar_path.stat().st_size != matches[0]["size_bytes"]
        or _sha256(calendar_path) != matches[0]["sha256"]
    ):
        raise SignalUniverseError("Verified trading calendar output differs")
    return Settings(database_root, typed_root, return_root, calendar_path, phase)


def _scaled_decimal(
    array: pa.ChunkedArray | pa.Array,
    *,
    scale: int,
    precision: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    combined = array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array
    valid = pc.is_valid(combined).to_numpy(zero_copy_only=False).astype(bool)
    narrowed = pc.cast(combined, pa.decimal128(precision, scale))
    multiplier = pa.scalar(Decimal(10**scale), type=pa.decimal128(scale + 1, 0))
    scaled = pc.cast(pc.multiply(narrowed, multiplier), pa.int64())
    values = pc.fill_null(scaled, 0).to_numpy(zero_copy_only=False).astype(np.int64)
    return values, valid


def _signal_calendar(settings: Settings) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(settings.calendar_path)
    dates = table["calendar_date"].combine_chunks().to_numpy(zero_copy_only=False).astype("datetime64[D]")
    ordinals = dates.astype(np.int64)
    month_end = table["is_observed_month_end"].combine_chunks().to_numpy(zero_copy_only=False).astype(bool)
    end_ordinal = np.datetime64(SCOPE_END_SIGNAL, "D").astype(np.int64)
    all_month_end_positions = np.flatnonzero(month_end)
    signal_positions = all_month_end_positions[
        (ordinals[all_month_end_positions] <= end_ordinal)
        & (
            ordinals[all_month_end_positions]
            >= np.datetime64("2000-01-01", "D").astype(np.int64)
        )
    ]
    if len(signal_positions) < 2 or ordinals[signal_positions[-1]] != end_ordinal:
        raise SignalUniverseError("Development signal boundary differs from the frozen calendar")
    entry_positions = signal_positions + 1
    next_month_end_index = np.searchsorted(
        all_month_end_positions, signal_positions, side="right"
    )
    if np.any(next_month_end_index >= len(all_month_end_positions)):
        raise SignalUniverseError("Calendar lacks the month end after a development signal")
    next_signal_positions = all_month_end_positions[next_month_end_index]
    exit_positions = next_signal_positions + 1
    if np.any(entry_positions >= len(ordinals)) or np.any(exit_positions >= len(ordinals)):
        raise SignalUniverseError("Calendar lacks entry/exit context after the last signal")
    if np.any(ordinals[entry_positions] <= ordinals[signal_positions]) or np.any(
        ordinals[exit_positions] <= ordinals[next_signal_positions]
    ):
        raise SignalUniverseError("Calendar entry/exit schedule is not strictly forward")
    return ordinals, ordinals[signal_positions], ordinals[entry_positions], ordinals[exit_positions]


def _load_market_and_returns(
    settings: Settings,
    signal_ordinals: np.ndarray,
) -> dict[str, np.ndarray]:
    market_path = settings.typed_root / "crsp_security_day.parquet"
    market = pq.read_table(
        market_path,
        columns=[
            "permno",
            "calendar_date",
            "conditional_type",
            "close_usd_per_share",
            "market_cap_usd",
            "volume_shares",
            "screening_dollar_volume_usd",
            "screening_price_valid",
            "market_cap_valid",
            "liquidity_session_valid",
        ],
    )
    permno = market["permno"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    day = (
        market["calendar_date"].combine_chunks().to_numpy(zero_copy_only=False)
        .astype("datetime64[D]")
        .astype(np.int64)
    )
    if np.any(
        (permno[1:] < permno[:-1])
        | ((permno[1:] == permno[:-1]) & (day[1:] <= day[:-1]))
    ):
        raise SignalUniverseError("Market table keys are not strictly sorted")
    signal_mask = np.isin(day, signal_ordinals)
    signal_index = np.flatnonzero(signal_mask)
    taken = pa.array(signal_index)
    close_scaled, close_valid = _scaled_decimal(market["close_usd_per_share"].combine_chunks().take(taken), scale=6)
    cap_scaled, cap_valid = _scaled_decimal(market["market_cap_usd"].combine_chunks().take(taken), scale=2)
    dollar_volume_scaled, dollar_volume_valid = _scaled_decimal(
        market["screening_dollar_volume_usd"], scale=6
    )
    volume = pc.fill_null(market["volume_shares"].combine_chunks(), -1).to_numpy(zero_copy_only=False).astype(np.int64)
    conditional_array = market["conditional_type"].combine_chunks().take(taken)
    price_flag_array = market["screening_price_valid"].combine_chunks().take(taken)
    cap_flag_array = market["market_cap_valid"].combine_chunks().take(taken)
    liquidity_array = market["liquidity_session_valid"].combine_chunks()
    if any(
        array.null_count
        for array in (conditional_array, price_flag_array, cap_flag_array, liquidity_array)
    ):
        raise SignalUniverseError("Typed market contract columns contain nulls")
    conditional = conditional_array.to_numpy(zero_copy_only=False).astype(str)
    price_flag = price_flag_array.to_numpy(zero_copy_only=False).astype(bool)
    cap_flag = cap_flag_array.to_numpy(zero_copy_only=False).astype(bool)
    liquidity_valid = liquidity_array.to_numpy(zero_copy_only=False).astype(bool)
    if np.any(liquidity_valid != dollar_volume_valid):
        raise SignalUniverseError("Typed liquidity validity and dollar-volume availability differ")
    if np.any(liquidity_valid & (volume < 0)):
        raise SignalUniverseError("Valid liquidity session has negative or null volume")

    return_path = settings.return_root / "crsp_security_day_return.parquet"
    return_file = pq.ParquetFile(return_path)
    return_usable_parts: list[np.ndarray] = []
    availability_parts: list[np.ndarray] = []
    availability_valid_parts: list[np.ndarray] = []
    offset = 0
    for group_index in range(return_file.num_row_groups):
        table = return_file.read_row_group(
            group_index,
            columns=[
                "permno",
                "calendar_date",
                "outcome_availability_date",
                "return_history_usable",
            ],
        )
        rows = table.num_rows
        group_permno = table["permno"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
        group_day = (
            table["calendar_date"].combine_chunks().to_numpy(zero_copy_only=False)
            .astype("datetime64[D]")
            .astype(np.int64)
        )
        if not np.array_equal(group_permno, permno[offset : offset + rows]) or not np.array_equal(
            group_day, day[offset : offset + rows]
        ):
            raise SignalUniverseError("Verified market and return keys differ")
        usable_array = table["return_history_usable"].combine_chunks()
        if usable_array.null_count:
            raise SignalUniverseError("Return-history usability contains nulls")
        return_usable_parts.append(
            usable_array.to_numpy(zero_copy_only=False).astype(bool)
        )
        availability_array = table["outcome_availability_date"].combine_chunks()
        availability_valid_parts.append(
            pc.is_valid(availability_array)
            .to_numpy(zero_copy_only=False)
            .astype(bool)
        )
        availability_parts.append(
            availability_array.to_numpy(zero_copy_only=False)
            .astype("datetime64[D]")
            .astype(np.int64)
        )
        offset += rows
    if offset != len(permno):
        raise SignalUniverseError("Verified market and return row counts differ")
    return {
        "permno": permno,
        "day": day,
        "signal_index": signal_index,
        "signal_permno": permno[signal_index],
        "signal_day": day[signal_index],
        "conditional": conditional,
        "close_scaled": close_scaled,
        "close_present": close_valid,
        "screening_price_valid": price_flag,
        "cap_scaled": cap_scaled,
        "cap_present": cap_valid,
        "market_cap_valid": cap_flag,
        "volume": volume,
        "dollar_volume_scaled": dollar_volume_scaled,
        "liquidity_valid": liquidity_valid,
        "return_usable": np.concatenate(return_usable_parts),
        "return_availability": np.concatenate(availability_parts),
        "return_availability_valid": np.concatenate(availability_valid_parts),
    }


def _median_twice(values: np.ndarray) -> int:
    size = len(values)
    if size == 0:
        return -1
    middle = size // 2
    partitioned = np.partition(values, (middle - 1, middle) if size % 2 == 0 else middle)
    if size % 2:
        return int(partitioned[middle]) * 2
    return int(partitioned[middle - 1]) + int(partitioned[middle])


def _rolling_metrics(
    arrays: dict[str, np.ndarray],
    calendar: np.ndarray,
    signal_ordinals: np.ndarray,
) -> dict[str, np.ndarray]:
    signal_index = arrays["signal_index"]
    signal_day = arrays["signal_day"]
    count = len(signal_index)
    valid_liquidity = np.zeros(count, dtype=np.int16)
    positive_volume = np.zeros(count, dtype=np.int16)
    median_twice = np.full(count, -1, dtype=np.int64)
    valid_returns = np.zeros(count, dtype=np.int16)
    liquidity_start = np.zeros(count, dtype=np.int32)
    return_start = np.zeros(count, dtype=np.int32)
    liquidity_calendar_sessions = np.zeros(count, dtype=np.int16)
    return_calendar_sessions = np.zeros(count, dtype=np.int16)
    signal_calendar_positions = np.searchsorted(calendar, signal_day)
    if np.any(calendar[signal_calendar_positions] != signal_day):
        raise SignalUniverseError("A signal row date is absent from the verified calendar")

    unique_permno, group_starts, group_counts = np.unique(
        arrays["permno"], return_index=True, return_counts=True
    )
    del unique_permno
    signal_cursor = 0
    for group_number, (group_start, group_count) in enumerate(
        zip(group_starts, group_counts, strict=True)
    ):
        group_end = int(group_start + group_count)
        first_signal = int(np.searchsorted(signal_index, group_start, side="left"))
        last_signal = int(np.searchsorted(signal_index, group_end, side="left"))
        if first_signal == last_signal:
            continue
        group_days = arrays["day"][group_start:group_end]
        for signal_position in range(first_signal, last_signal):
            global_row = int(signal_index[signal_position])
            local_end = global_row - int(group_start) + 1
            calendar_position = int(signal_calendar_positions[signal_position])
            liquidity_start_day = int(calendar[max(0, calendar_position - 59)])
            return_start_day = int(calendar[max(0, calendar_position - 251)])
            liquidity_start[signal_position] = liquidity_start_day
            return_start[signal_position] = return_start_day
            liquidity_calendar_sessions[signal_position] = min(
                60, calendar_position + 1
            )
            return_calendar_sessions[signal_position] = min(
                252, calendar_position + 1
            )
            liquidity_left = int(np.searchsorted(group_days, liquidity_start_day, side="left"))
            return_left = int(np.searchsorted(group_days, return_start_day, side="left"))
            liquidity_slice = slice(int(group_start) + liquidity_left, int(group_start) + local_end)
            valid = arrays["liquidity_valid"][liquidity_slice]
            valid_count = int(valid.sum())
            valid_liquidity[signal_position] = valid_count
            if valid_count:
                volume = arrays["volume"][liquidity_slice][valid]
                positive_volume[signal_position] = int(np.sum(volume > 0))
                median_twice[signal_position] = _median_twice(
                    arrays["dollar_volume_scaled"][liquidity_slice][valid]
                )
            return_slice = slice(int(group_start) + return_left, int(group_start) + local_end)
            valid_returns[signal_position] = int(
                np.sum(
                    arrays["return_usable"][return_slice]
                    & arrays["return_availability_valid"][return_slice]
                    & (arrays["return_availability"][return_slice] <= signal_day[signal_position])
                )
            )
            signal_cursor += 1
        if group_number % 2_000 == 0:
            print(f"universe rolling metrics: {signal_cursor:,}/{count:,} signal rows", flush=True)
    if signal_cursor != count:
        raise SignalUniverseError("Not every signal row received rolling metrics")
    return {
        "valid_liquidity": valid_liquidity,
        "positive_volume": positive_volume,
        "median_twice": median_twice,
        "valid_returns": valid_returns,
        "liquidity_start": liquidity_start,
        "return_start": return_start,
        "liquidity_calendar_sessions": liquidity_calendar_sessions,
        "return_calendar_sessions": return_calendar_sessions,
    }


def _match_identity_indices(
    *,
    interval_permno: np.ndarray,
    interval_start: np.ndarray,
    interval_end: np.ndarray,
    signal_permno: np.ndarray,
    signal_day: np.ndarray,
) -> np.ndarray:
    """Return the unique effective-dated identity row for every signal row."""

    if not (
        len(interval_permno) == len(interval_start) == len(interval_end)
        and len(signal_permno) == len(signal_day)
    ):
        raise SignalUniverseError("Identity matching arrays have inconsistent lengths")
    if np.any(interval_start > interval_end):
        raise SignalUniverseError("Identity interval starts after its end date")
    if len(interval_permno) > 1:
        sorted_intervals = (interval_permno[1:] > interval_permno[:-1]) | (
            (interval_permno[1:] == interval_permno[:-1])
            & (interval_start[1:] > interval_start[:-1])
        )
        if not np.all(sorted_intervals):
            raise SignalUniverseError("Identity intervals are not strictly key-sorted")
        overlaps = (interval_permno[1:] == interval_permno[:-1]) & (
            interval_start[1:] <= interval_end[:-1]
        )
        if np.any(overlaps):
            raise SignalUniverseError("Identity intervals overlap at inclusive boundaries")
    if len(signal_permno) > 1:
        sorted_signals = (signal_permno[1:] > signal_permno[:-1]) | (
            (signal_permno[1:] == signal_permno[:-1])
            & (signal_day[1:] > signal_day[:-1])
        )
        if not np.all(sorted_signals):
            raise SignalUniverseError("Signal rows are not strictly key-sorted")

    bounds: dict[int, tuple[int, int]] = {}
    unique, starts, counts = np.unique(
        interval_permno, return_index=True, return_counts=True
    )
    for permno, start, count in zip(unique, starts, counts, strict=True):
        bounds[int(permno)] = (int(start), int(start + count))

    match_index = np.full(len(signal_permno), -1, dtype=np.int64)
    signal_unique, signal_starts, signal_counts = np.unique(
        signal_permno, return_index=True, return_counts=True
    )
    for permno, signal_start, signal_count in zip(
        signal_unique, signal_starts, signal_counts, strict=True
    ):
        bound = bounds.get(int(permno))
        if bound is None:
            continue
        left, right = bound
        dates = signal_day[signal_start : signal_start + signal_count]
        local = np.searchsorted(interval_start[left:right], dates, side="right") - 1
        valid = local >= 0
        safe = np.maximum(local, 0) + left
        valid &= interval_end[safe] >= dates
        segment = match_index[signal_start : signal_start + signal_count]
        segment[valid] = safe[valid]
        match_index[signal_start : signal_start + signal_count] = segment
    missing = int(np.sum(match_index < 0))
    if missing:
        raise SignalUniverseError(
            f"Identity interval cardinality is zero for {missing:,} signal rows"
        )
    return match_index


def _identity_at_signal(settings: Settings, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    identity = pq.read_table(settings.typed_root / "crsp_identity_interval.parquet")
    required_nonnull = (
        "permno",
        "permco",
        "start_date",
        "end_date",
        "us_incorporation_flag",
        "security_type",
        "security_subtype",
        "share_type",
        "primary_exchange",
        "trading_status_flag",
    )
    null_columns = [name for name in required_nonnull if identity[name].null_count]
    if null_columns:
        raise SignalUniverseError(
            f"Identity contract columns contain nulls: {', '.join(null_columns)}"
        )
    interval_permno = identity["permno"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    interval_start = (
        identity["start_date"].combine_chunks().to_numpy(zero_copy_only=False)
        .astype("datetime64[D]")
        .astype(np.int64)
    )
    interval_end = (
        identity["end_date"].combine_chunks().to_numpy(zero_copy_only=False)
        .astype("datetime64[D]")
        .astype(np.int64)
    )
    match_index = _match_identity_indices(
        interval_permno=interval_permno,
        interval_start=interval_start,
        interval_end=interval_end,
        signal_permno=arrays["signal_permno"],
        signal_day=arrays["signal_day"],
    )
    matched = np.ones(len(match_index), dtype=bool)

    def integers(name: str) -> np.ndarray:
        source = identity[name].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
        return source[match_index]

    def dates(name: str) -> np.ndarray:
        source = (
            identity[name].combine_chunks().to_numpy(zero_copy_only=False)
            .astype("datetime64[D]")
            .astype(np.int64)
        )
        return source[match_index].astype(np.int32)

    def strings(name: str) -> np.ndarray:
        source = identity[name].combine_chunks().to_numpy(zero_copy_only=False).astype(str)
        return source[match_index]

    return {
        "matched": matched,
        "permco": integers("permco"),
        "identity_start": dates("start_date"),
        "identity_end": dates("end_date"),
        "us_incorporation_flag": strings("us_incorporation_flag"),
        "security_type": strings("security_type"),
        "security_subtype": strings("security_subtype"),
        "share_type": strings("share_type"),
        "primary_exchange": strings("primary_exchange"),
        "trading_status_flag": strings("trading_status_flag"),
    }


def _screen_and_deduplicate(
    *,
    identity: dict[str, Any],
    arrays: dict[str, np.ndarray],
    rolling: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply the frozen exact screens and common long/short issuer selection."""

    security_type_pass = identity["security_type"] == "EQTY"
    security_subtype_pass = identity["security_subtype"] == "COM"
    share_type_pass = np.isin(identity["share_type"], ["NS", "AD"])
    conditional_type_pass = arrays["conditional"] == "RW"
    trading_status_pass = identity["trading_status_flag"] == "A"
    primary_exchange_pass = np.isin(identity["primary_exchange"], ["N", "A", "Q"])
    base_price_valid_pass = (
        arrays["screening_price_valid"]
        & arrays["close_present"]
        & (arrays["close_scaled"] > 0)
    )
    base_market_cap_valid_pass = (
        arrays["market_cap_valid"]
        & arrays["cap_present"]
        & (arrays["cap_scaled"] > 0)
    )
    base_eligible = (
        identity["matched"]
        & security_type_pass
        & security_subtype_pass
        & share_type_pass
        & conditional_type_pass
        & trading_status_pass
        & primary_exchange_pass
        & base_price_valid_pass
        & base_market_cap_valid_pass
    )

    positive_denominator = rolling["valid_liquidity"] > 0
    long_price_pass = base_price_valid_pass & (arrays["close_scaled"] >= 5_000_000)
    long_market_cap_pass = base_market_cap_valid_pass & (
        arrays["cap_scaled"] >= 30_000_000_000
    )
    long_valid_liquidity_sessions_pass = rolling["valid_liquidity"] >= 40
    long_median_dollar_volume_pass = (
        rolling["median_twice"] >= 10_000_000_000_000
    )
    long_positive_volume_fraction_pass = positive_denominator & (
        rolling["positive_volume"] * 100
        >= rolling["valid_liquidity"] * 90
    )
    long_return_history_pass = rolling["valid_returns"] >= 126
    long_eligible_pre_dedup = (
        base_eligible
        & long_price_pass
        & long_market_cap_pass
        & long_valid_liquidity_sessions_pass
        & long_median_dollar_volume_pass
        & long_positive_volume_fraction_pass
        & long_return_history_pass
    )

    short_price_pass = base_price_valid_pass & (
        arrays["close_scaled"] >= 10_000_000
    )
    short_market_cap_pass = base_market_cap_valid_pass & (
        arrays["cap_scaled"] >= 100_000_000_000
    )
    short_median_dollar_volume_pass = (
        rolling["median_twice"] >= 40_000_000_000_000
    )
    short_positive_volume_fraction_pass = positive_denominator & (
        rolling["positive_volume"] * 100
        >= rolling["valid_liquidity"] * 95
    )
    short_eligible_pre_dedup = (
        long_eligible_pre_dedup
        & short_price_pass
        & short_market_cap_pass
        & short_median_dollar_volume_pass
        & short_positive_volume_fraction_pass
    )

    row_count = len(long_eligible_pre_dedup)
    issuer_line_count = np.zeros(row_count, dtype=np.int32)
    issuer_line_rank = np.full(row_count, -1, dtype=np.int32)
    issuer_line_selected = np.zeros(row_count, dtype=bool)
    issuer_dedup_reason_code = np.full(
        row_count,
        "NOT_APPLICABLE_NOT_LONG_ELIGIBLE_PRE_DEDUP",
        dtype=object,
    )
    candidates = np.flatnonzero(long_eligible_pre_dedup)
    if len(candidates):
        candidate_order = np.lexsort(
            (
                arrays["signal_permno"][candidates],
                -arrays["cap_scaled"][candidates],
                -rolling["median_twice"][candidates],
                identity["permco"][candidates],
                arrays["signal_day"][candidates],
            )
        )
        ordered = candidates[candidate_order]
        group_start_mask = np.ones(len(ordered), dtype=bool)
        group_start_mask[1:] = (
            (arrays["signal_day"][ordered[1:]] != arrays["signal_day"][ordered[:-1]])
            | (identity["permco"][ordered[1:]] != identity["permco"][ordered[:-1]])
        )
        group_starts = np.flatnonzero(group_start_mask)
        group_ends = np.r_[group_starts[1:], len(ordered)]
        for group_start, group_end in zip(group_starts, group_ends, strict=True):
            group = ordered[group_start:group_end]
            size = len(group)
            issuer_line_count[group] = size
            issuer_line_rank[group] = np.arange(1, size + 1, dtype=np.int32)
            selected = int(group[0])
            issuer_line_selected[selected] = True
            if size == 1:
                issuer_dedup_reason_code[selected] = "SOLE_LONG_ELIGIBLE_LINE"
                continue
            runner_up = int(group[1])
            if rolling["median_twice"][selected] > rolling["median_twice"][runner_up]:
                issuer_dedup_reason_code[selected] = (
                    "SELECTED_HIGHEST_MEDIAN_DOLLAR_VOLUME"
                )
            elif arrays["cap_scaled"][selected] > arrays["cap_scaled"][runner_up]:
                issuer_dedup_reason_code[selected] = "SELECTED_MARKET_CAP_TIEBREAK"
            else:
                issuer_dedup_reason_code[selected] = "SELECTED_LOWEST_PERMNO_TIEBREAK"
            for not_selected in group[1:]:
                index = int(not_selected)
                if rolling["median_twice"][index] < rolling["median_twice"][selected]:
                    reason = "NOT_SELECTED_LOWER_MEDIAN_DOLLAR_VOLUME"
                elif arrays["cap_scaled"][index] < arrays["cap_scaled"][selected]:
                    reason = "NOT_SELECTED_LOWER_MARKET_CAP_TIEBREAK"
                else:
                    reason = "NOT_SELECTED_HIGHER_PERMNO_TIEBREAK"
                issuer_dedup_reason_code[index] = reason

    long_eligible = long_eligible_pre_dedup & issuer_line_selected
    short_eligible = short_eligible_pre_dedup & issuer_line_selected
    if np.any(short_eligible & ~long_eligible):
        raise SignalUniverseError("Final short eligibility does not imply final long eligibility")

    return {
        "security_type_pass": security_type_pass,
        "security_subtype_pass": security_subtype_pass,
        "share_type_pass": share_type_pass,
        "conditional_type_pass": conditional_type_pass,
        "trading_status_pass": trading_status_pass,
        "primary_exchange_pass": primary_exchange_pass,
        "base_price_valid_pass": base_price_valid_pass,
        "base_market_cap_valid_pass": base_market_cap_valid_pass,
        "base_eligible": base_eligible,
        "long_price_pass": long_price_pass,
        "long_market_cap_pass": long_market_cap_pass,
        "long_valid_liquidity_sessions_pass": long_valid_liquidity_sessions_pass,
        "long_median_dollar_volume_pass": long_median_dollar_volume_pass,
        "long_positive_volume_fraction_pass": long_positive_volume_fraction_pass,
        "long_return_history_pass": long_return_history_pass,
        "long_eligible_pre_dedup": long_eligible_pre_dedup,
        "short_price_pass": short_price_pass,
        "short_market_cap_pass": short_market_cap_pass,
        "short_median_dollar_volume_pass": short_median_dollar_volume_pass,
        "short_positive_volume_fraction_pass": short_positive_volume_fraction_pass,
        "short_eligible_pre_dedup": short_eligible_pre_dedup,
        "issuer_long_eligible_line_count": issuer_line_count,
        "issuer_line_rank": issuer_line_rank,
        "issuer_line_selected": issuer_line_selected,
        "issuer_dedup_reason_code": issuer_dedup_reason_code,
        "long_eligible": long_eligible,
        "short_eligible": short_eligible,
    }


def _reason_codes(
    screens: dict[str, np.ndarray],
) -> tuple[list[str], list[str], list[str]]:
    def rendered(codes: list[str]) -> str:
        return "|".join(codes) if codes else "PASS"

    base_codes: list[str] = []
    long_codes: list[str] = []
    short_codes: list[str] = []
    for index in range(len(screens["base_eligible"])):
        base: list[str] = []
        if not screens["security_type_pass"][index]:
            base.append("BASE_SECURITY_TYPE_NOT_EQTY")
        if not screens["security_subtype_pass"][index]:
            base.append("BASE_SECURITY_SUBTYPE_NOT_COM")
        if not screens["share_type_pass"][index]:
            base.append("BASE_SHARE_TYPE_NOT_NS_OR_AD")
        if not screens["conditional_type_pass"][index]:
            base.append("BASE_CONDITIONAL_TYPE_NOT_RW")
        if not screens["trading_status_pass"][index]:
            base.append("BASE_TRADING_STATUS_NOT_A")
        if not screens["primary_exchange_pass"][index]:
            base.append("BASE_PRIMARY_EXCHANGE_NOT_N_A_Q")
        if not screens["base_price_valid_pass"][index]:
            base.append("BASE_PRICE_INVALID")
        if not screens["base_market_cap_valid_pass"][index]:
            base.append("BASE_MARKET_CAP_INVALID")
        base_codes.append(rendered(base))

        long: list[str] = []
        if not screens["base_eligible"][index]:
            long.append("LONG_NOT_BASE_ELIGIBLE")
        if screens["base_price_valid_pass"][index] and not screens["long_price_pass"][index]:
            long.append("LONG_PRICE_BELOW_5_USD")
        if screens["base_market_cap_valid_pass"][index] and not screens["long_market_cap_pass"][index]:
            long.append("LONG_MARKET_CAP_BELOW_300M_USD")
        if not screens["long_valid_liquidity_sessions_pass"][index]:
            long.append("LONG_VALID_LIQUIDITY_SESSIONS_BELOW_40")
        if not screens["long_median_dollar_volume_pass"][index]:
            long.append("LONG_MEDIAN_DOLLAR_VOLUME_BELOW_5M_USD")
        if not screens["long_positive_volume_fraction_pass"][index]:
            long.append("LONG_POSITIVE_VOLUME_FRACTION_BELOW_0_90")
        if not screens["long_return_history_pass"][index]:
            long.append("LONG_VALID_RETURN_ROWS_BELOW_126")
        if screens["long_eligible_pre_dedup"][index] and not screens["issuer_line_selected"][index]:
            long.append("LONG_NOT_SELECTED_ISSUER_LINE")
        long_codes.append(rendered(long))

        short: list[str] = []
        if not screens["long_eligible"][index]:
            short.append("SHORT_NOT_LONG_ELIGIBLE")
        if screens["base_price_valid_pass"][index] and not screens["short_price_pass"][index]:
            short.append("SHORT_PRICE_BELOW_10_USD")
        if screens["base_market_cap_valid_pass"][index] and not screens["short_market_cap_pass"][index]:
            short.append("SHORT_MARKET_CAP_BELOW_1B_USD")
        if not screens["short_median_dollar_volume_pass"][index]:
            short.append("SHORT_MEDIAN_DOLLAR_VOLUME_BELOW_20M_USD")
        if not screens["short_positive_volume_fraction_pass"][index]:
            short.append("SHORT_POSITIVE_VOLUME_FRACTION_BELOW_0_95")
        short_codes.append(rendered(short))

    if any(
        (code == "PASS") != bool(screens["base_eligible"][index])
        for index, code in enumerate(base_codes)
    ):
        raise SignalUniverseError("Base reason codes do not reconstruct eligibility")
    if any(
        (code == "PASS") != bool(screens["long_eligible"][index])
        for index, code in enumerate(long_codes)
    ):
        raise SignalUniverseError("Long reason codes do not reconstruct eligibility")
    if any(
        (code == "PASS") != bool(screens["short_eligible"][index])
        for index, code in enumerate(short_codes)
    ):
        raise SignalUniverseError("Short reason codes do not reconstruct eligibility")
    return base_codes, long_codes, short_codes


def compute_universe(settings: Settings | None = None) -> UniverseResult:
    settings = settings or load_settings()
    calendar, signals, entries, exits = _signal_calendar(settings)
    arrays = _load_market_and_returns(settings, signals)
    rolling = _rolling_metrics(arrays, calendar, signals)
    identity = _identity_at_signal(settings, arrays)
    screens = _screen_and_deduplicate(
        identity=identity,
        arrays=arrays,
        rolling=rolling,
    )
    base_reason_codes, long_reason_codes, short_reason_codes = _reason_codes(screens)

    def date_from_ordinal(value: int) -> dt.date:
        return dt.date(1970, 1, 1) + dt.timedelta(days=int(value))

    row_count = len(arrays["signal_day"])
    signal_dates = [date_from_ordinal(value) for value in arrays["signal_day"]]
    median_decimal = [
        None
        if value < 0
        else (Decimal(int(value)) / Decimal(2_000_000)).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_EVEN
        )
        for value in rolling["median_twice"]
    ]
    positive_fraction = [
        None
        if valid == 0
        else (Decimal(int(positive)) / Decimal(int(valid))).quantize(
            Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN
        )
        for positive, valid in zip(
            rolling["positive_volume"], rolling["valid_liquidity"], strict=True
        )
    ]
    close_decimal = [
        None
        if not arrays["close_present"][index]
        else (Decimal(int(arrays["close_scaled"][index])) / Decimal(1_000_000)).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_EVEN
        )
        for index in range(row_count)
    ]
    cap_decimal = [
        None
        if not arrays["cap_present"][index]
        else (Decimal(int(arrays["cap_scaled"][index])) / Decimal(100)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_EVEN
        )
        for index in range(row_count)
    ]
    issuer_rank = [
        None if value < 0 else int(value) for value in screens["issuer_line_rank"]
    ]
    identity_start = [
        date_from_ordinal(value) for value in identity["identity_start"]
    ]
    identity_end = [
        date_from_ordinal(value) for value in identity["identity_end"]
    ]
    liquidity_start = [
        date_from_ordinal(value) for value in rolling["liquidity_start"]
    ]
    return_start = [date_from_ordinal(value) for value in rolling["return_start"]]

    columns: dict[str, Any] = {
        "signal_date": signal_dates,
        "permno": arrays["signal_permno"],
        "permco": identity["permco"],
        "observation_date": signal_dates,
        "public_availability_date": signal_dates,
        "availability_not_after_signal": np.ones(row_count, dtype=bool),
        "identity_interval_start_date": identity_start,
        "identity_interval_end_date": identity_end,
        "security_type": identity["security_type"].tolist(),
        "security_subtype": identity["security_subtype"].tolist(),
        "share_type": identity["share_type"].tolist(),
        "conditional_type": arrays["conditional"].tolist(),
        "trading_status_flag": identity["trading_status_flag"].tolist(),
        "primary_exchange": identity["primary_exchange"].tolist(),
        "us_incorporation_flag": identity["us_incorporation_flag"].tolist(),
        "close_usd_per_share": close_decimal,
        "market_cap_usd": cap_decimal,
        "screening_price_valid": arrays["screening_price_valid"],
        "market_cap_valid": arrays["market_cap_valid"],
        "liquidity_window_start_date": liquidity_start,
        "liquidity_window_end_date": signal_dates,
        "liquidity_window_calendar_session_count": rolling["liquidity_calendar_sessions"],
        "valid_liquidity_session_count": rolling["valid_liquidity"],
        "positive_volume_session_count": rolling["positive_volume"],
        "median_daily_dollar_volume_60_usd": median_decimal,
        "positive_volume_fraction_60": positive_fraction,
        "return_window_start_date": return_start,
        "return_window_end_date": signal_dates,
        "return_window_calendar_session_count": rolling["return_calendar_sessions"],
        "valid_daily_return_count_252": rolling["valid_returns"],
        "return_observations_after_signal_count": np.zeros(row_count, dtype=np.int16),
        "security_type_pass": screens["security_type_pass"],
        "security_subtype_pass": screens["security_subtype_pass"],
        "share_type_pass": screens["share_type_pass"],
        "conditional_type_pass": screens["conditional_type_pass"],
        "trading_status_pass": screens["trading_status_pass"],
        "primary_exchange_pass": screens["primary_exchange_pass"],
        "base_price_valid_pass": screens["base_price_valid_pass"],
        "base_market_cap_valid_pass": screens["base_market_cap_valid_pass"],
        "base_eligible": screens["base_eligible"],
        "long_price_pass": screens["long_price_pass"],
        "long_market_cap_pass": screens["long_market_cap_pass"],
        "long_valid_liquidity_sessions_pass": screens["long_valid_liquidity_sessions_pass"],
        "long_median_dollar_volume_pass": screens["long_median_dollar_volume_pass"],
        "long_positive_volume_fraction_pass": screens["long_positive_volume_fraction_pass"],
        "long_return_history_pass": screens["long_return_history_pass"],
        "long_eligible_pre_dedup": screens["long_eligible_pre_dedup"],
        "short_price_pass": screens["short_price_pass"],
        "short_market_cap_pass": screens["short_market_cap_pass"],
        "short_median_dollar_volume_pass": screens["short_median_dollar_volume_pass"],
        "short_positive_volume_fraction_pass": screens["short_positive_volume_fraction_pass"],
        "short_eligible_pre_dedup": screens["short_eligible_pre_dedup"],
        "issuer_long_eligible_line_count": screens["issuer_long_eligible_line_count"],
        "issuer_line_rank": issuer_rank,
        "issuer_line_selected": screens["issuer_line_selected"],
        "issuer_dedup_reason_code": screens["issuer_dedup_reason_code"].tolist(),
        "long_eligible": screens["long_eligible"],
        "short_eligible": screens["short_eligible"],
        "base_reason_codes": base_reason_codes,
        "long_reason_codes": long_reason_codes,
        "short_reason_codes": short_reason_codes,
        "future_outcome_filter_applied": np.zeros(row_count, dtype=bool),
    }
    schema = _json(SCHEMA_PATH)
    expected_names = [item["name"] for item in schema["tables"][0]["columns"]]
    if list(columns) != expected_names:
        missing = sorted(set(expected_names) - set(columns))
        extra = sorted(set(columns) - set(expected_names))
        raise SignalUniverseError(
            f"Universe output differs from frozen schema; missing={missing}, extra={extra}"
        )

    output_order = np.lexsort((arrays["signal_permno"], arrays["signal_day"]))
    for name, values in tuple(columns.items()):
        if isinstance(values, np.ndarray):
            columns[name] = values[output_order]
        else:
            columns[name] = [values[int(index)] for index in output_order]

    long_counts = Counter(
        arrays["signal_day"][screens["long_eligible"]].tolist()
    )
    short_counts = Counter(
        arrays["signal_day"][screens["short_eligible"]].tolist()
    )
    long_by_signal = [long_counts.get(int(signal), 0) for signal in signals]
    short_by_signal = [short_counts.get(int(signal), 0) for signal in signals]
    summary = {
        "signal_dates": len(signals),
        "first_signal_date": date_from_ordinal(signals[0]).isoformat(),
        "last_signal_date": date_from_ordinal(signals[-1]).isoformat(),
        "first_entry_date": date_from_ordinal(entries[0]).isoformat(),
        "last_exit_date": date_from_ordinal(exits[-1]).isoformat(),
        "rows": row_count,
        "identity_matched_rows": row_count,
        "base_screen_rows": int(screens["base_eligible"].sum()),
        "long_pre_dedup_rows": int(screens["long_eligible_pre_dedup"].sum()),
        "issuer_primary_rows": int(screens["issuer_line_selected"].sum()),
        "long_eligible_rows": int(screens["long_eligible"].sum()),
        "short_pre_dedup_rows": int(screens["short_eligible_pre_dedup"].sum()),
        "short_eligible_rows": int(screens["short_eligible"].sum()),
        "foreign_long_eligible_rows": int(
            np.sum(
                screens["long_eligible"]
                & (identity["us_incorporation_flag"] != "Y")
            )
        ),
        "adr_long_eligible_rows": int(
            np.sum(screens["long_eligible"] & (identity["share_type"] == "AD"))
        ),
        "signal_long_count_min": min(long_by_signal, default=0),
        "signal_long_count_max": max(long_by_signal, default=0),
        "signal_short_count_min": min(short_by_signal, default=0),
        "signal_short_count_max": max(short_by_signal, default=0),
        "point_in_time_violations": 0,
        "future_outcome_filter_applied": False,
    }
    return UniverseResult(columns=columns, summary=summary)
