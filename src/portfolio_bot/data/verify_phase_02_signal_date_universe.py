"""Independent verifier for the development signal-date universe.

This module deliberately does not import the primary universe computation or
publication modules.  It authenticates their frozen inputs and published
receipts, then reconstructs all row-level eligibility and issuer-selection
semantics from the sealed Parquet output.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data.database_location import resolve_recorded_database_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/signal_date_universe_policy_v1.toml"
SCHEMA_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/signal_date_universe_schema_v1.json"
PATHS_PATH = PROJECT_ROOT / "configs/paths.toml"
PRIMARY_POINTER_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_SIGNAL_DATE_UNIVERSE_PRIMARY.json"
VERIFIED_POINTER_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_SIGNAL_DATE_UNIVERSE_VERIFIED.json"
VERIFICATION_PARENT = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/signal_date_universe_verifications"
PRIMARY_STATUS = "signal_date_universe_primary_passed_pending_independent_verification"
PASS_STATUS = "verified_passed"
FAIL_STATUS = "verification_failed"
EXPECTED_ROWS = 1_263_048


class SignalUniverseVerificationError(RuntimeError):
    """Raised when verification cannot be executed safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise SignalUniverseVerificationError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(_canonical_json(value))
    os.replace(temporary, path)


@dataclass
class Checks:
    rows: list[dict[str, Any]]

    def add(
        self,
        check_id: str,
        *,
        category: str,
        expected: Any,
        actual: Any,
        evidence: str,
    ) -> None:
        self.rows.append(
            {
                "check_id": check_id,
                "category": category,
                "status": "PASS" if expected == actual else "FAIL",
                "expected": expected,
                "actual": actual,
                "evidence": evidence,
            }
        )


@dataclass(frozen=True)
class Context:
    database_root: Path
    pointer: dict[str, Any]
    manifest: dict[str, Any]
    policy: dict[str, Any]
    schema_document: dict[str, Any]
    artifact_dir: Path
    interim_dir: Path
    quality_dir: Path
    output_path: Path
    calendar_path: Path


@dataclass
class ScanResult:
    rows: int
    summary: dict[str, Any]
    violations: dict[str, int]
    signal_rows: list[dict[str, Any]]
    group_rows: list[dict[str, Any]]
    reason_rows: list[dict[str, Any]]


def _arrow_type(value: str) -> pa.DataType:
    fixed = {
        "bool": pa.bool_(),
        "date32": pa.date32(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "utf8": pa.string(),
    }
    if value in fixed:
        return fixed[value]
    if value.startswith("decimal128(") and value.endswith(")"):
        precision, scale = map(int, value[11:-1].split(","))
        return pa.decimal128(precision, scale)
    raise SignalUniverseVerificationError(f"Unsupported schema type {value!r}")


def expected_arrow_schema(document: Mapping[str, Any]) -> pa.Schema:
    table = document["tables"][0]
    fields = []
    for item in table["columns"]:
        metadata = {
            key.encode(): ("null" if item.get(key) is None else str(item[key])).encode()
            for key in ("role", "source_column", "unit")
        }
        fields.append(
            pa.field(
                item["name"],
                _arrow_type(item["type"]),
                nullable=bool(item["nullable"]),
                metadata=metadata,
            )
        )
    table_metadata: dict[bytes, bytes] = {}
    for key in (
        "expected_rows",
        "expected_rows_status",
        "grain",
        "primary_key",
        "primary_source",
        "role",
        "sort_key",
        "table",
    ):
        value = table[key]
        encoded = (
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            if isinstance(value, (list, dict, bool, int)) or value is None
            else str(value)
        )
        table_metadata[key.encode()] = encoded.encode()
    return pa.schema(fields, metadata=table_metadata)


def _database_root() -> Path:
    with PATHS_PATH.open("rb") as handle:
        config = tomllib.load(handle)
    return Path(os.environ.get("PORTFOLIO_DATABASE_ROOT", config["database"]["root"]))


def _resolve_calendar_path(database_root: Path, pointer: Mapping[str, Any]) -> Path:
    artifact = PROJECT_ROOT / str(pointer["primary_artifact_path"])
    manifest = _json(artifact / "manifest.json")
    if _sha256(artifact / "manifest.json") != pointer["primary_manifest_sha256"]:
        raise SignalUniverseVerificationError("Calendar primary manifest hash differs")
    matches = [
        item for item in manifest["outputs"]
        if str(item["path"]).endswith("/trading_calendar.parquet")
    ]
    if len(matches) != 1:
        raise SignalUniverseVerificationError("Calendar output is not unique")
    recorded = Path(matches[0]["path"])
    path = (
        resolve_recorded_database_path(recorded, current_database_root=database_root)
        if recorded.is_absolute()
        else database_root / recorded
    )
    if path.stat().st_size != matches[0]["size_bytes"] or _sha256(path) != matches[0]["sha256"]:
        raise SignalUniverseVerificationError("Calendar output receipt differs")
    return path


def load_context() -> Context:
    database_root = _database_root()
    pointer = _json(PRIMARY_POINTER_PATH)
    artifact_dir = PROJECT_ROOT / str(pointer["artifact_path"])
    manifest_path = artifact_dir / "manifest.json"
    manifest = _json(manifest_path)
    with POLICY_PATH.open("rb") as handle:
        policy = tomllib.load(handle)
    schema_document = _json(SCHEMA_PATH)
    calendar_pointer_path = PROJECT_ROOT / policy["upstream"]["calendar"]["verified_pointer"]
    calendar_pointer = _json(calendar_pointer_path)
    return Context(
        database_root=database_root,
        pointer=pointer,
        manifest=manifest,
        policy=policy,
        schema_document=schema_document,
        artifact_dir=artifact_dir,
        interim_dir=database_root / str(pointer["interim_path"]),
        quality_dir=database_root / str(pointer["quality_path"]),
        output_path=database_root / str(pointer["interim_path"]) / "crsp_signal_date_universe.parquet",
        calendar_path=_resolve_calendar_path(database_root, calendar_pointer),
    )


def _bool(table: pa.Table, name: str) -> np.ndarray:
    return table[name].combine_chunks().to_numpy(zero_copy_only=False).astype(bool)


def _ints(table: pa.Table, name: str) -> np.ndarray:
    return table[name].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)


def _strings(table: pa.Table, name: str) -> np.ndarray:
    return table[name].combine_chunks().to_numpy(zero_copy_only=False).astype(str)


def _dates(table: pa.Table, name: str) -> np.ndarray:
    return (
        table[name].combine_chunks().to_numpy(zero_copy_only=False)
        .astype("datetime64[D]").astype(np.int64)
    )


def _scaled_decimal(table: pa.Table, name: str, scale: int) -> tuple[np.ndarray, np.ndarray]:
    column = table[name].combine_chunks()
    valid = pc.is_valid(column).to_numpy(zero_copy_only=False).astype(bool)
    narrowed = pc.cast(column, pa.decimal128(20, scale))
    multiplier = pa.scalar(Decimal(10**scale), type=pa.decimal128(scale + 1, 0))
    scaled = pc.cast(pc.multiply(narrowed, multiplier), pa.int64())
    values = pc.fill_null(scaled, 0).to_numpy(zero_copy_only=False).astype(np.int64)
    return values, valid


def _round_fraction_scaled(numerator: np.ndarray, denominator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = denominator > 0
    result = np.zeros(len(numerator), dtype=np.int64)
    product = numerator[valid].astype(np.int64) * np.int64(1_000_000_000_000)
    divisor = denominator[valid].astype(np.int64)
    quotient, remainder = np.divmod(product, divisor)
    increment = (remainder * 2 > divisor) | (
        (remainder * 2 == divisor) & (quotient % 2 == 1)
    )
    result[valid] = quotient + increment.astype(np.int64)
    return result, valid


def expected_reason_codes(values: Mapping[str, bool]) -> tuple[str, str, str]:
    def render(items: Iterable[str]) -> str:
        result = list(items)
        return "|".join(result) if result else "PASS"

    base = render(
        code for flag, code in (
            ("security_type_pass", "BASE_SECURITY_TYPE_NOT_EQTY"),
            ("security_subtype_pass", "BASE_SECURITY_SUBTYPE_NOT_COM"),
            ("share_type_pass", "BASE_SHARE_TYPE_NOT_NS_OR_AD"),
            ("conditional_type_pass", "BASE_CONDITIONAL_TYPE_NOT_RW"),
            ("trading_status_pass", "BASE_TRADING_STATUS_NOT_A"),
            ("primary_exchange_pass", "BASE_PRIMARY_EXCHANGE_NOT_N_A_Q"),
            ("base_price_valid_pass", "BASE_PRICE_INVALID"),
            ("base_market_cap_valid_pass", "BASE_MARKET_CAP_INVALID"),
        ) if not values[flag]
    )
    long_items: list[str] = []
    if not values["base_eligible"]:
        long_items.append("LONG_NOT_BASE_ELIGIBLE")
    if values["base_price_valid_pass"] and not values["long_price_pass"]:
        long_items.append("LONG_PRICE_BELOW_5_USD")
    if values["base_market_cap_valid_pass"] and not values["long_market_cap_pass"]:
        long_items.append("LONG_MARKET_CAP_BELOW_300M_USD")
    for flag, code in (
        ("long_valid_liquidity_sessions_pass", "LONG_VALID_LIQUIDITY_SESSIONS_BELOW_40"),
        ("long_median_dollar_volume_pass", "LONG_MEDIAN_DOLLAR_VOLUME_BELOW_5M_USD"),
        ("long_positive_volume_fraction_pass", "LONG_POSITIVE_VOLUME_FRACTION_BELOW_0_90"),
        ("long_return_history_pass", "LONG_VALID_RETURN_ROWS_BELOW_126"),
    ):
        if not values[flag]:
            long_items.append(code)
    if values["long_eligible_pre_dedup"] and not values["issuer_line_selected"]:
        long_items.append("LONG_NOT_SELECTED_ISSUER_LINE")

    short_items: list[str] = []
    if not values["long_eligible"]:
        short_items.append("SHORT_NOT_LONG_ELIGIBLE")
    if values["base_price_valid_pass"] and not values["short_price_pass"]:
        short_items.append("SHORT_PRICE_BELOW_10_USD")
    if values["base_market_cap_valid_pass"] and not values["short_market_cap_pass"]:
        short_items.append("SHORT_MARKET_CAP_BELOW_1B_USD")
    for flag, code in (
        ("short_median_dollar_volume_pass", "SHORT_MEDIAN_DOLLAR_VOLUME_BELOW_20M_USD"),
        ("short_positive_volume_fraction_pass", "SHORT_POSITIVE_VOLUME_FRACTION_BELOW_0_95"),
    ):
        if not values[flag]:
            short_items.append(code)
    return base, render(long_items), render(short_items)


def _calendar_expectations(context: Context) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    table = pq.read_table(context.calendar_path)
    dates = _dates(table, "calendar_date")
    month_end = _bool(table, "is_observed_month_end")
    positions = np.flatnonzero(
        month_end
        & (dates >= np.datetime64("2000-01-01", "D").astype(np.int64))
        & (dates <= np.datetime64("2014-10-31", "D").astype(np.int64))
    )
    expected: dict[int, tuple[int, int, int, int]] = {}
    for position in positions:
        expected[int(dates[position])] = (
            int(dates[max(0, position - 59)]),
            min(60, int(position) + 1),
            int(dates[max(0, position - 251)]),
            min(252, int(position) + 1),
        )
    return dates[positions], expected


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def scan_universe(context: Context) -> ScanResult:
    parquet = pq.ParquetFile(context.output_path)
    expected_signals, calendar_expected = _calendar_expectations(context)
    violations: Counter[str] = Counter()
    by_signal: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    by_group: dict[tuple[int, str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    reason_counts = {name: Counter() for name in ("base", "long", "short")}
    candidates: dict[tuple[int, int], list[tuple[Any, ...]]] = defaultdict(list)
    prior_key: tuple[int, int] | None = None
    seen_signals: set[int] = set()
    rows = base_total = long_pre_total = short_pre_total = 0
    selected_total = long_total = short_total = foreign_long = adr_long = 0
    bool_names = [
        "security_type_pass", "security_subtype_pass", "share_type_pass",
        "conditional_type_pass", "trading_status_pass", "primary_exchange_pass",
        "base_price_valid_pass", "base_market_cap_valid_pass", "base_eligible",
        "long_price_pass", "long_market_cap_pass", "long_valid_liquidity_sessions_pass",
        "long_median_dollar_volume_pass", "long_positive_volume_fraction_pass",
        "long_return_history_pass", "long_eligible_pre_dedup", "short_price_pass",
        "short_market_cap_pass", "short_median_dollar_volume_pass",
        "short_positive_volume_fraction_pass", "short_eligible_pre_dedup",
        "issuer_line_selected", "long_eligible", "short_eligible",
    ]

    for batch in parquet.iter_batches(batch_size=131_072):
        table = pa.Table.from_batches([batch])
        size = table.num_rows
        rows += size
        violations["required_nulls"] += sum(
            table[field.name].null_count
            for field in parquet.schema_arrow
            if not field.nullable
        )
        signal = _dates(table, "signal_date")
        permno = _ints(table, "permno")
        permco = _ints(table, "permco")
        if size:
            keys = list(zip(signal.tolist(), permno.tolist(), strict=True))
            if prior_key is not None and keys[0] <= prior_key:
                violations["pk_order_or_duplicate"] += 1
            violations["pk_order_or_duplicate"] += sum(
                keys[index] <= keys[index - 1] for index in range(1, size)
            )
            prior_key = keys[-1]
        observation = _dates(table, "observation_date")
        availability = _dates(table, "public_availability_date")
        identity_start = _dates(table, "identity_interval_start_date")
        identity_end = _dates(table, "identity_interval_end_date")
        liquidity_start = _dates(table, "liquidity_window_start_date")
        liquidity_end = _dates(table, "liquidity_window_end_date")
        return_start = _dates(table, "return_window_start_date")
        return_end = _dates(table, "return_window_end_date")
        liquidity_calendar_count = _ints(table, "liquidity_window_calendar_session_count")
        valid_liquidity = _ints(table, "valid_liquidity_session_count")
        positive_volume = _ints(table, "positive_volume_session_count")
        return_calendar_count = _ints(table, "return_window_calendar_session_count")
        valid_returns = _ints(table, "valid_daily_return_count_252")
        after_signal = _ints(table, "return_observations_after_signal_count")
        close, close_present = _scaled_decimal(table, "close_usd_per_share", 6)
        cap, cap_present = _scaled_decimal(table, "market_cap_usd", 2)
        median, median_present = _scaled_decimal(table, "median_daily_dollar_volume_60_usd", 6)
        fraction, fraction_present = _scaled_decimal(table, "positive_volume_fraction_60", 12)
        expected_fraction, expected_fraction_present = _round_fraction_scaled(
            positive_volume, valid_liquidity
        )
        actual_bool = {name: _bool(table, name) for name in bool_names}
        price_flag = _bool(table, "screening_price_valid")
        cap_flag = _bool(table, "market_cap_valid")
        strings = {
            name: _strings(table, name)
            for name in (
                "security_type", "security_subtype", "share_type", "conditional_type",
                "trading_status_flag", "primary_exchange", "us_incorporation_flag",
                "issuer_dedup_reason_code", "base_reason_codes", "long_reason_codes",
                "short_reason_codes",
            )
        }
        expected_bool = {
            "security_type_pass": strings["security_type"] == "EQTY",
            "security_subtype_pass": strings["security_subtype"] == "COM",
            "share_type_pass": np.isin(strings["share_type"], ["NS", "AD"]),
            "conditional_type_pass": strings["conditional_type"] == "RW",
            "trading_status_pass": strings["trading_status_flag"] == "A",
            "primary_exchange_pass": np.isin(strings["primary_exchange"], ["N", "A", "Q"]),
            "base_price_valid_pass": price_flag & close_present & (close > 0),
            "base_market_cap_valid_pass": cap_flag & cap_present & (cap > 0),
        }
        expected_bool["base_eligible"] = np.logical_and.reduce(list(expected_bool.values()))
        expected_bool.update(
            {
                "long_price_pass": expected_bool["base_price_valid_pass"] & (close >= 5_000_000),
                "long_market_cap_pass": expected_bool["base_market_cap_valid_pass"] & (cap >= 30_000_000_000),
                "long_valid_liquidity_sessions_pass": valid_liquidity >= 40,
                "long_median_dollar_volume_pass": median_present & (median >= 5_000_000_000_000),
                "long_positive_volume_fraction_pass": (valid_liquidity > 0) & (positive_volume * 100 >= valid_liquidity * 90),
                "long_return_history_pass": valid_returns >= 126,
                "short_price_pass": expected_bool["base_price_valid_pass"] & (close >= 10_000_000),
                "short_market_cap_pass": expected_bool["base_market_cap_valid_pass"] & (cap >= 100_000_000_000),
                "short_median_dollar_volume_pass": median_present & (median >= 20_000_000_000_000),
                "short_positive_volume_fraction_pass": (valid_liquidity > 0) & (positive_volume * 100 >= valid_liquidity * 95),
            }
        )
        expected_bool["long_eligible_pre_dedup"] = np.logical_and.reduce(
            [expected_bool[name] for name in (
                "base_eligible", "long_price_pass", "long_market_cap_pass",
                "long_valid_liquidity_sessions_pass", "long_median_dollar_volume_pass",
                "long_positive_volume_fraction_pass", "long_return_history_pass",
            )]
        )
        expected_bool["short_eligible_pre_dedup"] = np.logical_and.reduce(
            [expected_bool[name] for name in (
                "long_eligible_pre_dedup", "short_price_pass", "short_market_cap_pass",
                "short_median_dollar_volume_pass", "short_positive_volume_fraction_pass",
            )]
        )
        expected_bool["issuer_line_selected"] = actual_bool["issuer_line_selected"]
        expected_bool["long_eligible"] = expected_bool["long_eligible_pre_dedup"] & actual_bool["issuer_line_selected"]
        expected_bool["short_eligible"] = expected_bool["short_eligible_pre_dedup"] & actual_bool["issuer_line_selected"]
        for name, expected in expected_bool.items():
            violations[f"boolean_{name}"] += int(np.sum(actual_bool[name] != expected))

        violations["observation_date"] += int(np.sum(observation != signal))
        violations["public_availability_date"] += int(np.sum(availability != signal))
        violations["availability_flag"] += int(np.sum(~_bool(table, "availability_not_after_signal")))
        violations["identity_interval"] += int(np.sum((identity_start > signal) | (identity_end < signal)))
        violations["window_end"] += int(np.sum((liquidity_end != signal) | (return_end != signal)))
        violations["return_after_signal"] += int(np.sum(after_signal != 0))
        violations["future_filter"] += int(np.sum(_bool(table, "future_outcome_filter_applied")))
        violations["window_count_shape"] += int(np.sum(
            (valid_liquidity < 0) | (valid_liquidity > liquidity_calendar_count)
            | (positive_volume < 0) | (positive_volume > valid_liquidity)
            | (valid_returns < 0) | (valid_returns > return_calendar_count)
        ))
        violations["median_null_shape"] += int(np.sum(median_present != (valid_liquidity > 0)))
        violations["fraction_value_or_null"] += int(np.sum(
            (fraction_present != expected_fraction_present)
            | (fraction_present & (fraction != expected_fraction))
        ))
        for index, day in enumerate(signal):
            expected_window = calendar_expected.get(int(day))
            if expected_window is None:
                violations["signal_not_calendar_month_end"] += 1
            elif expected_window != (
                int(liquidity_start[index]), int(liquidity_calendar_count[index]),
                int(return_start[index]), int(return_calendar_count[index]),
            ):
                violations["calendar_window"] += 1
        seen_signals.update(int(value) for value in signal)

        ranks = table["issuer_line_rank"].combine_chunks().to_pylist()
        line_count = _ints(table, "issuer_long_eligible_line_count")
        for index in range(size):
            values = {name: bool(actual_bool[name][index]) for name in bool_names}
            expected_reasons = expected_reason_codes(values)
            actual_reasons = (
                strings["base_reason_codes"][index],
                strings["long_reason_codes"][index],
                strings["short_reason_codes"][index],
            )
            violations["reason_code_exact"] += expected_reasons != actual_reasons
            for family, rendered in zip(("base", "long", "short"), actual_reasons, strict=True):
                reason_counts[family].update(str(rendered).split("|"))
            if values["long_eligible_pre_dedup"]:
                if ranks[index] is None or not median_present[index] or not cap_present[index]:
                    violations["candidate_required_value"] += 1
                else:
                    candidates[(int(signal[index]), int(permco[index]))].append(
                        (
                            int(permno[index]), int(median[index]), int(cap[index]),
                            int(line_count[index]), int(ranks[index]),
                            values["issuer_line_selected"], strings["issuer_dedup_reason_code"][index],
                        )
                    )
            else:
                violations["noncandidate_dedup_shape"] += int(
                    line_count[index] != 0 or ranks[index] is not None
                    or values["issuer_line_selected"]
                    or strings["issuer_dedup_reason_code"][index]
                    != "NOT_APPLICABLE_NOT_LONG_ELIGIBLE_PRE_DEDUP"
                )
            target = by_signal[int(signal[index])]
            group = by_group[(
                dt.date(1970, 1, 1).year
                + int(np.datetime64(int(signal[index]), "D").astype("datetime64[Y]").astype(int)),
                strings["primary_exchange"][index], strings["share_type"][index],
            )]
            increments = (1, int(values["base_eligible"]), int(values["long_eligible"]), int(values["short_eligible"]))
            for position, value in enumerate(increments):
                target[position] += value
                group[position] += value
        base_total += int(actual_bool["base_eligible"].sum())
        long_pre_total += int(actual_bool["long_eligible_pre_dedup"].sum())
        short_pre_total += int(actual_bool["short_eligible_pre_dedup"].sum())
        selected_total += int(actual_bool["issuer_line_selected"].sum())
        long_total += int(actual_bool["long_eligible"].sum())
        short_total += int(actual_bool["short_eligible"].sum())
        foreign_long += int(np.sum(actual_bool["long_eligible"] & (strings["us_incorporation_flag"] != "Y")))
        adr_long += int(np.sum(actual_bool["long_eligible"] & (strings["share_type"] == "AD")))

    for group in candidates.values():
        ordered = sorted(group, key=lambda row: (-row[1], -row[2], row[0]))
        size = len(ordered)
        for position, row in enumerate(ordered, start=1):
            permno, median, cap, actual_count, actual_rank, selected, reason = row
            violations["issuer_count_or_rank"] += int(actual_count != size or actual_rank != position)
            violations["issuer_selection"] += int(selected != (position == 1))
            if size == 1:
                expected_reason = "SOLE_LONG_ELIGIBLE_LINE"
            elif position == 1:
                runner = ordered[1]
                expected_reason = (
                    "SELECTED_HIGHEST_MEDIAN_DOLLAR_VOLUME" if median > runner[1]
                    else "SELECTED_MARKET_CAP_TIEBREAK" if cap > runner[2]
                    else "SELECTED_LOWEST_PERMNO_TIEBREAK"
                )
            else:
                winner = ordered[0]
                expected_reason = (
                    "NOT_SELECTED_LOWER_MEDIAN_DOLLAR_VOLUME" if median < winner[1]
                    else "NOT_SELECTED_LOWER_MARKET_CAP_TIEBREAK" if cap < winner[2]
                    else "NOT_SELECTED_HIGHER_PERMNO_TIEBREAK"
                )
            violations["issuer_reason"] += reason != expected_reason

    signal_rows = [
        {
            "signal_date": (dt.date(1970, 1, 1) + dt.timedelta(days=day)).isoformat(),
            "rows": values[0], "base_eligible": values[1],
            "long_eligible": values[2], "short_eligible": values[3],
        }
        for day, values in sorted(by_signal.items())
    ]
    group_rows = [
        {
            "signal_year": key[0], "primary_exchange": key[1], "share_type": key[2],
            "rows": values[0], "base_eligible": values[1],
            "long_eligible": values[2], "short_eligible": values[3],
        }
        for key, values in sorted(by_group.items())
    ]
    reason_rows = [
        {"family": family, "reason_code": reason, "row_count": count}
        for family in ("base", "long", "short")
        for reason, count in sorted(reason_counts[family].items())
    ]
    long_counts = [by_signal[int(day)][2] for day in expected_signals]
    short_counts = [by_signal[int(day)][3] for day in expected_signals]
    calendar_dates = pq.read_table(context.calendar_path, columns=["calendar_date", "is_observed_month_end"])
    all_dates = _dates(calendar_dates, "calendar_date")
    all_month_end = np.flatnonzero(_bool(calendar_dates, "is_observed_month_end"))
    first_signal_position = int(np.searchsorted(all_dates, expected_signals[0]))
    last_signal_position = int(np.searchsorted(all_dates, expected_signals[-1]))
    next_month_end = all_month_end[np.searchsorted(all_month_end, last_signal_position, side="right")]
    summary = {
        "signal_dates": len(expected_signals),
        "first_signal_date": (dt.date(1970, 1, 1) + dt.timedelta(days=int(expected_signals[0]))).isoformat(),
        "last_signal_date": (dt.date(1970, 1, 1) + dt.timedelta(days=int(expected_signals[-1]))).isoformat(),
        "first_entry_date": (dt.date(1970, 1, 1) + dt.timedelta(days=int(all_dates[first_signal_position + 1]))).isoformat(),
        "last_exit_date": (dt.date(1970, 1, 1) + dt.timedelta(days=int(all_dates[next_month_end + 1]))).isoformat(),
        "rows": rows, "identity_matched_rows": rows, "base_screen_rows": base_total,
        "long_pre_dedup_rows": long_pre_total, "issuer_primary_rows": selected_total,
        "long_eligible_rows": long_total, "short_pre_dedup_rows": short_pre_total,
        "short_eligible_rows": short_total, "foreign_long_eligible_rows": foreign_long,
        "adr_long_eligible_rows": adr_long, "signal_long_count_min": min(long_counts),
        "signal_long_count_max": max(long_counts), "signal_short_count_min": min(short_counts),
        "signal_short_count_max": max(short_counts), "point_in_time_violations": 0,
        "future_outcome_filter_applied": False,
    }
    violations["signal_set"] += int(set(expected_signals.tolist()) != seen_signals)
    return ScanResult(rows, summary, dict(violations), signal_rows, group_rows, reason_rows)


def _authenticate(context: Context, checks: Checks) -> None:
    pointer = context.pointer
    manifest = context.manifest
    manifest_path = context.artifact_dir / "manifest.json"
    checks.add(
        "PRIMARY_POINTER_AND_MANIFEST",
        category="authentication",
        expected={"status": PRIMARY_STATUS, "authorized": False, "manifest": True, "identity": True},
        actual={
            "status": pointer.get("status"),
            "authorized": pointer.get("signal_date_universe_authorized"),
            "manifest": _sha256(manifest_path) == pointer.get("manifest_sha256"),
            "identity": pointer.get("build_id") == manifest.get("build_id")
            and pointer.get("contract_id") == manifest.get("contract_id"),
        },
        evidence="primary pointer and complete manifest bytes",
    )
    local_bindings = {
        "policy_sha256": _sha256(POLICY_PATH),
        "schema_sha256": _sha256(SCHEMA_PATH),
        "typed_verified_pointer_sha256": _sha256(PROJECT_ROOT / context.policy["upstream"]["typed_backbone"]["verified_pointer"]),
        "return_verified_pointer_sha256": _sha256(PROJECT_ROOT / context.policy["upstream"]["return_reconciliation"]["verified_pointer"]),
        "calendar_verified_pointer_sha256": _sha256(PROJECT_ROOT / context.policy["upstream"]["calendar"]["verified_pointer"]),
    }
    checks.add(
        "POLICY_SCHEMA_AND_UPSTREAM_BINDINGS",
        category="authentication",
        expected={key: manifest["bindings"][key] for key in local_bindings},
        actual=local_bindings,
        evidence="current frozen files hashed independently",
    )
    upstream_actual: dict[str, Any] = {}
    upstream_expected: dict[str, Any] = {}
    for name, spec in context.policy["upstream"].items():
        pointer_value = _json(PROJECT_ROOT / spec["verified_pointer"])
        authorization = spec.get("required_authorization")
        upstream_actual[name] = {
            "sha256": _sha256(PROJECT_ROOT / spec["verified_pointer"]),
            "status": pointer_value.get("status"),
            "build_id": pointer_value.get("build_id"),
            "authorization": True if authorization is None else bool(pointer_value.get("authorizations", {}).get(authorization)),
        }
        upstream_expected[name] = {
            "sha256": spec["verified_pointer_sha256"],
            "status": spec["required_status"],
            "build_id": spec["required_build_id"],
            "authorization": True,
        }
    checks.add(
        "UPSTREAM_VERIFIED_POINTER_STATES",
        category="authentication",
        expected=upstream_expected,
        actual=upstream_actual,
        evidence="three frozen independently verified upstream pointers",
    )
    output = manifest["output"]
    quality_actual = {
        name: {"sha256": _sha256(context.quality_dir / name), "size_bytes": (context.quality_dir / name).stat().st_size}
        for name in manifest["quality_receipts"]
    }
    checks.add(
        "OUTPUT_AND_QUALITY_FILE_RECEIPTS",
        category="authentication",
        expected={
            "output": {"sha256": output["sha256"], "size_bytes": output["size_bytes"], "rows": output["rows"]},
            "quality": manifest["quality_receipts"],
        },
        actual={
            "output": {"sha256": _sha256(context.output_path), "size_bytes": context.output_path.stat().st_size, "rows": pq.ParquetFile(context.output_path).metadata.num_rows},
            "quality": quality_actual,
        },
        evidence="full output hash plus every aggregate quality receipt",
    )


def verify(*, publish: bool = True) -> dict[str, Any]:
    context = load_context()
    checks = Checks([])
    _authenticate(context, checks)
    parquet = pq.ParquetFile(context.output_path)
    checks.add(
        "ARROW_SCHEMA_EXACT",
        category="schema",
        expected=True,
        actual=parquet.schema_arrow.equals(expected_arrow_schema(context.schema_document), check_metadata=True),
        evidence="complete Arrow field order, types, nullability, and metadata",
    )
    scan = scan_universe(context)
    checks.add(
        "ROW_COUNT_PRIMARY_KEY_AND_SEMANTICS",
        category="full_scan",
        expected={"rows": EXPECTED_ROWS, "violations": {}},
        actual={"rows": scan.rows, "violations": {key: value for key, value in sorted(scan.violations.items()) if value}},
        evidence="all 1,263,048 rows and all eligibility columns",
    )
    checks.add(
        "PRIMARY_SUMMARY_RECONSTRUCTED",
        category="aggregate",
        expected=context.manifest["summary"],
        actual=scan.summary,
        evidence="full-scan independent aggregates plus verified calendar",
    )
    receipt_matches = {
        "counts_by_signal.csv": _csv_bytes(scan.signal_rows) == (context.quality_dir / "counts_by_signal.csv").read_bytes(),
        "counts_by_year_exchange_share_type.csv": _csv_bytes(scan.group_rows) == (context.quality_dir / "counts_by_year_exchange_share_type.csv").read_bytes(),
        "reason_counts.csv": _csv_bytes(scan.reason_rows) == (context.quality_dir / "reason_counts.csv").read_bytes(),
    }
    checks.add(
        "AGGREGATE_QUALITY_RECEIPTS_RECONSTRUCTED",
        category="aggregate",
        expected={name: True for name in receipt_matches},
        actual=receipt_matches,
        evidence="independently rendered aggregate-only CSV bytes",
    )
    expected_checks = [
        {"check": "row_conservation", "status": "pass", "detail": scan.rows},
        {"check": "unique_signal_date_permno", "status": "pass", "detail": 0},
        {"check": "identity_join_cardinality_one", "status": "pass", "detail": scan.rows},
        {"check": "availability_after_signal", "status": "pass", "detail": 0},
        {"check": "future_outcome_filter_applied", "status": "pass", "detail": 0},
        {"check": "short_not_long_violations", "status": "pass", "detail": 0},
        {"check": "selected_issuer_duplicates", "status": "pass", "detail": 0},
    ]
    checks.add(
        "PRIMARY_CHECKS_AND_SUCCESS_RECEIPTS",
        category="aggregate",
        expected={"checks": True, "success_summary": True},
        actual={
            "checks": _csv_bytes(expected_checks) == (context.quality_dir / "checks.csv").read_bytes(),
            "success_summary": (context.interim_dir / "_SUCCESS.json").read_bytes() == (context.quality_dir / "summary.json").read_bytes(),
        },
        evidence="primary aggregate checks and duplicated success receipt",
    )
    ordered = sorted(checks.rows, key=lambda row: row["check_id"])
    fail_count = sum(row["status"] == "FAIL" for row in ordered)
    status = PASS_STATUS if fail_count == 0 else FAIL_STATUS
    implementation_sha = _sha256(Path(__file__))
    verification_seed = {
        "build_id": context.pointer["build_id"],
        "primary_manifest_sha256": context.pointer["manifest_sha256"],
        "verifier_implementation_sha256": implementation_sha,
    }
    verification_id = "verify-" + hashlib.sha256(_canonical_json(verification_seed)).hexdigest()[:16]
    authorizations = {
        "signal_date_universe_authorized": status == PASS_STATUS,
        "features_labels_models_portfolios_pnl_performance_or_backtests_authorized": False,
        "validation_or_final_test_access_authorized": False,
    }
    report = {
        "schema_version": 1,
        "status": status,
        "verification_id": verification_id,
        "build_id": context.pointer["build_id"],
        "contract_id": context.pointer["contract_id"],
        "primary_manifest_sha256": context.pointer["manifest_sha256"],
        "verifier_implementation_sha256": implementation_sha,
        "checks": ordered,
        "summary": {"check_count": len(ordered), "pass_count": len(ordered) - fail_count, "fail_count": fail_count, "rows_scanned": scan.rows},
        "independence": {"primary_compute_imported": False, "primary_publisher_imported": False, "full_parquet_rows_scanned": scan.rows},
        "authorizations": authorizations,
    }
    if publish and status == PASS_STATUS:
        _publish_verified(report)
    return {"status": status, "verification_id": verification_id, "check_count": len(ordered), "fail_count": fail_count, "report": report}


def _differences_bytes(report: Mapping[str, Any]) -> bytes:
    rows = [row for row in report["checks"] if row["status"] == "FAIL"]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["check_id", "category", "expected_json", "actual_json"])
    for row in rows:
        writer.writerow([
            row["check_id"], row["category"],
            json.dumps(row["expected"], sort_keys=True, separators=(",", ":")),
            json.dumps(row["actual"], sort_keys=True, separators=(",", ":")),
        ])
    return stream.getvalue().encode()


def _publish_verified(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("status") != PASS_STATUS or report.get("summary", {}).get("fail_count") != 0:
        raise SignalUniverseVerificationError("Only a zero-difference report may publish")
    verification_id = str(report["verification_id"])
    final = VERIFICATION_PARENT / verification_id
    stage = VERIFICATION_PARENT / (".staging-" + verification_id)
    if final.exists() or stage.exists():
        raise SignalUniverseVerificationError("Verification evidence path already exists")
    stage.mkdir(parents=True)
    try:
        report_path = stage / "independent_check.json"
        differences_path = stage / "differences.csv"
        report_path.write_bytes(_canonical_json(report))
        differences_path.write_bytes(_differences_bytes(report))
        pointer = {
            "schema_version": 1,
            "status": PASS_STATUS,
            "verification_id": verification_id,
            "verification_path": str(final.relative_to(PROJECT_ROOT)),
            "build_id": report["build_id"],
            "contract_id": report["contract_id"],
            "primary_manifest_sha256": report["primary_manifest_sha256"],
            "independent_check_sha256": _sha256(report_path),
            "differences_sha256": _sha256(differences_path),
            "signal_date_universe_authorized": True,
            "authorizations": dict(report["authorizations"]),
        }
        os.replace(stage, final)
        try:
            _atomic_json(VERIFIED_POINTER_PATH, pointer)
        except Exception:
            os.replace(final, stage)
            raise
        return pointer
    except Exception:
        raise
