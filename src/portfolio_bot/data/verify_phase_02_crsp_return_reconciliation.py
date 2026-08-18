"""Independent verifier for the Phase 2 CRSP return gate.

This module deliberately does not import the primary return builder or its
policy functions.  It scans the published Parquets, reconstructs event-group
and row-level invariants, authenticates every consumed file, and publishes a
verified pointer only when every aggregate check passes.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import tomllib
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data.database_location import resolve_recorded_database_path
from portfolio_bot.data.phase_02_crsp_typed_backbone_core import load_schema_registry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRIMARY_POINTER_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CRSP_RETURN_RECONCILIATION_PRIMARY.json"
VERIFIED_POINTER_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CRSP_RETURN_RECONCILIATION_VERIFIED.json"
SCHEMA_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/crsp_return_reconciliation_schema_v1.json"
POLICY_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/crsp_return_reconciliation_policy_v1.toml"
LOCK_PATH = PROJECT_ROOT / "configs/phase_02_canonicalization/crsp_return_reconciliation_empirical_lock_v1.json"
PATHS_PATH = PROJECT_ROOT / "configs/paths.toml"
CALENDAR_POINTER_PATH = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CALENDAR_VERIFIED.json"
ARTIFACT_PARENT = PROJECT_ROOT / "artifacts/phase_02_canonical_panel/crsp_return_reconciliation_verifications"
PASS_STATUS = "verified_passed"
FAIL_STATUS = "verification_failed"
RETURN_QUANTUM = Decimal("0.000000000001")
RETURN_FLOOR = Decimal("-1.000000000000")


class ReturnVerificationError(RuntimeError):
    """Raised when independent evidence cannot be scanned or published."""


def _canonical_json(value: Any) -> bytes:
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


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ReturnVerificationError(f"Expected a JSON object at {path}")
    return value


def _database_path(recorded: str, database_root: Path) -> Path:
    path = Path(recorded)
    if not path.is_absolute():
        return database_root / path
    return resolve_recorded_database_path(path, current_database_root=database_root)


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
        passed = actual == expected
        self.rows.append(
            {
                "check_id": check_id,
                "category": category,
                "status": "PASS" if passed else "FAIL",
                "expected": expected,
                "actual": actual,
                "evidence": evidence,
                "message": (
                    "Independent evidence reproduced the frozen expectation."
                    if passed
                    else "Independent evidence differs from the frozen expectation."
                ),
            }
        )


def _validated_groups(path: Path, spec: Any) -> Iterator[pa.Table]:
    parquet = pq.ParquetFile(path)
    expected_schema = spec.arrow_schema()
    if parquet.metadata.num_rows != spec.expected_rows:
        raise ReturnVerificationError(f"Row count differs for {path.name}")
    if not parquet.schema_arrow.equals(expected_schema, check_metadata=True):
        raise ReturnVerificationError(f"Exact schema differs for {path.name}")
    required = [field.name for field in expected_schema if not field.nullable]
    previous_key: tuple[Any, ...] | None = None
    scanned = 0
    for group_index in range(parquet.num_row_groups):
        table = parquet.read_row_group(group_index)
        scanned += table.num_rows
        for name in required:
            if table[name].null_count:
                raise ReturnVerificationError(
                    f"Required column {name!r} contains nulls in {path.name}"
                )
        keys = [
            table[name].combine_chunks().to_numpy(zero_copy_only=False)
            for name in spec.primary_key
        ]
        if table.num_rows:
            first_key = tuple(array[0] for array in keys)
            last_key = tuple(array[-1] for array in keys)
            if previous_key is not None and first_key <= previous_key:
                raise ReturnVerificationError(f"Cross-group key order differs in {path.name}")
            equal = np.ones(table.num_rows - 1, dtype=bool)
            descending = np.zeros(table.num_rows - 1, dtype=bool)
            for array in keys:
                descending |= equal & (array[1:] < array[:-1])
                equal &= array[1:] == array[:-1]
            if np.any(descending) or np.any(equal):
                raise ReturnVerificationError(f"Primary key order/uniqueness differs in {path.name}")
            previous_key = last_key
        yield table
    if scanned != spec.expected_rows:
        raise ReturnVerificationError(f"Streaming row count differs in {path.name}")


def _read_validated(path: Path, spec: Any) -> pa.Table:
    return pa.concat_tables(list(_validated_groups(path, spec)))


def _manifest_calendar(database_root: Path) -> np.ndarray:
    pointer = _json(CALENDAR_POINTER_PATH)
    if pointer.get("status") != PASS_STATUS:
        raise ReturnVerificationError("Calendar verified pointer is not passed")
    manifest_path = PROJECT_ROOT / pointer["primary_artifact_path"] / "manifest.json"
    if _sha256(manifest_path) != pointer["primary_manifest_sha256"]:
        raise ReturnVerificationError("Calendar manifest binding differs")
    manifest = _json(manifest_path)
    matches = [
        item for item in manifest["outputs"] if item["path"].endswith("/trading_calendar.parquet")
    ]
    if len(matches) != 1:
        raise ReturnVerificationError("Calendar output is not unique")
    item = matches[0]
    path = _database_path(item["path"], database_root)
    if path.stat().st_size != item["size_bytes"] or _sha256(path) != item["sha256"]:
        raise ReturnVerificationError("Calendar output binding differs")
    table = pq.read_table(path)
    name = next(name for name in table.column_names if "date" in name.lower())
    values = table[name].combine_chunks().to_numpy(zero_copy_only=False).astype("datetime64[D]")
    ordinals = values.astype(np.int64)
    if np.any(ordinals[1:] <= ordinals[:-1]):
        raise ReturnVerificationError("Calendar is not strictly ordered")
    return ordinals


def _distribution_checks(
    table: pa.Table,
    calendar: np.ndarray,
    checks: Checks,
) -> dict[str, int]:
    permno = table["permno"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    ex_date = (
        table["ex_date"].combine_chunks().to_numpy(zero_copy_only=False).astype("datetime64[D]").astype(np.int64)
    )
    positions = np.searchsorted(calendar, ex_date, side="left")
    effective = ex_date.copy()
    inside = positions < len(calendar)
    effective[inside] = calendar[positions[inside]]
    starts = np.r_[
        0,
        np.flatnonzero((permno[1:] != permno[:-1]) | (effective[1:] != effective[:-1])) + 1,
    ]
    ends = np.r_[starts[1:], table.num_rows]
    statuses = table["mapping_status"].combine_chunks().to_numpy(zero_copy_only=False).astype(str)
    assigned = table["assigned_exactly_once"].combine_chunks().to_numpy(zero_copy_only=False).astype(bool)
    mapped = pc.is_valid(table["mapped_return_date"].combine_chunks()).to_numpy(zero_copy_only=False)
    mapped_start = pc.is_valid(
        table["mapped_return_span_start_date"].combine_chunks()
    ).to_numpy(zero_copy_only=False)
    group_status: Counter[str] = Counter()
    inconsistent_groups = 0
    for start, end in zip(starts, ends, strict=True):
        unique = set(statuses[start:end].tolist())
        if len(unique) != 1 or np.any(assigned[start:end] != assigned[start]):
            inconsistent_groups += 1
            continue
        group_status[next(iter(unique))] += 1
    checks.add(
        "DISTRIBUTION_EFFECTIVE_GROUP_COUNT",
        category="distribution",
        expected=259104,
        actual=len(starts),
        evidence="independent verified-calendar grouping",
    )
    checks.add(
        "DISTRIBUTION_GROUP_STATUS_COUNTS",
        category="distribution",
        expected={
            "assigned_unique_containing_return_span": 258512,
            "off_calendar_assigned_unique_containing_return_span": 83,
            "structural_after_last_security_day": 412,
            "structural_missing_return_internal_gap": 59,
            "structural_no_scoped_security_day": 38,
        },
        actual=dict(sorted(group_status.items())),
        evidence="one status reconstructed per effective security/date group",
    )
    checks.add(
        "DISTRIBUTION_ASSIGNMENT_NULL_CONSISTENCY",
        category="distribution",
        expected={
            "inconsistent_groups": 0,
            "assigned_without_mapped_date": 0,
            "assigned_without_mapped_start": 0,
            "unassigned_with_mapped_interval": 0,
        },
        actual={
            "inconsistent_groups": inconsistent_groups,
            "assigned_without_mapped_date": int(np.sum(assigned & ~mapped)),
            "assigned_without_mapped_start": int(np.sum(assigned & ~mapped_start)),
            "unassigned_with_mapped_interval": int(
                np.sum(~assigned & (mapped | mapped_start))
            ),
        },
        evidence="ledger flags and mapped endpoints",
    )
    return {
        "assigned_event_rows": int(assigned.sum()),
        "unassigned_event_rows": int((~assigned).sum()),
    }


def _delisting_checks(table: pa.Table, checks: Checks) -> dict[str, int]:
    status = table["treatment_status"].combine_chunks().to_numpy(zero_copy_only=False).astype(str)
    counts = dict(sorted(Counter(status.tolist()).items()))
    expected = {
        "missing_nonperformance_terminal_leg_unresolved": 170,
        "missing_performance_imputed_on_delisting_date_once": 374,
        "observed_event_return_attached_to_delisting_date_once": 7847,
    }
    checks.add(
        "DELISTING_TREATMENT_COUNTS",
        category="delisting",
        expected=expected,
        actual=counts,
        evidence="complete delisting ledger scan",
    )
    posting = table["posting_date"].combine_chunks().to_pylist()
    amount = table["amount_date"].combine_chunks().to_pylist()
    next_date = table["next_date"].combine_chunks().to_pylist()
    available = table["outcome_availability_date"].combine_chunks().to_pylist()
    availability_differences = sum(
        actual != max(value for value in (post, amt, nxt) if value is not None)
        for post, amt, nxt, actual in zip(posting, amount, next_date, available, strict=True)
    )
    checks.add(
        "DELISTING_OUTCOME_AVAILABILITY",
        category="point_in_time",
        expected=0,
        actual=availability_differences,
        evidence="max(DelDlyDt, DelAmtDt, DelNextDt) reconstructed independently",
    )
    assigned = table["assigned_exactly_once"].combine_chunks().to_numpy(zero_copy_only=False).astype(bool)
    imputed = table["imputation_applied"].combine_chunks().to_numpy(zero_copy_only=False).astype(bool)
    effective = table["effective_delisting_return"].combine_chunks().to_pylist()
    observed = table["observed_delisting_return"].combine_chunks().to_pylist()
    exchange = table["primary_exchange"].combine_chunks().to_numpy(zero_copy_only=False).astype(str)
    value_differences = 0
    for index, treatment in enumerate(status):
        if treatment.startswith("observed_"):
            value_differences += effective[index] != observed[index]
        elif treatment.startswith("missing_performance_"):
            expected_value = Decimal("-0.550000000000") if exchange[index] == "Q" else Decimal("-0.300000000000")
            value_differences += effective[index] != expected_value
        else:
            value_differences += effective[index] is not None
    checks.add(
        "DELISTING_EXACT_VALUES_AND_ASSIGNMENT",
        category="delisting",
        expected={"value_differences": 0, "assigned_false": 0, "imputed": 374},
        actual={
            "value_differences": int(value_differences),
            "assigned_false": int((~assigned).sum()),
            "imputed": int(imputed.sum()),
        },
        evidence="exact Decimal event values and assignment flags",
    )
    return {"events": table.num_rows, "adjustments": sum(value is not None for value in effective)}


def _daily_checks(
    path: Path,
    spec: Any,
    checks: Checks,
    price_return_lock: Mapping[str, Any],
) -> dict[str, Any]:
    states: Counter[str] = Counter()
    reconciliation: Counter[str] = Counter()
    distribution_count = 0
    delisting_count = 0
    adjustment_count = 0
    authorized_true = 0
    usable_null_mismatches = 0
    availability_before_date = 0
    state_null_mismatches = 0
    event_formula_differences = 0
    verified_span_without_start = 0
    unresolved_span_status_mismatches = 0
    pre_scope_multi_span_rows = 0
    pre_scope_multi_span_shape_mismatches = 0
    retx_valid_rows = 0
    retx_total_null_asymmetry_rows = 0
    retx_at_or_below_minus_one_rows = 0
    ordinary_one_session_rows = 0
    ordinary_one_session_observed_rows = 0
    eligible_price_return_rows = 0
    eligible_price_return_late_rows = 0
    rows = 0
    value_states = {"observed", "observed_plus_delisting_adjustment", "imputed_delisting"}
    for group_index, table in enumerate(_validated_groups(path, spec)):
        rows += table.num_rows
        state = table["authoritative_return_state"].combine_chunks().to_numpy(zero_copy_only=False).astype(str)
        rec = table["reconciliation_status"].combine_chunks().to_numpy(zero_copy_only=False).astype(str)
        states.update(state.tolist())
        reconciliation.update(rec.tolist())
        distribution_count += int(pc.sum(table["distribution_event_count"]).as_py())
        delisting_count += int(pc.sum(table["delisting_event_count"]).as_py())
        adjustment_valid = pc.is_valid(table["delisting_adjustment_return"]).to_numpy(zero_copy_only=False)
        adjustment_count += int(adjustment_valid.sum())
        authoritative_valid = pc.is_valid(table["authoritative_total_return"]).to_numpy(zero_copy_only=False)
        raw_total_valid = pc.is_valid(table["raw_total_return"])
        retx_valid = pc.is_valid(table["raw_return_ex_distributions"])
        retx_valid_rows += int(pc.sum(pc.cast(retx_valid, pa.int64())).as_py())
        retx_total_null_asymmetry_rows += int(
            pc.sum(pc.cast(pc.xor(raw_total_valid, retx_valid), pa.int64())).as_py()
        )
        retx_at_or_below_minus_one_rows += int(
            pc.sum(
                pc.cast(
                    pc.fill_null(
                        pc.less_equal(table["raw_return_ex_distributions"], -1),
                        False,
                    ),
                    pa.int64(),
                )
            ).as_py()
        )
        usable = table["return_history_usable"].combine_chunks().to_numpy(zero_copy_only=False).astype(bool)
        authorized = table["return_history_authorized"].combine_chunks().to_numpy(zero_copy_only=False).astype(bool)
        authorized_true += int(authorized.sum())
        usable_null_mismatches += int(np.sum(usable != authoritative_valid))
        calendar_date = table["calendar_date"].combine_chunks().to_numpy(zero_copy_only=False).astype("datetime64[D]")
        availability = table["outcome_availability_date"].combine_chunks().to_numpy(zero_copy_only=False).astype("datetime64[D]")
        availability_before_date += int(np.sum(availability < calendar_date))
        expected_valid = np.isin(state, list(value_states))
        state_null_mismatches += int(np.sum(expected_valid != authoritative_valid))
        starts = pc.is_valid(
            table["return_span_start_date"].combine_chunks()
        ).to_numpy(zero_copy_only=False)
        span_status = (
            table["span_alignment_status"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(str)
        )
        verified_span_without_start += int(
            np.sum(np.char.startswith(span_status, "verified_") & ~starts)
        )
        leg_class = (
            table["return_leg_class"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(str)
        )
        ordinary_one = leg_class == "ordinary_one_session"
        observed = state == "observed"
        ordinary_one_session_rows += int(ordinary_one.sum())
        ordinary_one_session_observed_rows += int((ordinary_one & observed).sum())
        mapped = (
            table["mapped_trading_periods"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        covered = (
            table["covered_trading_sessions"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        retx_values = table["raw_return_ex_distributions"].combine_chunks().to_pylist()
        valid_gross = np.asarray(
            [value is not None and value > -1 for value in retx_values],
            dtype=bool,
        )
        eligible_price = (
            ordinary_one
            & observed
            & (mapped == 1)
            & (covered == 1)
            & usable
            & valid_gross
        )
        eligible_price_return_rows += int(eligible_price.sum())
        eligible_price_return_late_rows += int(
            np.sum(eligible_price & (availability > calendar_date))
        )
        boundary_status = span_status == "pre_scope_multi_session_span_unresolved"
        pre_scope_multi_span_rows += int(boundary_status.sum())
        pre_scope_multi_span_shape_mismatches += int(
            np.sum(
                boundary_status
                & (
                    starts
                    | (leg_class != "ordinary_multi_session")
                    | (state != "quarantined_unresolved_span")
                )
            )
        )
        pre_scope_multi_span_shape_mismatches += int(
            np.sum(
                (leg_class == "ordinary_multi_session")
                & ~starts
                & ~boundary_status
            )
        )
        unresolved = state == "quarantined_unresolved_span"
        unresolved_span_status_mismatches += int(
            np.sum(
                unresolved
                & ~np.isin(
                    span_status,
                    [
                        "pre_scope_multi_session_span_unresolved",
                        "no_return_span",
                    ],
                )
            )
        )

        event_rows = np.flatnonzero(np.isin(state, ["observed_plus_delisting_adjustment", "imputed_delisting"]))
        if len(event_rows):
            raw = table["raw_total_return"].combine_chunks().take(pa.array(event_rows)).to_pylist()
            adjustment = table["delisting_adjustment_return"].combine_chunks().take(pa.array(event_rows)).to_pylist()
            authoritative = table["authoritative_total_return"].combine_chunks().take(pa.array(event_rows)).to_pylist()
            for before, event, actual in zip(raw, adjustment, authoritative, strict=True):
                expected = max(
                    RETURN_FLOOR,
                    ((Decimal(1) + before) * (Decimal(1) + event) - Decimal(1)).quantize(
                        RETURN_QUANTUM, rounding=ROUND_HALF_EVEN
                    ),
                )
                event_formula_differences += actual != expected
        if group_index % 20 == 0:
            print(f"independent daily scan: {rows:,}/{spec.expected_rows:,} rows", flush=True)

    expected_states = {
        "imputed_delisting": 374,
        "missing": 536990,
        "observed": 26224286,
        "observed_plus_delisting_adjustment": 7847,
        "quarantined_MV": 39,
        "quarantined_unresolved_span": 4,
        "suppressed_delisting_posting_payload": 8383,
        "unresolved_nonperformance_delisting": 170,
    }
    checks.add(
        "DAILY_AUTHORITY_STATE_COUNTS",
        category="daily",
        expected=expected_states,
        actual=dict(sorted(states.items())),
        evidence="full independent row-group scan",
    )
    checks.add(
        "DAILY_AUTHORIZATION_AND_AVAILABILITY",
        category="daily",
        expected={
            "authorized_true": 0,
            "usable_null_mismatches": 0,
            "availability_before_date": 0,
            "state_null_mismatches": 0,
            "verified_span_without_start": 0,
            "unresolved_span_status_mismatches": 0,
            "pre_scope_multi_span_rows": 1,
            "pre_scope_multi_span_shape_mismatches": 0,
        },
        actual={
            "authorized_true": authorized_true,
            "usable_null_mismatches": usable_null_mismatches,
            "availability_before_date": availability_before_date,
            "state_null_mismatches": state_null_mismatches,
            "verified_span_without_start": verified_span_without_start,
            "unresolved_span_status_mismatches": unresolved_span_status_mismatches,
            "pre_scope_multi_span_rows": pre_scope_multi_span_rows,
            "pre_scope_multi_span_shape_mismatches": pre_scope_multi_span_shape_mismatches,
        },
        evidence="row authorization, canonical null state, and point-in-time guards",
    )
    checks.add(
        "DAILY_EVENT_FORMULA_AND_RECONCILIATION",
        category="daily",
        expected={"event_formula_differences": 0, "unexplained_total_minus_retx": 0},
        actual={
            "event_formula_differences": int(event_formula_differences),
            "unexplained_total_minus_retx": int(reconciliation.get("unexplained_total_minus_retx", 0)),
        },
        evidence="exact Decimal terminal-event formula and complete status scan",
    )
    expected_price_return = {
        "population_rows": int(price_return_lock["population_rows"]),
        "numeric_DlyRetx_rows": int(price_return_lock["numeric_DlyRetx_rows"]),
        "DlyRet_vs_DlyRetx_null_asymmetry_rows": int(
            price_return_lock["DlyRet_vs_DlyRetx_null_asymmetry_rows"]
        ),
        "ordinary_one_session_rows": int(
            price_return_lock["ordinary_one_session_rows"]
        ),
        "ordinary_one_session_observed_rows": int(
            price_return_lock["ordinary_one_session_observed_rows"]
        ),
        "eligible_observed_one_session_DlyRetx_rows": int(
            price_return_lock["eligible_observed_one_session_DlyRetx_rows"]
        ),
        "eligible_rows_available_after_observation_date": int(
            price_return_lock["eligible_rows_available_after_observation_date"]
        ),
        "all_numeric_DlyRetx_at_or_below_minus_one_rows": int(
            price_return_lock["all_numeric_DlyRetx_at_or_below_minus_one_rows"]
        ),
    }
    checks.add(
        "DAILY_PRICE_RETURN_FEATURE_AUTHORITY",
        category="daily",
        expected=expected_price_return,
        actual={
            "population_rows": rows,
            "numeric_DlyRetx_rows": retx_valid_rows,
            "DlyRet_vs_DlyRetx_null_asymmetry_rows": (
                retx_total_null_asymmetry_rows
            ),
            "ordinary_one_session_rows": ordinary_one_session_rows,
            "ordinary_one_session_observed_rows": (
                ordinary_one_session_observed_rows
            ),
            "eligible_observed_one_session_DlyRetx_rows": (
                eligible_price_return_rows
            ),
            "eligible_rows_available_after_observation_date": (
                eligible_price_return_late_rows
            ),
            "all_numeric_DlyRetx_at_or_below_minus_one_rows": (
                retx_at_or_below_minus_one_rows
            ),
        },
        evidence=(
            "full independent row-group scan of the narrowly authorized CRSP "
            "price-return source"
        ),
    )
    return {
        "rows": rows,
        "distribution_count": distribution_count,
        "delisting_count": delisting_count,
        "adjustment_count": adjustment_count,
        "reconciliation": dict(sorted(reconciliation.items())),
    }


def _monthly_checks(path: Path, spec: Any, checks: Checks) -> dict[str, Any]:
    rows = 0
    reconciliation: Counter[str] = Counter()
    hard_statuses = 0
    count_differences = 0
    tolerance_differences = 0
    for table in _validated_groups(path, spec):
        rows += table.num_rows
        status = table["reconciliation_status"].combine_chunks().to_numpy(zero_copy_only=False).astype(str)
        reconciliation.update(status.tolist())
        numeric_status = table["numeric_transport_status"].combine_chunks().to_numpy(zero_copy_only=False).astype(str)
        hard_statuses += int(np.char.startswith(numeric_status, "hard_").sum())
        observation = table["daily_observation_count"].combine_chunks().to_numpy(zero_copy_only=False)
        usable = table["usable_daily_return_count"].combine_chunks().to_numpy(zero_copy_only=False)
        missing = table["missing_daily_return_count"].combine_chunks().to_numpy(zero_copy_only=False)
        count_differences += int(np.sum(observation - usable != missing))
        tolerance = table["tolerance_used"].combine_chunks().to_pylist()
        tolerance_differences += sum(value != Decimal("0.000060000000") for value in tolerance)
    checks.add(
        "MONTHLY_RECONCILIATION_GATE",
        category="monthly",
        expected={
            "status_counts": {"accounted": 1284297},
            "hard_statuses": 0,
            "count_differences": 0,
            "tolerance_differences": 0,
        },
        actual={
            "status_counts": dict(sorted(reconciliation.items())),
            "hard_statuses": hard_statuses,
            "count_differences": count_differences,
            "tolerance_differences": tolerance_differences,
        },
        evidence="full monthly table scan with independently checked arithmetic identities",
    )
    return {"rows": rows, "reconciliation": dict(sorted(reconciliation.items()))}


def _differences_csv(checks: list[dict[str, Any]]) -> bytes:
    fields = ("check_id", "category", "expected", "actual", "evidence", "message")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for check in checks:
        if check["status"] == "PASS":
            continue
        writer.writerow(
            {
                name: (
                    json.dumps(check[name], sort_keys=True, separators=(",", ":"))
                    if isinstance(check[name], (dict, list, bool)) or check[name] is None
                    else check[name]
                )
                for name in fields
            }
        )
    return buffer.getvalue().encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(_canonical_json(value))
    os.replace(temporary, path)


def verify_and_publish(*, publish: bool = True) -> dict[str, Any]:
    with PATHS_PATH.open("rb") as handle:
        paths = tomllib.load(handle)
    database_root = Path(os.environ.get("PORTFOLIO_DATABASE_ROOT", paths["database"]["root"]))
    pointer = _json(PRIMARY_POINTER_PATH)
    if pointer.get("status") != "return_reconciliation_primary_passed_pending_independent_verification":
        raise ReturnVerificationError("Primary pointer is not pending independent verification")
    manifest_path = PROJECT_ROOT / pointer["artifact_path"] / "manifest.json"
    if _sha256(manifest_path) != pointer["manifest_sha256"]:
        raise ReturnVerificationError("Primary manifest hash differs from pointer")
    manifest = _json(manifest_path)
    if manifest.get("build_id") != pointer.get("build_id"):
        raise ReturnVerificationError("Primary build identity differs")
    lock = _json(LOCK_PATH)
    schema = load_schema_registry(SCHEMA_PATH)
    interim_root = database_root / pointer["interim_path"]
    outputs = {item["path"]: item for item in manifest["outputs"]}
    checks = Checks([])

    checks.add(
        "AUTHORITY_BINDINGS",
        category="provenance",
        expected={
            "empirical_lock_sha256": _sha256(LOCK_PATH),
            "policy_sha256": _sha256(POLICY_PATH),
            "schema_sha256": _sha256(SCHEMA_PATH),
        },
        actual={key: manifest["bindings"][key] for key in ("empirical_lock_sha256", "policy_sha256", "schema_sha256")},
        evidence="live frozen files compared with primary manifest",
    )
    raw_root = database_root / paths["database"]["raw"]
    checks.add(
        "RAW_EVENT_LEDGER_HASHES",
        category="provenance",
        expected={
            "distribution": lock["bindings"]["raw_distribution_sha256"],
            "delisting": lock["bindings"]["raw_delisting_sha256"],
        },
        actual={
            "distribution": _sha256(raw_root / "crsp_distribution_ciz.csv"),
            "delisting": _sha256(raw_root / "crsp_delisting_ciz.csv"),
        },
        evidence="fresh complete SHA-256 scans",
    )

    paths_by_name: dict[str, Path] = {}
    output_differences = 0
    for name, spec in schema.tables.items():
        path = interim_root / f"{name}.parquet"
        item = outputs[path.name]
        if path.stat().st_size != item["size_bytes"] or _sha256(path) != item["sha256"]:
            output_differences += 1
        paths_by_name[name] = path
    checks.add(
        "PRIMARY_OUTPUT_HASHES_AND_SIZES",
        category="publication",
        expected=0,
        actual=output_differences,
        evidence="all four complete Parquet byte streams",
    )

    calendar = _manifest_calendar(database_root)
    distribution = _read_validated(
        paths_by_name["crsp_distribution_event"], schema.tables["crsp_distribution_event"]
    )
    distribution_summary = _distribution_checks(distribution, calendar, checks)
    delisting = _read_validated(
        paths_by_name["crsp_delisting_event"], schema.tables["crsp_delisting_event"]
    )
    delisting_summary = _delisting_checks(delisting, checks)
    daily_summary = _daily_checks(
        paths_by_name["crsp_security_day_return"],
        schema.tables["crsp_security_day_return"],
        checks,
        lock["adjusted_price_path_feature_lock"],
    )
    monthly_summary = _monthly_checks(
        paths_by_name["crsp_security_month_return_reconciliation"],
        schema.tables["crsp_security_month_return_reconciliation"],
        checks,
    )
    checks.add(
        "CROSS_TABLE_EVENT_CONSERVATION",
        category="cross_table",
        expected={
            "distribution_count": distribution_summary["assigned_event_rows"],
            "delisting_count": delisting_summary["events"],
            "adjustment_count": delisting_summary["adjustments"],
        },
        actual={
            "distribution_count": daily_summary["distribution_count"],
            "delisting_count": daily_summary["delisting_count"],
            "adjustment_count": daily_summary["adjustment_count"],
        },
        evidence="event-ledger totals reconciled to the complete daily table",
    )

    ordered_checks = sorted(checks.rows, key=lambda item: item["check_id"])
    fail_count = sum(item["status"] == "FAIL" for item in ordered_checks)
    status = PASS_STATUS if fail_count == 0 else FAIL_STATUS
    implementation_sha = _sha256(Path(__file__))
    report_without_id = {
        "schema_version": 1,
        "status": status,
        "build_id": pointer["build_id"],
        "contract_id": pointer["contract_id"],
        "primary_manifest_sha256": pointer["manifest_sha256"],
        "verifier_implementation_sha256": implementation_sha,
        "independence": {
            "primary_return_builder_imported_or_called": False,
            "event_raw_hashes_recomputed": True,
            "four_parquet_tables_fully_scanned": True,
            "security_level_rows_published": False,
        },
        "summary": {
            "check_count": len(ordered_checks),
            "pass_count": len(ordered_checks) - fail_count,
            "fail_count": fail_count,
            "difference_count": fail_count,
            "parquet_rows_scanned": (
                distribution.num_rows
                + delisting.num_rows
                + daily_summary["rows"]
                + monthly_summary["rows"]
            ),
        },
        "authorizations": {
            "return_history_authorized": fail_count == 0,
            "adjusted_price_path_feature_authorized": fail_count == 0,
            "beta_estimation_authorized": False,
            "features_labels_models_portfolios_pnl_performance_or_backtests_authorized": False,
            "validation_or_final_test_access_authorized": False,
            "remaining_blockers": (
                ["universe_and_auxiliary_source_panel_not_complete", "factor_source_and_residual_target_contract_not_complete"]
                if fail_count == 0
                else ["return_reconciliation_independent_verification_failed"]
            ),
        },
        "checks": ordered_checks,
    }
    verification_id = "verify-" + hashlib.sha256(_canonical_json(report_without_id)).hexdigest()[:16]
    report = {**report_without_id, "verification_id": verification_id}
    differences = _differences_csv(ordered_checks)
    result = {
        "status": status,
        "verification_id": verification_id,
        "check_count": len(ordered_checks),
        "fail_count": fail_count,
        "report": report,
    }
    if not publish:
        return result

    destination = ARTIFACT_PARENT / verification_id
    stage = ARTIFACT_PARENT / (".staging-" + verification_id)
    if destination.exists() or stage.exists():
        raise ReturnVerificationError("Verification destination already exists")
    stage.mkdir(parents=True)
    try:
        (stage / "independent_check.json").write_bytes(_canonical_json(report))
        (stage / "differences.csv").write_bytes(differences)
        os.replace(stage, destination)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    if status == PASS_STATUS:
        verified_pointer = {
            "schema_version": 1,
            "status": PASS_STATUS,
            "build_id": pointer["build_id"],
            "contract_id": pointer["contract_id"],
            "verification_id": verification_id,
            "verification_path": str(destination.relative_to(PROJECT_ROOT)),
            "primary_manifest_sha256": pointer["manifest_sha256"],
            "independent_check_sha256": _sha256(destination / "independent_check.json"),
            "differences_sha256": _sha256(destination / "differences.csv"),
            "authorizations": report["authorizations"],
        }
        _atomic_json(VERIFIED_POINTER_PATH, verified_pointer)
        result["verified_pointer"] = verified_pointer
    return result
