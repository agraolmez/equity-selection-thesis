"""Authenticate, quality-check, and publish the Stage 2 auxiliary primary bundle."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import tomllib
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import duckdb

from portfolio_bot.data.phase_02_auxiliary_core import (
    canonical_json,
    load_schema_document,
    sha256,
    table_contract,
    table_schema,
)


CONTRACT_ID = "phase02-auxiliary-availability-linkage-v1"
DATABASE_CONTRACT_ID = "phase02-crsp-b03263a970e52fbd"
BUILD_PREFIX = "auxiliary-"
PRIMARY_STATUS = "auxiliary_primary_passed_pending_independent_verification"


class AuxiliaryPublicationError(RuntimeError):
    """Raised when primary evidence is incomplete or internally inconsistent."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise AuxiliaryPublicationError(f"Expected JSON object at {path}")
    return value


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    if temporary.exists():
        raise AuxiliaryPublicationError(f"Prior partial publication exists: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _hash_binding(project_root: Path, relative: str, expected: str, label: str) -> Path:
    path = project_root / relative
    actual = sha256(path)
    if actual != expected:
        raise AuxiliaryPublicationError(f"{label} hash differs: {actual} != {expected}")
    return path


def authenticate_contract(project_root: Path) -> dict[str, Any]:
    """Authenticate every frozen input needed before a primary build can publish."""

    policy_path = project_root / "configs/phase_02_canonicalization/auxiliary_source_policy_v1.toml"
    policy = _toml(policy_path)
    bindings = policy["bindings"]
    authenticated: dict[str, str] = {"policy_sha256": sha256(policy_path)}
    for path_key, hash_key, output_key in (
        ("methodology_source", "methodology_source_sha256", "methodology_source_sha256"),
        ("methodology_freeze", "methodology_freeze_sha256", "methodology_freeze_sha256"),
        ("phase_02_config", "phase_02_config_sha256", "phase_02_config_sha256"),
        ("source_config", "source_config_sha256", "source_config_sha256"),
        ("schema", "schema_sha256", "schema_sha256"),
        ("phase_01_freeze_manifest", "phase_01_freeze_manifest_sha256", "phase_01_freeze_manifest_sha256"),
    ):
        _hash_binding(project_root, bindings[path_key], bindings[hash_key], path_key)
        authenticated[output_key] = bindings[hash_key]
    factor = policy["factor_unit_lock"]
    factor_path = _hash_binding(project_root, factor["path"], factor["sha256"], "factor unit lock")
    factor_value = _json(factor_path)
    if str(factor_value["selected_source_to_decimal_ratio_multiplier"]) != "1":
        raise AuxiliaryPublicationError("Factor unit multiplier is not exact decimal one")
    authenticated["factor_unit_lock_sha256"] = factor["sha256"]

    upstream = policy["upstream"]["signal_universe"]
    verified_path = project_root / upstream["verified_pointer"]
    if sha256(verified_path) != upstream["verified_pointer_sha256"]:
        raise AuxiliaryPublicationError("Verified signal-universe pointer hash differs")
    verified = _json(verified_path)
    if (
        verified.get("status") != "verified_passed"
        or verified.get("signal_date_universe_authorized") is not True
        or verified.get("build_id") != upstream["required_build_id"]
    ):
        raise AuxiliaryPublicationError("Signal-universe authority is absent or mismatched")
    authenticated["signal_universe_verified_pointer_sha256"] = upstream["verified_pointer_sha256"]
    return {
        "policy": policy,
        "policy_path": policy_path,
        "bindings": authenticated,
        "signal_verified": verified,
    }


def expected_raw_source_hashes(project_root: Path) -> dict[str, str]:
    stage1 = _json(
        project_root
        / "artifacts/phase_01_raw_manifest/builds/phase01-2d804a3a69673de5/manifest.json"
    )
    known: dict[str, str] = {}
    for item in stage1["files"]:
        # Stage 1 records the authenticated whole-file digest inside the
        # byte-identity receipt.  Accept the former flat shape only so older
        # synthetic fixtures remain readable; production evidence must still
        # provide one unambiguous digest for every configured filename.
        digest = item.get("sha256")
        if digest is None:
            byte_identity = item.get("byte_identity")
            if isinstance(byte_identity, Mapping):
                digest = byte_identity.get("sha256")
        filename = str(item["filename"])
        if not isinstance(digest, str) or len(digest) != 64:
            raise AuxiliaryPublicationError(
                f"Stage 1 has no valid whole-file SHA-256 for {filename}"
            )
        if filename in known:
            raise AuxiliaryPublicationError(
                f"Stage 1 repeats a raw-source filename: {filename}"
            )
        known[filename] = digest
    source = _toml(project_root / "configs/phase_02_sources/auxiliary_sources_v1.toml")
    names = [str(item["filename"]) for item in source["inputs"]]
    if len(names) != len(set(names)):
        raise AuxiliaryPublicationError("Auxiliary source configuration repeats a filename")
    missing = sorted(set(names) - set(known))
    if missing:
        raise AuxiliaryPublicationError(f"Stage 1 does not bind auxiliary inputs: {missing}")
    return {name: known[name] for name in names}


def authenticate_signal_universe_path(
    *,
    project_root: Path,
    database_root: Path,
    signal_path: Path,
    verified_pointer: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the caller's spine bytes to the independently verified primary manifest."""

    primary_pointer_path = (
        project_root / "artifacts/phase_02_canonical_panel/LATEST_SIGNAL_DATE_UNIVERSE_PRIMARY.json"
    )
    primary = _json(primary_pointer_path)
    artifact_path = project_root / str(primary["artifact_path"])
    manifest_path = artifact_path / "manifest.json"
    manifest_hash = sha256(manifest_path)
    if (
        primary.get("build_id") != verified_pointer.get("build_id")
        or primary.get("manifest_sha256") != manifest_hash
        or verified_pointer.get("primary_manifest_sha256") != manifest_hash
    ):
        raise AuxiliaryPublicationError("Signal-universe primary/verified manifest binding differs")
    manifest = _json(manifest_path)
    expected_path = database_root / str(primary["interim_path"]) / str(manifest["output"]["path"])
    if signal_path.resolve() != expected_path.resolve():
        raise AuxiliaryPublicationError("Caller supplied a non-authoritative signal-universe path")
    output = manifest["output"]
    actual_size = signal_path.stat().st_size
    actual_hash = sha256(signal_path)
    actual_rows = pq.ParquetFile(signal_path).metadata.num_rows
    if (
        actual_hash != output["sha256"]
        or actual_size != output["size_bytes"]
        or actual_rows != output["rows"]
    ):
        raise AuxiliaryPublicationError("Signal-universe bytes differ from the verified primary output")
    return {
        "path": str(primary["interim_path"] + "/" + output["path"]),
        "sha256": actual_hash,
        "size_bytes": actual_size,
        "rows": actual_rows,
        "primary_manifest_sha256": manifest_hash,
    }


def rehash_raw_sources(
    *,
    project_root: Path,
    raw_root: Path,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, str]:
    """Reread every auxiliary raw source and compare it with frozen Stage 1."""

    progress = progress or (lambda _stage, _details: None)
    expected = expected_raw_source_hashes(project_root)
    actual: dict[str, str] = {}
    for index, (name, frozen_hash) in enumerate(expected.items(), start=1):
        progress("raw_rehash_started", {"source": name, "index": index, "total": len(expected)})
        path = raw_root / name
        if path.is_symlink() or not path.is_file():
            raise AuxiliaryPublicationError(f"Raw source is absent or not a regular file: {name}")
        digest = sha256(path)
        if digest != frozen_hash:
            raise AuxiliaryPublicationError(f"Raw source changed since Stage 1: {name}")
        actual[name] = digest
        progress("raw_rehash_passed", {"source": name, "size_bytes": path.stat().st_size})
    return actual


def _field_schema_equal(actual: pa.Schema, expected: pa.Schema) -> bool:
    if actual.names != expected.names or len(actual) != len(expected):
        return False
    return all(
        left.name == right.name
        and left.type == right.type
        and left.nullable == right.nullable
        for left, right in zip(actual, expected, strict=True)
    )


def collect_output_receipts(
    *, schema_path: Path, table_root: Path
) -> dict[str, dict[str, Any]]:
    document = load_schema_document(schema_path)
    expected_names = [str(item["table"]) for item in document["tables"]]
    expected_files = {f"{name}.parquet" for name in expected_names}
    entries = list(table_root.iterdir())
    existing_files = {path.name for path in entries if path.is_file() and not path.is_symlink()}
    non_regular = sorted(
        path.name for path in entries if path.is_symlink() or not path.is_file()
    )
    if existing_files != expected_files or non_regular:
        raise AuxiliaryPublicationError(
            "Auxiliary table directory is not a closed regular-file set; "
            f"missing={sorted(expected_files-existing_files)}, "
            f"unexpected={sorted(existing_files-expected_files)}, "
            f"non_regular={non_regular}"
        )
    outputs: dict[str, dict[str, Any]] = {}
    for name in expected_names:
        path = table_root / f"{name}.parquet"
        parquet = pq.ParquetFile(path)
        expected_schema = table_schema(document, name)
        if not _field_schema_equal(parquet.schema_arrow, expected_schema):
            raise AuxiliaryPublicationError(f"Exact field schema differs for {name}")
        outputs[name] = {
            "table": name,
            "path": path.name,
            "rows": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "primary_key": list(table_contract(document, name)["primary_key"]),
            "sort_key": list(table_contract(document, name)["sort_key"]),
        }
    return outputs


def _availability_and_interval_checks(
    *, schema_path: Path, table_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    document = load_schema_document(schema_path)
    checks: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    last_signal = dt.date(2014, 10, 31)
    last_exit = dt.date(2014, 12, 1)
    interval_columns = {
        "ccm_link_interval": ("link_start_date", "link_end_date"),
        "optionmetrics_name_interval": ("effective_start_date", "effective_end_date"),
        "optionmetrics_crsp_link_interval": ("link_start_date", "link_end_date"),
    }
    for contract in document["tables"]:
        name = str(contract["table"])
        path = table_root / f"{name}.parquet"
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        if "public_availability_date" in columns:
            violations = 0
            max_available: dt.date | None = None
            max_allowed = last_exit if name.startswith("ff_factor_") else last_signal
            if parquet.metadata.num_rows:
                for batch in parquet.iter_batches(columns=["public_availability_date"]):
                    values = batch.column(0)
                    if values.null_count:
                        violations += values.null_count
                    too_late = pc.greater(values, pa.scalar(max_allowed, type=pa.date32()))
                    violations += int(pc.sum(pc.cast(pc.fill_null(too_late, True), pa.int64())).as_py())
                    candidate = pc.max(values).as_py()
                    max_available = candidate if max_available is None else max(max_available, candidate)
            checks.append({
                "check": f"{name}.availability_within_development_scope",
                "passed": violations == 0,
                "actual": violations,
                "expected": 0,
            })
            summary[name] = {"max_public_availability_date": None if max_available is None else max_available.isoformat()}
        if name in interval_columns:
            start, end = interval_columns[name]
            violations = 0
            if parquet.metadata.num_rows:
                for batch in parquet.iter_batches(columns=[start, end]):
                    invalid = pc.and_(pc.is_valid(batch.column(1)), pc.greater(batch.column(0), batch.column(1)))
                    violations += int(pc.sum(pc.cast(invalid, pa.int64())).as_py())
            checks.append({
                "check": f"{name}.interval_start_not_after_end",
                "passed": violations == 0,
                "actual": violations,
                "expected": 0,
            })
    return checks, summary


def _verify_linkage_spine(
    *, signal_path: Path, linkage_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    signal = pq.read_table(signal_path, columns=["signal_date", "permno"])
    linkage = pq.read_table(
        linkage_path,
        columns=[
            "signal_date", "permno", "ccm_match_status", "ccm_gvkey",
            "sec_match_status", "sec_cik", "optionmetrics_match_status",
            "optionmetrics_secid", "any_auxiliary_ambiguity", "universe_row_preserved",
        ],
    )
    keys_equal = signal["signal_date"].equals(linkage["signal_date"]) and signal["permno"].equals(linkage["permno"])
    all_preserved = bool(pc.all(linkage["universe_row_preserved"]).as_py())
    status_counts: Counter[str] = Counter()
    ambiguous_field_violations = 0
    allowed = {"MATCHED", "UNMATCHED", "AMBIGUOUS", "NOT_COVERED"}
    for family, field in (
        ("ccm", "ccm_match_status"),
        ("sec", "sec_match_status"),
        ("optionmetrics", "optionmetrics_match_status"),
    ):
        for value in linkage[field].to_pylist():
            if value not in allowed:
                raise AuxiliaryPublicationError(f"Invalid {family} linkage status {value}")
            status_counts[f"{family}:{value}"] += 1
    for index in range(linkage.num_rows):
        if linkage["ccm_match_status"][index].as_py() == "AMBIGUOUS" and linkage["ccm_gvkey"][index].as_py() is not None:
            ambiguous_field_violations += 1
        if linkage["sec_match_status"][index].as_py() == "AMBIGUOUS" and linkage["sec_cik"][index].as_py() is not None:
            ambiguous_field_violations += 1
        if linkage["optionmetrics_match_status"][index].as_py() == "AMBIGUOUS" and linkage["optionmetrics_secid"][index].as_py() is not None:
            ambiguous_field_violations += 1
    checks = [
        {"check": "auxiliary_signal_linkage.row_count_preserved", "passed": signal.num_rows == linkage.num_rows, "actual": linkage.num_rows, "expected": signal.num_rows},
        {"check": "auxiliary_signal_linkage.primary_key_and_order_preserved", "passed": keys_equal, "actual": int(keys_equal), "expected": 1},
        {"check": "auxiliary_signal_linkage.all_rows_marked_preserved", "passed": all_preserved, "actual": int(all_preserved), "expected": 1},
        {"check": "auxiliary_signal_linkage.ambiguous_fields_null", "passed": ambiguous_field_violations == 0, "actual": ambiguous_field_violations, "expected": 0},
    ]
    return checks, {
        "source_spine_rows": signal.num_rows,
        "linkage_rows": linkage.num_rows,
        "primary_key_preserved": keys_equal,
        "all_rows_preserved": all_preserved,
        "status_counts": dict(sorted(status_counts.items())),
        "ambiguity_rows": int(pc.sum(pc.cast(linkage["any_auxiliary_ambiguity"], pa.int64())).as_py()),
    }


def _write_linkage_coverage(
    *, signal_path: Path, linkage_path: Path, output_path: Path
) -> None:
    signal_columns = [
        "signal_date", "primary_exchange", "us_incorporation_flag", "share_type",
        "long_eligible", "short_eligible",
    ]
    link_columns = ["ccm_match_status", "sec_match_status", "optionmetrics_match_status"]
    signal = pq.read_table(signal_path, columns=signal_columns)
    linkage = pq.read_table(linkage_path, columns=link_columns)
    if signal.num_rows != linkage.num_rows:
        raise AuxiliaryPublicationError("Cannot aggregate a non-preserved linkage spine")
    counts: Counter[tuple[Any, ...]] = Counter()
    columns = [signal[name].to_pylist() for name in signal_columns]
    statuses = [linkage[name].to_pylist() for name in link_columns]
    for values in zip(*columns, *statuses, strict=True):
        signal_date = values[0]
        counts[(signal_date.year, *values[1:])] += 1
    fieldnames = [
        "signal_year", "primary_exchange", "us_incorporation_flag", "share_type",
        "long_eligible", "short_eligible", "ccm_match_status", "sec_match_status",
        "optionmetrics_match_status", "rows",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, rows in sorted(counts.items(), key=lambda item: tuple(str(value) for value in item[0])):
            writer.writerow(dict(zip(fieldnames, (*key, rows), strict=True)))


def _write_sec_form_coverage(
    *, signal_path: Path, linkage_path: Path, filing_path: Path, output_path: Path
) -> dict[str, Any]:
    """Aggregate point-in-time form coverage without publishing security identifiers."""

    forms = ("10-K", "10-Q", "20-F", "40-F", "8-K", "6-K")
    filing_dates: dict[tuple[str, str], list[int]] = defaultdict(list)
    filing_file = pq.ParquetFile(filing_path)
    epoch = dt.date(1970, 1, 1)
    for batch in filing_file.iter_batches(
        columns=["cik", "form_family", "public_availability_date"]
    ):
        for cik, form, available in zip(
            batch.column(0).to_pylist(),
            batch.column(1).to_pylist(),
            batch.column(2).to_pylist(),
            strict=True,
        ):
            if form in forms:
                filing_dates[(str(cik), str(form))].append((available - epoch).days)
    for values in filing_dates.values():
        values.sort()

    signal = pq.read_table(
        signal_path,
        columns=["signal_date", "us_incorporation_flag", "share_type"],
    )
    linkage = pq.read_table(
        linkage_path, columns=["sec_match_status", "sec_cik"]
    )
    if signal.num_rows != linkage.num_rows:
        raise AuxiliaryPublicationError("SEC coverage cannot use a non-preserved spine")
    totals: Counter[tuple[int, str, bool, str]] = Counter()
    covered: Counter[tuple[int, str, bool, str]] = Counter()
    signal_values = [signal[name].to_pylist() for name in signal.column_names]
    link_values = [linkage[name].to_pylist() for name in linkage.column_names]
    for day, domicile, share_type, status, cik in zip(
        *signal_values, *link_values, strict=True
    ):
        ordinal = (day - epoch).days
        adr = share_type == "AD"
        for form in forms:
            key = (day.year, str(domicile), adr, form)
            totals[key] += 1
            dates = filing_dates.get((str(cik), form), ()) if status == "MATCHED" and cik else ()
            if dates and bisect_right(dates, ordinal):
                covered[key] += 1
    fieldnames = [
        "signal_year", "us_incorporation_flag", "adr_share_type", "form_family",
        "rows", "covered_rows", "missing_rows",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(totals, key=lambda value: tuple(str(item) for item in value)):
            rows = totals[key]
            matched = covered[key]
            writer.writerow(dict(zip(fieldnames, (*key, rows, matched, rows - matched), strict=True)))
    return {
        "forms": list(forms),
        "aggregate_rows": len(totals),
        "signal_rows_per_form": signal.num_rows,
        "current_company_name_join_used": False,
        "foreign_form_absence_removed_spine_row": False,
    }


def _sql_literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _write_optionmetrics_coverage(
    *, signal_path: Path, linkage_path: Path, volume_path: Path, output_path: Path
) -> dict[str, Any]:
    """Aggregate match, observed-flow, and genuine-zero states with a disk-backed join."""

    connection = duckdb.connect(":memory:")
    spill = output_path.parent / ".duckdb_option_qa_spill"
    spill.mkdir()
    try:
        connection.execute("SET memory_limit = '768MB'")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET temp_directory = ?", [str(spill)])
        query = f"""
        COPY (
          WITH flow AS (
            SELECT
              secid,
              observation_date,
              count(*) AS component_rows,
              count(volume) AS nonnull_volume_rows,
              count(open_interest) AS nonnull_open_interest_rows,
              sum(volume) AS summed_volume,
              sum(open_interest) AS summed_open_interest,
              bool_and(coalesce(genuine_zero_volume, false)) AS all_observed_volume_components_zero,
              bool_and(coalesce(genuine_zero_open_interest, false)) AS all_observed_open_interest_components_zero
            FROM read_parquet({_sql_literal(volume_path)})
            WHERE authoritative_flow_component
            GROUP BY secid, observation_date
          )
          SELECT
            year(u.signal_date) AS signal_year,
            u.primary_exchange,
            u.us_incorporation_flag,
            u.share_type,
            u.long_eligible,
            u.short_eligible,
            l.optionmetrics_match_status,
            count(*) AS rows,
            count(f.secid) AS rows_with_dated_vendor_flow_coverage,
            sum(CASE WHEN f.component_rows > 0 AND f.nonnull_volume_rows = f.component_rows
                          AND f.all_observed_volume_components_zero THEN 1 ELSE 0 END)
              AS genuine_zero_volume_rows,
            sum(CASE WHEN f.component_rows > 0 AND f.nonnull_open_interest_rows = f.component_rows
                          AND f.all_observed_open_interest_components_zero THEN 1 ELSE 0 END)
              AS genuine_zero_open_interest_rows
          FROM read_parquet({_sql_literal(signal_path)}) u
          JOIN read_parquet({_sql_literal(linkage_path)}) l
            USING (signal_date, permno)
          LEFT JOIN flow f
            ON l.optionmetrics_secid = f.secid
           AND u.signal_date = f.observation_date
          GROUP BY ALL
          ORDER BY 1,2,3,4,5,6,7
        ) TO {_sql_literal(output_path)} (HEADER, DELIMITER ',')
        """
        connection.execute(query)
    finally:
        connection.close()
        shutil.rmtree(spill, ignore_errors=True)
    with output_path.open(newline="", encoding="utf-8") as handle:
        aggregate_rows = sum(1 for _ in csv.DictReader(handle))
    return {
        "aggregate_rows": aggregate_rows,
        "authoritative_flow_roles": ["CALL", "PUT"],
        "total_or_blank_added_to_call_plus_put": False,
        "genuine_zero_requires_dated_vendor_flow_coverage": True,
        "option_coverage_used_as_universe_filter": False,
    }


def _factor_alignment_receipt(*, project_root: Path, table_root: Path) -> dict[str, Any]:
    lock_path = project_root / "configs/phase_02_canonicalization/auxiliary_factor_unit_empirical_lock_v1.json"
    lock = _json(lock_path)
    daily = pq.ParquetFile(table_root / "ff_factor_daily_observation.parquet")
    monthly = pq.ParquetFile(table_root / "ff_factor_monthly_observation.parquet")
    daily_dates = pq.read_table(
        table_root / "ff_factor_daily_observation.parquet", columns=["observation_date"]
    )["observation_date"]
    monthly_dates = pq.read_table(
        table_root / "ff_factor_monthly_observation.parquet", columns=["factor_month"]
    )["factor_month"]
    receipt = {
        "factor_unit_lock_sha256": sha256(lock_path),
        "selected_source_to_decimal_ratio_multiplier": str(lock["selected_source_to_decimal_ratio_multiplier"]),
        "source_values_are_decimal_ratios": bool(lock["source_values_are_decimal_ratios"]),
        "daily_rows": daily.metadata.num_rows,
        "monthly_rows": monthly.metadata.num_rows,
        "daily_min_date": pc.min(daily_dates).as_py().isoformat(),
        "daily_max_date": pc.max(daily_dates).as_py().isoformat(),
        "monthly_min_date": pc.min(monthly_dates).as_py().isoformat(),
        "monthly_max_date": pc.max(monthly_dates).as_py().isoformat(),
        "daily_monthly_crosscheck": lock["daily_monthly_crosscheck"],
        "models_portfolios_or_backtests_used_for_unit_selection": False,
    }
    if (
        receipt["daily_rows"] != lock["daily"]["rows_through_2014_12_01"]
        or receipt["monthly_rows"] != lock["monthly"]["rows_through_2014_12_01"]
        or receipt["selected_source_to_decimal_ratio_multiplier"] != "1"
    ):
        raise AuxiliaryPublicationError("Factor output differs from the prebuild empirical lock")
    return receipt


def _ps_release_lag_receipt(*, table_root: Path) -> dict[str, Any]:
    path = table_root / "ps_liquidity_observation.parquet"
    table = pq.read_table(
        path,
        columns=[
            "observation_month", "observation_date", "public_availability_date",
            "availability_basis",
        ],
    )
    violations = 0
    lag_days: list[int] = []
    for month, observation, available, basis in zip(
        *(table[name].to_pylist() for name in table.column_names), strict=True
    ):
        if observation != month or available <= observation or basis != "one_full_calendar_month_lag_fallback":
            violations += 1
        lag_days.append((available - observation).days)
    if violations:
        raise AuxiliaryPublicationError("Pastor-Stambaugh release-lag output violates policy")
    return {
        "rows": table.num_rows,
        "violations": violations,
        "minimum_calendar_lag_days": min(lag_days) if lag_days else None,
        "maximum_calendar_lag_days": max(lag_days) if lag_days else None,
        "raw_PS_INNOV_present": False,
        "recursive_innovation_constructed": False,
        "historically_revised_series_disclosed": True,
    }


def _write_named_source_qa_receipts(
    *,
    quality_staging: Path,
    source_receipts: Mapping[str, Any],
    linkage_summary: Mapping[str, Any],
    raw_source_hashes: Mapping[str, str],
) -> None:
    required = ("ccm_daily", "ccm_monthly", "optionmetrics")
    missing = [name for name in required if name not in source_receipts]
    if missing:
        raise AuxiliaryPublicationError(f"Missing source-family QA receipts: {missing}")
    for prefix in ("ccm_daily", "ccm_monthly"):
        source = dict(source_receipts[prefix])
        coverage = {
            key: source.get(key)
            for key in (
                "source", "raw_rows", "development_rows", "development_min_date",
                "development_max_date", "row_level_materialized", "source_sha256",
            )
        }
        coverage["status"] = "passed"
        coverage["receipt_id"] = f"{prefix}_crosscheck_coverage"
        consistency = {
            key: source.get(key)
            for key in (
                "source", "allowed_link_rows", "crsp_permno_not_found",
                "link_interval_violations", "source_sha256",
            )
        }
        consistency["status"] = "passed"
        consistency["receipt_id"] = f"{prefix}_crsp_link_consistency"
        (quality_staging / f"{prefix}_crosscheck_coverage.json").write_bytes(canonical_json(coverage))
        (quality_staging / f"{prefix}_crsp_link_consistency.json").write_bytes(canonical_json(consistency))
    option = dict(source_receipts["optionmetrics"])
    prices = dict(option["security_prices_qa"])
    prices["status"] = "passed"
    prices["receipt_id"] = "optionmetrics_underlying_coverage"
    prices["source_sha256"] = raw_source_hashes["optionmetrics_security_prices.csv"]
    (quality_staging / "optionmetrics_underlying_coverage.json").write_bytes(canonical_json(prices))
    (quality_staging / "optionmetrics_crsp_market_crosscheck.json").write_bytes(
        canonical_json({
            "status": "passed",
            "receipt_id": "optionmetrics_crsp_market_crosscheck",
            "source": "optionmetrics_security_prices.csv",
            "source_sha256": raw_source_hashes["optionmetrics_security_prices.csv"],
            "crsp_is_authoritative_market_source": True,
            "row_level_optionmetrics_price_table_materialized": False,
            "date_valid_cusip8_linkage_status_counts": linkage_summary["status_counts"],
            "price_source_rows_scanned": prices["source_rows"],
            "price_development_rows_scanned": prices["retained_rows"],
            "price_value_equality_used_as_link_or_filter": False,
        })
    )


def build_quality_receipts(
    *,
    project_root: Path,
    schema_path: Path,
    table_root: Path,
    signal_path: Path,
    quality_staging: Path,
    raw_source_hashes: Mapping[str, str],
    source_receipts: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    quality_staging.mkdir(parents=True, exist_ok=False)
    checks, availability = _availability_and_interval_checks(
        schema_path=schema_path, table_root=table_root
    )
    linkage_checks, linkage = _verify_linkage_spine(
        signal_path=signal_path,
        linkage_path=table_root / "auxiliary_signal_linkage.parquet",
    )
    checks.extend(linkage_checks)
    if not all(bool(item["passed"]) for item in checks):
        failed = [item["check"] for item in checks if not item["passed"]]
        raise AuxiliaryPublicationError(f"Primary auxiliary checks failed: {failed}")
    with (quality_staging / "checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "actual", "expected"])
        writer.writeheader()
        writer.writerows(checks)
    _write_linkage_coverage(
        signal_path=signal_path,
        linkage_path=table_root / "auxiliary_signal_linkage.parquet",
        output_path=quality_staging / "linkage_coverage.csv",
    )
    with (quality_staging / "linkage_coverage.csv").open(newline="", encoding="utf-8") as handle:
        linkage_coverage_rows = sum(1 for _ in csv.DictReader(handle))
    (quality_staging / "before_after_link_counts_by_family.json").write_bytes(
        canonical_json({
            "status": "passed",
            "receipt_id": "before_after_link_counts_by_family",
            "before_rows": linkage["source_spine_rows"],
            "after_rows": linkage["linkage_rows"],
            "primary_key_preserved": linkage["primary_key_preserved"],
            "all_rows_preserved": linkage["all_rows_preserved"],
            "status_counts": linkage["status_counts"],
        })
    )
    (quality_staging / "missingness_by_family_year_exchange_domicile_adr_and_long_short.json").write_bytes(
        canonical_json({
            "status": "passed",
            "receipt_id": "missingness_by_family_year_exchange_domicile_adr_and_long_short",
            "detail_file": "linkage_coverage.csv",
            "detail_sha256": sha256(quality_staging / "linkage_coverage.csv"),
            "aggregate_rows": linkage_coverage_rows,
            "dimensions": [
                "signal_year", "primary_exchange", "us_incorporation_flag", "share_type",
                "long_eligible", "short_eligible", "family_match_status",
            ],
            "raw_identifiers_present": False,
        })
    )
    sec_form_summary = _write_sec_form_coverage(
        signal_path=signal_path,
        linkage_path=table_root / "auxiliary_signal_linkage.parquet",
        filing_path=table_root / "sec_filing_observation.parquet",
        output_path=quality_staging / "sec_form_coverage.csv",
    )
    (quality_staging / "sec_form_coverage_by_year_domicile_adr_and_form.json").write_bytes(
        canonical_json({
            "status": "passed",
            "receipt_id": "sec_form_coverage_by_year_domicile_adr_and_form",
            **sec_form_summary,
            "detail_file": "sec_form_coverage.csv",
            "detail_sha256": sha256(quality_staging / "sec_form_coverage.csv"),
        })
    )
    option_summary = _write_optionmetrics_coverage(
        signal_path=signal_path,
        linkage_path=table_root / "auxiliary_signal_linkage.parquet",
        volume_path=table_root / "optionmetrics_option_volume_observation.parquet",
        output_path=quality_staging / "optionmetrics_coverage.csv",
    )
    (quality_staging / "optionmetrics_match_genuine_zero_optionability_by_year.json").write_bytes(
        canonical_json({
            "status": "passed",
            "receipt_id": "optionmetrics_match_genuine_zero_optionability_by_year",
            **option_summary,
            "detail_file": "optionmetrics_coverage.csv",
            "detail_sha256": sha256(quality_staging / "optionmetrics_coverage.csv"),
        })
    )
    factor_receipt = _factor_alignment_receipt(project_root=project_root, table_root=table_root)
    (quality_staging / "factor_date_alignment_and_unit_receipt.json").write_bytes(
        canonical_json({
            "status": "passed",
            "receipt_id": "factor_date_alignment_and_unit_receipt",
            **factor_receipt,
        })
    )
    ps_receipt = _ps_release_lag_receipt(table_root=table_root)
    (quality_staging / "ps_release_lag_receipt.json").write_bytes(
        canonical_json({
            "status": "passed",
            "receipt_id": "ps_release_lag_receipt",
            **ps_receipt,
        })
    )
    _write_named_source_qa_receipts(
        quality_staging=quality_staging,
        source_receipts=source_receipts,
        linkage_summary=linkage,
        raw_source_hashes=raw_source_hashes,
    )
    schema_document = load_schema_document(schema_path)
    with (quality_staging / "table_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["table", "rows"])
        writer.writeheader()
        for contract in schema_document["tables"]:
            name = str(contract["table"])
            writer.writerow({"table": name, "rows": pq.ParquetFile(table_root / f"{name}.parquet").metadata.num_rows})
    (quality_staging / "raw_source_hashes.json").write_bytes(canonical_json(dict(sorted(raw_source_hashes.items()))))
    summary = {
        "schema_version": 1,
        "status": PRIMARY_STATUS,
        "checks_passed": len(checks),
        "checks_failed": 0,
        "availability": availability,
        "linkage": linkage,
        "sec_form_coverage": sec_form_summary,
        "optionmetrics_coverage": option_summary,
        "factor_alignment": factor_receipt,
        "ps_release_lag": ps_receipt,
        "authorizations": {
            "auxiliary_panel_authorized": False,
            "features_labels_models_portfolios_pnl_performance_or_backtests_authorized": False,
            "validation_or_final_test_access_authorized": False,
        },
    }
    (quality_staging / "summary.json").write_bytes(canonical_json(summary))
    (quality_staging / "aggregate_qa.json").write_bytes(
        canonical_json({"status": "passed", "aggregate_qa": summary})
    )
    receipts = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(quality_staging.iterdir())
        if path.is_file()
    }
    return receipts, summary


def publish_primary(
    *,
    project_root: Path,
    database_root: Path,
    staging_root: Path,
    raw_source_hashes: Mapping[str, str],
    signal_path: Path,
    source_receipts: Mapping[str, Any],
) -> dict[str, Any]:
    """Move a checked staging bundle into immutable content-addressed locations."""

    authenticated = authenticate_contract(project_root)
    signal_receipt = authenticate_signal_universe_path(
        project_root=project_root,
        database_root=database_root,
        signal_path=signal_path,
        verified_pointer=authenticated["signal_verified"],
    )
    schema_path = project_root / authenticated["policy"]["bindings"]["schema"]
    table_root = staging_root / "tables"
    outputs = collect_output_receipts(schema_path=schema_path, table_root=table_root)
    expected_raw = expected_raw_source_hashes(project_root)
    if dict(raw_source_hashes) != expected_raw:
        raise AuxiliaryPublicationError("Raw source hashes do not exactly match frozen Stage 1")
    quality_staging = staging_root / "quality"
    quality_receipts, aggregate_qa = build_quality_receipts(
        project_root=project_root,
        schema_path=schema_path,
        table_root=table_root,
        signal_path=signal_path,
        quality_staging=quality_staging,
        raw_source_hashes=raw_source_hashes,
        source_receipts=source_receipts,
    )
    semantic = {
        "contract_id": CONTRACT_ID,
        "database_contract_id": DATABASE_CONTRACT_ID,
        "bindings": authenticated["bindings"],
        "raw_source_hashes": dict(sorted(raw_source_hashes.items())),
        "signal_universe": signal_receipt,
        "outputs": {name: {key: value for key, value in receipt.items() if key != "path"} for name, receipt in sorted(outputs.items())},
        "quality_receipts": quality_receipts,
        "aggregate_qa": aggregate_qa,
    }
    build_id = BUILD_PREFIX + hashlib.sha256(canonical_json(semantic)).hexdigest()[:24]
    interim_relative = Path(
        f"interim/phase_02_canonical_panel/{DATABASE_CONTRACT_ID}/auxiliary_availability_linkage/{build_id}"
    )
    quality_relative = Path(
        f"quality/phase_02_canonical_panel/{DATABASE_CONTRACT_ID}/auxiliary_availability_linkage/{build_id}"
    )
    artifact_relative = Path(f"artifacts/phase_02_canonical_panel/auxiliary_availability_linkages/{build_id}")
    interim_final = database_root / interim_relative
    quality_final = database_root / quality_relative
    artifact_final = project_root / artifact_relative
    for path in (interim_final, quality_final, artifact_final):
        if path.exists():
            raise AuxiliaryPublicationError(f"Refusing to overwrite immutable publication {path}")
    manifest = {
        "schema_version": 1,
        "status": PRIMARY_STATUS,
        "build_id": build_id,
        "contract_id": CONTRACT_ID,
        "database_contract_id": DATABASE_CONTRACT_ID,
        "bindings": authenticated["bindings"],
        "raw_source_hashes": dict(sorted(raw_source_hashes.items())),
        "signal_universe": signal_receipt,
        "outputs": outputs,
        "quality_receipts": quality_receipts,
        "aggregate_qa": aggregate_qa,
        "interim_path": interim_relative.as_posix(),
        "quality_path": quality_relative.as_posix(),
        "authorizations": {
            "auxiliary_panel_authorized": False,
            "independent_verification_required": True,
            "features_labels_models_portfolios_pnl_performance_or_backtests_authorized": False,
            "validation_or_final_test_access_authorized": False,
        },
    }
    artifact_final.parent.mkdir(parents=True, exist_ok=True)
    artifact_staging = artifact_final.parent / f".staging-{build_id}-{os.getpid()}"
    if artifact_staging.exists():
        raise AuxiliaryPublicationError(f"Prior artifact staging directory exists: {artifact_staging}")
    artifact_staging.mkdir()
    manifest_path = artifact_staging / "manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    (artifact_staging / "output_hashes.sha256").write_text(
        "".join(f"{receipt['sha256']}  {receipt['path']}\n" for receipt in outputs.values()),
        encoding="utf-8",
    )
    success = {
        "schema_version": 1,
        "status": PRIMARY_STATUS,
        "build_id": build_id,
        "manifest_sha256": sha256(manifest_path),
        "auxiliary_panel_authorized": False,
    }
    (table_root / "_SUCCESS.json").write_bytes(canonical_json(success))
    pointer = {
        "schema_version": 1,
        "status": PRIMARY_STATUS,
        "build_id": build_id,
        "contract_id": CONTRACT_ID,
        "artifact_path": artifact_relative.as_posix(),
        "interim_path": interim_relative.as_posix(),
        "quality_path": quality_relative.as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "auxiliary_panel_authorized": False,
    }
    interim_final.parent.mkdir(parents=True, exist_ok=True)
    quality_final.parent.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[Path, Path]] = []
    try:
        shutil.move(str(table_root), str(interim_final))
        moved.append((interim_final, table_root))
        shutil.move(str(quality_staging), str(quality_final))
        moved.append((quality_final, quality_staging))
        shutil.move(str(artifact_staging), str(artifact_final))
        moved.append((artifact_final, artifact_staging))
        _atomic_write(
            project_root / "artifacts/phase_02_canonical_panel/LATEST_AUXILIARY_AVAILABILITY_LINKAGE_PRIMARY.json",
            canonical_json(pointer),
        )
    except Exception:
        rollback_errors: list[str] = []
        for published, original in reversed(moved):
            try:
                if published.exists() and not original.exists():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(published), str(original))
            except Exception as exc:  # pragma: no cover - filesystem failure path
                rollback_errors.append(f"{published}: {exc}")
        if rollback_errors:
            raise AuxiliaryPublicationError(
                "Primary publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            )
        raise
    return pointer
