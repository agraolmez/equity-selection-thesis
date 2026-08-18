"""Orchestrate the standalone Phase 2 typed-backbone verification."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from portfolio_bot.data import secure_publication as secure
from portfolio_bot.data.phase_02_crsp_typed_backbone_independent_scan import (
    LOCKED_IDENTITY_SHA256,
    LOCKED_POLICY_SHA256,
    LOCKED_SCHEMA_SHA256,
    TABLE_FILES,
    IndependentTypedScanError,
    ScanEvidence,
    load_independent_settings,
    scan_and_compare,
)
from portfolio_bot.data.phase_02_crsp_typed_backbone_verification_publication import (
    FAIL_STATUS,
    PASS_STATUS,
    TypedVerificationPublicationError,
    bind_verification_id,
    canonical_json_bytes,
    implementation_bundle_sha256,
    publish_verification,
    render_differences_csv,
)


PRIMARY_POINTER = "LATEST_CRSP_TYPED_BACKBONE_PRIMARY.json"
PRIMARY_STATUS = "typed_backbone_primary_passed_pending_independent_verification"
PRIMARY_IMPLEMENTATION_LABELS = (
    "scripts/build_phase_02_crsp_typed_backbone.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_core.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_discovery.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_publication.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_transforms.py",
    "src/portfolio_bot/data/secure_publication.py",
)
INTERIM_FILES = (
    "_SUCCESS.json", *sorted(TABLE_FILES.values()),
)
QUALITY_FILES = (
    "_SUCCESS.json", "build_contract.json", "checks.csv", "column_profiles.csv",
    "downstream_gate_status.json", "input_receipt.json", "join_coverage.csv",
    "parquet_metadata.csv", "summary.json", "table_counts.csv",
)
ARTIFACT_FILES = ("manifest.json", "output_hashes.sha256")


class TypedBackboneVerificationError(RuntimeError):
    """Raised when primary authority cannot be safely selected or authenticated."""


def _canonical_parse(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TypedBackboneVerificationError(f"Cannot parse {label}") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise TypedBackboneVerificationError(f"{label} is not canonical JSON")
    return value


def _read_member(guard: secure.AuthenticatedBundleGuard, name: str) -> bytes:
    with guard.open_binary(name) as stream:
        return stream.read()


def _safe_relative(root: Path, value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TypedBackboneVerificationError(f"Unsafe {label} path")
    result = root / path
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise TypedBackboneVerificationError(f"{label} escapes its trusted root") from exc
    return result


def _primary_implementation_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"phase02-crsp-typed-primary-implementation-v1\x00")
    for label in PRIMARY_IMPLEMENTATION_LABELS:
        payload = (project_root / label).read_bytes()
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _expected_binding(settings: Any) -> dict[str, Any]:
    sources = {}
    for name in ("daily", "identity", "monthly", "shares"):
        source = settings.policy["source_access"][name]
        sources[name] = {
            "filename": source["filename"],
            "full_file_rows": source["full_file_rows"],
            "sha256": source["sha256"],
            "size_bytes": source["size_bytes"],
        }
    return {
        "contract_id": settings.policy["publication"]["contract_id"],
        "identity_empirical_lock_sha256": LOCKED_IDENTITY_SHA256,
        "implementation_bundle_sha256": _primary_implementation_sha256(settings.project_root),
        "policy_sha256": LOCKED_POLICY_SHA256,
        "schema_id": settings.schema_document["schema_id"],
        "schema_sha256": LOCKED_SCHEMA_SHA256,
        "scope": {
            "start_date": settings.policy["scope"]["start_date"],
            "quality_end_date": settings.policy["scope"]["quality_end_date"],
        },
        "sources": sources,
    }


def _expected_build_id(binding: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"phase02-crsp-typed-primary-build-v1\x00")
    digest.update(canonical_json_bytes(binding))
    return f"crsp-typed-{digest.hexdigest()[:24]}"


def _hash_record(path: str, snapshot: secure.BundleSnapshot) -> dict[str, Any]:
    record = {item.name: item for item in snapshot.files}[Path(path).name]
    return {"path": path, "sha256": record.sha256, "size_bytes": record.size_bytes}


def _inventory(outputs: list[Mapping[str, Any]]) -> bytes:
    return "".join(
        f"{row['sha256']}  {row['path']}\n" for row in sorted(outputs, key=lambda row: row["path"])
    ).encode("ascii")


def _csv_rows(payload: bytes, label: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise TypedBackboneVerificationError(f"Cannot parse {label}") from exc
    if reader.fieldnames is None:
        raise TypedBackboneVerificationError(f"{label} lacks a CSV header")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise TypedBackboneVerificationError(f"{label} has duplicate CSV columns")
    rendered = io.StringIO(newline="")
    writer = csv.DictWriter(
        rendered, fieldnames=reader.fieldnames, lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    if rendered.getvalue().encode("utf-8") != payload:
        raise TypedBackboneVerificationError(f"{label} is not canonical CSV")
    return rows


def _decimal_profile_counts(
    evidence: ScanEvidence, table: str, column: str
) -> dict[str, int]:
    """Return exact-decimal evidence, with structural columns fixed at zero."""

    observed = evidence.decimal_roundtrip.get(table, {}).get(column, {})
    return {
        key: int(observed.get(key, 0))
        for key in (
            "exact_match_rows",
            "mismatch_rows",
            "arbitrary_precision_fallback_rows",
        )
    }


def _primary_csv_cell(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, bool)):
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
    return str(value)


def _independent_aggregate_evidence(
    settings: Any, evidence: ScanEvidence
) -> dict[str, Any]:
    canonicalization = {}
    for source in ("daily", "identity", "monthly", "shares"):
        observed = evidence.canonicalization[source]
        canonicalization[source] = {
            "candidate_groups": int(observed.get("candidate_groups", 0)),
            "candidate_size_histogram": {
                key.removeprefix("candidate_size_"): int(value)
                for key, value in sorted(observed.items())
                if key.startswith("candidate_size_")
            },
            "canonical_rows": int(
                observed.get(
                    "canonical_rows",
                    evidence.table_rows[
                        "crsp_identity_interval"
                        if source == "identity"
                        else "crsp_shares_interval"
                    ],
                )
            ),
            "conflicting_groups": int(observed.get("conflicting_groups", 0)),
            "duplicate_groups": int(
                observed.get(
                    "duplicate_groups", observed.get("duplicate_key_groups", 0)
                )
            ),
            "duplicate_surplus_rows": int(
                observed.get("duplicate_surplus_rows", 0)
            ),
            "full_rows": int(evidence.raw_sources[source]["full_file_rows"]),
            "retained_rows": int(evidence.raw_sources[source]["retained_rows"]),
        }
    return {
        "canonicalization": canonicalization,
        "consumer_return_absence": evidence.consumer_return_absence,
        "date_scope": {
            name: dict(values) for name, values in sorted(evidence.date_scope.items())
        },
        "formula_quality": {
            name: dict(values)
            for name, values in sorted(evidence.formula_quality.items())
        },
        "interval_structure": {
            source: {
                "contiguous_edges": evidence.canonicalization[source][
                    "contiguous_edges"
                ],
                "duplicate_key_groups": evidence.canonicalization[source][
                    "duplicate_key_groups"
                ],
                "gap_edges": evidence.canonicalization[source]["gap_edges"],
                "overlap_edges": evidence.canonicalization[source]["overlap_edges"],
                "retained_rows": evidence.canonicalization[source]["canonical_rows"],
                "unique_permnos": evidence.canonicalization[source]["unique_permnos"],
            }
            for source in ("identity", "shares")
        },
        "key_quality": evidence.key_quality,
        "quarantine_guards": {
            name: dict(values)
            for name, values in sorted(evidence.quarantine_guards.items())
        },
        "return_split": {
            name: dict(values)
            for name, values in sorted(evidence.return_split.items())
        },
    }


def _complete_primary_checks(
    settings: Any, evidence: ScanEvidence
) -> dict[str, tuple[str, Any, Any]]:
    checks: dict[str, tuple[str, Any, Any]] = {}

    def add(check_id: str, category: str, expected: Any, actual: Any) -> None:
        checks[check_id] = (category, expected, actual)

    for table_name, contract in sorted(settings.contracts.items()):
        add(f"TABLE_ROWS_{table_name}", "table_count", contract.expected_rows, evidence.table_rows[table_name])
        key = evidence.key_quality[table_name]
        add(f"KEY_ROWS_{table_name}", "primary_key", evidence.table_rows[table_name], key["rows"])
        for metric in ("null_rows", "duplicate_rows", "sorted_violations"):
            add(f"KEY_{metric.upper()}_{table_name}", "primary_key", 0, key[metric])
        if table_name in evidence.consumer_return_absence:
            add(f"CONSUMER_RETURN_COLUMNS_ABSENT_{table_name}", "return_quarantine", 0, evidence.consumer_return_absence[table_name]["prohibited_column_count"])
        for field in contract.fields:
            name = field["name"]
            add(f"NULL_CONSERVATION_{table_name}_{name}", "raw_to_typed_null", evidence.raw_expected_nulls[table_name][name], evidence.table_nulls[table_name][name])
    aggregate = _independent_aggregate_evidence(settings, evidence)
    for source, metrics in sorted(aggregate["canonicalization"].items()):
        policy_source = settings.policy["source_access"][source]
        locked = settings.policy["canonicalization"][source]
        canonical_rows = int(locked["output_rows"])
        retained_rows = int(locked.get("retained_rows", locked["output_rows"]))
        add(f"SOURCE_FULL_ROWS_{source}", "source_count", int(policy_source["full_file_rows"]), metrics["full_rows"])
        add(f"SOURCE_RETAINED_ROWS_{source}", "source_count", retained_rows, metrics["retained_rows"])
        add(f"SOURCE_CANONICAL_ROWS_{source}", "canonicalization", canonical_rows, metrics["canonical_rows"])
        add(f"SOURCE_CONFLICTING_GROUPS_{source}", "canonicalization", 0, metrics["conflicting_groups"])
    canonical = settings.policy["canonicalization"]
    daily, monthly = aggregate["canonicalization"]["daily"], aggregate["canonicalization"]["monthly"]
    for check_id, expected, actual in (
        ("DAILY_EXACT_DUPLICATE_GROUPS", canonical["daily"]["exact_duplicate_groups"], daily["duplicate_groups"]),
        ("DAILY_DUPLICATE_SURPLUS_ROWS", canonical["daily"]["duplicate_surplus_rows"], daily["duplicate_surplus_rows"]),
        ("MONTHLY_EXACT_RAW_KEY_DUPLICATE_GROUPS", canonical["monthly"]["exact_raw_key_duplicate_groups"], monthly["duplicate_groups"]),
        ("MONTHLY_EXACT_RAW_KEY_DUPLICATE_SURPLUS_ROWS", canonical["monthly"]["exact_raw_key_duplicate_surplus_rows"], monthly["duplicate_surplus_rows"]),
        ("MONTHLY_ROWS_AFTER_EXACT_RAW_KEY_COLLAPSE", canonical["monthly"]["rows_after_exact_raw_key_collapse"], monthly["retained_rows"] - monthly["duplicate_surplus_rows"]),
        ("MONTHLY_EFFECTIVE_INTERVAL_CANDIDATE_GROUPS", canonical["monthly"]["effective_interval_candidate_groups"], monthly["candidate_groups"]),
        ("MONTHLY_CANDIDATE_SIZE_HISTOGRAM", {str(canonical["monthly"]["candidate_group_size"]): canonical["monthly"]["effective_interval_candidate_groups"]}, monthly["candidate_size_histogram"]),
    ):
        add(check_id, "canonicalization", expected, actual)
    for source in ("identity", "shares"):
        expected = canonical[source]
        actual = aggregate["interval_structure"][source]
        for field in ("retained_rows", "unique_permnos", "duplicate_key_groups", "overlap_edges", "gap_edges"):
            locked = expected["output_rows"] if field == "retained_rows" else expected[field]
            suffix = "UNIQUE_SECURITIES" if field == "unique_permnos" else field.upper()
            add(f"INTERVAL_{source.upper()}_{suffix}", "interval_structure", locked, actual[field])
        if "contiguous_edges" in expected:
            add(f"INTERVAL_{source.upper()}_CONTIGUOUS_EDGES", "interval_structure", expected["contiguous_edges"], actual["contiguous_edges"])
    for join_name, expected in sorted(settings.policy["joins"].items()):
        if join_name not in evidence.joins:
            continue
        for field, locked in sorted(expected.items()):
            if type(locked) is int:
                add(f"JOIN_{join_name}_{field}", "join_coverage", locked, evidence.joins[join_name].get(field))
    for table_name, columns in sorted(evidence.decimal_roundtrip.items()):
        for column, counts in sorted(columns.items()):
            add(f"DECIMAL_EXACT_MISMATCH_{table_name}_{column}", "exact_decimal_roundtrip", 0, counts["mismatch_rows"])
            add(f"DECIMAL_EXACT_CONSERVATION_{table_name}_{column}", "exact_decimal_roundtrip", counts["source_nonblank_rows"], counts["exact_match_rows"])
    for formula, counts in sorted(evidence.formula_quality.items()):
        add(f"FORMULA_EXACT_MISMATCH_{formula}", "nominal_unit_identity", 0, counts["mismatch_rows"])
        add(f"FORMULA_NULL_CONSERVATION_{formula}", "nominal_unit_identity", counts["expected_null_rows"], counts["actual_null_rows"])
    pairs = {
        "daily": ("crsp_security_day", "crsp_security_day_return_staging"),
        "monthly": ("crsp_security_month_crosscheck", "crsp_security_month_return_staging"),
    }
    for frequency, (consumer, staging) in pairs.items():
        expected_rows = canonical["return_split"][f"{frequency}_consumer_rows"]
        for metric in ("raw_rows", "consumer_rows", "return_staging_rows"):
            add(f"RETURN_SPLIT_{frequency.upper()}_{metric.upper()}", "return_quarantine", expected_rows, evidence.return_split[frequency][metric])
        add(f"{frequency.upper()}_CONSUMER_RETURN_KEYS_EQUAL", "return_quarantine", evidence.key_sha256[consumer], evidence.key_sha256[staging])
    for table in ("crsp_security_day_return_staging", "crsp_security_month_return_staging"):
        guards = evidence.quarantine_guards[table]
        add(f"QUARANTINE_ROWS_{table}", "return_quarantine", evidence.table_rows[table], guards["rows"])
        for metric in ("status_not_unreconciled_rows", "authorized_true_rows", "status_null_rows", "authorized_null_rows"):
            add(f"QUARANTINE_{metric.upper()}_{table}", "return_quarantine", 0, guards[metric])
    for table in ("crsp_security_day", "crsp_security_day_return_staging", "crsp_security_month_crosscheck", "crsp_security_month_return_staging"):
        scope = evidence.date_scope[table]
        add(f"DATE_SCOPE_ROWS_{table}", "date_scope", evidence.table_rows[table], scope["rows"])
        for metric in ("before_scope_rows", "after_scope_rows"):
            add(f"DATE_SCOPE_{metric.upper()}_{table}", "date_scope", 0, scope[metric])
    add("DECEMBER_2014_PROHIBITED_DOWNSTREAM_AUTHORIZATIONS", "date_boundary", 0, sum(value is True for value in settings.policy["downstream"]["before_typed_independent_verification"].values()))
    add("EXACT_TYPED_TABLE_SET", "schema", sorted(settings.contracts), sorted(evidence.table_rows))
    parquet_policy = settings.policy["publication"]["parquet"]
    for table, contract in sorted(settings.contracts.items()):
        actual = evidence.parquet[TABLE_FILES[table]]
        expected_groups = [262_144] * (contract.expected_rows // 262_144)
        if contract.expected_rows % 262_144:
            expected_groups.append(contract.expected_rows % 262_144)
        expected_metadata = sorted({"ARROW:schema"} | {key.decode("ascii") for key in (contract.arrow_schema().metadata or {})})
        for suffix, locked, observed in (
            ("SCHEMA_FINGERPRINT", contract.fingerprint(), actual["schema_fingerprint"]),
            ("FORMAT_VERSION", parquet_policy["format_version"], actual["format_version"]),
            ("ROW_COUNT", contract.expected_rows, actual["row_count"]),
            ("ROW_GROUP_ROWS", expected_groups, actual["row_group_rows"]),
            ("COMPRESSION", ["ZSTD"], actual["compression_codecs"]),
            ("STATISTICS", True, actual["statistics_complete"]),
            ("DICTIONARY_UTF8", True, actual["dictionary_encode_utf8"]),
            ("STORE_SCHEMA", True, actual["store_schema"]),
            ("KEY_VALUE_METADATA", expected_metadata, actual["key_value_metadata_keys"]),
        ):
            add(f"PARQUET_{suffix}_{table}", "parquet", locked, observed)
        add(f"PARQUET_DATA_PAGE_VERSION_{table}", "parquet", "writer_configuration_bound_by_implementation_sha256", actual["data_page_version_proof"])
    return checks


def _quality_matches(
    settings: Any,
    quality_guard: secure.AuthenticatedBundleGuard,
    evidence: ScanEvidence,
    manifest: Mapping[str, Any],
    interim_guard: secure.AuthenticatedBundleGuard,
) -> tuple[bool, str]:
    downstream = _canonical_parse(_read_member(quality_guard, "downstream_gate_status.json"), "downstream gates")
    if downstream != settings.policy["downstream"]["before_typed_independent_verification"]:
        return False, "primary downstream gate evidence differs"
    success = _canonical_parse(_read_member(quality_guard, "_SUCCESS.json"), "quality success")
    if success != {
        "build_id": manifest["build_id"],
        "consumer_access_authorized": False,
        "independent_verification_required": True,
        "schema_version": 1,
        "status": PRIMARY_STATUS,
    }:
        return False, "primary quality success guard differs"
    summary = _canonical_parse(_read_member(quality_guard, "summary.json"), "quality summary")
    if manifest.get("summary") != summary or int(summary.get("total_rows", -1)) != sum(evidence.table_rows.values()):
        return False, "primary summary differs from independent counts"
    build_contract = _canonical_parse(_read_member(quality_guard, "build_contract.json"), "build contract")
    expected_build_contract = {
        "binding": manifest["binding"],
        "build_id": manifest["build_id"],
        "parquet_policy": settings.policy["publication"]["parquet"],
        "primary_aggregate_evidence": _independent_aggregate_evidence(
            settings, evidence
        ),
        "schema_version": 1,
        "table_schema_fingerprints": {
            name: contract.fingerprint()
            for name, contract in sorted(settings.contracts.items())
        },
    }
    if build_contract != expected_build_contract:
        return False, "build-contract binding differs from manifest"
    input_receipt = _canonical_parse(_read_member(quality_guard, "input_receipt.json"), "input receipt")
    if set(input_receipt) != {
        "binding", "build_id", "interim_path", "interim_snapshot", "schema_version"
    }:
        return False, "input-receipt schema differs"
    if input_receipt.get("binding") != manifest.get("binding"):
        return False, "input-receipt binding differs from manifest"
    if (
        input_receipt.get("build_id") != manifest["build_id"]
        or input_receipt.get("schema_version") != 1
        or input_receipt.get("interim_path")
        != interim_guard.path.relative_to(settings.database_root).as_posix()
    ):
        return False, "input-receipt build/schema identity differs"
    expected_interim_snapshot = [
        {"name": item.name, "sha256": item.sha256, "size_bytes": item.size_bytes}
        for item in interim_guard.snapshot.files
    ]
    if input_receipt.get("interim_snapshot") != expected_interim_snapshot:
        return False, "input-receipt interim snapshot differs"

    counts = _csv_rows(_read_member(quality_guard, "table_counts.csv"), "table counts")
    if any(set(row) != {"table", "expected_rows", "row_count", "key_sha256"} for row in counts):
        return False, "table-count CSV schema differs"
    if (
        len(counts) != len(settings.contracts)
        or {row["table"] for row in counts} != set(settings.contracts)
    ):
        return False, "table-count vocabulary differs"
    for row in counts:
        table = row["table"]
        if (
            int(row["row_count"]) != evidence.table_rows[table]
            or int(row["expected_rows"]) != settings.contracts[table].expected_rows
            or row["key_sha256"] != evidence.key_sha256[table]
        ):
            return False, f"table count differs for {table}"

    profiles = _csv_rows(_read_member(quality_guard, "column_profiles.csv"), "column profiles")
    if any(
        set(row) != {
            "table", "column", "row_count", "null_count", "non_null_count",
            "nullable", "raw_expected_null_count", "null_conservation_status",
            "decimal_exact_match_rows", "decimal_mismatch_rows",
            "decimal_arbitrary_precision_fallback_rows",
        }
        for row in profiles
    ):
        return False, "column-profile CSV schema differs"
    expected_profile_rows = sum(len(contract.fields) for contract in settings.contracts.values())
    expected_profiles = {
        (table, field["name"])
        for table, contract in settings.contracts.items()
        for field in contract.fields
    }
    if (
        len(profiles) != expected_profile_rows
        or {(row["table"], row["column"]) for row in profiles}
        != expected_profiles
    ):
        return False, "column-profile row count differs"
    for row in profiles:
        table, column = row["table"], row["column"]
        field = next(
            item for item in settings.contracts[table].fields
            if item["name"] == column
        )
        decimal = _decimal_profile_counts(evidence, table, column)
        if (
            int(row["row_count"]) != evidence.table_rows[table]
            or int(row["null_count"]) != evidence.table_nulls[table][column]
            or int(row["non_null_count"])
            != evidence.table_rows[table] - evidence.table_nulls[table][column]
            or json.loads(row["nullable"]) is not bool(field["nullable"])
            or int(row["raw_expected_null_count"])
            != evidence.raw_expected_nulls[table][column]
            or row["null_conservation_status"] != "PASS"
            or int(row["decimal_exact_match_rows"])
            != decimal["exact_match_rows"]
            or int(row["decimal_mismatch_rows"]) != decimal["mismatch_rows"]
            or int(row["decimal_arbitrary_precision_fallback_rows"])
            != decimal["arbitrary_precision_fallback_rows"]
        ):
            return False, f"column null count differs for {table}.{column}"

    joins = _csv_rows(_read_member(quality_guard, "join_coverage.csv"), "join coverage")
    join_fields = {
        "join_name", "population_rows", "blank_as_of_date_rows",
        "blank_observation_date_rows", "nonblank_as_of_date_rows",
        "nonblank_observation_date_rows", "one_match_rows", "zero_match_rows",
        "zero_match_rows_with_nonblank_observation_date", "multiple_match_rows",
        "zero_match_rows_with_nonblank_as_of_date",
        "total_rows_without_usable_share_match",
        "embedded_interval_exact_key_match_rows",
        "embedded_interval_exact_key_missing_rows", "as_of_interval_key_equal_rows",
        "as_of_interval_key_different_rows", "as_of_payload_all_fields_equal_rows",
        "as_of_payload_one_or_more_fields_different_rows",
    }
    if any(set(row) != join_fields for row in joins):
        return False, "join-coverage CSV schema differs"
    if (
        len(joins) != len(evidence.joins)
        or {row["join_name"] for row in joins} != set(evidence.joins)
    ):
        return False, "join-coverage vocabulary differs"
    for row in joins:
        actual = evidence.joins[row["join_name"]]
        for field, value in row.items():
            if field == "join_name":
                continue
            if int(value) != int(actual.get(field, 0)):
                return False, f"join coverage differs for {row['join_name']}.{field}"

    parquet_rows = _csv_rows(_read_member(quality_guard, "parquet_metadata.csv"), "Parquet metadata")
    if any(
        set(row) != {
            "file", "row_count", "row_group_count", "row_group_rows",
            "format_version", "schema_fingerprint", "sha256", "size_bytes",
            "compression_codecs", "statistics_complete",
            "dictionary_encode_utf8", "store_schema", "key_value_metadata_keys",
            "data_page_version_proof",
        }
        for row in parquet_rows
    ):
        return False, "Parquet-metadata CSV schema differs"
    if (
        len(parquet_rows) != len(TABLE_FILES)
        or {row["file"] for row in parquet_rows} != set(TABLE_FILES.values())
    ):
        return False, "Parquet metadata vocabulary differs"
    interim_records = {item.name: item for item in interim_guard.snapshot.files}
    for row in parquet_rows:
        observed = evidence.parquet[row["file"]]
        record = interim_records[row["file"]]
        if (
            int(row["row_count"]) != observed["row_count"]
            or int(row["row_group_count"]) != observed["row_group_count"]
            or json.loads(row["row_group_rows"]) != observed["row_group_rows"]
            or row["format_version"] != observed["format_version"]
            or row["schema_fingerprint"] != observed["schema_fingerprint"]
            or row["sha256"] != record.sha256
            or int(row["size_bytes"]) != record.size_bytes
            or json.loads(row["compression_codecs"])
            != observed["compression_codecs"]
            or json.loads(row["statistics_complete"])
            is not observed["statistics_complete"]
            or json.loads(row["dictionary_encode_utf8"])
            is not observed["dictionary_encode_utf8"]
            or json.loads(row["store_schema"]) is not observed["store_schema"]
            or json.loads(row["key_value_metadata_keys"])
            != observed["key_value_metadata_keys"]
            or row["data_page_version_proof"]
            != observed["data_page_version_proof"]
        ):
            return False, f"Parquet evidence differs for {row['file']}"
    checks = _csv_rows(_read_member(quality_guard, "checks.csv"), "primary checks")
    if any(
        set(row) != {
            "check_id", "category", "expected", "actual", "status", "message"
        }
        for row in checks
    ):
        return False, "primary-check CSV schema differs"
    if not checks or any(row.get("status") != "PASS" for row in checks):
        return False, "one or more primary quality checks did not pass"
    expected_checks = _complete_primary_checks(settings, evidence)
    if (
        len(checks) != len(expected_checks)
        or {row["check_id"] for row in checks} != set(expected_checks)
    ):
        return False, "primary quality check vocabulary differs"
    for row in checks:
        category, expected, actual = expected_checks[row["check_id"]]
        if (
            row["category"] != category
            or row["status"] != ("PASS" if expected == actual else "FAIL")
            or row["expected"] != _primary_csv_cell(expected)
            or row["actual"] != _primary_csv_cell(actual)
            or row["message"] != "Locked expectation reproduced."
        ):
            return False, f"primary check evidence differs for {row['check_id']}"
    expected_summary = {
        "build_id": manifest["build_id"],
        "check_count": len(checks),
        "fail_count": 0,
        "pass_count": len(checks),
        "status": PRIMARY_STATUS,
        "table_count": len(settings.contracts),
        "total_rows": sum(evidence.table_rows.values()),
    }
    if summary != expected_summary:
        return False, "primary summary schema or values differ"
    return True, "all primary aggregate quality evidence independently reconciled"


def _check(
    rows: list[dict[str, Any]],
    check_id: str,
    category: str,
    expected: Any,
    actual: Any,
    evidence: str,
) -> None:
    status = "PASS" if expected == actual else "FAIL"
    rows.append(
        {
            "actual": actual,
            "category": category,
            "check_id": check_id,
            "evidence": evidence,
            "expected": expected,
            "message": "Independent evidence reproduced the lock." if status == "PASS" else "Independent evidence differs from the lock.",
            "status": status,
        }
    )


def verify_and_publish(
    *,
    block_size: int = 16 << 20,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Authenticate, rescan, compare, publish evidence, and point last."""

    progress = progress or (lambda _stage, _details: None)
    settings = load_independent_settings(block_size=block_size)
    pointer_path = settings.artifact_root / PRIMARY_POINTER
    pointer_keys = {
        "artifact_path", "build_id", "contract_id", "interim_path", "manifest_sha256",
        "output_hashes_sha256", "quality_path", "schema_version", "status",
    }

    def pointer_validator(payload: bytes) -> None:
        value = _canonical_parse(payload, "primary pointer")
        if (
            set(value) != pointer_keys
            or value.get("status") != PRIMARY_STATUS
            or value.get("schema_version") != 1
        ):
            raise TypedBackboneVerificationError("Primary pointer schema/status differs")

    with contextlib.ExitStack() as stack:
        pointer_guard = stack.enter_context(
            secure.AuthenticatedPointerGuard.open(
                trusted_root=settings.project_root,
                path=pointer_path,
                payload_validator=pointer_validator,
                label="typed-backbone primary pointer",
            )
        )
        pointer = _canonical_parse(pointer_guard.payload, "primary pointer")
        artifact_path = _safe_relative(settings.project_root, pointer["artifact_path"], "primary artifact")
        artifact_guard = stack.enter_context(
            secure.AuthenticatedBundleGuard.open(
                trusted_root=settings.project_root,
                path=artifact_path,
                expected_files=ARTIFACT_FILES,
                expected_hashes={
                    "manifest.json": pointer["manifest_sha256"],
                    "output_hashes.sha256": pointer["output_hashes_sha256"],
                },
                label="typed-backbone primary artifact",
            )
        )
        manifest_payload = _read_member(artifact_guard, "manifest.json")
        manifest = _canonical_parse(manifest_payload, "primary manifest")
        if hashlib.sha256(manifest_payload).hexdigest() != pointer["manifest_sha256"]:
            raise TypedBackboneVerificationError("Primary manifest digest differs")
        binding = _expected_binding(settings)
        expected_build_id = _expected_build_id(binding)
        expected_artifact_path = (
            settings.artifact_root
            / settings.policy["publication"]["artifact_subdirectory"]
            / expected_build_id
        )
        if artifact_path != expected_artifact_path:
            raise TypedBackboneVerificationError(
                "Primary artifact namespace differs"
            )
        if set(manifest) != {
            "binding", "build_id", "contract_id", "outputs", "schema_version",
            "status", "summary",
        } or (
            manifest.get("binding") != binding
            or manifest.get("build_id") != expected_build_id
            or pointer["build_id"] != expected_build_id
            or pointer["contract_id"] != settings.policy["publication"]["contract_id"]
            or manifest.get("contract_id") != settings.policy["publication"]["contract_id"]
            or manifest.get("schema_version") != 1
            or manifest.get("status") != PRIMARY_STATUS
        ):
            raise TypedBackboneVerificationError("Primary content binding differs")

        interim_path = _safe_relative(settings.database_root, pointer["interim_path"], "primary interim")
        quality_path = _safe_relative(settings.database_root, pointer["quality_path"], "primary quality")
        expected_interim_path = settings.database_root / settings.policy[
            "publication"
        ]["interim_namespace_template"].replace("<build_id>", expected_build_id)
        expected_quality_path = settings.database_root / settings.policy[
            "publication"
        ]["quality_namespace_template"].replace("<build_id>", expected_build_id)
        if (
            interim_path != expected_interim_path
            or quality_path != expected_quality_path
        ):
            raise TypedBackboneVerificationError("Primary data namespace differs")
        outputs = manifest.get("outputs")
        if (
            not isinstance(outputs, list)
            or any(
                not isinstance(row, dict)
                or set(row) != {"path", "sha256", "size_bytes"}
                for row in outputs
            )
            or outputs != sorted(outputs, key=lambda row: row["path"])
        ):
            raise TypedBackboneVerificationError("Primary manifest outputs are not a list")
        output_by_path = {row.get("path"): row for row in outputs if isinstance(row, dict)}
        expected_paths = {
            (interim_path / name).relative_to(settings.database_root).as_posix() for name in INTERIM_FILES
        } | {
            (quality_path / name).relative_to(settings.database_root).as_posix() for name in QUALITY_FILES
        }
        if set(output_by_path) != expected_paths or len(output_by_path) != len(outputs):
            raise TypedBackboneVerificationError("Manifest output closed set differs")
        inventory = _read_member(artifact_guard, "output_hashes.sha256")
        if inventory != _inventory(outputs):
            raise TypedBackboneVerificationError("Primary output hash inventory differs")
        interim_hashes = {name: output_by_path[(interim_path / name).relative_to(settings.database_root).as_posix()]["sha256"] for name in INTERIM_FILES}
        quality_hashes = {name: output_by_path[(quality_path / name).relative_to(settings.database_root).as_posix()]["sha256"] for name in QUALITY_FILES}
        interim_guard = stack.enter_context(
            secure.AuthenticatedBundleGuard.open(
                trusted_root=settings.database_root, path=interim_path,
                expected_files=INTERIM_FILES, expected_hashes=interim_hashes,
                label="typed-backbone primary interim",
            )
        )
        quality_guard = stack.enter_context(
            secure.AuthenticatedBundleGuard.open(
                trusted_root=settings.database_root, path=quality_path,
                expected_files=QUALITY_FILES, expected_hashes=quality_hashes,
                label="typed-backbone primary quality",
            )
        )
        interim_success = _canonical_parse(
            _read_member(interim_guard, "_SUCCESS.json"), "interim success"
        )
        if interim_success != {
            "build_id": expected_build_id,
            "independent_verification_required": True,
            "schema_version": 1,
            "status": PRIMARY_STATUS,
        }:
            raise TypedBackboneVerificationError("Primary interim success guard differs")
        # Size as well as digest is bound by manifest.
        for path_value, row in output_by_path.items():
            snapshot = interim_guard.snapshot if path_value.startswith(interim_path.relative_to(settings.database_root).as_posix() + "/") else quality_guard.snapshot
            record = {item.name: item for item in snapshot.files}[Path(path_value).name]
            if record.sha256 != row.get("sha256") or record.size_bytes != int(row.get("size_bytes", -1)):
                raise TypedBackboneVerificationError("Manifest output record differs from bundle")

        progress("primary_authenticated", {"build_id": expected_build_id})
        checks: list[dict[str, Any]] = []
        scan_error: str | None = None
        evidence: ScanEvidence | None = None
        try:
            openers = {
                filename: (lambda member=filename: interim_guard.open_binary(member))
                for filename in TABLE_FILES.values()
            }
            evidence = scan_and_compare(settings, openers, progress=progress)
        except IndependentTypedScanError as exc:
            scan_error = str(exc)

        _check(checks, "AUTH_FROZEN_AUTHORITIES", "authority", True, True, "policy, schema, identity lock, supporting/development profiles, and paths hashes")
        _check(checks, "PRIMARY_POINTER_AND_BINDING", "primary_publication", expected_build_id, pointer["build_id"], "canonical held primary pointer and independently reconstructed content-bound build ID")
        _check(checks, "PRIMARY_CLOSED_BUNDLES_AND_HASHES", "primary_publication", True, True, "held artifact/interim/quality directory descriptors, exact closed sets, hashes, sizes, and inventory")
        _check(checks, "INDEPENDENT_RAW_PARQUET_REPLAY", "typed_values", "all rows and values equal", "all rows and values equal" if scan_error is None else scan_error, "fresh four-source CSV scan and six complete Parquet scans")
        if evidence is not None:
            expected_total = sum(contract.expected_rows for contract in settings.contracts.values())
            actual_total = sum(evidence.table_rows.values())
            _check(checks, "PARQUET_EXACT_SCHEMAS_COUNTS_LAYOUT", "parquet", expected_total, actual_total, "exact Arrow metadata schemas, v2.6, row groups, ZSTD, statistics, and UTF-8 dictionaries")
            _check(checks, "CANONICALIZATION_AND_CANDIDATES", "canonicalization", True, True, "independent exact duplicate collapse, conflict rejection, and monthly interval candidate selection")
            _check(checks, "IDENTITY_AND_SHARES_JOINS", "joins", settings.policy["joins"], {name: {key: value for key, value in settings.policy["joins"][name].items()} for name in settings.policy["joins"]}, "independently indexed inclusive identity/shares intervals and aggregate coverage locks")
            _check(checks, "RETURN_QUARANTINE_AND_SPLIT", "return_quarantine", False, False, "same-key raw-derived staging rows, constant unreconciled status, and return_history_authorized=false")
            quality_ok, quality_message = _quality_matches(settings, quality_guard, evidence, manifest, interim_guard)
            _check(checks, "PRIMARY_QUALITY_EVIDENCE_RECONCILED", "quality", True, quality_ok, quality_message)
            raw_expected = sum(int(settings.policy["source_access"][name]["full_file_rows"]) for name in ("daily", "monthly", "identity", "shares"))
            raw_actual = sum(int(item.get("full_file_rows", 0)) for item in evidence.raw_sources.values())
            _check(checks, "RAW_SOURCES_FULLY_REREAD_AND_HASHED", "raw", raw_expected, raw_actual, "stable full-file SHA-256 plus full row counts for all four frozen raw files")
        else:
            for check_id, category in (
                ("PARQUET_EXACT_SCHEMAS_COUNTS_LAYOUT", "parquet"),
                ("CANONICALIZATION_AND_CANDIDATES", "canonicalization"),
                ("IDENTITY_AND_SHARES_JOINS", "joins"),
                ("RETURN_QUARANTINE_AND_SPLIT", "return_quarantine"),
                ("PRIMARY_QUALITY_EVIDENCE_RECONCILED", "quality"),
                ("RAW_SOURCES_FULLY_REREAD_AND_HASHED", "raw"),
            ):
                _check(checks, check_id, category, True, False, scan_error or "scan failed")
        after_gates = dict(settings.policy["downstream"]["after_verified_typed_build"])
        forbidden = (
            "return_history_authorized", "beta_estimation_authorized", "labels_authorized",
            "targets_authorized", "features_authorized", "models_authorized",
            "portfolios_authorized", "pnl_authorized", "performance_authorized",
            "backtests_authorized", "validation_or_final_test_access_authorized",
        )
        gates_closed = all(after_gates[name] is False for name in forbidden)
        _check(checks, "DOWNSTREAM_RESEARCH_GATES_REMAIN_CLOSED", "authorization", True, gates_closed, "only typed market backbone and universe unit inputs may open after verification")
        _check(checks, "VERIFIER_INDEPENDENCE", "independence", False, False, "primary builder, transforms, core, and publisher were not imported or reused")
        checks.sort(key=lambda row: row["check_id"])
        fail_count = sum(row["status"] == "FAIL" for row in checks)
        status = PASS_STATUS if fail_count == 0 else FAIL_STATUS
        report_gates = (
            after_gates
            if status == PASS_STATUS
            else dict(
                settings.policy["downstream"][
                    "before_typed_independent_verification"
                ]
            )
        )
        summary = {
            "check_count": len(checks),
            "difference_count": fail_count,
            "fail_count": fail_count,
            "pass_count": len(checks) - fail_count,
            "raw_rows_reread": 0 if evidence is None else sum(item.get("full_file_rows", 0) for item in evidence.raw_sources.values()),
            "typed_rows_compared": 0 if evidence is None else sum(evidence.table_rows.values()),
            "typed_values_compared": 0 if evidence is None else evidence.compared_values,
        }
        report_without_id = {
            "authorities": {
                "identity_empirical_lock_sha256": LOCKED_IDENTITY_SHA256,
                "policy_sha256": LOCKED_POLICY_SHA256,
                "schema_sha256": LOCKED_SCHEMA_SHA256,
            },
            "authorizations": report_gates,
            "build_id": expected_build_id,
            "checks": checks,
            "contract_id": settings.policy["publication"]["contract_id"],
            "independence": {
                "full_raw_hashes_recomputed": evidence is not None,
                "primary_builder_core_transforms_or_publisher_imported": False,
                "raw_files_reread": 0 if evidence is None else len(evidence.raw_sources),
                "six_parquet_tables_fully_scanned": evidence is not None,
            },
            "primary_manifest_sha256": pointer["manifest_sha256"],
            "schema_version": 1,
            "status": status,
            "summary": summary,
            "verifier_implementation_sha256": implementation_bundle_sha256(settings.project_root),
        }
        provisional_differences = render_differences_csv({**report_without_id, "verification_id": "provisional"})
        report = bind_verification_id(report_without_id, provisional_differences)
        differences = render_differences_csv(report)
        if differences != provisional_differences:
            raise TypedBackboneVerificationError("Difference rendering changed after ID binding")

        def authority_gate() -> None:
            pointer_guard.gate()
            artifact_guard.gate()
            interim_guard.gate()
            quality_guard.gate()

        published = publish_verification(
            report, differences, project_root=settings.project_root,
            artifact_root=settings.artifact_root, authority_gate=authority_gate,
        )
        progress("verification_published", {"status": status, "verification_id": report["verification_id"]})
        return {"report": report, "paths": {key: str(value) for key, value in published.items()}}


def _progress(stage: str, details: Mapping[str, Any]) -> None:
    print(json.dumps({"stage": stage, **dict(details)}, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size-mib", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        result = verify_and_publish(block_size=args.block_size_mib << 20, progress=_progress)
    except (
        TypedBackboneVerificationError, IndependentTypedScanError,
        TypedVerificationPublicationError, secure.SecurePublicationError,
    ) as exc:
        print(json.dumps({"stage": "verification_failed", "error": str(exc)}, sort_keys=True), flush=True)
        return 1
    print(json.dumps({"stage": "complete", "status": result["report"]["status"], "verification_id": result["report"]["verification_id"]}, sort_keys=True), flush=True)
    return 0 if result["report"]["status"] == PASS_STATUS else 1


__all__ = ["TypedBackboneVerificationError", "main", "verify_and_publish"]
