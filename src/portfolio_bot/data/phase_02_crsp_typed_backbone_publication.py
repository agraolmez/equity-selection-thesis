"""Deterministic immutable publication for the Phase 2 typed CRSP backbone."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.data import phase_02_crsp_typed_backbone as builder
from portfolio_bot.data import secure_publication as secure


SCHEMA_VERSION = 1
STATUS = "typed_backbone_primary_passed_pending_independent_verification"
INTERIM_FILES = (
    "_SUCCESS.json",
    "crsp_identity_interval.parquet",
    "crsp_security_day.parquet",
    "crsp_security_day_return_staging.parquet",
    "crsp_security_month_crosscheck.parquet",
    "crsp_security_month_return_staging.parquet",
    "crsp_shares_interval.parquet",
)
QUALITY_FILES = (
    "_SUCCESS.json",
    "build_contract.json",
    "checks.csv",
    "column_profiles.csv",
    "downstream_gate_status.json",
    "input_receipt.json",
    "join_coverage.csv",
    "parquet_metadata.csv",
    "summary.json",
    "table_counts.csv",
)
ARTIFACT_FILES = ("manifest.json", "output_hashes.sha256")
PRIMARY_POINTER = "LATEST_CRSP_TYPED_BACKBONE_PRIMARY.json"
BUILD_IMPLEMENTATION_LABELS = (
    "scripts/build_phase_02_crsp_typed_backbone.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_core.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_discovery.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_publication.py",
    "src/portfolio_bot/data/phase_02_crsp_typed_backbone_transforms.py",
    "src/portfolio_bot/data/secure_publication.py",
)
BUILD_ID_DOMAIN = b"phase02-crsp-typed-primary-build-v1\x00"


class TypedBackbonePublicationError(RuntimeError):
    """Raised when the primary publication cannot be proven complete."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TypedBackbonePublicationError("Value is not canonical JSON") from exc
    return (rendered + "\n").encode("utf-8")


def canonical_csv_bytes(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: (
                    json.dumps(
                        row[field],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    if isinstance(row[field], (dict, list, tuple, bool))
                    else row[field]
                )
                for field in fieldnames
            }
        )
    return buffer.getvalue().encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _implementation_bundle(settings: builder.TypedBackboneSettings) -> str:
    digest = hashlib.sha256()
    digest.update(b"phase02-crsp-typed-primary-implementation-v1\x00")
    for label in BUILD_IMPLEMENTATION_LABELS:
        path = settings.project_root / label
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise TypedBackbonePublicationError(
                f"Cannot read build implementation component {label}"
            ) from exc
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def primary_binding(settings: builder.TypedBackboneSettings) -> dict[str, Any]:
    return {
        "contract_id": settings.policy["publication"]["contract_id"],
        "identity_empirical_lock_sha256": settings.policy["upstream"][
            "identity_empirical_lock"
        ]["sha256"],
        "implementation_bundle_sha256": _implementation_bundle(settings),
        "policy_sha256": builder.LOCKED_POLICY_SHA256,
        "schema_id": settings.registry.schema_id,
        "schema_sha256": builder.LOCKED_SCHEMA_SHA256,
        "scope": {
            "start_date": settings.start_date.isoformat(),
            "quality_end_date": settings.end_date.isoformat(),
        },
        "sources": {
            name: {
                "filename": spec.filename,
                "full_file_rows": spec.full_file_rows,
                "sha256": spec.sha256,
                "size_bytes": spec.size_bytes,
            }
            for name, spec in sorted(settings.sources.items())
        },
    }


def build_id_for(settings: builder.TypedBackboneSettings) -> str:
    digest = hashlib.sha256()
    digest.update(BUILD_ID_DOMAIN)
    digest.update(canonical_json_bytes(primary_binding(settings)))
    return f"crsp-typed-{digest.hexdigest()[:24]}"


def _replace_template(template: str, build_id: str) -> Path:
    value = template.replace("<build_id>", build_id)
    if "<" in value or ">" in value:
        raise TypedBackbonePublicationError("Publication template is unresolved")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TypedBackbonePublicationError("Publication template is unsafe")
    return path


def publication_paths(
    settings: builder.TypedBackboneSettings, build_id: str
) -> dict[str, Path]:
    publication = settings.policy["publication"]
    interim_relative = _replace_template(
        str(publication["interim_namespace_template"]), build_id
    )
    quality_relative = _replace_template(
        str(publication["quality_namespace_template"]), build_id
    )
    artifact_relative = Path(str(publication["artifact_subdirectory"])) / build_id
    return {
        "interim": settings.database_root / interim_relative,
        "quality": settings.database_root / quality_relative,
        "artifact": settings.artifact_root / artifact_relative,
        "pointer": settings.artifact_root / str(publication["primary_pointer"]),
    }


def _relative_output(path: Path, settings: builder.TypedBackboneSettings) -> str:
    try:
        return path.relative_to(settings.database_root).as_posix()
    except ValueError:
        try:
            return path.relative_to(settings.project_root).as_posix()
        except ValueError as exc:
            raise TypedBackbonePublicationError(
                "Published output is outside the locked roots"
            ) from exc


def _parquet_validator(
    settings: builder.TypedBackboneSettings,
    metadata: dict[str, dict[str, Any]],
) -> secure.FileValidator:
    by_filename = {
        builder.PARQUET_FILENAMES[name]: spec
        for name, spec in settings.registry.tables.items()
    }

    def validate(name: str, descriptor: int, record: secure.FileRecord) -> None:
        if name == "_SUCCESS.json":
            return
        spec = by_filename.get(name)
        if spec is None:
            raise TypedBackbonePublicationError(f"Unexpected Parquet file {name}")
        duplicate = os.dup(descriptor)
        try:
            with os.fdopen(duplicate, "rb", buffering=0) as handle:
                parquet = pq.ParquetFile(pa.PythonFile(handle, mode="r"))
                arrow_schema = parquet.schema_arrow
                if not arrow_schema.equals(spec.arrow_schema(), check_metadata=True):
                    raise TypedBackbonePublicationError(
                        f"Parquet schema differs for {name}"
                    )
                file_metadata = parquet.metadata
                if file_metadata.num_rows != spec.expected_rows:
                    raise TypedBackbonePublicationError(
                        f"Parquet row count differs for {name}"
                    )
                row_group_rows = [
                    file_metadata.row_group(index).num_rows
                    for index in range(file_metadata.num_row_groups)
                ]
                if any(
                    rows != builder.ROW_GROUP_SIZE for rows in row_group_rows[:-1]
                ) or (
                    row_group_rows
                    and not 0 < row_group_rows[-1] <= builder.ROW_GROUP_SIZE
                ):
                    raise TypedBackbonePublicationError(
                        f"Parquet row-group sizes differ for {name}"
                    )
                compression_codecs: set[str] = set()
                statistics_complete = True
                utf8_dictionary_complete = True
                utf8_indexes = {
                    index
                    for index, field in enumerate(spec.fields)
                    if field.logical_type == "utf8"
                }
                for row_group_index in range(file_metadata.num_row_groups):
                    row_group = file_metadata.row_group(row_group_index)
                    for column_index in range(row_group.num_columns):
                        column = row_group.column(column_index)
                        compression_codecs.add(column.compression)
                        statistics_complete &= column.statistics is not None
                        if column_index in utf8_indexes:
                            utf8_dictionary_complete &= bool(
                                {"RLE_DICTIONARY", "PLAIN_DICTIONARY"}
                                & set(column.encodings)
                            )
                if compression_codecs != {"ZSTD"}:
                    raise TypedBackbonePublicationError(
                        f"Parquet compression differs for {name}"
                    )
                if not statistics_complete:
                    raise TypedBackbonePublicationError(
                        f"Parquet statistics are absent for {name}"
                    )
                if utf8_indexes and not utf8_dictionary_complete:
                    raise TypedBackbonePublicationError(
                        f"Parquet UTF-8 dictionary encoding differs for {name}"
                    )
                key_value_metadata = file_metadata.metadata or {}
                store_schema = b"ARROW:schema" in key_value_metadata
                if not store_schema:
                    raise TypedBackbonePublicationError(
                        f"Parquet stored Arrow schema is absent for {name}"
                    )
                metadata[name] = {
                    "compression_codecs": sorted(compression_codecs),
                    "data_page_version_proof": (
                        "writer_configuration_bound_by_implementation_sha256"
                    ),
                    "dictionary_encode_utf8": utf8_dictionary_complete,
                    "file_size_bytes": record.size_bytes,
                    "format_version": file_metadata.format_version,
                    "key_value_metadata_keys": sorted(
                        key.decode("ascii", errors="strict")
                        for key in key_value_metadata
                    ),
                    "row_count": file_metadata.num_rows,
                    "row_group_count": file_metadata.num_row_groups,
                    "row_group_rows": row_group_rows,
                    "schema_fingerprint": spec.fingerprint(),
                    "sha256": record.sha256,
                    "statistics_complete": statistics_complete,
                    "store_schema": store_schema,
                }
        except TypedBackbonePublicationError:
            raise
        except (OSError, pa.ArrowException) as exc:
            raise TypedBackbonePublicationError(
                f"Cannot validate Parquet file {name}"
            ) from exc

    return validate


def _check_rows(
    settings: builder.TypedBackboneSettings,
    evidence: builder.BuildEvidence,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, category: str, expected: Any, actual: Any) -> None:
        status = "PASS" if expected == actual else "FAIL"
        checks.append(
            {
                "actual": actual,
                "category": category,
                "check_id": check_id,
                "expected": expected,
                "message": (
                    "Locked expectation reproduced."
                    if status == "PASS"
                    else "Locked expectation differs."
                ),
                "status": status,
            }
        )

    for table_name, spec in sorted(settings.registry.tables.items()):
        add(
            f"TABLE_ROWS_{table_name}",
            "table_count",
            spec.expected_rows,
            evidence.table_rows[table_name],
        )
        key = evidence.key_quality.get(table_name, {})
        add(
            f"KEY_ROWS_{table_name}",
            "primary_key",
            evidence.table_rows[table_name],
            key.get("rows"),
        )
        for metric in ("null_rows", "duplicate_rows", "sorted_violations"):
            add(
                f"KEY_{metric.upper()}_{table_name}",
                "primary_key",
                0,
                key.get(metric),
            )
        absence = evidence.consumer_return_absence.get(table_name)
        if absence is not None:
            add(
                f"CONSUMER_RETURN_COLUMNS_ABSENT_{table_name}",
                "return_quarantine",
                0,
                absence.get("prohibited_column_count"),
            )
        for field in spec.fields:
            add(
                f"NULL_CONSERVATION_{table_name}_{field.name}",
                "raw_to_typed_null",
                evidence.raw_expected_nulls[table_name][field.name],
                evidence.table_nulls[table_name][field.name],
            )
    for source, metrics in sorted(evidence.canonicalization.items()):
        spec = settings.sources[source]
        add(
            f"SOURCE_FULL_ROWS_{source}",
            "source_count",
            spec.full_file_rows,
            metrics.full_rows,
        )
        add(
            f"SOURCE_RETAINED_ROWS_{source}",
            "source_count",
            spec.retained_rows,
            metrics.retained_rows,
        )
        add(
            f"SOURCE_CANONICAL_ROWS_{source}",
            "canonicalization",
            spec.canonical_rows,
            metrics.canonical_rows,
        )
        add(
            f"SOURCE_CONFLICTING_GROUPS_{source}",
            "canonicalization",
            0,
            metrics.conflicting_groups,
        )
    canonical = settings.policy["canonicalization"]
    daily = evidence.canonicalization["daily"]
    monthly = evidence.canonicalization["monthly"]
    for check_id, expected, actual in (
        (
            "DAILY_EXACT_DUPLICATE_GROUPS",
            canonical["daily"]["exact_duplicate_groups"],
            daily.duplicate_groups,
        ),
        (
            "DAILY_DUPLICATE_SURPLUS_ROWS",
            canonical["daily"]["duplicate_surplus_rows"],
            daily.duplicate_surplus_rows,
        ),
        (
            "MONTHLY_EXACT_RAW_KEY_DUPLICATE_GROUPS",
            canonical["monthly"]["exact_raw_key_duplicate_groups"],
            monthly.duplicate_groups,
        ),
        (
            "MONTHLY_EXACT_RAW_KEY_DUPLICATE_SURPLUS_ROWS",
            canonical["monthly"]["exact_raw_key_duplicate_surplus_rows"],
            monthly.duplicate_surplus_rows,
        ),
        (
            "MONTHLY_ROWS_AFTER_EXACT_RAW_KEY_COLLAPSE",
            canonical["monthly"]["rows_after_exact_raw_key_collapse"],
            monthly.retained_rows - monthly.duplicate_surplus_rows,
        ),
        (
            "MONTHLY_EFFECTIVE_INTERVAL_CANDIDATE_GROUPS",
            canonical["monthly"]["effective_interval_candidate_groups"],
            monthly.candidate_groups,
        ),
        (
            "MONTHLY_CANDIDATE_SIZE_HISTOGRAM",
            {
                str(canonical["monthly"]["candidate_group_size"]): canonical["monthly"][
                    "effective_interval_candidate_groups"
                ]
            },
            {
                str(size): count
                for size, count in sorted(monthly.candidate_size_histogram.items())
            },
        ),
    ):
        add(check_id, "canonicalization", expected, actual)

    for name in ("identity", "shares"):
        expected = canonical[name]
        actual = evidence.interval_structure.get(name, {})
        for field in (
            "retained_rows",
            "unique_permnos",
            "duplicate_key_groups",
            "overlap_edges",
            "gap_edges",
        ):
            locked = (
                expected["output_rows"] if field == "retained_rows" else expected[field]
            )
            add(
                f"INTERVAL_{name.upper()}_{('UNIQUE_SECURITIES' if field == 'unique_permnos' else field.upper())}",
                "interval_structure",
                locked,
                actual.get(field),
            )
        if "contiguous_edges" in expected:
            add(
                f"INTERVAL_{name.upper()}_CONTIGUOUS_EDGES",
                "interval_structure",
                expected["contiguous_edges"],
                actual.get("contiguous_edges"),
            )

    for join_name, expected in sorted(settings.policy["joins"].items()):
        if join_name not in evidence.joins:
            continue
        actual = evidence.joins[join_name]
        for field, locked in sorted(expected.items()):
            if type(locked) is int:
                add(
                    f"JOIN_{join_name}_{field}",
                    "join_coverage",
                    locked,
                    actual.get(field),
                )

    for table_name, columns in sorted(evidence.decimal_roundtrip.items()):
        for column_name, counts in sorted(columns.items()):
            add(
                f"DECIMAL_EXACT_MISMATCH_{table_name}_{column_name}",
                "exact_decimal_roundtrip",
                0,
                counts["mismatch_rows"],
            )
            add(
                f"DECIMAL_EXACT_CONSERVATION_{table_name}_{column_name}",
                "exact_decimal_roundtrip",
                counts["source_nonblank_rows"],
                counts["exact_match_rows"],
            )

    for formula_id, counts in sorted(evidence.formula_quality.items()):
        add(
            f"FORMULA_EXACT_MISMATCH_{formula_id}",
            "nominal_unit_identity",
            0,
            counts["mismatch_rows"],
        )
        add(
            f"FORMULA_NULL_CONSERVATION_{formula_id}",
            "nominal_unit_identity",
            counts["expected_null_rows"],
            counts["actual_null_rows"],
        )

    return_split = canonical["return_split"]
    split_tables = {
        "daily": ("crsp_security_day", "crsp_security_day_return_staging"),
        "monthly": (
            "crsp_security_month_crosscheck",
            "crsp_security_month_return_staging",
        ),
    }
    for frequency, (consumer_name, staging_name) in split_tables.items():
        counts = evidence.return_split[frequency]
        expected_rows = return_split[f"{frequency}_consumer_rows"]
        for metric in ("raw_rows", "consumer_rows", "return_staging_rows"):
            add(
                f"RETURN_SPLIT_{frequency.upper()}_{metric.upper()}",
                "return_quarantine",
                expected_rows,
                counts[metric],
            )
        add(
            f"{frequency.upper()}_CONSUMER_RETURN_KEYS_EQUAL",
            "return_quarantine",
            evidence.key_sha256[consumer_name],
            evidence.key_sha256[staging_name],
        )

    for table_name in (
        "crsp_security_day_return_staging",
        "crsp_security_month_return_staging",
    ):
        guards = evidence.quarantine_guards[table_name]
        add(
            f"QUARANTINE_ROWS_{table_name}",
            "return_quarantine",
            evidence.table_rows[table_name],
            guards["rows"],
        )
        for metric in (
            "status_not_unreconciled_rows",
            "authorized_true_rows",
            "status_null_rows",
            "authorized_null_rows",
        ):
            add(
                f"QUARANTINE_{metric.upper()}_{table_name}",
                "return_quarantine",
                0,
                guards[metric],
            )

    for table_name in (
        "crsp_security_day",
        "crsp_security_day_return_staging",
        "crsp_security_month_crosscheck",
        "crsp_security_month_return_staging",
    ):
        scope = evidence.date_scope[table_name]
        add(
            f"DATE_SCOPE_ROWS_{table_name}",
            "date_scope",
            evidence.table_rows[table_name],
            scope["rows"],
        )
        for metric in ("before_scope_rows", "after_scope_rows"):
            add(
                f"DATE_SCOPE_{metric.upper()}_{table_name}",
                "date_scope",
                0,
                scope[metric],
            )

    prohibited_true = sum(
        value is True
        for value in settings.policy["downstream"][
            "before_typed_independent_verification"
        ].values()
    )
    add(
        "DECEMBER_2014_PROHIBITED_DOWNSTREAM_AUTHORIZATIONS",
        "date_boundary",
        0,
        prohibited_true,
    )
    add(
        "EXACT_TYPED_TABLE_SET",
        "schema",
        sorted(settings.registry.tables),
        sorted(evidence.table_rows),
    )

    parquet_policy = settings.policy["publication"]["parquet"]
    for table_name, spec in sorted(settings.registry.tables.items()):
        filename = builder.PARQUET_FILENAMES[table_name]
        actual = evidence.parquet.get(filename, {})
        expected_row_groups = [builder.ROW_GROUP_SIZE] * (
            spec.expected_rows // builder.ROW_GROUP_SIZE
        )
        remainder = spec.expected_rows % builder.ROW_GROUP_SIZE
        if remainder:
            expected_row_groups.append(remainder)
        expected_metadata_keys = sorted(
            {"ARROW:schema"}
            | {
                key.decode("ascii", errors="strict")
                for key in (spec.arrow_schema().metadata or {})
            }
        )
        for suffix, locked, observed in (
            (
                "SCHEMA_FINGERPRINT",
                spec.fingerprint(),
                actual.get("schema_fingerprint"),
            ),
            (
                "FORMAT_VERSION",
                parquet_policy["format_version"],
                actual.get("format_version"),
            ),
            ("ROW_COUNT", spec.expected_rows, actual.get("row_count")),
            ("ROW_GROUP_ROWS", expected_row_groups, actual.get("row_group_rows")),
            ("COMPRESSION", ["ZSTD"], actual.get("compression_codecs")),
            ("STATISTICS", True, actual.get("statistics_complete")),
            ("DICTIONARY_UTF8", True, actual.get("dictionary_encode_utf8")),
            ("STORE_SCHEMA", True, actual.get("store_schema")),
            (
                "KEY_VALUE_METADATA",
                expected_metadata_keys,
                actual.get("key_value_metadata_keys"),
            ),
        ):
            add(
                f"PARQUET_{suffix}_{table_name}",
                "parquet",
                locked,
                observed,
            )
        add(
            f"PARQUET_DATA_PAGE_VERSION_{table_name}",
            "parquet",
            "writer_configuration_bound_by_implementation_sha256",
            actual.get("data_page_version_proof"),
        )
    return sorted(checks, key=lambda row: row["check_id"])


def render_quality_payloads(
    settings: builder.TypedBackboneSettings,
    *,
    build_id: str,
    binding: Mapping[str, Any],
    evidence: builder.BuildEvidence,
    interim_path: Path,
    interim_snapshot: secure.BundleSnapshot,
) -> dict[str, bytes]:
    checks = _check_rows(settings, evidence)
    if any(row["status"] != "PASS" for row in checks):
        raise TypedBackbonePublicationError("A primary quality check failed")
    table_counts = [
        {
            "expected_rows": settings.registry.tables[name].expected_rows,
            "key_sha256": evidence.key_sha256[name],
            "row_count": evidence.table_rows[name],
            "table": name,
        }
        for name in sorted(settings.registry.tables)
    ]
    column_profiles = [
        {
            "column": field.name,
            "decimal_arbitrary_precision_fallback_rows": evidence.decimal_roundtrip[
                table_name
            ][field.name]["arbitrary_precision_fallback_rows"],
            "decimal_exact_match_rows": evidence.decimal_roundtrip[table_name][
                field.name
            ]["exact_match_rows"],
            "decimal_mismatch_rows": evidence.decimal_roundtrip[table_name][field.name][
                "mismatch_rows"
            ],
            "non_null_count": evidence.table_rows[table_name]
            - evidence.table_nulls[table_name][field.name],
            "null_count": evidence.table_nulls[table_name][field.name],
            "null_conservation_status": (
                "PASS"
                if evidence.raw_expected_nulls[table_name][field.name]
                == evidence.table_nulls[table_name][field.name]
                else "FAIL"
            ),
            "nullable": field.nullable,
            "raw_expected_null_count": evidence.raw_expected_nulls[table_name][
                field.name
            ],
            "row_count": evidence.table_rows[table_name],
            "table": table_name,
        }
        for table_name, spec in sorted(settings.registry.tables.items())
        for field in spec.fields
    ]
    join_fields = (
        "join_name",
        "population_rows",
        "blank_as_of_date_rows",
        "blank_observation_date_rows",
        "nonblank_as_of_date_rows",
        "nonblank_observation_date_rows",
        "one_match_rows",
        "zero_match_rows",
        "zero_match_rows_with_nonblank_observation_date",
        "zero_match_rows_with_nonblank_as_of_date",
        "total_rows_without_usable_share_match",
        "multiple_match_rows",
        "embedded_interval_exact_key_match_rows",
        "embedded_interval_exact_key_missing_rows",
        "as_of_interval_key_equal_rows",
        "as_of_interval_key_different_rows",
        "as_of_payload_all_fields_equal_rows",
        "as_of_payload_one_or_more_fields_different_rows",
    )
    joins = [
        {
            field: (name if field == "join_name" else values.get(field, 0))
            for field in join_fields
        }
        for name, values in sorted(evidence.joins.items())
    ]
    parquet_rows = [
        {
            "file": name,
            "compression_codecs": values["compression_codecs"],
            "data_page_version_proof": values["data_page_version_proof"],
            "dictionary_encode_utf8": values["dictionary_encode_utf8"],
            "format_version": values["format_version"],
            "key_value_metadata_keys": values["key_value_metadata_keys"],
            "row_count": values["row_count"],
            "row_group_count": values["row_group_count"],
            "row_group_rows": values["row_group_rows"],
            "schema_fingerprint": values["schema_fingerprint"],
            "sha256": values["sha256"],
            "size_bytes": values["file_size_bytes"],
            "statistics_complete": values["statistics_complete"],
            "store_schema": values["store_schema"],
        }
        for name, values in sorted(evidence.parquet.items())
    ]
    input_receipt = {
        "binding": dict(binding),
        "build_id": build_id,
        "interim_path": _relative_output(interim_path, settings),
        "interim_snapshot": [
            {
                "name": record.name,
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
            }
            for record in interim_snapshot.files
        ],
        "schema_version": SCHEMA_VERSION,
    }
    summary = {
        "build_id": build_id,
        "check_count": len(checks),
        "fail_count": 0,
        "pass_count": len(checks),
        "status": STATUS,
        "table_count": len(table_counts),
        "total_rows": sum(row["row_count"] for row in table_counts),
    }
    success = {
        "build_id": build_id,
        "consumer_access_authorized": False,
        "independent_verification_required": True,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
    }
    return {
        "_SUCCESS.json": canonical_json_bytes(success),
        "build_contract.json": canonical_json_bytes(
            {
                "binding": dict(binding),
                "build_id": build_id,
                "parquet_policy": dict(settings.policy["publication"]["parquet"]),
                "primary_aggregate_evidence": {
                    "canonicalization": {
                        name: metrics.as_dict()
                        for name, metrics in sorted(evidence.canonicalization.items())
                    },
                    "consumer_return_absence": evidence.consumer_return_absence,
                    "date_scope": {
                        name: dict(values)
                        for name, values in sorted(evidence.date_scope.items())
                    },
                    "formula_quality": {
                        name: dict(values)
                        for name, values in sorted(evidence.formula_quality.items())
                    },
                    "interval_structure": evidence.interval_structure,
                    "key_quality": evidence.key_quality,
                    "quarantine_guards": {
                        name: dict(values)
                        for name, values in sorted(evidence.quarantine_guards.items())
                    },
                    "return_split": {
                        name: dict(values)
                        for name, values in sorted(evidence.return_split.items())
                    },
                },
                "schema_version": SCHEMA_VERSION,
                "table_schema_fingerprints": {
                    name: spec.fingerprint()
                    for name, spec in sorted(settings.registry.tables.items())
                },
            }
        ),
        "checks.csv": canonical_csv_bytes(
            ("check_id", "category", "expected", "actual", "status", "message"),
            checks,
        ),
        "column_profiles.csv": canonical_csv_bytes(
            (
                "table",
                "column",
                "row_count",
                "null_count",
                "non_null_count",
                "nullable",
                "raw_expected_null_count",
                "null_conservation_status",
                "decimal_exact_match_rows",
                "decimal_mismatch_rows",
                "decimal_arbitrary_precision_fallback_rows",
            ),
            column_profiles,
        ),
        "downstream_gate_status.json": canonical_json_bytes(
            settings.policy["downstream"]["before_typed_independent_verification"]
        ),
        "input_receipt.json": canonical_json_bytes(input_receipt),
        "join_coverage.csv": canonical_csv_bytes(join_fields, joins),
        "parquet_metadata.csv": canonical_csv_bytes(
            (
                "file",
                "row_count",
                "row_group_count",
                "row_group_rows",
                "format_version",
                "schema_fingerprint",
                "sha256",
                "size_bytes",
                "compression_codecs",
                "statistics_complete",
                "dictionary_encode_utf8",
                "store_schema",
                "key_value_metadata_keys",
                "data_page_version_proof",
            ),
            parquet_rows,
        ),
        "summary.json": canonical_json_bytes(summary),
        "table_counts.csv": canonical_csv_bytes(
            ("table", "expected_rows", "row_count", "key_sha256"), table_counts
        ),
    }


def _write_payload_bundle(
    *,
    trusted_root: Path,
    destination: Path,
    expected_files: Sequence[str],
    payloads: Mapping[str, bytes],
    label: str,
) -> secure.PublishedBundle:
    if set(payloads) != set(expected_files):
        raise TypedBackbonePublicationError(f"{label} payload set differs")
    with secure.SecureStagingDirectory.create(
        trusted_root=trusted_root,
        destination=destination,
        expected_files=expected_files,
        label=label,
    ) as stage:
        for name in sorted(payloads):
            with stage.open_exclusive_binary(name) as stream:
                stream.write(payloads[name])
        snapshot = stage.snapshot()
        return stage.publish(expected_hashes=snapshot.hashes)


def _output_records(
    settings: builder.TypedBackboneSettings,
    path: Path,
    snapshot: secure.BundleSnapshot,
) -> list[dict[str, Any]]:
    return [
        {
            "path": _relative_output(path / record.name, settings),
            "sha256": record.sha256,
            "size_bytes": record.size_bytes,
        }
        for record in snapshot.files
    ]


def _inventory_bytes(outputs: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [
        f"{row['sha256']}  {row['path']}\n"
        for row in sorted(outputs, key=lambda row: row["path"])
    ]
    return "".join(lines).encode("ascii")


def publish_primary(
    settings: builder.TypedBackboneSettings | None = None,
    *,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build, publish, and point to one primary candidate; verification stays closed."""

    settings = settings or builder.load_production_settings()
    progress = progress or (lambda _stage, _details: None)
    build_id = build_id_for(settings)
    binding = primary_binding(settings)
    paths = publication_paths(settings, build_id)
    parquet_metadata: dict[str, dict[str, Any]] = {}
    interim_success = canonical_json_bytes(
        {
            "build_id": build_id,
            "independent_verification_required": True,
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
        }
    )
    with secure.SecureStagingDirectory.create(
        trusted_root=settings.database_root,
        destination=paths["interim"],
        expected_files=INTERIM_FILES,
        label="typed-backbone interim bundle",
    ) as interim_stage:
        with contextlib.ExitStack() as stack:
            streams = {
                name: stack.enter_context(
                    interim_stage.open_exclusive_binary(builder.PARQUET_FILENAMES[name])
                )
                for name in builder.TABLE_NAMES
            }
            evidence = builder.compile_typed_tables(
                settings, streams, progress=progress
            )
        with interim_stage.open_exclusive_binary("_SUCCESS.json") as stream:
            stream.write(interim_success)
        validator = _parquet_validator(settings, parquet_metadata)
        interim_snapshot = interim_stage.snapshot(file_validator=validator)
        evidence.parquet = parquet_metadata
        interim_bundle = interim_stage.publish(
            expected_hashes=interim_snapshot.hashes,
            file_validator=validator,
        )
    progress("interim_bundle_published", {"build_id": build_id})

    quality_payloads = render_quality_payloads(
        settings,
        build_id=build_id,
        binding=binding,
        evidence=evidence,
        interim_path=paths["interim"],
        interim_snapshot=interim_bundle.snapshot,
    )
    quality_bundle = _write_payload_bundle(
        trusted_root=settings.database_root,
        destination=paths["quality"],
        expected_files=QUALITY_FILES,
        payloads=quality_payloads,
        label="typed-backbone quality bundle",
    )
    progress("quality_bundle_published", {"build_id": build_id})

    outputs = sorted(
        _output_records(settings, paths["interim"], interim_bundle.snapshot)
        + _output_records(settings, paths["quality"], quality_bundle.snapshot),
        key=lambda row: row["path"],
    )
    manifest = {
        "binding": binding,
        "build_id": build_id,
        "contract_id": settings.policy["publication"]["contract_id"],
        "outputs": outputs,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "summary": json.loads(quality_payloads["summary.json"]),
    }
    artifact_payloads = {
        "manifest.json": canonical_json_bytes(manifest),
        "output_hashes.sha256": _inventory_bytes(outputs),
    }
    artifact_bundle = _write_payload_bundle(
        trusted_root=settings.project_root,
        destination=paths["artifact"],
        expected_files=ARTIFACT_FILES,
        payloads=artifact_payloads,
        label="typed-backbone artifact bundle",
    )
    progress("artifact_bundle_published", {"build_id": build_id})

    pointer = {
        "artifact_path": _relative_output(paths["artifact"], settings),
        "build_id": build_id,
        "contract_id": settings.policy["publication"]["contract_id"],
        "interim_path": _relative_output(paths["interim"], settings),
        "manifest_sha256": artifact_bundle.snapshot.hashes["manifest.json"],
        "output_hashes_sha256": artifact_bundle.snapshot.hashes["output_hashes.sha256"],
        "quality_path": _relative_output(paths["quality"], settings),
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
    }

    with contextlib.ExitStack() as guards:
        interim_guard = guards.enter_context(
            secure.AuthenticatedBundleGuard.open(
                trusted_root=settings.database_root,
                path=paths["interim"],
                expected_files=INTERIM_FILES,
                expected_hashes=interim_bundle.snapshot.hashes,
                file_validator=_parquet_validator(settings, {}),
                label="typed-backbone published interim bundle",
                _permit_authenticated_directory_ctime_refresh=True,
            )
        )
        quality_guard = guards.enter_context(
            secure.AuthenticatedBundleGuard.open(
                trusted_root=settings.database_root,
                path=paths["quality"],
                expected_files=QUALITY_FILES,
                expected_hashes=quality_bundle.snapshot.hashes,
                label="typed-backbone published quality bundle",
                _permit_authenticated_directory_ctime_refresh=True,
            )
        )
        artifact_guard = guards.enter_context(
            secure.AuthenticatedBundleGuard.open(
                trusted_root=settings.project_root,
                path=paths["artifact"],
                expected_files=ARTIFACT_FILES,
                expected_hashes=artifact_bundle.snapshot.hashes,
                label="typed-backbone published artifact bundle",
                _permit_authenticated_directory_ctime_refresh=True,
            )
        )

        def gate() -> None:
            interim_guard.gate()
            quality_guard.gate()
            artifact_guard.gate()
            if (
                _implementation_bundle(settings)
                != binding["implementation_bundle_sha256"]
            ):
                raise TypedBackbonePublicationError(
                    "Executed implementation changed before primary pointer"
                )
            if (
                builder._sha256_file(settings.policy_path)
                != builder.LOCKED_POLICY_SHA256
            ):
                raise TypedBackbonePublicationError(
                    "Typed-backbone policy changed before primary pointer"
                )
            if (
                builder._sha256_file(settings.schema_path)
                != builder.LOCKED_SCHEMA_SHA256
            ):
                raise TypedBackbonePublicationError(
                    "Typed-backbone schema changed before primary pointer"
                )
            if artifact_bundle.snapshot.hashes["manifest.json"] != _sha256_bytes(
                artifact_payloads["manifest.json"]
            ):
                raise TypedBackbonePublicationError(
                    "Manifest changed before primary pointer"
                )

        secure.replace_pointer_last(
            paths["pointer"],
            canonical_json_bytes(pointer),
            trusted_root=settings.project_root,
            pre_replace_gate=gate,
            post_replace_gate=gate,
            label="typed-backbone primary pointer",
        )
    progress("primary_pointer_published", {"build_id": build_id})
    return pointer
