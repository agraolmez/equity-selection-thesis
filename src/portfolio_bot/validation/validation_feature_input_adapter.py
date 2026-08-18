"""Authenticated path adapter for empirical Stage 11 feature formulas.

The Stage 6 formula implementations consume one physical Parquet path per
source table.  Stage 11 deliberately keeps the verified development history
and the post-development extension as separate authorities.  This module is
the narrow bridge between those designs: it authenticates both authorities,
creates a sorted and uniquely keyed auxiliary-history merge, and returns the
exact 26-table ``FeatureStoreInputs`` catalog expected by the unchanged
formula adapters.

It never opens a label, forward return, portfolio, trade, P&L, or 2020+ row.
The 2014-11-28 OptionMetrics context spine is retained as a separate internal
catalog and is never included in the scored universe catalog.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data.phase_02_auxiliary_core import table_schema
from portfolio_bot.features.feature_store_inputs import (
    FeatureStoreInputs,
    InputTable,
    resolve_input_catalog,
)
from portfolio_bot.validation.validation_data_contract import (
    INTERNAL_CONTEXT_SIGNAL,
    LAST_SIGNAL,
    ValidationDataContract,
    validate_validation_data_contract,
)
from portfolio_bot.validation.validation_universe_auxiliary import (
    CONTEXT_LINKAGE_FILE,
    CONTEXT_UNIVERSE_FILE,
    RECEIPT_FILE,
    SCORED_LINKAGE_FILE,
    SCORED_UNIVERSE_FILE,
    POLICY_PATH as UNIVERSE_AUXILIARY_POLICY_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUXILIARY_SCHEMA_PATH = (
    PROJECT_ROOT
    / "configs/phase_02_canonicalization/auxiliary_source_schema_v1.json"
)
FIRST_EXTENSION_AVAILABILITY = dt.date(2014, 11, 1)
LAST_PERMITTED_EXIT = dt.date(2019, 12, 2)
INPUT_ADAPTER_ID = "stage11-verified-feature-input-adapter-v3"
FEATURE_MERGE_DUCKDB_THREADS = 6
FEATURE_MERGE_DUCKDB_MEMORY_LIMIT = "8GB"
_ZERO_ROW_AUXILIARY_MERGE_METADATA_KEY = (
    b"portfolio_bot.zero_row_auxiliary_merge_provenance"
)
_ZERO_ROW_AUXILIARY_MERGE_PROVENANCE_VERSION = 1

_FULL_CONTEXT_TABLES = (
    "crsp_security_day.parquet",
    "crsp_security_day_return.parquet",
    "crsp_identity_interval.parquet",
    "crsp_shares_interval.parquet",
    "ff_factor_daily_observation.parquet",
    "ff_factor_monthly_observation.parquet",
    "ps_liquidity_observation.parquet",
)
_FACTOR_TABLES = frozenset(
    {
        "ff_factor_daily_observation.parquet",
        "ff_factor_monthly_observation.parquet",
        "ps_liquidity_observation.parquet",
    }
)
_LINKAGE_TABLE = "auxiliary_signal_linkage.parquet"
_SPINE_TABLE = "crsp_signal_date_universe.parquet"
_CALENDAR_TABLE = "trading_calendar.parquet"
_REBALANCE_TABLE = "rebalance_calendar.parquet"


class ValidationFeatureInputAdapterError(RuntimeError):
    """Raised when the empirical feature-source bridge is not trustworthy."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFeatureInputAdapterError(message)


def _require_predictor_authority(authority: object) -> ValidationDataContract:
    """Revalidate the no-outcome technical contract, never outcome authority."""

    _check(
        type(authority) is ValidationDataContract,
        "Empirical predictor materialization requires the exact validation-data authority",
    )
    current = validate_validation_data_contract()
    assert isinstance(authority, ValidationDataContract)
    _check(
        authority.policy_sha256 == current.policy_sha256
        and authority.validation_signals == current.validation_signals
        and authority.raw_feature_schema.equals(
            current.raw_feature_schema, check_metadata=True
        )
        and authority.family_feature_names == current.family_feature_names,
        "Validation-data predictor authority is stale or differs",
    )
    return authority


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


def _json(path: Path, label: str) -> dict[str, Any]:
    _check(path.is_file() and not path.is_symlink(), f"Missing or unsafe {label}")
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationFeatureInputAdapterError(f"Invalid {label}") from exc
    _check(isinstance(value, dict), f"{label} is not an object")
    return value


def _sha256(path: Path) -> str:
    before = path.stat(follow_symlinks=False)
    _check(stat.S_ISREG(before.st_mode), f"Not an ordinary file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat(follow_symlinks=False)
    _check(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        f"File changed while hashing: {path}",
    )
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(_canonical_json(value))
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _provenance_zero_row_auxiliary_merge(
    path: Path,
    *,
    development: Path,
    extension: Path,
    destination_name: str,
    merge_operator: str,
) -> dict[str, Any]:
    """Make a fresh empty merge auditable without changing its field schema."""

    development_sha256 = _sha256(development)
    extension_sha256 = _sha256(extension)
    provenance = {
        "schema_version": _ZERO_ROW_AUXILIARY_MERGE_PROVENANCE_VERSION,
        "materialization": "independent_zero_row_auxiliary_merge",
        "adapter_id": INPUT_ADAPTER_ID,
        "destination_table": destination_name,
        "development_rows": 0,
        "extension_rows": 0,
        "development_sha256": development_sha256,
        "extension_sha256": extension_sha256,
        "merge_operator": merge_operator,
    }
    parquet = pq.ParquetFile(path)
    _check(
        parquet.metadata.num_rows == 0 and parquet.metadata.num_row_groups == 0,
        f"Zero-row auxiliary merge physical shape differs: {destination_name}",
    )
    field_schema = parquet.schema_arrow
    schema_metadata = dict(field_schema.metadata or {})
    schema_metadata[_ZERO_ROW_AUXILIARY_MERGE_METADATA_KEY] = _canonical_json(
        provenance
    )
    provenanced_schema = field_schema.with_metadata(schema_metadata)
    replacement = path.with_name(f".{path.name}.provenance-{uuid.uuid4().hex}")
    try:
        # Closing a ParquetWriter without writing a batch preserves the
        # DuckDB result's zero-row/zero-row-group shape.  Reusing its Arrow
        # schema preserves every field, type, nullability, and field metadata.
        with pq.ParquetWriter(
            replacement,
            provenanced_schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        ):
            pass
        observed = pq.ParquetFile(replacement)
        _check(
            observed.metadata.num_rows == 0
            and observed.metadata.num_row_groups == 0
            and observed.schema_arrow.remove_metadata().equals(
                field_schema.remove_metadata(), check_metadata=True
            )
            and (observed.schema_arrow.metadata or {}).get(
                _ZERO_ROW_AUXILIARY_MERGE_METADATA_KEY
            )
            == _canonical_json(provenance),
            f"Zero-row auxiliary merge provenance differs: {destination_name}",
        )
        os.replace(replacement, path)
    finally:
        if replacement.exists() or replacement.is_symlink():
            replacement.unlink()
    _check(
        _sha256(path) not in {development_sha256, extension_sha256},
        f"Zero-row auxiliary merge is not freshly materialized: {destination_name}",
    )
    return provenance


def _verify_parquet_receipt(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _check(path.is_file() and not path.is_symlink(), f"Missing or unsafe {label}")
    parquet = pq.ParquetFile(path)
    _check(int(receipt.get("rows", -1)) == parquet.metadata.num_rows, f"{label} rows differ")
    if "row_groups" in receipt:
        _check(
            int(receipt["row_groups"]) == parquet.metadata.num_row_groups,
            f"{label} row groups differ",
        )
    _check(int(receipt.get("size_bytes", -1)) == path.stat().st_size, f"{label} size differs")
    _check(receipt.get("sha256") == _sha256(path), f"{label} hash differs")


def _auxiliary_specs() -> dict[
    str,
    tuple[tuple[str, ...], tuple[str, ...], pa.Schema],
]:
    document = _json(AUXILIARY_SCHEMA_PATH, "frozen auxiliary schema")
    tables = document.get("tables")
    _check(isinstance(tables, list), "Frozen auxiliary schema tables differ")
    result: dict[
        str,
        tuple[tuple[str, ...], tuple[str, ...], pa.Schema],
    ] = {}
    for value in tables:
        _check(isinstance(value, dict), "Frozen auxiliary table specification differs")
        name = f"{value.get('table')}.parquet"
        primary = value.get("primary_key")
        order = value.get("sort_key")
        _check(
            isinstance(primary, list)
            and bool(primary)
            and isinstance(order, list)
            and bool(order)
            and all(isinstance(item, str) and item for item in (*primary, *order)),
            f"Frozen auxiliary keys differ: {name}",
        )
        result[name] = (
            tuple(primary),
            tuple(dict.fromkeys((*order, *primary))),
            table_schema(document, str(value.get("table"))),
        )
    expected = set(result) - _FACTOR_TABLES - {_LINKAGE_TABLE}
    _check(len(expected) == 14, "Auxiliary history table set is not the frozen fourteen")
    return {name: result[name] for name in sorted(expected)}


def _authenticate_universe_root(root: Path) -> dict[str, Any]:
    receipt = _json(root / RECEIPT_FILE, "validation universe receipt")
    _check(
        receipt.get("status") == "stage11_validation_universe_built_no_outcome_access",
        "Validation universe is not complete",
    )
    for key, filename in (
        ("scored", SCORED_UNIVERSE_FILE),
        ("internal_context", CONTEXT_UNIVERSE_FILE),
    ):
        value = receipt.get(key)
        _check(isinstance(value, dict) and value.get("path") == filename, f"Universe {key} receipt differs")
        _verify_parquet_receipt(root / filename, value, label=f"universe {key}")
    _check(
        receipt["scored"].get("signal_dates") == 59
        and receipt["scored"].get("maximum_signal_date") == LAST_SIGNAL.isoformat()
        and receipt["internal_context"].get("signal_dates") == 1
        and receipt["internal_context"].get("minimum_signal_date")
        == INTERNAL_CONTEXT_SIGNAL.isoformat(),
        "Validation universe scope differs",
    )
    return receipt


def _authenticate_auxiliary_extension(
    root: Path,
    *,
    development_auxiliary_root: Path,
) -> dict[str, Any]:
    receipt = _json(root / RECEIPT_FILE, "validation auxiliary receipt")
    _check(
        receipt.get("status")
        == "stage11_validation_auxiliary_normalization_built_no_outcomes",
        "Validation auxiliary extension is not complete",
    )
    bindings = receipt.get("bindings")
    _check(
        isinstance(bindings, dict)
        and Path(str(bindings.get("development_auxiliary_root"))).resolve()
        == development_auxiliary_root.resolve(),
        "Validation auxiliary development binding differs",
    )
    _check(
        bindings.get("policy_sha256") == _sha256(UNIVERSE_AUXILIARY_POLICY_PATH),
        "Validation auxiliary policy binding is stale or differs",
    )
    scope = receipt.get("scope")
    _check(
        isinstance(scope, dict)
        and scope.get("extension_start") == FIRST_EXTENSION_AVAILABILITY.isoformat()
        and scope.get("last_scored_signal") == LAST_SIGNAL.isoformat()
        and scope.get("factor_last_observation") == LAST_SIGNAL.isoformat(),
        "Validation auxiliary physical predictor scope differs",
    )
    checks = receipt.get("boundary_checks")
    _check(isinstance(checks, dict) and bool(checks), "Auxiliary boundary receipts are absent")
    tables = root / "tables"
    _check(tables.is_dir() and not tables.is_symlink(), "Auxiliary extension table root is unsafe")
    _check(
        {path.name for path in tables.iterdir() if path.is_file()}
        == set(checks),
        "Auxiliary extension directory is not its receipted closed table set",
    )
    for name, value in checks.items():
        _check(isinstance(value, dict), f"Auxiliary boundary receipt differs: {name}")
        maximum = value.get("maximum_gate_date")
        _check(
            maximum is None or dt.date.fromisoformat(str(maximum)) <= LAST_SIGNAL,
            f"Auxiliary boundary exceeds the no-outcome cutoff: {name}",
        )
        _verify_parquet_receipt(tables / name, value, label=f"auxiliary extension {name}")

    checkpoints = receipt.get("checkpoint_receipts")
    _check(isinstance(checkpoints, dict) and bool(checkpoints), "Auxiliary checkpoint receipts are absent")
    checkpoint_outputs: dict[str, Mapping[str, Any]] = {}
    for checkpoint, value in checkpoints.items():
        _check(
            isinstance(value, dict)
            and value.get("status") == "checkpoint_complete"
            and isinstance(value.get("outputs"), dict),
            f"Auxiliary checkpoint differs: {checkpoint}",
        )
        for name, output in value["outputs"].items():
            _check(name not in checkpoint_outputs and isinstance(output, dict), f"Auxiliary checkpoint output repeats: {name}")
            checkpoint_outputs[name] = output
            boundary = checks.get(name)
            _check(
                isinstance(boundary, dict)
                and all(
                    output.get(field) == boundary.get(field)
                    for field in ("rows", "row_groups", "sha256", "size_bytes")
                ),
                f"Checkpoint/boundary identity differs: {name}",
            )
    _check(set(checkpoint_outputs) == set(checks), "Checkpoint and boundary table sets differ")
    return receipt


def _authenticate_full_context(root: Path) -> dict[str, Any]:
    receipt = _json(root / "full_predictor_context_receipt.json", "full predictor context receipt")
    _check(
        receipt.get("status") == "stage11_full_predictor_context_composed_no_forward_access",
        "Full predictor context is not complete",
    )
    _check(
        receipt.get("historical_returns", {}).get("rows_after_last_signal") == 0
        and receipt.get("development_overlap_differences") == 0
        and receipt.get("duplicate_keys") == 0
        and receipt.get("market_return_key_differences") == 0,
        "Full predictor context QA differs",
    )
    receipts = receipt.get("tables")
    _check(isinstance(receipts, dict) and set(receipts) == set(_FULL_CONTEXT_TABLES), "Full predictor table set differs")
    tables = root / "tables"
    for name in _FULL_CONTEXT_TABLES:
        value = receipts[name]
        _check(isinstance(value, dict), f"Full predictor receipt differs: {name}")
        _verify_parquet_receipt(tables / name, value, label=f"full predictor context {name}")
    return receipt


def _date_bounds(path: Path, column: str) -> tuple[dt.date | None, dt.date | None]:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows == 0:
        return None, None
    minimum: dt.date | None = None
    maximum: dt.date | None = None
    for index in range(parquet.metadata.num_row_groups):
        values = parquet.read_row_group(index, columns=[column])[column]
        values = pc.drop_null(values)
        if len(values) == 0:
            continue
        low = pc.min(values).as_py()
        high = pc.max(values).as_py()
        minimum = low if minimum is None else min(minimum, low)
        maximum = high if maximum is None else max(maximum, high)
    return minimum, maximum


def _merge_auxiliary_table(
    connection: duckdb.DuckDBPyConnection,
    *,
    development: Path,
    extension: Path,
    destination: Path,
    expected_schema: pa.Schema,
    primary_key: Sequence[str],
    sort_key: Sequence[str],
) -> dict[str, Any]:
    development_file = pq.ParquetFile(development)
    extension_file = pq.ParquetFile(extension)
    development_schema = development_file.schema_arrow
    extension_schema = extension_file.schema_arrow
    _check(
        extension_schema.equals(expected_schema, check_metadata=False),
        f"Extension schema differs from frozen contract: {destination.name}",
    )
    _check(
        len(development_schema) == len(extension_schema)
        and all(
            development_field.name == extension_field.name
            and development_field.type == extension_field.type
            for development_field, extension_field in zip(
                development_schema,
                extension_schema,
            )
        ),
        f"Development/extension schema differs: {destination.name}",
    )
    development_contract_nullability_narrowings = tuple(
        development_field.name
        for development_field, contract_field in zip(
            development_schema,
            expected_schema,
            strict=True,
        )
        if not development_field.nullable and contract_field.nullable
    )
    _check(
        not development_contract_nullability_narrowings,
        f"Development/extension schema differs: {destination.name}",
    )
    development_physical_nullability_widened_columns = tuple(
        development_field.name
        for development_field, contract_field in zip(
            development_schema,
            expected_schema,
            strict=True,
        )
        if development_field.nullable and not contract_field.nullable
    )
    field_names = set(development_schema.names)
    _check(set(primary_key).issubset(field_names) and set(sort_key).issubset(field_names), f"Auxiliary merge keys differ: {destination.name}")
    dev_sql = f"read_parquet('{_sql_path(development)}')"
    ext_sql = f"read_parquet('{_sql_path(extension)}')"
    development_required_column_null_rows = 0
    development_required_column_null_rows_by_column: dict[str, int] = {}
    if development_physical_nullability_widened_columns:
        null_predicate = " OR ".join(
            f"{_quoted(name)} IS NULL"
            for name in development_physical_nullability_widened_columns
        )
        per_column_expressions = ", ".join(
            f"count(*) FILTER (WHERE {_quoted(name)} IS NULL)"
            for name in development_physical_nullability_widened_columns
        )
        null_counts = connection.execute(
            f"SELECT count(*) FILTER (WHERE {null_predicate}), "
            f"{per_column_expressions} FROM {dev_sql}"
        ).fetchone()
        development_required_column_null_rows = int(null_counts[0])
        development_required_column_null_rows_by_column = dict(
            zip(
                development_physical_nullability_widened_columns,
                (int(value) for value in null_counts[1:]),
                strict=True,
            )
        )
        _check(
            development_required_column_null_rows == 0
            and not any(development_required_column_null_rows_by_column.values()),
            f"Development required-column nulls violate extension schema: {destination.name}",
        )
    histories_are_date_disjoint = False
    if "public_availability_date" in field_names:
        _dev_min, dev_max = _date_bounds(development, "public_availability_date")
        ext_min, ext_max = _date_bounds(extension, "public_availability_date")
        _check(dev_max is None or dev_max < FIRST_EXTENSION_AVAILABILITY, f"Development auxiliary history crosses extension: {destination.name}")
        _check(ext_min is None or ext_min >= FIRST_EXTENSION_AVAILABILITY, f"Auxiliary extension overlaps development: {destination.name}")
        _check(ext_max is None or ext_max <= LAST_SIGNAL, f"Auxiliary extension exceeds validation scope: {destination.name}")
        histories_are_date_disjoint = dev_max is None or ext_min is None or dev_max < ext_min
    quoted_primary = ",".join(_quoted(name) for name in primary_key)
    quoted_order = ",".join(_quoted(name) for name in sort_key)
    for label, source in (("development", dev_sql), ("extension", ext_sql)):
        duplicates = int(
            connection.execute(
                f"SELECT count(*) FROM (SELECT {quoted_primary} FROM {source} GROUP BY {quoted_primary} HAVING count(*) > 1)"
            ).fetchone()[0]
        )
        _check(duplicates == 0, f"{label} auxiliary keys duplicate: {destination.name}")
    dev_rows = development_file.metadata.num_rows
    ext_rows = extension_file.metadata.num_rows
    join_predicate = " AND ".join(
        f'd.{_quoted(name)} IS NOT DISTINCT FROM e.{_quoted(name)}'
        for name in primary_key
    )
    overlap_rows = int(
        connection.execute(
            f"SELECT count(*) FROM {dev_sql} d JOIN {ext_sql} e ON {join_predicate}"
        ).fetchone()[0]
    )

    # Compustat source IDs are stable raw-record identities.  When the
    # extension normalizer sees one of those records again, its deliberately
    # truncated pre-extension SEC index can replace an already verified
    # filing date with a later fallback availability date.  The verified
    # development record remains the authoritative point-in-time version.
    # Such overlaps are admitted only when every non-timing source field is
    # identical and the extension availability is strictly later.  Any
    # changed economic/source value remains fatal.
    retiming_columns = tuple(
        name
        for name in (
            "public_availability_date",
            "availability_basis",
            "stale_after_date",
        )
        if name in field_names
    )
    non_timing_columns = tuple(
        name
        for name in development_file.schema_arrow.names
        if name not in set(primary_key) and name not in set(retiming_columns)
    )
    retiming_difference_rows = 0
    development_precedence_rows = 0
    interval_continuation_rows = 0
    if destination.name == "optionmetrics_name_interval.parquet" and overlap_rows:
        _check(
            "effective_end_date" in field_names,
            "OptionMetrics name-interval end date is absent",
        )
        interval_payload_columns = tuple(
            name
            for name in development_file.schema_arrow.names
            if name not in set(primary_key) | {"effective_end_date"}
        )
        interval_payload_difference = " OR ".join(
            f'd.{_quoted(name)} IS DISTINCT FROM e.{_quoted(name)}'
            for name in interval_payload_columns
        ) or "FALSE"
        interval_payload_difference_rows = int(
            connection.execute(
                f"SELECT count(*) FROM {dev_sql} d JOIN {ext_sql} e "
                f"ON {join_predicate} WHERE {interval_payload_difference}"
            ).fetchone()[0]
        )
        _check(
            interval_payload_difference_rows == 0,
            "OptionMetrics name-interval continuation changes identity payload",
        )
        interval_end_violations = int(
            connection.execute(
                f"SELECT count(*) FROM {dev_sql} d JOIN {ext_sql} e "
                f"ON {join_predicate} WHERE "
                f"e.effective_end_date < d.effective_end_date "
                f"OR e.effective_end_date > DATE '{LAST_SIGNAL.isoformat()}'"
            ).fetchone()[0]
        )
        _check(
            interval_end_violations == 0,
            "OptionMetrics name-interval continuation has an invalid end date",
        )
        interval_continuation_rows = overlap_rows
    if histories_are_date_disjoint and overlap_rows:
        non_timing_difference = " OR ".join(
            f'd.{_quoted(name)} IS DISTINCT FROM e.{_quoted(name)}'
            for name in non_timing_columns
        ) or "FALSE"
        non_timing_difference_rows = int(
            connection.execute(
                f"SELECT count(*) FROM {dev_sql} d JOIN {ext_sql} e "
                f"ON {join_predicate} WHERE {non_timing_difference}"
            ).fetchone()[0]
        )
        _check(
            non_timing_difference_rows == 0,
            f"Development/extension overlap changes non-timing payload: {destination.name}",
        )
        precedence_violations = int(
            connection.execute(
                f"SELECT count(*) FROM {dev_sql} d JOIN {ext_sql} e "
                f"ON {join_predicate} WHERE NOT "
                f"(d.public_availability_date < e.public_availability_date)"
            ).fetchone()[0]
        )
        _check(
            precedence_violations == 0,
            f"Development/extension overlap does not preserve earlier verified availability: {destination.name}",
        )
        retiming_difference = " OR ".join(
            f'd.{_quoted(name)} IS DISTINCT FROM e.{_quoted(name)}'
            for name in retiming_columns
        ) or "FALSE"
        retiming_difference_rows = int(
            connection.execute(
                f"SELECT count(*) FROM {dev_sql} d JOIN {ext_sql} e "
                f"ON {join_predicate} WHERE {retiming_difference}"
            ).fetchone()[0]
        )
        development_precedence_rows = overlap_rows

    # The large observation histories are proven date-disjoint above, so a
    # global DISTINCT would add a needless second external sort over hundreds
    # of millions of rows.  Stable-ID retiming overlaps use verified-
    # development precedence.  Interval authorities can overlap and retain
    # exact-row deduplication before their primary-key gate.
    if interval_continuation_rows:
        merge_operator = "EXTENSION_INTERVAL_CONTINUATION"
        development_select = (
            f"SELECT d.* FROM {dev_sql} d WHERE NOT EXISTS "
            f"(SELECT 1 FROM {ext_sql} e WHERE {join_predicate})"
        )
        combined_sql = f"{development_select} UNION ALL SELECT * FROM {ext_sql}"
    elif histories_are_date_disjoint and overlap_rows:
        merge_operator = "DEVELOPMENT_PRECEDENCE_ANTI_JOIN"
        extension_select = (
            f"SELECT e.* FROM {ext_sql} e WHERE NOT EXISTS "
            f"(SELECT 1 FROM {dev_sql} d WHERE {join_predicate})"
        )
        combined_sql = f"SELECT * FROM {dev_sql} UNION ALL {extension_select}"
    else:
        merge_operator = (
            "UNION ALL"
            if histories_are_date_disjoint or dev_rows == 0 or ext_rows == 0
            else "UNION"
        )
        combined_sql = (
            f"SELECT * FROM {dev_sql} {merge_operator} SELECT * FROM {ext_sql}"
        )
    temporary = destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex}")
    zero_row_materialization_provenance: dict[str, Any] | None = None
    try:
        connection.execute(
            f"""
            COPY (
                SELECT * FROM (
                    {combined_sql}
                ) combined
                ORDER BY {quoted_order}
            ) TO '{_sql_path(temporary)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)
            """
        )
        if dev_rows == 0 and ext_rows == 0:
            zero_row_materialization_provenance = (
                _provenance_zero_row_auxiliary_merge(
                    temporary,
                    development=development,
                    extension=extension,
                    destination_name=destination.name,
                    merge_operator=merge_operator,
                )
            )
        merged_sql = f"read_parquet('{_sql_path(temporary)}')"
        conflicts = int(
            connection.execute(
                f"SELECT count(*) FROM (SELECT {quoted_primary} FROM {merged_sql} GROUP BY {quoted_primary} HAVING count(*) > 1)"
            ).fetchone()[0]
        )
        _check(conflicts == 0, f"Development/extension primary-key conflict: {destination.name}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    merged = pq.ParquetFile(destination)
    _check(
        zero_row_materialization_provenance is not None
        or _ZERO_ROW_AUXILIARY_MERGE_METADATA_KEY
        not in (merged.schema_arrow.metadata or {}),
        f"Zero-row auxiliary merge provenance escaped its exact scope: {destination.name}",
    )
    deduplicated = dev_rows + ext_rows - merged.metadata.num_rows
    _check(deduplicated >= 0, f"Auxiliary merge cardinality differs: {destination.name}")
    if development_precedence_rows or interval_continuation_rows:
        _check(
            deduplicated
            == development_precedence_rows + interval_continuation_rows,
            f"Precedence/continuation merge cardinality differs: {destination.name}",
        )
    receipt = {
        "development_rows": dev_rows,
        "extension_rows": ext_rows,
        "primary_key_overlap_rows": overlap_rows,
        "development_precedence_rows": development_precedence_rows,
        "interval_continuation_rows": interval_continuation_rows,
        "retiming_metadata_difference_rows": retiming_difference_rows,
        "identical_overlap_rows_deduplicated": (
            deduplicated
            - development_precedence_rows
            - interval_continuation_rows
        ),
        "merge_operator": merge_operator,
        "conflicting_primary_keys": 0,
        "rows": merged.metadata.num_rows,
        "row_groups": merged.metadata.num_row_groups,
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
        "development_physical_nullability_widened_columns": list(
            development_physical_nullability_widened_columns
        ),
        "development_required_column_null_rows": (
            development_required_column_null_rows
        ),
        "development_required_column_null_rows_by_column": (
            development_required_column_null_rows_by_column
        ),
        "primary_key": list(primary_key),
        "sort_key": list(sort_key),
    }
    if zero_row_materialization_provenance is not None:
        receipt["zero_row_materialization_provenance"] = (
            zero_row_materialization_provenance
        )
    return receipt


def _write_filtered_table(
    source: Path,
    destination: Path,
    *,
    filter_expression: pa.compute.Expression,
) -> dict[str, Any]:
    table = pq.read_table(source, filters=filter_expression, use_threads=False)
    temporary = destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex}")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            row_group_size=131_072,
            use_dictionary=True,
            write_statistics=True,
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    parquet = pq.ParquetFile(destination)
    return {
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
    }


def _table_receipt(path: Path, *, gate: str, name: str) -> InputTable:
    parquet = pq.ParquetFile(path)
    return InputTable(
        gate=gate,
        name=name,
        path=path.resolve(),
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        rows=parquet.metadata.num_rows,
        row_groups=parquet.metadata.num_row_groups,
    )


def _common_real_root(paths: Sequence[Path]) -> Path:
    common = Path(os.path.commonpath([str(path.resolve()) for path in paths]))
    _check(common.is_dir() and not common.is_symlink(), "Feature input paths lack one real common root")
    return common


@dataclass(frozen=True, slots=True)
class ValidationFeatureInputBundle:
    scored_inputs: FeatureStoreInputs
    optionmetrics_working_inputs: FeatureStoreInputs
    receipt_path: Path
    logical_history_root: Path
    input_bundle_sha256: str
    source_receipts: Mapping[str, str]


def compose_verified_validation_feature_inputs(
    *,
    predictor_authority: ValidationDataContract,
    validation_universe_root: Path,
    validation_auxiliary_root: Path,
    full_predictor_context_root: Path,
    output_root: Path,
    development_inputs: FeatureStoreInputs | None = None,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> ValidationFeatureInputBundle:
    """Create and receipt the one logical history used by empirical formulas."""

    _require_predictor_authority(predictor_authority)
    development = development_inputs or resolve_input_catalog(authenticate_hashes=True)
    _check(isinstance(development, FeatureStoreInputs), "Development feature catalog differs")
    auxiliary_parents = {
        table.path.parent.resolve()
        for table in development.tables.values()
        if table.gate == "auxiliary"
    }
    _check(len(auxiliary_parents) == 1, "Verified development auxiliary root is ambiguous")
    development_auxiliary_root = next(iter(auxiliary_parents))

    universe_root = validation_universe_root.resolve()
    auxiliary_root = validation_auxiliary_root.resolve()
    context_root = full_predictor_context_root.resolve()
    output = output_root.resolve()
    for path, label in (
        (universe_root, "validation universe root"),
        (auxiliary_root, "validation auxiliary root"),
        (context_root, "full predictor context root"),
    ):
        _check(path.is_dir() and not path.is_symlink(), f"Missing or unsafe {label}")
    if output.exists():
        _check(output.is_dir() and not output.is_symlink() and not any(output.iterdir()), "Feature-input output is not empty/safe")
    else:
        output.mkdir(parents=True, exist_ok=False)
    tables = output / "tables"
    tables.mkdir()
    try:
        universe_receipt = _authenticate_universe_root(universe_root)
        auxiliary_receipt = _authenticate_auxiliary_extension(
            auxiliary_root,
            development_auxiliary_root=development_auxiliary_root,
        )
        context_receipt = _authenticate_full_context(context_root)
        specifications = _auxiliary_specs()
        connection = duckdb.connect(":memory:")
        merge_receipts: dict[str, Any] = {}
        try:
            connection.execute("SET preserve_insertion_order=false")
            connection.execute(f"SET threads={FEATURE_MERGE_DUCKDB_THREADS}")
            connection.execute(
                f"SET memory_limit='{FEATURE_MERGE_DUCKDB_MEMORY_LIMIT}'"
            )
            spill = output / ".duckdb_spill"
            spill.mkdir()
            connection.execute("SET temp_directory=?", [str(spill)])
            for ordinal, (name, (primary, order, schema)) in enumerate(
                specifications.items(), start=1
            ):
                _require_predictor_authority(predictor_authority)
                if progress is not None:
                    progress("stage11_auxiliary_history_merge_started", {"table": name, "ordinal": ordinal, "total": len(specifications)})
                merge_receipts[name] = _merge_auxiliary_table(
                    connection,
                    development=development_auxiliary_root / name,
                    extension=auxiliary_root / "tables" / name,
                    destination=tables / name,
                    expected_schema=schema,
                    primary_key=primary,
                    sort_key=order,
                )
                if progress is not None:
                    progress("stage11_auxiliary_history_merge_complete", {"table": name, "rows": merge_receipts[name]["rows"]})
        finally:
            connection.close()
            spill = output / ".duckdb_spill"
            if spill.exists():
                try:
                    spill.rmdir()
                except OSError:
                    pass

        calendar_receipts = {
            _CALENDAR_TABLE: _write_filtered_table(
                development.path(_CALENDAR_TABLE),
                tables / _CALENDAR_TABLE,
                filter_expression=pc.field("calendar_date") <= pa.scalar(LAST_PERMITTED_EXIT),
            ),
            _REBALANCE_TABLE: _write_filtered_table(
                development.path(_REBALANCE_TABLE),
                tables / _REBALANCE_TABLE,
                filter_expression=(pc.field("signal_date") <= pa.scalar(LAST_SIGNAL))
                & (pc.field("exit_date") <= pa.scalar(LAST_PERMITTED_EXIT)),
            ),
        }

        scored_linkage = auxiliary_root / "tables" / SCORED_LINKAGE_FILE
        context_linkage = auxiliary_root / "tables" / CONTEXT_LINKAGE_FILE
        scored_universe = universe_root / SCORED_UNIVERSE_FILE
        context_universe = universe_root / CONTEXT_UNIVERSE_FILE
        # These were already authenticated above through checkpoint/boundary and
        # universe receipts.  Keep the contexts physically separate.
        replacements: dict[str, Path] = {
            **{name: context_root / "tables" / name for name in _FULL_CONTEXT_TABLES},
            **{name: tables / name for name in specifications},
            _SPINE_TABLE: scored_universe,
            _LINKAGE_TABLE: scored_linkage,
            _CALENDAR_TABLE: tables / _CALENDAR_TABLE,
            _REBALANCE_TABLE: tables / _REBALANCE_TABLE,
        }
        scored_tables: dict[str, InputTable] = {}
        for name, original in development.tables.items():
            path = replacements.get(name, original.path)
            scored_tables[name] = _table_receipt(path, gate=original.gate, name=name)
        _check(set(scored_tables) == set(development.tables) and len(scored_tables) == 26, "Validation feature catalog is not the frozen 26-table set")
        all_paths = [table.path for table in scored_tables.values()]
        database_root = _common_real_root(all_paths)
        scored_inputs = FeatureStoreInputs(database_root, MappingProxyType(scored_tables))

        # OptionMetrics alone receives a 60-signal internal working spine so
        # the scored 2014-12-31 one-month change can reference 2014-11-28.
        # The context row is removed by the empirical row adapter before any
        # candidate payload is written.
        working_spine = tables / "optionmetrics_working_signal_date_universe.parquet"
        working_linkage = tables / "optionmetrics_working_auxiliary_signal_linkage.parquet"
        combine = duckdb.connect(":memory:")
        try:
            for first, second, destination in (
                (context_universe, scored_universe, working_spine),
                (context_linkage, scored_linkage, working_linkage),
            ):
                combine.execute(
                    f"""
                    COPY (
                        SELECT * FROM (
                            SELECT * FROM read_parquet('{_sql_path(first)}')
                            UNION ALL
                            SELECT * FROM read_parquet('{_sql_path(second)}')
                        ) source
                        ORDER BY signal_date, permno
                    ) TO '{_sql_path(destination)}'
                    (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)
                    """
                )
                duplicates = int(
                    combine.execute(
                        f"""
                        SELECT count(*) FROM (
                            SELECT signal_date, permno
                            FROM read_parquet('{_sql_path(destination)}')
                            GROUP BY signal_date, permno HAVING count(*) > 1
                        )
                        """
                    ).fetchone()[0]
                )
                _check(duplicates == 0, "OptionMetrics working context keys duplicate")
        finally:
            combine.close()

        working_tables = dict(scored_tables)
        working_tables[_SPINE_TABLE] = _table_receipt(
            working_spine,
            gate=development.tables[_SPINE_TABLE].gate,
            name=_SPINE_TABLE,
        )
        working_tables[_LINKAGE_TABLE] = _table_receipt(
            working_linkage,
            gate=development.tables[_LINKAGE_TABLE].gate,
            name=_LINKAGE_TABLE,
        )
        optionmetrics_working_inputs = FeatureStoreInputs(
            _common_real_root([table.path for table in working_tables.values()]),
            MappingProxyType(working_tables),
        )

        source_receipts = {
            "validation_universe_receipt_sha256": _sha256(universe_root / RECEIPT_FILE),
            "validation_auxiliary_receipt_sha256": _sha256(auxiliary_root / RECEIPT_FILE),
            "full_predictor_context_receipt_sha256": _sha256(context_root / "full_predictor_context_receipt.json"),
        }
        content = {
            "schema_version": 1,
            "status": "stage11_verified_logical_feature_history_complete",
            "adapter_id": INPUT_ADAPTER_ID,
            "scope": {
                "internal_context_signal": INTERNAL_CONTEXT_SIGNAL.isoformat(),
                "first_scored_signal": "2014-12-31",
                "last_scored_signal": LAST_SIGNAL.isoformat(),
                "scored_signal_count": 59,
                "final_period_rows_read": 0,
                "labels_forward_returns_weights_trades_pnl_rows_read": 0,
            },
            "source_receipts": source_receipts,
            "development_auxiliary_root": development_auxiliary_root.as_posix(),
            "auxiliary_history_merges": merge_receipts,
            "execution_resources": {
                "duckdb_threads": FEATURE_MERGE_DUCKDB_THREADS,
                "duckdb_memory_limit": FEATURE_MERGE_DUCKDB_MEMORY_LIMIT,
                "scientific_scope_or_formula_change": False,
            },
            "calendar_tables": calendar_receipts,
            "scored_catalog": {
                name: {
                    "path": table.path.as_posix(),
                    "sha256": table.sha256,
                    "size_bytes": table.size_bytes,
                    "rows": table.rows,
                    "row_groups": table.row_groups,
                    "gate": table.gate,
                }
                for name, table in sorted(scored_tables.items())
            },
            "internal_optionmetrics_context": {
                "signal_date": INTERNAL_CONTEXT_SIGNAL.isoformat(),
                "published_rows": 0,
                "working_signal_count": 60,
                "working_spine_sha256": working_tables[_SPINE_TABLE].sha256,
                "working_linkage_sha256": working_tables[_LINKAGE_TABLE].sha256,
                "source_context_spine_sha256": _sha256(context_universe),
                "source_context_linkage_sha256": _sha256(context_linkage),
            },
            "source_authorizations_remain_false": {
                "model_fitting": False,
                "validation_outcomes": False,
                "final_2020_2025_access": False,
            },
        }
        fingerprint = hashlib.sha256(_canonical_json(content)).hexdigest()
        receipt = {
            **content,
            "input_bundle_sha256": fingerprint,
            "input_bundle_id": f"stage11-feature-inputs-{fingerprint[:24]}",
        }
        receipt_path = output / "validation_feature_input_bundle_receipt.json"
        _atomic_json(receipt_path, receipt)
        return ValidationFeatureInputBundle(
            scored_inputs=scored_inputs,
            optionmetrics_working_inputs=optionmetrics_working_inputs,
            receipt_path=receipt_path,
            logical_history_root=output,
            input_bundle_sha256=fingerprint,
            source_receipts=MappingProxyType(source_receipts),
        )
    except Exception:
        # An incomplete merge has no authority and is always rebuilt cleanly.
        if output.exists() and not (output / "validation_feature_input_bundle_receipt.json").is_file():
            shutil.rmtree(output)
        raise


def load_validation_feature_input_bundle(
    root: Path,
    *,
    predictor_authority: ValidationDataContract,
) -> ValidationFeatureInputBundle:
    """Reauthenticate a completed logical-history bundle for a later CLI step."""

    _require_predictor_authority(predictor_authority)
    output = root.resolve()
    _check(output.is_dir() and not output.is_symlink(), "Feature-input bundle root is unsafe")
    receipt_path = output / "validation_feature_input_bundle_receipt.json"
    receipt = _json(receipt_path, "feature-input bundle receipt")
    _check(
        receipt.get("status") == "stage11_verified_logical_feature_history_complete"
        and receipt.get("adapter_id") == INPUT_ADAPTER_ID,
        "Feature-input bundle status/adapter differs",
    )
    fingerprint = receipt.get("input_bundle_sha256")
    identity = receipt.get("input_bundle_id")
    _check(
        isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and identity == f"stage11-feature-inputs-{fingerprint[:24]}",
        "Feature-input bundle identity differs",
    )
    content = {
        key: value
        for key, value in receipt.items()
        if key not in {"input_bundle_sha256", "input_bundle_id"}
    }
    _check(hashlib.sha256(_canonical_json(content)).hexdigest() == fingerprint, "Feature-input bundle fingerprint differs")
    catalog = receipt.get("scored_catalog")
    _check(isinstance(catalog, dict) and len(catalog) == 26, "Feature-input catalog differs")
    scored: dict[str, InputTable] = {}
    for name, value in catalog.items():
        _check(isinstance(value, dict), f"Feature-input catalog receipt differs: {name}")
        path = Path(str(value.get("path"))).resolve()
        table = InputTable(
            gate=str(value.get("gate")),
            name=name,
            path=path,
            sha256=str(value.get("sha256")),
            size_bytes=int(value.get("size_bytes", -1)),
            rows=int(value.get("rows", -1)),
            row_groups=int(value.get("row_groups", -1)),
        )
        _verify_parquet_receipt(
            path,
            {
                "sha256": table.sha256,
                "size_bytes": table.size_bytes,
                "rows": table.rows,
                "row_groups": table.row_groups,
            },
            label=f"feature-input catalog {name}",
        )
        scored[name] = table
    scored_inputs = FeatureStoreInputs(
        _common_real_root([table.path for table in scored.values()]),
        MappingProxyType(scored),
    )
    context = receipt.get("internal_optionmetrics_context")
    _check(isinstance(context, dict) and context.get("working_signal_count") == 60, "OptionMetrics working context receipt differs")
    working = dict(scored)
    for name, filename, hash_key in (
        (_SPINE_TABLE, "optionmetrics_working_signal_date_universe.parquet", "working_spine_sha256"),
        (_LINKAGE_TABLE, "optionmetrics_working_auxiliary_signal_linkage.parquet", "working_linkage_sha256"),
    ):
        path = output / "tables" / filename
        table = _table_receipt(path, gate=scored[name].gate, name=name)
        _check(table.sha256 == context.get(hash_key), f"OptionMetrics working table hash differs: {name}")
        working[name] = table
    working_inputs = FeatureStoreInputs(
        _common_real_root([table.path for table in working.values()]),
        MappingProxyType(working),
    )
    source_receipts = receipt.get("source_receipts")
    _check(isinstance(source_receipts, dict), "Feature-input source receipts differ")
    return ValidationFeatureInputBundle(
        scored_inputs=scored_inputs,
        optionmetrics_working_inputs=working_inputs,
        receipt_path=receipt_path,
        logical_history_root=output,
        input_bundle_sha256=fingerprint,
        source_receipts=MappingProxyType({str(key): str(value) for key, value in source_receipts.items()}),
    )


__all__ = [
    "INPUT_ADAPTER_ID",
    "ValidationFeatureInputAdapterError",
    "ValidationFeatureInputBundle",
    "compose_verified_validation_feature_inputs",
    "load_validation_feature_input_bundle",
]
