"""Bounded-memory quality artifacts for the frozen Stage 6 feature store.

This module audits an already materialized raw feature Parquet and the seven
lineage shards.  It returns deterministic Arrow tables and canonical summary
JSON bytes; publication and external filesystem writes remain the caller's
responsibility.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import tomllib
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.features.feature_lineage import VALUE_STATUS_SET
from portfolio_bot.features.feature_store_contract import (
    AUDIT_FIELDS,
    DEFAULT_POLICY,
    LINEAGE_SCHEMA,
    PROJECT_ROOT,
    expanded_features,
    validate_feature_store_contract,
)
from portfolio_bot.features.phase_03_contract import canonical_json_bytes


FEATURE_MISSINGNESS_SCHEMA = pa.schema(
    [
        pa.field("feature_name", pa.string(), nullable=False),
        pa.field("feature_family", pa.string(), nullable=False),
        pa.field("signal_year", pa.int16(), nullable=False),
        pa.field("exchange", pa.string(), nullable=False),
        pa.field("domicile", pa.string(), nullable=False),
        pa.field("adr_status", pa.string(), nullable=False),
        pa.field("long_eligible", pa.bool_(), nullable=False),
        pa.field("short_eligible", pa.bool_(), nullable=False),
        pa.field("value_status", pa.string(), nullable=False),
        pa.field("row_count", pa.int64(), nullable=False),
        pa.field("group_total", pa.int64(), nullable=False),
        pa.field("row_fraction", pa.float64(), nullable=False),
    ]
)
FEATURE_COVERAGE_SCHEMA = pa.schema(
    [
        pa.field("feature_name", pa.string(), nullable=False),
        pa.field("feature_family", pa.string(), nullable=False),
        pa.field("signal_year", pa.int16(), nullable=False),
        pa.field("total_rows", pa.int64(), nullable=False),
        pa.field("present_rows", pa.int64(), nullable=False),
        pa.field("missing_rows", pa.int64(), nullable=False),
        pa.field("coverage_rate", pa.float64(), nullable=False),
    ]
)
FEATURE_OUTLIERS_SCHEMA = pa.schema(
    [
        pa.field("feature_name", pa.string(), nullable=False),
        pa.field("signal_year", pa.int16(), nullable=False),
        pa.field("finite_count", pa.int64(), nullable=False),
        pa.field("missing_count", pa.int64(), nullable=False),
        pa.field("nonfinite_count", pa.int64(), nullable=False),
        pa.field("minimum", pa.float64(), nullable=True),
        pa.field("p01", pa.float64(), nullable=True),
        pa.field("median", pa.float64(), nullable=True),
        pa.field("p99", pa.float64(), nullable=True),
        pa.field("maximum", pa.float64(), nullable=True),
    ]
)
FEATURE_TIMING_CHECKS_SCHEMA = pa.schema(
    [
        pa.field("check_name", pa.string(), nullable=False),
        pa.field("observed_count", pa.int64(), nullable=False),
        pa.field("expected_count", pa.int64(), nullable=False),
        pa.field("passed", pa.bool_(), nullable=False),
        pa.field("severity", pa.string(), nullable=False),
        pa.field("details", pa.string(), nullable=False),
    ]
)
FEATURE_SOURCE_COVERAGE_SCHEMA = pa.schema(
    [
        pa.field("source_family", pa.string(), nullable=False),
        pa.field("signal_year", pa.int16(), nullable=False),
        pa.field("exchange", pa.string(), nullable=False),
        pa.field("domicile", pa.string(), nullable=False),
        pa.field("adr_status", pa.string(), nullable=False),
        pa.field("coverage_status", pa.string(), nullable=False),
        pa.field("row_count", pa.int64(), nullable=False),
        pa.field("group_total", pa.int64(), nullable=False),
        pa.field("coverage_fraction", pa.float64(), nullable=False),
    ]
)

REQUIRED_PROVENANCE_COUNTERS = frozenset(
    {
        "optionmetrics_total_rows_used",
        "raw_pastor_stambaugh_innovation_used",
    }
)
_BUILD_ID = re.compile(r"phase03-features-[0-9a-f]{24}\Z")
_NUMERIC_KINDS = {"continuous_firm", "binary_firm", "aggregate_state"}
_SEGMENT_FEATURES = (
    "business_segment_count",
    "geographic_segment_count",
    "business_segment_revenue_hhi",
    "business_segment_revenue_hhi_change_1y",
)


class FeatureStoreQualityError(RuntimeError):
    """Raised when quality inputs are unsafe or a required audit fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeatureStoreQualityError(message)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _regular_parquet(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    candidate = (
        candidate
        if candidate.is_absolute()
        else (Path.cwd() / candidate)
    )
    _require(not candidate.is_symlink(), f"Symlink prohibited for {label}")
    result = candidate.resolve()
    _require(result.is_file(), f"Missing {label}: {result}")
    try:
        pq.ParquetFile(result)
    except Exception as exc:
        raise FeatureStoreQualityError(
            f"{label} is not readable Parquet: {result}"
        ) from exc
    return result


def _arrow_table(rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> pa.Table:
    try:
        return pa.Table.from_pylist(list(rows), schema=schema)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise FeatureStoreQualityError(
            f"Quality output violates schema: {schema.names}"
        ) from exc


@dataclass(frozen=True)
class _QualityContext:
    policy: Mapping[str, Any]
    rules: Mapping[str, Any]
    features: tuple[Mapping[str, Any], ...]
    family_order: tuple[str, ...]
    expected_raw_schema: pa.Schema
    hard_gate_expectations: Mapping[str, int]
    quality_flags: Mapping[str, bool]
    summary_contract: Mapping[str, Any]


def _load_context(policy_path: Path) -> _QualityContext:
    policy_path = policy_path.resolve()
    validated = validate_feature_store_contract(policy_path)
    with policy_path.open("rb") as handle:
        policy = tomllib.load(handle)
    registry_path = _project_path(policy["semantic_contract"]["registry_path"])
    rules_path = _project_path(policy["physical_rules"]["path"])
    registry = json.loads(registry_path.read_text("utf-8"))
    rules = json.loads(rules_path.read_text("utf-8"))
    _require(
        isinstance(registry, dict) and isinstance(rules, dict),
        "Frozen registry or physical rules are not objects",
    )
    features = tuple(expanded_features(registry))
    numeric_gates = {
        key: int(value)
        for key, value in policy["quality_gates"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    flags = {
        key: bool(value)
        for key, value in policy["quality_gates"].items()
        if isinstance(value, bool)
    }
    summary_contract = rules["publication_payloads"][
        "quality_summary_value_contract"
    ]
    _require(
        len(features) == 163
        and tuple(validated["family_order"])
        == tuple(policy["build"]["family_order"]),
        "Frozen quality feature contract changed",
    )
    return _QualityContext(
        policy=policy,
        rules=rules,
        features=features,
        family_order=tuple(validated["family_order"]),
        expected_raw_schema=validated["raw_schema"],
        hard_gate_expectations=numeric_gates,
        quality_flags=flags,
        summary_contract=summary_contract,
    )


@dataclass(frozen=True)
class FeatureStoreQualityArtifacts:
    feature_missingness: pa.Table
    feature_coverage: pa.Table
    feature_outliers: pa.Table
    feature_timing_checks: pa.Table
    feature_source_coverage: pa.Table
    quality_summary: Mapping[str, Any]

    @property
    def quality_summary_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.quality_summary)

    @property
    def passed(self) -> bool:
        return (
            self.quality_summary["status"]
            == "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
        )


def _connect(
    *,
    memory_limit_mb: int,
    temporary_directory: Path | None,
) -> duckdb.DuckDBPyConnection:
    _require(
        isinstance(memory_limit_mb, int)
        and not isinstance(memory_limit_mb, bool)
        and memory_limit_mb >= 64,
        "DuckDB memory_limit_mb must be an integer of at least 64",
    )
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET threads = 1")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET memory_limit = '{memory_limit_mb}MB'")
    if temporary_directory is not None:
        candidate = Path(temporary_directory).expanduser()
        candidate = (
            candidate
            if candidate.is_absolute()
            else (Path.cwd() / candidate)
        )
        _require(
            not candidate.is_symlink(),
            "DuckDB temporary directory must not be a symlink",
        )
        temp = candidate.resolve()
        _require(
            temp.is_dir(),
            "DuckDB temporary directory must be a real directory",
        )
        connection.execute(
            f"SET temp_directory = {_sql_literal(temp.as_posix())}"
        )
    return connection


def _schema_mismatch_count(
    observed: pa.Schema,
    expected: pa.Schema,
) -> tuple[int, int]:
    name_mismatches = sum(
        left != right
        for left, right in zip_longest(
            observed.names,
            expected.names,
            fillvalue=None,
        )
    )
    expected_by_name = {field.name: field for field in expected}
    type_mismatches = 0
    for field in observed:
        expected_field = expected_by_name.get(field.name)
        if expected_field is not None and (
            field.type != expected_field.type
            or field.nullable != expected_field.nullable
        ):
            type_mismatches += 1
    type_mismatches += len(set(expected.names) - set(observed.names))
    return name_mismatches, type_mismatches


def _register_views(
    connection: duckdb.DuckDBPyConnection,
    *,
    raw_path: Path,
    lineage_paths: Mapping[str, Path],
    features: Sequence[Mapping[str, Any]],
    raw_columns: frozenset[str],
) -> None:
    connection.execute(
        "CREATE VIEW raw_store AS SELECT * FROM read_parquet("
        f"{_sql_literal(raw_path.as_posix())})"
    )
    lineage_parts = [
        "SELECT "
        f"{_sql_literal(family)}::VARCHAR AS feature_family, * "
        f"FROM read_parquet({_sql_literal(lineage_paths[family].as_posix())})"
        for family in lineage_paths
    ]
    connection.execute(
        "CREATE VIEW all_lineage AS " + " UNION ALL ".join(lineage_parts)
    )
    feature_map = pa.table(
        {
            "feature_name": [str(feature["name"]) for feature in features],
            "feature_family": [str(feature["family"]) for feature in features],
            "feature_kind": [str(feature["kind"]) for feature in features],
            "feature_ordinal": list(range(1, len(features) + 1)),
        }
    )
    connection.register("feature_map_arrow", feature_map)
    connection.execute(
        "CREATE VIEW feature_map AS SELECT * FROM feature_map_arrow"
    )

    domicile = (
        "COALESCE(NULLIF(TRIM(domicile_country_category), ''), "
        "'UNKNOWN_COUNTRY')"
        if "domicile_country_category" in raw_columns
        else "'UNKNOWN_COUNTRY'"
    )
    adr = (
        "CASE WHEN adr_indicator = 1 THEN 'ADR' "
        "WHEN adr_indicator = 0 THEN 'NON_ADR' ELSE 'UNKNOWN' END"
        if "adr_indicator" in raw_columns
        else "'UNKNOWN'"
    )
    connection.execute(
        """
        CREATE VIEW raw_dimensions AS
        WITH normalized AS (
            SELECT
                signal_date,
                permno,
                COALESCE(NULLIF(TRIM(primary_exchange), ''), 'UNKNOWN')
                    AS exchange,
                """
        + domicile
        + """ AS domicile,
                """
        + adr
        + """ AS adr_status,
                long_eligible,
                short_eligible
            FROM raw_store
        )
        SELECT
            signal_date,
            permno,
            MIN(exchange) AS exchange,
            MIN(domicile) AS domicile,
            MIN(adr_status) AS adr_status,
            BOOL_OR(long_eligible) AS long_eligible,
            BOOL_OR(short_eligible) AS short_eligible
        FROM normalized
        GROUP BY signal_date, permno
        """
    )
    connection.execute(
        """
        CREATE VIEW raw_years AS
        SELECT
            CAST(YEAR(signal_date) AS SMALLINT) AS signal_year,
            COUNT(*)::BIGINT AS total_rows
        FROM raw_store
        GROUP BY YEAR(signal_date)
        """
    )


def _feature_missingness(
    connection: duckdb.DuckDBPyConnection,
) -> pa.Table:
    result = connection.execute(
        """
        WITH grouped AS (
            SELECT
                l.feature_name,
                f.feature_family,
                CAST(YEAR(l.signal_date) AS SMALLINT) AS signal_year,
                r.exchange,
                r.domicile,
                r.adr_status,
                r.long_eligible,
                r.short_eligible,
                l.value_status,
                COUNT(*)::BIGINT AS row_count
            FROM all_lineage AS l
            JOIN feature_map AS f USING (feature_name)
            JOIN raw_dimensions AS r USING (signal_date, permno)
            GROUP BY ALL
        ),
        totaled AS (
            SELECT
                *,
                SUM(row_count) OVER (
                    PARTITION BY feature_name, signal_year, exchange,
                        domicile, adr_status, long_eligible, short_eligible
                )::BIGINT AS group_total
            FROM grouped
        )
        SELECT
            feature_name,
            feature_family,
            signal_year,
            exchange,
            domicile,
            adr_status,
            long_eligible,
            short_eligible,
            value_status,
            row_count,
            group_total,
            row_count::DOUBLE / group_total::DOUBLE AS row_fraction
        FROM totaled
        ORDER BY feature_name, signal_year, exchange, domicile, adr_status,
            long_eligible, short_eligible, value_status
        """
    ).fetch_arrow_table()
    return _arrow_table(result.to_pylist(), FEATURE_MISSINGNESS_SCHEMA)


def _feature_coverage(
    connection: duckdb.DuckDBPyConnection,
) -> pa.Table:
    result = connection.execute(
        """
        WITH present AS (
            SELECT
                l.feature_name,
                CAST(YEAR(l.signal_date) AS SMALLINT) AS signal_year,
                COUNT(DISTINCT (l.signal_date, l.permno))::BIGINT
                    AS present_rows
            FROM all_lineage AS l
            JOIN feature_map AS f USING (feature_name)
            JOIN raw_dimensions AS r USING (signal_date, permno)
            WHERE l.value_status = 'PRESENT'
            GROUP BY l.feature_name, YEAR(l.signal_date)
        )
        SELECT
            f.feature_name,
            f.feature_family,
            y.signal_year,
            y.total_rows,
            COALESCE(p.present_rows, 0)::BIGINT AS present_rows,
            (y.total_rows - COALESCE(p.present_rows, 0))::BIGINT
                AS missing_rows,
            COALESCE(p.present_rows, 0)::DOUBLE
                / y.total_rows::DOUBLE AS coverage_rate
        FROM feature_map AS f
        CROSS JOIN raw_years AS y
        LEFT JOIN present AS p USING (feature_name, signal_year)
        ORDER BY f.feature_name, y.signal_year
        """
    ).fetch_arrow_table()
    return _arrow_table(result.to_pylist(), FEATURE_COVERAGE_SCHEMA)


def _feature_source_coverage(
    connection: duckdb.DuckDBPyConnection,
    missingness: pa.Table,
) -> pa.Table:
    connection.register("feature_missingness_arrow", missingness)
    result = connection.execute(
        """
        WITH grouped AS (
            SELECT
                feature_family AS source_family,
                signal_year,
                exchange,
                domicile,
                adr_status,
                value_status AS coverage_status,
                SUM(row_count)::BIGINT AS row_count
            FROM feature_missingness_arrow
            GROUP BY ALL
        ),
        totaled AS (
            SELECT
                *,
                SUM(row_count) OVER (
                    PARTITION BY source_family, signal_year, exchange,
                        domicile, adr_status
                )::BIGINT AS group_total
            FROM grouped
        )
        SELECT
            source_family,
            signal_year,
            exchange,
            domicile,
            adr_status,
            coverage_status,
            row_count,
            group_total,
            row_count::DOUBLE / group_total::DOUBLE AS coverage_fraction
        FROM totaled
        ORDER BY source_family, signal_year, exchange, domicile, adr_status,
            coverage_status
        """
    ).fetch_arrow_table()
    connection.unregister("feature_missingness_arrow")
    return _arrow_table(result.to_pylist(), FEATURE_SOURCE_COVERAGE_SCHEMA)


def _feature_outliers(
    connection: duckdb.DuckDBPyConnection,
    *,
    features: Sequence[Mapping[str, Any]],
    raw_columns: frozenset[str],
) -> pa.Table:
    rows: list[dict[str, Any]] = []
    numeric = sorted(
        (
            feature
            for feature in features
            if feature["kind"] in _NUMERIC_KINDS
        ),
        key=lambda item: str(item["name"]).encode("utf-8"),
    )
    for feature in numeric:
        name = str(feature["name"])
        if name not in raw_columns:
            missing = connection.execute(
                """
                SELECT signal_year, total_rows
                FROM raw_years
                ORDER BY signal_year
                """
            ).fetchall()
            rows.extend(
                {
                    "feature_name": name,
                    "signal_year": int(year),
                    "finite_count": 0,
                    "missing_count": int(total),
                    "nonfinite_count": 0,
                    "minimum": None,
                    "p01": None,
                    "median": None,
                    "p99": None,
                    "maximum": None,
                }
                for year, total in missing
            )
            continue
        column = _identifier(name)
        values = connection.execute(
            f"""
            SELECT
                CAST(YEAR(signal_date) AS SMALLINT) AS signal_year,
                COUNT(*) FILTER (
                    WHERE {column} IS NOT NULL
                    AND ISFINITE(CAST({column} AS DOUBLE))
                )::BIGINT AS finite_count,
                COUNT(*) FILTER (
                    WHERE {column} IS NULL
                )::BIGINT AS missing_count,
                COUNT(*) FILTER (
                    WHERE {column} IS NOT NULL
                    AND NOT ISFINITE(CAST({column} AS DOUBLE))
                )::BIGINT AS nonfinite_count,
                MIN(CAST({column} AS DOUBLE)) FILTER (
                    WHERE {column} IS NOT NULL
                    AND ISFINITE(CAST({column} AS DOUBLE))
                ) AS minimum,
                QUANTILE_CONT(CAST({column} AS DOUBLE), 0.01) FILTER (
                    WHERE {column} IS NOT NULL
                    AND ISFINITE(CAST({column} AS DOUBLE))
                ) AS p01,
                QUANTILE_CONT(CAST({column} AS DOUBLE), 0.5) FILTER (
                    WHERE {column} IS NOT NULL
                    AND ISFINITE(CAST({column} AS DOUBLE))
                ) AS median,
                QUANTILE_CONT(CAST({column} AS DOUBLE), 0.99) FILTER (
                    WHERE {column} IS NOT NULL
                    AND ISFINITE(CAST({column} AS DOUBLE))
                ) AS p99,
                MAX(CAST({column} AS DOUBLE)) FILTER (
                    WHERE {column} IS NOT NULL
                    AND ISFINITE(CAST({column} AS DOUBLE))
                ) AS maximum
            FROM raw_store
            GROUP BY YEAR(signal_date)
            ORDER BY signal_year
            """
        ).fetchall()
        rows.extend(
            {
                "feature_name": name,
                "signal_year": int(value[0]),
                "finite_count": int(value[1]),
                "missing_count": int(value[2]),
                "nonfinite_count": int(value[3]),
                "minimum": value[4],
                "p01": value[5],
                "median": value[6],
                "p99": value[7],
                "maximum": value[8],
            }
            for value in values
        )
    return _arrow_table(rows, FEATURE_OUTLIERS_SCHEMA)


def _raw_duplicate_count(
    connection: duckdb.DuckDBPyConnection,
) -> int:
    value = connection.execute(
        """
        SELECT COALESCE(SUM(row_count - 1), 0)::BIGINT
        FROM (
            SELECT COUNT(*)::BIGINT AS row_count
            FROM raw_store
            GROUP BY signal_date, permno
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    return int(value)


def _lineage_key_counts(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[int, int]:
    duplicate, unique_valid = connection.execute(
        """
        WITH grouped AS (
            SELECT
                signal_date,
                permno,
                feature_name,
                COUNT(*)::BIGINT AS row_count
            FROM all_lineage
            GROUP BY signal_date, permno, feature_name
        )
        SELECT
            COALESCE(SUM(row_count - 1), 0)::BIGINT
                AS duplicate_lineage_keys,
            COUNT(*) FILTER (
                WHERE f.feature_name IS NOT NULL
                AND r.permno IS NOT NULL
            )::BIGINT AS unique_valid_keys
        FROM grouped AS g
        LEFT JOIN feature_map AS f USING (feature_name)
        LEFT JOIN raw_dimensions AS r USING (signal_date, permno)
        """
    ).fetchone()
    return int(duplicate), int(unique_valid)


def _lineage_aggregate_counts(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, int]:
    valid_statuses = ", ".join(
        _sql_literal(value)
        for value in sorted(VALUE_STATUS_SET)
    )
    row = connection.execute(
        f"""
        SELECT
            COUNT(*)::BIGINT AS lineage_rows,
            COUNT(*) FILTER (
                WHERE l.value_status = 'PRESENT'
            )::BIGINT AS present_values,
            COUNT(*) FILTER (
                WHERE l.value_status = 'MISSING_SOURCE_UNAVAILABLE'
            )::BIGINT AS structural_missing_values,
            COUNT(*) FILTER (
                WHERE l.max_observation_date > l.signal_date
            )::BIGINT AS observation_date_after_signal,
            COUNT(*) FILTER (
                WHERE l.max_public_availability_date > l.signal_date
            )::BIGINT AS public_availability_date_after_signal,
            COUNT(*) FILTER (
                WHERE l.value_status = 'PRESENT'
                AND l.stale_after_date < l.signal_date
            )::BIGINT AS stale_feature_uses,
            COUNT(*) FILTER (
                WHERE f.feature_name IS NULL
            )::BIGINT AS unknown_feature_names,
            COUNT(*) FILTER (
                WHERE r.permno IS NULL
            )::BIGINT AS orphan_lineage_keys,
            COUNT(*) FILTER (
                WHERE f.feature_name IS NOT NULL
                AND f.feature_family <> l.feature_family
            )::BIGINT AS lineage_family_mismatch,
            COUNT(*) FILTER (
                WHERE l.value_status NOT IN ({valid_statuses})
            )::BIGINT AS unknown_value_statuses
        FROM all_lineage AS l
        LEFT JOIN feature_map AS f USING (feature_name)
        LEFT JOIN raw_dimensions AS r USING (signal_date, permno)
        """
    ).fetchone()
    names = (
        "lineage_rows",
        "present_values",
        "structural_missing_values",
        "observation_date_after_signal",
        "public_availability_date_after_signal",
        "stale_feature_uses",
        "unknown_feature_names",
        "orphan_lineage_keys",
        "lineage_family_mismatch",
        "unknown_value_statuses",
    )
    return {name: int(value) for name, value in zip(names, row, strict=True)}


def _option_noncoverage_zeros(
    connection: duckdb.DuckDBPyConnection,
    *,
    features: Sequence[Mapping[str, Any]],
    raw_columns: frozenset[str],
) -> int:
    option_names = [
        str(feature["name"])
        for feature in features
        if feature["family"] == "optionmetrics"
        and feature["name"] in raw_columns
    ]
    if not option_names:
        return 0
    zero_parts = [
        "SELECT signal_date, permno, "
        f"{_sql_literal(name)}::VARCHAR AS feature_name "
        f"FROM raw_store WHERE {_identifier(name)} = 0"
        for name in option_names
    ]
    value = connection.execute(
        """
        WITH zero_values AS (
        """
        + " UNION ALL ".join(zero_parts)
        + """
        )
        SELECT COUNT(*)::BIGINT
        FROM zero_values AS z
        JOIN all_lineage AS l USING (signal_date, permno, feature_name)
        WHERE l.feature_family = 'optionmetrics'
            AND l.value_status = 'MISSING_NOT_COVERED'
        """
    ).fetchone()[0]
    return int(value)


def _structural_missingness(
    connection: duckdb.DuckDBPyConnection,
    raw_columns: frozenset[str],
) -> dict[str, int]:
    segment_terms = [
        f"CASE WHEN {_identifier(name)} IS NOT NULL THEN 1 ELSE 0 END"
        for name in _SEGMENT_FEATURES
        if name in raw_columns
    ]
    segment_expression = " + ".join(segment_terms) if segment_terms else "0"
    quarterly_expression = (
        "COUNT(*) FILTER ("
        "WHERE fundamental_source_basis = 'TTM' "
        "AND share_growth_yoy IS NOT NULL)"
        if {
            "fundamental_source_basis",
            "share_growth_yoy",
        }.issubset(raw_columns)
        else "0"
    )
    qualifying_expression = (
        "COUNT(*) FILTER ("
        "WHERE latest_interim_financial_filing_form = "
        "'QUALIFYING_FINANCIAL_RESULTS_6-K')"
        if "latest_interim_financial_filing_form" in raw_columns
        else "0"
    )
    row = connection.execute(
        f"""
        SELECT
            COALESCE(SUM({segment_expression}), 0)::BIGINT,
            ({quarterly_expression})::BIGINT,
            ({qualifying_expression})::BIGINT
        FROM raw_store
        """
    ).fetchone()
    return {
        "segment_feature_values": int(row[0]),
        "quarterly_share_growth_values": int(row[1]),
        "qualifying_financial_results_6k_values": int(row[2]),
    }


def _timing_checks(
    *,
    hard_gates: Mapping[str, int],
    expected_hard_gates: Mapping[str, int],
    integrity_checks: Mapping[str, int],
) -> pa.Table:
    rows: list[dict[str, Any]] = []
    for name in sorted(expected_hard_gates, key=lambda value: value.encode("utf-8")):
        observed = int(hard_gates[name])
        expected = int(expected_hard_gates[name])
        rows.append(
            {
                "check_name": name,
                "observed_count": observed,
                "expected_count": expected,
                "passed": observed == expected,
                "severity": "CRITICAL",
                "details": (
                    "Frozen hard gate from feature_store_policy_v1.toml; "
                    f"expected={expected}; observed={observed}"
                ),
            }
        )
    for name in sorted(integrity_checks, key=lambda value: value.encode("utf-8")):
        observed = int(integrity_checks[name])
        rows.append(
            {
                "check_name": name,
                "observed_count": observed,
                "expected_count": 0,
                "passed": observed == 0,
                "severity": "CRITICAL",
                "details": (
                    "Stage 6 structural/integrity check; expected=0; "
                    f"observed={observed}"
                ),
            }
        )
    rows.sort(key=lambda row: row["check_name"].encode("utf-8"))
    return _arrow_table(rows, FEATURE_TIMING_CHECKS_SCHEMA)


def _summary(
    *,
    build_id: str,
    context: _QualityContext,
    counts: Mapping[str, int],
    hard_gates: Mapping[str, int],
    structural_missingness: Mapping[str, int],
    passed: bool,
) -> dict[str, Any]:
    scope = context.summary_contract["scope"]["exact_values"]
    input_builds = context.summary_contract["input_builds"]["exact_values"]
    result = {
        "schema_version": 1,
        "contract_id": str(context.policy["contract_id"]),
        "policy_id": str(context.policy["policy_id"]),
        "build_id": build_id,
        "status": (
            "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
            if passed
            else "PRIMARY_FAILED"
        ),
        "scope": dict(scope),
        "counts": {key: int(value) for key, value in counts.items()},
        "hard_gates": {
            key: int(hard_gates[key])
            for key in context.hard_gate_expectations
        },
        "quality_policy_flags": dict(context.quality_flags),
        "structural_missingness": {
            key: int(structural_missingness[key])
            for key in context.summary_contract[
                "structural_missingness"
            ]["exact_keys"]
        },
        "input_builds": dict(input_builds),
    }
    expected_keys = context.rules["publication_payloads"][
        "quality_summary_exact_keys"
    ]
    _require(
        set(result) == set(expected_keys),
        "Generated quality summary keys differ from frozen contract",
    )
    for key in (
        "raw_rows",
        "raw_feature_columns",
        "feature_families",
        "lineage_rows",
        "present_values",
        "missing_values",
        "structural_missing_values",
    ):
        _require(
            key in result["counts"]
            and isinstance(result["counts"][key], int)
            and result["counts"][key] >= 0,
            f"Invalid quality summary count: {key}",
        )
    return result


def build_feature_store_quality_artifacts(
    *,
    build_id: str,
    raw_store_path: str | Path,
    lineage_paths: Mapping[str, str | Path],
    provenance_counters: Mapping[str, int],
    policy_path: Path = DEFAULT_POLICY,
    memory_limit_mb: int = 512,
    temporary_directory: Path | None = None,
    require_pass: bool = True,
) -> FeatureStoreQualityArtifacts:
    """Build all six frozen Stage 6 quality payloads without publishing them."""

    _require(
        isinstance(build_id, str) and _BUILD_ID.fullmatch(build_id) is not None,
        "Invalid Stage 6 feature-store build_id",
    )
    _require(
        set(provenance_counters) == REQUIRED_PROVENANCE_COUNTERS,
        "Provenance counters must contain the exact non-reconstructable gates",
    )
    for name, value in provenance_counters.items():
        _require(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0,
            f"Invalid provenance counter: {name}",
        )
    context = _load_context(policy_path)
    _require(
        set(lineage_paths) == set(context.family_order),
        "Lineage shard families differ from the frozen seven-family contract",
    )
    raw_path = _regular_parquet(raw_store_path, "raw feature store")
    resolved_lineage = {
        family: _regular_parquet(
            lineage_paths[family],
            f"{family} lineage shard",
        )
        for family in context.family_order
    }
    all_paths = [raw_path, *resolved_lineage.values()]
    _require(
        len({path.as_posix() for path in all_paths}) == len(all_paths),
        "Quality inputs reuse one physical Parquet path",
    )
    raw_schema = pq.ParquetFile(raw_path).schema_arrow
    raw_columns = frozenset(raw_schema.names)
    required_audit = {field.name for field in AUDIT_FIELDS}
    _require(
        required_audit.issubset(raw_columns),
        "Raw feature store is missing an audit column required for QA",
    )
    for family, path in resolved_lineage.items():
        observed = pq.ParquetFile(path).schema_arrow
        _require(
            observed.equals(LINEAGE_SCHEMA, check_metadata=True),
            f"Lineage schema differs for family: {family}",
        )

    name_mismatches, type_mismatches = _schema_mismatch_count(
        raw_schema,
        context.expected_raw_schema,
    )
    connection = _connect(
        memory_limit_mb=memory_limit_mb,
        temporary_directory=temporary_directory,
    )
    try:
        _register_views(
            connection,
            raw_path=raw_path,
            lineage_paths=resolved_lineage,
            features=context.features,
            raw_columns=raw_columns,
        )
        missingness = _feature_missingness(connection)
        coverage = _feature_coverage(connection)
        outliers = _feature_outliers(
            connection,
            features=context.features,
            raw_columns=raw_columns,
        )
        source_coverage = _feature_source_coverage(
            connection,
            missingness,
        )

        raw_rows = int(
            connection.execute(
                "SELECT COUNT(*)::BIGINT FROM raw_store"
            ).fetchone()[0]
        )
        raw_duplicates = _raw_duplicate_count(connection)
        lineage_duplicates, unique_valid_lineage = _lineage_key_counts(
            connection
        )
        lineage = _lineage_aggregate_counts(connection)
        raw_unique_keys = int(
            connection.execute(
                "SELECT COUNT(*)::BIGINT FROM raw_dimensions"
            ).fetchone()[0]
        )
        expected_for_observed_raw = raw_unique_keys * len(context.features)
        missing_lineage = max(
            expected_for_observed_raw - unique_valid_lineage,
            0,
        )
        last_signal_date = dt.date.fromisoformat(
            str(context.policy["scope"]["last_signal_date"])
        )
        first_signal_date = dt.date.fromisoformat(
            str(context.policy["scope"]["first_signal_date"])
        )
        future_rows = int(
            connection.execute(
                "SELECT COUNT(*)::BIGINT FROM raw_store "
                "WHERE signal_date > ?",
                [last_signal_date],
            ).fetchone()[0]
        )
        pre_scope_rows = int(
            connection.execute(
                "SELECT COUNT(*)::BIGINT FROM raw_store "
                "WHERE signal_date < ?",
                [first_signal_date],
            ).fetchone()[0]
        )
        nonfinite = sum(
            int(value)
            for value in outliers["nonfinite_count"].to_pylist()
        )
        option_noncoverage_zeros = _option_noncoverage_zeros(
            connection,
            features=context.features,
            raw_columns=raw_columns,
        )
        structural = _structural_missingness(
            connection,
            raw_columns,
        )

        raw_feature_columns = max(
            0,
            len(raw_schema.names) - len(AUDIT_FIELDS),
        )
        hard_gates = {
            "duplicate_raw_keys": raw_duplicates,
            "raw_rows": raw_rows,
            "raw_feature_columns": raw_feature_columns,
            "duplicate_lineage_keys": lineage_duplicates,
            "missing_lineage_keys": missing_lineage,
            "lineage_rows": lineage["lineage_rows"],
            "observation_date_after_signal": lineage[
                "observation_date_after_signal"
            ],
            "public_availability_date_after_signal": lineage[
                "public_availability_date_after_signal"
            ],
            "stale_feature_uses": lineage["stale_feature_uses"],
            "future_predictor_rows": future_rows,
            "nonfinite_published_numeric_values": nonfinite,
            "unknown_feature_names": lineage["unknown_feature_names"],
            "feature_order_mismatch": name_mismatches,
            "optionmetrics_total_rows_used": int(
                provenance_counters["optionmetrics_total_rows_used"]
            ),
            "optionmetrics_noncoverage_encoded_as_zero":
                option_noncoverage_zeros,
            "raw_pastor_stambaugh_innovation_used": int(
                provenance_counters[
                    "raw_pastor_stambaugh_innovation_used"
                ]
            ),
        }
        _require(
            set(hard_gates) == set(context.hard_gate_expectations),
            "Computed hard-gate keys differ from the frozen policy",
        )
        integrity = {
            "lineage_family_mismatch": lineage[
                "lineage_family_mismatch"
            ],
            "orphan_lineage_keys": lineage["orphan_lineage_keys"],
            "pre_scope_predictor_rows": pre_scope_rows,
            "raw_schema_type_mismatch": type_mismatches,
            "segment_feature_values": structural[
                "segment_feature_values"
            ],
            "quarterly_share_growth_values": structural[
                "quarterly_share_growth_values"
            ],
            "qualifying_financial_results_6k_values": structural[
                "qualifying_financial_results_6k_values"
            ],
            "unknown_value_statuses": lineage[
                "unknown_value_statuses"
            ],
        }
        hard_pass = all(
            hard_gates[name] == expected
            for name, expected in context.hard_gate_expectations.items()
        )
        integrity_pass = all(value == 0 for value in integrity.values())
        timing = _timing_checks(
            hard_gates=hard_gates,
            expected_hard_gates=context.hard_gate_expectations,
            integrity_checks=integrity,
        )
        counts = {
            "raw_rows": raw_rows,
            "raw_feature_columns": raw_feature_columns,
            "feature_families": len(resolved_lineage),
            "lineage_rows": lineage["lineage_rows"],
            "present_values": lineage["present_values"],
            "missing_values": (
                lineage["lineage_rows"] - lineage["present_values"]
            ),
            "structural_missing_values": lineage[
                "structural_missing_values"
            ],
        }
        summary = _summary(
            build_id=build_id,
            context=context,
            counts=counts,
            hard_gates=hard_gates,
            structural_missingness=structural,
            passed=hard_pass and integrity_pass,
        )
        artifacts = FeatureStoreQualityArtifacts(
            feature_missingness=missingness,
            feature_coverage=coverage,
            feature_outliers=outliers,
            feature_timing_checks=timing,
            feature_source_coverage=source_coverage,
            quality_summary=summary,
        )
    finally:
        connection.close()
    if require_pass and not artifacts.passed:
        failed = [
            row["check_name"]
            for row in artifacts.feature_timing_checks.to_pylist()
            if not row["passed"]
        ]
        raise FeatureStoreQualityError(
            "Stage 6 quality gates failed: " + ", ".join(failed)
        )
    return artifacts
