"""Pure Stage 7 quality, split, timing, and summary builders.

The functions in this module consume already-derived development label rows.
They do not read source data, construct forward outcomes, fit models, or
publish artifacts.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pyarrow as pa

from portfolio_bot.features.label_store_contract import (
    LabelStoreContractError,
    published_schemas,
)


@dataclass(frozen=True)
class MonthLabelQuality:
    """The three quality tables derived from one signal-month audit slice."""

    label_coverage: pa.Table
    label_status_counts: pa.Table
    label_distribution: pa.Table


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LabelStoreContractError(message)


def _schema(
    rules: Mapping[str, Any],
    table_name: str,
) -> pa.Schema:
    schemas = published_schemas(rules)
    _require(
        table_name in schemas,
        f"Missing frozen Stage 7 schema: {table_name}",
    )
    return schemas[table_name]


def _require_table_schema(
    table: pa.Table,
    schema: pa.Schema,
    label: str,
) -> None:
    _require(isinstance(table, pa.Table), f"{label} is not an Arrow table")
    _require(
        table.schema.equals(schema, check_metadata=False),
        f"{label} schema differs from the frozen Stage 7 schema",
    )
    for field in schema:
        if not field.nullable:
            _require(
                table[field.name].null_count == 0,
                f"{label} has nulls in non-null field: {field.name}",
            )


def _require_no_nonfinite(table: pa.Table, label: str) -> None:
    for field in table.schema:
        if not pa.types.is_floating(field.type):
            continue
        for value in table[field.name].to_pylist():
            if value is not None:
                _require(
                    math.isfinite(float(value)),
                    f"{label} has a nonfinite value: {field.name}",
                )


def _sort_atom(value: object) -> object:
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, dt.date):
        return value.toordinal()
    return value


def _row_sort_key(
    row: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[object, ...]:
    return tuple(_sort_atom(row[name]) for name in names)


def _table_keys(
    table: pa.Table,
    names: Sequence[str],
) -> list[tuple[object, ...]]:
    columns = [table[name].to_pylist() for name in names]
    return [
        tuple(column[index] for column in columns)
        for index in range(table.num_rows)
    ]


def _require_unique_sorted_keys(
    table: pa.Table,
    names: Sequence[str],
    label: str,
) -> None:
    keys = _table_keys(table, names)
    _require(
        all(all(value is not None for value in key) for key in keys),
        f"{label} has a null primary key",
    )
    _require(
        len(keys) == len(set(keys)),
        f"{label} has duplicate primary keys",
    )
    normalized = [
        tuple(_sort_atom(value) for value in key)
        for key in keys
    ]
    _require(
        all(right >= left for left, right in zip(normalized, normalized[1:])),
        f"{label} is not sorted by its frozen key",
    )


def _arrow_table(
    rows: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
    *,
    key: Sequence[str],
    label: str,
) -> pa.Table:
    try:
        table = pa.Table.from_pylist(list(rows), schema=schema)
    except (ArrowError, TypeError, ValueError) as exc:
        raise LabelStoreContractError(
            f"Cannot construct {label}"
        ) from exc
    _require_table_schema(table, schema, label)
    _require_no_nonfinite(table, label)
    _require_unique_sorted_keys(table, key, label)
    return table


try:
    ArrowError = pa.ArrowException
except AttributeError:  # pragma: no cover - compatibility for older Arrow
    ArrowError = Exception


def _require_conventions(rules: Mapping[str, Any]) -> None:
    conventions = rules.get("implementation_conventions")
    _require(
        isinstance(conventions, Mapping),
        "Stage 7 implementation conventions are missing",
    )
    status_counts = conventions.get("status_counts")
    coverage = conventions.get("coverage")
    distribution = conventions.get("distribution")
    split_flags = conventions.get("split_flags")
    _require(
        isinstance(status_counts, Mapping)
        and status_counts.get("density")
        == "dense_zero_filled_signal_date_by_dimension_by_allowed_status_domain"
        and status_counts.get("denominator")
        == "all audit-spine rows in the signal date",
        "Stage 7 status-count convention differs",
    )
    _require(
        isinstance(coverage, Mapping)
        and coverage.get("denominator") == "long-eligible audit rows"
        and coverage.get("zero_denominator_rate") == 0.0,
        "Stage 7 coverage convention differs",
    )
    _require(
        isinstance(distribution, Mapping)
        and distribution.get("population")
        == "long-eligible processed rows within signal date"
        and distribution.get("missing_count")
        == "population rows minus finite_count"
        and distribution.get("p01_p99_algorithm")
        == "Hyndman-Fan type 7 linear interpolation"
        and distribution.get("mean_primitive")
        == "math.fsum divided by finite_count"
        and distribution.get("zero_finite_statistics") == "null",
        "Stage 7 distribution convention differs",
    )
    _require(
        isinstance(split_flags, Mapping)
        and split_flags.get("included_in_training")
        == (
            "role TRAIN_ELIGIBLE and primary month valid and "
            "month available by fit"
        )
        and split_flags.get("included_in_validation")
        == "role VALIDATION and primary month valid"
        and split_flags.get("purged_or_embargoed")
        == (
            "role PURGED or holding month equals the frozen embargo "
            "holding month"
        )
        and split_flags.get("embargo_boundary_checked_outside_membership")
        is True,
        "Stage 7 split-flag convention differs",
    )


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


def _type7_quantile(ordered: Sequence[float], probability: float) -> float:
    _require(bool(ordered), "Cannot calculate a quantile of an empty vector")
    _require(
        0.0 <= probability <= 1.0,
        "Stage 7 quantile probability is outside [0,1]",
    )
    if len(ordered) == 1:
        return _positive_zero(float(ordered[0]))
    location = (len(ordered) - 1) * probability
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    fraction = location - lower
    if lower == upper:
        return _positive_zero(float(ordered[lower]))
    result = math.fsum(
        (
            float(ordered[lower]),
            fraction * (float(ordered[upper]) - float(ordered[lower])),
        )
    )
    _require(math.isfinite(result), "Stage 7 quantile is nonfinite")
    return _positive_zero(result)


def _single_month_identity(audit: pa.Table) -> tuple[dt.date, str]:
    signal_dates = set(audit["signal_date"].to_pylist())
    holding_months = set(audit["holding_month"].to_pylist())
    _require(audit.num_rows > 0, "Stage 7 month audit is empty")
    _require(
        len(signal_dates) == 1 and None not in signal_dates,
        "Stage 7 month audit spans multiple signal dates",
    )
    _require(
        len(holding_months) == 1 and None not in holding_months,
        "Stage 7 month audit spans multiple holding months",
    )
    signal_date = next(iter(signal_dates))
    holding_month = next(iter(holding_months))
    _require(
        isinstance(signal_date, dt.date) and isinstance(holding_month, str),
        "Stage 7 month identity has invalid types",
    )
    _validate_holding_month(holding_month)
    return signal_date, holding_month


def _validate_status_columns(
    audit: pa.Table,
    rules: Mapping[str, Any],
) -> None:
    domains = rules.get("value_domains")
    _require(isinstance(domains, Mapping), "Stage 7 value domains are missing")
    for column_name in (
        "outcome_status",
        "primary_label_status",
        "magnitude_label_status",
    ):
        allowed = domains.get(column_name)
        _require(
            isinstance(allowed, list)
            and bool(allowed)
            and len(allowed) == len(set(allowed)),
            f"Invalid Stage 7 status domain: {column_name}",
        )
        observed = set(audit[column_name].to_pylist())
        _require(
            observed.issubset(set(allowed)),
            f"Stage 7 audit has an unknown {column_name}",
        )


def build_month_label_quality(
    audit: pa.Table,
    primary_month_valid: bool,
    rules: Mapping[str, Any],
) -> MonthLabelQuality:
    """Build coverage, dense status counts, and distributions for one month."""

    _require_conventions(rules)
    _require(
        isinstance(primary_month_valid, bool),
        "Stage 7 primary-month-valid flag must be Boolean",
    )
    audit_schema = _schema(rules, "development_label_audit.parquet")
    _require_table_schema(audit, audit_schema, "Stage 7 month audit")
    _require_no_nonfinite(audit, "Stage 7 month audit")
    _require_unique_sorted_keys(
        audit,
        ("signal_date", "permno"),
        "Stage 7 month audit",
    )
    _validate_status_columns(audit, rules)
    signal_date, holding_month = _single_month_identity(audit)

    long_flags = [bool(value) for value in audit["long_eligible"].to_pylist()]
    beta_flags = [bool(value) for value in audit["beta_complete"].to_pylist()]
    outcome_statuses = audit["outcome_status"].to_pylist()
    primary_statuses = audit["primary_label_status"].to_pylist()
    magnitude_statuses = audit["magnitude_label_status"].to_pylist()
    eligible_count = sum(long_flags)
    beta_count = sum(
        long and complete
        for long, complete in zip(long_flags, beta_flags, strict=True)
    )
    outcome_count = sum(
        long and status == "PRESENT"
        for long, status in zip(long_flags, outcome_statuses, strict=True)
    )
    primary_count = sum(
        long and status == "PRESENT"
        for long, status in zip(long_flags, primary_statuses, strict=True)
    )
    magnitude_count = sum(
        long and status == "PRESENT"
        for long, status in zip(long_flags, magnitude_statuses, strict=True)
    )

    def rate(numerator: int) -> float:
        if eligible_count == 0:
            return 0.0
        return _positive_zero(numerator / eligible_count)

    coverage_rows = [
        {
            "signal_date": signal_date,
            "holding_month": holding_month,
            "eligible_count": eligible_count,
            "beta_complete_count": beta_count,
            "outcome_present_count": outcome_count,
            "primary_present_count": primary_count,
            "magnitude_present_count": magnitude_count,
            "eligible_denominator": eligible_count,
            "outcome_coverage_rate": rate(outcome_count),
            "primary_coverage_rate": rate(primary_count),
            "magnitude_coverage_rate": rate(magnitude_count),
            "primary_month_valid": primary_month_valid,
        }
    ]
    coverage = _arrow_table(
        coverage_rows,
        _schema(rules, "label_coverage.parquet"),
        key=("signal_date",),
        label="Stage 7 label coverage",
    )

    domains = rules["value_domains"]
    dimension_specs = (
        ("outcome", "outcome_status", "outcome_status"),
        ("primary", "primary_label_status", "primary_label_status"),
        ("magnitude", "magnitude_label_status", "magnitude_label_status"),
    )
    status_rows: list[dict[str, Any]] = []
    dimension_total = audit.num_rows
    for dimension, column_name, domain_name in dimension_specs:
        values = audit[column_name].to_pylist()
        counts = {status: 0 for status in domains[domain_name]}
        for value in values:
            counts[value] += 1
        for status, row_count in counts.items():
            fraction = (
                0.0
                if dimension_total == 0
                else _positive_zero(row_count / dimension_total)
            )
            status_rows.append(
                {
                    "signal_date": signal_date,
                    "holding_month": holding_month,
                    "status_dimension": dimension,
                    "status": status,
                    "row_count": row_count,
                    "dimension_total": dimension_total,
                    "row_fraction": fraction,
                }
            )
    status_key = ("signal_date", "status_dimension", "status")
    status_rows.sort(key=lambda row: _row_sort_key(row, status_key))
    status_counts = _arrow_table(
        status_rows,
        _schema(rules, "label_status_counts.parquet"),
        key=status_key,
        label="Stage 7 label status counts",
    )

    target_names = rules["value_domains"]["distribution_target_name"]
    _require(
        isinstance(target_names, list)
        and len(target_names) == len(set(target_names)) == 5,
        "Stage 7 distribution target domain differs",
    )
    distribution_rows: list[dict[str, Any]] = []
    for target_name in target_names:
        _require(
            target_name in audit.schema.names,
            f"Stage 7 distribution target is absent: {target_name}",
        )
        population = [
            value
            for value, long in zip(
                audit[target_name].to_pylist(),
                long_flags,
                strict=True,
            )
            if long
        ]
        finite = sorted(float(value) for value in population if value is not None)
        finite_count = len(finite)
        statistics: dict[str, float | None]
        if finite_count == 0:
            statistics = {
                "minimum": None,
                "p01": None,
                "median": None,
                "mean": None,
                "p99": None,
                "maximum": None,
            }
        else:
            mean = math.fsum(finite) / finite_count
            _require(math.isfinite(mean), "Stage 7 distribution mean is nonfinite")
            statistics = {
                "minimum": _positive_zero(finite[0]),
                "p01": _type7_quantile(finite, 0.01),
                "median": _type7_quantile(finite, 0.50),
                "mean": _positive_zero(mean),
                "p99": _type7_quantile(finite, 0.99),
                "maximum": _positive_zero(finite[-1]),
            }
        distribution_rows.append(
            {
                "signal_date": signal_date,
                "holding_month": holding_month,
                "target_name": target_name,
                "finite_count": finite_count,
                "missing_count": len(population) - finite_count,
                **statistics,
            }
        )
    distribution_key = ("signal_date", "target_name")
    distribution_rows.sort(
        key=lambda row: _row_sort_key(row, distribution_key)
    )
    distribution = _arrow_table(
        distribution_rows,
        _schema(rules, "label_distribution.parquet"),
        key=distribution_key,
        label="Stage 7 label distribution",
    )
    return MonthLabelQuality(
        label_coverage=coverage,
        label_status_counts=status_counts,
        label_distribution=distribution,
    )


def _validate_holding_month(value: object) -> str:
    _require(isinstance(value, str), "Stage 7 holding month is not text")
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise LabelStoreContractError(
            f"Invalid Stage 7 holding month: {value!r}"
        ) from exc
    _require(
        parsed.strftime("%Y-%m") == value,
        f"Invalid Stage 7 holding month: {value!r}",
    )
    return value


def _folds(rules: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    splits = rules.get("splits")
    _require(isinstance(splits, Mapping), "Stage 7 split rules are missing")
    folds = splits.get("folds")
    _require(
        isinstance(folds, list)
        and len(folds) == int(rules["scope"]["required_folds"]) == 4,
        "Stage 7 fold count differs",
    )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in folds:
        _require(isinstance(raw, Mapping), "Invalid Stage 7 fold specification")
        fold_id = raw.get("fold_id")
        _require(
            isinstance(fold_id, str) and fold_id and fold_id not in seen,
            "Duplicate or invalid Stage 7 fold ID",
        )
        seen.add(fold_id)
        try:
            fit_signal_date = dt.date.fromisoformat(
                str(raw["fit_signal_date"])
            )
        except (KeyError, ValueError) as exc:
            raise LabelStoreContractError(
                f"Invalid Stage 7 fit date: {fold_id}"
            ) from exc
        train_last = _validate_holding_month(
            raw.get("train_last_holding_month")
        )
        purged = _validate_holding_month(raw.get("purged_holding_month"))
        validation_first = _validate_holding_month(
            raw.get("validation_first_holding_month")
        )
        validation_last = _validate_holding_month(
            raw.get("validation_last_holding_month")
        )
        _require(
            train_last < purged < validation_first <= validation_last,
            f"Stage 7 fold boundaries are invalid: {fold_id}",
        )
        result.append(
            {
                "fold_id": fold_id,
                "fit_signal_date": fit_signal_date,
                "train_last_holding_month": train_last,
                "purged_holding_month": purged,
                "validation_first_holding_month": validation_first,
                "validation_last_holding_month": validation_last,
            }
        )
    return tuple(result)


def _split_role(holding_month: str, fold: Mapping[str, Any]) -> str:
    if holding_month <= fold["train_last_holding_month"]:
        return "TRAIN_ELIGIBLE"
    if holding_month == fold["purged_holding_month"]:
        return "PURGED"
    if (
        fold["validation_first_holding_month"]
        <= holding_month
        <= fold["validation_last_holding_month"]
    ):
        return "VALIDATION"
    return "OUTSIDE"


def build_split_membership(
    month_summaries: pa.Table,
    rules: Mapping[str, Any],
) -> pa.Table:
    """Expand month summaries into the four frozen development-fold rows."""

    _require_conventions(rules)
    summary_schema = _schema(
        rules, "development_label_month_summary.parquet"
    )
    _require_table_schema(
        month_summaries,
        summary_schema,
        "Stage 7 month summaries",
    )
    _require_no_nonfinite(month_summaries, "Stage 7 month summaries")
    _require_unique_sorted_keys(
        month_summaries,
        ("signal_date",),
        "Stage 7 month summaries",
    )
    allowed_month_statuses = set(rules["value_domains"]["month_status"])
    observed_statuses = set(month_summaries["month_status"].to_pylist())
    _require(
        observed_statuses.issubset(allowed_month_statuses),
        "Stage 7 month summaries contain an unknown status",
    )
    rows = month_summaries.to_pylist()
    seen_holding_months: set[str] = set()
    for row in rows:
        holding_month = _validate_holding_month(row["holding_month"])
        _require(
            holding_month not in seen_holding_months,
            "Stage 7 month summaries duplicate a holding month",
        )
        seen_holding_months.add(holding_month)
        expected_valid = row["month_status"] == "PRESENT"
        _require(
            bool(row["primary_month_valid"]) is expected_valid,
            "Stage 7 month-valid flag conflicts with month status",
        )
        _require(
            not expected_valid
            or row["month_label_available_date"] is not None,
            "Valid Stage 7 month has no availability date",
        )

    folds = _folds(rules)
    embargo_month = _validate_holding_month(
        rules["splits"]["development_embargo_holding_month"]
    )
    split_rows: list[dict[str, Any]] = []
    for month in rows:
        holding_month = month["holding_month"]
        month_valid = bool(month["primary_month_valid"])
        available_date = month["month_label_available_date"]
        for fold in folds:
            role = _split_role(holding_month, fold)
            available_by_fit = (
                available_date is not None
                and available_date <= fold["fit_signal_date"]
            )
            split_rows.append(
                {
                    "fold_id": fold["fold_id"],
                    "signal_date": month["signal_date"],
                    "holding_month": holding_month,
                    "fit_signal_date": fold["fit_signal_date"],
                    "role": role,
                    "month_label_available_date": available_date,
                    "available_by_fit": available_by_fit,
                    "included_in_training": (
                        role == "TRAIN_ELIGIBLE"
                        and month_valid
                        and available_by_fit
                    ),
                    "included_in_validation": (
                        role == "VALIDATION" and month_valid
                    ),
                    "purged_or_embargoed": (
                        role == "PURGED" or holding_month == embargo_month
                    ),
                }
            )
    key = ("fold_id", "signal_date")
    split_rows.sort(key=lambda row: _row_sort_key(row, key))
    return _arrow_table(
        split_rows,
        _schema(rules, "label_split_membership.parquet"),
        key=key,
        label="Stage 7 label split membership",
    )


def _integer_quality_gates(
    rules: Mapping[str, Any],
) -> dict[str, int]:
    gates = rules.get("quality_gates")
    _require(isinstance(gates, Mapping), "Stage 7 quality gates are missing")
    result = {
        str(name): int(value)
        for name, value in gates.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    _require(
        len(result) == 57,
        "Stage 7 integer hard-gate count differs",
    )
    _require(
        all(value >= 0 for value in result.values()),
        "Stage 7 hard-gate expectation is negative",
    )
    return result


def _observed_gate_values(
    observed_gates: Mapping[str, int],
    rules: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    _require(
        isinstance(observed_gates, Mapping),
        "Stage 7 observed gates are not a mapping",
    )
    expected = _integer_quality_gates(rules)
    _require(
        set(observed_gates) == set(expected),
        "Stage 7 observed hard-gate key set differs",
    )
    observed: dict[str, int] = {}
    for name, value in observed_gates.items():
        _require(
            isinstance(name, str) and name,
            "Stage 7 observed gate has an invalid name",
        )
        _require(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0,
            f"Stage 7 observed gate is not a nonnegative integer: {name}",
        )
        observed[name] = int(value)
    return observed, expected


def build_label_timing_checks(
    observed_gates: Mapping[str, int],
    rules: Mapping[str, Any],
) -> pa.Table:
    """Create one deterministic timing row for every integer hard gate."""

    observed, expected = _observed_gate_values(observed_gates, rules)
    names = sorted(expected, key=lambda value: value.encode("utf-8"))
    rows = [
        {
            "check_name": name,
            "observed_count": observed[name],
            "expected_count": expected[name],
            "passed": observed[name] == expected[name],
            "severity": "CRITICAL",
            "details": (
                "Frozen Stage 7 hard gate; "
                f"expected={expected[name]}; observed={observed[name]}"
            ),
        }
        for name in names
    ]
    return _arrow_table(
        rows,
        _schema(rules, "label_timing_checks.parquet"),
        key=("check_name",),
        label="Stage 7 label timing checks",
    )


def _sorted_int_mapping(
    value: Mapping[str, int],
    label: str,
) -> dict[str, int]:
    _require(isinstance(value, Mapping) and bool(value), f"{label} is empty")
    result: dict[str, int] = {}
    for key in sorted(value, key=lambda item: str(item).encode("utf-8")):
        item = value[key]
        _require(
            isinstance(key, str) and key,
            f"{label} has an invalid key",
        )
        _require(
            isinstance(item, int)
            and not isinstance(item, bool)
            and item >= 0,
            f"{label} has a non-integer or negative value: {key}",
        )
        result[key] = int(item)
    return result


def _sorted_json_mapping(
    value: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping) and bool(value), f"{label} is empty")
    result = {
        key: value[key]
        for key in sorted(value, key=lambda item: str(item).encode("utf-8"))
    }
    _require(
        all(isinstance(key, str) and key for key in result),
        f"{label} has an invalid key",
    )
    try:
        json.dumps(result, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LabelStoreContractError(f"{label} is not finite JSON") from exc
    return result


def build_quality_summary(
    build_id: str,
    contract_id: str,
    policy_id: str,
    observed_gates: Mapping[str, int],
    counts: Mapping[str, int],
    input_builds: Mapping[str, Any],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact-key Stage 7 quality summary payload."""

    _require_conventions(rules)
    _require(
        isinstance(build_id, str) and bool(build_id),
        "Stage 7 quality-summary build ID is invalid",
    )
    _require(
        contract_id == rules.get("contract_id"),
        "Stage 7 quality-summary contract ID differs",
    )
    _require(
        isinstance(policy_id, str) and bool(policy_id),
        "Stage 7 quality-summary policy ID is invalid",
    )
    observed, expected = _observed_gate_values(observed_gates, rules)
    ordered_observed = {
        name: observed[name]
        for name in sorted(observed, key=lambda value: value.encode("utf-8"))
    }
    ordered_counts = _sorted_int_mapping(counts, "Stage 7 summary counts")
    ordered_inputs = _sorted_json_mapping(
        input_builds, "Stage 7 summary input builds"
    )
    flags = {
        str(name): bool(value)
        for name, value in rules["quality_gates"].items()
        if isinstance(value, bool)
    }
    _require(
        flags
        == {
            "label_coverage_is_descriptive_not_pass_fail": True,
            "missing_outcomes_are_never_zero_filled": True,
        },
        "Stage 7 quality policy flags differ",
    )
    summary = {
        "schema_version": 1,
        "contract_id": contract_id,
        "policy_id": policy_id,
        "build_id": build_id,
        "status": (
            "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
            if all(observed[name] == expected[name] for name in expected)
            else "PRIMARY_FAILED"
        ),
        "scope": {
            key: rules["scope"][key]
            for key in sorted(
                rules["scope"],
                key=lambda value: value.encode("utf-8"),
            )
        },
        "counts": ordered_counts,
        "hard_gates": ordered_observed,
        "quality_policy_flags": flags,
        "input_builds": ordered_inputs,
    }
    required_keys = rules["publication_payloads"][
        "quality_summary_exact_keys"
    ]
    _require(
        set(summary) == set(required_keys),
        "Stage 7 quality-summary keys differ",
    )
    try:
        json.dumps(summary, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LabelStoreContractError(
            "Stage 7 quality summary is not finite JSON"
        ) from exc
    return summary
