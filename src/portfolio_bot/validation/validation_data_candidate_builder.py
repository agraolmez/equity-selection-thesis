"""Scope-aware Stage 11 feature and risk/cost candidate materialization.

The builder accepts only already-normalized, in-memory predictor inputs.  It
does not resolve or scan raw vendor paths and it has no label, outcome,
forward-return, portfolio, trade, or P&L interface.  Feature values are the
outputs of the unchanged Stage 6 family formulas; risk and cost calculations
delegate to the unchanged Stage 8 numerical implementation through a new
date-scope adapter.

The internal 2014-11-28 row exists solely to seed the exact one-month
OptionMetrics lag.  It is validated separately and can never enter a returned
or written payload.
"""

from __future__ import annotations

import datetime as dt
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.features.feature_lineage import (
    FeatureFamilyResult,
    FeatureLineage,
)
from portfolio_bot.risk_cost import materialization as frozen_stage8
from portfolio_bot.validation.validation_data_contract import (
    FIRST_PROHIBITED_SIGNAL,
    INTERNAL_CONTEXT_SIGNAL,
    LAST_SIGNAL,
    ValidationDataContract,
    unexpected_forbidden_columns,
)
from portfolio_bot.validation.validation_feature_semantics import (
    semantic_value_is_missing,
)


SEGMENT_FEATURES = (
    "business_segment_count",
    "geographic_segment_count",
    "business_segment_revenue_hhi",
    "business_segment_revenue_hhi_change_1y",
)
OPTIONMETRICS_FAMILY = "optionmetrics"
RETURN_COLUMNS = (
    "permno",
    "calendar_date",
    "return_span_end_date",
    "outcome_availability_date",
    "covered_trading_sessions",
    "return_history_usable",
    "authoritative_total_return",
)
RISK_SPINE_COLUMNS = (
    "signal_date",
    "permno",
    "permco",
    "long_eligible",
    "short_eligible",
    "primary_exchange",
    "ff12_sector_category",
    "domicile_country_category",
    "median_dollar_volume_20d",
    "median_estimated_full_spread_20d",
)
FACTOR_COLUMNS = (
    "observation_date",
    *frozen_stage8.FACTOR_NAMES,
    "rf",
    "public_availability_date",
)


class ValidationDataCandidateBuilderError(RuntimeError):
    """Raised when a scope, schema, lineage, or numerical gate fails closed."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationDataCandidateBuilderError(message)


def _date(value: object, label: str) -> dt.date:
    _check(
        isinstance(value, dt.date) and not isinstance(value, dt.datetime),
        f"{label} must be an exact date",
    )
    return value


def _integer(value: object, label: str) -> int:
    _check(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return int(value)


def _finite_or_nonnumeric(value: object) -> bool:
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _validate_lineage(
    value: object,
    lineage: FeatureLineage,
    *,
    signal_date: dt.date,
    feature_name: str,
) -> None:
    semantic_missing = semantic_value_is_missing(feature_name, value)
    if lineage.value_status == "PRESENT":
        _check(
            not semantic_missing,
            f"Present feature has the frozen semantic-missing encoding: {feature_name}",
        )
        _check(
            _finite_or_nonnumeric(value),
            f"Present feature is nonfinite: {feature_name}",
        )
        _check(
            lineage.max_observation_date is not None
            and lineage.max_public_availability_date is not None
            and lineage.max_observation_date <= signal_date
            and lineage.max_public_availability_date <= signal_date,
            f"Feature lineage exceeds its signal date: {feature_name}",
        )
        _check(
            lineage.stale_after_date is None or signal_date <= lineage.stale_after_date,
            f"Stale feature was published: {feature_name}",
        )
    else:
        _check(
            semantic_missing,
            f"Missing lineage carries a semantically present value: {feature_name}",
        )
        _check(
            lineage.max_observation_date is None
            and lineage.max_public_availability_date is None
            and lineage.stale_after_date is None,
            f"Missing feature carries lineage dates: {feature_name}",
        )


@dataclass(frozen=True, slots=True)
class ValidationFeatureRow:
    """One scored security row with all seven frozen family results."""

    signal_date: dt.date
    permno: int
    permco: int
    long_eligible: bool
    short_eligible: bool
    primary_exchange: str | None
    share_type: str | None
    family_results: Mapping[str, FeatureFamilyResult]


@dataclass(frozen=True, slots=True)
class OptionMetricsContextRow:
    """One non-publishable 2014-11-28 OptionMetrics family result."""

    signal_date: dt.date
    permno: int
    result: FeatureFamilyResult


@dataclass(frozen=True, slots=True)
class OptionMetricsContextReceipt:
    signal_date: dt.date
    input_rows: int
    unique_securities: int
    purpose: str
    published_rows: int
    label_outcome_or_forward_rows_read: int
    final_period_rows_read: int


@dataclass(frozen=True, slots=True)
class ValidationFeatureMaterialization:
    predictor: pa.Table
    lineage_by_family: Mapping[str, pa.Table]
    context_receipt: OptionMetricsContextReceipt
    quality: Mapping[str, int | str | bool]


@dataclass(frozen=True, slots=True)
class StreamingFeatureWriteReceipt:
    destination: Path
    feature_rows: int
    lineage_rows: int
    long_eligible_rows: int
    signal_dates: int
    maximum_feature_batch_rows: int
    maximum_lineage_batch_rows: int
    context_receipt: OptionMetricsContextReceipt


ReturnBatchFactory = Callable[[int], Iterable[pa.RecordBatch]]


@dataclass(frozen=True, slots=True)
class ValidationRiskHistoryInputs:
    """Closed in-memory input interface for the frozen Stage 8 formulas."""

    trading_calendar: pa.Table
    factors: pa.Table
    return_batch_factory: ReturnBatchFactory
    label_rows_read: int = 0
    outcome_rows_read: int = 0
    final_period_rows_read: int = 0

    def read_trading_calendar(self) -> pa.Table:
        return self.trading_calendar

    def read_factors(self) -> pa.Table:
        return self.factors

    def iter_return_batches(self, *, batch_rows: int) -> Iterable[pa.RecordBatch]:
        yield from self.return_batch_factory(batch_rows)


@dataclass(frozen=True, slots=True)
class ValidationRiskCostMaterialization:
    audit: pa.Table
    processed: pa.Table
    factor_covariance: pa.Table
    month_summary: pa.Table
    quality: Mapping[str, int | bool]


@dataclass(frozen=True, slots=True)
class ValidationCandidateTables:
    payloads: Mapping[str, pa.Table]
    feature_quality: Mapping[str, int | str | bool]
    risk_cost_quality: Mapping[str, int | bool]
    context_receipt: OptionMetricsContextReceipt


def _validate_family_result(
    result: FeatureFamilyResult,
    *,
    feature_names: tuple[str, ...],
    signal_date: dt.date,
) -> None:
    _check(tuple(result.values) == feature_names, "Feature-family value order differs")
    _check(
        tuple(result.lineage) == feature_names, "Feature-family lineage order differs"
    )
    for name in feature_names:
        _validate_lineage(
            result.values[name],
            result.lineage[name],
            signal_date=signal_date,
            feature_name=name,
        )


def _validate_nonnullable_output(
    output: Mapping[str, object], schema: pa.Schema
) -> None:
    missing = [
        field.name
        for field in schema
        if not field.nullable and output.get(field.name) is None
    ]
    _check(not missing, f"Nonnullable frozen feature fields are null: {missing}")


def _validate_context_rows(
    rows: Sequence[OptionMetricsContextRow],
    contract: ValidationDataContract,
) -> OptionMetricsContextReceipt:
    _check(bool(rows), "The exact OptionMetrics context signal is missing")
    names = contract.family_feature_names[OPTIONMETRICS_FAMILY]
    previous: int | None = None
    for row in sorted(rows, key=lambda item: item.permno):
        _check(
            row.signal_date == INTERNAL_CONTEXT_SIGNAL,
            "OptionMetrics context date differs",
        )
        permno = _integer(row.permno, "context PERMNO")
        _check(previous is None or permno > previous, "Context PERMNOs are duplicate")
        previous = permno
        _validate_family_result(
            row.result,
            feature_names=names,
            signal_date=INTERNAL_CONTEXT_SIGNAL,
        )
    return OptionMetricsContextReceipt(
        signal_date=INTERNAL_CONTEXT_SIGNAL,
        input_rows=len(rows),
        unique_securities=len(rows),
        purpose="exact_one_month_optionmetrics_lag_only",
        published_rows=0,
        label_outcome_or_forward_rows_read=0,
        final_period_rows_read=0,
    )


def materialize_validation_features(
    scored_rows: Sequence[ValidationFeatureRow],
    optionmetrics_context_rows: Sequence[OptionMetricsContextRow],
    contract: ValidationDataContract,
) -> ValidationFeatureMaterialization:
    """Assemble exact 170-column predictors and seven lineage payloads."""

    context_receipt = _validate_context_rows(optionmetrics_context_rows, contract)
    ordered = sorted(scored_rows, key=lambda item: (item.signal_date, item.permno))
    _check(bool(ordered), "Validation feature input is empty")
    expected_dates = frozenset(contract.scored_signal_dates)
    observed_dates: set[dt.date] = set()
    previous: tuple[dt.date, int] | None = None
    physical_rows: list[dict[str, object]] = []
    lineage_rows: dict[str, list[dict[str, object]]] = {
        family: [] for family in contract.family_feature_names
    }
    segment_nonmissing = 0
    for item in ordered:
        signal = _date(item.signal_date, "feature signal_date")
        permno = _integer(item.permno, "feature PERMNO")
        _integer(item.permco, "feature PERMCO")
        _check(
            signal in expected_dates, f"Feature signal escaped Stage 11 scope: {signal}"
        )
        _check(
            signal != INTERNAL_CONTEXT_SIGNAL and signal < FIRST_PROHIBITED_SIGNAL,
            "Context/final feature was published",
        )
        key = (signal, permno)
        _check(previous is None or key > previous, "Feature keys are duplicate")
        previous = key
        observed_dates.add(signal)
        _check(
            type(item.long_eligible) is bool and type(item.short_eligible) is bool,
            "Eligibility values must be booleans",
        )
        _check(
            not item.short_eligible or item.long_eligible,
            "Short eligibility must imply long eligibility",
        )
        _check(
            tuple(item.family_results) == tuple(contract.family_feature_names),
            "Feature-family set or order differs",
        )
        output: dict[str, object] = {
            "signal_date": signal,
            "permno": permno,
            "permco": item.permco,
            "long_eligible": item.long_eligible,
            "short_eligible": item.short_eligible,
            "primary_exchange": item.primary_exchange,
            "share_type": item.share_type,
        }
        for family, names in contract.family_feature_names.items():
            result = item.family_results[family]
            _validate_family_result(result, feature_names=names, signal_date=signal)
            for name in names:
                value = result.values[name]
                lineage = result.lineage[name]
                output[name] = value
                lineage_rows[family].append(
                    {
                        "signal_date": signal,
                        "permno": permno,
                        "feature_name": name,
                        "max_observation_date": lineage.max_observation_date,
                        "max_public_availability_date": lineage.max_public_availability_date,
                        "stale_after_date": lineage.stale_after_date,
                        "value_status": lineage.value_status,
                    }
                )
        for name in SEGMENT_FEATURES:
            segment_nonmissing += int(output[name] is not None)
            lineage = item.family_results["ownership_filings_events_segments"].lineage[
                name
            ]
            _check(
                output[name] is None
                and lineage.value_status == "MISSING_SOURCE_UNAVAILABLE",
                "Frozen-inactive segment support was changed",
            )
        _validate_nonnullable_output(output, contract.raw_feature_schema)
        physical_rows.append(output)

    _check(
        observed_dates == expected_dates,
        "Validation features do not contain the exact 59 scored signals",
    )
    _check(segment_nonmissing == 0, "Frozen-inactive segment values were populated")
    try:
        predictor = pa.Table.from_pylist(
            physical_rows, schema=contract.raw_feature_schema
        )
        lineage_tables = {
            family: pa.Table.from_pylist(rows, schema=contract.lineage_schema)
            for family, rows in lineage_rows.items()
        }
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ValidationDataCandidateBuilderError(
            "Feature values do not conform to the frozen Stage 6 schema"
        ) from exc
    _check(
        tuple(predictor.schema.names) == tuple(contract.raw_feature_schema.names),
        "Predictor schema order differs",
    )
    _check(len(predictor.schema) == 170, "Predictor schema is not exactly 170 columns")
    lineage_count = sum(table.num_rows for table in lineage_tables.values())
    _check(
        lineage_count == predictor.num_rows * 163, "Feature lineage arithmetic differs"
    )
    quality: Mapping[str, int | str | bool] = MappingProxyType(
        {
            "feature_rows": predictor.num_rows,
            "signal_dates": len(observed_dates),
            "physical_columns": len(predictor.schema),
            "economic_features": 163,
            "lineage_rows": lineage_count,
            "context_rows_published": 0,
            "frozen_category_vocabulary_reused": True,
            "validation_vocabulary_expansions": 0,
            "frozen_model_support_changes": 0,
            "inactive_segment_nonmissing_values": segment_nonmissing,
            "label_outcome_or_forward_rows_read": 0,
            "final_period_rows_read": 0,
        }
    )
    return ValidationFeatureMaterialization(
        predictor=predictor,
        lineage_by_family=MappingProxyType(lineage_tables),
        context_receipt=context_receipt,
        quality=quality,
    )


def _materialize_feature_chunk(
    rows: Sequence[ValidationFeatureRow],
    contract: ValidationDataContract,
    *,
    allowed_signal_dates: frozenset[dt.date] | None = None,
) -> tuple[pa.Table, Mapping[str, pa.Table], set[dt.date], int, int]:
    """Materialize one bounded, already ordered chunk without domain closure.

    ``allowed_signal_dates`` exists only for the authenticated development
    replay gate.  Normal validation callers cannot widen the domain because
    they omit it and therefore receive the frozen 59-signal contract scope.
    """

    physical: list[dict[str, object]] = []
    lineage_rows: dict[str, list[dict[str, object]]] = {
        family: [] for family in contract.family_feature_names
    }
    dates: set[dt.date] = set()
    long_rows = 0
    segment_nonmissing = 0
    previous: tuple[dt.date, int] | None = None
    permitted_signals = (
        frozenset(contract.scored_signal_dates)
        if allowed_signal_dates is None
        else allowed_signal_dates
    )
    _check(bool(permitted_signals), "Feature chunk signal allowlist is empty")
    for item in rows:
        signal = _date(item.signal_date, "feature signal_date")
        permno = _integer(item.permno, "feature PERMNO")
        _integer(item.permco, "feature PERMCO")
        _check(
            signal in permitted_signals, "Feature chunk escaped its authenticated scope"
        )
        key = (signal, permno)
        _check(
            previous is None or key > previous,
            "Feature chunk keys are not sorted unique",
        )
        previous = key
        _check(
            type(item.long_eligible) is bool and type(item.short_eligible) is bool,
            "Eligibility values must be booleans",
        )
        _check(
            not item.short_eligible or item.long_eligible,
            "Short eligibility must imply long eligibility",
        )
        _check(
            tuple(item.family_results) == tuple(contract.family_feature_names),
            "Feature-family set or order differs",
        )
        output: dict[str, object] = {
            "signal_date": signal,
            "permno": permno,
            "permco": item.permco,
            "long_eligible": item.long_eligible,
            "short_eligible": item.short_eligible,
            "primary_exchange": item.primary_exchange,
            "share_type": item.share_type,
        }
        for family, names in contract.family_feature_names.items():
            result = item.family_results[family]
            _validate_family_result(result, feature_names=names, signal_date=signal)
            for name in names:
                value = result.values[name]
                lineage = result.lineage[name]
                output[name] = value
                lineage_rows[family].append(
                    {
                        "signal_date": signal,
                        "permno": permno,
                        "feature_name": name,
                        "max_observation_date": lineage.max_observation_date,
                        "max_public_availability_date": lineage.max_public_availability_date,
                        "stale_after_date": lineage.stale_after_date,
                        "value_status": lineage.value_status,
                    }
                )
        for name in SEGMENT_FEATURES:
            lineage = item.family_results["ownership_filings_events_segments"].lineage[
                name
            ]
            segment_nonmissing += int(output[name] is not None)
            _check(
                output[name] is None
                and lineage.value_status == "MISSING_SOURCE_UNAVAILABLE",
                "Frozen-inactive segment support was changed",
            )
        _validate_nonnullable_output(output, contract.raw_feature_schema)
        dates.add(signal)
        long_rows += int(item.long_eligible)
        physical.append(output)
    try:
        feature_table = pa.Table.from_pylist(
            physical, schema=contract.raw_feature_schema
        )
        lineage_tables = MappingProxyType(
            {
                family: pa.Table.from_pylist(values, schema=contract.lineage_schema)
                for family, values in lineage_rows.items()
            }
        )
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ValidationDataCandidateBuilderError(
            "Feature chunk does not conform to the frozen Stage 6 schema"
        ) from exc
    return feature_table, lineage_tables, dates, long_rows, segment_nonmissing


def write_validation_features_streaming(
    scored_batches: Iterable[Sequence[ValidationFeatureRow]],
    optionmetrics_context_rows: Sequence[OptionMetricsContextRow],
    destination: Path,
    contract: ValidationDataContract,
    *,
    maximum_rows_per_chunk: int = 2_048,
) -> StreamingFeatureWriteReceipt:
    """Write the eight feature roles with bounded Python/Arrow memory.

    Providers may materialize one month at a time.  Every provider batch is
    subdivided again, so the largest Arrow feature table is bounded by
    ``maximum_rows_per_chunk`` and the largest lineage table is bounded by that
    row count times the largest frozen family width.
    """

    _check(maximum_rows_per_chunk > 0, "Feature stream chunk size must be positive")
    context = _validate_context_rows(optionmetrics_context_rows, contract)
    root = Path(os.path.abspath(destination.expanduser()))
    created_root = not root.exists()
    if root.exists():
        _check(root.is_dir() and not root.is_symlink(), "Unsafe feature destination")
        _check(not any(root.iterdir()), "Feature destination is not empty")
    else:
        root.mkdir(parents=False, exist_ok=False)
    roles = ("validation_predictor_payload",) + tuple(
        f"validation_feature_lineage_{family}"
        for family in contract.family_feature_names
    )
    writers: dict[str, pq.ParquetWriter] = {}
    partials: dict[str, Path] = {}
    stream_complete = False
    try:
        for role in roles:
            specification = contract.payload_specs[role]
            partial = root / f".{specification.filename}.partial"
            partials[role] = partial
            writers[role] = pq.ParquetWriter(
                partial,
                specification.schema,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
        previous: tuple[dt.date, int] | None = None
        observed_dates: set[dt.date] = set()
        feature_rows = 0
        lineage_count = 0
        long_rows = 0
        segment_nonmissing = 0
        max_feature = 0
        max_lineage = 0
        for provider_batch in scored_batches:
            batch = tuple(provider_batch)
            _check(bool(batch), "Feature provider emitted an empty batch")
            for offset in range(0, len(batch), maximum_rows_per_chunk):
                chunk = batch[offset : offset + maximum_rows_per_chunk]
                first_key = (chunk[0].signal_date, chunk[0].permno)
                _check(
                    previous is None or first_key > previous,
                    "Feature stream order regressed",
                )
                feature, lineages, dates, chunk_long, chunk_segments = (
                    _materialize_feature_chunk(chunk, contract)
                )
                final_key = (chunk[-1].signal_date, chunk[-1].permno)
                previous = final_key
                writers["validation_predictor_payload"].write_table(feature)
                for family, table in lineages.items():
                    writers[f"validation_feature_lineage_{family}"].write_table(table)
                    max_lineage = max(max_lineage, table.num_rows)
                    lineage_count += table.num_rows
                observed_dates.update(dates)
                feature_rows += feature.num_rows
                long_rows += chunk_long
                segment_nonmissing += chunk_segments
                max_feature = max(max_feature, feature.num_rows)
        _check(feature_rows > 0, "Feature stream is empty")
        _check(
            tuple(sorted(observed_dates)) == contract.scored_signal_dates,
            "Feature stream does not contain the exact 59 scored signals",
        )
        _check(
            lineage_count == feature_rows * 163,
            "Feature stream lineage arithmetic differs",
        )
        _check(
            segment_nonmissing == 0, "Feature stream populated frozen-inactive segments"
        )
        stream_complete = True
    finally:
        for writer in writers.values():
            writer.close()
        if not stream_complete:
            for partial in partials.values():
                if partial.is_file() and not partial.is_symlink():
                    partial.unlink()
            if created_root and root.is_dir() and not any(root.iterdir()):
                root.rmdir()
    for role in roles:
        specification = contract.payload_specs[role]
        os.replace(partials[role], root / specification.filename)
    return StreamingFeatureWriteReceipt(
        destination=root,
        feature_rows=feature_rows,
        lineage_rows=lineage_count,
        long_eligible_rows=long_rows,
        signal_dates=len(observed_dates),
        maximum_feature_batch_rows=max_feature,
        maximum_lineage_batch_rows=max_lineage,
        context_receipt=context,
    )


def _validate_risk_inputs(
    inputs: ValidationRiskHistoryInputs,
    contract: ValidationDataContract,
) -> None:
    _check(
        inputs.label_rows_read
        == inputs.outcome_rows_read
        == inputs.final_period_rows_read
        == 0,
        "Risk/cost input reports a prohibited read",
    )
    calendar = inputs.trading_calendar
    _check(
        tuple(calendar.schema.names) == ("calendar_date",),
        "Trading-calendar projection differs",
    )
    days = calendar["calendar_date"].to_pylist()
    _check(
        bool(days) and all(type(value) is dt.date for value in days),
        "Trading calendar is empty or invalid",
    )
    _check(days == sorted(set(days)), "Trading calendar is not sorted unique")
    _check(
        days[-1] == LAST_SIGNAL, "Trading calendar must end at the last scored signal"
    )
    _check(
        set(contract.scored_signal_dates).issubset(days),
        "Trading calendar omits a scored signal",
    )

    factors = inputs.factors
    _check(tuple(factors.schema.names) == FACTOR_COLUMNS, "Factor projection differs")
    observations = factors["observation_date"].to_pylist()
    availability = factors["public_availability_date"].to_pylist()
    _check(bool(observations), "Factor input is empty")
    _check(
        observations == sorted(set(observations)),
        "Factor observations are not sorted unique",
    )
    _check(
        all(
            type(observed) is dt.date
            and type(available) is dt.date
            and observed <= LAST_SIGNAL
            and available <= LAST_SIGNAL
            for observed, available in zip(observations, availability, strict=True)
        ),
        "Factor input contains a post-validation-cutoff row",
    )


def _scope_aware_return_scan(
    inputs: ValidationRiskHistoryInputs,
    target_map: Mapping[int, np.ndarray],
    state: Any,
    factors: Any,
    *,
    batch_rows: int,
) -> tuple[int, int]:
    last_signal_day = int((LAST_SIGNAL - dt.date(1970, 1, 1)).days)
    carry_permno: int | None = None
    carry_days = np.asarray([], dtype=np.int32)
    carry_availability = np.asarray([], dtype=np.int32)
    carry_returns = np.asarray([], dtype=np.float64)
    delivered = 0
    used = 0
    last_seen_permno: int | None = None

    def finish() -> None:
        nonlocal used
        if carry_permno is None:
            return
        targets = target_map.get(carry_permno)
        if targets is not None:
            frozen_stage8._process_security_history(
                carry_permno,
                carry_days,
                carry_availability,
                carry_returns,
                targets,
                state,
                factors,
            )
            used += len(carry_days)

    for batch in inputs.iter_return_batches(batch_rows=batch_rows):
        _check(
            isinstance(batch, pa.RecordBatch), "Return input did not yield RecordBatch"
        )
        _check(tuple(batch.schema.names) == RETURN_COLUMNS, "Return projection differs")
        _check(
            not unexpected_forbidden_columns(
                list(batch.schema.names), allowed_columns=list(RETURN_COLUMNS)
            ),
            "Return projection exposes a prohibited field",
        )
        delivered += batch.num_rows
        permnos = frozen_stage8._int_values(
            frozen_stage8._column(batch, "permno"), np.int64
        )
        days = frozen_stage8._date_days(frozen_stage8._column(batch, "calendar_date"))
        span_end = frozen_stage8._date_days(
            frozen_stage8._column(batch, "return_span_end_date")
        )
        availability = frozen_stage8._date_days(
            frozen_stage8._column(batch, "outcome_availability_date")
        )
        _check(
            np.all(days <= last_signal_day)
            and np.all(span_end <= last_signal_day)
            and np.all(availability <= last_signal_day),
            "Return input contains a post-validation-cutoff row",
        )
        covered = frozen_stage8._int_values(
            frozen_stage8._column(batch, "covered_trading_sessions"), np.int16
        )
        usable = pc.fill_null(
            frozen_stage8._column(batch, "return_history_usable"), False
        ).to_numpy(zero_copy_only=False)
        values = frozen_stage8._float_values(
            frozen_stage8._column(batch, "authoritative_total_return")
        )
        valid = (
            usable
            & (covered == 1)
            & (span_end == days)
            & (availability != frozen_stage8._MISSING_DAY)
            & np.isfinite(values)
            & (values >= -1.0)
        )
        permnos = permnos[valid]
        days = days[valid]
        availability = availability[valid]
        values = values[valid]
        if permnos.size == 0:
            continue
        _check(
            np.all(permnos[1:] >= permnos[:-1]), "Return batch is not sorted by PERMNO"
        )
        _check(
            last_seen_permno is None or int(permnos[0]) >= last_seen_permno,
            "Return scan order regressed",
        )
        last_seen_permno = int(permnos[-1])
        boundaries = np.flatnonzero(permnos[1:] != permnos[:-1]) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(permnos)]
        for start, end in zip(starts, ends, strict=True):
            group_permno = int(permnos[start])
            group_days = days[start:end]
            group_availability = availability[start:end]
            group_returns = values[start:end]
            if carry_permno is None:
                carry_permno = group_permno
                carry_days = group_days.copy()
                carry_availability = group_availability.copy()
                carry_returns = group_returns.copy()
            elif group_permno == carry_permno:
                carry_days = np.concatenate((carry_days, group_days))
                carry_availability = np.concatenate(
                    (carry_availability, group_availability)
                )
                carry_returns = np.concatenate((carry_returns, group_returns))
            else:
                finish()
                carry_permno = group_permno
                carry_days = group_days.copy()
                carry_availability = group_availability.copy()
                carry_returns = group_returns.copy()
    finish()
    return delivered, used


def materialize_validation_risk_cost(
    predictor: pa.Table,
    inputs: ValidationRiskHistoryInputs,
    contract: ValidationDataContract,
    *,
    batch_rows: int = 262_144,
) -> ValidationRiskCostMaterialization:
    """Run frozen risk/cost numerics under the exact 59-signal adapter."""

    _check(
        predictor.schema.equals(contract.raw_feature_schema, check_metadata=True),
        "Risk/cost predictor schema differs",
    )
    return _materialize_validation_risk_cost_spine(
        predictor.select(list(RISK_SPINE_COLUMNS)),
        inputs,
        contract,
        batch_rows=batch_rows,
    )


def _materialize_validation_risk_cost_spine(
    feature_spine: pa.Table,
    inputs: ValidationRiskHistoryInputs,
    contract: ValidationDataContract,
    *,
    batch_rows: int,
) -> ValidationRiskCostMaterialization:
    _check(
        tuple(feature_spine.schema.names) == RISK_SPINE_COLUMNS,
        "Risk/cost feature projection differs",
    )
    _validate_risk_inputs(inputs, contract)
    order = pc.sort_indices(
        feature_spine,
        sort_keys=[("signal_date", "ascending"), ("permno", "ascending")],
    )
    spine = feature_spine.take(order)
    signals = frozen_stage8._date_days(spine["signal_date"])
    permnos = frozen_stage8._int_values(spine["permno"], np.int64)
    _check(
        len(signals) > 0
        and not np.any((signals[1:] == signals[:-1]) & (permnos[1:] == permnos[:-1])),
        "Risk/cost spine keys are empty or duplicate",
    )
    unique_signals = np.unique(signals)
    expected_signal_days = np.asarray(
        [(value - dt.date(1970, 1, 1)).days for value in contract.scored_signal_dates],
        dtype=np.int32,
    )
    _check(
        np.array_equal(unique_signals, expected_signal_days),
        "Risk/cost signal scope differs",
    )
    long_mask = pc.fill_null(spine["long_eligible"], False).to_numpy(
        zero_copy_only=False
    )
    eligible_indices = np.flatnonzero(long_mask)
    _check(eligible_indices.size > 0, "Validation risk/cost has no long-eligible row")
    eligible_signals = signals[eligible_indices]
    eligible_permnos = permnos[eligible_indices]
    state = frozen_stage8._new_state(eligible_signals, eligible_permnos)
    factor_context = frozen_stage8._factor_context(inputs, unique_signals)
    targets = frozen_stage8._target_map(eligible_permnos)
    delivered, used = _scope_aware_return_scan(
        inputs,
        targets,
        state,
        factor_context,
        batch_rows=batch_rows,
    )

    eligible_spine = spine.take(pa.array(eligible_indices, type=pa.int64()))
    month_rows = frozen_stage8._finalize_cross_sections(
        state,
        sectors=eligible_spine["ff12_sector_category"].to_pylist(),
        short_eligible=pc.fill_null(eligible_spine["short_eligible"], False).to_numpy(
            zero_copy_only=False
        ),
        spreads=frozen_stage8._float_values(
            eligible_spine["median_estimated_full_spread_20d"]
        ),
        adv=frozen_stage8._float_values(eligible_spine["median_dollar_volume_20d"]),
        factors=factor_context,
    )
    audit = frozen_stage8._build_audit_table(spine, eligible_indices, state)
    processed = audit.filter(audit["long_eligible"])
    covariance = frozen_stage8._factor_covariance_table(unique_signals, factor_context)
    completed = frozen_stage8._complete_month_rows(signals, month_rows, factor_context)
    summary = pa.Table.from_pylist(
        completed,
        schema=contract.payload_specs["validation_risk_cost_month_summary"].schema,
    )
    expected = {
        "validation_risk_cost_audit": audit,
        "validation_risk_cost_inputs": processed,
        "validation_factor_covariance": covariance,
        "validation_risk_cost_month_summary": summary,
    }
    for role, table in expected.items():
        _check(
            table.schema.equals(
                contract.payload_specs[role].schema, check_metadata=True
            ),
            f"Frozen Stage 8 schema differs for {role}",
        )
    _check(audit.num_rows == feature_spine.num_rows, "Risk audit row count differs")
    _check(processed.num_rows == eligible_indices.size, "Risk input row count differs")
    _check(covariance.num_rows == 59 * 36, "Factor covariance is not 59 x 36")
    _check(summary.num_rows == 59, "Risk/cost summary is not 59 rows")
    quality: Mapping[str, int | bool] = MappingProxyType(
        {
            "audit_rows": audit.num_rows,
            "risk_input_rows": processed.num_rows,
            "factor_covariance_rows": covariance.num_rows,
            "month_summary_rows": summary.num_rows,
            "return_rows_delivered": delivered,
            "eligible_security_history_rows_processed": used,
            "post_cutoff_return_rows_used": 0,
            "post_cutoff_factor_rows_used": 0,
            "label_rows_read": 0,
            "outcome_rows_read": 0,
            "final_period_rows_read": 0,
            "frozen_stage8_numerics_reused": True,
            "feature_projection_columns": len(RISK_SPINE_COLUMNS),
            "feature_projection_bytes": feature_spine.nbytes,
        }
    )
    return ValidationRiskCostMaterialization(
        audit=audit,
        processed=processed,
        factor_covariance=covariance,
        month_summary=summary,
        quality=quality,
    )


def materialize_validation_risk_cost_from_parquet(
    predictor_path: Path,
    inputs: ValidationRiskHistoryInputs,
    contract: ValidationDataContract,
    *,
    batch_rows: int = 262_144,
) -> ValidationRiskCostMaterialization:
    """Read only the ten Stage 8 spine columns, never all 170 predictors."""

    path = Path(os.path.abspath(predictor_path.expanduser()))
    _check(path.is_file() and not path.is_symlink(), "Unsafe predictor Parquet path")
    parquet = pq.ParquetFile(path)
    _check(
        parquet.schema_arrow.equals(contract.raw_feature_schema, check_metadata=True),
        "Predictor Parquet schema differs before risk projection",
    )
    spine = pq.read_table(path, columns=list(RISK_SPINE_COLUMNS), use_threads=False)
    return _materialize_validation_risk_cost_spine(
        spine,
        inputs,
        contract,
        batch_rows=batch_rows,
    )


def write_validation_risk_cost_payloads(
    materialization: ValidationRiskCostMaterialization,
    destination: Path,
    contract: ValidationDataContract,
    *,
    row_group_size: int = 131_072,
) -> Path:
    """Append the four exact risk/cost roles to an eight-role feature root."""

    root = Path(os.path.abspath(destination.expanduser()))
    _check(root.is_dir() and not root.is_symlink(), "Unsafe risk/cost destination")
    feature_roles = {"validation_predictor_payload"} | {
        f"validation_feature_lineage_{family}"
        for family in contract.family_feature_names
    }
    _check(
        {path.name for path in root.iterdir()}
        == {contract.payload_specs[role].filename for role in feature_roles},
        "Risk/cost destination does not contain exactly the eight feature roles",
    )
    tables = {
        "validation_risk_cost_audit": materialization.audit,
        "validation_risk_cost_inputs": materialization.processed,
        "validation_factor_covariance": materialization.factor_covariance,
        "validation_risk_cost_month_summary": materialization.month_summary,
    }
    for role, table in tables.items():
        specification = contract.payload_specs[role]
        _check(
            table.schema.equals(specification.schema, check_metadata=True),
            f"Risk/cost schema differs: {role}",
        )
        partial = root / f".{specification.filename}.partial"
        pq.write_table(
            table,
            partial,
            compression="zstd",
            row_group_size=row_group_size,
            use_dictionary=True,
            write_statistics=True,
        )
        os.replace(partial, root / specification.filename)
    _check(
        {path.name for path in root.iterdir()}
        == {
            specification.filename for specification in contract.payload_specs.values()
        },
        "Completed candidate directory is not the exact twelve-role set",
    )
    return root


def write_validation_candidate_streaming(
    scored_batches: Iterable[Sequence[ValidationFeatureRow]],
    optionmetrics_context_rows: Sequence[OptionMetricsContextRow],
    risk_inputs: ValidationRiskHistoryInputs,
    destination: Path,
    contract: ValidationDataContract,
    *,
    maximum_feature_rows_per_chunk: int = 2_048,
    return_batch_rows: int = 262_144,
) -> tuple[StreamingFeatureWriteReceipt, ValidationRiskCostMaterialization]:
    """Bounded feature write followed by a ten-column risk/cost projection."""

    feature_receipt = write_validation_features_streaming(
        scored_batches,
        optionmetrics_context_rows,
        destination,
        contract,
        maximum_rows_per_chunk=maximum_feature_rows_per_chunk,
    )
    predictor_path = (
        feature_receipt.destination
        / contract.payload_specs["validation_predictor_payload"].filename
    )
    risk = materialize_validation_risk_cost_from_parquet(
        predictor_path,
        risk_inputs,
        contract,
        batch_rows=return_batch_rows,
    )
    write_validation_risk_cost_payloads(risk, feature_receipt.destination, contract)
    return feature_receipt, risk


def materialize_validation_candidate_tables(
    scored_rows: Sequence[ValidationFeatureRow],
    optionmetrics_context_rows: Sequence[OptionMetricsContextRow],
    risk_inputs: ValidationRiskHistoryInputs,
    contract: ValidationDataContract,
    *,
    batch_rows: int = 262_144,
) -> ValidationCandidateTables:
    """Build all twelve exact candidate roles without reading a raw path."""

    features = materialize_validation_features(
        scored_rows,
        optionmetrics_context_rows,
        contract,
    )
    risk = materialize_validation_risk_cost(
        features.predictor,
        risk_inputs,
        contract,
        batch_rows=batch_rows,
    )
    payloads: dict[str, pa.Table] = {
        "validation_predictor_payload": features.predictor,
        "validation_risk_cost_audit": risk.audit,
        "validation_risk_cost_inputs": risk.processed,
        "validation_factor_covariance": risk.factor_covariance,
        "validation_risk_cost_month_summary": risk.month_summary,
    }
    payloads.update(
        {
            f"validation_feature_lineage_{family}": table
            for family, table in features.lineage_by_family.items()
        }
    )
    _check(
        set(payloads) == set(contract.payload_specs),
        "Candidate payload role set differs",
    )
    return ValidationCandidateTables(
        payloads=MappingProxyType(payloads),
        feature_quality=features.quality,
        risk_cost_quality=risk.quality,
        context_receipt=features.context_receipt,
    )


def write_validation_candidate_tables(
    candidate: ValidationCandidateTables,
    destination: Path,
    contract: ValidationDataContract,
    *,
    row_group_size: int = 131_072,
) -> Path:
    """Write only the twelve contract roles into one closed candidate root."""

    root = Path(os.path.abspath(destination.expanduser()))
    if root.exists():
        _check(root.is_dir() and not root.is_symlink(), "Unsafe candidate destination")
        _check(not any(root.iterdir()), "Candidate destination is not empty")
    else:
        root.mkdir(parents=False, exist_ok=False)
    _check(
        set(candidate.payloads) == set(contract.payload_specs),
        "Candidate role set differs",
    )
    for role, specification in contract.payload_specs.items():
        table = candidate.payloads[role]
        _check(
            table.schema.equals(specification.schema, check_metadata=True),
            f"Candidate table schema differs: {role}",
        )
        pq.write_table(
            table,
            root / specification.filename,
            compression="zstd",
            row_group_size=row_group_size,
            use_dictionary=True,
            write_statistics=True,
        )
    return root


__all__ = [
    "FACTOR_COLUMNS",
    "OPTIONMETRICS_FAMILY",
    "OptionMetricsContextReceipt",
    "OptionMetricsContextRow",
    "RETURN_COLUMNS",
    "RISK_SPINE_COLUMNS",
    "SEGMENT_FEATURES",
    "ValidationCandidateTables",
    "ValidationDataCandidateBuilderError",
    "ValidationFeatureMaterialization",
    "ValidationFeatureRow",
    "ValidationRiskCostMaterialization",
    "ValidationRiskHistoryInputs",
    "StreamingFeatureWriteReceipt",
    "materialize_validation_candidate_tables",
    "materialize_validation_features",
    "materialize_validation_risk_cost",
    "materialize_validation_risk_cost_from_parquet",
    "write_validation_candidate_streaming",
    "write_validation_candidate_tables",
    "write_validation_features_streaming",
    "write_validation_risk_cost_payloads",
]
