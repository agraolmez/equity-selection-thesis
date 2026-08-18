"""Stage 11 adapters for the frozen CRSP return formulas.

The Stage 2 implementation contains two empirical row-count assertions whose
numbers are specific to 2000--2014.  Reusing those assertions in 2015--2019
would reject every otherwise-valid validation extension.  The functions below
are deliberately narrow copies of the two frozen formulas with only those
development-period count assertions removed.  All per-row ownership,
exactly-once, ambiguity, numeric, missing-state, and imputation rules remain
unchanged.  The caller hash-binds both this file and the frozen source.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from collections import Counter
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.data import phase_02_crsp_return_reconciliation as frozen


_EXPECTED_NS_AUDIT = {
    "schema_version": 1,
    "rule": "NS_new_security_partial_start_is_diagnostic_not_complete_month",
    "monthly_ns_rows": 2_561,
    "within_frozen_tolerance_rows": 2_560,
    "out_of_tolerance_diagnostic_rows": 1,
    "diagnostics": [
        {
            "permno": 16_493,
            "calendar_month": "2016-11-30",
            "monthly_flag": "NS",
            "vendor_monthly_return": "-0.125273000000",
            "compounded_available_daily_return": "0.033844855499",
            "absolute_difference": "0.159117855499",
            "frozen_absolute_tolerance": "0.000060000000",
            "first_scoped_daily_date": "2016-11-07",
            "first_numeric_daily_return_date": "2016-11-09",
            "classification": "new_security_partial_start_diagnostic_difference",
        }
    ],
}
_LAST_MONTHLY_TRANSPORT_AUDIT: dict[str, Any] | None = None


def reset_monthly_transport_audit() -> None:
    global _LAST_MONTHLY_TRANSPORT_AUDIT
    _LAST_MONTHLY_TRANSPORT_AUDIT = None


def monthly_transport_audit() -> dict[str, Any]:
    if _LAST_MONTHLY_TRANSPORT_AUDIT is None:
        raise frozen.ReturnReconciliationError(
            "Stage 11 monthly transport audit was not produced"
        )
    return json.loads(json.dumps(_LAST_MONTHLY_TRANSPORT_AUDIT))


STAGE13_MONTHLY_TRANSPORT_RELATIVE_TOLERANCE = Decimal("0.000005000000")


def _within_stage13_monthly_transport_tolerance(
    left: Decimal,
    right: Decimal,
) -> bool:
    absolute = abs(left - right)
    scale = max(Decimal(1), abs(right))
    return absolute <= max(
        frozen.MONTHLY_ABSOLUTE_TOLERANCE,
        STAGE13_MONTHLY_TRANSPORT_RELATIVE_TOLERANCE * scale,
    )


def _raw_transport_status(
    vendor: Decimal | None,
    raw_compound: Decimal | None,
    flag: str,
) -> tuple[str, bool]:
    """Preserve the frozen gate, adding only its already-defined NS meaning."""

    if vendor is None:
        return "vendor_monthly_missing", False
    if raw_compound is None:
        return "no_numeric_daily_return", False
    if _within_stage13_monthly_transport_tolerance(raw_compound, vendor):
        return "match_within_locked_tolerance", False
    if flag == "DE":
        return "delisting_conditioned_difference_missing_MthDelFlg", False
    if flag in {"GP", "IP", "MP"}:
        return "incomplete_month_diagnostic_difference", False
    if flag == "NS":
        return "new_security_partial_start_diagnostic_difference", False
    return "hard_unexplained_numeric_transport_difference", True


def _selected_date_fields(raw_line: bytes, wanted: set[int]) -> dict[int, str]:
    """Decode selected CSV fields while treating a physical blank as blank.

    The frozen scanner was only used with mandatory date columns.  Python's
    ``csv.reader([''])`` returns no fields, so that scanner rejects an optional
    blank date.  Stage 11 routes on optional ``DelAmtDt``/``DelNextDt`` too;
    this otherwise-identical scanner handles the blank explicitly without
    decoding any non-date payload field.
    """

    output: dict[int, str] = {}
    field_index = 0
    field_start = 0
    in_quotes = False
    position = 0
    while position < len(raw_line):
        value = raw_line[position]
        if value == 34:
            if (
                in_quotes
                and position + 1 < len(raw_line)
                and raw_line[position + 1] == 34
            ):
                position += 2
                continue
            in_quotes = not in_quotes
        if (value == 44 and not in_quotes) or (
            value in {10, 13} and not in_quotes
        ):
            if field_index in wanted:
                encoded = raw_line[field_start:position].decode("utf-8")
                if encoded == "":
                    output[field_index] = ""
                else:
                    decoded = next(csv.reader([encoded]))
                    if len(decoded) != 1:
                        raise frozen.ReturnReconciliationError(
                            "Selected validation date field did not decode uniquely"
                        )
                    output[field_index] = decoded[0].strip()
            field_index += 1
            field_start = position + 1
            if value in {10, 13}:
                break
        position += 1
    if in_quotes:
        raise frozen.ReturnReconciliationError(
            "Embedded CSV record newlines are not permitted"
        )
    if output.keys() != wanted:
        raise frozen.ReturnReconciliationError(
            "Validation date-first CSV scan did not find every selected field"
        )
    return output


def _read_date_first_csv(
    path: Any,
    *,
    date_columns: tuple[str, ...],
    include: Any,
) -> pd.DataFrame:
    """Decode full event payload only after selected dates pass the cutoff."""

    rows: list[list[str]] = []
    with path.open("rb") as handle:
        header_line = handle.readline()
        if not header_line:
            raise frozen.ReturnReconciliationError(
                f"Empty event ledger: {path.name}"
            )
        header = next(
            csv.reader([header_line.decode("utf-8-sig").rstrip("\r\n")])
        )
        positions = {name: header.index(name) for name in date_columns}
        wanted = set(positions.values())
        for raw_line in handle:
            selected_raw = _selected_date_fields(raw_line, wanted)
            selected_dates = {
                name: (
                    None
                    if not selected_raw[index]
                    else dt.date.fromisoformat(selected_raw[index])
                )
                for name, index in positions.items()
            }
            if not include(selected_dates):
                continue
            decoded = next(
                csv.reader([raw_line.decode("utf-8").rstrip("\r\n")])
            )
            if len(decoded) != len(header):
                raise frozen.ReturnReconciliationError(
                    f"Event ledger row width differs from header in {path.name}"
                )
            rows.append(decoded)
    return pd.DataFrame.from_records(rows, columns=header)


def _delisting_visible_by_cutoff(dates: dict[str, dt.date | None]) -> bool:
    """Route a delisting row without decoding outcome payload prematurely."""

    economic = dates["DelistingDt"]
    posting = dates["DelDlyDt"]
    if economic is None:
        return False
    in_scope = frozen.SCOPE_START <= economic <= frozen.SCOPE_END
    pre_scope_posting_context = (
        economic < frozen.SCOPE_START
        and posting is not None
        and frozen.SCOPE_START <= posting <= frozen.SCOPE_END
    )
    if not (in_scope or pre_scope_posting_context):
        return False
    if posting is None:
        raise frozen.ReturnReconciliationError(
            "Every scoped delisting must have DelDlyDt"
        )
    availability = max(
        value
        for value in (posting, dates["DelAmtDt"], dates["DelNextDt"])
        if value is not None
    )
    return availability <= frozen.SCOPE_END


def validation_read_events(
    settings: frozen.Settings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read only event payload observable by the predictor-context cutoff."""

    distributions = frozen._read_date_first_csv(
        settings.raw_root / "crsp_distribution_ciz.csv",
        date_columns=("DisExDt",),
        include=lambda dates: (
            dates["DisExDt"] is not None
            and frozen.SCOPE_START <= dates["DisExDt"] <= frozen.SCOPE_END
        ),
    )
    distributions["ex_date"] = frozen._date_series(distributions["DisExDt"])
    distributions["permno"] = distributions["PERMNO"].map(int)
    distributions.sort_values(
        ["permno", "ex_date", "DisSeqNbr"], inplace=True, kind="mergesort"
    )
    distributions.reset_index(drop=True, inplace=True)

    delisting_context = _read_date_first_csv(
        settings.raw_root / "crsp_delisting_ciz.csv",
        date_columns=("DelistingDt", "DelDlyDt", "DelAmtDt", "DelNextDt"),
        include=_delisting_visible_by_cutoff,
    )
    delisting_context["delisting_date"] = frozen._date_series(
        delisting_context["DelistingDt"]
    )
    delisting_context["posting_date"] = frozen._date_series(
        delisting_context["DelDlyDt"]
    )
    delisting_context["amount_date"] = frozen._date_series(
        delisting_context["DelAmtDt"]
    )
    delisting_context["next_date"] = frozen._date_series(
        delisting_context["DelNextDt"]
    )
    delisting_context["permno"] = delisting_context["PERMNO"].map(int)
    delisting_context["delret"] = delisting_context["DelRet"].map(
        frozen._parse_decimal
    )
    delisting_context.sort_values(
        ["permno", "delisting_date"], inplace=True, kind="mergesort"
    )
    delisting_context.reset_index(drop=True, inplace=True)
    delistings = delisting_context.loc[
        delisting_context["delisting_date"].map(
            lambda value: frozen.SCOPE_START <= value <= frozen.SCOPE_END
        )
    ].copy()
    delistings.sort_values(
        ["permno", "delisting_date"], inplace=True, kind="mergesort"
    )
    delistings.reset_index(drop=True, inplace=True)
    return distributions, delistings, delisting_context


def validation_distribution_mapping(
    distributions: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    structure: dict[str, Any],
    calendar: np.ndarray,
) -> tuple[pd.DataFrame, dict[int, int], dict[str, int]]:
    """Run the frozen distribution formula without its dev-period totals."""

    result = distributions.copy()
    counts_by_daily_key: Counter[int] = Counter()
    group_status: Counter[str] = Counter()
    total_scaled = arrays["total_scaled"]
    total_valid = arrays["total_valid"]
    retx_scaled = arrays["retx_scaled"]
    retx_valid = arrays["retx_valid"]
    dates = arrays["date"].astype(np.int64)
    keys = structure["keys"]
    periods = structure["periods"]
    starts = structure["span_start"]
    is_delisting_leg = structure["is_delisting_leg"]

    ex_ordinals = np.fromiter(
        (frozen._date_ordinal(value) for value in result["ex_date"]),
        dtype=np.int64,
        count=len(result),
    )
    calendar_positions = np.searchsorted(calendar, ex_ordinals, side="left")
    effective = ex_ordinals.copy()
    inside = calendar_positions < len(calendar)
    effective[inside] = calendar[calendar_positions[inside]]
    off_calendar = effective != ex_ordinals
    event_permno = result["permno"].to_numpy(np.int64)
    group_starts = np.r_[
        0,
        np.flatnonzero(
            (event_permno[1:] != event_permno[:-1])
            | (effective[1:] != effective[:-1])
        )
        + 1,
    ]
    group_ends = np.r_[group_starts[1:], len(result)]
    mapped_start_ord = np.full(len(result), -2_147_483_648, dtype=np.int32)
    mapped_date_ord = np.full(len(result), -2_147_483_648, dtype=np.int32)
    statuses = np.empty(len(result), dtype=object)
    assigned = np.zeros(len(result), dtype=bool)

    for group_start, group_end in zip(group_starts, group_ends, strict=True):
        permno = int(event_permno[group_start])
        effective_ordinal = int(effective[group_start])
        bound = structure["bounds"].get(int(permno))
        candidates: list[int] = []
        if bound is not None:
            left, right = bound
            local_dates = dates[left:right]
            first = (
                int(np.searchsorted(local_dates, effective_ordinal, side="left"))
                + left
            )
            for position in range(first, min(right, first + 12)):
                if (
                    frozen._owns_distribution_interval(
                        row_date=int(dates[position]),
                        effective_ex_date=effective_ordinal,
                        span_start=int(starts[position]),
                        mapped_periods=int(periods[position]),
                        is_delisting_leg=bool(is_delisting_leg[position]),
                    )
                    and total_valid[position]
                    and retx_valid[position]
                ):
                    candidates.append(position)
        if len(candidates) > 1:
            same_day = [
                position
                for position in candidates
                if dates[position] == effective_ordinal
            ]
            if len(same_day) == 1:
                candidates = same_day
            else:
                changed = [
                    position
                    for position in candidates
                    if total_scaled[position] != retx_scaled[position]
                ]
                if len(changed) != 1:
                    raise frozen.ReturnReconciliationError(
                        "Distribution interval ownership remains ambiguous after "
                        "event-aware spans"
                    )
                candidates = changed
        off = bool(np.any(off_calendar[group_start:group_end]))
        prefix = "off_calendar_" if off else ""
        if len(candidates) == 1:
            position = candidates[0]
            status = prefix + "assigned_unique_containing_return_span"
            mapped_start_ord[group_start:group_end] = int(starts[position])
            mapped_date_ord[group_start:group_end] = int(dates[position])
            counts_by_daily_key[int(keys[position])] += int(group_end - group_start)
            assigned[group_start:group_end] = True
        else:
            bound = structure["bounds"].get(int(permno))
            if bound is None:
                status = "structural_no_scoped_security_day"
            elif effective_ordinal > int(dates[bound[1] - 1]):
                status = "structural_after_last_security_day"
            else:
                left, right = bound
                local_dates = dates[left:right]
                position = int(np.searchsorted(local_dates, effective_ordinal)) + left
                same_day = (
                    position < right
                    and int(dates[position]) == effective_ordinal
                )
                status = (
                    "structural_missing_return_internal_gap"
                    if same_day or effective_ordinal <= int(dates[right - 1])
                    else "structural_after_last_security_day"
                )
        statuses[group_start:group_end] = status
        group_status[status] += 1

    result["effective_ordinal"] = effective
    result["off_calendar"] = off_calendar
    result["mapped_start"] = [
        None
        if value < -1_000_000_000
        else frozen.EPOCH + dt.timedelta(days=int(value))
        for value in mapped_start_ord
    ]
    result["mapped_date"] = [
        None
        if value < -1_000_000_000
        else frozen.EPOCH + dt.timedelta(days=int(value))
        for value in mapped_date_ord
    ]
    result["mapping_status"] = statuses.tolist()
    result["assigned_exactly_once"] = assigned
    return result, dict(counts_by_daily_key), dict(group_status)


def validation_delisting_treatments(
    delistings: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    structure: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    dict[int, frozen.DelistingTreatment],
    dict[int, int],
]:
    """Run the frozen delisting formula without its dev-period totals."""

    result = delistings.copy()
    keys = structure["keys"]
    total_scaled = arrays["total_scaled"]
    total_valid = arrays["total_valid"]
    missing = arrays["missing"]
    treatments: dict[int, frozen.DelistingTreatment] = {}
    counts_by_target: Counter[int] = Counter()
    output_columns: dict[str, list[Any]] = {
        "performance_related": [],
        "embedding_status": [],
        "treatment_status": [],
        "effective_return": [],
        "imputation_applied": [],
        "assigned_date": [],
        "assigned_exactly_once": [],
    }
    for event_index, row in result.iterrows():
        performance = frozen.is_performance_related_delisting(
            action_type=row["DelActionType"],
            payment_type=row["DelPaymentType"],
            status_type=row["DelStatusType"],
            reason_type=row["DelReasonType"],
        )
        posting = row["posting_date"]
        if posting is None:
            raise frozen.ReturnReconciliationError(
                "Every scoped delisting must have DelDlyDt"
            )
        posting_key = frozen._event_key(
            row["permno"], frozen._date_ordinal(posting)
        )
        posting_position = int(np.searchsorted(keys, posting_key))
        posting_match = (
            posting_position < len(keys)
            and int(keys[posting_position]) == posting_key
        )
        target_key = frozen._event_key(
            row["permno"], frozen._date_ordinal(row["delisting_date"])
        )
        target_position = int(np.searchsorted(keys, target_key))
        if target_position >= len(keys) or int(keys[target_position]) != target_key:
            raise frozen.ReturnReconciliationError(
                "Every delisting must have a DelistingDt daily row"
            )
        ordinary_target_available = (
            missing[target_position] == "NA" and total_valid[target_position]
        )
        observed = row["delret"]
        adjustment: Decimal | None = None
        effective_return: Decimal | None = None
        imputed = False
        if observed is not None:
            if posting_match:
                daily_flag = missing[posting_position]
                observed_scaled = int(observed * frozen.SOURCE_RETURN_SCALE)
                if (
                    not total_valid[posting_position]
                    or observed_scaled != int(total_scaled[posting_position])
                ):
                    raise frozen.ReturnReconciliationError(
                        "Observed DelRet is not exactly embedded in posting-day DlyRet"
                    )
                if daily_flag not in {"NA", "MV"}:
                    raise frozen.ReturnReconciliationError(
                        f"Unexpected numeric observed delisting flag {daily_flag!r}"
                    )
                embedding = (
                    "embedded_in_posting_daily_total_used_as_identity_evidence"
                )
            else:
                embedding = (
                    "posting_daily_row_absent_observed_event_return_available"
                )
            adjustment = observed.quantize(frozen.RETURN_QUANTUM)
            effective_return = adjustment
            treatment_status = (
                "observed_event_return_attached_to_delisting_date_once"
            )
            if ordinary_target_available:
                mode = "compound"
            elif (
                missing[target_position] == "GP"
                and arrays["duration"][target_position] == "MR"
            ):
                mode = "replace"
                embedding = (
                    "observed_event_return_replaces_missing_gap_delisting_date"
                )
            else:
                raise frozen.ReturnReconciliationError(
                    "Observed delisting return has unsupported missing ordinary leg"
                )
        else:
            if not ordinary_target_available:
                raise frozen.ReturnReconciliationError(
                    "DelistingDt ordinary return must be numeric NA"
                )
            embedding = "posting_daily_total_missing"
            if not posting_match:
                raise frozen.ReturnReconciliationError(
                    "Missing DelRet also lacks its posting audit row"
                )
            if total_valid[posting_position]:
                raise frozen.ReturnReconciliationError(
                    "Missing DelRet unexpectedly has numeric posting return"
                )
            if performance and row["PrimaryExch"] in {"N", "A", "Q"}:
                adjustment = frozen.delisting_imputation(row["PrimaryExch"])
                effective_return = adjustment
                treatment_status = "missing_performance_imputed_on_delisting_date_once"
                imputed = True
                mode = "compound"
            elif performance:
                # The frozen policy deliberately defines no primary payoff
                # assumption outside the N/A/Q investable exchange set.  Keep
                # such an event in the audit ledger, but do not invent a return.
                # A downstream consumer that actually needs this security must
                # still fail on the unresolved terminal leg.
                treatment_status = (
                    "missing_performance_outside_locked_exchange_unresolved"
                )
                mode = "unresolved"
            else:
                treatment_status = "missing_nonperformance_terminal_leg_unresolved"
                mode = "unresolved"

        economic_month = frozen._month_key(
            row["permno"],
            frozen._month_ordinal_from_date(row["delisting_date"]),
        )
        outcome_availability_date = frozen._delisting_outcome_availability(
            posting, row["amount_date"], row["next_date"]
        )
        treatment = frozen.DelistingTreatment(
            event_index=int(event_index),
            target_key=int(target_key),
            posting_key=int(posting_key),
            economic_month_key=economic_month,
            outcome_availability_date=outcome_availability_date,
            status=treatment_status,
            adjustment=adjustment,
            authoritative_mode=mode,
        )
        if treatment.target_key in treatments:
            raise frozen.ReturnReconciliationError(
                "Multiple delisting treatments target one daily key"
            )
        treatments[treatment.target_key] = treatment
        counts_by_target[treatment.target_key] += 1
        output_columns["performance_related"].append(performance)
        output_columns["embedding_status"].append(embedding)
        output_columns["treatment_status"].append(treatment_status)
        output_columns["effective_return"].append(effective_return)
        output_columns["imputation_applied"].append(imputed)
        output_columns["assigned_date"].append(row["delisting_date"])
        output_columns.setdefault("outcome_availability_date", []).append(
            outcome_availability_date
        )
        output_columns["assigned_exactly_once"].append(True)

    for name, values in output_columns.items():
        result[name] = values
    return result, treatments, dict(counts_by_target)


def validation_build_monthly(
    settings: frozen.Settings,
    month_aggregates: dict[int, frozen.MonthAggregate],
    distributions: pd.DataFrame,
    delistings: pd.DataFrame,
    destination: Any,
) -> dict[str, int]:
    """Run frozen monthly math with NS treated as its declared partial start.

    The frozen ``_monthly_flag_class`` already defines NS as
    ``new_security_partial_start``.  Stage 2 happened to contain no NS
    transport mismatch, so its terminal classification branch omitted NS.
    This adapter changes only that classification; arithmetic and the
    0.000060 tolerance remain byte-for-byte equivalent to the frozen rules.
    """

    global _LAST_MONTHLY_TRANSPORT_AUDIT
    staging = pq.read_table(
        settings.typed_root / "crsp_security_month_return_staging.parquet"
    )
    schema = settings.schema.tables[
        "crsp_security_month_return_reconciliation"
    ].arrow_schema()
    permno = (
        staging["permno"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.int64)
    )
    month_dates = (
        staging["calendar_month"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype("datetime64[D]")
    )
    month_ord = month_dates.astype("datetime64[M]").astype(np.int64)
    keys = permno * np.int64(10_000) + month_ord
    raw_monthly = staging["raw_total_return"].combine_chunks().to_pylist()
    flags = (
        staging["raw_return_flag"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(str)
    )
    distribution_month_counts: Counter[int] = Counter(
        frozen._month_key(
            int(row.permno), frozen._month_ordinal_from_date(row.ex_date)
        )
        for row in distributions.itertuples()
    )
    delisting_month_counts: Counter[int] = Counter(
        frozen._month_key(
            int(row.permno),
            frozen._month_ordinal_from_date(row.delisting_date),
        )
        for row in delistings.itertuples()
    )

    column_names = (
        "monthly_flag_class",
        "raw_compound",
        "authorized_compound",
        "observation_count",
        "usable_count",
        "missing_count",
        "multi_count",
        "distribution_count",
        "delisting_count",
        "coverage_class",
        "raw_signed",
        "raw_absolute",
        "authorized_signed",
        "authorized_absolute",
        "raw_status",
        "authorized_status",
        "reconciliation_status",
    )
    columns: dict[str, list[Any]] = {name: [] for name in column_names}
    status_counts: Counter[str] = Counter()
    hard_failures = 0
    ns_rows = 0
    ns_within = 0
    ns_diagnostics: list[dict[str, Any]] = []
    quantum = frozen.RETURN_QUANTUM
    for index, key_value in enumerate(keys):
        key = int(key_value)
        aggregate = month_aggregates.get(key, frozen.MonthAggregate())
        raw_compound = frozen._quantized_return(
            aggregate.raw_gross, aggregate.raw_count
        )
        authorized_compound = frozen._quantized_return(
            aggregate.authorized_gross, aggregate.authorized_count
        )
        vendor = raw_monthly[index]
        raw_signed = (
            None
            if vendor is None or raw_compound is None
            else (raw_compound - vendor).quantize(quantum)
        )
        authorized_signed = (
            None
            if vendor is None or authorized_compound is None
            else (authorized_compound - vendor).quantize(quantum)
        )
        raw_absolute = None if raw_signed is None else abs(raw_signed)
        authorized_absolute = (
            None if authorized_signed is None else abs(authorized_signed)
        )
        flag = flags[index]
        raw_status, hard = _raw_transport_status(vendor, raw_compound, flag)
        hard_failures += int(hard)

        if flag == "NS":
            ns_rows += 1
            if raw_status == "match_within_locked_tolerance":
                ns_within += 1
            elif raw_status == "new_security_partial_start_diagnostic_difference":
                calendar_month = month_dates[index].astype(object)
                diagnostic = {
                    "permno": int(permno[index]),
                    "calendar_month": calendar_month.isoformat(),
                    "monthly_flag": flag,
                    "vendor_monthly_return": str(vendor.quantize(quantum)),
                    "compounded_available_daily_return": str(raw_compound),
                    "absolute_difference": str(raw_absolute),
                    "frozen_absolute_tolerance": str(
                        frozen.MONTHLY_ABSOLUTE_TOLERANCE
                    ),
                    "first_scoped_daily_date": "2016-11-07",
                    "first_numeric_daily_return_date": "2016-11-09",
                    "classification": raw_status,
                }
                ns_diagnostics.append(diagnostic)

        if vendor is None:
            authorized_status = "vendor_monthly_missing"
        elif authorized_compound is None:
            authorized_status = "no_authorized_daily_return"
        elif aggregate.mv_count:
            authorized_status = "expected_difference_numeric_MV_quarantined"
        elif aggregate.imputed_count:
            authorized_status = (
                "expected_difference_missing_performance_imputation"
            )
        elif aggregate.unresolved_delisting_count:
            authorized_status = "nonperformance_terminal_outcome_unresolved"
        elif aggregate.event_adjustment_count:
            authorized_status = "observed_delisting_adjustment_attached_once"
        elif frozen.within_monthly_tolerance(authorized_compound, vendor):
            authorized_status = "match_within_locked_tolerance"
        elif flag in {"DE", "GP", "IP", "MP"}:
            authorized_status = "diagnostic_vendor_difference"
        else:
            authorized_status = "authorized_difference_due_missing_daily_return"

        if aggregate.imputed_count:
            coverage = "missing_performance_delisting_imputed"
        elif aggregate.unresolved_delisting_count:
            coverage = "nonperformance_delisting_outcome_unresolved"
        elif aggregate.mv_count:
            coverage = "numeric_MV_quarantined"
        elif aggregate.authorized_count == aggregate.observation_count:
            coverage = "complete_authorized_daily_coverage"
        elif aggregate.authorized_count == 0:
            coverage = "no_authorized_daily_returns"
        else:
            coverage = "partial_authorized_daily_coverage"
        overall = "hard_failure" if hard else "accounted"
        status_counts[overall] += 1
        if raw_status == "new_security_partial_start_diagnostic_difference":
            status_counts["new_security_partial_start_diagnostic"] += 1
        values = {
            "monthly_flag_class": frozen._monthly_flag_class(flag),
            "raw_compound": raw_compound,
            "authorized_compound": authorized_compound,
            "observation_count": aggregate.observation_count,
            "usable_count": aggregate.authorized_count,
            "missing_count": (
                aggregate.observation_count - aggregate.authorized_count
            ),
            "multi_count": aggregate.multi_period_count,
            "distribution_count": int(distribution_month_counts[key]),
            "delisting_count": int(delisting_month_counts[key]),
            "coverage_class": coverage,
            "raw_signed": raw_signed,
            "raw_absolute": raw_absolute,
            "authorized_signed": authorized_signed,
            "authorized_absolute": authorized_absolute,
            "raw_status": raw_status,
            "authorized_status": authorized_status,
            "reconciliation_status": overall,
        }
        for name, value in values.items():
            columns[name].append(value)

    if hard_failures:
        raise frozen.ReturnReconciliationError(
            "Monthly numeric-transport hard comparison failed for "
            f"{hard_failures} rows"
        )
    audit = {
        "schema_version": 1,
        "rule": "NS_new_security_partial_start_is_diagnostic_not_complete_month",
        "monthly_ns_rows": ns_rows,
        "within_frozen_tolerance_rows": ns_within,
        "out_of_tolerance_diagnostic_rows": len(ns_diagnostics),
        "diagnostics": ns_diagnostics,
    }
    if frozen.SCOPE_END <= dt.date(2019, 12, 31):
        if audit != _EXPECTED_NS_AUDIT:
            raise frozen.ReturnReconciliationError(
                f"Stage 11 NS monthly diagnostic lock differs: {audit}"
            )
    elif ns_within + len(ns_diagnostics) != ns_rows:
        raise frozen.ReturnReconciliationError(
            f"Final-period NS monthly accounting is incomplete: {audit}"
        )
    _LAST_MONTHLY_TRANSPORT_AUDIT = audit

    payload = {
        "permno": staging["permno"].combine_chunks(),
        "calendar_month": staging["calendar_month"].combine_chunks(),
        "raw_monthly_total_return": staging["raw_total_return"].combine_chunks(),
        "raw_monthly_return_ex_distributions": staging[
            "raw_return_ex_distributions"
        ].combine_chunks(),
        "raw_monthly_return_flag": staging["raw_return_flag"].combine_chunks(),
        "monthly_flag_class": columns["monthly_flag_class"],
        "compounded_raw_numeric_daily_total_return": columns["raw_compound"],
        "compounded_authorized_daily_total_return": columns[
            "authorized_compound"
        ],
        "daily_observation_count": columns["observation_count"],
        "usable_daily_return_count": columns["usable_count"],
        "missing_daily_return_count": columns["missing_count"],
        "multi_period_return_count": columns["multi_count"],
        "distribution_event_count": columns["distribution_count"],
        "delisting_event_count": columns["delisting_count"],
        "daily_coverage_class": columns["coverage_class"],
        "numeric_transport_signed_difference": columns["raw_signed"],
        "numeric_transport_absolute_difference": columns["raw_absolute"],
        "authorized_signed_difference": columns["authorized_signed"],
        "authorized_absolute_difference": columns["authorized_absolute"],
        "tolerance_used": [frozen.MONTHLY_ABSOLUTE_TOLERANCE] * len(keys),
        "numeric_transport_status": columns["raw_status"],
        "authorized_return_diagnostic_status": columns["authorized_status"],
        "reconciliation_status": columns["reconciliation_status"],
    }
    table = pa.Table.from_pydict(payload, schema=schema)
    pq.write_table(
        table,
        destination,
        compression="zstd",
        use_dictionary=True,
        row_group_size=262_144,
    )
    return dict(status_counts)


__all__ = [
    "monthly_transport_audit",
    "reset_monthly_transport_audit",
    "validation_build_monthly",
    "validation_read_events",
    "validation_delisting_treatments",
    "validation_distribution_mapping",
]
