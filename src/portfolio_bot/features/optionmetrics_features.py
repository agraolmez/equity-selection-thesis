"""Pure point-in-time OptionMetrics feature formulas for roadmap Stage 6."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from portfolio_bot.features.feature_lineage import (
    FeatureFamilyResult,
    FeatureLineage,
)


FEATURE_NAMES = (
    "option_call_volume_21d",
    "option_put_volume_21d",
    "option_combined_volume_21d",
    "option_call_volume_63d",
    "option_put_volume_63d",
    "option_combined_volume_63d",
    "option_call_open_interest_eom",
    "option_put_open_interest_eom",
    "option_combined_open_interest_eom",
    "log_option_put_call_volume_ratio_21d",
    "log_option_put_call_open_interest_ratio_eom",
    "option_call_minus_put_volume_share_21d",
    "option_total_volume_change_1m",
    "option_total_volume_change_3m",
    "option_total_open_interest_change_1m",
    "option_total_open_interest_change_3m",
    "option_volume_to_underlying_volume_21d",
    "optionmetrics_covered_indicator",
    "optionmetrics_coverage_age_months",
    "option_historical_volatility_30d",
    "option_historical_volatility_60d",
    "option_historical_volatility_90d",
    "option_historical_volatility_180d",
    "option_historical_volatility_360d",
    "option_historical_volatility_30d_minus_90d",
    "option_historical_volatility_60d_minus_180d",
)

VENDOR_HORIZONS = {30: 30, 60: 60, 90: 91, 180: 182, 360: 365}


class OptionMetricsFeatureError(RuntimeError):
    """Raised when an allegedly authoritative point-in-time grain is ambiguous."""


@dataclass(frozen=True)
class OptionMetricsResult:
    values: Mapping[str, float | int | None]
    observation_dates: Mapping[str, dt.date | None]
    public_availability_dates: Mapping[str, dt.date | None]
    stale_after_dates: Mapping[str, dt.date | None]
    statuses: Mapping[str, str]

    def as_family_result(self) -> FeatureFamilyResult:
        lineage = {
            name: FeatureLineage(
                self.observation_dates[name],
                self.public_availability_dates[name],
                self.stale_after_dates[name],
                self.statuses[name],
            )
            for name in self.values
        }
        return FeatureFamilyResult(values=self.values, lineage=lineage)


def _number(value: Any, *, nonnegative: bool = True) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        return None
    return result


def _completed_months(start: dt.date, end: dt.date) -> int:
    return max(0, (end.year - start.year) * 12 + end.month - start.month)


def _latest_date(values: Iterable[dt.date | None]) -> dt.date | None:
    dates = [value for value in values if isinstance(value, dt.date)]
    return max(dates) if dates else None


def _empty(status: str) -> OptionMetricsResult:
    return OptionMetricsResult(
        values={name: None for name in FEATURE_NAMES},
        observation_dates={name: None for name in FEATURE_NAMES},
        public_availability_dates={name: None for name in FEATURE_NAMES},
        stale_after_dates={name: None for name in FEATURE_NAMES},
        statuses={name: status for name in FEATURE_NAMES},
    )


def _eligible_flow_index(
    rows: Iterable[Mapping[str, Any]],
    secid: int,
    signal_date: dt.date,
    link_interval_start: dt.date,
) -> dict[tuple[dt.date, str], Mapping[str, Any]]:
    result: dict[tuple[dt.date, str], Mapping[str, Any]] = {}
    for row in rows:
        if int(row.get("secid", -1)) != secid:
            continue
        observation = row.get("observation_date")
        availability = row.get("public_availability_date")
        role = str(row.get("row_role", "")).strip().upper()
        if (
            role not in {"CALL", "PUT"}
            or row.get("authoritative_flow_component") is not True
            or not isinstance(observation, dt.date)
            or not isinstance(availability, dt.date)
            or observation < link_interval_start
            or observation > signal_date
            or availability > signal_date
        ):
            continue
        key = (observation, role)
        if key in result:
            raise OptionMetricsFeatureError(f"Duplicate authoritative option-flow key: {key}")
        result[key] = row
    return result


def _valid_field(row: Mapping[str, Any], field: str) -> float | None:
    value = _number(row.get(field))
    if value is None:
        return None
    if value == 0.0 and row.get(f"genuine_zero_{field}") is not True:
        return None
    return value


def _pair(
    index: Mapping[tuple[dt.date, str], Mapping[str, Any]],
    day: dt.date,
    field: str,
) -> tuple[float, float, tuple[Mapping[str, Any], Mapping[str, Any]]] | None:
    call_row = index.get((day, "CALL"))
    put_row = index.get((day, "PUT"))
    if call_row is None or put_row is None:
        return None
    call = _valid_field(call_row, field)
    put = _valid_field(put_row, field)
    if call is None or put is None:
        return None
    return call, put, (call_row, put_row)


def _lineage(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dt.date | None, dt.date | None]:
    materialized = list(rows)
    return (
        _latest_date(row.get("observation_date") for row in materialized),
        _latest_date(row.get("public_availability_date") for row in materialized),
    )


def _set(
    values: dict[str, float | int | None],
    observations: dict[str, dt.date | None],
    availability: dict[str, dt.date | None],
    statuses: dict[str, str],
    name: str,
    value: float | int | None,
    lineage: tuple[dt.date | None, dt.date | None] = (None, None),
    *,
    missing_status: str = "MISSING_INVALID_INPUT",
) -> None:
    values[name] = value
    if value is None:
        observations[name] = None
        availability[name] = None
        statuses[name] = missing_status
    else:
        observations[name], availability[name] = lineage
        statuses[name] = "PRESENT"


def compute_optionmetrics_features(
    *,
    link_status: str,
    secid: int | None,
    link_interval_start: dt.date | None,
    link_interval_end: dt.date | None,
    signal_date: dt.date,
    sessions: Sequence[dt.date],
    flow_rows: Iterable[Mapping[str, Any]],
    historical_volatility_rows: Iterable[Mapping[str, Any]],
    underlying_volume_by_session: Mapping[dt.date, Any],
    lagged_endpoints: Mapping[int, Mapping[str, Any]] | None = None,
    expected_lag_signal_dates: Mapping[int, dt.date] | None = None,
) -> OptionMetricsResult:
    """Compute all 26 OptionMetrics concepts without nearest-date substitution.

    ``lagged_endpoints`` must be keyed by exact prior rebalance-month lags (1 and
    3), as resolved by the caller. Each value may contain
    ``combined_volume_21d``, ``combined_open_interest_eom``, ``signal_date``,
    and ``public_availability_date``.
    """

    if link_status != "MATCHED":
        return _empty(
            "MISSING_AMBIGUOUS_LINK"
            if link_status == "AMBIGUOUS"
            else "MISSING_NOT_COVERED"
        )
    if secid is None or link_interval_start is None:
        return _empty("MISSING_INVALID_INPUT")
    if link_interval_start > signal_date:
        raise OptionMetricsFeatureError("OptionMetrics link starts after the signal")
    if link_interval_end is not None and signal_date > link_interval_end:
        raise OptionMetricsFeatureError("OptionMetrics link ends before the signal")
    ordered_sessions = list(sessions)
    if (
        not ordered_sessions
        or ordered_sessions != sorted(set(ordered_sessions))
        or ordered_sessions[-1] != signal_date
        or any(day > signal_date for day in ordered_sessions)
    ):
        raise OptionMetricsFeatureError("Sessions must be unique, sorted, and end at signal_date")

    values: dict[str, float | int | None] = {name: None for name in FEATURE_NAMES}
    observations: dict[str, dt.date | None] = {name: None for name in FEATURE_NAMES}
    availability: dict[str, dt.date | None] = {name: None for name in FEATURE_NAMES}
    statuses: dict[str, str] = {name: "MISSING_INSUFFICIENT_HISTORY" for name in FEATURE_NAMES}
    flow_index = _eligible_flow_index(
        flow_rows, int(secid), signal_date, link_interval_start
    )

    volume_windows: dict[int, tuple[float, float, tuple[Mapping[str, Any], ...]] | None] = {}
    for length in (21, 63):
        if len(ordered_sessions) < length:
            volume_windows[length] = None
            continue
        call_total = put_total = 0.0
        used: list[Mapping[str, Any]] = []
        for day in ordered_sessions[-length:]:
            pair = _pair(flow_index, day, "volume")
            if pair is None:
                used = []
                break
            call, put, rows = pair
            call_total += call
            put_total += put
            used.extend(rows)
        volume_windows[length] = (
            (call_total, put_total, tuple(used)) if used else None
        )
        suffix = f"{length}d"
        window = volume_windows[length]
        lineage = _lineage(window[2]) if window is not None else (None, None)
        _set(values, observations, availability, statuses, f"option_call_volume_{suffix}", None if window is None else window[0], lineage, missing_status="MISSING_INSUFFICIENT_HISTORY")
        _set(values, observations, availability, statuses, f"option_put_volume_{suffix}", None if window is None else window[1], lineage, missing_status="MISSING_INSUFFICIENT_HISTORY")
        _set(values, observations, availability, statuses, f"option_combined_volume_{suffix}", None if window is None else window[0] + window[1], lineage, missing_status="MISSING_INSUFFICIENT_HISTORY")

    signal_pair = _pair(flow_index, signal_date, "open_interest")
    signal_lineage = _lineage(signal_pair[2]) if signal_pair is not None else (None, None)
    call_oi = None if signal_pair is None else signal_pair[0]
    put_oi = None if signal_pair is None else signal_pair[1]
    _set(values, observations, availability, statuses, "option_call_open_interest_eom", call_oi, signal_lineage)
    _set(values, observations, availability, statuses, "option_put_open_interest_eom", put_oi, signal_lineage)
    _set(values, observations, availability, statuses, "option_combined_open_interest_eom", None if signal_pair is None else call_oi + put_oi, signal_lineage)

    window21 = volume_windows[21]
    if window21 is not None:
        call21, put21, rows21 = window21
        lineage21 = _lineage(rows21)
        _set(values, observations, availability, statuses, "log_option_put_call_volume_ratio_21d", math.log((put21 + 1.0) / (call21 + 1.0)), lineage21)
        denominator = call21 + put21
        share = 0.0 if denominator == 0.0 else (call21 - put21) / denominator
        _set(values, observations, availability, statuses, "option_call_minus_put_volume_share_21d", share, lineage21)
        underlying = [_number(underlying_volume_by_session.get(day)) for day in ordered_sessions[-21:]]
        underlying_total = None if any(value is None for value in underlying) else float(sum(underlying))
        ratio = None if underlying_total is None or underlying_total <= 0.0 else (call21 + put21) / underlying_total
        _set(values, observations, availability, statuses, "option_volume_to_underlying_volume_21d", ratio, lineage21)
    if signal_pair is not None:
        _set(values, observations, availability, statuses, "log_option_put_call_open_interest_ratio_eom", math.log((put_oi + 1.0) / (call_oi + 1.0)), signal_lineage)

    coverage_rows = (
        flow_index.get((signal_date, "CALL")),
        flow_index.get((signal_date, "PUT")),
    )
    has_signal_coverage = all(row is not None for row in coverage_rows)
    coverage_lineage = (
        _lineage(row for row in coverage_rows if row is not None)
        if has_signal_coverage
        else (None, None)
    )
    _set(values, observations, availability, statuses, "optionmetrics_covered_indicator", 1 if has_signal_coverage else None, coverage_lineage, missing_status="MISSING_INSUFFICIENT_HISTORY")
    complete_dates = sorted(
        day
        for day in {key[0] for key in flow_index}
        if link_interval_start <= day <= signal_date
        and (day, "CALL") in flow_index
        and (day, "PUT") in flow_index
    )
    age = _completed_months(complete_dates[0], signal_date) if complete_dates and has_signal_coverage else None
    _set(values, observations, availability, statuses, "optionmetrics_coverage_age_months", age, coverage_lineage, missing_status="MISSING_INSUFFICIENT_HISTORY")

    current_sources = {
        "combined_volume_21d": (values["option_combined_volume_21d"], observations["option_combined_volume_21d"], availability["option_combined_volume_21d"]),
        "combined_open_interest_eom": (values["option_combined_open_interest_eom"], observations["option_combined_open_interest_eom"], availability["option_combined_open_interest_eom"]),
    }
    for lag in (1, 3):
        prior = (lagged_endpoints or {}).get(lag)
        expected_signal = (expected_lag_signal_dates or {}).get(lag)
        expected_valid = (
            isinstance(expected_signal, dt.date)
            and expected_signal < signal_date
            and (signal_date.year * 12 + signal_date.month)
            - (expected_signal.year * 12 + expected_signal.month)
            == lag
        )
        prior_signal = None if prior is None else prior.get("signal_date")
        prior_availability = (
            None if prior is None else prior.get("public_availability_date")
        )
        prior_timing_valid = (
            expected_valid
            and prior_signal == expected_signal
            and isinstance(prior_availability, dt.date)
            and prior_availability <= signal_date
        )
        for stem, feature in (
            ("combined_volume_21d", f"option_total_volume_change_{lag}m"),
            ("combined_open_interest_eom", f"option_total_open_interest_change_{lag}m"),
        ):
            current_value, current_observation, current_availability = current_sources[stem]
            prior_value = _number(
                None if prior is None or not prior_timing_valid else prior.get(stem)
            )
            value = None if current_value is None or prior_value is None else math.log1p(float(current_value)) - math.log1p(prior_value)
            lineage = (
                _latest_date((current_observation, None if prior is None else prior.get("signal_date"))),
                _latest_date((current_availability, None if prior is None else prior.get("public_availability_date"))),
            )
            _set(values, observations, availability, statuses, feature, value, lineage, missing_status="MISSING_INSUFFICIENT_HISTORY")

    hv_index: dict[int, Mapping[str, Any]] = {}
    for row in historical_volatility_rows:
        if (
            int(row.get("secid", -1)) != int(secid)
            or row.get("observation_date") != signal_date
            or row.get("supported_horizon") is not True
            or not isinstance(row.get("public_availability_date"), dt.date)
            or row["public_availability_date"] > signal_date
        ):
            continue
        days = int(row.get("days", -1))
        if days in hv_index:
            raise OptionMetricsFeatureError(f"Duplicate historical-volatility key: {days}")
        hv_index[days] = row
    for nominal, vendor in VENDOR_HORIZONS.items():
        row = hv_index.get(vendor)
        value = None if row is None else _number(row.get("volatility"))
        lineage = (signal_date, row.get("public_availability_date")) if value is not None else (None, None)
        _set(values, observations, availability, statuses, f"option_historical_volatility_{nominal}d", value, lineage)
    for name, left, right in (
        ("option_historical_volatility_30d_minus_90d", "option_historical_volatility_30d", "option_historical_volatility_90d"),
        ("option_historical_volatility_60d_minus_180d", "option_historical_volatility_60d", "option_historical_volatility_180d"),
    ):
        value = None if values[left] is None or values[right] is None else float(values[left]) - float(values[right])
        lineage = (
            _latest_date((observations[left], observations[right])),
            _latest_date((availability[left], availability[right])),
        )
        _set(values, observations, availability, statuses, name, value, lineage)

    stale_after = {
        name: link_interval_end if value is not None else None
        for name, value in values.items()
    }
    allowed_statuses = {
        "PRESENT",
        "MISSING_AMBIGUOUS_LINK",
        "MISSING_INSUFFICIENT_HISTORY",
        "MISSING_INVALID_INPUT",
        "MISSING_NOT_COVERED",
        "MISSING_SOURCE_UNAVAILABLE",
        "MISSING_STALE",
    }
    if not set(statuses.values()).issubset(allowed_statuses):
        raise OptionMetricsFeatureError("OptionMetrics value status escaped frozen vocabulary")
    return OptionMetricsResult(values, observations, availability, stale_after, statuses)
