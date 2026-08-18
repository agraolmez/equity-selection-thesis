"""Point-in-time aggregate-state features for roadmap Stage 6.

The builders in this module consume only the three verified factor/liquidity
row families.  They deliberately do not read files, preprocess the resulting
features, or construct labels, model inputs, portfolios, or performance.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from portfolio_bot.features.feature_math import realized_volatility
from portfolio_bot.features.feature_lineage import (
    FeatureFamilyResult,
    FeatureLineage,
    make_lineage,
)


FACTOR_NAMES = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
TRAILING_FACTOR_MONTHS = (1, 3, 12)
MINIMUM_PS_PRIOR_PAIRS = 60

AGGREGATE_STATE_FEATURE_NAMES = (
    *(
        f"factor_{factor}_trailing_{months}m"
        for months in TRAILING_FACTOR_MONTHS
        for factor in FACTOR_NAMES
    ),
    "risk_free_rate_1m",
    "market_realized_volatility_21d",
    "market_realized_volatility_63d",
    "pastor_stambaugh_liquidity_level",
    "pastor_stambaugh_recursive_innovation",
)


class AggregateStateFeatureError(RuntimeError):
    """Raised when a supposedly verified aggregate source is ambiguous."""


@dataclass(frozen=True)
class AggregateStateResult(FeatureFamilyResult):
    """The 23 raw economic features in their frozen registry order."""


def _date(row: Mapping[str, Any], field: str, source: str) -> dt.date:
    value = row.get(field)
    if not isinstance(value, dt.date) or isinstance(value, dt.datetime):
        raise AggregateStateFeatureError(
            f"{source} row has a non-date {field}: {value!r}"
        )
    return value


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _unique_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        marker = id(row)
        if marker not in seen:
            seen.add(marker)
            result.append(row)
    return tuple(result)


def _lineage(
    value: object,
    rows: Iterable[Mapping[str, Any]],
    *,
    missing_status: str,
) -> FeatureLineage:
    contributors = _unique_rows(rows)
    observation = (
        max(_date(row, "observation_date", "aggregate lineage") for row in contributors)
        if contributors
        else None
    )
    availability = (
        max(
            _date(row, "public_availability_date", "aggregate lineage")
            for row in contributors
        )
        if contributors
        else None
    )
    return make_lineage(
        value,
        observation,
        availability,
        missing_status=missing_status,
    )


def _month_ordinal(value: dt.date) -> int:
    return value.year * 12 + value.month - 1


def _known_by_signal(
    row: Mapping[str, Any], signal_date: dt.date, source: str
) -> bool:
    return (
        _date(row, "observation_date", source) <= signal_date
        and _date(row, "public_availability_date", source) <= signal_date
    )


def _monthly_factor_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        factor_month = _date(row, "factor_month", "monthly factor")
        observation_date = _date(row, "observation_date", "monthly factor")
        _date(row, "public_availability_date", "monthly factor")
        if factor_month != observation_date:
            raise AggregateStateFeatureError(
                "Monthly factor_month must equal observation_date"
            )
        month = _month_ordinal(factor_month)
        if month in result:
            raise AggregateStateFeatureError(
                f"Duplicate monthly factor calendar month: {factor_month:%Y-%m}"
            )
        result[month] = row
    return result


def _monthly_factor_result(
    rows: Iterable[Mapping[str, Any]], signal_date: dt.date
) -> tuple[dict[str, float | None], dict[str, FeatureLineage]]:
    index = _monthly_factor_index(rows)
    signal_month = _month_ordinal(signal_date)
    values: dict[str, float | None] = {}
    lineage: dict[str, FeatureLineage] = {}
    for months in TRAILING_FACTOR_MONTHS:
        requested = range(signal_month - months + 1, signal_month + 1)
        for factor in FACTOR_NAMES:
            inputs: list[float] = []
            contributors: list[Mapping[str, Any]] = []
            missing_status = "MISSING_INSUFFICIENT_HISTORY"
            for month in requested:
                row = index.get(month)
                if row is None:
                    inputs = []
                    break
                if not _known_by_signal(row, signal_date, "monthly factor"):
                    inputs = []
                    missing_status = "MISSING_SOURCE_UNAVAILABLE"
                    break
                value = _finite_number(row.get(factor))
                if value is None:
                    inputs = []
                    missing_status = "MISSING_INVALID_INPUT"
                    break
                inputs.append(value)
                contributors.append(row)
            total = (
                float(np.sum(np.asarray(inputs, dtype=np.float64), dtype=np.float64))
                if len(inputs) == months
                else None
            )
            values[f"factor_{factor}_trailing_{months}m"] = (
                total if total is not None and math.isfinite(total) else None
            )
            name = f"factor_{factor}_trailing_{months}m"
            lineage[name] = _lineage(
                values[name],
                contributors,
                missing_status=missing_status,
            )

    eligible = [
        (month, row)
        for month, row in index.items()
        if month <= signal_month
        and _known_by_signal(row, signal_date, "monthly factor")
    ]
    latest = max(eligible, default=None, key=lambda item: item[0])
    rf = None if latest is None else _finite_number(latest[1].get("rf"))
    values["risk_free_rate_1m"] = rf
    lineage["risk_free_rate_1m"] = _lineage(
        rf,
        () if latest is None else (latest[1],),
        missing_status=(
            "MISSING_SOURCE_UNAVAILABLE"
            if latest is None
            else "MISSING_INVALID_INPUT"
        ),
    )
    return values, lineage


def compute_monthly_factor_features(
    rows: Iterable[Mapping[str, Any]], signal_date: dt.date
) -> dict[str, float | None]:
    """Build 18 trailing factor sums and the latest known monthly RF.

    Factor sums require the requested consecutive calendar months through the
    signal month.  An unavailable or invalid month makes only the affected
    factor/window missing; it is never replaced with an older month.
    """

    return _monthly_factor_result(rows, signal_date)[0]


def _daily_factor_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    seen: set[dt.date] = set()
    result: list[Mapping[str, Any]] = []
    for row in rows:
        observation_date = _date(row, "observation_date", "daily factor")
        _date(row, "public_availability_date", "daily factor")
        if observation_date in seen:
            raise AggregateStateFeatureError(
                f"Duplicate daily factor session: {observation_date.isoformat()}"
            )
        seen.add(observation_date)
        result.append(row)
    result.sort(key=lambda row: row["observation_date"])
    return result


def _market_volatility_result(
    rows: Iterable[Mapping[str, Any]], signal_date: dt.date
) -> tuple[dict[str, float | None], dict[str, FeatureLineage]]:
    history = [
        row
        for row in _daily_factor_rows(rows)
        if row["observation_date"] <= signal_date
    ]
    ends_at_signal = bool(history) and history[-1]["observation_date"] == signal_date
    values: dict[str, float | None] = {}
    lineage: dict[str, FeatureLineage] = {}
    for sessions in (21, 63):
        value: float | None = None
        contributors: list[Mapping[str, Any]] = []
        missing_status = "MISSING_INSUFFICIENT_HISTORY"
        if ends_at_signal and len(history) >= sessions:
            window = history[-sessions:]
            inputs = [_finite_number(row.get("mktrf")) for row in window]
            if any(
                not _known_by_signal(row, signal_date, "daily factor")
                for row in window
            ):
                missing_status = "MISSING_SOURCE_UNAVAILABLE"
            elif any(factor is None for factor in inputs):
                missing_status = "MISSING_INVALID_INPUT"
            else:
                contributors = list(window)
                value = realized_volatility(
                    factor for factor in inputs if factor is not None
                )
                if value is None:
                    missing_status = "MISSING_INVALID_INPUT"
        name = f"market_realized_volatility_{sessions}d"
        values[name] = value
        lineage[name] = _lineage(
            value,
            contributors,
            missing_status=missing_status,
        )
    return values, lineage


def compute_market_volatility_features(
    rows: Iterable[Mapping[str, Any]], signal_date: dt.date
) -> dict[str, float | None]:
    """Build annualized 21/63-session volatility from daily MKT-RF.

    The supplied verified daily rows define the global trading-session order.
    The window must end on the signal session and every selected factor row
    must already be publicly available at that after-close signal.
    """

    return _market_volatility_result(rows, signal_date)[0]


def _pastor_stambaugh_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        observation_month = _date(
            row, "observation_month", "Pastor-Stambaugh"
        )
        observation_date = _date(
            row, "observation_date", "Pastor-Stambaugh"
        )
        _date(row, "public_availability_date", "Pastor-Stambaugh")
        if observation_month != observation_date:
            raise AggregateStateFeatureError(
                "Pastor-Stambaugh observation_month must equal observation_date"
            )
        month = _month_ordinal(observation_month)
        if month in result:
            raise AggregateStateFeatureError(
                "Duplicate Pastor-Stambaugh observation month: "
                f"{observation_month:%Y-%m}"
            )
        result[month] = row
    return result


def _recursive_ps_innovation_result(
    index: Mapping[int, Mapping[str, Any]],
    candidate_month: int,
    signal_date: dt.date,
) -> tuple[float | None, tuple[Mapping[str, Any], ...], str]:
    candidate = index[candidate_month]
    current_level = _finite_number(candidate.get("ps_level"))
    previous = index.get(candidate_month - 1)
    previous_level = None if previous is None else _finite_number(previous.get("ps_level"))
    if previous is None:
        return None, (), "MISSING_INSUFFICIENT_HISTORY"
    if current_level is None or previous_level is None:
        return None, (), "MISSING_INVALID_INPUT"

    known = {
        month: row
        for month, row in index.items()
        if month < candidate_month
        and _known_by_signal(row, signal_date, "Pastor-Stambaugh")
    }
    lagged: list[float] = []
    responses: list[float] = []
    endpoints: list[int] = []
    contributing_rows: list[Mapping[str, Any]] = [candidate]
    for endpoint in sorted(known):
        prior = known.get(endpoint - 1)
        if prior is None:
            continue
        lag = _finite_number(prior.get("ps_level"))
        response = _finite_number(known[endpoint].get("ps_level"))
        if lag is None or response is None:
            continue
        lagged.append(lag)
        responses.append(response)
        endpoints.append(endpoint)
        contributing_rows.extend((prior, known[endpoint]))

    if (
        len(lagged) < MINIMUM_PS_PRIOR_PAIRS
        or not endpoints
        or endpoints[-1] != candidate_month - 1
    ):
        return None, (), "MISSING_INSUFFICIENT_HISTORY"
    lag_vector = np.asarray(lagged, dtype=np.float64)
    response_vector = np.asarray(responses, dtype=np.float64)
    if (
        not np.isfinite(lag_vector).all()
        or not np.isfinite(response_vector).all()
        or float(np.ptp(lag_vector)) == 0.0
    ):
        return None, (), "MISSING_INVALID_INPUT"
    design = np.column_stack(
        [np.ones(lag_vector.size, dtype=np.float64), lag_vector]
    )
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, response_vector, rcond=1e-12
        )
    except np.linalg.LinAlgError:
        return None, (), "MISSING_INVALID_INPUT"
    if rank != 2 or not np.isfinite(coefficients).all():
        return None, (), "MISSING_INVALID_INPUT"
    innovation = float(
        current_level
        - (float(coefficients[0]) + float(coefficients[1]) * previous_level)
    )
    if not math.isfinite(innovation):
        return None, (), "MISSING_INVALID_INPUT"
    return innovation, _unique_rows(contributing_rows), "MISSING_INVALID_INPUT"


def _recursive_ps_innovation(
    index: Mapping[int, Mapping[str, Any]],
    candidate_month: int,
    signal_date: dt.date,
) -> float | None:
    return _recursive_ps_innovation_result(index, candidate_month, signal_date)[0]


def _pastor_stambaugh_result(
    rows: Iterable[Mapping[str, Any]], signal_date: dt.date
) -> tuple[dict[str, float | None], dict[str, FeatureLineage]]:
    index = _pastor_stambaugh_index(rows)
    missing = {
        "pastor_stambaugh_liquidity_level": None,
        "pastor_stambaugh_recursive_innovation": None,
    }
    missing_lineage = {
        name: _lineage(
            None,
            (),
            missing_status="MISSING_SOURCE_UNAVAILABLE",
        )
        for name in missing
    }
    if not index:
        return missing, missing_lineage
    final_row = index[max(index)]
    series_end_month = _month_ordinal(
        _date(
            final_row,
            "public_availability_date",
            "Pastor-Stambaugh",
        )
    )
    if _month_ordinal(signal_date) > series_end_month:
        return missing, missing_lineage

    eligible = [
        month
        for month, row in index.items()
        if _known_by_signal(row, signal_date, "Pastor-Stambaugh")
    ]
    if not eligible:
        return missing, missing_lineage
    candidate_month = max(eligible)
    candidate = index[candidate_month]
    level = _finite_number(candidate.get("ps_level"))
    innovation, innovation_rows, innovation_status = (
        _recursive_ps_innovation_result(index, candidate_month, signal_date)
        if level is not None
        else (None, (), "MISSING_INVALID_INPUT")
    )
    values = {
        "pastor_stambaugh_liquidity_level": level,
        "pastor_stambaugh_recursive_innovation": innovation,
    }
    lineage = {
        "pastor_stambaugh_liquidity_level": _lineage(
            level,
            (candidate,),
            missing_status="MISSING_INVALID_INPUT",
        ),
        "pastor_stambaugh_recursive_innovation": _lineage(
            innovation,
            innovation_rows,
            missing_status=innovation_status,
        ),
    }
    return values, lineage


def compute_pastor_stambaugh_features(
    rows: Iterable[Mapping[str, Any]], signal_date: dt.date
) -> dict[str, float | None]:
    """Build the latest level and past-only expanding AR(1) innovation.

    The final source observation may be used in its availability month, but
    neither feature is carried into a later calendar month after verified
    series coverage ends.
    """

    return _pastor_stambaugh_result(rows, signal_date)[0]


def compute_aggregate_state_features(
    monthly_factor_rows: Iterable[Mapping[str, Any]],
    daily_factor_rows: Iterable[Mapping[str, Any]],
    pastor_stambaugh_rows: Iterable[Mapping[str, Any]],
    signal_date: dt.date,
) -> AggregateStateResult:
    """Build all 23 raw aggregate-state features for one after-close signal."""

    if not isinstance(signal_date, dt.date) or isinstance(signal_date, dt.datetime):
        raise AggregateStateFeatureError(f"Invalid signal_date: {signal_date!r}")
    values: dict[str, float | None] = {}
    lineage: dict[str, FeatureLineage] = {}
    monthly_values, monthly_lineage = _monthly_factor_result(
        monthly_factor_rows, signal_date
    )
    daily_values, daily_lineage = _market_volatility_result(
        daily_factor_rows, signal_date
    )
    ps_values, ps_lineage = _pastor_stambaugh_result(
        pastor_stambaugh_rows, signal_date
    )
    for family_values, family_lineage in (
        (monthly_values, monthly_lineage),
        (daily_values, daily_lineage),
        (ps_values, ps_lineage),
    ):
        values.update(family_values)
        lineage.update(family_lineage)
    if (
        tuple(values) != AGGREGATE_STATE_FEATURE_NAMES
        or tuple(lineage) != AGGREGATE_STATE_FEATURE_NAMES
    ):
        raise AggregateStateFeatureError("Aggregate-state feature order drifted")
    return AggregateStateResult(values=values, lineage=lineage)
