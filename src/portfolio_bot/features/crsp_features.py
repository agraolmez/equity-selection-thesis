"""Pure deterministic CRSP feature builders for roadmap Stage 6.

The functions in this module operate only on already-typed, in-memory rows.  They
do not perform I/O, preprocessing, label construction, or model fitting.
"""

from __future__ import annotations

import calendar
import datetime as dt
import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from portfolio_bot.features.feature_math import (
    compounded_return,
    corwin_schultz_pair,
    distance_to_high,
    downside_semivolatility,
    ewls_factor_fit,
    log_price_trend,
    maximum_drawdown,
    ordinary_median,
    price_to_moving_average,
    realized_volatility,
    sample_excess_kurtosis,
    sample_skewness,
    type7_quantile,
)
from portfolio_bot.features.feature_lineage import (
    FeatureFamilyResult,
    FeatureLineage,
    make_lineage,
)


PRICE_TREND_FEATURE_NAMES = (
    "return_total_5d",
    "return_total_63d",
    "return_total_126d",
    "return_total_252d",
    "momentum_6_1",
    "momentum_12_1",
    "momentum_24_12",
    "reversal_21d",
    "price_to_ma_21d",
    "price_to_ma_63d",
    "price_to_ma_126d",
    "price_to_ma_252d",
    "distance_to_high_252d",
    "log_price_trend_slope_63d",
    "log_price_trend_r2_63d",
    "log_price_trend_slope_126d",
    "log_price_trend_r2_126d",
    "log_price_trend_slope_252d",
    "log_price_trend_r2_252d",
    "max_drawdown_252d",
)

FACTOR_NAMES = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
BETA_FEATURE_NAMES = tuple(f"beta_label_{name}" for name in FACTOR_NAMES)
RISK_DISTRIBUTION_FEATURE_NAMES = (
    "realized_volatility_21d",
    "realized_volatility_63d",
    "realized_volatility_126d",
    "realized_volatility_252d",
    "downside_semivolatility_63d",
    "downside_semivolatility_252d",
    *BETA_FEATURE_NAMES,
    "beta_label_idiosyncratic_volatility",
    "return_skewness_252d",
    "return_excess_kurtosis_252d",
    "maximum_daily_return_21d",
    "minimum_daily_return_21d",
    "maximum_daily_return_63d",
    "minimum_daily_return_63d",
    "volatility_63d_vs_trailing_median_252d",
)

SIZE_LIQUIDITY_FEATURE_NAMES = (
    "log_absolute_price",
    "log_market_cap",
    "median_dollar_volume_20d",
    "median_dollar_volume_60d",
    "share_turnover_20d",
    "share_turnover_60d",
    "amihud_illiquidity_20d",
    "amihud_illiquidity_60d",
    "median_estimated_full_spread_20d",
    "zero_return_share_60d",
    "zero_volume_share_60d",
    "dollar_volume_shock_20d_vs_prior_252d",
    "months_since_first_tradable",
)

CRSP_FEATURE_NAMES = (
    *PRICE_TREND_FEATURE_NAMES,
    *RISK_DISTRIBUTION_FEATURE_NAMES,
    *SIZE_LIQUIDITY_FEATURE_NAMES,
)


class CrspFeatureError(RuntimeError):
    """Raised for ambiguous structural keys in otherwise canonical input rows."""


@dataclass(frozen=True)
class CrspFeatureInput:
    """In-memory inputs for one security at one signal date.

    ``sessions`` is the verified global CRSP session sequence and may include
    history before the security first traded.  Rows need not be pre-sorted.
    """

    permno: int
    signal_date: dt.date
    sessions: Sequence[dt.date]
    market_rows: Sequence[Mapping[str, Any]]
    return_rows: Sequence[Mapping[str, Any]]
    factor_rows: Sequence[Mapping[str, Any]] = ()
    shares_rows: Sequence[Mapping[str, Any]] = ()
    listing_history_rows: Sequence[Mapping[str, Any]] = ()
    long_eligible: bool = False
    primary_exchange: str | None = None
    prepared_cache: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )


def _prepared_cache(data: CrspFeatureInput) -> Any | None:
    """Return an authenticated duck-typed cache attached by the fast engine."""

    cache = data.prepared_cache
    if cache is None:
        return None
    matches = getattr(cache, "matches", None)
    if not callable(matches) or not matches(data):
        raise CrspFeatureError("Prepared CRSP cache does not match feature input")
    return cache


def _empty(names: Sequence[str]) -> dict[str, float | None]:
    return {name: None for name in names}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _date(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    return value if isinstance(value, dt.date) else None


def _session_context(data: CrspFeatureInput) -> tuple[tuple[dt.date, ...], int] | None:
    cache = _prepared_cache(data)
    if cache is not None:
        return cache.session_context
    sessions = tuple(data.sessions)
    if (
        not sessions
        or any(not isinstance(value, dt.date) for value in sessions)
        or any(left >= right for left, right in zip(sessions, sessions[1:]))
    ):
        return None
    try:
        signal_index = sessions.index(data.signal_date)
    except ValueError:
        return None
    return sessions, signal_index


def _window_dates(
    context: tuple[tuple[dt.date, ...], int] | None,
    start_offset: int,
    end_offset: int,
) -> tuple[dt.date, ...] | None:
    if context is None or start_offset > end_offset or end_offset > 0:
        return None
    sessions, signal_index = context
    start = signal_index + start_offset
    end = signal_index + end_offset
    if start < 0 or end >= len(sessions):
        return None
    return sessions[start : end + 1]


def _index_rows(
    rows: Iterable[Mapping[str, Any]], date_field: str
) -> dict[dt.date, Mapping[str, Any]]:
    result: dict[dt.date, Mapping[str, Any]] = {}
    for row in rows:
        value = _date(row.get(date_field))
        if value is None:
            continue
        if value in result:
            raise CrspFeatureError(f"Duplicate {date_field}: {value.isoformat()}")
        result[value] = row
    return result


def _market_values(
    index: Mapping[dt.date, Mapping[str, Any]],
    dates: Sequence[dt.date] | None,
    field: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> list[float] | None:
    if dates is None:
        return None
    values: list[float] = []
    for date in dates:
        row = index.get(date)
        value = None if row is None else _number(row.get(field))
        if (
            value is None
            or (positive and value <= 0.0)
            or (nonnegative and value < 0.0)
        ):
            return None
        values.append(value)
    return values


def _return_leg_coverage(
    row: Mapping[str, Any], sessions: Sequence[dt.date]
) -> tuple[int, ...] | None:
    start = _date(row.get("return_span_start_date"))
    end = _date(row.get("return_span_end_date"))
    declared = row.get("covered_trading_sessions")
    if start is None or end is None or start >= end or isinstance(declared, bool):
        return None
    try:
        declared_count = int(declared)
    except (TypeError, ValueError, OverflowError):
        return None
    first = bisect_right(sessions, start)
    after_last = bisect_right(sessions, end)
    covered = tuple(range(first, after_last))
    if declared_count <= 0 or len(covered) != declared_count:
        return None
    return covered


PreparedReturnLeg = tuple[tuple[int, ...], float]


def _prepare_return_legs(data: CrspFeatureInput) -> tuple[PreparedReturnLeg, ...]:
    cache = _prepared_cache(data)
    if cache is not None:
        return cache.prepared_return_legs
    context = _session_context(data)
    if context is None:
        return ()
    sessions, _ = context
    result: list[PreparedReturnLeg] = []
    for row in data.return_rows:
        if row.get("return_history_usable") is not True:
            continue
        observation = _date(row.get("calendar_date"))
        span_end = _date(row.get("return_span_end_date"))
        availability = _date(row.get("outcome_availability_date"))
        value = _number(row.get("authoritative_total_return"))
        coverage = _return_leg_coverage(row, sessions)
        if (
            observation is not None
            and observation <= data.signal_date
            and span_end == observation
            and availability is not None
            and availability <= data.signal_date
            and value is not None
            and value >= -1.0
            and coverage is not None
        ):
            result.append((coverage, value))
    return tuple(result)


def _exact_return_values(
    data: CrspFeatureInput,
    dates: Sequence[dt.date] | None,
    *,
    one_session_only: bool,
    prepared: Sequence[PreparedReturnLeg] | None = None,
) -> list[float] | None:
    cache = _prepared_cache(data)
    if cache is not None:
        return cache.exact_return_values(
            dates,
            one_session_only=one_session_only,
        )
    context = _session_context(data)
    if context is None or dates is None:
        return None
    sessions, _ = context
    position = {date: index for index, date in enumerate(sessions)}
    try:
        target = {position[date] for date in dates}
    except KeyError:
        return None
    selected: list[tuple[int, tuple[int, ...], float]] = []
    for coverage, value in prepared if prepared is not None else _prepare_return_legs(data):
        if one_session_only and len(coverage) != 1:
            continue
        overlap = target.intersection(coverage)
        if not overlap:
            continue
        # A leg crossing either requested endpoint cannot be prorated.
        if not set(coverage).issubset(target):
            return None
        selected.append((coverage[0], coverage, value))
    counts = {index: 0 for index in target}
    for _, coverage, _ in selected:
        for index in coverage:
            if index in counts:
                counts[index] += 1
    if not counts or any(count != 1 for count in counts.values()):
        return None
    selected.sort(key=lambda item: item[0])
    return [item[2] for item in selected]


def _exact_price_return_rows(
    data: CrspFeatureInput,
    dates: Sequence[dt.date] | None,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[float, ...]] | None:
    """Select exact, ordinary one-session CRSP price-return transitions."""

    if dates is None or not dates:
        return None
    rows = _exact_return_contributor_rows(
        data,
        dates,
        one_session_only=True,
    )
    context = _session_context(data)
    if rows is None or context is None or len(rows) != len(dates):
        return None
    sessions, _ = context
    positions = {day: index for index, day in enumerate(sessions)}
    values: list[float] = []
    for expected_day, row in zip(dates, rows, strict=True):
        position = positions.get(expected_day)
        if position is None or position == 0:
            return None
        mapped = row.get("mapped_trading_periods", 1)
        covered = row.get("covered_trading_sessions")
        try:
            mapped_count = int(mapped)
            covered_count = int(covered)
        except (TypeError, ValueError, OverflowError):
            return None
        value = _number(row.get("raw_return_ex_distributions"))
        if (
            _date(row.get("calendar_date")) != expected_day
            or _date(row.get("return_span_start_date"))
            != sessions[position - 1]
            or _date(row.get("return_span_end_date")) != expected_day
            or mapped_count != 1
            or covered_count != 1
            or row.get("return_leg_class", "ordinary_one_session")
            != "ordinary_one_session"
            or row.get("authoritative_return_state", "observed")
            != "observed"
            or value is None
            or value <= -1.0
        ):
            return None
        values.append(value)
    return tuple(rows), tuple(values)


def _adjusted_price_return_path(
    data: CrspFeatureInput,
    dates: Sequence[dt.date] | None,
) -> list[float] | None:
    """Build a split-invariant normalized path from CRSP price returns.

    The oldest level is one.  Each later level compounds CRSP DlyRetx, which
    measures price appreciation without ordinary dividends while incorporating
    the price factors needed for splits and other qualifying distributions.
    Literal positive closes remain a completeness gate, but never determine a
    path level.  Every transition must be an exact ordinary one-session row.
    """

    if dates is None or not dates:
        return None
    cache = _prepared_cache(data)
    market = (
        cache.market_index
        if cache is not None
        else _index_rows(data.market_rows, "calendar_date")
    )
    if _market_values(
        market,
        dates,
        "close_usd_per_share",
        positive=True,
    ) is None:
        return None
    if len(dates) == 1:
        return [1.0]
    selected = _exact_price_return_rows(data, dates[1:])
    if selected is None:
        return None
    _, returns = selected
    levels = [1.0]
    wealth = 1.0
    for value in returns:
        if not math.isfinite(value) or value <= -1.0:
            return None
        wealth *= 1.0 + value
        if not math.isfinite(wealth) or wealth <= 0.0:
            return None
        levels.append(wealth)
    return levels


def _valid_one_session_return_map(
    data: CrspFeatureInput,
    prepared: Sequence[PreparedReturnLeg] | None = None,
) -> dict[dt.date, float]:
    cache = _prepared_cache(data)
    if cache is not None:
        return cache.valid_one_session_return_map
    context = _session_context(data)
    if context is None:
        return {}
    sessions, _ = context
    result: dict[dt.date, float] = {}
    ambiguous: set[dt.date] = set()
    for coverage, value in prepared if prepared is not None else _prepare_return_legs(data):
        if len(coverage) != 1:
            continue
        date = sessions[coverage[0]]
        if date in result:
            ambiguous.add(date)
        else:
            result[date] = value
    for date in ambiguous:
        result.pop(date, None)
    return result


def _finite_result(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def build_price_trend_features(data: CrspFeatureInput) -> dict[str, float | None]:
    """Build the frozen 20 price/trend features for one security-signal row."""

    values = _empty(PRICE_TREND_FEATURE_NAMES)
    context = _session_context(data)
    if context is None:
        return values
    prepared = _prepare_return_legs(data)

    return_windows = {
        "return_total_5d": (-4, 0),
        "return_total_63d": (-62, 0),
        "return_total_126d": (-125, 0),
        "return_total_252d": (-251, 0),
        "momentum_6_1": (-126, -22),
        "momentum_12_1": (-252, -22),
        "momentum_24_12": (-504, -253),
    }
    for name, (start, end) in return_windows.items():
        dates = _window_dates(context, start, end)
        returns = _exact_return_values(
            data, dates, one_session_only=False, prepared=prepared
        )
        values[name] = None if returns is None else compounded_return(returns)

    reversal_dates = _window_dates(context, -20, 0)
    reversal_returns = _exact_return_values(
        data, reversal_dates, one_session_only=False, prepared=prepared
    )
    reversal = None if reversal_returns is None else compounded_return(reversal_returns)
    values["reversal_21d"] = None if reversal is None else -reversal

    for window in (21, 63, 126, 252):
        dates = _window_dates(context, -window + 1, 0)
        adjusted_path = _adjusted_price_return_path(data, dates)
        values[f"price_to_ma_{window}d"] = (
            None
            if adjusted_path is None
            else price_to_moving_average(adjusted_path)
        )

    dates_252 = _window_dates(context, -251, 0)
    adjusted_path_252 = _adjusted_price_return_path(data, dates_252)
    values["distance_to_high_252d"] = (
        None
        if adjusted_path_252 is None
        else distance_to_high(adjusted_path_252)
    )
    for window in (63, 126, 252):
        dates = _window_dates(context, -window + 1, 0)
        adjusted_path = _adjusted_price_return_path(data, dates)
        slope, r_squared = (
            (None, None)
            if adjusted_path is None
            else log_price_trend(adjusted_path)
        )
        values[f"log_price_trend_slope_{window}d"] = slope
        values[f"log_price_trend_r2_{window}d"] = r_squared

    drawdown_returns = _exact_return_values(
        data, dates_252, one_session_only=False, prepared=prepared
    )
    values["max_drawdown_252d"] = (
        None if drawdown_returns is None else maximum_drawdown(drawdown_returns)
    )
    return {name: _finite_result(value) for name, value in values.items()}


def _exact_one_session_window(
    data: CrspFeatureInput,
    context: tuple[tuple[dt.date, ...], int] | None,
    window: int,
    *,
    end_offset: int = 0,
    prepared: Sequence[PreparedReturnLeg] | None = None,
) -> list[float] | None:
    dates = _window_dates(context, end_offset - window + 1, end_offset)
    return _exact_return_values(
        data, dates, one_session_only=True, prepared=prepared
    )


def _beta_fit(
    data: CrspFeatureInput, prepared: Sequence[PreparedReturnLeg] | None = None
):
    context = _session_context(data)
    if context is None:
        return None
    sessions, signal_index = context
    start_index = max(0, signal_index - 251)
    returns = _valid_one_session_return_map(data, prepared)
    cache = _prepared_cache(data)
    factors = (
        cache.factor_index
        if cache is not None
        else _index_rows(data.factor_rows, "observation_date")
    )
    security_values: list[float] = []
    risk_free: list[float] = []
    factor_values: list[list[float]] = []
    ages: list[int] = []
    for index in range(start_index, signal_index + 1):
        date = sessions[index]
        security_return = returns.get(date)
        factor_row = factors.get(date)
        if security_return is None or factor_row is None:
            continue
        availability = _date(factor_row.get("public_availability_date"))
        rf = _number(factor_row.get("rf"))
        row_factors = [_number(factor_row.get(name)) for name in FACTOR_NAMES]
        if (
            availability is None
            or availability > data.signal_date
            or rf is None
            or any(value is None for value in row_factors)
        ):
            continue
        security_values.append(security_return)
        risk_free.append(rf)
        factor_values.append([float(value) for value in row_factors])
        ages.append(signal_index - index)
    return ewls_factor_fit(
        security_values,
        risk_free,
        np.asarray(factor_values, dtype=np.float64),
        ages,
        half_life_sessions=126.0,
        minimum_observations=126,
    )


def build_risk_distribution_features(
    data: CrspFeatureInput,
) -> dict[str, float | None]:
    """Build 20 unprocessed risk/distribution features.

    Beta slopes are returned before the required signal-date cross-sectional
    clipping.  :func:`build_crsp_feature_rows` applies that final clipping.
    """

    values = _empty(RISK_DISTRIBUTION_FEATURE_NAMES)
    context = _session_context(data)
    if context is None:
        return values
    prepared = _prepare_return_legs(data)

    return_cache: dict[int, list[float] | None] = {}
    for window in (21, 63, 126, 252):
        return_cache[window] = _exact_one_session_window(
            data, context, window, prepared=prepared
        )
        returns = return_cache[window]
        values[f"realized_volatility_{window}d"] = (
            None if returns is None else realized_volatility(returns)
        )
    for window in (63, 252):
        returns = return_cache[window]
        values[f"downside_semivolatility_{window}d"] = (
            None if returns is None else downside_semivolatility(returns)
        )

    fit = _beta_fit(data, prepared)
    if fit is not None:
        for name, slope in zip(BETA_FEATURE_NAMES, fit.slopes):
            values[name] = slope
        values["beta_label_idiosyncratic_volatility"] = (
            fit.annualized_idiosyncratic_volatility
        )

    returns_252 = return_cache[252]
    if returns_252 is not None:
        values["return_skewness_252d"] = sample_skewness(returns_252)
        values["return_excess_kurtosis_252d"] = sample_excess_kurtosis(
            returns_252
        )
    for window in (21, 63):
        returns = return_cache[window]
        if returns is not None:
            values[f"maximum_daily_return_{window}d"] = max(returns)
            values[f"minimum_daily_return_{window}d"] = min(returns)

    current_volatility = values["realized_volatility_63d"]
    cache = _prepared_cache(data)
    if cache is not None:
        values["volatility_63d_vs_trailing_median_252d"] = (
            cache.volatility_63d_vs_trailing_median_252d(
                current_volatility
            )
        )
    else:
        prior_estimates: list[float] = []
        for end_offset in range(-252, 0):
            returns = _exact_one_session_window(
                data,
                context,
                63,
                end_offset=end_offset,
                prepared=prepared,
            )
            estimate = (
                None
                if returns is None
                else realized_volatility(returns)
            )
            if estimate is None:
                prior_estimates = []
                break
            prior_estimates.append(estimate)
        prior_median = (
            ordinary_median(prior_estimates)
            if len(prior_estimates) == 252
            else None
        )
        if (
            current_volatility is not None
            and prior_median is not None
            and prior_median > 0.0
        ):
            values["volatility_63d_vs_trailing_median_252d"] = (
                current_volatility / prior_median - 1.0
            )
    return {name: _finite_result(value) for name, value in values.items()}


def _shares_on_dates(
    data: CrspFeatureInput,
    dates: Sequence[dt.date] | None,
) -> list[float] | None:
    if dates is None:
        return None
    cache = _prepared_cache(data)
    if cache is not None:
        return cache.shares_on_dates(dates)
    values: list[float] = []
    for date in dates:
        candidates: list[Mapping[str, Any]] = []
        for row in data.shares_rows:
            start = _date(row.get("start_date"))
            end = _date(row.get("end_date"))
            if start is not None and start <= date and (end is None or date <= end):
                candidates.append(row)
        if len(candidates) != 1:
            return None
        shares = _number(candidates[0].get("shares_outstanding"))
        if shares is None or shares <= 0.0:
            return None
        values.append(shares)
    return values


def _quoted_full_spread(row: Mapping[str, Any]) -> float | None:
    bid = _number(row.get("bid_usd_per_share"))
    ask = _number(row.get("ask_usd_per_share"))
    if bid is None or ask is None or bid <= 0.0 or ask < bid:
        return None
    midpoint = (ask + bid) / 2.0
    if midpoint <= 0.0:
        return None
    spread = (ask - bid) / midpoint
    return spread if math.isfinite(spread) and spread <= 0.20 else None


def security_level_spread(data: CrspFeatureInput) -> float | None:
    """Return the quote or Corwin--Schultz 20-session security estimate."""

    context = _session_context(data)
    dates = _window_dates(context, -19, 0)
    if dates is None:
        return None
    cache = _prepared_cache(data)
    market = (
        cache.market_index
        if cache is not None
        else _index_rows(data.market_rows, "calendar_date")
    )
    rows = [market.get(date) for date in dates]
    quotes = [
        spread
        for row in rows
        if row is not None and (spread := _quoted_full_spread(row)) is not None
    ]
    if len(quotes) >= 10:
        return ordinary_median(quotes)

    pairs: list[float] = []
    for prior, following in zip(rows, rows[1:]):
        if prior is None or following is None:
            continue
        inputs = (
            _number(prior.get("high_usd_per_share")),
            _number(prior.get("low_usd_per_share")),
            _number(prior.get("close_usd_per_share")),
            _number(following.get("high_usd_per_share")),
            _number(following.get("low_usd_per_share")),
        )
        if any(value is None for value in inputs):
            continue
        try:
            estimate = corwin_schultz_pair(*[float(value) for value in inputs])
        except (OverflowError, ValueError):
            estimate = None
        if estimate is not None:
            pairs.append(estimate)
    return ordinary_median(pairs) if len(pairs) >= 10 else None


def _signal_market_row(data: CrspFeatureInput) -> Mapping[str, Any] | None:
    cache = _prepared_cache(data)
    market = (
        cache.market_index
        if cache is not None
        else _index_rows(data.market_rows, "calendar_date")
    )
    return market.get(data.signal_date)


def _positive_signal_value(data: CrspFeatureInput, field: str) -> float | None:
    row = _signal_market_row(data)
    value = None if row is None else _number(row.get(field))
    return value if value is not None and value > 0.0 else None


def _assign_quintiles(
    inputs: Sequence[CrspFeatureInput], values: Mapping[tuple[dt.date, int], float | None]
) -> dict[tuple[dt.date, int], int]:
    eligible = [
        data
        for data in inputs
        if data.long_eligible
        and (value := values.get((data.signal_date, data.permno))) is not None
        and math.isfinite(value)
        and value > 0.0
    ]
    by_exchange: dict[str, list[CrspFeatureInput]] = defaultdict(list)
    for data in eligible:
        if data.primary_exchange in {"N", "A", "Q"}:
            by_exchange[data.primary_exchange].append(data)
    result: dict[tuple[dt.date, int], int] = {}
    for exchange_group in by_exchange.values():
        exchange_group.sort(
            key=lambda data: (
                values[(data.signal_date, data.permno)],
                data.permno,
            )
        )
        count = len(exchange_group)
        for rank, data in enumerate(exchange_group):
            result[(data.signal_date, data.permno)] = (
                math.floor(5 * rank / count) + 1
            )
    return result


def resolve_spread_hierarchy(
    inputs: Sequence[CrspFeatureInput],
    security_spreads: Mapping[tuple[dt.date, int], float | None] | None = None,
) -> dict[tuple[dt.date, int], float | None]:
    """Resolve the full signal-date quote/CS/5x5 peer hierarchy."""

    result: dict[tuple[dt.date, int], float | None] = {}
    by_signal: dict[dt.date, list[CrspFeatureInput]] = defaultdict(list)
    for data in inputs:
        by_signal[data.signal_date].append(data)

    supplied = security_spreads or {}
    for signal_date, group in by_signal.items():
        local: dict[tuple[dt.date, int], float | None] = {}
        for data in group:
            key = (signal_date, data.permno)
            candidate = supplied[key] if key in supplied else security_level_spread(data)
            local[key] = (
                candidate
                if candidate is not None
                and math.isfinite(candidate)
                and candidate >= 0.0
                else None
            )
        caps = {
            (signal_date, data.permno): _positive_signal_value(data, "market_cap_usd")
            for data in group
        }
        advs: dict[tuple[dt.date, int], float | None] = {}
        for data in group:
            context = _session_context(data)
            cache = _prepared_cache(data)
            market = (
                cache.market_index
                if cache is not None
                else _index_rows(data.market_rows, "calendar_date")
            )
            values = _market_values(
                market,
                _window_dates(context, -19, 0),
                "screening_dollar_volume_usd",
                nonnegative=True,
            )
            advs[(signal_date, data.permno)] = (
                None if values is None else ordinary_median(values)
            )
        size_bin = _assign_quintiles(group, caps)
        adv_bin = _assign_quintiles(group, advs)

        contributors = [
            data
            for data in group
            if data.long_eligible
            and (spread := local[(signal_date, data.permno)]) is not None
            and math.isfinite(spread)
        ]

        def peer_median(candidates: Iterable[CrspFeatureInput]) -> float | None:
            values = [local[(signal_date, item.permno)] for item in candidates]
            finite = [float(value) for value in values if value is not None and math.isfinite(value)]
            return ordinary_median(finite) if len(finite) >= 20 else None

        for data in group:
            key = (signal_date, data.permno)
            if local[key] is not None:
                result[key] = local[key]
                continue
            if not data.long_eligible or key not in size_bin or key not in adv_bin:
                result[key] = None
                continue
            exchange = data.primary_exchange
            size = size_bin[key]
            adv = adv_bin[key]
            candidate_groups = [
                [item for item in contributors if item.primary_exchange == exchange and size_bin.get((signal_date, item.permno)) == size and adv_bin.get((signal_date, item.permno)) == adv],
                [item for item in contributors if item.primary_exchange == exchange and size_bin.get((signal_date, item.permno)) == size and abs(adv_bin.get((signal_date, item.permno), -99) - adv) <= 1],
                [item for item in contributors if item.primary_exchange == exchange and size_bin.get((signal_date, item.permno)) == size and abs(adv_bin.get((signal_date, item.permno), -99) - adv) <= 2],
                [item for item in contributors if item.primary_exchange == exchange and abs(size_bin.get((signal_date, item.permno), -99) - size) <= 1],
                [item for item in contributors if item.primary_exchange == exchange and abs(size_bin.get((signal_date, item.permno), -99) - size) <= 2],
                [item for item in contributors if item.primary_exchange == exchange],
                contributors,
            ]
            resolved = None
            for candidates in candidate_groups:
                resolved = peer_median(candidates)
                if resolved is not None:
                    break
            result[key] = resolved
    return result


def _completed_calendar_months(start: dt.date, end: dt.date) -> int:
    months = (end.year - start.year) * 12 + end.month - start.month
    start_eom = start.day == calendar.monthrange(start.year, start.month)[1]
    end_eom = end.day == calendar.monthrange(end.year, end.month)[1]
    if end.day < start.day and not (start_eom and end_eom):
        months -= 1
    return max(0, months)


def months_since_first_tradable(data: CrspFeatureInput) -> float | None:
    """Observed-session age; lifecycle begin/end fields are intentionally unused."""

    cache = _prepared_cache(data)
    if cache is not None:
        return cache.months_since_first_tradable()

    aliases = {
        "security_type": ("security_type", "source_security_type"),
        "security_subtype": ("security_subtype", "source_security_subtype"),
        "share_type": ("share_type", "source_share_type"),
        "conditional_type": ("conditional_type",),
        "trading_status": (
            "trading_status",
            "trading_status_flag",
            "source_trading_status_flag",
        ),
        "exchange": ("primary_exchange", "source_primary_exchange"),
    }

    def text(row: Mapping[str, Any], concept: str) -> str | None:
        for field in aliases[concept]:
            value = row.get(field)
            if value is not None:
                normalized = str(value).strip().upper()
                if normalized:
                    return normalized
        return None

    dates = []
    rows = data.listing_history_rows or data.market_rows
    for row in rows:
        date = _date(row.get("calendar_date"))
        if date is None or date > data.signal_date:
            continue
        if (
            text(row, "security_type") == "EQTY"
            and text(row, "security_subtype") == "COM"
            and text(row, "share_type") in {"NS", "AD"}
            and text(row, "conditional_type") == "RW"
            and text(row, "trading_status") == "A"
            and text(row, "exchange") in {"N", "A", "Q"}
        ):
            dates.append(date)
    if not dates:
        return None
    return float(_completed_calendar_months(min(dates), data.signal_date))


def build_size_liquidity_features(
    data: CrspFeatureInput,
    *,
    resolved_spread: float | None = None,
) -> dict[str, float | None]:
    """Build the frozen 13 size/liquidity/friction features."""

    values = _empty(SIZE_LIQUIDITY_FEATURE_NAMES)
    context = _session_context(data)
    if context is None:
        return values
    cache = _prepared_cache(data)
    market = (
        cache.market_index
        if cache is not None
        else _index_rows(data.market_rows, "calendar_date")
    )
    prepared = _prepare_return_legs(data)
    signal_row = market.get(data.signal_date)
    close = None if signal_row is None else _number(signal_row.get("close_usd_per_share"))
    market_cap = None if signal_row is None else _number(signal_row.get("market_cap_usd"))
    if close is not None and close != 0.0:
        values["log_absolute_price"] = math.log(abs(close))
    if market_cap is not None and market_cap > 0.0:
        values["log_market_cap"] = math.log(market_cap)

    dollar_volume: dict[int, list[float] | None] = {}
    for window in (20, 60):
        dates = _window_dates(context, -window + 1, 0)
        volumes = _market_values(
            market,
            dates,
            "screening_dollar_volume_usd",
            nonnegative=True,
        )
        dollar_volume[window] = volumes
        values[f"median_dollar_volume_{window}d"] = (
            None if volumes is None else ordinary_median(volumes)
        )

        share_volume = _market_values(
            market, dates, "volume_shares", nonnegative=True
        )
        shares = _shares_on_dates(data, dates)
        if share_volume is not None and shares is not None:
            turnover = [volume / outstanding for volume, outstanding in zip(share_volume, shares)]
            mean_turnover = float(np.mean(np.asarray(turnover, dtype=np.float64)))
            values[f"share_turnover_{window}d"] = _finite_result(mean_turnover)

        one_session = _exact_return_values(
            data, dates, one_session_only=True, prepared=prepared
        )
        if one_session is not None and volumes is not None and all(value > 0.0 for value in volumes):
            amihud = float(
                np.mean(
                    np.asarray(
                        [abs(ret) / volume for ret, volume in zip(one_session, volumes)],
                        dtype=np.float64,
                    )
                )
            )
            values[f"amihud_illiquidity_{window}d"] = _finite_result(amihud)

    values["median_estimated_full_spread_20d"] = (
        security_level_spread(data) if resolved_spread is None else resolved_spread
    )

    dates_60 = _window_dates(context, -59, 0)
    one_session_60 = _exact_return_values(
        data, dates_60, one_session_only=True, prepared=prepared
    )
    volumes_60 = _market_values(
        market, dates_60, "volume_shares", nonnegative=True
    )
    if one_session_60 is not None:
        values["zero_return_share_60d"] = sum(value == 0.0 for value in one_session_60) / 60.0
    if volumes_60 is not None:
        values["zero_volume_share_60d"] = sum(value == 0.0 for value in volumes_60) / 60.0

    current = _market_values(
        market,
        _window_dates(context, -19, 0),
        "screening_dollar_volume_usd",
        nonnegative=True,
    )
    prior = _market_values(
        market,
        _window_dates(context, -271, -20),
        "screening_dollar_volume_usd",
        nonnegative=True,
    )
    current_median = None if current is None else ordinary_median(current)
    prior_median = None if prior is None else ordinary_median(prior)
    if current_median is not None and prior_median is not None and prior_median > 0.0:
        values["dollar_volume_shock_20d_vs_prior_252d"] = current_median / prior_median - 1.0
    values["months_since_first_tradable"] = months_since_first_tradable(data)
    return {name: _finite_result(value) for name, value in values.items()}


def _clip_beta_slopes(
    group: Sequence[CrspFeatureInput],
    values: Mapping[tuple[dt.date, int], dict[str, float | None]],
) -> None:
    for feature in BETA_FEATURE_NAMES:
        population = [
            values[(data.signal_date, data.permno)][feature]
            for data in group
            if data.long_eligible
            and values[(data.signal_date, data.permno)][feature] is not None
        ]
        lower = type7_quantile(population, 0.01)
        upper = type7_quantile(population, 0.99)
        for data in group:
            row = values[(data.signal_date, data.permno)]
            slope = row[feature]
            row[feature] = (
                None
                if slope is None or lower is None or upper is None
                else min(upper, max(lower, slope))
            )


def _build_crsp_feature_rows_unclipped(
    items: Sequence[CrspFeatureInput],
) -> tuple[
    dict[tuple[dt.date, int], dict[str, float | None]],
    dict[dt.date, list[CrspFeatureInput]],
]:
    keys = [(item.signal_date, item.permno) for item in items]
    if len(keys) != len(set(keys)):
        raise CrspFeatureError("Duplicate signal_date/permno input")
    spread = resolve_spread_hierarchy(items)
    result: dict[tuple[dt.date, int], dict[str, float | None]] = {}
    by_signal: dict[dt.date, list[CrspFeatureInput]] = defaultdict(list)
    for data in items:
        key = (data.signal_date, data.permno)
        row = {
            **build_price_trend_features(data),
            **build_risk_distribution_features(data),
            **build_size_liquidity_features(data, resolved_spread=spread.get(key)),
        }
        result[key] = row
        by_signal[data.signal_date].append(data)
    return result, by_signal


def build_crsp_feature_rows(
    inputs: Iterable[CrspFeatureInput],
) -> dict[tuple[dt.date, int], dict[str, float | None]]:
    """Build all 53 CRSP features, including cross-sectional finalizations."""

    items = list(inputs)
    result, by_signal = _build_crsp_feature_rows_unclipped(items)
    for group in by_signal.values():
        _clip_beta_slopes(group, result)
    return result


def _prepared_return_contributors(
    data: CrspFeatureInput,
) -> tuple[tuple[tuple[int, ...], Mapping[str, Any]], ...]:
    """Return the exact canonical legs eligible at this after-close signal."""

    cache = _prepared_cache(data)
    if cache is not None:
        return cache.prepared_return_contributors
    context = _session_context(data)
    if context is None:
        return ()
    sessions, _ = context
    result: list[tuple[tuple[int, ...], Mapping[str, Any]]] = []
    for row in data.return_rows:
        observation = _date(row.get("calendar_date"))
        span_end = _date(row.get("return_span_end_date"))
        availability = _date(row.get("outcome_availability_date"))
        value = _number(row.get("authoritative_total_return"))
        coverage = _return_leg_coverage(row, sessions)
        if (
            row.get("return_history_usable") is True
            and observation is not None
            and observation <= data.signal_date
            and span_end == observation
            and availability is not None
            and availability <= data.signal_date
            and value is not None
            and value >= -1.0
            and coverage is not None
        ):
            result.append((coverage, row))
    return tuple(result)


def _exact_return_contributor_rows(
    data: CrspFeatureInput,
    dates: Sequence[dt.date] | None,
    *,
    one_session_only: bool,
) -> tuple[Mapping[str, Any], ...] | None:
    cache = _prepared_cache(data)
    if cache is not None:
        return cache.exact_return_contributor_rows(
            dates,
            one_session_only=one_session_only,
        )
    context = _session_context(data)
    if context is None or dates is None:
        return None
    sessions, _ = context
    position = {date: index for index, date in enumerate(sessions)}
    try:
        target = {position[date] for date in dates}
    except KeyError:
        return None
    selected: list[tuple[int, tuple[int, ...], Mapping[str, Any]]] = []
    for coverage, row in _prepared_return_contributors(data):
        if one_session_only and len(coverage) != 1:
            continue
        if not target.intersection(coverage):
            continue
        if not set(coverage).issubset(target):
            return None
        selected.append((coverage[0], coverage, row))
    counts = {index: 0 for index in target}
    for _, coverage, _ in selected:
        for index in coverage:
            if index in counts:
                counts[index] += 1
    if not counts or any(count != 1 for count in counts.values()):
        return None
    selected.sort(key=lambda item: item[0])
    return tuple(item[2] for item in selected)


def _max_contributor_dates(
    contributors: Iterable[tuple[dt.date | None, dt.date | None]],
) -> tuple[dt.date | None, dt.date | None]:
    pairs = list(contributors)
    observations = [value for value, _ in pairs if value is not None]
    availability = [value for _, value in pairs if value is not None]
    return (
        max(observations) if observations else None,
        max(availability) if availability else None,
    )


def _return_lineage_dates(
    rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[dt.date | None, dt.date | None]:
    if rows is None:
        return None, None
    return _max_contributor_dates(
        (
            _date(row.get("calendar_date")),
            _date(row.get("outcome_availability_date")),
        )
        for row in rows
    )


def _market_lineage_dates(
    data: CrspFeatureInput, dates: Sequence[dt.date] | None
) -> tuple[dt.date | None, dt.date | None]:
    if dates is None:
        return None, None
    cache = _prepared_cache(data)
    market = (
        cache.market_index
        if cache is not None
        else _index_rows(data.market_rows, "calendar_date")
    )
    used = [market.get(day) for day in dates]
    if any(row is None for row in used):
        return None, None
    # Canonical CRSP market fields are known after the dated close.
    latest = max(dates)
    return latest, latest


def _beta_lineage_dates(
    data: CrspFeatureInput,
) -> tuple[dt.date | None, dt.date | None]:
    cache = _prepared_cache(data)
    if cache is not None:
        return cache.beta_lineage_dates()
    context = _session_context(data)
    if context is None:
        return None, None
    sessions, signal_index = context
    start_index = max(0, signal_index - 251)

    one_session: dict[dt.date, Mapping[str, Any]] = {}
    ambiguous: set[dt.date] = set()
    for coverage, row in _prepared_return_contributors(data):
        if len(coverage) != 1:
            continue
        day = sessions[coverage[0]]
        if day in one_session:
            ambiguous.add(day)
        else:
            one_session[day] = row
    for day in ambiguous:
        one_session.pop(day, None)

    cache = _prepared_cache(data)
    factors = (
        cache.factor_index
        if cache is not None
        else _index_rows(data.factor_rows, "observation_date")
    )
    contributors: list[tuple[dt.date | None, dt.date | None]] = []
    for index in range(start_index, signal_index + 1):
        day = sessions[index]
        return_row = one_session.get(day)
        factor_row = factors.get(day)
        if return_row is None or factor_row is None:
            continue
        factor_availability = _date(factor_row.get("public_availability_date"))
        factor_values = [_number(factor_row.get(name)) for name in FACTOR_NAMES]
        if (
            factor_availability is None
            or factor_availability > data.signal_date
            or _number(factor_row.get("rf")) is None
            or any(value is None for value in factor_values)
        ):
            continue
        contributors.extend(
            (
                (
                    _date(return_row.get("calendar_date")),
                    _date(return_row.get("outcome_availability_date")),
                ),
                (_date(factor_row.get("observation_date")), factor_availability),
            )
        )
    return _max_contributor_dates(contributors)


def _security_spread_lineage_dates(
    data: CrspFeatureInput,
) -> tuple[dt.date | None, dt.date | None]:
    context = _session_context(data)
    dates = _window_dates(context, -19, 0)
    if dates is None:
        return None, None
    cache = _prepared_cache(data)
    market = (
        cache.market_index
        if cache is not None
        else _index_rows(data.market_rows, "calendar_date")
    )
    rows = [market.get(day) for day in dates]
    quotes = [
        row
        for row in rows
        if row is not None and _quoted_full_spread(row) is not None
    ]
    if len(quotes) >= 10:
        quote_dates = [_date(row.get("calendar_date")) for row in quotes]
        latest = max(day for day in quote_dates if day is not None)
        return latest, latest

    used_dates: set[dt.date] = set()
    pair_count = 0
    for prior, following in zip(rows, rows[1:]):
        if prior is None or following is None:
            continue
        inputs = (
            _number(prior.get("high_usd_per_share")),
            _number(prior.get("low_usd_per_share")),
            _number(prior.get("close_usd_per_share")),
            _number(following.get("high_usd_per_share")),
            _number(following.get("low_usd_per_share")),
        )
        if any(value is None for value in inputs):
            continue
        try:
            estimate = corwin_schultz_pair(*[float(value) for value in inputs])
        except (OverflowError, ValueError):
            estimate = None
        if estimate is None:
            continue
        pair_count += 1
        for row in (prior, following):
            day = _date(row.get("calendar_date"))
            if day is not None:
                used_dates.add(day)
    if pair_count < 10 or not used_dates:
        return None, None
    latest = max(used_dates)
    return latest, latest


_RETURN_WINDOWS: dict[str, tuple[int, int, bool]] = {
    "return_total_5d": (-4, 0, False),
    "return_total_63d": (-62, 0, False),
    "return_total_126d": (-125, 0, False),
    "return_total_252d": (-251, 0, False),
    "momentum_6_1": (-126, -22, False),
    "momentum_12_1": (-252, -22, False),
    "momentum_24_12": (-504, -253, False),
    "reversal_21d": (-20, 0, False),
    "max_drawdown_252d": (-251, 0, False),
    "realized_volatility_21d": (-20, 0, True),
    "realized_volatility_63d": (-62, 0, True),
    "realized_volatility_126d": (-125, 0, True),
    "realized_volatility_252d": (-251, 0, True),
    "downside_semivolatility_63d": (-62, 0, True),
    "downside_semivolatility_252d": (-251, 0, True),
    "return_skewness_252d": (-251, 0, True),
    "return_excess_kurtosis_252d": (-251, 0, True),
    "maximum_daily_return_21d": (-20, 0, True),
    "minimum_daily_return_21d": (-20, 0, True),
    "maximum_daily_return_63d": (-62, 0, True),
    "minimum_daily_return_63d": (-62, 0, True),
    "volatility_63d_vs_trailing_median_252d": (-314, 0, True),
}

_ADJUSTED_PRICE_PATH_WINDOWS: dict[str, tuple[int, int]] = {
    "price_to_ma_21d": (-20, 0),
    "price_to_ma_63d": (-62, 0),
    "price_to_ma_126d": (-125, 0),
    "price_to_ma_252d": (-251, 0),
    "distance_to_high_252d": (-251, 0),
    "log_price_trend_slope_63d": (-62, 0),
    "log_price_trend_r2_63d": (-62, 0),
    "log_price_trend_slope_126d": (-125, 0),
    "log_price_trend_r2_126d": (-125, 0),
    "log_price_trend_slope_252d": (-251, 0),
    "log_price_trend_r2_252d": (-251, 0),
}

_MARKET_WINDOWS: dict[str, tuple[int, int]] = {
    "median_dollar_volume_20d": (-19, 0),
    "median_dollar_volume_60d": (-59, 0),
    "share_turnover_20d": (-19, 0),
    "share_turnover_60d": (-59, 0),
    "amihud_illiquidity_20d": (-19, 0),
    "amihud_illiquidity_60d": (-59, 0),
    "zero_volume_share_60d": (-59, 0),
    "dollar_volume_shock_20d_vs_prior_252d": (-271, 0),
}


def _missing_status_for_crsp(data: CrspFeatureInput, name: str) -> str:
    if _session_context(data) is None:
        return "MISSING_INSUFFICIENT_HISTORY"
    if name == "median_estimated_full_spread_20d" and not data.long_eligible:
        return "MISSING_NOT_COVERED"
    if name == "months_since_first_tradable":
        if not data.listing_history_rows and not data.market_rows:
            return "MISSING_SOURCE_UNAVAILABLE"
        return "MISSING_INSUFFICIENT_HISTORY"
    return_dependent = (
        name in _RETURN_WINDOWS
        or name in _ADJUSTED_PRICE_PATH_WINDOWS
        or name in BETA_FEATURE_NAMES
        or name == "beta_label_idiosyncratic_volatility"
        or name.startswith("amihud_illiquidity_")
        or name == "zero_return_share_60d"
    )
    if return_dependent:
        if not data.return_rows:
            return "MISSING_SOURCE_UNAVAILABLE"
        if name in BETA_FEATURE_NAMES or name == "beta_label_idiosyncratic_volatility":
            if not data.factor_rows:
                return "MISSING_SOURCE_UNAVAILABLE"
        if name in BETA_FEATURE_NAMES or name == "beta_label_idiosyncratic_volatility":
            start, end = -125, 0
        elif name.startswith("amihud_illiquidity_"):
            start, end = (-19, 0) if name.endswith("20d") else (-59, 0)
        elif name == "zero_return_share_60d":
            start, end = -59, 0
        elif name in _ADJUSTED_PRICE_PATH_WINDOWS:
            start, end = _ADJUSTED_PRICE_PATH_WINDOWS[name]
        else:
            start, end, _ = _RETURN_WINDOWS[name]
        if _window_dates(_session_context(data), start, end) is None:
            return "MISSING_INSUFFICIENT_HISTORY"
    if name in _MARKET_WINDOWS or name in {
        "log_absolute_price",
        "log_market_cap",
        "median_estimated_full_spread_20d",
        "amihud_illiquidity_20d",
        "amihud_illiquidity_60d",
    }:
        if not data.market_rows:
            return "MISSING_SOURCE_UNAVAILABLE"
        if name in _MARKET_WINDOWS:
            start, end = _MARKET_WINDOWS[name]
            if _window_dates(_session_context(data), start, end) is None:
                return "MISSING_INSUFFICIENT_HISTORY"
        elif name == "median_estimated_full_spread_20d":
            if _window_dates(_session_context(data), -19, 0) is None:
                return "MISSING_INSUFFICIENT_HISTORY"
    if name.startswith("share_turnover_") and not data.shares_rows:
        return "MISSING_SOURCE_UNAVAILABLE"
    return "MISSING_INVALID_INPUT"


def _feature_lineage_dates(
    data: CrspFeatureInput,
    name: str,
    value: float | None,
) -> tuple[dt.date | None, dt.date | None]:
    if value is None:
        return None, None
    context = _session_context(data)
    if name in _RETURN_WINDOWS:
        start, end, one_session = _RETURN_WINDOWS[name]
        rows = _exact_return_contributor_rows(
            data,
            _window_dates(context, start, end),
            one_session_only=one_session,
        )
        return _return_lineage_dates(rows)
    if name in _ADJUSTED_PRICE_PATH_WINDOWS:
        start, end = _ADJUSTED_PRICE_PATH_WINDOWS[name]
        dates = _window_dates(context, start, end)
        selected = _exact_price_return_rows(
            data,
            None if dates is None else dates[1:],
        )
        rows = None if selected is None else selected[0]
        return _return_lineage_dates(rows)
    if name in BETA_FEATURE_NAMES or name == "beta_label_idiosyncratic_volatility":
        return _beta_lineage_dates(data)
    if name in _MARKET_WINDOWS:
        start, end = _MARKET_WINDOWS[name]
        market_dates = _market_lineage_dates(data, _window_dates(context, start, end))
        if name.startswith("amihud_"):
            window = 20 if name.endswith("20d") else 60
            return_dates = _return_lineage_dates(
                _exact_return_contributor_rows(
                    data,
                    _window_dates(context, -window + 1, 0),
                    one_session_only=True,
                )
            )
            return _max_contributor_dates((market_dates, return_dates))
        return market_dates
    if name in {"log_absolute_price", "log_market_cap"}:
        return _market_lineage_dates(data, (data.signal_date,))
    if name == "median_estimated_full_spread_20d":
        own_spread = security_level_spread(data)
        if own_spread is not None:
            return _security_spread_lineage_dates(data)
        # Peer-imputed values use the target's signal cap and complete 20-day
        # ADV assignment; the target signal close is therefore a contributor.
        return _market_lineage_dates(data, _window_dates(context, -19, 0))
    if name == "zero_return_share_60d":
        return _return_lineage_dates(
            _exact_return_contributor_rows(
                data,
                _window_dates(context, -59, 0),
                one_session_only=True,
            )
        )
    if name == "months_since_first_tradable":
        cache = _prepared_cache(data)
        if cache is not None:
            return cache.listing_lineage_dates()
        rows = data.listing_history_rows or data.market_rows
        eligible_dates: list[dt.date] = []
        for row in rows:
            day = _date(row.get("calendar_date"))
            if day is None or day > data.signal_date:
                continue
            single = CrspFeatureInput(
                permno=data.permno,
                signal_date=day,
                sessions=(day,),
                market_rows=(row,),
                return_rows=(),
                listing_history_rows=(row,),
            )
            if months_since_first_tradable(single) == 0.0:
                eligible_dates.append(day)
        if not eligible_dates:
            return None, None
        latest = max(eligible_dates)
        return latest, latest
    return None, None


def _type7_bound_contributors(
    group: Sequence[CrspFeatureInput],
    raw_values: Mapping[tuple[dt.date, int], Mapping[str, float | None]],
    feature: str,
    quantile: float,
) -> tuple[CrspFeatureInput, ...]:
    population = [
        data
        for data in group
        if data.long_eligible
        and raw_values[(data.signal_date, data.permno)][feature] is not None
    ]
    population.sort(
        key=lambda data: (
            float(raw_values[(data.signal_date, data.permno)][feature]),
            data.permno,
        )
    )
    if not population:
        return ()
    position = (len(population) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    return tuple(
        population[index]
        for index in sorted({lower_index, upper_index})
    )


def _binding_beta_clip_lineage(
    group: Sequence[CrspFeatureInput],
    raw_values: Mapping[tuple[dt.date, int], Mapping[str, float | None]],
    final_values: Mapping[tuple[dt.date, int], Mapping[str, float | None]],
) -> dict[tuple[tuple[dt.date, int], str], tuple[dt.date | None, dt.date | None]]:
    result: dict[
        tuple[tuple[dt.date, int], str],
        tuple[dt.date | None, dt.date | None],
    ] = {}
    for feature in BETA_FEATURE_NAMES:
        population = [
            raw_values[(data.signal_date, data.permno)][feature]
            for data in group
            if data.long_eligible
            and raw_values[(data.signal_date, data.permno)][feature] is not None
        ]
        lower = type7_quantile(population, 0.01)
        upper = type7_quantile(population, 0.99)
        if lower is None or upper is None:
            continue
        for data in group:
            key = (data.signal_date, data.permno)
            raw = raw_values[key][feature]
            final = final_values[key][feature]
            if raw is None or final is None:
                continue
            if raw < lower:
                peers = _type7_bound_contributors(
                    group, raw_values, feature, 0.01
                )
            elif raw > upper:
                peers = _type7_bound_contributors(
                    group, raw_values, feature, 0.99
                )
            else:
                continue
            result[(key, feature)] = _max_contributor_dates(
                _beta_lineage_dates(peer) for peer in peers
            )
    return result


def build_crsp_feature_rows_traced(
    inputs: Iterable[CrspFeatureInput],
) -> dict[tuple[dt.date, int], FeatureFamilyResult]:
    """Build all 53 CRSP values with publication-ready per-feature lineage."""

    items = list(inputs)
    raw_values, by_signal = _build_crsp_feature_rows_unclipped(items)
    values_by_key = {
        key: dict(values) for key, values in raw_values.items()
    }
    clip_lineage: dict[
        tuple[tuple[dt.date, int], str],
        tuple[dt.date | None, dt.date | None],
    ] = {}
    for group in by_signal.values():
        _clip_beta_slopes(group, values_by_key)
        clip_lineage.update(
            _binding_beta_clip_lineage(group, raw_values, values_by_key)
        )
    item_by_key = {(item.signal_date, item.permno): item for item in items}
    traced: dict[tuple[dt.date, int], FeatureFamilyResult] = {}
    for key, values in values_by_key.items():
        data = item_by_key[key]
        ordered_values = {name: values[name] for name in CRSP_FEATURE_NAMES}
        lineage: dict[str, FeatureLineage] = {}
        for name, value in ordered_values.items():
            observation, availability = _feature_lineage_dates(data, name, value)
            bound_dates = clip_lineage.get((key, name))
            if bound_dates is not None:
                observation, availability = _max_contributor_dates(
                    ((observation, availability), bound_dates)
                )
            lineage[name] = make_lineage(
                value,
                observation,
                availability,
                missing_status=_missing_status_for_crsp(data, name),
            )
        traced[key] = FeatureFamilyResult(ordered_values, lineage)
    return traced


def build_crsp_unfinalized_feature_result(
    data: CrspFeatureInput,
) -> FeatureFamilyResult:
    """Build one security-signal row before peer spread/beta finalization."""

    values = {
        **build_price_trend_features(data),
        **build_risk_distribution_features(data),
        **build_size_liquidity_features(
            data, resolved_spread=security_level_spread(data)
        ),
    }
    ordered_values = {name: values[name] for name in CRSP_FEATURE_NAMES}
    lineage: dict[str, FeatureLineage] = {}
    for name, value in ordered_values.items():
        observation, availability = _feature_lineage_dates(data, name, value)
        lineage[name] = make_lineage(
            value,
            observation,
            availability,
            missing_status=_missing_status_for_crsp(data, name),
        )
    return FeatureFamilyResult(values=ordered_values, lineage=lineage)


assert len(PRICE_TREND_FEATURE_NAMES) == 20
assert len(RISK_DISTRIBUTION_FEATURE_NAMES) == 20
assert len(SIZE_LIQUIDITY_FEATURE_NAMES) == 13
assert len(CRSP_FEATURE_NAMES) == 53
