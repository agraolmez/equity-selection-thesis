"""Cached per-security execution engine for the frozen CRSP feature formulas.

The reference builders in :mod:`portfolio_bot.features.crsp_features` remain
the formula authority.  This module only prepares immutable indexes and exact
return-leg selections once per security/signal so those builders do not rescan
the same multi-year history dozens of times.
"""

from __future__ import annotations

import datetime as dt
import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from portfolio_bot.features.crsp_features import (
    CrspFeatureError,
    CrspFeatureInput,
    _completed_calendar_months,
    _date,
    _index_rows,
    _number,
    _return_leg_coverage,
    build_crsp_unfinalized_feature_result,
)
from portfolio_bot.features.feature_lineage import FeatureFamilyResult
from portfolio_bot.features.feature_math import ordinary_median


class FastCrspFeatureError(RuntimeError):
    """Raised when a security history cannot form an exact cached context."""


@dataclass(frozen=True)
class _ReturnLeg:
    ordinal: int
    coverage: tuple[int, ...]
    value: float
    row: Mapping[str, Any]
    observation_date: dt.date
    availability_date: dt.date


class _ProjectedOneSessionMap:
    """Map-like view over immutable immediate legs plus a signal overlay."""

    def __init__(self, cache: _SignalCache, attribute: str) -> None:
        self._cache = cache
        self._attribute = attribute

    def get(self, day: dt.date, default: Any = None) -> Any:
        leg = self._cache._one_session_leg(day)
        if leg is None:
            return default
        return leg if self._attribute == "leg" else getattr(
            leg, self._attribute
        )


def _tradable_date(row: Mapping[str, Any]) -> dt.date | None:
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

    def text(concept: str) -> str | None:
        for field in aliases[concept]:
            value = row.get(field)
            if value is not None:
                normalized = str(value).strip().upper()
                if normalized:
                    return normalized
        return None

    day = _date(row.get("calendar_date"))
    if (
        day is not None
        and text("security_type") == "EQTY"
        and text("security_subtype") == "COM"
        and text("share_type") in {"NS", "AD"}
        and text("conditional_type") == "RW"
        and text("trading_status") == "A"
        and text("exchange") in {"N", "A", "Q"}
    ):
        return day
    return None


class _SignalCache:
    """Duck-typed cache consumed by the unchanged reference formula helpers."""

    def __init__(
        self,
        engine: CachedCrspSecurityEngine,
        *,
        permno: int,
        signal_date: dt.date,
    ) -> None:
        self._engine = engine
        self._permno = int(permno)
        self._signal_date = signal_date
        signal_index = engine.session_position.get(signal_date)
        self.session_context = (
            None
            if signal_index is None
            else (engine.sessions, signal_index)
        )
        self._signal_index = signal_index
        self.market_index = engine.market_index
        self.factor_index = engine.factor_index

        direct = engine.direct_return_legs_by_position
        if direct is None:
            eligible = tuple(
                leg
                for leg in engine.return_legs
                if leg.observation_date <= signal_date
                and leg.availability_date <= signal_date
            )
            self._eligible_legs = eligible
            self.prepared_return_legs = tuple(
                (leg.coverage, leg.value) for leg in eligible
            )
            self.prepared_return_contributors = tuple(
                (leg.coverage, leg.row) for leg in eligible
            )
            by_index: dict[int, list[int]] = {}
            one_value: dict[dt.date, float] = {}
            one_row: dict[dt.date, Mapping[str, Any]] = {}
            one_leg: dict[dt.date, _ReturnLeg] = {}
            ambiguous: set[dt.date] = set()
            for leg_id, leg in enumerate(eligible):
                for index in leg.coverage:
                    by_index.setdefault(index, []).append(leg_id)
                if len(leg.coverage) == 1:
                    day = engine.sessions[leg.coverage[0]]
                    if day in one_value:
                        ambiguous.add(day)
                    else:
                        one_value[day] = leg.value
                        one_row[day] = leg.row
                        one_leg[day] = leg
            for day in ambiguous:
                one_value.pop(day, None)
                one_row.pop(day, None)
                one_leg.pop(day, None)
            self._leg_ids_by_index = {
                index: tuple(ids) for index, ids in by_index.items()
            }
            self.valid_one_session_return_map = one_value
            self._valid_one_session_row_map = one_row
            self._valid_one_session_leg_map = one_leg
            self._exception_legs_by_index: dict[
                int, tuple[_ReturnLeg, ...]
            ] = {}
            self._one_session_overlay: dict[
                int, tuple[_ReturnLeg, ...]
            ] = {}
        else:
            eligible_exceptions = tuple(
                leg
                for leg in engine.exception_return_legs
                if leg.observation_date <= signal_date
                and leg.availability_date <= signal_date
            )
            self._eligible_legs = eligible_exceptions
            # All formula helpers delegate selections back to this cache in
            # overlay mode; materializing the multi-year immediate prefix here
            # would defeat the shared-index optimization.
            self.prepared_return_legs = ()
            self.prepared_return_contributors = ()
            exception_by_index: dict[int, list[_ReturnLeg]] = {}
            one_session_overlay: dict[int, list[_ReturnLeg]] = {}
            for leg in eligible_exceptions:
                for index in leg.coverage:
                    exception_by_index.setdefault(index, []).append(leg)
                if len(leg.coverage) == 1:
                    one_session_overlay.setdefault(
                        leg.coverage[0], []
                    ).append(leg)
            self._exception_legs_by_index = {
                index: tuple(items)
                for index, items in exception_by_index.items()
            }
            self._one_session_overlay = {
                index: tuple(items)
                for index, items in one_session_overlay.items()
            }
            self._leg_ids_by_index = {}
            self.valid_one_session_return_map = _ProjectedOneSessionMap(
                self, "value"
            )
            self._valid_one_session_row_map = _ProjectedOneSessionMap(
                self, "row"
            )
            self._valid_one_session_leg_map = _ProjectedOneSessionMap(
                self, "leg"
            )
        self._direct_legs_by_position = direct
        self._selection_cache: dict[
            tuple[tuple[int, ...], bool],
            tuple[_ReturnLeg, ...] | None,
        ] = {}

    def matches(self, data: CrspFeatureInput) -> bool:
        """Fail closed if a cache is accidentally attached to another input."""

        return (
            data.permno == self._permno
            and data.signal_date == self._signal_date
            and data.sessions is self._engine.sessions
            and data.market_rows is self._engine.market_rows
            and data.return_rows is self._engine.return_rows
            and data.factor_rows is self._engine.factor_rows
            and data.shares_rows is self._engine.shares_rows
            and data.listing_history_rows is self._engine.listing_history_rows
        )

    def _positions(
        self,
        dates: Sequence[dt.date] | None,
    ) -> tuple[int, ...] | None:
        if dates is None or self.session_context is None:
            return None
        try:
            return tuple(self._engine.session_position[day] for day in dates)
        except KeyError:
            return None

    def _selection(
        self,
        dates: Sequence[dt.date] | None,
        *,
        one_session_only: bool,
    ) -> tuple[_ReturnLeg, ...] | None:
        positions = self._positions(dates)
        if not positions:
            return None
        key = (positions, one_session_only)
        if key in self._selection_cache:
            return self._selection_cache[key]

        if self._direct_legs_by_position is not None:
            if (
                self._signal_index is None
                or any(position > self._signal_index for position in positions)
            ):
                self._selection_cache[key] = None
                return None
            if one_session_only:
                direct = tuple(
                    self._one_session_leg_by_position(position)
                    for position in positions
                )
                if any(leg is None for leg in direct):
                    self._selection_cache[key] = None
                    return None
                selected = tuple(
                    leg for leg in direct if leg is not None
                )
                self._selection_cache[key] = selected
                return selected

            candidates: dict[int, _ReturnLeg] = {}
            for position in positions:
                immediate = self._direct_legs_by_position[position]
                if immediate is not None:
                    candidates[immediate.ordinal] = immediate
                for leg in self._exception_legs_by_index.get(
                    position, ()
                ):
                    candidates[leg.ordinal] = leg
            selected = tuple(
                candidates[ordinal]
                for ordinal in sorted(
                    candidates,
                    key=lambda item: (
                        candidates[item].coverage[0],
                        item,
                    ),
                )
            )
            target = set(positions)
            if any(
                not set(leg.coverage).issubset(target)
                for leg in selected
            ):
                self._selection_cache[key] = None
                return None
            counts = {position: 0 for position in target}
            for leg in selected:
                for position in leg.coverage:
                    if position in counts:
                        counts[position] += 1
            if any(count != 1 for count in counts.values()):
                self._selection_cache[key] = None
                return None
            self._selection_cache[key] = selected
            return selected

        if one_session_only:
            rows: list[_ReturnLeg] = []
            for day in dates or ():
                leg = self._valid_one_session_leg_map.get(day)
                if leg is None:
                    self._selection_cache[key] = None
                    return None
                rows.append(leg)
            selected = tuple(rows)
            self._selection_cache[key] = selected
            return selected

        target = set(positions)
        candidate_ids: set[int] = set()
        for position in positions:
            candidate_ids.update(self._leg_ids_by_index.get(position, ()))
        selected = tuple(
            self._eligible_legs[leg_id]
            for leg_id in sorted(
                candidate_ids,
                key=lambda item: (
                    self._eligible_legs[item].coverage[0],
                    item,
                ),
            )
        )
        if any(not set(leg.coverage).issubset(target) for leg in selected):
            self._selection_cache[key] = None
            return None
        counts = {position: 0 for position in target}
        for leg in selected:
            for position in leg.coverage:
                if position in counts:
                    counts[position] += 1
        if any(count != 1 for count in counts.values()):
            self._selection_cache[key] = None
            return None
        self._selection_cache[key] = selected
        return selected

    def _one_session_leg_by_position(
        self,
        position: int,
    ) -> _ReturnLeg | None:
        candidates: list[_ReturnLeg] = []
        if self._direct_legs_by_position is not None:
            immediate = self._direct_legs_by_position[position]
            if immediate is not None:
                candidates.append(immediate)
        candidates.extend(self._one_session_overlay.get(position, ()))
        return candidates[0] if len(candidates) == 1 else None

    def _one_session_leg(self, day: dt.date) -> _ReturnLeg | None:
        position = self._engine.session_position.get(day)
        if (
            position is None
            or self._signal_index is None
            or position > self._signal_index
        ):
            return None
        return self._one_session_leg_by_position(position)

    def exact_return_values(
        self,
        dates: Sequence[dt.date] | None,
        *,
        one_session_only: bool,
    ) -> list[float] | None:
        selected = self._selection(
            dates,
            one_session_only=one_session_only,
        )
        return None if selected is None else [leg.value for leg in selected]

    def exact_return_contributor_rows(
        self,
        dates: Sequence[dt.date] | None,
        *,
        one_session_only: bool,
    ) -> tuple[Mapping[str, Any], ...] | None:
        selected = self._selection(
            dates,
            one_session_only=one_session_only,
        )
        return None if selected is None else tuple(leg.row for leg in selected)

    def beta_lineage_dates(
        self,
    ) -> tuple[dt.date | None, dt.date | None]:
        cached = getattr(self, "_beta_lineage_cache", None)
        if cached is not None:
            return cached
        if self.session_context is None:
            result = (None, None)
            self._beta_lineage_cache = result
            return result
        sessions, signal_index = self.session_context
        observations: list[dt.date] = []
        availability_dates: list[dt.date] = []
        for index in range(max(0, signal_index - 251), signal_index + 1):
            day = sessions[index]
            return_row = self._valid_one_session_row_map.get(day)
            factor_row = self.factor_index.get(day)
            if return_row is None or factor_row is None:
                continue
            factor_availability = _date(
                factor_row.get("public_availability_date")
            )
            factor_values = [
                _number(factor_row.get(name))
                for name in ("mktrf", "smb", "hml", "rmw", "cma", "umd")
            ]
            if (
                factor_availability is None
                or factor_availability > self._signal_date
                or _number(factor_row.get("rf")) is None
                or any(value is None for value in factor_values)
            ):
                continue
            return_observation = _date(return_row.get("calendar_date"))
            return_availability = _date(
                return_row.get("outcome_availability_date")
            )
            factor_observation = _date(
                factor_row.get("observation_date")
            )
            if return_observation is not None:
                observations.append(return_observation)
            if factor_observation is not None:
                observations.append(factor_observation)
            if return_availability is not None:
                availability_dates.append(return_availability)
            availability_dates.append(factor_availability)
        result = (
            max(observations) if observations else None,
            max(availability_dates) if availability_dates else None,
        )
        self._beta_lineage_cache = result
        return result

    def volatility_63d_vs_trailing_median_252d(
        self,
        current_volatility: float | None,
    ) -> float | None:
        """Evaluate the 252 overlapping prior windows without Python rescans.

        The reference formula requests 252 consecutive 63-session windows
        ending from ``signal - 252`` through ``signal - 1``.  Their union is
        exactly the 314 sessions immediately preceding the signal.  The
        signal cache already contains only eligible, unambiguous one-session
        legs, so a dense sliding view is semantically identical to the
        reference loop and preserves its fail-closed missingness rule.
        """

        if current_volatility is None or self.session_context is None:
            return None
        sessions, signal_index = self.session_context
        first = signal_index - 314
        if first < 0:
            return None
        legs = [
            self._valid_one_session_leg_map.get(sessions[index])
            for index in range(first, signal_index)
        ]
        if any(leg is None for leg in legs):
            return None
        values = np.asarray(
            [leg.value for leg in legs if leg is not None],
            dtype=np.float64,
        )
        if values.shape != (314,) or not np.isfinite(values).all():
            return None
        windows = np.lib.stride_tricks.sliding_window_view(
            values,
            window_shape=63,
        )
        if windows.shape != (252, 63):
            return None
        estimates = (
            math.sqrt(252.0)
            * np.std(windows, axis=1, ddof=1)
        )
        prior_median = ordinary_median(estimates)
        if prior_median is None or prior_median <= 0.0:
            return None
        result = current_volatility / prior_median - 1.0
        return float(result) if math.isfinite(result) else None

    def shares_on_dates(
        self,
        dates: Sequence[dt.date],
    ) -> list[float] | None:
        values: list[float] = []
        for day in dates:
            if day not in self._engine.share_value_by_date:
                candidates = []
                for row in self._engine.shares_rows:
                    start = _date(row.get("start_date"))
                    end = _date(row.get("end_date"))
                    if start is not None and start <= day and (
                        end is None or day <= end
                    ):
                        candidates.append(row)
                value = None
                if len(candidates) == 1:
                    shares = _number(candidates[0].get("shares_outstanding"))
                    if shares is not None and shares > 0.0:
                        value = shares
                self._engine.share_value_by_date[day] = value
            value = self._engine.share_value_by_date[day]
            if value is None:
                return None
            values.append(value)
        return values

    def months_since_first_tradable(self) -> float | None:
        end = bisect_right(
            self._engine.tradable_dates,
            self._signal_date,
        )
        if end == 0:
            return None
        return float(
            _completed_calendar_months(
                self._engine.tradable_dates[0],
                self._signal_date,
            )
        )

    def listing_lineage_dates(
        self,
    ) -> tuple[dt.date | None, dt.date | None]:
        end = bisect_right(
            self._engine.tradable_dates,
            self._signal_date,
        )
        if end == 0:
            return None, None
        latest = self._engine.tradable_dates[end - 1]
        return latest, latest


class CachedCrspSecurityEngine:
    """Prepare one security history once, then score all of its signals."""

    def __init__(
        self,
        *,
        sessions: Sequence[dt.date],
        market_rows: Sequence[Mapping[str, Any]],
        return_rows: Sequence[Mapping[str, Any]],
        factor_rows: Sequence[Mapping[str, Any]],
        shares_rows: Sequence[Mapping[str, Any]],
        listing_history_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.sessions = tuple(sessions)
        if (
            not self.sessions
            or any(
                not isinstance(day, dt.date)
                or isinstance(day, dt.datetime)
                for day in self.sessions
            )
            or any(
                left >= right
                for left, right in zip(
                    self.sessions,
                    self.sessions[1:],
                )
            )
        ):
            raise FastCrspFeatureError(
                "Sessions must be unique, sorted date values"
            )
        self.session_position = {
            day: index for index, day in enumerate(self.sessions)
        }
        self.market_rows = tuple(market_rows)
        self.return_rows = tuple(return_rows)
        self.factor_rows = tuple(factor_rows)
        self.shares_rows = tuple(shares_rows)
        self.listing_history_rows = (
            self.market_rows
            if listing_history_rows is None
            else tuple(listing_history_rows)
        )
        try:
            self.market_index = _index_rows(
                self.market_rows,
                "calendar_date",
            )
            self.factor_index = _index_rows(
                self.factor_rows,
                "observation_date",
            )
        except CrspFeatureError as exc:
            raise FastCrspFeatureError(str(exc)) from exc

        legs: list[_ReturnLeg] = []
        for row in self.return_rows:
            if row.get("return_history_usable") is not True:
                continue
            observation = _date(row.get("calendar_date"))
            span_end = _date(row.get("return_span_end_date"))
            availability = _date(row.get("outcome_availability_date"))
            value = _number(row.get("authoritative_total_return"))
            coverage = _return_leg_coverage(row, self.sessions)
            if (
                observation is not None
                and span_end == observation
                and availability is not None
                and value is not None
                and value >= -1.0
                and coverage is not None
            ):
                legs.append(
                    _ReturnLeg(
                        ordinal=len(legs),
                        coverage=coverage,
                        value=value,
                        row=row,
                        observation_date=observation,
                        availability_date=availability,
                    )
                )
        self.return_legs = tuple(legs)
        one_session_counts: dict[int, int] = {}
        for leg in self.return_legs:
            if len(leg.coverage) == 1:
                position = leg.coverage[0]
                one_session_counts[position] = (
                    one_session_counts.get(position, 0) + 1
                )
        direct: list[_ReturnLeg | None] = [None] * len(self.sessions)
        exceptions: list[_ReturnLeg] = []
        for leg in self.return_legs:
            if (
                len(leg.coverage) != 1
                or leg.availability_date > leg.observation_date
                or one_session_counts[leg.coverage[0]] != 1
            ):
                exceptions.append(leg)
            else:
                direct[leg.coverage[0]] = leg
        self.direct_return_legs_by_position = tuple(direct)
        self.exception_return_legs = tuple(exceptions)
        listing_rows = self.listing_history_rows or self.market_rows
        self.tradable_dates = tuple(
            sorted(
                day
                for row in listing_rows
                if (day := _tradable_date(row)) is not None
            )
        )
        self.share_value_by_date: dict[dt.date, float | None] = {}
        self._last_signal_cache_key: tuple[int, dt.date] | None = None
        self._last_signal_cache: _SignalCache | None = None

    def input_for_signal(
        self,
        *,
        permno: int,
        signal_date: dt.date,
        long_eligible: bool,
        primary_exchange: str | None,
    ) -> CrspFeatureInput:
        key = (int(permno), signal_date)
        cache = (
            self._last_signal_cache
            if self._last_signal_cache_key == key
            else None
        )
        if cache is None:
            cache = _SignalCache(
                self,
                permno=permno,
                signal_date=signal_date,
            )
            self._last_signal_cache_key = key
            self._last_signal_cache = cache
        return CrspFeatureInput(
            permno=int(permno),
            signal_date=signal_date,
            sessions=self.sessions,
            market_rows=self.market_rows,
            return_rows=self.return_rows,
            factor_rows=self.factor_rows,
            shares_rows=self.shares_rows,
            listing_history_rows=self.listing_history_rows,
            long_eligible=bool(long_eligible),
            primary_exchange=primary_exchange,
            prepared_cache=cache,
        )

    def build_unfinalized_result(
        self,
        *,
        permno: int,
        signal_date: dt.date,
        long_eligible: bool,
        primary_exchange: str | None,
    ) -> FeatureFamilyResult:
        data = self.input_for_signal(
            permno=permno,
            signal_date=signal_date,
            long_eligible=long_eligible,
            primary_exchange=primary_exchange,
        )
        return build_crsp_unfinalized_feature_result(data)
