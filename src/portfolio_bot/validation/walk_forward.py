"""Append-only walk-forward chronology primitives for Stage 11.

The ledger records a forecast commitment before any corresponding outcome can
be attached.  Training selection is then reconstructed solely from committed
records whose exit and public-availability dates precede the refit signal,
followed by the frozen one-holding-period purge.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, replace

import numpy as np

from portfolio_bot.validation.calendar import (
    ValidationCalendarRow,
    ValidationCalendarError,
    validate_validation_calendar,
)


PRIMARY_REFIT_MONTHS = frozenset({3, 6, 9, 12})
ONE_HOLDING_PERIOD_PURGE = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_REBALANCE_CALENDAR_SHA256 = (
    "cc03002bb9c349c07f14d699bf3667fd4ccbcd012f20579b1ce1c22dd19f2591"
)


class WalkForwardError(ValueError):
    """Raised when an append would violate pseudo-live chronology."""


def _date(value: object, *, name: str) -> dt.date:
    if type(value) is not dt.date:
        raise WalkForwardError(f"{name} must be a datetime.date")
    return value


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WalkForwardError(f"{name} must be a nonempty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class MonthlyScheduleDecision:
    """Monthly scoring is unconditional; parameter refits are quarterly."""

    signal_date: dt.date
    score_and_optimize: bool
    refit_parameters_and_calibration: bool


@dataclass(frozen=True, slots=True)
class AuthenticatedValidationCalendar:
    """Exact Stage 11 calendar rows bound to the verified Phase 2 payload."""

    rows: tuple[ValidationCalendarRow, ...]
    source_payload_sha256: str

    def __post_init__(self) -> None:
        if self.source_payload_sha256 != EXPECTED_REBALANCE_CALENDAR_SHA256:
            raise WalkForwardError(
                "calendar authority does not bind the verified rebalance payload"
            )
        try:
            validated = validate_validation_calendar(self.rows)
        except ValidationCalendarError as exc:
            raise WalkForwardError("calendar authority is not the exact validation calendar") from exc
        object.__setattr__(self, "rows", validated)

    def row_for_signal_date(self, signal_date: dt.date) -> ValidationCalendarRow:
        signal = _date(signal_date, name="signal_date")
        matches = tuple(row for row in self.rows if row.signal_date == signal)
        if len(matches) != 1:
            raise WalkForwardError(
                "signal_date is not present exactly once in the authenticated calendar"
            )
        return matches[0]


def monthly_schedule_decision(
    signal_date: dt.date,
    *,
    calendar_authority: AuthenticatedValidationCalendar,
) -> MonthlyScheduleDecision:
    signal = _date(signal_date, name="signal_date")
    if type(calendar_authority) is not AuthenticatedValidationCalendar:
        raise WalkForwardError("monthly scheduling requires authenticated calendar authority")
    calendar_authority.row_for_signal_date(signal)
    return MonthlyScheduleDecision(
        signal_date=signal,
        score_and_optimize=True,
        refit_parameters_and_calibration=signal.month in PRIMARY_REFIT_MONTHS,
    )


@dataclass(frozen=True, slots=True)
class ForecastCommit:
    """One immutable forecast identity committed at its signal date."""

    forecast_id: str
    signal_date: dt.date
    entry_date: dt.date
    exit_date: dt.date
    payload_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "forecast_id", _identifier(self.forecast_id, name="forecast_id")
        )
        signal = _date(self.signal_date, name="signal_date")
        entry = _date(self.entry_date, name="entry_date")
        exit_date = _date(self.exit_date, name="exit_date")
        if not signal < entry < exit_date:
            raise WalkForwardError("forecast dates must satisfy signal < entry < exit")
        if not isinstance(self.payload_sha256, str) or _SHA256.fullmatch(
            self.payload_sha256
        ) is None:
            raise WalkForwardError("payload_sha256 must be lowercase hexadecimal SHA-256")

    @property
    def holding_month(self) -> str:
        return f"{self.entry_date.year:04d}-{self.entry_date.month:02d}"


@dataclass(frozen=True, slots=True)
class OutcomeCommit:
    """Synthetic/development outcome attached only after its forecast exists."""

    forecast_id: str
    exit_date: dt.date
    availability_date: dt.date
    outcome_value: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "forecast_id", _identifier(self.forecast_id, name="forecast_id")
        )
        exit_date = _date(self.exit_date, name="exit_date")
        availability = _date(self.availability_date, name="availability_date")
        if availability < exit_date:
            raise WalkForwardError("outcome cannot be available before its exit")
        try:
            value = float(self.outcome_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise WalkForwardError("outcome_value must be numeric") from exc
        if not np.isfinite(value):
            raise WalkForwardError("outcome_value must be finite")
        object.__setattr__(self, "outcome_value", value)


@dataclass(frozen=True, slots=True)
class TrainingLabelSelection:
    """Auditable included, purged, and unavailable records at one refit."""

    refit_signal_date: dt.date
    included: tuple[OutcomeCommit, ...]
    purged: tuple[ForecastCommit, ...]
    unavailable_forecast_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WalkForwardLedger:
    """Immutable chronological ledger; append methods return a new instance."""

    forecasts: tuple[ForecastCommit, ...] = ()
    outcomes: tuple[OutcomeCommit, ...] = ()
    last_commit_date: dt.date | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(item, ForecastCommit) for item in self.forecasts):
            raise WalkForwardError("forecasts must contain ForecastCommit records")
        if any(not isinstance(item, OutcomeCommit) for item in self.outcomes):
            raise WalkForwardError("outcomes must contain OutcomeCommit records")
        forecast_ids = tuple(item.forecast_id for item in self.forecasts)
        outcome_ids = tuple(item.forecast_id for item in self.outcomes)
        if len(set(forecast_ids)) != len(forecast_ids):
            raise WalkForwardError("forecast IDs must be unique")
        if len(set(outcome_ids)) != len(outcome_ids):
            raise WalkForwardError("each forecast may receive only one outcome")
        if not set(outcome_ids) <= set(forecast_ids):
            raise WalkForwardError("every outcome must reference a committed forecast")
        if tuple(item.signal_date for item in self.forecasts) != tuple(
            sorted(item.signal_date for item in self.forecasts)
        ):
            raise WalkForwardError("forecast commitments must be chronologically ordered")
        if tuple(item.availability_date for item in self.outcomes) != tuple(
            sorted(item.availability_date for item in self.outcomes)
        ):
            raise WalkForwardError("outcome commitments must be chronologically ordered")

        forecast_by_id = {item.forecast_id: item for item in self.forecasts}
        for outcome in self.outcomes:
            forecast = forecast_by_id[outcome.forecast_id]
            if outcome.exit_date != forecast.exit_date:
                raise WalkForwardError(
                    "outcome exit_date differs from forecast commitment"
                )

        event_dates = tuple(item.signal_date for item in self.forecasts) + tuple(
            item.availability_date for item in self.outcomes
        )
        if not event_dates:
            if self.last_commit_date is not None:
                raise WalkForwardError("an empty ledger cannot have a last_commit_date")
        else:
            if self.last_commit_date is None:
                raise WalkForwardError(
                    "a populated ledger requires its exact last_commit_date"
                )
            last_commit = _date(self.last_commit_date, name="last_commit_date")
            if last_commit != max(event_dates):
                raise WalkForwardError(
                    "last_commit_date must equal the latest committed event date"
                )

    def _require_not_backdated(self, event_date: dt.date) -> None:
        if self.last_commit_date is not None and event_date < self.last_commit_date:
            raise WalkForwardError("ledger append is backdated relative to prior commits")

    def commit_forecast(self, forecast: ForecastCommit) -> "WalkForwardLedger":
        """Commit a forecast at the signal date, before any outcome attachment."""

        if not isinstance(forecast, ForecastCommit):
            raise TypeError("forecast must be ForecastCommit")
        if forecast.forecast_id in {item.forecast_id for item in self.forecasts}:
            raise WalkForwardError("forecast_id has already been committed")
        self._require_not_backdated(forecast.signal_date)
        return replace(
            self,
            forecasts=self.forecasts + (forecast,),
            last_commit_date=forecast.signal_date,
        )

    def commit_outcome(self, outcome: OutcomeCommit) -> "WalkForwardLedger":
        """Attach an outcome only after its exact forecast and exit are committed."""

        if not isinstance(outcome, OutcomeCommit):
            raise TypeError("outcome must be OutcomeCommit")
        forecast_by_id = {item.forecast_id: item for item in self.forecasts}
        forecast = forecast_by_id.get(outcome.forecast_id)
        if forecast is None:
            raise WalkForwardError("outcome arrived before its forecast commitment")
        if outcome.forecast_id in {item.forecast_id for item in self.outcomes}:
            raise WalkForwardError("forecast already has a committed outcome")
        if outcome.exit_date != forecast.exit_date:
            raise WalkForwardError("outcome exit_date differs from forecast commitment")
        self._require_not_backdated(outcome.availability_date)
        return replace(
            self,
            outcomes=self.outcomes + (outcome,),
            last_commit_date=outcome.availability_date,
        )

    def training_labels_for_refit(
        self,
        refit_signal_date: dt.date,
        *,
        calendar_authority: AuthenticatedValidationCalendar,
    ) -> TrainingLabelSelection:
        """Select completed labels and purge the current overlapping holding period.

        This method is intentionally restricted to the frozen quarterly
        parameter/calibration refits.  Monthly calls still score and optimize,
        but they reuse the latest fitted parameters.
        """

        refit = _date(refit_signal_date, name="refit_signal_date")
        if not monthly_schedule_decision(
            refit,
            calendar_authority=calendar_authority,
        ).refit_parameters_and_calibration:
            raise WalkForwardError(
                "primary training-label selection is permitted only at quarterly refits"
            )
        complete = tuple(
            outcome
            for outcome in self.outcomes
            if outcome.exit_date <= refit and outcome.availability_date <= refit
        )
        # The one-period purge concerns the *currently open* holding interval,
        # not the most recently completed interval.  For example, at a
        # 2014-12-31 refit the November holding return is already observable
        # and remains usable, while the December holding forecast (whose exit
        # occurs in January) is the overlapping, purged record.
        purged = tuple(
            forecast
            for forecast in self.forecasts
            if forecast.entry_date <= refit < forecast.exit_date
        )
        purged_months = {item.holding_month for item in purged}
        if len(purged_months) > ONE_HOLDING_PERIOD_PURGE:
            raise WalkForwardError(
                "more than one holding period overlaps the refit signal"
            )
        included = complete
        included_ids = {item.forecast_id for item in included}
        purged_ids = {item.forecast_id for item in purged}
        unavailable = tuple(
            item.forecast_id
            for item in self.forecasts
            if item.forecast_id not in included_ids
            and item.forecast_id not in purged_ids
        )
        return TrainingLabelSelection(
            refit_signal_date=refit,
            included=included,
            purged=purged,
            unavailable_forecast_ids=unavailable,
        )


__all__ = [
    "AuthenticatedValidationCalendar",
    "EXPECTED_REBALANCE_CALENDAR_SHA256",
    "ONE_HOLDING_PERIOD_PURGE",
    "PRIMARY_REFIT_MONTHS",
    "ForecastCommit",
    "MonthlyScheduleDecision",
    "OutcomeCommit",
    "TrainingLabelSelection",
    "WalkForwardError",
    "WalkForwardLedger",
    "monthly_schedule_decision",
]
