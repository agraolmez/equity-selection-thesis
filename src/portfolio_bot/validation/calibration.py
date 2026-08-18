"""Past-only score-to-residual-return calibration for Stage 11.

Only genuine development OOF forecasts and previously committed validation
forecasts whose outcomes are already available may enter.  The implementation
uses the frozen zero-intercept Huber regression, gives each calendar month
equal mass before applying the 120-month half-life, and truncates a nonpositive
slope to zero.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import HuberRegressor

from portfolio_bot.selectors.weights import month_ordinal


HUBER_EPSILON = 1.345
MEMORY_HALF_LIFE_MONTHS = 120.0
PERMITTED_CALIBRATION_SOURCES = frozenset(
    {"development_oof", "validation_prior_live"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HOLDING_MONTH = re.compile(r"^[0-9]{4}-(?:0[1-9]|1[0-2])$")


class CalibrationError(ValueError):
    """Raised when calibration observations are not genuinely past-only."""


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    forecast_id: str
    holding_month: str
    permno: int
    score: float
    residual_return: float
    signal_date: dt.date
    entry_date: dt.date
    exit_date: dt.date
    forecast_payload_sha256: str
    forecast_committed_at: dt.datetime
    outcome_available_at: dt.datetime
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.forecast_id, str) or not self.forecast_id.strip():
            raise CalibrationError("forecast_id must be a nonempty string")
        object.__setattr__(self, "forecast_id", self.forecast_id.strip())
        if not isinstance(self.holding_month, str) or _HOLDING_MONTH.fullmatch(
            self.holding_month
        ) is None:
            raise CalibrationError("holding_month must use YYYY-MM")
        try:
            month_ordinal(self.holding_month)
        except (TypeError, ValueError) as exc:
            raise CalibrationError("holding_month must use YYYY-MM") from exc
        if type(self.permno) is not int or self.permno <= 0:
            raise CalibrationError("permno must be a positive integer")
        try:
            score = float(self.score)
            residual_return = float(self.residual_return)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CalibrationError(
                "calibration score and outcome must be numeric"
            ) from exc
        if not math.isfinite(score) or not math.isfinite(residual_return):
            raise CalibrationError("calibration score and outcome must be finite")
        if not -0.5 <= score <= 0.5:
            raise CalibrationError("calibration score must be within [-0.5, 0.5]")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "residual_return", residual_return)

        if type(self.signal_date) is not dt.date:
            raise CalibrationError("signal_date must be a datetime.date")
        if type(self.entry_date) is not dt.date:
            raise CalibrationError("entry_date must be a datetime.date")
        if type(self.exit_date) is not dt.date:
            raise CalibrationError("exit_date must be a datetime.date")
        if not self.signal_date < self.entry_date < self.exit_date:
            raise CalibrationError(
                "forecast dates must satisfy signal < entry < exit"
            )
        expected_holding_month = (
            f"{self.entry_date.year:04d}-{self.entry_date.month:02d}"
        )
        if self.holding_month != expected_holding_month:
            raise CalibrationError(
                "holding_month must match the forecast entry month"
            )
        if not isinstance(self.forecast_payload_sha256, str) or _SHA256.fullmatch(
            self.forecast_payload_sha256
        ) is None:
            raise CalibrationError(
                "forecast_payload_sha256 must be lowercase hexadecimal SHA-256"
            )
        if not isinstance(self.forecast_committed_at, dt.datetime) or not isinstance(
            self.outcome_available_at, dt.datetime
        ):
            raise CalibrationError("calibration timestamps must be datetimes")
        if (
            self.forecast_committed_at.tzinfo is None
            or self.outcome_available_at.tzinfo is None
        ):
            raise CalibrationError("calibration timestamps must be timezone-aware")
        committed_utc = self.forecast_committed_at.astimezone(dt.timezone.utc)
        available_utc = self.outcome_available_at.astimezone(dt.timezone.utc)
        if committed_utc.date() != self.signal_date:
            raise CalibrationError(
                "forecast commitment timestamp must bind to signal_date"
            )
        entry_start_utc = dt.datetime.combine(
            self.entry_date, dt.time.min, tzinfo=dt.timezone.utc
        )
        if committed_utc >= entry_start_utc:
            raise CalibrationError("forecast must be committed before entry")
        if available_utc.date() < self.exit_date:
            raise CalibrationError("outcome cannot be available before forecast exit")
        if committed_utc >= available_utc:
            raise CalibrationError("forecast must be committed before its outcome")
        if self.source not in PERMITTED_CALIBRATION_SOURCES:
            raise CalibrationError("calibration source must be OOF or prior-live")


@dataclass(frozen=True, slots=True)
class CalibrationReceipt:
    fit_signal_date: dt.date
    kappa: float
    unconstrained_slope: float
    observations: int
    distinct_months: int
    first_holding_month: str
    last_holding_month: str
    development_oof_observations: int
    validation_prior_live_observations: int
    equal_month_mass: bool
    half_life_months: float
    huber_epsilon: float
    nonpositive_slope_truncated_to_zero: bool


def _fit_cutoff(fit_signal_date: dt.date) -> dt.datetime:
    if type(fit_signal_date) is not dt.date:
        raise CalibrationError("fit_signal_date must be a datetime.date")
    return dt.datetime.combine(
        fit_signal_date,
        dt.time(23, 59, 59, 999999),
        tzinfo=dt.timezone.utc,
    )


def calibration_observation_weights(
    observations: Iterable[CalibrationObservation],
    *,
    fit_signal_date: dt.date,
    half_life_months: float = MEMORY_HALF_LIFE_MONTHS,
) -> np.ndarray:
    """Return equal-month, exponentially decayed row weights with mean one."""

    rows = tuple(observations)
    if not rows:
        raise CalibrationError("at least one calibration observation is required")
    if not math.isfinite(float(half_life_months)) or float(half_life_months) <= 0.0:
        raise CalibrationError("half_life_months must be positive and finite")
    cutoff = _fit_cutoff(fit_signal_date)
    fit_month = month_ordinal(fit_signal_date)
    counts: dict[str, int] = {}
    seen_security_months: set[tuple[str, int]] = set()
    for row in rows:
        if not isinstance(row, CalibrationObservation):
            raise CalibrationError("calibration rows must be CalibrationObservation")
        age = fit_month - month_ordinal(row.holding_month)
        if age <= 0:
            raise CalibrationError("current or future holding month entered calibration")
        if row.outcome_available_at.astimezone(dt.timezone.utc) > cutoff:
            raise CalibrationError("a calibration outcome is not available at fit time")
        security_month = (row.holding_month, row.permno)
        if security_month in seen_security_months:
            raise CalibrationError(
                "duplicate (holding_month, permno) calibration observation"
            )
        seen_security_months.add(security_month)
        counts[row.holding_month] = counts.get(row.holding_month, 0) + 1
    raw = np.asarray(
        [
            (2.0 ** (-float(fit_month - month_ordinal(row.holding_month)) / half_life_months))
            / counts[row.holding_month]
            for row in rows
        ],
        dtype=np.float64,
    )
    total = float(raw.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise CalibrationError("calibration weights have invalid total mass")
    result = raw * (len(raw) / total)
    result.setflags(write=False)
    return result


def fit_nonnegative_huber_calibration(
    observations: Iterable[CalibrationObservation],
    *,
    fit_signal_date: dt.date,
    half_life_months: float = MEMORY_HALF_LIFE_MONTHS,
) -> CalibrationReceipt:
    """Fit the frozen zero-intercept Huber slope and enforce ``kappa >= 0``."""

    rows = tuple(observations)
    weights = calibration_observation_weights(
        rows,
        fit_signal_date=fit_signal_date,
        half_life_months=half_life_months,
    )
    score = np.asarray([row.score for row in rows], dtype=np.float64)
    raw_outcome = np.asarray(
        [row.residual_return for row in rows], dtype=np.float64
    )
    # The frozen calibration target is the within-month cross-sectionally
    # demeaned residual return.  Do this here rather than trusting a caller to
    # supply a pre-demeaned column; otherwise a changing monthly residual mean
    # could leak into the score-to-return slope.
    outcome = raw_outcome.copy()
    month_positions: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        month_positions.setdefault(row.holding_month, []).append(index)
    for positions in month_positions.values():
        indices = np.asarray(positions, dtype=np.int64)
        outcome[indices] -= float(np.mean(raw_outcome[indices], dtype=np.float64))
    if np.all(score == 0.0):
        slope = 0.0
    else:
        model = HuberRegressor(
            epsilon=HUBER_EPSILON,
            alpha=0.0,
            fit_intercept=False,
            max_iter=1000,
            tol=1e-10,
        )
        try:
            model.fit(score.reshape(-1, 1), outcome, sample_weight=weights)
        except (ValueError, FloatingPointError) as exc:
            raise CalibrationError("Huber calibration failed") from exc
        slope = float(model.coef_[0])
        if not math.isfinite(slope):
            raise CalibrationError("Huber calibration returned a nonfinite slope")
    kappa = max(0.0, slope)
    months = sorted({row.holding_month for row in rows}, key=month_ordinal)
    return CalibrationReceipt(
        fit_signal_date=fit_signal_date,
        kappa=kappa,
        unconstrained_slope=slope,
        observations=len(rows),
        distinct_months=len(months),
        first_holding_month=months[0],
        last_holding_month=months[-1],
        development_oof_observations=sum(
            row.source == "development_oof" for row in rows
        ),
        validation_prior_live_observations=sum(
            row.source == "validation_prior_live" for row in rows
        ),
        equal_month_mass=True,
        half_life_months=float(half_life_months),
        huber_epsilon=HUBER_EPSILON,
        nonpositive_slope_truncated_to_zero=slope <= 0.0,
    )


__all__ = [
    "CalibrationError",
    "CalibrationObservation",
    "CalibrationReceipt",
    "HUBER_EPSILON",
    "MEMORY_HALF_LIFE_MONTHS",
    "PERMITTED_CALIBRATION_SOURCES",
    "calibration_observation_weights",
    "fit_nonnegative_huber_calibration",
]
