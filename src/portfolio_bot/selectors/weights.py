"""Observation-weight and inner-split rules for Stage 9 selectors."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import numpy as np


HALF_LIFE_MONTHS = 120.0
EARLY_STOPPING_MONTHS = 12


def month_ordinal(value: str | dt.date) -> int:
    """Return a zero-based calendar-month ordinal."""

    if isinstance(value, dt.date):
        return value.year * 12 + value.month - 1
    year_text, month_text = value.split("-", 1)
    year, month = int(year_text), int(month_text)
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid calendar month: {value!r}")
    return year * 12 + month - 1


def month_balanced_memory_weights(
    holding_months: Sequence[str],
    *,
    fit_signal_date: dt.date,
    half_life_months: float = HALF_LIFE_MONTHS,
) -> np.ndarray:
    """Give each month equal mass, apply memory decay, and normalize to mean one.

    Normalizing the final vector to sum to the number of observations preserves
    the frozen relative weights while matching the conventions used by the ML
    solvers.  A global scale factor has no effect on the weighted MSE.
    """

    if not holding_months:
        raise ValueError("At least one holding month is required")
    if not np.isfinite(half_life_months) or half_life_months <= 0:
        raise ValueError("half_life_months must be positive and finite")
    months = np.asarray(tuple(holding_months), dtype=object)
    unique, counts = np.unique(months, return_counts=True)
    count_by_month = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    fit_ordinal = month_ordinal(fit_signal_date)
    raw = np.empty(months.size, dtype=np.float64)
    for index, value in enumerate(months.tolist()):
        age = fit_ordinal - month_ordinal(str(value))
        if age < 0:
            raise ValueError("A training holding month occurs after its fit date")
        raw[index] = (
            2.0 ** (-float(age) / float(half_life_months))
            / float(count_by_month[value])
        )
    total = float(np.sum(raw, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Observation weights have invalid total mass")
    return raw * (float(raw.size) / total)


def latest_month_mask(
    holding_months: Sequence[str],
    *,
    count: int = EARLY_STOPPING_MONTHS,
) -> np.ndarray:
    """Mark the latest ``count`` distinct months as the early-stopping block."""

    if count <= 0:
        raise ValueError("count must be positive")
    values = np.asarray(tuple(holding_months), dtype=object)
    ordered = sorted({str(value) for value in values.tolist()}, key=month_ordinal)
    if len(ordered) <= count:
        raise ValueError("Training history is too short for the inner split")
    selected = set(ordered[-count:])
    return np.asarray([str(value) in selected for value in values], dtype=bool)
