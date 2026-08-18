"""Shared publication contract for Stage 6 feature values and lineage."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Mapping


VALUE_STATUSES = (
    "PRESENT",
    "MISSING_AMBIGUOUS_LINK",
    "MISSING_INSUFFICIENT_HISTORY",
    "MISSING_INVALID_INPUT",
    "MISSING_NOT_COVERED",
    "MISSING_SOURCE_UNAVAILABLE",
    "MISSING_STALE",
)
VALUE_STATUS_SET = frozenset(VALUE_STATUSES)


class FeatureLineageError(RuntimeError):
    """Raised when a feature result cannot satisfy the frozen lineage schema."""


@dataclass(frozen=True)
class FeatureLineage:
    max_observation_date: dt.date | None
    max_public_availability_date: dt.date | None
    stale_after_date: dt.date | None
    value_status: str

    def __post_init__(self) -> None:
        if self.value_status not in VALUE_STATUS_SET:
            raise FeatureLineageError(
                f"Unknown feature value status: {self.value_status}"
            )
        dates = (
            self.max_observation_date,
            self.max_public_availability_date,
            self.stale_after_date,
        )
        if any(
            value is not None
            and (not isinstance(value, dt.date) or isinstance(value, dt.datetime))
            for value in dates
        ):
            raise FeatureLineageError("Lineage dates must be date values or null")
        if self.value_status == "PRESENT":
            if (
                self.max_observation_date is None
                or self.max_public_availability_date is None
            ):
                raise FeatureLineageError(
                    "Present values require observation and availability dates"
                )
        elif any(value is not None for value in dates):
            raise FeatureLineageError("Missing values must have null lineage dates")


@dataclass(frozen=True)
class FeatureFamilyResult:
    values: Mapping[str, Any]
    lineage: Mapping[str, FeatureLineage]

    def __post_init__(self) -> None:
        if tuple(self.values) != tuple(self.lineage):
            raise FeatureLineageError(
                "Feature values and lineage must have identical ordered keys"
            )


def make_lineage(
    value: Any,
    observation_date: dt.date | None,
    public_availability_date: dt.date | None,
    stale_after_date: dt.date | None = None,
    missing_status: str = "MISSING_INVALID_INPUT",
    *,
    is_missing: bool | None = None,
) -> FeatureLineage:
    """Create one lineage record and null all dates for a missing value."""

    missing = value is None if is_missing is None else is_missing
    if missing:
        if missing_status == "PRESENT":
            raise FeatureLineageError("Missing feature cannot use PRESENT status")
        return FeatureLineage(None, None, None, missing_status)
    return FeatureLineage(
        observation_date,
        public_availability_date,
        stale_after_date,
        "PRESENT",
    )
