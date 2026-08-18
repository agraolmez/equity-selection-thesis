"""Pure Stage 11 validation-calendar boundaries.

This module deliberately performs no filesystem access.  It describes the 59
complete validation holding months and validates rows supplied by a separately
authenticated rebalance-calendar loader.  In particular, it cannot widen the
window to the embargoed December 2019 holding interval or to the final period.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence


FIRST_VALIDATION_HOLDING_MONTH = "2015-01"
LAST_VALIDATION_HOLDING_MONTH = "2019-11"
EMBARGOED_HOLDING_MONTH = "2019-12"
VALIDATION_HOLDING_MONTH_COUNT = 59

FIRST_VALIDATION_SIGNAL_DATE = dt.date(2014, 12, 31)
FIRST_VALIDATION_ENTRY_DATE = dt.date(2015, 1, 2)
FIRST_VALIDATION_EXIT_DATE = dt.date(2015, 2, 2)
LAST_VALIDATION_SIGNAL_DATE = dt.date(2019, 10, 31)
LAST_VALIDATION_ENTRY_DATE = dt.date(2019, 11, 1)
LAST_VALIDATION_EXIT_DATE = dt.date(2019, 12, 2)
FIRST_FINAL_PERIOD_ENTRY_DATE = dt.date(2020, 1, 2)

_MONTH_PATTERN = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
_CALENDAR_ROW_FIELDS = frozenset(
    {"holding_month", "signal_date", "entry_date", "exit_date"}
)


class ValidationCalendarError(ValueError):
    """Raised when a proposed validation calendar crosses a frozen boundary."""


def _parse_month(value: object, *, field: str) -> tuple[int, int]:
    if not isinstance(value, str) or _MONTH_PATTERN.fullmatch(value) is None:
        raise ValidationCalendarError(f"{field} must use exact YYYY-MM format")
    year_text, month_text = value.split("-")
    return int(year_text), int(month_text)


def _shift_month(value: str, offset: int) -> str:
    year, month = _parse_month(value, field="month")
    absolute = year * 12 + (month - 1) + offset
    shifted_year, shifted_index = divmod(absolute, 12)
    return f"{shifted_year:04d}-{shifted_index + 1:02d}"


def _month_of_date(value: dt.date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def _require_date(value: object, *, field: str) -> dt.date:
    # ``datetime`` is a date subclass, but accepting it would leave an implicit
    # timezone/time-of-day convention in a daily calendar contract.
    if type(value) is not dt.date:
        raise ValidationCalendarError(f"{field} must be a datetime.date")
    return value


@dataclass(frozen=True, slots=True)
class ValidationHoldingMonthDescriptor:
    """One immutable member of the frozen 59-month validation domain."""

    ordinal: int
    holding_month: str
    signal_month: str
    exit_month: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 1:
            raise ValidationCalendarError("ordinal must be a positive integer")
        _parse_month(self.holding_month, field="holding_month")
        if self.signal_month != _shift_month(self.holding_month, -1):
            raise ValidationCalendarError("signal_month must precede holding_month")
        if self.exit_month != _shift_month(self.holding_month, 1):
            raise ValidationCalendarError("exit_month must follow holding_month")


def _build_descriptors() -> tuple[ValidationHoldingMonthDescriptor, ...]:
    result: list[ValidationHoldingMonthDescriptor] = []
    month = FIRST_VALIDATION_HOLDING_MONTH
    ordinal = 1
    while month <= LAST_VALIDATION_HOLDING_MONTH:
        result.append(
            ValidationHoldingMonthDescriptor(
                ordinal=ordinal,
                holding_month=month,
                signal_month=_shift_month(month, -1),
                exit_month=_shift_month(month, 1),
            )
        )
        month = _shift_month(month, 1)
        ordinal += 1
    if len(result) != VALIDATION_HOLDING_MONTH_COUNT:
        raise RuntimeError("internal validation holding-month count mismatch")
    return tuple(result)


VALIDATION_HOLDING_MONTH_DESCRIPTORS = _build_descriptors()
VALIDATION_HOLDING_MONTHS = tuple(
    descriptor.holding_month
    for descriptor in VALIDATION_HOLDING_MONTH_DESCRIPTORS
)
_DESCRIPTOR_BY_MONTH = MappingProxyType(
    {
        descriptor.holding_month: descriptor
        for descriptor in VALIDATION_HOLDING_MONTH_DESCRIPTORS
    }
)


def validation_holding_month_descriptor(
    holding_month: str,
) -> ValidationHoldingMonthDescriptor:
    """Return the frozen descriptor, rejecting embargo/final-period months."""

    _parse_month(holding_month, field="holding_month")
    try:
        return _DESCRIPTOR_BY_MONTH[holding_month]
    except KeyError as exc:
        raise ValidationCalendarError(
            "holding_month is outside January 2015 through November 2019"
        ) from exc


@dataclass(frozen=True, slots=True)
class ValidationCalendarRow:
    """One authenticated-calendar row proposed for the Stage 11 loader."""

    holding_month: str
    signal_date: dt.date
    entry_date: dt.date
    exit_date: dt.date

    def __post_init__(self) -> None:
        _parse_month(self.holding_month, field="holding_month")
        _require_date(self.signal_date, field="signal_date")
        _require_date(self.entry_date, field="entry_date")
        _require_date(self.exit_date, field="exit_date")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ValidationCalendarRow":
        """Parse an exact row shape; missing and extra fields fail closed."""

        if not isinstance(value, Mapping):
            raise ValidationCalendarError("calendar row must be a mapping")
        fields = frozenset(value)
        if fields != _CALENDAR_ROW_FIELDS:
            missing = sorted(_CALENDAR_ROW_FIELDS - fields)
            extra = sorted(fields - _CALENDAR_ROW_FIELDS)
            raise ValidationCalendarError(
                f"calendar row field mismatch; missing={missing}, extra={extra}"
            )
        return cls(
            holding_month=value["holding_month"],  # type: ignore[arg-type]
            signal_date=value["signal_date"],  # type: ignore[arg-type]
            entry_date=value["entry_date"],  # type: ignore[arg-type]
            exit_date=value["exit_date"],  # type: ignore[arg-type]
        )


def _coerce_row(value: object) -> ValidationCalendarRow:
    if isinstance(value, ValidationCalendarRow):
        return value
    if isinstance(value, Mapping):
        return ValidationCalendarRow.from_mapping(value)
    raise ValidationCalendarError(
        "calendar rows must be ValidationCalendarRow objects or exact mappings"
    )


def validate_validation_calendar(
    rows: Sequence[ValidationCalendarRow | Mapping[str, object]],
) -> tuple[ValidationCalendarRow, ...]:
    """Validate the exact Stage 11 calendar and return an immutable tuple.

    Interior trading dates come from the authenticated Phase 2 calendar.  This
    pure boundary guard verifies their month roles, consecutive rebalance
    handoff, exact first/last observations, and the complete frozen domain.
    """

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValidationCalendarError("calendar rows must be a sequence")
    normalized = tuple(_coerce_row(row) for row in rows)
    if len(normalized) != VALIDATION_HOLDING_MONTH_COUNT:
        raise ValidationCalendarError(
            f"validation calendar must contain exactly "
            f"{VALIDATION_HOLDING_MONTH_COUNT} rows"
        )
    actual_months = tuple(row.holding_month for row in normalized)
    if actual_months != VALIDATION_HOLDING_MONTHS:
        raise ValidationCalendarError(
            "validation holding months must be the exact ordered "
            "January 2015 through November 2019 sequence"
        )

    for index, (row, descriptor) in enumerate(
        zip(normalized, VALIDATION_HOLDING_MONTH_DESCRIPTORS, strict=True)
    ):
        if _month_of_date(row.signal_date) != descriptor.signal_month:
            raise ValidationCalendarError(
                f"{row.holding_month} signal_date is not in its signal month"
            )
        if _month_of_date(row.entry_date) != descriptor.holding_month:
            raise ValidationCalendarError(
                f"{row.holding_month} entry_date is outside its holding month"
            )
        if _month_of_date(row.exit_date) != descriptor.exit_month:
            raise ValidationCalendarError(
                f"{row.holding_month} exit_date is outside the next month"
            )
        if not row.signal_date < row.entry_date < row.exit_date:
            raise ValidationCalendarError(
                f"{row.holding_month} must satisfy signal < entry < exit"
            )
        if index and normalized[index - 1].exit_date != row.entry_date:
            raise ValidationCalendarError(
                f"{row.holding_month} entry must equal the prior holding exit"
            )

    first = normalized[0]
    if (
        first.signal_date,
        first.entry_date,
        first.exit_date,
    ) != (
        FIRST_VALIDATION_SIGNAL_DATE,
        FIRST_VALIDATION_ENTRY_DATE,
        FIRST_VALIDATION_EXIT_DATE,
    ):
        raise ValidationCalendarError("first validation interval is not exact")

    last = normalized[-1]
    if (
        last.signal_date,
        last.entry_date,
        last.exit_date,
    ) != (
        LAST_VALIDATION_SIGNAL_DATE,
        LAST_VALIDATION_ENTRY_DATE,
        LAST_VALIDATION_EXIT_DATE,
    ):
        raise ValidationCalendarError("last validation interval is not exact")
    if last.exit_date >= FIRST_FINAL_PERIOD_ENTRY_DATE:
        raise ValidationCalendarError("validation calendar crosses into 2020")
    return normalized


__all__ = [
    "EMBARGOED_HOLDING_MONTH",
    "FIRST_FINAL_PERIOD_ENTRY_DATE",
    "FIRST_VALIDATION_ENTRY_DATE",
    "FIRST_VALIDATION_EXIT_DATE",
    "FIRST_VALIDATION_HOLDING_MONTH",
    "FIRST_VALIDATION_SIGNAL_DATE",
    "LAST_VALIDATION_ENTRY_DATE",
    "LAST_VALIDATION_EXIT_DATE",
    "LAST_VALIDATION_HOLDING_MONTH",
    "LAST_VALIDATION_SIGNAL_DATE",
    "VALIDATION_HOLDING_MONTH_COUNT",
    "VALIDATION_HOLDING_MONTH_DESCRIPTORS",
    "VALIDATION_HOLDING_MONTHS",
    "ValidationCalendarError",
    "ValidationCalendarRow",
    "ValidationHoldingMonthDescriptor",
    "validate_validation_calendar",
    "validation_holding_month_descriptor",
]
