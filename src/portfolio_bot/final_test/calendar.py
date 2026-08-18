"""Pure, closed calendar domain for the Stage 13 final test.

No file is opened here.  The complete primary period is January 2020 through
August 2025 (68 months).  The separately maintained core extension contains
the three complete months September through November 2025.  December 2025 is
partial and descriptive only; it can never enter confirmatory inference.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Sequence


FIRST_PRIMARY_FULL_HOLDING_MONTH = "2020-01"
LAST_PRIMARY_FULL_HOLDING_MONTH = "2025-08"
PRIMARY_FULL_HOLDING_MONTH_COUNT = 68

FIRST_CORE_COMPLETE_HOLDING_MONTH = "2025-09"
LAST_CORE_COMPLETE_HOLDING_MONTH = "2025-11"
CORE_COMPLETE_HOLDING_MONTH_COUNT = 3

PARTIAL_DECEMBER_HOLDING_MONTH = "2025-12"
FINAL_COMPLETE_HOLDING_MONTH_COUNT = 71
FINAL_HOLDING_MONTH_COUNT = 72

_MONTH_PATTERN = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


class FinalCalendarError(ValueError):
    """Raised when a month sequence crosses a frozen Stage 13 boundary."""


class FinalPeriodSegment(str, Enum):
    """Mutually exclusive final-period reporting segments."""

    PRIMARY_FULL = "primary_full"
    CORE_COMPLETE = "core_complete"
    PARTIAL_DECEMBER_DESCRIPTIVE = "partial_december_descriptive"


@dataclass(frozen=True, slots=True)
class FrozenHoldingIntervalAnchor:
    """One methodology-stated final-period signal/entry/exit anchor.

    The complete 72-row trading calendar remains an authenticated empirical
    input to Stage 13.  These six rows are the exact dates stated in
    Methodology v1 and therefore provide data-independent boundary checks for
    the first/last primary interval, every core interval, and partial December.
    """

    holding_month: str
    signal_date: dt.date
    entry_date: dt.date
    exit_date: dt.date

    def __post_init__(self) -> None:
        descriptor = final_holding_month_descriptor(self.holding_month)
        for field_name in ("signal_date", "entry_date", "exit_date"):
            if type(getattr(self, field_name)) is not dt.date:
                raise FinalCalendarError(f"{field_name} must be a datetime.date")
        if not self.signal_date < self.entry_date < self.exit_date:
            raise FinalCalendarError(
                "frozen interval dates must satisfy signal < entry < exit"
            )
        signal_month = shift_month(self.holding_month, -1)
        observed_signal_month = self.signal_date.strftime("%Y-%m")
        observed_entry_month = self.entry_date.strftime("%Y-%m")
        if observed_signal_month != signal_month:
            raise FinalCalendarError("anchor signal date is outside the prior month")
        if observed_entry_month != self.holding_month:
            raise FinalCalendarError("anchor entry date is outside the holding month")
        observed_exit_month = self.exit_date.strftime("%Y-%m")
        if descriptor.complete_holding_interval:
            if observed_exit_month != shift_month(self.holding_month, 1):
                raise FinalCalendarError(
                    "complete anchor exit date is outside the next month"
                )
        elif (
            descriptor.segment is FinalPeriodSegment.PARTIAL_DECEMBER_DESCRIPTIVE
            and observed_exit_month != self.holding_month
        ):
            raise FinalCalendarError(
                "partial-December administrative exit must remain in December"
            )


def _parse_month(value: object, *, field: str = "holding_month") -> tuple[int, int]:
    if not isinstance(value, str) or _MONTH_PATTERN.fullmatch(value) is None:
        raise FinalCalendarError(f"{field} must use exact YYYY-MM format")
    year_text, month_text = value.split("-")
    return int(year_text), int(month_text)


def shift_month(value: str, offset: int) -> str:
    """Shift one canonical month without consulting an empirical calendar."""

    if type(offset) is not int:
        raise FinalCalendarError("month offset must be an integer")
    year, month = _parse_month(value)
    absolute = year * 12 + month - 1 + offset
    shifted_year, shifted_index = divmod(absolute, 12)
    return f"{shifted_year:04d}-{shifted_index + 1:02d}"


def _inclusive_month_range(first: str, last: str) -> tuple[str, ...]:
    _parse_month(first, field="first")
    _parse_month(last, field="last")
    if first > last:
        raise RuntimeError("internal final-period month range is reversed")
    result: list[str] = []
    current = first
    while current <= last:
        result.append(current)
        current = shift_month(current, 1)
    return tuple(result)


PRIMARY_FULL_HOLDING_MONTHS = _inclusive_month_range(
    FIRST_PRIMARY_FULL_HOLDING_MONTH,
    LAST_PRIMARY_FULL_HOLDING_MONTH,
)
CORE_COMPLETE_HOLDING_MONTHS = _inclusive_month_range(
    FIRST_CORE_COMPLETE_HOLDING_MONTH,
    LAST_CORE_COMPLETE_HOLDING_MONTH,
)
FINAL_COMPLETE_HOLDING_MONTHS = (
    PRIMARY_FULL_HOLDING_MONTHS + CORE_COMPLETE_HOLDING_MONTHS
)
FINAL_HOLDING_MONTHS = FINAL_COMPLETE_HOLDING_MONTHS + (PARTIAL_DECEMBER_HOLDING_MONTH,)

if len(PRIMARY_FULL_HOLDING_MONTHS) != PRIMARY_FULL_HOLDING_MONTH_COUNT:
    raise RuntimeError("internal primary final-period month count mismatch")
if len(CORE_COMPLETE_HOLDING_MONTHS) != CORE_COMPLETE_HOLDING_MONTH_COUNT:
    raise RuntimeError("internal complete core-extension month count mismatch")
if len(FINAL_COMPLETE_HOLDING_MONTHS) != FINAL_COMPLETE_HOLDING_MONTH_COUNT:
    raise RuntimeError("internal complete final-period month count mismatch")
if len(FINAL_HOLDING_MONTHS) != FINAL_HOLDING_MONTH_COUNT:
    raise RuntimeError("internal complete-plus-partial month count mismatch")


@dataclass(frozen=True, slots=True)
class FinalHoldingMonthDescriptor:
    """One immutable member of the frozen Stage 13 reporting domain."""

    ordinal: int
    holding_month: str
    segment: FinalPeriodSegment
    complete_holding_interval: bool
    confirmatory_inference_eligible: bool

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 1 <= self.ordinal <= 72:
            raise FinalCalendarError("ordinal must be in the closed range 1..72")
        _parse_month(self.holding_month)
        if not isinstance(self.segment, FinalPeriodSegment):
            raise FinalCalendarError("segment must be a FinalPeriodSegment")
        if type(self.complete_holding_interval) is not bool:
            raise FinalCalendarError("complete_holding_interval must be boolean")
        if type(self.confirmatory_inference_eligible) is not bool:
            raise FinalCalendarError("confirmatory_inference_eligible must be boolean")
        if self.confirmatory_inference_eligible and (
            self.segment is not FinalPeriodSegment.PRIMARY_FULL
            or not self.complete_holding_interval
        ):
            raise FinalCalendarError(
                "only complete primary months may enter confirmatory inference"
            )
        if self.segment is FinalPeriodSegment.PARTIAL_DECEMBER_DESCRIPTIVE and (
            self.complete_holding_interval
            or self.confirmatory_inference_eligible
            or self.holding_month != PARTIAL_DECEMBER_HOLDING_MONTH
        ):
            raise FinalCalendarError(
                "partial December must remain incomplete and descriptive only"
            )


def _descriptor_for_position(
    ordinal: int,
    holding_month: str,
) -> FinalHoldingMonthDescriptor:
    if holding_month in PRIMARY_FULL_HOLDING_MONTHS:
        return FinalHoldingMonthDescriptor(
            ordinal=ordinal,
            holding_month=holding_month,
            segment=FinalPeriodSegment.PRIMARY_FULL,
            complete_holding_interval=True,
            confirmatory_inference_eligible=True,
        )
    if holding_month in CORE_COMPLETE_HOLDING_MONTHS:
        return FinalHoldingMonthDescriptor(
            ordinal=ordinal,
            holding_month=holding_month,
            segment=FinalPeriodSegment.CORE_COMPLETE,
            complete_holding_interval=True,
            confirmatory_inference_eligible=False,
        )
    return FinalHoldingMonthDescriptor(
        ordinal=ordinal,
        holding_month=holding_month,
        segment=FinalPeriodSegment.PARTIAL_DECEMBER_DESCRIPTIVE,
        complete_holding_interval=False,
        confirmatory_inference_eligible=False,
    )


FINAL_HOLDING_MONTH_DESCRIPTORS = tuple(
    _descriptor_for_position(ordinal, holding_month)
    for ordinal, holding_month in enumerate(FINAL_HOLDING_MONTHS, start=1)
)
_DESCRIPTOR_BY_MONTH = MappingProxyType(
    {
        descriptor.holding_month: descriptor
        for descriptor in FINAL_HOLDING_MONTH_DESCRIPTORS
    }
)


def final_holding_month_descriptor(
    holding_month: str,
) -> FinalHoldingMonthDescriptor:
    """Return a descriptor only for the closed January-2020–December-2025 set."""

    _parse_month(holding_month)
    try:
        return _DESCRIPTOR_BY_MONTH[holding_month]
    except KeyError as exc:
        raise FinalCalendarError(
            "holding_month is outside the frozen Stage 13 final-period domain"
        ) from exc


FROZEN_HOLDING_INTERVAL_ANCHORS = MappingProxyType(
    {
        anchor.holding_month: anchor
        for anchor in (
            FrozenHoldingIntervalAnchor(
                holding_month="2020-01",
                signal_date=dt.date(2019, 12, 31),
                entry_date=dt.date(2020, 1, 2),
                exit_date=dt.date(2020, 2, 3),
            ),
            FrozenHoldingIntervalAnchor(
                holding_month="2025-08",
                signal_date=dt.date(2025, 7, 31),
                entry_date=dt.date(2025, 8, 1),
                exit_date=dt.date(2025, 9, 2),
            ),
            FrozenHoldingIntervalAnchor(
                holding_month="2025-09",
                signal_date=dt.date(2025, 8, 29),
                entry_date=dt.date(2025, 9, 2),
                exit_date=dt.date(2025, 10, 1),
            ),
            FrozenHoldingIntervalAnchor(
                holding_month="2025-10",
                signal_date=dt.date(2025, 9, 30),
                entry_date=dt.date(2025, 10, 1),
                exit_date=dt.date(2025, 11, 3),
            ),
            FrozenHoldingIntervalAnchor(
                holding_month="2025-11",
                signal_date=dt.date(2025, 10, 31),
                entry_date=dt.date(2025, 11, 3),
                exit_date=dt.date(2025, 12, 1),
            ),
            FrozenHoldingIntervalAnchor(
                holding_month="2025-12",
                signal_date=dt.date(2025, 11, 28),
                entry_date=dt.date(2025, 12, 1),
                exit_date=dt.date(2025, 12, 31),
            ),
        )
    }
)


def require_frozen_holding_interval_anchor(
    *,
    holding_month: str,
    signal_date: dt.date,
    entry_date: dt.date,
    exit_date: dt.date,
) -> FrozenHoldingIntervalAnchor:
    """Require an exact Methodology-v1 anchor without opening calendar data."""

    try:
        expected = FROZEN_HOLDING_INTERVAL_ANCHORS[holding_month]
    except (KeyError, TypeError) as exc:
        raise FinalCalendarError(
            "holding_month is not one of the six frozen interval anchors"
        ) from exc
    observed = FrozenHoldingIntervalAnchor(
        holding_month=holding_month,
        signal_date=signal_date,
        entry_date=entry_date,
        exit_date=exit_date,
    )
    if observed != expected:
        raise FinalCalendarError(
            f"{holding_month} signal/entry/exit dates differ from the frozen anchor"
        )
    return observed


def validate_final_month_sequence(
    holding_months: Sequence[str] | Iterable[str],
    *,
    allowed_segments: frozenset[FinalPeriodSegment] | None = None,
    require_contiguous: bool = True,
    allow_cross_segment: bool = False,
) -> tuple[str, ...]:
    """Validate an ordered final-period subset without splicing by default."""

    if isinstance(holding_months, (str, bytes)):
        raise FinalCalendarError("holding_months must be a month sequence")
    try:
        months = tuple(holding_months)
    except TypeError as exc:
        raise FinalCalendarError("holding_months must be iterable") from exc
    if not months:
        raise FinalCalendarError("holding_months cannot be empty")
    if type(require_contiguous) is not bool:
        raise FinalCalendarError("require_contiguous must be boolean")
    if type(allow_cross_segment) is not bool:
        raise FinalCalendarError("allow_cross_segment must be boolean")
    descriptors = tuple(final_holding_month_descriptor(month) for month in months)
    ordinals = tuple(descriptor.ordinal for descriptor in descriptors)
    if len(set(months)) != len(months) or tuple(sorted(ordinals)) != ordinals:
        raise FinalCalendarError(
            "holding_months must be unique and in chronological order"
        )
    if require_contiguous and any(
        right != left + 1 for left, right in zip(ordinals, ordinals[1:])
    ):
        raise FinalCalendarError("holding_months must be contiguous")
    if allowed_segments is not None:
        if (
            not isinstance(allowed_segments, frozenset)
            or not allowed_segments
            or any(
                not isinstance(segment, FinalPeriodSegment)
                for segment in allowed_segments
            )
        ):
            raise FinalCalendarError(
                "allowed_segments must be a nonempty frozen set of segments"
            )
        if any(
            descriptor.segment not in allowed_segments for descriptor in descriptors
        ):
            raise FinalCalendarError(
                "holding_months cross the role's frozen period segment"
            )
    observed_segments = frozenset(descriptor.segment for descriptor in descriptors)
    if len(observed_segments) > 1 and not allow_cross_segment:
        raise FinalCalendarError(
            "primary, core-extension, and partial-December months may not be spliced"
        )
    return months


def require_exact_primary_inference_months(
    holding_months: Sequence[str] | Iterable[str],
) -> tuple[str, ...]:
    """Fail closed unless inference uses exactly the 68 primary full months."""

    months = validate_final_month_sequence(
        holding_months,
        allowed_segments=frozenset({FinalPeriodSegment.PRIMARY_FULL}),
    )
    if months != PRIMARY_FULL_HOLDING_MONTHS:
        raise FinalCalendarError(
            "confirmatory inference requires exactly January 2020 through "
            "August 2025 (68 complete months)"
        )
    return months


def require_exact_core_extension_months(
    holding_months: Sequence[str] | Iterable[str],
) -> tuple[str, ...]:
    """Fail closed unless the series is exactly Sep--Nov 2025."""

    months = validate_final_month_sequence(
        holding_months,
        allowed_segments=frozenset({FinalPeriodSegment.CORE_COMPLETE}),
    )
    if months != CORE_COMPLETE_HOLDING_MONTHS:
        raise FinalCalendarError(
            "core-extension description requires exactly September through "
            "November 2025 (three complete months)"
        )
    return months


def require_exact_partial_december_month(
    holding_months: Sequence[str] | Iterable[str],
) -> tuple[str, ...]:
    """Fail closed unless the series contains partial December 2025 only."""

    months = validate_final_month_sequence(
        holding_months,
        allowed_segments=frozenset({FinalPeriodSegment.PARTIAL_DECEMBER_DESCRIPTIVE}),
    )
    if months != (PARTIAL_DECEMBER_HOLDING_MONTH,):
        raise FinalCalendarError("partial description requires exactly December 2025")
    return months


__all__ = [
    "CORE_COMPLETE_HOLDING_MONTH_COUNT",
    "CORE_COMPLETE_HOLDING_MONTHS",
    "FINAL_COMPLETE_HOLDING_MONTH_COUNT",
    "FINAL_COMPLETE_HOLDING_MONTHS",
    "FINAL_HOLDING_MONTH_COUNT",
    "FINAL_HOLDING_MONTH_DESCRIPTORS",
    "FINAL_HOLDING_MONTHS",
    "FIRST_CORE_COMPLETE_HOLDING_MONTH",
    "FIRST_PRIMARY_FULL_HOLDING_MONTH",
    "FROZEN_HOLDING_INTERVAL_ANCHORS",
    "FinalCalendarError",
    "FinalHoldingMonthDescriptor",
    "FinalPeriodSegment",
    "FrozenHoldingIntervalAnchor",
    "LAST_CORE_COMPLETE_HOLDING_MONTH",
    "LAST_PRIMARY_FULL_HOLDING_MONTH",
    "PARTIAL_DECEMBER_HOLDING_MONTH",
    "PRIMARY_FULL_HOLDING_MONTH_COUNT",
    "PRIMARY_FULL_HOLDING_MONTHS",
    "final_holding_month_descriptor",
    "require_exact_core_extension_months",
    "require_exact_partial_december_month",
    "require_exact_primary_inference_months",
    "require_frozen_holding_interval_anchor",
    "shift_month",
    "validate_final_month_sequence",
]
