"""Pure, frozen semantics for the Phase 2 CRSP return gate.

The functions in this module contain no filesystem or publication logic.  In
particular, delisting classification reconstructs the official CRSP legacy
DLSTCD tuple rather than inferring performance from a reason label alone.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN


RETURN_QUANTUM = Decimal("0.000000000001")
RETURN_FLOOR = Decimal("-1.000000000000")
MONTHLY_ABSOLUTE_TOLERANCE = Decimal("0.000060000000")

ONE_PERIOD_DURATIONS = frozenset({"D1", "D2", "D3", "D4", "DU"})
MULTI_PERIODS = {f"P{number}": number + 1 for number in range(1, 10)}
KNOWN_DURATIONS = ONE_PERIOD_DURATIONS | frozenset(MULTI_PERIODS) | {"DD", "MR"}
KNOWN_MISSING_FLAGS = frozenset({"DG", "DM", "DP", "GP", "MP", "MV", "NA", "NS", "NT", "RA"})
KNOWN_MONTHLY_FLAGS = frozenset({"CR", "DE", "GP", "IP", "MP", "NS", "NT"})

# Official CIZ reconstruction of legacy DLSTCD
# {500,520,550,551,552,560,561,570,573,574,575,580,581,582,583,584}.
PERFORMANCE_REASONS = frozenset(
    {
        "UNAV",
        "MVOT",
        "MTMK",
        "SHLD",
        "LP",
        "INSC",
        "INSF",
        "CORQ",
        "DERE",
        "BKPY",
        "OFFRE",
        "DELQ",
        "FARG",
        "EQRQ",
        "DEEX",
        "FING",
    }
)


class ReturnPolicyError(ValueError):
    """Raised when a value cannot be treated under the frozen policy."""


def trading_periods(duration_flag: str, *, is_delisting_leg: bool = False) -> int | None:
    """Return ordinary factor-window periods, or ``None`` for an event leg.

    P1--P9 are multi-session ordinary returns only when they are not the
    separately posted delisting payoff.  DD and every event-linked P row are
    terminal event legs and must never create an overlapping market-factor
    window.
    """

    if duration_flag not in KNOWN_DURATIONS:
        raise ReturnPolicyError(f"Unknown daily duration flag {duration_flag!r}")
    if is_delisting_leg or duration_flag == "DD":
        return None
    if duration_flag in ONE_PERIOD_DURATIONS:
        return 1
    if duration_flag in MULTI_PERIODS:
        return MULTI_PERIODS[duration_flag]
    if duration_flag == "MR":
        return 0
    raise AssertionError(duration_flag)


def observed_total_is_usable(
    value: Decimal | None,
    missing_flag: str,
    duration_flag: str,
) -> bool:
    """Whether CRSP confirms a complete observed total-return payload."""

    if missing_flag not in KNOWN_MISSING_FLAGS:
        raise ReturnPolicyError(f"Unknown daily missing flag {missing_flag!r}")
    if duration_flag not in KNOWN_DURATIONS:
        raise ReturnPolicyError(f"Unknown daily duration flag {duration_flag!r}")
    if value is None or value < RETURN_FLOOR:
        return False
    # MV can carry a number but officially means a corporate-action value is
    # missing.  Preserving it for QA is different from authorizing it.
    return missing_flag == "NA" and duration_flag != "MR"


def is_performance_related_delisting(
    *, action_type: str, payment_type: str, status_type: str, reason_type: str
) -> bool:
    """Reconstruct the locked performance-related legacy DLSTCD tuple."""

    return (
        action_type == "GDR"
        and payment_type == "PRCF"
        and status_type == "VCL"
        and reason_type in PERFORMANCE_REASONS
    )


def delisting_imputation(exchange: str) -> Decimal:
    """Return the frozen Shumway-style missing-payoff assumption."""

    if exchange in {"N", "A"}:
        return Decimal("-0.300000000000")
    if exchange == "Q":
        return Decimal("-0.550000000000")
    raise ReturnPolicyError(
        f"No primary missing-performance-delisting assumption for exchange {exchange!r}"
    )


def combine_nonoverlapping_returns(*values: Decimal) -> Decimal:
    """Compound distinct legs once and enforce the economic return floor."""

    gross = Decimal(1)
    for value in values:
        if value < RETURN_FLOOR:
            raise ReturnPolicyError(f"Return below -100%: {value}")
        gross *= Decimal(1) + value
    result = (gross - Decimal(1)).quantize(RETURN_QUANTUM, rounding=ROUND_HALF_EVEN)
    return max(RETURN_FLOOR, result)


def within_monthly_tolerance(left: Decimal | None, right: Decimal | None) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) <= MONTHLY_ABSOLUTE_TOLERANCE
