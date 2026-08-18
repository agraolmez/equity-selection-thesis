"""Deterministic numerical primitives for the Stage 7 development labels.

The primary label builder consumes the already-verified Stage 6 beta slopes.
This module therefore evaluates forward outcomes and cross-sectional targets;
it does not re-estimate or re-clip betas.  Independent beta recomputation is a
separate verification responsibility.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Mapping, Sequence


PRESENT = "PRESENT"
NOT_LONG_ELIGIBLE = "NOT_LONG_ELIGIBLE"
MISSING_BETA = "MISSING_BETA"
MISSING_NO_POST_ENTRY_RETURN = "MISSING_NO_POST_ENTRY_RETURN"
MISSING_DUPLICATE_RETURN_LEG = "MISSING_DUPLICATE_RETURN_LEG"
MISSING_RETURN_GAP = "MISSING_RETURN_GAP"
MISSING_MULTI_SESSION_RETURN = "MISSING_MULTI_SESSION_RETURN"
MISSING_INTERVAL_STRADDLE = "MISSING_INTERVAL_STRADDLE"
MISSING_UNUSABLE_RETURN = "MISSING_UNUSABLE_RETURN"
MISSING_UNRESOLVED_DELISTING = "MISSING_UNRESOLVED_DELISTING"
MISSING_FACTOR = "MISSING_FACTOR"
MISSING_NONFINITE_INPUT = "MISSING_NONFINITE_INPUT"
MISSING_MONTH_MINIMUM = "MISSING_MONTH_MINIMUM"
MISSING_ZERO_MAD = "MISSING_ZERO_MAD"
NO_ELIGIBLE_SECURITIES = "NO_ELIGIBLE_SECURITIES"

OUTCOME_PRECEDENCE = (
    NOT_LONG_ELIGIBLE,
    MISSING_BETA,
    MISSING_DUPLICATE_RETURN_LEG,
    MISSING_INTERVAL_STRADDLE,
    MISSING_MULTI_SESSION_RETURN,
    MISSING_UNRESOLVED_DELISTING,
    MISSING_RETURN_GAP,
    MISSING_NO_POST_ENTRY_RETURN,
    MISSING_UNUSABLE_RETURN,
    MISSING_FACTOR,
    MISSING_NONFINITE_INPUT,
)

FACTOR_NAMES = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
ORDINARY_LEG_CLASS = "ordinary_one_session"
TERMINAL_LEG_CLASS = "ordinary_plus_terminal_delisting_adjustment"
UNRESOLVED_LEG_CLASS = "ordinary_with_unresolved_terminal_delisting"
SUPPRESSED_LEG_CLASSES = frozenset(
    {
        "terminal_delisting_posting_audit_suppressed",
        "pre_scope_terminal_delisting_context_suppressed",
    }
)


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ForwardLeg:
    """One reconciled security-return row offered to a holding interval."""

    calendar_date: dt.date
    span_start_date: dt.date | None
    span_end_date: dt.date | None
    return_value: float | None
    outcome_availability_date: dt.date
    outcome_class: str
    return_history_usable: bool = True
    verified_return_pointer_authorized: bool = True
    covered_session_count: int | None = 1
    delisting_status: str = "not_terminal"

    @property
    def is_suppressed_posting(self) -> bool:
        return (
            self.outcome_class in SUPPRESSED_LEG_CLASSES
            or "suppressed" in self.outcome_class
        )

    @property
    def is_terminal(self) -> bool:
        return self.outcome_class == TERMINAL_LEG_CLASS

    @property
    def is_unresolved_delisting(self) -> bool:
        return (
            self.outcome_class == UNRESOLVED_LEG_CLASS
            or "unresolved" in self.outcome_class
            or "unresolved" in self.delisting_status
        )


@dataclass(frozen=True)
class FactorObservation:
    """The exact-date RF, FF5, and momentum factor observation."""

    calendar_date: dt.date
    rf: float | None
    mktrf: float | None
    smb: float | None
    hml: float | None
    rmw: float | None
    cma: float | None
    umd: float | None
    public_availability_date: dt.date | None = None

    def factor_values(self) -> tuple[float | None, ...]:
        return (self.mktrf, self.smb, self.hml, self.rmw, self.cma, self.umd)


@dataclass(frozen=True)
class ForwardContribution:
    calendar_date: dt.date
    return_leg_class: str
    return_value: float
    rf: float
    factor_values: tuple[float, ...]
    systematic_component: float
    excess_contribution: float
    residual_contribution: float
    outcome_availability_date: dt.date
    factor_public_availability_date: dt.date
    terminal_delisting: bool


@dataclass(frozen=True)
class ForwardOutcome:
    status: str
    residual_sum: float | None
    excess_sum: float | None
    available_date: dt.date | None
    expected_session_count: int
    used_session_count: int
    terminal_date: dt.date | None
    contributions: tuple[ForwardContribution, ...]

    @property
    def is_present(self) -> bool:
        return self.status == PRESENT


def _missing_forward_outcome(
    status: str,
    expected_session_count: int,
) -> ForwardOutcome:
    return ForwardOutcome(
        status=status,
        residual_sum=None,
        excess_sum=None,
        available_date=None,
        expected_session_count=expected_session_count,
        used_session_count=0,
        terminal_date=None,
        contributions=(),
    )


def _first_outcome_status(flags: set[str]) -> str:
    for status in OUTCOME_PRECEDENCE:
        if status in flags:
            return status
    return PRESENT


def evaluate_forward_outcome(
    entry_date: dt.date,
    exit_date: dt.date,
    required_sessions: Sequence[dt.date],
    betas: Sequence[float | None],
    legs_by_date: Mapping[dt.date, Sequence[ForwardLeg]],
    factors_by_date: Mapping[dt.date, FactorObservation],
    *,
    long_eligible: bool = True,
) -> ForwardOutcome:
    """Evaluate the literal Stage 7 daily residual/excess-return formulas.

    Required sessions must be the verified trading sessions strictly after the
    entry close through and including the exit close.  A valid terminal payoff
    ends the security path; later sessions represent cash and contribute zero.
    """

    sessions = tuple(required_sessions)
    expected_count = len(sessions)
    flags: set[str] = set()
    if not long_eligible:
        flags.add(NOT_LONG_ELIGIBLE)

    beta_values: tuple[float, ...] | None = None
    if len(betas) != len(FACTOR_NAMES) or not all(_finite(value) for value in betas):
        flags.add(MISSING_BETA)
    else:
        beta_values = tuple(float(value) for value in betas)

    if (
        entry_date >= exit_date
        or not sessions
        or tuple(sorted(set(sessions))) != sessions
        or sessions[0] <= entry_date
        or sessions[-1] != exit_date
        or any(date > exit_date for date in sessions)
    ):
        if not sessions:
            flags.add(MISSING_NO_POST_ENTRY_RETURN)
        else:
            flags.add(MISSING_INTERVAL_STRADDLE)

    for date, candidates in legs_by_date.items():
        if date <= entry_date and any(
            leg.is_terminal and not leg.is_suppressed_posting for leg in candidates
        ):
            flags.add(MISSING_UNUSABLE_RETURN)
        if date > exit_date and any(
            not leg.is_suppressed_posting
            and leg.span_start_date is not None
            and leg.span_end_date is not None
            and leg.span_start_date < exit_date < leg.span_end_date
            for leg in candidates
        ):
            flags.add(MISSING_INTERVAL_STRADDLE)

    if flags:
        status = _first_outcome_status(flags)
        if status in {
            NOT_LONG_ELIGIBLE,
            MISSING_BETA,
            MISSING_NO_POST_ENTRY_RETURN,
        }:
            return _missing_forward_outcome(status, expected_count)

    contributions: list[ForwardContribution] = []
    prior_session = entry_date
    terminal_date: dt.date | None = None
    path_ended = False
    offered_count = 0

    for calendar_date in sessions:
        if path_ended:
            break
        raw_candidates = tuple(legs_by_date.get(calendar_date, ()))
        candidates = tuple(
            leg for leg in raw_candidates if not leg.is_suppressed_posting
        )
        offered_count += len(candidates)
        if len(candidates) > 1:
            flags.add(MISSING_DUPLICATE_RETURN_LEG)
            prior_session = calendar_date
            continue
        if not candidates:
            flags.add(MISSING_RETURN_GAP)
            prior_session = calendar_date
            continue

        leg = candidates[0]
        ends_path = leg.is_terminal
        if leg.calendar_date != calendar_date:
            flags.add(MISSING_INTERVAL_STRADDLE)
        if (
            leg.span_start_date != prior_session
            or leg.span_end_date != calendar_date
        ):
            flags.add(MISSING_INTERVAL_STRADDLE)
        if (
            leg.covered_session_count != 1
            or leg.outcome_class == "ordinary_multi_session"
        ):
            flags.add(MISSING_MULTI_SESSION_RETURN)
        if leg.is_unresolved_delisting:
            flags.add(MISSING_UNRESOLVED_DELISTING)
        if leg.outcome_class not in {ORDINARY_LEG_CLASS, TERMINAL_LEG_CLASS}:
            if not leg.is_unresolved_delisting:
                flags.add(MISSING_UNUSABLE_RETURN)
        if (
            not leg.return_history_usable
            or not leg.verified_return_pointer_authorized
        ):
            flags.add(MISSING_UNUSABLE_RETURN)
        if leg.return_value is None:
            flags.add(MISSING_UNUSABLE_RETURN)
        elif not _finite(leg.return_value):
            flags.add(MISSING_NONFINITE_INPUT)
        elif float(leg.return_value) < -1.0:
            flags.add(MISSING_UNUSABLE_RETURN)

        factor = factors_by_date.get(calendar_date)
        if (
            factor is None
            or factor.calendar_date != calendar_date
            or (
                factor.public_availability_date is not None
                and factor.public_availability_date > calendar_date
            )
        ):
            flags.add(MISSING_FACTOR)
        elif not _finite(factor.rf) or not all(
            _finite(value) for value in factor.factor_values()
        ):
            flags.add(MISSING_NONFINITE_INPUT)

        if flags:
            path_ended = ends_path
            prior_session = calendar_date
            continue
        assert beta_values is not None
        assert factor is not None
        assert leg.return_value is not None
        factor_values = tuple(float(value) for value in factor.factor_values())
        rf = float(factor.rf)
        security_return = float(leg.return_value)
        systematic = math.fsum(
            beta * value for beta, value in zip(beta_values, factor_values, strict=True)
        )
        excess = math.fsum((security_return, -rf))
        residual = math.fsum((security_return, -rf, -systematic))
        if not all(_finite(value) for value in (systematic, excess, residual)):
            flags.add(MISSING_NONFINITE_INPUT)
            prior_session = calendar_date
            continue
        contribution = ForwardContribution(
            calendar_date=calendar_date,
            return_leg_class=leg.outcome_class,
            return_value=security_return,
            rf=rf,
            factor_values=factor_values,
            systematic_component=_positive_zero(systematic),
            excess_contribution=_positive_zero(excess),
            residual_contribution=_positive_zero(residual),
            outcome_availability_date=leg.outcome_availability_date,
            factor_public_availability_date=(
                factor.public_availability_date or factor.calendar_date
            ),
            terminal_delisting=leg.is_terminal,
        )
        contributions.append(contribution)
        if ends_path:
            terminal_date = calendar_date
            path_ended = True
        prior_session = calendar_date

    if offered_count == 0 and terminal_date is None:
        flags.discard(MISSING_RETURN_GAP)
        flags.add(MISSING_NO_POST_ENTRY_RETURN)
    status = _first_outcome_status(flags)
    if status != PRESENT:
        return _missing_forward_outcome(status, expected_count)

    if not contributions:
        return _missing_forward_outcome(MISSING_NO_POST_ENTRY_RETURN, expected_count)
    excess_sum = _positive_zero(
        math.fsum(item.excess_contribution for item in contributions)
    )
    residual_sum = _positive_zero(
        math.fsum(item.residual_contribution for item in contributions)
    )
    available_date = max(
        exit_date,
        *(item.outcome_availability_date for item in contributions),
    )
    return ForwardOutcome(
        status=PRESENT,
        residual_sum=residual_sum,
        excess_sum=excess_sum,
        available_date=available_date,
        expected_session_count=expected_count,
        used_session_count=len(contributions),
        terminal_date=terminal_date,
        contributions=tuple(contributions),
    )


@dataclass(frozen=True)
class RankResult:
    values: tuple[float | None, ...]
    numerators: tuple[int | None, ...]
    finite_count: int
    numerator_sum: int


def centered_average_rank(
    values: Sequence[float | None],
) -> RankResult:
    """Return exact-tie centered ranks and their integer audit numerators."""

    ranked_values: list[float | None] = [None] * len(values)
    numerators: list[int | None] = [None] * len(values)
    finite = [
        (float(value), index)
        for index, value in enumerate(values)
        if value is not None and _finite(value)
    ]
    finite.sort(key=lambda item: (item[0], item[1]))
    count = len(finite)
    position = 0
    while position < count:
        end = position + 1
        while end < count and finite[end][0] == finite[position][0]:
            end += 1
        numerator = position + end - count
        rank = _positive_zero(numerator / (2.0 * count))
        for _, original_index in finite[position:end]:
            ranked_values[original_index] = rank
            numerators[original_index] = numerator
        position = end
    numerator_sum = sum(
        value for value in numerators if value is not None
    )
    return RankResult(
        values=tuple(ranked_values),
        numerators=tuple(numerators),
        finite_count=count,
        numerator_sum=numerator_sum,
    )


def _ordinary_median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return float(ordered[middle])
    return _positive_zero(math.fsum((ordered[middle - 1], ordered[middle])) / 2.0)


@dataclass(frozen=True)
class MadResult:
    values: tuple[float | None, ...]
    median: float | None
    mad: float | None
    status: str


def mad_standardized(
    values: Sequence[float | None],
) -> MadResult:
    """Median/MAD standardize finite values with no epsilon and a ±5 clip."""

    finite = [
        float(value)
        for value in values
        if value is not None and _finite(value)
    ]
    if not finite:
        return MadResult(
            values=tuple(None for _ in values),
            median=None,
            mad=None,
            status=MISSING_NONFINITE_INPUT,
        )
    median = _ordinary_median(finite)
    mad = _ordinary_median([abs(value - median) for value in finite])
    if mad == 0.0:
        return MadResult(
            values=tuple(None for _ in values),
            median=_positive_zero(median),
            mad=0.0,
            status=MISSING_ZERO_MAD,
        )
    denominator = 1.4826 * mad
    standardized: list[float | None] = []
    for value in values:
        if value is None or not _finite(value):
            standardized.append(None)
            continue
        result = (float(value) - median) / denominator
        standardized.append(_positive_zero(min(5.0, max(-5.0, result))))
    return MadResult(
        values=tuple(standardized),
        median=_positive_zero(median),
        mad=_positive_zero(mad),
        status=PRESENT,
    )


@dataclass(frozen=True)
class MonthTargets:
    residual_rank: tuple[float | None, ...]
    excess_rank: tuple[float | None, ...]
    residual_rank_numerators: tuple[int | None, ...]
    excess_rank_numerators: tuple[int | None, ...]
    residual_mad_standardized: tuple[float | None, ...]
    valid_count: int
    month_status: str
    magnitude_status: str
    residual_median: float | None
    residual_mad: float | None
    residual_rank_numerator_sum: int | None
    excess_rank_numerator_sum: int | None


def build_month_targets(
    residuals: Sequence[float | None],
    excesses: Sequence[float | None],
    valid_mask: Sequence[bool],
    long_eligible: Sequence[bool],
    *,
    minimum_count: int = 100,
) -> MonthTargets:
    """Build all three targets from one primary-valid long-only population."""

    size = len(residuals)
    if not (
        len(excesses) == size
        and len(valid_mask) == size
        and len(long_eligible) == size
    ):
        raise ValueError("Stage 7 month-target inputs have unequal lengths")
    if isinstance(minimum_count, bool) or minimum_count <= 0:
        raise ValueError("Stage 7 minimum label count must be positive")
    eligible_count = sum(bool(value) for value in long_eligible)
    common = tuple(
        bool(valid_mask[index]) and bool(long_eligible[index])
        for index in range(size)
    )
    valid_count = sum(common)
    for index, included in enumerate(common):
        if included and (
            residuals[index] is None
            or excesses[index] is None
            or not _finite(residuals[index])
            or not _finite(excesses[index])
        ):
            raise ValueError(
                "Primary-valid Stage 7 row has a missing/nonfinite outcome"
            )

    empty_values = tuple(None for _ in range(size))
    if eligible_count == 0:
        return MonthTargets(
            residual_rank=empty_values,
            excess_rank=empty_values,
            residual_rank_numerators=empty_values,
            excess_rank_numerators=empty_values,
            residual_mad_standardized=empty_values,
            valid_count=0,
            month_status=NO_ELIGIBLE_SECURITIES,
            magnitude_status=MISSING_MONTH_MINIMUM,
            residual_median=None,
            residual_mad=None,
            residual_rank_numerator_sum=None,
            excess_rank_numerator_sum=None,
        )
    if valid_count < minimum_count:
        return MonthTargets(
            residual_rank=empty_values,
            excess_rank=empty_values,
            residual_rank_numerators=empty_values,
            excess_rank_numerators=empty_values,
            residual_mad_standardized=empty_values,
            valid_count=valid_count,
            month_status=MISSING_MONTH_MINIMUM,
            magnitude_status=MISSING_MONTH_MINIMUM,
            residual_median=None,
            residual_mad=None,
            residual_rank_numerator_sum=None,
            excess_rank_numerator_sum=None,
        )

    residual_population = tuple(
        residuals[index] if common[index] else None for index in range(size)
    )
    excess_population = tuple(
        excesses[index] if common[index] else None for index in range(size)
    )
    residual_rank = centered_average_rank(residual_population)
    excess_rank = centered_average_rank(excess_population)
    residual_mad = mad_standardized(residual_population)
    if (
        residual_rank.finite_count != valid_count
        or excess_rank.finite_count != valid_count
        or residual_rank.numerator_sum != 0
        or excess_rank.numerator_sum != 0
    ):
        raise ArithmeticError("Stage 7 rank population/audit identity failed")
    return MonthTargets(
        residual_rank=residual_rank.values,
        excess_rank=excess_rank.values,
        residual_rank_numerators=residual_rank.numerators,
        excess_rank_numerators=excess_rank.numerators,
        residual_mad_standardized=residual_mad.values,
        valid_count=valid_count,
        month_status=PRESENT,
        magnitude_status=residual_mad.status,
        residual_median=residual_mad.median,
        residual_mad=residual_mad.mad,
        residual_rank_numerator_sum=residual_rank.numerator_sum,
        excess_rank_numerator_sum=excess_rank.numerator_sum,
    )


def security_label_available_date(
    exit_date: dt.date,
    contributing_outcome_availability_dates: Sequence[dt.date],
) -> dt.date | None:
    if not contributing_outcome_availability_dates:
        return None
    return max(exit_date, *contributing_outcome_availability_dates)


def month_label_available_date(
    security_available_dates: Sequence[dt.date | None],
    primary_valid: Sequence[bool],
    long_eligible: Sequence[bool],
) -> dt.date | None:
    if not (
        len(security_available_dates) == len(primary_valid) == len(long_eligible)
    ):
        raise ValueError("Stage 7 month availability inputs have unequal lengths")
    dates = [
        date
        for date, valid, eligible in zip(
            security_available_dates,
            primary_valid,
            long_eligible,
            strict=True,
        )
        if valid and eligible and date is not None
    ]
    if any(
        valid and eligible and date is None
        for date, valid, eligible in zip(
            security_available_dates,
            primary_valid,
            long_eligible,
            strict=True,
        )
    ):
        raise ValueError(
            "Primary-valid long Stage 7 row has no availability date"
        )
    return max(dates) if dates else None


def label_is_available(
    month_available_date: dt.date | None,
    fit_signal_date: dt.date,
) -> bool:
    return (
        month_available_date is not None
        and month_available_date <= fit_signal_date
    )


def derive_label_statuses(
    outcome_status: str,
    month_status: str,
    *,
    zero_mad: bool,
) -> tuple[str, str]:
    """Apply the frozen outcome -> primary -> magnitude status precedence."""

    if outcome_status != PRESENT:
        return outcome_status, outcome_status
    if month_status != PRESENT:
        return MISSING_MONTH_MINIMUM, MISSING_MONTH_MINIMUM
    if zero_mad:
        return PRESENT, MISSING_ZERO_MAD
    return PRESENT, PRESENT


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    fit_signal_date: dt.date
    train_last_holding_month: str
    purged_holding_month: str
    validation_first_holding_month: str
    validation_last_holding_month: str


@dataclass(frozen=True)
class SplitAssignment:
    fold_id: str
    fit_signal_date: dt.date
    role: str


DEVELOPMENT_FOLDS = (
    FoldSpec(
        "dev_fold_1",
        dt.date(2006, 12, 29),
        "2006-11",
        "2006-12",
        "2007-01",
        "2008-12",
    ),
    FoldSpec(
        "dev_fold_2",
        dt.date(2008, 12, 31),
        "2008-11",
        "2008-12",
        "2009-01",
        "2010-12",
    ),
    FoldSpec(
        "dev_fold_3",
        dt.date(2010, 12, 31),
        "2010-11",
        "2010-12",
        "2011-01",
        "2012-12",
    ),
    FoldSpec(
        "dev_fold_4",
        dt.date(2012, 12, 31),
        "2012-11",
        "2012-12",
        "2013-01",
        "2014-11",
    ),
)


def _validate_holding_month(value: str) -> str:
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Stage 7 holding month: {value!r}") from exc
    normalized = parsed.strftime("%Y-%m")
    if normalized != value:
        raise ValueError(f"Invalid Stage 7 holding month: {value!r}")
    return normalized


def assign_split_membership(
    holding_month: str,
) -> tuple[SplitAssignment, ...]:
    """Assign one holding month to each frozen expanding-development fold."""

    month = _validate_holding_month(holding_month)
    result: list[SplitAssignment] = []
    for fold in DEVELOPMENT_FOLDS:
        if month <= fold.train_last_holding_month:
            role = "TRAIN_ELIGIBLE"
        elif month == fold.purged_holding_month:
            role = "PURGED"
        elif (
            fold.validation_first_holding_month
            <= month
            <= fold.validation_last_holding_month
        ):
            role = "VALIDATION"
        else:
            role = "OUTSIDE"
        result.append(
            SplitAssignment(
                fold_id=fold.fold_id,
                fit_signal_date=fold.fit_signal_date,
                role=role,
            )
        )
    return tuple(result)
