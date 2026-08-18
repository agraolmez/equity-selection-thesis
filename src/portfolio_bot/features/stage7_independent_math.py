"""Clean-room numerical primitives for independent Stage 7 verification.

This module deliberately does not import the Stage 6 feature builders or any
Stage 7 primary computation module.  Its constants and formulas are restated
from the frozen Stage 7 rules so the verifier has a genuinely independent
calculation path.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


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

EWLS_LOOKBACK_SESSIONS = 252
EWLS_MINIMUM_OBSERVATIONS = 126
EWLS_HALF_LIFE_SESSIONS = 126.0
EWLS_RCOND = 1e-12
EWLS_REQUIRED_RANK = 7
BETA_CLIP_LOWER = 0.01
BETA_CLIP_UPPER = 0.99
MAD_SCALE = 1.4826
MAD_LOWER_CLIP = -5.0
MAD_UPPER_CLIP = 5.0
MINIMUM_VALID_LABELS = 100
DEVELOPMENT_EMBARGO_HOLDING_MONTH = "2014-12"


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def type7_quantile(
    values: Sequence[float | None],
    probability: float,
) -> float | None:
    """Return a Hyndman--Fan type-7 quantile over finite values."""

    if not _finite(probability) or not 0.0 <= float(probability) <= 1.0:
        raise ValueError("Type-7 probability must be finite and within [0, 1]")
    ordered = sorted(
        float(value) for value in values if value is not None and _finite(value)
    )
    if not ordered:
        return None
    if len(ordered) == 1:
        return _positive_zero(ordered[0])
    position = (len(ordered) - 1) * float(probability)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return _positive_zero(ordered[lower_index])
    fraction = position - lower_index
    result = math.fsum(
        (
            ordered[lower_index],
            fraction * (ordered[upper_index] - ordered[lower_index]),
        )
    )
    if not math.isfinite(result):
        raise ArithmeticError("Type-7 quantile produced a nonfinite result")
    return _positive_zero(result)


@dataclass(frozen=True)
class IndependentEwlsResult:
    intercept: float
    slopes: tuple[float, ...]
    observation_count: int


def fit_ewls_factor_model(
    security_returns: Sequence[float],
    risk_free_rates: Sequence[float],
    factors: Sequence[Sequence[float]] | np.ndarray,
    ages: Sequence[int | float],
    *,
    half_life_sessions: float = EWLS_HALF_LIFE_SESSIONS,
    minimum_observations: int = EWLS_MINIMUM_OBSERVATIONS,
    rcond: float = EWLS_RCOND,
) -> IndependentEwlsResult | None:
    """Fit the frozen six-factor, intercept-bearing sqrt-weighted EWLS."""

    if (
        isinstance(minimum_observations, bool)
        or not isinstance(minimum_observations, int)
        or minimum_observations <= 0
        or not _finite(half_life_sessions)
        or float(half_life_sessions) <= 0.0
        or not _finite(rcond)
        or float(rcond) < 0.0
    ):
        return None
    try:
        returns = np.asarray(security_returns, dtype=np.float64)
        rf = np.asarray(risk_free_rates, dtype=np.float64)
        factor_matrix = np.asarray(factors, dtype=np.float64)
        age_vector = np.asarray(ages, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        returns.ndim != 1
        or rf.ndim != 1
        or age_vector.ndim != 1
        or factor_matrix.ndim != 2
        or factor_matrix.shape[1] != len(FACTOR_NAMES)
        or returns.size != rf.size
        or returns.size != age_vector.size
        or returns.size != factor_matrix.shape[0]
        or returns.size < minimum_observations
        or returns.size > EWLS_LOOKBACK_SESSIONS
        or not np.isfinite(returns).all()
        or not np.isfinite(rf).all()
        or not np.isfinite(factor_matrix).all()
        or not np.isfinite(age_vector).all()
        or np.any(age_vector < 0.0)
        or np.any(age_vector > EWLS_LOOKBACK_SESSIONS - 1)
        or np.any(age_vector != np.floor(age_vector))
        or np.unique(age_vector).size != age_vector.size
    ):
        return None
    design = np.column_stack((np.ones(returns.size, dtype=np.float64), factor_matrix))
    dependent = returns - rf
    weights = np.exp2(-age_vector / float(half_life_sessions))
    roots = np.sqrt(weights)
    weighted_design = design * roots[:, None]
    weighted_dependent = dependent * roots
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(
            weighted_design,
            weighted_dependent,
            rcond=float(rcond),
        )
    except np.linalg.LinAlgError:
        return None
    if rank != EWLS_REQUIRED_RANK or not np.isfinite(coefficients).all():
        return None
    return IndependentEwlsResult(
        intercept=_positive_zero(float(coefficients[0])),
        slopes=tuple(_positive_zero(float(value)) for value in coefficients[1:]),
        observation_count=int(returns.size),
    )


@dataclass(frozen=True)
class IndependentBetaClipResult:
    slopes: tuple[tuple[float | None, ...], ...]
    lower_bounds: tuple[float | None, ...]
    upper_bounds: tuple[float | None, ...]


def clip_beta_slopes(
    slopes: Sequence[Sequence[float | None]],
    long_eligible: Sequence[bool],
    *,
    lower_probability: float = BETA_CLIP_LOWER,
    upper_probability: float = BETA_CLIP_UPPER,
) -> IndependentBetaClipResult:
    """Clip each factor using finite long-only bounds, then apply to all rows."""

    if len(slopes) != len(long_eligible):
        raise ValueError("Beta rows and eligibility flags have unequal lengths")
    if (
        not _finite(lower_probability)
        or not _finite(upper_probability)
        or not 0.0 <= float(lower_probability) <= float(upper_probability) <= 1.0
    ):
        raise ValueError("Invalid beta clipping probabilities")
    normalized: list[tuple[float | None, ...]] = []
    for row in slopes:
        if len(row) != len(FACTOR_NAMES):
            raise ValueError("Every beta row must contain six factor slopes")
        normalized.append(
            tuple(
                float(value) if value is not None and _finite(value) else None
                for value in row
            )
        )
    lower: list[float | None] = []
    upper: list[float | None] = []
    for factor_index in range(len(FACTOR_NAMES)):
        population = [
            row[factor_index]
            for row, eligible in zip(normalized, long_eligible, strict=True)
            if bool(eligible) and row[factor_index] is not None
        ]
        lower.append(type7_quantile(population, float(lower_probability)))
        upper.append(type7_quantile(population, float(upper_probability)))
    clipped: list[tuple[float | None, ...]] = []
    for row in normalized:
        output: list[float | None] = []
        for value, low, high in zip(row, lower, upper, strict=True):
            if value is None or low is None or high is None:
                output.append(None)
            else:
                output.append(_positive_zero(min(high, max(low, value))))
        clipped.append(tuple(output))
    return IndependentBetaClipResult(
        slopes=tuple(clipped),
        lower_bounds=tuple(lower),
        upper_bounds=tuple(upper),
    )


@dataclass(frozen=True)
class IndependentForwardLeg:
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
class IndependentFactorObservation:
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
class IndependentForwardContribution:
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
class IndependentForwardOutcome:
    status: str
    residual_sum: float | None
    excess_sum: float | None
    available_date: dt.date | None
    expected_session_count: int
    used_session_count: int
    terminal_date: dt.date | None
    contributions: tuple[IndependentForwardContribution, ...]

    @property
    def is_present(self) -> bool:
        return self.status == PRESENT


def _missing_outcome(
    status: str,
    expected_session_count: int,
) -> IndependentForwardOutcome:
    return IndependentForwardOutcome(
        status=status,
        residual_sum=None,
        excess_sum=None,
        available_date=None,
        expected_session_count=expected_session_count,
        used_session_count=0,
        terminal_date=None,
        contributions=(),
    )


def _outcome_status(flags: set[str]) -> str:
    for status in OUTCOME_PRECEDENCE:
        if status in flags:
            return status
    return PRESENT


def evaluate_forward_outcome(
    entry_date: dt.date,
    exit_date: dt.date,
    required_sessions: Sequence[dt.date],
    betas: Sequence[float | None],
    legs_by_date: Mapping[dt.date, Sequence[IndependentForwardLeg]],
    factors_by_date: Mapping[dt.date, IndependentFactorObservation],
    *,
    long_eligible: bool = True,
) -> IndependentForwardOutcome:
    """Reconstruct one frozen arithmetic forward excess/residual outcome."""

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

    interval_is_valid = (
        entry_date < exit_date
        and bool(sessions)
        and tuple(sorted(set(sessions))) == sessions
        and sessions[0] > entry_date
        and sessions[-1] == exit_date
        and all(date <= exit_date for date in sessions)
    )
    if not interval_is_valid:
        flags.add(
            MISSING_NO_POST_ENTRY_RETURN if not sessions else MISSING_INTERVAL_STRADDLE
        )

    for date, raw_candidates in legs_by_date.items():
        candidates = tuple(raw_candidates)
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

    first_status = _outcome_status(flags)
    if first_status in {
        NOT_LONG_ELIGIBLE,
        MISSING_BETA,
        MISSING_NO_POST_ENTRY_RETURN,
    }:
        return _missing_outcome(first_status, expected_count)

    contributions: list[IndependentForwardContribution] = []
    prior_session = entry_date
    terminal_date: dt.date | None = None
    path_ended = False
    offered_count = 0

    for calendar_date in sessions:
        if path_ended:
            break
        candidates = tuple(
            leg
            for leg in legs_by_date.get(calendar_date, ())
            if not leg.is_suppressed_posting
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
        if leg.span_start_date != prior_session or leg.span_end_date != calendar_date:
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
        if not leg.return_history_usable or not leg.verified_return_pointer_authorized:
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
        return_value = float(leg.return_value)
        systematic = math.fsum(
            beta * factor_value
            for beta, factor_value in zip(beta_values, factor_values, strict=True)
        )
        excess = math.fsum((return_value, -rf))
        residual = math.fsum((return_value, -rf, -systematic))
        if not all(_finite(value) for value in (systematic, excess, residual)):
            flags.add(MISSING_NONFINITE_INPUT)
            prior_session = calendar_date
            continue
        contributions.append(
            IndependentForwardContribution(
                calendar_date=calendar_date,
                return_leg_class=leg.outcome_class,
                return_value=return_value,
                rf=rf,
                factor_values=factor_values,
                systematic_component=_positive_zero(systematic),
                excess_contribution=_positive_zero(excess),
                residual_contribution=_positive_zero(residual),
                outcome_availability_date=leg.outcome_availability_date,
                factor_public_availability_date=(
                    factor.public_availability_date or factor.calendar_date
                ),
                terminal_delisting=ends_path,
            )
        )
        if ends_path:
            terminal_date = calendar_date
            path_ended = True
        prior_session = calendar_date

    if offered_count == 0 and terminal_date is None:
        flags.discard(MISSING_RETURN_GAP)
        flags.add(MISSING_NO_POST_ENTRY_RETURN)
    status = _outcome_status(flags)
    if status != PRESENT:
        return _missing_outcome(status, expected_count)
    if not contributions:
        return _missing_outcome(MISSING_NO_POST_ENTRY_RETURN, expected_count)
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
    return IndependentForwardOutcome(
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
class IndependentRankResult:
    values: tuple[float | None, ...]
    numerators: tuple[int | None, ...]
    finite_count: int
    numerator_sum: int


def centered_average_rank(
    values: Sequence[float | None],
) -> IndependentRankResult:
    """Return exact-tie centered ranks and integer audit numerators."""

    ranked: list[float | None] = [None] * len(values)
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
        value = _positive_zero(numerator / (2.0 * count))
        for _, original_index in finite[position:end]:
            ranked[original_index] = value
            numerators[original_index] = numerator
        position = end
    numerator_sum = sum(value for value in numerators if value is not None)
    return IndependentRankResult(
        values=tuple(ranked),
        numerators=tuple(numerators),
        finite_count=count,
        numerator_sum=numerator_sum,
    )


def _ordinary_median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Median requires at least one value")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return _positive_zero(ordered[middle])
    return _positive_zero(math.fsum((ordered[middle - 1], ordered[middle])) / 2.0)


@dataclass(frozen=True)
class IndependentMadResult:
    values: tuple[float | None, ...]
    median: float | None
    mad: float | None
    status: str


def mad_standardized(
    values: Sequence[float | None],
) -> IndependentMadResult:
    """Median/MAD standardize finite values, without epsilon, clipped to ±5."""

    finite = [float(value) for value in values if value is not None and _finite(value)]
    if not finite:
        return IndependentMadResult(
            values=tuple(None for _ in values),
            median=None,
            mad=None,
            status=MISSING_NONFINITE_INPUT,
        )
    median = _ordinary_median(finite)
    mad = _ordinary_median([abs(value - median) for value in finite])
    if mad == 0.0:
        return IndependentMadResult(
            values=tuple(None for _ in values),
            median=_positive_zero(median),
            mad=0.0,
            status=MISSING_ZERO_MAD,
        )
    denominator = MAD_SCALE * mad
    standardized: list[float | None] = []
    for value in values:
        if value is None or not _finite(value):
            standardized.append(None)
            continue
        result = (float(value) - median) / denominator
        standardized.append(
            _positive_zero(min(MAD_UPPER_CLIP, max(MAD_LOWER_CLIP, result)))
        )
    return IndependentMadResult(
        values=tuple(standardized),
        median=_positive_zero(median),
        mad=_positive_zero(mad),
        status=PRESENT,
    )


@dataclass(frozen=True)
class IndependentMonthTargets:
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
    minimum_count: int = MINIMUM_VALID_LABELS,
) -> IndependentMonthTargets:
    """Build all frozen targets from one primary-valid, long-only mask."""

    size = len(residuals)
    if not (
        len(excesses) == size and len(valid_mask) == size and len(long_eligible) == size
    ):
        raise ValueError("Month-target inputs have unequal lengths")
    if (
        isinstance(minimum_count, bool)
        or not isinstance(minimum_count, int)
        or minimum_count <= 0
    ):
        raise ValueError("Minimum label count must be a positive integer")
    eligible_count = sum(bool(value) for value in long_eligible)
    common = tuple(
        bool(valid_mask[index]) and bool(long_eligible[index]) for index in range(size)
    )
    valid_count = sum(common)
    for index, included in enumerate(common):
        if included and (
            residuals[index] is None
            or excesses[index] is None
            or not _finite(residuals[index])
            or not _finite(excesses[index])
        ):
            raise ValueError("Primary-valid row has a missing/nonfinite outcome")

    empty = tuple(None for _ in range(size))
    if eligible_count == 0:
        return IndependentMonthTargets(
            residual_rank=empty,
            excess_rank=empty,
            residual_rank_numerators=empty,
            excess_rank_numerators=empty,
            residual_mad_standardized=empty,
            valid_count=0,
            month_status=NO_ELIGIBLE_SECURITIES,
            magnitude_status=MISSING_MONTH_MINIMUM,
            residual_median=None,
            residual_mad=None,
            residual_rank_numerator_sum=None,
            excess_rank_numerator_sum=None,
        )
    if valid_count < minimum_count:
        return IndependentMonthTargets(
            residual_rank=empty,
            excess_rank=empty,
            residual_rank_numerators=empty,
            excess_rank_numerators=empty,
            residual_mad_standardized=empty,
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
    magnitude = mad_standardized(residual_population)
    if (
        residual_rank.finite_count != valid_count
        or excess_rank.finite_count != valid_count
        or residual_rank.numerator_sum != 0
        or excess_rank.numerator_sum != 0
    ):
        raise ArithmeticError("Independent rank population identity failed")
    return IndependentMonthTargets(
        residual_rank=residual_rank.values,
        excess_rank=excess_rank.values,
        residual_rank_numerators=residual_rank.numerators,
        excess_rank_numerators=excess_rank.numerators,
        residual_mad_standardized=magnitude.values,
        valid_count=valid_count,
        month_status=PRESENT,
        magnitude_status=magnitude.status,
        residual_median=magnitude.median,
        residual_mad=magnitude.mad,
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
    if not (len(security_available_dates) == len(primary_valid) == len(long_eligible)):
        raise ValueError("Month-availability inputs have unequal lengths")
    result: list[dt.date] = []
    for date, valid, eligible in zip(
        security_available_dates,
        primary_valid,
        long_eligible,
        strict=True,
    ):
        if bool(valid) and bool(eligible):
            if date is None:
                raise ValueError("Primary-valid long row has no availability date")
            result.append(date)
    return max(result) if result else None


def label_is_available(
    month_available_date: dt.date | None,
    fit_signal_date: dt.date,
) -> bool:
    return month_available_date is not None and month_available_date <= fit_signal_date


def derive_label_statuses(
    outcome_status: str,
    month_status: str,
    *,
    zero_mad: bool,
) -> tuple[str, str]:
    if outcome_status != PRESENT:
        return outcome_status, outcome_status
    if month_status != PRESENT:
        return MISSING_MONTH_MINIMUM, MISSING_MONTH_MINIMUM
    if zero_mad:
        return PRESENT, MISSING_ZERO_MAD
    return PRESENT, PRESENT


@dataclass(frozen=True)
class IndependentFoldSpec:
    fold_id: str
    fit_signal_date: dt.date
    train_last_holding_month: str
    purged_holding_month: str
    validation_first_holding_month: str
    validation_last_holding_month: str


@dataclass(frozen=True)
class IndependentSplitAssignment:
    fold_id: str
    fit_signal_date: dt.date
    role: str


@dataclass(frozen=True)
class IndependentSplitMembership:
    fold_id: str
    fit_signal_date: dt.date
    role: str
    available_by_fit: bool
    included_in_training: bool
    included_in_validation: bool
    purged_or_embargoed: bool


DEVELOPMENT_FOLDS = (
    IndependentFoldSpec(
        "dev_fold_1",
        dt.date(2006, 12, 29),
        "2006-11",
        "2006-12",
        "2007-01",
        "2008-12",
    ),
    IndependentFoldSpec(
        "dev_fold_2",
        dt.date(2008, 12, 31),
        "2008-11",
        "2008-12",
        "2009-01",
        "2010-12",
    ),
    IndependentFoldSpec(
        "dev_fold_3",
        dt.date(2010, 12, 31),
        "2010-11",
        "2010-12",
        "2011-01",
        "2012-12",
    ),
    IndependentFoldSpec(
        "dev_fold_4",
        dt.date(2012, 12, 31),
        "2012-11",
        "2012-12",
        "2013-01",
        "2014-11",
    ),
)


def _holding_month(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Invalid holding month: {value!r}")
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise ValueError(f"Invalid holding month: {value!r}") from exc
    if parsed.strftime("%Y-%m") != value:
        raise ValueError(f"Invalid holding month: {value!r}")
    return value


def assign_split_membership(
    holding_month: str,
    *,
    folds: Sequence[IndependentFoldSpec] = DEVELOPMENT_FOLDS,
) -> tuple[IndependentSplitAssignment, ...]:
    """Assign a holding month to every frozen development fold."""

    month = _holding_month(holding_month)
    result: list[IndependentSplitAssignment] = []
    for fold in folds:
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
            IndependentSplitAssignment(
                fold_id=fold.fold_id,
                fit_signal_date=fold.fit_signal_date,
                role=role,
            )
        )
    return tuple(result)


def build_split_membership(
    holding_month: str,
    *,
    month_valid: bool,
    month_available_date: dt.date | None,
    folds: Sequence[IndependentFoldSpec] = DEVELOPMENT_FOLDS,
    embargo_holding_month: str = DEVELOPMENT_EMBARGO_HOLDING_MONTH,
) -> tuple[IndependentSplitMembership, ...]:
    """Derive frozen availability, training, validation, and purge flags."""

    month = _holding_month(holding_month)
    embargo = _holding_month(embargo_holding_month)
    assignments = assign_split_membership(month, folds=folds)
    result: list[IndependentSplitMembership] = []
    for assignment in assignments:
        available = label_is_available(
            month_available_date,
            assignment.fit_signal_date,
        )
        result.append(
            IndependentSplitMembership(
                fold_id=assignment.fold_id,
                fit_signal_date=assignment.fit_signal_date,
                role=assignment.role,
                available_by_fit=available,
                included_in_training=(
                    assignment.role == "TRAIN_ELIGIBLE"
                    and bool(month_valid)
                    and available
                ),
                included_in_validation=(
                    assignment.role == "VALIDATION" and bool(month_valid)
                ),
                purged_or_embargoed=(assignment.role == "PURGED" or month == embargo),
            )
        )
    return tuple(result)
