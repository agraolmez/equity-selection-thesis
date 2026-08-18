"""Dynamic monthly Stage 11 adapter into the frozen Stage 10 allocator.

This module is pure and table-agnostic.  It joins an already-scored monthly
cross-section to current market/risk inputs and inherited holdings, then calls
the existing :class:`CandidateUniverse` and
``CandidateSelection.to_problem_data`` APIs through a Stage 11 entry-gating
adapter.  No storage access is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Sequence

import numpy as np

from portfolio_bot.allocation.candidates import (
    DEFAULT_MINIMUM_CANDIDATES,
    DEFAULT_TAIL_FRACTION,
)
from portfolio_bot.allocation.domain import (
    FACTOR_COUNT,
    PRIMARY_BORROW_SCENARIO,
    UNKNOWN_COUNTRY,
    UNKNOWN_SECTOR,
    AllocationProblemData,
    CandidateSelection,
    CandidateUniverse,
)
from portfolio_bot.validation.candidate_adapter import (
    form_stage11_monthly_candidates,
)


MAXIMUM_RISK_CARRY_SESSIONS = 21
MAXIMUM_ADV_TRADE_FRACTION = 0.10
SCORE_CENTERING_TOLERANCE = 1e-8

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ProblemBuilderError(ValueError):
    """Raised when a monthly problem cannot be assembled without invention."""


def _vector(
    value: object,
    *,
    dtype: np.dtype | type,
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=dtype).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProblemBuilderError(f"{name} cannot be converted") from exc
    if result.ndim != 1:
        raise ProblemBuilderError(f"{name} must be one-dimensional")
    result.setflags(write=False)
    return result


def _bool_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"b", "i", "u"}:
        raise ProblemBuilderError(f"{name} must be a boolean vector")
    if raw.dtype.kind != "b" and not np.isin(raw, (0, 1)).all():
        raise ProblemBuilderError(f"{name} must contain only boolean values")
    return _vector(raw, dtype=np.bool_, name=name)


def _matrix(value: object, *, rows: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProblemBuilderError(f"{name} cannot be converted") from exc
    if result.shape != (rows, FACTOR_COUNT):
        raise ProblemBuilderError(f"{name} must have shape ({rows}, {FACTOR_COUNT})")
    result.setflags(write=False)
    return result


def _text_vector(
    value: Sequence[object] | np.ndarray,
    *,
    rows: int,
    name: str,
    missing: str,
) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.ndim != 1 or len(raw) != rows:
        raise ProblemBuilderError(f"{name} must be a length-{rows} vector")
    result = np.asarray(
        [
            str(item).strip().upper() if item is not None and str(item).strip() else missing
            for item in raw
        ],
        dtype=np.str_,
    )
    result.setflags(write=False)
    return result


def _same_length(expected: int, **values: np.ndarray) -> None:
    mismatch = sorted(name for name, value in values.items() if len(value) != expected)
    if mismatch:
        raise ProblemBuilderError("input lengths differ for: " + ", ".join(mismatch))


def _identity(permno: np.ndarray, *, name: str) -> None:
    if np.any(permno <= 0) or len(np.unique(permno)) != len(permno):
        raise ProblemBuilderError(f"{name} PERMNO values must be positive and unique")


@dataclass(frozen=True, slots=True)
class AuthenticatedMonthlyIdentity:
    """Hash-bound identity shared by every input to one monthly decision.

    The component tables may come from different files, but they must have
    been sealed into the same as-of bundle.  Requiring the exact same signal
    date and bundle digest prevents a caller from mixing snapshots from
    different months or publication runs.
    """

    signal_date: date
    source_bundle_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.signal_date, datetime) or not isinstance(
            self.signal_date, date
        ):
            raise ProblemBuilderError("identity.signal_date must be a date")
        if not isinstance(self.source_bundle_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.source_bundle_sha256
        ):
            raise ProblemBuilderError(
                "identity.source_bundle_sha256 must be a lowercase SHA-256 digest"
            )


@dataclass(frozen=True, slots=True, eq=False)
class ScoredSnapshot:
    """Current score/alpha cross-section before execution-cost gating."""

    identity: AuthenticatedMonthlyIdentity
    permno: np.ndarray
    score: np.ndarray
    calibration_kappa_monthly: float
    long_eligible: np.ndarray
    short_eligible: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.identity, AuthenticatedMonthlyIdentity):
            raise ProblemBuilderError(
                "scored.identity must be AuthenticatedMonthlyIdentity"
            )
        permno = _vector(self.permno, dtype=np.int64, name="scored.permno")
        score = _vector(self.score, dtype=np.float64, name="scored.score")
        try:
            kappa = float(self.calibration_kappa_monthly)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProblemBuilderError(
                "scored.calibration_kappa_monthly must be numeric"
            ) from exc
        long = _bool_vector(self.long_eligible, name="scored.long_eligible")
        short = _bool_vector(self.short_eligible, name="scored.short_eligible")
        _same_length(
            len(permno),
            score=score,
            long_eligible=long,
            short_eligible=short,
        )
        _identity(permno, name="scored")
        if np.any(short & ~long):
            raise ProblemBuilderError("short eligibility must be a subset of long")
        if not np.isfinite(score[long]).all():
            raise ProblemBuilderError("eligible scores must be finite")
        if np.isinf(score).any():
            raise ProblemBuilderError("scores may be finite or NaN, not infinite")
        if np.any(np.abs(score[long]) > 0.5 + 1e-12):
            raise ProblemBuilderError("eligible scores must lie within [-0.5, 0.5]")
        if long.any() and abs(float(np.mean(score[long]))) > SCORE_CENTERING_TOLERANCE:
            raise ProblemBuilderError("eligible scores must be cross-sectionally centered")
        if not np.isfinite(kappa) or kappa < 0.0:
            raise ProblemBuilderError(
                "scored.calibration_kappa_monthly must be finite and nonnegative"
            )
        for name, value in (
            ("permno", permno),
            ("score", score),
            ("calibration_kappa_monthly", 0.0 if kappa == 0.0 else kappa),
            ("long_eligible", long),
            ("short_eligible", short),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True, eq=False)
class MarketSnapshot:
    """Current execution, cost, classification, and risk observations."""

    identity: AuthenticatedMonthlyIdentity
    permno: np.ndarray
    permco: np.ndarray
    execution_price: np.ndarray
    full_spread: np.ndarray
    adv20_usd: np.ndarray
    impact_volatility_daily: np.ndarray
    factor_betas: np.ndarray
    specific_variance_monthly: np.ndarray
    sector: np.ndarray
    country: np.ndarray
    new_long_trade_allowed: np.ndarray
    new_short_trade_allowed: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.identity, AuthenticatedMonthlyIdentity):
            raise ProblemBuilderError(
                "market.identity must be AuthenticatedMonthlyIdentity"
            )
        permno = _vector(self.permno, dtype=np.int64, name="market.permno")
        permco = _vector(self.permco, dtype=np.int64, name="market.permco")
        price = _vector(
            self.execution_price,
            dtype=np.float64,
            name="market.execution_price",
        )
        spread = _vector(self.full_spread, dtype=np.float64, name="market.full_spread")
        adv = _vector(self.adv20_usd, dtype=np.float64, name="market.adv20_usd")
        impact = _vector(
            self.impact_volatility_daily,
            dtype=np.float64,
            name="market.impact_volatility_daily",
        )
        specific = _vector(
            self.specific_variance_monthly,
            dtype=np.float64,
            name="market.specific_variance_monthly",
        )
        new_long = _bool_vector(
            self.new_long_trade_allowed,
            name="market.new_long_trade_allowed",
        )
        new_short = _bool_vector(
            self.new_short_trade_allowed,
            name="market.new_short_trade_allowed",
        )
        betas = _matrix(self.factor_betas, rows=len(permno), name="market.factor_betas")
        sector = _text_vector(
            self.sector,
            rows=len(permno),
            name="market.sector",
            missing=UNKNOWN_SECTOR,
        )
        country = _text_vector(
            self.country,
            rows=len(permno),
            name="market.country",
            missing=UNKNOWN_COUNTRY,
        )
        _same_length(
            len(permno),
            permco=permco,
            execution_price=price,
            full_spread=spread,
            adv20_usd=adv,
            impact_volatility_daily=impact,
            specific_variance_monthly=specific,
            sector=sector,
            country=country,
            new_long_trade_allowed=new_long,
            new_short_trade_allowed=new_short,
        )
        _identity(permno, name="market")
        if np.any(permco <= 0):
            raise ProblemBuilderError("market PERMCO values must be positive")
        for name, values in (
            ("execution_price", price),
            ("full_spread", spread),
            ("adv20_usd", adv),
            ("impact_volatility_daily", impact),
            ("specific_variance_monthly", specific),
            ("factor_betas", betas),
        ):
            if np.isinf(values).any():
                raise ProblemBuilderError(f"market {name} may be finite or NaN, not infinite")
        if np.any(new_short & ~new_long):
            raise ProblemBuilderError(
                "market short-entry authority must be a subset of long-entry authority"
            )
        for name, value in (
            ("permno", permno),
            ("permco", permco),
            ("execution_price", price),
            ("full_spread", spread),
            ("adv20_usd", adv),
            ("impact_volatility_daily", impact),
            ("factor_betas", betas),
            ("specific_variance_monthly", specific),
            ("sector", sector),
            ("country", country),
            ("new_long_trade_allowed", new_long),
            ("new_short_trade_allowed", new_short),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True, eq=False)
class InheritedHoldings:
    """Opening weights plus the last authenticated risk state for each holding."""

    identity: AuthenticatedMonthlyIdentity
    permno: np.ndarray
    permco: np.ndarray
    previous_weight: np.ndarray
    last_factor_betas: np.ndarray
    last_specific_variance_monthly: np.ndarray
    risk_age_sessions: np.ndarray
    sector: np.ndarray
    country: np.ndarray
    current_long_eligible: np.ndarray
    current_short_eligible: np.ndarray
    required_exit: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.identity, AuthenticatedMonthlyIdentity):
            raise ProblemBuilderError(
                "inherited.identity must be AuthenticatedMonthlyIdentity"
            )
        permno = _vector(self.permno, dtype=np.int64, name="inherited.permno")
        permco = _vector(self.permco, dtype=np.int64, name="inherited.permco")
        weight = _vector(
            self.previous_weight,
            dtype=np.float64,
            name="inherited.previous_weight",
        )
        specific = _vector(
            self.last_specific_variance_monthly,
            dtype=np.float64,
            name="inherited.last_specific_variance_monthly",
        )
        raw_age = np.asarray(self.risk_age_sessions)
        if raw_age.ndim != 1 or raw_age.dtype.kind not in {"i", "u"}:
            raise ProblemBuilderError("inherited risk_age_sessions must be integers")
        age = _vector(raw_age, dtype=np.int64, name="inherited.risk_age_sessions")
        current_long = _bool_vector(
            self.current_long_eligible,
            name="inherited.current_long_eligible",
        )
        current_short = _bool_vector(
            self.current_short_eligible,
            name="inherited.current_short_eligible",
        )
        required = _bool_vector(self.required_exit, name="inherited.required_exit")
        betas = _matrix(
            self.last_factor_betas,
            rows=len(permno),
            name="inherited.last_factor_betas",
        )
        sector = _text_vector(
            self.sector,
            rows=len(permno),
            name="inherited.sector",
            missing=UNKNOWN_SECTOR,
        )
        country = _text_vector(
            self.country,
            rows=len(permno),
            name="inherited.country",
            missing=UNKNOWN_COUNTRY,
        )
        _same_length(
            len(permno),
            permco=permco,
            previous_weight=weight,
            last_specific_variance_monthly=specific,
            risk_age_sessions=age,
            sector=sector,
            country=country,
            current_long_eligible=current_long,
            current_short_eligible=current_short,
            required_exit=required,
        )
        _identity(permno, name="inherited")
        if np.any(permco <= 0):
            raise ProblemBuilderError("inherited PERMCO values must be positive")
        if not np.isfinite(weight).all() or np.any(weight == 0.0):
            raise ProblemBuilderError("inherited weights must be finite and nonzero")
        if not np.isfinite(betas).all():
            raise ProblemBuilderError("last inherited factor betas must be finite")
        if not np.isfinite(specific).all() or np.any(specific <= 0.0):
            raise ProblemBuilderError(
                "last inherited specific variances must be positive and finite"
            )
        if np.any(age < 0):
            raise ProblemBuilderError("risk ages must be nonnegative")
        if np.any(current_short & ~current_long):
            raise ProblemBuilderError(
                "inherited short eligibility must be a subset of long eligibility"
            )
        for name, value in (
            ("permno", permno),
            ("permco", permco),
            ("previous_weight", weight),
            ("last_factor_betas", betas),
            ("last_specific_variance_monthly", specific),
            ("risk_age_sessions", age),
            ("sector", sector),
            ("country", country),
            ("current_long_eligible", current_long),
            ("current_short_eligible", current_short),
            ("required_exit", required),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def empty(
        cls,
        identity: AuthenticatedMonthlyIdentity,
    ) -> "InheritedHoldings":
        return cls(
            identity=identity,
            permno=np.empty(0, dtype=np.int64),
            permco=np.empty(0, dtype=np.int64),
            previous_weight=np.empty(0, dtype=np.float64),
            last_factor_betas=np.empty((0, FACTOR_COUNT), dtype=np.float64),
            last_specific_variance_monthly=np.empty(0, dtype=np.float64),
            risk_age_sessions=np.empty(0, dtype=np.int64),
            sector=np.empty(0, dtype=np.str_),
            country=np.empty(0, dtype=np.str_),
            current_long_eligible=np.empty(0, dtype=np.bool_),
            current_short_eligible=np.empty(0, dtype=np.bool_),
            required_exit=np.empty(0, dtype=np.bool_),
        )


@dataclass(frozen=True, slots=True, eq=False)
class MonthlyRiskInputs:
    """Authenticated covariance and conservative cross-sectional fallbacks."""

    identity: AuthenticatedMonthlyIdentity
    factor_covariance_monthly: np.ndarray
    specific_variance_p95_monthly: float
    required_exit_impact_volatility_p95_daily: float

    def __post_init__(self) -> None:
        if not isinstance(self.identity, AuthenticatedMonthlyIdentity):
            raise ProblemBuilderError("risk.identity must be AuthenticatedMonthlyIdentity")
        covariance = np.asarray(
            self.factor_covariance_monthly,
            dtype=np.float64,
        ).copy()
        if covariance.shape != (FACTOR_COUNT, FACTOR_COUNT) or not np.isfinite(
            covariance
        ).all():
            raise ProblemBuilderError("factor covariance must be a finite 6 by 6 matrix")
        try:
            variance_p95 = float(self.specific_variance_p95_monthly)
            exit_volatility_p95 = float(
                self.required_exit_impact_volatility_p95_daily
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProblemBuilderError("monthly risk scalars must be numeric") from exc
        if not np.isfinite(variance_p95) or variance_p95 <= 0.0:
            raise ProblemBuilderError("specific variance p95 must be positive and finite")
        if not np.isfinite(exit_volatility_p95) or exit_volatility_p95 < 0.0:
            raise ProblemBuilderError(
                "required-exit impact volatility p95 must be finite and nonnegative"
            )
        covariance.setflags(write=False)
        object.__setattr__(self, "factor_covariance_monthly", covariance)
        object.__setattr__(self, "specific_variance_p95_monthly", variance_p95)
        object.__setattr__(
            self,
            "required_exit_impact_volatility_p95_daily",
            exit_volatility_p95,
        )


def _readonly_ids(values: list[int]) -> np.ndarray:
    result = np.asarray(sorted(values), dtype=np.int64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class MonthlyProblemBuild:
    """Frozen Stage 10 objects plus conservative adapter diagnostics."""

    identity: AuthenticatedMonthlyIdentity
    universe: CandidateUniverse
    selection: CandidateSelection
    problem: AllocationProblemData
    long_entry_allowed: np.ndarray
    short_entry_allowed: np.ndarray
    blocked_new_trade_permno: np.ndarray
    frozen_holding_permno: np.ndarray
    carried_risk_permno: np.ndarray
    stale_p95_risk_permno: np.ndarray
    required_exit_volatility_substituted_permno: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.identity, AuthenticatedMonthlyIdentity):
            raise ProblemBuilderError(
                "monthly build identity must be AuthenticatedMonthlyIdentity"
            )
        long_entry = _bool_vector(
            self.long_entry_allowed,
            name="monthly_build.long_entry_allowed",
        )
        short_entry = _bool_vector(
            self.short_entry_allowed,
            name="monthly_build.short_entry_allowed",
        )
        if (
            len(long_entry) != self.universe.n_assets
            or len(short_entry) != self.universe.n_assets
        ):
            raise ProblemBuilderError(
                "monthly build entry masks must match the candidate universe"
            )
        if np.any(long_entry & ~self.universe.long_eligible):
            raise ProblemBuilderError(
                "monthly build long entry mask exceeds scoring eligibility"
            )
        if np.any(short_entry & ~self.universe.short_eligible):
            raise ProblemBuilderError(
                "monthly build short entry mask exceeds short eligibility"
            )
        if np.any(short_entry & ~long_entry):
            raise ProblemBuilderError(
                "monthly build short entry mask exceeds long entry authority"
            )
        object.__setattr__(self, "long_entry_allowed", long_entry)
        object.__setattr__(self, "short_entry_allowed", short_entry)
        for name in (
            "blocked_new_trade_permno",
            "frozen_holding_permno",
            "carried_risk_permno",
            "stale_p95_risk_permno",
            "required_exit_volatility_substituted_permno",
        ):
            value = _vector(getattr(self, name), dtype=np.int64, name=name)
            object.__setattr__(self, name, value)


def build_monthly_problem(
    scored: ScoredSnapshot,
    market: MarketSnapshot,
    inherited: InheritedHoldings,
    risk: MonthlyRiskInputs,
    *,
    nav_dollars: float,
    borrow_scenario: str = PRIMARY_BORROW_SCENARIO,
    minimum_candidates: int = DEFAULT_MINIMUM_CANDIDATES,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
) -> MonthlyProblemBuild:
    """Union scored names and holdings, then build one frozen allocation problem.

    Missing execution or cost inputs give a row zero trade capacity.  Missing
    current risk instead closes a separate entry gate.  Neither condition
    deletes an otherwise eligible security from the frozen score ranking or
    changes its score.  A holding with missing current risk is reduction-only:
    last risk is carried for at most 21 sessions, after which its last betas are
    retained and idiosyncratic variance is replaced by the supplied
    contemporaneous 95th percentile.
    """

    if not isinstance(scored, ScoredSnapshot):
        raise TypeError("scored must be ScoredSnapshot")
    if not isinstance(market, MarketSnapshot):
        raise TypeError("market must be MarketSnapshot")
    if not isinstance(inherited, InheritedHoldings):
        raise TypeError("inherited must be InheritedHoldings")
    if not isinstance(risk, MonthlyRiskInputs):
        raise TypeError("risk must be MonthlyRiskInputs")
    identities = (scored.identity, market.identity, inherited.identity, risk.identity)
    if any(identity != identities[0] for identity in identities[1:]):
        raise ProblemBuilderError(
            "scored, market, inherited, and risk inputs must share one "
            "authenticated monthly identity"
        )
    try:
        nav = float(nav_dollars)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProblemBuilderError("nav_dollars must be numeric") from exc
    if not np.isfinite(nav) or nav <= 0.0:
        raise ProblemBuilderError("nav_dollars must be positive and finite")
    variance_p95 = risk.specific_variance_p95_monthly
    exit_volatility_p95 = risk.required_exit_impact_volatility_p95_daily
    covariance = risk.factor_covariance_monthly

    market_index = {int(value): index for index, value in enumerate(market.permno)}
    scored_index = {int(value): index for index, value in enumerate(scored.permno)}
    inherited_index = {
        int(value): index for index, value in enumerate(inherited.permno)
    }
    missing_market_identity = sorted(set(scored_index) - set(market_index))
    if missing_market_identity:
        raise ProblemBuilderError(
            "scored names lack current identity rows: "
            + ", ".join(str(value) for value in missing_market_identity)
        )
    permnos = np.asarray(
        sorted(set(scored_index).union(inherited_index)),
        dtype=np.int64,
    )
    if permnos.size == 0:
        raise ProblemBuilderError("monthly scored/holding union is empty")
    n = len(permnos)
    permco = np.empty(n, dtype=np.int64)
    score = np.full(n, np.nan, dtype=np.float64)
    expected = np.full(n, np.nan, dtype=np.float64)
    long_eligible = np.zeros(n, dtype=np.bool_)
    short_eligible = np.zeros(n, dtype=np.bool_)
    long_entry_allowed = np.zeros(n, dtype=np.bool_)
    short_entry_allowed = np.zeros(n, dtype=np.bool_)
    current_screen_long = np.zeros(n, dtype=np.bool_)
    current_screen_short = np.zeros(n, dtype=np.bool_)
    previous = np.zeros(n, dtype=np.float64)
    required_exit = np.zeros(n, dtype=np.bool_)
    execution_price = np.full(n, np.nan, dtype=np.float64)
    spread = np.full(n, np.nan, dtype=np.float64)
    adv = np.full(n, np.nan, dtype=np.float64)
    impact = np.full(n, np.nan, dtype=np.float64)
    betas = np.zeros((n, FACTOR_COUNT), dtype=np.float64)
    specific = np.full(n, variance_p95, dtype=np.float64)
    sector = np.full(n, UNKNOWN_SECTOR, dtype=object)
    country = np.full(n, UNKNOWN_COUNTRY, dtype=object)

    blocked_new: list[int] = []
    frozen_holding: list[int] = []
    carried: list[int] = []
    stale: list[int] = []
    exit_substitution: list[int] = []

    for union_index, raw_permno in enumerate(permnos):
        security = int(raw_permno)
        score_row = scored_index.get(security)
        market_row = market_index.get(security)
        holding_row = inherited_index.get(security)
        if market_row is not None:
            permco[union_index] = int(market.permco[market_row])
            execution_price[union_index] = market.execution_price[market_row]
            spread[union_index] = market.full_spread[market_row]
            adv[union_index] = market.adv20_usd[market_row]
            impact[union_index] = market.impact_volatility_daily[market_row]
            sector[union_index] = market.sector[market_row]
            country[union_index] = market.country[market_row]
        elif holding_row is not None:
            permco[union_index] = int(inherited.permco[holding_row])
            sector[union_index] = inherited.sector[holding_row]
            country[union_index] = inherited.country[holding_row]
        else:  # impossible because every scored name requires market identity
            raise RuntimeError("union row lacks identity")
        if holding_row is not None:
            if (
                market_row is not None
                and int(market.permco[market_row]) != int(inherited.permco[holding_row])
            ):
                raise ProblemBuilderError(
                    f"current and inherited PERMCO differ for PERMNO {security}"
                )
            previous[union_index] = inherited.previous_weight[holding_row]
            required_exit[union_index] = inherited.required_exit[holding_row]
            current_screen_long[union_index] = inherited.current_long_eligible[
                holding_row
            ]
            current_screen_short[union_index] = inherited.current_short_eligible[
                holding_row
            ]
        if score_row is not None:
            score[union_index] = scored.score[score_row]
            long_eligible[union_index] = scored.long_eligible[score_row]
            short_eligible[union_index] = scored.short_eligible[score_row]
            assert market_row is not None
            long_entry_allowed[union_index] = (
                scored.long_eligible[score_row]
                and market.new_long_trade_allowed[market_row]
            )
            short_entry_allowed[union_index] = (
                scored.short_eligible[score_row]
                and market.new_short_trade_allowed[market_row]
            )
            if holding_row is not None and (
                current_screen_long[union_index] != long_eligible[union_index]
                or current_screen_short[union_index] != short_eligible[union_index]
            ):
                raise ProblemBuilderError(
                    f"scored and inherited eligibility differ for PERMNO {security}"
                )
            current_screen_long[union_index] = long_eligible[union_index]
            current_screen_short[union_index] = short_eligible[union_index]

        # Decide sign-specific forced exits before resolving missing impact
        # volatility below.  Otherwise a newly ineligible inherited holding
        # with no current impact estimate would miss the frozen p95 exit-cost
        # substitution and become accidentally untradeable.
        if holding_row is not None and previous[union_index] != 0.0:
            applicable_eligible = (
                current_screen_long[union_index]
                if previous[union_index] > 0.0
                else current_screen_short[union_index]
            )
            if not applicable_eligible:
                required_exit[union_index] = True

        current_risk_valid = bool(
            market_row is not None
            and np.isfinite(market.factor_betas[market_row]).all()
            and np.isfinite(market.specific_variance_monthly[market_row])
            and market.specific_variance_monthly[market_row] > 0.0
        )
        if current_risk_valid:
            assert market_row is not None
            betas[union_index] = market.factor_betas[market_row]
            specific[union_index] = market.specific_variance_monthly[market_row]
        elif holding_row is not None:
            betas[union_index] = inherited.last_factor_betas[holding_row]
            age = int(inherited.risk_age_sessions[holding_row])
            if age <= MAXIMUM_RISK_CARRY_SESSIONS:
                specific[union_index] = inherited.last_specific_variance_monthly[
                    holding_row
                ]
                carried.append(security)
            else:
                specific[union_index] = variance_p95
                stale.append(security)
                required_exit[union_index] = True
            # A carried estimate keeps the holding measurable but can never
            # justify increasing it.  Preserve any authenticated score and its
            # full scoring-universe membership; the distinct entry gate and
            # candidate adapter make it reduction-only during the 21-session
            # carry.  Once stale, the required-exit flag makes it continue to
            # zero under the frozen p95 substitution rule.
            long_entry_allowed[union_index] = False
            short_entry_allowed[union_index] = False
        else:
            # No current or inherited risk authority: block this new row from
            # candidate formation.  Its authenticated score remains in the
            # contemporaneous scoring cross-section, while conservative risk
            # placeholders remain irrelevant because it cannot be selected.
            long_entry_allowed[union_index] = False
            short_entry_allowed[union_index] = False

        if (
            required_exit[union_index]
            and not (
                np.isfinite(impact[union_index]) and impact[union_index] >= 0.0
            )
        ):
            impact[union_index] = exit_volatility_p95
            exit_substitution.append(security)

    # The calibration slope belongs to the selector's authenticated score.
    # Risk/execution gates must never translate, rescale, or rerank that score.
    expected = scored.calibration_kappa_monthly * score
    ordinary_variance = (
        np.einsum("ij,jk,ik->i", betas, covariance, betas) + specific
    )
    if not np.isfinite(ordinary_variance).all() or np.any(ordinary_variance <= 0.0):
        raise ProblemBuilderError("resolved forecast variances must be positive")
    forecast_volatility = np.sqrt(ordinary_variance)

    valid_execution = np.isfinite(execution_price) & (execution_price > 0.0)
    valid_cost = (
        valid_execution
        & np.isfinite(spread)
        & (spread >= 0.0)
        & np.isfinite(adv)
        & (adv > 0.0)
        & np.isfinite(impact)
        & (impact >= 0.0)
    )
    capacity = np.zeros(n, dtype=np.float64)
    capacity[valid_cost] = (
        MAXIMUM_ADV_TRADE_FRACTION * adv[valid_cost] / nav
    )
    for index, security in enumerate(permnos):
        if (
            previous[index] == 0.0
            and long_eligible[index]
            and (not valid_cost[index] or not long_entry_allowed[index])
        ):
            blocked_new.append(int(security))
        if previous[index] != 0.0 and not valid_cost[index]:
            frozen_holding.append(int(security))

    universe = CandidateUniverse(
        permno=permnos,
        permco=permco,
        score=score,
        long_eligible=long_eligible,
        short_eligible=short_eligible,
        previous_weight=previous,
        adv20_usd=adv,
        sector=np.asarray(sector, dtype=np.str_),
        country=np.asarray(country, dtype=np.str_),
        required_exit=required_exit,
    )
    selection = form_stage11_monthly_candidates(
        universe,
        long_entry_allowed=long_entry_allowed,
        short_entry_allowed=short_entry_allowed,
        minimum_candidates=minimum_candidates,
        tail_fraction=tail_fraction,
    )
    problem = selection.to_problem_data(
        expected_return=expected,
        factor_betas=betas,
        factor_covariance_monthly=covariance,
        specific_variance_monthly=specific,
        forecast_volatility=forecast_volatility,
        full_spread=spread,
        impact_volatility_daily=impact,
        borrow_scenario=borrow_scenario,
        trade_capacity_weight=capacity,
        nav_dollars=nav,
        market_variance_monthly=float(covariance[0, 0]),
    )
    return MonthlyProblemBuild(
        identity=scored.identity,
        universe=universe,
        selection=selection,
        problem=problem,
        long_entry_allowed=long_entry_allowed,
        short_entry_allowed=short_entry_allowed,
        blocked_new_trade_permno=_readonly_ids(blocked_new),
        frozen_holding_permno=_readonly_ids(frozen_holding),
        carried_risk_permno=_readonly_ids(carried),
        stale_p95_risk_permno=_readonly_ids(stale),
        required_exit_volatility_substituted_permno=_readonly_ids(
            exit_substitution
        ),
    )


__all__ = [
    "MAXIMUM_ADV_TRADE_FRACTION",
    "MAXIMUM_RISK_CARRY_SESSIONS",
    "SCORE_CENTERING_TOLERANCE",
    "AuthenticatedMonthlyIdentity",
    "InheritedHoldings",
    "MarketSnapshot",
    "MonthlyRiskInputs",
    "MonthlyProblemBuild",
    "ProblemBuilderError",
    "ScoredSnapshot",
    "build_monthly_problem",
]
