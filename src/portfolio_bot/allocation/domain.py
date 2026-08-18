"""Immutable, solver-facing domain objects for roadmap Stage 10.

The objects in this module are deliberately pure: they perform no filesystem
access, model fitting, optimization, or portfolio accounting.  Every NumPy
array is defensively copied and made read-only so a validated problem cannot
be mutated underneath a conic solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from portfolio_bot.allocation.risk import build_augmented_risk_root


FACTOR_COUNT = 6
LONG_SIDE = np.int8(1)
SHORT_SIDE = np.int8(-1)
UNUSED_SIDE = np.int8(0)

USA = "USA"
UNKNOWN_COUNTRY = "UNKNOWN_COUNTRY"
UNKNOWN_FOREIGN = "UNKNOWN_FOREIGN"
UNKNOWN_SECTOR = "UNKNOWN_SECTOR"

ACTIVE = "ACTIVE"
INSUFFICIENT_DISJOINT_CANDIDATES = "INSUFFICIENT_DISJOINT_CANDIDATES"

PRIMARY_BORROW_SCENARIO = "primary"
STRESS_BORROW_SCENARIO = "stress"
PRIMARY_SHORT_BORROW_RATE_ANNUAL = 0.01
STRESS_SHORT_BORROW_RATE_ANNUAL = 0.03
STRESS_LOWEST_LIQUIDITY_BORROW_RATE_ANNUAL = 0.10

# The risk inputs are serialized as decimal monthly variances.  This tolerance
# accepts round-trip noise well below a basis point of annualized volatility,
# while still rejecting a forecast-volatility vector from a different model.
FORECAST_VOLATILITY_RTOL = 1e-7
FORECAST_VOLATILITY_ATOL = 1e-10
MARKET_VARIANCE_RTOL = 1e-12
MARKET_VARIANCE_ATOL = 1e-15
TRADE_CAPACITY_ATOL = 1e-12

MANDATORY_CORPORATE_ACTION_HANDOFF = (
    "mandatory corporate actions must be accounted in the upstream return/"
    "position engine before allocation"
)


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype | type,
    name: str,
    ndim: int = 1,
) -> np.ndarray:
    try:
        result = np.array(value, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} cannot be converted to {np.dtype(dtype).name}"
        ) from exc
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    result.setflags(write=False)
    return result


def _readonly_bool(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.dtype.kind not in {"b", "i", "u"}:
        raise ValueError(f"{name} must contain boolean values")
    if raw.dtype.kind != "b" and not np.isin(raw, (0, 1)).all():
        raise ValueError(f"{name} must contain only boolean values")
    return _readonly_array(raw, dtype=np.bool_, name=name)


def _normalize_text(
    values: Sequence[object],
    *,
    name: str,
    missing_value: str,
) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    normalized: list[str] = []
    for value in raw:
        if value is None:
            normalized.append(missing_value)
            continue
        text = str(value).strip().upper()
        normalized.append(text if text else missing_value)
    result = np.asarray(normalized, dtype=np.str_)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    result.setflags(write=False)
    return result


def _require_same_length(length: int, **arrays: np.ndarray) -> None:
    mismatches = sorted(name for name, value in arrays.items() if len(value) != length)
    if mismatches:
        raise ValueError(f"array lengths differ for: {', '.join(mismatches)}")


def _readonly_mask(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def derive_borrow_rate_annual(
    side: object,
    borrow_liquidity_quintile: object,
    *,
    scenario: str,
) -> np.ndarray:
    """Map the frozen short-union ADV quintiles to annual borrow rates.

    Long rows always receive zero.  Under the primary scenario every short is
    charged 1%.  Under the conservative stress, quintile-one shorts are
    charged 10% *total* and every other short 3%.
    """

    normalized_side = _readonly_array(side, dtype=np.int8, name="side")
    quintile = _readonly_array(
        borrow_liquidity_quintile,
        dtype=np.int8,
        name="borrow_liquidity_quintile",
    )
    if len(normalized_side) != len(quintile):
        raise ValueError("borrow-rate input lengths differ")
    if not np.isin(normalized_side, (LONG_SIDE, SHORT_SIDE)).all():
        raise ValueError("side must contain only +1 and -1")
    if np.any((quintile < 0) | (quintile > 5)):
        raise ValueError("borrow liquidity quintiles must be in [0, 5]")
    short = normalized_side == SHORT_SIDE
    if np.any(short & (quintile == 0)) or np.any((~short) & (quintile != 0)):
        raise ValueError("borrow quintiles must be positive exactly for short rows")
    if scenario not in {PRIMARY_BORROW_SCENARIO, STRESS_BORROW_SCENARIO}:
        raise ValueError("borrow scenario must be 'primary' or 'stress'")

    rates = np.zeros(len(normalized_side), dtype=np.float64)
    if scenario == PRIMARY_BORROW_SCENARIO:
        rates[short] = PRIMARY_SHORT_BORROW_RATE_ANNUAL
    else:
        rates[short] = STRESS_SHORT_BORROW_RATE_ANNUAL
        rates[short & (quintile == 1)] = STRESS_LOWEST_LIQUIDITY_BORROW_RATE_ANNUAL
    rates.setflags(write=False)
    return rates


@dataclass(frozen=True, eq=False)
class CandidateUniverse:
    """One complete monthly scoring universe plus inherited holdings.

    ``long_eligible`` identifies the full cross-section on which quintiles are
    formed.  ``short_eligible`` is the stricter short screen and must be a
    subset.  Rows that are no longer eligible may remain solely to carry an
    inherited position toward zero.
    """

    permno: np.ndarray
    permco: np.ndarray
    score: np.ndarray
    long_eligible: np.ndarray
    short_eligible: np.ndarray
    previous_weight: np.ndarray
    adv20_usd: np.ndarray
    sector: np.ndarray
    country: np.ndarray
    required_exit: np.ndarray
    mandatory_corporate_action: np.ndarray | None = None

    def __post_init__(self) -> None:
        permno = _readonly_array(self.permno, dtype=np.int64, name="permno")
        permco = _readonly_array(self.permco, dtype=np.int64, name="permco")
        score = _readonly_array(self.score, dtype=np.float64, name="score")
        long_eligible = _readonly_bool(self.long_eligible, name="long_eligible")
        short_eligible = _readonly_bool(self.short_eligible, name="short_eligible")
        previous = _readonly_array(
            self.previous_weight, dtype=np.float64, name="previous_weight"
        )
        adv = _readonly_array(self.adv20_usd, dtype=np.float64, name="adv20_usd")
        sector = _normalize_text(
            self.sector, name="sector", missing_value=UNKNOWN_SECTOR
        )
        country = _normalize_text(
            self.country, name="country", missing_value=UNKNOWN_COUNTRY
        )
        required_exit = _readonly_bool(self.required_exit, name="required_exit")
        mandatory_corporate_action = (
            _readonly_bool(
                np.zeros(len(permno), dtype=np.bool_),
                name="mandatory_corporate_action",
            )
            if self.mandatory_corporate_action is None
            else _readonly_bool(
                self.mandatory_corporate_action,
                name="mandatory_corporate_action",
            )
        )

        n = len(permno)
        if n == 0:
            raise ValueError("candidate universe must contain at least one row")
        _require_same_length(
            n,
            permco=permco,
            score=score,
            long_eligible=long_eligible,
            short_eligible=short_eligible,
            previous_weight=previous,
            adv20_usd=adv,
            sector=sector,
            country=country,
            required_exit=required_exit,
            mandatory_corporate_action=mandatory_corporate_action,
        )
        if np.any(permno <= 0) or len(np.unique(permno)) != n:
            raise ValueError("permno must be positive and unique")
        if np.any(permco <= 0):
            raise ValueError("permco must be positive")
        if not np.isfinite(previous).all():
            raise ValueError("previous_weight must be finite")
        if np.isinf(adv).any():
            raise ValueError("adv20_usd may be finite or NaN, but not infinite")
        if np.any(short_eligible & ~long_eligible):
            raise ValueError("short_eligible must be a subset of long_eligible")
        if np.any(required_exit & (previous == 0.0)):
            raise ValueError("required_exit requires an inherited nonzero position")
        if mandatory_corporate_action.any():
            raise ValueError(MANDATORY_CORPORATE_ACTION_HANDOFF)

        scoring_scores = score[long_eligible]
        if not np.isfinite(scoring_scores).all():
            raise ValueError("every scoring-universe score must be finite")
        if np.any(np.abs(scoring_scores) > 0.5 + 1e-12):
            raise ValueError("scoring-universe scores must lie in [-0.5, 0.5]")
        # Published selector scores are float32, so retain a small allowance
        # for their otherwise zero-centered cross-sectional mean.
        if scoring_scores.size and abs(float(np.mean(scoring_scores))) > 1e-8:
            raise ValueError("monthly scoring-universe scores must be centered")

        for field, value in (
            ("permno", permno),
            ("permco", permco),
            ("score", score),
            ("long_eligible", long_eligible),
            ("short_eligible", short_eligible),
            ("previous_weight", previous),
            ("adv20_usd", adv),
            ("sector", sector),
            ("country", country),
            ("required_exit", required_exit),
            ("mandatory_corporate_action", mandatory_corporate_action),
        ):
            object.__setattr__(self, field, value)

    @property
    def n_assets(self) -> int:
        return len(self.permno)

    @property
    def held_mask(self) -> np.ndarray:
        return _readonly_mask(self.previous_weight != 0.0)

    @property
    def foreign_mask(self) -> np.ndarray:
        """Mark every non-USA bucket, including both unknown-country buckets."""

        return _readonly_mask(self.country != USA)


@dataclass(frozen=True, eq=False)
class AllocationProblemData:
    """Validated one-month numeric input for the common conic allocator.

    Rows contain only active candidates and inherited holdings.  ``side`` is
    fixed for the solve; a zero-weight row may enter only on its assigned side,
    while a nonzero inherited position must keep its existing sign.  An empty
    row set is the explicit all-cash/risk-off case.  ``risk_root`` and
    ``borrow_rate_annual`` are derived internally from authoritative primitive
    inputs and cannot be supplied by callers.  Pending mandatory corporate
    actions are prohibited because their payoff belongs to upstream accounting.
    """

    permno: np.ndarray
    permco: np.ndarray
    side: np.ndarray
    previous_weight: np.ndarray
    reduction_only: np.ndarray
    required_exit: np.ndarray
    expected_return: np.ndarray
    score: np.ndarray
    factor_betas: np.ndarray
    factor_covariance_monthly: np.ndarray
    specific_variance_monthly: np.ndarray
    forecast_volatility: np.ndarray
    full_spread: np.ndarray
    adv20_usd: np.ndarray
    impact_volatility_daily: np.ndarray
    borrow_liquidity_quintile: np.ndarray
    borrow_scenario: str
    trade_capacity_weight: np.ndarray
    sector: np.ndarray
    country: np.ndarray
    nav_dollars: float
    market_variance_monthly: float
    mandatory_corporate_action: np.ndarray | None = None
    risk_root: np.ndarray = field(init=False, repr=False)
    borrow_rate_annual: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        vectors = {
            "permno": _readonly_array(self.permno, dtype=np.int64, name="permno"),
            "permco": _readonly_array(self.permco, dtype=np.int64, name="permco"),
            "side": _readonly_array(self.side, dtype=np.int8, name="side"),
            "previous_weight": _readonly_array(
                self.previous_weight, dtype=np.float64, name="previous_weight"
            ),
            "reduction_only": _readonly_bool(
                self.reduction_only, name="reduction_only"
            ),
            "required_exit": _readonly_bool(self.required_exit, name="required_exit"),
            "expected_return": _readonly_array(
                self.expected_return, dtype=np.float64, name="expected_return"
            ),
            "score": _readonly_array(self.score, dtype=np.float64, name="score"),
            "forecast_volatility": _readonly_array(
                self.forecast_volatility,
                dtype=np.float64,
                name="forecast_volatility",
            ),
            "full_spread": _readonly_array(
                self.full_spread, dtype=np.float64, name="full_spread"
            ),
            "adv20_usd": _readonly_array(
                self.adv20_usd, dtype=np.float64, name="adv20_usd"
            ),
            "impact_volatility_daily": _readonly_array(
                self.impact_volatility_daily,
                dtype=np.float64,
                name="impact_volatility_daily",
            ),
            "borrow_liquidity_quintile": _readonly_array(
                self.borrow_liquidity_quintile,
                dtype=np.int8,
                name="borrow_liquidity_quintile",
            ),
            "trade_capacity_weight": _readonly_array(
                self.trade_capacity_weight,
                dtype=np.float64,
                name="trade_capacity_weight",
            ),
        }
        factor_betas = _readonly_array(
            self.factor_betas, dtype=np.float64, name="factor_betas", ndim=2
        )
        factor_covariance = _readonly_array(
            self.factor_covariance_monthly,
            dtype=np.float64,
            name="factor_covariance_monthly",
            ndim=2,
        )
        specific_variance = _readonly_array(
            self.specific_variance_monthly,
            dtype=np.float64,
            name="specific_variance_monthly",
        )
        sector = _normalize_text(
            self.sector, name="sector", missing_value=UNKNOWN_SECTOR
        )
        country = _normalize_text(
            self.country, name="country", missing_value=UNKNOWN_COUNTRY
        )

        n = len(vectors["permno"])
        mandatory_corporate_action = (
            _readonly_bool(
                np.zeros(n, dtype=np.bool_),
                name="mandatory_corporate_action",
            )
            if self.mandatory_corporate_action is None
            else _readonly_bool(
                self.mandatory_corporate_action,
                name="mandatory_corporate_action",
            )
        )
        _require_same_length(
            n,
            **{name: value for name, value in vectors.items() if name != "permno"},
            sector=sector,
            country=country,
            specific_variance_monthly=specific_variance,
            mandatory_corporate_action=mandatory_corporate_action,
        )
        if factor_betas.shape != (n, FACTOR_COUNT):
            raise ValueError("factor_betas must have shape (n_assets, 6)")
        if factor_covariance.shape != (FACTOR_COUNT, FACTOR_COUNT):
            raise ValueError("factor_covariance_monthly must have shape (6, 6)")
        if np.any(vectors["permno"] <= 0) or len(np.unique(vectors["permno"])) != n:
            raise ValueError("permno must be positive and unique")
        if np.any(vectors["permco"] <= 0):
            raise ValueError("permco must be positive")
        if not np.isin(vectors["side"], (LONG_SIDE, SHORT_SIDE)).all():
            raise ValueError("side must contain only +1 and -1")

        previous = vectors["previous_weight"]
        side = vectors["side"].astype(np.float64, copy=False)
        if not np.isfinite(previous).all() or np.any(side * previous < -1e-15):
            raise ValueError("inherited weights cannot reverse their fixed side")
        reduction = vectors["reduction_only"]
        required = vectors["required_exit"]
        if np.any(reduction & (previous == 0.0)):
            raise ValueError("reduction_only requires an inherited position")
        if np.any(required & ~reduction):
            raise ValueError("required_exit rows must be reduction_only")

        finite_names = (
            "expected_return",
            "score",
            "forecast_volatility",
            "trade_capacity_weight",
        )
        if (
            not np.isfinite(factor_betas).all()
            or not np.isfinite(factor_covariance).all()
        ):
            raise ValueError("risk arrays must be finite")
        for name in finite_names:
            if not np.isfinite(vectors[name]).all():
                raise ValueError(f"{name} must be finite")
        if np.any(vectors["forecast_volatility"] <= 0.0):
            raise ValueError("forecast_volatility must be positive")
        if np.any(vectors["trade_capacity_weight"] < 0.0):
            raise ValueError("trade_capacity_weight must be nonnegative")
        if not np.isfinite(specific_variance).all() or np.any(specific_variance <= 0.0):
            raise ValueError(
                "specific_variance_monthly must be finite and strictly positive"
            )
        for name in ("full_spread", "adv20_usd", "impact_volatility_daily"):
            if np.isinf(vectors[name]).any():
                raise ValueError(f"{name} may be finite or NaN, but not infinite")
        valid_cost_inputs = (
            np.isfinite(vectors["full_spread"])
            & (vectors["full_spread"] >= 0.0)
            & np.isfinite(vectors["adv20_usd"])
            & (vectors["adv20_usd"] > 0.0)
            & np.isfinite(vectors["impact_volatility_daily"])
            & (vectors["impact_volatility_daily"] >= 0.0)
        )
        if np.any((~valid_cost_inputs) & (vectors["trade_capacity_weight"] > 0.0)):
            raise ValueError(
                "rows with missing/invalid trade-cost inputs must have zero trade capacity"
            )

        try:
            nav = float(self.nav_dollars)
            market_variance = float(self.market_variance_monthly)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("allocation scalars must be numeric") from exc
        if not np.isfinite(nav) or nav <= 0.0:
            raise ValueError("nav_dollars must be positive and finite")
        if not np.isfinite(market_variance) or market_variance < 0.0:
            raise ValueError("market_variance_monthly must be finite and nonnegative")
        if not np.isclose(
            market_variance,
            float(factor_covariance[0, 0]),
            rtol=MARKET_VARIANCE_RTOL,
            atol=MARKET_VARIANCE_ATOL,
        ):
            raise ValueError(
                "market_variance_monthly must equal factor covariance mktrf variance"
            )
        if mandatory_corporate_action.any():
            raise ValueError(MANDATORY_CORPORATE_ACTION_HANDOFF)

        positive_adv = np.isfinite(vectors["adv20_usd"]) & (vectors["adv20_usd"] > 0.0)
        maximum_capacity = np.zeros(n, dtype=np.float64)
        maximum_capacity[positive_adv] = 0.10 * vectors["adv20_usd"][positive_adv] / nav
        if np.any(
            vectors["trade_capacity_weight"] > maximum_capacity + TRADE_CAPACITY_ATOL
        ):
            raise ValueError("trade_capacity_weight exceeds 10% of ADV divided by NAV")

        # The optimizer never accepts an externally supplied square root: it
        # is reconstructed from the authoritative primitive risk estimates.
        risk_root = _readonly_array(
            build_augmented_risk_root(
                factor_betas,
                factor_covariance,
                specific_variance,
                market_variance_monthly=market_variance,
            ),
            dtype=np.float64,
            name="risk_root",
            ndim=2,
        )
        ordinary_variance = (
            np.einsum(
                "ij,jk,ik->i",
                factor_betas,
                factor_covariance,
                factor_betas,
            )
            + specific_variance
        )
        implied_forecast_volatility = np.sqrt(ordinary_variance)
        if not np.allclose(
            vectors["forecast_volatility"],
            implied_forecast_volatility,
            rtol=FORECAST_VOLATILITY_RTOL,
            atol=FORECAST_VOLATILITY_ATOL,
        ):
            raise ValueError("forecast_volatility must equal sqrt(diag(B F B' + D))")

        if not isinstance(self.borrow_scenario, str):
            raise ValueError("borrow_scenario must be a string")
        borrow_scenario = self.borrow_scenario.strip().lower()
        borrow_rate = derive_borrow_rate_annual(
            vectors["side"],
            vectors["borrow_liquidity_quintile"],
            scenario=borrow_scenario,
        )

        for field, value in vectors.items():
            object.__setattr__(self, field, value)
        object.__setattr__(self, "factor_betas", factor_betas)
        object.__setattr__(self, "factor_covariance_monthly", factor_covariance)
        object.__setattr__(self, "specific_variance_monthly", specific_variance)
        object.__setattr__(self, "risk_root", risk_root)
        object.__setattr__(self, "borrow_scenario", borrow_scenario)
        object.__setattr__(self, "borrow_rate_annual", borrow_rate)
        object.__setattr__(self, "sector", sector)
        object.__setattr__(self, "country", country)
        object.__setattr__(self, "nav_dollars", nav)
        object.__setattr__(self, "market_variance_monthly", market_variance)
        object.__setattr__(
            self,
            "mandatory_corporate_action",
            mandatory_corporate_action,
        )

    @property
    def n_assets(self) -> int:
        return len(self.permno)

    @property
    def long_mask(self) -> np.ndarray:
        return _readonly_mask(self.side == LONG_SIDE)

    @property
    def short_mask(self) -> np.ndarray:
        return _readonly_mask(self.side == SHORT_SIDE)

    @property
    def foreign_mask(self) -> np.ndarray:
        return _readonly_mask(self.country != USA)


@dataclass(frozen=True, eq=False)
class CandidateSelection:
    """Deterministic candidate decision over one :class:`CandidateUniverse`."""

    universe: CandidateUniverse
    long_candidate: np.ndarray
    short_candidate: np.ndarray
    selected: np.ndarray
    side: np.ndarray
    reduction_only: np.ndarray
    borrow_liquidity_quintile: np.ndarray
    long_entry_boundary: float | None
    short_entry_boundary: float | None
    tail_size: int
    minimum_candidates: int
    available_long_candidates: int
    available_short_candidates: int
    overlap_count: int
    status: str

    def __post_init__(self) -> None:
        if not isinstance(self.universe, CandidateUniverse):
            raise TypeError("universe must be CandidateUniverse")
        n = self.universe.n_assets
        long_candidate = _readonly_bool(self.long_candidate, name="long_candidate")
        short_candidate = _readonly_bool(self.short_candidate, name="short_candidate")
        selected = _readonly_bool(self.selected, name="selected")
        side = _readonly_array(self.side, dtype=np.int8, name="side")
        reduction = _readonly_bool(self.reduction_only, name="reduction_only")
        quintile = _readonly_array(
            self.borrow_liquidity_quintile,
            dtype=np.int8,
            name="borrow_liquidity_quintile",
        )
        _require_same_length(
            n,
            long_candidate=long_candidate,
            short_candidate=short_candidate,
            selected=selected,
            side=side,
            reduction_only=reduction,
            borrow_liquidity_quintile=quintile,
        )
        if np.any(long_candidate & short_candidate):
            raise ValueError("long and short candidates must be disjoint")
        held = self.universe.previous_weight != 0.0
        if not np.array_equal(selected, long_candidate | short_candidate | held):
            raise ValueError("selected must equal candidates union inherited holdings")
        expected_side = np.zeros(n, dtype=np.int8)
        expected_side[long_candidate] = LONG_SIDE
        expected_side[short_candidate] = SHORT_SIDE
        expected_side[self.universe.previous_weight > 0.0] = LONG_SIDE
        expected_side[self.universe.previous_weight < 0.0] = SHORT_SIDE
        if not np.array_equal(side, expected_side):
            raise ValueError("side differs from candidates and inherited signs")
        if np.any(reduction & ~held) or np.any(reduction & ~selected):
            raise ValueError("only inherited selected positions can be reduction_only")
        if np.any(self.universe.required_exit & ~reduction):
            raise ValueError("required exits must be reduction_only")
        if np.any(long_candidate & (self.universe.previous_weight < 0.0)):
            raise ValueError("an inherited short cannot reverse to long")
        if np.any(short_candidate & (self.universe.previous_weight > 0.0)):
            raise ValueError("an inherited long cannot reverse to short")
        if np.any((quintile < 0) | (quintile > 5)):
            raise ValueError("borrow liquidity quintiles must be in [0, 5]")
        short_union = short_candidate | (self.universe.previous_weight < 0.0)
        if np.any(quintile[short_union] == 0) or np.any(quintile[~short_union] != 0):
            raise ValueError(
                "borrow quintiles must cover candidates plus inherited shorts"
            )
        if self.status not in {ACTIVE, INSUFFICIENT_DISJOINT_CANDIDATES}:
            raise ValueError("unknown candidate-selection status")
        integer_metadata = {
            "tail_size": self.tail_size,
            "minimum_candidates": self.minimum_candidates,
            "available_long_candidates": self.available_long_candidates,
            "available_short_candidates": self.available_short_candidates,
            "overlap_count": self.overlap_count,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_metadata.values()
        ):
            raise ValueError("candidate-selection counts must be integers")
        if self.tail_size <= 0 or self.minimum_candidates <= 0:
            raise ValueError("tail and minimum candidate counts must be positive")
        if any(
            value < 0 or value > n
            for name, value in integer_metadata.items()
            if name not in {"tail_size", "minimum_candidates"}
        ):
            raise ValueError("candidate-selection availability counts are out of range")
        for name, value in (
            ("long_entry_boundary", self.long_entry_boundary),
            ("short_entry_boundary", self.short_entry_boundary),
        ):
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or None")
        if self.status == ACTIVE:
            if (
                int(long_candidate.sum()) < self.minimum_candidates
                or int(short_candidate.sum()) < self.minimum_candidates
            ):
                raise ValueError("ACTIVE selection does not meet minimum side counts")
            if (
                self.available_long_candidates != int(long_candidate.sum())
                or self.available_short_candidates != int(short_candidate.sum())
                or self.overlap_count != 0
            ):
                raise ValueError(
                    "ACTIVE availability metadata differs from candidate masks"
                )
        elif long_candidate.any() or short_candidate.any() or np.any(held & ~reduction):
            raise ValueError(
                "insufficient selection must take no new risk and reduce holdings"
            )

        for name, value in (
            ("long_candidate", long_candidate),
            ("short_candidate", short_candidate),
            ("selected", selected),
            ("side", side),
            ("reduction_only", reduction),
            ("borrow_liquidity_quintile", quintile),
        ):
            object.__setattr__(self, name, value)

    @property
    def new_risk_allowed(self) -> bool:
        return self.status == ACTIVE

    @property
    def solver_indices(self) -> np.ndarray:
        result = np.flatnonzero(self.selected).astype(np.int64, copy=False)
        result.setflags(write=False)
        return result

    @property
    def n_assets(self) -> int:
        return int(self.selected.sum())

    @property
    def foreign_mask(self) -> np.ndarray:
        result = self.universe.foreign_mask[self.selected].copy()
        result.setflags(write=False)
        return result

    def to_problem_data(
        self,
        *,
        expected_return: np.ndarray,
        factor_betas: np.ndarray,
        factor_covariance_monthly: np.ndarray,
        specific_variance_monthly: np.ndarray,
        forecast_volatility: np.ndarray,
        full_spread: np.ndarray,
        impact_volatility_daily: np.ndarray,
        borrow_scenario: str,
        trade_capacity_weight: np.ndarray,
        nav_dollars: float,
        market_variance_monthly: float,
    ) -> AllocationProblemData:
        """Subset full-universe arrays and derive risk/borrow solver inputs.

        Nonfinite alpha inputs are neutralized only for selected inherited
        holdings that are both ineligible and reduction-only.  Any eligible or
        new candidate with a nonfinite value fails closed.
        """

        index = self.solver_indices
        n = self.universe.n_assets

        def vector(
            value: object,
            name: str,
            *,
            neutralize_ineligible_reduction_only: bool = False,
            allow_nonfinite: bool = False,
        ) -> np.ndarray:
            array = np.asarray(value)
            if array.ndim != 1 or len(array) != n:
                raise ValueError(f"{name} must be a full-universe vector")
            try:
                selected_value = np.asarray(array[index], dtype=np.float64).copy()
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            nonfinite = ~np.isfinite(selected_value)
            if nonfinite.any() and not allow_nonfinite:
                permitted = (
                    self.reduction_only[index] & ~self.universe.long_eligible[index]
                )
                if not neutralize_ineligible_reduction_only or np.any(
                    nonfinite & ~permitted
                ):
                    raise ValueError(
                        f"{name} is nonfinite for an eligible or new candidate"
                    )
                selected_value[nonfinite] = 0.0
            return selected_value

        betas = np.asarray(factor_betas)
        if betas.shape != (n, FACTOR_COUNT):
            raise ValueError("factor_betas must be a full-universe (n, 6) matrix")
        covariance = np.asarray(factor_covariance_monthly)
        if covariance.shape != (FACTOR_COUNT, FACTOR_COUNT):
            raise ValueError("factor_covariance_monthly must be a 6 by 6 matrix")
        return AllocationProblemData(
            permno=self.universe.permno[index],
            permco=self.universe.permco[index],
            side=self.side[index],
            previous_weight=self.universe.previous_weight[index],
            reduction_only=self.reduction_only[index],
            required_exit=self.universe.required_exit[index],
            expected_return=vector(
                expected_return,
                "expected_return",
                neutralize_ineligible_reduction_only=True,
            ),
            score=vector(
                self.universe.score,
                "score",
                neutralize_ineligible_reduction_only=True,
            ),
            factor_betas=betas[index],
            factor_covariance_monthly=covariance,
            specific_variance_monthly=vector(
                specific_variance_monthly,
                "specific_variance_monthly",
            ),
            forecast_volatility=vector(forecast_volatility, "forecast_volatility"),
            # Missing cost inputs are valid only when the separately supplied
            # trade capacity is exactly zero.  AllocationProblemData enforces
            # that relationship, allowing an inherited holding to be frozen
            # without pretending that a nonzero trade has a known cost.
            full_spread=vector(
                full_spread,
                "full_spread",
                allow_nonfinite=True,
            ),
            adv20_usd=self.universe.adv20_usd[index],
            impact_volatility_daily=vector(
                impact_volatility_daily,
                "impact_volatility_daily",
                allow_nonfinite=True,
            ),
            borrow_liquidity_quintile=self.borrow_liquidity_quintile[index],
            borrow_scenario=borrow_scenario,
            trade_capacity_weight=vector(
                trade_capacity_weight, "trade_capacity_weight"
            ),
            sector=self.universe.sector[index],
            country=self.universe.country[index],
            nav_dollars=nav_dollars,
            market_variance_monthly=market_variance_monthly,
            mandatory_corporate_action=self.universe.mandatory_corporate_action[index],
        )


__all__ = [
    "ACTIVE",
    "FACTOR_COUNT",
    "FORECAST_VOLATILITY_ATOL",
    "FORECAST_VOLATILITY_RTOL",
    "INSUFFICIENT_DISJOINT_CANDIDATES",
    "LONG_SIDE",
    "MANDATORY_CORPORATE_ACTION_HANDOFF",
    "PRIMARY_BORROW_SCENARIO",
    "PRIMARY_SHORT_BORROW_RATE_ANNUAL",
    "SHORT_SIDE",
    "STRESS_BORROW_SCENARIO",
    "STRESS_LOWEST_LIQUIDITY_BORROW_RATE_ANNUAL",
    "STRESS_SHORT_BORROW_RATE_ANNUAL",
    "UNKNOWN_COUNTRY",
    "UNKNOWN_FOREIGN",
    "UNKNOWN_SECTOR",
    "USA",
    "AllocationProblemData",
    "CandidateSelection",
    "CandidateUniverse",
    "derive_borrow_rate_annual",
]
