"""Deterministic conservative-cost primitives for roadmap Stage 8.

The functions in this module estimate cost inputs and rates only.  They do not
apply costs to a portfolio return series and therefore do not implement a P&L
or backtest engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


COMMISSION_AND_OPERATIONAL_SLIPPAGE = 0.0001
REFERENCE_NAV_DOLLARS = 100_000_000.0
NAV_SENSITIVITY_DOLLARS = (25_000_000.0, 100_000_000.0, 500_000_000.0)


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _is_finite_nonnegative(value: object) -> bool:
    result = _finite_float(value)
    return result is not None and result >= 0.0


@dataclass(frozen=True)
class ImpactVolatilityEstimate:
    """Daily impact volatility and the global-session window that supplied it."""

    daily_volatility: float
    source_window_sessions: int
    valid_observations: int


@dataclass(frozen=True)
class TransactionCostScenario:
    """Frozen spread and square-root-impact multipliers for one cost scenario."""

    name: str
    spread_multiplier: float
    impact_coefficient: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must be nonempty")
        if not _is_finite_nonnegative(self.spread_multiplier):
            raise ValueError("spread multiplier must be finite and nonnegative")
        if not _is_finite_nonnegative(self.impact_coefficient):
            raise ValueError("impact coefficient must be finite and nonnegative")


PRIMARY_COST_SCENARIO = TransactionCostScenario(
    name="primary",
    spread_multiplier=1.0,
    impact_coefficient=0.10,
)
STANDARD_STRESS_COST_SCENARIO = TransactionCostScenario(
    name="standard_stress",
    spread_multiplier=2.0,
    impact_coefficient=0.20,
)
SEVERE_IMPACT_COST_SCENARIO = TransactionCostScenario(
    name="severe_impact",
    spread_multiplier=1.0,
    impact_coefficient=0.50,
)
TRANSACTION_COST_SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        PRIMARY_COST_SCENARIO,
        STANDARD_STRESS_COST_SCENARIO,
        SEVERE_IMPACT_COST_SCENARIO,
    )
}


@dataclass(frozen=True)
class PeriodicRate:
    """A simple annual rate expressed on the frozen optimizer and daily clocks."""

    annual: float
    optimizer_monthly: float
    daily: float

    @classmethod
    def from_annual(cls, annual: float) -> PeriodicRate:
        if not _is_finite_nonnegative(annual):
            raise ValueError("annual rate must be finite and nonnegative")
        annual_value = float(annual)
        return cls(
            annual=annual_value,
            optimizer_monthly=annual_value / 12.0,
            daily=annual_value / 252.0,
        )


@dataclass(frozen=True)
class BorrowRateSchedule:
    """Primary and total stress borrow rates for a security."""

    primary: PeriodicRate
    stress: PeriodicRate


@dataclass(frozen=True)
class FinancingCostSchedule:
    """Financing charges on gross exposure above one, in return units."""

    excess_gross_exposure: float
    primary: PeriodicRate
    stress: PeriodicRate


PRIMARY_BORROW_RATE = PeriodicRate.from_annual(0.01)
REGULAR_STRESS_BORROW_RATE = PeriodicRate.from_annual(0.03)
ILLIQUID_STRESS_BORROW_RATE = PeriodicRate.from_annual(0.10)
PRIMARY_FINANCING_RATE = PeriodicRate.from_annual(0.005)
STRESS_FINANCING_RATE = PeriodicRate.from_annual(0.015)


def _valid_observations(
    values: Iterable[float | None], *, latest_sessions: int
) -> np.ndarray:
    full_sequence = list(values)
    if len(full_sequence) < latest_sessions:
        return np.asarray([], dtype=np.float64)
    sequence = full_sequence[-latest_sessions:]
    valid: list[float] = []
    for value in sequence:
        finite = _finite_float(value)
        if finite is not None:
            valid.append(finite)
    return np.asarray(valid, dtype=np.float64)


def estimate_impact_volatility(
    latest_20_global_sessions: Iterable[float | None],
    latest_60_global_sessions: Iterable[float | None],
) -> ImpactVolatilityEstimate | None:
    """Estimate unannualized daily impact volatility using the frozen fallback.

    Missing or non-finite returns are omitted, but the session windows themselves
    remain global-session windows.  The 20-session estimate is selected whenever
    it has at least 15 valid observations.  Only when it misses that threshold is
    the 60-session fallback considered, with its minimum of 40 valid observations.
    """

    for window, minimum, values in (
        (20, 15, latest_20_global_sessions),
        (60, 40, latest_60_global_sessions),
    ):
        vector = _valid_observations(values, latest_sessions=window)
        if vector.size < minimum:
            continue
        volatility = float(np.std(vector, ddof=1))
        if math.isfinite(volatility) and volatility >= 0.0:
            return ImpactVolatilityEstimate(
                daily_volatility=volatility,
                source_window_sessions=window,
                valid_observations=int(vector.size),
            )
    return None


def required_exit_volatility_p95(
    cross_sectional_daily_volatilities: Iterable[float | None],
) -> float | None:
    """Return the type-7 cross-sectional 95th percentile for a required exit."""

    values: list[float] = []
    for value in cross_sectional_daily_volatilities:
        finite = _finite_float(value)
        if finite is not None and finite >= 0.0:
            values.append(finite)
    if not values:
        return None
    result = float(np.quantile(np.asarray(values), 0.95, method="linear"))
    return result if math.isfinite(result) and result >= 0.0 else None


def transaction_cost_return(
    delta_weight: float,
    *,
    full_spread: float | None,
    adv_dollars: float | None,
    impact_volatility: float | None,
    nav_dollars: float = REFERENCE_NAV_DOLLARS,
    scenario: TransactionCostScenario | str = PRIMARY_COST_SCENARIO,
    required_exit: bool = False,
    required_exit_volatility_fallback: float | None = None,
) -> float | None:
    """Estimate one-way transaction cost in NAV-return units.

    Nonzero new trades fail closed when spread, ADV, or impact volatility is
    unavailable or invalid.  A required exit may substitute the supplied type-7
    p95 volatility only when its security-level volatility is unavailable.
    """

    delta = _finite_float(delta_weight)
    if delta is None:
        return None
    absolute_delta = abs(delta)
    if absolute_delta == 0.0:
        return 0.0

    if isinstance(scenario, str):
        try:
            selected_scenario = TRANSACTION_COST_SCENARIOS[scenario]
        except KeyError as exc:
            raise ValueError(f"unknown transaction-cost scenario: {scenario}") from exc
    elif isinstance(scenario, TransactionCostScenario):
        selected_scenario = scenario
    else:
        raise TypeError("scenario must be a name or TransactionCostScenario")

    spread = _finite_float(full_spread)
    adv = _finite_float(adv_dollars)
    nav = _finite_float(nav_dollars)
    volatility = _finite_float(impact_volatility)
    if volatility is None or volatility < 0.0:
        fallback = _finite_float(required_exit_volatility_fallback)
        volatility = fallback if required_exit else None
    if (
        spread is None
        or spread < 0.0
        or adv is None
        or adv <= 0.0
        or nav is None
        or nav <= 0.0
        or volatility is None
        or volatility < 0.0
    ):
        return None

    participation = absolute_delta * nav / adv
    impact = (
        selected_scenario.impact_coefficient
        * volatility
        * math.sqrt(participation)
    )
    per_traded_dollar = (
        selected_scenario.spread_multiplier * spread / 2.0
        + COMMISSION_AND_OPERATIONAL_SLIPPAGE
        + impact
    )
    result = absolute_delta * per_traded_dollar
    return float(result) if math.isfinite(result) and result >= 0.0 else None


def borrow_rate_schedule(*, lowest_liquidity_quintile: bool) -> BorrowRateSchedule:
    """Return primary and total stress borrow rates for one security."""

    stress = (
        ILLIQUID_STRESS_BORROW_RATE
        if lowest_liquidity_quintile
        else REGULAR_STRESS_BORROW_RATE
    )
    return BorrowRateSchedule(primary=PRIMARY_BORROW_RATE, stress=stress)


def financing_cost_schedule(gross_exposure: float) -> FinancingCostSchedule | None:
    """Return financing charges on ``max(gross_exposure - 1, 0)``."""

    gross = _finite_float(gross_exposure)
    if gross is None or gross < 0.0:
        return None
    excess = max(gross - 1.0, 0.0)
    return FinancingCostSchedule(
        excess_gross_exposure=excess,
        primary=PeriodicRate.from_annual(excess * PRIMARY_FINANCING_RATE.annual),
        stress=PeriodicRate.from_annual(excess * STRESS_FINANCING_RATE.annual),
    )


def assign_adv_liquidity_quintiles(
    permnos: Sequence[int], adv_dollars: Sequence[float | None]
) -> np.ndarray:
    """Assign deterministic ADV quintiles, preserving the caller's row order.

    Ranks sort ascending by ADV and then PERMNO.  Missing and nonpositive ADV are
    treated as a single worst-liquidity group ordered by PERMNO.  With zero-based
    rank ``r`` among ``N`` observations, the assigned quintile is
    ``floor(5 * r / N) + 1``; quintile 1 is therefore least liquid.
    """

    if len(permnos) != len(adv_dollars):
        raise ValueError("PERMNO and ADV sequences must have equal length")
    size = len(permnos)
    if size == 0:
        return np.asarray([], dtype=np.int8)

    normalized_permnos: list[int] = []
    for value in permnos:
        finite = _finite_float(value)
        if finite is None or not finite.is_integer():
            raise ValueError("PERMNO values must be finite integers")
        normalized_permnos.append(int(finite))
    if len(set(normalized_permnos)) != size:
        raise ValueError("PERMNO values must be unique")

    def sort_key(index: int) -> tuple[int, float, int]:
        adv = _finite_float(adv_dollars[index])
        if adv is None or adv <= 0.0:
            return (0, 0.0, normalized_permnos[index])
        return (1, adv, normalized_permnos[index])

    ordered_indices = sorted(range(size), key=sort_key)
    result = np.empty(size, dtype=np.int8)
    for rank, original_index in enumerate(ordered_indices):
        result[original_index] = math.floor(5 * rank / size) + 1
    return result


__all__ = [
    "BorrowRateSchedule",
    "COMMISSION_AND_OPERATIONAL_SLIPPAGE",
    "FinancingCostSchedule",
    "ILLIQUID_STRESS_BORROW_RATE",
    "ImpactVolatilityEstimate",
    "NAV_SENSITIVITY_DOLLARS",
    "PRIMARY_BORROW_RATE",
    "PRIMARY_COST_SCENARIO",
    "PRIMARY_FINANCING_RATE",
    "PeriodicRate",
    "REFERENCE_NAV_DOLLARS",
    "REGULAR_STRESS_BORROW_RATE",
    "SEVERE_IMPACT_COST_SCENARIO",
    "STANDARD_STRESS_COST_SCENARIO",
    "STRESS_FINANCING_RATE",
    "TRANSACTION_COST_SCENARIOS",
    "TransactionCostScenario",
    "assign_adv_liquidity_quintiles",
    "borrow_rate_schedule",
    "estimate_impact_volatility",
    "financing_cost_schedule",
    "required_exit_volatility_p95",
    "transaction_cost_return",
]
