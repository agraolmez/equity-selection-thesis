"""Pure fixed-holding portfolio accounting for Stage 11 preflight tests.

The implementation is deliberately detached from storage and calendar code.
Positions are represented by signed dollar values.  This is algebraically the
same as holding fixed shares while applying an authoritative total-return
series: longs have positive values, shorts have negative values, and neither
side is renormalized between rebalances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from portfolio_bot.allocation.costs import (
    PRIMARY_FINANCING_ANNUAL_RATE,
    transaction_cost_numpy,
)


TRADING_SESSIONS_PER_YEAR = 252.0


class AccountingError(ValueError):
    """Raised when an accounting input would create an ambiguous ledger."""


def _readonly_vector(
    value: object,
    *,
    dtype: np.dtype | type,
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=dtype).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise AccountingError(f"{name} cannot be converted") from exc
    if result.ndim != 1:
        raise AccountingError(f"{name} must be one-dimensional")
    result.setflags(write=False)
    return result


def _finite_scalar(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AccountingError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise AccountingError(f"{name} must be finite")
    return result


def _same_length(expected: int, **values: np.ndarray) -> None:
    mismatches = sorted(name for name, value in values.items() if len(value) != expected)
    if mismatches:
        raise AccountingError(
            "accounting vector lengths differ for: " + ", ".join(mismatches)
        )


@dataclass(frozen=True, slots=True, eq=False)
class PortfolioState:
    """Immutable signed market values and cash at one instant.

    Zero-valued rows may remain in the state so the monthly problem and the
    daily-return matrix can retain one stable asset order.  They have no
    economic effect.
    """

    permno: np.ndarray
    position_value_dollars: np.ndarray
    cash_dollars: float

    def __post_init__(self) -> None:
        permno = _readonly_vector(self.permno, dtype=np.int64, name="permno")
        values = _readonly_vector(
            self.position_value_dollars,
            dtype=np.float64,
            name="position_value_dollars",
        )
        if len(permno) != len(values):
            raise AccountingError("permno and position values have different lengths")
        if np.any(permno <= 0) or len(np.unique(permno)) != len(permno):
            raise AccountingError("permno must be positive and unique")
        if not np.isfinite(values).all():
            raise AccountingError("position values must be finite")
        cash = _finite_scalar(self.cash_dollars, name="cash_dollars")
        nav = cash + float(np.sum(values, dtype=np.float64))
        if not np.isfinite(nav) or nav <= 0.0:
            raise AccountingError("portfolio NAV must remain positive and finite")
        object.__setattr__(self, "permno", permno)
        object.__setattr__(self, "position_value_dollars", values)
        object.__setattr__(self, "cash_dollars", cash)

    @classmethod
    def all_cash(cls, nav_dollars: float) -> "PortfolioState":
        """Create the mandated 100%-cash starting form for any positive NAV."""

        nav = _finite_scalar(nav_dollars, name="nav_dollars")
        if nav <= 0.0:
            raise AccountingError("nav_dollars must be positive")
        return cls(
            permno=np.empty(0, dtype=np.int64),
            position_value_dollars=np.empty(0, dtype=np.float64),
            cash_dollars=nav,
        )

    @property
    def nav_dollars(self) -> float:
        return self.cash_dollars + float(
            np.sum(self.position_value_dollars, dtype=np.float64)
        )

    @property
    def weights(self) -> np.ndarray:
        result = self.position_value_dollars / self.nav_dollars
        result.setflags(write=False)
        return result

    @property
    def cash_weight(self) -> float:
        return self.cash_dollars / self.nav_dollars

    @property
    def gross_exposure_dollars(self) -> float:
        return float(np.sum(np.abs(self.position_value_dollars), dtype=np.float64))


@dataclass(frozen=True, slots=True)
class RebalanceAccounting:
    """One execution-close rebalance with its single transaction-cost debit."""

    state: PortfolioState
    pre_trade_nav_dollars: float
    post_trade_nav_dollars: float
    transaction_cost_dollars: float
    transaction_cost_return: float
    one_way_turnover: float


@dataclass(frozen=True, slots=True)
class DailyAccounting:
    """One close-to-close drift step using opening holdings for daily charges."""

    state: PortfolioState
    opening_nav_dollars: float
    closing_nav_dollars: float
    net_total_return: float
    risk_free_return: float
    net_excess_return: float
    borrow_cost_dollars: float
    financing_cost_dollars: float


@dataclass(frozen=True, slots=True, eq=False)
class HoldingPeriodAccounting:
    """Complete rebalance-plus-holding-period accounting receipt."""

    state: PortfolioState
    pre_trade_nav_dollars: float
    daily_nav_dollars: np.ndarray
    transaction_cost_dollars: float
    borrow_cost_dollars: float
    financing_cost_dollars: float
    net_total_return: float
    risk_free_total_return: float
    net_excess_return: float

    def __post_init__(self) -> None:
        path = _readonly_vector(
            self.daily_nav_dollars,
            dtype=np.float64,
            name="daily_nav_dollars",
        )
        if not np.isfinite(path).all() or np.any(path <= 0.0):
            raise AccountingError("daily NAV path must be positive and finite")
        object.__setattr__(self, "daily_nav_dollars", path)

    @property
    def transaction_cost_return(self) -> float:
        return self.transaction_cost_dollars / self.pre_trade_nav_dollars

    @property
    def borrow_cost_return(self) -> float:
        return self.borrow_cost_dollars / self.pre_trade_nav_dollars

    @property
    def financing_cost_return(self) -> float:
        return self.financing_cost_dollars / self.pre_trade_nav_dollars


def _aligned_previous_values(
    state: PortfolioState,
    target_permno: np.ndarray,
) -> np.ndarray:
    target_index = {int(value): index for index, value in enumerate(target_permno)}
    missing = [
        int(permno)
        for permno, value in zip(
            state.permno,
            state.position_value_dollars,
            strict=True,
        )
        if value != 0.0 and int(permno) not in target_index
    ]
    if missing:
        raise AccountingError(
            "target universe omitted inherited nonzero holdings: "
            + ", ".join(str(value) for value in sorted(missing))
        )
    result = np.zeros(len(target_permno), dtype=np.float64)
    for permno, value in zip(
        state.permno,
        state.position_value_dollars,
        strict=True,
    ):
        index = target_index.get(int(permno))
        if index is not None:
            result[index] = float(value)
    return result


def rebalance_portfolio(
    state: PortfolioState,
    *,
    target_permno: Sequence[int] | np.ndarray,
    target_weights: Sequence[float] | np.ndarray,
    execution_price: Sequence[float] | np.ndarray,
    full_spread: Sequence[float] | np.ndarray,
    adv_dollars: Sequence[float] | np.ndarray,
    impact_volatility_daily: Sequence[float] | np.ndarray,
) -> RebalanceAccounting:
    """Execute one target and charge transaction costs exactly once.

    Target weights are fractions of *pre-cost* NAV, matching the Stage 10
    optimization convention.  A nonzero trade requires a finite positive
    execution price even though the subsequent total-return accounting can be
    performed directly in signed value space.
    """

    if not isinstance(state, PortfolioState):
        raise TypeError("state must be PortfolioState")
    permno = _readonly_vector(target_permno, dtype=np.int64, name="target_permno")
    weights = _readonly_vector(
        target_weights, dtype=np.float64, name="target_weights"
    )
    prices = _readonly_vector(execution_price, dtype=np.float64, name="execution_price")
    spread = _readonly_vector(full_spread, dtype=np.float64, name="full_spread")
    adv = _readonly_vector(adv_dollars, dtype=np.float64, name="adv_dollars")
    volatility = _readonly_vector(
        impact_volatility_daily,
        dtype=np.float64,
        name="impact_volatility_daily",
    )
    n = len(permno)
    _same_length(
        n,
        target_weights=weights,
        execution_price=prices,
        full_spread=spread,
        adv_dollars=adv,
        impact_volatility_daily=volatility,
    )
    if np.any(permno <= 0) or len(np.unique(permno)) != n:
        raise AccountingError("target_permno must be positive and unique")
    if not np.isfinite(weights).all():
        raise AccountingError("target_weights must be finite")

    pre_nav = state.nav_dollars
    previous_values = _aligned_previous_values(state, permno)
    previous_weights = previous_values / pre_nav
    delta = weights - previous_weights
    missing_execution = (~np.isfinite(prices)) | (prices <= 0.0)
    if np.any(missing_execution & (delta != 0.0)):
        blocked = permno[missing_execution & (delta != 0.0)]
        raise AccountingError(
            "nonzero trade lacks a valid execution price for PERMNO "
            + ", ".join(str(int(value)) for value in blocked)
        )

    transaction_return = float(
        transaction_cost_numpy(
            delta,
            full_spread=spread,
            adv_dollars=adv,
            impact_volatility=volatility,
            nav_dollars=pre_nav,
        )
    )
    transaction_dollars = pre_nav * transaction_return
    target_values = weights * pre_nav
    cash = pre_nav - float(np.sum(target_values, dtype=np.float64)) - transaction_dollars
    result = PortfolioState(
        permno=permno,
        position_value_dollars=target_values,
        cash_dollars=cash,
    )
    expected_post_nav = pre_nav - transaction_dollars
    if not np.isclose(
        result.nav_dollars,
        expected_post_nav,
        rtol=1e-13,
        atol=max(1e-8, pre_nav * 1e-14),
    ):
        raise AccountingError("rebalance ledger does not reconcile")
    return RebalanceAccounting(
        state=result,
        pre_trade_nav_dollars=pre_nav,
        post_trade_nav_dollars=result.nav_dollars,
        transaction_cost_dollars=transaction_dollars,
        transaction_cost_return=transaction_return,
        one_way_turnover=0.5 * float(np.sum(np.abs(delta), dtype=np.float64)),
    )


def apply_daily_accounting(
    state: PortfolioState,
    *,
    asset_total_returns: Sequence[float] | np.ndarray,
    daily_risk_free_rate: float,
    borrow_rate_annual: Sequence[float] | np.ndarray,
    financing_rate_annual: float = PRIMARY_FINANCING_ANNUAL_RATE,
) -> DailyAccounting:
    """Drift one day without rebalancing and debit opening-notional charges."""

    if not isinstance(state, PortfolioState):
        raise TypeError("state must be PortfolioState")
    returns = _readonly_vector(
        asset_total_returns,
        dtype=np.float64,
        name="asset_total_returns",
    )
    borrow_rates = _readonly_vector(
        borrow_rate_annual,
        dtype=np.float64,
        name="borrow_rate_annual",
    )
    _same_length(len(state.permno), returns=returns, borrow_rates=borrow_rates)
    if not np.isfinite(returns).all() or np.any(returns < -1.0):
        raise AccountingError(
            "asset total returns must be finite and no less than -100%"
        )
    if not np.isfinite(borrow_rates).all() or np.any(borrow_rates < 0.0):
        raise AccountingError("borrow rates must be finite and nonnegative")
    risk_free = _finite_scalar(daily_risk_free_rate, name="daily_risk_free_rate")
    if risk_free <= -1.0:
        raise AccountingError("daily_risk_free_rate must exceed -100%")
    financing = _finite_scalar(
        financing_rate_annual,
        name="financing_rate_annual",
    )
    if financing < 0.0:
        raise AccountingError("financing_rate_annual must be nonnegative")

    opening_nav = state.nav_dollars
    opening_values = state.position_value_dollars
    borrow_cost = float(
        np.sum(np.maximum(-opening_values, 0.0) * borrow_rates, dtype=np.float64)
        / TRADING_SESSIONS_PER_YEAR
    )
    financing_base = max(state.gross_exposure_dollars - opening_nav, 0.0)
    financing_cost = financing_base * financing / TRADING_SESSIONS_PER_YEAR
    closing_values = opening_values * (1.0 + returns)
    closing_cash = (
        state.cash_dollars * (1.0 + risk_free) - borrow_cost - financing_cost
    )
    closing_state = PortfolioState(
        permno=state.permno,
        position_value_dollars=closing_values,
        cash_dollars=closing_cash,
    )
    net_total = closing_state.nav_dollars / opening_nav - 1.0
    return DailyAccounting(
        state=closing_state,
        opening_nav_dollars=opening_nav,
        closing_nav_dollars=closing_state.nav_dollars,
        net_total_return=net_total,
        risk_free_return=risk_free,
        net_excess_return=net_total - risk_free,
        borrow_cost_dollars=borrow_cost,
        financing_cost_dollars=financing_cost,
    )


def run_rebalanced_holding_period(
    state: PortfolioState,
    *,
    target_permno: Sequence[int] | np.ndarray,
    target_weights: Sequence[float] | np.ndarray,
    execution_price: Sequence[float] | np.ndarray,
    full_spread: Sequence[float] | np.ndarray,
    adv_dollars: Sequence[float] | np.ndarray,
    impact_volatility_daily: Sequence[float] | np.ndarray,
    asset_total_returns: Sequence[Sequence[float]] | np.ndarray,
    daily_risk_free_rates: Sequence[float] | np.ndarray | float,
    borrow_rate_annual: Sequence[float] | np.ndarray,
    financing_rate_annual: float = PRIMARY_FINANCING_ANNUAL_RATE,
) -> HoldingPeriodAccounting:
    """Run one monthly target followed by fixed-holding daily accounting."""

    rebalance = rebalance_portfolio(
        state,
        target_permno=target_permno,
        target_weights=target_weights,
        execution_price=execution_price,
        full_spread=full_spread,
        adv_dollars=adv_dollars,
        impact_volatility_daily=impact_volatility_daily,
    )
    returns = np.asarray(asset_total_returns, dtype=np.float64)
    if returns.ndim == 1:
        returns = returns.reshape(1, -1)
    if returns.ndim != 2 or returns.shape[1] != len(rebalance.state.permno):
        raise AccountingError(
            "asset_total_returns must have shape (days, target assets)"
        )
    if returns.shape[0] == 0:
        raise AccountingError("a holding period must contain at least one trading day")
    if np.isscalar(daily_risk_free_rates):
        risk_free = np.full(returns.shape[0], float(daily_risk_free_rates))
    else:
        risk_free = np.asarray(daily_risk_free_rates, dtype=np.float64)
    if risk_free.ndim != 1 or len(risk_free) != returns.shape[0]:
        raise AccountingError("daily risk-free rates must have one value per day")

    current = rebalance.state
    nav_path: list[float] = []
    borrow_total = 0.0
    financing_total = 0.0
    for day, risk_free_rate in zip(returns, risk_free, strict=True):
        daily = apply_daily_accounting(
            current,
            asset_total_returns=day,
            daily_risk_free_rate=float(risk_free_rate),
            borrow_rate_annual=borrow_rate_annual,
            financing_rate_annual=financing_rate_annual,
        )
        current = daily.state
        nav_path.append(current.nav_dollars)
        borrow_total += daily.borrow_cost_dollars
        financing_total += daily.financing_cost_dollars

    risk_free_total = float(np.prod(1.0 + risk_free, dtype=np.float64) - 1.0)
    net_total = current.nav_dollars / rebalance.pre_trade_nav_dollars - 1.0
    return HoldingPeriodAccounting(
        state=current,
        pre_trade_nav_dollars=rebalance.pre_trade_nav_dollars,
        daily_nav_dollars=np.asarray(nav_path, dtype=np.float64),
        transaction_cost_dollars=rebalance.transaction_cost_dollars,
        borrow_cost_dollars=borrow_total,
        financing_cost_dollars=financing_total,
        net_total_return=net_total,
        risk_free_total_return=risk_free_total,
        net_excess_return=net_total - risk_free_total,
    )


__all__ = [
    "TRADING_SESSIONS_PER_YEAR",
    "AccountingError",
    "DailyAccounting",
    "HoldingPeriodAccounting",
    "PortfolioState",
    "RebalanceAccounting",
    "apply_daily_accounting",
    "rebalance_portfolio",
    "run_rebalanced_holding_period",
]
