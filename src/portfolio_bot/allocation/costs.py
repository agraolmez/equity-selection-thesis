"""Stage 10 allocation-cost primitives with three equivalent backends.

NumPy functions are the independent audit math.  CVXPY functions build the
convex optimizer terms, while Torch functions provide differentiable copies
for the decision-focused challenger.  This module calculates one-month cost
in NAV-return units only; it does not create trades, P&L, or a backtest.

Required-exit volatility is deliberately not estimated here.  Callers must
pass the already-resolved security estimate or frozen cross-sectional exit
fallback.  An unavailable or invalid market input makes a nonzero trade
prohibited, while an exactly zero trade remains feasible and costs zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cvxpy as cp
import numpy as np
from scipy import sparse
import torch
from numpy.typing import ArrayLike

from portfolio_bot.risk_cost.cost_math import (
    COMMISSION_AND_OPERATIONAL_SLIPPAGE,
    PRIMARY_BORROW_RATE,
    PRIMARY_COST_SCENARIO,
    PRIMARY_FINANCING_RATE,
    REFERENCE_NAV_DOLLARS,
    TRANSACTION_COST_SCENARIOS,
    TransactionCostScenario,
)


Reduction = Literal["sum", "none"]

COMMISSION_RATE = COMMISSION_AND_OPERATIONAL_SLIPPAGE
PRIMARY_BORROW_ANNUAL_RATE = PRIMARY_BORROW_RATE.annual
PRIMARY_FINANCING_ANNUAL_RATE = PRIMARY_FINANCING_RATE.annual


class CostInputError(ValueError):
    """Raised when an evaluated nonzero trade lacks valid cost inputs."""


@dataclass(frozen=True)
class CvxpyCostTerm:
    """A convex cost expression plus its fail-closed trade constraints."""

    per_asset: cp.Expression
    total: cp.Expression
    constraints: tuple[cp.Constraint, ...]
    trade_permitted: np.ndarray


@dataclass(frozen=True)
class NumpyTotalCost:
    """Auditable NumPy decomposition of one month's optimizer cost."""

    transaction: float
    borrow: float
    financing: float
    total: float


@dataclass(frozen=True)
class CvxpyTotalCost:
    """CVXPY total-cost decomposition and fail-closed trade constraints."""

    transaction: cp.Expression
    borrow: cp.Expression
    financing: cp.Expression
    total: cp.Expression
    constraints: tuple[cp.Constraint, ...]
    trade_permitted: np.ndarray


@dataclass(frozen=True)
class TorchTotalCost:
    """Differentiable Torch decomposition of one month's optimizer cost."""

    transaction: torch.Tensor
    borrow: torch.Tensor
    financing: torch.Tensor
    total: torch.Tensor


def _scenario_values(
    scenario: TransactionCostScenario | str,
) -> tuple[float, float]:
    if isinstance(scenario, str):
        try:
            selected = TRANSACTION_COST_SCENARIOS[scenario]
        except KeyError as exc:
            raise ValueError(f"unknown transaction-cost scenario: {scenario}") from exc
    elif isinstance(scenario, TransactionCostScenario):
        selected = scenario
    else:
        raise TypeError("scenario must be a name or TransactionCostScenario")
    return float(selected.spread_multiplier), float(selected.impact_coefficient)


def _validate_reduction(reduction: Reduction) -> None:
    if reduction not in {"sum", "none"}:
        raise ValueError("reduction must be 'sum' or 'none'")


def _float_array(value: object, *, name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric or missing") from exc


def _broadcast_numpy_inputs(
    delta_weight: ArrayLike,
    full_spread: ArrayLike | None,
    adv_dollars: ArrayLike | None,
    impact_volatility: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    delta = _float_array(delta_weight, name="delta_weight")
    spread = _float_array(full_spread, name="full_spread")
    adv = _float_array(adv_dollars, name="adv_dollars")
    volatility = _float_array(impact_volatility, name="impact_volatility")
    try:
        delta, spread, adv, volatility = np.broadcast_arrays(
            delta, spread, adv, volatility
        )
    except ValueError as exc:
        raise ValueError(
            "transaction-cost inputs are not broadcast-compatible"
        ) from exc
    if not np.isfinite(delta).all():
        raise CostInputError("delta_weight must be finite")
    return tuple(
        np.asarray(value, dtype=np.float64)
        for value in (delta, spread, adv, volatility)
    )  # type: ignore[return-value]


def _valid_nav(nav_dollars: object) -> tuple[float, bool]:
    try:
        nav = float(nav_dollars)
    except (TypeError, ValueError, OverflowError):
        return 0.0, False
    return (nav, True) if math.isfinite(nav) and nav > 0.0 else (0.0, False)


def _transaction_coefficients(
    full_spread: np.ndarray,
    adv_dollars: np.ndarray,
    impact_volatility: np.ndarray,
    *,
    nav_dollars: object,
    scenario: TransactionCostScenario | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spread_multiplier, impact_coefficient = _scenario_values(scenario)
    nav, nav_valid = _valid_nav(nav_dollars)
    permitted = (
        np.isfinite(full_spread)
        & (full_spread >= 0.0)
        & np.isfinite(adv_dollars)
        & (adv_dollars > 0.0)
        & np.isfinite(impact_volatility)
        & (impact_volatility >= 0.0)
        & nav_valid
    )
    safe_spread = np.where(permitted, full_spread, 0.0)
    safe_adv = np.where(permitted, adv_dollars, 1.0)
    safe_volatility = np.where(permitted, impact_volatility, 0.0)
    linear_coefficient = spread_multiplier * safe_spread / 2.0 + COMMISSION_RATE
    impact_scale = (
        impact_coefficient * safe_volatility * np.sqrt(nav / safe_adv)
        if nav_valid
        else np.zeros_like(safe_volatility)
    )
    linear_coefficient = np.where(permitted, linear_coefficient, 0.0)
    impact_scale = np.where(permitted, impact_scale, 0.0)
    return linear_coefficient, impact_scale, np.asarray(permitted, dtype=np.bool_)


def transaction_cost_numpy(
    delta_weight: ArrayLike,
    *,
    full_spread: ArrayLike | None,
    adv_dollars: ArrayLike | None,
    impact_volatility: ArrayLike | None,
    nav_dollars: float = REFERENCE_NAV_DOLLARS,
    scenario: TransactionCostScenario | str = PRIMARY_COST_SCENARIO,
    reduction: Reduction = "sum",
) -> float | np.ndarray:
    """Evaluate frozen transaction cost with scalar or vector NumPy audit math.

    The algebra is ``a*|dw| + k*|dw|**1.5``, where ``a`` contains the
    half-spread and one-basis-point charge and ``k`` contains square-root
    impact, daily volatility, NAV, and ADV.
    """

    _validate_reduction(reduction)
    delta, spread, adv, volatility = _broadcast_numpy_inputs(
        delta_weight, full_spread, adv_dollars, impact_volatility
    )
    linear, impact, permitted = _transaction_coefficients(
        spread,
        adv,
        volatility,
        nav_dollars=nav_dollars,
        scenario=scenario,
    )
    if np.any((~permitted) & (delta != 0.0)):
        raise CostInputError(
            "nonzero trade lacks valid spread, ADV, volatility, or NAV"
        )
    absolute_delta = np.abs(delta)
    per_asset = linear * absolute_delta + impact * np.power(absolute_delta, 1.5)
    if reduction == "sum":
        return float(np.sum(per_asset, dtype=np.float64))
    result = np.array(per_asset, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _cvxpy_expression(value: cp.Expression | ArrayLike) -> cp.Expression:
    return value if isinstance(value, cp.Expression) else cp.Constant(value)


def _broadcast_to_expression_shape(
    value: ArrayLike | None,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = _float_array(value, name=name)
    try:
        return np.asarray(np.broadcast_to(array, shape), dtype=np.float64)
    except ValueError as exc:
        raise ValueError(
            f"{name} does not broadcast to decision shape {shape}"
        ) from exc


def transaction_cost_cvxpy(
    delta_weight: cp.Expression | ArrayLike,
    *,
    full_spread: ArrayLike | None,
    adv_dollars: ArrayLike | None,
    impact_volatility: ArrayLike | None,
    nav_dollars: float = REFERENCE_NAV_DOLLARS,
    scenario: TransactionCostScenario | str = PRIMARY_COST_SCENARIO,
) -> CvxpyCostTerm:
    """Build the convex transaction-cost term and invalid-trade equalities.

    Invalid market inputs are replaced only inside zeroed coefficients and are
    paired with an explicit ``delta_weight == 0`` constraint.  Consequently
    they cannot silently make a nonzero optimizer trade free.
    """

    delta = _cvxpy_expression(delta_weight)
    if len(delta.shape) > 1:
        raise ValueError("delta_weight must be scalar or one-dimensional")
    shape = delta.shape
    spread = _broadcast_to_expression_shape(
        full_spread, shape=shape, name="full_spread"
    )
    adv = _broadcast_to_expression_shape(adv_dollars, shape=shape, name="adv_dollars")
    volatility = _broadcast_to_expression_shape(
        impact_volatility, shape=shape, name="impact_volatility"
    )
    linear, impact, permitted = _transaction_coefficients(
        spread,
        adv,
        volatility,
        nav_dollars=nav_dollars,
        scenario=scenario,
    )
    # Construct only economically active atoms.  Multiplying an invalid row's
    # power-cone atom by zero still leaves a degenerate cone in the canonical
    # problem and can turn an otherwise exact CLARABEL solve into
    # ``optimal_inaccurate``.  Invalid rows are instead represented by a zero
    # constant plus the explicit no-trade equality below.
    if permitted.ndim == 0:
        if bool(permitted):
            absolute_delta = cp.abs(delta)
            impact_term = (
                float(impact) * cp.power(absolute_delta, 1.5)
                if float(impact) > 0.0
                else cp.Constant(0.0)
            )
            per_asset = float(linear) * absolute_delta + impact_term
        else:
            per_asset = cp.Constant(0.0)
    else:
        size = int(np.prod(delta.shape))
        permitted_indices = np.flatnonzero(permitted)
        if permitted_indices.size:
            # Preserve the per-security output while presenting one vectorized
            # expression graph to CVXPY instead of hundreds of scalar atoms.
            absolute = cp.abs(delta[permitted_indices])
            active_terms: cp.Expression = cp.multiply(
                linear[permitted_indices], absolute
            )
            positive_positions = np.flatnonzero(
                impact[permitted_indices] > 0.0
            )
            if positive_positions.size:
                impact_terms = cp.multiply(
                    impact[permitted_indices[positive_positions]],
                    cp.power(absolute[positive_positions], 1.5),
                )
                impact_scatter = sparse.csr_matrix(
                    (
                        np.ones(positive_positions.size, dtype=np.float64),
                        (positive_positions, np.arange(positive_positions.size)),
                    ),
                    shape=(permitted_indices.size, positive_positions.size),
                )
                active_terms = active_terms + impact_scatter @ impact_terms
            output_scatter = sparse.csr_matrix(
                (
                    np.ones(permitted_indices.size, dtype=np.float64),
                    (permitted_indices, np.arange(permitted_indices.size)),
                ),
                shape=(size, permitted_indices.size),
            )
            per_asset = output_scatter @ active_terms
        else:
            per_asset = cp.Constant(np.zeros(size, dtype=np.float64))
    constraints: list[cp.Constraint] = []
    if permitted.ndim == 0:
        if not bool(permitted):
            constraints.append(delta == 0.0)
    else:
        invalid_indices = np.flatnonzero(~permitted)
        if invalid_indices.size:
            constraints.append(delta[invalid_indices] == 0.0)
    mask = np.array(permitted, dtype=np.bool_, copy=True)
    mask.setflags(write=False)
    return CvxpyCostTerm(
        per_asset=per_asset,
        total=cp.sum(per_asset),
        constraints=tuple(constraints),
        trade_permitted=mask,
    )


def _torch_delta(value: torch.Tensor | ArrayLike) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        # Torch warns (and permits undefined behaviour) when a read-only NumPy
        # view is wrapped without copying.  Stage 10 domain arrays are
        # deliberately immutable, so make a private writable copy first.
        source = (
            np.array(value, copy=True)
            if isinstance(value, np.ndarray) and not value.flags.writeable
            else value
        )
        tensor = torch.as_tensor(source)
    if tensor.is_complex():
        raise CostInputError("delta_weight must be real")
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all().detach().cpu().item()):
        raise CostInputError("delta_weight must be finite")
    return tensor


def _torch_market_input(
    value: torch.Tensor | ArrayLike | None,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        return torch.full_like(reference, float("nan"))
    source = (
        np.array(value, copy=True)
        if isinstance(value, np.ndarray) and not value.flags.writeable
        else value
    )
    return torch.as_tensor(source, dtype=reference.dtype, device=reference.device)


def transaction_cost_torch(
    delta_weight: torch.Tensor | ArrayLike,
    *,
    full_spread: torch.Tensor | ArrayLike | None,
    adv_dollars: torch.Tensor | ArrayLike | None,
    impact_volatility: torch.Tensor | ArrayLike | None,
    nav_dollars: float = REFERENCE_NAV_DOLLARS,
    scenario: TransactionCostScenario | str = PRIMARY_COST_SCENARIO,
    reduction: Reduction = "sum",
) -> torch.Tensor:
    """Evaluate the differentiable Torch equivalent of transaction cost."""

    _validate_reduction(reduction)
    delta = _torch_delta(delta_weight)
    spread = _torch_market_input(full_spread, reference=delta)
    adv = _torch_market_input(adv_dollars, reference=delta)
    volatility = _torch_market_input(impact_volatility, reference=delta)
    try:
        delta, spread, adv, volatility = torch.broadcast_tensors(
            delta, spread, adv, volatility
        )
    except RuntimeError as exc:
        raise ValueError(
            "transaction-cost inputs are not broadcast-compatible"
        ) from exc
    spread_multiplier, impact_coefficient = _scenario_values(scenario)
    nav, nav_valid = _valid_nav(nav_dollars)
    permitted = (
        torch.isfinite(spread)
        & (spread >= 0.0)
        & torch.isfinite(adv)
        & (adv > 0.0)
        & torch.isfinite(volatility)
        & (volatility >= 0.0)
    )
    if not nav_valid:
        permitted = torch.zeros_like(permitted, dtype=torch.bool)
    invalid_nonzero = (~permitted) & (delta != 0.0)
    if bool(invalid_nonzero.any().detach().cpu().item()):
        raise CostInputError(
            "nonzero trade lacks valid spread, ADV, volatility, or NAV"
        )
    safe_spread = torch.where(permitted, spread, torch.zeros_like(spread))
    safe_adv = torch.where(permitted, adv, torch.ones_like(adv))
    safe_volatility = torch.where(permitted, volatility, torch.zeros_like(volatility))
    linear = spread_multiplier * safe_spread / 2.0 + COMMISSION_RATE
    impact = (
        impact_coefficient
        * safe_volatility
        * torch.sqrt(
            torch.as_tensor(nav, dtype=delta.dtype, device=delta.device) / safe_adv
        )
    )
    linear = torch.where(permitted, linear, torch.zeros_like(linear))
    impact = torch.where(permitted, impact, torch.zeros_like(impact))
    absolute_delta = torch.abs(delta)
    per_asset = linear * absolute_delta + impact * absolute_delta.pow(1.5)
    return per_asset.sum() if reduction == "sum" else per_asset


def _annual_rate_array(annual_rate: ArrayLike, *, name: str) -> np.ndarray:
    rate = _float_array(annual_rate, name=name)
    if not np.isfinite(rate).all() or np.any(rate < 0.0):
        raise CostInputError(f"{name} must be finite and nonnegative")
    return rate


def borrow_cost_numpy(
    weight: ArrayLike,
    *,
    annual_rate: ArrayLike = PRIMARY_BORROW_ANNUAL_RATE,
    reduction: Reduction = "sum",
) -> float | np.ndarray:
    """Charge annual-rate/12 on positive short notional only."""

    _validate_reduction(reduction)
    weights = _float_array(weight, name="weight")
    rates = _annual_rate_array(annual_rate, name="annual_rate")
    try:
        weights, rates = np.broadcast_arrays(weights, rates)
    except ValueError as exc:
        raise ValueError("weight and annual_rate are not broadcast-compatible") from exc
    if not np.isfinite(weights).all():
        raise CostInputError("weight must be finite")
    per_asset = np.maximum(-weights, 0.0) * rates / 12.0
    if reduction == "sum":
        return float(np.sum(per_asset, dtype=np.float64))
    result = np.array(per_asset, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def borrow_cost_cvxpy(
    weight: cp.Expression | ArrayLike,
    *,
    annual_rate: ArrayLike = PRIMARY_BORROW_ANNUAL_RATE,
) -> cp.Expression:
    """Build the convex monthly borrow-cost expression."""

    weights = _cvxpy_expression(weight)
    if len(weights.shape) > 1:
        raise ValueError("weight must be scalar or one-dimensional")
    rates = _annual_rate_array(annual_rate, name="annual_rate")
    try:
        rates = np.asarray(np.broadcast_to(rates, weights.shape), dtype=np.float64)
    except ValueError as exc:
        raise ValueError("annual_rate does not broadcast to weight shape") from exc
    return cp.sum(cp.multiply(rates / 12.0, cp.pos(-weights)))


def borrow_cost_torch(
    weight: torch.Tensor | ArrayLike,
    *,
    annual_rate: torch.Tensor | ArrayLike = PRIMARY_BORROW_ANNUAL_RATE,
    reduction: Reduction = "sum",
) -> torch.Tensor:
    """Evaluate the differentiable monthly Torch borrow cost."""

    _validate_reduction(reduction)
    weights = _torch_delta(weight)
    rate_source = (
        np.array(annual_rate, copy=True)
        if isinstance(annual_rate, np.ndarray) and not annual_rate.flags.writeable
        else annual_rate
    )
    rates = torch.as_tensor(
        rate_source,
        dtype=weights.dtype,
        device=weights.device,
    )
    try:
        weights, rates = torch.broadcast_tensors(weights, rates)
    except RuntimeError as exc:
        raise ValueError("weight and annual_rate are not broadcast-compatible") from exc
    if not bool((torch.isfinite(rates) & (rates >= 0.0)).all().detach().cpu().item()):
        raise CostInputError("annual_rate must be finite and nonnegative")
    # ReLU chooses the neutral zero subgradient at a zero holding, unlike
    # torch.clamp whose boundary subgradient is one in current Torch releases.
    per_asset = torch.relu(-weights) * rates / 12.0
    return per_asset.sum() if reduction == "sum" else per_asset


def _annual_rate_scalar(annual_rate: object, *, name: str) -> float:
    try:
        rate = float(annual_rate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CostInputError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(rate) or rate < 0.0:
        raise CostInputError(f"{name} must be finite and nonnegative")
    return rate


def financing_cost_numpy(
    weight: ArrayLike,
    *,
    annual_rate: float = PRIMARY_FINANCING_ANNUAL_RATE,
) -> float:
    """Charge annual-rate/12 on gross exposure above one NAV."""

    weights = _float_array(weight, name="weight")
    if not np.isfinite(weights).all():
        raise CostInputError("weight must be finite")
    rate = _annual_rate_scalar(annual_rate, name="annual_rate")
    gross = float(np.sum(np.abs(weights), dtype=np.float64))
    return max(gross - 1.0, 0.0) * rate / 12.0


def financing_cost_cvxpy(
    weight: cp.Expression | ArrayLike,
    *,
    annual_rate: float = PRIMARY_FINANCING_ANNUAL_RATE,
) -> cp.Expression:
    """Build the convex monthly financing-cost expression."""

    weights = _cvxpy_expression(weight)
    if len(weights.shape) > 1:
        raise ValueError("weight must be scalar or one-dimensional")
    rate = _annual_rate_scalar(annual_rate, name="annual_rate")
    return cp.pos(cp.norm1(weights) - 1.0) * (rate / 12.0)


def financing_cost_torch(
    weight: torch.Tensor | ArrayLike,
    *,
    annual_rate: float = PRIMARY_FINANCING_ANNUAL_RATE,
) -> torch.Tensor:
    """Evaluate the differentiable monthly Torch financing cost."""

    weights = _torch_delta(weight)
    rate = _annual_rate_scalar(annual_rate, name="annual_rate")
    gross = torch.sum(torch.abs(weights))
    return torch.clamp(gross - 1.0, min=0.0) * (rate / 12.0)


def _custom_cost_scenario(
    *,
    impact_coefficient: float,
    spread_multiplier: float,
) -> TransactionCostScenario:
    return TransactionCostScenario(
        name="stage10_total_cost",
        spread_multiplier=spread_multiplier,
        impact_coefficient=impact_coefficient,
    )


def numpy_total_cost(
    weights: ArrayLike,
    previous_weights: ArrayLike,
    full_spread: ArrayLike | None,
    adv20_usd: ArrayLike | None,
    impact_volatility_daily: ArrayLike | None,
    borrow_rate_annual: ArrayLike,
    nav_dollars: float,
    impact_coefficient: float = 0.10,
    spread_multiplier: float = 1.0,
    financing_rate_annual: float = 0.005,
) -> NumpyTotalCost:
    """Return the complete NumPy optimizer-cost decomposition."""

    current = _float_array(weights, name="weights")
    previous = _float_array(previous_weights, name="previous_weights")
    try:
        current, previous = np.broadcast_arrays(current, previous)
    except ValueError as exc:
        raise ValueError(
            "weights and previous_weights are not broadcast-compatible"
        ) from exc
    if not np.isfinite(current).all() or not np.isfinite(previous).all():
        raise CostInputError("weights and previous_weights must be finite")
    scenario = _custom_cost_scenario(
        impact_coefficient=impact_coefficient,
        spread_multiplier=spread_multiplier,
    )
    transaction = float(
        transaction_cost_numpy(
            current - previous,
            full_spread=full_spread,
            adv_dollars=adv20_usd,
            impact_volatility=impact_volatility_daily,
            nav_dollars=nav_dollars,
            scenario=scenario,
        )
    )
    borrow = float(borrow_cost_numpy(current, annual_rate=borrow_rate_annual))
    financing = financing_cost_numpy(current, annual_rate=financing_rate_annual)
    return NumpyTotalCost(
        transaction=transaction,
        borrow=borrow,
        financing=financing,
        total=transaction + borrow + financing,
    )


def cvxpy_total_cost(
    weights: cp.Expression | ArrayLike,
    previous_weights: cp.Expression | ArrayLike,
    full_spread: ArrayLike | None,
    adv20_usd: ArrayLike | None,
    impact_volatility_daily: ArrayLike | None,
    borrow_rate_annual: ArrayLike,
    nav_dollars: float,
    impact_coefficient: float = 0.10,
    spread_multiplier: float = 1.0,
    financing_rate_annual: float = 0.005,
) -> CvxpyTotalCost:
    """Return all CVXPY optimizer-cost terms and invalid-trade constraints."""

    current = _cvxpy_expression(weights)
    previous = _cvxpy_expression(previous_weights)
    if current.shape != previous.shape:
        raise ValueError("weights and previous_weights must have identical shapes")
    scenario = _custom_cost_scenario(
        impact_coefficient=impact_coefficient,
        spread_multiplier=spread_multiplier,
    )
    transaction_term = transaction_cost_cvxpy(
        current - previous,
        full_spread=full_spread,
        adv_dollars=adv20_usd,
        impact_volatility=impact_volatility_daily,
        nav_dollars=nav_dollars,
        scenario=scenario,
    )
    borrow = borrow_cost_cvxpy(current, annual_rate=borrow_rate_annual)
    financing = financing_cost_cvxpy(current, annual_rate=financing_rate_annual)
    total = transaction_term.total + borrow + financing
    return CvxpyTotalCost(
        transaction=transaction_term.total,
        borrow=borrow,
        financing=financing,
        total=total,
        constraints=transaction_term.constraints,
        trade_permitted=transaction_term.trade_permitted,
    )


def torch_total_cost(
    weights: torch.Tensor | ArrayLike,
    previous_weights: torch.Tensor | ArrayLike,
    full_spread: torch.Tensor | ArrayLike | None,
    adv20_usd: torch.Tensor | ArrayLike | None,
    impact_volatility_daily: torch.Tensor | ArrayLike | None,
    borrow_rate_annual: torch.Tensor | ArrayLike,
    nav_dollars: float,
    impact_coefficient: float = 0.10,
    spread_multiplier: float = 1.0,
    financing_rate_annual: float = 0.005,
) -> TorchTotalCost:
    """Return the complete differentiable Torch cost decomposition."""

    current = _torch_delta(weights)
    previous_source = (
        np.array(previous_weights, copy=True)
        if isinstance(previous_weights, np.ndarray)
        and not previous_weights.flags.writeable
        else previous_weights
    )
    previous = torch.as_tensor(
        previous_source,
        dtype=current.dtype,
        device=current.device,
    )
    try:
        current, previous = torch.broadcast_tensors(current, previous)
    except RuntimeError as exc:
        raise ValueError(
            "weights and previous_weights are not broadcast-compatible"
        ) from exc
    if not bool(torch.isfinite(previous).all().detach().cpu().item()):
        raise CostInputError("previous_weights must be finite")
    scenario = _custom_cost_scenario(
        impact_coefficient=impact_coefficient,
        spread_multiplier=spread_multiplier,
    )
    transaction = transaction_cost_torch(
        current - previous,
        full_spread=full_spread,
        adv_dollars=adv20_usd,
        impact_volatility=impact_volatility_daily,
        nav_dollars=nav_dollars,
        scenario=scenario,
    )
    borrow = borrow_cost_torch(current, annual_rate=borrow_rate_annual)
    financing = financing_cost_torch(current, annual_rate=financing_rate_annual)
    return TorchTotalCost(
        transaction=transaction,
        borrow=borrow,
        financing=financing,
        total=transaction + borrow + financing,
    )


__all__ = [
    "COMMISSION_RATE",
    "CvxpyCostTerm",
    "CvxpyTotalCost",
    "CostInputError",
    "NumpyTotalCost",
    "PRIMARY_BORROW_ANNUAL_RATE",
    "PRIMARY_FINANCING_ANNUAL_RATE",
    "TorchTotalCost",
    "borrow_cost_cvxpy",
    "borrow_cost_numpy",
    "borrow_cost_torch",
    "cvxpy_total_cost",
    "financing_cost_cvxpy",
    "financing_cost_numpy",
    "financing_cost_torch",
    "numpy_total_cost",
    "torch_total_cost",
    "transaction_cost_cvxpy",
    "transaction_cost_numpy",
    "transaction_cost_torch",
]
