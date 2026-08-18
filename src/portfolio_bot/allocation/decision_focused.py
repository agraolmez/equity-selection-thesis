"""Decision-focused allocation challenger for Stage 10.

The differentiable SCS projection in this module is a training primitive only.
Executable inference always re-solves every seed and the final ensemble with
the common deterministic CLARABEL path in :mod:`portfolio_bot.allocation.solver`.
The training helpers are data-source agnostic and therefore can be exercised
with fictitious in-memory observations without opening any empirical payload.
"""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Callable, Iterable, Protocol, Sequence

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from torch import nn

from portfolio_bot.allocation.constraints import (
    independent_constraint_diagnostics,
    ordinary_constraints,
)
from portfolio_bot.allocation.costs import (
    PRIMARY_FINANCING_ANNUAL_RATE,
    TorchTotalCost,
    torch_total_cost,
    transaction_cost_torch,
)
from portfolio_bot.allocation.domain import AllocationProblemData
from portfolio_bot.allocation.solver import (
    PROJECTION_RIDGE,
    AllocationResult,
    AllocationSolveError,
    solve_projection,
)


HIDDEN_LAYERS = (32, 16)
DROPOUT = 0.10
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
PATIENCE = 20
EARLY_STOPPING_MONTHS = 24
DECISION_SEEDS = (11, 23, 47, 71, 101)
OUTPUTS_PER_SECURITY = 3
ANNUALIZATION_PERIODS = 12.0
TRADING_SESSIONS_PER_YEAR = 252.0
SHARPE_VARIANCE_RIDGE = 1e-8
TRAINING_PROJECTION_TOLERANCE = 5e-5
WEIGHT_HALF_LIFE_MONTHS = 120.0
QUARTERLY_REFIT_MONTHS = (3, 6, 9, 12)
# With an immutable-prior CvxpyLayer, the stopped prior changes the trade and
# turnover constraints every month and every epoch.  Reusing such a layer
# would be incorrect; rebuilding and CLARABEL-preflighting it hundreds of
# thousands of times is not a production-safe Stage 11 path.  Keep empirical
# execution closed until a separately verified prior-parameterized layer is
# implemented.  The complete lifecycle remains runnable only on explicit
# synthetic fixtures for Stage 10 verification.
PRODUCTION_TRAINING_RUNTIME_CERTIFIED = False

_SCS_SOLVER_ARGS: dict[str, object] = {
    "solve_method": "SCS",
    "eps_abs": 1e-6,
    "eps_rel": 1e-6,
    "eps_infeas": 1e-7,
    "max_iters": 100_000,
    "use_indirect": False,
    "normalize": True,
    "adaptive_scale": False,
    "acceleration_lookback": 0,
}


class DecisionFocusedInactiveError(RuntimeError):
    """The challenger cannot safely produce a training or inference weight."""


def _vocabulary(values: Sequence[object], *, name: str) -> tuple[str, ...]:
    result = tuple(str(value).strip().upper() for value in values)
    if not result or any(not value for value in result):
        raise ValueError(f"{name} must be nonempty and contain nonblank labels")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique labels")
    return result


def _reference_parameter(module: nn.Module) -> torch.Tensor:
    try:
        return next(module.parameters())
    except StopIteration as exc:  # pragma: no cover - fixed network has parameters
        raise RuntimeError("decision-focused network has no parameters") from exc


def _torch_copy(
    value: torch.Tensor | np.ndarray | Sequence[float] | float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Convert read-only NumPy inputs without exposing non-writable storage."""

    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    array = np.array(value, copy=True)
    return torch.as_tensor(array, dtype=dtype, device=device)


def decision_features(
    data: AllocationProblemData,
    *,
    sector_vocabulary: Sequence[str],
    country_vocabulary: Sequence[str],
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode score, six betas, risk, liquidity, holding, and categories."""

    if not isinstance(data, AllocationProblemData):
        raise TypeError("data must be AllocationProblemData")
    sectors = _vocabulary(sector_vocabulary, name="sector_vocabulary")
    countries = _vocabulary(country_vocabulary, name="country_vocabulary")
    unknown_sector = sorted(set(data.sector.tolist()).difference(sectors))
    unknown_country = sorted(set(data.country.tolist()).difference(countries))
    if unknown_sector:
        raise ValueError(
            f"sector labels are outside the frozen vocabulary: {unknown_sector}"
        )
    if unknown_country:
        raise ValueError(
            f"country labels are outside the frozen vocabulary: {unknown_country}"
        )

    liquidity = np.divide(
        data.adv20_usd,
        data.nav_dollars,
        out=np.zeros(data.n_assets, dtype=np.float64),
        where=np.isfinite(data.adv20_usd) & (data.adv20_usd > 0.0),
    )
    numeric = np.column_stack(
        (
            data.score,
            data.factor_betas,
            data.forecast_volatility,
            liquidity,
            data.previous_weight,
        )
    )
    sector_columns = np.column_stack(
        [data.sector == label for label in sectors]
    ).astype(np.float64)
    country_columns = np.column_stack(
        [data.country == label for label in countries]
    ).astype(np.float64)
    encoded = np.column_stack((numeric, sector_columns, country_columns))
    if not np.isfinite(encoded).all():
        raise ValueError("decision-focused features must be finite")
    return torch.as_tensor(np.array(encoded, copy=True), dtype=dtype, device=device)


@dataclass(frozen=True)
class DecisionRawAllocation:
    security_outputs: torch.Tensor
    raw_weights: torch.Tensor
    long_gate: torch.Tensor
    short_gate: torch.Tensor


class DecisionFocusedAllocator(nn.Module):
    """Frozen shared-security MLP with a logit and two pooled gross gates."""

    def __init__(
        self,
        *,
        sector_vocabulary: Sequence[str],
        country_vocabulary: Sequence[str],
        seed: int,
    ) -> None:
        super().__init__()
        if isinstance(seed, bool) or int(seed) not in DECISION_SEEDS:
            raise ValueError(f"seed must be one of {DECISION_SEEDS}")
        self.seed = int(seed)
        self.sector_vocabulary = _vocabulary(
            sector_vocabulary, name="sector_vocabulary"
        )
        self.country_vocabulary = _vocabulary(
            country_vocabulary, name="country_vocabulary"
        )
        input_width = 10 + len(self.sector_vocabulary) + len(self.country_vocabulary)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network = nn.Sequential(
                nn.Linear(input_width, HIDDEN_LAYERS[0]),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1]),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_LAYERS[1], OUTPUTS_PER_SECURITY),
            )

    @property
    def input_width(self) -> int:
        return int(self.network[0].in_features)  # type: ignore[union-attr]

    def encode(self, data: AllocationProblemData) -> torch.Tensor:
        reference = _reference_parameter(self)
        return decision_features(
            data,
            sector_vocabulary=self.sector_vocabulary,
            country_vocabulary=self.country_vocabulary,
            dtype=reference.dtype,
            device=reference.device,
        )

    def raw_allocation(self, data: AllocationProblemData) -> DecisionRawAllocation:
        output = self.network(self.encode(data))
        if output.shape != (data.n_assets, OUTPUTS_PER_SECURITY):
            raise RuntimeError("decision-focused network output shape differs")
        active = ~data.reduction_only
        long_mask = torch.as_tensor(
            np.array(data.long_mask & active, copy=True),
            dtype=torch.bool,
            device=output.device,
        )
        short_mask = torch.as_tensor(
            np.array(data.short_mask & active, copy=True),
            dtype=torch.bool,
            device=output.device,
        )
        raw = torch.zeros(data.n_assets, dtype=output.dtype, device=output.device)
        zero = output.new_zeros(())
        if bool(long_mask.any().detach().cpu().item()):
            long_gate = torch.sigmoid(output[long_mask, 1].mean())
            long_notional = torch.softmax(output[long_mask, 0], dim=0)
            raw = raw.masked_scatter(long_mask, long_gate * long_notional)
        else:
            long_gate = zero
        if bool(short_mask.any().detach().cpu().item()):
            short_gate = torch.sigmoid(output[short_mask, 2].mean())
            short_notional = torch.softmax(output[short_mask, 0], dim=0)
            raw = raw.masked_scatter(short_mask, -short_gate * short_notional)
        else:
            short_gate = zero
        return DecisionRawAllocation(output, raw, long_gate, short_gate)

    def forward(self, data: AllocationProblemData) -> torch.Tensor:
        return self.raw_allocation(data).raw_weights


class DifferentiableProjection(nn.Module):
    """SCS/diffcp training projection; its output is never executable.

    Construction first proves that the ordinary constraint set is reachable by
    the deterministic CLARABEL path.  A genuinely inherited infeasibility (or
    an inability to certify ordinary feasibility) makes this experimental
    challenger explicitly inactive rather than invoking emergency slacks.
    """

    def __init__(self, data: AllocationProblemData) -> None:
        super().__init__()
        if not isinstance(data, AllocationProblemData):
            raise TypeError("data must be AllocationProblemData")
        self.data = data
        self.problem: cp.Problem | None = None
        self.raw_parameter: cp.Parameter | None = None
        self.weight_variable: cp.Variable | None = None
        self.layer: CvxpyLayer | None = None

        try:
            feasibility = solve_projection(
                data,
                data.previous_weight,
                architecture="decision_focused_training_preflight",
            )
        except (AllocationSolveError, ValueError) as exc:
            raise DecisionFocusedInactiveError(
                "ordinary differentiable projection feasibility could not be certified"
            ) from exc
        if feasibility.mode != "ordinary":
            raise DecisionFocusedInactiveError(
                "ordinary differentiable projection is infeasible under inherited state"
            )
        if data.n_assets == 0:
            return

        weights = cp.Variable(data.n_assets, name="decision_focused_weights")
        raw_target = cp.Parameter(data.n_assets, name="raw_target")
        caps = np.where(data.long_mask, 0.0100, 0.0075)
        objective = cp.Minimize(
            0.5 * cp.sum_squares(cp.multiply(1.0 / caps, weights - raw_target))
            + PROJECTION_RIDGE * cp.sum_squares(weights)
        )
        problem = cp.Problem(objective, ordinary_constraints(data, weights))
        if not problem.is_dcp() or not problem.is_dpp():
            raise DecisionFocusedInactiveError(
                "common decision-focused projection is not DCP/DPP"
            )
        self.problem = problem
        self.raw_parameter = raw_target
        self.weight_variable = weights
        try:
            self.layer = CvxpyLayer(
                problem,
                parameters=[raw_target],
                variables=[weights],
                solver="DIFFCP",
                solver_args=dict(_SCS_SOLVER_ARGS),
            )
        except Exception as exc:  # dependency incompatibility means inactive
            raise DecisionFocusedInactiveError(
                "CVXPYLayers/diffcp training projection is unavailable"
            ) from exc

    @property
    def parameter_count(self) -> int:
        return 0 if self.problem is None else len(self.problem.parameters())

    def project_for_training(self, raw_target: torch.Tensor) -> torch.Tensor:
        """Return a gradient-carrying, non-executable training weight vector."""

        if not isinstance(raw_target, torch.Tensor):
            raise TypeError("raw_target must be a torch.Tensor")
        if raw_target.ndim != 1 or raw_target.shape[0] != self.data.n_assets:
            raise ValueError("raw_target must have shape (n_assets,)")
        if not raw_target.is_floating_point() or raw_target.is_complex():
            raise ValueError("raw_target must be a real floating tensor")
        if not bool(torch.isfinite(raw_target).all().detach().cpu().item()):
            raise ValueError("raw_target must be finite")
        if self.data.n_assets == 0:
            return raw_target.new_empty((0,))
        if self.layer is None:  # pragma: no cover - construction invariant
            raise DecisionFocusedInactiveError("training projection is unavailable")
        try:
            (weights,) = self.layer(raw_target)
        except Exception as exc:
            raise DecisionFocusedInactiveError(
                "SCS training projection failed"
            ) from exc
        if not bool(torch.isfinite(weights).all().detach().cpu().item()):
            raise DecisionFocusedInactiveError(
                "SCS training projection emitted nonfinite weights"
            )

        # An equality with zero capacity is mathematical, not approximate.
        fixed_mask = torch.as_tensor(
            np.array(self.data.trade_capacity_weight == 0.0, copy=True),
            dtype=torch.bool,
            device=weights.device,
        )
        prior = _torch_copy(
            self.data.previous_weight,
            dtype=weights.dtype,
            device=weights.device,
        ).detach()
        verified = torch.where(fixed_mask, prior, weights)
        diagnostics = independent_constraint_diagnostics(
            self.data,
            verified.detach().cpu().numpy(),
        )
        if diagnostics.maximum_violation > TRAINING_PROJECTION_TOLERANCE:
            raise DecisionFocusedInactiveError(
                "SCS training projection failed independent feasibility verification: "
                f"{diagnostics.maximum_violation:.9g}"
            )
        return verified

    def forward(self, raw_target: torch.Tensor) -> torch.Tensor:
        """Alias required by ``nn.Module``; explicitly a training operation."""

        return self.project_for_training(raw_target)


def stopped_prior(weights: torch.Tensor) -> torch.Tensor:
    """Return a storage-independent prior holding with its gradient stopped."""

    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch.Tensor")
    return weights.detach().clone()


def torch_decision_total_cost(
    data: AllocationProblemData,
    weights: torch.Tensor,
    *,
    previous_weights: torch.Tensor | np.ndarray | None = None,
) -> TorchTotalCost:
    """Common monthly cost primitive retained for optimizer equivalence tests."""

    if not isinstance(data, AllocationProblemData):
        raise TypeError("data must be AllocationProblemData")
    prior = (
        data.previous_weight.copy() if previous_weights is None else previous_weights
    )
    return torch_total_cost(
        weights,
        prior,
        data.full_spread.copy(),
        data.adv20_usd.copy(),
        data.impact_volatility_daily.copy(),
        data.borrow_rate_annual.copy(),
        data.nav_dollars,
    )


@dataclass(frozen=True)
class DailyHoldingAccounting:
    """Differentiable within-month holdings, cash, costs, and terminal state."""

    net_excess_return: torch.Tensor
    net_total_return: torch.Tensor
    risk_free_total_return: torch.Tensor
    terminal_weights: torch.Tensor
    terminal_cash_weight: torch.Tensor
    daily_nav: torch.Tensor
    transaction_cost: torch.Tensor
    borrow_cost: torch.Tensor
    financing_cost: torch.Tensor


def torch_daily_holding_accounting(
    data: AllocationProblemData,
    weights: torch.Tensor,
    asset_total_returns: torch.Tensor | np.ndarray,
    daily_risk_free_rates: torch.Tensor | np.ndarray | float,
    *,
    previous_weights: torch.Tensor | np.ndarray | None = None,
) -> DailyHoldingAccounting:
    """Apply one rebalance cost and then drift signed holdings every day.

    Asset returns must be total returns, so dividends are already present.
    Short positions are represented by negative asset values and therefore use
    the same signed compounding as longs.  Borrow and financing accrue on each
    day's opening holdings at annual-rate/252; the risk-free rate accrues on
    cash daily.  The reported monthly excess return subtracts the compounded
    daily risk-free return from the resulting net portfolio total return.
    """

    if not isinstance(data, AllocationProblemData):
        raise TypeError("data must be AllocationProblemData")
    if not isinstance(weights, torch.Tensor) or weights.ndim != 1:
        raise TypeError("weights must be a one-dimensional torch.Tensor")
    if weights.shape[0] != data.n_assets or not weights.is_floating_point():
        raise ValueError("weights must be a real n_assets vector")
    if weights.is_complex() or not bool(
        torch.isfinite(weights).all().detach().cpu().item()
    ):
        raise ValueError("weights must be finite and real")

    returns = _torch_copy(
        asset_total_returns,
        dtype=weights.dtype,
        device=weights.device,
    )
    if returns.ndim == 1:
        returns = returns.unsqueeze(0)
    if returns.ndim != 2 or returns.shape[1] != data.n_assets:
        raise ValueError("asset_total_returns must have shape (days, n_assets)")
    if not bool(torch.isfinite(returns).all().detach().cpu().item()) or bool(
        (returns < -1.0).any().detach().cpu().item()
    ):
        raise ValueError("asset total returns must be finite and no less than -100%")
    days = returns.shape[0]
    risk_free = _torch_copy(
        daily_risk_free_rates,
        dtype=weights.dtype,
        device=weights.device,
    ).reshape(-1)
    if risk_free.numel() == 1 and days != 1:
        risk_free = risk_free.expand(days)
    if (
        risk_free.shape != (days,)
        or not bool(torch.isfinite(risk_free).all().detach().cpu().item())
        or bool((risk_free <= -1.0).any().detach().cpu().item())
    ):
        raise ValueError("daily_risk_free_rates must be finite with one value per day")

    prior_value = data.previous_weight if previous_weights is None else previous_weights
    prior = _torch_copy(prior_value, dtype=weights.dtype, device=weights.device)
    if prior.shape != weights.shape or not bool(
        torch.isfinite(prior).all().detach().cpu().item()
    ):
        raise ValueError("previous_weights must be a finite n_assets vector")
    prior = stopped_prior(prior)
    transaction = transaction_cost_torch(
        weights - prior,
        full_spread=data.full_spread.copy(),
        adv_dollars=data.adv20_usd.copy(),
        impact_volatility=data.impact_volatility_daily.copy(),
        nav_dollars=data.nav_dollars,
    )
    borrow_rates = _torch_copy(
        data.borrow_rate_annual,
        dtype=weights.dtype,
        device=weights.device,
    )
    financing_rate = weights.new_tensor(PRIMARY_FINANCING_ANNUAL_RATE)

    holdings = weights
    cash = weights.new_tensor(1.0) - weights.sum() - transaction
    nav = holdings.sum() + cash
    nav_path: list[torch.Tensor] = []
    borrow_total = weights.new_zeros(())
    financing_total = weights.new_zeros(())
    for index in range(days):
        borrow_charge = (
            torch.sum(torch.relu(-holdings) * borrow_rates) / TRADING_SESSIONS_PER_YEAR
        )
        financing_charge = (
            torch.relu(torch.sum(torch.abs(holdings)) - nav)
            * financing_rate
            / TRADING_SESSIONS_PER_YEAR
        )
        holdings = holdings * (1.0 + returns[index])
        cash = cash * (1.0 + risk_free[index]) - borrow_charge - financing_charge
        nav = holdings.sum() + cash
        if (
            not bool(torch.isfinite(nav).detach().cpu().item())
            or float(nav.detach().cpu().item()) <= 0.0
        ):
            raise DecisionFocusedInactiveError(
                "daily training accounting produced nonpositive or nonfinite NAV"
            )
        nav_path.append(nav)
        borrow_total = borrow_total + borrow_charge
        financing_total = financing_total + financing_charge

    risk_free_total = torch.prod(1.0 + risk_free) - 1.0
    net_total = nav - 1.0
    terminal_weights = holdings / nav
    terminal_cash = cash / nav
    daily_nav = torch.stack(nav_path) if nav_path else weights.new_empty((0,))
    return DailyHoldingAccounting(
        net_excess_return=net_total - risk_free_total,
        net_total_return=net_total,
        risk_free_total_return=risk_free_total,
        terminal_weights=terminal_weights,
        terminal_cash_weight=terminal_cash,
        daily_nav=daily_nav,
        transaction_cost=transaction,
        borrow_cost=borrow_total,
        financing_cost=financing_total,
    )


def torch_net_excess_return(
    data: AllocationProblemData,
    weights: torch.Tensor,
    asset_total_returns: torch.Tensor | np.ndarray,
    risk_free_rate: torch.Tensor | np.ndarray | float,
    *,
    previous_weights: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    """Return the daily-drifted net excess return used by training."""

    return torch_daily_holding_accounting(
        data,
        weights,
        asset_total_returns,
        risk_free_rate,
        previous_weights=previous_weights,
    ).net_excess_return


def negative_annualized_net_excess_sharpe(
    net_excess_returns: torch.Tensor | Iterable[float],
    *,
    observation_weights: torch.Tensor | Iterable[float] | None = None,
) -> torch.Tensor:
    """Frozen negative annualized Sharpe with population variance and ridge."""

    values = (
        net_excess_returns
        if isinstance(net_excess_returns, torch.Tensor)
        else torch.as_tensor(tuple(net_excess_returns), dtype=torch.float64)
    )
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("net_excess_returns must be a nonempty vector")
    if (
        not values.is_floating_point()
        or values.is_complex()
        or not bool(torch.isfinite(values).all().detach().cpu().item())
    ):
        raise ValueError("net_excess_returns must be finite real values")
    if observation_weights is None:
        weights = torch.ones_like(values) / values.numel()
    else:
        weights = _torch_copy(
            observation_weights,
            dtype=values.dtype,
            device=values.device,
        )
        if (
            weights.shape != values.shape
            or not bool(
                (torch.isfinite(weights) & (weights >= 0.0)).all().detach().cpu().item()
            )
            or float(weights.sum().detach().cpu().item()) <= 0.0
        ):
            raise ValueError(
                "observation_weights must be finite, nonnegative, and conformable"
            )
        weights = weights / weights.sum()
    mean = torch.sum(weights * values)
    population_variance = torch.sum(weights * torch.square(values - mean))
    return (
        -math.sqrt(ANNUALIZATION_PERIODS)
        * mean
        / torch.sqrt(population_variance + SHARPE_VARIANCE_RIDGE)
    )


def build_seeded_allocators(
    *,
    sector_vocabulary: Sequence[str],
    country_vocabulary: Sequence[str],
) -> tuple[DecisionFocusedAllocator, ...]:
    """Construct exactly the five preregistered seed-specific networks."""

    return tuple(
        DecisionFocusedAllocator(
            sector_vocabulary=sector_vocabulary,
            country_vocabulary=country_vocabulary,
            seed=seed,
        )
        for seed in DECISION_SEEDS
    )


@dataclass(frozen=True)
class SeedInferenceAudit:
    seed: int
    raw_weights: np.ndarray
    projected: AllocationResult

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_weights, dtype=np.float64).copy()
        if raw.ndim != 1 or not np.isfinite(raw).all():
            raise ValueError("raw inference weights must be a finite vector")
        raw.setflags(write=False)
        object.__setattr__(self, "raw_weights", raw)


@dataclass(frozen=True)
class DecisionInferenceResult:
    """Auditable CLARABEL-only inference result for all seeds and ensemble."""

    seed_results: tuple[SeedInferenceAudit, ...]
    averaged_seed_weights: np.ndarray
    final_projection: AllocationResult

    def __post_init__(self) -> None:
        average = np.asarray(self.averaged_seed_weights, dtype=np.float64).copy()
        if average.ndim != 1 or not np.isfinite(average).all():
            raise ValueError("averaged seed weights must be a finite vector")
        average.setflags(write=False)
        object.__setattr__(self, "averaged_seed_weights", average)

    @property
    def weights(self) -> np.ndarray:
        return self.final_projection.weights


def _same_problem(left: AllocationProblemData, right: AllocationProblemData) -> bool:
    return (
        left.n_assets == right.n_assets
        and np.array_equal(left.permno, right.permno)
        and np.array_equal(left.side, right.side)
        and np.array_equal(left.previous_weight, right.previous_weight)
    )


def project_seed_ensemble(
    models: Sequence[DecisionFocusedAllocator],
    data: AllocationProblemData,
    training_projection: DifferentiableProjection | None = None,
) -> DecisionInferenceResult:
    """Run deterministic executable inference with CLARABEL only.

    ``training_projection`` is accepted solely as a coherence assertion for
    older callers; it is never evaluated and no SCS weight enters inference.
    Every model's prior train/eval state is restored even if a solve fails.
    """

    if not isinstance(data, AllocationProblemData):
        raise TypeError("data must be AllocationProblemData")
    model_tuple = tuple(models)
    if tuple(model.seed for model in model_tuple) != DECISION_SEEDS:
        raise ValueError("models must appear once each in frozen seed order")
    sectors = model_tuple[0].sector_vocabulary
    countries = model_tuple[0].country_vocabulary
    if any(
        model.sector_vocabulary != sectors or model.country_vocabulary != countries
        for model in model_tuple
    ):
        raise ValueError("all seed models must share frozen vocabularies")
    if training_projection is not None and not _same_problem(
        training_projection.data, data
    ):
        raise ValueError("training projection and inference problem differ")
    # This also fails closed on a category outside the common vocabulary.
    for model in model_tuple:
        if model.input_width != 10 + len(sectors) + len(countries):
            raise ValueError("seed model input width differs from frozen vocabulary")
        model.encode(data)

    prior_modes = tuple(model.training for model in model_tuple)
    audits: list[SeedInferenceAudit] = []
    try:
        for model in model_tuple:
            model.eval()
        with torch.no_grad():
            for model in model_tuple:
                raw = model(data).detach().cpu().numpy().astype(np.float64, copy=True)
                projected = solve_projection(
                    data,
                    raw,
                    architecture="decision_focused_neural_projected",
                )
                audits.append(SeedInferenceAudit(model.seed, raw, projected))
        average = np.mean(
            np.stack([audit.projected.weights for audit in audits], axis=0),
            axis=0,
        )
        final = solve_projection(
            data,
            average,
            architecture="decision_focused_neural_projected",
        )
        return DecisionInferenceResult(tuple(audits), average, final)
    finally:
        for model, was_training in zip(model_tuple, prior_modes, strict=True):
            model.train(was_training)


def _month_end(value: object, *, name: str) -> date:
    if isinstance(value, datetime):
        result = value.date()
    elif isinstance(value, date):
        result = value
    elif isinstance(value, np.datetime64):
        text = np.datetime_as_string(value, unit="D")
        result = date.fromisoformat(text)
    elif isinstance(value, str):
        result = date.fromisoformat(value)
    else:
        raise TypeError(f"{name} must be a date or ISO date string")
    if result.day != calendar.monthrange(result.year, result.month)[1]:
        raise ValueError(f"{name} must be a calendar month-end")
    return result


def validate_quarterly_refit_date(value: object) -> date:
    """Accept only the frozen March/June/September/December month-end cadence."""

    result = _month_end(value, name="refit_date")
    if result.month not in QUARTERLY_REFIT_MONTHS:
        raise ValueError(
            "decision-focused updates are permitted only at quarterly month-ends"
        )
    return result


@dataclass(frozen=True)
class DecisionTrainingMonth:
    """One completed synthetic lifecycle month with a prebuilt problem.

    A real walk-forward caller cannot pre-freeze this row set because inherited
    names, signs, and reduction-only states depend on the preceding simulated
    portfolio.  Production use therefore also requires a point-in-time dynamic
    problem builder; the Stage 10 generic engine intentionally has no data
    access and the production runtime gate remains closed.
    """

    month_end: date | datetime | np.datetime64 | str
    problem: AllocationProblemData
    daily_total_returns: np.ndarray
    daily_risk_free_rates: np.ndarray
    completed: bool = True

    def __post_init__(self) -> None:
        month = _month_end(self.month_end, name="training month")
        if not isinstance(self.problem, AllocationProblemData):
            raise TypeError("problem must be AllocationProblemData")
        returns = np.asarray(self.daily_total_returns, dtype=np.float64).copy()
        risk_free = (
            np.asarray(self.daily_risk_free_rates, dtype=np.float64).reshape(-1).copy()
        )
        if returns.ndim != 2 or returns.shape[1] != self.problem.n_assets:
            raise ValueError("daily_total_returns must have shape (days, n_assets)")
        if returns.shape[0] == 0 or risk_free.shape != (returns.shape[0],):
            raise ValueError(
                "each completed month needs aligned nonempty daily returns and rf"
            )
        if not np.isfinite(returns).all() or np.any(returns < -1.0):
            raise ValueError("daily total returns must be finite and at least -100%")
        if not np.isfinite(risk_free).all() or np.any(risk_free <= -1.0):
            raise ValueError(
                "daily risk-free rates must be finite and greater than -100%"
            )
        if self.completed is not True:
            raise ValueError("training observations must be completed holding months")
        returns.setflags(write=False)
        risk_free.setflags(write=False)
        object.__setattr__(self, "month_end", month)
        object.__setattr__(self, "daily_total_returns", returns)
        object.__setattr__(self, "daily_risk_free_rates", risk_free)


def weighted_expanding_month_weights(
    months: Sequence[DecisionTrainingMonth],
    *,
    refit_date: object,
) -> np.ndarray:
    """Return normalized 10-year-half-life expanding-history month weights."""

    refit = validate_quarterly_refit_date(refit_date)
    observations = tuple(months)
    if not observations:
        raise ValueError("training history must not be empty")
    dates = tuple(item.month_end for item in observations)
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("training months must be strictly chronological")
    ages = np.asarray(
        [12 * (refit.year - item.year) + refit.month - item.month for item in dates],
        dtype=np.float64,
    )
    if np.any(ages < 0.0):
        raise ValueError("training months cannot follow the refit month")
    result = np.exp2(-ages / WEIGHT_HALF_LIFE_MONTHS)
    result /= result.sum()
    result.setflags(write=False)
    return result


class TrainingProjection(Protocol):
    def project_for_training(self, raw_target: torch.Tensor) -> torch.Tensor: ...


ProjectionFactory = Callable[[AllocationProblemData], TrainingProjection]


@dataclass(frozen=True)
class TrainingSequenceResult:
    net_excess_returns: torch.Tensor
    terminal_weights: tuple[torch.Tensor, ...]


def _prior_for_problem(
    problem: AllocationProblemData,
    prior_by_permno: dict[int, torch.Tensor],
) -> np.ndarray:
    current_ids = {int(value) for value in problem.permno.tolist()}
    missing = sorted(
        identifier
        for identifier, value in prior_by_permno.items()
        if identifier not in current_ids and abs(float(value.cpu().item())) > 1e-12
    )
    if missing:
        raise DecisionFocusedInactiveError(
            "a nonzero drifted holding disappeared without an accounted payoff: "
            f"{missing}"
        )
    return np.asarray(
        [
            (
                float(prior_by_permno[int(identifier)].cpu().item())
                if int(identifier) in prior_by_permno
                else 0.0
            )
            for identifier in problem.permno
        ],
        dtype=np.float64,
    )


def run_chronological_training_sequence(
    model: DecisionFocusedAllocator,
    months: Sequence[DecisionTrainingMonth],
    *,
    projection_factory: ProjectionFactory = DifferentiableProjection,
) -> TrainingSequenceResult:
    """Run the complete sequence from cash with stopped, drifted prior holdings."""

    observations = tuple(months)
    if not observations:
        raise ValueError("training sequence must not be empty")
    if any(
        left.month_end >= right.month_end
        for left, right in zip(observations, observations[1:])
    ):
        raise ValueError("training sequence must be strictly chronological")
    prior_by_permno: dict[int, torch.Tensor] = {}
    returns: list[torch.Tensor] = []
    terminals: list[torch.Tensor] = []
    for observation in observations:
        prior_numpy = _prior_for_problem(observation.problem, prior_by_permno)
        try:
            problem = replace(observation.problem, previous_weight=prior_numpy)
        except ValueError as exc:
            raise DecisionFocusedInactiveError(
                "chronological stopped prior is incoherent with the next problem"
            ) from exc
        projection = projection_factory(problem)
        raw = model(problem)
        projected = projection.project_for_training(raw)
        prior_tensor = _torch_copy(
            prior_numpy,
            dtype=projected.dtype,
            device=projected.device,
        ).detach()
        accounting = torch_daily_holding_accounting(
            problem,
            projected,
            observation.daily_total_returns,
            observation.daily_risk_free_rates,
            previous_weights=prior_tensor,
        )
        terminal = stopped_prior(accounting.terminal_weights)
        prior_by_permno = {
            int(identifier): terminal[index]
            for index, identifier in enumerate(problem.permno)
        }
        returns.append(accounting.net_excess_return)
        terminals.append(terminal)
    return TrainingSequenceResult(torch.stack(returns), tuple(terminals))


@dataclass(frozen=True)
class SeedTrainingResult:
    seed: int
    selected_epoch_count: int
    early_stopping_validation_losses: tuple[float, ...]
    refit_training_losses: tuple[float, ...]
    model: DecisionFocusedAllocator
    synthetic_test_mode: bool


def _training_controls(
    *,
    synthetic_test_mode: bool,
    maximum_epochs_for_test: int | None,
    patience_for_test: int | None,
    validation_months_for_test: int | None,
    projection_factory: ProjectionFactory | None,
) -> tuple[int, int, int, ProjectionFactory]:
    overrides = (
        maximum_epochs_for_test,
        patience_for_test,
        validation_months_for_test,
        projection_factory,
    )
    if any(value is not None for value in overrides) and not synthetic_test_mode:
        raise ValueError("training overrides require explicit synthetic_test_mode=True")
    maximum = (
        MAX_EPOCHS if maximum_epochs_for_test is None else int(maximum_epochs_for_test)
    )
    patience = PATIENCE if patience_for_test is None else int(patience_for_test)
    validation = (
        EARLY_STOPPING_MONTHS
        if validation_months_for_test is None
        else int(validation_months_for_test)
    )
    if maximum <= 0 or patience <= 0 or validation <= 0:
        raise ValueError("epoch, patience, and validation controls must be positive")
    factory = (
        DifferentiableProjection if projection_factory is None else projection_factory
    )
    return maximum, patience, validation, factory


def train_decision_focused_seed(
    months: Sequence[DecisionTrainingMonth],
    *,
    refit_date: object,
    sector_vocabulary: Sequence[str],
    country_vocabulary: Sequence[str],
    seed: int,
    synthetic_test_mode: bool = False,
    maximum_epochs_for_test: int | None = None,
    patience_for_test: int | None = None,
    validation_months_for_test: int | None = None,
    projection_factory: ProjectionFactory | None = None,
) -> SeedTrainingResult:
    """Early-stop on latest 24 months, then reset and refit for that epoch count."""

    refit = validate_quarterly_refit_date(refit_date)
    observations = tuple(months)
    maximum, patience, validation_count, factory = _training_controls(
        synthetic_test_mode=synthetic_test_mode,
        maximum_epochs_for_test=maximum_epochs_for_test,
        patience_for_test=patience_for_test,
        validation_months_for_test=validation_months_for_test,
        projection_factory=projection_factory,
    )
    if not synthetic_test_mode and not PRODUCTION_TRAINING_RUNTIME_CERTIFIED:
        raise DecisionFocusedInactiveError(
            "production decision-focused training is inactive: the stopped prior "
            "changes projection constraints each month and epoch, while safe "
            "prior-parameterized CVXPYLayer reuse and dynamic inherited-row problem "
            "rebuilding have not been independently certified"
        )
    if len(observations) <= validation_count:
        raise ValueError(
            "training history needs at least one fit month before validation"
        )
    memory_weights = weighted_expanding_month_weights(observations, refit_date=refit)
    split = len(observations) - validation_count

    def new_model() -> DecisionFocusedAllocator:
        return DecisionFocusedAllocator(
            sector_vocabulary=sector_vocabulary,
            country_vocabulary=country_vocabulary,
            seed=seed,
        ).double()

    validation_history: list[float] = []
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        selection_model = new_model()
        optimizer = torch.optim.Adam(selection_model.parameters(), lr=LEARNING_RATE)
        for epoch in range(1, maximum + 1):
            selection_model.train()
            optimizer.zero_grad(set_to_none=True)
            sequence = run_chronological_training_sequence(
                selection_model,
                observations,
                projection_factory=factory,
            )
            fit_loss = negative_annualized_net_excess_sharpe(
                sequence.net_excess_returns[:split],
                observation_weights=memory_weights[:split],
            )
            if not bool(torch.isfinite(fit_loss).detach().cpu().item()):
                raise DecisionFocusedInactiveError("nonfinite early-stopping fit loss")
            fit_loss.backward()
            optimizer.step()

            selection_model.eval()
            validation_sequence = run_chronological_training_sequence(
                selection_model,
                observations,
                projection_factory=factory,
            )
            validation_loss = negative_annualized_net_excess_sharpe(
                validation_sequence.net_excess_returns[split:],
                observation_weights=memory_weights[split:],
            )
            observed = float(validation_loss.detach().cpu().item())
            if not math.isfinite(observed):
                raise DecisionFocusedInactiveError("nonfinite validation loss")
            validation_history.append(observed)
            if observed < best_loss:
                best_loss = observed
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_epoch <= 0:  # pragma: no cover - finite validation invariant
            raise DecisionFocusedInactiveError("early stopping selected no epoch")

        # Same seed and architecture are reset; only the selected epoch count
        # crosses from the held-out stopping exercise into the full-history fit.
        torch.manual_seed(int(seed))
        refit_model = new_model()
        refit_optimizer = torch.optim.Adam(refit_model.parameters(), lr=LEARNING_RATE)
        refit_history: list[float] = []
        for _ in range(best_epoch):
            refit_model.train()
            refit_optimizer.zero_grad(set_to_none=True)
            sequence = run_chronological_training_sequence(
                refit_model,
                observations,
                projection_factory=factory,
            )
            loss = negative_annualized_net_excess_sharpe(
                sequence.net_excess_returns,
                observation_weights=memory_weights,
            )
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise DecisionFocusedInactiveError("nonfinite full-history refit loss")
            loss.backward()
            refit_optimizer.step()
            refit_history.append(float(loss.detach().cpu().item()))
        refit_model.eval()
    return SeedTrainingResult(
        seed=int(seed),
        selected_epoch_count=best_epoch,
        early_stopping_validation_losses=tuple(validation_history),
        refit_training_losses=tuple(refit_history),
        model=refit_model,
        synthetic_test_mode=synthetic_test_mode,
    )


def train_decision_focused_ensemble(
    months: Sequence[DecisionTrainingMonth],
    *,
    refit_date: object,
    sector_vocabulary: Sequence[str],
    country_vocabulary: Sequence[str],
) -> tuple[SeedTrainingResult, ...]:
    """Train exactly the five fixed seeds under production-frozen controls."""

    return tuple(
        train_decision_focused_seed(
            months,
            refit_date=refit_date,
            sector_vocabulary=sector_vocabulary,
            country_vocabulary=country_vocabulary,
            seed=seed,
        )
        for seed in DECISION_SEEDS
    )


__all__ = [
    "ANNUALIZATION_PERIODS",
    "DECISION_SEEDS",
    "DROPOUT",
    "DailyHoldingAccounting",
    "DecisionFocusedAllocator",
    "DecisionFocusedInactiveError",
    "DecisionInferenceResult",
    "DecisionRawAllocation",
    "DecisionTrainingMonth",
    "DifferentiableProjection",
    "EARLY_STOPPING_MONTHS",
    "HIDDEN_LAYERS",
    "LEARNING_RATE",
    "MAX_EPOCHS",
    "OUTPUTS_PER_SECURITY",
    "PATIENCE",
    "PRODUCTION_TRAINING_RUNTIME_CERTIFIED",
    "QUARTERLY_REFIT_MONTHS",
    "SHARPE_VARIANCE_RIDGE",
    "SeedInferenceAudit",
    "SeedTrainingResult",
    "TRAINING_PROJECTION_TOLERANCE",
    "TrainingSequenceResult",
    "WEIGHT_HALF_LIFE_MONTHS",
    "build_seeded_allocators",
    "decision_features",
    "negative_annualized_net_excess_sharpe",
    "project_seed_ensemble",
    "run_chronological_training_sequence",
    "stopped_prior",
    "torch_daily_holding_accounting",
    "torch_decision_total_cost",
    "torch_net_excess_return",
    "train_decision_focused_ensemble",
    "train_decision_focused_seed",
    "validate_quarterly_refit_date",
    "weighted_expanding_month_weights",
]
