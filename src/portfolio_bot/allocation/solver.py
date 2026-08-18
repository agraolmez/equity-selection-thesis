"""Deterministic convex solves and inherited-violation fallback for Stage 10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cvxpy as cp
import numpy as np

from portfolio_bot.allocation.constraints import (
    ConstraintDiagnostics,
    hard_trade_and_sign_constraints,
    independent_constraint_diagnostics,
    ordinary_constraints,
    slackable_constraint_specs,
)
from portfolio_bot.allocation.costs import cvxpy_total_cost, numpy_total_cost
from portfolio_bot.allocation.domain import AllocationProblemData
from portfolio_bot.numerical_materiality import (
    NumericalMaterialityError,
    SOLVER_FEASIBILITY_ATOL,
    audit_strategy_month_residue_dollars,
    audit_strategy_month_residue_weights,
    is_immaterial_weight,
)


PRIMARY_SOLVER_SETTINGS = {
    "max_iter": 500,
    "tol_gap_abs": 1e-9,
    "tol_gap_rel": 1e-9,
    "tol_feas": 1e-9,
    "equilibrate_enable": True,
    "iterative_refinement_enable": True,
}
RETRY_SOLVER_SETTINGS = {
    "max_iter": 2_000,
    "tol_gap_abs": 1e-11,
    "tol_gap_rel": 1e-11,
    "tol_feas": 1e-11,
    "equilibrate_enable": True,
    "iterative_refinement_enable": True,
}
INDEPENDENT_ABSOLUTE_TOLERANCE = SOLVER_FEASIBILITY_ATOL
EMERGENCY_WORSENING_TOLERANCE = SOLVER_FEASIBILITY_ATOL
LEXICOGRAPHIC_LOCK = 1e-9
PROJECTION_RIDGE = 1e-8
# Expected monthly returns and costs are small decimals.  Multiplication by a
# fixed positive constant changes neither the optimizer nor the economic
# meaning, but materially improves conic solver conditioning.  Result objects
# always report the original, unscaled return-unit objective.
ECONOMIC_OBJECTIVE_NUMERICAL_SCALE = 10_000.0
STATIC_COMPLIANCE_UNREACHABLE = -1

SolveKind = Literal["hybrid", "projection"]


class AllocationSolveError(RuntimeError):
    """Raised when the frozen solve hierarchy cannot certify a result."""


class _CertifiedOrdinaryInfeasible(RuntimeError):
    """Internal signal carrying the complete ordinary feasibility history."""

    def __init__(self, status_history: tuple[str, ...]) -> None:
        super().__init__("ordinary targets are certified infeasible")
        self.status_history = status_history


@dataclass(frozen=True)
class AllocationResult:
    architecture: str
    weights: np.ndarray
    residue_dollars: np.ndarray
    mode: Literal["ordinary", "emergency"]
    solver_status: str
    solver_status_history: tuple[str, ...]
    solver_attempt: Literal["primary", "retry", "emergency"]
    objective_value: float
    transaction_cost: float
    borrow_cost: float
    financing_cost: float
    total_cost: float
    diagnostics: ConstraintDiagnostics
    normalized_slacks: dict[str, float]
    binding_trade_limits: tuple[int, ...]
    # 0 means compliant, a positive integer is the minimum static-input
    # horizon, and -1 means unreachable without changing frozen inputs.
    estimated_months_to_compliance: int
    cash_weight: float

    def __post_init__(self) -> None:
        vector = np.asarray(self.weights, dtype=np.float64).copy()
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("allocation result weights must be a finite vector")
        vector.setflags(write=False)
        object.__setattr__(self, "weights", vector)
        residue = np.asarray(self.residue_dollars, dtype=np.float64).copy()
        if residue.shape != vector.shape or not np.isfinite(residue).all():
            raise ValueError(
                "allocation result residue_dollars must align with weights"
            )
        try:
            audit_strategy_month_residue_dollars(residue)
        except NumericalMaterialityError as exc:
            raise ValueError(
                "allocation result residue_dollars failed materiality audit"
            ) from exc
        residue.setflags(write=False)
        object.__setattr__(self, "residue_dollars", residue)
        history = tuple(str(value) for value in self.solver_status_history)
        if not history:
            raise ValueError("solver_status_history must not be empty")
        object.__setattr__(self, "solver_status_history", history)
        if not np.isfinite(float(self.cash_weight)):
            raise ValueError("cash_weight must be finite")
        months = self.estimated_months_to_compliance
        if isinstance(months, bool) or not isinstance(months, int) or months < -1:
            raise ValueError(
                "estimated_months_to_compliance must be -1, zero, or positive"
            )


def _projection_expression(
    data: AllocationProblemData,
    weights: cp.Variable,
    raw_weights: np.ndarray,
) -> cp.Expression:
    caps = np.where(data.long_mask, 0.0100, 0.0075)
    return 0.5 * cp.sum_squares(
        cp.multiply(1.0 / caps, weights - raw_weights)
    ) + PROJECTION_RIDGE * cp.sum_squares(weights)


def _objective_and_cost_constraints(
    data: AllocationProblemData,
    weights: cp.Variable,
    *,
    kind: SolveKind,
    raw_weights: np.ndarray | None,
) -> tuple[cp.Minimize | cp.Maximize, list[cp.Constraint]]:
    if kind == "hybrid":
        return _economic_objective_and_cost_constraints(data, weights)
    if raw_weights is None:
        raise ValueError("projection solve requires raw_weights")
    raw = np.asarray(raw_weights, dtype=np.float64)
    if raw.shape != (data.n_assets,) or not np.isfinite(raw).all():
        raise ValueError("raw_weights must be a finite n_assets vector")
    return cp.Minimize(_projection_expression(data, weights, raw)), []


def _economic_objective_and_cost_constraints(
    data: AllocationProblemData,
    weights: cp.Variable,
) -> tuple[cp.Maximize, list[cp.Constraint]]:
    """Return the frozen net-alpha-minus-cost objective at numerical scale."""

    breakdown = cvxpy_total_cost(
        weights,
        data.previous_weight,
        data.full_spread,
        data.adv20_usd,
        data.impact_volatility_daily,
        data.borrow_rate_annual,
        data.nav_dollars,
    )
    economic_value = data.expected_return @ weights - breakdown.total
    return (
        cp.Maximize(ECONOMIC_OBJECTIVE_NUMERICAL_SCALE * economic_value),
        list(breakdown.constraints),
    )


def _solve_status(problem: cp.Problem, settings: dict[str, object]) -> str:
    """Run CLARABEL once and preserve its exact status or failure class."""

    try:
        problem.solve(solver="CLARABEL", verbose=False, **settings)
    except cp.error.SolverError as exc:
        return f"solver_error:{type(exc).__name__}"
    except Exception as exc:  # pragma: no cover - defensive fail-closed guard
        return f"unexpected_error:{type(exc).__name__}"
    return str(problem.status) if problem.status is not None else "unknown"


def _numpy_costs(data: AllocationProblemData, weights: np.ndarray):
    return numpy_total_cost(
        weights,
        data.previous_weight,
        data.full_spread,
        data.adv20_usd,
        data.impact_volatility_daily,
        data.borrow_rate_annual,
        data.nav_dollars,
    )


def _canonicalize_solver_weights_with_residue(
    data: AllocationProblemData,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap solver residue and return its audited dollar evidence."""

    vector = np.asarray(weights, dtype=np.float64).copy()
    delta = vector - data.previous_weight
    fixed = data.trade_capacity_weight == 0.0
    if np.any(np.abs(delta[fixed]) > INDEPENDENT_ABSOLUTE_TOLERANCE):
        raise AllocationSolveError("zero-capacity trade equality was not solved")
    vector[fixed] = data.previous_weight[fixed]
    wrong_sign = data.side.astype(np.float64, copy=False) * vector < 0.0
    immaterial_solver_residue = (
        is_immaterial_weight(vector, data.nav_dollars)
        & (wrong_sign | data.required_exit)
        & ~fixed
    )
    near_zero = ((np.abs(vector) <= 1e-14) & ~fixed) | (
        immaterial_solver_residue
    )
    try:
        audit_strategy_month_residue_weights(
            vector[near_zero],
            data.nav_dollars,
        )
    except NumericalMaterialityError as exc:
        raise AllocationSolveError(
            "strategy-month numerical residue failed materiality audit"
        ) from exc
    residue_dollars = np.zeros_like(vector)
    residue_dollars[near_zero] = vector[near_zero] * data.nav_dollars
    vector[near_zero] = 0.0
    return vector, residue_dollars


def _canonicalize_solver_weights(
    data: AllocationProblemData,
    weights: np.ndarray,
) -> np.ndarray:
    """Preserve the historical vector-only canonicalization API."""

    vector, _residue_dollars = _canonicalize_solver_weights_with_residue(
        data,
        weights,
    )
    return vector


def _effective_trade_capacity(data: AllocationProblemData) -> np.ndarray:
    """Return the tighter of supplied and independently rebuilt ADV limits."""

    valid_adv = np.isfinite(data.adv20_usd) & (data.adv20_usd > 0.0)
    adv_capacity = np.where(
        valid_adv,
        0.10 * data.adv20_usd / float(data.nav_dollars),
        0.0,
    )
    return np.minimum(data.trade_capacity_weight, adv_capacity)


def _binding_trade_limits(
    data: AllocationProblemData,
    weights: np.ndarray,
) -> tuple[int, ...]:
    used = np.abs(weights - data.previous_weight)
    tolerance = 5e-8
    capacity = _effective_trade_capacity(data)
    indices = np.flatnonzero(used >= np.maximum(capacity - tolerance, 0.0))
    # A zero-capacity equality is itself a binding operational limit and must
    # be disclosed even though the corresponding executed trade is zero.
    return tuple(
        int(data.permno[index])
        for index in indices
        if used[index] > tolerance or capacity[index] == 0.0
    )


def _static_compliance_constraints(
    data: AllocationProblemData,
    weights: cp.Variable,
    start: np.ndarray,
    months: int,
) -> list[cp.Constraint]:
    """Build a cumulative-capacity static-input compliance feasibility set.

    The convention freezes securities, sides, classifications, betas, risk,
    NAV, and per-month capacities at the current rebalance.  Capacity and the
    one-way turnover budget accumulate linearly for ``months``; all prospective
    portfolio targets must be met at the horizon.  Reduction-only positions
    may only move toward zero.  It is an operational estimate, not a forecast.
    """

    delta = weights - start
    constraints: list[cp.Constraint] = []
    long_indices = np.flatnonzero(data.long_mask)
    short_indices = np.flatnonzero(data.short_mask)
    if long_indices.size:
        constraints.append(weights[long_indices] >= 0.0)
    if short_indices.size:
        constraints.append(weights[short_indices] <= 0.0)
    for index in np.flatnonzero(data.reduction_only):
        if data.side[index] > 0:
            constraints.append(weights[index] <= start[index])
        else:
            constraints.append(weights[index] >= start[index])
    constraints.extend(
        [
            cp.abs(delta) <= float(months) * _effective_trade_capacity(data),
            0.5 * cp.norm1(delta) <= float(months) * 0.50,
        ]
    )
    constraints.extend(
        spec.expression <= spec.limit
        for spec in slackable_constraint_specs(data, weights)
    )
    return constraints


def _static_compliance_feasible(
    data: AllocationProblemData,
    start: np.ndarray,
    months: int,
) -> bool:
    weights = cp.Variable(data.n_assets, name=f"compliance_weights_{months}")
    problem = cp.Problem(
        cp.Minimize(0.0),
        _static_compliance_constraints(data, weights, start, months),
    )
    return _solve_status(problem, RETRY_SOLVER_SETTINGS) == cp.OPTIMAL


def _estimated_months_to_compliance(
    data: AllocationProblemData,
    weights: np.ndarray,
    diagnostics: ConstraintDiagnostics,
) -> int:
    if diagnostics.maximum_normalized_violation <= INDEPENDENT_ABSOLUTE_TOLERANCE:
        return 0
    if data.n_assets == 0:
        return 0

    capacity = _effective_trade_capacity(data)
    movable = capacity > 0.0
    # Zeroing every movable holding and allowing every empty row to reach its
    # applicable cap gives a finite conservative search horizon.  If even that
    # horizon cannot meet all targets, -1 means unreachable under frozen rows
    # and capacities (rather than emitting an ambiguous null).
    applicable_cap = np.where(data.long_mask, 0.0100, 0.0075)
    start = np.asarray(weights, dtype=np.float64)
    required_movement = np.maximum(np.abs(start), applicable_cap)
    ratios = np.divide(
        required_movement,
        capacity,
        out=np.zeros_like(required_movement),
        where=movable,
    )
    turnover_upper = int(np.ceil(float(np.sum(np.abs(start)) + np.sum(applicable_cap))))
    upper = max(
        1,
        turnover_upper,
        int(np.ceil(float(np.max(ratios, initial=0.0)))),
    )
    if not _static_compliance_feasible(data, start, upper):
        return STATIC_COMPLIANCE_UNREACHABLE
    low = 1
    while low < upper:
        middle = (low + upper) // 2
        if _static_compliance_feasible(data, start, middle):
            upper = middle
        else:
            low = middle + 1
    return low


def _make_result(
    data: AllocationProblemData,
    weights: np.ndarray,
    *,
    architecture: str,
    kind: SolveKind,
    raw_weights: np.ndarray | None,
    mode: Literal["ordinary", "emergency"],
    solver_attempt: Literal["primary", "retry", "emergency"],
    solver_status: str,
    solver_status_history: tuple[str, ...],
    normalized_slacks: dict[str, float] | None = None,
    residue_dollars: np.ndarray | None = None,
) -> AllocationResult:
    vector = np.asarray(weights, dtype=np.float64)
    residue = (
        np.zeros_like(vector)
        if residue_dollars is None
        else np.asarray(residue_dollars, dtype=np.float64)
    )
    diagnostics = independent_constraint_diagnostics(data, vector)
    costs = _numpy_costs(data, vector)
    if kind == "hybrid" or mode == "emergency":
        objective_value = float(data.expected_return @ vector - costs.total)
    else:
        assert raw_weights is not None
        caps = np.where(data.long_mask, 0.0100, 0.0075)
        objective_value = float(
            0.5 * np.sum(((vector - raw_weights) / caps) ** 2)
            + PROJECTION_RIDGE * np.dot(vector, vector)
        )
    return AllocationResult(
        architecture=architecture,
        weights=vector,
        residue_dollars=residue,
        mode=mode,
        solver_status=solver_status,
        solver_status_history=solver_status_history,
        solver_attempt=solver_attempt,
        objective_value=objective_value,
        transaction_cost=float(costs.transaction),
        borrow_cost=float(costs.borrow),
        financing_cost=float(costs.financing),
        total_cost=float(costs.total),
        diagnostics=diagnostics,
        normalized_slacks=dict(normalized_slacks or {}),
        binding_trade_limits=_binding_trade_limits(data, vector),
        estimated_months_to_compliance=_estimated_months_to_compliance(
            data,
            vector,
            diagnostics,
        ),
        cash_weight=float(1.0 - diagnostics.net),
    )


def _ordinary_solve(
    data: AllocationProblemData,
    *,
    kind: SolveKind,
    raw_weights: np.ndarray | None,
    architecture: str,
) -> AllocationResult:
    weights = cp.Variable(data.n_assets, name="weights")
    objective, cost_constraints = _objective_and_cost_constraints(
        data,
        weights,
        kind=kind,
        raw_weights=raw_weights,
    )
    constraints = ordinary_constraints(data, weights) + cost_constraints
    problem = cp.Problem(objective, constraints)
    history: list[str] = []
    for label, settings in (
        ("primary", PRIMARY_SOLVER_SETTINGS),
        ("retry", RETRY_SOLVER_SETTINGS),
    ):
        status = _solve_status(problem, settings)
        history.append(f"ordinary_{label}:{status}")
        if status != cp.OPTIMAL:
            continue
        if weights.value is None:
            history.append(f"ordinary_{label}_audit:missing_primal")
            continue
        try:
            vector, residue_dollars = _canonicalize_solver_weights_with_residue(
                data,
                np.asarray(weights.value, dtype=np.float64).reshape(-1),
            )
        except (AllocationSolveError, TypeError, ValueError):
            history.append(f"ordinary_{label}_audit:invalid_primal")
            continue
        diagnostics = independent_constraint_diagnostics(data, vector)
        if diagnostics.maximum_violation > INDEPENDENT_ABSOLUTE_TOLERANCE:
            history.append(f"ordinary_{label}_audit:constraint_violation")
            continue
        return _make_result(
            data,
            vector,
            architecture=architecture,
            kind=kind,
            raw_weights=raw_weights,
            mode="ordinary",
            solver_attempt=label,
            solver_status=status,
            solver_status_history=tuple(history),
            residue_dollars=residue_dollars,
        )

    # Emergency is not a generic response to an objective-solve numerical
    # failure.  After both objective attempts fail to produce an independently
    # valid exact optimum, a separate zero-objective solve over the identical
    # constraints decides feasibility.  Only its exact infeasible certificate
    # permits the emergency hierarchy; every other status fails closed.
    feasibility = cp.Problem(cp.Minimize(0.0), constraints)
    feasibility_status = _solve_status(feasibility, RETRY_SOLVER_SETTINGS)
    history.append(f"ordinary_feasibility_retry:{feasibility_status}")
    if feasibility_status != cp.INFEASIBLE:
        raise AllocationSolveError(
            "ordinary feasibility was not exactly certified infeasible: "
            + ", ".join(history)
        )
    raise _CertifiedOrdinaryInfeasible(tuple(history))


def _emergency_solve(
    data: AllocationProblemData,
    *,
    kind: SolveKind,
    raw_weights: np.ndarray | None,
    architecture: str,
    ordinary_status_history: tuple[str, ...],
) -> AllocationResult:
    pretrade_diagnostics = independent_constraint_diagnostics(
        data,
        data.previous_weight,
    )
    always_hard_prefixes = (
        "sign:",
        "trade_capacity:",
        "adv_capacity:",
        "reduction_only:",
    )
    for name, value in pretrade_diagnostics.violations.items():
        if name == "turnover" or name.startswith(always_hard_prefixes):
            if value > INDEPENDENT_ABSOLUTE_TOLERANCE:
                raise AllocationSolveError(
                    f"pretrade violates an always-hard emergency rule: {name}"
                )

    weights = cp.Variable(data.n_assets, name="emergency_weights")
    # All three architectures share the methodology's economic third step in
    # emergency mode.  Projection distance is deliberately not a fallback
    # objective, and the same transaction/borrow/financing terms remain live.
    objective, cost_constraints = _economic_objective_and_cost_constraints(
        data, weights
    )
    constraints = hard_trade_and_sign_constraints(data, weights) + cost_constraints
    specs = slackable_constraint_specs(data, weights)
    slack_variables: dict[str, cp.Variable] = {}
    pretrade_slacks: dict[str, float] = {}
    for spec in specs:
        pre = spec.pretrade_normalized_violation
        if pre <= 0.0:
            constraints.append(spec.expression <= spec.limit)
            continue
        slack = cp.Variable(nonneg=True, name=f"slack_{len(slack_variables)}")
        constraints.extend(
            [
                spec.expression <= spec.limit + spec.normalizer * slack,
                slack <= pre,
            ]
        )
        slack_variables[spec.name] = slack
        pretrade_slacks[spec.name] = pre
    if not slack_variables:
        raise AllocationSolveError(
            "ordinary problem failed but no inherited target violation permits slack"
        )

    slack_vector = cp.hstack(list(slack_variables.values()))
    maximum = cp.Variable(nonneg=True, name="maximum_normalized_slack")
    step1_constraints = constraints + [slack_vector <= maximum]
    step1 = cp.Problem(cp.Minimize(maximum), step1_constraints)
    history = list(ordinary_status_history)
    step1_status = _solve_status(step1, RETRY_SOLVER_SETTINGS)
    history.append(f"emergency_minimax:{step1_status}")
    if step1_status != cp.OPTIMAL:
        raise AllocationSolveError(
            "emergency maximum-slack step failed: " + ", ".join(history)
        )
    optimum_maximum = float(maximum.value)

    step2_constraints = step1_constraints + [
        maximum <= optimum_maximum + LEXICOGRAPHIC_LOCK
    ]
    slack_sum = cp.sum(slack_vector)
    step2 = cp.Problem(cp.Minimize(slack_sum), step2_constraints)
    step2_status = _solve_status(step2, RETRY_SOLVER_SETTINGS)
    history.append(f"emergency_sum:{step2_status}")
    if step2_status != cp.OPTIMAL:
        raise AllocationSolveError(
            "emergency sum-slack step failed: " + ", ".join(history)
        )
    optimum_sum = float(slack_sum.value)

    final_constraints = step2_constraints + [
        slack_sum <= optimum_sum + LEXICOGRAPHIC_LOCK
    ]
    final_problem = cp.Problem(objective, final_constraints)
    final_status = _solve_status(final_problem, PRIMARY_SOLVER_SETTINGS)
    history.append(f"emergency_economic_primary:{final_status}")
    if final_status != cp.OPTIMAL:
        final_status = _solve_status(final_problem, RETRY_SOLVER_SETTINGS)
        history.append(f"emergency_economic_retry:{final_status}")
    if final_status != cp.OPTIMAL:
        raise AllocationSolveError(
            "emergency economic-objective step failed: " + ", ".join(history)
        )
    vector, residue_dollars = _canonicalize_solver_weights_with_residue(
        data,
        np.asarray(weights.value, dtype=np.float64).reshape(-1),
    )
    diagnostics = independent_constraint_diagnostics(data, vector)
    for name, value in diagnostics.violations.items():
        if name.startswith(always_hard_prefixes) or name == "turnover":
            if value > INDEPENDENT_ABSOLUTE_TOLERANCE:
                raise AllocationSolveError(f"emergency hard constraint failed: {name}")
    normalized_slacks = {
        name: max(diagnostics.normalized_violations.get(name, 0.0), 0.0)
        for name in slack_variables
    }
    for name, observed in normalized_slacks.items():
        prior = max(pretrade_diagnostics.normalized_violations.get(name, 0.0), 0.0)
        if observed > prior + EMERGENCY_WORSENING_TOLERANCE:
            raise AllocationSolveError(f"emergency violation worsened: {name}")
        if observed > pretrade_slacks[name] + EMERGENCY_WORSENING_TOLERANCE:
            raise AllocationSolveError(
                f"emergency slack exceeded inherited bound: {name}"
            )
    for name, value in diagnostics.violations.items():
        if (
            name not in slack_variables
            and name != "turnover"
            and not name.startswith(always_hard_prefixes)
            and value > INDEPENDENT_ABSOLUTE_TOLERANCE
        ):
            raise AllocationSolveError(
                f"emergency created a non-inherited target violation: {name}"
            )
    return _make_result(
        data,
        vector,
        architecture=architecture,
        kind=kind,
        raw_weights=raw_weights,
        mode="emergency",
        solver_attempt="emergency",
        solver_status=final_status,
        solver_status_history=tuple(history),
        normalized_slacks=normalized_slacks,
        residue_dollars=residue_dollars,
    )


def solve_hybrid(data: AllocationProblemData) -> AllocationResult:
    """Solve the confirmatory net-alpha primary under the common constraints."""

    architecture = "hybrid_alpha_risk_cost_convex"
    if data.n_assets == 0:
        return _make_result(
            data,
            np.empty(0, dtype=np.float64),
            architecture=architecture,
            kind="hybrid",
            raw_weights=None,
            mode="ordinary",
            solver_attempt="primary",
            solver_status=cp.OPTIMAL,
            solver_status_history=("all_cash_no_assets:optimal",),
        )
    try:
        return _ordinary_solve(
            data,
            kind="hybrid",
            raw_weights=None,
            architecture=architecture,
        )
    except _CertifiedOrdinaryInfeasible as certified:
        return _emergency_solve(
            data,
            kind="hybrid",
            raw_weights=None,
            architecture=architecture,
            ordinary_status_history=certified.status_history,
        )


def solve_projection(
    data: AllocationProblemData,
    raw_weights: object,
    *,
    architecture: str,
) -> AllocationResult:
    """Solve the unique cap-scaled projection shared by both challengers."""

    raw = np.asarray(raw_weights, dtype=np.float64)
    if raw.shape != (data.n_assets,) or not np.isfinite(raw).all():
        raise ValueError("raw_weights must be a finite n_assets vector")
    if data.n_assets == 0:
        return _make_result(
            data,
            raw,
            architecture=architecture,
            kind="projection",
            raw_weights=raw,
            mode="ordinary",
            solver_attempt="primary",
            solver_status=cp.OPTIMAL,
            solver_status_history=("all_cash_no_assets:optimal",),
        )
    try:
        return _ordinary_solve(
            data,
            kind="projection",
            raw_weights=raw,
            architecture=architecture,
        )
    except _CertifiedOrdinaryInfeasible as certified:
        return _emergency_solve(
            data,
            kind="projection",
            raw_weights=raw,
            architecture=architecture,
            ordinary_status_history=certified.status_history,
        )


__all__ = [
    "AllocationResult",
    "AllocationSolveError",
    "ECONOMIC_OBJECTIVE_NUMERICAL_SCALE",
    "EMERGENCY_WORSENING_TOLERANCE",
    "INDEPENDENT_ABSOLUTE_TOLERANCE",
    "LEXICOGRAPHIC_LOCK",
    "PRIMARY_SOLVER_SETTINGS",
    "PROJECTION_RIDGE",
    "RETRY_SOLVER_SETTINGS",
    "STATIC_COMPLIANCE_UNREACHABLE",
    "solve_hybrid",
    "solve_projection",
]
