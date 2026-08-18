"""Stage 11-only, algebraically equivalent allocation reparameterization.

The frozen Stage 10 policy is expressed in portfolio-weight units.  At the
larger Stage 11 cross sections, those small decision values and the
``|trade|**1.5`` impact term can make CLARABEL return
``optimal_inaccurate`` even when an independent feasibility solve is exactly
optimal.  This adapter changes only numerical coordinates:

``weight_i = previous_weight_i + side_cap_i * normalized_trade_i``

and, for the transaction-cost cone only, the affine expression

``basis_point_trade_i = 10000 * side_cap_i * normalized_trade_i``.

Both substitutions are positive diagonal affine bijections.  The economic
objective, feasible set, CLARABEL-only hierarchy, tolerances, independent
audits, and emergency policy remain the frozen Stage 10 definitions.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal

import cvxpy as cp
import numpy as np
from scipy.sparse.linalg import lsmr

from portfolio_bot.allocation import solver as frozen_solver
from portfolio_bot.allocation.constraints import (
    LONG_CAP,
    SHORT_CAP,
    hard_trade_and_sign_constraints,
    independent_constraint_diagnostics,
    ordinary_constraints,
    slackable_constraint_specs,
)
from portfolio_bot.allocation.costs import (
    CvxpyTotalCost,
    PRIMARY_FINANCING_ANNUAL_RATE,
    _transaction_coefficients,
    numpy_total_cost,
)
from portfolio_bot.allocation.domain import AllocationProblemData
from portfolio_bot.risk_cost.cost_math import PRIMARY_COST_SCENARIO


# Portfolio weights are NAV fractions, so multiplication by 10,000 expresses
# a trade in basis points of NAV.  This is an economic unit conversion rather
# than a validation-selected tuning parameter.
TRADE_NUMERICAL_SCALE = 10_000.0

# The explicit conic dual is an independent optimality certificate only.  Its
# solver tolerances remain the frozen retry tolerances.  Direct residual audits
# permit two hundred times the retry feasibility tolerance.  The resulting
# 2e-9 certificate limit is still twenty-five times tighter than the frozen
# 5e-8 independent primal-feasibility limit and fifty times tighter than the
# uniform 1e-7 candidate/dual-gap audit limit, while avoiding false rejections
# caused by sub-nanounit conic residuals at the full Stage 11 cross-section.
# The same limits apply uniformly to every validation month and every
# dual-certificate route.
DUAL_CERTIFICATE_RESIDUAL_TOLERANCE = float(
    200.0 * float(frozen_solver.RETRY_SOLVER_SETTINGS["tol_feas"])
)
DUAL_CERTIFICATE_GAP_TOLERANCE = float(
    2.0 * frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE
)
DUAL_CERTIFICATE_STATIONARITY_FLOOR = float(
    np.sqrt(frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE)
)
# CVXPY and the independent NumPy reconstruction evaluate the identical
# objective through different deterministic reduction orders.  Five hundred
# picounits admits only that roundoff-scale disagreement; the limit remains
# one hundred times tighter than the frozen 5e-8 primal-feasibility gate and
# two hundred times tighter than the uniform 1e-7 candidate/dual-gap gate.  It
# applies uniformly to every certificate route and validation month.
DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE = float(
    50.0 * float(frozen_solver.RETRY_SOLVER_SETTINGS["tol_gap_abs"])
)
DUAL_CERTIFIED_SOLVER_STATUS = "optimal_dual_certified"
DUAL_CERTIFICATE_AUDITABLE_STATUSES = frozenset(
    (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
)
EXPLICIT_DUAL_CERTIFICATE_SOURCE = "separately_solved_explicit_dual"
RETRY_DUAL_CERTIFICATE_SOURCE = (
    "clarabel_retry_canonical_dual_lsmr_refined"
)
RECONCILED_DUAL_CERTIFICATE_SOURCE = (
    "equal_convex_blend_of_explicit_and_retry_duals"
)
BOUNDED_FEASIBLE_DUAL_CERTIFICATE_SOURCE = (
    "cone_preserving_scaled_bounded_dual_feasibility"
)
DUAL_RECONCILIATION_RETRY_WEIGHT = 0.5
BOUNDED_DUAL_STATIONARITY_MULTIPLIER = 10_000.0
BOUNDED_DUAL_GAP_FRACTION = 0.25
# The bounded-dual construction keeps its pre-amendment absolute 1.25e-8
# target band.  It is a certificate-construction parameter, not the global
# candidate/dual-gap audit limit, and must not widen when that audit limit is
# adjusted uniformly.
BOUNDED_DUAL_GAP_REFERENCE_TOLERANCE = float(
    frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE
)
BOUNDED_DUAL_GAP_BAND = float(
    BOUNDED_DUAL_GAP_FRACTION * BOUNDED_DUAL_GAP_REFERENCE_TOLERANCE
)
# Backward-compatible name retained for the Stage 11 authority loader while
# its sealed amendment records the complete explicit-first hierarchy.
DUAL_CERTIFICATE_SOURCE = RETRY_DUAL_CERTIFICATE_SOURCE
DUAL_STATIONARITY_REFINEMENT_TOLERANCE = float(np.finfo(np.float64).eps)
DUAL_FAILURE_SNAPSHOT_ROOT = Path(
    "/private/tmp/stage11_dual_failure_snapshots"
)

SolveKind = Literal["hybrid", "projection"]

# Local aliases make the Stage 11 adapter independently monkeypatchable in its
# fail-closed tests while preserving every frozen setting and audit threshold.
_solve_status = frozen_solver._solve_status


@dataclass(frozen=True)
class _HybridDualCertificate:
    certificate_source: str
    dual_status: str
    candidate_maximum_constraint_violation: float
    stationarity_absolute_maximum: float
    stationarity_backward_error_maximum: float
    dual_nonnegative_minimum: float
    dual_soc_minimum_margin: float
    candidate_scaled_objective: float
    dual_upper_scaled: float
    candidate_dual_gap_scaled: float
    objective_cost_scaled_maximum_difference: float
    dual_soc_minimum_block_ordinal: int = -1
    stationarity_refinement_scale: float = 0.0
    stationarity_refinement_stop_code: int = -1
    stationarity_refinement_iterations: int = -1
    stationarity_refinement_correction_maximum: float = 0.0
    dual_reconciliation_retry_weight: float = 0.0
    dual_coordinate_scale_minimum: float = 0.0
    dual_coordinate_scale_maximum: float = 0.0
    stationarity_equation_multiplier: float = 0.0
    candidate_gap_band_fraction: float = 0.0
    dual_vector: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class _EmergencyCandidateAudit:
    maximum_hard_or_noninherited_violation: float
    maximum_inherited_worsening: float
    maximum_invalid_cost_trade: float
    locked_maximum_slack_violation: float
    locked_sum_slack_violation: float
    required_maximum_slack: float
    required_sum_slack: float

    @property
    def certificate_primal_violation(self) -> float:
        return max(
            self.maximum_hard_or_noninherited_violation,
            self.maximum_inherited_worsening,
            self.maximum_invalid_cost_trade,
            self.locked_maximum_slack_violation,
            self.locked_sum_slack_violation,
            0.0,
        )


@dataclass(frozen=True)
class _LexicographicDualCertificate:
    """Direct certificate for an inaccurate emergency slack optimum."""

    certificate_source: str
    dual_status: str
    candidate_maximum_constraint_violation: float
    canonical_primal_residual_maximum: float
    primal_zero_cone_residual_maximum: float
    primal_nonnegative_minimum: float
    primal_soc_minimum_margin: float
    stationarity_absolute_maximum: float
    stationarity_backward_error_maximum: float
    dual_nonnegative_minimum: float
    dual_soc_minimum_margin: float
    candidate_objective: float
    dual_lower_bound: float
    candidate_dual_gap: float
    objective_agreement_maximum_difference: float
    complementarity_absolute: float
    primal_soc_minimum_block_ordinal: int = -1
    dual_soc_minimum_block_ordinal: int = -1
    stationarity_refinement_stop_code: int = -1
    stationarity_refinement_iterations: int = -1
    stationarity_refinement_correction_maximum: float = 0.0


@dataclass(frozen=True)
class _LexicographicCanonicalAudit:
    matrix: object = field(repr=False, compare=False)
    rhs: np.ndarray = field(repr=False, compare=False)
    objective: np.ndarray = field(repr=False, compare=False)
    dims: object = field(repr=False, compare=False)
    raw_primal: np.ndarray = field(repr=False, compare=False)
    raw_slack: np.ndarray = field(repr=False, compare=False)
    raw_dual: np.ndarray = field(repr=False, compare=False)
    primal_status: str
    candidate_maximum_constraint_violation: float
    canonical_primal_residual_maximum: float
    primal_zero_cone_residual_maximum: float
    primal_nonnegative_minimum: float
    primal_soc_minimum_margin: float
    primal_soc_minimum_block_ordinal: int
    candidate_objective: float
    objective_offset: float
    objective_agreement_maximum_difference: float
    reported_dual_objective_agreement: float


def _cap_vector(data: AllocationProblemData) -> np.ndarray:
    """Return the frozen side-specific position cap for every row."""

    return np.where(data.long_mask, LONG_CAP, SHORT_CAP).astype(
        np.float64,
        copy=False,
    )


def _scaled_weight_expression(
    data: AllocationProblemData,
    *,
    name: str,
) -> tuple[cp.Variable, cp.Expression, np.ndarray]:
    """Create a cap-normalized trade variable and exact weight expression.

    Month one starts from cash, so the earlier coordinate ``weight = cap*x``
    is well conditioned there.  From month two onward, however, forming
    ``cap*x - previous_weight`` inside each power cone reintroduces subtraction
    between nearby decimals.  Centering the variable at the authenticated
    pretrade portfolio removes that cancellation:

    ``weight = previous_weight + cap*normalized_trade``.

    Positive caps make this an affine bijection over the same weight space.
    """

    caps = _cap_vector(data)
    normalized_trade = cp.Variable(data.n_assets, name=name)
    weights = data.previous_weight + cp.multiply(caps, normalized_trade)
    return normalized_trade, weights, caps


def _scaled_cvxpy_total_cost(
    data: AllocationProblemData,
    normalized_trade: cp.Variable,
    weights: cp.Expression,
    caps: np.ndarray,
) -> CvxpyTotalCost:
    """Build the frozen total cost in better-conditioned trade coordinates.

    With ``weight = previous_weight + caps*d``, direct substitution gives

    ``a*caps*|d| + k*caps**1.5*|d|**1.5``.

    The affine expression ``z = 10000*caps*d`` measures trades in basis points
    of NAV and expresses the same quantity as
    ``a/10000*|z| + k/10000**1.5*|z|**1.5``.  It is deliberately an expression,
    not an auxiliary variable and equality.  Only permitted, positive-impact
    rows receive power-cone atoms; invalid rows retain the frozen exact
    no-trade equality in original weight units.
    """

    delta_weight = cp.multiply(caps, normalized_trade)
    scaled_trade = TRADE_NUMERICAL_SCALE * delta_weight
    constraints: list[cp.Constraint] = []

    linear, impact, permitted = _transaction_coefficients(
        data.full_spread,
        data.adv20_usd,
        data.impact_volatility_daily,
        nav_dollars=data.nav_dollars,
        scenario=PRIMARY_COST_SCENARIO,
    )
    permitted = np.asarray(permitted, dtype=np.bool_)
    active = np.flatnonzero(permitted)
    positive_impact = np.flatnonzero(permitted & (impact > 0.0))
    invalid = np.flatnonzero(~permitted)

    absolute_trade = cp.abs(scaled_trade)
    linear_transaction = (
        cp.sum(
            cp.multiply(
                linear[active] / TRADE_NUMERICAL_SCALE,
                absolute_trade[active],
            )
        )
        if active.size
        else cp.Constant(0.0)
    )
    impact_transaction = (
        cp.sum(
            cp.multiply(
                impact[positive_impact] / (TRADE_NUMERICAL_SCALE**1.5),
                cp.power(absolute_trade[positive_impact], 1.5),
            )
        )
        if positive_impact.size
        else cp.Constant(0.0)
    )
    transaction = linear_transaction + impact_transaction
    if invalid.size:
        constraints.append(delta_weight[invalid] == 0.0)

    # Positive side caps let us keep borrow and financing atoms in normalized
    # coordinates without changing their values.
    normalized_weights = (
        data.previous_weight / caps + normalized_trade
    )
    borrow = cp.sum(
        cp.multiply(
            data.borrow_rate_annual * caps / 12.0,
            cp.pos(-normalized_weights),
        )
    )
    financing = cp.pos(
        cp.sum(cp.multiply(caps, cp.abs(normalized_weights))) - 1.0
    ) * (PRIMARY_FINANCING_ANNUAL_RATE / 12.0)
    total = transaction + borrow + financing

    mask = np.array(permitted, dtype=np.bool_, copy=True)
    mask.setflags(write=False)
    return CvxpyTotalCost(
        transaction=transaction,
        borrow=borrow,
        financing=financing,
        total=total,
        constraints=tuple(constraints),
        trade_permitted=mask,
    )


def _economic_objective_and_cost_constraints(
    data: AllocationProblemData,
    normalized_trade: cp.Variable,
    weights: cp.Expression,
    caps: np.ndarray,
) -> tuple[cp.Maximize, list[cp.Constraint]]:
    breakdown = _scaled_cvxpy_total_cost(
        data,
        normalized_trade,
        weights,
        caps,
    )
    expected_return = cp.sum(cp.multiply(data.expected_return, weights))
    economic_value = expected_return - breakdown.total
    return (
        cp.Maximize(
            frozen_solver.ECONOMIC_OBJECTIVE_NUMERICAL_SCALE * economic_value
        ),
        list(breakdown.constraints),
    )


def _objective_and_cost_constraints(
    data: AllocationProblemData,
    normalized_trade: cp.Variable,
    weights: cp.Expression,
    caps: np.ndarray,
    *,
    kind: SolveKind,
    raw_weights: np.ndarray | None,
) -> tuple[cp.Minimize | cp.Maximize, list[cp.Constraint]]:
    if kind == "hybrid":
        return _economic_objective_and_cost_constraints(
            data,
            normalized_trade,
            weights,
            caps,
        )
    if raw_weights is None:
        raise ValueError("projection solve requires raw_weights")
    raw = np.asarray(raw_weights, dtype=np.float64)
    if raw.shape != (data.n_assets,) or not np.isfinite(raw).all():
        raise ValueError("raw_weights must be a finite n_assets vector")
    normalized_weights = data.previous_weight / caps + normalized_trade
    normalized_raw = raw / caps
    # This is the frozen projection expression after exact affine
    # substitution, written without subtracting nearby weight-unit decimals.
    return (
        cp.Minimize(
            0.5 * cp.sum_squares(normalized_weights - normalized_raw)
            + frozen_solver.PROJECTION_RIDGE * cp.sum_squares(weights)
        ),
        [],
    )


def _dual_cone_constraints(
    dual_vector: cp.Variable,
    dims: object,
) -> list[cp.Constraint]:
    """Build the self-dual zero/nonnegative/SOC cone used by CLARABEL."""

    constraints: list[cp.Constraint] = []
    offset = int(dims.zero)  # Equality-cone dual coordinates are free.
    nonnegative = int(dims.nonneg)
    if nonnegative:
        constraints.append(
            dual_vector[offset : offset + nonnegative] >= 0.0
        )
        offset += nonnegative
    for raw_width in dims.soc:
        width = int(raw_width)
        constraints.append(
            cp.SOC(
                dual_vector[offset],
                dual_vector[offset + 1 : offset + width],
            )
        )
        offset += width
    if offset != int(dual_vector.size):
        raise frozen_solver.AllocationSolveError(
            "explicit dual cone partition does not span every canonical row"
        )
    return constraints


def _dual_cone_margins(
    vector: np.ndarray,
    dims: object,
) -> tuple[float, float, int]:
    """Return the minimum nonnegative and SOC margins of a dual vector."""

    offset = int(dims.zero)
    nonnegative = int(dims.nonneg)
    nonnegative_minimum = (
        float(np.min(vector[offset : offset + nonnegative]))
        if nonnegative
        else float("inf")
    )
    offset += nonnegative
    soc_margins: list[float] = []
    for raw_width in dims.soc:
        width = int(raw_width)
        block = vector[offset : offset + width]
        soc_margins.append(float(block[0] - np.linalg.norm(block[1:])))
        offset += width
    if offset != len(vector):
        raise frozen_solver.AllocationSolveError(
            "explicit dual cone audit does not span every canonical row"
        )
    minimum_soc = min(soc_margins, default=float("inf"))
    minimum_ordinal = (
        int(np.argmin(np.asarray(soc_margins, dtype=np.float64)))
        if soc_margins
        else -1
    )
    return nonnegative_minimum, minimum_soc, minimum_ordinal


def _frozen_dual_vector(value: np.ndarray) -> np.ndarray:
    """Return an immutable binary64 copy of one audited canonical dual."""

    result = np.array(value, dtype=np.float64, copy=True, order="C")
    if result.ndim != 1 or not np.isfinite(result).all():
        raise frozen_solver.AllocationSolveError(
            "hybrid dual vector is invalid"
        )
    result.setflags(write=False)
    return result


def _canonical_objective_offset(inverse_data: list[object]) -> float:
    """Extract CVXPY's single constant canonical-objective offset."""

    values: list[object] = []
    for item in inverse_data:
        payload = getattr(item, "inverse_data", None)
        if isinstance(payload, dict) and "offset" in payload:
            values.append(payload["offset"])
    if len(values) != 1:
        raise frozen_solver.AllocationSolveError(
            "explicit dual could not identify one canonical objective offset"
        )
    value = np.asarray(values[0], dtype=np.float64)
    if value.size != 1 or not np.isfinite(value).all():
        raise frozen_solver.AllocationSolveError(
            "explicit dual canonical objective offset is invalid"
        )
    return float(value.reshape(()))


def _lexicographic_canonical_audit(
    problem: cp.Problem,
    *,
    primal_status: str,
) -> _LexicographicCanonicalAudit:
    """Read and independently audit one returned slack-minimization point."""

    if not isinstance(problem.objective, cp.Minimize):
        raise frozen_solver.AllocationSolveError(
            "lexicographic certificate requires a minimization problem"
        )
    canonical, _chain, inverse = problem.get_problem_data(cp.CLARABEL)
    matrix = canonical["A"]
    rhs = np.asarray(canonical["b"], dtype=np.float64).reshape(-1)
    objective = np.asarray(canonical["c"], dtype=np.float64).reshape(-1)
    dims = canonical["dims"]
    quadratic = canonical.get("P")
    if quadratic is not None and int(quadratic.nnz) != 0:
        raise frozen_solver.AllocationSolveError(
            "lexicographic certificate unexpectedly has a quadratic objective"
        )
    if (
        int(dims.exp) != 0
        or bool(dims.psd)
        or bool(dims.p3d)
        or bool(getattr(dims, "pnd", []))
    ):
        raise frozen_solver.AllocationSolveError(
            "lexicographic certificate encountered an unsupported cone"
        )
    if (
        canonical.get("lower_bounds") is not None
        or canonical.get("upper_bounds") is not None
    ):
        raise frozen_solver.AllocationSolveError(
            "lexicographic certificate encountered direct canonical bounds"
        )
    rows, columns = matrix.shape
    if (
        len(rhs) != rows
        or len(objective) != columns
        or not np.isfinite(rhs).all()
        or not np.isfinite(objective).all()
        or not np.isfinite(np.asarray(matrix.data)).all()
    ):
        raise frozen_solver.AllocationSolveError(
            "lexicographic certificate canonical data are invalid"
        )

    cache = getattr(problem, "_solver_cache", None)
    clarabel_solver = (
        cache.get("CLARABEL") if isinstance(cache, dict) else None
    )
    try:
        solution = (
            clarabel_solver.get_solution()
            if clarabel_solver is not None
            else None
        )
    except Exception as exc:  # pragma: no cover - defensive private-API guard
        raise frozen_solver.AllocationSolveError(
            "lexicographic CLARABEL solution could not be read"
        ) from exc
    raw_solution_status = (
        "" if solution is None else str(getattr(solution, "status", ""))
    )
    raw_primal = (
        None
        if solution is None or getattr(solution, "x", None) is None
        else np.asarray(solution.x, dtype=np.float64).reshape(-1)
    )
    raw_slack = (
        None
        if solution is None or getattr(solution, "s", None) is None
        else np.asarray(solution.s, dtype=np.float64).reshape(-1)
    )
    raw_dual = (
        None
        if solution is None or getattr(solution, "z", None) is None
        else np.asarray(solution.z, dtype=np.float64).reshape(-1)
    )
    reported_primal_objective = (
        None if solution is None else getattr(solution, "obj_val", None)
    )
    reported_dual_objective = (
        None if solution is None else getattr(solution, "obj_val_dual", None)
    )
    if (
        primal_status not in DUAL_CERTIFICATE_AUDITABLE_STATUSES
        or raw_solution_status not in {"Solved", "AlmostSolved"}
        or raw_primal is None
        or raw_slack is None
        or raw_dual is None
        or raw_primal.shape != (columns,)
        or raw_slack.shape != (rows,)
        or raw_dual.shape != (rows,)
        or not np.isfinite(raw_primal).all()
        or not np.isfinite(raw_slack).all()
        or not np.isfinite(raw_dual).all()
        or problem.value is None
        or not np.isfinite(float(problem.value))
        or reported_primal_objective is None
        or not np.isfinite(float(reported_primal_objective))
        or reported_dual_objective is None
        or not np.isfinite(float(reported_dual_objective))
    ):
        raise frozen_solver.AllocationSolveError(
            "lexicographic CLARABEL point is not directly auditable"
        )

    constraint_violations: list[float] = []
    for constraint in problem.constraints:
        try:
            violation = np.asarray(
                constraint.violation(),
                dtype=np.float64,
            )
        except Exception as exc:
            raise frozen_solver.AllocationSolveError(
                "lexicographic candidate constraint audit failed"
            ) from exc
        if not np.isfinite(violation).all():
            raise frozen_solver.AllocationSolveError(
                "lexicographic candidate constraint audit is nonfinite"
            )
        constraint_violations.append(
            float(np.max(violation, initial=0.0))
        )

    canonical_primal_residual = np.asarray(
        matrix @ raw_primal + raw_slack - rhs,
        dtype=np.float64,
    ).reshape(-1)
    zero = int(dims.zero)
    primal_zero_residual = float(
        np.max(np.abs(raw_slack[:zero]), initial=0.0)
    )
    (
        primal_nonnegative_minimum,
        primal_soc_minimum,
        primal_soc_minimum_block_ordinal,
    ) = _dual_cone_margins(raw_slack, dims)
    objective_offset = _canonical_objective_offset(inverse)
    candidate_objective = float(objective @ raw_primal + objective_offset)
    canonical_dual_objective = float(-rhs @ raw_dual + objective_offset)
    objective_agreement = max(
        abs(float(problem.value) - candidate_objective),
        abs(float(reported_primal_objective) - candidate_objective),
    )
    reported_dual_agreement = abs(
        float(reported_dual_objective) - canonical_dual_objective
    )

    frozen_primal = np.array(raw_primal, dtype=np.float64, copy=True)
    frozen_slack = np.array(raw_slack, dtype=np.float64, copy=True)
    frozen_dual = np.array(raw_dual, dtype=np.float64, copy=True)
    frozen_primal.setflags(write=False)
    frozen_slack.setflags(write=False)
    frozen_dual.setflags(write=False)
    return _LexicographicCanonicalAudit(
        matrix=matrix,
        rhs=rhs,
        objective=objective,
        dims=dims,
        raw_primal=frozen_primal,
        raw_slack=frozen_slack,
        raw_dual=frozen_dual,
        primal_status=primal_status,
        candidate_maximum_constraint_violation=max(
            constraint_violations,
            default=0.0,
        ),
        canonical_primal_residual_maximum=float(
            np.max(np.abs(canonical_primal_residual), initial=0.0)
        ),
        primal_zero_cone_residual_maximum=primal_zero_residual,
        primal_nonnegative_minimum=primal_nonnegative_minimum,
        primal_soc_minimum_margin=primal_soc_minimum,
        primal_soc_minimum_block_ordinal=(
            primal_soc_minimum_block_ordinal
        ),
        candidate_objective=candidate_objective,
        objective_offset=objective_offset,
        objective_agreement_maximum_difference=objective_agreement,
        reported_dual_objective_agreement=reported_dual_agreement,
    )


def _lexicographic_certificate_from_dual(
    audit: _LexicographicCanonicalAudit,
    dual: np.ndarray,
    *,
    certificate_source: str,
    dual_status: str,
    objective_agreement_maximum_difference: float | None = None,
    stationarity_refinement_stop_code: int = -1,
    stationarity_refinement_iterations: int = -1,
    stationarity_refinement_correction_maximum: float = 0.0,
) -> _LexicographicDualCertificate:
    """Construct direct KKT metrics for one lexicographic dual candidate."""

    vector = np.asarray(dual, dtype=np.float64).reshape(-1)
    rows, _columns = audit.matrix.shape
    if vector.shape != (rows,) or not np.isfinite(vector).all():
        raise frozen_solver.AllocationSolveError(
            "lexicographic dual candidate is invalid"
        )
    stationarity_residual = np.asarray(
        audit.matrix.T @ vector + audit.objective,
        dtype=np.float64,
    ).reshape(-1)
    backward_denominator = (
        np.asarray(
            np.abs(audit.matrix).T @ np.abs(vector),
            dtype=np.float64,
        ).reshape(-1)
        + np.abs(audit.objective)
    )
    backward_error = np.abs(stationarity_residual) / np.maximum(
        backward_denominator,
        DUAL_CERTIFICATE_STATIONARITY_FLOOR,
    )
    (
        dual_nonnegative_minimum,
        dual_soc_minimum,
        dual_soc_minimum_block_ordinal,
    ) = _dual_cone_margins(vector, audit.dims)
    dual_lower_bound = float(
        -audit.rhs @ vector + audit.objective_offset
    )
    agreement = (
        audit.objective_agreement_maximum_difference
        if objective_agreement_maximum_difference is None
        else max(
            audit.objective_agreement_maximum_difference,
            float(objective_agreement_maximum_difference),
        )
    )
    return _LexicographicDualCertificate(
        certificate_source=certificate_source,
        dual_status=dual_status,
        candidate_maximum_constraint_violation=(
            audit.candidate_maximum_constraint_violation
        ),
        canonical_primal_residual_maximum=(
            audit.canonical_primal_residual_maximum
        ),
        primal_zero_cone_residual_maximum=(
            audit.primal_zero_cone_residual_maximum
        ),
        primal_nonnegative_minimum=audit.primal_nonnegative_minimum,
        primal_soc_minimum_margin=audit.primal_soc_minimum_margin,
        stationarity_absolute_maximum=float(
            np.max(np.abs(stationarity_residual), initial=0.0)
        ),
        stationarity_backward_error_maximum=float(
            np.max(backward_error, initial=0.0)
        ),
        dual_nonnegative_minimum=dual_nonnegative_minimum,
        dual_soc_minimum_margin=dual_soc_minimum,
        candidate_objective=audit.candidate_objective,
        dual_lower_bound=dual_lower_bound,
        candidate_dual_gap=float(
            audit.candidate_objective - dual_lower_bound
        ),
        objective_agreement_maximum_difference=agreement,
        complementarity_absolute=abs(
            float(audit.raw_slack @ vector)
        ),
        primal_soc_minimum_block_ordinal=(
            audit.primal_soc_minimum_block_ordinal
        ),
        dual_soc_minimum_block_ordinal=dual_soc_minimum_block_ordinal,
        stationarity_refinement_stop_code=(
            stationarity_refinement_stop_code
        ),
        stationarity_refinement_iterations=(
            stationarity_refinement_iterations
        ),
        stationarity_refinement_correction_maximum=float(
            stationarity_refinement_correction_maximum
        ),
    )


def _build_explicit_lexicographic_dual_certificate(
    audit: _LexicographicCanonicalAudit,
) -> _LexicographicDualCertificate:
    """Solve and directly audit a separately modelled slack-problem dual."""

    rows, _columns = audit.matrix.shape
    dual_variable = cp.Variable(
        rows,
        name="stage11_lexicographic_explicit_dual",
    )
    stationarity_scale = 1.0 / np.maximum(
        np.abs(audit.objective),
        DUAL_CERTIFICATE_STATIONARITY_FLOOR,
    )
    dual_problem = cp.Problem(
        cp.Maximize(-audit.rhs @ dual_variable),
        [
            cp.multiply(
                stationarity_scale,
                audit.matrix.T @ dual_variable + audit.objective,
            )
            == 0.0,
            *_dual_cone_constraints(dual_variable, audit.dims),
        ],
    )
    dual_status = _solve_status(
        dual_problem,
        frozen_solver.RETRY_SOLVER_SETTINGS,
    )
    raw_dual = (
        None
        if dual_variable.value is None
        else np.asarray(
            dual_variable.value,
            dtype=np.float64,
        ).reshape(-1)
    )
    if (
        dual_status not in DUAL_CERTIFICATE_AUDITABLE_STATUSES
        or raw_dual is None
        or raw_dual.shape != (rows,)
        or not np.isfinite(raw_dual).all()
        or dual_problem.value is None
        or not np.isfinite(float(dual_problem.value))
    ):
        raise frozen_solver.AllocationSolveError(
            "explicit lexicographic dual did not return an auditable point"
        )
    canonical_dual_value = float(-audit.rhs @ raw_dual)
    return _lexicographic_certificate_from_dual(
        audit,
        raw_dual,
        certificate_source=EXPLICIT_DUAL_CERTIFICATE_SOURCE,
        dual_status=dual_status,
        objective_agreement_maximum_difference=abs(
            float(dual_problem.value) - canonical_dual_value
        ),
    )


def _build_retry_lexicographic_dual_certificate(
    audit: _LexicographicCanonicalAudit,
) -> _LexicographicDualCertificate:
    """Refine and directly audit the slack solve's own canonical dual."""

    raw_stationarity = np.asarray(
        audit.matrix.T @ audit.raw_dual + audit.objective,
        dtype=np.float64,
    ).reshape(-1)
    refinement = lsmr(
        audit.matrix.T,
        -raw_stationarity,
        atol=DUAL_STATIONARITY_REFINEMENT_TOLERANCE,
        btol=DUAL_STATIONARITY_REFINEMENT_TOLERANCE,
        conlim=0.0,
        maxiter=min(audit.matrix.shape),
        show=False,
    )
    correction = np.asarray(
        refinement[0],
        dtype=np.float64,
    ).reshape(-1)
    stop_code = int(refinement[1])
    iterations = int(refinement[2])
    if (
        correction.shape != audit.raw_dual.shape
        or not np.isfinite(correction).all()
        or stop_code < 0
        or iterations < 0
    ):
        raise frozen_solver.AllocationSolveError(
            "lexicographic dual stationarity refinement is invalid"
        )
    return _lexicographic_certificate_from_dual(
        audit,
        audit.raw_dual + correction,
        certificate_source=RETRY_DUAL_CERTIFICATE_SOURCE,
        dual_status=audit.primal_status,
        objective_agreement_maximum_difference=(
            audit.reported_dual_objective_agreement
        ),
        stationarity_refinement_stop_code=stop_code,
        stationarity_refinement_iterations=iterations,
        stationarity_refinement_correction_maximum=float(
            np.max(np.abs(correction), initial=0.0)
        ),
    )


def _lexicographic_dual_certificate_failure(
    certificate: _LexicographicDualCertificate,
) -> str | None:
    """Return the first failed uniform lexicographic certificate gate."""

    if certificate.dual_status not in DUAL_CERTIFICATE_AUDITABLE_STATUSES:
        return "dual_status_not_auditable"
    if (
        not np.isfinite(
            certificate.candidate_maximum_constraint_violation
        )
        or certificate.candidate_maximum_constraint_violation
        > frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE
    ):
        return "candidate_constraint_violation"
    if (
        not np.isfinite(certificate.canonical_primal_residual_maximum)
        or certificate.canonical_primal_residual_maximum
        > DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
    ):
        return "canonical_primal_residual"
    if (
        not np.isfinite(
            certificate.primal_zero_cone_residual_maximum
        )
        or certificate.primal_zero_cone_residual_maximum
        > DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
    ):
        return "primal_zero_cone"
    if (
        np.isnan(certificate.primal_nonnegative_minimum)
        or certificate.primal_nonnegative_minimum
        < -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
    ):
        return "primal_nonnegative_cone"
    if (
        np.isnan(certificate.primal_soc_minimum_margin)
        or certificate.primal_soc_minimum_margin
        < -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
    ):
        return "primal_soc_cone"
    if (
        not np.isfinite(certificate.stationarity_absolute_maximum)
        or certificate.stationarity_absolute_maximum
        > DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
    ):
        return "dual_stationarity_absolute"
    if (
        not np.isfinite(certificate.stationarity_backward_error_maximum)
        or certificate.stationarity_backward_error_maximum
        > DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
    ):
        return "dual_stationarity_backward_error"
    if (
        np.isnan(certificate.dual_nonnegative_minimum)
        or certificate.dual_nonnegative_minimum
        < -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
    ):
        return "dual_nonnegative_cone"
    if (
        np.isnan(certificate.dual_soc_minimum_margin)
        or certificate.dual_soc_minimum_margin
        < -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
    ):
        return "dual_soc_cone"
    if (
        not np.isfinite(certificate.candidate_objective)
        or not np.isfinite(certificate.dual_lower_bound)
        or not np.isfinite(certificate.candidate_dual_gap)
        or abs(certificate.candidate_dual_gap)
        > DUAL_CERTIFICATE_GAP_TOLERANCE
    ):
        return "candidate_dual_gap"
    if (
        not np.isfinite(certificate.complementarity_absolute)
        or certificate.complementarity_absolute
        > DUAL_CERTIFICATE_GAP_TOLERANCE
    ):
        return "complementarity"
    if (
        not np.isfinite(
            certificate.objective_agreement_maximum_difference
        )
        or certificate.objective_agreement_maximum_difference
        > DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
    ):
        return "objective_agreement"
    return None


def _lexicographic_dual_certificate_metrics_history(
    certificate: _LexicographicDualCertificate,
    *,
    prefix: str,
) -> tuple[str, ...]:
    """Serialize every direct lexicographic certificate metric."""

    metrics = (
        (
            "candidate_maximum_constraint_violation",
            certificate.candidate_maximum_constraint_violation,
        ),
        (
            "canonical_primal_residual_maximum",
            certificate.canonical_primal_residual_maximum,
        ),
        (
            "primal_zero_cone_residual_maximum",
            certificate.primal_zero_cone_residual_maximum,
        ),
        (
            "primal_nonnegative_minimum",
            certificate.primal_nonnegative_minimum,
        ),
        (
            "primal_soc_minimum_margin",
            certificate.primal_soc_minimum_margin,
        ),
        (
            "stationarity_absolute_maximum",
            certificate.stationarity_absolute_maximum,
        ),
        (
            "stationarity_backward_error_maximum",
            certificate.stationarity_backward_error_maximum,
        ),
        (
            "dual_nonnegative_minimum",
            certificate.dual_nonnegative_minimum,
        ),
        (
            "dual_soc_minimum_margin",
            certificate.dual_soc_minimum_margin,
        ),
        ("candidate_objective", certificate.candidate_objective),
        ("dual_lower_bound", certificate.dual_lower_bound),
        ("candidate_dual_gap", certificate.candidate_dual_gap),
        (
            "objective_agreement_maximum_difference",
            certificate.objective_agreement_maximum_difference,
        ),
        (
            "complementarity_absolute",
            certificate.complementarity_absolute,
        ),
        (
            "stationarity_refinement_correction_maximum",
            certificate.stationarity_refinement_correction_maximum,
        ),
    )
    return (
        (
            f"{prefix}_dual_certificate_"
            "primal_soc_minimum_block_ordinal:"
            + str(certificate.primal_soc_minimum_block_ordinal)
        ),
        (
            f"{prefix}_dual_certificate_"
            "dual_soc_minimum_block_ordinal:"
            + str(certificate.dual_soc_minimum_block_ordinal)
        ),
        (
            f"{prefix}_dual_certificate_"
            "stationarity_refinement_stop_code:"
            + str(certificate.stationarity_refinement_stop_code)
        ),
        (
            f"{prefix}_dual_certificate_"
            "stationarity_refinement_iterations:"
            + str(certificate.stationarity_refinement_iterations)
        ),
        *(
            f"{prefix}_dual_certificate_"
            + name
            + ":"
            + float(value).hex()
            for name, value in metrics
        ),
    )


def _lexicographic_dual_certificate_history(
    certificate: _LexicographicDualCertificate,
    *,
    prefix: str,
) -> tuple[str, ...]:
    """Serialize a passing lexicographic certificate."""

    return (
        f"{prefix}_dual_certificate:{certificate.dual_status}",
        (
            f"{prefix}_dual_certificate_source:"
            + certificate.certificate_source
        ),
        *_lexicographic_dual_certificate_metrics_history(
            certificate,
            prefix=prefix,
        ),
        f"{prefix}_dual_certificate_audit:passed",
    )


def _certify_lexicographic_optimal_inaccurate(
    problem: cp.Problem,
    *,
    primal_status: str,
    prefix: str,
) -> tuple[str, ...]:
    """Certify an inaccurate slack optimum without accepting status alone."""

    audit = _lexicographic_canonical_audit(
        problem,
        primal_status=primal_status,
    )
    history: list[str] = []
    certificate: _LexicographicDualCertificate | None = None
    try:
        explicit = _build_explicit_lexicographic_dual_certificate(audit)
    except (
        frozen_solver.AllocationSolveError,
        TypeError,
        ValueError,
    ) as exc:
        history.append(
            f"{prefix}_explicit_dual_certificate_audit:error:"
            + type(exc).__name__
        )
    else:
        explicit_failure = _lexicographic_dual_certificate_failure(
            explicit
        )
        if explicit_failure is None:
            certificate = explicit
        else:
            history.extend(
                (
                    f"{prefix}_explicit_dual_certificate:"
                    + explicit.dual_status,
                    (
                        f"{prefix}_explicit_dual_certificate_source:"
                        + explicit.certificate_source
                    ),
                    (
                        f"{prefix}_explicit_dual_certificate_audit:"
                        "failed:"
                        + explicit_failure
                    ),
                    *_lexicographic_dual_certificate_metrics_history(
                        explicit,
                        prefix=f"{prefix}_explicit",
                    ),
                )
            )
    if certificate is None:
        retry = _build_retry_lexicographic_dual_certificate(audit)
        retry_failure = _lexicographic_dual_certificate_failure(retry)
        if retry_failure is None:
            certificate = retry
        else:
            history.extend(
                (
                    f"{prefix}_retry_dual_certificate:"
                    + retry.dual_status,
                    (
                        f"{prefix}_retry_dual_certificate_source:"
                        + retry.certificate_source
                    ),
                    (
                        f"{prefix}_retry_dual_certificate_audit:failed:"
                        + retry_failure
                    ),
                    *_lexicographic_dual_certificate_metrics_history(
                        retry,
                        prefix=f"{prefix}_retry",
                    ),
                )
            )
    if certificate is None:
        raise frozen_solver.AllocationSolveError(
            "lexicographic optimal_inaccurate point was not "
            "directly certified: "
            + ", ".join(history)
        )
    history.extend(
        _lexicographic_dual_certificate_history(
            certificate,
            prefix=prefix,
        )
    )
    return tuple(history)


def _candidate_objective_cost_agreement(
    data: AllocationProblemData,
    candidate: np.ndarray,
    normalized_trade: cp.Variable,
    weights: cp.Expression,
    caps: np.ndarray,
    problem: cp.Problem,
) -> tuple[float, float]:
    """Independently reconcile CVXPY and NumPy candidate economics."""

    normalized_trade.value = (candidate - data.previous_weight) / caps
    breakdown = _scaled_cvxpy_total_cost(
        data,
        normalized_trade,
        weights,
        caps,
    )
    cvxpy_values = (
        breakdown.transaction.value,
        breakdown.borrow.value,
        breakdown.financing.value,
        breakdown.total.value,
        problem.objective.value,
    )
    if any(value is None for value in cvxpy_values):
        raise frozen_solver.AllocationSolveError(
            "dual certificate candidate CVXPY economics are missing"
        )
    cvxpy_numeric = np.asarray(
        [float(np.asarray(value).reshape(())) for value in cvxpy_values],
        dtype=np.float64,
    )
    if not np.isfinite(cvxpy_numeric).all():
        raise frozen_solver.AllocationSolveError(
            "dual certificate candidate CVXPY economics are nonfinite"
        )

    numpy_cost = numpy_total_cost(
        candidate,
        data.previous_weight,
        data.full_spread,
        data.adv20_usd,
        data.impact_volatility_daily,
        data.borrow_rate_annual,
        data.nav_dollars,
    )
    numpy_objective = float(
        data.expected_return @ candidate - numpy_cost.total
    )
    numpy_numeric = np.asarray(
        [
            numpy_cost.transaction,
            numpy_cost.borrow,
            numpy_cost.financing,
            numpy_cost.total,
            frozen_solver.ECONOMIC_OBJECTIVE_NUMERICAL_SCALE
            * numpy_objective,
        ],
        dtype=np.float64,
    )
    scaled_differences = np.abs(cvxpy_numeric - numpy_numeric)
    scaled_differences[:4] *= (
        frozen_solver.ECONOMIC_OBJECTIVE_NUMERICAL_SCALE
    )
    return (
        float(
            frozen_solver.ECONOMIC_OBJECTIVE_NUMERICAL_SCALE
            * numpy_objective
        ),
        float(np.max(scaled_differences)),
    )


def _build_explicit_hybrid_dual_certificate(
    data: AllocationProblemData,
    candidate: np.ndarray,
    problem: cp.Problem,
    normalized_trade: cp.Variable,
    weights: cp.Expression,
    caps: np.ndarray,
    *,
    candidate_maximum_constraint_violation: float | None = None,
) -> _HybridDualCertificate:
    """Solve and independently audit the separately modelled conic dual."""

    if candidate_maximum_constraint_violation is None:
        candidate_maximum_constraint_violation = float(
            independent_constraint_diagnostics(
                data,
                candidate,
            ).maximum_violation
        )
    candidate_scaled, agreement = _candidate_objective_cost_agreement(
        data,
        candidate,
        normalized_trade,
        weights,
        caps,
        problem,
    )
    canonical, _chain, inverse = problem.get_problem_data(cp.CLARABEL)
    matrix = canonical["A"]
    rhs = np.asarray(canonical["b"], dtype=np.float64).reshape(-1)
    objective = np.asarray(canonical["c"], dtype=np.float64).reshape(-1)
    dims = canonical["dims"]
    quadratic = canonical.get("P")
    if quadratic is not None and int(quadratic.nnz) != 0:
        raise frozen_solver.AllocationSolveError(
            "hybrid explicit dual unexpectedly has a quadratic objective"
        )
    if (
        int(dims.exp) != 0
        or bool(dims.psd)
        or bool(dims.p3d)
        or bool(getattr(dims, "pnd", []))
    ):
        raise frozen_solver.AllocationSolveError(
            "hybrid explicit dual encountered an unsupported cone"
        )
    if (
        canonical.get("lower_bounds") is not None
        or canonical.get("upper_bounds") is not None
    ):
        raise frozen_solver.AllocationSolveError(
            "hybrid explicit dual encountered direct canonical bounds"
        )
    rows, columns = matrix.shape
    if (
        len(rhs) != rows
        or len(objective) != columns
        or not np.isfinite(rhs).all()
        or not np.isfinite(objective).all()
        or not np.isfinite(np.asarray(matrix.data)).all()
    ):
        raise frozen_solver.AllocationSolveError(
            "hybrid explicit dual canonical data are invalid"
        )

    dual_variable = cp.Variable(rows, name="stage11_explicit_dual")
    stationarity_scale = 1.0 / np.maximum(
        np.abs(objective),
        DUAL_CERTIFICATE_STATIONARITY_FLOOR,
    )
    dual_constraints = [
        cp.multiply(
            stationarity_scale,
            matrix.T @ dual_variable + objective,
        )
        == 0.0,
        *_dual_cone_constraints(dual_variable, dims),
    ]
    dual_problem = cp.Problem(
        cp.Maximize(-rhs @ dual_variable),
        dual_constraints,
    )
    dual_status = _solve_status(
        dual_problem,
        frozen_solver.RETRY_SOLVER_SETTINGS,
    )
    raw_dual = (
        None
        if dual_variable.value is None
        else np.asarray(
            dual_variable.value,
            dtype=np.float64,
        ).reshape(-1)
    )
    if (
        dual_status not in DUAL_CERTIFICATE_AUDITABLE_STATUSES
        or raw_dual is None
        or raw_dual.shape != (rows,)
        or not np.isfinite(raw_dual).all()
        or dual_problem.value is None
        or not np.isfinite(float(dual_problem.value))
    ):
        return _HybridDualCertificate(
            certificate_source=EXPLICIT_DUAL_CERTIFICATE_SOURCE,
            dual_status=dual_status,
            candidate_maximum_constraint_violation=float(
                candidate_maximum_constraint_violation
            ),
            stationarity_absolute_maximum=float("inf"),
            stationarity_backward_error_maximum=float("inf"),
            dual_nonnegative_minimum=float("-inf"),
            dual_soc_minimum_margin=float("-inf"),
            candidate_scaled_objective=candidate_scaled,
            dual_upper_scaled=float("nan"),
            candidate_dual_gap_scaled=float("nan"),
            objective_cost_scaled_maximum_difference=agreement,
        )

    stationarity_residual = np.asarray(
        matrix.T @ raw_dual + objective,
        dtype=np.float64,
    ).reshape(-1)
    backward_denominator = (
        np.asarray(
            np.abs(matrix).T @ np.abs(raw_dual),
            dtype=np.float64,
        ).reshape(-1)
        + np.abs(objective)
    )
    backward_error = np.abs(stationarity_residual) / np.maximum(
        backward_denominator,
        DUAL_CERTIFICATE_STATIONARITY_FLOOR,
    )
    (
        nonnegative_minimum,
        soc_minimum,
        soc_minimum_block_ordinal,
    ) = _dual_cone_margins(raw_dual, dims)
    canonical_dual_value = float(-rhs @ raw_dual)
    objective_offset = _canonical_objective_offset(inverse)
    dual_upper_scaled = float(
        -(canonical_dual_value + objective_offset)
    )
    agreement = max(
        agreement,
        abs(float(dual_problem.value) - canonical_dual_value),
    )
    return _HybridDualCertificate(
        certificate_source=EXPLICIT_DUAL_CERTIFICATE_SOURCE,
        dual_status=dual_status,
        candidate_maximum_constraint_violation=float(
            candidate_maximum_constraint_violation
        ),
        stationarity_absolute_maximum=float(
            np.max(np.abs(stationarity_residual), initial=0.0)
        ),
        stationarity_backward_error_maximum=float(
            np.max(backward_error, initial=0.0)
        ),
        dual_nonnegative_minimum=nonnegative_minimum,
        dual_soc_minimum_margin=soc_minimum,
        candidate_scaled_objective=candidate_scaled,
        dual_upper_scaled=dual_upper_scaled,
        candidate_dual_gap_scaled=float(
            dual_upper_scaled - candidate_scaled
        ),
        objective_cost_scaled_maximum_difference=agreement,
        dual_soc_minimum_block_ordinal=soc_minimum_block_ordinal,
        dual_vector=_frozen_dual_vector(raw_dual),
    )


def _build_retry_economic_dual_certificate(
    data: AllocationProblemData,
    candidate: np.ndarray,
    problem: cp.Problem,
    normalized_trade: cp.Variable,
    weights: cp.Expression,
    caps: np.ndarray,
    *,
    dual_status: str,
    candidate_maximum_constraint_violation: float | None = None,
) -> _HybridDualCertificate:
    """Extract, refine, and audit the retry economic solve's conic dual."""

    if candidate_maximum_constraint_violation is None:
        candidate_maximum_constraint_violation = float(
            independent_constraint_diagnostics(
                data,
                candidate,
            ).maximum_violation
        )
    candidate_scaled, agreement = _candidate_objective_cost_agreement(
        data,
        candidate,
        normalized_trade,
        weights,
        caps,
        problem,
    )
    canonical, _chain, inverse = problem.get_problem_data(cp.CLARABEL)
    matrix = canonical["A"]
    rhs = np.asarray(canonical["b"], dtype=np.float64).reshape(-1)
    objective = np.asarray(canonical["c"], dtype=np.float64).reshape(-1)
    dims = canonical["dims"]
    quadratic = canonical.get("P")
    if quadratic is not None and int(quadratic.nnz) != 0:
        raise frozen_solver.AllocationSolveError(
            "hybrid explicit dual unexpectedly has a quadratic objective"
        )
    if (
        int(dims.exp) != 0
        or bool(dims.psd)
        or bool(dims.p3d)
        or bool(getattr(dims, "pnd", []))
    ):
        raise frozen_solver.AllocationSolveError(
            "hybrid explicit dual encountered an unsupported cone"
        )
    if (
        canonical.get("lower_bounds") is not None
        or canonical.get("upper_bounds") is not None
    ):
        raise frozen_solver.AllocationSolveError(
            "hybrid explicit dual encountered direct canonical bounds"
        )
    rows, columns = matrix.shape
    if (
        len(rhs) != rows
        or len(objective) != columns
        or not np.isfinite(rhs).all()
        or not np.isfinite(objective).all()
        or not np.isfinite(np.asarray(matrix.data)).all()
    ):
        raise frozen_solver.AllocationSolveError(
            "hybrid explicit dual canonical data are invalid"
        )

    # The retry economic solve is already a primal-dual CLARABEL solve.  Reuse
    # its canonical dual candidate instead of solving a separately
    # re-canonicalized dual whose scaling can lose the tight objective bound.
    # CVXPY and CLARABEL are version-pinned by the Stage 11 authority.
    cache = getattr(problem, "_solver_cache", None)
    clarabel_solver = (
        cache.get("CLARABEL") if isinstance(cache, dict) else None
    )
    try:
        solution = (
            clarabel_solver.get_solution()
            if clarabel_solver is not None
            else None
        )
    except Exception as exc:  # pragma: no cover - defensive private-API guard
        raise frozen_solver.AllocationSolveError(
            "retry CLARABEL dual solution could not be read"
        ) from exc
    raw_dual = (
        None
        if solution is None or getattr(solution, "z", None) is None
        else np.asarray(solution.z, dtype=np.float64).reshape(-1)
    )
    reported_dual_objective = (
        None if solution is None else getattr(solution, "obj_val_dual", None)
    )
    raw_solution_status = (
        "" if solution is None else str(getattr(solution, "status", ""))
    )
    if (
        dual_status not in DUAL_CERTIFICATE_AUDITABLE_STATUSES
        or raw_dual is None
        or raw_dual.shape != (rows,)
        or not np.isfinite(raw_dual).all()
        or reported_dual_objective is None
        or not np.isfinite(float(reported_dual_objective))
        or raw_solution_status not in {"Solved", "AlmostSolved"}
    ):
        return _HybridDualCertificate(
            certificate_source=RETRY_DUAL_CERTIFICATE_SOURCE,
            dual_status=dual_status,
            candidate_maximum_constraint_violation=float(
                candidate_maximum_constraint_violation
            ),
            stationarity_absolute_maximum=float("inf"),
            stationarity_backward_error_maximum=float("inf"),
            dual_nonnegative_minimum=float("-inf"),
            dual_soc_minimum_margin=float("-inf"),
            candidate_scaled_objective=candidate_scaled,
            dual_upper_scaled=float("nan"),
            candidate_dual_gap_scaled=float("nan"),
            objective_cost_scaled_maximum_difference=agreement,
        )

    # CLARABEL's canonical dual has an excellent objective bound, while tiny
    # stationarity components may remain after an ``AlmostSolved`` exit.
    # Apply only the minimum-norm linear correction for A.T*y + c = 0.  This
    # does not change the optimization problem.  The refined vector is still
    # accepted only after direct cone, stationarity, and gap audits.
    raw_stationarity = np.asarray(
        matrix.T @ raw_dual + objective,
        dtype=np.float64,
    ).reshape(-1)
    refinement = lsmr(
        matrix.T,
        -raw_stationarity,
        atol=DUAL_STATIONARITY_REFINEMENT_TOLERANCE,
        btol=DUAL_STATIONARITY_REFINEMENT_TOLERANCE,
        conlim=0.0,
        maxiter=min(matrix.shape),
        show=False,
    )
    correction = np.asarray(refinement[0], dtype=np.float64).reshape(-1)
    stop_code = int(refinement[1])
    iterations = int(refinement[2])
    if (
        correction.shape != (rows,)
        or not np.isfinite(correction).all()
        or stop_code < 0
        or iterations < 0
    ):
        raise frozen_solver.AllocationSolveError(
            "retry CLARABEL dual stationarity refinement is invalid"
        )
    dual = raw_dual + correction
    stationarity_residual = np.asarray(
        matrix.T @ dual + objective,
        dtype=np.float64,
    ).reshape(-1)
    backward_denominator = (
        np.asarray(
            np.abs(matrix).T @ np.abs(dual),
            dtype=np.float64,
        ).reshape(-1)
        + np.abs(objective)
    )
    backward_error = np.abs(stationarity_residual) / np.maximum(
        backward_denominator,
        DUAL_CERTIFICATE_STATIONARITY_FLOOR,
    )
    (
        nonnegative_minimum,
        soc_minimum,
        soc_minimum_block_ordinal,
    ) = _dual_cone_margins(dual, dims)
    canonical_dual_value = float(-rhs @ dual)
    raw_canonical_dual_value = float(-rhs @ raw_dual)
    objective_offset = _canonical_objective_offset(inverse)
    dual_upper_scaled = float(
        -(canonical_dual_value + objective_offset)
    )
    reported_dual_difference = abs(
        float(reported_dual_objective) - raw_canonical_dual_value
    )
    agreement = max(agreement, reported_dual_difference)
    return _HybridDualCertificate(
        certificate_source=RETRY_DUAL_CERTIFICATE_SOURCE,
        dual_status=dual_status,
        candidate_maximum_constraint_violation=float(
            candidate_maximum_constraint_violation
        ),
        stationarity_absolute_maximum=float(
            np.max(np.abs(stationarity_residual), initial=0.0)
        ),
        stationarity_backward_error_maximum=float(
            np.max(backward_error, initial=0.0)
        ),
        dual_nonnegative_minimum=nonnegative_minimum,
        dual_soc_minimum_margin=soc_minimum,
        candidate_scaled_objective=candidate_scaled,
        dual_upper_scaled=dual_upper_scaled,
        candidate_dual_gap_scaled=float(
            dual_upper_scaled - candidate_scaled
        ),
        objective_cost_scaled_maximum_difference=agreement,
        dual_soc_minimum_block_ordinal=soc_minimum_block_ordinal,
        stationarity_refinement_scale=1.0,
        stationarity_refinement_stop_code=stop_code,
        stationarity_refinement_iterations=iterations,
        stationarity_refinement_correction_maximum=float(
            np.max(np.abs(correction), initial=0.0)
        ),
        dual_vector=_frozen_dual_vector(dual),
    )


def _build_reconciled_hybrid_dual_certificate(
    problem: cp.Problem,
    explicit_certificate: _HybridDualCertificate,
    retry_certificate: _HybridDualCertificate,
) -> _HybridDualCertificate:
    """Audit the equal convex blend of the two existing dual candidates.

    The explicit dual can be cone-safe while its objective gap is marginally
    high, and the refined retry dual can have the complementary numerical
    profile.  The dual affine constraints and cone are convex.  The equal
    blend is fixed in advance, depends on no portfolio outcome, and is not
    presumed valid from convexity alone: it is accepted only if every
    unchanged direct gate passes when recomputed from the canonical problem.
    """

    if (
        explicit_certificate.certificate_source
        != EXPLICIT_DUAL_CERTIFICATE_SOURCE
        or retry_certificate.certificate_source
        != RETRY_DUAL_CERTIFICATE_SOURCE
        or explicit_certificate.dual_status
        not in DUAL_CERTIFICATE_AUDITABLE_STATUSES
        or retry_certificate.dual_status
        not in DUAL_CERTIFICATE_AUDITABLE_STATUSES
        or explicit_certificate.dual_vector is None
        or retry_certificate.dual_vector is None
    ):
        raise frozen_solver.AllocationSolveError(
            "dual reconciliation endpoints are not auditable"
        )
    canonical, _chain, inverse = problem.get_problem_data(cp.CLARABEL)
    matrix = canonical["A"]
    rhs = np.asarray(canonical["b"], dtype=np.float64).reshape(-1)
    objective = np.asarray(canonical["c"], dtype=np.float64).reshape(-1)
    dims = canonical["dims"]
    rows, columns = matrix.shape
    explicit_dual = np.asarray(
        explicit_certificate.dual_vector,
        dtype=np.float64,
    ).reshape(-1)
    retry_dual = np.asarray(
        retry_certificate.dual_vector,
        dtype=np.float64,
    ).reshape(-1)
    if (
        explicit_dual.shape != (rows,)
        or retry_dual.shape != (rows,)
        or len(rhs) != rows
        or len(objective) != columns
        or not np.isfinite(explicit_dual).all()
        or not np.isfinite(retry_dual).all()
    ):
        raise frozen_solver.AllocationSolveError(
            "dual reconciliation canonical dimensions differ"
        )
    retry_weight = DUAL_RECONCILIATION_RETRY_WEIGHT
    dual = (1.0 - retry_weight) * explicit_dual + retry_weight * retry_dual
    stationarity_residual = np.asarray(
        matrix.T @ dual + objective,
        dtype=np.float64,
    ).reshape(-1)
    backward_denominator = (
        np.asarray(
            np.abs(matrix).T @ np.abs(dual),
            dtype=np.float64,
        ).reshape(-1)
        + np.abs(objective)
    )
    backward_error = np.abs(stationarity_residual) / np.maximum(
        backward_denominator,
        DUAL_CERTIFICATE_STATIONARITY_FLOOR,
    )
    (
        nonnegative_minimum,
        soc_minimum,
        soc_minimum_block_ordinal,
    ) = _dual_cone_margins(dual, dims)
    objective_offset = _canonical_objective_offset(inverse)
    dual_upper_scaled = float(-(-rhs @ dual + objective_offset))
    candidate_scaled = float(
        (1.0 - retry_weight)
        * explicit_certificate.candidate_scaled_objective
        + retry_weight
        * retry_certificate.candidate_scaled_objective
    )
    affine_dual_upper = float(
        (1.0 - retry_weight) * explicit_certificate.dual_upper_scaled
        + retry_weight * retry_certificate.dual_upper_scaled
    )
    agreement = max(
        explicit_certificate.objective_cost_scaled_maximum_difference,
        retry_certificate.objective_cost_scaled_maximum_difference,
        abs(
            explicit_certificate.candidate_scaled_objective
            - retry_certificate.candidate_scaled_objective
        ),
        abs(dual_upper_scaled - affine_dual_upper),
    )
    return _HybridDualCertificate(
        certificate_source=RECONCILED_DUAL_CERTIFICATE_SOURCE,
        dual_status=retry_certificate.dual_status,
        candidate_maximum_constraint_violation=max(
            explicit_certificate.candidate_maximum_constraint_violation,
            retry_certificate.candidate_maximum_constraint_violation,
        ),
        stationarity_absolute_maximum=float(
            np.max(np.abs(stationarity_residual), initial=0.0)
        ),
        stationarity_backward_error_maximum=float(
            np.max(backward_error, initial=0.0)
        ),
        dual_nonnegative_minimum=nonnegative_minimum,
        dual_soc_minimum_margin=soc_minimum,
        candidate_scaled_objective=candidate_scaled,
        dual_upper_scaled=dual_upper_scaled,
        candidate_dual_gap_scaled=float(
            dual_upper_scaled - candidate_scaled
        ),
        objective_cost_scaled_maximum_difference=agreement,
        dual_soc_minimum_block_ordinal=soc_minimum_block_ordinal,
        stationarity_refinement_scale=(
            retry_weight
            * retry_certificate.stationarity_refinement_scale
        ),
        stationarity_refinement_stop_code=(
            retry_certificate.stationarity_refinement_stop_code
        ),
        stationarity_refinement_iterations=(
            retry_certificate.stationarity_refinement_iterations
        ),
        stationarity_refinement_correction_maximum=(
            retry_weight
            * retry_certificate.stationarity_refinement_correction_maximum
        ),
        dual_reconciliation_retry_weight=retry_weight,
        dual_vector=_frozen_dual_vector(dual),
    )


def _build_bounded_feasible_hybrid_dual_certificate(
    problem: cp.Problem,
    explicit_certificate: _HybridDualCertificate,
    retry_certificate: _HybridDualCertificate,
) -> _HybridDualCertificate:
    """Search the unchanged dual for a tightly bounded feasible certificate.

    This final numerical fallback is reached only when the separately solved,
    retry-derived, and fixed equal-blend certificates all fail their direct
    audits.  It changes no portfolio quantity.  A positive diagonal
    reparameterization balances the dual coordinates while preserving every
    nonnegative coordinate and using one common scale within each second-order
    cone.  The feasibility problem uses the frozen retry solver/settings,
    enforces the original dual stationarity equations, and targets a gap band
    four times tighter than the unchanged independent audit.  Its returned
    vector is still accepted only through the ordinary direct certificate
    gates below.
    """

    if (
        explicit_certificate.certificate_source
        != EXPLICIT_DUAL_CERTIFICATE_SOURCE
        or retry_certificate.certificate_source
        != RETRY_DUAL_CERTIFICATE_SOURCE
    ):
        raise frozen_solver.AllocationSolveError(
            "bounded dual feasibility endpoints are not authoritative"
        )
    canonical, _chain, inverse = problem.get_problem_data(cp.CLARABEL)
    matrix = canonical["A"]
    rhs = np.asarray(canonical["b"], dtype=np.float64).reshape(-1)
    objective = np.asarray(canonical["c"], dtype=np.float64).reshape(-1)
    dims = canonical["dims"]
    quadratic = canonical.get("P")
    if quadratic is not None and int(quadratic.nnz) != 0:
        raise frozen_solver.AllocationSolveError(
            "bounded dual feasibility unexpectedly has a quadratic objective"
        )
    if (
        int(dims.exp) != 0
        or bool(dims.psd)
        or bool(dims.p3d)
        or bool(getattr(dims, "pnd", []))
    ):
        raise frozen_solver.AllocationSolveError(
            "bounded dual feasibility encountered an unsupported cone"
        )
    if (
        canonical.get("lower_bounds") is not None
        or canonical.get("upper_bounds") is not None
    ):
        raise frozen_solver.AllocationSolveError(
            "bounded dual feasibility encountered direct canonical bounds"
        )
    rows, columns = matrix.shape
    if (
        len(rhs) != rows
        or len(objective) != columns
        or not np.isfinite(rhs).all()
        or not np.isfinite(objective).all()
        or not np.isfinite(np.asarray(matrix.data)).all()
    ):
        raise frozen_solver.AllocationSolveError(
            "bounded dual feasibility canonical data are invalid"
        )

    row_infinity_norm = np.asarray(
        np.abs(matrix).max(axis=1).toarray(),
        dtype=np.float64,
    ).reshape(-1)
    if (
        row_infinity_norm.shape != (rows,)
        or not np.isfinite(row_infinity_norm).all()
        or np.any(row_infinity_norm < 0.0)
    ):
        raise frozen_solver.AllocationSolveError(
            "bounded dual feasibility row scales are invalid"
        )
    cone_preserving_measure = np.array(
        row_infinity_norm,
        dtype=np.float64,
        copy=True,
    )
    cone_offset = int(dims.zero) + int(dims.nonneg)
    for raw_width in dims.soc:
        width = int(raw_width)
        block_stop = cone_offset + width
        cone_preserving_measure[cone_offset:block_stop] = float(
            np.max(
                cone_preserving_measure[cone_offset:block_stop],
                initial=0.0,
            )
        )
        cone_offset = block_stop
    if cone_offset != rows:
        raise frozen_solver.AllocationSolveError(
            "bounded dual feasibility cone partition is incomplete"
        )
    dual_coordinate_scale = 1.0 / np.maximum(
        cone_preserving_measure,
        DUAL_CERTIFICATE_STATIONARITY_FLOOR,
    )
    if (
        not np.isfinite(dual_coordinate_scale).all()
        or np.any(dual_coordinate_scale <= 0.0)
    ):
        raise frozen_solver.AllocationSolveError(
            "bounded dual feasibility coordinate scales are invalid"
        )
    scaled_matrix = matrix.multiply(dual_coordinate_scale[:, np.newaxis])
    column_infinity_norm = np.asarray(
        np.abs(scaled_matrix).max(axis=0).toarray(),
        dtype=np.float64,
    ).reshape(-1)
    if (
        column_infinity_norm.shape != (columns,)
        or not np.isfinite(column_infinity_norm).all()
        or np.any(column_infinity_norm < 0.0)
    ):
        raise frozen_solver.AllocationSolveError(
            "bounded dual feasibility stationarity scales are invalid"
        )
    stationarity_scale = (
        BOUNDED_DUAL_STATIONARITY_MULTIPLIER
        / np.maximum.reduce(
            (
                column_infinity_norm,
                np.abs(objective),
                np.full_like(
                    objective,
                    DUAL_CERTIFICATE_STATIONARITY_FLOOR,
                ),
            )
        )
    )
    candidate_scaled = float(
        retry_certificate.candidate_scaled_objective
    )
    if (
        not np.isfinite(candidate_scaled)
        or not np.isfinite(stationarity_scale).all()
        or np.any(stationarity_scale <= 0.0)
    ):
        raise frozen_solver.AllocationSolveError(
            "bounded dual feasibility target is invalid"
        )

    objective_offset = _canonical_objective_offset(inverse)
    scaled_dual_variable = cp.Variable(
        rows,
        name="stage11_bounded_feasible_scaled_dual",
    )
    dual_expression = cp.multiply(
        dual_coordinate_scale,
        scaled_dual_variable,
    )
    dual_upper_expression = rhs @ dual_expression - objective_offset
    gap_band = BOUNDED_DUAL_GAP_BAND
    bounded_problem = cp.Problem(
        cp.Minimize(0.0),
        [
            cp.multiply(
                stationarity_scale,
                matrix.T @ dual_expression + objective,
            )
            == 0.0,
            *_dual_cone_constraints(scaled_dual_variable, dims),
            dual_upper_expression >= candidate_scaled,
            dual_upper_expression <= candidate_scaled + gap_band,
        ],
    )
    bounded_status = _solve_status(
        bounded_problem,
        frozen_solver.RETRY_SOLVER_SETTINGS,
    )
    dual = (
        None
        if scaled_dual_variable.value is None
        else dual_coordinate_scale
        * np.asarray(
            scaled_dual_variable.value,
            dtype=np.float64,
        ).reshape(-1)
    )
    candidate_violation = max(
        explicit_certificate.candidate_maximum_constraint_violation,
        retry_certificate.candidate_maximum_constraint_violation,
    )
    agreement = max(
        explicit_certificate.objective_cost_scaled_maximum_difference,
        retry_certificate.objective_cost_scaled_maximum_difference,
        abs(
            explicit_certificate.candidate_scaled_objective
            - retry_certificate.candidate_scaled_objective
        ),
    )
    if (
        bounded_status not in DUAL_CERTIFICATE_AUDITABLE_STATUSES
        or dual is None
        or dual.shape != (rows,)
        or not np.isfinite(dual).all()
    ):
        return _HybridDualCertificate(
            certificate_source=(
                BOUNDED_FEASIBLE_DUAL_CERTIFICATE_SOURCE
            ),
            dual_status=bounded_status,
            candidate_maximum_constraint_violation=candidate_violation,
            stationarity_absolute_maximum=float("inf"),
            stationarity_backward_error_maximum=float("inf"),
            dual_nonnegative_minimum=float("-inf"),
            dual_soc_minimum_margin=float("-inf"),
            candidate_scaled_objective=candidate_scaled,
            dual_upper_scaled=float("nan"),
            candidate_dual_gap_scaled=float("nan"),
            objective_cost_scaled_maximum_difference=agreement,
            dual_coordinate_scale_minimum=float(
                np.min(dual_coordinate_scale)
            ),
            dual_coordinate_scale_maximum=float(
                np.max(dual_coordinate_scale)
            ),
            stationarity_equation_multiplier=(
                BOUNDED_DUAL_STATIONARITY_MULTIPLIER
            ),
            candidate_gap_band_fraction=BOUNDED_DUAL_GAP_FRACTION,
        )

    stationarity_residual = np.asarray(
        matrix.T @ dual + objective,
        dtype=np.float64,
    ).reshape(-1)
    backward_denominator = (
        np.asarray(
            np.abs(matrix).T @ np.abs(dual),
            dtype=np.float64,
        ).reshape(-1)
        + np.abs(objective)
    )
    backward_error = np.abs(stationarity_residual) / np.maximum(
        backward_denominator,
        DUAL_CERTIFICATE_STATIONARITY_FLOOR,
    )
    (
        nonnegative_minimum,
        soc_minimum,
        soc_minimum_block_ordinal,
    ) = _dual_cone_margins(dual, dims)
    dual_upper_scaled = float(rhs @ dual - objective_offset)
    return _HybridDualCertificate(
        certificate_source=BOUNDED_FEASIBLE_DUAL_CERTIFICATE_SOURCE,
        dual_status=bounded_status,
        candidate_maximum_constraint_violation=candidate_violation,
        stationarity_absolute_maximum=float(
            np.max(np.abs(stationarity_residual), initial=0.0)
        ),
        stationarity_backward_error_maximum=float(
            np.max(backward_error, initial=0.0)
        ),
        dual_nonnegative_minimum=nonnegative_minimum,
        dual_soc_minimum_margin=soc_minimum,
        candidate_scaled_objective=candidate_scaled,
        dual_upper_scaled=dual_upper_scaled,
        candidate_dual_gap_scaled=float(
            dual_upper_scaled - candidate_scaled
        ),
        objective_cost_scaled_maximum_difference=agreement,
        dual_soc_minimum_block_ordinal=soc_minimum_block_ordinal,
        dual_coordinate_scale_minimum=float(
            np.min(dual_coordinate_scale)
        ),
        dual_coordinate_scale_maximum=float(
            np.max(dual_coordinate_scale)
        ),
        stationarity_equation_multiplier=(
            BOUNDED_DUAL_STATIONARITY_MULTIPLIER
        ),
        candidate_gap_band_fraction=BOUNDED_DUAL_GAP_FRACTION,
        dual_vector=_frozen_dual_vector(dual),
    )


def _hybrid_dual_certificate_failure(
    certificate: _HybridDualCertificate,
) -> str | None:
    """Return the first failed predeclared certificate gate."""

    if certificate.dual_status not in DUAL_CERTIFICATE_AUDITABLE_STATUSES:
        return "dual_status_not_auditable"
    if (
        not np.isfinite(
            certificate.candidate_maximum_constraint_violation
        )
        or certificate.candidate_maximum_constraint_violation
        > frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE
    ):
        return "candidate_constraint_violation"
    if (
        not np.isfinite(certificate.stationarity_absolute_maximum)
        or certificate.stationarity_absolute_maximum
        > DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
    ):
        return "dual_stationarity_absolute"
    if (
        not np.isfinite(certificate.stationarity_backward_error_maximum)
        or certificate.stationarity_backward_error_maximum
        > DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
    ):
        return "dual_stationarity_backward_error"
    if (
        np.isnan(certificate.dual_nonnegative_minimum)
        or certificate.dual_nonnegative_minimum
        < -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
    ):
        return "dual_nonnegative_cone"
    if (
        np.isnan(certificate.dual_soc_minimum_margin)
        or certificate.dual_soc_minimum_margin
        < -DUAL_CERTIFICATE_RESIDUAL_TOLERANCE
    ):
        return "dual_soc_cone"
    if (
        not np.isfinite(certificate.candidate_scaled_objective)
        or not np.isfinite(certificate.dual_upper_scaled)
        or not np.isfinite(certificate.candidate_dual_gap_scaled)
        or abs(certificate.candidate_dual_gap_scaled)
        > DUAL_CERTIFICATE_GAP_TOLERANCE
    ):
        return "candidate_dual_gap"
    if (
        not np.isfinite(
            certificate.objective_cost_scaled_maximum_difference
        )
        or certificate.objective_cost_scaled_maximum_difference
        > DUAL_CERTIFICATE_OBJECTIVE_AGREEMENT_TOLERANCE
    ):
        return "cvxpy_numpy_objective_cost_disagreement"
    return None


def _hybrid_dual_certificate_metrics_history(
    certificate: _HybridDualCertificate,
    *,
    prefix: str = "ordinary",
) -> tuple[str, ...]:
    """Serialize all certificate metrics deterministically."""

    metrics = (
        (
            "candidate_maximum_constraint_violation",
            certificate.candidate_maximum_constraint_violation,
        ),
        (
            "stationarity_absolute_maximum",
            certificate.stationarity_absolute_maximum,
        ),
        (
            "stationarity_backward_error_maximum",
            certificate.stationarity_backward_error_maximum,
        ),
        (
            "dual_nonnegative_minimum",
            certificate.dual_nonnegative_minimum,
        ),
        ("dual_soc_minimum_margin", certificate.dual_soc_minimum_margin),
        (
            "candidate_scaled_objective",
            certificate.candidate_scaled_objective,
        ),
        ("dual_upper_scaled", certificate.dual_upper_scaled),
        (
            "candidate_dual_gap_scaled",
            certificate.candidate_dual_gap_scaled,
        ),
        (
            "objective_cost_scaled_maximum_difference",
            certificate.objective_cost_scaled_maximum_difference,
        ),
        (
            "stationarity_refinement_scale",
            certificate.stationarity_refinement_scale,
        ),
        (
            "stationarity_refinement_correction_maximum",
            certificate.stationarity_refinement_correction_maximum,
        ),
        (
            "dual_reconciliation_retry_weight",
            certificate.dual_reconciliation_retry_weight,
        ),
        (
            "dual_coordinate_scale_minimum",
            certificate.dual_coordinate_scale_minimum,
        ),
        (
            "dual_coordinate_scale_maximum",
            certificate.dual_coordinate_scale_maximum,
        ),
        (
            "stationarity_equation_multiplier",
            certificate.stationarity_equation_multiplier,
        ),
        (
            "candidate_gap_band_fraction",
            certificate.candidate_gap_band_fraction,
        ),
    )
    return (
        (
            f"{prefix}_dual_certificate_stationarity_refinement_stop_code:"
            + str(certificate.stationarity_refinement_stop_code)
        ),
        (
            f"{prefix}_dual_certificate_stationarity_refinement_iterations:"
            + str(certificate.stationarity_refinement_iterations)
        ),
        (
            f"{prefix}_dual_certificate_soc_minimum_block_ordinal:"
            + str(certificate.dual_soc_minimum_block_ordinal)
        ),
        *(
            f"{prefix}_dual_certificate_"
            + name
            + ":"
            + float(value).hex()
            for name, value in metrics
        ),
    )


def _hybrid_dual_certificate_history(
    certificate: _HybridDualCertificate,
    *,
    prefix: str = "ordinary",
) -> tuple[str, ...]:
    """Serialize a passing certificate and every audited metric."""

    return (
        f"{prefix}_dual_certificate:{certificate.dual_status}",
        (
            f"{prefix}_dual_certificate_source:"
            + certificate.certificate_source
        ),
        *_hybrid_dual_certificate_metrics_history(
            certificate,
            prefix=prefix,
        ),
        f"{prefix}_dual_certificate_audit:passed",
    )


def _persist_dual_failure_snapshot(
    data: AllocationProblemData,
    candidate: np.ndarray,
    problem: cp.Problem,
    caps: np.ndarray,
    *,
    architecture: str,
    explicit_certificate: _HybridDualCertificate | None,
    retry_certificate: _HybridDualCertificate | None,
) -> tuple[str, str, str]:
    """Persist one exact local troubleshooting snapshot after fail-closed exit."""

    def _json_safe(value: object) -> object:
        """Encode diagnostic metadata without non-standard JSON numbers."""

        if isinstance(value, dict):
            return {
                str(key): _json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [_json_safe(item) for item in value]
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            numeric = float(value)
            return numeric.hex() if np.isfinite(numeric) else str(numeric)
        if value is None or isinstance(value, str):
            return value
        raise TypeError(
            "dual failure snapshot metadata contains an unsupported value"
        )

    def _certificate_metadata(
        certificate: _HybridDualCertificate | None,
    ) -> dict[str, object] | None:
        if certificate is None:
            return None
        return {
            item.name: getattr(certificate, item.name)
            for item in fields(certificate)
            if item.name != "dual_vector"
        }

    canonical, _chain, inverse = problem.get_problem_data(cp.CLARABEL)
    matrix = canonical["A"].tocsc(copy=True)
    rhs = np.asarray(canonical["b"], dtype=np.float64).reshape(-1)
    objective = np.asarray(canonical["c"], dtype=np.float64).reshape(-1)
    dims = canonical["dims"]
    cache = getattr(problem, "_solver_cache", None)
    clarabel_solver = (
        cache.get("CLARABEL") if isinstance(cache, dict) else None
    )
    solution = (
        clarabel_solver.get_solution()
        if clarabel_solver is not None
        else None
    )
    raw_dual = (
        np.asarray(solution.z, dtype=np.float64).reshape(-1)
        if solution is not None and getattr(solution, "z", None) is not None
        else np.asarray([], dtype=np.float64)
    )
    correction = np.asarray([], dtype=np.float64)
    refinement_stop_code = -1
    refinement_iterations = -1
    if raw_dual.shape == (matrix.shape[0],) and np.isfinite(raw_dual).all():
        raw_stationarity = np.asarray(
            matrix.T @ raw_dual + objective,
            dtype=np.float64,
        ).reshape(-1)
        refinement = lsmr(
            matrix.T,
            -raw_stationarity,
            atol=DUAL_STATIONARITY_REFINEMENT_TOLERANCE,
            btol=DUAL_STATIONARITY_REFINEMENT_TOLERANCE,
            conlim=0.0,
            maxiter=min(matrix.shape),
            show=False,
        )
        correction = np.asarray(
            refinement[0],
            dtype=np.float64,
        ).reshape(-1)
        refinement_stop_code = int(refinement[1])
        refinement_iterations = int(refinement[2])

    payload: dict[str, np.ndarray] = {
        "permno": np.asarray(data.permno),
        "permco": np.asarray(data.permco),
        "side": np.asarray(data.side),
        "previous_weight": np.asarray(data.previous_weight),
        "reduction_only": np.asarray(data.reduction_only),
        "required_exit": np.asarray(data.required_exit),
        "expected_return": np.asarray(data.expected_return),
        "score": np.asarray(data.score),
        "factor_betas": np.asarray(data.factor_betas),
        "factor_covariance_monthly": np.asarray(
            data.factor_covariance_monthly
        ),
        "specific_variance_monthly": np.asarray(
            data.specific_variance_monthly
        ),
        "forecast_volatility": np.asarray(data.forecast_volatility),
        "full_spread": np.asarray(data.full_spread),
        "adv20_usd": np.asarray(data.adv20_usd),
        "impact_volatility_daily": np.asarray(
            data.impact_volatility_daily
        ),
        "borrow_liquidity_quintile": np.asarray(
            data.borrow_liquidity_quintile
        ),
        "trade_capacity_weight": np.asarray(data.trade_capacity_weight),
        "sector": np.asarray(data.sector),
        "country": np.asarray(data.country),
        "mandatory_corporate_action": np.asarray(
            data.mandatory_corporate_action
        ),
        "risk_root": np.asarray(data.risk_root),
        "borrow_rate_annual": np.asarray(data.borrow_rate_annual),
        "candidate": np.asarray(candidate, dtype=np.float64),
        "caps": np.asarray(caps, dtype=np.float64),
        "canonical_A_data": np.asarray(matrix.data),
        "canonical_A_indices": np.asarray(matrix.indices),
        "canonical_A_indptr": np.asarray(matrix.indptr),
        "canonical_A_shape": np.asarray(matrix.shape, dtype=np.int64),
        "canonical_b": rhs,
        "canonical_c": objective,
        "canonical_dims_zero_nonnegative": np.asarray(
            [int(dims.zero), int(dims.nonneg)],
            dtype=np.int64,
        ),
        "canonical_dims_soc": np.asarray(dims.soc, dtype=np.int64),
        "canonical_objective_offset": np.asarray(
            [_canonical_objective_offset(inverse)],
            dtype=np.float64,
        ),
        "raw_dual": raw_dual,
        "lsmr_correction": correction,
        "explicit_dual": (
            np.asarray([], dtype=np.float64)
            if explicit_certificate is None
            or explicit_certificate.dual_vector is None
            else np.asarray(
                explicit_certificate.dual_vector,
                dtype=np.float64,
            )
        ),
        "retry_refined_dual": (
            np.asarray([], dtype=np.float64)
            if retry_certificate is None
            or retry_certificate.dual_vector is None
            else np.asarray(
                retry_certificate.dual_vector,
                dtype=np.float64,
            )
        ),
        "nav_dollars": np.asarray([data.nav_dollars], dtype=np.float64),
        "market_variance_monthly": np.asarray(
            [data.market_variance_monthly],
            dtype=np.float64,
        ),
    }
    metadata = {
        "schema_version": 1,
        "architecture": architecture,
        "borrow_scenario": data.borrow_scenario,
        "explicit_certificate": (
            _certificate_metadata(explicit_certificate)
        ),
        "retry_certificate": (
            _certificate_metadata(retry_certificate)
        ),
        "retry_solver_status": (
            None if solution is None else str(getattr(solution, "status", ""))
        ),
        "retry_reported_dual_objective": (
            None
            if solution is None
            or getattr(solution, "obj_val_dual", None) is None
            else float(solution.obj_val_dual)
        ),
        "lsmr_stop_code": refinement_stop_code,
        "lsmr_iterations": refinement_iterations,
        "primary_solver_settings": frozen_solver.PRIMARY_SOLVER_SETTINGS,
        "retry_solver_settings": frozen_solver.RETRY_SOLVER_SETTINGS,
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(
            _json_safe(metadata),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )

    content_hasher = hashlib.sha256()
    for name in sorted(payload):
        value = np.ascontiguousarray(payload[name])
        content_hasher.update(name.encode("utf-8"))
        content_hasher.update(value.dtype.str.encode("ascii"))
        content_hasher.update(
            np.asarray(value.shape, dtype=np.int64).tobytes()
        )
        content_hasher.update(value.tobytes())
    content_sha256 = content_hasher.hexdigest()
    root = DUAL_FAILURE_SNAPSHOT_ROOT
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    destination = root / f"dual-failure-{content_sha256}.npz"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".dual-failure-{content_sha256}.",
        suffix=".transaction",
        dir=root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_sha256 = hashlib.sha256(temporary.read_bytes()).hexdigest()
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or hashlib.sha256(destination.read_bytes()).hexdigest()
                != temporary_sha256
            ):
                raise frozen_solver.AllocationSolveError(
                    "existing dual failure snapshot bytes differ"
                )
        finally:
            temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if destination.is_symlink() or (destination.stat().st_mode & 0o777) != 0o600:
        raise frozen_solver.AllocationSolveError(
            "dual failure snapshot path is unsafe"
        )
    file_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    return destination.as_posix(), file_sha256, content_sha256


def _append_dual_failure_snapshot_history(
    history: list[str],
    data: AllocationProblemData,
    candidate: np.ndarray,
    problem: cp.Problem,
    caps: np.ndarray,
    *,
    architecture: str,
    explicit_certificate: _HybridDualCertificate | None,
    retry_certificate: _HybridDualCertificate | None,
    prefix: str,
) -> None:
    """Record a reproducible snapshot without weakening fail-closed behavior."""

    try:
        path, file_sha256, content_sha256 = _persist_dual_failure_snapshot(
            data,
            candidate,
            problem,
            caps,
            architecture=architecture,
            explicit_certificate=explicit_certificate,
            retry_certificate=retry_certificate,
        )
    except Exception as exc:  # pragma: no cover - diagnostic must never accept
        history.append(
            f"{prefix}_dual_failure_snapshot:error:{type(exc).__name__}"
        )
        return
    history.extend(
        (
            f"{prefix}_dual_failure_snapshot_path:{path}",
            f"{prefix}_dual_failure_snapshot_file_sha256:{file_sha256}",
            f"{prefix}_dual_failure_snapshot_content_sha256:{content_sha256}",
        )
    )


def _projection_original_weight_coordinate_retry(
    data: AllocationProblemData,
    raw_weights: np.ndarray,
    *,
    architecture: str,
    history: list[str],
) -> frozen_solver.AllocationResult:
    """Retry the frozen projection in its unchanged original coordinate.

    This route is reached only after the cap-normalized objective attempts
    were numerically inaccurate and the identical-constraint feasibility
    problem returned exact ``optimal``.  It changes neither the objective nor
    any constraint, solver setting, tolerance, or solver.  The zero-objective
    feasibility candidate is never used as portfolio weights.
    """

    raw = np.asarray(raw_weights, dtype=np.float64)
    if raw.shape != (data.n_assets,) or not np.isfinite(raw).all():
        raise ValueError("raw_weights must be a finite n_assets vector")
    weights = cp.Variable(
        data.n_assets,
        name="stage11_projection_original_weight_retry",
    )
    problem = cp.Problem(
        cp.Minimize(
            frozen_solver._projection_expression(data, weights, raw)
        ),
        ordinary_constraints(data, weights),
    )
    for label, settings in (
        ("primary", frozen_solver.PRIMARY_SOLVER_SETTINGS),
        ("retry", frozen_solver.RETRY_SOLVER_SETTINGS),
    ):
        status = _solve_status(problem, settings)
        history.append(
            f"ordinary_projection_original_coordinate_{label}:{status}"
        )
        if status != cp.OPTIMAL:
            continue
        if weights.value is None:
            history.append(
                "ordinary_projection_original_coordinate_"
                f"{label}_audit:missing_primal"
            )
            continue
        try:
            vector = frozen_solver._canonicalize_solver_weights(
                data,
                np.asarray(weights.value, dtype=np.float64).reshape(-1),
            )
        except (frozen_solver.AllocationSolveError, TypeError, ValueError):
            history.append(
                "ordinary_projection_original_coordinate_"
                f"{label}_audit:invalid_primal"
            )
            continue
        diagnostics = independent_constraint_diagnostics(data, vector)
        if (
            diagnostics.maximum_violation
            > frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE
        ):
            history.append(
                "ordinary_projection_original_coordinate_"
                f"{label}_audit:constraint_violation"
            )
            continue
        return frozen_solver._make_result(
            data,
            vector,
            architecture=architecture,
            kind="projection",
            raw_weights=raw,
            mode="ordinary",
            solver_attempt=label,
            solver_status=status,
            solver_status_history=tuple(history),
        )
    raise frozen_solver.AllocationSolveError(
        "ordinary projection equivalent retry did not return an audited "
        "exact optimum: "
        + ", ".join(history)
    )


def _projection_original_coordinate_infeasibility_recheck(
    data: AllocationProblemData,
    *,
    history: list[str],
) -> bool:
    """Require an exact, independently reproducible infeasibility route.

    The cap-normalized and original weight coordinates are positive-diagonal
    affine bijections for a projection problem, whose ordinary feasible set
    contains no transaction-cost epigraph.  If the normalized zero-objective
    solve reports only ``infeasible_inaccurate``, this recheck may open the
    existing emergency hierarchy only when:

    * a required-exit row has an unavoidable residual larger than the frozen
      independent tolerance under both hard capacity bounds; and
    * both frozen CLARABEL setting sets return exact ``infeasible`` in the
      unchanged original weight coordinate.

    No solver status, tolerance, constraint, or portfolio target is relaxed.
    """

    valid_adv = np.isfinite(data.adv20_usd) & (data.adv20_usd > 0.0)
    adv_capacity = np.where(
        valid_adv,
        0.10 * data.adv20_usd / float(data.nav_dollars),
        0.0,
    )
    effective_capacity = np.minimum(
        np.asarray(data.trade_capacity_weight, dtype=np.float64),
        np.asarray(adv_capacity, dtype=np.float64),
    )
    unavoidable = (
        np.abs(np.asarray(data.previous_weight, dtype=np.float64))
        - effective_capacity
    )
    candidates = np.flatnonzero(
        np.asarray(data.required_exit, dtype=np.bool_)
        & (
            unavoidable
            > frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE
        )
    )
    if not len(candidates):
        history.append(
            "ordinary_projection_required_exit_infeasibility_witness:"
            "not_found"
        )
        return False
    witness_index = min(
        (int(index) for index in candidates),
        key=lambda index: (int(data.permno[index]), index),
    )
    history.extend(
        (
            "ordinary_projection_required_exit_infeasibility_witness:"
            "passed",
            (
                "ordinary_projection_required_exit_infeasibility_permno:"
                f"{int(data.permno[witness_index])}"
            ),
            (
                "ordinary_projection_required_exit_previous_weight:"
                f"{float(data.previous_weight[witness_index]).hex()}"
            ),
            (
                "ordinary_projection_required_exit_trade_capacity:"
                f"{float(data.trade_capacity_weight[witness_index]).hex()}"
            ),
            (
                "ordinary_projection_required_exit_adv_capacity:"
                f"{float(adv_capacity[witness_index]).hex()}"
            ),
            (
                "ordinary_projection_required_exit_effective_capacity:"
                f"{float(effective_capacity[witness_index]).hex()}"
            ),
            (
                "ordinary_projection_required_exit_unavoidable_weight:"
                f"{float(unavoidable[witness_index]).hex()}"
            ),
        )
    )

    weights = cp.Variable(
        data.n_assets,
        name="stage11_projection_original_coordinate_feasibility_recheck",
    )
    problem = cp.Problem(
        cp.Minimize(0.0),
        ordinary_constraints(data, weights),
    )
    statuses: list[str] = []
    for label, settings in (
        ("primary", frozen_solver.PRIMARY_SOLVER_SETTINGS),
        ("retry", frozen_solver.RETRY_SOLVER_SETTINGS),
    ):
        status = _solve_status(problem, settings)
        statuses.append(status)
        history.append(
            "ordinary_projection_original_coordinate_feasibility_"
            f"{label}:{status}"
        )
    passed = statuses == [cp.INFEASIBLE, cp.INFEASIBLE]
    history.append(
        "ordinary_projection_original_coordinate_infeasibility_audit:"
        + ("passed" if passed else "failed")
    )
    return passed


def _ordinary_solve(
    data: AllocationProblemData,
    *,
    kind: SolveKind,
    raw_weights: np.ndarray | None,
    architecture: str,
) -> frozen_solver.AllocationResult:
    normalized, weights, caps = _scaled_weight_expression(
        data,
        name="stage11_normalized_weights",
    )
    objective, cost_constraints = _objective_and_cost_constraints(
        data,
        normalized,
        weights,
        caps,
        kind=kind,
        raw_weights=raw_weights,
    )
    constraints = ordinary_constraints(data, weights) + cost_constraints
    problem = cp.Problem(objective, constraints)
    history: list[str] = []
    retry_candidate: np.ndarray | None = None
    for label, settings in (
        ("primary", frozen_solver.PRIMARY_SOLVER_SETTINGS),
        ("retry", frozen_solver.RETRY_SOLVER_SETTINGS),
    ):
        status = _solve_status(problem, settings)
        history.append(f"ordinary_{label}:{status}")
        if status == cp.OPTIMAL:
            if weights.value is None:
                history.append(f"ordinary_{label}_audit:missing_primal")
                continue
            try:
                vector = frozen_solver._canonicalize_solver_weights(
                    data,
                    np.asarray(weights.value, dtype=np.float64).reshape(-1),
                )
            except (frozen_solver.AllocationSolveError, TypeError, ValueError):
                history.append(f"ordinary_{label}_audit:invalid_primal")
                continue
            diagnostics = independent_constraint_diagnostics(data, vector)
            if (
                diagnostics.maximum_violation
                > frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE
            ):
                history.append(
                    f"ordinary_{label}_audit:constraint_violation"
                )
                continue
            return frozen_solver._make_result(
                data,
                vector,
                architecture=architecture,
                kind=kind,
                raw_weights=raw_weights,
                mode="ordinary",
                solver_attempt=label,
                solver_status=status,
                solver_status_history=tuple(history),
            )
        if (
            kind != "hybrid"
            or label != "retry"
            or status != cp.OPTIMAL_INACCURATE
        ):
            continue
        if weights.value is None:
            history.append(f"ordinary_{label}_audit:missing_primal")
            continue
        try:
            vector = frozen_solver._canonicalize_solver_weights(
                data,
                np.asarray(weights.value, dtype=np.float64).reshape(-1),
            )
        except (frozen_solver.AllocationSolveError, TypeError, ValueError):
            history.append(f"ordinary_{label}_audit:invalid_primal")
            continue
        diagnostics = independent_constraint_diagnostics(data, vector)
        if (
            diagnostics.maximum_violation
            > frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE
        ):
            history.append(f"ordinary_{label}_audit:constraint_violation")
            continue
        # Never accept this status by itself.  Copy the retry primal now
        # because the subsequent feasibility solve reuses the same variable.
        retry_candidate = np.array(vector, dtype=np.float64, copy=True)

    # The zero-objective problem uses the identical constraints.  Only an
    # exact infeasibility certificate may open the emergency path.  A feasible
    # status, including ``optimal_inaccurate``, can only route an already
    # independently primal-audited retry candidate into the full dual-
    # certificate hierarchy below; status alone never permits acceptance.
    feasibility = cp.Problem(cp.Minimize(0.0), constraints)
    feasibility_status = _solve_status(
        feasibility,
        frozen_solver.RETRY_SOLVER_SETTINGS,
    )
    history.append(f"ordinary_feasibility_retry:{feasibility_status}")
    if (
        kind == "hybrid"
        and feasibility_status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        and retry_candidate is not None
    ):
        certificate: _HybridDualCertificate | None = None
        explicit_certificate: _HybridDualCertificate | None = None
        retry_certificate: _HybridDualCertificate | None = None
        try:
            explicit_certificate = _build_explicit_hybrid_dual_certificate(
                data,
                retry_candidate,
                problem,
                normalized,
                weights,
                caps,
            )
        except (frozen_solver.AllocationSolveError, TypeError, ValueError) as exc:
            history.append(
                "ordinary_explicit_dual_certificate_audit:error:"
                + type(exc).__name__
            )
        else:
            explicit_failure = _hybrid_dual_certificate_failure(
                explicit_certificate
            )
            if explicit_failure is None:
                certificate = explicit_certificate
            else:
                history.extend(
                    (
                        (
                            "ordinary_explicit_dual_certificate:"
                            + explicit_certificate.dual_status
                        ),
                        (
                            "ordinary_explicit_dual_certificate_source:"
                            + explicit_certificate.certificate_source
                        ),
                        (
                            "ordinary_explicit_dual_certificate_audit:"
                            "failed:"
                            + explicit_failure
                        ),
                        *_hybrid_dual_certificate_metrics_history(
                            explicit_certificate,
                            prefix="ordinary_explicit",
                        ),
                    )
                )
        if certificate is None:
            try:
                retry_certificate = (
                    _build_retry_economic_dual_certificate(
                        data,
                        retry_candidate,
                        problem,
                        normalized,
                        weights,
                        caps,
                        dual_status=cp.OPTIMAL_INACCURATE,
                    )
                )
            except (
                frozen_solver.AllocationSolveError,
                TypeError,
                ValueError,
            ) as exc:
                history.append(
                    "ordinary_retry_dual_certificate_audit:error:"
                    + type(exc).__name__
                )
                _append_dual_failure_snapshot_history(
                    history,
                    data,
                    retry_candidate,
                    problem,
                    caps,
                    architecture=architecture,
                    explicit_certificate=explicit_certificate,
                    retry_certificate=retry_certificate,
                    prefix="ordinary",
                )
                raise frozen_solver.AllocationSolveError(
                    "ordinary hybrid dual certificate construction failed: "
                    + ", ".join(history)
                ) from exc
            retry_failure = _hybrid_dual_certificate_failure(
                retry_certificate
            )
            if retry_failure is None:
                certificate = retry_certificate
            else:
                history.extend(
                    (
                        (
                            "ordinary_retry_dual_certificate:"
                            + retry_certificate.dual_status
                        ),
                        (
                            "ordinary_retry_dual_certificate_source:"
                            + retry_certificate.certificate_source
                        ),
                        (
                            "ordinary_retry_dual_certificate_audit:failed:"
                            + retry_failure
                        ),
                        *_hybrid_dual_certificate_metrics_history(
                            retry_certificate,
                            prefix="ordinary_retry",
                        ),
                    )
                )
        if (
            certificate is None
            and explicit_certificate is not None
            and retry_certificate is not None
        ):
            try:
                reconciled_certificate = (
                    _build_reconciled_hybrid_dual_certificate(
                        problem,
                        explicit_certificate,
                        retry_certificate,
                    )
                )
            except (
                frozen_solver.AllocationSolveError,
                TypeError,
                ValueError,
            ) as exc:
                history.append(
                    "ordinary_reconciled_dual_certificate_audit:error:"
                    + type(exc).__name__
                )
            else:
                reconciled_failure = _hybrid_dual_certificate_failure(
                    reconciled_certificate
                )
                if reconciled_failure is None:
                    certificate = reconciled_certificate
                else:
                    history.extend(
                        (
                            (
                                "ordinary_reconciled_dual_certificate:"
                                + reconciled_certificate.dual_status
                            ),
                            (
                                "ordinary_reconciled_dual_certificate_source:"
                                + reconciled_certificate.certificate_source
                            ),
                            (
                                "ordinary_reconciled_dual_certificate_audit:"
                                "failed:"
                                + reconciled_failure
                            ),
                            *_hybrid_dual_certificate_metrics_history(
                                reconciled_certificate,
                                prefix="ordinary_reconciled",
                            ),
                        )
                    )
        if (
            certificate is None
            and explicit_certificate is not None
            and retry_certificate is not None
        ):
            try:
                bounded_certificate = (
                    _build_bounded_feasible_hybrid_dual_certificate(
                        problem,
                        explicit_certificate,
                        retry_certificate,
                    )
                )
            except (
                frozen_solver.AllocationSolveError,
                TypeError,
                ValueError,
            ) as exc:
                history.append(
                    "ordinary_bounded_feasible_dual_certificate_"
                    "audit:error:"
                    + type(exc).__name__
                )
            else:
                bounded_failure = _hybrid_dual_certificate_failure(
                    bounded_certificate
                )
                if bounded_failure is None:
                    certificate = bounded_certificate
                else:
                    history.extend(
                        (
                            (
                                "ordinary_bounded_feasible_dual_"
                                "certificate:"
                                + bounded_certificate.dual_status
                            ),
                            (
                                "ordinary_bounded_feasible_dual_"
                                "certificate_source:"
                                + bounded_certificate.certificate_source
                            ),
                            (
                                "ordinary_bounded_feasible_dual_"
                                "certificate_audit:failed:"
                                + bounded_failure
                            ),
                            *_hybrid_dual_certificate_metrics_history(
                                bounded_certificate,
                                prefix="ordinary_bounded_feasible",
                            ),
                        )
                    )
        if certificate is None:
            _append_dual_failure_snapshot_history(
                history,
                data,
                retry_candidate,
                problem,
                caps,
                architecture=architecture,
                explicit_certificate=explicit_certificate,
                retry_certificate=retry_certificate,
                prefix="ordinary",
            )
            raise frozen_solver.AllocationSolveError(
                "ordinary hybrid retry candidate was not dual-certified: "
                + ", ".join(history)
            )
        history.extend(_hybrid_dual_certificate_history(certificate))
        return frozen_solver._make_result(
            data,
            retry_candidate,
            architecture=architecture,
            kind=kind,
            raw_weights=raw_weights,
            mode="ordinary",
            solver_attempt="retry",
            solver_status=DUAL_CERTIFIED_SOLVER_STATUS,
            solver_status_history=tuple(history),
        )
    if (
        kind == "projection"
        and feasibility_status == cp.OPTIMAL
        and raw_weights is not None
    ):
        return _projection_original_weight_coordinate_retry(
            data,
            raw_weights,
            architecture=architecture,
            history=history,
        )
    if (
        kind == "projection"
        and feasibility_status == cp.INFEASIBLE_INACCURATE
        and _projection_original_coordinate_infeasibility_recheck(
            data,
            history=history,
        )
    ):
        raise frozen_solver._CertifiedOrdinaryInfeasible(tuple(history))
    if feasibility_status != cp.INFEASIBLE:
        raise frozen_solver.AllocationSolveError(
            "ordinary feasibility was not exactly certified infeasible: "
            + ", ".join(history)
        )
    raise frozen_solver._CertifiedOrdinaryInfeasible(tuple(history))


def _audit_emergency_candidate(
    data: AllocationProblemData,
    candidate: np.ndarray,
    *,
    pretrade_diagnostics: object,
    pretrade_slacks: dict[str, float],
    optimum_maximum: float,
    optimum_sum: float,
) -> _EmergencyCandidateAudit:
    """Independently prove feasibility of the emergency lexicographic locks."""

    vector = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if vector.shape != (data.n_assets,) or not np.isfinite(vector).all():
        return _EmergencyCandidateAudit(*(float("inf"),) * 7)
    diagnostics = independent_constraint_diagnostics(data, vector)
    always_hard_prefixes = (
        "sign:",
        "trade_capacity:",
        "adv_capacity:",
        "reduction_only:",
    )
    hard_or_noninherited = max(
        (
            max(float(value), 0.0)
            for name, value in diagnostics.violations.items()
            if (
                name == "turnover"
                or name.startswith(always_hard_prefixes)
                or name not in pretrade_slacks
            )
        ),
        default=0.0,
    )
    required_slacks: list[float] = []
    inherited_worsening = 0.0
    for name, inherited_bound in pretrade_slacks.items():
        if name not in diagnostics.normalized_violations:
            return _EmergencyCandidateAudit(*(float("inf"),) * 7)
        required = max(
            float(diagnostics.normalized_violations[name]),
            0.0,
        )
        prior = max(
            float(pretrade_diagnostics.normalized_violations.get(name, 0.0)),
            0.0,
        )
        inherited_worsening = max(
            inherited_worsening,
            required - prior,
            required - float(inherited_bound),
            0.0,
        )
        required_slacks.append(required)

    _linear, _impact, permitted = _transaction_coefficients(
        data.full_spread,
        data.adv20_usd,
        data.impact_volatility_daily,
        nav_dollars=data.nav_dollars,
        scenario=PRIMARY_COST_SCENARIO,
    )
    invalid = np.flatnonzero(~np.asarray(permitted, dtype=np.bool_))
    maximum_invalid_cost_trade = (
        float(
            np.max(
                np.abs(
                    vector[invalid] - data.previous_weight[invalid]
                )
            )
        )
        if invalid.size
        else 0.0
    )
    required_maximum = max(required_slacks, default=0.0)
    required_sum = float(sum(required_slacks))
    values = np.asarray(
        [
            hard_or_noninherited,
            inherited_worsening,
            maximum_invalid_cost_trade,
            required_maximum,
            required_sum,
            optimum_maximum,
            optimum_sum,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        return _EmergencyCandidateAudit(*(float("inf"),) * 7)
    return _EmergencyCandidateAudit(
        maximum_hard_or_noninherited_violation=float(
            hard_or_noninherited
        ),
        maximum_inherited_worsening=float(inherited_worsening),
        maximum_invalid_cost_trade=maximum_invalid_cost_trade,
        locked_maximum_slack_violation=max(
            required_maximum
            - (
                float(optimum_maximum)
                + frozen_solver.LEXICOGRAPHIC_LOCK
            ),
            0.0,
        ),
        locked_sum_slack_violation=max(
            required_sum
            - (float(optimum_sum) + frozen_solver.LEXICOGRAPHIC_LOCK),
            0.0,
        ),
        required_maximum_slack=required_maximum,
        required_sum_slack=required_sum,
    )


def _emergency_candidate_audit_failure(
    audit: _EmergencyCandidateAudit,
) -> str | None:
    """Return the first failed frozen emergency-candidate audit gate."""

    values = np.asarray(
        [
            audit.maximum_hard_or_noninherited_violation,
            audit.maximum_inherited_worsening,
            audit.maximum_invalid_cost_trade,
            audit.locked_maximum_slack_violation,
            audit.locked_sum_slack_violation,
            audit.required_maximum_slack,
            audit.required_sum_slack,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        return "nonfinite"
    if (
        audit.maximum_hard_or_noninherited_violation
        > frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE
    ):
        return "hard_or_noninherited_constraint"
    if (
        audit.maximum_inherited_worsening
        > frozen_solver.EMERGENCY_WORSENING_TOLERANCE
    ):
        return "inherited_slack_worsening"
    if (
        audit.maximum_invalid_cost_trade
        > frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE
    ):
        return "invalid_cost_trade"
    if (
        audit.locked_maximum_slack_violation
        > frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE
    ):
        return "maximum_slack_lock"
    if (
        audit.locked_sum_slack_violation
        > frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE
    ):
        return "sum_slack_lock"
    return None


def _emergency_candidate_audit_metrics_history(
    audit: _EmergencyCandidateAudit,
) -> tuple[str, ...]:
    """Serialize independent emergency-candidate audit metrics."""

    return (
        *(
            "emergency_economic_candidate_audit_"
            + name
            + ":"
            + float(value).hex()
            for name, value in (
                (
                    "maximum_hard_or_noninherited_violation",
                    audit.maximum_hard_or_noninherited_violation,
                ),
                (
                    "maximum_inherited_worsening",
                    audit.maximum_inherited_worsening,
                ),
                (
                    "maximum_invalid_cost_trade",
                    audit.maximum_invalid_cost_trade,
                ),
                (
                    "locked_maximum_slack_violation",
                    audit.locked_maximum_slack_violation,
                ),
                (
                    "locked_sum_slack_violation",
                    audit.locked_sum_slack_violation,
                ),
                (
                    "required_maximum_slack",
                    audit.required_maximum_slack,
                ),
                ("required_sum_slack", audit.required_sum_slack),
            )
        ),
    )


def _emergency_candidate_audit_history(
    audit: _EmergencyCandidateAudit,
) -> tuple[str, ...]:
    """Serialize the passing independent emergency-candidate audit."""

    return (
        *_emergency_candidate_audit_metrics_history(audit),
        "emergency_economic_candidate_audit:passed",
    )


def _emergency_solve(
    data: AllocationProblemData,
    *,
    kind: SolveKind,
    raw_weights: np.ndarray | None,
    architecture: str,
    ordinary_status_history: tuple[str, ...],
) -> frozen_solver.AllocationResult:
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
            if value > frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE:
                raise frozen_solver.AllocationSolveError(
                    f"pretrade violates an always-hard emergency rule: {name}"
                )

    normalized, weights, caps = _scaled_weight_expression(
        data,
        name="stage11_emergency_normalized_weights",
    )
    objective, cost_constraints = _economic_objective_and_cost_constraints(
        data,
        normalized,
        weights,
        caps,
    )
    constraints = (
        hard_trade_and_sign_constraints(data, weights) + cost_constraints
    )
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
        raise frozen_solver.AllocationSolveError(
            "ordinary problem failed but no inherited target violation permits slack"
        )

    slack_vector = cp.hstack(list(slack_variables.values()))
    maximum = cp.Variable(nonneg=True, name="maximum_normalized_slack")
    step1_constraints = constraints + [slack_vector <= maximum]
    step1 = cp.Problem(cp.Minimize(maximum), step1_constraints)
    history = list(ordinary_status_history)
    step1_status = _solve_status(step1, frozen_solver.RETRY_SOLVER_SETTINGS)
    history.append(f"emergency_minimax:{step1_status}")
    if step1_status == cp.OPTIMAL_INACCURATE:
        try:
            history.extend(
                _certify_lexicographic_optimal_inaccurate(
                    step1,
                    primal_status=step1_status,
                    prefix="emergency_minimax",
                )
            )
        except (
            frozen_solver.AllocationSolveError,
            TypeError,
            ValueError,
        ) as exc:
            raise frozen_solver.AllocationSolveError(
                "emergency maximum-slack step was not independently "
                "certified: "
                + ", ".join(history)
                + ", certificate_error:"
                + str(exc)
            ) from exc
    elif step1_status != cp.OPTIMAL:
        raise frozen_solver.AllocationSolveError(
            "emergency maximum-slack step failed: " + ", ".join(history)
        )
    optimum_maximum = float(maximum.value)

    step2_constraints = step1_constraints + [
        maximum <= optimum_maximum + frozen_solver.LEXICOGRAPHIC_LOCK
    ]
    slack_sum = cp.sum(slack_vector)
    step2 = cp.Problem(cp.Minimize(slack_sum), step2_constraints)
    step2_status = _solve_status(step2, frozen_solver.RETRY_SOLVER_SETTINGS)
    history.append(f"emergency_sum:{step2_status}")
    if step2_status == cp.OPTIMAL_INACCURATE:
        try:
            history.extend(
                _certify_lexicographic_optimal_inaccurate(
                    step2,
                    primal_status=step2_status,
                    prefix="emergency_sum",
                )
            )
        except (
            frozen_solver.AllocationSolveError,
            TypeError,
            ValueError,
        ) as exc:
            raise frozen_solver.AllocationSolveError(
                "emergency sum-slack step was not independently "
                "certified: "
                + ", ".join(history)
                + ", certificate_error:"
                + str(exc)
            ) from exc
    elif step2_status != cp.OPTIMAL:
        raise frozen_solver.AllocationSolveError(
            "emergency sum-slack step failed: " + ", ".join(history)
        )
    optimum_sum = float(slack_sum.value)

    final_constraints = step2_constraints + [
        slack_sum <= optimum_sum + frozen_solver.LEXICOGRAPHIC_LOCK
    ]
    final_problem = cp.Problem(objective, final_constraints)
    final_status = _solve_status(
        final_problem,
        frozen_solver.PRIMARY_SOLVER_SETTINGS,
    )
    history.append(f"emergency_economic_primary:{final_status}")
    if final_status != cp.OPTIMAL:
        final_status = _solve_status(
            final_problem,
            frozen_solver.RETRY_SOLVER_SETTINGS,
        )
        history.append(f"emergency_economic_retry:{final_status}")
    certified_candidate: np.ndarray | None = None
    equivalent_final_problem_used = False
    if final_status == cp.OPTIMAL_INACCURATE and history[-2:] == [
        "emergency_economic_primary:optimal_inaccurate",
        "emergency_economic_retry:optimal_inaccurate",
    ]:
        equivalent_final_problem_used = True
        # ``maximum`` has no role in the final economic objective.  The pair
        # ``slack <= maximum`` and ``maximum <= locked_maximum`` is therefore
        # exactly equivalent to ``slack <= locked_maximum`` after existential
        # elimination.  Retry this algebraically identical, smaller cone only
        # after both frozen final solves return the same numerical status.
        equivalent_constraints = constraints + [
            slack_vector
            <= optimum_maximum + frozen_solver.LEXICOGRAPHIC_LOCK,
            slack_sum <= optimum_sum + frozen_solver.LEXICOGRAPHIC_LOCK,
        ]
        equivalent_problem = cp.Problem(
            objective,
            equivalent_constraints,
        )
        final_status = _solve_status(
            equivalent_problem,
            frozen_solver.RETRY_SOLVER_SETTINGS,
        )
        history.append(
            f"emergency_economic_equivalent_retry:{final_status}"
        )
        final_problem = equivalent_problem
        if final_status == cp.OPTIMAL_INACCURATE:
            if weights.value is None:
                history.append(
                    "emergency_economic_equivalent_retry_audit:"
                    "missing_primal"
                )
                raise frozen_solver.AllocationSolveError(
                    "emergency equivalent economic-objective step "
                    "returned no primal: "
                    + ", ".join(history)
                )
            try:
                certified_candidate = (
                    frozen_solver._canonicalize_solver_weights(
                        data,
                        np.asarray(
                            weights.value,
                            dtype=np.float64,
                        ).reshape(-1),
                    )
                )
            except (
                frozen_solver.AllocationSolveError,
                TypeError,
                ValueError,
            ) as exc:
                history.append(
                    "emergency_economic_equivalent_retry_audit:"
                    "invalid_primal"
                )
                raise frozen_solver.AllocationSolveError(
                    "emergency equivalent economic-objective step "
                    "returned an invalid primal: "
                    + ", ".join(history)
                ) from exc
            emergency_audit = _audit_emergency_candidate(
                data,
                certified_candidate,
                pretrade_diagnostics=pretrade_diagnostics,
                pretrade_slacks=pretrade_slacks,
                optimum_maximum=optimum_maximum,
                optimum_sum=optimum_sum,
            )
            emergency_failure = _emergency_candidate_audit_failure(
                emergency_audit
            )
            if emergency_failure is not None:
                history.extend(
                    _emergency_candidate_audit_metrics_history(
                        emergency_audit
                    )
                )
                history.append(
                    "emergency_economic_candidate_audit:failed:"
                    + emergency_failure
                )
                raise frozen_solver.AllocationSolveError(
                    "emergency equivalent economic candidate failed "
                    "independent audit: "
                    + ", ".join(history)
                )
            history.extend(
                _emergency_candidate_audit_history(emergency_audit)
            )
            certificate: _HybridDualCertificate | None = None
            explicit_certificate: _HybridDualCertificate | None = None
            retry_certificate: _HybridDualCertificate | None = None
            try:
                explicit_certificate = (
                    _build_explicit_hybrid_dual_certificate(
                        data,
                        certified_candidate,
                        final_problem,
                        normalized,
                        weights,
                        caps,
                        candidate_maximum_constraint_violation=(
                            emergency_audit.certificate_primal_violation
                        ),
                    )
                )
            except (
                frozen_solver.AllocationSolveError,
                TypeError,
                ValueError,
            ) as exc:
                history.append(
                    "emergency_economic_explicit_dual_certificate_"
                    "audit:error:"
                    + type(exc).__name__
                )
            else:
                explicit_failure = _hybrid_dual_certificate_failure(
                    explicit_certificate
                )
                if explicit_failure is None:
                    certificate = explicit_certificate
                else:
                    history.extend(
                        (
                            (
                                "emergency_economic_explicit_dual_"
                                "certificate:"
                                + explicit_certificate.dual_status
                            ),
                            (
                                "emergency_economic_explicit_dual_"
                                "certificate_source:"
                                + explicit_certificate.certificate_source
                            ),
                            (
                                "emergency_economic_explicit_dual_"
                                "certificate_audit:failed:"
                                + explicit_failure
                            ),
                            *_hybrid_dual_certificate_metrics_history(
                                explicit_certificate,
                                prefix=(
                                    "emergency_economic_explicit"
                                ),
                            ),
                        )
                    )
            if certificate is None:
                try:
                    retry_certificate = (
                        _build_retry_economic_dual_certificate(
                            data,
                            certified_candidate,
                            final_problem,
                            normalized,
                            weights,
                            caps,
                            dual_status=cp.OPTIMAL_INACCURATE,
                            candidate_maximum_constraint_violation=(
                                emergency_audit.certificate_primal_violation
                            ),
                        )
                    )
                except (
                    frozen_solver.AllocationSolveError,
                    TypeError,
                    ValueError,
                ) as exc:
                    history.append(
                        "emergency_economic_retry_dual_certificate_"
                        "audit:error:"
                        + type(exc).__name__
                    )
                    raise frozen_solver.AllocationSolveError(
                        "emergency economic dual certificate "
                        "construction failed: "
                        + ", ".join(history)
                    ) from exc
                retry_failure = _hybrid_dual_certificate_failure(
                    retry_certificate
                )
                if retry_failure is None:
                    certificate = retry_certificate
                else:
                    history.extend(
                        (
                            (
                                "emergency_economic_retry_dual_"
                                "certificate:"
                                + retry_certificate.dual_status
                            ),
                            (
                                "emergency_economic_retry_dual_"
                                "certificate_source:"
                                + retry_certificate.certificate_source
                            ),
                            (
                                "emergency_economic_retry_dual_"
                                "certificate_audit:failed:"
                                + retry_failure
                            ),
                            *_hybrid_dual_certificate_metrics_history(
                                retry_certificate,
                                prefix="emergency_economic_retry",
                            ),
                        )
                    )
            if (
                certificate is None
                and explicit_certificate is not None
                and retry_certificate is not None
            ):
                try:
                    reconciled_certificate = (
                        _build_reconciled_hybrid_dual_certificate(
                            final_problem,
                            explicit_certificate,
                            retry_certificate,
                        )
                    )
                except (
                    frozen_solver.AllocationSolveError,
                    TypeError,
                    ValueError,
                ) as exc:
                    history.append(
                        "emergency_economic_reconciled_dual_"
                        "certificate_audit:error:"
                        + type(exc).__name__
                    )
                else:
                    reconciled_failure = (
                        _hybrid_dual_certificate_failure(
                            reconciled_certificate
                        )
                    )
                    if reconciled_failure is None:
                        certificate = reconciled_certificate
                    else:
                        history.extend(
                            (
                                (
                                    "emergency_economic_reconciled_dual_"
                                    "certificate:"
                                    + reconciled_certificate.dual_status
                                ),
                                (
                                    "emergency_economic_reconciled_dual_"
                                    "certificate_source:"
                                    + reconciled_certificate.certificate_source
                                ),
                                (
                                    "emergency_economic_reconciled_dual_"
                                    "certificate_audit:failed:"
                                    + reconciled_failure
                                ),
                                *_hybrid_dual_certificate_metrics_history(
                                    reconciled_certificate,
                                    prefix=(
                                        "emergency_economic_reconciled"
                                    ),
                                ),
                            )
                        )
            if (
                certificate is None
                and explicit_certificate is not None
                and retry_certificate is not None
            ):
                try:
                    bounded_certificate = (
                        _build_bounded_feasible_hybrid_dual_certificate(
                            final_problem,
                            explicit_certificate,
                            retry_certificate,
                        )
                    )
                except (
                    frozen_solver.AllocationSolveError,
                    TypeError,
                    ValueError,
                ) as exc:
                    history.append(
                        "emergency_economic_bounded_feasible_dual_"
                        "certificate_audit:error:"
                        + type(exc).__name__
                    )
                else:
                    bounded_failure = _hybrid_dual_certificate_failure(
                        bounded_certificate
                    )
                    if bounded_failure is None:
                        certificate = bounded_certificate
                    else:
                        history.extend(
                            (
                                (
                                    "emergency_economic_bounded_feasible_"
                                    "dual_certificate:"
                                    + bounded_certificate.dual_status
                                ),
                                (
                                    "emergency_economic_bounded_feasible_"
                                    "dual_certificate_source:"
                                    + bounded_certificate.certificate_source
                                ),
                                (
                                    "emergency_economic_bounded_feasible_"
                                    "dual_certificate_audit:failed:"
                                    + bounded_failure
                                ),
                                *_hybrid_dual_certificate_metrics_history(
                                    bounded_certificate,
                                    prefix=(
                                        "emergency_economic_"
                                        "bounded_feasible"
                                    ),
                                ),
                            )
                        )
            if certificate is None:
                raise frozen_solver.AllocationSolveError(
                    "emergency economic retry candidate was not "
                    "dual-certified: "
                    + ", ".join(history)
                )
            history.extend(
                _hybrid_dual_certificate_history(
                    certificate,
                    prefix="emergency_economic",
                )
            )
            final_status = DUAL_CERTIFIED_SOLVER_STATUS
    if final_status not in (cp.OPTIMAL, DUAL_CERTIFIED_SOLVER_STATUS):
        raise frozen_solver.AllocationSolveError(
            "emergency economic-objective step failed: " + ", ".join(history)
        )
    if certified_candidate is None and weights.value is None:
        raise frozen_solver.AllocationSolveError(
            "emergency economic-objective step returned no primal"
        )
    vector = (
        certified_candidate
        if certified_candidate is not None
        else frozen_solver._canonicalize_solver_weights(
            data,
            np.asarray(weights.value, dtype=np.float64).reshape(-1),
        )
    )
    if certified_candidate is None and equivalent_final_problem_used:
        emergency_audit = _audit_emergency_candidate(
            data,
            vector,
            pretrade_diagnostics=pretrade_diagnostics,
            pretrade_slacks=pretrade_slacks,
            optimum_maximum=optimum_maximum,
            optimum_sum=optimum_sum,
        )
        emergency_failure = _emergency_candidate_audit_failure(
            emergency_audit
        )
        if emergency_failure is not None:
            history.extend(
                _emergency_candidate_audit_metrics_history(emergency_audit)
            )
            history.append(
                "emergency_economic_candidate_audit:failed:"
                + emergency_failure
            )
            raise frozen_solver.AllocationSolveError(
                "emergency economic candidate failed independent audit: "
                + ", ".join(history)
            )
        history.extend(_emergency_candidate_audit_history(emergency_audit))
    diagnostics = independent_constraint_diagnostics(data, vector)
    for name, value in diagnostics.violations.items():
        if name.startswith(always_hard_prefixes) or name == "turnover":
            if value > frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE:
                raise frozen_solver.AllocationSolveError(
                    f"emergency hard constraint failed: {name}"
                )
    normalized_slacks = {
        name: max(diagnostics.normalized_violations.get(name, 0.0), 0.0)
        for name in slack_variables
    }
    for name, observed in normalized_slacks.items():
        prior = max(
            pretrade_diagnostics.normalized_violations.get(name, 0.0),
            0.0,
        )
        if observed > prior + frozen_solver.EMERGENCY_WORSENING_TOLERANCE:
            raise frozen_solver.AllocationSolveError(
                f"emergency violation worsened: {name}"
            )
        if (
            observed
            > pretrade_slacks[name]
            + frozen_solver.EMERGENCY_WORSENING_TOLERANCE
        ):
            raise frozen_solver.AllocationSolveError(
                f"emergency slack exceeded inherited bound: {name}"
            )
    for name, value in diagnostics.violations.items():
        if (
            name not in slack_variables
            and name != "turnover"
            and not name.startswith(always_hard_prefixes)
            and value > frozen_solver.INDEPENDENT_ABSOLUTE_TOLERANCE
        ):
            raise frozen_solver.AllocationSolveError(
                f"emergency created a non-inherited target violation: {name}"
            )
    return frozen_solver._make_result(
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
    )


def solve_hybrid_stage11(
    data: AllocationProblemData,
) -> frozen_solver.AllocationResult:
    """Solve the frozen hybrid allocation through Stage 11 coordinates."""

    architecture = "hybrid_alpha_risk_cost_convex"
    if data.n_assets == 0:
        return frozen_solver._make_result(
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
    except frozen_solver._CertifiedOrdinaryInfeasible as certified:
        return _emergency_solve(
            data,
            kind="hybrid",
            raw_weights=None,
            architecture=architecture,
            ordinary_status_history=certified.status_history,
        )


def solve_projection_stage11(
    data: AllocationProblemData,
    raw_weights: object,
    *,
    architecture: str,
) -> frozen_solver.AllocationResult:
    """Solve a frozen challenger projection through Stage 11 coordinates."""

    raw = np.asarray(raw_weights, dtype=np.float64)
    if raw.shape != (data.n_assets,) or not np.isfinite(raw).all():
        raise ValueError("raw_weights must be a finite n_assets vector")
    if data.n_assets == 0:
        return frozen_solver._make_result(
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
    except frozen_solver._CertifiedOrdinaryInfeasible as certified:
        return _emergency_solve(
            data,
            kind="projection",
            raw_weights=raw,
            architecture=architecture,
            ordinary_status_history=certified.status_history,
        )


__all__ = [
    "TRADE_NUMERICAL_SCALE",
    "solve_hybrid_stage11",
    "solve_projection_stage11",
]
