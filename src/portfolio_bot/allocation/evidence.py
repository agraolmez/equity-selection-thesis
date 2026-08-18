"""Deterministic, fictitious evidence for the Stage 10 allocators.

The evidence runner is intentionally incapable of opening a database or an
upstream payload.  It constructs only in-memory synthetic cross-sections and
exercises the three frozen allocation architectures, common constraints,
cost backends, candidate rules, and emergency waterfall.  Its canonical JSON
hash is the numeric replay contract used by primary publication and by the
independent verifier.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from typing import Any, Mapping

import cvxpy as cp
import numpy as np
import torch

from portfolio_bot.allocation.candidates import form_monthly_candidates
from portfolio_bot.allocation.constraints import (
    ANNUAL_VARIANCE_LIMIT,
    COUNTRY_NET_LIMIT,
    FACTOR_EXPOSURE_LIMIT,
    FOREIGN_NET_LIMIT,
    GROSS_LIMIT,
    LONG_CAP,
    NET_NAV_LIMIT,
    ONE_WAY_TURNOVER_LIMIT,
    SECTOR_GROSS_LIMIT,
    SECTOR_NET_LIMIT,
    SHORT_CAP,
    independent_constraint_diagnostics,
    slackable_constraint_specs,
)
from portfolio_bot.allocation.contract import (
    BoundaryCounters,
    canonical_json_bytes,
    enforce_stage10_boundary,
)
from portfolio_bot.allocation.costs import (
    CostInputError,
    cvxpy_total_cost,
    numpy_total_cost,
    torch_total_cost,
)
from portfolio_bot.allocation.decision_focused import (
    DECISION_SEEDS,
    PRODUCTION_TRAINING_RUNTIME_CERTIFIED,
    DecisionFocusedInactiveError,
    DecisionFocusedAllocator,
    DecisionTrainingMonth,
    DifferentiableProjection,
    build_seeded_allocators,
    project_seed_ensemble,
    run_chronological_training_sequence,
    train_decision_focused_seed,
    torch_daily_holding_accounting,
    validate_quarterly_refit_date,
    weighted_expanding_month_weights,
)
from portfolio_bot.allocation.domain import (
    ACTIVE,
    INSUFFICIENT_DISJOINT_CANDIDATES,
    AllocationProblemData,
    CandidateUniverse,
    derive_borrow_rate_annual,
)
from portfolio_bot.allocation.fixtures import balanced_synthetic_problem
from portfolio_bot.allocation.rank import allocate_rank
from portfolio_bot.allocation.risk import augmented_risk_diagnostics
from portfolio_bot.allocation.solver import (
    INDEPENDENT_ABSOLUTE_TOLERANCE,
    solve_hybrid,
    solve_projection,
)


EVIDENCE_SCHEMA_VERSION = 2
FORWARD_TOLERANCE = 5e-8
TRAINING_PROJECTION_TOLERANCE = 5e-5
GRADIENT_ABSOLUTE_TOLERANCE = 5e-4
GRADIENT_RELATIVE_TOLERANCE = 5e-3


class Stage10EvidenceError(RuntimeError):
    """Raised when one synthetic implementation gate does not pass."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite_float(value: object) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise Stage10EvidenceError("synthetic evidence contains a nonfinite value")
    return result


def _vector_hash(value: object) -> str:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.isfinite(vector).all():
        raise Stage10EvidenceError("synthetic evidence vector is nonfinite")
    return _sha256([float(item) for item in vector])


def _raises(
    exception_type: type[BaseException],
    operation: object,
    *args: object,
    message_contains: str | None = None,
    **kwargs: object,
) -> tuple[bool, str | None]:
    """Execute one negative-path check and retain the observed exception."""

    if not callable(operation):
        raise TypeError("operation must be callable")
    try:
        operation(*args, **kwargs)
    except exception_type as exc:
        message = str(exc)
        return (
            message_contains is None or message_contains in message,
            f"{type(exc).__name__}: {message}",
        )
    except Exception as exc:  # pragma: no cover - evidence reports wrong failure class
        return False, f"{type(exc).__name__}: {exc}"
    return False, None


def _checks_pass(checks: Mapping[str, object]) -> bool:
    """Return true only for a nonempty mapping of literal executed booleans."""

    return bool(checks) and all(
        isinstance(value, bool) and value for value in checks.values()
    )


def _result_summary(result: object) -> dict[str, Any]:
    diagnostics = result.diagnostics
    return {
        "architecture": result.architecture,
        "mode": result.mode,
        "solver_status": result.solver_status,
        "solver_attempt": result.solver_attempt,
        "weights_sha256": _vector_hash(result.weights),
        "nonzero_weights": int(np.count_nonzero(np.abs(result.weights) > 1e-10)),
        "long_gross": _finite_float(diagnostics.long_gross),
        "short_gross": _finite_float(diagnostics.short_gross),
        "gross": _finite_float(diagnostics.gross),
        "net": _finite_float(diagnostics.net),
        "one_way_turnover": _finite_float(diagnostics.one_way_turnover),
        "augmented_annual_volatility": _finite_float(
            diagnostics.augmented_annual_volatility
        ),
        "maximum_constraint_violation": _finite_float(diagnostics.maximum_violation),
        "factor_exposures": [
            _finite_float(value) for value in diagnostics.factor_exposures
        ],
        "objective_value": _finite_float(result.objective_value),
        "transaction_cost": _finite_float(result.transaction_cost),
        "borrow_cost": _finite_float(result.borrow_cost),
        "financing_cost": _finite_float(result.financing_cost),
        "total_cost": _finite_float(result.total_cost),
        "normalized_slacks": {
            name: _finite_float(value)
            for name, value in sorted(result.normalized_slacks.items())
        },
        "binding_trade_limits": [int(value) for value in result.binding_trade_limits],
        "estimated_months_to_compliance": result.estimated_months_to_compliance,
    }


def _candidate_evidence() -> dict[str, Any]:
    count = 120
    scores = np.linspace(-0.5, 0.5, count, dtype=np.float64)
    scores -= float(np.mean(scores))
    previous = np.zeros(count, dtype=np.float64)
    previous[57] = -0.002
    adv = np.linspace(10_000_000.0, 1_200_000_000.0, count, dtype=np.float64)
    adv[57] = np.nan
    universe = CandidateUniverse(
        permno=np.arange(7_000_001, 7_000_001 + count, dtype=np.int64),
        permco=np.arange(8_000_001, 8_000_001 + count, dtype=np.int64),
        score=scores,
        long_eligible=np.ones(count, dtype=np.bool_),
        short_eligible=np.ones(count, dtype=np.bool_),
        previous_weight=previous,
        adv20_usd=adv,
        sector=np.where(np.arange(count) % 2 == 0, "SYNTH_A", "SYNTH_B"),
        country=np.where(np.arange(count) % 3 == 0, "USA", "SYNTH_FOREIGN"),
        required_exit=np.zeros(count, dtype=np.bool_),
    )
    selection = form_monthly_candidates(universe)
    inherited_short_quintile = int(selection.borrow_liquidity_quintile[57])

    # This stricter screen makes two independent 50-name scans overlap on 20
    # securities even though a disjoint 50/50 assignment exists.  The frozen
    # joint augmenting-path repair must find it rather than suppressing risk.
    repair_short_eligible = np.zeros(count, dtype=np.bool_)
    repair_short_eligible[40:] = True
    repair_universe = replace(
        universe,
        short_eligible=repair_short_eligible,
        previous_weight=np.zeros(count, dtype=np.float64),
        adv20_usd=np.linspace(
            10_000_000.0,
            1_200_000_000.0,
            count,
            dtype=np.float64,
        ),
    )
    independent_long = np.zeros(count, dtype=np.bool_)
    independent_short = np.zeros(count, dtype=np.bool_)
    independent_long[70:120] = True
    independent_short[40:90] = True
    independent_overlap = int(np.sum(independent_long & independent_short))
    repaired = form_monthly_candidates(repair_universe)
    repaired_replay = form_monthly_candidates(repair_universe)

    # Exercise the complete candidate -> quintile -> internally derived
    # primary/stress borrow-rate path.  No caller-supplied rate is accepted.
    covariance = np.diag(np.asarray([0.0025, 0.001, 0.001, 0.001, 0.001, 0.001]))
    specific = np.full(count, 0.20**2, dtype=np.float64)
    forecast = np.full(count, 0.20, dtype=np.float64)
    capacity = np.where(
        np.isfinite(adv) & (adv > 0.0),
        0.05 * adv / 100_000_000.0,
        0.0,
    )
    adapter_inputs = dict(
        expected_return=np.zeros(count, dtype=np.float64),
        factor_betas=np.zeros((count, 6), dtype=np.float64),
        factor_covariance_monthly=covariance,
        specific_variance_monthly=specific,
        forecast_volatility=forecast,
        full_spread=np.full(count, 0.002, dtype=np.float64),
        impact_volatility_daily=np.full(count, 0.02, dtype=np.float64),
        trade_capacity_weight=capacity,
        nav_dollars=100_000_000.0,
        market_variance_monthly=0.0025,
    )
    primary_problem = selection.to_problem_data(
        borrow_scenario="primary",
        **adapter_inputs,
    )
    stress_problem = selection.to_problem_data(
        borrow_scenario="stress",
        **adapter_inputs,
    )
    short = primary_problem.short_mask
    lowest = stress_problem.borrow_liquidity_quintile == 1
    signature_has_external_rate = (
        "borrow_rate_annual" in inspect.signature(AllocationProblemData).parameters
    )
    checks = {
        "active": selection.status == ACTIVE,
        "minimum_long_candidates": int(selection.long_candidate.sum()) >= 50,
        "minimum_short_candidates": int(selection.short_candidate.sum()) >= 50,
        "candidate_sets_disjoint": not bool(
            np.any(selection.long_candidate & selection.short_candidate)
        ),
        "missing_adv_is_lowest_borrow_quintile": inherited_short_quintile == 1,
        "short_union_fully_ranked": bool(
            np.all(
                selection.borrow_liquidity_quintile[
                    selection.short_candidate | (previous < 0.0)
                ]
                > 0
            )
        ),
        "independent_scans_would_overlap": independent_overlap == 20,
        "joint_repair_remains_active": repaired.status == ACTIVE,
        "joint_repair_has_exact_minimum_each_side": (
            int(repaired.long_candidate.sum()) == 50
            and int(repaired.short_candidate.sum()) == 50
        ),
        "joint_repair_is_disjoint": not bool(
            np.any(repaired.long_candidate & repaired.short_candidate)
        ),
        "joint_repair_preserves_valid_positional_long_tail": bool(
            repaired.long_candidate[96:120].all()
        ),
        "joint_repair_is_deterministic": bool(
            np.array_equal(repaired.long_candidate, repaired_replay.long_candidate)
            and np.array_equal(
                repaired.short_candidate, repaired_replay.short_candidate
            )
        ),
        "primary_borrow_mapping_integrated": bool(
            np.all(primary_problem.borrow_rate_annual[~short] == 0.0)
            and np.all(primary_problem.borrow_rate_annual[short] == 0.01)
        ),
        "stress_borrow_mapping_integrated": bool(
            np.all(stress_problem.borrow_rate_annual[lowest] == 0.10)
            and np.all(
                stress_problem.borrow_rate_annual[stress_problem.short_mask & ~lowest]
                == 0.03
            )
        ),
        "borrow_rate_cannot_be_supplied_by_caller": not signature_has_external_rate,
    }
    if not all(checks.values()):
        raise Stage10EvidenceError(f"synthetic candidate gate failed: {checks}")
    return {
        "fixture_rows": count,
        "fixture_identifiers_are_fictitious": True,
        "status": selection.status,
        "tail_size": selection.tail_size,
        "long_candidates": int(selection.long_candidate.sum()),
        "short_candidates": int(selection.short_candidate.sum()),
        "overlap_count": selection.overlap_count,
        "inherited_missing_adv_short_quintile": inherited_short_quintile,
        "joint_repair": {
            "independent_scan_overlap_count": independent_overlap,
            "status": repaired.status,
            "long_candidates": int(repaired.long_candidate.sum()),
            "short_candidates": int(repaired.short_candidate.sum()),
            "overlap_count": repaired.overlap_count,
            "long_weights_sha256": _vector_hash(
                repaired.long_candidate.astype(np.float64)
            ),
            "short_weights_sha256": _vector_hash(
                repaired.short_candidate.astype(np.float64)
            ),
        },
        "integrated_borrow": {
            "primary_rate_sha256": _vector_hash(primary_problem.borrow_rate_annual),
            "stress_rate_sha256": _vector_hash(stress_problem.borrow_rate_annual),
            "primary_short_rate_values": sorted(
                set(float(value) for value in primary_problem.borrow_rate_annual[short])
            ),
            "stress_short_rate_values": sorted(
                set(
                    float(value)
                    for value in stress_problem.borrow_rate_annual[
                        stress_problem.short_mask
                    ]
                )
            ),
        },
        "checks": checks,
    }


def _cost_evidence() -> dict[str, Any]:
    weights = np.asarray([0.60, -0.55], dtype=np.float64)
    previous = np.asarray([0.40, -0.35], dtype=np.float64)
    spread = np.asarray([0.0020, 0.0030], dtype=np.float64)
    adv = np.asarray([400_000_000.0, 250_000_000.0], dtype=np.float64)
    volatility = np.asarray([0.020, 0.025], dtype=np.float64)
    borrow = np.asarray([0.010, 0.030], dtype=np.float64)
    numpy_cost = numpy_total_cost(
        weights,
        previous,
        spread,
        adv,
        volatility,
        borrow,
        100_000_000.0,
    )
    cvx_cost = cvxpy_total_cost(
        cp.Constant(weights),
        cp.Constant(previous),
        spread,
        adv,
        volatility,
        borrow,
        100_000_000.0,
    )
    torch_cost = torch_total_cost(
        torch.tensor(weights, dtype=torch.float64),
        torch.tensor(previous, dtype=torch.float64),
        torch.tensor(spread, dtype=torch.float64),
        torch.tensor(adv, dtype=torch.float64),
        torch.tensor(volatility, dtype=torch.float64),
        torch.tensor(borrow, dtype=torch.float64),
        100_000_000.0,
    )
    components = ("transaction", "borrow", "financing", "total")
    numpy_values = {
        name: _finite_float(getattr(numpy_cost, name)) for name in components
    }
    cvx_values = {
        name: _finite_float(getattr(cvx_cost, name).value) for name in components
    }
    torch_values = {
        name: _finite_float(getattr(torch_cost, name).detach().cpu().item())
        for name in components
    }
    differences = [
        abs(numpy_values[name] - backend[name])
        for backend in (cvx_values, torch_values)
        for name in components
    ]
    maximum_difference = max(differences)
    missing_input_failed_closed = False
    try:
        numpy_total_cost(
            np.asarray([0.01]),
            np.asarray([0.0]),
            np.asarray([0.002]),
            np.asarray([np.nan]),
            np.asarray([0.02]),
            np.asarray([0.01]),
            100_000_000.0,
        )
    except CostInputError:
        missing_input_failed_closed = True
    missing_input_zero_trade = numpy_total_cost(
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([0.002]),
        np.asarray([np.nan]),
        np.asarray([0.02]),
        np.asarray([0.01]),
        100_000_000.0,
    )
    checks = {
        "transaction_positive": numpy_values["transaction"] > 0.0,
        "borrow_positive": numpy_values["borrow"] > 0.0,
        "financing_positive": numpy_values["financing"] > 0.0,
        "decomposition_exact": abs(
            numpy_values["total"]
            - numpy_values["transaction"]
            - numpy_values["borrow"]
            - numpy_values["financing"]
        )
        <= 1e-15,
        "three_backends_equivalent": maximum_difference <= 1e-12,
        "missing_input_nonzero_trade_fails_closed": missing_input_failed_closed,
        "missing_input_zero_trade_remains_feasible": missing_input_zero_trade.total
        == 0.0,
    }
    if not all(checks.values()):
        raise Stage10EvidenceError(f"synthetic cost gate failed: {checks}")
    return {
        "fixture_identifiers_are_fictitious": True,
        "numpy": numpy_values,
        "cvxpy": cvx_values,
        "torch": torch_values,
        "maximum_backend_absolute_difference": maximum_difference,
        "checks": checks,
    }


def _domain_and_risk_evidence() -> dict[str, Any]:
    """Exercise authoritative derivation and fail-closed liquidity inputs."""

    data = balanced_synthetic_problem()
    ordinary_covariance = (
        data.factor_betas @ data.factor_covariance_monthly @ data.factor_betas.T
        + np.diag(data.specific_variance_monthly)
    )
    expected_augmented = ordinary_covariance + (
        4.0
        * data.market_variance_monthly
        * np.outer(data.factor_betas[:, 0], data.factor_betas[:, 0])
    )
    root_reconstruction_difference = float(
        np.max(np.abs(data.risk_root.T @ data.risk_root - expected_augmented))
    )
    implied_forecast = np.sqrt(np.diag(ordinary_covariance))
    forecast_difference = float(
        np.max(np.abs(data.forecast_volatility - implied_forecast))
    )
    risk = augmented_risk_diagnostics(
        np.asarray([0.004, 0.003, 0.002, 0.001, -0.004, -0.003, -0.002, -0.001]),
        data.factor_betas,
        data.factor_covariance_monthly,
        data.specific_variance_monthly,
        market_variance_monthly=data.market_variance_monthly,
    )

    bad_forecast = data.forecast_volatility.copy()
    bad_forecast[0] += 1e-5
    forecast_rejected, forecast_exception = _raises(
        ValueError,
        replace,
        data,
        forecast_volatility=bad_forecast,
        message_contains="sqrt(diag",
    )
    bad_market_rejected, market_exception = _raises(
        ValueError,
        replace,
        data,
        market_variance_monthly=data.market_variance_monthly + 1e-8,
        message_contains="factor covariance",
    )

    adv = data.adv20_usd.copy()
    capacity = data.trade_capacity_weight.copy()
    adv[0] = 1_000_000.0
    capacity[0] = 0.00100001
    capacity_rejected, capacity_exception = _raises(
        ValueError,
        replace,
        data,
        adv20_usd=adv,
        trade_capacity_weight=capacity,
        message_contains="exceeds 10% of ADV",
    )
    capacity[0] = 0.001
    constrained = replace(data, adv20_usd=adv, trade_capacity_weight=capacity)
    probe = constrained.previous_weight.copy()
    probe[0] = 0.0015
    diagnostic = independent_constraint_diagnostics(constrained, probe)
    observed_adv_violation = diagnostic.violations[
        f"adv_capacity:{int(constrained.permno[0])}"
    ]

    missing_adv = data.adv20_usd.copy()
    missing_capacity = data.trade_capacity_weight.copy()
    missing_adv[0] = np.nan
    missing_capacity[0] = 1e-8
    missing_rejected, missing_exception = _raises(
        ValueError,
        replace,
        data,
        adv20_usd=missing_adv,
        trade_capacity_weight=missing_capacity,
        message_contains="zero trade capacity",
    )
    missing_capacity[0] = 0.0
    blocked = replace(
        data,
        adv20_usd=missing_adv,
        trade_capacity_weight=missing_capacity,
    )
    blocked_result = solve_hybrid(blocked)

    checks = {
        "risk_root_rebuilt_from_primitives": root_reconstruction_difference <= 1e-12,
        "forecast_volatility_rebuilt_from_ordinary_covariance": forecast_difference
        <= 1e-12,
        "external_risk_root_not_a_constructor_input": "risk_root"
        not in inspect.signature(AllocationProblemData).parameters,
        "forecast_mismatch_rejected": forecast_rejected,
        "market_variance_mismatch_rejected": bad_market_rejected,
        "capacity_above_ten_percent_adv_rejected": capacity_rejected,
        "adv_diagnostic_recomputes_exact_limit": abs(observed_adv_violation - 0.0005)
        <= 1e-15,
        "missing_adv_positive_capacity_rejected": missing_rejected,
        "missing_adv_zero_capacity_is_exactly_fixed": bool(
            abs(blocked_result.weights[0] - blocked.previous_weight[0])
            <= INDEPENDENT_ABSOLUTE_TOLERANCE
        ),
        "missing_adv_binding_limit_disclosed": int(blocked.permno[0])
        in blocked_result.binding_trade_limits,
        "independent_augmented_risk_finite": bool(
            np.isfinite(risk.augmented_annual_variance)
            and np.isfinite(risk.factor_exposures).all()
        ),
    }
    if not _checks_pass(checks):
        raise Stage10EvidenceError(f"domain/risk synthetic gate failed: {checks}")
    return {
        "fixture_identifiers_are_fictitious": bool(
            np.all((data.permno >= 900001) & (data.permno <= 900008))
        ),
        "risk_root_reconstruction_maximum_absolute_difference": root_reconstruction_difference,
        "forecast_volatility_maximum_absolute_difference": forecast_difference,
        "augmented_annual_variance_probe": _finite_float(
            risk.augmented_annual_variance
        ),
        "adv_capacity_rejection_exception": capacity_exception,
        "missing_adv_rejection_exception": missing_exception,
        "forecast_rejection_exception": forecast_exception,
        "market_variance_rejection_exception": market_exception,
        "adv_diagnostic_observed_violation": _finite_float(observed_adv_violation),
        "zero_capacity_binding_trade_limits": list(blocked_result.binding_trade_limits),
        "checks": checks,
    }


def _manual_constraint_violations(
    data: AllocationProblemData,
    weights: np.ndarray,
) -> dict[str, float]:
    """Independent literal transcription of every frozen constraint formula."""

    vector = np.asarray(weights, dtype=np.float64)
    previous = data.previous_weight
    long_mask = data.long_mask
    short_mask = data.short_mask
    long_gross = float(np.sum(vector[long_mask]))
    short_gross = float(-np.sum(vector[short_mask]))
    gross = long_gross + short_gross
    net = long_gross - short_gross
    delta = vector - previous
    factor = data.factor_betas.T @ vector
    ordinary_monthly_variance = float(
        factor @ data.factor_covariance_monthly @ factor
        + np.dot(data.specific_variance_monthly, vector**2)
    )
    augmented_annual_variance = 12.0 * (
        ordinary_monthly_variance
        + 4.0 * data.market_variance_monthly * float(factor[0]) ** 2
    )
    violations: dict[str, float] = {
        "gross": gross - GROSS_LIMIT,
        "net_positive": net - NET_NAV_LIMIT,
        "net_negative": -net - NET_NAV_LIMIT,
        "dynamic_net_positive": 0.95 * long_gross - 1.05 * short_gross,
        "dynamic_net_negative": 0.95 * short_gross - 1.05 * long_gross,
        "augmented_risk": augmented_annual_variance - ANNUAL_VARIANCE_LIMIT,
        "turnover": 0.5 * float(np.sum(np.abs(delta))) - ONE_WAY_TURNOVER_LIMIT,
    }
    for index, identifier in enumerate(data.permno):
        permno = int(identifier)
        cap = LONG_CAP if data.side[index] > 0 else SHORT_CAP
        if data.side[index] > 0:
            violations[f"position_long:{permno}"] = float(vector[index] - cap)
            violations[f"sign:{permno}"] = float(-vector[index])
            if data.required_exit[index]:
                violations[f"required_exit_long:{permno}"] = float(vector[index])
        else:
            violations[f"position_short:{permno}"] = float(-vector[index] - cap)
            violations[f"sign:{permno}"] = float(vector[index])
            if data.required_exit[index]:
                violations[f"required_exit_short:{permno}"] = float(-vector[index])
        violations[f"trade_capacity:{permno}"] = float(
            abs(delta[index]) - data.trade_capacity_weight[index]
        )
        adv = float(data.adv20_usd[index])
        adv_capacity = (
            0.10 * adv / data.nav_dollars if np.isfinite(adv) and adv > 0.0 else 0.0
        )
        violations[f"adv_capacity:{permno}"] = float(abs(delta[index]) - adv_capacity)
        if data.reduction_only[index]:
            violations[f"reduction_only:{permno}"] = float(
                abs(vector[index]) - abs(previous[index])
            )
    for issuer in sorted(set(int(value) for value in data.permco.tolist())):
        mask = data.permco == issuer
        if np.any(mask & long_mask):
            violations[f"issuer_long:{issuer}"] = float(
                np.sum(vector[mask & long_mask]) - LONG_CAP
            )
        if np.any(mask & short_mask):
            violations[f"issuer_short:{issuer}"] = float(
                -np.sum(vector[mask & short_mask]) - SHORT_CAP
            )
    for index, name in enumerate(("mktrf", "smb", "hml", "rmw", "cma", "umd")):
        violations[f"factor_positive:{name}"] = float(
            factor[index] - FACTOR_EXPOSURE_LIMIT
        )
        violations[f"factor_negative:{name}"] = float(
            -factor[index] - FACTOR_EXPOSURE_LIMIT
        )
    for sector in sorted(set(str(value) for value in data.sector.tolist())):
        mask = data.sector == sector
        value = float(np.sum(vector[mask]))
        violations[f"sector_net_positive:{sector}"] = value - SECTOR_NET_LIMIT
        violations[f"sector_net_negative:{sector}"] = -value - SECTOR_NET_LIMIT
        violations[f"sector_gross:{sector}"] = float(
            np.sum(np.abs(vector[mask])) - SECTOR_GROSS_LIMIT
        )
    for country in sorted(set(str(value) for value in data.country.tolist())):
        if country == "USA":
            continue
        value = float(np.sum(vector[data.country == country]))
        violations[f"country_positive:{country}"] = value - COUNTRY_NET_LIMIT
        violations[f"country_negative:{country}"] = -value - COUNTRY_NET_LIMIT
    if np.any(data.foreign_mask):
        value = float(np.sum(vector[data.foreign_mask]))
        violations["foreign_positive"] = value - FOREIGN_NET_LIMIT
        violations["foreign_negative"] = -value - FOREIGN_NET_LIMIT
    return violations


def _constraint_detection_evidence() -> dict[str, Any]:
    data = balanced_synthetic_problem()
    inherited = balanced_synthetic_problem(inherited_violation=True)
    required = inherited.required_exit.copy()
    required[[0, 4]] = True
    inherited = replace(inherited, required_exit=required)
    probes: list[tuple[object, np.ndarray]] = []
    all_sides = np.where(data.long_mask, 1.0, -1.0)
    probes.append((data, all_sides))
    single_long = np.zeros(data.n_assets)
    single_long[0] = 1.0
    probes.append((data, single_long))
    foreign = np.zeros(data.n_assets)
    foreign[2] = 0.20
    probes.append((data, foreign))
    wrong_sign = np.zeros(data.n_assets)
    wrong_sign[0] = -0.01
    probes.append((data, wrong_sign))
    increased_inherited = inherited.previous_weight.copy()
    increased_inherited[0] = 0.030
    probes.append((inherited, increased_inherited))
    probes.append((inherited, inherited.previous_weight.copy()))
    observed: dict[str, float] = {}
    for probe_data, weights in probes:
        values = independent_constraint_diagnostics(probe_data, weights).violations
        for name, value in values.items():
            observed[name] = max(observed.get(name, float("-inf")), float(value))
    families: Mapping[str, tuple[str, ...]] = {
        "gross": ("gross",),
        "net": ("net_positive", "net_negative"),
        "dynamic_net": ("dynamic_net_positive", "dynamic_net_negative"),
        "position": ("position_",),
        "issuer": ("issuer_",),
        "augmented_risk": ("augmented_risk",),
        "factor": ("factor_",),
        "sector": ("sector_",),
        "country": ("country_",),
        "aggregate_foreign": ("foreign_",),
        "turnover": ("turnover",),
        "trade_capacity": ("trade_capacity:",),
        "fixed_sign": ("sign:",),
        "reduction_only": ("reduction_only:",),
        "required_exit": ("required_exit_",),
    }
    detected: dict[str, bool] = {}
    maximum_detected_violation: dict[str, float] = {}
    for family, prefixes in families.items():
        matching = [
            value
            for name, value in observed.items()
            if any(name == prefix or name.startswith(prefix) for prefix in prefixes)
        ]
        maximum = max(matching, default=float("-inf"))
        detected[family] = maximum > 0.0
        maximum_detected_violation[family] = _finite_float(maximum)
    formula_weights = np.asarray(
        [0.023, -0.003, 0.18, 0.02, -0.024, 0.002, -0.17, -0.02],
        dtype=np.float64,
    )
    diagnostic_violations = independent_constraint_diagnostics(
        inherited,
        formula_weights,
    ).violations
    manual_violations = _manual_constraint_violations(inherited, formula_weights)
    formula_keys_match = set(diagnostic_violations) == set(manual_violations)
    formula_maximum_difference = max(
        (
            abs(diagnostic_violations[name] - manual_violations[name])
            for name in diagnostic_violations.keys() & manual_violations.keys()
        ),
        default=float("inf"),
    )

    ordinary = balanced_synthetic_problem()
    raw = np.where(ordinary.long_mask, 0.75, -0.75)
    enforced = solve_projection(
        ordinary,
        raw,
        architecture="all_constraint_enforcement_probe",
    )
    enforced_diagnostics = independent_constraint_diagnostics(
        ordinary,
        enforced.weights,
    )
    variable = cp.Variable(ordinary.n_assets)
    spec_names = {spec.name for spec in slackable_constraint_specs(ordinary, variable)}
    diagnostic_names = set(enforced_diagnostics.violations)
    required_spec_prefixes = {
        "gross": ("gross",),
        "net": ("net_positive", "net_negative"),
        "dynamic_net": ("dynamic_net_positive", "dynamic_net_negative"),
        "position": ("position_long:", "position_short:"),
        "issuer": ("issuer_long:", "issuer_short:"),
        "risk": ("augmented_risk",),
        "factor": ("factor_positive:", "factor_negative:"),
        "sector": ("sector_net_", "sector_gross:"),
        "country": ("country_positive:", "country_negative:"),
        "foreign": ("foreign_positive", "foreign_negative"),
    }
    spec_family_present = {
        family: all(
            any(name == prefix or name.startswith(prefix) for name in spec_names)
            for prefix in prefixes
        )
        for family, prefixes in required_spec_prefixes.items()
    }
    hard_family_present = {
        family: any(
            name == prefix or name.startswith(prefix)
            for name in diagnostic_names
            for prefix in prefixes
        )
        for family, prefixes in {
            "sign": ("sign:",),
            "trade_capacity": ("trade_capacity:",),
            "adv_capacity": ("adv_capacity:",),
            "turnover": ("turnover",),
        }.items()
    }
    checks = {
        "every_family_detects_a_deliberate_violation": all(detected.values()),
        "independent_formula_key_set_exact": formula_keys_match,
        "independent_formula_values_exact": formula_maximum_difference <= 1e-14,
        "every_slackable_formula_is_installed": all(spec_family_present.values()),
        "every_always_hard_formula_is_audited": all(hard_family_present.values()),
        "ordinary_solver_enforces_all_installed_constraints": (
            enforced.mode == "ordinary"
            and enforced_diagnostics.maximum_violation <= INDEPENDENT_ABSOLUTE_TOLERANCE
        ),
        "ordinary_solver_has_no_slack": not bool(enforced.normalized_slacks),
    }
    if not _checks_pass(checks):
        raise Stage10EvidenceError(f"constraint detector gate failed: {checks}")
    return {
        "fixture_identifiers_are_fictitious": True,
        "families": list(families),
        "detected": detected,
        "maximum_detected_violation": maximum_detected_violation,
        "manual_formula_key_count": len(manual_violations),
        "diagnostic_formula_key_count": len(diagnostic_violations),
        "manual_formula_maximum_absolute_difference": _finite_float(
            formula_maximum_difference
        ),
        "slackable_formula_families_present": spec_family_present,
        "always_hard_formula_families_present": hard_family_present,
        "solver_enforcement": _result_summary(enforced),
        "checks": checks,
    }


def _empty_problem(data: AllocationProblemData) -> AllocationProblemData:
    return replace(
        data,
        permno=data.permno[:0],
        permco=data.permco[:0],
        side=data.side[:0],
        previous_weight=data.previous_weight[:0],
        reduction_only=data.reduction_only[:0],
        required_exit=data.required_exit[:0],
        expected_return=data.expected_return[:0],
        score=data.score[:0],
        factor_betas=data.factor_betas[:0, :],
        specific_variance_monthly=data.specific_variance_monthly[:0],
        forecast_volatility=data.forecast_volatility[:0],
        full_spread=data.full_spread[:0],
        adv20_usd=data.adv20_usd[:0],
        impact_volatility_daily=data.impact_volatility_daily[:0],
        borrow_liquidity_quintile=data.borrow_liquidity_quintile[:0],
        trade_capacity_weight=data.trade_capacity_weight[:0],
        sector=data.sector[:0],
        country=data.country[:0],
        mandatory_corporate_action=data.mandatory_corporate_action[:0],
    )


def _required_exit_problem(*, constrained_capacity: bool) -> AllocationProblemData:
    data = balanced_synthetic_problem(zero_alpha=True)
    previous = data.previous_weight.copy()
    reduction = data.reduction_only.copy()
    required = data.required_exit.copy()
    capacity = data.trade_capacity_weight.copy()
    previous[[0, 4]] = [0.005, -0.005]
    reduction[[0, 4]] = True
    required[[0, 4]] = True
    if constrained_capacity:
        capacity[[0, 4]] = 0.001
    return replace(
        data,
        previous_weight=previous,
        reduction_only=reduction,
        required_exit=required,
        trade_capacity_weight=capacity,
    )


def _solver_flow_evidence() -> dict[str, Any]:
    """Exercise ordinary, all-cash, exact-infeasible and emergency behavior."""

    data = balanced_synthetic_problem()
    active = solve_hybrid(data)
    cash = solve_hybrid(balanced_synthetic_problem(zero_alpha=True))
    empty = _empty_problem(data)
    empty_hybrid = solve_hybrid(empty)
    empty_projection = solve_projection(
        empty,
        np.empty(0, dtype=np.float64),
        architecture="empty_projection",
    )

    required_ordinary_data = _required_exit_problem(constrained_capacity=False)
    required_ordinary = solve_hybrid(required_ordinary_data)
    required_emergency_data = _required_exit_problem(constrained_capacity=True)
    required_emergency = solve_hybrid(required_emergency_data)

    reduction_previous = data.previous_weight.copy()
    reduction_mask = data.reduction_only.copy()
    reduction_previous[[0, 4]] = [0.005, -0.005]
    reduction_mask[[0, 4]] = True
    reduction_data = replace(
        data,
        previous_weight=reduction_previous,
        reduction_only=reduction_mask,
    )
    reduction_result = solve_hybrid(reduction_data)

    emergency_data = balanced_synthetic_problem(inherited_violation=True)
    emergency_positive_raw = solve_projection(
        emergency_data,
        np.full(emergency_data.n_assets, 0.25),
        architecture="emergency_objective_positive_raw",
    )
    emergency_negative_raw = solve_projection(
        emergency_data,
        np.full(emergency_data.n_assets, -0.25),
        architecture="emergency_objective_negative_raw",
    )
    pretrade = independent_constraint_diagnostics(
        emergency_data,
        emergency_data.previous_weight,
    )
    posttrade = independent_constraint_diagnostics(
        emergency_data,
        emergency_positive_raw.weights,
    )
    new_or_worsened = sorted(
        name
        for name, observed in posttrade.normalized_violations.items()
        if observed
        > max(pretrade.normalized_violations.get(name, 0.0), 0.0) + FORWARD_TOLERANCE
    )

    required_history = required_emergency.solver_status_history
    ordinary_prefix = required_history[:3]
    emergency_suffix = required_history[3:]
    required_indices = np.flatnonzero(required_emergency_data.required_exit)
    exact_required_targets = bool(
        np.max(np.abs(required_ordinary.weights[required_indices]))
        <= INDEPENDENT_ABSOLUTE_TOLERANCE
    )
    constrained_expected = np.asarray([0.004, -0.004])
    constrained_actual = required_emergency.weights[required_indices]
    checks = {
        "hybrid_active_ordinary": active.mode == "ordinary"
        and active.diagnostics.gross > 0.0,
        "hybrid_zero_alpha_allows_cash": bool(np.max(np.abs(cash.weights)) <= 1e-8),
        "zero_asset_hybrid_bypasses_to_cash": (
            empty_hybrid.weights.size == 0
            and empty_hybrid.cash_weight == 1.0
            and empty_hybrid.solver_status_history == ("all_cash_no_assets:optimal",)
        ),
        "zero_asset_projection_bypasses_to_cash": (
            empty_projection.weights.size == 0
            and empty_projection.cash_weight == 1.0
            and empty_projection.solver_status_history
            == ("all_cash_no_assets:optimal",)
        ),
        "required_exit_exact_zero_in_ordinary": required_ordinary.mode == "ordinary"
        and exact_required_targets,
        "required_exit_enters_emergency_when_capacity_blocks_zero": bool(
            required_emergency.mode == "emergency"
            and np.max(np.abs(constrained_actual - constrained_expected))
            <= INDEPENDENT_ABSOLUTE_TOLERANCE
        ),
        "ordinary_waterfall_has_two_exact_infeasible_objective_statuses": (
            len(ordinary_prefix) == 3
            and ordinary_prefix[0] == "ordinary_primary:infeasible"
            and ordinary_prefix[1] == "ordinary_retry:infeasible"
        ),
        "emergency_requires_exact_feasibility_infeasibility_certificate": (
            len(ordinary_prefix) == 3
            and ordinary_prefix[2] == "ordinary_feasibility_retry:infeasible"
        ),
        "emergency_lexicographic_steps_all_exact_optimal": (
            any(value == "emergency_minimax:optimal" for value in emergency_suffix)
            and any(value == "emergency_sum:optimal" for value in emergency_suffix)
            and emergency_suffix[-1].endswith(":optimal")
        ),
        "reduction_only_numerical_case_exact_optimal": (
            reduction_result.solver_status == "optimal"
            and reduction_result.diagnostics.maximum_violation
            <= INDEPENDENT_ABSOLUTE_TOLERANCE
            and reduction_result.transaction_cost > 0.0
        ),
        "challenger_emergency_uses_economic_not_raw_objective": bool(
            np.max(
                np.abs(emergency_positive_raw.weights - emergency_negative_raw.weights)
            )
            <= INDEPENDENT_ABSOLUTE_TOLERANCE
        ),
        "emergency_creates_or_worsens_no_violation": not bool(new_or_worsened),
        "emergency_reduction_only_never_increases": bool(
            np.all(
                np.abs(emergency_positive_raw.weights[emergency_data.reduction_only])
                <= np.abs(emergency_data.previous_weight[emergency_data.reduction_only])
                + FORWARD_TOLERANCE
            )
        ),
        "emergency_reports_slacks": bool(emergency_positive_raw.normalized_slacks),
        "emergency_reports_binding_trade_limits": bool(
            emergency_positive_raw.binding_trade_limits
        ),
        "emergency_reports_cash_exactly": abs(
            emergency_positive_raw.cash_weight
            - (1.0 - emergency_positive_raw.diagnostics.net)
        )
        <= 1e-15,
        "emergency_reports_compliance_horizon": (
            isinstance(emergency_positive_raw.estimated_months_to_compliance, int)
            and emergency_positive_raw.estimated_months_to_compliance != 0
        ),
        "required_exit_compliance_horizon_is_four_months": (
            required_emergency.estimated_months_to_compliance == 4
        ),
    }
    if not _checks_pass(checks):
        raise Stage10EvidenceError(f"solver-flow synthetic gate failed: {checks}")
    return {
        "active": _result_summary(active),
        "cash": _result_summary(cash),
        "empty_hybrid": _result_summary(empty_hybrid),
        "empty_projection": _result_summary(empty_projection),
        "required_exit_ordinary": _result_summary(required_ordinary),
        "required_exit_emergency": _result_summary(required_emergency),
        "numerical_reduction_only": _result_summary(reduction_result),
        "emergency_positive_raw": _result_summary(emergency_positive_raw),
        "emergency_negative_raw": _result_summary(emergency_negative_raw),
        "emergency_new_or_worsened_violation_names": new_or_worsened,
        "checks": checks,
    }


def _decision_focused_evidence() -> dict[str, Any]:
    data = balanced_synthetic_problem()
    sectors = tuple(sorted(set(str(value) for value in data.sector.tolist())))
    countries = tuple(sorted(set(str(value) for value in data.country.tolist())))
    model = DecisionFocusedAllocator(
        sector_vocabulary=sectors,
        country_vocabulary=countries,
        seed=11,
    ).double()
    model.eval()
    forward = model.raw_allocation(data)
    fixed_raw = torch.tensor(
        [0.004, 0.003, 0.002, 0.001, -0.004, -0.003, -0.002, -0.001],
        dtype=torch.float64,
        requires_grad=True,
    )
    layer = DifferentiableProjection(data)
    seeded_models = tuple(
        item.double().eval()
        for item in build_seeded_allocators(
            sector_vocabulary=sectors,
            country_vocabulary=countries,
        )
    )
    inference = project_seed_ensemble(
        seeded_models,
        data,
        layer,
    )
    ensemble_vector = inference.weights
    seed_weights = tuple(item.projected.weights for item in inference.seed_results)
    ensemble_diagnostics = independent_constraint_diagnostics(data, ensemble_vector)
    layer_weights = layer.project_for_training(fixed_raw)
    clarabel = solve_projection(
        data,
        fixed_raw.detach().cpu().numpy(),
        architecture="decision_focused_neural_projected",
    )
    layer_vector = layer_weights.detach().cpu().numpy()
    forward_difference = float(np.max(np.abs(layer_vector - clarabel.weights)))
    layer_diagnostics = independent_constraint_diagnostics(data, layer_vector)

    coefficients = torch.tensor(
        [0.7, -0.2, 0.1, -0.3, 0.4, -0.6, 0.2, -0.1],
        dtype=torch.float64,
    )
    torch.dot(layer_weights, coefficients).backward()
    if fixed_raw.grad is None:
        raise Stage10EvidenceError("differentiable projection returned no gradient")
    analytic = fixed_raw.grad.detach().cpu().numpy()
    epsilon = 1e-5
    finite = np.empty(data.n_assets, dtype=np.float64)
    baseline = fixed_raw.detach()
    for index in range(data.n_assets):
        step = torch.zeros_like(baseline)
        step[index] = epsilon
        above = torch.dot(layer.project_for_training(baseline + step), coefficients)
        below = torch.dot(layer.project_for_training(baseline - step), coefficients)
        finite[index] = float((above - below).detach().cpu().item() / (2.0 * epsilon))
    absolute = np.abs(analytic - finite)
    scale = np.maximum(np.maximum(np.abs(analytic), np.abs(finite)), 1e-12)
    relative = absolute / scale
    maximum_absolute = float(np.max(absolute))
    maximum_relative = float(np.max(relative))

    # An SCS failure must deactivate training, while executable inference must
    # remain CLARABEL-only and never touch that failed training layer.
    class _FailingTrainingLayer(torch.nn.Module):
        def forward(self, *_args: object, **_kwargs: object) -> torch.Tensor:
            raise RuntimeError("synthetic SCS failure")

    original_layer = layer.layer
    layer.layer = _FailingTrainingLayer()
    scs_failed_closed, scs_failure = _raises(
        DecisionFocusedInactiveError,
        layer.project_for_training,
        baseline,
        message_contains="SCS training projection failed",
    )
    clarabel_without_scs = project_seed_ensemble(seeded_models, data, layer)
    layer.layer = original_layer

    zero_previous = data.previous_weight.copy()
    zero_previous[0] = 0.004
    zero_adv = data.adv20_usd.copy()
    zero_capacity = data.trade_capacity_weight.copy()
    zero_adv[0] = np.nan
    zero_capacity[0] = 0.0
    zero_problem = replace(
        data,
        previous_weight=zero_previous,
        adv20_usd=zero_adv,
        trade_capacity_weight=zero_capacity,
    )
    zero_projection = DifferentiableProjection(zero_problem)
    zero_raw = fixed_raw.detach().clone().requires_grad_(True)
    zero_weights = zero_projection.project_for_training(zero_raw)
    zero_coordinate_exact = bool(
        zero_weights[0].detach().cpu().item() == zero_previous[0]
    )

    # Independently replay daily fixed-share accounting, including one-time
    # transaction cost and daily risk-free/borrow/financing accrual.
    accounting_weights = torch.tensor(
        [0.24, 0.22, 0.20, 0.18, -0.13, -0.12, -0.11, -0.10],
        dtype=torch.float64,
        requires_grad=True,
    )
    daily_returns = np.asarray(
        [
            [0.010, -0.004, 0.003, 0.001, -0.006, 0.002, -0.001, 0.004],
            [-0.002, 0.005, 0.000, 0.003, 0.004, -0.003, 0.002, -0.005],
            [0.004, 0.001, -0.002, 0.006, -0.001, 0.003, -0.004, 0.002],
        ],
        dtype=np.float64,
    )
    daily_rf = np.asarray([0.00010, 0.00011, 0.00009], dtype=np.float64)
    accounting = torch_daily_holding_accounting(
        data,
        accounting_weights,
        daily_returns,
        daily_rf,
    )
    transaction = numpy_total_cost(
        accounting_weights.detach().cpu().numpy(),
        data.previous_weight,
        data.full_spread,
        data.adv20_usd,
        data.impact_volatility_daily,
        data.borrow_rate_annual,
        data.nav_dollars,
    ).transaction
    manual_holdings = accounting_weights.detach().cpu().numpy().copy()
    manual_cash = 1.0 - float(np.sum(manual_holdings)) - transaction
    manual_nav = float(np.sum(manual_holdings) + manual_cash)
    manual_borrow = 0.0
    manual_financing = 0.0
    for day_return, risk_free in zip(daily_returns, daily_rf, strict=True):
        borrow_charge = float(
            np.sum(np.maximum(-manual_holdings, 0.0) * data.borrow_rate_annual) / 252.0
        )
        financing_charge = (
            max(float(np.sum(np.abs(manual_holdings))) - manual_nav, 0.0)
            * 0.005
            / 252.0
        )
        manual_holdings *= 1.0 + day_return
        manual_cash = manual_cash * (1.0 + risk_free) - borrow_charge - financing_charge
        manual_nav = float(np.sum(manual_holdings) + manual_cash)
        manual_borrow += borrow_charge
        manual_financing += financing_charge
    manual_rf_total = float(np.prod(1.0 + daily_rf) - 1.0)
    manual_net_excess = manual_nav - 1.0 - manual_rf_total
    accounting_loss = -accounting.net_excess_return
    accounting_loss.backward()

    # Exercise the data-source-agnostic chronological lifecycle from cash,
    # stopped drifted priors, quarterly cadence, half-life weighting, held-out
    # early stopping, reset, and full-history refit.
    months = tuple(
        DecisionTrainingMonth(
            month_end=value,
            problem=data,
            daily_total_returns=daily_returns[:1] * scale,
            daily_risk_free_rates=daily_rf[:1],
        )
        for value, scale in (
            ("2014-01-31", 0.8),
            ("2014-02-28", 1.0),
            ("2014-03-31", 1.2),
        )
    )
    observed_priors: list[np.ndarray] = []

    class _SyntheticLifecycleProjection:
        def __init__(self, problem: AllocationProblemData) -> None:
            self.problem = problem

        def project_for_training(self, raw_target: torch.Tensor) -> torch.Tensor:
            return 0.005 * raw_target

    def lifecycle_factory(
        problem: AllocationProblemData,
    ) -> _SyntheticLifecycleProjection:
        observed_priors.append(problem.previous_weight.copy())
        return _SyntheticLifecycleProjection(problem)

    lifecycle_model = (
        DecisionFocusedAllocator(
            sector_vocabulary=sectors,
            country_vocabulary=countries,
            seed=11,
        )
        .double()
        .eval()
    )
    lifecycle = run_chronological_training_sequence(
        lifecycle_model,
        months,
        projection_factory=lifecycle_factory,
    )
    lifecycle_training = train_decision_focused_seed(
        months,
        refit_date="2014-03-31",
        sector_vocabulary=sectors,
        country_vocabulary=countries,
        seed=11,
        synthetic_test_mode=True,
        maximum_epochs_for_test=2,
        patience_for_test=1,
        validation_months_for_test=1,
        projection_factory=_SyntheticLifecycleProjection,
    )
    unsorted_rejected, unsorted_exception = _raises(
        ValueError,
        run_chronological_training_sequence,
        lifecycle_model,
        tuple(reversed(months)),
        projection_factory=lifecycle_factory,
        message_contains="strictly chronological",
    )
    monthly_refit_rejected, refit_exception = _raises(
        ValueError,
        validate_quarterly_refit_date,
        "2014-02-28",
        message_contains="quarterly",
    )
    decade_months = (
        DecisionTrainingMonth("2004-03-31", data, daily_returns[:1], daily_rf[:1]),
        DecisionTrainingMonth("2014-03-31", data, daily_returns[:1], daily_rf[:1]),
    )
    memory = weighted_expanding_month_weights(
        decade_months,
        refit_date="2014-03-31",
    )

    checks = {
        "fixed_forward_finite": bool(
            torch.isfinite(forward.raw_weights).all().detach().cpu().item()
        ),
        "five_frozen_seeds_exercised": (
            tuple(item.seed for item in seeded_models) == DECISION_SEEDS
            and len(seed_weights) == len(DECISION_SEEDS)
        ),
        "seed_ensemble_final_projection_finite": bool(
            np.isfinite(ensemble_vector).all()
        ),
        "seed_ensemble_final_projection_feasible": (
            ensemble_diagnostics.maximum_violation <= TRAINING_PROJECTION_TOLERANCE
        ),
        "two_gates_in_unit_interval": bool(
            0.0 <= float(forward.long_gate) <= 1.0
            and 0.0 <= float(forward.short_gate) <= 1.0
        ),
        "common_projection_equivalent": forward_difference <= 2e-6,
        "training_projection_feasible": (
            layer_diagnostics.maximum_violation <= TRAINING_PROJECTION_TOLERANCE
        ),
        "analytic_gradient_finite": bool(np.isfinite(analytic).all()),
        "finite_difference_gradient_finite": bool(np.isfinite(finite).all()),
        "gradient_absolute_gate": maximum_absolute <= GRADIENT_ABSOLUTE_TOLERANCE,
        "gradient_relative_gate": maximum_relative <= GRADIENT_RELATIVE_TOLERANCE,
        "scs_training_failure_fails_closed": scs_failed_closed,
        "clarabel_inference_does_not_evaluate_scs_layer": bool(
            np.isfinite(clarabel_without_scs.weights).all()
            and all(
                item.projected.mode == "ordinary"
                for item in clarabel_without_scs.seed_results
            )
            and clarabel_without_scs.final_projection.mode == "ordinary"
        ),
        "zero_capacity_training_coordinate_is_exact_pretrade": zero_coordinate_exact,
        "zero_capacity_training_projection_remains_feasible": (
            independent_constraint_diagnostics(
                zero_problem,
                zero_weights.detach().cpu().numpy(),
            ).maximum_violation
            <= TRAINING_PROJECTION_TOLERANCE
        ),
        "daily_accounting_matches_independent_nav_replay": abs(
            accounting.net_excess_return.detach().cpu().item() - manual_net_excess
        )
        <= 1e-12,
        "daily_borrow_accrual_matches_independent_replay": abs(
            accounting.borrow_cost.detach().cpu().item() - manual_borrow
        )
        <= 1e-12,
        "daily_financing_accrual_matches_independent_replay": abs(
            accounting.financing_cost.detach().cpu().item() - manual_financing
        )
        <= 1e-12,
        "daily_transaction_cost_charged_once": abs(
            accounting.transaction_cost.detach().cpu().item() - transaction
        )
        <= 1e-12,
        "daily_accounting_gradient_finite": bool(
            accounting_weights.grad is not None
            and torch.isfinite(accounting_weights.grad).all().detach().cpu().item()
        ),
        "training_sequence_starts_from_cash": bool(
            observed_priors
            and np.array_equal(observed_priors[0], np.zeros(data.n_assets))
        ),
        "training_sequence_uses_stopped_drifted_prior": bool(
            len(observed_priors) == len(months)
            and np.allclose(
                observed_priors[1],
                lifecycle.terminal_weights[0].detach().cpu().numpy(),
                rtol=0.0,
                atol=1e-15,
            )
            and all(not value.requires_grad for value in lifecycle.terminal_weights)
        ),
        "training_sequence_rejects_nonchronological_months": unsorted_rejected,
        "training_refit_rejects_nonquarterly_month": monthly_refit_rejected,
        "weighted_expanding_half_life_is_ten_years": abs(
            float(memory[1] / memory[0]) - 2.0
        )
        <= 1e-12,
        "early_stopping_then_full_history_refit_executed": (
            lifecycle_training.synthetic_test_mode
            and 1 <= lifecycle_training.selected_epoch_count <= 2
            and len(lifecycle_training.refit_training_losses)
            == lifecycle_training.selected_epoch_count
            and len(lifecycle_training.early_stopping_validation_losses) >= 1
        ),
    }
    if not _checks_pass(checks):
        raise Stage10EvidenceError(f"decision-focused synthetic gate failed: {checks}")
    return {
        "fixture_identifiers_are_fictitious": True,
        "network_seed": 11,
        "ensemble_seeds": list(DECISION_SEEDS),
        "input_width": model.input_width,
        "long_gate": _finite_float(forward.long_gate.detach().cpu().item()),
        "short_gate": _finite_float(forward.short_gate.detach().cpu().item()),
        "raw_weights_sha256": _vector_hash(forward.raw_weights.detach().cpu().numpy()),
        "seed_projected_weights_sha256": [
            _vector_hash(value) for value in seed_weights
        ],
        "ensemble_final_weights_sha256": _vector_hash(ensemble_vector),
        "ensemble_final_maximum_constraint_violation": _finite_float(
            ensemble_diagnostics.maximum_violation
        ),
        "layer_weights_sha256": _vector_hash(layer_vector),
        "clarabel_weights_sha256": _vector_hash(clarabel.weights),
        "layer_vs_clarabel_maximum_absolute_difference": forward_difference,
        "layer_maximum_constraint_violation": _finite_float(
            layer_diagnostics.maximum_violation
        ),
        "analytic_gradient_sha256": _vector_hash(analytic),
        "finite_difference_gradient_sha256": _vector_hash(finite),
        "gradient_maximum_absolute_difference": maximum_absolute,
        "gradient_maximum_relative_difference": maximum_relative,
        "scs_failure_exception": scs_failure,
        "zero_capacity_training_weight": _finite_float(
            zero_weights[0].detach().cpu().item()
        ),
        "clarabel_inference_final": _result_summary(
            clarabel_without_scs.final_projection
        ),
        "daily_accounting": {
            "net_excess_return": _finite_float(
                accounting.net_excess_return.detach().cpu().item()
            ),
            "manual_net_excess_return": _finite_float(manual_net_excess),
            "transaction_cost": _finite_float(
                accounting.transaction_cost.detach().cpu().item()
            ),
            "borrow_cost": _finite_float(accounting.borrow_cost.detach().cpu().item()),
            "financing_cost": _finite_float(
                accounting.financing_cost.detach().cpu().item()
            ),
            "terminal_weights_sha256": _vector_hash(
                accounting.terminal_weights.detach().cpu().numpy()
            ),
        },
        "training_lifecycle": {
            "synthetic_lifecycle_only": not PRODUCTION_TRAINING_RUNTIME_CERTIFIED,
            "empirical_production_training_runtime_certified": bool(
                PRODUCTION_TRAINING_RUNTIME_CERTIFIED
            ),
            "empirical_production_training_limitation": (
                "inactive pending Stage 11 dynamic candidate/problem adapter and "
                "certified reusable projection runtime"
            ),
            "chronological_month_count": len(months),
            "first_prior_weights_sha256": _vector_hash(observed_priors[0]),
            "second_prior_weights_sha256": _vector_hash(observed_priors[1]),
            "first_terminal_weights_sha256": _vector_hash(
                lifecycle.terminal_weights[0].detach().cpu().numpy()
            ),
            "selected_epoch_count": lifecycle_training.selected_epoch_count,
            "early_stopping_observations": len(
                lifecycle_training.early_stopping_validation_losses
            ),
            "full_history_refit_epochs": len(lifecycle_training.refit_training_losses),
            "ten_year_memory_weights": [float(value) for value in memory],
            "unsorted_rejection_exception": unsorted_exception,
            "nonquarterly_rejection_exception": refit_exception,
        },
        "checks": checks,
    }


def build_synthetic_allocator_evidence() -> dict[str, Any]:
    """Run and return the canonical Stage 10 synthetic evidence contract."""

    counters = BoundaryCounters()
    enforce_stage10_boundary(counters, synthetic_fixture=True)
    data = balanced_synthetic_problem()
    active = solve_hybrid(data)
    cash = solve_hybrid(balanced_synthetic_problem(zero_alpha=True))
    rank_raw, rank_result = allocate_rank(
        data,
        long_entry_boundary=0.10,
        short_entry_boundary=-0.10,
    )
    rank_common_replay = solve_projection(
        data,
        rank_raw.weights,
        architecture="rank_volatility_projected",
    )
    common_raw = np.asarray(
        [0.004, 0.003, 0.002, 0.001, -0.004, -0.003, -0.002, -0.001]
    )
    common_first = solve_projection(data, common_raw, architecture="common_projection")
    common_replay = solve_projection(data, common_raw, architecture="common_projection")
    emergency_data = balanced_synthetic_problem(inherited_violation=True)
    emergency = solve_projection(
        emergency_data,
        emergency_data.previous_weight,
        architecture="synthetic_emergency_projection",
    )
    emergency_pretrade = independent_constraint_diagnostics(
        emergency_data,
        emergency_data.previous_weight,
    )
    emergency_posttrade = independent_constraint_diagnostics(
        emergency_data,
        emergency.weights,
    )
    emergency_new_or_worsened = sorted(
        name
        for name, observed in emergency_posttrade.normalized_violations.items()
        if observed
        > max(emergency_pretrade.normalized_violations.get(name, 0.0), 0.0)
        + FORWARD_TOLERANCE
    )
    common_weight_difference = float(
        np.max(np.abs(common_first.weights - common_replay.weights))
    )
    common_objective_difference = abs(
        float(common_first.objective_value) - float(common_replay.objective_value)
    )
    ordinary_results = (
        active,
        cash,
        rank_result,
        rank_common_replay,
        common_first,
        common_replay,
    )
    architecture_checks = {
        "hybrid_active_ordinary": active.mode == "ordinary"
        and active.diagnostics.gross > 0.0,
        "hybrid_cash_when_alpha_cannot_cover_costs": bool(
            np.max(np.abs(cash.weights)) <= 1e-8
        ),
        "rank_equal_raw_legs": abs(
            float(np.sum(rank_raw.weights[data.long_mask]))
            + float(np.sum(rank_raw.weights[data.short_mask]))
        )
        <= 1e-12,
        "rank_risk_scaled": (
            float(np.sqrt(12.0) * np.linalg.norm(data.risk_root @ rank_raw.weights))
            <= 0.10 + 1e-12
        ),
        "rank_matches_direct_common_projection": bool(
            np.max(np.abs(rank_result.weights - rank_common_replay.weights)) <= 1e-9
        ),
        "ordinary_cases_have_no_slacks": all(
            not result.normalized_slacks for result in ordinary_results
        ),
        "ordinary_cases_independently_feasible": all(
            result.diagnostics.maximum_violation <= FORWARD_TOLERANCE
            for result in ordinary_results
        ),
        "common_projection_deterministic_weights": common_weight_difference <= 1e-9,
        "common_projection_deterministic_objective": common_objective_difference
        <= 1e-10,
        "emergency_entered": emergency.mode == "emergency",
        "emergency_has_inherited_slacks": bool(emergency.normalized_slacks),
        "emergency_has_no_new_or_worsened_violation": not emergency_new_or_worsened,
        "emergency_did_not_increase_reduction_rows": bool(
            np.all(
                np.abs(emergency.weights[emergency_data.reduction_only])
                <= np.abs(emergency_data.previous_weight[emergency_data.reduction_only])
                + FORWARD_TOLERANCE
            )
        ),
        "emergency_respected_trade_capacity": bool(
            np.all(
                np.abs(emergency.weights - emergency_data.previous_weight)
                <= emergency_data.trade_capacity_weight + FORWARD_TOLERANCE
            )
        ),
    }
    if not _checks_pass(architecture_checks):
        raise Stage10EvidenceError(
            f"allocator architecture synthetic gate failed: {architecture_checks}"
        )
    decision_evidence = _decision_focused_evidence()
    candidate_evidence = _candidate_evidence()
    cost_evidence = _cost_evidence()
    domain_risk_evidence = _domain_and_risk_evidence()
    constraint_evidence = _constraint_detection_evidence()
    solver_flow_evidence = _solver_flow_evidence()
    counter_values = counters.to_dict()
    scope_zero = all(value == 0 for value in counter_values.values())
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "SYNTHETIC_ALLOCATOR_EVIDENCE_PASSED",
        "scope": {
            "stage": 10,
            "synthetic_only": scope_zero,
            "fixture_identifiers_are_fictitious": bool(
                np.all((data.permno >= 900001) & (data.permno <= 900008))
            ),
            **counter_values,
            "sharpe_or_performance_report_created": False,
            "selector_refit_or_selection_performed": False,
            "ensemble_weight_selection_performed": False,
            "decision_focused_real_data_training_performed": False,
            "decision_focused_empirical_production_training_ready": bool(
                PRODUCTION_TRAINING_RUNTIME_CERTIFIED
            ),
        },
        "architectures": {
            "hybrid_active": _result_summary(active),
            "hybrid_cash": _result_summary(cash),
            "rank_raw": {
                "weights_sha256": _vector_hash(rank_raw.weights),
                "long_strength_sum": _finite_float(rank_raw.long_strength_sum),
                "short_strength_sum": _finite_float(rank_raw.short_strength_sum),
                "unscaled_augmented_annual_volatility": _finite_float(
                    rank_raw.unscaled_augmented_annual_volatility
                ),
                "scale": _finite_float(rank_raw.scale),
            },
            "rank_projected": _result_summary(rank_result),
            "rank_direct_common_projection_replay": _result_summary(rank_common_replay),
            "common_projection": _result_summary(common_first),
            "common_projection_replay": _result_summary(common_replay),
            "emergency_projection": _result_summary(emergency),
            "emergency_new_or_worsened_violation_names": emergency_new_or_worsened,
            "decision_focused": decision_evidence,
            "checks": architecture_checks,
        },
        "candidate_and_borrow": candidate_evidence,
        "costs": cost_evidence,
        "authoritative_domain_and_risk": domain_risk_evidence,
        "constraint_family_detection": constraint_evidence,
        "solver_flow": solver_flow_evidence,
    }
    evidence["gates"] = {
        "only_synthetic_allocator_cases_executed": scope_zero,
        "all_three_architecture_contracts_executed": (
            active.architecture == "hybrid_alpha_risk_cost_convex"
            and rank_result.architecture == "rank_volatility_projected"
            and decision_evidence["clarabel_inference_final"]["architecture"]
            == "decision_focused_neural_projected"
        ),
        "ordinary_case_slack_count_zero": architecture_checks[
            "ordinary_cases_have_no_slacks"
        ],
        "candidate_joint_repair_and_borrow_mapping_pass": _checks_pass(
            candidate_evidence["checks"]
        ),
        "authoritative_risk_and_adv_derivation_pass": _checks_pass(
            domain_risk_evidence["checks"]
        ),
        "all_constraint_formulas_and_solver_enforcement_pass": _checks_pass(
            constraint_evidence["checks"]
        ),
        "transaction_borrow_financing_backend_parity_pass": _checks_pass(
            cost_evidence["checks"]
        ),
        "ordinary_all_cash_required_exit_and_emergency_flow_pass": _checks_pass(
            solver_flow_evidence["checks"]
        ),
        "rank_projection_matches_common_projection": architecture_checks[
            "rank_matches_direct_common_projection"
        ],
        "decision_focused_training_inference_accounting_lifecycle_pass": _checks_pass(
            decision_evidence["checks"]
        ),
        "deterministic_replay_pass": (
            architecture_checks["common_projection_deterministic_weights"]
            and architecture_checks["common_projection_deterministic_objective"]
            and candidate_evidence["checks"]["joint_repair_is_deterministic"]
        ),
    }
    evidence["pending_gates"] = {
        "full_repository_regression": "PENDING_EXTERNAL_TEST_RUN",
        "independent_verification": "PENDING_PRIMARY_PUBLICATION",
    }
    evidence["limitations"] = {
        "decision_focused_empirical_production_training": (
            "INACTIVE_PENDING_STAGE11_DYNAMIC_ADAPTER_AND_CERTIFIED_REUSABLE_PROJECTION"
        )
    }
    if not _checks_pass(evidence["gates"]):
        raise Stage10EvidenceError("one or more final synthetic evidence gates failed")
    return evidence


def synthetic_evidence_sha256(evidence: Mapping[str, Any]) -> str:
    """Return the canonical identity of a completed evidence dictionary."""

    return hashlib.sha256(canonical_json_bytes(dict(evidence))).hexdigest()


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "Stage10EvidenceError",
    "build_synthetic_allocator_evidence",
    "synthetic_evidence_sha256",
]
