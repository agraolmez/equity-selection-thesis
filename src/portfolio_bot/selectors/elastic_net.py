"""Conditional Elastic Net path for the Stage 9 linear comparator."""

from __future__ import annotations

import math
import importlib.metadata
import hashlib
import os
import platform
import re
import stat
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from portfolio_bot.selectors.contract import validate_runtime_versions
from portfolio_bot.selectors.model_specs import (
    ELASTIC_NET_LAMBDA_POINTS,
    ELASTIC_NET_RHO_GRID,
    ElasticNetConfig,
    elastic_net_configurations,
    elastic_net_lambda_value,
)
from portfolio_bot.selectors.native_process import (
    configure_single_thread_environment,
    require_spawned_native_worker,
)

configure_single_thread_environment()
require_spawned_native_worker("elastic_net")

import numpy as np
from scipy.linalg import lstsq, solve_triangular
from sklearn.linear_model import enet_path
from sklearn.exceptions import ConvergenceWarning
from skglm.datafits import QuadraticHessian
from skglm.penalties import L1_plus_L2
from skglm.solvers import AndersonCD
from threadpoolctl import threadpool_limits

from portfolio_bot.selectors.cache import (
    job_id_from_identity,
    load_completed_job,
    publish_completed_job,
)


RHO_GRID = ELASTIC_NET_RHO_GRID
LAMBDA_POINTS = ELASTIC_NET_LAMBDA_POINTS
MAX_ITERATIONS = 20_000
TOLERANCE = 1e-6
NONZERO_THRESHOLD = 1e-12
MAX_IN_MEMORY_WEIGHTED_DESIGN_BYTES = 512 * 1024 * 1024
_AUTHORITATIVE_PARENT_JOB = re.compile(r"stage09-fit-[0-9a-f]{64}\Z")
_LEGACY_WEIGHTED_SUFFIX = ".weighted-f32.memmap"
_FLOAT64_WEIGHTED_SUFFIX = ".weighted-f64.memmap"
_CHECKPOINT_DIRECTORY = "rho_checkpoints"
_CHECKPOINT_PURPOSE = "stage09_elastic_net_authenticated_rho_resume"
_CHECKPOINT_STATUS = "AUTHENTICATED_PARTIAL_RHO_ONLY_NOT_PUBLISHABLE"
_CACHE_READ_ATTEMPTS = 3
_SKGLM_REQUIRED_VERSION = "0.5"
_SKGLM_OBSERVED_VERSION = importlib.metadata.version("skglm")
_NUMBA_REQUIRED_VERSION = "0.61.0"
_NUMBA_OBSERVED_VERSION = importlib.metadata.version("numba")
_LLVMLITE_REQUIRED_VERSION = "0.44.0"
_LLVMLITE_OBSERVED_VERSION = importlib.metadata.version("llvmlite")
_SCIPY_OBSERVED_VERSION = importlib.metadata.version("scipy")
_SCIPY_REQUIRED_VERSION = "1.15.3"
_WORKING_SET_MAX_ITERATIONS = 100
_WORKING_SET_MAX_EPOCHS = 50_000
_WORKING_SET_INITIAL_SIZE = 32
_WORKING_SET_STRATEGY = "subdiff"
_WORKING_SET_COARSE_TOLERANCE_MULTIPLIER = 0.1
_WORKING_SET_REFINEMENT_TOLERANCE_MULTIPLIER = 0.01
_WORKING_SET_MAX_PASSES = 2
_ACTIVE_SET_MAX_ITERATIONS = 32
_LASSO_ORTHANT_MAX_ITERATIONS = 64
_ACTIVE_SET_BACKWARD_ERROR_MULTIPLIER = 256.0
_ACTIVE_SET_ORTHANT_TIE_MULTIPLIER = 64.0
_WORKING_SET_SOLVER = (
    "skglm_0_5_anderson_cd_hessian_singular_safe_lasso_orthant_full_kkt_dual_certified"
)
if (
    _SKGLM_OBSERVED_VERSION != _SKGLM_REQUIRED_VERSION
    or _NUMBA_OBSERVED_VERSION != _NUMBA_REQUIRED_VERSION
    or _LLVMLITE_OBSERVED_VERSION != _LLVMLITE_REQUIRED_VERSION
    or _SCIPY_OBSERVED_VERSION != _SCIPY_REQUIRED_VERSION
):
    raise RuntimeError(
        "Stage 9 Elastic Net working-set runtime differs: "
        f"skglm={_SKGLM_OBSERVED_VERSION}, numba={_NUMBA_OBSERVED_VERSION}, "
        f"llvmlite={_LLVMLITE_OBSERVED_VERSION}, scipy={_SCIPY_OBSERVED_VERSION}"
    )
_CHECKPOINT_AUTHORIZATIONS = {
    "stage_09_parent_fit_complete": False,
    "completed_search_bundle_authorized": False,
    "stage_10_or_later_authorized": False,
    "portfolio_or_backtest_authorized": False,
    "validation_or_final_access_authorized": False,
}
_CHECKPOINT_SCIENTIFIC_SCOPE = {
    "development_only": True,
    "target": "residual_return_rank",
    "parent_fit_complete": False,
    "portfolio_or_backtest_permitted": False,
    "validation_2015_2019_accessed": False,
    "final_2020_2025_accessed": False,
}


@dataclass(frozen=True)
class ElasticNetFitReceipt:
    config_id: str
    fold_id: str
    input_features: int
    train_rows: int
    outer_validation_rows: int
    lambda_ref: float
    lambda_value: float
    rho: float
    lambda_index: int
    nonzero_coefficients: int
    initial_dual_gap: float
    post_secondary_dual_gap: float
    dual_gap: float
    dual_gap_limit: float
    initial_kkt_max_violation: float
    initial_kkt_limit: float
    post_secondary_kkt_max_violation: float
    post_secondary_kkt_limit: float
    kkt_max_violation: float
    kkt_limit: float
    iterations: int
    initial_iterations: int
    secondary_iterations: int
    tertiary_iterations: int
    refinement_iterations: int
    solver_passes: int
    initial_solver_stop_criterion: float
    initial_solver_stop_limit: float
    secondary_solver_stop_criterion: float
    secondary_solver_stop_limit: float
    solver_stop_criterion: float
    solver_stop_limit: float
    refinement_route: str
    initial_coefficient_sha256: str
    final_coefficient_sha256: str
    orthant_iterations: int
    orthant_cholesky_attempts: int
    orthant_gelsy_attempts: int
    orthant_rank_deficient_solves: int
    orthant_max_active_dimension: int
    orthant_min_solve_rank: int
    orthant_support_transitions: int
    orthant_boundary_transitions: int
    orthant_dropped_coordinates: int
    orthant_added_coordinates: int
    orthant_trace: list[dict[str, Any]]
    converged: bool
    elapsed_seconds: float
    solver: str
    dtype: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _WorkingSetPathResult:
    alphas: np.ndarray
    coefficients: np.ndarray
    initial_dual_gaps: np.ndarray
    post_secondary_dual_gaps: np.ndarray
    dual_gaps: np.ndarray
    initial_kkt_maximum: np.ndarray
    initial_kkt_limits: np.ndarray
    post_secondary_kkt_maximum: np.ndarray
    post_secondary_kkt_limits: np.ndarray
    total_iterations: np.ndarray
    initial_iterations: np.ndarray
    secondary_iterations: np.ndarray
    tertiary_iterations: np.ndarray
    refinement_iterations: np.ndarray
    solver_passes: np.ndarray
    initial_stop_criteria: np.ndarray
    secondary_stop_criteria: np.ndarray
    final_stop_criteria: np.ndarray
    initial_stop_limits: np.ndarray
    secondary_stop_limits: np.ndarray
    final_stop_limits: np.ndarray
    orthant_traces: tuple[list[dict[str, Any]], ...]
    refinement_routes: tuple[str, ...]
    initial_coefficient_sha256s: tuple[str, ...]
    final_coefficient_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class _ActiveSetPolishResult:
    coefficients: np.ndarray
    iterations: int
    resolved: bool
    trace: list[dict[str, Any]]


def _authoritative_float64_workspace(legacy_work_path: Path) -> tuple[str, Path]:
    """Translate the unchanged handler's legacy name into a safe float64 path.

    The shared Stage 9 handler still passes ``*.weighted-f32.memmap`` and
    requires that exact path to remain absent after the native call.  Changing
    that shared handler would invalidate completed NN and XGBoost job
    identities, so Elastic Net privately uses a sibling float64 workspace.
    """

    legacy = Path(legacy_work_path)
    if not legacy.is_absolute() or ".." in legacy.parts:
        raise ValueError("Authoritative Elastic Net work_path must be absolute and traversal-free")
    if not legacy.name.endswith(_LEGACY_WEIGHTED_SUFFIX):
        raise ValueError("Authoritative Elastic Net work_path has an invalid legacy suffix")
    parent_job_id = legacy.name[: -len(_LEGACY_WEIGHTED_SUFFIX)]
    if _AUTHORITATIVE_PARENT_JOB.fullmatch(parent_job_id) is None:
        raise ValueError("Authoritative Elastic Net work_path lacks its parent job identity")
    if legacy.exists() or legacy.is_symlink():
        raise RuntimeError("Legacy Elastic Net float32 work_path must remain absent")

    workspace = legacy.with_name(parent_job_id + _FLOAT64_WEIGHTED_SUFFIX)
    if workspace.is_symlink():
        raise RuntimeError("Unsafe Elastic Net float64 workspace symlink")
    if workspace.exists():
        status = workspace.stat(follow_symlinks=False)
        if not stat.S_ISREG(status.st_mode):
            raise RuntimeError("Unsafe Elastic Net float64 workspace object")
        # A killed worker may leave only this reconstructible workspace.  Rho
        # checkpoints live in a separate content-addressed directory.
        workspace.unlink()
    return parent_job_id, workspace


def _checkpoint_identity(
    *,
    parent_job_id: str,
    fold_id: str,
    rho_index: int,
    rho: float,
    configurations: tuple[ElasticNetConfig, ...],
    alphas: np.ndarray,
    lambda_ref: float,
    train_rows: int,
    input_features: int,
    outer_validation_rows: int,
    max_iter: int,
    tolerance: float,
    y_second_moment: float,
) -> dict[str, Any]:
    penalties = np.asarray(alphas, dtype=np.float64)
    if _AUTHORITATIVE_PARENT_JOB.fullmatch(parent_job_id) is None:
        raise ValueError("Elastic Net checkpoint parent job ID is invalid")
    if (
        not 0 <= int(rho_index) < len(RHO_GRID)
        or float(rho) != float(RHO_GRID[int(rho_index)])
        or len(configurations) != LAMBDA_POINTS
        or penalties.shape != (LAMBDA_POINTS,)
        or not np.isfinite(penalties).all()
        or np.any(penalties <= 0.0)
    ):
        raise ValueError("Elastic Net checkpoint path identity is invalid")
    if any(
        config.rho != float(rho)
        or config.lambda_index != position
        for position, config in enumerate(configurations, start=1)
    ):
        raise ValueError("Elastic Net checkpoint configurations are not canonical")
    if (
        train_rows < 2
        or input_features < 1
        or outer_validation_rows < 1
        or max_iter < 1
        or not math.isfinite(float(tolerance))
        or tolerance <= 0.0
        or not math.isfinite(float(y_second_moment))
        or y_second_moment <= 0.0
    ):
        raise ValueError("Elastic Net checkpoint dimensions or solver limits are invalid")
    return {
        "schema_version": 6,
        "purpose": _CHECKPOINT_PURPOSE,
        "parent_path_job_id": parent_job_id,
        "parent_job_identity_sha256": parent_job_id.removeprefix("stage09-fit-"),
        "family": "elastic_net",
        "fold_id": fold_id,
        "seed": None,
        "rho_index": int(rho_index),
        "rho_hex": float(rho).hex(),
        "config_ids": [config.config_id for config in configurations],
        "lambda_ref_hex": float(lambda_ref).hex(),
        "lambda_values_hex": [float(value).hex() for value in penalties],
        "train_rows": int(train_rows),
        "input_features": int(input_features),
        "outer_validation_rows": int(outer_validation_rows),
        "frozen_estimator_max_iterations_binding": int(max_iter),
        "tolerance_hex": float(tolerance).hex(),
        "y_second_moment_hex": float(y_second_moment).hex(),
        "dual_gap_limit_formula": (
            "tolerance_times_y_second_moment_for_positive_rho;"
            "tolerance_times_max_one_y_second_moment_for_ridge"
        ),
        "selection": "cyclic",
        "positive_rho_solver": _WORKING_SET_SOLVER,
        "positive_rho_solver_dependency": (
            f"skglm=={_SKGLM_REQUIRED_VERSION}"
        ),
        "positive_rho_jit_dependencies": {
            "numba": _NUMBA_REQUIRED_VERSION,
            "llvmlite": _LLVMLITE_REQUIRED_VERSION,
        },
        "positive_rho_datafit": "skglm.QuadraticHessian",
        "positive_rho_hessian_dtype": "<f8",
        "working_set_max_iterations": _WORKING_SET_MAX_ITERATIONS,
        "working_set_max_epochs": _WORKING_SET_MAX_EPOCHS,
        "working_set_initial_size": _WORKING_SET_INITIAL_SIZE,
        "working_set_strategy": _WORKING_SET_STRATEGY,
        "working_set_coarse_tolerance_multiplier_hex": (
            _WORKING_SET_COARSE_TOLERANCE_MULTIPLIER.hex()
        ),
        "working_set_refinement_tolerance_multiplier_hex": (
            _WORKING_SET_REFINEMENT_TOLERANCE_MULTIPLIER.hex()
        ),
        "working_set_max_passes": _WORKING_SET_MAX_PASSES,
        "certificate_refinement_trigger": (
            "independent_kkt_or_dual_gap_exceeds_frozen_limit"
        ),
        "certificate_refinement_scope": "only_failed_alphas_in_ascending_path_order",
        "certificate_refinement_route": (
            "rho_in_open_unit_interval_active_set_then_strict_cd_if_unresolved_"
            "or_hard_failure;"
            "rho_one_singular_safe_same_alpha_orthant_or_hard_failure"
        ),
        "active_set_objective": (
            "half_beta_gram_beta_minus_rhs_beta_plus_alpha_rho_l1_plus_"
            "half_alpha_one_minus_rho_l2_squared"
        ),
        "active_set_scaling": "raw_normalized_weighted_centered_gram_no_standardization",
        "active_set_initial_support": "abs_same_alpha_coarse_beta_gt_nonzero_threshold",
        "active_set_initial_sign": "sign_same_alpha_coarse_beta",
        "active_set_fixed_orthant_equation": (
            "gram_AA_plus_alpha_one_minus_rho_identity_beta_A_equals_"
            "rhs_A_minus_alpha_rho_sign_A"
        ),
        "active_set_subset_solver": (
            "numpy_float64_cholesky_scipy_solve_triangular_no_jitter"
        ),
        "active_set_subset_solver_dependency": f"scipy=={_SCIPY_OBSERVED_VERSION}",
        "active_set_full_support_solver": (
            "computed_once_raw_gram_float64_eigendecomposition_shared_with_"
            "ridge_path_no_jitter"
        ),
        "active_set_orthant_rule": (
            "move_to_first_zero_boundary_drop_all_numerically_tied_crossings_"
            "before_opposite_sign_reentry"
        ),
        "active_set_violator_rule": (
            "add_all_inactive_abs_smooth_gt_alpha_rho_plus_backward_error_"
            "in_sorted_feature_index_order_with_sign_smooth"
        ),
        "active_set_nonzero_threshold_hex": NONZERO_THRESHOLD.hex(),
        "active_set_backward_error_multiplier_hex": (
            _ACTIVE_SET_BACKWARD_ERROR_MULTIPLIER.hex()
        ),
        "active_set_backward_error_formula": (
            "multiplier_eps_times_max_one_abs_rhs_alpha_feature_count_times_"
            "max_abs_gram_times_max_abs_beta_l2_times_max_abs_beta"
        ),
        "active_set_orthant_tie_multiplier_hex": (
            _ACTIVE_SET_ORTHANT_TIE_MULTIPLIER.hex()
        ),
        "active_set_max_iterations": _ACTIVE_SET_MAX_ITERATIONS,
        "lasso_orthant_max_iterations": _LASSO_ORTHANT_MAX_ITERATIONS,
        "lasso_orthant_linear_solver": (
            "numpy_float64_cholesky_first_then_scipy_linalg_lstsq_on_"
            "linear_algebra_error"
        ),
        "lasso_orthant_lstsq_dependency": f"scipy=={_SCIPY_REQUIRED_VERSION}",
        "lasso_orthant_lstsq_cond": None,
        "lasso_orthant_lstsq_driver": "gelsy",
        "lasso_orthant_lstsq_check_finite": False,
        "lasso_orthant_runtime_architecture": platform.machine(),
        "lasso_orthant_numpy_version": importlib.metadata.version("numpy"),
        "lasso_orthant_diagonal_jitter": False,
        "lasso_orthant_unresolved_policy": "hard_failure_no_strict_cd_fallback",
        "lasso_orthant_receipt_evidence": (
            "per_alpha_ordered_iteration_trace_with_solver_dimension_rank_"
            "rank_deficiency_and_boundary_drop_add_transitions"
        ),
        "active_set_stop_evidence": (
            "normalized_independent_full_kkt_with_frozen_tolerance_limit"
        ),
        "strict_cd_fallback_grouping": (
            "maximal_contiguous_ascending_strict_required_path_indices"
        ),
        "strict_cd_fallback_group_initialization": (
            "first_group_index_same_alpha_coarse_coefficient"
        ),
        "strict_cd_fallback_within_group_warm_start": (
            "previous_alpha_solution_in_frozen_descending_path_order"
        ),
        "strict_cd_fallback_replacement_scope": (
            "strict_required_columns_only_passing_columns_bitwise_unchanged"
        ),
        "strict_cd_fallback_stop_evidence": (
            "anderson_cd_stop_with_refinement_tolerance_limit"
        ),
        "certificate_final_recompute": (
            "all_alphas_independent_dual_gap_and_full_kkt"
        ),
        "working_set_fit_intercept": False,
        "working_set_estimator_warm_start": False,
        "path_previous_alpha_warm_start": True,
        "positive_coefficients_only": False,
        "weighted_design_dtype": "<f8",
        "forecast_dtype": "<f4",
        "rho_checkpoint_resume_policy": (
            "all_development_folds_may_resume_authenticated_completed_rho_paths"
        ),
    }


def _checkpoint_location(
    *,
    work_root: Path,
    identity: Mapping[str, Any],
) -> tuple[Path, str]:
    parent_job_id = str(identity["parent_path_job_id"])
    checkpoint_job_id = job_id_from_identity(identity)
    destination = (
        Path(work_root)
        / _CHECKPOINT_DIRECTORY
        / parent_job_id
        / f"{checkpoint_job_id}.npz"
    )
    return destination, checkpoint_job_id


def _checkpoint_schema(outer_validation_rows: int) -> dict[str, dict[str, Any]]:
    return {
        "predictions": {
            "dtype": "<f4",
            "shape": [int(outer_validation_rows), LAMBDA_POINTS],
        }
    }


def _load_completed_checkpoint_with_retry(
    path: Path,
    *,
    trusted_root: Path,
    identity: Mapping[str, Any],
    schema: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    for _attempt in range(_CACHE_READ_ATTEMPTS):
        receipt = load_completed_job(
            path,
            trusted_root=trusted_root,
            expected_identity=identity,
            expected_array_schema=schema,
        )
        if receipt is not None:
            return receipt
    return None


def _validated_checkpoint_receipts(
    values: object,
    *,
    fold_id: str,
    configurations: tuple[ElasticNetConfig, ...],
    alphas: np.ndarray,
    lambda_ref: float,
    train_rows: int,
    input_features: int,
    outer_validation_rows: int,
    tolerance: float,
    y_second_moment: float,
) -> tuple[ElasticNetFitReceipt, ...]:
    if not isinstance(values, list) or len(values) != len(configurations):
        raise RuntimeError("Elastic Net checkpoint model-fit ledger width differs")
    penalties = np.asarray(alphas, dtype=np.float64)
    receipts: list[ElasticNetFitReceipt] = []
    for position, (raw, config) in enumerate(
        zip(values, configurations, strict=True)
    ):
        expected_solver = (
            "weighted_centered_ridge_eigendecomposition"
            if config.rho == 0.0
            else _WORKING_SET_SOLVER
        )
        is_ridge = config.rho == 0.0
        if not isinstance(raw, Mapping):
            raise RuntimeError("Elastic Net checkpoint model-fit receipt is malformed")
        try:
            receipt = ElasticNetFitReceipt(**dict(raw))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Elastic Net checkpoint model-fit receipt schema differs") from exc
        integer_evidence = (
            receipt.input_features,
            receipt.train_rows,
            receipt.outer_validation_rows,
            receipt.lambda_index,
            receipt.nonzero_coefficients,
            receipt.iterations,
            receipt.initial_iterations,
            receipt.secondary_iterations,
            receipt.tertiary_iterations,
            receipt.refinement_iterations,
            receipt.solver_passes,
            receipt.orthant_iterations,
            receipt.orthant_cholesky_attempts,
            receipt.orthant_gelsy_attempts,
            receipt.orthant_rank_deficient_solves,
            receipt.orthant_max_active_dimension,
            receipt.orthant_min_solve_rank,
            receipt.orthant_support_transitions,
            receipt.orthant_boundary_transitions,
            receipt.orthant_dropped_coordinates,
            receipt.orthant_added_coordinates,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integer_evidence
        ):
            raise RuntimeError("Elastic Net checkpoint iteration evidence differs")
        floating_evidence = (
            receipt.lambda_ref,
            receipt.lambda_value,
            receipt.rho,
            receipt.initial_dual_gap,
            receipt.post_secondary_dual_gap,
            receipt.dual_gap,
            receipt.dual_gap_limit,
            receipt.initial_kkt_max_violation,
            receipt.initial_kkt_limit,
            receipt.post_secondary_kkt_max_violation,
            receipt.post_secondary_kkt_limit,
            receipt.kkt_max_violation,
            receipt.kkt_limit,
            receipt.initial_solver_stop_criterion,
            receipt.initial_solver_stop_limit,
            receipt.secondary_solver_stop_criterion,
            receipt.secondary_solver_stop_limit,
            receipt.solver_stop_criterion,
            receipt.solver_stop_limit,
            receipt.elapsed_seconds,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in floating_evidence
        ):
            raise RuntimeError("Elastic Net checkpoint floating-point evidence differs")
        coarse_stop_limit = (
            float(tolerance) * _WORKING_SET_COARSE_TOLERANCE_MULTIPLIER
        )
        refinement_stop_limit = (
            float(tolerance) * _WORKING_SET_REFINEMENT_TOLERANCE_MULTIPLIER
        )
        expected_dual_gap_limit = float(tolerance) * (
            max(1.0, float(y_second_moment))
            if is_ridge
            else float(y_second_moment)
        )
        trace_summary = _orthant_trace_summary(
            receipt.orthant_trace,
            input_features=input_features,
        )
        valid_trace_summary = all(
            getattr(receipt, key) == value
            for key, value in trace_summary.items()
        )
        valid_coefficient_hashes = all(
            isinstance(value, str)
            and len(value) == 64
            and set(value) <= set("0123456789abcdef")
            for value in (
                receipt.initial_coefficient_sha256,
                receipt.final_coefficient_sha256,
            )
        )
        if is_ridge:
            valid_iteration_evidence = (
                receipt.iterations == 1
                and receipt.initial_iterations == 1
                and receipt.secondary_iterations == 0
                and receipt.tertiary_iterations == 0
                and receipt.refinement_iterations == 0
                and receipt.solver_passes == 1
                and receipt.refinement_route == "ridge_closed_form"
                and not receipt.orthant_trace
                and receipt.initial_coefficient_sha256
                == receipt.final_coefficient_sha256
            )
            valid_stop_limits = (
                float(receipt.initial_solver_stop_limit).hex()
                == float(receipt.secondary_solver_stop_limit).hex()
                == float(receipt.solver_stop_limit).hex()
                and float(receipt.initial_solver_stop_criterion).hex()
                == float(receipt.secondary_solver_stop_criterion).hex()
                == float(receipt.solver_stop_criterion).hex()
            )
            valid_certificate_evidence = (
                float(receipt.initial_dual_gap).hex()
                == float(receipt.post_secondary_dual_gap).hex()
                == float(receipt.dual_gap).hex()
                and float(receipt.initial_kkt_max_violation).hex()
                == float(receipt.post_secondary_kkt_max_violation).hex()
                == float(receipt.kkt_max_violation).hex()
                and float(receipt.initial_kkt_limit).hex()
                == float(receipt.post_secondary_kkt_limit).hex()
                == float(receipt.kkt_limit).hex()
            )
        else:
            active_stop_limit = float(tolerance)
            final_limit_hex = float(receipt.solver_stop_limit).hex()
            secondary_limit_hex = float(
                receipt.secondary_solver_stop_limit
            ).hex()
            used_active_set = secondary_limit_hex == active_stop_limit.hex()
            used_secondary_cd = (
                secondary_limit_hex == refinement_stop_limit.hex()
            )
            active_stop_criterion = (
                math.inf
                if float(receipt.kkt_limit) <= 0.0
                else (
                    float(receipt.kkt_max_violation)
                    * float(tolerance)
                    / float(receipt.kkt_limit)
                )
            )
            post_secondary_active_stop_criterion = (
                math.inf
                if float(receipt.post_secondary_kkt_limit) <= 0.0
                else (
                    float(receipt.post_secondary_kkt_max_violation)
                    * float(tolerance)
                    / float(receipt.post_secondary_kkt_limit)
                )
            )
            initial_certificate_failed = (
                receipt.initial_dual_gap > receipt.dual_gap_limit
                or receipt.initial_kkt_max_violation > receipt.initial_kkt_limit
            )
            initial_certificate_passed = not initial_certificate_failed
            post_secondary_certificate_failed = (
                receipt.post_secondary_dual_gap > receipt.dual_gap_limit
                or receipt.post_secondary_kkt_max_violation
                > receipt.post_secondary_kkt_limit
            )
            post_secondary_certificate_passed = not post_secondary_certificate_failed
            initial_to_secondary_unchanged = (
                float(receipt.initial_dual_gap).hex()
                == float(receipt.post_secondary_dual_gap).hex()
                and float(receipt.initial_kkt_max_violation).hex()
                == float(receipt.post_secondary_kkt_max_violation).hex()
                and float(receipt.initial_kkt_limit).hex()
                == float(receipt.post_secondary_kkt_limit).hex()
            )
            secondary_to_final_unchanged = (
                float(receipt.post_secondary_dual_gap).hex()
                == float(receipt.dual_gap).hex()
                and float(receipt.post_secondary_kkt_max_violation).hex()
                == float(receipt.kkt_max_violation).hex()
                and float(receipt.post_secondary_kkt_limit).hex()
                == float(receipt.kkt_limit).hex()
            )
            trace_required = (
                receipt.rho == 1.0
                and receipt.solver_passes == 2
                and receipt.refinement_route == "rho_one_singular_safe_orthant"
            )
            if receipt.rho == 1.0:
                valid_route = (
                    (
                        receipt.solver_passes == 1
                        and receipt.refinement_route == "none"
                        and not receipt.orthant_trace
                        and receipt.initial_coefficient_sha256
                        == receipt.final_coefficient_sha256
                    )
                    or (
                        trace_required
                        and bool(receipt.orthant_trace)
                        and receipt.orthant_iterations
                        == receipt.secondary_iterations
                        and 1 <= receipt.orthant_iterations
                        <= _LASSO_ORTHANT_MAX_ITERATIONS
                        and receipt.orthant_trace[-1]["support_after"]
                        == receipt.nonzero_coefficients
                    )
                )
            else:
                valid_route = (
                    not receipt.orthant_trace
                    and (
                        (
                            receipt.solver_passes == 1
                            and receipt.refinement_route == "none"
                            and receipt.initial_coefficient_sha256
                            == receipt.final_coefficient_sha256
                        )
                        or (
                            receipt.solver_passes == 2
                            and (
                                (
                                    used_active_set
                                    and receipt.refinement_route
                                    == "rho_lt_one_active_set"
                                )
                                or (
                                    used_secondary_cd
                                    and receipt.refinement_route
                                    == "rho_lt_one_grouped_strict_cd"
                                )
                            )
                        )
                    )
                )
            valid_iteration_evidence = (
                receipt.solver_passes in {1, 2}
                and 0 <= receipt.initial_iterations < _WORKING_SET_MAX_ITERATIONS
                and 0 <= receipt.secondary_iterations
                < (
                    _LASSO_ORTHANT_MAX_ITERATIONS + 1
                    if receipt.rho == 1.0
                    else _ACTIVE_SET_MAX_ITERATIONS + _WORKING_SET_MAX_ITERATIONS
                )
                and receipt.tertiary_iterations == 0
                and receipt.refinement_iterations
                == receipt.secondary_iterations
                and receipt.iterations
                == receipt.initial_iterations + receipt.refinement_iterations
                and receipt.iterations
                < (
                    _WORKING_SET_MAX_PASSES * _WORKING_SET_MAX_ITERATIONS
                    + _ACTIVE_SET_MAX_ITERATIONS
                )
                and (
                    (
                        receipt.solver_passes == 1
                        and receipt.secondary_iterations == 0
                        and receipt.tertiary_iterations == 0
                    )
                    or (
                        receipt.solver_passes == 2
                        and receipt.tertiary_iterations == 0
                        and (
                            (
                                used_active_set
                                and 1 <= receipt.secondary_iterations
                                <= (
                                    _LASSO_ORTHANT_MAX_ITERATIONS
                                    if receipt.rho == 1.0
                                    else _ACTIVE_SET_MAX_ITERATIONS
                                )
                            )
                            or (
                                used_secondary_cd
                                and receipt.rho < 1.0
                                and receipt.secondary_iterations >= 1
                            )
                        )
                    )
                )
                and valid_route
            )
            valid_stop_limits = (
                float(receipt.initial_solver_stop_limit).hex()
                == coarse_stop_limit.hex()
                and (
                    (
                        receipt.solver_passes == 1
                        and final_limit_hex == coarse_stop_limit.hex()
                        and float(receipt.secondary_solver_stop_limit).hex()
                        == coarse_stop_limit.hex()
                        and float(receipt.initial_solver_stop_criterion).hex()
                        == float(receipt.secondary_solver_stop_criterion).hex()
                        == float(receipt.solver_stop_criterion).hex()
                    )
                    or (
                        receipt.solver_passes == 2
                        and (
                            (
                                used_active_set
                                and 0.0 < receipt.rho <= 1.0
                                and final_limit_hex == active_stop_limit.hex()
                                and float(receipt.secondary_solver_stop_limit).hex()
                                == active_stop_limit.hex()
                                and float(
                                    receipt.secondary_solver_stop_criterion
                                ).hex()
                                == post_secondary_active_stop_criterion.hex()
                                and float(receipt.solver_stop_criterion).hex()
                                == active_stop_criterion.hex()
                            )
                            or (
                                used_secondary_cd
                                and receipt.rho < 1.0
                                and final_limit_hex
                                == refinement_stop_limit.hex()
                                and float(receipt.secondary_solver_stop_limit).hex()
                                == refinement_stop_limit.hex()
                            )
                        )
                        and float(receipt.secondary_solver_stop_criterion).hex()
                        == float(receipt.solver_stop_criterion).hex()
                    )
                )
            )
            valid_certificate_evidence = (
                (
                    receipt.solver_passes == 1
                    and initial_certificate_passed
                    and initial_to_secondary_unchanged
                    and secondary_to_final_unchanged
                )
                or (
                    receipt.solver_passes == 2
                    and initial_certificate_failed
                    and post_secondary_certificate_passed
                    and secondary_to_final_unchanged
                )
            )
        if (
            receipt.config_id != config.config_id
            or receipt.fold_id != fold_id
            or receipt.input_features != input_features
            or receipt.train_rows != train_rows
            or receipt.outer_validation_rows != outer_validation_rows
            or float(receipt.lambda_ref).hex() != float(lambda_ref).hex()
            or float(receipt.lambda_value).hex() != float(penalties[position]).hex()
            or float(receipt.rho).hex() != float(config.rho).hex()
            or receipt.lambda_index != config.lambda_index
            or not isinstance(receipt.converged, bool)
            or not receipt.converged
            or receipt.nonzero_coefficients < 0
            or receipt.nonzero_coefficients > input_features
            or not valid_iteration_evidence
            or not valid_stop_limits
            or not valid_certificate_evidence
            or not valid_trace_summary
            or not valid_coefficient_hashes
            or float(receipt.dual_gap_limit).hex()
            != expected_dual_gap_limit.hex()
            or receipt.initial_solver_stop_criterion < 0.0
            or receipt.initial_solver_stop_limit < 0.0
            or receipt.initial_solver_stop_criterion
            > receipt.initial_solver_stop_limit
            or receipt.secondary_solver_stop_criterion < 0.0
            or receipt.secondary_solver_stop_limit < 0.0
            or receipt.secondary_solver_stop_criterion
            > receipt.secondary_solver_stop_limit
            or receipt.solver_stop_criterion < 0.0
            or receipt.solver_stop_limit < 0.0
            or receipt.solver_stop_criterion > receipt.solver_stop_limit
            or receipt.initial_dual_gap < 0.0
            or receipt.post_secondary_dual_gap < 0.0
            or receipt.dual_gap < 0.0
            or receipt.dual_gap_limit < 0.0
            or receipt.dual_gap > receipt.dual_gap_limit
            or receipt.initial_kkt_max_violation < 0.0
            or receipt.initial_kkt_limit < 0.0
            or receipt.post_secondary_kkt_max_violation < 0.0
            or receipt.post_secondary_kkt_limit < 0.0
            or receipt.kkt_max_violation < 0.0
            or receipt.kkt_limit < 0.0
            or receipt.kkt_max_violation > receipt.kkt_limit
            or receipt.elapsed_seconds < 0.0
            or receipt.solver != expected_solver
            or receipt.dtype != "float64"
        ):
            raise RuntimeError("Elastic Net checkpoint model-fit evidence differs")
        receipts.append(receipt)
    return tuple(receipts)


def _load_rho_checkpoint(
    *,
    work_root: Path,
    identity: Mapping[str, Any],
    fold_id: str,
    configurations: tuple[ElasticNetConfig, ...],
    alphas: np.ndarray,
    lambda_ref: float,
    train_rows: int,
    input_features: int,
    outer_validation_rows: int,
) -> tuple[np.ndarray, tuple[ElasticNetFitReceipt, ...]] | None:
    path, checkpoint_job_id = _checkpoint_location(
        work_root=work_root,
        identity=identity,
    )
    schema = _checkpoint_schema(outer_validation_rows)
    outer = _load_completed_checkpoint_with_retry(
        path,
        trusted_root=work_root,
        identity=identity,
        schema=schema,
    )
    if outer is None:
        return None
    fit = outer.get("fit_receipt")
    if not isinstance(fit, Mapping) or (
        fit.get("schema_version") != 6
        or fit.get("status") != _CHECKPOINT_STATUS
        or fit.get("purpose") != _CHECKPOINT_PURPOSE
        or fit.get("checkpoint_job_id") != checkpoint_job_id
        or fit.get("parent_path_job_id") != identity["parent_path_job_id"]
        or fit.get("fold_id") != fold_id
        or fit.get("rho_index") != identity["rho_index"]
        or fit.get("rho_hex") != identity["rho_hex"]
        or fit.get("scientific_scope") != _CHECKPOINT_SCIENTIFIC_SCOPE
        or fit.get("authorizations") != _CHECKPOINT_AUTHORIZATIONS
        or fit.get("publication_fit_receipts_permitted") is not False
    ):
        raise RuntimeError("Elastic Net checkpoint authorization or identity differs")
    receipts = _validated_checkpoint_receipts(
        fit.get("model_fit"),
        fold_id=fold_id,
        configurations=configurations,
        alphas=alphas,
        lambda_ref=lambda_ref,
        train_rows=train_rows,
        input_features=input_features,
        outer_validation_rows=outer_validation_rows,
        tolerance=float.fromhex(str(identity["tolerance_hex"])),
        y_second_moment=float.fromhex(str(identity["y_second_moment_hex"])),
    )
    with np.load(path, allow_pickle=False) as archive:
        predictions = np.asarray(archive["predictions"], dtype=np.float32).copy()
    after = _load_completed_checkpoint_with_retry(
        path,
        trusted_root=work_root,
        identity=identity,
        schema=schema,
    )
    if after != outer:
        raise RuntimeError("Elastic Net checkpoint changed while loading")
    return predictions, receipts


def _publish_rho_checkpoint(
    *,
    work_root: Path,
    identity: Mapping[str, Any],
    fold_id: str,
    predictions: np.ndarray,
    receipts: tuple[ElasticNetFitReceipt, ...],
) -> None:
    path, checkpoint_job_id = _checkpoint_location(
        work_root=work_root,
        identity=identity,
    )
    forecast = np.ascontiguousarray(predictions, dtype=np.float32)
    schema = _checkpoint_schema(int(identity["outer_validation_rows"]))
    fit_receipt = {
        "schema_version": 6,
        "status": _CHECKPOINT_STATUS,
        "purpose": _CHECKPOINT_PURPOSE,
        "checkpoint_job_id": checkpoint_job_id,
        "parent_path_job_id": identity["parent_path_job_id"],
        "fold_id": fold_id,
        "rho_index": identity["rho_index"],
        "rho_hex": identity["rho_hex"],
        "publication_fit_receipts_permitted": False,
        "scientific_scope": dict(_CHECKPOINT_SCIENTIFIC_SCOPE),
        "authorizations": dict(_CHECKPOINT_AUTHORIZATIONS),
        "model_fit": [receipt.as_dict() for receipt in receipts],
    }
    last_error: RuntimeError | None = None
    for _attempt in range(_CACHE_READ_ATTEMPTS):
        try:
            publish_completed_job(
                path,
                trusted_root=work_root,
                job_identity=identity,
                predictions=forecast,
                fit_receipt=fit_receipt,
                expected_array_schema=schema,
            )
            last_error = None
            break
        except RuntimeError as exc:
            if "Stage 9 cache file changed during hashing:" not in str(exc):
                raise
            last_error = exc
    if last_error is not None:
        raise last_error
    authenticated = _load_completed_checkpoint_with_retry(
        path,
        trusted_root=work_root,
        identity=identity,
        schema=schema,
    )
    if authenticated is None:
        raise RuntimeError("Published Elastic Net rho checkpoint failed reauthentication")


def _emit_path_progress(
    *, fold_id: str, rho_index: int, rho: float, status: str
) -> None:
    print(
        f"Stage 9 Elastic Net {fold_id}: rho path {rho_index + 1}/{len(RHO_GRID)} "
        f"(rho={rho:g}) {status}",
        flush=True,
    )


def lambda_value(config: ElasticNetConfig, lambda_ref: float) -> float:
    return elastic_net_lambda_value(config, lambda_ref)


def _validated(
    X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(X, dtype=np.float32)
    target = np.asarray(y, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("Invalid Elastic Net design shape")
    if target.shape != (features.shape[0],) or weights.shape != target.shape:
        raise ValueError("Elastic Net input lengths differ")
    if not (
        np.isfinite(features).all()
        and np.isfinite(target).all()
        and np.isfinite(weights).all()
        and np.all(weights > 0)
    ):
        raise ValueError("Elastic Net inputs must be finite with positive weights")
    weights = weights * (float(weights.size) / float(np.sum(weights, dtype=np.float64)))
    return features, target, weights


def weighted_lambda_reference(
    X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, *, chunk_rows: int = 8192
) -> tuple[float, float, np.ndarray]:
    """Return the frozen weighted lasso zero-slope threshold and centering data."""

    features, target, weights = _validated(X, y, sample_weight)
    weight_sum = float(np.sum(weights, dtype=np.float64))
    y_mean = float(np.dot(weights, target) / weight_sum)
    x_mean = np.zeros(features.shape[1], dtype=np.float64)
    xy = np.zeros(features.shape[1], dtype=np.float64)
    centered_y = target - y_mean
    for start in range(0, features.shape[0], chunk_rows):
        stop = min(features.shape[0], start + chunk_rows)
        block = np.asarray(features[start:stop], dtype=np.float64)
        block_weight = weights[start:stop]
        x_mean += block.T @ block_weight
        xy += block.T @ (block_weight * centered_y[start:stop])
    x_mean /= weight_sum
    reference = float(np.max(np.abs(xy)) / weight_sum)
    if not math.isfinite(reference):
        raise FloatingPointError("Elastic Net lambda reference is nonfinite")
    return reference, y_mean, x_mean


def prepare_weighted_centered_design(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    x_mean: np.ndarray,
    y_mean: float,
    destination: Path | None = None,
    chunk_rows: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct float64 sqrt-weighted centered arrays.

    Authoritative fits pass ``destination`` so the large interaction-expanded
    design remains disk backed.  Float64 is required because the same array is
    used to build the precomputed normal equations and to obtain sklearn's
    dual certificate without a float32 reduction.
    """

    features, target, weights = _validated(X, y, sample_weight)
    means = np.asarray(x_mean, dtype=np.float64)
    if means.shape != (features.shape[1],) or not np.isfinite(means).all():
        raise ValueError("Invalid Elastic Net feature means")
    if destination is None:
        weighted = np.empty(features.shape, dtype=np.float64, order="F")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        weighted = np.memmap(
            destination,
            mode="w+",
            dtype=np.float64,
            shape=features.shape,
            order="F",
        )
    root_weight = np.sqrt(weights, dtype=np.float64)
    for start in range(0, features.shape[0], chunk_rows):
        stop = min(features.shape[0], start + chunk_rows)
        block = np.asarray(features[start:stop], dtype=np.float64)
        block = (block - means) * root_weight[start:stop, None]
        weighted[start:stop] = block
    weighted_y = np.asarray(
        (target - float(y_mean)) * root_weight,
        dtype=np.float64,
    )
    if isinstance(weighted, np.memmap):
        weighted.flush()
    return weighted, weighted_y


def _require_converged_path(
    *,
    dual_gaps: np.ndarray,
    iterations: np.ndarray,
    weighted_y: np.ndarray,
    tolerance: float,
    max_iter: int,
    convergence_warnings: tuple[str, ...],
) -> None:
    """Require sklearn's dual certificate for every requested path point.

    ``enet_path`` scales its stopping threshold by ``||y||^2`` internally and
    returns dual gaps divided by the sample count.  The matching returned-gap
    threshold is therefore ``tol * ||y||^2 / n``.  A finite forecast is not a
    substitute for this convergence/KKT certificate.
    """

    gaps = np.asarray(dual_gaps, dtype=np.float64)
    counts = np.asarray(iterations, dtype=np.int64)
    if gaps.ndim != 1 or counts.shape != gaps.shape:
        raise RuntimeError("Elastic Net solver returned invalid convergence diagnostics")
    y_norm_per_row = float(
        np.dot(
            np.asarray(weighted_y, dtype=np.float64),
            np.asarray(weighted_y, dtype=np.float64),
        )
        / float(np.asarray(weighted_y).size)
    )
    gap_limit = float(tolerance) * y_norm_per_row
    failed = np.flatnonzero(
        ~np.isfinite(gaps)
        | (np.abs(gaps) > gap_limit)
        | (counts <= 0)
        | (counts > int(max_iter))
    )
    if convergence_warnings or failed.size:
        details = (
            f"failed_path_indices={failed.tolist()}, gap_limit={gap_limit:.17g}, "
            f"warnings={list(convergence_warnings)}"
        )
        raise RuntimeError(f"Elastic Net path failed its convergence/KKT gate: {details}")


def _nonnegative_dual_gap(value: float) -> float:
    """Normalize accepted solver round-off for publication evidence.

    ``sklearn`` can return a dual gap a few ulps below zero even though the
    convergence gate (which correctly uses its absolute magnitude) passes.
    A mathematical duality gap is nonnegative, so only that signed round-off
    is clipped to zero after the convergence and KKT checks have succeeded.
    """

    gap = float(value)
    if not math.isfinite(gap):
        raise RuntimeError("Elastic Net solver returned a nonfinite dual gap")
    return max(gap, 0.0)


def _float64_raw_gram_and_rhs(
    X: np.ndarray,
    y: np.ndarray,
    *,
    chunk_rows: int = 2_048,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate raw float64 ``X.T @ X`` and ``X.T @ y`` in row blocks."""

    sample_count, feature_count = X.shape
    gram = np.zeros((feature_count, feature_count), dtype=np.float64)
    rhs = np.zeros(feature_count, dtype=np.float64)
    for start in range(0, sample_count, chunk_rows):
        stop = min(sample_count, start + chunk_rows)
        block = np.asarray(X[start:stop], dtype=np.float64)
        target = np.asarray(y[start:stop], dtype=np.float64)
        gram += block.T @ block
        rhs += block.T @ target
    return np.ascontiguousarray(gram), np.ascontiguousarray(rhs)


def _coordinate_descent_path(
    X: np.ndarray,
    y: np.ndarray,
    *,
    rho: float,
    alphas: np.ndarray,
    max_iter: int,
    tolerance: float,
    raw_gram: np.ndarray | None,
    raw_rhs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Run one deterministic sklearn path, optionally from normal equations.

    Passing ``raw_gram`` and ``raw_rhs`` selects sklearn's Gram coordinate-
    descent implementation.  Passing both as ``None`` retains a small-matrix
    direct-solver reference used only by equivalence tests.
    """

    design = np.asarray(X)
    target = np.asarray(y)
    penalties = np.asarray(alphas, dtype=np.float64)
    if design.dtype != np.dtype("float64") or target.dtype != np.dtype("float64"):
        raise RuntimeError("Elastic Net coordinate descent requires float64 inputs")
    if design.ndim != 2 or target.shape != (design.shape[0],):
        raise ValueError("Elastic Net coordinate-descent inputs have invalid dimensions")
    if penalties.ndim != 1 or penalties.size < 1 or not np.isfinite(penalties).all():
        raise ValueError("Elastic Net coordinate-descent penalties are invalid")
    if (raw_gram is None) != (raw_rhs is None):
        raise ValueError("Elastic Net precomputed Gram and Xy must be supplied together")

    if raw_gram is None:
        precompute: bool | np.ndarray = False
        solver_rhs = None
    else:
        precompute = np.asarray(raw_gram, dtype=np.float64, order="C")
        solver_rhs = np.asarray(raw_rhs, dtype=np.float64, order="C")
        if precompute.shape != (design.shape[1], design.shape[1]):
            raise ValueError("Elastic Net precomputed Gram has invalid dimensions")
        if solver_rhs.shape != (design.shape[1],):
            raise ValueError("Elastic Net precomputed Xy has invalid dimensions")
        if not np.isfinite(precompute).all() or not np.isfinite(solver_rhs).all():
            raise FloatingPointError("Elastic Net precomputed normal equations are nonfinite")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        returned_alphas, coefficients, dual_gaps, iterations = enet_path(
            design,
            target,
            l1_ratio=float(rho),
            alphas=penalties,
            precompute=precompute,
            Xy=solver_rhs,
            copy_X=False,
            max_iter=max_iter,
            tol=tolerance,
            selection="cyclic",
            return_n_iter=True,
            check_input=False,
        )
    convergence_warnings = tuple(
        str(value.message)
        for value in caught
        if issubclass(value.category, ConvergenceWarning)
    )
    return (
        np.asarray(returned_alphas, dtype=np.float64),
        np.asarray(coefficients, dtype=np.float64),
        np.asarray(dual_gaps, dtype=np.float64),
        np.asarray(iterations, dtype=np.int64),
        convergence_warnings,
    )


def _elastic_net_dual_gaps(
    gram: np.ndarray,
    rhs: np.ndarray,
    y_second_moment: float,
    coefficients: np.ndarray,
    alphas: np.ndarray,
    *,
    rho: float,
) -> np.ndarray:
    """Compute normalized primal-dual gaps for the frozen Elastic Net objective."""

    normal = np.asarray(gram, dtype=np.float64)
    cross_product = np.asarray(rhs, dtype=np.float64)
    beta = np.asarray(coefficients, dtype=np.float64)
    penalties = np.asarray(alphas, dtype=np.float64)
    mixing = float(rho)
    if (
        beta.ndim != 2
        or normal.shape != (beta.shape[0], beta.shape[0])
        or cross_product.shape != (beta.shape[0],)
        or penalties.shape != (beta.shape[1],)
        or not 0.0 < mixing <= 1.0
        or not math.isfinite(float(y_second_moment))
        or float(y_second_moment) < 0.0
    ):
        raise ValueError("Elastic Net dual-gap inputs are invalid")

    gram_beta = normal @ beta
    beta_rhs = np.sum(beta * cross_product[:, None], axis=0)
    residual_mse = (
        float(y_second_moment)
        - 2.0 * beta_rhs
        + np.sum(beta * gram_beta, axis=0)
    )
    residual_mse = np.maximum(residual_mse, 0.0)
    l1_penalties = penalties * mixing
    l2_penalties = penalties * (1.0 - mixing)
    correlations = cross_product[:, None] - gram_beta
    dual_correlations = correlations - l2_penalties[None, :] * beta
    y_residual_inner_product = float(y_second_moment) - beta_rhs
    maximum_dual_correlation = np.max(np.abs(dual_correlations), axis=0)
    scale = np.ones_like(penalties)
    nonzero = maximum_dual_correlation > 0.0
    scale[nonzero] = np.minimum(
        1.0,
        l1_penalties[nonzero] / maximum_dual_correlation[nonzero],
    )
    scaled = scale < 1.0
    residual_term = np.where(
        scaled,
        0.5 * residual_mse * (1.0 + np.square(scale)),
        residual_mse,
    )
    gaps = (
        residual_term
        + l1_penalties * np.sum(np.abs(beta), axis=0)
        - scale * y_residual_inner_product
        + 0.5
        * l2_penalties
        * (1.0 + np.square(scale))
        * np.sum(np.square(beta), axis=0)
    )
    primal = (
        0.5 * residual_mse
        + l1_penalties * np.sum(np.abs(beta), axis=0)
        + 0.5 * l2_penalties * np.sum(np.square(beta), axis=0)
    )
    dual = primal - gaps
    numerical_floor = 128.0 * np.finfo(np.float64).eps * np.maximum(
        1.0,
        np.maximum(np.abs(primal), np.abs(dual)),
    )
    if (
        not np.isfinite(gaps).all()
        or bool(np.any(gaps < -numerical_floor))
    ):
        raise RuntimeError("Elastic Net primal-dual certificate is invalid")
    return np.maximum(gaps, 0.0)


def _active_set_backward_error_threshold(
    gram: np.ndarray,
    rhs: np.ndarray,
    coefficients: np.ndarray,
    *,
    alpha: float,
    rho: float,
) -> float:
    """Deterministic float64 guard for inactive-set entry comparisons."""

    normal = np.asarray(gram, dtype=np.float64)
    cross_product = np.asarray(rhs, dtype=np.float64)
    beta = np.asarray(coefficients, dtype=np.float64)
    beta_maximum = float(np.max(np.abs(beta), initial=0.0))
    scale = max(
        1.0,
        float(np.max(np.abs(cross_product), initial=0.0)),
        float(alpha),
        float(normal.shape[0])
        * float(np.max(np.abs(normal), initial=0.0))
        * beta_maximum,
        float(alpha) * (1.0 - float(rho)) * beta_maximum,
    )
    return (
        _ACTIVE_SET_BACKWARD_ERROR_MULTIPLIER
        * np.finfo(np.float64).eps
        * scale
    )


def _coefficient_sha256(coefficients: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(coefficients, dtype=np.float64))
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("Elastic Net coefficient hash input is invalid")
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _orthant_trace_summary(
    trace: object,
    *,
    input_features: int,
) -> dict[str, int]:
    if not isinstance(trace, list) or len(trace) > _LASSO_ORTHANT_MAX_ITERATIONS:
        raise RuntimeError("Elastic Net lasso orthant trace is invalid")
    allowed_transitions = {
        "drop_zero_origin_wrong_sign",
        "orthant_boundary",
        "drop_wrong_or_zero",
        "add_violators",
        "resolved",
    }
    cholesky_attempts = 0
    gelsy_attempts = 0
    rank_deficient_solves = 0
    maximum_active_dimension = 0
    solve_ranks: list[int] = []
    support_transitions = 0
    boundary_transitions = 0
    dropped_total = 0
    added_total = 0
    expected_support: int | None = None
    for ordinal, raw in enumerate(trace, start=1):
        if not isinstance(raw, Mapping) or set(raw) != {
            "iteration",
            "active_dimension",
            "support_after",
            "cholesky_attempted",
            "gelsy_attempted",
            "linear_solver",
            "reported_rank",
            "rank_deficient",
            "transition",
            "boundary_crossings",
            "dropped_coordinates",
            "added_coordinates",
        }:
            raise RuntimeError("Elastic Net lasso orthant trace schema differs")
        integer_keys = (
            "iteration",
            "active_dimension",
            "support_after",
            "reported_rank",
            "boundary_crossings",
            "dropped_coordinates",
            "added_coordinates",
        )
        if any(
            not isinstance(raw[key], int) or isinstance(raw[key], bool)
            for key in integer_keys
        ) or any(
            not isinstance(raw[key], bool)
            for key in ("cholesky_attempted", "gelsy_attempted", "rank_deficient")
        ):
            raise RuntimeError("Elastic Net lasso orthant trace types differ")
        if not isinstance(raw["linear_solver"], str) or not isinstance(
            raw["transition"], str
        ):
            raise RuntimeError("Elastic Net lasso orthant trace labels differ")
        iteration = int(raw["iteration"])
        active_dimension = int(raw["active_dimension"])
        support_after = int(raw["support_after"])
        rank = int(raw["reported_rank"])
        boundary = int(raw["boundary_crossings"])
        dropped = int(raw["dropped_coordinates"])
        added = int(raw["added_coordinates"])
        solver = raw["linear_solver"]
        transition = raw["transition"]
        if (
            iteration != ordinal
            or not 0 <= active_dimension <= input_features
            or not 0 <= support_after <= input_features
            or expected_support is not None
            and active_dimension != expected_support
            or not 0 <= rank <= active_dimension
            or not 0 <= boundary <= dropped <= input_features
            or not 0 <= added <= input_features
            or support_after != active_dimension - dropped + added
            or transition not in allowed_transitions
            or solver not in {"none", "cholesky", "gelsy"}
        ):
            raise RuntimeError("Elastic Net lasso orthant transition differs")
        cholesky_attempted = bool(raw["cholesky_attempted"])
        gelsy_attempted = bool(raw["gelsy_attempted"])
        rank_deficient = bool(raw["rank_deficient"])
        if solver == "none":
            valid_solver = (
                active_dimension == 0
                and not cholesky_attempted
                and not gelsy_attempted
                and rank == 0
                and not rank_deficient
            )
        elif solver == "cholesky":
            valid_solver = (
                active_dimension > 0
                and cholesky_attempted
                and not gelsy_attempted
                and rank == active_dimension
                and not rank_deficient
            )
        else:
            valid_solver = (
                active_dimension > 0
                and cholesky_attempted
                and gelsy_attempted
                and rank_deficient == (rank < active_dimension)
            )
        if not valid_solver:
            raise RuntimeError("Elastic Net lasso orthant solver evidence differs")
        if transition in {
            "drop_zero_origin_wrong_sign",
            "drop_wrong_or_zero",
        }:
            valid_action = dropped > 0 and added == 0 and boundary == 0
        elif transition == "orthant_boundary":
            valid_action = boundary > 0 and dropped > 0 and added == 0
        elif transition == "add_violators":
            valid_action = added > 0 and dropped == 0 and boundary == 0
        else:
            valid_action = dropped == 0 and added == 0 and boundary == 0
        if not valid_action or (transition == "resolved" and ordinal != len(trace)):
            raise RuntimeError("Elastic Net lasso orthant action evidence differs")
        cholesky_attempts += int(cholesky_attempted)
        gelsy_attempts += int(gelsy_attempted)
        rank_deficient_solves += int(rank_deficient)
        maximum_active_dimension = max(maximum_active_dimension, active_dimension)
        if solver in {"cholesky", "gelsy"}:
            solve_ranks.append(rank)
        support_transitions += int(support_after != active_dimension)
        boundary_transitions += boundary
        dropped_total += dropped
        added_total += added
        expected_support = support_after
    if trace and trace[-1]["transition"] != "resolved":
        raise RuntimeError("Elastic Net lasso orthant trace did not resolve")
    return {
        "orthant_iterations": len(trace),
        "orthant_cholesky_attempts": cholesky_attempts,
        "orthant_gelsy_attempts": gelsy_attempts,
        "orthant_rank_deficient_solves": rank_deficient_solves,
        "orthant_max_active_dimension": maximum_active_dimension,
        "orthant_min_solve_rank": min(solve_ranks, default=0),
        "orthant_support_transitions": support_transitions,
        "orthant_boundary_transitions": boundary_transitions,
        "orthant_dropped_coordinates": dropped_total,
        "orthant_added_coordinates": added_total,
    }


def _active_set_polish(
    *,
    rho: float,
    alpha: float,
    gram: np.ndarray,
    rhs: np.ndarray,
    coefficient_initialization: np.ndarray,
    gram_eigenvalues: np.ndarray | None = None,
    gram_eigenvectors: np.ndarray | None = None,
) -> _ActiveSetPolishResult:
    """Polish one Elastic Net point on deterministic orthants.

    Every restricted system is the raw, normalized frozen objective; no
    feature rescaling or diagonal jitter is introduced.  A coefficient whose
    fixed-orthant solution would change sign first moves to the zero boundary
    and is dropped, so an opposite sign can enter only through the full
    inactive KKT correlation on a later iteration.  The caller independently
    recomputes the full KKT and dual certificates before accepting a result.
    """

    normal = np.asarray(gram, dtype=np.float64)
    cross_product = np.asarray(rhs, dtype=np.float64)
    beta = np.asarray(coefficient_initialization, dtype=np.float64).copy()
    mixing = float(rho)
    penalty = float(alpha)
    if (
        normal.ndim != 2
        or normal.shape[0] != normal.shape[1]
        or cross_product.shape != (normal.shape[0],)
        or beta.shape != cross_product.shape
        or not 0.0 < mixing <= 1.0
        or not math.isfinite(penalty)
        or penalty <= 0.0
        or not np.isfinite(normal).all()
        or not np.isfinite(cross_product).all()
        or not np.isfinite(beta).all()
    ):
        raise ValueError("Elastic Net active-set polishing inputs are invalid")
    if (gram_eigenvalues is None) != (gram_eigenvectors is None):
        raise ValueError("Elastic Net cached Gram eigensystem must be supplied together")
    eigenvalues: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    if gram_eigenvalues is not None:
        eigenvalues = np.asarray(gram_eigenvalues, dtype=np.float64)
        eigenvectors = np.asarray(gram_eigenvectors, dtype=np.float64)
        if (
            eigenvalues.shape != (normal.shape[0],)
            or eigenvectors.shape != normal.shape
            or not np.isfinite(eigenvalues).all()
            or not np.isfinite(eigenvectors).all()
        ):
            raise ValueError("Elastic Net cached Gram eigensystem is invalid")

    l1_penalty = penalty * mixing
    l2_penalty = penalty * (1.0 - mixing)
    active = np.abs(beta) > NONZERO_THRESHOLD
    beta[~active] = 0.0
    signs = np.zeros(beta.shape, dtype=np.float64)
    signs[active] = np.sign(beta[active])
    seen_states: set[tuple[bytes, bytes]] = set()
    trace: list[dict[str, Any]] = []
    maximum_iterations = (
        _LASSO_ORTHANT_MAX_ITERATIONS
        if mixing == 1.0
        else _ACTIVE_SET_MAX_ITERATIONS
    )

    def record(
        *,
        iteration: int,
        active_dimension: int,
        cholesky_attempted: bool,
        gelsy_attempted: bool,
        linear_solver: str,
        reported_rank: int,
        transition: str,
        support_after: int | None = None,
        boundary_crossings: int = 0,
        dropped_coordinates: int = 0,
        added_coordinates: int = 0,
    ) -> None:
        trace.append(
            {
                "iteration": int(iteration),
                "active_dimension": int(active_dimension),
                "support_after": int(
                    active_dimension if support_after is None else support_after
                ),
                "cholesky_attempted": bool(cholesky_attempted),
                "gelsy_attempted": bool(gelsy_attempted),
                "linear_solver": str(linear_solver),
                "reported_rank": int(reported_rank),
                "rank_deficient": bool(
                    linear_solver == "gelsy"
                    and reported_rank < active_dimension
                ),
                "transition": str(transition),
                "boundary_crossings": int(boundary_crossings),
                "dropped_coordinates": int(dropped_coordinates),
                "added_coordinates": int(added_coordinates),
            }
        )

    for iteration in range(1, maximum_iterations + 1):
        state = (active.tobytes(), signs.astype(np.int8, copy=False).tobytes())
        if state in seen_states:
            active_dimension = int(np.count_nonzero(active))
            record(
                iteration=iteration,
                active_dimension=active_dimension,
                cholesky_attempted=False,
                gelsy_attempted=False,
                linear_solver="none",
                reported_rank=0,
                transition="cycle",
            )
            return _ActiveSetPolishResult(beta, len(trace), False, trace)
        seen_states.add(state)
        active_indices = np.flatnonzero(active)
        active_dimension = int(active_indices.size)
        candidate = np.zeros_like(beta)
        cholesky_attempted = False
        gelsy_attempted = False
        linear_solver = "none"
        reported_rank = 0
        if active_indices.size:
            fixed_rhs = (
                cross_product[active_indices]
                - l1_penalty * signs[active_indices]
            )
            if not np.isfinite(fixed_rhs).all():
                raise RuntimeError("Elastic Net active-set restricted RHS is nonfinite")
            try:
                if (
                    mixing < 1.0
                    and active_indices.size == normal.shape[0]
                    and eigenvalues is not None
                ):
                    assert eigenvectors is not None
                    denominators = eigenvalues + l2_penalty
                    if (
                        not np.isfinite(denominators).all()
                        or bool(np.any(denominators <= 0.0))
                    ):
                        raise RuntimeError(
                            "Elastic Net active-set spectral system is invalid"
                        )
                    candidate_values = eigenvectors @ (
                        (eigenvectors.T @ fixed_rhs) / denominators
                    )
                    linear_solver = "spectral"
                    reported_rank = active_dimension
                else:
                    restricted = np.asarray(
                        normal[np.ix_(active_indices, active_indices)],
                        dtype=np.float64,
                        order="C",
                    ).copy()
                    restricted.flat[:: restricted.shape[0] + 1] += l2_penalty
                    if (
                        not np.isfinite(restricted).all()
                        or not np.isfinite(fixed_rhs).all()
                    ):
                        raise RuntimeError(
                            "Elastic Net active-set restricted system is nonfinite"
                        )
                    cholesky_attempted = True
                    try:
                        cholesky = np.linalg.cholesky(restricted)
                    except np.linalg.LinAlgError:
                        if mixing < 1.0:
                            raise
                        gelsy_attempted = True
                        try:
                            candidate_values, _, rank, singular_values = lstsq(
                                restricted,
                                fixed_rhs,
                                cond=None,
                                lapack_driver="gelsy",
                                check_finite=False,
                            )
                        except (np.linalg.LinAlgError, ValueError) as exc:
                            raise RuntimeError(
                                "Elastic Net lasso GELSY restricted solve failed"
                            ) from exc
                        if (
                            np.asarray(candidate_values).shape
                            != (active_dimension,)
                            or not isinstance(rank, (int, np.integer))
                            or isinstance(rank, (bool, np.bool_))
                            or not 0 <= int(rank) <= active_dimension
                            or singular_values is not None
                        ):
                            raise RuntimeError(
                                "Elastic Net lasso GELSY evidence is invalid"
                            )
                        linear_solver = "gelsy"
                        reported_rank = int(rank)
                    else:
                        candidate_values = solve_triangular(
                            cholesky.T,
                            solve_triangular(
                                cholesky,
                                fixed_rhs,
                                lower=True,
                                check_finite=False,
                            ),
                            lower=False,
                            check_finite=False,
                        )
                        linear_solver = "cholesky"
                        reported_rank = active_dimension
            except np.linalg.LinAlgError:
                record(
                    iteration=iteration,
                    active_dimension=active_dimension,
                    cholesky_attempted=cholesky_attempted,
                    gelsy_attempted=gelsy_attempted,
                    linear_solver=linear_solver,
                    reported_rank=reported_rank,
                    transition="linear_failure",
                )
                return _ActiveSetPolishResult(beta, len(trace), False, trace)
            if (
                np.asarray(candidate_values).shape != (active_dimension,)
                or not np.isfinite(candidate_values).all()
            ):
                raise RuntimeError(
                    "Elastic Net active-set restricted solution is invalid"
                )

            current_values = beta[active_indices]
            active_signs = signs[active_indices]
            zero_origin_wrong_sign = (
                (np.abs(current_values) <= NONZERO_THRESHOLD)
                & (candidate_values * active_signs <= NONZERO_THRESHOLD)
            )
            if bool(np.any(zero_origin_wrong_sign)):
                dropped = active_indices[zero_origin_wrong_sign]
                beta[dropped] = 0.0
                active[dropped] = False
                signs[dropped] = 0.0
                record(
                    iteration=iteration,
                    active_dimension=active_dimension,
                    support_after=int(np.count_nonzero(active)),
                    cholesky_attempted=cholesky_attempted,
                    gelsy_attempted=gelsy_attempted,
                    linear_solver=linear_solver,
                    reported_rank=reported_rank,
                    transition="drop_zero_origin_wrong_sign",
                    dropped_coordinates=int(dropped.size),
                )
                continue

            delta = candidate_values - current_values
            toward_zero = current_values * delta < 0.0
            ratios = np.full(current_values.shape, np.inf, dtype=np.float64)
            ratios[toward_zero] = -current_values[toward_zero] / delta[toward_zero]
            crossings = toward_zero & (ratios >= 0.0) & (ratios < 1.0)
            if bool(np.any(crossings)):
                boundary = float(np.min(ratios[crossings]))
                beta[active_indices] = current_values + boundary * delta
                tie_limit = (
                    _ACTIVE_SET_ORTHANT_TIE_MULTIPLIER
                    * np.finfo(np.float64).eps
                    * max(1.0, abs(boundary))
                )
                tied = crossings & (ratios <= boundary + tie_limit)
                dropped = active_indices[
                    tied | (np.abs(beta[active_indices]) <= NONZERO_THRESHOLD)
                ]
                beta[dropped] = 0.0
                active[dropped] = False
                signs[dropped] = 0.0
                record(
                    iteration=iteration,
                    active_dimension=active_dimension,
                    support_after=int(np.count_nonzero(active)),
                    cholesky_attempted=cholesky_attempted,
                    gelsy_attempted=gelsy_attempted,
                    linear_solver=linear_solver,
                    reported_rank=reported_rank,
                    transition="orthant_boundary",
                    boundary_crossings=int(np.count_nonzero(tied)),
                    dropped_coordinates=int(dropped.size),
                )
                continue

            candidate[active_indices] = candidate_values
            wrong_or_zero = (
                (candidate_values * active_signs <= 0.0)
                | (np.abs(candidate_values) <= NONZERO_THRESHOLD)
            )
            if bool(np.any(wrong_or_zero)):
                kept = ~wrong_or_zero
                beta.fill(0.0)
                beta[active_indices[kept]] = candidate_values[kept]
                dropped = active_indices[wrong_or_zero]
                active[dropped] = False
                signs[dropped] = 0.0
                record(
                    iteration=iteration,
                    active_dimension=active_dimension,
                    support_after=int(np.count_nonzero(active)),
                    cholesky_attempted=cholesky_attempted,
                    gelsy_attempted=gelsy_attempted,
                    linear_solver=linear_solver,
                    reported_rank=reported_rank,
                    transition="drop_wrong_or_zero",
                    dropped_coordinates=int(dropped.size),
                )
                continue
        beta = candidate
        smooth = cross_product - normal @ beta - l2_penalty * beta
        if not np.isfinite(smooth).all():
            raise RuntimeError("Elastic Net active-set smooth correlation is nonfinite")
        numerical_threshold = _active_set_backward_error_threshold(
            normal,
            cross_product,
            beta,
            alpha=penalty,
            rho=mixing,
        )
        violating = np.flatnonzero(
            (~active)
            & (np.abs(smooth) > l1_penalty + numerical_threshold)
        )
        if violating.size == 0:
            record(
                iteration=iteration,
                active_dimension=active_dimension,
                support_after=int(np.count_nonzero(active)),
                cholesky_attempted=cholesky_attempted,
                gelsy_attempted=gelsy_attempted,
                linear_solver=linear_solver,
                reported_rank=reported_rank,
                transition="resolved",
            )
            return _ActiveSetPolishResult(beta, len(trace), True, trace)
        # ``flatnonzero`` fixes feature-index order; all violators enter at once.
        active[violating] = True
        signs[violating] = np.sign(smooth[violating])
        beta[violating] = 0.0
        record(
            iteration=iteration,
            active_dimension=active_dimension,
            support_after=int(np.count_nonzero(active)),
            cholesky_attempted=cholesky_attempted,
            gelsy_attempted=gelsy_attempted,
            linear_solver=linear_solver,
            reported_rank=reported_rank,
            transition="add_violators",
            added_coordinates=int(violating.size),
        )

    return _ActiveSetPolishResult(beta, len(trace), False, trace)


def _contiguous_index_runs(indices: np.ndarray) -> tuple[np.ndarray, ...]:
    """Partition sorted unique path indices into maximal contiguous runs."""

    values = np.asarray(indices, dtype=np.int64)
    if (
        values.ndim != 1
        or bool(np.any(values < 0))
        or (values.size > 1 and bool(np.any(np.diff(values) <= 0)))
    ):
        raise ValueError("Elastic Net strict-refinement indices are invalid")
    if values.size == 0:
        return ()
    boundaries = np.flatnonzero(np.diff(values) != 1) + 1
    return tuple(
        np.asarray(run, dtype=np.int64)
        for run in np.split(values, boundaries)
    )


def _working_set_coordinate_descent_path(
    *,
    rho: float,
    alphas: np.ndarray,
    tolerance: float,
    gram: np.ndarray,
    rhs: np.ndarray,
    y_second_moment: float,
    gram_eigenvalues: np.ndarray | None = None,
    gram_eigenvectors: np.ndarray | None = None,
) -> _WorkingSetPathResult:
    """Solve and, only when necessary, certificate-refine a Hessian path.

    ``QuadraticHessian`` minimizes ``0.5 * beta.T @ gram @ beta - rhs.T @ beta``
    before the unchanged Elastic Net penalty.  This is the weighted-centered
    least-squares objective up to the response-only constant, while avoiding
    repeated coordinate updates across every training row.  A deterministic
    coarse path is followed, only for failed certificate points, by an exact
    raw-scale active-set polish. For ``rho < 1``, maximal contiguous runs of
    still-unresolved points retain the v5 strict Anderson-CD fallback and
    hard-fail if it cannot pass the independent certificates. Failed lasso
    points instead use a singular-safe Cholesky/GELSY orthant polish with the
    same fail-closed policy.
    """

    normal = np.asarray(gram, dtype=np.float64, order="F")
    linear = -np.asarray(rhs, dtype=np.float64)
    penalties = np.asarray(alphas, dtype=np.float64)
    if (
        normal.ndim != 2
        or normal.shape[0] != normal.shape[1]
        or linear.shape != (normal.shape[0],)
        or penalties.ndim != 1
        or penalties.size < 1
        or not np.isfinite(normal).all()
        or not np.isfinite(linear).all()
        or not np.isfinite(penalties).all()
        or np.any(penalties <= 0.0)
        or not 0.0 < float(rho) <= 1.0
    ):
        raise ValueError("Elastic Net Hessian working-set path inputs are invalid")
    symmetry_limit = 128.0 * np.finfo(np.float64).eps * max(
        1.0,
        float(np.max(np.abs(normal))),
    )
    if not np.allclose(normal, normal.T, rtol=0.0, atol=symmetry_limit):
        raise RuntimeError("Elastic Net normalized Hessian is not symmetric")

    coarse_tolerance = (
        float(tolerance) * _WORKING_SET_COARSE_TOLERANCE_MULTIPLIER
    )
    refinement_tolerance = (
        float(tolerance) * _WORKING_SET_REFINEMENT_TOLERANCE_MULTIPLIER
    )

    def solve(
        solve_alphas: np.ndarray,
        *,
        solver_tolerance: float,
        coefficient_initialization: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        requested = np.asarray(solve_alphas, dtype=np.float64)
        solver = AndersonCD(
            max_iter=_WORKING_SET_MAX_ITERATIONS,
            max_epochs=_WORKING_SET_MAX_EPOCHS,
            p0=min(_WORKING_SET_INITIAL_SIZE, normal.shape[1]),
            tol=float(solver_tolerance),
            ws_strategy=_WORKING_SET_STRATEGY,
            fit_intercept=False,
            warm_start=False,
            verbose=0,
        )
        penalty = L1_plus_L2(
            alpha=float(requested[0]),
            l1_ratio=float(rho),
            positive=False,
        )
        raw_alphas, raw_beta, raw_stops, raw_counts = solver.path(
            normal,
            linear,
            QuadraticHessian(),
            penalty,
            alphas=requested,
            w_init=coefficient_initialization,
            return_n_iter=True,
        )
        returned = np.asarray(raw_alphas, dtype=np.float64)
        beta = np.asarray(raw_beta, dtype=np.float64)
        stops = np.asarray(raw_stops, dtype=np.float64)
        counts = np.asarray(raw_counts, dtype=np.int64)
        if (
            returned.shape != requested.shape
            or beta.shape != (normal.shape[1], requested.size)
            or stops.shape != requested.shape
            or counts.shape != requested.shape
            or not np.array_equal(returned, requested)
            or not np.isfinite(beta).all()
        ):
            raise RuntimeError("Elastic Net working-set solver changed the frozen path")
        failed = np.flatnonzero(
            ~np.isfinite(stops)
            | (stops < 0.0)
            | (stops > float(solver_tolerance))
            | (counts < 0)
            | (counts >= _WORKING_SET_MAX_ITERATIONS)
        )
        if failed.size:
            raise RuntimeError(
                "Elastic Net working-set path failed its convergence gate: "
                f"indices={failed.tolist()}, max_stop={float(np.nanmax(stops)):.17g}"
            )
        return returned, beta, stops, counts

    returned, coarse_beta, coarse_stops, coarse_counts = solve(
        penalties,
        solver_tolerance=coarse_tolerance,
        coefficient_initialization=None,
    )
    coarse_dual_gaps = _elastic_net_dual_gaps(
        gram,
        rhs,
        y_second_moment,
        coarse_beta,
        penalties,
        rho=float(rho),
    )
    gap_limit = float(tolerance) * float(y_second_moment)
    coarse_kkt, coarse_kkt_limits = _path_kkt_violations(
        gram,
        rhs,
        coarse_beta,
        penalties,
        rho=float(rho),
        tolerance=float(tolerance),
    )
    failed_certificate = np.flatnonzero(
        ~np.isfinite(coarse_dual_gaps)
        | (coarse_dual_gaps < 0.0)
        | (coarse_dual_gaps > gap_limit)
        | ~np.isfinite(coarse_kkt)
        | (coarse_kkt < 0.0)
        | (coarse_kkt > coarse_kkt_limits)
    )

    beta = coarse_beta.copy(order="F")
    final_stops = coarse_stops.copy()
    final_stop_limits = np.full(penalties.shape, coarse_tolerance, dtype=np.float64)
    refinement_counts = np.zeros(penalties.shape, dtype=np.int64)
    solver_passes = np.ones(penalties.shape, dtype=np.int64)
    active_iteration_counts = np.zeros(penalties.shape, dtype=np.int64)
    orthant_traces: list[list[dict[str, Any]]] = [
        [] for _ in range(penalties.size)
    ]
    refinement_routes = ["none" for _ in range(penalties.size)]
    strict_required: list[int] = []
    for path_index in failed_certificate:
        single_alpha = penalties[path_index : path_index + 1]
        if float(rho) <= 1.0:
            polished = _active_set_polish(
                rho=float(rho),
                alpha=float(single_alpha[0]),
                gram=normal,
                rhs=-linear,
                coefficient_initialization=coarse_beta[:, path_index],
                gram_eigenvalues=gram_eigenvalues,
                gram_eigenvectors=gram_eigenvectors,
            )
            active_iteration_counts[path_index] = int(polished.iterations)
            if float(rho) == 1.0:
                orthant_traces[path_index] = [
                    dict(value) for value in polished.trace
                ]
            active_candidate = np.asarray(
                polished.coefficients,
                dtype=np.float64,
            )
            if polished.resolved:
                candidate_matrix = active_candidate[:, None]
                try:
                    candidate_dual_gap = _elastic_net_dual_gaps(
                        normal,
                        -linear,
                        y_second_moment,
                        candidate_matrix,
                        single_alpha,
                        rho=float(rho),
                    )
                    candidate_kkt, candidate_kkt_limits = _path_kkt_violations(
                        normal,
                        -linear,
                        candidate_matrix,
                        single_alpha,
                        rho=float(rho),
                        tolerance=float(tolerance),
                    )
                except RuntimeError:
                    candidate_dual_gap = np.asarray([math.inf], dtype=np.float64)
                    candidate_kkt = np.asarray([math.inf], dtype=np.float64)
                    candidate_kkt_limits = np.asarray(
                        [float(tolerance)],
                        dtype=np.float64,
                    )
                if (
                    np.isfinite(candidate_dual_gap).all()
                    and candidate_dual_gap[0] <= gap_limit
                    and np.isfinite(candidate_kkt).all()
                    and np.isfinite(candidate_kkt_limits).all()
                    and candidate_kkt_limits[0] > 0.0
                    and candidate_kkt[0] <= candidate_kkt_limits[0]
                ):
                    beta[:, path_index] = active_candidate
                    final_stops[path_index] = (
                        candidate_kkt[0]
                        * float(tolerance)
                        / candidate_kkt_limits[0]
                    )
                    final_stop_limits[path_index] = float(tolerance)
                    refinement_counts[path_index] = active_iteration_counts[
                        path_index
                    ]
                    solver_passes[path_index] = 2
                    refinement_routes[path_index] = (
                        "rho_one_singular_safe_orthant"
                        if float(rho) == 1.0
                        else "rho_lt_one_active_set"
                    )
                    continue
        if float(rho) == 1.0:
            raise RuntimeError(
                "Elastic Net bounded lasso orthant polish failed: "
                f"path_index={int(path_index)}, resolved={polished.resolved}"
            )
        strict_required.append(int(path_index))

    for run in _contiguous_index_runs(
        np.asarray(strict_required, dtype=np.int64)
    ):
        first_index = int(run[0])
        stop_index = int(run[-1]) + 1
        grouped_alphas = penalties[first_index:stop_index]
        _, refined_beta, refined_stops, refined_iterations = solve(
            grouped_alphas,
            solver_tolerance=refinement_tolerance,
            coefficient_initialization=coarse_beta[:, first_index].copy(),
        )
        beta[:, run] = refined_beta
        final_stops[run] = refined_stops
        final_stop_limits[run] = refinement_tolerance
        refinement_counts[run] = (
            active_iteration_counts[run] + refined_iterations
        )
        solver_passes[run] = 2
        for path_index in run:
            refinement_routes[int(path_index)] = "rho_lt_one_grouped_strict_cd"

    post_secondary_dual_gaps = _elastic_net_dual_gaps(
        gram,
        rhs,
        y_second_moment,
        beta,
        penalties,
        rho=float(rho),
    )
    post_secondary_kkt, post_secondary_kkt_limits = _path_kkt_violations(
        gram,
        rhs,
        beta,
        penalties,
        rho=float(rho),
        tolerance=float(tolerance),
    )
    failed_post_secondary = np.flatnonzero(
        ~np.isfinite(post_secondary_dual_gaps)
        | (post_secondary_dual_gaps < 0.0)
        | (post_secondary_dual_gaps > gap_limit)
        | ~np.isfinite(post_secondary_kkt)
        | (post_secondary_kkt < 0.0)
        | (post_secondary_kkt > post_secondary_kkt_limits)
    )
    active_polished = (
        (solver_passes == 2)
        & (final_stop_limits == float(tolerance))
    )
    final_stops[active_polished] = (
        post_secondary_kkt[active_polished]
        * float(tolerance)
        / post_secondary_kkt_limits[active_polished]
    )
    secondary_iteration_counts = refinement_counts.copy()
    tertiary_iteration_counts = np.zeros(penalties.shape, dtype=np.int64)
    secondary_stops = final_stops.copy()
    secondary_stop_limits = final_stop_limits.copy()

    if failed_post_secondary.size:
        raise RuntimeError(
            "Elastic Net certificate refinement failed its secondary gate: "
            f"indices={failed_post_secondary.tolist()}, "
            f"max_gap={float(np.nanmax(post_secondary_dual_gaps)):.17g}, "
            f"gap_limit={gap_limit:.17g}"
        )

    dual_gaps = _elastic_net_dual_gaps(
        gram,
        rhs,
        y_second_moment,
        beta,
        penalties,
        rho=float(rho),
    )
    final_kkt, final_kkt_limits = _path_kkt_violations(
        gram,
        rhs,
        beta,
        penalties,
        rho=float(rho),
        tolerance=float(tolerance),
    )
    failed_final = np.flatnonzero(
        ~np.isfinite(dual_gaps)
        | (dual_gaps < 0.0)
        | (dual_gaps > gap_limit)
        | ~np.isfinite(final_kkt)
        | (final_kkt < 0.0)
        | (final_kkt > final_kkt_limits)
    )
    if failed_final.size:
        raise RuntimeError(
            "Elastic Net certificate refinement failed its final gate: "
            f"indices={failed_final.tolist()}, "
            f"max_gap={float(np.nanmax(dual_gaps)):.17g}, "
            f"gap_limit={gap_limit:.17g}"
        )
    total_counts = coarse_counts + refinement_counts
    if bool(
        np.any(
            total_counts
            >= (
                _WORKING_SET_MAX_PASSES * _WORKING_SET_MAX_ITERATIONS
                + _ACTIVE_SET_MAX_ITERATIONS
            )
        )
    ):
        raise RuntimeError("Elastic Net certificate refinement iteration evidence is invalid")
    initial_coefficient_sha256s = tuple(
        _coefficient_sha256(coarse_beta[:, path_index])
        for path_index in range(penalties.size)
    )
    final_coefficient_sha256s = tuple(
        _coefficient_sha256(beta[:, path_index])
        for path_index in range(penalties.size)
    )
    return _WorkingSetPathResult(
        alphas=returned,
        coefficients=beta,
        initial_dual_gaps=coarse_dual_gaps,
        post_secondary_dual_gaps=post_secondary_dual_gaps,
        dual_gaps=dual_gaps,
        initial_kkt_maximum=coarse_kkt,
        initial_kkt_limits=coarse_kkt_limits,
        post_secondary_kkt_maximum=post_secondary_kkt,
        post_secondary_kkt_limits=post_secondary_kkt_limits,
        total_iterations=total_counts,
        initial_iterations=coarse_counts,
        secondary_iterations=secondary_iteration_counts,
        tertiary_iterations=tertiary_iteration_counts,
        refinement_iterations=refinement_counts,
        solver_passes=solver_passes,
        initial_stop_criteria=coarse_stops,
        secondary_stop_criteria=secondary_stops,
        final_stop_criteria=final_stops,
        initial_stop_limits=np.full(
            penalties.shape,
            coarse_tolerance,
            dtype=np.float64,
        ),
        secondary_stop_limits=secondary_stop_limits,
        final_stop_limits=final_stop_limits,
        orthant_traces=tuple(orthant_traces),
        refinement_routes=tuple(refinement_routes),
        initial_coefficient_sha256s=initial_coefficient_sha256s,
        final_coefficient_sha256s=final_coefficient_sha256s,
    )


def _path_kkt_violations(
    gram: np.ndarray,
    rhs: np.ndarray,
    coefficients: np.ndarray,
    alphas: np.ndarray,
    *,
    rho: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact path stationarity from the float64 normal equations."""

    beta = np.asarray(coefficients, dtype=np.float64)
    penalties = np.asarray(alphas, dtype=np.float64)
    normal = np.asarray(gram, dtype=np.float64)
    cross_product = np.asarray(rhs, dtype=np.float64)
    if beta.ndim != 2 or normal.shape != (beta.shape[0], beta.shape[0]):
        raise RuntimeError("Elastic Net coefficients have invalid KKT dimensions")
    if penalties.shape != (beta.shape[1],) or cross_product.shape != (beta.shape[0],):
        raise RuntimeError("Elastic Net alphas have invalid KKT dimensions")
    correlations = cross_product[:, None] - normal @ beta
    smooth = correlations - penalties[None, :] * (1.0 - float(rho)) * beta
    l1 = penalties[None, :] * float(rho)
    active = np.abs(beta) > NONZERO_THRESHOLD
    violations = np.where(
        active,
        np.abs(smooth - l1 * np.sign(beta)),
        np.maximum(np.abs(smooth) - l1, 0.0),
    )
    maximum = np.max(violations, axis=0)
    scale = np.maximum(
        1.0,
        np.maximum(np.max(np.abs(correlations), axis=0), penalties),
    )
    limits = float(tolerance) * scale
    return maximum, limits


def _require_kkt(
    maximum: np.ndarray,
    limits: np.ndarray,
    *,
    label: str,
) -> None:
    values = np.asarray(maximum, dtype=np.float64)
    bounds = np.asarray(limits, dtype=np.float64)
    failed = np.flatnonzero(
        ~np.isfinite(values) | ~np.isfinite(bounds) | (values < 0.0) | (values > bounds)
    )
    if failed.size:
        raise RuntimeError(
            f"Elastic Net {label} failed its KKT gate: indices={failed.tolist()}, "
            f"max_violation={float(np.nanmax(values)):.17g}"
        )


def _zero_signal_result(
    *,
    fold_id: str,
    y_mean: float,
    validation_rows: int,
    input_features: int,
) -> tuple[np.ndarray, tuple[ElasticNetFitReceipt, ...]]:
    configs = elastic_net_configurations()
    # Methodology v1 explicitly requires a literal zero forecast when the
    # fold has no weighted linear signal.  The weighted target mean should
    # already be zero under equal month mass, but it is not used as a
    # floating-point substitute for that rule.
    predictions = np.zeros((validation_rows, len(configs)), dtype=np.float32)
    zero_coefficient_sha256 = _coefficient_sha256(
        np.zeros(input_features, dtype=np.float64)
    )
    receipts = tuple(
        ElasticNetFitReceipt(
            config_id=config.config_id,
            fold_id=fold_id,
            input_features=input_features,
            train_rows=0,
            outer_validation_rows=validation_rows,
            lambda_ref=0.0,
            lambda_value=0.0,
            rho=config.rho,
            lambda_index=config.lambda_index,
            nonzero_coefficients=0,
            initial_dual_gap=0.0,
            post_secondary_dual_gap=0.0,
            dual_gap=0.0,
            dual_gap_limit=0.0,
            initial_kkt_max_violation=0.0,
            initial_kkt_limit=TOLERANCE,
            post_secondary_kkt_max_violation=0.0,
            post_secondary_kkt_limit=TOLERANCE,
            kkt_max_violation=0.0,
            kkt_limit=TOLERANCE,
            iterations=0,
            initial_iterations=0,
            secondary_iterations=0,
            tertiary_iterations=0,
            refinement_iterations=0,
            solver_passes=1,
            initial_solver_stop_criterion=0.0,
            initial_solver_stop_limit=TOLERANCE,
            secondary_solver_stop_criterion=0.0,
            secondary_solver_stop_limit=TOLERANCE,
            solver_stop_criterion=0.0,
            solver_stop_limit=TOLERANCE,
            refinement_route="zero_signal",
            initial_coefficient_sha256=zero_coefficient_sha256,
            final_coefficient_sha256=zero_coefficient_sha256,
            orthant_iterations=0,
            orthant_cholesky_attempts=0,
            orthant_gelsy_attempts=0,
            orthant_rank_deficient_solves=0,
            orthant_max_active_dimension=0,
            orthant_min_solve_rank=0,
            orthant_support_transitions=0,
            orthant_boundary_transitions=0,
            orthant_dropped_coordinates=0,
            orthant_added_coordinates=0,
            orthant_trace=[],
            converged=True,
            elapsed_seconds=0.0,
            solver="zero_signal",
            dtype="float32",
        )
        for config in configs
    )
    return predictions, receipts


def _fit_predict_outer_fold_impl(
    *,
    fold_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_outer_validation: np.ndarray,
    work_path: Path | None = None,
    max_iter: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE,
    checkpoint_parent_job_id: str | None = None,
    _test_interrupt_after_rho_checkpoints: int | None = None,
) -> tuple[np.ndarray, tuple[ElasticNetFitReceipt, ...]]:
    """Fit all 250 fold-relative configurations as five warm paths."""

    started = time.monotonic()
    features, target, weights = _validated(X_train, y_train, w_train)
    validation = np.asarray(X_outer_validation, dtype=np.float32)
    if validation.ndim != 2 or validation.shape[1] != features.shape[1] or not np.isfinite(validation).all():
        raise ValueError("Invalid Elastic Net validation matrix")
    reference, y_mean, x_mean = weighted_lambda_reference(features, target, weights)
    if reference == 0.0:
        predictions, receipts = _zero_signal_result(
            fold_id=fold_id,
            y_mean=y_mean,
            validation_rows=validation.shape[0],
            input_features=features.shape[1],
        )
        receipts = tuple(
            ElasticNetFitReceipt(
                **{
                    **receipt.as_dict(),
                    "train_rows": features.shape[0],
                    "elapsed_seconds": float(time.monotonic() - started),
                }
            )
            for receipt in receipts
        )
        return predictions, receipts
    checkpoint_weights = weights * (
        float(weights.size) / float(np.sum(weights, dtype=np.float64))
    )
    checkpoint_weighted_target = (
        target - float(y_mean)
    ) * np.sqrt(checkpoint_weights, dtype=np.float64)
    checkpoint_y_second_moment = float(
        np.dot(checkpoint_weighted_target, checkpoint_weighted_target)
        / float(target.size)
    )
    if (
        not math.isfinite(checkpoint_y_second_moment)
        or checkpoint_y_second_moment <= 0.0
    ):
        raise RuntimeError("Elastic Net checkpoint response scale is invalid")
    if checkpoint_parent_job_id is not None:
        if work_path is None:
            raise ValueError("Elastic Net rho checkpoints require a disk-backed work_path")
        if _AUTHORITATIVE_PARENT_JOB.fullmatch(checkpoint_parent_job_id) is None:
            raise ValueError("Elastic Net rho checkpoint parent job identity is invalid")
    if _test_interrupt_after_rho_checkpoints is not None:
        if (
            not fold_id.startswith("NONAUTHORITATIVE_TEST::")
            or checkpoint_parent_job_id is None
            or not isinstance(_test_interrupt_after_rho_checkpoints, int)
            or isinstance(_test_interrupt_after_rho_checkpoints, bool)
            or not 1 <= _test_interrupt_after_rho_checkpoints <= len(RHO_GRID)
        ):
            raise ValueError(
                "The Elastic Net interruption hook is restricted to checkpointed "
                "nonauthoritative tests"
            )
    weighted_design_bytes = int(features.size * np.dtype(np.float64).itemsize)
    if work_path is None and weighted_design_bytes > MAX_IN_MEMORY_WEIGHTED_DESIGN_BYTES:
        raise ValueError(
            "A disk-backed work_path is required for a large Elastic Net weighted design"
        )

    weighted_X: np.ndarray | None = None
    scratch_requested = work_path is not None
    configs = elastic_net_configurations()
    prediction_by_id: dict[str, np.ndarray] = {}
    receipt_by_id: dict[str, ElasticNetFitReceipt] = {}
    resumed_rho_indices: set[int] = set()
    newly_published_checkpoints = 0

    def ordered_result() -> tuple[np.ndarray, tuple[ElasticNetFitReceipt, ...]]:
        ordered_predictions = np.column_stack(
            [prediction_by_id[config.config_id] for config in configs]
        ).astype(np.float32, copy=False)
        if (
            ordered_predictions.shape != (validation.shape[0], len(configs))
            or not np.isfinite(ordered_predictions).all()
        ):
            raise FloatingPointError("Elastic Net emitted invalid predictions")
        ordered_receipts = tuple(
            receipt_by_id[config.config_id] for config in configs
        )
        return ordered_predictions, ordered_receipts

    try:
        if checkpoint_parent_job_id is not None:
            assert work_path is not None
            for rho_index, rho in enumerate(RHO_GRID):
                group = tuple(config for config in configs if config.rho == rho)
                alphas = np.asarray(
                    [lambda_value(config, reference) for config in group],
                    dtype=np.float64,
                )
                identity = _checkpoint_identity(
                    parent_job_id=checkpoint_parent_job_id,
                    fold_id=fold_id,
                    rho_index=rho_index,
                    rho=rho,
                    configurations=group,
                    alphas=alphas,
                    lambda_ref=reference,
                    train_rows=features.shape[0],
                    input_features=features.shape[1],
                    outer_validation_rows=validation.shape[0],
                    max_iter=max_iter,
                    tolerance=tolerance,
                    y_second_moment=checkpoint_y_second_moment,
                )
                cached = _load_rho_checkpoint(
                    work_root=work_path.parent,
                    identity=identity,
                    fold_id=fold_id,
                    configurations=group,
                    alphas=alphas,
                    lambda_ref=reference,
                    train_rows=features.shape[0],
                    input_features=features.shape[1],
                    outer_validation_rows=validation.shape[0],
                )
                if cached is None:
                    continue
                path_predictions, path_receipts = cached
                for path_index, config in enumerate(group):
                    prediction_by_id[config.config_id] = path_predictions[
                        :, path_index
                    ]
                    receipt_by_id[config.config_id] = path_receipts[path_index]
                resumed_rho_indices.add(rho_index)
                _emit_path_progress(
                    fold_id=fold_id,
                    rho_index=rho_index,
                    rho=rho,
                    status="resumed from authenticated checkpoint",
                )
            if len(resumed_rho_indices) == len(RHO_GRID):
                return ordered_result()
            _emit_path_progress(
                fold_id=fold_id,
                rho_index=min(
                    set(range(len(RHO_GRID))).difference(resumed_rho_indices)
                ),
                rho=RHO_GRID[
                    min(set(range(len(RHO_GRID))).difference(resumed_rho_indices))
                ],
                status=(
                    "is the first unfinished path; preparing the shared float64 "
                    "weighted design and normal equations"
                ),
            )
        weighted_X, weighted_y = prepare_weighted_centered_design(
            features,
            target,
            weights,
            x_mean=x_mean,
            y_mean=y_mean,
            destination=work_path,
        )
        elapsed_offset = time.monotonic()
        raw_gram, raw_rhs = _float64_raw_gram_and_rhs(weighted_X, weighted_y)
        normal_equation_scale = float(weighted_X.shape[0])
        gram = raw_gram / normal_equation_scale
        rhs = raw_rhs / normal_equation_scale
        y_second_moment = float(
            np.dot(
                np.asarray(weighted_y, dtype=np.float64),
                np.asarray(weighted_y, dtype=np.float64),
            )
            / normal_equation_scale
        )
        if y_second_moment.hex() != checkpoint_y_second_moment.hex():
            raise RuntimeError(
                "Elastic Net checkpoint response scale differs from its design"
            )

        def publish_and_reauthenticate_checkpoint(
            *,
            identity: Mapping[str, Any],
            group: tuple[ElasticNetConfig, ...],
            alphas: np.ndarray,
            predictions: np.ndarray,
            receipts: tuple[ElasticNetFitReceipt, ...],
        ) -> None:
            assert work_path is not None
            validated_receipts = _validated_checkpoint_receipts(
                [receipt.as_dict() for receipt in receipts],
                fold_id=fold_id,
                configurations=group,
                alphas=alphas,
                lambda_ref=reference,
                train_rows=features.shape[0],
                input_features=features.shape[1],
                outer_validation_rows=validation.shape[0],
                tolerance=tolerance,
                y_second_moment=y_second_moment,
            )
            if validated_receipts != receipts:
                raise RuntimeError(
                    "Elastic Net checkpoint receipt validation changed evidence"
                )
            _publish_rho_checkpoint(
                work_root=work_path.parent,
                identity=identity,
                fold_id=fold_id,
                predictions=predictions,
                receipts=validated_receipts,
            )
            reloaded = _load_rho_checkpoint(
                work_root=work_path.parent,
                identity=identity,
                fold_id=fold_id,
                configurations=group,
                alphas=alphas,
                lambda_ref=reference,
                train_rows=features.shape[0],
                input_features=features.shape[1],
                outer_validation_rows=validation.shape[0],
            )
            if (
                reloaded is None
                or not np.array_equal(
                    np.asarray(reloaded[0], dtype=np.float32),
                    np.asarray(predictions, dtype=np.float32),
                )
                or reloaded[1] != validated_receipts
            ):
                raise RuntimeError(
                    "Published Elastic Net rho checkpoint failed semantic replay"
                )
        # The exact positive-rho solver below uses only the normalized Hessian
        # and linear term.  Release the 2–3 GB production memmap before any
        # path solve so a bounded-runtime termination cannot strand it.
        if isinstance(weighted_X, np.memmap):
            weighted_X.flush()
        weighted_X = None
        if (
            scratch_requested
            and work_path is not None
            and (work_path.exists() or work_path.is_symlink())
        ):
            os.unlink(work_path)
        # This eigensystem is also the exact full-support active-set backend.
        # Compute it once even when rho=0 resumes from a checkpoint so cold and
        # resumed positive-rho paths take the identical deterministic route.
        gram_eigenvalues, gram_eigenvectors = np.linalg.eigh(gram)
        for rho_index, rho in enumerate(RHO_GRID):
            if rho_index in resumed_rho_indices:
                continue
            group = tuple(config for config in configs if config.rho == rho)
            alphas = np.asarray(
                [lambda_value(config, reference) for config in group],
                dtype=np.float64,
            )
            checkpoint_identity = (
                None
                if checkpoint_parent_job_id is None
                else _checkpoint_identity(
                    parent_job_id=checkpoint_parent_job_id,
                    fold_id=fold_id,
                    rho_index=rho_index,
                    rho=rho,
                    configurations=group,
                    alphas=alphas,
                    lambda_ref=reference,
                    train_rows=features.shape[0],
                    input_features=features.shape[1],
                    outer_validation_rows=validation.shape[0],
                    max_iter=max_iter,
                    tolerance=tolerance,
                    y_second_moment=checkpoint_y_second_moment,
                )
            )
            if checkpoint_identity is not None:
                _emit_path_progress(
                    fold_id=fold_id,
                    rho_index=rho_index,
                    rho=rho,
                    status="started",
            )
            if rho == 0.0:
                ridge_eigenvalues = np.maximum(gram_eigenvalues, 0.0)
                rotated = gram_eigenvectors.T @ rhs
                coefficient_path = gram_eigenvectors @ (
                    rotated[:, None]
                    / (ridge_eigenvalues[:, None] + alphas[None, :])
                )
                ridge_stationarity = (
                    rhs[:, None]
                    - gram @ coefficient_path
                    - alphas[None, :] * coefficient_path
                )
                ridge_kkt = np.max(np.abs(ridge_stationarity), axis=0)
                ridge_limits = tolerance * np.maximum(
                    1.0,
                    np.maximum(float(np.max(np.abs(rhs))), alphas),
                )
                _require_kkt(ridge_kkt, ridge_limits, label="ridge path")
                gram_beta = gram @ coefficient_path
                beta_rhs = np.sum(coefficient_path * rhs[:, None], axis=0)
                beta_gram_beta = np.sum(coefficient_path * gram_beta, axis=0)
                beta_norm = np.sum(np.square(coefficient_path), axis=0)
                residual_mse = y_second_moment - 2.0 * beta_rhs + beta_gram_beta
                primal = 0.5 * residual_mse + 0.5 * alphas * beta_norm
                dual = (
                    y_second_moment
                    - beta_rhs
                    - 0.5 * residual_mse
                    - np.sum(np.square(gram_beta - rhs[:, None]), axis=0)
                    / (2.0 * alphas)
                )
                ridge_dual_gaps = np.maximum(primal - dual, 0.0)
                ridge_gap_limit = tolerance * max(1.0, y_second_moment)
                if (
                    not np.isfinite(ridge_dual_gaps).all()
                    or bool(np.any(ridge_dual_gaps > ridge_gap_limit))
                ):
                    raise RuntimeError("Elastic Net ridge path failed its dual-gap gate")
                for path_index, (config, alpha) in enumerate(
                    zip(group, alphas, strict=True)
                ):
                    coefficients = coefficient_path[:, path_index]
                    intercept = y_mean - float(np.dot(x_mean, coefficients))
                    prediction = validation @ coefficients + intercept
                    prediction_by_id[config.config_id] = np.asarray(prediction, dtype=np.float32)
                    receipt_by_id[config.config_id] = ElasticNetFitReceipt(
                        config_id=config.config_id,
                        fold_id=fold_id,
                        input_features=features.shape[1],
                        train_rows=features.shape[0],
                        outer_validation_rows=validation.shape[0],
                        lambda_ref=reference,
                        lambda_value=float(alpha),
                        rho=rho,
                        lambda_index=config.lambda_index,
                        nonzero_coefficients=int(np.count_nonzero(np.abs(coefficients) > NONZERO_THRESHOLD)),
                        initial_dual_gap=float(ridge_dual_gaps[path_index]),
                        post_secondary_dual_gap=float(ridge_dual_gaps[path_index]),
                        dual_gap=float(ridge_dual_gaps[path_index]),
                        dual_gap_limit=float(ridge_gap_limit),
                        initial_kkt_max_violation=float(ridge_kkt[path_index]),
                        initial_kkt_limit=float(ridge_limits[path_index]),
                        post_secondary_kkt_max_violation=float(ridge_kkt[path_index]),
                        post_secondary_kkt_limit=float(ridge_limits[path_index]),
                        kkt_max_violation=float(ridge_kkt[path_index]),
                        kkt_limit=float(ridge_limits[path_index]),
                        iterations=1,
                        initial_iterations=1,
                        secondary_iterations=0,
                        tertiary_iterations=0,
                        refinement_iterations=0,
                        solver_passes=1,
                        initial_solver_stop_criterion=float(ridge_kkt[path_index]),
                        initial_solver_stop_limit=float(ridge_limits[path_index]),
                        secondary_solver_stop_criterion=float(ridge_kkt[path_index]),
                        secondary_solver_stop_limit=float(ridge_limits[path_index]),
                        solver_stop_criterion=float(ridge_kkt[path_index]),
                        solver_stop_limit=float(ridge_limits[path_index]),
                        refinement_route="ridge_closed_form",
                        initial_coefficient_sha256=_coefficient_sha256(coefficients),
                        final_coefficient_sha256=_coefficient_sha256(coefficients),
                        orthant_iterations=0,
                        orthant_cholesky_attempts=0,
                        orthant_gelsy_attempts=0,
                        orthant_rank_deficient_solves=0,
                        orthant_max_active_dimension=0,
                        orthant_min_solve_rank=0,
                        orthant_support_transitions=0,
                        orthant_boundary_transitions=0,
                        orthant_dropped_coordinates=0,
                        orthant_added_coordinates=0,
                        orthant_trace=[],
                        converged=True,
                        elapsed_seconds=float(time.monotonic() - elapsed_offset),
                        solver="weighted_centered_ridge_eigendecomposition",
                        dtype="float64",
                    )
                if checkpoint_identity is not None:
                    assert work_path is not None
                    path_predictions = np.column_stack(
                        [prediction_by_id[config.config_id] for config in group]
                    ).astype(np.float32, copy=False)
                    path_receipts = tuple(
                        receipt_by_id[config.config_id] for config in group
                    )
                    publish_and_reauthenticate_checkpoint(
                        identity=checkpoint_identity,
                        group=group,
                        alphas=alphas,
                        predictions=path_predictions,
                        receipts=path_receipts,
                    )
                    newly_published_checkpoints += 1
                    _emit_path_progress(
                        fold_id=fold_id,
                        rho_index=rho_index,
                        rho=rho,
                        status="published an authenticated checkpoint",
                    )
                    if (
                        _test_interrupt_after_rho_checkpoints
                        == newly_published_checkpoints
                    ):
                        raise RuntimeError(
                            "NONAUTHORITATIVE_TEST interruption after authenticated "
                            "Elastic Net rho checkpoint"
                        )
                continue
            path_result = _working_set_coordinate_descent_path(
                rho=rho,
                alphas=alphas,
                tolerance=tolerance,
                gram=gram,
                rhs=rhs,
                y_second_moment=y_second_moment,
                gram_eigenvalues=gram_eigenvalues,
                gram_eigenvectors=gram_eigenvectors,
            )
            returned_alphas = path_result.alphas
            coefficients = path_result.coefficients
            dual_gaps = path_result.dual_gaps
            if not np.allclose(returned_alphas, alphas, rtol=0.0, atol=0.0):
                raise RuntimeError("Elastic Net solver changed the frozen lambda path")
            kkt_maximum, kkt_limits = _path_kkt_violations(
                gram,
                rhs,
                coefficients,
                alphas,
                rho=rho,
                tolerance=tolerance,
            )
            _require_kkt(kkt_maximum, kkt_limits, label=f"rho={rho:g} path")
            intercepts = y_mean - x_mean @ coefficients
            path_predictions = validation @ coefficients + intercepts
            dual_gap_limit = float(tolerance) * y_second_moment
            for path_index, config in enumerate(group):
                gap = _nonnegative_dual_gap(float(dual_gaps[path_index]))
                iteration_count = int(path_result.total_iterations[path_index])
                coefficient = coefficients[:, path_index]
                orthant_trace = [
                    dict(value)
                    for value in path_result.orthant_traces[path_index]
                ]
                orthant_summary = _orthant_trace_summary(
                    orthant_trace,
                    input_features=features.shape[1],
                )
                prediction_by_id[config.config_id] = np.asarray(
                    path_predictions[:, path_index], dtype=np.float32
                )
                receipt_by_id[config.config_id] = ElasticNetFitReceipt(
                    config_id=config.config_id,
                    fold_id=fold_id,
                    input_features=features.shape[1],
                    train_rows=features.shape[0],
                    outer_validation_rows=validation.shape[0],
                    lambda_ref=reference,
                    lambda_value=float(alphas[path_index]),
                    rho=rho,
                    lambda_index=config.lambda_index,
                    nonzero_coefficients=int(np.count_nonzero(np.abs(coefficient) > NONZERO_THRESHOLD)),
                    initial_dual_gap=float(
                        path_result.initial_dual_gaps[path_index]
                    ),
                    post_secondary_dual_gap=float(
                        path_result.post_secondary_dual_gaps[path_index]
                    ),
                    dual_gap=gap,
                    dual_gap_limit=dual_gap_limit,
                    initial_kkt_max_violation=float(
                        path_result.initial_kkt_maximum[path_index]
                    ),
                    initial_kkt_limit=float(
                        path_result.initial_kkt_limits[path_index]
                    ),
                    post_secondary_kkt_max_violation=float(
                        path_result.post_secondary_kkt_maximum[path_index]
                    ),
                    post_secondary_kkt_limit=float(
                        path_result.post_secondary_kkt_limits[path_index]
                    ),
                    kkt_max_violation=float(kkt_maximum[path_index]),
                    kkt_limit=float(kkt_limits[path_index]),
                    iterations=iteration_count,
                    initial_iterations=int(
                        path_result.initial_iterations[path_index]
                    ),
                    secondary_iterations=int(
                        path_result.secondary_iterations[path_index]
                    ),
                    tertiary_iterations=int(
                        path_result.tertiary_iterations[path_index]
                    ),
                    refinement_iterations=int(
                        path_result.refinement_iterations[path_index]
                    ),
                    solver_passes=int(path_result.solver_passes[path_index]),
                    initial_solver_stop_criterion=float(
                        path_result.initial_stop_criteria[path_index]
                    ),
                    initial_solver_stop_limit=float(
                        path_result.initial_stop_limits[path_index]
                    ),
                    secondary_solver_stop_criterion=float(
                        path_result.secondary_stop_criteria[path_index]
                    ),
                    secondary_solver_stop_limit=float(
                        path_result.secondary_stop_limits[path_index]
                    ),
                    solver_stop_criterion=float(
                        path_result.final_stop_criteria[path_index]
                    ),
                    solver_stop_limit=float(
                        path_result.final_stop_limits[path_index]
                    ),
                    refinement_route=path_result.refinement_routes[path_index],
                    initial_coefficient_sha256=(
                        path_result.initial_coefficient_sha256s[path_index]
                    ),
                    final_coefficient_sha256=(
                        path_result.final_coefficient_sha256s[path_index]
                    ),
                    **orthant_summary,
                    orthant_trace=orthant_trace,
                    converged=True,
                    elapsed_seconds=float(time.monotonic() - elapsed_offset),
                    solver=_WORKING_SET_SOLVER,
                    dtype="float64",
                )
            if checkpoint_identity is not None:
                assert work_path is not None
                checkpoint_predictions = np.column_stack(
                    [prediction_by_id[config.config_id] for config in group]
                ).astype(np.float32, copy=False)
                checkpoint_receipts = tuple(
                    receipt_by_id[config.config_id] for config in group
                )
                publish_and_reauthenticate_checkpoint(
                    identity=checkpoint_identity,
                    group=group,
                    alphas=alphas,
                    predictions=checkpoint_predictions,
                    receipts=checkpoint_receipts,
                )
                newly_published_checkpoints += 1
                _emit_path_progress(
                    fold_id=fold_id,
                    rho_index=rho_index,
                    rho=rho,
                    status="published an authenticated checkpoint",
                )
                if (
                    _test_interrupt_after_rho_checkpoints
                    == newly_published_checkpoints
                ):
                    raise RuntimeError(
                        "NONAUTHORITATIVE_TEST interruption after authenticated "
                        "Elastic Net rho checkpoint"
                    )
        return ordered_result()
    finally:
        if isinstance(weighted_X, np.memmap):
            weighted_X.flush()
        weighted_X = None
        if scratch_requested and work_path is not None and (work_path.exists() or work_path.is_symlink()):
            os.unlink(work_path)


def fit_predict_outer_fold(
    *,
    fold_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_outer_validation: np.ndarray,
    work_path: Path,
) -> tuple[np.ndarray, tuple[ElasticNetFitReceipt, ...]]:
    """Fit the canonical 250-point path in an authoritative spawn worker."""

    require_spawned_native_worker("elastic_net")
    validate_runtime_versions()
    if fold_id not in {"dev_fold_1", "dev_fold_2", "dev_fold_3", "dev_fold_4"}:
        raise ValueError("Authoritative Stage 9 Elastic Net fold is outside the frozen folds")
    if work_path is None:
        raise ValueError("Authoritative Elastic Net fits require a disk-backed work_path")
    parent_job_id, float64_work_path = _authoritative_float64_workspace(work_path)
    with threadpool_limits(limits=1):
        result = _fit_predict_outer_fold_impl(
            fold_id=fold_id,
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_outer_validation=X_outer_validation,
            work_path=float64_work_path,
            max_iter=MAX_ITERATIONS,
            tolerance=TOLERANCE,
            checkpoint_parent_job_id=parent_job_id,
        )
    if work_path.exists() or work_path.is_symlink():
        raise RuntimeError("Legacy Elastic Net float32 work_path was unexpectedly created")
    if float64_work_path.exists() or float64_work_path.is_symlink():
        raise RuntimeError("Elastic Net float64 workspace was not removed")
    return result


def _fit_predict_outer_fold_nonauthoritative_for_test(
    *,
    fold_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_outer_validation: np.ndarray,
    work_path: Path | None,
    max_iter: int = 2_000,
    tolerance: float = TOLERANCE,
    checkpoint_parent_job_id: str | None = None,
    _test_interrupt_after_rho_checkpoints: int | None = None,
) -> tuple[np.ndarray, tuple[ElasticNetFitReceipt, ...]]:
    """Reduced native fit for tests; its receipts are never publishable."""

    require_spawned_native_worker("elastic_net")
    with threadpool_limits(limits=1):
        return _fit_predict_outer_fold_impl(
            fold_id=f"NONAUTHORITATIVE_TEST::{fold_id}",
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_outer_validation=X_outer_validation,
            work_path=work_path,
            max_iter=max_iter,
            tolerance=tolerance,
            checkpoint_parent_job_id=checkpoint_parent_job_id,
            _test_interrupt_after_rho_checkpoints=(
                _test_interrupt_after_rho_checkpoints
            ),
        )
