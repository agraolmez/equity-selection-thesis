"""Family-isolated native fitting and scoring for Stage 11 selectors.

The public functions in this module must execute in a multiprocessing
``spawn`` child claimed by exactly one native family.  They reuse the frozen
Stage 9 numerical implementations, while removing only Stage 9's deliberate
development-fold and three-seed admission checks.

There are no filesystem or validation-outcome readers here.  The caller
supplies arrays that have already passed :mod:`selector_runtime` chronology
checks and persists the returned :class:`NativeModelArtifact` through the
authenticated checkpoint publisher.
"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from portfolio_bot.selectors.model_specs import (
    canonical_xgboost_configuration,
    elastic_net_configurations,
    neural_configurations,
)
from portfolio_bot.selectors.native_process import require_spawned_native_worker
from portfolio_bot.validation.selector_runtime import (
    FileBackedArrayRequest,
    MODEL_FORMAT_BY_FAMILY,
    LoadedSelectorCheckpoint,
    NativeModelArtifact,
    SelectorRuntimeError,
    ValidationRefitArrays,
    ValidationRefitIdentity,
    ValidationRefitPathRequest,
    array_sha256,
    load_array_for_spawn,
    load_validation_refit_arrays_from_paths,
    pack_numeric_arrays,
    unpack_numeric_arrays,
    validate_locked_selector,
)
from portfolio_bot.validation.locked_selection import LOCKED_SHORTLIST


@dataclass(frozen=True, slots=True)
class _NonauthoritativeTestControls:
    """Small synthetic-fit controls that can never produce a publishable receipt."""

    neural_max_epochs: int = 4
    neural_patience: int = 2
    neural_batch_size: int = 16
    xgboost_max_rounds: int = 30
    xgboost_early_stopping_rounds: int = 5
    elastic_tolerance: float = 1e-6


def _module(family: str) -> Any:
    require_spawned_native_worker(family)
    path = {
        "neural_network": "portfolio_bot.selectors.neural_model",
        "xgboost": "portfolio_bot.selectors.xgboost_model",
        "elastic_net": "portfolio_bot.selectors.elastic_net",
    }.get(family)
    if path is None:
        raise SelectorRuntimeError("unknown Stage 11 selector family")
    return importlib.import_module(path)


def _validate_arrays(identity: ValidationRefitIdentity, arrays: ValidationRefitArrays) -> None:
    observed = arrays.identity_payload()
    if observed != dict(identity.training_data):
        raise SelectorRuntimeError("native fit arrays differ from their refit identity")
    if identity.family in {"neural_network", "xgboost"}:
        Xi = np.asarray(arrays.X_inner_train)
        Xv = np.asarray(arrays.X_inner_validation)
        Xf = np.asarray(arrays.X_full_train)
        if Xi.ndim != 2 or Xv.ndim != 2 or Xf.ndim != 2:
            raise SelectorRuntimeError("stochastic selector matrices must be two-dimensional")
        if Xi.shape[1] != Xv.shape[1]:
            raise SelectorRuntimeError("inner selector matrix widths differ")


def _nn_fit(
    identity: ValidationRefitIdentity,
    arrays: ValidationRefitArrays,
    *,
    controls: _NonauthoritativeTestControls | None,
) -> NativeModelArtifact:
    module = _module("neural_network")
    if controls is None:
        module.validate_runtime_versions()
    config = {value.config_id: value for value in neural_configurations()}[
        identity.config_id
    ]
    seed = int(identity.seed)
    max_epochs = module.MAX_EPOCHS if controls is None else controls.neural_max_epochs
    patience = module.PATIENCE if controls is None else controls.neural_patience
    batch_size = module.BATCH_SIZE if controls is None else controls.neural_batch_size
    best_epoch, best_loss, examined, stopped, inner_parameters = module.choose_epoch_count(
        arrays.X_inner_train,
        arrays.y_inner_train,
        arrays.w_inner_train,
        arrays.X_inner_validation,
        arrays.y_inner_validation,
        arrays.w_inner_validation,
        config=config,
        seed=seed,
        threads=1,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=module.MIN_DELTA,
        batch_size=batch_size,
    )
    module._seed_runtime(seed, threads=1)
    model = module.ShallowSelectorNetwork(arrays.X_full_train.shape[1], config)
    module._fit_epochs(
        model,
        arrays.X_full_train,
        arrays.y_full_train,
        arrays.w_full_train,
        config=config,
        seed=seed,
        epochs=best_epoch,
        batch_size=batch_size,
    )
    state_arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in sorted(model.state_dict().items())
    }
    model_payload = pack_numeric_arrays(state_arrays)
    status = (
        "NONAUTHORITATIVE_SYNTHETIC_TEST_ONLY"
        if controls is not None
        else "AUTHORITATIVE_STAGE11_MODEL_FIT_COMPLETE"
    )
    return NativeModelArtifact(
        family="neural_network",
        model_format=MODEL_FORMAT_BY_FAMILY["neural_network"],
        model_payload=model_payload,
        fit_receipt={
            "schema_version": 1,
            "status": status,
            "family": "neural_network",
            "config_id": identity.config_id,
            "configuration": asdict(config),
            "seed": seed,
            "refit_signal_date": identity.refit_signal_date.isoformat(),
            "train_rows": arrays.full_rows,
            "inner_train_rows": arrays.inner_train_rows,
            "inner_validation_rows": arrays.inner_validation_rows,
            "input_features": int(arrays.X_full_train.shape[1]),
            "inner_input_features": int(arrays.X_inner_train.shape[1]),
            "best_epoch": int(best_epoch),
            "best_inner_loss": float(best_loss),
            "epochs_examined": int(examined),
            "stopped_early": bool(stopped),
            "parameter_count": int(module.trainable_parameter_count(model)),
            "inner_parameter_count": int(inner_parameters),
            "model_payload_sha256": array_sha256(
                np.frombuffer(model_payload, dtype=np.uint8)
            ),
            "device": "cpu",
            "threads": 1,
            "deterministic_algorithms": True,
            "validation_selection_metrics_read": False,
            "portfolio_or_final_data_read": False,
        },
    )


def _xgboost_fit(
    identity: ValidationRefitIdentity,
    arrays: ValidationRefitArrays,
    *,
    controls: _NonauthoritativeTestControls | None,
) -> NativeModelArtifact:
    module = _module("xgboost")
    if controls is None:
        module.validate_runtime_versions()
    config = canonical_xgboost_configuration(identity.config_id)
    seed = int(identity.seed)
    Xi, yi, wi = module._validated(
        arrays.X_inner_train, arrays.y_inner_train, arrays.w_inner_train
    )
    Xv, yv, wv = module._validated(
        arrays.X_inner_validation,
        arrays.y_inner_validation,
        arrays.w_inner_validation,
    )
    Xf, yf, wf = module._validated(
        arrays.X_full_train, arrays.y_full_train, arrays.w_full_train
    )
    assert yi is not None and wi is not None and yv is not None and wv is not None
    assert yf is not None and wf is not None
    inner_child = module.child_share_to_min_child_weight(config.child_share, wi)
    full_child = module.child_share_to_min_child_weight(config.child_share, wf)
    max_rounds = (
        module.MAX_BOOSTING_ROUNDS if controls is None else controls.xgboost_max_rounds
    )
    stopping = (
        module.EARLY_STOPPING_ROUNDS
        if controls is None
        else controls.xgboost_early_stopping_rounds
    )
    inner_train = module.xgb.DMatrix(Xi, label=yi, weight=wi, nthread=1)
    inner_validation = module.xgb.DMatrix(Xv, label=yv, weight=wv, nthread=1)
    early_model = module.xgb.train(
        module._params(config, seed=seed, min_child_weight=inner_child, threads=1),
        inner_train,
        num_boost_round=max_rounds,
        evals=[(inner_validation, "inner_validation")],
        early_stopping_rounds=stopping,
        verbose_eval=False,
    )
    best_tree_count = int(early_model.best_iteration) + 1
    best_score = float(early_model.best_score)
    if best_tree_count <= 0 or not math.isfinite(best_score):
        raise SelectorRuntimeError("XGBoost early stopping produced invalid evidence")
    del early_model, inner_train, inner_validation
    full_train = module.xgb.DMatrix(Xf, label=yf, weight=wf, nthread=1)
    final_model = module.xgb.train(
        module._params(config, seed=seed, min_child_weight=full_child, threads=1),
        full_train,
        num_boost_round=best_tree_count,
        verbose_eval=False,
    )
    model_payload = bytes(final_model.save_raw(raw_format="ubj"))
    status = (
        "NONAUTHORITATIVE_SYNTHETIC_TEST_ONLY"
        if controls is not None
        else "AUTHORITATIVE_STAGE11_MODEL_FIT_COMPLETE"
    )
    return NativeModelArtifact(
        family="xgboost",
        model_format=MODEL_FORMAT_BY_FAMILY["xgboost"],
        model_payload=model_payload,
        fit_receipt={
            "schema_version": 1,
            "status": status,
            "family": "xgboost",
            "config_id": identity.config_id,
            "configuration": asdict(config),
            "seed": seed,
            "refit_signal_date": identity.refit_signal_date.isoformat(),
            "train_rows": arrays.full_rows,
            "inner_train_rows": arrays.inner_train_rows,
            "inner_validation_rows": arrays.inner_validation_rows,
            "input_features": int(Xf.shape[1]),
            "inner_input_features": int(Xi.shape[1]),
            "min_child_weight_inner": float(inner_child),
            "min_child_weight_full": float(full_child),
            "best_tree_count": best_tree_count,
            "best_inner_rmse": best_score,
            "xgboost_version": module.xgb.__version__,
            "model_payload_sha256": array_sha256(
                np.frombuffer(model_payload, dtype=np.uint8)
            ),
            "device": "cpu",
            "threads": 1,
            "validation_selection_metrics_read": False,
            "portfolio_or_final_data_read": False,
        },
    )


def _elastic_artifact_from_solution(
    *,
    identity: ValidationRefitIdentity,
    arrays: Any,
    config: Any,
    features: np.ndarray,
    reference: float,
    lambda_value: float,
    coefficients: np.ndarray,
    intercept: float,
    dual_gap: float,
    dual_gap_limit: float,
    kkt: float,
    kkt_limit: float,
    iterations: int,
    route: str,
    controls: _NonauthoritativeTestControls | None,
    grouped_certificate: Mapping[str, Any] | None = None,
) -> NativeModelArtifact:
    module = importlib.import_module("portfolio_bot.selectors.elastic_net")
    model_payload = pack_numeric_arrays(
        {
            "coefficients": np.asarray(coefficients, dtype=np.float64),
            "intercept": np.asarray([intercept], dtype=np.float64),
        }
    )
    status = (
        "NONAUTHORITATIVE_SYNTHETIC_TEST_ONLY"
        if controls is not None
        else "AUTHORITATIVE_STAGE11_MODEL_FIT_COMPLETE"
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "family": "elastic_net",
        "config_id": identity.config_id,
        "configuration": asdict(config),
        "seed": None,
        "refit_signal_date": identity.refit_signal_date.isoformat(),
        "train_rows": arrays.full_rows,
        "input_features": int(features.shape[1]),
        "lambda_ref": float(reference),
        "lambda_value": float(lambda_value),
        "rho": float(config.rho),
        "lambda_index": int(config.lambda_index),
        "nonzero_coefficients": int(
            np.count_nonzero(np.abs(coefficients) > module.NONZERO_THRESHOLD)
        ),
        "dual_gap": dual_gap,
        "dual_gap_limit": dual_gap_limit,
        "kkt_max_violation": kkt,
        "kkt_limit": kkt_limit,
        "iterations": iterations,
        "refinement_route": route,
        "solver": (
            "weighted_centered_ridge_eigendecomposition"
            if config.rho == 0.0 and reference != 0.0
            else module._WORKING_SET_SOLVER
            if reference != 0.0
            else "zero_signal"
        ),
        "model_payload_sha256": array_sha256(
            np.frombuffer(model_payload, dtype=np.uint8)
        ),
        "device": "cpu",
        "threads": 1,
        "validation_selection_metrics_read": False,
        "portfolio_or_final_data_read": False,
    }
    if grouped_certificate is not None:
        receipt["grouped_refit"] = True
        receipt["grouped_shared_numerics"] = dict(grouped_certificate)
    return NativeModelArtifact(
        family="elastic_net",
        model_format=MODEL_FORMAT_BY_FAMILY["elastic_net"],
        model_payload=model_payload,
        fit_receipt=receipt,
    )


def _elastic_net_fit(
    identity: ValidationRefitIdentity,
    arrays: ValidationRefitArrays,
    *,
    scratch_path: Path | None,
    controls: _NonauthoritativeTestControls | None,
) -> NativeModelArtifact:
    module = _module("elastic_net")
    if controls is None:
        module.validate_runtime_versions()
    config = {value.config_id: value for value in elastic_net_configurations()}[
        identity.config_id
    ]
    features, target, weights = module._validated(
        arrays.X_full_train, arrays.y_full_train, arrays.w_full_train
    )
    reference, y_mean, x_mean = module.weighted_lambda_reference(
        features, target, weights
    )
    tolerance = module.TOLERANCE if controls is None else controls.elastic_tolerance
    route = "zero_signal"
    iterations = 0
    dual_gap = 0.0
    dual_gap_limit = 0.0
    kkt = 0.0
    kkt_limit = tolerance
    lambda_value = 0.0
    if reference == 0.0:
        coefficients = np.zeros(features.shape[1], dtype=np.float64)
        intercept = 0.0
    else:
        destination: Path | None = None
        if scratch_path is not None:
            destination = Path(os.path.abspath(Path(scratch_path).expanduser()))
            if not destination.is_absolute() or destination.exists() or destination.is_symlink():
                raise SelectorRuntimeError("Elastic Net scratch path is unsafe or already exists")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.parent.is_symlink():
                raise SelectorRuntimeError("Elastic Net scratch parent is unsafe")
        weighted_X: np.ndarray | None = None
        try:
            weighted_X, weighted_y = module.prepare_weighted_centered_design(
                features,
                target,
                weights,
                x_mean=x_mean,
                y_mean=y_mean,
                destination=destination,
            )
            raw_gram, raw_rhs = module._float64_raw_gram_and_rhs(weighted_X, weighted_y)
            scale = float(features.shape[0])
            gram = raw_gram / scale
            rhs = raw_rhs / scale
            y_second_moment = float(np.dot(weighted_y, weighted_y) / scale)
            prefix = tuple(
                value
                for value in elastic_net_configurations()
                if value.rho == config.rho and value.lambda_index <= config.lambda_index
            )
            alphas = np.asarray(
                [module.lambda_value(value, reference) for value in prefix],
                dtype=np.float64,
            )
            lambda_value = float(alphas[-1])
            if config.rho == 0.0:
                eigenvalues, eigenvectors = np.linalg.eigh(gram)
                eigenvalues = np.maximum(eigenvalues, 0.0)
                rotated = eigenvectors.T @ rhs
                path = eigenvectors @ (
                    rotated[:, None] / (eigenvalues[:, None] + alphas[None, :])
                )
                coefficients = np.asarray(path[:, -1], dtype=np.float64)
                stationarity = rhs - gram @ coefficients - lambda_value * coefficients
                kkt = float(np.max(np.abs(stationarity)))
                kkt_limit = float(
                    tolerance
                    * max(1.0, float(np.max(np.abs(rhs))), lambda_value)
                )
                if not math.isfinite(kkt) or kkt > kkt_limit:
                    raise SelectorRuntimeError("Elastic Net ridge fit failed its KKT gate")
                route = "ridge_closed_form"
                iterations = 1
                residual_mse = float(
                    y_second_moment
                    - 2.0 * np.dot(coefficients, rhs)
                    + coefficients @ gram @ coefficients
                )
                primal = 0.5 * residual_mse + 0.5 * lambda_value * float(
                    np.dot(coefficients, coefficients)
                )
                correlation = gram @ coefficients - rhs
                dual = (
                    y_second_moment
                    - float(np.dot(coefficients, rhs))
                    - 0.5 * residual_mse
                    - float(np.dot(correlation, correlation)) / (2.0 * lambda_value)
                )
                dual_gap = max(float(primal - dual), 0.0)
                dual_gap_limit = float(tolerance * max(1.0, y_second_moment))
                if dual_gap > dual_gap_limit:
                    raise SelectorRuntimeError("Elastic Net ridge fit failed its dual-gap gate")
            else:
                eigenvalues, eigenvectors = np.linalg.eigh(gram)
                path_result = module._working_set_coordinate_descent_path(
                    rho=config.rho,
                    alphas=alphas,
                    tolerance=tolerance,
                    gram=gram,
                    rhs=rhs,
                    y_second_moment=y_second_moment,
                    gram_eigenvalues=eigenvalues,
                    gram_eigenvectors=eigenvectors,
                )
                coefficients = np.asarray(path_result.coefficients[:, -1], dtype=np.float64)
                kkt_values, kkt_limits = module._path_kkt_violations(
                    gram,
                    rhs,
                    path_result.coefficients[:, -1:],
                    alphas[-1:],
                    rho=config.rho,
                    tolerance=tolerance,
                )
                module._require_kkt(kkt_values, kkt_limits, label="Stage 11 selected path")
                kkt = float(kkt_values[-1])
                kkt_limit = float(kkt_limits[-1])
                dual_gap = float(path_result.dual_gaps[-1])
                dual_gap_limit = float(tolerance * y_second_moment)
                iterations = int(path_result.total_iterations[-1])
                route = str(path_result.refinement_routes[-1])
            intercept = float(y_mean - np.dot(x_mean, coefficients))
        finally:
            if isinstance(weighted_X, np.memmap):
                weighted_X.flush()
            weighted_X = None
            if destination is not None and (destination.exists() or destination.is_symlink()):
                destination.unlink()
    return _elastic_artifact_from_solution(
        identity=identity,
        arrays=arrays,
        config=config,
        features=features,
        reference=reference,
        lambda_value=lambda_value,
        coefficients=coefficients,
        intercept=intercept,
        dual_gap=dual_gap,
        dual_gap_limit=dual_gap_limit,
        kkt=kkt,
        kkt_limit=kkt_limit,
        iterations=iterations,
        route=route,
        controls=controls,
    )


def _fit_locked_elastic_net_group_impl(
    *,
    identities: Sequence[ValidationRefitIdentity],
    arrays: Any,
    scratch_path: Path | None,
    controls: _NonauthoritativeTestControls | None,
) -> tuple[NativeModelArtifact, ...]:
    """Fit the three locked ENets from one design/Gram/eigendecomposition."""

    expected_ids = tuple(LOCKED_SHORTLIST["elastic_net"])
    by_id = {identity.config_id: identity for identity in identities}
    if len(identities) != len(by_id) or tuple(sorted(by_id)) != tuple(sorted(expected_ids)):
        raise SelectorRuntimeError("grouped Elastic Net refit must contain exactly three locked configs")
    ordered = tuple(by_id[config_id] for config_id in expected_ids)
    first = ordered[0]
    for identity in ordered:
        validate_locked_selector(identity.family, identity.config_id, identity.seed)
        if (
            identity.family != "elastic_net"
            or identity.seed is not None
            or identity.refit_signal_date != first.refit_signal_date
            or dict(identity.training_data) != dict(first.training_data)
            or identity.full_preprocessor_sha256 != first.full_preprocessor_sha256
            or identity.inner_preprocessor_sha256 is not None
            or identity.validation_calendar_sha256 != first.validation_calendar_sha256
            or identity.model_code_sha256 != first.model_code_sha256
            or identity.runtime_versions_sha256 != first.runtime_versions_sha256
            or identity.stage9_shortlist_sha256 != first.stage9_shortlist_sha256
        ):
            raise SelectorRuntimeError("grouped Elastic Net refit identities do not share one refit")
    _validate_arrays(first, arrays)
    module = _module("elastic_net")
    if controls is None:
        module.validate_runtime_versions()
    configurations = {value.config_id: value for value in elastic_net_configurations()}
    features, target, weights = module._validated(
        arrays.X_full_train, arrays.y_full_train, arrays.w_full_train
    )
    reference, y_mean, x_mean = module.weighted_lambda_reference(
        features, target, weights
    )
    tolerance = module.TOLERANCE if controls is None else controls.elastic_tolerance
    destination: Path | None = None
    if scratch_path is not None:
        destination = Path(os.path.abspath(Path(scratch_path).expanduser()))
        if not destination.is_absolute() or destination.exists() or destination.is_symlink():
            raise SelectorRuntimeError("Elastic Net scratch path is unsafe or already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.is_symlink():
            raise SelectorRuntimeError("Elastic Net scratch parent is unsafe")

    weighted_X: np.ndarray | None = None
    solutions: list[NativeModelArtifact] = []
    try:
        if reference == 0.0:
            certificate: Mapping[str, Any] = {
                "config_ids": list(expected_ids),
                "weighted_design_computations": 0,
                "normal_equation_computations": 0,
                "eigendecompositions": 0,
                "gram_sha256": None,
                "rhs_sha256": None,
                "eigenvalues_sha256": None,
                "eigenvectors_sha256": None,
            }
            for identity in ordered:
                config = configurations[identity.config_id]
                solutions.append(
                    _elastic_artifact_from_solution(
                        identity=identity,
                        arrays=arrays,
                        config=config,
                        features=features,
                        reference=reference,
                        lambda_value=0.0,
                        coefficients=np.zeros(features.shape[1], dtype=np.float64),
                        intercept=0.0,
                        dual_gap=0.0,
                        dual_gap_limit=0.0,
                        kkt=0.0,
                        kkt_limit=tolerance,
                        iterations=0,
                        route="zero_signal",
                        controls=controls,
                        grouped_certificate=certificate,
                    )
                )
            return tuple(solutions)

        weighted_X, weighted_y = module.prepare_weighted_centered_design(
            features,
            target,
            weights,
            x_mean=x_mean,
            y_mean=y_mean,
            destination=destination,
        )
        raw_gram, raw_rhs = module._float64_raw_gram_and_rhs(weighted_X, weighted_y)
        scale = float(features.shape[0])
        gram = raw_gram / scale
        rhs = raw_rhs / scale
        y_second_moment = float(np.dot(weighted_y, weighted_y) / scale)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        certificate = {
            "config_ids": list(expected_ids),
            "weighted_design_computations": 1,
            "normal_equation_computations": 1,
            "eigendecompositions": 1,
            "gram_sha256": array_sha256(gram),
            "rhs_sha256": array_sha256(rhs),
            "eigenvalues_sha256": array_sha256(eigenvalues),
            "eigenvectors_sha256": array_sha256(eigenvectors),
        }
        all_configs = elastic_net_configurations()
        for identity in ordered:
            config = configurations[identity.config_id]
            prefix = tuple(
                value
                for value in all_configs
                if value.rho == config.rho and value.lambda_index <= config.lambda_index
            )
            alphas = np.asarray(
                [module.lambda_value(value, reference) for value in prefix],
                dtype=np.float64,
            )
            lambda_value = float(alphas[-1])
            if config.rho == 0.0:
                ridge_eigenvalues = np.maximum(eigenvalues, 0.0)
                rotated = eigenvectors.T @ rhs
                path = eigenvectors @ (
                    rotated[:, None]
                    / (ridge_eigenvalues[:, None] + alphas[None, :])
                )
                coefficients = np.asarray(path[:, -1], dtype=np.float64)
                stationarity = rhs - gram @ coefficients - lambda_value * coefficients
                kkt = float(np.max(np.abs(stationarity)))
                kkt_limit = float(
                    tolerance * max(1.0, float(np.max(np.abs(rhs))), lambda_value)
                )
                if not math.isfinite(kkt) or kkt > kkt_limit:
                    raise SelectorRuntimeError("Grouped Elastic Net ridge failed its KKT gate")
                residual_mse = float(
                    y_second_moment
                    - 2.0 * np.dot(coefficients, rhs)
                    + coefficients @ gram @ coefficients
                )
                primal = 0.5 * residual_mse + 0.5 * lambda_value * float(
                    np.dot(coefficients, coefficients)
                )
                correlation = gram @ coefficients - rhs
                dual = (
                    y_second_moment
                    - float(np.dot(coefficients, rhs))
                    - 0.5 * residual_mse
                    - float(np.dot(correlation, correlation)) / (2.0 * lambda_value)
                )
                dual_gap = max(float(primal - dual), 0.0)
                dual_gap_limit = float(tolerance * max(1.0, y_second_moment))
                if dual_gap > dual_gap_limit:
                    raise SelectorRuntimeError("Grouped Elastic Net ridge failed its dual-gap gate")
                iterations = 1
                route = "ridge_closed_form"
            else:
                path_result = module._working_set_coordinate_descent_path(
                    rho=config.rho,
                    alphas=alphas,
                    tolerance=tolerance,
                    gram=gram,
                    rhs=rhs,
                    y_second_moment=y_second_moment,
                    gram_eigenvalues=eigenvalues,
                    gram_eigenvectors=eigenvectors,
                )
                coefficients = np.asarray(path_result.coefficients[:, -1], dtype=np.float64)
                kkt_values, kkt_limits = module._path_kkt_violations(
                    gram,
                    rhs,
                    path_result.coefficients[:, -1:],
                    alphas[-1:],
                    rho=config.rho,
                    tolerance=tolerance,
                )
                module._require_kkt(kkt_values, kkt_limits, label="Stage 11 grouped selected path")
                kkt = float(kkt_values[-1])
                kkt_limit = float(kkt_limits[-1])
                dual_gap = float(path_result.dual_gaps[-1])
                dual_gap_limit = float(tolerance * y_second_moment)
                iterations = int(path_result.total_iterations[-1])
                route = str(path_result.refinement_routes[-1])
            intercept = float(y_mean - np.dot(x_mean, coefficients))
            solutions.append(
                _elastic_artifact_from_solution(
                    identity=identity,
                    arrays=arrays,
                    config=config,
                    features=features,
                    reference=reference,
                    lambda_value=lambda_value,
                    coefficients=coefficients,
                    intercept=intercept,
                    dual_gap=dual_gap,
                    dual_gap_limit=dual_gap_limit,
                    kkt=kkt,
                    kkt_limit=kkt_limit,
                    iterations=iterations,
                    route=route,
                    controls=controls,
                    grouped_certificate=certificate,
                )
            )
    finally:
        if isinstance(weighted_X, np.memmap):
            weighted_X.flush()
        weighted_X = None
        if destination is not None and (destination.exists() or destination.is_symlink()):
            destination.unlink()
    return tuple(solutions)


def fit_locked_elastic_net_group(
    *,
    identities: Sequence[ValidationRefitIdentity],
    arrays: Any,
    elastic_scratch_path: Path | None = None,
) -> tuple[NativeModelArtifact, ...]:
    """Authoritative three-config ENet refit with one shared heavy setup."""

    return _fit_locked_elastic_net_group_impl(
        identities=identities,
        arrays=arrays,
        scratch_path=elastic_scratch_path,
        controls=None,
    )


def _fit_locked_elastic_net_group_nonauthoritative_for_test(
    *,
    identities: Sequence[ValidationRefitIdentity],
    arrays: Any,
    elastic_scratch_path: Path | None = None,
    controls: _NonauthoritativeTestControls | None = None,
) -> tuple[NativeModelArtifact, ...]:
    return _fit_locked_elastic_net_group_impl(
        identities=identities,
        arrays=arrays,
        scratch_path=elastic_scratch_path,
        controls=controls or _NonauthoritativeTestControls(),
    )


def fit_locked_selector_model(
    *,
    identity: ValidationRefitIdentity,
    arrays: ValidationRefitArrays,
    elastic_scratch_path: Path | None = None,
) -> NativeModelArtifact:
    """Fit one authoritative Stage 11 checkpoint inside its native worker."""

    validate_locked_selector(identity.family, identity.config_id, identity.seed)
    _validate_arrays(identity, arrays)
    if identity.family == "neural_network":
        if elastic_scratch_path is not None:
            raise SelectorRuntimeError("NN fit cannot carry Elastic Net scratch state")
        return _nn_fit(identity, arrays, controls=None)
    if identity.family == "xgboost":
        if elastic_scratch_path is not None:
            raise SelectorRuntimeError("XGBoost fit cannot carry Elastic Net scratch state")
        return _xgboost_fit(identity, arrays, controls=None)
    return _elastic_net_fit(
        identity,
        arrays,
        scratch_path=elastic_scratch_path,
        controls=None,
    )


def fit_locked_selector_model_from_paths(
    *,
    identity: ValidationRefitIdentity,
    request: ValidationRefitPathRequest,
    elastic_scratch_path: Path | None = None,
) -> NativeModelArtifact:
    """Authenticate read-only child mmaps, then run one locked native fit."""

    require_spawned_native_worker(identity.family)
    arrays = load_validation_refit_arrays_from_paths(request)
    return fit_locked_selector_model(
        identity=identity,
        arrays=arrays,  # type: ignore[arg-type]
        elastic_scratch_path=elastic_scratch_path,
    )


def fit_locked_elastic_net_group_from_paths(
    *,
    identities: Sequence[ValidationRefitIdentity],
    request: ValidationRefitPathRequest,
    elastic_scratch_path: Path | None = None,
) -> tuple[NativeModelArtifact, ...]:
    """Load one mmap request and fit all three locked ENets in one child."""

    require_spawned_native_worker("elastic_net")
    arrays = load_validation_refit_arrays_from_paths(request)
    return fit_locked_elastic_net_group(
        identities=identities,
        arrays=arrays,
        elastic_scratch_path=elastic_scratch_path,
    )


def _fit_locked_elastic_net_group_from_paths_nonauthoritative_for_test(
    *,
    identities: Sequence[ValidationRefitIdentity],
    request: ValidationRefitPathRequest,
    elastic_scratch_path: Path | None = None,
    controls: _NonauthoritativeTestControls | None = None,
) -> tuple[NativeModelArtifact, ...]:
    """Exercise the exact mmap/group route without producing authority."""

    require_spawned_native_worker("elastic_net")
    arrays = load_validation_refit_arrays_from_paths(request)
    return _fit_locked_elastic_net_group_nonauthoritative_for_test(
        identities=identities,
        arrays=arrays,
        elastic_scratch_path=elastic_scratch_path,
        controls=controls,
    )


def _fit_locked_selector_model_nonauthoritative_for_test(
    *,
    identity: ValidationRefitIdentity,
    arrays: ValidationRefitArrays,
    elastic_scratch_path: Path | None = None,
    controls: _NonauthoritativeTestControls | None = None,
) -> NativeModelArtifact:
    """Reduced synthetic fit; the returned status cannot pass publication QA."""

    test_controls = controls or _NonauthoritativeTestControls()
    validate_locked_selector(identity.family, identity.config_id, identity.seed)
    _validate_arrays(identity, arrays)
    if identity.family == "neural_network":
        return _nn_fit(identity, arrays, controls=test_controls)
    if identity.family == "xgboost":
        return _xgboost_fit(identity, arrays, controls=test_controls)
    return _elastic_net_fit(
        identity,
        arrays,
        scratch_path=elastic_scratch_path,
        controls=test_controls,
    )


def predict_locked_selector_model(
    *,
    checkpoint: LoadedSelectorCheckpoint,
    full_frozen_matrix: np.ndarray,
) -> np.ndarray:
    """Score a batch from persisted model and full-preprocessor state."""

    identity = checkpoint.identity
    family = str(identity["family"])
    config_id = str(identity["config_id"])
    seed = identity["seed"]
    validate_locked_selector(family, config_id, seed)
    module = _module(family)
    X = checkpoint.full_preprocessor.apply_support(full_frozen_matrix)
    if family == "neural_network":
        config = {value.config_id: value for value in neural_configurations()}[config_id]
        model = module.ShallowSelectorNetwork(X.shape[1], config)
        arrays = unpack_numeric_arrays(checkpoint.model_payload)
        expected_names = tuple(sorted(model.state_dict()))
        if tuple(sorted(arrays)) != expected_names:
            raise SelectorRuntimeError("NN checkpoint tensor names differ from architecture")
        state = {
            name: module.torch.from_numpy(np.asarray(arrays[name]).copy())
            for name in expected_names
        }
        model.load_state_dict(state, strict=True)
        prediction = module.predict_network(model, X, batch_size=module.BATCH_SIZE)
    elif family == "xgboost":
        model = module.xgb.Booster()
        model.load_model(bytearray(checkpoint.model_payload))
        prediction = np.asarray(model.inplace_predict(X), dtype=np.float32)
    else:
        arrays = unpack_numeric_arrays(checkpoint.model_payload)
        if set(arrays) != {"coefficients", "intercept"}:
            raise SelectorRuntimeError("Elastic Net checkpoint tensor names differ")
        coefficients = np.asarray(arrays["coefficients"], dtype=np.float64)
        intercept = np.asarray(arrays["intercept"], dtype=np.float64)
        if coefficients.shape != (X.shape[1],) or intercept.shape != (1,):
            raise SelectorRuntimeError("Elastic Net checkpoint dimensions differ")
        prediction = np.asarray(
            np.asarray(X, dtype=np.float64) @ coefficients + float(intercept[0]),
            dtype=np.float32,
        )
    if prediction.shape != (X.shape[0],) or not np.isfinite(prediction).all():
        raise SelectorRuntimeError("selector checkpoint emitted invalid raw forecasts")
    return np.ascontiguousarray(prediction, dtype=np.float32)


def predict_locked_selector_model_from_path(
    *,
    checkpoint: LoadedSelectorCheckpoint,
    request: FileBackedArrayRequest,
) -> np.ndarray:
    """Authenticate a read-only scoring mmap before native prediction."""

    family = str(checkpoint.identity["family"])
    require_spawned_native_worker(family)
    matrix = load_array_for_spawn(request)
    return predict_locked_selector_model(
        checkpoint=checkpoint,
        full_frozen_matrix=matrix,
    )


__all__ = [
    "fit_locked_elastic_net_group",
    "fit_locked_elastic_net_group_from_paths",
    "fit_locked_selector_model",
    "fit_locked_selector_model_from_paths",
    "predict_locked_selector_model",
    "predict_locked_selector_model_from_path",
]
