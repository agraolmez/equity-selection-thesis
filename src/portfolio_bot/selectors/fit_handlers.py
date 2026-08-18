"""Fold-scoped authoritative Stage 9 native-family fit handlers.

This module deliberately imports no model framework.  It is imported only
after :func:`selectors.native_worker.load_native_family_module` has claimed a
spawn child for exactly one native family.  A handler authenticates and opens
one prepared fold once, fixes all row partitions and feature supports once,
and publishes each fit through the content-addressed cache without returning
forecast arrays over a multiprocessing queue.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from portfolio_bot.selectors.cache import (
    canonical_json_bytes,
    job_id_from_identity,
    load_completed_job,
    publish_completed_job,
    sha256_file,
)
from portfolio_bot.selectors.contract import PROJECT_ROOT, validate_contract
from portfolio_bot.selectors.model_specs import (
    ELASTIC_NET_PATH_CONFIG_ID,
    canonical_xgboost_configuration,
    neural_configurations,
)
from portfolio_bot.selectors.native_worker import NativeFitSpec


_FAMILIES = frozenset({"neural_network", "xgboost", "elastic_net"})
_SHA256 = frozenset("0123456789abcdef")
_FAMILY_CODE_DEPENDENCIES = {
    "neural_network": "src/portfolio_bot/selectors/neural_model.py",
    "xgboost": "src/portfolio_bot/selectors/xgboost_model.py",
    "elastic_net": "src/portfolio_bot/selectors/elastic_net.py",
}
_COMMON_CODE_DEPENDENCIES = (
    "src/portfolio_bot/selectors/cache.py",
    "src/portfolio_bot/selectors/contract.py",
    "src/portfolio_bot/selectors/fit_handlers.py",
    "src/portfolio_bot/selectors/model_specs.py",
    "src/portfolio_bot/selectors/native_process.py",
    "src/portfolio_bot/selectors/native_worker.py",
    "src/portfolio_bot/selectors/prepared_data.py",
    "src/portfolio_bot/selectors/splits.py",
    "src/portfolio_bot/selectors/weights.py",
)
_APFS_STABLE_READ_ATTEMPTS = 3


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256


def family_code_sha256(family: str) -> str:
    """Hash the exact family-scoped source dependency set."""

    if family not in _FAMILIES:
        raise ValueError("Unknown Stage 9 fit-handler family")
    names = (*_COMMON_CODE_DEPENDENCIES, _FAMILY_CODE_DEPENDENCIES[family])
    hashes: dict[str, str] = {}
    for name in sorted(names):
        path = PROJECT_ROOT / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Unsafe Stage 9 fit code dependency: {name}")
        hashes[name] = sha256_file(path)
    return hashlib.sha256(canonical_json_bytes(hashes)).hexdigest()


def prepared_input_bundle_sha256(metadata: Mapping[str, Any]) -> str:
    """Bind a fit to the authenticated seven-table prepared input bundle."""

    receipts = metadata.get("input_receipts")
    if not isinstance(receipts, list) or len(receipts) != 7:
        raise RuntimeError("Prepared Stage 9 input receipts are incomplete")
    payload = {
        "contract_id": metadata.get("contract_id"),
        "prepared_build_id": metadata.get("build_id"),
        "prepared_content_fingerprint": metadata.get("content_fingerprint"),
        "input_receipts": receipts,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=bool))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def support_binding_sha256(
    family: str,
    fold_id: str,
    *,
    inner_base_support: np.ndarray | None = None,
    full_base_support: np.ndarray | None = None,
    elastic_support: np.ndarray | None = None,
) -> str:
    """Return the deterministic family/fold support-mask binding."""

    if family in {"neural_network", "xgboost"}:
        if inner_base_support is None or full_base_support is None or elastic_support is not None:
            raise ValueError("Base-family support binding requires inner and full masks only")
        masks = {
            "inner_base": _array_sha256(inner_base_support),
            "full_base": _array_sha256(full_base_support),
        }
    elif family == "elastic_net":
        if elastic_support is None or inner_base_support is not None or full_base_support is not None:
            raise ValueError("Elastic Net support binding requires the full elastic mask only")
        masks = {"full_elastic": _array_sha256(elastic_support)}
    else:
        raise ValueError("Unknown Stage 9 support family")
    return hashlib.sha256(
        canonical_json_bytes({"family": family, "fold_id": fold_id, "masks": masks})
    ).hexdigest()


def _safe_absolute_root(value: object, *, label: str, create: bool) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an absolute path string")
    root = Path(value)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError(f"{label} must be absolute and traversal-free")
    root = Path(os.path.abspath(root.expanduser()))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise RuntimeError(f"Unsafe {label} component: {current}")
        elif create:
            current.mkdir(mode=0o700)
        else:
            raise RuntimeError(f"Missing {label}: {root}")
    return root


def _active_columns(matrix: np.ndarray, rows: np.ndarray, support: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(np.asarray(support, dtype=bool))
    if indices.size == 0:
        raise RuntimeError("A Stage 9 fit has no supported predictor columns")
    result = np.asarray(matrix[np.ix_(np.asarray(rows, dtype=np.int64), indices)], dtype=np.float32)
    if result.shape != (len(rows), indices.size) or not np.isfinite(result).all():
        raise RuntimeError("Stage 9 prepared fit partition is invalid")
    return np.ascontiguousarray(result)


def _target(rows: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(rows["residual_return_rank"], errors="coerce").to_numpy(np.float64)
    result = np.asarray(values[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    if result.shape != (len(indices),) or not np.isfinite(result).all():
        raise RuntimeError("Stage 9 training target partition is invalid")
    return result


def _indices_sha256(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _load_completed_job_with_bounded_retry(
    path: Path, **kwargs: Any
) -> dict[str, Any] | None:
    """Retry a transient APFS metadata settlement without accepting instability."""

    for _attempt in range(_APFS_STABLE_READ_ATTEMPTS):
        receipt = load_completed_job(path, **kwargs)
        if receipt is not None:
            return receipt
    return None


def _publish_completed_job_with_bounded_retry(
    path: Path, **kwargs: Any
) -> dict[str, Any]:
    """Retry only the fail-closed temporary-file stable-hash race.

    Every individual attempt still requires an unchanged full stat identity
    around its hash.  Other runtime errors are never retried or suppressed.
    """

    last_error: RuntimeError | None = None
    for _attempt in range(_APFS_STABLE_READ_ATTEMPTS):
        try:
            return publish_completed_job(path, **kwargs)
        except RuntimeError as exc:
            if "Stage 9 cache file changed during hashing:" not in str(exc):
                raise
            last_error = exc
    assert last_error is not None
    raise last_error


class _FoldFitHandler:
    def __init__(
        self,
        *,
        family: str,
        native_module: Any,
        prepared: Any,
        cache_root: Path,
        scratch_root: Path | None,
        gamma_authorization: Mapping[str, Any] | None,
    ) -> None:
        self.family = family
        self.native_module = native_module
        self.prepared = prepared
        self.fold = prepared.fold
        self.fold_id = str(self.fold.fold_id)
        self.cache_root = cache_root
        self.scratch_root = scratch_root
        self.gamma_authorization = (
            None if gamma_authorization is None else dict(gamma_authorization)
        )
        self.input_bundle_sha256 = prepared_input_bundle_sha256(prepared.metadata)
        self.code_sha256 = family_code_sha256(family)
        self._neural_configs = {value.config_id: value for value in neural_configurations()}
        self._fixed_arrays: dict[str, np.ndarray] = {}
        self._xgboost_context: Any | None = None

        if family in {"neural_network", "xgboost"}:
            inner_support = np.asarray(prepared.inner_base_support, dtype=bool)
            full_support = np.asarray(prepared.full_base_support, dtype=bool)
            self.support_sha256 = support_binding_sha256(
                family,
                self.fold_id,
                inner_base_support=inner_support,
                full_base_support=full_support,
            )
            fixed = {
                "X_inner_train": _active_columns(
                    prepared.inner_base_matrix, self.fold.inner_train, inner_support
                ),
                "y_inner_train": _target(prepared.rows, self.fold.inner_train),
                "w_inner_train": np.asarray(self.fold.inner_train_weights, dtype=np.float64),
                "X_inner_validation": _active_columns(
                    prepared.inner_base_matrix, self.fold.inner_validation, inner_support
                ),
                "y_inner_validation": _target(prepared.rows, self.fold.inner_validation),
                "w_inner_validation": np.asarray(
                    self.fold.inner_validation_weights, dtype=np.float64
                ),
                "X_full_train": _active_columns(
                    prepared.full_base_matrix, self.fold.full_train, full_support
                ),
                "y_full_train": _target(prepared.rows, self.fold.full_train),
                "w_full_train": np.asarray(self.fold.full_train_weights, dtype=np.float64),
                "X_outer_validation": _active_columns(
                    prepared.full_base_matrix, self.fold.outer_validation, full_support
                ),
            }
            if family == "xgboost":
                self._xgboost_context = native_module.prepare_xgboost_fold_context(
                    fold_id=self.fold_id,
                    **fixed,
                )
            else:
                self._fixed_arrays = fixed
            self.feature_evidence = {
                "inner_matrix": f"base_matrix_inner_{self.fold_id}.npy",
                "full_matrix": f"base_matrix_full_{self.fold_id}.npy",
                "inner_supported_features": int(inner_support.sum()),
                "full_supported_features": int(full_support.sum()),
                "early_stop_preprocessing_fit_scope": "inner_train_only",
                "refit_and_oof_preprocessing_fit_scope": "full_outer_training_only",
            }
        else:
            support = np.asarray(prepared.elastic_support, dtype=bool)
            self.support_sha256 = support_binding_sha256(
                family,
                self.fold_id,
                elastic_support=support,
            )
            self._fixed_arrays = {
                "X_train": _active_columns(
                    prepared.elastic_matrix, self.fold.full_train, support
                ),
                "y_train": _target(prepared.rows, self.fold.full_train),
                "w_train": np.asarray(self.fold.full_train_weights, dtype=np.float64),
                "X_outer_validation": _active_columns(
                    prepared.elastic_matrix, self.fold.outer_validation, support
                ),
            }
            self.feature_evidence = {
                "full_matrix": f"elastic_matrix_full_{self.fold_id}.npy",
                "full_supported_features": int(support.sum()),
                "fit_scope": "full_outer_training_only",
                "disk_backed_weighted_centered_design": True,
            }

    def _validate_spec(self, specification: NativeFitSpec) -> tuple[dict[str, Any], Path]:
        spec = specification.validated()
        if spec.family != self.family or spec.fold_id != self.fold_id:
            raise ValueError("Stage 9 fit specification escaped its initialized family/fold shard")
        if Path(spec.prepared_root).resolve() != self.prepared.root.resolve():
            raise ValueError("Stage 9 fit specification changed its authenticated prepared root")
        if spec.input_bundle_sha256 != self.input_bundle_sha256:
            raise ValueError("Stage 9 fit input-bundle binding differs")
        if spec.code_sha256 != self.code_sha256:
            raise ValueError("Stage 9 fit code binding differs")
        if spec.support_sha256 != self.support_sha256:
            raise ValueError("Stage 9 fit support binding differs")
        if spec.sealed_core_receipt_sha256 is not None:
            authorization = self.gamma_authorization
            if (
                self.family != "xgboost"
                or not isinstance(authorization, dict)
                or authorization.get("sealed_core_receipt_sha256")
                != spec.sealed_core_receipt_sha256
                or spec.config_payload_sha256
                not in set(authorization.get("authorized_config_payload_sha256s", ()))
            ):
                raise ValueError("Stage 9 gamma fit differs from its sealed core authorization")
        elif self.gamma_authorization is not None:
            raise ValueError("A sealed gamma shard cannot serve an XGBoost core fit")
        identity = spec.job_identity()
        expected_path = self.cache_root / f"{job_id_from_identity(identity)}.npz"
        observed_path = Path(spec.cache_path)
        if observed_path != expected_path:
            raise ValueError("Stage 9 cache path is not the canonical content-addressed job path")
        return identity, observed_path

    def _model_fit(self, spec: NativeFitSpec, *, job_id: str) -> tuple[np.ndarray, Any, Any]:
        if self.family == "neural_network":
            config = self._neural_configs[spec.config_id]
            predictions, receipt = self.native_module.fit_predict_outer_fold(
                config=config,
                fold_id=self.fold_id,
                seed=int(spec.seed),
                **self._fixed_arrays,
            )
            return predictions, receipt, asdict(config)
        if self.family == "xgboost":
            config = canonical_xgboost_configuration(spec.config_id)
            predictions, receipt = self._xgboost_context.fit_predict(
                config=config,
                seed=int(spec.seed),
                gamma_authorization=self.gamma_authorization,
            )
            return predictions, receipt, asdict(config)
        if spec.config_id != ELASTIC_NET_PATH_CONFIG_ID or self.scratch_root is None:
            raise ValueError("Elastic Net shard requires the canonical path job and scratch root")
        scratch = self.scratch_root / f"{job_id}.weighted-f32.memmap"
        if scratch.is_symlink():
            raise RuntimeError("Unsafe Stage 9 Elastic Net scratch symlink")
        if scratch.exists():
            if not scratch.is_file():
                raise RuntimeError("Unsafe Stage 9 Elastic Net scratch object")
            scratch.unlink()
        predictions, receipts = self.native_module.fit_predict_outer_fold(
            fold_id=self.fold_id,
            work_path=scratch,
            **self._fixed_arrays,
        )
        if scratch.exists() or scratch.is_symlink():
            raise RuntimeError("Elastic Net weighted-design scratch was not removed")
        return predictions, receipts, {
            "config_id": ELASTIC_NET_PATH_CONFIG_ID,
            "rho_values": [0.0, 0.25, 0.5, 0.75, 1.0],
            "lambda_points_per_rho": 50,
            "configuration_count": 250,
        }

    def _fit_receipt(
        self,
        spec: NativeFitSpec,
        predictions: np.ndarray,
        model_receipt: Any,
        configuration: Mapping[str, Any],
    ) -> dict[str, Any]:
        model_evidence = (
            [value.as_dict() for value in model_receipt]
            if isinstance(model_receipt, tuple)
            else model_receipt.as_dict()
        )
        publication_receipts = self._publication_fit_receipts(
            spec, predictions, model_receipt
        )
        return {
            "schema_version": 1,
            "status": "AUTHORITATIVE_STAGE09_FIT_COMPLETE",
            "family": self.family,
            "config_id": spec.config_id,
            "fold_id": self.fold_id,
            "seed": spec.seed,
            "scientific_scope": {
                "development_only": True,
                "target": "residual_return_rank",
                "loss": "month_weighted_mse",
                "genuine_oof": True,
                "portfolio_or_backtest_permitted": False,
                "validation_2015_2019_accessed": False,
                "final_2020_2025_accessed": False,
            },
            "prepared_cache": {
                "build_id": self.prepared.metadata["build_id"],
                "content_fingerprint": self.prepared.metadata["content_fingerprint"],
                "root": str(self.prepared.root),
            },
            "bindings": {
                "rules_sha256": spec.rules_sha256,
                "runtime_versions_sha256": spec.runtime_versions_sha256,
                "input_bundle_sha256": self.input_bundle_sha256,
                "code_sha256": self.code_sha256,
                "support_sha256": self.support_sha256,
                "config_payload_sha256": spec.config_payload_sha256,
                "sealed_core_receipt_sha256": spec.sealed_core_receipt_sha256,
            },
            "configuration": dict(configuration),
            "features": dict(self.feature_evidence),
            "partitions": {
                "full_train_rows": int(self.fold.full_train.size),
                "inner_train_rows": int(self.fold.inner_train.size),
                "inner_purged_rows": int(self.fold.inner_purged.size),
                "inner_validation_rows": int(self.fold.inner_validation.size),
                "outer_validation_rows": int(self.fold.outer_validation.size),
                "outer_evaluation_rows": int(self.fold.outer_evaluation.size),
                "full_train_indices_sha256": _indices_sha256(self.fold.full_train),
                "inner_train_indices_sha256": _indices_sha256(self.fold.inner_train),
                "inner_purged_indices_sha256": _indices_sha256(self.fold.inner_purged),
                "inner_validation_indices_sha256": _indices_sha256(
                    self.fold.inner_validation
                ),
                "outer_validation_indices_sha256": _indices_sha256(
                    self.fold.outer_validation
                ),
            },
            "weights": {
                "month_initial_total_weight": 1.0,
                "memory_half_life_months": 120.0,
                "normalization": "mean_one_sum_equals_fit_row_count",
                "full_train_sum": float(np.sum(self.fold.full_train_weights)),
                "inner_train_sum": float(np.sum(self.fold.inner_train_weights)),
                "inner_validation_sum": float(
                    np.sum(self.fold.inner_validation_weights)
                ),
            },
            "forecast": {
                "rows": int(np.asarray(predictions).shape[0]),
                "columns": (
                    1 if np.asarray(predictions).ndim == 1 else int(np.asarray(predictions).shape[1])
                ),
                "dtype": "float32",
                "all_finite": bool(np.isfinite(predictions).all()),
            },
            "model_fit": model_evidence,
            # This normalized ledger is intentionally stored beside the full
            # native receipt evidence.  Four Elastic Net path jobs expand to
            # 1,000 per-configuration receipts; stochastic jobs remain 1:1.
            "publication_fit_receipts": publication_receipts,
        }

    def _publication_fit_receipts(
        self,
        spec: NativeFitSpec,
        predictions: np.ndarray,
        model_receipt: Any,
    ) -> list[dict[str, Any]]:
        forecast = np.asarray(predictions, dtype=np.float32)
        parent_job_id = job_id_from_identity(spec.job_identity())
        native_receipts = model_receipt if isinstance(model_receipt, tuple) else (model_receipt,)
        if self.family == "elastic_net":
            if forecast.ndim != 2 or forecast.shape[1] != len(native_receipts):
                raise RuntimeError("Elastic Net receipt/prediction path widths differ")
        elif forecast.ndim != 1 or len(native_receipts) != 1:
            raise RuntimeError("Stochastic fit receipt/prediction dimensions differ")

        result: list[dict[str, Any]] = []
        for position, native in enumerate(native_receipts):
            evidence = native.as_dict()
            config_id = str(evidence["config_id"])
            raw = forecast if self.family != "elastic_net" else forecast[:, position]
            raw = np.ascontiguousarray(raw, dtype=np.float32)
            if self.family == "elastic_net":
                child_payload = {
                    "parent_path_job_id": parent_job_id,
                    "family": self.family,
                    "config_id": config_id,
                    "fold_id": self.fold_id,
                    "seed": None,
                }
                child_job_id = f"stage09-fit-{hashlib.sha256(canonical_json_bytes(child_payload)).hexdigest()}"
            else:
                child_job_id = parent_job_id
            if self.family == "neural_network":
                iterations = int(evidence["epochs_examined"])
                inner_best = int(evidence["best_epoch"])
                refit_count = int(evidence["best_epoch"])
                converged = True
                fit_status = "EARLY_STOPPED_AND_REFIT" if evidence["stopped_early"] else "MAX_EPOCHS_EXAMINED_AND_REFIT"
            elif self.family == "xgboost":
                iterations = int(evidence["best_tree_count"])
                inner_best = int(evidence["best_tree_count"])
                refit_count = int(evidence["best_tree_count"])
                converged = True
                fit_status = "EARLY_STOPPED_AND_REFIT"
            else:
                iterations = max(1, int(evidence["iterations"]))
                inner_best = None
                refit_count = int(evidence["iterations"])
                converged = bool(evidence["converged"])
                fit_status = str(evidence["solver"])
            result.append(
                {
                    "model_family": self.family,
                    "config_id": config_id,
                    "fold_id": self.fold_id,
                    "seed": spec.seed,
                    "job_id": child_job_id,
                    "job_status": "COMPLETE",
                    "prediction_sha256": hashlib.sha256(raw.tobytes(order="C")).hexdigest(),
                    "prediction_rows": int(raw.size),
                    "input_feature_count": int(evidence["input_features"]),
                    "input_rows": int(self.fold.full_train.size + self.fold.outer_validation.size),
                    "training_rows": int(evidence["train_rows"]),
                    "inner_training_rows": (
                        int(evidence["inner_train_rows"])
                        if self.family != "elastic_net"
                        else None
                    ),
                    "inner_validation_rows": (
                        int(evidence["inner_validation_rows"])
                        if self.family != "elastic_net"
                        else None
                    ),
                    "outer_validation_rows": int(evidence["outer_validation_rows"]),
                    "parameter_count": (
                        int(evidence["parameter_count"])
                        if self.family == "neural_network"
                        else None
                    ),
                    "best_tree_count": (
                        int(evidence["best_tree_count"])
                        if self.family == "xgboost"
                        else None
                    ),
                    "nonzero_coefficients": (
                        int(evidence["nonzero_coefficients"])
                        if self.family == "elastic_net"
                        else None
                    ),
                    "dual_gap": (
                        float(evidence["dual_gap"])
                        if self.family == "elastic_net"
                        else None
                    ),
                    "kkt_max_violation": (
                        float(evidence["kkt_max_violation"])
                        if self.family == "elastic_net"
                        else None
                    ),
                    "kkt_limit": (
                        float(evidence["kkt_limit"])
                        if self.family == "elastic_net"
                        else None
                    ),
                    "iterations_or_epochs": iterations,
                    "elapsed_seconds": float(evidence["elapsed_seconds"]),
                    "inner_best_iteration": inner_best,
                    "refit_iteration_count": refit_count,
                    "fit_converged": converged,
                    "fit_status": fit_status,
                }
            )
        return result

    @staticmethod
    def _completion(receipt: Mapping[str, Any], *, cache_path: Path, cache_hit: bool) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "COMPLETE",
            "cache_hit": cache_hit,
            "job_id": receipt["job_id"],
            "job_identity_sha256": receipt["job_identity_sha256"],
            "prediction_sha256": receipt["prediction_sha256"],
            "fit_receipt_sha256": receipt["fit_receipt_sha256"],
            "cache_path": str(cache_path),
        }

    def __call__(self, specification: NativeFitSpec) -> dict[str, Any]:
        identity, cache_path = self._validate_spec(specification)
        schema = specification.expected_array_schema(self.fold.outer_validation.size)
        cached = _load_completed_job_with_bounded_retry(
            cache_path,
            trusted_root=self.cache_root,
            expected_identity=identity,
            expected_array_schema=schema,
        )
        if cached is not None:
            return self._completion(cached, cache_path=cache_path, cache_hit=True)
        job_id = job_id_from_identity(identity)
        predictions, model_receipt, configuration = self._model_fit(
            specification, job_id=job_id
        )
        fit_receipt = self._fit_receipt(
            specification, predictions, model_receipt, configuration
        )
        published = _publish_completed_job_with_bounded_retry(
            cache_path,
            trusted_root=self.cache_root,
            job_identity=identity,
            predictions=np.asarray(predictions, dtype=np.float32),
            fit_receipt=fit_receipt,
            expected_array_schema=schema,
        )
        return self._completion(published, cache_path=cache_path, cache_hit=False)

    def close(self) -> None:
        self._fixed_arrays.clear()
        self._xgboost_context = None
        self.prepared = None


def create_fold_fit_handler(
    *, native_module: Any, initialization_payload: dict[str, Any]
) -> _FoldFitHandler:
    """Authenticate one family/fold cache and return its long-lived handler.

    Required payload fields are ``family``, ``fold_id``, ``prepared_root``,
    ``expected_build_id``, ``expected_content_fingerprint``, and ``cache_root``.
    Elastic Net additionally requires ``scratch_root``.  XGBoost gamma shards
    carry the sealed ``gamma_authorization`` already checked by the bootstrap.
    """

    family = str(initialization_payload.get("family"))
    fold_id = str(initialization_payload.get("fold_id"))
    if family not in _FAMILIES:
        raise ValueError("Stage 9 handler initialization family is not frozen")
    prepared_root = _safe_absolute_root(
        initialization_payload.get("prepared_root"),
        label="Stage 9 prepared root",
        create=False,
    )
    cache_root = _safe_absolute_root(
        initialization_payload.get("cache_root"),
        label="Stage 9 fit cache root",
        create=True,
    )
    scratch_root = None
    if family == "elastic_net":
        scratch_root = _safe_absolute_root(
            initialization_payload.get("scratch_root"),
            label="Stage 9 Elastic Net scratch root",
            create=True,
        )
    elif initialization_payload.get("scratch_root") is not None:
        raise ValueError("Only Elastic Net handlers may carry a scratch root")
    gamma_authorization = initialization_payload.get("gamma_authorization")
    if gamma_authorization is not None and family != "xgboost":
        raise ValueError("Only XGBoost handlers may carry gamma authorization")
    expected_build_id = initialization_payload.get("expected_build_id")
    expected_fingerprint = initialization_payload.get("expected_content_fingerprint")
    if not isinstance(expected_build_id, str) or not _is_sha256(expected_fingerprint):
        raise ValueError("Stage 9 handler lacks a trusted prepared-build identity")

    # Imported only inside the already family-claimed spawn child.
    from portfolio_bot.selectors.prepared_data import load_prepared_selector_fold

    prepared = load_prepared_selector_fold(
        prepared_root,
        expected_build_id=expected_build_id,
        expected_content_fingerprint=expected_fingerprint,
        fold_id=fold_id,
        family=family,
    )
    policy, _ = validate_contract()
    if policy["physical_rules"]["sha256"] != initialization_payload.get(
        "rules_sha256", policy["physical_rules"]["sha256"]
    ):
        raise ValueError("Stage 9 handler initialization rules binding differs")
    return _FoldFitHandler(
        family=family,
        native_module=native_module,
        prepared=prepared,
        cache_root=cache_root,
        scratch_root=scratch_root,
        gamma_authorization=gamma_authorization,
    )


__all__ = [
    "create_fold_fit_handler",
    "family_code_sha256",
    "prepared_input_bundle_sha256",
    "support_binding_sha256",
]
