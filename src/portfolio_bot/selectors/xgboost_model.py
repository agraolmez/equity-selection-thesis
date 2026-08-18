"""Frozen XGBoost selector grid and deterministic fold fitting."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from portfolio_bot.selectors.contract import validate_runtime_versions
from portfolio_bot.selectors.model_specs import (
    DEVELOPMENT_SEEDS,
    XGBoostConfig,
    configuration_payload_sha256,
    core_xgboost_configurations,
    gamma_challenger_configurations,
    require_canonical_xgboost_config,
)
from portfolio_bot.selectors.native_process import (
    configure_single_thread_environment,
    require_spawned_native_worker,
)

configure_single_thread_environment()
require_spawned_native_worker("xgboost")

import numpy as np
import xgboost as xgb


MAX_BOOSTING_ROUNDS = 6_000
EARLY_STOPPING_ROUNDS = 200


@dataclass(frozen=True)
class XGBoostFitReceipt:
    config_id: str
    fold_id: str
    seed: int
    input_features: int
    inner_input_features: int
    train_rows: int
    inner_train_rows: int
    inner_validation_rows: int
    outer_validation_rows: int
    min_child_weight_inner: float
    min_child_weight_full: float
    best_tree_count: int
    best_inner_rmse: float
    elapsed_seconds: float
    xgboost_version: str
    device: str
    threads: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class XGBoostPreparedFold:
    """One fold's reusable, single-threaded DMatrix context for a long-lived shard."""

    fold_id: str
    inner_train: Any
    inner_validation: Any
    full_train: Any
    outer_validation: np.ndarray
    input_features: int
    inner_input_features: int
    inner_train_rows: int
    inner_validation_rows: int
    full_train_rows: int
    inner_weight_sum: float
    full_weight_sum: float
    threads: int = 1

    def _fit(
        self,
        *,
        config: XGBoostConfig,
        seed: int,
        max_boosting_rounds: int,
        early_stopping_rounds: int,
    ) -> tuple[np.ndarray, XGBoostFitReceipt]:
        started = time.monotonic()
        inner_child = float(config.child_share * self.inner_weight_sum)
        full_child = float(config.child_share * self.full_weight_sum)
        early_model = xgb.train(
            _params(
                config,
                seed=seed,
                min_child_weight=inner_child,
                threads=self.threads,
            ),
            self.inner_train,
            num_boost_round=max_boosting_rounds,
            evals=[(self.inner_validation, "inner_validation")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        best_tree_count = int(early_model.best_iteration) + 1
        best_score = float(early_model.best_score)
        if best_tree_count <= 0 or not math.isfinite(best_score):
            raise RuntimeError("XGBoost early stopping produced an invalid result")
        del early_model
        final_model = xgb.train(
            _params(
                config,
                seed=seed,
                min_child_weight=full_child,
                threads=self.threads,
            ),
            self.full_train,
            num_boost_round=best_tree_count,
            verbose_eval=False,
        )
        predictions = np.asarray(
            final_model.inplace_predict(self.outer_validation), dtype=np.float32
        )
        if predictions.shape != (self.outer_validation.shape[0],) or not np.isfinite(
            predictions
        ).all():
            raise FloatingPointError("XGBoost emitted invalid predictions")
        return predictions, XGBoostFitReceipt(
            config_id=config.config_id,
            fold_id=self.fold_id,
            seed=seed,
            input_features=self.input_features,
            inner_input_features=self.inner_input_features,
            train_rows=self.full_train_rows,
            inner_train_rows=self.inner_train_rows,
            inner_validation_rows=self.inner_validation_rows,
            outer_validation_rows=self.outer_validation.shape[0],
            min_child_weight_inner=inner_child,
            min_child_weight_full=full_child,
            best_tree_count=best_tree_count,
            best_inner_rmse=best_score,
            elapsed_seconds=float(time.monotonic() - started),
            xgboost_version=xgb.__version__,
            device="cpu",
            threads=self.threads,
        )

    def fit_predict(
        self,
        *,
        config: XGBoostConfig,
        seed: int,
        gamma_authorization: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, XGBoostFitReceipt]:
        require_spawned_native_worker("xgboost")
        _require_authoritative_xgboost_job(
            config=config,
            seed=seed,
            fold_id=self.fold_id,
            gamma_authorization=gamma_authorization,
        )
        return self._fit(
            config=config,
            seed=seed,
            max_boosting_rounds=MAX_BOOSTING_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )

    def _fit_predict_nonauthoritative_for_test(
        self,
        *,
        config: XGBoostConfig,
        seed: int,
        max_boosting_rounds: int = 30,
        early_stopping_rounds: int = 5,
    ) -> tuple[np.ndarray, XGBoostFitReceipt]:
        require_spawned_native_worker("xgboost")
        return self._fit(
            config=config,
            seed=seed,
            max_boosting_rounds=max_boosting_rounds,
            early_stopping_rounds=early_stopping_rounds,
        )


def _validated(
    X: np.ndarray, y: np.ndarray | None = None, weight: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    features = np.asarray(X, dtype=np.float32, order="C")
    if features.ndim != 2 or features.shape[1] < 1 or not np.isfinite(features).all():
        raise ValueError("XGBoost features must be a finite two-dimensional matrix")
    target = None if y is None else np.asarray(y, dtype=np.float32)
    weights = None if weight is None else np.asarray(weight, dtype=np.float32)
    if target is not None:
        if target.shape != (features.shape[0],) or not np.isfinite(target).all():
            raise ValueError("Invalid XGBoost target")
    if weights is not None:
        if weights.shape != (features.shape[0],) or not (
            np.isfinite(weights).all() and np.all(weights > 0)
        ):
            raise ValueError("Invalid XGBoost weights")
        weights = weights * (float(weights.size) / float(np.sum(weights, dtype=np.float64)))
    return features, target, weights


def child_share_to_min_child_weight(child_share: float, sample_weight: np.ndarray) -> float:
    weights = np.asarray(sample_weight, dtype=np.float64)
    if not 0 < child_share <= 1 or weights.ndim != 1 or not (
        np.isfinite(weights).all() and np.all(weights > 0)
    ):
        raise ValueError("Invalid child share or sample weights")
    return float(child_share * np.sum(weights, dtype=np.float64))


def _params(
    config: XGBoostConfig,
    *,
    seed: int,
    min_child_weight: float,
    threads: int,
) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "tree_method": "hist",
        "grow_policy": "depthwise",
        "max_bin": 256,
        "max_depth": config.max_depth,
        "eta": config.eta,
        "min_child_weight": min_child_weight,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "lambda": config.reg_lambda,
        "alpha": config.reg_alpha,
        "gamma": config.gamma,
        "base_score": 0.0,
        "sampling_method": "uniform",
        "max_delta_step": 0.0,
        "seed": seed,
        "seed_per_iteration": False,
        "device": "cpu",
        "nthread": threads,
        "verbosity": 0,
    }


def _prepare_xgboost_fold_context_impl(
    *,
    fold_id: str,
    X_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    w_inner_train: np.ndarray,
    X_inner_validation: np.ndarray,
    y_inner_validation: np.ndarray,
    w_inner_validation: np.ndarray,
    X_full_train: np.ndarray,
    y_full_train: np.ndarray,
    w_full_train: np.ndarray,
    X_outer_validation: np.ndarray,
    threads: int,
) -> XGBoostPreparedFold:
    Xi, yi, wi = _validated(X_inner_train, y_inner_train, w_inner_train)
    Xv, yv, wv = _validated(X_inner_validation, y_inner_validation, w_inner_validation)
    Xf, yf, wf = _validated(X_full_train, y_full_train, w_full_train)
    Xo, _, _ = _validated(X_outer_validation)
    assert yi is not None and wi is not None and yv is not None and wv is not None
    assert yf is not None and wf is not None
    if Xi.shape[1] != Xv.shape[1]:
        raise ValueError("Prepared XGBoost inner feature widths differ")
    if Xf.shape[1] != Xo.shape[1]:
        raise ValueError("Prepared XGBoost full/outer feature widths differ")
    return XGBoostPreparedFold(
        fold_id=fold_id,
        inner_train=xgb.DMatrix(Xi, label=yi, weight=wi, nthread=threads),
        inner_validation=xgb.DMatrix(Xv, label=yv, weight=wv, nthread=threads),
        full_train=xgb.DMatrix(Xf, label=yf, weight=wf, nthread=threads),
        outer_validation=Xo,
        input_features=Xf.shape[1],
        inner_input_features=Xi.shape[1],
        inner_train_rows=Xi.shape[0],
        inner_validation_rows=Xv.shape[0],
        full_train_rows=Xf.shape[0],
        inner_weight_sum=float(np.sum(wi, dtype=np.float64)),
        full_weight_sum=float(np.sum(wf, dtype=np.float64)),
        threads=threads,
    )


def prepare_xgboost_fold_context(
    *,
    fold_id: str,
    X_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    w_inner_train: np.ndarray,
    X_inner_validation: np.ndarray,
    y_inner_validation: np.ndarray,
    w_inner_validation: np.ndarray,
    X_full_train: np.ndarray,
    y_full_train: np.ndarray,
    w_full_train: np.ndarray,
    X_outer_validation: np.ndarray,
) -> XGBoostPreparedFold:
    """Build reusable DMatrix objects once for an authoritative family/fold shard."""

    require_spawned_native_worker("xgboost")
    validate_runtime_versions()
    if fold_id not in {"dev_fold_1", "dev_fold_2", "dev_fold_3", "dev_fold_4"}:
        raise ValueError("Authoritative Stage 9 XGBoost fold is outside the frozen folds")
    return _prepare_xgboost_fold_context_impl(
        fold_id=fold_id,
        X_inner_train=X_inner_train,
        y_inner_train=y_inner_train,
        w_inner_train=w_inner_train,
        X_inner_validation=X_inner_validation,
        y_inner_validation=y_inner_validation,
        w_inner_validation=w_inner_validation,
        X_full_train=X_full_train,
        y_full_train=y_full_train,
        w_full_train=w_full_train,
        X_outer_validation=X_outer_validation,
        threads=1,
    )


def _fit_predict_outer_fold_impl(
    *,
    config: XGBoostConfig,
    fold_id: str,
    seed: int,
    X_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    w_inner_train: np.ndarray,
    X_inner_validation: np.ndarray,
    y_inner_validation: np.ndarray,
    w_inner_validation: np.ndarray,
    X_full_train: np.ndarray,
    y_full_train: np.ndarray,
    w_full_train: np.ndarray,
    X_outer_validation: np.ndarray,
    threads: int = 1,
    max_boosting_rounds: int = MAX_BOOSTING_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> tuple[np.ndarray, XGBoostFitReceipt]:
    if threads <= 0:
        raise ValueError("threads must be positive")
    started = time.monotonic()
    Xi, yi, wi = _validated(X_inner_train, y_inner_train, w_inner_train)
    Xv, yv, wv = _validated(X_inner_validation, y_inner_validation, w_inner_validation)
    Xf, yf, wf = _validated(X_full_train, y_full_train, w_full_train)
    Xo, _, _ = _validated(X_outer_validation)
    assert yi is not None and wi is not None and yv is not None and wv is not None
    assert yf is not None and wf is not None
    inner_train_rows = Xi.shape[0]
    inner_input_features = Xi.shape[1]
    inner_validation_rows = Xv.shape[0]
    full_train_rows = Xf.shape[0]
    outer_validation_rows = Xo.shape[0]
    if Xi.shape[1] != Xv.shape[1]:
        raise ValueError("XGBoost inner feature widths differ")
    if Xf.shape[1] != Xo.shape[1]:
        raise ValueError("XGBoost full/outer feature widths differ")
    inner_child = child_share_to_min_child_weight(config.child_share, wi)
    full_child = child_share_to_min_child_weight(config.child_share, wf)
    inner_train = xgb.DMatrix(Xi, label=yi, weight=wi, nthread=threads)
    inner_validation = xgb.DMatrix(Xv, label=yv, weight=wv, nthread=threads)
    early_model = xgb.train(
        _params(
            config,
            seed=seed,
            min_child_weight=inner_child,
            threads=threads,
        ),
        inner_train,
        num_boost_round=max_boosting_rounds,
        evals=[(inner_validation, "inner_validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    best_tree_count = int(early_model.best_iteration) + 1
    best_score = float(early_model.best_score)
    if best_tree_count <= 0 or not math.isfinite(best_score):
        raise RuntimeError("XGBoost early stopping produced an invalid result")
    del early_model, inner_train, inner_validation, Xi, yi, wi, Xv, yv, wv
    full_train = xgb.DMatrix(Xf, label=yf, weight=wf, nthread=threads)
    final_model = xgb.train(
        _params(
            config,
            seed=seed,
            min_child_weight=full_child,
            threads=threads,
        ),
        full_train,
        num_boost_round=best_tree_count,
        verbose_eval=False,
    )
    predictions = np.asarray(final_model.inplace_predict(Xo), dtype=np.float32)
    if predictions.shape != (Xo.shape[0],) or not np.isfinite(predictions).all():
        raise FloatingPointError("XGBoost emitted invalid predictions")
    receipt = XGBoostFitReceipt(
        config_id=config.config_id,
        fold_id=fold_id,
        seed=seed,
        input_features=Xf.shape[1],
        inner_input_features=inner_input_features,
        train_rows=full_train_rows,
        inner_train_rows=inner_train_rows,
        inner_validation_rows=inner_validation_rows,
        outer_validation_rows=outer_validation_rows,
        min_child_weight_inner=inner_child,
        min_child_weight_full=full_child,
        best_tree_count=best_tree_count,
        best_inner_rmse=best_score,
        elapsed_seconds=float(time.monotonic() - started),
        xgboost_version=xgb.__version__,
        device="cpu",
        threads=threads,
    )
    return predictions, receipt


def _require_authoritative_xgboost_job(
    *,
    config: XGBoostConfig,
    seed: int,
    fold_id: str,
    gamma_authorization: Mapping[str, Any] | None,
) -> XGBoostConfig:
    canonical = require_canonical_xgboost_config(config)
    if canonical.challenger:
        authorized_hashes = (
            ()
            if gamma_authorization is None
            else gamma_authorization.get("authorized_config_payload_sha256s", ())
        )
        sealed_receipt = (
            None
            if gamma_authorization is None
            else gamma_authorization.get("sealed_core_receipt_sha256")
        )
        if (
            not isinstance(sealed_receipt, str)
            or len(sealed_receipt) != 64
            or configuration_payload_sha256(canonical) not in set(authorized_hashes)
        ):
            raise ValueError("Authoritative gamma XGBoost config lacks sealed authorization")
    elif gamma_authorization is not None:
        raise ValueError("Core XGBoost fit cannot carry gamma authorization")
    if seed not in DEVELOPMENT_SEEDS:
        raise ValueError("Authoritative Stage 9 XGBoost seed is outside {11, 23, 47}")
    if fold_id not in {"dev_fold_1", "dev_fold_2", "dev_fold_3", "dev_fold_4"}:
        raise ValueError("Authoritative Stage 9 XGBoost fold is outside the frozen folds")
    return canonical


def fit_predict_outer_fold(
    *,
    config: XGBoostConfig,
    fold_id: str,
    seed: int,
    X_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    w_inner_train: np.ndarray,
    X_inner_validation: np.ndarray,
    y_inner_validation: np.ndarray,
    w_inner_validation: np.ndarray,
    X_full_train: np.ndarray,
    y_full_train: np.ndarray,
    w_full_train: np.ndarray,
    X_outer_validation: np.ndarray,
    gamma_authorization: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, XGBoostFitReceipt]:
    """Run one canonical core-grid development fit in its spawn worker."""

    require_spawned_native_worker("xgboost")
    validate_runtime_versions()
    _require_authoritative_xgboost_job(
        config=config,
        seed=seed,
        fold_id=fold_id,
        gamma_authorization=gamma_authorization,
    )
    return _fit_predict_outer_fold_impl(
        config=config,
        fold_id=fold_id,
        seed=seed,
        X_inner_train=X_inner_train,
        y_inner_train=y_inner_train,
        w_inner_train=w_inner_train,
        X_inner_validation=X_inner_validation,
        y_inner_validation=y_inner_validation,
        w_inner_validation=w_inner_validation,
        X_full_train=X_full_train,
        y_full_train=y_full_train,
        w_full_train=w_full_train,
        X_outer_validation=X_outer_validation,
        threads=1,
        max_boosting_rounds=MAX_BOOSTING_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )


def _fit_predict_outer_fold_nonauthoritative_for_test(
    *,
    config: XGBoostConfig,
    fold_id: str,
    seed: int,
    X_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    w_inner_train: np.ndarray,
    X_inner_validation: np.ndarray,
    y_inner_validation: np.ndarray,
    w_inner_validation: np.ndarray,
    X_full_train: np.ndarray,
    y_full_train: np.ndarray,
    w_full_train: np.ndarray,
    X_outer_validation: np.ndarray,
    threads: int = 1,
    max_boosting_rounds: int = 30,
    early_stopping_rounds: int = 5,
) -> tuple[np.ndarray, XGBoostFitReceipt]:
    """Reduced native fit for tests; its receipt is never publishable."""

    require_spawned_native_worker("xgboost")
    return _fit_predict_outer_fold_impl(
        config=config,
        fold_id=f"NONAUTHORITATIVE_TEST::{fold_id}",
        seed=seed,
        X_inner_train=X_inner_train,
        y_inner_train=y_inner_train,
        w_inner_train=w_inner_train,
        X_inner_validation=X_inner_validation,
        y_inner_validation=y_inner_validation,
        w_inner_validation=w_inner_validation,
        X_full_train=X_full_train,
        y_full_train=y_full_train,
        w_full_train=w_full_train,
        X_outer_validation=X_outer_validation,
        threads=threads,
        max_boosting_rounds=max_boosting_rounds,
        early_stopping_rounds=early_stopping_rounds,
    )
