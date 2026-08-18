"""Deterministic PyTorch implementation of the frozen shallow NN selector."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from portfolio_bot.selectors.contract import validate_runtime_versions
from portfolio_bot.selectors.model_specs import (
    DEVELOPMENT_SEEDS,
    NeuralConfig,
    neural_configurations,
    require_canonical_neural_config,
)
from portfolio_bot.selectors.native_process import (
    configure_single_thread_environment,
    require_spawned_native_worker,
)

configure_single_thread_environment()
require_spawned_native_worker("neural_network")

import numpy as np
import torch
from torch import nn


BATCH_SIZE = 4_096
MAX_EPOCHS = 200
PATIENCE = 20
MIN_DELTA = 1e-5
_INTEROP_THREADS_CONFIGURED = False


@dataclass(frozen=True)
class NeuralFitReceipt:
    config_id: str
    fold_id: str
    seed: int
    input_features: int
    inner_input_features: int
    train_rows: int
    inner_train_rows: int
    inner_validation_rows: int
    outer_validation_rows: int
    best_epoch: int
    best_inner_loss: float
    epochs_examined: int
    stopped_early: bool
    parameter_count: int
    inner_parameter_count: int
    elapsed_seconds: float
    device: str
    dtype: str
    deterministic_algorithms: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _seed_runtime(seed: int, *, threads: int) -> None:
    global _INTEROP_THREADS_CONFIGURED
    if threads <= 0:
        raise ValueError("threads must be positive")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if not _INTEROP_THREADS_CONFIGURED:
        torch.set_num_interop_threads(threads)
        _INTEROP_THREADS_CONFIGURED = True
    torch.use_deterministic_algorithms(True)


class ShallowSelectorNetwork(nn.Module):
    def __init__(self, input_features: int, config: NeuralConfig) -> None:
        super().__init__()
        if input_features <= 0:
            raise ValueError("input_features must be positive")
        blocks: list[nn.Module] = []
        previous = input_features
        for width in config.hidden_layers:
            linear = nn.Linear(previous, width, bias=True)
            batch_norm = nn.BatchNorm1d(
                width,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
            nn.init.ones_(batch_norm.weight)
            nn.init.zeros_(batch_norm.bias)
            blocks.extend((linear, batch_norm, nn.ReLU(), nn.Dropout(config.dropout)))
            previous = width
        output = nn.Linear(previous, 1, bias=True)
        blocks.append(output)
        self.layers = nn.Sequential(*blocks)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters() if value.requires_grad))


def _kernel_l1(model: nn.Module) -> torch.Tensor:
    terms = [
        torch.sum(torch.abs(module.weight))
        for module in model.modules()
        if isinstance(module, nn.Linear)
    ]
    if not terms:
        raise AssertionError("A selector network must contain a linear kernel")
    return torch.stack(terms).sum()


def _validate_arrays(
    X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(X, dtype=np.float32, order="C")
    target = np.asarray(y, dtype=np.float32)
    weights = np.asarray(sample_weight, dtype=np.float32)
    if features.ndim != 2 or target.ndim != 1 or weights.ndim != 1:
        raise ValueError("Invalid NN input dimensionality")
    if features.shape[0] != target.size or target.size != weights.size:
        raise ValueError("NN input lengths differ")
    if target.size < 2 or features.shape[1] < 1:
        raise ValueError("NN fit requires at least two rows and one feature")
    if not (
        np.isfinite(features).all()
        and np.isfinite(target).all()
        and np.isfinite(weights).all()
        and np.all(weights > 0)
    ):
        raise ValueError("NN inputs must be finite with positive weights")
    weights = weights * (float(weights.size) / float(np.sum(weights, dtype=np.float64)))
    return features, target, weights.astype(np.float32, copy=False)


def _batch_slices(size: int, batch_size: int) -> list[slice]:
    if batch_size < 2:
        raise ValueError("batch_size must be at least two for BatchNorm")
    starts = list(range(0, size, batch_size))
    slices = [slice(start, min(size, start + batch_size)) for start in starts]
    if len(slices) > 1 and slices[-1].stop - slices[-1].start == 1:
        previous = slices[-2]
        slices[-2] = slice(previous.start, slices[-1].stop)
        slices.pop()
    return slices


def _weighted_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    model: nn.Module,
    l1: float,
) -> torch.Tensor:
    loss = torch.mean(weight * torch.square(prediction - target))
    if l1:
        loss = loss + float(l1) * _kernel_l1(model)
    return loss


def _fit_epochs(
    model: ShallowSelectorNetwork,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    config: NeuralConfig,
    seed: int,
    epochs: int,
    batch_size: int,
) -> None:
    features, target, weights = _validate_arrays(X, y, sample_weight)
    x_tensor = torch.from_numpy(features)
    y_tensor = torch.from_numpy(target)
    w_tensor = torch.from_numpy(weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    slices = _batch_slices(target.size, batch_size)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(target.size, generator=generator)
        for batch_slice in slices:
            index = order[batch_slice]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_tensor[index])
            loss = _weighted_objective(
                prediction,
                y_tensor[index],
                w_tensor[index],
                model=model,
                l1=config.l1,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite NN training loss")
            loss.backward()
            optimizer.step()


@torch.no_grad()
def _evaluate(
    model: ShallowSelectorNetwork,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    l1: float,
    batch_size: int,
) -> float:
    features, target, weights = _validate_arrays(X, y, sample_weight)
    model.eval()
    total = 0.0
    weight_total = 0.0
    x_tensor = torch.from_numpy(features)
    y_tensor = torch.from_numpy(target)
    w_tensor = torch.from_numpy(weights)
    for start in range(0, target.size, batch_size):
        stop = min(target.size, start + batch_size)
        prediction = model(x_tensor[start:stop])
        errors = torch.square(prediction - y_tensor[start:stop])
        total += float(torch.sum(w_tensor[start:stop] * errors).item())
        weight_total += float(torch.sum(w_tensor[start:stop]).item())
    value = total / weight_total
    if l1:
        value += float(l1) * float(_kernel_l1(model).item())
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite NN validation loss")
    return value


def choose_epoch_count(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    w_validation: np.ndarray,
    *,
    config: NeuralConfig,
    seed: int,
    threads: int = 1,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    min_delta: float = MIN_DELTA,
    batch_size: int = BATCH_SIZE,
) -> tuple[int, float, int, bool, int]:
    """Choose the best epoch on the latest-12-month inner block."""

    _seed_runtime(seed, threads=threads)
    features, target, weights = _validate_arrays(X_train, y_train, w_train)
    model = ShallowSelectorNetwork(features.shape[1], config)
    parameter_count = trainable_parameter_count(model)
    x_tensor = torch.from_numpy(features)
    y_tensor = torch.from_numpy(target)
    w_tensor = torch.from_numpy(weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    slices = _batch_slices(target.size, batch_size)
    best_epoch = 0
    best_loss = math.inf
    stale = 0
    examined = 0
    stopped = False
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(target.size, generator=generator)
        for batch_slice in slices:
            index = order[batch_slice]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_tensor[index])
            training_loss = _weighted_objective(
                prediction,
                y_tensor[index],
                w_tensor[index],
                model=model,
                l1=config.l1,
            )
            if not torch.isfinite(training_loss):
                raise FloatingPointError("Nonfinite NN training loss")
            training_loss.backward()
            optimizer.step()
        examined = epoch
        loss = _evaluate(
            model,
            X_validation,
            y_validation,
            w_validation,
            l1=config.l1,
            batch_size=batch_size,
        )
        if best_loss - loss > min_delta:
            best_loss = loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                stopped = True
                break
    if best_epoch <= 0:
        raise RuntimeError("NN early stopping did not select an epoch")
    return best_epoch, best_loss, examined, stopped, parameter_count


@torch.no_grad()
def predict_network(
    model: ShallowSelectorNetwork,
    X: np.ndarray,
    *,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    features = np.asarray(X, dtype=np.float32, order="C")
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Prediction matrix must be finite and two-dimensional")
    model.eval()
    tensor = torch.from_numpy(features)
    pieces = []
    for start in range(0, features.shape[0], batch_size):
        value = model(tensor[start : start + batch_size]).cpu().numpy()
        pieces.append(np.asarray(value, dtype=np.float32))
    result = np.concatenate(pieces)
    if result.shape != (features.shape[0],) or not np.isfinite(result).all():
        raise FloatingPointError("NN emitted invalid predictions")
    return result


def _fit_predict_outer_fold_impl(
    *,
    config: NeuralConfig,
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
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    min_delta: float = MIN_DELTA,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, NeuralFitReceipt]:
    started = time.monotonic()
    best_epoch, best_loss, examined, stopped, inner_parameter_count = choose_epoch_count(
        X_inner_train,
        y_inner_train,
        w_inner_train,
        X_inner_validation,
        y_inner_validation,
        w_inner_validation,
        config=config,
        seed=seed,
        threads=threads,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        batch_size=batch_size,
    )
    _seed_runtime(seed, threads=threads)
    full_input_features = int(np.asarray(X_full_train).shape[1])
    model = ShallowSelectorNetwork(full_input_features, config)
    parameter_count = trainable_parameter_count(model)
    _fit_epochs(
        model,
        X_full_train,
        y_full_train,
        w_full_train,
        config=config,
        seed=seed,
        epochs=best_epoch,
        batch_size=batch_size,
    )
    predictions = predict_network(model, X_outer_validation, batch_size=batch_size)
    receipt = NeuralFitReceipt(
        config_id=config.config_id,
        fold_id=fold_id,
        seed=seed,
        input_features=full_input_features,
        inner_input_features=int(np.asarray(X_inner_train).shape[1]),
        train_rows=int(np.asarray(X_full_train).shape[0]),
        inner_train_rows=int(np.asarray(X_inner_train).shape[0]),
        inner_validation_rows=int(np.asarray(X_inner_validation).shape[0]),
        outer_validation_rows=int(np.asarray(X_outer_validation).shape[0]),
        best_epoch=best_epoch,
        best_inner_loss=best_loss,
        epochs_examined=examined,
        stopped_early=stopped,
        parameter_count=parameter_count,
        inner_parameter_count=inner_parameter_count,
        elapsed_seconds=float(time.monotonic() - started),
        device="cpu",
        dtype="float32",
        deterministic_algorithms=True,
    )
    return predictions, receipt


def fit_predict_outer_fold(
    *,
    config: NeuralConfig,
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
) -> tuple[np.ndarray, NeuralFitReceipt]:
    """Run one canonical development fit in its authoritative spawn worker."""

    require_spawned_native_worker("neural_network")
    validate_runtime_versions()
    require_canonical_neural_config(config)
    if seed not in DEVELOPMENT_SEEDS:
        raise ValueError("Authoritative Stage 9 NN seed is outside {11, 23, 47}")
    if fold_id not in {"dev_fold_1", "dev_fold_2", "dev_fold_3", "dev_fold_4"}:
        raise ValueError("Authoritative Stage 9 NN fold is outside the frozen folds")
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
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        batch_size=BATCH_SIZE,
    )


def _fit_predict_outer_fold_nonauthoritative_for_test(
    *,
    config: NeuralConfig,
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
    max_epochs: int = 3,
    patience: int = 2,
    min_delta: float = 0.0,
    batch_size: int = 16,
) -> tuple[np.ndarray, NeuralFitReceipt]:
    """Reduced native fit for tests; its receipt is never publishable."""

    require_spawned_native_worker("neural_network")
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
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        batch_size=batch_size,
    )
