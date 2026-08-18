"""V6.1 neural selectors and causal multi-model ensembles."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from portfolio_bot.research_v6.models import SelectorMLP, rank_within_month


def _sample_pairs(
    target: np.ndarray,
    months: Sequence[str],
    *,
    rng: np.random.Generator,
    pairs_per_row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = pd.Series(months, dtype="string")
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    sign_parts: list[np.ndarray] = []
    for indices in series.groupby(series, sort=False).groups.values():
        positions = np.asarray(indices, dtype=np.int64)
        if len(positions) < 2:
            continue
        count = pairs_per_row * len(positions)
        left = rng.choice(positions, size=count, replace=True)
        right = rng.choice(positions, size=count, replace=True)
        sign = np.sign(target[left] - target[right])
        keep = sign != 0.0
        left_parts.append(left[keep])
        right_parts.append(right[keep])
        sign_parts.append(sign[keep].astype(np.float32))
    if not left_parts:
        raise ValueError("pairwise MLP sampling produced no distinct-label pairs")
    return np.concatenate(left_parts), np.concatenate(right_parts), np.concatenate(sign_parts)


def fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    months: Sequence[str],
    *,
    objective: str,
    seed: int,
    hidden_layers: Sequence[int],
    dropout: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    pairs_per_row: int,
    gradient_clip: float,
) -> SelectorMLP:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SelectorMLP(x.shape[1], hidden_layers, dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    values = torch.from_numpy(np.asarray(x, dtype=np.float32))
    target = torch.from_numpy(np.asarray(y, dtype=np.float32))
    model.train()
    for _ in range(int(epochs)):
        if objective == "pairwise":
            left, right, sign = _sample_pairs(
                y, months, rng=rng, pairs_per_row=int(pairs_per_row)
            )
            order = rng.permutation(len(left))
            for start in range(0, len(order), int(batch_size)):
                selected = order[start : start + int(batch_size)]
                direction = torch.from_numpy(sign[selected])
                loss = torch.nn.functional.softplus(
                    -direction * (model(values[left[selected]]) - model(values[right[selected]]))
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
                optimizer.step()
        elif objective == "continuous_rank":
            order = rng.permutation(len(values))
            for start in range(0, len(order), int(batch_size)):
                selected = order[start : start + int(batch_size)]
                loss = torch.nn.functional.smooth_l1_loss(
                    model(values[selected]), target[selected], beta=0.10
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
                optimizer.step()
        else:
            raise ValueError(f"unknown MLP objective: {objective}")
    model.eval()
    return model


def predict_mlp(model: SelectorMLP, x: np.ndarray, batch_size: int = 16_384) -> np.ndarray:
    values = torch.from_numpy(np.asarray(x, dtype=np.float32))
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            parts.append(model(values[start : start + batch_size]).numpy())
    return np.concatenate(parts).astype(np.float64)


def _rank_ic(score: np.ndarray, target: np.ndarray) -> float:
    valid = np.isfinite(score) & np.isfinite(target)
    if valid.sum() < 10:
        return math.nan
    left = pd.Series(score[valid]).rank(method="average").to_numpy(float)
    right = pd.Series(target[valid]).rank(method="average").to_numpy(float)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _bounded_weights(
    quality: np.ndarray, minimum: float, maximum: float
) -> np.ndarray:
    count = len(quality)
    if count == 0 or minimum * count > 1.0 or maximum * count < 1.0:
        raise ValueError("ensemble weight bounds are infeasible")
    positive = np.maximum(np.asarray(quality, dtype=float), 0.0)
    if positive.sum() <= 0.0:
        positive = np.ones(count, dtype=float)
    share = positive / positive.sum()
    weights = np.full(count, minimum, dtype=float)
    remaining = 1.0 - minimum * count
    active = np.ones(count, dtype=bool)
    while remaining > 1e-12 and active.any():
        active_share = share[active]
        if active_share.sum() <= 0.0:
            active_share = np.ones(active.sum(), dtype=float)
        addition = remaining * active_share / active_share.sum()
        positions = np.flatnonzero(active)
        capacity = maximum - weights[positions]
        applied = np.minimum(addition, capacity)
        weights[positions] += applied
        remaining -= float(applied.sum())
        active[positions[capacity - applied <= 1e-12]] = False
        if np.all(applied <= 1e-15):
            break
    return weights / weights.sum()


def add_equal_ensemble(
    frame: pd.DataFrame,
    *,
    components: Sequence[str],
    output_column: str,
) -> pd.DataFrame:
    result = frame.copy()
    blended = result.loc[:, list(components)].mean(axis=1).to_numpy(float)
    result[output_column] = rank_within_month(blended, result["holding_month"])
    return result


def add_dynamic_ensemble(
    frame: pd.DataFrame,
    *,
    components: Sequence[str],
    target_column: str,
    output_column: str,
    half_life_months: float,
    minimum_history_months: int,
    minimum_component_weight: float,
    maximum_component_weight: float,
) -> pd.DataFrame:
    """Blend any number of components using strictly prior monthly rank IC."""

    required = {"holding_month", "permno", target_column, *components}
    if not required.issubset(frame.columns):
        raise ValueError("dynamic ensemble input omits required columns")
    result = frame.sort_values(["holding_month", "permno"], kind="mergesort").copy()
    history: list[np.ndarray] = []
    monthly_weights: list[dict[str, float | str]] = []
    decay = math.log(2.0) / float(half_life_months)
    for month, group in result.groupby("holding_month", observed=True, sort=True):
        finite_history = [values for values in history if np.isfinite(values).all()]
        if len(finite_history) < int(minimum_history_months):
            weights = np.full(len(components), 1.0 / len(components))
        else:
            ages = np.arange(len(finite_history) - 1, -1, -1, dtype=float)
            time_weight = np.exp(-decay * ages)
            quality = np.average(np.vstack(finite_history), axis=0, weights=time_weight)
            weights = _bounded_weights(
                np.maximum(quality, 0.0),
                float(minimum_component_weight),
                float(maximum_component_weight),
            )
        row: dict[str, float | str] = {"holding_month": str(month)}
        for component, weight in zip(components, weights):
            row[f"{output_column}__weight__{component}"] = float(weight)
        monthly_weights.append(row)
        target = group[target_column].to_numpy(float)
        history.append(
            np.asarray(
                [_rank_ic(group[component].to_numpy(float), target) for component in components]
            )
        )
    weights = pd.DataFrame(monthly_weights)
    result = result.merge(weights, on="holding_month", validate="many_to_one")
    blended = np.zeros(len(result), dtype=float)
    for component in components:
        blended += (
            result[f"{output_column}__weight__{component}"].to_numpy(float)
            * result[component].to_numpy(float)
        )
    result[output_column] = rank_within_month(blended, result["holding_month"])
    return result


__all__ = [
    "add_dynamic_ensemble",
    "add_equal_ensemble",
    "fit_mlp",
    "predict_mlp",
]
