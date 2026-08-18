"""Lean pairwise selectors and causal dynamic ensemble for V6."""

from __future__ import annotations

import dataclasses
import math
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import ElasticNet
from xgboost import XGBRanker, XGBRegressor


def _group_sizes(months: Sequence[str]) -> list[int]:
    series = pd.Series(months, dtype="string")
    if not series.is_monotonic_increasing:
        raise ValueError("ranking rows must be sorted by holding month")
    return series.groupby(series, sort=False).size().astype(int).tolist()


def rank_within_month(raw: Sequence[float], months: Sequence[str]) -> np.ndarray:
    frame = pd.DataFrame({"month": list(months), "raw": np.asarray(raw, dtype=float)})
    if not np.isfinite(frame["raw"]).all():
        raise ValueError("selector predictions must be finite")
    count = frame.groupby("month", observed=True)["raw"].transform("size")
    rank = frame.groupby("month", observed=True)["raw"].rank(method="average")
    return ((rank - 0.5) / count).to_numpy(np.float64)


@dataclasses.dataclass(frozen=True, slots=True)
class SelectorPrediction:
    model_id: str
    raw_score: np.ndarray


def fit_xgboost_ranker(
    x: np.ndarray,
    y: np.ndarray,
    months: Sequence[str],
    *,
    seed: int = 20260811,
    **parameters: float | int | str,
) -> XGBRanker:
    defaults: dict[str, float | int | str] = dict(
        objective="rank:pairwise",
        eval_metric="ndcg",
        n_estimators=180,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=20.0,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.02,
        reg_lambda=2.0,
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
    )
    defaults.update(parameters)
    model = XGBRanker(**defaults)
    model.fit(x, y, group=_group_sizes(months), verbose=False)
    return model


def fit_xgboost_pointwise(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 20260811,
    **parameters: float | int | str,
) -> XGBRegressor:
    """Fit the continuous-rank control with a robust pointwise loss."""

    defaults: dict[str, float | int | str] = dict(
        objective="reg:pseudohubererror",
        eval_metric="mae",
        n_estimators=180,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=20.0,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.02,
        reg_lambda=2.0,
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
    )
    defaults.update(parameters)
    model = XGBRegressor(**defaults)
    model.fit(x, y, verbose=False)
    return model


def fit_elastic_net(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 20260811,
    alpha: float = 0.001,
    l1_ratio: float = 0.50,
) -> ElasticNet:
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=3_000,
        tol=1e-4,
        selection="random",
        random_state=seed,
    )
    model.fit(x, y)
    return model


class SelectorMLP(torch.nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_layers: Sequence[int] = (128, 64),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        previous = input_features
        for width in hidden_layers:
            layers.extend(
                (
                    torch.nn.Linear(previous, int(width)),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(float(dropout)),
                )
            )
            previous = int(width)
        layers.append(torch.nn.Linear(previous, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def _sample_pairs(
    y: np.ndarray,
    months: Sequence[str],
    *,
    rng: np.random.Generator,
    pairs_per_row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    month_series = pd.Series(months, dtype="string")
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    sign_parts: list[np.ndarray] = []
    for indices in month_series.groupby(month_series, sort=False).groups.values():
        positions = np.asarray(indices, dtype=np.int64)
        if len(positions) < 2:
            continue
        draw_count = pairs_per_row * len(positions)
        left = rng.choice(positions, size=draw_count, replace=True)
        right = rng.choice(positions, size=draw_count, replace=True)
        sign = np.sign(y[left] - y[right])
        keep = sign != 0.0
        left_parts.append(left[keep])
        right_parts.append(right[keep])
        sign_parts.append(sign[keep].astype(np.float32))
    if not left_parts:
        raise ValueError("pairwise training produced no distinct-label pairs")
    return np.concatenate(left_parts), np.concatenate(right_parts), np.concatenate(sign_parts)


def fit_pairwise_mlp(
    x: np.ndarray,
    y: np.ndarray,
    months: Sequence[str],
    *,
    seed: int = 20260811,
    epochs: int = 6,
    pairs_per_row: int = 2,
    batch_size: int = 8_192,
    hidden_layers: Sequence[int] = (128, 64),
    dropout: float = 0.10,
) -> SelectorMLP:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SelectorMLP(x.shape[1], hidden_layers, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    values = torch.from_numpy(np.asarray(x, dtype=np.float32))
    model.train()
    for _ in range(epochs):
        left, right, sign = _sample_pairs(
            y, months, rng=rng, pairs_per_row=pairs_per_row
        )
        order = rng.permutation(len(left))
        for start in range(0, len(order), batch_size):
            selection = order[start : start + batch_size]
            left_score = model(values[left[selection]])
            right_score = model(values[right[selection]])
            direction = torch.from_numpy(sign[selection])
            loss = torch.nn.functional.softplus(
                -direction * (left_score - right_score)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    return model


def fit_pointwise_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 20260811,
    epochs: int = 6,
    batch_size: int = 8_192,
    hidden_layers: Sequence[int] = (128, 64),
    dropout: float = 0.10,
) -> SelectorMLP:
    """Fit a robust continuous centered-rank neural selector."""

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SelectorMLP(x.shape[1], hidden_layers, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    values = torch.from_numpy(np.asarray(x, dtype=np.float32))
    target = torch.from_numpy(np.asarray(y, dtype=np.float32))
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(values))
        for start in range(0, len(order), batch_size):
            selection = order[start : start + batch_size]
            loss = torch.nn.functional.smooth_l1_loss(
                model(values[selection]), target[selection], beta=0.10
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    return model


def predict_mlp(model: SelectorMLP, x: np.ndarray, *, batch_size: int = 16_384) -> np.ndarray:
    values = torch.from_numpy(np.asarray(x, dtype=np.float32))
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            parts.append(model(values[start : start + batch_size]).numpy())
    return np.concatenate(parts).astype(np.float64)


def _rank_ic(score: np.ndarray, target: np.ndarray) -> float:
    valid = np.isfinite(score) & np.isfinite(target)
    if valid.sum() < 3:
        return math.nan
    left = pd.Series(score[valid]).rank(method="average").to_numpy(float)
    right = pd.Series(target[valid]).rank(method="average").to_numpy(float)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def dynamic_ensemble(
    predictions: pd.DataFrame,
    *,
    xgb_column: str = "xgboost_rank",
    mlp_column: str = "mlp_rank",
    target_column: str,
    half_life_months: float = 12.0,
    minimum_history_months: int = 24,
    minimum_weight: float = 0.20,
) -> pd.DataFrame:
    """Weight XGBoost and MLP using only prior realized monthly rank IC."""

    required = {"holding_month", xgb_column, mlp_column, target_column}
    if not required.issubset(predictions.columns):
        raise ValueError("ensemble input omits required columns")
    result = predictions.sort_values(
        ["holding_month", "permno"], kind="mergesort"
    ).copy()
    monthly: list[tuple[str, float, float]] = []
    for month, group in result.groupby("holding_month", observed=True, sort=True):
        monthly.append(
            (
                str(month),
                _rank_ic(group[xgb_column].to_numpy(float), group[target_column].to_numpy(float)),
                _rank_ic(group[mlp_column].to_numpy(float), group[target_column].to_numpy(float)),
            )
        )
    histories: list[tuple[float, float]] = []
    weight_rows: list[dict[str, float | str]] = []
    decay = math.log(2.0) / half_life_months
    for month, xgb_ic, mlp_ic in monthly:
        finite_history = [(x, m) for x, m in histories if np.isfinite(x) and np.isfinite(m)]
        if len(finite_history) < minimum_history_months:
            xgb_weight = 0.5
        else:
            ages = np.arange(len(finite_history) - 1, -1, -1, dtype=float)
            weights = np.exp(-decay * ages)
            x_quality = float(np.average([x for x, _ in finite_history], weights=weights))
            m_quality = float(np.average([m for _, m in finite_history], weights=weights))
            quality = np.maximum(np.asarray([x_quality, m_quality]), 0.0)
            if quality.sum() <= 0.0:
                xgb_weight = 0.5
            else:
                share = float(quality[0] / quality.sum())
                xgb_weight = minimum_weight + (1.0 - 2.0 * minimum_weight) * share
        weight_rows.append(
            {
                "holding_month": month,
                "xgboost_weight": xgb_weight,
                "mlp_weight": 1.0 - xgb_weight,
            }
        )
        histories.append((xgb_ic, mlp_ic))
    result = result.merge(pd.DataFrame(weight_rows), on="holding_month", validate="many_to_one")
    blended = (
        result["xgboost_weight"].to_numpy(float) * result[xgb_column].to_numpy(float)
        + result["mlp_weight"].to_numpy(float) * result[mlp_column].to_numpy(float)
    )
    result["ensemble_rank"] = rank_within_month(blended, result["holding_month"])
    return result


__all__ = [
    "SelectorMLP",
    "dynamic_ensemble",
    "fit_elastic_net",
    "fit_pairwise_mlp",
    "fit_pointwise_mlp",
    "fit_xgboost_pointwise",
    "fit_xgboost_ranker",
    "predict_mlp",
    "rank_within_month",
]
