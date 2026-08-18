"""Causal signal builders for the six V6.3 signal-to-weight experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from portfolio_bot.research_v6.models import rank_within_month


KEYS = ["holding_month", "signal_date", "permno", "permco"]
TARGET_COLUMNS = {
    "long_net": "target_rank_long_net",
    "short_net": "target_rank_short_net",
    "ff5_momentum": "target_rank_ff5_momentum",
}


def _percentile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.any():
        result.loc[valid] = numeric.loc[valid].rank(method="average", pct=True)
    return result


def add_confidence_diagnostics(
    frame: pd.DataFrame,
    *,
    base_column: str,
    component_columns: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    """Add outcome-blind component disagreement and one-month instability.

    Both diagnostics are converted to within-month percentiles so they remain
    comparable across model families and changing universe breadth.
    """

    required = {"holding_month", "permno", base_column, *component_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"confidence input omits columns: {sorted(missing)}")
    result = frame.sort_values(["holding_month", "permno"], kind="mergesort").copy()
    components = result.loc[:, list(component_columns)].apply(pd.to_numeric, errors="coerce")
    disagreement = components.max(axis=1) - components.min(axis=1)
    result[f"{prefix}_component_disagreement"] = disagreement
    result[f"{prefix}_component_disagreement_pct"] = result.groupby(
        "holding_month", observed=True, sort=False
    )[f"{prefix}_component_disagreement"].transform(_percentile)

    month_number = pd.PeriodIndex(result["holding_month"].astype(str), freq="M").asi8
    result["_month_number"] = month_number
    prior = result[["permno", "_month_number", base_column]].copy()
    prior["_month_number"] += 1
    prior = prior.rename(columns={base_column: "_prior_score"})
    result = result.merge(
        prior,
        on=["permno", "_month_number"],
        how="left",
        validate="one_to_one",
    )
    instability = (
        pd.to_numeric(result[base_column], errors="coerce")
        - pd.to_numeric(result["_prior_score"], errors="coerce")
    ).abs()
    result[f"{prefix}_instability"] = instability
    result[f"{prefix}_instability_pct"] = result.groupby(
        "holding_month", observed=True, sort=False
    )[f"{prefix}_instability"].transform(_percentile)
    result[f"{prefix}_uncertainty"] = (
        0.70 * result[f"{prefix}_component_disagreement_pct"].fillna(0.50)
        + 0.30 * result[f"{prefix}_instability_pct"].fillna(0.50)
    ).clip(0.0, 1.0)
    return result.drop(columns=["_month_number", "_prior_score"])


def build_three_month_target(rows: pd.DataFrame) -> pd.DataFrame:
    """Build a consecutive three-month excess-return rank and availability date."""

    required = {
        "holding_month",
        "permno",
        "unranked_excess_return",
        "label_available_date",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"three-month target input omits: {sorted(missing)}")
    result = rows[["holding_month", "permno", "unranked_excess_return", "label_available_date"]].copy()
    result["holding_month"] = result["holding_month"].astype(str)
    result["_ordinal"] = pd.PeriodIndex(result["holding_month"], freq="M").asi8
    result["unranked_excess_return"] = pd.to_numeric(
        result["unranked_excess_return"], errors="coerce"
    )
    result["label_available_date"] = pd.to_datetime(
        result["label_available_date"], errors="coerce"
    )
    lookup = result.set_index(["permno", "_ordinal"])[
        ["unranked_excess_return", "label_available_date"]
    ]
    values: list[pd.DataFrame] = []
    base = result[["permno", "_ordinal"]].copy()
    for lead in range(3):
        keys = pd.MultiIndex.from_arrays(
            [base["permno"].to_numpy(), base["_ordinal"].to_numpy() + lead]
        )
        joined = lookup.reindex(keys).reset_index(drop=True)
        joined.columns = [f"return_{lead}", f"available_{lead}"]
        values.append(joined)
    joined = pd.concat([result.reset_index(drop=True), *values], axis=1)
    consecutive = np.ones(len(joined), dtype=bool)
    compounded = np.ones(len(joined), dtype=float)
    availability = pd.Series(pd.NaT, index=joined.index, dtype="datetime64[ns]")
    for lead in range(3):
        current_return = pd.to_numeric(joined[f"return_{lead}"], errors="coerce")
        current_available = pd.to_datetime(joined[f"available_{lead}"], errors="coerce")
        consecutive &= current_return.notna().to_numpy() & current_available.notna().to_numpy()
        compounded *= 1.0 + current_return.fillna(0.0).to_numpy(float)
        availability = pd.concat([availability, current_available], axis=1).max(axis=1)
    joined["target_return_3m"] = np.where(consecutive, compounded - 1.0, np.nan)
    joined["target_available_date_3m"] = availability.where(consecutive)
    def centered_rank(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        valid = numeric.notna()
        ranked = pd.Series(np.nan, index=values.index, dtype=float)
        count = int(valid.sum())
        if count:
            ranks = numeric.loc[valid].rank(method="average")
            ranked.loc[valid] = (ranks - 0.5) / count - 0.5
        return ranked

    joined["target_rank_return_3m"] = joined.groupby(
        "holding_month", observed=True, sort=False
    )["target_return_3m"].transform(centered_rank)
    return joined[
        [
            "holding_month",
            "permno",
            "target_return_3m",
            "target_rank_return_3m",
            "target_available_date_3m",
        ]
    ]


def feature_expert_indices(
    metadata: Mapping[str, Any], experts: Mapping[str, Sequence[str]]
) -> dict[str, np.ndarray]:
    """Map every encoded feature to one and only one economic expert block."""

    families = dict(metadata["feature_family"])
    sources = list(metadata["source_features"])
    family_to_expert: dict[str, str] = {}
    for expert, expert_families in experts.items():
        for family in expert_families:
            if family in family_to_expert:
                raise ValueError(f"feature family assigned twice: {family}")
            family_to_expert[str(family)] = str(expert)
    unknown = sorted(set(families.values()).difference(family_to_expert))
    if unknown:
        raise ValueError(f"expert map omits feature families: {unknown}")
    result: dict[str, list[int]] = {str(expert): [] for expert in experts}
    for index, source in enumerate(sources):
        result[family_to_expert[str(families[source])]].append(index)
    arrays = {name: np.asarray(indices, dtype=np.int64) for name, indices in result.items()}
    covered = np.concatenate(list(arrays.values()))
    if sorted(covered.tolist()) != list(range(len(sources))):
        raise RuntimeError("expert features do not partition the encoded matrix")
    return arrays


def feature_group_coverage(
    matrix: np.ndarray,
    metadata: Mapping[str, Any],
    expert_indices: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return row-level observed shares from explicit source missingness flags."""

    names = list(metadata["feature_names"])
    result: dict[str, np.ndarray] = {}
    for expert, indices in expert_indices.items():
        missing = np.asarray(
            [index for index in indices if names[int(index)].endswith("__missing")],
            dtype=np.int64,
        )
        if not len(missing):
            result[expert] = np.ones(matrix.shape[0], dtype=np.float32)
            continue
        flags = np.asarray(matrix[:, missing], dtype=np.float32)
        result[expert] = np.clip(1.0 - flags.mean(axis=1), 0.0, 1.0)
    return result


def _rank_ic(score: np.ndarray, target: np.ndarray) -> float:
    valid = np.isfinite(score) & np.isfinite(target)
    if valid.sum() < 20:
        return math.nan
    left = pd.Series(score[valid]).rank(method="average").to_numpy(float)
    right = pd.Series(target[valid]).rank(method="average").to_numpy(float)
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def causal_specialist_mixture(
    predictions: pd.DataFrame,
    rows: pd.DataFrame,
    coverage: Mapping[str, np.ndarray],
    *,
    components: Sequence[str],
    target_column: str,
    temperature: float,
    minimum_weight: float,
    history_months: int,
) -> pd.DataFrame:
    """Blend specialist ranks using strictly available trailing IC and coverage."""

    if temperature <= 0.0:
        raise ValueError("specialist temperature must be positive")
    if minimum_weight < 0.0 or minimum_weight * len(components) >= 1.0:
        raise ValueError("specialist minimum weight is infeasible")
    result = predictions.sort_values(["holding_month", "permno"], kind="mergesort").copy()
    aligned = rows[[*KEYS, target_column, "label_available_date"]].copy()
    aligned["holding_month"] = aligned["holding_month"].astype(str)
    aligned["signal_date"] = pd.to_datetime(aligned["signal_date"], errors="raise")
    aligned["label_available_date"] = pd.to_datetime(
        aligned["label_available_date"], errors="coerce"
    )
    result = result.merge(aligned, on=KEYS, how="left", validate="one_to_one")
    # Build trailing IC from completed prior months directly. The loop above
    # cannot observe a forecast month's target; compute historical monthly ICs
    # once, then look them up only when their availability predates the signal.
    monthly_ic: list[dict[str, Any]] = []
    for month, group in result.groupby("holding_month", observed=True, sort=True):
        row: dict[str, Any] = {
            "holding_month": str(month),
            "available": pd.to_datetime(group["label_available_date"], errors="coerce").max(),
        }
        for component in components:
            row[component] = _rank_ic(
                group[component].to_numpy(float),
                pd.to_numeric(group[target_column], errors="coerce").to_numpy(float),
            )
        monthly_ic.append(row)
    ic_frame = pd.DataFrame(monthly_ic)

    blended = np.zeros(len(result), dtype=float)
    weight_records: list[dict[str, Any]] = []
    for month, group in result.groupby("holding_month", observed=True, sort=True):
        signal_date = pd.Timestamp(group["signal_date"].iloc[0])
        history = ic_frame.loc[
            ic_frame["holding_month"].lt(str(month))
            & pd.to_datetime(ic_frame["available"], errors="coerce").le(signal_date)
        ].tail(int(history_months))
        if len(history) < 12:
            quality = np.zeros(len(components), dtype=float)
        else:
            values = history.loc[:, list(components)].to_numpy(float)
            count = np.isfinite(values).sum(axis=0)
            quality = np.divide(
                np.nansum(values, axis=0),
                count,
                out=np.zeros(len(components), dtype=float),
                where=count > 0,
            )
        logits = np.maximum(quality, 0.0) / float(temperature)
        logits -= logits.max(initial=0.0)
        weights = np.exp(logits)
        weights /= weights.sum()
        weights = minimum_weight + (1.0 - minimum_weight * len(components)) * weights
        stock_weights = np.zeros((len(group), len(components)), dtype=float)
        for position, component in enumerate(components):
            cov = np.asarray(coverage[component], dtype=float)[group.index.to_numpy()]
            stock_weights[:, position] = weights[position] * (0.25 + 0.75 * cov)
        stock_weights /= stock_weights.sum(axis=1, keepdims=True)
        score = np.sum(
            stock_weights
            * group.loc[:, list(components)].to_numpy(float),
            axis=1,
        )
        blended[group.index.to_numpy()] = score
        record: dict[str, Any] = {"holding_month": str(month)}
        for component, weight, quality_value in zip(components, weights, quality):
            record[f"weight_{component}"] = float(weight)
            record[f"trailing_ic_{component}"] = float(quality_value)
        weight_records.append(record)
    result["score_rank"] = rank_within_month(blended, result["holding_month"])
    result.attrs["monthly_weights"] = pd.DataFrame(weight_records)
    return result


class MultiTaskSelector(torch.nn.Module):
    """Shared representation with rank, horizon, and tail-probability heads."""

    def __init__(
        self,
        input_features: int,
        hidden_layers: Sequence[int] = (96, 48),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        previous = int(input_features)
        for width in hidden_layers:
            layers.extend(
                [
                    torch.nn.Linear(previous, int(width)),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(float(dropout)),
                ]
            )
            previous = int(width)
        self.trunk = torch.nn.Sequential(*layers)
        self.rank_head = torch.nn.Linear(previous, 4)
        self.tail_head = torch.nn.Linear(previous, 3)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(values)
        return self.rank_head(hidden), self.tail_head(hidden)


@dataclass(frozen=True, slots=True)
class MultiTaskPrediction:
    ranks: np.ndarray
    tail_probabilities: np.ndarray


def fit_multitask_selector(
    x: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    hidden_layers: Sequence[int],
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> MultiTaskSelector:
    """Fit masked Huber rank heads plus explicitly tail-aware BCE heads."""

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model = MultiTaskSelector(x.shape[1], hidden_layers, dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    values = torch.from_numpy(np.asarray(x, dtype=np.float32))
    target_values = torch.from_numpy(np.nan_to_num(targets, nan=0.0).astype(np.float32))
    valid = torch.from_numpy(np.isfinite(targets))
    model.train()
    for _ in range(int(epochs)):
        order = rng.permutation(len(values))
        for start in range(0, len(order), int(batch_size)):
            selected = order[start : start + int(batch_size)]
            rank_prediction, tail_logits = model(values[selected])
            batch_target = target_values[selected]
            batch_valid = valid[selected]
            rank_loss = torch.zeros((), dtype=torch.float32)
            used = 0
            for head in range(4):
                mask = batch_valid[:, head]
                if bool(mask.any()):
                    rank_loss = rank_loss + torch.nn.functional.smooth_l1_loss(
                        rank_prediction[mask, head],
                        batch_target[mask, head],
                        beta=0.10,
                    )
                    used += 1
            tail_loss = torch.zeros((), dtype=torch.float32)
            tail_used = 0
            for head in range(3):
                mask = batch_valid[:, head]
                if bool(mask.any()):
                    label = (batch_target[mask, head] >= 0.30).to(torch.float32)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        tail_logits[mask, head], label, reduction="none"
                    )
                    tail_loss = tail_loss + (loss * (1.0 + 3.0 * label)).mean()
                    tail_used += 1
            if used == 0:
                continue
            loss = rank_loss / used
            if tail_used:
                loss = loss + 0.25 * tail_loss / tail_used
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    return model


def predict_multitask_selector(
    model: MultiTaskSelector, x: np.ndarray, *, batch_size: int = 16_384
) -> MultiTaskPrediction:
    values = torch.from_numpy(np.asarray(x, dtype=np.float32))
    ranks: list[np.ndarray] = []
    tails: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), int(batch_size)):
            rank, logits = model(values[start : start + int(batch_size)])
            ranks.append(rank.numpy())
            tails.append(torch.sigmoid(logits).numpy())
    return MultiTaskPrediction(
        ranks=np.vstack(ranks).astype(np.float64),
        tail_probabilities=np.vstack(tails).astype(np.float64),
    )


def compose_multitask_scores(
    prediction: pd.DataFrame, *, horizon_blend: float
) -> pd.DataFrame:
    """Translate multitask outputs into monthly long, short, and alpha ranks."""

    blend = float(horizon_blend)
    if blend < 0.0 or blend > 1.0:
        raise ValueError("horizon blend must lie in [0, 1]")
    result = prediction.copy()
    one_long = result["rank_head_long"].to_numpy(float)
    one_short = result["rank_head_short"].to_numpy(float)
    one_ff5 = result["rank_head_ff5"].to_numpy(float)
    horizon = result["rank_head_3m"].to_numpy(float)
    result["score_long_net"] = rank_within_month(
        (1.0 - blend) * one_long + blend * horizon,
        result["holding_month"],
    )
    result["score_short_net"] = rank_within_month(
        (1.0 - blend) * one_short - blend * horizon,
        result["holding_month"],
    )
    result["score_ff5_momentum"] = rank_within_month(
        one_ff5, result["holding_month"]
    )
    return result


__all__ = [
    "KEYS",
    "TARGET_COLUMNS",
    "MultiTaskPrediction",
    "MultiTaskSelector",
    "add_confidence_diagnostics",
    "build_three_month_target",
    "causal_specialist_mixture",
    "compose_multitask_scores",
    "feature_expert_indices",
    "feature_group_coverage",
    "fit_multitask_selector",
    "predict_multitask_selector",
]
