"""Efficient, development-only evaluation of Stage 9 OOF forecasts.

The public primitives in :mod:`portfolio_bot.selectors.metrics` define the
scientific statistics.  This module applies those rules to dense prediction
cubes without first expanding hundreds of configurations into tens of
millions of duplicate key rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from portfolio_bot.selectors.metrics import (
    BOOTSTRAP_MASTER_SEED,
    BOOTSTRAP_REPLICATES,
    DEVELOPMENT_FOLD_IDS,
    DEVELOPMENT_SEEDS,
    aggregate_selector_metrics,
    make_moving_block_bootstrap_plan,
)


OOF_KEY_COLUMNS = (
    "fold_id",
    "signal_date",
    "holding_month",
    "permno",
)


@dataclass(frozen=True)
class FamilyEvaluation:
    """Compact Stage 9 evaluation for one selector family."""

    model_family: str
    config_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    raw_forecast_mean: np.ndarray
    monthly_scores: np.ndarray
    monthly_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    config_metrics: pd.DataFrame
    bootstrap_months: tuple[object, ...]
    bootstrap_indices: np.ndarray


def _validate_oof_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        *OOF_KEY_COLUMNS,
        "unranked_residual_return",
        "outcome_valid",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"OOF rows are missing required columns: {missing}")
    frame = rows.reset_index(drop=True).copy()
    if frame.empty or frame.duplicated(list(OOF_KEY_COLUMNS)).any():
        raise ValueError("OOF row keys must be nonempty and unique")
    if frame[list(OOF_KEY_COLUMNS)].isna().any().any():
        raise ValueError("OOF row keys may not be null")
    actual_folds = tuple(sorted(frame["fold_id"].astype(str).unique()))
    if actual_folds != tuple(sorted(DEVELOPMENT_FOLD_IDS)):
        raise ValueError("OOF rows do not contain the four frozen folds")
    if frame["outcome_valid"].isna().any():
        raise ValueError("outcome_valid may not be null")
    outcome_valid = frame["outcome_valid"].astype(bool).to_numpy()
    residual = pd.to_numeric(
        frame["unranked_residual_return"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(residual[outcome_valid]).all():
        raise ValueError("Valid OOF residual returns must be finite")
    if np.isfinite(residual[~outcome_valid]).any():
        raise ValueError("Invalid OOF outcomes must not expose a residual value")
    frame["fold_id"] = frame["fold_id"].astype(str)
    frame["outcome_valid"] = outcome_valid
    frame["unranked_residual_return"] = residual
    return frame


def _validate_predictions(
    predictions: np.ndarray,
    *,
    row_count: int,
    config_count: int,
    seeds: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float32)
    expected = (
        (row_count, config_count)
        if not seeds
        else (row_count, config_count, len(seeds))
    )
    if values.shape != expected:
        raise ValueError(
            f"Prediction cube shape differs: expected {expected}, found {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Every OOF forecast must be finite")
    return values


def _center_monthly(raw_mean: np.ndarray, month_positions: Sequence[np.ndarray]) -> np.ndarray:
    scores = np.empty(raw_mean.shape, dtype=np.float32)
    for positions in month_positions:
        block = np.asarray(raw_mean[positions], dtype=np.float64)
        ranks = rankdata(block, method="average", axis=0)
        centered = ((ranks - 0.5) / float(len(positions))) - 0.5
        scores[positions] = centered.astype(np.float32)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Monthly score construction emitted a nonfinite value")
    return scores


def _columnwise_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    spearman: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return correlations and frozen status strings for all columns."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("Correlation matrix dimensions differ")
    if x.shape[0] < 2:
        return (
            np.zeros(x.shape[1], dtype=np.float64),
            np.full(x.shape[1], "INSUFFICIENT_OBSERVATIONS", dtype=object),
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Correlation inputs must be finite")
    if spearman:
        x = rankdata(x, method="average", axis=0)
        y = rankdata(y, method="average")
    x -= np.mean(x, axis=0, dtype=np.float64)
    y -= float(np.mean(y, dtype=np.float64))
    numerator = x.T @ y
    x_ss = np.sum(np.square(x), axis=0, dtype=np.float64)
    y_ss = float(np.dot(y, y))
    values = np.zeros(x.shape[1], dtype=np.float64)
    statuses = np.full(x.shape[1], "OK", dtype=object)
    x_constant = x_ss == 0.0
    y_constant = y_ss == 0.0
    usable = np.zeros_like(x_constant, dtype=bool) if y_constant else ~x_constant
    values[usable] = numerator[usable] / np.sqrt(x_ss[usable] * y_ss)
    values = np.clip(values, -1.0, 1.0)
    if y_constant:
        statuses[:] = "CONSTANT_RIGHT"
        statuses[x_constant] = "CONSTANT_BOTH"
    else:
        statuses[x_constant] = "CONSTANT_LEFT"
    if not np.isfinite(values).all():
        raise FloatingPointError("Correlation calculation emitted a nonfinite value")
    return values, statuses


def evaluate_family_predictions(
    rows: pd.DataFrame,
    *,
    model_family: str,
    config_ids: Sequence[str],
    predictions: np.ndarray,
    seeds: Sequence[int] = (),
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_MASTER_SEED,
) -> FamilyEvaluation:
    """Evaluate one complete family of genuine Stage 9 OOF forecasts.

    Stochastic predictions have shape ``rows x configurations x seeds``.
    Elastic Net predictions have shape ``rows x configurations`` and an empty
    seed tuple.  Seed-specific raw forecasts are always averaged before any
    monthly percentile or IC is calculated.
    """

    if not isinstance(model_family, str) or not model_family:
        raise ValueError("model_family must be a nonempty string")
    configs = tuple(str(value) for value in config_ids)
    if not configs or len(configs) != len(set(configs)):
        raise ValueError("config_ids must be nonempty and unique")
    seed_tuple = tuple(int(value) for value in seeds)
    if seed_tuple and seed_tuple != DEVELOPMENT_SEEDS:
        raise ValueError("Stochastic Stage 9 evaluation requires seeds 11, 23, 47")
    frame = _validate_oof_rows(rows)
    values = _validate_predictions(
        predictions,
        row_count=len(frame),
        config_count=len(configs),
        seeds=seed_tuple,
    )
    raw_mean = (
        np.mean(values, axis=2, dtype=np.float64).astype(np.float32)
        if seed_tuple
        else np.asarray(values, dtype=np.float32)
    )

    month_groups = frame.groupby("holding_month", sort=True, observed=True).indices
    ordered_months = tuple(month_groups)
    month_positions = tuple(
        np.asarray(month_groups[month], dtype=np.int64) for month in ordered_months
    )
    scores = _center_monthly(raw_mean, month_positions)
    residual = frame["unranked_residual_return"].to_numpy(dtype=np.float64)
    valid_outcome = frame["outcome_valid"].to_numpy(dtype=bool)

    records: list[dict[str, object]] = []
    for month, positions in zip(ordered_months, month_positions, strict=True):
        month_frame = frame.iloc[positions]
        fold_values = month_frame["fold_id"].unique()
        signal_values = month_frame["signal_date"].unique()
        if len(fold_values) != 1 or len(signal_values) != 1:
            raise ValueError("Each OOF holding month must map to one fold and signal date")
        evaluation_mask = valid_outcome[positions]
        evaluation_positions = positions[evaluation_mask]
        rank_ic, rank_status = _columnwise_correlation(
            scores[evaluation_positions],
            residual[evaluation_positions],
            spearman=True,
        )
        linear_ic, linear_status = _columnwise_correlation(
            raw_mean[evaluation_positions],
            residual[evaluation_positions],
            spearman=False,
        )
        for index, config_id in enumerate(configs):
            records.append(
                {
                    "model_family": model_family,
                    "config_id": config_id,
                    "fold_id": str(fold_values[0]),
                    "signal_date": signal_values[0],
                    "holding_month": month,
                    "scored_rows": int(len(positions)),
                    "evaluation_rows": int(len(evaluation_positions)),
                    "excluded_outcome_rows": int(len(positions) - len(evaluation_positions)),
                    "rank_ic": float(rank_ic[index]),
                    "rank_ic_status": str(rank_status[index]),
                    "linear_ic": float(linear_ic[index]),
                    "linear_ic_status": str(linear_status[index]),
                }
            )
    monthly = pd.DataFrame.from_records(records).sort_values(
        ["model_family", "config_id", "holding_month"], kind="mergesort"
    ).reset_index(drop=True)
    plan = make_moving_block_bootstrap_plan(
        ordered_months,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    fold_metrics, config_metrics = aggregate_selector_metrics(monthly, plan=plan)
    return FamilyEvaluation(
        model_family=model_family,
        config_ids=configs,
        seeds=seed_tuple,
        raw_forecast_mean=raw_mean,
        monthly_scores=scores,
        monthly_metrics=monthly,
        fold_metrics=fold_metrics,
        config_metrics=config_metrics,
        bootstrap_months=plan.months,
        bootstrap_indices=plan.indices,
    )


__all__ = [
    "FamilyEvaluation",
    "OOF_KEY_COLUMNS",
    "evaluate_family_predictions",
]
