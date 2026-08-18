"""Exact Stage 7 split consumption for Stage 9 model workers."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_bot.selectors.weights import (
    latest_month_mask,
    month_balanced_memory_weights,
    month_ordinal,
)


@dataclass(frozen=True)
class FoldIndices:
    fold_id: str
    fit_signal_date: dt.date
    full_train: np.ndarray
    inner_train: np.ndarray
    inner_purged: np.ndarray
    inner_validation: np.ndarray
    outer_validation: np.ndarray
    outer_evaluation: np.ndarray
    full_train_weights: np.ndarray
    inner_train_weights: np.ndarray
    inner_validation_weights: np.ndarray


def _date(value: object) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


def build_fold_indices(
    rows: pd.DataFrame,
    split_membership: pd.DataFrame,
    *,
    target_column: str = "residual_return_rank",
    expected_folds: int = 4,
) -> tuple[FoldIndices, ...]:
    """Build train/inner/outer row indices from the published split truth."""

    required_rows = {"signal_date", "permno", "holding_month", target_column}
    required_splits = {
        "fold_id",
        "signal_date",
        "holding_month",
        "fit_signal_date",
        "included_in_training",
        "included_in_validation",
        "available_by_fit",
        "purged_or_embargoed",
    }
    if not required_rows <= set(rows) or not required_splits <= set(split_membership):
        raise ValueError("Stage 9 split inputs are missing required columns")
    if rows.duplicated(["signal_date", "permno"]).any():
        raise ValueError("Stage 9 selector rows contain duplicate keys")
    target = pd.to_numeric(rows[target_column], errors="coerce").to_numpy(np.float64)
    valid_target = np.isfinite(target)
    result: list[FoldIndices] = []
    for fold_id, membership in split_membership.groupby("fold_id", sort=True):
        membership = membership.sort_values("signal_date", kind="stable")
        fit_dates = {_date(value) for value in membership["fit_signal_date"]}
        if len(fit_dates) != 1:
            raise ValueError(f"Fold {fold_id} has multiple fit dates")
        fit_date = fit_dates.pop()
        if bool(
            membership.loc[membership["included_in_training"], "available_by_fit"]
            .eq(False)
            .any()
        ):
            raise ValueError(f"Fold {fold_id} includes unavailable training labels")
        if bool(
            (
                membership["included_in_training"]
                & membership["included_in_validation"]
            ).any()
        ):
            raise ValueError(f"Fold {fold_id} has train/validation month overlap")
        role = membership[
            ["signal_date", "holding_month", "included_in_training", "included_in_validation"]
        ]
        joined = rows[["signal_date", "holding_month"]].merge(
            role,
            on=["signal_date", "holding_month"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        if joined[["included_in_training", "included_in_validation"]].isna().any().any():
            raise ValueError(f"Fold {fold_id} does not cover every selector month")
        train_mask = joined["included_in_training"].to_numpy(bool) & valid_target
        validation_mask = joined["included_in_validation"].to_numpy(bool)
        evaluation_mask = validation_mask & valid_target
        train_index = np.flatnonzero(train_mask).astype(np.int64)
        outer_index = np.flatnonzero(validation_mask).astype(np.int64)
        evaluation_index = np.flatnonzero(evaluation_mask).astype(np.int64)
        if not train_index.size or not outer_index.size or not evaluation_index.size:
            raise ValueError(f"Fold {fold_id} has an empty required partition")
        train_months = rows.iloc[train_index]["holding_month"].astype(str).tolist()
        inner_validation_local = latest_month_mask(train_months)
        validation_month_ordinals = {
            month_ordinal(value)
            for value, included in zip(train_months, inner_validation_local, strict=True)
            if included
        }
        inner_boundary = min(validation_month_ordinals)
        inner_purge_local = np.asarray(
            [month_ordinal(value) == inner_boundary - 1 for value in train_months],
            dtype=bool,
        )
        if not inner_purge_local.any():
            raise ValueError(f"Fold {fold_id} has no one-period inner-boundary purge month")
        inner_validation_index = train_index[inner_validation_local]
        inner_purged_index = train_index[inner_purge_local]
        inner_train_index = train_index[~inner_validation_local & ~inner_purge_local]
        if not inner_train_index.size:
            raise ValueError(f"Fold {fold_id} has no rows before its purged inner boundary")
        full_weights = month_balanced_memory_weights(
            train_months, fit_signal_date=fit_date
        )
        inner_train_weights = month_balanced_memory_weights(
            rows.iloc[inner_train_index]["holding_month"].astype(str).tolist(),
            fit_signal_date=fit_date,
        )
        inner_validation_weights = month_balanced_memory_weights(
            rows.iloc[inner_validation_index]["holding_month"].astype(str).tolist(),
            fit_signal_date=fit_date,
        )
        result.append(
            FoldIndices(
                fold_id=str(fold_id),
                fit_signal_date=fit_date,
                full_train=train_index,
                inner_train=inner_train_index,
                inner_purged=inner_purged_index,
                inner_validation=inner_validation_index,
                outer_validation=outer_index,
                outer_evaluation=evaluation_index,
                full_train_weights=full_weights,
                inner_train_weights=inner_train_weights,
                inner_validation_weights=inner_validation_weights,
            )
        )
    if len(result) != expected_folds:
        raise ValueError(f"Expected {expected_folds} folds, found {len(result)}")
    all_outer = np.concatenate([fold.outer_validation for fold in result])
    if np.unique(all_outer).size != all_outer.size:
        raise ValueError("OOF validation rows overlap between folds")
    return tuple(result)
