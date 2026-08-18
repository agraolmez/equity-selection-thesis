"""Frozen Stage 9 complexity summaries and deterministic tie fields."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from portfolio_bot.selectors.model_specs import (
    XGBoostConfig,
    core_xgboost_configurations,
)


def _require_receipts(
    receipts: pd.DataFrame,
    *,
    family: str,
    config_ids: Sequence[str],
    repetitions: int,
    value_column: str,
) -> pd.Series:
    required = {"model_family", "config_id", value_column}
    if not required <= set(receipts):
        raise ValueError(f"Fit receipts are missing {family} complexity fields")
    frame = receipts.loc[receipts["model_family"].astype(str).eq(family)].copy()
    expected = tuple(str(value) for value in config_ids)
    if set(frame["config_id"].astype(str)) != set(expected):
        raise ValueError(f"Fit receipts do not contain the complete {family} grid")
    counts = frame.groupby("config_id", observed=True).size()
    if not counts.eq(repetitions).all():
        raise ValueError(f"Every {family} config requires {repetitions} fit receipts")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all() or (values < 0).any():
        raise ValueError(f"{family} complexity receipts must be finite and nonnegative")
    frame[value_column] = values
    return frame.groupby("config_id", observed=True)[value_column].median()


def neural_complexity(
    receipts: pd.DataFrame,
    *,
    config_ids: Sequence[str],
) -> pd.DataFrame:
    """Median realized trainable parameter count across the 12 OOF fits."""

    values = _require_receipts(
        receipts,
        family="neural_network",
        config_ids=config_ids,
        repetitions=12,
        value_column="parameter_count",
    )
    return pd.DataFrame(
        {
            "config_id": list(config_ids),
            "complexity": [float(values.loc[value]) for value in config_ids],
        }
    )


def xgboost_complexity(
    receipts: pd.DataFrame,
    *,
    configurations: Sequence[XGBoostConfig],
) -> pd.DataFrame:
    """Depth times median early-stopped tree count over four folds/three seeds."""

    configs = tuple(configurations)
    ids = tuple(value.config_id for value in configs)
    trees = _require_receipts(
        receipts,
        family="xgboost",
        config_ids=ids,
        repetitions=12,
        value_column="best_tree_count",
    )
    records = []
    for config in configs:
        records.append(
            {
                "config_id": config.config_id,
                "complexity": float(config.max_depth * trees.loc[config.config_id]),
                # Ascending means stronger regularization after complexity ties.
                "regularization_tuple": (
                    -float(config.reg_lambda),
                    -float(config.reg_alpha),
                    -float(config.child_share),
                    float(config.subsample),
                    float(config.colsample_bytree),
                    float(config.eta),
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def elastic_net_complexity(
    receipts: pd.DataFrame,
    *,
    config_ids: Sequence[str],
) -> pd.DataFrame:
    """Median number of nonzero penalized coefficients over four folds."""

    values = _require_receipts(
        receipts,
        family="elastic_net",
        config_ids=config_ids,
        repetitions=4,
        value_column="nonzero_coefficients",
    )
    return pd.DataFrame(
        {
            "config_id": list(config_ids),
            "complexity": [float(values.loc[value]) for value in config_ids],
        }
    )


def attach_complexity(
    metrics: pd.DataFrame,
    complexity: pd.DataFrame,
    *,
    family: str,
) -> pd.DataFrame:
    """One-to-one merge a family metric table with audited complexity."""

    family_metrics = metrics.loc[metrics["model_family"].astype(str).eq(family)].copy()
    if family_metrics.empty or family_metrics["config_id"].duplicated().any():
        raise ValueError(f"{family} configuration metrics are absent or duplicated")
    if complexity["config_id"].duplicated().any():
        raise ValueError(f"{family} complexity rows are duplicated")
    result = family_metrics.merge(
        complexity,
        on="config_id",
        how="left",
        validate="one_to_one",
    )
    if result["complexity"].isna().any():
        raise ValueError(f"{family} configuration complexity is incomplete")
    return result


__all__ = [
    "attach_complexity",
    "elastic_net_complexity",
    "neural_complexity",
    "xgboost_complexity",
]
