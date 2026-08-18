"""Outcome-blind preprocessing for the V6 tabular selector models."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from portfolio_bot.research_v6.data import FEATURE_ROOT


DEVELOPMENT_AGGREGATE_HISTORY = Path(
    "/Users/agraemirolmez/Desktop/Thesis/interim/phase_03_features/"
    "phase03-feature-label-split-v1/feature_store_builds/"
    "phase03-features-fda21f0e51b8276c10246f9f/feature_store_raw.parquet"
)


@dataclasses.dataclass(frozen=True, slots=True)
class EncodedPanel:
    values: np.ndarray
    feature_names: tuple[str, ...]
    source_features: tuple[str, ...]
    keys: pd.DataFrame


def _centered_rank_winsorized(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(numeric)
    output = np.zeros(len(numeric), dtype=np.float32)
    if finite.any():
        population = numeric[finite]
        lower, upper = np.quantile(population, [0.01, 0.99], method="linear")
        clipped = np.clip(population, lower, upper)
        ranks = pd.Series(clipped).rank(method="average").to_numpy(np.float64)
        output[finite] = ((ranks - 0.5) / len(ranks) - 0.5).astype(np.float32)
    return output, ~finite


def _past_only_month_zscore(
    month_values: pd.Series, *, clip: float = 5.0
) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(month_values, errors="coerce").to_numpy(np.float64)
    transformed = np.zeros(len(values), dtype=np.float32)
    missing = np.ones(len(values), dtype=bool)
    history: list[float] = []
    for index, value in enumerate(values):
        if len(history) >= 2 and np.isfinite(value):
            mean = float(np.mean(history, dtype=np.float64))
            std = float(np.std(history, ddof=1, dtype=np.float64))
            if math.isfinite(std) and std > 0.0:
                transformed[index] = np.float32(np.clip((value - mean) / std, -clip, clip))
                missing[index] = False
        if np.isfinite(value):
            history.append(float(value))
    return (
        pd.Series(transformed, index=month_values.index),
        pd.Series(missing.astype(np.float32), index=month_values.index),
    )


def _is_missing_scalar(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _tokens(value: Any) -> tuple[str, ...]:
    if _is_missing_scalar(value):
        return ()
    if isinstance(value, np.ndarray):
        raw = value.tolist()
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = [value]
    return tuple(str(item) for item in raw if not _is_missing_scalar(item))


def _aggregate_month_history(
    panel: pd.DataFrame,
    aggregate_features: Sequence[str],
) -> pd.DataFrame:
    development = pd.read_parquet(
        DEVELOPMENT_AGGREGATE_HISTORY,
        columns=["signal_date", *aggregate_features],
    )
    current = panel.loc[:, ["signal_date", *aggregate_features]]
    combined = pd.concat([development, current], ignore_index=True)
    combined["signal_date"] = pd.to_datetime(combined["signal_date"], errors="raise")
    rows: list[dict[str, Any]] = []
    for signal_date, group in combined.groupby("signal_date", observed=True, sort=True):
        row: dict[str, Any] = {"signal_date": pd.Timestamp(signal_date)}
        for feature in aggregate_features:
            values = pd.to_numeric(group[feature], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            if len(finite) and not np.allclose(finite, finite[0], rtol=0.0, atol=0.0):
                raise ValueError(f"aggregate feature {feature} varies within a signal date")
            row[feature] = float(finite[0]) if len(finite) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("signal_date", kind="mergesort").drop_duplicates(
        "signal_date", keep="last"
    )


def encode_panel(
    panel: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    category_vocabulary_path: Path = FEATURE_ROOT / "category_vocabulary.parquet",
) -> EncodedPanel:
    """Encode one chronological modeling slice without consulting outcomes."""

    required_keys = ["holding_month", "signal_date", "permno", "permco"]
    if not set(required_keys).issubset(panel.columns):
        raise ValueError("panel omits V6 keys")
    features = registry["feature_name"].astype(str).tolist()
    missing = sorted(set(features).difference(panel.columns))
    if missing:
        raise ValueError(f"panel omits registered features: {missing}")
    work = panel.sort_values(["holding_month", "permno"], kind="mergesort").reset_index(
        drop=True
    )
    columns: list[np.ndarray] = []
    names: list[str] = []
    sources: list[str] = []
    month_groups = work.groupby("holding_month", observed=True, sort=False).groups

    for row in registry.itertuples(index=False):
        feature = str(row.feature_name)
        kind = str(row.kind)
        if kind == "continuous_firm":
            value = np.zeros(len(work), dtype=np.float32)
            indicator = np.zeros(len(work), dtype=np.float32)
            for indices in month_groups.values():
                ranked, absent = _centered_rank_winsorized(work.loc[indices, feature])
                value[np.asarray(indices)] = ranked
                indicator[np.asarray(indices)] = absent.astype(np.float32)
            columns.extend((value, indicator))
            names.extend((feature, feature + "__missing"))
            sources.extend((feature, feature))
        elif kind == "binary_firm":
            numeric = pd.to_numeric(work[feature], errors="coerce").to_numpy(float)
            absent = ~np.isfinite(numeric)
            columns.extend(
                (
                    np.where(absent, 0.0, numeric).astype(np.float32),
                    absent.astype(np.float32),
                )
            )
            names.extend((feature, feature + "__missing"))
            sources.extend((feature, feature))

    aggregate_features = registry.loc[
        registry["kind"].eq("aggregate_state"), "feature_name"
    ].astype(str).tolist()
    if aggregate_features:
        history = _aggregate_month_history(work, aggregate_features)
        score_table = history.loc[:, ["signal_date"]].copy()
        for feature in aggregate_features:
            score, absent = _past_only_month_zscore(history[feature])
            score_table[feature] = score
            score_table[feature + "__missing"] = absent
        row_scores = work.loc[:, ["signal_date"]].merge(
            score_table, on="signal_date", how="left", validate="many_to_one"
        )
        for feature in aggregate_features:
            columns.extend(
                (
                    row_scores[feature].fillna(0.0).to_numpy(np.float32),
                    row_scores[feature + "__missing"].fillna(1.0).to_numpy(np.float32),
                )
            )
            names.extend((feature, feature + "__missing"))
            sources.extend((feature, feature))

    vocabulary = pd.read_parquet(category_vocabulary_path)
    categorical = registry.loc[
        registry["kind"].isin(("categorical_firm", "categorical_multihot_firm"))
    ]
    for row in categorical.itertuples(index=False):
        feature = str(row.feature_name)
        kind = str(row.kind)
        feature_vocabulary = vocabulary.loc[
            vocabulary["feature_name"].eq(feature)
        ].sort_values("category_ordinal", kind="mergesort")
        if feature_vocabulary.empty:
            raise ValueError(f"category vocabulary is empty for {feature}")
        token_sets = work[feature].map(_tokens)
        unknown_rows = token_sets.map(len).eq(0).to_numpy()
        known_values = set(feature_vocabulary["category_value"].astype(str))
        for item in feature_vocabulary.itertuples(index=False):
            category = str(item.category_value)
            encoded_name = str(item.encoded_column_name)
            if bool(item.is_unknown):
                active = unknown_rows | token_sets.map(
                    lambda values: bool(values) and not any(v in known_values for v in values)
                ).to_numpy()
            elif kind == "categorical_multihot_firm":
                active = token_sets.map(lambda values: category in values).to_numpy()
            else:
                active = token_sets.map(lambda values: bool(values) and values[0] == category).to_numpy()
            columns.append(active.astype(np.float32))
            names.append(encoded_name)
            sources.append(feature)

    if not columns:
        raise ValueError("V6 encoder produced no columns")
    values = np.column_stack(columns).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("V6 encoded matrix contains nonfinite values")
    if (
        len(names) != values.shape[1]
        or len(sources) != values.shape[1]
        or len(set(names)) != len(names)
    ):
        raise ValueError("V6 encoded feature identities are invalid")
    return EncodedPanel(
        values=values,
        feature_names=tuple(names),
        source_features=tuple(sources),
        keys=work.loc[:, required_keys].copy(),
    )


__all__ = ["EncodedPanel", "encode_panel"]
