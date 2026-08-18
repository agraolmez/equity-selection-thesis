"""Prospective one-month encoder for the frozen V6 850-column schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_bot.research_v6.preprocessing import (
    DEVELOPMENT_AGGREGATE_HISTORY,
    EncodedPanel,
    _centered_rank_winsorized,
    _past_only_month_zscore,
    _tokens,
)


class LivePreprocessingError(RuntimeError):
    """The prospective panel cannot reproduce the frozen encoder."""


def _collapse_aggregate_months(
    frame: pd.DataFrame, aggregate_features: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = frame[["signal_date", *aggregate_features]].copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"], errors="raise")
    for signal_date, group in work.groupby("signal_date", observed=True, sort=True):
        row: dict[str, Any] = {"signal_date": pd.Timestamp(signal_date)}
        for feature in aggregate_features:
            numeric = pd.to_numeric(group[feature], errors="coerce").to_numpy(float)
            finite = numeric[np.isfinite(numeric)]
            if len(finite) and not np.allclose(finite, finite[0], rtol=0.0, atol=0.0):
                raise LivePreprocessingError(
                    f"aggregate feature {feature} varies within {pd.Timestamp(signal_date).date()}"
                )
            row[feature] = float(finite[0]) if len(finite) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_history_before(
    *,
    signal_date: pd.Timestamp,
    aggregate_features: list[str],
    historical_model_root: Path,
) -> pd.DataFrame:
    development = pd.read_parquet(
        DEVELOPMENT_AGGREGATE_HISTORY,
        columns=["signal_date", *aggregate_features],
    )
    parts = [_collapse_aggregate_months(development, aggregate_features)]
    for path in sorted((historical_model_root / "months").glob("????-??.parquet")):
        frame = pd.read_parquet(path, columns=["signal_date", *aggregate_features])
        month_date = pd.to_datetime(frame["signal_date"], errors="raise")
        if month_date.nunique() != 1:
            raise LivePreprocessingError(f"historical aggregate month is ambiguous: {path}")
        if pd.Timestamp(month_date.iloc[0]) < signal_date:
            parts.append(_collapse_aggregate_months(frame, aggregate_features))
    history = pd.concat(parts, ignore_index=True)
    # Development and model-feature authorities may meet at a boundary.  Their
    # common month must agree exactly before one copy is retained.
    collapsed = _collapse_aggregate_months(history, aggregate_features)
    return collapsed.sort_values("signal_date", kind="mergesort").reset_index(drop=True)


def encode_live_panel(
    panel: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    category_vocabulary_path: Path,
    historical_model_root: Path,
) -> EncodedPanel:
    """Encode one current month using the frozen historical preprocessing state."""

    keys = ["holding_month", "signal_date", "permno", "permco"]
    if not set(keys).issubset(panel.columns):
        raise LivePreprocessingError("live panel omits frozen model keys")
    features = registry["feature_name"].astype(str).tolist()
    missing = sorted(set(features) - set(panel.columns))
    if missing:
        raise LivePreprocessingError(f"live panel omits registry features: {missing}")
    work = panel.sort_values(["holding_month", "permno"], kind="mergesort").reset_index(drop=True)
    if work["holding_month"].astype(str).nunique() != 1:
        raise LivePreprocessingError("prospective encoding requires exactly one month")
    signal_dates = pd.to_datetime(work["signal_date"], errors="raise")
    if signal_dates.nunique() != 1:
        raise LivePreprocessingError("prospective encoding requires one signal date")
    columns: list[np.ndarray] = []
    names: list[str] = []
    sources: list[str] = []

    for row in registry.itertuples(index=False):
        feature = str(row.feature_name)
        kind = str(row.kind)
        if kind == "continuous_firm":
            value, absent = _centered_rank_winsorized(work[feature])
            columns.extend((value, absent.astype(np.float32)))
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
        signal_date = pd.Timestamp(signal_dates.iloc[0])
        history = _aggregate_history_before(
            signal_date=signal_date,
            aggregate_features=aggregate_features,
            historical_model_root=historical_model_root,
        )
        current = _collapse_aggregate_months(work, aggregate_features)
        timeline = pd.concat([history, current], ignore_index=True).sort_values(
            "signal_date", kind="mergesort"
        ).reset_index(drop=True)
        if timeline["signal_date"].duplicated().any():
            raise LivePreprocessingError("current aggregate month overlaps historical inputs")
        current_index = timeline.index[timeline["signal_date"].eq(signal_date)]
        if len(current_index) != 1:
            raise LivePreprocessingError("current aggregate state was not identified")
        current_index_value = int(current_index[0])
        for feature in aggregate_features:
            score, absent = _past_only_month_zscore(timeline[feature])
            value = float(score.loc[current_index_value])
            missing_value = float(absent.loc[current_index_value])
            columns.extend(
                (
                    np.full(len(work), value, dtype=np.float32),
                    np.full(len(work), missing_value, dtype=np.float32),
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
            raise LivePreprocessingError(f"category vocabulary is empty for {feature}")
        token_sets = work[feature].map(_tokens)
        unknown_rows = token_sets.map(len).eq(0).to_numpy()
        known_values = set(feature_vocabulary["category_value"].astype(str))
        for item in feature_vocabulary.itertuples(index=False):
            category = str(item.category_value)
            if bool(item.is_unknown):
                active = unknown_rows | token_sets.map(
                    lambda values: bool(values)
                    and not any(value in known_values for value in values)
                ).to_numpy()
            elif kind == "categorical_multihot_firm":
                active = token_sets.map(lambda values: category in values).to_numpy()
            else:
                active = token_sets.map(
                    lambda values: bool(values) and values[0] == category
                ).to_numpy()
            columns.append(active.astype(np.float32))
            names.append(str(item.encoded_column_name))
            sources.append(feature)

    values = np.column_stack(columns).astype(np.float32, copy=False)
    if not np.isfinite(values).all() or len(names) != values.shape[1]:
        raise LivePreprocessingError("prospective encoded matrix is invalid")
    return EncodedPanel(
        values=values,
        feature_names=tuple(names),
        source_features=tuple(sources),
        keys=work[keys].copy(),
    )


__all__ = ["LivePreprocessingError", "encode_live_panel"]
