"""Outcome-blind V6.1 feature views and direct CAM score construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from portfolio_bot.research_v6.models import rank_within_month
from portfolio_bot.research_v6.selector_runtime import atomic_json, atomic_parquet, canonical_sha256


def _metadata_identity(metadata: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "feature_names": metadata["feature_names"],
            "source_features": metadata["source_features"],
            "feature_family": metadata["feature_family"],
        }
    )


def _registry_sources(metadata: Mapping[str, Any]) -> list[str]:
    sources = list(metadata["feature_family"])
    encoded_sources = set(map(str, metadata["source_features"]))
    if set(sources) != encoded_sources:
        raise ValueError("V6 metadata source registry and encoded sources disagree")
    return sources


def _value_and_missing_indices(
    metadata: Mapping[str, Any], source: str
) -> tuple[int | None, int | None]:
    names = list(map(str, metadata["feature_names"]))
    lookup = {name: index for index, name in enumerate(names)}
    return lookup.get(source), lookup.get(source + "__missing")


def _correlation_reduced_sources(
    matrix: np.ndarray,
    rows: pd.DataFrame,
    metadata: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]], pd.DataFrame]:
    """Select source features from training inputs without consulting outcomes."""

    end_month = str(policy["training_end"])
    train_index = np.flatnonzero(rows["holding_month"].astype(str).le(end_month)).astype(
        np.int64
    )
    if len(train_index) == 0:
        raise ValueError("correlation-reduction training slice is empty")
    sources = _registry_sources(metadata)
    candidates: list[str] = []
    categorical: list[str] = []
    coverage: dict[str, float] = {}
    value_indices: list[int] = []
    missing_indices: list[int | None] = []
    noninformative: list[dict[str, Any]] = []
    for source in sources:
        value_index, missing_index = _value_and_missing_indices(metadata, source)
        if value_index is None:
            categorical.append(source)
            continue
        values = np.asarray(matrix[train_index, value_index], dtype=np.float32)
        if missing_index is None:
            observed = np.isfinite(values)
        else:
            observed = np.asarray(matrix[train_index, missing_index] < 0.5, dtype=bool)
        coverage[source] = float(observed.mean())
        observed_values = values[observed]
        if len(observed_values) < int(policy["minimum_pair_observations"]):
            noninformative.append(
                {"source_feature": source, "reason": "insufficient_observations"}
            )
            continue
        if float(np.nanstd(observed_values.astype(np.float64))) <= 1e-12:
            noninformative.append({"source_feature": source, "reason": "constant"})
            continue
        candidates.append(source)
        value_indices.append(value_index)
        missing_indices.append(missing_index)

    values = np.asarray(matrix[np.ix_(train_index, np.asarray(value_indices))], dtype=np.float32)
    for ordinal, missing_index in enumerate(missing_indices):
        if missing_index is not None:
            absent = np.asarray(matrix[train_index, missing_index] >= 0.5, dtype=bool)
            values[absent, ordinal] = np.nan
    # Firm inputs are already cross-sectionally rank transformed. Compute their
    # pairwise-observed rank correlation with matrix products so the preparation
    # scales linearly in rows instead of reranking every feature pair.
    observed = np.isfinite(values).astype(np.float32)
    clean = np.nan_to_num(values, nan=0.0).astype(np.float32, copy=False)
    count = observed.T @ observed
    sum_left = clean.T @ observed
    sum_square_left = np.square(clean).T @ observed
    cross = clean.T @ clean
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = cross - sum_left * sum_left.T / count
        variance_left = sum_square_left - np.square(sum_left) / count
        variance_right = sum_square_left.T - np.square(sum_left.T) / count
        correlation_values = numerator / np.sqrt(variance_left * variance_right)
    correlation_values[count < int(policy["minimum_pair_observations"])] = np.nan
    np.fill_diagonal(correlation_values, 1.0)
    correlation = pd.DataFrame(correlation_values, index=candidates, columns=candidates)
    threshold = float(policy["absolute_rank_correlation_threshold"])
    registry_ordinal = {source: ordinal for ordinal, source in enumerate(sources)}
    ordered = sorted(
        candidates,
        key=lambda source: (-coverage[source], registry_ordinal[source], source),
    )
    retained: list[str] = []
    dropped: list[dict[str, Any]] = list(noninformative)
    edge_rows: list[dict[str, Any]] = []
    for source in ordered:
        links: list[tuple[float, str]] = []
        for representative in retained:
            value = correlation.at[source, representative]
            if np.isfinite(value) and abs(float(value)) > threshold:
                links.append((abs(float(value)), representative))
        if links:
            absolute_value, representative = max(links, key=lambda item: (item[0], item[1]))
            signed = float(correlation.at[source, representative])
            dropped.append(
                {
                    "source_feature": source,
                    "reason": "correlated",
                    "representative": representative,
                    "rank_correlation": signed,
                    "absolute_rank_correlation": absolute_value,
                }
            )
        else:
            retained.append(source)
    for left_position, left in enumerate(candidates):
        for right in candidates[left_position + 1 :]:
            value = correlation.at[left, right]
            if np.isfinite(value) and abs(float(value)) > threshold:
                edge_rows.append(
                    {
                        "left_source": left,
                        "right_source": right,
                        "rank_correlation": float(value),
                        "absolute_rank_correlation": abs(float(value)),
                    }
                )
    selected = sorted(
        set(retained).union(categorical), key=lambda source: registry_ordinal[source]
    )
    return selected, dropped, pd.DataFrame(edge_rows)


def prepare_feature_views(
    *,
    matrix: np.ndarray,
    rows: pd.DataFrame,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    input_identity: str,
    output: Path,
) -> dict[str, Any]:
    """Materialize a receipt-bound V6.1 feature-view manifest."""

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "feature_views.json"
    policy = config["feature_views"]["correlation_reduction"]
    identity = canonical_sha256(
        {
            "input_identity": input_identity,
            "metadata_identity": _metadata_identity(metadata),
            "policy": policy,
            "direct_cam_sources": config["feature_views"]["direct_cam_sources"],
        }
    )
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing.get("identity") != identity:
            raise ValueError("existing V6.1 feature-view manifest has a different identity")
        return existing

    all_sources = _registry_sources(metadata)
    direct_cam = set(map(str, config["feature_views"]["direct_cam_sources"]))
    if not direct_cam.issubset(all_sources):
        raise ValueError("direct CAM source list is not present in the V6 input cache")
    correlation_sources, dropped, edges = _correlation_reduced_sources(
        matrix, rows, metadata, policy
    )
    views = {
        "full": all_sources,
        "cam_free": [source for source in all_sources if source not in direct_cam],
        "correlation_reduced": correlation_sources,
    }
    encoded_sources = np.asarray(metadata["source_features"], dtype=str)
    payload: dict[str, Any] = {
        "schema": "research_v6_1_feature_views_v1",
        "identity": identity,
        "input_identity": input_identity,
        "metadata_identity": _metadata_identity(metadata),
        "correlation_policy": policy,
        "views": {},
        "correlation_dropped": dropped,
        "outcomes_used": False,
    }
    for view, sources in views.items():
        index = np.flatnonzero(np.isin(encoded_sources, sources)).astype(np.int64)
        if len(index) == 0:
            raise ValueError(f"V6.1 feature view {view} is empty")
        payload["views"][view] = {
            "raw_sources": sources,
            "raw_source_count": len(sources),
            "encoded_indices": index.tolist(),
            "encoded_feature_count": len(index),
        }
    if edges.empty:
        edges = pd.DataFrame(
            columns=[
                "left_source",
                "right_source",
                "rank_correlation",
                "absolute_rank_correlation",
            ]
        )
    atomic_parquet(edges, output / "correlation_edges.parquet")
    atomic_json(payload, manifest_path)
    return payload


def direct_cam_predictions(
    *,
    matrix: np.ndarray,
    rows: pd.DataFrame,
    metadata: Mapping[str, Any],
    source: str,
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    """Return the unmodeled point-in-time CAM level as a monthly stock rank."""

    value_index, missing_index = _value_and_missing_indices(metadata, source)
    if value_index is None:
        raise ValueError(f"direct CAM source {source} has no encoded value column")
    keep = rows["holding_month"].astype(str).between(start_month, end_month).to_numpy()
    index = np.flatnonzero(keep).astype(np.int64)
    result = rows.iloc[index][
        ["holding_month", "signal_date", "permno", "permco"]
    ].copy()
    raw = np.asarray(matrix[index, value_index], dtype=float)
    result["raw_score"] = raw
    result["score_rank"] = rank_within_month(raw, result["holding_month"])
    result["cam_available"] = (
        True
        if missing_index is None
        else np.asarray(matrix[index, missing_index] < 0.5, dtype=bool)
    )
    return result


__all__ = ["direct_cam_predictions", "prepare_feature_views"]
