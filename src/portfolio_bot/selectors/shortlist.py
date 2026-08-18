"""Frozen Stage 9 selector gates and deterministic family shortlisting."""

from __future__ import annotations

import math
from numbers import Real
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


TIE_TOLERANCE = 1e-12
ELASTIC_NET_FAMILY = "elastic_net"
SHORTLIST_COLUMNS = (
    "model_family",
    "config_id",
    "shortlist_order",
    "selection_reasons",
    "benchmark_only_fallback",
    "gate_pass",
    "mean_rank_ic",
    "rank_ic_bootstrap_se",
    "positive_fold_count",
    "fold_dispersion",
    "complexity",
    "family_best_mean_rank_ic",
    "family_best_bootstrap_se",
    "one_se_threshold",
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _sortable_value(value: object) -> tuple[int, object]:
    """Return a deterministic ascending key for a pre-oriented tie-break field."""

    if isinstance(value, (bool, np.bool_)):
        return (0, int(value))
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Regularization tie-break values must be finite")
        return (0, numeric)
    if isinstance(value, str):
        return (1, value)
    if isinstance(value, (tuple, list)):
        return (2, tuple(_sortable_value(item) for item in value))
    if value is None:
        raise ValueError("Regularization tie-break values may not be null")
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        raise ValueError("Regularization tie-break values may not be null")
    return (3, repr(value))


def _regularization_key(
    row: pd.Series,
    columns: Sequence[str],
) -> tuple[tuple[int, object], ...]:
    return tuple(_sortable_value(row[column]) for column in columns)


def _tie_break(
    candidates: pd.DataFrame,
    *,
    regularization_columns: Sequence[str],
) -> pd.Series:
    if candidates.empty:
        raise ValueError("Cannot tie-break an empty candidate set")
    ordered = sorted(
        (row for _, row in candidates.iterrows()),
        key=lambda row: (
            _regularization_key(row, regularization_columns),
            str(row["config_id"]),
        ),
    )
    return ordered[0]


def _choose_maximum(
    candidates: pd.DataFrame,
    column: str,
    *,
    regularization_columns: Sequence[str],
    tolerance: float,
) -> pd.Series:
    optimum = float(candidates[column].max())
    tied = candidates.loc[candidates[column] >= optimum - tolerance]
    return _tie_break(tied, regularization_columns=regularization_columns)


def _choose_minimum(
    candidates: pd.DataFrame,
    column: str,
    *,
    regularization_columns: Sequence[str],
    tolerance: float,
) -> pd.Series:
    optimum = float(candidates[column].min())
    tied = candidates.loc[candidates[column] <= optimum + tolerance]
    return _tie_break(tied, regularization_columns=regularization_columns)


def _empty_shortlist() -> pd.DataFrame:
    return pd.DataFrame(columns=list(SHORTLIST_COLUMNS))


def _validate_family_metrics(
    metrics: pd.DataFrame,
    *,
    regularization_columns: Sequence[str],
) -> None:
    required = (
        "model_family",
        "config_id",
        "mean_rank_ic",
        "rank_ic_bootstrap_se",
        "positive_fold_count",
        "fold_dispersion",
        "gate_pass",
        "complexity",
        *regularization_columns,
    )
    _require_columns(metrics, required)
    if metrics["config_id"].duplicated().any():
        raise ValueError("Configuration identifiers must be unique within a family")
    numeric_columns = (
        "mean_rank_ic",
        "rank_ic_bootstrap_se",
        "positive_fold_count",
        "fold_dispersion",
        "complexity",
    )
    for column in numeric_columns:
        values = pd.to_numeric(metrics[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite")
    if (pd.to_numeric(metrics["rank_ic_bootstrap_se"]) < 0.0).any():
        raise ValueError("rank_ic_bootstrap_se may not be negative")
    if (pd.to_numeric(metrics["fold_dispersion"]) < 0.0).any():
        raise ValueError("fold_dispersion may not be negative")
    if (pd.to_numeric(metrics["complexity"]) < 0.0).any():
        raise ValueError("complexity may not be negative")
    positive_fold_count = pd.to_numeric(metrics["positive_fold_count"])
    if ((positive_fold_count < 0) | (positive_fold_count > 4)).any():
        raise ValueError("positive_fold_count must be within [0, 4]")

    expected_gate = (
        (pd.to_numeric(metrics["mean_rank_ic"]) > 0.0)
        & (positive_fold_count >= 3)
    )
    supplied_gate = metrics["gate_pass"].astype(bool)
    if not np.array_equal(expected_gate.to_numpy(), supplied_gate.to_numpy()):
        raise ValueError("gate_pass does not match the frozen 3-of-4 positive-IC gate")
    for _, row in metrics.iterrows():
        _regularization_key(row, regularization_columns)


def shortlist_family(
    config_metrics: pd.DataFrame,
    *,
    model_family: str,
    regularization_columns: Sequence[str],
    elastic_net_family: str = ELASTIC_NET_FAMILY,
    tolerance: float = TIE_TOLERANCE,
) -> pd.DataFrame:
    """Retain at most three distinct configurations for one model family.

    Tie-break columns must be supplied in the frozen model-specific order and
    oriented so their ascending tuple means stronger preferred regularization.
    Final ties use ``config_id``.  Only Elastic Net receives the explicitly
    labelled benchmark-only fallback when no configuration passes the gate.
    """

    if not isinstance(model_family, str) or not model_family:
        raise ValueError("model_family must be a nonempty string")
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    family = config_metrics.loc[
        config_metrics["model_family"].astype(str) == model_family
    ].copy()
    if family.empty:
        return _empty_shortlist()
    _validate_family_metrics(
        family,
        regularization_columns=regularization_columns,
    )
    tolerance = float(tolerance)
    eligible = family.loc[family["gate_pass"].astype(bool)].copy()

    if eligible.empty:
        if model_family != elastic_net_family:
            return _empty_shortlist()
        fallback = _choose_maximum(
            family,
            "mean_rank_ic",
            regularization_columns=regularization_columns,
            tolerance=tolerance,
        )
        best_mean = float(fallback["mean_rank_ic"])
        best_se = float(fallback["rank_ic_bootstrap_se"])
        output = fallback.to_dict()
        output.update(
            {
                "shortlist_order": 1,
                "selection_reasons": "benchmark_only_fallback",
                "benchmark_only_fallback": True,
                "family_best_mean_rank_ic": best_mean,
                "family_best_bootstrap_se": best_se,
                "one_se_threshold": best_mean - best_se,
            }
        )
        return pd.DataFrame([output]).loc[:, list(SHORTLIST_COLUMNS)]

    best = _choose_maximum(
        eligible,
        "mean_rank_ic",
        regularization_columns=regularization_columns,
        tolerance=tolerance,
    )
    best_mean = float(best["mean_rank_ic"])
    best_se = float(best["rank_ic_bootstrap_se"])
    threshold = best_mean - best_se
    one_se = eligible.loc[
        pd.to_numeric(eligible["mean_rank_ic"]) >= threshold - tolerance
    ].copy()

    simplest = _choose_minimum(
        one_se,
        "complexity",
        regularization_columns=regularization_columns,
        tolerance=tolerance,
    )
    stable = _choose_minimum(
        one_se,
        "fold_dispersion",
        regularization_columns=regularization_columns,
        tolerance=tolerance,
    )
    selections = (
        (best, "highest_mean_rank_ic"),
        (simplest, "simplest_within_one_se"),
        (stable, "lowest_fold_dispersion_within_one_se"),
    )

    by_config: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for row, reason in selections:
        config_id = str(row["config_id"])
        if config_id not in by_config:
            output = row.to_dict()
            output.update(
                {
                    "selection_reasons": reason,
                    "benchmark_only_fallback": False,
                    "family_best_mean_rank_ic": best_mean,
                    "family_best_bootstrap_se": best_se,
                    "one_se_threshold": threshold,
                }
            )
            by_config[config_id] = output
            order.append(config_id)
        else:
            prior = str(by_config[config_id]["selection_reasons"])
            by_config[config_id]["selection_reasons"] = f"{prior}|{reason}"

    rows: list[dict[str, object]] = []
    for shortlist_order, config_id in enumerate(order, start=1):
        row = by_config[config_id]
        row["shortlist_order"] = shortlist_order
        rows.append(row)
    return pd.DataFrame(rows).loc[:, list(SHORTLIST_COLUMNS)]


def shortlist_all_families(
    config_metrics: pd.DataFrame,
    *,
    regularization_columns_by_family: Mapping[str, Sequence[str]],
    elastic_net_family: str = ELASTIC_NET_FAMILY,
    tolerance: float = TIE_TOLERANCE,
) -> pd.DataFrame:
    """Shortlist every supplied family in deterministic family-name order."""

    _require_columns(config_metrics, ("model_family",))
    outputs: list[pd.DataFrame] = []
    for model_family in sorted(str(value) for value in config_metrics["model_family"].unique()):
        if model_family not in regularization_columns_by_family:
            raise ValueError(
                f"No regularization tie-break tuple supplied for {model_family!r}"
            )
        selected = shortlist_family(
            config_metrics,
            model_family=model_family,
            regularization_columns=regularization_columns_by_family[model_family],
            elastic_net_family=elastic_net_family,
            tolerance=tolerance,
        )
        if not selected.empty:
            outputs.append(selected)
    if not outputs:
        return _empty_shortlist()
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["model_family", "shortlist_order"], kind="mergesort"
    ).reset_index(drop=True)
