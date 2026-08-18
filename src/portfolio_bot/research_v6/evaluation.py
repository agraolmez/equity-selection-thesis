"""Compact selector metrics used by the lean V6 screen."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def monthly_selector_metrics(
    frame: pd.DataFrame,
    *,
    score_columns: Sequence[str],
    target_column: str,
    economic_column: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for month, group in frame.groupby("holding_month", observed=True, sort=True):
        target = pd.to_numeric(group[target_column], errors="coerce").to_numpy(float)
        economic = (
            pd.to_numeric(group[economic_column], errors="coerce").to_numpy(float)
            if economic_column is not None
            else target
        )
        for score_column in score_columns:
            score = pd.to_numeric(group[score_column], errors="coerce").to_numpy(float)
            valid = np.isfinite(score) & np.isfinite(target) & np.isfinite(economic)
            if valid.sum() < 10:
                continue
            score_rank = pd.Series(score[valid]).rank(method="average").to_numpy(float)
            target_rank = pd.Series(target[valid]).rank(method="average").to_numpy(float)
            order = np.argsort(score[valid], kind="mergesort")
            tail = max(int(np.floor(valid.sum() * 0.10)), 1)
            top = float(np.mean(economic[valid][order[-tail:]]))
            bottom = float(np.mean(economic[valid][order[:tail]]))
            decile_means = []
            for positions in np.array_split(order, 10):
                decile_means.append(float(np.mean(economic[valid][positions])))
            monotonicity = _correlation(
                np.arange(1.0, 11.0), np.asarray(decile_means, dtype=float)
            )
            rows.append(
                {
                    "holding_month": str(month),
                    "model_id": score_column,
                    "observations": int(valid.sum()),
                    "rank_ic": _correlation(score_rank, target_rank),
                    "economic_top_decile_mean": top,
                    "economic_bottom_decile_mean": bottom,
                    "economic_top_minus_bottom": top - bottom,
                    "decile_monotonicity": monotonicity,
                }
            )
    return pd.DataFrame(rows)


def _hac_mean(values: pd.Series, lags: int) -> tuple[float, float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    count = len(clean)
    if count < 3:
        return math.nan, math.nan, math.nan, math.nan
    centered = clean - clean.mean()
    long_run = float(np.dot(centered, centered) / count)
    for lag in range(1, min(lags, count - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        long_run += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run, 0.0) / count)
    mean = float(clean.mean())
    statistic = mean / standard_error if standard_error > 0.0 else math.nan
    return mean, standard_error, statistic, mean - 1.96 * standard_error


def summarize_selector_metrics(
    monthly: pd.DataFrame, *, hac_lags: int = 6
) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for model_id, group in monthly.groupby("model_id", observed=True, sort=True):
        ordered = group.sort_values("holding_month", kind="mergesort")
        mean, standard_error, statistic, lower = _hac_mean(
            ordered["rank_ic"], hac_lags
        )
        rolling_window = min(24, max(12, len(ordered) // 2))
        rolling = ordered["rank_ic"].rolling(rolling_window).mean().dropna()
        midpoint = max(len(ordered) // 2, 1)
        rows.append(
            {
                "model_id": str(model_id),
                "months": int(ordered["holding_month"].nunique()),
                "mean_rank_ic": mean,
                "median_rank_ic": float(ordered["rank_ic"].median()),
                "rank_ic_standard_deviation": float(ordered["rank_ic"].std(ddof=1)),
                "rank_ic_hac_standard_error": standard_error,
                "rank_ic_hac_t": statistic,
                "rank_ic_hac_95_lower": lower,
                "positive_ic_fraction": float((ordered["rank_ic"] > 0.0).mean()),
                "early_mean_rank_ic": float(ordered["rank_ic"].iloc[:midpoint].mean()),
                "recent_mean_rank_ic": float(ordered["rank_ic"].iloc[midpoint:].mean()),
                "rolling_window_months": rolling_window,
                "minimum_rolling_mean_rank_ic": (
                    float(rolling.min()) if not rolling.empty else math.nan
                ),
                "mean_economic_top_decile": float(
                    ordered["economic_top_decile_mean"].mean()
                ),
                "mean_economic_top_minus_bottom": float(
                    ordered["economic_top_minus_bottom"].mean()
                ),
                "mean_decile_monotonicity": float(
                    ordered["decile_monotonicity"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_rank_ic", ascending=False, kind="mergesort"
    )


__all__ = ["monthly_selector_metrics", "summarize_selector_metrics"]
