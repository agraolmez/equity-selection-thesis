"""Leakage-safe Stage 9 selector scoring and development metrics.

This module contains only deterministic numerical and tabular primitives.  It
does not read data, fit a selector, select a portfolio, or inspect data outside
the development period.  Raw stochastic forecasts are averaged across the
frozen seeds before the monthly score is formed.  Every long-eligible row stays
in that score cross-section even when its realized outcome is unavailable;
only the correlation calculation excludes invalid outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEVELOPMENT_SEEDS = (11, 23, 47)
DEVELOPMENT_FOLD_IDS = (
    "dev_fold_1",
    "dev_fold_2",
    "dev_fold_3",
    "dev_fold_4",
)
BOOTSTRAP_BLOCK_MONTHS = 6
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_MASTER_SEED = 20_260_714

SCORE_KEY_COLUMNS = (
    "model_family",
    "config_id",
    "fold_id",
    "signal_date",
    "holding_month",
    "permno",
)
SCORE_GROUP_COLUMNS = SCORE_KEY_COLUMNS[:-1]
CONFIG_KEY_COLUMNS = ("model_family", "config_id")

CORRELATION_OK = "OK"
CORRELATION_INSUFFICIENT = "INSUFFICIENT_OBSERVATIONS"
CORRELATION_NONFINITE = "NONFINITE_INPUT"
CORRELATION_CONSTANT_LEFT = "CONSTANT_LEFT"
CORRELATION_CONSTANT_RIGHT = "CONSTANT_RIGHT"
CORRELATION_CONSTANT_BOTH = "CONSTANT_BOTH"
CORRELATION_NONFINITE_RESULT = "NONFINITE_RESULT"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _finite_float_vector(values: Sequence[object], *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return vector


def centered_average_rank(values: Sequence[object]) -> np.ndarray:
    """Return frozen centered average ranks for a complete score cross-section.

    For ``N`` finite forecasts, the score is
    ``((average_rank - 0.5) / N) - 0.5``.  Missing forecasts are not silently
    removed because doing so would change the long-eligible scoring universe.
    """

    vector = _finite_float_vector(values, name="forecast values")
    if vector.size == 0:
        return np.asarray([], dtype=np.float64)
    ranks = (
        pd.Series(vector, copy=False)
        .rank(method="average", ascending=True)
        .to_numpy(dtype=np.float64)
    )
    scores = ((ranks - 0.5) / float(vector.size)) - 0.5
    scores[scores == 0.0] = 0.0
    return scores


def center_monthly_forecasts(
    forecasts: pd.DataFrame,
    *,
    raw_forecast_column: str = "raw_forecast_mean",
) -> pd.DataFrame:
    """Center already-averaged raw forecasts within every validation month.

    The input must contain exactly one row per score key and only rows belonging
    to the validation scoring universe.  If a ``long_eligible`` column is
    supplied, false rows are removed before validation and scoring.
    """

    _require_columns(forecasts, (*SCORE_KEY_COLUMNS, raw_forecast_column))
    result = forecasts.copy()
    if "long_eligible" in result.columns:
        if result["long_eligible"].isna().any():
            raise ValueError("long_eligible may not be null")
        result = result.loc[result["long_eligible"].astype(bool)].copy()
    if result.empty:
        raise ValueError("No long-eligible forecasts were supplied")
    if result.duplicated(list(SCORE_KEY_COLUMNS)).any():
        raise ValueError("Forecast score keys must be unique")

    raw = pd.to_numeric(result[raw_forecast_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(raw).all():
        raise ValueError("Every long-eligible raw forecast must be finite")
    result[raw_forecast_column] = raw
    result["score"] = np.nan

    for _, positions in result.groupby(
        list(SCORE_GROUP_COLUMNS), sort=True, observed=True
    ).indices.items():
        indices = np.asarray(positions, dtype=np.int64)
        result.iloc[
            indices, result.columns.get_loc("score")
        ] = centered_average_rank(result.iloc[indices][raw_forecast_column])

    return result.sort_values(list(SCORE_KEY_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )


def score_seed_averaged_forecasts(
    seed_forecasts: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] = DEVELOPMENT_SEEDS,
    seed_column: str = "seed",
    raw_forecast_column: str = "raw_forecast",
) -> pd.DataFrame:
    """Average raw seed forecasts first and then form monthly scores.

    This ordering is scientifically material: seed-specific scores or ICs are
    never averaged.  Every security/config/fold/month key must have exactly the
    expected seed set.
    """

    _require_columns(
        seed_forecasts,
        (*SCORE_KEY_COLUMNS, seed_column, raw_forecast_column),
    )
    expected = tuple(sorted(int(seed) for seed in expected_seeds))
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_seeds must be a nonempty unique sequence")

    frame = seed_forecasts.copy()
    if "long_eligible" in frame.columns:
        if frame["long_eligible"].isna().any():
            raise ValueError("long_eligible may not be null")
        frame = frame.loc[frame["long_eligible"].astype(bool)].copy()
    if frame.empty:
        raise ValueError("No long-eligible seed forecasts were supplied")
    if frame.duplicated([*SCORE_KEY_COLUMNS, seed_column]).any():
        raise ValueError("Seed forecast keys must be unique")

    try:
        frame[seed_column] = frame[seed_column].astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Seeds must be integers") from exc
    raw = pd.to_numeric(frame[raw_forecast_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(raw).all():
        raise ValueError("Every long-eligible raw seed forecast must be finite")
    frame[raw_forecast_column] = raw

    records: list[dict[str, object]] = []
    for key, group in frame.groupby(
        list(SCORE_KEY_COLUMNS), sort=True, observed=True
    ):
        actual = tuple(sorted(int(value) for value in group[seed_column]))
        if actual != expected:
            raise ValueError(
                f"Forecast key {key!r} has seeds {actual}; expected {expected}"
            )
        record = dict(zip(SCORE_KEY_COLUMNS, key, strict=True))
        record["raw_forecast_mean"] = float(
            np.mean(group[raw_forecast_column].to_numpy(dtype=np.float64))
        )
        record["seed_count"] = len(expected)
        records.append(record)

    averaged = pd.DataFrame.from_records(records)
    return center_monthly_forecasts(averaged)


@dataclass(frozen=True)
class CorrelationResult:
    value: float
    status: str
    observations: int


def correlation_with_status(
    left: Sequence[object],
    right: Sequence[object],
    *,
    method: str,
    minimum_observations: int = 2,
) -> CorrelationResult:
    """Calculate Pearson or Spearman correlation with fail-closed status.

    Insufficient, nonfinite, or constant inputs return a literal zero together
    with a diagnostic status; no month disappears from the equal-month mean.
    """

    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    if (
        isinstance(minimum_observations, bool)
        or not isinstance(minimum_observations, int)
        or minimum_observations < 2
    ):
        raise ValueError("minimum_observations must be an integer of at least 2")
    try:
        x = np.asarray(left, dtype=np.float64)
        y = np.asarray(right, dtype=np.float64)
    except (TypeError, ValueError):
        return CorrelationResult(0.0, CORRELATION_NONFINITE, 0)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("Correlation inputs must be equal-length vectors")
    observations = int(x.size)
    if observations < minimum_observations:
        return CorrelationResult(0.0, CORRELATION_INSUFFICIENT, observations)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return CorrelationResult(0.0, CORRELATION_NONFINITE, observations)
    if method == "spearman":
        x = pd.Series(x, copy=False).rank(method="average").to_numpy(dtype=float)
        y = pd.Series(y, copy=False).rank(method="average").to_numpy(dtype=float)

    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    x_ss = float(np.dot(x_centered, x_centered))
    y_ss = float(np.dot(y_centered, y_centered))
    x_constant = x_ss == 0.0
    y_constant = y_ss == 0.0
    if x_constant and y_constant:
        return CorrelationResult(0.0, CORRELATION_CONSTANT_BOTH, observations)
    if x_constant:
        return CorrelationResult(0.0, CORRELATION_CONSTANT_LEFT, observations)
    if y_constant:
        return CorrelationResult(0.0, CORRELATION_CONSTANT_RIGHT, observations)

    value = float(np.dot(x_centered, y_centered) / math.sqrt(x_ss * y_ss))
    if not math.isfinite(value):
        return CorrelationResult(0.0, CORRELATION_NONFINITE_RESULT, observations)
    # Protect against harmless floating overshoot while preserving the value.
    value = min(1.0, max(-1.0, value))
    if value == 0.0:
        value = 0.0
    return CorrelationResult(value, CORRELATION_OK, observations)


def monthly_selector_metrics(
    scored_forecasts: pd.DataFrame,
    *,
    residual_column: str = "unranked_residual_return",
    outcome_valid_column: str | None = None,
) -> pd.DataFrame:
    """Return one authoritative rank IC and descriptive linear IC per month.

    The score is assumed to have been constructed before outcomes were joined.
    Nonfinite residual outcomes and rows explicitly marked invalid do not enter
    either correlation, but remain counted in ``scored_rows``.
    """

    required = (
        *SCORE_KEY_COLUMNS,
        "raw_forecast_mean",
        "score",
        residual_column,
    )
    _require_columns(scored_forecasts, required)
    if outcome_valid_column is not None:
        _require_columns(scored_forecasts, (outcome_valid_column,))
    if scored_forecasts.duplicated(list(SCORE_KEY_COLUMNS)).any():
        raise ValueError("Scored forecast keys must be unique")

    rows: list[dict[str, object]] = []
    for group_key, group in scored_forecasts.groupby(
        list(SCORE_GROUP_COLUMNS), sort=True, observed=True
    ):
        residual = pd.to_numeric(group[residual_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        valid = np.isfinite(residual)
        if outcome_valid_column is not None:
            valid_flag = group[outcome_valid_column]
            if valid_flag.isna().any():
                raise ValueError(f"{outcome_valid_column} may not be null")
            valid &= valid_flag.astype(bool).to_numpy()

        score = pd.to_numeric(group["score"], errors="coerce").to_numpy(dtype=float)
        raw = pd.to_numeric(group["raw_forecast_mean"], errors="coerce").to_numpy(
            dtype=float
        )
        rank_result = correlation_with_status(
            score[valid], residual[valid], method="spearman"
        )
        linear_result = correlation_with_status(
            raw[valid], residual[valid], method="pearson"
        )
        record = dict(zip(SCORE_GROUP_COLUMNS, group_key, strict=True))
        record.update(
            {
                "scored_rows": int(len(group)),
                "evaluation_rows": int(np.sum(valid)),
                "excluded_outcome_rows": int(len(group) - np.sum(valid)),
                "rank_ic": rank_result.value,
                "rank_ic_status": rank_result.status,
                "linear_ic": linear_result.value,
                "linear_ic_status": linear_result.status,
            }
        )
        rows.append(record)

    return pd.DataFrame.from_records(rows).sort_values(
        list(SCORE_GROUP_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)


@dataclass(frozen=True)
class MovingBlockBootstrapPlan:
    """One shared non-circular overlapping moving-block resample plan."""

    months: tuple[object, ...]
    indices: np.ndarray
    block_length: int
    replicates: int
    seed: int


def make_moving_block_bootstrap_plan(
    months: Sequence[object],
    *,
    block_length: int = BOOTSTRAP_BLOCK_MONTHS,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_MASTER_SEED,
) -> MovingBlockBootstrapPlan:
    """Build an order-independent PCG64 moving-block plan.

    Months are canonicalized into sorted order before any random draw.  Blocks
    are sampled from every overlapping start ``0..N-L`` and never wrap from the
    end of the sample to its beginning.
    """

    if (
        isinstance(block_length, bool)
        or not isinstance(block_length, int)
        or block_length <= 0
    ):
        raise ValueError("block_length must be a positive integer")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("replicates must be an integer of at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    month_index = pd.Index(months)
    if month_index.hasnans:
        raise ValueError("Bootstrap months may not be null")
    if month_index.has_duplicates:
        raise ValueError("Bootstrap months must be unique")
    try:
        ordered_months = tuple(month_index.sort_values().tolist())
    except (TypeError, ValueError) as exc:
        raise ValueError("Bootstrap months must have one sortable type") from exc
    count = len(ordered_months)
    if count < block_length:
        raise ValueError("The month count must be at least the block length")

    blocks_per_replication = math.ceil(count / block_length)
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(
        0,
        count - block_length + 1,
        size=(replicates, blocks_per_replication),
        dtype=np.int32,
    )
    offsets = np.arange(block_length, dtype=np.int32)
    indices = (starts[:, :, None] + offsets).reshape(replicates, -1)[:, :count]
    indices.setflags(write=False)
    return MovingBlockBootstrapPlan(
        months=ordered_months,
        indices=indices,
        block_length=block_length,
        replicates=replicates,
        seed=seed,
    )


def bootstrap_equal_month_mean_se(
    month_values: pd.DataFrame,
    plan: MovingBlockBootstrapPlan,
    *,
    month_column: str = "holding_month",
    value_column: str = "rank_ic",
) -> float:
    """Return the sample SD of shared-plan bootstrap equal-month means."""

    _require_columns(month_values, (month_column, value_column))
    if month_values[month_column].duplicated().any():
        raise ValueError("Bootstrap input must have one row per month")
    by_month = month_values.set_index(month_column)[value_column]
    if set(by_month.index) != set(plan.months):
        raise ValueError("Every configuration must contain exactly the plan months")
    values = pd.to_numeric(by_month.reindex(plan.months), errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap monthly values must be finite")
    resampled_means = np.mean(values[plan.indices], axis=1, dtype=np.float64)
    result = float(np.std(resampled_means, ddof=1))
    if not math.isfinite(result):
        raise ValueError("Bootstrap standard error is nonfinite")
    return 0.0 if result == 0.0 else result


def aggregate_selector_metrics(
    monthly_metrics: pd.DataFrame,
    *,
    plan: MovingBlockBootstrapPlan,
    expected_fold_ids: Sequence[str] = DEVELOPMENT_FOLD_IDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return tidy fold and configuration metrics using equal month mass."""

    _require_columns(
        monthly_metrics,
        (
            *CONFIG_KEY_COLUMNS,
            "fold_id",
            "holding_month",
            "rank_ic",
            "linear_ic",
        ),
    )
    if monthly_metrics.duplicated(
        [*CONFIG_KEY_COLUMNS, "holding_month"]
    ).any():
        raise ValueError("Each configuration must have one metric row per month")
    expected_folds = tuple(expected_fold_ids)
    if len(expected_folds) != 4 or len(set(expected_folds)) != 4:
        raise ValueError("The frozen development gate requires four unique folds")

    fold_rows: list[dict[str, object]] = []
    config_rows: list[dict[str, object]] = []
    for config_key, config in monthly_metrics.groupby(
        list(CONFIG_KEY_COLUMNS), sort=True, observed=True
    ):
        actual_folds = tuple(sorted(str(value) for value in config["fold_id"].unique()))
        if actual_folds != tuple(sorted(expected_folds)):
            raise ValueError(
                f"Configuration {config_key!r} has folds {actual_folds}; "
                f"expected {tuple(sorted(expected_folds))}"
            )
        if set(config["holding_month"]) != set(plan.months):
            raise ValueError(
                f"Configuration {config_key!r} does not cover the shared plan months"
            )

        config_fold_means: list[float] = []
        for fold_id, fold in config.groupby("fold_id", sort=True, observed=True):
            rank_values = pd.to_numeric(fold["rank_ic"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            linear_values = pd.to_numeric(
                fold["linear_ic"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            if not np.isfinite(rank_values).all() or not np.isfinite(
                linear_values
            ).all():
                raise ValueError("Monthly IC values must be finite")
            rank_mean = float(np.mean(rank_values))
            config_fold_means.append(rank_mean)
            fold_rows.append(
                {
                    "model_family": config_key[0],
                    "config_id": config_key[1],
                    "fold_id": fold_id,
                    "months": int(len(fold)),
                    "mean_rank_ic": rank_mean,
                    "mean_linear_ic": float(np.mean(linear_values)),
                }
            )

        rank_values = pd.to_numeric(config["rank_ic"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        linear_values = pd.to_numeric(
            config["linear_ic"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        mean_rank_ic = float(np.mean(rank_values))
        positive_fold_count = int(sum(value > 0.0 for value in config_fold_means))
        fold_dispersion = float(np.std(config_fold_means, ddof=1))
        config_rows.append(
            {
                "model_family": config_key[0],
                "config_id": config_key[1],
                "months": int(len(config)),
                "folds": len(config_fold_means),
                "mean_rank_ic": mean_rank_ic,
                "mean_linear_ic": float(np.mean(linear_values)),
                "rank_ic_bootstrap_se": bootstrap_equal_month_mean_se(
                    config,
                    plan,
                    value_column="rank_ic",
                ),
                "linear_ic_bootstrap_se": bootstrap_equal_month_mean_se(
                    config,
                    plan,
                    value_column="linear_ic",
                ),
                "positive_fold_count": positive_fold_count,
                "fold_dispersion": fold_dispersion,
                "gate_pass": bool(
                    mean_rank_ic > 0.0 and positive_fold_count >= 3
                ),
            }
        )

    folds = pd.DataFrame.from_records(fold_rows).sort_values(
        [*CONFIG_KEY_COLUMNS, "fold_id"], kind="mergesort"
    ).reset_index(drop=True)
    configs = pd.DataFrame.from_records(config_rows).sort_values(
        list(CONFIG_KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    return folds, configs


@dataclass(frozen=True)
class SelectorMetricTables:
    monthly: pd.DataFrame
    fold: pd.DataFrame
    config: pd.DataFrame
    bootstrap_plan: MovingBlockBootstrapPlan


def evaluate_selector_metrics(
    scored_forecasts: pd.DataFrame,
    *,
    residual_column: str = "unranked_residual_return",
    outcome_valid_column: str | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_MASTER_SEED,
) -> SelectorMetricTables:
    """Evaluate all configurations with one shared Stage 9 resample plan."""

    monthly = monthly_selector_metrics(
        scored_forecasts,
        residual_column=residual_column,
        outcome_valid_column=outcome_valid_column,
    )
    plan = make_moving_block_bootstrap_plan(
        monthly["holding_month"].drop_duplicates().tolist(),
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    folds, configs = aggregate_selector_metrics(monthly, plan=plan)
    return SelectorMetricTables(
        monthly=monthly,
        fold=folds,
        config=configs,
        bootstrap_plan=plan,
    )
