#!/usr/bin/env python3
"""Run the lean V6 selector screen before any portfolio construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from portfolio_bot.research_v6.data import (  # noqa: E402
    TARGET_COLUMNS,
    build_target_panel,
    load_feature_registry,
    load_predictor_panel,
)
from portfolio_bot.research_v6.evaluation import (  # noqa: E402
    monthly_selector_metrics,
    summarize_selector_metrics,
)
from portfolio_bot.research_v6.models import (  # noqa: E402
    dynamic_ensemble,
    fit_elastic_net,
    fit_pairwise_mlp,
    predict_mlp,
    rank_within_month,
)
from portfolio_bot.research_v6.preprocessing import encode_panel  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts/research_v6/selector_smoke_v2"
TARGET_IDS = {
    "capm": "target_rank_capm",
    "ff5_momentum": "target_rank_ff5_momentum",
    "long_net": "target_rank_long_net",
    "short_net": "target_rank_short_net",
}
FINANCE_PYTHON = Path("/opt/anaconda3/envs/finance/bin/python")
XGBOOST_WORKER = ROOT / "scripts/run_v6_xgboost_worker.py"


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _feature_importance(
    feature_names: tuple[str, ...], xgb_importance: np.ndarray, elastic_model
) -> pd.DataFrame:
    xgb = np.asarray(xgb_importance, dtype=float)
    elastic = np.abs(np.asarray(elastic_model.coef_, dtype=float))
    return pd.DataFrame(
        {
            "feature_name": feature_names,
            "xgboost_gain_importance": xgb,
            "elastic_net_absolute_coefficient": elastic,
        }
    ).sort_values(
        ["xgboost_gain_importance", "elastic_net_absolute_coefficient"],
        ascending=False,
        kind="mergesort",
    )


def _xgboost_worker_prediction(
    *,
    matrix_path: Path,
    train: np.ndarray,
    forecast: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    target_output: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not FINANCE_PYTHON.is_file():
        raise FileNotFoundError(f"stable XGBoost runtime not found: {FINANCE_PYTHON}")
    train_index = np.flatnonzero(train).astype(np.int64)
    forecast_index = np.flatnonzero(forecast).astype(np.int64)
    group_sizes = (
        pd.Series(months[train], dtype="string")
        .groupby(pd.Series(months[train], dtype="string"), sort=False)
        .size()
        .to_numpy(np.int64)
    )
    paths = {
        "train_index": target_output / "xgb_train_index.npy",
        "forecast_index": target_output / "xgb_forecast_index.npy",
        "target": target_output / "xgb_train_target.npy",
        "groups": target_output / "xgb_group_sizes.npy",
        "prediction": target_output / "xgb_prediction.npy",
        "importance": target_output / "xgb_importance.npy",
    }
    np.save(paths["train_index"], train_index)
    np.save(paths["forecast_index"], forecast_index)
    np.save(paths["target"], target[train])
    np.save(paths["groups"], group_sizes)
    subprocess.run(
        [
            str(FINANCE_PYTHON),
            str(XGBOOST_WORKER),
            "--matrix",
            str(matrix_path),
            "--train-index",
            str(paths["train_index"]),
            "--forecast-index",
            str(paths["forecast_index"]),
            "--train-target",
            str(paths["target"]),
            "--group-sizes",
            str(paths["groups"]),
            "--prediction-output",
            str(paths["prediction"]),
            "--importance-output",
            str(paths["importance"]),
        ],
        cwd=ROOT,
        check=True,
    )
    return np.load(paths["prediction"]), np.load(paths["importance"])


def _run_target(
    *,
    target_id: str,
    target_column: str,
    matrix: np.ndarray,
    keys_and_targets: pd.DataFrame,
    feature_names: tuple[str, ...],
    matrix_path: Path,
    output: Path,
) -> None:
    target_output = output / target_id
    target_output.mkdir(parents=True, exist_ok=True)
    completion = target_output / "completion.json"
    if completion.is_file():
        print(f"{target_id}: compatible completion exists; skipping", flush=True)
        return
    y = pd.to_numeric(keys_and_targets[target_column], errors="coerce").to_numpy(float)
    months = keys_and_targets["holding_month"].astype(str).to_numpy()
    train = (months >= "2015-01") & (months <= "2019-11") & np.isfinite(y)
    forecast = (months >= "2020-01") & (months <= "2025-08") & np.isfinite(y)
    if train.sum() < 10_000 or forecast.sum() < 10_000:
        raise RuntimeError(f"{target_id}: insufficient train or forecast observations")
    x_train = matrix[train]
    y_train = y[train]
    train_months = months[train]
    print(
        f"{target_id}: fitting {train.sum():,} rows, forecasting {forecast.sum():,}",
        flush=True,
    )
    xgb_prediction, xgb_importance = _xgboost_worker_prediction(
        matrix_path=matrix_path,
        train=train,
        forecast=forecast,
        target=y,
        months=months,
        target_output=target_output,
    )
    mlp_model = fit_pairwise_mlp(x_train, y_train, train_months)
    elastic_model = fit_elastic_net(x_train, y_train)

    selected = keys_and_targets.loc[
        forecast,
        ["holding_month", "signal_date", "permno", "permco", target_column],
    ].copy()
    selected["xgboost_rank"] = rank_within_month(
        xgb_prediction, selected["holding_month"]
    )
    selected["mlp_rank"] = rank_within_month(
        predict_mlp(mlp_model, matrix[forecast]), selected["holding_month"]
    )
    selected["elastic_net_rank"] = rank_within_month(
        elastic_model.predict(matrix[forecast]), selected["holding_month"]
    )
    selected = dynamic_ensemble(selected, target_column=target_column)
    selected["period"] = np.where(
        selected["holding_month"].le("2023-12"), "validation", "historical_test"
    )
    metrics = monthly_selector_metrics(
        selected,
        score_columns=(
            "xgboost_rank",
            "mlp_rank",
            "ensemble_rank",
            "elastic_net_rank",
        ),
        target_column=target_column,
    ).merge(
        selected.loc[:, ["holding_month", "period"]].drop_duplicates(),
        on="holding_month",
        validate="many_to_one",
    )
    summary = pd.concat(
        [
            summarize_selector_metrics(group).assign(period=period)
            for period, group in metrics.groupby("period", observed=True, sort=True)
        ],
        ignore_index=True,
    )
    importance = _feature_importance(feature_names, xgb_importance, elastic_model)
    _atomic_parquet(selected, target_output / "predictions.parquet")
    _atomic_parquet(metrics, target_output / "monthly_metrics.parquet")
    _atomic_parquet(summary, target_output / "summary.parquet")
    _atomic_parquet(importance, target_output / "feature_importance.parquet")
    _atomic_json(
        {
            "schema": "research_v6_selector_smoke_completion_v2",
            "target_id": target_id,
            "target_column": target_column,
            "train_start": "2015-01",
            "train_end": "2019-11",
            "validation_start": "2020-01",
            "validation_end": "2023-12",
            "historical_test_start": "2024-01",
            "historical_test_end": "2025-08",
            "december_2019_predictors_pending": True,
            "news_features_pending": True,
            "portfolio_constructed": False,
            "train_rows": int(train.sum()),
            "forecast_rows": int(forecast.sum()),
            "encoded_feature_count": int(matrix.shape[1]),
        },
        completion,
    )
    print(summary.to_string(index=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=tuple(TARGET_IDS),
        default=tuple(TARGET_IDS),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    registry = load_feature_registry()
    predictors = load_predictor_panel(registry)
    targets = build_target_panel(predictors)
    modeling = predictors.merge(
        targets,
        on=["holding_month", "signal_date", "permno", "permco"],
        how="inner",
        validate="one_to_one",
    )
    modeling = modeling.loc[modeling["holding_month"].between("2015-01", "2025-08")]
    encoded = encode_panel(modeling, registry)
    matrix_path = output / "encoded_values.npy"
    if not matrix_path.is_file():
        temporary_matrix = matrix_path.with_suffix(".npy.tmp")
        with temporary_matrix.open("wb") as destination:
            np.save(destination, encoded.values)
        temporary_matrix.replace(matrix_path)
    target_values = encoded.keys.merge(
        modeling.loc[:, ["holding_month", "signal_date", "permno", "permco", *TARGET_COLUMNS]],
        on=["holding_month", "signal_date", "permno", "permco"],
        how="left",
        validate="one_to_one",
    )
    if len(target_values) != len(encoded.values):
        raise RuntimeError("encoded rows do not align with target rows")
    _atomic_json(
        {
            "schema": "research_v6_selector_smoke_run_v2",
            "raw_feature_count": int(len(registry)),
            "encoded_feature_count": int(encoded.values.shape[1]),
            "modeling_rows": int(len(modeling)),
            "modeling_months": int(modeling["holding_month"].nunique()),
            "december_2019_predictors_pending": True,
            "news_features_pending": True,
            "exploratory_not_thesis_grade": True,
        },
        output / "run_manifest.json",
    )
    for target_id in args.targets:
        _run_target(
            target_id=target_id,
            target_column=TARGET_IDS[target_id],
            matrix=encoded.values,
            keys_and_targets=target_values,
            feature_names=encoded.feature_names,
            matrix_path=matrix_path,
            output=output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
