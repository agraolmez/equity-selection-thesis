#!/usr/bin/env python3
"""Native XGBoost worker for V6 pairwise and continuous-rank selectors.

The project Python 3.13/XGBoost 3.2 wheel segfaults during ranker fitting on
this Mac.  This worker is invoked with the existing finance environment, whose
XGBoost 3.0.2 ranker is stable.  It receives only local NumPy files and never
opens research outcomes or credentials.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRanker, XGBRegressor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--forecast-index", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path)
    parser.add_argument("--train-target", type=Path, required=True)
    parser.add_argument("--group-sizes", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=("pairwise", "continuous_rank"),
        default="pairwise",
    )
    parser.add_argument("--parameters-json")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--importance-output", type=Path, required=True)
    args = parser.parse_args()

    matrix = np.load(args.matrix, mmap_mode="r")
    train_index = np.load(args.train_index)
    forecast_index = np.load(args.forecast_index)
    feature_index = (
        np.load(args.feature_index)
        if args.feature_index is not None
        else np.arange(matrix.shape[1], dtype=np.int64)
    )
    target = np.load(args.train_target)
    group_sizes = np.load(args.group_sizes).astype(int).tolist()
    if sum(group_sizes) != len(train_index) or len(target) != len(train_index):
        raise ValueError("XGBoost worker inputs are not aligned")
    parameters = (
        json.loads(args.parameters_json)
        if args.parameters_json
        else {
            "n_estimators": 180,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 20.0,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.02,
            "reg_lambda": 2.0,
            "tree_method": "hist",
            "n_jobs": 4,
        }
    )
    train_values = matrix[np.ix_(train_index, feature_index)]
    forecast_values = matrix[np.ix_(forecast_index, feature_index)]
    if args.objective == "pairwise":
        model = XGBRanker(
            objective="rank:pairwise",
            eval_metric="ndcg",
            random_state=args.seed,
            **parameters,
        )
        model.fit(train_values, target, group=group_sizes, verbose=False)
    else:
        model = XGBRegressor(
            objective="reg:pseudohubererror",
            eval_metric="mae",
            random_state=args.seed,
            **parameters,
        )
        model.fit(train_values, target, verbose=False)
    np.save(args.prediction_output, model.predict(forecast_values))
    np.save(args.importance_output, model.feature_importances_)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
