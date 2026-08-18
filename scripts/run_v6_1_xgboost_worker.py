#!/usr/bin/env python3
"""Stable XGBoost worker with V6.1 early stopping and refit modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRanker, XGBRegressor


def _groups(months: np.ndarray) -> list[int]:
    if len(months) == 0:
        return []
    boundaries = np.flatnonzero(months[1:] != months[:-1]) + 1
    return np.diff(np.r_[0, boundaries, len(months)]).astype(int).tolist()


def _relevance(target: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Map each query monotonically onto 32 integer relevance grades."""

    result = np.empty(len(target), dtype=float)
    start = 0
    for size in _groups(months):
        stop = start + size
        query_target = target[start:stop]
        sorter = np.argsort(query_target, kind="mergesort")
        sorted_target = query_target[sorter]
        boundaries = np.r_[0, np.flatnonzero(sorted_target[1:] != sorted_target[:-1]) + 1, size]
        ranks = np.empty(size, dtype=float)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            ranks[sorter[left:right]] = (left + right - 1) / 2.0
        result[start:stop] = np.minimum(np.floor(ranks * 32.0 / size), 31.0)
        start = stop
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--forecast-index", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--train-target", type=Path, required=True)
    parser.add_argument("--train-months", type=Path, required=True)
    parser.add_argument("--validation-index", type=Path)
    parser.add_argument("--validation-target", type=Path)
    parser.add_argument("--validation-months", type=Path)
    parser.add_argument("--objective", choices=("pairwise", "continuous_rank"), required=True)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--importance-output", type=Path, required=True)
    parser.add_argument("--best-iteration-output", type=Path, required=True)
    args = parser.parse_args()

    matrix = np.load(args.matrix, mmap_mode="r")
    train_index = np.load(args.train_index).astype(np.int64)
    forecast_index = np.load(args.forecast_index).astype(np.int64)
    feature_index = np.load(args.feature_index).astype(np.int64)
    target = np.load(args.train_target).astype(float)
    train_months = np.load(args.train_months, allow_pickle=False).astype(str)
    if len(train_index) != len(target) or len(target) != len(train_months):
        raise ValueError("V6.1 XGBoost training inputs are not aligned")
    parameters = json.loads(args.parameters_json)
    early_stopping_rounds = parameters.pop("early_stopping_rounds", None)
    pair_method = parameters.pop("pair_method", "topk")
    pairs_per_sample = parameters.pop("pairs_per_sample", 32)
    validation_index = (
        np.load(args.validation_index).astype(np.int64)
        if args.validation_index is not None
        else None
    )
    validation_target = (
        np.load(args.validation_target).astype(float)
        if args.validation_target is not None
        else None
    )
    validation_months = (
        np.load(args.validation_months, allow_pickle=False).astype(str)
        if args.validation_months is not None
        else None
    )
    train_values = matrix[np.ix_(train_index, feature_index)]
    forecast_values = matrix[np.ix_(forecast_index, feature_index)]
    fit_kwargs: dict[str, object] = {"verbose": False}
    if validation_index is not None:
        if validation_target is None or validation_months is None:
            raise ValueError("validation index requires target and month inputs")
        validation_values = matrix[np.ix_(validation_index, feature_index)]
        fit_kwargs["eval_set"] = [(validation_values, validation_target)]
    if args.objective == "pairwise":
        target = _relevance(target, train_months)
        parameters.update(
            {
                "objective": "rank:ndcg",
                "eval_metric": "ndcg",
                "lambdarank_pair_method": pair_method,
                "lambdarank_num_pair_per_sample": int(pairs_per_sample),
            }
        )
        if validation_index is not None:
            validation_target = _relevance(validation_target, validation_months)
            fit_kwargs["eval_set"] = [(validation_values, validation_target)]
            fit_kwargs["eval_group"] = [_groups(validation_months)]
        if early_stopping_rounds is not None and validation_index is not None:
            parameters["early_stopping_rounds"] = int(early_stopping_rounds)
        model = XGBRanker(random_state=args.seed, **parameters)
        model.fit(train_values, target, group=_groups(train_months), **fit_kwargs)
    else:
        parameters.update({"objective": "reg:pseudohubererror", "eval_metric": "mae"})
        if early_stopping_rounds is not None and validation_index is not None:
            parameters["early_stopping_rounds"] = int(early_stopping_rounds)
        model = XGBRegressor(random_state=args.seed, **parameters)
        model.fit(train_values, target, **fit_kwargs)
    best_iteration = getattr(model, "best_iteration", None)
    effective_trees = (
        int(best_iteration) + 1
        if best_iteration is not None
        else int(parameters["n_estimators"])
    )
    np.save(args.prediction_output, model.predict(forecast_values))
    np.save(args.importance_output, model.feature_importances_)
    args.best_iteration_output.write_text(
        json.dumps({"effective_trees": effective_trees}, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
