#!/usr/bin/env python3
"""Fit one frozen V6.1 XGBoost component and score a separate live matrix."""

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
    result = np.empty(len(target), dtype=float)
    start = 0
    for size in _groups(months):
        stop = start + size
        query = target[start:stop]
        sorter = np.argsort(query, kind="mergesort")
        ordered = query[sorter]
        boundaries = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, size]
        ranks = np.empty(size, dtype=float)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            ranks[sorter[left:right]] = (left + right - 1) / 2.0
        result[start:stop] = np.minimum(np.floor(ranks * 32.0 / size), 31.0)
        start = stop
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-matrix", type=Path, required=True)
    parser.add_argument("--live-matrix", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--train-target", type=Path, required=True)
    parser.add_argument("--train-months", type=Path, required=True)
    parser.add_argument("--objective", choices=("pairwise", "continuous_rank"), required=True)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--importance-output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    historical = np.load(args.historical_matrix, mmap_mode="r")
    live = np.load(args.live_matrix, mmap_mode="r")
    train_index = np.load(args.train_index).astype(np.int64)
    feature_index = np.load(args.feature_index).astype(np.int64)
    target = np.load(args.train_target).astype(float)
    months = np.load(args.train_months, allow_pickle=False).astype(str)
    if len(train_index) != len(target) or len(target) != len(months):
        raise ValueError("XGBoost live training inputs are misaligned")
    train_values = historical[np.ix_(train_index, feature_index)]
    live_values = live[:, feature_index]
    parameters = json.loads(args.parameters_json)
    parameters.pop("early_stopping_rounds", None)
    pair_method = parameters.pop("pair_method", "topk")
    pairs_per_sample = int(parameters.pop("pairs_per_sample", 32))
    if args.objective == "pairwise":
        parameters.update(
            {
                "objective": "rank:ndcg",
                "eval_metric": "ndcg",
                "lambdarank_pair_method": pair_method,
                "lambdarank_num_pair_per_sample": pairs_per_sample,
            }
        )
        model = XGBRanker(random_state=args.seed, **parameters)
        model.fit(train_values, _relevance(target, months), group=_groups(months), verbose=False)
    else:
        parameters.update({"objective": "reg:pseudohubererror", "eval_metric": "mae"})
        model = XGBRegressor(random_state=args.seed, **parameters)
        model.fit(train_values, target, verbose=False)
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.prediction_output, model.predict(live_values))
    np.save(args.importance_output, model.feature_importances_)
    model.save_model(args.model_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
