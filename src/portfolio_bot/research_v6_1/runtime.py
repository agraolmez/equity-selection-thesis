"""Causal, atomic V6.1 walk-forward runtime."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from portfolio_bot.research_v6.models import fit_elastic_net, rank_within_month
from portfolio_bot.research_v6.selector_runtime import (
    atomic_json,
    atomic_parquet,
    canonical_sha256,
    is_refit_month,
    is_retune_month,
    mean_monthly_rank_ic,
)
from portfolio_bot.research_v6_1.models import fit_mlp, predict_mlp


ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "scripts/run_v6_1_xgboost_worker.py"
STABLE_PYTHON = Path("/opt/anaconda3/envs/finance/bin/python")


def _training_indices(
    rows: pd.DataFrame,
    target: np.ndarray,
    *,
    forecast_month: str,
    signal_date: pd.Timestamp,
    history: str,
    history_months: int | None,
) -> np.ndarray:
    month = rows["holding_month"].astype(str)
    available = pd.to_datetime(rows["label_available_date"], errors="coerce")
    eligible = month.lt(forecast_month) & available.le(signal_date) & np.isfinite(target)
    if history == "rolling":
        observed = sorted(month.loc[eligible].unique())
        if not history_months or history_months < 24:
            raise ValueError("rolling V6.1 history requires at least 24 months")
        eligible &= month.isin(set(observed[-history_months:]))
    elif history != "expanding":
        raise ValueError(f"unknown V6.1 history policy: {history}")
    return np.flatnonzero(eligible.to_numpy()).astype(np.int64)


def _tuning_split(
    rows: pd.DataFrame, train_index: np.ndarray, months: int
) -> tuple[np.ndarray, np.ndarray]:
    observed = rows.iloc[train_index]["holding_month"].astype(str)
    unique = sorted(observed.unique())
    if len(unique) <= months + 12:
        raise ValueError("V6.1 tuning split has insufficient history")
    holdout = observed.isin(set(unique[-months:])).to_numpy()
    return train_index[~holdout], train_index[holdout]


def _profile_id(profile: Mapping[str, Any]) -> str:
    return str(profile.get("profile_id") or canonical_sha256(profile)[:12])


def _run_xgboost(
    *,
    matrix: np.ndarray,
    feature_index: np.ndarray,
    train_index: np.ndarray,
    forecast_index: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    objective: str,
    parameters: Mapping[str, Any],
    seed: int,
    validation_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    filename = getattr(matrix, "filename", None)
    if filename is None:
        raise ValueError("V6.1 XGBoost requires a memory-mapped input matrix")
    with tempfile.TemporaryDirectory(prefix="v6_1_xgb_") as temporary_name:
        temporary = Path(temporary_name)
        paths = {
            "train": temporary / "train.npy",
            "forecast": temporary / "forecast.npy",
            "features": temporary / "features.npy",
            "target": temporary / "target.npy",
            "months": temporary / "months.npy",
            "prediction": temporary / "prediction.npy",
            "importance": temporary / "importance.npy",
            "best": temporary / "best.json",
        }
        np.save(paths["train"], train_index)
        np.save(paths["forecast"], forecast_index)
        np.save(paths["features"], feature_index)
        np.save(paths["target"], target[train_index])
        np.save(paths["months"], months[train_index].astype("U7"))
        command = [
            str(STABLE_PYTHON), str(WORKER),
            "--matrix", str(filename),
            "--train-index", str(paths["train"]),
            "--forecast-index", str(paths["forecast"]),
            "--feature-index", str(paths["features"]),
            "--train-target", str(paths["target"]),
            "--train-months", str(paths["months"]),
            "--objective", objective,
            "--parameters-json", json.dumps(dict(parameters), sort_keys=True),
            "--seed", str(int(seed)),
            "--prediction-output", str(paths["prediction"]),
            "--importance-output", str(paths["importance"]),
            "--best-iteration-output", str(paths["best"]),
        ]
        if validation_index is not None:
            paths.update(
                {
                    "validation": temporary / "validation.npy",
                    "validation_target": temporary / "validation_target.npy",
                    "validation_months": temporary / "validation_months.npy",
                }
            )
            np.save(paths["validation"], validation_index)
            np.save(paths["validation_target"], target[validation_index])
            np.save(paths["validation_months"], months[validation_index].astype("U7"))
            command.extend(
                [
                    "--validation-index", str(paths["validation"]),
                    "--validation-target", str(paths["validation_target"]),
                    "--validation-months", str(paths["validation_months"]),
                ]
            )
        subprocess.run(command, cwd=ROOT, check=True)
        best = json.loads(paths["best"].read_text())
        return (
            np.load(paths["prediction"]).astype(float),
            np.load(paths["importance"]).astype(float),
            int(best["effective_trees"]),
        )


def _fit_predict(
    *,
    family: str,
    objective: str,
    profile: Mapping[str, Any],
    shared: Mapping[str, Any],
    matrix: np.ndarray,
    feature_index: np.ndarray,
    train_index: np.ndarray,
    forecast_index: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    seeds: Sequence[int],
) -> tuple[np.ndarray, np.ndarray | None, dict[int, np.ndarray]]:
    predictions: list[np.ndarray] = []
    importances: list[np.ndarray] = []
    if family == "xgboost":
        parameters = {**dict(profile), **dict(shared)}
        parameters.pop("profile_id", None)
        for seed in seeds:
            prediction, importance, _ = _run_xgboost(
                matrix=matrix,
                feature_index=feature_index,
                train_index=train_index,
                forecast_index=forecast_index,
                target=target,
                months=months,
                objective=objective,
                parameters=parameters,
                seed=int(seed),
            )
            predictions.append(prediction)
            importances.append(importance)
    elif family == "neural_network":
        x_train = np.asarray(matrix[np.ix_(train_index, feature_index)], dtype=np.float32)
        x_forecast = np.asarray(matrix[np.ix_(forecast_index, feature_index)], dtype=np.float32)
        for seed in seeds:
            model = fit_mlp(
                x_train,
                target[train_index],
                months[train_index],
                objective=objective,
                seed=int(seed),
                hidden_layers=profile["hidden_layers"],
                dropout=float(profile["dropout"]),
                epochs=int(profile["epochs"]),
                learning_rate=float(profile["learning_rate"]),
                weight_decay=float(profile["weight_decay"]),
                batch_size=int(shared["batch_size"]),
                pairs_per_row=int(shared["pairs_per_row"]),
                gradient_clip=float(shared["gradient_clip"]),
            )
            predictions.append(predict_mlp(model, x_forecast))
    elif family == "elastic_net":
        x_train = np.asarray(matrix[np.ix_(train_index, feature_index)], dtype=np.float32)
        x_forecast = np.asarray(matrix[np.ix_(forecast_index, feature_index)], dtype=np.float32)
        for seed in seeds:
            model = fit_elastic_net(
                x_train,
                target[train_index],
                seed=int(seed),
                alpha=float(profile["alpha"]),
                l1_ratio=float(profile["l1_ratio"]),
            )
            predictions.append(model.predict(x_forecast).astype(float))
            importances.append(np.abs(model.coef_.astype(float)))
    else:
        raise ValueError(f"unknown V6.1 model family: {family}")
    return (
        np.mean(np.vstack(predictions), axis=0),
        np.mean(np.vstack(importances), axis=0) if importances else None,
        {int(seed): prediction for seed, prediction in zip(seeds, predictions)},
    )


def choose_profile(
    *,
    family: str,
    objective: str,
    profiles: Sequence[Mapping[str, Any]],
    shared: Mapping[str, Any],
    matrix: np.ndarray,
    feature_index: np.ndarray,
    rows: pd.DataFrame,
    target: np.ndarray,
    train_index: np.ndarray,
    months: np.ndarray,
    holdout_months: int,
    seed: int,
    tie_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subtrain, holdout = _tuning_split(rows, train_index, holdout_months)
    evidence: list[dict[str, Any]] = []
    for ordinal, profile in enumerate(profiles):
        selected = dict(profile)
        if family == "xgboost":
            parameters = {**selected, **dict(shared)}
            parameters.pop("profile_id", None)
            prediction, _, effective_trees = _run_xgboost(
                matrix=matrix,
                feature_index=feature_index,
                train_index=subtrain,
                forecast_index=holdout,
                target=target,
                months=months,
                objective=objective,
                parameters=parameters,
                seed=seed,
                validation_index=holdout,
            )
        else:
            prediction, _, _ = _fit_predict(
                family=family,
                objective=objective,
                profile=selected,
                shared=shared,
                matrix=matrix,
                feature_index=feature_index,
                train_index=subtrain,
                forecast_index=holdout,
                target=target,
                months=months,
                seeds=[seed],
            )
            effective_trees = None
        evidence.append(
            {
                "profile_id": _profile_id(selected),
                "profile_ordinal": ordinal,
                "mean_rank_ic": mean_monthly_rank_ic(
                    months[holdout], prediction, target[holdout]
                ),
                "effective_trees": effective_trees,
            }
        )
    best = max(float(item["mean_rank_ic"]) for item in evidence)
    admissible = [item for item in evidence if float(item["mean_rank_ic"]) >= best - tie_tolerance]
    winner = min(admissible, key=lambda item: (int(item["profile_ordinal"]), str(item["profile_id"])))
    selected = dict(profiles[int(winner["profile_ordinal"])])
    if family == "xgboost" and winner["effective_trees"] is not None:
        selected["n_estimators"] = int(winner["effective_trees"])
        selected["profile_id"] = str(winner["profile_id"]) + f"__trees{winner['effective_trees']}"
    return selected, evidence


def walk_forward_predictions(
    *,
    matrix: np.ndarray,
    rows: pd.DataFrame,
    feature_index: np.ndarray,
    target_column: str,
    family: str,
    objective: str,
    candidate: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    shared: Mapping[str, Any],
    start_month: str,
    end_month: str,
    seeds: Sequence[int],
    tuning: Mapping[str, Any],
    output: Path,
    run_identity: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = pd.to_numeric(rows[target_column], errors="coerce").to_numpy(float)
    months = rows["holding_month"].astype(str).to_numpy()
    forecast_months = sorted(
        rows.loc[rows["holding_month"].between(start_month, end_month), "holding_month"]
        .astype(str).unique()
    )
    scheduled = [
        month for position, month in enumerate(forecast_months)
        if is_refit_month(month, str(candidate["refit"]), first=position == 0)
    ]
    starts = {month: forecast_months.index(month) for month in scheduled}
    selected_profile: Mapping[str, Any] | None = None
    prediction_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    for position, refit_month in enumerate(scheduled):
        stop = starts[scheduled[position + 1]] if position + 1 < len(scheduled) else len(forecast_months)
        block_months = forecast_months[starts[refit_month] : stop]
        checkpoint_root = output / "checkpoints"
        prediction_path = checkpoint_root / f"{refit_month}.parquet"
        importance_path = checkpoint_root / f"{refit_month}.importance.parquet"
        receipt_path = checkpoint_root / f"{refit_month}.json"
        block_identity = canonical_sha256(
            {
                "run_identity": run_identity,
                "family": family,
                "objective": objective,
                "candidate": candidate,
                "refit_month": refit_month,
                "block_months": block_months,
                "seeds": list(seeds),
                "feature_index": feature_index.tolist(),
            }
        )
        if prediction_path.is_file() and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("block_identity") != block_identity:
                raise ValueError(f"V6.1 checkpoint identity mismatch: {receipt_path}")
            selected_profile = dict(receipt["selected_profile"])
            prediction_parts.append(pd.read_parquet(prediction_path))
            if importance_path.is_file():
                importance_parts.append(pd.read_parquet(importance_path))
            tuning_rows.extend(receipt.get("tuning_evidence", []))
            continue
        signal_dates = pd.to_datetime(
            rows.loc[rows["holding_month"].eq(refit_month), "signal_date"], errors="raise"
        ).unique()
        if len(signal_dates) != 1:
            raise ValueError(f"invalid V6.1 signal date for {refit_month}")
        train_index = _training_indices(
            rows,
            target,
            forecast_month=refit_month,
            signal_date=pd.Timestamp(signal_dates[0]),
            history=str(candidate["history"]),
            history_months=(int(candidate["history_months"]) if candidate.get("history_months") else None),
        )
        retune = is_retune_month(
            refit_month, str(candidate["retune"]), first=selected_profile is None
        )
        evidence: list[dict[str, Any]] = []
        if retune:
            selected_profile, evidence = choose_profile(
                family=family,
                objective=objective,
                profiles=profiles,
                shared=shared,
                matrix=matrix,
                feature_index=feature_index,
                rows=rows,
                target=target,
                train_index=train_index,
                months=months,
                holdout_months=int(tuning["holdout_months"]),
                seed=int(tuning["tuning_seed"]),
                tie_tolerance=float(tuning["profile_tie_tolerance"]),
            )
            evidence = [
                {**item, "refit_month": refit_month, "retuned": True}
                for item in evidence
            ]
            tuning_rows.extend(evidence)
        if selected_profile is None:
            raise RuntimeError("V6.1 walk-forward has no selected profile")
        forecast_index = np.flatnonzero(np.isin(months, block_months)).astype(np.int64)
        raw, importance, seed_predictions = _fit_predict(
            family=family,
            objective=objective,
            profile=selected_profile,
            shared=shared,
            matrix=matrix,
            feature_index=feature_index,
            train_index=train_index,
            forecast_index=forecast_index,
            target=target,
            months=months,
            seeds=seeds,
        )
        prediction = rows.iloc[forecast_index][
            ["holding_month", "signal_date", "permno", "permco"]
        ].copy()
        prediction["raw_score"] = raw
        prediction["score_rank"] = rank_within_month(raw, prediction["holding_month"])
        for seed, seed_raw in seed_predictions.items():
            prediction[f"score_rank__seed_{seed}"] = rank_within_month(
                seed_raw, prediction["holding_month"]
            )
        prediction["refit_month"] = refit_month
        prediction["selected_profile_id"] = _profile_id(selected_profile)
        atomic_parquet(prediction, prediction_path)
        if importance is not None:
            importance_frame = pd.DataFrame(
                {"refit_month": refit_month, "feature_index": feature_index, "importance": importance}
            )
            atomic_parquet(importance_frame, importance_path)
            importance_parts.append(importance_frame)
        atomic_json(
            {
                "schema": "research_v6_1_refit_checkpoint_v1",
                "block_identity": block_identity,
                "selected_profile": dict(selected_profile),
                "train_rows": len(train_index),
                "train_first_month": str(months[train_index].min()),
                "train_last_month": str(months[train_index].max()),
                "forecast_months": block_months,
                "tuning_evidence": evidence,
            },
            receipt_path,
        )
        prediction_parts.append(prediction)
    prediction = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["holding_month", "permno"], kind="mergesort"
    )
    tuning_frame = pd.DataFrame(tuning_rows)
    importance_frame = (
        pd.concat(importance_parts, ignore_index=True)
        if importance_parts
        else pd.DataFrame(columns=["refit_month", "feature_index", "importance"])
    )
    atomic_parquet(prediction, output / "predictions.parquet")
    atomic_parquet(tuning_frame, output / "tuning_evidence.parquet")
    atomic_parquet(importance_frame, output / "feature_importance.parquet")
    return prediction, tuning_frame, importance_frame


__all__ = ["choose_profile", "walk_forward_predictions"]
