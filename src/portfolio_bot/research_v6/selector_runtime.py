"""Causal, checkpointed walk-forward runtime for the V6 selector study."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from portfolio_bot.research_v6.evaluation import monthly_selector_metrics
from portfolio_bot.research_v6.models import (
    fit_elastic_net,
    fit_pairwise_mlp,
    fit_pointwise_mlp,
    predict_mlp,
    rank_within_month,
)


ROOT = Path(__file__).resolve().parents[3]
XGBOOST_WORKER = ROOT / "scripts/run_v6_xgboost_worker.py"
STABLE_PYTHON = Path("/opt/anaconda3/envs/finance/bin/python")


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (pd.Timestamp, Path)):
            return str(value)
        raise TypeError(f"cannot serialize {type(value).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=convert) + "\n"
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def mean_monthly_rank_ic(
    months: Sequence[str], score: np.ndarray, target: np.ndarray
) -> float:
    frame = pd.DataFrame(
        {"holding_month": np.asarray(months), "score": score, "target": target}
    )
    values: list[float] = []
    for _, group in frame.groupby("holding_month", observed=True, sort=True):
        valid = np.isfinite(group["score"]) & np.isfinite(group["target"])
        if valid.sum() < 10:
            continue
        left = group.loc[valid, "score"].rank(method="average").to_numpy(float)
        right = group.loc[valid, "target"].rank(method="average").to_numpy(float)
        if np.std(left) == 0.0 or np.std(right) == 0.0:
            values.append(0.0)
        else:
            values.append(float(np.corrcoef(left, right)[0, 1]))
    return float(np.mean(values)) if values else -math.inf


def is_refit_month(month: str, cadence: str, *, first: bool = False) -> bool:
    if first:
        return True
    number = int(str(month).split("-")[1])
    if cadence == "annual":
        return number == 1
    if cadence == "quarterly":
        return number in {1, 4, 7, 10}
    raise ValueError(f"unknown refit cadence: {cadence}")


def is_retune_month(month: str, cadence: str, *, first: bool = False) -> bool:
    if first:
        return True
    year, number = (int(value) for value in str(month).split("-"))
    if number != 1:
        return False
    if cadence == "annual":
        return True
    if cadence == "biennial":
        return year % 2 == 0
    raise ValueError(f"unknown retune cadence: {cadence}")


def _training_indices(
    rows: pd.DataFrame,
    target: np.ndarray,
    *,
    forecast_month: str,
    signal_date: pd.Timestamp,
    history: str,
    history_months: int | None,
) -> np.ndarray:
    months = rows["holding_month"].astype(str)
    availability = pd.to_datetime(rows["label_available_date"], errors="coerce")
    eligible = (
        months.lt(forecast_month)
        & availability.le(signal_date)
        & np.isfinite(target)
    )
    if history == "rolling":
        observed = sorted(months.loc[eligible].unique())
        if not history_months or history_months < 24:
            raise ValueError("rolling history requires at least 24 observed months")
        keep = set(observed[-history_months:])
        eligible &= months.isin(keep)
    elif history != "expanding":
        raise ValueError(f"unknown training history: {history}")
    return np.flatnonzero(eligible.to_numpy()).astype(np.int64)


def _tuning_split(
    rows: pd.DataFrame, train_index: np.ndarray, holdout_months: int
) -> tuple[np.ndarray, np.ndarray]:
    train_months = rows.iloc[train_index]["holding_month"].astype(str)
    observed = sorted(train_months.unique())
    if len(observed) <= holdout_months + 12:
        raise ValueError("insufficient history for causal internal tuning")
    holdout_set = set(observed[-holdout_months:])
    holdout_mask = train_months.isin(holdout_set).to_numpy()
    return train_index[~holdout_mask], train_index[holdout_mask]


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
) -> tuple[np.ndarray, np.ndarray | None]:
    predictions: list[np.ndarray] = []
    importances: list[np.ndarray] = []
    y_train = np.asarray(target[train_index], dtype=float)
    train_months = months[train_index]
    if family == "xgboost":
        parameters = {
            "n_estimators": int(profile["n_estimators"]),
            "max_depth": int(profile["max_depth"]),
            "min_child_weight": float(profile["min_child_weight"]),
            **{key: value for key, value in shared.items() if key != "profile_id"},
        }
        for seed in seeds:
            if objective not in {"pairwise", "continuous_rank"}:
                raise ValueError(f"unknown XGBoost objective: {objective}")
            matrix_filename = getattr(matrix, "filename", None)
            if matrix_filename is None:
                raise ValueError("production XGBoost requires a memory-mapped matrix")
            with tempfile.TemporaryDirectory(prefix="v6_xgb_") as temporary_name:
                temporary = Path(temporary_name)
                paths = {
                    "train": temporary / "train.npy",
                    "forecast": temporary / "forecast.npy",
                    "features": temporary / "features.npy",
                    "target": temporary / "target.npy",
                    "groups": temporary / "groups.npy",
                    "prediction": temporary / "prediction.npy",
                    "importance": temporary / "importance.npy",
                }
                np.save(paths["train"], train_index)
                np.save(paths["forecast"], forecast_index)
                np.save(paths["features"], feature_index)
                np.save(paths["target"], y_train)
                groups = (
                    pd.Series(train_months, dtype="string")
                    .groupby(pd.Series(train_months, dtype="string"), sort=False)
                    .size()
                    .to_numpy(np.int64)
                )
                np.save(paths["groups"], groups)
                subprocess.run(
                    [
                        str(STABLE_PYTHON),
                        str(XGBOOST_WORKER),
                        "--matrix",
                        str(matrix_filename),
                        "--train-index",
                        str(paths["train"]),
                        "--forecast-index",
                        str(paths["forecast"]),
                        "--feature-index",
                        str(paths["features"]),
                        "--train-target",
                        str(paths["target"]),
                        "--group-sizes",
                        str(paths["groups"]),
                        "--objective",
                        objective,
                        "--parameters-json",
                        json.dumps(parameters, sort_keys=True),
                        "--seed",
                        str(int(seed)),
                        "--prediction-output",
                        str(paths["prediction"]),
                        "--importance-output",
                        str(paths["importance"]),
                    ],
                    cwd=ROOT,
                    check=True,
                )
                predictions.append(np.load(paths["prediction"]).astype(float))
                importances.append(np.load(paths["importance"]).astype(float))
    elif family == "neural_network":
        x_train = np.asarray(
            matrix[np.ix_(train_index, feature_index)], dtype=np.float32
        )
        x_forecast = np.asarray(
            matrix[np.ix_(forecast_index, feature_index)], dtype=np.float32
        )
        common = dict(
            hidden_layers=tuple(int(value) for value in profile["hidden_layers"]),
            dropout=float(profile["dropout"]),
            epochs=int(profile["epochs"]),
            batch_size=int(shared["batch_size"]),
        )
        for seed in seeds:
            if objective == "pairwise":
                model = fit_pairwise_mlp(
                    x_train,
                    y_train,
                    train_months,
                    seed=int(seed),
                    pairs_per_row=int(shared["pairs_per_row"]),
                    **common,
                )
            elif objective == "continuous_rank":
                model = fit_pointwise_mlp(
                    x_train, y_train, seed=int(seed), **common
                )
            else:
                raise ValueError(f"unknown neural-network objective: {objective}")
            predictions.append(predict_mlp(model, x_forecast))
    elif family == "elastic_net":
        x_train = np.asarray(
            matrix[np.ix_(train_index, feature_index)], dtype=np.float32
        )
        x_forecast = np.asarray(
            matrix[np.ix_(forecast_index, feature_index)], dtype=np.float32
        )
        for seed in seeds:
            model = fit_elastic_net(
                x_train,
                y_train,
                seed=int(seed),
                alpha=float(profile["alpha"]),
                l1_ratio=float(profile["l1_ratio"]),
            )
            predictions.append(np.asarray(model.predict(x_forecast), dtype=float))
            importances.append(np.abs(np.asarray(model.coef_, dtype=float)))
    else:
        raise ValueError(f"unknown model family: {family}")
    prediction = np.mean(np.vstack(predictions), axis=0)
    importance = np.mean(np.vstack(importances), axis=0) if importances else None
    return prediction, importance


def _profile_id(profile: Mapping[str, Any]) -> str:
    return str(profile.get("profile_id") or canonical_sha256(profile)[:12])


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
    tuning_seed: int,
    tie_tolerance: float,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    subtrain, holdout = _tuning_split(rows, train_index, holdout_months)
    evidence: list[dict[str, Any]] = []
    for ordinal, profile in enumerate(profiles):
        prediction, _ = _fit_predict(
            family=family,
            objective=objective,
            profile=profile,
            shared=shared,
            matrix=matrix,
            feature_index=feature_index,
            train_index=subtrain,
            forecast_index=holdout,
            target=target,
            months=months,
            seeds=[tuning_seed],
        )
        evidence.append(
            {
                "profile_id": _profile_id(profile),
                "profile_ordinal": ordinal,
                "mean_rank_ic": mean_monthly_rank_ic(
                    months[holdout], prediction, target[holdout]
                ),
            }
        )
    best = max(item["mean_rank_ic"] for item in evidence)
    admissible = [
        item for item in evidence if item["mean_rank_ic"] >= best - tie_tolerance
    ]
    selected = min(admissible, key=lambda item: (item["profile_ordinal"], item["profile_id"]))
    return profiles[int(selected["profile_ordinal"])], evidence


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
    """Create causal forecasts and resume at completed refit blocks."""

    target = pd.to_numeric(rows[target_column], errors="coerce").to_numpy(float)
    months = rows["holding_month"].astype(str).to_numpy()
    forecast_months = sorted(
        rows.loc[rows["holding_month"].between(start_month, end_month), "holding_month"]
        .astype(str)
        .unique()
    )
    scheduled = [
        month
        for position, month in enumerate(forecast_months)
        if is_refit_month(month, str(candidate["refit"]), first=position == 0)
    ]
    block_starts = {month: forecast_months.index(month) for month in scheduled}
    selected_profile: Mapping[str, Any] | None = None
    prediction_parts: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    importance_parts: list[pd.DataFrame] = []
    for refit_position, refit_month in enumerate(scheduled):
        next_position = (
            block_starts[scheduled[refit_position + 1]]
            if refit_position + 1 < len(scheduled)
            else len(forecast_months)
        )
        block_months = forecast_months[block_starts[refit_month] : next_position]
        checkpoint = output / "checkpoints" / f"{refit_month}.parquet"
        importance_checkpoint = output / "checkpoints" / f"{refit_month}.importance.parquet"
        receipt_path = output / "checkpoints" / f"{refit_month}.json"
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
        if checkpoint.is_file() and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("block_identity") != block_identity:
                raise ValueError(f"checkpoint identity mismatch: {receipt_path}")
            selected_id = str(receipt["selected_profile_id"])
            selected_profile = next(
                profile for profile in profiles if _profile_id(profile) == selected_id
            )
            prediction_parts.append(pd.read_parquet(checkpoint))
            if importance_checkpoint.is_file():
                importance_parts.append(pd.read_parquet(importance_checkpoint))
            tuning_rows.extend(receipt.get("tuning_evidence", []))
            continue
        signal_dates = pd.to_datetime(
            rows.loc[rows["holding_month"].eq(refit_month), "signal_date"],
            errors="raise",
        ).unique()
        if len(signal_dates) != 1:
            raise ValueError(f"forecast month {refit_month} has invalid signal dates")
        train_index = _training_indices(
            rows,
            target,
            forecast_month=refit_month,
            signal_date=pd.Timestamp(signal_dates[0]),
            history=str(candidate["history"]),
            history_months=(
                int(candidate["history_months"])
                if candidate.get("history_months") is not None
                else None
            ),
        )
        retune = is_retune_month(
            refit_month,
            str(candidate["retune"]),
            first=selected_profile is None,
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
                tuning_seed=int(tuning["tuning_seed"]),
                tie_tolerance=float(tuning["profile_tie_tolerance"]),
            )
            for item in evidence:
                tuning_rows.append(
                    {**item, "refit_month": refit_month, "retuned": True}
                )
        if selected_profile is None:
            raise RuntimeError("walk-forward fit has no selected profile")
        forecast_index = np.flatnonzero(np.isin(months, block_months)).astype(np.int64)
        raw, importance = _fit_predict(
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
        prediction["refit_month"] = refit_month
        prediction["selected_profile_id"] = _profile_id(selected_profile)
        atomic_parquet(prediction, checkpoint)
        atomic_json(
            {
                "schema": "research_v6_refit_checkpoint_v1",
                "block_identity": block_identity,
                "selected_profile_id": _profile_id(selected_profile),
                "train_rows": int(len(train_index)),
                "train_first_month": str(months[train_index].min()),
                "train_last_month": str(months[train_index].max()),
                "forecast_months": block_months,
                "tuning_evidence": evidence,
            },
            receipt_path,
        )
        prediction_parts.append(prediction)
        if importance is not None:
            importance_frame = pd.DataFrame(
                {
                    "refit_month": refit_month,
                    "feature_index": feature_index,
                    "importance": importance,
                }
            )
            atomic_parquet(importance_frame, importance_checkpoint)
            importance_parts.append(importance_frame)
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["holding_month", "permno"], kind="mergesort"
    )
    tuning_frame = pd.DataFrame(tuning_rows)
    importance_frame = (
        pd.concat(importance_parts, ignore_index=True)
        if importance_parts
        else pd.DataFrame(columns=["refit_month", "feature_index", "importance"])
    )
    atomic_parquet(predictions, output / "predictions.parquet")
    atomic_parquet(tuning_frame, output / "tuning_evidence.parquet")
    atomic_parquet(importance_frame, output / "feature_importance.parquet")
    return predictions, tuning_frame, importance_frame


def evaluate_predictions(
    predictions: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    target_column: str,
    economic_column: str,
    model_id: str,
) -> pd.DataFrame:
    joined = predictions.merge(
        rows[
            [
                "holding_month",
                "permno",
                target_column,
                economic_column,
            ]
        ],
        on=["holding_month", "permno"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"score_rank": model_id})
    return monthly_selector_metrics(
        joined,
        score_columns=[model_id],
        target_column=target_column,
        economic_column=economic_column,
    )


__all__ = [
    "atomic_json",
    "atomic_parquet",
    "canonical_sha256",
    "choose_profile",
    "evaluate_predictions",
    "is_refit_month",
    "is_retune_month",
    "mean_monthly_rank_ic",
    "walk_forward_predictions",
]
