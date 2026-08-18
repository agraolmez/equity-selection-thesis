#!/usr/bin/env python3
"""Run the bounded V6.7 neural daily-overlay comparison.

V6.7 is a separate post-diagnostic mechanism study.  It reads the immutable
V6.6 prepared daily panel and frozen XGBoost comparator, adds compact MLP
regression/classification candidates, selects one challenger on 2023, and
evaluates that single frozen challenger on January 2024--August 2025.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_v6_6_intramonth_interventions as base  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/research_v6_7/daily_mlp_overlay.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_7/daily_mlp_overlay"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _resolve(relative: str) -> Path:
    return ROOT / relative


def _verify(config_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config = json.loads(config_path.read_text())
    if config.get("status") != "AUTHORIZED_POST_DIAGNOSTIC_STUDY":
        raise RuntimeError("V6.7 daily MLP overlay is not authorized")
    sources = config["sources"]
    checks: list[tuple[Path, str, str]] = []
    for name in (
        "v6_6_config",
        "v6_6_runner",
        "v6_6_result",
        "v6_6_selection",
        "prepared_position_days",
        "frozen_xgboost_validation_predictions",
        "frozen_xgboost_historical_predictions",
    ):
        item = sources[name]
        checks.append((_resolve(item["path"]), str(item["sha256"]), name))
    for name, item in sources["frozen_historical_replays"].items():
        checks.append((_resolve(item["path"]), str(item["sha256"]), name))
    encoded = sources["v6_encoded_inputs"]
    checks.extend(
        [
            (_resolve(encoded["rows"]), str(encoded["rows_sha256"]), "encoded rows"),
            (_resolve(encoded["matrix"]), str(encoded["matrix_sha256"]), "encoded matrix"),
        ]
    )
    for path, expected, label in checks:
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(f"{label} identity changed: {path}")
    code_identity = hashlib.sha256()
    for path in (Path(__file__).resolve(), _resolve(sources["v6_6_runner"]["path"])):
        code_identity.update(path.name.encode())
        code_identity.update(bytes.fromhex(_sha256(path)))
    identity = {
        "config_sha256": _sha256(config_path),
        "code_identity": code_identity.hexdigest(),
        "v6_input_identity": str(encoded["input_identity"]),
        "v6_6_result_sha256": str(sources["v6_6_result"]["sha256"]),
        "prepared_days_sha256": str(sources["prepared_position_days"]["sha256"]),
    }
    return config, identity


def _output_path(output_root: Path, identity: Mapping[str, str]) -> Path:
    return output_root / (
        f"{identity['config_sha256'][:12]}-"
        f"{identity['v6_input_identity'][:12]}-"
        f"{identity['code_identity'][:12]}"
    )


class DailyMLP(torch.nn.Module):
    def __init__(self, input_features: int, hidden_layers: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        previous = int(input_features)
        for width in hidden_layers:
            layers.extend(
                [
                    torch.nn.Linear(previous, int(width)),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(float(dropout)),
                ]
            )
            previous = int(width)
        layers.append(torch.nn.Linear(previous, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def _standardize(
    x_train: np.ndarray, x_forecast: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    train = np.asarray(x_train, dtype=np.float32).copy()
    forecast = np.asarray(x_forecast, dtype=np.float32).copy()
    train[~np.isfinite(train)] = 0.0
    forecast[~np.isfinite(forecast)] = 0.0
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    constant = scale < 1e-6
    scale[constant] = 1.0
    train -= mean
    train /= scale
    forecast -= mean
    forecast /= scale
    np.clip(train, -12.0, 12.0, out=train)
    np.clip(forecast, -12.0, 12.0, out=forecast)
    return train, forecast, {
        "constant_feature_count": int(constant.sum()),
        "maximum_absolute_standardized_train_value": float(np.max(np.abs(train), initial=0.0)),
    }


def _fit_seed_predict(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_forecast: np.ndarray,
    profile: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    torch.manual_seed(int(seed))
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    generator = np.random.default_rng(int(seed))
    model = DailyMLP(
        x_train.shape[1],
        tuple(int(value) for value in profile["hidden_layers"]),
        float(profile["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    train_tensor = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    objective = str(profile["objective"])
    target_mean = 0.0
    target_scale = 1.0
    if objective == "signed_return_regression":
        lower, upper = np.quantile(
            y_train,
            [float(profile["target_winsor_lower"]), float(profile["target_winsor_upper"])],
        )
        target = np.clip(np.asarray(y_train, dtype=np.float32), lower, upper)
        target_mean = float(target.mean())
        target_scale = max(float(target.std()), 1e-6)
        target_tensor = torch.from_numpy(((target - target_mean) / target_scale).astype(np.float32))
    elif objective == "adverse_return_classifier":
        target_tensor = torch.from_numpy((np.asarray(y_train, dtype=float) < 0.0).astype(np.float32))
    else:
        raise ValueError(f"unknown MLP objective: {objective}")
    batch_size = int(profile["batch_size"])
    final_loss = math.nan
    model.train()
    for _ in range(int(profile["epochs"])):
        order = generator.permutation(len(train_tensor))
        loss_total = 0.0
        observed = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            output = model(train_tensor[selected])
            if objective == "signed_return_regression":
                loss = torch.nn.functional.smooth_l1_loss(
                    output, target_tensor[selected], beta=0.25
                )
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output, target_tensor[selected]
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach()) * len(selected)
            observed += len(selected)
        final_loss = loss_total / max(observed, 1)
    model.eval()
    forecast_tensor = torch.from_numpy(np.asarray(x_forecast, dtype=np.float32))
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(forecast_tensor), 16_384):
            raw = model(forecast_tensor[start : start + 16_384])
            if objective == "signed_return_regression":
                score = raw * target_scale + target_mean
            else:
                score = 0.5 - torch.sigmoid(raw)
            pieces.append(score.numpy())
    prediction = np.concatenate(pieces).astype(np.float64)
    if not np.isfinite(prediction).all():
        raise RuntimeError("MLP produced non-finite predictions")
    return prediction, {
        "seed": int(seed),
        "final_training_loss": float(final_loss),
        "target_mean": float(target_mean),
        "target_scale": float(target_scale),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
    }


def _prediction_checkpoint(
    *,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    output: Path,
    profile: Mapping[str, Any],
    horizon: int,
    phase: str,
    refit_month: str,
    forecast_months: list[str],
    train_cutoff: pd.Timestamp,
    identity: Mapping[str, str],
) -> pd.DataFrame:
    block_identity = _canonical_sha(
        {
            "identity": identity,
            "profile": profile,
            "horizon": int(horizon),
            "phase": phase,
            "refit_month": refit_month,
            "forecast_months": forecast_months,
            "train_cutoff": str(train_cutoff),
            "seeds": config["mlp_models"]["seeds"],
        }
    )
    root = output / "models" / phase / str(profile["id"]) / f"h{horizon}"
    prediction_path = root / f"{refit_month}.parquet"
    receipt_path = root / f"{refit_month}.json"
    if prediction_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("block_identity") != block_identity:
            raise RuntimeError(f"V6.7 model checkpoint identity mismatch: {receipt_path}")
        return pd.read_parquet(prediction_path)
    target_column = f"target_signed_return_{int(horizon)}d"
    availability_column = f"label_available_date_{int(horizon)}d"
    train_mask = (
        frame["holding_month"].lt(refit_month)
        & pd.to_datetime(frame[availability_column], errors="coerce").le(train_cutoff)
        & pd.to_numeric(frame[target_column], errors="coerce").notna()
    )
    forecast_mask = frame["holding_month"].isin(forecast_months)
    train_index = np.flatnonzero(train_mask.to_numpy())
    forecast_index = np.flatnonzero(forecast_mask.to_numpy())
    if len(train_index) < 5_000 or len(forecast_index) == 0:
        raise RuntimeError(
            f"insufficient V6.7 data for {phase}/{refit_month}/{profile['id']}/h{horizon}"
        )
    x_train = base._feature_matrix(frame, config, train_index)
    x_forecast = base._feature_matrix(frame, config, forecast_index)
    x_train, x_forecast, scaling = _standardize(x_train, x_forecast)
    y_train = frame.iloc[train_index][target_column].to_numpy(float)
    seed_predictions: list[np.ndarray] = []
    seed_receipts: list[dict[str, float | int]] = []
    for seed in config["mlp_models"]["seeds"]:
        prediction, seed_receipt = _fit_seed_predict(
            x_train=x_train,
            y_train=y_train,
            x_forecast=x_forecast,
            profile=profile,
            seed=int(seed),
        )
        seed_predictions.append(prediction)
        seed_receipts.append(seed_receipt)
    stacked = np.vstack(seed_predictions)
    prediction = frame.iloc[forecast_index][
        ["holding_month", "calendar_date", "permno"]
    ].copy()
    prediction["score"] = stacked.mean(axis=0)
    prediction["seed_score_std"] = stacked.std(axis=0, ddof=1)
    prediction["model_id"] = str(profile["id"])
    prediction["horizon"] = int(horizon)
    prediction["refit_month"] = refit_month
    _atomic_parquet(prediction, prediction_path)
    _atomic_json(
        receipt_path,
        {
            "schema": "research_v6_7_mlp_refit_v1",
            "block_identity": block_identity,
            "profile": profile,
            "horizon": int(horizon),
            "phase": phase,
            "refit_month": refit_month,
            "forecast_months": forecast_months,
            "train_cutoff": str(train_cutoff),
            "train_rows": int(len(train_index)),
            "forecast_rows": int(len(forecast_index)),
            "train_first_month": str(frame.iloc[train_index]["holding_month"].min()),
            "train_last_month": str(frame.iloc[train_index]["holding_month"].max()),
            "feature_scaling": scaling,
            "seed_receipts": seed_receipts,
            "mean_seed_prediction_std": float(prediction["seed_score_std"].mean()),
        },
    )
    return prediction


def _zscore(values: Iterable[float], mean: float | None = None, std: float | None = None) -> tuple[np.ndarray, float, float]:
    array = np.asarray(list(values), dtype=float)
    center = float(array.mean()) if mean is None else float(mean)
    scale = max(float(array.std(ddof=1)), 1e-8) if std is None else max(float(std), 1e-8)
    return (array - center) / scale, center, scale


def _merge_ensemble(
    mlp: pd.DataFrame,
    xgb: pd.DataFrame,
    scaling: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    merged = mlp[["holding_month", "calendar_date", "permno", "score"]].merge(
        xgb[["holding_month", "calendar_date", "permno", "score"]],
        on=["holding_month", "calendar_date", "permno"],
        suffixes=("_mlp", "_xgb"),
        validate="one_to_one",
    )
    mlp_z, mlp_mean, mlp_std = _zscore(
        merged["score_mlp"],
        None if scaling is None else float(scaling["mlp_mean"]),
        None if scaling is None else float(scaling["mlp_std"]),
    )
    xgb_z, xgb_mean, xgb_std = _zscore(
        merged["score_xgb"],
        None if scaling is None else float(scaling["xgb_mean"]),
        None if scaling is None else float(scaling["xgb_std"]),
    )
    merged["score"] = 0.5 * (mlp_z + xgb_z)
    return merged[["holding_month", "calendar_date", "permno", "score"]], {
        "mlp_mean": mlp_mean,
        "mlp_std": mlp_std,
        "xgb_mean": xgb_mean,
        "xgb_std": xgb_std,
    }


def _validation_predictions(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    output: Path,
    identity: Mapping[str, str],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    chronology = config["chronology"]
    forecast_months = base._month_sequence(
        chronology["selection_start"], chronology["selection_end"]
    )
    cutoff = pd.Timestamp(
        frame.loc[frame["holding_month"].eq(forecast_months[0]), "signal_date"].iloc[0]
    )
    result: dict[str, pd.DataFrame] = {}
    for profile in config["mlp_models"]["profiles"]:
        for horizon in config["mlp_models"]["targets"]:
            key = f"{profile['id']}__h{int(horizon)}"
            result[key] = _prediction_checkpoint(
                frame=frame,
                config=config,
                output=output,
                profile=profile,
                horizon=int(horizon),
                phase="selection",
                refit_month=forecast_months[0],
                forecast_months=forecast_months,
                train_cutoff=cutoff,
                identity=identity,
            )
    xgb = pd.read_parquet(
        _resolve(config["sources"]["frozen_xgboost_validation_predictions"]["path"])
    )
    scaling: dict[str, dict[str, float]] = {}
    horizon = int(config["mlp_models"]["ensemble"]["eligible_horizon"])
    for profile in config["mlp_models"]["profiles"]:
        component_key = f"{profile['id']}__h{horizon}"
        ensemble_key = f"ensemble_{profile['id']}_xgb_depth_2__h{horizon}"
        result[ensemble_key], scaling[ensemble_key] = _merge_ensemble(
            result[component_key], xgb
        )
    return result, scaling


def _historical_prediction(
    *,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    output: Path,
    identity: Mapping[str, str],
    selected: Mapping[str, Any],
) -> pd.DataFrame:
    model_key = str(selected["model_key"])
    horizon = int(selected["horizon"])
    component_key = model_key
    if model_key.startswith("ensemble_"):
        component_key = model_key.removeprefix("ensemble_").split("_xgb_depth_2__h", 1)[0]
    else:
        component_key = model_key.rsplit("__h", 1)[0]
    profiles = {str(value["id"]): value for value in config["mlp_models"]["profiles"]}
    profile = profiles[component_key]
    test_months = base._month_sequence(
        config["chronology"]["historical_test_start"],
        config["chronology"]["historical_test_end"],
    )
    refits = [month for month in test_months if month.endswith(("-01", "-04", "-07", "-10"))]
    parts: list[pd.DataFrame] = []
    for position, refit in enumerate(refits):
        start = test_months.index(refit)
        stop = test_months.index(refits[position + 1]) if position + 1 < len(refits) else len(test_months)
        block = test_months[start:stop]
        cutoff = pd.Timestamp(
            frame.loc[frame["holding_month"].eq(refit), "signal_date"].iloc[0]
        )
        parts.append(
            _prediction_checkpoint(
                frame=frame,
                config=config,
                output=output,
                profile=profile,
                horizon=horizon,
                phase="historical_test",
                refit_month=refit,
                forecast_months=block,
                train_cutoff=cutoff,
                identity=identity,
            )
        )
    mlp = pd.concat(parts, ignore_index=True)
    if model_key.startswith("ensemble_"):
        xgb = pd.read_parquet(
            _resolve(config["sources"]["frozen_xgboost_historical_predictions"]["path"])
        )
        result, _ = _merge_ensemble(mlp, xgb, selected["ensemble_scaling"])
    else:
        result = mlp[["holding_month", "calendar_date", "permno", "score"]].copy()
    _atomic_parquet(result, output / "models" / "selected_historical_test_predictions.parquet")
    return result


def _replay_checkpoint(
    *,
    root: Path,
    days: pd.DataFrame,
    replay_config: Mapping[str, Any],
    policy: Mapping[str, Any],
    evaluation_start: str,
    evaluation_end: str,
    scores: pd.DataFrame,
    identity: Mapping[str, str],
    score_identity: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target = root / str(policy["id"])
    receipt_path = target / "receipt.json"
    replay_identity = _canonical_sha(
        {
            "identity": identity,
            "policy": policy,
            "evaluation_start": evaluation_start,
            "evaluation_end": evaluation_end,
            "score_identity": score_identity,
        }
    )
    required = [
        target / "monthly.parquet",
        target / "daily.parquet",
        target / "actions.parquet",
        target / "summary.json",
    ]
    if receipt_path.is_file() and all(path.is_file() for path in required):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("replay_identity") != replay_identity:
            raise RuntimeError(f"V6.7 replay checkpoint identity mismatch: {receipt_path}")
        return (
            pd.read_parquet(required[0]),
            pd.read_parquet(required[1]),
            pd.read_parquet(required[2]),
            json.loads(required[3].read_text()),
        )
    monthly, daily, actions, summary = base._replay_policy(
        days=days,
        config=replay_config,
        policy=policy,
        scores=scores,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )
    base._save_replay(root, str(policy["id"]), monthly, daily, actions, summary)
    _atomic_json(
        receipt_path,
        {
            "schema": "research_v6_7_replay_checkpoint_v1",
            "replay_identity": replay_identity,
            "policy_id": policy["id"],
            "evaluation_start": evaluation_start,
            "evaluation_end": evaluation_end,
        },
    )
    return monthly, daily, actions, summary


def _paired(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    joined = left[["holding_month", "net_excess_return"]].merge(
        right[["holding_month", "net_excess_return"]],
        on="holding_month",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    difference = joined["net_excess_return_right"] - joined["net_excess_return_left"]
    return {
        "annualized_difference": float(difference.mean() * 12.0),
        "paired_hac_t": float(base._hac_t(difference, lags=6)),
        "positive_difference_fraction": float((difference > 0.0).mean()),
    }


def _result_markdown(result: Mapping[str, Any]) -> str:
    validation = result["selection"]
    historical = result["historical_test"]
    selected = validation["selected_v6_7_challenger"]
    return f"""# Research V6.7 daily MLP overlay result

## Decision

`{result['decision']['terminal_decision']}`

{result['decision']['interpretation']}

## 2023 selection

- Frozen monthly control: {validation['monthly_control']['annualized_net_excess_return']:.2%} annualized net excess return, {validation['monthly_control']['net_sharpe']:.3f} Sharpe.
- Frozen V6.6 XGBoost overlay: {validation['frozen_xgboost']['annualized_net_excess_return']:.2%}, {validation['frozen_xgboost']['net_sharpe']:.3f} Sharpe.
- Selected V6.7 challenger: `{selected['policy_id']}` using `{selected['model_key']}`, {selected['summary']['annualized_net_excess_return']:.2%}, {selected['summary']['net_sharpe']:.3f} Sharpe.

## January 2024--August 2025 historical evidence

- Monthly control: {historical['monthly_control']['annualized_net_excess_return']:.2%} annualized net excess return, {historical['monthly_control']['annualized_realized_volatility']:.2%} volatility, {historical['monthly_control']['net_sharpe']:.3f} Sharpe, {historical['monthly_control']['maximum_drawdown']:.2%} maximum drawdown.
- Frozen XGBoost overlay: {historical['frozen_xgboost']['annualized_net_excess_return']:.2%}, {historical['frozen_xgboost']['annualized_realized_volatility']:.2%}, {historical['frozen_xgboost']['net_sharpe']:.3f}, {historical['frozen_xgboost']['maximum_drawdown']:.2%}.
- Selected MLP/ensemble challenger: {historical['selected_v6_7_challenger']['annualized_net_excess_return']:.2%}, {historical['selected_v6_7_challenger']['annualized_realized_volatility']:.2%}, {historical['selected_v6_7_challenger']['net_sharpe']:.3f}, {historical['selected_v6_7_challenger']['maximum_drawdown']:.2%}.
- Paired annualized difference versus monthly control: {result['paired_differences']['versus_monthly_control']['annualized_difference']:.2%}, HAC t {result['paired_differences']['versus_monthly_control']['paired_hac_t']:.2f}.
- Paired annualized difference versus frozen XGBoost: {result['paired_differences']['versus_frozen_xgboost']['annualized_difference']:.2%}, HAC t {result['paired_differences']['versus_frozen_xgboost']['paired_hac_t']:.2f}.

## Interpretation boundary

This study was explicitly opened after the V6.6 historical outcome was known.
It can identify a useful neural mechanism or a paper-trading shadow candidate,
but it cannot create untouched confirmation from January 2024--August 2025.
Prospective evidence can begin no earlier than September 2026.
"""


def run(config_path: Path, output_root: Path) -> Path:
    config, identity = _verify(config_path)
    output = _output_path(output_root, identity)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema": "research_v6_7_daily_mlp_run_v1",
            "status": "RUNNING",
            "identity": identity,
        },
    )
    replay_config = json.loads(
        _resolve(config["sources"]["v6_6_config"]["path"]).read_text()
    )
    days = pd.read_parquet(_resolve(config["sources"]["prepared_position_days"]["path"]))
    required = {
        "holding_month",
        "signal_date",
        "calendar_date",
        "permno",
        "encoded_row",
        "target_signed_return_1d",
        "target_signed_return_5d",
        "label_available_date_1d",
        "label_available_date_5d",
    }
    missing = sorted(required.difference(days.columns))
    if missing:
        raise RuntimeError(f"prepared V6.6 panel omits columns: {missing}")
    if days.duplicated(["holding_month", "calendar_date", "permno"]).any():
        raise RuntimeError("prepared V6.6 panel contains duplicate position-days")

    validation_predictions, ensemble_scaling = _validation_predictions(
        days, config, output, identity
    )
    v6_6_selection = json.loads(
        _resolve(config["sources"]["v6_6_selection"]["path"]).read_text()
    )
    selection_root = output / "selection"
    candidates: list[dict[str, Any]] = []
    for model_key, prediction in validation_predictions.items():
        horizon = int(model_key.rsplit("__h", 1)[1])
        values = prediction["score"].to_numpy(float)
        for intensity in config["mlp_models"]["decision_intensities"]:
            exit_threshold = min(
                float(np.quantile(values, float(intensity["exit_quantile"]))), 0.0
            )
            trim_threshold = min(
                float(np.quantile(values, float(intensity["trim_quantile"]))), 0.0
            )
            policy_id = f"predictive__{model_key}__{intensity['id']}"
            predictive = {
                "model_key": model_key,
                "horizon": horizon,
                "intensity": str(intensity["id"]),
                "exit_threshold": exit_threshold,
                "trim_threshold": trim_threshold,
            }
            policy = {"id": policy_id, "predictive": predictive}
            monthly, daily, actions, summary = _replay_checkpoint(
                root=selection_root,
                days=days,
                replay_config=replay_config,
                policy=policy,
                evaluation_start=config["chronology"]["selection_start"],
                evaluation_end=config["chronology"]["selection_end"],
                scores=prediction,
                identity=identity,
                score_identity=model_key,
            )
            candidate = {
                "kind": "v6_7_predictive",
                "policy_id": policy_id,
                "policy": policy,
                "model_key": model_key,
                "horizon": horizon,
                "intensity": str(intensity["id"]),
                "exit_threshold": exit_threshold,
                "trim_threshold": trim_threshold,
                "summary": summary,
            }
            if model_key in ensemble_scaling:
                candidate["ensemble_scaling"] = ensemble_scaling[model_key]
            candidates.append(candidate)
    selected = max(candidates, key=base._candidate_order)
    candidate_rows = [
        {
            "kind": value["kind"],
            "policy_id": value["policy_id"],
            "model_key": value["model_key"],
            "horizon": value["horizon"],
            "intensity": value["intensity"],
            **value["summary"],
        }
        for value in candidates
    ]
    candidate_rows.append(
        {
            "kind": "frozen_v6_6_comparator",
            "policy_id": v6_6_selection["selected_predictive"]["policy_id"],
            "model_key": v6_6_selection["selected_predictive"]["model_key"],
            "horizon": v6_6_selection["selected_predictive"]["horizon"],
            "intensity": v6_6_selection["selected_predictive"]["intensity"],
            **v6_6_selection["selected_predictive"]["summary"],
        }
    )
    _atomic_parquet(pd.DataFrame(candidate_rows), selection_root / "candidate_summary.parquet")
    _atomic_json(
        selection_root / "selection.json",
        {
            "schema": "research_v6_7_daily_mlp_selection_v1",
            "monthly_control": v6_6_selection["control"],
            "frozen_xgboost": v6_6_selection["selected_predictive"]["summary"],
            "selected_v6_7_challenger": selected,
        },
    )

    historical_scores = _historical_prediction(
        frame=days,
        config=config,
        output=output,
        identity=identity,
        selected=selected,
    )
    selected_policy = {
        "id": "selected_v6_7_challenger",
        "predictive": selected["policy"]["predictive"],
    }
    challenger_monthly, challenger_daily, challenger_actions, challenger_summary = _replay_checkpoint(
        root=output / "historical_test",
        days=days,
        replay_config=replay_config,
        policy=selected_policy,
        evaluation_start=config["chronology"]["historical_test_start"],
        evaluation_end=config["chronology"]["historical_test_end"],
        scores=historical_scores,
        identity=identity,
        score_identity=str(selected["model_key"]),
    )
    predecessor_result = json.loads(
        _resolve(config["sources"]["v6_6_result"]["path"]).read_text()
    )
    replay_sources = config["sources"]["frozen_historical_replays"]
    control_monthly = pd.read_parquet(
        _resolve(replay_sources["monthly_control_monthly"]["path"])
    )
    xgb_monthly = pd.read_parquet(
        _resolve(replay_sources["frozen_xgboost_monthly"]["path"])
    )
    control_summary = predecessor_result["historical_test"]["monthly_control"]
    xgb_summary = predecessor_result["historical_test"]["selected_predictive"]
    versus_control = _paired(control_monthly, challenger_monthly)
    versus_xgb = _paired(xgb_monthly, challenger_monthly)
    tolerance = float(
        config["promotion"][
            "historical_test_maximum_drawdown_not_worse_than_control_by_more_than"
        ]
    )
    gates = {
        "validation_selected_before_historical_evaluation": True,
        "historical_sharpe_higher_than_monthly_control": float(challenger_summary["net_sharpe"])
        > float(control_summary["net_sharpe"]),
        "historical_return_higher_than_monthly_control": float(
            challenger_summary["annualized_net_excess_return"]
        )
        > float(control_summary["annualized_net_excess_return"]),
        "historical_sharpe_higher_than_frozen_xgboost": float(challenger_summary["net_sharpe"])
        > float(xgb_summary["net_sharpe"]),
        "historical_drawdown_within_tolerance": float(challenger_summary["maximum_drawdown"])
        >= float(control_summary["maximum_drawdown"]) - tolerance,
        "paired_hac_t_versus_control": float(versus_control["paired_hac_t"])
        >= float(config["promotion"]["paired_monthly_difference_hac_t_minimum"]),
    }
    qualifies = all(gates.values())
    terminal = (
        "V6_7_MLP_OVERLAY_FORWARD_SHADOW_CANDIDATE_ONLY"
        if qualifies
        else "RETAIN_MONTHLY_REBALANCE_CONTROL_NO_DAILY_MLP_PROMOTION"
    )
    interpretation = (
        "The validation-selected neural challenger clears the bounded historical economic gates. Because the same historical period was already visible before V6.7, it may only enter prospective shadow paper trading; it does not replace the monthly primary from this evidence."
        if qualifies
        else "The validation-selected neural challenger does not clear every frozen return, Sharpe, drawdown, and paired-inference gate. The monthly portfolio therefore remains primary; the neural daily score may be logged only as a non-trading diagnostic."
    )
    result = {
        "schema": "research_v6_7_daily_mlp_result_v1",
        "status": "COMPLETE",
        "identity": identity,
        "data_quality": {
            "position_days": int(len(days)),
            "months": int(days["holding_month"].nunique()),
            "unique_position_day_keys": int(
                days[["holding_month", "calendar_date", "permno"]]
                .drop_duplicates()
                .shape[0]
            ),
            "encoded_feature_count": int(
                config["sources"]["v6_encoded_inputs"]["encoded_feature_count"]
            ),
            "dynamic_feature_count": len(base.DYNAMIC_COLUMNS),
            "mlp_seed_count": len(config["mlp_models"]["seeds"]),
            "validation_candidate_count": len(candidates),
        },
        "selection": {
            "period": f"{config['chronology']['selection_start']}--{config['chronology']['selection_end']}",
            "monthly_control": v6_6_selection["control"],
            "frozen_xgboost": v6_6_selection["selected_predictive"]["summary"],
            "selected_v6_7_challenger": selected,
        },
        "historical_test": {
            "monthly_control": control_summary,
            "frozen_xgboost": xgb_summary,
            "selected_v6_7_challenger": challenger_summary,
        },
        "paired_differences": {
            "versus_monthly_control": versus_control,
            "versus_frozen_xgboost": versus_xgb,
        },
        "promotion_gates": gates,
        "decision": {
            "terminal_decision": terminal,
            "all_gates_pass": qualifies,
            "permitted_status_if_all_gates_pass": config["promotion"][
                "promotion_status_even_if_all_gates_pass"
            ],
            "interpretation": interpretation,
        },
    }
    _atomic_json(output / "result.json", result)
    _atomic_text(output / "RESULT.md", _result_markdown(result))
    _atomic_json(
        manifest_path,
        {
            "schema": "research_v6_7_daily_mlp_run_v1",
            "status": "COMPLETE",
            "identity": identity,
            "result": "result.json",
            "report": "RESULT.md",
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(run(arguments.config.resolve(), arguments.output_root.resolve()))


if __name__ == "__main__":
    main()
