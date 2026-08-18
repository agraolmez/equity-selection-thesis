"""Causal refit and current scoring for the frozen V6.8 monthly selectors."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRanker, XGBRegressor

from portfolio_bot.research_v6.models import SelectorMLP, rank_within_month
from portfolio_bot.research_v6.selector_runtime import (
    canonical_sha256,
    is_refit_month,
    is_retune_month,
)
from portfolio_bot.research_v6_1.models import fit_mlp, predict_mlp
from portfolio_bot.research_v6_1.runtime import choose_profile


class ProspectiveSelectorError(RuntimeError):
    """The current selector refit or score package failed closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


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
    eligible = months.lt(forecast_month) & availability.le(signal_date) & np.isfinite(target)
    if history == "rolling":
        observed = sorted(months.loc[eligible].unique())
        if history_months is None or history_months < 24:
            raise ProspectiveSelectorError("rolling selector history is invalid")
        eligible &= months.isin(set(observed[-history_months:]))
    elif history != "expanding":
        raise ProspectiveSelectorError(f"unknown history policy: {history}")
    index = np.flatnonzero(eligible.to_numpy()).astype(np.int64)
    if len(index) == 0:
        raise ProspectiveSelectorError("no label-available training rows")
    return index


def _profiles(config: Mapping[str, Any], family: str) -> list[dict[str, Any]]:
    if family == "xgboost":
        return [dict(value) for value in config["xgboost"]["profiles"]]
    if family == "neural_network":
        return [dict(value) for value in config["neural_network"]["profiles"]]
    raise ProspectiveSelectorError(f"unsupported prospective family: {family}")


def _shared(config: Mapping[str, Any], family: str) -> Mapping[str, Any]:
    return config["xgboost"]["shared"] if family == "xgboost" else config["neural_network"]["shared"]


def _fit_mlp_component(
    *,
    historical: np.ndarray,
    live: np.ndarray,
    feature_index: np.ndarray,
    train_index: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    objective: str,
    profile: Mapping[str, Any],
    shared: Mapping[str, Any],
    seeds: list[int],
    model_root: Path,
) -> np.ndarray:
    x_train = np.asarray(historical[np.ix_(train_index, feature_index)], dtype=np.float32)
    x_live = np.asarray(live[:, feature_index], dtype=np.float32)
    predictions: list[np.ndarray] = []
    for seed in seeds:
        model = fit_mlp(
            x_train,
            target[train_index],
            months[train_index],
            objective=objective,
            seed=seed,
            hidden_layers=profile["hidden_layers"],
            dropout=float(profile["dropout"]),
            epochs=int(profile["epochs"]),
            learning_rate=float(profile["learning_rate"]),
            weight_decay=float(profile["weight_decay"]),
            batch_size=int(shared["batch_size"]),
            pairs_per_row=int(shared["pairs_per_row"]),
            gradient_clip=float(shared["gradient_clip"]),
        )
        predictions.append(predict_mlp(model, x_live))
        model_root.mkdir(parents=True, exist_ok=True)
        target_path = model_root / f"seed-{seed}.pt"
        temporary = target_path.with_suffix(".pt.tmp")
        torch.save(model.state_dict(), temporary)
        os.replace(temporary, target_path)
    return np.mean(np.vstack(predictions), axis=0)


def _fit_xgboost_component(
    *,
    project_root: Path,
    historical_path: Path,
    live_path: Path,
    feature_index: np.ndarray,
    train_index: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    objective: str,
    profile: Mapping[str, Any],
    shared: Mapping[str, Any],
    seeds: list[int],
    model_root: Path,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    parameters = {**dict(profile), **dict(shared)}
    parameters.pop("profile_id", None)
    with tempfile.TemporaryDirectory(prefix="v6_8_live_xgb_") as temporary_name:
        temporary = Path(temporary_name)
        np.save(temporary / "train.npy", train_index)
        np.save(temporary / "features.npy", feature_index)
        np.save(temporary / "target.npy", target[train_index])
        np.save(temporary / "months.npy", months[train_index].astype("U7"))
        model_root.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            prediction_path = temporary / f"prediction-{seed}.npy"
            importance_path = model_root / f"seed-{seed}.importance.npy"
            model_path = model_root / f"seed-{seed}.json"
            command = [
                "/opt/anaconda3/envs/finance/bin/python",
                str(project_root / "scripts/run_v6_8_xgboost_worker.py"),
                "--historical-matrix", str(historical_path),
                "--live-matrix", str(live_path),
                "--train-index", str(temporary / "train.npy"),
                "--feature-index", str(temporary / "features.npy"),
                "--train-target", str(temporary / "target.npy"),
                "--train-months", str(temporary / "months.npy"),
                "--objective", objective,
                "--parameters-json", json.dumps(parameters, sort_keys=True),
                "--seed", str(seed),
                "--prediction-output", str(prediction_path),
                "--importance-output", str(importance_path),
                "--model-output", str(model_path),
            ]
            subprocess.run(command, cwd=project_root, check=True)
            predictions.append(np.load(prediction_path).astype(float))
    return np.mean(np.vstack(predictions), axis=0)


def _predict_mlp_state(
    *,
    live: np.ndarray,
    feature_index: np.ndarray,
    profile: Mapping[str, Any],
    seeds: list[int],
    model_root: Path,
) -> np.ndarray:
    x_live = np.asarray(live[:, feature_index], dtype=np.float32)
    predictions: list[np.ndarray] = []
    for seed in seeds:
        path = model_root / f"seed-{seed}.pt"
        if not path.is_file():
            raise ProspectiveSelectorError(f"MLP state is incomplete: {path}")
        model = SelectorMLP(
            len(feature_index), profile["hidden_layers"], float(profile["dropout"])
        )
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        predictions.append(predict_mlp(model, x_live))
    return np.mean(np.vstack(predictions), axis=0)


def _predict_xgboost_state(
    *,
    live: np.ndarray,
    feature_index: np.ndarray,
    objective: str,
    seeds: list[int],
    model_root: Path,
) -> np.ndarray:
    x_live = np.asarray(live[:, feature_index], dtype=np.float32)
    predictions: list[np.ndarray] = []
    for seed in seeds:
        path = model_root / f"seed-{seed}.json"
        if not path.is_file():
            raise ProspectiveSelectorError(f"XGBoost state is incomplete: {path}")
        model = XGBRanker() if objective == "pairwise" else XGBRegressor()
        model.load_model(path)
        predictions.append(np.asarray(model.predict(x_live), dtype=float))
    return np.mean(np.vstack(predictions), axis=0)


def _model_hashes(model_root: Path, family: str, seeds: list[int]) -> dict[str, str]:
    suffix = ".pt" if family == "neural_network" else ".json"
    paths = {f"seed-{seed}": model_root / f"seed-{seed}{suffix}" for seed in seeds}
    if any(not path.is_file() for path in paths.values()):
        raise ProspectiveSelectorError("selector state omits a frozen seed model")
    return {name: _sha256(path) for name, path in paths.items()}


def materialize_prospective_selector_scores(
    *,
    config_path: Path,
    feature_package: Path,
    output: Path,
    state_root: Path | None = None,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "research_v6_8_prospective_selector_v1":
        raise ProspectiveSelectorError("prospective selector config schema differs")
    package_receipt = json.loads((feature_package / "receipt.json").read_text(encoding="utf-8"))
    if package_receipt.get("status") != "PASS":
        raise ProspectiveSelectorError("current feature package has not passed every source group")
    live_path = feature_package / "encoded_values.npy"
    if package_receipt.get("encoded_values_sha256") != _sha256(live_path):
        raise ProspectiveSelectorError("current encoded matrix differs from its package receipt")
    live_rows = pd.read_parquet(feature_package / "encoded_rows.parquet")
    live = np.load(live_path, mmap_mode="r")
    if len(live_rows) != len(live):
        raise ProspectiveSelectorError("current encoded rows and matrix differ")
    holding_months = live_rows["holding_month"].astype(str)
    signal_dates = pd.to_datetime(live_rows["signal_date"], errors="raise")
    if holding_months.nunique() != 1 or signal_dates.nunique() != 1:
        raise ProspectiveSelectorError("current selector package must contain one month")
    forecast_month = str(holding_months.iloc[0])
    signal_date = pd.Timestamp(signal_dates.iloc[0])

    v61_path = _resolve(project_root, str(config["v6_1_selector_config"]))
    v61 = json.loads(v61_path.read_text(encoding="utf-8"))
    historical_path = _resolve(project_root, str(config["historical_matrix"]))
    rows_path = _resolve(project_root, str(config["historical_rows"]))
    metadata_path = _resolve(project_root, str(config["historical_metadata"]))
    views_path = _resolve(project_root, str(config["frozen_feature_views"]))
    historical = np.load(historical_path, mmap_mode="r")
    rows = pd.read_parquet(rows_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    views = json.loads(views_path.read_text(encoding="utf-8"))
    if historical.shape[0] != len(rows) or historical.shape[1] != live.shape[1]:
        raise ProspectiveSelectorError("historical and current encoded schemas differ")
    seeds = [int(value) for value in config["seeds"]]
    months = rows["holding_month"].astype(str).to_numpy()
    scored = live_rows.copy()
    component_receipts: dict[str, Any] = {}
    state_root = state_root or _resolve(project_root, str(config["state_directory"]))
    stable_hashes = {
        "config_sha256": _sha256(config_path),
        "v6_1_selector_config_sha256": _sha256(v61_path),
        "historical_matrix_sha256": _sha256(historical_path),
        "historical_rows_sha256": _sha256(rows_path),
        "feature_views_sha256": _sha256(views_path),
    }
    input_hashes = {
        **stable_hashes,
        "feature_package_sha256": _sha256(feature_package / "receipt.json"),
    }

    for component, specification in config["components"].items():
        family = str(specification["family"])
        target_name = str(specification["target"])
        target_column = str(specification["target_column"])
        objective = str(specification["objective"])
        view = str(specification["feature_view"])
        feature_index = np.asarray(views["views"][view]["encoded_indices"], dtype=np.int64)
        target = pd.to_numeric(rows[target_column], errors="coerce").to_numpy(float)
        schedule = v61["operating_policy"][family][target_name]
        train_index = _training_indices(
            rows,
            target,
            forecast_month=forecast_month,
            signal_date=signal_date,
            history=str(schedule["history"]),
            history_months=(int(schedule["history_months"]) if schedule.get("history_months") else None),
        )
        component_root = output / "components" / component
        state_component = state_root / component
        model_root = state_component / "models"
        profile_path = state_component / "profile.json"
        model_receipt_path = state_component / "model_receipt.json"
        prediction_path = component_root / "prediction.npy"
        receipt_path = component_root / "receipt.json"
        profile_identity = canonical_sha256(
            {
                **stable_hashes,
                "component": component,
                "specification": specification,
                "feature_index": feature_index.tolist(),
                "seeds": seeds,
            }
        )
        prior_profile: dict[str, Any] | None = None
        if profile_path.is_file():
            candidate = json.loads(profile_path.read_text(encoding="utf-8"))
            if candidate.get("identity") != profile_identity:
                raise ProspectiveSelectorError(
                    f"selector profile state identity changed: {component}"
                )
            prior_profile = candidate
        retune_due = is_retune_month(
            forecast_month, str(schedule["retune"]), first=prior_profile is None
        )
        retune = bool(
            retune_due
            and (
                prior_profile is None
                or prior_profile.get("selected_for_month") != forecast_month
            )
        )
        if retune:
            profile, tuning = choose_profile(
                family=family,
                objective=objective,
                profiles=_profiles(v61, family),
                shared=_shared(v61, family),
                matrix=historical,
                feature_index=feature_index,
                rows=rows,
                target=target,
                train_index=train_index,
                months=months,
                holdout_months=int(v61["causal_tuning"]["holdout_months"]),
                seed=int(v61["causal_tuning"]["tuning_seed"]),
                tie_tolerance=float(v61["causal_tuning"]["profile_tie_tolerance"]),
            )
            profile_receipt = {
                "schema": "research_v6_8_prospective_profile_v1",
                "status": "PASS",
                "identity": profile_identity,
                "selected_profile": profile,
                "selected_for_month": forecast_month,
                "tuning_evidence": tuning,
            }
            _atomic_json(profile_path, profile_receipt)
        else:
            assert prior_profile is not None
            profile_receipt = prior_profile
            profile = dict(profile_receipt["selected_profile"])
            tuning = list(profile_receipt["tuning_evidence"])

        prior_model: dict[str, Any] | None = None
        if model_receipt_path.is_file():
            prior_model = json.loads(model_receipt_path.read_text(encoding="utf-8"))
            if (
                prior_model.get("profile_identity") != profile_identity
                or prior_model.get("selected_profile") != profile
            ):
                prior_model = None
        refit_due = is_refit_month(
            forecast_month, str(schedule["refit"]), first=prior_model is None
        ) or retune
        refit = bool(
            refit_due
            and (
                prior_model is None
                or prior_model.get("fitted_for_month") != forecast_month
            )
        )
        if refit:
            if family == "neural_network":
                raw = _fit_mlp_component(
                    historical=historical,
                    live=live,
                    feature_index=feature_index,
                    train_index=train_index,
                    target=target,
                    months=months,
                    objective=objective,
                    profile=profile,
                    shared=_shared(v61, family),
                    seeds=seeds,
                    model_root=model_root,
                )
            else:
                raw = _fit_xgboost_component(
                    project_root=project_root,
                    historical_path=historical_path,
                    live_path=live_path,
                    feature_index=feature_index,
                    train_index=train_index,
                    target=target,
                    months=months,
                    objective=objective,
                    profile=profile,
                    shared=_shared(v61, family),
                    seeds=seeds,
                    model_root=model_root,
                )
            model_receipt = {
                "schema": "research_v6_8_prospective_model_state_v1",
                "status": "PASS",
                "profile_identity": profile_identity,
                "selected_profile": profile,
                "fitted_for_month": forecast_month,
                "train_rows": int(len(train_index)),
                "train_first_month": str(months[train_index].min()),
                "train_last_month": str(months[train_index].max()),
                "training_index_sha256": canonical_sha256(train_index.tolist()),
                "model_sha256s": _model_hashes(model_root, family, seeds),
            }
            _atomic_json(model_receipt_path, model_receipt)
        else:
            assert prior_model is not None
            model_receipt = prior_model
            if model_receipt.get("model_sha256s") != _model_hashes(model_root, family, seeds):
                raise ProspectiveSelectorError(f"selector model state changed: {component}")
            if family == "neural_network":
                raw = _predict_mlp_state(
                    live=live, feature_index=feature_index, profile=profile,
                    seeds=seeds, model_root=model_root,
                )
            else:
                raw = _predict_xgboost_state(
                    live=live, feature_index=feature_index, objective=objective,
                    seeds=seeds, model_root=model_root,
                )

        identity = canonical_sha256(
            {
                **input_hashes,
                "component": component,
                "model_state_receipt_sha256": _sha256(model_receipt_path),
            }
        )
        if prediction_path.is_file() and receipt_path.is_file():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            if prior.get("identity") != identity:
                raise ProspectiveSelectorError(f"component checkpoint identity changed: {component}")
            raw = np.load(prediction_path)
            component_receipts[component] = prior
        else:
            component_root.mkdir(parents=True, exist_ok=True)
            temporary = prediction_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as destination:
                np.save(destination, raw)
            os.replace(temporary, prediction_path)
            receipt = {
                "schema": "research_v6_8_prospective_component_v1",
                "status": "PASS",
                "identity": identity,
                "component": component,
                "selected_profile": profile,
                "tuning_evidence": tuning,
                "retuned": bool(retune),
                "refitted": bool(refit),
                "model_fitted_for_month": model_receipt["fitted_for_month"],
                "train_rows": int(len(train_index)),
                "train_first_month": str(months[train_index].min()),
                "train_last_month": str(months[train_index].max()),
                "forecast_month": forecast_month,
                "feature_count": int(len(feature_index)),
                "prediction_sha256": _sha256(prediction_path),
            }
            _atomic_json(receipt_path, receipt)
            component_receipts[component] = receipt
        scored[component] = rank_within_month(raw, holding_months)

    source = str(config["direct_cam_source"])
    names = list(map(str, metadata["feature_names"]))
    try:
        cam_index = names.index(source)
        cam_missing_index = names.index(source + "__missing")
    except ValueError as error:
        raise ProspectiveSelectorError("direct CAM columns differ from frozen metadata") from error
    scored["direct_cam"] = rank_within_month(np.asarray(live[:, cam_index], dtype=float), holding_months)
    scored["direct_cam_available"] = np.asarray(live[:, cam_missing_index] < 0.5, dtype=bool)
    scored["ff5_momentum_score"] = rank_within_month(
        (scored["ff5_momentum_mlp"] + scored["direct_cam"]).to_numpy(float) / 2.0,
        holding_months,
    )
    scored["long_net_score"] = rank_within_month(
        (scored["long_net_mlp"] + scored["direct_cam"]).to_numpy(float) / 2.0,
        holding_months,
    )
    scored["short_net_score"] = rank_within_month(
        (scored["short_net_xgboost"] + scored["short_net_mlp"]).to_numpy(float) / 2.0,
        holding_months,
    )
    score_path = output / "selector_scores.parquet"
    _atomic_parquet(score_path, scored)
    receipt = {
        "schema": "research_v6_8_prospective_selector_receipt_v1",
        "status": "PASS",
        "forecast_month": forecast_month,
        "signal_date": signal_date.date().isoformat(),
        "rows": int(len(scored)),
        "seeds": seeds,
        "components": component_receipts,
        "scores_sha256": _sha256(score_path),
        "feature_package_receipt_sha256": _sha256(feature_package / "receipt.json"),
    }
    _atomic_json(output / "receipt.json", receipt)
    return receipt


__all__ = ["ProspectiveSelectorError", "materialize_prospective_selector_scores"]
