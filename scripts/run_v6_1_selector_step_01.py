#!/usr/bin/env python3
"""Prepare, smoke-test, or run the separate Research V6.1 selector study."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
STABLE_PYTHON = Path("/opt/anaconda3/envs/finance/bin/python")
DEFAULT_CONFIG = ROOT / "configs/research_v6_1/selector_step_01.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/research_v6_1/selector_step_01"
CODE_PATHS = (
    ROOT / "scripts/run_v6_1_selector_step_01.py",
    ROOT / "scripts/run_v6_1_xgboost_worker.py",
    ROOT / "src/portfolio_bot/research_v6_1/feature_views.py",
    ROOT / "src/portfolio_bot/research_v6_1/models.py",
    ROOT / "src/portfolio_bot/research_v6_1/runtime.py",
    ROOT / "src/portfolio_bot/research_v6/evaluation.py",
    ROOT / "src/portfolio_bot/research_v6/models.py",
    ROOT / "src/portfolio_bot/research_v6/selector_runtime.py",
)

from portfolio_bot.research_v6.evaluation import (  # noqa: E402
    monthly_selector_metrics,
    summarize_selector_metrics,
)
from portfolio_bot.research_v6.selector_runtime import (  # noqa: E402
    atomic_json,
    atomic_parquet,
    canonical_sha256,
)
from portfolio_bot.research_v6_1.feature_views import (  # noqa: E402
    direct_cam_predictions,
    prepare_feature_views,
)
from portfolio_bot.research_v6_1.models import (  # noqa: E402
    add_dynamic_ensemble,
    add_equal_ensemble,
)
from portfolio_bot.research_v6_1.runtime import walk_forward_predictions  # noqa: E402


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_identity(config_path: Path) -> str:
    return canonical_sha256(
        {
            "config_sha256": _file_sha256(config_path),
            "code": {str(path.relative_to(ROOT)): _file_sha256(path) for path in CODE_PATHS},
        }
    )


def _v6_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    root = ROOT / str(config["inputs"]["v6_run"])
    return {
        "root": root,
        "run_manifest": root / "run_manifest.json",
        "matrix": root / "input_cache/encoded_values.npy",
        "rows": root / "input_cache/rows.parquet",
        "metadata": root / "input_cache/feature_metadata.json",
        "receipt": root / "input_cache/receipt.json",
    }


def _verify_inputs(config: Mapping[str, Any], *, load_matrix: bool = False) -> dict[str, Any]:
    paths = _v6_paths(config)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V6.1 predecessor inputs are missing: {missing}")
    run = json.loads(paths["run_manifest"].read_text())
    receipt = json.loads(paths["receipt"].read_text())
    metadata = json.loads(paths["metadata"].read_text())
    expected = config["inputs"]
    if run.get("status") != "COMPLETE" or sorted(run.get("targets", [])) != [
        "capm", "ff5_momentum", "long_net", "short_net"
    ]:
        raise ValueError("V6 predecessor run is not complete for all four targets")
    checks = {
        "V6 contract": (run.get("config_sha256"), expected["v6_contract_sha256"]),
        "V6 code": (run.get("code_identity"), expected["v6_code_identity"]),
        "V6 input": (run.get("input_identity"), expected["v6_input_identity"]),
        "cache input": (receipt.get("input_identity"), expected["v6_input_identity"]),
    }
    for label, (observed, wanted) in checks.items():
        if observed != wanted:
            raise ValueError(f"{label} identity changed: {observed} != {wanted}")
    rows = pd.read_parquet(paths["rows"])
    if len(rows) != int(expected["expected_rows"]):
        raise ValueError("V6.1 predecessor row count changed")
    if rows["holding_month"].nunique() != int(expected["expected_months"]):
        raise ValueError("V6.1 predecessor month count changed")
    if not rows["holding_month"].astype(str).eq("2019-12").any():
        raise ValueError("V6.1 predecessor cache omits December 2019")
    if len(metadata["feature_family"]) != int(expected["expected_raw_features"]):
        raise ValueError("V6.1 predecessor raw feature count changed")
    if len(metadata["feature_names"]) != int(expected["expected_encoded_features"]):
        raise ValueError("V6.1 predecessor encoded feature count changed")
    result: dict[str, Any] = {"paths": paths, "run": run, "receipt": receipt, "metadata": metadata, "rows": rows}
    if load_matrix:
        matrix = np.load(paths["matrix"], mmap_mode="r")
        if matrix.shape != (int(expected["expected_rows"]), int(expected["expected_encoded_features"])):
            raise ValueError("V6.1 predecessor matrix shape changed")
        result["matrix"] = matrix
    return result


def _elastic_profiles(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"profile_id": f"enet_a{alpha:g}_l{ratio:g}", "alpha": alpha, "l1_ratio": ratio}
        for alpha in config["elastic_net"]["alpha_grid"]
        for ratio in config["elastic_net"]["l1_ratio_grid"]
    ]


def _model_settings(config: Mapping[str, Any], family: str) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    if family == "xgboost":
        return list(config["xgboost"]["profiles"]), config["xgboost"]["shared"]
    if family == "neural_network":
        return list(config["neural_network"]["profiles"]), config["neural_network"]["shared"]
    if family == "elastic_net":
        return _elastic_profiles(config), {}
    raise ValueError(f"unknown V6.1 family: {family}")


def _tuning(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["causal_tuning"]
    return {
        "holdout_months": int(source["holdout_months"]),
        "tuning_seed": int(source["tuning_seed"]),
        "profile_tie_tolerance": float(source["profile_tie_tolerance"]),
    }


def _summary(
    prediction: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    score_column: str,
    target_column: str,
    economic_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = prediction.merge(
        rows[["holding_month", "permno", target_column, economic_column]],
        on=["holding_month", "permno"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"score_rank": score_column})
    monthly = monthly_selector_metrics(
        joined,
        score_columns=[score_column],
        target_column=target_column,
        economic_column=economic_column,
    )
    return monthly, summarize_selector_metrics(monthly)


def _choose_objective(summary: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    finite = summary.loc[np.isfinite(summary["mean_rank_ic"])].copy()
    if finite.empty:
        raise RuntimeError("V6.1 objective selection has no finite validation result")
    tolerance = float(config["causal_tuning"]["objective_tie_tolerance"])
    admissible = finite.loc[finite["mean_rank_ic"].ge(float(finite["mean_rank_ic"].max()) - tolerance)].copy()
    preferred = str(config["causal_tuning"]["objective_tie_preference"])
    admissible["objective_preference"] = admissible["objective"].ne(preferred).astype(int)
    return admissible.sort_values(["objective_preference", "objective"], kind="mergesort").iloc[0]


def _seed_stability(prediction: pd.DataFrame, *, component: str) -> pd.DataFrame:
    seed_columns = sorted(column for column in prediction if column.startswith("score_rank__seed_"))
    rows: list[dict[str, Any]] = []
    for left_position, left in enumerate(seed_columns):
        for right in seed_columns[left_position + 1 :]:
            monthly: list[float] = []
            for _, group in prediction.groupby("holding_month", observed=True, sort=True):
                monthly.append(float(group[left].corr(group[right], method="spearman")))
            rows.append(
                {
                    "component": component,
                    "left_seed": int(left.rsplit("_", 1)[1]),
                    "right_seed": int(right.rsplit("_", 1)[1]),
                    "mean_monthly_rank_correlation": float(np.nanmean(monthly)),
                    "minimum_monthly_rank_correlation": float(np.nanmin(monthly)),
                }
            )
    return pd.DataFrame(rows)


def _turnover(frame: pd.DataFrame, score_columns: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    months = sorted(frame["holding_month"].astype(str).unique())
    for score in score_columns:
        previous: pd.DataFrame | None = None
        for month in months:
            current = frame.loc[frame["holding_month"].eq(month), ["permno", score]].dropna()
            if previous is not None:
                common = previous.merge(current, on="permno", suffixes=("_previous", "_current"))
                persistence = (
                    float(common[f"{score}_previous"].corr(common[f"{score}_current"], method="spearman"))
                    if len(common) >= 10 else np.nan
                )
                tail = max(len(current) // 10, 1)
                current_top = set(current.nlargest(tail, score)["permno"])
                previous_tail = max(len(previous) // 10, 1)
                previous_top = set(previous.nlargest(previous_tail, score)["permno"])
                rows.append(
                    {
                        "holding_month": month,
                        "model_id": score,
                        "common_stocks": len(common),
                        "rank_persistence": persistence,
                        "top_decile_retention": len(current_top & previous_top) / max(len(current_top), 1),
                        "top_decile_churn": 1.0 - len(current_top & previous_top) / max(len(current_top), 1),
                    }
                )
            previous = current
    return pd.DataFrame(rows)


def _subgroup_metrics(
    frame: pd.DataFrame,
    *,
    score_columns: Sequence[str],
    target_column: str,
    economic_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    labels = {"size_input": ["small", "middle", "large"], "liquidity_input": ["low", "middle", "high"]}
    for dimension, names in labels.items():
        rank = frame.groupby("holding_month", observed=True)[dimension].rank(method="average", pct=True)
        group = pd.cut(rank, bins=[0.0, 1 / 3, 2 / 3, 1.0], labels=names, include_lowest=True)
        for subgroup in names:
            selected = frame.loc[group.eq(subgroup)]
            metrics = monthly_selector_metrics(
                selected,
                score_columns=score_columns,
                target_column=target_column,
                economic_column=economic_column,
            )
            if not metrics.empty:
                parts.append(metrics.assign(dimension=dimension, subgroup=subgroup))
    monthly = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    summaries: list[pd.DataFrame] = []
    if not monthly.empty:
        for (dimension, subgroup), group in monthly.groupby(["dimension", "subgroup"], observed=True):
            summaries.append(
                summarize_selector_metrics(group).assign(dimension=dimension, subgroup=subgroup)
            )
    return monthly, pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def _run_model(
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
    start: str,
    end: str,
    seeds: Sequence[int],
    tuning: Mapping[str, Any],
    output: Path,
    run_identity: str,
) -> pd.DataFrame:
    prediction, _, _ = walk_forward_predictions(
        matrix=matrix,
        rows=rows,
        feature_index=feature_index,
        target_column=target_column,
        family=family,
        objective=objective,
        candidate=candidate,
        profiles=profiles,
        shared=shared,
        start_month=start,
        end_month=end,
        seeds=seeds,
        tuning=tuning,
        output=output,
        run_identity=run_identity,
    )
    return prediction


def _run_target(
    target_id: str,
    config_path: str,
    output_path: str,
    feature_manifest_path: str,
    code_identity: str,
) -> str:
    config = json.loads(Path(config_path).read_text())
    verified = _verify_inputs(config, load_matrix=True)
    matrix = verified["matrix"]
    rows = verified["rows"]
    metadata = verified["metadata"]
    feature_manifest = json.loads(Path(feature_manifest_path).read_text())
    output = Path(output_path) / target_id
    output.mkdir(parents=True, exist_ok=True)
    target_spec = config["targets"][target_id]
    target_column = str(target_spec["rank_column"])
    economic_column = str(target_spec["economic_column"])
    chronology = config["chronology"]
    validation_seed = list(config["causal_tuning"]["validation_final_fit_seeds"])
    test_seeds = list(config["causal_tuning"]["historical_test_final_fit_seeds"])
    tuning = _tuning(config)
    base_identity = canonical_sha256(
        {
            "code_identity": code_identity,
            "input_identity": config["inputs"]["v6_input_identity"],
            "feature_manifest": feature_manifest["identity"],
            "target": target_id,
        }
    )
    validation_choices: list[dict[str, Any]] = []
    component_predictions: dict[str, pd.DataFrame] = {}
    seed_frames: list[pd.DataFrame] = []
    for view in config["feature_views"]["views"]:
        feature_index = np.asarray(feature_manifest["views"][view]["encoded_indices"], dtype=np.int64)
        for family in ("xgboost", "neural_network", "elastic_net"):
            profiles, shared = _model_settings(config, family)
            candidate = config["operating_policy"][family][target_id] if family != "elastic_net" else config["operating_policy"]["elastic_net"]
            objectives = ["continuous_rank"] if family == "elastic_net" else list(config["objectives"])
            candidates: list[dict[str, Any]] = []
            validation_predictions: dict[str, pd.DataFrame] = {}
            for objective in objectives:
                task = f"{view}__{family}__{objective}"
                prediction = _run_model(
                    matrix=matrix,
                    rows=rows,
                    feature_index=feature_index,
                    target_column=target_column,
                    family=family,
                    objective=objective,
                    candidate=candidate,
                    profiles=profiles,
                    shared=shared,
                    start=chronology["validation_start"],
                    end=chronology["validation_end"],
                    seeds=validation_seed,
                    tuning=tuning,
                    output=output / "validation_candidates" / task,
                    run_identity=canonical_sha256({"base": base_identity, "task": task}),
                )
                _, summary = _summary(
                    prediction, rows,
                    score_column=task,
                    target_column=target_column,
                    economic_column=economic_column,
                )
                candidates.append(
                    {
                        **summary.iloc[0].to_dict(),
                        "feature_view": view,
                        "family": family,
                        "objective": objective,
                        "task_id": task,
                    }
                )
                validation_predictions[objective] = prediction
            choice = _choose_objective(pd.DataFrame(candidates), config)
            validation_choices.extend(candidates)
            selected_objective = str(choice["objective"])
            task = f"{view}__{family}__{selected_objective}"
            test = _run_model(
                matrix=matrix,
                rows=rows,
                feature_index=feature_index,
                target_column=target_column,
                family=family,
                objective=selected_objective,
                candidate=candidate,
                profiles=profiles,
                shared=shared,
                start=chronology["historical_test_start"],
                end=chronology["historical_test_end"],
                seeds=test_seeds,
                tuning=tuning,
                output=output / "historical_test" / task,
                run_identity=canonical_sha256({"base": base_identity, "task": task, "period": "historical_test"}),
            )
            component = f"{view}__{'mlp' if family == 'neural_network' else family}"
            component_predictions[component] = pd.concat(
                [validation_predictions[selected_objective], test], ignore_index=True
            ).sort_values(["holding_month", "permno"], kind="mergesort")
            stability = _seed_stability(test, component=component)
            if not stability.empty:
                seed_frames.append(stability)

    validation_summary = pd.DataFrame(validation_choices)
    validation_summary["selected_objective"] = False
    for (view, family), group in validation_summary.groupby(["feature_view", "family"], observed=True):
        choice = _choose_objective(group, config)
        validation_summary.loc[choice.name, "selected_objective"] = True
    atomic_parquet(validation_summary, output / "validation_objective_summary.parquet")

    selected_rows = rows.loc[
        rows["holding_month"].between(chronology["validation_start"], chronology["historical_test_end"]),
        ["holding_month", "signal_date", "permno", "permco", target_column, economic_column],
    ].copy()
    official = selected_rows
    stratification = config["evaluation"]["size_liquidity_stratification"]
    encoded_lookup = {str(name): index for index, name in enumerate(metadata["feature_names"])}
    for output_column, source in (
        ("size_input", str(stratification["size_source"])),
        ("liquidity_input", str(stratification["liquidity_source"])),
    ):
        if source not in encoded_lookup:
            raise ValueError(f"V6.1 stratification source is missing: {source}")
        official[output_column] = np.asarray(
            matrix[official.index.to_numpy(np.int64), encoded_lookup[source]], dtype=float
        )
    for component, prediction in component_predictions.items():
        official = official.merge(
            prediction[["holding_month", "permno", "score_rank"]].rename(columns={"score_rank": component}),
            on=["holding_month", "permno"],
            validate="one_to_one",
        )
    direct_cam = direct_cam_predictions(
        matrix=matrix,
        rows=rows,
        metadata=metadata,
        source=str(config["feature_views"]["direct_cam_level"]),
        start_month=chronology["validation_start"],
        end_month=chronology["historical_test_end"],
    )
    official = official.merge(
        direct_cam[["holding_month", "permno", "score_rank", "cam_available"]].rename(columns={"score_rank": "cam_direct"}),
        on=["holding_month", "permno"],
        validate="one_to_one",
    )

    v6_path = Path(config_path).resolve().parents[2] / str(config["inputs"]["v6_run"])
    v6 = pd.read_parquet(v6_path / target_id / "official_predictions.parquet")
    comparator_columns = ["xgboost_rank", "mlp_rank", "elastic_net_rank", "ensemble_rank"]
    official = official.merge(
        v6[["holding_month", "permno", *comparator_columns]].rename(columns={column: f"v6_frozen__{column}" for column in comparator_columns}),
        on=["holding_month", "permno"],
        validate="one_to_one",
    )

    score_columns = list(component_predictions) + ["cam_direct"] + [f"v6_frozen__{column}" for column in comparator_columns]
    dynamic_config = config["ensembles"]["dynamic"]
    for name, components in config["ensembles"]["sets"].items():
        equal_column = f"ensemble__{name}__equal"
        dynamic_column = f"ensemble__{name}__dynamic"
        official = add_equal_ensemble(official, components=components, output_column=equal_column)
        official = add_dynamic_ensemble(
            official,
            components=components,
            target_column=target_column,
            output_column=dynamic_column,
            half_life_months=float(dynamic_config["half_life_months"]),
            minimum_history_months=int(dynamic_config["minimum_history_months"]),
            minimum_component_weight=float(dynamic_config["minimum_component_weight"]),
            maximum_component_weight=float(dynamic_config["maximum_component_weight"]),
        )
        score_columns.extend([equal_column, dynamic_column])
    official["period"] = np.where(
        official["holding_month"].le(chronology["validation_end"]), "validation", "historical_test"
    )
    monthly = monthly_selector_metrics(
        official,
        score_columns=score_columns,
        target_column=target_column,
        economic_column=economic_column,
    ).merge(official[["holding_month", "period"]].drop_duplicates(), on="holding_month", validate="many_to_one")
    summary = pd.concat(
        [summarize_selector_metrics(group).assign(period=period) for period, group in monthly.groupby("period", observed=True, sort=True)],
        ignore_index=True,
    )
    gate = config["evaluation"]["interesting_selector_gate"]
    test_summary = summary.loc[summary["period"].eq("historical_test")].copy()
    test_summary["interesting_selector"] = (
        test_summary["mean_rank_ic"].ge(float(gate["historical_test_mean_rank_ic_minimum"]))
        & test_summary["rank_ic_hac_t"].ge(float(gate["historical_test_hac_t_minimum"]))
        & test_summary["early_mean_rank_ic"].gt(0.0)
        & test_summary["recent_mean_rank_ic"].gt(0.0)
        & test_summary["minimum_rolling_mean_rank_ic"].gt(0.0)
        & test_summary["mean_economic_top_minus_bottom"].gt(0.0)
    )
    pivot = monthly.pivot(index=["holding_month", "period"], columns="model_id", values="rank_ic").reset_index()
    paired_rows: list[pd.DataFrame] = []
    family_comparator = {
        "xgboost": "v6_frozen__xgboost_rank",
        "mlp": "v6_frozen__mlp_rank",
        "elastic_net": "v6_frozen__elastic_net_rank",
    }
    for component in component_predictions:
        family = component.split("__", 1)[1]
        comparator = family_comparator[family]
        paired_rows.append(
            pivot[["holding_month", "period"]].assign(
                model_id=component,
                v6_comparator=comparator,
                paired_rank_ic_delta=pivot[component] - pivot[comparator],
            )
        )
    paired = pd.concat(paired_rows, ignore_index=True)
    turnover = _turnover(official, score_columns)
    subgroup_monthly, subgroup_summary = _subgroup_metrics(
        official.loc[official["period"].eq("historical_test")],
        score_columns=score_columns,
        target_column=target_column,
        economic_column=economic_column,
    )
    atomic_parquet(official, output / "official_predictions.parquet")
    atomic_parquet(monthly, output / "official_monthly_metrics.parquet")
    atomic_parquet(summary, output / "official_summary.parquet")
    atomic_parquet(test_summary, output / "historical_test_gate_evaluation.parquet")
    atomic_parquet(paired, output / "paired_v6_rank_ic_deltas.parquet")
    atomic_parquet(turnover, output / "rank_turnover.parquet")
    atomic_parquet(subgroup_monthly, output / "size_liquidity_monthly_metrics.parquet")
    atomic_parquet(subgroup_summary, output / "size_liquidity_summary.parquet")
    atomic_parquet(
        pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame(),
        output / "seed_stability.parquet",
    )
    atomic_json(
        {
            "schema": "research_v6_1_target_completion_v1",
            "target": target_id,
            "base_identity": base_identity,
            "feature_view_identity": feature_manifest["identity"],
            "score_columns": score_columns,
            "portfolio_constructed": False,
            "optimizer_run": False,
        },
        output / "completion.json",
    )
    return target_id


def _label_shuffle_smoke(
    config: Mapping[str, Any],
    verified: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    output: Path,
    code_identity: str,
) -> None:
    policy = config["evaluation"]["label_shuffle_smoke"]
    if not policy["enabled"]:
        return
    rows = verified["rows"].copy()
    target_column = str(config["targets"]["long_net"]["rank_column"])
    rng = np.random.default_rng(int(policy["seed"]))
    rows[target_column] = rows.groupby("holding_month", observed=True)[target_column].transform(
        lambda values: rng.permutation(values.to_numpy())
    )
    profiles, shared = _model_settings(config, "elastic_net")
    feature_index = np.asarray(feature_manifest["views"][policy["feature_view"]]["encoded_indices"], dtype=np.int64)
    candidate = config["operating_policy"]["elastic_net"]
    prediction = _run_model(
        matrix=verified["matrix"], rows=rows, feature_index=feature_index,
        target_column=target_column, family="elastic_net", objective="continuous_rank",
        candidate=candidate, profiles=profiles[:3], shared=shared,
        start="2023-01", end="2023-12", seeds=[int(policy["seed"])], tuning=_tuning(config),
        output=output / "controls/label_shuffle",
        run_identity=canonical_sha256({"code": code_identity, "control": policy}),
    )
    monthly, summary = _summary(
        prediction, rows, score_column="label_shuffle_elastic_net",
        target_column=target_column,
        economic_column=str(config["targets"]["long_net"]["economic_column"]),
    )
    mean_ic = float(summary.iloc[0]["mean_rank_ic"])
    summary["passes_smoke_bound"] = abs(mean_ic) <= float(policy["maximum_absolute_mean_ic"])
    atomic_parquet(monthly, output / "controls/label_shuffle/monthly_metrics.parquet")
    atomic_parquet(summary, output / "controls/label_shuffle/summary.parquet")


def _smoke_test() -> None:
    rng = np.random.default_rng(20260813)
    months = pd.period_range("2017-01", "2020-01", freq="M")
    records: list[dict[str, Any]] = []
    values: list[list[float]] = []
    for month in months:
        signal = (month - 1).to_timestamp(how="end").normalize()
        available = month.to_timestamp(how="end").normalize()
        for stock in range(64):
            first, second, third = rng.normal(size=3)
            records.append(
                {
                    "holding_month": str(month), "signal_date": signal,
                    "permno": 10_000 + stock, "permco": 20_000 + stock,
                    "label_available_date": available,
                    "target": first - 0.2 * second,
                }
            )
            values.append([first, second, third])
    rows = pd.DataFrame(records)
    with tempfile.TemporaryDirectory(prefix="v6_1_smoke_") as temporary_name:
        temporary = Path(temporary_name)
        matrix_path = temporary / "matrix.npy"
        np.save(matrix_path, np.asarray(values, dtype=np.float32))
        matrix = np.load(matrix_path, mmap_mode="r")
        common = dict(
            matrix=matrix, rows=rows, feature_index=np.arange(3), target_column="target",
            candidate={"candidate_id": "annual", "refit": "annual", "history": "expanding", "retune": "annual"},
            start="2020-01", end="2020-01", seeds=[7],
            tuning={"holdout_months": 12, "tuning_seed": 7, "profile_tie_tolerance": 0.002},
        )
        tasks = [
            ("elastic_net", "continuous_rank", [{"profile_id": "enet", "alpha": 0.01, "l1_ratio": 0.5}], {}),
            ("neural_network", "pairwise", [{"profile_id": "mlp", "hidden_layers": [8], "dropout": 0.0, "epochs": 1, "learning_rate": 0.001, "weight_decay": 0.0001}], {"batch_size": 128, "pairs_per_row": 1, "gradient_clip": 5.0}),
            ("xgboost", "pairwise", [{"profile_id": "xgb", "max_depth": 2, "n_estimators": 12, "learning_rate": 0.1, "min_child_weight": 2, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 2.0}], {"tree_method": "hist", "max_bin": 32, "n_jobs": 1, "early_stopping_rounds": 3, "pair_method": "topk", "pairs_per_sample": 4}),
        ]
        for family, objective, profiles, shared in tasks:
            prediction = _run_model(
                family=family, objective=objective, profiles=profiles, shared=shared,
                output=temporary / family, run_identity=f"smoke-{family}", **common,
            )
            if len(prediction) != 64 or not np.isfinite(prediction["score_rank"]).all():
                raise ValueError(f"V6.1 {family} smoke test failed")
        sample = prediction[["holding_month", "permno"]].copy()
        sample["a"] = np.linspace(0.01, 0.99, 64)
        sample["b"] = sample["a"].iloc[::-1].to_numpy()
        sample["target"] = np.arange(64)
        blended = add_dynamic_ensemble(
            sample, components=["a", "b"], target_column="target", output_column="blend",
            half_life_months=12, minimum_history_months=24,
            minimum_component_weight=0.05, maximum_component_weight=0.70,
        )
        if not np.isfinite(blended["blend"]).all():
            raise ValueError("V6.1 ensemble smoke test failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--targets", nargs="+", choices=("capm", "ff5_momentum", "long_net", "short_net"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--prepare-views", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--native-runtime", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.native_runtime and Path(sys.executable).resolve() != STABLE_PYTHON.resolve():
        if not STABLE_PYTHON.is_file():
            raise FileNotFoundError(f"stable V6.1 runtime is missing: {STABLE_PYTHON}")
        os.execv(str(STABLE_PYTHON), [str(STABLE_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:], "--native-runtime"])

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    verified = _verify_inputs(config, load_matrix=args.prepare_views)
    if args.check_only:
        run_state = (
            "full fit is authorized"
            if config.get("status") == "FROZEN_READY_TO_RUN"
            and config["execution"]["full_run_allowed_by_current_status"]
            else "full fit remains locked"
        )
        print(
            "READY: V6.1 contract, immutable V6 input, 318 features, "
            "three views, four targets, depth-2-to-6 XGBoost, MLP, Elastic Net, "
            f"direct CAM and causal ensembles; {run_state}.", flush=True
        )
        return 0
    if args.smoke_test:
        _smoke_test()
        print("PASS: V6.1 synthetic XGBoost, MLP, Elastic Net and ensemble smoke tests.", flush=True)
        return 0
    code_identity = _code_identity(config_path)
    output = args.output_root.resolve() / f"{_file_sha256(config_path)[:12]}-{config['inputs']['v6_input_identity'][:12]}-{code_identity[:12]}"
    output.mkdir(parents=True, exist_ok=True)
    feature_matrix = (
        verified["matrix"]
        if "matrix" in verified
        else np.load(verified["paths"]["matrix"], mmap_mode="r")
    )
    feature_manifest = prepare_feature_views(
        matrix=feature_matrix,
        rows=verified["rows"], metadata=verified["metadata"], config=config,
        input_identity=str(config["inputs"]["v6_input_identity"]), output=output / "feature_views",
    )
    if args.prepare_views:
        views = feature_manifest["views"]
        print(
            "PREPARED: " + ", ".join(
                f"{name}={spec['raw_source_count']} raw/{spec['encoded_feature_count']} encoded"
                for name, spec in views.items()
            ) + "; no model fit started.", flush=True
        )
        return 0
    if config.get("status") != "FROZEN_READY_TO_RUN" or not config["execution"]["full_run_allowed_by_current_status"]:
        raise PermissionError(
            "V6.1 full fit is locked. Review the prepared contract, then explicitly approve changing status to FROZEN_READY_TO_RUN."
        )
    targets = args.targets or list(config["targets"])
    workers = max(1, min(args.workers or int(config["execution"]["default_parallel_targets"]), len(targets)))
    manifest_base = {
        "schema": "research_v6_1_selector_run_manifest_v1",
        "config": str(config_path), "config_sha256": _file_sha256(config_path),
        "code_identity": code_identity, "v6_input_identity": config["inputs"]["v6_input_identity"],
        "feature_view_identity": feature_manifest["identity"], "targets": targets,
        "workers": workers, "portfolio_constructed": False, "optimizer_run": False,
    }
    atomic_json({**manifest_base, "status": "RUNNING"}, output / "run_manifest.json")
    completed: list[str] = []
    arguments = (str(config_path), str(output), str(output / "feature_views/feature_views.json"), code_identity)
    try:
        _label_shuffle_smoke(
            config,
            {**verified, "matrix": np.load(verified["paths"]["matrix"], mmap_mode="r")},
            feature_manifest,
            output,
            code_identity,
        )
        if workers == 1:
            for target in targets:
                completed.append(_run_target(target, *arguments))
                print(f"completed V6.1 target: {target}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run_target, target, *arguments): target for target in targets}
                for future in as_completed(futures):
                    completed.append(future.result())
                    print(f"completed V6.1 target: {completed[-1]}", flush=True)
    except BaseException as error:
        atomic_json(
            {
                **manifest_base,
                "status": "FAILED",
                "completed_targets": sorted(completed),
                "failure_type": type(error).__name__,
                "failure_summary": str(error)[:2000],
            },
            output / "run_manifest.json",
        )
        raise
    atomic_json(
        {**manifest_base, "status": "COMPLETE", "targets": sorted(completed)},
        output / "run_manifest.json",
    )
    print(f"V6.1 selector study complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
