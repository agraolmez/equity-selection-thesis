#!/usr/bin/env python3
"""Run the frozen, selector-only V6 walk-forward comparison.

The controller prepares one immutable encoded input cache, then runs targets in
parallel.  Every refit block is checkpointed and reused only under the same
contract, code, input, model, and forecast identity.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

STABLE_PYTHON = Path("/opt/anaconda3/envs/finance/bin/python")
if (
    Path(sys.executable).resolve() != STABLE_PYTHON.resolve()
    and "--native-runtime" not in sys.argv
):
    if not STABLE_PYTHON.is_file():
        raise FileNotFoundError(f"stable V6 runtime is missing: {STABLE_PYTHON}")
    os.execv(
        str(STABLE_PYTHON),
        [
            str(STABLE_PYTHON),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--native-runtime",
        ],
    )

from portfolio_bot.research_v6.data import (  # noqa: E402
    V6Authorities,
    build_target_panel,
    load_feature_registry,
    load_model_feature_panel,
    load_model_feature_registry,
    load_predictor_panel,
)
from portfolio_bot.research_v6.evaluation import (  # noqa: E402
    monthly_selector_metrics,
    summarize_selector_metrics,
)
from portfolio_bot.research_v6.models import dynamic_ensemble  # noqa: E402
from portfolio_bot.research_v6.preprocessing import encode_panel  # noqa: E402
from portfolio_bot.research_v6.selector_runtime import (  # noqa: E402
    atomic_json,
    atomic_parquet,
    canonical_sha256,
    evaluate_predictions,
    walk_forward_predictions,
)


DEFAULT_CONFIG = ROOT / "configs/research_v6/selector_step_01.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/research_v6/selector_step_01"
CODE_PATHS = (
    ROOT / "scripts/run_v6_selector_step_01.py",
    ROOT / "src/portfolio_bot/research_v6/data.py",
    ROOT / "src/portfolio_bot/research_v6/preprocessing.py",
    ROOT / "src/portfolio_bot/research_v6/models.py",
    ROOT / "src/portfolio_bot/research_v6/evaluation.py",
    ROOT / "src/portfolio_bot/research_v6/selector_runtime.py",
)


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


def _source_inventory(authorities: V6Authorities) -> list[dict[str, Any]]:
    paths = [
        authorities.validation_label_audit,
        authorities.validation_factors,
        authorities.validation_reserve_factors,
        authorities.validation_costs,
        authorities.final_labels,
        authorities.final_outcomes,
        authorities.final_accounting,
        authorities.final_costs,
        authorities.december_2019_targets,
        authorities.december_2019_costs,
    ]
    rows = []
    for path in paths:
        if path.is_file():
            stat = path.stat()
            rows.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return rows


def _frame_identity(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(columns).encode())
    digest.update(
        pd.util.hash_pandas_object(frame.loc[:, columns], index=False)
        .to_numpy(np.uint64)
        .tobytes()
    )
    return digest.hexdigest()


def _elastic_profiles(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "profile_id": f"enet_a{alpha:g}_l{ratio:g}",
            "alpha": float(alpha),
            "l1_ratio": float(ratio),
        }
        for alpha in config["elastic_net"]["alpha_grid"]
        for ratio in config["elastic_net"]["l1_ratio_grid"]
    ]


def _tuning_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["causal_tuning"]
    return {
        "holdout_months": int(source["holdout_months"]),
        "tuning_seed": int(source["tuning_seed"]),
        "profile_tie_tolerance": 0.002,
    }


def _prepare_cache(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output: Path,
    code_identity: str,
) -> tuple[Path, Path, Path, str]:
    cache = output / "input_cache"
    matrix_path = cache / "encoded_values.npy"
    rows_path = cache / "rows.parquet"
    metadata_path = cache / "feature_metadata.json"
    receipt_path = cache / "receipt.json"
    authorities = V6Authorities()
    source_inventory = _source_inventory(authorities)
    preliminary_identity = canonical_sha256(
        {
            "code_identity": code_identity,
            "feature_build_identity": config["inputs"]["model_feature_build_identity"],
            "source_inventory": source_inventory,
        }
    )
    if all(path.is_file() for path in (matrix_path, rows_path, metadata_path, receipt_path)):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("preliminary_identity") != preliminary_identity:
            raise ValueError(
                "existing V6 input cache has a different identity; use a new output lineage"
            )
        return matrix_path, rows_path, metadata_path, str(receipt["input_identity"])

    cache.mkdir(parents=True, exist_ok=True)
    primary_registry = load_model_feature_registry(feature_view="primary_full")
    model_panel = load_model_feature_panel(primary_registry)
    predecessor_registry = load_feature_registry(authorities)
    predecessor_panel = load_predictor_panel(predecessor_registry, authorities=authorities)
    targets = build_target_panel(predecessor_panel, authorities=authorities)
    target_columns = [
        "label_available_date",
        "unranked_excess_return",
        "alpha_capm",
        "alpha_ff5_momentum",
        "long_net_opportunity",
        "short_net_opportunity",
        "target_rank_capm",
        "target_rank_ff5_momentum",
        "target_rank_long_net",
        "target_rank_short_net",
    ]
    modeling = model_panel.merge(
        targets[
            ["holding_month", "signal_date", "permno", "permco", *target_columns]
        ],
        on=["holding_month", "signal_date", "permno", "permco"],
        how="left",
        validate="one_to_one",
    )
    encoded = encode_panel(modeling, primary_registry)
    rows = encoded.keys.merge(
        modeling[
            ["holding_month", "signal_date", "permno", "permco", *target_columns]
        ],
        on=["holding_month", "signal_date", "permno", "permco"],
        how="left",
        validate="one_to_one",
    )
    if len(rows) != int(config["inputs"]["expected_rows"]):
        raise ValueError(f"V6 row count changed: {len(rows)}")
    if encoded.values.shape[1] != int(config["inputs"]["expected_primary_encoded_features"]):
        raise ValueError(f"V6 encoded feature count changed: {encoded.values.shape[1]}")
    if rows["holding_month"].nunique() != int(config["inputs"]["expected_months"]):
        raise ValueError("V6 month count changed")
    if rows.duplicated(["holding_month", "permno"]).any():
        raise ValueError("V6 cache contains duplicate stock-months")
    if not np.isfinite(encoded.values).all():
        raise ValueError("V6 cache contains nonfinite encoded inputs")
    if not rows["holding_month"].eq("2019-12").any():
        raise ValueError("V6 cache omits December 2019")

    temporary_matrix = matrix_path.with_suffix(".npy.tmp")
    with temporary_matrix.open("wb") as destination:
        np.save(destination, encoded.values)
    temporary_matrix.replace(matrix_path)
    atomic_parquet(rows, rows_path)
    metadata = {
        "feature_names": list(encoded.feature_names),
        "source_features": list(encoded.source_features),
        "feature_family": dict(
            zip(
                primary_registry["feature_name"].astype(str),
                primary_registry["family"].astype(str),
            )
        ),
        "cam_free_raw_features": primary_registry.loc[
            primary_registry["cam_free"].fillna(False).astype(bool), "feature_name"
        ].astype(str).tolist(),
        "cam_only_raw_features": primary_registry.loc[
            primary_registry["cam_only"].fillna(False).astype(bool), "feature_name"
        ].astype(str).tolist(),
    }
    atomic_json(metadata, metadata_path)
    row_identity = _frame_identity(
        rows,
        [
            "holding_month",
            "signal_date",
            "permno",
            "permco",
            "label_available_date",
            *[item["rank_column"] for item in config["targets"].values()],
        ],
    )
    input_identity = canonical_sha256(
        {
            "preliminary_identity": preliminary_identity,
            "row_identity": row_identity,
            "encoded_feature_names": encoded.feature_names,
            "source_features": encoded.source_features,
            "shape": encoded.values.shape,
        }
    )
    atomic_json(
        {
            "schema": "research_v6_selector_input_cache_v1",
            "preliminary_identity": preliminary_identity,
            "input_identity": input_identity,
            "row_identity": row_identity,
            "rows": len(rows),
            "months": int(rows["holding_month"].nunique()),
            "raw_features": len(primary_registry),
            "encoded_features": encoded.values.shape[1],
            "december_2019_rows": int(rows["holding_month"].eq("2019-12").sum()),
            "source_inventory": source_inventory,
        },
        receipt_path,
    )
    return matrix_path, rows_path, metadata_path, input_identity


def _feature_index(metadata: Mapping[str, Any], view: str) -> np.ndarray:
    source = np.asarray(metadata["source_features"], dtype=str)
    if view == "primary_full":
        keep = np.ones(len(source), dtype=bool)
    elif view == "cam_free":
        keep = np.isin(source, metadata["cam_free_raw_features"])
    elif view == "cam_only":
        keep = np.isin(source, metadata["cam_only_raw_features"])
    elif view == "news_free":
        news = {
            feature
            for feature, family in metadata["feature_family"].items()
            if family == "alpaca_related_news"
        }
        keep = ~np.isin(source, sorted(news))
    else:
        raise ValueError(f"unknown feature view: {view}")
    index = np.flatnonzero(keep).astype(np.int64)
    if len(index) == 0:
        raise ValueError(f"feature view {view} is empty")
    return index


def _candidate_choice(summary: pd.DataFrame) -> pd.Series:
    finite = summary.loc[np.isfinite(summary["mean_rank_ic"])].copy()
    if finite.empty:
        raise RuntimeError("no finite validation candidate result")
    best = float(finite["mean_rank_ic"].max())
    admissible = finite.loc[finite["mean_rank_ic"].ge(best - 0.002)].copy()
    admissible["objective_preference"] = admissible["objective"].ne(
        "continuous_rank"
    ).astype(int)
    admissible["refit_preference"] = admissible["refit"].ne("annual").astype(int)
    admissible["retune_preference"] = admissible["retune"].ne("annual").astype(int)
    admissible["history_preference"] = admissible["history"].ne("expanding").astype(int)
    return admissible.sort_values(
        [
            "objective_preference",
            "refit_preference",
            "retune_preference",
            "history_preference",
            "task_id",
        ],
        kind="mergesort",
    ).iloc[0]


def _candidate_from_choice(config: Mapping[str, Any], choice: pd.Series) -> Mapping[str, Any]:
    return next(
        candidate
        for candidate in config["operating_candidates"]
        if candidate["candidate_id"] == choice["candidate_id"]
    )


def _run_one(
    *,
    matrix: np.ndarray,
    rows: pd.DataFrame,
    feature_index: np.ndarray,
    target_column: str,
    family: str,
    objective: str,
    candidate: Mapping[str, Any],
    profiles: list[dict[str, Any]],
    shared: Mapping[str, Any],
    start: str,
    end: str,
    seeds: list[int],
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
    matrix_path: str,
    rows_path: str,
    metadata_path: str,
    input_identity: str,
    code_identity: str,
    diagnostics: bool,
) -> str:
    config = json.loads(Path(config_path).read_text())
    output = Path(output_path) / target_id
    output.mkdir(parents=True, exist_ok=True)
    matrix = np.load(matrix_path, mmap_mode="r")
    rows = pd.read_parquet(rows_path)
    metadata = json.loads(Path(metadata_path).read_text())
    primary_index = _feature_index(metadata, "primary_full")
    target_spec = config["targets"][target_id]
    target_column = target_spec["rank_column"]
    economic_column = target_spec["economic_column"]
    chronology = config["chronology"]
    tuning = _tuning_policy(config)
    validation_seed = list(config["causal_tuning"]["validation_final_fit_seeds"])
    test_seeds = list(config["causal_tuning"]["historical_test_final_fit_seeds"])
    base_identity = canonical_sha256(
        {
            "input_identity": input_identity,
            "code_identity": code_identity,
            "target_id": target_id,
        }
    )
    validation_rows: list[dict[str, Any]] = []
    validation_predictions: dict[str, pd.DataFrame] = {}
    family_settings = {
        "xgboost": (config["xgboost"]["profiles"], config["xgboost"]["shared"]),
        "neural_network": (
            config["neural_network"]["profiles"],
            config["neural_network"]["shared"],
        ),
    }
    for family, (profiles, shared) in family_settings.items():
        for objective in config["objectives"]:
            for candidate in config["operating_candidates"]:
                task_id = f"{family}__{objective}__{candidate['candidate_id']}"
                task_output = output / "validation_candidates" / task_id
                prediction = _run_one(
                    matrix=matrix,
                    rows=rows,
                    feature_index=primary_index,
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
                    output=task_output,
                    run_identity=canonical_sha256({"base": base_identity, "task": task_id}),
                )
                metrics = evaluate_predictions(
                    prediction,
                    rows,
                    target_column=target_column,
                    economic_column=economic_column,
                    model_id=task_id,
                )
                summary = summarize_selector_metrics(metrics).iloc[0].to_dict()
                validation_rows.append(
                    {
                        **summary,
                        "task_id": task_id,
                        "family": family,
                        "objective": objective,
                        "candidate_id": candidate["candidate_id"],
                        "refit": candidate["refit"],
                        "retune": candidate["retune"],
                        "history": candidate["history"],
                    }
                )
                validation_predictions[task_id] = prediction
    validation_summary = pd.DataFrame(validation_rows)
    atomic_parquet(validation_summary, output / "validation_candidate_summary.parquet")
    choices = {
        family: _candidate_choice(validation_summary.loc[validation_summary["family"].eq(family)])
        for family in family_settings
    }
    selection = pd.DataFrame([choice.to_dict() for choice in choices.values()])
    atomic_parquet(selection, output / "validation_selected_candidates.parquet")

    selected_predictions: dict[str, pd.DataFrame] = {}
    selected_test_outputs: dict[str, Path] = {}
    for family, choice in choices.items():
        candidate = _candidate_from_choice(config, choice)
        profiles, shared = family_settings[family]
        objective = str(choice["objective"])
        task_id = f"selected_{family}__{objective}__{candidate['candidate_id']}"
        test_output = output / "historical_test" / task_id
        test = _run_one(
            matrix=matrix,
            rows=rows,
            feature_index=primary_index,
            target_column=target_column,
            family=family,
            objective=objective,
            candidate=candidate,
            profiles=profiles,
            shared=shared,
            start=chronology["historical_test_start"],
            end=chronology["historical_test_end"],
            seeds=test_seeds,
            tuning=tuning,
            output=test_output,
            run_identity=canonical_sha256({"base": base_identity, "task": task_id}),
        )
        selected_predictions[family] = pd.concat(
            [validation_predictions[str(choice["task_id"])], test], ignore_index=True
        ).sort_values(["holding_month", "permno"], kind="mergesort")
        selected_test_outputs[family] = test_output

    elastic_candidate = {
        "candidate_id": "elastic_quarterly_expanding_annual_retune",
        "refit": config["elastic_net"]["refit"],
        "history": config["elastic_net"]["history"],
        "retune": config["elastic_net"]["retune"],
    }
    elastic_profiles = _elastic_profiles(config)
    elastic_shared: dict[str, Any] = {}
    elastic_parts = []
    elastic_test_output: Path | None = None
    for period, start, end, seeds in (
        (
            "validation",
            chronology["validation_start"],
            chronology["validation_end"],
            validation_seed,
        ),
        (
            "historical_test",
            chronology["historical_test_start"],
            chronology["historical_test_end"],
            test_seeds,
        ),
    ):
        period_output = output / period / "elastic_net_benchmark"
        elastic_parts.append(
            _run_one(
                matrix=matrix,
                rows=rows,
                feature_index=primary_index,
                target_column=target_column,
                family="elastic_net",
                objective="continuous_rank",
                candidate=elastic_candidate,
                profiles=elastic_profiles,
                shared=elastic_shared,
                start=start,
                end=end,
                seeds=seeds,
                tuning=tuning,
                output=period_output,
                run_identity=canonical_sha256(
                    {"base": base_identity, "task": f"elastic_{period}"}
                ),
            )
        )
        if period == "historical_test":
            elastic_test_output = period_output
    elastic = pd.concat(elastic_parts, ignore_index=True)

    official = selected_predictions["xgboost"][
        ["holding_month", "signal_date", "permno", "permco", "score_rank"]
    ].rename(columns={"score_rank": "xgboost_rank"})
    official = official.merge(
        selected_predictions["neural_network"][
            ["holding_month", "permno", "score_rank"]
        ].rename(columns={"score_rank": "mlp_rank"}),
        on=["holding_month", "permno"],
        validate="one_to_one",
    ).merge(
        elastic[["holding_month", "permno", "score_rank"]].rename(
            columns={"score_rank": "elastic_net_rank"}
        ),
        on=["holding_month", "permno"],
        validate="one_to_one",
    ).merge(
        rows[["holding_month", "permno", target_column, economic_column]],
        on=["holding_month", "permno"],
        how="left",
        validate="one_to_one",
    )
    ensemble_config = config["dynamic_ensemble"]
    official = dynamic_ensemble(
        official,
        target_column=target_column,
        half_life_months=float(ensemble_config["half_life_months"]),
        minimum_history_months=int(ensemble_config["minimum_history_months"]),
        minimum_weight=float(ensemble_config["component_weight_bounds"][0]),
    )
    official["period"] = np.where(
        official["holding_month"].le(chronology["validation_end"]),
        "validation",
        "historical_test",
    )
    official_metrics = monthly_selector_metrics(
        official,
        score_columns=[
            "xgboost_rank",
            "mlp_rank",
            "ensemble_rank",
            "elastic_net_rank",
        ],
        target_column=target_column,
        economic_column=economic_column,
    ).merge(
        official[["holding_month", "period"]].drop_duplicates(),
        on="holding_month",
        validate="many_to_one",
    )
    summary = pd.concat(
        [
            summarize_selector_metrics(group).assign(period=period)
            for period, group in official_metrics.groupby("period", observed=True, sort=True)
        ],
        ignore_index=True,
    )
    annual = (
        official_metrics.assign(year=official_metrics["holding_month"].str[:4])
        .groupby(["model_id", "period", "year"], observed=True)["rank_ic"]
        .mean()
        .rename("mean_rank_ic")
        .reset_index()
    )
    test_summary = summary.loc[summary["period"].eq("historical_test")].copy()
    gate = config["evaluation"]["interesting_selector_gate"]
    test_summary["interesting_selector"] = (
        test_summary["mean_rank_ic"].ge(float(gate["historical_test_mean_rank_ic_minimum"]))
        & test_summary["rank_ic_hac_t"].ge(float(gate["historical_test_hac_t_minimum"]))
        & test_summary["early_mean_rank_ic"].gt(0.0)
        & test_summary["recent_mean_rank_ic"].gt(0.0)
        & test_summary["minimum_rolling_mean_rank_ic"].gt(0.0)
        & test_summary["mean_economic_top_minus_bottom"].gt(0.0)
    )
    atomic_parquet(official, output / "official_predictions.parquet")
    atomic_parquet(official_metrics, output / "official_monthly_metrics.parquet")
    atomic_parquet(summary, output / "official_summary.parquet")
    atomic_parquet(annual, output / "official_annual_metrics.parquet")
    atomic_parquet(test_summary, output / "historical_test_gate_evaluation.parquet")

    importance_rows: list[pd.DataFrame] = []
    importance_sources = [
        ("xgboost_rank", selected_test_outputs["xgboost"]),
        ("elastic_net_rank", elastic_test_output),
    ]
    feature_names = np.asarray(metadata["feature_names"], dtype=str)
    source_features = np.asarray(metadata["source_features"], dtype=str)
    for model_id, model_output in importance_sources:
        if model_output is None:
            continue
        path = model_output / "feature_importance.parquet"
        if not path.is_file():
            continue
        values = pd.read_parquet(path)
        if values.empty:
            continue
        aggregate = values.groupby("feature_index", observed=True)["importance"].mean().reset_index()
        indices = aggregate["feature_index"].to_numpy(int)
        aggregate["encoded_feature_name"] = feature_names[indices]
        aggregate["source_feature"] = source_features[indices]
        aggregate["family"] = aggregate["source_feature"].map(metadata["feature_family"])
        aggregate["model_id"] = model_id
        importance_rows.append(aggregate)
    if importance_rows:
        importance = pd.concat(importance_rows, ignore_index=True).sort_values(
            ["model_id", "importance"], ascending=[True, False], kind="mergesort"
        )
        atomic_parquet(importance, output / "official_feature_importance.parquet")

    diagnostic_rows: list[pd.DataFrame] = []
    if diagnostics:
        xgb_choice = choices["xgboost"]
        diagnostic_specs = (
            ("cam_free", "xgboost", xgb_choice),
            ("news_free", "xgboost", xgb_choice),
            ("cam_only", "elastic_net", None),
        )
        for view, family, choice in diagnostic_specs:
            index = _feature_index(metadata, view)
            if family == "xgboost":
                candidate = _candidate_from_choice(config, choice)
                profiles, shared = family_settings[family]
                objective = str(choice["objective"])
            else:
                candidate = elastic_candidate
                profiles, shared = elastic_profiles, elastic_shared
                objective = "continuous_rank"
            for period, start, end, seeds in (
                ("validation", chronology["validation_start"], chronology["validation_end"], validation_seed),
                ("historical_test", chronology["historical_test_start"], chronology["historical_test_end"], test_seeds),
            ):
                prediction = _run_one(
                    matrix=matrix,
                    rows=rows,
                    feature_index=index,
                    target_column=target_column,
                    family=family,
                    objective=objective,
                    candidate=candidate,
                    profiles=profiles,
                    shared=shared,
                    start=start,
                    end=end,
                    seeds=seeds,
                    tuning=tuning,
                    output=output / "diagnostics" / view / period,
                    run_identity=canonical_sha256(
                        {"base": base_identity, "view": view, "period": period}
                    ),
                )
                metrics = evaluate_predictions(
                    prediction,
                    rows,
                    target_column=target_column,
                    economic_column=economic_column,
                    model_id=f"{view}_{family}",
                )
                diagnostic_rows.append(
                    summarize_selector_metrics(metrics).assign(
                        feature_view=view, family=family, period=period
                    )
                )
        atomic_parquet(
            pd.concat(diagnostic_rows, ignore_index=True),
            output / "diagnostic_feature_view_summary.parquet",
        )

    atomic_json(
        {
            "schema": "research_v6_selector_target_completion_v1",
            "target_id": target_id,
            "target_column": target_column,
            "economic_column": economic_column,
            "base_identity": base_identity,
            "selected_xgboost": choices["xgboost"].to_dict(),
            "selected_neural_network": choices["neural_network"].to_dict(),
            "historical_test_months": int(
                official.loc[official["period"].eq("historical_test"), "holding_month"].nunique()
            ),
            "portfolio_constructed": False,
            "optimizer_run": False,
            "diagnostics_complete": diagnostics,
        },
        output / "completion.json",
    )
    return target_id


def _check_only(config: Mapping[str, Any]) -> None:
    manifest_path = Path(
        "/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/model_features_v2/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "BACKTEST_READY":
        raise ValueError("V6 feature manifest is not backtest-ready")
    if manifest.get("build_identity") != config["inputs"]["model_feature_build_identity"]:
        raise ValueError("V6 feature build identity changed")
    registry = load_model_feature_registry(feature_view="primary_full")
    if len(registry) != int(config["inputs"]["primary_raw_features"]):
        raise ValueError("V6 primary feature count changed")
    for month in pd.period_range(
        config["inputs"]["first_month"], config["inputs"]["last_month"], freq="M"
    ).astype(str):
        path = manifest_path.parent / "months" / f"{month}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
    print(
        "READY: frozen V6 contract, 318-feature panel, 128 months, four targets, "
        "pairwise and continuous selectors; no backtest started.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--targets", nargs="+", choices=("capm", "ff5_momentum", "long_net", "short_net"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--native-runtime", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.native_runtime and Path(sys.executable).resolve() != STABLE_PYTHON.resolve():
        if not STABLE_PYTHON.is_file():
            raise FileNotFoundError(f"stable V6 runtime is missing: {STABLE_PYTHON}")
        completed = subprocess.run(
            [str(STABLE_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:], "--native-runtime"],
            cwd=ROOT,
            check=False,
        )
        return int(completed.returncode)

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("status") != "FROZEN_READY_TO_RUN":
        raise ValueError("V6 selector contract is not frozen and ready")
    if args.check_only:
        _check_only(config)
        return 0
    code_identity = _code_identity(config_path)
    contract_short = _file_sha256(config_path)[:12]
    build_short = str(config["inputs"]["model_feature_build_identity"])[:12]
    output = args.output_root.resolve() / f"{contract_short}-{build_short}-{code_identity[:12]}"
    output.mkdir(parents=True, exist_ok=True)
    matrix_path, rows_path, metadata_path, input_identity = _prepare_cache(
        config=config,
        config_path=config_path,
        output=output,
        code_identity=code_identity,
    )
    if args.prepare_only:
        print(
            f"PREPARED: compatible V6 encoded input cache at {output}; no model fit started.",
            flush=True,
        )
        return 0
    targets = args.targets or list(config["targets"])
    workers = args.workers or int(config["execution"]["default_parallel_targets"])
    workers = max(1, min(workers, len(targets)))
    atomic_json(
        {
            "schema": "research_v6_selector_run_manifest_v1",
            "status": "RUNNING",
            "config": str(config_path),
            "config_sha256": _file_sha256(config_path),
            "code_identity": code_identity,
            "input_identity": input_identity,
            "targets": targets,
            "workers": workers,
            "diagnostics": not args.skip_diagnostics,
            "portfolio_constructed": False,
        },
        output / "run_manifest.json",
    )
    completed_targets: list[str] = []
    if workers == 1:
        for target in targets:
            completed_targets.append(
                _run_target(
                    target,
                    str(config_path),
                    str(output),
                    str(matrix_path),
                    str(rows_path),
                    str(metadata_path),
                    input_identity,
                    code_identity,
                    not args.skip_diagnostics,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_target,
                    target,
                    str(config_path),
                    str(output),
                    str(matrix_path),
                    str(rows_path),
                    str(metadata_path),
                    input_identity,
                    code_identity,
                    not args.skip_diagnostics,
                ): target
                for target in targets
            }
            for future in as_completed(futures):
                completed_targets.append(future.result())
                print(f"completed target: {completed_targets[-1]}", flush=True)
    atomic_json(
        {
            "schema": "research_v6_selector_run_manifest_v1",
            "status": "COMPLETE",
            "config": str(config_path),
            "config_sha256": _file_sha256(config_path),
            "code_identity": code_identity,
            "input_identity": input_identity,
            "targets": sorted(completed_targets),
            "diagnostics": not args.skip_diagnostics,
            "portfolio_constructed": False,
            "optimizer_run": False,
        },
        output / "run_manifest.json",
    )
    print(f"V6 selector run complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
