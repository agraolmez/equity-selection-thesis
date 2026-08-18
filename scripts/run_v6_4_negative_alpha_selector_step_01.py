#!/usr/bin/env python3
"""Run the separate V6.4 FF5 negative-alpha specialist selector study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
for path in (ROOT, SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_v6_1_selector_step_01 as base  # noqa: E402
from portfolio_bot.research_v6.evaluation import (  # noqa: E402
    monthly_selector_metrics,
    summarize_selector_metrics,
)
from portfolio_bot.research_v6.models import rank_within_month  # noqa: E402
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


STABLE_PYTHON = Path("/opt/anaconda3/envs/finance/bin/python")
DEFAULT_CONFIG = ROOT / "configs/research_v6_4/negative_alpha_selector_step_01.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/research_v6_4/negative_alpha_selector_step_01"
TARGET_ID = "negative_ff5_alpha"
CODE_PATHS = (
    ROOT / "scripts/run_v6_4_negative_alpha_selector_step_01.py",
    ROOT / "scripts/run_v6_1_selector_step_01.py",
    ROOT / "scripts/run_v6_1_xgboost_worker.py",
    ROOT / "src/portfolio_bot/research_v6_1/feature_views.py",
    ROOT / "src/portfolio_bot/research_v6_1/models.py",
    ROOT / "src/portfolio_bot/research_v6_1/runtime.py",
    ROOT / "src/portfolio_bot/research_v6/evaluation.py",
    ROOT / "src/portfolio_bot/research_v6/models.py",
    ROOT / "src/portfolio_bot/research_v6/selector_runtime.py",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    contract = json.loads(config_path.read_text())
    base_path = ROOT / str(contract["base_selector_contract"]["path"])
    observed = _file_sha256(base_path)
    expected = str(contract["base_selector_contract"]["sha256"])
    if observed != expected:
        raise ValueError(f"frozen V6.1 selector contract changed: {observed} != {expected}")
    effective = copy.deepcopy(json.loads(base_path.read_text()))
    target = contract["target"]
    effective["schema"] = str(contract["schema"])
    effective["status"] = str(contract["status"])
    effective["scope"] = copy.deepcopy(contract["scope"])
    effective["targets"] = {
        TARGET_ID: {
            "rank_column": str(target["rank_column"]),
            "economic_column": str(target["economic_column"]),
        }
    }
    for family in ("xgboost", "neural_network"):
        effective["operating_policy"][family] = {
            TARGET_ID: copy.deepcopy(
                json.loads(base_path.read_text())["operating_policy"][family]["ff5_momentum"]
            )
        }
    effective["execution"]["default_parallel_targets"] = 1
    effective["execution"]["full_run_allowed_by_current_status"] = bool(
        contract["execution"]["full_run_allowed_by_current_status"]
    )
    return contract, effective, base_path


def _code_identity(config_path: Path, base_path: Path) -> str:
    return canonical_sha256(
        {
            "contract_sha256": _file_sha256(config_path),
            "base_contract_sha256": _file_sha256(base_path),
            "code": {
                str(path.relative_to(ROOT)): _file_sha256(path) for path in CODE_PATHS
            },
        }
    )


def attach_negative_alpha_target(
    rows: pd.DataFrame, contract: Mapping[str, Any]
) -> pd.DataFrame:
    """Create a high-is-bad-alpha target without changing any source outcomes."""

    target = contract["target"]
    source_rank = pd.to_numeric(rows[str(target["source_rank_column"])], errors="coerce")
    source_economic = pd.to_numeric(
        rows[str(target["source_economic_column"])], errors="coerce"
    )
    result = rows.copy()
    result[str(target["rank_column"])] = -source_rank
    result[str(target["economic_column"])] = -source_economic
    valid = source_rank.notna()
    if not np.allclose(
        result.loc[valid, str(target["rank_column"])].to_numpy(float),
        -source_rank.loc[valid].to_numpy(float),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("negative-alpha rank is not the exact FF5-rank reversal")
    return result


def reverse_monthly_score(values: Sequence[float], months: Sequence[str]) -> np.ndarray:
    return rank_within_month(-np.asarray(values, dtype=float), months)


def _load_frozen_control(
    contract: Mapping[str, Any], *, start: str, end: str
) -> pd.DataFrame:
    policy = contract["frozen_control"]
    path = ROOT / str(policy["path"])
    if _file_sha256(path) != str(policy["sha256"]):
        raise ValueError("frozen V6.1 FF5-alpha control prediction identity changed")
    score_column = str(policy["score_column"])
    output_column = str(policy["negative_alpha_control_column"])
    frame = pd.read_parquet(path, columns=["holding_month", "permno", score_column])
    frame["holding_month"] = frame["holding_month"].astype(str)
    frame = frame.loc[frame["holding_month"].between(start, end)].copy()
    frame[output_column] = reverse_monthly_score(
        frame[score_column].to_numpy(float), frame["holding_month"].to_numpy(str)
    )
    if frame.duplicated(["holding_month", "permno"]).any():
        raise ValueError("frozen FF5-alpha control contains duplicate stock-months")
    return frame[["holding_month", "permno", output_column]]


def _tail_diagnostics(
    frame: pd.DataFrame,
    *,
    score_columns: Sequence[str],
    alpha_column: str,
    periods: Mapping[str, tuple[str, str]],
    tail_fractions: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for period, (start, end) in periods.items():
        subset = frame.loc[frame["holding_month"].between(start, end)]
        for month, group in subset.groupby("holding_month", observed=True, sort=True):
            realized_alpha = pd.to_numeric(group[alpha_column], errors="coerce").to_numpy(float)
            for model_id in score_columns:
                score = pd.to_numeric(group[model_id], errors="coerce").to_numpy(float)
                valid = np.isfinite(score) & np.isfinite(realized_alpha)
                if valid.sum() < 20:
                    continue
                valid_positions = np.flatnonzero(valid)
                order = valid_positions[np.argsort(score[valid], kind="mergesort")]
                for fraction in tail_fractions:
                    count = max(int(np.ceil(len(order) * float(fraction))), 1)
                    selected = order[-count:]
                    alpha = realized_alpha[selected]
                    rows.append(
                        {
                            "period": period,
                            "holding_month": str(month),
                            "model_id": model_id,
                            "tail_fraction": float(fraction),
                            "selected_stocks": int(count),
                            "negative_alpha_fraction": float(np.mean(alpha < 0.0)),
                            "mean_realized_ff5_alpha": float(np.mean(alpha)),
                            "median_realized_ff5_alpha": float(np.median(alpha)),
                        }
                    )
    monthly = pd.DataFrame(rows)
    summary = (
        monthly.groupby(["period", "model_id", "tail_fraction"], observed=True)
        .agg(
            months=("holding_month", "nunique"),
            mean_selected_stocks=("selected_stocks", "mean"),
            mean_negative_alpha_fraction=("negative_alpha_fraction", "mean"),
            mean_realized_ff5_alpha=("mean_realized_ff5_alpha", "mean"),
            median_month_realized_ff5_alpha=("median_realized_ff5_alpha", "median"),
        )
        .reset_index()
    )
    return monthly, summary


def _overlap_diagnostics(
    frame: pd.DataFrame,
    *,
    challenger: str,
    control: str,
    tail_fractions: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (period, month), group in frame.groupby(
        ["period", "holding_month"], observed=True, sort=True
    ):
        for fraction in tail_fractions:
            count = max(int(np.ceil(len(group) * float(fraction))), 1)
            challenger_set = set(group.nlargest(count, challenger)["permno"])
            control_set = set(group.nlargest(count, control)["permno"])
            intersection = len(challenger_set & control_set)
            union = len(challenger_set | control_set)
            rows.append(
                {
                    "period": str(period),
                    "holding_month": str(month),
                    "tail_fraction": float(fraction),
                    "tail_size": int(count),
                    "intersection": int(intersection),
                    "conditional_overlap": intersection / count,
                    "jaccard": intersection / max(union, 1),
                }
            )
    return pd.DataFrame(rows)


def _run_target(
    *,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    verified: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    output: Path,
    code_identity: str,
) -> None:
    matrix = verified["matrix"]
    rows = attach_negative_alpha_target(verified["rows"], contract)
    metadata = verified["metadata"]
    target_spec = config["targets"][TARGET_ID]
    target_column = str(target_spec["rank_column"])
    economic_column = str(target_spec["economic_column"])
    source_alpha_column = str(contract["target"]["source_economic_column"])
    chronology = config["chronology"]
    tuning = base._tuning(config)
    validation_seeds = list(config["causal_tuning"]["validation_final_fit_seeds"])
    test_seeds = list(config["causal_tuning"]["historical_test_final_fit_seeds"])
    target_output = output / TARGET_ID
    target_output.mkdir(parents=True, exist_ok=True)
    base_identity = canonical_sha256(
        {
            "code_identity": code_identity,
            "input_identity": config["inputs"]["v6_input_identity"],
            "feature_manifest": feature_manifest["identity"],
            "target": contract["target"],
            "frozen_control_sha256": contract["frozen_control"]["sha256"],
        }
    )

    validation_choices: list[dict[str, Any]] = []
    component_predictions: dict[str, pd.DataFrame] = {}
    seed_frames: list[pd.DataFrame] = []
    for view in config["feature_views"]["views"]:
        feature_index = np.asarray(
            feature_manifest["views"][view]["encoded_indices"], dtype=np.int64
        )
        for family in ("xgboost", "neural_network", "elastic_net"):
            profiles, shared = base._model_settings(config, family)
            candidate = (
                config["operating_policy"][family][TARGET_ID]
                if family != "elastic_net"
                else config["operating_policy"]["elastic_net"]
            )
            objectives = (
                ["continuous_rank"]
                if family == "elastic_net"
                else list(config["objectives"])
            )
            candidates: list[dict[str, Any]] = []
            validation_predictions: dict[str, pd.DataFrame] = {}
            for objective in objectives:
                task = f"{view}__{family}__{objective}"
                prediction = base._run_model(
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
                    seeds=validation_seeds,
                    tuning=tuning,
                    output=target_output / "validation_candidates" / task,
                    run_identity=canonical_sha256({"base": base_identity, "task": task}),
                )
                _, summary = base._summary(
                    prediction,
                    rows,
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
            choice = base._choose_objective(pd.DataFrame(candidates), config)
            validation_choices.extend(candidates)
            selected_objective = str(choice["objective"])
            task = f"{view}__{family}__{selected_objective}"
            test = base._run_model(
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
                output=target_output / "historical_test" / task,
                run_identity=canonical_sha256(
                    {"base": base_identity, "task": task, "period": "historical_test"}
                ),
            )
            component = f"{view}__{'mlp' if family == 'neural_network' else family}"
            component_predictions[component] = pd.concat(
                [validation_predictions[selected_objective], test], ignore_index=True
            ).sort_values(["holding_month", "permno"], kind="mergesort")
            stability = base._seed_stability(test, component=component)
            if not stability.empty:
                seed_frames.append(stability)

    validation_summary = pd.DataFrame(validation_choices)
    validation_summary["selected_objective"] = False
    for (_, _), group in validation_summary.groupby(
        ["feature_view", "family"], observed=True
    ):
        choice = base._choose_objective(group, config)
        validation_summary.loc[choice.name, "selected_objective"] = True
    atomic_parquet(
        validation_summary, target_output / "validation_objective_summary.parquet"
    )

    selected_rows = rows.loc[
        rows["holding_month"].between(
            chronology["validation_start"], chronology["historical_test_end"]
        ),
        [
            "holding_month",
            "signal_date",
            "permno",
            "permco",
            source_alpha_column,
            target_column,
            economic_column,
        ],
    ].copy()
    official = selected_rows
    stratification = config["evaluation"]["size_liquidity_stratification"]
    encoded_lookup = {
        str(name): index for index, name in enumerate(metadata["feature_names"])
    }
    for output_column, source in (
        ("size_input", str(stratification["size_source"])),
        ("liquidity_input", str(stratification["liquidity_source"])),
    ):
        official[output_column] = np.asarray(
            matrix[official.index.to_numpy(np.int64), encoded_lookup[source]], dtype=float
        )
    for component, prediction in component_predictions.items():
        official = official.merge(
            prediction[["holding_month", "permno", "score_rank"]].rename(
                columns={"score_rank": component}
            ),
            on=["holding_month", "permno"],
            validate="one_to_one",
        )

    cam = direct_cam_predictions(
        matrix=matrix,
        rows=rows,
        metadata=metadata,
        source=str(config["feature_views"]["direct_cam_level"]),
        start_month=chronology["validation_start"],
        end_month=chronology["historical_test_end"],
    )
    cam["cam_direct"] = reverse_monthly_score(
        cam["score_rank"].to_numpy(float), cam["holding_month"].to_numpy(str)
    )
    official = official.merge(
        cam[["holding_month", "permno", "cam_direct", "cam_available"]],
        on=["holding_month", "permno"],
        validate="one_to_one",
    )
    control_column = str(contract["frozen_control"]["negative_alpha_control_column"])
    control = _load_frozen_control(
        contract,
        start=chronology["validation_start"],
        end=chronology["historical_test_end"],
    )
    official = official.merge(
        control, on=["holding_month", "permno"], validate="one_to_one"
    )

    score_columns = list(component_predictions) + ["cam_direct", control_column]
    dynamic = config["ensembles"]["dynamic"]
    for name, components in config["ensembles"]["sets"].items():
        equal_column = f"ensemble__{name}__equal"
        dynamic_column = f"ensemble__{name}__dynamic"
        official = add_equal_ensemble(
            official, components=components, output_column=equal_column
        )
        official = add_dynamic_ensemble(
            official,
            components=components,
            target_column=target_column,
            output_column=dynamic_column,
            half_life_months=float(dynamic["half_life_months"]),
            minimum_history_months=int(dynamic["minimum_history_months"]),
            minimum_component_weight=float(dynamic["minimum_component_weight"]),
            maximum_component_weight=float(dynamic["maximum_component_weight"]),
        )
        score_columns.extend([equal_column, dynamic_column])
    official["period"] = np.where(
        official["holding_month"].le(chronology["validation_end"]),
        "validation",
        "historical_test",
    )

    monthly = monthly_selector_metrics(
        official,
        score_columns=score_columns,
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
            for period, group in monthly.groupby("period", observed=True, sort=True)
        ],
        ignore_index=True,
    )
    validation = summary.loc[summary["period"].eq("validation")].copy()
    challengers = validation.loc[validation["model_id"].ne(control_column)].sort_values(
        ["mean_rank_ic", "model_id"], ascending=[False, True], kind="mergesort"
    )
    challenger = str(challengers.iloc[0]["model_id"])
    control_validation = validation.loc[validation["model_id"].eq(control_column)].iloc[0]
    challenger_validation = challengers.iloc[0]
    selection = {
        "schema": "research_v6_4_negative_alpha_validation_selection_v1",
        "selection_period": "validation",
        "selection_metric": "mean_rank_ic",
        "challenger_model_id": challenger,
        "challenger_validation_mean_rank_ic": float(challenger_validation["mean_rank_ic"]),
        "control_model_id": control_column,
        "control_validation_mean_rank_ic": float(control_validation["mean_rank_ic"]),
        "challenger_minus_control_validation_mean_rank_ic": float(
            challenger_validation["mean_rank_ic"] - control_validation["mean_rank_ic"]
        ),
        "historical_test_used_for_selection": False,
    }

    pivot = monthly.pivot(
        index=["holding_month", "period"], columns="model_id", values="rank_ic"
    ).reset_index()
    paired = pivot[["holding_month", "period"]].copy()
    for model_id in score_columns:
        if model_id == control_column:
            continue
        paired[model_id] = pivot[model_id] - pivot[control_column]

    turnover = base._turnover(official, score_columns)
    subgroup_monthly, subgroup_summary = base._subgroup_metrics(
        official.loc[official["period"].eq("historical_test")],
        score_columns=score_columns,
        target_column=target_column,
        economic_column=economic_column,
    )
    tail_monthly, tail_summary = _tail_diagnostics(
        official,
        score_columns=score_columns,
        alpha_column=source_alpha_column,
        periods={
            "validation": (
                chronology["validation_start"],
                chronology["validation_end"],
            ),
            "historical_test": (
                chronology["historical_test_start"],
                chronology["historical_test_end"],
            ),
        },
        tail_fractions=contract["selection"]["tail_diagnostics"],
    )
    overlap = _overlap_diagnostics(
        official,
        challenger=challenger,
        control=control_column,
        tail_fractions=contract["selection"]["tail_diagnostics"],
    )

    gate = config["evaluation"]["interesting_selector_gate"]
    test_summary = summary.loc[summary["period"].eq("historical_test")].copy()
    test_summary["interesting_selector"] = (
        test_summary["mean_rank_ic"].ge(
            float(gate["historical_test_mean_rank_ic_minimum"])
        )
        & test_summary["rank_ic_hac_t"].ge(
            float(gate["historical_test_hac_t_minimum"])
        )
        & test_summary["early_mean_rank_ic"].gt(0.0)
        & test_summary["recent_mean_rank_ic"].gt(0.0)
        & test_summary["minimum_rolling_mean_rank_ic"].gt(0.0)
        & test_summary["mean_economic_top_minus_bottom"].gt(0.0)
    )

    atomic_parquet(official, target_output / "official_predictions.parquet")
    atomic_parquet(monthly, target_output / "official_monthly_metrics.parquet")
    atomic_parquet(summary, target_output / "official_summary.parquet")
    atomic_parquet(test_summary, target_output / "historical_test_gate_evaluation.parquet")
    atomic_parquet(paired, target_output / "paired_control_rank_ic_deltas.parquet")
    atomic_parquet(turnover, target_output / "rank_turnover.parquet")
    atomic_parquet(subgroup_monthly, target_output / "size_liquidity_monthly_metrics.parquet")
    atomic_parquet(subgroup_summary, target_output / "size_liquidity_summary.parquet")
    atomic_parquet(tail_monthly, target_output / "negative_alpha_tail_monthly.parquet")
    atomic_parquet(tail_summary, target_output / "negative_alpha_tail_summary.parquet")
    atomic_parquet(overlap, target_output / "challenger_control_overlap.parquet")
    atomic_parquet(
        pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame(),
        target_output / "seed_stability.parquet",
    )
    atomic_json(selection, target_output / "validation_selection.json")
    atomic_json(
        {
            "schema": "research_v6_4_negative_alpha_target_completion_v1",
            "target": TARGET_ID,
            "base_identity": base_identity,
            "feature_view_identity": feature_manifest["identity"],
            "score_columns": score_columns,
            "validation_selected_challenger": challenger,
            "portfolio_constructed": False,
            "optimizer_run": False,
        },
        target_output / "completion.json",
    )


def _label_shuffle_smoke(
    *,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    verified: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    output: Path,
    code_identity: str,
) -> None:
    policy = config["evaluation"]["label_shuffle_smoke"]
    if not policy["enabled"]:
        return
    rows = attach_negative_alpha_target(verified["rows"], contract)
    target_column = str(contract["target"]["rank_column"])
    economic_column = str(contract["target"]["economic_column"])
    rng = np.random.default_rng(int(policy["seed"]))
    rows[target_column] = rows.groupby("holding_month", observed=True)[
        target_column
    ].transform(lambda values: rng.permutation(values.to_numpy()))
    profiles, shared = base._model_settings(config, "elastic_net")
    feature_index = np.asarray(
        feature_manifest["views"][policy["feature_view"]]["encoded_indices"],
        dtype=np.int64,
    )
    prediction = base._run_model(
        matrix=verified["matrix"],
        rows=rows,
        feature_index=feature_index,
        target_column=target_column,
        family="elastic_net",
        objective="continuous_rank",
        candidate=config["operating_policy"]["elastic_net"],
        profiles=profiles[:3],
        shared=shared,
        start="2023-01",
        end="2023-12",
        seeds=[int(policy["seed"])],
        tuning=base._tuning(config),
        output=output / "controls/label_shuffle",
        run_identity=canonical_sha256({"code": code_identity, "control": policy}),
    )
    monthly, summary = base._summary(
        prediction,
        rows,
        score_column="label_shuffle_elastic_net",
        target_column=target_column,
        economic_column=economic_column,
    )
    mean_ic = float(summary.iloc[0]["mean_rank_ic"])
    summary["passes_smoke_bound"] = abs(mean_ic) <= float(
        policy["maximum_absolute_mean_ic"]
    )
    atomic_parquet(monthly, output / "controls/label_shuffle/monthly_metrics.parquet")
    atomic_parquet(summary, output / "controls/label_shuffle/summary.parquet")
    if not bool(summary.iloc[0]["passes_smoke_bound"]):
        raise RuntimeError("V6.4 negative-alpha label-shuffle control failed")


def _smoke_test() -> None:
    base._smoke_test()
    sample = pd.DataFrame(
        {
            "holding_month": ["2020-01"] * 5,
            "target_rank_ff5_momentum": [-0.4, -0.2, 0.0, 0.2, 0.4],
            "alpha_ff5_momentum": [-0.08, -0.02, 0.0, 0.03, 0.09],
        }
    )
    contract = {
        "target": {
            "source_rank_column": "target_rank_ff5_momentum",
            "source_economic_column": "alpha_ff5_momentum",
            "rank_column": "target_rank_negative_ff5_alpha",
            "economic_column": "negative_ff5_alpha_opportunity",
        }
    }
    result = attach_negative_alpha_target(sample, contract)
    np.testing.assert_allclose(
        result["target_rank_negative_ff5_alpha"],
        -sample["target_rank_ff5_momentum"],
    )
    np.testing.assert_allclose(
        result["negative_ff5_alpha_opportunity"],
        -sample["alpha_ff5_momentum"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--prepare-views", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--native-runtime", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.native_runtime and Path(sys.executable).resolve() != STABLE_PYTHON.resolve():
        if not STABLE_PYTHON.is_file():
            raise FileNotFoundError(f"stable V6.4 runtime is missing: {STABLE_PYTHON}")
        os.execv(
            str(STABLE_PYTHON),
            [str(STABLE_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:], "--native-runtime"],
        )

    config_path = args.config.resolve()
    contract, config, base_path = _load_contract(config_path)
    if args.smoke_test:
        _smoke_test()
        print("PASS: V6.4 target, XGBoost, MLP, Elastic Net, and ensemble smoke tests.", flush=True)
        return 0
    verified = base._verify_inputs(config, load_matrix=args.prepare_views)
    if args.check_only:
        print(
            "READY: separate FF5 negative-alpha specialist; three feature views, "
            "XGBoost, MLP, Elastic Net, equal/dynamic ensembles, frozen reversed-alpha control, "
            "and no portfolio or optimizer.",
            flush=True,
        )
        return 0

    code_identity = _code_identity(config_path, base_path)
    output = args.output_root.resolve() / (
        f"{_file_sha256(config_path)[:12]}-"
        f"{config['inputs']['v6_input_identity'][:12]}-{code_identity[:12]}"
    )
    output.mkdir(parents=True, exist_ok=True)
    matrix = (
        verified["matrix"]
        if "matrix" in verified
        else np.load(verified["paths"]["matrix"], mmap_mode="r")
    )
    feature_manifest = prepare_feature_views(
        matrix=matrix,
        rows=verified["rows"],
        metadata=verified["metadata"],
        config=config,
        input_identity=str(config["inputs"]["v6_input_identity"]),
        output=output / "feature_views",
    )
    if args.prepare_views:
        print(
            "PREPARED: "
            + ", ".join(
                f"{name}={spec['raw_source_count']} raw/{spec['encoded_feature_count']} encoded"
                for name, spec in feature_manifest["views"].items()
            )
            + "; no model fit started.",
            flush=True,
        )
        return 0
    if contract.get("status") != "FROZEN_READY_TO_RUN" or not contract["execution"][
        "full_run_allowed_by_current_status"
    ]:
        raise PermissionError("V6.4 full fit is not authorized by the frozen contract")

    manifest = {
        "schema": "research_v6_4_negative_alpha_selector_run_manifest_v1",
        "config": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "base_config_sha256": _file_sha256(base_path),
        "code_identity": code_identity,
        "v6_input_identity": config["inputs"]["v6_input_identity"],
        "feature_view_identity": feature_manifest["identity"],
        "targets": [TARGET_ID],
        "portfolio_constructed": False,
        "optimizer_run": False,
    }
    atomic_json({**manifest, "status": "RUNNING"}, output / "run_manifest.json")
    verified_with_matrix = {**verified, "matrix": matrix}
    try:
        _label_shuffle_smoke(
            contract=contract,
            config=config,
            verified=verified_with_matrix,
            feature_manifest=feature_manifest,
            output=output,
            code_identity=code_identity,
        )
        _run_target(
            contract=contract,
            config=config,
            verified=verified_with_matrix,
            feature_manifest=feature_manifest,
            output=output,
            code_identity=code_identity,
        )
    except BaseException as error:
        atomic_json(
            {
                **manifest,
                "status": "FAILED",
                "failure_type": type(error).__name__,
                "failure_summary": str(error)[:2000],
            },
            output / "run_manifest.json",
        )
        raise
    atomic_json({**manifest, "status": "COMPLETE"}, output / "run_manifest.json")
    print(f"V6.4 negative-alpha selector study complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
