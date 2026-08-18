#!/usr/bin/env python3
"""Refit only the frozen selector components used by the chosen V6.1 portfolio.

This is a computational-reproducibility run.  It reuses the authenticated V6
feature matrix and the already-frozen model/objective choices, but it does not
reuse any fitted selector, selector prediction, or selector checkpoint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

STABLE_PYTHON = Path("/opt/anaconda3/envs/finance/bin/python")
DEFAULT_CONFIG = ROOT / "configs/research_v6_1/selector_step_01.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/research_v6_1/primary_selector_clean_reproduction"
ORIGINAL_RUN = ROOT / "artifacts/research_v6_1/selector_step_01/481e306553c2-16046ee58521-d5dadd9cd9c3"

from scripts.run_v6_1_selector_step_01 import (  # noqa: E402
    _code_identity,
    _file_sha256,
    _model_settings,
    _run_model,
    _tuning,
    _verify_inputs,
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
from portfolio_bot.research_v6_1.models import add_equal_ensemble  # noqa: E402


COMPONENTS: tuple[dict[str, str], ...] = (
    {"target": "capm", "view": "cam_free", "family": "neural_network", "objective": "continuous_rank"},
    {"target": "ff5_momentum", "view": "cam_free", "family": "neural_network", "objective": "pairwise"},
    {"target": "long_net", "view": "cam_free", "family": "neural_network", "objective": "continuous_rank"},
    {"target": "short_net", "view": "correlation_reduced", "family": "xgboost", "objective": "pairwise"},
    {"target": "short_net", "view": "correlation_reduced", "family": "neural_network", "objective": "pairwise"},
)


def _component_name(family: str, view: str) -> str:
    return f"{view}__{'mlp' if family == 'neural_network' else family}"


def _fit_component(
    spec: dict[str, str],
    config_path: str,
    output_path: str,
    feature_manifest_path: str,
    code_identity: str,
) -> dict[str, str]:
    config = json.loads(Path(config_path).read_text())
    verified = _verify_inputs(config, load_matrix=True)
    matrix = verified["matrix"]
    rows = verified["rows"]
    feature_manifest = json.loads(Path(feature_manifest_path).read_text())
    target = spec["target"]
    view = spec["view"]
    family = spec["family"]
    objective = spec["objective"]
    component = _component_name(family, view)
    target_column = str(config["targets"][target]["rank_column"])
    profiles, shared = _model_settings(config, family)
    candidate = config["operating_policy"][family][target]
    tuning = _tuning(config)
    chronology = config["chronology"]
    feature_index = np.asarray(feature_manifest["views"][view]["encoded_indices"], dtype=np.int64)
    base_identity = canonical_sha256(
        {
            "code_identity": code_identity,
            "input_identity": config["inputs"]["v6_input_identity"],
            "feature_manifest": feature_manifest["identity"],
            "target": target,
        }
    )
    task = f"{view}__{family}__{objective}"
    root = Path(output_path) / "components" / target / task
    validation = _run_model(
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
        seeds=list(config["causal_tuning"]["validation_final_fit_seeds"]),
        tuning=tuning,
        output=root / "validation",
        run_identity=canonical_sha256({"base": base_identity, "task": task}),
    )
    historical = _run_model(
        matrix=matrix,
        rows=rows,
        feature_index=feature_index,
        target_column=target_column,
        family=family,
        objective=objective,
        candidate=candidate,
        profiles=profiles,
        shared=shared,
        start=chronology["historical_test_start"],
        end=chronology["historical_test_end"],
        seeds=list(config["causal_tuning"]["historical_test_final_fit_seeds"]),
        tuning=tuning,
        output=root / "historical_test",
        run_identity=canonical_sha256(
            {"base": base_identity, "task": task, "period": "historical_test"}
        ),
    )
    combined = pd.concat([validation, historical], ignore_index=True).sort_values(
        ["holding_month", "permno"], kind="mergesort"
    )
    prediction_path = root / "component_predictions.parquet"
    atomic_parquet(combined, prediction_path)
    return {
        "target": target,
        "view": view,
        "family": family,
        "objective": objective,
        "component": component,
        "path": str(prediction_path),
    }


def _comparison(new: pd.DataFrame, original: pd.DataFrame, column: str) -> dict[str, Any]:
    keys = ["holding_month", "signal_date", "permno", "permco"]
    left = new[keys + [column]].sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = original[keys + [column]].sort_values(keys, kind="mergesort").reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise RuntimeError(f"clean reproduction row keys differ for {column}")
    delta = left[column].to_numpy(float) - right[column].to_numpy(float)
    monthly_corr = []
    joined = left.assign(original=right[column].to_numpy(float))
    for _, group in joined.groupby("holding_month", observed=True, sort=True):
        monthly_corr.append(float(group[column].corr(group["original"], method="spearman")))
    return {
        "rows": int(len(left)),
        "exact_value_matches": int(np.count_nonzero(delta == 0.0)),
        "maximum_absolute_score_difference": float(np.max(np.abs(delta))),
        "mean_absolute_score_difference": float(np.mean(np.abs(delta))),
        "mean_monthly_rank_correlation": float(np.mean(monthly_corr)),
        "minimum_monthly_rank_correlation": float(np.min(monthly_corr)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--native-runtime", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.native_runtime and Path(sys.executable).resolve() != STABLE_PYTHON.resolve():
        os.execv(
            str(STABLE_PYTHON),
            [str(STABLE_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:], "--native-runtime"],
        )

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    verified = _verify_inputs(config, load_matrix=True)
    code_identity = _code_identity(config_path)
    run_id = (
        f"{_file_sha256(config_path)[:12]}-"
        f"{config['inputs']['v6_input_identity'][:12]}-{code_identity[:12]}"
    )
    output = args.output_root.resolve() / run_id
    output.mkdir(parents=True, exist_ok=True)
    feature_manifest = prepare_feature_views(
        matrix=verified["matrix"],
        rows=verified["rows"],
        metadata=verified["metadata"],
        config=config,
        input_identity=str(config["inputs"]["v6_input_identity"]),
        output=output / "feature_views",
    )
    manifest_base = {
        "schema": "research_v6_1_primary_selector_clean_reproduction_v1",
        "purpose": "Computational reproducibility of only the frozen production selector components",
        "config": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "code_identity": code_identity,
        "v6_input_identity": config["inputs"]["v6_input_identity"],
        "feature_view_identity": feature_manifest["identity"],
        "fitted_models_reused": False,
        "selector_predictions_reused": False,
        "selector_checkpoints_reused": False,
        "components": list(COMPONENTS),
    }
    atomic_json({**manifest_base, "status": "RUNNING"}, output / "run_manifest.json")

    results: list[dict[str, str]] = []
    workers = max(1, min(int(args.workers), len(COMPONENTS)))
    arguments = (str(config_path), str(output), str(output / "feature_views/feature_views.json"), code_identity)
    if workers == 1:
        for spec in COMPONENTS:
            result = _fit_component(dict(spec), *arguments)
            results.append(result)
            print(f"DONE {result['target']} {result['component']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fit_component, dict(spec), *arguments): spec for spec in COMPONENTS}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"DONE {result['target']} {result['component']}", flush=True)

    rows = verified["rows"]
    chronology = config["chronology"]
    selected = rows.loc[
        rows["holding_month"].between(chronology["validation_start"], chronology["historical_test_end"])
    ].copy()
    cam = direct_cam_predictions(
        matrix=verified["matrix"],
        rows=rows,
        metadata=verified["metadata"],
        source=str(config["feature_views"]["direct_cam_level"]),
        start_month=chronology["validation_start"],
        end_month=chronology["historical_test_end"],
    )[["holding_month", "permno", "score_rank"]].rename(columns={"score_rank": "cam_direct"})

    by_target: dict[str, list[dict[str, str]]] = {}
    for result in results:
        by_target.setdefault(result["target"], []).append(result)
    comparisons: dict[str, Any] = {}
    signal_column_by_target = {
        "capm": "ensemble__cam_free_mlp_plus_cam__equal",
        "ff5_momentum": "ensemble__cam_free_mlp_plus_cam__equal",
        "long_net": "ensemble__cam_free_mlp_plus_cam__equal",
        "short_net": "ensemble__correlation_reduced_xgb_mlp__equal",
    }
    for target in ("capm", "ff5_momentum", "long_net", "short_net"):
        target_spec = config["targets"][target]
        target_rank = str(target_spec["rank_column"])
        economic = str(target_spec["economic_column"])
        frame = selected[["holding_month", "signal_date", "permno", "permco", target_rank, economic]].copy()
        components: list[str] = []
        for result in sorted(by_target[target], key=lambda item: item["component"]):
            prediction = pd.read_parquet(result["path"], columns=["holding_month", "permno", "score_rank"])
            frame = frame.merge(
                prediction.rename(columns={"score_rank": result["component"]}),
                on=["holding_month", "permno"],
                validate="one_to_one",
            )
            components.append(result["component"])
        if target != "short_net":
            frame = frame.merge(cam, on=["holding_month", "permno"], validate="one_to_one")
            components.append("cam_direct")
        signal_column = signal_column_by_target[target]
        frame = add_equal_ensemble(frame, components=components, output_column=signal_column)
        frame["period"] = np.where(
            frame["holding_month"].le(chronology["validation_end"]), "validation", "historical_test"
        )
        final = frame[["holding_month", "signal_date", "permno", "permco", economic, signal_column, "period"]]
        target_root = output / target
        target_root.mkdir(parents=True, exist_ok=True)
        target_path = target_root / "official_predictions.parquet"
        atomic_parquet(final, target_path)
        original = pd.read_parquet(ORIGINAL_RUN / target / "official_predictions.parquet")
        comparisons[target] = {
            **_comparison(final, original, signal_column),
            "signal_column": signal_column,
            "new_prediction_sha256": _file_sha256(target_path),
            "original_prediction_sha256": _file_sha256(ORIGINAL_RUN / target / "official_predictions.parquet"),
        }
        print(
            f"COMPARE {target} max_abs={comparisons[target]['maximum_absolute_score_difference']:.12g} "
            f"rank_corr={comparisons[target]['mean_monthly_rank_correlation']:.12g}",
            flush=True,
        )

    atomic_json(comparisons, output / "signal_comparison.json")
    atomic_json({**manifest_base, "status": "COMPLETE", "comparisons": comparisons}, output / "run_manifest.json")
    print(f"COMPLETE {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
