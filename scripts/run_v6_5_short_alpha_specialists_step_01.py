#!/usr/bin/env python3
"""Run V6.5 side-specific CAPM and FF5 short-alpha selector research."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
for path in (ROOT, SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_v6_1_selector_step_01 as base  # noqa: E402
from scripts import run_v6_4_negative_alpha_selector_step_01 as engine  # noqa: E402
from portfolio_bot.research_v6.evaluation import _hac_mean  # noqa: E402
from portfolio_bot.research_v6.selector_runtime import (  # noqa: E402
    atomic_json,
    atomic_parquet,
    canonical_sha256,
)
from portfolio_bot.research_v6_1.feature_views import prepare_feature_views  # noqa: E402


STABLE_PYTHON = Path("/opt/anaconda3/envs/finance/bin/python")
DEFAULT_CONFIG = ROOT / "configs/research_v6_5/short_alpha_specialists_step_01.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/research_v6_5/short_alpha_specialists_step_01"
CODE_PATHS = (
    ROOT / "scripts/run_v6_5_short_alpha_specialists_step_01.py",
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


def _load(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    contract = json.loads(config_path.read_text())
    base_path = ROOT / str(contract["base_selector_contract"]["path"])
    observed = _file_sha256(base_path)
    expected = str(contract["base_selector_contract"]["sha256"])
    if observed != expected:
        raise ValueError(f"frozen V6.1 selector contract changed: {observed} != {expected}")
    config = json.loads(base_path.read_text())
    return contract, config, base_path


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


def centered_downside_rank(alpha: pd.Series, months: pd.Series) -> np.ndarray:
    """Rank only negative-alpha severity; non-negative outcomes tie at zero."""

    numeric = pd.to_numeric(alpha, errors="coerce").to_numpy(float)
    finite = np.isfinite(numeric)
    opportunity = np.full(len(numeric), np.nan, dtype=np.float64)
    opportunity[finite] = np.maximum(-numeric[finite], 0.0)
    frame = pd.DataFrame({"month": months.astype(str).to_numpy(), "value": opportunity})
    rank = frame.groupby("month", observed=True)["value"].rank(method="average")
    count = frame.groupby("month", observed=True)["value"].transform("count")
    return ((rank - 0.5) / count - 0.5).to_numpy(np.float64)


def _target_contract(
    contract: Mapping[str, Any], target_id: str
) -> dict[str, Any]:
    spec = contract["targets"][target_id]
    source_rank = f"v65_source_rank__{target_id}"
    control = spec["frozen_control"]
    return {
        "schema": contract["schema"],
        "status": contract["status"],
        "scope": contract["scope"],
        "target": {
            "target_id": target_id,
            "source_rank_column": source_rank,
            "source_economic_column": spec["source_alpha_column"],
            "rank_column": spec["rank_column"],
            "economic_column": spec["economic_column"],
            "construction": spec["construction"],
            "higher_is_more_attractive_short_alpha": True,
        },
        "frozen_control": {
            "path": control["path"],
            "sha256": control["sha256"],
            "score_column": control["score_column"],
            "negative_alpha_control_column": control["control_column"],
        },
        "selection": contract["selection"],
    }


def _target_config(
    base_config: Mapping[str, Any], contract: Mapping[str, Any], target_id: str
) -> dict[str, Any]:
    spec = contract["targets"][target_id]
    source_target = str(spec["source_target_id"])
    config = copy.deepcopy(base_config)
    config["schema"] = str(contract["schema"])
    config["status"] = str(contract["status"])
    config["scope"] = copy.deepcopy(contract["scope"])
    config["targets"] = {
        target_id: {
            "rank_column": str(spec["rank_column"]),
            "economic_column": str(spec["economic_column"]),
        }
    }
    for family in ("xgboost", "neural_network"):
        config["operating_policy"][family] = {
            target_id: copy.deepcopy(
                base_config["operating_policy"][family][source_target]
            )
        }
    config["execution"]["default_parallel_targets"] = 1
    config["execution"]["full_run_allowed_by_current_status"] = True
    return config


def _rows_for_target(
    rows: pd.DataFrame, target_contract: Mapping[str, Any]
) -> pd.DataFrame:
    result = rows.copy()
    target = target_contract["target"]
    desired = centered_downside_rank(
        result[str(target["source_economic_column"])], result["holding_month"]
    )
    result[str(target["source_rank_column"])] = -desired
    attached = engine.attach_negative_alpha_target(result, target_contract)
    observed = attached[str(target["rank_column"])].to_numpy(float)
    if not np.allclose(observed, desired, rtol=0.0, atol=0.0, equal_nan=True):
        raise ValueError("downside-specific target attachment changed its ordering")
    return result


def _paired_hac(
    monthly: pd.DataFrame, challenger: str, control: str, period: str
) -> dict[str, float]:
    pivot = monthly.loc[monthly["period"].eq(period)].pivot(
        index="holding_month", columns="model_id", values="rank_ic"
    )
    values = pivot[challenger] - pivot[control]
    mean, standard_error, statistic, lower = _hac_mean(values, 6)
    return {
        "mean": float(mean),
        "standard_error": float(standard_error),
        "hac_t": float(statistic),
        "lower_95": float(lower),
    }


def _tail_row(path: Path, period: str, model_id: str) -> Mapping[str, Any]:
    frame = pd.read_parquet(path)
    row = frame.loc[
        frame["period"].eq(period)
        & frame["model_id"].eq(model_id)
        & np.isclose(frame["tail_fraction"], 0.10)
    ]
    if len(row) != 1:
        raise ValueError(f"missing unique top-decile diagnostic for {period}/{model_id}")
    return row.iloc[0].to_dict()


def _publish_promotion(target_output: Path, target_contract: Mapping[str, Any]) -> None:
    selection_path = target_output / "validation_selection.json"
    selection = json.loads(selection_path.read_text())
    challenger = str(selection["challenger_model_id"])
    control = str(selection["control_model_id"])
    monthly = pd.read_parquet(target_output / "official_monthly_metrics.parquet")
    validation_delta = _paired_hac(monthly, challenger, control, "validation")
    test_delta = _paired_hac(monthly, challenger, control, "historical_test")
    tail_path = target_output / "negative_alpha_tail_summary.parquet"
    challenger_validation_tail = _tail_row(tail_path, "validation", challenger)
    control_validation_tail = _tail_row(tail_path, "validation", control)
    challenger_test_tail = _tail_row(tail_path, "historical_test", challenger)
    control_test_tail = _tail_row(tail_path, "historical_test", control)
    rule = target_contract["selection"]["promotion_vs_reversed_control"]
    validation_passes = {
        "mean_ic_improvement": validation_delta["mean"]
        >= float(rule["minimum_mean_ic_improvement"]),
        "paired_hac_t": validation_delta["hac_t"]
        >= float(rule["minimum_paired_hac_t"]),
        "top_decile_mean_alpha_more_negative": float(
            challenger_validation_tail["mean_realized_ff5_alpha"]
        )
        < float(control_validation_tail["mean_realized_ff5_alpha"]),
        "top_decile_negative_fraction_not_lower": float(
            challenger_validation_tail["mean_negative_alpha_fraction"]
        )
        >= float(control_validation_tail["mean_negative_alpha_fraction"]),
    }
    selected_passed_validation = all(validation_passes.values())
    test_confirmed = (
        selected_passed_validation
        and test_delta["mean"] > 0.0
        and float(challenger_test_tail["mean_realized_ff5_alpha"])
        < float(control_test_tail["mean_realized_ff5_alpha"])
        and float(challenger_test_tail["mean_negative_alpha_fraction"])
        >= float(control_test_tail["mean_negative_alpha_fraction"])
    )
    result = {
        "schema": "research_v6_5_short_alpha_promotion_evaluation_v1",
        "target": str(target_contract["target"]["target_id"]),
        "challenger": challenger,
        "reversed_control": control,
        "validation_delta": validation_delta,
        "validation_tail": {
            "challenger": challenger_validation_tail,
            "control": control_validation_tail,
        },
        "validation_rules": validation_passes,
        "selected_challenger_passed_validation": selected_passed_validation,
        "historical_test_delta": test_delta,
        "historical_test_tail": {
            "challenger": challenger_test_tail,
            "control": control_test_tail,
        },
        "historical_test_confirmed": test_confirmed,
        "portfolio_gate_test_justified": test_confirmed,
    }
    atomic_json(result, target_output / "promotion_evaluation.json")


def _run_one(
    *,
    target_id: str,
    contract: Mapping[str, Any],
    base_config: Mapping[str, Any],
    verified: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    output: Path,
    code_identity: str,
) -> None:
    target_contract = _target_contract(contract, target_id)
    config = _target_config(base_config, contract, target_id)
    rows = _rows_for_target(verified["rows"], target_contract)
    target_verified = {**verified, "rows": rows}
    engine.TARGET_ID = target_id
    engine._label_shuffle_smoke(
        contract=target_contract,
        config=config,
        verified=target_verified,
        feature_manifest=feature_manifest,
        output=output / target_id,
        code_identity=code_identity,
    )
    engine._run_target(
        contract=target_contract,
        config=config,
        verified=target_verified,
        feature_manifest=feature_manifest,
        output=output,
        code_identity=code_identity,
    )
    _publish_promotion(output / target_id, target_contract)
    atomic_json(
        {
            "schema": "research_v6_5_target_status_v1",
            "target": target_id,
            "status": "COMPLETE",
        },
        output / target_id / "v6_5_status.json",
    )


def _smoke_test() -> None:
    sample = pd.DataFrame(
        {
            "holding_month": ["2020-01"] * 5,
            "alpha": [-0.10, -0.02, 0.00, 0.03, 0.10],
        }
    )
    observed = centered_downside_rank(sample["alpha"], sample["holding_month"])
    expected = np.asarray([0.4, 0.2, -0.2, -0.2, -0.2])
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)
    if np.allclose(observed, -np.asarray([-0.4, -0.2, 0.0, 0.2, 0.4])):
        raise AssertionError("short specialist target accidentally equals reversed full rank")
    base._smoke_test()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--prepare-views", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--target", choices=["short_capm_alpha", "short_ff5_alpha"])
    parser.add_argument("--native-runtime", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.native_runtime and Path(sys.executable).resolve() != STABLE_PYTHON.resolve():
        if not STABLE_PYTHON.is_file():
            raise FileNotFoundError(f"stable V6.5 runtime is missing: {STABLE_PYTHON}")
        os.execv(
            str(STABLE_PYTHON),
            [str(STABLE_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:], "--native-runtime"],
        )

    config_path = args.config.resolve()
    contract, base_config, base_path = _load(config_path)
    if args.smoke_test:
        _smoke_test()
        print("PASS: V6.5 downside-specific target and inherited model runtime smoke tests.", flush=True)
        return 0
    verified = base._verify_inputs(base_config, load_matrix=args.prepare_views)
    if args.check_only:
        print(
            "READY: side-specific CAPM and FF5 short-alpha selectors; three feature views, "
            "XGBoost, MLP, Elastic Net, equal/dynamic ensembles, reversed controls, and no portfolio.",
            flush=True,
        )
        return 0

    code_identity = _code_identity(config_path, base_path)
    output = args.output_root.resolve() / (
        f"{_file_sha256(config_path)[:12]}-"
        f"{base_config['inputs']['v6_input_identity'][:12]}-{code_identity[:12]}"
    )
    output.mkdir(parents=True, exist_ok=True)
    matrix = verified.get("matrix")
    if matrix is None:
        matrix = np.load(verified["paths"]["matrix"], mmap_mode="r")
    feature_manifest = prepare_feature_views(
        matrix=matrix,
        rows=verified["rows"],
        metadata=verified["metadata"],
        config=base_config,
        input_identity=str(base_config["inputs"]["v6_input_identity"]),
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
    if contract["status"] != "FROZEN_READY_TO_RUN" or not contract["execution"][
        "full_run_allowed_by_current_status"
    ]:
        raise PermissionError("V6.5 full fit is not authorized")

    verified_with_matrix = {**verified, "matrix": matrix}
    manifest = {
        "schema": "research_v6_5_short_alpha_selector_run_manifest_v1",
        "config": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "base_config_sha256": _file_sha256(base_path),
        "code_identity": code_identity,
        "v6_input_identity": base_config["inputs"]["v6_input_identity"],
        "feature_view_identity": feature_manifest["identity"],
        "targets": list(contract["targets"]),
        "portfolio_constructed": False,
        "optimizer_run": False,
    }
    if args.target:
        _run_one(
            target_id=args.target,
            contract=contract,
            base_config=base_config,
            verified=verified_with_matrix,
            feature_manifest=feature_manifest,
            output=output,
            code_identity=code_identity,
        )
        print(f"V6.5 target complete: {args.target}", flush=True)
        return 0

    atomic_json({**manifest, "status": "RUNNING"}, output / "run_manifest.json")
    commands = [
        [
            str(STABLE_PYTHON),
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--output-root",
            str(args.output_root.resolve()),
            "--target",
            target_id,
            "--native-runtime",
        ]
        for target_id in contract["targets"]
    ]
    processes = [subprocess.Popen(command) for command in commands]
    exit_codes = [process.wait() for process in processes]
    if any(code != 0 for code in exit_codes):
        atomic_json(
            {**manifest, "status": "FAILED", "target_exit_codes": exit_codes},
            output / "run_manifest.json",
        )
        raise RuntimeError(f"V6.5 target process failed: {exit_codes}")
    atomic_json({**manifest, "status": "COMPLETE"}, output / "run_manifest.json")
    print(f"V6.5 short-alpha specialist study complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
