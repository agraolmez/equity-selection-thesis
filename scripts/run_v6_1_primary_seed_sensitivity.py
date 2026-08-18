#!/usr/bin/env python3
"""Evaluate the frozen primary portfolio using each prespecified test-fit seed."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from scripts.run_v6_1_portfolio_step_02 import _atomic_json, _atomic_parquet, run  # noqa: E402
from scripts.run_v6_1_selector_step_01 import _verify_inputs  # noqa: E402
from portfolio_bot.research_v6_1.feature_views import direct_cam_predictions  # noqa: E402
from portfolio_bot.research_v6_1.models import add_equal_ensemble  # noqa: E402


SELECTOR_RUN = ROOT / (
    "artifacts/research_v6_1/primary_selector_clean_reproduction/"
    "481e306553c2-16046ee58521-d5dadd9cd9c3"
)
SELECTOR_CONFIG = ROOT / "configs/research_v6_1/selector_step_01.json"
PORTFOLIO_CONFIG = ROOT / "configs/research_v6_1/portfolio_step_02.json"
OUTPUT_ROOT = ROOT / "artifacts/research_v6_1/portfolio_step_02_seed_sensitivity"
VARIANT = "return_plus_ff5_momentum__optimized_10pct_vol"
SEEDS = (17, 42, 101)
KEYS = ["holding_month", "signal_date", "permno", "permco"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _component_path(target: str, task: str) -> Path:
    return SELECTOR_RUN / "components" / target / task / "component_predictions.parquet"


def _seed_signal_files(seed: int) -> Path:
    selector_config = json.loads(SELECTOR_CONFIG.read_text())
    verified = _verify_inputs(selector_config, load_matrix=True)
    rows = verified["rows"]
    chronology = selector_config["chronology"]
    cam = direct_cam_predictions(
        matrix=verified["matrix"],
        rows=rows,
        metadata=verified["metadata"],
        source=str(selector_config["feature_views"]["direct_cam_level"]),
        start_month=chronology["validation_start"],
        end_month=chronology["historical_test_end"],
    )[["holding_month", "permno", "score_rank"]].rename(columns={"score_rank": "cam_direct"})
    root = SELECTOR_RUN / "seed_sensitivity" / f"seed_{seed}"
    specifications = {
        "capm": [("cam_free__mlp", "cam_free__neural_network__continuous_rank")],
        "ff5_momentum": [("cam_free__mlp", "cam_free__neural_network__pairwise")],
        "long_net": [("cam_free__mlp", "cam_free__neural_network__continuous_rank")],
        "short_net": [
            ("correlation_reduced__xgboost", "correlation_reduced__xgboost__pairwise"),
            ("correlation_reduced__mlp", "correlation_reduced__neural_network__pairwise"),
        ],
    }
    signal_columns = {
        "capm": "ensemble__cam_free_mlp_plus_cam__equal",
        "ff5_momentum": "ensemble__cam_free_mlp_plus_cam__equal",
        "long_net": "ensemble__cam_free_mlp_plus_cam__equal",
        "short_net": "ensemble__correlation_reduced_xgb_mlp__equal",
    }
    bindings: dict[str, Path] = {}
    for target, components in specifications.items():
        original = pd.read_parquet(SELECTOR_RUN / target / "official_predictions.parquet")
        economic = str(selector_config["targets"][target]["economic_column"])
        frame = original[KEYS + [economic, "period"]].copy()
        component_names: list[str] = []
        for component, task in components:
            prediction = pd.read_parquet(_component_path(target, task))
            selected = prediction["score_rank"].copy()
            historical = prediction["holding_month"].ge(chronology["historical_test_start"])
            selected.loc[historical] = prediction.loc[historical, f"score_rank__seed_{seed}"]
            source = prediction[["holding_month", "permno"]].copy()
            source[component] = selected
            frame = frame.merge(source, on=["holding_month", "permno"], validate="one_to_one")
            component_names.append(component)
        if target != "short_net":
            frame = frame.merge(cam, on=["holding_month", "permno"], validate="one_to_one")
            component_names.append("cam_direct")
        signal = signal_columns[target]
        frame = add_equal_ensemble(frame, components=component_names, output_column=signal)
        path = root / target / "official_predictions.parquet"
        _atomic_parquet(path, frame[KEYS + [economic, signal, "period"]])
        bindings[target] = path
    _atomic_json(
        root / "run_manifest.json",
        {
            "schema": "research_v6_1_primary_seed_sensitivity_v1",
            "status": "COMPLETE",
            "seed": seed,
            "validation_seed": 42,
            "historical_test_seed": seed,
            "purpose": "Diagnostic only; no seed-based winner selection",
        },
    )
    return root


def _portfolio_contract(selector_root: Path, seed: int) -> Path:
    config = copy.deepcopy(json.loads(PORTFOLIO_CONFIG.read_text()))
    config["question"] = f"Primary portfolio sensitivity to prespecified historical-test seed {seed}."
    config["scope"]["selector_retraining_authorized"] = True
    config["selector_run"]["path"] = _relative(selector_root)
    config["selector_run"]["manifest_sha256"] = _sha256(selector_root / "run_manifest.json")
    for binding in config["source_bindings"]:
        identifier = str(binding["id"])
        if identifier.endswith("_predictions"):
            target = identifier.removesuffix("_predictions")
            path = selector_root / target / "official_predictions.parquet"
            binding["path"] = _relative(path)
            binding["sha256"] = _sha256(path)
    contract = selector_root / "portfolio_contract.json"
    _atomic_json(contract, config)
    return contract


def main() -> int:
    outputs: list[dict[str, object]] = []
    for seed in SEEDS:
        selector_root = _seed_signal_files(seed)
        contract = _portfolio_contract(selector_root, seed)
        portfolio_output = run(
            contract,
            OUTPUT_ROOT / f"seed_{seed}",
            check_only=False,
            selected_variants={VARIANT},
            smoke_month=None,
            max_months=None,
        )
        if portfolio_output is None:
            raise RuntimeError(f"seed {seed} did not produce a portfolio")
        summary = pd.read_parquet(portfolio_output / "variant_summary.parquet").iloc[0]
        outputs.append(
            {
                "seed": seed,
                "portfolio_output": str(portfolio_output),
                "annualized_net_excess_return_primary": float(summary["annualized_net_excess_return_primary"]),
                "annualized_realized_volatility_primary": float(summary["annualized_realized_volatility_primary"]),
                "net_sharpe_primary": float(summary["net_sharpe_primary"]),
                "maximum_drawdown_primary": float(summary["maximum_drawdown_primary"]),
            }
        )
    destination = OUTPUT_ROOT / "seed_sensitivity_summary.json"
    _atomic_json(destination, outputs)
    print(json.dumps(outputs, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
