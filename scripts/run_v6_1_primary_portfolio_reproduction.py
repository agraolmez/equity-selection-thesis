#!/usr/bin/env python3
"""Run the frozen primary V6.1 portfolio from independently rebuilt signals."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v6_1_portfolio_step_02 import _atomic_json, run  # noqa: E402


SELECTOR_RUN = ROOT / (
    "artifacts/research_v6_1/primary_selector_clean_reproduction/"
    "481e306553c2-16046ee58521-d5dadd9cd9c3"
)
BASE_CONFIG = ROOT / "configs/research_v6_1/portfolio_step_02.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_1/portfolio_step_02_clean_reproduction"
VARIANT = "return_plus_ff5_momentum__optimized_10pct_vol"
ORIGINAL_PORTFOLIO = ROOT / (
    "artifacts/research_v6_1/portfolio_step_02/"
    "48f429eb0d7f-53a6425a3d92-7256b91cc393"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _build_contract() -> Path:
    manifest = SELECTOR_RUN / "run_manifest.json"
    state = json.loads(manifest.read_text())
    if state.get("status") != "COMPLETE":
        raise RuntimeError("clean primary-selector reproduction is not complete")
    config = copy.deepcopy(json.loads(BASE_CONFIG.read_text()))
    config["question"] = (
        "Does the frozen primary V6.1 portfolio reproduce when every selected "
        "model component is refitted from scratch with the same frozen seeds?"
    )
    config["scope"]["selector_retraining_authorized"] = True
    config["scope"]["selector_choices_frozen"] = True
    config["selector_run"]["path"] = _relative(SELECTOR_RUN)
    config["selector_run"]["manifest_sha256"] = _sha256(manifest)
    for binding in config["source_bindings"]:
        identifier = str(binding["id"])
        if not identifier.endswith("_predictions"):
            continue
        target = identifier.removesuffix("_predictions")
        prediction = SELECTOR_RUN / target / "official_predictions.parquet"
        binding["path"] = _relative(prediction)
        binding["sha256"] = _sha256(prediction)
    contract = SELECTOR_RUN / "portfolio_reproduction_contract.json"
    _atomic_json(contract, config)
    return contract


def _verify_reproduction(output: Path) -> dict[str, object]:
    old_monthly = pd.read_parquet(ORIGINAL_PORTFOLIO / "monthly_results.parquet")
    old_monthly = old_monthly.loc[old_monthly["variant"].eq(VARIANT)].sort_values(
        "holding_month", kind="mergesort"
    ).reset_index(drop=True)
    new_monthly = pd.read_parquet(output / "monthly_results.parquet").sort_values(
        "holding_month", kind="mergesort"
    ).reset_index(drop=True)
    same_monthly_columns = set(old_monthly.columns) == set(new_monthly.columns)
    if same_monthly_columns:
        new_monthly = new_monthly.loc[:, old_monthly.columns]
    numeric = old_monthly.select_dtypes(include="number").columns.intersection(new_monthly.columns)
    numeric_delta = (old_monthly[numeric] - new_monthly[numeric]).abs().to_numpy()
    old_summary = pd.read_parquet(ORIGINAL_PORTFOLIO / "variant_summary.parquet")
    old_summary = old_summary.loc[old_summary["variant"].eq(VARIANT)].reset_index(drop=True)
    new_summary = pd.read_parquet(output / "variant_summary.parquet").reset_index(drop=True)
    identical_positions = 0
    position_months = 0
    new_month_root = output / "variants" / VARIANT / "months"
    old_month_root = ORIGINAL_PORTFOLIO / "variants" / VARIANT / "months"
    for path in sorted(new_month_root.glob("*.parquet")):
        original = old_month_root / path.name
        if not original.exists():
            continue
        position_months += 1
        identical_positions += _sha256(path) == _sha256(original)
    return {
        "variant": VARIANT,
        "monthly_rows": int(len(new_monthly)),
        "monthly_columns_equal_ignoring_order": same_monthly_columns,
        "monthly_values_exact_after_column_alignment": bool(old_monthly.equals(new_monthly)),
        "summary_frame_exact": bool(old_summary.equals(new_summary)),
        "maximum_absolute_numeric_difference": float(np.nanmax(numeric_delta)),
        "position_months": position_months,
        "byte_identical_position_months": identical_positions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = _build_contract()
    output = run(
        contract,
        args.output_root.resolve(),
        check_only=False,
        selected_variants={VARIANT},
        smoke_month=None,
        max_months=None,
    )
    if output is None:
        raise RuntimeError("portfolio reproduction did not produce an output")
    verification = _verify_reproduction(output)
    _atomic_json(output / "reproduction_verification.json", verification)
    if not (
        verification["monthly_columns_equal_ignoring_order"]
        and verification["monthly_values_exact_after_column_alignment"]
        and verification["summary_frame_exact"]
        and verification["position_months"] == verification["byte_identical_position_months"] == 20
    ):
        raise RuntimeError(f"portfolio reproduction differs: {verification}")
    print(json.dumps(verification, indent=2), flush=True)
    print(f"REPRODUCTION_COMPLETE {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
