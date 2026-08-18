#!/usr/bin/env python3
"""Run the exact V6.1 Step 02 no-forced-short ablation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_v6_1_portfolio_step_02 as base  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/research_v6_1/portfolio_short_floor_ablation.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_1/portfolio_short_floor_ablation"
VARIANT_ID = "return_plus_ff5_momentum__optimized_10pct_vol__no_forced_short"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_identity() -> str:
    files = [
        Path(__file__).resolve(),
        ROOT / "scripts/run_v6_1_portfolio_step_02.py",
        ROOT / "src/portfolio_bot/research_v6_1/portfolio.py",
        ROOT / "src/portfolio_bot/allocation/costs.py",
        ROOT / "src/portfolio_bot/risk_cost/cost_math.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def verify_and_compose(
    overlay_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str, dict[str, Path]]:
    overlay = json.loads(overlay_path.read_text())
    if overlay.get("status") != "FROZEN_READY_TO_RUN":
        raise RuntimeError("short-floor ablation contract is not frozen")
    base_path = ROOT / str(overlay["base_contract"]["path"])
    if _sha256(base_path) != overlay["base_contract"]["sha256"]:
        raise RuntimeError("bound Step 02 contract identity changed")
    base_config, _, input_identity, paths = base.verify_contract(base_path)
    result_root = ROOT / str(overlay["base_result"]["path"])
    manifest_path = result_root / "run_manifest.json"
    if _sha256(manifest_path) != overlay["base_result"]["manifest_sha256"]:
        raise RuntimeError("bound Step 02 result identity changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != overlay["base_result"]["required_status"]:
        raise RuntimeError("bound Step 02 result is not complete")
    if manifest.get("config_sha256") != overlay["base_contract"]["sha256"]:
        raise RuntimeError("bound Step 02 result used another contract")

    # The economic experiment is composed from the authenticated Step 02
    # contract.  Only the two explicitly declared values below can differ.
    if float(base_config["optimizer"]["minimum_short_gross"]) != 0.30:
        raise RuntimeError("Step 02 explicit short floor is not the expected 0.30")
    if float(base_config["optimizer"]["maximum_net"]) != 0.30:
        raise RuntimeError("Step 02 maximum net is not the expected 0.30")
    if float(base_config["optimizer"]["minimum_long_gross"]) != 0.50:
        raise RuntimeError("Step 02 minimum long gross is not the expected 0.50")
    composed = copy.deepcopy(base_config)
    composed["optimizer"]["minimum_short_gross"] = 0.0
    composed["optimizer"]["maximum_net"] = 1.0

    frozen = overlay["frozen_variant"]
    expected = {
        "policy": "return_plus_ff5_momentum",
        "kind": "optimized",
        "method": "optimized",
        "volatility": 0.10,
    }
    if frozen != expected:
        raise RuntimeError("ablation variant differs from the frozen FF5 control")
    return overlay, composed, _sha256(overlay_path), input_identity, paths


def _variant(overlay: dict[str, Any]) -> dict[str, Any]:
    return {"id": VARIANT_ID, **overlay["frozen_variant"]}


def run(
    config_path: Path,
    output_root: Path,
    *,
    check_only: bool,
    smoke_month: str | None,
) -> Path | None:
    overlay, config, config_sha, input_identity, paths = verify_and_compose(config_path)
    code_identity = _code_identity()
    identity = {
        "config_sha256": config_sha,
        "input_identity": input_identity,
        "code_identity": code_identity,
    }
    if check_only:
        print(
            "READY: exact FF5 Step 02 path; explicit short floor 0.30->0.00; "
            "maximum net 0.30->1.00 solely to remove the implicit 0.20 short floor; "
            "all other economics inherited unchanged",
            flush=True,
        )
        return None

    output = output_root / f"{config_sha[:12]}-{input_identity[:12]}-{code_identity[:12]}"
    if smoke_month:
        output = output / "smoke"
    variant = _variant(overlay)
    base._atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_1_exact_short_floor_ablation_run_v1",
            "status": "RUNNING",
            **identity,
            "config": str(config_path.resolve()),
            "variant": VARIANT_ID,
            "smoke_month": smoke_month,
            "only_economic_changes": overlay["only_economic_changes"],
        },
    )
    inputs = base.load_inputs(config, paths)
    months = pd.period_range(
        config["evidence"]["portfolio_start"], config["evidence"]["portfolio_end"], freq="M"
    ).astype(str).tolist()
    if smoke_month:
        if smoke_month not in months:
            raise ValueError("smoke month is outside the frozen Step 02 window")
        months = [smoke_month]

    completed, previous, nav, summaries = base._load_resume_state(
        output, VARIANT_ID, months, identity
    )
    month_cache: dict[str, tuple[pd.DataFrame, Any]] = {}
    for month in months[completed:]:
        print(f"START {VARIANT_ID} {month}", flush=True)
        try:
            if month not in month_cache:
                month_cache[month] = base._month_frame(inputs, config, month)
            frame, covariance_root = month_cache[month]
            _, positions, summary = base._solve_variant_month(
                frame, covariance_root, config, variant, previous, nav
            )
            parquet_path, receipt_path = base._checkpoint_paths(output, VARIANT_ID, month)
            base._atomic_parquet(parquet_path, positions)
            base._atomic_json(
                receipt_path,
                {
                    **identity,
                    "variant": VARIANT_ID,
                    "holding_month": month,
                    "summary": summary,
                },
            )
            previous = {
                int(row.permno): float(row.end_drift_weight)
                for row in positions.itertuples(index=False)
                if abs(float(row.end_drift_weight)) > 1e-12
            }
            nav = float(summary["nav_end_primary"])
            summaries.append(summary)
            print(
                f"DONE {month} net={summary['net_excess_return_primary']:.6f} "
                f"L={summary['long_gross']:.3f} S={summary['short_gross']:.3f} "
                f"netexp={summary['net']:.3f}",
                flush=True,
            )
        except Exception as exc:
            base._atomic_json(
                output / "error.json",
                {
                    **identity,
                    "holding_month": month,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    monthly = pd.DataFrame(summaries).drop_duplicates(
        ["variant", "holding_month"], keep="last"
    ).sort_values("holding_month", kind="mergesort").reset_index(drop=True)
    summary = pd.DataFrame([base._aggregate_variant(monthly)])
    base._atomic_parquet(output / "monthly_results.parquet", monthly)
    base._atomic_parquet(output / "variant_summary.parquet", summary)

    control = pd.read_parquet(
        ROOT / str(overlay["base_result"]["path"]) / "variant_summary.parquet"
    )
    control = control.loc[
        control["variant"].eq(overlay["base_result"]["control_variant"])
    ].copy()
    control["lineage"] = "v6_1_step_02_forced_short_control"
    challenger = summary.copy()
    challenger["lineage"] = "v6_1_exact_no_forced_short_ablation"
    shared = sorted(set(control.columns) & set(challenger.columns))
    base._atomic_parquet(
        output / "comparison_summary.parquet",
        pd.concat([control[shared], challenger[shared]], ignore_index=True),
    )
    base._atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_1_exact_short_floor_ablation_run_v1",
            "status": "SMOKE_COMPLETE" if smoke_month else "COMPLETE",
            **identity,
            "config": str(config_path.resolve()),
            "variant": VARIANT_ID,
            "completed_months": int(len(monthly)),
            "expected_months": int(len(months)),
            "selector_retrained": False,
            "retuned": False,
            "minimum_short_gross": 0.0,
            "maximum_net": 1.0,
            "only_economic_changes": overlay["only_economic_changes"],
        },
    )
    print(f"COMPLETE {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-month")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.config.resolve(),
        arguments.output_root.resolve(),
        check_only=arguments.check_only,
        smoke_month=arguments.smoke_month,
    )
