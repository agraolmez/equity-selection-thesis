"""Feature-level summary for the completed V6.8 LSEG market parity gate."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_bot.research_v6_8.lseg_market_parity import LsegMarketParityError


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_lseg_market_derived_parity(
    *, config_path: Path, run_directory: Path
) -> dict[str, Any]:
    """Publish one feature-level record across all completed months."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_receipt_path = run_directory / "receipt.json"
    run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    expected = int(run_receipt["months_expected"])
    if run_receipt.get("scope") != "FULL_WINDOW" or int(
        run_receipt.get("months_completed", 0)
    ) != expected:
        raise LsegMarketParityError("derived-market full window is incomplete")
    parts: list[pd.DataFrame] = []
    metric_hashes: dict[str, str] = {}
    for path in sorted(run_directory.glob("months/*/feature_metrics.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "holding_month", path.parent.name)
        parts.append(frame)
        metric_hashes[path.parent.name] = _sha256(path)
    if len(parts) != expected:
        raise LsegMarketParityError("derived-market monthly metric count differs")
    detail = pd.concat(parts, ignore_index=True)
    summary = (
        detail.groupby("feature", sort=False)
        .agg(
            months_evaluated=("holding_month", "nunique"),
            months_passing=("pilot_gate_pass", "sum"),
            minimum_rank_correlation=("spearman_rank_correlation", "min"),
            median_rank_correlation=("spearman_rank_correlation", "median"),
            maximum_missingness_difference=(
                "absolute_missingness_difference",
                "max",
            ),
            maximum_median_rank_difference=(
                "median_absolute_rank_difference",
                "max",
            ),
            median_relative_difference=(
                "median_absolute_relative_difference",
                "median",
            ),
        )
        .reset_index()
    )
    threshold = float(config["full_window_gate"]["minimum_passing_month_share"])
    summary["passing_month_share"] = (
        summary["months_passing"] / summary["months_evaluated"]
    )
    summary["feature_level_parity_pass"] = summary["passing_month_share"] >= threshold
    if len(summary) != len(config["features"]):
        raise LsegMarketParityError("derived-market feature summary count differs")
    csv_path = run_directory / "feature_window_summary.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    summary.to_csv(temporary, index=False)
    os.replace(temporary, csv_path)
    passed = summary.loc[summary["feature_level_parity_pass"], "feature"].tolist()
    failed = summary.loc[~summary["feature_level_parity_pass"], "feature"].tolist()
    receipt = {
        "schema": "research_v6_8_lseg_market_derived_feature_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL_FEATURE_PARITY_PASS_BUNDLE_FAIL",
        "months": expected,
        "features_tested": int(len(summary)),
        "features_passing_feature_level_gate": int(len(passed)),
        "features_failing_feature_level_gate": int(len(failed)),
        "feature_level_minimum_passing_month_share": threshold,
        "passing_features": passed,
        "failing_features": failed,
        "excluded_cross_sectional_features": config[
            "excluded_cross_sectional_features"
        ],
        "identity": {
            "config_sha256": _sha256(config_path),
            "run_receipt_sha256": _sha256(run_receipt_path),
            "monthly_feature_metric_sha256": metric_hashes,
            "summary_code_sha256": _sha256(Path(__file__)),
        },
    }
    receipt_path = run_directory / "feature_summary_receipt.json"
    temporary_receipt = receipt_path.with_suffix(".json.tmp")
    temporary_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_receipt, receipt_path)
    return receipt


__all__ = ["summarize_lseg_market_derived_parity"]
