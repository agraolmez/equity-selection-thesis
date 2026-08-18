#!/usr/bin/env python3
"""Focused independent verification for the V6 318-feature panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DATABASE = Path("/Users/agraemirolmez/Desktop/Thesis")
BASE = DATABASE / "processed/research_v6/model_features_v1"
OUTPUT = DATABASE / "processed/research_v6/model_features_v2"
ENGINEERED = (
    "industry_momentum_12_1_vw",
    "momentum_acceleration_6m",
    "share_turnover_volatility_63d",
    "lseg_forward_eps_yield_fy1",
    "lseg_forward_eps_growth_fy2_vs_fy1",
    "lseg_forward_revenue_growth_fy2_vs_fy1",
    "lseg_price_target_upside",
    "lseg_price_target_revision_1m",
    "lseg_price_target_revision_3m",
    "lseg_recommendation_upgrade_1m",
    "lseg_recommendation_upgrade_3m",
    "quarterly_earnings_change_to_market_cap",
    "quarterly_sales_change_to_market_cap",
    "consecutive_quarterly_earnings_increases",
    "days_since_latest_quarterly_report",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def main() -> int:
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    registry = pd.read_csv(OUTPUT / "feature_registry.csv")
    base_registry = pd.read_csv(BASE / "feature_registry.csv")
    checks: dict[str, int | str | bool] = {
        "manifest_backtest_ready": manifest.get("status") == "BACKTEST_READY",
        "selector_backtest_not_run": manifest.get("selector_backtest_run_by_this_build") is False,
        "registry_features": len(registry),
        "registry_duplicate_names": int(registry["feature_name"].duplicated().sum()),
        "engineered_registry_features": int(registry["feature_name"].isin(ENGINEERED).sum()),
    }

    month_files = sorted((OUTPUT / "months").glob("*.parquet"))
    checks["monthly_files"] = len(month_files)
    checks["monthly_receipts"] = len(list((OUTPUT / "manifests").glob("*.json")))
    receipt_failures = 0
    receipt_rows = 0
    for path in month_files:
        receipt = json.loads((OUTPUT / "manifests" / f"{path.stem}.json").read_text())
        rows = pq.ParquetFile(path).metadata.num_rows
        receipt_rows += rows
        receipt_failures += int(
            receipt.get("status") != "COMPLETE"
            or receipt.get("sha256") != _sha256(path)
            or int(receipt.get("rows", -1)) != rows
            or int(receipt.get("features", -1)) != 318
        )
    checks["receipt_rows"] = receipt_rows
    checks["receipt_failures"] = receipt_failures

    connection = duckdb.connect(database=":memory:")
    connection.execute("SET threads=6")
    try:
        new_glob = str(OUTPUT / "months/*.parquet")
        base_glob = str(BASE / "months/*.parquet")
        scope = connection.execute(
            f"""
            SELECT
                count(*),
                count(DISTINCT holding_month),
                count(*) - count(DISTINCT holding_month || ':' || permno),
                min(holding_month),
                max(holding_month)
            FROM read_parquet('{new_glob}', union_by_name=true)
            """
        ).fetchone()
        if scope is None:
            raise RuntimeError("scope audit returned no row")
        checks.update(
            {
                "rows": int(scope[0]),
                "months": int(scope[1]),
                "duplicate_month_permno_keys": int(scope[2]),
                "first_month": str(scope[3]),
                "last_month": str(scope[4]),
            }
        )

        comparisons = []
        for column in base_registry["feature_name"].astype(str):
            name = _quoted(column)
            comparisons.append(
                "count(*) FILTER (WHERE NOT "
                f"((base.{name} = current.{name}) OR "
                f"(base.{name} IS NULL AND current.{name} IS NULL)))"
            )
        mismatches = 0
        for start in range(0, len(comparisons), 40):
            row = connection.execute(
                f"""
                SELECT {', '.join(comparisons[start:start + 40])}
                FROM read_parquet('{base_glob}', union_by_name=true) AS base
                INNER JOIN read_parquet('{new_glob}', union_by_name=true) AS current
                  USING (holding_month, signal_date, permno, permco)
                """
            ).fetchone()
            if row is None:
                raise RuntimeError("base-value audit returned no row")
            mismatches += sum(int(value) for value in row)
        checks["base_value_mismatches"] = mismatches

        engineered_projection = ", ".join(_quoted(column) for column in ENGINEERED)
        finite_frame = connection.execute(
            f"""
            SELECT {engineered_projection}
            FROM read_parquet('{new_glob}', union_by_name=true)
            """
        ).fetch_df()
        numeric = finite_frame.to_numpy(dtype=float)
        checks["engineered_infinities"] = int(np.isinf(numeric).sum())
        checks["engineered_all_empty_features"] = int(
            finite_frame.notna().sum().eq(0).sum()
        )
        sanity = connection.execute(
            f"""
            SELECT
                count(*) FILTER (WHERE days_since_latest_quarterly_report < 0),
                count(*) FILTER (
                    WHERE consecutive_quarterly_earnings_increases
                          NOT BETWEEN 0 AND 8
                ),
                min(days_since_latest_quarterly_report),
                max(days_since_latest_quarterly_report)
            FROM read_parquet('{new_glob}', union_by_name=true)
            """
        ).fetchone()
        if sanity is None:
            raise RuntimeError("point-in-time sanity audit returned no row")
        checks["future_quarterly_report_age_rows"] = int(sanity[0])
        checks["earnings_increase_range_failures"] = int(sanity[1])
        checks["minimum_quarterly_report_age_days"] = float(sanity[2])
        checks["maximum_quarterly_report_age_days"] = float(sanity[3])
    finally:
        connection.close()

    expected = {
        "manifest_backtest_ready": True,
        "selector_backtest_not_run": True,
        "registry_features": 318,
        "registry_duplicate_names": 0,
        "engineered_registry_features": 15,
        "monthly_files": 128,
        "monthly_receipts": 128,
        "receipt_rows": 278897,
        "receipt_failures": 0,
        "rows": 278897,
        "months": 128,
        "duplicate_month_permno_keys": 0,
        "first_month": "2015-01",
        "last_month": "2025-08",
        "base_value_mismatches": 0,
        "engineered_infinities": 0,
        "engineered_all_empty_features": 0,
        "future_quarterly_report_age_rows": 0,
        "earnings_increase_range_failures": 0,
    }
    failures = {
        key: {"expected": value, "observed": checks.get(key)}
        for key, value in expected.items()
        if checks.get(key) != value
    }
    print(json.dumps({"checks": checks, "failures": failures}, sort_keys=True, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
