#!/usr/bin/env python3
"""Audit whether this checkout and an external data root can reproduce V6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = Path("/Users/agraemirolmez/Desktop/Thesis")
EXPECTED_BUILD_IDENTITY = (
    "01c2e95ca6b928446b3524f1c621db01cc6285858ca39ac7646441d1265de2d7"
)

REQUIRED_REPOSITORY_PATHS = (
    "pyproject.toml",
    "configs/phase_01_raw_snapshot.toml",
    "configs/phase_03_features/phase_03_contract_v1.toml",
    "configs/phase_04_models/selector_policy_v1.toml",
    "configs/phase_05_validation/stage_11_validation_data_policy_v1.toml",
    "configs/phase_06_final_test/stage_12_final_system_v1.json",
    "configs/research_v6/selector_step_01.json",
    "configs/research_v6_1/selector_step_01.json",
    "configs/research_v6_1/portfolio_step_02.json",
    "configs/research_v6_1/net_return_frontier.json",
    "scripts/run_phase_01_manifest.py",
    "scripts/build_phase_02_crsp_typed_backbone.py",
    "scripts/build_stage_06_feature_store.py",
    "scripts/build_stage_07_label_store.py",
    "scripts/build_stage_08_risk_cost.py",
    "scripts/run_stage_09_selector_search.py",
    "scripts/build_stage_10_allocators.py",
    "scripts/run_stage_11_validation.py",
    "scripts/build_stage_12_final_freeze.py",
    "scripts/build_stage_13_alpha_diagnostic.py",
    "scripts/download_lseg_selector_data.py",
    "scripts/download_lseg_usd_repair.py",
    "scripts/download_v6_alpaca_news.py",
    "scripts/materialize_v6_news_features.py",
    "scripts/materialize_v6_model_features.py",
    "scripts/materialize_v6_engineered_features.py",
    "scripts/verify_v6_engineered_features.py",
    "scripts/run_v6_selector_step_01.py",
    "scripts/run_v6_1_selector_step_01.py",
    "scripts/reproduce_v6_1_primary_selectors.py",
    "scripts/run_v6_1_primary_portfolio_reproduction.py",
    "scripts/run_v6_1_net_return_frontier.py",
    "src/portfolio_bot/research_v6/selector_runtime.py",
    "src/portfolio_bot/research_v6_1/portfolio.py",
    "evidence/v6_selector/run_manifest.json",
    "evidence/v6_1_net_return_frontier/result.json",
)

REQUIRED_DATABASE_PATHS = (
    "interim/phase_05_validation/stage11-validation-data-v1/candidate_payloads/validation_feature_store_raw.parquet",
    "processed/phase_06_final_test/stage13-final-inputs-v2/primary_predictors.parquet",
    "processed/research_v6/december_2019_v1/predictors.parquet",
    "raw/research_v6/lseg/input_universe.parquet",
    "raw/research_v6/lseg/contract.json",
    "raw/research_v6/lseg_usd_v1/contract.json",
    "processed/research_v6/news_features_v2/manifest.json",
    "processed/research_v6/model_features_v1/manifest.json",
    "processed/research_v6/model_features_v2/manifest.json",
)


def _manifest_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False}
    payload = json.loads(path.read_text())
    return {
        "present": True,
        "status": payload.get("status"),
        "build_identity": payload.get("build_identity"),
        "months": payload.get("months", payload.get("holding_months")),
        "rows": payload.get("rows"),
        "first_holding_month": payload.get("first_holding_month"),
        "last_holding_month": payload.get("last_holding_month"),
    }


def audit(database: Path, *, repository_only: bool = False) -> dict[str, Any]:
    missing_repository = [
        relative
        for relative in REQUIRED_REPOSITORY_PATHS
        if not (PROJECT / relative).exists()
    ]
    result: dict[str, Any] = {
        "repository_root": str(PROJECT),
        "repository_complete": not missing_repository,
        "missing_repository_paths": missing_repository,
    }
    if repository_only:
        return result

    missing_database = [
        relative
        for relative in REQUIRED_DATABASE_PATHS
        if not (database / relative).exists()
    ]
    feature_manifest = _manifest_summary(
        database / "processed/research_v6/model_features_v2/manifest.json"
    )
    raw_news_receipts = len(
        list((database / "raw/research_v6/alpaca_news").glob("*.manifest.json"))
    )
    lseg_receipts = len(list((database / "raw/research_v6/lseg/receipts").glob("*.json")))
    feature_months = len(
        list((database / "processed/research_v6/model_features_v2/months").glob("*.parquet"))
    )
    identity_matches = feature_manifest.get("build_identity") == EXPECTED_BUILD_IDENTITY
    scope_matches = (
        feature_manifest.get("status") == "BACKTEST_READY"
        and feature_manifest.get("months") == 128
        and feature_manifest.get("rows") == 278_897
        and feature_manifest.get("first_holding_month") == "2015-01"
        and feature_manifest.get("last_holding_month") == "2025-08"
    )
    result.update(
        {
            "database_root": str(database),
            "canonical_database_root": database.resolve() == DEFAULT_DATABASE.resolve(),
            "missing_database_paths": missing_database,
            "raw_news_month_receipts": raw_news_receipts,
            "lseg_month_receipts": lseg_receipts,
            "model_feature_months": feature_months,
            "feature_manifest": feature_manifest,
            "feature_identity_matches_frozen_contract": identity_matches,
            "feature_scope_matches_frozen_contract": scope_matches,
            "data_complete": (
                not missing_database
                and raw_news_receipts == 128
                and lseg_receipts == 128
                and feature_months == 128
                and identity_matches
                and scope_matches
            ),
        }
    )
    result["ready_to_start_reproduction"] = (
        result["repository_complete"] and result["data_complete"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-root", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--repository-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    result = audit(args.database_root, repository_only=args.repository_only)
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("Repository closure:", "PASS" if result["repository_complete"] else "FAIL")
        if not args.repository_only:
            print("External frozen inputs:", "PASS" if result["data_complete"] else "FAIL")
            print(
                "Ready to start full V6 replay:",
                "YES" if result["ready_to_start_reproduction"] else "NO",
            )
        for key in ("missing_repository_paths", "missing_database_paths"):
            for path in result.get(key, []):
                print(f"MISSING: {path}")
    complete = result["repository_complete"] and (
        args.repository_only or result["data_complete"]
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
