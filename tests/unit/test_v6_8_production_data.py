from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from portfolio_bot.research_v6_8.production_data import (
    build_feature_source_matrix,
    summarize_feature_sources,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/research_v6_8/production_feature_sources.json"
REGISTRY = Path(
    "/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/model_features_v2/feature_registry.parquet"
)


def test_all_318_features_are_assigned_once() -> None:
    contract = json.loads(CONFIG.read_text("utf-8"))
    matrix = build_feature_source_matrix(contract, pd.read_parquet(REGISTRY))
    assert len(matrix) == 318
    assert matrix["model_ordinal"].tolist() == list(range(1, 319))
    assert matrix["feature_name"].is_unique
    assert matrix["production_group"].notna().all()


def test_refresh_class_counts_match_the_frozen_source_decision() -> None:
    contract = json.loads(CONFIG.read_text("utf-8"))
    matrix = build_feature_source_matrix(contract, pd.read_parquet(REGISTRY))
    assert matrix.groupby("refresh_class").size().to_dict() == {
        "CAUSAL_NOWCAST_OR_MISSING_POLICY_REQUIRED": 26,
        "PRESERVE_FROZEN_STRUCTURAL_MISSINGNESS": 2,
        "PROVIDER_BRIDGE_REQUIRES_PARITY": 121,
        "SAME_SOURCE_REFRESH_PATH": 169,
    }
    assert summarize_feature_sources(matrix)["feature_count"].sum() == 318


def test_yfinance_is_never_a_primary_source() -> None:
    contract = json.loads(CONFIG.read_text("utf-8"))
    matrix = build_feature_source_matrix(contract, pd.read_parquet(REGISTRY))
    assert contract["production_policy"]["yfinance_primary_allowed"] is False
    assert not matrix["yfinance_primary_allowed"].any()
    assert not matrix["primary_source"].str.contains("Yahoo", case=False).any()


def test_pastor_stambaugh_preserves_the_frozen_missing_path() -> None:
    contract = json.loads(CONFIG.read_text("utf-8"))
    matrix = build_feature_source_matrix(contract, pd.read_parquet(REGISTRY))
    rows = matrix[matrix["feature_name"].str.startswith("pastor_stambaugh")]
    assert len(rows) == 2
    assert set(rows["refresh_class"]) == {"PRESERVE_FROZEN_STRUCTURAL_MISSINGNESS"}


def test_lseg_opra_is_a_parity_bridge_not_an_exact_optionmetrics_refresh() -> None:
    contract = json.loads(CONFIG.read_text("utf-8"))
    matrix = build_feature_source_matrix(contract, pd.read_parquet(REGISTRY))
    rows = matrix[matrix["production_group"].eq("optionmetrics")]
    assert len(rows) == 26
    assert set(rows["refresh_class"]) == {"PROVIDER_BRIDGE_REQUIRES_PARITY"}
    assert rows["primary_source"].str.contains("LSEG OPRA", case=False).all()
