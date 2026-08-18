from __future__ import annotations

import pandas as pd

from portfolio_bot.research_v6_8.live_feature_package import (
    REQUIRED_SOURCE_GROUPS,
    assemble_live_feature_panel,
)


def test_live_feature_assembly_retains_universe_and_derives_cross_section() -> None:
    features = [
        (1, "momentum_6_1", "continuous_firm", "market_price_liquidity"),
        (2, "momentum_12_1", "continuous_firm", "market_price_liquidity"),
        (3, "log_market_cap", "continuous_firm", "market_price_liquidity"),
        (4, "fundamental_source_basis", "categorical_firm", "valuation_fundamentals"),
        (5, "institutional_ownership_ratio", "continuous_firm", "institutional_ownership"),
        (6, "filing_count_30d", "continuous_firm", "sec_filings_events"),
        (7, "sic2_industry_category", "categorical_firm", "security_identity"),
        (8, "optionmetrics_covered_indicator", "binary_firm", "optionmetrics"),
        (9, "beta_label_mktrf", "continuous_firm", "factor_dependent"),
        (10, "lseg_cam_country_rank", "continuous_firm", "lseg_analyst_cam"),
        (11, "news_source_available", "binary_firm", "alpaca_related_news"),
    ]
    registry = pd.DataFrame(
        features, columns=["model_ordinal", "feature_name", "kind", "production_group"]
    )
    source_matrix = registry[["model_ordinal", "feature_name", "production_group"]]
    universe = pd.DataFrame(
        {
            "live_security_id": ["a", "b"],
            "holding_month": ["2026-09", "2026-09"],
            "signal_date": ["2026-08-31", "2026-08-31"],
            "TickerSymbol": ["AAA", "BBB"],
            "RIC": ["AAA.N", "BBB.N"],
            "PermID": ["1", "2"],
            "issuer_key": ["i1", "i2"],
        }
    )
    group_frames: dict[str, pd.DataFrame] = {}
    values = {
        "momentum_6_1": [0.10, 0.20],
        "momentum_12_1": [0.25, 0.35],
        "log_market_cap": [10.0, 11.0],
        "fundamental_source_basis": ["UNKNOWN", "UNKNOWN"],
        "institutional_ownership_ratio": [None, None],
        "filing_count_30d": [0.0, 1.0],
        "sic2_industry_category": ["35", "35"],
        "optionmetrics_covered_indicator": [None, None],
        "beta_label_mktrf": [0.9, 1.1],
        "lseg_cam_country_rank": [80.0, 20.0],
        "news_source_available": [1, 1],
    }
    for group in REQUIRED_SOURCE_GROUPS:
        names = source_matrix.loc[
            source_matrix["production_group"].eq(group), "feature_name"
        ].tolist()
        frame = pd.DataFrame({"live_security_id": ["a", "b"]})
        for name in names:
            frame[name] = values[name]
        if group == "market_price_liquidity":
            frame["long_eligible"] = [True, False]
        group_frames[group] = frame
    panel, audit = assemble_live_feature_panel(
        universe=universe,
        registry=registry,
        source_matrix=source_matrix,
        group_frames=group_frames,
    )
    assert audit["status"] == "PASS"
    assert audit["rows"] == 2
    assert audit["eligible_rows"] == 1
    assert panel["momentum_acceleration_6m"].notna().all()
    assert panel["industry_momentum_12_1_vw"].notna().all()
