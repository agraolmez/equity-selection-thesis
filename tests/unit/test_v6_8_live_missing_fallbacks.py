from __future__ import annotations

import json

import pandas as pd

from portfolio_bot.research_v6_8.live_missing_fallbacks import materialize_missing_fallback


def test_fundamental_fallback_is_explicit_and_hash_bound(tmp_path) -> None:
    universe = tmp_path / "universe.parquet"
    registry = tmp_path / "registry.parquet"
    config = tmp_path / "config.json"
    output = tmp_path / "output"
    pd.DataFrame({"live_security_id": ["a", "b"]}).to_parquet(universe, index=False)
    pd.DataFrame(
        {
            "model_ordinal": [54, 55],
            "feature_name": ["earnings_to_price", "fundamental_source_basis"],
            "kind": ["continuous_firm", "categorical_firm"],
        }
    ).to_parquet(registry, index=False)
    config.write_text(
        json.dumps(
            {
                "schema": "research_v6_8_live_missing_fallbacks_v1",
                "allowed_groups": {
                    "valuation_fundamentals": {
                        "ordinal_ranges": [[54, 55]],
                        "categorical_constants": {"fundamental_source_basis": "UNKNOWN"},
                        "reason": "not equivalent",
                    }
                },
            }
        )
    )
    receipt = materialize_missing_fallback(
        config_path=config,
        group="valuation_fundamentals",
        universe_path=universe,
        registry_path=registry,
        output=output,
    )
    frame = pd.read_parquet(output / "valuation_fundamentals_features.parquet")
    assert receipt["status"] == "PASS"
    assert receipt["proxy_values_used"] is False
    assert frame["earnings_to_price"].isna().all()
    assert frame["fundamental_source_basis"].eq("UNKNOWN").all()
