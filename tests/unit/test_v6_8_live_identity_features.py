from __future__ import annotations

import json

import pandas as pd

from portfolio_bot.research_v6_8.live_identity_features import build_identity_features


def _config() -> dict:
    return {
        "exchange_mapping": {
            "NYSE Consolidated": "N",
            "NASDAQ Global Select Consolidated": "Q",
        }
    }


def test_identity_bridge_uses_unambiguous_sec_sic_and_fundamental_currency(tmp_path) -> None:
    universe = pd.DataFrame(
        {
            "live_security_id": ["1", "2"],
            "IssuerCikNumber": ["0000320193", "0001|0002"],
            "IssuerCountry": ["USA", "NLD"],
            "ExchangeName": ["NASDAQ Global Select Consolidated", "Other"],
            "adr_indicator": [0, 1],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "live_security_id": ["1", "2"],
            "fundamental_fic": ["USA", "NLD"],
            "fundamental_loc": ["USA", "USA"],
            "reporting_currency": ["USD", "EUR"],
        }
    )
    root = tmp_path / "submissions"
    root.mkdir()
    (root / "CIK0000320193.json").write_text(json.dumps({"sic": "3571"}))
    result = build_identity_features(
        universe,
        config=_config(),
        fundamentals=fundamentals,
        sec_submissions=root,
    ).set_index("live_security_id")
    assert result.loc["1", "sic2_industry_category"] == "35"
    assert result.loc["1", "exchange_category"] == "Q"
    assert result.loc["1", "us_incorporation_indicator"] == 1
    assert result.loc["2", "sic2_industry_category"] == "UNKNOWN_INDUSTRY"
    assert result.loc["2", "adr_indicator"] == 1
    assert result.loc["2", "foreign_listing_indicator"] == 1
    assert result.loc["2", "reporting_currency_category"] == "EUR"
