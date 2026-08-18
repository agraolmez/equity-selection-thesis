from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_bot.research_v6_8.live_market_features import (
    LiveMarketFeatureError,
    _daily_history_frame,
    apply_frozen_eligibility,
    apply_market_feature_policy,
)


def test_market_policy_preserves_one_row_per_live_security() -> None:
    frame = pd.DataFrame(
        {"live_security_id": ["a", "b"], "return_total_5d": [0.01, -0.02]}
    )
    result = apply_market_feature_policy(
        frame,
        passing_features=["return_total_5d"],
        provider_missing_features=[],
        pending_cross_sectional_features=[],
    )
    assert result["live_security_id"].tolist() == ["a", "b"]


def test_daily_history_retains_decimal_total_return_at_security_session_grain() -> None:
    history = pd.DataFrame(
        {
            "Price Close": [10.0, 10.2],
            "Total Return": [1.0, -0.5],
            "Accumulated Volume": [100.0, 120.0],
        },
        index=pd.to_datetime(["2026-08-13", "2026-08-14"]),
    )
    result = _daily_history_frame(
        history, live_security_id="q1", symbol="ABC", ric="ABC.N"
    )
    assert result["total_return"].tolist() == [0.01, -0.005]
    assert result["live_security_id"].tolist() == ["q1", "q1"]


def test_failed_and_pending_market_features_are_explicitly_missing() -> None:
    frame = pd.DataFrame({"live_security_id": ["a"], "return_total_5d": [0.01]})
    result = apply_market_feature_policy(
        frame,
        passing_features=["return_total_5d"],
        provider_missing_features=["log_market_cap"],
        pending_cross_sectional_features=["industry_momentum_12_1_vw"],
    )
    assert np.isnan(result.loc[0, "log_market_cap"])
    assert np.isnan(result.loc[0, "industry_momentum_12_1_vw"])


def test_feature_dispositions_must_not_overlap() -> None:
    with pytest.raises(LiveMarketFeatureError, match="overlaps"):
        apply_market_feature_policy(
            pd.DataFrame({"a": [1.0]}),
            passing_features=["a"],
            provider_missing_features=["a"],
            pending_cross_sectional_features=[],
        )


def test_frozen_eligibility_uses_all_five_economic_rules() -> None:
    rows = pd.DataFrame(
        {
            "signal_price_usd": [5.0, 4.99],
            "issue_market_cap_usd": [300_000_000.0, 1_000_000_000.0],
            "median_dollar_volume_60d": [5_000_000.0, 10_000_000.0],
            "zero_volume_share_60d": [0.10, 0.0],
            "usable_daily_returns": [126, 252],
        }
    )
    result = apply_frozen_eligibility(
        rows,
        minimum_price_usd=5.0,
        minimum_market_cap_usd=300_000_000.0,
        minimum_median_dollar_volume_60d=5_000_000.0,
        minimum_positive_volume_share_60d=0.9,
        minimum_usable_daily_returns=126,
    )
    assert result.tolist() == [True, False]
