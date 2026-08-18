from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_bot.research_v6_8.live_risk_cost import build_live_risk_cost
from portfolio_bot.risk_cost.risk_math import FACTOR_NAMES


def test_live_risk_cost_builds_complete_frozen_inputs() -> None:
    rng = np.random.default_rng(7)
    sessions = pd.bdate_range("2025-08-01", periods=252)
    factor_values = rng.normal(0.0, 0.01, size=(len(sessions), len(FACTOR_NAMES)))
    factors = pd.DataFrame(factor_values, columns=FACTOR_NAMES)
    factors.insert(0, "observation_date", sessions)
    factors["rf"] = 0.0001
    market_rows = []
    history_rows = []
    identity_rows = []
    for security in range(12):
        security_id = f"s{security}"
        beta = np.linspace(0.2, 0.7, len(FACTOR_NAMES)) + security * 0.01
        returns = factor_values @ beta + rng.normal(0.0, 0.005, len(sessions)) + 0.0001
        market_rows.append(
            {
                "live_security_id": security_id,
                "symbol": f"S{security}",
                "long_eligible": True,
                "signal_price_usd": 20.0,
                "issue_market_cap_usd": 2_000_000_000.0,
                "median_dollar_volume_60d": 25_000_000.0,
                "zero_volume_share_60d": 0.01,
                "median_estimated_full_spread_20d": 0.002,
                "median_dollar_volume_20d": 20_000_000.0 + security,
            }
        )
        identity_rows.append(
            {"live_security_id": security_id, "sic2_industry_category": "35"}
        )
        history_rows.extend(
            {
                "live_security_id": security_id,
                "observation_date": day,
                "total_return": value,
            }
            for day, value in zip(sessions, returns, strict=True)
        )
    config = {
        "risk": {
            "lookback_global_sessions": 252,
            "minimum_observations": 126,
            "half_life_sessions": 63.0,
            "sector_minimum_securities": 10,
            "specific_variance_prior_observations": 60.0,
            "monthly_session_scale": 21.0,
        },
        "short_eligibility": {
            "minimum_price_usd": 10.0,
            "minimum_market_cap_usd": 1_000_000_000.0,
            "minimum_median_dollar_volume_60d": 20_000_000.0,
            "minimum_positive_volume_share_60d": 0.95,
        },
    }
    risk, covariance = build_live_risk_cost(
        market=pd.DataFrame(market_rows),
        history=pd.DataFrame(history_rows),
        identity=pd.DataFrame(identity_rows),
        daily_factors=factors,
        signal_date=sessions[-1],
        config=config,
    )
    assert len(risk) == 12
    assert risk["new_long_trade_allowed"].all()
    assert risk["new_short_trade_allowed"].all()
    assert risk["specific_variance_monthly"].gt(0).all()
    assert risk["primary_borrow_rate_annual"].eq(0.01).all()
    assert len(covariance) == 36
    assert covariance["covariance_monthly"].notna().all()
