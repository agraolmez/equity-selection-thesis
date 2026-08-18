from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_bot.research_v6_1.portfolio_step_03 import (
    causal_direct_rank_payoff,
    causal_shrunk_linear_calibration,
    stressed_round_trip_hurdle,
)


def _history() -> pd.DataFrame:
    rows = []
    for month in pd.period_range("2020-01", "2021-12", freq="M").astype(str):
        for score in (0.1, 0.3, 0.5, 0.7, 0.9):
            rows.append(
                {
                    "month": month,
                    "score": score,
                    "outcome": score - 0.45,
                    "available": pd.Timestamp("2022-01-15"),
                }
            )
    rows.append(
        {
            "month": "2022-01",
            "score": 0.9,
            "outcome": -100.0,
            "available": pd.Timestamp("2023-01-01"),
        }
    )
    return pd.DataFrame(rows)


def test_step03_linear_and_direct_mappings_are_causal_and_ordered() -> None:
    common = {
        "score_column": "score",
        "outcome_column": "outcome",
        "month_column": "month",
        "availability_column": "available",
        "signal_date": pd.Timestamp("2022-02-01"),
        "minimum_history_months": 24,
        "shrinkage_prior_months": 24.0,
    }
    score = np.asarray([0.1, 0.5, 0.9])
    linear = causal_shrunk_linear_calibration(_history(), score, **common)
    direct = causal_direct_rank_payoff(_history(), score, threshold=0.8, **common)
    assert linear.history_months == 24
    assert np.all(np.diff(linear.values) >= 0.0)
    assert direct.values[0] == 0.0
    assert direct.values[1] == 0.0
    assert direct.values[2] > 0.0


def test_step03_stressed_cost_hurdle_adds_borrow_only_to_shorts() -> None:
    common = {
        "full_spread": np.asarray([0.002]),
        "adv20_usd": np.asarray([100_000_000.0]),
        "impact_volatility_daily": np.asarray([0.02]),
        "stress_borrow_rate_annual": np.asarray([0.12]),
        "nav_dollars": 100_000_000.0,
        "representative_position_weight": 0.001,
        "spread_multiplier": 2.0,
        "impact_coefficient": 0.20,
        "commission_each_way": 0.0001,
    }
    long = stressed_round_trip_hurdle(include_borrow=False, **common)
    short = stressed_round_trip_hurdle(include_borrow=True, **common)
    assert np.isclose(short[0] - long[0], 0.01)
