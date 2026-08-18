from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_bot.research_v6_1.portfolio import FACTOR_NAMES
from scripts import run_v6_8_daily_reoptimization as subject


def _config() -> dict:
    return {
        "reoptimization": {
            "maximum_gross": 0.5,
            "minimum_net": -0.5,
            "maximum_net": 0.5,
            "maximum_position_weight": 0.3,
            "forecast_volatility_limit": 0.2,
            "market_beta_absolute_limit": 0.2,
            "other_factor_absolute_limit": 0.2,
            "sic2_industry_net_absolute_limit": 0.5,
            "sic2_industry_gross_limit": 0.5,
            "maximum_one_way_adv_participation": 0.1,
            "solver": "CLARABEL",
            "solver_tolerance": 1e-8,
        },
        "account_and_costs": {
            "starting_nav_dollars": 100_000.0,
            "rebalance_gross_cap": 0.5,
            "financing_rate_annual": 0.0625,
            "spread_multiplier": 1.0,
            "impact_coefficient": 0.1,
            "commission_rate": 0.0001,
        },
    }


def _frame(expected: tuple[float, float] = (0.05, 0.03)) -> pd.DataFrame:
    values: dict[str, object] = {
        "permno": [1, 2],
        "sic2_industry_category": ["10", "20"],
        "adv20_usd": [10_000_000.0, 10_000_000.0],
        "full_spread_20d": [0.001, 0.001],
        "impact_volatility_daily": [0.01, 0.01],
        "primary_borrow_rate_annual": [0.01, 0.01],
        "specific_variance_monthly": [0.0001, 0.0001],
        "monthly_expected_stock_excess_return": list(expected),
    }
    for name in FACTOR_NAMES:
        values[f"beta_risk_{name}"] = [0.0, 0.0]
    return pd.DataFrame(values)


def test_survivor_refill_fixes_warned_name_and_refills_only_survivor() -> None:
    current = np.asarray([0.2, 0.2])
    result, receipt = subject._solve_survivor_refill(
        current_weights=current,
        warned=np.asarray([True, False]),
        frame=_frame(),
        covariance_root=np.zeros((len(FACTOR_NAMES), len(FACTOR_NAMES))),
        nav_dollars=100_000.0,
        refill_budget=0.1,
        pre_action_gross=0.5,
        remaining_fraction=0.5,
        config=_config(),
    )
    assert result[0] == pytest.approx(0.2, abs=2e-6)
    assert result[1] > 0.299
    assert np.abs(result - current).sum() <= 0.100002
    assert receipt["refill_gross"] > 0.099


def test_survivor_refill_leaves_cash_when_incremental_payoff_is_negative() -> None:
    current = np.asarray([0.2, 0.2])
    result, receipt = subject._solve_survivor_refill(
        current_weights=current,
        warned=np.asarray([True, False]),
        frame=_frame((0.05, -0.03)),
        covariance_root=np.zeros((len(FACTOR_NAMES), len(FACTOR_NAMES))),
        nav_dollars=100_000.0,
        refill_budget=0.1,
        pre_action_gross=0.5,
        remaining_fraction=0.5,
        config=_config(),
    )
    np.testing.assert_allclose(result, current, atol=2e-6)
    assert receipt["refill_gross"] < 2e-6


def test_month_replay_executes_saved_warning_then_reoptimizes_same_close() -> None:
    rows: list[dict[str, object]] = []
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    for day_index, date in enumerate(dates):
        for permno, expected in ((1, 0.05), (2, 0.03)):
            row: dict[str, object] = {
                "holding_month": "2024-01",
                "calendar_date": date,
                "permno": permno,
                "target_weight": 0.2,
                "stock_total_return": 0.0,
                "rf": 0.0,
                "sessions_in_month": 3,
                "day_index": day_index,
                "full_spread_20d": 0.001,
                "adv20_usd": 10_000_000.0,
                "impact_volatility_daily": 0.01,
                "primary_borrow_rate_annual": 0.01,
                "monthly_expected_stock_excess_return": expected,
                "specific_variance_monthly": 0.0001,
                "sic2_industry_category": str(permno * 10),
            }
            for name in FACTOR_NAMES:
                row[f"beta_risk_{name}"] = 0.0
            rows.append(row)
    actions = pd.DataFrame(
        [
            {
                "holding_month": "2024-01",
                "calendar_date": pd.Timestamp("2024-01-03"),
                "permno": 1,
                "action_type": "exit",
                "old_fraction": 1.0,
                "new_fraction": 0.0,
            }
        ]
    )
    monthly, daily, warnings, events, trades, state = subject._month_replay(
        month="2024-01",
        month_frame=pd.DataFrame(rows),
        action_schedule=actions,
        covariance_root=np.zeros((len(FACTOR_NAMES), len(FACTOR_NAMES))),
        state={"nav": 100_000.0, "exposure": {}, "risk_cache": {}},
        config=_config(),
        policy_id="test",
    )
    assert len(monthly) == 1
    assert len(daily) == 3
    assert len(warnings) == 1
    assert len(events) == 1
    assert events.iloc[0]["refill_gross"] > 0.099
    exit_trade = trades.loc[trades["permno"].eq(1)].iloc[0]
    survivor_trade = trades.loc[trades["permno"].eq(2)].iloc[0]
    assert exit_trade["final_weight"] == pytest.approx(0.0, abs=2e-6)
    assert survivor_trade["refill_delta"] > 0.099
    assert 1 not in state["exposure"] and "1" not in state["exposure"]
