from __future__ import annotations

import pandas as pd

from scripts.run_v6_1_maximum_return_scaling_frontier import (
    _multiplier_grid,
    _select_tiers,
)


def test_multiplier_grid_is_complete_and_deterministic() -> None:
    config = {
        "multiplier_grid": [
            {"start": 0.5, "end": 0.6, "step": 0.05},
            {"start": 1.0, "end": 1.1, "step": 0.1},
        ]
    }
    assert _multiplier_grid(config) == [0.5, 0.55, 0.6, 1.0, 1.1]


def test_tier_selection_uses_return_metric_and_constraints() -> None:
    validation = pd.DataFrame(
        [
            {
                "profile": "constant_1.00",
                "multiplier": 1.0,
                "net_total_return_cagr": 0.10,
                "net_sharpe": 1.0,
                "annualized_stress_net_excess_return": 0.05,
                "maximum_gross": 1.4,
                "maximum_one_way_adv_participation": 0.09,
                "maximum_absolute_position": 0.015,
                "maximum_one_way_turnover": 0.50,
            },
            {
                "profile": "constant_1.20",
                "multiplier": 1.2,
                "net_total_return_cagr": 0.14,
                "net_sharpe": 0.9,
                "annualized_stress_net_excess_return": 0.04,
                "maximum_gross": 1.7,
                "maximum_one_way_adv_participation": 0.12,
                "maximum_absolute_position": 0.018,
                "maximum_one_way_turnover": 0.60,
            },
            {
                "profile": "constant_2.00",
                "multiplier": 2.0,
                "net_total_return_cagr": 0.20,
                "net_sharpe": 0.7,
                "annualized_stress_net_excess_return": -0.01,
                "maximum_gross": 2.8,
                "maximum_one_way_adv_participation": 0.25,
                "maximum_absolute_position": 0.03,
                "maximum_one_way_turnover": 1.0,
            },
        ]
    )
    config = {
        "selection": {
            "metric": "net_total_return_cagr",
            "tiers": {
                "realistic": {
                    "maximum_gross": 1.5,
                    "maximum_one_way_adv_participation": 0.10,
                    "require_positive_stress_excess_return": True,
                },
                "expanded": {
                    "maximum_gross": 2.0,
                    "maximum_one_way_adv_participation": 0.125,
                    "require_positive_stress_excess_return": True,
                },
                "absolute": {},
            },
        }
    }
    assert _select_tiers(validation, config) == {
        "realistic": "constant_1.00",
        "expanded": "constant_1.20",
        "absolute": "constant_2.00",
    }
