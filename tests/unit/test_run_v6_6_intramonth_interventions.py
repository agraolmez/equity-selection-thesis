from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT_PATH = WORKSPACE / "scripts/run_v6_6_intramonth_interventions.py"
SPEC = importlib.util.spec_from_file_location("run_v6_6_intramonth_interventions", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def test_transaction_cost_is_zero_without_a_trade_and_positive_with_one() -> None:
    risk = pd.DataFrame(
        {
            "full_spread_20d": [0.002],
            "adv20_usd": [10_000_000.0],
            "impact_volatility_daily": [0.02],
        }
    )
    costs = {
        "spread_multiplier": 1.0,
        "commission_rate": 0.0001,
        "impact_coefficient": 0.1,
    }
    assert study._transaction_cost(np.array([0.0]), risk, 100_000.0, costs) == 0.0
    assert study._transaction_cost(np.array([0.1]), risk, 100_000.0, costs) > 0.0


def test_stop_rule_can_exit_after_a_prior_profit_trim_but_never_add() -> None:
    rule = {
        "take_profit": 0.20,
        "take_profit_fraction": 0.50,
        "stop_loss": -0.10,
        "stop_loss_fraction": 0.0,
    }
    assert study._rule_target(pd.Series({"signed_position_return": 0.25, "signed_peak_to_current_drawdown": 0.0}), rule, 1.0) == 0.5
    assert study._rule_target(pd.Series({"signed_position_return": 0.25, "signed_peak_to_current_drawdown": 0.0}), rule, 0.4) == 0.4
    assert study._rule_target(pd.Series({"signed_position_return": -0.12, "signed_peak_to_current_drawdown": -0.2}), rule, 0.5) == 0.0


def test_hac_t_has_the_sign_of_a_constant_nonzero_difference() -> None:
    assert study._hac_t([0.01] * 12) > 0.0
    assert study._hac_t([-0.01] * 12) < 0.0
