from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_v6_4_negative_alpha_selector_step_01.py"
SPEC = importlib.util.spec_from_file_location("run_v6_4_negative_alpha", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _contract() -> dict[str, object]:
    return {
        "target": {
            "source_rank_column": "target_rank_ff5_momentum",
            "source_economic_column": "alpha_ff5_momentum",
            "rank_column": "target_rank_negative_ff5_alpha",
            "economic_column": "negative_ff5_alpha_opportunity",
        }
    }


def test_negative_alpha_target_is_exact_sign_reversal() -> None:
    rows = pd.DataFrame(
        {
            "holding_month": ["2024-01"] * 5,
            "target_rank_ff5_momentum": [-0.4, -0.2, 0.0, 0.2, 0.4],
            "alpha_ff5_momentum": [-0.08, -0.03, 0.0, 0.02, 0.11],
        }
    )
    result = MODULE.attach_negative_alpha_target(rows, _contract())
    np.testing.assert_allclose(
        result["target_rank_negative_ff5_alpha"],
        -rows["target_rank_ff5_momentum"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result["negative_ff5_alpha_opportunity"],
        -rows["alpha_ff5_momentum"],
        rtol=0.0,
        atol=0.0,
    )


def test_monthly_score_reversal_preserves_month_boundaries() -> None:
    months = np.repeat(["2024-01", "2024-02"], 4)
    score = np.tile([0.1, 0.3, 0.7, 0.9], 2)
    reversed_score = MODULE.reverse_monthly_score(score, months)
    for position in (0, 4):
        np.testing.assert_allclose(
            reversed_score[position : position + 4],
            [0.875, 0.625, 0.375, 0.125],
        )


def test_tail_diagnostics_identify_negative_alpha_hits() -> None:
    frame = pd.DataFrame(
        {
            "holding_month": ["2024-01"] * 20,
            "permno": np.arange(20),
            "score": np.arange(20, dtype=float),
            "alpha_ff5_momentum": np.linspace(0.10, -0.09, 20),
        }
    )
    monthly, summary = MODULE._tail_diagnostics(
        frame,
        score_columns=["score"],
        alpha_column="alpha_ff5_momentum",
        periods={"historical_test": ("2024-01", "2024-01")},
        tail_fractions=[0.10],
    )
    assert len(monthly) == 1
    assert monthly.loc[0, "negative_alpha_fraction"] == 1.0
    assert summary.loc[0, "mean_negative_alpha_fraction"] == 1.0
