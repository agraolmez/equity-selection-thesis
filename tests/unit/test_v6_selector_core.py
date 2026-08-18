from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_bot.research_v6.data import _attach_net_values
from portfolio_bot.research_v6.models import dynamic_ensemble, rank_within_month
from portfolio_bot.research_v6.selector_runtime import (
    is_refit_month,
    is_retune_month,
    walk_forward_predictions,
)


def test_rank_within_month_is_separate_and_monotonic() -> None:
    score = np.asarray([3.0, 1.0, 2.0, 20.0, 10.0])
    month = ["2020-01"] * 3 + ["2020-02"] * 2
    rank = rank_within_month(score, month)
    assert np.allclose(rank[:3], [5 / 6, 1 / 6, 3 / 6])
    assert np.allclose(rank[3:], [0.75, 0.25])


def test_dynamic_ensemble_never_uses_current_month_outcome() -> None:
    rows = []
    for month_index in range(25):
        month = f"202{month_index // 12}-{month_index % 12 + 1:02d}"
        for permno in range(10):
            rows.append(
                {
                    "holding_month": month,
                    "permno": permno,
                    "xgboost_rank": (permno + 0.5) / 10,
                    "mlp_rank": (9 - permno + 0.5) / 10,
                    "target": permno,
                }
            )
    frame = pd.DataFrame(rows)
    original = dynamic_ensemble(frame, target_column="target")
    changed = frame.copy()
    changed.loc[changed["holding_month"].eq("2022-01"), "target"] *= -1
    rerun = dynamic_ensemble(changed, target_column="target")
    before = original.loc[original["holding_month"].eq("2022-01"), "xgboost_weight"].unique()
    after = rerun.loc[rerun["holding_month"].eq("2022-01"), "xgboost_weight"].unique()
    assert np.array_equal(before, after)


def test_dynamic_ensemble_uses_meaningful_bounded_performance_weights() -> None:
    rows = []
    months = pd.period_range("2020-01", periods=25, freq="M").astype(str)
    for month in months:
        for permno in range(10):
            rows.append(
                {
                    "holding_month": month,
                    "permno": permno,
                    "xgboost_rank": (permno + 0.5) / 10,
                    "mlp_rank": (9 - permno + 0.5) / 10,
                    "target": permno,
                }
            )
    result = dynamic_ensemble(pd.DataFrame(rows), target_column="target")
    final = result.loc[result["holding_month"].eq(months[-1])]
    assert final["xgboost_weight"].nunique() == 1
    assert final["xgboost_weight"].iloc[0] == 0.8
    assert final["mlp_weight"].iloc[0] == pytest.approx(0.2)


def test_refit_and_retune_schedules_are_prespecified() -> None:
    assert is_refit_month("2024-01", "annual")
    assert not is_refit_month("2024-04", "annual")
    assert is_refit_month("2024-04", "quarterly")
    assert is_retune_month("2024-01", "annual")
    assert is_retune_month("2024-01", "biennial")
    assert not is_retune_month("2025-01", "biennial")


def test_walk_forward_elastic_checkpoint_resumes(tmp_path) -> None:
    rng = np.random.default_rng(11)
    months = pd.period_range("2017-01", "2020-03", freq="M")
    rows = []
    values = []
    for month in months:
        signal = (month - 1).to_timestamp(how="end").normalize()
        available = month.to_timestamp(how="end").normalize()
        for permno in range(12):
            first = rng.normal()
            second = rng.normal()
            rows.append(
                {
                    "holding_month": str(month),
                    "signal_date": signal,
                    "permno": 10_000 + permno,
                    "permco": 20_000 + permno,
                    "label_available_date": available,
                    "target": first + 0.1 * second,
                }
            )
            values.append([first, second])
    frame = pd.DataFrame(rows)
    matrix = np.asarray(values, dtype=np.float32)
    parameters = dict(
        matrix=matrix,
        rows=frame,
        feature_index=np.asarray([0, 1]),
        target_column="target",
        family="elastic_net",
        objective="continuous_rank",
        candidate={
            "candidate_id": "annual_expanding",
            "refit": "annual",
            "history": "expanding",
            "retune": "annual",
        },
        profiles=[
            {"profile_id": "simple", "alpha": 0.01, "l1_ratio": 0.1},
            {"profile_id": "sparse", "alpha": 0.01, "l1_ratio": 0.9},
        ],
        shared={},
        start_month="2020-01",
        end_month="2020-03",
        seeds=[42],
        tuning={
            "holdout_months": 12,
            "tuning_seed": 42,
            "profile_tie_tolerance": 0.002,
        },
        output=tmp_path,
        run_identity="synthetic-v6-test",
    )
    first, _, _ = walk_forward_predictions(**parameters)
    second, _, _ = walk_forward_predictions(**parameters)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    assert (tmp_path / "checkpoints/2020-01.json").is_file()


def test_net_cost_join_normalizes_signal_date_type() -> None:
    outcome = pd.DataFrame(
        {
            "holding_month": ["2019-12"],
            "signal_date": [pd.Timestamp("2019-11-29")],
            "permno": [10_001],
            "permco": [20_001],
            "long_eligible": [True],
            "short_eligible": [True],
            "unranked_excess_return": [0.02],
        }
    )
    costs = pd.DataFrame(
        {
            "holding_month": ["2019-12"],
            "signal_date": [pd.Timestamp("2019-11-29").date()],
            "permno": [10_001],
            "permco": [20_001],
            "full_spread_20d": [0.001],
            "adv20_usd": [1_000_000_000.0],
            "impact_volatility_daily": [0.02],
            "primary_borrow_rate_annual": [0.03],
            "new_long_trade_allowed": [True],
            "new_short_trade_allowed": [True],
        }
    )

    result = _attach_net_values(outcome, costs)

    assert len(result) == 1
    assert np.isfinite(result.loc[0, "long_net_opportunity"])
    assert np.isfinite(result.loc[0, "short_net_opportunity"])
