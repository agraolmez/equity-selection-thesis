from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_bot.research_v6_3.portfolio import (
    causal_opportunity_percentiles,
    latent_residual_risk,
    opportunity_multiplier,
)
from portfolio_bot.research_v6_3.signals import (
    add_confidence_diagnostics,
    build_three_month_target,
    feature_expert_indices,
)


def test_confidence_diagnostics_are_outcome_free_and_bounded() -> None:
    frame = pd.DataFrame(
        {
            "holding_month": ["2020-01", "2020-01", "2020-02", "2020-02"],
            "permno": [1, 2, 1, 2],
            "base": [0.2, 0.8, 0.4, 0.7],
            "left": [0.1, 0.9, 0.2, 0.6],
            "right": [0.3, 0.7, 0.6, 0.8],
        }
    )
    result = add_confidence_diagnostics(
        frame,
        base_column="base",
        component_columns=["left", "right"],
        prefix="signal",
    )
    assert result["signal_uncertainty"].between(0.0, 1.0).all()
    assert result.loc[result["holding_month"].eq("2020-01"), "signal_instability"].isna().all()
    assert result.loc[result["holding_month"].eq("2020-02"), "signal_instability"].notna().all()


def test_three_month_target_requires_consecutive_future_months() -> None:
    records: list[dict[str, object]] = []
    returns = {1: [0.10, 0.00, -0.05, 0.02], 2: [0.00, 0.01, 0.02, 0.03]}
    for permno, values in returns.items():
        for period, value in zip(pd.period_range("2020-01", periods=4, freq="M"), values):
            records.append(
                {
                    "holding_month": str(period),
                    "permno": permno,
                    "unranked_excess_return": value,
                    "label_available_date": period.end_time.normalize(),
                }
            )
    result = build_three_month_target(pd.DataFrame(records))
    january = result.loc[result["holding_month"].eq("2020-01")].set_index("permno")
    assert np.isclose(january.loc[1, "target_return_3m"], 1.10 * 1.00 * 0.95 - 1.0)
    assert january.loc[1, "target_available_date_3m"] == pd.Timestamp("2020-03-31")
    assert january.loc[1, "target_rank_return_3m"] > january.loc[2, "target_rank_return_3m"]
    assert result.loc[result["holding_month"].eq("2020-03"), "target_return_3m"].isna().all()


def test_expert_partition_covers_every_encoded_feature_once() -> None:
    metadata = {
        "source_features": ["price", "analyst", "news", "book"],
        "feature_family": {
            "price": "price_trend",
            "analyst": "analyst_estimates",
            "news": "news",
            "book": "fundamentals",
        },
    }
    experts = {
        "market": ["price_trend"],
        "analyst": ["analyst_estimates"],
        "alternative": ["news"],
        "fundamental": ["fundamentals"],
    }
    partition = feature_expert_indices(metadata, experts)
    covered = np.concatenate(list(partition.values()))
    assert sorted(covered.tolist()) == [0, 1, 2, 3]


def test_opportunity_scaling_is_causal_and_bounded() -> None:
    scores = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    percentile = causal_opportunity_percentiles(scores, minimum_history_months=1)
    assert percentile.loc["a"] == 0.5
    assert percentile.loc["b"] == 1.0
    assert percentile.loc["c"] == 1.0
    assert opportunity_multiplier(1.0, strength=1.5, minimum=0.65, maximum=1.35) == 1.35
    assert opportunity_multiplier(0.0, strength=1.5, minimum=0.65, maximum=1.35) == 0.65


def test_latent_risk_adds_only_trailing_residual_factors() -> None:
    factor_names = ["mktrf", "smb", "hml", "rmw", "cma", "mom"]
    factor_columns = [f"beta_risk_{name}" for name in factor_names]
    permnos = np.arange(1, 7)
    months = pd.period_range("2019-01", periods=24, freq="M")
    rows: list[dict[str, object]] = []
    for month_number, month in enumerate(months):
        for permno in permnos:
            row: dict[str, object] = {
                "holding_month": str(month),
                "permno": int(permno),
                "stock_excess_return": 0.001 * month_number + 0.002 * permno,
                "label_available_date": month.end_time.normalize(),
            }
            for position, name in enumerate(factor_names):
                row[name] = 0.001 * (position + 1)
                row[f"beta_risk_{name}"] = 0.1 * (position + 1)
            rows.append(row)
    current = pd.DataFrame(
        {
            "holding_month": "2021-01",
            "signal_date": pd.Timestamp("2021-01-31"),
            "permno": permnos,
            "specific_variance_monthly": 0.02,
            **{column: 0.1 for column in factor_columns},
        }
    )
    result = latent_residual_risk(
        current,
        pd.DataFrame(rows),
        np.eye(6) * 0.02,
        factor_columns=factor_columns,
        latent_factors=3,
        latent_variance_share=0.5,
        trailing_months=24,
        minimum_months=12,
        specific_variance_floor_fraction=0.25,
    )
    assert result.observations == 24
    assert result.factor_betas.shape == (6, 9)
    assert result.covariance_root.shape == (9, 9)
    assert np.all(result.specific_variance_monthly >= 0.005)
