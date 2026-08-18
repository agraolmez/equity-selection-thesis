from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_bot.research_v6_1.feature_views import (
    direct_cam_predictions,
    prepare_feature_views,
)
from portfolio_bot.research_v6_1.models import add_dynamic_ensemble


def _feature_fixture() -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(61)
    rows = pd.DataFrame(
        {
            "holding_month": np.repeat(["2018-01", "2018-02"], 20),
            "signal_date": np.repeat(
                [pd.Timestamp("2017-12-29"), pd.Timestamp("2018-01-31")], 20
            ),
            "permno": np.tile(np.arange(20), 2),
            "permco": np.tile(np.arange(20), 2),
        }
    )
    first = rng.normal(size=40).astype(np.float32)
    second = (first * 2.0).astype(np.float32)
    third = rng.normal(size=40).astype(np.float32)
    cam = np.tile(np.linspace(-0.49, 0.49, 20), 2).astype(np.float32)
    matrix = np.column_stack(
        [
            first,
            np.zeros(40),
            second,
            np.zeros(40),
            third,
            np.zeros(40),
            cam,
            np.zeros(40),
            np.tile([1.0, 0.0], 20),
            np.tile([0.0, 1.0], 20),
        ]
    ).astype(np.float32)
    metadata: dict[str, object] = {
        "feature_names": [
            "first",
            "first__missing",
            "second",
            "second__missing",
            "third",
            "third__missing",
            "lseg_cam_country_rank",
            "lseg_cam_country_rank__missing",
            "exchange__nyse",
            "exchange__other",
        ],
        "source_features": [
            "first",
            "first",
            "second",
            "second",
            "third",
            "third",
            "lseg_cam_country_rank",
            "lseg_cam_country_rank",
            "exchange",
            "exchange",
        ],
        "feature_family": {
            "first": "price",
            "second": "price",
            "third": "fundamental",
            "lseg_cam_country_rank": "lseg",
            "exchange": "classification",
        },
    }
    return matrix, rows, metadata


def test_correlation_reduction_is_outcome_blind_and_moves_source_columns_together(
    tmp_path,
) -> None:
    matrix, rows, metadata = _feature_fixture()
    config = {
        "feature_views": {
            "direct_cam_sources": ["lseg_cam_country_rank"],
            "correlation_reduction": {
                "training_end": "2019-12",
                "absolute_rank_correlation_threshold": 0.90,
                "minimum_pair_observations": 10,
                "priority": [
                    "observed_fraction_desc",
                    "registry_ordinal_asc",
                    "feature_name_asc",
                ],
                "outcomes_used": False,
                "categorical_sources_pruned": False,
                "encoded_columns_move_with_source": True,
            },
        }
    }
    manifest = prepare_feature_views(
        matrix=matrix,
        rows=rows,
        metadata=metadata,
        config=config,
        input_identity="synthetic",
        output=tmp_path,
    )
    reduced = manifest["views"]["correlation_reduced"]["raw_sources"]
    assert "first" in reduced
    assert "second" not in reduced
    assert "third" in reduced
    assert "exchange" in reduced
    assert manifest["outcomes_used"] is False
    indices = manifest["views"]["correlation_reduced"]["encoded_indices"]
    assert 0 in indices and 1 in indices


def test_direct_cam_is_unmodeled_monthly_rank() -> None:
    matrix, rows, metadata = _feature_fixture()
    result = direct_cam_predictions(
        matrix=matrix,
        rows=rows,
        metadata=metadata,
        source="lseg_cam_country_rank",
        start_month="2018-01",
        end_month="2018-02",
    )
    for _, month in result.groupby("holding_month", observed=True):
        assert month.sort_values("raw_score")["score_rank"].is_monotonic_increasing
        assert month["score_rank"].min() == pytest.approx(0.025)
        assert month["score_rank"].max() == pytest.approx(0.975)


def test_multimodel_dynamic_ensemble_is_causal_and_bounded() -> None:
    records = []
    months = pd.period_range("2020-01", periods=26, freq="M").astype(str)
    for month in months:
        for stock in range(20):
            records.append(
                {
                    "holding_month": month,
                    "permno": stock,
                    "good": (stock + 0.5) / 20,
                    "bad": (19 - stock + 0.5) / 20,
                    "flat": 0.5,
                    "target": stock,
                }
            )
    frame = pd.DataFrame(records)
    first = add_dynamic_ensemble(
        frame,
        components=["good", "bad", "flat"],
        target_column="target",
        output_column="blend",
        half_life_months=12,
        minimum_history_months=24,
        minimum_component_weight=0.05,
        maximum_component_weight=0.70,
    )
    changed = frame.copy()
    changed.loc[changed["holding_month"].eq(months[-1]), "target"] *= -1
    second = add_dynamic_ensemble(
        changed,
        components=["good", "bad", "flat"],
        target_column="target",
        output_column="blend",
        half_life_months=12,
        minimum_history_months=24,
        minimum_component_weight=0.05,
        maximum_component_weight=0.70,
    )
    weight_columns = [column for column in first if "__weight__" in column]
    current_first = first.loc[first["holding_month"].eq(months[-1]), weight_columns]
    current_second = second.loc[second["holding_month"].eq(months[-1]), weight_columns]
    np.testing.assert_allclose(current_first, current_second)
    weights = current_first.iloc[0].to_numpy(float)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= 0.05
    assert weights.max() <= 0.70
    assert weights[0] == pytest.approx(0.70)
