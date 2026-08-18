from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_bot.research_v6_8.lseg_market_parity import (
    _bridge_values,
    _identity,
    _monthly_pass,
    _months,
)


def test_months_are_inclusive() -> None:
    assert _months("2024-11", "2025-02") == [
        "2024-11",
        "2024-12",
        "2025-01",
        "2025-02",
    ]


def test_bridge_values_use_total_return_percent_and_consolidated_volume() -> None:
    rows = 252
    frame = pd.DataFrame(
        {
            "Total Return": [1.0] * rows,
            "Price Close": [10.0] * rows,
            "Accumulated Volume": [100.0] * rows,
        }
    )
    frame.attrs["unadjusted_signal_price"] = 40.0
    values = _bridge_values(1, frame)
    assert values["permno"] == 1
    assert values["median_dollar_volume_60d"] == 1000.0
    assert values["log_absolute_price"] == pytest.approx(np.log(40.0))
    assert values["return_total_5d"] == pytest.approx(1.01**5 - 1.0)
    assert values["realized_volatility_21d"] == pytest.approx(0.0, abs=1e-15)


def test_monthly_gate_requires_resolution_and_all_feature_metrics() -> None:
    detail = pd.DataFrame(
        {
            "spearman_rank_correlation": [0.99, 0.98],
            "median_absolute_rank_difference": [0.01, 0.02],
            "absolute_missingness_difference": [0.0, 0.01],
        }
    )
    gate = {
        "minimum_resolved_instrument_share": 0.95,
        "minimum_spearman_rank_correlation": 0.95,
        "maximum_median_absolute_rank_difference": 0.05,
        "maximum_absolute_missingness_difference": 0.02,
    }
    assert _monthly_pass(detail, resolved_share=0.95, gate=gate)
    assert not _monthly_pass(detail, resolved_share=0.90, gate=gate)
    detail.loc[1, "spearman_rank_correlation"] = 0.80
    assert not _monthly_pass(detail, resolved_share=1.00, gate=gate)


def test_bridge_values_preserve_short_history_as_feature_missingness() -> None:
    frame = pd.DataFrame(
        {
            "Total Return": [0.5] * 63,
            "Price Close": [20.0] * 63,
            "Accumulated Volume": [200.0] * 63,
        }
    )
    values = _bridge_values(2, frame)
    assert pd.notna(values["return_total_63d"])
    assert pd.isna(values["return_total_126d"])
    assert pd.notna(values["realized_volatility_63d"])
    assert pd.isna(values["realized_volatility_126d"])


def test_checkpoint_identity_binds_imported_resolver(tmp_path) -> None:
    paths = [tmp_path / name for name in ("config.json", "archive.parquet", "model.parquet")]
    for index, path in enumerate(paths):
        path.write_text(str(index), encoding="utf-8")
    identity = _identity(
        config_path=paths[0],
        archive_month=paths[1],
        model_feature_month=paths[2],
    )
    assert len(identity["code_sha256"]) == 64
    assert len(identity["resolver_code_sha256"]) == 64
