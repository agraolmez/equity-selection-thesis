from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT_PATH = WORKSPACE / "scripts/run_v6_7_daily_mlp_overlay.py"
SPEC = importlib.util.spec_from_file_location("run_v6_7_daily_mlp_overlay", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def test_standardize_is_finite_and_handles_constant_features() -> None:
    train = np.asarray([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]], dtype=np.float32)
    forecast = np.asarray([[1.0, np.nan]], dtype=np.float32)
    scaled_train, scaled_forecast, receipt = study._standardize(train, forecast)
    assert np.isfinite(scaled_train).all()
    assert np.isfinite(scaled_forecast).all()
    assert receipt["constant_feature_count"] == 1


def test_equal_ensemble_uses_validation_scaling() -> None:
    keys = {
        "holding_month": ["2023-01", "2023-01"],
        "calendar_date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
        "permno": [1, 2],
    }
    mlp = pd.DataFrame({**keys, "score": [-1.0, 1.0]})
    xgb = pd.DataFrame({**keys, "score": [-2.0, 2.0]})
    ensemble, scaling = study._merge_ensemble(mlp, xgb)
    frozen, frozen_scaling = study._merge_ensemble(mlp, xgb, scaling)
    assert np.allclose(ensemble["score"], frozen["score"])
    assert scaling == frozen_scaling
    assert ensemble.loc[ensemble["permno"].eq(1), "score"].iloc[0] < 0.0


def test_paired_difference_is_challenger_minus_baseline() -> None:
    baseline = pd.DataFrame(
        {"holding_month": ["2024-01", "2024-02", "2024-03"], "net_excess_return": [0.01, 0.01, 0.01]}
    )
    challenger = pd.DataFrame(
        {"holding_month": ["2024-01", "2024-02", "2024-03"], "net_excess_return": [0.02, 0.02, 0.02]}
    )
    result = study._paired(baseline, challenger)
    assert result["annualized_difference"] == 0.12
    assert result["paired_hac_t"] > 0.0
