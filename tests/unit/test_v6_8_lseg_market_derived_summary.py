from __future__ import annotations

import json

import pandas as pd

from portfolio_bot.research_v6_8.lseg_market_derived_summary import (
    summarize_lseg_market_derived_parity,
)


def test_feature_summary_applies_the_prespecified_month_share(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "features": ["good", "bad"],
                "full_window_gate": {"minimum_passing_month_share": 0.5},
                "excluded_cross_sectional_features": ["pending"],
            }
        ),
        encoding="utf-8",
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "receipt.json").write_text(
        json.dumps(
            {"scope": "FULL_WINDOW", "months_expected": 2, "months_completed": 2}
        ),
        encoding="utf-8",
    )
    for month, good, bad in [("2024-01", True, False), ("2024-02", True, True)]:
        target = run / "months" / month
        target.mkdir(parents=True)
        pd.DataFrame(
            {
                "feature": ["good", "bad"],
                "pilot_gate_pass": [good, bad],
                "spearman_rank_correlation": [1.0, 0.8],
                "absolute_missingness_difference": [0.0, 0.1],
                "median_absolute_rank_difference": [0.0, 0.1],
                "median_absolute_relative_difference": [0.0, 0.2],
            }
        ).to_csv(target / "feature_metrics.csv", index=False)
    result = summarize_lseg_market_derived_parity(
        config_path=config, run_directory=run
    )
    assert result["features_passing_feature_level_gate"] == 2
    assert result["failing_features"] == []
    assert (run / "feature_window_summary.csv").exists()
