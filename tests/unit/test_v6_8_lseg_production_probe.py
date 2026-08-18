from __future__ import annotations

import pandas as pd

from portfolio_bot.research_v6_8.lseg_production_probe import (
    _lseg_identifier_text,
    _market_bridge_metrics,
    compare_replay,
    normalize_lseg_frame,
)


def test_normalize_lseg_frame_matches_frozen_downloader_names() -> None:
    frame = pd.DataFrame(
        {
            "Instrument": ["037833100"],
            "TR.EPSMean(Period=FY1)": [7.5],
            "TR.EPSMean(Period=FY1).CalcDate": ["2026-08-14"],
        }
    )
    result = normalize_lseg_frame(frame, "fy1")
    assert result.columns.tolist() == [
        "cusip9",
        "fy1__tr_epsmean_period_fy1",
        "fy1__tr_epsmean_period_fy1_calcdate",
    ]
    assert result.loc[0, "cusip9"] == "037833100"


def test_compare_replay_preserves_missingness_and_checks_dates() -> None:
    archive = pd.DataFrame(
        {
            "cusip9": ["A", "B", "C"],
            "signal__value": [1.0, None, 3.0],
            "signal__value_date": ["2025-06-30", None, "2025-06-30"],
        }
    )
    requery = archive.copy()
    detail, summary = compare_replay(archive, requery)
    assert summary["weighted_numeric_exact_share"] == 1.0
    assert summary["maximum_missingness_difference"] == 0.0
    assert set(detail["kind"]) == {"numeric", "date"}


def test_market_bridge_metrics_apply_frozen_rank_gate() -> None:
    reference = pd.DataFrame(
        {"permno": [1, 2, 3, 4], "feature": [1.0, 2.0, 3.0, 4.0]}
    )
    bridge = pd.DataFrame(
        {"permno": [1, 2, 3, 4], "feature": [10.0, 20.0, 30.0, 40.0]}
    )
    detail, summary = _market_bridge_metrics(reference, bridge, ["feature"])
    assert summary["features_passing_pilot_gate"] == 1
    assert detail.loc[0, "spearman_rank_correlation"] == 1.0
    assert detail.loc[0, "median_absolute_rank_difference"] == 0.0


def test_lseg_permanent_identifier_does_not_gain_float_suffix() -> None:
    assert _lseg_identifier_text(55_839_222_168) == "55839222168"
    assert _lseg_identifier_text(55_839_222_168.0) == "55839222168"
    assert _lseg_identifier_text(None) is None
