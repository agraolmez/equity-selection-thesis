import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence/post_thesis_v6_8_replay"


def _inputs() -> tuple[pd.DataFrame, dict[str, object]]:
    monthly = pd.read_csv(EVIDENCE / "monthly_returns.csv")
    summary = json.loads((EVIDENCE / "summary.json").read_text("utf-8"))
    return monthly, summary


def test_monthly_series_is_complete_unique_and_status_bound() -> None:
    monthly, _ = _inputs()
    expected = pd.period_range("2024-01", "2026-07", freq="M").astype(str)
    assert monthly["holding_month"].tolist() == expected.tolist()
    assert len(monthly) == 31
    assert monthly["holding_month"].is_unique
    assert not monthly[["net_return", "net_excess_return"]].isna().any().any()
    assert monthly["evidence_status"].value_counts().to_dict() == {
        "historical_development": 20,
        "retrospective_operational_diagnostic": 11,
    }


def test_headline_metrics_recompute_from_public_monthly_returns() -> None:
    monthly, summary = _inputs()
    wealth = np.cumprod(1.0 + monthly["net_return"].to_numpy(float))
    cumulative_return = wealth[-1] - 1.0
    cagr = wealth[-1] ** (12.0 / len(monthly)) - 1.0
    annualized_excess = monthly["net_excess_return"].mean() * 12.0
    annualized_volatility = (
        monthly["net_excess_return"].std(ddof=1) * np.sqrt(12.0)
    )
    sharpe = annualized_excess / annualized_volatility

    assert np.isclose(cumulative_return, summary["cumulative_net_total_return"])
    assert np.isclose(cagr, summary["total_return_cagr"])
    assert np.isclose(annualized_excess, summary["annualized_net_excess_return"])
    assert np.isclose(
        annualized_volatility,
        summary["annualized_excess_volatility"],
    )
    assert np.isclose(sharpe, summary["net_sharpe"])
    assert np.isclose(100_000.0 * wealth[-1], summary["wealth_100k"])
    assert summary["status"] == (
        "PASS_DIAGNOSTIC_STITCHED_NOT_THESIS_OR_LIVE_EVIDENCE"
    )
