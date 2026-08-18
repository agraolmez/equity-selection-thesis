from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_bot.research_v6.engineered_features import (
    ENGINEERED_FEATURE_NAMES,
    derive_panel_engineered_features,
)


def _row(month: str, permno: int, *, industry: str, momentum: float, cap: float) -> dict[str, object]:
    return {
        "holding_month": month,
        "signal_date": pd.Period(month, freq="M").start_time,
        "permno": permno,
        "permco": permno + 100_000,
        "momentum_6_1": 0.10,
        "momentum_12_1": momentum,
        "log_market_cap": np.log(cap),
        "log_absolute_price": np.log(20.0),
        "sic2_industry_category": industry,
        "lseg_epsmean_fy1": 2.0,
        "lseg_epsmean_fy2": 2.4,
        "lseg_revenuemean_fy1": 100.0,
        "lseg_revenuemean_fy2": 110.0,
        "lseg_pricetargetmean": 25.0,
        "lseg_recmean": 3.0,
    }


def test_panel_engineering_uses_value_weighted_industry_and_exact_calendar_lags() -> None:
    rows = [
        _row("2020-01", 1, industry="10", momentum=0.10, cap=1.0),
        _row("2020-01", 2, industry="10", momentum=0.20, cap=2.0),
        _row("2020-01", 3, industry="10", momentum=0.40, cap=1.0),
        _row("2020-02", 1, industry="10", momentum=0.15, cap=1.0),
        _row("2020-04", 1, industry="10", momentum=0.25, cap=1.0),
    ]
    rows[3]["lseg_pricetargetmean"] = 27.5
    rows[3]["lseg_recmean"] = 2.0
    rows[4]["lseg_pricetargetmean"] = 33.0
    rows[4]["lseg_recmean"] = 1.0

    result = derive_panel_engineered_features(pd.DataFrame(rows))

    january_first = result.loc[
        result["holding_month"].eq("2020-01") & result["permno"].eq(1)
    ].iloc[0]
    assert np.isclose(
        january_first["industry_momentum_12_1_vw"],
        (0.10 * 1.0 + 0.20 * 2.0 + 0.40 * 1.0) / 4.0,
    )
    february = result.loc[result["holding_month"].eq("2020-02")].iloc[0]
    assert np.isclose(february["lseg_price_target_revision_1m"], 0.10)
    assert np.isclose(february["lseg_recommendation_upgrade_1m"], 1.0)
    april = result.loc[result["holding_month"].eq("2020-04")].iloc[0]
    assert np.isnan(april["lseg_price_target_revision_1m"])
    assert np.isclose(april["lseg_price_target_revision_3m"], 0.32)
    assert np.isclose(april["lseg_recommendation_upgrade_3m"], 2.0)


def test_panel_engineering_normalizes_levels_and_decomposes_momentum() -> None:
    frame = pd.DataFrame(
        [_row("2020-01", index, industry="20", momentum=0.21, cap=float(index)) for index in (1, 2, 3)]
    )
    result = derive_panel_engineered_features(frame)
    first = result.iloc[0]

    # 12-1 return 21% and recent six-month return 10% imply an older six-month
    # return of 10%, so acceleration is zero.
    assert np.isclose(first["momentum_acceleration_6m"], 0.0)
    assert np.isclose(first["lseg_forward_eps_yield_fy1"], 0.10)
    assert np.isclose(first["lseg_forward_eps_growth_fy2_vs_fy1"], 0.20)
    assert np.isclose(first["lseg_forward_revenue_growth_fy2_vs_fy1"], 0.10)
    assert np.isclose(first["lseg_price_target_upside"], 0.25)
    assert len(ENGINEERED_FEATURE_NAMES) == 15
