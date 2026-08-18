from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from portfolio_bot.research_v6_8.live_lseg_analyst import (
    transform_lseg_analyst_features,
)


def test_current_lseg_transform_builds_direct_and_engineered_features() -> None:
    names = [f"f{i}" for i in range(89)]
    names[0] = "lseg_fy1_information_age_days"
    names[1] = "lseg_epsmean_fy1"
    names[2] = "lseg_epsmean_fy2"
    names[3] = "lseg_revenuemean_fy1"
    names[4] = "lseg_revenuemean_fy2"
    names[5] = "lseg_recmean"
    names[6] = "lseg_pricetargetmean"
    names[7] = "lseg_cam_country_rank"
    names[8] = "lseg_cam_rank_change_1m"
    names[9] = "lseg_cam_rank_change_3m"
    names[10] = "lseg_cam_rank_change_6m"
    sources = [f"raw_{i}" for i in range(89)]
    sources[0] = "fy1_date"
    registry = pd.DataFrame(
        {
            "model_ordinal": list(range(159, 248)) + list(range(307, 315)),
            "feature_name": names
            + [
                "lseg_forward_eps_yield_fy1",
                "lseg_forward_eps_growth_fy2_vs_fy1",
                "lseg_forward_revenue_growth_fy2_vs_fy1",
                "lseg_price_target_upside",
                "lseg_price_target_revision_1m",
                "lseg_price_target_revision_3m",
                "lseg_recommendation_upgrade_1m",
                "lseg_recommendation_upgrade_3m",
            ],
            "source_column": sources + ["engineered"] * 8,
            "transformation": ["signal_date_minus_vendor_date_days"]
            + ["numeric"] * 88
            + ["engineered"] * 8,
        }
    )
    universe = pd.DataFrame({"live_security_id": ["q1"], "RIC": ["ABC.N"]})
    raw = pd.DataFrame({"address_key": ["ABC.N"], **{name: [1.0] for name in sources}})
    raw["fy1_date"] = pd.to_datetime(["2026-07-31"])
    raw["raw_1"] = 2.0
    raw["raw_2"] = 3.0
    raw["raw_3"] = 100.0
    raw["raw_4"] = 120.0
    raw["raw_5"] = 2.0
    raw["raw_6"] = 15.0
    raw["raw_7"] = 25.0
    result = transform_lseg_analyst_features(
        universe=universe,
        raw=raw,
        usd=pd.DataFrame({"address_key": ["ABC.N"]}),
        registry=registry,
        market_features=pd.DataFrame(
            {"live_security_id": ["q1"], "signal_price_usd": [10.0]}
        ),
        signal_date=date(2026, 8, 17),
    )
    assert result.loc[0, "lseg_fy1_information_age_days"] == 17.0
    assert result.loc[0, "lseg_forward_eps_yield_fy1"] == 0.2
    assert result.loc[0, "lseg_forward_eps_growth_fy2_vs_fy1"] == 0.5
    assert result.loc[0, "lseg_price_target_upside"] == 0.5
    assert np.isnan(result.loc[0, "lseg_cam_rank_change_1m"])
