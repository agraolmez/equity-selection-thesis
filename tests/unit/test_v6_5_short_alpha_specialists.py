from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_v6_5_short_alpha_specialists_step_01 import centered_downside_rank


def test_downside_target_focuses_on_negative_alpha_tail() -> None:
    frame = pd.DataFrame(
        {
            "holding_month": ["2020-01"] * 5,
            "alpha": [-0.10, -0.02, 0.00, 0.03, 0.10],
        }
    )
    observed = centered_downside_rank(frame["alpha"], frame["holding_month"])
    np.testing.assert_allclose(
        observed,
        np.asarray([0.4, 0.2, -0.2, -0.2, -0.2]),
        rtol=0.0,
        atol=1e-15,
    )


def test_downside_target_is_not_full_rank_reversal() -> None:
    frame = pd.DataFrame(
        {
            "holding_month": ["2020-01"] * 5,
            "alpha": [-0.10, -0.02, 0.00, 0.03, 0.10],
        }
    )
    observed = centered_downside_rank(frame["alpha"], frame["holding_month"])
    reversed_full_rank = np.asarray([0.4, 0.2, 0.0, -0.2, -0.4])
    assert not np.allclose(observed, reversed_full_rank)


def test_downside_target_is_month_local() -> None:
    frame = pd.DataFrame(
        {
            "holding_month": ["2020-01"] * 3 + ["2020-02"] * 3,
            "alpha": [-0.03, 0.01, 0.02, -0.30, 0.10, 0.20],
        }
    )
    observed = centered_downside_rank(frame["alpha"], frame["holding_month"])
    np.testing.assert_allclose(observed[:3], observed[3:], rtol=0.0, atol=1e-15)


def test_downside_target_preserves_missing_labels() -> None:
    frame = pd.DataFrame(
        {
            "holding_month": ["2020-01"] * 4,
            "alpha": [-0.03, 0.01, np.nan, 0.02],
        }
    )
    observed = centered_downside_rank(frame["alpha"], frame["holding_month"])
    assert np.isnan(observed[2])
    assert np.isfinite(observed[[0, 1, 3]]).all()
