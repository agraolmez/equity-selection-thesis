from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_bot.research_v6_8.lseg_market_derived_parity import (
    _batch_member,
    _derived_bridge_values,
)


def test_derived_market_formulas_use_capitalization_equivalent_turnover() -> None:
    rows = 520
    index = pd.bdate_range("2023-01-02", periods=rows)
    returns = np.full(rows, 1.0)
    price = 10.0 * np.cumprod(1.0 + returns / 100.0)
    history = pd.DataFrame(
        {
            "Price Close": price,
            "Total Return": returns,
            "Accumulated Volume": np.full(rows, 200.0),
            "Outstanding Shares": np.full(rows, 1000.0),
            "Company Market Cap": price * 1000.0,
        },
        index=index,
    )
    raw = pd.DataFrame(
        {
            "TRDPRC_1": price,
            "ACVOL_UNS": np.full(rows, 100.0),
            "HIGH_1": price * 1.01,
            "LOW_1": price * 0.99,
            "BID": price * 0.999,
            "ASK": price * 1.001,
        },
        index=index,
    )
    values = _derived_bridge_values(7, history, raw)
    assert values["return_total_5d"] == pytest.approx(1.01**5 - 1.0)
    assert values["median_dollar_volume_20d"] == pytest.approx(
        np.median(price[-20:] * 200.0)
    )
    assert values["share_turnover_20d"] == pytest.approx(0.2)
    assert values["zero_return_share_60d"] == 0.0
    assert values["share_turnover_volatility_63d"] == pytest.approx(0.0)


def test_short_history_preserves_long_window_missingness() -> None:
    index = pd.bdate_range("2025-01-02", periods=100)
    history = pd.DataFrame(
        {
            "Price Close": np.linspace(10.0, 20.0, 100),
            "Total Return": np.zeros(100),
            "Accumulated Volume": np.ones(100),
            "Outstanding Shares": np.full(100, 10.0),
            "Company Market Cap": np.linspace(100.0, 200.0, 100),
        },
        index=index,
    )
    raw = pd.DataFrame(
        {
            "TRDPRC_1": np.linspace(10.0, 20.0, 100),
            "ACVOL_UNS": np.ones(100),
            "HIGH_1": np.linspace(10.1, 20.2, 100),
            "LOW_1": np.linspace(9.9, 19.8, 100),
            "BID": np.linspace(9.99, 19.98, 100),
            "ASK": np.linspace(10.01, 20.02, 100),
        },
        index=index,
    )
    values = _derived_bridge_values(8, history, raw)
    assert np.isfinite(values["return_total_63d"])
    assert np.isnan(values["return_total_126d"])
    assert np.isnan(values["momentum_12_1"])


def test_nontrading_rows_added_by_point_in_time_fields_are_not_sessions() -> None:
    index = pd.bdate_range("2023-01-02", periods=520)
    history = pd.DataFrame(
        {
            "Price Close": np.full(520, 10.0),
            "Total Return": np.zeros(520),
            "Accumulated Volume": np.ones(520),
            "Outstanding Shares": np.full(520, 100.0),
            "Company Market Cap": np.full(520, 1000.0),
        },
        index=index,
    )
    extra_date = index[-20] + pd.Timedelta(days=1)
    history.loc[extra_date] = [np.nan, np.nan, np.nan, 100.0, 1000.0]
    raw = pd.DataFrame(
        {
            "TRDPRC_1": np.full(520, 10.0),
            "ACVOL_UNS": np.ones(520),
            "HIGH_1": np.full(520, 10.1),
            "LOW_1": np.full(520, 9.9),
            "BID": np.full(520, 9.99),
            "ASK": np.full(520, 10.01),
        },
        index=index,
    )
    values = _derived_bridge_values(9, history, raw)
    assert values["return_total_252d"] == 0.0
    assert values["zero_return_share_60d"] == 1.0


def test_batch_member_extracts_one_instrument() -> None:
    columns = pd.MultiIndex.from_product(
        [["AAA.O", "BBB.N"], ["Price Close", "Total Return"]]
    )
    frame = pd.DataFrame([[10.0, 1.0, 20.0, 2.0]], columns=columns)
    member = _batch_member(frame, "BBB.N")
    assert member is not None
    assert member.columns.tolist() == ["Price Close", "Total Return"]
    assert member.iloc[0].tolist() == [20.0, 2.0]
    assert _batch_member(frame, "MISSING") is None
