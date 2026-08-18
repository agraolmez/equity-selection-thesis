"""Outcome-blind transformations for the V6 engineered feature block.

The functions in this module consume only information present at a monthly
signal.  Raw CRSP turnover history and point-in-time Compustat observations are
materialized separately by ``scripts/materialize_v6_engineered_features.py``;
this module owns the inexpensive cross-sectional and exact-calendar-lag
transforms of the already authenticated V6 panel.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


ENGINEERED_FEATURE_SPECS: tuple[Mapping[str, object], ...] = (
    {
        "feature_name": "industry_momentum_12_1_vw",
        "family": "price_trends_path",
        "unit": "decimal_return",
        "transformation": (
            "same_signal_sic2_value_weighted_momentum_12_1_including_stock;"
            "minimum_two_industry_contributors"
        ),
    },
    {
        "feature_name": "momentum_acceleration_6m",
        "family": "price_trends_path",
        "unit": "decimal_return_difference",
        "transformation": (
            "momentum_6_1_minus_implied_older_six_month_return_from_momentum_12_1"
        ),
    },
    {
        "feature_name": "share_turnover_volatility_63d",
        "family": "size_liquidity_trading_frictions",
        "unit": "daily_fraction_standard_deviation",
        "transformation": (
            "sample_standard_deviation_of_daily_volume_shares_divided_by_"
            "date_valid_shares_over_exact_63_sessions"
        ),
    },
    {
        "feature_name": "lseg_forward_eps_yield_fy1",
        "family": "lseg_analyst_expectations",
        "unit": "ratio",
        "transformation": "lseg_epsmean_fy1_divided_by_signal_absolute_price",
    },
    {
        "feature_name": "lseg_forward_eps_growth_fy2_vs_fy1",
        "family": "lseg_analyst_expectations",
        "unit": "signed_growth_ratio",
        "transformation": "fy2_minus_fy1_eps_mean_divided_by_absolute_fy1_eps_mean",
    },
    {
        "feature_name": "lseg_forward_revenue_growth_fy2_vs_fy1",
        "family": "lseg_analyst_expectations",
        "unit": "signed_growth_ratio",
        "transformation": (
            "fy2_minus_fy1_revenue_mean_divided_by_absolute_fy1_revenue_mean"
        ),
    },
    {
        "feature_name": "lseg_price_target_upside",
        "family": "lseg_analyst_expectations",
        "unit": "ratio",
        "transformation": "price_target_mean_divided_by_signal_absolute_price_minus_one",
    },
    {
        "feature_name": "lseg_price_target_revision_1m",
        "family": "lseg_analyst_expectations",
        "unit": "signed_growth_ratio",
        "transformation": "exact_calendar_1m_price_target_mean_percent_change_same_permno",
    },
    {
        "feature_name": "lseg_price_target_revision_3m",
        "family": "lseg_analyst_expectations",
        "unit": "signed_growth_ratio",
        "transformation": "exact_calendar_3m_price_target_mean_percent_change_same_permno",
    },
    {
        "feature_name": "lseg_recommendation_upgrade_1m",
        "family": "lseg_analyst_expectations",
        "unit": "recommendation_scale_change",
        "transformation": (
            "exact_calendar_1m_prior_minus_current_recmean_same_permno;"
            "positive_is_upgrade_on_lseg_1_strong_buy_5_strong_sell_scale"
        ),
    },
    {
        "feature_name": "lseg_recommendation_upgrade_3m",
        "family": "lseg_analyst_expectations",
        "unit": "recommendation_scale_change",
        "transformation": (
            "exact_calendar_3m_prior_minus_current_recmean_same_permno;"
            "positive_is_upgrade_on_lseg_1_strong_buy_5_strong_sell_scale"
        ),
    },
    {
        "feature_name": "quarterly_earnings_change_to_market_cap",
        "family": "valuation_fundamentals",
        "unit": "ratio",
        "transformation": (
            "latest_public_quarter_ibq_minus_same_fiscal_quarter_prior_year_"
            "scaled_to_usd_signal_market_cap"
        ),
    },
    {
        "feature_name": "quarterly_sales_change_to_market_cap",
        "family": "valuation_fundamentals",
        "unit": "ratio",
        "transformation": (
            "latest_public_quarter_saleq_minus_same_fiscal_quarter_prior_year_"
            "scaled_to_usd_signal_market_cap"
        ),
    },
    {
        "feature_name": "consecutive_quarterly_earnings_increases",
        "family": "valuation_fundamentals",
        "unit": "quarters",
        "transformation": (
            "consecutive_latest_quarters_up_to_eight_with_ibq_above_same_"
            "fiscal_quarter_prior_year"
        ),
    },
    {
        "feature_name": "days_since_latest_quarterly_report",
        "family": "valuation_fundamentals",
        "unit": "days",
        "transformation": "signal_date_minus_latest_public_quarter_availability_date",
    },
)

ENGINEERED_FEATURE_NAMES = tuple(
    str(item["feature_name"]) for item in ENGINEERED_FEATURE_SPECS
)

PANEL_INPUT_COLUMNS = (
    "holding_month",
    "signal_date",
    "permno",
    "permco",
    "momentum_6_1",
    "momentum_12_1",
    "log_market_cap",
    "log_absolute_price",
    "sic2_industry_category",
    "lseg_epsmean_fy1",
    "lseg_epsmean_fy2",
    "lseg_revenuemean_fy1",
    "lseg_revenuemean_fy2",
    "lseg_pricetargetmean",
    "lseg_recmean",
)


def _finite(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(np.float64)


def _finite_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0.0)
    )
    output[valid] = numerator[valid] / denominator[valid]
    output[~np.isfinite(output)] = np.nan
    return output


def _exact_calendar_lag(
    frame: pd.DataFrame,
    column: str,
    months: int,
) -> np.ndarray:
    if months <= 0:
        raise ValueError("months must be positive")
    if frame.duplicated(["holding_month", "permno"]).any():
        raise ValueError("exact lag input has duplicate stock-month keys")
    period = pd.PeriodIndex(frame["holding_month"].astype(str), freq="M")
    ordinal = period.astype("int64").to_numpy()
    values = _finite(frame[column])
    lookup = pd.Series(
        values,
        index=pd.MultiIndex.from_arrays(
            [ordinal, frame["permno"].to_numpy(np.int64)],
            names=["month_ordinal", "permno"],
        ),
    )
    requested = pd.MultiIndex.from_arrays(
        [ordinal - months, frame["permno"].to_numpy(np.int64)],
        names=lookup.index.names,
    )
    return lookup.reindex(requested).to_numpy(np.float64)


def _industry_momentum(frame: pd.DataFrame) -> np.ndarray:
    momentum = _finite(frame["momentum_12_1"])
    log_market_cap = _finite(frame["log_market_cap"])
    market_cap = np.full(len(frame), np.nan, dtype=np.float64)
    valid_cap = np.isfinite(log_market_cap)
    market_cap[valid_cap] = np.exp(log_market_cap[valid_cap])
    industry = frame["sic2_industry_category"].astype("string")
    valid = (
        np.isfinite(momentum)
        & np.isfinite(market_cap)
        & (market_cap > 0.0)
        & industry.notna().to_numpy()
    )
    work = pd.DataFrame(
        {
            "holding_month": frame["holding_month"].astype(str),
            "industry": industry,
            "weighted": np.where(valid, momentum * market_cap, 0.0),
            "weight": np.where(valid, market_cap, 0.0),
            "contributor": valid.astype(np.int64),
        }
    )
    grouped = work.groupby(
        ["holding_month", "industry"], observed=True, sort=False, dropna=False
    )
    total_weighted = grouped["weighted"].transform("sum").to_numpy(np.float64)
    total_weight = grouped["weight"].transform("sum").to_numpy(np.float64)
    total_count = grouped["contributor"].transform("sum").to_numpy(np.int64)
    output = _finite_divide(total_weighted, total_weight)
    output[(total_count < 2) | industry.isna().to_numpy()] = np.nan
    return output


def derive_panel_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the 11 engineered features derived from the monthly V6 panel."""

    missing = sorted(set(PANEL_INPUT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"engineered panel input omits columns: {missing}")
    work = frame.loc[:, PANEL_INPUT_COLUMNS].copy()
    work["holding_month"] = work["holding_month"].astype(str)
    work["signal_date"] = pd.to_datetime(work["signal_date"], errors="raise")
    work["permno"] = pd.to_numeric(work["permno"], errors="raise").astype("int64")
    work["permco"] = pd.to_numeric(work["permco"], errors="raise").astype("int64")
    work = work.sort_values(["holding_month", "permno"], kind="mergesort").reset_index(
        drop=True
    )
    if work.duplicated(["holding_month", "permno"]).any():
        raise ValueError("engineered panel input contains duplicate stock-months")

    output = work.loc[:, ["holding_month", "signal_date", "permno", "permco"]].copy()
    output["industry_momentum_12_1_vw"] = _industry_momentum(work)

    momentum_6_1 = _finite(work["momentum_6_1"])
    momentum_12_1 = _finite(work["momentum_12_1"])
    older_six_month = _finite_divide(1.0 + momentum_12_1, 1.0 + momentum_6_1) - 1.0
    invalid_wealth = (1.0 + momentum_6_1 <= 0.0) | (1.0 + momentum_12_1 <= 0.0)
    older_six_month[invalid_wealth] = np.nan
    output["momentum_acceleration_6m"] = momentum_6_1 - older_six_month

    log_price = _finite(work["log_absolute_price"])
    price = np.full(len(work), np.nan, dtype=np.float64)
    price_valid = np.isfinite(log_price)
    price[price_valid] = np.exp(log_price[price_valid])
    eps_fy1 = _finite(work["lseg_epsmean_fy1"])
    eps_fy2 = _finite(work["lseg_epsmean_fy2"])
    revenue_fy1 = _finite(work["lseg_revenuemean_fy1"])
    revenue_fy2 = _finite(work["lseg_revenuemean_fy2"])
    price_target = _finite(work["lseg_pricetargetmean"])
    recommendation = _finite(work["lseg_recmean"])

    output["lseg_forward_eps_yield_fy1"] = _finite_divide(eps_fy1, price)
    output["lseg_forward_eps_growth_fy2_vs_fy1"] = _finite_divide(
        eps_fy2 - eps_fy1, np.abs(eps_fy1)
    )
    output["lseg_forward_revenue_growth_fy2_vs_fy1"] = _finite_divide(
        revenue_fy2 - revenue_fy1, np.abs(revenue_fy1)
    )
    output["lseg_price_target_upside"] = _finite_divide(price_target, price) - 1.0

    for lag in (1, 3):
        prior_target = _exact_calendar_lag(work, "lseg_pricetargetmean", lag)
        output[f"lseg_price_target_revision_{lag}m"] = _finite_divide(
            price_target - prior_target, np.abs(prior_target)
        )
        prior_recommendation = _exact_calendar_lag(work, "lseg_recmean", lag)
        upgrade = prior_recommendation - recommendation
        upgrade[~(np.isfinite(prior_recommendation) & np.isfinite(recommendation))] = np.nan
        output[f"lseg_recommendation_upgrade_{lag}m"] = upgrade

    numeric = output.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("engineered panel transformations produced infinity")
    return output


__all__ = [
    "ENGINEERED_FEATURE_NAMES",
    "ENGINEERED_FEATURE_SPECS",
    "PANEL_INPUT_COLUMNS",
    "derive_panel_engineered_features",
]
