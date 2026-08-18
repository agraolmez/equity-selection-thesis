"""Prospective frozen Stage 8 risk and cost inputs for V6.8 paper trading."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from portfolio_bot.risk_cost.cost_math import (
    assign_adv_liquidity_quintiles,
    borrow_rate_schedule,
    estimate_impact_volatility,
)
from portfolio_bot.risk_cost.risk_math import (
    FACTOR_NAMES,
    estimate_factor_covariance,
    fit_security_risk_ewls,
    regularize_factor_model_covariance,
    shrink_residual_variances,
)


class LiveRiskCostError(RuntimeError):
    """The prospective risk/cost input block failed closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def build_live_risk_cost(
    *,
    market: pd.DataFrame,
    history: pd.DataFrame,
    identity: pd.DataFrame,
    daily_factors: pd.DataFrame,
    signal_date: pd.Timestamp,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if market.empty or market["live_security_id"].duplicated().any():
        raise LiveRiskCostError("market risk/cost grain is invalid")
    risk = config["risk"]
    history = history.copy()
    factors = daily_factors.copy()
    history["observation_date"] = pd.to_datetime(
        history["observation_date"], errors="raise"
    ).dt.normalize()
    factors["observation_date"] = pd.to_datetime(
        factors["observation_date"], errors="raise"
    ).dt.normalize()
    cutoff = pd.Timestamp(signal_date).normalize()
    history = history.loc[history["observation_date"].le(cutoff)].copy()
    factors = factors.loc[factors["observation_date"].le(cutoff)].copy()
    sessions = sorted(pd.Timestamp(value) for value in history["observation_date"].unique())
    lookback_count = int(risk["lookback_global_sessions"])
    if len(sessions) < lookback_count:
        raise LiveRiskCostError("current market history has fewer than 252 global sessions")
    sessions = sessions[-lookback_count:]
    ages = {day: len(sessions) - 1 - position for position, day in enumerate(sessions)}
    factor_index = factors.set_index("observation_date")
    factor_dates = [day for day in sessions if day in factor_index.index]
    factor_values = factor_index.loc[factor_dates, list(FACTOR_NAMES)].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_factor = np.isfinite(factor_values.to_numpy(float)).all(axis=1)
    factor_dates = [day for day, valid in zip(factor_dates, valid_factor, strict=True) if valid]
    factor_values = factor_index.loc[factor_dates, list(FACTOR_NAMES)].to_numpy(float)
    covariance = estimate_factor_covariance(
        factor_values,
        [ages[day] for day in factor_dates],
        half_life_sessions=float(risk["half_life_sessions"]),
        minimum_observations=int(risk["minimum_observations"]),
    )
    if covariance is None:
        raise LiveRiskCostError("official pre-cutoff factors cannot form covariance")
    covariance_daily = np.asarray(covariance.covariance_daily, dtype=float)
    rf = pd.to_numeric(factor_index["rf"], errors="coerce")
    identity_index = identity.set_index("live_security_id")
    history_groups = {
        str(key): group.set_index("observation_date").sort_index()
        for key, group in history.groupby("live_security_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    raw_specific: list[float | None] = []
    observation_counts: list[int | None] = []
    sectors: list[str | None] = []
    scoring: list[bool] = []
    for market_row in market.itertuples(index=False):
        security_id = str(market_row.live_security_id)
        group = history_groups.get(security_id)
        fit = None
        impact = None
        if group is not None:
            security_return = pd.to_numeric(group["total_return"], errors="coerce")
            dates = [
                day for day in sessions
                if day in security_return.index
                and day in factor_index.index
                and np.isfinite(security_return.loc[day])
                and np.isfinite(rf.loc[day])
                and np.isfinite(
                    factor_index.loc[day, list(FACTOR_NAMES)].to_numpy(dtype=float)
                ).all()
            ]
            fit = fit_security_risk_ewls(
                [float(security_return.loc[day]) for day in dates],
                [float(rf.loc[day]) for day in dates],
                factor_index.loc[dates, list(FACTOR_NAMES)].to_numpy(float),
                [ages[day] for day in dates],
                half_life_sessions=float(risk["half_life_sessions"]),
                minimum_observations=int(risk["minimum_observations"]),
            )
            aligned = security_return.reindex(sessions)
            impact = estimate_impact_volatility(
                aligned.iloc[-20:].tolist(), aligned.iloc[-60:].tolist()
            )
        sector_value = (
            identity_index.loc[security_id, "sic2_industry_category"]
            if security_id in identity_index.index
            else None
        )
        sector = None if pd.isna(sector_value) else str(sector_value)
        eligible = bool(getattr(market_row, "long_eligible"))
        short_screen = config["short_eligibility"]
        price = float(getattr(market_row, "signal_price_usd", np.nan))
        market_cap = float(getattr(market_row, "issue_market_cap_usd", np.nan))
        dollar_volume_60d = float(
            getattr(market_row, "median_dollar_volume_60d", np.nan)
        )
        zero_volume_share = float(
            getattr(market_row, "zero_volume_share_60d", np.nan)
        )
        short_eligible = bool(
            eligible
            and np.isfinite(price)
            and price >= float(short_screen["minimum_price_usd"])
            and np.isfinite(market_cap)
            and market_cap >= float(short_screen["minimum_market_cap_usd"])
            and np.isfinite(dollar_volume_60d)
            and dollar_volume_60d
            >= float(short_screen["minimum_median_dollar_volume_60d"])
            and np.isfinite(zero_volume_share)
            and 1.0 - zero_volume_share
            >= float(short_screen["minimum_positive_volume_share_60d"])
        )
        spread = float(getattr(market_row, "median_estimated_full_spread_20d", np.nan))
        adv = float(getattr(market_row, "median_dollar_volume_20d", np.nan))
        row = {
            "live_security_id": security_id,
            "symbol": str(getattr(market_row, "symbol", "")).upper(),
            "long_eligible": eligible,
            "short_eligible": short_eligible,
            "sic2_industry_category": sector or "UNKNOWN_INDUSTRY",
            "risk_observation_count": 0 if fit is None else int(fit.observation_count),
            "beta_risk_intercept": np.nan if fit is None else float(fit.intercept),
            **{
                f"beta_risk_{name}": np.nan if fit is None else float(fit.betas[index])
                for index, name in enumerate(FACTOR_NAMES)
            },
            "specific_variance_raw_daily": np.nan if fit is None else float(fit.residual_variance_daily),
            "full_spread_20d": spread,
            "adv20_usd": adv,
            "impact_volatility_daily": np.nan if impact is None else float(impact.daily_volatility),
            "impact_volatility_source": "MISSING" if impact is None else f"{impact.source_window_sessions}_SESSION",
        }
        rows.append(row)
        raw_specific.append(None if fit is None else float(fit.residual_variance_daily))
        observation_counts.append(None if fit is None else int(fit.observation_count))
        sectors.append(sector)
        scoring.append(eligible)
    result = pd.DataFrame(rows)
    shrunk = shrink_residual_variances(
        raw_specific,
        observation_counts,
        sectors,
        scoring,
        minimum_observations=int(risk["minimum_observations"]),
        sector_minimum_securities=int(risk["sector_minimum_securities"]),
        prior_observations=float(risk["specific_variance_prior_observations"]),
    )
    if shrunk is None:
        raise LiveRiskCostError("specific-risk shrinkage inputs are invalid")
    result["shrinkage_target_daily"] = [
        np.nan if value is None else value.reference_median_daily for value in shrunk
    ]
    result["shrinkage_target_source"] = [
        "MISSING" if value is None else value.reference_scope.upper() for value in shrunk
    ]
    result["shrinkage_population_count"] = [
        0 if value is None else value.reference_security_count for value in shrunk
    ]
    result["specific_variance_shrunk_daily"] = [
        np.nan if value is None else value.shrunk_variance_daily for value in shrunk
    ]
    present = (
        result[[f"beta_risk_{name}" for name in FACTOR_NAMES]].notna().all(axis=1).to_numpy()
        & result["specific_variance_shrunk_daily"].notna().to_numpy()
    )
    if not present.any():
        raise LiveRiskCostError("no eligible security has complete prospective risk")
    regularized = regularize_factor_model_covariance(
        result.loc[present, [f"beta_risk_{name}" for name in FACTOR_NAMES]].to_numpy(float),
        covariance_daily,
        result.loc[present, "specific_variance_shrunk_daily"].to_numpy(float),
        monthly_session_scale=float(risk["monthly_session_scale"]),
    )
    if regularized is None:
        raise LiveRiskCostError("prospective risk regularization failed")
    result["risk_ridge_daily"] = np.nan
    result["specific_variance_daily"] = np.nan
    result["specific_variance_monthly"] = np.nan
    result.loc[present, "risk_ridge_daily"] = regularized.ridge_daily
    result.loc[present, "specific_variance_daily"] = regularized.specific_variances_daily_post_ridge
    result.loc[present, "specific_variance_monthly"] = regularized.specific_variances_monthly_post_ridge

    spread_values = result["full_spread_20d"].to_numpy(float)
    adv_values = result["adv20_usd"].to_numpy(float)
    impact_values = result["impact_volatility_daily"].to_numpy(float)
    complete_cost = (
        np.isfinite(spread_values)
        & (spread_values >= 0.0)
        & np.isfinite(adv_values)
        & (adv_values > 0.0)
        & np.isfinite(impact_values)
        & (impact_values >= 0.0)
    )
    allowed = present & complete_cost
    result["new_long_trade_allowed"] = allowed & result["long_eligible"].to_numpy(bool)
    result["new_short_trade_allowed"] = allowed & result["short_eligible"].to_numpy(bool)
    result["primary_borrow_rate_annual"] = np.nan
    result["stress_borrow_rate_annual"] = np.nan
    eligible_indices = np.flatnonzero(
        allowed
        & (
            result["long_eligible"].to_numpy(bool)
            | result["short_eligible"].to_numpy(bool)
        )
    )
    quintiles = assign_adv_liquidity_quintiles(
        list(range(len(eligible_indices))),
        result.loc[eligible_indices, "adv20_usd"].tolist(),
    )
    for position, quintile in zip(eligible_indices, quintiles, strict=True):
        schedule = borrow_rate_schedule(lowest_liquidity_quintile=int(quintile) == 1)
        result.loc[position, "primary_borrow_rate_annual"] = schedule.primary.annual
        result.loc[position, "stress_borrow_rate_annual"] = schedule.stress.annual
    result["risk_status"] = np.where(present, "PRESENT", "MISSING_RISK")
    result["cost_input_status"] = np.where(complete_cost, "PRESENT", "MISSING_COST_INPUT")
    covariance_rows = []
    monthly = np.asarray(regularized.factor_covariance_monthly, dtype=float)
    for row_index, row_name in enumerate(FACTOR_NAMES):
        for column_index, column_name in enumerate(FACTOR_NAMES):
            covariance_rows.append(
                {
                    "factor_row": row_name,
                    "factor_col": column_name,
                    "factor_row_ordinal": row_index,
                    "factor_col_ordinal": column_index,
                    "factor_observation_count": covariance.observation_count,
                    "covariance_daily": covariance_daily[row_index, column_index],
                    "covariance_monthly": monthly[row_index, column_index],
                }
            )
    return result, pd.DataFrame(covariance_rows)


def materialize_live_risk_cost(
    *,
    config_path: Path,
    market_features_path: Path,
    market_history_path: Path,
    identity_path: Path,
    factor_snapshot: Path,
    output: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "research_v6_8_live_risk_cost_v1":
        raise LiveRiskCostError("live risk/cost config schema differs")
    market = pd.read_parquet(market_features_path)
    history = pd.read_parquet(market_history_path)
    identity = pd.read_parquet(identity_path)
    factor_receipt = json.loads((factor_snapshot / "receipt.json").read_text(encoding="utf-8"))
    daily_factor_path = factor_snapshot / "daily_factors.parquet"
    if (
        factor_receipt.get("status") != "PASS"
        or factor_receipt.get("daily_sha256") != _sha256(daily_factor_path)
    ):
        raise LiveRiskCostError("official factor snapshot is not authenticated")
    dates = pd.to_datetime(market["latest_market_session"], errors="coerce")
    if dates.dropna().empty:
        raise LiveRiskCostError("market block has no current session")
    signal_date = pd.Timestamp(dates.max())
    result, covariance = build_live_risk_cost(
        market=market,
        history=history,
        identity=identity,
        daily_factors=pd.read_parquet(daily_factor_path),
        signal_date=signal_date,
        config=config,
    )
    risk_path = output / "risk_cost.parquet"
    covariance_path = output / "factor_covariance.parquet"
    _atomic_parquet(risk_path, result)
    _atomic_parquet(covariance_path, covariance)
    receipt = {
        "schema": "research_v6_8_live_risk_cost_receipt_v1",
        "status": "PASS",
        "signal_date": signal_date.date().isoformat(),
        "rows": int(len(result)),
        "risk_present_rows": int(result["risk_status"].eq("PRESENT").sum()),
        "new_trade_allowed_rows": int(result["new_long_trade_allowed"].sum()),
        "factor_covariance_rows": int(len(covariance)),
        "risk_cost_sha256": _sha256(risk_path),
        "factor_covariance_sha256": _sha256(covariance_path),
        "factor_snapshot_receipt_sha256": _sha256(factor_snapshot / "receipt.json"),
        "market_features_sha256": _sha256(market_features_path),
        "market_history_sha256": _sha256(market_history_path),
        "identity_sha256": _sha256(identity_path),
    }
    _atomic_json(output / "receipt.json", receipt)
    return receipt


__all__ = ["LiveRiskCostError", "build_live_risk_cost", "materialize_live_risk_cost"]
