"""Checkpointed parity for V6.8 stock-level LSEG market formulas."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from portfolio_bot.features.feature_math import (
    compounded_return,
    corwin_schultz_pair,
    distance_to_high,
    downside_semivolatility,
    log_price_trend,
    maximum_drawdown,
    price_to_moving_average,
    realized_volatility,
    sample_excess_kurtosis,
    sample_skewness,
)
from portfolio_bot.research_v6_8.lseg_market_parity import (
    LsegMarketParityError,
    _market_bridge_metrics,
    _monthly_pass,
    _months,
)


DEFAULT_FIELDS = (
    "TR.PriceClose",
    "TR.TotalReturn",
    "TR.AccumulatedVolume",
    "TR.CompanyMarketCap",
)
RAW_FIELDS = ("TRDPRC_1", "ACVOL_UNS", "HIGH_1", "LOW_1", "BID", "ASK")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce").astype("float64")


def _complete(values: pd.Series, window: int) -> np.ndarray | None:
    if len(values) < window:
        return None
    result = values.iloc[-window:].to_numpy(np.float64)
    return result if np.isfinite(result).all() else None


def _compound(values: pd.Series, window: int) -> float:
    sample = _complete(values, window)
    return np.nan if sample is None else float(compounded_return(sample))


def _offset_compound(values: pd.Series, start: int, end: int) -> float:
    # Inclusive offsets relative to the final observed session (offset zero).
    first = len(values) - 1 + start
    last = len(values) - 1 + end
    if first < 0 or last < first:
        return np.nan
    sample = values.iloc[first : last + 1].to_numpy(np.float64)
    if not np.isfinite(sample).all():
        return np.nan
    result = compounded_return(sample)
    return np.nan if result is None else float(result)


def _security_spread(raw: pd.DataFrame) -> float:
    if len(raw) < 20:
        return np.nan
    work = raw.iloc[-20:]
    bid = _numeric(work, "BID")
    ask = _numeric(work, "ASK")
    midpoint = (bid + ask) / 2.0
    quoted = (ask - bid) / midpoint
    valid = (
        np.isfinite(quoted)
        & np.isfinite(midpoint)
        & (bid > 0.0)
        & (ask >= bid)
        & (midpoint > 0.0)
        & (quoted <= 0.20)
    )
    if int(valid.sum()) >= 10:
        return float(np.median(quoted[valid]))
    high = _numeric(work, "HIGH_1")
    low = _numeric(work, "LOW_1")
    close = _numeric(work, "TRDPRC_1")
    estimates: list[float] = []
    for index in range(19):
        estimate = corwin_schultz_pair(
            high.iloc[index],
            low.iloc[index],
            close.iloc[index],
            high.iloc[index + 1],
            low.iloc[index + 1],
        )
        if estimate is not None:
            estimates.append(estimate)
    return float(np.median(estimates)) if len(estimates) >= 10 else np.nan


def _finite_or_nan(value: float | None) -> float:
    return float(value) if value is not None and np.isfinite(value) else np.nan


def _derived_bridge_values(
    permno: int,
    history: pd.DataFrame,
    raw_history: pd.DataFrame,
) -> dict[str, Any]:
    """Rebuild the 47 stock-level frozen market features from LSEG."""

    history = history.sort_index().copy()
    raw_history = raw_history.sort_index().copy()
    # Point-in-time shares and market cap can add non-trading calendar rows
    # (notably for ADRs).  The frozen formulas use the verified US trading
    # session clock, so retain only rows with the three consolidated market
    # observations that define an actual security session.
    core = pd.concat(
        [
            _numeric(history, "Price Close"),
            _numeric(history, "Total Return"),
            _numeric(history, "Accumulated Volume"),
        ],
        axis=1,
    )
    history = history.loc[core.notna().all(axis=1)].copy()
    total_return = _numeric(history, "Total Return") / 100.0
    price = _numeric(history, "Price Close")
    adjusted_volume = _numeric(history, "Accumulated Volume")
    market_cap = _numeric(history, "Company Market Cap")
    raw_price = _numeric(raw_history, "TRDPRC_1")
    raw_volume = _numeric(raw_history, "ACVOL_UNS")
    raw_volume = raw_volume.reindex(history.index)
    dollar_volume = price * adjusted_volume
    # Turnover is algebraically dollar volume / market cap.  This avoids a
    # vendor-unit trap in TR.SharesOutstanding (company/common/ADR units can
    # differ across issuers) while remaining identical to volume/shares when
    # price, volume and capitalization describe the same listed security.
    turnover = dollar_volume / market_cap

    values: dict[str, Any] = {"permno": int(permno)}
    for window in (5, 63, 126, 252):
        values[f"return_total_{window}d"] = _compound(total_return, window)
    values["momentum_6_1"] = _offset_compound(total_return, -126, -22)
    values["momentum_12_1"] = _offset_compound(total_return, -252, -22)
    values["momentum_24_12"] = _offset_compound(total_return, -504, -253)
    reversal = _compound(total_return, 21)
    values["reversal_21d"] = -reversal if np.isfinite(reversal) else np.nan

    for window in (21, 63, 126, 252):
        path = _complete(price, window)
        values[f"price_to_ma_{window}d"] = (
            np.nan if path is None else _finite_or_nan(price_to_moving_average(path))
        )
    path_252 = _complete(price, 252)
    values["distance_to_high_252d"] = (
        np.nan if path_252 is None else _finite_or_nan(distance_to_high(path_252))
    )
    for window in (63, 126, 252):
        path = _complete(price, window)
        slope, r_squared = (None, None) if path is None else log_price_trend(path)
        values[f"log_price_trend_slope_{window}d"] = _finite_or_nan(slope)
        values[f"log_price_trend_r2_{window}d"] = _finite_or_nan(r_squared)
    returns_252 = _complete(total_return, 252)
    values["max_drawdown_252d"] = (
        np.nan if returns_252 is None else _finite_or_nan(maximum_drawdown(returns_252))
    )

    return_windows: dict[int, np.ndarray | None] = {}
    for window in (21, 63, 126, 252):
        sample = _complete(total_return, window)
        return_windows[window] = sample
        values[f"realized_volatility_{window}d"] = (
            np.nan if sample is None else _finite_or_nan(realized_volatility(sample))
        )
    for window in (63, 252):
        sample = return_windows[window]
        values[f"downside_semivolatility_{window}d"] = (
            np.nan if sample is None else _finite_or_nan(downside_semivolatility(sample))
        )
    if returns_252 is None:
        values["return_skewness_252d"] = np.nan
        values["return_excess_kurtosis_252d"] = np.nan
    else:
        values["return_skewness_252d"] = _finite_or_nan(sample_skewness(returns_252))
        values["return_excess_kurtosis_252d"] = _finite_or_nan(
            sample_excess_kurtosis(returns_252)
        )
    for window in (21, 63):
        sample = return_windows[window]
        values[f"maximum_daily_return_{window}d"] = (
            np.nan if sample is None else float(np.max(sample))
        )
        values[f"minimum_daily_return_{window}d"] = (
            np.nan if sample is None else float(np.min(sample))
        )

    trailing = _complete(total_return, 315)
    if trailing is None:
        values["volatility_63d_vs_trailing_median_252d"] = np.nan
    else:
        prior = [realized_volatility(trailing[index : index + 63]) for index in range(252)]
        current = realized_volatility(trailing[-63:])
        if current is None or any(value is None for value in prior):
            values["volatility_63d_vs_trailing_median_252d"] = np.nan
        else:
            denominator = float(np.median(np.asarray(prior, dtype=np.float64)))
            values["volatility_63d_vs_trailing_median_252d"] = (
                float(current / denominator - 1.0) if denominator > 0.0 else np.nan
            )

    signal_raw_price = raw_price.dropna()
    signal_market_cap = market_cap.dropna()
    values["log_absolute_price"] = (
        float(np.log(abs(signal_raw_price.iloc[-1])))
        if not signal_raw_price.empty and signal_raw_price.iloc[-1] != 0.0
        else np.nan
    )
    values["log_market_cap"] = (
        float(np.log(signal_market_cap.iloc[-1]))
        if not signal_market_cap.empty and signal_market_cap.iloc[-1] > 0.0
        else np.nan
    )
    for window in (20, 60):
        dollars = _complete(dollar_volume, window)
        values[f"median_dollar_volume_{window}d"] = (
            np.nan if dollars is None else float(np.median(dollars))
        )
        turns = _complete(turnover, window)
        values[f"share_turnover_{window}d"] = (
            np.nan if turns is None else float(np.mean(turns))
        )
        returns = _complete(total_return, window)
        values[f"amihud_illiquidity_{window}d"] = (
            np.nan
            if returns is None or dollars is None or np.any(dollars <= 0.0)
            else float(np.mean(np.abs(returns) / dollars))
        )
    values["median_estimated_full_spread_20d"] = _security_spread(raw_history)
    returns_60 = _complete(total_return, 60)
    volume_60 = _complete(raw_volume, 60)
    values["zero_return_share_60d"] = (
        np.nan if returns_60 is None else float(np.mean(returns_60 == 0.0))
    )
    values["zero_volume_share_60d"] = (
        np.nan if volume_60 is None else float(np.mean(volume_60 == 0.0))
    )
    dollars_272 = _complete(dollar_volume, 272)
    values["dollar_volume_shock_20d_vs_prior_252d"] = np.nan
    if dollars_272 is not None:
        current_median = float(np.median(dollars_272[-20:]))
        prior_median = float(np.median(dollars_272[:-20]))
        if prior_median > 0.0:
            values["dollar_volume_shock_20d_vs_prior_252d"] = (
                current_median / prior_median - 1.0
            )
    momentum_6_1 = values["momentum_6_1"]
    momentum_12_1 = values["momentum_12_1"]
    values["momentum_acceleration_6m"] = np.nan
    if (
        np.isfinite(momentum_6_1)
        and np.isfinite(momentum_12_1)
        and 1.0 + momentum_6_1 > 0.0
        and 1.0 + momentum_12_1 > 0.0
    ):
        older = (1.0 + momentum_12_1) / (1.0 + momentum_6_1) - 1.0
        values["momentum_acceleration_6m"] = momentum_6_1 - older
    turns_63 = _complete(turnover, 63)
    values["share_turnover_volatility_63d"] = (
        np.nan if turns_63 is None else float(np.std(turns_63, ddof=1))
    )
    return values


def _request_history(
    ld: Any,
    identifier: str | list[str],
    fields: Iterable[str],
    *,
    start: str,
    end: str,
    adjustments: str | None = None,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            arguments: dict[str, Any] = {
                "universe": [identifier] if isinstance(identifier, str) else identifier,
                "fields": list(fields),
                "interval": "1D",
                "start": start,
                "end": end,
            }
            if adjustments is not None:
                arguments["adjustments"] = adjustments
            return ld.get_history(**arguments)
        except Exception as error:  # pragma: no cover - live provider path
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    raise LsegMarketParityError(f"LSEG history request failed: {last_error}")


def _batch_member(frame: pd.DataFrame, identifier: str) -> pd.DataFrame | None:
    """Extract one requested instrument from an LSEG batch response."""

    if isinstance(frame.columns, pd.MultiIndex):
        instruments = frame.columns.get_level_values(0)
        if identifier not in instruments:
            return None
        result = frame[identifier].copy()
    else:
        result = frame.copy()
    result = result.dropna(how="all")
    return None if result.empty else result


def _identity(
    config_path: Path,
    model_month: Path,
    upstream_mapping: Path,
    run_months: list[str],
) -> dict[str, Any]:
    return {
        "config_sha256": _sha256(config_path),
        "code_sha256": _sha256(Path(__file__)),
        "model_month_sha256": _sha256(model_month),
        "upstream_mapping_sha256": _sha256(upstream_mapping),
        "run_months": run_months,
    }


def run_lseg_market_derived_parity(
    *,
    config_path: Path,
    output: Path,
    model_feature_months: Path,
    pilot_only: bool = False,
) -> dict[str, Any]:
    """Run or resume the stock-level derived-market parity gate."""

    try:
        import lseg.data as ld
    except ImportError as error:  # pragma: no cover
        raise LsegMarketParityError("run with /opt/anaconda3/envs/finance/bin/python") from error

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "research_v6_8_lseg_market_derived_parity_v1":
        raise LsegMarketParityError("derived parity config schema differs")
    run_months = (
        [str(config["pilot_month"])]
        if pilot_only
        else _months(config["holding_month_start"], config["holding_month_end"])
    )
    features = [str(value) for value in config["features"]]
    upstream = config_path.parents[2] / config["upstream_market_parity"]
    output.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    completed: list[str] = []
    ld.open_session()
    try:
        for month in run_months:
            month_output = output / "months" / month
            model_path = model_feature_months / f"{month}.parquet"
            mapping_path = upstream / "months" / month / "identifier_metrics.csv"
            if not model_path.exists() or not mapping_path.exists():
                raise LsegMarketParityError(f"missing derived parity input for {month}")
            identity = _identity(config_path, model_path, mapping_path, run_months)
            receipt_path = month_output / "receipt.json"
            if receipt_path.exists():
                existing = json.loads(receipt_path.read_text(encoding="utf-8"))
                if existing.get("identity") == identity:
                    receipts.append(existing)
                    completed.append(month)
                    continue
            _atomic_json(
                output / "progress.json",
                {
                    "schema": "research_v6_8_lseg_market_derived_parity_progress_v1",
                    "status": "RUNNING",
                    "updated_at_utc": _utc_now(),
                    "completed_months": completed,
                    "current_month": month,
                    "current_security": None,
                },
            )
            model = pd.read_parquet(model_path, columns=["signal_date", "permno", *features])
            mapping = pd.read_csv(
                mapping_path,
                dtype={"consolidated_ric": "string", "history_identifier": "string"},
            )
            mapping = mapping.loc[
                mapping["mapping_status"].eq("CONSOLIDATED_RIC_RESOLVED")
            ].copy()
            reference = mapping[["permno", "history_identifier"]].merge(
                model, on="permno", how="left", validate="one_to_one"
            )
            signal_dates = pd.to_datetime(reference["signal_date"], errors="raise").dt.date
            if signal_dates.nunique() != 1:
                raise LsegMarketParityError(f"{month} does not have one signal date")
            signal_date = signal_dates.iloc[0].isoformat()
            start = (
                pd.Timestamp(signal_date)
                - timedelta(days=int(config["history_calendar_days"]))
            ).date().isoformat()
            rows: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            identifiers = reference["history_identifier"].astype(str).tolist()
            batch_history: pd.DataFrame | None = None
            batch_raw: pd.DataFrame | None = None
            try:
                batch_history = _request_history(
                    ld, identifiers, DEFAULT_FIELDS, start=start, end=signal_date
                )
                batch_raw = _request_history(
                    ld,
                    identifiers,
                    RAW_FIELDS,
                    start=start,
                    end=signal_date,
                    adjustments="unadjusted",
                )
            except Exception:
                # A single unavailable permanent identifier must not invalidate
                # the entire month; the loop below falls back per security.
                batch_history = None
                batch_raw = None
            for position, row in reference.iterrows():
                identifier = str(row["history_identifier"])
                _atomic_json(
                    output / "progress.json",
                    {
                        "schema": "research_v6_8_lseg_market_derived_parity_progress_v1",
                        "status": "RUNNING",
                        "updated_at_utc": _utc_now(),
                        "completed_months": completed,
                        "current_month": month,
                        "current_security": int(position) + 1,
                        "securities_in_month": int(len(reference)),
                    },
                )
                try:
                    history = (
                        None
                        if batch_history is None
                        else _batch_member(batch_history, identifier)
                    )
                    raw = None if batch_raw is None else _batch_member(batch_raw, identifier)
                    if history is None:
                        history = _request_history(
                            ld, identifier, DEFAULT_FIELDS, start=start, end=signal_date
                        )
                    if raw is None:
                        raw = _request_history(
                            ld,
                            identifier,
                            RAW_FIELDS,
                            start=start,
                            end=signal_date,
                            adjustments="unadjusted",
                        )
                    if history.empty or raw.empty:
                        raise LsegMarketParityError("empty LSEG history")
                    rows.append(_derived_bridge_values(int(row["permno"]), history, raw))
                except Exception as error:  # pragma: no cover - live provider path
                    errors.append(
                        {
                            "permno": int(row["permno"]),
                            "history_identifier": identifier,
                            "error_type": type(error).__name__,
                            "error": str(error)[:500],
                        }
                    )
            bridge = pd.DataFrame(rows)
            if bridge.empty:
                detail = pd.DataFrame(
                    {
                        "feature": features,
                        "comparable_rows": 0,
                        "spearman_rank_correlation": np.nan,
                        "median_absolute_rank_difference": np.nan,
                        "median_absolute_relative_difference": np.nan,
                        "absolute_missingness_difference": 1.0,
                        "pilot_gate_pass": False,
                    }
                )
            else:
                detail, _ = _market_bridge_metrics(
                    reference[["permno", *features]], bridge[["permno", *features]], features
                )
            resolved_share = len(bridge) / len(reference)
            passed = _monthly_pass(
                detail, resolved_share=resolved_share, gate=config["monthly_gate"]
            )
            _atomic_csv(month_output / "feature_metrics.csv", detail)
            _atomic_csv(month_output / "request_errors.csv", pd.DataFrame(errors))
            receipt = {
                "schema": "research_v6_8_lseg_market_derived_parity_month_v1",
                "created_at_utc": _utc_now(),
                "status": "PASS" if passed else "FAIL",
                "holding_month": month,
                "signal_date": signal_date,
                "identity": identity,
                "sample_size": int(len(reference)),
                "resolved_instruments": int(len(bridge)),
                "resolved_instrument_share": resolved_share,
                "passing_features": int(detail["pilot_gate_pass"].sum()),
                "compared_features": int(len(features)),
                "minimum_rank_correlation": (
                    None
                    if detail["spearman_rank_correlation"].dropna().empty
                    else float(detail["spearman_rank_correlation"].min())
                ),
                "maximum_missingness_difference": float(
                    detail["absolute_missingness_difference"].max()
                ),
                "no_economic_values_persisted": True,
            }
            _atomic_json(receipt_path, receipt)
            receipts.append(receipt)
            completed.append(month)
    finally:
        ld.close_session()

    monthly = pd.DataFrame(receipts).sort_values("holding_month")
    passing_share = float(monthly["status"].eq("PASS").mean())
    resolved_share = float(
        monthly["resolved_instruments"].sum() / monthly["sample_size"].sum()
    )
    full_gate = config["full_window_gate"]
    passed = bool(
        passing_share >= float(full_gate["minimum_passing_month_share"])
        and resolved_share
        >= float(full_gate["minimum_all_month_resolved_instrument_share"])
    )
    _atomic_csv(output / "monthly_summary.csv", monthly)
    final = {
        "schema": "research_v6_8_lseg_market_derived_parity_v1",
        "created_at_utc": _utc_now(),
        "status": "PASS" if passed else "FAIL",
        "scope": "PILOT" if pilot_only else "FULL_WINDOW",
        "months_expected": len(run_months),
        "months_completed": len(monthly),
        "months_passing": int(monthly["status"].eq("PASS").sum()),
        "passing_month_share": passing_share,
        "all_month_resolved_instrument_share": resolved_share,
        "features_tested": len(features),
        "excluded_cross_sectional_features": config["excluded_cross_sectional_features"],
    }
    _atomic_json(output / "receipt.json", final)
    _atomic_json(
        output / "progress.json",
        {
            "schema": "research_v6_8_lseg_market_derived_parity_progress_v1",
            "status": "COMPLETE",
            "updated_at_utc": _utc_now(),
            "completed_months": run_months,
        },
    )
    return final


__all__ = ["_derived_bridge_values", "run_lseg_market_derived_parity"]
