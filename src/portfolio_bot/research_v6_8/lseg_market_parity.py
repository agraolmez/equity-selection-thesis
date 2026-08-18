"""Resumable full-window CRSP-to-LSEG market parity for V6.8.

The runner compares a deterministic, market-cap-stratified monthly sample of
frozen market features with the same formulas rebuilt from LSEG consolidated
US equity histories.  Each month is an atomic checkpoint bound to its config,
code, and two input files.  Venue-only quote RICs are never accepted because
their volume can omit consolidated trading activity.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_bot.research_v6_8.lseg_production_probe import (
    _consolidated_ric_history,
    _market_bridge_metrics,
)


class LsegMarketParityError(RuntimeError):
    """A parity input, checkpoint identity, or live LSEG request is invalid."""


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


def _months(start: str, end: str) -> list[str]:
    first = pd.Period(start, freq="M")
    last = pd.Period(end, freq="M")
    if first > last:
        raise LsegMarketParityError("holding_month_start follows holding_month_end")
    return [str(value) for value in pd.period_range(first, last, freq="M")]


def _identity(
    *,
    config_path: Path,
    archive_month: Path,
    model_feature_month: Path,
) -> dict[str, str]:
    return {
        "config_sha256": _sha256(config_path),
        "code_sha256": _sha256(Path(__file__)),
        "resolver_code_sha256": _sha256(
            Path(__file__).with_name("lseg_production_probe.py")
        ),
        "archive_month_sha256": _sha256(archive_month),
        "model_feature_month_sha256": _sha256(model_feature_month),
    }


def _sample_population(
    archive: pd.DataFrame,
    model_features: pd.DataFrame,
    *,
    features: list[str],
    sample_size: int,
) -> pd.DataFrame:
    population = archive.merge(
        model_features[["permno", *features]],
        on="permno",
        how="inner",
        validate="one_to_one",
    )
    normalized_ticker = population["ticker"].astype("string").str.strip()
    population = population.loc[
        normalized_ticker.notna()
        & normalized_ticker.ne("")
        & population["log_absolute_price"].notna()
        & population["market_cap_usd"].notna()
    ].sort_values(["market_cap_usd", "permno"], kind="mergesort")
    population = population.reset_index(drop=True)
    if len(population) < sample_size:
        raise LsegMarketParityError(
            f"only {len(population)} valid rows for sample size {sample_size}"
        )
    indices = np.linspace(0, len(population) - 1, sample_size).round().astype(int)
    sample = population.iloc[indices].drop_duplicates("permno").copy()
    if len(sample) != sample_size:
        raise LsegMarketParityError("deterministic sample contains duplicate PERMNOs")
    return sample


def _bridge_values(permno: int, history: pd.DataFrame) -> dict[str, Any]:
    total_return = pd.to_numeric(history["Total Return"], errors="coerce") / 100.0
    price = pd.to_numeric(history["Price Close"], errors="coerce")
    volume = pd.to_numeric(history["Accumulated Volume"], errors="coerce")
    valid = total_return.notna() & price.notna() & volume.notna()
    total_return = total_return[valid]
    price = price[valid]
    volume = volume[valid]
    values: dict[str, Any] = {
        "permno": int(permno),
        "log_absolute_price": float(
            np.log(abs(history.attrs.get("unadjusted_signal_price", price.iloc[-1])))
        ),
        "median_dollar_volume_60d": (
            float(np.median((price * volume).iloc[-60:]))
            if len(price) >= 60
            else np.nan
        ),
    }
    for window in (5, 63, 126, 252):
        values[f"return_total_{window}d"] = (
            float(np.prod(1.0 + total_return.iloc[-window:]) - 1.0)
            if len(total_return) >= window
            else np.nan
        )
    for window in (21, 63, 126, 252):
        values[f"realized_volatility_{window}d"] = (
            float(total_return.iloc[-window:].std(ddof=1) * np.sqrt(252.0))
            if len(total_return) >= window
            else np.nan
        )
    return values


def _monthly_pass(
    detail: pd.DataFrame,
    *,
    resolved_share: float,
    gate: dict[str, float],
) -> bool:
    if resolved_share < float(gate["minimum_resolved_instrument_share"]):
        return False
    valid = detail["spearman_rank_correlation"].notna()
    if not valid.all():
        return False
    return bool(
        (detail.loc[valid, "spearman_rank_correlation"]
         >= float(gate["minimum_spearman_rank_correlation"])).all()
        and (detail.loc[valid, "median_absolute_rank_difference"]
             <= float(gate["maximum_median_absolute_rank_difference"])).all()
        and (detail["absolute_missingness_difference"]
             <= float(gate["maximum_absolute_missingness_difference"])).all()
    )


def run_lseg_market_parity(
    *,
    config_path: Path,
    output: Path,
    archive_months: Path,
    model_feature_months: Path,
) -> dict[str, Any]:
    """Run or resume the configured full-window market parity study."""

    try:
        import lseg.data as ld
    except ImportError as error:  # pragma: no cover - environment-specific
        raise LsegMarketParityError(
            "run with /opt/anaconda3/envs/finance/bin/python"
        ) from error

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "research_v6_8_lseg_market_parity_v1":
        raise LsegMarketParityError("market parity config schema differs")
    months = _months(config["holding_month_start"], config["holding_month_end"])
    features = [str(value) for value in config["features"]]
    sample_size = int(config["sample_size_per_month"])
    output.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    monthly_receipts: list[dict[str, Any]] = []

    ld.open_session()
    try:
        for month in months:
            month_output = output / "months" / month
            archive_month = archive_months / f"{month}.parquet"
            model_feature_month = model_feature_months / f"{month}.parquet"
            if not archive_month.exists() or not model_feature_month.exists():
                raise LsegMarketParityError(f"missing input month {month}")
            identity = _identity(
                config_path=config_path,
                archive_month=archive_month,
                model_feature_month=model_feature_month,
            )
            receipt_path = month_output / "receipt.json"
            if receipt_path.exists():
                existing = json.loads(receipt_path.read_text(encoding="utf-8"))
                if existing.get("identity") == identity and existing.get("status") in {
                    "PASS", "FAIL"
                }:
                    monthly_receipts.append(existing)
                    completed.append(month)
                    continue

            _atomic_json(
                output / "progress.json",
                {
                    "schema": "research_v6_8_lseg_market_parity_progress_v1",
                    "status": "RUNNING",
                    "updated_at_utc": _utc_now(),
                    "completed_months": completed,
                    "current_month": month,
                },
            )
            archive = pd.read_parquet(archive_month)
            model = pd.read_parquet(
                model_feature_month,
                columns=["permno", *features],
            )
            sample = _sample_population(
                archive,
                model,
                features=features,
                sample_size=sample_size,
            )
            signal_dates = sample["signal_date"].astype(str).unique().tolist()
            if len(signal_dates) != 1:
                raise LsegMarketParityError(f"{month} has multiple signal dates")
            signal_date = signal_dates[0]
            start = (
                pd.Timestamp(signal_date) - timedelta(days=520)
            ).date().isoformat()
            bridge_rows: list[dict[str, Any]] = []
            mapping_rows: list[dict[str, Any]] = []
            for _, row in sample.iterrows():
                try:
                    resolved = _consolidated_ric_history(
                        ld,
                        ticker=str(row["ticker"]).strip(),
                        cusip9=str(row["cusip9"]).strip(),
                        expected_signal_price=float(np.exp(row["log_absolute_price"])),
                        start=start,
                        end=signal_date,
                    )
                except Exception as error:  # pragma: no cover - live provider path
                    mapping_rows.append(
                        {
                            "permno": int(row["permno"]),
                            "ticker": str(row["ticker"]),
                            "mapping_status": "REQUEST_ERROR",
                            "consolidated_ric": None,
                            "error_type": type(error).__name__,
                        }
                    )
                    continue
                if resolved is None:
                    mapping_rows.append(
                        {
                            "permno": int(row["permno"]),
                            "ticker": str(row["ticker"]),
                            "mapping_status": "UNRESOLVED",
                            "consolidated_ric": None,
                            "error_type": None,
                        }
                    )
                    continue
                ric, history = resolved
                try:
                    bridge_rows.append(_bridge_values(int(row["permno"]), history))
                    mapping_rows.append(
                        {
                            "permno": int(row["permno"]),
                            "ticker": str(row["ticker"]),
                            "mapping_status": "CONSOLIDATED_RIC_RESOLVED",
                            "consolidated_ric": ric,
                            "history_identifier": history.attrs.get(
                                "lseg_history_identifier", ric
                            ),
                            "error_type": None,
                        }
                    )
                except LsegMarketParityError as error:
                    mapping_rows.append(
                        {
                            "permno": int(row["permno"]),
                            "ticker": str(row["ticker"]),
                            "mapping_status": "INSUFFICIENT_HISTORY",
                            "consolidated_ric": ric,
                            "error_type": type(error).__name__,
                        }
                    )
            mapping = pd.DataFrame(mapping_rows)
            bridge = pd.DataFrame(bridge_rows)
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
                    sample[["permno", *features]],
                    bridge[["permno", *features]],
                    features,
                )
            resolved_count = int(
                mapping["mapping_status"].eq("CONSOLIDATED_RIC_RESOLVED").sum()
            )
            resolved_share = resolved_count / sample_size
            passed = _monthly_pass(
                detail,
                resolved_share=resolved_share,
                gate=config["monthly_gate"],
            )
            _atomic_csv(month_output / "feature_metrics.csv", detail)
            _atomic_csv(month_output / "identifier_metrics.csv", mapping)
            receipt = {
                "schema": "research_v6_8_lseg_market_parity_month_v1",
                "created_at_utc": _utc_now(),
                "status": "PASS" if passed else "FAIL",
                "holding_month": month,
                "signal_date": signal_date,
                "identity": identity,
                "sample_size": sample_size,
                "resolved_instruments": resolved_count,
                "resolved_instrument_share": resolved_share,
                "passing_features": int(detail["pilot_gate_pass"].sum()),
                "compared_features": len(features),
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
            monthly_receipts.append(receipt)
            completed.append(month)
    finally:
        ld.close_session()

    monthly = pd.DataFrame(monthly_receipts).sort_values("holding_month")
    passing_month_share = float(monthly["status"].eq("PASS").mean())
    resolved_share = float(monthly["resolved_instruments"].sum() / (len(monthly) * sample_size))
    full_gate = config["full_window_gate"]
    passed = bool(
        passing_month_share >= float(full_gate["minimum_passing_month_share"])
        and resolved_share
        >= float(full_gate["minimum_all_month_resolved_instrument_share"])
    )
    _atomic_csv(output / "monthly_summary.csv", monthly)
    receipt = {
        "schema": "research_v6_8_lseg_market_parity_v1",
        "created_at_utc": _utc_now(),
        "status": "PASS" if passed else "FAIL",
        "config_sha256": _sha256(config_path),
        "code_sha256": _sha256(Path(__file__)),
        "resolver_code_sha256": _sha256(
            Path(__file__).with_name("lseg_production_probe.py")
        ),
        "months_expected": len(months),
        "months_completed": len(monthly),
        "months_passing": int(monthly["status"].eq("PASS").sum()),
        "passing_month_share": passing_month_share,
        "all_month_resolved_instrument_share": resolved_share,
        "scope": "10 representative market primitives plus consolidated-instrument bridge",
        "remaining_scope": "41 derived market-family features, fundamentals, identity, ownership, options, factors, and current signal materialization",
    }
    _atomic_json(output / "receipt.json", receipt)
    _atomic_json(
        output / "progress.json",
        {
            "schema": "research_v6_8_lseg_market_parity_progress_v1",
            "status": "COMPLETE",
            "updated_at_utc": _utc_now(),
            "completed_months": months,
        },
    )
    return receipt


__all__ = [
    "LsegMarketParityError",
    "run_lseg_market_parity",
]
