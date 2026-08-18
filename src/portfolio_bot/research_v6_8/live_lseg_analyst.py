"""Checkpointed current LSEG analyst/CAM features for V6.8 paper trading."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ESTIMATE_BASES = ("Revenue", "GrossIncome", "PreTaxProfit", "NetProfit", "EPS", "DPS")
HORIZONS = ("FY1", "FY2", "FQ1")
SIGNAL_FIELDS = (
    "TR.RecMean.Date",
    "TR.RecMean",
    "TR.RecMedian",
    "TR.RecHigh",
    "TR.RecLow",
    "TR.NumOfRecommendations",
    "TR.PriceTargetMean.Date",
    "TR.PriceTargetMean",
    "TR.PriceTargetMedian",
    "TR.PriceTargetHigh",
    "TR.PriceTargetLow",
    "TR.PriceTargetNumIncEstimates",
    "TR.CombinedAlphaCountryRank.Date",
    "TR.CombinedAlphaCountryRank",
)
INT64_NULL = np.iinfo(np.int64).min


class LiveLsegAnalystError(RuntimeError):
    """The same-source LSEG analyst snapshot failed its production gate."""


def estimate_fields(horizon: str) -> list[str]:
    fields = [f"TR.EPSMean(Period={horizon}).CalcDate"]
    for base in ESTIMATE_BASES:
        fields.extend(
            [
                f"TR.{base}Mean(Period={horizon})",
                f"TR.{base}NumIncEstimates(Period={horizon})",
            ]
        )
    for base in ("EPS", "Revenue"):
        fields.extend(
            [
                f"TR.{base}Median(Period={horizon})",
                f"TR.{base}High(Period={horizon})",
                f"TR.{base}Low(Period={horizon})",
                f"TR.{base}StdDev(Period={horizon})",
            ]
        )
    fields.extend(
        [
            f"TR.EpsSmartEst(Period={horizon})",
            f"TR.EpsSENumIncEst(Period={horizon})",
            f"TR.RevenueSmartEst(Period={horizon})",
        ]
    )
    return fields


def currency_fields() -> list[str]:
    fields: list[str] = []
    for horizon in HORIZONS:
        fields.extend(
            f"TR.{base}(Period={horizon})"
            for base in (
                "RevenueMean",
                "GrossIncomeMean",
                "PreTaxProfitMean",
                "NetProfitMean",
                "EPSMean",
                "DPSMean",
                "EPSMedian",
                "EPSHigh",
                "EPSLow",
                "EPSStdDev",
                "RevenueMedian",
                "RevenueHigh",
                "RevenueLow",
                "RevenueStdDev",
                "EpsSmartEst",
                "RevenueSmartEst",
            )
        )
    fields.extend(
        ("TR.PriceTargetMean", "TR.PriceTargetMedian", "TR.PriceTargetHigh", "TR.PriceTargetLow")
    )
    return fields


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
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def normalize_lseg_frame(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Reproduce the historical V6 LSEG header normalization exactly."""

    if frame.empty:
        return pd.DataFrame(columns=["address_key"])
    result = frame.copy()
    instrument = next(
        (column for column in result if str(column).upper() == "INSTRUMENT"),
        result.columns[0],
    )
    result = result.rename(columns={instrument: "address_key"})
    result["address_key"] = result["address_key"].astype("string").str.strip()
    names: list[str] = []
    seen: dict[str, int] = {}
    for column in result:
        if column == "address_key":
            names.append(column)
            continue
        base = str(column).strip().lower()
        for old, new in ((".", "_"), ("(", "_"), (")", ""), ("=", "_"), (",", "_"), (" ", "_")):
            base = base.replace(old, new)
        name = f"{prefix}__{base}"
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    result.columns = names
    if result["address_key"].duplicated().any():
        raise LiveLsegAnalystError("LSEG analyst response duplicates an address key")
    return result


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    left = pd.to_numeric(numerator, errors="coerce").to_numpy(float)
    right = pd.to_numeric(denominator, errors="coerce").to_numpy(float)
    output = np.full(len(left), np.nan, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right) & (right != 0.0)
    output[valid] = left[valid] / right[valid]
    return pd.Series(output, index=numerator.index)


def transform_lseg_analyst_features(
    *,
    universe: pd.DataFrame,
    raw: pd.DataFrame,
    usd: pd.DataFrame,
    registry: pd.DataFrame,
    market_features: pd.DataFrame,
    signal_date: date,
    history: Mapping[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Map current LSEG responses to the exact 97-feature registry block."""

    merged = universe[["live_security_id", "RIC"]].rename(columns={"RIC": "address_key"})
    merged = merged.merge(raw, on="address_key", how="left", validate="one_to_one")
    merged = merged.merge(usd, on="address_key", how="left", validate="one_to_one")
    output = merged[["live_security_id"]].copy()
    direct = registry.loc[registry["model_ordinal"].between(159, 247)].copy()
    if len(direct) != 89:
        raise LiveLsegAnalystError("frozen direct LSEG registry block differs")
    for row in direct.itertuples(index=False):
        name = str(row.feature_name)
        if "rank_change_" in name:
            output[name] = np.nan
            continue
        source = str(row.source_column)
        if source not in merged:
            raise LiveLsegAnalystError(f"LSEG response omitted frozen source column {source}")
        values = merged[source]
        if str(row.transformation) == "signal_date_minus_vendor_date_days":
            timestamps = pd.to_datetime(values, errors="coerce")
            age = (pd.Timestamp(signal_date) - timestamps).dt.total_seconds() / 86_400.0
            if age.dropna().lt(0).any():
                raise LiveLsegAnalystError(f"future LSEG date escaped for {name}")
            output[name] = age.astype(float)
        else:
            numeric = pd.to_numeric(values, errors="coerce").astype(float)
            output[name] = numeric.mask(numeric.eq(float(INT64_NULL)))
    market = market_features[["live_security_id", "signal_price_usd"]]
    work = output.merge(market, on="live_security_id", how="left", validate="one_to_one")
    price = pd.to_numeric(work["signal_price_usd"], errors="coerce").abs()
    output["lseg_forward_eps_yield_fy1"] = _safe_ratio(work["lseg_epsmean_fy1"], price)
    output["lseg_forward_eps_growth_fy2_vs_fy1"] = _safe_ratio(
        pd.to_numeric(work["lseg_epsmean_fy2"], errors="coerce")
        - pd.to_numeric(work["lseg_epsmean_fy1"], errors="coerce"),
        pd.to_numeric(work["lseg_epsmean_fy1"], errors="coerce").abs(),
    )
    output["lseg_forward_revenue_growth_fy2_vs_fy1"] = _safe_ratio(
        pd.to_numeric(work["lseg_revenuemean_fy2"], errors="coerce")
        - pd.to_numeric(work["lseg_revenuemean_fy1"], errors="coerce"),
        pd.to_numeric(work["lseg_revenuemean_fy1"], errors="coerce").abs(),
    )
    output["lseg_price_target_upside"] = _safe_ratio(
        work["lseg_pricetargetmean"], price
    ) - 1.0
    history = history or {}
    current = output.set_index("live_security_id")
    for lag in (1, 3):
        prior = history.get(lag)
        if prior is None:
            output[f"lseg_price_target_revision_{lag}m"] = np.nan
            output[f"lseg_recommendation_upgrade_{lag}m"] = np.nan
            continue
        prior = prior.set_index("live_security_id")
        prior_target = pd.to_numeric(
            prior.reindex(current.index)["lseg_pricetargetmean"], errors="coerce"
        )
        prior_rec = pd.to_numeric(
            prior.reindex(current.index)["lseg_recmean"], errors="coerce"
        )
        target = pd.to_numeric(current["lseg_pricetargetmean"], errors="coerce")
        rec = pd.to_numeric(current["lseg_recmean"], errors="coerce")
        output[f"lseg_price_target_revision_{lag}m"] = _safe_ratio(
            target - prior_target, prior_target.abs()
        ).to_numpy()
        output[f"lseg_recommendation_upgrade_{lag}m"] = (prior_rec - rec).to_numpy()
    expected = registry.loc[
        registry["model_ordinal"].between(159, 247)
        | registry["model_ordinal"].between(307, 314),
        "feature_name",
    ].astype(str).tolist()
    if len(expected) != 97 or set(expected) - set(output):
        raise LiveLsegAnalystError("current LSEG feature block is incomplete")
    result = output[["live_security_id", *expected]]
    if len(result) != len(universe) or result["live_security_id"].duplicated().any():
        raise LiveLsegAnalystError("current LSEG feature grain differs from the live universe")
    return result


def _request_with_retry(
    ld: Any,
    instruments: list[str],
    fields: list[str],
    signal_date: str,
    *,
    currency: str | None = None,
) -> pd.DataFrame:
    parameters = {"SDate": signal_date, "EDate": signal_date, "Frq": "D"}
    if currency:
        parameters["Curn"] = currency
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            return ld.get_data(
                instruments,
                fields,
                parameters=parameters,
                header_type=ld.HeaderType.NAME,
            )
        except Exception as error:  # pragma: no cover - provider path
            last_error = error
            if attempt < 4:
                time.sleep(min(30, 2**attempt))
    raise LiveLsegAnalystError(f"LSEG analyst request failed: {last_error}")


def materialize_live_lseg_analyst_features(
    *,
    config_path: Path,
    universe_path: Path,
    market_features_path: Path,
    registry_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Download, checkpoint, and publish one current LSEG analyst block."""

    try:
        import lseg.data as ld
    except ImportError as error:  # pragma: no cover - environment specific
        raise LiveLsegAnalystError("run with the finance Python environment") from error
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "research_v6_8_live_lseg_analyst_features_v1":
        raise LiveLsegAnalystError("live analyst config schema differs")
    universe = pd.read_parquet(universe_path)
    market = pd.read_parquet(market_features_path)
    registry = pd.read_parquet(registry_path)
    signal_dates = pd.to_datetime(universe["signal_date"], errors="raise").dt.date
    if signal_dates.nunique() != 1:
        raise LiveLsegAnalystError("live analyst universe must have one signal date")
    signal_date = signal_dates.iloc[0]
    identity = {
        "config_sha256": _sha256(config_path),
        "universe_sha256": _sha256(universe_path),
        "market_sha256": _sha256(market_features_path),
        "registry_sha256": _sha256(registry_path),
        "code_sha256": _sha256(Path(__file__)),
    }
    batch_size = int(config["batch_size"])
    chunk_root = output / "chunks"
    completed: list[int] = []
    ld.open_session()
    try:
        for chunk_number, start in enumerate(range(0, len(universe), batch_size), start=1):
            base = universe.iloc[start : start + batch_size].copy()
            checkpoint = chunk_root / f"{chunk_number:04d}.parquet"
            checkpoint_receipt = chunk_root / f"{chunk_number:04d}.json"
            if checkpoint.exists() and checkpoint_receipt.exists():
                prior = json.loads(checkpoint_receipt.read_text(encoding="utf-8"))
                if (
                    prior.get("identity") == identity
                    and prior.get("output_sha256") == _sha256(checkpoint)
                    and prior.get("input_rows") == len(base)
                ):
                    completed.append(chunk_number)
                    continue
            _atomic_json(
                output / "progress.json",
                {
                    "schema": "research_v6_8_live_lseg_analyst_progress_v1",
                    "status": "RUNNING",
                    "completed_chunks": completed,
                    "current_chunk": chunk_number,
                    "total_chunks": int(np.ceil(len(universe) / batch_size)),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            instruments = base["RIC"].astype(str).tolist()
            merged_raw = base[["live_security_id", "RIC"]].rename(columns={"RIC": "address_key"})
            for horizon in HORIZONS:
                response = normalize_lseg_frame(
                    _request_with_retry(
                        ld, instruments, estimate_fields(horizon), signal_date.isoformat()
                    ),
                    horizon.lower(),
                )
                merged_raw = merged_raw.merge(
                    response, on="address_key", how="left", validate="one_to_one"
                )
            signals = normalize_lseg_frame(
                _request_with_retry(
                    ld, instruments, list(SIGNAL_FIELDS), signal_date.isoformat()
                ),
                "signals",
            )
            merged_raw = merged_raw.merge(
                signals, on="address_key", how="left", validate="one_to_one"
            )
            usd = normalize_lseg_frame(
                _request_with_retry(
                    ld,
                    instruments,
                    currency_fields(),
                    signal_date.isoformat(),
                    currency=str(config["currency"]),
                ),
                "usd",
            )
            transformed = transform_lseg_analyst_features(
                universe=base,
                raw=merged_raw.drop(columns="live_security_id"),
                usd=usd,
                registry=registry,
                market_features=market.loc[
                    market["live_security_id"].isin(base["live_security_id"])
                ],
                signal_date=signal_date,
            )
            _atomic_parquet(checkpoint, transformed)
            _atomic_json(
                checkpoint_receipt,
                {
                    "schema": "research_v6_8_live_lseg_analyst_chunk_v1",
                    "identity": identity,
                    "chunk": chunk_number,
                    "input_rows": int(len(base)),
                    "output_sha256": _sha256(checkpoint),
                },
            )
            completed.append(chunk_number)
    finally:
        ld.close_session()
    parts = [pd.read_parquet(path) for path in sorted(chunk_root.glob("*.parquet"))]
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(result) != len(universe) or result["live_security_id"].duplicated().any():
        raise LiveLsegAnalystError("final current analyst feature grain differs")
    data_path = output / "analyst_features.parquet"
    _atomic_parquet(data_path, result)
    value_columns = [column for column in result if column != "live_security_id"]
    receipt = {
        "schema": "research_v6_8_live_lseg_analyst_receipt_v1",
        "status": "PASS",
        "signal_date": signal_date.isoformat(),
        "rows": int(len(result)),
        "feature_count": int(len(value_columns)),
        "rows_with_any_value": int(result[value_columns].notna().any(axis=1).sum()),
        "cam_coverage_share": float(result["lseg_cam_country_rank"].notna().mean()),
        "exact_lag_fields_missing_until_live_history_accumulates": [
            "lseg_cam_rank_change_1m",
            "lseg_cam_rank_change_3m",
            "lseg_cam_rank_change_6m",
            "lseg_price_target_revision_1m",
            "lseg_price_target_revision_3m",
            "lseg_recommendation_upgrade_1m",
            "lseg_recommendation_upgrade_3m",
        ],
        "identity": identity,
        "output_sha256": _sha256(data_path),
    }
    _atomic_json(output / "receipt.json", receipt)
    _atomic_json(
        output / "progress.json",
        {
            "schema": "research_v6_8_live_lseg_analyst_progress_v1",
            "status": "COMPLETE",
            "completed_chunks": completed,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return receipt


__all__ = [
    "LiveLsegAnalystError",
    "currency_fields",
    "estimate_fields",
    "materialize_live_lseg_analyst_features",
    "normalize_lseg_frame",
    "transform_lseg_analyst_features",
]
