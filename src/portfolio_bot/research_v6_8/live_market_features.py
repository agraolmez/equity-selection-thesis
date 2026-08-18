"""Checkpointed current LSEG market features for V6.8 paper trading."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from portfolio_bot.research_v6_8.lseg_market_derived_parity import (
    DEFAULT_FIELDS,
    RAW_FIELDS,
    _batch_member,
    _derived_bridge_values,
    _request_history,
)


class LiveMarketFeatureError(RuntimeError):
    """The current market feature snapshot failed a required gate."""


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


def _chunk_identity(config_path: Path, universe_path: Path) -> dict[str, str]:
    return {
        "config_sha256": _sha256(config_path),
        "universe_sha256": _sha256(universe_path),
        "code_sha256": _sha256(Path(__file__)),
    }


def apply_market_feature_policy(
    rows: pd.DataFrame,
    *,
    passing_features: list[str],
    provider_missing_features: list[str],
    pending_cross_sectional_features: list[str],
) -> pd.DataFrame:
    """Keep certified formulas and represent every unsupported formula as missing."""

    expected = [
        *passing_features,
        *provider_missing_features,
        *pending_cross_sectional_features,
    ]
    if len(expected) != len(set(expected)):
        raise LiveMarketFeatureError("market feature disposition overlaps")
    missing_passing = sorted(set(passing_features) - set(rows.columns))
    if missing_passing:
        raise LiveMarketFeatureError(
            f"certified market formulas were not materialized: {', '.join(missing_passing)}"
        )
    result = rows.copy()
    for name in [*provider_missing_features, *pending_cross_sectional_features]:
        result[name] = np.nan
    return result


def apply_frozen_eligibility(
    rows: pd.DataFrame,
    *,
    minimum_price_usd: float,
    minimum_market_cap_usd: float,
    minimum_median_dollar_volume_60d: float,
    minimum_positive_volume_share_60d: float,
    minimum_usable_daily_returns: int,
) -> pd.Series:
    """Return the exact production version of the frozen long-eligibility screen."""

    required = {
        "signal_price_usd",
        "issue_market_cap_usd",
        "median_dollar_volume_60d",
        "zero_volume_share_60d",
        "usable_daily_returns",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise LiveMarketFeatureError(
            f"eligibility inputs are missing: {', '.join(missing)}"
        )
    positive_volume_share = 1.0 - pd.to_numeric(
        rows["zero_volume_share_60d"], errors="coerce"
    )
    return (
        pd.to_numeric(rows["signal_price_usd"], errors="coerce").ge(
            float(minimum_price_usd)
        )
        & pd.to_numeric(rows["issue_market_cap_usd"], errors="coerce").ge(
            float(minimum_market_cap_usd)
        )
        & pd.to_numeric(rows["median_dollar_volume_60d"], errors="coerce").ge(
            float(minimum_median_dollar_volume_60d)
        )
        & positive_volume_share.ge(float(minimum_positive_volume_share_60d))
        & pd.to_numeric(rows["usable_daily_returns"], errors="coerce").ge(
            int(minimum_usable_daily_returns)
        )
    )


def _history_metadata(
    history: pd.DataFrame, raw_history: pd.DataFrame
) -> dict[str, Any]:
    core_columns = ["Price Close", "Total Return", "Accumulated Volume"]
    core = history.reindex(columns=core_columns).apply(pd.to_numeric, errors="coerce")
    trading = history.loc[core.notna().all(axis=1)].copy()
    raw_price = pd.to_numeric(
        raw_history.get("TRDPRC_1", pd.Series(dtype="float64")), errors="coerce"
    ).dropna()
    return {
        "usable_daily_returns": int(
            pd.to_numeric(trading.get("Total Return"), errors="coerce").notna().sum()
        ),
        "signal_price_usd": (
            float(raw_price.iloc[-1]) if not raw_price.empty else np.nan
        ),
        "latest_market_session": (
            pd.Timestamp(trading.index.max()).date().isoformat()
            if not trading.empty
            else None
        ),
    }


def _daily_history_frame(
    history: pd.DataFrame,
    *,
    live_security_id: str,
    symbol: str,
    ric: str,
) -> pd.DataFrame:
    """Retain the causal daily inputs needed by downstream factor exposures."""

    values = pd.DataFrame(
        {
            "price_close": pd.to_numeric(
                history.get("Price Close"), errors="coerce"
            ),
            "total_return": pd.to_numeric(
                history.get("Total Return"), errors="coerce"
            )
            / 100.0,
            "volume": pd.to_numeric(
                history.get("Accumulated Volume"), errors="coerce"
            ),
        },
        index=history.index,
    ).dropna()
    values.index = pd.to_datetime(values.index, errors="coerce")
    values = values.loc[values.index.notna()].copy()
    values.index.name = "observation_date"
    values = values.reset_index()
    values.insert(0, "ric", str(ric))
    values.insert(0, "symbol", str(symbol))
    values.insert(0, "live_security_id", str(live_security_id))
    return values


def materialize_live_market_features(
    *,
    config_path: Path,
    universe_path: Path,
    output: Path,
    maximum_securities: int | None = None,
) -> dict[str, Any]:
    """Build or resume one current market-family feature snapshot."""

    try:
        import lseg.data as ld
    except ImportError as error:  # pragma: no cover - environment specific
        raise LiveMarketFeatureError(
            "run with /opt/anaconda3/envs/finance/bin/python"
        ) from error

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "research_v6_8_live_market_features_v1":
        raise LiveMarketFeatureError("live-market config schema differs")
    universe = pd.read_parquet(universe_path)
    if maximum_securities is not None:
        universe = universe.head(int(maximum_securities)).copy()
    if universe.empty or universe["live_security_id"].duplicated().any():
        raise LiveMarketFeatureError("live universe is empty or has duplicate security IDs")
    signal_dates = pd.to_datetime(universe["signal_date"], errors="raise").dt.date
    if signal_dates.nunique() != 1:
        raise LiveMarketFeatureError("live universe must have one signal date")
    signal_date = signal_dates.iloc[0]
    start_date = signal_date - timedelta(days=int(config["history_calendar_days"]))
    identity = _chunk_identity(config_path, universe_path)
    batch_size = int(config["batch_size"])
    chunk_root = output / "chunks"
    history_root = output / "history_chunks"
    output.mkdir(parents=True, exist_ok=True)

    completed: list[int] = []
    ld.open_session()
    try:
        for chunk_number, start in enumerate(range(0, len(universe), batch_size), start=1):
            chunk = universe.iloc[start : start + batch_size].copy()
            checkpoint = chunk_root / f"{chunk_number:04d}.parquet"
            history_checkpoint = history_root / f"{chunk_number:04d}.parquet"
            checkpoint_receipt = chunk_root / f"{chunk_number:04d}.json"
            if (
                checkpoint.exists()
                and history_checkpoint.exists()
                and checkpoint_receipt.exists()
            ):
                prior = json.loads(checkpoint_receipt.read_text(encoding="utf-8"))
                if (
                    prior.get("identity") == identity
                    and prior.get("output_sha256") == _sha256(checkpoint)
                    and prior.get("history_sha256")
                    == _sha256(history_checkpoint)
                    and prior.get("input_rows") == len(chunk)
                ):
                    completed.append(chunk_number)
                    continue
            _atomic_json(
                output / "progress.json",
                {
                    "schema": "research_v6_8_live_market_progress_v1",
                    "status": "RUNNING",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_chunks": completed,
                    "current_chunk": chunk_number,
                    "total_chunks": int(np.ceil(len(universe) / batch_size)),
                },
            )
            # Quote PermID is the stable security identity, but LSEG's
            # unadjusted interday fields resolve on the consolidated RIC.
            identifiers = chunk["RIC"].astype(str).tolist()
            adjusted_batch: pd.DataFrame | None = None
            raw_batch: pd.DataFrame | None = None
            try:
                adjusted_batch = _request_history(
                    ld,
                    identifiers,
                    DEFAULT_FIELDS,
                    start=start_date.isoformat(),
                    end=signal_date.isoformat(),
                )
                raw_batch = _request_history(
                    ld,
                    identifiers,
                    RAW_FIELDS,
                    start=start_date.isoformat(),
                    end=signal_date.isoformat(),
                    adjustments="unadjusted",
                )
            except Exception:
                adjusted_batch = None
                raw_batch = None
            rows: list[dict[str, Any]] = []
            daily_histories: list[pd.DataFrame] = []
            errors: list[dict[str, Any]] = []
            for row in chunk.itertuples(index=False):
                identifier = str(row.RIC)
                try:
                    adjusted = (
                        None
                        if adjusted_batch is None
                        else _batch_member(adjusted_batch, identifier)
                    )
                    raw = None if raw_batch is None else _batch_member(raw_batch, identifier)
                    if adjusted is None:
                        adjusted = _request_history(
                            ld,
                            identifier,
                            DEFAULT_FIELDS,
                            start=start_date.isoformat(),
                            end=signal_date.isoformat(),
                        )
                    if raw is None:
                        raw = _request_history(
                            ld,
                            identifier,
                            RAW_FIELDS,
                            start=start_date.isoformat(),
                            end=signal_date.isoformat(),
                            adjustments="unadjusted",
                        )
                    values = _derived_bridge_values(0, adjusted, raw)
                    values.pop("permno", None)
                    values.update(_history_metadata(adjusted, raw))
                    values.update(
                        {
                            "live_security_id": str(row.live_security_id),
                            "symbol": str(row.TickerSymbol),
                            "ric": str(row.RIC),
                            "issuer_key": str(row.issuer_key),
                            "issue_market_cap_usd": float(row.MktCapIssueUsd),
                        }
                    )
                    rows.append(values)
                    daily_histories.append(
                        _daily_history_frame(
                            adjusted,
                            live_security_id=str(row.live_security_id),
                            symbol=str(row.TickerSymbol),
                            ric=str(row.RIC),
                        )
                    )
                except Exception as error:  # pragma: no cover - provider path
                    errors.append(
                        {
                            "live_security_id": str(row.live_security_id),
                            "symbol": str(row.TickerSymbol),
                            "ric": str(row.RIC),
                            "issuer_key": str(row.issuer_key),
                            "issue_market_cap_usd": float(row.MktCapIssueUsd),
                            "error_type": type(error).__name__,
                            "error": str(error)[:500],
                        }
                    )
            result = pd.DataFrame(rows)
            if errors and result.empty:
                # Retain the failed keys so the final resolved-share gate can
                # fail transparently instead of collapsing to a zero-row file.
                result = pd.DataFrame(errors)[
                    [
                        "live_security_id",
                        "symbol",
                        "ric",
                        "issuer_key",
                        "issue_market_cap_usd",
                    ]
                ].copy()
            elif errors:
                error_frame = pd.DataFrame(errors)[
                    [
                        "live_security_id",
                        "symbol",
                        "ric",
                        "issuer_key",
                        "issue_market_cap_usd",
                    ]
                ].copy()
                for column in result.columns:
                    if column not in error_frame:
                        error_frame[column] = np.nan
                result = pd.concat(
                    [result, error_frame[result.columns]], ignore_index=True
                )
            daily_history = (
                pd.concat(daily_histories, ignore_index=True)
                if daily_histories
                else pd.DataFrame(
                    columns=[
                        "live_security_id",
                        "symbol",
                        "ric",
                        "observation_date",
                        "price_close",
                        "total_return",
                        "volume",
                    ]
                )
            )
            _atomic_parquet(checkpoint, result)
            _atomic_parquet(history_checkpoint, daily_history)
            _atomic_json(
                checkpoint_receipt,
                {
                    "schema": "research_v6_8_live_market_chunk_v1",
                    "identity": identity,
                    "chunk": chunk_number,
                    "input_rows": int(len(chunk)),
                    "resolved_rows": int(len(rows)),
                    "error_rows": int(len(errors)),
                    "output_sha256": _sha256(checkpoint),
                    "history_rows": int(len(daily_history)),
                    "history_sha256": _sha256(history_checkpoint),
                },
            )
            completed.append(chunk_number)
    finally:
        ld.close_session()

    parts = [pd.read_parquet(path) for path in sorted(chunk_root.glob("*.parquet"))]
    materialized = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if materialized.empty:
        raise LiveMarketFeatureError("no current market histories were materialized")
    if len(materialized) != len(universe):
        raise LiveMarketFeatureError(
            "market output row count differs from the live-universe grain"
        )
    if materialized["live_security_id"].duplicated().any():
        raise LiveMarketFeatureError(
            "market output contains duplicate live security identities"
        )
    materialized = apply_market_feature_policy(
        materialized,
        passing_features=list(config["feature_disposition"]["passing"]),
        provider_missing_features=list(
            config["feature_disposition"]["provider_mismatch_missing"]
        ),
        pending_cross_sectional_features=list(
            config["feature_disposition"]["cross_sectional_pending_missing"]
        ),
    )
    screen = config["eligibility"]
    materialized["long_eligible"] = apply_frozen_eligibility(
        materialized,
        minimum_price_usd=float(screen["minimum_price_usd"]),
        minimum_market_cap_usd=float(screen["minimum_market_cap_usd"]),
        minimum_median_dollar_volume_60d=float(
            screen["minimum_median_dollar_volume_60d"]
        ),
        minimum_positive_volume_share_60d=float(
            screen["minimum_positive_volume_share_60d"]
        ),
        minimum_usable_daily_returns=int(screen["minimum_usable_daily_returns"]),
    )
    ordered_features = [
        *config["feature_disposition"]["passing"],
        *config["feature_disposition"]["provider_mismatch_missing"],
        *config["feature_disposition"]["cross_sectional_pending_missing"],
    ]
    controls = [
        "live_security_id",
        "symbol",
        "ric",
        "issuer_key",
        "issue_market_cap_usd",
        "signal_price_usd",
        "usable_daily_returns",
        "latest_market_session",
        "long_eligible",
    ]
    materialized = materialized[[*controls, *ordered_features]].sort_values(
        ["long_eligible", "issue_market_cap_usd", "live_security_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    data_path = output / "market_features.parquet"
    _atomic_parquet(data_path, materialized)
    history_parts = [
        pd.read_parquet(path) for path in sorted(history_root.glob("*.parquet"))
    ]
    daily_history = (
        pd.concat(history_parts, ignore_index=True)
        if history_parts
        else pd.DataFrame()
    )
    if daily_history.empty:
        raise LiveMarketFeatureError("no daily market history was preserved")
    if daily_history.duplicated(["live_security_id", "observation_date"]).any():
        raise LiveMarketFeatureError(
            "daily market history contains duplicate security-session keys"
        )
    history_path = output / "daily_market_history.parquet"
    _atomic_parquet(history_path, daily_history)
    latest_sessions = materialized["latest_market_session"].dropna().astype(str)
    receipt = {
        "schema": "research_v6_8_live_market_features_receipt_v1",
        "status": "PASS" if materialized["long_eligible"].any() else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_date": signal_date.isoformat(),
        "input_rows": int(len(universe)),
        "materialized_rows": int(len(materialized)),
        "eligible_rows": int(materialized["long_eligible"].sum()),
        "resolved_row_share": float(materialized["latest_market_session"].notna().mean()),
        "latest_market_session": latest_sessions.max() if not latest_sessions.empty else None,
        "certified_feature_count": int(len(config["feature_disposition"]["passing"])),
        "provider_missing_feature_count": int(
            len(config["feature_disposition"]["provider_mismatch_missing"])
        ),
        "pending_cross_sectional_feature_count": int(
            len(config["feature_disposition"]["cross_sectional_pending_missing"])
        ),
        "identity": identity,
        "output_sha256": _sha256(data_path),
        "daily_history_rows": int(len(daily_history)),
        "daily_history_sha256": _sha256(history_path),
    }
    _atomic_json(output / "receipt.json", receipt)
    _atomic_json(
        output / "progress.json",
        {
            "schema": "research_v6_8_live_market_progress_v1",
            "status": "COMPLETE",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_chunks": completed,
        },
    )
    return receipt


__all__ = [
    "LiveMarketFeatureError",
    "apply_frozen_eligibility",
    "apply_market_feature_policy",
    "materialize_live_market_features",
]
