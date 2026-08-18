"""Low-rate, resumable repair of missing V6.8 LSEG market chunks."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_bot.research_v6_8.live_market_features import (
    _atomic_json,
    _atomic_parquet,
    _batch_member,
    _chunk_identity,
    _daily_history_frame,
    _derived_bridge_values,
    _history_metadata,
    _request_history,
    _sha256,
)
from portfolio_bot.research_v6_8.lseg_market_derived_parity import (
    DEFAULT_FIELDS,
    RAW_FIELDS,
)


class LiveMarketRepairError(RuntimeError):
    """A repair unit could not be completed without weakening data quality."""


class LiveMarketRepairRateLimited(LiveMarketRepairError):
    """LSEG asked the repair process to stop and cool down."""


class LiveMarketRepairQuotaExhausted(LiveMarketRepairRateLimited):
    """The Workspace daily request allowance has no repair capacity left."""


def read_desktop_throttle_limits() -> dict[str, int]:
    """Read LSEG Workspace's local, credential-free usage-limit endpoint."""

    for port in (9000, 9001):
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/api/throttling/limits", timeout=3
            ) as response:
                payload = json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            continue
        return {
            key: int(payload[key])
            for key in (
                "daily_request_max",
                "daily_request_remaining",
                "request_max",
                "request_remaining",
                "daily_volume_max",
                "daily_volume_remaining",
            )
        }
    raise LiveMarketRepairError("LSEG Workspace throttle endpoint is unavailable")


def _code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _is_rate_limit(error: BaseException) -> bool:
    return "too many requests" in str(error).lower()


def _request(
    ld: Any,
    identifiers: str | list[str],
    fields: tuple[str, ...],
    *,
    start: str,
    end: str,
    adjustments: str | None = None,
    pause_seconds: float,
) -> pd.DataFrame:
    try:
        result = _request_history(
            ld,
            identifiers,
            fields,
            start=start,
            end=end,
            adjustments=adjustments,
        )
    except Exception as error:
        if _is_rate_limit(error):
            raise LiveMarketRepairRateLimited(
                "LSEG rate limit reached; repair stopped without publishing the active unit"
            ) from error
        raise
    time.sleep(max(float(pause_seconds), 0.0))
    return result


def repair_missing_market_chunks(
    *,
    config_path: Path,
    universe_path: Path,
    output: Path,
    repair_batch_size: int = 5,
    request_pause_seconds: float = 3.0,
) -> dict[str, Any]:
    """Repair only absent original chunks and preserve every clean checkpoint."""

    try:
        import lseg.data as ld
    except ImportError as error:  # pragma: no cover - environment specific
        raise LiveMarketRepairError(
            "run with /opt/anaconda3/envs/finance/bin/python"
        ) from error
    config = json.loads(config_path.read_text(encoding="utf-8"))
    universe = pd.read_parquet(universe_path)
    signal_dates = pd.to_datetime(universe["signal_date"], errors="raise").dt.date
    if signal_dates.nunique() != 1:
        raise LiveMarketRepairError("repair universe must contain one signal date")
    signal_date = signal_dates.iloc[0]
    start_date = signal_date - timedelta(days=int(config["history_calendar_days"]))
    original_batch_size = int(config["batch_size"])
    if repair_batch_size <= 0 or repair_batch_size > original_batch_size:
        raise LiveMarketRepairError("repair batch size is outside the safe interval")
    identity = _chunk_identity(config_path, universe_path)
    chunk_root = output / "chunks"
    history_root = output / "history_chunks"
    repair_root = output / "repair_checkpoints"
    total_chunks = int(np.ceil(len(universe) / original_batch_size))
    repaired_chunks: list[int] = []
    missing_chunks = [
        number
        for number in range(1, total_chunks + 1)
        if not all(
            path.is_file()
            for path in (
                chunk_root / f"{number:04d}.json",
                chunk_root / f"{number:04d}.parquet",
                history_root / f"{number:04d}.parquet",
            )
        )
    ]
    limits = read_desktop_throttle_limits()
    missing_rows = sum(
        len(
            universe.iloc[
                (number - 1) * original_batch_size : number * original_batch_size
            ]
        )
        for number in missing_chunks
    )
    minimum_request_budget = 2 * missing_rows + 50
    if limits["daily_request_remaining"] < minimum_request_budget:
        _atomic_json(
            output / "repair_progress.json",
            {
                "schema": "research_v6_8_live_market_repair_progress_v1",
                "status": "DAILY_QUOTA_EXHAUSTED",
                "missing_chunks": missing_chunks,
                "minimum_request_budget": minimum_request_budget,
                "daily_request_remaining": limits["daily_request_remaining"],
                "daily_request_max": limits["daily_request_max"],
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise LiveMarketRepairQuotaExhausted(
            "LSEG daily request allowance is insufficient for the remaining repair"
        )
    ld.open_session()
    try:
        for chunk_number in missing_chunks:
            start_position = (chunk_number - 1) * original_batch_size
            chunk = universe.iloc[
                start_position : start_position + original_batch_size
            ].copy()
            unit_root = repair_root / f"{chunk_number:04d}"
            results: list[pd.DataFrame] = []
            histories: list[pd.DataFrame] = []
            for unit_number, offset in enumerate(
                range(0, len(chunk), repair_batch_size), start=1
            ):
                unit = chunk.iloc[offset : offset + repair_batch_size].copy()
                unit_data = unit_root / f"{unit_number:04d}.parquet"
                unit_history = unit_root / f"{unit_number:04d}.history.parquet"
                unit_receipt = unit_root / f"{unit_number:04d}.json"
                if unit_data.is_file() and unit_history.is_file() and unit_receipt.is_file():
                    prior = json.loads(unit_receipt.read_text(encoding="utf-8"))
                    if (
                        prior.get("original_identity") == identity
                        and prior.get("repair_code_sha256") == _code_sha256()
                        and prior.get("output_sha256") == _sha256(unit_data)
                        and prior.get("history_sha256") == _sha256(unit_history)
                    ):
                        results.append(pd.read_parquet(unit_data))
                        histories.append(pd.read_parquet(unit_history))
                        continue
                _atomic_json(
                    output / "repair_progress.json",
                    {
                        "schema": "research_v6_8_live_market_repair_progress_v1",
                        "status": "RUNNING",
                        "chunk": chunk_number,
                        "unit": unit_number,
                        "missing_chunks": missing_chunks,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                identifiers = unit["RIC"].astype(str).tolist()
                try:
                    adjusted_batch = _request(
                        ld,
                        identifiers,
                        DEFAULT_FIELDS,
                        start=start_date.isoformat(),
                        end=signal_date.isoformat(),
                        pause_seconds=request_pause_seconds,
                    )
                    raw_batch = _request(
                        ld,
                        identifiers,
                        RAW_FIELDS,
                        start=start_date.isoformat(),
                        end=signal_date.isoformat(),
                        adjustments="unadjusted",
                        pause_seconds=request_pause_seconds,
                    )
                except LiveMarketRepairRateLimited:
                    _atomic_json(
                        output / "repair_progress.json",
                        {
                            "schema": "research_v6_8_live_market_repair_progress_v1",
                            "status": "RATE_LIMITED",
                            "chunk": chunk_number,
                            "unit": unit_number,
                            "missing_chunks": missing_chunks,
                            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    raise
                rows: list[dict[str, Any]] = []
                daily: list[pd.DataFrame] = []
                for row in unit.itertuples(index=False):
                    identifier = str(row.RIC)
                    adjusted = _batch_member(adjusted_batch, identifier)
                    raw = _batch_member(raw_batch, identifier)
                    repair_error: str | None = None
                    try:
                        if adjusted is None:
                            adjusted = _request(
                                ld,
                                identifier,
                                DEFAULT_FIELDS,
                                start=start_date.isoformat(),
                                end=signal_date.isoformat(),
                                pause_seconds=request_pause_seconds,
                            )
                        if raw is None:
                            raw = _request(
                                ld,
                                identifier,
                                RAW_FIELDS,
                                start=start_date.isoformat(),
                                end=signal_date.isoformat(),
                                adjustments="unadjusted",
                                pause_seconds=request_pause_seconds,
                            )
                        values = _derived_bridge_values(0, adjusted, raw)
                        values.pop("permno", None)
                        values.update(_history_metadata(adjusted, raw))
                        daily.append(
                            _daily_history_frame(
                                adjusted,
                                live_security_id=str(row.live_security_id),
                                symbol=str(row.TickerSymbol),
                                ric=identifier,
                            )
                        )
                    except LiveMarketRepairRateLimited:
                        raise
                    except Exception as error:  # provider-specific permanent absence
                        values = {}
                        repair_error = f"{type(error).__name__}: {str(error)[:300]}"
                    values.update(
                        {
                            "live_security_id": str(row.live_security_id),
                            "symbol": str(row.TickerSymbol),
                            "ric": identifier,
                            "issuer_key": str(row.issuer_key),
                            "issue_market_cap_usd": float(row.MktCapIssueUsd),
                            "repair_error": repair_error,
                        }
                    )
                    rows.append(values)
                unit_result = pd.DataFrame(rows)
                unit_daily = (
                    pd.concat(daily, ignore_index=True)
                    if daily
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
                _atomic_parquet(unit_data, unit_result)
                _atomic_parquet(unit_history, unit_daily)
                _atomic_json(
                    unit_receipt,
                    {
                        "schema": "research_v6_8_live_market_repair_unit_v1",
                        "original_identity": identity,
                        "repair_code_sha256": _code_sha256(),
                        "chunk": chunk_number,
                        "unit": unit_number,
                        "input_rows": len(unit),
                        "error_rows": int(unit_result["repair_error"].notna().sum()),
                        "output_sha256": _sha256(unit_data),
                        "history_sha256": _sha256(unit_history),
                    },
                )
                results.append(unit_result)
                histories.append(unit_daily)
            chunk_result = pd.concat(results, ignore_index=True)
            chunk_history = pd.concat(histories, ignore_index=True)
            if len(chunk_result) != len(chunk):
                raise LiveMarketRepairError("repaired chunk differs from original grain")
            checkpoint = chunk_root / f"{chunk_number:04d}.parquet"
            history_checkpoint = history_root / f"{chunk_number:04d}.parquet"
            checkpoint_receipt = chunk_root / f"{chunk_number:04d}.json"
            _atomic_parquet(checkpoint, chunk_result)
            _atomic_parquet(history_checkpoint, chunk_history)
            error_rows = int(chunk_result["repair_error"].notna().sum())
            _atomic_json(
                checkpoint_receipt,
                {
                    "schema": "research_v6_8_live_market_chunk_v1",
                    "identity": identity,
                    "chunk": chunk_number,
                    "input_rows": len(chunk),
                    "resolved_rows": len(chunk) - error_rows,
                    "error_rows": error_rows,
                    "output_sha256": _sha256(checkpoint),
                    "history_rows": len(chunk_history),
                    "history_sha256": _sha256(history_checkpoint),
                    "repair_code_sha256": _code_sha256(),
                    "repair_batch_size": repair_batch_size,
                },
            )
            repaired_chunks.append(chunk_number)
    finally:
        ld.close_session()
    receipt = {
        "schema": "research_v6_8_live_market_repair_receipt_v1",
        "status": "PASS",
        "repaired_chunks": repaired_chunks,
        "remaining_missing_chunks": [
            number
            for number in range(1, total_chunks + 1)
            if not (chunk_root / f"{number:04d}.json").is_file()
        ],
        "repair_code_sha256": _code_sha256(),
        "original_identity": identity,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output / "repair_receipt.json", receipt)
    return receipt


__all__ = [
    "LiveMarketRepairError",
    "LiveMarketRepairQuotaExhausted",
    "LiveMarketRepairRateLimited",
    "read_desktop_throttle_limits",
    "repair_missing_market_chunks",
]
