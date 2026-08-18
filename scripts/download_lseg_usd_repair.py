#!/usr/bin/env python3
"""Download the currency-bearing V6 LSEG fields in a common USD currency.

The completed ``raw/research_v6/lseg`` archive is preserved verbatim.  This
script creates a separate, resumable repair lineage containing only fields
whose cross-sectional levels are otherwise incomparable across reporting
currencies.  Every request uses the original historical CUSIP9, exact signal
date, and LSEG's ``Curn=USD`` parameter.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lseg.data as ld
import pandas as pd
import pyarrow.parquet as pq


DATABASE = Path("/Users/agraemirolmez/Desktop/Thesis")
SOURCE = DATABASE / "raw/research_v6/lseg"
OUTPUT = DATABASE / "raw/research_v6/lseg_usd_v1"
UNIVERSE = SOURCE / "input_universe.parquet"
CHUNK_SIZE = 1000
HORIZONS = ("FY1", "FY2", "FQ1")


def _currency_fields() -> list[str]:
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
        (
            "TR.PriceTargetMean",
            "TR.PriceTargetMedian",
            "TR.PriceTargetHigh",
            "TR.PriceTargetLow",
        )
    )
    return fields


FIELDS = _currency_fields()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["cusip9"])
    frame = frame.copy()
    instrument = next(
        (column for column in frame if str(column).upper() == "INSTRUMENT"),
        frame.columns[0],
    )
    frame = frame.rename(columns={instrument: "cusip9"})
    frame["cusip9"] = frame["cusip9"].astype("string").str.strip().str.upper()
    names: list[str] = []
    seen: dict[str, int] = {}
    for column in frame:
        if column == "cusip9":
            names.append(column)
            continue
        name = str(column).strip().lower()
        for old, new in ((".", "_"), ("(", "_"), (")", ""), ("=", "_"), (",", "_"), (" ", "_")):
            name = name.replace(old, new)
        name = f"usd__{name}"
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    frame.columns = names
    for column in frame.columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates("cusip9", keep="last")


def _request(instruments: list[str], signal_date: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            return ld.get_data(
                instruments,
                FIELDS,
                parameters={
                    "SDate": signal_date,
                    "EDate": signal_date,
                    "Frq": "D",
                    "Curn": "USD",
                },
                header_type=ld.HeaderType.NAME,
            )
        except Exception as error:
            last_error = error
            if attempt == 5:
                break
            time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"LSEG USD request failed after retries: {last_error}")


def _contract() -> dict[str, Any]:
    parent = json.loads((SOURCE / "contract.json").read_text())
    payload: dict[str, Any] = {
        "schema": "research_v6_lseg_usd_repair_v1",
        "source_contract_sha256": parent["contract_sha256"],
        "universe_sha256": _sha256(UNIVERSE),
        "addressing": "historical_cusip9",
        "cutoff": "exact_authenticated_signal_date",
        "currency": "USD",
        "fields": FIELDS,
        "chunk_size": CHUNK_SIZE,
        "lseg_data_version": getattr(ld, "__version__", "unknown"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["contract_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def _valid(month: str, contract_sha256: str) -> bool:
    data = OUTPUT / "months" / f"{month}.parquet"
    receipt = OUTPUT / "receipts" / f"{month}.json"
    if not data.is_file() or not receipt.is_file():
        return False
    try:
        value = json.loads(receipt.read_text())
        return (
            value.get("contract_sha256") == contract_sha256
            and value.get("output_sha256") == _sha256(data)
            and value.get("rows") == pq.ParquetFile(data).metadata.num_rows
        )
    except Exception:
        return False


def _download_month(frame: pd.DataFrame, contract_sha256: str) -> None:
    month = str(frame["holding_month"].iloc[0])
    dates = frame["signal_date"].astype(str).unique().tolist()
    if len(dates) != 1:
        raise RuntimeError(f"{month}: expected one signal date")
    signal_date = dates[0]
    parts: list[pd.DataFrame] = []
    total_chunks = (len(frame) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk, start in enumerate(range(0, len(frame), CHUNK_SIZE), start=1):
        base = frame.iloc[start : start + CHUNK_SIZE][
            ["holding_month", "signal_date", "permno", "permco", "cusip9"]
        ].copy()
        values = _normalize(_request(base["cusip9"].astype(str).tolist(), signal_date))
        parts.append(base.merge(values, on="cusip9", how="left", validate="one_to_one"))
        _atomic_json(
            OUTPUT / "progress.json",
            {
                "status": "RUNNING",
                "updated_at_utc": _utc_now(),
                "holding_month": month,
                "chunk": chunk,
                "chunks": total_chunks,
                "rows_completed_in_month": sum(len(part) for part in parts),
                "rows_in_month": len(frame),
            },
        )
    output = pd.concat(parts, ignore_index=True)
    if output.duplicated(["holding_month", "permno"]).any():
        raise RuntimeError(f"{month}: duplicate output keys")
    destination = OUTPUT / "months" / f"{month}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT / ".staging" / f"{month}.parquet.tmp"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    value_columns = [column for column in output if column.startswith("usd__")]
    _atomic_json(
        OUTPUT / "receipts" / f"{month}.json",
        {
            "status": "COMPLETE",
            "created_at_utc": _utc_now(),
            "holding_month": month,
            "signal_date": signal_date,
            "contract_sha256": contract_sha256,
            "rows": len(output),
            "value_columns": len(value_columns),
            "rows_with_any_usd_value": int(output[value_columns].notna().any(axis=1).sum()),
            "output_sha256": _sha256(destination),
        },
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = _contract()
    _atomic_json(OUTPUT / "contract.json", contract)
    universe = pd.read_parquet(UNIVERSE)
    months = sorted(universe["holding_month"].astype(str).unique())
    ld.open_session()
    try:
        for month in months:
            if _valid(month, contract["contract_sha256"]):
                continue
            _download_month(
                universe.loc[universe["holding_month"].eq(month)].reset_index(drop=True),
                contract["contract_sha256"],
            )
            completed = sum(_valid(value, contract["contract_sha256"]) for value in months)
            print(f"completed {month} ({completed}/{len(months)})", flush=True)
        _atomic_json(
            OUTPUT / "progress.json",
            {
                "status": "COMPLETE",
                "updated_at_utc": _utc_now(),
                "completed_months": len(months),
                "total_months": len(months),
            },
        )
    finally:
        ld.close_session()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        _atomic_json(
            OUTPUT / "progress.json",
            {"status": "FAILED", "updated_at_utc": _utc_now(), "error": str(error)},
        )
        raise
