#!/usr/bin/env python3
"""Resumable point-in-time LSEG selector-data download.

The script reads authenticated CRSP signal universes, addresses LSEG by
historical CUSIP (not today's ticker), and publishes one atomic parquet plus
receipt per holding month. Secrets are accepted only through LSEG_APP_KEY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lseg.data as ld
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("/Users/agraemirolmez/Desktop/Thesis/raw/research_v6/lseg")
VALIDATION_UNIVERSE = Path(
    "/Users/agraemirolmez/Desktop/Thesis/interim/phase_05_validation/"
    "stage11-validation-data-v1/universe_extension/universe/"
    "validation_signal_date_universe.parquet"
)
FINAL_WORKING_UNIVERSE = Path(
    "/Users/agraemirolmez/Desktop/Thesis/cache/phase_06_final_test/"
    "stage13-final-source-replay-v1/"
    "082c42dd57e8077f47cf409a12aefd8ec2ee98b732839496a320f8f84abeb66e/"
    "formula_work/predictor_formula/input_subsets/"
    "d9425a00ff93b1b6af5170346561325bdccb8eb94e30b0426ace877796a6438a/"
    "working/crsp_signal_date_universe.parquet"
)
IDENTITY_INTERVALS = Path(
    "/Users/agraemirolmez/Desktop/Thesis/cache/phase_06_final_test/"
    "stage13-final-source-replay-v1/"
    "082c42dd57e8077f47cf409a12aefd8ec2ee98b732839496a320f8f84abeb66e/"
    "formula_work/decision_market/typed/crsp_identity_interval.parquet"
)

START_MONTH = "2015-01"
END_MONTH = "2025-08"
CHUNK_SIZE = 1000
ESTIMATE_BASES = ["Revenue", "GrossIncome", "PreTaxProfit", "NetProfit", "EPS", "DPS"]
HORIZONS = ("FY1", "FY2", "FQ1")
SERIALIZATION_MIGRATION_ID = "lseg-numeric-string-coercion-v2"
SERIALIZATION_MIGRATION_PARENT_CONTRACT_SHA256 = (
    "06678b805fbcb8b45bed65ea3b78c796775bf2c39ad70c0d86a944d791333d69"
)

warnings.filterwarnings("ignore", category=FutureWarning, module=r"lseg\.data.*")


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


SIGNAL_FIELDS = [
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
]

SELECTOR_FEATURE_VIEWS = {
    "cam_free": {
        "description": "All raw LSEG estimate, recommendation, and target-price fields; CAM excluded.",
        "exclude_raw_fields": [
            "signals__tr_combinedalphacountryrank_date",
            "signals__tr_combinedalphacountryrank",
        ],
    },
    "full_information": {
        "description": "The CAM-free view plus point-in-time StarMine CAM rank and causal lagged changes.",
        "include_cam_derived_fields": [
            "cam_rank_change_1m",
            "cam_rank_change_3m",
            "cam_rank_change_6m",
        ],
    },
    "cam_only": {
        "description": "Vendor-model benchmark using only CAM rank, its causal lagged changes, and missingness.",
        "include_raw_fields": [
            "signals__tr_combinedalphacountryrank_date",
            "signals__tr_combinedalphacountryrank",
        ],
        "include_cam_derived_fields": [
            "cam_rank_change_1m",
            "cam_rank_change_3m",
            "cam_rank_change_6m",
            "cam_missing",
        ],
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    os.replace(tmp, path)


def cusip_check_digit(cusip8: str) -> str | None:
    value = str(cusip8).strip().upper()
    if len(value) != 8:
        return None
    total = 0
    for index, char in enumerate(value):
        if char.isdigit():
            digit = int(char)
        elif "A" <= char <= "Z":
            digit = ord(char) - ord("A") + 10
        elif char == "*":
            digit = 36
        elif char == "@":
            digit = 37
        elif char == "#":
            digit = 38
        else:
            return None
        digit *= 1 if index % 2 == 0 else 2
        total += digit // 10 + digit % 10
    return str((10 - total % 10) % 10)


def prepare_universe() -> Path:
    columns = [
        "signal_date",
        "permno",
        "permco",
        "market_cap_usd",
        "median_daily_dollar_volume_60_usd",
        "long_eligible",
        "short_eligible",
    ]
    validation = pq.read_table(VALIDATION_UNIVERSE, columns=columns).to_pandas()
    final = pq.read_table(FINAL_WORKING_UNIVERSE, columns=columns).to_pandas()
    universe = pd.concat([validation, final], ignore_index=True)
    universe["holding_month"] = (
        pd.to_datetime(universe["signal_date"]).dt.to_period("M") + 1
    ).astype(str)
    universe = universe[
        universe["holding_month"].between(START_MONTH, END_MONTH)
        & universe["long_eligible"].fillna(False)
    ].copy()
    universe["signal_date"] = pd.to_datetime(universe["signal_date"]).dt.date

    identities = pq.read_table(
        IDENTITY_INTERVALS,
        columns=[
            "permno",
            "permco",
            "start_date",
            "end_date",
            "cusip",
            "ticker",
            "issuer_name",
            "primary_exchange",
        ],
    ).to_pandas()
    identities["start_date"] = pd.to_datetime(identities["start_date"]).dt.date
    identities["end_date"] = pd.to_datetime(identities["end_date"]).dt.date
    joined = universe.merge(identities, on="permno", how="left", suffixes=("", "_identity"))
    joined = joined[
        joined["start_date"].notna()
        & (joined["start_date"] <= joined["signal_date"])
        & (joined["end_date"] >= joined["signal_date"])
    ].copy()
    joined["cusip8"] = joined["cusip"].astype("string").str.strip().str.upper()

    def to_cusip9(value: object) -> object:
        if not isinstance(value, str):
            return pd.NA
        check = cusip_check_digit(value)
        return value + check if check is not None else pd.NA

    joined["cusip9"] = joined["cusip8"].map(to_cusip9)
    joined = joined.dropna(subset=["cusip9"])
    joined = joined.sort_values(["holding_month", "permno", "start_date"]).drop_duplicates(
        ["holding_month", "permno"], keep="last"
    )
    joined = joined[
        [
            "holding_month",
            "signal_date",
            "permno",
            "permco",
            "cusip8",
            "cusip9",
            "ticker",
            "issuer_name",
            "primary_exchange",
            "market_cap_usd",
            "median_daily_dollar_volume_60_usd",
            "long_eligible",
            "short_eligible",
        ]
    ]
    if joined.duplicated(["holding_month", "permno"]).any():
        raise RuntimeError("duplicate holding_month/permno in prepared universe")
    expected = pd.period_range(START_MONTH, END_MONTH, freq="M").astype(str)
    missing_months = sorted(set(expected) - set(joined["holding_month"]))
    if missing_months:
        raise RuntimeError(f"prepared universe is missing months: {missing_months}")

    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "input_universe.parquet"
    tmp = ROOT / ".input_universe.parquet.tmp"
    joined.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    atomic_json(
        ROOT / "input_universe_receipt.json",
        {
            "created_at_utc": utc_now(),
            "start_month": START_MONTH,
            "end_month": END_MONTH,
            "rows": int(len(joined)),
            "months": int(joined["holding_month"].nunique()),
            "unique_permnos": int(joined["permno"].nunique()),
            "validation_universe_sha256": sha256_file(VALIDATION_UNIVERSE),
            "final_working_universe_sha256": sha256_file(FINAL_WORKING_UNIVERSE),
            "identity_intervals_sha256": sha256_file(IDENTITY_INTERVALS),
            "output_sha256": sha256_file(path),
        },
    )
    return path


def build_contract(universe_path: Path) -> dict[str, Any]:
    fields = {horizon: estimate_fields(horizon) for horizon in HORIZONS}
    fields["signals"] = SIGNAL_FIELDS
    payload = {
        "schema_version": 1,
        "start_month": START_MONTH,
        "end_month": END_MONTH,
        "chunk_size": CHUNK_SIZE,
        "universe_sha256": sha256_file(universe_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "lseg_data_version": getattr(ld, "__version__", "unknown"),
        "addressing": "historical_cusip9",
        "cutoff": "exact_authenticated_signal_date",
        "fields": fields,
        "selector_feature_views": SELECTOR_FEATURE_VIEWS,
        "authorized_serialization_migration": {
            "migration_id": SERIALIZATION_MIGRATION_ID,
            "parent_contract_sha256": SERIALIZATION_MIGRATION_PARENT_CONTRACT_SHA256,
            "semantic_change": "empty vendor strings become null and non-date vendor numeric strings are parsed before parquet serialization",
            "raw_nonempty_values_changed": False,
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["contract_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def normalize_frame(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["cusip9"])
    frame = frame.copy()
    instrument = next(
        (column for column in frame.columns if str(column).upper() == "INSTRUMENT"),
        frame.columns[0],
    )
    frame = frame.rename(columns={instrument: "cusip9"})
    frame["cusip9"] = frame["cusip9"].astype("string").str.strip().str.upper()
    names: list[str] = []
    seen: dict[str, int] = {}
    for column in frame.columns:
        if column == "cusip9":
            names.append(column)
            continue
        base = str(column).strip().lower()
        for old, new in [
            (".", "_"),
            ("(", "_"),
            (")", ""),
            ("=", "_"),
            (",", "_"),
            (" ", "_"),
        ]:
            base = base.replace(old, new)
        name = f"{prefix}__{base}"
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    frame.columns = names
    count_column = f"{prefix}__tr_numofrecommendations"
    if count_column in frame and pd.api.types.is_datetime64_any_dtype(frame[count_column]):
        frame[count_column] = frame[count_column].astype("int64")
    return frame.drop_duplicates("cusip9", keep="last")


def request_with_retry(
    instruments: list[str], fields: list[str], signal_date: str
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return ld.get_data(
                instruments,
                fields,
                parameters={"SDate": signal_date, "EDate": signal_date, "Frq": "D"},
                header_type=ld.HeaderType.NAME,
            )
        except Exception as error:
            last_error = error
            if attempt == 3:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"LSEG request failed after retries: {last_error}")


def valid_receipt(month: str, contract_sha256: str) -> bool:
    data_path = ROOT / "months" / f"{month}.parquet"
    receipt_path = ROOT / "receipts" / f"{month}.json"
    if not data_path.exists() or not receipt_path.exists():
        return False
    try:
        receipt = json.loads(receipt_path.read_text())
        return (
            receipt.get("contract_sha256") == contract_sha256
            and receipt.get("output_sha256") == sha256_file(data_path)
            and pq.ParquetFile(data_path).metadata.num_rows == receipt.get("rows")
        )
    except Exception:
        return False


def migrate_parent_receipts(contract_sha256: str) -> None:
    migrated: list[dict[str, Any]] = []
    receipt_root = ROOT / "receipts"
    if not receipt_root.exists():
        return
    for receipt_path in sorted(receipt_root.glob("*.json")):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("contract_sha256") == contract_sha256:
            continue
        if (
            receipt.get("contract_sha256")
            != SERIALIZATION_MIGRATION_PARENT_CONTRACT_SHA256
        ):
            raise RuntimeError(
                f"{receipt_path.name}: unexpected checkpoint contract identity"
            )
        data_path = ROOT / "months" / f"{receipt_path.stem}.parquet"
        if (
            not data_path.exists()
            or receipt.get("output_sha256") != sha256_file(data_path)
            or pq.ParquetFile(data_path).metadata.num_rows != receipt.get("rows")
        ):
            raise RuntimeError(
                f"{receipt_path.name}: parent checkpoint failed migration authentication"
            )
        parent_hash = receipt["contract_sha256"]
        history = receipt.get("migration_history", [])
        if receipt.get("migrated_from_contract_sha256"):
            history.append(
                {
                    "migration_id": receipt.get("migration_id"),
                    "from_contract_sha256": receipt.get(
                        "migrated_from_contract_sha256"
                    ),
                    "to_contract_sha256": parent_hash,
                    "payload_bytes_changed": receipt.get(
                        "migration_payload_bytes_changed", False
                    ),
                }
            )
        receipt["contract_sha256"] = contract_sha256
        receipt["migrated_from_contract_sha256"] = parent_hash
        receipt["migration_id"] = SERIALIZATION_MIGRATION_ID
        receipt["migration_history"] = history
        receipt["migration_payload_bytes_changed"] = False
        receipt["migration_at_utc"] = utc_now()
        atomic_json(receipt_path, receipt)
        migrated.append(
            {
                "holding_month": receipt_path.stem,
                "output_sha256": receipt["output_sha256"],
                "rows": receipt["rows"],
            }
        )
    atomic_json(
        ROOT / "migrations" / f"{SERIALIZATION_MIGRATION_ID}.json",
        {
            "created_at_utc": utc_now(),
            "migration_id": SERIALIZATION_MIGRATION_ID,
            "authorization": "explicit user approval in local Codex thread on 2026-08-12",
            "parent_contract_sha256": SERIALIZATION_MIGRATION_PARENT_CONTRACT_SHA256,
            "replacement_contract_sha256": contract_sha256,
            "payload_bytes_changed": False,
            "migrated_checkpoint_count": len(migrated),
            "migrated_checkpoints": migrated,
        },
    )


def coerce_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        series = frame[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue
        text = series.astype("string")
        blank = text.str.strip().eq("").fillna(False)
        if blank.any():
            series = series.mask(blank, pd.NA)
            frame[column] = series
        is_vendor_numeric = column.startswith(
            ("fy1__", "fy2__", "fq1__", "signals__")
        ) and not (
            column.endswith("_date") or "calcdate" in column
        )
        if is_vendor_numeric:
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = series.notna() & numeric.isna()
            if invalid.any():
                examples = series[invalid].astype(str).drop_duplicates().head(5).tolist()
                raise RuntimeError(
                    f"{column}: unexpected nonnumeric LSEG values: {examples}"
                )
            frame[column] = numeric
    return frame


def download_month(
    month_frame: pd.DataFrame, month: str, contract_sha256: str
) -> None:
    signal_dates = month_frame["signal_date"].astype(str).unique().tolist()
    if len(signal_dates) != 1:
        raise RuntimeError(f"{month}: expected one signal date, got {signal_dates}")
    signal_date = signal_dates[0]
    chunks: list[pd.DataFrame] = []
    total_chunks = (len(month_frame) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_index, start in enumerate(
        range(0, len(month_frame), CHUNK_SIZE), start=1
    ):
        base = month_frame.iloc[start : start + CHUNK_SIZE].copy()
        instruments = base["cusip9"].astype(str).tolist()
        merged = base
        for horizon in HORIZONS:
            response = request_with_retry(
                instruments, estimate_fields(horizon), signal_date
            )
            merged = merged.merge(
                normalize_frame(response, horizon.lower()), on="cusip9", how="left"
            )
        response = request_with_retry(instruments, SIGNAL_FIELDS, signal_date)
        merged = merged.merge(
            normalize_frame(response, "signals"), on="cusip9", how="left"
        )
        chunks.append(merged)
        atomic_json(
            ROOT / "progress.json",
            {
                "updated_at_utc": utc_now(),
                "status": "RUNNING",
                "holding_month": month,
                "signal_date": signal_date,
                "chunk": chunk_index,
                "chunks": total_chunks,
                "rows_completed_in_month": int(sum(len(x) for x in chunks)),
                "rows_in_month": int(len(month_frame)),
            },
        )

    output = coerce_for_parquet(pd.concat(chunks, ignore_index=True))
    if output.duplicated(["holding_month", "permno"]).any():
        raise RuntimeError(f"{month}: duplicate output keys")
    data_path = ROOT / "months" / f"{month}.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    staging = ROOT / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    tmp = staging / f"{month}.parquet.tmp"
    output.to_parquet(tmp, index=False)
    os.replace(tmp, data_path)
    feature_columns = [column for column in output if column not in month_frame.columns]
    atomic_json(
        ROOT / "receipts" / f"{month}.json",
        {
            "created_at_utc": utc_now(),
            "holding_month": month,
            "signal_date": signal_date,
            "contract_sha256": contract_sha256,
            "rows": int(len(output)),
            "feature_columns": int(len(feature_columns)),
            "rows_with_any_lseg_value": int(
                output[feature_columns].notna().any(axis=1).sum()
            ),
            "output_sha256": sha256_file(data_path),
        },
    )


def run() -> None:
    if not os.environ.get("LSEG_APP_KEY"):
        raise RuntimeError("LSEG_APP_KEY is not set")
    universe_path = prepare_universe()
    contract = build_contract(universe_path)
    atomic_json(ROOT / "contract.json", contract)
    migrate_parent_receipts(contract["contract_sha256"])
    universe = pd.read_parquet(universe_path)
    months = sorted(universe["holding_month"].unique())
    ld.open_session(app_key=os.environ["LSEG_APP_KEY"])
    try:
        for month in months:
            if valid_receipt(month, contract["contract_sha256"]):
                continue
            month_frame = universe[
                universe["holding_month"] == month
            ].reset_index(drop=True)
            download_month(month_frame, month, contract["contract_sha256"])
            completed = sum(
                valid_receipt(value, contract["contract_sha256"]) for value in months
            )
            print(f"completed {month} ({completed}/{len(months)})", flush=True)
        atomic_json(
            ROOT / "progress.json",
            {
                "updated_at_utc": utc_now(),
                "status": "COMPLETE",
                "completed_months": len(months),
                "total_months": len(months),
            },
        )
    finally:
        ld.close_session()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        print(prepare_universe())
        return
    run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        atomic_json(
            ROOT / "progress.json",
            {"updated_at_utc": utc_now(), "status": "FAILED", "error": str(exc)},
        )
        raise
