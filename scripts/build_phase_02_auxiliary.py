#!/usr/bin/env python3
"""Build and publish the development-only Stage 2 auxiliary data bundle."""

from __future__ import annotations

import argparse
import json
import os
import time
import tomllib
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data.phase_02_auxiliary_compustat_factors import (
    build_family as build_compustat_factor_family,
    scan_ccm_market_crosscheck,
)
from portfolio_bot.data.phase_02_auxiliary_core import (
    TradingCalendar,
    canonical_json,
    load_schema_document,
)
from portfolio_bot.data.phase_02_auxiliary_linkage import build_auxiliary_signal_linkage
from portfolio_bot.data.phase_02_auxiliary_publication import (
    publish_primary,
    rehash_raw_sources,
)
from portfolio_bot.data.phase_02_optionmetrics import (
    OptionMetricsSettings,
    build_optionmetrics_family,
)
from portfolio_bot.data.phase_02_sec_restatement_auxiliary import (
    build_sec_restatement_family,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_CONTRACT_ID = "phase02-crsp-b03263a970e52fbd"
STAGING_NAME = ".staging-primary-20260715"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return value


def _save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    temporary.write_bytes(canonical_json(dict(value)))
    os.replace(temporary, path)


def _progress(stage: str, details: Mapping[str, Any]) -> None:
    print(json.dumps({"stage": stage, "details": dict(details), "time": time.time()}, sort_keys=True), flush=True)


def _database_root() -> Path:
    override = os.environ.get("PORTFOLIO_DATABASE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    with (PROJECT_ROOT / "configs/paths.toml").open("rb") as handle:
        return Path(tomllib.load(handle)["database"]["root"]).expanduser().resolve()


def _fixed_paths(database_root: Path) -> dict[str, Path]:
    base = database_root / f"interim/phase_02_canonical_panel/{DATABASE_CONTRACT_ID}"
    signal_primary = _json(
        PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_SIGNAL_DATE_UNIVERSE_PRIMARY.json"
    )
    return {
        "calendar": base / "calendar/calendar-ae2e4ba4a280d66e/trading_calendar.parquet",
        "identity": base / "crsp_typed_backbone/crsp-typed-2ab2f0b7910a1b4b33d1ca32/crsp_identity_interval.parquet",
        "security_day": base / "crsp_typed_backbone/crsp-typed-2ab2f0b7910a1b4b33d1ca32/crsp_security_day.parquet",
        "signal": database_root / signal_primary["interim_path"] / "crsp_signal_date_universe.parquet",
    }


def _calendar(path: Path) -> TradingCalendar:
    table = pq.read_table(path)
    date_name = "calendar_date" if "calendar_date" in table.column_names else next(
        name for name in table.column_names if "date" in name.lower()
    )
    return TradingCalendar(table[date_name].combine_chunks().to_pylist())


def _require_absent_or_receipted(receipt: Path, targets: list[Path], label: str) -> bool:
    if receipt.is_file():
        if not all(path.is_file() for path in targets):
            raise RuntimeError(f"{label} receipt exists but outputs are incomplete")
        return False
    existing = [path.name for path in targets if path.exists()]
    if existing:
        raise RuntimeError(f"Unreceipted partial {label} outputs require review: {existing}")
    return True


def run(database_root: Path, staging_root: Path) -> dict[str, Any]:
    raw_root = database_root / "raw"
    table_root = staging_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    paths = _fixed_paths(database_root)
    for label, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"Required verified {label} input is absent: {path}")
    schema_path = PROJECT_ROOT / "configs/phase_02_canonicalization/auxiliary_source_schema_v1.json"
    schema = load_schema_document(schema_path)

    sec_receipt_path = staging_root / "sec_restatement_receipt.json"
    sec_targets = [
        table_root / f"{name}.parquet"
        for name in (
            "sec_filing_observation", "sec_8k_item_observation",
            "sec_13f_filing_observation", "sec_13f_holding_observation",
            "restatement_event_observation", "restatement_crsp_link",
        )
    ]
    if _require_absent_or_receipted(sec_receipt_path, sec_targets, "SEC/restatement"):
        _progress("sec_restatement_started", {})
        started = time.perf_counter()
        receipt = build_sec_restatement_family(
            raw_root=raw_root,
            trading_calendar_path=paths["calendar"],
            crsp_identity_path=paths["identity"],
            staging_dir=table_root,
            batch_rows=250_000,
            progress=_progress,
        )
        receipt["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        _save(sec_receipt_path, receipt)
        _progress("sec_restatement_complete", {"elapsed_seconds": receipt["elapsed_seconds"]})

    comp_receipt_path = staging_root / "compustat_factor_receipt.json"
    comp_names = (
        "ccm_link_interval", "compustat_annual_observation",
        "compustat_quarterly_observation", "compustat_segment_observation",
        "ff_factor_daily_observation", "ff_factor_monthly_observation",
        "ps_liquidity_observation",
    )
    comp_targets = [table_root / f"{name}.parquet" for name in comp_names]
    if _require_absent_or_receipted(comp_receipt_path, comp_targets, "Compustat/factor"):
        _progress("compustat_factors_started", {})
        started = time.perf_counter()
        receipt = build_compustat_factor_family(
            raw_root=raw_root,
            output_dir=table_root,
            schema_document=schema,
            calendar=_calendar(paths["calendar"]),
            sec_filing_path=table_root / "sec_filing_observation.parquet",
        )
        receipt["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        _save(comp_receipt_path, receipt)
        _progress("compustat_factors_complete", {"elapsed_seconds": receipt["elapsed_seconds"]})

    option_receipt_path = staging_root / "optionmetrics_receipt.json"
    option_names = (
        "optionmetrics_name_interval", "optionmetrics_crsp_link_interval",
        "optionmetrics_option_volume_observation", "optionmetrics_historical_volatility_observation",
    )
    option_targets = [table_root / f"{name}.parquet" for name in option_names]
    if _require_absent_or_receipted(option_receipt_path, option_targets, "OptionMetrics"):
        _progress("optionmetrics_started", {})
        started = time.perf_counter()
        receipt = build_optionmetrics_family(
            OptionMetricsSettings(
                raw_root=raw_root,
                crsp_identity_path=paths["identity"],
                staging_dir=table_root,
            ),
            progress=_progress,
        )
        receipt["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        _save(option_receipt_path, receipt)
        _progress("optionmetrics_complete", {"elapsed_seconds": receipt["elapsed_seconds"]})

    identity_permnos = pq.read_table(paths["identity"], columns=["permno"])["permno"]
    identity_permnos = pc.unique(identity_permnos.combine_chunks())
    ccm_receipts: dict[str, Any] = {}
    for key, filename in (
        ("ccm_daily", "ccm_security_daily.csv"),
        ("ccm_monthly", "ccm_security_monthly.csv"),
    ):
        receipt_path = staging_root / f"{key}_qa_receipt.json"
        if receipt_path.is_file():
            ccm_receipts[key] = _json(receipt_path)
        else:
            _progress(f"{key}_qa_started", {})
            receipt = scan_ccm_market_crosscheck(
                raw_path=raw_root / filename,
                valid_crsp_permnos=identity_permnos,
            )
            _save(receipt_path, receipt)
            ccm_receipts[key] = receipt
            _progress(f"{key}_qa_complete", {"raw_rows": receipt["raw_rows"]})

    linkage_path = table_root / "auxiliary_signal_linkage.parquet"
    linkage_receipt_path = staging_root / "auxiliary_signal_linkage_receipt.json"
    if _require_absent_or_receipted(linkage_receipt_path, [linkage_path], "auxiliary linkage"):
        _progress("auxiliary_signal_linkage_started", {})
        receipt = build_auxiliary_signal_linkage(
            signal_universe_path=paths["signal"],
            ccm_link_path=table_root / "ccm_link_interval.parquet",
            optionmetrics_link_path=table_root / "optionmetrics_crsp_link_interval.parquet",
            output_path=linkage_path,
        )
        _save(linkage_receipt_path, receipt)
        _progress("auxiliary_signal_linkage_complete", {"rows": receipt["rows"]})

    raw_hash_path = staging_root / "raw_source_hashes.json"
    if raw_hash_path.is_file():
        raw_hashes = _json(raw_hash_path)
    else:
        _progress("raw_source_rehash_started", {})
        raw_hashes = rehash_raw_sources(
            project_root=PROJECT_ROOT, raw_root=raw_root, progress=_progress
        )
        _save(raw_hash_path, raw_hashes)
        _progress("raw_source_rehash_complete", {"sources": len(raw_hashes)})

    source_receipts = {
        **ccm_receipts,
        "optionmetrics": _json(option_receipt_path),
        "sec_restatement": _json(sec_receipt_path),
        "compustat_factors": _json(comp_receipt_path),
    }
    _progress("auxiliary_primary_publication_started", {})
    pointer = publish_primary(
        project_root=PROJECT_ROOT,
        database_root=database_root,
        staging_root=staging_root,
        raw_source_hashes=raw_hashes,
        signal_path=paths["signal"],
        source_receipts=source_receipts,
    )
    _progress("auxiliary_primary_publication_complete", {"build_id": pointer["build_id"]})
    return pointer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--staging-root", type=Path, default=None)
    arguments = parser.parse_args()
    database_root = (arguments.database_root or _database_root()).expanduser().resolve()
    staging_root = arguments.staging_root or (
        database_root
        / f"interim/phase_02_canonical_panel/{DATABASE_CONTRACT_ID}/auxiliary_availability_linkage/{STAGING_NAME}"
    )
    pointer = run(database_root, staging_root.resolve())
    print(json.dumps(pointer, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
