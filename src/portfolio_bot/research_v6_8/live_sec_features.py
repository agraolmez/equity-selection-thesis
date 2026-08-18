"""Checkpointed SEC submissions/event features for V6.8 paper trading."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from portfolio_bot.features.ownership_event_features import (
    compute_restatement_features,
    compute_sec_event_features,
)


SEC_FEATURES = (
    "days_since_latest_annual_filing",
    "days_since_latest_interim_financial_filing",
    "latest_annual_filing_form",
    "latest_interim_financial_filing_form",
    "count_8k_30d",
    "count_8k_90d",
    "count_6k_30d",
    "count_6k_90d",
    "recency_8k_30d",
    "recency_8k_90d",
    "recency_6k_30d",
    "recency_6k_90d",
    "restatement_disclosed_90d",
    "restatement_disclosed_365d",
    "eight_k_item_families_30d",
    "eight_k_item_families_90d",
)
SEC_RELEVANT_FORM_FAMILIES = frozenset(
    {"10-K", "20-F", "40-F", "10-Q", "8-K", "6-K"}
)


class LiveSecFeatureError(RuntimeError):
    """The SEC point-in-time bridge failed a production requirement."""


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


def normalize_single_cik(value: Any) -> str | None:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    return text if re.fullmatch(r"\d{10}", text) else None


def parse_sec_submissions(
    payload: Mapping[str, Any], *, cik: str, signal_cutoff: datetime
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert SEC recent submissions into the frozen filing/item grains."""

    recent = payload.get("filings", {}).get("recent", {})
    accessions = list(recent.get("accessionNumber", []))
    columns = {
        "filingDate": list(recent.get("filingDate", [])),
        "reportDate": list(recent.get("reportDate", [])),
        "acceptanceDateTime": list(recent.get("acceptanceDateTime", [])),
        "form": list(recent.get("form", [])),
        "items": list(recent.get("items", [])),
    }
    if any(len(values) != len(accessions) for values in columns.values()):
        raise LiveSecFeatureError("SEC submissions recent arrays have unequal lengths")
    filings: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    seen_accessions: dict[str, tuple[Any, ...]] = {}
    for index, accession in enumerate(accessions):
        acceptance = pd.to_datetime(
            columns["acceptanceDateTime"][index], errors="coerce", utc=True
        )
        if pd.isna(acceptance):
            continue
        accepted_at = acceptance.to_pydatetime()
        if accepted_at > signal_cutoff.astimezone(timezone.utc):
            continue
        filing_date = pd.to_datetime(columns["filingDate"][index], errors="coerce")
        report_date = pd.to_datetime(columns["reportDate"][index], errors="coerce")
        if pd.isna(filing_date):
            continue
        form = str(columns["form"][index]).strip().upper()
        family = form.removesuffix("/A")
        if family not in SEC_RELEVANT_FORM_FAMILIES:
            continue
        accession_key = str(accession)
        signature = (
            filing_date.date(),
            None if pd.isna(report_date) else report_date.date(),
            accepted_at,
            family,
            str(columns["items"][index] or ""),
        )
        if accession_key in seen_accessions:
            if seen_accessions[accession_key] != signature:
                raise LiveSecFeatureError(
                    f"SEC accession {accession_key} has conflicting duplicate metadata"
                )
            continue
        seen_accessions[accession_key] = signature
        row = {
            "cik": cik,
            "source_record_id": accession_key,
            "form_family": family,
            "filing_date": filing_date.date(),
            "report_date": None if pd.isna(report_date) else report_date.date(),
            "observation_date": filing_date.date(),
            "public_availability_date": accepted_at.date(),
            "accepted_at_utc": accepted_at,
        }
        filings.append(row)
        if family == "8-K":
            item_text = str(columns["items"][index] or "")
            for item in sorted(
                {token.strip() for token in item_text.split(",") if token.strip()}
            ):
                items.append(
                    {
                        **row,
                        "source_record_id": f"{accession}|{item}",
                        "item_number": item,
                    }
                )
    return filings, items


def sec_feature_row(
    *,
    live_security_id: str,
    cik: str | None,
    signal_date: date,
    filing_rows: list[dict[str, Any]],
    item_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    matched = cik is not None
    events = compute_sec_event_features(
        sec_match_status="MATCHED" if matched else "UNMATCHED",
        cik=cik,
        signal_date=signal_date,
        filing_rows=filing_rows,
        item_rows=item_rows,
    )
    restatement_rows = [
        row for row in item_rows if str(row.get("item_number", "")) == "4.02"
    ]
    restatements = compute_restatement_features(
        positively_covered=matched,
        signal_date=signal_date,
        event_rows=restatement_rows,
    )
    values = {**events, **restatements}
    if set(values) != set(SEC_FEATURES):
        raise LiveSecFeatureError("SEC feature formula block differs from the registry")
    return {"live_security_id": str(live_security_id), **values}


def materialize_live_sec_features(
    *, config_path: Path, universe_path: Path, output: Path
) -> dict[str, Any]:
    """Download one submissions snapshot per unambiguous CIK and publish 16 features."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "research_v6_8_live_sec_features_v1":
        raise LiveSecFeatureError("live SEC config schema differs")
    env_name = str(config["user_agent_environment"])
    user_agent = os.environ.get(env_name, "").strip()
    if not user_agent or "@" not in user_agent:
        raise LiveSecFeatureError(
            f"{env_name} must contain a truthful SEC contact name and email"
        )
    universe = pd.read_parquet(universe_path)
    signal_dates = pd.to_datetime(universe["signal_date"], errors="raise").dt.date
    if signal_dates.nunique() != 1:
        raise LiveSecFeatureError("live SEC universe must have one signal date")
    signal_date = signal_dates.iloc[0]
    cutoff_time = datetime.strptime(
        str(config["signal_cutoff_local_time"]), "%H:%M:%S"
    ).time()
    cutoff = datetime.combine(
        signal_date,
        cutoff_time,
        tzinfo=ZoneInfo(str(config["signal_cutoff_timezone"])),
    )
    raw_root = output / "submissions"
    unique_ciks = sorted(
        {
            cik
            for cik in universe["IssuerCikNumber"].map(normalize_single_cik)
            if cik is not None
        }
    )
    minimum_interval = 1.0 / float(config["maximum_requests_per_second"])
    failures: list[dict[str, str]] = []
    last_request = 0.0
    for position, cik in enumerate(unique_ciks, start=1):
        path = raw_root / f"CIK{cik}.json"
        if path.is_file():
            continue
        elapsed = time.monotonic() - last_request
        if elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)
        url = str(config["submissions_url_template"]).format(cik=cik)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": user_agent},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            failures.append({"cik": cik, "error": str(error)[:300]})
            continue
        finally:
            last_request = time.monotonic()
        _atomic_json(path, payload)
        if position % 25 == 0:
            _atomic_json(
                output / "progress.json",
                {
                    "schema": "research_v6_8_live_sec_progress_v1",
                    "status": "RUNNING",
                    "completed_ciks": position - len(failures),
                    "total_ciks": len(unique_ciks),
                    "failures": failures,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
    if failures:
        _atomic_json(
            output / "progress.json",
            {
                "schema": "research_v6_8_live_sec_progress_v1",
                "status": "FAILED",
                "failures": failures,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise LiveSecFeatureError(f"{len(failures)} SEC submissions requests failed")
    parsed: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for cik in unique_ciks:
        payload = json.loads((raw_root / f"CIK{cik}.json").read_text(encoding="utf-8"))
        parsed[cik] = parse_sec_submissions(payload, cik=cik, signal_cutoff=cutoff)
    rows: list[dict[str, Any]] = []
    for row in universe.itertuples(index=False):
        cik = normalize_single_cik(row.IssuerCikNumber)
        filings, items = parsed.get(cik, ([], []))
        rows.append(
            sec_feature_row(
                live_security_id=str(row.live_security_id),
                cik=cik,
                signal_date=signal_date,
                filing_rows=filings,
                item_rows=items,
            )
        )
    result = pd.DataFrame(rows)
    if len(result) != len(universe) or result["live_security_id"].duplicated().any():
        raise LiveSecFeatureError("live SEC output grain differs from the universe")
    data_path = output / "sec_features.parquet"
    _atomic_parquet(data_path, result)
    receipt = {
        "schema": "research_v6_8_live_sec_features_receipt_v1",
        "status": "PASS",
        "signal_date": signal_date.isoformat(),
        "rows": int(len(result)),
        "unique_unambiguous_ciks": int(len(unique_ciks)),
        "unambiguous_cik_coverage_share": float(
            universe["IssuerCikNumber"].map(normalize_single_cik).notna().mean()
        ),
        "feature_count": len(SEC_FEATURES),
        "output_sha256": _sha256(data_path),
        "secrets_in_receipt": False,
    }
    _atomic_json(output / "receipt.json", receipt)
    _atomic_json(
        output / "progress.json",
        {
            "schema": "research_v6_8_live_sec_progress_v1",
            "status": "COMPLETE",
            "completed_ciks": len(unique_ciks),
            "total_ciks": len(unique_ciks),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return receipt


__all__ = [
    "LiveSecFeatureError",
    "SEC_FEATURES",
    "materialize_live_sec_features",
    "normalize_single_cik",
    "parse_sec_submissions",
    "sec_feature_row",
]
