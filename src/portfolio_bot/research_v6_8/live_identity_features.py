"""Current identity/category features for the frozen V6.8 input schema."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from portfolio_bot.features.ownership_event_features import compute_identity_features
from portfolio_bot.research_v6_8.live_sec_features import normalize_single_cik


IDENTITY_FEATURES = (
    "foreign_listing_indicator",
    "adr_indicator",
    "us_incorporation_indicator",
    "exchange_category",
    "sic2_industry_category",
    "domicile_country_category",
    "reporting_currency_category",
)


class LiveIdentityFeatureError(RuntimeError):
    """The current identity bridge failed a frozen-schema requirement."""


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


def _clean(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    return text or None


def _sec_sic(sec_submissions: Path | None, cik: str | None) -> str | None:
    if sec_submissions is None or cik is None:
        return None
    path = sec_submissions / f"CIK{cik}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    sic = _clean(payload.get("sic"))
    return sic if sic is not None and sic.isdigit() and len(sic) <= 4 else None


def build_identity_features(
    universe: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    fundamentals: pd.DataFrame | None = None,
    sec_submissions: Path | None = None,
) -> pd.DataFrame:
    """Build the seven model identity features at one row per live security."""

    required = {
        "live_security_id",
        "IssuerCikNumber",
        "IssuerCountry",
        "ExchangeName",
        "adr_indicator",
    }
    missing = sorted(required - set(universe.columns))
    if missing:
        raise LiveIdentityFeatureError(
            f"live universe omits identity columns: {', '.join(missing)}"
        )
    if universe["live_security_id"].duplicated().any():
        raise LiveIdentityFeatureError("live universe duplicates security identities")
    fundamental_by_id: dict[str, Mapping[str, Any]] = {}
    if fundamentals is not None:
        if "live_security_id" not in fundamentals or fundamentals["live_security_id"].duplicated().any():
            raise LiveIdentityFeatureError("fundamental identity grain is invalid")
        fundamental_by_id = {
            str(row["live_security_id"]): row
            for row in fundamentals.to_dict(orient="records")
        }
    # The frozen helper also computes FF12, which is diagnostic-only and is
    # deliberately discarded here.  A residual-only mapping keeps that
    # irrelevant branch total without introducing a second sector policy.
    ff12_dummy = {
        "unknown_category": "UNKNOWN_SECTOR",
        "sectors": [{"code": "12_OTHER", "residual_valid_sic": True}],
    }
    rows: list[dict[str, Any]] = []
    exchange_mapping = {
        str(key): str(value) for key, value in config["exchange_mapping"].items()
    }
    for row in universe.to_dict(orient="records"):
        security_id = str(row["live_security_id"])
        fundamental = fundamental_by_id.get(security_id, {})
        issuer_country = _clean(row.get("IssuerCountry"))
        fic = _clean(fundamental.get("fundamental_fic")) or issuer_country
        loc = _clean(fundamental.get("fundamental_loc"))
        currency = _clean(fundamental.get("reporting_currency"))
        cik = normalize_single_cik(row.get("IssuerCikNumber"))
        sic = _sec_sic(sec_submissions, cik)
        exchange = exchange_mapping.get(str(row.get("ExchangeName")), "UNKNOWN")
        values = compute_identity_features(
            {
                "share_type": "AD" if int(row.get("adr_indicator", 0)) == 1 else "NS",
                "us_incorporation_flag": (
                    None if issuer_country is None else ("Y" if issuer_country == "USA" else "N")
                ),
                "primary_exchange": exchange,
                "sic_code": sic,
            },
            fundamental_fic=fic,
            fundamental_loc=loc,
            reporting_currency=currency,
            ff12_mapping=ff12_dummy,
        )
        rows.append(
            {
                "live_security_id": security_id,
                **{name: values[name] for name in IDENTITY_FEATURES},
                "identity_cik": cik,
                "identity_sic_source": "SEC_SUBMISSIONS" if sic is not None else "UNAVAILABLE",
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(universe) or result["live_security_id"].duplicated().any():
        raise LiveIdentityFeatureError("identity output grain differs from universe")
    return result


def materialize_live_identity_features(
    *,
    config_path: Path,
    universe_path: Path,
    output: Path,
    fundamentals_path: Path | None = None,
    sec_submissions: Path | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "research_v6_8_live_identity_features_v1":
        raise LiveIdentityFeatureError("live identity config schema differs")
    universe = pd.read_parquet(universe_path)
    fundamentals = (
        pd.read_parquet(fundamentals_path)
        if fundamentals_path is not None and fundamentals_path.is_file()
        else None
    )
    result = build_identity_features(
        universe,
        config=config,
        fundamentals=fundamentals,
        sec_submissions=sec_submissions,
    )
    data_path = output / "identity_features.parquet"
    _atomic_parquet(data_path, result)
    receipt = {
        "schema": "research_v6_8_live_identity_features_receipt_v1",
        "status": "PASS",
        "rows": int(len(result)),
        "feature_count": len(IDENTITY_FEATURES),
        "known_exchange_share": float(result["exchange_category"].ne("UNKNOWN").mean()),
        "known_sic2_share": float(result["sic2_industry_category"].ne("UNKNOWN_INDUSTRY").mean()),
        "known_domicile_share": float(result["domicile_country_category"].ne("UNKNOWN_COUNTRY").mean()),
        "known_reporting_currency_share": float(result["reporting_currency_category"].ne("UNKNOWN_CURRENCY").mean()),
        "output_sha256": _sha256(data_path),
    }
    _atomic_json(output / "receipt.json", receipt)
    return receipt


__all__ = [
    "IDENTITY_FEATURES",
    "LiveIdentityFeatureError",
    "build_identity_features",
    "materialize_live_identity_features",
]
