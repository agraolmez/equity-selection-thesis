"""Current LSEG universe discovery for prospective V6.8 paper signals.

The builder deliberately discovers the security cross-section at the signal
cutoff rather than carrying the final historical CRSP universe forward.  It
keeps only consolidated, active US ordinary-share and ADR quotes and applies
the frozen price and market-cap pre-screen.  The remaining liquidity and
trading-history gates are applied after daily history is materialized.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


class LsegLiveUniverseError(RuntimeError):
    """The current LSEG universe cannot be authenticated."""


DISCOVERY_SELECT = (
    "RIC,DocumentTitle,CUSIP,ISIN,ExchangeCountry,RCSExchangeCountryLeaf,"
    "AssetState,RCSAssetCategoryLeaf,IsPrimaryRIC,IsPrimaryIssueRIC,"
    "IsExchangeTradable,Currency,TickerSymbol,PermID,PI,IssuerOAPermID,"
    "IssuerPI,IssuerCikNumber,IssuerCountry,SICC,AvgVol30D,PriceClose,"
    "MktCapIssueUsd,MktCapCompanyUsd,CommonName,ExchangeName"
)


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


def _normalized_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def select_live_universe(
    discovery: pd.DataFrame,
    *,
    minimum_price: float,
    minimum_market_cap_usd: float,
) -> pd.DataFrame:
    """Validate discovery rows and select one consolidated line per issuer."""

    required = {
        "RIC",
        "DocumentTitle",
        "ExchangeCountry",
        "AssetState",
        "RCSAssetCategoryLeaf",
        "IsPrimaryIssueRIC",
        "IsExchangeTradable",
        "Currency",
        "TickerSymbol",
        "PermID",
        "PI",
        "IssuerOAPermID",
        "IssuerPI",
        "PriceClose",
        "MktCapIssueUsd",
        "CommonName",
        "ExchangeName",
    }
    missing = sorted(required - set(discovery.columns))
    if missing:
        raise LsegLiveUniverseError(
            f"LSEG discovery omitted required columns: {', '.join(missing)}"
        )
    frame = discovery.copy()
    for column in (
        "RIC",
        "DocumentTitle",
        "ExchangeCountry",
        "AssetState",
        "RCSAssetCategoryLeaf",
        "Currency",
        "TickerSymbol",
        "PermID",
        "PI",
        "IssuerOAPermID",
        "IssuerPI",
        "CUSIP",
        "ISIN",
        "IssuerCikNumber",
        "IssuerCountry",
        "SICC",
        "CommonName",
        "ExchangeName",
    ):
        if column in frame:
            frame[column] = _normalized_text(frame[column])
    for column in (
        "PriceClose",
        "MktCapIssueUsd",
        "MktCapCompanyUsd",
        "AvgVol30D",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    allowed_categories = {
        "Ordinary Share",
        "Fully Paid Ordinary Share",
        "American Depository Receipt",
    }
    consolidated = frame["DocumentTitle"].str.contains(
        "Consolidated", case=False, na=False
    )
    eligible = (
        frame["ExchangeCountry"].eq("USA")
        & frame["AssetState"].eq("AC")
        & frame["IsPrimaryIssueRIC"].fillna(False).astype(bool)
        & frame["IsExchangeTradable"].fillna(False).astype(bool)
        & frame["Currency"].eq("USD")
        & frame["RCSAssetCategoryLeaf"].isin(allowed_categories)
        & consolidated
        & frame["RIC"].notna()
        & frame["TickerSymbol"].notna()
        & frame["PriceClose"].ge(float(minimum_price))
        & frame["MktCapIssueUsd"].ge(float(minimum_market_cap_usd))
    )
    frame = frame.loc[eligible].copy()
    if frame.empty:
        raise LsegLiveUniverseError("no consolidated US securities passed the pre-screen")
    if frame["RIC"].duplicated().any():
        raise LsegLiveUniverseError("LSEG discovery returned duplicate consolidated RICs")

    issuer_key = frame["IssuerOAPermID"].copy()
    issuer_key = issuer_key.fillna(frame["IssuerPI"])
    cusip6 = frame.get("CUSIP", pd.Series(pd.NA, index=frame.index)).str[:6]
    issuer_key = issuer_key.fillna(cusip6)
    issuer_key = issuer_key.fillna(frame["CommonName"].str.upper())
    frame["issuer_key"] = issuer_key
    if frame["issuer_key"].isna().any():
        raise LsegLiveUniverseError("a consolidated candidate has no stable issuer key")

    # Historical formation retains one traded security line per issuer.  Keep
    # the largest line; use provider average volume and RIC only as deterministic
    # tie-breakers.  The exact 60-session liquidity gate is applied downstream.
    frame = frame.sort_values(
        ["issuer_key", "MktCapIssueUsd", "AvgVol30D", "RIC"],
        ascending=[True, False, False, True],
        na_position="last",
        kind="mergesort",
    ).drop_duplicates("issuer_key", keep="first")
    frame["adr_indicator"] = frame["RCSAssetCategoryLeaf"].eq(
        "American Depository Receipt"
    ).astype("int8")
    frame["preliminary_dollar_volume_30d"] = (
        frame["PriceClose"] * frame["AvgVol30D"]
    )
    return frame.sort_values(
        ["MktCapIssueUsd", "RIC"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def discover_live_universe(
    *,
    config_path: Path,
    output: Path,
    signal_date: date,
) -> dict[str, Any]:
    """Query LSEG once and publish an atomic, secret-free universe snapshot."""

    try:
        import lseg.data as ld
    except ImportError as error:  # pragma: no cover - environment specific
        raise LsegLiveUniverseError(
            "run with /opt/anaconda3/envs/finance/bin/python"
        ) from error

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "research_v6_8_lseg_live_universe_v1":
        raise LsegLiveUniverseError("live-universe config schema differs")
    screen = config["pre_screen"]
    filter_expression = str(config["lseg_discovery"]["filter"])
    snapshot_path = output / "universe.parquet"
    receipt_path = output / "receipt.json"

    ld.open_session()
    try:
        raw = ld.discovery.search(
            view=ld.discovery.Views.EQUITY_QUOTES,
            filter=filter_expression,
            select=DISCOVERY_SELECT,
            top=int(config["lseg_discovery"]["maximum_rows"]),
            order_by="MktCapIssueUsd desc",
        )
    finally:
        ld.close_session()
    selected = select_live_universe(
        raw,
        minimum_price=float(screen["minimum_price_usd"]),
        minimum_market_cap_usd=float(screen["minimum_market_cap_usd"]),
    )
    selected.insert(0, "signal_date", pd.Timestamp(signal_date))
    selected.insert(0, "holding_month", str(pd.Period(signal_date, freq="M") + 1))
    selected.insert(0, "live_security_id", selected["PermID"].astype(str))
    _atomic_parquet(snapshot_path, selected)
    receipt = {
        "schema": "research_v6_8_lseg_live_universe_receipt_v1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_date": signal_date.isoformat(),
        "holding_month": str(pd.Period(signal_date, freq="M") + 1),
        "raw_discovery_rows": int(len(raw)),
        "selected_issuer_rows": int(len(selected)),
        "ordinary_share_rows": int(selected["adr_indicator"].eq(0).sum()),
        "adr_rows": int(selected["adr_indicator"].eq(1).sum()),
        "unique_quote_permid": int(selected["PermID"].nunique()),
        "unique_issuer_key": int(selected["issuer_key"].nunique()),
        "minimum_price_usd": float(selected["PriceClose"].min()),
        "minimum_market_cap_usd": float(selected["MktCapIssueUsd"].min()),
        "config_sha256": _sha256(config_path),
        "output_sha256": _sha256(snapshot_path),
        "provider_snapshot_is_current_not_backdated": True,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


__all__ = [
    "LsegLiveUniverseError",
    "discover_live_universe",
    "select_live_universe",
]
