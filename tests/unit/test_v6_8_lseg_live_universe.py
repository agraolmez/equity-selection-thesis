from __future__ import annotations

import pandas as pd
import pytest

from portfolio_bot.research_v6_8.lseg_live_universe import (
    LsegLiveUniverseError,
    select_live_universe,
)


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "RIC": "AAA.O",
        "DocumentTitle": "Issuer A, Ordinary Share, NASDAQ Consolidated",
        "CUSIP": "000000001",
        "ISIN": "US000000001",
        "ExchangeCountry": "USA",
        "AssetState": "AC",
        "RCSAssetCategoryLeaf": "Ordinary Share",
        "IsPrimaryIssueRIC": True,
        "IsExchangeTradable": True,
        "Currency": "USD",
        "TickerSymbol": "AAA",
        "PermID": "quote-a",
        "PI": "pi-a",
        "IssuerOAPermID": "issuer-a",
        "IssuerPI": "issuer-pi-a",
        "IssuerCikNumber": "1",
        "IssuerCountry": "USA",
        "SICC": "3571",
        "PriceClose": 20.0,
        "MktCapIssueUsd": 500_000_000.0,
        "MktCapCompanyUsd": 500_000_000.0,
        "AvgVol30D": 1_000_000.0,
        "CommonName": "ISSUER A ORD",
        "ExchangeName": "NASDAQ Consolidated",
    }
    row.update(updates)
    return row


def test_selects_largest_consolidated_line_per_issuer() -> None:
    frame = pd.DataFrame(
        [
            _row(),
            _row(
                RIC="AAB.O",
                TickerSymbol="AAB",
                PermID="quote-b",
                MktCapIssueUsd=700_000_000.0,
            ),
            _row(
                RIC="ADR.N",
                TickerSymbol="ADR",
                PermID="quote-adr",
                IssuerOAPermID="issuer-adr",
                RCSAssetCategoryLeaf="American Depository Receipt",
                MktCapIssueUsd=900_000_000.0,
            ),
            _row(
                RIC="VENUE.Z",
                TickerSymbol="VENUE",
                PermID="quote-venue",
                IssuerOAPermID="issuer-venue",
                DocumentTitle="Issuer Venue, Ordinary Share, Cboe BZX Exchange",
            ),
        ]
    )
    result = select_live_universe(
        frame, minimum_price=5.0, minimum_market_cap_usd=300_000_000.0
    )
    assert result["RIC"].tolist() == ["ADR.N", "AAB.O"]
    assert result.set_index("RIC").loc["ADR.N", "adr_indicator"] == 1
    assert result["issuer_key"].nunique() == len(result)


def test_rejects_missing_required_discovery_field() -> None:
    frame = pd.DataFrame([_row()]).drop(columns="IssuerPI")
    with pytest.raises(LsegLiveUniverseError, match="IssuerPI"):
        select_live_universe(
            frame, minimum_price=5.0, minimum_market_cap_usd=300_000_000.0
        )
