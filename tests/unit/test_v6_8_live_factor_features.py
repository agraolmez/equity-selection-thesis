from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from portfolio_bot.research_v6_8.live_factor_features import (
    canonicalize_factor_archives,
)


def _archive(name: str, text: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def test_factor_archives_merge_at_exact_monthly_and_daily_grains() -> None:
    payloads = {
        "ff5_monthly": _archive(
            "ff5.csv", ",Mkt-RF,SMB,HML,RMW,CMA,RF\n202606,1,2,3,4,5,0.3\n"
        ),
        "momentum_monthly": _archive("mom.csv", ",Mom\n202606,6\n"),
        "ff5_daily": _archive(
            "ff5d.csv",
            ",Mkt-RF,SMB,HML,RMW,CMA,RF\n20260629,1,2,3,4,5,0.01\n20260630,2,3,4,5,6,0.01\n",
        ),
        "momentum_daily": _archive(
            "momd.csv", ",Mom\n20260629,6\n20260630,7\n"
        ),
    }
    monthly, daily = canonicalize_factor_archives(
        payloads, fetched_at=datetime(2026, 8, 17, 10, tzinfo=timezone.utc)
    )
    assert monthly.iloc[0]["observation_date"].date().isoformat() == "2026-06-30"
    assert monthly.iloc[0]["mktrf"] == 0.01
    assert monthly.iloc[0]["umd"] == 0.06
    assert daily.iloc[-1]["umd"] == 0.07
    assert str(daily.iloc[-1]["public_availability_date"].date()) == "2026-08-17"
