from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from portfolio_bot.research_v6_8.live_sec_features import (
    normalize_single_cik,
    parse_sec_submissions,
)


def test_multi_cik_is_not_guessed() -> None:
    assert normalize_single_cik("0000320193") == "0000320193"
    assert normalize_single_cik("0000320193|0001234567") is None


def test_sec_parser_respects_exact_signal_close_and_splits_8k_items() -> None:
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["a", "b"],
                "filingDate": ["2026-08-17", "2026-08-17"],
                "reportDate": ["2026-06-30", "2026-06-30"],
                "acceptanceDateTime": ["2026-08-17T15:30:00-04:00", "2026-08-17T16:30:00-04:00"],
                "form": ["8-K", "8-K"],
                "items": ["2.02,9.01", "4.02"],
            }
        }
    }
    filings, items = parse_sec_submissions(
        payload,
        cik="0000320193",
        signal_cutoff=datetime(
            2026, 8, 17, 16, 0, tzinfo=ZoneInfo("America/New_York")
        ),
    )
    assert [row["source_record_id"] for row in filings] == ["a"]
    assert [row["item_number"] for row in items] == ["2.02", "9.01"]


def test_sec_parser_deduplicates_identical_accessions() -> None:
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["same", "same"],
                "filingDate": ["2026-08-17", "2026-08-17"],
                "reportDate": ["2026-06-30", "2026-06-30"],
                "acceptanceDateTime": [
                    "2026-08-17T15:30:00-04:00",
                    "2026-08-17T15:30:00-04:00",
                ],
                "form": ["8-K", "8-K"],
                "items": ["2.02,9.01", "2.02,9.01"],
            }
        }
    }
    filings, items = parse_sec_submissions(
        payload,
        cik="0000320193",
        signal_cutoff=datetime(
            2026, 8, 17, 16, 0, tzinfo=ZoneInfo("America/New_York")
        ),
    )
    assert [row["source_record_id"] for row in filings] == ["same"]
    assert [row["item_number"] for row in items] == ["2.02", "9.01"]


def test_sec_parser_ignores_conflicting_irrelevant_form_duplicates() -> None:
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["same", "same"],
                "filingDate": ["2026-07-08", "2026-07-08"],
                "reportDate": ["", ""],
                "acceptanceDateTime": [
                    "2026-07-08T20:41:20Z",
                    "2026-07-08T16:41:20Z",
                ],
                "form": ["SC 13D/A", "SC TO-T/A"],
                "items": ["", ""],
            }
        }
    }
    filings, items = parse_sec_submissions(
        payload,
        cik="0001326200",
        signal_cutoff=datetime(
            2026, 8, 17, 16, 0, tzinfo=ZoneInfo("America/New_York")
        ),
    )
    assert filings == []
    assert items == []
