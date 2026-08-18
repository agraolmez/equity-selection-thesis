from __future__ import annotations

import json
from pathlib import Path

from portfolio_bot.research_v6.news import (
    AlpacaNewsCredentials,
    download_news_month,
    load_alpaca_credentials,
)
from scripts.download_v6_alpaca_news import _months
from scripts.materialize_v6_news_features import _clean_text


class _TwoPageClient:
    def __init__(self) -> None:
        self.tokens: list[str | None] = []

    def fetch_page(self, *, start: str, end: str, page_token: str | None = None):
        self.tokens.append(page_token)
        if page_token is None:
            return {"news": [{"id": 1, "headline": "first"}], "next_page_token": "two"}
        return {"news": [{"id": 2, "headline": "second"}], "next_page_token": None}


def test_credentials_file_is_loaded_without_repr_leak(tmp_path: Path) -> None:
    source = tmp_path / "paper.env"
    source.write_text(
        "APCA_API_KEY_ID=paper-key\nAPCA_API_SECRET_KEY=paper-secret\n",
        encoding="utf-8",
    )
    credentials = load_alpaca_credentials(source)
    assert credentials == AlpacaNewsCredentials("paper-key", "paper-secret")
    assert "paper-key" not in repr(credentials)
    assert "paper-secret" not in repr(credentials)


def test_month_download_is_checkpointed_and_reusable(tmp_path: Path) -> None:
    client = _TwoPageClient()
    first = download_news_month(client, month="2019-12", output_root=tmp_path)
    assert first.article_count == 2
    assert first.page_count == 2
    assert client.tokens == [None, "two"]
    rows = [json.loads(line) for line in first.output_path.read_text().splitlines()]
    assert [row["id"] for row in rows] == [1, 2]

    second_client = _TwoPageClient()
    second = download_news_month(second_client, month="2019-12", output_root=tmp_path)
    assert second.resumed is True
    assert second.sha256 == first.sha256
    assert second_client.tokens == []


def test_month_range_is_inclusive() -> None:
    assert _months("2019-11", "2020-01") == ["2019-11", "2019-12", "2020-01"]


def test_feature_text_deduplicates_repeated_payload_segments() -> None:
    text, full = _clean_text(
        {
            "headline": "Profit rises",
            "summary": "Profit rises",
            "content": "<p>Demand is strong</p>",
        }
    )
    assert text == "Profit rises Demand is strong"
    assert full is True
