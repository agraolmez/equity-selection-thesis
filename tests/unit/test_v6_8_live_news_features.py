from __future__ import annotations

import datetime as dt

import pandas as pd

from portfolio_bot.research_v6_8.live_news_features import build_live_news_features


def test_live_news_preserves_direct_family_and_confirmed_industry_rules() -> None:
    universe = pd.DataFrame(
        {
            "live_security_id": ["a", "b", "c", "d"],
            "TickerSymbol": ["AAA", "AAB", "BBB", "CCC"],
            "issuer_key": ["issuer-a", "issuer-a", "issuer-b", "issuer-c"],
        }
    )
    identity = pd.DataFrame(
        {
            "live_security_id": ["a", "b", "c", "d"],
            "sic2_industry_category": ["35", "35", "35", "60"],
        }
    )
    cutoff = dt.datetime(2026, 8, 14, 20, tzinfo=dt.timezone.utc)
    articles = [
        {
            "id": 1,
            "created_at": "2026-08-14T18:00:00Z",
            "updated_at": "2026-08-14T19:00:00Z",
            "symbols": ["AAA", "BBB"],
            "headline": "Strong gain",
            "summary": "uncertain outlook",
            "content": "Strong gain with uncertain outlook",
        }
    ]
    result, stats = build_live_news_features(
        universe=universe,
        identity=identity,
        articles=articles,
        cutoff=cutoff,
        lexicon=({"STRONG", "GAIN"}, set(), {"UNCERTAIN"}),
    )
    rows = result.set_index("live_security_id")
    assert rows.loc["a", "news_direct_1d_count"] == 1
    assert rows.loc["b", "news_direct_1d_count"] == 0
    assert rows.loc["b", "news_family_1d_count"] == 1
    assert rows.loc["c", "news_direct_1d_count"] == 1
    assert rows.loc["a", "news_industry_1d_count"] == 1
    assert rows.loc["b", "news_industry_1d_count"] == 1
    assert rows.loc["c", "news_industry_1d_count"] == 1
    assert rows.loc["d", "news_industry_1d_count"] == 0
    assert stats["mapped_articles"] == 1
    assert stats["related_rows"] == 3


def test_future_updated_article_is_excluded() -> None:
    universe = pd.DataFrame(
        {"live_security_id": ["a"], "TickerSymbol": ["AAA"], "issuer_key": ["issuer-a"]}
    )
    identity = pd.DataFrame(
        {"live_security_id": ["a"], "sic2_industry_category": ["35"]}
    )
    cutoff = dt.datetime(2026, 8, 14, 20, tzinfo=dt.timezone.utc)
    result, stats = build_live_news_features(
        universe=universe,
        identity=identity,
        articles=[
            {
                "id": 1,
                "created_at": "2026-08-14T18:00:00Z",
                "updated_at": "2026-08-14T21:00:00Z",
                "symbols": ["AAA"],
                "headline": "gain",
            }
        ],
        cutoff=cutoff,
        lexicon=({"GAIN"}, set(), set()),
    )
    assert result.loc[0, "news_related"] == 0
    assert stats["mapped_articles"] == 0
