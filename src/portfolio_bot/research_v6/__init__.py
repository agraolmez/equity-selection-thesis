"""Lean V6 selector research components.

V6 is an additive lineage.  It may read authenticated predecessor data, but it
never writes into or mutates a closed predecessor publication.
"""

from portfolio_bot.research_v6.news import (
    AlpacaNewsClient,
    AlpacaNewsCredentials,
    NewsDownloadResult,
    download_news_month,
    load_alpaca_credentials,
)

__all__ = [
    "AlpacaNewsClient",
    "AlpacaNewsCredentials",
    "NewsDownloadResult",
    "download_news_month",
    "load_alpaca_credentials",
]
