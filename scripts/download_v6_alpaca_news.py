#!/usr/bin/env python3
"""Download checkpointed monthly Alpaca news without trading API access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_bot.research_v6.news import (
    AlpacaNewsClient,
    download_news_month,
    load_alpaca_credentials,
)


DEFAULT_OUTPUT = Path(
    "/Users/agraemirolmez/Desktop/Thesis/raw/research_v6/alpaca_news"
)


def _months(first: str, last: str) -> list[str]:
    import pandas as pd

    start = pd.Period(first, freq="M")
    end = pd.Period(last, freq="M")
    if start > end:
        raise ValueError("start month must not be after end month")
    return [str(value) for value in pd.period_range(start, end, freq="M")]


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--month", help="one calendar month in YYYY-MM form")
    group.add_argument("--start-month", help="first month in a batch")
    parser.add_argument("--end-month", help="last month in a batch")
    parser.add_argument(
        "--priority-month",
        help="download this month first when it is inside the requested batch",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--credentials-file", type=Path)
    args = parser.parse_args()

    credentials = load_alpaca_credentials(args.credentials_file)
    if args.start_month and not args.end_month:
        parser.error("--start-month requires --end-month")
    if args.end_month and not args.start_month:
        parser.error("--end-month requires --start-month")
    months = [args.month] if args.month else _months(args.start_month, args.end_month)
    if args.priority_month in months:
        months.remove(args.priority_month)
        months.insert(0, args.priority_month)
    client = AlpacaNewsClient(credentials)
    for month in months:
        result = download_news_month(client, month=month, output_root=args.output_root)
        print(
            json.dumps(
                {
                    "month": result.month,
                    "article_count": result.article_count,
                    "page_count": result.page_count,
                    "sha256": result.sha256,
                    "output_path": str(result.output_path),
                    "manifest_path": str(result.manifest_path),
                    "resumed": result.resumed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
