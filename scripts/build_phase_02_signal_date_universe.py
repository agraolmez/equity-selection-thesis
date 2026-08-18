#!/usr/bin/env python3
"""Build the development-only Stage 2 signal-date universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_bot.data.phase_02_signal_date_universe_publication import build


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-database-root", type=Path)
    parser.add_argument("--no-publish-pointer", action="store_true")
    args = parser.parse_args()
    result = build(
        output_database_root=args.output_database_root,
        publish_pointer=not args.no_publish_pointer,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
