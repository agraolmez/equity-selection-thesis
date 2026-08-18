#!/usr/bin/env python3
"""Build the development-only CRSP return reconciliation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_bot.data.phase_02_crsp_return_reconciliation import build, load_settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-database-root",
        type=Path,
        help="Optional output root for a non-authoritative rehearsal.",
    )
    parser.add_argument(
        "--no-publish-pointer",
        action="store_true",
        help="Do not advance the primary project pointer.",
    )
    args = parser.parse_args()
    settings = load_settings(output_database_root=args.output_database_root)
    publish_pointer = not args.no_publish_pointer and settings.is_production
    result = build(settings, publish_pointer=publish_pointer)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
