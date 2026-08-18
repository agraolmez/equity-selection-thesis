#!/usr/bin/env python3
"""Create the reviewed, report-ready Phase 1 summary artifacts."""

from __future__ import annotations

import json

from portfolio_bot.data.phase_01_review import write_review_artifacts


def main() -> int:
    paths = write_review_artifacts()
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
