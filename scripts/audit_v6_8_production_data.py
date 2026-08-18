#!/usr/bin/env python3
"""Publish the frozen V6.8 feature-source and local-readiness audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_bot.research_v6_8.production_data import audit_production_sources


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research_v6_8/production_feature_sources.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research_v6_8/production_data_readiness/20260816",
    )
    args = parser.parse_args()
    readiness = audit_production_sources(args.config, args.output)
    print(json.dumps(readiness, indent=2, sort_keys=True))
    return 0 if readiness["status"] != "BLOCKED_FOR_PAPER_ORDERS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
