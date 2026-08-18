#!/usr/bin/env python3
"""Repair only missing V6.8 LSEG market chunks at a conservative request rate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_bot.research_v6_8.live_market_repair import (
    LiveMarketRepairQuotaExhausted,
    LiveMarketRepairRateLimited,
    repair_missing_market_chunks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research_v6_8/live_market_features.json",
    )
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-batch-size", type=int, default=5)
    parser.add_argument("--request-pause-seconds", type=float, default=3.0)
    args = parser.parse_args()
    try:
        receipt = repair_missing_market_chunks(
            config_path=args.config.resolve(),
            universe_path=args.universe.resolve(),
            output=args.output.resolve(),
            repair_batch_size=args.repair_batch_size,
            request_pause_seconds=args.request_pause_seconds,
        )
    except LiveMarketRepairQuotaExhausted as error:
        print(json.dumps({"status": "DAILY_QUOTA_EXHAUSTED", "message": str(error)}))
        return 75
    except LiveMarketRepairRateLimited as error:
        print(json.dumps({"status": "RATE_LIMITED", "message": str(error)}))
        return 75
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
