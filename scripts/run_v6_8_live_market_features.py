#!/usr/bin/env python3
"""Build one checkpointed current V6.8 market-feature snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.research_v6_8.live_market_features import (
    materialize_live_market_features,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/research_v6_8/live_market_features.json"
    )
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-securities", type=int)
    args = parser.parse_args()
    receipt = materialize_live_market_features(
        config_path=args.config.resolve(),
        universe_path=args.universe.resolve(),
        output=args.output.resolve(),
        maximum_securities=args.maximum_securities,
    )
    print(json.dumps({"output": str(args.output.resolve()), **receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
