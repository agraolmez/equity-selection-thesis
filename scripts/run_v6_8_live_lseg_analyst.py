#!/usr/bin/env python3
"""Materialize current LSEG analyst/CAM features for V6.8 paper trading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_bot.research_v6_8.live_lseg_analyst import (
    materialize_live_lseg_analyst_features,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research_v6_8/live_lseg_analyst_features.json"),
    )
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--market-features", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(
            "/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/model_features_v2/feature_registry.parquet"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize_live_lseg_analyst_features(
        config_path=args.config,
        universe_path=args.universe,
        market_features_path=args.market_features,
        registry_path=args.registry,
        output=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
