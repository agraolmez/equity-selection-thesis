#!/usr/bin/env python3
"""Build one point-in-time Alpaca related-news block for V6.8 paper trading."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_bot.research_v6_8.live_news_features import materialize_live_news_features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research_v6_8/live_news_features.json")
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/model_features_v2/feature_registry.parquet"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize_live_news_features(
        config_path=args.config.resolve(),
        universe_path=args.universe.resolve(),
        identity_path=args.identity.resolve(),
        registry_path=args.registry.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
