#!/usr/bin/env python3
"""Materialize one explicitly authorized V6.8 provider-mismatch fallback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_bot.research_v6_8.live_missing_fallbacks import materialize_missing_fallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research_v6_8/live_missing_fallbacks.json")
    parser.add_argument(
        "--group",
        required=True,
        choices=("valuation_fundamentals", "institutional_ownership", "optionmetrics"),
    )
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path,
        default=Path("/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/model_features_v2/feature_registry.parquet"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize_missing_fallback(
        config_path=args.config.resolve(), group=args.group,
        universe_path=args.universe.resolve(), registry_path=args.registry.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
