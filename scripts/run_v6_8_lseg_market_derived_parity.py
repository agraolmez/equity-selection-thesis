#!/usr/bin/env python3
"""Run the V6.8 stock-level LSEG derived-market parity gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_bot.research_v6_8.lseg_market_derived_parity import (
    run_lseg_market_derived_parity,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research_v6_8/lseg_market_derived_parity.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research_v6_8/lseg_market_derived_parity/20260817",
    )
    parser.add_argument(
        "--model-feature-months",
        type=Path,
        default=Path(
            "/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/model_features_v2/months"
        ),
    )
    parser.add_argument("--pilot", action="store_true")
    arguments = parser.parse_args()
    receipt = run_lseg_market_derived_parity(
        config_path=arguments.config.resolve(),
        output=arguments.output.resolve(),
        model_feature_months=arguments.model_feature_months.resolve(),
        pilot_only=arguments.pilot,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
