#!/usr/bin/env python3
"""Build the current frozen risk/cost inputs for V6.8 paper trading."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_bot.research_v6_8.live_risk_cost import materialize_live_risk_cost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research_v6_8/live_risk_cost.json")
    parser.add_argument("--market-features", type=Path, required=True)
    parser.add_argument("--market-history", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--factor-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize_live_risk_cost(
        config_path=args.config.resolve(), market_features_path=args.market_features.resolve(),
        market_history_path=args.market_history.resolve(), identity_path=args.identity.resolve(),
        factor_snapshot=args.factor_snapshot.resolve(), output=args.output.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
