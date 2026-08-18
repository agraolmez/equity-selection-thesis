#!/usr/bin/env python3
"""Capture the read-only Alpaca paper state used by one V6.8 decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_bot.research_v6_8.live_broker_snapshot import materialize_live_broker_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-config", type=Path, default=ROOT / "configs/research_v6_8/alpaca_paper_deployment.json")
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize_live_broker_snapshot(
        deployment_config_path=args.deployment_config.resolve(),
        universe_path=args.universe.resolve(), output=args.output.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
