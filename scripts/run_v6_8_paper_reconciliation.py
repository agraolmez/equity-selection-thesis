#!/usr/bin/env python3
"""Reconcile a submitted V6.8 Alpaca paper rebalance after the close."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_bot.research_v6_8.paper_reconciliation import materialize_paper_reconciliation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployment-config",
        type=Path,
        default=ROOT / "configs/research_v6_8/alpaca_paper_deployment.json",
    )
    parser.add_argument("--deployment-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize_paper_reconciliation(
        deployment_config_path=args.deployment_config.resolve(),
        deployment_receipt_path=args.deployment_receipt.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS_FILLED_AND_RECONCILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
