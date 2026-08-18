#!/usr/bin/env python3
"""Build one current frozen V6.8 paper target book."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_bot.research_v6_8.prospective_portfolio import materialize_prospective_portfolio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research_v6_8/prospective_portfolio.json")
    parser.add_argument("--selector-scores", type=Path, required=True)
    parser.add_argument("--risk-cost", type=Path, required=True)
    parser.add_argument("--factor-covariance", type=Path, required=True)
    parser.add_argument("--broker-snapshot", type=Path, required=True)
    parser.add_argument("--prior-signal", type=Path)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize_prospective_portfolio(
        config_path=args.config.resolve(), selector_scores_path=args.selector_scores.resolve(),
        risk_cost_path=args.risk_cost.resolve(), factor_covariance_path=args.factor_covariance.resolve(),
        broker_snapshot=args.broker_snapshot.resolve(), output=args.output.resolve(),
        prior_signal_path=None if args.prior_signal is None else args.prior_signal.resolve(),
        technical_smoke=args.technical_smoke,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
