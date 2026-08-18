#!/usr/bin/env python3
"""Materialize current V6.8 security identity features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_bot.research_v6_8.live_identity_features import materialize_live_identity_features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research_v6_8/live_identity_features.json")
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fundamentals", type=Path)
    parser.add_argument("--sec-submissions", type=Path)
    args = parser.parse_args()
    receipt = materialize_live_identity_features(
        config_path=args.config.resolve(),
        universe_path=args.universe.resolve(),
        output=args.output.resolve(),
        fundamentals_path=None if args.fundamentals is None else args.fundamentals.resolve(),
        sec_submissions=None if args.sec_submissions is None else args.sec_submissions.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
