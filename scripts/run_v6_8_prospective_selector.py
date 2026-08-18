#!/usr/bin/env python3
"""Refit frozen monthly selectors and score one current paper universe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_bot.research_v6_8.prospective_selector import materialize_prospective_selector_scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research_v6_8/prospective_selector.json")
    parser.add_argument("--feature-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    receipt = materialize_prospective_selector_scores(
        config_path=args.config.resolve(),
        feature_package=args.feature_package.resolve(),
        output=args.output.resolve(),
        state_root=None if args.state is None else args.state.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
