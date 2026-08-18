#!/usr/bin/env python3
"""Assemble and encode one current V6.8 feature package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_bot.research_v6_8.live_feature_package import materialize_live_feature_package


def _group_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--feature-group must be GROUP=/absolute/path.parquet")
        group, raw_path = value.split("=", 1)
        if group in result:
            raise ValueError(f"duplicate feature group: {group}")
        result[group] = Path(raw_path).expanduser().resolve()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/research_v6_8/production_feature_sources.json")
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--feature-group", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize_live_feature_package(
        contract_path=args.contract.resolve(),
        universe_path=args.universe.resolve(),
        group_paths=_group_paths(args.feature_group),
        output=args.output.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
