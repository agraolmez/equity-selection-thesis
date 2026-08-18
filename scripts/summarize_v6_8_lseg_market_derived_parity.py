#!/usr/bin/env python3
"""Summarize the completed V6.8 stock-level LSEG market parity run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_bot.research_v6_8.lseg_market_derived_summary import (
    summarize_lseg_market_derived_parity,
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
        "--run-directory",
        type=Path,
        default=ROOT / "artifacts/research_v6_8/lseg_market_derived_parity/20260817",
    )
    arguments = parser.parse_args()
    result = summarize_lseg_market_derived_parity(
        config_path=arguments.config.resolve(),
        run_directory=arguments.run_directory.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
