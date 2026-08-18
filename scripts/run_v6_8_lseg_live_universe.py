#!/usr/bin/env python3
"""Materialize one current LSEG universe snapshot for V6.8 paper trading."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.research_v6_8.lseg_live_universe import discover_live_universe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/research_v6_8/lseg_live_universe.json"
    )
    parser.add_argument("--signal-date", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Users/agraemirolmez/Desktop/Thesis/raw/research_v6_8/lseg_universe"),
    )
    args = parser.parse_args()
    output = args.output_root.resolve() / args.signal_date.isoformat()
    receipt = discover_live_universe(
        config_path=args.config.resolve(), output=output, signal_date=args.signal_date
    )
    print(json.dumps({"output": str(output), **receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
