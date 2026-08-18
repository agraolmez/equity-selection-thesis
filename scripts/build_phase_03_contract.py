#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portfolio_bot.features.phase_03_contract import main


if __name__ == "__main__":
    raise SystemExit(main())
