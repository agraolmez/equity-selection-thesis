#!/usr/bin/env python3
"""Run the standalone Phase 2 typed-CRSP-backbone verifier."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.data.verify_phase_02_crsp_typed_backbone import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
