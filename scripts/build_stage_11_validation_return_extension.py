#!/usr/bin/env python3
"""Reconcile Stage 11 predictor returns through 2019-10-31 only.

The reserved 2019-11-01 through 2019-12-02 outcome interval is inaccessible
through this pre-unblind command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.validation.validation_market_extension import (  # noqa: E402
    DATABASE_ROOT,
    build_return_extension,
)


DEFAULT_BASE = DATABASE_ROOT / "interim/phase_05_validation/stage11-validation-data-v1"
DEFAULT_TYPED = DEFAULT_BASE / "market_extension/typed"
DEFAULT_OUTPUT = DEFAULT_BASE / "market_extension/returns"


def _progress(event: str, details: Mapping[str, Any]) -> None:
    print(json.dumps({"event": event, **dict(details)}, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--typed-directory", type=Path, default=DEFAULT_TYPED)
    parser.add_argument("--output-base-directory", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    receipt = build_return_extension(
        arguments.typed_directory.expanduser().resolve(),
        arguments.output_base_directory.expanduser().resolve(),
        progress=_progress,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
