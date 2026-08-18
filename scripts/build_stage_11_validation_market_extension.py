#!/usr/bin/env python3
"""Build the isolated CRSP predictor context through 2019-10-31.

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
    build_typed_market_extension,
)


DEFAULT_OUTPUT = (
    DATABASE_ROOT
    / "interim/phase_05_validation/stage11-validation-data-v1/market_extension/typed"
)


def _progress(event: str, details: Mapping[str, Any]) -> None:
    print(json.dumps({"event": event, **dict(details)}, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--block-size-mib", type=int, default=64)
    arguments = parser.parse_args(argv)
    result = build_typed_market_extension(
        arguments.output_directory.expanduser().resolve(),
        block_size_bytes=arguments.block_size_mib << 20,
        progress=_progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
