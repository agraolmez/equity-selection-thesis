#!/usr/bin/env python3
"""Independently verify the Stage 11 predictor market/return extensions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.validation.validation_market_extension_verification import (  # noqa: E402
    independently_verify_validation_market_extensions,
    write_verification_report,
)


DATABASE_ROOT = Path("/Users/agraemirolmez/Desktop/Thesis")
DEFAULT_BASE = (
    DATABASE_ROOT
    / "interim/phase_05_validation/stage11-validation-data-v1/market_extension"
)
DEFAULT_TYPED = DEFAULT_BASE / "typed"
DEFAULT_RETURNS = DEFAULT_BASE / "returns"
DEFAULT_REPORT = (
    DATABASE_ROOT
    / "quality/phase_05_validation/stage11-validation-data-v1/"
    "market_extension_independent_verification.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--typed-directory", type=Path, default=DEFAULT_TYPED)
    parser.add_argument("--return-base-directory", type=Path, default=DEFAULT_RETURNS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args(argv)
    report = independently_verify_validation_market_extensions(
        arguments.typed_directory,
        arguments.return_base_directory,
    )
    write_verification_report(arguments.report, report)
    print(json.dumps(dict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
