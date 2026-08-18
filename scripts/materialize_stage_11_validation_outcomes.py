#!/usr/bin/env python3
"""Sealed Stage 11 outcome/label/daily-market materialization entrypoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.validation.outcome_materialization import (  # noqa: E402
    Stage11OutcomeMaterializationError,
    materialize_from_frozen_manifest,
)
from portfolio_bot.validation.validation_runner import (  # noqa: E402
    Stage11ValidationSealedError,
    open_stage11_validation_unblind,
)


def main(argv: list[str] | None = None) -> int:
    selected = sys.argv[1:] if argv is None else argv
    if selected:
        print(
            json.dumps(
                {
                    "status": "validation_sealed",
                    "error_type": "Stage11ValidationSealedError",
                    "error": "caller-supplied paths and options are prohibited",
                    "outcome_rows_read": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        with open_stage11_validation_unblind() as authority:
            result = materialize_from_frozen_manifest(authority)
            print(json.dumps(dict(result.receipt), sort_keys=True, indent=2))
            return 0
    except (
        Stage11ValidationSealedError,
        Stage11OutcomeMaterializationError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "validation_sealed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "outcome_rows_read": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
