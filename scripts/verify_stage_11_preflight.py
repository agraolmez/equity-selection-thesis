#!/usr/bin/env python3
"""Independently replay and verify the Stage 11 no-outcome preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.validation.contract import DEFAULT_POLICY  # noqa: E402
from portfolio_bot.validation.preflight import (  # noqa: E402
    Stage11PreflightRunError,
    independently_verify_stage11_preflight,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--primary-pointer", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = independently_verify_stage11_preflight(
            policy_path=args.policy,
            primary_pointer_path=args.primary_pointer,
        )
    except (Stage11PreflightRunError, RuntimeError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "verification_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "PREFLIGHT_VERIFIED_PASSED",
                "build_id": result.build_id,
                "verification_id": result.verification_id,
                "verification_path": result.bundle_path.as_posix(),
                "verified_pointer_path": result.pointer_path.as_posix(),
                "difference_count": 0,
                "outcome_rows_read": 0,
                "validation_unblind_authorized_now": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
