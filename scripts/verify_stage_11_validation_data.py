#!/usr/bin/env python3
"""Independently verify a Stage 11 no-outcome validation-data candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.validation.validation_data_contract import DEFAULT_POLICY  # noqa: E402
from portfolio_bot.validation.validation_data_independent_verification import (  # noqa: E402
    independently_verify_candidate_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = independently_verify_candidate_manifest(
            args.manifest,
            args.candidate_root,
            policy_path=args.policy,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "validation_data_verification_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(dict(result.report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
