#!/usr/bin/env python3
"""Independently verify Stage 9 selector results and publish Stage-10 authority."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio_bot.selectors.contract import DEFAULT_POLICY
from portfolio_bot.selectors.independent_verification import (
    SelectorVerificationError,
    verify_stage9_selector_results,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently replay Stage 9 selector metrics and shortlist."
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--primary-pointer", type=Path)
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Run every verification check without replacing the verified pointer.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_stage9_selector_results(
            policy_path=args.policy,
            primary_pointer_path=args.primary_pointer,
            publish=not args.no_publish,
            progress=lambda message: print(
                f"[stage9-verify] {message}", file=sys.stderr, flush=True
            ),
        )
    except (SelectorVerificationError, RuntimeError, ValueError, OSError) as exc:
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
                "passed": result.passed,
                "build_id": result.build_id,
                "verification_id": result.verification_id,
                "difference_count": len(result.differences),
                "differences": [value.as_dict() for value in result.differences],
                "verified_pointer_path": (
                    None
                    if result.verified_pointer_path is None
                    else str(result.verified_pointer_path)
                ),
                "stage_10_authorized": bool(
                    result.passed and not args.no_publish
                ),
                "stage_11_or_later_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
