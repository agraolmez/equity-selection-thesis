#!/usr/bin/env python3
"""Independently verify and publish the synthetic-only Stage 10 allocators."""

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

from portfolio_bot.allocation.contract import DEFAULT_POLICY
from portfolio_bot.allocation.independent_verification import (
    AllocatorVerificationError,
    verify_stage10_allocators,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate, replay, regress, and independently verify the "
            "synthetic-only Stage 10 allocator implementation."
        )
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--primary-pointer", type=Path)
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Run all checks without replacing the verified pointer.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_stage10_allocators(
            policy_path=args.policy,
            primary_pointer_path=args.primary_pointer,
            publish=not args.no_publish,
        )
    except (AllocatorVerificationError, RuntimeError, ValueError, OSError) as exc:
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
                "status": "VERIFIED_PASSED",
                "passed": result.passed,
                "build_id": result.build_id,
                "verification_id": result.verification_id,
                "verification_path": (
                    None
                    if result.verification_path is None
                    else str(result.verification_path)
                ),
                "verified_pointer_path": (
                    None
                    if result.verified_pointer_path is None
                    else str(result.verified_pointer_path)
                ),
                "difference_count": result.checks["difference_count"],
                "stage_11_validation_technical_gate_passed": True,
                "stage_11_requires_explicit_user_approval": True,
                "stage_11_execution_started": False,
                "stage_12_or_later_authorized": False,
                "final_period_access_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
