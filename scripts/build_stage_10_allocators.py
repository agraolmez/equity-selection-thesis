#!/usr/bin/env python3
"""Publish the synthetic-only Stage 10 allocator implementation evidence."""

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
from portfolio_bot.allocation.publication import (
    AllocatorPublicationError,
    publish_primary_allocator_evidence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish the immutable, synthetic-only Stage 10 allocator primary "
            "bundle. This command does not read empirical payload rows or run a backtest."
        )
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = publish_primary_allocator_evidence(policy_path=args.policy)
    except (AllocatorPublicationError, RuntimeError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "publication_error",
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
                "status": result.pointer["status"],
                "build_id": result.build_id,
                "manifest_path": str(result.manifest_path),
                "pointer_path": str(result.pointer_path),
                "reused_existing": result.reused_existing,
                "synthetic_only": result.manifest["synthetic_evidence"]["scope"][
                    "synthetic_only"
                ],
                "independent_verification_authorized": result.pointer["authorizations"][
                    "independent_verification_authorized"
                ],
                "full_repository_regression": result.manifest["pending_gates"][
                    "full_repository_regression"
                ],
                "independent_verification": result.manifest["pending_gates"][
                    "independent_verification"
                ],
                "decision_focused_empirical_production_training_ready": result.manifest[
                    "summary"
                ]["decision_focused_empirical_production_training_ready"],
                "stage_11_authorized": result.pointer["authorizations"][
                    "stage_11_validation_authorized"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
