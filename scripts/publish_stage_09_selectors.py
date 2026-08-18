#!/usr/bin/env python3
"""Publish one completed Stage 9 selector-search bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio_bot.selectors.contract import DEFAULT_POLICY
from portfolio_bot.selectors.publication import (
    SelectorPublicationError,
    publish_primary_selector_results,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish immutable development-only Stage 9 selector results."
    )
    parser.add_argument(
        "--search-bundle",
        required=True,
        type=Path,
        help="Closed completed-search directory beneath the configured database root.",
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = publish_primary_selector_results(
            args.search_bundle,
            policy_path=args.policy,
        )
    except (SelectorPublicationError, RuntimeError, ValueError, OSError) as exc:
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
                "stage_10_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
