#!/usr/bin/env python3
"""Print a read-only JSON or Markdown summary of Stage 9 selector results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio_bot.selectors.contract import DEFAULT_POLICY
from portfolio_bot.selectors.reporting import (
    SelectorReportingError,
    render_selector_results_markdown,
    summarize_selector_results,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate and summarize a completed Stage 9 search bundle or "
            "primary pointer without fitting, publishing, or changing authority."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--search-bundle",
        type=Path,
        help="Sealed completed-search directory under the configured database root.",
    )
    source.add_argument(
        "--primary-pointer",
        type=Path,
        help="Published Stage 9 primary pointer to authenticate and summarize.",
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output representation written to standard output (default: json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = summarize_selector_results(
            search_bundle_path=args.search_bundle,
            primary_pointer_path=args.primary_pointer,
            policy_path=args.policy,
        )
        if args.format == "markdown":
            output = render_selector_results_markdown(summary)
        else:
            output = json.dumps(
                summary,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
    except (SelectorReportingError, RuntimeError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "stage09_summary_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
