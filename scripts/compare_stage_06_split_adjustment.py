"""Run the pinned old-versus-corrected Stage 6 scientific comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portfolio_bot.features.stage6_split_adjustment_comparison import (
    Stage6ComparisonError,
    compare_stage6_split_adjustment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--batch-rows", type=int, default=16_384)
    arguments = parser.parse_args()
    try:
        result = compare_stage6_split_adjustment(
            project_root=arguments.project_root,
            database_root=arguments.database_root,
            batch_rows=arguments.batch_rows,
        )
    except Stage6ComparisonError as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "comparison_id": result["comparison_id"],
        "receipt_path": result["receipt_path"],
        "csv_path": result["csv_path"],
        "summary": result["summary"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
