#!/usr/bin/env python3
"""Compare retired and corrected Stage 9 prepared matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_bot.selectors.prepared_invariance import (
    compare_prepared_caches,
    write_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--old-build-id", required=True)
    parser.add_argument("--new-build-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=4096)
    args = parser.parse_args()
    receipt = compare_prepared_caches(
        args.old_root,
        args.new_root,
        old_build_id=args.old_build_id,
        new_build_id=args.new_build_id,
        batch_rows=args.batch_rows,
        progress=lambda message: print(message, flush=True),
    )
    receipt_sha256 = write_receipt(receipt, args.output)
    print(
        json.dumps(
            {
                "comparison_id": receipt["comparison_id"],
                "output": str(args.output.resolve()),
                "receipt_sha256": receipt_sha256,
                "status": receipt["status"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
