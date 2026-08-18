#!/usr/bin/env python3
"""Build and publish the locked Phase 2 typed CRSP primary candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.data.phase_02_crsp_typed_backbone import (  # noqa: E402
    load_production_settings,
)
from portfolio_bot.data.phase_02_crsp_typed_backbone_publication import (  # noqa: E402
    publish_primary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block-size-mib",
        type=int,
        default=16,
        help="Arrow CSV block size in MiB (default: 16)",
    )
    args = parser.parse_args()
    if args.block_size_mib <= 0:
        parser.error("--block-size-mib must be positive")

    def progress(stage: str, details: object) -> None:
        print(
            json.dumps(
                {"details": details, "stage": stage},
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    settings = load_production_settings(
        block_size_bytes=args.block_size_mib << 20
    )
    pointer = publish_primary(settings, progress=progress)
    print(json.dumps(pointer, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
