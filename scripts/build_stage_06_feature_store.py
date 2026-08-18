#!/usr/bin/env python3
"""Materialize, audit, and publish the frozen Stage 6 feature store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio_bot.features.feature_store_family_orchestration import (
    materialize_stage6_family_cache,
)
from portfolio_bot.features.feature_store_inputs import resolve_input_catalog
from portfolio_bot.features.feature_store_publication import (
    publish_primary_feature_store,
)


def _progress(event: Mapping[str, Any]) -> None:
    print(
        json.dumps(dict(event), sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build only the 2000–2014 development feature store. "
            "Labels, models, portfolios, P&L, validation, and final-test "
            "payloads remain prohibited."
        )
    )
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--crsp-checkpoint-rows", type=int, default=4_096)
    parser.add_argument("--crsp-spool-path", type=Path)
    parser.add_argument("--quality-memory-limit-mb", type=int, default=1_024)
    args = parser.parse_args()

    inputs = resolve_input_catalog(authenticate_hashes=False)
    cached = materialize_stage6_family_cache(
        inputs,
        batch_size=args.batch_size,
        crsp_checkpoint_rows=args.crsp_checkpoint_rows,
        crsp_spool_path=args.crsp_spool_path,
        progress=_progress,
    )
    published = publish_primary_feature_store(
        inputs=inputs,
        build_identity=cached.build_identity,
        family_batches=cached,
        provenance_counters={
            "optionmetrics_total_rows_used": 0,
            "raw_pastor_stambaugh_innovation_used": 0,
        },
        quality_memory_limit_mb=args.quality_memory_limit_mb,
    )
    print(
        json.dumps(
            {
                "event": "stage_06_primary_complete",
                "build_id": published.build_id,
                "interim_path": published.interim_path.as_posix(),
                "processed_path": published.processed_path.as_posix(),
                "quality_path": published.quality_path.as_posix(),
                "manifest_path": published.manifest_path.as_posix(),
                "pointer_path": published.pointer_path.as_posix(),
                "reused_existing": published.reused_existing,
                "stage_07_authorized": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
