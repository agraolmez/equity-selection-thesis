#!/usr/bin/env python3
"""Recover an interrupted Stage 6 cache resume, then run the normal build."""

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
from portfolio_bot.features.feature_store_materialization import (
    compute_feature_store_build_identity,
)
from portfolio_bot.features.feature_store_publication import (
    publish_primary_feature_store,
)
from portfolio_bot.features.stage6_resume_recovery import (
    quarantine_identical_retired_candidates,
    reuse_authenticated_existing_batches,
)


def _progress(event: Mapping[str, Any]) -> None:
    print(json.dumps(dict(event), sort_keys=True, separators=(",", ":")), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-build-id", required=True)
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--crsp-checkpoint-rows", type=int, default=4_096)
    parser.add_argument("--quality-memory-limit-mb", type=int, default=1_024)
    args = parser.parse_args()

    inputs = resolve_input_catalog(authenticate_hashes=False)
    identity = compute_feature_store_build_identity(inputs)
    if identity.build_id != args.expected_build_id:
        raise RuntimeError(
            f"Stage 6 recovery identity differs: {identity.build_id} != {args.expected_build_id}"
        )
    recovery = quarantine_identical_retired_candidates(
        inputs=inputs,
        identity=identity,
        artifact_root=PROJECT_ROOT / "artifacts" / "phase_03_features",
    )
    _progress(
        {
            "event": "stage_06_resume_candidates_quarantined",
            "recovery_id": recovery.recovery_id,
            "moved_candidates": recovery.moved_candidates,
            "quarantine_path": None
            if recovery.quarantine_path is None
            else recovery.quarantine_path.as_posix(),
            "receipt_path": None
            if recovery.receipt_path is None
            else recovery.receipt_path.as_posix(),
        }
    )
    with reuse_authenticated_existing_batches():
        cached = materialize_stage6_family_cache(
            inputs,
            batch_size=args.batch_size,
            crsp_checkpoint_rows=args.crsp_checkpoint_rows,
            progress=_progress,
        )
    if cached.build_identity.build_id != args.expected_build_id:
        raise RuntimeError("Recovered Stage 6 build identity changed")
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
    _progress(
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
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
