#!/usr/bin/env python3
"""Run the audited router-only Stage 6 cache migration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio_bot.features.feature_store_inputs import resolve_input_catalog
from portfolio_bot.features.feature_store_materialization import (
    compute_feature_store_build_identity,
)
from portfolio_bot.features.migrate_crsp_router_cache import (
    STOPPED_BUILD_CONFIRMATION,
    RouterMigrationSettings,
    contract_from_policy,
    migrate_router_cache,
    production_router_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate only a complete Stage 6 CRSP router after the running "
            "build has been stopped. Family outputs are never reused."
        )
    )
    parser.add_argument("--old-build-id", required=True)
    parser.add_argument("--expected-new-build-id", required=True)
    parser.add_argument(
        "--old-identity-receipt",
        type=Path,
        required=True,
        help="An authenticated old cache _SUCCESS.json used only for identity hashes.",
    )
    parser.add_argument(
        "--confirm-build-stopped",
        action="store_true",
        help="Required acknowledgement; this command never stops the build itself.",
    )
    parser.add_argument("--comparison-batch-rows", type=int, default=1_024)
    args = parser.parse_args()
    if not args.confirm_build_stopped:
        parser.error("--confirm-build-stopped is required")

    inputs = resolve_input_catalog(authenticate_hashes=False)
    identity = compute_feature_store_build_identity(inputs)
    if identity.build_id != args.expected_new_build_id:
        parser.error(
            "Current authenticated identity does not equal --expected-new-build-id"
        )
    result = migrate_router_cache(
        RouterMigrationSettings(
            router_root=production_router_root(inputs.database_root),
            artifact_root=PROJECT_ROOT / "artifacts" / "phase_03_features",
            old_build_id=args.old_build_id,
            expected_new_build_id=args.expected_new_build_id,
            old_identity_receipt=args.old_identity_receipt,
            build_stopped_confirmation=STOPPED_BUILD_CONFIRMATION,
            comparison_batch_rows=args.comparison_batch_rows,
        ),
        new_identity=identity,
        contract=contract_from_policy(),
    )
    print(
        json.dumps(
            {
                "event": "stage_06_crsp_router_migration_complete",
                "old_build_id": result.old_build_id,
                "new_build_id": result.new_build_id,
                "target_directory": result.target_directory.as_posix(),
                "quarantined_partial_directory": (
                    result.quarantined_partial_directory.as_posix()
                ),
                "receipt_path": result.receipt_path.as_posix(),
                "receipt_file_sha256": result.receipt_file_sha256,
                "family_outputs_reused": False,
                "raw_files_written": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
