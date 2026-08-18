"""Run independent Stage 6 feature-store verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portfolio_bot.features.verify_feature_store import (
    DEFAULT_POLICY,
    Stage6VerificationError,
    verify_stage6_feature_store,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--primary-pointer", type=Path)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--raw-batch-rows", type=int, default=16_384)
    parser.add_argument("--lineage-batch-rows", type=int, default=2_048)
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Run every check without publishing verification authority.",
    )
    arguments = parser.parse_args()
    try:
        result = verify_stage6_feature_store(
            primary_manifest_path=arguments.primary_manifest,
            primary_pointer_path=arguments.primary_pointer,
            database_root=arguments.database_root,
            artifact_root=arguments.artifact_root,
            policy_path=arguments.policy,
            publish=not arguments.no_publish,
            raw_batch_rows=arguments.raw_batch_rows,
            lineage_batch_rows=arguments.lineage_batch_rows,
        )
    except Stage6VerificationError as exc:
        print(
            json.dumps(
                {"status": "verification_error", "error": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    payload = {
        "status": "verified_passed" if result.passed else "verified_failed",
        "build_id": result.build_id,
        "verification_id": result.verification_id,
        "differences": [
            difference.as_dict() for difference in result.differences
        ],
        "verification_path": (
            None
            if result.verification_path is None
            else result.verification_path.as_posix()
        ),
        "verified_pointer_path": (
            None
            if result.verified_pointer_path is None
            else result.verified_pointer_path.as_posix()
        ),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
