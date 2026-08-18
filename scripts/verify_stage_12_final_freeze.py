#!/usr/bin/env python3
"""Independently verify and close Stage 12 without authorizing Stage 13."""

from __future__ import annotations

import json

from portfolio_bot.final_freeze.inventory import (
    build_environment_lock,
    build_project_file_inventory,
    canonical_json_bytes,
    load_json_object,
    sha256_bytes,
)
from portfolio_bot.final_freeze.publication import DEFAULT_POLICY, PROJECT_ROOT
from portfolio_bot.final_freeze.regression import (
    checkpoint_receipt_for_bundle,
    run_or_reuse_test_checkpoint,
)
from portfolio_bot.final_freeze.verification import (
    replay_primary_stage12,
    verify_and_publish_stage12,
)


def main() -> int:
    policy, _ = load_json_object(DEFAULT_POLICY)
    replay = replay_primary_stage12()
    if replay.difference_count:
        raise RuntimeError(
            f"primary Stage 12 replay has {replay.difference_count} differences"
        )
    inventory = build_project_file_inventory(policy, project_root=PROJECT_ROOT)
    environment = build_environment_lock(project_root=PROJECT_ROOT)
    environment_sha = sha256_bytes(canonical_json_bytes(environment))
    primary_nodes = replay.prepared.documents["test_receipts.json"]["nodes"]
    dependencies = tuple(
        sorted(str(node["checkpoint_id"]) for node in primary_nodes)
    )
    checkpoint_root = (
        PROJECT_ROOT
        / str(policy["outputs"]["artifact_root"])
        / str(policy["outputs"]["checkpoint_namespace"])
    )
    focused = run_or_reuse_test_checkpoint(
        node_name="independent_stage12_focused_replay",
        command=tuple(str(value) for value in policy["tests"]["focused_command"]),
        project_root=PROJECT_ROOT,
        checkpoint_root=checkpoint_root,
        project_tree_sha256=str(inventory["tree_sha256"]),
        environment_sha256=environment_sha,
        dependency_ids=dependencies,
        extra_environment={"PYTHONHASHSEED": "0"},
    )
    receipt = checkpoint_receipt_for_bundle(focused, project_root=PROJECT_ROOT)
    publication = verify_and_publish_stage12(receipt)
    result = {
        "stage": 12,
        "verification_id": publication.verification_id,
        "status": policy["outputs"]["verified_status"],
        "verification_bundle_path": publication.bundle.path.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "verified_pointer_path": publication.pointer_path.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "verified_pointer_sha256": publication.pointer_sha256,
        "verification_report_sha256": publication.report_sha256,
        "difference_count": 0,
        "independent_checkpoint_id": focused.checkpoint_id,
        "independent_checkpoint_reused": focused.reused_existing,
        "stage13_technical_gate_passed": True,
        "stage13_technical_blockers": [],
        "stage13_requires_separate_user_approval": True,
        "stage13_execution_authorized_now": False,
        "final_period_access_authorized_now": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
