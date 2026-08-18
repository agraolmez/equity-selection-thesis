#!/usr/bin/env python3
"""Build the complete Stage 12 pre-final freeze without final-period access."""

from __future__ import annotations

import json
from pathlib import Path

from portfolio_bot.final_freeze.inventory import (
    build_environment_lock,
    build_project_file_inventory,
    canonical_json_bytes,
    load_json_object,
    sha256_bytes,
)
from portfolio_bot.final_freeze.publication import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
    prepare_stage12_freeze,
    publish_stage12_freeze,
    test_receipt_document,
)
from portfolio_bot.final_freeze.regression import (
    checkpoint_receipt_for_bundle,
    run_or_reuse_test_checkpoint,
)


def main() -> int:
    policy, _ = load_json_object(DEFAULT_POLICY)
    inventory = build_project_file_inventory(policy, project_root=PROJECT_ROOT)
    environment = build_environment_lock(project_root=PROJECT_ROOT)
    environment_sha = sha256_bytes(canonical_json_bytes(environment))
    checkpoint_root = (
        PROJECT_ROOT
        / str(policy["outputs"]["artifact_root"])
        / str(policy["outputs"]["checkpoint_namespace"])
    )
    focused = run_or_reuse_test_checkpoint(
        node_name="focused_stage12_protocol",
        command=tuple(str(value) for value in policy["tests"]["focused_command"]),
        project_root=PROJECT_ROOT,
        checkpoint_root=checkpoint_root,
        project_tree_sha256=str(inventory["tree_sha256"]),
        environment_sha256=environment_sha,
        extra_environment={"PYTHONHASHSEED": "0"},
    )
    independent = run_or_reuse_test_checkpoint(
        node_name="independent_synthetic_regression",
        command=tuple(
            str(value)
            for value in policy["tests"]["independent_synthetic_command"]
        ),
        project_root=PROJECT_ROOT,
        checkpoint_root=checkpoint_root,
        project_tree_sha256=str(inventory["tree_sha256"]),
        environment_sha256=environment_sha,
        dependency_ids=(focused.checkpoint_id,),
        extra_environment={"PYTHONHASHSEED": "0"},
    )
    current = build_project_file_inventory(policy, project_root=PROJECT_ROOT)
    if current["tree_sha256"] != inventory["tree_sha256"]:
        raise RuntimeError("project tree changed while running Stage 12 tests")
    tests = test_receipt_document(
        [
            checkpoint_receipt_for_bundle(focused, project_root=PROJECT_ROOT),
            checkpoint_receipt_for_bundle(
                independent, project_root=PROJECT_ROOT
            ),
        ]
    )
    prepared = prepare_stage12_freeze(tests)
    publication = publish_stage12_freeze(prepared)
    result = {
        "stage": 12,
        "freeze_id": publication.freeze_id,
        "status": prepared.policy["outputs"]["primary_status"],
        "bundle_path": publication.bundle.path.relative_to(PROJECT_ROOT).as_posix(),
        "pointer_path": publication.pointer_path.relative_to(PROJECT_ROOT).as_posix(),
        "pointer_sha256": publication.pointer_sha256,
        "manifest_sha256": publication.manifest_sha256,
        "focused_checkpoint_id": focused.checkpoint_id,
        "focused_checkpoint_reused": focused.reused_existing,
        "independent_checkpoint_id": independent.checkpoint_id,
        "independent_checkpoint_reused": independent.reused_existing,
        "final_period_rows_read": 0,
        "stage13_execution_authorized_now": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
