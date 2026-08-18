#!/usr/bin/env python3
"""Run the exact Phase 1 suite and bind JUnit to unchanged implementation bytes."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
import xml.etree.ElementTree as et
from pathlib import Path
from typing import Any

from portfolio_bot.data.phase_01_evidence import (
    PHASE_01_TEST_IDS,
    implementation_receipt,
    stable_file_bytes_receipt,
    stable_file_receipt,
)
from portfolio_bot.data.phase_01_review import PROJECT_ROOT, load_review_context


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _stable_bytes(root: Path, path: Path) -> tuple[bytes, dict[str, Any]]:
    display_path = path.relative_to(root).as_posix()
    return stable_file_bytes_receipt(
        path, display_path=display_path, include_identity=True
    )


def _stable_json(root: Path, path: Path) -> dict[str, Any]:
    raw, receipt = _stable_bytes(root, path)
    return {
        "receipt": receipt,
        "payload": json.loads(raw.decode("utf-8")),
    }


def _minimal_implementation(
    receipts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {key: item[key] for key in ("path", "size_bytes", "sha256")}
        for item in receipts
    ]


def _source_selection(root: Path, context: dict[str, Any]) -> dict[str, Any]:
    artifact_root = context["artifact_root"]
    summary = context["summary"]
    verified = _stable_json(root, artifact_root / "LATEST_VERIFIED.json")
    primary = _stable_json(root, artifact_root / "LATEST_PRIMARY.json")
    expected_verified = {
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "verification_id": summary["verification_id"],
        "primary_path": context["primary_directory"]
        .relative_to(artifact_root)
        .as_posix(),
        "verification_path": context["verification_directory"]
        .relative_to(artifact_root)
        .as_posix(),
        "verification_status": "passed",
    }
    expected_primary = {
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "path": expected_verified["primary_path"],
        "primary_status": "primary_passed_pending_independent_verification",
        "verification_status": "not_run",
    }
    if verified["payload"] != expected_verified:
        raise RuntimeError("LATEST_VERIFIED does not exactly select the tested context")
    if primary["payload"] != expected_primary:
        raise RuntimeError("LATEST_PRIMARY does not exactly select the tested context")
    return {
        "artifact_root": artifact_root.relative_to(root).as_posix(),
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "verification_id": summary["verification_id"],
        "selected_primary_directory": context["primary_directory"]
        .relative_to(root)
        .as_posix(),
        "selected_verification_directory": context["verification_directory"]
        .relative_to(root)
        .as_posix(),
        "latest_verified": verified,
        "latest_primary": primary,
    }


def _junit_summary(
    root: Path, path: Path
) -> tuple[dict[str, int], list[str], dict[str, Any]]:
    raw, receipt = _stable_bytes(root, path)
    document = et.fromstring(raw)
    if document.tag == "testsuite":
        suites = [document]
    elif document.tag == "testsuites":
        suites = list(document.findall("testsuite"))
    else:
        raise RuntimeError(f"Unexpected JUnit root element: {document.tag}")
    if not suites:
        raise RuntimeError("JUnit contains no test suites")

    cases = [case for suite in suites for case in suite.findall("testcase")]
    result_counts = {
        result: sum(len(case.findall(result)) for case in cases)
        for result in ("failure", "error", "skipped")
    }
    suite_summary = {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
    }
    observed_summary = {
        "tests": len(cases),
        "failures": result_counts["failure"],
        "errors": result_counts["error"],
        "skipped": result_counts["skipped"],
    }
    if suite_summary != observed_summary:
        raise RuntimeError(
            "JUnit suite counters disagree with testcase result elements: "
            f"suite={suite_summary}, testcase={observed_summary}"
        )
    if any(not case.attrib.get(key) for case in cases for key in ("classname", "name")):
        raise RuntimeError("JUnit testcase is missing classname or name")

    summary = {
        "tests": len(cases),
        "failures": result_counts["failure"],
        "errors": result_counts["error"],
        "skipped": result_counts["skipped"],
    }
    test_ids = sorted(
        f"{case.attrib.get('classname')}::{case.attrib.get('name')}"
        for case in cases
    )
    if len(test_ids) != len(set(test_ids)):
        raise RuntimeError("JUnit contains duplicate testcase IDs")
    junit = {
        **receipt,
        "root_element": document.tag,
        "suite_count": len(suites),
        "testcase_count": len(cases),
        "child_result_counts": result_counts,
    }
    return summary, test_ids, junit


def main() -> int:
    project_root = PROJECT_ROOT.resolve()
    context = load_review_context(project_root)
    summary = context["summary"]
    run_directory = (
        project_root
        / "runs"
        / "phase_01_raw_snapshot"
        / summary["phase_01_build_id"]
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    junit_path = run_directory / "pytest.xml"
    receipt_path = run_directory / "test_execution_receipt.json"
    started = dt.datetime.now(dt.timezone.utc)
    source_selection_before = _source_selection(project_root, context)
    implementation_integrity_before = implementation_receipt(
        project_root, include_identity=True
    )
    implementation_before = _minimal_implementation(implementation_integrity_before)
    python_executable = Path(sys.executable).resolve(strict=True)
    executable_before = stable_file_receipt(
        python_executable,
        display_path=str(python_executable),
        include_identity=True,
    )
    command = [
        str(python_executable),
        "-m",
        "pytest",
        "-q",
        "tests",
        f"--junitxml={junit_path}",
    ]
    completed = subprocess.run(command, cwd=project_root, check=False, text=True)
    implementation_integrity_after = implementation_receipt(
        project_root, include_identity=True
    )
    implementation_after = _minimal_implementation(implementation_integrity_after)
    if implementation_integrity_after != implementation_integrity_before:
        raise RuntimeError("Phase 1 implementation changed while pytest was running")
    executable_after = stable_file_receipt(
        python_executable,
        display_path=str(python_executable),
        include_identity=True,
    )
    if executable_after != executable_before:
        raise RuntimeError("Python executable changed while pytest was running")
    source_selection_after = _source_selection(project_root, context)
    if source_selection_after != source_selection_before:
        raise RuntimeError("Phase 1 source selection changed while pytest was running")
    test_summary, test_ids, junit = _junit_summary(project_root, junit_path)
    expected_summary = {
        "tests": len(PHASE_01_TEST_IDS),
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    if (
        type(completed.returncode) is not int
        or completed.returncode != 0
        or test_summary != expected_summary
        or test_ids != sorted(PHASE_01_TEST_IDS)
        or junit["root_element"] != "testsuites"
        or junit["suite_count"] != 1
        or junit["testcase_count"] != len(PHASE_01_TEST_IDS)
        or junit["child_result_counts"]
        != {"failure": 0, "error": 0, "skipped": 0}
    ):
        raise RuntimeError(
            f"Phase 1 exact test contract failed: return={completed.returncode}, "
            f"summary={test_summary}, ids={test_ids}"
        )
    completed_at = dt.datetime.now(dt.timezone.utc)
    receipt = {
        "schema_version": 2,
        "receipt_type": "phase_01_exact_test_execution",
        "status": "passed",
        "run_started_utc": started.isoformat(),
        "run_completed_utc": completed_at.isoformat(),
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "verification_id": summary["verification_id"],
        "cwd": str(project_root),
        "runtime": {
            "python_invocation": sys.executable,
            "python_executable": str(python_executable),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "pytest_version": importlib.metadata.version("pytest"),
            "executable_before": executable_before,
            "executable_after": executable_after,
        },
        "command": command,
        "return_code": completed.returncode,
        "source_selection_before": source_selection_before,
        "source_selection_after": source_selection_after,
        "test_summary": test_summary,
        "test_ids": test_ids,
        "junit": junit,
        "implementation_before": implementation_before,
        "implementation_after": implementation_after,
        "implementation_integrity_before": implementation_integrity_before,
        "implementation_integrity_after": implementation_integrity_after,
    }
    _atomic_json(receipt_path, receipt)
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
