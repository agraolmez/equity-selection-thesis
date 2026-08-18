#!/usr/bin/env python3
"""Render/browser-QA Phase 1 and bind the exact input/output bytes in one run."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

from portfolio_bot.data.phase_01_evidence import stable_file_receipt
from portfolio_bot.data.phase_01_review import PROJECT_ROOT, load_review_context


def _receipt(root: Path, path: Path) -> dict[str, Any]:
    return stable_file_receipt(
        path,
        display_path=path.relative_to(root).as_posix(),
        include_identity=True,
    )


def _engine_receipt(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return stable_file_receipt(
        resolved,
        display_path=str(resolved),
        include_identity=True,
    )


def _version_receipt(executable: Path) -> dict[str, Any]:
    command = [str(executable), "--version"]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if type(completed.returncode) is not int or completed.returncode != 0:
        raise RuntimeError(
            f"Could not identify QA engine version: {command}, "
            f"return={completed.returncode}, stderr={completed.stderr[-1000:]}"
        )
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_sha256": hashlib.sha256(
            completed.stdout.encode("utf-8")
        ).hexdigest(),
        "stderr_sha256": hashlib.sha256(
            completed.stderr.encode("utf-8")
        ).hexdigest(),
    }


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _discover_delivery_script() -> Path:
    matches = sorted(
        (Path.home() / ".codex" / "plugins" / "cache" / "openai-curated-remote" / "data-analytics").glob(
            "*/skills/build-report/scripts/deliver_portable_artifact.mjs"
        )
    )
    if not matches:
        raise FileNotFoundError("Could not locate the Data Analytics report-delivery script")
    return matches[-1]


def _discover_node() -> Path:
    configured = os.environ.get("CODEX_NODE_EXECUTABLE")
    if configured:
        return Path(configured)
    candidates = sorted(
        (Path.home() / ".cache" / "codex-runtimes").glob(
            "*/dependencies/node/bin/node"
        )
    )
    if not candidates:
        raise FileNotFoundError("Could not locate the bundled Node.js executable")
    return candidates[-1]


def _discover_chromium() -> Path:
    configured = os.environ.get("CHROMIUM_EXECUTABLE_PATH")
    if configured:
        return Path(configured)
    fixed_candidates = (
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        Path.home()
        / "Applications"
        / "Google Chrome.app"
        / "Contents"
        / "MacOS"
        / "Google Chrome",
        Path.home()
        / "Applications"
        / "Chromium.app"
        / "Contents"
        / "MacOS"
        / "Chromium",
    )
    for candidate in fixed_candidates:
        if candidate.is_file():
            return candidate
    search_roots = (
        Path.home() / ".playwright-browsers",
        Path.home() / ".cache" / "ms-playwright",
        Path.home() / "Library" / "Caches" / "ms-playwright",
    )
    for search_root in search_roots:
        if search_root.is_dir():
            matches = sorted(search_root.rglob("chrome-headless-shell"))
            if matches:
                return matches[-1]
    raise FileNotFoundError(
        "Could not locate Chromium; set CHROMIUM_EXECUTABLE_PATH explicitly"
    )


def _validate_result(
    result: dict[str, Any],
    *,
    root: Path,
) -> None:
    expected_keys = {
        "counts",
        "html",
        "ok",
        "sourceDialog",
        "sourceInteraction",
        "stages",
        "timings",
        "viewports",
    }
    if set(result) != expected_keys:
        raise RuntimeError(
            f"Browser QA result key inventory changed: {sorted(result)}"
        )
    timing_keys = {
        "browserRunMs",
        "contentChecksMs",
        "readerReadyMs",
        "responsiveChecksMs",
        "sourceDialogMs",
        "staticChartExtractionMs",
        "totalMs",
        "validateAndPackageMs",
        "verificationMs",
    }
    timings = result.get("timings")
    if (
        not isinstance(timings, dict)
        or set(timings) != timing_keys
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
            for value in timings.values()
        )
    ):
        raise RuntimeError("Browser QA timing schema changed or is invalid")
    expected_result = (
        result.get("ok") is True
        and result.get("stages")
        == {"validation": "passed", "package": "passed", "verification": "passed"}
        and result.get("counts")
        == {"blocks": 19, "charts": 1, "html": 0, "metrics": 5, "tables": 3}
        and result.get("sourceDialog") == "passed"
        and result.get("sourceInteraction") == "keyboard_menu_semantic_click"
        and sorted(result.get("viewports", [])) == [390, 1440]
        and Path(result.get("html", "")).resolve()
        == (
            root
            / "reports"
            / "phase_01_raw_snapshot"
            / "phase_01_verification_report.html"
        ).resolve()
    )
    if not expected_result:
        raise RuntimeError(f"Phase 1 browser QA did not meet its exact contract: {result}")


def main() -> int:
    root = PROJECT_ROOT.resolve()
    context = load_review_context(root)
    summary = context["summary"]
    report_directory = root / "reports" / "phase_01_raw_snapshot"
    artifact = report_directory / "artifact.json"
    html = report_directory / "phase_01_verification_report.html"
    source_database = report_directory / "phase_01_review.sqlite"
    run_directory = root / "runs" / "phase_01_raw_snapshot" / summary["phase_01_build_id"]
    receipt_path = run_directory / "report_delivery_receipt.json"
    started = dt.datetime.now(dt.timezone.utc)
    artifact_before = _receipt(root, artifact)
    source_database_before = _receipt(root, source_database)
    delivery_script = _discover_delivery_script().resolve(strict=True)
    node = _discover_node().resolve(strict=True)
    chromium = _discover_chromium().resolve(strict=True)
    engine_before = {
        "delivery_script": _engine_receipt(delivery_script),
        "node_executable": _engine_receipt(node),
        "node_version": _version_receipt(node),
        "chromium_executable": _engine_receipt(chromium),
        "chromium_version": _version_receipt(chromium),
    }
    command = [
        str(node),
        str(delivery_script),
        "--input",
        str(artifact),
        "--output",
        str(html),
        "--timeout-ms",
        "20000",
        "--screenshot",
        "/private/tmp/phase_01_report_failure.png",
    ]
    environment = os.environ.copy()
    environment["CHROMIUM_EXECUTABLE_PATH"] = str(chromium)
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if type(completed.returncode) is not int or completed.returncode != 0:
        raise RuntimeError(
            "Phase 1 browser QA process failed: "
            f"return={completed.returncode}, stderr={completed.stderr[-2000:]}"
        )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("Phase 1 browser QA produced no JSON result")
    result = json.loads(output_lines[-1])
    if not isinstance(result, dict):
        raise RuntimeError("Phase 1 browser QA result is not a JSON object")
    artifact_after = _receipt(root, artifact)
    if artifact_after != artifact_before:
        raise RuntimeError("Report artifact changed while browser QA was running")
    source_database_after = _receipt(root, source_database)
    if source_database_after != source_database_before:
        raise RuntimeError("Report source database changed while browser QA was running")
    html_after = _receipt(root, html)
    engine_after = {
        "delivery_script": _engine_receipt(delivery_script),
        "node_executable": _engine_receipt(node),
        "node_version": _version_receipt(node),
        "chromium_executable": _engine_receipt(chromium),
        "chromium_version": _version_receipt(chromium),
    }
    if engine_after != engine_before:
        raise RuntimeError("Browser QA engine changed while verification was running")
    completed_at = dt.datetime.now(dt.timezone.utc)
    _validate_result(
        result,
        root=root,
    )
    receipt = {
        "schema_version": 3,
        "receipt_type": "phase_01_causal_browser_qa",
        "status": "passed",
        "run_started_utc": started.isoformat(),
        "run_completed_utc": completed_at.isoformat(),
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "verification_id": summary["verification_id"],
        "cwd": str(root),
        "command": command,
        "environment": {"CHROMIUM_EXECUTABLE_PATH": str(chromium)},
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
        "artifact_before_qa": artifact_before,
        "artifact_after_qa": artifact_after,
        "source_database_before_qa": source_database_before,
        "source_database_after_qa": source_database_after,
        "html_after_qa": html_after,
        "qa_engine": {
            "before": engine_before,
            "after": engine_after,
        },
        "qa_result": result,
    }
    _atomic_json(receipt_path, receipt)
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
