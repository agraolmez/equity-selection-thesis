"""Validate the complete, reader-facing Phase 1 handoff before freezing it."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
import xml.etree.ElementTree as et
from pathlib import Path
from typing import Any

from portfolio_bot.data.phase_01_review import (
    FAMILY_SQL,
    FILE_SQL,
    HEADLINE_SQL,
    PROJECT_ROOT,
    PROVENANCE_SQL,
    SCOPE_SQL,
    build_report_artifact,
    build_source_notes,
    load_review_context,
    provenance_record,
    report_provenance,
    validation_failures,
    with_provenance_columns,
)
from portfolio_bot.data.phase_01_notebook import (
    PHASE_01_NOTEBOOK_RELATIVE_PATH,
    build_phase_01_notebook,
    canonical_phase_01_notebook_bytes,
)
from portfolio_bot.data.phase_01_evidence import (
    PHASE_01_TEST_IDS,
    implementation_receipt,
    sha256_file,
    stable_file_bytes_receipt,
    stable_file_receipt,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_receipt(project_root: Path, path: Path) -> dict[str, Any]:
    before = path.stat()
    receipt = {
        "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
        "size_bytes": before.st_size,
        "sha256": sha256_file(path),
    }
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError(f"Handoff evidence changed while hashing: {path}")
    return receipt


def _integrity_receipt(project_root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        display_path = resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        display_path = str(resolved)
    return stable_file_receipt(
        resolved,
        display_path=display_path,
        include_identity=True,
    )


def _minimal_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: receipt[key] for key in ("path", "size_bytes", "sha256")
    }


def _identity_receipt_valid(receipt: Any) -> bool:
    identity_keys = {
        "device",
        "inode",
        "mode",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
    }
    return (
        isinstance(receipt, dict)
        and set(receipt) == {"path", "size_bytes", "sha256", "identity"}
        and isinstance(receipt.get("path"), str)
        and type(receipt.get("size_bytes")) is int
        and receipt["size_bytes"] >= 0
        and isinstance(receipt.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"]) is not None
        and isinstance(receipt.get("identity"), dict)
        and set(receipt["identity"]) == identity_keys
        and all(type(receipt["identity"][key]) is int for key in identity_keys)
        and receipt["identity"]["size_bytes"] == receipt["size_bytes"]
    )


def _recorded_content_matches_current(
    recorded: Any, current: dict[str, Any]
) -> bool:
    return (
        _identity_receipt_valid(recorded)
        and _minimal_receipt(recorded) == _minimal_receipt(current)
    )


def _normalized_cell_source(value: Any) -> Any:
    return "".join(value) if isinstance(value, list) else value


def _stable_json_with_receipt(
    project_root: Path, path: Path
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    display_path = resolved.relative_to(project_root.resolve()).as_posix()
    payload, receipt = stable_file_bytes_receipt(
        resolved,
        display_path=display_path,
        include_identity=True,
    )
    return {
        "receipt": receipt,
        "payload": json.loads(payload.decode("utf-8")),
    }


def _expected_test_source_selection(
    project_root: Path, context: dict[str, Any]
) -> dict[str, Any]:
    artifact_root = context["artifact_root"]
    summary = context["summary"]
    latest_verified = _stable_json_with_receipt(
        project_root, artifact_root / "LATEST_VERIFIED.json"
    )
    latest_primary = _stable_json_with_receipt(
        project_root, artifact_root / "LATEST_PRIMARY.json"
    )
    primary_relative = context["primary_directory"].relative_to(artifact_root).as_posix()
    verification_relative = context["verification_directory"].relative_to(
        artifact_root
    ).as_posix()
    expected_verified_payload = {
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "verification_id": summary["verification_id"],
        "primary_path": primary_relative,
        "verification_path": verification_relative,
        "verification_status": "passed",
    }
    expected_primary_payload = {
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "path": primary_relative,
        "primary_status": "primary_passed_pending_independent_verification",
        "verification_status": "not_run",
    }
    if (
        latest_verified["payload"] != expected_verified_payload
        or latest_primary["payload"] != expected_primary_payload
    ):
        raise RuntimeError("Current Phase 1 source-selection pointers are not canonical")
    return {
        "artifact_root": artifact_root.relative_to(project_root).as_posix(),
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "verification_id": summary["verification_id"],
        "selected_primary_directory": context["primary_directory"]
        .relative_to(project_root)
        .as_posix(),
        "selected_verification_directory": context["verification_directory"]
        .relative_to(project_root)
        .as_posix(),
        "latest_verified": latest_verified,
        "latest_primary": latest_primary,
    }


def _version_receipt(executable: Path) -> dict[str, Any]:
    command = [str(executable.resolve(strict=True)), "--version"]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
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


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in rows
    ]


def _check(
    check_id: str,
    passed: bool,
    observed: Any,
    expected: Any,
    message: str,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "expected": expected,
        "message": message,
    }


def _utc_timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        return None
    return parsed


def validate_handoff(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = project_root.resolve()
    context = load_review_context(project_root)
    summary = context["summary"]
    build_id = summary["phase_01_build_id"]
    run_directory = project_root / "runs" / "phase_01_raw_snapshot" / build_id
    report_directory = project_root / "reports" / "phase_01_raw_snapshot"
    notebook_path = project_root / PHASE_01_NOTEBOOK_RELATIVE_PATH
    notebook_execution_path = run_directory / "notebook_execution_receipt.json"
    pytest_path = run_directory / "pytest.xml"
    test_execution_path = run_directory / "test_execution_receipt.json"
    receipt_path = run_directory / "report_delivery_receipt.json"
    html_path = report_directory / "phase_01_verification_report.html"
    artifact_path = report_directory / "artifact.json"
    database_path = report_directory / "phase_01_review.sqlite"

    checks: list[dict[str, Any]] = []
    evidence_failures = validation_failures(context)
    checks.append(
        _check(
            "PUBLISHED_EVIDENCE_RECONCILES",
            not evidence_failures,
            evidence_failures,
            [],
            "The primary, independent, registry, date, and hash invariants must all pass.",
        )
    )

    artifact = _read_json(artifact_path)
    manifest = artifact["manifest"]
    snapshot = artifact["snapshot"]
    expected_counts = {"blocks": 15, "cards": 5, "charts": 1, "tables": 3}
    observed_counts = {
        "blocks": len(manifest.get("blocks", [])),
        "cards": len(manifest.get("cards", [])),
        "charts": len(manifest.get("charts", [])),
        "tables": len(manifest.get("tables", [])),
    }
    checks.append(
        _check(
            "REPORT_READING_PATH_COMPLETE",
            observed_counts == expected_counts
            and manifest.get("title") == "Phase 1 Raw Data Snapshot: Verification Report"
            and snapshot.get("status") == "ready",
            observed_counts,
            expected_counts,
            "The technical report must contain its title, narrative, headline metrics, visual evidence, exact tables, limitations, and next steps.",
        )
    )

    expected_artifact = build_report_artifact(
        context,
        generated_at=artifact.get("manifest", {}).get("generatedAt"),
    )
    expected_datasets = expected_artifact["snapshot"]["datasets"]
    expected_datasets = {
        "headline": expected_datasets["headline"],
        "family_summary": sorted(
            expected_datasets["family_summary"],
            key=lambda row: (-row["parsed_rows"], row["family"]),
        ),
        "file_summary": sorted(
            expected_datasets["file_summary"], key=lambda row: row["filename"]
        ),
        "scope_rows": sorted(
            expected_datasets["scope_rows"], key=lambda row: row["claim"]
        ),
    }
    expected_artifact["snapshot"]["datasets"] = expected_datasets
    summary_artifact = _read_json(report_directory / "phase_01_summary.json")
    family_csv_matches = _csv_rows(
        report_directory / "source_family_summary.csv"
    ) == _as_csv_rows(with_provenance_columns(context, context["family_summary"]))
    file_csv_matches = _csv_rows(
        report_directory / "source_file_summary.csv"
    ) == _as_csv_rows(with_provenance_columns(context, context["file_summary"]))
    source_notes_match = (
        report_directory / "source_notes.md"
    ).read_text(encoding="utf-8") == build_source_notes(context)
    report_matches_context = (
        artifact == expected_artifact
        and summary_artifact.get("summary") == summary
        and summary_artifact.get("provenance") == report_provenance(context)
        and summary_artifact.get("evidence_reconciliation")
        == context["reconciliation"]
        and summary_artifact.get("verified_pointer") == context["verified_pointer"]
        and summary_artifact.get("scope_boundary") == context["manifest"]["scope_boundary"]
        and summary_artifact.get("validation_failures") == []
        and family_csv_matches
        and file_csv_matches
        and source_notes_match
    )
    checks.append(
        _check(
            "REPORT_MATCHES_CURRENT_VERIFIED_CONTEXT",
            report_matches_context,
            {
                "full_artifact_matches": artifact == expected_artifact,
                "summary_matches": summary_artifact.get("summary") == summary,
                "summary_provenance_matches": summary_artifact.get("provenance")
                == report_provenance(context),
                "summary_reconciliation_matches": summary_artifact.get(
                    "evidence_reconciliation"
                )
                == context["reconciliation"],
                "verified_pointer_matches": summary_artifact.get("verified_pointer")
                == context["verified_pointer"],
                "family_csv_matches": family_csv_matches,
                "file_csv_matches": file_csv_matches,
                "source_notes_match": source_notes_match,
            },
            {
                "full_artifact_matches": True,
                "summary_matches": True,
                "summary_provenance_matches": True,
                "summary_reconciliation_matches": True,
                "verified_pointer_matches": True,
                "family_csv_matches": True,
                "file_csv_matches": True,
                "source_notes_match": True,
            },
            "The complete report, ancillary tables, and notes must equal the current verified context and generator.",
        )
    )

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        queried = {
            "headline": [dict(row) for row in connection.execute(HEADLINE_SQL)],
            "family_summary": [dict(row) for row in connection.execute(FAMILY_SQL)],
            "file_summary": [dict(row) for row in connection.execute(FILE_SQL)],
            "scope_rows": [dict(row) for row in connection.execute(SCOPE_SQL)],
        }
        queried_provenance = [
            dict(row) for row in connection.execute(PROVENANCE_SQL)
        ]
    finally:
        connection.close()
    expected_provenance = [provenance_record(context)]
    provenance_matches = (
        queried_provenance == expected_provenance
        and artifact.get("provenance", {}).get("raw_snapshot_id")
        == summary["raw_snapshot_id"]
        and artifact.get("provenance", {}).get("phase_01_build_id") == build_id
        and artifact.get("provenance", {}).get("verification_id")
        == summary["verification_id"]
        and all(
            artifact.get("provenance", {}).get(key) == value
            for key, value in expected_provenance[0].items()
        )
    )
    checks.append(
        _check(
            "REPORT_SQL_REPRODUCES_SNAPSHOT",
            queried == snapshot.get("datasets") == expected_datasets
            and provenance_matches,
            {
                **{key: len(value) for key, value in queried.items()},
                "provenance": len(queried_provenance),
                "provenance_matches": provenance_matches,
            },
            {
                "headline": 1,
                "family_summary": 8,
                "file_summary": 25,
                "scope_rows": 6,
                "provenance": 1,
                "provenance_matches": True,
            },
            "Every report dataset and its one-row evidence identity must equal the saved runnable SQLite queries.",
        )
    )

    loaded_receipt = _read_json(receipt_path)
    receipt = loaded_receipt if isinstance(loaded_receipt, dict) else {}
    artifact_integrity = _integrity_receipt(project_root, artifact_path)
    database_integrity = _integrity_receipt(project_root, database_path)
    html_integrity = _integrity_receipt(project_root, html_path)
    qa_result = receipt.get("qa_result", {})
    qa_result = qa_result if isinstance(qa_result, dict) else {}
    browser_receipt_keys = {
        "schema_version",
        "receipt_type",
        "status",
        "run_started_utc",
        "run_completed_utc",
        "raw_snapshot_id",
        "phase_01_build_id",
        "verification_id",
        "cwd",
        "command",
        "environment",
        "return_code",
        "stdout",
        "stderr",
        "stdout_sha256",
        "stderr_sha256",
        "artifact_before_qa",
        "artifact_after_qa",
        "source_database_before_qa",
        "source_database_after_qa",
        "html_after_qa",
        "qa_engine",
        "qa_result",
    }
    browser_started = _utc_timestamp(receipt.get("run_started_utc"))
    browser_completed = _utc_timestamp(receipt.get("run_completed_utc"))

    qa_engine = receipt.get("qa_engine", {})
    qa_engine = qa_engine if isinstance(qa_engine, dict) else {}
    engine_before = qa_engine.get("before", {})
    engine_after = qa_engine.get("after", {})
    engine_keys = {
        "delivery_script",
        "node_executable",
        "node_version",
        "chromium_executable",
        "chromium_version",
    }
    engine_schema_valid = (
        set(qa_engine) == {"before", "after"}
        and isinstance(engine_before, dict)
        and isinstance(engine_after, dict)
        and set(engine_before) == engine_keys
        and set(engine_after) == engine_keys
    )
    current_engine: dict[str, Any] = {}
    if engine_schema_valid:
        try:
            node_path = Path(engine_before["node_executable"]["path"])
            chromium_path = Path(engine_before["chromium_executable"]["path"])
            current_engine = {
                "delivery_script": stable_file_receipt(
                    Path(engine_before["delivery_script"]["path"]),
                    display_path=engine_before["delivery_script"]["path"],
                    include_identity=True,
                ),
                "node_executable": stable_file_receipt(
                    node_path,
                    display_path=engine_before["node_executable"]["path"],
                    include_identity=True,
                ),
                "node_version": _version_receipt(node_path),
                "chromium_executable": stable_file_receipt(
                    chromium_path,
                    display_path=engine_before["chromium_executable"]["path"],
                    include_identity=True,
                ),
                "chromium_version": _version_receipt(chromium_path),
            }
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            current_engine = {}
    version_receipt_keys = {
        "command",
        "return_code",
        "stdout",
        "stderr",
        "stdout_sha256",
        "stderr_sha256",
    }
    version_receipts_valid = all(
        isinstance(engine_before.get(name), dict)
        and set(engine_before[name]) == version_receipt_keys
        and type(engine_before[name].get("return_code")) is int
        and engine_before[name].get("return_code") == 0
        for name in ("node_version", "chromium_version")
    )
    engine_matches = (
        engine_schema_valid
        and version_receipts_valid
        and engine_before == engine_after
        and engine_before == current_engine
    )

    expected_command: list[str] | None = None
    expected_environment: dict[str, str] | None = None
    if engine_matches:
        expected_command = [
            engine_before["node_executable"]["path"],
            engine_before["delivery_script"]["path"],
            "--input",
            str(artifact_path),
            "--output",
            str(html_path),
            "--timeout-ms",
            "20000",
            "--screenshot",
            "/private/tmp/phase_01_report_failure.png",
        ]
        expected_environment = {
            "CHROMIUM_EXECUTABLE_PATH": engine_before["chromium_executable"][
                "path"
            ]
        }

    stdout = receipt.get("stdout")
    stderr = receipt.get("stderr")
    parsed_stdout_result: Any = None
    if isinstance(stdout, str):
        output_lines = [line for line in stdout.splitlines() if line.strip()]
        if output_lines:
            try:
                parsed_stdout_result = json.loads(output_lines[-1])
            except json.JSONDecodeError:
                parsed_stdout_result = None
    stream_hashes_match = (
        isinstance(stdout, str)
        and isinstance(stderr, str)
        and receipt.get("stdout_sha256")
        == hashlib.sha256(stdout.encode("utf-8")).hexdigest()
        and receipt.get("stderr_sha256")
        == hashlib.sha256(stderr.encode("utf-8")).hexdigest()
        and parsed_stdout_result == qa_result
    )

    qa_result_keys = {
        "counts",
        "html",
        "ok",
        "sourceDialog",
        "sourceInteraction",
        "stages",
        "timings",
        "viewports",
    }
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
    timings = qa_result.get("timings")
    qa_result_valid = (
        set(qa_result) == qa_result_keys
        and qa_result.get("ok") is True
        and qa_result.get("stages")
        == {"validation": "passed", "package": "passed", "verification": "passed"}
        and qa_result.get("counts")
        == {"blocks": 19, "charts": 1, "html": 0, "metrics": 5, "tables": 3}
        and qa_result.get("sourceDialog") == "passed"
        and qa_result.get("sourceInteraction") == "keyboard_menu_semantic_click"
        and sorted(qa_result.get("viewports", [])) == [390, 1440]
        and Path(qa_result.get("html", "")).resolve() == html_path.resolve()
        and isinstance(timings, dict)
        and set(timings) == timing_keys
        and all(
            type(value) in {int, float} and value >= 0
            for value in timings.values()
        )
    )
    artifact_causality_valid = (
        receipt.get("artifact_before_qa") == receipt.get("artifact_after_qa")
        and _recorded_content_matches_current(
            receipt.get("artifact_after_qa"), artifact_integrity
        )
    )
    database_causality_valid = (
        receipt.get("source_database_before_qa")
        == receipt.get("source_database_after_qa")
        and _recorded_content_matches_current(
            receipt.get("source_database_after_qa"), database_integrity
        )
    )
    html_causality_valid = _recorded_content_matches_current(
        receipt.get("html_after_qa"), html_integrity
    )
    receipt_passed = (
        set(receipt) == browser_receipt_keys
        and type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 3
        and receipt.get("receipt_type") == "phase_01_causal_browser_qa"
        and receipt.get("status") == "passed"
        and browser_started is not None
        and browser_completed is not None
        and browser_started <= browser_completed
        and receipt.get("raw_snapshot_id") == summary["raw_snapshot_id"]
        and receipt.get("phase_01_build_id") == build_id
        and receipt.get("verification_id") == summary["verification_id"]
        and receipt.get("cwd") == str(project_root)
        and type(receipt.get("return_code")) is int
        and receipt.get("return_code") == 0
        and receipt.get("command") == expected_command
        and receipt.get("environment") == expected_environment
        and artifact_causality_valid
        and database_causality_valid
        and html_causality_valid
        and engine_matches
        and stream_hashes_match
        and qa_result_valid
    )
    checks.append(
        _check(
            "REPORT_BROWSER_QA_PASSED",
            receipt_passed,
            {
                "stages": qa_result.get("stages"),
                "sourceDialog": qa_result.get("sourceDialog"),
                "viewports": qa_result.get("viewports"),
                "artifact_before_qa": receipt.get("artifact_before_qa"),
                "artifact_after_qa": receipt.get("artifact_after_qa"),
                "source_database_before_qa": receipt.get(
                    "source_database_before_qa"
                ),
                "source_database_after_qa": receipt.get(
                    "source_database_after_qa"
                ),
                "html_after_qa": receipt.get("html_after_qa"),
                "engine_matches": engine_matches,
                "stream_hashes_match": stream_hashes_match,
                "artifact_causality_valid": artifact_causality_valid,
                "database_causality_valid": database_causality_valid,
                "html_causality_valid": html_causality_valid,
            },
            {
                "stages": {"validation": "passed", "package": "passed", "verification": "passed"},
                "sourceDialog": "passed",
                "viewports": [1440, 390],
                "artifact_before_qa": receipt.get("artifact_before_qa"),
                "artifact_after_qa": receipt.get("artifact_after_qa"),
                "source_database_before_qa": receipt.get(
                    "source_database_before_qa"
                ),
                "source_database_after_qa": receipt.get(
                    "source_database_after_qa"
                ),
                "html_after_qa": receipt.get("html_after_qa"),
                "engine_matches": True,
                "stream_hashes_match": True,
                "artifact_causality_valid": True,
                "database_causality_valid": True,
                "html_causality_valid": True,
            },
            "One browser-QA run must bind unchanged input artifact bytes and the exact delivered HTML bytes.",
        )
    )

    html = html_path.read_text(encoding="utf-8")
    html_markers = {
        "title": "<title>Phase 1 Raw Data Snapshot: Verification Report</title>" in html,
        "semantic_fallback": 'id="data-analytics-portable-fallback"' in html,
        "enhanced_reader": 'id="data-analytics-portable-reader"' in html,
        "embedded_chart": "portable-static-chart" in html,
    }
    checks.append(
        _check(
            "REPORT_SELF_CONTAINED_MARKERS_PRESENT",
            all(html_markers.values()) and html_path.stat().st_size > 100_000,
            {**html_markers, "size_bytes": html_path.stat().st_size},
            "all markers true and size_bytes > 100000",
            "The delivered HTML must include both enhanced and semantic readers plus its embedded chart.",
        )
    )

    notebook = _read_json(notebook_path)
    expected_notebook = build_phase_01_notebook(project_root)
    expected_cell_contract = [
        {
            "id": cell.get("id"),
            "cell_type": cell.get("cell_type"),
            "source": _normalized_cell_source(cell.get("source")),
            "metadata": dict(cell.get("metadata", {})),
        }
        for cell in expected_notebook.get("cells", [])
    ]
    observed_cell_contract = [
        {
            "id": cell.get("id"),
            "cell_type": cell.get("cell_type"),
            "source": _normalized_cell_source(cell.get("source")),
            "metadata": {
                key: value
                for key, value in dict(cell.get("metadata", {})).items()
                if key != "execution"
            },
        }
        for cell in notebook.get("cells", [])
    ]
    expected_notebook_metadata = dict(expected_notebook.get("metadata", {}))
    observed_notebook_metadata = notebook.get("metadata", {})
    notebook_metadata_matches = (
        observed_notebook_metadata.get("kernelspec")
        == expected_notebook_metadata.get("kernelspec")
        and observed_notebook_metadata.get("language_info", {}).get("name")
        == "python"
        and str(observed_notebook_metadata.get("language_info", {}).get("version", "")).startswith(
            "3"
        )
        and all(
            observed_notebook_metadata.get(key) == expected_notebook_metadata.get(key)
            for key in (
                "phase",
                "role",
                "raw_snapshot_id",
                "phase_01_build_id",
                "verification_id",
                "phase_status",
            )
        )
    )
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    notebook_errors = [
        {"cell": index, "ename": output.get("ename"), "evalue": output.get("evalue")}
        for index, cell in enumerate(code_cells, start=1)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    execution_counts = [cell.get("execution_count") for cell in code_cells]
    final_output_text = json.dumps(code_cells[-1].get("outputs", []), sort_keys=True)
    notebook_passed = (
        len(code_cells) == 7
        and not notebook_errors
        and execution_counts == list(range(1, len(code_cells) + 1))
        and observed_cell_contract == expected_cell_contract
        and notebook_metadata_matches
        and summary["raw_snapshot_id"] in final_output_text
        and summary["phase_01_build_id"] in final_output_text
        and summary["verification_id"] in final_output_text
        and "verified_pending_explicit_phase_freeze" in final_output_text
    )
    checks.append(
        _check(
            "NOTEBOOK_EXECUTED_TOP_TO_BOTTOM",
            notebook_passed,
            {
                "code_cells": len(code_cells),
                "execution_counts": execution_counts,
                "errors": notebook_errors,
                "source_contract_matches": observed_cell_contract == expected_cell_contract,
                "metadata_matches": notebook_metadata_matches,
                "result_ids_present": all(
                    value in final_output_text
                    for value in (
                        summary["raw_snapshot_id"],
                        summary["phase_01_build_id"],
                        summary["verification_id"],
                    )
                ),
            },
            {
                "code_cells": 7,
                "execution_counts": [1, 2, 3, 4, 5, 6, 7],
                "errors": [],
                "source_contract_matches": True,
                "metadata_matches": True,
                "result_ids_present": True,
            },
            "The exact context-bound notebook source must execute top-to-bottom and emit its machine-readable gate result.",
        )
    )

    canonical_notebook_bytes = canonical_phase_01_notebook_bytes(project_root)
    expected_canonical_receipt = {
        "path": PHASE_01_NOTEBOOK_RELATIVE_PATH.as_posix(),
        "size_bytes": len(canonical_notebook_bytes),
        "sha256": hashlib.sha256(canonical_notebook_bytes).hexdigest(),
    }
    executed_notebook_receipt = _file_receipt(project_root, notebook_path)
    loaded_notebook_execution_receipt = _read_json(notebook_execution_path)
    notebook_execution_receipt = (
        loaded_notebook_execution_receipt
        if isinstance(loaded_notebook_execution_receipt, dict)
        else {}
    )
    notebook_execution_keys = {
        "schema_version",
        "phase",
        "status",
        "run_started_utc",
        "run_completed_utc",
        "raw_snapshot_id",
        "phase_01_build_id",
        "verification_id",
        "cwd",
        "canonical_notebook_before",
        "executed_notebook_after",
        "execution",
        "engine",
        "implementation_before",
        "implementation_after",
    }
    expected_final_result = {
        "phase": "01_raw_snapshot",
        "status": "verified_pending_explicit_phase_freeze",
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": build_id,
        "verification_id": summary["verification_id"],
        "phase_2_permitted": False,
    }
    expected_notebook_execution = {
        "code_cell_count": 7,
        "execution_counts": [1, 2, 3, 4, 5, 6, 7],
        "error_count": 0,
        "source_contract_matches": True,
        "metadata_context_matches": True,
        "final_result": expected_final_result,
    }
    expected_notebook_engine = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "nbformat_version": importlib.metadata.version("nbformat"),
        "nbclient_version": importlib.metadata.version("nbclient"),
        "kernel_name": expected_notebook_metadata["kernelspec"]["name"],
        "timeout_seconds": 600,
        "allow_errors": False,
        "record_timing": False,
    }
    notebook_started = _utc_timestamp(
        notebook_execution_receipt.get("run_started_utc")
    )
    notebook_completed = _utc_timestamp(
        notebook_execution_receipt.get("run_completed_utc")
    )
    current_implementation = implementation_receipt(project_root)
    causal_notebook_receipt_passed = (
        set(notebook_execution_receipt) == notebook_execution_keys
        and type(notebook_execution_receipt.get("schema_version")) is int
        and notebook_execution_receipt.get("schema_version") == 1
        and notebook_execution_receipt.get("phase") == "01_raw_snapshot"
        and notebook_execution_receipt.get("status") == "passed"
        and notebook_started is not None
        and notebook_completed is not None
        and notebook_started <= notebook_completed
        and notebook_execution_receipt.get("raw_snapshot_id")
        == summary["raw_snapshot_id"]
        and notebook_execution_receipt.get("phase_01_build_id") == build_id
        and notebook_execution_receipt.get("verification_id")
        == summary["verification_id"]
        and isinstance(notebook_execution_receipt.get("cwd"), str)
        and Path(notebook_execution_receipt["cwd"]).resolve() == project_root
        and notebook_execution_receipt.get("canonical_notebook_before")
        == expected_canonical_receipt
        and notebook_execution_receipt.get("executed_notebook_after")
        == executed_notebook_receipt
        and notebook_execution_receipt.get("execution")
        == expected_notebook_execution
        and notebook_execution_receipt.get("engine") == expected_notebook_engine
        and notebook_execution_receipt.get("implementation_before")
        == current_implementation
        and notebook_execution_receipt.get("implementation_after")
        == current_implementation
    )
    checks.append(
        _check(
            "NOTEBOOK_EXECUTION_BINDS_IMPLEMENTATION",
            causal_notebook_receipt_passed,
            {
                "schema_keys_match": set(notebook_execution_receipt)
                == notebook_execution_keys,
                "status": notebook_execution_receipt.get("status"),
                "time_order_valid": notebook_started is not None
                and notebook_completed is not None
                and notebook_started <= notebook_completed,
                "canonical_notebook_before": notebook_execution_receipt.get(
                    "canonical_notebook_before"
                ),
                "executed_notebook_after": notebook_execution_receipt.get(
                    "executed_notebook_after"
                ),
                "execution": notebook_execution_receipt.get("execution"),
                "engine": notebook_execution_receipt.get("engine"),
                "implementation_before_matches": notebook_execution_receipt.get(
                    "implementation_before"
                )
                == current_implementation,
                "implementation_after_matches": notebook_execution_receipt.get(
                    "implementation_after"
                )
                == current_implementation,
            },
            {
                "schema_keys_match": True,
                "status": "passed",
                "time_order_valid": True,
                "canonical_notebook_before": expected_canonical_receipt,
                "executed_notebook_after": executed_notebook_receipt,
                "execution": expected_notebook_execution,
                "engine": expected_notebook_engine,
                "implementation_before_matches": True,
                "implementation_after_matches": True,
            },
            "One causal execution must bind the exact canonical source, unchanged implementation, execution engine, and published notebook bytes.",
        )
    )

    junit_bytes, junit_integrity = stable_file_bytes_receipt(
        pytest_path,
        display_path=pytest_path.relative_to(project_root).as_posix(),
        include_identity=True,
    )
    junit = et.fromstring(junit_bytes)
    if junit.tag == "testsuite":
        suites = [junit]
    elif junit.tag == "testsuites":
        suites = list(junit.findall("testsuite"))
    else:
        suites = []
    testcases = [case for suite in suites for case in suite.findall("testcase")]
    observed_test_ids = sorted(
        f"{case.attrib.get('classname')}::{case.attrib.get('name')}"
        for case in testcases
    )
    expected_test_ids = sorted(PHASE_01_TEST_IDS)
    suite_test_summary = {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
    }
    child_result_counts = {
        result: sum(len(case.findall(result)) for case in testcases)
        for result in ("failure", "error", "skipped")
    }
    test_summary = {
        "tests": len(testcases),
        "failures": child_result_counts["failure"],
        "errors": child_result_counts["error"],
        "skipped": child_result_counts["skipped"],
    }
    junit_structure_valid = (
        junit.tag in {"testsuite", "testsuites"}
        and len(suites) == 1
        and suite_test_summary == test_summary
        and len(observed_test_ids) == len(set(observed_test_ids))
        and all(
            case.attrib.get("classname") and case.attrib.get("name")
            for case in testcases
        )
    )
    checks.append(
        _check(
            "AUTOMATED_TEST_SUITE_PASSED",
            junit_structure_valid
            and test_summary["tests"] == len(expected_test_ids)
            and test_summary["failures"] == 0
            and test_summary["errors"] == 0
            and test_summary["skipped"] == 0
            and observed_test_ids == expected_test_ids,
            {
                **test_summary,
                "test_ids": observed_test_ids,
                "junit_structure_valid": junit_structure_valid,
            },
            {
                "tests": len(expected_test_ids),
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "test_ids": expected_test_ids,
                "junit_structure_valid": True,
            },
            "The exact Phase 1 test registry must pass without failures, errors, or skips.",
        )
    )

    tested_implementation = implementation_receipt(project_root)
    tested_implementation_integrity = implementation_receipt(
        project_root, include_identity=True
    )
    pytest_receipt = {
        **_minimal_receipt(junit_integrity),
        "test_ids": observed_test_ids,
    }
    loaded_test_execution_receipt = _read_json(test_execution_path)
    test_execution_receipt = (
        loaded_test_execution_receipt
        if isinstance(loaded_test_execution_receipt, dict)
        else {}
    )
    test_receipt_keys = {
        "schema_version",
        "receipt_type",
        "status",
        "run_started_utc",
        "run_completed_utc",
        "raw_snapshot_id",
        "phase_01_build_id",
        "verification_id",
        "cwd",
        "runtime",
        "command",
        "return_code",
        "source_selection_before",
        "source_selection_after",
        "test_summary",
        "test_ids",
        "junit",
        "implementation_before",
        "implementation_after",
        "implementation_integrity_before",
        "implementation_integrity_after",
    }
    expected_source_selection = _expected_test_source_selection(
        project_root, context
    )
    python_executable = Path(sys.executable).resolve(strict=True)
    executable_integrity = _integrity_receipt(project_root, python_executable)
    expected_runtime = {
        "python_invocation": sys.executable,
        "python_executable": str(python_executable),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pytest_version": importlib.metadata.version("pytest"),
        "executable_before": executable_integrity,
        "executable_after": executable_integrity,
    }
    expected_command = [
        str(python_executable),
        "-m",
        "pytest",
        "-q",
        "tests",
        f"--junitxml={pytest_path}",
    ]
    expected_junit_structure = {
        "root_element": junit.tag,
        "suite_count": len(suites),
        "testcase_count": len(testcases),
        "child_result_counts": child_result_counts,
    }
    recorded_junit = test_execution_receipt.get("junit")
    recorded_junit_identity = (
        {
            key: recorded_junit.get(key)
            for key in ("path", "size_bytes", "sha256", "identity")
        }
        if isinstance(recorded_junit, dict)
        else {}
    )
    junit_receipt_valid = (
        isinstance(recorded_junit, dict)
        and set(recorded_junit)
        == {
            "path",
            "size_bytes",
            "sha256",
            "identity",
            *expected_junit_structure.keys(),
        }
        and _recorded_content_matches_current(
            recorded_junit_identity, junit_integrity
        )
        and all(
            recorded_junit.get(key) == value
            for key, value in expected_junit_structure.items()
        )
    )
    test_started = _utc_timestamp(test_execution_receipt.get("run_started_utc"))
    test_completed = _utc_timestamp(
        test_execution_receipt.get("run_completed_utc")
    )
    causal_test_receipt_passed = (
        set(test_execution_receipt) == test_receipt_keys
        and type(test_execution_receipt.get("schema_version")) is int
        and test_execution_receipt.get("schema_version") == 2
        and test_execution_receipt.get("receipt_type")
        == "phase_01_exact_test_execution"
        and test_execution_receipt.get("status") == "passed"
        and test_started is not None
        and test_completed is not None
        and test_started <= test_completed
        and type(test_execution_receipt.get("return_code")) is int
        and test_execution_receipt.get("return_code") == 0
        and test_execution_receipt.get("raw_snapshot_id") == summary["raw_snapshot_id"]
        and test_execution_receipt.get("phase_01_build_id") == build_id
        and test_execution_receipt.get("verification_id") == summary["verification_id"]
        and test_execution_receipt.get("cwd") == str(project_root)
        and test_execution_receipt.get("runtime") == expected_runtime
        and test_execution_receipt.get("command") == expected_command
        and test_execution_receipt.get("source_selection_before")
        == expected_source_selection
        and test_execution_receipt.get("source_selection_after")
        == expected_source_selection
        and test_execution_receipt.get("test_summary") == test_summary
        and test_execution_receipt.get("test_ids") == observed_test_ids
        and junit_receipt_valid
        and test_execution_receipt.get("implementation_before") == tested_implementation
        and test_execution_receipt.get("implementation_after") == tested_implementation
        and test_execution_receipt.get("implementation_integrity_before")
        == tested_implementation_integrity
        and test_execution_receipt.get("implementation_integrity_after")
        == tested_implementation_integrity
    )
    checks.append(
        _check(
            "TEST_EXECUTION_BINDS_IMPLEMENTATION",
            causal_test_receipt_passed,
            {
                "status": test_execution_receipt.get("status"),
                "test_summary": test_execution_receipt.get("test_summary"),
                "test_ids": test_execution_receipt.get("test_ids"),
                "junit": test_execution_receipt.get("junit"),
                "junit_receipt_valid": junit_receipt_valid,
                "time_order_valid": test_started is not None
                and test_completed is not None
                and test_started <= test_completed,
                "runtime_matches": test_execution_receipt.get("runtime")
                == expected_runtime,
                "command_matches": test_execution_receipt.get("command")
                == expected_command,
                "source_selection_matches": test_execution_receipt.get(
                    "source_selection_before"
                )
                == expected_source_selection
                and test_execution_receipt.get("source_selection_after")
                == expected_source_selection,
                "implementation_before_matches": test_execution_receipt.get(
                    "implementation_before"
                )
                == tested_implementation,
                "implementation_after_matches": test_execution_receipt.get(
                    "implementation_after"
                )
                == tested_implementation,
            },
            {
                "status": "passed",
                "test_summary": test_summary,
                "test_ids": observed_test_ids,
                "junit": {
                    **_minimal_receipt(junit_integrity),
                    **expected_junit_structure,
                },
                "junit_receipt_valid": True,
                "time_order_valid": True,
                "runtime_matches": True,
                "command_matches": True,
                "source_selection_matches": True,
                "implementation_before_matches": True,
                "implementation_after_matches": True,
            },
            "The test runner must bind one unchanged implementation inventory to the exact passing JUnit bytes.",
        )
    )

    failed = [item for item in checks if item["status"] == "FAIL"]
    return {
        "schema_version": 1,
        "phase": "01_raw_snapshot",
        "assessment": "Ready to freeze" if not failed else "Needs revision",
        "validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": build_id,
        "verification_id": summary["verification_id"],
        "checks": checks,
        "failed_check_count": len(failed),
        "issues_found": [],
        "required_caveats": [
            "Coverage dates are not public-availability dates.",
            "Phase 1 does not establish canonical joins, feature eligibility, leakage safety, model skill, or portfolio performance.",
            "Source extraction provenance was not supplied separately; filesystem dates and maximum observations are not treated as source as-of dates.",
        ],
        "next_permitted_action": (
            "Publish the immutable Phase 1 freeze, then begin Phase 2 canonical-panel contracts."
            if not failed
            else "Resolve failed handoff checks before freezing Phase 1."
        ),
        "evidence": {
            "primary": str(context["primary_directory"].relative_to(project_root)),
            "verification": str(context["verification_directory"].relative_to(project_root)),
            "report": str(html_path.relative_to(project_root)),
            "report_artifact": str(artifact_path.relative_to(project_root)),
            "report_database": str(database_path.relative_to(project_root)),
            "notebook": str(notebook_path.relative_to(project_root)),
            "notebook_execution_receipt": str(
                notebook_execution_path.relative_to(project_root)
            ),
            "pytest": str(pytest_path.relative_to(project_root)),
            "report_delivery_receipt": str(receipt_path.relative_to(project_root)),
            "test_execution_receipt": str(
                test_execution_path.relative_to(project_root)
            ),
        },
        "pytest_receipt": pytest_receipt,
        "notebook_execution_receipt": notebook_execution_receipt,
        "test_execution_receipt": test_execution_receipt,
        "tested_implementation": tested_implementation,
    }


def write_validation_report(project_root: Path = PROJECT_ROOT) -> Path:
    report = validate_handoff(project_root)
    if report["failed_check_count"]:
        raise RuntimeError(
            "Phase 1 handoff validation failed: "
            + "; ".join(
                check["check_id"] for check in report["checks"] if check["status"] == "FAIL"
            )
        )
    output = (
        project_root.resolve()
        / "runs"
        / "phase_01_raw_snapshot"
        / report["phase_01_build_id"]
        / "validation_report.json"
    )
    _write_json_atomic(output, report)
    return output
