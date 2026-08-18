"""Content-addressed, resumable test receipts for the Stage 12 freeze.

Only successful test nodes are published.  A receipt is reusable when its
command, complete frozen project-tree identity, environment identity, and
upstream dependency identities all match exactly.  Failed attempts therefore
never overwrite or invalidate an earlier authenticated success.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from portfolio_bot.data.secure_publication import SecureStagingDirectory
from portfolio_bot.final_freeze.inventory import (
    Stage12InventoryError,
    canonical_json_bytes,
    load_json_object,
    require,
    sha256_bytes,
)


CHECKPOINT_FILES = ("_SUCCESS.json", "receipt.json")
SAFE_RUNNER = "scripts/run_stage_12_safe_regression.py"
SANITIZED_ENVIRONMENT = {
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "TZ": "UTC",
}


class Stage12RegressionError(RuntimeError):
    """Raised when a required Stage 12 regression node does not pass."""


@dataclass(frozen=True, slots=True)
class TestCheckpoint:
    checkpoint_id: str
    path: Path
    receipt: Mapping[str, Any]
    reused_existing: bool


def _write_member(
    stage: SecureStagingDirectory,
    name: str,
    payload: bytes,
) -> str:
    with stage.open_exclusive_binary(name) as stream:
        written = stream.write(payload)
        if written != len(payload):
            raise Stage12RegressionError(f"short checkpoint write: {name}")
    return sha256_bytes(payload)


def _checkpoint_fingerprint(
    *,
    node_name: str,
    command: Sequence[str],
    project_tree_sha256: str,
    environment_sha256: str,
    dependency_ids: Sequence[str],
    environment_overrides: Mapping[str, str],
    process_environment: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    payload = {
        "schema_version": 1,
        "node_name": node_name,
        "command": [str(part) for part in command],
        "project_tree_sha256": project_tree_sha256,
        "environment_sha256": environment_sha256,
        "environment_overrides": dict(sorted(environment_overrides.items())),
        "process_environment": dict(sorted(process_environment.items())),
        "dependency_ids": sorted(str(value) for value in dependency_ids),
    }
    return payload, sha256_bytes(canonical_json_bytes(payload))


def _parse_junit(path: Path) -> tuple[dict[str, Any], bytes, list[str]]:
    try:
        payload = path.read_bytes()
        root = ET.fromstring(payload)
    except (OSError, ET.ParseError) as exc:
        raise Stage12RegressionError("pytest did not produce valid JUnit XML") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        suites = list(root.iter("testsuite"))
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        for name in counts:
            raw = suite.attrib.get(name, "0")
            try:
                counts[name] += int(raw)
            except ValueError as exc:
                raise Stage12RegressionError(
                    f"invalid JUnit {name} count"
                ) from exc
    identities = sorted(
        "::".join(
            value
            for value in (
                case.attrib.get("file", ""),
                case.attrib.get("classname", ""),
                case.attrib.get("name", ""),
            )
            if value
        )
        for case in root.iter("testcase")
    )
    return ({
        **counts,
        "passed": counts["tests"]
        - counts["failures"]
        - counts["errors"]
        - counts["skipped"],
        "test_case_identity_sha256": sha256_bytes(
            canonical_json_bytes(identities)
        ),
    }, payload, identities)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _command_test_selectors(command: Sequence[object]) -> list[str]:
    return [str(value) for value in command if str(value).startswith("tests/")]


def _validate_access_audit_mapping(
    audit: Mapping[str, Any],
    *,
    command: Sequence[object],
    project_root: Path,
) -> None:
    counters = audit.get("counters")
    guard_scope = audit.get("guard_scope")
    selected_tests = audit.get("selected_tests")
    expected_tests = _command_test_selectors(command)
    raw_audit = {
        key: value
        for key, value in audit.items()
        if key not in {"receipt_sha256", "receipt_size_bytes"}
    }
    expected_forbidden_roots = [
        str((project_root / component).resolve(strict=False))
        for component in ("raw", "interim", "processed")
    ] + [str((Path.home() / "Desktop").resolve(strict=False))]
    authorized_metadata = (
        guard_scope.get("authorized_external_metadata")
        if isinstance(guard_scope, Mapping)
        else None
    )
    expected_measurement_scope = (
        "this safe-regression subprocess, from AccessGuard audit-hook "
        "installation through pytest.main return"
    )
    expected_zero_open_claim = (
        "No forbidden-root dispatch succeeded through the enumerated "
        "Python-audit, native-reader, and DuckDB entry points during "
        "this guarded test execution."
    )
    expected_excluded_scope = [
        "activity before guard installation",
        "the parent Stage 12 build process",
        "prior amendment or Stage 13 executions",
        "external processes",
        "unenumerated native entry points",
    ]
    expected_native_entrypoints = [
        "Python audit open/process-spawn events",
        "PyArrow ParquetFile and parquet/csv/dataset path APIs",
        "Pandas and Polars reader path APIs",
        "DuckDB connect, SQL/query, and direct reader path APIs",
    ]
    if (
        audit.get("schema_version") != 1
        or audit.get("status") != "NO_FORBIDDEN_PAYLOAD_ACCESS"
        or not isinstance(counters, Mapping)
        or selected_tests != expected_tests
        or audit.get("selected_tests_sha256")
        != sha256_bytes(canonical_json_bytes(expected_tests))
        or audit.get("pytest_exit_code") != 0
        or not isinstance(guard_scope, Mapping)
        or guard_scope.get("forbidden_roots") != expected_forbidden_roots
        or guard_scope.get("child_processes_permitted") is not False
        or guard_scope.get("native_reader_wrappers_installed") is not True
        or guard_scope.get("explicit_test_allowlist_required") is not True
        or guard_scope.get("measurement_scope") != expected_measurement_scope
        or guard_scope.get("zero_open_claim") != expected_zero_open_claim
        or guard_scope.get("excluded_scope") != expected_excluded_scope
        or guard_scope.get("guarded_native_entrypoints")
        != expected_native_entrypoints
        or guard_scope.get("audited_dependency_site_path")
        != str(
            (
                Path.home()
                / f".local/lib/python{sys.version_info.major}."
                f"{sys.version_info.minor}/site-packages"
            ).resolve(strict=False)
        )
        or not isinstance(authorized_metadata, Mapping)
        or len(authorized_metadata) != 2
        or any(
            not isinstance(path, str)
            or not path.endswith("/manifest.json")
            or "/Desktop/Thesis/interim/phase_05_validation/" not in path
            or not _is_sha256(digest)
            for path, digest in authorized_metadata.items()
        )
        # The guard opens the authenticated Stage 11 execution manifest once.
        # Its two allowlisted clean-candidate descendants remain closed during
        # a metadata-only Stage 12 regression; counting their registration as
        # payload opens would manufacture access that did not occur.
        or int(counters.get("authenticated_external_metadata_open_events", -1))
        != 1
        or int(counters.get("audit_events_seen", 0)) <= 0
        or int(counters.get("filesystem_open_events_seen", 0)) <= 0
        or int(counters.get("native_reader_calls_checked", 0)) <= 0
        or int(counters.get("native_path_values_checked", 0)) <= 0
        or int(counters.get("duckdb_sql_entrypoint_calls_checked", 0)) <= 0
        or any(
            int(counters.get(name, -1)) != 0
            for name in (
                "forbidden_access_attempts_denied",
                "process_spawn_attempts_denied",
                "raw_payload_files_opened",
                "interim_payload_files_opened",
                "processed_payload_files_opened",
                "desktop_payload_files_opened",
                "final_period_payload_files_opened",
                "final_period_rows_read",
            )
        )
        or not _is_sha256(audit.get("receipt_sha256"))
        or audit.get("receipt_sha256")
        != sha256_bytes(canonical_json_bytes(raw_audit))
        or audit.get("receipt_size_bytes")
        != len(canonical_json_bytes(raw_audit))
    ):
        raise Stage12RegressionError("safe regression access audit is not clean")


def _load_access_audit(path: Path) -> tuple[Mapping[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage12RegressionError(
            "safe regression did not produce a valid access audit"
        ) from exc
    if not isinstance(value, Mapping):
        raise Stage12RegressionError("safe regression access audit is not an object")
    counters = value.get("counters")
    guard_scope = value.get("guard_scope")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "NO_FORBIDDEN_PAYLOAD_ACCESS"
        or not isinstance(counters, Mapping)
        or not isinstance(guard_scope, Mapping)
        or guard_scope.get("measurement_scope")
        != (
            "this safe-regression subprocess, from AccessGuard audit-hook "
            "installation through pytest.main return"
        )
        or guard_scope.get("zero_open_claim")
        != (
            "No forbidden-root dispatch succeeded through the enumerated "
            "Python-audit, native-reader, and DuckDB entry points during "
            "this guarded test execution."
        )
        or guard_scope.get("excluded_scope")
        != [
            "activity before guard installation",
            "the parent Stage 12 build process",
            "prior amendment or Stage 13 executions",
            "external processes",
            "unenumerated native entry points",
        ]
        or int(counters.get("native_path_values_checked", 0)) <= 0
        or int(counters.get("duckdb_sql_entrypoint_calls_checked", 0)) <= 0
        or int(counters.get("forbidden_access_attempts_denied", -1)) != 0
        or int(counters.get("process_spawn_attempts_denied", -1)) != 0
        or int(counters.get("raw_payload_files_opened", -1)) != 0
        or int(counters.get("interim_payload_files_opened", -1)) != 0
        or int(counters.get("processed_payload_files_opened", -1)) != 0
        or int(counters.get("desktop_payload_files_opened", -1)) != 0
        or int(counters.get("final_period_payload_files_opened", -1)) != 0
        or int(counters.get("final_period_rows_read", -1)) != 0
    ):
        raise Stage12RegressionError("safe regression access audit is not clean")
    return value, payload


def _validate_existing(
    path: Path,
    *,
    fingerprint: Mapping[str, Any],
    fingerprint_sha256: str,
    project_root: Path,
) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_dir():
        raise Stage12RegressionError("checkpoint is not a real directory")
    try:
        receipt, receipt_file = load_json_object(path / "receipt.json")
        success, success_file = load_json_object(path / "_SUCCESS.json")
    except Stage12InventoryError as exc:
        raise Stage12RegressionError(
            f"checkpoint exists but cannot be authenticated: {path}"
        ) from exc
    expected_names = {item.name for item in path.iterdir()}
    if expected_names != set(CHECKPOINT_FILES):
        raise Stage12RegressionError("checkpoint closed-file set differs")
    expected = dict(fingerprint)
    expected_checkpoint_id = (
        f"stage12-checkpoint-{fingerprint.get('node_name')}-"
        f"{fingerprint_sha256[:24]}"
    )
    if (
        receipt.get("input_fingerprint") != expected
        or receipt.get("input_fingerprint_sha256") != fingerprint_sha256
        or receipt.get("status") != "PASSED"
        or success.get("status") != "PASSED"
        or success.get("checkpoint_id") != receipt.get("checkpoint_id")
        or success.get("receipt_sha256") != receipt_file.sha256
        or success_file.sha256 != sha256_bytes(canonical_json_bytes(success))
        or receipt.get("checkpoint_id") != expected_checkpoint_id
        or success.get("checkpoint_id") != expected_checkpoint_id
        or path.name != expected_checkpoint_id
        or set(success) != {
            "schema_version",
            "checkpoint_id",
            "status",
            "receipt_sha256",
        }
        or success.get("schema_version") != 1
    ):
        raise Stage12RegressionError("checkpoint semantic identity differs")
    summary = receipt.get("summary")
    if not isinstance(summary, Mapping):
        raise Stage12RegressionError("checkpoint summary is missing")
    audit = receipt.get("access_audit")
    junit = receipt.get("junit_receipt")
    command = fingerprint.get("command")
    if not isinstance(command, list):
        raise Stage12RegressionError("checkpoint command is missing")
    if isinstance(audit, Mapping):
        _validate_access_audit_mapping(
            audit,
            command=command,
            project_root=project_root,
        )
    identities = junit.get("test_case_identities") if isinstance(junit, Mapping) else None
    if (
        receipt.get("schema_version") != 2
        or int(summary.get("tests", 0)) <= 0
        or int(summary.get("failures", -1)) != 0
        or int(summary.get("errors", -1)) != 0
        or int(summary.get("skipped", -1)) != 0
        or not isinstance(audit, Mapping)
        or audit.get("status") != "NO_FORBIDDEN_PAYLOAD_ACCESS"
        or not isinstance(junit, Mapping)
        or not _is_sha256(junit.get("sha256"))
        or type(junit.get("size_bytes")) is not int
        or int(junit.get("size_bytes", 0)) <= 0
        or not isinstance(identities, list)
        or len(identities) != int(summary.get("tests", -1))
        or not all(isinstance(value, str) and value for value in identities)
        or identities != sorted(identities)
        or len(identities) != len(set(identities))
        or int(summary.get("passed", -1)) != int(summary.get("tests", -2))
        or summary.get("test_case_identity_sha256")
        != sha256_bytes(canonical_json_bytes(sorted(identities)))
        or junit.get("test_case_identity_sha256")
        != summary.get("test_case_identity_sha256")
        or receipt.get("exit_code") != 0
        or receipt.get("memoization")
        != {
            "reused_prior_stage": False,
            "reusable_only_for_exact_input_fingerprint": True,
            "failed_attempts_published": False,
        }
    ):
        raise Stage12RegressionError("checkpoint is not a successful test run")
    return receipt


def run_or_reuse_test_checkpoint(
    *,
    node_name: str,
    command: Sequence[str],
    project_root: Path,
    checkpoint_root: Path,
    project_tree_sha256: str,
    environment_sha256: str,
    dependency_ids: Sequence[str] = (),
    extra_environment: Mapping[str, str] | None = None,
) -> TestCheckpoint:
    """Run one test command or reuse its exact authenticated passed receipt."""

    if not command:
        raise Stage12RegressionError("test command is empty")
    if re.fullmatch(r"[a-z0-9_]+", node_name) is None:
        raise Stage12RegressionError("test node name is unsafe")
    root = project_root.absolute()
    checkpoints = checkpoint_root.absolute()
    try:
        checkpoints.relative_to(root)
    except ValueError as exc:
        raise Stage12RegressionError("checkpoint root escaped project root") from exc
    overrides = {str(key): str(value) for key, value in (extra_environment or {}).items()}
    if set(overrides) - {"PYTHONHASHSEED"}:
        raise Stage12RegressionError("unsupported Stage 12 test environment override")
    process_environment = {**SANITIZED_ENVIRONMENT, **overrides}
    fingerprint, digest = _checkpoint_fingerprint(
        node_name=node_name,
        command=command,
        project_tree_sha256=project_tree_sha256,
        environment_sha256=environment_sha256,
        dependency_ids=dependency_ids,
        environment_overrides=overrides,
        process_environment=process_environment,
    )
    checkpoint_id = f"stage12-checkpoint-{node_name}-{digest[:24]}"
    destination = checkpoints / checkpoint_id
    existing = _validate_existing(
        destination,
        fingerprint=fingerprint,
        fingerprint_sha256=digest,
        project_root=root,
    )
    if existing is not None:
        return TestCheckpoint(
            checkpoint_id=checkpoint_id,
            path=destination,
            receipt=existing,
            reused_existing=True,
        )

    executable = Path(str(command[0]))
    if executable.is_absolute():
        try:
            executable.relative_to(root)
        except ValueError as exc:
            raise Stage12RegressionError(
                "test executable must be inside the project"
            ) from exc
    elif ".." in executable.parts:
        raise Stage12RegressionError("test executable path is unsafe")
    if len(command) < 2 or str(command[1]) != SAFE_RUNNER:
        raise Stage12RegressionError(
            "Stage 12 regression must use the frozen safe runner"
        )

    checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(process_environment)
    with tempfile.TemporaryDirectory(
        prefix=f"{checkpoint_id}.", dir=checkpoints.parent
    ) as temporary:
        junit = Path(temporary) / "pytest.xml"
        access_audit_path = Path(temporary) / "access_audit.json"
        invoked = [str(part) for part in command]
        invoked.append(f"--junitxml={junit}")
        invoked.append(f"--stage12-access-audit={access_audit_path}")
        completed = subprocess.run(
            invoked,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            check=False,
        )
        if not junit.is_file():
            tail = completed.stdout[-4000:]
            raise Stage12RegressionError(
                f"{node_name} produced no JUnit receipt (exit "
                f"{completed.returncode}):\n{tail}"
            )
        summary, junit_payload, test_case_identities = _parse_junit(junit)
        try:
            access_audit, access_audit_payload = _load_access_audit(
                access_audit_path
            )
        except Stage12RegressionError as exc:
            tail = completed.stdout[-8000:]
            raise Stage12RegressionError(
                f"{node_name} access audit failed after exit "
                f"{completed.returncode}: {exc}\n{tail}"
            ) from exc
        if (
            completed.returncode != 0
            or int(summary["failures"]) != 0
            or int(summary["errors"]) != 0
            or int(summary["skipped"]) != 0
        ):
            tail = completed.stdout[-8000:]
            raise Stage12RegressionError(
                f"{node_name} failed (exit {completed.returncode}, "
                f"{summary}):\n{tail}"
            )

    receipt = {
        "schema_version": 2,
        "checkpoint_id": checkpoint_id,
        "status": "PASSED",
        "input_fingerprint": fingerprint,
        "input_fingerprint_sha256": digest,
        "summary": summary,
        "exit_code": completed.returncode,
        "junit_receipt": {
            "sha256": sha256_bytes(junit_payload),
            "size_bytes": len(junit_payload),
            "test_case_identity_sha256": summary["test_case_identity_sha256"],
            "test_case_identities": test_case_identities,
        },
        "access_audit": {
            **dict(access_audit),
            "receipt_sha256": sha256_bytes(access_audit_payload),
            "receipt_size_bytes": len(access_audit_payload),
        },
        "memoization": {
            "reused_prior_stage": False,
            "reusable_only_for_exact_input_fingerprint": True,
            "failed_attempts_published": False,
        },
    }
    receipt_bytes = canonical_json_bytes(receipt)
    success = {
        "schema_version": 1,
        "checkpoint_id": checkpoint_id,
        "status": "PASSED",
        "receipt_sha256": sha256_bytes(receipt_bytes),
    }
    success_bytes = canonical_json_bytes(success)
    hashes = {
        "receipt.json": sha256_bytes(receipt_bytes),
        "_SUCCESS.json": sha256_bytes(success_bytes),
    }
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    with SecureStagingDirectory.create(
        trusted_root=root,
        destination=destination,
        expected_files=CHECKPOINT_FILES,
        label=f"Stage 12 {node_name} checkpoint",
    ) as stage:
        _write_member(stage, "receipt.json", receipt_bytes)
        _write_member(stage, "_SUCCESS.json", success_bytes)
        snapshot = stage.snapshot(expected_hashes=hashes)
        published = stage.publish(expected_hashes=snapshot.hashes)
    validated = _validate_existing(
        published.path,
        fingerprint=fingerprint,
        fingerprint_sha256=digest,
        project_root=root,
    )
    if validated is None:
        raise Stage12RegressionError("published checkpoint disappeared")
    return TestCheckpoint(
        checkpoint_id=checkpoint_id,
        path=published.path,
        receipt=validated,
        reused_existing=published.reused_existing,
    )


def checkpoint_receipt_for_bundle(
    checkpoint: TestCheckpoint,
    *,
    project_root: Path,
) -> dict[str, Any]:
    receipt_file = checkpoint.path / "receipt.json"
    success_file = checkpoint.path / "_SUCCESS.json"
    receipt, receipt_record = load_json_object(receipt_file)
    success, success_record = load_json_object(success_file)
    require(
        success.get("receipt_sha256") == receipt_record.sha256,
        "checkpoint success/receipt hash differs",
    )
    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "relative_path": checkpoint.path.relative_to(project_root).as_posix(),
        "receipt_sha256": receipt_record.sha256,
        "success_sha256": success_record.sha256,
        "status": receipt["status"],
        "summary": dict(receipt["summary"]),
        "input_fingerprint_sha256": receipt["input_fingerprint_sha256"],
        "junit_receipt": dict(receipt["junit_receipt"]),
        "access_audit": dict(receipt["access_audit"]),
    }


__all__ = [
    "Stage12RegressionError",
    "TestCheckpoint",
    "checkpoint_receipt_for_bundle",
    "run_or_reuse_test_checkpoint",
]
