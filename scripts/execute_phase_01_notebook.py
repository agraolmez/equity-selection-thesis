#!/usr/bin/env python3
"""Execute the exact Phase 1 notebook source and publish causal evidence."""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import sys
import uuid
from pathlib import Path
from typing import Any

import nbformat as nbf
from nbclient import NotebookClient

from portfolio_bot.data.phase_01_evidence import implementation_receipt
from portfolio_bot.data.phase_01_notebook import (
    PHASE_01_NOTEBOOK_RELATIVE_PATH,
    build_phase_01_notebook,
    canonical_phase_01_notebook_bytes,
)
from portfolio_bot.data.phase_01_review import PROJECT_ROOT, load_review_context


SCHEMA_VERSION = 1
EXECUTION_TIMEOUT_SECONDS = 600
EXPECTED_CODE_CELL_COUNT = 7
RECEIPT_KEYS = {
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


def _fingerprint(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _stable_file_receipt(
    project_root: Path,
    path: Path,
) -> tuple[dict[str, Any], bytes]:
    """Read one regular file while proving its identity stayed unchanged."""
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"Notebook evidence must be a non-symlink regular file: {path}")
    with path.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        if _fingerprint(opened_before) != _fingerprint(before):
            raise RuntimeError(f"Notebook evidence changed while opening: {path}")
        content = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = path.lstat()
    if not (
        _fingerprint(before)
        == _fingerprint(opened_before)
        == _fingerprint(opened_after)
        == _fingerprint(after)
    ):
        raise RuntimeError(f"Notebook evidence changed while reading: {path}")
    if len(content) != before.st_size:
        raise RuntimeError(f"Notebook evidence size changed while reading: {path}")
    return (
        {
            "path": path.relative_to(project_root.resolve()).as_posix(),
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
        content,
    )


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Any) -> None:
    content = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, content)


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


def _canonical_cell_contract(notebook: nbf.NotebookNode) -> list[dict[str, Any]]:
    return [
        {
            "id": cell.get("id"),
            "cell_type": cell.get("cell_type"),
            "source": cell.get("source"),
            "metadata": {
                key: value
                for key, value in dict(cell.get("metadata", {})).items()
                if key != "execution"
            },
        }
        for cell in notebook.get("cells", [])
    ]


def _text_plain(output: nbf.NotebookNode) -> str | None:
    data = output.get("data", {})
    value = data.get("text/plain")
    if isinstance(value, list):
        return "".join(value)
    return value if isinstance(value, str) else None


def _validate_executed_notebook(
    executed: nbf.NotebookNode,
    canonical: nbf.NotebookNode,
    expected_result: dict[str, Any],
) -> dict[str, Any]:
    if _canonical_cell_contract(executed) != _canonical_cell_contract(canonical):
        raise RuntimeError("Notebook cell sources or reader-facing metadata changed during execution")

    context_keys = (
        "phase",
        "role",
        "raw_snapshot_id",
        "phase_01_build_id",
        "verification_id",
        "phase_status",
    )
    if any(
        executed.get("metadata", {}).get(key)
        != canonical.get("metadata", {}).get(key)
        for key in context_keys
    ):
        raise RuntimeError("Notebook context metadata changed during execution")

    code_cells = [
        cell for cell in executed.get("cells", []) if cell.get("cell_type") == "code"
    ]
    execution_counts = [cell.get("execution_count") for cell in code_cells]
    errors = [
        {
            "code_cell": index,
            "ename": output.get("ename"),
            "evalue": output.get("evalue"),
        }
        for index, cell in enumerate(code_cells, start=1)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if len(code_cells) != EXPECTED_CODE_CELL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CODE_CELL_COUNT} code cells, found {len(code_cells)}"
        )
    if execution_counts != list(range(1, EXPECTED_CODE_CELL_COUNT + 1)):
        raise RuntimeError(f"Notebook execution counts are not sequential: {execution_counts}")
    if errors:
        raise RuntimeError(f"Notebook contains execution errors: {errors}")

    final_results: list[Any] = []
    for output in code_cells[-1].get("outputs", []):
        if output.get("output_type") not in {"execute_result", "display_data"}:
            continue
        text = _text_plain(output)
        if text is None:
            continue
        try:
            final_results.append(ast.literal_eval(text))
        except (SyntaxError, ValueError):
            continue
    if final_results != [expected_result]:
        raise RuntimeError(
            "Notebook final code cell did not emit the exact Phase 1 gate result: "
            f"{final_results}"
        )

    return {
        "code_cell_count": len(code_cells),
        "execution_counts": execution_counts,
        "error_count": len(errors),
        "source_contract_matches": True,
        "metadata_context_matches": True,
        "final_result": expected_result,
    }


def _render_notebook(notebook: nbf.NotebookNode) -> bytes:
    rendered = nbf.writes(notebook)
    if not rendered.endswith("\n"):
        rendered += "\n"
    return rendered.encode("utf-8")


def _engine_contract(canonical: nbf.NotebookNode) -> dict[str, Any]:
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "nbformat_version": importlib.metadata.version("nbformat"),
        "nbclient_version": importlib.metadata.version("nbclient"),
        "kernel_name": canonical["metadata"]["kernelspec"]["name"],
        "timeout_seconds": EXECUTION_TIMEOUT_SECONDS,
        "allow_errors": False,
        "record_timing": False,
    }


def _content_receipt(project_root: Path, path: Path, content: bytes) -> dict[str, Any]:
    return {
        "path": path.relative_to(project_root.resolve()).as_posix(),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _existing_execution_is_authenticated(
    *,
    project_root: Path,
    receipt_path: Path,
    current_receipt: dict[str, Any],
    current_bytes: bytes,
    canonical: nbf.NotebookNode,
    canonical_receipt: dict[str, Any],
    expected_result: dict[str, Any],
    engine: dict[str, Any],
    implementation: list[dict[str, Any]],
) -> bool:
    """Accept an already-executed notebook only with a complete causal receipt."""
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return False
    try:
        _, receipt_bytes = _stable_file_receipt(project_root, receipt_path)
        receipt = json.loads(receipt_bytes.decode("utf-8"))
        executed = nbf.reads(current_bytes.decode("utf-8"), as_version=4)
        execution = _validate_executed_notebook(executed, canonical, expected_result)
    except (OSError, UnicodeDecodeError, ValueError, nbf.ValidationError, RuntimeError):
        return False
    if not isinstance(receipt, dict):
        return False
    started = _utc_timestamp(receipt.get("run_started_utc"))
    completed = _utc_timestamp(receipt.get("run_completed_utc"))
    return (
        set(receipt) == RECEIPT_KEYS
        and type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("phase") == "01_raw_snapshot"
        and receipt.get("status") == "passed"
        and started is not None
        and completed is not None
        and started <= completed
        and receipt.get("raw_snapshot_id") == expected_result["raw_snapshot_id"]
        and receipt.get("phase_01_build_id") == expected_result["phase_01_build_id"]
        and receipt.get("verification_id") == expected_result["verification_id"]
        and receipt.get("cwd") == str(project_root.resolve())
        and receipt.get("canonical_notebook_before") == canonical_receipt
        and receipt.get("executed_notebook_after") == current_receipt
        and receipt.get("execution") == execution
        and receipt.get("engine") == engine
        and receipt.get("implementation_before") == implementation
        and receipt.get("implementation_after") == implementation
    )


def main() -> int:
    project_root = PROJECT_ROOT.resolve()
    context = load_review_context(project_root)
    summary = context["summary"]
    notebook_path = project_root / PHASE_01_NOTEBOOK_RELATIVE_PATH
    run_directory = (
        project_root
        / "runs"
        / "phase_01_raw_snapshot"
        / summary["phase_01_build_id"]
    )
    receipt_path = run_directory / "notebook_execution_receipt.json"
    staged_notebook_path = notebook_path.with_name(
        f".{notebook_path.name}.executed.pending"
    )

    implementation_before = implementation_receipt(project_root)
    canonical_bytes = canonical_phase_01_notebook_bytes(project_root)
    canonical = nbf.reads(canonical_bytes.decode("utf-8"), as_version=4)
    code_cells = [
        cell for cell in canonical.get("cells", []) if cell.get("cell_type") == "code"
    ]
    if (
        len(code_cells) != EXPECTED_CODE_CELL_COUNT
        or any(cell.get("execution_count") is not None for cell in code_cells)
        or any(cell.get("outputs") for cell in code_cells)
    ):
        raise RuntimeError("Canonical Phase 1 notebook must be clean and unexecuted")

    expected_result = {
        "phase": "01_raw_snapshot",
        "status": "verified_pending_explicit_phase_freeze",
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "verification_id": summary["verification_id"],
        "phase_2_permitted": False,
    }
    canonical_metadata = build_phase_01_notebook(project_root).get("metadata", {})
    expected_metadata_context = {
        "phase": "01_raw_snapshot",
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "verification_id": summary["verification_id"],
        "phase_status": "verified_pending_explicit_phase_freeze",
    }
    if any(
        canonical.get("metadata", {}).get(key) != value
        or canonical_metadata.get(key) != value
        for key, value in expected_metadata_context.items()
    ):
        raise RuntimeError("Canonical notebook metadata does not match the verified context")

    engine = _engine_contract(canonical)
    expected_canonical_receipt = _content_receipt(
        project_root, notebook_path, canonical_bytes
    )
    current_receipt, observed_bytes = _stable_file_receipt(project_root, notebook_path)
    if observed_bytes != canonical_bytes:
        if _existing_execution_is_authenticated(
            project_root=project_root,
            receipt_path=receipt_path,
            current_receipt=current_receipt,
            current_bytes=observed_bytes,
            canonical=canonical,
            canonical_receipt=expected_canonical_receipt,
            expected_result=expected_result,
            engine=engine,
            implementation=implementation_before,
        ):
            staged_notebook_path.unlink(missing_ok=True)
            print(receipt_path)
            return 0
        raise RuntimeError(
            "The notebook is neither canonical unexecuted source nor an already-executed "
            "artifact authenticated by the current causal receipt. Run "
            "`PYTHONPATH=src python scripts/build_phase_01_notebook.py` to restore the "
            "canonical source, then rerun this executor."
        )
    if current_receipt != expected_canonical_receipt:
        raise RuntimeError("Canonical notebook receipt differs from its deterministic bytes")

    started = dt.datetime.now(dt.timezone.utc)
    client = NotebookClient(
        canonical,
        timeout=EXECUTION_TIMEOUT_SECONDS,
        kernel_name=canonical["metadata"]["kernelspec"]["name"],
        allow_errors=False,
        record_timing=False,
        resources={"metadata": {"path": str(project_root)}},
    )
    executed = client.execute()
    execution = _validate_executed_notebook(executed, canonical, expected_result)
    executed_bytes = _render_notebook(executed)
    try:
        _atomic_bytes(staged_notebook_path, executed_bytes)
        staged_receipt, staged_bytes = _stable_file_receipt(
            project_root, staged_notebook_path
        )
        if staged_bytes != executed_bytes:
            raise RuntimeError("Staged notebook bytes differ from the validated execution")
        executed_receipt = {
            **staged_receipt,
            "path": PHASE_01_NOTEBOOK_RELATIVE_PATH.as_posix(),
        }

        implementation_after = implementation_receipt(project_root)
        if implementation_after != implementation_before:
            raise RuntimeError(
                "Phase 1 implementation changed while the notebook was executing; "
                "the canonical notebook was not replaced"
            )
        completed = dt.datetime.now(dt.timezone.utc)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "phase": "01_raw_snapshot",
            "status": "passed",
            "run_started_utc": started.isoformat(),
            "run_completed_utc": completed.isoformat(),
            "raw_snapshot_id": summary["raw_snapshot_id"],
            "phase_01_build_id": summary["phase_01_build_id"],
            "verification_id": summary["verification_id"],
            "cwd": str(project_root),
            "canonical_notebook_before": current_receipt,
            "executed_notebook_after": executed_receipt,
            "execution": execution,
            "engine": engine,
            "implementation_before": implementation_before,
            "implementation_after": implementation_after,
        }

        # Commit the receipt first.  If publication stops here, the canonical
        # notebook remains rerunnable and this receipt is safely overwritten.
        _atomic_json(receipt_path, receipt)
        os.replace(staged_notebook_path, notebook_path)
        published_receipt, published_bytes = _stable_file_receipt(
            project_root, notebook_path
        )
        if published_bytes != executed_bytes or published_receipt != executed_receipt:
            raise RuntimeError(
                "Published notebook bytes differ from the receipt-authenticated stage"
            )
    finally:
        staged_notebook_path.unlink(missing_ok=True)

    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
