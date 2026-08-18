#!/usr/bin/env python3
"""Run non-authoritative Phase 2 CRSP numeric/unit discovery.

The compiler itself is non-writing.  This helper may preserve its aggregate-only
return value in an explicitly requested temporary path so the evidence can be
reviewed without publishing anything into the project or database trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from portfolio_bot.data.phase_02_crsp_numeric_unit_profile import (
    canonical_json_bytes,
    load_settings,
    run_numeric_unit_discovery,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validated_ephemeral_output(value: str, *, forbidden_roots: tuple[Path, ...]) -> Path:
    path = Path(value).expanduser().absolute()
    allowed_roots = {
        Path("/private/tmp").resolve(strict=False),
        Path(tempfile.gettempdir()).resolve(strict=False),
    }
    resolved = path.resolve(strict=False)
    if not any(_is_below(resolved, root) for root in allowed_roots):
        raise SystemExit("--output must be below /private/tmp or the OS temporary root")
    if any(_is_below(resolved, root.resolve(strict=False)) for root in forbidden_roots):
        raise SystemExit("--output may not be inside project or database roots")
    if path.exists() or path.is_symlink():
        raise SystemExit("--output already exists; refusing to overwrite it")
    return path


def _review_record(result: Mapping[str, Any]) -> dict[str, Any]:
    quality_payloads = result["quality_payloads"]
    tables = result["tables"]
    aggregate_result = {
        key: value for key, value in result.items() if key != "quality_payloads"
    }
    return {
        "schema_version": 1,
        "aggregate_result": aggregate_result,
        "quality_payload_sha256": {
            name: _sha256(payload)
            for name, payload in sorted(quality_payloads.items())
        },
        "table_sha256": {
            name: _sha256(canonical_json_bytes(rows))
            for name, rows in sorted(tables.items())
        },
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write while preserving discovery review record")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True,
        help="New aggregate-only JSON path below /private/tmp or the OS temp root.",
    )
    arguments = parser.parse_args()
    settings = load_settings()
    output = _validated_ephemeral_output(
        arguments.output,
        forbidden_roots=(settings.project_root, settings.database_root),
    )

    print("numeric-unit discovery: authenticated aggregate scan starting", flush=True)
    result = run_numeric_unit_discovery(settings)
    record = _review_record(result)
    payload = canonical_json_bytes(record)
    _write_exclusive(output, payload)
    receipt = {
        "build_id": result["build_id"],
        "output_path": output.as_posix(),
        "output_sha256": _sha256(payload),
        "summary": result["summary"],
        "table_sha256": record["table_sha256"],
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
