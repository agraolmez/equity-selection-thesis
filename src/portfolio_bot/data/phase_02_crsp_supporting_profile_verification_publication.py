"""Safely publish independent verification of the CRSP supporting profile.

Passing and failing reports both receive an immutable two-file evidence bundle.
Only a zero-difference passing report may move the verified pointer, and that
pointer is committed last after all primary, raw, and verifier bytes are
reauthenticated.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = 1
PROFILE_NAME = "crsp_development_supporting_source_structure"
PASS_STATUS = "verified_passed"
FAIL_STATUS = "verification_failed"
VERIFICATION_FILES = ("differences.csv", "independent_check.json")
IMPLEMENTATION_COMPONENT_LABELS = (
    "scripts/verify_phase_02_crsp_supporting_profile.py",
    "src/portfolio_bot/data/phase_02_crsp_supporting_profile_independent_scan.py",
    "src/portfolio_bot/data/phase_02_crsp_supporting_profile_verification_publication.py",
    "src/portfolio_bot/data/verify_phase_02_crsp_supporting_profile.py",
)
DIFFERENCE_FIELDS = (
    "check_id",
    "category",
    "expected",
    "actual",
    "evidence",
    "message",
)
REPORT_KEYS = (
    "build_id",
    "checks",
    "contract_id",
    "independence",
    "primary_manifest_sha256",
    "profile_name",
    "schema_version",
    "status",
    "summary",
    "verification_id",
    "verifier_implementation_sha256",
)
CHECK_KEYS = (
    "actual",
    "category",
    "check_id",
    "evidence",
    "expected",
    "message",
    "status",
)
SUMMARY_KEYS = (
    "check_count",
    "difference_count",
    "fail_count",
    "full_file_rows_read",
    "pass_count",
    "retained_rows",
    "source_count",
)
INDEPENDENCE_KEYS = (
    "aggregate_only",
    "full_raw_hashes_recomputed",
    "grouping_depends_on_input_sort",
    "primary_compiler_imported",
    "primary_publisher_imported",
    "raw_files_reread",
)
VERIFIED_POINTER_KEYS = (
    "build_id",
    "contract_id",
    "differences_sha256",
    "independent_check_sha256",
    "primary_manifest_sha256",
    "schema_version",
    "status",
    "verification_id",
    "verification_path",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_VERIFICATION_ID_RE = re.compile(r"verify-[0-9a-f]{16}\Z")
_VERIFICATION_ID_DOMAIN = b"phase02-crsp-supporting-profile-independent-v1\x00"
_IMPLEMENTATION_BUNDLE_DOMAIN = b"phase02-crsp-supporting-verifier-bundle-v1\x00"


class SupportingVerificationPublicationError(RuntimeError):
    """Raised when independent evidence cannot be published safely."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise SupportingVerificationPublicationError(
            f"Value is not canonical JSON: {exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _assert_no_symlink_components(path: Path, label: str) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise SupportingVerificationPublicationError(
                f"Symlink is prohibited in {label} path: {current}"
            )


def _read_authenticated(path: Path, label: str) -> bytes:
    path = _lexical_absolute(path)
    _assert_no_symlink_components(path, label)
    try:
        before = path.lstat()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise SupportingVerificationPublicationError(
            f"Cannot open authenticated {label}: {path}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or before.st_dev != opened.st_dev
            or before.st_ino != opened.st_ino
        ):
            raise SupportingVerificationPublicationError(
                f"Authenticated {label} is not the selected regular file"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        final_descriptor = os.fstat(descriptor)
        after = path.lstat()
        if (
            opened.st_dev != final_descriptor.st_dev
            or opened.st_ino != final_descriptor.st_ino
            or opened.st_size != final_descriptor.st_size
            or opened.st_mtime_ns != final_descriptor.st_mtime_ns
            or final_descriptor.st_dev != after.st_dev
            or final_descriptor.st_ino != after.st_ino
            or final_descriptor.st_size != after.st_size
            or final_descriptor.st_mtime_ns != after.st_mtime_ns
        ):
            raise SupportingVerificationPublicationError(
                f"Authenticated {label} changed while being read"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _safe_component(value: Any, label: str) -> str:
    text = str(value)
    if (
        not text
        or text in {".", ".."}
        or Path(text).name != text
        or "/" in text
        or "\\" in text
    ):
        raise SupportingVerificationPublicationError(
            f"Unsafe {label} path component: {text!r}"
        )
    return text


def _assert_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SupportingVerificationPublicationError(f"{label} is not SHA-256")
    return value


def _canonical_cell(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def render_differences_csv(report: Mapping[str, Any]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(DIFFERENCE_FIELDS),
        delimiter=",",
        quotechar='"',
        doublequote=True,
        escapechar=None,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    writer.writeheader()
    for check in report.get("checks", []):
        if check.get("status") == "PASS":
            continue
        writer.writerow(
            {
                "check_id": check["check_id"],
                "category": check["category"],
                "expected": _canonical_cell(check["expected"]),
                "actual": _canonical_cell(check["actual"]),
                "evidence": check["evidence"],
                "message": check["message"],
            }
        )
    return output.getvalue().encode("utf-8")


def derive_verification_id(
    report: Mapping[str, Any], differences_payload: bytes
) -> str:
    identity = dict(report)
    identity["verification_id"] = ""
    digest = hashlib.sha256()
    digest.update(_VERIFICATION_ID_DOMAIN)
    digest.update(canonical_json_bytes(identity))
    digest.update(differences_payload)
    return f"verify-{digest.hexdigest()[:16]}"


def bind_verification_id(
    report: Mapping[str, Any], differences_payload: bytes
) -> dict[str, Any]:
    bound = json.loads(canonical_json_bytes(dict(report)).decode("utf-8"))
    bound["verification_id"] = derive_verification_id(bound, differences_payload)
    return bound


def _normalize_implementation_paths(
    implementation_paths: Mapping[str, Path] | Sequence[Path],
    project_root: Path,
) -> dict[str, Path]:
    declared: dict[str, Path] = {}
    if isinstance(implementation_paths, Mapping):
        items = implementation_paths.items()
    else:
        items = (("", path) for path in implementation_paths)
    for label, raw_path in items:
        path = _lexical_absolute(raw_path)
        _assert_no_symlink_components(path, "verifier implementation")
        try:
            derived = path.relative_to(project_root).as_posix()
        except ValueError as exc:
            raise SupportingVerificationPublicationError(
                "Verifier implementation is outside project root"
            ) from exc
        if label and label != derived:
            raise SupportingVerificationPublicationError(
                "Declared implementation label differs from path"
            )
        if derived in declared:
            raise SupportingVerificationPublicationError(
                "Duplicate verifier implementation component"
            )
        declared[derived] = path
    if set(declared) != set(IMPLEMENTATION_COMPONENT_LABELS):
        raise SupportingVerificationPublicationError(
            "Verifier implementation component set differs"
        )
    return dict(sorted(declared.items()))


def implementation_bundle_sha256(
    implementation_paths: Mapping[str, Path] | Sequence[Path],
    *,
    project_root: Path,
) -> str:
    project_root = _lexical_absolute(project_root)
    normalized = _normalize_implementation_paths(implementation_paths, project_root)
    digest = hashlib.sha256()
    digest.update(_IMPLEMENTATION_BUNDLE_DOMAIN)
    for label, path in normalized.items():
        payload = _read_authenticated(path, f"implementation {label}")
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _validate_report_and_payload(
    report: Mapping[str, Any], differences_payload: bytes
) -> tuple[dict[str, Any], bytes]:
    if set(report) != set(REPORT_KEYS):
        raise SupportingVerificationPublicationError("Report schema differs")
    if (
        report["schema_version"] != SCHEMA_VERSION
        or report["profile_name"] != PROFILE_NAME
    ):
        raise SupportingVerificationPublicationError("Report identity differs")
    _safe_component(report["build_id"], "build_id")
    _safe_component(report["contract_id"], "contract_id")
    _assert_sha(report["primary_manifest_sha256"], "primary manifest hash")
    _assert_sha(report["verifier_implementation_sha256"], "implementation hash")
    if report["status"] not in {PASS_STATUS, FAIL_STATUS}:
        raise SupportingVerificationPublicationError("Report status differs")
    independence = report["independence"]
    if not isinstance(independence, Mapping) or set(independence) != set(
        INDEPENDENCE_KEYS
    ):
        raise SupportingVerificationPublicationError("Independence schema differs")
    expected_independence = {
        "aggregate_only": True,
        "full_raw_hashes_recomputed": True,
        "grouping_depends_on_input_sort": False,
        "primary_compiler_imported": False,
        "primary_publisher_imported": False,
        "raw_files_reread": 5,
    }
    if dict(independence) != expected_independence:
        raise SupportingVerificationPublicationError(
            "Report does not prove independent five-source aggregate scanning"
        )
    checks = report["checks"]
    if not isinstance(checks, list) or not checks:
        raise SupportingVerificationPublicationError("Report checks are missing")
    previous = ""
    for check in checks:
        if not isinstance(check, Mapping) or set(check) != set(CHECK_KEYS):
            raise SupportingVerificationPublicationError("Check schema differs")
        check_id = check["check_id"]
        if not isinstance(check_id, str) or not check_id or check_id <= previous:
            raise SupportingVerificationPublicationError(
                "Checks are not uniquely sorted"
            )
        previous = check_id
        if check["status"] not in {"PASS", "FAIL"}:
            raise SupportingVerificationPublicationError("Check status differs")
        for key in ("category", "evidence", "message"):
            if not isinstance(check[key], str):
                raise SupportingVerificationPublicationError(f"Check {key} is not text")
    summary = report["summary"]
    if not isinstance(summary, Mapping) or set(summary) != set(SUMMARY_KEYS):
        raise SupportingVerificationPublicationError("Report summary schema differs")
    for key, value in summary.items():
        if type(value) is not int or value < 0:
            raise SupportingVerificationPublicationError(
                f"Summary {key} is not a nonnegative integer"
            )
    failures = sum(check["status"] == "FAIL" for check in checks)
    if (
        summary["check_count"] != len(checks)
        or summary["fail_count"] != failures
        or summary["difference_count"] != failures
        or summary["pass_count"] != len(checks) - failures
        or (report["status"] == PASS_STATUS) != (failures == 0)
    ):
        raise SupportingVerificationPublicationError(
            "Report status/summary does not reconcile"
        )
    expected_differences = render_differences_csv(report)
    if differences_payload != expected_differences:
        raise SupportingVerificationPublicationError(
            "Differences payload differs from failed checks"
        )
    expected_id = derive_verification_id(report, differences_payload)
    if (
        not isinstance(report["verification_id"], str)
        or _VERIFICATION_ID_RE.fullmatch(report["verification_id"]) is None
        or report["verification_id"] != expected_id
    ):
        raise SupportingVerificationPublicationError("Verification ID differs")
    report_bytes = canonical_json_bytes(report)
    return json.loads(report_bytes.decode("utf-8")), report_bytes


def _assert_optional_regular(path: Path, label: str) -> None:
    _assert_no_symlink_components(path, label)
    if path.exists() and not path.is_file():
        raise SupportingVerificationPublicationError(
            f"Existing {label} is not a regular file"
        )


def _read_closed_directory(directory: Path, label: str) -> dict[str, bytes]:
    _assert_no_symlink_components(directory, label)
    if not directory.is_dir():
        raise SupportingVerificationPublicationError(f"{label} is not a directory")
    result: dict[str, bytes] = {}
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.is_symlink() or not entry.is_file():
            raise SupportingVerificationPublicationError(
                f"{label} contains a non-regular entry: {entry.name}"
            )
        result[entry.name] = _read_authenticated(entry, f"{label}/{entry.name}")
    return result


def _assert_existing_matches(
    directory: Path, payloads: Mapping[str, bytes], label: str
) -> bool:
    _assert_no_symlink_components(directory, label)
    if not directory.exists() and not directory.is_symlink():
        return False
    if _read_closed_directory(directory, label) != dict(payloads):
        raise SupportingVerificationPublicationError(
            f"Existing {label} differs from deterministic payload"
        )
    return True


def _remove_stage(stage: Path) -> None:
    try:
        mode = stage.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        stage.unlink(missing_ok=True)
    else:
        shutil.rmtree(stage, ignore_errors=True)


def _stage_payloads(parent: Path, prefix: str, payloads: Mapping[str, bytes]) -> Path:
    _assert_no_symlink_components(parent, "verification staging parent")
    parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(parent, "verification staging parent")
    stage = Path(tempfile.mkdtemp(prefix=f".{prefix}.staging-", dir=parent))
    descriptor: int | None = None
    try:
        descriptor = os.open(
            stage,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        path_stat = stage.lstat()
        if opened.st_dev != path_stat.st_dev or opened.st_ino != path_stat.st_ino:
            raise SupportingVerificationPublicationError(
                "Verification staging identity changed"
            )
        for name, payload in payloads.items():
            if name not in VERIFICATION_FILES:
                raise SupportingVerificationPublicationError(
                    f"Unexpected verification file: {name}"
                )
            file_descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=descriptor,
            )
            try:
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(file_descriptor, remaining)
                    if written <= 0:
                        raise SupportingVerificationPublicationError(
                            "Short verification staging write"
                        )
                    remaining = remaining[written:]
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
    except Exception:
        _remove_stage(stage)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if _read_closed_directory(stage, "verification staging") != dict(payloads):
        _remove_stage(stage)
        raise SupportingVerificationPublicationError("Staged evidence bytes differ")
    return stage


def _publish_stage(stage: Path, final: Path, payloads: Mapping[str, bytes]) -> None:
    if _assert_existing_matches(final, payloads, "verification bundle"):
        _remove_stage(stage)
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(final, "verification bundle")
    try:
        os.rename(stage, final)
    except OSError:
        if _assert_existing_matches(final, payloads, "verification bundle"):
            _remove_stage(stage)
            return
        raise
    if _read_closed_directory(final, "verification bundle") != dict(payloads):
        raise SupportingVerificationPublicationError(
            "Published verification bytes changed"
        )


def _write_atomic_file(
    path: Path,
    payload: bytes,
    *,
    pre_replace_gate: Callable[[], None] | None = None,
) -> None:
    _assert_no_symlink_components(path, "verified pointer")
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(path, "verified pointer")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if pre_replace_gate is not None:
            pre_replace_gate()
        _assert_no_symlink_components(path, "verified pointer")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalize_snapshot(snapshot: Mapping[Path, str]) -> dict[Path, str]:
    normalized: dict[Path, str] = {}
    for raw_path, digest in snapshot.items():
        path = _lexical_absolute(raw_path)
        _assert_sha(digest, f"snapshot hash for {path}")
        if path in normalized:
            raise SupportingVerificationPublicationError("Duplicate snapshot path")
        normalized[path] = digest
    if not normalized:
        raise SupportingVerificationPublicationError("Final snapshot is empty")
    return normalized


def _assert_snapshot(snapshot: Mapping[Path, str]) -> None:
    for path, expected in snapshot.items():
        actual = _sha256_bytes(_read_authenticated(path, f"snapshot {path}"))
        if actual != expected:
            raise SupportingVerificationPublicationError(
                f"Final context changed: {path}"
            )


def _raw_stat(path: Path) -> dict[str, int]:
    _assert_no_symlink_components(path, "raw source")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise SupportingVerificationPublicationError("Raw source is not regular")
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(metadata.st_mode),
        "size_bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
    }


def _normalize_raw_snapshot(
    snapshot: Mapping[Path, Mapping[str, int]]
) -> dict[Path, dict[str, int]]:
    normalized: dict[Path, dict[str, int]] = {}
    required = {"device", "inode", "mode", "size_bytes", "mtime_ns"}
    for raw_path, values in snapshot.items():
        path = _lexical_absolute(raw_path)
        if set(values) != required or any(
            type(value) is not int for value in values.values()
        ):
            raise SupportingVerificationPublicationError(
                "Raw operational snapshot schema differs"
            )
        normalized[path] = dict(values)
    if len(normalized) != 5:
        raise SupportingVerificationPublicationError(
            "Raw operational snapshot must contain five sources"
        )
    return normalized


def _assert_raw_snapshot(snapshot: Mapping[Path, Mapping[str, int]]) -> None:
    for path, expected in snapshot.items():
        if _raw_stat(path) != dict(expected):
            raise SupportingVerificationPublicationError(
                f"Raw source changed before verified-pointer commit: {path.name}"
            )


def publish_independent_verification(
    report: Mapping[str, Any],
    differences_payload: bytes,
    *,
    artifact_root: Path,
    project_root: Path,
    config: Mapping[str, Any],
    verifier_implementation_paths: Mapping[str, Path] | Sequence[Path],
    final_snapshot: Mapping[Path, str],
    raw_operational_snapshot: Mapping[Path, Mapping[str, int]],
    primary_manifest_path: Path,
) -> dict[str, Path]:
    """Publish immutable evidence and a passing verified pointer, if eligible."""
    project_root = _lexical_absolute(project_root)
    artifact_root = _lexical_absolute(artifact_root)
    _assert_no_symlink_components(project_root, "project root")
    _assert_no_symlink_components(artifact_root, "artifact root")
    try:
        artifact_root.relative_to(project_root)
    except ValueError as exc:
        raise SupportingVerificationPublicationError(
            "Artifact root is outside project root"
        ) from exc
    frozen_report, report_payload = _validate_report_and_payload(
        report, differences_payload
    )
    normalized_implementations = _normalize_implementation_paths(
        verifier_implementation_paths, project_root
    )
    if (
        implementation_bundle_sha256(
            normalized_implementations, project_root=project_root
        )
        != frozen_report["verifier_implementation_sha256"]
    ):
        raise SupportingVerificationPublicationError(
            "Verifier implementation differs from report binding"
        )
    snapshots = _normalize_snapshot(final_snapshot)
    raw_snapshot = _normalize_raw_snapshot(raw_operational_snapshot)
    _assert_snapshot(snapshots)
    _assert_raw_snapshot(raw_snapshot)
    primary_manifest_path = _lexical_absolute(primary_manifest_path)
    primary_manifest_sha = _sha256_bytes(
        _read_authenticated(primary_manifest_path, "primary manifest")
    )
    if primary_manifest_sha != frozen_report["primary_manifest_sha256"]:
        raise SupportingVerificationPublicationError(
            "Primary manifest differs from report binding"
        )

    verification_subdirectory = _safe_component(
        config["outputs"]["verification_subdirectory"],
        "verification subdirectory",
    )
    build_id = _safe_component(frozen_report["build_id"], "build_id")
    verification_id = _safe_component(
        frozen_report["verification_id"], "verification_id"
    )
    verification_directory = _lexical_absolute(
        artifact_root / verification_subdirectory / build_id / verification_id
    )
    payloads = {
        "differences.csv": differences_payload,
        "independent_check.json": report_payload,
    }
    if set(payloads) != set(config["outputs"]["closed_verification_file_set"]):
        raise SupportingVerificationPublicationError(
            "Verification closed file set differs from config"
        )
    stage = _stage_payloads(verification_directory.parent, verification_id, payloads)
    try:
        _publish_stage(stage, verification_directory, payloads)
    finally:
        _remove_stage(stage)
    result = {"verification": verification_directory}

    def context_gate() -> None:
        _assert_snapshot(snapshots)
        _assert_raw_snapshot(raw_snapshot)
        if (
            implementation_bundle_sha256(
                normalized_implementations, project_root=project_root
            )
            != frozen_report["verifier_implementation_sha256"]
        ):
            raise SupportingVerificationPublicationError(
                "Verifier implementation changed during evidence publication"
            )
        if (
            _sha256_bytes(
                _read_authenticated(primary_manifest_path, "primary manifest")
            )
            != frozen_report["primary_manifest_sha256"]
        ):
            raise SupportingVerificationPublicationError(
                "Primary manifest changed during evidence publication"
            )
        if (
            _read_closed_directory(verification_directory, "verification bundle")
            != payloads
        ):
            raise SupportingVerificationPublicationError(
                "Verification bundle changed during evidence publication"
            )

    context_gate()

    # Failure evidence is durable, but a failed run never reads or mutates the
    # verified pointer endpoint.
    if frozen_report["status"] != PASS_STATUS:
        return result

    if frozen_report["summary"]["difference_count"] != 0:
        raise SupportingVerificationPublicationError(
            "Passing report contains differences"
        )
    try:
        verification_path = verification_directory.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise SupportingVerificationPublicationError(
            "Verification path is outside project root"
        ) from exc
    pointer = {
        "build_id": build_id,
        "contract_id": frozen_report["contract_id"],
        "differences_sha256": _sha256_bytes(differences_payload),
        "independent_check_sha256": _sha256_bytes(report_payload),
        "primary_manifest_sha256": frozen_report["primary_manifest_sha256"],
        "schema_version": SCHEMA_VERSION,
        "status": PASS_STATUS,
        "verification_id": verification_id,
        "verification_path": verification_path,
    }
    if set(pointer) != set(VERIFIED_POINTER_KEYS):
        raise SupportingVerificationPublicationError("Verified pointer schema differs")
    pointer_payload = canonical_json_bytes(pointer)
    pointer_name = _safe_component(
        config["outputs"]["verified_pointer"], "verified pointer filename"
    )
    pointer_path = _lexical_absolute(artifact_root / pointer_name)
    _assert_optional_regular(pointer_path, "verified pointer")

    def final_gate() -> None:
        context_gate()
        _assert_optional_regular(pointer_path, "verified pointer")

    final_gate()
    pointer_matches = pointer_path.exists() and (
        _read_authenticated(pointer_path, "verified pointer") == pointer_payload
    )
    if not pointer_matches:
        _write_atomic_file(pointer_path, pointer_payload, pre_replace_gate=final_gate)
    result["pointer"] = pointer_path
    return result


__all__ = [
    "SupportingVerificationPublicationError",
    "bind_verification_id",
    "canonical_json_bytes",
    "derive_verification_id",
    "implementation_bundle_sha256",
    "publish_independent_verification",
    "render_differences_csv",
]
