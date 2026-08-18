"""Publish standalone Phase 2 CRSP profile-verification evidence safely.

This module deliberately does not import either the primary profile compiler or
its publisher.  It receives an already constructed independent-check report and
an already rendered differences payload, validates their exact public contract,
and publishes only those two immutable files.  A verified pointer is committed
last and only for a zero-difference passing report.
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
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = 1
PROFILE_NAME = "crsp_development_duplicate_and_flag_profile"
PASS_STATUS = "verified_passed"
FAIL_STATUS = "verification_failed"

VERIFICATION_SUBDIRECTORY = "crsp_development_profile_verifications"
PRIMARY_BUILD_SUBDIRECTORY = "crsp_development_profiles"
PRIMARY_POINTER_NAME = "LATEST_CRSP_DEVELOPMENT_PROFILE_PRIMARY.json"
VERIFIED_POINTER_NAME = "LATEST_CRSP_DEVELOPMENT_PROFILE_VERIFIED.json"
VERIFICATION_FILES = ("differences.csv", "independent_check.json")
EMPTY_DIFFERENCES_SHA256 = (
    "3cb2ecb4f25c911917a8dce931f6c68d91a4c230306b2d44d572dd770e55dacf"
)
IMPLEMENTATION_COMPONENT_LABELS = (
    "scripts/verify_phase_02_crsp_development_profile.py",
    "src/portfolio_bot/data/phase_02_crsp_profile_independent_scan.py",
    "src/portfolio_bot/data/phase_02_crsp_profile_verification_publication.py",
    "src/portfolio_bot/data/verify_phase_02_crsp_development_profile.py",
)
FINAL_CONTEXT_LABELS = (
    "artifacts/phase_02_canonical_panel/LATEST_CALENDAR_VERIFIED.json",
    "artifacts/phase_02_canonical_panel/LATEST_CONTRACT_VERIFIED.json",
    "configs/paths.toml",
    "configs/phase_02_canonical_panel.toml",
    "configs/phase_02_canonicalization/crsp_development_v1.toml",
    "src/portfolio_bot/data/phase_02_crsp_development_profile.py",
)
RAW_SOURCE_RELATIVE_PATHS = (
    "raw/crsp_daily_ciz.csv",
    "raw/crsp_monthly_ciz.csv",
)
RAW_OPERATIONAL_STAT_KEYS = (
    "device",
    "inode",
    "mode",
    "mtime_ns",
    "size_bytes",
)
BASELINE_CHECK_IDS = (
    "CONFIG_PHASE_SHA256",
    "CONFIG_PROFILE_SHA256",
    "MONTHLY_MISSING_PRICE_DATE_EXACT",
    "MULTIPLICITY_FINGERPRINTS_EXACT",
    "PRIMARY_ARTIFACT_BYTES_MANIFEST_JSON",
    "PRIMARY_ARTIFACT_BYTES_OUTPUT_HASHES_SHA256",
    "PRIMARY_ARTIFACT_FILE_SET",
    "PRIMARY_ARTIFACT_PATH_EXACT",
    "PRIMARY_BINDING_RECONSTRUCTED",
    "PRIMARY_BUILD_ID_CONTENT_BOUND",
    "PRIMARY_MANIFEST_IDENTITY",
    "PRIMARY_MANIFEST_OUTPUTS_EXACT",
    "PRIMARY_MANIFEST_POINTER_HASH",
    "PRIMARY_MANIFEST_SCHEMA_AND_SERIALIZATION",
    "PRIMARY_OUTPUT_HASHES_MATCH_FILES",
    "PRIMARY_OUTPUT_HASH_INVENTORY_SCHEMA",
    "PRIMARY_POINTER_BINDING",
    "PRIMARY_POINTER_BYTES_RECONSTRUCTED",
    "PRIMARY_POINTER_SCHEMA_AND_SERIALIZATION",
    "PRIMARY_QUALITY_BYTES_CANONICALIZATION_RULES_JSON",
    "PRIMARY_QUALITY_BYTES_CHECKS_CSV",
    "PRIMARY_QUALITY_BYTES_DUPLICATE_COLUMN_DIFFERENCES_CSV",
    "PRIMARY_QUALITY_BYTES_DUPLICATE_MULTIPLICITY_CSV",
    "PRIMARY_QUALITY_BYTES_DUPLICATE_SUMMARY_CSV",
    "PRIMARY_QUALITY_BYTES_FLAG_DOMAINS_CSV",
    "PRIMARY_QUALITY_BYTES_INPUT_RECEIPT_JSON",
    "PRIMARY_QUALITY_BYTES_SUMMARY_JSON",
    "PRIMARY_QUALITY_BYTES__SUCCESS_JSON",
    "PRIMARY_QUALITY_FILE_SET",
    "PRIMARY_QUALITY_SCHEMAS_EXACT",
    "PRIMARY_TABLE_POPULATIONS_EXACT",
    "RAW_DAILY_SHA256_RECOMPUTED",
    "RAW_DAILY_STABLE_ALL_PASSES",
    "RAW_LOCKED_REGRESSION_COUNTS",
    "RAW_MONTHLY_SHA256_RECOMPUTED",
    "RAW_MONTHLY_STABLE_ALL_PASSES",
    "UPSTREAM_CALENDAR_IDENTITY",
    "UPSTREAM_CALENDAR_POINTER_SHA256",
    "UPSTREAM_CONTRACT_IDENTITY",
    "UPSTREAM_CONTRACT_POINTER_SHA256",
)
PRIMARY_QUALITY_FILENAMES = (
    "_SUCCESS.json",
    "canonicalization_rules.json",
    "checks.csv",
    "duplicate_column_differences.csv",
    "duplicate_multiplicity.csv",
    "duplicate_summary.csv",
    "flag_domains.csv",
    "input_receipt.json",
    "summary.json",
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
    "daily_full_file_rows_read",
    "daily_retained_rows",
    "difference_count",
    "fail_count",
    "monthly_full_file_rows_read",
    "monthly_retained_rows",
    "pass_count",
)
INDEPENDENCE_KEYS = (
    "full_raw_hashes_recomputed",
    "grouping_depends_on_input_sort",
    "primary_builder_imported",
    "raw_files_reread",
)
DIFFERENCE_COLUMNS = (
    "check_id",
    "category",
    "expected",
    "actual",
    "evidence",
    "message",
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
_VERIFICATION_ID_DOMAIN = (
    b"phase02-crsp-development-profile-independent-verification-v1\x00"
)
_IMPLEMENTATION_BUNDLE_DOMAIN = (
    b"phase02-crsp-development-profile-verifier-implementation-bundle-v1\x00"
)


class VerificationPublicationError(RuntimeError):
    """Raised when verification evidence cannot be published exactly."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the locked canonical JSON representation with one trailing LF."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise VerificationPublicationError(
            f"Value is not canonical-JSON serializable: {exc}"
        ) from exc
    return (rendered + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    except OSError as exc:
        raise VerificationPublicationError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _lexical_absolute(path: Path) -> Path:
    """Normalize an absolute path lexically without dereferencing symlinks."""

    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path, label: str) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise VerificationPublicationError(
                f"Cannot authenticate {label} path component {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise VerificationPublicationError(
                f"Symlink is prohibited in {label} path: {current}"
            )


def _read_authenticated_regular_file(path: Path, label: str) -> bytes:
    """Read a regular file through a no-follow descriptor bound to its path."""

    path = _lexical_absolute(path)
    _assert_no_symlink_components(path, label)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise VerificationPublicationError(f"Cannot open {label} {path}: {exc}") from exc
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = path.lstat()
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or descriptor_stat.st_dev != path_stat.st_dev
            or descriptor_stat.st_ino != path_stat.st_ino
        ):
            raise VerificationPublicationError(
                f"{label} path identity differs from its opened file: {path}"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        final_stat = path.lstat()
        if (
            not stat.S_ISREG(final_stat.st_mode)
            or final_stat.st_dev != descriptor_stat.st_dev
            or final_stat.st_ino != descriptor_stat.st_ino
            or final_stat.st_size != descriptor_stat.st_size
            or final_stat.st_mtime_ns != descriptor_stat.st_mtime_ns
        ):
            raise VerificationPublicationError(
                f"{label} changed while it was being authenticated: {path}"
            )
        return b"".join(chunks)
    except OSError as exc:
        raise VerificationPublicationError(f"Cannot read {label} {path}: {exc}") from exc
    finally:
        os.close(descriptor)


def _normalize_implementation_paths(
    implementation_paths: Mapping[str, Path] | Sequence[Path],
    project_root: Path,
) -> dict[str, Path]:
    project_root = _lexical_absolute(project_root)
    _assert_no_symlink_components(project_root, "project root")
    declared_items: list[tuple[str | None, Path]] = []
    if isinstance(implementation_paths, Mapping):
        for declared_label, path in implementation_paths.items():
            if not isinstance(declared_label, str):
                raise VerificationPublicationError(
                    "Implementation component labels must be strings"
                )
            declared_items.append((declared_label, Path(path)))
    elif isinstance(implementation_paths, Sequence) and not isinstance(
        implementation_paths, (str, bytes, bytearray)
    ):
        declared_items.extend((None, Path(path)) for path in implementation_paths)
    else:
        raise VerificationPublicationError(
            "Implementation paths must be a mapping or a path sequence"
        )

    normalized: dict[str, Path] = {}
    for declared_label, raw_path in declared_items:
        path = _lexical_absolute(raw_path)
        _assert_no_symlink_components(path, "verifier implementation component")
        try:
            derived_label = path.relative_to(project_root).as_posix()
        except ValueError as exc:
            raise VerificationPublicationError(
                f"Verifier implementation component is outside project root: {path}"
            ) from exc
        if declared_label is not None and declared_label != derived_label:
            raise VerificationPublicationError(
                "Declared implementation label differs from its project-relative path"
            )
        if derived_label in normalized:
            raise VerificationPublicationError(
                f"Duplicate verifier implementation label: {derived_label}"
            )
        normalized[derived_label] = path
    if set(normalized) != set(IMPLEMENTATION_COMPONENT_LABELS):
        raise VerificationPublicationError(
            "Verifier implementation component set differs: "
            f"expected {sorted(IMPLEMENTATION_COMPONENT_LABELS)}, "
            f"actual {sorted(normalized)}"
        )
    return {label: normalized[label] for label in sorted(normalized)}


def implementation_bundle_sha256(
    implementation_paths: Mapping[str, Path] | Sequence[Path],
    *,
    project_root: Path,
) -> str:
    """Hash the exact four-file verifier implementation with stable framing."""

    normalized = _normalize_implementation_paths(
        implementation_paths, _lexical_absolute(project_root)
    )
    digest = hashlib.sha256()
    digest.update(_IMPLEMENTATION_BUNDLE_DOMAIN)
    digest.update(len(normalized).to_bytes(8, "big", signed=False))
    for label in sorted(normalized):
        label_bytes = label.encode("utf-8")
        payload = _read_authenticated_regular_file(
            normalized[label], f"verifier implementation component {label}"
        )
        digest.update(len(label_bytes).to_bytes(8, "big", signed=False))
        digest.update(label_bytes)
        digest.update(len(payload).to_bytes(8, "big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def _assert_safe_component(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise VerificationPublicationError(f"{label} must be a string")
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise VerificationPublicationError(
            f"Unsafe {label} path component: {value!r}"
        )
    return value


def _assert_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise VerificationPublicationError(
            f"{label} must be a lowercase 64-hex SHA-256 digest"
        )
    return value


def _assert_exact_keys(value: Any, expected: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerificationPublicationError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise VerificationPublicationError(f"{label} keys must be strings")
    actual = set(value)
    required = set(expected)
    if actual != required:
        raise VerificationPublicationError(
            f"{label} keys differ: expected {sorted(required)}, actual {sorted(actual)}"
        )
    return value


def _assert_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise VerificationPublicationError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _canonical_json_cell(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")[:-1]


def render_differences_csv(report: Mapping[str, Any]) -> bytes:
    """Render the exact differences CSV from failing checks in a report.

    The function validates check row shape and ordering sufficiently to make the
    output deterministic.  Full report validation occurs before publication.
    """

    checks = report.get("checks")
    if not isinstance(checks, list):
        raise VerificationPublicationError("report checks must be an array")
    output = io.StringIO(newline="")
    writer = csv.writer(
        output,
        delimiter=",",
        quotechar='"',
        doublequote=True,
        escapechar=None,
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writerow(DIFFERENCE_COLUMNS)
    for index, check in enumerate(checks):
        row = _assert_exact_keys(check, CHECK_KEYS, f"check[{index}]")
        if row["status"] != "FAIL":
            continue
        writer.writerow(
            (
                row["check_id"],
                row["category"],
                _canonical_json_cell(row["expected"]),
                _canonical_json_cell(row["actual"]),
                row["evidence"],
                row["message"],
            )
        )
    return output.getvalue().encode("utf-8")


def derive_verification_id(
    report: Mapping[str, Any], differences_payload: bytes
) -> str:
    """Derive the content identity while excluding the self-referential ID."""

    if not isinstance(report, Mapping):
        raise VerificationPublicationError("report must be an object")
    if not isinstance(differences_payload, bytes):
        raise VerificationPublicationError("differences payload must be bytes")
    identity_report = dict(report)
    identity_report.pop("verification_id", None)
    digest = hashlib.sha256()
    digest.update(_VERIFICATION_ID_DOMAIN)
    digest.update(canonical_json_bytes(identity_report))
    digest.update(b"differences.csv\x00")
    digest.update(differences_payload)
    return f"verify-{digest.hexdigest()[:16]}"


def bind_verification_id(
    report: Mapping[str, Any], differences_payload: bytes
) -> dict[str, Any]:
    """Return a copy of a report with its deterministic verification ID bound."""

    bound = dict(report)
    bound.pop("verification_id", None)
    bound["verification_id"] = derive_verification_id(bound, differences_payload)
    return bound


def _validate_report_and_payload(
    report: Mapping[str, Any], differences_payload: bytes
) -> tuple[dict[str, Any], bytes]:
    report_object = _assert_exact_keys(report, REPORT_KEYS, "independent report")
    if not isinstance(differences_payload, bytes):
        raise VerificationPublicationError("differences payload must be bytes")

    if (
        type(report_object["schema_version"]) is not int
        or report_object["schema_version"] != SCHEMA_VERSION
    ):
        raise VerificationPublicationError("independent report schema_version must be 1")
    if report_object["profile_name"] != PROFILE_NAME:
        raise VerificationPublicationError("independent report profile_name differs")
    _assert_safe_component(report_object["build_id"], "build_id")
    _assert_safe_component(report_object["contract_id"], "contract_id")
    _assert_sha256(report_object["primary_manifest_sha256"], "primary manifest hash")
    _assert_sha256(
        report_object["verifier_implementation_sha256"],
        "verifier implementation hash",
    )

    checks = report_object["checks"]
    if not isinstance(checks, list) or not checks:
        raise VerificationPublicationError("independent report checks must be nonempty")
    check_ids: list[str] = []
    pass_count = 0
    fail_count = 0
    for index, check in enumerate(checks):
        row = _assert_exact_keys(check, CHECK_KEYS, f"check[{index}]")
        for key in ("category", "check_id", "evidence", "message"):
            if not isinstance(row[key], str):
                raise VerificationPublicationError(
                    f"check[{index}].{key} must be a string"
                )
        if not row["check_id"]:
            raise VerificationPublicationError(f"check[{index}].check_id is blank")
        # Compare canonical JSON bytes, not Python's loose equality: JSON boolean
        # true must never be accepted as the integer 1 (nor false as 0).
        expected_bytes = canonical_json_bytes(row["expected"])
        actual_bytes = canonical_json_bytes(row["actual"])
        truthful_status = "PASS" if actual_bytes == expected_bytes else "FAIL"
        if row["status"] != truthful_status:
            raise VerificationPublicationError(
                f"check[{index}] status is false: actual/expected equality requires "
                f"{truthful_status}"
            )
        if row["status"] == "PASS":
            pass_count += 1
        elif row["status"] == "FAIL":
            fail_count += 1
        else:
            raise VerificationPublicationError(
                f"check[{index}].status must be PASS or FAIL"
            )
        check_ids.append(row["check_id"])
    if check_ids != sorted(check_ids) or len(set(check_ids)) != len(check_ids):
        raise VerificationPublicationError(
            "independent report checks must have ascending unique check_id values"
        )
    if tuple(check_ids) != BASELINE_CHECK_IDS:
        raise VerificationPublicationError(
            "independent report check vocabulary differs from the exact 40-check baseline"
        )

    summary = _assert_exact_keys(
        report_object["summary"], SUMMARY_KEYS, "independent report summary"
    )
    for key in SUMMARY_KEYS:
        _assert_int(summary[key], f"summary.{key}")
    expected_summary_counts = {
        "check_count": len(checks),
        "difference_count": fail_count,
        "fail_count": fail_count,
        "pass_count": pass_count,
    }
    for key, expected in expected_summary_counts.items():
        if summary[key] != expected:
            raise VerificationPublicationError(
                f"summary.{key} differs: expected {expected}, actual {summary[key]}"
            )

    independence = _assert_exact_keys(
        report_object["independence"],
        INDEPENDENCE_KEYS,
        "independent report independence",
    )
    required_independence = {
        "full_raw_hashes_recomputed": True,
        "grouping_depends_on_input_sort": False,
        "primary_builder_imported": False,
        "raw_files_reread": 2,
    }
    for key, expected in required_independence.items():
        actual = independence[key]
        if type(actual) is not type(expected) or actual != expected:
            raise VerificationPublicationError(
                f"independence.{key} differs: expected {expected!r}, actual {actual!r}"
            )

    expected_status = PASS_STATUS if fail_count == 0 else FAIL_STATUS
    if report_object["status"] != expected_status:
        raise VerificationPublicationError(
            f"independent report status differs: expected {expected_status!r}"
        )
    expected_differences = render_differences_csv(report_object)
    if differences_payload != expected_differences:
        raise VerificationPublicationError(
            "differences payload differs from the exact failing-check projection"
        )
    if fail_count == 0 and _sha256_bytes(differences_payload) != EMPTY_DIFFERENCES_SHA256:
        raise VerificationPublicationError(
            "Header-only differences payload differs from its locked SHA-256"
        )

    expected_verification_id = derive_verification_id(
        report_object, differences_payload
    )
    verification_id = report_object["verification_id"]
    if (
        not isinstance(verification_id, str)
        or _VERIFICATION_ID_RE.fullmatch(verification_id) is None
        or verification_id != expected_verification_id
    ):
        raise VerificationPublicationError(
            "verification_id differs from the deterministic report identity"
        )

    # Freeze the validated semantic report into plain canonical JSON data.
    report_bytes = canonical_json_bytes(report_object)
    frozen_report = json.loads(report_bytes.decode("utf-8"))
    return frozen_report, report_bytes


def _assert_optional_regular_file(path: Path, label: str) -> None:
    _assert_no_symlink_components(path, label)
    if path.exists() and not path.is_file():
        raise VerificationPublicationError(f"Existing {label} is not a regular file")


def _read_closed_directory(directory: Path, label: str) -> dict[str, bytes]:
    _assert_no_symlink_components(directory, label)
    if not directory.is_dir():
        raise VerificationPublicationError(f"Existing {label} is not a real directory")
    payloads: dict[str, bytes] = {}
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.is_symlink() or not entry.is_file():
            raise VerificationPublicationError(
                f"Existing {label} contains a non-regular entry: {entry.name}"
            )
        payloads[entry.name] = entry.read_bytes()
    return payloads


def _assert_existing_matches(
    final: Path, payloads: Mapping[str, bytes], label: str
) -> bool:
    _assert_no_symlink_components(final, label)
    if not final.exists() and not final.is_symlink():
        return False
    observed = _read_closed_directory(final, label)
    if observed != dict(payloads):
        expected_hashes = {
            name: _sha256_bytes(payload) for name, payload in payloads.items()
        }
        observed_hashes = {
            name: _sha256_bytes(payload) for name, payload in observed.items()
        }
        raise VerificationPublicationError(
            f"Existing {label} differs from deterministic payload: "
            f"expected {expected_hashes}, actual {observed_hashes}"
        )
    return True


def _remove_staging_entry(stage: Path) -> None:
    try:
        mode = stage.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        stage.unlink(missing_ok=True)
    else:
        shutil.rmtree(stage, ignore_errors=True)


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise VerificationPublicationError("Short write while staging evidence")
        remaining = remaining[written:]


def _read_anchored_payloads(directory_descriptor: int) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for name in sorted(os.listdir(directory_descriptor)):
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise VerificationPublicationError(
                    f"Staged entry is not a regular file: {name}"
                )
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1 << 20):
                chunks.append(chunk)
            payloads[name] = b"".join(chunks)
        finally:
            os.close(descriptor)
    return payloads


def _stage_payloads(parent: Path, prefix: str, payloads: Mapping[str, bytes]) -> Path:
    _assert_no_symlink_components(parent, "verification staging parent")
    parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(parent, "verification staging parent")
    stage = Path(tempfile.mkdtemp(prefix=f".{prefix}.staging-", dir=parent))
    directory_descriptor: int | None = None
    try:
        _assert_no_symlink_components(stage, "verification staging directory")
        directory_descriptor = os.open(
            stage,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        path_metadata = stage.lstat()
        descriptor_metadata = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(descriptor_metadata.st_mode)
            or path_metadata.st_dev != descriptor_metadata.st_dev
            or path_metadata.st_ino != descriptor_metadata.st_ino
        ):
            raise VerificationPublicationError(
                "Verification staging path identity changed before writes"
            )
        for name, payload in payloads.items():
            if name not in VERIFICATION_FILES:
                raise VerificationPublicationError(
                    f"Unexpected verification payload filename: {name!r}"
                )
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_descriptor,
            )
            try:
                _write_all(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        if _read_anchored_payloads(directory_descriptor) != dict(payloads):
            raise VerificationPublicationError(
                "Staged verification bytes differ immediately after writing"
            )
        final_metadata = stage.lstat()
        if (
            not stat.S_ISDIR(final_metadata.st_mode)
            or final_metadata.st_dev != descriptor_metadata.st_dev
            or final_metadata.st_ino != descriptor_metadata.st_ino
        ):
            raise VerificationPublicationError(
                "Verification staging path identity changed during writes"
            )
    except Exception:
        _remove_staging_entry(stage)
        raise
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    return stage


def _publish_staged_directory(
    stage: Path, final: Path, payloads: Mapping[str, bytes]
) -> None:
    if _assert_existing_matches(final, payloads, "verification bundle"):
        _remove_staging_entry(stage)
        return
    _assert_no_symlink_components(final.parent, "verification bundle parent")
    final.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(final, "verification bundle")
    try:
        os.rename(stage, final)
    except OSError:
        if _assert_existing_matches(final, payloads, "verification bundle"):
            _remove_staging_entry(stage)
            return
        raise
    if _read_closed_directory(final, "verification bundle") != dict(payloads):
        raise VerificationPublicationError(
            "Published verification bytes changed after atomic rename"
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
        _assert_no_symlink_components(path, "verified pointer")
        if pre_replace_gate is not None:
            pre_replace_gate()
        _assert_no_symlink_components(path, "verified pointer")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _project_relative_path(path: Path, project_root: Path) -> str:
    try:
        relative = path.relative_to(project_root)
    except ValueError as exc:
        raise VerificationPublicationError(
            "Verification destination is outside the configured project root"
        ) from exc
    rendered = relative.as_posix()
    pure = PurePosixPath(rendered)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in rendered
    ):
        raise VerificationPublicationError(
            "Verification path is not a safe project-root-relative POSIX path"
        )
    return rendered


def _normalize_snapshot_mapping(
    snapshot: Mapping[str | Path, str],
) -> dict[Path, str]:
    if not isinstance(snapshot, Mapping):
        raise VerificationPublicationError("Primary snapshot must be a path/hash mapping")
    normalized: dict[Path, str] = {}
    for raw_path, raw_digest in snapshot.items():
        if not isinstance(raw_path, (str, os.PathLike)):
            raise VerificationPublicationError(
                "Primary snapshot keys must be filesystem paths"
            )
        path = _lexical_absolute(Path(raw_path))
        if path in normalized:
            raise VerificationPublicationError(
                f"Primary snapshot repeats a normalized path: {path}"
            )
        normalized[path] = _assert_sha256(
            raw_digest, f"primary snapshot digest for {path}"
        )
    return dict(sorted(normalized.items(), key=lambda item: item[0].as_posix()))


def _assert_primary_snapshot_hashes(snapshot: Mapping[Path, str]) -> None:
    observed: dict[Path, str] = {}
    for path in sorted(snapshot, key=lambda item: item.as_posix()):
        payload = _read_authenticated_regular_file(path, "primary snapshot file")
        observed[path] = _sha256_bytes(payload)
    if observed != dict(snapshot):
        changed = [
            path.as_posix()
            for path in sorted(snapshot, key=lambda item: item.as_posix())
            if observed.get(path) != snapshot[path]
        ]
        raise VerificationPublicationError(
            f"Primary snapshot changed during verification: {changed}"
        )


def _final_context_paths(project_root: Path) -> dict[str, Path]:
    root = _lexical_absolute(project_root)
    return {
        label: _lexical_absolute(root / PurePosixPath(label))
        for label in FINAL_CONTEXT_LABELS
    }


def capture_final_context_snapshot(*, project_root: Path) -> dict[str, str]:
    """Capture hashes for the exact six small final-context dependencies."""

    paths = _final_context_paths(project_root)
    return {
        path.as_posix(): _sha256_bytes(
            _read_authenticated_regular_file(path, f"final context {label}")
        )
        for label, path in sorted(paths.items())
    }


def _validate_final_context_snapshot(
    snapshot: Mapping[str | Path, str], *, project_root: Path
) -> dict[Path, str]:
    normalized = _normalize_snapshot_mapping(snapshot)
    expected = set(_final_context_paths(project_root).values())
    if set(normalized) != expected:
        raise VerificationPublicationError(
            "Final context snapshot path set differs from the exact six-file set"
        )
    _assert_final_context_snapshot_unchanged(normalized)
    return normalized


def _assert_final_context_snapshot_unchanged(snapshot: Mapping[Path, str]) -> None:
    observed = {
        path: _sha256_bytes(
            _read_authenticated_regular_file(path, "final context dependency")
        )
        for path in sorted(snapshot, key=lambda item: item.as_posix())
    }
    if observed != dict(snapshot):
        changed = [
            path.as_posix()
            for path in sorted(snapshot, key=lambda item: item.as_posix())
            if observed.get(path) != snapshot[path]
        ]
        raise VerificationPublicationError(
            f"Final context snapshot changed during verification: {changed}"
        )


def _raw_source_paths(database_root: Path) -> dict[str, Path]:
    root = _lexical_absolute(database_root)
    return {
        label: _lexical_absolute(root / PurePosixPath(label))
        for label in RAW_SOURCE_RELATIVE_PATHS
    }


def _raw_operational_stat(path: Path) -> dict[str, int]:
    path = _lexical_absolute(path)
    _assert_no_symlink_components(path, "raw operational source")
    try:
        value = path.lstat()
    except OSError as exc:
        raise VerificationPublicationError(
            f"Cannot lstat raw operational source {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(value.st_mode):
        raise VerificationPublicationError(
            f"Raw operational source is not a regular file: {path}"
        )
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "mtime_ns": int(value.st_mtime_ns),
        "size_bytes": int(value.st_size),
    }


def capture_raw_operational_snapshot(
    *, database_root: Path
) -> dict[str, dict[str, int]]:
    """Capture metadata only for both exact raw CRSP sources; read no raw bytes."""

    return {
        path.as_posix(): _raw_operational_stat(path)
        for _, path in sorted(_raw_source_paths(database_root).items())
    }


def _validate_raw_operational_snapshot(
    snapshot: Mapping[str | Path, Mapping[str, int]], *, database_root: Path
) -> dict[Path, dict[str, int]]:
    if not isinstance(snapshot, Mapping):
        raise VerificationPublicationError(
            "Raw operational snapshot must be a path/stat mapping"
        )
    normalized: dict[Path, dict[str, int]] = {}
    for raw_path, raw_stat in snapshot.items():
        if not isinstance(raw_path, (str, os.PathLike)):
            raise VerificationPublicationError(
                "Raw operational snapshot keys must be filesystem paths"
            )
        path = _lexical_absolute(Path(raw_path))
        if path in normalized:
            raise VerificationPublicationError(
                f"Raw operational snapshot repeats a normalized path: {path}"
            )
        row = _assert_exact_keys(
            raw_stat, RAW_OPERATIONAL_STAT_KEYS, f"raw operational stat {path}"
        )
        normalized[path] = {
            key: _assert_int(row[key], f"raw operational stat {path}.{key}")
            for key in RAW_OPERATIONAL_STAT_KEYS
        }
        if not stat.S_ISREG(normalized[path]["mode"]):
            raise VerificationPublicationError(
                f"Raw operational snapshot mode is not regular: {path}"
            )
    expected = set(_raw_source_paths(database_root).values())
    if set(normalized) != expected:
        raise VerificationPublicationError(
            "Raw operational snapshot path set differs from the exact two-file set"
        )
    _assert_raw_operational_snapshot_unchanged(normalized)
    return dict(sorted(normalized.items(), key=lambda item: item[0].as_posix()))


def _assert_raw_operational_snapshot_unchanged(
    snapshot: Mapping[Path, Mapping[str, int]],
) -> None:
    observed = {
        path: _raw_operational_stat(path)
        for path in sorted(snapshot, key=lambda item: item.as_posix())
    }
    if observed != dict(snapshot):
        changed = [
            path.as_posix()
            for path in sorted(snapshot, key=lambda item: item.as_posix())
            if observed.get(path) != snapshot[path]
        ]
        raise VerificationPublicationError(
            f"Raw operational snapshot changed during verification: {changed}"
        )


def _validate_primary_snapshot(
    snapshot: Mapping[str | Path, str],
    *,
    artifact_root: Path,
    database_root: Path,
    report: Mapping[str, Any],
    primary_manifest_path: Path,
) -> dict[Path, str]:
    """Authenticate the exact pointer, artifact, and nine quality-file set."""

    normalized = _normalize_snapshot_mapping(snapshot)
    database_root = _lexical_absolute(database_root)
    _assert_no_symlink_components(database_root, "database root")
    build_id = report["build_id"]
    contract_id = report["contract_id"]
    primary_pointer_path = _lexical_absolute(artifact_root / PRIMARY_POINTER_NAME)
    inventory_path = _lexical_absolute(primary_manifest_path.parent / "output_hashes.sha256")
    fixed_paths = {primary_pointer_path, primary_manifest_path, inventory_path}
    missing_fixed = sorted(
        (path.as_posix() for path in fixed_paths if path not in normalized)
    )
    if missing_fixed:
        raise VerificationPublicationError(
            f"Primary snapshot omits fixed dependencies: {missing_fixed}"
        )
    if normalized[primary_manifest_path] != report["primary_manifest_sha256"]:
        raise VerificationPublicationError(
            "Primary snapshot manifest digest differs from the report binding"
        )

    # Authenticate the supplied start snapshot before using its manifest paths.
    _assert_primary_snapshot_hashes(normalized)
    manifest_payload = _read_authenticated_regular_file(
        primary_manifest_path, "primary manifest"
    )
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationPublicationError(
            f"Primary manifest is not valid UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise VerificationPublicationError("Primary manifest is not an object")
    for key, expected in (
        ("build_id", build_id),
        ("contract_id", contract_id),
        ("profile_name", PROFILE_NAME),
    ):
        if manifest.get(key) != expected:
            raise VerificationPublicationError(
                f"Primary manifest {key} differs from the independent report"
            )

    quality_prefix = PurePosixPath(
        "quality",
        "phase_02_canonical_panel",
        contract_id,
        "crsp_development_profile",
        build_id,
    )
    expected_labels = [
        (quality_prefix / name).as_posix()
        for name in sorted(PRIMARY_QUALITY_FILENAMES)
    ]
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(expected_labels):
        raise VerificationPublicationError(
            "Primary manifest does not list exactly nine quality outputs"
        )
    observed_labels: list[str] = []
    output_by_label: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(outputs):
        entry = _assert_exact_keys(
            entry, ("path", "sha256", "size_bytes"), f"manifest output[{index}]"
        )
        label = entry["path"]
        if not isinstance(label, str):
            raise VerificationPublicationError(
                f"manifest output[{index}].path must be a string"
            )
        _assert_sha256(entry["sha256"], f"manifest output[{index}].sha256")
        _assert_int(entry["size_bytes"], f"manifest output[{index}].size_bytes")
        observed_labels.append(label)
        output_by_label[label] = entry
    if observed_labels != expected_labels or len(output_by_label) != len(expected_labels):
        raise VerificationPublicationError(
            "Primary manifest quality paths differ from the exact locked layout"
        )

    quality_paths = {
        _lexical_absolute(database_root / PurePosixPath(label))
        for label in expected_labels
    }
    expected_snapshot_paths = fixed_paths | quality_paths
    if set(normalized) != expected_snapshot_paths:
        raise VerificationPublicationError(
            "Primary snapshot path set differs from the exact 12-file dependency set"
        )
    for label in expected_labels:
        path = _lexical_absolute(database_root / PurePosixPath(label))
        entry = output_by_label[label]
        if normalized[path] != entry["sha256"]:
            raise VerificationPublicationError(
                f"Primary snapshot hash differs from manifest output {label}"
            )
        if path.stat().st_size != entry["size_bytes"]:
            raise VerificationPublicationError(
                f"Primary snapshot size differs from manifest output {label}"
            )

    expected_inventory = "".join(
        f"{output_by_label[label]['sha256']}  {label}\n" for label in expected_labels
    ).encode("utf-8")
    if _read_authenticated_regular_file(
        inventory_path, "primary output hash inventory"
    ) != expected_inventory:
        raise VerificationPublicationError(
            "Primary output hash inventory differs from the manifest outputs"
        )
    pointer_payload = _read_authenticated_regular_file(
        primary_pointer_path, "primary pointer"
    )
    try:
        primary_pointer = json.loads(pointer_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationPublicationError(
            f"Primary pointer is not valid UTF-8 JSON: {exc}"
        ) from exc
    expected_pointer_binding = {
        "build_id": build_id,
        "contract_id": contract_id,
        "manifest_sha256": report["primary_manifest_sha256"],
    }
    if not isinstance(primary_pointer, dict) or any(
        primary_pointer.get(key) != value
        for key, value in expected_pointer_binding.items()
    ):
        raise VerificationPublicationError(
            "Primary pointer does not bind the independently verified manifest"
        )
    return normalized


def _assert_paths(
    artifact_root: Path,
    project_root: Path,
    database_root: Path,
) -> tuple[Path, Path, Path]:
    artifact_root = _lexical_absolute(artifact_root)
    project_root = _lexical_absolute(project_root)
    database_root = _lexical_absolute(database_root)
    _assert_no_symlink_components(project_root, "project root")
    _assert_no_symlink_components(artifact_root, "artifact root")
    _assert_no_symlink_components(database_root, "database root")
    try:
        artifact_root.resolve(strict=False).relative_to(
            project_root.resolve(strict=False)
        )
    except ValueError as exc:
        raise VerificationPublicationError(
            "Artifact root is outside the configured project root"
        ) from exc
    return artifact_root, project_root, database_root


def _reauthenticate_for_pointer(
    report: Mapping[str, Any],
    differences_payload: bytes,
    report_bytes: bytes,
    bundle_directory: Path,
    bundle_payloads: Mapping[str, bytes],
    primary_snapshot: Mapping[Path, str],
    final_context_snapshot: Mapping[Path, str],
    raw_operational_snapshot: Mapping[Path, Mapping[str, int]],
    verifier_implementation_paths: Mapping[str, Path],
    project_root: Path,
    pointer_path: Path,
) -> None:
    """Recheck every byte dependency immediately before pointer mutation."""

    _, rerendered_report = _validate_report_and_payload(
        report, differences_payload
    )
    if rerendered_report != report_bytes:
        raise VerificationPublicationError(
            "Independent report bytes changed before pointer commit"
        )
    if _read_closed_directory(bundle_directory, "verification bundle") != dict(
        bundle_payloads
    ):
        raise VerificationPublicationError(
            "Verification bundle changed before pointer commit"
        )
    _assert_primary_snapshot_hashes(primary_snapshot)
    _assert_final_context_snapshot_unchanged(final_context_snapshot)
    _assert_raw_operational_snapshot_unchanged(raw_operational_snapshot)
    if (
        implementation_bundle_sha256(
            verifier_implementation_paths, project_root=project_root
        )
        != report["verifier_implementation_sha256"]
    ):
        raise VerificationPublicationError(
            "Verifier implementation bundle changed before verified-pointer commit"
        )
    _assert_optional_regular_file(pointer_path, "verified pointer")


def publish_independent_verification(
    report: Mapping[str, Any],
    differences_payload: bytes,
    *,
    artifact_root: Path,
    project_root: Path,
    database_root: Path,
    verifier_implementation_paths: Mapping[str, Path] | Sequence[Path],
    primary_snapshot: Mapping[str | Path, str],
    final_context_snapshot: Mapping[str | Path, str],
    raw_operational_snapshot: Mapping[
        str | Path, Mapping[str, int]
    ],
    primary_manifest_path: Path | None = None,
) -> dict[str, Path]:
    """Publish an exact evidence bundle and, on pass only, its verified pointer.

    Failure evidence uses the same deterministic immutable directory but never
    creates, replaces, or otherwise reads the contents of the verified pointer.
    """

    frozen_report, report_bytes = _validate_report_and_payload(
        report, differences_payload
    )
    artifact_root, project_root, database_root = _assert_paths(
        artifact_root, project_root, database_root
    )
    normalized_final_context_snapshot = _validate_final_context_snapshot(
        final_context_snapshot, project_root=project_root
    )
    normalized_raw_operational_snapshot = _validate_raw_operational_snapshot(
        raw_operational_snapshot, database_root=database_root
    )
    normalized_implementation_paths = _normalize_implementation_paths(
        verifier_implementation_paths, project_root
    )
    if (
        implementation_bundle_sha256(
            normalized_implementation_paths, project_root=project_root
        )
        != frozen_report["verifier_implementation_sha256"]
    ):
        raise VerificationPublicationError(
            "Verifier implementation bundle differs from the report binding"
        )

    build_id = _assert_safe_component(frozen_report["build_id"], "build_id")
    verification_id = _assert_safe_component(
        frozen_report["verification_id"], "verification_id"
    )
    bundle_directory = _lexical_absolute(
        artifact_root
        / VERIFICATION_SUBDIRECTORY
        / build_id
        / verification_id
    )
    expected_bundle_directory = _lexical_absolute(
        artifact_root
        / "crsp_development_profile_verifications"
        / build_id
        / verification_id
    )
    if bundle_directory != expected_bundle_directory:
        raise VerificationPublicationError(
            "Verification bundle selected an unexpected directory"
        )
    _assert_no_symlink_components(bundle_directory, "verification bundle")
    pointer_path = _lexical_absolute(artifact_root / VERIFIED_POINTER_NAME)
    _assert_optional_regular_file(pointer_path, "verified pointer")
    if primary_manifest_path is None:
        primary_manifest_path = (
            artifact_root
            / PRIMARY_BUILD_SUBDIRECTORY
            / build_id
            / "manifest.json"
        )
    primary_manifest_path = _lexical_absolute(primary_manifest_path)
    expected_primary_manifest_path = _lexical_absolute(
        artifact_root
        / PRIMARY_BUILD_SUBDIRECTORY
        / build_id
        / "manifest.json"
    )
    if primary_manifest_path != expected_primary_manifest_path:
        raise VerificationPublicationError(
            "Primary manifest path differs from the exact bound build artifact"
        )
    _assert_no_symlink_components(primary_manifest_path, "primary manifest")
    normalized_primary_snapshot = _validate_primary_snapshot(
        primary_snapshot,
        artifact_root=artifact_root,
        database_root=database_root,
        report=frozen_report,
        primary_manifest_path=primary_manifest_path,
    )

    bundle_payloads = {
        "differences.csv": differences_payload,
        "independent_check.json": report_bytes,
    }
    if tuple(sorted(bundle_payloads)) != tuple(sorted(VERIFICATION_FILES)):
        raise VerificationPublicationError("Verification bundle file set differs")
    _assert_existing_matches(
        bundle_directory, bundle_payloads, "verification bundle"
    )
    stage: Path | None = None
    try:
        stage = _stage_payloads(
            bundle_directory.parent, verification_id, bundle_payloads
        )
        _publish_staged_directory(stage, bundle_directory, bundle_payloads)
        stage = None
    finally:
        if stage is not None:
            _remove_staging_entry(stage)

    result = {"bundle": bundle_directory}
    if frozen_report["status"] != PASS_STATUS:
        _assert_primary_snapshot_hashes(normalized_primary_snapshot)
        _assert_final_context_snapshot_unchanged(normalized_final_context_snapshot)
        _assert_raw_operational_snapshot_unchanged(
            normalized_raw_operational_snapshot
        )
        if implementation_bundle_sha256(
            normalized_implementation_paths, project_root=project_root
        ) != frozen_report["verifier_implementation_sha256"]:
            raise VerificationPublicationError(
                "Verifier implementation bundle changed before failure-evidence completion"
            )
        # Deliberately return without reading, creating, or replacing the pointer.
        return result

    _reauthenticate_for_pointer(
        frozen_report,
        differences_payload,
        report_bytes,
        bundle_directory,
        bundle_payloads,
        normalized_primary_snapshot,
        normalized_final_context_snapshot,
        normalized_raw_operational_snapshot,
        normalized_implementation_paths,
        project_root,
        pointer_path,
    )
    verification_path = _project_relative_path(bundle_directory, project_root)
    pointer = {
        "build_id": build_id,
        "contract_id": frozen_report["contract_id"],
        "differences_sha256": _sha256_bytes(differences_payload),
        "independent_check_sha256": _sha256_bytes(report_bytes),
        "primary_manifest_sha256": frozen_report["primary_manifest_sha256"],
        "schema_version": SCHEMA_VERSION,
        "status": PASS_STATUS,
        "verification_id": verification_id,
        "verification_path": verification_path,
    }
    if tuple(sorted(pointer)) != tuple(sorted(VERIFIED_POINTER_KEYS)):
        raise VerificationPublicationError("Verified pointer key set differs")
    pointer_bytes = canonical_json_bytes(pointer)

    def final_commit_gate() -> None:
        """Run after pointer-temp fsync and immediately before atomic replace."""

        observed = _read_closed_directory(bundle_directory, "verification bundle")
        if observed != bundle_payloads:
            raise VerificationPublicationError(
                "Verification bundle changed at final pointer boundary"
            )
        if (
            _sha256_bytes(observed["differences.csv"])
            != pointer["differences_sha256"]
        ):
            raise VerificationPublicationError("Final differences hash differs")
        if (
            _sha256_bytes(observed["independent_check.json"])
            != pointer["independent_check_sha256"]
        ):
            raise VerificationPublicationError("Final independent-check hash differs")
        _assert_primary_snapshot_hashes(normalized_primary_snapshot)
        _assert_final_context_snapshot_unchanged(normalized_final_context_snapshot)
        _assert_raw_operational_snapshot_unchanged(
            normalized_raw_operational_snapshot
        )
        if (
            implementation_bundle_sha256(
                normalized_implementation_paths, project_root=project_root
            )
            != frozen_report["verifier_implementation_sha256"]
        ):
            raise VerificationPublicationError(
                "Verifier implementation bundle changed at the final pointer boundary"
            )
        _assert_optional_regular_file(pointer_path, "verified pointer")

    _assert_optional_regular_file(pointer_path, "verified pointer")
    pointer_already_exact = (
        pointer_path.exists() and pointer_path.read_bytes() == pointer_bytes
    )
    if pointer_already_exact:
        final_commit_gate()
    else:
        _write_atomic_file(
            pointer_path,
            pointer_bytes,
            pre_replace_gate=final_commit_gate,
        )
    if pointer_path.read_bytes() != pointer_bytes:
        raise VerificationPublicationError(
            "Verified pointer bytes differ immediately after commit"
        )
    result["pointer"] = pointer_path
    return result


__all__ = [
    "BASELINE_CHECK_IDS",
    "EMPTY_DIFFERENCES_SHA256",
    "FAIL_STATUS",
    "FINAL_CONTEXT_LABELS",
    "IMPLEMENTATION_COMPONENT_LABELS",
    "PASS_STATUS",
    "PRIMARY_POINTER_NAME",
    "PRIMARY_QUALITY_FILENAMES",
    "RAW_OPERATIONAL_STAT_KEYS",
    "RAW_SOURCE_RELATIVE_PATHS",
    "VERIFICATION_FILES",
    "VERIFICATION_SUBDIRECTORY",
    "VERIFIED_POINTER_NAME",
    "VerificationPublicationError",
    "bind_verification_id",
    "canonical_json_bytes",
    "capture_final_context_snapshot",
    "capture_raw_operational_snapshot",
    "derive_verification_id",
    "implementation_bundle_sha256",
    "publish_independent_verification",
    "render_differences_csv",
]
