"""Standalone verifier for the Phase 2 CRSP development profile.

The module intentionally never imports the primary profile compiler or its
publisher.  It authenticates the published primary bundle, asks the separate
independent scanner to reconstruct every aggregate from the two raw CSVs, and
compares independently rendered output bytes with the sealed primary files.
Only aggregate evidence is allowed into the verification report.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from portfolio_bot.data.phase_02_crsp_profile_verification_publication import (
    bind_verification_id,
    publish_independent_verification,
    render_differences_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_CONFIG = (
    PROJECT_ROOT / "configs" / "phase_02_canonicalization" / "crsp_development_v1.toml"
)
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.toml"
DEFAULT_PHASE_CONFIG = PROJECT_ROOT / "configs" / "phase_02_canonical_panel.toml"
DEFAULT_PRIMARY_POINTER = (
    PROJECT_ROOT
    / "artifacts"
    / "phase_02_canonical_panel"
    / "LATEST_CRSP_DEVELOPMENT_PROFILE_PRIMARY.json"
)
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel"
VERIFIER_IMPLEMENTATION_PATH = Path(__file__).resolve()
PRIMARY_IMPLEMENTATION_PATH = (
    PROJECT_ROOT / "src" / "portfolio_bot" / "data" / "phase_02_crsp_development_profile.py"
)

LOCKED_PROFILE_CONFIG_SHA256 = (
    "645b6ff51480d4e3cbcad313e3488c5a8d45df272ca4af6e7010e870296b4c35"
)
LOCKED_PHASE_CONFIG_SHA256 = (
    "fb727b10375d5eaf5a8e72629934daeb2f5790c7b09a46287df08c972db00adb"
)
BYTE_CHUNK_SIZE = 8 << 20
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

DIFFERENCE_FIELDS = (
    "check_id",
    "category",
    "expected",
    "actual",
    "evidence",
    "message",
)


class ProfileVerificationError(RuntimeError):
    """Raised when independent verification cannot proceed safely."""


@dataclasses.dataclass(frozen=True)
class VerificationSettings:
    project_root: Path
    profile_config_path: Path
    paths_config_path: Path
    phase_config_path: Path
    primary_pointer_path: Path
    artifact_root: Path
    database_root: Path
    raw_root: Path
    quality_root: Path
    scratch_root: Path
    config: dict[str, Any]
    paths: dict[str, Any]
    phase: dict[str, Any]
    block_size_bytes: int = 16 << 20
    enforce_locked_config_hashes: bool = True


def canonical_json_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (canonical_json_text(value) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = BYTE_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_stable_regular_bytes(path: Path, label: str) -> bytes:
    """Read one small dependency through a no-follow, path-bound descriptor."""

    path = _lexical_absolute(path)
    _assert_no_symlink_components(path, label)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ProfileVerificationError(f"Cannot open {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or opened.st_dev != named.st_dev
            or opened.st_ino != named.st_ino
        ):
            raise ProfileVerificationError(f"{label.capitalize()} path identity differs")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        final_opened = os.fstat(descriptor)
        final_named = path.lstat()
        identity_fields = ("st_dev", "st_ino", "st_mode", "st_mtime_ns", "st_size")
        if any(
            getattr(final_opened, field) != getattr(opened, field)
            or getattr(final_named, field) != getattr(opened, field)
            for field in identity_fields
        ):
            raise ProfileVerificationError(f"{label.capitalize()} changed while read")
        return b"".join(chunks)
    except OSError as exc:
        raise ProfileVerificationError(f"Cannot read {label}: {path}") from exc
    finally:
        os.close(descriptor)


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return _lexical_absolute(path if path.is_absolute() else project_root / path)


def _project_label(path: Path, project_root: Path) -> str:
    lexical = _lexical_absolute(path)
    try:
        return lexical.relative_to(_lexical_absolute(project_root)).as_posix()
    except ValueError:
        return lexical.as_posix()


def _assert_no_symlink_components(path: Path, label: str) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ProfileVerificationError(f"Symlink is prohibited in {label}: {current}")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileVerificationError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProfileVerificationError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            return fields, list(reader)
    except (OSError, csv.Error) as exc:
        raise ProfileVerificationError(f"Cannot read CSV {path}: {exc}") from exc


def _wire(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return canonical_json_text(value)
    return str(value)


def _csv_bytes(fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(fields),
        delimiter=",",
        quotechar='"',
        doublequote=True,
        escapechar=None,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _wire(row.get(field)) for field in fields})
    return output.getvalue().encode("utf-8")


def load_settings(
    profile_config_path: Path = DEFAULT_PROFILE_CONFIG,
    paths_config_path: Path = DEFAULT_PATHS_CONFIG,
    phase_config_path: Path = DEFAULT_PHASE_CONFIG,
    *,
    primary_pointer_path: Path = DEFAULT_PRIMARY_POINTER,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    project_root: Path = PROJECT_ROOT,
    database_root: Path | None = None,
    raw_root: Path | None = None,
    quality_root: Path | None = None,
    scratch_root: Path | None = None,
    block_size_bytes: int = 16 << 20,
    enforce_locked_config_hashes: bool = True,
) -> VerificationSettings:
    project_root = _lexical_absolute(project_root)
    profile_config_path = _lexical_absolute(profile_config_path)
    paths_config_path = _lexical_absolute(paths_config_path)
    phase_config_path = _lexical_absolute(phase_config_path)
    with profile_config_path.open("rb") as handle:
        config = tomllib.load(handle)
    with paths_config_path.open("rb") as handle:
        paths = tomllib.load(handle)
    with phase_config_path.open("rb") as handle:
        phase = tomllib.load(handle)
    configured_database = Path(paths["database"]["root"]).expanduser()
    database = _lexical_absolute(
        database_root
        or Path(os.environ.get("PORTFOLIO_DATABASE_ROOT", configured_database))
    )
    namespace = phase["storage"]["quality_namespace"]
    resolved_raw = _lexical_absolute(raw_root or database / paths["database"]["raw"])
    resolved_quality = _lexical_absolute(
        quality_root or database / paths["database"]["quality"] / namespace
    )
    resolved_scratch = _lexical_absolute(
        scratch_root
        or database
        / paths["database"].get("cache", "cache")
        / namespace
        / "crsp_development_profile_verifier"
    )
    return VerificationSettings(
        project_root=project_root,
        profile_config_path=profile_config_path,
        paths_config_path=paths_config_path,
        phase_config_path=phase_config_path,
        primary_pointer_path=_lexical_absolute(primary_pointer_path),
        artifact_root=_lexical_absolute(artifact_root),
        database_root=database,
        raw_root=resolved_raw,
        quality_root=resolved_quality,
        scratch_root=resolved_scratch,
        config=config,
        paths=paths,
        phase=phase,
        block_size_bytes=block_size_bytes,
        enforce_locked_config_hashes=enforce_locked_config_hashes,
    )


class _Checks:
    def __init__(self, project_root: Path, database_root: Path | None = None) -> None:
        self.project_root = project_root
        self.database_root = database_root
        self.rows: list[dict[str, Any]] = []

    def _evidence_label(self, path: Path) -> str:
        lexical = _lexical_absolute(path)
        try:
            return lexical.relative_to(_lexical_absolute(self.project_root)).as_posix()
        except ValueError:
            pass
        if self.database_root is not None:
            try:
                relative = lexical.relative_to(
                    _lexical_absolute(self.database_root)
                ).as_posix()
                return f"database/{relative}"
            except ValueError:
                pass
        # Evidence labels must be portable and must not disclose a host-specific
        # home or temporary-directory prefix.  The basename is sufficient for
        # an out-of-root diagnostic and cannot affect any data comparison.
        return f"external/{lexical.name}"

    def add(
        self,
        check_id: str,
        category: str,
        *,
        expected: Any,
        actual: Any,
        passed: bool,
        evidence: str | Path,
        message: str,
    ) -> None:
        evidence_text = (
            self._evidence_label(evidence)
            if isinstance(evidence, Path)
            else evidence
        )
        self.rows.append(
            {
                "actual": actual,
                "category": category,
                "check_id": check_id,
                "evidence": evidence_text,
                "expected": expected,
                "message": message,
                "status": "PASS" if passed else "FAIL",
            }
        )

    def sorted_rows(self) -> list[dict[str, Any]]:
        rows = sorted(self.rows, key=lambda row: row["check_id"])
        identifiers = [row["check_id"] for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ProfileVerificationError("Independent check identifiers are not unique")
        return rows


def _assert_exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    if set(value) != set(expected):
        raise ProfileVerificationError(
            f"{label} keys differ: expected {sorted(expected)}, actual {sorted(value)}"
        )


def _parse_hash_inventory(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    problems: list[str] = []
    if not path.is_file() or path.is_symlink():
        return values, ["missing or nonregular inventory"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, label = line.partition("  ")
        if not separator or SHA256_RE.fullmatch(digest) is None or not label:
            problems.append(f"malformed line {line_number}")
            continue
        if label in values:
            problems.append(f"duplicate path at line {line_number}")
            continue
        pure = PurePosixPath(label)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            problems.append(f"unsafe path at line {line_number}")
            continue
        values[label] = digest
    return values, problems


def operational_stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "mtime_ns": int(value.st_mtime_ns),
        "size_bytes": int(value.st_size),
    }


def _safe_project_relative(label: Any, project_root: Path, name: str) -> Path:
    if not isinstance(label, str):
        raise ProfileVerificationError(f"{name} must be a project-relative string")
    pure = PurePosixPath(label)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in label
    ):
        raise ProfileVerificationError(f"Unsafe {name}: {label!r}")
    result = _lexical_absolute(project_root / Path(*pure.parts))
    try:
        result.relative_to(_lexical_absolute(project_root))
    except ValueError as exc:
        raise ProfileVerificationError(f"{name} escapes the project root") from exc
    return result


def _safe_component(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise ProfileVerificationError(f"Unsafe {name}: {value!r}")
    return value


def _directory_file_names(directory: Path) -> list[str]:
    _assert_no_symlink_components(directory, "primary directory")
    if not directory.is_dir():
        raise ProfileVerificationError(f"Missing primary directory: {directory}")
    names: list[str] = []
    for entry in directory.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ProfileVerificationError(
                f"Primary directory contains a nonregular entry: {entry.name}"
            )
        names.append(entry.name)
    return sorted(names)


def _schema_type_ok(value: Any, logical_type: str) -> bool:
    if logical_type == "object":
        return isinstance(value, dict)
    if logical_type == "array":
        return isinstance(value, list)
    if logical_type == "int64":
        return isinstance(value, int) and not isinstance(value, bool)
    if logical_type == "bool":
        return isinstance(value, bool)
    if logical_type == "sha256":
        return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
    if logical_type in {
        "string",
        "project_root_relative_posix_path",
        "database_root_relative_posix_path",
    }:
        return isinstance(value, str)
    return True


def _json_schema_problems(
    value: Any, schema: Mapping[str, Any], *, label: str
) -> list[str]:
    problems: list[str] = []
    if not isinstance(value, dict):
        return [f"{label} is not an object"]
    keys = list(schema["top_level_keys"])
    if set(value) != set(keys):
        problems.append(
            f"{label} keys differ: expected {sorted(keys)}, actual {sorted(value)}"
        )
        return problems
    for key, logical_type in zip(keys, schema["top_level_types"]):
        if not _schema_type_ok(value[key], logical_type):
            problems.append(f"{label}.{key} is not {logical_type}")
    return problems


def _canonical_json_problems(path: Path, value: Mapping[str, Any]) -> list[str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return [f"cannot read bytes: {exc}"]
    return [] if payload == canonical_json_bytes(value) else ["noncanonical JSON bytes"]


def _snapshot_hashes(
    paths: Iterable[Path], *, label: str = "sealed dependency"
) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        _assert_no_symlink_components(path, f"{label} path")
        if not path.is_file():
            raise ProfileVerificationError(f"{label.capitalize()} file is missing: {path}")
        result[_lexical_absolute(path).as_posix()] = sha256_file(path)
    return dict(sorted(result.items()))


def assert_snapshot_unchanged(
    snapshot: Mapping[str, str], *, label: str
) -> None:
    observed = _snapshot_hashes(
        (Path(path) for path in snapshot), label=label
    )
    if observed != dict(snapshot):
        raise ProfileVerificationError(
            f"{label.capitalize()} changed during independent verification"
        )


def assert_primary_snapshot_unchanged(snapshot: Mapping[str, str]) -> None:
    assert_snapshot_unchanged(snapshot, label="primary pointer or sealed bundle")


def _authenticate_upstream_pointer(
    settings: VerificationSettings,
    checks: _Checks,
    source_name: str,
) -> dict[str, Any]:
    locked = settings.config["upstream"][source_name]
    pointer_path = _safe_project_relative(
        locked["verified_pointer"], settings.project_root, f"{source_name} verified pointer"
    )
    pointer_bytes = _read_stable_regular_bytes(
        pointer_path, f"{source_name} verified pointer"
    )
    actual_hash = sha256_bytes(pointer_bytes)
    checks.add(
        f"UPSTREAM_{source_name.upper()}_POINTER_SHA256",
        "upstream_authentication",
        expected=locked["verified_pointer_sha256"],
        actual=actual_hash,
        passed=actual_hash == locked["verified_pointer_sha256"],
        evidence=pointer_path,
        message=f"The locked {source_name} verified pointer bytes are unchanged.",
    )
    try:
        pointer = json.loads(pointer_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileVerificationError(
            f"Cannot decode {source_name} verified pointer"
        ) from exc
    if not isinstance(pointer, dict):
        raise ProfileVerificationError(
            f"{source_name.capitalize()} verified pointer is not an object"
        )
    expected_status = locked.get("required_status", "verified_passed")
    identifiers = {
        "status": expected_status,
        "verification_id": locked["verification_id"],
    }
    if source_name == "calendar":
        identifiers["build_id"] = locked["build_id"]
        identifiers["contract_id"] = settings.config["upstream"]["contract"]["contract_id"]
    else:
        identifiers["contract_id"] = locked["contract_id"]
    actual = {key: pointer.get(key) for key in identifiers}
    checks.add(
        f"UPSTREAM_{source_name.upper()}_IDENTITY",
        "upstream_authentication",
        expected=identifiers,
        actual=actual,
        passed=actual == identifiers,
        evidence=pointer_path,
        message=f"The independently verified {source_name} identity matches the profile lock.",
    )
    return {"path": pointer_path, "sha256": actual_hash, "pointer": pointer}


def authenticate_primary_bundle(
    settings: VerificationSettings, checks: _Checks
) -> dict[str, Any]:
    """Authenticate and parse the complete primary bundle before raw scanning."""

    for path, label in (
        (settings.profile_config_path, "profile config"),
        (settings.paths_config_path, "paths config"),
        (settings.phase_config_path, "phase config"),
        (settings.primary_pointer_path, "primary pointer"),
        (settings.artifact_root, "artifact root"),
        (settings.quality_root, "quality root"),
        (settings.raw_root, "raw root"),
    ):
        _assert_no_symlink_components(path, label)

    profile_bytes = _read_stable_regular_bytes(
        settings.profile_config_path, "profile config"
    )
    paths_bytes = _read_stable_regular_bytes(settings.paths_config_path, "paths config")
    phase_bytes = _read_stable_regular_bytes(settings.phase_config_path, "phase config")
    try:
        fresh_config = tomllib.loads(profile_bytes.decode("utf-8"))
        fresh_paths = tomllib.loads(paths_bytes.decode("utf-8"))
        fresh_phase = tomllib.loads(phase_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProfileVerificationError(
            "A verifier configuration is not stable UTF-8 TOML"
        ) from exc
    if (
        fresh_config != settings.config
        or fresh_paths != settings.paths
        or fresh_phase != settings.phase
    ):
        raise ProfileVerificationError(
            "In-memory verifier settings differ from freshly authenticated TOML bytes"
        )
    profile_hash = sha256_bytes(profile_bytes)
    paths_hash = sha256_bytes(paths_bytes)
    phase_hash = sha256_bytes(phase_bytes)
    expected_profile_hash = (
        LOCKED_PROFILE_CONFIG_SHA256
        if settings.enforce_locked_config_hashes
        else profile_hash
    )
    expected_phase_hash = (
        LOCKED_PHASE_CONFIG_SHA256 if settings.enforce_locked_config_hashes else phase_hash
    )
    checks.add(
        "CONFIG_PROFILE_SHA256",
        "configuration",
        expected=expected_profile_hash,
        actual=profile_hash,
        passed=profile_hash == expected_profile_hash,
        evidence=settings.profile_config_path,
        message="The verifier uses the exact locked profile configuration bytes.",
    )
    checks.add(
        "CONFIG_PHASE_SHA256",
        "configuration",
        expected=expected_phase_hash,
        actual=phase_hash,
        passed=phase_hash == expected_phase_hash,
        evidence=settings.phase_config_path,
        message="The verifier uses the exact locked Phase 2 configuration bytes.",
    )
    if profile_hash != expected_profile_hash or phase_hash != expected_phase_hash:
        raise ProfileVerificationError("A locked verifier configuration hash differs")

    contract = _authenticate_upstream_pointer(settings, checks, "contract")
    calendar = _authenticate_upstream_pointer(settings, checks, "calendar")

    pointer_path = settings.primary_pointer_path
    pointer = _read_json_object(pointer_path)
    pointer_schema = settings.config["outputs"]["schemas"]["primary_pointer"]
    pointer_problems = _json_schema_problems(pointer, pointer_schema, label="primary pointer")
    pointer_problems.extend(_canonical_json_problems(pointer_path, pointer))
    checks.add(
        "PRIMARY_POINTER_SCHEMA_AND_SERIALIZATION",
        "primary_structure",
        expected=[],
        actual=pointer_problems,
        passed=not pointer_problems,
        evidence=pointer_path,
        message="The primary pointer has the exact locked schema and canonical bytes.",
    )
    if pointer_problems:
        raise ProfileVerificationError("Primary pointer structure is unsafe")
    build_id = _safe_component(pointer["build_id"], "primary build_id")
    contract_id = _safe_component(pointer["contract_id"], "primary contract_id")

    expected_pointer_literals = {
        "calendar_build_id": settings.config["upstream"]["calendar"]["build_id"],
        "contract_id": settings.config["upstream"]["contract"]["contract_id"],
        "schema_version": int(settings.config["schema_version"]),
        "status": settings.config["outputs"]["serialization"]["published_primary_status"],
    }
    actual_pointer_literals = {
        key: pointer.get(key) for key in expected_pointer_literals
    }
    checks.add(
        "PRIMARY_POINTER_BINDING",
        "primary_binding",
        expected=expected_pointer_literals,
        actual=actual_pointer_literals,
        passed=actual_pointer_literals == expected_pointer_literals,
        evidence=pointer_path,
        message="The primary pointer selects the locked profile inputs and pending status.",
    )

    artifact_directory = _safe_project_relative(
        pointer["artifact_path"], settings.project_root, "primary artifact path"
    )
    expected_artifact_directory = _lexical_absolute(
        settings.artifact_root
        / settings.config["outputs"]["artifact_subdirectory"]
        / build_id
    )
    checks.add(
        "PRIMARY_ARTIFACT_PATH_EXACT",
        "primary_structure",
        expected=_project_label(expected_artifact_directory, settings.project_root),
        actual=_project_label(artifact_directory, settings.project_root),
        passed=artifact_directory == expected_artifact_directory,
        evidence=pointer_path,
        message="The pointer names only the deterministic profile artifact directory.",
    )
    if artifact_directory != expected_artifact_directory:
        raise ProfileVerificationError("Primary artifact path differs from the locked layout")

    expected_artifact_files = sorted(
        settings.config["outputs"]["closed_sets"]["artifact_build_files"]
    )
    actual_artifact_files = _directory_file_names(artifact_directory)
    checks.add(
        "PRIMARY_ARTIFACT_FILE_SET",
        "primary_structure",
        expected=expected_artifact_files,
        actual=actual_artifact_files,
        passed=actual_artifact_files == expected_artifact_files,
        evidence=artifact_directory,
        message="The primary artifact directory is an exact two-file closed set.",
    )

    manifest_path = artifact_directory / "manifest.json"
    inventory_path = artifact_directory / "output_hashes.sha256"
    manifest = _read_json_object(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    manifest_problems = _json_schema_problems(
        manifest,
        settings.config["outputs"]["schemas"]["manifest"],
        label="primary manifest",
    )
    manifest_problems.extend(_canonical_json_problems(manifest_path, manifest))
    checks.add(
        "PRIMARY_MANIFEST_SCHEMA_AND_SERIALIZATION",
        "primary_structure",
        expected=[],
        actual=manifest_problems,
        passed=not manifest_problems,
        evidence=manifest_path,
        message="The primary manifest has the exact locked schema and canonical bytes.",
    )
    checks.add(
        "PRIMARY_MANIFEST_POINTER_HASH",
        "primary_integrity",
        expected=pointer["manifest_sha256"],
        actual=manifest_hash,
        passed=manifest_hash == pointer["manifest_sha256"],
        evidence=manifest_path,
        message="The primary manifest bytes match the pointer SHA-256.",
    )
    if manifest_problems:
        raise ProfileVerificationError("Primary manifest structure is unsafe")

    quality_directory = _lexical_absolute(
        settings.quality_root
        / contract_id
        / settings.config["outputs"]["quality_subdirectory"]
        / build_id
    )
    expected_quality_files = sorted(
        settings.config["outputs"]["closed_quality_file_set"]
    )
    actual_quality_files = _directory_file_names(quality_directory)
    checks.add(
        "PRIMARY_QUALITY_FILE_SET",
        "primary_structure",
        expected=expected_quality_files,
        actual=actual_quality_files,
        passed=actual_quality_files == expected_quality_files,
        evidence=quality_directory,
        message="The primary quality directory is the exact nine-file closed set.",
    )

    inventory, inventory_problems = _parse_hash_inventory(inventory_path)
    expected_inventory_paths = [
        (path.relative_to(settings.database_root)).as_posix()
        for path in (quality_directory / name for name in expected_quality_files)
    ]
    if list(inventory) != sorted(inventory):
        inventory_problems.append("inventory rows are not path-sorted")
    if sorted(inventory) != sorted(expected_inventory_paths):
        inventory_problems.append("inventory membership differs from the quality closed set")
    checks.add(
        "PRIMARY_OUTPUT_HASH_INVENTORY_SCHEMA",
        "primary_structure",
        expected=[],
        actual=inventory_problems,
        passed=not inventory_problems,
        evidence=inventory_path,
        message="The primary output inventory has exact paths, order, and digest syntax.",
    )

    quality_payloads = {
        name: (quality_directory / name).read_bytes() for name in expected_quality_files
    }
    observed_hashes = {
        (quality_directory / name).relative_to(settings.database_root).as_posix():
        sha256_bytes(quality_payloads[name])
        for name in expected_quality_files
    }
    checks.add(
        "PRIMARY_OUTPUT_HASHES_MATCH_FILES",
        "primary_integrity",
        expected=inventory,
        actual=observed_hashes,
        passed=observed_hashes == inventory,
        evidence=inventory_path,
        message="Every primary quality output matches the sealed SHA-256 inventory.",
    )
    expected_manifest_outputs = [
        {
            "path": label,
            "sha256": observed_hashes[label],
            "size_bytes": len(quality_payloads[Path(label).name]),
        }
        for label in sorted(observed_hashes)
    ]
    checks.add(
        "PRIMARY_MANIFEST_OUTPUTS_EXACT",
        "primary_integrity",
        expected=expected_manifest_outputs,
        actual=manifest.get("outputs"),
        passed=manifest.get("outputs") == expected_manifest_outputs,
        evidence=manifest_path,
        message="Manifest output paths, sizes, order, and hashes exactly describe the quality files.",
    )

    identity_expected = {
        "build_id": pointer["build_id"],
        "calendar_build_id": pointer["calendar_build_id"],
        "contract_id": pointer["contract_id"],
        "profile_name": settings.config["profile_name"],
        "schema_version": int(settings.config["schema_version"]),
        "status": settings.config["outputs"]["serialization"]["published_primary_status"],
    }
    identity_actual = {key: manifest.get(key) for key in identity_expected}
    checks.add(
        "PRIMARY_MANIFEST_IDENTITY",
        "primary_binding",
        expected=identity_expected,
        actual=identity_actual,
        passed=identity_actual == identity_expected,
        evidence=manifest_path,
        message="The manifest identity is consistent with the pointer and locked profile.",
    )

    json_names = {
        "_SUCCESS.json": "summary",
        "canonicalization_rules.json": "canonicalization_rules",
        "input_receipt.json": "input_receipt",
        "summary.json": "summary",
    }
    parsed_json: dict[str, dict[str, Any]] = {}
    schema_problems: list[str] = []
    for name, schema_name in json_names.items():
        path = quality_directory / name
        value = _read_json_object(path)
        parsed_json[name] = value
        schema_problems.extend(
            _json_schema_problems(
                value,
                settings.config["outputs"]["schemas"][schema_name],
                label=name,
            )
        )
        schema_problems.extend(f"{name}: {item}" for item in _canonical_json_problems(path, value))
    csv_names = (
        "checks.csv",
        "duplicate_column_differences.csv",
        "duplicate_multiplicity.csv",
        "duplicate_summary.csv",
        "flag_domains.csv",
    )
    parsed_csv: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for name in csv_names:
        fields, rows = _read_csv(quality_directory / name)
        parsed_csv[name] = (fields, rows)
        schema_name = name.removesuffix(".csv")
        expected_fields = list(
            settings.config["outputs"]["schemas"][schema_name]["columns"]
        )
        if fields != expected_fields:
            schema_problems.append(f"{name} header differs")
    checks.add(
        "PRIMARY_QUALITY_SCHEMAS_EXACT",
        "primary_structure",
        expected=[],
        actual=schema_problems,
        passed=not schema_problems,
        evidence=quality_directory,
        message="All primary quality files have exact locked schemas and canonical JSON serialization.",
    )

    snapshot_paths = [
        pointer_path,
        manifest_path,
        inventory_path,
        *(quality_directory / name for name in expected_quality_files),
    ]
    snapshot = _snapshot_hashes(snapshot_paths, label="primary snapshot")
    final_context_paths = [
        settings.profile_config_path,
        settings.paths_config_path,
        settings.phase_config_path,
        contract["path"],
        calendar["path"],
        PRIMARY_IMPLEMENTATION_PATH,
    ]
    final_context_snapshot = _snapshot_hashes(
        final_context_paths, label="final context snapshot"
    )
    expected_context_hashes = {
        _lexical_absolute(settings.profile_config_path).as_posix(): profile_hash,
        _lexical_absolute(settings.paths_config_path).as_posix(): paths_hash,
        _lexical_absolute(settings.phase_config_path).as_posix(): phase_hash,
        _lexical_absolute(contract["path"]).as_posix(): contract["sha256"],
        _lexical_absolute(calendar["path"]).as_posix(): calendar["sha256"],
        _lexical_absolute(PRIMARY_IMPLEMENTATION_PATH).as_posix(): manifest.get(
            "binding", {}
        ).get("implementation_sha256"),
    }
    if final_context_snapshot != dict(sorted(expected_context_hashes.items())):
        raise ProfileVerificationError(
            "Final context snapshot differs from the authenticated primary binding"
        )
    return {
        "calendar": calendar,
        "contract": contract,
        "pointer": pointer,
        "pointer_path": pointer_path,
        "artifact_directory": artifact_directory,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_hash,
        "inventory": inventory,
        "inventory_path": inventory_path,
        "quality_directory": quality_directory,
        "quality_payloads": quality_payloads,
        "parsed_json": parsed_json,
        "parsed_csv": parsed_csv,
        "snapshot": snapshot,
        "final_context_snapshot": final_context_snapshot,
        "config_sha256": profile_hash,
        "phase_config_sha256": phase_hash,
    }


def _scan_mapping(scan: Any) -> Mapping[str, Any]:
    if isinstance(scan, Mapping):
        return scan
    if hasattr(scan, "to_dict"):
        value = scan.to_dict()
        if isinstance(value, Mapping):
            return value
    raise ProfileVerificationError("Independent scanner returned no aggregate mapping")


def _stage_map(scan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        "daily_raw_key": scan["daily"]["stage"],
        **scan["monthly"]["stages"],
    }


def _duplicate_summary_rows(
    config: Mapping[str, Any], scan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    stages = _stage_map(scan)
    fields = config["outputs"]["schemas"]["duplicate_summary"]["columns"]
    return [
        {field: stages[stage_name][field] for field in fields}
        for stage_name in config["outputs"]["population_registry"]["duplicate_stage_order"]
    ]


def _duplicate_multiplicity_rows(
    config: Mapping[str, Any], scan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    stages = _stage_map(scan)
    result: list[dict[str, Any]] = []
    for stage_name in config["outputs"]["population_registry"]["duplicate_stage_order"]:
        stage = stages[stage_name]
        for multiplicity, group_count in sorted(
            ((int(key), int(value)) for key, value in stage["histogram"].items()),
            key=lambda item: item[0],
        ):
            if group_count <= 0:
                continue
            result.append(
                {
                    "source": stage["source"],
                    "stage": stage_name,
                    "multiplicity": multiplicity,
                    "group_count": group_count,
                    "affected_rows": multiplicity * group_count,
                }
            )
    return result


def _duplicate_column_difference_rows(
    config: Mapping[str, Any], scan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    schema = config["outputs"]["schemas"]["duplicate_column_differences"]
    expected = config["expected"]["monthly_candidate_column_difference_groups"]
    observed = scan["monthly"]["candidate_diagnostics"]["column_difference_groups"]
    return [
        {
            "source": schema["source_literal"],
            "stage": schema["stage_literal"],
            "column": column,
            "groups_differing": int(observed[column]),
            "expected_groups_differing": int(expected.get(column, 0)),
            "status": (
                "PASS"
                if int(observed[column]) == int(expected.get(column, 0))
                else "FAIL"
            ),
        }
        for column in config["sources"]["monthly"]["header"]
    ]


def _flag_domain_rows(
    config: Mapping[str, Any], scan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source_name, stage in (
        ("daily", "daily_retained_raw_rows"),
        ("monthly", "monthly_retained_raw_rows"),
    ):
        source_scan = scan[source_name]
        filename = config["sources"][source_name]["filename"]
        hard_domains = config["expected"]["flag_domains"][source_name]
        for column in config["profiled_categorical_columns"][source_name]:
            policy = "hard_gate" if column in hard_domains else "discovery"
            expected = set(hard_domains.get(column, []))
            frequencies = source_scan["categorical_counts"][column]
            ordered = sorted(
                frequencies,
                key=lambda value: (value != "", value.encode("utf-8")),
            )
            for value in ordered:
                if value == "":
                    classification = "blank"
                elif policy == "discovery":
                    classification = "discovered_nonblank"
                elif value in expected:
                    classification = "expected_nonblank"
                else:
                    classification = "unexpected_nonblank"
                result.append(
                    {
                        "source": filename,
                        "stage": stage,
                        "column": column,
                        "raw_value": value,
                        "row_count": int(frequencies[value]),
                        "is_blank": value == "",
                        "domain_policy": policy,
                        "classification": classification,
                    }
                )
    return result


def _primary_check_row(
    check_id: str,
    *,
    expected: Any,
    actual: Any,
    evidence: str,
    message: str,
    severity: str = "hard",
    passed: bool | None = None,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "severity": severity,
        "status": "PASS" if (actual == expected if passed is None else passed) else "FAIL",
        "expected": canonical_json_text(expected),
        "actual": canonical_json_text(actual),
        "evidence": evidence,
        "message": message,
    }


def _primary_operational_stats(
    scan: Mapping[str, Any], source_name: str
) -> tuple[dict[str, int], dict[str, int]]:
    stats = scan["operational_stats"][source_name]
    before = stats.get("before_hash") or stats.get("before")
    after = (
        stats.get("final")
        or stats.get("after_sparse_replay")
        or stats.get("after")
    )
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise ProfileVerificationError(
            f"Independent scanner omitted {source_name} operational identity"
        )
    fields = ("device", "inode", "mode", "mtime_ns", "size_bytes")
    return (
        {field: int(before[field]) for field in fields},
        {field: int(after[field]) for field in fields},
    )


def _raw_operational_snapshot(
    settings: VerificationSettings, scan: Mapping[str, Any]
) -> dict[str, dict[str, int]]:
    """Bind both raw files at the scanner's final all-source barrier."""

    result: dict[str, dict[str, int]] = {}
    fields = ("device", "inode", "mode", "mtime_ns", "size_bytes")
    for source_name in ("daily", "monthly"):
        source_stats = scan["operational_stats"][source_name]
        final = source_stats.get("final")
        if not isinstance(final, Mapping):
            raise ProfileVerificationError(
                f"Independent scanner omitted the final {source_name} raw identity"
            )
        sealed = {field: int(final[field]) for field in fields}
        path = _lexical_absolute(
            settings.raw_root
            / settings.config["sources"][source_name]["filename"]
        )
        if operational_stat(path) != sealed:
            raise ProfileVerificationError(
                f"Authenticated {source_name} raw source changed after scanning"
            )
        result[path.as_posix()] = sealed
    return dict(sorted(result.items()))


def _build_primary_checks(
    settings: VerificationSettings,
    primary: Mapping[str, Any],
    scan: Mapping[str, Any],
    difference_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    config = settings.config
    calendar_pointer = primary["calendar"]["pointer"]
    contract_pointer = primary["contract"]["pointer"]
    checks = [
        _primary_check_row(
            "UPSTREAM_CALENDAR_VERIFIED",
            expected="verified_passed",
            actual=calendar_pointer.get("status"),
            evidence=_project_label(primary["calendar"]["path"], settings.project_root),
            message="The profile consumes the independently verified observed calendar.",
        ),
        _primary_check_row(
            "UPSTREAM_CONTRACT_VERIFIED",
            expected="verified_passed",
            actual=contract_pointer.get("status"),
            evidence=_project_label(primary["contract"]["path"], settings.project_root),
            message="The profile consumes the independently verified CRSP contract.",
        ),
    ]
    for source_name in ("daily", "monthly"):
        source_scan = scan[source_name]
        before, after = _primary_operational_stats(scan, source_name)
        checks.extend(
            (
                _primary_check_row(
                    f"HEADER_{source_name.upper()}_EXACT",
                    expected=list(config["sources"][source_name]["header"]),
                    actual=list(source_scan["header"]),
                    evidence=f"{source_name} Arrow schema",
                    message="All 36 decoded string columns match the locked header in order.",
                ),
                _primary_check_row(
                    f"RAW_{source_name.upper()}_IDENTITY_BEFORE",
                    expected=before,
                    actual=before,
                    evidence=config["sources"][source_name]["filename"],
                    message="The raw operational identity matched the frozen Phase 1 recheck before scanning.",
                ),
                _primary_check_row(
                    f"RAW_{source_name.upper()}_STABLE_DURING_SCAN",
                    expected=before,
                    actual=after,
                    evidence=config["sources"][source_name]["filename"],
                    message="The raw file did not change during the complete scan.",
                ),
                _primary_check_row(
                    f"SCOPE_{source_name.upper()}_NONDATE_EXCLUSION",
                    expected=0,
                    actual=source_scan["scope_gate"][
                        "excluded_non_date_values_compared_or_aggregated"
                    ],
                    evidence=f"{source_name} date-first scope receipt",
                    message="No out-of-scope non-date value entered a statistic, comparison, fingerprint, or output.",
                ),
            )
        )
        expected_domains = config["expected"]["flag_domains"][source_name]
        actual_domains = {
            column: sorted(
                (
                    value
                    for value in source_scan["categorical_counts"][column]
                    if value != ""
                ),
                key=lambda value: value.encode("utf-8"),
            )
            for column in expected_domains
        }
        checks.append(
            _primary_check_row(
                f"DOMAIN_{source_name.upper()}_HARD_GATES",
                expected={column: list(values) for column, values in expected_domains.items()},
                actual=actual_domains,
                evidence="flag_domains.csv source population",
                message="Every locked nonblank categorical domain is reproduced exactly.",
            )
        )

    daily = scan["daily"]
    monthly = scan["monthly"]
    expected_daily = config["expected"]["daily_raw_key"]
    expected_raw = config["expected"]["monthly_raw_key"]
    expected_pre = config["expected"][
        "monthly_canonical_key_before_raw_exact_collapse"
    ]
    expected_post = config["expected"][
        "monthly_canonical_key_after_raw_exact_collapse"
    ]
    daily_actual = {key: daily["stage"][key] for key in expected_daily}
    raw_stage = monthly["stages"]["monthly_raw_key"]
    pre_stage = monthly["stages"][
        "monthly_canonical_key_before_raw_exact_collapse"
    ]
    post_stage = monthly["stages"][
        "monthly_canonical_key_after_raw_exact_collapse"
    ]
    candidate = monthly["candidate_diagnostics"]
    raw_actual = {key: raw_stage[key] for key in expected_raw}
    pre_actual = {
        key: (
            candidate[key]
            if key == "raw_exact_duplicate_groups_embedded_in_effective_interval_candidates"
            else pre_stage[key]
        )
        for key in expected_pre
    }
    post_actual = {
        key: (
            candidate[key]
            if key.startswith("groups_with_") or key == "candidate_group_size"
            else post_stage[key]
        )
        for key in expected_post
    }
    checks.extend(
        (
            _primary_check_row(
                "REGRESSION_DAILY_RAW_KEY",
                expected=expected_daily,
                actual=daily_actual,
                evidence="daily typed-key accumulator",
                message="All locked daily duplicate metrics reproduce exactly.",
            ),
            _primary_check_row(
                "REGRESSION_MONTHLY_RAW_KEY",
                expected=expected_raw,
                actual=raw_actual,
                evidence="monthly typed four-part raw-key accumulator",
                message="All locked monthly raw-key duplicate metrics reproduce exactly.",
            ),
            _primary_check_row(
                "REGRESSION_MONTHLY_CANONICAL_PRECOLLAPSE",
                expected=expected_pre,
                actual=pre_actual,
                evidence="monthly precanonical typed-key accumulator",
                message="All locked precollapse canonical metrics reproduce exactly.",
            ),
            _primary_check_row(
                "REGRESSION_MONTHLY_CANONICAL_POSTCOLLAPSE",
                expected=expected_post,
                actual=post_actual,
                evidence="monthly postcollapse typed-key accumulator",
                message="All locked postcollapse canonical and interval-match metrics reproduce exactly.",
            ),
        )
    )
    row_conservation = {
        "daily": daily["stage"]["input_rows"] - daily["stage"]["surplus_rows"]
        == daily["stage"]["output_rows"],
        "monthly_raw": raw_stage["input_rows"] - raw_stage["surplus_rows"]
        == raw_stage["output_rows"],
        "monthly_pre": pre_stage["input_rows"] - pre_stage["surplus_rows"]
        == pre_stage["output_rows"],
        "monthly_post": post_stage["input_rows"] - post_stage["surplus_rows"]
        == post_stage["output_rows"],
    }
    expected_pattern = config["expected"]["monthly_candidate_diagnostics"]
    pattern_keys = (
        "selected_conditional_type",
        "selected_trading_status",
        "rejected_conditional_type",
        "rejected_trading_status",
        "groups_with_observed_pattern",
    )
    actual_pattern = {key: candidate[key] for key in pattern_keys}
    locked_pattern = {key: expected_pattern[key] for key in pattern_keys}
    multiplicity_health: dict[str, dict[str, bool]] = {}
    for stage_name, stage in _stage_map(scan).items():
        histogram = {int(key): int(value) for key, value in stage["histogram"].items()}
        digest = stage["multiplicity_fingerprint_sha256"]
        multiplicity_health[stage_name] = {
            "fingerprint_is_lowercase_sha256": (
                isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None
            ),
            "histogram_group_count_matches": sum(histogram.values())
            == stage["distinct_groups"],
            "histogram_row_count_matches": sum(
                multiplicity * count for multiplicity, count in histogram.items()
            )
            == stage["input_rows"],
            "maximum_group_size_matches": max(histogram, default=0)
            == stage["maximum_group_size"],
        }
    checks.extend(
        (
            _primary_check_row(
                "FULL_FILE_ROW_COUNTS",
                expected={
                    "daily": int(config["sources"]["daily"]["full_file_rows"]),
                    "monthly": int(config["sources"]["monthly"]["full_file_rows"]),
                },
                actual={
                    "daily": int(daily["full_file_rows_scanned"]),
                    "monthly": int(monthly["full_file_rows_scanned"]),
                },
                evidence="complete monolithic CSV scan receipts",
                message="Both full files were read completely while only retained rows fed profile statistics.",
            ),
            _primary_check_row(
                "INTERVAL_ENDPOINT_DIAGNOSTICS",
                expected={"daily_inverted": 0, "monthly_inverted": 0},
                actual={
                    "daily_inverted": daily["additional_diagnostics"][
                        "inverted_interval_rows"
                    ],
                    "monthly_inverted": monthly["additional_diagnostics"][
                        "inverted_interval_rows"
                    ],
                },
                evidence="strict retained-row interval parser",
                message="Every retained effective interval is valid and non-inverted.",
            ),
            _primary_check_row(
                "MONTHLY_MISSING_PRICE_DATE_DIAGNOSTICS",
                expected="diagnostic_count_reported",
                actual={
                    "inside_candidate_rows": candidate[
                        "missing_price_date_rows_inside_candidates"
                    ],
                    "overall_retained_rows": monthly["additional_diagnostics"][
                        "missing_price_date_rows_overall"
                    ],
                },
                evidence="retained monthly rows and post-collapse candidate groups",
                message="Blank monthly price dates are counted; only singleton keys may pass through with one.",
                severity="info",
                passed=True,
            ),
            _primary_check_row(
                "MONTHLY_POSTCOLLAPSE_CANDIDATE_SIZE_HISTOGRAM",
                expected={
                    str(expected_post["candidate_group_size"]): expected_post[
                        "effective_interval_candidate_groups"
                    ]
                },
                actual={
                    str(key): value
                    for key, value in candidate["candidate_size_histogram"].items()
                },
                evidence="post-collapse monthly candidate collector",
                message="Every effective-interval candidate group has the locked multiplicity.",
            ),
            _primary_check_row(
                "MULTIPLICITY_HISTOGRAM_MAXIMUM_AND_FINGERPRINTS",
                expected={
                    stage_name: {key: True for key in health}
                    for stage_name, health in multiplicity_health.items()
                },
                actual=multiplicity_health,
                evidence="duplicate_multiplicity.csv and duplicate_summary.csv",
                message="Ordered typed-key histograms, maxima, and domain-separated aggregate fingerprints are internally valid.",
            ),
            _primary_check_row(
                "ROW_CONSERVATION_ALL_STAGES",
                expected={key: True for key in row_conservation},
                actual=row_conservation,
                evidence="duplicate_summary.csv",
                message="Every exact-collapse and canonical-resolution row equation conserves rows.",
            ),
            _primary_check_row(
                "MONTHLY_CANDIDATE_PATTERN_DIAGNOSTIC",
                expected=locked_pattern,
                actual=actual_pattern,
                evidence="monthly selected and rejected interval diagnostics",
                message="The locked selected/rejected diagnostic pattern reproduces but never selects a row.",
            ),
            _primary_check_row(
                "MONTHLY_CANDIDATE_SAME_ROW_SELECTION",
                expected=0,
                actual=candidate["price_and_calendar_selected_different_rows"],
                evidence="monthly post-collapse candidate diagnostics",
                message="Price date and calendar date selected the same unique interval row.",
            ),
            _primary_check_row(
                "MONTHLY_COLUMN_DIFFERENCE_COMPLEMENT_ZERO",
                expected=0,
                actual=sum(row["status"] != "PASS" for row in difference_rows),
                evidence="duplicate_column_differences.csv",
                message="Only locked candidate columns differ, at their exact expected group counts.",
            ),
            _primary_check_row(
                "NO_ROW_LEVEL_KEY_METADATA_RETURNED",
                expected=False,
                actual=False,
                evidence="publishable scanner return objects",
                message="No raw security key or row-level duplicate metadata reaches immutable outputs.",
            ),
            _primary_check_row(
                "NO_DOWNSTREAM_ANALYTICS_BUILT",
                expected={
                    "features": False,
                    "targets": False,
                    "portfolios": False,
                    "pnl": False,
                },
                actual={
                    "features": False,
                    "targets": False,
                    "portfolios": False,
                    "pnl": False,
                },
                evidence="locked profile scope",
                message="This quality profile creates no features, targets, portfolios, P&L, or performance result.",
            ),
        )
    )
    return sorted(checks, key=lambda row: row["check_id"])


def reconstruct_primary_documents(
    settings: VerificationSettings,
    primary: Mapping[str, Any],
    scan_result: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Independently render all nine quality files and both artifact files."""

    scan = _scan_mapping(scan_result)
    config = settings.config
    pointer = primary["pointer"]
    duplicate_summary = _duplicate_summary_rows(config, scan)
    duplicate_multiplicity = _duplicate_multiplicity_rows(config, scan)
    duplicate_differences = _duplicate_column_difference_rows(config, scan)
    flag_domains = _flag_domain_rows(config, scan)
    primary_checks = _build_primary_checks(
        settings, primary, scan, duplicate_differences
    )
    primary_implementation_path = (
        settings.project_root
        / "src"
        / "portfolio_bot"
        / "data"
        / "phase_02_crsp_development_profile.py"
    )
    implementation_sha256 = sha256_file(primary_implementation_path)
    binding = {
        "calendar_build_id": config["upstream"]["calendar"]["build_id"],
        "calendar_pointer_sha256": config["upstream"]["calendar"][
            "verified_pointer_sha256"
        ],
        "calendar_verification_id": config["upstream"]["calendar"][
            "verification_id"
        ],
        "config_sha256": primary["config_sha256"],
        "contract_id": config["upstream"]["contract"]["contract_id"],
        "contract_pointer_sha256": config["upstream"]["contract"][
            "verified_pointer_sha256"
        ],
        "contract_verification_id": config["upstream"]["contract"][
            "verification_id"
        ],
        "daily_raw_sha256": scan["raw_hashes"]["daily"],
        "implementation_sha256": implementation_sha256,
        "monthly_raw_sha256": scan["raw_hashes"]["monthly"],
    }
    build_id = pointer["build_id"]
    status = config["outputs"]["serialization"]["published_primary_status"]
    daily = scan["daily"]
    monthly = scan["monthly"]
    summary = {
        "build_id": build_id,
        "check_count": len(primary_checks),
        "daily_full_file_rows_scanned": int(daily["full_file_rows_scanned"]),
        "daily_retained_rows": int(daily["retained_rows"]),
        "daily_unique_security_days": int(daily["stage"]["output_rows"]),
        "fail_count": sum(row["status"] != "PASS" for row in primary_checks),
        "features_built": False,
        "monthly_full_file_rows_scanned": int(monthly["full_file_rows_scanned"]),
        "monthly_retained_rows": int(monthly["retained_rows"]),
        "monthly_unique_security_months": int(
            monthly["stages"][
                "monthly_canonical_key_after_raw_exact_collapse"
            ]["output_rows"]
        ),
        "pass_count": sum(row["status"] == "PASS" for row in primary_checks),
        "performance_results_seen": False,
        "pnl_built": False,
        "portfolios_built": False,
        "profile_name": config["profile_name"],
        "schema_version": int(config["schema_version"]),
        "status": status,
        "targets_built": False,
    }
    canonicalization_rules = {
        "comparison": config["comparison"],
        "config_sha256": primary["config_sha256"],
        "key_and_date_semantics": config["key_and_date_semantics"],
        "ordering_and_fingerprints": config["ordering_and_fingerprints"],
        "profile_name": config["profile_name"],
        "profile_version": config["profile_version"],
        "rules": config["rules"],
        "scope": config["scope"],
        "sources": config["sources"],
    }
    raw_inputs: dict[str, Any] = {}
    for source_name in ("daily", "monthly"):
        before, after = _primary_operational_stats(scan, source_name)
        source = config["sources"][source_name]
        source_scan = scan[source_name]
        raw_inputs[source_name] = {
            "expected_full_file_rows": int(source["full_file_rows"]),
            "expected_sha256": source["sha256"],
            "expected_size_bytes": int(source["size_bytes"]),
            "filename": source["filename"],
            "full_file_rows_scanned": int(source_scan["full_file_rows_scanned"]),
            "operational_stat_after": after,
            "operational_stat_before": before,
            "retained_rows": int(source_scan["retained_rows"]),
        }
    input_receipt = {
        "build_id": build_id,
        "columns_read": {
            "daily": list(config["sources"]["daily"]["header"]),
            "monthly": list(config["sources"]["monthly"]["header"]),
        },
        "config_sha256": primary["config_sha256"],
        "implementation_sha256": implementation_sha256,
        "profile_name": config["profile_name"],
        "raw_inputs": raw_inputs,
        "schema_version": int(config["schema_version"]),
        "scope": {
            "end_date": config["scope"]["end_date"],
            "start_date": config["scope"]["start_date"],
        },
        "upstream": {
            "calendar_build_id": config["upstream"]["calendar"]["build_id"],
            "calendar_pointer_sha256": config["upstream"]["calendar"][
                "verified_pointer_sha256"
            ],
            "calendar_verification_id": config["upstream"]["calendar"][
                "verification_id"
            ],
            "contract_id": config["upstream"]["contract"]["contract_id"],
            "contract_pointer_sha256": config["upstream"]["contract"][
                "verified_pointer_sha256"
            ],
            "contract_verification_id": config["upstream"]["contract"][
                "verification_id"
            ],
        },
    }
    schemas = config["outputs"]["schemas"]
    quality_payloads = {
        "_SUCCESS.json": canonical_json_bytes(summary),
        "canonicalization_rules.json": canonical_json_bytes(canonicalization_rules),
        "checks.csv": _csv_bytes(schemas["checks"]["columns"], primary_checks),
        "duplicate_column_differences.csv": _csv_bytes(
            schemas["duplicate_column_differences"]["columns"],
            duplicate_differences,
        ),
        "duplicate_multiplicity.csv": _csv_bytes(
            schemas["duplicate_multiplicity"]["columns"], duplicate_multiplicity
        ),
        "duplicate_summary.csv": _csv_bytes(
            schemas["duplicate_summary"]["columns"], duplicate_summary
        ),
        "flag_domains.csv": _csv_bytes(
            schemas["flag_domains"]["columns"], flag_domains
        ),
        "input_receipt.json": canonical_json_bytes(input_receipt),
        "summary.json": canonical_json_bytes(summary),
    }
    quality_directory = primary["quality_directory"]
    output_entries = [
        {
            "path": (quality_directory / name)
            .relative_to(settings.database_root)
            .as_posix(),
            "sha256": sha256_bytes(quality_payloads[name]),
            "size_bytes": len(quality_payloads[name]),
        }
        for name in sorted(quality_payloads)
    ]
    manifest = {
        "binding": binding,
        "build_id": build_id,
        "calendar_build_id": binding["calendar_build_id"],
        "contract_id": binding["contract_id"],
        "outputs": output_entries,
        "profile_name": config["profile_name"],
        "schema_version": int(config["schema_version"]),
        "status": status,
        "summary": summary,
    }
    inventory_bytes = "".join(
        f"{entry['sha256']}  {entry['path']}\n" for entry in output_entries
    ).encode("utf-8")
    manifest_bytes = canonical_json_bytes(manifest)
    expected_pointer = {
        "artifact_path": primary["artifact_directory"]
        .relative_to(settings.project_root)
        .as_posix(),
        "build_id": build_id,
        "calendar_build_id": binding["calendar_build_id"],
        "contract_id": binding["contract_id"],
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "schema_version": int(config["schema_version"]),
        "status": status,
    }
    return {
        "binding": binding,
        "build_id": build_id,
        "canonicalization_rules": canonicalization_rules,
        "checks": primary_checks,
        "duplicate_column_differences": duplicate_differences,
        "duplicate_multiplicity": duplicate_multiplicity,
        "duplicate_summary": duplicate_summary,
        "flag_domains": flag_domains,
        "input_receipt": input_receipt,
        "summary": summary,
        "quality_payloads": quality_payloads,
        "manifest": manifest,
        "manifest_bytes": manifest_bytes,
        "output_hashes_bytes": inventory_bytes,
        "primary_pointer": expected_pointer,
        "primary_pointer_bytes": canonical_json_bytes(expected_pointer),
    }


def compare_reconstruction_to_primary(
    settings: VerificationSettings,
    primary: Mapping[str, Any],
    scan_result: Mapping[str, Any] | Any,
    reconstructed: Mapping[str, Any],
    checks: _Checks,
) -> None:
    scan = _scan_mapping(scan_result)
    config = settings.config
    for source_name in ("daily", "monthly"):
        expected_hash = config["sources"][source_name]["sha256"]
        actual_hash = scan["raw_hashes"][source_name]
        checks.add(
            f"RAW_{source_name.upper()}_SHA256_RECOMPUTED",
            "raw_authentication",
            expected=expected_hash,
            actual=actual_hash,
            passed=actual_hash == expected_hash,
            evidence=settings.raw_root / config["sources"][source_name]["filename"],
            message="The independent byte pass reproduced the locked whole-file SHA-256.",
        )
        before, after = _primary_operational_stats(scan, source_name)
        checks.add(
            f"RAW_{source_name.upper()}_STABLE_ALL_PASSES",
            "raw_authentication",
            expected=before,
            actual=after,
            passed=before == after,
            evidence=settings.raw_root / config["sources"][source_name]["filename"],
            message="The raw operational identity remained stable across hash, key, and sparse replay passes.",
        )

    monthly_candidate = scan["monthly"]["candidate_diagnostics"]
    pre_stage = scan["monthly"]["stages"][
        "monthly_canonical_key_before_raw_exact_collapse"
    ]
    post_stage = scan["monthly"]["stages"][
        "monthly_canonical_key_after_raw_exact_collapse"
    ]
    expected_pre = config["expected"][
        "monthly_canonical_key_before_raw_exact_collapse"
    ]
    expected_post = config["expected"][
        "monthly_canonical_key_after_raw_exact_collapse"
    ]
    locked_regressions = {
        "daily_raw_key": {
            key: scan["daily"]["stage"][key]
            for key in config["expected"]["daily_raw_key"]
        },
        "monthly_raw_key": {
            key: scan["monthly"]["stages"]["monthly_raw_key"][key]
            for key in config["expected"]["monthly_raw_key"]
        },
        "monthly_canonical_key_before_raw_exact_collapse": {
            key: (
                monthly_candidate[key]
                if key
                == "raw_exact_duplicate_groups_embedded_in_effective_interval_candidates"
                else pre_stage[key]
            )
            for key in expected_pre
        },
        "monthly_canonical_key_after_raw_exact_collapse": {
            key: (
                monthly_candidate[key]
                if key.startswith("groups_with_") or key == "candidate_group_size"
                else post_stage[key]
            )
            for key in expected_post
        },
    }
    expected_regressions = {
        "daily_raw_key": config["expected"]["daily_raw_key"],
        "monthly_raw_key": config["expected"]["monthly_raw_key"],
        "monthly_canonical_key_before_raw_exact_collapse": expected_pre,
        "monthly_canonical_key_after_raw_exact_collapse": expected_post,
    }
    checks.add(
        "RAW_LOCKED_REGRESSION_COUNTS",
        "independent_reconstruction",
        expected=expected_regressions,
        actual=locked_regressions,
        passed=locked_regressions == expected_regressions,
        evidence="independent range-partition grouping",
        message="Independent order-insensitive grouping reproduces all four locked duplicate-stage populations.",
    )
    fingerprint_actual = {
        stage_name: stage["multiplicity_fingerprint_sha256"]
        for stage_name, stage in _stage_map(scan).items()
    }
    primary_summary_rows = primary["parsed_csv"]["duplicate_summary.csv"][1]
    fingerprint_expected = {
        row["stage"]: row["multiplicity_fingerprint_sha256"]
        for row in primary_summary_rows
    }
    checks.add(
        "MULTIPLICITY_FINGERPRINTS_EXACT",
        "independent_reconstruction",
        expected=fingerprint_expected,
        actual=fingerprint_actual,
        passed=fingerprint_actual == fingerprint_expected,
        evidence="four independent globally typed-key-sorted SHA-256 accumulators",
        message="All four key-multiplicity fingerprints reproduce without input-order dependence.",
    )
    missing_actual = {
        "inside_candidate_rows": scan["monthly"]["candidate_diagnostics"][
            "missing_price_date_rows_inside_candidates"
        ],
        "overall_retained_rows": scan["monthly"]["additional_diagnostics"][
            "missing_price_date_rows_overall"
        ],
    }
    primary_check_rows = {
        row["check_id"]: row for row in primary["parsed_csv"]["checks.csv"][1]
    }
    try:
        missing_expected = json.loads(
            primary_check_rows["MONTHLY_MISSING_PRICE_DATE_DIAGNOSTICS"]["actual"]
        )
    except (KeyError, json.JSONDecodeError):
        missing_expected = None
    checks.add(
        "MONTHLY_MISSING_PRICE_DATE_EXACT",
        "independent_reconstruction",
        expected=missing_expected,
        actual=missing_actual,
        passed=missing_actual == missing_expected,
        evidence="independent retained-row and post-collapse candidate diagnostics",
        message="The verifier independently reproduces both missing MthPrcDt populations.",
    )

    expected_binding = reconstructed["binding"]
    actual_binding = primary["manifest"].get("binding")
    checks.add(
        "PRIMARY_BINDING_RECONSTRUCTED",
        "primary_binding",
        expected=expected_binding,
        actual=actual_binding,
        passed=actual_binding == expected_binding,
        evidence=primary["manifest_path"],
        message="The primary binding is independently reconstructed from current locked inputs and raw hashes.",
    )
    binding_build_id = f"crsp-profile-{sha256_bytes(canonical_json_bytes(expected_binding))[:16]}"
    checks.add(
        "PRIMARY_BUILD_ID_CONTENT_BOUND",
        "primary_binding",
        expected=binding_build_id,
        actual=primary["pointer"]["build_id"],
        passed=primary["pointer"]["build_id"] == binding_build_id,
        evidence=primary["pointer_path"],
        message="The primary build identifier matches its deterministic immutable binding identity.",
    )

    for name in sorted(reconstructed["quality_payloads"]):
        expected_payload = reconstructed["quality_payloads"][name]
        actual_payload = primary["quality_payloads"][name]
        checks.add(
            f"PRIMARY_QUALITY_BYTES_{name.upper().replace('.', '_')}",
            "primary_reproduction",
            expected={
                "sha256": sha256_bytes(expected_payload),
                "size_bytes": len(expected_payload),
            },
            actual={
                "sha256": sha256_bytes(actual_payload),
                "size_bytes": len(actual_payload),
            },
            passed=actual_payload == expected_payload,
            evidence=primary["quality_directory"] / name,
            message="Independent aggregate reconstruction reproduces the exact primary output bytes.",
        )
    artifact_payloads = {
        "manifest.json": reconstructed["manifest_bytes"],
        "output_hashes.sha256": reconstructed["output_hashes_bytes"],
    }
    for name, expected_payload in artifact_payloads.items():
        path = primary["artifact_directory"] / name
        actual_payload = path.read_bytes()
        checks.add(
            f"PRIMARY_ARTIFACT_BYTES_{name.upper().replace('.', '_')}",
            "primary_reproduction",
            expected={
                "sha256": sha256_bytes(expected_payload),
                "size_bytes": len(expected_payload),
            },
            actual={
                "sha256": sha256_bytes(actual_payload),
                "size_bytes": len(actual_payload),
            },
            passed=actual_payload == expected_payload,
            evidence=path,
            message="The independently rendered artifact bytes exactly reproduce the primary artifact.",
        )
    actual_pointer_payload = primary["pointer_path"].read_bytes()
    expected_pointer_payload = reconstructed["primary_pointer_bytes"]
    checks.add(
        "PRIMARY_POINTER_BYTES_RECONSTRUCTED",
        "primary_reproduction",
        expected=sha256_bytes(expected_pointer_payload),
        actual=sha256_bytes(actual_pointer_payload),
        passed=actual_pointer_payload == expected_pointer_payload,
        evidence=primary["pointer_path"],
        message="The independently rendered primary pointer exactly matches the published pointer.",
    )

    row_counts = {
        "checks": len(reconstructed["checks"]),
        "duplicate_column_differences": len(
            reconstructed["duplicate_column_differences"]
        ),
        "duplicate_multiplicity": len(reconstructed["duplicate_multiplicity"]),
        "duplicate_summary": len(reconstructed["duplicate_summary"]),
        "flag_domains": len(reconstructed["flag_domains"]),
    }
    expected_row_counts = {
        "checks": 27,
        "duplicate_column_differences": int(
            config["outputs"]["schemas"]["duplicate_column_differences"]["row_count"]
        ),
        "duplicate_multiplicity": 45,
        "duplicate_summary": int(
            config["outputs"]["schemas"]["duplicate_summary"]["row_count"]
        ),
        "flag_domains": 133,
    }
    checks.add(
        "PRIMARY_TABLE_POPULATIONS_EXACT",
        "primary_structure",
        expected=expected_row_counts,
        actual=row_counts,
        passed=row_counts == expected_row_counts,
        evidence=primary["quality_directory"],
        message="All reconstructed primary tables have the exact locked populations.",
    )


def verifier_implementation_paths(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    root = _lexical_absolute(project_root)
    labels = (
        "scripts/verify_phase_02_crsp_development_profile.py",
        "src/portfolio_bot/data/phase_02_crsp_profile_independent_scan.py",
        "src/portfolio_bot/data/phase_02_crsp_profile_verification_publication.py",
        "src/portfolio_bot/data/verify_phase_02_crsp_development_profile.py",
    )
    return {label: root / Path(*PurePosixPath(label).parts) for label in labels}


def build_independent_report(
    settings: VerificationSettings,
    primary: Mapping[str, Any],
    scan_result: Mapping[str, Any] | Any,
    checks: _Checks,
    *,
    verifier_implementation_sha256: str,
) -> dict[str, Any]:
    scan = _scan_mapping(scan_result)
    rows = checks.sorted_rows()
    pass_count = sum(row["status"] == "PASS" for row in rows)
    fail_count = len(rows) - pass_count
    status = (
        settings.config["outputs"]["serialization"]["published_verified_status"]
        if fail_count == 0
        else settings.config["outputs"]["serialization"]["failed_verification_status"]
    )
    scan_independence = scan.get("independence", {})
    independence = {
        "full_raw_hashes_recomputed": scan_independence.get(
            "full_raw_hashes_recomputed", False
        ),
        "grouping_depends_on_input_sort": scan_independence.get(
            "grouping_depends_on_input_sort", True
        ),
        "primary_builder_imported": scan_independence.get(
            "primary_builder_imported", True
        ),
        "raw_files_reread": scan_independence.get("raw_files_reread", 0),
    }
    report = {
        "build_id": primary["pointer"]["build_id"],
        "checks": rows,
        "contract_id": primary["pointer"]["contract_id"],
        "independence": independence,
        "primary_manifest_sha256": primary["manifest_sha256"],
        "profile_name": settings.config["profile_name"],
        "schema_version": int(settings.config["schema_version"]),
        "status": status,
        "summary": {
            "check_count": len(rows),
            "daily_full_file_rows_read": int(
                scan["daily"]["full_file_rows_scanned"]
            ),
            "daily_retained_rows": int(scan["daily"]["retained_rows"]),
            "difference_count": fail_count,
            "fail_count": fail_count,
            "monthly_full_file_rows_read": int(
                scan["monthly"]["full_file_rows_scanned"]
            ),
            "monthly_retained_rows": int(scan["monthly"]["retained_rows"]),
            "pass_count": pass_count,
        },
        "verification_id": "",
        "verifier_implementation_sha256": verifier_implementation_sha256,
    }
    differences = render_differences_csv(report)
    return bind_verification_id(report, differences)


def _scanner_call(settings: VerificationSettings, scratch_directory: Path) -> Mapping[str, Any]:
    # Local import is intentional: an AST audit can prove this verifier imports
    # neither primary module, while the independent scanner remains replaceable
    # and separately content-bound.
    from portfolio_bot.data.phase_02_crsp_profile_independent_scan import (
        scan_crsp_profile_independently,
    )

    raw_paths = {
        source_name: settings.raw_root
        / settings.config["sources"][source_name]["filename"]
        for source_name in ("daily", "monthly")
    }
    result = scan_crsp_profile_independently(
        settings.config,
        raw_paths,
        scratch_root=scratch_directory,
        block_size_bytes=settings.block_size_bytes,
    )
    return _scan_mapping(result)


def independently_verify_profile(
    settings: VerificationSettings | None = None,
    *,
    scan_result: Mapping[str, Any] | Any | None = None,
) -> dict[str, Any]:
    """Authenticate, scan, reconstruct, and return an uncommitted report bundle."""

    settings = settings or load_settings()
    checks = _Checks(settings.project_root, settings.database_root)
    primary = authenticate_primary_bundle(settings, checks)
    from portfolio_bot.data.phase_02_crsp_profile_verification_publication import (
        implementation_bundle_sha256,
    )

    implementation_paths = verifier_implementation_paths(settings.project_root)
    implementation_sha256_before = implementation_bundle_sha256(
        implementation_paths, project_root=settings.project_root
    )
    if scan_result is None:
        _assert_no_symlink_components(settings.scratch_root, "verifier scratch root")
        settings.scratch_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _assert_no_symlink_components(settings.scratch_root, "verifier scratch root")
        with tempfile.TemporaryDirectory(
            prefix="crsp-profile-verify-", dir=settings.scratch_root
        ) as temporary:
            scratch_directory = Path(temporary)
            scratch_directory.chmod(0o700)
            scan = _scanner_call(settings, scratch_directory)
    else:
        scan = _scan_mapping(scan_result)
        checks.add(
            "INDEPENDENCE_INJECTED_SCAN_RESULT",
            "independence",
            expected=False,
            actual=True,
            passed=False,
            evidence="in-memory test hook",
            message="Injected scan aggregates are prohibited from verified publication.",
        )
    reconstructed = reconstruct_primary_documents(settings, primary, scan)
    compare_reconstruction_to_primary(
        settings, primary, scan, reconstructed, checks
    )

    implementation_sha256_after = implementation_bundle_sha256(
        implementation_paths, project_root=settings.project_root
    )
    if implementation_sha256_after != implementation_sha256_before:
        raise ProfileVerificationError(
            "Verifier implementation bundle changed during independent scanning"
        )
    assert_snapshot_unchanged(
        primary["final_context_snapshot"], label="final context snapshot"
    )
    raw_operational_snapshot = _raw_operational_snapshot(settings, scan)
    report = build_independent_report(
        settings,
        primary,
        scan,
        checks,
        verifier_implementation_sha256=implementation_sha256_before,
    )
    differences = render_differences_csv(report)
    assert_primary_snapshot_unchanged(primary["snapshot"])
    return {
        "differences_payload": differences,
        "final_context_snapshot": primary["final_context_snapshot"],
        "implementation_paths": implementation_paths,
        "primary": primary,
        "raw_operational_snapshot": raw_operational_snapshot,
        "reconstructed": reconstructed,
        "report": report,
        "scan": scan,
    }


def verify_and_publish_profile(
    settings: VerificationSettings | None = None,
    *,
    scan_result: Mapping[str, Any] | Any | None = None,
) -> dict[str, Any]:
    if scan_result is not None:
        raise ProfileVerificationError(
            "Injected scan results are prohibited from verification publication"
        )
    settings = settings or load_settings()
    result = independently_verify_profile(settings)
    primary = result["primary"]
    published = publish_independent_verification(
        result["report"],
        result["differences_payload"],
        artifact_root=settings.artifact_root,
        database_root=settings.database_root,
        final_context_snapshot=result["final_context_snapshot"],
        project_root=settings.project_root,
        raw_operational_snapshot=result["raw_operational_snapshot"],
        verifier_implementation_paths=result["implementation_paths"],
        primary_manifest_path=primary["manifest_path"],
        primary_snapshot=primary["snapshot"],
    )
    return {**result, "published": published}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently verify the Phase 2 CRSP development profile"
    )
    parser.add_argument("--profile-config", type=Path, default=DEFAULT_PROFILE_CONFIG)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--phase-config", type=Path, default=DEFAULT_PHASE_CONFIG)
    parser.add_argument("--primary-pointer", type=Path, default=DEFAULT_PRIMARY_POINTER)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--quality-root", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--block-size-bytes", type=int, default=16 << 20)
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Run all verification work but do not publish the immutable bundle or pointer.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    settings = load_settings(
        arguments.profile_config,
        arguments.paths_config,
        arguments.phase_config,
        primary_pointer_path=arguments.primary_pointer,
        artifact_root=arguments.artifact_root,
        database_root=arguments.database_root,
        raw_root=arguments.raw_root,
        quality_root=arguments.quality_root,
        scratch_root=arguments.scratch_root,
        block_size_bytes=arguments.block_size_bytes,
    )
    result = (
        independently_verify_profile(settings)
        if arguments.no_publish
        else verify_and_publish_profile(settings)
    )
    report = result["report"]
    output = {
        "build_id": report["build_id"],
        "difference_count": report["summary"]["difference_count"],
        "status": report["status"],
        "verification_id": report["verification_id"],
    }
    if "published" in result:
        output["published"] = {
            key: _project_label(path, settings.project_root)
            for key, path in result["published"].items()
        }
    print(canonical_json_text(output))
    return 0 if report["status"] == "verified_passed" else 1


__all__ = [
    "ProfileVerificationError",
    "VerificationSettings",
    "assert_primary_snapshot_unchanged",
    "authenticate_primary_bundle",
    "build_independent_report",
    "canonical_json_bytes",
    "canonical_json_text",
    "compare_reconstruction_to_primary",
    "independently_verify_profile",
    "load_settings",
    "main",
    "reconstruct_primary_documents",
    "sha256_bytes",
    "sha256_file",
    "verifier_implementation_paths",
    "verify_and_publish_profile",
]
