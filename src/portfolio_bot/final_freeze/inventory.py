"""Deterministic metadata-only inventories for the Stage 12 pre-final freeze."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_POINTER_STATUS_VALUES = frozenset(
    {
        "phase_01_frozen",
        "passed",
        "verified_passed",
        "VERIFIED_PASSED",
        "STAGE11_VALIDATION_EXECUTION_VERIFIED",
        "STAGE12_PREFINAL_METHODOLOGY_AMENDMENT_FROZEN",
        "STAGE12_POST_UNBLIND_TECHNICAL_AMENDMENT_AUTHORIZED",
        "STAGE12_POST_UNBLIND_HASH_AUDIT_INCIDENT_DISCLOSED",
        "STAGE12_POST_UNBLIND_SCRATCH_TOPOLOGY_AMENDMENT_AUTHORIZED",
        "STAGE13_FAILED_CLOSED_ON_PARENT_SCRATCH_TOPOLOGY_CONTRACT_PREMISE_MISMATCH",
        "STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_AUTHORIZED",
        "STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_V2_AUTHORIZED",
        "STAGE12_ZERO_ROW_MERGE_PROVENANCE_AMENDMENT_AUTHORIZED",
        "STAGE12_DECEMBER_MARKET_INPUT_SCOPE_PARTITION_AMENDMENT_AUTHORIZED",
        "FAILED_CLOSED_BEFORE_DECISION_BARRIER_MISSING_FINAL_MARKET_INPUT_2025_12",
        "STAGE13_DECISION_CHECKPOINT_V4_TO_V5_REBIND_AUTHORIZED",
        "BLOCKED_PENDING_NARROW_TECHNICAL_AMENDMENT_AUTHORIZATION",
        "STAGE12_ENTRY_MARKET_CANONICALIZATION_SCOPE_ALIGNMENT_AMENDMENT_AUTHORIZED",
        "BLOCKED_PENDING_NARROW_ENTRY_MARKET_CANONICALIZATION_AND_TOMBSTONE_COVERAGE_AMENDMENT_AUTHORIZATION",
        "ENTRY_MARKET_SHARE_COVERAGE_PREFLIGHT_COMPLETE_NO_THIRD_PROJECTION_BLOCKER",
        "STAGE12_CHECKPOINT_REBIND_METADATA_CORRECTION_AMENDMENT_AUTHORIZED",
        "STAGE13_FAILED_CLOSED_ON_CHECKPOINT_REBIND_METADATA_AUTHORITY_MISMATCH",
        "STAGE13_CHECKPOINT_REBIND_METADATA_PREFLIGHT_COMPLETE",
        "THESIS_ERROR_MATERIALITY_POLICY_ACTIVE",
        "STAGE12_ACCOUNTING_MISSING_EXECUTION_COST_TRANSPORT_AMENDMENT_AUTHORIZED",
        "STAGE13_FAILED_CLOSED_BEFORE_FIRST_OUTCOME_MONTH_ON_ADV20_NULLABILITY_CONTRADICTION",
        "STAGE13_ACCOUNTING_MISSING_EXECUTION_COST_PREFLIGHT_COMPLETE",
        "STAGE12_LOGICAL_FINGERPRINT_TRANSPORT_AMENDMENT_AUTHORIZED",
        "STAGE13_FAILED_CLOSED_BEFORE_DECISION_BARRIER_ON_PHYSICAL_FINGERPRINT_FALSE_DIFFERENCE",
        "STAGE13_FINAL_MARKET_INPUT_LOGICAL_FINGERPRINT_PREFLIGHT_COMPLETE",
        "STAGE12_FACTOR_COVARIANCE_ORDINAL_BASIS_ALIGNMENT_AMENDMENT_AUTHORIZED",
        "STAGE13_FAILED_CLOSED_BEFORE_NORMALIZED_SOURCE_PUBLICATION_ON_FACTOR_COVARIANCE_ORDINAL_BASIS_MISMATCH",
        "STAGE13_FACTOR_COVARIANCE_ORDINAL_BASIS_PREFLIGHT_COMPLETE",
        "STAGE13_RESUME3_FAILED_CLOSED_BEFORE_FIRST_ECONOMIC_CHECKPOINT",
        "STAGE13_EQUAL_WEIGHT_MISSING_COST_GATE_PREFLIGHT_COMPLETE",
        "STAGE12_EQUAL_WEIGHT_MISSING_COST_GATE_AMENDMENT_AUTHORIZED",
        "post_unblind_economic_data_gap_treatment",
    }
)


class Stage12InventoryError(RuntimeError):
    """Raised when an inventory input is unsafe, incomplete, or changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage12InventoryError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class StableFile:
    path: Path
    sha256: str
    size_bytes: int
    payload: bytes | None = None


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without resolving or following symlinks."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def confined_existing_path(
    path: Path,
    *,
    trusted_root: Path,
    file_required: bool,
    label: str,
) -> Path:
    """Prove that an existing path is lexically and physically below a root.

    The check deliberately happens before callers open the path.  Every
    descendant component is inspected without following symlinks and the
    resolved path is then required to remain under the resolved root.
    """

    root = _lexical_absolute(trusted_root)
    selected = _lexical_absolute(path)
    try:
        relative = selected.relative_to(root)
    except ValueError as exc:
        raise Stage12InventoryError(f"{label} escaped its trusted root") from exc
    require(
        relative.parts and ".." not in relative.parts,
        f"unsafe {label} relative path",
    )
    try:
        root_metadata = root.stat(follow_symlinks=False)
    except OSError as exc:
        raise Stage12InventoryError(f"missing trusted root for {label}") from exc
    require(
        stat.S_ISDIR(root_metadata.st_mode) and not stat.S_ISLNK(root_metadata.st_mode),
        f"unsafe trusted root for {label}",
    )
    current = root
    for index, component in enumerate(relative.parts):
        require(
            component not in {"", ".", ".."}
            and "/" not in component
            and "\\" not in component
            and "\x00" not in component,
            f"unsafe {label} path component",
        )
        current = current / component
        try:
            metadata = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise Stage12InventoryError(f"missing {label}: {selected}") from exc
        require(
            not stat.S_ISLNK(metadata.st_mode),
            f"symlinked {label} component: {current}",
        )
        final = index == len(relative.parts) - 1
        if final and file_required:
            require(
                stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
                f"unsafe {label} file: {current}",
            )
        else:
            require(stat.S_ISDIR(metadata.st_mode), f"unsafe {label} directory")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = selected.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise Stage12InventoryError(
            f"{label} resolved outside its trusted root"
        ) from exc
    return selected


def confined_relative_path(
    trusted_root: Path,
    relative: object,
    *,
    file_required: bool,
    label: str,
) -> Path:
    """Validate one untrusted relative path before resolving or reading it."""

    text = str(relative)
    candidate = Path(text)
    require(
        bool(text)
        and not candidate.is_absolute()
        and candidate.as_posix() == text
        and ".." not in candidate.parts
        and "." not in candidate.parts
        and "\\" not in text
        and "\x00" not in text,
        f"unsafe relative {label}",
    )
    selected = _lexical_absolute(_lexical_absolute(trusted_root) / candidate)
    return confined_existing_path(
        selected,
        trusted_root=trusted_root,
        file_required=file_required,
        label=label,
    )


def stable_file(
    path: Path,
    *,
    keep_payload: bool = False,
    trusted_root: Path | None = None,
) -> StableFile:
    """Hash one ordinary single-link file while proving stable identity."""

    selected = (
        confined_existing_path(
            path,
            trusted_root=trusted_root,
            file_required=True,
            label="inventory file",
        )
        if trusted_root is not None
        else _lexical_absolute(path)
    )
    try:
        before = selected.stat(follow_symlinks=False)
    except OSError as exc:
        raise Stage12InventoryError(f"cannot stat inventory file: {selected}") from exc
    require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_nlink == 1,
        f"unsafe inventory file: {selected}",
    )
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if keep_payload else None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(selected, flags)
    except OSError as exc:
        raise Stage12InventoryError(f"cannot open inventory file: {selected}") from exc
    try:
        opened = os.fstat(descriptor)
        require(
            all(
                getattr(before, field) == getattr(opened, field)
                for field in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            ),
            f"inventory file identity changed before open: {selected}",
        )
        while block := os.read(descriptor, 8 << 20):
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise Stage12InventoryError(f"cannot read inventory file: {selected}") from exc
    finally:
        os.close(descriptor)
    try:
        path_after = selected.stat(follow_symlinks=False)
    except OSError as exc:
        raise Stage12InventoryError(
            f"inventory file disappeared after hashing: {selected}"
        ) from exc
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    require(
        all(getattr(before, field) == getattr(after, field) for field in fields)
        and all(
            getattr(after, field) == getattr(path_after, field) for field in fields
        ),
        f"inventory file changed while hashing: {selected}",
    )
    return StableFile(
        path=selected,
        sha256=digest.hexdigest(),
        size_bytes=int(after.st_size),
        payload=None if chunks is None else b"".join(chunks),
    )


def load_json_object(
    path: Path,
    *,
    trusted_root: Path | None = None,
) -> tuple[dict[str, Any], StableFile]:
    record = stable_file(path, keep_payload=True, trusted_root=trusted_root)
    assert record.payload is not None
    try:
        value = json.loads(record.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage12InventoryError(f"invalid JSON object: {path}") from exc
    require(isinstance(value, dict), f"JSON value must be an object: {path}")
    return value, record


def _project_relative(path: Path, *, project_root: Path) -> str:
    try:
        relative = path.absolute().relative_to(project_root.absolute())
    except ValueError as exc:
        raise Stage12InventoryError("inventory path escaped project root") from exc
    require(
        relative.parts
        and ".." not in relative.parts
        and not any(
            part.startswith(".") and part != ".gitignore" for part in relative.parts
        ),
        f"unsafe project inventory path: {relative}",
    )
    return relative.as_posix()


def _iter_regular_files(root: Path, *, project_root: Path) -> Iterable[Path]:
    confined_existing_path(
        root,
        trusted_root=project_root,
        file_required=False,
        label="project inventory root",
    )
    for path in sorted(root.rglob("*")):
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if path.is_symlink():
            raise Stage12InventoryError(f"symlink in frozen inventory: {path}")
        if path.is_file():
            yield path


def build_project_file_inventory(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Hash the complete executable/configuration/test surface."""

    inventory = policy.get("inventory")
    require(isinstance(inventory, Mapping), "inventory policy is missing")
    records: dict[str, dict[str, Any]] = {}

    def add(path: Path, *, domain: str) -> None:
        relative = _project_relative(path, project_root=project_root)
        require(relative not in records, f"duplicate inventory path: {relative}")
        file = stable_file(path, trusted_root=project_root)
        records[relative] = {
            "domain": domain,
            "logical_root": "project://",
            "relative_path": relative,
            "sha256": file.sha256,
            "size_bytes": file.size_bytes,
            "required_for_stage13": (
                relative.startswith(("src/", "scripts/", "configs/"))
                or relative == "pyproject.toml"
            ),
            "required_for_stage14": (
                relative.startswith(("src/", "scripts/", "configs/"))
                or relative == "pyproject.toml"
            ),
            "identity_mode": "full_file_sha256",
            "physical_hash_verified": True,
        }

    python_roots = inventory.get("include_python_under")
    require(isinstance(python_roots, list), "Python inventory roots are missing")
    for name in python_roots:
        root = confined_relative_path(
            project_root,
            name,
            file_required=False,
            label="Python inventory root",
        )
        for path in _iter_regular_files(root, project_root=project_root):
            if path.suffix == ".py":
                domain = "test" if str(name).startswith("tests/") else "code"
                add(path, domain=domain)

    fixture_roots = inventory.get("include_test_fixture_under", [])
    require(isinstance(fixture_roots, list), "test fixture roots are invalid")
    for name in fixture_roots:
        root = confined_relative_path(
            project_root,
            name,
            file_required=False,
            label="test fixture inventory root",
        )
        for path in _iter_regular_files(root, project_root=project_root):
            if path.suffix != ".pyc":
                add(path, domain="test_fixture")

    config_roots = inventory.get("include_configuration_under")
    require(isinstance(config_roots, list), "configuration roots are missing")
    for name in config_roots:
        root = confined_relative_path(
            project_root,
            name,
            file_required=False,
            label="configuration inventory root",
        )
        for path in _iter_regular_files(root, project_root=project_root):
            add(path, domain="configuration")

    exact = inventory.get("include_exact_files")
    require(isinstance(exact, list), "exact inventory file list is missing")
    for name in exact:
        path = confined_relative_path(
            project_root,
            name,
            file_required=True,
            label="exact inventory file",
        )
        add(
            path,
            domain=(
                "methodology"
                if str(name).startswith("outputs/methodology")
                else "environment_definition"
            ),
        )
    ordered = [records[name] for name in sorted(records)]
    tree_sha = sha256_bytes(canonical_json_bytes(ordered))
    counts: dict[str, int] = {}
    bytes_by_domain: dict[str, int] = {}
    for record in ordered:
        domain = str(record["domain"])
        counts[domain] = counts.get(domain, 0) + 1
        bytes_by_domain[domain] = bytes_by_domain.get(domain, 0) + int(
            record["size_bytes"]
        )
    return {
        "schema_version": 1,
        "inventory_id": f"stage12-project-tree-{tree_sha[:24]}",
        "revision_method": "canonical_sha256_file_inventory_and_tree_digest",
        "git_repository_present": (project_root / ".git").is_dir(),
        "tree_sha256": tree_sha,
        "file_count": len(ordered),
        "counts_by_domain": dict(sorted(counts.items())),
        "bytes_by_domain": dict(sorted(bytes_by_domain.items())),
        "files": ordered,
    }


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def build_environment_lock(*, project_root: Path) -> dict[str, Any]:
    """Capture a portable exact package/version environment receipt."""

    project = confined_relative_path(
        project_root,
        "pyproject.toml",
        file_required=True,
        label="project environment definition",
    )
    project_record = stable_file(
        project,
        keep_payload=True,
        trusted_root=project_root,
    )
    assert project_record.payload is not None
    try:
        pyproject = tomllib.loads(project_record.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise Stage12InventoryError("invalid pyproject.toml") from exc
    dependencies = pyproject.get("project", {}).get("dependencies")
    require(isinstance(dependencies, list), "project dependency lock is missing")
    installed_versions: dict[str, set[str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str) or not name:
            continue
        normalized = _normalized_distribution_name(name)
        version = str(distribution.version)
        installed_versions.setdefault(normalized, set()).add(version)
    direct: list[dict[str, str]] = []
    for requirement in dependencies:
        match = re.fullmatch(r"\s*([A-Za-z0-9_.-]+)==([^\s;]+)\s*", str(requirement))
        require(match is not None, f"dependency is not exactly pinned: {requirement}")
        assert match is not None
        name = _normalized_distribution_name(match.group(1))
        required_version = match.group(2)
        require(
            name in installed_versions,
            f"direct dependency is not installed: {name}",
        )
        require(
            installed_versions[name] == {required_version},
            f"direct dependency version differs: {name}",
        )
        direct.append({"name": name, "version": required_version})
    packages = [
        {"name": name, "versions": sorted(installed_versions[name])}
        for name in sorted(installed_versions)
    ]
    package_lock_sha = sha256_bytes(canonical_json_bytes(packages))
    executable = Path(sys.executable)
    executable_target = executable.resolve(strict=True)
    executable_record = stable_file(executable_target)
    return {
        "schema_version": 1,
        "environment_id": f"stage12-environment-{package_lock_sha[:24]}",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_version_info": list(sys.version_info[:5]),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "byteorder": sys.byteorder,
        "executable_invocation_name": executable.name,
        "executable_resolves_outside_project": (
            project_root.absolute() not in executable_target.parents
        ),
        "executable_sha256": executable_record.sha256,
        "pyproject_sha256": project_record.sha256,
        "direct_dependencies": sorted(direct, key=lambda item: item["name"]),
        "installed_distributions": packages,
        "installed_distribution_count": len(packages),
        "installed_distribution_lock_sha256": package_lock_sha,
        "deterministic_seeds": [11, 23, 47, 71, 101, 20260714],
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def authenticate_authority_inventory(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Authenticate required upstream pointers without opening data payloads."""

    required = policy.get("required_authorities")
    require(isinstance(required, list) and required, "authority list is missing")
    records: list[dict[str, Any]] = []
    roles: set[str] = set()
    for item in required:
        require(isinstance(item, Mapping), "authority entry must be an object")
        role = str(item.get("role"))
        relative = str(item.get("path"))
        expected_sha = str(item.get("sha256"))
        require(role not in roles and role, "authority roles must be unique")
        roles.add(role)
        require(_HEX64.fullmatch(expected_sha) is not None, "authority hash is invalid")
        path = confined_relative_path(
            project_root,
            relative,
            file_required=True,
            label=f"{role} authority pointer",
        )
        value, record = load_json_object(path, trusted_root=project_root)
        require(record.sha256 == expected_sha, f"{role} pointer SHA-256 differs")
        status = value.get("status", value.get("verification_status"))
        require(status in _POINTER_STATUS_VALUES, f"{role} pointer status differs")
        records.append(
            {
                "role": role,
                "logical_root": "project://",
                "relative_path": relative,
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
                "status": status,
                "build_id": value.get(
                    "build_id",
                    value.get("primary_build_id", value.get("phase_01_build_id")),
                ),
                "verification_id": value.get("verification_id"),
                "payload_file_opened": False,
            }
        )
        if role == "validation_selection":
            require(
                value.get("primary_build_id")
                == "stage11-validation-8d5e7f863ee093089520457e"
                and value.get("verification_id") == "verify-86a1e2d42b4c84f0"
                and value.get("difference_count") == 0,
                "Stage 11 verified identity differs",
            )
            authorizations = value.get("authorizations")
            require(
                isinstance(authorizations, Mapping)
                and authorizations.get("final_2020_plus_access_authorized") is False
                and authorizations.get("stage12_or_later_authorized") is False,
                "Stage 11 final boundary differs",
            )
    raw_namespace = confined_relative_path(
        project_root,
        "artifacts/phase_01_raw_manifest",
        file_required=False,
        label="raw manifest namespace",
    )
    raw_pointer_path = confined_relative_path(
        raw_namespace,
        "LATEST_FROZEN.json",
        file_required=True,
        label="raw freeze pointer",
    )
    raw_pointer, _ = load_json_object(
        raw_pointer_path,
        trusted_root=raw_namespace,
    )
    raw_manifest_path = confined_relative_path(
        raw_namespace,
        raw_pointer["path"],
        file_required=True,
        label="raw transitive manifest",
    )
    raw_manifest = stable_file(raw_manifest_path, trusted_root=raw_namespace)
    require(
        raw_manifest.sha256 == raw_pointer.get("sha256"),
        "raw freeze manifest SHA-256 differs",
    )
    records.append(
        {
            "role": "raw_snapshot_transitive_manifest",
            "logical_root": "project://",
            "relative_path": _project_relative(
                raw_manifest_path, project_root=project_root
            ),
            "sha256": raw_manifest.sha256,
            "size_bytes": raw_manifest.size_bytes,
            "status": "phase_01_frozen_manifest",
            "build_id": raw_pointer.get("phase_01_build_id"),
            "verification_id": raw_pointer.get("verification_id"),
            "payload_file_opened": False,
        }
    )
    records.sort(key=lambda item: (str(item["role"]), str(item["relative_path"])))
    inventory_sha = sha256_bytes(canonical_json_bytes(records))
    return {
        "schema_version": 1,
        "authority_inventory_id": f"stage12-authorities-{inventory_sha[:24]}",
        "authority_inventory_sha256": inventory_sha,
        "authority_count": len(records),
        "records": records,
        "transitive_raw_file_hash_registry_bound": True,
        "raw_payload_files_opened": 0,
        "development_payload_rows_read": 0,
        "validation_payload_rows_read": 0,
        "final_period_payload_rows_read": 0,
        "stage13_outputs_opened": 0,
        "maximum_empirical_date_read": None,
    }


def inventory_summary_sha256(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


__all__ = [
    "StableFile",
    "Stage12InventoryError",
    "authenticate_authority_inventory",
    "build_environment_lock",
    "build_project_file_inventory",
    "canonical_json_bytes",
    "confined_existing_path",
    "confined_relative_path",
    "inventory_summary_sha256",
    "load_json_object",
    "require",
    "sha256_bytes",
    "stable_file",
]
