"""Safely publish the Phase 2 CRSP supporting-source profile.

The compiler owns every deterministic evidence byte.  This module authenticates
those bytes, publishes two exact immutable directories, records the local run,
and moves the mutable primary pointer last.  It deliberately has no code path
that writes canonical, interim, processed, feature, target, or return data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import portfolio_bot.data.phase_02_crsp_supporting_profile as supporting_compiler
from portfolio_bot.data.phase_02_crsp_supporting_profile import (
    SupportingProfileError,
    SupportingProfileSettings,
    canonical_json_bytes,
    compile_supporting_profile,
    load_settings,
    render_artifact_documents,
    render_quality_payloads,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_supporting_sources_development_v1.toml"
)
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.toml"
DEFAULT_PHASE_CONFIG = PROJECT_ROOT / "configs" / "phase_02_canonical_panel.toml"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "phase_02_canonical_panel"
CLI_RELATIVE_PATH = Path("scripts/build_phase_02_crsp_supporting_profile.py")
PUBLISHED_STATUS = "supporting_profile_primary_passed_pending_independent_verification"


class SupportingProfilePublicationError(SupportingProfileError):
    """Raised when the exact supporting profile cannot be published safely."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _lexical_absolute(path: str | Path) -> Path:
    """Normalize a path without dereferencing symlink ancestry."""
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
        except OSError as exc:
            raise SupportingProfilePublicationError(
                f"Cannot authenticate {label} path component {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise SupportingProfilePublicationError(
                f"Symlink is prohibited in {label} path: {current}"
            )


def _assert_safe_component(value: Any, label: str) -> str:
    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or Path(component).name != component
        or "/" in component
        or "\\" in component
    ):
        raise SupportingProfilePublicationError(
            f"Unsafe {label} path component: {component!r}"
        )
    return component


def _assert_optional_regular_file(path: Path, label: str) -> None:
    _assert_no_symlink_components(path, label)
    if path.exists() and not path.is_file():
        raise SupportingProfilePublicationError(
            f"Existing {label} is not a regular file"
        )


def _relative_to(path: Path, root: Path, label: str) -> Path:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise SupportingProfilePublicationError(
            f"{label} is outside its allowed root"
        ) from exc


def _assert_allowed_roots(settings: SupportingProfileSettings) -> None:
    project_root = _lexical_absolute(settings.project_root)
    database_root = _lexical_absolute(settings.database_root)
    raw_root = _lexical_absolute(settings.raw_root)
    quality_root = _lexical_absolute(settings.quality_root)
    for label, root in (
        ("project root", project_root),
        ("database root", database_root),
        ("raw root", raw_root),
        ("quality root", quality_root),
        ("artifact root", _lexical_absolute(settings.artifact_root)),
        ("run root", _lexical_absolute(settings.run_root)),
    ):
        _assert_no_symlink_components(root, label)

    raw_relative = _relative_to(raw_root, database_root, "raw root")
    if not raw_relative.parts or raw_relative.parts[0] != "raw":
        raise SupportingProfilePublicationError(
            "Supporting-profile raw reads must be rooted under database/raw/"
        )
    quality_relative = _relative_to(quality_root, database_root, "quality root")
    if not quality_relative.parts or quality_relative.parts[0] != "quality":
        raise SupportingProfilePublicationError(
            "Supporting-profile database writes must be rooted under database/quality/"
        )

    for label, root in (
        ("artifact root", _lexical_absolute(settings.artifact_root)),
        ("run root", _lexical_absolute(settings.run_root)),
    ):
        _relative_to(root, project_root, label)
        try:
            database_relative = root.resolve(strict=False).relative_to(
                database_root.resolve(strict=False)
            )
        except ValueError:
            continue
        if not database_relative.parts or database_relative.parts[0] != "quality":
            raise SupportingProfilePublicationError(
                f"{label} may not write outside database/quality/ inside the database"
            )


def _read_authenticated_regular_file(path: Path, label: str) -> bytes:
    """Read bytes through a no-follow descriptor and reject a changing inode."""
    path = _lexical_absolute(path)
    _assert_no_symlink_components(path, label)
    try:
        path_before = path.lstat()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise SupportingProfilePublicationError(
            f"Cannot open authenticated {label}: {path}"
        ) from exc
    try:
        descriptor_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_before.st_mode)
            or path_before.st_dev != descriptor_before.st_dev
            or path_before.st_ino != descriptor_before.st_ino
        ):
            raise SupportingProfilePublicationError(
                f"Authenticated {label} is not the selected regular file"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        descriptor_after = os.fstat(descriptor)
        path_after = path.lstat()
        stable = (
            descriptor_before.st_dev == descriptor_after.st_dev == path_after.st_dev
            and descriptor_before.st_ino == descriptor_after.st_ino == path_after.st_ino
            and descriptor_before.st_size
            == descriptor_after.st_size
            == path_after.st_size
            and descriptor_before.st_mtime_ns
            == descriptor_after.st_mtime_ns
            == path_after.st_mtime_ns
        )
        if not stable:
            raise SupportingProfilePublicationError(
                f"Authenticated {label} changed while it was being read"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _assert_sha256_path(path: Path, expected: str, label: str) -> bytes:
    payload = _read_authenticated_regular_file(path, label)
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise SupportingProfilePublicationError(
            f"{label} SHA-256 differs: expected {expected}, actual {actual}"
        )
    return payload


def _project_path(settings: SupportingProfileSettings, value: Any, label: str) -> Path:
    text = str(value)
    posix = PurePosixPath(text)
    if (
        posix.is_absolute()
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
        or "\\" in text
    ):
        raise SupportingProfilePublicationError(
            f"Unsafe project-relative {label}: {text!r}"
        )
    path = _lexical_absolute(settings.project_root / Path(*posix.parts))
    _relative_to(path, _lexical_absolute(settings.project_root), label)
    return path


def _assert_pointer_identity(
    payload: bytes, expected: Mapping[str, Any], label: str
) -> None:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupportingProfilePublicationError(
            f"Authenticated {label} is not UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict) or any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise SupportingProfilePublicationError(
            f"Authenticated {label} identity differs from the locked config"
        )


def _authenticate_external_context(
    settings: SupportingProfileSettings, compiled: Mapping[str, Any]
) -> None:
    """Rehash config, upstream evidence, raw sources, and compiler bytes."""
    try:
        binding = compiled["binding"]
        expected_config_sha = binding["profile_config_sha256"]
    except (KeyError, TypeError) as exc:
        raise SupportingProfilePublicationError(
            "Compiled profile lacks its external binding"
        ) from exc
    profile_payload = _assert_sha256_path(
        settings.profile_config_path, expected_config_sha, "profile config"
    )
    try:
        fresh_config = tomllib.loads(profile_payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SupportingProfilePublicationError(
            "Authenticated profile config is not valid UTF-8 TOML"
        ) from exc
    if fresh_config != settings.config:
        raise SupportingProfilePublicationError(
            "In-memory settings differ from freshly authenticated profile config"
        )

    _assert_sha256_path(
        settings.paths_config_path, binding["paths_config_sha256"], "paths config"
    )
    phase_entry = fresh_config["upstream"]["phase_02"]
    phase_path = _project_path(settings, phase_entry["phase_config"], "Phase 2 config")
    if phase_path != _lexical_absolute(settings.phase_config_path):
        raise SupportingProfilePublicationError(
            "Settings phase config differs from the locked project-relative path"
        )
    _assert_sha256_path(
        phase_path, phase_entry["phase_config_sha256"], "Phase 2 config"
    )
    if binding["phase_config_sha256"] != phase_entry["phase_config_sha256"]:
        raise SupportingProfilePublicationError(
            "Compiled Phase 2 config binding differs from the locked config"
        )

    methodology = fresh_config["upstream"]["methodology"]
    _assert_sha256_path(
        _project_path(settings, methodology["freeze_manifest"], "methodology freeze"),
        methodology["freeze_manifest_sha256"],
        "methodology freeze",
    )
    if binding["methodology_freeze_sha256"] != methodology["freeze_manifest_sha256"]:
        raise SupportingProfilePublicationError(
            "Compiled methodology binding differs from the locked config"
        )

    phase_01 = fresh_config["upstream"]["phase_01"]
    _assert_sha256_path(
        _project_path(settings, phase_01["freeze_manifest"], "Phase 1 freeze"),
        phase_01["freeze_manifest_sha256"],
        "Phase 1 freeze",
    )
    if binding["phase_01_freeze_manifest_sha256"] != phase_01["freeze_manifest_sha256"]:
        raise SupportingProfilePublicationError(
            "Compiled Phase 1 binding differs from the locked config"
        )

    contract = fresh_config["upstream"]["contract"]
    contract_payload = _assert_sha256_path(
        _project_path(settings, contract["verified_pointer"], "contract pointer"),
        contract["verified_pointer_sha256"],
        "contract pointer",
    )
    _assert_pointer_identity(
        contract_payload,
        {
            "contract_id": contract["contract_id"],
            "verification_id": contract["verification_id"],
            "status": contract["required_status"],
        },
        "contract pointer",
    )
    if (
        binding["contract_id"] != contract["contract_id"]
        or binding["contract_verification_id"] != contract["verification_id"]
    ):
        raise SupportingProfilePublicationError(
            "Compiled contract binding differs from the locked config"
        )

    calendar = fresh_config["upstream"]["calendar"]
    calendar_payload = _assert_sha256_path(
        _project_path(settings, calendar["verified_pointer"], "calendar pointer"),
        calendar["verified_pointer_sha256"],
        "calendar pointer",
    )
    _assert_pointer_identity(
        calendar_payload,
        {
            "build_id": calendar["build_id"],
            "verification_id": calendar["verification_id"],
            "status": calendar["required_status"],
        },
        "calendar pointer",
    )
    if (
        binding["calendar_build_id"] != calendar["build_id"]
        or binding["calendar_verification_id"] != calendar["verification_id"]
    ):
        raise SupportingProfilePublicationError(
            "Compiled calendar binding differs from the locked config"
        )

    prior = fresh_config["upstream"]["daily_monthly_profile"]
    prior_payload = _assert_sha256_path(
        _project_path(
            settings, prior["verified_pointer"], "daily/monthly profile pointer"
        ),
        prior["verified_pointer_sha256"],
        "daily/monthly profile pointer",
    )
    _assert_pointer_identity(
        prior_payload,
        {
            "build_id": prior["build_id"],
            "verification_id": prior["verification_id"],
            "primary_manifest_sha256": prior["primary_manifest_sha256"],
            "status": prior["required_status"],
        },
        "daily/monthly profile pointer",
    )
    if (
        binding["daily_monthly_profile_build_id"] != prior["build_id"]
        or binding["daily_monthly_profile_verification_id"] != prior["verification_id"]
    ):
        raise SupportingProfilePublicationError(
            "Compiled daily/monthly-profile binding differs from the locked config"
        )

    for source in fresh_config["sources"]:
        filename = _assert_safe_component(source["filename"], "raw filename")
        payload = _assert_sha256_path(
            _lexical_absolute(settings.raw_root / filename),
            source["sha256"],
            f"raw source {filename}",
        )
        if len(payload) != int(source["size_bytes"]):
            raise SupportingProfilePublicationError(
                f"Raw source {filename} size differs from the locked config"
            )

    compiler_path = _lexical_absolute(Path(supporting_compiler.__file__))
    compiler_sha = _sha256_bytes(
        _read_authenticated_regular_file(compiler_path, "compiler implementation")
    )
    if compiler_sha != binding["implementation_sha256"]:
        raise SupportingProfilePublicationError(
            "Compiler implementation changed after profile compilation"
        )


def _implementation_binding(
    settings: SupportingProfileSettings, compiled: Mapping[str, Any]
) -> dict[str, str]:
    compiler_path = _lexical_absolute(Path(supporting_compiler.__file__))
    publisher_path = _lexical_absolute(Path(__file__))
    cli_path = _project_path(settings, CLI_RELATIVE_PATH.as_posix(), "CLI script")
    binding = {
        "cli_sha256": _sha256_bytes(
            _read_authenticated_regular_file(cli_path, "CLI implementation")
        ),
        "compiler_sha256": _sha256_bytes(
            _read_authenticated_regular_file(compiler_path, "compiler implementation")
        ),
        "publisher_sha256": _sha256_bytes(
            _read_authenticated_regular_file(publisher_path, "publisher implementation")
        ),
    }
    expected = compiled.get("binding", {}).get("implementation_sha256")
    if binding["compiler_sha256"] != expected:
        raise SupportingProfilePublicationError(
            "Compiler implementation differs from the compiled binding"
        )
    return binding


def _assert_locked_compiled_context(
    settings: SupportingProfileSettings, compiled: Mapping[str, Any]
) -> None:
    try:
        build_id = _assert_safe_component(compiled["build_id"], "build_id")
        contract_id = _assert_safe_component(
            compiled["binding"]["contract_id"], "contract_id"
        )
        summary = compiled["summary"]
    except (KeyError, TypeError) as exc:
        raise SupportingProfilePublicationError(
            "Compiled profile lacks its publication identity"
        ) from exc
    if not contract_id:
        raise SupportingProfilePublicationError("Compiled contract_id is blank")
    if summary.get("build_id") != build_id:
        raise SupportingProfilePublicationError(
            "Summary build_id differs from compiled build_id"
        )
    if summary.get("status") != PUBLISHED_STATUS:
        raise SupportingProfilePublicationError(
            "Only a passing primary supporting profile may be published"
        )
    if int(summary.get("fail_count", -1)) != 0:
        raise SupportingProfilePublicationError(
            "A supporting profile with failed checks may not be published"
        )
    config_payload = _read_authenticated_regular_file(
        settings.profile_config_path, "profile config"
    )
    if _sha256_bytes(config_payload) != compiled["binding"]["profile_config_sha256"]:
        raise SupportingProfilePublicationError(
            "Compiled profile does not bind the selected config bytes"
        )
    try:
        fresh_config = tomllib.loads(config_payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SupportingProfilePublicationError(
            "Profile config is not valid UTF-8 TOML"
        ) from exc
    if fresh_config != settings.config:
        raise SupportingProfilePublicationError(
            "In-memory settings differ from freshly parsed profile config"
        )


def _quality_directory(
    settings: SupportingProfileSettings, compiled: Mapping[str, Any]
) -> Path:
    contract_id = _assert_safe_component(
        compiled["binding"]["contract_id"], "contract_id"
    )
    subdirectory = _assert_safe_component(
        settings.config["outputs"]["quality_subdirectory"],
        "quality subdirectory",
    )
    build_id = _assert_safe_component(compiled["build_id"], "build_id")
    return _lexical_absolute(
        settings.quality_root / contract_id / subdirectory / build_id
    )


def _validate_payload_names(
    payloads: Mapping[str, bytes], expected_names: Sequence[str], label: str
) -> dict[str, bytes]:
    expected = set(expected_names)
    actual = set(payloads)
    if len(expected) != len(tuple(expected_names)):
        raise SupportingProfilePublicationError(
            f"Configured {label} closed file set contains duplicates"
        )
    if actual != expected:
        raise SupportingProfilePublicationError(
            f"{label} closed file set differs: expected {sorted(expected)}, "
            f"actual {sorted(actual)}"
        )
    validated: dict[str, bytes] = {}
    for name in sorted(payloads):
        _assert_safe_component(name, f"{label} filename")
        payload = payloads[name]
        if not isinstance(payload, bytes):
            raise SupportingProfilePublicationError(
                f"{label} payload {name!r} is not bytes"
            )
        validated[name] = payload
    return validated


def _validate_rendered_documents(
    settings: SupportingProfileSettings,
    compiled: Mapping[str, Any],
    quality_directory: Path,
    quality_payloads: Mapping[str, bytes],
    artifact_directory: Path,
    artifact_payloads: Mapping[str, bytes],
    documents: Mapping[str, Any],
) -> None:
    database_root = _lexical_absolute(settings.database_root)
    try:
        quality_relative = quality_directory.relative_to(database_root)
    except ValueError as exc:
        raise SupportingProfilePublicationError(
            "Quality destination is outside the configured database root"
        ) from exc
    if not quality_relative.parts or quality_relative.parts[0] != "quality":
        raise SupportingProfilePublicationError(
            "Manifest quality paths must begin with quality/"
        )
    outputs = [
        {
            "path": (quality_relative / name).as_posix(),
            "sha256": _sha256_bytes(quality_payloads[name]),
            "size_bytes": len(quality_payloads[name]),
        }
        for name in sorted(quality_payloads)
    ]
    expected_manifest = {
        "binding": compiled["binding"],
        "build_id": compiled["build_id"],
        "contract_id": compiled["binding"]["contract_id"],
        "outputs": outputs,
        "profile_name": settings.config["profile_name"],
        "schema_version": int(settings.config["schema_version"]),
        "status": compiled["summary"]["status"],
        "summary": compiled["summary"],
    }
    manifest_bytes = canonical_json_bytes(expected_manifest)
    if documents.get("manifest") != expected_manifest:
        raise SupportingProfilePublicationError(
            "Rendered manifest differs from the exact authenticated manifest"
        )
    if artifact_payloads.get("manifest.json") != manifest_bytes:
        raise SupportingProfilePublicationError(
            "Rendered manifest bytes differ from the authenticated object"
        )
    inventory = "".join(
        f"{item['sha256']}  {item['path']}\n" for item in outputs
    ).encode("utf-8")
    if artifact_payloads.get("output_hashes.sha256") != inventory:
        raise SupportingProfilePublicationError(
            "Rendered output inventory differs from the quality payloads"
        )
    try:
        artifact_label = artifact_directory.relative_to(
            _lexical_absolute(settings.project_root)
        ).as_posix()
    except ValueError as exc:
        raise SupportingProfilePublicationError(
            "Artifact path is not project-root-relative"
        ) from exc
    artifact_posix = PurePosixPath(artifact_label)
    if (
        artifact_posix.is_absolute()
        or not artifact_posix.parts
        or any(part in {"", ".", ".."} for part in artifact_posix.parts)
        or "\\" in artifact_label
    ):
        raise SupportingProfilePublicationError(
            "Artifact path is not a safe project-relative path"
        )
    expected_pointer = {
        "artifact_path": artifact_label,
        "build_id": compiled["build_id"],
        "contract_id": compiled["binding"]["contract_id"],
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "schema_version": int(settings.config["schema_version"]),
        "status": compiled["summary"]["status"],
    }
    if documents.get("pointer") != expected_pointer:
        raise SupportingProfilePublicationError(
            "Rendered pointer differs from the exact authenticated pointer"
        )
    if documents.get("pointer_bytes") != canonical_json_bytes(expected_pointer):
        raise SupportingProfilePublicationError(
            "Rendered pointer bytes differ from the authenticated object"
        )


def _read_closed_directory(directory: Path, label: str) -> dict[str, bytes]:
    _assert_no_symlink_components(directory, label)
    if not directory.is_dir():
        raise SupportingProfilePublicationError(
            f"Existing {label} is not a real directory"
        )
    result: dict[str, bytes] = {}
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.is_symlink() or not entry.is_file():
            raise SupportingProfilePublicationError(
                f"Existing {label} contains a non-regular entry: {entry.name}"
            )
        result[entry.name] = _read_authenticated_regular_file(
            entry, f"{label} file {entry.name}"
        )
    return result


def _assert_existing_matches(
    final: Path, payloads: Mapping[str, bytes], label: str
) -> bool:
    _assert_no_symlink_components(final, label)
    if not final.exists() and not final.is_symlink():
        return False
    actual = _read_closed_directory(final, label)
    if actual != dict(payloads):
        raise SupportingProfilePublicationError(
            f"Existing {label} differs from deterministic payload"
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
            raise SupportingProfilePublicationError(
                "Short write while staging immutable payload"
            )
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
                raise SupportingProfilePublicationError(
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
    _assert_no_symlink_components(parent, "staging parent")
    parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(parent, "staging parent")
    stage = Path(tempfile.mkdtemp(prefix=f".{prefix}.staging-", dir=parent))
    directory_descriptor: int | None = None
    try:
        _assert_no_symlink_components(stage, "staging directory")
        directory_descriptor = os.open(
            stage,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        path_metadata = stage.lstat()
        descriptor_metadata = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(descriptor_metadata.st_mode)
            or path_metadata.st_dev != descriptor_metadata.st_dev
            or path_metadata.st_ino != descriptor_metadata.st_ino
        ):
            raise SupportingProfilePublicationError(
                "Staging path identity changed before anchored writes"
            )
        for name, payload in payloads.items():
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_descriptor,
            )
            try:
                _write_all(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        if _read_anchored_payloads(directory_descriptor) != dict(payloads):
            raise SupportingProfilePublicationError(
                "Staged bytes differ immediately after writing"
            )
        final_metadata = stage.lstat()
        if (
            not stat.S_ISDIR(final_metadata.st_mode)
            or final_metadata.st_dev != descriptor_metadata.st_dev
            or final_metadata.st_ino != descriptor_metadata.st_ino
        ):
            raise SupportingProfilePublicationError(
                "Staging path identity changed during anchored writes"
            )
    except Exception:
        _remove_staging_entry(stage)
        raise
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    return stage


def _publish_staged_directory(
    stage: Path, final: Path, payloads: Mapping[str, bytes], label: str
) -> None:
    if _assert_existing_matches(final, payloads, label):
        _remove_staging_entry(stage)
        return
    _assert_no_symlink_components(final.parent, f"{label} parent")
    final.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(final, label)
    try:
        os.rename(stage, final)
    except OSError:
        if _assert_existing_matches(final, payloads, label):
            _remove_staging_entry(stage)
            return
        raise
    if _read_closed_directory(final, label) != dict(payloads):
        raise SupportingProfilePublicationError(
            f"Published {label} bytes changed after rename"
        )


def _write_atomic_file(
    path: Path,
    payload: bytes,
    *,
    pre_replace_gate: Callable[[], None] | None = None,
) -> None:
    _assert_no_symlink_components(path, "atomic file")
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(path, "atomic file")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_no_symlink_components(path, "atomic file")
        if pre_replace_gate is not None:
            pre_replace_gate()
        _assert_no_symlink_components(path, "atomic file")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_receipt_path(
    settings: SupportingProfileSettings, compiled: Mapping[str, Any]
) -> Path:
    return _lexical_absolute(
        settings.run_root
        / _assert_safe_component(compiled["binding"]["contract_id"], "contract_id")
        / _assert_safe_component(
            settings.config["outputs"]["quality_subdirectory"],
            "quality subdirectory",
        )
        / _assert_safe_component(compiled["build_id"], "build_id")
        / "build_receipt.json"
    )


def _final_pointer_object(
    documents: Mapping[str, Any], implementation: Mapping[str, str]
) -> tuple[dict[str, Any], bytes]:
    pointer = dict(documents["pointer"])
    pointer["implementation"] = dict(implementation)
    return pointer, canonical_json_bytes(pointer)


def _publish_run_receipt(
    settings: SupportingProfileSettings,
    compiled: Mapping[str, Any],
    final_pointer: Mapping[str, Any],
    final_pointer_bytes: bytes,
    quality_directory: Path,
    receipt_path: Path,
    implementation: Mapping[str, str],
    *,
    elapsed_seconds: float | None,
) -> Path:
    _assert_optional_regular_file(receipt_path, "run receipt")
    identity = {
        "build_id": compiled["build_id"],
        "contract_id": compiled["binding"]["contract_id"],
        "implementation": dict(implementation),
        "manifest_sha256": final_pointer["manifest_sha256"],
        "pointer_sha256": _sha256_bytes(final_pointer_bytes),
        "status": compiled["summary"]["status"],
    }
    if receipt_path.exists():
        try:
            receipt_payload = _read_authenticated_regular_file(
                receipt_path, "run receipt"
            )
            existing = json.loads(receipt_payload.decode("utf-8"))
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            SupportingProfilePublicationError,
        ) as exc:
            raise SupportingProfilePublicationError(
                "Existing run receipt is unreadable; primary pointer was not moved"
            ) from exc
        if any(existing.get(key) != value for key, value in identity.items()):
            raise SupportingProfilePublicationError(
                "Existing run receipt selects a different publication"
            )
        return receipt_path
    receipt = {
        "artifact_path": final_pointer["artifact_path"],
        **identity,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": round(max(0.0, float(elapsed_seconds or 0.0)), 6),
        "quality_path": quality_directory.as_posix(),
        "schema_version": 1,
    }
    _write_atomic_file(receipt_path, canonical_json_bytes(receipt))
    return receipt_path


def _final_pointer_commit_gate(
    settings: SupportingProfileSettings,
    compiled: Mapping[str, Any],
    quality_directory: Path,
    quality_payloads: Mapping[str, bytes],
    artifact_directory: Path,
    artifact_payloads: Mapping[str, bytes],
    documents: Mapping[str, Any],
    implementation: Mapping[str, str],
    pointer_path: Path,
    receipt_path: Path,
) -> None:
    _assert_locked_compiled_context(settings, compiled)
    _authenticate_external_context(settings, compiled)
    if _implementation_binding(settings, compiled) != dict(implementation):
        raise SupportingProfilePublicationError(
            "Publication implementation bundle changed before pointer commit"
        )
    rerendered = _validate_payload_names(
        render_quality_payloads(settings.config, compiled, settings=settings),
        settings.config["outputs"]["closed_quality_file_set"],
        "re-authenticated quality",
    )
    if rerendered != dict(quality_payloads):
        raise SupportingProfilePublicationError(
            "Re-authenticated quality bytes differ before pointer commit"
        )
    _assert_optional_regular_file(pointer_path, "primary pointer")
    _assert_optional_regular_file(receipt_path, "run receipt")
    _assert_existing_matches(quality_directory, quality_payloads, "quality")
    _assert_existing_matches(artifact_directory, artifact_payloads, "artifact")
    _validate_rendered_documents(
        settings,
        compiled,
        quality_directory,
        quality_payloads,
        artifact_directory,
        artifact_payloads,
        documents,
    )


def publish_supporting_profile(
    compiled: Mapping[str, Any],
    settings: SupportingProfileSettings | None = None,
    *,
    elapsed_seconds: float | None = None,
) -> dict[str, Path]:
    """Publish a passing compiled profile and commit its primary pointer last."""
    settings = settings or load_settings()
    _assert_allowed_roots(settings)
    _assert_locked_compiled_context(settings, compiled)
    implementation = _implementation_binding(settings, compiled)

    quality_directory = _quality_directory(settings, compiled)
    _assert_no_symlink_components(quality_directory, "quality destination")
    quality_payloads = _validate_payload_names(
        render_quality_payloads(settings.config, compiled, settings=settings),
        settings.config["outputs"]["closed_quality_file_set"],
        "quality",
    )
    documents = render_artifact_documents(
        settings, compiled, quality_directory, quality_payloads
    )
    artifact_subdirectory = _assert_safe_component(
        settings.config["outputs"]["artifact_subdirectory"],
        "artifact subdirectory",
    )
    artifact_directory = _lexical_absolute(Path(documents["artifact_directory"]))
    expected_artifact_directory = _lexical_absolute(
        settings.artifact_root
        / artifact_subdirectory
        / _assert_safe_component(compiled["build_id"], "build_id")
    )
    if artifact_directory != expected_artifact_directory:
        raise SupportingProfilePublicationError(
            "Artifact renderer selected an unexpected directory"
        )
    _assert_no_symlink_components(artifact_directory, "artifact destination")
    artifact_payloads = _validate_payload_names(
        documents["artifact_payloads"],
        settings.config["outputs"]["closed_artifact_file_set"],
        "artifact",
    )
    _validate_rendered_documents(
        settings,
        compiled,
        quality_directory,
        quality_payloads,
        artifact_directory,
        artifact_payloads,
        documents,
    )
    final_pointer, final_pointer_bytes = _final_pointer_object(
        documents, implementation
    )

    pointer_name = _assert_safe_component(
        settings.config["outputs"]["primary_pointer"], "primary pointer filename"
    )
    pointer_path = _lexical_absolute(settings.artifact_root / pointer_name)
    receipt_path = _run_receipt_path(settings, compiled)
    _assert_optional_regular_file(pointer_path, "primary pointer")
    _assert_optional_regular_file(receipt_path, "run receipt")
    _assert_existing_matches(quality_directory, quality_payloads, "quality")
    _assert_existing_matches(artifact_directory, artifact_payloads, "artifact")

    quality_stage: Path | None = None
    artifact_stage: Path | None = None
    try:
        quality_stage = _stage_payloads(
            quality_directory.parent, compiled["build_id"], quality_payloads
        )
        artifact_stage = _stage_payloads(
            artifact_directory.parent, compiled["build_id"], artifact_payloads
        )
        _publish_staged_directory(
            quality_stage, quality_directory, quality_payloads, "quality"
        )
        quality_stage = None
        _publish_staged_directory(
            artifact_stage, artifact_directory, artifact_payloads, "artifact"
        )
        artifact_stage = None
    finally:
        if quality_stage is not None:
            _remove_staging_entry(quality_stage)
        if artifact_stage is not None:
            _remove_staging_entry(artifact_stage)

    receipt_path = _publish_run_receipt(
        settings,
        compiled,
        final_pointer,
        final_pointer_bytes,
        quality_directory,
        receipt_path,
        implementation,
        elapsed_seconds=elapsed_seconds,
    )

    def commit_gate() -> None:
        _final_pointer_commit_gate(
            settings,
            compiled,
            quality_directory,
            quality_payloads,
            artifact_directory,
            artifact_payloads,
            documents,
            implementation,
            pointer_path,
            receipt_path,
        )

    commit_gate()
    _assert_optional_regular_file(pointer_path, "primary pointer")
    pointer_matches = pointer_path.exists() and (
        _read_authenticated_regular_file(pointer_path, "primary pointer")
        == final_pointer_bytes
    )
    if not pointer_matches:
        _write_atomic_file(
            pointer_path, final_pointer_bytes, pre_replace_gate=commit_gate
        )
    return {
        "artifact": artifact_directory,
        "pointer": pointer_path,
        "quality": quality_directory,
        "receipt": receipt_path,
    }


def build_supporting_profile(
    settings: SupportingProfileSettings | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    settings = settings or load_settings()
    started = time.monotonic()
    compiled = compile_supporting_profile(settings)
    paths = publish_supporting_profile(
        compiled, settings, elapsed_seconds=time.monotonic() - started
    )
    return compiled, paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-config", type=Path, default=DEFAULT_PROFILE_CONFIG)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--phase-config", type=Path, default=DEFAULT_PHASE_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--block-size-bytes", type=int, default=16 << 20)
    args = parser.parse_args(argv)
    settings = load_settings(
        args.profile_config,
        args.paths_config,
        args.phase_config,
        artifact_root=args.artifact_root,
        run_root=args.run_root,
        block_size_bytes=args.block_size_bytes,
    )
    compiled, paths = build_supporting_profile(settings)
    print(
        json.dumps(
            {
                "artifact_path": paths["artifact"].as_posix(),
                "build_id": compiled["build_id"],
                "quality_path": paths["quality"].as_posix(),
                "status": compiled["summary"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
