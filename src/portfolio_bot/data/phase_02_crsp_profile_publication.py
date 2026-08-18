"""Atomically publish the deterministic Phase 2 CRSP development profile.

All byte-affecting rendering lives in ``phase_02_crsp_development_profile``.
This module is intentionally limited to closed-set filesystem publication, the
mutable primary pointer, a non-immutable run receipt, and the command-line
entry point.
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
from typing import Any, Mapping, Sequence

import portfolio_bot.data.phase_02_crsp_development_profile as profile_compiler
from portfolio_bot.data.phase_02_crsp_development_profile import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_PATHS_CONFIG,
    DEFAULT_PHASE_CONFIG,
    DEFAULT_PROFILE_CONFIG,
    DEFAULT_RUN_ROOT,
    ProfileBuildError,
    ProfileSettings,
    canonical_json_bytes,
    compile_profile_build,
    load_settings,
    render_artifact_documents,
    render_quality_payloads,
)


LOCKED_PROFILE_CONFIG_SHA256 = (
    "645b6ff51480d4e3cbcad313e3488c5a8d45df272ca4af6e7010e870296b4c35"
)


class ProfilePublicationError(ProfileBuildError):
    """Raised when immutable profile publication cannot complete exactly."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute normalized path without dereferencing symlinks."""
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path, label: str) -> None:
    """Reject a symlink at the target or in any currently existing ancestor."""
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ProfilePublicationError(
                f"Cannot authenticate {label} path component {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise ProfilePublicationError(
                f"Symlink is prohibited in {label} path: {current}"
            )


def _assert_optional_regular_file(path: Path, label: str) -> None:
    _assert_no_symlink_components(path, label)
    if path.exists() and not path.is_file():
        raise ProfilePublicationError(f"Existing {label} is not a regular file")


def _quality_directory(settings: ProfileSettings, compiled: Mapping[str, Any]) -> Path:
    return _lexical_absolute(
        settings.quality_root
        / compiled["binding"]["contract_id"]
        / settings.config["outputs"]["quality_subdirectory"]
        / compiled["build_id"]
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
        raise ProfilePublicationError(f"Unsafe {label} path component: {component!r}")
    return component


def _assert_allowed_roots(settings: ProfileSettings) -> None:
    database_root = _lexical_absolute(settings.database_root)
    quality_root = _lexical_absolute(settings.quality_root)
    project_root = _lexical_absolute(settings.project_root)
    _assert_no_symlink_components(database_root, "database root")
    _assert_no_symlink_components(quality_root, "quality root")
    _assert_no_symlink_components(project_root, "project root")
    database_resolved = database_root.resolve(strict=False)
    try:
        quality_relative = quality_root.resolve(strict=False).relative_to(
            database_resolved
        )
    except ValueError as exc:
        raise ProfilePublicationError(
            "The quality root is outside the configured database root"
        ) from exc
    if not quality_relative.parts or quality_relative.parts[0] != "quality":
        raise ProfilePublicationError("Profile outputs must be rooted under quality/")
    for label, root in (
        ("artifact", settings.artifact_root),
        ("run", settings.run_root),
    ):
        lexical_root = _lexical_absolute(root)
        _assert_no_symlink_components(lexical_root, f"{label} root")
        if label == "artifact":
            try:
                lexical_root.resolve(strict=False).relative_to(
                    project_root.resolve(strict=False)
                )
            except ValueError as exc:
                raise ProfilePublicationError(
                    "Artifact root is outside the configured project root"
                ) from exc
        try:
            relative = lexical_root.resolve(strict=False).relative_to(database_resolved)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] in {"raw", "interim"}:
            raise ProfilePublicationError(
                f"The {label} root may not write below database/{relative.parts[0]}"
            )


def _assert_locked_context(
    settings: ProfileSettings, compiled: Mapping[str, Any]
) -> None:
    actual_config_sha256 = _sha256_bytes(settings.profile_config_path.read_bytes())
    if actual_config_sha256 != LOCKED_PROFILE_CONFIG_SHA256:
        raise ProfilePublicationError(
            "Profile config differs from the locked pre-publication contract"
        )
    with settings.profile_config_path.open("rb") as handle:
        fresh_config = tomllib.load(handle)
    if fresh_config != settings.config:
        raise ProfilePublicationError(
            "In-memory profile settings differ from freshly parsed locked config"
        )
    try:
        binding_sha256 = compiled["binding"]["config_sha256"]
        build_id = _assert_safe_component(compiled["build_id"], "build_id")
        contract_id = _assert_safe_component(
            compiled["binding"]["contract_id"], "contract_id"
        )
    except (KeyError, TypeError) as exc:
        raise ProfilePublicationError("Compiled profile lacks its binding identity") from exc
    if binding_sha256 != actual_config_sha256:
        raise ProfilePublicationError(
            "Compiled profile binding does not select the locked config bytes"
        )
    if compiled["summary"].get("build_id") != build_id:
        raise ProfilePublicationError("Summary build_id differs from compiled build_id")
    if not contract_id:
        raise ProfilePublicationError("Compiled profile contract_id is blank")
    required_status = settings.config["outputs"]["serialization"][
        "published_primary_status"
    ]
    if compiled["summary"].get("status") != required_status:
        raise ProfilePublicationError(
            "Only a passing primary profile may enter immutable publication"
        )
    if int(compiled["summary"].get("fail_count", -1)) != 0:
        raise ProfilePublicationError("A profile with failed checks may not be published")


def _assert_bound_implementation_fresh(compiled: Mapping[str, Any]) -> None:
    try:
        expected = compiled["binding"]["implementation_sha256"]
    except (KeyError, TypeError) as exc:
        raise ProfilePublicationError(
            "Compiled profile lacks its implementation binding"
        ) from exc
    implementation_path = Path(profile_compiler.__file__).resolve()
    actual = _sha256_bytes(implementation_path.read_bytes())
    if actual != expected:
        raise ProfilePublicationError(
            "Primary implementation bytes changed before pointer commit"
        )


def _validate_payload_names(
    payloads: Mapping[str, bytes], expected_names: Sequence[str], label: str
) -> dict[str, bytes]:
    expected = set(expected_names)
    actual = set(payloads)
    if actual != expected:
        raise ProfilePublicationError(
            f"{label} closed file set differs: expected {sorted(expected)}, "
            f"actual {sorted(actual)}"
        )
    validated: dict[str, bytes] = {}
    for name in sorted(payloads):
        if Path(name).name != name or name in {".", ".."}:
            raise ProfilePublicationError(f"Unsafe {label} filename: {name!r}")
        payload = payloads[name]
        if not isinstance(payload, bytes):
            raise ProfilePublicationError(f"{label} payload {name!r} is not bytes")
        validated[name] = payload
    return validated


def _validate_rendered_paths_and_bindings(
    settings: ProfileSettings,
    compiled: Mapping[str, Any],
    quality_directory: Path,
    quality_payloads: Mapping[str, bytes],
    artifact_directory: Path,
    artifact_payloads: Mapping[str, bytes],
    documents: Mapping[str, Any],
) -> None:
    """Double-check that rendered documents select only the exact destinations."""
    database_root = _lexical_absolute(settings.database_root)
    try:
        quality_relative = quality_directory.relative_to(database_root)
    except ValueError as exc:
        raise ProfilePublicationError(
            "Exact quality destination is outside the configured database root"
        ) from exc
    if not quality_relative.parts or quality_relative.parts[0] != "quality":
        raise ProfilePublicationError("Quality inventory paths must begin with quality/")
    expected_outputs = [
        {
            "path": (quality_relative / name).as_posix(),
            "sha256": _sha256_bytes(quality_payloads[name]),
            "size_bytes": len(quality_payloads[name]),
        }
        for name in sorted(quality_payloads)
    ]
    manifest = documents.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ProfilePublicationError("Rendered manifest object is missing")
    expected_manifest = {
        "binding": compiled["binding"],
        "build_id": compiled["build_id"],
        "calendar_build_id": compiled["binding"]["calendar_build_id"],
        "contract_id": compiled["binding"]["contract_id"],
        "outputs": expected_outputs,
        "profile_name": settings.config["profile_name"],
        "schema_version": int(settings.config["schema_version"]),
        "status": compiled["summary"]["status"],
        "summary": compiled["summary"],
    }
    expected_manifest_bytes = canonical_json_bytes(expected_manifest)
    if canonical_json_bytes(dict(manifest)) != expected_manifest_bytes:
        raise ProfilePublicationError(
            "Rendered manifest differs from the exact authenticated manifest"
        )
    manifest_bytes = artifact_payloads.get("manifest.json")
    if manifest_bytes != expected_manifest_bytes:
        raise ProfilePublicationError("Rendered manifest bytes differ from its object")
    expected_inventory = "".join(
        f"{item['sha256']}  {item['path']}\n" for item in expected_outputs
    ).encode("utf-8")
    if artifact_payloads.get("output_hashes.sha256") != expected_inventory:
        raise ProfilePublicationError(
            "Rendered output inventory differs from the exact quality payloads"
        )
    pointer = documents.get("pointer")
    if not isinstance(pointer, Mapping):
        raise ProfilePublicationError("Rendered primary pointer object is missing")
    try:
        artifact_label = artifact_directory.relative_to(
            _lexical_absolute(settings.project_root)
        ).as_posix()
    except ValueError as exc:
        raise ProfilePublicationError(
            "Artifact root is unsupported because artifact_path is not project-root-relative"
        ) from exc
    artifact_posix = PurePosixPath(str(artifact_label))
    if (
        artifact_posix.is_absolute()
        or not artifact_posix.parts
        or any(part in {"", ".", ".."} for part in artifact_posix.parts)
        or "\\" in str(artifact_label)
    ):
        raise ProfilePublicationError(
            "Artifact root is unsupported because artifact_path is not project-root-relative"
        )
    expected_pointer = {
        "artifact_path": artifact_label,
        "build_id": compiled["build_id"],
        "calendar_build_id": compiled["binding"]["calendar_build_id"],
        "contract_id": compiled["binding"]["contract_id"],
        "manifest_sha256": _sha256_bytes(expected_manifest_bytes),
        "schema_version": int(settings.config["schema_version"]),
        "status": compiled["summary"]["status"],
    }
    expected_pointer_bytes = canonical_json_bytes(expected_pointer)
    if canonical_json_bytes(dict(pointer)) != expected_pointer_bytes:
        raise ProfilePublicationError(
            "Rendered pointer differs from the exact authenticated primary pointer"
        )
    pointer_bytes = documents.get("pointer_bytes")
    if pointer_bytes != expected_pointer_bytes:
        raise ProfilePublicationError("Rendered pointer bytes differ from its object")


def _read_closed_directory(directory: Path, label: str) -> dict[str, bytes]:
    _assert_no_symlink_components(directory, label)
    if not directory.is_dir():
        raise ProfilePublicationError(f"Existing {label} path is not a real directory")
    payloads: dict[str, bytes] = {}
    for entry in sorted(directory.iterdir(), key=lambda path: path.name):
        if entry.is_symlink() or not entry.is_file():
            raise ProfilePublicationError(
                f"Existing {label} directory contains a non-regular entry: {entry.name}"
            )
        payloads[entry.name] = entry.read_bytes()
    return payloads


def _assert_existing_matches(
    final: Path, payloads: Mapping[str, bytes], label: str
) -> bool:
    _assert_no_symlink_components(final, label)
    if not final.exists() and not final.is_symlink():
        return False
    actual = _read_closed_directory(final, label)
    if actual != dict(payloads):
        expected_hashes = {
            name: _sha256_bytes(payload) for name, payload in payloads.items()
        }
        actual_hashes = {name: _sha256_bytes(payload) for name, payload in actual.items()}
        raise ProfilePublicationError(
            f"Existing {label} directory differs from deterministic payload: "
            f"expected {expected_hashes}, actual {actual_hashes}"
        )
    return True


def _remove_staging_entry(stage: Path) -> None:
    """Remove only the staging entry itself; never follow a replacement symlink."""
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
            raise ProfilePublicationError("Short write while staging immutable payload")
        remaining = remaining[written:]


def _read_anchored_payloads(directory_descriptor: int) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    names = sorted(os.listdir(directory_descriptor))
    for name in names:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ProfilePublicationError(
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
            raise ProfilePublicationError(
                "Staging path identity changed before anchored writes"
            )
        for name, payload in payloads.items():
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
            raise ProfilePublicationError("Staged bytes differ immediately after writing")
        final_path_metadata = stage.lstat()
        if (
            final_path_metadata.st_dev != descriptor_metadata.st_dev
            or final_path_metadata.st_ino != descriptor_metadata.st_ino
            or not stat.S_ISDIR(final_path_metadata.st_mode)
        ):
            raise ProfilePublicationError(
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
        # A concurrent identical publisher may have won the rename race.
        if _assert_existing_matches(final, payloads, label):
            _remove_staging_entry(stage)
            return
        raise
    if _read_closed_directory(final, label) != dict(payloads):
        raise ProfilePublicationError(f"Published {label} bytes changed after rename")


def _final_pointer_commit_gate(
    settings: ProfileSettings,
    compiled: Mapping[str, Any],
    quality_directory: Path,
    quality_payloads: Mapping[str, bytes],
    artifact_directory: Path,
    artifact_payloads: Mapping[str, bytes],
    documents: Mapping[str, Any],
    pointer_path: Path,
    receipt_path: Path,
) -> None:
    """Jointly re-authenticate every pointer dependency at the commit boundary."""
    _assert_locked_context(settings, compiled)
    reauthenticated_quality = _validate_payload_names(
        render_quality_payloads(settings.config, compiled, settings=settings),
        settings.config["outputs"]["closed_quality_file_set"],
        "re-authenticated quality",
    )
    if reauthenticated_quality != dict(quality_payloads):
        raise ProfilePublicationError(
            "Re-authenticated quality payload differs before pointer commit"
        )
    _assert_bound_implementation_fresh(compiled)
    _assert_optional_regular_file(pointer_path, "primary pointer")
    _assert_optional_regular_file(receipt_path, "run receipt")
    _assert_existing_matches(quality_directory, quality_payloads, "quality")
    _assert_existing_matches(artifact_directory, artifact_payloads, "artifact")
    _validate_rendered_paths_and_bindings(
        settings,
        compiled,
        quality_directory,
        quality_payloads,
        artifact_directory,
        artifact_payloads,
        documents,
    )


def _write_atomic_file(path: Path, payload: bytes) -> None:
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
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_receipt_path(
    settings: ProfileSettings, compiled: Mapping[str, Any]
) -> Path:
    return _lexical_absolute(
        settings.run_root
        / compiled["binding"]["contract_id"]
        / settings.config["outputs"]["quality_subdirectory"]
        / compiled["build_id"]
        / "build_receipt.json"
    )


def _publish_run_receipt(
    settings: ProfileSettings,
    compiled: Mapping[str, Any],
    documents: Mapping[str, Any],
    quality_directory: Path,
    receipt_path: Path,
    *,
    elapsed_seconds: float | None,
) -> Path:
    contract_id = compiled["binding"]["contract_id"]
    build_id = compiled["build_id"]
    _assert_optional_regular_file(receipt_path, "run receipt")
    if receipt_path.exists():
        try:
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfilePublicationError(
                "Existing run receipt is unreadable; primary pointer was not moved"
            ) from exc
        expected_identity = {
            "build_id": build_id,
            "contract_id": contract_id,
            "manifest_sha256": documents["pointer"]["manifest_sha256"],
            "status": compiled["summary"]["status"],
        }
        if any(existing.get(key) != value for key, value in expected_identity.items()):
            raise ProfilePublicationError(
                "Existing run receipt selects a different immutable publication"
            )
        return receipt_path
    receipt = {
        "artifact_path": documents["pointer"]["artifact_path"],
        "build_id": build_id,
        "calendar_build_id": compiled["binding"]["calendar_build_id"],
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract_id": contract_id,
        "elapsed_seconds": round(max(0.0, float(elapsed_seconds or 0.0)), 6),
        "manifest_sha256": documents["pointer"]["manifest_sha256"],
        "quality_path": quality_directory.as_posix(),
        "schema_version": 1,
        "status": compiled["summary"]["status"],
    }
    _write_atomic_file(receipt_path, canonical_json_bytes(receipt))
    return receipt_path


def publish_profile_build(
    compiled: Mapping[str, Any],
    settings: ProfileSettings | None = None,
    *,
    elapsed_seconds: float | None = None,
) -> dict[str, Path]:
    """Publish a passing compiled profile without transforming rendered bytes."""
    settings = settings or load_settings()
    _assert_allowed_roots(settings)
    _assert_locked_context(settings, compiled)
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
    artifact_directory = _lexical_absolute(Path(documents["artifact_directory"]))
    expected_artifact_directory = _lexical_absolute(
        settings.artifact_root
        / settings.config["outputs"]["artifact_subdirectory"]
        / compiled["build_id"]
    )
    if artifact_directory != expected_artifact_directory:
        raise ProfilePublicationError("Artifact renderer selected an unexpected directory")
    _assert_no_symlink_components(artifact_directory, "artifact destination")
    artifact_payloads = _validate_payload_names(
        documents["artifact_payloads"],
        settings.config["outputs"]["closed_sets"]["artifact_build_files"],
        "artifact",
    )
    _validate_rendered_paths_and_bindings(
        settings,
        compiled,
        quality_directory,
        quality_payloads,
        artifact_directory,
        artifact_payloads,
        documents,
    )
    pointer_name = settings.config["outputs"]["primary_pointer"]
    if Path(pointer_name).name != pointer_name:
        raise ProfilePublicationError("Primary pointer filename is not a safe basename")
    pointer_path = _lexical_absolute(settings.artifact_root / pointer_name)
    receipt_path = _run_receipt_path(settings, compiled)
    # These mutable endpoints are authenticated before immutable staging so a
    # symlink cannot cause a late follow-on read or write outside the run.
    _assert_optional_regular_file(pointer_path, "primary pointer")
    _assert_optional_regular_file(receipt_path, "run receipt")
    # Preflight immutable destinations before creating any staging directory.
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
        documents,
        quality_directory,
        receipt_path,
        elapsed_seconds=elapsed_seconds,
    )
    _final_pointer_commit_gate(
        settings,
        compiled,
        quality_directory,
        quality_payloads,
        artifact_directory,
        artifact_payloads,
        documents,
        pointer_path,
        receipt_path,
    )
    pointer_bytes = documents["pointer_bytes"]
    if not isinstance(pointer_bytes, bytes):
        raise ProfilePublicationError("Rendered primary pointer is not bytes")
    _assert_optional_regular_file(pointer_path, "primary pointer")
    if not pointer_path.exists() or pointer_path.read_bytes() != pointer_bytes:
        _write_atomic_file(pointer_path, pointer_bytes)
    return {
        "artifact": artifact_directory,
        "pointer": pointer_path,
        "quality": quality_directory,
        "receipt": receipt_path,
    }


def build_profile(
    settings: ProfileSettings | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Compile, then publish; compilation failure cannot move the pointer."""
    settings = settings or load_settings()
    started = time.monotonic()
    compiled = compile_profile_build(settings)
    paths = publish_profile_build(
        compiled, settings, elapsed_seconds=time.monotonic() - started
    )
    return compiled, paths


def main(argv: Sequence[str] | None = None) -> int:
    """Run the primary CRSP development-profile build."""
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
    compiled, paths = build_profile(settings)
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
