"""Publish a locked Phase 2 CRSP numeric/unit profile safely.

The numeric compiler is deliberately not imported here.  It owns all semantic
work and rendering; this module accepts its already-rendered deterministic
bytes, authenticates their complete external snapshot and implementation
bindings, installs two immutable closed directories, and replaces the mutable
primary pointer last.

No function in this module can write raw, interim, processed, cache, canonical
market, feature, target, portfolio, return, or performance data.
"""

from __future__ import annotations

import csv
import copy
import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import tomllib
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence


SCHEMA_VERSION = 1
PROFILE_NAME = "crsp_development_numeric_unit_profile"
PUBLISHED_STATUS = "numeric_unit_primary_passed_pending_independent_verification"
LOCKED_PROFILE_STATUS = "locked_after_aggregate_discovery_before_primary_profile"

QUALITY_FILES = (
    "_SUCCESS.json",
    "capitalization_diagnostics.csv",
    "checks.csv",
    "downstream_gate_status.json",
    "identity_inventory.csv",
    "input_receipt.json",
    "numeric_inventory.csv",
    "profile_contract.json",
    "return_inventory.csv",
    "summary.json",
)
ARTIFACT_FILES = ("manifest.json", "output_hashes.sha256")
QUALITY_SUBDIRECTORY = "crsp_numeric_unit_profile"
ARTIFACT_SUBDIRECTORY = "crsp_numeric_unit_profiles"
PRIMARY_POINTER = "LATEST_CRSP_NUMERIC_UNIT_PRIMARY.json"

_BUILD_ID_RE = re.compile(r"crsp-numeric-[0-9a-f]{24}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SNAPSHOT_DOMAIN = b"phase02-crsp-numeric-unit-snapshots-v1\x00"
_RAW_PATHS = {
    "crsp_daily_ciz.csv",
    "crsp_monthly_ciz.csv",
    "crsp_shares_outstanding_ciz.csv",
}
PRODUCTION_LOCK_KEYS = frozenset(
    {
        "daily_numeric_expectations_locked",
        "daily_return_cross_tabs_locked",
        "daily_price_volume_expectations_locked",
        "monthly_numeric_expectations_locked",
        "monthly_observation_date_expectations_locked",
        "monthly_crosscheck_expectations_locked",
        "shares_join_expectations_locked",
        "capitalization_diagnostics_locked",
        "non_ADR_scale_population_locked",
        "non_ADR_scale_threshold_locked",
        "all_required_locks_true",
    }
)
_SUMMARY_COUNT_FIELDS = (
    "check_count",
    "pass_count",
    "fail_count",
    "diagnostic_count",
    "hard_failure_count",
)
_AGGREGATE_TABLES = frozenset(
    {
        "checks",
        "numeric_inventory",
        "return_inventory",
        "identity_inventory",
        "capitalization_diagnostics",
    }
)
CHECKS_FIELDS = (
    "check_id",
    "stage",
    "family",
    "source",
    "stratum",
    "severity",
    "expected",
    "actual",
    "tolerance_rule",
    "status",
    "downstream_consequence",
)
_CHECK_STATUSES = frozenset({"PASS", "FAIL", "DIAGNOSTIC"})
_CHECK_SEVERITIES = frozenset({"hard", "diagnostic"})

_INVENTORY_FILES = {
    "capitalization_diagnostics": "capitalization_diagnostics.csv",
    "identity_inventory": "identity_inventory.csv",
    "numeric_inventory": "numeric_inventory.csv",
    "return_inventory": "return_inventory.csv",
}
_MEASUREMENT_EXCLUDED_FIELDS = {
    "numeric_inventory": frozenset({"status", "severity"}),
    "return_inventory": frozenset({"semantic_rule", "status", "severity"}),
    "identity_inventory": frozenset({"severity", "status", "tolerance_rule"}),
    "capitalization_diagnostics": frozenset(
        {"severity", "status", "threshold_rule"}
    ),
}
_NONNEGATIVE_INTEGER_FIELDS = {
    "numeric_inventory": frozenset(
        {
            "year",
            "population_rows",
            "blank_rows",
            "valid_in_domain_rows",
            "valid_out_of_domain_rows",
            "malformed_rows",
            "nonfinite_rows",
            "overflow_rows",
            "negative_zero_rows",
        }
    ),
    "return_inventory": frozenset({"year", "population_rows"}),
    "identity_inventory": frozenset(
        {
            "year",
            "population_rows",
            "comparable_rows",
            "pass_rows",
            "fail_rows",
            "excluded_rows",
        }
    ),
    "capitalization_diagnostics": frozenset(
        {
            "year",
            "security_line_count",
            "population_rows",
            "comparable_rows",
            "excluded_rows",
            "zero_share_match_rows",
            "one_share_match_rows",
            "multiple_share_match_rows",
            "formula_pass_rows",
            "formula_fail_rows",
        }
    ),
}
_OPTIONAL_SIGNED_INTEGER_FIELDS = {
    "return_inventory": frozenset({"mapped_trading_periods"})
}
_OPTIONAL_NONNEGATIVE_INTEGER_FIELDS = {
    "return_inventory": frozenset({"coherent_rows", "incoherent_rows"})
}
_NON_ADR_COUNT_FIELDS = (
    "population_rows",
    "comparable_rows",
    "excluded_rows",
    "zero_share_match_rows",
    "one_share_match_rows",
    "multiple_share_match_rows",
    "formula_pass_rows",
    "formula_fail_rows",
)

REQUIRED_EMPIRICAL_PROJECTION_CHECK_IDS = frozenset(
    {
        f"empirical_{table}_measurement_projection_{segment}"
        for table in (
            "capitalization_diagnostics",
            "identity_inventory",
            "numeric_inventory",
            "return_inventory",
        )
        for segment in (
            "full",
            "development_2000_2014_11",
            "embargo_2014_12",
        )
    }
)
REQUIRED_PRODUCTION_CHECK_IDS = frozenset(
    {
        "capitalization_diagnostics_hard_failures",
        "identity_pooled_active_segment_zero_comparable_groups",
        "identity_active_segment_population_conservation",
        "identity_inventory_hard_failures",
        "non_adr_capitalization_segment_counts",
        "non_adr_capitalization_theory_threshold",
        "numeric_inventory_hard_failures",
        "production_profile_status",
        "return_inventory_hard_failures",
        "source_daily_canonical_rows",
        "source_daily_full_rows",
        "source_daily_retained_rows",
        "source_monthly_canonical_rows",
        "source_monthly_full_rows",
        "source_monthly_retained_rows",
        "source_shares_full_rows",
        "source_shares_retained_rows",
    }
) | REQUIRED_EMPIRICAL_PROJECTION_CHECK_IDS


class NumericUnitPublicationError(RuntimeError):
    """Raised when a numeric/unit primary publication is not provably safe."""


class NumericUnitPublicationSettings(Protocol):
    """Structural settings contract expected from the future compiler."""

    project_root: Path
    database_root: Path
    raw_root: Path
    quality_root: Path
    artifact_root: Path
    run_root: Path
    profile_config_path: Path
    config: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class AuthenticatedSnapshot:
    """Portable description of one immutable publication input.

    ``relative_path`` is resolved only below ``project_root`` or ``raw_root``.
    The descriptor itself, rather than an absolute local path or file stat, is
    bound into the compiler's ``snapshot_inventory_sha256``.
    """

    snapshot_id: str
    root: str
    relative_path: str
    sha256: str
    size_bytes: int


def canonical_json_bytes(value: Any) -> bytes:
    """Render the canonical JSON representation used by Phase 2 artifacts."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise NumericUnitPublicationError(
            f"Value is not canonical JSON: {exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _assert_safe_component(value: Any, label: str) -> str:
    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or Path(component).name != component
        or "/" in component
        or "\\" in component
    ):
        raise NumericUnitPublicationError(
            f"Unsafe {label} path component: {component!r}"
        )
    return component


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
            raise NumericUnitPublicationError(
                f"Cannot authenticate {label} path component"
            ) from exc
        if stat.S_ISLNK(mode):
            raise NumericUnitPublicationError(
                f"Symlink is prohibited in {label} path: {current}"
            )


def _relative_to(path: Path, root: Path, label: str) -> Path:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise NumericUnitPublicationError(
            f"{label} is outside its allowed root"
        ) from exc


def _safe_relative_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
    ):
        raise NumericUnitPublicationError(f"Unsafe {label}: {value!r}")
    return path


def _assert_allowed_roots(settings: NumericUnitPublicationSettings) -> None:
    project_root = _lexical_absolute(settings.project_root)
    database_root = _lexical_absolute(settings.database_root)
    raw_root = _lexical_absolute(settings.raw_root)
    quality_root = _lexical_absolute(settings.quality_root)
    artifact_root = _lexical_absolute(settings.artifact_root)
    run_root = _lexical_absolute(settings.run_root)
    for label, root in (
        ("project root", project_root),
        ("database root", database_root),
        ("raw root", raw_root),
        ("quality root", quality_root),
        ("artifact root", artifact_root),
        ("run root", run_root),
    ):
        _assert_no_symlink_components(root, label)

    raw_relative = _relative_to(raw_root, database_root, "raw root")
    if not raw_relative.parts or raw_relative.parts[0] != "raw":
        raise NumericUnitPublicationError(
            "Numeric-unit raw reads must be rooted under database/raw/"
        )
    quality_relative = _relative_to(quality_root, database_root, "quality root")
    if not quality_relative.parts or quality_relative.parts[0] != "quality":
        raise NumericUnitPublicationError(
            "Numeric-unit database writes must be rooted under database/quality/"
        )
    _relative_to(artifact_root, project_root, "artifact root")
    _relative_to(run_root, project_root, "run root")
    for label, root in (("artifact root", artifact_root), ("run root", run_root)):
        try:
            database_relative = root.resolve(strict=False).relative_to(
                database_root.resolve(strict=False)
            )
        except ValueError:
            continue
        if not database_relative.parts or database_relative.parts[0] != "quality":
            raise NumericUnitPublicationError(
                f"{label} may not write outside database/quality/ inside the database"
            )


def _authenticated_file_stat(path: Path, label: str) -> os.stat_result:
    _assert_no_symlink_components(path, label)
    try:
        value = path.lstat()
    except OSError as exc:
        raise NumericUnitPublicationError(
            f"Cannot stat authenticated {label}"
        ) from exc
    if not stat.S_ISREG(value.st_mode):
        raise NumericUnitPublicationError(
            f"Authenticated {label} is not a regular file"
        )
    return value


def _hash_authenticated_regular_file(path: Path, label: str) -> tuple[str, int]:
    """Stream a no-follow file hash and reject any operational identity change."""

    path = _lexical_absolute(path)
    before = _authenticated_file_stat(path, label)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise NumericUnitPublicationError(
            f"Cannot open authenticated {label}"
        ) from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise NumericUnitPublicationError(
                f"Authenticated {label} is not the selected regular file"
            )
        while chunk := os.read(descriptor, 8 << 20):
            digest.update(chunk)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = _authenticated_file_stat(path, label)
    stable = (
        opened.st_dev == closed.st_dev == after.st_dev
        and opened.st_ino == closed.st_ino == after.st_ino
        and opened.st_size == closed.st_size == after.st_size
        and opened.st_mtime_ns == closed.st_mtime_ns == after.st_mtime_ns
    )
    if not stable:
        raise NumericUnitPublicationError(
            f"Authenticated {label} changed while it was being hashed"
        )
    return digest.hexdigest(), int(closed.st_size)


def _read_authenticated_regular_file(path: Path, label: str) -> bytes:
    path = _lexical_absolute(path)
    before = _authenticated_file_stat(path, label)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise NumericUnitPublicationError(
            f"Cannot open authenticated {label}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise NumericUnitPublicationError(
                f"Authenticated {label} is not the selected regular file"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = _authenticated_file_stat(path, label)
    if not (
        opened.st_dev == closed.st_dev == after.st_dev
        and opened.st_ino == closed.st_ino == after.st_ino
        and opened.st_size == closed.st_size == after.st_size
        and opened.st_mtime_ns == closed.st_mtime_ns == after.st_mtime_ns
    ):
        raise NumericUnitPublicationError(
            f"Authenticated {label} changed while it was being read"
        )
    return b"".join(chunks)


def _snapshot_descriptors(
    snapshots: Sequence[AuthenticatedSnapshot],
) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    ids: set[str] = set()
    locations: set[tuple[str, str]] = set()
    raw_paths: set[str] = set()
    for snapshot in snapshots:
        if (
            not snapshot.snapshot_id
            or any(ord(character) < 32 for character in snapshot.snapshot_id)
        ):
            raise NumericUnitPublicationError("Snapshot id is blank or contains control bytes")
        if snapshot.snapshot_id in ids:
            raise NumericUnitPublicationError("Snapshot inventory contains duplicate ids")
        ids.add(snapshot.snapshot_id)
        if snapshot.root not in {"project", "raw"}:
            raise NumericUnitPublicationError("Snapshot root must be project or raw")
        relative = _safe_relative_path(snapshot.relative_path, "snapshot path")
        location = (snapshot.root, relative.as_posix())
        if location in locations:
            raise NumericUnitPublicationError(
                "Snapshot inventory contains duplicate file locations"
            )
        locations.add(location)
        if _SHA256_RE.fullmatch(snapshot.sha256) is None:
            raise NumericUnitPublicationError("Snapshot SHA-256 is malformed")
        if isinstance(snapshot.size_bytes, bool) or snapshot.size_bytes < 0:
            raise NumericUnitPublicationError("Snapshot size is invalid")
        if snapshot.root == "raw":
            if len(relative.parts) != 1:
                raise NumericUnitPublicationError(
                    "Raw numeric-unit snapshots must be direct raw filenames"
                )
            raw_paths.add(relative.as_posix())
        descriptors.append(
            {
                "relative_path": relative.as_posix(),
                "root": snapshot.root,
                "sha256": snapshot.sha256,
                "size_bytes": int(snapshot.size_bytes),
                "snapshot_id": snapshot.snapshot_id,
            }
        )
    if raw_paths != _RAW_PATHS:
        raise NumericUnitPublicationError(
            "Numeric-unit publication must bind exactly daily, monthly, and shares raw sources"
        )
    return sorted(descriptors, key=lambda item: item["snapshot_id"])


def snapshot_inventory_sha256(
    snapshots: Sequence[AuthenticatedSnapshot],
) -> str:
    """Return the portable hash the compiler must place in its binding."""

    digest = hashlib.sha256()
    digest.update(_SNAPSHOT_DOMAIN)
    digest.update(canonical_json_bytes(_snapshot_descriptors(snapshots)))
    return digest.hexdigest()


def _configured_snapshot_expectations(config: Mapping[str, Any]) -> dict[str, tuple[str, str, str, int | None]]:
    """Return mandatory snapshot id -> (root, path, sha, optional size)."""

    try:
        result: dict[str, tuple[str, str, str, int | None]] = {
            "policy": (
                "project",
                str(config["policy"]["path"]),
                str(config["policy"]["sha256"]),
                None,
            ),
            "empirical_lock": (
                "project",
                str(config["empirical_lock"]["path"]),
                str(config["empirical_lock"]["sha256"]),
                None,
            ),
            "methodology_freeze": (
                "project",
                str(config["upstream"]["methodology"]["freeze_manifest"]),
                str(config["upstream"]["methodology"]["freeze_manifest_sha256"]),
                None,
            ),
            "phase_01_freeze": (
                "project",
                str(config["upstream"]["phase_01"]["freeze_manifest"]),
                str(config["upstream"]["phase_01"]["freeze_manifest_sha256"]),
                None,
            ),
            "phase_02_config": (
                "project",
                str(config["upstream"]["phase_02"]["phase_config"]),
                str(config["upstream"]["phase_02"]["phase_config_sha256"]),
                None,
            ),
            "crsp_backbone_contract": (
                "project",
                str(config["upstream"]["phase_02"]["crsp_backbone_contract"]),
                str(config["upstream"]["phase_02"]["crsp_backbone_contract_sha256"]),
                None,
            ),
            "contract_verified_pointer": (
                "project",
                str(config["upstream"]["contract"]["verified_pointer"]),
                str(config["upstream"]["contract"]["verified_pointer_sha256"]),
                None,
            ),
            "calendar_verified_pointer": (
                "project",
                str(config["upstream"]["calendar"]["verified_pointer"]),
                str(config["upstream"]["calendar"]["verified_pointer_sha256"]),
                None,
            ),
            "daily_monthly_profile_verified_pointer": (
                "project",
                str(config["upstream"]["daily_monthly_profile"]["verified_pointer"]),
                str(config["upstream"]["daily_monthly_profile"]["verified_pointer_sha256"]),
                None,
            ),
            "supporting_profile_verified_pointer": (
                "project",
                str(config["upstream"]["supporting_profile"]["verified_pointer"]),
                str(config["upstream"]["supporting_profile"]["verified_pointer_sha256"]),
                None,
            ),
        }
        for source_name, snapshot_id in (
            ("daily", "raw_daily"),
            ("monthly", "raw_monthly"),
            ("shares", "raw_shares"),
        ):
            source = config["sources"][source_name]
            result[snapshot_id] = (
                "raw",
                str(source["filename"]),
                str(source["sha256"]),
                int(source["size_bytes"]),
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise NumericUnitPublicationError(
            "Locked profile config lacks its required snapshot bindings"
        ) from exc
    return result


def _validate_snapshot_contract(
    config: Mapping[str, Any], snapshots: Sequence[AuthenticatedSnapshot]
) -> list[dict[str, Any]]:
    descriptors = _snapshot_descriptors(snapshots)
    by_id = {item["snapshot_id"]: item for item in descriptors}
    for snapshot_id, expected in _configured_snapshot_expectations(config).items():
        actual = by_id.get(snapshot_id)
        if actual is None:
            raise NumericUnitPublicationError(
                f"Snapshot inventory lacks required input {snapshot_id}"
            )
        root, path, sha256, size = expected
        if (
            actual["root"] != root
            or actual["relative_path"] != path
            or actual["sha256"] != sha256
            or (size is not None and actual["size_bytes"] != size)
        ):
            raise NumericUnitPublicationError(
                f"Snapshot {snapshot_id} differs from the locked profile config"
            )
    return descriptors


def _authenticate_snapshots(
    settings: NumericUnitPublicationSettings,
    config: Mapping[str, Any],
    snapshots: Sequence[AuthenticatedSnapshot],
    expected_inventory_sha256: str,
) -> None:
    descriptors = _validate_snapshot_contract(config, snapshots)
    if snapshot_inventory_sha256(snapshots) != expected_inventory_sha256:
        raise NumericUnitPublicationError(
            "Snapshot inventory differs from the compiled binding"
        )
    project_root = _lexical_absolute(settings.project_root)
    raw_root = _lexical_absolute(settings.raw_root)
    for descriptor in descriptors:
        root = project_root if descriptor["root"] == "project" else raw_root
        relative = PurePosixPath(descriptor["relative_path"])
        path = _lexical_absolute(root / Path(*relative.parts))
        _relative_to(path, root, f"snapshot {descriptor['snapshot_id']}")
        actual_sha, actual_size = _hash_authenticated_regular_file(
            path, f"snapshot {descriptor['snapshot_id']}"
        )
        if (
            actual_sha != descriptor["sha256"]
            or actual_size != descriptor["size_bytes"]
        ):
            raise NumericUnitPublicationError(
                f"Snapshot {descriptor['snapshot_id']} bytes differ"
            )


def _authenticate_calendar_chain(
    settings: NumericUnitPublicationSettings,
    config: Mapping[str, Any],
) -> None:
    """Re-authenticate the verified pointer, manifest, and calendar bytes.

    The calendar CSV is a semantic compiler input for validating ``MthPrcDt``.
    Its verified pointer is already part of the portable snapshot inventory, but
    the pointer's manifest and CSV must also remain unchanged through the final
    pointer commit.
    """

    try:
        calendar = config["upstream"]["calendar"]
        pointer_relative = _safe_relative_path(
            str(calendar["verified_pointer"]), "calendar pointer path"
        )
        expected_pointer_sha = str(calendar["verified_pointer_sha256"])
        expected_build_id = calendar["build_id"]
        expected_verification_id = calendar["verification_id"]
        expected_status = calendar["required_status"]
    except (KeyError, TypeError) as exc:
        raise NumericUnitPublicationError(
            "Locked profile lacks its verified calendar binding"
        ) from exc
    if _SHA256_RE.fullmatch(expected_pointer_sha) is None:
        raise NumericUnitPublicationError(
            "Verified calendar pointer SHA-256 is malformed"
        )
    project_root = _lexical_absolute(settings.project_root)
    pointer_path = _lexical_absolute(
        project_root / Path(*pointer_relative.parts)
    )
    _relative_to(pointer_path, project_root, "calendar pointer")
    pointer_payload = _read_authenticated_regular_file(
        pointer_path, "verified calendar pointer"
    )
    if _sha256_bytes(pointer_payload) != expected_pointer_sha:
        raise NumericUnitPublicationError(
            "Verified calendar pointer bytes differ"
        )
    try:
        pointer = json.loads(pointer_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NumericUnitPublicationError(
            "Verified calendar pointer is not UTF-8 JSON"
        ) from exc
    if not isinstance(pointer, dict) or any(
        pointer.get(key) != expected
        for key, expected in (
            ("build_id", expected_build_id),
            ("verification_id", expected_verification_id),
            ("status", expected_status),
        )
    ):
        raise NumericUnitPublicationError(
            "Verified calendar pointer identity/status differs"
        )
    artifact_relative_raw = pointer.get("primary_artifact_path")
    manifest_sha = str(pointer.get("primary_manifest_sha256", ""))
    if (
        not isinstance(artifact_relative_raw, str)
        or _SHA256_RE.fullmatch(manifest_sha) is None
    ):
        raise NumericUnitPublicationError(
            "Verified calendar pointer lacks a valid primary manifest binding"
        )
    artifact_relative = _safe_relative_path(
        artifact_relative_raw, "calendar artifact path"
    )
    manifest_path = _lexical_absolute(
        project_root / Path(*artifact_relative.parts) / "manifest.json"
    )
    _relative_to(manifest_path, project_root, "calendar manifest")
    manifest_payload = _read_authenticated_regular_file(
        manifest_path, "verified calendar manifest"
    )
    if _sha256_bytes(manifest_payload) != manifest_sha:
        raise NumericUnitPublicationError(
            "Verified calendar manifest bytes differ"
        )
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NumericUnitPublicationError(
            "Verified calendar manifest is not UTF-8 JSON"
        ) from exc
    outputs = manifest.get("outputs") if isinstance(manifest, dict) else None
    if not isinstance(outputs, list):
        raise NumericUnitPublicationError(
            "Verified calendar manifest lacks an outputs list"
        )
    matches = [
        item
        for item in outputs
        if isinstance(item, dict)
        and Path(str(item.get("path", ""))).name == "trading_calendar.csv"
    ]
    if len(matches) != 1:
        raise NumericUnitPublicationError(
            "Verified calendar manifest lacks exactly one trading_calendar.csv"
        )
    receipt = matches[0]
    calendar_path_raw = receipt.get("path")
    calendar_sha = str(receipt.get("sha256", ""))
    calendar_size = receipt.get("size_bytes")
    if (
        not isinstance(calendar_path_raw, str)
        or not Path(calendar_path_raw).expanduser().is_absolute()
        or _SHA256_RE.fullmatch(calendar_sha) is None
        or isinstance(calendar_size, bool)
        or not isinstance(calendar_size, int)
        or calendar_size < 0
    ):
        raise NumericUnitPublicationError(
            "Verified calendar output receipt is malformed"
        )
    calendar_path = _lexical_absolute(calendar_path_raw)
    database_root = _lexical_absolute(settings.database_root)
    calendar_relative = _relative_to(
        calendar_path, database_root, "verified trading calendar"
    )
    if (
        not calendar_relative.parts
        or calendar_relative.parts[0] not in {"interim", "quality"}
    ):
        raise NumericUnitPublicationError(
            "Verified trading calendar is outside database interim/quality"
        )
    actual_sha, actual_size = _hash_authenticated_regular_file(
        calendar_path, "verified trading calendar"
    )
    if actual_sha != calendar_sha or actual_size != calendar_size:
        raise NumericUnitPublicationError(
            "Verified trading calendar bytes differ"
        )


def _read_locked_profile_config(
    settings: NumericUnitPublicationSettings, compiled: Mapping[str, Any]
) -> dict[str, Any]:
    profile_path = _lexical_absolute(settings.profile_config_path)
    _relative_to(
        profile_path, _lexical_absolute(settings.project_root), "profile config"
    )
    payload = _read_authenticated_regular_file(profile_path, "profile config")
    try:
        expected_sha = compiled["binding"]["profile_config_sha256"]
    except (KeyError, TypeError) as exc:
        raise NumericUnitPublicationError(
            "Compiled profile lacks its profile-config binding"
        ) from exc
    if (
        _SHA256_RE.fullmatch(str(expected_sha)) is None
        or _sha256_bytes(payload) != expected_sha
    ):
        raise NumericUnitPublicationError(
            "Compiled profile does not bind the selected profile config bytes"
        )
    try:
        config = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise NumericUnitPublicationError(
            "Profile config is not valid UTF-8 TOML"
        ) from exc
    if config != settings.config:
        raise NumericUnitPublicationError(
            "In-memory settings differ from freshly authenticated profile config"
        )
    try:
        configured_policy_sha = str(config["policy"]["sha256"])
        compiled_policy_sha = str(compiled["binding"]["policy_sha256"])
    except (KeyError, TypeError) as exc:
        raise NumericUnitPublicationError(
            "Locked profile lacks its policy SHA-256 binding"
        ) from exc
    if (
        _SHA256_RE.fullmatch(configured_policy_sha) is None
        or compiled_policy_sha != configured_policy_sha
    ):
        raise NumericUnitPublicationError(
            "Compiled policy SHA-256 differs from the locked profile config"
        )
    try:
        configured_empirical_sha = str(config["empirical_lock"]["sha256"])
        compiled_empirical_sha = str(
            compiled["binding"]["empirical_lock_sha256"]
        )
    except (KeyError, TypeError) as exc:
        raise NumericUnitPublicationError(
            "Locked profile lacks its empirical-lock SHA-256 binding"
        ) from exc
    if (
        _SHA256_RE.fullmatch(configured_empirical_sha) is None
        or compiled_empirical_sha != configured_empirical_sha
    ):
        raise NumericUnitPublicationError(
            "Compiled empirical-lock SHA-256 differs from the locked profile config"
        )
    if config.get("status") != LOCKED_PROFILE_STATUS:
        raise NumericUnitPublicationError(
            "Unlocked numeric-unit profile config cannot be published"
        )
    locks = config.get("production_locks")
    if (
        not isinstance(locks, dict)
        or set(locks) != PRODUCTION_LOCK_KEYS
        or any(value is not True for value in locks.values())
    ):
        raise NumericUnitPublicationError(
            "Numeric-unit production locks are incomplete"
        )
    try:
        outputs = config["outputs"]
        checks = (
            int(config["schema_version"]) == SCHEMA_VERSION,
            config["profile_name"] == PROFILE_NAME,
            outputs["quality_subdirectory"] == QUALITY_SUBDIRECTORY,
            outputs["artifact_subdirectory"] == ARTIFACT_SUBDIRECTORY,
            outputs["primary_pointer"] == PRIMARY_POINTER,
            set(outputs["closed_quality_file_set"]) == set(QUALITY_FILES),
            len(outputs["closed_quality_file_set"]) == len(QUALITY_FILES),
            set(outputs["closed_artifact_file_set"]) == set(ARTIFACT_FILES),
            len(outputs["closed_artifact_file_set"]) == len(ARTIFACT_FILES),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NumericUnitPublicationError(
            "Profile config lacks the numeric-unit publication contract"
        ) from exc
    if not all(checks):
        raise NumericUnitPublicationError(
            "Profile config publication contract differs from the frozen architecture"
        )
    return config


def _frozen_compiled_view(compiled: Mapping[str, Any]) -> dict[str, Any]:
    try:
        view = {
            "binding": copy.deepcopy(compiled["binding"]),
            "build_id": compiled["build_id"],
            "mode": compiled["mode"],
            "summary": copy.deepcopy(compiled["summary"]),
        }
    except (KeyError, TypeError) as exc:
        raise NumericUnitPublicationError(
            "Compiled profile lacks its publication identity"
        ) from exc
    return json.loads(canonical_json_bytes(view).decode("utf-8"))


def _assert_locked_compiled_context(compiled: Mapping[str, Any]) -> None:
    try:
        build_id = str(compiled["build_id"])
        mode = compiled["mode"]
        binding = compiled["binding"]
        summary = compiled["summary"]
        contract_id = _assert_safe_component(binding["contract_id"], "contract_id")
    except (KeyError, TypeError) as exc:
        raise NumericUnitPublicationError(
            "Compiled profile lacks its publication identity"
        ) from exc
    if _BUILD_ID_RE.fullmatch(build_id) is None:
        raise NumericUnitPublicationError("Compiled build_id is malformed")
    if not contract_id:
        raise NumericUnitPublicationError("Compiled contract_id is blank")
    if mode != "production":
        raise NumericUnitPublicationError(
            "Only production-mode numeric-unit evidence may be published"
        )
    expected_summary_fields = {
        "aggregate_row_counts",
        "build_id",
        "check_count",
        "diagnostic_count",
        "fail_count",
        "hard_failure_count",
        "mode",
        "pass_count",
        "production_authorized",
        "profile_status",
        "schema_version",
        "source_counts",
        "status",
    }
    if not isinstance(summary, dict) or set(summary) != expected_summary_fields:
        raise NumericUnitPublicationError(
            "Compiled production summary schema differs"
        )
    if (
        isinstance(summary["schema_version"], bool)
        or summary["schema_version"] != SCHEMA_VERSION
    ):
        raise NumericUnitPublicationError(
            "Compiled production summary schema_version differs"
        )
    if summary.get("build_id") != build_id:
        raise NumericUnitPublicationError(
            "Summary build_id differs from compiled build_id"
        )
    if summary.get("mode") != mode:
        raise NumericUnitPublicationError(
            "Summary mode differs from compiled production mode"
        )
    if summary.get("profile_status") != LOCKED_PROFILE_STATUS:
        raise NumericUnitPublicationError(
            "Summary profile_status is not the locked production status"
        )
    if summary.get("production_authorized") is not True:
        raise NumericUnitPublicationError(
            "Summary does not authorize primary production publication"
        )
    if summary.get("status") != PUBLISHED_STATUS:
        raise NumericUnitPublicationError(
            "Only a passing primary numeric-unit profile may be published"
        )
    counts: dict[str, int] = {}
    for field in _SUMMARY_COUNT_FIELDS:
        value = summary[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise NumericUnitPublicationError(
                f"Summary {field} is not a nonnegative integer"
            )
        counts[field] = value
    if counts["check_count"] != (
        counts["pass_count"]
        + counts["fail_count"]
        + counts["diagnostic_count"]
    ):
        raise NumericUnitPublicationError(
            "Summary check counts do not conserve"
        )
    if counts["fail_count"] != 0 or counts["hard_failure_count"] != 0:
        raise NumericUnitPublicationError(
            "A numeric-unit profile with failed checks may not be published"
        )
    aggregate_counts = summary["aggregate_row_counts"]
    if (
        not isinstance(aggregate_counts, dict)
        or set(aggregate_counts) != _AGGREGATE_TABLES
    ):
        raise NumericUnitPublicationError(
            "Summary aggregate-row-count schema differs"
        )
    for value in aggregate_counts.values():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise NumericUnitPublicationError(
                "Summary aggregate row count is invalid"
            )
    if aggregate_counts["checks"] != counts["check_count"]:
        raise NumericUnitPublicationError(
            "Summary check_count differs from aggregate row counts"
        )
    source_counts = summary["source_counts"]
    if not isinstance(source_counts, dict) or set(source_counts) != {
        "daily",
        "monthly",
        "shares",
    }:
        raise NumericUnitPublicationError("Summary source-count schema differs")
    for source, values in source_counts.items():
        if not isinstance(values, dict) or set(values) != {
            "canonical_rows",
            "full_rows",
            "retained_rows",
        }:
            raise NumericUnitPublicationError(
                f"Summary source-count schema differs for {source}"
            )
        for value in values.values():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise NumericUnitPublicationError(
                    f"Summary source count is invalid for {source}"
                )
    for key in (
        "implementation_sha256",
        "policy_sha256",
        "empirical_lock_sha256",
        "profile_config_sha256",
        "snapshot_inventory_sha256",
    ):
        if _SHA256_RE.fullmatch(str(binding.get(key, ""))) is None:
            raise NumericUnitPublicationError(
                f"Compiled binding {key} is not SHA-256"
            )


def _validate_compiler_owned_checks(
    payload: bytes,
    summary: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    """Validate the rendered production gate table against its summary.

    The publisher does not reproduce compiler analytics, but it must establish
    that the bytes it installs contain the complete, passing production gate
    set claimed by ``summary.json``.  Treating this CSV as opaque would allow a
    stale or mutated compiler result to omit a mandatory gate while retaining a
    superficially passing summary.
    """

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NumericUnitPublicationError(
            "Compiler-owned checks.csv is not valid UTF-8"
        ) from exc
    if "\x00" in text:
        raise NumericUnitPublicationError(
            "Compiler-owned checks.csv contains a NUL byte"
        )

    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise NumericUnitPublicationError(
                "Compiler-owned checks.csv is empty"
            ) from exc
        if tuple(header) != CHECKS_FIELDS:
            raise NumericUnitPublicationError(
                "Compiler-owned checks.csv header/schema differs"
            )
        rows: list[dict[str, str]] = []
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(CHECKS_FIELDS):
                raise NumericUnitPublicationError(
                    "Compiler-owned checks.csv row width differs at line "
                    f"{line_number}"
                )
            rows.append(dict(zip(CHECKS_FIELDS, values, strict=True)))
    except csv.Error as exc:
        raise NumericUnitPublicationError(
            "Compiler-owned checks.csv is not valid RFC CSV"
        ) from exc

    by_id: dict[str, dict[str, str]] = {}
    status_counts = {status: 0 for status in _CHECK_STATUSES}
    failed_rows: list[str] = []
    hard_failed_rows: list[str] = []
    for row in rows:
        check_id = row["check_id"]
        if not check_id:
            raise NumericUnitPublicationError(
                "Compiler-owned checks.csv contains a blank check_id"
            )
        if check_id in by_id:
            raise NumericUnitPublicationError(
                f"Compiler-owned checks.csv contains duplicate check_id {check_id!r}"
            )
        by_id[check_id] = row
        status = row["status"]
        severity = row["severity"]
        if status not in _CHECK_STATUSES:
            raise NumericUnitPublicationError(
                f"Compiler-owned checks.csv has invalid status for {check_id!r}"
            )
        if severity not in _CHECK_SEVERITIES:
            raise NumericUnitPublicationError(
                f"Compiler-owned checks.csv has invalid severity for {check_id!r}"
            )
        status_counts[status] += 1
        if status == "FAIL":
            failed_rows.append(check_id)
            if severity == "hard":
                hard_failed_rows.append(check_id)

    if failed_rows or hard_failed_rows:
        raise NumericUnitPublicationError(
            "Compiler-owned checks.csv contains failed checks: "
            f"failed={sorted(failed_rows)}, hard_failed={sorted(hard_failed_rows)}"
        )

    try:
        expected_counts = {
            "check_count": summary["check_count"],
            "pass_count": summary["pass_count"],
            "fail_count": summary["fail_count"],
            "diagnostic_count": summary["diagnostic_count"],
        }
    except (KeyError, TypeError) as exc:
        raise NumericUnitPublicationError(
            "Compiled summary lacks check counts for checks.csv validation"
        ) from exc
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in expected_counts.values()
    ):
        raise NumericUnitPublicationError(
            "Compiled summary check counts are invalid for checks.csv validation"
        )
    actual_counts = {
        "check_count": len(rows),
        "pass_count": status_counts["PASS"],
        "fail_count": status_counts["FAIL"],
        "diagnostic_count": status_counts["DIAGNOSTIC"],
    }
    if actual_counts != expected_counts:
        raise NumericUnitPublicationError(
            "Compiler-owned checks.csv counts differ from compiled summary: "
            f"expected={expected_counts}, actual={actual_counts}"
        )

    actual_ids = set(by_id)
    missing = sorted(REQUIRED_PRODUCTION_CHECK_IDS - actual_ids)
    unexpected = sorted(actual_ids - REQUIRED_PRODUCTION_CHECK_IDS)
    if missing or unexpected:
        raise NumericUnitPublicationError(
            "Compiler-owned checks.csv differs from the exact production check set: "
            f"missing={missing}, unexpected={unexpected}"
        )
    weakened = sorted(
        check_id
        for check_id in REQUIRED_PRODUCTION_CHECK_IDS
        if by_id[check_id]["status"] != "PASS"
        or by_id[check_id]["severity"] != "hard"
    )
    if weakened:
        raise NumericUnitPublicationError(
            "Mandatory production checks are not passing hard gates: "
            f"{weakened}"
        )
    return tuple(rows)


def _parse_canonical_json_object(payload: bytes, label: str) -> dict[str, Any]:
    """Parse one compiler document and require its exact canonical wire form."""

    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NumericUnitPublicationError(
            f"Compiler-owned {label} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise NumericUnitPublicationError(
            f"Compiler-owned {label} is not a JSON object"
        )
    try:
        canonical = canonical_json_bytes(value)
    except NumericUnitPublicationError as exc:
        raise NumericUnitPublicationError(
            f"Compiler-owned {label} is not canonical JSON"
        ) from exc
    if payload != canonical:
        raise NumericUnitPublicationError(
            f"Compiler-owned {label} is not exact canonical JSON"
        )
    return value


def _parse_canonical_json_cell(value: str, label: str) -> Any:
    """Parse the compiler's canonical JSON-without-final-LF check cells."""

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise NumericUnitPublicationError(
            f"Compiler-owned {label} is not valid JSON"
        ) from exc
    try:
        canonical = canonical_json_bytes(parsed)[:-1].decode("utf-8")
    except NumericUnitPublicationError as exc:
        raise NumericUnitPublicationError(
            f"Compiler-owned {label} is not canonical JSON"
        ) from exc
    if value != canonical:
        raise NumericUnitPublicationError(
            f"Compiler-owned {label} is not exact canonical JSON"
        )
    return parsed


def _validate_summary_document(
    payload: bytes, compiled: Mapping[str, Any]
) -> None:
    document = _parse_canonical_json_object(payload, "summary.json")
    if document != compiled["summary"]:
        raise NumericUnitPublicationError(
            "Compiler-owned summary.json differs from compiled summary"
        )


def _validate_compiler_owned_json_documents(
    payloads: Mapping[str, bytes],
    compiled: Mapping[str, Any],
    config: Mapping[str, Any],
    snapshots: Sequence[AuthenticatedSnapshot],
) -> dict[str, dict[str, Any]]:
    """Validate every non-summary compiler JSON contract independently."""

    summary = compiled["summary"]
    configured_quality_files = list(
        config["outputs"]["closed_quality_file_set"]
    )
    expected_success = {
        "aggregate_only": True,
        "build_id": compiled["build_id"],
        "closed_quality_file_set": configured_quality_files,
        "publication_performed": False,
        "schema_version": SCHEMA_VERSION,
        "status": summary["status"],
    }
    success = _parse_canonical_json_object(payloads["_SUCCESS.json"], "_SUCCESS.json")
    if success != expected_success:
        raise NumericUnitPublicationError(
            "Compiler-owned _SUCCESS.json contract differs"
        )

    expected_contract = {
        "DlyRetx_is_availability_requirement": False,
        "capitalization_formula_ids": list(
            config["outputs"]["vocabularies"]["capitalization_formula_id"]
        ),
        "capitalization_quantiles": {
            "estimator": "full_population_type7_linear_interpolation_over_disk_backed_Decimal_ratios_with_at_least_50_significant_digits",
            "hard_threshold_use": False,
            "sampled": False,
        },
        "closed_quality_file_set": configured_quality_files,
        "daily_price_flag_is_eligibility_filter": False,
        "empirical_lock_sha256": compiled["binding"]["empirical_lock_sha256"],
        "event_reconciliation_in_scope": False,
        "joins": {
            "MthCalDt_substitution_permitted": False,
            "daily_shares": "PERMNO plus inclusive DlyCalDt share interval",
            "monthly_daily": "monthly (PERMNO,MthPrcDt) equals daily (PERMNO,DlyCalDt)",
            "monthly_shares": "PERMNO plus inclusive nonblank MthPrcDt share interval",
        },
        "policy_sha256": compiled["binding"]["policy_sha256"],
        "profile_config_sha256": compiled["binding"]["profile_config_sha256"],
        "profile_name": config["profile_name"],
        "profile_status": config["status"],
        "profile_version": config["profile_version"],
        "raw_security_keys_may_be_published": False,
        "schema_version": SCHEMA_VERSION,
        "schemas": config["outputs"]["schemas"],
        "scope": {
            "december_2014_is_quality_only": True,
            "quality_end_date": config["scope"]["quality_end_date"],
            "start_date": config["scope"]["start_date"],
        },
    }
    contract = _parse_canonical_json_object(
        payloads["profile_contract.json"], "profile_contract.json"
    )
    if contract != expected_contract:
        raise NumericUnitPublicationError(
            "Compiler-owned profile_contract.json contract differs"
        )

    expected_downstream = {
        "backtests_authorized": False,
        "beta_estimation_authorized": False,
        "blockers": list(config["downstream"]["blockers"]),
        "build_id": compiled["build_id"],
        "labels_authorized": False,
        "profile_status": config["status"],
        "return_history_authorized": False,
        "schema_version": SCHEMA_VERSION,
        "typed_market_backbone_authorized": False,
        "universe_unit_inputs_authorized": False,
    }
    downstream = _parse_canonical_json_object(
        payloads["downstream_gate_status.json"], "downstream_gate_status.json"
    )
    if downstream != expected_downstream:
        raise NumericUnitPublicationError(
            "Compiler-owned downstream_gate_status.json contract differs"
        )

    receipt = _parse_canonical_json_object(
        payloads["input_receipt.json"], "input_receipt.json"
    )
    required_receipt_keys = {
        "authentication",
        "portable_snapshots",
        "raw_sources",
        "schema_version",
        "upstreams",
    }
    if set(receipt) != required_receipt_keys or receipt["schema_version"] != SCHEMA_VERSION:
        raise NumericUnitPublicationError(
            "Compiler-owned input_receipt.json schema differs"
        )
    if receipt["authentication"] != {"raw_inputs": True, "upstreams": True}:
        raise NumericUnitPublicationError(
            "Compiler-owned input_receipt.json is not fully authenticated"
        )

    descriptors = _snapshot_descriptors(snapshots)
    if receipt["portable_snapshots"] != descriptors:
        raise NumericUnitPublicationError(
            "Compiler-owned input_receipt.json portable snapshots differ"
        )
    descriptors_by_id = {item["snapshot_id"]: item for item in descriptors}
    upstream_ids = {
        "methodology_freeze",
        "phase_01_freeze",
        "phase_02_config",
        "crsp_backbone_contract",
        "contract_verified_pointer",
        "calendar_verified_pointer",
        "daily_monthly_profile_verified_pointer",
        "supporting_profile_verified_pointer",
    }
    expected_upstreams = [
        {
            "relative_path": descriptors_by_id[snapshot_id]["relative_path"],
            "sha256": descriptors_by_id[snapshot_id]["sha256"],
            "size_bytes": descriptors_by_id[snapshot_id]["size_bytes"],
            "snapshot_id": snapshot_id,
        }
        for snapshot_id in sorted(upstream_ids)
    ]
    if receipt["upstreams"] != expected_upstreams:
        raise NumericUnitPublicationError(
            "Compiler-owned input_receipt.json upstream receipts differ"
        )

    raw_sources = receipt["raw_sources"]
    if not isinstance(raw_sources, list) or len(raw_sources) != 3:
        raise NumericUnitPublicationError(
            "Compiler-owned input_receipt.json raw-source set differs"
        )
    raw_by_source: dict[str, Mapping[str, Any]] = {}
    raw_keys = {
        "canonical_rows",
        "filename",
        "full_rows",
        "retained_rows",
        "sha256",
        "size_bytes",
        "source",
    }
    for raw in raw_sources:
        if not isinstance(raw, dict) or set(raw) != raw_keys:
            raise NumericUnitPublicationError(
                "Compiler-owned input_receipt.json raw-source schema differs"
            )
        source = raw.get("source")
        if source not in {"daily", "monthly", "shares"} or source in raw_by_source:
            raise NumericUnitPublicationError(
                "Compiler-owned input_receipt.json raw-source identity differs"
            )
        raw_by_source[str(source)] = raw
    if [item["source"] for item in raw_sources] != ["daily", "monthly", "shares"]:
        raise NumericUnitPublicationError(
            "Compiler-owned input_receipt.json raw-source order differs"
        )
    for source in ("daily", "monthly", "shares"):
        raw = raw_by_source[source]
        configured = config["sources"][source]
        snapshot = descriptors_by_id[f"raw_{source}"]
        if any(
            raw[key] != expected
            for key, expected in (
                ("filename", configured["filename"]),
                ("sha256", snapshot["sha256"]),
                ("size_bytes", snapshot["size_bytes"]),
                ("full_rows", configured["full_file_rows"]),
                ("retained_rows", configured["verified_retained_rows"]),
            )
        ):
            raise NumericUnitPublicationError(
                f"Compiler-owned input_receipt.json {source} receipt differs"
            )
        if "verified_canonical_rows" in configured and raw["canonical_rows"] != configured[
            "verified_canonical_rows"
        ]:
            raise NumericUnitPublicationError(
                f"Compiler-owned input_receipt.json {source} canonical count differs"
            )
        if any(
            isinstance(raw[field], bool)
            or not isinstance(raw[field], int)
            or raw[field] < 0
            for field in ("full_rows", "retained_rows", "canonical_rows", "size_bytes")
        ):
            raise NumericUnitPublicationError(
                f"Compiler-owned input_receipt.json {source} counts are invalid"
            )
        if summary["source_counts"][source] != {
            "canonical_rows": raw["canonical_rows"],
            "full_rows": raw["full_rows"],
            "retained_rows": raw["retained_rows"],
        }:
            raise NumericUnitPublicationError(
                f"Compiler-owned input_receipt.json {source} summary counts differ"
            )
    return {
        "downstream_gate_status": downstream,
        "input_receipt": receipt,
        "profile_contract": contract,
        "success": success,
    }


def _canonical_integer(value: str, label: str, *, signed: bool) -> int:
    pattern = r"(?:0|[1-9][0-9]*|-[1-9][0-9]*)" if signed else r"(?:0|[1-9][0-9]*)"
    if re.fullmatch(pattern, value, flags=re.ASCII) is None:
        raise NumericUnitPublicationError(
            f"Compiler-owned {label} is not a canonical integer"
        )
    return int(value)


def _parse_inventory_csv(
    table: str,
    payload: bytes,
    fields: Sequence[str],
    expected_row_count: int,
) -> tuple[dict[str, Any], ...]:
    label = _INVENTORY_FILES[table]
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NumericUnitPublicationError(
            f"Compiler-owned {label} is not valid UTF-8"
        ) from exc
    if "\x00" in text:
        raise NumericUnitPublicationError(
            f"Compiler-owned {label} contains a NUL byte"
        )
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise NumericUnitPublicationError(
                f"Compiler-owned {label} is empty"
            ) from exc
        if tuple(header) != tuple(fields):
            raise NumericUnitPublicationError(
                f"Compiler-owned {label} header/schema differs"
            )
        parsed_rows: list[dict[str, Any]] = []
        nonnegative = _NONNEGATIVE_INTEGER_FIELDS[table]
        optional_signed = _OPTIONAL_SIGNED_INTEGER_FIELDS.get(table, frozenset())
        optional_nonnegative = _OPTIONAL_NONNEGATIVE_INTEGER_FIELDS.get(
            table, frozenset()
        )
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(fields):
                raise NumericUnitPublicationError(
                    f"Compiler-owned {label} row width differs at line {line_number}"
                )
            row: dict[str, Any] = {}
            for field, value in zip(fields, values, strict=True):
                cell_label = f"{label} {field} at line {line_number}"
                if field in nonnegative:
                    row[field] = _canonical_integer(value, cell_label, signed=False)
                elif field in optional_signed:
                    row[field] = (
                        "" if value == "" else _canonical_integer(value, cell_label, signed=True)
                    )
                elif field in optional_nonnegative:
                    row[field] = (
                        "" if value == "" else _canonical_integer(value, cell_label, signed=False)
                    )
                else:
                    row[field] = value
            parsed_rows.append(row)
    except csv.Error as exc:
        raise NumericUnitPublicationError(
            f"Compiler-owned {label} is not valid RFC CSV"
        ) from exc
    if len(parsed_rows) != expected_row_count:
        raise NumericUnitPublicationError(
            f"Compiler-owned {label} row count differs from compiled summary"
        )
    return tuple(parsed_rows)


def _validate_inventory_balances(
    table: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    for index, row in enumerate(rows, start=2):
        if row.get("status") not in _CHECK_STATUSES or row.get("severity") not in _CHECK_SEVERITIES:
            raise NumericUnitPublicationError(
                f"Compiler-owned {_INVENTORY_FILES[table]} status/severity differs at line {index}"
            )
        if table == "numeric_inventory":
            states = sum(
                int(row[field])
                for field in (
                    "blank_rows",
                    "valid_in_domain_rows",
                    "valid_out_of_domain_rows",
                    "malformed_rows",
                    "nonfinite_rows",
                    "overflow_rows",
                )
            )
            balanced = row["population_rows"] == states and row["negative_zero_rows"] <= (
                row["valid_in_domain_rows"] + row["valid_out_of_domain_rows"]
            )
        elif table == "return_inventory":
            coherent = row["coherent_rows"]
            incoherent = row["incoherent_rows"]
            balanced = (coherent == "" and incoherent == "") or (
                coherent != ""
                and incoherent != ""
                and row["population_rows"] == coherent + incoherent
            )
        else:
            exclusions = _parse_canonical_json_cell(
                str(row["exclusions"]), f"{_INVENTORY_FILES[table]} exclusions at line {index}"
            )
            if not isinstance(exclusions, dict) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in exclusions.values()
            ):
                raise NumericUnitPublicationError(
                    f"Compiler-owned {_INVENTORY_FILES[table]} exclusions differ at line {index}"
                )
            if table == "identity_inventory":
                balanced = (
                    row["population_rows"] == row["comparable_rows"] + row["excluded_rows"]
                    and row["comparable_rows"] == row["pass_rows"] + row["fail_rows"]
                    and row["excluded_rows"] == sum(exclusions.values())
                )
            else:
                balanced = (
                    row["population_rows"] == row["comparable_rows"] + row["excluded_rows"]
                    and row["comparable_rows"]
                    == row["formula_pass_rows"] + row["formula_fail_rows"]
                    and row["excluded_rows"] == sum(exclusions.values())
                )
        if not balanced:
            raise NumericUnitPublicationError(
                f"Compiler-owned {_INVENTORY_FILES[table]} balance differs at line {index}"
            )


def _measurement_projection(
    table: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    period_segment: str | None = None,
) -> dict[str, Any]:
    excluded = _MEASUREMENT_EXCLUDED_FIELDS[table]
    projected = [
        {key: value for key, value in row.items() if key not in excluded}
        for row in rows
        if period_segment is None or row["period_segment"] == period_segment
    ]
    return {
        "row_count": len(projected),
        "sha256": _sha256_bytes(canonical_json_bytes(projected)),
    }


def _read_empirical_projection_lock(
    settings: NumericUnitPublicationSettings,
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    relative = _safe_relative_path(
        str(config["empirical_lock"]["path"]), "empirical-lock path"
    )
    project_root = _lexical_absolute(settings.project_root)
    path = _lexical_absolute(project_root / Path(*relative.parts))
    _relative_to(path, project_root, "empirical lock")
    payload = _read_authenticated_regular_file(path, "empirical lock")
    if _sha256_bytes(payload) != config["empirical_lock"]["sha256"]:
        raise NumericUnitPublicationError("Authenticated empirical-lock bytes differ")
    try:
        lock = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NumericUnitPublicationError(
            "Authenticated empirical lock is not UTF-8 JSON"
        ) from exc
    projections = lock.get("measurement_projections") if isinstance(lock, dict) else None
    if not isinstance(projections, dict) or set(projections) != set(_INVENTORY_FILES):
        raise NumericUnitPublicationError(
            "Authenticated empirical-lock projection set differs"
        )
    segment_names = {"development_2000_2014_11", "embargo_2014_12"}
    for table, expected in projections.items():
        if not isinstance(expected, dict) or set(expected) != {
            "policy_derived_fields_excluded",
            "row_count",
            "segments",
            "sha256",
        }:
            raise NumericUnitPublicationError(
                f"Authenticated empirical-lock projection schema differs for {table}"
            )
        if expected["policy_derived_fields_excluded"] != sorted(
            _MEASUREMENT_EXCLUDED_FIELDS[table]
        ) or not isinstance(expected["segments"], dict) or set(expected["segments"]) != segment_names:
            raise NumericUnitPublicationError(
                f"Authenticated empirical-lock projection contract differs for {table}"
            )
        for projection in (expected, *expected["segments"].values()):
            if (
                not isinstance(projection, dict)
                or isinstance(projection.get("row_count"), bool)
                or not isinstance(projection.get("row_count"), int)
                or projection["row_count"] < 0
                or _SHA256_RE.fullmatch(str(projection.get("sha256", ""))) is None
            ):
                raise NumericUnitPublicationError(
                    f"Authenticated empirical-lock projection value differs for {table}"
                )
    try:
        capitalization = lock["transparent_aggregates"]["capitalization"][
            "non_ADR_NS_N_primary_threshold"
        ]
    except (KeyError, TypeError) as exc:
        raise NumericUnitPublicationError(
            "Authenticated empirical lock lacks its non-ADR threshold"
        ) from exc
    selector = capitalization.get("selector") if isinstance(capitalization, dict) else None
    expected_selector = {
        "ShareType": "NS",
        "effective_interval_ShrAdrFlg": "N",
        "interval_date": "MthPrcDt",
        "interval_match_count": 1,
        "primary_exchange_scope": "all_observed_no_exchange_filter",
        "required_valid_operands": ["MthCap", "MthPrc", "interval_ShrOut"],
    }
    if (
        not isinstance(capitalization, dict)
        or capitalization.get("authoritative_formula_id")
        != "monthly_interval_cap_price_shares"
        or capitalization.get("periodicity") != "monthly"
        or capitalization.get("required_fail_rows") != 0
        or capitalization.get("required_multiple_share_match_rows") != 0
        or capitalization.get("required_pass_rate") != "1"
        or selector != expected_selector
        or not isinstance(capitalization.get("segments"), dict)
        or set(capitalization["segments"]) != segment_names
    ):
        raise NumericUnitPublicationError(
            "Authenticated empirical-lock non-ADR threshold differs"
        )
    for segment, counts in capitalization["segments"].items():
        if not isinstance(counts, dict) or set(counts) != set(_NON_ADR_COUNT_FIELDS):
            raise NumericUnitPublicationError(
                f"Authenticated empirical-lock non-ADR counts differ for {segment}"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts.values()
        ):
            raise NumericUnitPublicationError(
                f"Authenticated empirical-lock non-ADR counts are invalid for {segment}"
            )
    return projections, capitalization


def _identity_zero_comparable_active_segments(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pooled: dict[tuple[str, str, int, str], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    for row in rows:
        if row["severity"] != "hard":
            continue
        key = (
            str(row["identity_id"]),
            str(row["source"]),
            int(row["year"]),
            str(row["period_segment"]),
        )
        pooled[key][0] += int(row["population_rows"])
        pooled[key][1] += int(row["comparable_rows"])
    failures = [
        {
            "identity_id": identity_id,
            "source": source,
            "year": year,
            "period_segment": segment,
            "population_rows": population,
            "comparable_rows": comparable,
        }
        for (identity_id, source, year, segment), (population, comparable)
        in pooled.items()
        if population > 0 and comparable == 0
    ]
    return sorted(
        failures,
        key=lambda row: (
            str(row["identity_id"]).encode("utf-8"),
            str(row["source"]).encode("utf-8"),
            int(row["year"]),
            str(row["period_segment"]).encode("utf-8"),
        ),
    )


def _identity_active_segment_population_differences(
    numeric_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    references = {
        "daily_price_volume": ("daily", "daily", "DlyPrc"),
        "monthly_daily_price": ("monthly_daily", "monthly", "MthPrc"),
        "monthly_daily_cap": ("monthly_daily", "monthly", "MthCap"),
    }
    expected: Counter[tuple[str, str, int, str]] = Counter()
    reference_columns = {
        (numeric_source, column)
        for _identity_source, numeric_source, column in references.values()
    }
    for row in numeric_rows:
        source_column = (str(row["source"]), str(row["column"]))
        if source_column not in reference_columns:
            continue
        expected[
            (
                source_column[0],
                source_column[1],
                int(row["year"]),
                str(row["period_segment"]),
            )
        ] += int(row["population_rows"])
    actual: Counter[tuple[str, str, int, str]] = Counter()
    for row in identity_rows:
        identity_id = str(row["identity_id"])
        if identity_id not in references:
            continue
        actual[
            (
                identity_id,
                str(row["source"]),
                int(row["year"]),
                str(row["period_segment"]),
            )
        ] += int(row["population_rows"])
    differences: list[dict[str, Any]] = []
    for identity_id, (identity_source, numeric_source, column) in references.items():
        groups = {
            (year, segment)
            for source, expected_column, year, segment in expected
            if source == numeric_source and expected_column == column
        } | {
            (year, segment)
            for actual_identity, source, year, segment in actual
            if actual_identity == identity_id and source == identity_source
        }
        for year, segment in groups:
            expected_rows = expected[(numeric_source, column, year, segment)]
            actual_rows = actual[(identity_id, identity_source, year, segment)]
            if expected_rows != actual_rows:
                differences.append(
                    {
                        "identity_id": identity_id,
                        "source": identity_source,
                        "year": year,
                        "period_segment": segment,
                        "expected_population_rows": expected_rows,
                        "actual_population_rows": actual_rows,
                    }
                )
    return sorted(
        differences,
        key=lambda row: (
            str(row["identity_id"]).encode("utf-8"),
            str(row["source"]).encode("utf-8"),
            int(row["year"]),
            str(row["period_segment"]).encode("utf-8"),
        ),
    )


def _non_adr_capitalization_segment_counts(
    rows: Sequence[Mapping[str, Any]],
    capitalization: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    selector = capitalization["selector"]
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        if (
            row["formula_id"] != capitalization["authoritative_formula_id"]
            or row["periodicity"] != capitalization["periodicity"]
            or row["share_type"] != selector["ShareType"]
            or row["shr_adr_flag"] != selector["effective_interval_ShrAdrFlg"]
        ):
            continue
        segment = str(row["period_segment"])
        cell = counts.setdefault(
            segment, {field: 0 for field in _NON_ADR_COUNT_FIELDS}
        )
        for field in _NON_ADR_COUNT_FIELDS:
            cell[field] += int(row[field])
    return {segment: counts[segment] for segment in sorted(counts)}


def _rate_text(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    with localcontext() as context:
        context.prec = 50
        value = (Decimal(numerator) / Decimal(denominator)).quantize(
            Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN
        )
    rendered = format(value, "f").rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0", "+0"} else rendered


def _validate_remaining_semantic_gates(
    rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
    checks_by_id: Mapping[str, Mapping[str, str]],
    capitalization: Mapping[str, Any],
) -> None:
    pooled = _identity_zero_comparable_active_segments(
        rows_by_table["identity_inventory"]
    )
    populations = _identity_active_segment_population_differences(
        rows_by_table["numeric_inventory"], rows_by_table["identity_inventory"]
    )
    for check_id, observed in (
        ("identity_pooled_active_segment_zero_comparable_groups", pooled),
        ("identity_active_segment_population_conservation", populations),
    ):
        check = checks_by_id[check_id]
        expected = _parse_canonical_json_cell(
            check["expected"], f"checks.csv expected for {check_id}"
        )
        actual = _parse_canonical_json_cell(
            check["actual"], f"checks.csv actual for {check_id}"
        )
        if expected != [] or actual != observed or observed:
            raise NumericUnitPublicationError(
                f"Compiler-owned identity coverage gate differs for {check_id}"
            )

    segment_counts = _non_adr_capitalization_segment_counts(
        rows_by_table["capitalization_diagnostics"], capitalization
    )
    expected_segment_counts = capitalization["segments"]
    segment_check_id = "non_adr_capitalization_segment_counts"
    segment_check = checks_by_id[segment_check_id]
    check_expected_counts = _parse_canonical_json_cell(
        segment_check["expected"],
        f"checks.csv expected for {segment_check_id}",
    )
    check_actual_counts = _parse_canonical_json_cell(
        segment_check["actual"], f"checks.csv actual for {segment_check_id}"
    )
    if (
        segment_counts != expected_segment_counts
        or check_expected_counts != expected_segment_counts
        or check_actual_counts != segment_counts
    ):
        raise NumericUnitPublicationError(
            "Compiler-owned non-ADR capitalization segment-count gate differs"
        )

    totals = {
        field: sum(counts[field] for counts in segment_counts.values())
        for field in _NON_ADR_COUNT_FIELDS
    }
    expected_theory = {
        "formula_fail_rows": capitalization["required_fail_rows"],
        "formula_pass_rate": capitalization["required_pass_rate"],
        "multiple_share_match_rows": capitalization[
            "required_multiple_share_match_rows"
        ],
        "pass_rule": "quantization_interval_overlap",
    }
    actual_theory = {
        "formula_fail_rows": totals["formula_fail_rows"],
        "formula_pass_rate": _rate_text(
            totals["formula_pass_rows"], totals["comparable_rows"]
        ),
        "multiple_share_match_rows": totals["multiple_share_match_rows"],
        "pass_rule": "quantization_interval_overlap",
    }
    theory_check_id = "non_adr_capitalization_theory_threshold"
    theory_check = checks_by_id[theory_check_id]
    check_expected_theory = _parse_canonical_json_cell(
        theory_check["expected"], f"checks.csv expected for {theory_check_id}"
    )
    check_actual_theory = _parse_canonical_json_cell(
        theory_check["actual"], f"checks.csv actual for {theory_check_id}"
    )
    if (
        actual_theory != expected_theory
        or check_expected_theory != expected_theory
        or check_actual_theory != actual_theory
    ):
        raise NumericUnitPublicationError(
            "Compiler-owned non-ADR capitalization theory gate differs"
        )


def _validate_compiler_owned_inventories(
    payloads: Mapping[str, bytes],
    compiled: Mapping[str, Any],
    config: Mapping[str, Any],
    checks: Sequence[Mapping[str, str]],
    settings: NumericUnitPublicationSettings,
) -> dict[str, tuple[dict[str, Any], ...]]:
    rows_by_table: dict[str, tuple[dict[str, Any], ...]] = {}
    for table, filename in _INVENTORY_FILES.items():
        fields = tuple(config["outputs"]["schemas"][table])
        rows = _parse_inventory_csv(
            table,
            payloads[filename],
            fields,
            compiled["summary"]["aggregate_row_counts"][table],
        )
        _validate_inventory_balances(table, rows)
        rows_by_table[table] = rows

    checks_by_id = {row["check_id"]: row for row in checks}
    projections, capitalization = _read_empirical_projection_lock(
        settings, config
    )
    for table, rows in rows_by_table.items():
        expected_table = projections[table]
        for segment in ("full", "development_2000_2014_11", "embargo_2014_12"):
            expected = (
                {"row_count": expected_table["row_count"], "sha256": expected_table["sha256"]}
                if segment == "full"
                else expected_table["segments"][segment]
            )
            actual = _measurement_projection(
                table,
                rows,
                period_segment=None if segment == "full" else segment,
            )
            check_id = f"empirical_{table}_measurement_projection_{segment}"
            check = checks_by_id[check_id]
            check_expected = _parse_canonical_json_cell(
                check["expected"], f"checks.csv expected for {check_id}"
            )
            check_actual = _parse_canonical_json_cell(
                check["actual"], f"checks.csv actual for {check_id}"
            )
            if actual != expected or check_expected != expected or check_actual != actual:
                raise NumericUnitPublicationError(
                    f"Compiler-owned empirical projection differs for {check_id}"
                )

    for table, rows in rows_by_table.items():
        check_id = f"{table}_hard_failures"
        check = checks_by_id[check_id]
        expected = _parse_canonical_json_cell(
            check["expected"], f"checks.csv expected for {check_id}"
        )
        actual = _parse_canonical_json_cell(
            check["actual"], f"checks.csv actual for {check_id}"
        )
        observed = sum(row["status"] == "FAIL" for row in rows)
        if expected != 0 or actual != observed or observed != 0:
            raise NumericUnitPublicationError(
                f"Compiler-owned aggregate failure gate differs for {check_id}"
            )

    for source in ("daily", "monthly", "shares"):
        for metric, configured_field in (
            ("full_rows", "full_file_rows"),
            ("retained_rows", "verified_retained_rows"),
            ("canonical_rows", "verified_canonical_rows"),
        ):
            if configured_field not in config["sources"][source]:
                continue
            check_id = f"source_{source}_{metric}"
            check = checks_by_id[check_id]
            expected = _parse_canonical_json_cell(
                check["expected"], f"checks.csv expected for {check_id}"
            )
            actual = _parse_canonical_json_cell(
                check["actual"], f"checks.csv actual for {check_id}"
            )
            locked = config["sources"][source][configured_field]
            observed = compiled["summary"]["source_counts"][source][metric]
            if expected != locked or actual != observed or observed != locked:
                raise NumericUnitPublicationError(
                    f"Compiler-owned source-count gate differs for {check_id}"
                )

    profile_check = checks_by_id["production_profile_status"]
    profile_expected = _parse_canonical_json_cell(
        profile_check["expected"], "checks.csv expected for production_profile_status"
    )
    profile_actual = _parse_canonical_json_cell(
        profile_check["actual"], "checks.csv actual for production_profile_status"
    )
    if profile_expected != config["status"] or profile_actual != config["status"]:
        raise NumericUnitPublicationError(
            "Compiler-owned production profile-status gate differs"
        )
    _validate_remaining_semantic_gates(
        rows_by_table, checks_by_id, capitalization
    )
    return rows_by_table


def _validate_semantic_build_id(
    compiled: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
    inventories: Mapping[str, Sequence[Mapping[str, Any]]],
    checks: Sequence[Mapping[str, str]],
) -> None:
    """Reconstruct the compiler's complete pre-ID semantic seed."""

    tables = {
        table: [dict(row) for row in inventories[table]]
        for table in sorted(inventories)
    }
    tables["checks"] = [dict(row) for row in checks]
    build_seed = {
        "binding": compiled["binding"],
        "downstream_gate_status_without_build_id": {
            key: value
            for key, value in documents["downstream_gate_status"].items()
            if key != "build_id"
        },
        "input_receipt": documents["input_receipt"],
        "mode": compiled["mode"],
        "profile_contract": documents["profile_contract"],
        "success_without_build_id": {
            key: value
            for key, value in documents["success"].items()
            if key != "build_id"
        },
        "summary_without_build_id": {
            key: value
            for key, value in compiled["summary"].items()
            if key != "build_id"
        },
        "tables": tables,
    }
    expected_build_id = "crsp-numeric-" + hashlib.sha256(
        canonical_json_bytes(build_seed)
    ).hexdigest()[:24]
    if compiled["build_id"] != expected_build_id:
        raise NumericUnitPublicationError(
            "Compiled build_id differs from the reconstructed semantic build"
        )


def _implementation_binding(
    settings: NumericUnitPublicationSettings,
    compiled: Mapping[str, Any],
    *,
    compiler_path: Path,
    cli_path: Path,
) -> dict[str, str]:
    project_root = _lexical_absolute(settings.project_root)
    compiler_path = _lexical_absolute(compiler_path)
    cli_path = _lexical_absolute(cli_path)
    _relative_to(compiler_path, project_root, "compiler implementation")
    _relative_to(cli_path, project_root, "CLI implementation")
    compiler_sha, _ = _hash_authenticated_regular_file(
        compiler_path, "compiler implementation"
    )
    cli_sha, _ = _hash_authenticated_regular_file(cli_path, "CLI implementation")
    publisher_sha, _ = _hash_authenticated_regular_file(
        _lexical_absolute(Path(__file__)), "publisher implementation"
    )
    if compiler_sha != compiled["binding"]["implementation_sha256"]:
        raise NumericUnitPublicationError(
            "Compiler implementation differs from the compiled binding"
        )
    return {
        "cli_sha256": cli_sha,
        "compiler_sha256": compiler_sha,
        "publisher_sha256": publisher_sha,
    }


def _validate_payload_names(
    payloads: Mapping[str, bytes], expected_names: Sequence[str], label: str
) -> dict[str, bytes]:
    expected = set(expected_names)
    if len(expected) != len(tuple(expected_names)):
        raise NumericUnitPublicationError(
            f"Configured {label} closed file set contains duplicates"
        )
    if set(payloads) != expected:
        raise NumericUnitPublicationError(
            f"{label} closed file set differs: expected {sorted(expected)}, "
            f"actual {sorted(payloads)}"
        )
    result: dict[str, bytes] = {}
    for name in sorted(payloads):
        _assert_safe_component(name, f"{label} filename")
        payload = payloads[name]
        if not isinstance(payload, bytes):
            raise NumericUnitPublicationError(
                f"{label} payload {name!r} is not bytes"
            )
        result[name] = payload
    return result


def _quality_directory(
    settings: NumericUnitPublicationSettings,
    compiled: Mapping[str, Any],
) -> Path:
    return _lexical_absolute(
        settings.quality_root
        / _assert_safe_component(compiled["binding"]["contract_id"], "contract_id")
        / QUALITY_SUBDIRECTORY
        / _assert_safe_component(compiled["build_id"], "build_id")
    )


def _normalize_documents(documents: Mapping[str, Any]) -> dict[str, Any]:
    try:
        manifest = json.loads(canonical_json_bytes(documents["manifest"]).decode("utf-8"))
        pointer = json.loads(canonical_json_bytes(documents["pointer"]).decode("utf-8"))
        pointer_bytes = documents["pointer_bytes"]
        artifact_directory = _lexical_absolute(Path(documents["artifact_directory"]))
        artifact_payloads = _validate_payload_names(
            documents["artifact_payloads"], ARTIFACT_FILES, "artifact"
        )
    except (KeyError, TypeError) as exc:
        raise NumericUnitPublicationError(
            "Rendered artifact documents are incomplete"
        ) from exc
    if not isinstance(pointer_bytes, bytes):
        raise NumericUnitPublicationError("Rendered pointer bytes are not bytes")
    return {
        "artifact_directory": artifact_directory,
        "artifact_payloads": artifact_payloads,
        "manifest": manifest,
        "pointer": pointer,
        "pointer_bytes": pointer_bytes,
    }


def _validate_rendered_documents(
    settings: NumericUnitPublicationSettings,
    compiled: Mapping[str, Any],
    quality_directory: Path,
    quality_payloads: Mapping[str, bytes],
    documents: Mapping[str, Any],
) -> None:
    artifact_directory = documents["artifact_directory"]
    expected_artifact_directory = _lexical_absolute(
        settings.artifact_root / ARTIFACT_SUBDIRECTORY / compiled["build_id"]
    )
    if artifact_directory != expected_artifact_directory:
        raise NumericUnitPublicationError(
            "Artifact renderer selected an unexpected directory"
        )
    _assert_no_symlink_components(artifact_directory, "artifact destination")
    database_root = _lexical_absolute(settings.database_root)
    try:
        quality_relative = quality_directory.relative_to(database_root)
    except ValueError as exc:
        raise NumericUnitPublicationError(
            "Quality destination is outside the configured database root"
        ) from exc
    if not quality_relative.parts or quality_relative.parts[0] != "quality":
        raise NumericUnitPublicationError(
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
        "profile_name": PROFILE_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": compiled["summary"]["status"],
        "summary": compiled["summary"],
    }
    manifest_bytes = canonical_json_bytes(expected_manifest)
    if documents["manifest"] != expected_manifest:
        raise NumericUnitPublicationError(
            "Rendered manifest differs from the exact authenticated manifest"
        )
    if documents["artifact_payloads"]["manifest.json"] != manifest_bytes:
        raise NumericUnitPublicationError(
            "Rendered manifest bytes differ from the authenticated object"
        )
    inventory = "".join(
        f"{item['sha256']}  {item['path']}\n" for item in outputs
    ).encode("utf-8")
    if documents["artifact_payloads"]["output_hashes.sha256"] != inventory:
        raise NumericUnitPublicationError(
            "Rendered output inventory differs from the quality payloads"
        )
    artifact_label = artifact_directory.relative_to(
        _lexical_absolute(settings.project_root)
    ).as_posix()
    _safe_relative_path(artifact_label, "artifact label")
    expected_pointer = {
        "artifact_path": artifact_label,
        "build_id": compiled["build_id"],
        "contract_id": compiled["binding"]["contract_id"],
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "schema_version": SCHEMA_VERSION,
        "status": compiled["summary"]["status"],
    }
    if documents["pointer"] != expected_pointer:
        raise NumericUnitPublicationError(
            "Rendered pointer differs from the exact authenticated pointer"
        )
    if documents["pointer_bytes"] != canonical_json_bytes(expected_pointer):
        raise NumericUnitPublicationError(
            "Rendered pointer bytes differ from the authenticated object"
        )


def _read_closed_directory(directory: Path, label: str) -> dict[str, bytes]:
    _assert_no_symlink_components(directory, label)
    if not directory.is_dir():
        raise NumericUnitPublicationError(
            f"Existing {label} is not a real directory"
        )
    result: dict[str, bytes] = {}
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.is_symlink() or not entry.is_file():
            raise NumericUnitPublicationError(
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
    if _read_closed_directory(final, label) != dict(payloads):
        raise NumericUnitPublicationError(
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
            raise NumericUnitPublicationError(
                "Short write while staging immutable payload"
            )
        remaining = remaining[written:]


def _read_anchored_payloads(directory_descriptor: int) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for name in sorted(os.listdir(directory_descriptor)):
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise NumericUnitPublicationError(
                    f"Staged entry is not a regular file: {name}"
                )
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1 << 20):
                chunks.append(chunk)
            result[name] = b"".join(chunks)
        finally:
            os.close(descriptor)
    return result


def _stage_payloads(parent: Path, prefix: str, payloads: Mapping[str, bytes]) -> Path:
    _assert_no_symlink_components(parent, "staging parent")
    parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(parent, "staging parent")
    stage = Path(tempfile.mkdtemp(prefix=f".{prefix}.staging-", dir=parent))
    directory_descriptor: int | None = None
    try:
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
            raise NumericUnitPublicationError(
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
            raise NumericUnitPublicationError(
                "Staged bytes differ immediately after writing"
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
        raise NumericUnitPublicationError(
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
        if pre_replace_gate is not None:
            pre_replace_gate()
        _assert_no_symlink_components(path, "atomic file")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _assert_optional_regular_file(path: Path, label: str) -> None:
    _assert_no_symlink_components(path, label)
    if path.exists() and not path.is_file():
        raise NumericUnitPublicationError(f"Existing {label} is not a regular file")


def _run_receipt_path(
    settings: NumericUnitPublicationSettings, compiled: Mapping[str, Any]
) -> Path:
    return _lexical_absolute(
        settings.run_root
        / _assert_safe_component(compiled["binding"]["contract_id"], "contract_id")
        / QUALITY_SUBDIRECTORY
        / _assert_safe_component(compiled["build_id"], "build_id")
        / "build_receipt.json"
    )


def _publish_run_receipt(
    receipt_path: Path,
    compiled: Mapping[str, Any],
    final_pointer: Mapping[str, Any],
    final_pointer_bytes: bytes,
    quality_directory: Path,
    implementation: Mapping[str, str],
    *,
    elapsed_seconds: float | None,
) -> Path:
    identity = {
        "build_id": compiled["build_id"],
        "contract_id": compiled["binding"]["contract_id"],
        "implementation": dict(implementation),
        "manifest_sha256": final_pointer["manifest_sha256"],
        "pointer_sha256": _sha256_bytes(final_pointer_bytes),
        "status": compiled["summary"]["status"],
    }
    _assert_optional_regular_file(receipt_path, "run receipt")
    if receipt_path.exists():
        try:
            existing = json.loads(
                _read_authenticated_regular_file(receipt_path, "run receipt").decode(
                    "utf-8"
                )
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NumericUnitPublicationError(
                "Existing run receipt is unreadable; primary pointer was not moved"
            ) from exc
        if any(existing.get(key) != value for key, value in identity.items()):
            raise NumericUnitPublicationError(
                "Existing run receipt selects a different publication"
            )
        return receipt_path
    receipt = {
        "artifact_path": final_pointer["artifact_path"],
        **identity,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": round(max(0.0, float(elapsed_seconds or 0.0)), 6),
        "quality_path": quality_directory.as_posix(),
        "schema_version": SCHEMA_VERSION,
    }
    _write_atomic_file(receipt_path, canonical_json_bytes(receipt))
    return receipt_path


def publish_numeric_unit_profile(
    compiled: Mapping[str, Any],
    settings: NumericUnitPublicationSettings,
    quality_payloads: Mapping[str, bytes],
    documents: Mapping[str, Any],
    *,
    snapshots: Sequence[AuthenticatedSnapshot],
    compiler_path: Path,
    cli_path: Path,
    elapsed_seconds: float | None = None,
) -> dict[str, Path]:
    """Install a passing rendered profile and commit its primary pointer last."""

    _assert_allowed_roots(settings)
    try:
        compiler_quality_payloads = _validate_payload_names(
            compiled["quality_payloads"],
            QUALITY_FILES,
            "compiler-owned quality",
        )
    except (KeyError, TypeError) as exc:
        raise NumericUnitPublicationError(
            "Compiled profile lacks its compiler-owned quality payloads"
        ) from exc
    compiled_view = _frozen_compiled_view(compiled)
    _assert_locked_compiled_context(compiled_view)
    _validate_summary_document(
        compiler_quality_payloads["summary.json"], compiled_view
    )
    compiler_checks = _validate_compiler_owned_checks(
        compiler_quality_payloads["checks.csv"], compiled_view["summary"]
    )
    config = _read_locked_profile_config(settings, compiled_view)
    snapshot_tuple = tuple(snapshots)
    _validate_snapshot_contract(config, snapshot_tuple)
    expected_snapshot_sha = compiled_view["binding"]["snapshot_inventory_sha256"]
    _authenticate_snapshots(
        settings, config, snapshot_tuple, expected_snapshot_sha
    )
    _authenticate_calendar_chain(settings, config)
    compiler_documents = _validate_compiler_owned_json_documents(
        compiler_quality_payloads, compiled_view, config, snapshot_tuple
    )
    compiler_inventories = _validate_compiler_owned_inventories(
        compiler_quality_payloads,
        compiled_view,
        config,
        compiler_checks,
        settings,
    )
    _validate_semantic_build_id(
        compiled_view,
        compiler_documents,
        compiler_inventories,
        compiler_checks,
    )
    implementation = _implementation_binding(
        settings,
        compiled_view,
        compiler_path=compiler_path,
        cli_path=cli_path,
    )
    quality_payloads = _validate_payload_names(
        quality_payloads, QUALITY_FILES, "quality"
    )
    if quality_payloads != compiler_quality_payloads:
        raise NumericUnitPublicationError(
            "Supplied quality bytes differ from compiler-owned payloads"
        )
    documents = _normalize_documents(documents)
    quality_directory = _quality_directory(settings, compiled_view)
    _assert_no_symlink_components(quality_directory, "quality destination")
    _validate_rendered_documents(
        settings, compiled_view, quality_directory, quality_payloads, documents
    )
    artifact_directory = documents["artifact_directory"]
    artifact_payloads = documents["artifact_payloads"]

    final_pointer = dict(documents["pointer"])
    final_pointer["implementation"] = dict(implementation)
    final_pointer_bytes = canonical_json_bytes(final_pointer)
    pointer_path = _lexical_absolute(settings.artifact_root / PRIMARY_POINTER)
    receipt_path = _run_receipt_path(settings, compiled_view)
    _assert_optional_regular_file(pointer_path, "primary pointer")
    _assert_optional_regular_file(receipt_path, "run receipt")
    _assert_existing_matches(quality_directory, quality_payloads, "quality")
    _assert_existing_matches(artifact_directory, artifact_payloads, "artifact")

    quality_stage: Path | None = None
    artifact_stage: Path | None = None
    try:
        quality_stage = _stage_payloads(
            quality_directory.parent, compiled_view["build_id"], quality_payloads
        )
        artifact_stage = _stage_payloads(
            artifact_directory.parent, compiled_view["build_id"], artifact_payloads
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
        receipt_path,
        compiled_view,
        final_pointer,
        final_pointer_bytes,
        quality_directory,
        implementation,
        elapsed_seconds=elapsed_seconds,
    )

    def commit_gate() -> None:
        _assert_locked_compiled_context(compiled_view)
        fresh_config = _read_locked_profile_config(settings, compiled_view)
        if fresh_config != config:
            raise NumericUnitPublicationError(
                "Locked profile config changed before pointer commit"
            )
        _authenticate_snapshots(
            settings, config, snapshot_tuple, expected_snapshot_sha
        )
        _authenticate_calendar_chain(settings, config)
        if _implementation_binding(
            settings,
            compiled_view,
            compiler_path=compiler_path,
            cli_path=cli_path,
        ) != implementation:
            raise NumericUnitPublicationError(
                "Publication implementation bundle changed before pointer commit"
            )
        _assert_existing_matches(quality_directory, quality_payloads, "quality")
        _assert_existing_matches(artifact_directory, artifact_payloads, "artifact")
        _validate_rendered_documents(
            settings, compiled_view, quality_directory, quality_payloads, documents
        )
        _assert_optional_regular_file(pointer_path, "primary pointer")
        _assert_optional_regular_file(receipt_path, "run receipt")

    commit_gate()
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
