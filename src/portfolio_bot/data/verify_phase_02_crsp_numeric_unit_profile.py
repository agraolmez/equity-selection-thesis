"""Independent verifier for the locked Phase 2 CRSP numeric/unit profile.

This module deliberately does not import the primary compiler or publisher.
It authenticates their published bytes, hashes (but never decodes) the three
raw CSVs, reconstructs the typed aggregate tables and semantic build ID, and
re-evaluates every mandatory production gate from the published CSV evidence.

Passing and failing semantic reports may be preserved as immutable evidence.
Only a zero-difference passing report may move the verified pointer, which is
committed last after the complete verification context is reauthenticated.
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
import secrets
import stat
import tomllib
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_CONFIG = (
    PROJECT_ROOT
    / "configs/phase_02_canonicalization/crsp_numeric_unit_profile_development_v1.toml"
)
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs/paths.toml"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/phase_02_canonical_panel"

SCHEMA_VERSION = 1
PROFILE_NAME = "crsp_development_numeric_unit_profile"
PRIMARY_STATUS = "numeric_unit_primary_passed_pending_independent_verification"
LOCKED_PROFILE_STATUS = "locked_after_aggregate_discovery_before_primary_profile"
PASS_STATUS = "verified_passed"
FAIL_STATUS = "verification_failed"

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
BUILD_ID_RE = re.compile(r"crsp-numeric-[0-9a-f]{24}\Z")
VERIFICATION_ID_RE = re.compile(r"verify-[0-9a-f]{16}\Z")
SNAPSHOT_INVENTORY_DOMAIN = b"phase02-crsp-numeric-unit-snapshots-v1\x00"
VERIFICATION_ID_DOMAIN = b"phase02-crsp-numeric-unit-independent-v1\x00"
IMPLEMENTATION_BUNDLE_DOMAIN = b"phase02-crsp-numeric-unit-verifier-bundle-v1\x00"

SOURCE_NAMES = ("daily", "monthly", "shares")
RAW_SNAPSHOT_IDS = {
    "daily": "raw_daily",
    "monthly": "raw_monthly",
    "shares": "raw_shares",
}
UPSTREAM_SNAPSHOT_PATHS = {
    "methodology_freeze": ("methodology", "freeze_manifest"),
    "phase_01_freeze": ("phase_01", "freeze_manifest"),
    "phase_02_config": ("phase_02", "phase_config"),
    "crsp_backbone_contract": ("phase_02", "crsp_backbone_contract"),
    "contract_verified_pointer": ("contract", "verified_pointer"),
    "calendar_verified_pointer": ("calendar", "verified_pointer"),
    "daily_monthly_profile_verified_pointer": (
        "daily_monthly_profile",
        "verified_pointer",
    ),
    "supporting_profile_verified_pointer": (
        "supporting_profile",
        "verified_pointer",
    ),
}
UPSTREAM_SNAPSHOT_LOCKS = {
    "methodology_freeze": ("methodology", "freeze_manifest", "freeze_manifest_sha256"),
    "phase_01_freeze": ("phase_01", "freeze_manifest", "freeze_manifest_sha256"),
    "phase_02_config": ("phase_02", "phase_config", "phase_config_sha256"),
    "crsp_backbone_contract": (
        "phase_02", "crsp_backbone_contract", "crsp_backbone_contract_sha256"
    ),
    "contract_verified_pointer": (
        "contract", "verified_pointer", "verified_pointer_sha256"
    ),
    "calendar_verified_pointer": (
        "calendar", "verified_pointer", "verified_pointer_sha256"
    ),
    "daily_monthly_profile_verified_pointer": (
        "daily_monthly_profile", "verified_pointer", "verified_pointer_sha256"
    ),
    "supporting_profile_verified_pointer": (
        "supporting_profile", "verified_pointer", "verified_pointer_sha256"
    ),
}
EXPECTED_SNAPSHOT_IDS = frozenset(
    {"policy", "empirical_lock", *UPSTREAM_SNAPSHOT_PATHS, *RAW_SNAPSHOT_IDS.values()}
)

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
VERIFICATION_FILES = ("differences.csv", "independent_check.json")

CSV_FIELDS: dict[str, tuple[str, ...]] = {
    "checks": (
        "check_id", "stage", "family", "source", "stratum", "severity",
        "expected", "actual", "tolerance_rule", "status",
        "downstream_consequence",
    ),
    "numeric_inventory": (
        "source", "column", "year", "period_segment", "stratum",
        "numeric_kind", "population_rows", "blank_rows",
        "valid_in_domain_rows", "valid_out_of_domain_rows", "malformed_rows",
        "nonfinite_rows", "overflow_rows", "negative_zero_rows", "min_value",
        "max_value", "quantum_counts", "status", "severity",
    ),
    "return_inventory": (
        "source", "periodicity", "year", "period_segment", "return_flag",
        "duration_flag", "total_return_state", "returnx_state",
        "mapped_trading_periods", "semantic_rule", "population_rows",
        "coherent_rows", "incoherent_rows", "status", "severity",
    ),
    "identity_inventory": (
        "identity_id", "source", "year", "period_segment", "stratum",
        "severity", "population_rows", "comparable_rows", "pass_rows",
        "fail_rows", "excluded_rows", "exclusions", "coverage_rate",
        "pass_rate", "max_absolute_error", "max_relative_error",
        "tolerance_rule", "status",
    ),
    "capitalization_diagnostics": (
        "formula_id", "periodicity", "year", "period_segment",
        "primary_exchange", "share_type", "shr_adr_flag",
        "security_line_count", "population_rows", "comparable_rows",
        "excluded_rows", "exclusions", "zero_share_match_rows",
        "one_share_match_rows", "multiple_share_match_rows",
        "formula_pass_rows", "formula_fail_rows", "ratio_p01", "ratio_p05",
        "ratio_p25", "ratio_p50", "ratio_p75", "ratio_p95", "ratio_p99",
        "formula_pass_rate", "threshold_rule", "status", "severity",
    ),
}

CSV_FILENAMES = {
    "checks": "checks.csv",
    "numeric_inventory": "numeric_inventory.csv",
    "return_inventory": "return_inventory.csv",
    "identity_inventory": "identity_inventory.csv",
    "capitalization_diagnostics": "capitalization_diagnostics.csv",
}
SORT_FIELDS = {
    "checks": ("check_id",),
    "numeric_inventory": ("source", "column", "period_segment", "year", "stratum"),
    "return_inventory": (
        "source", "periodicity", "period_segment", "year", "return_flag",
        "duration_flag", "total_return_state", "returnx_state",
    ),
    "identity_inventory": (
        "identity_id", "source", "period_segment", "year", "stratum",
    ),
    "capitalization_diagnostics": (
        "formula_id", "periodicity", "period_segment", "year",
        "primary_exchange", "share_type", "shr_adr_flag",
    ),
}

NONNEGATIVE_INTEGER_FIELDS = {
    "numeric_inventory": frozenset(
        {
            "year", "population_rows", "blank_rows", "valid_in_domain_rows",
            "valid_out_of_domain_rows", "malformed_rows", "nonfinite_rows",
            "overflow_rows", "negative_zero_rows",
        }
    ),
    "return_inventory": frozenset({"year", "population_rows"}),
    "identity_inventory": frozenset(
        {
            "year", "population_rows", "comparable_rows", "pass_rows",
            "fail_rows", "excluded_rows",
        }
    ),
    "capitalization_diagnostics": frozenset(
        {
            "year", "security_line_count", "population_rows",
            "comparable_rows", "excluded_rows", "zero_share_match_rows",
            "one_share_match_rows", "multiple_share_match_rows",
            "formula_pass_rows", "formula_fail_rows",
        }
    ),
}
OPTIONAL_SIGNED_INTEGER_FIELDS = {
    "return_inventory": frozenset({"mapped_trading_periods"})
}
OPTIONAL_NONNEGATIVE_INTEGER_FIELDS = {
    "return_inventory": frozenset({"coherent_rows", "incoherent_rows"})
}
JSON_CELL_FIELDS = {
    "checks": frozenset({"stratum", "expected", "actual"}),
    "numeric_inventory": frozenset({"stratum", "quantum_counts"}),
    "identity_inventory": frozenset({"stratum", "exclusions"}),
    "capitalization_diagnostics": frozenset({"exclusions"}),
}

MEASUREMENT_EXCLUDED_FIELDS = {
    "numeric_inventory": frozenset({"status", "severity"}),
    "return_inventory": frozenset({"semantic_rule", "status", "severity"}),
    "identity_inventory": frozenset({"severity", "status", "tolerance_rule"}),
    "capitalization_diagnostics": frozenset(
        {"severity", "status", "threshold_rule"}
    ),
}
EMPIRICAL_TABLES = (
    "capitalization_diagnostics",
    "identity_inventory",
    "numeric_inventory",
    "return_inventory",
)
NON_ADR_COUNT_FIELDS = (
    "population_rows",
    "comparable_rows",
    "excluded_rows",
    "zero_share_match_rows",
    "one_share_match_rows",
    "multiple_share_match_rows",
    "formula_pass_rows",
    "formula_fail_rows",
)
SEGMENTS = ("development_2000_2014_11", "embargo_2014_12")

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
    | {
        f"empirical_{table}_measurement_projection_{segment}"
        for table in EMPIRICAL_TABLES
        for segment in ("full", *SEGMENTS)
    }
)
REQUIRED_VERIFIER_CHECK_IDS = frozenset(
    {
        "AUTHENTICATED_PRIMARY_BUNDLE",
        "AUTHENTICATED_LOCKED_CONTEXT",
        "PRIMARY_GATE_SET",
        "NS_N_SEGMENT_THEORY_INDEPENDENT",
        "SUMMARY_CONSERVATION",
        "PRIMARY_DOCUMENT_CONTRACTS",
        "SEMANTIC_BUILD_ID",
        *{
            f"PRIMARY_GATE__{check_id}"
            for check_id in REQUIRED_PRODUCTION_CHECK_IDS
        },
        *{
            f"INVENTORY_SEMANTICS__{table}"
            for table in EMPIRICAL_TABLES
        },
    }
)
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

PRIMARY_IMPLEMENTATION_PATHS = {
    "compiler_sha256": "src/portfolio_bot/data/phase_02_crsp_numeric_unit_profile.py",
    "publisher_sha256": "src/portfolio_bot/data/phase_02_crsp_numeric_unit_profile_publication.py",
    "cli_sha256": "scripts/build_crsp_numeric_unit_profile.py",
}
VERIFIER_IMPLEMENTATION_LABELS = (
    "scripts/verify_phase_02_crsp_numeric_unit_profile.py",
    "src/portfolio_bot/data/verify_phase_02_crsp_numeric_unit_profile.py",
)

REPORT_KEYS = frozenset(
    {
        "build_id", "checks", "contract_id", "independence",
        "primary_manifest_sha256", "profile_name", "schema_version", "status",
        "summary", "verification_id", "verifier_implementation_sha256",
    }
)
CHECK_KEYS = frozenset(
    {"actual", "category", "check_id", "evidence", "expected", "message", "status"}
)
SUMMARY_KEYS = frozenset(
    {
        "check_count", "difference_count", "fail_count", "pass_count",
        "primary_gate_count", "projection_count", "source_count",
    }
)
INDEPENDENCE_KEYS = frozenset(
    {
        "aggregate_only", "empirical_projections_recomputed",
        "full_raw_hashes_recomputed", "primary_compiler_imported",
        "primary_publisher_imported", "raw_files_hashed",
        "raw_payload_rows_decoded", "semantic_build_id_recomputed",
    }
)
DIFFERENCE_FIELDS = (
    "check_id", "category", "expected", "actual", "evidence", "message"
)
VERIFIED_POINTER_KEYS = frozenset(
    {
        "build_id", "contract_id", "differences_sha256",
        "independent_check_sha256", "primary_manifest_sha256", "schema_version",
        "status", "verification_id", "verification_path",
    }
)


class NumericUnitVerificationError(RuntimeError):
    """Raised when a safe independent evidence boundary cannot be established."""


@dataclasses.dataclass(frozen=True)
class SettingsProvenance:
    """Exact caller/environment choices used to derive selected paths."""

    phase_config_override: Path | None
    primary_pointer_override: Path | None
    artifact_root_override: Path | None
    database_root_override: Path | None
    database_environment_override: str | None
    raw_root_override: Path | None
    quality_root_override: Path | None


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
    interim_root: Path
    quality_root: Path
    config: dict[str, Any]
    profile_config_payload: bytes
    paths_config_payload: bytes
    phase_config_payload: bytes
    provenance: SettingsProvenance


@dataclasses.dataclass(frozen=True)
class AuthenticatedSettings:
    """Immutable settings bytes, roots, and output names used by verification."""

    profile_config_payload: bytes
    paths_config_payload: bytes
    phase_config_payload: bytes
    profile_config_sha256: str
    paths_config_sha256: str
    phase_config_sha256: str
    project_root: Path
    profile_config_path: Path
    paths_config_path: Path
    phase_config_path: Path
    primary_pointer_path: Path
    artifact_root: Path
    database_root: Path
    raw_root: Path
    interim_root: Path
    quality_root: Path
    artifact_subdirectory: str
    quality_subdirectory: str
    verification_subdirectory: str
    primary_pointer_name: str
    verified_pointer_name: str
    closed_artifact_file_set: tuple[str, ...]
    closed_quality_file_set: tuple[str, ...]
    closed_verification_file_set: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _ClosedDirectoryState:
    path: Path
    trusted_root: Path
    label: str
    device: int
    inode: int
    names: tuple[str, ...]
    hashes: tuple[tuple[str, str], ...]


@dataclasses.dataclass(frozen=True)
class _ValidatedReport:
    payload: bytes
    value: Mapping[str, Any]


@dataclasses.dataclass
class _StableRegularAnchor:
    path: Path
    trusted_root: Path
    label: str
    descriptor: int
    opened: os.stat_result
    parent_descriptor: int
    parent_opened: os.stat_result
    name: str


def canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise NumericUnitVerificationError(f"Value is not canonical JSON: {exc}") from exc
    return (rendered + "\n").encode("utf-8")


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value)[:-1].decode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _fsync_file(descriptor: int, label: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise NumericUnitVerificationError(f"Cannot fsync {label}") from exc


def _fsync_directory(descriptor: int, label: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise NumericUnitVerificationError(f"Cannot fsync {label} directory") from exc


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_trusted_root_fd(
    trusted_root: Path, label: str
) -> tuple[int, os.stat_result]:
    """Open one explicit trust anchor directly, without walking from ``/``."""

    root = _lexical_absolute(trusted_root)
    if not root.is_absolute() or root.anchor != os.sep:
        raise NumericUnitVerificationError(
            f"{label} trusted root must be an absolute path"
        )
    descriptor: int | None = None
    try:
        before = os.stat(root, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise NumericUnitVerificationError(
                f"Symlink is prohibited for {label} trusted root: {root}"
            )
        descriptor = os.open(root, _directory_flags())
        opened = os.fstat(descriptor)
        after = os.stat(root, follow_symlinks=False)
    except NumericUnitVerificationError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise NumericUnitVerificationError(
            f"Cannot open {label} trusted root: {root}"
        ) from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not _same_inode(before, opened)
        or not _same_inode(opened, after)
    ):
        os.close(descriptor)
        raise NumericUnitVerificationError(
            f"{label} trusted-root identity changed: {root}"
        )
    return descriptor, opened


def _assert_trusted_root_path_identity(
    trusted_root: Path, expected: os.stat_result, label: str
) -> None:
    descriptor, observed = _open_trusted_root_fd(trusted_root, label)
    try:
        if not _same_inode(expected, observed):
            raise NumericUnitVerificationError(
                f"{label} trusted-root identity changed"
            )
    finally:
        os.close(descriptor)


def _relative_to_trusted_root(path: Path, trusted_root: Path, label: str) -> Path:
    absolute = _lexical_absolute(path)
    root = _lexical_absolute(trusted_root)
    try:
        return absolute.relative_to(root)
    except ValueError as exc:
        raise NumericUnitVerificationError(
            f"{label} escapes trusted root: {root}"
        ) from exc


def _trusted_root_for_path(
    path: Path, trusted_roots: Sequence[Path], label: str
) -> Path:
    """Select the narrowest explicit allowed root containing ``path``."""

    absolute = _lexical_absolute(path)
    candidates: list[Path] = []
    for candidate in trusted_roots:
        root = _lexical_absolute(candidate)
        try:
            absolute.relative_to(root)
        except ValueError:
            continue
        candidates.append(root)
    if not candidates:
        raise NumericUnitVerificationError(
            f"{label} is outside every explicit trusted root"
        )
    return max(candidates, key=lambda value: len(value.parts))


def _open_directory_fd_secure(
    path: Path, label: str, *, create: bool, trusted_root: Path
) -> tuple[int, os.stat_result]:
    """Traverse below an explicit root fd-relative without following links."""

    absolute = _lexical_absolute(path)
    root = _lexical_absolute(trusted_root)
    relative = _relative_to_trusted_root(absolute, root, label)
    current_fd, root_opened = _open_trusted_root_fd(root, label)
    traversed = root
    try:
        for raw_part in relative.parts:
            part = _safe_component(raw_part, f"{label} path")
            traversed = traversed / part
            created_component = False
            try:
                before = os.stat(
                    part, dir_fd=current_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                if not create:
                    raise NumericUnitVerificationError(
                        f"Cannot open {label}: {absolute}"
                    )
                try:
                    os.mkdir(part, 0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise NumericUnitVerificationError(
                        f"Cannot create {label}: {traversed}"
                    ) from exc
                else:
                    created_component = True
                    _fsync_directory(current_fd, f"new {label} parent")
                try:
                    before = os.stat(
                        part, dir_fd=current_fd, follow_symlinks=False
                    )
                except OSError as exc:
                    raise NumericUnitVerificationError(
                        f"Cannot authenticate created {label}: {traversed}"
                    ) from exc
            except OSError as exc:
                raise NumericUnitVerificationError(
                    f"Cannot inspect {label}: {traversed}"
                ) from exc
            if stat.S_ISLNK(before.st_mode):
                raise NumericUnitVerificationError(
                    f"Symlink is prohibited in {label} path: {traversed}"
                )
            child_fd: int | None = None
            try:
                child_fd = os.open(part, _directory_flags(), dir_fd=current_fd)
                opened = os.fstat(child_fd)
                after = os.stat(
                    part, dir_fd=current_fd, follow_symlinks=False
                )
            except OSError as exc:
                if child_fd is not None:
                    os.close(child_fd)
                raise NumericUnitVerificationError(
                    f"Cannot anchor {label}: {traversed}"
                ) from exc
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not _same_inode(before, opened)
                or not _same_inode(opened, after)
            ):
                os.close(child_fd)
                raise NumericUnitVerificationError(
                    f"{label} directory identity changed: {traversed}"
                )
            try:
                if created_component:
                    _fsync_directory(child_fd, f"new {label} ancestor")
            except Exception:
                os.close(child_fd)
                raise
            os.close(current_fd)
            current_fd = child_fd
        _assert_trusted_root_path_identity(root, root_opened, label)
        opened = os.fstat(current_fd)
        return current_fd, opened
    except Exception:
        os.close(current_fd)
        raise


def _assert_directory_path_identity(
    path: Path, expected: os.stat_result, label: str, *, trusted_root: Path
) -> None:
    descriptor, observed = _open_directory_fd_secure(
        path, label, create=False, trusted_root=trusted_root
    )
    try:
        if not _same_inode(expected, observed):
            raise NumericUnitVerificationError(f"{label} path identity changed")
    finally:
        os.close(descriptor)


def _stable_descriptor(
    path: Path, label: str, *, trusted_root: Path
) -> _StableRegularAnchor:
    path = _lexical_absolute(path)
    trusted_root = _lexical_absolute(trusted_root)
    parent_descriptor, parent_opened = _open_directory_fd_secure(
        path.parent,
        f"{label} parent",
        create=False,
        trusted_root=trusted_root,
    )
    name = _safe_component(path.name, label)
    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise NumericUnitVerificationError(
                f"Symlink is prohibited for {label}: {path}"
            )
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except NumericUnitVerificationError:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
        raise NumericUnitVerificationError(f"Cannot open {label}: {path}") from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or not _same_inode(before, opened)
        or not _same_inode(opened, after)
    ):
        os.close(descriptor)
        os.close(parent_descriptor)
        raise NumericUnitVerificationError(f"{label} is not the selected regular file")
    return _StableRegularAnchor(
        path=path,
        trusted_root=trusted_root,
        label=label,
        descriptor=descriptor,
        opened=opened,
        parent_descriptor=parent_descriptor,
        parent_opened=parent_opened,
        name=name,
    )


def _assert_descriptor_unchanged(anchor: _StableRegularAnchor) -> None:
    current = os.fstat(anchor.descriptor)
    after = os.stat(
        anchor.name,
        dir_fd=anchor.parent_descriptor,
        follow_symlinks=False,
    )
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(anchor.opened, field) != getattr(current, field)
        for field in identity
    ) or any(getattr(current, field) != getattr(after, field) for field in identity):
        raise NumericUnitVerificationError(
            f"{anchor.label} changed while being read"
        )
    _assert_directory_path_identity(
        anchor.path.parent,
        anchor.parent_opened,
        f"{anchor.label} parent",
        trusted_root=anchor.trusted_root,
    )


def _read_stable(path: Path, label: str, *, trusted_root: Path) -> bytes:
    anchor = _stable_descriptor(path, label, trusted_root=trusted_root)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(anchor.descriptor, 1 << 20):
            chunks.append(chunk)
        _assert_descriptor_unchanged(anchor)
        return b"".join(chunks)
    finally:
        os.close(anchor.descriptor)
        os.close(anchor.parent_descriptor)


def _operational_stat(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _hash_stable(
    path: Path, label: str, *, trusted_root: Path
) -> tuple[str, int, dict[str, int]]:
    anchor = _stable_descriptor(path, label, trusted_root=trusted_root)
    digest = hashlib.sha256()
    try:
        while chunk := os.read(anchor.descriptor, 8 << 20):
            digest.update(chunk)
        _assert_descriptor_unchanged(anchor)
        final = os.fstat(anchor.descriptor)
        return digest.hexdigest(), int(final.st_size), _operational_stat(final)
    finally:
        os.close(anchor.descriptor)
        os.close(anchor.parent_descriptor)


def _read_canonical_json(
    path: Path, label: str, *, trusted_root: Path
) -> tuple[dict[str, Any], bytes]:
    payload = _read_stable(path, label, trusted_root=trusted_root)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NumericUnitVerificationError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise NumericUnitVerificationError(f"{label} is not a canonical JSON object")
    return value, payload


def _parse_canonical_json_payload(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NumericUnitVerificationError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise NumericUnitVerificationError(f"{label} is not canonical object JSON")
    return value


def _read_json_loose(
    path: Path, label: str, *, trusted_root: Path
) -> dict[str, Any]:
    payload = _read_stable(path, label, trusted_root=trusted_root)
    return _parse_json_loose_payload(payload, label)


def _parse_json_loose_payload(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NumericUnitVerificationError(f"{label} is not JSON") from exc
    if not isinstance(value, dict):
        raise NumericUnitVerificationError(f"{label} is not a JSON object")
    return value


def _safe_component(value: Any, label: str) -> str:
    text = str(value)
    if (
        not text
        or text in {".", ".."}
        or Path(text).name != text
        or "/" in text
        or "\\" in text
    ):
        raise NumericUnitVerificationError(f"Unsafe {label} component: {text!r}")
    return text


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    text = str(value)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
    ):
        raise NumericUnitVerificationError(f"Unsafe {label}: {text!r}")
    return path


def _project_path(settings: VerificationSettings, value: Any, label: str) -> Path:
    relative = _safe_relative(value, label)
    path = _lexical_absolute(settings.project_root / Path(*relative.parts))
    try:
        path.relative_to(settings.project_root)
    except ValueError as exc:
        raise NumericUnitVerificationError(f"{label} escapes project root") from exc
    return path


def _raw_path(settings: VerificationSettings, filename: Any, label: str) -> Path:
    name = _safe_component(filename, label)
    path = _lexical_absolute(settings.raw_root / name)
    if path.parent != settings.raw_root:
        raise NumericUnitVerificationError(f"{label} escapes raw root")
    return path


def _closed_directory(
    path: Path,
    expected_names: Sequence[str],
    label: str,
    *,
    trusted_root: Path,
) -> tuple[dict[str, bytes], _ClosedDirectoryState]:
    path = _lexical_absolute(path)
    trusted_root = _lexical_absolute(trusted_root)
    expected = tuple(_safe_component(name, f"{label} file") for name in expected_names)
    if len(expected) != len(set(expected)):
        raise NumericUnitVerificationError(f"{label} expected file set duplicates")
    descriptor, opened = _open_directory_fd_secure(
        path, label, create=False, trusted_root=trusted_root
    )
    try:
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as exc:
            raise NumericUnitVerificationError(f"Cannot list anchored {label}") from exc
        if set(names) != set(expected) or len(names) != len(expected):
            raise NumericUnitVerificationError(
                f"{label} closed file set differs: {names}"
            )
        observed = {
            name: _read_regular_at(descriptor, name, f"{label}/{name}")
            for name in names
        }
        _assert_directory_path_identity(
            path, opened, label, trusted_root=trusted_root
        )
        state = _ClosedDirectoryState(
            path=path,
            trusted_root=trusted_root,
            label=label,
            device=int(opened.st_dev),
            inode=int(opened.st_ino),
            names=tuple(names),
            hashes=tuple(
                (name, sha256_bytes(observed[name])) for name in names
            ),
        )
        return observed, state
    finally:
        os.close(descriptor)


def _parse_toml_payload(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise NumericUnitVerificationError(f"{label} is not UTF-8 TOML") from exc
    if not isinstance(value, dict):
        raise NumericUnitVerificationError(f"{label} is not a TOML document")
    return value


def _normalized_override(value: Path | None) -> Path | None:
    return None if value is None else _lexical_absolute(value)


def _derive_selected_paths(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    paths: Mapping[str, Any],
    phase: Mapping[str, Any],
    provenance: SettingsProvenance,
) -> dict[str, Path]:
    try:
        configured_database = Path(paths["database"]["root"]).expanduser()
        selected_database_value: str | Path = (
            provenance.database_root_override
            or provenance.database_environment_override
            or configured_database
        )
        selected_database = _lexical_absolute(selected_database_value)
        selected_artifact = _lexical_absolute(
            provenance.artifact_root_override
            or project_root / "artifacts/phase_02_canonical_panel"
        )
        selected_raw = _lexical_absolute(
            provenance.raw_root_override
            or selected_database / paths["database"]["raw"]
        )
        selected_quality = _lexical_absolute(
            provenance.quality_root_override
            or selected_database
            / paths["database"]["quality"]
            / phase["storage"]["quality_namespace"]
        )
        selected_interim = _lexical_absolute(
            selected_database
            / paths["database"]["interim"]
            / phase["storage"]["interim_namespace"]
        )
        phase_path = _lexical_absolute(
            provenance.phase_config_override
            or project_root / config["upstream"]["phase_02"]["phase_config"]
        )
        pointer = _lexical_absolute(
            provenance.primary_pointer_override
            or selected_artifact / config["outputs"]["primary_pointer"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NumericUnitVerificationError(
            f"Cannot derive verifier settings from authenticated TOML: {exc}"
        ) from exc
    return {
        "phase_config_path": phase_path,
        "primary_pointer_path": pointer,
        "artifact_root": selected_artifact,
        "database_root": selected_database,
        "raw_root": selected_raw,
        "interim_root": selected_interim,
        "quality_root": selected_quality,
    }


def load_settings(
    profile_config_path: Path = DEFAULT_PROFILE_CONFIG,
    paths_config_path: Path = DEFAULT_PATHS_CONFIG,
    phase_config_path: Path | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
    primary_pointer_path: Path | None = None,
    artifact_root: Path | None = None,
    database_root: Path | None = None,
    raw_root: Path | None = None,
    quality_root: Path | None = None,
) -> VerificationSettings:
    project_root = _lexical_absolute(project_root)
    profile_config_path = _lexical_absolute(profile_config_path)
    paths_config_path = _lexical_absolute(paths_config_path)
    profile_payload = _read_stable(
        profile_config_path,
        "numeric-unit profile config",
        trusted_root=project_root,
    )
    paths_payload = _read_stable(
        paths_config_path, "paths config", trusted_root=project_root
    )
    config = _parse_toml_payload(profile_payload, "numeric-unit profile config")
    paths = _parse_toml_payload(paths_payload, "paths config")
    explicit_phase = _normalized_override(phase_config_path)
    environment_database = (
        None
        if database_root is not None
        else os.environ.get("PORTFOLIO_DATABASE_ROOT")
    )
    provenance = SettingsProvenance(
        phase_config_override=explicit_phase,
        primary_pointer_override=_normalized_override(primary_pointer_path),
        artifact_root_override=_normalized_override(artifact_root),
        database_root_override=_normalized_override(database_root),
        database_environment_override=environment_database,
        raw_root_override=_normalized_override(raw_root),
        quality_root_override=_normalized_override(quality_root),
    )
    provisional_phase_path = _lexical_absolute(
        explicit_phase
        or project_root / config["upstream"]["phase_02"]["phase_config"]
    )
    phase_payload = _read_stable(
        provisional_phase_path, "phase config", trusted_root=project_root
    )
    phase = _parse_toml_payload(phase_payload, "phase config")
    selected = _derive_selected_paths(
        project_root=project_root,
        config=config,
        paths=paths,
        phase=phase,
        provenance=provenance,
    )
    return VerificationSettings(
        project_root=project_root,
        profile_config_path=profile_config_path,
        paths_config_path=paths_config_path,
        phase_config_path=selected["phase_config_path"],
        primary_pointer_path=selected["primary_pointer_path"],
        artifact_root=selected["artifact_root"],
        database_root=selected["database_root"],
        raw_root=selected["raw_root"],
        interim_root=selected["interim_root"],
        quality_root=selected["quality_root"],
        config=config,
        profile_config_payload=profile_payload,
        paths_config_payload=paths_payload,
        phase_config_payload=phase_payload,
        provenance=provenance,
    )


def _authenticate_settings(
    settings: VerificationSettings,
    expected: AuthenticatedSettings | None = None,
) -> tuple[AuthenticatedSettings, dict[str, Any]]:
    profile_payload = _read_stable(
        settings.profile_config_path,
        "numeric-unit profile config",
        trusted_root=settings.project_root,
    )
    paths_payload = _read_stable(
        settings.paths_config_path,
        "paths config",
        trusted_root=settings.project_root,
    )
    phase_payload = _read_stable(
        settings.phase_config_path,
        "phase config",
        trusted_root=settings.project_root,
    )
    if profile_payload != settings.profile_config_payload:
        raise NumericUnitVerificationError(
            "Profile config bytes differ from settings provenance"
        )
    if paths_payload != settings.paths_config_payload:
        raise NumericUnitVerificationError(
            "Paths config bytes differ from settings provenance"
        )
    if phase_payload != settings.phase_config_payload:
        raise NumericUnitVerificationError(
            "Phase config bytes differ from settings provenance; "
            "Configured phase_02_config SHA-256 lock differs"
        )
    config = _parse_toml_payload(profile_payload, "numeric-unit profile config")
    paths = _parse_toml_payload(paths_payload, "paths config")
    phase = _parse_toml_payload(phase_payload, "phase config")
    if config != settings.config:
        raise NumericUnitVerificationError(
            "In-memory profile config differs from authenticated profile bytes"
        )
    selected = _derive_selected_paths(
        project_root=settings.project_root,
        config=config,
        paths=paths,
        phase=phase,
        provenance=settings.provenance,
    )
    locked_phase_path = _lexical_absolute(
        settings.project_root / config["upstream"]["phase_02"]["phase_config"]
    )
    if selected["phase_config_path"] != locked_phase_path:
        raise NumericUnitVerificationError(
            "Selected phase config differs from the locked profile path"
        )
    for field, selected_value in selected.items():
        actual = getattr(settings, field)
        if actual != selected_value:
            label = field.replace("_", " ")
            raise NumericUnitVerificationError(
                f"Selected {label} differs or escapes project authenticated provenance"
            )
    try:
        outputs = config["outputs"]
        authenticated = AuthenticatedSettings(
            profile_config_payload=profile_payload,
            paths_config_payload=paths_payload,
            phase_config_payload=phase_payload,
            profile_config_sha256=sha256_bytes(profile_payload),
            paths_config_sha256=sha256_bytes(paths_payload),
            phase_config_sha256=sha256_bytes(phase_payload),
            project_root=settings.project_root,
            profile_config_path=settings.profile_config_path,
            paths_config_path=settings.paths_config_path,
            phase_config_path=settings.phase_config_path,
            primary_pointer_path=selected["primary_pointer_path"],
            artifact_root=selected["artifact_root"],
            database_root=selected["database_root"],
            raw_root=selected["raw_root"],
            interim_root=selected["interim_root"],
            quality_root=selected["quality_root"],
            artifact_subdirectory=_safe_component(
                outputs["artifact_subdirectory"], "artifact subdirectory"
            ),
            quality_subdirectory=_safe_component(
                outputs["quality_subdirectory"], "quality subdirectory"
            ),
            verification_subdirectory=_safe_component(
                outputs["verification_subdirectory"],
                "verification subdirectory",
            ),
            primary_pointer_name=_safe_component(
                outputs["primary_pointer"], "primary pointer"
            ),
            verified_pointer_name=_safe_component(
                outputs["verified_pointer"], "verified pointer"
            ),
            closed_artifact_file_set=tuple(
                _safe_component(name, "closed artifact file")
                for name in outputs["closed_artifact_file_set"]
            ),
            closed_quality_file_set=tuple(
                _safe_component(name, "closed quality file")
                for name in outputs["closed_quality_file_set"]
            ),
            closed_verification_file_set=tuple(
                _safe_component(name, "closed verification file")
                for name in outputs["closed_verification_file_set"]
            ),
        )
    except (KeyError, TypeError) as exc:
        raise NumericUnitVerificationError(
            f"Authenticated output settings differ: {exc}"
        ) from exc
    for names, label in (
        (authenticated.closed_artifact_file_set, "artifact"),
        (authenticated.closed_quality_file_set, "quality"),
        (authenticated.closed_verification_file_set, "verification"),
    ):
        if len(names) != len(set(names)):
            raise NumericUnitVerificationError(
                f"Authenticated closed {label} set duplicates"
            )
    if expected is not None and authenticated != expected:
        raise NumericUnitVerificationError(
            "Authenticated settings changed since the selected context"
        )
    return authenticated, config


def _config_from_authenticated_settings(
    authenticated: AuthenticatedSettings,
) -> dict[str, Any]:
    return _parse_toml_payload(
        authenticated.profile_config_payload, "authenticated profile config"
    )


class _Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        check_id: str,
        category: str,
        *,
        expected: Any,
        actual: Any,
        passed: bool,
        evidence: str,
        message: str,
    ) -> None:
        self.rows.append(
            {
                "check_id": check_id,
                "category": category,
                "expected": expected,
                "actual": actual,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
                "message": message,
            }
        )

    def sorted(self) -> list[dict[str, Any]]:
        rows = sorted(self.rows, key=lambda row: row["check_id"].encode("utf-8"))
        ids = [row["check_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise NumericUnitVerificationError("Independent check IDs are duplicated")
        return rows


def _canonical_integer(value: str, label: str, *, signed: bool = False) -> int:
    pattern = r"(?:0|[1-9][0-9]*|-[1-9][0-9]*)" if signed else r"(?:0|[1-9][0-9]*)"
    if re.fullmatch(pattern, value, flags=re.ASCII) is None:
        raise NumericUnitVerificationError(f"{label} is not a canonical integer")
    return int(value)


def _parse_json_cell(value: str, label: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise NumericUnitVerificationError(f"{label} is not JSON") from exc
    if canonical_json_text(parsed) != value:
        raise NumericUnitVerificationError(f"{label} is not canonical JSON")
    return parsed


def _wire(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _render_csv(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
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
        writer.writerow({field: _wire(row[field]) for field in fields})
    return output.getvalue().encode("utf-8")


def _sort_component(value: Any) -> tuple[int, Any]:
    return (0, value) if isinstance(value, int) else (1, str(value).encode("utf-8"))


def _parse_table(payload: bytes, table: str) -> list[dict[str, Any]]:
    fields = CSV_FIELDS[table]
    try:
        text = payload.decode("utf-8")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader)
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise NumericUnitVerificationError(f"Cannot decode {table} CSV") from exc
    if tuple(header) != fields or "\x00" in text:
        raise NumericUnitVerificationError(f"{table} CSV header or encoding differs")
    nonnegative = NONNEGATIVE_INTEGER_FIELDS.get(table, frozenset())
    optional_signed = OPTIONAL_SIGNED_INTEGER_FIELDS.get(table, frozenset())
    optional_nonnegative = OPTIONAL_NONNEGATIVE_INTEGER_FIELDS.get(table, frozenset())
    rows: list[dict[str, Any]] = []
    try:
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(fields):
                raise NumericUnitVerificationError(
                    f"{table} row width differs at line {line_number}"
                )
            row: dict[str, Any] = {}
            for field, value in zip(fields, values, strict=True):
                label = f"{table}.{field} line {line_number}"
                if field in nonnegative:
                    row[field] = _canonical_integer(value, label)
                elif field in optional_signed:
                    row[field] = "" if value == "" else _canonical_integer(
                        value, label, signed=True
                    )
                elif field in optional_nonnegative:
                    row[field] = "" if value == "" else _canonical_integer(value, label)
                else:
                    row[field] = value
                if field in JSON_CELL_FIELDS.get(table, frozenset()):
                    _parse_json_cell(value, label)
            rows.append(row)
    except csv.Error as exc:
        raise NumericUnitVerificationError(f"{table} is not strict RFC CSV") from exc
    if _render_csv(fields, rows) != payload:
        raise NumericUnitVerificationError(f"{table} CSV wire representation differs")
    expected_order = sorted(
        rows,
        key=lambda row: tuple(
            _sort_component(row[field]) for field in SORT_FIELDS[table]
        ),
    )
    if rows != expected_order:
        raise NumericUnitVerificationError(f"{table} rows are not canonically sorted")
    return rows


def _parse_hash_inventory(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NumericUnitVerificationError("Output hash inventory is not UTF-8") from exc
    if text and not text.endswith("\n"):
        raise NumericUnitVerificationError("Output hash inventory lacks trailing LF")
    result: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or SHA256_RE.fullmatch(parts[0]) is None or not parts[1]:
            raise NumericUnitVerificationError("Output hash inventory is malformed")
        if parts[1] in result:
            raise NumericUnitVerificationError("Output hash inventory has duplicates")
        result[parts[1]] = parts[0]
    if list(result) != sorted(result):
        raise NumericUnitVerificationError("Output hash inventory is not sorted")
    return result


def authenticate_primary_bundle(settings: VerificationSettings) -> dict[str, Any]:
    settings_auth, config = _authenticate_settings(settings)
    final_snapshot: dict[Path, str] = {}
    pointer, pointer_payload = _read_canonical_json(
        settings_auth.primary_pointer_path,
        "numeric-unit primary pointer",
        trusted_root=settings_auth.artifact_root,
    )
    final_snapshot[settings_auth.primary_pointer_path] = sha256_bytes(pointer_payload)
    expected_pointer_keys = {
        "artifact_path", "build_id", "contract_id", "implementation",
        "manifest_sha256", "schema_version", "status",
    }
    if set(pointer) != expected_pointer_keys:
        raise NumericUnitVerificationError("Primary pointer schema differs")
    if pointer["schema_version"] != SCHEMA_VERSION or pointer["status"] != PRIMARY_STATUS:
        raise NumericUnitVerificationError("Primary pointer status/version differs")
    build_id = _safe_component(pointer["build_id"], "build_id")
    if BUILD_ID_RE.fullmatch(build_id) is None:
        raise NumericUnitVerificationError("Primary build_id is malformed")
    contract_id = _safe_component(pointer["contract_id"], "contract_id")
    locked_contract_id = _safe_component(
        config["upstream"]["contract"]["contract_id"],
        "locked contract_id",
    )
    if contract_id != locked_contract_id:
        raise NumericUnitVerificationError(
            "Primary contract_id differs from the locked upstream contract_id"
        )
    artifact_subdirectory = settings_auth.artifact_subdirectory
    artifact_directory = _lexical_absolute(
        settings_auth.artifact_root / artifact_subdirectory / build_id
    )
    if _project_path(settings, pointer["artifact_path"], "primary artifact path") != artifact_directory:
        raise NumericUnitVerificationError("Primary artifact path differs")
    artifact_payloads, artifact_state = _closed_directory(
        artifact_directory,
        settings_auth.closed_artifact_file_set,
        "numeric-unit primary artifact",
        trusted_root=settings_auth.artifact_root,
    )
    if set(artifact_payloads) != set(ARTIFACT_FILES):
        raise NumericUnitVerificationError("Primary artifact contract differs")
    for name, payload in artifact_payloads.items():
        final_snapshot[artifact_directory / name] = sha256_bytes(payload)
    manifest_payload = artifact_payloads["manifest.json"]
    manifest_sha = sha256_bytes(manifest_payload)
    if pointer["manifest_sha256"] != manifest_sha:
        raise NumericUnitVerificationError("Primary manifest hash differs from pointer")
    manifest = _parse_canonical_json_payload(manifest_payload, "primary manifest")
    if set(manifest) != {
        "binding", "build_id", "contract_id", "outputs", "profile_name",
        "schema_version", "status", "summary",
    }:
        raise NumericUnitVerificationError("Primary manifest schema differs")
    if any(
        (
            manifest["build_id"] != build_id,
            manifest["contract_id"] != contract_id,
            manifest["profile_name"] != PROFILE_NAME,
            manifest["schema_version"] != SCHEMA_VERSION,
            manifest["status"] != PRIMARY_STATUS,
        )
    ):
        raise NumericUnitVerificationError("Primary manifest identity differs")
    binding = manifest["binding"]
    if not isinstance(binding, dict) or set(binding) != {
        "contract_id", "implementation_sha256", "policy_sha256",
        "empirical_lock_sha256", "profile_config_sha256",
        "snapshot_inventory_sha256",
    }:
        raise NumericUnitVerificationError("Primary manifest binding schema differs")
    if binding["contract_id"] != contract_id:
        raise NumericUnitVerificationError("Manifest contract binding differs")
    for key in (
        "implementation_sha256", "policy_sha256", "empirical_lock_sha256",
        "profile_config_sha256", "snapshot_inventory_sha256",
    ):
        if SHA256_RE.fullmatch(str(binding[key])) is None:
            raise NumericUnitVerificationError(f"Manifest {key} is not SHA-256")

    implementation = pointer["implementation"]
    if not isinstance(implementation, dict) or set(implementation) != set(
        PRIMARY_IMPLEMENTATION_PATHS
    ):
        raise NumericUnitVerificationError("Primary implementation schema differs")
    for key, relative in PRIMARY_IMPLEMENTATION_PATHS.items():
        path = _project_path(settings, relative, f"primary {key}")
        payload = _read_stable(
            path, f"primary {key}", trusted_root=settings_auth.project_root
        )
        digest = sha256_bytes(payload)
        final_snapshot[path] = digest
        if implementation[key] != digest:
            raise NumericUnitVerificationError(f"Primary {key} bytes differ")
    if binding["implementation_sha256"] != implementation["compiler_sha256"]:
        raise NumericUnitVerificationError("Compiler implementation binding differs")

    quality_subdirectory = settings_auth.quality_subdirectory
    quality_directory = _lexical_absolute(
        settings_auth.quality_root / contract_id / quality_subdirectory / build_id
    )
    quality_payloads, quality_state = _closed_directory(
        quality_directory,
        settings_auth.closed_quality_file_set,
        "numeric-unit primary quality",
        trusted_root=settings_auth.quality_root,
    )
    if set(quality_payloads) != set(QUALITY_FILES):
        raise NumericUnitVerificationError("Primary quality contract differs")
    for name, payload in quality_payloads.items():
        final_snapshot[quality_directory / name] = sha256_bytes(payload)
    try:
        quality_relative = quality_directory.relative_to(settings_auth.database_root)
    except ValueError as exc:
        raise NumericUnitVerificationError("Quality directory escapes database root") from exc
    if not quality_relative.parts or quality_relative.parts[0] != "quality":
        raise NumericUnitVerificationError("Quality output is outside quality namespace")
    expected_outputs = [
        {
            "path": (quality_relative / name).as_posix(),
            "sha256": sha256_bytes(quality_payloads[name]),
            "size_bytes": len(quality_payloads[name]),
        }
        for name in sorted(quality_payloads)
    ]
    if manifest["outputs"] != expected_outputs:
        raise NumericUnitVerificationError("Manifest outputs differ from quality bytes")
    if _parse_hash_inventory(artifact_payloads["output_hashes.sha256"]) != {
        item["path"]: item["sha256"] for item in expected_outputs
    }:
        raise NumericUnitVerificationError("Output hash inventory differs")

    json_documents = {
        name: _parse_canonical_json_payload(payload, f"primary {name}")
        for name, payload in quality_payloads.items()
        if name.endswith(".json")
    }
    tables = {
        table: _parse_table(quality_payloads[filename], table)
        for table, filename in CSV_FILENAMES.items()
    }
    if manifest["summary"] != json_documents["summary.json"]:
        raise NumericUnitVerificationError("Manifest summary differs from summary.json")
    return {
        "artifact_directory": artifact_directory,
        "binding": binding,
        "build_id": build_id,
        "closed_directories": (artifact_state, quality_state),
        "contract_id": contract_id,
        "final_snapshot": final_snapshot,
        "json": json_documents,
        "manifest": manifest,
        "manifest_path": artifact_directory / "manifest.json",
        "manifest_sha256": manifest_sha,
        "pointer": pointer,
        "quality_directory": quality_directory,
        "quality_payloads": quality_payloads,
        "settings_auth": settings_auth,
        "tables": tables,
    }


def _descriptor_inventory_sha(descriptors: Sequence[Mapping[str, Any]]) -> str:
    required = {"relative_path", "root", "sha256", "size_bytes", "snapshot_id"}
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    locations: set[tuple[str, str]] = set()
    for item in descriptors:
        if not isinstance(item, Mapping) or set(item) != required:
            raise NumericUnitVerificationError("Portable snapshot schema differs")
        snapshot_id = str(item["snapshot_id"])
        root = str(item["root"])
        relative = _safe_relative(item["relative_path"], "snapshot relative path")
        digest = str(item["sha256"])
        size = item["size_bytes"]
        if not snapshot_id or snapshot_id in ids or any(ord(ch) < 32 for ch in snapshot_id):
            raise NumericUnitVerificationError("Portable snapshot ID differs")
        if root not in {"project", "raw"}:
            raise NumericUnitVerificationError("Portable snapshot root differs")
        if (root, relative.as_posix()) in locations:
            raise NumericUnitVerificationError("Portable snapshot location duplicates")
        if SHA256_RE.fullmatch(digest) is None:
            raise NumericUnitVerificationError("Portable snapshot hash differs")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise NumericUnitVerificationError("Portable snapshot size differs")
        ids.add(snapshot_id)
        locations.add((root, relative.as_posix()))
        normalized.append(
            {
                "relative_path": relative.as_posix(),
                "root": root,
                "sha256": digest,
                "size_bytes": size,
                "snapshot_id": snapshot_id,
            }
        )
    normalized.sort(key=lambda item: item["snapshot_id"])
    return sha256_bytes(SNAPSHOT_INVENTORY_DOMAIN + canonical_json_bytes(normalized))


def _project_snapshot_spec(config: Mapping[str, Any]) -> dict[str, str]:
    result = {
        "policy": str(config["policy"]["path"]),
        "empirical_lock": str(config["empirical_lock"]["path"]),
    }
    for snapshot_id, (section, key) in UPSTREAM_SNAPSHOT_PATHS.items():
        result[snapshot_id] = str(config["upstream"][section][key])
    return result


def _calendar_output_path(
    settings_auth: AuthenticatedSettings, value: Any, label: str
) -> Path:
    candidate = _lexical_absolute(str(value))
    if not Path(str(value)).is_absolute():
        raise NumericUnitVerificationError(f"{label} must be an absolute database path")
    allowed = False
    for root in (settings_auth.interim_root, settings_auth.quality_root):
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        allowed = True
        break
    if not allowed:
        raise NumericUnitVerificationError(
            f"{label} is outside the allowed database interim/quality roots"
        )
    return candidate


def _authenticate_calendar_chain(
    settings: VerificationSettings,
    settings_auth: AuthenticatedSettings,
    config: Mapping[str, Any],
    pointer: Mapping[str, Any],
    final_snapshot: dict[Path, str],
) -> dict[str, Any]:
    calendar = config["upstream"]["calendar"]
    contract_id = config["upstream"]["contract"]["contract_id"]
    required_pointer_fields = {
        "build_id", "contract_id", "primary_artifact_path",
        "primary_manifest_sha256", "schema_version", "status", "verification_id"
    }
    if not required_pointer_fields <= set(pointer):
        raise NumericUnitVerificationError("Calendar verified pointer chain is incomplete")
    if any(
        (
            pointer.get("schema_version") != SCHEMA_VERSION,
            pointer.get("build_id") != calendar["build_id"],
            pointer.get("contract_id") != contract_id,
            pointer.get("status") != calendar["required_status"],
            pointer.get("verification_id") != calendar["verification_id"],
            SHA256_RE.fullmatch(str(pointer.get("primary_manifest_sha256", "")))
            is None,
        )
    ):
        raise NumericUnitVerificationError("Calendar verified pointer identity differs")
    artifact_directory = _project_path(
        settings, pointer["primary_artifact_path"], "calendar primary artifact"
    )
    try:
        artifact_directory.relative_to(settings.artifact_root)
    except ValueError as exc:
        raise NumericUnitVerificationError(
            "Calendar primary artifact is outside the configured artifact root"
        ) from exc
    manifest_path = artifact_directory / "manifest.json"
    manifest_payload = _read_stable(
        manifest_path,
        "calendar primary manifest",
        trusted_root=settings_auth.artifact_root,
    )
    manifest_sha = sha256_bytes(manifest_payload)
    final_snapshot[manifest_path] = manifest_sha
    if manifest_sha != pointer["primary_manifest_sha256"]:
        raise NumericUnitVerificationError("Calendar primary manifest SHA-256 differs")
    manifest = _parse_canonical_json_payload(
        manifest_payload, "calendar primary manifest"
    )
    if any(
        (
            manifest.get("schema_version") != SCHEMA_VERSION,
            manifest.get("build_id") != calendar["build_id"],
            manifest.get("contract_id") != contract_id,
            manifest.get("status")
            != "calendar_primary_passed_pending_independent_verification",
            manifest.get("verification_id")
            != config["upstream"]["contract"]["verification_id"],
        )
    ):
        raise NumericUnitVerificationError("Calendar primary manifest identity differs")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise NumericUnitVerificationError("Calendar primary outputs are absent")
    matches = [
        item
        for item in outputs
        if isinstance(item, Mapping)
        and Path(str(item.get("path", ""))).name == "trading_calendar.csv"
    ]
    if len(matches) != 1:
        raise NumericUnitVerificationError(
            "Calendar primary manifest must contain exactly one trading_calendar.csv"
        )
    receipt = matches[0]
    if set(receipt) != {"path", "sha256", "size_bytes"}:
        raise NumericUnitVerificationError("Trading-calendar receipt schema differs")
    if (
        SHA256_RE.fullmatch(str(receipt["sha256"])) is None
        or isinstance(receipt["size_bytes"], bool)
        or not isinstance(receipt["size_bytes"], int)
        or receipt["size_bytes"] < 0
    ):
        raise NumericUnitVerificationError("Trading-calendar receipt value differs")
    calendar_path = _calendar_output_path(
        settings_auth, receipt["path"], "verified trading calendar"
    )
    calendar_root = _trusted_root_for_path(
        calendar_path,
        (settings_auth.interim_root, settings_auth.quality_root),
        "verified trading calendar",
    )
    calendar_payload = _read_stable(
        calendar_path,
        "verified trading calendar",
        trusted_root=calendar_root,
    )
    calendar_sha = sha256_bytes(calendar_payload)
    final_snapshot[calendar_path] = calendar_sha
    if calendar_sha != receipt["sha256"] or len(calendar_payload) != receipt["size_bytes"]:
        raise NumericUnitVerificationError("Verified trading-calendar bytes differ")
    return {
        "calendar_path": calendar_path,
        "manifest_path": manifest_path,
        "sha256": calendar_sha,
        "size_bytes": len(calendar_payload),
    }


def _assert_upstream_identities(
    settings: VerificationSettings,
    settings_auth: AuthenticatedSettings,
    config: Mapping[str, Any],
    project_payloads: Mapping[str, bytes],
    final_snapshot: dict[Path, str],
) -> dict[str, Any]:
    methodology = _parse_json_loose_payload(
        project_payloads["methodology_freeze"], "methodology freeze"
    )
    if methodology.get("status") != "frozen_preimplementation":
        raise NumericUnitVerificationError("Methodology freeze status differs")
    phase_01 = _parse_json_loose_payload(
        project_payloads["phase_01_freeze"], "Phase 1 freeze"
    )
    if phase_01.get("status") != "phase_01_frozen":
        raise NumericUnitVerificationError("Phase 1 freeze status differs")
    pointer_specs = {
        "contract_verified_pointer": ("contract", "contract_id"),
        "calendar_verified_pointer": ("calendar", "build_id"),
        "daily_monthly_profile_verified_pointer": (
            "daily_monthly_profile", "build_id"
        ),
        "supporting_profile_verified_pointer": ("supporting_profile", "build_id"),
    }
    for snapshot_id, (section, identity_key) in pointer_specs.items():
        locked = config["upstream"][section]
        pointer = _parse_json_loose_payload(
            project_payloads[snapshot_id], f"{section} pointer"
        )
        if any(
            (
                pointer.get("status") != locked["required_status"],
                pointer.get(identity_key) != locked[identity_key],
                pointer.get("verification_id") != locked["verification_id"],
            )
        ):
            raise NumericUnitVerificationError(f"{section} pointer identity differs")
        if section in {"daily_monthly_profile", "supporting_profile"} and (
            pointer.get("primary_manifest_sha256")
            != locked["primary_manifest_sha256"]
        ):
            raise NumericUnitVerificationError(
                f"{section} primary-manifest binding differs"
            )
    calendar_pointer = _parse_json_loose_payload(
        project_payloads["calendar_verified_pointer"], "calendar pointer"
    )
    return _authenticate_calendar_chain(
        settings, settings_auth, config, calendar_pointer, final_snapshot
    )


def _validate_empirical_lock(
    lock: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    empirical = config["empirical_lock"]
    if (
        lock.get("schema_version") != empirical["schema_version"]
        or lock.get("lock_id") != empirical["lock_id"]
    ):
        raise NumericUnitVerificationError("Empirical lock identity differs")
    contract = lock.get("measurement_projection_contract")
    if not isinstance(contract, Mapping) or any(
        (
            contract.get("algorithm_id")
            != "sha256_canonical_json_lf_projected_rows_v1",
            contract.get("checks_table_is_not_an_empirical_projection") is not True,
            contract.get("rows_are_sorted_by_profile_output_sort_keys_before_projection")
            is not True,
            contract.get("cell_values_are_preserved_without_reparsing") is not True,
        )
    ):
        raise NumericUnitVerificationError("Empirical projection contract differs")
    projections = lock.get("measurement_projections")
    if not isinstance(projections, Mapping) or set(projections) != set(EMPIRICAL_TABLES):
        raise NumericUnitVerificationError("Empirical projection table set differs")
    for table, expected in projections.items():
        if not isinstance(expected, Mapping) or set(expected) != {
            "policy_derived_fields_excluded", "row_count", "segments", "sha256"
        }:
            raise NumericUnitVerificationError(f"Empirical {table} projection schema differs")
        if expected["policy_derived_fields_excluded"] != sorted(
            MEASUREMENT_EXCLUDED_FIELDS[table]
        ) or set(expected["segments"]) != set(SEGMENTS):
            raise NumericUnitVerificationError(f"Empirical {table} projection contract differs")
        values = [expected, *expected["segments"].values()]
        if any(
            isinstance(value.get("row_count"), bool)
            or not isinstance(value.get("row_count"), int)
            or value["row_count"] < 0
            or SHA256_RE.fullmatch(str(value.get("sha256", ""))) is None
            for value in values
        ):
            raise NumericUnitVerificationError(f"Empirical {table} projection value differs")
        if expected["row_count"] != sum(
            expected["segments"][segment]["row_count"] for segment in SEGMENTS
        ):
            raise NumericUnitVerificationError(f"Empirical {table} row balance differs")
    cap = lock.get("transparent_aggregates", {}).get("capitalization", {}).get(
        "non_ADR_NS_N_primary_threshold"
    )
    if not isinstance(cap, Mapping):
        raise NumericUnitVerificationError("Empirical NS/N threshold is absent")
    if cap.get("selector") != {
        "ShareType": "NS",
        "effective_interval_ShrAdrFlg": "N",
        "interval_date": "MthPrcDt",
        "interval_match_count": 1,
        "primary_exchange_scope": "all_observed_no_exchange_filter",
        "required_valid_operands": ["MthCap", "MthPrc", "interval_ShrOut"],
    }:
        raise NumericUnitVerificationError("Empirical NS/N selector differs")
    if any(
        (
            cap.get("authoritative_formula_id")
            != "monthly_interval_cap_price_shares",
            cap.get("periodicity") != "monthly",
            cap.get("required_comparable_rows_greater_than_zero") is not True,
            cap.get("required_fail_rows") != 0,
            cap.get("required_multiple_share_match_rows") != 0,
            cap.get("required_zero_share_match_rows") != 0,
            cap.get("required_pass_rate") != "1",
            cap.get("development_threshold_must_be_evaluated_separately") is not True,
            cap.get("segments_are_enforced_independently") is not True,
        )
    ):
        raise NumericUnitVerificationError("Empirical NS/N threshold rule differs")
    if set(cap.get("segments", {})) != set(SEGMENTS):
        raise NumericUnitVerificationError("Empirical NS/N segment set differs")
    segment_counts: dict[str, dict[str, int]] = {}
    for segment in SEGMENTS:
        values = cap["segments"][segment]
        if not isinstance(values, Mapping) or set(values) != set(NON_ADR_COUNT_FIELDS):
            raise NumericUnitVerificationError("Empirical NS/N count schema differs")
        segment_counts[segment] = {field: int(values[field]) for field in NON_ADR_COUNT_FIELDS}
    return {
        "measurement_projections": projections,
        "non_adr_capitalization": cap,
        "segment_counts": segment_counts,
    }


def authenticate_locked_context(
    settings: VerificationSettings, primary: Mapping[str, Any]
) -> dict[str, Any]:
    settings_auth, config = _authenticate_settings(
        settings, primary["settings_auth"]
    )
    binding = primary["binding"]
    locked_contract_id = config["upstream"]["contract"]["contract_id"]
    if (
        primary["contract_id"] != locked_contract_id
        or binding["contract_id"] != locked_contract_id
    ):
        raise NumericUnitVerificationError(
            "Primary contract binding differs from the locked upstream contract"
        )
    final_snapshot: dict[Path, str] = dict(primary["final_snapshot"])
    profile_sha = settings_auth.profile_config_sha256
    final_snapshot[settings_auth.profile_config_path] = profile_sha
    if profile_sha != binding["profile_config_sha256"]:
        raise NumericUnitVerificationError("Profile config binding differs")
    final_snapshot[settings_auth.paths_config_path] = settings_auth.paths_config_sha256
    final_snapshot[settings_auth.phase_config_path] = settings_auth.phase_config_sha256
    if config.get("schema_version") != SCHEMA_VERSION or config.get("profile_name") != PROFILE_NAME:
        raise NumericUnitVerificationError("Profile config identity differs")
    if config.get("status") != LOCKED_PROFILE_STATUS:
        raise NumericUnitVerificationError("Profile config is not production locked")
    locks = config.get("production_locks")
    if not isinstance(locks, Mapping) or set(locks) != PRODUCTION_LOCK_KEYS or any(
        value is not True for value in locks.values()
    ):
        raise NumericUnitVerificationError("Production lock set differs")
    if config.get("capitalization_formulas", {}).get("primary_non_ADR_scale_formula") != "monthly_interval_formula":
        raise NumericUnitVerificationError("Primary capitalization formula differs")
    if config["outputs"]["schemas"] != {
        table: list(fields) for table, fields in CSV_FIELDS.items()
    }:
        raise NumericUnitVerificationError("Locked output schemas differ")
    if config["outputs"]["sort_keys"] != {
        table: list(fields) for table, fields in SORT_FIELDS.items()
    } | {
        "string_sort_rule": "ascending UTF-8 byte order",
        "integer_sort_rule": "ascending numeric order",
    }:
        raise NumericUnitVerificationError("Locked output sort contract differs")

    project_specs = _project_snapshot_spec(config)
    if set(project_specs) != {
        "policy", "empirical_lock", *UPSTREAM_SNAPSHOT_PATHS
    }:
        raise NumericUnitVerificationError("Configured project snapshot set differs")
    expected_descriptors: list[dict[str, Any]] = []
    project_payloads: dict[str, bytes] = {}
    for snapshot_id, relative in project_specs.items():
        path = _project_path(settings, relative, f"snapshot {snapshot_id}")
        payload = _read_stable(
            path,
            f"snapshot {snapshot_id}",
            trusted_root=settings_auth.project_root,
        )
        digest = sha256_bytes(payload)
        if snapshot_id == "policy":
            locked_digest = config["policy"]["sha256"]
        elif snapshot_id == "empirical_lock":
            locked_digest = config["empirical_lock"]["sha256"]
        else:
            section, path_key, sha_key = UPSTREAM_SNAPSHOT_LOCKS[snapshot_id]
            if relative != config["upstream"][section][path_key]:
                raise NumericUnitVerificationError(
                    f"Configured {snapshot_id} path lock differs"
                )
            locked_digest = config["upstream"][section][sha_key]
        if SHA256_RE.fullmatch(str(locked_digest)) is None or digest != locked_digest:
            raise NumericUnitVerificationError(
                f"Configured {snapshot_id} SHA-256 lock differs"
            )
        project_payloads[snapshot_id] = payload
        final_snapshot[path] = digest
        expected_descriptors.append(
            {
                "snapshot_id": snapshot_id,
                "root": "project",
                "relative_path": _safe_relative(relative, snapshot_id).as_posix(),
                "sha256": digest,
                "size_bytes": len(payload),
            }
        )
    if sha256_bytes(project_payloads["policy"]) != config["policy"]["sha256"] or binding["policy_sha256"] != config["policy"]["sha256"]:
        raise NumericUnitVerificationError("Policy binding differs")
    try:
        policy = tomllib.loads(project_payloads["policy"].decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise NumericUnitVerificationError("Policy is not TOML") from exc
    if policy.get("status") != config["policy"]["required_status"]:
        raise NumericUnitVerificationError("Policy status differs")
    empirical_sha = sha256_bytes(project_payloads["empirical_lock"])
    if empirical_sha != config["empirical_lock"]["sha256"] or binding["empirical_lock_sha256"] != empirical_sha:
        raise NumericUnitVerificationError("Empirical-lock binding differs")
    try:
        empirical_lock = json.loads(project_payloads["empirical_lock"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NumericUnitVerificationError("Empirical lock is not JSON") from exc
    if not isinstance(empirical_lock, dict):
        raise NumericUnitVerificationError("Empirical lock is not an object")
    empirical = _validate_empirical_lock(empirical_lock, config)
    if project_payloads["phase_02_config"] != settings_auth.phase_config_payload:
        raise NumericUnitVerificationError(
            "Authenticated phase config bytes differ from the locked snapshot bytes"
        )
    calendar_chain = _assert_upstream_identities(
        settings, settings_auth, config, project_payloads, final_snapshot
    )
    phase_config_expected = _project_path(
        settings, config["upstream"]["phase_02"]["phase_config"], "phase config"
    )
    if phase_config_expected != settings_auth.phase_config_path:
        raise NumericUnitVerificationError("Selected phase config differs")

    raw_operational: dict[Path, dict[str, int]] = {}
    raw_computed: dict[str, dict[str, Any]] = {}
    for source in SOURCE_NAMES:
        configured = config["sources"][source]
        path = _raw_path(settings, configured["filename"], f"raw {source}")
        digest, size, operational = _hash_stable(
            path, f"raw {source}", trusted_root=settings_auth.raw_root
        )
        raw_operational[path] = operational
        if digest != configured["sha256"] or size != configured["size_bytes"]:
            raise NumericUnitVerificationError(f"Raw {source} binding differs")
        raw_computed[source] = {"sha256": digest, "size_bytes": size}
        expected_descriptors.append(
            {
                "snapshot_id": RAW_SNAPSHOT_IDS[source],
                "root": "raw",
                "relative_path": configured["filename"],
                "sha256": digest,
                "size_bytes": size,
            }
        )
    expected_descriptors.sort(key=lambda item: item["snapshot_id"])
    if (
        len(expected_descriptors) != len(EXPECTED_SNAPSHOT_IDS)
        or {item["snapshot_id"] for item in expected_descriptors}
        != EXPECTED_SNAPSHOT_IDS
    ):
        raise NumericUnitVerificationError("Exact portable snapshot set differs")

    receipt = primary["json"]["input_receipt.json"]
    if set(receipt) != {
        "authentication", "portable_snapshots", "raw_sources", "schema_version", "upstreams"
    } or receipt["schema_version"] != SCHEMA_VERSION:
        raise NumericUnitVerificationError("Input receipt schema differs")
    if receipt["authentication"] != {"raw_inputs": True, "upstreams": True}:
        raise NumericUnitVerificationError("Input receipt is not fully authenticated")
    if receipt["portable_snapshots"] != expected_descriptors:
        raise NumericUnitVerificationError("Portable snapshot descriptors differ")
    if _descriptor_inventory_sha(expected_descriptors) != binding["snapshot_inventory_sha256"]:
        raise NumericUnitVerificationError("Snapshot inventory binding differs")
    by_id = {item["snapshot_id"]: item for item in expected_descriptors}
    expected_upstreams = [
        {
            "relative_path": by_id[snapshot_id]["relative_path"],
            "sha256": by_id[snapshot_id]["sha256"],
            "size_bytes": by_id[snapshot_id]["size_bytes"],
            "snapshot_id": snapshot_id,
        }
        for snapshot_id in sorted(UPSTREAM_SNAPSHOT_PATHS)
    ]
    if receipt["upstreams"] != expected_upstreams:
        raise NumericUnitVerificationError("Input upstream receipts differ")
    raw_sources = receipt["raw_sources"]
    if not isinstance(raw_sources, list) or [item.get("source") for item in raw_sources] != list(SOURCE_NAMES):
        raise NumericUnitVerificationError("Raw source receipt set/order differs")
    raw_by_source: dict[str, dict[str, Any]] = {}
    for item in raw_sources:
        if not isinstance(item, dict) or set(item) != {
            "canonical_rows", "filename", "full_rows", "retained_rows",
            "sha256", "size_bytes", "source",
        }:
            raise NumericUnitVerificationError("Raw source receipt schema differs")
        raw_by_source[item["source"]] = item
    source_expectations = empirical_lock.get("source_expectations")
    if not isinstance(source_expectations, Mapping) or set(source_expectations) != set(SOURCE_NAMES):
        raise NumericUnitVerificationError("Empirical source expectation set differs")
    summary = primary["json"]["summary.json"]
    for source in SOURCE_NAMES:
        configured = config["sources"][source]
        receipt_row = raw_by_source[source]
        canonical = configured.get("verified_canonical_rows", configured["verified_retained_rows"])
        expected_counts = {
            "canonical_rows": canonical,
            "full_rows": configured["full_file_rows"],
            "retained_rows": configured["verified_retained_rows"],
        }
        expected_receipt = {
            **expected_counts,
            "filename": configured["filename"],
            "sha256": raw_computed[source]["sha256"],
            "size_bytes": raw_computed[source]["size_bytes"],
            "source": source,
        }
        if receipt_row != expected_receipt or summary["source_counts"][source] != expected_counts:
            raise NumericUnitVerificationError(f"{source} source counts differ")
        empirical_source = source_expectations[source]
        for key, expected in (
            ("filename", configured["filename"]),
            ("sha256", configured["sha256"]),
            ("size_bytes", configured["size_bytes"]),
            ("full_file_rows", configured["full_file_rows"]),
            ("retained_rows", configured["verified_retained_rows"]),
            ("canonical_rows", canonical),
        ):
            if empirical_source.get(key) != expected:
                raise NumericUnitVerificationError(
                    f"Empirical {source} source binding differs for {key}"
                )
    return {
        "empirical": empirical,
        "empirical_lock": empirical_lock,
        "calendar_chain": calendar_chain,
        "expected_descriptors": expected_descriptors,
        "final_snapshot": final_snapshot,
        "policy": policy,
        "raw_by_source": raw_by_source,
        "raw_operational": raw_operational,
        "settings_auth": settings_auth,
    }


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    with localcontext() as context:
        context.prec = 50
        value = Decimal(numerator) / Decimal(denominator)
        value = value.quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN)
    rendered = format(value, "f").rstrip("0").rstrip(".")
    return rendered or "0"


def _expected_numeric_status(row: Mapping[str, Any]) -> tuple[str, str]:
    hard_outside = {
        "DlyRet", "DlyRetx", "DlyVol", "MthRet", "MthRetx", "MthDisCnt",
        "MthVol", "MthPrcVolMissCnt", "ShrOut",
    }
    hard_failures = (
        row["malformed_rows"] + row["nonfinite_rows"] + row["overflow_rows"]
        + (row["valid_out_of_domain_rows"] if row["column"] in hard_outside else 0)
    )
    hard_rule = row["column"] in hard_outside
    diagnostic = row["valid_out_of_domain_rows"] > 0 and hard_failures == 0
    status = "FAIL" if hard_failures else "DIAGNOSTIC" if diagnostic else "PASS"
    severity = "hard" if hard_rule or hard_failures else "diagnostic"
    return status, severity


def _inventory_semantics(
    tables: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    differences: dict[str, list[dict[str, Any]]] = {
        table: [] for table in EMPIRICAL_TABLES
    }
    for index, row in enumerate(tables["numeric_inventory"], start=2):
        states = sum(
            row[field]
            for field in (
                "blank_rows", "valid_in_domain_rows", "valid_out_of_domain_rows",
                "malformed_rows", "nonfinite_rows", "overflow_rows",
            )
        )
        quantum = _parse_json_cell(row["quantum_counts"], "numeric quantum_counts")
        expected_status, expected_severity = _expected_numeric_status(row)
        valid = (
            isinstance(quantum, dict)
            and all(type(value) is int and value >= 0 for value in quantum.values())
            and row["population_rows"] == states
            and row["negative_zero_rows"]
            <= row["valid_in_domain_rows"] + row["valid_out_of_domain_rows"]
            and sum(quantum.values())
            == row["valid_in_domain_rows"] + row["valid_out_of_domain_rows"]
            and row["status"] == expected_status
            and row["severity"] == expected_severity
        )
        if not valid:
            differences["numeric_inventory"].append({"line": index, "column": row["column"]})
    for index, row in enumerate(tables["return_inventory"], start=2):
        coherent = row["coherent_rows"]
        incoherent = row["incoherent_rows"]
        locked = coherent != "" or incoherent != ""
        expected_status = (
            "FAIL" if locked and incoherent else "PASS" if locked else "DIAGNOSTIC"
        )
        expected_severity = "hard" if locked else "diagnostic"
        balanced = (
            coherent != ""
            and incoherent != ""
            and row["population_rows"] == coherent + incoherent
        ) if locked else coherent == incoherent == ""
        if not balanced or row["status"] != expected_status or row["severity"] != expected_severity:
            differences["return_inventory"].append({"line": index, "return_flag": row["return_flag"]})
    for index, row in enumerate(tables["identity_inventory"], start=2):
        exclusions = _parse_json_cell(row["exclusions"], "identity exclusions")
        expected_status = (
            "FAIL" if row["severity"] == "hard" and row["fail_rows"]
            else "DIAGNOSTIC"
            if row["fail_rows"] or (row["population_rows"] > 0 and row["comparable_rows"] == 0)
            else "PASS"
        )
        valid = (
            isinstance(exclusions, dict)
            and all(type(value) is int and value >= 0 for value in exclusions.values())
            and row["severity"] in {"hard", "diagnostic"}
            and row["population_rows"] == row["comparable_rows"] + row["excluded_rows"]
            and row["comparable_rows"] == row["pass_rows"] + row["fail_rows"]
            and row["excluded_rows"] == sum(exclusions.values())
            and row["coverage_rate"] == _rate(row["comparable_rows"], row["population_rows"])
            and row["pass_rate"] == _rate(row["pass_rows"], row["comparable_rows"])
            and row["status"] == expected_status
        )
        if not valid:
            differences["identity_inventory"].append({"line": index, "identity_id": row["identity_id"]})
    threshold = "full_population_Decimal_ratio_quantiles_are_diagnostic_only_hard_threshold_uses_exact_quantization_overlap"
    for index, row in enumerate(tables["capitalization_diagnostics"], start=2):
        exclusions = _parse_json_cell(row["exclusions"], "capitalization exclusions")
        hard = row["multiple_share_match_rows"] > 0
        valid = (
            isinstance(exclusions, dict)
            and all(type(value) is int and value >= 0 for value in exclusions.values())
            and row["population_rows"] == row["comparable_rows"] + row["excluded_rows"]
            and row["comparable_rows"] == row["formula_pass_rows"] + row["formula_fail_rows"]
            and row["excluded_rows"] == sum(exclusions.values())
            and row["population_rows"]
            == row["zero_share_match_rows"] + row["one_share_match_rows"] + row["multiple_share_match_rows"]
            and row["formula_pass_rate"] == _rate(row["formula_pass_rows"], row["comparable_rows"])
            and row["threshold_rule"] == threshold
            and row["status"] == ("FAIL" if hard else "DIAGNOSTIC")
            and row["severity"] == ("hard" if hard else "diagnostic")
        )
        if not valid:
            differences["capitalization_diagnostics"].append({"line": index, "formula_id": row["formula_id"]})
    return differences


def _identity_zero_comparable_segments(
    rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    pooled: dict[tuple[str, str, int, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["severity"] != "hard":
            continue
        key = (
            row["identity_id"], row["source"], row["year"], row["period_segment"]
        )
        pooled[key][0] += row["population_rows"]
        pooled[key][1] += row["comparable_rows"]
    result = [
        {
            "identity_id": key[0], "source": key[1], "year": key[2],
            "period_segment": key[3], "population_rows": values[0],
            "comparable_rows": values[1],
        }
        for key, values in pooled.items()
        if values[0] > 0 and values[1] == 0
    ]
    return sorted(
        result,
        key=lambda row: (
            row["identity_id"].encode("utf-8"), row["source"].encode("utf-8"),
            row["year"], row["period_segment"].encode("utf-8"),
        ),
    )


IDENTITY_REFERENCE_COLUMNS = {
    "daily_price_volume": ("daily_price_volume", "daily", "daily", "DlyPrc"),
    "monthly_daily_price": ("monthly_daily_price", "monthly_daily", "monthly", "MthPrc"),
    "monthly_daily_cap": ("monthly_daily_cap", "monthly_daily", "monthly", "MthCap"),
}


def _identity_population_differences(
    numeric_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected: Counter[tuple[str, str, int, str]] = Counter()
    reference_pairs = {(value[2], value[3]) for value in IDENTITY_REFERENCE_COLUMNS.values()}
    for row in numeric_rows:
        if (row["source"], row["column"]) in reference_pairs:
            expected[(row["source"], row["column"], row["year"], row["period_segment"])] += row["population_rows"]
    actual: Counter[tuple[str, str, int, str]] = Counter()
    for row in identity_rows:
        if row["identity_id"] in IDENTITY_REFERENCE_COLUMNS:
            actual[(row["identity_id"], row["source"], row["year"], row["period_segment"])] += row["population_rows"]
    result: list[dict[str, Any]] = []
    for identity_id, (_, identity_source, numeric_source, column) in IDENTITY_REFERENCE_COLUMNS.items():
        groups = {
            (year, segment)
            for source, observed_column, year, segment in expected
            if source == numeric_source and observed_column == column
        } | {
            (year, segment)
            for observed_identity, source, year, segment in actual
            if observed_identity == identity_id and source == identity_source
        }
        for year, segment in groups:
            expected_rows = expected[(numeric_source, column, year, segment)]
            actual_rows = actual[(identity_id, identity_source, year, segment)]
            if expected_rows != actual_rows:
                result.append(
                    {
                        "identity_id": identity_id, "source": identity_source,
                        "year": year, "period_segment": segment,
                        "expected_population_rows": expected_rows,
                        "actual_population_rows": actual_rows,
                    }
                )
    return sorted(
        result,
        key=lambda row: (
            row["identity_id"].encode("utf-8"), row["source"].encode("utf-8"),
            row["year"], row["period_segment"].encode("utf-8"),
        ),
    )


def _measurement_projection(
    table: str,
    rows: Sequence[Mapping[str, Any]],
    period_segment: str | None = None,
) -> dict[str, Any]:
    excluded = MEASUREMENT_EXCLUDED_FIELDS[table]
    projected = [
        {key: value for key, value in row.items() if key not in excluded}
        for row in rows
        if period_segment is None or row["period_segment"] == period_segment
    ]
    return {"row_count": len(projected), "sha256": sha256_bytes(canonical_json_bytes(projected))}


def _non_adr_segment_counts(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        if (
            row["formula_id"] != "monthly_interval_cap_price_shares"
            or row["periodicity"] != "monthly"
            or row["share_type"] != "NS"
            or row["shr_adr_flag"] != "N"
        ):
            continue
        segment = row["period_segment"]
        cell = result.setdefault(segment, {field: 0 for field in NON_ADR_COUNT_FIELDS})
        for field in NON_ADR_COUNT_FIELDS:
            cell[field] += row[field]
    return {segment: result[segment] for segment in sorted(result)}


def _recomputed_gate_values(
    settings: VerificationSettings,
    primary: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[dict[str, tuple[Any, Any]], dict[str, Any]]:
    tables = primary["tables"]
    config = _config_from_authenticated_settings(context["settings_auth"])
    projections = context["empirical"]["measurement_projections"]
    values: dict[str, tuple[Any, Any]] = {}
    semantic_differences = _inventory_semantics(tables)
    for table in EMPIRICAL_TABLES:
        # Replay the compiler gate exactly. Independent row-semantic defects are
        # reported separately by INVENTORY_SEMANTICS__* and must not redefine
        # the primary gate's published status-count contract.
        published_failed_rows = sum(
            row["status"] == "FAIL" for row in tables[table]
        )
        values[f"{table}_hard_failures"] = (0, published_failed_rows)
    pooled = _identity_zero_comparable_segments(tables["identity_inventory"])
    populations = _identity_population_differences(
        tables["numeric_inventory"], tables["identity_inventory"]
    )
    values["identity_pooled_active_segment_zero_comparable_groups"] = ([], pooled)
    values["identity_active_segment_population_conservation"] = ([], populations)
    for table in EMPIRICAL_TABLES:
        expected = projections[table]
        values[f"empirical_{table}_measurement_projection_full"] = (
            {"row_count": expected["row_count"], "sha256": expected["sha256"]},
            _measurement_projection(table, tables[table]),
        )
        for segment in SEGMENTS:
            values[f"empirical_{table}_measurement_projection_{segment}"] = (
                expected["segments"][segment],
                _measurement_projection(table, tables[table], segment),
            )
    for source in SOURCE_NAMES:
        configured = config["sources"][source]
        receipt = context["raw_by_source"][source]
        for metric, configured_field in (
            ("full_rows", "full_file_rows"),
            ("retained_rows", "verified_retained_rows"),
            ("canonical_rows", "verified_canonical_rows"),
        ):
            if configured_field in configured:
                values[f"source_{source}_{metric}"] = (
                    configured[configured_field], receipt[metric]
                )
    actual_counts = _non_adr_segment_counts(tables["capitalization_diagnostics"])
    expected_counts = context["empirical"]["segment_counts"]
    values["non_adr_capitalization_segment_counts"] = (expected_counts, actual_counts)
    totals = {
        field: sum(segment[field] for segment in actual_counts.values())
        for field in NON_ADR_COUNT_FIELDS
    }
    expected_theory = {
        "formula_pass_rate": "1", "formula_fail_rows": 0,
        "multiple_share_match_rows": 0,
        "pass_rule": "quantization_interval_overlap",
    }
    actual_theory = {
        "formula_pass_rate": _rate(totals["formula_pass_rows"], totals["comparable_rows"]),
        "formula_fail_rows": totals["formula_fail_rows"],
        "multiple_share_match_rows": totals["multiple_share_match_rows"],
        "pass_rule": "quantization_interval_overlap",
    }
    values["non_adr_capitalization_theory_threshold"] = (
        expected_theory, actual_theory
    )
    values["production_profile_status"] = (
        LOCKED_PROFILE_STATUS, config["status"]
    )
    segment_theory = {
        segment: (
            segment in actual_counts
            and actual_counts[segment]["comparable_rows"] > 0
            and actual_counts[segment]["excluded_rows"] == 0
            and actual_counts[segment]["zero_share_match_rows"] == 0
            and actual_counts[segment]["multiple_share_match_rows"] == 0
            and actual_counts[segment]["formula_fail_rows"] == 0
            and actual_counts[segment]["formula_pass_rows"]
            == actual_counts[segment]["comparable_rows"]
            and actual_counts[segment]["one_share_match_rows"]
            == actual_counts[segment]["comparable_rows"]
            and actual_counts[segment]["population_rows"]
            == actual_counts[segment]["comparable_rows"]
            and _rate(
                actual_counts[segment]["formula_pass_rows"],
                actual_counts[segment]["comparable_rows"],
            ) == "1"
        )
        for segment in SEGMENTS
    }
    extras = {
        "inventory_semantic_differences": semantic_differences,
        "identity_population_differences": populations,
        "pooled_zero_comparable": pooled,
        "segment_theory": segment_theory,
    }
    if set(values) != REQUIRED_PRODUCTION_CHECK_IDS:
        raise NumericUnitVerificationError(
            "Independent gate reconstruction does not cover the exact 29-gate set"
        )
    return values, extras


def _expected_documents(
    settings: VerificationSettings,
    primary: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    config = _config_from_authenticated_settings(primary["settings_auth"])
    binding = primary["binding"]
    build_id = primary["build_id"]
    summary = primary["json"]["summary.json"]
    return {
        "_SUCCESS.json": {
            "aggregate_only": True,
            "build_id": build_id,
            "closed_quality_file_set": list(config["outputs"]["closed_quality_file_set"]),
            "publication_performed": False,
            "schema_version": SCHEMA_VERSION,
            "status": PRIMARY_STATUS,
        },
        "profile_contract.json": {
            "DlyRetx_is_availability_requirement": False,
            "capitalization_formula_ids": list(
                config["outputs"]["vocabularies"]["capitalization_formula_id"]
            ),
            "capitalization_quantiles": {
                "estimator": "full_population_type7_linear_interpolation_over_disk_backed_Decimal_ratios_with_at_least_50_significant_digits",
                "hard_threshold_use": False,
                "sampled": False,
            },
            "closed_quality_file_set": list(config["outputs"]["closed_quality_file_set"]),
            "daily_price_flag_is_eligibility_filter": False,
            "empirical_lock_sha256": binding["empirical_lock_sha256"],
            "event_reconciliation_in_scope": False,
            "joins": {
                "MthCalDt_substitution_permitted": False,
                "daily_shares": "PERMNO plus inclusive DlyCalDt share interval",
                "monthly_daily": "monthly (PERMNO,MthPrcDt) equals daily (PERMNO,DlyCalDt)",
                "monthly_shares": "PERMNO plus inclusive nonblank MthPrcDt share interval",
            },
            "policy_sha256": binding["policy_sha256"],
            "profile_config_sha256": binding["profile_config_sha256"],
            "profile_name": PROFILE_NAME,
            "profile_status": LOCKED_PROFILE_STATUS,
            "profile_version": config["profile_version"],
            "raw_security_keys_may_be_published": False,
            "schema_version": SCHEMA_VERSION,
            "schemas": config["outputs"]["schemas"],
            "scope": {
                "december_2014_is_quality_only": True,
                "quality_end_date": config["scope"]["quality_end_date"],
                "start_date": config["scope"]["start_date"],
            },
        },
        "downstream_gate_status.json": {
            "backtests_authorized": False,
            "beta_estimation_authorized": False,
            "blockers": list(config["downstream"]["blockers"]),
            "build_id": build_id,
            "labels_authorized": False,
            "profile_status": LOCKED_PROFILE_STATUS,
            "return_history_authorized": False,
            "schema_version": SCHEMA_VERSION,
            "typed_market_backbone_authorized": False,
            "universe_unit_inputs_authorized": False,
        },
        "summary.json": summary,
    }


def _summary_expected(
    settings: VerificationSettings,
    primary: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    tables = primary["tables"]
    return {
        "aggregate_row_counts": {
            table: len(rows) for table, rows in sorted(tables.items())
        },
        "build_id": primary["build_id"],
        "check_count": 29,
        "diagnostic_count": 0,
        "fail_count": 0,
        "hard_failure_count": 0,
        "mode": "production",
        "pass_count": 29,
        "production_authorized": True,
        "profile_status": LOCKED_PROFILE_STATUS,
        "schema_version": SCHEMA_VERSION,
        "source_counts": {
            source: {
                "canonical_rows": context["raw_by_source"][source]["canonical_rows"],
                "full_rows": context["raw_by_source"][source]["full_rows"],
                "retained_rows": context["raw_by_source"][source]["retained_rows"],
            }
            for source in SOURCE_NAMES
        },
        "status": PRIMARY_STATUS,
    }


def _semantic_build_id(primary: Mapping[str, Any]) -> str:
    documents = primary["json"]
    build_seed = {
        "binding": primary["binding"],
        "downstream_gate_status_without_build_id": {
            key: value
            for key, value in documents["downstream_gate_status.json"].items()
            if key != "build_id"
        },
        "input_receipt": documents["input_receipt.json"],
        "mode": "production",
        "profile_contract": documents["profile_contract.json"],
        "success_without_build_id": {
            key: value
            for key, value in documents["_SUCCESS.json"].items()
            if key != "build_id"
        },
        "summary_without_build_id": {
            key: value
            for key, value in documents["summary.json"].items()
            if key != "build_id"
        },
        "tables": {
            table: [dict(row) for row in primary["tables"][table]]
            for table in sorted(primary["tables"])
        },
    }
    return "crsp-numeric-" + sha256_bytes(canonical_json_bytes(build_seed))[:24]


def add_semantic_checks(
    settings: VerificationSettings,
    primary: Mapping[str, Any],
    context: Mapping[str, Any],
    checks: _Checks,
) -> None:
    values, extras = _recomputed_gate_values(settings, primary, context)
    gate_rows = primary["tables"]["checks"]
    by_id = {row["check_id"]: row for row in gate_rows}
    exact_set = len(by_id) == len(gate_rows) == 29 and set(by_id) == REQUIRED_PRODUCTION_CHECK_IDS
    checks.add(
        "PRIMARY_GATE_SET",
        "primary_gates",
        expected=sorted(REQUIRED_PRODUCTION_CHECK_IDS),
        actual=sorted(by_id),
        passed=exact_set,
        evidence="checks.csv",
        message="The primary gate table contains exactly the 29 locked production gates.",
    )
    for check_id in sorted(REQUIRED_PRODUCTION_CHECK_IDS):
        expected, actual = values[check_id]
        row = by_id.get(check_id)
        if row is None:
            checks.add(
                f"PRIMARY_GATE__{check_id}",
                "primary_gates",
                expected={
                    "expected": expected,
                    "actual": actual,
                    "severity": "hard",
                    "status": "PASS",
                },
                actual=None,
                passed=False,
                evidence=f"checks.csv::{check_id}",
                message="The mandatory independently reconstructed gate is absent.",
            )
            continue
        row_expected = _parse_json_cell(row["expected"], f"gate {check_id} expected")
        row_actual = _parse_json_cell(row["actual"], f"gate {check_id} actual")
        passed = (
            row["severity"] == "hard"
            and row["status"] == "PASS"
            and row_expected == expected
            and row_actual == actual
            and expected == actual
        )
        checks.add(
            f"PRIMARY_GATE__{check_id}",
            "primary_gates",
            expected={"expected": expected, "actual": actual, "severity": "hard", "status": "PASS"},
            actual={"expected": row_expected, "actual": row_actual, "severity": row["severity"], "status": row["status"]},
            passed=passed,
            evidence=f"checks.csv::{check_id}",
            message="The published gate agrees with the independent aggregate recomputation.",
        )

    for table in EMPIRICAL_TABLES:
        differences = extras["inventory_semantic_differences"][table]
        checks.add(
            f"INVENTORY_SEMANTICS__{table}",
            "inventory_conservation",
            expected=[],
            actual=differences,
            passed=not differences,
            evidence=CSV_FILENAMES[table],
            message="Balances, exclusion identities, derived rates, severities, and statuses reconcile.",
        )
    checks.add(
        "NS_N_SEGMENT_THEORY_INDEPENDENT",
        "capitalization_scale",
        expected={segment: True for segment in SEGMENTS},
        actual=extras["segment_theory"],
        passed=all(extras["segment_theory"].values()),
        evidence="capitalization_diagnostics.csv; empirical lock",
        message="Each development/embargo NS/N segment independently satisfies the exact interval-share theory threshold.",
    )
    summary_expected = _summary_expected(settings, primary, context)
    summary_actual = primary["json"]["summary.json"]
    checks.add(
        "SUMMARY_CONSERVATION",
        "primary_documents",
        expected=summary_expected,
        actual=summary_actual,
        passed=summary_actual == summary_expected,
        evidence="summary.json; parsed CSV row counts; input_receipt.json",
        message="The production summary is exact and conserves all gate/table/source counts.",
    )
    expected_documents = _expected_documents(settings, primary)
    document_actual = {
        name: primary["json"][name]
        for name in ("_SUCCESS.json", "profile_contract.json", "downstream_gate_status.json")
    }
    document_expected = {
        name: expected_documents[name]
        for name in ("_SUCCESS.json", "profile_contract.json", "downstream_gate_status.json")
    }
    checks.add(
        "PRIMARY_DOCUMENT_CONTRACTS",
        "primary_documents",
        expected=document_expected,
        actual=document_actual,
        passed=document_actual == document_expected,
        evidence="_SUCCESS.json; profile_contract.json; downstream_gate_status.json",
        message="Success, profile, and downstream documents preserve the locked authorization boundary.",
    )
    reconstructed = _semantic_build_id(primary)
    checks.add(
        "SEMANTIC_BUILD_ID",
        "semantic_identity",
        expected=reconstructed,
        actual=primary["build_id"],
        passed=reconstructed == primary["build_id"],
        evidence="binding + typed CSV rows + four pre-ID JSON documents",
        message="The build ID is independently reconstructed without compiler logic imports.",
    )


def verifier_implementation_paths(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    root = _lexical_absolute(project_root)
    return {label: _lexical_absolute(root / label) for label in VERIFIER_IMPLEMENTATION_LABELS}


def implementation_bundle_sha256(
    implementation_paths: Mapping[str, Path], *, project_root: Path
) -> str:
    root = _lexical_absolute(project_root)
    if set(implementation_paths) != set(VERIFIER_IMPLEMENTATION_LABELS):
        raise NumericUnitVerificationError("Verifier implementation set differs")
    digest = hashlib.sha256()
    digest.update(IMPLEMENTATION_BUNDLE_DOMAIN)
    for label in sorted(implementation_paths):
        path = _lexical_absolute(implementation_paths[label])
        try:
            derived = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise NumericUnitVerificationError("Verifier implementation escapes project") from exc
        if derived != label:
            raise NumericUnitVerificationError("Verifier implementation label differs")
        payload = _read_stable(
            path,
            f"verifier implementation {label}",
            trusted_root=root,
        )
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _canonical_cell(value: Any) -> str:
    return canonical_json_text(value)


def render_differences_csv(report: Mapping[str, Any]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(DIFFERENCE_FIELDS), lineterminator="\n")
    writer.writeheader()
    for check in report.get("checks", []):
        if check["status"] == "PASS":
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


def derive_verification_id(report: Mapping[str, Any], differences_payload: bytes) -> str:
    identity = dict(report)
    identity["verification_id"] = ""
    digest = hashlib.sha256()
    digest.update(VERIFICATION_ID_DOMAIN)
    digest.update(canonical_json_bytes(identity))
    digest.update(differences_payload)
    return f"verify-{digest.hexdigest()[:16]}"


def bind_verification_id(report: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    frozen = json.loads(canonical_json_bytes(dict(report)).decode("utf-8"))
    provisional_differences = render_differences_csv(frozen)
    frozen["verification_id"] = derive_verification_id(frozen, provisional_differences)
    differences = render_differences_csv(frozen)
    if differences != provisional_differences:
        raise NumericUnitVerificationError("Verification ID changed differences payload")
    return frozen, differences


def build_independent_report(
    primary: Mapping[str, Any], checks: _Checks, implementation_sha256: str
) -> dict[str, Any]:
    rows = checks.sorted()
    failures = sum(row["status"] == "FAIL" for row in rows)
    report = {
        "build_id": primary["build_id"],
        "checks": rows,
        "contract_id": primary["contract_id"],
        "independence": {
            "aggregate_only": True,
            "empirical_projections_recomputed": 12,
            "full_raw_hashes_recomputed": True,
            "primary_compiler_imported": False,
            "primary_publisher_imported": False,
            "raw_files_hashed": 3,
            "raw_payload_rows_decoded": False,
            "semantic_build_id_recomputed": True,
        },
        "primary_manifest_sha256": primary["manifest_sha256"],
        "profile_name": PROFILE_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": PASS_STATUS if failures == 0 else FAIL_STATUS,
        "summary": {
            "check_count": len(rows),
            "difference_count": failures,
            "fail_count": failures,
            "pass_count": len(rows) - failures,
            "primary_gate_count": 29,
            "projection_count": 12,
            "source_count": 3,
        },
        "verification_id": "",
        "verifier_implementation_sha256": implementation_sha256,
    }
    return bind_verification_id(report)[0]


def _report_from_authenticated_context(
    settings: VerificationSettings,
    primary: Mapping[str, Any],
    context: Mapping[str, Any],
    implementation_sha256: str,
) -> dict[str, Any]:
    checks = _Checks()
    checks.add(
        "AUTHENTICATED_PRIMARY_BUNDLE",
        "authentication",
        expected=True,
        actual=True,
        passed=True,
        evidence="primary pointer, closed artifact/quality sets, manifest and output inventory",
        message="The complete published primary bundle authenticates.",
    )
    checks.add(
        "AUTHENTICATED_LOCKED_CONTEXT",
        "authentication",
        expected=True,
        actual=True,
        passed=True,
        evidence="profile, policy, empirical lock, eight upstreams, full calendar chain, and three full raw hashes",
        message="Every portable snapshot, calendar-chain artifact, and upstream/raw binding authenticates.",
    )
    add_semantic_checks(settings, primary, context, checks)
    return build_independent_report(primary, checks, implementation_sha256)


def independently_verify_numeric_unit_profile(
    settings: VerificationSettings | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    implementations = verifier_implementation_paths(settings.project_root)
    implementation_before = implementation_bundle_sha256(
        implementations, project_root=settings.project_root
    )
    primary = authenticate_primary_bundle(settings)
    context = authenticate_locked_context(settings, primary)
    implementation_after = implementation_bundle_sha256(
        implementations, project_root=settings.project_root
    )
    if implementation_after != implementation_before:
        raise NumericUnitVerificationError("Verifier implementation changed during verification")
    report = _report_from_authenticated_context(
        settings, primary, context, implementation_before
    )
    differences = render_differences_csv(report)
    if derive_verification_id(report, differences) != report["verification_id"]:
        raise NumericUnitVerificationError("Verification report identity differs")
    return {
        "context": context,
        "differences_payload": differences,
        "final_snapshot": context["final_snapshot"],
        "implementation_paths": implementations,
        "primary": primary,
        "raw_operational_snapshot": context["raw_operational"],
        "report": report,
    }


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze_json(child) for child in value)
    return value


def _validate_report(
    report: Mapping[str, Any], differences_payload: bytes
) -> _ValidatedReport:
    # Reconstruct first so every validation and later publication decision is
    # detached from caller-owned dictionaries/lists.
    report_payload = canonical_json_bytes(report)
    report = _parse_canonical_json_payload(
        report_payload, "verification report"
    )
    differences_payload = bytes(differences_payload)
    if set(report) != REPORT_KEYS:
        raise NumericUnitVerificationError("Verification report schema differs")
    if report["schema_version"] != SCHEMA_VERSION or report["profile_name"] != PROFILE_NAME:
        raise NumericUnitVerificationError("Verification report identity differs")
    if BUILD_ID_RE.fullmatch(str(report["build_id"])) is None:
        raise NumericUnitVerificationError("Verification build ID differs")
    _safe_component(report["contract_id"], "verification contract_id")
    if SHA256_RE.fullmatch(str(report["primary_manifest_sha256"])) is None or SHA256_RE.fullmatch(str(report["verifier_implementation_sha256"])) is None:
        raise NumericUnitVerificationError("Verification hash binding differs")
    if report["status"] not in {PASS_STATUS, FAIL_STATUS}:
        raise NumericUnitVerificationError("Verification status differs")
    independence = report["independence"]
    if not isinstance(independence, Mapping) or set(independence) != INDEPENDENCE_KEYS or dict(independence) != {
        "aggregate_only": True,
        "empirical_projections_recomputed": 12,
        "full_raw_hashes_recomputed": True,
        "primary_compiler_imported": False,
        "primary_publisher_imported": False,
        "raw_files_hashed": 3,
        "raw_payload_rows_decoded": False,
        "semantic_build_id_recomputed": True,
    }:
        raise NumericUnitVerificationError("Verification independence contract differs")
    rows = report["checks"]
    if not isinstance(rows, list) or not rows:
        raise NumericUnitVerificationError("Verification checks are absent")
    previous = ""
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != CHECK_KEYS:
            raise NumericUnitVerificationError("Verification check schema differs")
        if not isinstance(row["check_id"], str) or not row["check_id"] or row["check_id"] <= previous:
            raise NumericUnitVerificationError("Verification checks are not uniquely sorted")
        previous = row["check_id"]
        if row["status"] not in {"PASS", "FAIL"}:
            raise NumericUnitVerificationError("Verification check status differs")
        if any(not isinstance(row[key], str) for key in ("category", "evidence", "message")):
            raise NumericUnitVerificationError("Verification check text differs")
    observed_check_ids = {row["check_id"] for row in rows}
    if (
        len(rows) != len(REQUIRED_VERIFIER_CHECK_IDS)
        or observed_check_ids != REQUIRED_VERIFIER_CHECK_IDS
    ):
        raise NumericUnitVerificationError(
            "Verification report check-ID set differs from the exact independent contract"
        )
    summary = report["summary"]
    if not isinstance(summary, Mapping) or set(summary) != SUMMARY_KEYS or any(
        type(value) is not int or value < 0 for value in summary.values()
    ):
        raise NumericUnitVerificationError("Verification summary schema differs")
    failures = sum(row["status"] == "FAIL" for row in rows)
    if any(
        (
            summary["check_count"] != len(rows),
            summary["fail_count"] != failures,
            summary["difference_count"] != failures,
            summary["pass_count"] != len(rows) - failures,
            summary["primary_gate_count"] != 29,
            summary["projection_count"] != 12,
            summary["source_count"] != 3,
            (report["status"] == PASS_STATUS) != (failures == 0),
        )
    ):
        raise NumericUnitVerificationError("Verification summary does not reconcile")
    if render_differences_csv(report) != differences_payload:
        raise NumericUnitVerificationError("Differences CSV differs from failed checks")
    if VERIFICATION_ID_RE.fullmatch(str(report["verification_id"])) is None or derive_verification_id(report, differences_payload) != report["verification_id"]:
        raise NumericUnitVerificationError("Verification ID differs")
    return _ValidatedReport(
        payload=report_payload,
        value=_deep_freeze_json(report),
    )


def _assert_snapshot(
    snapshot: Mapping[Path, str], *, trusted_roots: Sequence[Path]
) -> None:
    if not snapshot:
        raise NumericUnitVerificationError("Final authentication snapshot is empty")
    for raw_path, expected in snapshot.items():
        path = _lexical_absolute(raw_path)
        if SHA256_RE.fullmatch(str(expected)) is None:
            raise NumericUnitVerificationError("Final snapshot hash differs")
        trusted_root = _trusted_root_for_path(
            path, trusted_roots, f"final snapshot {path}"
        )
        if sha256_bytes(
            _read_stable(
                path,
                f"final snapshot {path}",
                trusted_root=trusted_root,
            )
        ) != expected:
            raise NumericUnitVerificationError(f"Final context changed: {path}")


def _assert_closed_directory_state(state: _ClosedDirectoryState) -> None:
    descriptor, opened = _open_directory_fd_secure(
        state.path,
        state.label,
        create=False,
        trusted_root=state.trusted_root,
    )
    try:
        if int(opened.st_dev) != state.device or int(opened.st_ino) != state.inode:
            raise NumericUnitVerificationError(
                f"{state.label} directory identity changed"
            )
        payloads = _read_closed_directory_fd(
            descriptor, state.label, state.names
        )
        observed_hashes = tuple(
            (name, sha256_bytes(payloads[name])) for name in sorted(payloads)
        )
        if observed_hashes != state.hashes:
            raise NumericUnitVerificationError(
                f"{state.label} closed directory bytes changed"
            )
        _assert_directory_path_identity(
            state.path,
            opened,
            state.label,
            trusted_root=state.trusted_root,
        )
    finally:
        os.close(descriptor)


def _assert_closed_directories(
    states: Sequence[_ClosedDirectoryState],
) -> None:
    if len(states) != 2 or {state.label for state in states} != {
        "numeric-unit primary artifact",
        "numeric-unit primary quality",
    }:
        raise NumericUnitVerificationError(
            "Authenticated primary closed-directory state differs"
        )
    for state in states:
        _assert_closed_directory_state(state)


def _normalized_snapshot(snapshot: Mapping[Path, str]) -> dict[Path, str]:
    normalized: dict[Path, str] = {}
    for raw_path, digest in snapshot.items():
        path = _lexical_absolute(raw_path)
        if path in normalized or SHA256_RE.fullmatch(str(digest)) is None:
            raise NumericUnitVerificationError("Final snapshot schema differs")
        normalized[path] = str(digest)
    return normalized


def _normalized_raw_operational(
    snapshot: Mapping[Path, Mapping[str, int]],
) -> dict[Path, dict[str, int]]:
    normalized: dict[Path, dict[str, int]] = {}
    for raw_path, values in snapshot.items():
        path = _lexical_absolute(raw_path)
        if path in normalized or not isinstance(values, Mapping):
            raise NumericUnitVerificationError("Raw operational snapshot schema differs")
        normalized[path] = dict(values)
    return normalized


def _assert_raw_operational(
    snapshot: Mapping[Path, Mapping[str, int]], *, trusted_root: Path
) -> None:
    if len(snapshot) != 3:
        raise NumericUnitVerificationError("Raw operational snapshot must contain three sources")
    required = {"device", "inode", "mode", "size_bytes", "mtime_ns", "ctime_ns"}
    for raw_path, expected in snapshot.items():
        path = _lexical_absolute(raw_path)
        anchor = _stable_descriptor(
            path,
            "raw operational snapshot",
            trusted_root=trusted_root,
        )
        try:
            _assert_descriptor_unchanged(anchor)
            current = os.fstat(anchor.descriptor)
            if set(expected) != required or _operational_stat(current) != dict(expected):
                raise NumericUnitVerificationError(f"Raw source changed: {path.name}")
        finally:
            os.close(anchor.descriptor)
            os.close(anchor.parent_descriptor)


@dataclasses.dataclass
class _DirectoryAnchor:
    path: Path
    trusted_root: Path
    label: str
    descriptor: int
    opened: os.stat_result
    closed: bool = False


@dataclasses.dataclass
class _StagedVerification:
    parent: _DirectoryAnchor
    name: str
    descriptor: int
    opened: os.stat_result
    published: bool = False
    closed: bool = False


def _open_directory_anchor(
    path: Path, label: str, *, create: bool, trusted_root: Path
) -> _DirectoryAnchor:
    path = _lexical_absolute(path)
    trusted_root = _lexical_absolute(trusted_root)
    descriptor, opened = _open_directory_fd_secure(
        path, label, create=create, trusted_root=trusted_root
    )
    return _DirectoryAnchor(
        path=path,
        trusted_root=trusted_root,
        label=label,
        descriptor=descriptor,
        opened=opened,
    )


def _assert_directory_anchor(anchor: _DirectoryAnchor) -> None:
    if anchor.closed:
        raise NumericUnitVerificationError(f"{anchor.label} anchor is closed")
    try:
        opened = os.fstat(anchor.descriptor)
    except OSError as exc:
        raise NumericUnitVerificationError(
            f"Cannot reauthenticate {anchor.label} directory"
        ) from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not _same_inode(anchor.opened, opened)
    ):
        raise NumericUnitVerificationError(
            f"{anchor.label} parent path identity changed"
        )
    try:
        _assert_directory_path_identity(
            anchor.path,
            opened,
            anchor.label,
            trusted_root=anchor.trusted_root,
        )
    except NumericUnitVerificationError as exc:
        raise NumericUnitVerificationError(
            f"{anchor.label} parent path identity changed"
        ) from exc


def _close_directory_anchor(anchor: _DirectoryAnchor) -> None:
    if not anchor.closed:
        os.close(anchor.descriptor)
        anchor.closed = True


def _read_regular_at(directory_fd: int, name: str, label: str) -> bytes:
    name = _safe_component(name, label)
    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise NumericUnitVerificationError(
                f"Symlink is prohibited for anchored {label}"
            )
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
    except NumericUnitVerificationError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise NumericUnitVerificationError(f"Cannot open anchored {label}") from exc
    try:
        if not stat.S_ISREG(opened.st_mode) or not _same_inode(before, opened):
            raise NumericUnitVerificationError(
                f"Anchored {label} is not the selected regular file"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        current = os.fstat(descriptor)
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(opened, field) != getattr(current, field) for field in stable_fields) or any(
            getattr(current, field) != getattr(after, field) for field in stable_fields
        ):
            raise NumericUnitVerificationError(f"Anchored {label} changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_closed_directory_fd(
    directory_fd: int,
    label: str,
    expected_names: Sequence[str] = VERIFICATION_FILES,
) -> dict[str, bytes]:
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as exc:
        raise NumericUnitVerificationError(f"Cannot list anchored {label}") from exc
    expected = tuple(expected_names)
    if set(names) != set(expected) or len(names) != len(expected):
        raise NumericUnitVerificationError(f"{label} closed file set differs")
    return {
        name: _read_regular_at(directory_fd, name, f"{label}/{name}")
        for name in names
    }


def _open_child_directory(
    parent: _DirectoryAnchor, name: str, label: str
) -> tuple[int, os.stat_result]:
    _assert_directory_anchor(parent)
    name = _safe_component(name, label)
    descriptor: int | None = None
    try:
        before = os.stat(
            name, dir_fd=parent.descriptor, follow_symlinks=False
        )
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent.descriptor,
        )
        opened = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise NumericUnitVerificationError(f"Cannot open anchored {label}") from exc
    if not stat.S_ISDIR(opened.st_mode) or not _same_inode(before, opened):
        os.close(descriptor)
        raise NumericUnitVerificationError(f"Anchored {label} identity differs")
    return descriptor, opened


def _read_closed_child(
    parent: _DirectoryAnchor,
    name: str,
    label: str,
    expected_names: Sequence[str] = VERIFICATION_FILES,
) -> dict[str, bytes]:
    descriptor, opened = _open_child_directory(parent, name, label)
    try:
        payloads = _read_closed_directory_fd(descriptor, label, expected_names)
        after = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        if not _same_inode(opened, after):
            raise NumericUnitVerificationError(f"Anchored {label} changed")
        _assert_directory_anchor(parent)
        return payloads
    finally:
        os.close(descriptor)


def _read_closed_verification(
    path: Path, *, trusted_root: Path
) -> dict[str, bytes]:
    path = _lexical_absolute(path)
    parent = _open_directory_anchor(
        path.parent,
        "verification bundle parent",
        create=False,
        trusted_root=trusted_root,
    )
    try:
        return _read_closed_child(parent, path.name, "verification bundle")
    finally:
        _close_directory_anchor(parent)


def _write_payload_at(
    directory_fd: int, name: str, payload: bytes, label: str
) -> os.stat_result:
    name = _safe_component(name, label)
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise NumericUnitVerificationError(f"Cannot create anchored {label}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise NumericUnitVerificationError(f"Anchored {label} is not regular")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise NumericUnitVerificationError(f"Short anchored write for {label}")
            remaining = remaining[written:]
        _fsync_file(descriptor, label)
        current = os.fstat(descriptor)
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not _same_inode(opened, current)
            or not _same_inode(current, after)
            or current.st_size != len(payload)
            or after.st_size != len(payload)
        ):
            raise NumericUnitVerificationError(f"Anchored {label} identity changed")
        return current
    finally:
        os.close(descriptor)


def _stage_verification(
    parent_path: Path,
    prefix: str,
    payloads: Mapping[str, bytes],
    *,
    trusted_root: Path,
) -> _StagedVerification:
    if set(payloads) != set(VERIFICATION_FILES):
        raise NumericUnitVerificationError("Unexpected verification payload set")
    parent = _open_directory_anchor(
        parent_path,
        "verification staging parent",
        create=True,
        trusted_root=trusted_root,
    )
    stage: _StagedVerification | None = None
    try:
        for _attempt in range(128):
            name = f".{_safe_component(prefix, 'staging prefix')}.staging-{secrets.token_hex(8)}"
            try:
                os.mkdir(name, 0o700, dir_fd=parent.descriptor)
                break
            except FileExistsError:
                continue
        else:
            raise NumericUnitVerificationError("Cannot allocate verification stage")
        _fsync_directory(parent.descriptor, "verification staging parent after mkdir")
        descriptor, opened = _open_child_directory(
            parent, name, "verification staging directory"
        )
        stage = _StagedVerification(
            parent=parent, name=name, descriptor=descriptor, opened=opened
        )
        for filename, payload in sorted(payloads.items()):
            _write_payload_at(
                descriptor, filename, payload, f"verification staging/{filename}"
            )
        if _read_closed_directory_fd(descriptor, "verification staging") != dict(payloads):
            raise NumericUnitVerificationError("Staged verification bytes differ")
        _fsync_directory(descriptor, "verification staging after payloads")
        relative = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        if not _same_inode(opened, relative):
            raise NumericUnitVerificationError("Verification staging identity changed")
        _assert_directory_anchor(parent)
        return stage
    except Exception:
        if stage is not None:
            _discard_stage(stage)
        else:
            _close_directory_anchor(parent)
        raise


def _stage_name_for_inode(stage: _StagedVerification) -> str | None:
    try:
        names = os.listdir(stage.parent.descriptor)
    except OSError:
        return None
    for name in names:
        try:
            observed = os.stat(
                name, dir_fd=stage.parent.descriptor, follow_symlinks=False
            )
        except OSError:
            continue
        if stat.S_ISDIR(observed.st_mode) and _same_inode(stage.opened, observed):
            return name
    return None


def _discard_stage(stage: _StagedVerification) -> None:
    if stage.closed:
        return
    try:
        if not stage.published:
            try:
                for name in os.listdir(stage.descriptor):
                    os.unlink(name, dir_fd=stage.descriptor)
                _fsync_directory(
                    stage.descriptor, "discarded verification staging"
                )
            except OSError:
                pass
            selected_name = _stage_name_for_inode(stage)
            if selected_name is not None:
                try:
                    os.rmdir(selected_name, dir_fd=stage.parent.descriptor)
                    _fsync_directory(
                        stage.parent.descriptor,
                        "verification staging parent after discard",
                    )
                except OSError:
                    pass
    finally:
        os.close(stage.descriptor)
        stage.closed = True
        _close_directory_anchor(stage.parent)


def _publish_stage(
    stage: _StagedVerification, final: Path, payloads: Mapping[str, bytes]
) -> None:
    final = _lexical_absolute(final)
    if final.parent != stage.parent.path:
        raise NumericUnitVerificationError("Verification final parent differs from stage anchor")
    final_name = _safe_component(final.name, "verification destination")
    _assert_directory_anchor(stage.parent)
    relative_stage = os.stat(
        stage.name, dir_fd=stage.parent.descriptor, follow_symlinks=False
    )
    if not _same_inode(stage.opened, relative_stage):
        raise NumericUnitVerificationError("Verification stage path identity changed")
    try:
        os.stat(final_name, dir_fd=stage.parent.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        final_exists = False
    else:
        final_exists = True
    if final_exists:
        if _read_closed_child(
            stage.parent, final_name, "existing verification bundle"
        ) != dict(payloads):
            raise NumericUnitVerificationError("Existing immutable verification differs")
        existing_fd, _existing_opened = _open_child_directory(
            stage.parent, final_name, "existing verification bundle"
        )
        try:
            _fsync_directory(existing_fd, "existing verification bundle")
        finally:
            os.close(existing_fd)
        _fsync_directory(
            stage.parent.descriptor,
            "verification parent with existing immutable bundle",
        )
        _discard_stage(stage)
        return
    try:
        os.rename(
            stage.name,
            final_name,
            src_dir_fd=stage.parent.descriptor,
            dst_dir_fd=stage.parent.descriptor,
        )
    except OSError:
        try:
            existing = _read_closed_child(
                stage.parent, final_name, "concurrent verification bundle"
            )
        except NumericUnitVerificationError:
            raise
        if existing != dict(payloads):
            raise NumericUnitVerificationError("Concurrent immutable verification differs")
        existing_fd, _existing_opened = _open_child_directory(
            stage.parent, final_name, "concurrent verification bundle"
        )
        try:
            _fsync_directory(existing_fd, "concurrent verification bundle")
        finally:
            os.close(existing_fd)
        _fsync_directory(
            stage.parent.descriptor,
            "verification parent with concurrent immutable bundle",
        )
        _discard_stage(stage)
        return
    published = os.stat(
        final_name, dir_fd=stage.parent.descriptor, follow_symlinks=False
    )
    if not _same_inode(stage.opened, published):
        raise NumericUnitVerificationError("Published verification identity differs")
    if _read_closed_directory_fd(stage.descriptor, "published verification") != dict(payloads):
        raise NumericUnitVerificationError("Published verification bytes differ")
    _assert_directory_anchor(stage.parent)
    _fsync_directory(stage.descriptor, "published verification bundle")
    _fsync_directory(
        stage.parent.descriptor, "verification parent after immutable rename"
    )
    _assert_directory_anchor(stage.parent)
    stage.published = True


def _write_atomic(
    path: Path,
    payload: bytes,
    pre_replace_gate: Any,
    post_replace_gate: Any | None = None,
    *,
    trusted_root: Path,
) -> None:
    path = _lexical_absolute(path)
    name = _safe_component(path.name, "verified pointer")
    parent = _open_directory_anchor(
        path.parent,
        "verified pointer parent",
        create=True,
        trusted_root=trusted_root,
    )
    temporary_name = f".{name}.{secrets.token_hex(8)}.tmp"
    temporary_stat: os.stat_result | None = None
    existing: os.stat_result | None = None
    backup_name: str | None = None
    backup_stat: os.stat_result | None = None
    pointer_replaced = False
    preserve_backup = False
    post_replace_gate = post_replace_gate or pre_replace_gate

    def rollback_replaced_pointer() -> None:
        nonlocal pointer_replaced, backup_name, preserve_backup
        if not pointer_replaced or temporary_stat is None:
            return
        try:
            if backup_name is None:
                try:
                    current = os.stat(
                        name, dir_fd=parent.descriptor, follow_symlinks=False
                    )
                except FileNotFoundError:
                    current = None
                if current is not None:
                    if stat.S_ISDIR(current.st_mode):
                        os.rmdir(name, dir_fd=parent.descriptor)
                    else:
                        os.unlink(name, dir_fd=parent.descriptor)
            else:
                try:
                    selected_backup = os.stat(
                        backup_name,
                        dir_fd=parent.descriptor,
                        follow_symlinks=False,
                    )
                except OSError:
                    selected_backup = None
                if (
                    backup_stat is None
                    or selected_backup is None
                    or not stat.S_ISREG(selected_backup.st_mode)
                    or not _same_inode(backup_stat, selected_backup)
                ):
                    try:
                        current = os.stat(
                            name,
                            dir_fd=parent.descriptor,
                            follow_symlinks=False,
                        )
                        if stat.S_ISDIR(current.st_mode):
                            os.rmdir(name, dir_fd=parent.descriptor)
                        else:
                            os.unlink(name, dir_fd=parent.descriptor)
                    except FileNotFoundError:
                        pass
                    try:
                        os.unlink(backup_name, dir_fd=parent.descriptor)
                    except OSError:
                        pass
                    backup_name = None
                    os.fsync(parent.descriptor)
                    pointer_replaced = False
                    return
                os.rename(
                    backup_name,
                    name,
                    src_dir_fd=parent.descriptor,
                    dst_dir_fd=parent.descriptor,
                )
                backup_name = None
                restored = os.stat(
                    name, dir_fd=parent.descriptor, follow_symlinks=False
                )
                if not _same_inode(backup_stat, restored):
                    os.unlink(name, dir_fd=parent.descriptor)
            os.fsync(parent.descriptor)
            pointer_replaced = False
        except OSError:
            # If restoring the authenticated backup itself fails, fail closed:
            # remove the just-published inode, durably preserve the backup link,
            # and let the caller receive the original commit error.
            preserve_backup = backup_name is not None
            try:
                current = os.stat(
                    name, dir_fd=parent.descriptor, follow_symlinks=False
                )
                if stat.S_ISDIR(current.st_mode):
                    os.rmdir(name, dir_fd=parent.descriptor)
                else:
                    os.unlink(name, dir_fd=parent.descriptor)
                os.fsync(parent.descriptor)
                pointer_replaced = False
            except OSError:
                pass

    try:
        temporary_stat = _write_payload_at(
            parent.descriptor, temporary_name, payload, "verified pointer temporary"
        )
        pre_replace_gate()
        _assert_directory_anchor(parent)
        try:
            existing = os.stat(
                name, dir_fd=parent.descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise NumericUnitVerificationError(
                "Existing verified pointer is not a regular file"
            )
        anchored_temporary = os.stat(
            temporary_name, dir_fd=parent.descriptor, follow_symlinks=False
        )
        if not _same_inode(temporary_stat, anchored_temporary):
            raise NumericUnitVerificationError("Verified pointer temporary changed")
        if existing is not None:
            backup_name = f".{name}.{secrets.token_hex(8)}.rollback"
            try:
                os.link(
                    name,
                    backup_name,
                    src_dir_fd=parent.descriptor,
                    dst_dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
                backup_stat = os.stat(
                    backup_name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise NumericUnitVerificationError(
                    "Cannot preserve existing verified pointer for rollback"
                ) from exc
            if not _same_inode(existing, backup_stat):
                raise NumericUnitVerificationError(
                    "Verified pointer rollback identity differs"
                )
            _fsync_directory(
                parent.descriptor,
                "verified pointer parent after rollback-link creation",
            )
        os.rename(
            temporary_name,
            name,
            src_dir_fd=parent.descriptor,
            dst_dir_fd=parent.descriptor,
        )
        pointer_replaced = True
        try:
            published = os.stat(
                name, dir_fd=parent.descriptor, follow_symlinks=False
            )
            if not _same_inode(temporary_stat, published):
                raise NumericUnitVerificationError(
                    "Verified pointer identity differs"
                )
            _assert_directory_anchor(parent)
            # Reauthenticate once more after the rename. This closes the only
            # externally hookable window between the pre-rename gate and the
            # pointer commit; any mismatch atomically restores the old pointer
            # (or removes a newly created one) before returning an error.
            post_replace_gate()
            _assert_directory_anchor(parent)
            _fsync_directory(
                parent.descriptor, "verified pointer parent after pointer-last rename"
            )
            _assert_directory_anchor(parent)
            post_replace_gate()
        except Exception:
            rollback_replaced_pointer()
            raise
        if backup_name is not None:
            try:
                os.unlink(backup_name, dir_fd=parent.descriptor)
            except OSError:
                # The new pointer and its authenticated context are already
                # durable. A non-authority rollback link can be cleaned later.
                preserve_backup = True
            else:
                backup_name = None
                try:
                    _fsync_directory(
                        parent.descriptor,
                        "verified pointer parent after rollback-link cleanup",
                    )
                except NumericUnitVerificationError:
                    pass
    finally:
        try:
            temporary = os.stat(
                temporary_name, dir_fd=parent.descriptor, follow_symlinks=False
            )
        except (FileNotFoundError, OSError):
            temporary = None
        if (
            temporary is not None
            and temporary_stat is not None
            and _same_inode(temporary_stat, temporary)
        ):
            try:
                os.unlink(temporary_name, dir_fd=parent.descriptor)
                _fsync_directory(
                    parent.descriptor,
                    "verified pointer parent after temporary cleanup",
                )
            except OSError:
                pass
        if (
            backup_name is not None
            and backup_stat is not None
            and not preserve_backup
        ):
            try:
                observed_backup = os.stat(
                    backup_name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
                if _same_inode(backup_stat, observed_backup):
                    os.unlink(backup_name, dir_fd=parent.descriptor)
                    os.fsync(parent.descriptor)
            except OSError:
                pass
        _close_directory_anchor(parent)


def _read_optional_regular(
    path: Path, label: str, *, trusted_root: Path
) -> bytes | None:
    path = _lexical_absolute(path)
    parent = _open_directory_anchor(
        path.parent,
        f"{label} parent",
        create=False,
        trusted_root=trusted_root,
    )
    try:
        name = _safe_component(path.name, label)
        try:
            observed = os.stat(
                name, dir_fd=parent.descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(observed.st_mode):
            raise NumericUnitVerificationError(
                f"Symlink is prohibited for {label}: {path}"
            )
        if not stat.S_ISREG(observed.st_mode):
            raise NumericUnitVerificationError(f"{label} is not a regular file")
        payload = _read_regular_at(parent.descriptor, name, label)
        _assert_directory_anchor(parent)
        return payload
    finally:
        _close_directory_anchor(parent)


def _durably_confirm_regular(
    path: Path, label: str, expected_payload: bytes, *, trusted_root: Path
) -> None:
    anchor = _stable_descriptor(path, label, trusted_root=trusted_root)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(anchor.descriptor, 1 << 20):
            chunks.append(chunk)
        _assert_descriptor_unchanged(anchor)
        if b"".join(chunks) != expected_payload:
            raise NumericUnitVerificationError(
                f"Existing {label} bytes changed before durability confirmation"
            )
        _fsync_file(anchor.descriptor, f"existing {label}")
        _fsync_directory(
            anchor.parent_descriptor,
            f"{label} parent after idempotent durability confirmation",
        )
        _assert_descriptor_unchanged(anchor)
    finally:
        os.close(anchor.descriptor)
        os.close(anchor.parent_descriptor)


def publish_independent_verification(
    report: Mapping[str, Any],
    differences_payload: bytes,
    *,
    settings: VerificationSettings,
    verifier_implementation_paths: Mapping[str, Path],
    final_snapshot: Mapping[Path, str],
    raw_operational_snapshot: Mapping[Path, Mapping[str, int]],
    primary_manifest_path: Path,
) -> dict[str, Path]:
    validated = _validate_report(report, differences_payload)
    report_payload = validated.payload
    report_view = validated.value
    differences_payload = bytes(differences_payload)
    implementation_paths = {
        str(label): _lexical_absolute(path)
        for label, path in verifier_implementation_paths.items()
    }
    implementation_sha = implementation_bundle_sha256(
        implementation_paths, project_root=settings.project_root
    )
    if implementation_sha != report_view["verifier_implementation_sha256"]:
        raise NumericUnitVerificationError("Verifier implementation differs from report")

    # A public publisher call must not be able to substitute a forged PASS or
    # a partial hand-selected snapshot. Reauthenticate the selected primary and
    # every locked context byte, reconstruct the entire report, and require the
    # caller's evidence/snapshots to equal that authoritative reconstruction.
    authenticated_primary = authenticate_primary_bundle(settings)
    authenticated_context = authenticate_locked_context(
        settings, authenticated_primary
    )
    expected_report = _report_from_authenticated_context(
        settings, authenticated_primary, authenticated_context, implementation_sha
    )
    expected_differences = render_differences_csv(expected_report)
    if (
        report_payload != canonical_json_bytes(expected_report)
        or differences_payload != expected_differences
    ):
        raise NumericUnitVerificationError(
            "Verification report is not the independently reconstructed selected primary"
        )
    authoritative_snapshot = _normalized_snapshot(
        authenticated_context["final_snapshot"]
    )
    supplied_snapshot = _normalized_snapshot(final_snapshot)
    if supplied_snapshot != authoritative_snapshot:
        raise NumericUnitVerificationError(
            "Final snapshot is not the exact authenticated context path set"
        )
    authoritative_raw = _normalized_raw_operational(
        authenticated_context["raw_operational"]
    )
    supplied_raw = _normalized_raw_operational(raw_operational_snapshot)
    if supplied_raw != authoritative_raw:
        raise NumericUnitVerificationError(
            "Raw operational snapshot is not the exact authenticated source set"
        )
    selected_manifest_path = _lexical_absolute(
        authenticated_primary["manifest_path"]
    )
    primary_manifest_path = _lexical_absolute(primary_manifest_path)
    if primary_manifest_path != selected_manifest_path:
        raise NumericUnitVerificationError(
            "Primary manifest path is not the selected primary manifest"
        )
    final_snapshot = authoritative_snapshot
    raw_operational_snapshot = authoritative_raw
    settings_auth = authenticated_context["settings_auth"]
    if settings_auth != authenticated_primary["settings_auth"]:
        raise NumericUnitVerificationError(
            "Primary and locked-context settings authentication differ"
        )
    closed_directories = tuple(authenticated_primary["closed_directories"])
    trusted_snapshot_roots = (
        settings_auth.project_root,
        settings_auth.artifact_root,
        settings_auth.database_root,
        settings_auth.raw_root,
        settings_auth.interim_root,
        settings_auth.quality_root,
    )
    try:
        settings_auth.artifact_root.relative_to(settings_auth.project_root)
    except ValueError as exc:
        raise NumericUnitVerificationError("Artifact root escapes project root") from exc
    # Freeze every authenticated output component and destination before any
    # staging/publication hook can mutate caller-owned settings.
    subdirectory = settings_auth.verification_subdirectory
    verification_id = _safe_component(
        report_view["verification_id"], "verification_id"
    )
    build_id = _safe_component(report_view["build_id"], "verification build_id")
    verification_directory = _lexical_absolute(
        settings_auth.artifact_root / subdirectory / build_id / verification_id
    )
    pointer_path = _lexical_absolute(
        settings_auth.artifact_root / settings_auth.verified_pointer_name
    )
    verification_relative = verification_directory.relative_to(
        settings_auth.project_root
    ).as_posix()
    payloads = {
        "differences.csv": differences_payload,
        "independent_check.json": report_payload,
    }
    if set(payloads) != set(settings_auth.closed_verification_file_set):
        raise NumericUnitVerificationError("Verification closed file set differs")
    verification_published = False

    def context_gate() -> None:
        _authenticate_settings(settings, settings_auth)
        _assert_snapshot(
            final_snapshot, trusted_roots=trusted_snapshot_roots
        )
        _assert_raw_operational(
            raw_operational_snapshot, trusted_root=settings_auth.raw_root
        )
        _assert_closed_directories(closed_directories)
        if implementation_bundle_sha256(
            implementation_paths, project_root=settings_auth.project_root
        ) != report_view["verifier_implementation_sha256"]:
            raise NumericUnitVerificationError("Verifier implementation changed")
        if sha256_bytes(
            _read_stable(
                primary_manifest_path,
                "primary manifest",
                trusted_root=settings_auth.artifact_root,
            )
        ) != report_view["primary_manifest_sha256"]:
            raise NumericUnitVerificationError("Primary manifest changed")
        if (
            verification_published
            and _read_closed_verification(
                verification_directory,
                trusted_root=settings_auth.artifact_root,
            )
            != payloads
        ):
            raise NumericUnitVerificationError("Verification bundle changed")

    context_gate()
    stage = _stage_verification(
        verification_directory.parent,
        verification_id,
        payloads,
        trusted_root=settings_auth.artifact_root,
    )
    try:
        _publish_stage(stage, verification_directory, payloads)
        verification_published = True
    finally:
        _discard_stage(stage)
    result = {"verification": verification_directory}
    context_gate()
    if report_view["status"] != PASS_STATUS:
        return result
    if report_view["summary"]["difference_count"] != 0:
        raise NumericUnitVerificationError("Passing report contains differences")
    pointer = {
        "build_id": build_id,
        "contract_id": report_view["contract_id"],
        "differences_sha256": sha256_bytes(differences_payload),
        "independent_check_sha256": sha256_bytes(report_payload),
        "primary_manifest_sha256": report_view["primary_manifest_sha256"],
        "schema_version": SCHEMA_VERSION,
        "status": PASS_STATUS,
        "verification_id": verification_id,
        "verification_path": verification_relative,
    }
    if set(pointer) != VERIFIED_POINTER_KEYS:
        raise NumericUnitVerificationError("Verified pointer schema differs")
    pointer_payload = canonical_json_bytes(pointer)

    def pointer_context_gate() -> None:
        context_gate()
        if _read_optional_regular(
            pointer_path,
            "verified pointer",
            trusted_root=settings_auth.artifact_root,
        ) != pointer_payload:
            raise NumericUnitVerificationError(
                "Verified pointer bytes changed during final commit gate"
            )

    context_gate()
    existing_pointer = _read_optional_regular(
        pointer_path,
        "verified pointer",
        trusted_root=settings_auth.artifact_root,
    )
    if existing_pointer != pointer_payload:
        _write_atomic(
            pointer_path,
            pointer_payload,
            context_gate,
            pointer_context_gate,
            trusted_root=settings_auth.artifact_root,
        )
    else:
        context_gate()
        _durably_confirm_regular(
            pointer_path,
            "verified pointer",
            pointer_payload,
            trusted_root=settings_auth.artifact_root,
        )
        pointer_context_gate()
    result["pointer"] = pointer_path
    return result


def verify_and_publish_numeric_unit_profile(
    settings: VerificationSettings | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    result = independently_verify_numeric_unit_profile(settings)
    published = publish_independent_verification(
        result["report"],
        result["differences_payload"],
        settings=settings,
        verifier_implementation_paths=result["implementation_paths"],
        final_snapshot=result["final_snapshot"],
        raw_operational_snapshot=result["raw_operational_snapshot"],
        primary_manifest_path=result["primary"]["manifest_path"],
    )
    return {**result, "published": published}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-config", type=Path, default=DEFAULT_PROFILE_CONFIG)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--phase-config", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--primary-pointer", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--quality-root", type=Path)
    parser.add_argument("--no-publish", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    settings = load_settings(
        arguments.profile_config,
        arguments.paths_config,
        arguments.phase_config,
        project_root=arguments.project_root,
        primary_pointer_path=arguments.primary_pointer,
        artifact_root=arguments.artifact_root,
        database_root=arguments.database_root,
        raw_root=arguments.raw_root,
        quality_root=arguments.quality_root,
    )
    result = (
        independently_verify_numeric_unit_profile(settings)
        if arguments.no_publish
        else verify_and_publish_numeric_unit_profile(settings)
    )
    report = result["report"]
    print(
        canonical_json_text(
            {
                "build_id": report["build_id"],
                "difference_count": report["summary"]["difference_count"],
                "status": report["status"],
                "verification_id": report["verification_id"],
            }
        )
    )
    return 0 if report["status"] == PASS_STATUS else 1


__all__ = [
    "NumericUnitVerificationError",
    "VerificationSettings",
    "authenticate_locked_context",
    "authenticate_primary_bundle",
    "bind_verification_id",
    "canonical_json_bytes",
    "canonical_json_text",
    "derive_verification_id",
    "implementation_bundle_sha256",
    "independently_verify_numeric_unit_profile",
    "load_settings",
    "main",
    "publish_independent_verification",
    "render_differences_csv",
    "verifier_implementation_paths",
    "verify_and_publish_numeric_unit_profile",
]
