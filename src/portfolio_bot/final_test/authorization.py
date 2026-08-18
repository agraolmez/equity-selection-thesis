"""Exact, Stage-12-bound user authorization for the Stage 13 unblind.

A verified Stage 12 freeze is a necessary technical precondition; it is not
itself permission to inspect final-period outcomes.  The immutable receipt
defined here binds a separate explicit user decision to the exact verified
Stage 12 build, manifest, verification, and pointer identities.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


STAGE13_APPROVAL_SCHEMA_VERSION = 1
STAGE13_UNBLIND_SCOPE = "stage13_final_2020_2025_outcome_unblind"
STAGE12_VERIFIED_POINTER_PATH = (
    "artifacts/phase_06_final_test/LATEST_STAGE_12_VERIFIED.json"
)
STAGE12_VERIFIED_STATUS = "STAGE12_POST_UNBLIND_REPLACEMENT_FREEZE_VERIFIED"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{2,127}$")
_UTC_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_id",
        "scope",
        "approved",
        "approved_by",
        "approved_at_utc",
        "source_message_id",
        "source_message_sha256",
        "stage12_build_id",
        "stage12_verification_id",
        "stage12_manifest_sha256",
        "stage12_verified_pointer_path",
        "stage12_verified_pointer_sha256",
        "stage12_binding_sha256",
        "payload_sha256",
    }
)


class Stage13AuthorizationError(RuntimeError):
    """Raised when final-period authorization is absent, altered, or mismatched."""


def _require_real_path_beneath(
    path: Path,
    *,
    project_root: Path,
    file_required: bool,
) -> None:
    root = project_root.absolute()
    selected = path.absolute()
    try:
        relative = selected.relative_to(root)
    except ValueError as exc:
        raise Stage13AuthorizationError("authority path escaped project root") from exc
    current = root
    for index, component in enumerate(relative.parts):
        current = current / component
        try:
            metadata = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise Stage13AuthorizationError(
                "required Stage 12 authority path is absent"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise Stage13AuthorizationError(
                "Stage 12 authority path contains a symlink"
            )
        final = index == len(relative.parts) - 1
        expected = stat.S_ISREG(metadata.st_mode) if final and file_required else (
            stat.S_ISDIR(metadata.st_mode)
        )
        if not expected:
            raise Stage13AuthorizationError(
                "Stage 12 authority path has an unsafe file type"
            )


def _read_stable_json(
    path: Path,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], str]:
    _require_real_path_beneath(
        path,
        project_root=project_root,
        file_required=True,
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Stage13AuthorizationError(
            "cannot open Stage 12 authority safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > 16 * 1024 * 1024
        ):
            raise Stage13AuthorizationError("unsafe Stage 12 authority file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise Stage13AuthorizationError(
            "Stage 12 authority changed while being read"
        )
    payload = b"".join(chunks)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage13AuthorizationError(
            "Stage 12 authority is not valid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise Stage13AuthorizationError("Stage 12 authority must be an object")
    return value, _sha256_bytes(payload)


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise Stage13AuthorizationError(
            f"{field} must be 64 lowercase hexadecimal characters"
        )
    return value


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise Stage13AuthorizationError(
            f"{field} is not a valid bounded identifier"
        )
    return value


def _require_nonempty_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
    ):
        raise Stage13AuthorizationError(
            f"{field} must be nonempty normalized text of at most 256 characters"
        )
    return value


def _require_utc_timestamp(value: object) -> str:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise Stage13AuthorizationError(
            "approved_at_utc must be canonical ISO-8601 UTC"
        )
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Stage13AuthorizationError(
            "approved_at_utc is not a valid timestamp"
        ) from exc
    if parsed.utcoffset() != dt.timedelta(0):
        raise Stage13AuthorizationError("approved_at_utc must be UTC")
    return value


def _require_pointer_path(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Stage13AuthorizationError(
            "stage12_verified_pointer_path must be normalized text"
        )
    if "\\" in value:
        raise Stage13AuthorizationError(
            "stage12_verified_pointer_path must use POSIX separators"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise Stage13AuthorizationError(
            "stage12_verified_pointer_path must be repository-relative"
        )
    if (
        not path.parts
        or path.parts[0] != "artifacts"
        or path.suffix != ".json"
    ):
        raise Stage13AuthorizationError(
            "stage12_verified_pointer_path must name an artifacts JSON pointer"
        )
    return value


@dataclass(frozen=True, slots=True)
class Stage12VerifiedBinding:
    """The complete verified Stage 12 identity required by Stage 13."""

    build_id: str
    verification_id: str
    manifest_sha256: str
    verified_pointer_path: str
    verified_pointer_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.build_id, field="stage12_build_id")
        _require_identifier(
            self.verification_id,
            field="stage12_verification_id",
        )
        _require_hash(
            self.manifest_sha256,
            field="stage12_manifest_sha256",
        )
        _require_pointer_path(self.verified_pointer_path)
        _require_hash(
            self.verified_pointer_sha256,
            field="stage12_verified_pointer_sha256",
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "stage12_build_id": self.build_id,
            "stage12_verification_id": self.verification_id,
            "stage12_manifest_sha256": self.manifest_sha256,
            "stage12_verified_pointer_path": self.verified_pointer_path,
            "stage12_verified_pointer_sha256": self.verified_pointer_sha256,
        }

    @property
    def binding_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_mapping()))


def load_authenticated_stage12_binding(
    project_root: Path,
    *,
    replay_primary_bytes: bool = True,
) -> Stage12VerifiedBinding:
    """Load the one repository authority that may precede Stage 13 approval.

    The verified pointer, its exact three-member verification bundle, the
    primary pointer, and the primary manifest are authenticated here.  The
    production path additionally performs the independent Stage 12 byte replay.
    ``replay_primary_bytes=False`` exists only for isolated synthetic tests;
    the Stage 13 broker never exposes or uses that mode.
    """

    if not isinstance(project_root, Path) or not project_root.is_absolute():
        raise Stage13AuthorizationError("project_root must be an absolute Path")
    root = project_root.absolute()
    pointer_path = root / STAGE12_VERIFIED_POINTER_PATH
    pointer, pointer_sha = _read_stable_json(
        pointer_path,
        project_root=root,
    )
    authorizations = pointer.get("authorizations")
    if (
        pointer.get("schema_version") != 1
        or pointer.get("stage") != 12
        or pointer.get("status") != STAGE12_VERIFIED_STATUS
        or pointer.get("difference_count") != 0
        or not isinstance(authorizations, Mapping)
        or authorizations.get("stage13_technical_gate_passed") is not True
        or authorizations.get("stage13_requires_separate_user_approval")
        is not True
        or authorizations.get("stage13_execution_authorized_now") is not False
        or authorizations.get("final_period_access_authorized_now") is not False
    ):
        raise Stage13AuthorizationError(
            "verified Stage 12 pointer does not preserve the Stage 13 boundary"
        )
    build_id = _require_identifier(
        pointer.get("primary_freeze_id"),
        field="primary_freeze_id",
    )
    verification_id = _require_identifier(
        pointer.get("verification_id"),
        field="verification_id",
    )
    manifest_sha = _require_hash(
        pointer.get("primary_manifest_sha256"),
        field="primary_manifest_sha256",
    )
    primary_pointer_relative = pointer.get("primary_pointer_path")
    if (
        primary_pointer_relative
        != "artifacts/phase_06_final_test/LATEST_STAGE_12_PRIMARY.json"
    ):
        raise Stage13AuthorizationError("Stage 12 primary pointer path differs")
    primary_pointer, primary_pointer_sha = _read_stable_json(
        root / str(primary_pointer_relative),
        project_root=root,
    )
    if (
        primary_pointer_sha != pointer.get("primary_pointer_sha256")
        or primary_pointer.get("freeze_id") != build_id
        or primary_pointer.get("manifest_sha256") != manifest_sha
        or primary_pointer.get("authorizations", {}).get(
            "stage13_execution_authorized_now"
        )
        is not False
        or primary_pointer.get("authorizations", {}).get(
            "final_period_access_authorized_now"
        )
        is not False
    ):
        raise Stage13AuthorizationError("Stage 12 primary pointer differs")
    verification_relative = pointer.get("verification_bundle_path")
    if not isinstance(verification_relative, str):
        raise Stage13AuthorizationError("verification bundle path is missing")
    verification_path = (root / verification_relative).absolute()
    _require_real_path_beneath(
        verification_path,
        project_root=root,
        file_required=False,
    )
    names = {item.name for item in verification_path.iterdir()}
    if names != {
        "_SUCCESS.json",
        "byte_differences.json",
        "verification_report.json",
    }:
        raise Stage13AuthorizationError(
            "Stage 12 verification closed-member set differs"
        )
    verification_report, report_sha = _read_stable_json(
        verification_path / "verification_report.json",
        project_root=root,
    )
    verification_success, success_sha = _read_stable_json(
        verification_path / "_SUCCESS.json",
        project_root=root,
    )
    differences, _ = _read_stable_json(
        verification_path / "byte_differences.json",
        project_root=root,
    )
    if (
        report_sha != pointer.get("verification_report_sha256")
        or success_sha != pointer.get("verification_success_sha256")
        or verification_report.get("verification_id") != verification_id
        or verification_report.get("primary_freeze_id") != build_id
        or verification_report.get("primary_manifest_sha256") != manifest_sha
        or verification_report.get("difference_count") != 0
        or verification_report.get("authorizations") != authorizations
        or verification_success.get("verification_id") != verification_id
        or verification_success.get("difference_count") != 0
        or differences.get("verification_id") != verification_id
        or differences.get("difference_count") != 0
        or differences.get("differences") != []
    ):
        raise Stage13AuthorizationError(
            "Stage 12 verification bundle does not replay exactly"
        )
    bundle_relative = primary_pointer.get("bundle_path")
    if not isinstance(bundle_relative, str):
        raise Stage13AuthorizationError("Stage 12 primary bundle path is missing")
    manifest, observed_manifest_sha = _read_stable_json(
        root / bundle_relative / "pre_final_manifest.json",
        project_root=root,
    )
    if (
        observed_manifest_sha != manifest_sha
        or manifest.get("freeze_id") != build_id
        or manifest.get("authorizations", {}).get(
            "stage13_execution_authorized_now"
        )
        is not False
    ):
        raise Stage13AuthorizationError("Stage 12 primary manifest differs")
    if replay_primary_bytes:
        from portfolio_bot.final_freeze.verification import replay_primary_stage12

        replay = replay_primary_stage12(project_root=root)
        if (
            replay.difference_count != 0
            or replay.prepared.freeze_id != build_id
            or replay.pointer_sha256 != primary_pointer_sha
        ):
            raise Stage13AuthorizationError(
                "Stage 12 independent primary byte replay differs"
            )
    return Stage12VerifiedBinding(
        build_id=build_id,
        verification_id=verification_id,
        manifest_sha256=manifest_sha,
        verified_pointer_path=STAGE12_VERIFIED_POINTER_PATH,
        verified_pointer_sha256=pointer_sha,
    )


def _receipt_payload(
    *,
    schema_version: int,
    receipt_id: str,
    approved: bool,
    approved_by: str,
    approved_at_utc: str,
    source_message_id: str,
    source_message_sha256: str,
    binding: Stage12VerifiedBinding,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "receipt_id": receipt_id,
        "scope": STAGE13_UNBLIND_SCOPE,
        "approved": approved,
        "approved_by": approved_by,
        "approved_at_utc": approved_at_utc,
        "source_message_id": source_message_id,
        "source_message_sha256": source_message_sha256,
        **binding.to_mapping(),
        "stage12_binding_sha256": binding.binding_sha256,
    }


@dataclass(frozen=True, slots=True)
class Stage13ApprovalReceipt:
    """One immutable, content-authenticated Stage 13 user decision."""

    schema_version: int
    receipt_id: str
    scope: str
    approved: bool
    approved_by: str
    approved_at_utc: str
    source_message_id: str
    source_message_sha256: str
    stage12_build_id: str
    stage12_verification_id: str
    stage12_manifest_sha256: str
    stage12_verified_pointer_path: str
    stage12_verified_pointer_sha256: str
    stage12_binding_sha256: str
    payload_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != STAGE13_APPROVAL_SCHEMA_VERSION
        ):
            raise Stage13AuthorizationError(
                "unsupported Stage 13 approval schema version"
            )
        _require_identifier(self.receipt_id, field="receipt_id")
        if self.scope != STAGE13_UNBLIND_SCOPE:
            raise Stage13AuthorizationError("approval scope is not exact Stage 13")
        if type(self.approved) is not bool:
            raise Stage13AuthorizationError("approved must be boolean")
        _require_nonempty_text(self.approved_by, field="approved_by")
        _require_utc_timestamp(self.approved_at_utc)
        _require_identifier(self.source_message_id, field="source_message_id")
        _require_hash(
            self.source_message_sha256,
            field="source_message_sha256",
        )
        binding = self.stage12_binding
        _require_hash(
            self.stage12_binding_sha256,
            field="stage12_binding_sha256",
        )
        if self.stage12_binding_sha256 != binding.binding_sha256:
            raise Stage13AuthorizationError("Stage 12 binding digest mismatch")
        _require_hash(self.payload_sha256, field="payload_sha256")
        if self.payload_sha256 != self.compute_payload_sha256():
            raise Stage13AuthorizationError("approval receipt payload hash mismatch")

    @property
    def stage12_binding(self) -> Stage12VerifiedBinding:
        return Stage12VerifiedBinding(
            build_id=self.stage12_build_id,
            verification_id=self.stage12_verification_id,
            manifest_sha256=self.stage12_manifest_sha256,
            verified_pointer_path=self.stage12_verified_pointer_path,
            verified_pointer_sha256=self.stage12_verified_pointer_sha256,
        )

    def payload_without_digest(self) -> dict[str, object]:
        return _receipt_payload(
            schema_version=self.schema_version,
            receipt_id=self.receipt_id,
            approved=self.approved,
            approved_by=self.approved_by,
            approved_at_utc=self.approved_at_utc,
            source_message_id=self.source_message_id,
            source_message_sha256=self.source_message_sha256,
            binding=self.stage12_binding,
        )

    def compute_payload_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.payload_without_digest()))

    def to_mapping(self) -> dict[str, object]:
        result = self.payload_without_digest()
        result["payload_sha256"] = self.payload_sha256
        return result

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "Stage13ApprovalReceipt":
        if not isinstance(value, Mapping):
            raise Stage13AuthorizationError("approval receipt must be a mapping")
        fields = frozenset(value)
        if fields != _RECEIPT_FIELDS:
            missing = sorted(_RECEIPT_FIELDS - fields)
            extra = sorted(fields - _RECEIPT_FIELDS)
            raise Stage13AuthorizationError(
                f"approval receipt field mismatch; missing={missing}, extra={extra}"
            )
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            receipt_id=value["receipt_id"],  # type: ignore[arg-type]
            scope=value["scope"],  # type: ignore[arg-type]
            approved=value["approved"],  # type: ignore[arg-type]
            approved_by=value["approved_by"],  # type: ignore[arg-type]
            approved_at_utc=value["approved_at_utc"],  # type: ignore[arg-type]
            source_message_id=value["source_message_id"],  # type: ignore[arg-type]
            source_message_sha256=value["source_message_sha256"],  # type: ignore[arg-type]
            stage12_build_id=value["stage12_build_id"],  # type: ignore[arg-type]
            stage12_verification_id=value["stage12_verification_id"],  # type: ignore[arg-type]
            stage12_manifest_sha256=value["stage12_manifest_sha256"],  # type: ignore[arg-type]
            stage12_verified_pointer_path=value[
                "stage12_verified_pointer_path"
            ],  # type: ignore[arg-type]
            stage12_verified_pointer_sha256=value[
                "stage12_verified_pointer_sha256"
            ],  # type: ignore[arg-type]
            stage12_binding_sha256=value[
                "stage12_binding_sha256"
            ],  # type: ignore[arg-type]
            payload_sha256=value["payload_sha256"],  # type: ignore[arg-type]
        )


def issue_stage13_unblind_approval(
    *,
    receipt_id: str,
    approved: bool,
    approved_by: str,
    approved_at_utc: str,
    source_message_id: str,
    source_message: str,
    stage12_binding: Stage12VerifiedBinding,
) -> Stage13ApprovalReceipt:
    """Issue a separate user receipt; a Stage 12 binding alone is insufficient."""

    if not isinstance(stage12_binding, Stage12VerifiedBinding):
        raise Stage13AuthorizationError(
            "stage12_binding must be a verified Stage 12 binding"
        )
    if not isinstance(source_message, str) or not source_message:
        raise Stage13AuthorizationError(
            "a nonempty, separate Stage 13 user message is required"
        )
    _require_identifier(receipt_id, field="receipt_id")
    if type(approved) is not bool:
        raise Stage13AuthorizationError("approved must be boolean")
    _require_nonempty_text(approved_by, field="approved_by")
    _require_utc_timestamp(approved_at_utc)
    _require_identifier(source_message_id, field="source_message_id")
    source_message_sha256 = _sha256_bytes(source_message.encode("utf-8"))
    payload = _receipt_payload(
        schema_version=STAGE13_APPROVAL_SCHEMA_VERSION,
        receipt_id=receipt_id,
        approved=approved,
        approved_by=approved_by,
        approved_at_utc=approved_at_utc,
        source_message_id=source_message_id,
        source_message_sha256=source_message_sha256,
        binding=stage12_binding,
    )
    return Stage13ApprovalReceipt(
        schema_version=STAGE13_APPROVAL_SCHEMA_VERSION,
        receipt_id=receipt_id,
        scope=STAGE13_UNBLIND_SCOPE,
        approved=approved,
        approved_by=approved_by,
        approved_at_utc=approved_at_utc,
        source_message_id=source_message_id,
        source_message_sha256=source_message_sha256,
        stage12_build_id=stage12_binding.build_id,
        stage12_verification_id=stage12_binding.verification_id,
        stage12_manifest_sha256=stage12_binding.manifest_sha256,
        stage12_verified_pointer_path=stage12_binding.verified_pointer_path,
        stage12_verified_pointer_sha256=(
            stage12_binding.verified_pointer_sha256
        ),
        stage12_binding_sha256=stage12_binding.binding_sha256,
        payload_sha256=_sha256_bytes(_canonical_json_bytes(payload)),
    )


def _coerce_receipt(
    value: Stage13ApprovalReceipt | Mapping[str, object],
) -> Stage13ApprovalReceipt:
    if isinstance(value, Stage13ApprovalReceipt):
        return Stage13ApprovalReceipt.from_mapping(value.to_mapping())
    return Stage13ApprovalReceipt.from_mapping(value)


def require_stage13_unblind_authorization(
    receipt: Stage13ApprovalReceipt | Mapping[str, object] | None,
    *,
    expected_stage12_binding: Stage12VerifiedBinding,
) -> Stage13ApprovalReceipt:
    """Authenticate one exact approval against the verified Stage 12 identity."""

    if receipt is None:
        raise Stage13AuthorizationError(
            "a separate explicit Stage 13 approval receipt is required"
        )
    if not isinstance(expected_stage12_binding, Stage12VerifiedBinding):
        raise Stage13AuthorizationError(
            "expected_stage12_binding must be a verified Stage 12 binding"
        )
    authenticated = _coerce_receipt(receipt)
    if not authenticated.approved:
        raise Stage13AuthorizationError(
            "Stage 13 approval receipt records a denied decision"
        )
    if authenticated.stage12_binding != expected_stage12_binding:
        raise Stage13AuthorizationError(
            "Stage 13 approval does not bind the expected verified Stage 12"
        )
    return authenticated


__all__ = [
    "STAGE13_APPROVAL_SCHEMA_VERSION",
    "STAGE13_UNBLIND_SCOPE",
    "STAGE12_VERIFIED_POINTER_PATH",
    "STAGE12_VERIFIED_STATUS",
    "Stage12VerifiedBinding",
    "Stage13ApprovalReceipt",
    "Stage13AuthorizationError",
    "issue_stage13_unblind_approval",
    "load_authenticated_stage12_binding",
    "require_stage13_unblind_authorization",
]
