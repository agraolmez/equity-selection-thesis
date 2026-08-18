"""Immutable, scope-separated approval receipts for roadmap Stage 11.

An upstream technical pass is not human authorization.  These pure objects bind
one explicit decision to one policy hash and the current verified Stage 10
pointer.  Exact-scope checks prevent an implementation/preflight approval from
being reused to open validation outcomes or the final period.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


APPROVAL_RECEIPT_SCHEMA_VERSION = 1
STAGE10_VERIFIED_POINTER_SHA256 = (
    "af78595e3cd6f35535dd9d30ac018c6c053dfb9e798ff8f00779cf0a451d58a2"
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{2,127}$")
_UTC_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)


class AuthorizationError(RuntimeError):
    """Raised when an approval is absent, altered, denied, or too broad."""


class ApprovalScope(str, Enum):
    """Mutually exclusive permissions; no scope implies another scope."""

    IMPLEMENTATION_NO_OUTCOME_PREFLIGHT = (
        "stage11_implementation_no_outcome_preflight"
    )
    VALIDATION_OUTCOME_UNBLIND = "stage11_validation_2015_2019_outcome_unblind"
    FINAL_PERIOD_OUTCOME_UNBLIND = "stage13_final_2020_2025_outcome_unblind"


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
        "stage10_verified_pointer_sha256",
        "stage11_policy_sha256",
        "payload_sha256",
    }
)


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
        raise AuthorizationError(f"{field} must be 64 lowercase hex characters")
    return value


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise AuthorizationError(f"{field} is not a valid bounded identifier")
    return value


def _require_nonempty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AuthorizationError(f"{field} must be nonempty normalized text")
    if len(value) > 256:
        raise AuthorizationError(f"{field} exceeds 256 characters")
    return value


def _require_utc_timestamp(value: object) -> str:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise AuthorizationError("approved_at_utc must be canonical ISO-8601 UTC")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AuthorizationError("approved_at_utc is not a valid timestamp") from exc
    if parsed.utcoffset() != dt.timedelta(0):
        raise AuthorizationError("approved_at_utc must be UTC")
    return value


def _payload_without_digest(
    *,
    schema_version: int,
    receipt_id: str,
    scope: ApprovalScope,
    approved: bool,
    approved_by: str,
    approved_at_utc: str,
    source_message_id: str,
    source_message_sha256: str,
    stage10_verified_pointer_sha256: str,
    stage11_policy_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "receipt_id": receipt_id,
        "scope": scope.value,
        "approved": approved,
        "approved_by": approved_by,
        "approved_at_utc": approved_at_utc,
        "source_message_id": source_message_id,
        "source_message_sha256": source_message_sha256,
        "stage10_verified_pointer_sha256": stage10_verified_pointer_sha256,
        "stage11_policy_sha256": stage11_policy_sha256,
    }


@dataclass(frozen=True, slots=True)
class ApprovalReceipt:
    """One content-authenticated, immutable, exact-scope user decision."""

    schema_version: int
    receipt_id: str
    scope: ApprovalScope
    approved: bool
    approved_by: str
    approved_at_utc: str
    source_message_id: str
    source_message_sha256: str
    stage10_verified_pointer_sha256: str
    stage11_policy_sha256: str
    payload_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise AuthorizationError("schema_version must be an integer")
        if self.schema_version != APPROVAL_RECEIPT_SCHEMA_VERSION:
            raise AuthorizationError("unsupported approval-receipt schema version")
        _require_identifier(self.receipt_id, field="receipt_id")
        if not isinstance(self.scope, ApprovalScope):
            raise AuthorizationError("scope must be an ApprovalScope")
        if type(self.approved) is not bool:
            raise AuthorizationError("approved must be a boolean")
        _require_nonempty_text(self.approved_by, field="approved_by")
        _require_utc_timestamp(self.approved_at_utc)
        _require_identifier(self.source_message_id, field="source_message_id")
        _require_hash(self.source_message_sha256, field="source_message_sha256")
        _require_hash(
            self.stage10_verified_pointer_sha256,
            field="stage10_verified_pointer_sha256",
        )
        _require_hash(self.stage11_policy_sha256, field="stage11_policy_sha256")
        _require_hash(self.payload_sha256, field="payload_sha256")
        expected = self.compute_payload_sha256()
        if self.payload_sha256 != expected:
            raise AuthorizationError("approval receipt payload hash mismatch")

    def payload_without_digest(self) -> dict[str, object]:
        return _payload_without_digest(
            schema_version=self.schema_version,
            receipt_id=self.receipt_id,
            scope=self.scope,
            approved=self.approved,
            approved_by=self.approved_by,
            approved_at_utc=self.approved_at_utc,
            source_message_id=self.source_message_id,
            source_message_sha256=self.source_message_sha256,
            stage10_verified_pointer_sha256=self.stage10_verified_pointer_sha256,
            stage11_policy_sha256=self.stage11_policy_sha256,
        )

    def compute_payload_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.payload_without_digest()))

    def to_mapping(self) -> dict[str, object]:
        value = self.payload_without_digest()
        value["payload_sha256"] = self.payload_sha256
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ApprovalReceipt":
        """Parse an exact receipt shape and authenticate its entire payload."""

        if not isinstance(value, Mapping):
            raise AuthorizationError("approval receipt must be a mapping")
        fields = frozenset(value)
        if fields != _RECEIPT_FIELDS:
            missing = sorted(_RECEIPT_FIELDS - fields)
            extra = sorted(fields - _RECEIPT_FIELDS)
            raise AuthorizationError(
                f"approval receipt field mismatch; missing={missing}, extra={extra}"
            )
        try:
            scope = ApprovalScope(value["scope"])
        except (TypeError, ValueError) as exc:
            raise AuthorizationError("approval receipt has an unknown scope") from exc
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            receipt_id=value["receipt_id"],  # type: ignore[arg-type]
            scope=scope,
            approved=value["approved"],  # type: ignore[arg-type]
            approved_by=value["approved_by"],  # type: ignore[arg-type]
            approved_at_utc=value["approved_at_utc"],  # type: ignore[arg-type]
            source_message_id=value["source_message_id"],  # type: ignore[arg-type]
            source_message_sha256=value["source_message_sha256"],  # type: ignore[arg-type]
            stage10_verified_pointer_sha256=value[
                "stage10_verified_pointer_sha256"
            ],  # type: ignore[arg-type]
            stage11_policy_sha256=value["stage11_policy_sha256"],  # type: ignore[arg-type]
            payload_sha256=value["payload_sha256"],  # type: ignore[arg-type]
        )


def issue_approval_receipt(
    *,
    receipt_id: str,
    scope: ApprovalScope,
    approved: bool,
    approved_by: str,
    approved_at_utc: str,
    source_message_id: str,
    source_message: str,
    stage11_policy_sha256: str,
    stage10_verified_pointer_sha256: str = STAGE10_VERIFIED_POINTER_SHA256,
) -> ApprovalReceipt:
    """Create a receipt with a digest over every authorization-relevant field."""

    if not isinstance(source_message, str) or not source_message:
        raise AuthorizationError("source_message must be nonempty")
    message_sha256 = _sha256_bytes(source_message.encode("utf-8"))
    # Validate everything except the digest once before deriving it.
    _require_identifier(receipt_id, field="receipt_id")
    if not isinstance(scope, ApprovalScope):
        raise AuthorizationError("scope must be an ApprovalScope")
    if type(approved) is not bool:
        raise AuthorizationError("approved must be a boolean")
    _require_nonempty_text(approved_by, field="approved_by")
    _require_utc_timestamp(approved_at_utc)
    _require_identifier(source_message_id, field="source_message_id")
    _require_hash(stage11_policy_sha256, field="stage11_policy_sha256")
    _require_hash(
        stage10_verified_pointer_sha256,
        field="stage10_verified_pointer_sha256",
    )
    payload = _payload_without_digest(
        schema_version=APPROVAL_RECEIPT_SCHEMA_VERSION,
        receipt_id=receipt_id,
        scope=scope,
        approved=approved,
        approved_by=approved_by,
        approved_at_utc=approved_at_utc,
        source_message_id=source_message_id,
        source_message_sha256=message_sha256,
        stage10_verified_pointer_sha256=stage10_verified_pointer_sha256,
        stage11_policy_sha256=stage11_policy_sha256,
    )
    return ApprovalReceipt(
        schema_version=APPROVAL_RECEIPT_SCHEMA_VERSION,
        receipt_id=receipt_id,
        scope=scope,
        approved=approved,
        approved_by=approved_by,
        approved_at_utc=approved_at_utc,
        source_message_id=source_message_id,
        source_message_sha256=message_sha256,
        stage10_verified_pointer_sha256=stage10_verified_pointer_sha256,
        stage11_policy_sha256=stage11_policy_sha256,
        payload_sha256=_sha256_bytes(_canonical_json_bytes(payload)),
    )


def _coerce_receipt(value: ApprovalReceipt | Mapping[str, object]) -> ApprovalReceipt:
    if isinstance(value, ApprovalReceipt):
        # Re-run the immutable object's integrity check in case a caller used
        # low-level object mutation to bypass dataclass freezing.
        return ApprovalReceipt.from_mapping(value.to_mapping())
    return ApprovalReceipt.from_mapping(value)


def require_authorization(
    receipt: ApprovalReceipt | Mapping[str, object] | None,
    *,
    required_scope: ApprovalScope,
    expected_stage11_policy_sha256: str,
    expected_stage10_verified_pointer_sha256: str = STAGE10_VERIFIED_POINTER_SHA256,
) -> ApprovalReceipt:
    """Require one exact scope; approvals never inherit across boundaries."""

    if receipt is None:
        raise AuthorizationError("an explicit approval receipt is required")
    _require_hash(
        expected_stage11_policy_sha256,
        field="expected_stage11_policy_sha256",
    )
    _require_hash(
        expected_stage10_verified_pointer_sha256,
        field="expected_stage10_verified_pointer_sha256",
    )
    authenticated = _coerce_receipt(receipt)
    if not authenticated.approved:
        raise AuthorizationError("approval receipt records a denied decision")
    if authenticated.scope is not required_scope:
        raise AuthorizationError(
            f"approval scope {authenticated.scope.value!r} does not authorize "
            f"{required_scope.value!r}"
        )
    if (
        authenticated.stage10_verified_pointer_sha256
        != expected_stage10_verified_pointer_sha256
    ):
        raise AuthorizationError("Stage 10 verified-pointer binding mismatch")
    if authenticated.stage11_policy_sha256 != expected_stage11_policy_sha256:
        raise AuthorizationError("Stage 11 policy binding mismatch")
    return authenticated


def require_implementation_preflight_authorization(
    receipt: ApprovalReceipt | Mapping[str, object] | None,
    *,
    expected_stage11_policy_sha256: str,
) -> ApprovalReceipt:
    return require_authorization(
        receipt,
        required_scope=ApprovalScope.IMPLEMENTATION_NO_OUTCOME_PREFLIGHT,
        expected_stage11_policy_sha256=expected_stage11_policy_sha256,
    )


def require_validation_outcome_unblind_authorization(
    receipt: ApprovalReceipt | Mapping[str, object] | None,
    *,
    expected_stage11_policy_sha256: str,
) -> ApprovalReceipt:
    return require_authorization(
        receipt,
        required_scope=ApprovalScope.VALIDATION_OUTCOME_UNBLIND,
        expected_stage11_policy_sha256=expected_stage11_policy_sha256,
    )


def require_final_period_outcome_unblind_authorization(
    receipt: ApprovalReceipt | Mapping[str, object] | None,
    *,
    expected_stage11_policy_sha256: str,
) -> ApprovalReceipt:
    return require_authorization(
        receipt,
        required_scope=ApprovalScope.FINAL_PERIOD_OUTCOME_UNBLIND,
        expected_stage11_policy_sha256=expected_stage11_policy_sha256,
    )


__all__ = [
    "APPROVAL_RECEIPT_SCHEMA_VERSION",
    "STAGE10_VERIFIED_POINTER_SHA256",
    "ApprovalReceipt",
    "ApprovalScope",
    "AuthorizationError",
    "issue_approval_receipt",
    "require_authorization",
    "require_final_period_outcome_unblind_authorization",
    "require_implementation_preflight_authorization",
    "require_validation_outcome_unblind_authorization",
]
