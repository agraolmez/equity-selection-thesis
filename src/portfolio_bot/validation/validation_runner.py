"""Fail-closed authorization front door for empirical Stage 11 validation.

The exact Gate B approval is frozen at the one repository path below, and its
file and logical-payload digests are pinned in this source.  A caller still
cannot supply or replace an approval.  Execution additionally requires a
freshly republished, independently verified no-outcome preflight that binds
this approval, the frozen execution-input manifest, and the current code
(including this guard).

No function in this module resolves a market-data or outcome path.  The
context-manager API is deliberate: if a future empirical runner is connected,
the approval, the verified and primary pointers, and both immutable bundles
remain descriptor-anchored for the entire execution scope.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from portfolio_bot.data.secure_publication import (
    AuthenticatedBundleGuard,
    AuthenticatedPointerGuard,
    SecurePublicationError,
)
from portfolio_bot.validation.authorization import ApprovalScope
from portfolio_bot.validation.contract import (
    CONTRACT_ID,
    DEFAULT_POLICY,
    METHODOLOGY_FREEZE_SHA256,
    METHODOLOGY_SOURCE_SHA256,
    POLICY_ID,
    PREFLIGHT_POLICY_SHA256,
    PREFLIGHT_RULES_SHA256,
    PROJECT_ROOT,
    REQUIRED_PREFLIGHT_CHECK_NAMES,
    RULES_ID,
    STAGE10_PRIMARY_MANIFEST_SHA256,
    STAGE10_POLICY_SHA256,
    STAGE10_RULES_SHA256,
    STAGE10_VERIFIED_POINTER_SHA256,
    Stage11PreflightCounters,
    Stage11PreflightScopeError,
    authenticate_stage11_preflight_contract,
    canonical_json_bytes,
    enforce_stage11_preflight_boundary,
    sha256_file,
)
from portfolio_bot.validation.publication import (
    DEFAULT_IMPLEMENTATION_CODE_PATHS,
    DIFFERENCES_NAME,
    PRIMARY_FILES,
    PRIMARY_STATUS,
    SUCCESS_NAME,
    VERIFICATION_CHECKS_NAME,
    VERIFICATION_FILES,
    VERIFICATION_REPORT_NAME,
    VERIFIED_STATUS,
    _authenticate_primary,
    _validate_check_payload,
)


DEFAULT_PRIMARY_PREFLIGHT_POINTER = (
    PROJECT_ROOT
    / "artifacts/phase_05_validation/LATEST_STAGE_11_PREFLIGHT_PRIMARY.json"
)
DEFAULT_VERIFIED_PREFLIGHT_POINTER = (
    PROJECT_ROOT
    / "artifacts/phase_05_validation/LATEST_STAGE_11_PREFLIGHT_VERIFIED.json"
)

EXECUTION_APPROVAL_CONTRACT_RELATIVE_PATH = (
    "configs/phase_05_validation/"
    "stage_11_validation_execution_approval_v1.json"
)
DEFAULT_EXECUTION_APPROVAL_CONTRACT = (
    PROJECT_ROOT / EXECUTION_APPROVAL_CONTRACT_RELATIVE_PATH
)
EXECUTION_APPROVAL_CONTRACT_ID = (
    "phase05-stage11-validation-execution-approval-v1"
)

# The user's explicit decision is captured verbatim in the exact contract
# above, including the SHA-256 of those UTF-8 bytes.  No fabricated platform
# message identifier is part of the authority.  The contract is also part of
# DEFAULT_IMPLEMENTATION_CODE_PATHS, so the final no-outcome preflight binds
# both the decision and this guard before empirical execution can open.
FROZEN_EXECUTION_APPROVAL_CONTRACT_FILE_SHA256: str | None = (
    "47e32a555884cc7cb33c5c5b9f25eec0bd2b7cda7ba17027f5b845441665dae8"
)
FROZEN_EXECUTION_APPROVAL_CONTRACT_PAYLOAD_SHA256: str | None = (
    "07159cf4e9eafc70385168e96ca39b90df0fd8fe49cfcc3a68e146f0348d991a"
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_BUILD_ID = re.compile(r"stage11-preflight-[0-9a-f]{24}\Z")
_VERIFICATION_ID = re.compile(r"verify-[0-9a-f]{16}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}\Z")
_UTC_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z"
)

_EXECUTION_APPROVAL_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "stage",
        "scope",
        "approved",
        "approved_by",
        "approved_at_utc",
        "approval_text",
        "approval_text_sha256",
        "execution_input_manifest_sha256",
        "stage10_verified_pointer_sha256",
        "stage11_policy_sha256",
        "stage11_rules_sha256",
        "methodology_source_sha256",
        "methodology_freeze_sha256",
        "preflight_republication_required",
        "payload_sha256",
    }
)

_VERIFIED_POINTER_FIELDS = frozenset(
    {
        "schema_version",
        "stage",
        "contract_id",
        "policy_id",
        "rules_id",
        "build_id",
        "verification_id",
        "status",
        "content_fingerprint",
        "bundle_path",
        "bundle_files",
        "verification_report_sha256",
        "primary_pointer_sha256",
        "primary_manifest_sha256",
        "contract_receipt_sha256",
        "difference_count",
        "authorizations",
    }
)

_VERIFICATION_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "stage",
        "build_id",
        "status",
        "scope",
        "difference_count",
        "authorizations",
        "primary_pointer_sha256",
        "primary_manifest_sha256",
        "bindings",
        "evidence",
        "evidence_sha256",
        "counters",
        "summary",
        "outcome_or_performance_payload_members",
    }
)

_VERIFIED_AUTHORIZATIONS = {
    "independent_verification_completed": True,
    "no_outcome_preflight_technical_gate_passed": True,
    "validation_unblind_requires_separate_approval": True,
    "validation_outcome_or_forward_return_access_authorized_now": False,
    "real_validation_weights_or_trades_authorized_now": False,
    "real_validation_pnl_or_performance_authorized_now": False,
    "final_2020_2025_access_authorized": False,
    "stage12_or_later_authorized": False,
}


class Stage11ValidationSealedError(RuntimeError):
    """Raised before outcome-path resolution whenever validation is sealed."""


class _Stage11ValidationAuthorityLease:
    """Private liveness cell revoked when the descriptor-anchored gate closes."""

    __slots__ = ("__active",)

    def __init__(self) -> None:
        self.__active = True

    @property
    def active(self) -> bool:
        return self.__active

    def revoke(self, *, _issuance_token: object) -> None:
        if _issuance_token is not _VALIDATION_UNBLIND_AUTHORITY_ISSUANCE_TOKEN:
            raise Stage11ValidationSealedError(
                "Stage 11 validation authority lease cannot be changed by a caller"
            )
        self.__active = False


_VALIDATION_UNBLIND_AUTHORITY_ISSUANCE_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class Stage11ValidationUnblindAuthority:
    """Privately issued, hash-bound authority for one live unblind scope.

    Direct construction is deliberately unavailable.  The private lease is
    revoked as soon as :func:`open_stage11_validation_unblind` exits, so a
    retained dataclass value cannot be replayed to open a later data broker.
    """

    approval_contract_file_sha256: str
    approval_contract_payload_sha256: str
    execution_input_manifest_sha256: str
    verified_pointer_sha256: str
    build_id: str
    verification_id: str
    verification_report_sha256: str
    primary_pointer_sha256: str
    primary_manifest_sha256: str
    contract_receipt_sha256: str
    _lease: _Stage11ValidationAuthorityLease
    _descriptor_gate: Callable[[], None] | None

    def __init__(
        self,
        *,
        approval_contract_file_sha256: str,
        approval_contract_payload_sha256: str,
        execution_input_manifest_sha256: str,
        verified_pointer_sha256: str,
        build_id: str,
        verification_id: str,
        verification_report_sha256: str,
        primary_pointer_sha256: str,
        primary_manifest_sha256: str,
        contract_receipt_sha256: str,
        _descriptor_gate: Callable[[], None] | None = None,
        _issuance_token: object,
    ) -> None:
        if _issuance_token is not _VALIDATION_UNBLIND_AUTHORITY_ISSUANCE_TOKEN:
            raise Stage11ValidationSealedError(
                "Stage 11 validation authority must be issued by the "
                "descriptor-anchored repository gate"
            )
        hash_fields = {
            "approval_contract_file_sha256": approval_contract_file_sha256,
            "approval_contract_payload_sha256": approval_contract_payload_sha256,
            "execution_input_manifest_sha256": execution_input_manifest_sha256,
            "verified_pointer_sha256": verified_pointer_sha256,
            "verification_report_sha256": verification_report_sha256,
            "primary_pointer_sha256": primary_pointer_sha256,
            "primary_manifest_sha256": primary_manifest_sha256,
            "contract_receipt_sha256": contract_receipt_sha256,
        }
        for name, value in hash_fields.items():
            if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
                raise Stage11ValidationSealedError(
                    f"Stage 11 validation authority {name} is not a lowercase SHA-256"
                )
        if not isinstance(build_id, str) or _BUILD_ID.fullmatch(build_id) is None:
            raise Stage11ValidationSealedError(
                "Stage 11 validation authority build_id is malformed"
            )
        if (
            not isinstance(verification_id, str)
            or _VERIFICATION_ID.fullmatch(verification_id) is None
        ):
            raise Stage11ValidationSealedError(
                "Stage 11 validation authority verification_id is malformed"
            )
        for name, value in hash_fields.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "build_id", build_id)
        object.__setattr__(self, "verification_id", verification_id)
        object.__setattr__(self, "_lease", _Stage11ValidationAuthorityLease())
        object.__setattr__(self, "_descriptor_gate", _descriptor_gate)


def _issue_stage11_validation_unblind_authority(
    *,
    approval_contract_file_sha256: str,
    approval_contract_payload_sha256: str,
    execution_input_manifest_sha256: str,
    verified_pointer_sha256: str,
    build_id: str,
    verification_id: str,
    verification_report_sha256: str,
    primary_pointer_sha256: str,
    primary_manifest_sha256: str,
    contract_receipt_sha256: str,
    _descriptor_gate: Callable[[], None] | None = None,
) -> Stage11ValidationUnblindAuthority:
    """Private issuer used only after every descriptor-backed check passes."""

    return Stage11ValidationUnblindAuthority(
        approval_contract_file_sha256=approval_contract_file_sha256,
        approval_contract_payload_sha256=approval_contract_payload_sha256,
        execution_input_manifest_sha256=execution_input_manifest_sha256,
        verified_pointer_sha256=verified_pointer_sha256,
        build_id=build_id,
        verification_id=verification_id,
        verification_report_sha256=verification_report_sha256,
        primary_pointer_sha256=primary_pointer_sha256,
        primary_manifest_sha256=primary_manifest_sha256,
        contract_receipt_sha256=contract_receipt_sha256,
        _descriptor_gate=_descriptor_gate,
        _issuance_token=_VALIDATION_UNBLIND_AUTHORITY_ISSUANCE_TOKEN,
    )


def require_live_stage11_validation_unblind_authority(
    authority: object,
) -> Stage11ValidationUnblindAuthority:
    """Authenticate a live authority before every empirical access operation."""

    _require(
        type(authority) is Stage11ValidationUnblindAuthority,
        "Stage 11 validation execution requires the exact unblind authority type",
    )
    typed = authority
    _require(
        typed._lease.active,
        "Stage 11 validation unblind authority is outside its live gate scope",
    )
    frozen_file = FROZEN_EXECUTION_APPROVAL_CONTRACT_FILE_SHA256
    frozen_payload = FROZEN_EXECUTION_APPROVAL_CONTRACT_PAYLOAD_SHA256
    _require(
        isinstance(frozen_file, str)
        and _HEX64.fullmatch(frozen_file) is not None
        and isinstance(frozen_payload, str)
        and _HEX64.fullmatch(frozen_payload) is not None,
        "Stage 11 validation remains sealed: execution approval digests are not frozen",
    )
    _require(
        typed.approval_contract_file_sha256 == frozen_file
        and typed.approval_contract_payload_sha256 == frozen_payload,
        "Stage 11 validation unblind authority differs from the frozen approval",
    )
    from portfolio_bot.validation.execution_input_manifest import (
        FROZEN_EXECUTION_INPUT_MANIFEST_SHA256,
    )

    _require(
        isinstance(FROZEN_EXECUTION_INPUT_MANIFEST_SHA256, str)
        and _HEX64.fullmatch(FROZEN_EXECUTION_INPUT_MANIFEST_SHA256) is not None
        and typed.execution_input_manifest_sha256
        == FROZEN_EXECUTION_INPUT_MANIFEST_SHA256,
        "Stage 11 validation authority differs from the frozen execution input manifest",
    )
    if typed._descriptor_gate is not None:
        try:
            typed._descriptor_gate()
        except Stage11ValidationSealedError:
            raise
        except (SecurePublicationError, RuntimeError, ValueError, OSError) as exc:
            raise Stage11ValidationSealedError(
                "Stage 11 validation authority descriptor gate failed"
            ) from exc
    return typed


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11ValidationSealedError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11ValidationSealedError(f"invalid {label} JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _require_exact_keys(
    value: object,
    expected: frozenset[str] | set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    observed = frozenset(value)
    expected_set = frozenset(expected)
    _require(
        observed == expected_set,
        f"{label} fields differ; missing={sorted(expected_set - observed)}, "
        f"extra={sorted(observed - expected_set)}",
    )
    return value


def _require_sha256(value: object, *, label: str) -> str:
    _require(
        isinstance(value, str) and _HEX64.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _current_code_hashes() -> dict[str, str]:
    paths = tuple(DEFAULT_IMPLEMENTATION_CODE_PATHS)
    _require(
        len(paths) == len(set(paths)) and bool(paths),
        "verified preflight code-binding path set is malformed",
    )
    guard_path = "src/portfolio_bot/validation/validation_runner.py"
    _require(
        guard_path in paths,
        "verified preflight code bindings omit the validation-unblind guard",
    )
    result: dict[str, str] = {}
    for relative in paths:
        path = Path(relative)
        _require(
            bool(path.parts) and not path.is_absolute() and ".." not in path.parts,
            f"unsafe verified preflight code-binding path: {relative}",
        )
        selected = PROJECT_ROOT / path
        _require(
            selected.is_file() and not selected.is_symlink(),
            f"missing verified preflight code dependency: {relative}",
        )
        result[path.as_posix()] = sha256_file(selected)
    return dict(sorted(result.items()))


def _validate_execution_approval_contract(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the exact future repository-frozen execution contract."""

    contract = _require_exact_keys(
        value,
        _EXECUTION_APPROVAL_FIELDS,
        label="validation-execution approval contract",
    )
    _require(
        contract.get("schema_version") == 1
        and contract.get("contract_id") == EXECUTION_APPROVAL_CONTRACT_ID
        and contract.get("stage") == 11
        and contract.get("scope") == ApprovalScope.VALIDATION_OUTCOME_UNBLIND.value
        and contract.get("approved") is True
        and contract.get("preflight_republication_required") is True,
        "validation-execution approval contract identity or scope differs",
    )
    _require(
        isinstance(contract.get("approved_by"), str)
        and bool(str(contract["approved_by"]).strip())
        and contract["approved_by"] == str(contract["approved_by"]).strip()
        and len(str(contract["approved_by"])) <= 256,
        "validation-execution approval authority is malformed",
    )
    approval_text = contract.get("approval_text")
    _require(
        isinstance(approval_text, str)
        and bool(approval_text)
        and approval_text == approval_text.strip()
        and len(approval_text.encode("utf-8")) <= 8_192,
        "validation-execution approval text is malformed",
    )
    approved_at = contract.get("approved_at_utc")
    _require(
        isinstance(approved_at, str)
        and _UTC_TIMESTAMP.fullmatch(approved_at) is not None,
        "validation-execution approval timestamp is malformed",
    )
    try:
        parsed = dt.datetime.fromisoformat(approved_at[:-1] + "+00:00")
    except ValueError as exc:
        raise Stage11ValidationSealedError(
            "validation-execution approval timestamp is invalid"
        ) from exc
    _require(
        parsed.utcoffset() == dt.timedelta(0),
        "validation-execution approval timestamp is not UTC",
    )
    expected_authorities = {
        "stage10_verified_pointer_sha256": STAGE10_VERIFIED_POINTER_SHA256,
        "stage11_policy_sha256": PREFLIGHT_POLICY_SHA256,
        "stage11_rules_sha256": PREFLIGHT_RULES_SHA256,
        "methodology_source_sha256": METHODOLOGY_SOURCE_SHA256,
        "methodology_freeze_sha256": METHODOLOGY_FREEZE_SHA256,
    }
    for name, expected in expected_authorities.items():
        _require_sha256(contract.get(name), label=name)
        _require(
            contract.get(name) == expected,
            f"validation-execution approval authority differs for {name}",
        )
    from portfolio_bot.validation.execution_input_manifest import (
        FROZEN_EXECUTION_INPUT_MANIFEST_SHA256,
    )

    execution_manifest_sha = _require_sha256(
        contract.get("execution_input_manifest_sha256"),
        label="execution_input_manifest_sha256",
    )
    _require(
        execution_manifest_sha == FROZEN_EXECUTION_INPUT_MANIFEST_SHA256,
        "validation-execution approval does not bind the frozen execution input manifest",
    )
    approval_text_sha256 = _require_sha256(
        contract.get("approval_text_sha256"), label="approval_text_sha256"
    )
    _require(
        approval_text_sha256 == _sha256_bytes(approval_text.encode("utf-8")),
        "validation-execution approval text digest differs",
    )
    supplied_payload_sha = _require_sha256(
        contract.get("payload_sha256"), label="payload_sha256"
    )
    payload_without_digest = dict(contract)
    payload_without_digest.pop("payload_sha256")
    computed_payload_sha = _sha256_bytes(
        canonical_json_bytes(payload_without_digest)
    )
    _require(
        supplied_payload_sha == computed_payload_sha,
        "validation-execution approval payload digest differs",
    )
    _require(
        supplied_payload_sha == FROZEN_EXECUTION_APPROVAL_CONTRACT_PAYLOAD_SHA256,
        "validation-execution approval payload is not the separately frozen decision",
    )
    return contract


def _expected_authority_hashes(receipt: object) -> dict[str, str]:
    names = (
        "policy_sha256",
        "rules_sha256",
        "methodology_source_sha256",
        "methodology_freeze_sha256",
        "stage10_verified_pointer_sha256",
        "stage10_primary_manifest_sha256",
        "stage10_policy_sha256",
        "stage10_rules_sha256",
        "approval_file_sha256",
        "approval_payload_sha256",
    )
    result = {name: str(getattr(receipt, name)) for name in names}
    _require(
        all(_HEX64.fullmatch(value) is not None for value in result.values()),
        "authenticated preflight receipt contains a malformed authority hash",
    )
    return result


def _validated_preflight_counters(value: object) -> Stage11PreflightCounters:
    """Accept audited predictor reads while keeping every forbidden counter zero."""

    counters = _require_exact_keys(
        value,
        frozenset(Stage11PreflightCounters.__dataclass_fields__),
        label="verified preflight counters",
    )
    try:
        authenticated = Stage11PreflightCounters(**dict(counters))
        enforce_stage11_preflight_boundary(authenticated)
    except (TypeError, Stage11PreflightScopeError) as exc:
        raise Stage11ValidationSealedError(
            "verified preflight counters violate the no-outcome boundary"
        ) from exc
    return authenticated


def _validate_evidence(value: object) -> tuple[dict[str, object], ...]:
    _require(isinstance(value, list), "verification evidence must be a list")
    rows: list[dict[str, object]] = []
    for raw in value:
        row = _require_exact_keys(
            raw,
            {"check_name", "evidence_sha256", "observation_count"},
            label="verification evidence row",
        )
        _require(
            isinstance(row.get("check_name"), str)
            and row.get("check_name") in REQUIRED_PREFLIGHT_CHECK_NAMES,
            "verification evidence check name differs",
        )
        _require_sha256(row.get("evidence_sha256"), label="evidence_sha256")
        _require(
            type(row.get("observation_count")) is int
            and int(row["observation_count"]) >= 0,
            "verification evidence observation count is malformed",
        )
        rows.append(dict(row))
    names = tuple(str(row["check_name"]) for row in rows)
    _require(
        names == tuple(sorted(REQUIRED_PREFLIGHT_CHECK_NAMES)),
        "verification evidence membership or order differs",
    )
    return tuple(rows)


def _validate_summary(value: object) -> dict[str, object]:
    summary = _require_exact_keys(
        value,
        {"runner_id", "implementation_mode", "required_check_count"},
        label="verification summary",
    )
    runner_id = summary.get("runner_id")
    _require(
        isinstance(runner_id, str)
        and _IDENTIFIER.fullmatch(runner_id) is not None,
        "verification runner identity is malformed",
    )
    _require(
        summary.get("implementation_mode")
        == "synthetic_and_development_parity_no_outcome"
        and summary.get("required_check_count")
        == len(REQUIRED_PREFLIGHT_CHECK_NAMES),
        "verification summary identity differs",
    )
    return dict(summary)


def _validate_verified_pointer(value: Mapping[str, Any]) -> Mapping[str, Any]:
    pointer = _require_exact_keys(
        value, _VERIFIED_POINTER_FIELDS, label="verified preflight pointer"
    )
    _require(
        pointer.get("schema_version") == 1
        and pointer.get("stage") == 11
        and pointer.get("contract_id") == CONTRACT_ID
        and pointer.get("policy_id") == POLICY_ID
        and pointer.get("rules_id") == RULES_ID
        and pointer.get("status") == VERIFIED_STATUS
        and pointer.get("difference_count") == 0,
        "verified preflight pointer identity, status, or differences differ",
    )
    build_id = pointer.get("build_id")
    verification_id = pointer.get("verification_id")
    _require(
        isinstance(build_id, str) and _BUILD_ID.fullmatch(build_id) is not None,
        "verified preflight build identity is malformed",
    )
    _require(
        isinstance(verification_id, str)
        and _VERIFICATION_ID.fullmatch(verification_id) is not None,
        "verified preflight verification identity is malformed",
    )
    _require_sha256(pointer.get("content_fingerprint"), label="content_fingerprint")
    for name in (
        "verification_report_sha256",
        "primary_pointer_sha256",
        "primary_manifest_sha256",
        "contract_receipt_sha256",
    ):
        _require_sha256(pointer.get(name), label=name)
    _require(
        pointer.get("authorizations") == _VERIFIED_AUTHORIZATIONS,
        "verified preflight authorizations differ",
    )
    expected_bundle_path = (
        "artifacts/phase_05_validation/stage_11_preflight_verifications/"
        + verification_id
    )
    _require(
        pointer.get("bundle_path") == expected_bundle_path,
        "verified preflight bundle path is not the exact repository path",
    )
    hashes = _require_exact_keys(
        pointer.get("bundle_files"),
        set(VERIFICATION_FILES),
        label="verified preflight bundle hashes",
    )
    for name, digest in hashes.items():
        _require_sha256(digest, label=f"verified bundle member {name}")
    return pointer


@contextlib.contextmanager
def _open_frozen_execution_approval(
    *, code_hashes: Mapping[str, str]
) -> Iterator[tuple[Mapping[str, Any], AuthenticatedPointerGuard]]:
    file_sha = FROZEN_EXECUTION_APPROVAL_CONTRACT_FILE_SHA256
    payload_sha = FROZEN_EXECUTION_APPROVAL_CONTRACT_PAYLOAD_SHA256
    if file_sha is None or payload_sha is None:
        raise Stage11ValidationSealedError(
            "Stage 11 validation remains sealed: no separately frozen "
            "validation-execution approval contract digest exists. A new "
            "explicit approval must be frozen at the repository contract path, "
            "pinned in the guard, and followed by a republished independently "
            "verified no-outcome preflight."
        )
    _require_sha256(file_sha, label="frozen execution-approval file digest")
    _require_sha256(payload_sha, label="frozen execution-approval payload digest")
    _require(
        EXECUTION_APPROVAL_CONTRACT_RELATIVE_PATH in code_hashes,
        "validation remains sealed: the execution-approval contract is not "
        "part of the preflight's exact code-binding set; republish is required",
    )
    _require(
        code_hashes[EXECUTION_APPROVAL_CONTRACT_RELATIVE_PATH] == file_sha,
        "validation remains sealed: execution-approval contract code binding differs",
    )
    with AuthenticatedPointerGuard.open(
        trusted_root=PROJECT_ROOT,
        path=DEFAULT_EXECUTION_APPROVAL_CONTRACT,
        expected_sha256=file_sha,
        label="repository-frozen Stage 11 validation-execution approval contract",
    ) as approval_guard:
        contract = _validate_execution_approval_contract(
            _json(approval_guard.payload, label="validation-execution approval contract")
        )
        approval_guard.gate()
        yield contract, approval_guard
        approval_guard.gate()


@contextlib.contextmanager
def open_stage11_validation_unblind() -> Iterator[Stage11ValidationUnblindAuthority]:
    """Open the future gate and retain every authority through the caller scope.

    This is the only API a future empirical runner should use.  It has no path
    or receipt arguments, so caller-created JSON cannot become authority.
    Today it always raises because the separately frozen execution contract is
    intentionally absent.
    """

    try:
        code_hashes = _current_code_hashes()
        with contextlib.ExitStack() as stack:
            contract, approval_guard = stack.enter_context(
                _open_frozen_execution_approval(code_hashes=code_hashes)
            )
            verified_pointer_guard = stack.enter_context(
                AuthenticatedPointerGuard.open(
                    trusted_root=PROJECT_ROOT,
                    path=DEFAULT_VERIFIED_PREFLIGHT_POINTER,
                    label="repository-root Stage 11 verified preflight pointer",
                )
            )
            pointer = _validate_verified_pointer(
                _json(
                    verified_pointer_guard.payload,
                    label="verified preflight pointer",
                )
            )
            bundle_path = PROJECT_ROOT / str(pointer["bundle_path"])
            bundle_hashes = {
                str(name): str(digest)
                for name, digest in pointer["bundle_files"].items()
            }
            verified_bundle_guard = stack.enter_context(
                AuthenticatedBundleGuard.open(
                    trusted_root=PROJECT_ROOT,
                    path=bundle_path,
                    expected_files=VERIFICATION_FILES,
                    expected_hashes=bundle_hashes,
                    label="Stage 11 verified no-outcome preflight bundle",
                    _permit_authenticated_directory_ctime_refresh=True,
                )
            )

            with verified_bundle_guard.open_binary(VERIFICATION_REPORT_NAME) as stream:
                report_payload = stream.read()
            _require(
                _sha256_bytes(report_payload)
                == pointer["verification_report_sha256"],
                "verification report hash differs from the trusted pointer",
            )
            report = _require_exact_keys(
                _json(report_payload, label="verification report"),
                _VERIFICATION_REPORT_FIELDS,
                label="verification report",
            )
            _require(
                report.get("schema_version") == 1
                and report.get("stage") == 11
                and report.get("build_id") == pointer["build_id"]
                and report.get("status") == VERIFIED_STATUS
                and report.get("scope") == "NO_OUTCOME_PREFLIGHT_ONLY"
                and report.get("difference_count") == 0
                and report.get("outcome_or_performance_payload_members") == 0,
                "verification report identity, scope, or differences differ",
            )
            _require(
                report.get("authorizations") == _VERIFIED_AUTHORIZATIONS,
                "verification report authorizations differ",
            )
            _require(
                report.get("primary_pointer_sha256")
                == pointer["primary_pointer_sha256"]
                and report.get("primary_manifest_sha256")
                == pointer["primary_manifest_sha256"],
                "verification report primary bindings differ from the pointer",
            )

            counters = _validated_preflight_counters(report.get("counters"))
            receipt = authenticate_stage11_preflight_contract(
                DEFAULT_POLICY,
                project_root=PROJECT_ROOT,
                counters=counters,
            )
            expected_authority_hashes = _expected_authority_hashes(receipt)
            bindings = _require_exact_keys(
                report.get("bindings"),
                {"contract_receipt_sha256", "authority_hashes", "code_hashes"},
                label="verification report bindings",
            )
            _require(
                bindings.get("contract_receipt_sha256")
                == pointer["contract_receipt_sha256"]
                == receipt.sha256,
                "verification report contract-receipt binding differs",
            )
            _require(
                bindings.get("authority_hashes") == expected_authority_hashes,
                "verification report authority or policy hashes differ",
            )
            _require(
                bindings.get("code_hashes") == code_hashes,
                "verification report does not bind the exact current code set",
            )
            _require(
                code_hashes["src/portfolio_bot/validation/validation_runner.py"]
                == sha256_file(Path(__file__)),
                "verification report validation-unblind guard hash differs",
            )

            evidence = _validate_evidence(report.get("evidence"))
            _require(
                report.get("evidence_sha256")
                == _sha256_bytes(canonical_json_bytes(list(evidence))),
                "verification evidence digest differs",
            )
            verification_summary = _validate_summary(report.get("summary"))

            with verified_bundle_guard.open_binary(VERIFICATION_CHECKS_NAME) as stream:
                verification_checks_payload = stream.read()
            verification_checks = _validate_check_payload(
                verification_checks_payload,
                label="validation-unblind verification",
            )

            with verified_bundle_guard.open_binary(DIFFERENCES_NAME) as stream:
                differences_payload = stream.read()
            differences = _require_exact_keys(
                _json(differences_payload, label="verification differences"),
                {"schema_version", "difference_count", "differences"},
                label="verification differences",
            )
            _require(
                differences
                == {"schema_version": 1, "difference_count": 0, "differences": []},
                "verified preflight differences are not exactly empty",
            )

            with verified_bundle_guard.open_binary(SUCCESS_NAME) as stream:
                success_payload = stream.read()
            success = _require_exact_keys(
                _json(success_payload, label="verification success marker"),
                {
                    "schema_version",
                    "stage",
                    "build_id",
                    "verification_id",
                    "status",
                    "verification_report_sha256",
                    "verification_checks_sha256",
                    "differences_sha256",
                    "closed_files",
                    "outcome_or_performance_payload_members",
                },
                label="verification success marker",
            )
            _require(
                success.get("schema_version") == 1
                and success.get("stage") == 11
                and success.get("build_id") == pointer["build_id"]
                and success.get("verification_id") == pointer["verification_id"]
                and success.get("status")
                == "PREFLIGHT_VERIFICATION_BUNDLE_COMPLETE"
                and success.get("verification_report_sha256")
                == _sha256_bytes(report_payload)
                and success.get("verification_checks_sha256")
                == _sha256_bytes(verification_checks_payload)
                and success.get("differences_sha256")
                == _sha256_bytes(differences_payload)
                and success.get("closed_files") == list(VERIFICATION_FILES)
                and success.get("outcome_or_performance_payload_members") == 0,
                "verification success marker identity or member bindings differ",
            )

            primary = stack.enter_context(
                _authenticate_primary(
                    pointer_path=DEFAULT_PRIMARY_PREFLIGHT_POINTER,
                    trusted_root=PROJECT_ROOT,
                    members=PRIMARY_FILES,
                    receipt=receipt,
                    expected_code_hashes=code_hashes,
                )
            )
            _require(
                primary.pointer_sha256 == pointer["primary_pointer_sha256"]
                == report["primary_pointer_sha256"]
                and primary.manifest_sha256 == pointer["primary_manifest_sha256"]
                == report["primary_manifest_sha256"]
                and primary.build_id == pointer["build_id"],
                "verified preflight does not bind the current trusted primary bundle",
            )
            _require(
                primary.counters == counters.to_dict(),
                "primary and verified preflight counters differ",
            )
            _require(
                tuple(primary.checks) == tuple(verification_checks),
                "independent verification checks differ from the trusted primary",
            )
            _require(
                tuple(primary.evidence) == evidence,
                "independent verification evidence differs from the trusted primary",
            )
            _require(
                primary.summary.get("runner_id")
                != verification_summary.get("runner_id")
                and primary.summary.get("implementation_mode")
                == verification_summary.get("implementation_mode")
                and primary.summary.get("required_check_count")
                == verification_summary.get("required_check_count"),
                "independent verification runner identity is not distinct from primary",
            )

            def descriptor_gate() -> None:
                approval_guard.gate()
                verified_pointer_guard.gate()
                verified_bundle_guard.gate()
                primary.gate()

            authority = _issue_stage11_validation_unblind_authority(
                approval_contract_file_sha256=str(
                    FROZEN_EXECUTION_APPROVAL_CONTRACT_FILE_SHA256
                ),
                approval_contract_payload_sha256=str(
                    FROZEN_EXECUTION_APPROVAL_CONTRACT_PAYLOAD_SHA256
                ),
                execution_input_manifest_sha256=str(
                    contract["execution_input_manifest_sha256"]
                ),
                verified_pointer_sha256=verified_pointer_guard.payload_sha256,
                build_id=str(pointer["build_id"]),
                verification_id=str(pointer["verification_id"]),
                verification_report_sha256=str(
                    pointer["verification_report_sha256"]
                ),
                primary_pointer_sha256=str(pointer["primary_pointer_sha256"]),
                primary_manifest_sha256=str(pointer["primary_manifest_sha256"]),
                contract_receipt_sha256=str(pointer["contract_receipt_sha256"]),
                _descriptor_gate=descriptor_gate,
            )
            descriptor_gate()
            try:
                yield authority
            finally:
                # Revoke before any descriptor is released.  All empirical
                # broker operations recheck this liveness cell.
                authority._lease.revoke(
                    _issuance_token=_VALIDATION_UNBLIND_AUTHORITY_ISSUANCE_TOKEN
                )
                primary.gate()
                verified_bundle_guard.gate()
                verified_pointer_guard.gate()
                approval_guard.gate()
    except Stage11ValidationSealedError:
        raise
    except (SecurePublicationError, RuntimeError, ValueError, OSError) as exc:
        raise Stage11ValidationSealedError(
            f"Stage 11 validation remains sealed: authority authentication failed: {exc}"
        ) from exc


def require_stage11_validation_unblind(
    *,
    approval_path: Path | None = None,
    verified_preflight_pointer_path: Path | None = None,
) -> Stage11ValidationUnblindAuthority:
    """One-shot diagnostic check; future execution must use the context manager.

    Legacy arguments remain solely to fail with an explicit security error.
    Neither a caller-supplied receipt nor a caller-supplied pointer is accepted.
    """

    if approval_path is not None or verified_preflight_pointer_path is not None:
        raise Stage11ValidationSealedError(
            "caller-supplied approval receipts and preflight pointers are prohibited; "
            "only the exact repository-frozen authorities may open validation"
        )
    with open_stage11_validation_unblind() as authority:
        return authority


__all__ = [
    "DEFAULT_EXECUTION_APPROVAL_CONTRACT",
    "DEFAULT_PRIMARY_PREFLIGHT_POINTER",
    "DEFAULT_VERIFIED_PREFLIGHT_POINTER",
    "EXECUTION_APPROVAL_CONTRACT_ID",
    "EXECUTION_APPROVAL_CONTRACT_RELATIVE_PATH",
    "FROZEN_EXECUTION_APPROVAL_CONTRACT_FILE_SHA256",
    "FROZEN_EXECUTION_APPROVAL_CONTRACT_PAYLOAD_SHA256",
    "Stage11ValidationSealedError",
    "Stage11ValidationUnblindAuthority",
    "open_stage11_validation_unblind",
    "require_live_stage11_validation_unblind_authority",
    "require_stage11_validation_unblind",
]
