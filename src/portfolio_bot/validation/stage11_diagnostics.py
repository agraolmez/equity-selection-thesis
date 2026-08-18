"""Authenticated, performance-blind persistence for Stage 11 diagnostics.

This module is deliberately independent of the empirical portfolio loop.  It
provides a small persistence protocol that the loop can call at four explicit
boundaries:

* publish all three monthly allocation problems before any solver is called;
* publish one private post-accounting checkpoint after a month succeeds;
* publish one aggregate report after all three strategies have been attempted;
  and
* seal the diagnostic path only after the exact 59-month chain is complete.

Diagnostic authority is append-only.  Bundles are immutable, content
addressed, exact closed sets.  There is no mutable ``LATEST`` pointer: resume
derives the unique longest authenticated checkpoint chain.  A corrupt bundle,
hash mismatch, fork, gap, incompatible identity, symlink, hard link, unsafe
permission, or out-of-window month fails closed.

Public status and failure payloads contain no realized portfolio performance.
The private checkpoint state is opaque to this module because the accounting
engine owns its schema; it is stored in a mode-0600 file below mode-0700
directories and is never copied into a public/status receipt.

The module never reads a market, return, outcome, or final-period data source.
Its only accepted holding-month domain is January 2015 through November 2019.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import re
import stat
import struct
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from portfolio_bot.data.secure_publication import (
    AuthenticatedBundleGuard,
    SecurePublicationError,
    SecureStagingDirectory,
)
from portfolio_bot.validation.calendar import (
    FIRST_VALIDATION_HOLDING_MONTH,
    LAST_VALIDATION_EXIT_DATE,
    LAST_VALIDATION_HOLDING_MONTH,
    VALIDATION_HOLDING_MONTH_COUNT,
    VALIDATION_HOLDING_MONTHS,
    validation_holding_month_descriptor,
)


DIAGNOSTIC_PROTOCOL_ID = "stage11-authenticated-diagnostic-persistence-v1"
PROBLEM_BUNDLE_KIND = "stage11_monthly_allocation_problem_bundle"
CHECKPOINT_BUNDLE_KIND = "stage11_post_month_checkpoint"
FAILURE_BUNDLE_KIND = "stage11_aggregate_solver_failure_report"
COMPLETION_BUNDLE_KIND = "stage11_diagnostic_completion_seal"

PRIMARY_STRATEGY_ID = "ensemble_hybrid_primary"
ELASTIC_NET_CHALLENGER_ID = "elastic_net_hybrid_challenger"
RANK_CHALLENGER_ID = "ensemble_rank_challenger"
STAGE11_DIAGNOSTIC_STRATEGY_IDS = (
    PRIMARY_STRATEGY_ID,
    ELASTIC_NET_CHALLENGER_ID,
    RANK_CHALLENGER_ID,
)

PROBLEM_FILE_BY_STRATEGY = MappingProxyType(
    {
        PRIMARY_STRATEGY_ID: "ensemble_hybrid_primary.problem.bin",
        ELASTIC_NET_CHALLENGER_ID: (
            "elastic_net_hybrid_challenger.problem.bin"
        ),
        RANK_CHALLENGER_ID: "ensemble_rank_challenger.problem.bin",
    }
)

PROBLEM_MANIFEST_NAME = "manifest.json"
CHECKPOINT_MANIFEST_NAME = "manifest.json"
FAILURE_MANIFEST_NAME = "manifest.json"
COMPLETION_MANIFEST_NAME = "manifest.json"
PUBLIC_STATUS_NAME = "status.json"
FAILURE_REPORT_NAME = "failure_report.json"
PRIVATE_FAILURE_DETAILS_NAME = "failure_details.json"
PRIVATE_STATE_NAME = "checkpoint_state.bin"
SUCCESS_NAME = "_SUCCESS.json"

PROBLEM_MEMBER_NAMES = tuple(
    sorted((*PROBLEM_FILE_BY_STRATEGY.values(), PROBLEM_MANIFEST_NAME, SUCCESS_NAME))
)
CHECKPOINT_MEMBER_NAMES = tuple(
    sorted(
        (
            CHECKPOINT_MANIFEST_NAME,
            PRIVATE_STATE_NAME,
            PUBLIC_STATUS_NAME,
            SUCCESS_NAME,
        )
    )
)
FAILURE_MEMBER_NAMES = tuple(
    sorted(
        (
            FAILURE_MANIFEST_NAME,
            FAILURE_REPORT_NAME,
            PRIVATE_FAILURE_DETAILS_NAME,
            SUCCESS_NAME,
        )
    )
)
COMPLETION_MEMBER_NAMES = tuple(
    sorted((COMPLETION_MANIFEST_NAME, PUBLIC_STATUS_NAME, SUCCESS_NAME))
)

COMPATIBILITY_IDENTITY_FIELDS = (
    "methodology_sha256",
    "preflight_sha256",
    "gate_b_sha256",
    "execution_input_authority_sha256",
    "validation_calendar_sha256",
    "checkpoint_state_schema_sha256",
    "allocation_problem_schema_sha256",
    "accounting_transition_code_sha256",
    "diagnostic_workflow_sha256",
)

POST_MONTH_ATTESTATION_FIELDS = (
    "all_three_allocation_problems_saved_before_solve",
    "all_active_strategies_solved_or_inactive_receipted",
    "accounting_complete",
    "corporate_actions_applied",
    "mark_continuity_verified",
    "authority_gate_passed",
    "independent_checkpoint_reopen_passed",
    "stage12_rows_read_is_zero",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_FAILURE_CODE_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,159}\Z")
_SAFE_EXCEPTION_CLASS_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_.]{0,199}\Z"
)
_SAFE_PRIVATE_ARRAY_NAME_RE = re.compile(
    r"[a-z][a-z0-9_.:-]{0,159}\Z"
)
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_PROBLEM_DIRECTORY_RE = re.compile(
    r"(?P<ordinal>\d{3})-(?P<month>\d{4}-\d{2})-problems-"
    r"(?P<digest>[0-9a-f]{64})\Z"
)
_CHECKPOINT_DIRECTORY_RE = re.compile(
    r"(?P<ordinal>\d{3})-(?P<month>\d{4}-\d{2})-checkpoint-"
    r"(?P<digest>[0-9a-f]{64})\Z"
)
_FAILURE_DIRECTORY_RE = re.compile(
    r"(?P<ordinal>\d{3})-(?P<month>\d{4}-\d{2})-failures-"
    r"(?P<digest>[0-9a-f]{64})\Z"
)
_COMPLETION_DIRECTORY_RE = re.compile(
    r"stage11-diagnostic-complete-(?P<digest>[0-9a-f]{64})\Z"
)

_PROBLEM_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol_id",
        "holding_month",
        "holding_month_ordinal",
        "compatibility",
        "compatibility_sha256",
        "producer_implementation_sha256",
        "opening_checkpoint_sha256",
        "strategy_ids",
        "strategy_problem_files",
        "saved_before_any_solver_attempt",
        "boundary",
        "performance_fields_emitted",
    }
)
_CHECKPOINT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol_id",
        "holding_month",
        "holding_month_ordinal",
        "completed_month_count",
        "next_holding_month",
        "compatibility",
        "compatibility_sha256",
        "producer_implementation_sha256",
        "previous_checkpoint_sha256",
        "problem_bundle_sha256",
        "private_state",
        "post_month_attestation",
        "boundary",
        "performance_fields_emitted",
    }
)
_FAILURE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol_id",
        "holding_month",
        "holding_month_ordinal",
        "compatibility",
        "compatibility_sha256",
        "producer_implementation_sha256",
        "opening_checkpoint_sha256",
        "problem_bundle_sha256",
        "failure_report",
        "private_failure_details",
        "all_active_strategies_attempted_and_inactive_receipted",
        "checkpoint_advanced",
        "boundary",
        "performance_fields_emitted",
    }
)
_COMPLETION_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "protocol_id",
        "compatibility",
        "compatibility_sha256",
        "producer_implementation_sha256",
        "completed_month_count",
        "first_holding_month",
        "last_holding_month",
        "latest_checkpoint_sha256",
        "checkpoint_chain_sha256",
        "observed_failure_ids",
        "regression_case_sha256_by_failure_id",
        "regression_suite_sha256",
        "regression_suite_passed",
        "unresolved_failure_count",
        "boundary",
        "performance_fields_emitted",
    }
)

# These names may never occur as keys in a diagnostic public/status payload.
# ``performance_fields_emitted`` is the one explicit denial counter and is
# accepted only when its value is the integer zero.
_PROHIBITED_PUBLIC_KEY_TOKENS = frozenset(
    {
        "return",
        "returns",
        "realizedreturn",
        "realizedreturns",
        "netreturn",
        "grossreturn",
        "excessreturn",
        "residualreturn",
        "benchmarkreturn",
        "pnl",
        "profit",
        "profits",
        "profitandloss",
        "nav",
        "netassetvalue",
        "portfoliovalue",
        "accountvalue",
        "wealth",
        "sharpe",
        "sortino",
        "alpha",
        "beta",
        "drawdown",
        "volatility",
        "rankic",
        "linearic",
        "informationcoefficient",
        "performance",
        "performancestatistic",
        "performancestatistics",
        "monthlymetrics",
        "dailyaccounting",
        "transactioncost",
        "borrowcost",
        "financingcost",
        "turnover",
        "grossexposure",
        "netexposure",
        "cashweight",
        "objectivevalue",
        "weights",
        "trades",
    }
)

_PRIVATE_CONTAINER_MAGIC = b"STAGE11-PRIVATE-ARRAY-CONTAINER-V1\n"
_PRIVATE_CONTAINER_HEADER_FIELDS = frozenset(
    {"schema_version", "payload_kind", "metadata", "arrays"}
)
_PRIVATE_ARRAY_RECORD_FIELDS = frozenset(
    {"name", "offset", "size_bytes", "sha256", "dtype", "shape"}
)


class Stage11DiagnosticPersistenceError(RuntimeError):
    """Raised when diagnostic evidence is incomplete, unsafe, or incompatible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11DiagnosticPersistenceError(message)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: object, field: str) -> str:
    _require(_is_sha256(value), f"{field} must be a lowercase SHA-256")
    return str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise Stage11DiagnosticPersistenceError(
            "diagnostic JSON must be canonical and finite"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_exact_json(
    payload: bytes,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11DiagnosticPersistenceError(
            f"{label} is not canonical UTF-8 JSON"
        ) from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    _require(
        frozenset(value) == fields,
        f"{label} field set differs; "
        f"missing={sorted(fields - frozenset(value))}, "
        f"extra={sorted(frozenset(value) - fields)}",
    )
    _require(
        _canonical_json_bytes(value) == payload,
        f"{label} is not in canonical byte form",
    )
    return value


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _month_ordinal(holding_month: str) -> int:
    descriptor = validation_holding_month_descriptor(holding_month)
    return descriptor.ordinal


def _next_holding_month(ordinal: int) -> str | None:
    if ordinal == VALIDATION_HOLDING_MONTH_COUNT:
        return None
    return VALIDATION_HOLDING_MONTHS[ordinal]


def _boundary_payload() -> dict[str, Any]:
    return {
        "stage": 11,
        "first_holding_month": FIRST_VALIDATION_HOLDING_MONTH,
        "last_holding_month": LAST_VALIDATION_HOLDING_MONTH,
        "latest_permitted_observation_date": LAST_VALIDATION_EXIT_DATE.isoformat(),
        "stage12_access_authorized": False,
        "stage12_rows_read": 0,
        "final_period_rows_read": 0,
    }


def _validate_boundary(value: object) -> None:
    _require(value == _boundary_payload(), "Stage 11 boundary evidence differs")


def _normalize_public_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _contains_prohibited_public_token(value: object) -> bool:
    normalized = _normalize_public_key(value)
    return any(token in normalized for token in _PROHIBITED_PUBLIC_KEY_TOKENS)


def _failure_code_exposes_performance(code: str) -> bool:
    """Reject result-bearing labels while allowing constraint diagnostics."""

    tokens = tuple(token for token in re.split(r"[_.:-]+", code) if token)
    return any(
        token
        in {
            "return",
            "returns",
            "pnl",
            "profit",
            "loss",
            "nav",
            "sharpe",
            "alpha",
            "drawdown",
            "turnover",
            "performance",
        }
        for token in tokens
    )


def assert_stage11_diagnostic_public_payload_safe(value: object) -> None:
    """Reject realized-performance fields from any public diagnostic payload."""

    def walk(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            for raw_key, nested in item.items():
                _require(isinstance(raw_key, str), f"{path} keys must be strings")
                normalized = _normalize_public_key(raw_key)
                if normalized == "performancefieldsemitted":
                    _require(
                        type(nested) is int and nested == 0,
                        "performance_fields_emitted must be exactly zero",
                    )
                else:
                    _require(
                        not _contains_prohibited_public_token(normalized),
                        f"prohibited performance field in public payload: {raw_key}",
                    )
                walk(nested, f"{path}.{raw_key}")
            return
        if isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                walk(nested, f"{path}[{index}]")
            return
        _require(
            item is None
            or type(item) in {str, int, bool}
            or (type(item) is float and item == item and abs(item) != float("inf")),
            f"{path} contains a non-JSON or non-finite value",
        )

    walk(value, "public_payload")


@dataclass(frozen=True, slots=True)
class Stage11DiagnosticCompatibility:
    """Strict identities that determine whether persisted state may be resumed.

    The producer implementation hash is intentionally *not* part of this
    state-compatibility object.  Every bundle separately records that hash.
    This permits a solver-only correction at a failed month to resume the prior
    accounting checkpoint, while any methodology, input, calendar, state
    schema, accounting-transition, or workflow change still invalidates the
    chain.
    """

    methodology_sha256: str
    preflight_sha256: str
    gate_b_sha256: str
    execution_input_authority_sha256: str
    validation_calendar_sha256: str
    checkpoint_state_schema_sha256: str
    allocation_problem_schema_sha256: str
    accounting_transition_code_sha256: str
    diagnostic_workflow_sha256: str

    def __post_init__(self) -> None:
        for field in COMPATIBILITY_IDENTITY_FIELDS:
            _require_sha256(getattr(self, field), field)

    def as_dict(self) -> dict[str, str]:
        return {
            field: getattr(self, field)
            for field in COMPATIBILITY_IDENTITY_FIELDS
        }

    @property
    def sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.as_dict()))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "Stage11DiagnosticCompatibility":
        _require(
            frozenset(value) == frozenset(COMPATIBILITY_IDENTITY_FIELDS),
            "diagnostic compatibility identity field set differs",
        )
        return cls(
            **{
                field: _require_sha256(value[field], field)
                for field in COMPATIBILITY_IDENTITY_FIELDS
            }
        )


@dataclass(frozen=True, slots=True)
class MonthlyProblemBundleReceipt:
    holding_month: str
    ordinal: int
    bundle_sha256: str
    path: Path
    opening_checkpoint_sha256: str | None
    compatibility_sha256: str
    producer_implementation_sha256: str
    strategy_payloads: Mapping[str, bytes]

    def __post_init__(self) -> None:
        _require(self.ordinal == _month_ordinal(self.holding_month), "ordinal differs")
        _require_sha256(self.bundle_sha256, "bundle_sha256")
        _require_sha256(self.compatibility_sha256, "compatibility_sha256")
        _require_sha256(
            self.producer_implementation_sha256,
            "producer_implementation_sha256",
        )
        if self.opening_checkpoint_sha256 is not None:
            _require_sha256(
                self.opening_checkpoint_sha256,
                "opening_checkpoint_sha256",
            )
        _require(
            tuple(self.strategy_payloads) == STAGE11_DIAGNOSTIC_STRATEGY_IDS,
            "problem receipt strategy ordering differs",
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedDiagnosticCheckpoint:
    holding_month: str
    ordinal: int
    checkpoint_sha256: str
    previous_checkpoint_sha256: str | None
    problem_bundle_sha256: str
    path: Path
    compatibility_sha256: str
    producer_implementation_sha256: str
    private_state: bytes
    public_status: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require(self.ordinal == _month_ordinal(self.holding_month), "ordinal differs")
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        if self.previous_checkpoint_sha256 is not None:
            _require_sha256(
                self.previous_checkpoint_sha256,
                "previous_checkpoint_sha256",
            )
        _require_sha256(self.problem_bundle_sha256, "problem_bundle_sha256")
        _require_sha256(self.compatibility_sha256, "compatibility_sha256")
        _require_sha256(
            self.producer_implementation_sha256,
            "producer_implementation_sha256",
        )
        _require(bool(self.private_state), "checkpoint private state is empty")
        assert_stage11_diagnostic_public_payload_safe(self.public_status)


@dataclass(frozen=True, slots=True)
class StrategyDiagnosticResult:
    """One strategy result in an aggregate, performance-blind failure report."""

    strategy_id: str
    status: str
    failure_code: str | None = None
    exception_class: str | None = None
    private_detail_sha256: str | None = None
    inactive_reason: str | None = None
    private_detail: bytes | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require(
            self.strategy_id in STAGE11_DIAGNOSTIC_STRATEGY_IDS,
            "unknown Stage 11 diagnostic strategy",
        )
        _require(
            self.status in {"passed", "failed", "inactive"},
            "invalid strategy result status",
        )
        if self.status == "passed":
            _require(
                self.failure_code is None
                and self.exception_class is None
                and self.private_detail is None
                and self.private_detail_sha256 is None
                and self.inactive_reason is None,
                "passed result may not carry failure detail",
            )
            return
        if self.status == "inactive":
            _require(
                self.failure_code is None
                and self.exception_class is None
                and self.private_detail is None
                and self.private_detail_sha256 is None,
                "inactive result may not carry failure detail",
            )
            _require(
                isinstance(self.inactive_reason, str)
                and _SAFE_FAILURE_CODE_RE.fullmatch(self.inactive_reason) is not None,
                "inactive_reason is unsafe",
            )
            return
        _require(self.inactive_reason is None, "failed result may not be inactive")
        _require(
            isinstance(self.failure_code, str)
            and _SAFE_FAILURE_CODE_RE.fullmatch(self.failure_code) is not None,
            "failure_code is unsafe",
        )
        _require(
            not _failure_code_exposes_performance(self.failure_code),
            "failure_code may not encode a performance result",
        )
        _require(
            isinstance(self.exception_class, str)
            and _SAFE_EXCEPTION_CLASS_RE.fullmatch(self.exception_class) is not None,
            "exception_class is unsafe",
        )
        _require_sha256(self.private_detail_sha256, "private_detail_sha256")
        if self.private_detail is not None:
            _require(
                type(self.private_detail) is bytes
                and bool(self.private_detail)
                and _sha256_bytes(self.private_detail)
                == self.private_detail_sha256,
                "private failure detail differs from its digest",
            )

    @classmethod
    def passed(cls, strategy_id: str) -> "StrategyDiagnosticResult":
        return cls(strategy_id=strategy_id, status="passed")

    @classmethod
    def inactive(
        cls,
        strategy_id: str,
        *,
        inactive_reason: str,
    ) -> "StrategyDiagnosticResult":
        return cls(
            strategy_id=strategy_id,
            status="inactive",
            inactive_reason=inactive_reason,
        )

    @classmethod
    def failed(
        cls,
        strategy_id: str,
        *,
        failure_code: str,
        exception_class: str,
        private_detail: bytes,
    ) -> "StrategyDiagnosticResult":
        _require(
            isinstance(private_detail, bytes) and bool(private_detail),
            "private failure detail must be nonempty bytes",
        )
        return cls(
            strategy_id=strategy_id,
            status="failed",
            failure_code=failure_code,
            exception_class=exception_class,
            private_detail_sha256=_sha256_bytes(private_detail),
            private_detail=private_detail,
        )

    def as_public_dict(
        self,
        *,
        holding_month: str,
        problem_bundle_sha256: str,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "strategy_id": self.strategy_id,
            "status": self.status,
        }
        if self.status == "failed":
            identity = {
                "holding_month": holding_month,
                "problem_bundle_sha256": problem_bundle_sha256,
                "strategy_id": self.strategy_id,
                "failure_code": self.failure_code,
                "exception_class": self.exception_class,
            }
            value.update(identity)
            value["private_detail_sha256"] = self.private_detail_sha256
            value["failure_id"] = _sha256_bytes(_canonical_json_bytes(identity))
        elif self.status == "inactive":
            value["inactive_reason"] = self.inactive_reason
        return value


@dataclass(frozen=True, slots=True)
class Stage11AggregateFailureReport:
    """All three strategy attempts for one month, reported together."""

    holding_month: str
    problem_bundle_sha256: str
    strategy_results: tuple[StrategyDiagnosticResult, ...]

    def __post_init__(self) -> None:
        _month_ordinal(self.holding_month)
        _require_sha256(self.problem_bundle_sha256, "problem_bundle_sha256")
        _require(
            tuple(result.strategy_id for result in self.strategy_results)
            == STAGE11_DIAGNOSTIC_STRATEGY_IDS,
            "failure report must contain all three strategies in locked order",
        )
        _require(
            any(result.status == "failed" for result in self.strategy_results),
            "aggregate failure report must contain at least one failure",
        )

    @property
    def failure_ids(self) -> tuple[str, ...]:
        return tuple(
            result.as_public_dict(
                holding_month=self.holding_month,
                problem_bundle_sha256=self.problem_bundle_sha256,
            )["failure_id"]
            for result in self.strategy_results
            if result.status == "failed"
        )

    def as_public_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": 1,
            "protocol_id": DIAGNOSTIC_PROTOCOL_ID,
            "holding_month": self.holding_month,
            "holding_month_ordinal": _month_ordinal(self.holding_month),
            "problem_bundle_sha256": self.problem_bundle_sha256,
            "all_active_strategies_attempted_and_inactive_receipted": True,
            "strategy_results": [
                item.as_public_dict(
                    holding_month=self.holding_month,
                    problem_bundle_sha256=self.problem_bundle_sha256,
                )
                for item in self.strategy_results
            ],
            "failure_count": sum(
                item.status == "failed" for item in self.strategy_results
            ),
            "performance_fields_emitted": 0,
        }
        assert_stage11_diagnostic_public_payload_safe(result)
        return result


@dataclass(frozen=True, slots=True)
class DiagnosticCompletionReceipt:
    completion_sha256: str
    path: Path
    producer_implementation_sha256: str
    latest_checkpoint_sha256: str
    checkpoint_chain_sha256: str
    completed_month_count: int
    observed_failure_ids: tuple[str, ...]
    regression_case_sha256_by_failure_id: Mapping[str, str]
    regression_suite_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.completion_sha256, "completion_sha256")
        _require_sha256(
            self.producer_implementation_sha256,
            "producer_implementation_sha256",
        )
        _require_sha256(
            self.latest_checkpoint_sha256,
            "latest_checkpoint_sha256",
        )
        _require_sha256(self.checkpoint_chain_sha256, "checkpoint_chain_sha256")
        _require(
            self.completed_month_count == VALIDATION_HOLDING_MONTH_COUNT,
            "completion receipt must cover exactly 59 months",
        )
        for failure_id in self.observed_failure_ids:
            _require_sha256(failure_id, "observed_failure_id")
        _require(
            tuple(sorted(self.regression_case_sha256_by_failure_id))
            == self.observed_failure_ids,
            "completion regression mapping differs",
        )
        for digest in self.regression_case_sha256_by_failure_id.values():
            _require_sha256(digest, "regression_case_sha256")
        _require_sha256(
            self.regression_suite_sha256,
            "regression_suite_sha256",
        )


@dataclass(frozen=True, slots=True)
class DecodedPrivateArrayContainer:
    """Authenticated decoded content from a private state/problem payload."""

    payload_kind: str
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DiagnosticResumePoint:
    """Backend-facing resume result with a zero-based next-month index."""

    next_month_index: int
    next_holding_month: str | None
    completed_checkpoint: AuthenticatedDiagnosticCheckpoint | None
    state_arrays: Mapping[str, np.ndarray]
    state_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require(
            type(self.next_month_index) is int
            and 0 <= self.next_month_index <= VALIDATION_HOLDING_MONTH_COUNT,
            "next_month_index is outside the exact Stage 11 calendar",
        )
        _require(
            self.next_holding_month
            == (
                None
                if self.next_month_index == VALIDATION_HOLDING_MONTH_COUNT
                else VALIDATION_HOLDING_MONTHS[self.next_month_index]
            ),
            "resume next holding month differs from the frozen calendar",
        )
        if self.completed_checkpoint is None:
            _require(self.next_month_index == 0, "empty resume must start in January")
            _require(not self.state_arrays, "empty resume may not contain state arrays")
            _require(not self.state_metadata, "empty resume may not contain metadata")
        else:
            _require(
                self.completed_checkpoint.ordinal == self.next_month_index,
                "resume index differs from completed checkpoint count",
            )


@dataclass(frozen=True, slots=True)
class DiagnosticFailureStatus:
    """Blinded result of publishing one aggregate monthly failure report."""

    holding_month: str
    failure_report_sha256: str
    attempted_strategy_count: int
    failure_count: int
    checkpoint_advanced: bool
    performance_fields_emitted: int

    def __post_init__(self) -> None:
        _month_ordinal(self.holding_month)
        _require_sha256(
            self.failure_report_sha256,
            "failure_report_sha256",
        )
        _require(self.attempted_strategy_count == 3, "strategy count differs")
        _require(1 <= self.failure_count <= 3, "failure count differs")
        _require(self.checkpoint_advanced is False, "failure advanced checkpoint")
        _require(
            self.performance_fields_emitted == 0,
            "failure status exposes performance",
        )
        assert_stage11_diagnostic_public_payload_safe(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "holding_month": self.holding_month,
            "failure_report_sha256": self.failure_report_sha256,
            "attempted_strategy_count": self.attempted_strategy_count,
            "failure_count": self.failure_count,
            "checkpoint_advanced": self.checkpoint_advanced,
            "performance_fields_emitted": self.performance_fields_emitted,
        }


def _encode_private_array_container(
    *,
    payload_kind: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> bytes:
    _require(
        payload_kind
        in {
            "stage11_checkpoint_state",
            "stage11_strategy_problem",
        },
        "private payload kind differs",
    )
    _require(isinstance(arrays, Mapping), "private arrays must be a mapping")
    _require(isinstance(metadata, Mapping), "private metadata must be a mapping")
    # Canonical round-trip validates that metadata is finite JSON and detaches
    # it from mutable caller-owned mappings.  Performance values are allowed
    # here because this container is private mode-0600 state, never status.
    metadata_bytes = _canonical_json_bytes(dict(metadata))
    detached_metadata = json.loads(metadata_bytes.decode("utf-8"))

    encoded_arrays: list[tuple[str, bytes, np.ndarray]] = []
    for name in sorted(arrays):
        _require(
            isinstance(name, str)
            and _SAFE_PRIVATE_ARRAY_NAME_RE.fullmatch(name) is not None,
            f"unsafe private array name: {name!r}",
        )
        value = np.asarray(arrays[name])
        _require(
            not value.dtype.hasobject,
            f"private array {name} may not use object dtype",
        )
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        encoded_arrays.append((name, buffer.getvalue(), value))

    offset = 0
    records: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    for name, payload, value in encoded_arrays:
        records.append(
            {
                "name": name,
                "offset": offset,
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            }
        )
        chunks.append(payload)
        offset += len(payload)
    header = {
        "schema_version": 1,
        "payload_kind": payload_kind,
        "metadata": detached_metadata,
        "arrays": records,
    }
    header_bytes = _canonical_json_bytes(header)
    return (
        _PRIVATE_CONTAINER_MAGIC
        + struct.pack(">Q", len(header_bytes))
        + header_bytes
        + b"".join(chunks)
    )


def encode_stage11_checkpoint_state(
    *,
    state_arrays: Mapping[str, np.ndarray],
    state_metadata: Mapping[str, Any],
) -> bytes:
    """Encode private continuation arrays and metadata deterministically."""

    return _encode_private_array_container(
        payload_kind="stage11_checkpoint_state",
        arrays=state_arrays,
        metadata=state_metadata,
    )


def encode_stage11_strategy_problem(
    *,
    problem_arrays: Mapping[str, np.ndarray],
    problem_metadata: Mapping[str, Any],
) -> bytes:
    """Encode APD, rank-boundary, and MonthlyProblemBuild audit primitives.

    Callers should include every primitive needed for an exact replay.  In
    particular, rank payloads include the locked long/short entry boundaries;
    hybrid payloads include AllocationProblemData primitives; and all payloads
    include MonthlyProblemBuild audit arrays plus their source-bundle identity.
    """

    return _encode_private_array_container(
        payload_kind="stage11_strategy_problem",
        arrays=problem_arrays,
        metadata=problem_metadata,
    )


def decode_stage11_private_array_container(
    payload: bytes,
    *,
    expected_payload_kind: str,
) -> DecodedPrivateArrayContainer:
    """Decode and independently authenticate a private array container."""

    _require(type(payload) is bytes, "private container must be bytes")
    _require(
        payload.startswith(_PRIVATE_CONTAINER_MAGIC),
        "private container magic differs",
    )
    header_length_offset = len(_PRIVATE_CONTAINER_MAGIC)
    _require(
        len(payload) >= header_length_offset + 8,
        "private container header is truncated",
    )
    header_length = struct.unpack(
        ">Q",
        payload[header_length_offset : header_length_offset + 8],
    )[0]
    header_start = header_length_offset + 8
    header_end = header_start + header_length
    _require(
        header_length > 0 and header_end <= len(payload),
        "private container header length differs",
    )
    header = _parse_exact_json(
        payload[header_start:header_end],
        fields=_PRIVATE_CONTAINER_HEADER_FIELDS,
        label="private array container header",
    )
    _require(header["schema_version"] == 1, "private container schema differs")
    _require(
        header["payload_kind"] == expected_payload_kind,
        "private container payload kind differs",
    )
    _require(isinstance(header["metadata"], dict), "private metadata must be an object")
    raw_records = header["arrays"]
    _require(isinstance(raw_records, list), "private array records must be a list")
    names: list[str] = []
    expected_offset = 0
    arrays: dict[str, np.ndarray] = {}
    data = payload[header_end:]
    for raw_record in raw_records:
        _require(isinstance(raw_record, dict), "private array record must be an object")
        _require(
            frozenset(raw_record) == _PRIVATE_ARRAY_RECORD_FIELDS,
            "private array record field set differs",
        )
        name = raw_record["name"]
        _require(
            isinstance(name, str)
            and _SAFE_PRIVATE_ARRAY_NAME_RE.fullmatch(name) is not None,
            "private array name differs",
        )
        _require(name not in names, "duplicate private array name")
        names.append(name)
        _require(
            raw_record["offset"] == expected_offset,
            "private array offsets are not contiguous",
        )
        size = raw_record["size_bytes"]
        _require(type(size) is int and size > 0, "private array size differs")
        end = expected_offset + size
        _require(end <= len(data), "private array payload is truncated")
        array_bytes = data[expected_offset:end]
        _require(
            _sha256_bytes(array_bytes) == raw_record["sha256"],
            "private array payload hash differs",
        )
        try:
            stream = io.BytesIO(array_bytes)
            value = np.load(stream, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise Stage11DiagnosticPersistenceError(
                f"private array {name} cannot be decoded"
            ) from exc
        _require(
            stream.tell() == len(array_bytes),
            f"private array {name} contains trailing bytes",
        )
        _require(
            value.dtype.str == raw_record["dtype"],
            f"private array {name} dtype differs",
        )
        _require(
            list(value.shape) == raw_record["shape"],
            f"private array {name} shape differs",
        )
        value.setflags(write=False)
        arrays[name] = value
        expected_offset = end
    _require(names == sorted(names), "private array order is not canonical")
    _require(expected_offset == len(data), "private container has trailing bytes")
    return DecodedPrivateArrayContainer(
        payload_kind=expected_payload_kind,
        arrays=MappingProxyType(arrays),
        metadata=_deep_freeze_json(header["metadata"]),
    )


def _assert_no_post_stage11_dates(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_no_post_stage11_dates(nested, label=f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_no_post_stage11_dates(nested, label=f"{label}[{index}]")
        return
    if isinstance(value, str) and _ISO_DATE_RE.fullmatch(value) is not None:
        try:
            observed = dt.date.fromisoformat(value)
        except ValueError as exc:
            raise Stage11DiagnosticPersistenceError(
                f"{label} contains an invalid ISO date"
            ) from exc
        _require(
            observed <= LAST_VALIDATION_EXIT_DATE,
            f"{label} crosses the Stage 11 date boundary",
        )


def _validate_strategy_problem_payload(
    *,
    payload: bytes,
    holding_month: str,
    strategy_id: str,
) -> None:
    decoded = decode_stage11_private_array_container(
        payload,
        expected_payload_kind="stage11_strategy_problem",
    )
    _require(
        decoded.metadata.get("holding_month") == holding_month,
        "strategy problem holding month differs",
    )
    _require(
        decoded.metadata.get("strategy_id") == strategy_id,
        "strategy problem identity differs",
    )
    _require_sha256(
        decoded.metadata.get("source_bundle_sha256"),
        "strategy problem source_bundle_sha256",
    )
    _require(
        "allocation.permno" in decoded.arrays,
        "strategy problem omits allocation.permno",
    )
    _assert_no_post_stage11_dates(decoded.metadata, label="strategy problem metadata")
    for name, array in decoded.arrays.items():
        if array.dtype.kind in {"U", "S"}:
            for index, value in enumerate(np.asarray(array).reshape(-1)):
                _assert_no_post_stage11_dates(
                    str(value),
                    label=f"strategy problem array {name}[{index}]",
                )


class Stage11DiagnosticSession:
    """Backend-facing facade over the authenticated diagnostic protocol."""

    def __init__(
        self,
        *,
        root: Path,
        compatibility: Stage11DiagnosticCompatibility,
        producer_implementation_sha256: str,
        calendar_holding_months: Sequence[str] = VALIDATION_HOLDING_MONTHS,
    ) -> None:
        _require(
            tuple(calendar_holding_months) == VALIDATION_HOLDING_MONTHS,
            "diagnostic session requires the exact 59-month Stage 11 calendar",
        )
        self.root = initialize_stage11_diagnostic_root(root)
        self.compatibility = compatibility
        self.producer_implementation_sha256 = _require_sha256(
            producer_implementation_sha256,
            "producer_implementation_sha256",
        )
        self._authenticated_chain_cache: (
            tuple[AuthenticatedDiagnosticCheckpoint, ...] | None
        ) = None

    def _authenticated_chain(
        self,
    ) -> tuple[AuthenticatedDiagnosticCheckpoint, ...]:
        if self._authenticated_chain_cache is None:
            self._authenticated_chain_cache = (
                load_authenticated_checkpoint_chain(
                    root=self.root,
                    compatibility=self.compatibility,
                )
            )
        return self._authenticated_chain_cache

    def load_latest_checkpoint(self) -> DiagnosticResumePoint:
        chain = self._authenticated_chain()
        checkpoint = None if not chain else chain[-1]
        if checkpoint is None:
            return DiagnosticResumePoint(
                next_month_index=0,
                next_holding_month=VALIDATION_HOLDING_MONTHS[0],
                completed_checkpoint=None,
                state_arrays=MappingProxyType({}),
                state_metadata=MappingProxyType({}),
            )
        decoded = decode_stage11_private_array_container(
            checkpoint.private_state,
            expected_payload_kind="stage11_checkpoint_state",
        )
        next_index = checkpoint.ordinal
        return DiagnosticResumePoint(
            next_month_index=next_index,
            next_holding_month=(
                None
                if next_index == VALIDATION_HOLDING_MONTH_COUNT
                else VALIDATION_HOLDING_MONTHS[next_index]
            ),
            completed_checkpoint=checkpoint,
            state_arrays=decoded.arrays,
            state_metadata=decoded.metadata,
        )

    def save_problem_bundle(
        self,
        *,
        holding_month: str,
        strategy_problem_payloads: Mapping[str, bytes],
    ) -> MonthlyProblemBundleReceipt:
        resume = self.load_latest_checkpoint()
        _require(
            resume.next_holding_month == holding_month,
            "problem month is not the authenticated resume month",
        )
        opening = (
            None
            if resume.completed_checkpoint is None
            else resume.completed_checkpoint.checkpoint_sha256
        )
        return publish_monthly_problem_bundle(
            root=self.root,
            holding_month=holding_month,
            compatibility=self.compatibility,
            producer_implementation_sha256=self.producer_implementation_sha256,
            opening_checkpoint_sha256=opening,
            strategy_problem_payloads=strategy_problem_payloads,
        )

    def save_checkpoint(
        self,
        *,
        completed_month: str,
        next_month_index: int,
        state_arrays: Mapping[str, np.ndarray],
        state_metadata: Mapping[str, Any],
        problem_bundle_sha256: str,
        post_month_attestation: Mapping[str, object],
    ) -> AuthenticatedDiagnosticCheckpoint:
        ordinal = _month_ordinal(completed_month)
        _require(
            next_month_index == ordinal,
            "next_month_index must equal the completed calendar-prefix length",
        )
        resume = self.load_latest_checkpoint()
        _require(
            resume.next_month_index == ordinal - 1
            and resume.next_holding_month == completed_month,
            "checkpoint month is not the authenticated resume month",
        )
        opening = (
            None
            if resume.completed_checkpoint is None
            else resume.completed_checkpoint.checkpoint_sha256
        )
        private_state = encode_stage11_checkpoint_state(
            state_arrays=state_arrays,
            state_metadata=state_metadata,
        )
        checkpoint = publish_post_month_checkpoint(
            root=self.root,
            holding_month=completed_month,
            compatibility=self.compatibility,
            producer_implementation_sha256=self.producer_implementation_sha256,
            expected_previous_checkpoint_sha256=opening,
            problem_bundle_sha256=problem_bundle_sha256,
            private_state=private_state,
            post_month_attestation=post_month_attestation,
            _authenticated_chain=self._authenticated_chain(),
        )
        self._authenticated_chain_cache = (
            *self._authenticated_chain(),
            checkpoint,
        )
        return checkpoint

    def record_failures(
        self,
        *,
        report: Stage11AggregateFailureReport,
    ) -> DiagnosticFailureStatus:
        digest = publish_aggregate_failure_report(
            root=self.root,
            compatibility=self.compatibility,
            producer_implementation_sha256=self.producer_implementation_sha256,
            report=report,
            _authenticated_chain=self._authenticated_chain(),
        )
        return DiagnosticFailureStatus(
            holding_month=report.holding_month,
            failure_report_sha256=digest,
            attempted_strategy_count=3,
            failure_count=len(report.failure_ids),
            checkpoint_advanced=False,
            performance_fields_emitted=0,
        )

    def seal_completion(
        self,
        *,
        regression_case_sha256_by_failure_id: Mapping[str, str],
        regression_suite_sha256: str,
        regression_suite_passed: bool,
    ) -> DiagnosticCompletionReceipt:
        return publish_diagnostic_completion_seal(
            root=self.root,
            compatibility=self.compatibility,
            producer_implementation_sha256=self.producer_implementation_sha256,
            regression_case_sha256_by_failure_id=(
                regression_case_sha256_by_failure_id
            ),
            regression_suite_sha256=regression_suite_sha256,
            regression_suite_passed=regression_suite_passed,
        )


def _existing_components_are_not_symlinks(path: Path) -> None:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    cursor = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        cursor /= component
        try:
            observed = os.lstat(cursor)
        except FileNotFoundError:
            break
        _require(
            not stat.S_ISLNK(observed.st_mode),
            f"symlinked diagnostic path is prohibited: {cursor}",
        )


def _ensure_private_directory(path: Path, *, create: bool) -> Path:
    _require(isinstance(path, Path) and path.is_absolute(), "path must be absolute")
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    _existing_components_are_not_symlinks(lexical)
    if create and not lexical.exists():
        _require(lexical.parent.is_dir(), f"parent directory is missing: {lexical.parent}")
        lexical.mkdir(mode=0o700)
    _require(lexical.exists(), f"diagnostic directory is missing: {lexical}")
    observed = os.lstat(lexical)
    _require(
        stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
        f"diagnostic path is not a real directory: {lexical}",
    )
    _require(
        stat.S_IMODE(observed.st_mode) == 0o700,
        f"diagnostic directory must have mode 0700: {lexical}",
    )
    _require(
        observed.st_uid == os.getuid(),
        f"diagnostic directory is not owned by the current user: {lexical}",
    )
    return lexical


def _ensure_private_parent_directory(path: Path) -> Path:
    """Create one missing private parent below an authenticated anchor."""

    _require(isinstance(path, Path) and path.is_absolute(), "path must be absolute")
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    _existing_components_are_not_symlinks(lexical)
    if lexical.exists():
        observed = os.lstat(lexical)
        _require(
            stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
            f"diagnostic parent is not a real directory: {lexical}",
        )
        _require(
            observed.st_uid == os.getuid(),
            f"diagnostic parent is not owned by the current user: {lexical}",
        )
        _require(
            stat.S_IMODE(observed.st_mode) & 0o022 == 0,
            f"diagnostic parent is group/world writable: {lexical}",
        )
        return lexical

    anchor = lexical.parent
    _existing_components_are_not_symlinks(anchor)
    _require(
        anchor.exists(),
        f"diagnostic parent anchor is missing: {anchor}",
    )
    observed = os.lstat(anchor)
    _require(
        stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
        f"diagnostic parent anchor is not a real directory: {anchor}",
    )
    _require(
        observed.st_uid == os.getuid(),
        f"diagnostic parent anchor is not owned by the current user: {anchor}",
    )
    _require(
        stat.S_IMODE(observed.st_mode) & 0o022 == 0,
        f"diagnostic parent anchor is group/world writable: {anchor}",
    )
    try:
        os.mkdir(lexical, mode=0o700)
    except FileExistsError:
        # A concurrent creator is accepted only if the independent checks
        # below prove the exact owner-only, non-symlinked directory.
        pass
    return _ensure_private_directory(lexical, create=False)


def initialize_stage11_diagnostic_root(root: Path) -> Path:
    """Create and authenticate the private directory layout."""

    _require(isinstance(root, Path) and root.is_absolute(), "path must be absolute")
    lexical = Path(os.path.abspath(os.fspath(root.expanduser())))
    _ensure_private_parent_directory(lexical.parent)
    root = _ensure_private_directory(root, create=True)
    for name in (
        "problem_bundles",
        "checkpoints",
        "failure_reports",
        "completion_seals",
    ):
        child = root / name
        if not child.exists():
            child.mkdir(mode=0o700)
        _ensure_private_directory(child, create=False)
    return root


def _file_records(payloads: Mapping[str, bytes]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": name,
            "sha256": _sha256_bytes(payloads[name]),
            "size_bytes": len(payloads[name]),
        }
        for name in sorted(payloads)
    )


def _bundle_sha256(payloads: Mapping[str, bytes]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "digest_protocol": "stage11-closed-bundle-sha256-v1",
                "files": _file_records(payloads),
            }
        )
    )


def _publish_closed_bundle(
    *,
    root: Path,
    destination: Path,
    payloads: Mapping[str, bytes],
    label: str,
) -> str:
    expected_hashes = {
        name: _sha256_bytes(payload) for name, payload in payloads.items()
    }
    digest = _bundle_sha256(payloads)
    _require(
        destination.name.endswith(digest),
        f"{label} destination is not content addressed",
    )
    try:
        with SecureStagingDirectory.create(
            trusted_root=root,
            destination=destination,
            expected_files=tuple(payloads),
            label=label,
        ) as stage:
            for name in sorted(payloads):
                with stage.open_exclusive_binary(name) as stream:
                    stream.write(payloads[name])
            snapshot = stage.snapshot(expected_hashes=expected_hashes)
            _require(
                dict(snapshot.hashes) == expected_hashes,
                f"{label} staged hashes differ",
            )
            stage.publish(expected_hashes=expected_hashes)
    except SecurePublicationError as exc:
        raise Stage11DiagnosticPersistenceError(
            f"cannot atomically publish {label}"
        ) from exc
    return digest


def _diagnostic_file_identity(
    value: os.stat_result,
    *,
    include_ctime: bool,
) -> tuple[int, ...]:
    fields = [
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
    ]
    if include_ctime:
        fields.append("st_ctime_ns")
    return tuple(int(getattr(value, field)) for field in fields)


def _read_regular_file_pass(
    path: Path,
) -> tuple[
    bytes,
    tuple[os.stat_result, os.stat_result, os.stat_result, os.stat_result],
]:
    before = os.lstat(path)
    _require(
        not stat.S_ISLNK(before.st_mode),
        f"symlinked diagnostic member is prohibited: {path}",
    )
    _require(
        stat.S_ISREG(before.st_mode),
        f"diagnostic member is not a regular file: {path}",
    )
    _require(before.st_nlink == 1, f"hard-linked diagnostic member: {path}")
    _require(
        stat.S_IMODE(before.st_mode) == 0o600,
        f"diagnostic member must have mode 0600: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(path)
    identities = (before, opened, after_fd, after_path)
    return b"".join(chunks), identities


def _read_regular_file_stable(path: Path) -> bytes:
    first_payload, first_stats = _read_regular_file_pass(path)
    first_full_identities = tuple(
        _diagnostic_file_identity(value, include_ctime=True)
        for value in first_stats
    )
    if len(set(first_full_identities)) == 1:
        return first_payload

    first_non_ctime_identities = tuple(
        _diagnostic_file_identity(value, include_ctime=False)
        for value in first_stats
    )
    _require(
        len(set(first_non_ctime_identities)) == 1
        and len({int(value.st_ctime_ns) for value in first_stats}) > 1,
        f"diagnostic member changed while read: {path}",
    )

    second_payload, second_stats = _read_regular_file_pass(path)
    second_full_identities = tuple(
        _diagnostic_file_identity(value, include_ctime=True)
        for value in second_stats
    )
    second_non_ctime_identities = tuple(
        _diagnostic_file_identity(value, include_ctime=False)
        for value in second_stats
    )
    all_non_ctime_identities = (
        *first_non_ctime_identities,
        *second_non_ctime_identities,
    )
    _require(
        len(set(all_non_ctime_identities)) == 1,
        f"diagnostic member identity changed during ctime stabilization: {path}",
    )
    _require(
        second_payload == first_payload,
        f"diagnostic member bytes changed during ctime stabilization: {path}",
    )
    _require(
        first_full_identities[-1] == second_full_identities[0]
        and len(set(second_full_identities)) == 1,
        f"diagnostic member ctime did not stabilize: {path}",
    )
    return second_payload


def _read_authenticated_closed_bundle(
    *,
    root: Path,
    path: Path,
    expected_members: Sequence[str],
    expected_digest: str,
    label: str,
) -> Mapping[str, bytes]:
    _require_sha256(expected_digest, f"{label} expected digest")
    _require(path.parent.parent == root, f"{label} path escaped diagnostic root")
    _existing_components_are_not_symlinks(path)
    observed_dir = os.lstat(path)
    _require(
        stat.S_ISDIR(observed_dir.st_mode) and not stat.S_ISLNK(observed_dir.st_mode),
        f"{label} is not a real directory",
    )
    _require(
        stat.S_IMODE(observed_dir.st_mode) == 0o700,
        f"{label} directory must have mode 0700",
    )
    names = tuple(sorted(os.listdir(path)))
    expected = tuple(sorted(expected_members))
    _require(names == expected, f"{label} closed member set differs")
    first_payloads = {name: _read_regular_file_stable(path / name) for name in names}
    first_hashes = {
        name: _sha256_bytes(payload) for name, payload in first_payloads.items()
    }
    _require(
        _bundle_sha256(first_payloads) == expected_digest,
        f"{label} content-address digest mismatch",
    )
    try:
        with AuthenticatedBundleGuard.open(
            trusted_root=root,
            path=path,
            expected_files=expected,
            expected_hashes=first_hashes,
            label=label,
            _permit_authenticated_directory_ctime_refresh=True,
        ) as guard:
            second_payloads = {
                name: _read_regular_file_stable(path / name) for name in names
            }
            _require(
                second_payloads == first_payloads,
                f"{label} bytes changed during authenticated reopen",
            )
            guard.revalidate()
    except SecurePublicationError as exc:
        raise Stage11DiagnosticPersistenceError(
            f"{label} failed authenticated reopen"
        ) from exc
    return MappingProxyType(second_payloads)


def _safe_category_entries(category: Path) -> tuple[Path, ...]:
    result: list[Path] = []
    for name in sorted(os.listdir(category)):
        path = category / name
        observed = os.lstat(path)
        _require(
            not stat.S_ISLNK(observed.st_mode),
            f"symlink in diagnostic category is prohibited: {path}",
        )
        # Dot-prefixed stages and retired entries are never authority.  They
        # remain recovery evidence owned by the secure publication primitive.
        if name.startswith("."):
            _require(
                stat.S_ISDIR(observed.st_mode),
                f"unsafe recovery entry in diagnostic category: {path}",
            )
            continue
        _require(
            stat.S_ISDIR(observed.st_mode),
            f"non-directory diagnostic bundle entry: {path}",
        )
        result.append(path)
    return tuple(result)


def _compatibility_from_manifest(
    manifest: Mapping[str, Any],
    expected: Stage11DiagnosticCompatibility,
) -> None:
    _require(
        manifest["compatibility"] == expected.as_dict(),
        "diagnostic compatibility identities differ",
    )
    _require(
        manifest["compatibility_sha256"] == expected.sha256,
        "diagnostic compatibility digest differs",
    )


def _problem_directory_name(
    ordinal: int, holding_month: str, digest: str
) -> str:
    return f"{ordinal:03d}-{holding_month}-problems-{digest}"


def _checkpoint_directory_name(
    ordinal: int, holding_month: str, digest: str
) -> str:
    return f"{ordinal:03d}-{holding_month}-checkpoint-{digest}"


def _failure_directory_name(
    ordinal: int, holding_month: str, digest: str
) -> str:
    return f"{ordinal:03d}-{holding_month}-failures-{digest}"


def publish_monthly_problem_bundle(
    *,
    root: Path,
    holding_month: str,
    compatibility: Stage11DiagnosticCompatibility,
    producer_implementation_sha256: str,
    opening_checkpoint_sha256: str | None,
    strategy_problem_payloads: Mapping[str, bytes],
) -> MonthlyProblemBundleReceipt:
    """Atomically save all three allocation problems before solver entry."""

    root = initialize_stage11_diagnostic_root(root)
    ordinal = _month_ordinal(holding_month)
    producer = _require_sha256(
        producer_implementation_sha256,
        "producer_implementation_sha256",
    )
    if opening_checkpoint_sha256 is None:
        _require(ordinal == 1, "only January 2015 may have no opening checkpoint")
    else:
        _require_sha256(opening_checkpoint_sha256, "opening_checkpoint_sha256")
        _require(ordinal > 1, "January 2015 opening checkpoint must be null")
    _require(
        tuple(strategy_problem_payloads) == STAGE11_DIAGNOSTIC_STRATEGY_IDS,
        "all three strategy problem payloads are required in locked order",
    )
    for strategy_id, payload in strategy_problem_payloads.items():
        _require(
            type(payload) is bytes and bool(payload),
            f"{strategy_id} problem payload must be nonempty bytes",
        )
        _validate_strategy_problem_payload(
            payload=payload,
            holding_month=holding_month,
            strategy_id=strategy_id,
        )

    payload_records = [
        {
            "strategy_id": strategy_id,
            "file_name": PROBLEM_FILE_BY_STRATEGY[strategy_id],
            "sha256": _sha256_bytes(strategy_problem_payloads[strategy_id]),
            "size_bytes": len(strategy_problem_payloads[strategy_id]),
        }
        for strategy_id in STAGE11_DIAGNOSTIC_STRATEGY_IDS
    ]
    manifest = {
        "schema_version": 1,
        "kind": PROBLEM_BUNDLE_KIND,
        "protocol_id": DIAGNOSTIC_PROTOCOL_ID,
        "holding_month": holding_month,
        "holding_month_ordinal": ordinal,
        "compatibility": compatibility.as_dict(),
        "compatibility_sha256": compatibility.sha256,
        "producer_implementation_sha256": producer,
        "opening_checkpoint_sha256": opening_checkpoint_sha256,
        "strategy_ids": list(STAGE11_DIAGNOSTIC_STRATEGY_IDS),
        "strategy_problem_files": payload_records,
        "saved_before_any_solver_attempt": True,
        "boundary": _boundary_payload(),
        "performance_fields_emitted": 0,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    success = {
        "schema_version": 1,
        "status": "ALL_THREE_PROBLEMS_SAVED_BEFORE_SOLVE",
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "strategy_count": len(STAGE11_DIAGNOSTIC_STRATEGY_IDS),
        "performance_fields_emitted": 0,
    }
    assert_stage11_diagnostic_public_payload_safe(success)
    payloads = {
        PROBLEM_FILE_BY_STRATEGY[strategy_id]: strategy_problem_payloads[strategy_id]
        for strategy_id in STAGE11_DIAGNOSTIC_STRATEGY_IDS
    }
    payloads[PROBLEM_MANIFEST_NAME] = manifest_bytes
    payloads[SUCCESS_NAME] = _canonical_json_bytes(success)
    digest = _bundle_sha256(payloads)
    destination = (
        root
        / "problem_bundles"
        / _problem_directory_name(ordinal, holding_month, digest)
    )
    _publish_closed_bundle(
        root=root,
        destination=destination,
        payloads=payloads,
        label="Stage 11 monthly problem bundle",
    )
    return authenticate_monthly_problem_bundle(
        root=root,
        bundle_sha256=digest,
        compatibility=compatibility,
    )


def _find_unique_digest_bundle(
    *,
    category: Path,
    digest: str,
    pattern: re.Pattern[str],
    label: str,
) -> tuple[Path, re.Match[str]]:
    matches: list[tuple[Path, re.Match[str]]] = []
    for path in _safe_category_entries(category):
        parsed = pattern.fullmatch(path.name)
        _require(parsed is not None, f"unexpected {label} directory name: {path.name}")
        if parsed.group("digest") == digest:
            matches.append((path, parsed))
    _require(bool(matches), f"{label} digest is not present")
    _require(len(matches) == 1, f"{label} digest is ambiguous")
    return matches[0]


def authenticate_monthly_problem_bundle(
    *,
    root: Path,
    bundle_sha256: str,
    compatibility: Stage11DiagnosticCompatibility,
) -> MonthlyProblemBundleReceipt:
    """Authenticate and reopen one content-addressed pre-solve bundle."""

    root = initialize_stage11_diagnostic_root(root)
    digest = _require_sha256(bundle_sha256, "bundle_sha256")
    path, parsed = _find_unique_digest_bundle(
        category=root / "problem_bundles",
        digest=digest,
        pattern=_PROBLEM_DIRECTORY_RE,
        label="problem bundle",
    )
    ordinal = int(parsed.group("ordinal"))
    holding_month = parsed.group("month")
    _require(ordinal == _month_ordinal(holding_month), "problem directory ordinal differs")
    payloads = _read_authenticated_closed_bundle(
        root=root,
        path=path,
        expected_members=PROBLEM_MEMBER_NAMES,
        expected_digest=digest,
        label="Stage 11 monthly problem bundle",
    )
    manifest = _parse_exact_json(
        payloads[PROBLEM_MANIFEST_NAME],
        fields=_PROBLEM_MANIFEST_FIELDS,
        label="problem manifest",
    )
    _require(manifest["schema_version"] == 1, "problem schema version differs")
    _require(manifest["kind"] == PROBLEM_BUNDLE_KIND, "problem bundle kind differs")
    _require(manifest["protocol_id"] == DIAGNOSTIC_PROTOCOL_ID, "protocol differs")
    _require(manifest["holding_month"] == holding_month, "problem month differs")
    _require(manifest["holding_month_ordinal"] == ordinal, "problem ordinal differs")
    _compatibility_from_manifest(manifest, compatibility)
    producer = _require_sha256(
        manifest["producer_implementation_sha256"],
        "producer_implementation_sha256",
    )
    opening = manifest["opening_checkpoint_sha256"]
    if opening is not None:
        _require_sha256(opening, "opening_checkpoint_sha256")
    _require(
        manifest["strategy_ids"] == list(STAGE11_DIAGNOSTIC_STRATEGY_IDS),
        "problem strategy ids differ",
    )
    _require(
        manifest["saved_before_any_solver_attempt"] is True,
        "problem bundle lacks pre-solve attestation",
    )
    _validate_boundary(manifest["boundary"])
    _require(
        manifest["performance_fields_emitted"] == 0,
        "problem manifest exposes performance",
    )
    expected_records = [
        {
            "strategy_id": strategy_id,
            "file_name": PROBLEM_FILE_BY_STRATEGY[strategy_id],
            "sha256": _sha256_bytes(payloads[PROBLEM_FILE_BY_STRATEGY[strategy_id]]),
            "size_bytes": len(payloads[PROBLEM_FILE_BY_STRATEGY[strategy_id]]),
        }
        for strategy_id in STAGE11_DIAGNOSTIC_STRATEGY_IDS
    ]
    _require(
        manifest["strategy_problem_files"] == expected_records,
        "problem payload records differ",
    )
    for strategy_id in STAGE11_DIAGNOSTIC_STRATEGY_IDS:
        _validate_strategy_problem_payload(
            payload=payloads[PROBLEM_FILE_BY_STRATEGY[strategy_id]],
            holding_month=holding_month,
            strategy_id=strategy_id,
        )
    success = _parse_exact_json(
        payloads[SUCCESS_NAME],
        fields=frozenset(
            {
                "schema_version",
                "status",
                "manifest_sha256",
                "strategy_count",
                "performance_fields_emitted",
            }
        ),
        label="problem success receipt",
    )
    _require(
        success
        == {
            "schema_version": 1,
            "status": "ALL_THREE_PROBLEMS_SAVED_BEFORE_SOLVE",
            "manifest_sha256": _sha256_bytes(payloads[PROBLEM_MANIFEST_NAME]),
            "strategy_count": 3,
            "performance_fields_emitted": 0,
        },
        "problem success receipt differs",
    )
    assert_stage11_diagnostic_public_payload_safe(success)
    strategy_payloads = MappingProxyType(
        {
            strategy_id: payloads[PROBLEM_FILE_BY_STRATEGY[strategy_id]]
            for strategy_id in STAGE11_DIAGNOSTIC_STRATEGY_IDS
        }
    )
    return MonthlyProblemBundleReceipt(
        holding_month=holding_month,
        ordinal=ordinal,
        bundle_sha256=digest,
        path=path,
        opening_checkpoint_sha256=opening,
        compatibility_sha256=compatibility.sha256,
        producer_implementation_sha256=producer,
        strategy_payloads=strategy_payloads,
    )


def _validate_post_month_attestation(value: Mapping[str, object]) -> dict[str, bool]:
    _require(
        frozenset(value) == frozenset(POST_MONTH_ATTESTATION_FIELDS),
        "post-month attestation field set differs",
    )
    result = {field: value[field] for field in POST_MONTH_ATTESTATION_FIELDS}
    _require(
        all(item is True for item in result.values()),
        "every post-month attestation must be exactly true",
    )
    return result  # type: ignore[return-value]


def _checkpoint_status(
    *, holding_month: str, ordinal: int, problem_bundle_sha256: str
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "protocol_id": DIAGNOSTIC_PROTOCOL_ID,
        "status": "MONTH_DIAGNOSTICALLY_COMPLETE",
        "holding_month": holding_month,
        "completed_month_count": ordinal,
        "next_holding_month": _next_holding_month(ordinal),
        "problem_bundle_sha256": problem_bundle_sha256,
        "checkpoint_state": "PRIVATE_NOT_EXPOSED",
        "performance_fields_emitted": 0,
    }
    assert_stage11_diagnostic_public_payload_safe(value)
    return value


def publish_post_month_checkpoint(
    *,
    root: Path,
    holding_month: str,
    compatibility: Stage11DiagnosticCompatibility,
    producer_implementation_sha256: str,
    expected_previous_checkpoint_sha256: str | None,
    problem_bundle_sha256: str,
    private_state: bytes,
    post_month_attestation: Mapping[str, object],
    _authenticated_chain: Sequence[AuthenticatedDiagnosticCheckpoint]
    | None = None,
) -> AuthenticatedDiagnosticCheckpoint:
    """Publish a private checkpoint after every required month-end gate passes."""

    root = initialize_stage11_diagnostic_root(root)
    ordinal = _month_ordinal(holding_month)
    producer = _require_sha256(
        producer_implementation_sha256,
        "producer_implementation_sha256",
    )
    _require(
        type(private_state) is bytes and bool(private_state),
        "private checkpoint state must be nonempty bytes",
    )
    attestation = _validate_post_month_attestation(post_month_attestation)
    if _authenticated_chain is None:
        chain = load_authenticated_checkpoint_chain(
            root=root,
            compatibility=compatibility,
        )
        use_cached_chain = False
    else:
        chain = _reauthenticate_cached_checkpoint_tip(
            root=root,
            compatibility=compatibility,
            chain=tuple(_authenticated_chain),
        )
        use_cached_chain = True
    latest = None if not chain else chain[-1]
    actual_previous = None if latest is None else latest.checkpoint_sha256
    _require(
        actual_previous == expected_previous_checkpoint_sha256,
        "checkpoint optimistic-concurrency identity differs",
    )
    _require(
        ordinal == (1 if latest is None else latest.ordinal + 1),
        "checkpoint month is not the next month in the authenticated chain",
    )
    problem = authenticate_monthly_problem_bundle(
        root=root,
        bundle_sha256=problem_bundle_sha256,
        compatibility=compatibility,
    )
    _require(problem.holding_month == holding_month, "problem month differs")
    _require(
        problem.opening_checkpoint_sha256 == actual_previous,
        "problem bundle was not built from the authenticated opening checkpoint",
    )

    status = _checkpoint_status(
        holding_month=holding_month,
        ordinal=ordinal,
        problem_bundle_sha256=problem.bundle_sha256,
    )
    status_bytes = _canonical_json_bytes(status)
    state_record = {
        "file_name": PRIVATE_STATE_NAME,
        "sha256": _sha256_bytes(private_state),
        "size_bytes": len(private_state),
        "visibility": "private_mode_0600",
        "serialization_owned_by": "stage11_portfolio_accounting_loop",
    }
    manifest = {
        "schema_version": 1,
        "kind": CHECKPOINT_BUNDLE_KIND,
        "protocol_id": DIAGNOSTIC_PROTOCOL_ID,
        "holding_month": holding_month,
        "holding_month_ordinal": ordinal,
        "completed_month_count": ordinal,
        "next_holding_month": _next_holding_month(ordinal),
        "compatibility": compatibility.as_dict(),
        "compatibility_sha256": compatibility.sha256,
        "producer_implementation_sha256": producer,
        "previous_checkpoint_sha256": actual_previous,
        "problem_bundle_sha256": problem.bundle_sha256,
        "private_state": state_record,
        "post_month_attestation": attestation,
        "boundary": _boundary_payload(),
        "performance_fields_emitted": 0,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    success = {
        "schema_version": 1,
        "status": "POST_MONTH_CHECKPOINT_COMMITTED",
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "status_sha256": _sha256_bytes(status_bytes),
        "private_state_sha256": _sha256_bytes(private_state),
        "completed_month_count": ordinal,
        "performance_fields_emitted": 0,
    }
    assert_stage11_diagnostic_public_payload_safe(success)
    payloads = {
        CHECKPOINT_MANIFEST_NAME: manifest_bytes,
        PRIVATE_STATE_NAME: private_state,
        PUBLIC_STATUS_NAME: status_bytes,
        SUCCESS_NAME: _canonical_json_bytes(success),
    }
    digest = _bundle_sha256(payloads)
    destination = (
        root
        / "checkpoints"
        / _checkpoint_directory_name(ordinal, holding_month, digest)
    )
    _publish_closed_bundle(
        root=root,
        destination=destination,
        payloads=payloads,
        label="Stage 11 post-month checkpoint",
    )
    if not use_cached_chain:
        reopened = load_latest_authenticated_checkpoint(
            root=root,
            compatibility=compatibility,
        )
        _require(
            reopened is not None and reopened.checkpoint_sha256 == digest,
            "new checkpoint did not become the authenticated chain tip",
        )
        return reopened
    parsed = _CHECKPOINT_DIRECTORY_RE.fullmatch(destination.name)
    _require(parsed is not None, "new checkpoint directory name differs")
    reopened = _authenticate_checkpoint_path(
        root=root,
        path=destination,
        parsed=parsed,
        compatibility=compatibility,
    )
    _require(
        reopened.checkpoint_sha256 == digest
        and reopened.previous_checkpoint_sha256 == actual_previous,
        "new checkpoint authenticated identity differs",
    )
    _assert_checkpoint_namespace_matches_chain(
        root=root,
        chain=(*chain, reopened),
    )
    return reopened


def _authenticate_checkpoint_path(
    *,
    root: Path,
    path: Path,
    parsed: re.Match[str],
    compatibility: Stage11DiagnosticCompatibility,
) -> AuthenticatedDiagnosticCheckpoint:
    ordinal = int(parsed.group("ordinal"))
    holding_month = parsed.group("month")
    digest = parsed.group("digest")
    _require(ordinal == _month_ordinal(holding_month), "checkpoint ordinal differs")
    payloads = _read_authenticated_closed_bundle(
        root=root,
        path=path,
        expected_members=CHECKPOINT_MEMBER_NAMES,
        expected_digest=digest,
        label="Stage 11 post-month checkpoint",
    )
    manifest = _parse_exact_json(
        payloads[CHECKPOINT_MANIFEST_NAME],
        fields=_CHECKPOINT_MANIFEST_FIELDS,
        label="checkpoint manifest",
    )
    _require(manifest["schema_version"] == 1, "checkpoint schema differs")
    _require(manifest["kind"] == CHECKPOINT_BUNDLE_KIND, "checkpoint kind differs")
    _require(manifest["protocol_id"] == DIAGNOSTIC_PROTOCOL_ID, "protocol differs")
    _require(manifest["holding_month"] == holding_month, "checkpoint month differs")
    _require(manifest["holding_month_ordinal"] == ordinal, "checkpoint ordinal differs")
    _require(manifest["completed_month_count"] == ordinal, "completed count differs")
    _require(
        manifest["next_holding_month"] == _next_holding_month(ordinal),
        "next holding month differs",
    )
    _compatibility_from_manifest(manifest, compatibility)
    producer = _require_sha256(
        manifest["producer_implementation_sha256"],
        "producer_implementation_sha256",
    )
    previous = manifest["previous_checkpoint_sha256"]
    if previous is not None:
        _require_sha256(previous, "previous_checkpoint_sha256")
    problem_digest = _require_sha256(
        manifest["problem_bundle_sha256"],
        "problem_bundle_sha256",
    )
    expected_state_record = {
        "file_name": PRIVATE_STATE_NAME,
        "sha256": _sha256_bytes(payloads[PRIVATE_STATE_NAME]),
        "size_bytes": len(payloads[PRIVATE_STATE_NAME]),
        "visibility": "private_mode_0600",
        "serialization_owned_by": "stage11_portfolio_accounting_loop",
    }
    _require(
        manifest["private_state"] == expected_state_record,
        "private checkpoint state record differs",
    )
    _validate_post_month_attestation(manifest["post_month_attestation"])
    _validate_boundary(manifest["boundary"])
    _require(manifest["performance_fields_emitted"] == 0, "checkpoint leaks performance")
    status = _parse_exact_json(
        payloads[PUBLIC_STATUS_NAME],
        fields=frozenset(
            {
                "schema_version",
                "protocol_id",
                "status",
                "holding_month",
                "completed_month_count",
                "next_holding_month",
                "problem_bundle_sha256",
                "checkpoint_state",
                "performance_fields_emitted",
            }
        ),
        label="checkpoint public status",
    )
    _require(
        status
        == _checkpoint_status(
            holding_month=holding_month,
            ordinal=ordinal,
            problem_bundle_sha256=problem_digest,
        ),
        "checkpoint public status differs",
    )
    assert_stage11_diagnostic_public_payload_safe(status)
    success = _parse_exact_json(
        payloads[SUCCESS_NAME],
        fields=frozenset(
            {
                "schema_version",
                "status",
                "manifest_sha256",
                "status_sha256",
                "private_state_sha256",
                "completed_month_count",
                "performance_fields_emitted",
            }
        ),
        label="checkpoint success receipt",
    )
    _require(
        success
        == {
            "schema_version": 1,
            "status": "POST_MONTH_CHECKPOINT_COMMITTED",
            "manifest_sha256": _sha256_bytes(payloads[CHECKPOINT_MANIFEST_NAME]),
            "status_sha256": _sha256_bytes(payloads[PUBLIC_STATUS_NAME]),
            "private_state_sha256": _sha256_bytes(payloads[PRIVATE_STATE_NAME]),
            "completed_month_count": ordinal,
            "performance_fields_emitted": 0,
        },
        "checkpoint success receipt differs",
    )
    assert_stage11_diagnostic_public_payload_safe(success)
    return AuthenticatedDiagnosticCheckpoint(
        holding_month=holding_month,
        ordinal=ordinal,
        checkpoint_sha256=digest,
        previous_checkpoint_sha256=previous,
        problem_bundle_sha256=problem_digest,
        path=path,
        compatibility_sha256=compatibility.sha256,
        producer_implementation_sha256=producer,
        private_state=payloads[PRIVATE_STATE_NAME],
        public_status=MappingProxyType(status),
    )


def load_authenticated_checkpoint_chain(
    *,
    root: Path,
    compatibility: Stage11DiagnosticCompatibility,
) -> tuple[AuthenticatedDiagnosticCheckpoint, ...]:
    """Authenticate the unique contiguous chain; forks and gaps fail closed."""

    root = initialize_stage11_diagnostic_root(root)
    checkpoints: list[AuthenticatedDiagnosticCheckpoint] = []
    seen_ordinals: set[int] = set()
    for path in _safe_category_entries(root / "checkpoints"):
        parsed = _CHECKPOINT_DIRECTORY_RE.fullmatch(path.name)
        _require(parsed is not None, f"unexpected checkpoint directory: {path.name}")
        checkpoint = _authenticate_checkpoint_path(
            root=root,
            path=path,
            parsed=parsed,
            compatibility=compatibility,
        )
        _require(
            checkpoint.ordinal not in seen_ordinals,
            f"forked checkpoint chain at month {checkpoint.holding_month}",
        )
        seen_ordinals.add(checkpoint.ordinal)
        checkpoints.append(checkpoint)
    checkpoints.sort(key=lambda item: item.ordinal)
    expected_previous: str | None = None
    for expected_ordinal, checkpoint in enumerate(checkpoints, start=1):
        _require(
            checkpoint.ordinal == expected_ordinal,
            "checkpoint chain contains a gap or does not start at January 2015",
        )
        _require(
            checkpoint.previous_checkpoint_sha256 == expected_previous,
            "checkpoint hash chain differs",
        )
        problem = authenticate_monthly_problem_bundle(
            root=root,
            bundle_sha256=checkpoint.problem_bundle_sha256,
            compatibility=compatibility,
        )
        _require(problem.ordinal == checkpoint.ordinal, "checkpoint problem month differs")
        _require(
            problem.opening_checkpoint_sha256 == expected_previous,
            "checkpoint problem opening identity differs",
        )
        expected_previous = checkpoint.checkpoint_sha256
    return tuple(checkpoints)


def load_latest_authenticated_checkpoint(
    *,
    root: Path,
    compatibility: Stage11DiagnosticCompatibility,
) -> AuthenticatedDiagnosticCheckpoint | None:
    """Return the latest authenticated state without replaying earlier months."""

    chain = load_authenticated_checkpoint_chain(
        root=root,
        compatibility=compatibility,
    )
    return None if not chain else chain[-1]


def _assert_checkpoint_namespace_matches_chain(
    *,
    root: Path,
    chain: Sequence[AuthenticatedDiagnosticCheckpoint],
) -> None:
    """Reject forks or unexpected committed checkpoints without rereading state."""

    expected_names = tuple(sorted(checkpoint.path.name for checkpoint in chain))
    observed_names = tuple(
        sorted(path.name for path in _safe_category_entries(root / "checkpoints"))
    )
    _require(
        observed_names == expected_names,
        "checkpoint namespace differs from the authenticated chain",
    )


def _reauthenticate_cached_checkpoint_tip(
    *,
    root: Path,
    compatibility: Stage11DiagnosticCompatibility,
    chain: tuple[AuthenticatedDiagnosticCheckpoint, ...],
) -> tuple[AuthenticatedDiagnosticCheckpoint, ...]:
    """Reopen the cached tip and reject namespace forks before an append."""

    _assert_checkpoint_namespace_matches_chain(root=root, chain=chain)
    if not chain:
        return ()
    expected_previous: str | None = None
    for expected_ordinal, checkpoint in enumerate(chain, start=1):
        _require(
            checkpoint.ordinal == expected_ordinal
            and checkpoint.previous_checkpoint_sha256 == expected_previous,
            "cached checkpoint chain is not contiguous",
        )
        expected_previous = checkpoint.checkpoint_sha256
    tip = chain[-1]
    parsed = _CHECKPOINT_DIRECTORY_RE.fullmatch(tip.path.name)
    _require(parsed is not None, "cached checkpoint tip name differs")
    reopened = _authenticate_checkpoint_path(
        root=root,
        path=tip.path,
        parsed=parsed,
        compatibility=compatibility,
    )
    _require(
        reopened.checkpoint_sha256 == tip.checkpoint_sha256
        and reopened.previous_checkpoint_sha256
        == tip.previous_checkpoint_sha256
        and reopened.problem_bundle_sha256 == tip.problem_bundle_sha256
        and reopened.private_state == tip.private_state,
        "cached checkpoint tip changed before append",
    )
    return (*chain[:-1], reopened)


def _private_failure_details_bytes(
    report: Stage11AggregateFailureReport,
) -> bytes:
    details: list[dict[str, str]] = []
    for result in report.strategy_results:
        if result.status != "failed":
            continue
        _require(
            result.private_detail is not None,
            "new failure report requires the private failure detail",
        )
        details.append(
            {
                "strategy_id": result.strategy_id,
                "private_detail_hex": result.private_detail.hex(),
                "private_detail_sha256": str(result.private_detail_sha256),
            }
        )
    return _canonical_json_bytes(
        {
            "schema_version": 1,
            "visibility": "private_mode_0600",
            "details": details,
        }
    )


def publish_aggregate_failure_report(
    *,
    root: Path,
    compatibility: Stage11DiagnosticCompatibility,
    producer_implementation_sha256: str,
    report: Stage11AggregateFailureReport,
    _authenticated_chain: Sequence[AuthenticatedDiagnosticCheckpoint]
    | None = None,
) -> str:
    """Publish all three monthly strategy outcomes without performance values."""

    root = initialize_stage11_diagnostic_root(root)
    producer = _require_sha256(
        producer_implementation_sha256,
        "producer_implementation_sha256",
    )
    if _authenticated_chain is None:
        chain = load_authenticated_checkpoint_chain(
            root=root,
            compatibility=compatibility,
        )
    else:
        chain = _reauthenticate_cached_checkpoint_tip(
            root=root,
            compatibility=compatibility,
            chain=tuple(_authenticated_chain),
        )
    latest = None if not chain else chain[-1]
    opening = None if latest is None else latest.checkpoint_sha256
    expected_ordinal = 1 if latest is None else latest.ordinal + 1
    ordinal = _month_ordinal(report.holding_month)
    _require(
        ordinal == expected_ordinal,
        "failure report must describe the next uncompleted month",
    )
    problem = authenticate_monthly_problem_bundle(
        root=root,
        bundle_sha256=report.problem_bundle_sha256,
        compatibility=compatibility,
    )
    _require(problem.holding_month == report.holding_month, "failure problem month differs")
    _require(
        problem.opening_checkpoint_sha256 == opening,
        "failure problem opening checkpoint differs",
    )
    report_payload = report.as_public_dict()
    report_bytes = _canonical_json_bytes(report_payload)
    private_details_bytes = _private_failure_details_bytes(report)
    manifest = {
        "schema_version": 1,
        "kind": FAILURE_BUNDLE_KIND,
        "protocol_id": DIAGNOSTIC_PROTOCOL_ID,
        "holding_month": report.holding_month,
        "holding_month_ordinal": ordinal,
        "compatibility": compatibility.as_dict(),
        "compatibility_sha256": compatibility.sha256,
        "producer_implementation_sha256": producer,
        "opening_checkpoint_sha256": opening,
        "problem_bundle_sha256": problem.bundle_sha256,
        "failure_report": {
            "file_name": FAILURE_REPORT_NAME,
            "sha256": _sha256_bytes(report_bytes),
            "failure_ids": list(report.failure_ids),
            "failure_count": len(report.failure_ids),
        },
        "private_failure_details": {
            "file_name": PRIVATE_FAILURE_DETAILS_NAME,
            "sha256": _sha256_bytes(private_details_bytes),
            "size_bytes": len(private_details_bytes),
            "visibility": "private_mode_0600",
        },
        "all_active_strategies_attempted_and_inactive_receipted": True,
        "checkpoint_advanced": False,
        "boundary": _boundary_payload(),
        "performance_fields_emitted": 0,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    success = {
        "schema_version": 1,
        "status": "ALL_STRATEGY_FAILURES_REPORTED_TOGETHER",
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "failure_report_sha256": _sha256_bytes(report_bytes),
        "failure_count": len(report.failure_ids),
        "checkpoint_advanced": False,
        "performance_fields_emitted": 0,
    }
    assert_stage11_diagnostic_public_payload_safe(manifest)
    assert_stage11_diagnostic_public_payload_safe(report_payload)
    assert_stage11_diagnostic_public_payload_safe(success)
    payloads = {
        FAILURE_MANIFEST_NAME: manifest_bytes,
        FAILURE_REPORT_NAME: report_bytes,
        PRIVATE_FAILURE_DETAILS_NAME: private_details_bytes,
        SUCCESS_NAME: _canonical_json_bytes(success),
    }
    digest = _bundle_sha256(payloads)
    destination = (
        root
        / "failure_reports"
        / _failure_directory_name(ordinal, report.holding_month, digest)
    )
    _publish_closed_bundle(
        root=root,
        destination=destination,
        payloads=payloads,
        label="Stage 11 aggregate failure report",
    )
    _authenticate_failure_path(
        root=root,
        path=destination,
        parsed=_FAILURE_DIRECTORY_RE.fullmatch(destination.name),
        compatibility=compatibility,
    )
    return digest


def _authenticate_failure_path(
    *,
    root: Path,
    path: Path,
    parsed: re.Match[str] | None,
    compatibility: Stage11DiagnosticCompatibility,
) -> Stage11AggregateFailureReport:
    _require(parsed is not None, "failure report directory name differs")
    ordinal = int(parsed.group("ordinal"))
    holding_month = parsed.group("month")
    digest = parsed.group("digest")
    _require(ordinal == _month_ordinal(holding_month), "failure ordinal differs")
    payloads = _read_authenticated_closed_bundle(
        root=root,
        path=path,
        expected_members=FAILURE_MEMBER_NAMES,
        expected_digest=digest,
        label="Stage 11 aggregate failure report",
    )
    manifest = _parse_exact_json(
        payloads[FAILURE_MANIFEST_NAME],
        fields=_FAILURE_MANIFEST_FIELDS,
        label="failure manifest",
    )
    _require(manifest["schema_version"] == 1, "failure schema differs")
    _require(manifest["kind"] == FAILURE_BUNDLE_KIND, "failure kind differs")
    _require(manifest["protocol_id"] == DIAGNOSTIC_PROTOCOL_ID, "protocol differs")
    _require(manifest["holding_month"] == holding_month, "failure month differs")
    _require(manifest["holding_month_ordinal"] == ordinal, "failure ordinal differs")
    _compatibility_from_manifest(manifest, compatibility)
    _require_sha256(
        manifest["producer_implementation_sha256"],
        "producer_implementation_sha256",
    )
    opening = manifest["opening_checkpoint_sha256"]
    if opening is not None:
        _require_sha256(opening, "opening_checkpoint_sha256")
    problem_digest = _require_sha256(
        manifest["problem_bundle_sha256"],
        "problem_bundle_sha256",
    )
    _require(
        manifest[
            "all_active_strategies_attempted_and_inactive_receipted"
        ]
        is True,
        "attempt set differs",
    )
    _require(manifest["checkpoint_advanced"] is False, "failure advanced checkpoint")
    _validate_boundary(manifest["boundary"])
    _require(manifest["performance_fields_emitted"] == 0, "failure leaks performance")
    try:
        report_value = json.loads(payloads[FAILURE_REPORT_NAME].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11DiagnosticPersistenceError("failure report JSON is invalid") from exc
    _require(
        _canonical_json_bytes(report_value) == payloads[FAILURE_REPORT_NAME],
        "failure report is not canonical",
    )
    assert_stage11_diagnostic_public_payload_safe(report_value)
    _require(report_value["holding_month"] == holding_month, "failure report month differs")
    _require(
        report_value["problem_bundle_sha256"] == problem_digest,
        "failure report problem differs",
    )
    _require(
        report_value[
            "all_active_strategies_attempted_and_inactive_receipted"
        ]
        is True,
        "failure report did not attempt every active strategy or receipt inactivity",
    )
    try:
        private_value = json.loads(
            payloads[PRIVATE_FAILURE_DETAILS_NAME].decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11DiagnosticPersistenceError(
            "private failure details JSON is invalid"
        ) from exc
    _require(
        _canonical_json_bytes(private_value)
        == payloads[PRIVATE_FAILURE_DETAILS_NAME]
        and isinstance(private_value, dict)
        and set(private_value)
        == {"schema_version", "visibility", "details"}
        and private_value["schema_version"] == 1
        and private_value["visibility"] == "private_mode_0600"
        and isinstance(private_value["details"], list),
        "private failure details differ",
    )
    private_detail_by_strategy: dict[str, bytes] = {}
    for item in private_value["details"]:
        _require(
            isinstance(item, dict)
            and set(item)
            == {
                "strategy_id",
                "private_detail_hex",
                "private_detail_sha256",
            }
            and item["strategy_id"] not in private_detail_by_strategy,
            "private failure detail row differs",
        )
        try:
            detail = bytes.fromhex(item["private_detail_hex"])
        except (TypeError, ValueError) as exc:
            raise Stage11DiagnosticPersistenceError(
                "private failure detail encoding differs"
            ) from exc
        _require(
            bool(detail)
            and _sha256_bytes(detail) == item["private_detail_sha256"],
            "private failure detail digest differs",
        )
        private_detail_by_strategy[item["strategy_id"]] = detail
    raw_results = report_value["strategy_results"]
    _require(
        isinstance(raw_results, list) and len(raw_results) == 3,
        "failure strategy result count differs",
    )
    parsed_results: list[StrategyDiagnosticResult] = []
    for expected_strategy, value in zip(
        STAGE11_DIAGNOSTIC_STRATEGY_IDS,
        raw_results,
        strict=True,
    ):
        _require(
            isinstance(value, dict) and value.get("strategy_id") == expected_strategy,
            "failure strategy order differs",
        )
        if value.get("status") == "passed":
            _require(
                frozenset(value) == frozenset({"strategy_id", "status"}),
                "passed failure-report row has extra fields",
            )
            parsed_results.append(StrategyDiagnosticResult.passed(expected_strategy))
            continue
        if value.get("status") == "inactive":
            _require(
                frozenset(value)
                == frozenset({"strategy_id", "status", "inactive_reason"}),
                "inactive failure-report row field set differs",
            )
            parsed_results.append(
                StrategyDiagnosticResult.inactive(
                    expected_strategy,
                    inactive_reason=value["inactive_reason"],
                )
            )
            continue
        required = frozenset(
            {
                "strategy_id",
                "status",
                "holding_month",
                "problem_bundle_sha256",
                "failure_code",
                "exception_class",
                "private_detail_sha256",
                "failure_id",
            }
        )
        _require(
            frozenset(value) == required,
            "failed failure-report row field set differs",
        )
        candidate = StrategyDiagnosticResult(
            strategy_id=expected_strategy,
            status="failed",
            failure_code=value["failure_code"],
            exception_class=value["exception_class"],
            private_detail_sha256=value["private_detail_sha256"],
            private_detail=private_detail_by_strategy.get(expected_strategy),
        )
        expected_value = candidate.as_public_dict(
            holding_month=holding_month,
            problem_bundle_sha256=problem_digest,
        )
        _require(value == expected_value, "failure id or linkage differs")
        parsed_results.append(candidate)
    _require(
        set(private_detail_by_strategy)
        == {
            result.strategy_id
            for result in parsed_results
            if result.status == "failed"
        },
        "private failure detail strategy set differs",
    )
    report = Stage11AggregateFailureReport(
        holding_month=holding_month,
        problem_bundle_sha256=problem_digest,
        strategy_results=tuple(parsed_results),
    )
    _require(report.as_public_dict() == report_value, "failure report payload differs")
    expected_record = {
        "file_name": FAILURE_REPORT_NAME,
        "sha256": _sha256_bytes(payloads[FAILURE_REPORT_NAME]),
        "failure_ids": list(report.failure_ids),
        "failure_count": len(report.failure_ids),
    }
    _require(manifest["failure_report"] == expected_record, "failure record differs")
    _require(
        manifest["private_failure_details"]
        == {
            "file_name": PRIVATE_FAILURE_DETAILS_NAME,
            "sha256": _sha256_bytes(
                payloads[PRIVATE_FAILURE_DETAILS_NAME]
            ),
            "size_bytes": len(payloads[PRIVATE_FAILURE_DETAILS_NAME]),
            "visibility": "private_mode_0600",
        },
        "private failure detail manifest record differs",
    )
    success = json.loads(payloads[SUCCESS_NAME].decode("utf-8"))
    expected_success = {
        "schema_version": 1,
        "status": "ALL_STRATEGY_FAILURES_REPORTED_TOGETHER",
        "manifest_sha256": _sha256_bytes(payloads[FAILURE_MANIFEST_NAME]),
        "failure_report_sha256": _sha256_bytes(payloads[FAILURE_REPORT_NAME]),
        "failure_count": len(report.failure_ids),
        "checkpoint_advanced": False,
        "performance_fields_emitted": 0,
    }
    _require(
        success == expected_success
        and _canonical_json_bytes(success) == payloads[SUCCESS_NAME],
        "failure success receipt differs",
    )
    assert_stage11_diagnostic_public_payload_safe(success)
    return report


def load_authenticated_failure_reports(
    *,
    root: Path,
    compatibility: Stage11DiagnosticCompatibility,
) -> tuple[Stage11AggregateFailureReport, ...]:
    root = initialize_stage11_diagnostic_root(root)
    reports: list[Stage11AggregateFailureReport] = []
    for path in _safe_category_entries(root / "failure_reports"):
        parsed = _FAILURE_DIRECTORY_RE.fullmatch(path.name)
        _require(parsed is not None, f"unexpected failure directory: {path.name}")
        reports.append(
            _authenticate_failure_path(
                root=root,
                path=path,
                parsed=parsed,
                compatibility=compatibility,
            )
        )
    return tuple(
        sorted(
            reports,
            key=lambda report: (
                _month_ordinal(report.holding_month),
                report.problem_bundle_sha256,
                report.failure_ids,
            ),
        )
    )


def _checkpoint_chain_sha256(
    chain: Sequence[AuthenticatedDiagnosticCheckpoint],
) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            [
                {
                    "holding_month": checkpoint.holding_month,
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "previous_checkpoint_sha256": (
                        checkpoint.previous_checkpoint_sha256
                    ),
                    "problem_bundle_sha256": checkpoint.problem_bundle_sha256,
                }
                for checkpoint in chain
            ]
        )
    )


def publish_diagnostic_completion_seal(
    *,
    root: Path,
    compatibility: Stage11DiagnosticCompatibility,
    producer_implementation_sha256: str,
    regression_case_sha256_by_failure_id: Mapping[str, str],
    regression_suite_sha256: str,
    regression_suite_passed: bool,
) -> DiagnosticCompletionReceipt:
    """Seal diagnostics only after 59 months and every failure's regression."""

    root = initialize_stage11_diagnostic_root(root)
    producer = _require_sha256(
        producer_implementation_sha256,
        "producer_implementation_sha256",
    )
    regression_suite = _require_sha256(
        regression_suite_sha256,
        "regression_suite_sha256",
    )
    _require(regression_suite_passed is True, "regression suite must pass")
    chain = load_authenticated_checkpoint_chain(
        root=root,
        compatibility=compatibility,
    )
    _require(
        len(chain) == VALIDATION_HOLDING_MONTH_COUNT,
        "diagnostic completion requires exactly 59 checkpoints",
    )
    _require(
        tuple(item.holding_month for item in chain) == VALIDATION_HOLDING_MONTHS,
        "diagnostic completion month domain differs",
    )
    failure_reports = load_authenticated_failure_reports(
        root=root,
        compatibility=compatibility,
    )
    observed_failure_ids = tuple(
        sorted(
            {
                failure_id
                for report in failure_reports
                for failure_id in report.failure_ids
            }
        )
    )
    supplied_failure_ids = tuple(sorted(regression_case_sha256_by_failure_id))
    _require(
        supplied_failure_ids == observed_failure_ids,
        "every observed failure must map to exactly one permanent regression case",
    )
    regression_map: dict[str, str] = {}
    for failure_id in observed_failure_ids:
        _require_sha256(failure_id, "failure_id")
        regression_map[failure_id] = _require_sha256(
            regression_case_sha256_by_failure_id[failure_id],
            f"regression case for {failure_id}",
        )
    chain_sha256 = _checkpoint_chain_sha256(chain)
    latest = chain[-1]
    status = {
        "schema_version": 1,
        "protocol_id": DIAGNOSTIC_PROTOCOL_ID,
        "status": "DIAGNOSTIC_PATH_COMPLETE_READY_FOR_CLEAN_SEALED_RUN",
        "completed_month_count": VALIDATION_HOLDING_MONTH_COUNT,
        "first_holding_month": FIRST_VALIDATION_HOLDING_MONTH,
        "last_holding_month": LAST_VALIDATION_HOLDING_MONTH,
        "latest_checkpoint_sha256": latest.checkpoint_sha256,
        "checkpoint_chain_sha256": chain_sha256,
        "observed_failure_count": len(observed_failure_ids),
        "unresolved_failure_count": 0,
        "regression_suite_passed": True,
        "clean_sealed_run_started": False,
        "performance_fields_emitted": 0,
    }
    assert_stage11_diagnostic_public_payload_safe(status)
    status_bytes = _canonical_json_bytes(status)
    manifest = {
        "schema_version": 1,
        "kind": COMPLETION_BUNDLE_KIND,
        "protocol_id": DIAGNOSTIC_PROTOCOL_ID,
        "compatibility": compatibility.as_dict(),
        "compatibility_sha256": compatibility.sha256,
        "producer_implementation_sha256": producer,
        "completed_month_count": VALIDATION_HOLDING_MONTH_COUNT,
        "first_holding_month": FIRST_VALIDATION_HOLDING_MONTH,
        "last_holding_month": LAST_VALIDATION_HOLDING_MONTH,
        "latest_checkpoint_sha256": latest.checkpoint_sha256,
        "checkpoint_chain_sha256": chain_sha256,
        "observed_failure_ids": list(observed_failure_ids),
        "regression_case_sha256_by_failure_id": regression_map,
        "regression_suite_sha256": regression_suite,
        "regression_suite_passed": True,
        "unresolved_failure_count": 0,
        "boundary": _boundary_payload(),
        "performance_fields_emitted": 0,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    success = {
        "schema_version": 1,
        "status": "STAGE11_DIAGNOSTIC_COMPLETION_SEALED",
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "status_sha256": _sha256_bytes(status_bytes),
        "completed_month_count": VALIDATION_HOLDING_MONTH_COUNT,
        "unresolved_failure_count": 0,
        "performance_fields_emitted": 0,
    }
    assert_stage11_diagnostic_public_payload_safe(manifest)
    assert_stage11_diagnostic_public_payload_safe(success)
    payloads = {
        COMPLETION_MANIFEST_NAME: manifest_bytes,
        PUBLIC_STATUS_NAME: status_bytes,
        SUCCESS_NAME: _canonical_json_bytes(success),
    }
    digest = _bundle_sha256(payloads)
    destination = (
        root
        / "completion_seals"
        / f"stage11-diagnostic-complete-{digest}"
    )
    _publish_closed_bundle(
        root=root,
        destination=destination,
        payloads=payloads,
        label="Stage 11 diagnostic completion seal",
    )
    _authenticate_completion_seal(
        root=root,
        path=destination,
        compatibility=compatibility,
        expected_latest_checkpoint_sha256=latest.checkpoint_sha256,
        expected_checkpoint_chain_sha256=chain_sha256,
        expected_failure_ids=observed_failure_ids,
        expected_regression_map=regression_map,
        expected_regression_suite_sha256=regression_suite,
    )
    return DiagnosticCompletionReceipt(
        completion_sha256=digest,
        path=destination,
        producer_implementation_sha256=producer,
        latest_checkpoint_sha256=latest.checkpoint_sha256,
        checkpoint_chain_sha256=chain_sha256,
        completed_month_count=VALIDATION_HOLDING_MONTH_COUNT,
        observed_failure_ids=observed_failure_ids,
        regression_case_sha256_by_failure_id=MappingProxyType(regression_map),
        regression_suite_sha256=regression_suite,
    )


def load_authenticated_diagnostic_completion_seal(
    *,
    root: Path,
    compatibility: Stage11DiagnosticCompatibility,
) -> DiagnosticCompletionReceipt:
    """Load the unique completion seal against the current diagnostic evidence.

    A seal is authority only while it still describes the unique authenticated
    59-checkpoint chain and every currently observed diagnostic failure.  The
    regression mapping is read from the content-addressed manifest, validated
    as an exact one-to-one mapping over those failures, and then supplied to
    the independent seal authenticator.  Missing or multiple seals, a stale
    chain, a newly observed failure, malformed regression evidence, or any
    bundle-integrity problem fails closed.
    """

    root = initialize_stage11_diagnostic_root(root)
    completion_entries = _safe_category_entries(root / "completion_seals")
    _require(
        len(completion_entries) == 1,
        "exactly one diagnostic completion seal is required",
    )
    path = completion_entries[0]
    parsed = _COMPLETION_DIRECTORY_RE.fullmatch(path.name)
    _require(parsed is not None, "completion seal directory name differs")
    digest = parsed.group("digest")

    chain = load_authenticated_checkpoint_chain(
        root=root,
        compatibility=compatibility,
    )
    _require(
        len(chain) == VALIDATION_HOLDING_MONTH_COUNT,
        "diagnostic completion requires exactly 59 checkpoints",
    )
    _require(
        tuple(item.holding_month for item in chain) == VALIDATION_HOLDING_MONTHS,
        "diagnostic completion month domain differs",
    )
    latest = chain[-1]
    chain_sha256 = _checkpoint_chain_sha256(chain)

    failure_reports = load_authenticated_failure_reports(
        root=root,
        compatibility=compatibility,
    )
    observed_failure_ids = tuple(
        sorted(
            {
                failure_id
                for report in failure_reports
                for failure_id in report.failure_ids
            }
        )
    )

    # Read the manifest through the same closed-bundle guard before using any
    # of its values as expected identities for the full semantic authenticator.
    payloads = _read_authenticated_closed_bundle(
        root=root,
        path=path,
        expected_members=COMPLETION_MEMBER_NAMES,
        expected_digest=digest,
        label="Stage 11 diagnostic completion seal",
    )
    manifest = _parse_exact_json(
        payloads[COMPLETION_MANIFEST_NAME],
        fields=_COMPLETION_MANIFEST_FIELDS,
        label="completion manifest",
    )
    raw_regression_map = manifest["regression_case_sha256_by_failure_id"]
    _require(
        isinstance(raw_regression_map, dict),
        "completion regression mapping must be an object",
    )
    _require(
        tuple(sorted(raw_regression_map)) == observed_failure_ids,
        "completion regression mapping differs from observed failures",
    )
    regression_map: dict[str, str] = {}
    for failure_id in observed_failure_ids:
        _require_sha256(failure_id, "observed_failure_id")
        regression_map[failure_id] = _require_sha256(
            raw_regression_map[failure_id],
            f"regression case for {failure_id}",
        )
    regression_suite_sha256 = _require_sha256(
        manifest["regression_suite_sha256"],
        "regression_suite_sha256",
    )
    producer_implementation_sha256 = _require_sha256(
        manifest["producer_implementation_sha256"],
        "producer_implementation_sha256",
    )

    _authenticate_completion_seal(
        root=root,
        path=path,
        compatibility=compatibility,
        expected_latest_checkpoint_sha256=latest.checkpoint_sha256,
        expected_checkpoint_chain_sha256=chain_sha256,
        expected_failure_ids=observed_failure_ids,
        expected_regression_map=regression_map,
        expected_regression_suite_sha256=regression_suite_sha256,
    )
    # Reject a concurrent namespace fork introduced during authentication.
    _require(
        tuple(
            item.name
            for item in _safe_category_entries(root / "completion_seals")
        )
        == (path.name,),
        "diagnostic completion seal namespace changed during authentication",
    )
    return DiagnosticCompletionReceipt(
        completion_sha256=digest,
        path=path,
        producer_implementation_sha256=producer_implementation_sha256,
        latest_checkpoint_sha256=latest.checkpoint_sha256,
        checkpoint_chain_sha256=chain_sha256,
        completed_month_count=VALIDATION_HOLDING_MONTH_COUNT,
        observed_failure_ids=observed_failure_ids,
        regression_case_sha256_by_failure_id=MappingProxyType(regression_map),
        regression_suite_sha256=regression_suite_sha256,
    )


def _authenticate_completion_seal(
    *,
    root: Path,
    path: Path,
    compatibility: Stage11DiagnosticCompatibility,
    expected_latest_checkpoint_sha256: str,
    expected_checkpoint_chain_sha256: str,
    expected_failure_ids: tuple[str, ...],
    expected_regression_map: Mapping[str, str],
    expected_regression_suite_sha256: str,
) -> None:
    parsed = _COMPLETION_DIRECTORY_RE.fullmatch(path.name)
    _require(parsed is not None, "completion seal directory name differs")
    digest = parsed.group("digest")
    payloads = _read_authenticated_closed_bundle(
        root=root,
        path=path,
        expected_members=COMPLETION_MEMBER_NAMES,
        expected_digest=digest,
        label="Stage 11 diagnostic completion seal",
    )
    manifest = _parse_exact_json(
        payloads[COMPLETION_MANIFEST_NAME],
        fields=_COMPLETION_MANIFEST_FIELDS,
        label="completion manifest",
    )
    _require(manifest["schema_version"] == 1, "completion schema differs")
    _require(manifest["kind"] == COMPLETION_BUNDLE_KIND, "completion kind differs")
    _require(manifest["protocol_id"] == DIAGNOSTIC_PROTOCOL_ID, "protocol differs")
    _compatibility_from_manifest(manifest, compatibility)
    _require_sha256(
        manifest["producer_implementation_sha256"],
        "producer_implementation_sha256",
    )
    _require(
        manifest["completed_month_count"] == VALIDATION_HOLDING_MONTH_COUNT,
        "completion count differs",
    )
    _require(
        manifest["first_holding_month"] == FIRST_VALIDATION_HOLDING_MONTH
        and manifest["last_holding_month"] == LAST_VALIDATION_HOLDING_MONTH,
        "completion holding-month boundary differs",
    )
    _require(
        manifest["latest_checkpoint_sha256"]
        == expected_latest_checkpoint_sha256,
        "completion latest checkpoint differs",
    )
    _require(
        manifest["checkpoint_chain_sha256"]
        == expected_checkpoint_chain_sha256,
        "completion checkpoint chain differs",
    )
    _require(
        manifest["observed_failure_ids"] == list(expected_failure_ids),
        "completion observed failure ids differ",
    )
    _require(
        manifest["regression_case_sha256_by_failure_id"]
        == dict(expected_regression_map),
        "completion regression mapping differs",
    )
    _require(
        manifest["regression_suite_sha256"]
        == expected_regression_suite_sha256,
        "completion regression suite differs",
    )
    _require(manifest["regression_suite_passed"] is True, "regressions did not pass")
    _require(manifest["unresolved_failure_count"] == 0, "unresolved failures remain")
    _validate_boundary(manifest["boundary"])
    _require(manifest["performance_fields_emitted"] == 0, "completion leaks performance")
    status = _parse_exact_json(
        payloads[PUBLIC_STATUS_NAME],
        fields=frozenset(
            {
                "schema_version",
                "protocol_id",
                "status",
                "completed_month_count",
                "first_holding_month",
                "last_holding_month",
                "latest_checkpoint_sha256",
                "checkpoint_chain_sha256",
                "observed_failure_count",
                "unresolved_failure_count",
                "regression_suite_passed",
                "clean_sealed_run_started",
                "performance_fields_emitted",
            }
        ),
        label="completion public status",
    )
    success = _parse_exact_json(
        payloads[SUCCESS_NAME],
        fields=frozenset(
            {
                "schema_version",
                "status",
                "manifest_sha256",
                "status_sha256",
                "completed_month_count",
                "unresolved_failure_count",
                "performance_fields_emitted",
            }
        ),
        label="completion success receipt",
    )
    assert_stage11_diagnostic_public_payload_safe(status)
    assert_stage11_diagnostic_public_payload_safe(success)
    _require(
        status
        == {
            "schema_version": 1,
            "protocol_id": DIAGNOSTIC_PROTOCOL_ID,
            "status": "DIAGNOSTIC_PATH_COMPLETE_READY_FOR_CLEAN_SEALED_RUN",
            "completed_month_count": VALIDATION_HOLDING_MONTH_COUNT,
            "first_holding_month": FIRST_VALIDATION_HOLDING_MONTH,
            "last_holding_month": LAST_VALIDATION_HOLDING_MONTH,
            "latest_checkpoint_sha256": expected_latest_checkpoint_sha256,
            "checkpoint_chain_sha256": expected_checkpoint_chain_sha256,
            "observed_failure_count": len(expected_failure_ids),
            "unresolved_failure_count": 0,
            "regression_suite_passed": True,
            "clean_sealed_run_started": False,
            "performance_fields_emitted": 0,
        },
        "completion public status differs",
    )
    _require(
        success
        == {
            "schema_version": 1,
            "status": "STAGE11_DIAGNOSTIC_COMPLETION_SEALED",
            "manifest_sha256": _sha256_bytes(payloads[COMPLETION_MANIFEST_NAME]),
            "status_sha256": _sha256_bytes(payloads[PUBLIC_STATUS_NAME]),
            "completed_month_count": VALIDATION_HOLDING_MONTH_COUNT,
            "unresolved_failure_count": 0,
            "performance_fields_emitted": 0,
        },
        "completion success receipt differs",
    )


__all__ = [
    "AuthenticatedDiagnosticCheckpoint",
    "DecodedPrivateArrayContainer",
    "DiagnosticCompletionReceipt",
    "DiagnosticFailureStatus",
    "DiagnosticResumePoint",
    "DIAGNOSTIC_PROTOCOL_ID",
    "ELASTIC_NET_CHALLENGER_ID",
    "MonthlyProblemBundleReceipt",
    "POST_MONTH_ATTESTATION_FIELDS",
    "PRIVATE_FAILURE_DETAILS_NAME",
    "PRIMARY_STRATEGY_ID",
    "RANK_CHALLENGER_ID",
    "STAGE11_DIAGNOSTIC_STRATEGY_IDS",
    "Stage11AggregateFailureReport",
    "Stage11DiagnosticCompatibility",
    "Stage11DiagnosticPersistenceError",
    "Stage11DiagnosticSession",
    "StrategyDiagnosticResult",
    "assert_stage11_diagnostic_public_payload_safe",
    "authenticate_monthly_problem_bundle",
    "decode_stage11_private_array_container",
    "encode_stage11_checkpoint_state",
    "encode_stage11_strategy_problem",
    "initialize_stage11_diagnostic_root",
    "load_authenticated_checkpoint_chain",
    "load_authenticated_diagnostic_completion_seal",
    "load_authenticated_failure_reports",
    "load_latest_authenticated_checkpoint",
    "publish_aggregate_failure_report",
    "publish_diagnostic_completion_seal",
    "publish_monthly_problem_bundle",
    "publish_post_month_checkpoint",
]
