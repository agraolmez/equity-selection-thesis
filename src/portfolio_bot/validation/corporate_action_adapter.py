"""Fail-closed empirical adapter for Stage 11 delisting mechanics.

The Stage 7 delisting table contains vendor classifications, not an accounting
instruction.  In particular, the presence or absence of ``successor_permno``
does not by itself prove cash or stock consideration.  This module therefore
requires a separately reviewed, hash-frozen decision table.  Rules use only
declared code sets or explicit wildcards and every empirical row must match
exactly one rule.  Unknown codes, no match, or overlapping matches stop the
run before portfolio accounting.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pyarrow as pa

from portfolio_bot.validation.corporate_actions import (
    CASH_SETTLEMENT,
    CorporateAction,
)
from portfolio_bot.validation.outcome_materialization import DELISTING_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAPPING_MANIFEST_RELATIVE_PATH = (
    "configs/phase_05_validation/stage_11_corporate_action_mapping_v1.json"
)
FROZEN_MAPPING_MANIFEST_PATH = PROJECT_ROOT / MAPPING_MANIFEST_RELATIVE_PATH
MAPPING_MANIFEST_ID = "phase05-stage11-corporate-action-mapping-v1"
MAPPING_MANIFEST_STATUS = "FROZEN_BEFORE_VALIDATION_OUTCOMES"

# Comprehensive vendor domains were checked against all 8,391 authenticated
# 2000--2014 development delisting rows before validation outcomes were opened.
FROZEN_MAPPING_MANIFEST_SHA256: str | None = (
    "9ad6c56e6e3cfa15d24cf0090bafbe8a0fbc3efe1e7358e4e4cbecf86ca8aeb4"
)

VENDOR_MATCH_FIELDS = (
    "action_type",
    "status_type",
    "reason_type",
    "payment_type",
    "distribution_type",
    "treatment_status",
)
MATCH_FIELDS = (*VENDOR_MATCH_FIELDS, "successor_presence")
FIXED_INTERVAL_COLUMNS = (
    "signal_date",
    "holding_month",
    "entry_date",
    "exit_date",
)
DISPOSITIONS = frozenset({CASH_SETTLEMENT})
SUCCESSOR_POLICIES = frozenset({"ignored"})
UNRESOLVED_TERMINAL_TREATMENT = "missing_nonperformance_terminal_leg_unresolved"
UNRESOLVED_TERMINAL_TREATMENTS = frozenset(
    {
        UNRESOLVED_TERMINAL_TREATMENT,
        "missing_performance_outside_locked_exchange_unresolved",
    }
)
RESOLVED_TERMINAL_TREATMENTS = frozenset(
    {
        "observed_event_return_attached_to_delisting_date_once",
        "missing_performance_imputed_on_delisting_date_once",
    }
)
TREATMENT_STATUS_DOMAIN = (
    UNRESOLVED_TERMINAL_TREATMENT,
    "missing_performance_imputed_on_delisting_date_once",
    "observed_event_return_attached_to_delisting_date_once",
)
CRSP_PAYMENT_TYPES = (
    "CASH",
    "CNC",
    "COP",
    "CSHN",
    "CST",
    "CUS",
    "DW",
    "MAF",
    "MMI",
    "MUT",
    "NCOM",
    "NCOP",
    "OP",
    "PRCF",
    "SNC",
    "SOP",
    "STK",
    "UMAP",
    "UNAV",
    "USNC",
    "USOP",
    "USTK",
)
CRSP_ACTION_TYPES = (
    "GDR",
    "GEX",
    "GLI",
    "LOS",
    "MER",
)
CRSP_STATUS_TYPES = (
    "FPAY",
    "N/A",
    "NDC",
    "NDP",
    "NFC",
    "NFP",
    "NVP",
    "UNAV",
    "VCL",
)
CRSP_REASON_TYPES = (
    "BKPY",
    "CORQ",
    "DEEX",
    "DELQ",
    "DERE",
    "EQRQ",
    "FARG",
    "FDCV",
    "FING",
    "INSC",
    "INSF",
    "LP",
    "MTMK",
    "MVB",
    "MVCHI",
    "MVMF",
    "MVMO",
    "MVNM",
    "MVOT",
    "MVPAC",
    "MVPH",
    "MVTO",
    "N/A",
    "NACT",
    "OFFRE",
    "PUBI",
    "SERQ",
    "SHLD",
    "UNAV",
    "UNL",
    "VIO",
)
CRSP_DISTRIBUTION_TYPES = (
    "C1",
    "C2",
    "CS",
    "D1",
    "D2",
    "F1",
    "M2",
    "MU",
    "N1",
    "NA",
    "NO",
    "O1",
    "P1",
    "S1",
    "S2",
    "T1",
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ISSUANCE_TOKEN = object()


class CorporateActionAdapterError(RuntimeError):
    """Raised when empirical action classification is not fully determined."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorporateActionAdapterError(message)


def _canonical_json_bytes(value: object) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _row_value(value: object) -> object:
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, Decimal):
        # Arrow decimal128 values are exact source-record numerics.  Preserve
        # their scale and precision in the canonical source-row identity;
        # conversion through binary float would make the hash platform- and
        # representation-dependent.
        return format(value, "f")
    return value


def _vocabulary_sort_key(value: str | None) -> tuple[int, str]:
    return (0, "") if value is None else (1, value)


def _normalized_vocabulary(
    value: object,
    *,
    label: str,
) -> tuple[str | None, ...]:
    _require(isinstance(value, list) and bool(value), f"{label} must be a nonempty list")
    result = tuple(value)
    _require(
        all(item is None or isinstance(item, str) for item in result)
        and len(result) == len(set(result))
        and result == tuple(sorted(result, key=_vocabulary_sort_key)),
        f"{label} must be a canonical unique string/null vocabulary",
    )
    return result


@dataclass(frozen=True, slots=True)
class CorporateActionMappingRule:
    # ``None`` at the predicate level is a wildcard.  A frozenset containing
    # ``None`` explicitly matches a vendor null and is materially different.
    predicates: tuple[frozenset[str | None] | None, ...]
    disposition: str
    successor_policy: str
    terminal_total_return_already_applied: bool

    def __post_init__(self) -> None:
        _require(
            len(self.predicates) == len(MATCH_FIELDS)
            and all(
                predicate is None
                or (
                    type(predicate) is frozenset
                    and bool(predicate)
                    and all(value is None or isinstance(value, str) for value in predicate)
                )
                for predicate in self.predicates
            ),
            "corporate-action rule predicates are malformed",
        )
        _require(self.disposition in DISPOSITIONS, "unknown action disposition")
        _require(
            self.successor_policy in SUCCESSOR_POLICIES,
            "unknown action successor policy",
        )
        _require(
            type(self.terminal_total_return_already_applied) is bool,
            "terminal-return flag must be boolean",
        )
        expected_policy = "ignored"
        _require(
            self.successor_policy == expected_policy,
            "action disposition/successor policy combination is inconsistent",
        )
        _require(
            self.terminal_total_return_already_applied is True,
            "Stage 11 actions must acknowledge the authoritative terminal return",
        )

    def matches(self, values: tuple[str | None, ...]) -> bool:
        return all(
            predicate is None or value in predicate
            for predicate, value in zip(self.predicates, values, strict=True)
        )


@dataclass(frozen=True, slots=True, init=False)
class CorporateActionMappingAuthority:
    """Privately issued complete mapping held inside one live unblind scope."""

    manifest_sha256: str
    rules_sha256: str
    allowed_vocabularies: Mapping[str, frozenset[str | None]]
    rules: tuple[CorporateActionMappingRule, ...]
    _stage11_authority: object

    def __init__(
        self,
        *,
        manifest_sha256: str,
        rules_sha256: str,
        allowed_vocabularies: Mapping[str, frozenset[str | None]],
        rules: Sequence[CorporateActionMappingRule],
        stage11_authority: object,
        _issuance_token: object,
    ) -> None:
        if _issuance_token is not _ISSUANCE_TOKEN:
            raise CorporateActionAdapterError(
                "corporate-action mapping authority must be issued by its frozen factory"
            )
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "rules_sha256", rules_sha256)
        object.__setattr__(
            self,
            "allowed_vocabularies",
            MappingProxyType(dict(allowed_vocabularies)),
        )
        object.__setattr__(self, "rules", tuple(rules))
        object.__setattr__(self, "_stage11_authority", stage11_authority)

    def gate(self) -> None:
        from portfolio_bot.validation.validation_runner import (
            require_live_stage11_validation_unblind_authority,
        )

        require_live_stage11_validation_unblind_authority(self._stage11_authority)
        _require(
            FROZEN_MAPPING_MANIFEST_SHA256 == self.manifest_sha256,
            "corporate-action mapping manifest authority changed",
        )


def _exact_keys(value: object, expected: set[str], *, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    observed = set(value)
    _require(
        observed == expected,
        f"{label} fields differ; missing={sorted(expected - observed)}, "
        f"extra={sorted(observed - expected)}",
    )
    return value


def authenticate_corporate_action_mapping_manifest(
    stage11_authority: object,
) -> CorporateActionMappingAuthority:
    """Authenticate the one fixed mapping only after the live execution gate."""

    from portfolio_bot.validation.validation_runner import (
        require_live_stage11_validation_unblind_authority,
    )

    # This must precede the manifest existence check or path inspection.
    require_live_stage11_validation_unblind_authority(stage11_authority)
    expected_sha = FROZEN_MAPPING_MANIFEST_SHA256
    _require(
        isinstance(expected_sha, str) and _HEX64.fullmatch(expected_sha) is not None,
        "corporate-action mapping remains sealed: no reviewed manifest digest is frozen",
    )
    path = FROZEN_MAPPING_MANIFEST_PATH
    try:
        before = path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise CorporateActionAdapterError(
            "frozen corporate-action mapping manifest is absent"
        ) from exc
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_nlink == 1,
        "corporate-action mapping manifest is not an ordinary single-link file",
    )
    payload = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    _require(
        _stable_identity(before) == _stable_identity(after),
        "corporate-action mapping manifest changed while read",
    )
    manifest_sha = _sha256_bytes(payload)
    _require(manifest_sha == expected_sha, "corporate-action mapping manifest SHA-256 differs")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorporateActionAdapterError(
            "corporate-action mapping manifest is invalid JSON"
        ) from exc
    manifest = _exact_keys(
        raw,
        {
            "schema_version",
            "manifest_id",
            "status",
            "match_fields",
            "allowed_vocabularies",
            "rules",
            "rules_sha256",
            "payload_sha256",
        },
        label="corporate-action mapping manifest",
    )
    _require(payload == _canonical_json_bytes(manifest), "mapping manifest is not canonical JSON")
    _require(
        manifest["schema_version"] == 1
        and manifest["manifest_id"] == MAPPING_MANIFEST_ID
        and manifest["status"] == MAPPING_MANIFEST_STATUS
        and tuple(manifest["match_fields"]) == MATCH_FIELDS,
        "corporate-action mapping manifest identity differs",
    )
    supplied_payload_sha = manifest["payload_sha256"]
    _require(
        isinstance(supplied_payload_sha, str)
        and _HEX64.fullmatch(supplied_payload_sha) is not None,
        "mapping payload digest is malformed",
    )
    without_digest = dict(manifest)
    without_digest.pop("payload_sha256")
    _require(
        supplied_payload_sha == _sha256_bytes(_canonical_json_bytes(without_digest)),
        "mapping payload self-digest differs",
    )
    raw_vocabularies = _exact_keys(
        manifest["allowed_vocabularies"],
        set(MATCH_FIELDS),
        label="corporate-action allowed vocabularies",
    )
    allowed_vocabularies: dict[str, frozenset[str | None]] = {}
    for field in MATCH_FIELDS:
        values = _normalized_vocabulary(
            raw_vocabularies[field],
            label=f"corporate-action vocabulary {field}",
        )
        allowed_vocabularies[field] = frozenset(values)
    _require(
        tuple(raw_vocabularies["action_type"]) == CRSP_ACTION_TYPES
        and tuple(raw_vocabularies["status_type"]) == CRSP_STATUS_TYPES
        and tuple(raw_vocabularies["reason_type"]) == CRSP_REASON_TYPES
        and tuple(raw_vocabularies["payment_type"]) == CRSP_PAYMENT_TYPES
        and tuple(raw_vocabularies["distribution_type"])
        == CRSP_DISTRIBUTION_TYPES,
        "corporate-action vendor vocabulary differs from the official CRSP domains",
    )
    _require(
        tuple(raw_vocabularies["successor_presence"]) == ("absent", "present"),
        "corporate-action successor-presence vocabulary differs",
    )
    _require(
        tuple(raw_vocabularies["treatment_status"]) == TREATMENT_STATUS_DOMAIN,
        "corporate-action treatment-status vocabulary differs",
    )
    raw_rules = manifest["rules"]
    _require(isinstance(raw_rules, list) and raw_rules, "mapping rule set is empty")
    rules: list[CorporateActionMappingRule] = []
    seen_predicates: set[tuple[frozenset[str | None] | None, ...]] = set()
    normalized_payload: list[dict[str, object]] = []
    for position, raw_rule in enumerate(raw_rules):
        rule_value = _exact_keys(
            raw_rule,
            {
                "match",
                "disposition",
                "successor_policy",
                "terminal_total_return_already_applied",
            },
            label=f"corporate-action mapping rule {position}",
        )
        match_value = _exact_keys(
            rule_value["match"],
            set(MATCH_FIELDS),
            label=f"corporate-action mapping rule {position} match",
        )
        predicates: list[frozenset[str | None] | None] = []
        normalized_match: dict[str, object] = {}
        for field in MATCH_FIELDS:
            raw_predicate = match_value[field]
            if raw_predicate == "*":
                predicates.append(None)
                normalized_match[field] = "*"
                continue
            values = _normalized_vocabulary(
                raw_predicate,
                label=f"corporate-action rule {position} predicate {field}",
            )
            _require(
                set(values).issubset(allowed_vocabularies[field]),
                f"corporate-action rule {position} uses an undeclared {field} code",
            )
            predicates.append(frozenset(values))
            normalized_match[field] = list(values)
        predicate_tuple = tuple(predicates)
        rule = CorporateActionMappingRule(
            predicates=predicate_tuple,
            disposition=str(rule_value["disposition"]),
            successor_policy=str(rule_value["successor_policy"]),
            terminal_total_return_already_applied=rule_value[
                "terminal_total_return_already_applied"
            ],
        )
        _require(
            predicate_tuple not in seen_predicates,
            "corporate-action mapping contains a duplicate predicate",
        )
        seen_predicates.add(predicate_tuple)
        rules.append(rule)
        normalized_payload.append(
            {
                "match": normalized_match,
                "disposition": rule.disposition,
                "successor_policy": rule.successor_policy,
                "terminal_total_return_already_applied": (
                    rule.terminal_total_return_already_applied
                ),
            }
        )
    rules_sha = _sha256_bytes(_canonical_json_bytes(normalized_payload))
    _require(
        manifest["rules_sha256"] == rules_sha,
        "corporate-action mapping rule digest differs",
    )
    require_live_stage11_validation_unblind_authority(stage11_authority)
    return CorporateActionMappingAuthority(
        manifest_sha256=manifest_sha,
        rules_sha256=rules_sha,
        allowed_vocabularies=allowed_vocabularies,
        rules=rules,
        stage11_authority=stage11_authority,
        _issuance_token=_ISSUANCE_TOKEN,
    )


def _as_table(value: object) -> pa.Table:
    if isinstance(value, pa.Table):
        return value.combine_chunks()
    raise CorporateActionAdapterError("delisting input must be an Arrow table")


def map_delisting_rows_to_corporate_actions(
    table: object,
    *,
    authority: CorporateActionMappingAuthority,
) -> tuple[CorporateAction, ...]:
    """Map every row explicitly; unknown/ambiguous mechanics fail closed."""

    _require(
        type(authority) is CorporateActionMappingAuthority,
        "empirical action mapping requires the exact mapping authority",
    )
    authority.gate()
    source = _as_table(table)
    observed_columns = tuple(source.column_names)
    allowed_schemas = {
        tuple(DELISTING_COLUMNS),
        (*FIXED_INTERVAL_COLUMNS, *DELISTING_COLUMNS),
    }
    _require(
        observed_columns in allowed_schemas,
        "delisting input columns differ from the frozen materialization schema",
    )
    candidates: list[tuple[dict[str, Any], CorporateActionMappingRule, str]] = []
    for raw in source.to_pylist():
        _require(raw.get("assigned_exactly_once") is True, "delisting row is not assigned exactly once")
        treatment = raw.get("treatment_status")
        _require(
            treatment not in UNRESOLVED_TERMINAL_TREATMENTS,
            "unresolved terminal delisting cannot enter portfolio accounting",
        )
        _require(
            treatment in RESOLVED_TERMINAL_TREATMENTS,
            "delisting treatment is not a resolved terminal event",
        )
        successor = raw.get("successor_permno")
        key = (
            *(raw.get(name) for name in VENDOR_MATCH_FIELDS),
            "absent" if successor is None else "present",
        )
        for field, value in zip(MATCH_FIELDS, key, strict=True):
            _require(
                value in authority.allowed_vocabularies[field],
                f"unknown corporate-action vendor code for {field}: {value!r}",
            )
        matches = tuple(rule for rule in authority.rules if rule.matches(key))
        _require(
            len(matches) == 1,
            "corporate-action vendor tuple must match exactly one frozen rule: "
            + repr(key),
        )
        rule = matches[0]
        source_payload = {
            name: _row_value(raw.get(name)) for name in observed_columns
        }
        source_sha = _sha256_bytes(_canonical_json_bytes(source_payload))
        candidates.append((raw, rule, source_sha))

    candidates.sort(
        key=lambda item: (
            item[0].get("assigned_return_date"),
            int(item[0].get("permno")),
            item[0].get("delisting_date"),
            item[2],
        )
    )
    seen_source_hashes: set[str] = set()
    next_sequence_by_date: dict[dt.date, int] = {}
    actions: list[CorporateAction] = []
    for raw, rule, source_sha in candidates:
        _require(source_sha not in seen_source_hashes, "duplicate corporate-action source row")
        seen_source_hashes.add(source_sha)
        assigned = raw.get("assigned_return_date")
        _require(type(assigned) is dt.date, "assigned corporate-action date is malformed")
        sequence = next_sequence_by_date.get(assigned, 0) + 1
        next_sequence_by_date[assigned] = sequence
        action_id = "stage11-action-" + _sha256_bytes(
            _canonical_json_bytes(
                {
                    "source_record_sha256": source_sha,
                    "mapping_manifest_sha256": authority.manifest_sha256,
                    "rules_sha256": authority.rules_sha256,
                    "disposition": rule.disposition,
                }
            )
        )[:32]
        actions.append(
            CorporateAction(
                action_id=action_id,
                economic_event_date=raw.get("delisting_date"),
                assigned_return_date=assigned,
                public_availability_date=raw.get("outcome_availability_date"),
                action_sequence=sequence,
                source_record_sha256=source_sha,
                source_permno=raw.get("permno"),
                settlement_kind=rule.disposition,
                successor_permno=None,
                settlement_value_multiplier=1.0,
                terminal_total_return_already_applied=(
                    rule.terminal_total_return_already_applied
                ),
            )
        )
    authority.gate()
    return tuple(actions)


__all__ = [
    "CRSP_ACTION_TYPES",
    "CRSP_DISTRIBUTION_TYPES",
    "CRSP_PAYMENT_TYPES",
    "CRSP_REASON_TYPES",
    "CRSP_STATUS_TYPES",
    "CorporateActionAdapterError",
    "CorporateActionMappingAuthority",
    "CorporateActionMappingRule",
    "FROZEN_MAPPING_MANIFEST_PATH",
    "FROZEN_MAPPING_MANIFEST_SHA256",
    "MAPPING_MANIFEST_ID",
    "MAPPING_MANIFEST_RELATIVE_PATH",
    "MAPPING_MANIFEST_STATUS",
    "MATCH_FIELDS",
    "RESOLVED_TERMINAL_TREATMENTS",
    "TREATMENT_STATUS_DOMAIN",
    "UNRESOLVED_TERMINAL_TREATMENT",
    "VENDOR_MATCH_FIELDS",
    "authenticate_corporate_action_mapping_manifest",
    "map_delisting_rows_to_corporate_actions",
]
