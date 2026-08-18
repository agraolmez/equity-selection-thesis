"""Single sealed input front door for empirical Stage 11 execution.

The empirical runner accepts no path arguments.  This module opens one fixed,
hash-frozen repository manifest *after* a live validation-unblind authority is
authenticated, reauthenticates the independently verified Gate-A ten-source
bundle, predictor, and development OOF authorities, binds the outcome-
materialization registrations, and creates one shared execution broker.  An
input that is not named and content-addressed by that manifest cannot enter the
backtest.
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
from types import MappingProxyType
from typing import Any, Mapping

import pyarrow.parquet as pq

from portfolio_bot.selectors.contract import (
    SelectorContractError,
    canonical_json_bytes as selector_canonical_json_bytes,
)
from portfolio_bot.selectors.inputs import (
    Stage9Inputs,
    reauthenticate_stage9_inputs,
    resolve_stage9_inputs,
)
import portfolio_bot.validation.corporate_action_adapter as corporate_action_adapter
from portfolio_bot.validation.access import (
    FROZEN_STAGE6_RAW_SCHEMA_SHA256,
    PayloadRole,
    Stage11AccessError,
)
from portfolio_bot.validation.calendar import VALIDATION_HOLDING_MONTHS
from portfolio_bot.validation.corporate_action_adapter import (
    FROZEN_MAPPING_MANIFEST_PATH,
    MAPPING_MANIFEST_ID,
    CorporateActionMappingAuthority,
    authenticate_corporate_action_mapping_manifest,
)
from portfolio_bot.validation.execution_access import (
    ExecutionDateAxis,
    RegisteredExecutionPayload,
    ValidationExecutionAccessBroker,
)
from portfolio_bot.validation.execution_risk_inputs import (
    RISK_COST_REGISTRATION_NAMES,
    build_authenticated_risk_cost_registrations,
)
from portfolio_bot.validation.outcome_materialization import (
    CONTRACT_ID as OUTCOME_CONTRACT_ID,
    DEFAULT_INPUT_NAMES,
    FROZEN_INPUT_MANIFEST_PATH as OUTCOME_INPUT_MANIFEST_PATH,
    _validate_registration_contract,
)
from portfolio_bot.validation.predictor_adapter import (
    DATABASE_ROOT,
    AuthenticatedValidationPredictor,
    authenticate_validation_predictor_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXECUTION_INPUT_MANIFEST_RELATIVE_PATH = (
    "artifacts/phase_05_validation/STAGE_11_EXECUTION_INPUTS.json"
)
FROZEN_EXECUTION_INPUT_MANIFEST_PATH = (
    PROJECT_ROOT / EXECUTION_INPUT_MANIFEST_RELATIVE_PATH
)
EXECUTION_INPUT_MANIFEST_ID = "phase05-stage11-empirical-execution-inputs-v1"
EXECUTION_INPUT_MANIFEST_STATUS = "SEALED_STAGE11_EMPIRICAL_EXECUTION_INPUTS"

SOURCE_PLAN_RELATIVE_PATH = (
    "configs/phase_05_validation/stage_11_source_materialization_plan_v1.json"
)
SOURCE_PRIMARY_POINTER_RELATIVE_PATH = (
    "artifacts/phase_05_validation/STAGE_11_SOURCE_MATERIALIZATION_PRIMARY.json"
)
SOURCE_VERIFIED_POINTER_RELATIVE_PATH = (
    "artifacts/phase_05_validation/STAGE_11_SOURCE_MATERIALIZATION_VERIFIED.json"
)
SOURCE_PLAN_SHA256 = "02707c1f885ea67e866970dbe78537d31f47d4023b8a9d9d4eb603d213fbe594"
SOURCE_PLAN_PAYLOAD_SHA256 = (
    "dd07e9467ed4d73e77618e499c3d58b1575ce487c8359791573d451236714a4e"
)
SOURCE_PUBLICATION_ID = "phase05-stage11-source-materialization-publication-v1"
SOURCE_PRIMARY_STATUS = "STAGE11_SOURCE_MATERIALIZATION_COMPLETE"
SOURCE_VERIFICATION_IDENTITY = (
    "phase05-stage11-source-materialization-independent-verifier-v1"
)
SOURCE_VERIFIED_STATUS = "STAGE11_SOURCE_MATERIALIZATION_VERIFIED"
NORMALIZED_SOURCE_NAMES = (
    "validation_sessions_main",
    "validation_sessions_reserve",
    "validation_market_main",
    "validation_market_reserve",
    "validation_returns_main",
    "validation_returns_reserve",
    "validation_factors_main",
    "validation_factors_reserve",
    "validation_delistings_main",
    "validation_delistings_reserve",
)

# Frozen only after every empirical input and mapping rule was materialized,
# independently reproduced, and explicitly approved.  Any digest mismatch
# keeps the runner sealed before an empirical input can enter the broker.
FROZEN_EXECUTION_INPUT_MANIFEST_SHA256: str | None = (
    "0893c445dc99516fdc03cb31aa9253b128aed1c87ec4da0397148dbb9ea3cfca"
)

SELECTOR_VERIFIED_POINTER_RELATIVE_PATH = (
    "artifacts/phase_04_models/LATEST_SELECTOR_VERIFIED.json"
)
SELECTOR_PRIMARY_MANIFEST_RELATIVE_PATH = (
    "artifacts/phase_04_models/selector_builds/"
    "phase04-selector-results-09a59d61ed58e77d80448a97/manifest.json"
)
SELECTOR_VERIFIED_POINTER_SHA256 = (
    "e4ffe4fdd999d15f2a1afa796c76155abb71708a26efbde135a70f6ca77ca2f5"
)
SELECTOR_PRIMARY_MANIFEST_SHA256 = (
    "e25c06b473975aca6ff5299288aef0abec383f44c9d3e8b005d3464517be4bfa"
)
SELECTOR_BUILD_ID = "phase04-selector-results-09a59d61ed58e77d80448a97"
SELECTOR_VERIFICATION_ID = "verify-257784339805ef6a"
SELECTOR_INPUT_RECEIPTS_SHA256 = (
    "cda43256e8e86b34094105d08c26b2f9bc4d18c8fbd733039098072607860a38"
)
DEVELOPMENT_OOF_NAME = "selector_oof_forecasts.parquet"
DEVELOPMENT_OOF_SHA256 = (
    "dd967332b5a62500826fc47ecf59d2cbc83b33e23761eef554ab483bc1a346d0"
)
DEVELOPMENT_OOF_SCHEMA_SHA256 = (
    "ad63782ba58461d7b05ba0c082991616dd171e319b04bec2d8c416f0504ddb45"
)
DEVELOPMENT_OOF_SIZE_BYTES = 726_777_676
DEVELOPMENT_OOF_ROWS = 172_969

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_BUILD_ID = re.compile(r"stage11-sources-[0-9a-f]{24}\Z")
_SOURCE_VERIFICATION_ID = re.compile(r"verify-stage11-sources-[0-9a-f]{20}\Z")
_TOP_KEYS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "status",
        "scope",
        "predictor_authority",
        "normalized_sources",
        "outcome_materialization",
        "development_selector",
        "corporate_actions",
        "output_roots",
        "payload_sha256",
    }
)
_NORMALIZED_SOURCE_KEYS = frozenset(
    {
        "plan_path",
        "plan_sha256",
        "plan_payload_sha256",
        "primary_pointer_path",
        "primary_pointer_sha256",
        "verified_pointer_path",
        "verified_pointer_sha256",
        "source_authority_sha256",
        "primary_build_id",
        "primary_manifest_sha256",
        "verification_id",
        "verification_report_sha256",
        "execution_input_records_sha256",
    }
)
_SCOPE_KEYS = frozenset(
    {
        "first_signal_date",
        "last_signal_date",
        "last_exit_date",
        "first_holding_month",
        "last_holding_month",
        "holding_month_count",
        "embargoed_holding_month",
        "first_prohibited_final_date",
    }
)
_PREDICTOR_KEYS = frozenset(
    {
        "manifest_path",
        "manifest_sha256",
        "validation_data_build_id",
        "validation_data_verification_id",
        "candidate_manifest_sha256",
        "verification_report_sha256",
        "materialization_code_hashes_sha256",
        "payload_sha256",
        "payload_schema_sha256",
        "payload_rows",
    }
)
_OUTCOME_KEYS = frozenset(
    {"input_manifest_path", "input_manifest_sha256", "contract_id"}
)
_SELECTOR_KEYS = frozenset(
    {
        "verified_pointer_path",
        "verified_pointer_sha256",
        "primary_manifest_path",
        "primary_manifest_sha256",
        "build_id",
        "verification_id",
        "input_receipts_sha256",
        "oof_payload_sha256",
        "oof_schema_sha256",
        "oof_size_bytes",
        "oof_rows",
    }
)
_ACTION_KEYS = frozenset(
    {"mapping_manifest_path", "mapping_manifest_sha256", "mapping_manifest_id"}
)
_OUTPUT_KEYS = frozenset(
    {"outcome_tables", "selector_checkpoints", "raw_forecasts", "execution_results"}
)


class Stage11ExecutionInputError(RuntimeError):
    """Raised before execution when a sealed input binding differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11ExecutionInputError(message)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


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


def _exact_keys(
    value: object,
    expected: frozenset[str] | set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    observed = frozenset(value)
    selected = frozenset(expected)
    _require(
        observed == selected,
        f"{label} fields differ; missing={sorted(selected - observed)}, "
        f"extra={sorted(observed - selected)}",
    )
    return value


def _sha(value: object, *, label: str) -> str:
    _require(
        isinstance(value, str) and _HEX64.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _ordinary_file(
    path: Path, *, root: Path, label: str
) -> tuple[Path, os.stat_result]:
    root_absolute = Path(os.path.abspath(root.expanduser()))
    lexical = Path(os.path.abspath(path.expanduser()))
    try:
        lexical.relative_to(root_absolute)
    except ValueError as exc:
        raise Stage11ExecutionInputError(f"{label} escaped its trusted root") from exc
    _require(lexical.resolve(strict=False) == lexical, f"{label} uses a symlinked path")
    try:
        metadata = lexical.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise Stage11ExecutionInputError(f"{label} is absent") from exc
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_nlink == 1,
        f"{label} is not an ordinary single-link file",
    )
    return lexical, metadata


def _read_bound_file(
    path: Path,
    *,
    root: Path,
    expected_sha256: str,
    label: str,
) -> bytes:
    selected, before = _ordinary_file(path, root=root, label=label)
    payload = selected.read_bytes()
    after = selected.stat(follow_symlinks=False)
    _require(
        _stable_identity(before) == _stable_identity(after),
        f"{label} changed while read",
    )
    _require(_sha256_bytes(payload) == expected_sha256, f"{label} SHA-256 differs")
    return payload


def _json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11ExecutionInputError(f"{label} is invalid JSON") from exc
    _require(isinstance(value, Mapping), f"{label} must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class AuthenticatedStage11NormalizedSources:
    """Independently verified Gate-A records admitted to the Gate-B broker."""

    plan_sha256: str
    plan_payload_sha256: str
    source_authority_sha256: str
    primary_build_id: str
    primary_pointer_sha256: str
    primary_manifest_sha256: str
    verification_id: str
    verified_pointer_sha256: str
    verification_report_sha256: str
    execution_input_records_sha256: str
    records: Mapping[str, RegisteredExecutionPayload]
    _bound_file_identities: tuple[tuple[Path, tuple[int, ...]], ...]

    def gate(self) -> None:
        _require(
            set(self.records) == set(NORMALIZED_SOURCE_NAMES),
            "authenticated Gate-A source record set changed",
        )
        for path, expected in self._bound_file_identities:
            try:
                current = path.stat(follow_symlinks=False)
            except FileNotFoundError as exc:
                raise Stage11ExecutionInputError(
                    "authenticated Gate-A source file disappeared"
                ) from exc
            _require(
                _stable_identity(current) == expected,
                f"authenticated Gate-A source file changed: {path.name}",
            )


@dataclass(frozen=True, slots=True)
class AuthenticatedDevelopmentOOF:
    path: Path
    sha256: str
    size_bytes: int
    rows: int
    schema_sha256: str
    selector_build_id: str
    selector_verification_id: str
    input_receipts_sha256: str
    stage9_inputs: Stage9Inputs
    score_column_by_model: Mapping[tuple[str, str], str]
    _file_identity: tuple[int, ...]

    def gate(self) -> None:
        """Recheck the exact development OOF inode throughout execution."""

        current = self.path.stat(follow_symlinks=False)
        _require(
            _stable_identity(current) == self._file_identity,
            "development OOF changed during the live run",
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedStage11ExecutionInputs:
    """All empirical sources bound to one live authority and one broker."""

    manifest_path: Path
    manifest_sha256: str
    manifest_payload_sha256: str
    normalized_sources: AuthenticatedStage11NormalizedSources
    predictor: AuthenticatedValidationPredictor
    development_oof: AuthenticatedDevelopmentOOF
    corporate_action_authority: CorporateActionMappingAuthority
    registrations: Mapping[str, RegisteredExecutionPayload]
    output_roots: Mapping[str, Path]
    broker: ValidationExecutionAccessBroker
    _stage11_authority: object
    _manifest_identity: tuple[int, ...]

    def gate(self) -> None:
        from portfolio_bot.validation.validation_runner import (
            require_live_stage11_validation_unblind_authority,
        )

        require_live_stage11_validation_unblind_authority(self._stage11_authority)
        _require(
            FROZEN_EXECUTION_INPUT_MANIFEST_SHA256 == self.manifest_sha256,
            "execution input manifest authority changed",
        )
        current = self.manifest_path.stat(follow_symlinks=False)
        _require(
            _stable_identity(current) == self._manifest_identity,
            "execution input manifest changed during the live run",
        )
        self.normalized_sources.gate()
        self.development_oof.gate()
        self.corporate_action_authority.gate()
        self.broker.ledger()


def _validate_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest = _exact_keys(value, _TOP_KEYS, label="execution input manifest")
    _require(
        manifest["schema_version"] == 1
        and manifest["manifest_id"] == EXECUTION_INPUT_MANIFEST_ID
        and manifest["status"] == EXECUTION_INPUT_MANIFEST_STATUS,
        "execution input manifest identity differs",
    )
    scope = _exact_keys(manifest["scope"], _SCOPE_KEYS, label="execution scope")
    _require(
        dict(scope)
        == {
            "first_signal_date": "2014-12-31",
            "last_signal_date": "2019-10-31",
            "last_exit_date": "2019-12-02",
            "first_holding_month": VALIDATION_HOLDING_MONTHS[0],
            "last_holding_month": VALIDATION_HOLDING_MONTHS[-1],
            "holding_month_count": len(VALIDATION_HOLDING_MONTHS),
            "embargoed_holding_month": "2019-12",
            "first_prohibited_final_date": "2020-01-02",
        },
        "execution input manifest scope differs",
    )
    _exact_keys(
        manifest["predictor_authority"], _PREDICTOR_KEYS, label="predictor authority"
    )
    normalized = _exact_keys(
        manifest["normalized_sources"],
        _NORMALIZED_SOURCE_KEYS,
        label="normalized-source authority",
    )
    _validate_project_relative(
        normalized["plan_path"],
        SOURCE_PLAN_RELATIVE_PATH,
        label="normalized-source plan",
    )
    _validate_project_relative(
        normalized["primary_pointer_path"],
        SOURCE_PRIMARY_POINTER_RELATIVE_PATH,
        label="normalized-source primary pointer",
    )
    _validate_project_relative(
        normalized["verified_pointer_path"],
        SOURCE_VERIFIED_POINTER_RELATIVE_PATH,
        label="normalized-source verified pointer",
    )
    _require(
        normalized["plan_sha256"] == SOURCE_PLAN_SHA256
        and normalized["plan_payload_sha256"] == SOURCE_PLAN_PAYLOAD_SHA256,
        "normalized-source plan binding differs",
    )
    for name in (
        "primary_pointer_sha256",
        "verified_pointer_sha256",
        "source_authority_sha256",
        "primary_manifest_sha256",
        "verification_report_sha256",
        "execution_input_records_sha256",
    ):
        _sha(normalized[name], label=f"normalized-source {name}")
    _require(
        isinstance(normalized["primary_build_id"], str)
        and _SOURCE_BUILD_ID.fullmatch(normalized["primary_build_id"]) is not None
        and isinstance(normalized["verification_id"], str)
        and _SOURCE_VERIFICATION_ID.fullmatch(normalized["verification_id"])
        is not None,
        "normalized-source build or verification identity differs",
    )
    _exact_keys(
        manifest["outcome_materialization"], _OUTCOME_KEYS, label="outcome binding"
    )
    _exact_keys(
        manifest["development_selector"],
        _SELECTOR_KEYS,
        label="development selector binding",
    )
    _exact_keys(
        manifest["corporate_actions"], _ACTION_KEYS, label="corporate-action binding"
    )
    _exact_keys(manifest["output_roots"], _OUTPUT_KEYS, label="execution output roots")
    supplied = _sha(
        manifest["payload_sha256"], label="execution manifest payload_sha256"
    )
    without_digest = dict(manifest)
    without_digest.pop("payload_sha256")
    _require(
        supplied == _sha256_bytes(_canonical_json_bytes(without_digest)),
        "execution input manifest payload self-digest differs",
    )
    return manifest


def _validate_project_relative(value: object, expected: str, *, label: str) -> Path:
    _require(value == expected, f"{label} path differs")
    relative = PurePosixPath(str(value))
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"{label} path is unsafe",
    )
    return PROJECT_ROOT.joinpath(*relative.parts)


def _read_stable_json_file(
    path: Path,
    *,
    root: Path,
    expected_sha256: str,
    label: str,
    canonical: bool = True,
) -> tuple[Mapping[str, Any], tuple[int, ...]]:
    selected, before = _ordinary_file(path, root=root, label=label)
    payload = selected.read_bytes()
    after = selected.stat(follow_symlinks=False)
    _require(
        _stable_identity(before) == _stable_identity(after),
        f"{label} changed while read",
    )
    _require(_sha256_bytes(payload) == expected_sha256, f"{label} SHA-256 differs")
    value = _json(payload, label=label)
    if canonical:
        _require(
            payload == _canonical_json_bytes(value), f"{label} is not canonical JSON"
        )
    return value, _stable_identity(after)


def _hash_stable_member(
    path: Path,
    *,
    root: Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, tuple[int, ...]]:
    selected, before = _ordinary_file(path, root=root, label=label)
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    after = selected.stat(follow_symlinks=False)
    _require(
        _stable_identity(before) == _stable_identity(after),
        f"{label} changed while hashed",
    )
    _require(digest.hexdigest() == expected_sha256, f"{label} SHA-256 differs")
    return selected, _stable_identity(after)


def _closed_bundle(
    path: Path,
    *,
    root: Path,
    build_id: str,
    members: tuple[str, ...],
    label: str,
) -> Path:
    root_absolute = Path(os.path.abspath(root.expanduser()))
    _require(
        root.is_absolute()
        and root_absolute.resolve(strict=False) == root_absolute
        and (
            _SOURCE_BUILD_ID.fullmatch(build_id) is not None
            or _SOURCE_VERIFICATION_ID.fullmatch(build_id) is not None
        ),
        f"{label} fixed root or build identity is unsafe",
    )
    expected = root_absolute / build_id
    selected = Path(os.path.abspath(path.expanduser()))
    _require(
        selected == expected
        and selected.resolve(strict=False) == selected
        and selected.is_dir()
        and not selected.is_symlink(),
        f"{label} path is unsafe or outside its fixed root",
    )
    _require(
        {value.name for value in selected.iterdir()} == set(members),
        f"{label} is not a closed member set",
    )
    return selected


def _authenticate_normalized_sources(
    binding_value: Mapping[str, Any],
) -> AuthenticatedStage11NormalizedSources:
    """Read-only authentication of the complete Gate-A verified-pointer chain."""

    binding = _exact_keys(
        binding_value,
        _NORMALIZED_SOURCE_KEYS,
        label="normalized-source authority",
    )
    plan_path = _validate_project_relative(
        binding["plan_path"],
        SOURCE_PLAN_RELATIVE_PATH,
        label="normalized-source plan",
    )
    primary_pointer_path = _validate_project_relative(
        binding["primary_pointer_path"],
        SOURCE_PRIMARY_POINTER_RELATIVE_PATH,
        label="normalized-source primary pointer",
    )
    verified_pointer_path = _validate_project_relative(
        binding["verified_pointer_path"],
        SOURCE_VERIFIED_POINTER_RELATIVE_PATH,
        label="normalized-source verified pointer",
    )
    _require(
        binding["plan_sha256"] == SOURCE_PLAN_SHA256
        and binding["plan_payload_sha256"] == SOURCE_PLAN_PAYLOAD_SHA256,
        "normalized-source plan binding differs",
    )
    bound_identities: list[tuple[Path, tuple[int, ...]]] = []

    verified, identity = _read_stable_json_file(
        verified_pointer_path,
        root=PROJECT_ROOT,
        expected_sha256=_sha(
            binding["verified_pointer_sha256"],
            label="normalized-source verified pointer SHA-256",
        ),
        label="Gate-A verified pointer",
    )
    bound_identities.append((verified_pointer_path, identity))
    _exact_keys(
        verified,
        {
            "schema_version",
            "verification_identity",
            "status",
            "stage",
            "gate",
            "primary_build_id",
            "primary_pointer_sha256",
            "primary_manifest_sha256",
            "verification_id",
            "verification_bundle_path",
            "verification_bundle_files",
            "verification_report_sha256",
            "difference_count",
            "authority_sha256",
            "plan_sha256",
            "gate_b_validation_execution_authorized",
            "final_2020_2025_authorized",
        },
        label="Gate-A verified pointer",
    )
    _require(
        verified["schema_version"] == 1
        and verified["verification_identity"] == SOURCE_VERIFICATION_IDENTITY
        and verified["status"] == SOURCE_VERIFIED_STATUS
        and verified["stage"] == 11
        and verified["gate"] == "A"
        and verified["difference_count"] == 0
        and verified["gate_b_validation_execution_authorized"] is False
        and verified["final_2020_2025_authorized"] is False,
        "Gate-A verified pointer identity or authorization differs",
    )
    for name in (
        "primary_pointer_sha256",
        "primary_manifest_sha256",
        "verification_report_sha256",
        "authority_sha256",
        "plan_sha256",
    ):
        _sha(verified[name], label=f"Gate-A verified pointer {name}")
    _require(
        verified["plan_sha256"] == SOURCE_PLAN_SHA256
        and verified["primary_build_id"] == binding["primary_build_id"]
        and verified["primary_pointer_sha256"] == binding["primary_pointer_sha256"]
        and verified["primary_manifest_sha256"] == binding["primary_manifest_sha256"]
        and verified["verification_id"] == binding["verification_id"]
        and verified["verification_report_sha256"]
        == binding["verification_report_sha256"]
        and verified["authority_sha256"] == binding["source_authority_sha256"],
        "Gate-A verified pointer differs from the execution-input binding",
    )

    plan, identity = _read_stable_json_file(
        plan_path,
        root=PROJECT_ROOT,
        expected_sha256=SOURCE_PLAN_SHA256,
        label="Gate-A source plan",
        canonical=False,
    )
    bound_identities.append((plan_path, identity))
    _require(
        plan.get("schema_version") == 1
        and plan.get("plan_id") == "phase05-stage11-source-materialization-plan-v1"
        and plan.get("stage") == 11
        and plan.get("gate") == "A"
        and plan.get("payload_sha256") == SOURCE_PLAN_PAYLOAD_SHA256,
        "Gate-A source plan identity differs",
    )
    plan_without_digest = dict(plan)
    plan_without_digest.pop("payload_sha256", None)
    _require(
        _sha256_bytes(_canonical_json_bytes(plan_without_digest))
        == SOURCE_PLAN_PAYLOAD_SHA256,
        "Gate-A source plan payload digest differs",
    )
    outputs = plan.get("outputs")
    publication = plan.get("publication")
    _require(
        isinstance(outputs, Mapping)
        and set(outputs) == set(NORMALIZED_SOURCE_NAMES)
        and isinstance(publication, Mapping)
        and publication.get("primary_pointer") == SOURCE_PRIMARY_POINTER_RELATIVE_PATH
        and publication.get("verified_pointer")
        == SOURCE_VERIFIED_POINTER_RELATIVE_PATH,
        "Gate-A plan output or pointer contract differs",
    )

    primary, identity = _read_stable_json_file(
        primary_pointer_path,
        root=PROJECT_ROOT,
        expected_sha256=_sha(
            binding["primary_pointer_sha256"],
            label="normalized-source primary pointer SHA-256",
        ),
        label="Gate-A primary pointer",
    )
    bound_identities.append((primary_pointer_path, identity))
    _exact_keys(
        primary,
        {
            "schema_version",
            "publication_id",
            "status",
            "stage",
            "gate",
            "build_id",
            "bundle_path",
            "bundle_files",
            "manifest_sha256",
            "authority_sha256",
            "plan_sha256",
            "gate_b_validation_execution_authorized",
            "final_2020_2025_authorized",
        },
        label="Gate-A primary pointer",
    )
    _require(
        primary["schema_version"] == 1
        and primary["publication_id"] == SOURCE_PUBLICATION_ID
        and primary["status"] == SOURCE_PRIMARY_STATUS
        and primary["stage"] == 11
        and primary["gate"] == "A"
        and primary["build_id"] == binding["primary_build_id"]
        and primary["manifest_sha256"] == binding["primary_manifest_sha256"]
        and primary["authority_sha256"] == binding["source_authority_sha256"]
        and primary["plan_sha256"] == SOURCE_PLAN_SHA256
        and primary["gate_b_validation_execution_authorized"] is False
        and primary["final_2020_2025_authorized"] is False,
        "Gate-A primary pointer identity or authorization differs",
    )

    table_members = tuple(f"{name}.parquet" for name in sorted(NORMALIZED_SOURCE_NAMES))
    primary_members = (*table_members, "manifest.json", "_SUCCESS.json")
    primary_files = primary.get("bundle_files")
    _require(
        isinstance(primary_files, Mapping)
        and set(primary_files) == set(primary_members)
        and all(_HEX64.fullmatch(str(value)) for value in primary_files.values()),
        "Gate-A primary pointer member hashes differ",
    )
    primary_root = Path(str(publication.get("primary_bundle_root", "")))
    primary_bundle = _closed_bundle(
        Path(str(primary["bundle_path"])),
        root=primary_root,
        build_id=str(primary["build_id"]),
        members=primary_members,
        label="Gate-A primary bundle",
    )
    for name in primary_members:
        path, member_identity = _hash_stable_member(
            primary_bundle / name,
            root=primary_bundle,
            expected_sha256=str(primary_files[name]),
            label=f"Gate-A primary member {name}",
        )
        bound_identities.append((path, member_identity))

    manifest, _ = _read_stable_json_file(
        primary_bundle / "manifest.json",
        root=primary_bundle,
        expected_sha256=str(primary["manifest_sha256"]),
        label="Gate-A primary manifest",
    )
    _exact_keys(
        manifest,
        {
            "schema_version",
            "publication_id",
            "status",
            "stage",
            "gate",
            "build_id",
            "input_fingerprint",
            "authority",
            "implementation_code_hashes",
            "upstream_authorities",
            "source_identities",
            "tables",
            "execution_input_records",
            "scope",
            "authorizations",
        },
        label="Gate-A primary manifest",
    )
    manifest_authority = _exact_keys(
        manifest["authority"],
        {
            "authority_sha256",
            "approval_file_sha256",
            "approval_payload_sha256",
            "approval_text_sha256",
            "plan_sha256",
            "plan_payload_sha256",
        },
        label="Gate-A manifest authority",
    )
    _require(
        manifest["schema_version"] == 1
        and manifest["publication_id"] == SOURCE_PUBLICATION_ID
        and manifest["status"] == SOURCE_PRIMARY_STATUS
        and manifest["stage"] == 11
        and manifest["gate"] == "A"
        and manifest["build_id"] == primary["build_id"]
        and manifest_authority["authority_sha256"] == primary["authority_sha256"]
        and manifest_authority["plan_sha256"] == SOURCE_PLAN_SHA256
        and manifest_authority["plan_payload_sha256"] == SOURCE_PLAN_PAYLOAD_SHA256
        and manifest["implementation_code_hashes"]
        == plan.get("implementation_code_hashes")
        and manifest["upstream_authorities"] == plan.get("upstream_authorities")
        and manifest["source_identities"] == plan.get("sources")
        and manifest["scope"] == plan.get("scope"),
        "Gate-A primary manifest plan binding differs",
    )
    _require(
        manifest["authorizations"]
        == {
            "source_materialization_only": True,
            "gate_b_validation_execution_authorized": False,
            "labels_or_targets_created": False,
            "forecasts_or_model_selection_created": False,
            "portfolio_weights_or_trades_created": False,
            "pnl_or_performance_created": False,
            "december_2019_holding_month_included": False,
            "final_2020_2025_included": False,
            "stage12_or_later_authorized": False,
        },
        "Gate-A primary manifest authorization differs",
    )
    tables = manifest.get("tables")
    raw_records = manifest.get("execution_input_records")
    _require(
        isinstance(tables, Mapping)
        and set(tables) == set(NORMALIZED_SOURCE_NAMES)
        and isinstance(raw_records, Mapping)
        and set(raw_records) == set(NORMALIZED_SOURCE_NAMES),
        "Gate-A table or execution-record set differs",
    )
    records_digest = _sha256_bytes(_canonical_json_bytes(raw_records))
    _require(
        records_digest == binding["execution_input_records_sha256"],
        "Gate-A execution-record digest differs",
    )

    expected_roles = {
        "sessions": PayloadRole.VALIDATION_TRADING_SESSION_METADATA,
        "market": PayloadRole.VALIDATION_MARKET_INPUT,
        "returns": PayloadRole.VALIDATION_FORWARD_RETURN,
        "factors": PayloadRole.VALIDATION_ACCOUNTING_INPUT,
        "delistings": PayloadRole.VALIDATION_ACCOUNTING_INPUT,
    }
    expected_date_columns = {
        "sessions": "calendar_date",
        "market": "calendar_date",
        "returns": "calendar_date",
        "factors": "observation_date",
        "delistings": "assigned_return_date",
    }
    records: dict[str, RegisteredExecutionPayload] = {}
    for name in NORMALIZED_SOURCE_NAMES:
        contract = outputs[name]
        receipt = _exact_keys(
            tables[name],
            {
                "path",
                "sha256",
                "size_bytes",
                "rows",
                "row_groups",
                "schema_sha256",
                "minimum_date",
                "maximum_date",
            },
            label=f"Gate-A table receipt {name}",
        )
        raw = _exact_keys(
            raw_records[name],
            {
                "role",
                "path",
                "sha256",
                "size_bytes",
                "allowed_columns",
                "date_column",
                "date_axis",
                "minimum_date",
                "maximum_date",
                "holding_month_column",
                "trusted_schema_sha256",
            },
            label=f"Gate-A execution record {name}",
        )
        kind = name.removeprefix("validation_").rsplit("_", 1)[0]
        filename = f"{name}.parquet"
        path = primary_bundle / filename
        _require(
            isinstance(contract, Mapping)
            and contract.get("filename") == filename
            and contract.get("date_column") == expected_date_columns[kind]
            and contract.get("predictor_use_permitted") is False
            and receipt["path"] == filename
            and receipt["sha256"] == primary_files[filename] == raw["sha256"]
            and receipt["size_bytes"] == raw["size_bytes"] == path.stat().st_size
            and receipt["schema_sha256"] == contract.get("schema_sha256")
            and raw["path"] == str(path)
            and raw["role"] == expected_roles[kind].value
            and raw["allowed_columns"] == contract.get("allowed_columns")
            and raw["date_column"] == contract.get("date_column")
            and raw["date_axis"] == ExecutionDateAxis.OBSERVATION.value
            and raw["minimum_date"] == contract.get("minimum_date")
            and raw["maximum_date"] == contract.get("maximum_date")
            and raw["holding_month_column"] is None
            and raw["trusted_schema_sha256"] is None,
            f"Gate-A table contract differs: {name}",
        )
        _require(
            type(receipt["rows"]) is int
            and receipt["rows"] >= 0
            and type(receipt["row_groups"]) is int
            and receipt["row_groups"] >= 0,
            f"Gate-A table shape receipt differs: {name}",
        )
        parquet = pq.ParquetFile(path)
        _require(
            parquet.metadata.num_rows == receipt["rows"]
            and parquet.metadata.num_row_groups == receipt["row_groups"]
            and _sha256_bytes(parquet.schema_arrow.serialize().to_pybytes())
            == receipt["schema_sha256"]
            and parquet.schema_arrow.names == list(raw["allowed_columns"]),
            f"Gate-A table footer differs: {name}",
        )
        if receipt["rows"] == 0:
            _require(
                receipt["minimum_date"] is None and receipt["maximum_date"] is None,
                f"empty Gate-A table has date extrema: {name}",
            )
        else:
            try:
                observed_minimum = dt.date.fromisoformat(str(receipt["minimum_date"]))
                observed_maximum = dt.date.fromisoformat(str(receipt["maximum_date"]))
                contract_minimum = dt.date.fromisoformat(str(contract["minimum_date"]))
                contract_maximum = dt.date.fromisoformat(str(contract["maximum_date"]))
            except ValueError as exc:
                raise Stage11ExecutionInputError(
                    f"Gate-A table date receipt differs: {name}"
                ) from exc
            _require(
                contract_minimum
                <= observed_minimum
                <= observed_maximum
                <= contract_maximum,
                f"Gate-A table date extrema escaped its contract: {name}",
            )
        try:
            records[name] = RegisteredExecutionPayload(
                role=expected_roles[kind],
                path=path,
                sha256=str(raw["sha256"]),
                size_bytes=int(raw["size_bytes"]),
                allowed_columns=tuple(str(value) for value in raw["allowed_columns"]),
                date_column=str(raw["date_column"]),
                date_axis=ExecutionDateAxis(str(raw["date_axis"])),
                minimum_date=dt.date.fromisoformat(str(raw["minimum_date"])),
                maximum_date=dt.date.fromisoformat(str(raw["maximum_date"])),
                holding_month_column=None,
                trusted_schema_sha256=None,
            )
        except (TypeError, ValueError, Stage11AccessError) as exc:
            raise Stage11ExecutionInputError(
                f"invalid authenticated Gate-A execution record: {name}"
            ) from exc

    primary_success, _ = _read_stable_json_file(
        primary_bundle / "_SUCCESS.json",
        root=primary_bundle,
        expected_sha256=str(primary_files["_SUCCESS.json"]),
        label="Gate-A primary success receipt",
    )
    _require(
        dict(primary_success)
        == {
            "schema_version": 1,
            "status": SOURCE_PRIMARY_STATUS,
            "build_id": primary["build_id"],
            "manifest_sha256": primary["manifest_sha256"],
            "closed_members": list(primary_members),
        },
        "Gate-A primary success receipt differs",
    )

    verification_members = (
        "verification_report.json",
        "differences.json",
        "_SUCCESS.json",
    )
    verification_files = verified.get("verification_bundle_files")
    _require(
        isinstance(verification_files, Mapping)
        and set(verification_files) == set(verification_members)
        and all(_HEX64.fullmatch(str(value)) for value in verification_files.values()),
        "Gate-A verification pointer member hashes differ",
    )
    verification_root = Path(str(publication.get("verification_bundle_root", "")))
    verification_bundle = _closed_bundle(
        Path(str(verified["verification_bundle_path"])),
        root=verification_root,
        build_id=str(verified["verification_id"]),
        members=verification_members,
        label="Gate-A verification bundle",
    )
    for name in verification_members:
        path, member_identity = _hash_stable_member(
            verification_bundle / name,
            root=verification_bundle,
            expected_sha256=str(verification_files[name]),
            label=f"Gate-A verification member {name}",
        )
        bound_identities.append((path, member_identity))
    report, _ = _read_stable_json_file(
        verification_bundle / "verification_report.json",
        root=verification_bundle,
        expected_sha256=str(verified["verification_report_sha256"]),
        label="Gate-A verification report",
    )
    _require(
        report.get("schema_version") == 1
        and report.get("verification_identity") == SOURCE_VERIFICATION_IDENTITY
        and report.get("status") == SOURCE_VERIFIED_STATUS
        and report.get("stage") == 11
        and report.get("gate") == "A"
        and report.get("primary_build_id") == primary["build_id"]
        and report.get("primary_pointer_sha256") == binding["primary_pointer_sha256"]
        and report.get("primary_manifest_sha256") == primary["manifest_sha256"]
        and report.get("authority_sha256") == primary["authority_sha256"]
        and report.get("plan_sha256") == SOURCE_PLAN_SHA256
        and report.get("verification_id") == verified["verification_id"]
        and report.get("difference_count") == 0
        and report.get("differences") == [],
        "Gate-A independent verification report differs",
    )
    report_identity = dict(report)
    report_identity.pop("verification_id", None)
    _require(
        report["verification_id"]
        == "verify-stage11-sources-"
        + _sha256_bytes(_canonical_json_bytes(report_identity))[:20],
        "Gate-A verification identifier is not content addressed",
    )
    _require(
        report.get("checks")
        == {
            "primary_closed_set_authenticated": True,
            "main_upstream_receipts_reauthenticated_and_reprojected": True,
            "reserve_raw_extraction_independently_replayed": True,
            "all_output_byte_hashes_compared": True,
            "semantic_fallback_compared_on_byte_difference": True,
            "labels_forecasts_weights_trades_pnl_performance_created": False,
            "final_2020_plus_access_authorized": False,
            "gate_b_execution_authorized": False,
        },
        "Gate-A independent verification checks differ",
    )
    differences, _ = _read_stable_json_file(
        verification_bundle / "differences.json",
        root=verification_bundle,
        expected_sha256=str(verification_files["differences.json"]),
        label="Gate-A verification differences",
    )
    _require(
        dict(differences)
        == {"schema_version": 1, "difference_count": 0, "differences": []},
        "Gate-A verification differences are nonzero",
    )
    verification_success, _ = _read_stable_json_file(
        verification_bundle / "_SUCCESS.json",
        root=verification_bundle,
        expected_sha256=str(verification_files["_SUCCESS.json"]),
        label="Gate-A verification success receipt",
    )
    _require(
        dict(verification_success)
        == {
            "schema_version": 1,
            "status": SOURCE_VERIFIED_STATUS,
            "verification_id": verified["verification_id"],
            "report_sha256": verified["verification_report_sha256"],
            "differences_sha256": verification_files["differences.json"],
            "primary_manifest_sha256": primary["manifest_sha256"],
            "closed_members": list(verification_members),
        },
        "Gate-A verification success receipt differs",
    )
    result = AuthenticatedStage11NormalizedSources(
        plan_sha256=SOURCE_PLAN_SHA256,
        plan_payload_sha256=SOURCE_PLAN_PAYLOAD_SHA256,
        source_authority_sha256=str(primary["authority_sha256"]),
        primary_build_id=str(primary["build_id"]),
        primary_pointer_sha256=str(binding["primary_pointer_sha256"]),
        primary_manifest_sha256=str(primary["manifest_sha256"]),
        verification_id=str(verified["verification_id"]),
        verified_pointer_sha256=str(binding["verified_pointer_sha256"]),
        verification_report_sha256=str(verified["verification_report_sha256"]),
        execution_input_records_sha256=records_digest,
        records=MappingProxyType(records),
        _bound_file_identities=tuple(bound_identities),
    )
    result.gate()
    return result


def _authenticate_development_oof(
    binding: Mapping[str, Any],
) -> AuthenticatedDevelopmentOOF:
    pointer_path = _validate_project_relative(
        binding["verified_pointer_path"],
        SELECTOR_VERIFIED_POINTER_RELATIVE_PATH,
        label="selector verified pointer",
    )
    manifest_path = _validate_project_relative(
        binding["primary_manifest_path"],
        SELECTOR_PRIMARY_MANIFEST_RELATIVE_PATH,
        label="selector primary manifest",
    )
    expected = {
        "verified_pointer_sha256": SELECTOR_VERIFIED_POINTER_SHA256,
        "primary_manifest_sha256": SELECTOR_PRIMARY_MANIFEST_SHA256,
        "build_id": SELECTOR_BUILD_ID,
        "verification_id": SELECTOR_VERIFICATION_ID,
        "input_receipts_sha256": SELECTOR_INPUT_RECEIPTS_SHA256,
        "oof_payload_sha256": DEVELOPMENT_OOF_SHA256,
        "oof_schema_sha256": DEVELOPMENT_OOF_SCHEMA_SHA256,
        "oof_size_bytes": DEVELOPMENT_OOF_SIZE_BYTES,
        "oof_rows": DEVELOPMENT_OOF_ROWS,
    }
    for name, expected_value in expected.items():
        _require(
            binding[name] == expected_value,
            f"development selector binding differs: {name}",
        )
    pointer_payload = _read_bound_file(
        pointer_path,
        root=PROJECT_ROOT,
        expected_sha256=SELECTOR_VERIFIED_POINTER_SHA256,
        label="selector verified pointer",
    )
    pointer = _json(pointer_payload, label="selector verified pointer")
    _require(
        pointer.get("status") == "VERIFIED_PASSED"
        and pointer.get("build_id") == SELECTOR_BUILD_ID
        and pointer.get("verification_id") == SELECTOR_VERIFICATION_ID
        and pointer.get("primary_manifest_sha256") == SELECTOR_PRIMARY_MANIFEST_SHA256
        and pointer.get("authorizations", {}).get("independent_verification_completed")
        is True,
        "development selector verified authority differs",
    )
    primary_payload = _read_bound_file(
        manifest_path,
        root=PROJECT_ROOT,
        expected_sha256=SELECTOR_PRIMARY_MANIFEST_SHA256,
        label="selector primary manifest",
    )
    primary = _json(primary_payload, label="selector primary manifest")
    _require(
        primary.get("build_id") == SELECTOR_BUILD_ID
        and primary.get("status") == "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
        and primary.get("bindings", {}).get("input_receipts_sha256")
        == SELECTOR_INPUT_RECEIPTS_SHA256,
        "selector primary manifest identity differs",
    )
    try:
        stage9_inputs = resolve_stage9_inputs()
        reauthenticate_stage9_inputs(stage9_inputs)
    except (SelectorContractError, OSError, ValueError) as exc:
        raise Stage11ExecutionInputError(
            "authenticated Stage 9 development inputs differ"
        ) from exc
    input_receipts_sha256 = hashlib.sha256(
        selector_canonical_json_bytes(list(stage9_inputs.input_receipts))
    ).hexdigest()
    _require(
        input_receipts_sha256
        == binding["input_receipts_sha256"]
        == SELECTOR_INPUT_RECEIPTS_SHA256,
        "Stage 9 development input receipt digest differs",
    )
    try:
        interim = primary["outputs"]["interim"]
        receipt = interim["files"][DEVELOPMENT_OOF_NAME]
    except (KeyError, TypeError) as exc:
        raise Stage11ExecutionInputError(
            "selector primary manifest omitted the development OOF payload"
        ) from exc
    _require(
        receipt.get("sha256") == DEVELOPMENT_OOF_SHA256
        and receipt.get("schema_sha256") == DEVELOPMENT_OOF_SCHEMA_SHA256
        and receipt.get("size_bytes") == DEVELOPMENT_OOF_SIZE_BYTES
        and receipt.get("rows") == DEVELOPMENT_OOF_ROWS
        and receipt.get("path") == DEVELOPMENT_OOF_NAME,
        "development OOF receipt differs",
    )
    base = PurePosixPath(str(interim.get("path", "")))
    _require(
        bool(base.parts) and not base.is_absolute() and ".." not in base.parts,
        "development OOF base path is unsafe",
    )
    oof_path = DATABASE_ROOT.joinpath(*base.parts, DEVELOPMENT_OOF_NAME)
    selected, before = _ordinary_file(
        oof_path, root=DATABASE_ROOT, label="development OOF"
    )
    _require(
        before.st_size == DEVELOPMENT_OOF_SIZE_BYTES, "development OOF size differs"
    )
    _require(
        _sha256_file(selected) == DEVELOPMENT_OOF_SHA256,
        "development OOF SHA-256 differs",
    )
    after = selected.stat(follow_symlinks=False)
    _require(
        _stable_identity(before) == _stable_identity(after),
        "development OOF changed while hashed",
    )
    parquet = pq.ParquetFile(selected)
    schema_sha = _sha256_bytes(parquet.schema_arrow.serialize().to_pybytes())
    _require(
        parquet.metadata.num_rows == DEVELOPMENT_OOF_ROWS
        and schema_sha == DEVELOPMENT_OOF_SCHEMA_SHA256,
        "development OOF footer differs",
    )

    processed = primary.get("outputs", {}).get("processed")
    _require(
        isinstance(processed, Mapping), "selector processed output receipt is absent"
    )
    processed_files = processed.get("files")
    _require(
        isinstance(processed_files, Mapping),
        "selector processed file receipts are absent",
    )
    search_receipt = processed_files.get("selector_search_receipt.json")
    _require(isinstance(search_receipt, Mapping), "selector search receipt is absent")
    processed_base = PurePosixPath(str(processed.get("path", "")))
    _require(
        bool(processed_base.parts)
        and not processed_base.is_absolute()
        and ".." not in processed_base.parts,
        "selector processed output base path is unsafe",
    )
    _require(
        search_receipt.get("path") == "selector_search_receipt.json"
        and isinstance(search_receipt.get("sha256"), str)
        and _HEX64.fullmatch(str(search_receipt["sha256"])) is not None
        and type(search_receipt.get("size_bytes")) is int
        and int(search_receipt["size_bytes"]) > 0,
        "selector search receipt file binding differs",
    )
    search_path = DATABASE_ROOT.joinpath(
        *processed_base.parts, "selector_search_receipt.json"
    )
    search_payload = _read_bound_file(
        search_path,
        root=DATABASE_ROOT,
        expected_sha256=str(search_receipt["sha256"]),
        label="selector search receipt",
    )
    _require(
        len(search_payload) == int(search_receipt["size_bytes"]),
        "selector search receipt size differs",
    )
    search = _json(search_payload, label="selector search receipt")
    registry = search.get("column_registry")
    _require(isinstance(registry, Mapping), "selector score-column registry is absent")
    monthly_scores = registry.get("monthly_scores")
    _require(
        isinstance(monthly_scores, list) and len(monthly_scores) == 346,
        "selector monthly score-column registry differs",
    )
    score_column_by_model: dict[tuple[str, str], str] = {}
    for value in monthly_scores:
        _require(isinstance(value, Mapping), "selector score-column entry differs")
        key = (str(value.get("model_family", "")), str(value.get("config_id", "")))
        column = str(value.get("column_name", ""))
        _require(
            all(key)
            and column.startswith("score__")
            and key not in score_column_by_model,
            "selector score-column registry contains an invalid or duplicate entry",
        )
        score_column_by_model[key] = column
    _require(
        len(score_column_by_model) == 346
        and set(score_column_by_model.values()).issubset(parquet.schema_arrow.names),
        "selector score-column registry escaped the authenticated OOF schema",
    )
    return AuthenticatedDevelopmentOOF(
        path=selected,
        sha256=DEVELOPMENT_OOF_SHA256,
        size_bytes=DEVELOPMENT_OOF_SIZE_BYTES,
        rows=DEVELOPMENT_OOF_ROWS,
        schema_sha256=schema_sha,
        selector_build_id=SELECTOR_BUILD_ID,
        selector_verification_id=SELECTOR_VERIFICATION_ID,
        input_receipts_sha256=input_receipts_sha256,
        stage9_inputs=stage9_inputs,
        score_column_by_model=MappingProxyType(score_column_by_model),
        _file_identity=_stable_identity(after),
    )


def _authenticate_predictor(
    binding: Mapping[str, Any],
) -> AuthenticatedValidationPredictor:
    manifest_text = binding["manifest_path"]
    _require(
        isinstance(manifest_text, str) and bool(manifest_text),
        "predictor manifest path is invalid",
    )
    predictor = authenticate_validation_predictor_manifest(Path(manifest_text))
    receipt = predictor.validation_data_verification
    expected = {
        "manifest_sha256": predictor.manifest_sha256,
        "validation_data_build_id": receipt.build_id,
        "validation_data_verification_id": receipt.verification_id,
        "candidate_manifest_sha256": receipt.candidate_manifest_sha256,
        "verification_report_sha256": receipt.verification_report_sha256,
        "materialization_code_hashes_sha256": receipt.materialization_code_hashes_sha256,
        "payload_sha256": predictor.registration.sha256,
        "payload_schema_sha256": FROZEN_STAGE6_RAW_SCHEMA_SHA256,
        "payload_rows": predictor.payload_rows,
    }
    for name, observed in expected.items():
        _require(
            binding[name] == observed, f"predictor execution binding differs: {name}"
        )
    _require(
        predictor.manifest_path == Path(manifest_text),
        "authenticated predictor manifest path differs",
    )
    return predictor


def _parse_authenticated_outcome_manifest(
    payload: bytes,
) -> tuple[Mapping[str, RegisteredExecutionPayload], Path]:
    """Parse source bytes authenticated by the frozen execution manifest.

    No public function accepts a caller-supplied outcome-manifest digest.  The
    fixed file is read and hash-checked by the execution front door before its
    bytes can reach this parser.
    """

    value = _json(payload, label="outcome input manifest")
    _require(
        value.get("contract_id") == OUTCOME_CONTRACT_ID
        and isinstance(value.get("inputs"), Mapping),
        "outcome input manifest identity differs",
    )
    records: dict[str, RegisteredExecutionPayload] = {}
    for name, raw in value["inputs"].items():
        _require(isinstance(raw, Mapping), f"invalid outcome manifest record: {name}")
        try:
            records[str(name)] = RegisteredExecutionPayload(
                role=PayloadRole(str(raw["role"])),
                path=Path(str(raw["path"])),
                sha256=str(raw["sha256"]),
                size_bytes=int(raw["size_bytes"]),
                allowed_columns=tuple(str(item) for item in raw["allowed_columns"]),
                date_column=str(raw["date_column"]),
                date_axis=ExecutionDateAxis(str(raw["date_axis"])),
                minimum_date=dt.date.fromisoformat(str(raw["minimum_date"])),
                maximum_date=dt.date.fromisoformat(str(raw["maximum_date"])),
                holding_month_column=raw.get("holding_month_column"),
                trusted_schema_sha256=raw.get("trusted_schema_sha256"),
            )
        except (KeyError, TypeError, ValueError, Stage11AccessError) as exc:
            raise Stage11ExecutionInputError(
                f"invalid outcome input manifest record: {name}"
            ) from exc
    output_root = Path(str(value.get("output_root", "")))
    _require(output_root.is_absolute(), "outcome manifest output root must be absolute")
    _validate_registration_contract(records, DEFAULT_INPUT_NAMES)
    return MappingProxyType(records), output_root


def _output_roots(value: Mapping[str, Any]) -> Mapping[str, Path]:
    base = DATABASE_ROOT / "interim/phase_05_validation/stage_11_execution_v1"
    expected = {
        "outcome_tables": base / "outcome_tables",
        "selector_checkpoints": base / "selector_checkpoints",
        "raw_forecasts": base / "raw_forecasts",
        "execution_results": base / "execution_results",
    }
    result: dict[str, Path] = {}
    for name, fixed in expected.items():
        selected = Path(str(value[name]))
        _require(
            selected == fixed and selected.is_absolute(),
            f"execution output root differs: {name}",
        )
        _require(
            selected.resolve(strict=False) == selected,
            f"execution output root is symlinked: {name}",
        )
        result[name] = selected
    _require(len(set(result.values())) == len(result), "execution output roots collide")
    return MappingProxyType(result)


def authenticate_stage11_execution_inputs(
    stage11_authority: object,
) -> AuthenticatedStage11ExecutionInputs:
    """Open and bind every empirical input through the one frozen manifest."""

    from portfolio_bot.validation.validation_runner import (
        require_live_stage11_validation_unblind_authority,
    )

    # Authentication precedes the frozen-hash and manifest existence checks.
    require_live_stage11_validation_unblind_authority(stage11_authority)
    expected_manifest_sha = FROZEN_EXECUTION_INPUT_MANIFEST_SHA256
    _require(
        isinstance(expected_manifest_sha, str)
        and _HEX64.fullmatch(expected_manifest_sha) is not None,
        "Stage 11 execution inputs remain sealed: no reviewed manifest digest is frozen",
    )
    path, before = _ordinary_file(
        FROZEN_EXECUTION_INPUT_MANIFEST_PATH,
        root=PROJECT_ROOT,
        label="Stage 11 execution input manifest",
    )
    payload = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    _require(
        _stable_identity(before) == _stable_identity(after),
        "execution input manifest changed while read",
    )
    manifest_sha = _sha256_bytes(payload)
    _require(
        manifest_sha == expected_manifest_sha,
        "execution input manifest SHA-256 differs",
    )
    manifest = _validate_manifest(_json(payload, label="execution input manifest"))
    _require(
        payload == _canonical_json_bytes(manifest),
        "execution input manifest is not canonical JSON",
    )

    # Gate B consumes Gate A's immutable, independently verified output; it
    # never substitutes the broader execution authority for Gate A's narrow
    # materialization authority.  This full chain is checked before the
    # twelve-record outcome manifest is opened or any broker is constructed.
    normalized_sources = _authenticate_normalized_sources(
        manifest["normalized_sources"]
    )
    normalized_sources.gate()

    outcome_binding = manifest["outcome_materialization"]
    _validate_project_relative(
        outcome_binding["input_manifest_path"],
        OUTCOME_INPUT_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        label="outcome input manifest",
    )
    _require(
        outcome_binding["contract_id"] == OUTCOME_CONTRACT_ID,
        "outcome materialization contract differs",
    )
    outcome_manifest_sha = _sha(
        outcome_binding["input_manifest_sha256"],
        label="outcome input manifest SHA-256",
    )
    outcome_manifest_payload = _read_bound_file(
        OUTCOME_INPUT_MANIFEST_PATH,
        root=PROJECT_ROOT,
        expected_sha256=outcome_manifest_sha,
        label="outcome input manifest",
    )

    predictor = _authenticate_predictor(manifest["predictor_authority"])
    development_oof = _authenticate_development_oof(manifest["development_selector"])

    action_binding = manifest["corporate_actions"]
    _validate_project_relative(
        action_binding["mapping_manifest_path"],
        FROZEN_MAPPING_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        label="corporate-action mapping manifest",
    )
    _require(
        action_binding["mapping_manifest_id"] == MAPPING_MANIFEST_ID
        and action_binding["mapping_manifest_sha256"]
        == corporate_action_adapter.FROZEN_MAPPING_MANIFEST_SHA256,
        "corporate-action execution binding differs",
    )
    action_authority = authenticate_corporate_action_mapping_manifest(stage11_authority)

    outcome_records, declared_outcome_root = _parse_authenticated_outcome_manifest(
        outcome_manifest_payload,
    )
    _require(
        all(
            outcome_records.get(name) == normalized_sources.records[name]
            for name in NORMALIZED_SOURCE_NAMES
        ),
        "outcome manifest does not bind the exact verified Gate-A sources",
    )
    old_predictor = outcome_records[DEFAULT_INPUT_NAMES.predictor]
    authenticated_record = predictor.registration
    _require(
        old_predictor.path == authenticated_record.path
        and old_predictor.sha256 == authenticated_record.sha256
        and old_predictor.size_bytes == authenticated_record.size_bytes,
        "outcome and scoring manifests do not bind the same predictor payload",
    )
    shared_predictor = RegisteredExecutionPayload(
        role=PayloadRole.VALIDATION_PREDICTOR,
        path=authenticated_record.path,
        sha256=authenticated_record.sha256,
        size_bytes=authenticated_record.size_bytes,
        allowed_columns=authenticated_record.allowed_columns,
        date_column="signal_date",
        date_axis=ExecutionDateAxis.SIGNAL,
        minimum_date=authenticated_record.minimum_signal_date,
        maximum_date=authenticated_record.maximum_signal_date,
        holding_month_column=None,
        trusted_schema_sha256=authenticated_record.trusted_schema_sha256,
    )
    records = dict(outcome_records)
    records.update(normalized_sources.records)
    records[DEFAULT_INPUT_NAMES.predictor] = shared_predictor
    risk_cost_records = build_authenticated_risk_cost_registrations(predictor)
    _require(
        not set(records).intersection(risk_cost_records),
        "verified risk/cost registrations collide with outcome sources",
    )
    records.update(risk_cost_records)
    outcome_projection = {name: records[name] for name in DEFAULT_INPUT_NAMES.values()}
    _validate_registration_contract(outcome_projection, DEFAULT_INPUT_NAMES)
    _require(
        set(records)
        == set(DEFAULT_INPUT_NAMES.values()).union(RISK_COST_REGISTRATION_NAMES),
        "combined execution broker registration set differs",
    )

    outputs = _output_roots(manifest["output_roots"])
    _require(
        declared_outcome_root == outputs["outcome_tables"],
        "outcome materialization output root differs from the execution manifest",
    )
    broker = ValidationExecutionAccessBroker(stage11_authority, records)
    require_live_stage11_validation_unblind_authority(stage11_authority)
    result = AuthenticatedStage11ExecutionInputs(
        manifest_path=path,
        manifest_sha256=manifest_sha,
        manifest_payload_sha256=str(manifest["payload_sha256"]),
        normalized_sources=normalized_sources,
        predictor=predictor,
        development_oof=development_oof,
        corporate_action_authority=action_authority,
        registrations=MappingProxyType(records),
        output_roots=outputs,
        broker=broker,
        _stage11_authority=stage11_authority,
        _manifest_identity=_stable_identity(after),
    )
    result.gate()
    return result


__all__ = [
    "AuthenticatedDevelopmentOOF",
    "AuthenticatedStage11NormalizedSources",
    "AuthenticatedStage11ExecutionInputs",
    "DEVELOPMENT_OOF_NAME",
    "DEVELOPMENT_OOF_ROWS",
    "DEVELOPMENT_OOF_SCHEMA_SHA256",
    "DEVELOPMENT_OOF_SHA256",
    "DEVELOPMENT_OOF_SIZE_BYTES",
    "EXECUTION_INPUT_MANIFEST_ID",
    "EXECUTION_INPUT_MANIFEST_RELATIVE_PATH",
    "EXECUTION_INPUT_MANIFEST_STATUS",
    "FROZEN_EXECUTION_INPUT_MANIFEST_PATH",
    "FROZEN_EXECUTION_INPUT_MANIFEST_SHA256",
    "NORMALIZED_SOURCE_NAMES",
    "SOURCE_PLAN_PAYLOAD_SHA256",
    "SOURCE_PLAN_RELATIVE_PATH",
    "SOURCE_PLAN_SHA256",
    "SOURCE_PRIMARY_POINTER_RELATIVE_PATH",
    "SOURCE_VERIFIED_POINTER_RELATIVE_PATH",
    "Stage11ExecutionInputError",
    "authenticate_stage11_execution_inputs",
]
