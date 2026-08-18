"""Stage 10 authority, input-metadata, and execution-boundary contract.

Stage 10 is deliberately an implementation-only stage.  This module may read
and hash repository metadata (the frozen methodology, policy/rules, verified
pointers, and their primary manifests), but it never resolves or opens an
upstream database payload.  Real security rows, portfolio weights, trades,
P&L, performance statistics, and every 2015-or-later row remain out of scope.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import stat
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = PROJECT_ROOT / "configs/phase_04_models/allocation_policy_v1.toml"

POLICY_ID = "phase04-allocation-policy-v1"
CONTRACT_ID = "phase04-allocation-v1"
RULES_ID = "phase04-allocation-build-rules-v1"
STAGE = 10

FIRST_DEVELOPMENT_SIGNAL_DATE = dt.date(2000, 1, 31)
LAST_DEVELOPMENT_SIGNAL_DATE = dt.date(2014, 10, 31)
REQUIRED_DEVELOPMENT_SIGNAL_DATES = 178

ARCHITECTURE_IDS = (
    "hybrid_alpha_risk_cost_convex",
    "rank_volatility_projected",
    "decision_focused_neural_projected",
)

REQUIRED_RUNTIME_VERSIONS = {
    "cvxpy": "1.9.2",
    "cvxpylayers": "1.2.0",
    "clarabel": "0.11.1",
    "scs": "3.2.11",
    "diffcp": "1.1.9",
}

PERMITTED_REPOSITORY_METADATA = (
    "outputs/methodology_v1_source.md",
    "outputs/methodology_v1_freeze.json",
    "artifacts/phase_04_models/LATEST_SELECTOR_VERIFIED.json",
    (
        "artifacts/phase_04_models/selector_builds/"
        "phase04-selector-results-09a59d61ed58e77d80448a97/manifest.json"
    ),
    "artifacts/phase_04_models/LATEST_RISK_COST_VERIFIED.json",
    (
        "artifacts/phase_04_models/risk_cost_builds/"
        "phase04-risk-cost-8e2fe8061e776ff3aa527d09/manifest.json"
    ),
)

PROHIBITED_PAYLOAD_ROLES = (
    "real_selector_forecasts",
    "real_risk_cost_rows",
    "real_features_or_labels",
    "validation_or_final_data",
    "real_portfolio_weights",
    "real_trades",
    "real_pnl_or_performance",
)

_POLICY_PROHIBITION_FLAGS = (
    "real_security_month_payload_access_permitted",
    "real_selector_forecast_payload_access_permitted",
    "real_risk_cost_payload_access_permitted",
    "real_portfolio_weights_or_trades_permitted",
    "real_pnl_or_performance_permitted",
    "sharpe_or_portfolio_performance_reporting_permitted",
    "selector_refit_or_selection_permitted",
    "ensemble_weight_selection_permitted",
    "decision_focused_real_data_training_permitted",
    "validation_2015_2019_payload_access_permitted",
    "final_2020_2025_payload_access_permitted",
    "stage11_or_later_execution_permitted",
)

_RULE_PROHIBITION_FLAGS = (
    "real_security_month_payload_access_permitted",
    "real_selector_forecast_payload_access_permitted",
    "real_risk_cost_payload_access_permitted",
    "real_portfolio_weight_or_trade_generation_permitted",
    "real_pnl_or_performance_generation_permitted",
    "sharpe_or_portfolio_performance_reporting_permitted",
    "selector_refitting_or_selection_permitted",
    "ensemble_weight_selection_permitted",
    "decision_focused_real_data_training_permitted",
    "validation_2015_2019_payload_access_permitted",
    "final_2020_2025_payload_access_permitted",
    "stage_11_or_later_execution_permitted",
)

# These are metadata identities only.  Stage 10 is not authorized to open any
# of the corresponding external-database files.
_BOUND_PAYLOAD_SPECS = (
    (9, "interim", "selector_oof_forecasts.parquet", 172_969),
    (9, "processed", "selector_search_receipt.json", None),
    (9, "processed", "selector_shortlist.parquet", 9),
    (8, "interim", "development_risk_cost_audit.parquet", 1_263_048),
    (8, "processed", "development_risk_cost_inputs.parquet", 286_628),
    (8, "processed", "development_factor_covariance.parquet", 6_408),
    (8, "processed", "model_parameters.json", None),
)


class AllocationContractError(RuntimeError):
    """Raised when Stage 10 authority, metadata, or scope differs."""


class AllocationScopeError(AllocationContractError):
    """Raised when an empirical action attempts to cross the Stage 10 boundary."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AllocationContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
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


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _safe_project_file(
    value: str | Path,
    *,
    project_root: Path,
    label: str,
) -> Path:
    """Return one regular file beneath ``project_root`` without following links."""

    root = _lexical_absolute(project_root)
    candidate = Path(value).expanduser()
    path = _lexical_absolute(candidate if candidate.is_absolute() else root / candidate)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise AllocationContractError(
            f"Stage 10 {label} escaped the project root"
        ) from exc
    require(
        relative.parts and ".." not in relative.parts, f"Unsafe Stage 10 {label} path"
    )
    require(root.is_dir() and not root.is_symlink(), "Unsafe Stage 10 project root")
    current = root
    for part in relative.parts:
        current = current / part
        require(
            not current.is_symlink(), f"Symlinked Stage 10 {label} path is prohibited"
        )
    try:
        observed = path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise AllocationContractError(f"Missing Stage 10 {label}: {path}") from exc
    require(stat.S_ISREG(observed.st_mode), f"Stage 10 {label} is not a regular file")
    return path


def _authenticate_project_file(
    value: object,
    expected_sha256: object,
    *,
    project_root: Path,
    label: str,
) -> Path:
    path = _safe_project_file(str(value), project_root=project_root, label=label)
    require(sha256_file(path) == str(expected_sha256), f"Stage 10 {label} hash differs")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    require(isinstance(value, dict), f"Stage 10 {label} is not a JSON object")
    return value


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    resolved = _lexical_absolute(path)
    with resolved.open("rb") as handle:
        value = tomllib.load(handle)
    require(isinstance(value, dict), "Stage 10 policy is not a TOML table")
    return value


def load_rules(
    policy: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    binding = policy.get("physical_rules")
    require(isinstance(binding, Mapping), "Stage 10 physical-rules binding is missing")
    path = _authenticate_project_file(
        binding.get("path"),
        binding.get("sha256"),
        project_root=project_root,
        label="physical rules",
    )
    value = _read_json(path, label="physical rules")
    require(
        value.get("rules_id") == binding.get("required_rules_id"),
        "Stage 10 rules ID differs",
    )
    return value


def inspect_runtime_versions(
    *,
    version_reader: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str | None]:
    """Read distribution metadata without importing solver frameworks."""

    observed: dict[str, str | None] = {}
    for distribution in REQUIRED_RUNTIME_VERSIONS:
        try:
            observed[distribution] = version_reader(distribution)
        except importlib.metadata.PackageNotFoundError:
            observed[distribution] = None
    return observed


def validate_runtime_versions(
    *,
    version_reader: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str | None]:
    observed = inspect_runtime_versions(version_reader=version_reader)
    require(sys.version_info >= (3, 11), "Stage 10 requires Python 3.11 or later")
    for name, expected in REQUIRED_RUNTIME_VERSIONS.items():
        require(
            observed[name] == expected,
            f"Stage 10 runtime version differs for {name}: "
            f"expected {expected}, observed {observed[name]}",
        )
    return observed


@dataclass(frozen=True, slots=True)
class BoundaryCounters:
    """Auditable counters for actions that Stage 10 must leave at zero."""

    real_data_payload_rows_read: int = 0
    validation_or_final_rows_read: int = 0
    real_portfolio_weights_created: int = 0
    real_trades_created: int = 0
    real_pnl_or_performance_rows_created: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "real_data_payload_rows_read": self.real_data_payload_rows_read,
            "validation_or_final_rows_read": self.validation_or_final_rows_read,
            "real_portfolio_weights_created": self.real_portfolio_weights_created,
            "real_trades_created": self.real_trades_created,
            "real_pnl_or_performance_rows_created": self.real_pnl_or_performance_rows_created,
        }


def enforce_stage10_boundary(
    counters: BoundaryCounters = BoundaryCounters(),
    *,
    synthetic_fixture: bool,
    selector_refit_or_selection: bool = False,
    ensemble_weight_selection: bool = False,
    decision_focused_real_data_training: bool = False,
) -> None:
    """Fail closed unless an execution is synthetic and empirically empty."""

    if not synthetic_fixture:
        raise AllocationScopeError("Stage 10 execution must use a synthetic fixture")
    for name, value in counters.to_dict().items():
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            raise AllocationScopeError(f"Stage 10 prohibits nonzero {name}")
    if selector_refit_or_selection:
        raise AllocationScopeError("Stage 10 prohibits selector refitting or selection")
    if ensemble_weight_selection:
        raise AllocationScopeError("Stage 10 prohibits ensemble-weight selection")
    if decision_focused_real_data_training:
        raise AllocationScopeError(
            "Stage 10 prohibits decision-focused real-data training"
        )


@dataclass(frozen=True, slots=True)
class BoundPayloadIdentity:
    """Hash-bound upstream identity; never a permission to open the payload."""

    source_stage: int
    lifecycle: str
    name: str
    logical_path: str
    rows: int | None
    row_groups: int | None
    size_bytes: int
    sha256: str
    schema_sha256: str | None
    last_permitted_signal_date: str
    payload_open_permitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_stage": self.source_stage,
            "lifecycle": self.lifecycle,
            "name": self.name,
            "logical_path": self.logical_path,
            "rows": self.rows,
            "row_groups": self.row_groups,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "schema_sha256": self.schema_sha256,
            "last_permitted_signal_date": self.last_permitted_signal_date,
            "payload_open_permitted": self.payload_open_permitted,
        }


@dataclass(frozen=True, slots=True)
class Stage10ContractReceipt:
    """Small immutable authority receipt suitable for primary publication."""

    schema_version: int
    status: str
    stage: int
    policy_id: str
    contract_id: str
    rules_id: str
    policy_sha256: str
    rules_sha256: str
    methodology_source_sha256: str
    methodology_freeze_sha256: str
    stage09_build_id: str
    stage09_verification_id: str
    stage09_pointer_sha256: str
    stage09_manifest_sha256: str
    stage08_build_id: str
    stage08_verification_id: str
    stage08_pointer_sha256: str
    stage08_manifest_sha256: str
    first_development_signal_date: str
    last_development_signal_date: str
    bound_payloads: tuple[BoundPayloadIdentity, ...]
    runtime_versions: tuple[tuple[str, str], ...]
    prohibitions: tuple[str, ...]
    counters: BoundaryCounters

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "stage": self.stage,
            "policy_id": self.policy_id,
            "contract_id": self.contract_id,
            "rules_id": self.rules_id,
            "policy_sha256": self.policy_sha256,
            "rules_sha256": self.rules_sha256,
            "methodology_source_sha256": self.methodology_source_sha256,
            "methodology_freeze_sha256": self.methodology_freeze_sha256,
            "stage09_build_id": self.stage09_build_id,
            "stage09_verification_id": self.stage09_verification_id,
            "stage09_pointer_sha256": self.stage09_pointer_sha256,
            "stage09_manifest_sha256": self.stage09_manifest_sha256,
            "stage08_build_id": self.stage08_build_id,
            "stage08_verification_id": self.stage08_verification_id,
            "stage08_pointer_sha256": self.stage08_pointer_sha256,
            "stage08_manifest_sha256": self.stage08_manifest_sha256,
            "first_development_signal_date": self.first_development_signal_date,
            "last_development_signal_date": self.last_development_signal_date,
            "bound_payloads": [item.to_dict() for item in self.bound_payloads],
            "runtime_versions": dict(self.runtime_versions),
            "prohibitions": list(self.prohibitions),
            "counters": self.counters.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def _require_boolean_mapping(
    observed: object,
    expected: Mapping[str, bool],
    *,
    label: str,
) -> None:
    require(isinstance(observed, Mapping), f"Stage 10 {label} map is missing")
    for name, value in expected.items():
        require(observed.get(name) is value, f"Stage 10 {label} differs for {name}")


def _validate_methodology(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    methodology = policy.get("methodology")
    require(isinstance(methodology, Mapping), "Stage 10 methodology binding is missing")
    source_path = _authenticate_project_file(
        methodology.get("source"),
        methodology.get("source_sha256"),
        project_root=project_root,
        label="methodology source",
    )
    freeze_path = _authenticate_project_file(
        methodology.get("freeze_receipt"),
        methodology.get("freeze_receipt_sha256"),
        project_root=project_root,
        label="methodology freeze receipt",
    )
    freeze = _read_json(freeze_path, label="methodology freeze receipt")
    require(
        freeze.get("status") == "frozen_preimplementation"
        and freeze.get("methodology_version") == "v1"
        and freeze.get("empirical_results_seen") is False,
        "Stage 10 methodology freeze state differs",
    )
    files = freeze.get("files")
    require(
        isinstance(files, list), "Stage 10 methodology freeze file registry is missing"
    )
    source_entries = [
        item
        for item in files
        if isinstance(item, Mapping) and item.get("path") == methodology.get("source")
    ]
    require(
        len(source_entries) == 1, "Stage 10 methodology source is not uniquely frozen"
    )
    require(
        source_entries[0].get("sha256") == methodology.get("source_sha256"),
        "Stage 10 methodology freeze/source hash differs",
    )
    return source_path, freeze_path, freeze


def _validate_scope_and_inputs(
    policy: Mapping[str, Any], rules: Mapping[str, Any]
) -> None:
    scope = policy.get("scope")
    require(isinstance(scope, Mapping), "Stage 10 policy scope is missing")
    require(
        scope.get("first_development_signal_date")
        == FIRST_DEVELOPMENT_SIGNAL_DATE.isoformat()
        and scope.get("last_development_signal_date")
        == LAST_DEVELOPMENT_SIGNAL_DATE.isoformat()
        and scope.get("required_development_signal_dates")
        == REQUIRED_DEVELOPMENT_SIGNAL_DATES,
        "Stage 10 development date scope differs",
    )
    require(
        scope.get("upstream_metadata_authentication_permitted") is True,
        "Metadata authentication is not allowed",
    )
    require(
        scope.get("synthetic_fixture_execution_permitted") is True,
        "Synthetic Stage 10 tests are not allowed",
    )
    for name in _POLICY_PROHIBITION_FLAGS:
        require(
            scope.get(name) is False, f"Stage 10 policy unexpectedly permits {name}"
        )

    rule_scope = rules.get("scope")
    require(isinstance(rule_scope, Mapping), "Stage 10 rule scope is missing")
    require(
        rule_scope.get("stage") == STAGE
        and rule_scope.get("first_development_signal_date")
        == FIRST_DEVELOPMENT_SIGNAL_DATE.isoformat()
        and rule_scope.get("last_development_signal_date")
        == LAST_DEVELOPMENT_SIGNAL_DATE.isoformat()
        and rule_scope.get("development_signal_date_count")
        == REQUIRED_DEVELOPMENT_SIGNAL_DATES,
        "Stage 10 rule date scope differs",
    )
    require(
        rule_scope.get("upstream_metadata_authentication_permitted") is True,
        "Rule metadata authentication is not allowed",
    )
    require(
        rule_scope.get("synthetic_fixture_execution_permitted") is True,
        "Rule synthetic execution is not allowed",
    )
    for name in _RULE_PROHIBITION_FLAGS:
        require(
            rule_scope.get(name) is False, f"Stage 10 rules unexpectedly permit {name}"
        )

    inputs = policy.get("inputs")
    require(isinstance(inputs, Mapping), "Stage 10 input boundary is missing")
    require(
        inputs.get("only_closed_allowlist_permitted") is True
        and inputs.get("upstream_payload_open_permitted") is False
        and inputs.get("unregistered_input_permitted") is False
        and inputs.get(
            "synthetic_fixtures_must_be_generated_without_real_security_identifiers_or_values"
        )
        is True,
        "Stage 10 input boundary differs",
    )
    require(
        tuple(inputs.get("permitted_repository_metadata", ()))
        == PERMITTED_REPOSITORY_METADATA,
        "Stage 10 metadata allowlist differs",
    )
    require(
        tuple(inputs.get("prohibited_payload_roles", ())) == PROHIBITED_PAYLOAD_ROLES,
        "Stage 10 prohibited payload roles differ",
    )


def _validate_architecture_and_runtime_bindings(
    policy: Mapping[str, Any], rules: Mapping[str, Any]
) -> None:
    architectures = policy.get("architectures")
    require(
        isinstance(architectures, Mapping), "Stage 10 architecture registry is missing"
    )
    require(
        tuple(architectures.get("ordered_ids", ())) == ARCHITECTURE_IDS,
        "Stage 10 architecture order differs",
    )
    require(
        architectures.get("confirmatory_primary_id") == ARCHITECTURE_IDS[0]
        and architectures.get("challengers_may_replace_primary") is False,
        "Stage 10 confirmatory architecture role differs",
    )
    rule_architectures = rules.get("architectures")
    require(
        isinstance(rule_architectures, Mapping),
        "Stage 10 rule architecture registry is missing",
    )
    observed_ids = (
        rule_architectures["confirmatory_primary"]["id"],
        rule_architectures["rank_based_challenger"]["id"],
        rule_architectures["decision_focused_ml_challenger"]["id"],
    )
    require(
        observed_ids == ARCHITECTURE_IDS, "Stage 10 policy/rule architecture IDs differ"
    )

    policy_runtime = policy.get("runtime")
    rule_runtime = rules.get("runtime")
    require(
        isinstance(policy_runtime, Mapping) and isinstance(rule_runtime, Mapping),
        "Stage 10 runtime binding is missing",
    )
    require(
        dict(policy_runtime.get("required_versions", {})) == REQUIRED_RUNTIME_VERSIONS
        and dict(rule_runtime.get("required_versions", {}))
        == REQUIRED_RUNTIME_VERSIONS,
        "Stage 10 runtime version binding differs",
    )
    require(
        policy_runtime.get("forward_solver")
        == rule_runtime.get("forward_solver")
        == "CLARABEL"
        and policy_runtime.get("differentiable_cone_solver")
        == rule_runtime.get("differentiable_cone_solver")
        == "SCS"
        and policy_runtime.get("forward_solver_substitution_permitted") is False
        and rule_runtime.get("forward_solver_substitution_permitted") is False,
        "Stage 10 solver/runtime boundary differs",
    )


def _validate_cross_bindings(
    policy: Mapping[str, Any], rules: Mapping[str, Any]
) -> None:
    methodology = policy.get("methodology")
    scientific = rules.get("scientific_authority")
    require(
        isinstance(methodology, Mapping) and isinstance(scientific, Mapping),
        "Stage 10 scientific-authority binding is missing",
    )
    require(
        scientific.get("methodology_source") == methodology.get("source")
        and scientific.get("methodology_source_sha256")
        == methodology.get("source_sha256"),
        "Stage 10 policy/rule methodology binding differs",
    )
    authority = policy.get("authority")
    rule_authority = rules.get("upstream_authority")
    require(
        isinstance(authority, Mapping) and isinstance(rule_authority, Mapping),
        "Stage 10 authority binding is missing",
    )
    require(
        rule_authority.get("sole_immediate_authority")
        == authority.get("verified_stage09_pointer")
        and rule_authority.get("sole_immediate_authority_sha256")
        == authority.get("verified_stage09_pointer_sha256")
        and rule_authority.get("stage09_primary_manifest")
        == authority.get("required_stage09_primary_manifest")
        and rule_authority.get("stage09_primary_manifest_sha256")
        == authority.get("required_stage09_primary_manifest_sha256")
        and rule_authority.get("required_stage08_pointer")
        == authority.get("verified_stage08_pointer")
        and rule_authority.get("required_stage08_pointer_sha256")
        == authority.get("verified_stage08_pointer_sha256")
        and rule_authority.get("stage08_primary_manifest")
        == authority.get("required_stage08_primary_manifest")
        and rule_authority.get("stage08_primary_manifest_sha256")
        == authority.get("required_stage08_primary_manifest_sha256"),
        "Stage 10 policy/rule upstream hashes differ",
    )
    require(
        rule_authority.get("unregistered_input_permitted") is False
        and rule_authority.get("upstream_payload_open_permitted") is False
        and rule_authority.get("hash_bypass_permitted") is False
        and rule_authority.get("symlinked_authority_path_permitted") is False
        and authority.get("hash_bypass_permitted") is False
        and authority.get("symlinked_authority_path_permitted") is False,
        "Stage 10 upstream safety boundary differs",
    )


def validate_contract(
    policy_path: Path = DEFAULT_POLICY,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate frozen Stage 10 configuration without touching data payloads."""

    policy_file = _safe_project_file(
        policy_path, project_root=project_root, label="policy"
    )
    policy = load_policy(policy_file)
    require(
        policy.get("schema_version") == 1
        and policy.get("policy_id") == POLICY_ID
        and policy.get("contract_id") == CONTRACT_ID
        and policy.get("stage") == STAGE,
        "Stage 10 policy identity differs",
    )
    require(
        policy.get("status")
        == "frozen_definition_only_before_allocator_implementation",
        "Stage 10 policy status differs",
    )
    rules = load_rules(policy, project_root=project_root)
    require(
        rules.get("schema_version") == 1 and rules.get("rules_id") == RULES_ID,
        "Stage 10 rules identity differs",
    )
    _validate_methodology(policy, project_root=project_root)
    _validate_scope_and_inputs(policy, rules)
    _validate_architecture_and_runtime_bindings(policy, rules)
    _validate_cross_bindings(policy, rules)
    return policy, rules


def _authenticate_stage09(
    policy: Mapping[str, Any],
    rules: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = policy["authority"]
    pointer_path = _authenticate_project_file(
        authority["verified_stage09_pointer"],
        authority["verified_stage09_pointer_sha256"],
        project_root=project_root,
        label="Stage 9 verified pointer",
    )
    pointer = _read_json(pointer_path, label="Stage 9 verified pointer")
    require(
        pointer.get("schema_version") == 1
        and pointer.get("status") == authority["required_stage09_status"]
        and pointer.get("build_id") == authority["required_stage09_build_id"]
        and pointer.get("verification_id")
        == authority["required_stage09_verification_id"]
        and pointer.get("contract_id") == "phase04-selector-v1"
        and pointer.get("policy_id") == "phase04-selector-policy-v1",
        "Stage 9 verified authority identity differs",
    )
    required_authorizations = rules["upstream_authority"][
        "required_stage09_authorizations"
    ]
    _require_boolean_mapping(
        pointer.get("authorizations"),
        required_authorizations,
        label="Stage 9 authorizations",
    )
    require(
        pointer["authorizations"].get(authority["required_stage09_authorization"])
        is True
        and pointer["authorizations"].get(
            authority["required_stage09_selector_authorization"]
        )
        is True,
        "Stage 9 pointer does not authorize Stage 10 architecture work",
    )
    require(
        pointer.get("primary_manifest_sha256")
        == authority["required_stage09_primary_manifest_sha256"],
        "Stage 9 pointer/manifest binding differs",
    )
    manifest_path = _authenticate_project_file(
        authority["required_stage09_primary_manifest"],
        authority["required_stage09_primary_manifest_sha256"],
        project_root=project_root,
        label="Stage 9 primary manifest",
    )
    manifest = _read_json(manifest_path, label="Stage 9 primary manifest")
    require(
        manifest.get("build_id") == authority["required_stage09_build_id"]
        and manifest.get("contract_id") == "phase04-selector-v1"
        and manifest.get("policy_id") == "phase04-selector-policy-v1"
        and manifest.get("status") == "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION",
        "Stage 9 primary manifest identity differs",
    )
    require(
        manifest.get("inputs", {}).get("stage08_authority_build_id")
        == authority["required_stage08_build_id"]
        and manifest.get("inputs", {}).get("stage08_authority_verification_id")
        == authority["required_stage08_verification_id"]
        and manifest.get("inputs", {}).get("validation_or_final_rows_read") == 0,
        "Stage 9 manifest upstream or time boundary differs",
    )
    summary = manifest.get("summary", {})
    require(
        summary.get("configurations") == 346
        and summary.get("oof_rows") == 172_969
        and summary.get("present_oof_outcomes") == 172_799
        and summary.get("shortlisted_configurations") == 9,
        "Stage 9 manifest summary differs",
    )
    return pointer, manifest


def _authenticate_stage08(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = policy["authority"]
    pointer_path = _authenticate_project_file(
        authority["verified_stage08_pointer"],
        authority["verified_stage08_pointer_sha256"],
        project_root=project_root,
        label="Stage 8 verified pointer",
    )
    pointer = _read_json(pointer_path, label="Stage 8 verified pointer")
    require(
        pointer.get("schema_version") == 1
        and pointer.get("status") == "VERIFIED_PASSED"
        and pointer.get("build_id") == authority["required_stage08_build_id"]
        and pointer.get("verification_id")
        == authority["required_stage08_verification_id"]
        and pointer.get("contract_id") == "phase04-risk-cost-v1"
        and pointer.get("policy_id") == "phase04-risk-cost-policy-v1",
        "Stage 8 verified input identity differs",
    )
    require(
        pointer.get("primary_manifest_sha256")
        == authority["required_stage08_primary_manifest_sha256"],
        "Stage 8 pointer/manifest binding differs",
    )
    _require_boolean_mapping(
        pointer.get("authorizations"),
        {
            "independent_verification_completed": True,
            "stage_09_model_training_authorized": True,
            "stage_10_or_later_authorized": False,
            "models_portfolios_pnl_or_backtests_authorized": False,
            "validation_or_final_test_access_authorized": False,
        },
        label="Stage 8 authorizations",
    )
    manifest_path = _authenticate_project_file(
        authority["required_stage08_primary_manifest"],
        authority["required_stage08_primary_manifest_sha256"],
        project_root=project_root,
        label="Stage 8 primary manifest",
    )
    manifest = _read_json(manifest_path, label="Stage 8 primary manifest")
    require(
        manifest.get("build_id") == authority["required_stage08_build_id"]
        and manifest.get("contract_id") == "phase04-risk-cost-v1"
        and manifest.get("policy_id") == "phase04-risk-cost-policy-v1"
        and manifest.get("status") == "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION",
        "Stage 8 primary manifest identity differs",
    )
    gates = manifest.get("hard_gates", {})
    summary = manifest.get("summary", {})
    require(
        gates.get("post_signal_return_rows_used") == 0
        and gates.get("post_signal_factor_rows_used") == 0
        and gates.get("label_payload_rows_read") == 0
        and gates.get("processed_rows") == 286_628
        and gates.get("audit_rows") == 1_263_048
        and gates.get("factor_covariance_rows") == 6_408,
        "Stage 8 primary time or row-count gates differ",
    )
    require(
        summary.get("signal_dates") == REQUIRED_DEVELOPMENT_SIGNAL_DATES
        and summary.get("processed_rows") == 286_628
        and summary.get("audit_rows") == 1_263_048
        and summary.get("factor_covariance_rows") == 6_408,
        "Stage 8 primary summary differs",
    )
    return pointer, manifest


def _safe_relative_path(value: object, *, label: str) -> Path:
    path = Path(str(value))
    require(
        path.parts and not path.is_absolute() and ".." not in path.parts,
        f"Unsafe Stage 10 logical payload path for {label}",
    )
    return path


def _payload_identity(
    manifest: Mapping[str, Any],
    *,
    stage: int,
    lifecycle: str,
    name: str,
    expected_rows: int | None,
) -> BoundPayloadIdentity:
    outputs = manifest.get("outputs")
    require(isinstance(outputs, Mapping), f"Stage {stage} manifest outputs are missing")
    bundle = outputs.get(lifecycle)
    require(
        isinstance(bundle, Mapping),
        f"Stage {stage} {lifecycle} output bundle is missing",
    )
    files = bundle.get("files")
    require(
        isinstance(files, Mapping),
        f"Stage {stage} {lifecycle} file registry is missing",
    )
    entry = files.get(name)
    require(isinstance(entry, Mapping), f"Stage {stage} manifest omitted {name}")
    base = _safe_relative_path(bundle.get("path"), label=f"Stage {stage} {lifecycle}")
    relative = _safe_relative_path(entry.get("path"), label=name)
    rows = entry.get("rows")
    row_groups = entry.get("row_groups")
    if expected_rows is None:
        require(
            rows is None and row_groups is None,
            f"Stage {stage} JSON receipt shape differs for {name}",
        )
    else:
        require(rows == expected_rows, f"Stage {stage} row count differs for {name}")
        require(
            isinstance(row_groups, int) and row_groups > 0,
            f"Stage {stage} row groups differ for {name}",
        )
    sha = entry.get("sha256")
    schema_sha = entry.get("schema_sha256")
    require(
        isinstance(sha, str) and len(sha) == 64,
        f"Stage {stage} SHA is invalid for {name}",
    )
    if name.endswith(".parquet"):
        require(
            isinstance(schema_sha, str) and len(schema_sha) == 64,
            f"Stage {stage} schema SHA is invalid for {name}",
        )
    else:
        require(schema_sha is None, f"Stage {stage} JSON unexpectedly has a schema SHA")
    size = entry.get("size_bytes")
    require(
        isinstance(size, int) and size > 0, f"Stage {stage} size differs for {name}"
    )
    return BoundPayloadIdentity(
        source_stage=stage,
        lifecycle=lifecycle,
        name=name,
        logical_path=(base / relative).as_posix(),
        rows=rows,
        row_groups=row_groups,
        size_bytes=size,
        sha256=sha,
        schema_sha256=schema_sha,
        last_permitted_signal_date=LAST_DEVELOPMENT_SIGNAL_DATE.isoformat(),
        payload_open_permitted=False,
    )


def authenticate_stage10_contract(
    policy_path: Path = DEFAULT_POLICY,
    *,
    project_root: Path = PROJECT_ROOT,
    version_reader: Callable[[str], str] = importlib.metadata.version,
) -> Stage10ContractReceipt:
    """Authenticate Stage 10 metadata and return a publication-safe receipt.

    The function intentionally has no database-root argument and no payload
    reader.  All payload information comes from already hash-bound repository
    manifests.
    """

    policy, rules = validate_contract(policy_path, project_root=project_root)
    runtime = validate_runtime_versions(version_reader=version_reader)
    stage09_pointer, stage09_manifest = _authenticate_stage09(
        policy, rules, project_root=project_root
    )
    stage08_pointer, stage08_manifest = _authenticate_stage08(
        policy, project_root=project_root
    )
    payloads: list[BoundPayloadIdentity] = []
    for stage, lifecycle, name, expected_rows in _BOUND_PAYLOAD_SPECS:
        manifest = stage09_manifest if stage == 9 else stage08_manifest
        payloads.append(
            _payload_identity(
                manifest,
                stage=stage,
                lifecycle=lifecycle,
                name=name,
                expected_rows=expected_rows,
            )
        )
    require(
        len({item.name for item in payloads}) == len(payloads),
        "Stage 10 payload allowlist has duplicate names",
    )

    policy_file = _safe_project_file(
        policy_path, project_root=project_root, label="policy"
    )
    rules_file = _safe_project_file(
        policy["physical_rules"]["path"],
        project_root=project_root,
        label="physical rules",
    )
    methodology = policy["methodology"]
    authority = policy["authority"]
    counters = BoundaryCounters()
    require(
        all(value == 0 for value in counters.to_dict().values()),
        "Stage 10 boundary counters differ",
    )
    return Stage10ContractReceipt(
        schema_version=1,
        status="AUTHENTICATED_IMPLEMENTATION_ONLY",
        stage=STAGE,
        policy_id=POLICY_ID,
        contract_id=CONTRACT_ID,
        rules_id=RULES_ID,
        policy_sha256=sha256_file(policy_file),
        rules_sha256=sha256_file(rules_file),
        methodology_source_sha256=str(methodology["source_sha256"]),
        methodology_freeze_sha256=str(methodology["freeze_receipt_sha256"]),
        stage09_build_id=str(stage09_pointer["build_id"]),
        stage09_verification_id=str(stage09_pointer["verification_id"]),
        stage09_pointer_sha256=str(authority["verified_stage09_pointer_sha256"]),
        stage09_manifest_sha256=str(
            authority["required_stage09_primary_manifest_sha256"]
        ),
        stage08_build_id=str(stage08_pointer["build_id"]),
        stage08_verification_id=str(stage08_pointer["verification_id"]),
        stage08_pointer_sha256=str(authority["verified_stage08_pointer_sha256"]),
        stage08_manifest_sha256=str(
            authority["required_stage08_primary_manifest_sha256"]
        ),
        first_development_signal_date=FIRST_DEVELOPMENT_SIGNAL_DATE.isoformat(),
        last_development_signal_date=LAST_DEVELOPMENT_SIGNAL_DATE.isoformat(),
        bound_payloads=tuple(payloads),
        runtime_versions=tuple(
            (name, str(runtime[name])) for name in REQUIRED_RUNTIME_VERSIONS
        ),
        prohibitions=PROHIBITED_PAYLOAD_ROLES,
        counters=counters,
    )


build_contract_receipt = authenticate_stage10_contract


__all__ = [
    "ARCHITECTURE_IDS",
    "AllocationContractError",
    "AllocationScopeError",
    "BOUND_PAYLOAD_SPECS",
    "BoundaryCounters",
    "BoundPayloadIdentity",
    "CONTRACT_ID",
    "DEFAULT_POLICY",
    "FIRST_DEVELOPMENT_SIGNAL_DATE",
    "LAST_DEVELOPMENT_SIGNAL_DATE",
    "POLICY_ID",
    "PROHIBITED_PAYLOAD_ROLES",
    "PROJECT_ROOT",
    "REQUIRED_DEVELOPMENT_SIGNAL_DATES",
    "REQUIRED_RUNTIME_VERSIONS",
    "RULES_ID",
    "STAGE",
    "Stage10ContractReceipt",
    "authenticate_stage10_contract",
    "build_contract_receipt",
    "canonical_json_bytes",
    "enforce_stage10_boundary",
    "inspect_runtime_versions",
    "load_policy",
    "load_rules",
    "sha256_file",
    "validate_contract",
    "validate_runtime_versions",
]

# Public immutable view of the closed payload-name/lifecycle contract.  Hashes
# and schemas are taken only from the already authenticated primary manifests.
BOUND_PAYLOAD_SPECS = _BOUND_PAYLOAD_SPECS
