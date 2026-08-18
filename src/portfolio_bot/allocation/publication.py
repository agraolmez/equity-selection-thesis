"""Content-addressed primary publication for Stage 10 allocator evidence.

Only a compact synthetic manifest and its completion receipt are published.
The latest-primary pointer is replaced last and authorizes independent
verification only.  It cannot authorize empirical portfolio formation,
validation-period access, Stage 11, or any later stage.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from portfolio_bot.allocation.contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
    Stage10ContractReceipt,
    authenticate_stage10_contract,
    canonical_json_bytes,
    load_policy,
    sha256_file,
)
from portfolio_bot.allocation.evidence import (
    build_synthetic_allocator_evidence,
    synthetic_evidence_sha256,
)
from portfolio_bot.data.secure_publication import (
    PublishedBundle,
    SecureStagingDirectory,
    replace_pointer_last,
)


MANIFEST_NAME = "manifest.json"
SUCCESS_NAME = "_SUCCESS.json"
PRIMARY_FILES = (MANIFEST_NAME, SUCCESS_NAME)
BUILD_ID_PATTERN = re.compile(r"phase04-allocator-[0-9a-f]{24}\Z")

DEPENDENCY_CODE_PATHS = (
    "src/portfolio_bot/data/secure_publication.py",
    "src/portfolio_bot/allocation/__init__.py",
    "src/portfolio_bot/allocation/candidates.py",
    "src/portfolio_bot/allocation/constraints.py",
    "src/portfolio_bot/allocation/contract.py",
    "src/portfolio_bot/allocation/costs.py",
    "src/portfolio_bot/allocation/decision_focused.py",
    "src/portfolio_bot/allocation/domain.py",
    "src/portfolio_bot/allocation/evidence.py",
    "src/portfolio_bot/allocation/fixtures.py",
    "src/portfolio_bot/allocation/independent_verification.py",
    "src/portfolio_bot/allocation/publication.py",
    "src/portfolio_bot/allocation/rank.py",
    "src/portfolio_bot/allocation/risk.py",
    "src/portfolio_bot/allocation/solver.py",
    "scripts/build_stage_10_allocators.py",
    "scripts/verify_stage_10_allocators.py",
)


class AllocatorPublicationError(RuntimeError):
    """Raised when Stage 10 primary publication cannot be authenticated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AllocatorPublicationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _project_relative(path: Path, *, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise AllocatorPublicationError(
            "Stage 10 policy must be beneath project root"
        ) from exc


def _hash_paths(paths: Sequence[str], *, project_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in paths:
        path = project_root / name
        _require(
            path.is_file() and not path.is_symlink(),
            f"Missing or unsafe dependency: {name}",
        )
        result[name] = sha256_file(path)
    return result


@dataclass(frozen=True)
class AllocatorBuildIdentity:
    build_id: str
    content_fingerprint: str
    contract_receipt_sha256: str
    synthetic_evidence_sha256: str
    code_hashes: Mapping[str, str]
    configuration_hashes: Mapping[str, str]


@dataclass(frozen=True)
class PrimaryAllocatorPublication:
    build_id: str
    manifest_path: Path
    pointer_path: Path
    manifest: Mapping[str, Any]
    pointer: Mapping[str, Any]
    bundle: PublishedBundle
    reused_existing: bool


def compute_allocator_build_identity(
    receipt: Stage10ContractReceipt,
    evidence: Mapping[str, Any],
    *,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
) -> AllocatorBuildIdentity:
    """Hash the complete dependency-scoped synthetic implementation identity."""

    policy_file = Path(policy_path)
    if not policy_file.is_absolute():
        policy_file = project_root / policy_file
    policy = load_policy(policy_file)
    policy_relative = _project_relative(policy_file, project_root=project_root)
    configuration_paths = (
        policy_relative,
        str(policy["physical_rules"]["path"]),
        str(policy["methodology"]["source"]),
        str(policy["methodology"]["freeze_receipt"]),
        "pyproject.toml",
    )
    code_hashes = _hash_paths(DEPENDENCY_CODE_PATHS, project_root=project_root)
    configuration_hashes = _hash_paths(
        configuration_paths,
        project_root=project_root,
    )
    evidence_sha = synthetic_evidence_sha256(evidence)
    fingerprint_payload = {
        "schema_version": 1,
        "purpose": "stage10_synthetic_allocator_primary_only",
        "contract_id": receipt.contract_id,
        "policy_id": receipt.policy_id,
        "rules_id": receipt.rules_id,
        "contract_receipt_sha256": receipt.sha256,
        "synthetic_evidence_sha256": evidence_sha,
        "upstream": {
            "stage09_build_id": receipt.stage09_build_id,
            "stage09_verification_id": receipt.stage09_verification_id,
            "stage09_pointer_sha256": receipt.stage09_pointer_sha256,
            "stage09_manifest_sha256": receipt.stage09_manifest_sha256,
            "stage08_build_id": receipt.stage08_build_id,
            "stage08_verification_id": receipt.stage08_verification_id,
            "stage08_pointer_sha256": receipt.stage08_pointer_sha256,
            "stage08_manifest_sha256": receipt.stage08_manifest_sha256,
        },
        "runtime_versions": dict(receipt.runtime_versions),
        "code_hashes": dict(sorted(code_hashes.items())),
        "configuration_hashes": dict(sorted(configuration_hashes.items())),
    }
    fingerprint = _sha256_bytes(canonical_json_bytes(fingerprint_payload))
    identity = AllocatorBuildIdentity(
        build_id=f"phase04-allocator-{fingerprint[:24]}",
        content_fingerprint=fingerprint,
        contract_receipt_sha256=receipt.sha256,
        synthetic_evidence_sha256=evidence_sha,
        code_hashes=code_hashes,
        configuration_hashes=configuration_hashes,
    )
    _require(
        BUILD_ID_PATTERN.fullmatch(identity.build_id) is not None,
        "Malformed allocator build ID",
    )
    return identity


def build_primary_allocator_manifest(
    receipt: Stage10ContractReceipt,
    identity: AllocatorBuildIdentity,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the deterministic primary manifest without publishing it."""

    _require(
        synthetic_evidence_sha256(evidence) == identity.synthetic_evidence_sha256,
        "Synthetic evidence changed after identity computation",
    )
    gates = dict(evidence.get("gates", {}))
    _require(
        gates and all(isinstance(value, bool) and value for value in gates.values()),
        "Executed synthetic evidence gates did not all pass",
    )
    pending_gates = dict(evidence.get("pending_gates", {}))
    _require(
        pending_gates
        == {
            "full_repository_regression": "PENDING_EXTERNAL_TEST_RUN",
            "independent_verification": "PENDING_PRIMARY_PUBLICATION",
        },
        "Primary pending-gate contract differs",
    )
    scope = evidence.get("scope")
    _require(isinstance(scope, Mapping), "Synthetic evidence scope is missing")
    for name in (
        "real_data_payload_rows_read",
        "validation_or_final_rows_read",
        "real_portfolio_weights_created",
        "real_trades_created",
        "real_pnl_or_performance_rows_created",
    ):
        _require(scope.get(name) == 0, f"Stage 10 scope counter differs: {name}")
    upstream_authenticated = bool(
        receipt.status == "AUTHENTICATED_IMPLEMENTATION_ONLY"
        and len(receipt.bound_payloads) == 7
        and all(not item.payload_open_permitted for item in receipt.bound_payloads)
    )
    methodology_authenticated = bool(
        identity.configuration_hashes.get("outputs/methodology_v1_source.md")
        == receipt.methodology_source_sha256
    )
    runtime_authenticated = bool(receipt.runtime_versions) and all(
        bool(name) and bool(version) for name, version in receipt.runtime_versions
    )
    _require(upstream_authenticated, "Upstream authority receipt is not authenticated")
    _require(methodology_authenticated, "Methodology source hash is not authenticated")
    _require(runtime_authenticated, "Runtime receipt is empty or malformed")
    authorizations = {
        "independent_verification_authorized": True,
        "independent_verification_completed": False,
        "stage_11_validation_authorized": False,
        "stage_11_or_later_authorized": False,
        "stage_12_or_later_authorized": False,
        "validation_or_final_test_access_authorized": False,
        "real_portfolio_weights_or_trades_authorized": False,
        "real_pnl_or_performance_authorized": False,
    }
    return {
        "schema_version": 1,
        "contract_id": receipt.contract_id,
        "policy_id": receipt.policy_id,
        "rules_id": receipt.rules_id,
        "build_id": identity.build_id,
        "status": "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION",
        "authorizations": authorizations,
        "bindings": {
            "content_fingerprint": identity.content_fingerprint,
            "contract_receipt_sha256": identity.contract_receipt_sha256,
            "synthetic_evidence_sha256": identity.synthetic_evidence_sha256,
            "code_hashes": dict(sorted(identity.code_hashes.items())),
            "configuration_hashes": dict(sorted(identity.configuration_hashes.items())),
            "runtime_versions": dict(receipt.runtime_versions),
        },
        "upstream_authority": {
            "stage09_build_id": receipt.stage09_build_id,
            "stage09_verification_id": receipt.stage09_verification_id,
            "stage09_pointer_sha256": receipt.stage09_pointer_sha256,
            "stage09_manifest_sha256": receipt.stage09_manifest_sha256,
            "stage08_build_id": receipt.stage08_build_id,
            "stage08_verification_id": receipt.stage08_verification_id,
            "stage08_pointer_sha256": receipt.stage08_pointer_sha256,
            "stage08_manifest_sha256": receipt.stage08_manifest_sha256,
            "bound_payload_identities": [
                item.to_dict() for item in receipt.bound_payloads
            ],
            "upstream_payload_open_permitted": False,
        },
        "contract_receipt": receipt.to_dict(),
        "synthetic_evidence": dict(evidence),
        "hard_gates": {
            "upstream_authority_authenticated": upstream_authenticated,
            "methodology_hash_authenticated": methodology_authenticated,
            "runtime_versions_authenticated": runtime_authenticated,
            **gates,
            "real_data_payload_rows_read": 0,
            "validation_or_final_rows_read": 0,
            "real_portfolio_weights_or_trades_created": 0,
            "real_pnl_or_performance_rows_created": 0,
        },
        "pending_gates": pending_gates,
        "summary": {
            "architecture_count": 3,
            "architecture_ids": [
                "hybrid_alpha_risk_cost_convex",
                "rank_volatility_projected",
                "decision_focused_neural_projected",
            ],
            "evidence_scope": "deterministic_fictitious_synthetic_only",
            "empirical_security_rows_read": 0,
            "empirical_weights_trades_or_pnl_rows_created": 0,
            "independent_verification_pending": True,
            "stage_11_authorized": False,
            "decision_focused_empirical_production_training_ready": False,
            "decision_focused_empirical_production_training_limitation": evidence.get(
                "limitations", {}
            ).get("decision_focused_empirical_production_training"),
        },
    }


def _write_bytes(stage: SecureStagingDirectory, name: str, payload: bytes) -> str:
    with stage.open_exclusive_binary(name) as stream:
        written = stream.write(payload)
        _require(written == len(payload), f"Short Stage 10 publication write: {name}")
    return _sha256_bytes(payload)


def _publication_paths(
    policy: Mapping[str, Any],
    identity: AllocatorBuildIdentity,
    *,
    project_root: Path,
    artifact_root: Path | None,
    pointer_path: Path | None,
) -> tuple[Path, Path, Path]:
    outputs = policy["outputs"]
    if artifact_root is None:
        selected_root = project_root / str(outputs["artifact_root"])
    else:
        selected_root = Path(artifact_root)
    selected_root = selected_root.absolute()
    destination = (
        selected_root / str(outputs["primary_build_namespace"]) / identity.build_id
    )
    if pointer_path is None:
        pointer_name = Path(str(outputs["latest_primary_pointer"])).name
        selected_pointer = selected_root / pointer_name
    else:
        selected_pointer = Path(pointer_path).absolute()
    return selected_root, destination, selected_pointer


def publish_primary_allocator_evidence(
    *,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
    artifact_root: Path | None = None,
    pointer_path: Path | None = None,
    trusted_root: Path | None = None,
) -> PrimaryAllocatorPublication:
    """Authenticate, run synthetic gates, and publish pointer-last primary evidence."""

    receipt = authenticate_stage10_contract(policy_path, project_root=project_root)
    policy = load_policy(policy_path)
    outputs = policy.get("outputs")
    _require(isinstance(outputs, Mapping), "Stage 10 output contract is missing")
    _require(
        tuple(outputs.get("primary_required_files", ())) == PRIMARY_FILES,
        "Stage 10 primary closed file set differs",
    )
    _require(
        outputs.get("empirical_weight_trade_or_pnl_payload_permitted") is False,
        "Stage 10 output contract permits empirical payloads",
    )
    evidence = build_synthetic_allocator_evidence()
    identity = compute_allocator_build_identity(
        receipt,
        evidence,
        policy_path=policy_path,
        project_root=project_root,
    )
    manifest = build_primary_allocator_manifest(receipt, identity, evidence)
    manifest_payload = canonical_json_bytes(manifest)
    success_payload = canonical_json_bytes(
        {
            "schema_version": 1,
            "contract_id": receipt.contract_id,
            "policy_id": receipt.policy_id,
            "build_id": identity.build_id,
            "content_fingerprint": identity.content_fingerprint,
            "status": "PRIMARY_MANIFEST_COMPLETE",
            "manifest_sha256": _sha256_bytes(manifest_payload),
            "closed_files": list(PRIMARY_FILES),
            "empirical_payload_members": 0,
        }
    )
    selected_root, destination, selected_pointer = _publication_paths(
        policy,
        identity,
        project_root=project_root,
        artifact_root=artifact_root,
        pointer_path=pointer_path,
    )
    secure_root = (
        Path(trusted_root).absolute()
        if trusted_root is not None
        else (project_root.absolute() if artifact_root is None else selected_root)
    )

    def reauthenticate() -> None:
        current_receipt = authenticate_stage10_contract(
            policy_path,
            project_root=project_root,
        )
        _require(
            current_receipt.sha256 == receipt.sha256,
            "Stage 10 contract changed during publication",
        )
        current_identity = compute_allocator_build_identity(
            current_receipt,
            evidence,
            policy_path=policy_path,
            project_root=project_root,
        )
        _require(
            current_identity == identity,
            "Stage 10 dependency identity changed during publication",
        )

    with SecureStagingDirectory.create(
        trusted_root=secure_root,
        destination=destination,
        expected_files=PRIMARY_FILES,
        label="Stage 10 primary allocator evidence",
    ) as stage:
        manifest_sha = _write_bytes(stage, MANIFEST_NAME, manifest_payload)
        success_sha = _write_bytes(stage, SUCCESS_NAME, success_payload)
        snapshot = stage.snapshot(
            expected_hashes={MANIFEST_NAME: manifest_sha, SUCCESS_NAME: success_sha}
        )
        bundle = stage.publish(
            expected_hashes=snapshot.hashes,
            pre_publish_gate=reauthenticate,
            post_publish_gate=reauthenticate,
        )

    pointer = {
        "schema_version": 1,
        "contract_id": receipt.contract_id,
        "policy_id": receipt.policy_id,
        "rules_id": receipt.rules_id,
        "build_id": identity.build_id,
        "status": "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION",
        "primary_manifest_path": destination.relative_to(secure_root)
        .joinpath(MANIFEST_NAME)
        .as_posix(),
        "primary_manifest_sha256": manifest_sha,
        "primary_success_sha256": success_sha,
        "synthetic_evidence_sha256": identity.synthetic_evidence_sha256,
        "authorizations": manifest["authorizations"],
    }
    pointer_payload = canonical_json_bytes(pointer)
    pointer_replaced = replace_pointer_last(
        selected_pointer,
        pointer_payload,
        trusted_root=secure_root,
        pre_replace_gate=reauthenticate,
        post_replace_gate=reauthenticate,
        label="Stage 10 latest allocator primary pointer",
    )
    return PrimaryAllocatorPublication(
        build_id=identity.build_id,
        manifest_path=destination / MANIFEST_NAME,
        pointer_path=selected_pointer,
        manifest=manifest,
        pointer=pointer,
        bundle=bundle,
        reused_existing=bundle.reused_existing and not pointer_replaced,
    )


__all__ = [
    "AllocatorBuildIdentity",
    "AllocatorPublicationError",
    "BUILD_ID_PATTERN",
    "DEPENDENCY_CODE_PATHS",
    "MANIFEST_NAME",
    "PRIMARY_FILES",
    "PrimaryAllocatorPublication",
    "SUCCESS_NAME",
    "build_primary_allocator_manifest",
    "compute_allocator_build_identity",
    "publish_primary_allocator_evidence",
]
