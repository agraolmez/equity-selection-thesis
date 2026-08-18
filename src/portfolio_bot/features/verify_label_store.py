"""Independent artifact boundary for the frozen Stage 7 label store.

This module deliberately does not import the Stage 7 contract validator,
input resolver, label mathematics, return router, materializer, quality
builder, or publisher.  It reconstructs the reviewed byte contract itself,
authenticates the exact six inputs and every primary output, and supplies a
narrow callback boundary for an independently implemented numerical
recomputation engine.

Passing artifact checks alone never grants downstream authority.  A verified
pointer can be published only when a recomputation callback explicitly marks
its evidence complete, reports every frozen hard gate, and returns no
differences.
"""

from __future__ import annotations

import hashlib
import json
import datetime as dt
import numbers
import os
import re
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    SecurePublicationError,
    SecureStagingDirectory,
    replace_pointer_last,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = PROJECT_ROOT / "configs/phase_03_features/label_store_policy_v1.toml"
REVIEWED_POLICY_SHA256 = (
    "1f6f5649cfd1959a707c6e04465b70a334b1fbc4b32e9b81158baf521c2abcad"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BUILD_ID = re.compile(r"phase03-labels-[0-9a-f]{24}\Z")
_VERIFY_ID = re.compile(r"verify-[0-9a-f]{16}\Z")
_SAFE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*\Z")

_PRIMARY_STATUS = "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
_VERIFIED_STATUS = "VERIFIED_PASSED"
_PRIMARY_AUTHORIZATIONS = {
    "independent_verification_required": True,
    "models_portfolios_pnl_or_backtests_authorized": False,
    "stage_08_risk_and_cost_models_authorized": False,
    "stage_09_model_training_authorized": False,
    "validation_or_final_test_access_authorized": False,
}
_VERIFIED_AUTHORIZATIONS = {
    "independent_verification_completed": True,
    "models_portfolios_pnl_or_backtests_authorized": False,
    "stage_08_risk_and_cost_models_authorized": True,
    "stage_09_model_training_authorized": False,
    "validation_or_final_test_access_authorized": False,
}
_PRIMARY_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "policy_id",
        "build_id",
        "status",
        "bindings",
        "inputs",
        "outputs",
        "summary",
        "authorizations",
    }
)
_BINDING_KEYS = frozenset(
    {
        "content_fingerprint",
        "configuration_hashes",
        "code_hashes",
        "input_receipts_sha256",
    }
)
_INPUT_KEYS = frozenset({"input_builds", "table_receipts"})
_OUTPUT_KEYS = frozenset({"path", "success_sha256", "files"})
_FILE_RECEIPT_KEYS = frozenset(
    {"path", "sha256", "size_bytes", "rows", "row_groups", "schema_sha256"}
)
_SUCCESS_LOCATIONS = {
    "interim": ("interim_label_store", "PRIMARY_INTERIM_COMPLETE"),
    "processed": ("processed_label_store", "PRIMARY_PROCESSED_COMPLETE"),
    "quality": ("label_store_quality", "PRIMARY_QUALITY_PASSED"),
}
_PRIMARY_CODE_PATHS = frozenset(
    {
        "src/portfolio_bot/data/secure_publication.py",
        "src/portfolio_bot/features/phase_03_contract.py",
        "src/portfolio_bot/features/label_store_contract.py",
        "src/portfolio_bot/features/label_store_inputs.py",
        "src/portfolio_bot/features/label_math.py",
        "src/portfolio_bot/features/label_return_router.py",
        "src/portfolio_bot/features/label_store_materialization.py",
        "src/portfolio_bot/features/label_store_quality.py",
        "src/portfolio_bot/features/label_store_publication.py",
        "scripts/build_stage_07_label_store.py",
    }
)
_VERIFICATION_CODE_PATHS = (
    "scripts/verify_stage_07_label_store.py",
    "src/portfolio_bot/features/stage7_independent_math.py",
    "src/portfolio_bot/features/stage7_independent_recompute.py",
    "src/portfolio_bot/features/verify_label_store.py",
)
_GATE_TO_UPSTREAM = {
    "verified_calendar": "calendar",
    "verified_return_reconciliation": "return_reconciliation",
    "verified_signal_universe": "signal_universe",
    "verified_auxiliary": "auxiliary",
}


class Stage7VerificationError(RuntimeError):
    """Raised when independent Stage 7 verification cannot proceed safely."""


@dataclass(frozen=True)
class Difference:
    code: str
    expected: Any
    observed: Any
    details: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "expected": _json_safe(self.expected),
            "observed": _json_safe(self.observed),
            "details": self.details,
        }


@dataclass(frozen=True)
class RecomputeEvidence:
    """Evidence returned by an independent numerical recomputation engine."""

    checks: Mapping[str, Any]
    differences: tuple[Difference, ...] = ()
    complete: bool = False


@dataclass(frozen=True)
class InputReceipt:
    gate: str
    name: str
    sha256: str
    schema_sha256: str
    size_bytes: int
    rows: int
    row_groups: int
    path: Path
    stat_identity: tuple[int, ...] = field(repr=False)

    def manifest_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "name": self.name,
            "sha256": self.sha256,
            "schema_sha256": self.schema_sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "row_groups": self.row_groups,
        }


@dataclass(frozen=True)
class FrozenLabelContract:
    project_root: Path
    policy_path: Path
    policy: Mapping[str, Any]
    semantic: Mapping[str, Any]
    rules: Mapping[str, Any]
    input_names: tuple[str, ...]
    output_files: Mapping[str, tuple[str, ...]]
    output_roots: Mapping[str, str]
    published_schemas: Mapping[str, pa.Schema]
    published_schema_hashes: Mapping[str, str]
    configuration_hashes: Mapping[str, str]
    hard_gates: Mapping[str, int]


@dataclass(frozen=True)
class AuthenticatedOutput:
    kind: str
    path: Path
    files: Mapping[str, Path]
    hashes: Mapping[str, str]
    stat_identities: Mapping[str, tuple[int, ...]] = field(repr=False)


@dataclass(frozen=True)
class AuthenticatedLabelArtifacts:
    contract: FrozenLabelContract
    database_root: Path
    artifact_root: Path
    primary_pointer_path: Path
    primary_pointer_sha256: str
    primary_manifest_path: Path
    primary_manifest_sha256: str
    manifest: Mapping[str, Any]
    inputs: Mapping[str, InputReceipt]
    outputs: Mapping[str, AuthenticatedOutput]

    @property
    def build_id(self) -> str:
        return str(self.manifest["build_id"])


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    build_id: str | None
    verification_id: str | None
    checks: Mapping[str, Any]
    differences: tuple[Difference, ...]
    verification_path: Path | None
    verified_pointer_path: Path | None


RecomputeHook = Callable[[AuthenticatedLabelArtifacts], RecomputeEvidence]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage7VerificationError(message)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    raise Stage7VerificationError(
        f"Value is not canonical-JSON compatible: {type(value).__name__}"
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return _sha256_bytes(schema.serialize().to_pybytes())


def _json_object(path: Path, *, canonical: bool = False) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Stage7VerificationError(f"Invalid JSON object: {path}") from exc
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    if canonical:
        _require(
            payload == _canonical_json_bytes(value),
            f"JSON artifact is not canonical: {path}",
        )
    return value


def _toml_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise Stage7VerificationError(f"Invalid TOML object: {path}") from exc
    _require(isinstance(value, dict), f"Expected TOML object: {path}")
    return value


def _exact_keys(
    value: Any, expected: Sequence[str] | frozenset[str], label: str
) -> None:
    _require(
        isinstance(value, Mapping) and set(value) == set(expected),
        f"{label} keys differ",
    )


def _safe_hash_map(value: Any, label: str) -> dict[str, str]:
    _require(isinstance(value, Mapping) and bool(value), f"{label} is empty")
    result: dict[str, str] = {}
    for key, digest in value.items():
        _require(
            isinstance(key, str)
            and bool(key)
            and isinstance(digest, str)
            and _SHA256.fullmatch(digest) is not None,
            f"{label} is invalid",
        )
        result[key] = digest
    return dict(sorted(result.items()))


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_gid),
    )


def _same_stat_except_ctime(left: os.stat_result, right: os.stat_result) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
    )
    return all(getattr(left, name) == getattr(right, name) for name in fields)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _anchored_path(
    candidate: Path,
    *,
    trusted_root: Path,
    label: str,
    directory: bool,
) -> Path:
    root = _lexical_absolute(trusted_root)
    raw = _lexical_absolute(candidate)
    try:
        relative = raw.relative_to(root)
    except ValueError as exc:
        raise Stage7VerificationError(f"{label} escaped its trusted root") from exc
    _require(root.exists() and not root.is_symlink(), f"Unsafe trusted root: {root}")
    current = root
    for part in relative.parts:
        current = current / part
        _require(not current.is_symlink(), f"Symlinked {label}: {current}")
    try:
        resolved = raw.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise Stage7VerificationError(f"Missing or escaped {label}: {raw}") from exc
    observed = os.lstat(resolved)
    wanted = (
        stat.S_ISDIR(observed.st_mode) if directory else stat.S_ISREG(observed.st_mode)
    )
    _require(
        wanted and not stat.S_ISLNK(observed.st_mode), f"Unsafe {label}: {resolved}"
    )
    if not directory:
        _require(observed.st_nlink == 1, f"Hard-linked {label} is prohibited")
    return resolved


def _project_file(project_root: Path, value: str | Path, label: str) -> Path:
    supplied = Path(value).expanduser()
    candidate = supplied if supplied.is_absolute() else project_root / supplied
    return _anchored_path(
        candidate,
        trusted_root=project_root,
        label=label,
        directory=False,
    )


def _authenticate_file(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[str, tuple[int, ...]]:
    _require(_SHA256.fullmatch(expected_sha256) is not None, f"Invalid {label} hash")
    before = os.lstat(path)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_nlink == 1,
        f"Unsafe {label}",
    )
    observed_hash = _sha256_file(path)
    after = os.lstat(path)
    _require(observed_hash == expected_sha256, f"{label} content hash differs")
    _require(
        _stat_identity(before) == _stat_identity(after)
        or _same_stat_except_ctime(before, after),
        f"{label} changed while hashing",
    )
    if _stat_identity(before) != _stat_identity(after):
        _require(
            _sha256_file(path) == expected_sha256,
            f"{label} changed during ctime refresh",
        )
        after = os.lstat(path)
    return observed_hash, _stat_identity(after)


def _parse_schema_field(value: object) -> pa.Field:
    _require(isinstance(value, str), "Published schema column is not text")
    parts = value.split(":")
    _require(len(parts) == 3, f"Invalid Stage 7 schema column: {value!r}")
    name, kind, nullability = parts
    types = {
        "date32": pa.date32(),
        "int64": pa.int64(),
        "int32": pa.int32(),
        "int16": pa.int16(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
        "utf8": pa.string(),
    }
    _require(kind in types, f"Unsupported Stage 7 schema type: {kind}")
    _require(nullability in {"nullable", "not_null"}, "Invalid nullability")
    _require(bool(name), "Published schema column name is empty")
    return pa.field(name, types[kind], nullable=nullability == "nullable")


def _safe_relative(value: object, *, prefix: str, label: str) -> str:
    _require(isinstance(value, str), f"{label} namespace is not text")
    pure = PurePosixPath(value)
    _require(
        not pure.is_absolute()
        and bool(pure.parts)
        and ".." not in pure.parts
        and pure.parts[0] == prefix,
        f"Unsafe {label} namespace",
    )
    return pure.as_posix()


def _configuration_paths(
    policy: Mapping[str, Any],
    *,
    policy_path: Path,
    project_root: Path,
) -> tuple[Path, ...]:
    values: list[Path] = [
        policy_path,
        _project_file(project_root, policy["physical_rules"]["path"], "rules"),
        _project_file(
            project_root,
            policy["semantic_contract"]["policy_path"],
            "semantic contract",
        ),
        _project_file(
            project_root,
            policy["semantic_contract"]["verified_pointer"],
            "semantic pointer",
        ),
        _project_file(
            project_root,
            policy["verified_feature_store"]["policy_path"],
            "feature policy",
        ),
        _project_file(
            project_root,
            policy["verified_feature_store"]["verified_pointer"],
            "feature verified pointer",
        ),
        _project_file(
            project_root,
            policy["verified_feature_store"]["primary_manifest"],
            "feature manifest",
        ),
        _project_file(project_root, policy["database"]["paths_config"], "paths config"),
        _project_file(
            project_root, policy["database"]["relocation_receipt"], "relocation receipt"
        ),
    ]
    values.extend(
        _project_file(project_root, value, f"verified input pointer {name}")
        for name, value in policy["inputs"]["verified_pointers"].items()
        if not str(name).endswith("_sha256")
    )
    return tuple(sorted(set(values), key=lambda item: item.as_posix()))


def reconstruct_frozen_label_contract(
    policy_path: Path = DEFAULT_POLICY,
    *,
    project_root: Path = PROJECT_ROOT,
    reviewed_policy_sha256: str = REVIEWED_POLICY_SHA256,
) -> FrozenLabelContract:
    """Reconstruct the frozen Stage 7 contract without primary validators."""

    root = _lexical_absolute(project_root)
    policy_file = _project_file(root, policy_path, "Stage 7 policy")
    _require(
        _sha256_file(policy_file) == reviewed_policy_sha256,
        "Stage 7 policy is not the reviewed byte artifact",
    )
    policy = _toml_object(policy_file)
    _require(
        policy.get("schema_version") == 1
        and policy.get("stage") == 7
        and policy.get("policy_id") == "phase03-label-store-policy-v1"
        and policy.get("contract_id") == "phase03-feature-label-split-v1"
        and policy.get("status")
        == "frozen_development_only_before_primary_label_build",
        "Stage 7 policy header differs",
    )

    rules_spec = policy["physical_rules"]
    rules_path = _project_file(root, rules_spec["path"], "Stage 7 rules")
    _require(_sha256_file(rules_path) == rules_spec["sha256"], "Rules hash differs")
    rules = _json_object(rules_path)
    _require(
        rules.get("schema_version") == 1
        and rules.get("stage") == 7
        and rules.get("rules_id") == rules_spec["required_rules_id"]
        and rules.get("contract_id") == policy["contract_id"]
        and rules.get("status")
        == "frozen_stage_07_development_only_before_materialization",
        "Stage 7 rules header differs",
    )

    semantic_spec = policy["semantic_contract"]
    semantic_path = _project_file(
        root, semantic_spec["policy_path"], "semantic contract"
    )
    _require(
        _sha256_file(semantic_path) == semantic_spec["policy_sha256"],
        "Semantic contract hash differs",
    )
    semantic = _toml_object(semantic_path)

    hash_bound = (
        (
            semantic_spec,
            "verified_pointer",
            "verified_pointer_sha256",
            "semantic pointer",
        ),
        (
            policy["verified_feature_store"],
            "policy_path",
            "policy_sha256",
            "feature policy",
        ),
        (
            policy["verified_feature_store"],
            "verified_pointer",
            "verified_pointer_sha256",
            "feature pointer",
        ),
        (
            policy["verified_feature_store"],
            "primary_manifest",
            "primary_manifest_sha256",
            "feature manifest",
        ),
        (policy["database"], "paths_config", "paths_config_sha256", "paths config"),
        (
            policy["database"],
            "relocation_receipt",
            "relocation_receipt_sha256",
            "relocation receipt",
        ),
    )
    for spec, path_key, hash_key, label in hash_bound:
        target = _project_file(root, spec[path_key], label)
        _require(_sha256_file(target) == spec[hash_key], f"{label} hash differs")
    pointer_specs = policy["inputs"]["verified_pointers"]
    for name in ("calendar", "returns", "signal_universe", "auxiliary"):
        target = _project_file(root, pointer_specs[name], f"{name} verified pointer")
        _require(
            _sha256_file(target) == pointer_specs[f"{name}_sha256"],
            f"{name} pointer hash differs",
        )

    input_names = tuple(
        str(value) for value in policy["inputs"]["required_table_names"]
    )
    allowlist = rules.get("input_allowlist")
    _require(
        isinstance(allowlist, Mapping)
        and tuple(allowlist) == input_names
        and len(input_names) == len(set(input_names)) == 6,
        "Stage 7 input allowlist differs",
    )

    output_files: dict[str, tuple[str, ...]] = {}
    for kind in _SUCCESS_LOCATIONS:
        names = tuple(str(value) for value in policy["outputs"][kind]["required_files"])
        _require(
            bool(names)
            and len(names) == len(set(names))
            and all(
                Path(name).name == name and _SAFE_NAME.fullmatch(name) for name in names
            ),
            f"Unsafe Stage 7 {kind} closed file set",
        )
        output_files[kind] = names
    schema_specs = rules.get("published_table_schemas")
    _require(isinstance(schema_specs, Mapping), "Published schemas are missing")
    schemas = {
        str(name): pa.schema(
            [_parse_schema_field(value) for value in specification["columns"]]
        )
        for name, specification in schema_specs.items()
    }
    expected_schema_names = {
        name
        for names in output_files.values()
        for name in names
        if name.endswith(".parquet")
    }
    _require(set(schemas) == expected_schema_names, "Published schema set differs")

    hard_policy = {
        str(key): int(value)
        for key, value in policy["quality_gates"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    hard_rules = {
        str(key): int(value)
        for key, value in rules["quality_gates"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    _require(hard_policy == hard_rules and len(hard_policy) == 57, "Hard gates differ")
    publication = policy["publication"]
    _require(
        publication["primary_pointer_authorizes_downstream_use"] is False
        and publication["independent_verification_required"] is True
        and publication["verified_pointer_authorizes_stage_08_risk_and_cost_models"]
        is True
        and publication["verified_pointer_authorizes_stage_09_model_training"] is False
        and publication["verified_pointer_authorizes_portfolios_pnl_or_backtests"]
        is False
        and publication["verified_pointer_authorizes_validation_or_final_access"]
        is False,
        "Stage 7 publication authority differs",
    )
    for key in ("latest_primary_pointer", "latest_verified_pointer"):
        _safe_relative(publication[key], prefix="artifacts", label=key)

    configuration_hashes: dict[str, str] = {}
    for path in _configuration_paths(
        policy, policy_path=policy_file, project_root=root
    ):
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = path.as_posix()
        configuration_hashes[label] = _sha256_file(path)
    output_roots = {
        "interim": _safe_relative(
            policy["database"]["interim_build_root"], prefix="interim", label="interim"
        ),
        "processed": _safe_relative(
            policy["database"]["processed_build_root"],
            prefix="processed",
            label="processed",
        ),
        "quality": _safe_relative(
            policy["database"]["quality_build_root"], prefix="quality", label="quality"
        ),
    }
    return FrozenLabelContract(
        project_root=root,
        policy_path=policy_file,
        policy=policy,
        semantic=semantic,
        rules=rules,
        input_names=input_names,
        output_files=output_files,
        output_roots=output_roots,
        published_schemas=schemas,
        published_schema_hashes={
            name: _schema_sha256(schema) for name, schema in schemas.items()
        },
        configuration_hashes=dict(sorted(configuration_hashes.items())),
        hard_gates=dict(sorted(hard_policy.items())),
    )


def _manifest_output_receipts(
    manifest: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    outputs = manifest.get("outputs")
    if isinstance(outputs, list):
        candidates.extend(item for item in outputs if isinstance(item, Mapping))
    elif isinstance(outputs, Mapping):
        for key, item in outputs.items():
            if isinstance(item, Mapping) and "files" not in item:
                candidates.append({"table_key": key, **item})
    output = manifest.get("output")
    if isinstance(output, Mapping):
        candidates.append(output)
    result: dict[str, Mapping[str, Any]] = {}
    for item in candidates:
        value = item.get("path")
        if not isinstance(value, str) or not value.endswith(".parquet"):
            continue
        name = Path(value).name
        _require(name not in result, f"Duplicate upstream output receipt: {name}")
        result[name] = item
    return result


def _validate_upstream_authority(
    contract: FrozenLabelContract,
    *,
    gate: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    specification = contract.semantic["upstream"][gate]
    pointer_path = _project_file(
        contract.project_root,
        specification["verified_pointer"],
        f"{gate} verified pointer",
    )
    _require(
        _sha256_file(pointer_path) == specification["verified_pointer_sha256"],
        f"{gate} verified-pointer hash differs",
    )
    pointer = _json_object(pointer_path)
    _require(
        pointer.get("schema_version") == 1
        and pointer.get("status") == specification["required_status"]
        and pointer.get("build_id") == specification["required_build_id"]
        and pointer.get("verification_id") == specification["required_verification_id"]
        and pointer.get("contract_id") == specification["required_contract_id"],
        f"{gate} verified authority differs",
    )
    authorizations = pointer.get("authorizations")
    if specification.get("legacy_no_authorizations_object") is not True:
        _require(
            isinstance(authorizations, Mapping), f"{gate} authorizations are missing"
        )
        for name in specification.get(
            "required_authorizations", ()
        ):  # pragma: no branch
            _require(
                authorizations.get(name) is True, f"{gate} authority is closed: {name}"
            )
        for name in specification.get(
            "required_false_authorizations", ()
        ):  # pragma: no branch
            _require(
                authorizations.get(name) is False,
                f"{gate} downstream seal opened: {name}",
            )

    manifest_path = _project_file(
        contract.project_root,
        specification["primary_manifest"],
        f"{gate} primary manifest",
    )
    manifest_hash = _sha256_file(manifest_path)
    _require(
        manifest_hash == specification["primary_manifest_sha256"]
        and pointer.get("primary_manifest_sha256") == manifest_hash,
        f"{gate} primary-manifest binding differs",
    )
    manifest = _json_object(manifest_path)
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("build_id") == specification["required_build_id"]
        and manifest.get("contract_id") == specification["required_contract_id"],
        f"{gate} primary manifest identity differs",
    )
    return manifest, specification


def _validate_semantic_and_feature_authorities(contract: FrozenLabelContract) -> None:
    semantic = contract.policy["semantic_contract"]
    semantic_pointer = _json_object(
        _project_file(
            contract.project_root, semantic["verified_pointer"], "semantic pointer"
        )
    )
    _require(
        semantic_pointer.get("status") == "verified_passed"
        and semantic_pointer.get("build_id") == semantic["required_contract_build_id"]
        and semantic_pointer.get("verification_id")
        == semantic["required_contract_verification_id"]
        and semantic_pointer.get("contract_id") == contract.policy["contract_id"]
        and semantic_pointer.get("authorizations", {}).get(
            semantic["required_authorization"]
        )
        is True
        and semantic_pointer.get("authorizations", {}).get(
            "models_portfolios_pnl_performance_or_backtests_authorized"
        )
        is False
        and semantic_pointer.get("authorizations", {}).get(
            "validation_or_final_test_access_authorized"
        )
        is False,
        "Verified semantic-contract authority differs",
    )

    feature = contract.policy["verified_feature_store"]
    pointer_path = _project_file(
        contract.project_root, feature["verified_pointer"], "feature verified pointer"
    )
    pointer = _json_object(pointer_path)
    manifest_path = _project_file(
        contract.project_root, feature["primary_manifest"], "feature manifest"
    )
    _require(
        pointer.get("schema_version") == 1
        and pointer.get("status") == "VERIFIED_PASSED"
        and pointer.get("build_id") == feature["required_build_id"]
        and pointer.get("verification_id") == feature["required_verification_id"]
        and pointer.get("contract_id") == contract.policy["contract_id"]
        and pointer.get("primary_manifest_sha256") == feature["primary_manifest_sha256"]
        and pointer.get("authorizations", {}).get(feature["required_authorization"])
        is True
        and pointer.get("authorizations", {}).get(
            "models_portfolios_pnl_or_backtests_authorized"
        )
        is False
        and pointer.get("authorizations", {}).get(
            "validation_or_final_test_access_authorized"
        )
        is False
        and _sha256_file(manifest_path) == feature["primary_manifest_sha256"],
        "Verified feature-store authority differs",
    )


def _relocated_input_path(
    value: str,
    *,
    database_root: Path,
    historical_root: Path,
    interim_path: str | None,
) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        try:
            relative = raw.relative_to(historical_root)
        except ValueError:
            try:
                relative = raw.relative_to(database_root)
            except ValueError as exc:
                raise Stage7VerificationError(
                    "Upstream Stage 7 input escaped authenticated roots"
                ) from exc
        candidate = database_root / relative
    elif len(raw.parts) > 1:
        candidate = database_root / raw
    elif interim_path is not None:
        candidate = database_root / interim_path / raw
    else:
        raise Stage7VerificationError(f"Cannot resolve upstream Stage 7 input: {value}")
    return _anchored_path(
        candidate,
        trusted_root=database_root,
        label="Stage 7 input",
        directory=False,
    )


def _feature_store_input(
    contract: FrozenLabelContract,
    *,
    database_root: Path,
) -> tuple[Path, Mapping[str, Any]]:
    feature = contract.policy["verified_feature_store"]
    manifest = _json_object(
        _project_file(
            contract.project_root, feature["primary_manifest"], "feature manifest"
        )
    )
    interim = manifest.get("outputs", {}).get("interim")
    _require(isinstance(interim, Mapping), "Feature-store interim receipt is missing")
    files = interim.get("files")
    _require(isinstance(files, Mapping), "Feature-store closed file set is missing")
    receipt = files.get("feature_store_raw.parquet")
    _require(isinstance(receipt, Mapping), "Feature-store raw receipt is missing")
    relative = PurePosixPath(str(interim.get("path", "")))
    _require(
        not relative.is_absolute()
        and bool(relative.parts)
        and ".." not in relative.parts,
        "Unsafe feature-store input namespace",
    )
    path = _anchored_path(
        database_root / Path(*relative.parts) / "feature_store_raw.parquet",
        trusted_root=database_root,
        label="feature-store input",
        directory=False,
    )
    return path, receipt


def reconstruct_input_receipts(
    contract: FrozenLabelContract,
    *,
    database_root: Path | None = None,
) -> tuple[InputReceipt, ...]:
    """Resolve and fully hash-authenticate exactly six Stage 7 inputs."""

    _validate_semantic_and_feature_authorities(contract)
    declared = _lexical_absolute(Path(str(contract.policy["database"]["root"])))
    active = declared if database_root is None else _lexical_absolute(database_root)
    _require(active == declared, "Active Stage 7 database root differs from policy")
    active = _anchored_path(
        active, trusted_root=active, label="database root", directory=True
    )
    historical = _lexical_absolute(
        Path(str(contract.semantic["database"]["historical_root"]))
    )
    relocation_path = _project_file(
        contract.project_root,
        contract.policy["database"]["relocation_receipt"],
        "relocation receipt",
    )
    relocation = _json_object(relocation_path)
    _require(
        relocation.get("new_database_root") == active.as_posix()
        and relocation.get("old_database_root") == historical.as_posix(),
        "Stage 7 relocation authority differs",
    )

    upstream: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {
        gate: _validate_upstream_authority(contract, gate=gate)
        for gate in _GATE_TO_UPSTREAM.values()
    }
    receipts: list[InputReceipt] = []
    for name in contract.input_names:
        expected = contract.rules["input_allowlist"][name]
        gate = str(expected["gate"])
        if gate == "verified_stage_06":
            path, source = _feature_store_input(contract, database_root=active)
        else:
            _require(gate in _GATE_TO_UPSTREAM, f"Unknown Stage 7 input gate: {gate}")
            upstream_gate = _GATE_TO_UPSTREAM[gate]
            manifest, _specification = upstream[upstream_gate]
            source_receipts = _manifest_output_receipts(manifest)
            _require(name in source_receipts, f"Upstream receipt is missing: {name}")
            source = source_receipts[name]
            interim_path = manifest.get("interim_path")
            _require(
                interim_path is None or isinstance(interim_path, str),
                f"Invalid upstream namespace: {upstream_gate}",
            )
            path = _relocated_input_path(
                str(source["path"]),
                database_root=active,
                historical_root=historical,
                interim_path=interim_path,
            )

        expected_hash = str(expected["sha256"])
        _require(
            source.get("sha256") == expected_hash
            and source.get("size_bytes") == path.stat().st_size,
            f"Upstream receipt differs: {name}",
        )
        observed_hash, identity = _authenticate_file(
            path, expected_sha256=expected_hash, label=f"input {name}"
        )
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        groups = int(parquet.metadata.num_row_groups)
        schema_hash = _schema_sha256(parquet.schema_arrow)
        _require(
            rows == int(expected["rows"])
            and schema_hash == expected["schema_sha256"]
            and (source.get("rows") is None or int(source["rows"]) == rows)
            and set(expected["projected_columns"]).issubset(parquet.schema_arrow.names),
            f"Input metadata differs: {name}",
        )
        receipts.append(
            InputReceipt(
                gate=gate,
                name=name,
                sha256=observed_hash,
                schema_sha256=schema_hash,
                size_bytes=path.stat().st_size,
                rows=rows,
                row_groups=groups,
                path=path,
                stat_identity=identity,
            )
        )
    _require(
        tuple(item.name for item in receipts) == contract.input_names,
        "Resolved Stage 7 input order differs",
    )
    return tuple(receipts)


def _safe_artifact_root(path: Path, *, create: bool) -> Path:
    root = _lexical_absolute(path)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    _require(
        root.exists() and root.is_dir() and not root.is_symlink(),
        "Unsafe artifact root",
    )
    return root


def _read_primary(
    contract: FrozenLabelContract,
    *,
    artifact_root: Path,
    primary_pointer_path: Path | None,
    primary_manifest_path: Path | None,
) -> tuple[Path, str, Path, dict[str, Any], str]:
    pointer_candidate = (
        contract.project_root
        / str(contract.policy["publication"]["latest_primary_pointer"])
        if primary_pointer_path is None
        else primary_pointer_path
    )
    pointer_path = _anchored_path(
        pointer_candidate,
        trusted_root=artifact_root,
        label="Stage 7 primary pointer",
        directory=False,
    )
    pointer_payload = pointer_path.read_bytes()
    try:
        pointer = json.loads(pointer_payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Stage7VerificationError("Invalid Stage 7 primary pointer") from exc
    pointer_keys = contract.rules["publication_payloads"][
        "primary_pointer_required_keys"
    ]
    _exact_keys(pointer, pointer_keys, "Primary pointer")
    build_id = pointer.get("build_id")
    _require(
        pointer_payload == _canonical_json_bytes(pointer)
        and pointer.get("schema_version") == 1
        and pointer.get("contract_id") == contract.policy["contract_id"]
        and pointer.get("policy_id") == contract.policy["policy_id"]
        and isinstance(build_id, str)
        and _BUILD_ID.fullmatch(build_id) is not None
        and pointer.get("status") == _PRIMARY_STATUS
        and pointer.get("authorizations") == _PRIMARY_AUTHORIZATIONS
        and _SHA256.fullmatch(str(pointer.get("primary_manifest_sha256", "")))
        is not None,
        "Primary pointer binding differs",
    )
    pointer_hash = _sha256_bytes(pointer_payload)
    expected_manifest = (
        artifact_root
        / str(contract.policy["publication"]["primary_build_namespace"])
        / build_id
        / "manifest.json"
    )
    manifest_candidate = (
        expected_manifest if primary_manifest_path is None else primary_manifest_path
    )
    manifest_path = _anchored_path(
        manifest_candidate,
        trusted_root=artifact_root,
        label="Stage 7 primary manifest",
        directory=False,
    )
    _require(
        manifest_path == _lexical_absolute(expected_manifest),
        "Primary manifest is outside the content-addressed namespace",
    )
    manifest_payload = manifest_path.read_bytes()
    manifest_hash = _sha256_bytes(manifest_payload)
    _require(
        manifest_hash == pointer["primary_manifest_sha256"],
        "Primary manifest hash differs from pointer",
    )
    try:
        manifest = json.loads(manifest_payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Stage7VerificationError("Invalid Stage 7 primary manifest") from exc
    required_manifest_keys = contract.rules["publication_payloads"][
        "primary_manifest_required_keys"
    ]
    _exact_keys(manifest, required_manifest_keys, "Primary manifest")
    _exact_keys(manifest["bindings"], _BINDING_KEYS, "Primary bindings")
    _exact_keys(manifest["inputs"], _INPUT_KEYS, "Primary inputs")
    _exact_keys(manifest["outputs"], frozenset(_SUCCESS_LOCATIONS), "Primary outputs")
    _exact_keys(
        manifest["summary"],
        frozenset(
            {
                "audit_rows",
                "hard_gate_failures",
                "leg_lineage_rows",
                "processed_rows",
                "signal_dates",
                "split_membership_rows",
            }
        ),
        "Primary summary",
    )
    _require(
        manifest_payload == _canonical_json_bytes(manifest)
        and manifest.get("schema_version") == 1
        and manifest.get("contract_id") == contract.policy["contract_id"]
        and manifest.get("policy_id") == contract.policy["policy_id"]
        and manifest.get("build_id") == build_id
        and manifest.get("status") == _PRIMARY_STATUS
        and manifest.get("authorizations") == _PRIMARY_AUTHORIZATIONS,
        "Primary manifest header differs",
    )
    summary = manifest["summary"]
    scope = contract.policy["scope"]
    _require(
        summary["audit_rows"] == scope["required_audit_rows"]
        and summary["processed_rows"] == scope["required_long_eligible_rows"]
        and summary["signal_dates"] == scope["required_signal_dates"]
        and summary["split_membership_rows"] == scope["required_split_membership_rows"]
        and summary["hard_gate_failures"] == 0
        and isinstance(summary["leg_lineage_rows"], int)
        and summary["leg_lineage_rows"] >= 0,
        "Primary manifest summary differs",
    )
    return pointer_path, pointer_hash, manifest_path, manifest, manifest_hash


def _output_directory(
    contract: FrozenLabelContract,
    *,
    database_root: Path,
    build_id: str,
    kind: str,
    supplied: Any,
) -> Path:
    _require(isinstance(supplied, str), f"{kind} output path is invalid")
    raw = Path(supplied).expanduser()
    resolved = _lexical_absolute(raw if raw.is_absolute() else database_root / raw)
    expected = _lexical_absolute(database_root / contract.output_roots[kind] / build_id)
    _require(resolved == expected, f"{kind} output namespace differs")
    return _anchored_path(
        resolved,
        trusted_root=database_root,
        label=f"{kind} output directory",
        directory=True,
    )


def _authenticate_output(
    contract: FrozenLabelContract,
    manifest: Mapping[str, Any],
    *,
    database_root: Path,
    kind: str,
) -> AuthenticatedOutput:
    specification = manifest["outputs"][kind]
    _exact_keys(specification, _OUTPUT_KEYS, f"{kind} output receipt")
    path = _output_directory(
        contract,
        database_root=database_root,
        build_id=str(manifest["build_id"]),
        kind=kind,
        supplied=specification["path"],
    )
    expected_names = set(contract.output_files[kind])
    observed_names = {item.name for item in path.iterdir()}
    _require(observed_names == expected_names, f"{kind} closed file set differs")
    file_specs = specification["files"]
    _require(
        isinstance(file_specs, Mapping) and set(file_specs) == expected_names,
        f"{kind} manifest file set differs",
    )
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    identities: dict[str, tuple[int, ...]] = {}
    actual_receipts: dict[str, dict[str, Any]] = {}
    for name in contract.output_files[kind]:
        member = _anchored_path(
            path / name,
            trusted_root=database_root,
            label=f"{kind}/{name}",
            directory=False,
        )
        receipt = file_specs[name]
        _exact_keys(receipt, _FILE_RECEIPT_KEYS, f"{kind}/{name} receipt")
        expected_hash = str(receipt["sha256"])
        observed_hash, identity = _authenticate_file(
            member, expected_sha256=expected_hash, label=f"{kind}/{name}"
        )
        if name.endswith(".parquet"):
            parquet = pq.ParquetFile(member)
            rows: int | None = int(parquet.metadata.num_rows)
            groups: int | None = int(parquet.metadata.num_row_groups)
            schema_hash: str | None = _schema_sha256(parquet.schema_arrow)
            _require(
                name in contract.published_schemas
                and parquet.schema_arrow.equals(
                    contract.published_schemas[name], check_metadata=True
                )
                and schema_hash == contract.published_schema_hashes[name],
                f"{kind}/{name} schema differs from frozen contract",
            )
            declared_rows = contract.rules["published_table_schemas"][name]["rows"]
            _require(
                declared_rows is None or rows == int(declared_rows),
                f"{kind}/{name} frozen row count differs",
            )
        else:
            rows = groups = schema_hash = None
        observed_receipt = {
            "path": name,
            "sha256": observed_hash,
            "size_bytes": member.stat().st_size,
            "rows": rows,
            "row_groups": groups,
            "schema_sha256": schema_hash,
        }
        _require(dict(receipt) == observed_receipt, f"{kind}/{name} receipt differs")
        paths[name] = member
        hashes[name] = observed_hash
        identities[name] = identity
        if name != "_SUCCESS.json":
            actual_receipts[name] = observed_receipt

    _require(
        specification["success_sha256"] == hashes["_SUCCESS.json"],
        f"{kind} success binding differs",
    )
    success_payload = paths["_SUCCESS.json"].read_bytes()
    try:
        success = json.loads(success_payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Stage7VerificationError(f"Invalid {kind} success receipt") from exc
    success_keys = contract.rules["publication_payloads"]["success_required_keys"]
    _exact_keys(success, success_keys, f"{kind} success receipt")
    location, status_value = _SUCCESS_LOCATIONS[kind]
    _require(
        success_payload == _canonical_json_bytes(success)
        and success.get("schema_version") == 1
        and success.get("contract_id") == contract.policy["contract_id"]
        and success.get("policy_id") == contract.policy["policy_id"]
        and success.get("build_id") == manifest["build_id"]
        and success.get("location_kind") == location
        and success.get("status") == status_value
        and success.get("bindings") == manifest["bindings"]
        and success.get("table_receipts") == actual_receipts,
        f"{kind} success receipt binding differs",
    )
    if kind == "quality":
        quality = _json_object(paths["quality_summary.json"], canonical=True)
        _exact_keys(
            quality,
            contract.rules["publication_payloads"]["quality_summary_exact_keys"],
            "Quality summary",
        )
        _require(
            quality.get("schema_version") == 1
            and quality.get("contract_id") == contract.policy["contract_id"]
            and quality.get("policy_id") == contract.policy["policy_id"]
            and quality.get("build_id") == manifest["build_id"]
            and quality.get("status") == _PRIMARY_STATUS
            and quality.get("hard_gates") == contract.hard_gates,
            "Quality summary binding differs",
        )
    return AuthenticatedOutput(
        kind=kind,
        path=path,
        files=paths,
        hashes=hashes,
        stat_identities=identities,
    )


def verify_build_identity(
    contract: FrozenLabelContract,
    manifest: Mapping[str, Any],
    input_receipts: Sequence[InputReceipt],
    *,
    required_code_paths: frozenset[str] = _PRIMARY_CODE_PATHS,
) -> Mapping[str, Any]:
    """Rebuild the primary content fingerprint without primary code."""

    bindings = manifest["bindings"]
    configuration = _safe_hash_map(
        bindings["configuration_hashes"], "Configuration hashes"
    )
    _require(
        configuration == dict(contract.configuration_hashes),
        "Primary configuration hashes differ",
    )
    code = _safe_hash_map(bindings["code_hashes"], "Code hashes")
    _require(set(code) == set(required_code_paths), "Primary code-hash set differs")
    for label, expected_hash in code.items():
        pure = PurePosixPath(label)
        _require(
            not pure.is_absolute() and ".." not in pure.parts and pure.suffix == ".py",
            f"Unsafe primary code binding: {label}",
        )
        path = _project_file(contract.project_root, label, f"code binding {label}")
        _require(_sha256_file(path) == expected_hash, f"Code binding differs: {label}")

    identity_order = [item.manifest_dict() for item in input_receipts]
    manifest_order = [
        item.manifest_dict()
        for item in sorted(input_receipts, key=lambda receipt: receipt.name)
    ]
    _require(
        manifest["inputs"]["table_receipts"] == manifest_order,
        "Primary input receipts differ",
    )
    receipts_hash = _sha256_bytes(_canonical_json_bytes(identity_order))
    _require(
        bindings["input_receipts_sha256"] == receipts_hash,
        "Input receipt aggregate hash differs",
    )
    policy_inputs = contract.policy["inputs"]
    expected_builds = {
        "auxiliary": policy_inputs["auxiliary_build_id"],
        "calendar": policy_inputs["calendar_build_id"],
        "feature_store": policy_inputs["feature_store_build_id"],
        "feature_store_verification": policy_inputs["feature_store_verification_id"],
        "return_reconciliation": policy_inputs["return_build_id"],
        "signal_universe": policy_inputs["signal_universe_build_id"],
    }
    _require(
        manifest["inputs"]["input_builds"] == expected_builds,
        "Primary input build identities differ",
    )
    content = {
        "schema_version": 1,
        "contract_id": contract.policy["contract_id"],
        "policy_id": contract.policy["policy_id"],
        "configuration_hashes": configuration,
        "code_hashes": code,
        "input_receipts": identity_order,
    }
    fingerprint = _sha256_bytes(_canonical_json_bytes(content))
    build_id = f"phase03-labels-{fingerprint[:24]}"
    _require(
        bindings["content_fingerprint"] == fingerprint
        and manifest["build_id"] == build_id,
        "Deterministic Stage 7 build identity differs",
    )
    return {
        "content_fingerprint": fingerprint,
        "build_id": build_id,
        "input_receipts_sha256": receipts_hash,
        "configuration_hashes": configuration,
        "code_hashes": code,
    }


def authenticate_stage7_artifacts(
    *,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
    database_root: Path | None = None,
    artifact_root: Path | None = None,
    primary_pointer_path: Path | None = None,
    primary_manifest_path: Path | None = None,
    reviewed_policy_sha256: str = REVIEWED_POLICY_SHA256,
) -> AuthenticatedLabelArtifacts:
    """Authenticate the complete Stage 7 primary artifact graph."""

    contract = reconstruct_frozen_label_contract(
        policy_path,
        project_root=project_root,
        reviewed_policy_sha256=reviewed_policy_sha256,
    )
    database = _lexical_absolute(
        Path(str(contract.policy["database"]["root"]))
        if database_root is None
        else database_root
    )
    artifacts = _safe_artifact_root(
        (
            contract.project_root / str(contract.policy["publication"]["artifact_root"])
            if artifact_root is None
            else artifact_root
        ),
        create=False,
    )
    (
        pointer_path,
        pointer_hash,
        manifest_path,
        manifest,
        manifest_hash,
    ) = _read_primary(
        contract,
        artifact_root=artifacts,
        primary_pointer_path=primary_pointer_path,
        primary_manifest_path=primary_manifest_path,
    )
    inputs = reconstruct_input_receipts(contract, database_root=database)
    verify_build_identity(contract, manifest, inputs)
    outputs = {
        kind: _authenticate_output(
            contract,
            manifest,
            database_root=database,
            kind=kind,
        )
        for kind in _SUCCESS_LOCATIONS
    }
    return AuthenticatedLabelArtifacts(
        contract=contract,
        database_root=database,
        artifact_root=artifacts,
        primary_pointer_path=pointer_path,
        primary_pointer_sha256=pointer_hash,
        primary_manifest_path=manifest_path,
        primary_manifest_sha256=manifest_hash,
        manifest=manifest,
        inputs={item.name: item for item in inputs},
        outputs=outputs,
    )


def reauthenticate_stage7_artifacts(artifacts: AuthenticatedLabelArtifacts) -> None:
    """Full re-hash and compare before any authority publication."""

    observed = authenticate_stage7_artifacts(
        policy_path=artifacts.contract.policy_path,
        project_root=artifacts.contract.project_root,
        database_root=artifacts.database_root,
        artifact_root=artifacts.artifact_root,
        primary_pointer_path=artifacts.primary_pointer_path,
        primary_manifest_path=artifacts.primary_manifest_path,
        reviewed_policy_sha256=_sha256_file(artifacts.contract.policy_path),
    )
    _require(
        observed.primary_pointer_sha256 == artifacts.primary_pointer_sha256
        and observed.primary_manifest_sha256 == artifacts.primary_manifest_sha256
        and observed.manifest == artifacts.manifest
        and {name: receipt.manifest_dict() for name, receipt in observed.inputs.items()}
        == {name: receipt.manifest_dict() for name, receipt in artifacts.inputs.items()}
        and {kind: dict(output.hashes) for kind, output in observed.outputs.items()}
        == {kind: dict(output.hashes) for kind, output in artifacts.outputs.items()},
        "Stage 7 artifact graph changed during verification",
    )


def _artifact_checks(artifacts: AuthenticatedLabelArtifacts) -> dict[str, Any]:
    return {
        "artifact_graph_authenticated": True,
        "build_identity_matches": True,
        "closed_output_sets_authenticated": len(artifacts.outputs),
        "input_tables_authenticated": len(artifacts.inputs),
        "output_files_authenticated": sum(
            len(output.files) for output in artifacts.outputs.values()
        ),
        "primary_manifest_sha256": artifacts.primary_manifest_sha256,
    }


def _verification_code_hashes(project_root: Path) -> dict[str, str]:
    return {
        label: _sha256_file(
            _project_file(project_root, label, f"verification code {label}")
        )
        for label in _VERIFICATION_CODE_PATHS
    }


def _coerce_difference(value: Any) -> Difference:
    if isinstance(value, Difference):
        difference = value
    elif isinstance(value, Mapping):
        _exact_keys(
            value,
            frozenset({"code", "expected", "observed", "details"}),
            "Recomputation difference",
        )
        difference = Difference(
            code=value["code"],
            expected=value["expected"],
            observed=value["observed"],
            details=value["details"],
        )
    else:
        try:
            difference = Difference(
                code=value.code,
                expected=value.expected,
                observed=value.observed,
                details=value.details,
            )
        except AttributeError as exc:
            raise Stage7VerificationError(
                "Recomputation difference has an invalid shape"
            ) from exc
    _require(
        isinstance(difference.code, str)
        and bool(difference.code)
        and isinstance(difference.details, str)
        and bool(difference.details),
        "Recomputation difference code or details are invalid",
    )
    # Validate publication compatibility without altering the evidence values.
    _json_safe(difference.expected)
    _json_safe(difference.observed)
    return difference


def _coerce_recompute_evidence(value: Any) -> RecomputeEvidence:
    if isinstance(value, Mapping):
        _require(
            {"checks", "differences", "complete"}.issubset(value),
            "Recomputation evidence mapping is incomplete",
        )
        checks = value["checks"]
        differences = value["differences"]
        complete = value["complete"]
        metrics = value.get("metrics")
    else:
        try:
            checks = value.checks
            differences = value.differences
            complete = value.complete
            metrics = getattr(value, "metrics", None)
        except AttributeError as exc:
            raise Stage7VerificationError(
                "Recomputation hook returned an invalid evidence object"
            ) from exc
    _require(isinstance(checks, Mapping), "Recomputation checks are not a mapping")
    _require(
        isinstance(differences, (list, tuple)),
        "Recomputation differences are not a finite sequence",
    )
    _require(
        isinstance(complete, bool),
        "Recomputation completion flag is not boolean",
    )
    normalized_checks: Mapping[str, Any] = checks
    if metrics is not None:
        _require(
            isinstance(metrics, Mapping),
            "Recomputation metrics are not a mapping",
        )
        if "hard_gates" not in checks:
            normalized_checks = {
                "hard_gates": dict(checks),
                "metrics": dict(metrics),
            }
        else:
            _require(
                "metrics" not in checks,
                "Recomputation metrics are duplicated",
            )
            normalized_checks = {**dict(checks), "metrics": dict(metrics)}
    return RecomputeEvidence(
        checks=normalized_checks,
        differences=tuple(_coerce_difference(item) for item in differences),
        complete=complete,
    )


def _validate_recompute_evidence(
    contract: FrozenLabelContract,
    evidence: RecomputeEvidence,
) -> tuple[Difference, ...]:
    differences = list(evidence.differences)
    if not evidence.complete:
        differences.append(
            Difference(
                code="numerical_recomputation_incomplete",
                expected=True,
                observed=False,
                details="All Stage 7 numerical checks must complete before authority publication.",
            )
        )
        return tuple(differences)
    nested = evidence.checks.get("hard_gates")
    if isinstance(nested, Mapping):
        observed = dict(nested)
    else:
        observed = {
            name: evidence.checks[name]
            for name in contract.hard_gates
            if name in evidence.checks
        }
    if set(observed) != set(contract.hard_gates):
        differences.append(
            Difference(
                code="hard_gate_key_set",
                expected=sorted(contract.hard_gates),
                observed=sorted(str(value) for value in observed),
                details="The numerical verifier must report every frozen Stage 7 hard gate.",
            )
        )
        return tuple(differences)
    for name, value in observed.items():
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"Recomputation hard gate is not a nonnegative integer: {name}",
        )
    for name, expected in contract.hard_gates.items():
        if observed[name] != expected:
            differences.append(
                Difference(
                    code=f"hard_gate:{name}",
                    expected=expected,
                    observed=observed[name],
                    details=f"Frozen Stage 7 hard gate {name} differs.",
                )
            )
    return tuple(differences)


def _publish_immutable_bundle(
    *,
    artifact_root: Path,
    destination: Path,
    payloads: Mapping[str, bytes],
) -> None:
    _require(
        all(
            Path(name).name == name and _SAFE_NAME.fullmatch(name) for name in payloads
        ),
        "Unsafe verification bundle filename",
    )
    try:
        destination.relative_to(artifact_root)
    except ValueError as exc:
        raise Stage7VerificationError(
            "Verification bundle escaped artifact root"
        ) from exc
    expected_hashes = {
        name: _sha256_bytes(payload) for name, payload in payloads.items()
    }
    try:
        with SecureStagingDirectory.create(
            trusted_root=artifact_root,
            destination=destination,
            expected_files=tuple(payloads),
            label="Stage 7 independent verification",
        ) as stage:
            for name, payload in payloads.items():
                with stage.open_exclusive_binary(name) as stream:
                    stream.write(payload)
            snapshot = stage.snapshot(expected_hashes=expected_hashes)
            stage.publish(expected_hashes=snapshot.hashes)
    except SecurePublicationError as exc:
        raise Stage7VerificationError(
            "Cannot publish Stage 7 verification bundle"
        ) from exc


def _publish_verification(
    artifacts: AuthenticatedLabelArtifacts,
    *,
    checks: Mapping[str, Any],
    differences: tuple[Difference, ...],
    verification_code_hashes: Mapping[str, str],
    publication_artifact_root: Path | None = None,
    reauthenticate: Callable[[], None] | None = None,
) -> tuple[str, Path, Path]:
    _require(not differences, "Cannot publish Stage 7 authority with differences")
    code_hashes = _safe_hash_map(
        verification_code_hashes,
        "Verification code hashes",
    )
    _require(
        tuple(sorted(code_hashes)) == tuple(sorted(_VERIFICATION_CODE_PATHS)),
        "Verification code-hash set differs",
    )
    _require(
        checks.get("verification_code_hashes") == code_hashes,
        "Verification checks do not bind the code hashes",
    )
    _require(
        _verification_code_hashes(artifacts.contract.project_root) == code_hashes,
        "Verification code changed before publication",
    )
    root = _safe_artifact_root(
        (
            artifacts.artifact_root
            if publication_artifact_root is None
            else publication_artifact_root
        ),
        create=True,
    )
    difference_bytes = _canonical_json_bytes([])
    difference_hash = _sha256_bytes(difference_bytes)
    output_hashes = {
        kind: dict(sorted(output.hashes.items()))
        for kind, output in sorted(artifacts.outputs.items())
    }
    output_hashes_sha = _sha256_bytes(_canonical_json_bytes(output_hashes))
    canonical_checks = _json_safe(checks)
    _require(
        isinstance(canonical_checks, Mapping),
        "Verification checks are not a canonical object",
    )
    checks_bytes = _canonical_json_bytes(canonical_checks)
    identity = {
        "build_id": artifacts.build_id,
        "primary_manifest_sha256": artifacts.primary_manifest_sha256,
        "verification_code_hashes": code_hashes,
        "checks": canonical_checks,
        "differences_sha256": difference_hash,
        "output_hashes_sha256": output_hashes_sha,
    }
    verification_id = "verify-" + _sha256_bytes(_canonical_json_bytes(identity))[:16]
    _require(
        _VERIFY_ID.fullmatch(verification_id) is not None, "Invalid verification ID"
    )
    check_summary = {
        "check_count": len(checks),
        "pass_count": len(checks),
        "fail_count": 0,
        "results_sha256": _sha256_bytes(checks_bytes),
        "verification_code_hashes": code_hashes,
        "results": canonical_checks,
    }
    independent_check = {
        "schema_version": 1,
        "contract_id": artifacts.contract.policy["contract_id"],
        "policy_id": artifacts.contract.policy["policy_id"],
        "build_id": artifacts.build_id,
        "verification_id": verification_id,
        "status": _VERIFIED_STATUS,
        "primary_manifest_sha256": artifacts.primary_manifest_sha256,
        "checks": check_summary,
        "differences_sha256": difference_hash,
        "output_hashes_sha256": output_hashes_sha,
        "authorizations": _VERIFIED_AUTHORIZATIONS,
    }
    _exact_keys(
        independent_check,
        artifacts.contract.rules["publication_payloads"][
            "independent_check_required_keys"
        ],
        "Independent check",
    )
    check_payload = _canonical_json_bytes(independent_check)
    check_hash = _sha256_bytes(check_payload)
    pointer = {
        "schema_version": 1,
        "contract_id": artifacts.contract.policy["contract_id"],
        "policy_id": artifacts.contract.policy["policy_id"],
        "build_id": artifacts.build_id,
        "verification_id": verification_id,
        "status": _VERIFIED_STATUS,
        "primary_manifest_sha256": artifacts.primary_manifest_sha256,
        "independent_check_sha256": check_hash,
        "differences_sha256": difference_hash,
        "authorizations": _VERIFIED_AUTHORIZATIONS,
    }
    _exact_keys(
        pointer,
        artifacts.contract.rules["publication_payloads"][
            "verified_pointer_required_keys"
        ],
        "Verified pointer",
    )
    namespace = str(artifacts.contract.policy["publication"]["verification_namespace"])
    destination = root / namespace / artifacts.build_id / verification_id
    artifact_gate = reauthenticate or (lambda: None)

    def gate() -> None:
        artifact_gate()
        _require(
            _verification_code_hashes(artifacts.contract.project_root) == code_hashes,
            "Verification code changed during publication",
        )

    gate()
    _publish_immutable_bundle(
        artifact_root=root,
        destination=destination,
        payloads={
            "differences.json": difference_bytes,
            "independent_check.json": check_payload,
        },
    )
    gate()
    pointer_path = (
        root
        / Path(
            str(artifacts.contract.policy["publication"]["latest_verified_pointer"])
        ).name
    )
    try:
        replace_pointer_last(
            pointer_path,
            _canonical_json_bytes(pointer),
            trusted_root=root,
            pre_replace_gate=gate,
            post_replace_gate=gate,
            label="Stage 7 latest verified pointer",
        )
    except SecurePublicationError as exc:
        raise Stage7VerificationError(
            "Cannot replace Stage 7 verified pointer"
        ) from exc
    return verification_id, destination, pointer_path


def verify_stage7_label_store(
    *,
    recompute_hook: RecomputeHook | None,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
    database_root: Path | None = None,
    artifact_root: Path | None = None,
    primary_pointer_path: Path | None = None,
    primary_manifest_path: Path | None = None,
    publication_artifact_root: Path | None = None,
    reviewed_policy_sha256: str = REVIEWED_POLICY_SHA256,
    publish: bool = True,
) -> VerificationResult:
    """Authenticate artifacts, invoke independent recomputation, and seal on zero diff."""

    artifacts = authenticate_stage7_artifacts(
        policy_path=policy_path,
        project_root=project_root,
        database_root=database_root,
        artifact_root=artifact_root,
        primary_pointer_path=primary_pointer_path,
        primary_manifest_path=primary_manifest_path,
        reviewed_policy_sha256=reviewed_policy_sha256,
    )
    verification_code_hashes = _verification_code_hashes(
        artifacts.contract.project_root
    )
    if recompute_hook is None:
        evidence = RecomputeEvidence(
            checks={},
            differences=(
                Difference(
                    code="numerical_recomputation_missing",
                    expected="complete independent Stage 7 recomputation",
                    observed=None,
                    details="Artifact authentication alone cannot authorize Stage 8.",
                ),
            ),
            complete=False,
        )
    else:
        evidence = _coerce_recompute_evidence(recompute_hook(artifacts))
    differences = _validate_recompute_evidence(artifacts.contract, evidence)
    checks = {
        "artifact_authentication": _artifact_checks(artifacts),
        "verification_code_hashes": verification_code_hashes,
        "recomputation": dict(evidence.checks),
    }
    if differences:
        return VerificationResult(
            passed=False,
            build_id=artifacts.build_id,
            verification_id=None,
            checks=checks,
            differences=differences,
            verification_path=None,
            verified_pointer_path=None,
        )
    if publish:
        verification_id, verification_path, pointer_path = _publish_verification(
            artifacts,
            checks=checks,
            differences=(),
            verification_code_hashes=verification_code_hashes,
            publication_artifact_root=publication_artifact_root,
            reauthenticate=lambda: reauthenticate_stage7_artifacts(artifacts),
        )
    else:
        verification_id = verification_path = pointer_path = None
    return VerificationResult(
        passed=True,
        build_id=artifacts.build_id,
        verification_id=verification_id,
        checks=checks,
        differences=(),
        verification_path=verification_path,
        verified_pointer_path=pointer_path,
    )


__all__ = [
    "AuthenticatedLabelArtifacts",
    "AuthenticatedOutput",
    "DEFAULT_POLICY",
    "Difference",
    "FrozenLabelContract",
    "InputReceipt",
    "PROJECT_ROOT",
    "REVIEWED_POLICY_SHA256",
    "RecomputeEvidence",
    "RecomputeHook",
    "Stage7VerificationError",
    "VerificationResult",
    "authenticate_stage7_artifacts",
    "reauthenticate_stage7_artifacts",
    "reconstruct_frozen_label_contract",
    "reconstruct_input_receipts",
    "verify_build_identity",
    "verify_stage7_label_store",
]
