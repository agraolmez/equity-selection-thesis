"""Independent verifier for the frozen roadmap Stage 6 feature store.

This module intentionally does not import the Stage 6 builders, materializers,
registry generators, quality generator, or their assertions.  It reconstructs
the frozen contract from reviewed byte artifacts, authenticates every primary
output, and independently streams the raw and lineage payloads before it can
publish a verification authority pointer.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    SecurePublicationError,
    SecureStagingDirectory,
    replace_pointer_last,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "configs/phase_03_features/feature_store_policy_v1.toml"
)
REVIEWED_POLICY_SHA256 = (
    "1a66bfa4ad537e06829f585e080c356750c7360d486414a40a895de6f6b5380f"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BUILD_ID = re.compile(r"phase03-features-[0-9a-f]{24}\Z")
_SAFE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*\Z")
_VALUE_STATUSES = frozenset(
    {
        "PRESENT",
        "MISSING_AMBIGUOUS_LINK",
        "MISSING_INSUFFICIENT_HISTORY",
        "MISSING_INVALID_INPUT",
        "MISSING_NOT_COVERED",
        "MISSING_SOURCE_UNAVAILABLE",
        "MISSING_STALE",
    }
)
_AUDIT_SCHEMA = pa.schema(
    [
        pa.field("signal_date", pa.date32(), nullable=False),
        pa.field("permno", pa.int64(), nullable=False),
        pa.field("permco", pa.int64(), nullable=False),
        pa.field("long_eligible", pa.bool_(), nullable=False),
        pa.field("short_eligible", pa.bool_(), nullable=False),
        pa.field("primary_exchange", pa.string(), nullable=True),
        pa.field("share_type", pa.string(), nullable=True),
    ]
)
_LINEAGE_SCHEMA = pa.schema(
    [
        pa.field("signal_date", pa.date32(), nullable=False),
        pa.field("permno", pa.int64(), nullable=False),
        pa.field("feature_name", pa.string(), nullable=False),
        pa.field("max_observation_date", pa.date32(), nullable=True),
        pa.field(
            "max_public_availability_date",
            pa.date32(),
            nullable=True,
        ),
        pa.field("stale_after_date", pa.date32(), nullable=True),
        pa.field("value_status", pa.string(), nullable=False),
    ]
)
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
    {
        "path",
        "sha256",
        "size_bytes",
        "rows",
        "row_groups",
        "schema_sha256",
    }
)
_TABLE_RECEIPT_KEYS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "rows",
        "row_groups",
        "schema_sha256",
    }
)
_SUCCESS_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "policy_id",
        "build_id",
        "location_kind",
        "status",
        "configuration_hashes",
        "code_hashes",
        "input_receipts_sha256",
        "table_receipts",
    }
)
_SUMMARY_KEYS = frozenset(
    {
        "raw_rows",
        "economic_features",
        "feature_families",
        "lineage_rows",
        "feature_set_included_counts",
        "category_vocabulary_rows",
        "hard_gate_failures",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {
        "independent_verification_required",
        "stage_07_labels_authorized",
        "models_portfolios_pnl_or_backtests_authorized",
        "validation_or_final_test_access_authorized",
    }
)
_PRIMARY_AUTHORIZATIONS = {
    "independent_verification_required": True,
    "stage_07_labels_authorized": False,
    "models_portfolios_pnl_or_backtests_authorized": False,
    "validation_or_final_test_access_authorized": False,
}
_VERIFIED_AUTHORIZATIONS = {
    "independent_verification_completed": True,
    "stage_07_labels_authorized": True,
    "models_portfolios_pnl_or_backtests_authorized": False,
    "validation_or_final_test_access_authorized": False,
}
_SUCCESS_ENUMS = {
    "interim": ("interim_feature_store", "PRIMARY_INTERIM_COMPLETE"),
    "processed": (
        "processed_feature_registries",
        "PRIMARY_PROCESSED_COMPLETE",
    ),
    "quality": ("feature_store_quality", "PRIMARY_QUALITY_PASSED"),
}
_FEATURE_COMMON_FIELDS = frozenset(
    {"name", "family", "kind", "source_set", "unit", "formula_id"}
)
_BUILDER_CODE_PATHS = frozenset(
    {
        "src/portfolio_bot/data/secure_publication.py",
        "src/portfolio_bot/features/phase_03_contract.py",
        "src/portfolio_bot/features/feature_lineage.py",
        "src/portfolio_bot/features/feature_math.py",
        "src/portfolio_bot/features/crsp_features.py",
        "src/portfolio_bot/features/fast_crsp_features.py",
        "src/portfolio_bot/features/crsp_cross_section.py",
        "src/portfolio_bot/features/fundamental_features.py",
        "src/portfolio_bot/features/ownership_event_features.py",
        "src/portfolio_bot/features/optionmetrics_features.py",
        "src/portfolio_bot/features/aggregate_state_features.py",
        "src/portfolio_bot/features/feature_store_contract.py",
        "src/portfolio_bot/features/feature_store_inputs.py",
        "src/portfolio_bot/features/feature_store_materialization.py",
        "src/portfolio_bot/features/materialize_crsp.py",
        "src/portfolio_bot/features/materialize_crsp_streaming.py",
        "src/portfolio_bot/features/materialize_crsp_routing.py",
        "src/portfolio_bot/features/materialize_fundamentals.py",
        "src/portfolio_bot/features/materialize_ownership_events.py",
        "src/portfolio_bot/features/materialize_optionmetrics.py",
        "src/portfolio_bot/features/materialize_aggregate_state.py",
        "src/portfolio_bot/features/feature_store_registries.py",
        "src/portfolio_bot/features/feature_store_quality.py",
        "src/portfolio_bot/features/feature_store_family_orchestration.py",
        "src/portfolio_bot/features/feature_store_publication.py",
        "scripts/build_stage_06_feature_store.py",
    }
)


class Stage6VerificationError(RuntimeError):
    """Raised when independent verification cannot proceed safely."""


@dataclass(frozen=True)
class Difference:
    code: str
    expected: Any
    observed: Any
    details: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "expected": self.expected,
            "observed": self.observed,
            "details": self.details,
        }


@dataclass(frozen=True)
class InputReceipt:
    gate: str
    name: str
    sha256: str
    size_bytes: int
    rows: int
    row_groups: int
    path: Path

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "row_groups": self.row_groups,
        }


@dataclass(frozen=True)
class FrozenContract:
    project_root: Path
    policy_path: Path
    policy: Mapping[str, Any]
    semantic: Mapping[str, Any]
    rules: Mapping[str, Any]
    registry: Mapping[str, Any]
    ff12: Mapping[str, Any]
    features: tuple[Mapping[str, Any], ...]
    family_order: tuple[str, ...]
    family_features: Mapping[str, tuple[str, ...]]
    raw_schema: pa.Schema
    lineage_schema: pa.Schema
    published_schemas: Mapping[str, pa.Schema]
    output_files: Mapping[str, tuple[str, ...]]
    output_roots: Mapping[str, str]
    feature_sets: Mapping[str, tuple[str, ...]]
    value_statuses: frozenset[str]
    configuration_hashes: Mapping[str, str]


@dataclass(frozen=True)
class AuthenticatedOutput:
    kind: str
    path: Path
    files: Mapping[str, Path]
    hashes: Mapping[str, str]


@dataclass(frozen=True)
class RawAudit:
    raw_rows: int
    raw_key_differences: int
    raw_duplicate_or_order_errors: int
    raw_audit_value_differences: int
    nonfinite_numeric_values: int
    invalid_binary_values: int
    invalid_categorical_values: int
    signal_dates: int
    first_signal_date: dt.date | None
    last_signal_date: dt.date | None
    signal_year_counts: Mapping[int, int]
    dynamic_categories: Mapping[str, tuple[str, ...]]
    structural_missingness: Mapping[str, int]


@dataclass(frozen=True)
class LineageAudit:
    lineage_rows: int
    lineage_key_differences: int
    duplicate_lineage_keys: int
    unknown_feature_names: int
    unknown_value_statuses: int
    value_status_mismatches: int
    observation_date_after_signal: int
    public_availability_date_after_signal: int
    stale_feature_uses: int
    present_values_missing_dates: int
    missing_values_with_dates: int
    present_values: int
    missing_values: int
    structural_missing_values: int
    option_noncoverage_zeros: int


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    build_id: str | None
    verification_id: str | None
    checks: Mapping[str, Any]
    differences: tuple[Difference, ...]
    verification_path: Path | None
    verified_pointer_path: Path | None


def _canonical_json_bytes(value: Any) -> bytes:
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return _sha256_bytes(schema.serialize().to_pybytes())


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise Stage6VerificationError(f"Expected JSON object: {path}")
    return value


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise Stage6VerificationError(f"Expected TOML object: {path}")
    return value


def _project_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _regular_file(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    try:
        observed = os.lstat(lexical)
    except OSError as exc:
        raise Stage6VerificationError(f"Missing {label}: {lexical}") from exc
    if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise Stage6VerificationError(f"Unsafe {label}: {lexical}")
    return lexical


def _safe_relative(value: str, prefix: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise Stage6VerificationError(f"Unsafe relative namespace: {value}")
    if path.parts[0] != prefix:
        raise Stage6VerificationError(f"Wrong lifecycle namespace: {value}")
    return path.as_posix()


def _parse_column(value: str) -> pa.Field:
    parts = value.split(":")
    if len(parts) != 3:
        raise Stage6VerificationError(f"Invalid frozen column: {value}")
    name, type_name, nullability = parts
    types = {
        "utf8": pa.string(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
    }
    if type_name not in types or nullability not in {"nullable", "not_null"}:
        raise Stage6VerificationError(f"Unknown frozen column type: {value}")
    return pa.field(
        name,
        types[type_name],
        nullable=nullability == "nullable",
    )


def _expand_features(registry: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    inherited = ("family", "kind", "source_set", "unit", "formula_id")
    blocks = registry.get("blocks")
    if not isinstance(blocks, list):
        raise Stage6VerificationError("Feature registry blocks are absent")
    for block in blocks:
        if not isinstance(block, dict) or not all(
            key in block for key in (*inherited, "members")
        ):
            raise Stage6VerificationError("Feature block is incomplete")
        members = block["members"]
        if not isinstance(members, list) or not members:
            raise Stage6VerificationError("Feature block is empty")
        common = {key: block[key] for key in inherited}
        for member in members:
            if not isinstance(member, dict) or not isinstance(
                member.get("name"), str
            ):
                raise Stage6VerificationError("Feature member is invalid")
            result.append({**common, **member})
    names = [str(item["name"]) for item in result]
    if len(names) != len(set(names)):
        raise Stage6VerificationError("Feature registry has duplicate names")
    return tuple(result)


def _raw_schema(features: Sequence[Mapping[str, Any]]) -> pa.Schema:
    type_by_kind = {
        "continuous_firm": pa.float64(),
        "aggregate_state": pa.float64(),
        "binary_firm": pa.int8(),
        "categorical_firm": pa.string(),
        "categorical_multihot_firm": pa.list_(pa.string()),
    }
    fields = list(_AUDIT_SCHEMA)
    for feature in features:
        kind = str(feature["kind"])
        if kind not in type_by_kind:
            raise Stage6VerificationError(f"Unknown feature kind: {kind}")
        fields.append(
            pa.field(
                str(feature["name"]),
                type_by_kind[kind],
                nullable=kind
                not in {"categorical_firm", "categorical_multihot_firm"},
            )
        )
    return pa.schema(fields)


def _resolve_feature_sets(
    features: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    ordered = [str(feature["name"]) for feature in features]
    by_name = {str(feature["name"]): feature for feature in features}
    specs = rules.get("feature_sets")
    if not isinstance(specs, dict):
        raise Stage6VerificationError("Feature-set rules are absent")
    result: dict[str, tuple[str, ...]] = {}
    for set_name, spec in specs.items():
        if not isinstance(spec, dict):
            raise Stage6VerificationError("Feature-set rule is invalid")
        if spec.get("include_all_registry_features") is True:
            selected = set(ordered)
        else:
            families = set(spec.get("include_families", []))
            selected = {
                name
                for name, feature in by_name.items()
                if feature["family"] in families
            }
            explicit = set(spec.get("include_feature_names", []))
            if not explicit.issubset(by_name):
                raise Stage6VerificationError(
                    f"Unknown explicit feature in {set_name}"
                )
            selected.update(explicit)
        excluded = set(spec.get("exclude_feature_names", []))
        if not excluded.issubset(by_name):
            raise Stage6VerificationError(
                f"Unknown excluded feature in {set_name}"
            )
        selected.difference_update(excluded)
        values = tuple(name for name in ordered if name in selected)
        if len(values) != int(spec["expected_count"]):
            raise Stage6VerificationError(
                f"Feature-set count differs: {set_name}"
            )
        result[str(set_name)] = values
    return result


def reconstruct_frozen_contract(
    policy_path: Path = DEFAULT_POLICY,
    *,
    enforce_reviewed_policy: bool = True,
) -> FrozenContract:
    """Independently reconstruct every Stage 6 physical verification binding."""

    policy_path = _regular_file(policy_path.resolve(), "Stage 6 policy")
    if (
        enforce_reviewed_policy
        and _sha256_file(policy_path) != REVIEWED_POLICY_SHA256
    ):
        raise Stage6VerificationError(
            "Stage 6 policy is not the reviewed byte artifact"
        )
    root = policy_path.parents[2]
    policy = _toml(policy_path)
    if (
        policy.get("schema_version") != 1
        or policy.get("stage") != 6
        or policy.get("status")
        != "frozen_development_only_before_primary_feature_build"
    ):
        raise Stage6VerificationError("Stage 6 policy header differs")

    linked = {
        "semantic": (
            policy["semantic_contract"]["policy_path"],
            policy["semantic_contract"]["policy_sha256"],
        ),
        "registry": (
            policy["semantic_contract"]["registry_path"],
            policy["semantic_contract"]["registry_sha256"],
        ),
        "contract_pointer": (
            policy["semantic_contract"]["verified_pointer"],
            policy["semantic_contract"]["verified_pointer_sha256"],
        ),
        "rules": (
            policy["physical_rules"]["path"],
            policy["physical_rules"]["sha256"],
        ),
        "ff12": (
            policy["ff12_mapping"]["path"],
            policy["ff12_mapping"]["sha256"],
        ),
        "paths": (
            policy["database"]["paths_config"],
            policy["database"]["paths_config_sha256"],
        ),
        "relocation": (
            policy["database"]["relocation_receipt"],
            policy["database"]["relocation_receipt_sha256"],
        ),
    }
    linked_paths: dict[str, Path] = {}
    for label, (value, expected_hash) in linked.items():
        path = _regular_file(_project_path(root, value), label)
        observed = _sha256_file(path)
        if observed != expected_hash:
            raise Stage6VerificationError(
                f"Frozen linked hash differs: {label}"
            )
        linked_paths[label] = path

    semantic = _toml(linked_paths["semantic"])
    registry = _json(linked_paths["registry"])
    rules = _json(linked_paths["rules"])
    ff12 = _json(linked_paths["ff12"])
    pointer = _json(linked_paths["contract_pointer"])
    if (
        semantic.get("contract_id") != policy["contract_id"]
        or pointer.get("status") != "verified_passed"
        or pointer.get("build_id")
        != policy["semantic_contract"]["required_contract_build_id"]
        or pointer.get("verification_id")
        != policy["semantic_contract"]["required_contract_verification_id"]
        or pointer.get("authorizations", {}).get(
            policy["semantic_contract"]["required_authorization"]
        )
        is not True
    ):
        raise Stage6VerificationError(
            "Verified semantic authorization differs"
        )
    if (
        rules.get("rules_id")
        != policy["physical_rules"]["required_rules_id"]
        or registry.get("registry_id")
        != semantic["feature_registry"]["required_registry_id"]
        or ff12.get("mapping_id")
        != policy["ff12_mapping"]["required_mapping_id"]
    ):
        raise Stage6VerificationError("Frozen contract identifier differs")

    features = _expand_features(registry)
    family_order = tuple(str(value) for value in registry["family_order"])
    family_features = {
        family: tuple(
            str(feature["name"])
            for feature in features
            if feature["family"] == family
        )
        for family in family_order
    }
    if (
        len(features) != 163
        or tuple(policy["build"]["family_order"]) != family_order
        or any(not names for names in family_features.values())
    ):
        raise Stage6VerificationError("Frozen feature/family count differs")

    published_specs = rules["published_table_schemas"]
    published_schemas = {
        name: pa.schema(
            [_parse_column(value) for value in specification["columns"]]
        )
        for name, specification in published_specs.items()
    }
    output_files = {
        kind: tuple(str(value) for value in policy["outputs"][kind][
            "required_files"
        ])
        for kind in ("interim", "processed", "quality")
    }
    for kind, names in output_files.items():
        if len(names) != len(set(names)) or any(
            Path(name).name != name or _SAFE_NAME.fullmatch(name) is None
            for name in names
        ):
            raise Stage6VerificationError(
                f"Unsafe closed file set: {kind}"
            )
    output_roots = {
        "interim": _safe_relative(
            str(policy["database"]["interim_build_root"]), "interim"
        ),
        "processed": _safe_relative(
            str(policy["database"]["processed_build_root"]), "processed"
        ),
        "quality": _safe_relative(
            str(policy["database"]["quality_build_root"]), "quality"
        ),
    }
    configuration_paths = (
        policy_path,
        linked_paths["semantic"],
        linked_paths["registry"],
        linked_paths["contract_pointer"],
        linked_paths["rules"],
        linked_paths["ff12"],
        linked_paths["paths"],
        linked_paths["relocation"],
    )
    configuration_hashes: dict[str, str] = {}
    for path in configuration_paths:
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = path.as_posix()
        configuration_hashes[label] = _sha256_file(path)

    statuses = frozenset(
        str(value)
        for value in rules["physical_store"]["value_status_vocabulary"]
    )
    if statuses != _VALUE_STATUSES:
        raise Stage6VerificationError("Value-status vocabulary differs")
    raw = _raw_schema(features)
    if int(policy["scope"]["required_lineage_rows"]) != (
        int(policy["scope"]["required_spine_rows"]) * len(features)
    ):
        raise Stage6VerificationError("Frozen lineage arithmetic differs")
    return FrozenContract(
        project_root=root,
        policy_path=policy_path,
        policy=policy,
        semantic=semantic,
        rules=rules,
        registry=registry,
        ff12=ff12,
        features=features,
        family_order=family_order,
        family_features=family_features,
        raw_schema=raw,
        lineage_schema=_LINEAGE_SCHEMA,
        published_schemas=published_schemas,
        output_files=output_files,
        output_roots=output_roots,
        feature_sets=_resolve_feature_sets(features, rules),
        value_statuses=statuses,
        configuration_hashes=dict(sorted(configuration_hashes.items())),
    )


def _manifest_output_receipts(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    outputs = manifest.get("outputs")
    if isinstance(outputs, list):
        candidates.extend(
            item for item in outputs if isinstance(item, Mapping)
        )
    elif isinstance(outputs, Mapping):
        for key, value in outputs.items():
            if isinstance(value, Mapping):
                candidates.append({"table_key": key, **value})
    output = manifest.get("output")
    if isinstance(output, Mapping):
        candidates.append(output)
    result: dict[str, Mapping[str, Any]] = {}
    for item in candidates:
        value = item.get("path")
        if not isinstance(value, str) or not value.endswith(".parquet"):
            continue
        name = Path(value).name
        if name in result:
            raise Stage6VerificationError(
                f"Duplicate upstream output receipt: {name}"
            )
        result[name] = item
    return result


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
                raise Stage6VerificationError(
                    "Upstream output escaped authenticated roots"
                ) from exc
        candidate = database_root / relative
    elif len(raw.parts) > 1:
        candidate = database_root / raw
    elif interim_path is not None:
        candidate = database_root / interim_path / raw
    else:
        raise Stage6VerificationError(
            f"Cannot resolve upstream output: {value}"
        )
    lexical = Path(os.path.abspath(candidate))
    try:
        lexical.relative_to(database_root)
    except ValueError as exc:
        raise Stage6VerificationError(
            "Resolved upstream input escaped database root"
        ) from exc
    return lexical


def reconstruct_input_receipts(
    contract: FrozenContract,
    *,
    database_root: Path,
) -> tuple[InputReceipt, ...]:
    """Rehash and reconstruct the exact 26-table Stage 6 input allowlist."""

    database_root = Path(os.path.abspath(database_root.expanduser()))
    if database_root != Path(
        os.path.abspath(
            Path(str(contract.policy["database"]["root"])).expanduser()
        )
    ):
        raise Stage6VerificationError(
            "Active database root differs from frozen policy"
        )
    root_stat = os.lstat(database_root)
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(
        root_stat.st_mode
    ):
        raise Stage6VerificationError("Database root is unsafe")
    historical = Path(
        str(contract.semantic["database"]["historical_root"])
    ).expanduser()
    relocation = _json(
        _regular_file(
            _project_path(
                contract.project_root,
                contract.policy["database"]["relocation_receipt"],
            ),
            "database relocation receipt",
        )
    )
    if (
        relocation.get("new_database_root")
        != database_root.as_posix()
        or relocation.get("old_database_root")
        != historical.as_posix()
    ):
        raise Stage6VerificationError(
            "Database relocation binding differs"
        )
    receipts: list[InputReceipt] = []
    seen: set[str] = set()
    for gate, specification in contract.semantic["upstream"].items():
        manifest_path = _regular_file(
            _project_path(
                contract.project_root,
                specification["primary_manifest"],
            ),
            f"{gate} upstream manifest",
        )
        if _sha256_file(manifest_path) != specification[
            "primary_manifest_sha256"
        ]:
            raise Stage6VerificationError(
                f"Upstream manifest hash differs: {gate}"
            )
        manifest = _json(manifest_path)
        outputs = _manifest_output_receipts(manifest)
        interim = manifest.get("interim_path")
        if interim is not None and not isinstance(interim, str):
            raise Stage6VerificationError(
                f"Invalid upstream interim path: {gate}"
            )
        for name in specification["required_tables"]:
            if name in seen or name not in outputs:
                raise Stage6VerificationError(
                    f"Upstream allowlist differs: {name}"
                )
            seen.add(name)
            source = outputs[name]
            path = _regular_file(
                _relocated_input_path(
                    str(source["path"]),
                    database_root=database_root,
                    historical_root=historical,
                    interim_path=interim,
                ),
                f"input table {name}",
            )
            parquet = pq.ParquetFile(path)
            observed_hash = _sha256_file(path)
            expected_hash = source.get("sha256")
            expected_size = source.get("size_bytes")
            expected_rows = source.get("rows")
            if (
                not isinstance(expected_hash, str)
                or observed_hash != expected_hash
                or path.stat().st_size != expected_size
                or (
                    expected_rows is not None
                    and parquet.metadata.num_rows != int(expected_rows)
                )
            ):
                raise Stage6VerificationError(
                    f"Input receipt differs: {name}"
                )
            receipts.append(
                InputReceipt(
                    gate=str(gate),
                    name=str(name),
                    sha256=observed_hash,
                    size_bytes=path.stat().st_size,
                    rows=parquet.metadata.num_rows,
                    row_groups=parquet.metadata.num_row_groups,
                    path=path,
                )
            )
    receipts.sort(key=lambda item: item.name)
    if len(receipts) != 26:
        raise Stage6VerificationError(
            "Stage 6 input allowlist is not exactly 26 tables"
        )
    return tuple(receipts)


def _exact_keys(value: Any, expected: frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise Stage6VerificationError(f"{label} keys differ")


def _safe_hash_map(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise Stage6VerificationError(f"{label} is not a nonempty map")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise Stage6VerificationError(f"{label} is invalid")
        result[key] = digest
    return dict(sorted(result.items()))


def _read_primary(
    contract: FrozenContract,
    *,
    primary_manifest_path: Path,
    primary_pointer_path: Path,
    artifact_root: Path,
) -> tuple[dict[str, Any], str]:
    pointer_path = _regular_file(
        primary_pointer_path, "Stage 6 primary pointer"
    )
    pointer_bytes = pointer_path.read_bytes()
    pointer = json.loads(pointer_bytes)
    pointer_keys = frozenset(
        contract.rules["publication_payloads"][
            "primary_pointer_required_keys"
        ]
    )
    _exact_keys(pointer, pointer_keys, "Primary pointer")
    build_id = pointer.get("build_id")
    if (
        pointer_bytes != _canonical_json_bytes(pointer)
        or
        pointer.get("schema_version") != 1
        or pointer.get("contract_id") != contract.policy["contract_id"]
        or pointer.get("policy_id") != contract.policy["policy_id"]
        or not isinstance(build_id, str)
        or _BUILD_ID.fullmatch(build_id) is None
        or pointer.get("status")
        != "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
        or pointer.get("authorizations") != _PRIMARY_AUTHORIZATIONS
        or _SHA256.fullmatch(
            str(pointer.get("primary_manifest_sha256", ""))
        )
        is None
    ):
        raise Stage6VerificationError("Primary pointer binding differs")
    expected_manifest = (
        artifact_root
        / str(contract.policy["publication"]["primary_build_namespace"])
        / build_id
        / "manifest.json"
    )
    manifest_path = _regular_file(
        primary_manifest_path, "Stage 6 primary manifest"
    )
    if manifest_path != Path(os.path.abspath(expected_manifest)):
        raise Stage6VerificationError(
            "Primary manifest path is not the frozen build artifact"
        )
    payload = manifest_path.read_bytes()
    manifest_hash = _sha256_bytes(payload)
    if manifest_hash != pointer["primary_manifest_sha256"]:
        raise Stage6VerificationError(
            "Primary manifest hash differs from pointer"
        )
    manifest = json.loads(payload)
    _exact_keys(manifest, _PRIMARY_KEYS, "Primary manifest")
    _exact_keys(manifest["bindings"], _BINDING_KEYS, "Manifest bindings")
    _exact_keys(manifest["inputs"], _INPUT_KEYS, "Manifest inputs")
    _exact_keys(manifest["outputs"], frozenset(_SUCCESS_ENUMS), "Manifest outputs")
    _exact_keys(manifest["summary"], _SUMMARY_KEYS, "Manifest summary")
    _exact_keys(
        manifest["authorizations"],
        _AUTHORIZATION_KEYS,
        "Manifest authorizations",
    )
    if (
        payload != _canonical_json_bytes(manifest)
        or
        manifest.get("schema_version") != 1
        or manifest.get("contract_id") != contract.policy["contract_id"]
        or manifest.get("policy_id") != contract.policy["policy_id"]
        or manifest.get("build_id") != build_id
        or manifest.get("status")
        != "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
        or manifest["authorizations"] != _PRIMARY_AUTHORIZATIONS
    ):
        raise Stage6VerificationError("Primary manifest header differs")
    return manifest, manifest_hash


def _resolved_output_path(
    contract: FrozenContract,
    *,
    database_root: Path,
    build_id: str,
    kind: str,
    value: Any,
) -> Path:
    if not isinstance(value, str):
        raise Stage6VerificationError(f"{kind} output path is invalid")
    supplied = Path(value).expanduser()
    resolved = (
        Path(os.path.abspath(supplied))
        if supplied.is_absolute()
        else Path(os.path.abspath(database_root / supplied))
    )
    expected = Path(
        os.path.abspath(
            database_root / contract.output_roots[kind] / build_id
        )
    )
    if resolved != expected:
        raise Stage6VerificationError(
            f"{kind} output path differs from frozen namespace"
        )
    observed = os.lstat(resolved)
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(
        observed.st_mode
    ):
        raise Stage6VerificationError(f"{kind} output directory is unsafe")
    return resolved


def _authenticate_output(
    contract: FrozenContract,
    manifest: Mapping[str, Any],
    *,
    database_root: Path,
    kind: str,
) -> AuthenticatedOutput:
    specification = manifest["outputs"][kind]
    _exact_keys(specification, _OUTPUT_KEYS, f"{kind} manifest output")
    build_id = str(manifest["build_id"])
    path = _resolved_output_path(
        contract,
        database_root=database_root,
        build_id=build_id,
        kind=kind,
        value=specification["path"],
    )
    expected_names = set(contract.output_files[kind])
    observed_names = {
        item.name
        for item in path.iterdir()
        if item.name not in {".", ".."}
    }
    if observed_names != expected_names:
        raise Stage6VerificationError(
            f"{kind} output closed file set differs"
        )
    files_spec = specification["files"]
    if not isinstance(files_spec, Mapping) or set(files_spec) != expected_names:
        raise Stage6VerificationError(
            f"{kind} manifest file set differs"
        )
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    actual_receipts: dict[str, dict[str, Any]] = {}
    for name in contract.output_files[kind]:
        file_path = _regular_file(path / name, f"{kind} output {name}")
        receipt = files_spec[name]
        _exact_keys(receipt, _FILE_RECEIPT_KEYS, f"{kind}/{name} receipt")
        observed_hash = _sha256_file(file_path)
        observed_size = file_path.stat().st_size
        if not name.endswith(".parquet"):
            observed_rows = observed_groups = observed_schema = None
        else:
            parquet = pq.ParquetFile(file_path)
            observed_rows = parquet.metadata.num_rows
            observed_groups = parquet.metadata.num_row_groups
            observed_schema = _schema_sha256(parquet.schema_arrow)
        observed_receipt = {
            "path": name,
            "sha256": observed_hash,
            "size_bytes": observed_size,
            "rows": observed_rows,
            "row_groups": observed_groups,
            "schema_sha256": observed_schema,
        }
        if dict(receipt) != observed_receipt:
            raise Stage6VerificationError(
                f"{kind}/{name} receipt differs from bytes"
            )
        paths[name] = file_path
        hashes[name] = observed_hash
        if name != "_SUCCESS.json":
            actual_receipts[name] = observed_receipt
    if specification["success_sha256"] != hashes["_SUCCESS.json"]:
        raise Stage6VerificationError(
            f"{kind} success hash binding differs"
        )
    success_bytes = paths["_SUCCESS.json"].read_bytes()
    success = json.loads(success_bytes)
    _exact_keys(success, _SUCCESS_KEYS, f"{kind} success receipt")
    location, status = _SUCCESS_ENUMS[kind]
    bindings = manifest["bindings"]
    if (
        success.get("schema_version") != 1
        or success.get("contract_id") != contract.policy["contract_id"]
        or success.get("policy_id") != contract.policy["policy_id"]
        or success.get("build_id") != build_id
        or success.get("location_kind") != location
        or success.get("status") != status
        or success.get("configuration_hashes")
        != bindings["configuration_hashes"]
        or success.get("code_hashes") != bindings["code_hashes"]
        or success.get("input_receipts_sha256")
        != bindings["input_receipts_sha256"]
        or success.get("table_receipts") != actual_receipts
        or success_bytes != _canonical_json_bytes(success)
    ):
        raise Stage6VerificationError(
            f"{kind} success receipt binding differs"
        )
    return AuthenticatedOutput(
        kind=kind,
        path=path,
        files=paths,
        hashes=hashes,
    )


def verify_build_identity(
    contract: FrozenContract,
    manifest: Mapping[str, Any],
    input_receipts: Sequence[InputReceipt],
    *,
    required_code_paths: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Recompute the content fingerprint without calling builder identity code."""

    bindings = manifest["bindings"]
    configuration = _safe_hash_map(
        bindings["configuration_hashes"], "Configuration hashes"
    )
    if configuration != dict(contract.configuration_hashes):
        raise Stage6VerificationError(
            "Manifest configuration hashes differ"
        )
    code = _safe_hash_map(bindings["code_hashes"], "Code hashes")
    required = (
        _BUILDER_CODE_PATHS
        if required_code_paths is None
        else required_code_paths
    )
    if set(code) != set(required):
        raise Stage6VerificationError(
            "Manifest code-hash set differs from the independent allowlist"
        )
    for label, expected_hash in code.items():
        pure = PurePosixPath(label)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or pure.suffix != ".py"
        ):
            raise Stage6VerificationError(
                f"Unsafe code binding: {label}"
            )
        path = _regular_file(
            contract.project_root / pure, f"code binding {label}"
        )
        if _sha256_file(path) != expected_hash:
            raise Stage6VerificationError(
                f"Code binding hash differs: {label}"
            )
    ordered_receipts = [
        item.manifest_dict()
        for item in sorted(input_receipts, key=lambda value: value.name)
    ]
    if manifest["inputs"]["table_receipts"] != ordered_receipts:
        raise Stage6VerificationError(
            "Manifest input receipts differ"
        )
    receipts_hash = _sha256_bytes(
        _canonical_json_bytes(ordered_receipts)
    )
    if bindings["input_receipts_sha256"] != receipts_hash:
        raise Stage6VerificationError(
            "Input-receipt aggregate hash differs"
        )
    expected_input_builds = {
        "calendar": contract.policy["inputs"]["calendar_build_id"],
        "typed_backbone":
            contract.policy["inputs"]["typed_backbone_build_id"],
        "return_reconciliation":
            contract.policy["inputs"]["return_build_id"],
        "signal_universe":
            contract.policy["inputs"]["signal_universe_build_id"],
        "auxiliary": contract.policy["inputs"]["auxiliary_build_id"],
    }
    if manifest["inputs"]["input_builds"] != expected_input_builds:
        raise Stage6VerificationError("Manifest input builds differ")
    content = {
        "schema_version": 1,
        "contract_id": str(contract.policy["contract_id"]),
        "policy_id": str(contract.policy["policy_id"]),
        "configuration_hashes": configuration,
        "code_hashes": code,
        "input_receipts": ordered_receipts,
    }
    fingerprint = _sha256_bytes(_canonical_json_bytes(content))
    build_id = f"phase03-features-{fingerprint[:24]}"
    if (
        bindings["content_fingerprint"] != fingerprint
        or manifest["build_id"] != build_id
    ):
        raise Stage6VerificationError(
            "Deterministic feature-store build identity differs"
        )
    return {
        "content_fingerprint": fingerprint,
        "build_id": build_id,
        "input_receipts_sha256": receipts_hash,
        "configuration_hashes": configuration,
        "code_hashes": code,
    }


class _ParquetCursor:
    def __init__(
        self,
        path: Path,
        *,
        columns: Sequence[str] | None = None,
        batch_size: int = 65_536,
    ) -> None:
        self._schema = pq.ParquetFile(path).schema_arrow
        self._iterator = pq.ParquetFile(path).iter_batches(
            batch_size=batch_size,
            columns=None if columns is None else list(columns),
            use_threads=False,
        )
        self._batch: pa.RecordBatch | None = None
        self._offset = 0
        self.rows_read = 0

    def take(self, count: int) -> pa.Table:
        if count < 0:
            raise Stage6VerificationError("Negative cursor request")
        remaining = count
        pieces: list[pa.RecordBatch] = []
        while remaining:
            if (
                self._batch is None
                or self._offset >= self._batch.num_rows
            ):
                try:
                    self._batch = next(self._iterator)
                except StopIteration:
                    break
                self._offset = 0
            available = self._batch.num_rows - self._offset
            used = min(available, remaining)
            pieces.append(self._batch.slice(self._offset, used))
            self._offset += used
            remaining -= used
            self.rows_read += used
        if not pieces:
            schema = (
                self._schema
                if self._batch is None
                else self._batch.schema
            )
            return pa.Table.from_batches([], schema=schema)
        return pa.Table.from_batches(pieces)

    def exhausted(self) -> bool:
        if self._batch is not None and self._offset < self._batch.num_rows:
            return False
        try:
            self._batch = next(self._iterator)
        except StopIteration:
            return True
        self._offset = 0
        return self._batch.num_rows == 0 and self.exhausted()


def _normalize_category(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value).strip().upper()
    return normalized or None


def _count_true(value: pa.Array | pa.ChunkedArray) -> int:
    cleaned = pc.fill_null(value, False)
    result = pc.sum(pc.cast(cleaned, pa.int64())).as_py()
    return int(result or 0)


def _schema_equal(observed: pa.Schema, expected: pa.Schema) -> bool:
    return observed.equals(expected, check_metadata=True)


def _parquet_physical_schema(schema: pa.Schema) -> pa.Schema:
    fields: list[pa.Field] = []
    for field in schema:
        field_type = field.type
        if pa.types.is_list(field_type):
            field_type = pa.list_(
                pa.field(
                    "element",
                    field_type.value_type,
                    nullable=field_type.value_field.nullable,
                )
            )
        fields.append(
            pa.field(
                field.name,
                field_type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        )
    return pa.schema(fields, metadata=schema.metadata)


def audit_raw_store(
    contract: FrozenContract,
    *,
    raw_path: Path,
    spine_path: Path,
    batch_rows: int = 16_384,
) -> RawAudit:
    """Stream every raw row against the authenticated signal-universe spine."""

    if not isinstance(batch_rows, int) or batch_rows <= 0:
        raise Stage6VerificationError("Raw audit batch_rows is invalid")
    raw_file = pq.ParquetFile(raw_path)
    spine_file = pq.ParquetFile(spine_path)
    if not _schema_equal(
        raw_file.schema_arrow,
        _parquet_physical_schema(contract.raw_schema),
    ):
        raise Stage6VerificationError("Raw feature-store schema differs")
    spine_by_name = {
        field.name: field for field in spine_file.schema_arrow
    }
    if any(
        field.name not in spine_by_name
        or spine_by_name[field.name].type != field.type
        for field in _AUDIT_SCHEMA
    ):
        raise Stage6VerificationError(
            "Signal spine audit-column schema differs"
        )
    raw_cursor = _ParquetCursor(raw_path, batch_size=batch_rows)
    spine_cursor = _ParquetCursor(
        spine_path,
        columns=_AUDIT_SCHEMA.names,
        batch_size=batch_rows,
    )
    feature_by_name = {
        str(feature["name"]): feature for feature in contract.features
    }
    categorical = tuple(
        feature
        for feature in contract.features
        if feature["kind"]
        in {"categorical_firm", "categorical_multihot_firm"}
    )
    dynamic_categories: dict[str, set[str]] = {
        str(feature["name"]): set()
        for feature in categorical
        if not isinstance(feature.get("categories"), list)
        and feature["name"] != "ff12_sector_category"
    }
    raw_rows = key_differences = audit_differences = 0
    duplicate_or_order = nonfinite = invalid_binary = invalid_category = 0
    signal_dates: set[dt.date] = set()
    years: dict[int, int] = {}
    first_signal: dt.date | None = None
    last_signal: dt.date | None = None
    prior_key: tuple[dt.date, int] | None = None
    structural = {
        "segment_feature_values": 0,
        "quarterly_share_growth_values": 0,
        "qualifying_financial_results_6k_values": 0,
    }
    segment_names = (
        "business_segment_count",
        "geographic_segment_count",
        "business_segment_revenue_hhi",
        "business_segment_revenue_hhi_change_1y",
    )

    while True:
        raw = raw_cursor.take(batch_rows)
        if raw.num_rows == 0:
            break
        spine = spine_cursor.take(raw.num_rows)
        if spine.num_rows != raw.num_rows:
            key_differences += raw.num_rows - spine.num_rows
            break
        raw_rows += raw.num_rows
        for field in _AUDIT_SCHEMA:
            if not raw[field.name].equals(spine[field.name]):
                left = raw[field.name]
                right = spine[field.name]
                if len(left) == len(right):
                    audit_differences += _count_true(
                        pc.invert(
                            pc.fill_null(pc.equal(left, right), False)
                        )
                    )
                else:
                    audit_differences += abs(len(left) - len(right))
        raw_signals = raw["signal_date"].combine_chunks()
        raw_permnos = raw["permno"].combine_chunks()
        signal_values = raw_signals.to_pylist()
        permno_values = raw_permnos.to_pylist()
        for signal, permno in zip(
            signal_values, permno_values, strict=True
        ):
            key = (signal, int(permno))
            if prior_key is not None and key <= prior_key:
                duplicate_or_order += 1
            prior_key = key
            signal_dates.add(signal)
            years[signal.year] = years.get(signal.year, 0) + 1
            first_signal = (
                signal
                if first_signal is None
                else min(first_signal, signal)
            )
            last_signal = (
                signal
                if last_signal is None
                else max(last_signal, signal)
            )

        long_mask = raw["long_eligible"].combine_chunks().to_numpy(
            zero_copy_only=False
        )
        for feature in contract.features:
            name = str(feature["name"])
            column = raw[name].combine_chunks()
            if feature["kind"] in {"continuous_firm", "aggregate_state"}:
                values = column.to_numpy(zero_copy_only=False)
                nulls = column.is_null().to_numpy(zero_copy_only=False)
                nonfinite += int(
                    np.count_nonzero(~nulls & ~np.isfinite(values))
                )
            elif feature["kind"] == "binary_firm":
                values = column.to_numpy(zero_copy_only=False)
                nulls = column.is_null().to_numpy(zero_copy_only=False)
                invalid_binary += int(
                    np.count_nonzero(
                        ~nulls & (values != 0) & (values != 1)
                    )
                )

        for feature in categorical:
            name = str(feature["name"])
            fixed_categories = feature.get("categories")
            if isinstance(fixed_categories, list):
                allowed = {
                    _normalize_category(value)
                    for value in fixed_categories
                }
            elif name == "ff12_sector_category":
                allowed = set(_ff12_categories(contract))
            else:
                allowed = None
            column_values = raw[name].to_pylist()
            for eligible, value in zip(
                long_mask, column_values, strict=True
            ):
                values = (
                    value
                    if feature["kind"] == "categorical_multihot_firm"
                    else (value,)
                )
                if not isinstance(values, (list, tuple)) or not values:
                    invalid_category += 1
                    continue
                normalized_values: list[str] = []
                for item in values:
                    normalized = _normalize_category(item)
                    if normalized is None:
                        invalid_category += 1
                        continue
                    normalized_values.append(normalized)
                    invalid_category += int(normalized != item)
                    if allowed is not None and normalized not in allowed:
                        invalid_category += 1
                    if name == "sic2_industry_category" and not (
                        normalized == "UNKNOWN_INDUSTRY"
                        or re.fullmatch(r"[0-9]{2}", normalized)
                    ):
                        invalid_category += 1
                    if name == "domicile_country_category" and not (
                        normalized
                        in {"UNKNOWN_COUNTRY", "UNKNOWN_FOREIGN"}
                        or re.fullmatch(r"[A-Z]{3}", normalized)
                    ):
                        invalid_category += 1
                if len(normalized_values) != len(set(normalized_values)):
                    invalid_category += 1
                if feature["kind"] == "categorical_multihot_firm":
                    reserved = {"UNKNOWN", "NO_EVENT"}.intersection(
                        normalized_values
                    )
                    if reserved and len(normalized_values) != 1:
                        invalid_category += 1
                if eligible and name in dynamic_categories:
                    dynamic_categories[name].update(normalized_values)

        for name in segment_names:
            if name in feature_by_name:
                structural["segment_feature_values"] += int(
                    raw[name].length()
                    - raw[name].null_count
                )
        if {
            "fundamental_source_basis",
            "share_growth_yoy",
        }.issubset(raw.column_names):
            quarterly = pc.equal(
                raw["fundamental_source_basis"],
                "TTM",
            )
            structural["quarterly_share_growth_values"] += _count_true(
                pc.and_(
                    pc.fill_null(quarterly, False),
                    pc.invert(raw["share_growth_yoy"].is_null()),
                )
            )
        if "latest_interim_financial_filing_form" in raw.column_names:
            structural[
                "qualifying_financial_results_6k_values"
            ] += _count_true(
                pc.equal(
                    raw["latest_interim_financial_filing_form"],
                    "QUALIFYING_FINANCIAL_RESULTS_6-K",
                )
            )

    if not raw_cursor.exhausted():
        raise Stage6VerificationError("Raw cursor did not exhaust")
    if not spine_cursor.exhausted():
        key_differences += (
            spine_file.metadata.num_rows - spine_cursor.rows_read
        )
    if raw_rows != raw_file.metadata.num_rows:
        raise Stage6VerificationError("Raw stream row count differs")
    return RawAudit(
        raw_rows=raw_rows,
        raw_key_differences=int(key_differences),
        raw_duplicate_or_order_errors=int(duplicate_or_order),
        raw_audit_value_differences=int(audit_differences),
        nonfinite_numeric_values=int(nonfinite),
        invalid_binary_values=int(invalid_binary),
        invalid_categorical_values=int(invalid_category),
        signal_dates=len(signal_dates),
        first_signal_date=first_signal,
        last_signal_date=last_signal,
        signal_year_counts=dict(sorted(years.items())),
        dynamic_categories={
            name: tuple(
                sorted(values, key=lambda item: item.encode("utf-8"))
            )
            for name, values in sorted(dynamic_categories.items())
        },
        structural_missingness=structural,
    )


def _adjacent_duplicate_count(
    table: pa.Table,
    previous: tuple[dt.date, int, str] | None,
) -> tuple[int, tuple[dt.date, int, str] | None]:
    if table.num_rows == 0:
        return 0, previous
    signals = table["signal_date"].combine_chunks()
    permnos = table["permno"].combine_chunks()
    features = table["feature_name"].combine_chunks()
    count = 0
    if previous is not None:
        first = (
            signals[0].as_py(),
            int(permnos[0].as_py()),
            str(features[0].as_py()),
        )
        count += int(first == previous)
    if table.num_rows > 1:
        same = pc.and_(
            pc.and_(
                pc.equal(signals.slice(1), signals.slice(0, len(signals) - 1)),
                pc.equal(permnos.slice(1), permnos.slice(0, len(permnos) - 1)),
            ),
            pc.equal(
                features.slice(1),
                features.slice(0, len(features) - 1),
            ),
        )
        count += _count_true(same)
    last = (
        signals[-1].as_py(),
        int(permnos[-1].as_py()),
        str(features[-1].as_py()),
    )
    return count, last


def _semantic_missing_mask(
    feature: Mapping[str, Any],
    values: pa.Array,
) -> np.ndarray:
    """Return missingness under the frozen raw-store representation.

    Categorical UNKNOWN states are intentionally retained as non-null raw
    values while their lineage records the applicable non-PRESENT cause.
    """

    kind = str(feature["kind"])
    if kind == "categorical_firm":
        return np.fromiter(
            (
                value is None
                or (
                    isinstance(value, str)
                    and value.startswith("UNKNOWN")
                )
                for value in values.to_pylist()
            ),
            dtype=np.bool_,
            count=len(values),
        )
    if kind == "categorical_multihot_firm":
        return np.fromiter(
            (
                value is None or value == ["UNKNOWN"]
                for value in values.to_pylist()
            ),
            dtype=np.bool_,
            count=len(values),
        )
    return values.is_null().to_numpy(zero_copy_only=False)


def audit_lineage(
    contract: FrozenContract,
    *,
    raw_path: Path,
    lineage_paths: Mapping[str, Path],
    batch_rows: int = 2_048,
) -> LineageAudit:
    """Stream-check every logical lineage key and its raw-value consistency."""

    if set(lineage_paths) != set(contract.family_order):
        raise Stage6VerificationError("Lineage family set differs")
    totals = {
        "lineage_rows": 0,
        "lineage_key_differences": 0,
        "duplicate_lineage_keys": 0,
        "unknown_feature_names": 0,
        "unknown_value_statuses": 0,
        "value_status_mismatches": 0,
        "observation_date_after_signal": 0,
        "public_availability_date_after_signal": 0,
        "stale_feature_uses": 0,
        "present_values_missing_dates": 0,
        "missing_values_with_dates": 0,
        "present_values": 0,
        "missing_values": 0,
        "structural_missing_values": 0,
        "option_noncoverage_zeros": 0,
    }
    feature_by_name = {
        str(feature["name"]): feature
        for feature in contract.features
    }
    for family in contract.family_order:
        lineage_path = lineage_paths[family]
        lineage_file = pq.ParquetFile(lineage_path)
        if not _schema_equal(
            lineage_file.schema_arrow, contract.lineage_schema
        ):
            raise Stage6VerificationError(
                f"Lineage schema differs: {family}"
            )
        names = contract.family_features[family]
        feature_count = len(names)
        expected_family_rows = (
            pq.ParquetFile(raw_path).metadata.num_rows * feature_count
        )
        if lineage_file.metadata.num_rows != expected_family_rows:
            totals["lineage_key_differences"] += abs(
                lineage_file.metadata.num_rows - expected_family_rows
            )
        raw_cursor = _ParquetCursor(raw_path, batch_size=batch_rows)
        lineage_cursor = _ParquetCursor(
            lineage_path,
            batch_size=batch_rows * feature_count,
        )
        previous: tuple[dt.date, int, str] | None = None
        while True:
            raw = raw_cursor.take(batch_rows)
            if raw.num_rows == 0:
                break
            needed = raw.num_rows * feature_count
            lineage = lineage_cursor.take(needed)
            if lineage.num_rows != needed:
                totals["lineage_key_differences"] += (
                    needed - lineage.num_rows
                )
                break
            totals["lineage_rows"] += lineage.num_rows
            raw_signals = raw["signal_date"].combine_chunks().to_numpy(
                zero_copy_only=False
            )
            raw_permnos = raw["permno"].combine_chunks().to_numpy(
                zero_copy_only=False
            )
            expected_signal = pa.array(
                np.repeat(raw_signals, feature_count),
                type=pa.date32(),
            )
            expected_permno = pa.array(
                np.repeat(raw_permnos, feature_count),
                type=pa.int64(),
            )
            expected_feature = pa.array(
                list(names) * raw.num_rows,
                type=pa.string(),
            )
            for actual, expected in (
                (lineage["signal_date"].combine_chunks(), expected_signal),
                (lineage["permno"].combine_chunks(), expected_permno),
                (lineage["feature_name"].combine_chunks(), expected_feature),
            ):
                totals["lineage_key_differences"] += _count_true(
                    pc.invert(
                        pc.fill_null(pc.equal(actual, expected), False)
                    )
                )
            totals["unknown_feature_names"] += _count_true(
                pc.invert(
                    pc.fill_null(
                        pc.is_in(
                            lineage["feature_name"].combine_chunks(),
                            value_set=pa.array(
                                list(names), type=pa.string()
                            ),
                        ),
                        False,
                    )
                )
            )
            duplicates, previous = _adjacent_duplicate_count(
                lineage, previous
            )
            totals["duplicate_lineage_keys"] += duplicates

            status = lineage["value_status"].combine_chunks()
            present = pc.equal(status, "PRESENT")
            totals["present_values"] += _count_true(present)
            totals["missing_values"] += (
                lineage.num_rows - _count_true(present)
            )
            totals["structural_missing_values"] += _count_true(
                pc.equal(status, "MISSING_SOURCE_UNAVAILABLE")
            )
            known = pc.is_in(
                status,
                value_set=pa.array(
                    sorted(contract.value_statuses), type=pa.string()
                ),
            )
            totals["unknown_value_statuses"] += _count_true(
                pc.invert(pc.fill_null(known, False))
            )
            semantic_missing = np.stack(
                [
                    _semantic_missing_mask(
                        feature_by_name[name],
                        raw[name].combine_chunks(),
                    )
                    for name in names
                ],
                axis=1,
            ).reshape(-1)
            present_numpy = pc.fill_null(present, False).to_numpy(
                zero_copy_only=False
            )
            totals["value_status_mismatches"] += int(
                np.count_nonzero(semantic_missing == present_numpy)
            )
            if family == "optionmetrics":
                raw_zero = np.stack(
                    [
                        np.equal(
                            raw[name]
                            .combine_chunks()
                            .to_numpy(zero_copy_only=False),
                            0,
                        )
                        & ~raw[name]
                        .combine_chunks()
                        .is_null()
                        .to_numpy(zero_copy_only=False)
                        for name in names
                    ],
                    axis=1,
                ).reshape(-1)
                not_covered = pc.equal(
                    status, "MISSING_NOT_COVERED"
                ).to_numpy(zero_copy_only=False)
                totals["option_noncoverage_zeros"] += int(
                    np.count_nonzero(raw_zero & not_covered)
                )
            observation = lineage[
                "max_observation_date"
            ].combine_chunks()
            availability = lineage[
                "max_public_availability_date"
            ].combine_chunks()
            stale = lineage["stale_after_date"].combine_chunks()
            totals["observation_date_after_signal"] += _count_true(
                pc.greater(observation, expected_signal)
            )
            totals[
                "public_availability_date_after_signal"
            ] += _count_true(pc.greater(availability, expected_signal))
            totals["stale_feature_uses"] += _count_true(
                pc.and_(
                    pc.fill_null(present, False),
                    pc.less(stale, expected_signal),
                )
            )
            missing_required_dates = pc.or_(
                observation.is_null(), availability.is_null()
            )
            totals["present_values_missing_dates"] += _count_true(
                pc.and_(
                    pc.fill_null(present, False),
                    missing_required_dates,
                )
            )
            any_date = pc.or_(
                pc.or_(
                    pc.invert(observation.is_null()),
                    pc.invert(availability.is_null()),
                ),
                pc.invert(stale.is_null()),
            )
            totals["missing_values_with_dates"] += _count_true(
                pc.and_(
                    pc.invert(pc.fill_null(present, False)),
                    any_date,
                )
            )
        if not raw_cursor.exhausted():
            raise Stage6VerificationError(
                f"Raw cursor did not exhaust for {family}"
            )
        if not lineage_cursor.exhausted():
            totals["lineage_key_differences"] += (
                lineage_file.metadata.num_rows - lineage_cursor.rows_read
            )
    return LineageAudit(**totals)


def _feature_dictionary(contract: FrozenContract) -> pa.Table:
    rows: list[dict[str, Any]] = []
    for ordinal, feature in enumerate(contract.features, start=1):
        parameters = {
            key: value
            for key, value in feature.items()
            if key not in _FEATURE_COMMON_FIELDS
        }
        rows.append(
            {
                "feature_ordinal": ordinal,
                "feature_name": str(feature["name"]),
                "family": str(feature["family"]),
                "kind": str(feature["kind"]),
                "source_set": str(feature["source_set"]),
                "unit": str(feature["unit"]),
                "formula_id": str(feature["formula_id"]),
                "parameters_json": json.dumps(
                    parameters,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return pa.Table.from_pylist(
        rows,
        schema=contract.published_schemas[
            "feature_dictionary.parquet"
        ],
    )


def _feature_membership(contract: FrozenContract) -> pa.Table:
    ordinals = {
        str(feature["name"]): ordinal
        for ordinal, feature in enumerate(contract.features, start=1)
    }
    rows: list[dict[str, Any]] = []
    for set_name in sorted(
        contract.feature_sets, key=lambda value: value.encode("utf-8")
    ):
        included = set(contract.feature_sets[set_name])
        for feature in contract.features:
            name = str(feature["name"])
            rows.append(
                {
                    "feature_set": set_name,
                    "feature_ordinal": ordinals[name],
                    "feature_name": name,
                    "included": name in included,
                }
            )
    return pa.Table.from_pylist(
        rows,
        schema=contract.published_schemas[
            "feature_set_membership.parquet"
        ],
    )


def _category_slug(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(
        character
        for character in decomposed
        if ord(character) < 128
        and not unicodedata.combining(character)
    ).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    return slug or "category"


def _encoded_category_name(feature: str, value: str) -> str:
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{feature}__category__{_category_slug(value)}__{suffix}"


def _unknown_category(feature: Mapping[str, Any]) -> str:
    explicit = _normalize_category(feature.get("unknown"))
    if explicit is not None:
        return explicit
    categories = feature.get("categories")
    if isinstance(categories, list):
        values = [
            normalized
            for item in categories
            if (normalized := _normalize_category(item)) is not None
            and normalized.startswith("UNKNOWN")
        ]
        if len(values) != 1:
            raise Stage6VerificationError(
                f"Fixed category unknown differs: {feature['name']}"
            )
        return values[0]
    return "UNKNOWN"


def _ff12_categories(contract: FrozenContract) -> tuple[str, ...]:
    sectors = contract.ff12.get("sectors")
    if not isinstance(sectors, list) or len(sectors) != 12:
        raise Stage6VerificationError("FF12 category map differs")
    values = tuple(str(sector["code"]) for sector in sectors)
    return (*values, str(contract.ff12["unknown_category"]))


def _category_vocabulary(
    contract: FrozenContract,
    dynamic_categories: Mapping[str, tuple[str, ...]],
) -> pa.Table:
    categorical = tuple(
        feature
        for feature in contract.features
        if feature["kind"]
        in {"categorical_firm", "categorical_multihot_firm"}
    )
    rows: list[dict[str, Any]] = []
    for feature in sorted(
        categorical,
        key=lambda item: str(item["name"]).encode("utf-8"),
    ):
        name = str(feature["name"])
        unknown = _unknown_category(feature)
        fixed = feature.get("categories")
        if isinstance(fixed, list):
            categories = tuple(
                (
                    str(_normalize_category(value)),
                    "FROZEN_REGISTRY_ENUMERATION",
                )
                for value in fixed
            )
        elif name == "ff12_sector_category":
            categories = tuple(
                (value, "FROZEN_FF12_MAPPING")
                for value in _ff12_categories(contract)
            )
        else:
            reserved = [(unknown, "FROZEN_RESERVED_CATEGORY")]
            if feature["kind"] == "categorical_multihot_firm":
                reserved.append(
                    ("NO_EVENT", "FROZEN_RESERVED_CATEGORY")
                )
            observed = [
                (value, "LONG_ELIGIBLE_DEVELOPMENT")
                for value in dynamic_categories.get(name, ())
                if value not in {item[0] for item in reserved}
            ]
            categories = tuple((*reserved, *observed))
        if len(categories) != len({value for value, _ in categories}):
            raise Stage6VerificationError(
                f"Category vocabulary duplicates: {name}"
            )
        for ordinal, (value, population) in enumerate(
            categories, start=1
        ):
            rows.append(
                {
                    "feature_name": name,
                    "category_ordinal": ordinal,
                    "category_value": value,
                    "encoded_column_name": _encoded_category_name(
                        name, value
                    ),
                    "is_unknown": value == unknown,
                    "is_no_event": value == "NO_EVENT",
                    "population": population,
                }
            )
    return pa.Table.from_pylist(
        rows,
        schema=contract.published_schemas[
            "category_vocabulary.parquet"
        ],
    )


def verify_processed_registries(
    contract: FrozenContract,
    *,
    processed: AuthenticatedOutput,
    raw_audit: RawAudit,
) -> dict[str, int]:
    expected = {
        "feature_dictionary.parquet": _feature_dictionary(contract),
        "feature_set_membership.parquet": _feature_membership(contract),
        "category_vocabulary.parquet": _category_vocabulary(
            contract, raw_audit.dynamic_categories
        ),
    }
    differences: dict[str, int] = {}
    for name, expected_table in expected.items():
        observed = pq.read_table(processed.files[name])
        differences[name] = int(not observed.equals(expected_table))
    return {
        "processed_registry_differences": sum(differences.values()),
        "category_vocabulary_rows": expected[
            "category_vocabulary.parquet"
        ].num_rows,
    }


def _sort_token(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8")
    return value


def _verify_sorted_unique(
    table: pa.Table,
    *,
    primary_key: Sequence[str],
    sort_key: Sequence[str],
) -> tuple[int, int]:
    rows = table.select(
        list(dict.fromkeys((*primary_key, *sort_key)))
    ).to_pylist()
    duplicate_count = 0
    order_count = 0
    prior_primary: tuple[Any, ...] | None = None
    prior_sort: tuple[Any, ...] | None = None
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        primary = tuple(row[name] for name in primary_key)
        sort = tuple(_sort_token(row[name]) for name in sort_key)
        if primary in seen:
            duplicate_count += 1
        seen.add(primary)
        if prior_sort is not None and sort < prior_sort:
            order_count += 1
        prior_primary = primary
        prior_sort = sort
    del prior_primary
    return duplicate_count, order_count


def _invalid_outlier_values(values: Sequence[float | None]) -> bool:
    observed = [value for value in values if value is not None]
    return bool(
        any(not math.isfinite(value) for value in observed)
        or (observed and observed != sorted(observed))
    )


def verify_quality_payloads(
    contract: FrozenContract,
    *,
    quality: AuthenticatedOutput,
    raw_audit: RawAudit,
    lineage_audit: LineageAudit,
    build_id: str,
) -> dict[str, Any]:
    table_integrity = 0
    quality_tables: dict[str, pa.Table] = {}
    for name, schema in contract.published_schemas.items():
        if name not in quality.files:
            continue
        table = pq.read_table(quality.files[name])
        if not table.schema.equals(schema, check_metadata=True):
            table_integrity += 1
            continue
        specification = contract.rules["published_table_schemas"][name]
        duplicate, order = _verify_sorted_unique(
            table,
            primary_key=specification["primary_key"],
            sort_key=specification["sort"],
        )
        table_integrity += duplicate + order
        quality_tables[name] = table

    feature_family = {
        str(feature["name"]): str(feature["family"])
        for feature in contract.features
    }
    years = raw_audit.signal_year_counts
    missingness = quality_tables.get("feature_missingness.parquet")
    if missingness is not None:
        for row in missingness.to_pylist():
            table_integrity += int(
                row["feature_name"] not in feature_family
                or row["feature_family"]
                != feature_family.get(row["feature_name"])
                or row["value_status"] not in contract.value_statuses
                or row["row_count"] < 0
                or row["group_total"] <= 0
                or not math.isclose(
                    row["row_fraction"],
                    row["row_count"] / row["group_total"],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
    coverage = quality_tables.get("feature_coverage.parquet")
    if coverage is not None:
        expected_coverage_keys = {
            (name, year)
            for name in feature_family
            for year in years
        }
        observed_coverage_keys: set[tuple[str, int]] = set()
        for row in coverage.to_pylist():
            key = (row["feature_name"], int(row["signal_year"]))
            observed_coverage_keys.add(key)
            expected_total = years.get(int(row["signal_year"]))
            table_integrity += int(
                row["feature_family"]
                != feature_family.get(row["feature_name"])
                or expected_total is None
                or row["total_rows"] != expected_total
                or row["present_rows"] + row["missing_rows"]
                != row["total_rows"]
                or not math.isclose(
                    row["coverage_rate"],
                    row["present_rows"] / row["total_rows"],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
        table_integrity += len(
            expected_coverage_keys.symmetric_difference(
                observed_coverage_keys
            )
        )
    outliers = quality_tables.get("feature_outliers.parquet")
    if outliers is not None:
        numeric_names = {
            str(feature["name"])
            for feature in contract.features
            if feature["kind"]
            in {"continuous_firm", "binary_firm", "aggregate_state"}
        }
        expected_outlier_keys = {
            (name, year) for name in numeric_names for year in years
        }
        observed_outlier_keys: set[tuple[str, int]] = set()
        for row in outliers.to_pylist():
            key = (row["feature_name"], int(row["signal_year"]))
            observed_outlier_keys.add(key)
            expected_total = years.get(int(row["signal_year"]))
            values = [
                row[name]
                for name in ("minimum", "p01", "median", "p99", "maximum")
            ]
            table_integrity += int(
                row["feature_name"] not in numeric_names
                or expected_total is None
                or row["finite_count"]
                + row["missing_count"]
                + row["nonfinite_count"]
                != expected_total
                or row["nonfinite_count"] != 0
                or _invalid_outlier_values(values)
            )
        table_integrity += len(
            expected_outlier_keys.symmetric_difference(
                observed_outlier_keys
            )
        )
    source_coverage = quality_tables.get(
        "feature_source_coverage.parquet"
    )
    if source_coverage is not None:
        for row in source_coverage.to_pylist():
            table_integrity += int(
                row["row_count"] < 0
                or row["group_total"] <= 0
                or not math.isclose(
                    row["coverage_fraction"],
                    row["row_count"] / row["group_total"],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )

    summary_path = quality.files["quality_summary.json"]
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    exact_keys = set(
        contract.rules["publication_payloads"][
            "quality_summary_exact_keys"
        ]
    )
    if (
        not isinstance(summary, dict)
        or set(summary) != exact_keys
        or summary_bytes != _canonical_json_bytes(summary)
    ):
        raise Stage6VerificationError(
            "Quality-summary encoding or keys differ"
        )
    expected_scope = contract.rules["publication_payloads"][
        "quality_summary_value_contract"
    ]["scope"]["exact_values"]
    expected_inputs = contract.rules["publication_payloads"][
        "quality_summary_value_contract"
    ]["input_builds"]["exact_values"]
    hard_expected = {
        key: int(value)
        for key, value in contract.policy["quality_gates"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    reconstructed_hard = {
        "duplicate_raw_keys": raw_audit.raw_duplicate_or_order_errors,
        "raw_rows": raw_audit.raw_rows,
        "raw_feature_columns": len(contract.features),
        "duplicate_lineage_keys": lineage_audit.duplicate_lineage_keys,
        "missing_lineage_keys": lineage_audit.lineage_key_differences,
        "lineage_rows": lineage_audit.lineage_rows,
        "observation_date_after_signal":
            lineage_audit.observation_date_after_signal,
        "public_availability_date_after_signal":
            lineage_audit.public_availability_date_after_signal,
        "stale_feature_uses": lineage_audit.stale_feature_uses,
        "future_predictor_rows": int(
            raw_audit.first_signal_date
            != dt.date.fromisoformat(
                str(contract.policy["scope"]["first_signal_date"])
            )
            or raw_audit.last_signal_date
            != dt.date.fromisoformat(
                str(contract.policy["scope"]["last_signal_date"])
            )
        ),
        "nonfinite_published_numeric_values":
            raw_audit.nonfinite_numeric_values,
        "unknown_feature_names": lineage_audit.unknown_feature_names,
        "feature_order_mismatch": lineage_audit.lineage_key_differences,
        "optionmetrics_total_rows_used": 0,
        "optionmetrics_noncoverage_encoded_as_zero":
            lineage_audit.option_noncoverage_zeros,
        "raw_pastor_stambaugh_innovation_used": 0,
    }
    expected_counts = {
        "raw_rows": raw_audit.raw_rows,
        "raw_feature_columns": len(contract.features),
        "feature_families": len(contract.family_order),
        "lineage_rows": lineage_audit.lineage_rows,
        "present_values": lineage_audit.present_values,
        "missing_values": lineage_audit.missing_values,
        "structural_missing_values":
            lineage_audit.structural_missing_values,
    }
    flags = {
        key: value
        for key, value in contract.policy["quality_gates"].items()
        if isinstance(value, bool)
    }
    summary_differences = sum(
        (
            summary.get("schema_version") != 1,
            summary.get("contract_id") != contract.policy["contract_id"],
            summary.get("policy_id") != contract.policy["policy_id"],
            summary.get("build_id") != build_id,
            summary.get("status")
            != "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION",
            summary.get("scope") != expected_scope,
            summary.get("input_builds") != expected_inputs,
            summary.get("hard_gates") != reconstructed_hard,
            summary.get("quality_policy_flags") != flags,
            summary.get("structural_missingness")
            != raw_audit.structural_missingness,
        )
    )
    counts = summary.get("counts")
    expected_count_keys = {
        "raw_rows",
        "raw_feature_columns",
        "feature_families",
        "lineage_rows",
        "present_values",
        "missing_values",
        "structural_missing_values",
    }
    if (
        not isinstance(counts, Mapping)
        or set(counts) != expected_count_keys
        or any(
            not isinstance(value, int) or value < 0
            for value in counts.values()
        )
    ):
        summary_differences += 1
    else:
        for key in (
            "raw_rows",
            "raw_feature_columns",
            "feature_families",
            "lineage_rows",
            "present_values",
            "missing_values",
            "structural_missing_values",
        ):
            summary_differences += int(
                counts.get(key) != expected_counts[key]
            )
    summary_differences += int(
        reconstructed_hard != hard_expected
    )

    timing = quality_tables.get("feature_timing_checks.parquet")
    timing_failures = 1
    if timing is not None:
        timing_rows = timing.to_pylist()
        timing_failures = sum(
            not row["passed"]
            or row["observed_count"] != row["expected_count"]
            for row in timing_rows
        )
        by_name = {row["check_name"]: row for row in timing_rows}
        for key, value in reconstructed_hard.items():
            timing_failures += int(
                key not in by_name
                or by_name[key]["observed_count"] != value
                or by_name[key]["expected_count"] != hard_expected[key]
            )
    return {
        "quality_table_integrity_differences": int(table_integrity),
        "quality_summary_differences": int(summary_differences),
        "quality_timing_failures": int(timing_failures),
        "quality_summary": summary,
        "reconstructed_hard_gates": reconstructed_hard,
    }


def _difference(
    differences: list[Difference],
    code: str,
    expected: Any,
    observed: Any,
    details: str,
) -> None:
    if observed != expected:
        differences.append(
            Difference(code, expected, observed, details)
        )


def _safe_artifact_root(path: Path) -> Path:
    root = Path(os.path.abspath(path.expanduser()))
    root.mkdir(parents=True, exist_ok=True)
    observed = os.lstat(root)
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(
        observed.st_mode
    ):
        raise Stage6VerificationError("Verification artifact root is unsafe")
    return root


def _publish_immutable_bundle(
    *,
    artifact_root: Path,
    destination: Path,
    payloads: Mapping[str, bytes],
) -> None:
    try:
        destination.relative_to(artifact_root)
    except ValueError as exc:
        raise Stage6VerificationError(
            "Verification bundle escaped artifact root"
        ) from exc
    if any(
        _SAFE_NAME.fullmatch(name) is None or Path(name).name != name
        for name in payloads
    ):
        raise Stage6VerificationError(
            "Verification bundle filename is unsafe"
        )
    try:
        with SecureStagingDirectory.create(
            trusted_root=artifact_root,
            destination=destination,
            expected_files=tuple(payloads),
            label="Stage 6 independent verification",
        ) as stage:
            for name, payload in payloads.items():
                with stage.open_exclusive_binary(name) as stream:
                    stream.write(payload)
            snapshot = stage.snapshot(
                expected_hashes={
                    name: _sha256_bytes(payload)
                    for name, payload in payloads.items()
                }
            )
            stage.publish(expected_hashes=snapshot.hashes)
    except SecurePublicationError as exc:
        raise Stage6VerificationError(
            "Cannot publish independent verification bundle"
        ) from exc


def _replace_pointer_last(
    path: Path,
    payload: bytes,
    *,
    artifact_root: Path,
) -> None:
    try:
        replace_pointer_last(
            path,
            payload,
            trusted_root=artifact_root,
            pre_replace_gate=lambda: None,
            post_replace_gate=lambda: None,
            label="Stage 6 latest verified pointer",
        )
    except SecurePublicationError as exc:
        raise Stage6VerificationError(
            "Cannot replace Stage 6 verified pointer"
        ) from exc


def _publish_verification(
    contract: FrozenContract,
    *,
    artifact_root: Path,
    build_id: str,
    primary_manifest_sha256: str,
    checks: Mapping[str, Any],
    output_hashes: Mapping[str, Mapping[str, str]],
) -> tuple[str, Path, Path]:
    differences_bytes = _canonical_json_bytes([])
    differences_sha = _sha256_bytes(differences_bytes)
    checks_payload = {
        "check_count": len(checks),
        "pass_count": len(checks),
        "fail_count": 0,
        "results_sha256": _sha256_bytes(_canonical_json_bytes(checks)),
    }
    output_hashes_sha = _sha256_bytes(
        _canonical_json_bytes(output_hashes)
    )
    verifier_sha = _sha256_file(Path(__file__).resolve())
    identity = {
        "build_id": build_id,
        "primary_manifest_sha256": primary_manifest_sha256,
        "verifier_code_sha256": verifier_sha,
        "checks": checks,
        "differences_sha256": differences_sha,
        "output_hashes_sha256": output_hashes_sha,
    }
    verification_id = (
        "verify-" + _sha256_bytes(_canonical_json_bytes(identity))[:16]
    )
    check = {
        "schema_version": 1,
        "contract_id": contract.policy["contract_id"],
        "policy_id": contract.policy["policy_id"],
        "build_id": build_id,
        "verification_id": verification_id,
        "status": "VERIFIED_PASSED",
        "primary_manifest_sha256": primary_manifest_sha256,
        "checks": checks_payload,
        "differences_sha256": differences_sha,
        "output_hashes_sha256": output_hashes_sha,
        "authorizations": _VERIFIED_AUTHORIZATIONS,
    }
    required_check_keys = set(
        contract.rules["publication_payloads"][
            "verification_manifest_required_keys"
        ]
    )
    if set(check) != required_check_keys:
        raise Stage6VerificationError(
            "Independent-check keys differ from frozen contract"
        )
    check_bytes = _canonical_json_bytes(check)
    check_sha = _sha256_bytes(check_bytes)
    pointer = {
        "schema_version": 1,
        "contract_id": contract.policy["contract_id"],
        "policy_id": contract.policy["policy_id"],
        "build_id": build_id,
        "verification_id": verification_id,
        "status": "VERIFIED_PASSED",
        "primary_manifest_sha256": primary_manifest_sha256,
        "independent_check_sha256": check_sha,
        "differences_sha256": differences_sha,
        "authorizations": _VERIFIED_AUTHORIZATIONS,
    }
    required_pointer_keys = set(
        contract.rules["publication_payloads"][
            "verified_pointer_required_keys"
        ]
    )
    if set(pointer) != required_pointer_keys:
        raise Stage6VerificationError(
            "Verified-pointer keys differ from frozen contract"
        )
    namespace = str(
        contract.policy["publication"]["verification_namespace"]
    )
    destination = (
        artifact_root / namespace / build_id / verification_id
    )
    _publish_immutable_bundle(
        artifact_root=artifact_root,
        destination=destination,
        payloads={
            "differences.json": differences_bytes,
            "independent_check.json": check_bytes,
        },
    )
    pointer_path = artifact_root / Path(
        str(contract.policy["publication"]["latest_verified_pointer"])
    ).name
    _replace_pointer_last(
        pointer_path,
        _canonical_json_bytes(pointer),
        artifact_root=artifact_root,
    )
    return verification_id, destination, pointer_path


def verify_stage6_feature_store(
    *,
    primary_manifest_path: Path,
    primary_pointer_path: Path | None = None,
    database_root: Path | None = None,
    artifact_root: Path | None = None,
    policy_path: Path = DEFAULT_POLICY,
    publish: bool = True,
    raw_batch_rows: int = 16_384,
    lineage_batch_rows: int = 2_048,
    required_code_paths: frozenset[str] | None = None,
) -> VerificationResult:
    """Run the complete independent Stage 6 verification and publish on zero diff."""

    contract = reconstruct_frozen_contract(policy_path)
    database = (
        Path(str(contract.policy["database"]["root"])).expanduser()
        if database_root is None
        else database_root
    )
    artifacts = _safe_artifact_root(
        (
            contract.project_root
            / str(contract.policy["publication"]["artifact_root"])
        )
        if artifact_root is None
        else artifact_root
    )
    primary_pointer = (
        contract.project_root
        / str(contract.policy["publication"]["latest_primary_pointer"])
        if primary_pointer_path is None
        else primary_pointer_path
    )
    manifest, manifest_hash = _read_primary(
        contract,
        primary_manifest_path=primary_manifest_path,
        primary_pointer_path=primary_pointer,
        artifact_root=artifacts,
    )
    build_id = str(manifest["build_id"])
    input_receipts = reconstruct_input_receipts(
        contract, database_root=database
    )
    identity = verify_build_identity(
        contract,
        manifest,
        input_receipts,
        required_code_paths=required_code_paths,
    )
    outputs = {
        kind: _authenticate_output(
            contract,
            manifest,
            database_root=database,
            kind=kind,
        )
        for kind in ("interim", "processed", "quality")
    }
    raw_path = outputs["interim"].files["feature_store_raw.parquet"]
    spine_path = next(
        item.path
        for item in input_receipts
        if item.name == "crsp_signal_date_universe.parquet"
    )
    raw_audit = audit_raw_store(
        contract,
        raw_path=raw_path,
        spine_path=spine_path,
        batch_rows=raw_batch_rows,
    )
    lineage_paths = {
        family: outputs["interim"].files[
            f"feature_lineage_{family}.parquet"
        ]
        for family in contract.family_order
    }
    lineage_audit = audit_lineage(
        contract,
        raw_path=raw_path,
        lineage_paths=lineage_paths,
        batch_rows=lineage_batch_rows,
    )
    processed_audit = verify_processed_registries(
        contract,
        processed=outputs["processed"],
        raw_audit=raw_audit,
    )
    quality_audit = verify_quality_payloads(
        contract,
        quality=outputs["quality"],
        raw_audit=raw_audit,
        lineage_audit=lineage_audit,
        build_id=build_id,
    )

    checks: dict[str, Any] = {
        "raw_rows": raw_audit.raw_rows,
        "raw_key_differences": raw_audit.raw_key_differences,
        "raw_duplicate_or_order_errors":
            raw_audit.raw_duplicate_or_order_errors,
        "raw_audit_value_differences":
            raw_audit.raw_audit_value_differences,
        "nonfinite_numeric_values":
            raw_audit.nonfinite_numeric_values,
        "invalid_binary_values": raw_audit.invalid_binary_values,
        "invalid_categorical_values":
            raw_audit.invalid_categorical_values,
        "signal_dates": raw_audit.signal_dates,
        "lineage_rows": lineage_audit.lineage_rows,
        "lineage_key_differences":
            lineage_audit.lineage_key_differences,
        "duplicate_lineage_keys":
            lineage_audit.duplicate_lineage_keys,
        "unknown_feature_names":
            lineage_audit.unknown_feature_names,
        "unknown_value_statuses":
            lineage_audit.unknown_value_statuses,
        "value_status_mismatches":
            lineage_audit.value_status_mismatches,
        "observation_date_after_signal":
            lineage_audit.observation_date_after_signal,
        "public_availability_date_after_signal":
            lineage_audit.public_availability_date_after_signal,
        "stale_feature_uses": lineage_audit.stale_feature_uses,
        "present_values_missing_dates":
            lineage_audit.present_values_missing_dates,
        "missing_values_with_dates":
            lineage_audit.missing_values_with_dates,
        **{
            key: value
            for key, value in processed_audit.items()
            if key != "category_vocabulary_rows"
        },
        **{
            key: value
            for key, value in quality_audit.items()
            if isinstance(value, int)
        },
        "build_identity_matches": identity["build_id"] == build_id,
    }
    differences: list[Difference] = []
    scope = contract.policy["scope"]
    expected_zero_checks = (
        "raw_key_differences",
        "raw_duplicate_or_order_errors",
        "raw_audit_value_differences",
        "nonfinite_numeric_values",
        "invalid_binary_values",
        "invalid_categorical_values",
        "lineage_key_differences",
        "duplicate_lineage_keys",
        "unknown_feature_names",
        "unknown_value_statuses",
        "value_status_mismatches",
        "observation_date_after_signal",
        "public_availability_date_after_signal",
        "stale_feature_uses",
        "present_values_missing_dates",
        "missing_values_with_dates",
        "processed_registry_differences",
        "quality_table_integrity_differences",
        "quality_summary_differences",
        "quality_timing_failures",
    )
    for name in expected_zero_checks:
        _difference(
            differences,
            name,
            0,
            checks[name],
            f"Independent Stage 6 check {name} must equal zero.",
        )
    for name, expected, observed in (
        (
            "raw_rows",
            int(scope["required_spine_rows"]),
            raw_audit.raw_rows,
        ),
        (
            "signal_dates",
            int(scope["required_signal_dates"]),
            raw_audit.signal_dates,
        ),
        (
            "lineage_rows",
            int(scope["required_lineage_rows"]),
            lineage_audit.lineage_rows,
        ),
        (
            "first_signal_date",
            str(scope["first_signal_date"]),
            None
            if raw_audit.first_signal_date is None
            else raw_audit.first_signal_date.isoformat(),
        ),
        (
            "last_signal_date",
            str(scope["last_signal_date"]),
            None
            if raw_audit.last_signal_date is None
            else raw_audit.last_signal_date.isoformat(),
        ),
    ):
        _difference(
            differences,
            name,
            expected,
            observed,
            f"Frozen Stage 6 scope check {name} differs.",
        )
    expected_summary = {
        "raw_rows": int(scope["required_spine_rows"]),
        "economic_features": len(contract.features),
        "feature_families": len(contract.family_order),
        "lineage_rows": int(scope["required_lineage_rows"]),
        "feature_set_included_counts": {
            name: len(values)
            for name, values in sorted(contract.feature_sets.items())
        },
        "category_vocabulary_rows":
            processed_audit["category_vocabulary_rows"],
        "hard_gate_failures": 0,
    }
    _difference(
        differences,
        "primary_manifest_summary",
        expected_summary,
        manifest["summary"],
        "Primary summary must match independently reconstructed counts.",
    )
    _difference(
        differences,
        "build_identity",
        True,
        checks["build_identity_matches"],
        "Content-addressed build identity must be deterministic.",
    )
    if differences:
        return VerificationResult(
            passed=False,
            build_id=build_id,
            verification_id=None,
            checks=checks,
            differences=tuple(differences),
            verification_path=None,
            verified_pointer_path=None,
        )

    output_hashes = {
        kind: dict(sorted(output.hashes.items()))
        for kind, output in outputs.items()
    }
    if publish:
        if _sha256_file(primary_manifest_path) != manifest_hash:
            raise Stage6VerificationError(
                "Primary manifest changed during verification"
            )
        for receipt in input_receipts:
            if _sha256_file(receipt.path) != receipt.sha256:
                raise Stage6VerificationError(
                    f"Input changed during verification: {receipt.name}"
                )
        for output in outputs.values():
            for name, path in output.files.items():
                if _sha256_file(path) != output.hashes[name]:
                    raise Stage6VerificationError(
                        f"Output changed during verification: {name}"
                    )
        verification_id, verification_path, verified_pointer = (
            _publish_verification(
                contract,
                artifact_root=artifacts,
                build_id=build_id,
                primary_manifest_sha256=manifest_hash,
                checks=checks,
                output_hashes=output_hashes,
            )
        )
    else:
        verification_id = verification_path = verified_pointer = None
    return VerificationResult(
        passed=True,
        build_id=build_id,
        verification_id=verification_id,
        checks=checks,
        differences=(),
        verification_path=verification_path,
        verified_pointer_path=verified_pointer,
    )
