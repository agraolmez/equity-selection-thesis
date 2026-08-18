"""Repository-native raw-to-normalized builder for the Stage 13 final run.

The public entry point in this module is deliberately zero argument.  It
authenticates the verified Stage 12 freeze, the separate Stage 13 unblind
approval, and the transitive Phase 1 raw snapshot before an empirical reader
can be reached.  Callers cannot provide paths, dates, schemas, tables, or
reader callbacks.

The expensive empirical replay and the inexpensive normalization compiler are
kept separate.  The replay emits the native Stage 6/8/11 formula products in a
small sealed carrier; the compiler adds only the final holding-month axis and
performs the exact schema, chronology, key, and period checks used by the
publication layer.  This boundary is useful for two reasons:

* the unchanged Stage 6, Stage 8, Stage 7/11 implementations remain the sole
  economic formula authorities; and
* the full compiler can be certified with synthetic Arrow inputs without
  opening a 2020+ payload.

The production replay is imported lazily from ``final_source_replay``.  That
module owns the large raw scans and is reachable only after every authority in
this module has passed.  Keeping that import behind the gate also makes the
authorization-before-payload ordering directly testable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pyarrow as pa

from portfolio_bot.final_test.authorization import (
    Stage12VerifiedBinding,
    load_authenticated_stage12_binding,
    require_stage13_unblind_authorization,
)
from portfolio_bot.final_test.final_input_construction import (
    PROJECT_ROOT,
    _SOURCE_MONTHS,
    _SOURCE_NAMES,
    _load_fixed_approval,
    _source_schemas,
    _validate_normalized_source_tables,
)


BUILDER_ID = "stage13-final-raw-formula-builder-v1"
RAW_AUTHORITY_ID = "phase01-frozen-raw-authority-for-stage13-v1"
FORMULA_REPLAY_ID = "stage13-native-stage6-stage8-stage11-replay-v1"

RAW_FROZEN_POINTER = Path("artifacts/phase_01_raw_manifest/LATEST_FROZEN.json")
RAW_VERIFIED_POINTER = Path("artifacts/phase_01_raw_manifest/LATEST_VERIFIED.json")
RAW_AUTHORITY_ROOT = Path("artifacts/phase_01_raw_manifest")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TEST_REPLAY_TOKEN = object()
_REPLAY_SEAL = object()

# The native replay may change table-level scope only.  These are the exact
# economic implementations whose formulas it must call unchanged.
_FORMULA_CODE_MEMBERS = (
    Path("src/portfolio_bot/data/phase_02_auxiliary_core.py"),
    Path("src/portfolio_bot/data/phase_02_auxiliary_compustat_factors.py"),
    Path("src/portfolio_bot/data/phase_02_crsp_typed_backbone.py"),
    Path("src/portfolio_bot/data/phase_02_crsp_typed_backbone_core.py"),
    Path("src/portfolio_bot/data/phase_02_crsp_typed_backbone_discovery.py"),
    Path("src/portfolio_bot/data/phase_02_crsp_typed_backbone_transforms.py"),
    Path("src/portfolio_bot/data/phase_02_crsp_return_reconciliation.py"),
    Path("src/portfolio_bot/features/feature_store_inputs.py"),
    Path("src/portfolio_bot/features/feature_store_materialization.py"),
    Path("src/portfolio_bot/features/feature_math.py"),
    Path("src/portfolio_bot/features/crsp_features.py"),
    Path("src/portfolio_bot/features/fundamental_features.py"),
    Path("src/portfolio_bot/features/ownership_event_features.py"),
    Path("src/portfolio_bot/features/optionmetrics_features.py"),
    Path("src/portfolio_bot/features/aggregate_state_features.py"),
    Path("src/portfolio_bot/features/materialize_crsp.py"),
    Path("src/portfolio_bot/features/materialize_crsp_streaming.py"),
    Path("src/portfolio_bot/features/materialize_crsp_routing.py"),
    Path("src/portfolio_bot/features/materialize_fundamentals.py"),
    Path("src/portfolio_bot/features/materialize_ownership_events.py"),
    Path("src/portfolio_bot/features/materialize_optionmetrics.py"),
    Path("src/portfolio_bot/features/materialize_aggregate_state.py"),
    Path("src/portfolio_bot/features/label_math.py"),
    Path("src/portfolio_bot/features/label_store_materialization.py"),
    Path("src/portfolio_bot/risk_cost/risk_math.py"),
    Path("src/portfolio_bot/risk_cost/cost_math.py"),
    Path("src/portfolio_bot/risk_cost/materialization.py"),
    Path("src/portfolio_bot/validation/validation_source_extraction.py"),
    Path("src/portfolio_bot/validation/outcome_materialization.py"),
    Path("src/portfolio_bot/validation/validation_market_extension.py"),
    Path("src/portfolio_bot/validation/validation_return_adapter.py"),
    Path("src/portfolio_bot/validation/validation_universe_auxiliary.py"),
    Path("src/portfolio_bot/validation/validation_feature_input_adapter.py"),
    Path("src/portfolio_bot/validation/validation_fundamental_adapter.py"),
    Path("src/portfolio_bot/validation/validation_empirical_feature_stream.py"),
    Path("src/portfolio_bot/validation/validation_data_candidate_builder.py"),
    Path("src/portfolio_bot/validation/corporate_action_adapter.py"),
    Path("src/portfolio_bot/final_test/terminal_event_overlay.py"),
    Path("src/portfolio_bot/final_test/final_source_replay.py"),
    Path("src/portfolio_bot/final_test/final_source_builder.py"),
)

_FORMULA_CONFIG_MEMBERS = (
    Path("configs/phase_02_canonicalization/auxiliary_source_schema_v1.json"),
    Path("configs/phase_02_canonicalization/crsp_typed_backbone_policy_v1.toml"),
    Path("configs/phase_02_canonicalization/crsp_return_reconciliation_policy_v1.toml"),
    Path("configs/phase_03_features/feature_registry_v1.json"),
    Path("configs/phase_03_features/feature_build_rules_v1.json"),
    Path("configs/phase_03_features/label_build_rules_v1.json"),
    Path("configs/phase_04_models/risk_cost_policy_v1.toml"),
    Path("configs/phase_04_models/risk_cost_build_rules_v1.json"),
    Path("configs/phase_05_validation/stage_11_corporate_action_mapping_v1.json"),
    Path("configs/phase_06_final_test/stage_12_final_system_v1.json"),
    Path("configs/phase_06_final_test/stage_13_input_registry_v1.json"),
    Path(
        "configs/phase_06_final_test/"
        "stage_13_public_interest_last_tradable_close_liquidation_v1.json"
    ),
)

# Predictors/risk/execution products are published on the decision side;
# realized accounting, factors, labels, and outcomes are published afterward.
# This table-role sequence is not a raw-read barrier: later decisions correctly
# consume prior final-period returns under their own signal-date cutoff.  The
# runtime enforces the actual per-month decision-before-accounting chronology.
DECISION_SOURCE_NAMES = (
    "final_calendar",
    "final_predictors",
    "final_risk_cost",
    "final_factor_covariance",
    "final_risk_cost_month_summary",
    "final_market_inputs",
)
OUTCOME_SOURCE_NAMES = (
    "final_market_accounting",
    "final_factors",
    "final_benchmarks",
    "final_completed_labels",
    "final_outcomes",
    "partial_descriptive_outcome",
)
EXPECTED_SOURCE_PUBLICATION_SEQUENCE = DECISION_SOURCE_NAMES + OUTCOME_SOURCE_NAMES


class FinalSourceBuilderError(RuntimeError):
    """Raised when a final raw authority or native formula replay differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalSourceBuilderError(message)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FinalSourceBuilderError(
            "final source identity is not finite canonical JSON"
        ) from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_file_bytes(
    path: Path,
    *,
    trusted_root: Path,
    label: str,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> bytes:
    root = Path(os.path.abspath(trusted_root.expanduser()))
    selected = Path(os.path.abspath(path.expanduser()))
    try:
        relative = selected.relative_to(root)
    except ValueError as exc:
        raise FinalSourceBuilderError(f"{label} escaped its root") from exc
    _require(bool(relative.parts), f"{label} cannot equal its root")
    current = root
    for index, component in enumerate(relative.parts):
        current /= component
        try:
            metadata = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise FinalSourceBuilderError(f"{label} is absent") from exc
        final = index == len(relative.parts) - 1
        _require(not stat.S_ISLNK(metadata.st_mode), f"{label} is symlinked")
        _require(
            stat.S_ISREG(metadata.st_mode) if final else stat.S_ISDIR(metadata.st_mode),
            f"{label} path type differs",
        )
    descriptor = os.open(selected, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and before.st_size <= maximum_bytes,
            f"{label} identity is unsafe",
        )
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    _require(identity(before) == identity(after), f"{label} changed while read")
    return b"".join(chunks)


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalSourceBuilderError(f"{label} is not valid JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _safe_relative_json(value: object, *, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} is missing")
    relative = PurePosixPath(str(value))
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and "." not in relative.parts
        and relative.suffix == ".json",
        f"{label} path is unsafe",
    )
    return Path(*relative.parts)


def _hash_members(
    project_root: Path,
    members: Sequence[Path],
    *,
    label: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in members:
        payload = _stable_file_bytes(
            project_root / relative,
            trusted_root=project_root,
            label=f"{label} {relative.as_posix()}",
        )
        result[relative.as_posix()] = _sha256_bytes(payload)
    return result


@dataclass(frozen=True, slots=True)
class AuthenticatedFinalRawAuthority:
    """Metadata-only Phase 1 raw authority; no empirical row is opened."""

    identity: Mapping[str, object]
    raw_files: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class AuthenticatedFinalFormulaReplay:
    """Sealed native formula products before final month-axis normalization."""

    tables: Mapping[str, pa.Table]
    raw_source_identity: Mapping[str, object]
    source_publication_sequence: tuple[str, ...]
    formula_identity: Mapping[str, object]
    _seal: object

    def __post_init__(self) -> None:
        _require(self._seal is _REPLAY_SEAL, "final formula replay is unissued")
        _require(
            tuple(self.tables) == _SOURCE_NAMES,
            "final formula replay table order differs",
        )


@dataclass(frozen=True, slots=True)
class FinalNormalizedSourceBuild:
    """The twelve normalized Arrow tables and their transitive identities."""

    tables: Mapping[str, pa.Table]
    raw_source_identity: Mapping[str, object]
    builder_identity: Mapping[str, object]


def authenticate_final_raw_authority(
    project_root: Path = PROJECT_ROOT,
) -> AuthenticatedFinalRawAuthority:
    """Authenticate the frozen/verified Phase 1 graph without raw data reads."""

    _require(
        isinstance(project_root, Path) and project_root.is_absolute(),
        "project_root must be an absolute Path",
    )
    root = project_root.absolute()
    namespace = root / RAW_AUTHORITY_ROOT
    frozen_payload = _stable_file_bytes(
        root / RAW_FROZEN_POINTER,
        trusted_root=root,
        label="Phase 1 frozen pointer",
    )
    verified_payload = _stable_file_bytes(
        root / RAW_VERIFIED_POINTER,
        trusted_root=root,
        label="Phase 1 verified pointer",
    )
    frozen = _json_object(frozen_payload, label="Phase 1 frozen pointer")
    verified = _json_object(verified_payload, label="Phase 1 verified pointer")
    manifest_relative = _safe_relative_json(
        frozen.get("path"), label="Phase 1 transitive manifest"
    )
    manifest_payload = _stable_file_bytes(
        namespace / manifest_relative,
        trusted_root=namespace,
        label="Phase 1 transitive manifest",
    )
    manifest_sha = _sha256_bytes(manifest_payload)
    manifest = _json_object(manifest_payload, label="Phase 1 transitive manifest")

    _require(
        frozen.get("status") == "phase_01_frozen"
        and frozen.get("sha256") == manifest_sha
        and frozen.get("phase_01_build_id") == verified.get("phase_01_build_id")
        and frozen.get("raw_snapshot_id") == verified.get("raw_snapshot_id")
        and frozen.get("verification_id") == verified.get("verification_id")
        and verified.get("verification_status") == "passed",
        "Phase 1 frozen/verified identities differ",
    )
    identity = manifest.get("identity")
    _require(isinstance(identity, Mapping), "Phase 1 manifest identity is absent")
    sealed = identity.get("sealed_periods")
    _require(
        identity.get("phase") == "01_raw_snapshot"
        and identity.get("phase_01_build_id") == frozen.get("phase_01_build_id")
        and identity.get("raw_snapshot_id") == frozen.get("raw_snapshot_id")
        and isinstance(sealed, Mapping)
        and sealed.get("final_test_2020_2025") == "untouched_and_sealed",
        "Phase 1 final-period seal differs",
    )

    evidence_groups = identity.get("evidence_groups")
    _require(isinstance(evidence_groups, Mapping), "Phase 1 evidence graph is absent")
    primary_entries = evidence_groups.get("primary_build")
    _require(isinstance(primary_entries, list), "Phase 1 primary evidence is absent")
    primary_manifest_candidates = [
        item
        for item in primary_entries
        if isinstance(item, Mapping)
        and str(item.get("path", "")).endswith("/manifest.json")
    ]
    _require(
        len(primary_manifest_candidates) == 1,
        "Phase 1 primary manifest membership differs",
    )
    primary_record = primary_manifest_candidates[0]
    primary_relative = _safe_relative_json(
        primary_record.get("path"), label="Phase 1 primary manifest"
    )
    primary_payload = _stable_file_bytes(
        root / primary_relative,
        trusted_root=root,
        label="Phase 1 primary manifest",
    )
    _require(
        _sha256_bytes(primary_payload) == primary_record.get("sha256"),
        "Phase 1 primary manifest SHA-256 differs",
    )
    primary = _json_object(primary_payload, label="Phase 1 primary manifest")
    files = primary.get("files")
    _require(isinstance(files, list) and len(files) == 25, "raw file set differs")
    raw_files: dict[str, Mapping[str, object]] = {}
    for item in files:
        _require(isinstance(item, Mapping), "raw file record is invalid")
        filename = item.get("filename")
        byte_identity = item.get("byte_identity")
        _require(
            isinstance(filename, str)
            and filename == Path(filename).name
            and filename.endswith(".csv")
            and isinstance(byte_identity, Mapping)
            and _HEX64.fullmatch(str(byte_identity.get("sha256", ""))) is not None
            and type(byte_identity.get("size_bytes")) is int
            and int(byte_identity["size_bytes"]) > 0
            and filename not in raw_files,
            "raw file identity is invalid",
        )
        raw_files[filename] = MappingProxyType(
            {
                "sha256": str(byte_identity["sha256"]),
                "size_bytes": int(byte_identity["size_bytes"]),
                "family": str(item.get("family")),
                "role": str(item.get("role")),
            }
        )
    ordered_files = {name: dict(raw_files[name]) for name in sorted(raw_files)}
    raw_file_identity_sha = _sha256_bytes(_canonical_json_bytes(ordered_files))
    authority_identity = MappingProxyType(
        {
            "raw_authority_id": RAW_AUTHORITY_ID,
            "phase_01_build_id": str(frozen["phase_01_build_id"]),
            "raw_snapshot_id": str(frozen["raw_snapshot_id"]),
            "verification_id": str(frozen["verification_id"]),
            "frozen_pointer_path": RAW_FROZEN_POINTER.as_posix(),
            "frozen_pointer_sha256": _sha256_bytes(frozen_payload),
            "verified_pointer_path": RAW_VERIFIED_POINTER.as_posix(),
            "verified_pointer_sha256": _sha256_bytes(verified_payload),
            "transitive_manifest_path": (
                RAW_AUTHORITY_ROOT / manifest_relative
            ).as_posix(),
            "transitive_manifest_sha256": manifest_sha,
            "primary_manifest_path": primary_relative.as_posix(),
            "primary_manifest_sha256": _sha256_bytes(primary_payload),
            "raw_file_count": len(ordered_files),
            "raw_file_identities_sha256": raw_file_identity_sha,
            "raw_payload_files_opened": 0,
            "final_period_rows_read": 0,
        }
    )
    return AuthenticatedFinalRawAuthority(
        identity=authority_identity,
        raw_files=MappingProxyType(ordered_files),
    )


def _builder_implementation_identity(project_root: Path) -> Mapping[str, object]:
    code = _hash_members(
        project_root,
        _FORMULA_CODE_MEMBERS,
        label="final formula code",
    )
    config = _hash_members(
        project_root,
        _FORMULA_CONFIG_MEMBERS,
        label="final formula config",
    )
    return MappingProxyType(
        {
            "builder_id": BUILDER_ID,
            "formula_replay_id": FORMULA_REPLAY_ID,
            "formula_code_hashes": code,
            "formula_code_hashes_sha256": _sha256_bytes(_canonical_json_bytes(code)),
            "formula_config_hashes": config,
            "formula_config_hashes_sha256": _sha256_bytes(
                _canonical_json_bytes(config)
            ),
        }
    )


def _source_native_schema(target: pa.Schema) -> pa.Schema:
    """The replay schema uses ``source_holding_month`` before normalization."""

    _require(
        target.names and target.names[0] == "holding_month",
        "normalized target lacks the holding-month axis",
    )
    return pa.schema(
        [
            pa.field("source_holding_month", pa.string(), nullable=False),
            *list(target)[1:],
        ],
        metadata=target.metadata,
    )


def _normalize_native_table(
    table: pa.Table,
    *,
    target: pa.Schema,
    months: tuple[str, ...],
    label: str,
) -> pa.Table:
    _require(isinstance(table, pa.Table), f"{label} is not an Arrow table")
    source = table.combine_chunks()
    expected = _source_native_schema(target)
    _require(
        source.schema.equals(expected, check_metadata=True),
        f"{label} native formula schema differs",
    )
    values = source["source_holding_month"].to_pylist()
    _require(
        values
        and all(isinstance(value, str) for value in values)
        and tuple(sorted(set(values))) == months,
        f"{label} native holding-month domain differs",
    )
    renamed = source.rename_columns(target.names)
    _require(
        renamed.schema.equals(target, check_metadata=True),
        f"{label} normalized schema differs",
    )
    return renamed


def _issue_test_formula_replay(
    tables: Mapping[str, pa.Table],
    *,
    raw_source_identity: Mapping[str, object],
    source_publication_sequence: tuple[str, ...] = (
        EXPECTED_SOURCE_PUBLICATION_SEQUENCE
    ),
    formula_identity: Mapping[str, object] | None = None,
    _token: object = _TEST_REPLAY_TOKEN,
) -> AuthenticatedFinalFormulaReplay:
    """Private synthetic seam; never reachable from the public entry point."""

    _require(_token is _TEST_REPLAY_TOKEN, "test formula replay token differs")
    return AuthenticatedFinalFormulaReplay(
        tables=MappingProxyType(dict(tables)),
        raw_source_identity=MappingProxyType(dict(raw_source_identity)),
        source_publication_sequence=tuple(source_publication_sequence),
        formula_identity=MappingProxyType(
            dict(formula_identity or {"synthetic_formula_replay": True})
        ),
        _seal=_REPLAY_SEAL,
    )


def compile_final_normalized_sources(
    replay: AuthenticatedFinalFormulaReplay,
    *,
    project_root: Path = PROJECT_ROOT,
    stage12_binding_sha256: str | None = None,
    approval_payload_sha256: str | None = None,
) -> FinalNormalizedSourceBuild:
    """Normalize and independently certify one sealed native formula replay."""

    _require(
        type(replay) is AuthenticatedFinalFormulaReplay,
        "final source compiler requires an authenticated formula replay",
    )
    _require(
        replay.source_publication_sequence == EXPECTED_SOURCE_PUBLICATION_SEQUENCE,
        "decision/outcome source publication chronology differs",
    )
    _require(
        isinstance(replay.raw_source_identity, Mapping)
        and bool(replay.raw_source_identity),
        "raw source identity is absent",
    )
    _require(
        isinstance(replay.formula_identity, Mapping) and bool(replay.formula_identity),
        "formula identity is absent",
    )
    schemas = _source_schemas(project_root)
    normalized: dict[str, pa.Table] = {}
    for name in _SOURCE_NAMES:
        normalized[name] = _normalize_native_table(
            replay.tables[name],
            target=schemas[name],
            months=_SOURCE_MONTHS[name],
            label=name,
        )
    validated = _validate_normalized_source_tables(
        normalized,
        project_root=project_root,
    )
    table_identity = {
        name: {
            "rows": table.num_rows,
            "arrow_schema_sha256": _sha256_bytes(table.schema.serialize().to_pybytes()),
        }
        for name, table in validated.items()
    }
    builder: dict[str, object] = {
        "builder_id": BUILDER_ID,
        "formula_replay_id": FORMULA_REPLAY_ID,
        "decision_sources": list(DECISION_SOURCE_NAMES),
        "outcome_sources": list(OUTCOME_SOURCE_NAMES),
        "source_publication_sequence": list(replay.source_publication_sequence),
        "decision_source_tables_committed_before_outcome_source_open": True,
        "formula_identity": dict(replay.formula_identity),
        "table_identity": table_identity,
        "normalized_source_count": len(validated),
    }
    if stage12_binding_sha256 is not None:
        _require(
            _HEX64.fullmatch(stage12_binding_sha256) is not None,
            "Stage 12 binding SHA-256 is malformed",
        )
        builder["stage12_binding_sha256"] = stage12_binding_sha256
    if approval_payload_sha256 is not None:
        _require(
            _HEX64.fullmatch(approval_payload_sha256) is not None,
            "approval payload SHA-256 is malformed",
        )
        builder["approval_payload_sha256"] = approval_payload_sha256
    builder["builder_receipt_sha256"] = _sha256_bytes(_canonical_json_bytes(builder))
    return FinalNormalizedSourceBuild(
        tables=validated,
        raw_source_identity=MappingProxyType(dict(replay.raw_source_identity)),
        builder_identity=MappingProxyType(builder),
    )


def _execute_repository_native_formula_replay(
    *,
    raw_authority: AuthenticatedFinalRawAuthority,
    stage12_binding: Stage12VerifiedBinding,
    approval_payload_sha256: str,
    project_root: Path,
) -> AuthenticatedFinalFormulaReplay:
    """Run the fixed raw scans through unchanged Stage 6/8/11 formulas.

    The large replay implementation is isolated so importing this module and
    authenticating metadata cannot import a raw reader.  Its callable is also
    zero-argument from the user's perspective: every path and date is fixed by
    repository authority and the authenticated objects passed here.
    """

    # Delayed import is a security boundary, not an optional backend.  The
    # imported implementation is a frozen member of ``formula_identity``.
    from portfolio_bot.final_test.final_source_replay import (
        execute_repository_native_final_replay,
    )

    result = execute_repository_native_final_replay(
        raw_authority=raw_authority,
        stage12_binding_sha256=stage12_binding.binding_sha256,
        approval_payload_sha256=approval_payload_sha256,
    )
    _require(
        type(result) is AuthenticatedFinalFormulaReplay,
        "native final replay returned an unsealed result",
    )
    return result


def build_final_normalized_source_tables() -> FinalNormalizedSourceBuild:
    """Sealed production raw-to-12-normalized-Arrow entry point."""

    root = PROJECT_ROOT.absolute()
    binding = load_authenticated_stage12_binding(root)
    approval = _load_fixed_approval(root, binding=binding)
    authorized = require_stage13_unblind_authorization(
        approval,
        expected_stage12_binding=binding,
    )
    raw = authenticate_final_raw_authority(root)
    implementation = _builder_implementation_identity(root)
    replay = _execute_repository_native_formula_replay(
        raw_authority=raw,
        stage12_binding=binding,
        approval_payload_sha256=authorized.payload_sha256,
        project_root=root,
    )
    _require(
        replay.raw_source_identity.get("raw_snapshot_id")
        == raw.identity.get("raw_snapshot_id")
        and replay.raw_source_identity.get("raw_file_identities_sha256")
        == raw.identity.get("raw_file_identities_sha256"),
        "native replay raw authority differs",
    )
    replay_formula = dict(replay.formula_identity)
    replay_formula.update(implementation)
    rebound = AuthenticatedFinalFormulaReplay(
        tables=replay.tables,
        raw_source_identity=replay.raw_source_identity,
        source_publication_sequence=replay.source_publication_sequence,
        formula_identity=MappingProxyType(replay_formula),
        _seal=_REPLAY_SEAL,
    )
    return compile_final_normalized_sources(
        rebound,
        project_root=root,
        stage12_binding_sha256=binding.binding_sha256,
        approval_payload_sha256=authorized.payload_sha256,
    )


__all__ = [
    "BUILDER_ID",
    "FinalNormalizedSourceBuild",
    "FinalSourceBuilderError",
    "authenticate_final_raw_authority",
    "build_final_normalized_source_tables",
]
