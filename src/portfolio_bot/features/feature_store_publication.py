"""Primary Stage 6 feature-store assembly and secure publication.

The module consumes authenticated family-cache completions, stream-zips their
wide values into the raw feature store, concatenates the seven lineage shards,
builds the frozen processed/quality payloads, and publishes the primary
manifest and mutable primary pointer last.  It never constructs labels,
models, portfolios, P&L, validation data, or final-period data.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    AuthenticatedBundleGuard,
    PublishedBundle,
    SecurePublicationError,
    SecureStagingDirectory,
    replace_pointer_last,
)
from portfolio_bot.features.feature_store_contract import (
    DEFAULT_POLICY,
    LINEAGE_SCHEMA,
    PROJECT_ROOT,
    sha256_file,
    validate_feature_store_contract,
)
from portfolio_bot.features.feature_store_inputs import FeatureStoreInputs
from portfolio_bot.features.feature_store_family_orchestration import (
    FamilyCompletionReceipt,
    Stage6FamilyCacheResult,
    authenticate_family_completion,
)
from portfolio_bot.features.feature_store_materialization import (
    FamilyBatchReceipt,
    FeatureStoreBuildIdentity,
    TableReceipt,
    authenticate_materialized_family_batch,
    family_feature_names,
    family_value_schema,
    reauthenticate_feature_store_build_identity,
    validate_precomputed_feature_store_build_identity,
)
from portfolio_bot.features.feature_store_quality import (
    FEATURE_COVERAGE_SCHEMA,
    FEATURE_MISSINGNESS_SCHEMA,
    FEATURE_OUTLIERS_SCHEMA,
    FEATURE_SOURCE_COVERAGE_SCHEMA,
    FEATURE_TIMING_CHECKS_SCHEMA,
    FeatureStoreQualityArtifacts,
    build_feature_store_quality_artifacts,
)
from portfolio_bot.features.feature_store_registries import (
    CATEGORY_VOCABULARY_SCHEMA,
    FEATURE_DICTIONARY_SCHEMA,
    FEATURE_SET_MEMBERSHIP_SCHEMA,
    build_category_vocabulary,
    build_feature_dictionary,
    build_feature_set_membership,
)
from portfolio_bot.features.phase_03_contract import canonical_json_bytes


_INTERIM_LOCATION = "interim_feature_store"
_INTERIM_STATUS = "PRIMARY_INTERIM_COMPLETE"
_PROCESSED_LOCATION = "processed_feature_registries"
_PROCESSED_STATUS = "PRIMARY_PROCESSED_COMPLETE"
_QUALITY_LOCATION = "feature_store_quality"
_QUALITY_STATUS = "PRIMARY_QUALITY_PASSED"
_PRIMARY_STATUS = "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
_SUCCESS_NAME = "_SUCCESS.json"
_MANIFEST_NAME = "manifest.json"
_AUDIT_NAMES = (
    "signal_date",
    "permno",
    "permco",
    "long_eligible",
    "short_eligible",
    "primary_exchange",
    "share_type",
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
_PRIMARY_MANIFEST_KEYS = frozenset(
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
_POINTER_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "policy_id",
        "build_id",
        "status",
        "primary_manifest_sha256",
        "authorizations",
    }
)


class FeatureStorePublicationError(RuntimeError):
    """Raised when primary Stage 6 assembly or publication fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeatureStorePublicationError(message)


def _project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    _require(isinstance(value, dict), f"Expected TOML object: {path}")
    return value


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _parquet_physical_schema(schema: pa.Schema) -> pa.Schema:
    """Return PyArrow's canonical Parquet round-trip list child naming."""

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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PublicationFileReceipt:
    path: str
    sha256: str
    size_bytes: int
    rows: int | None
    row_groups: int | None
    schema_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "row_groups": self.row_groups,
            "schema_sha256": self.schema_sha256,
        }


@dataclass(frozen=True)
class PublishedOutputSet:
    path: Path
    bundle: PublishedBundle
    files: Mapping[str, PublicationFileReceipt]

    @property
    def success_sha256(self) -> str:
        return self.bundle.snapshot.hashes[_SUCCESS_NAME]


@dataclass(frozen=True)
class PrimaryFeatureStorePublication:
    build_id: str
    interim_path: Path
    processed_path: Path
    quality_path: Path
    manifest_path: Path
    pointer_path: Path
    manifest: Mapping[str, Any]
    pointer: Mapping[str, Any]
    reused_existing: bool


@dataclass(frozen=True)
class _PublicationContext:
    policy: Mapping[str, Any]
    contract: Mapping[str, Any]
    family_order: tuple[str, ...]
    raw_schema: pa.Schema
    required_raw_rows: int
    required_lineage_rows: int
    row_group_size: int
    compression: str
    dictionary: bool
    interim_files: tuple[str, ...]
    processed_files: tuple[str, ...]
    quality_files: tuple[str, ...]
    interim_path: Path
    processed_path: Path
    quality_path: Path
    manifest_directory: Path
    pointer_path: Path


@dataclass(frozen=True)
class _AssemblyBatch:
    path: Path
    build_id: str
    family: str
    batch_id: str
    row_count: int
    lineage_row_count: int


def _required_scope_counts(
    policy: Mapping[str, Any],
) -> tuple[int, int]:
    """Return frozen counts; isolated for small synthetic publication tests."""

    return (
        int(policy["scope"]["required_spine_rows"]),
        int(policy["scope"]["required_lineage_rows"]),
    )


def _context(
    *,
    identity: FeatureStoreBuildIdentity,
    inputs: FeatureStoreInputs,
    policy_path: Path,
    project_root: Path,
) -> _PublicationContext:
    contract = validate_feature_store_contract(policy_path)
    policy = _toml(policy_path)
    _require(
        identity.build_id.startswith("phase03-features-"),
        "Primary feature-store build ID is malformed",
    )
    _require(
        identity.contract_id == policy["contract_id"]
        and identity.policy_id == policy["policy_id"],
        "Primary identity differs from frozen policy",
    )
    family_order = tuple(str(value) for value in policy["build"]["family_order"])
    required_raw_rows, required_lineage_rows = _required_scope_counts(policy)
    database_root = Path(os.path.abspath(inputs.database_root.expanduser()))

    def database_path(key: str) -> Path:
        relative = Path(str(policy["database"][key]))
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts,
            f"Unsafe Stage 6 database namespace: {key}",
        )
        return database_root / relative / identity.build_id

    artifact_root = Path(str(policy["publication"]["artifact_root"]))
    _require(
        not artifact_root.is_absolute()
        and ".." not in artifact_root.parts,
        "Unsafe feature-store artifact namespace",
    )
    pointer_relative = Path(str(policy["publication"]["latest_primary_pointer"]))
    _require(
        not pointer_relative.is_absolute()
        and ".." not in pointer_relative.parts,
        "Unsafe feature-store primary pointer",
    )
    return _PublicationContext(
        policy=policy,
        contract=contract,
        family_order=family_order,
        raw_schema=contract["raw_schema"],
        required_raw_rows=required_raw_rows,
        required_lineage_rows=required_lineage_rows,
        row_group_size=int(policy["build"]["row_group_size"]),
        compression=str(policy["build"]["compression"]),
        dictionary=bool(policy["build"]["dictionary_encode_strings"]),
        interim_files=tuple(sorted(policy["outputs"]["interim"]["required_files"])),
        processed_files=tuple(sorted(policy["outputs"]["processed"]["required_files"])),
        quality_files=tuple(sorted(policy["outputs"]["quality"]["required_files"])),
        interim_path=database_path("interim_build_root"),
        processed_path=database_path("processed_build_root"),
        quality_path=database_path("quality_build_root"),
        manifest_directory=(
            project_root
            / artifact_root
            / str(policy["publication"]["primary_build_namespace"])
            / identity.build_id
        ),
        pointer_path=project_root / pointer_relative,
    )


def _receipt_from_cache(
    receipt: TableReceipt,
) -> PublicationFileReceipt:
    return PublicationFileReceipt(
        path=receipt.path,
        sha256=receipt.sha256,
        size_bytes=receipt.size_bytes,
        rows=receipt.rows,
        row_groups=receipt.row_groups,
        schema_sha256=receipt.schema_sha256,
    )


def _compare_family_receipts(
    provided: FamilyBatchReceipt,
    authenticated: FamilyBatchReceipt,
) -> None:
    _require(
        provided.path == authenticated.path
        and provided.build_id == authenticated.build_id
        and provided.family == authenticated.family
        and provided.batch_id == authenticated.batch_id
        and provided.row_count == authenticated.row_count
        and provided.lineage_row_count == authenticated.lineage_row_count
        and provided.table_receipts == authenticated.table_receipts
        and provided.success_sha256 == authenticated.success_sha256
        and provided.success_size_bytes == authenticated.success_size_bytes,
        f"Family completion changed: {provided.family}/{provided.batch_id}",
    )


def _completion_batches(
    receipt: FamilyCompletionReceipt,
    *,
    identity: FeatureStoreBuildIdentity,
    context: _PublicationContext,
    policy_path: Path,
) -> tuple[_AssemblyBatch, ...]:
    family = receipt.family
    feature_count = len(family_feature_names(family, policy_path))
    _require(
        receipt.build_id == identity.build_id
        and receipt.feature_count == feature_count
        and receipt.batch_count == len(receipt.batches)
        and receipt.row_count == context.required_raw_rows
        and receipt.lineage_row_count
        == context.required_raw_rows * feature_count
        and receipt.path.name == "_completion",
        f"Family completion summary differs: {family}",
    )
    expected_ordinal = 0
    previous_last = None
    batches: list[_AssemblyBatch] = []
    for summary in receipt.batches:
        _require(
            summary.ordinal == expected_ordinal
            and summary.row_count > 0
            and summary.lineage_row_count
            == summary.row_count * feature_count
            and (
                previous_last is None
                or previous_last < summary.first_key
            )
            and summary.first_key <= summary.last_key,
            f"Family completion batch order differs: "
            f"{family}/{summary.batch_id}",
        )
        path = receipt.path.parent / summary.batch_id
        _require(
            path.is_dir() and not path.is_symlink(),
            f"Family completion batch path is unsafe: "
            f"{family}/{summary.batch_id}",
        )
        batches.append(
            _AssemblyBatch(
                path=path,
                build_id=identity.build_id,
                family=family,
                batch_id=summary.batch_id,
                row_count=summary.row_count,
                lineage_row_count=summary.lineage_row_count,
            )
        )
        previous_last = summary.last_key
        expected_ordinal += 1
    _require(
        batches
        and sum(item.row_count for item in batches)
        == context.required_raw_rows,
        f"Family completion rows differ: {family}",
    )
    return tuple(batches)


def _authenticate_family_completions(
    *,
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    family_batches: (
        Mapping[
            str,
            Sequence[FamilyBatchReceipt] | FamilyCompletionReceipt,
        ]
        | Stage6FamilyCacheResult
    ),
    context: _PublicationContext,
    policy_path: Path,
) -> dict[str, tuple[_AssemblyBatch, ...]]:
    if isinstance(family_batches, Stage6FamilyCacheResult):
        _require(
            family_batches.build_identity == identity
            and family_batches.final_identity_reauthenticated is True,
            "Stage 6 cache result does not bind the supplied final identity",
        )
        completion_map: Mapping[str, FamilyCompletionReceipt] = (
            family_batches.family_receipt_map
        )
        _require(
            set(completion_map) == set(context.family_order),
            "Stage 6 cache result does not contain seven families",
        )
        # This is the explicit no-double-auth bridge.  The in-process
        # orchestrator result already authenticated every batch and performed
        # its final identity reauthentication.
        return {
            family: _completion_batches(
                completion_map[family],
                identity=identity,
                context=context,
                policy_path=policy_path,
            )
            for family in context.family_order
        }

    _require(
        set(family_batches) == set(context.family_order),
        "Family completion map differs from frozen seven-family order",
    )
    result: dict[str, tuple[_AssemblyBatch, ...]] = {}
    physical_paths: set[str] = set()
    for family in context.family_order:
        supplied = family_batches[family]
        if isinstance(supplied, FamilyCompletionReceipt):
            authenticated_completion = authenticate_family_completion(
                inputs=inputs,
                identity=identity,
                family=family,
                policy_path=policy_path,
            )
            _require(
                authenticated_completion == supplied
                or authenticated_completion.__dict__
                == {
                    **supplied.__dict__,
                    "reused_existing": True,
                },
                f"Family completion receipt changed: {family}",
            )
            result[family] = _completion_batches(
                authenticated_completion,
                identity=identity,
                context=context,
                policy_path=policy_path,
            )
            continue
        provided = tuple(supplied)
        _require(provided, f"Feature family has no completed batches: {family}")
        batch_ids = [receipt.batch_id for receipt in provided]
        _require(
            len(batch_ids) == len(set(batch_ids)),
            f"Feature family repeats a batch ID: {family}",
        )
        authenticated_items: list[_AssemblyBatch] = []
        for receipt in provided:
            _require(
                isinstance(receipt, FamilyBatchReceipt)
                and receipt.family == family
                and receipt.build_id == identity.build_id,
                f"Invalid family completion: {family}",
            )
            authenticated = authenticate_materialized_family_batch(
                inputs=inputs,
                family=family,
                batch_id=receipt.batch_id,
                policy_path=policy_path,
                build_identity=identity,
            )
            _compare_family_receipts(receipt, authenticated)
            path_label = authenticated.path.as_posix()
            _require(
                path_label not in physical_paths,
                "One cache completion was reused across families/batches",
            )
            physical_paths.add(path_label)
            authenticated_items.append(
                _AssemblyBatch(
                    path=authenticated.path,
                    build_id=authenticated.build_id,
                    family=authenticated.family,
                    batch_id=authenticated.batch_id,
                    row_count=authenticated.row_count,
                    lineage_row_count=authenticated.lineage_row_count,
                )
            )
        row_count = sum(item.row_count for item in authenticated_items)
        feature_count = len(family_feature_names(family, policy_path))
        lineage_count = sum(item.lineage_row_count for item in authenticated_items)
        _require(
            row_count == context.required_raw_rows,
            f"Family row count differs: {family}",
        )
        _require(
            lineage_count == row_count * feature_count,
            f"Family lineage count differs: {family}",
        )
        result[family] = tuple(authenticated_items)
    _require(
        sum(
            item.lineage_row_count
            for family in context.family_order
            for item in result[family]
        )
        == context.required_lineage_rows,
        "Combined lineage completion count differs",
    )
    return result


class _HashingSink:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()
        self._size = 0
        self.closed = False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._size

    def write(self, payload: bytes | bytearray | memoryview) -> int:
        _require(not self.closed, "Publication Parquet sink is closed")
        view = memoryview(payload)
        written = self._stream.write(view)
        _require(written == len(view), "Short Stage 6 publication write")
        self._digest.update(view)
        self._size += written
        return written

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        if not self.closed:
            self.flush()
            self.closed = True

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    @property
    def size_bytes(self) -> int:
        return self._size


def _write_parquet_batches(
    *,
    stage: SecureStagingDirectory,
    name: str,
    schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
    row_group_size: int,
    compression: str,
    dictionary: bool,
) -> PublicationFileReceipt:
    rows = row_groups = 0
    with stage.open_exclusive_binary(name) as stream:
        sink = _HashingSink(stream)
        writer = pq.ParquetWriter(
            sink,
            schema,
            compression=compression,
            use_dictionary=dictionary,
            version="2.6",
            write_statistics=True,
        )
        try:
            for batch in batches:
                _require(
                    isinstance(batch, pa.RecordBatch)
                    and batch.schema.equals(schema, check_metadata=True),
                    f"Published Parquet batch schema differs: {name}",
                )
                if batch.num_rows == 0:
                    continue
                writer.write_batch(batch, row_group_size=row_group_size)
                rows += batch.num_rows
                row_groups += math.ceil(batch.num_rows / row_group_size)
        finally:
            writer.close()
        sink.flush()
    return PublicationFileReceipt(
        path=name,
        sha256=sink.sha256,
        size_bytes=sink.size_bytes,
        rows=rows,
        row_groups=row_groups,
        schema_sha256=_schema_sha256(_parquet_physical_schema(schema)),
    )


def _write_parquet_table(
    *,
    stage: SecureStagingDirectory,
    name: str,
    table: pa.Table,
    row_group_size: int,
    compression: str,
    dictionary: bool,
) -> PublicationFileReceipt:
    batches = table.to_batches(max_chunksize=row_group_size)
    if not batches:
        batches = [pa.RecordBatch.from_arrays(
            [pa.array([], type=field.type) for field in table.schema],
            schema=table.schema,
        )]
    return _write_parquet_batches(
        stage=stage,
        name=name,
        schema=table.schema,
        batches=batches,
        row_group_size=row_group_size,
        compression=compression,
        dictionary=dictionary,
    )


def _write_bytes(
    stage: SecureStagingDirectory,
    name: str,
    payload: bytes,
) -> PublicationFileReceipt:
    with stage.open_exclusive_binary(name) as stream:
        stream.write(payload)
    return PublicationFileReceipt(
        path=name,
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        rows=None,
        row_groups=None,
        schema_sha256=None,
    )


def _success_payload(
    *,
    identity: FeatureStoreBuildIdentity,
    location_kind: str,
    status: str,
    receipts: Mapping[str, PublicationFileReceipt],
) -> bytes:
    payload = {
        "schema_version": 1,
        "contract_id": identity.contract_id,
        "policy_id": identity.policy_id,
        "build_id": identity.build_id,
        "location_kind": location_kind,
        "status": status,
        "configuration_hashes": identity.configuration_hash_map,
        "code_hashes": identity.code_hash_map,
        "input_receipts_sha256": identity.input_receipts_sha256,
        "table_receipts": {
            name: receipts[name].as_dict()
            for name in sorted(receipts)
        },
    }
    _require(set(payload) == _SUCCESS_KEYS, "Primary _SUCCESS keys differ")
    return canonical_json_bytes(payload)


def _parquet_validator(
    receipts: Mapping[str, PublicationFileReceipt],
    schemas: Mapping[str, pa.Schema],
):
    def validate(name: str, descriptor: int, record) -> None:
        receipt = receipts.get(name)
        if receipt is None:
            return
        _require(
            record.sha256 == receipt.sha256
            and record.size_bytes == receipt.size_bytes,
            f"Published file receipt differs: {name}",
        )
        if receipt.rows is None:
            return
        duplicate = os.dup(descriptor)
        try:
            with os.fdopen(duplicate, "rb") as handle:
                parquet = pq.ParquetFile(handle)
                observed_schema = parquet.schema_arrow
                observed_rows = parquet.metadata.num_rows
                observed_groups = parquet.metadata.num_row_groups
        except Exception as exc:
            raise FeatureStorePublicationError(
                f"Cannot validate published Parquet: {name}"
            ) from exc
        _require(
            observed_schema.equals(
                _parquet_physical_schema(schemas[name]),
                check_metadata=True,
            )
            and observed_rows == receipt.rows
            and observed_groups == receipt.row_groups
            and _schema_sha256(observed_schema) == receipt.schema_sha256,
            f"Published Parquet metadata differs: {name}",
        )

    return validate


def _publish_staged_set(
    *,
    trusted_root: Path,
    destination: Path,
    expected_files: Sequence[str],
    identity: FeatureStoreBuildIdentity,
    location_kind: str,
    status: str,
    table_writer,
    pre_publish_gate,
) -> PublishedOutputSet:
    receipts: dict[str, PublicationFileReceipt] = {}
    schemas: dict[str, pa.Schema] = {}
    with SecureStagingDirectory.create(
        trusted_root=trusted_root,
        destination=destination,
        expected_files=expected_files,
        label=f"Stage 6 {location_kind}",
    ) as stage:
        written, written_schemas = table_writer(stage)
        receipts.update(written)
        schemas.update(written_schemas)
        _require(
            set(receipts) == set(expected_files) - {_SUCCESS_NAME},
            f"{location_kind} payload set differs before _SUCCESS",
        )
        success = _success_payload(
            identity=identity,
            location_kind=location_kind,
            status=status,
            receipts=receipts,
        )
        _write_bytes(stage, _SUCCESS_NAME, success)
        validator = _parquet_validator(receipts, schemas)
        written_hashes = {
            _SUCCESS_NAME: _sha256_bytes(success),
            **{
                name: receipt.sha256
                for name, receipt in receipts.items()
            },
        }
        snapshot = stage.snapshot(
            expected_hashes=written_hashes,
            file_validator=validator,
        )
        bundle = stage.publish(
            expected_hashes=snapshot.hashes,
            file_validator=validator,
            pre_publish_gate=pre_publish_gate,
            post_publish_gate=pre_publish_gate,
        )
    return PublishedOutputSet(
        path=destination,
        bundle=bundle,
        files=dict(receipts),
    )


def _value_batch_iterator(
    receipts: Sequence[_AssemblyBatch],
    family: str,
    *,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    filename = f"feature_values_{family}.parquet"
    for receipt in receipts:
        parquet = pq.ParquetFile(receipt.path / filename)
        yield from parquet.iter_batches(batch_size=batch_size)


class _BatchCursor:
    def __init__(self, batches: Iterable[pa.RecordBatch]) -> None:
        self._batches = iter(batches)
        self._batch: pa.RecordBatch | None = None
        self._offset = 0

    def _ensure(self) -> None:
        while self._batch is None or self._offset >= self._batch.num_rows:
            try:
                self._batch = next(self._batches)
            except StopIteration:
                self._batch = None
                self._offset = 0
                return
            self._offset = 0
            if self._batch.num_rows:
                return

    @property
    def remaining(self) -> int | None:
        self._ensure()
        if self._batch is None:
            return None
        return self._batch.num_rows - self._offset

    def take(self, length: int) -> pa.RecordBatch:
        self._ensure()
        _require(
            self._batch is not None
            and 0 < length <= self._batch.num_rows - self._offset,
            "Family value cursor cannot satisfy aligned slice",
        )
        result = self._batch.slice(self._offset, length)
        self._offset += length
        return result


def _raw_batches(
    *,
    family_batches: Mapping[str, Sequence[_AssemblyBatch]],
    context: _PublicationContext,
    policy_path: Path,
) -> Iterator[pa.RecordBatch]:
    cursors = {
        family: _BatchCursor(
            _value_batch_iterator(
                family_batches[family],
                family,
                batch_size=context.row_group_size,
            )
        )
        for family in context.family_order
    }
    feature_names = {
        family: family_feature_names(family, policy_path)
        for family in context.family_order
    }
    emitted = 0
    previous_key: tuple[Any, int] | None = None
    while True:
        remaining = {
            family: cursor.remaining
            for family, cursor in cursors.items()
        }
        exhausted = {
            family for family, count in remaining.items() if count is None
        }
        if exhausted:
            _require(
                len(exhausted) == len(cursors),
                "Feature families ended at different raw-store rows",
            )
            break
        length = min(int(value) for value in remaining.values())
        slices = {
            family: cursors[family].take(length)
            for family in context.family_order
        }
        anchor = slices[context.family_order[0]]
        for family in context.family_order[1:]:
            candidate = slices[family]
            for name in _AUDIT_NAMES:
                _require(
                    anchor.column(
                        anchor.schema.get_field_index(name)
                    ).equals(
                        candidate.column(
                            candidate.schema.get_field_index(name)
                        )
                    ),
                    f"Family audit spine differs while assembling: {family}/{name}",
                )
        dates = anchor.column(
            anchor.schema.get_field_index("signal_date")
        ).to_pylist()
        permnos = anchor.column(
            anchor.schema.get_field_index("permno")
        ).to_pylist()
        for key in zip(dates, permnos, strict=True):
            normalized = (key[0], int(key[1]))
            _require(
                previous_key is None or previous_key < normalized,
                "Raw feature-store keys are not globally unique and sorted",
            )
            previous_key = normalized
        arrays = [
            anchor.column(anchor.schema.get_field_index(name))
            for name in _AUDIT_NAMES
        ]
        for family in context.family_order:
            batch = slices[family]
            arrays.extend(
                batch.column(batch.schema.get_field_index(name))
                for name in feature_names[family]
            )
        output = pa.RecordBatch.from_arrays(
            arrays,
            schema=context.raw_schema,
        )
        emitted += output.num_rows
        yield output
    _require(
        emitted == context.required_raw_rows,
        "Stream-assembled raw row count differs",
    )


def _lineage_batches(
    *,
    family: str,
    receipts: Sequence[_AssemblyBatch],
    expected_rows: int,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    filename = f"feature_lineage_{family}.parquet"
    emitted = 0
    for receipt in receipts:
        parquet = pq.ParquetFile(receipt.path / filename)
        for batch in parquet.iter_batches(batch_size=batch_size):
            _require(
                batch.schema.equals(LINEAGE_SCHEMA, check_metadata=True),
                f"Cache lineage schema differs: {family}",
            )
            emitted += batch.num_rows
            yield batch
    _require(
        emitted == expected_rows,
        f"Published lineage row count differs: {family}",
    )


def _publish_interim(
    *,
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    family_batches: Mapping[str, Sequence[_AssemblyBatch]],
    context: _PublicationContext,
    policy_path: Path,
) -> PublishedOutputSet:
    def writer(stage: SecureStagingDirectory):
        receipts: dict[str, PublicationFileReceipt] = {}
        schemas: dict[str, pa.Schema] = {}
        raw_name = "feature_store_raw.parquet"
        receipts[raw_name] = _write_parquet_batches(
            stage=stage,
            name=raw_name,
            schema=context.raw_schema,
            batches=_raw_batches(
                family_batches=family_batches,
                context=context,
                policy_path=policy_path,
            ),
            row_group_size=context.row_group_size,
            compression=context.compression,
            dictionary=context.dictionary,
        )
        schemas[raw_name] = context.raw_schema
        for family in context.family_order:
            name = f"feature_lineage_{family}.parquet"
            expected = (
                context.required_raw_rows
                * len(family_feature_names(family, policy_path))
            )
            receipts[name] = _write_parquet_batches(
                stage=stage,
                name=name,
                schema=LINEAGE_SCHEMA,
                batches=_lineage_batches(
                    family=family,
                    receipts=family_batches[family],
                    expected_rows=expected,
                    batch_size=context.row_group_size,
                ),
                row_group_size=context.row_group_size,
                compression=context.compression,
                dictionary=context.dictionary,
            )
            schemas[name] = LINEAGE_SCHEMA
        return receipts, schemas

    return _publish_staged_set(
        trusted_root=inputs.database_root,
        destination=context.interim_path,
        expected_files=context.interim_files,
        identity=identity,
        location_kind=_INTERIM_LOCATION,
        status=_INTERIM_STATUS,
        table_writer=writer,
        pre_publish_gate=lambda: validate_precomputed_feature_store_build_identity(
            inputs,
            identity,
            policy_path,
        ),
    )


def _iter_vocabulary_rows(
    raw_path: Path,
    policy_path: Path,
    *,
    batch_size: int,
) -> Iterator[Mapping[str, Any]]:
    registry = build_feature_dictionary(policy_path)
    categorical = [
        row["feature_name"]
        for row in registry.to_pylist()
        if row["kind"] in {
            "categorical_firm",
            "categorical_multihot_firm",
        }
    ]
    columns = ["signal_date", "long_eligible", *categorical]
    parquet = pq.ParquetFile(raw_path)
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):
        yield from batch.to_pylist()


def _publish_processed(
    *,
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    context: _PublicationContext,
    policy_path: Path,
) -> tuple[PublishedOutputSet, pa.Table, pa.Table, pa.Table]:
    dictionary = build_feature_dictionary(policy_path)
    membership = build_feature_set_membership(policy_path)
    vocabulary = build_category_vocabulary(
        _iter_vocabulary_rows(
            context.interim_path / "feature_store_raw.parquet",
            policy_path,
            batch_size=context.row_group_size,
        ),
        policy_path,
    )
    tables = {
        "feature_dictionary.parquet": dictionary,
        "feature_set_membership.parquet": membership,
        "category_vocabulary.parquet": vocabulary,
    }
    expected_schemas = {
        "feature_dictionary.parquet": FEATURE_DICTIONARY_SCHEMA,
        "feature_set_membership.parquet": FEATURE_SET_MEMBERSHIP_SCHEMA,
        "category_vocabulary.parquet": CATEGORY_VOCABULARY_SCHEMA,
    }
    for name, table in tables.items():
        _require(
            table.schema.equals(expected_schemas[name]),
            f"{name} does not match its frozen Stage 6 schema",
        )

    def writer(stage: SecureStagingDirectory):
        receipts = {
            name: _write_parquet_table(
                stage=stage,
                name=name,
                table=table,
                row_group_size=context.row_group_size,
                compression=context.compression,
                dictionary=context.dictionary,
            )
            for name, table in tables.items()
        }
        return receipts, {name: table.schema for name, table in tables.items()}

    published = _publish_staged_set(
        trusted_root=inputs.database_root,
        destination=context.processed_path,
        expected_files=context.processed_files,
        identity=identity,
        location_kind=_PROCESSED_LOCATION,
        status=_PROCESSED_STATUS,
        table_writer=writer,
        pre_publish_gate=lambda: validate_precomputed_feature_store_build_identity(
            inputs,
            identity,
            policy_path,
        ),
    )
    return published, dictionary, membership, vocabulary


def _publish_quality(
    *,
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    context: _PublicationContext,
    policy_path: Path,
    provenance_counters: Mapping[str, int],
    memory_limit_mb: int,
) -> tuple[PublishedOutputSet, FeatureStoreQualityArtifacts]:
    lineage_paths = {
        family: (
            context.interim_path
            / f"feature_lineage_{family}.parquet"
        )
        for family in context.family_order
    }
    temporary_directory = (
        Path(inputs.database_root)
        / str(context.policy["database"]["cache_root"])
    )
    temporary_directory.mkdir(parents=True, exist_ok=True)
    quality = build_feature_store_quality_artifacts(
        build_id=identity.build_id,
        raw_store_path=context.interim_path / "feature_store_raw.parquet",
        lineage_paths=lineage_paths,
        provenance_counters=provenance_counters,
        policy_path=policy_path,
        memory_limit_mb=memory_limit_mb,
        temporary_directory=temporary_directory,
        require_pass=True,
    )
    _require(quality.passed, "Stage 6 quality payload did not pass")
    tables = {
        "feature_missingness.parquet": quality.feature_missingness,
        "feature_coverage.parquet": quality.feature_coverage,
        "feature_outliers.parquet": quality.feature_outliers,
        "feature_timing_checks.parquet": quality.feature_timing_checks,
        "feature_source_coverage.parquet": quality.feature_source_coverage,
    }
    expected_schemas = {
        "feature_missingness.parquet": FEATURE_MISSINGNESS_SCHEMA,
        "feature_coverage.parquet": FEATURE_COVERAGE_SCHEMA,
        "feature_outliers.parquet": FEATURE_OUTLIERS_SCHEMA,
        "feature_timing_checks.parquet": FEATURE_TIMING_CHECKS_SCHEMA,
        "feature_source_coverage.parquet": FEATURE_SOURCE_COVERAGE_SCHEMA,
    }
    for name, table in tables.items():
        _require(
            table.schema.equals(expected_schemas[name], check_metadata=True),
            f"Quality Arrow schema differs before publication: {name}",
        )

    def writer(stage: SecureStagingDirectory):
        receipts = {
            name: _write_parquet_table(
                stage=stage,
                name=name,
                table=table,
                row_group_size=context.row_group_size,
                compression=context.compression,
                dictionary=context.dictionary,
            )
            for name, table in tables.items()
        }
        summary_name = "quality_summary.json"
        receipts[summary_name] = _write_bytes(
            stage,
            summary_name,
            quality.quality_summary_json_bytes,
        )
        return receipts, {
            name: table.schema for name, table in tables.items()
        }

    published = _publish_staged_set(
        trusted_root=inputs.database_root,
        destination=context.quality_path,
        expected_files=context.quality_files,
        identity=identity,
        location_kind=_QUALITY_LOCATION,
        status=_QUALITY_STATUS,
        table_writer=writer,
        pre_publish_gate=lambda: validate_precomputed_feature_store_build_identity(
            inputs,
            identity,
            policy_path,
        ),
    )
    return published, quality


def _input_stat_snapshot(
    inputs: FeatureStoreInputs,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    result: list[tuple[str, tuple[int, ...]]] = []
    for name in sorted(inputs.tables):
        path = inputs.tables[name].path
        observed = os.stat(path, follow_symlinks=False)
        _require(
            stat.S_ISREG(observed.st_mode)
            and not stat.S_ISLNK(observed.st_mode),
            f"Input identity is not a regular file: {name}",
        )
        result.append(
            (
                name,
                (
                    observed.st_dev,
                    observed.st_ino,
                    observed.st_mode,
                    observed.st_nlink,
                    observed.st_size,
                    observed.st_mtime_ns,
                    observed.st_ctime_ns,
                ),
            )
        )
    return tuple(result)


def _binding_hash_gate(
    identity: FeatureStoreBuildIdentity,
) -> None:
    for label, expected in (
        *identity.configuration_hashes,
        *identity.code_hashes,
    ):
        candidate = Path(label)
        path = (
            candidate
            if candidate.is_absolute()
            else PROJECT_ROOT / candidate
        )
        _require(
            path.is_file()
            and not path.is_symlink()
            and sha256_file(path) == expected,
            f"Feature-store binding changed before primary pointer: {label}",
        )


def _relative_database_path(path: Path, inputs: FeatureStoreInputs) -> str:
    try:
        return path.resolve().relative_to(
            Path(inputs.database_root).resolve()
        ).as_posix()
    except ValueError as exc:
        raise FeatureStorePublicationError(
            "Published output escaped database root"
        ) from exc


def _manifest_output(
    output: PublishedOutputSet,
    inputs: FeatureStoreInputs,
) -> dict[str, Any]:
    files = {
        name: output.files[name].as_dict()
        for name in sorted(output.files)
    }
    success_record = next(
        record
        for record in output.bundle.snapshot.files
        if record.name == _SUCCESS_NAME
    )
    files[_SUCCESS_NAME] = PublicationFileReceipt(
        path=_SUCCESS_NAME,
        sha256=success_record.sha256,
        size_bytes=success_record.size_bytes,
        rows=None,
        row_groups=None,
        schema_sha256=None,
    ).as_dict()
    return {
        "path": _relative_database_path(output.path, inputs),
        "success_sha256": output.success_sha256,
        "files": {
            name: files[name] for name in sorted(files)
        },
    }


def _input_builds(policy: Mapping[str, Any]) -> dict[str, str]:
    return {
        "calendar": str(policy["inputs"]["calendar_build_id"]),
        "typed_backbone": str(policy["inputs"]["typed_backbone_build_id"]),
        "return_reconciliation": str(
            policy["inputs"]["return_build_id"]
        ),
        "signal_universe": str(
            policy["inputs"]["signal_universe_build_id"]
        ),
        "auxiliary": str(policy["inputs"]["auxiliary_build_id"]),
    }


def _authorizations() -> dict[str, bool]:
    return {
        "independent_verification_required": True,
        "stage_07_labels_authorized": False,
        "models_portfolios_pnl_or_backtests_authorized": False,
        "validation_or_final_test_access_authorized": False,
    }


def _manifest(
    *,
    identity: FeatureStoreBuildIdentity,
    context: _PublicationContext,
    inputs: FeatureStoreInputs,
    interim: PublishedOutputSet,
    processed: PublishedOutputSet,
    quality_output: PublishedOutputSet,
    membership: pa.Table,
    vocabulary: pa.Table,
    quality: FeatureStoreQualityArtifacts,
) -> dict[str, Any]:
    included_counts: dict[str, int] = {}
    for row in membership.to_pylist():
        if row["included"]:
            included_counts[row["feature_set"]] = (
                included_counts.get(row["feature_set"], 0) + 1
            )
    hard_gate_failures = sum(
        not bool(row["passed"])
        for row in quality.feature_timing_checks.to_pylist()
    )
    result = {
        "schema_version": 1,
        "contract_id": identity.contract_id,
        "policy_id": identity.policy_id,
        "build_id": identity.build_id,
        "status": _PRIMARY_STATUS,
        "bindings": {
            "content_fingerprint": identity.content_fingerprint,
            "configuration_hashes": identity.configuration_hash_map,
            "code_hashes": identity.code_hash_map,
            "input_receipts_sha256": identity.input_receipts_sha256,
        },
        "inputs": {
            "input_builds": _input_builds(context.policy),
            "table_receipts": [
                item.as_dict()
                for item in sorted(
                    identity.input_receipts,
                    key=lambda item: item.name,
                )
            ],
        },
        "outputs": {
            "interim": _manifest_output(interim, inputs),
            "processed": _manifest_output(processed, inputs),
            "quality": _manifest_output(quality_output, inputs),
        },
        "summary": {
            "raw_rows": context.required_raw_rows,
            "economic_features": 163,
            "feature_families": len(context.family_order),
            "lineage_rows": context.required_lineage_rows,
            "feature_set_included_counts": {
                name: included_counts[name]
                for name in sorted(included_counts)
            },
            "category_vocabulary_rows": vocabulary.num_rows,
            "hard_gate_failures": hard_gate_failures,
        },
        "authorizations": _authorizations(),
    }
    _require(
        set(result) == _PRIMARY_MANIFEST_KEYS,
        "Primary manifest top-level keys differ",
    )
    _require(
        included_counts
        == {
            "full_feature_primary": 163,
            "reduced_auxiliary_core": 111,
            "market_price_liquidity_core": 79,
        },
        "Primary manifest feature-set counts differ",
    )
    _require(
        hard_gate_failures == 0,
        "Primary manifest cannot publish failed quality gates",
    )
    return result


def _publish_manifest(
    *,
    project_root: Path,
    context: _PublicationContext,
    manifest: Mapping[str, Any],
    gate,
) -> PublishedBundle:
    payload = canonical_json_bytes(manifest)
    with SecureStagingDirectory.create(
        trusted_root=project_root,
        destination=context.manifest_directory,
        expected_files=(_MANIFEST_NAME,),
        label="Stage 6 primary manifest",
    ) as stage:
        with stage.open_exclusive_binary(_MANIFEST_NAME) as stream:
            stream.write(payload)
        snapshot = stage.snapshot(
            expected_hashes={
                _MANIFEST_NAME: _sha256_bytes(payload)
            }
        )
        return stage.publish(
            expected_hashes=snapshot.hashes,
            pre_publish_gate=gate,
            post_publish_gate=gate,
        )


def publish_primary_feature_store(
    *,
    inputs: FeatureStoreInputs,
    build_identity: FeatureStoreBuildIdentity,
    family_batches: (
        Mapping[
            str,
            Sequence[FamilyBatchReceipt] | FamilyCompletionReceipt,
        ]
        | Stage6FamilyCacheResult
    ),
    provenance_counters: Mapping[str, int],
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
    quality_memory_limit_mb: int = 512,
) -> PrimaryFeatureStorePublication:
    """Assemble and securely publish one primary Stage 6 feature-store build."""

    policy_path = policy_path.resolve()
    project_root = project_root.expanduser().resolve()
    _require(
        project_root.is_dir() and not project_root.is_symlink(),
        "Stage 6 project root must be a real directory",
    )
    validate_precomputed_feature_store_build_identity(
        inputs,
        build_identity,
        policy_path,
    )
    context = _context(
        identity=build_identity,
        inputs=inputs,
        policy_path=policy_path,
        project_root=project_root,
    )
    authenticated_batches = _authenticate_family_completions(
        inputs=inputs,
        identity=build_identity,
        family_batches=family_batches,
        context=context,
        policy_path=policy_path,
    )
    interim = _publish_interim(
        inputs=inputs,
        identity=build_identity,
        family_batches=authenticated_batches,
        context=context,
        policy_path=policy_path,
    )
    processed, _dictionary, membership, vocabulary = _publish_processed(
        inputs=inputs,
        identity=build_identity,
        context=context,
        policy_path=policy_path,
    )
    quality_output, quality = _publish_quality(
        inputs=inputs,
        identity=build_identity,
        context=context,
        policy_path=policy_path,
        provenance_counters=provenance_counters,
        memory_limit_mb=quality_memory_limit_mb,
    )

    before_reauthentication = _input_stat_snapshot(inputs)
    reauthenticate_feature_store_build_identity(
        inputs,
        build_identity,
        policy_path,
    )
    after_reauthentication = _input_stat_snapshot(inputs)
    _require(
        before_reauthentication == after_reauthentication,
        "Stage 6 inputs changed during final full reauthentication",
    )
    manifest = _manifest(
        identity=build_identity,
        context=context,
        inputs=inputs,
        interim=interim,
        processed=processed,
        quality_output=quality_output,
        membership=membership,
        vocabulary=vocabulary,
        quality=quality,
    )
    expected_sets = {
        "interim": context.interim_files,
        "processed": context.processed_files,
        "quality": context.quality_files,
    }
    outputs = {
        "interim": interim,
        "processed": processed,
        "quality": quality_output,
    }

    with contextlib.ExitStack() as guards:
        data_guards = {
            name: guards.enter_context(
                AuthenticatedBundleGuard.open(
                    trusted_root=inputs.database_root,
                    path=outputs[name].path,
                    expected_files=expected_sets[name],
                    expected_hashes=outputs[name].bundle.snapshot.hashes,
                    label=f"Stage 6 published {name} bundle",
                    _permit_authenticated_directory_ctime_refresh=True,
                )
            )
            for name in ("interim", "processed", "quality")
        }

        def data_gate() -> None:
            for guard in data_guards.values():
                guard.gate()
            _require(
                _input_stat_snapshot(inputs) == after_reauthentication,
                "Stage 6 inputs changed after final full reauthentication",
            )
            validate_precomputed_feature_store_build_identity(
                inputs,
                build_identity,
                policy_path,
            )
            _binding_hash_gate(build_identity)

        artifact_bundle = _publish_manifest(
            project_root=project_root,
            context=context,
            manifest=manifest,
            gate=data_gate,
        )
        manifest_path = context.manifest_directory / _MANIFEST_NAME
        manifest_guard = guards.enter_context(
            AuthenticatedBundleGuard.open(
                trusted_root=project_root,
                path=context.manifest_directory,
                expected_files=(_MANIFEST_NAME,),
                expected_hashes=artifact_bundle.snapshot.hashes,
                label="Stage 6 published primary manifest",
                _permit_authenticated_directory_ctime_refresh=True,
            )
        )
        pointer = {
            "schema_version": 1,
            "contract_id": build_identity.contract_id,
            "policy_id": build_identity.policy_id,
            "build_id": build_identity.build_id,
            "status": _PRIMARY_STATUS,
            "primary_manifest_sha256": artifact_bundle.snapshot.hashes[
                _MANIFEST_NAME
            ],
            "authorizations": _authorizations(),
        }
        _require(
            set(pointer) == _POINTER_KEYS,
            "Primary pointer keys differ",
        )

        def pointer_gate() -> None:
            data_gate()
            manifest_guard.gate()
            _require(
                sha256_file(manifest_path)
                == pointer["primary_manifest_sha256"],
                "Primary manifest changed before pointer publication",
            )

        pointer_replaced = replace_pointer_last(
            context.pointer_path,
            canonical_json_bytes(pointer),
            trusted_root=project_root,
            pre_replace_gate=pointer_gate,
            post_replace_gate=pointer_gate,
            label="Stage 6 latest primary pointer",
        )
    reused = (
        interim.bundle.reused_existing
        and processed.bundle.reused_existing
        and quality_output.bundle.reused_existing
        and artifact_bundle.reused_existing
        and not pointer_replaced
    )
    return PrimaryFeatureStorePublication(
        build_id=build_identity.build_id,
        interim_path=context.interim_path,
        processed_path=context.processed_path,
        quality_path=context.quality_path,
        manifest_path=context.manifest_directory / _MANIFEST_NAME,
        pointer_path=context.pointer_path,
        manifest=manifest,
        pointer=pointer,
        reused_existing=reused,
    )


def publish_primary_feature_store_from_cache_result(
    *,
    inputs: FeatureStoreInputs,
    cache_result: Stage6FamilyCacheResult,
    provenance_counters: Mapping[str, int],
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
    quality_memory_limit_mb: int = 512,
) -> PrimaryFeatureStorePublication:
    """Publish directly from the in-process orchestrator result.

    This bridge deliberately avoids reauthenticating every family batch a
    second time.  ``materialize_stage6_family_cache`` already authenticated
    those batches, sealed the seven completion receipts, and performed its
    final build-identity reauthentication before returning the typed result.
    The publisher still performs the separately required final full identity
    reauthentication after primary data/quality assembly.
    """

    _require(
        isinstance(cache_result, Stage6FamilyCacheResult),
        "Primary bridge requires Stage6FamilyCacheResult",
    )
    return publish_primary_feature_store(
        inputs=inputs,
        build_identity=cache_result.build_identity,
        family_batches=cache_result,
        provenance_counters=provenance_counters,
        policy_path=policy_path,
        project_root=project_root,
        quality_memory_limit_mb=quality_memory_limit_mb,
    )
