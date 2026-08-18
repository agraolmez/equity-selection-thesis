"""Authenticated byte-preserving rebind of Stage 13 decision checkpoints.

This module provides a deliberately narrow bridge between two Stage 13 replay
contexts.  It does not authorize a Stage 13 run and it does not reinterpret,
recalculate, or relabel a checkpoint.  An authority document pins one closed
parent checkpoint set.  Every parent receipt and Parquet payload is
authenticated before any replacement payload is published; payload bytes are
then copied into the replacement context and a fresh canonical receipt is
created by changing only ``context_sha256``.

The public callable has no discovery behavior.  Its authority path and file
SHA-256 are supplied by the frozen caller, the parent context is selected by
that authority, and the replacement context must already have a canonical
``context.json``.  The completion receipt is the final write, so its presence
proves that the complete pinned set was rebound and reauthenticated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


REBIND_ID = "stage13-decision-checkpoint-v4-to-v5-rebind-v1"
REBIND_STATUS = "STAGE13_DECISION_CHECKPOINT_V4_TO_V5_REBIND_AUTHORIZED"
COMPLETION_STATUS = "STAGE13_DECISION_CHECKPOINT_V4_TO_V5_REBIND_COMPLETE"
COMPLETION_RECEIPT = "stage13_decision_checkpoint_rebind_receipt.json"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MONTH = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "checkpoint_id",
        "context_sha256",
        "table_name",
        "holding_month",
        "rows",
        "parquet_sha256",
        "logical_content_sha256",
        "arrow_schema_sha256",
        "relative_path",
    }
)


class SourceCheckpointRebindError(RuntimeError):
    """Raised when the pinned checkpoint rebind cannot be authenticated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceCheckpointRebindError(message)


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
        raise SourceCheckpointRebindError(
            "checkpoint rebind metadata is not finite canonical JSON"
        ) from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceCheckpointRebindError(f"{label} is absent or invalid") from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _table_fingerprint(table: pa.Table) -> str:
    source = table.combine_chunks()
    digest = hashlib.sha256(source.schema.serialize().to_pybytes())
    for batch in source.to_batches(max_chunksize=131_072):
        digest.update(batch.serialize().to_pybytes())
    return digest.hexdigest()


def _aggregate_digest(records: Iterable[tuple[str, str]]) -> str:
    """Hash sorted ``path NUL binary-SHA256`` records without ambiguity."""

    normalized = sorted((str(path), str(digest)) for path, digest in records)
    _require(
        len(normalized) == len({path for path, _digest in normalized}),
        "aggregate paths repeat",
    )
    result = hashlib.sha256()
    for path, digest in normalized:
        _require(
            bool(path) and "\x00" not in path and _HEX64.fullmatch(digest) is not None,
            "aggregate record is malformed",
        )
        result.update(path.encode("utf-8"))
        result.update(b"\x00")
        result.update(bytes.fromhex(digest))
    return result.hexdigest()


def _month_range(first: str, last: str) -> tuple[str, ...]:
    _require(
        _MONTH.fullmatch(first) is not None and _MONTH.fullmatch(last) is not None,
        "checkpoint month boundary is malformed",
    )
    _require(first <= last, "checkpoint month boundary is reversed")
    result: list[str] = []
    year, month = (int(value) for value in first.split("-"))
    while True:
        current = f"{year:04d}-{month:02d}"
        result.append(current)
        if current == last:
            break
        month += 1
        if month == 13:
            year += 1
            month = 1
        _require(len(result) <= 1_200, "checkpoint month boundary is unbounded")
    return tuple(result)


def _safe_absolute_child(path: Path, *, root: Path, label: str) -> Path:
    trusted = Path(os.path.abspath(root.expanduser()))
    selected = Path(os.path.abspath(path.expanduser()))
    try:
        relative = selected.relative_to(trusted)
    except ValueError as exc:
        raise SourceCheckpointRebindError(f"{label} escaped its root") from exc
    _require(bool(relative.parts), f"{label} cannot equal its root")
    cursor = trusted
    for component in relative.parts:
        cursor /= component
        if not cursor.exists() and not cursor.is_symlink():
            break
        metadata = cursor.stat(follow_symlinks=False)
        _require(
            stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
            f"{label} path is unsafe",
        )
    return selected


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.rebind-partial")
    if temporary.exists() or temporary.is_symlink():
        metadata = temporary.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
            "checkpoint rebind temporary receipt is unsafe",
        )
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _require(
            not path.exists() and not path.is_symlink(),
            "checkpoint receipt appeared concurrently",
        )
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class _Family:
    name: str
    months: tuple[str, ...]
    rows: int
    arrow_schema_sha256: str


@dataclass(frozen=True, slots=True)
class _Pair:
    family: _Family
    month: str
    payload_relative: str
    receipt_relative: str


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    relative_path: str
    kind: str
    mode: int
    device: int
    inode: int
    links: int
    size_bytes: int
    modified_ns: int
    changed_ns: int
    sha256: str | None

    def mapping(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "kind": self.kind,
            "mode": self.mode,
            "device": self.device,
            "inode": self.inode,
            "links": self.links,
            "size_bytes": self.size_bytes,
            "modified_ns": self.modified_ns,
            "changed_ns": self.changed_ns,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _Authority:
    path: Path
    sha256: str
    document: Mapping[str, object]
    parent_context: Mapping[str, object]
    families: tuple[_Family, ...]
    pairs: tuple[_Pair, ...]
    checkpoint_id: str
    aggregates: Mapping[str, object]
    parent_decision_market_cache: Mapping[str, object]


def _load_authority(path: Path, expected_sha256: str) -> _Authority:
    _require(
        _HEX64.fullmatch(expected_sha256) is not None, "authority SHA-256 is malformed"
    )
    metadata = path.stat(follow_symlinks=False)
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_nlink == 1,
        "checkpoint rebind authority is unsafe",
    )
    observed_sha256 = _sha256_file(path)
    _require(observed_sha256 == expected_sha256, "checkpoint rebind authority differs")
    document = _json(path, label="checkpoint rebind authority")
    parent = document.get("parent_context")
    checkpoint = document.get("checkpoint_contract")
    raw_families = document.get("source_families")
    aggregates = document.get("aggregate_pins")
    transition = document.get("replacement_context_rules")
    safety = document.get("safety")
    authorization = document.get("repair_authorization")
    excluded = document.get("excluded_parent_state")
    parent_cache = document.get("parent_decision_market_cache")
    authority_relative = document.get("authority_path")
    _require(
        document.get("schema_version") == 1
        and document.get("status") == REBIND_STATUS
        and document.get("rebind_id") == REBIND_ID
        and isinstance(parent, Mapping)
        and isinstance(checkpoint, Mapping)
        and isinstance(raw_families, Mapping)
        and isinstance(aggregates, Mapping)
        and isinstance(transition, Mapping)
        and isinstance(safety, Mapping)
        and isinstance(authorization, Mapping)
        and isinstance(excluded, Mapping)
        and isinstance(parent_cache, Mapping)
        and isinstance(authority_relative, str)
        and bool(authority_relative)
        and not Path(authority_relative).is_absolute()
        and ".." not in Path(authority_relative).parts,
        "checkpoint rebind authority structure differs",
    )
    _require(
        authorization.get("source_message_exact")
        == "yes, please fix these types of mistakes beforehand"
        and _sha256_bytes(
            str(authorization.get("source_message_exact")).encode("utf-8")
        )
        == authorization.get("source_message_sha256")
        and authorization.get("source_message_sha256")
        == "70c78b3f5dfe084851ae8072eaeb393f7c84c18c2380f920f4f80759b4013627"
        and authorization.get("stage13_relaunch_authorized") is False,
        "checkpoint rebind repair authorization differs",
    )
    _require(
        transition.get("raw_identity_must_match") is True
        and transition.get("stage12_binding_must_differ") is True
        and transition.get("stage13_approval_must_differ") is True
        and transition.get("fresh_stage13_approval_required") is True
        and transition.get("prior_stage13_approval_valid") is False,
        "checkpoint rebind context-transition rules differ",
    )
    _require(
        safety.get("parent_access_read_only") is True
        and safety.get("parent_tree_identical_before_after") is True
        and safety.get("payload_copy_bytes_exact") is True
        and safety.get("hardlinks_symlinks_aliases_forbidden") is True
        and safety.get("fresh_receipt_changes_only_context_sha256") is True
        and safety.get("idempotent_states")
        == [
            "neither_payload_nor_receipt_copy_then_publish",
            "payload_without_receipt_authenticate_then_publish_receipt",
            "payload_and_receipt_reauthenticate",
            "receipt_without_payload_fail_closed",
        ]
        and safety.get("post_completion_state_verify_only") is True
        and safety.get("completion_receipt_published_last") is True
        and safety.get("stage12_tests_synthetic_only") is True,
        "checkpoint rebind safety contract differs",
    )
    _require(
        excluded.get("final_market_inputs_2025_12_present") is False
        and excluded.get("decision_source_barrier_present") is False
        and excluded.get("outcome_source_checkpoint_count") == 0
        and excluded.get("economic_checkpoint_count") == 0
        and excluded.get("excluded_state_may_not_be_synthesized_by_rebind") is True,
        "checkpoint rebind excluded parent state differs",
    )
    _require(
        set(parent_cache)
        == {
            "reuse_mode",
            "relative_path",
            "typed_relative_path",
            "typed_receipt_sha256",
            "returns_relative_path",
            "return_receipt_sha256",
            "legacy_market_extension_source_sha256",
            "reconciliation_source_sha256",
            "return_adapter_source_sha256",
            "scope",
            "safety",
        }
        and parent_cache.get("reuse_mode")
        == "authenticated_read_only_direct_parent_paths"
        and parent_cache.get("relative_path") == "formula_work/decision_market"
        and parent_cache.get("typed_relative_path") == "typed"
        and parent_cache.get("returns_relative_path") == "returns"
        and all(
            _HEX64.fullmatch(str(parent_cache.get(name))) is not None
            for name in (
                "typed_receipt_sha256",
                "return_receipt_sha256",
                "legacy_market_extension_source_sha256",
                "reconciliation_source_sha256",
                "return_adapter_source_sha256",
            )
        )
        and parent_cache.get("scope")
        == {
            "extension_start": "2019-11-01",
            "last_signal": "2025-11-28",
            "last_exit": "2025-12-31",
            "first_context_signal": "2019-10-31",
        }
        and parent_cache.get("safety")
        == {
            "selected_subtrees_read_only": True,
            "selected_subtrees_reauthenticated_before_reuse": True,
            "selected_subtrees_identical_before_after": True,
            "unselected_siblings_not_opened": True,
            "hardlinks_symlinks_special_nodes_forbidden": True,
            "typed_and_return_builder_semantics_unchanged": True,
            "entry_projection_change_semantically_disjoint": True,
            "replacement_cache_copy_authorized": False,
            "fallback_rebuild_on_authentication_failure": False,
            "interrupted_siblings_may_not_be_opened_or_quarantined": True,
        },
        "parent decision-market cache authority differs",
    )
    checkpoint_id = str(checkpoint.get("checkpoint_id"))
    _require(
        checkpoint.get("receipt_schema_version") == 1
        and checkpoint.get("receipt_status")
        == "FINAL_FORMULA_MONTH_CHECKPOINT_COMPLETE"
        and checkpoint.get("receipt_change_fields") == ["context_sha256"]
        and bool(checkpoint_id),
        "checkpoint receipt contract differs",
    )
    families: list[_Family] = []
    pairs: list[_Pair] = []
    for name, raw in sorted(raw_families.items()):
        _require(
            isinstance(name, str) and isinstance(raw, Mapping),
            "source-family contract differs",
        )
        first = str(raw.get("first_holding_month"))
        last = str(raw.get("last_holding_month"))
        months = _month_range(first, last)
        rows = raw.get("total_rows")
        schema_sha = str(raw.get("arrow_schema_sha256"))
        _require(
            raw.get("checkpoint_count") == len(months)
            and type(rows) is int
            and rows > 0
            and _HEX64.fullmatch(schema_sha) is not None,
            f"source-family identity differs: {name}",
        )
        family = _Family(
            name=name,
            months=months,
            rows=rows,
            arrow_schema_sha256=schema_sha,
        )
        families.append(family)
        for month in months:
            pairs.append(
                _Pair(
                    family=family,
                    month=month,
                    payload_relative=f"months/{name}/{month}.parquet",
                    receipt_relative=f"receipts/{name}/{month}.json",
                )
            )
    _require(bool(families), "checkpoint rebind source-family set is empty")
    _require(
        aggregates.get("checkpoint_count") == len(pairs)
        and aggregates.get("receipt_count") == len(pairs)
        and aggregates.get("total_rows") == sum(family.rows for family in families)
        and type(aggregates.get("total_payload_bytes")) is int
        and int(aggregates["total_payload_bytes"]) > 0
        and type(aggregates.get("total_receipt_bytes")) is int
        and int(aggregates["total_receipt_bytes"]) > 0
        and aggregates.get("aggregate_method")
        == "sha256_sorted_relative_path_nul_binary_digest_v1"
        and all(
            _HEX64.fullmatch(str(aggregates.get(name))) is not None
            for name in (
                "receipt_file_set_sha256",
                "payload_content_set_sha256",
                "logical_content_set_sha256",
            )
        ),
        "checkpoint rebind aggregate contract differs",
    )
    return _Authority(
        path=path,
        sha256=observed_sha256,
        document=MappingProxyType(document),
        parent_context=MappingProxyType(dict(parent)),
        families=tuple(families),
        pairs=tuple(pairs),
        checkpoint_id=checkpoint_id,
        aggregates=MappingProxyType(dict(aggregates)),
        parent_decision_market_cache=MappingProxyType(dict(parent_cache)),
    )


def _expected_scope_paths(pairs: Sequence[_Pair]) -> tuple[set[str], set[str]]:
    files = {
        relative
        for pair in pairs
        for relative in (pair.payload_relative, pair.receipt_relative)
    }
    directories = {"months", "receipts"}
    for relative in files:
        path = Path(relative)
        for parent in path.parents:
            if parent == Path("."):
                break
            directories.add(parent.as_posix())
    return files, directories


def _closed_parent_snapshot(
    root: Path, pairs: Sequence[_Pair]
) -> tuple[_FileIdentity, ...]:
    expected_files, expected_directories = _expected_scope_paths(pairs)
    observed: list[_FileIdentity] = []

    def visit(path: Path, relative: str) -> None:
        metadata = path.stat(follow_symlinks=False)
        _require(
            not stat.S_ISLNK(metadata.st_mode),
            "parent checkpoint scope contains a symlink",
        )
        if stat.S_ISDIR(metadata.st_mode):
            _require(
                relative in expected_directories,
                "parent checkpoint scope contains an extra directory",
            )
            observed.append(
                _FileIdentity(
                    relative_path=relative,
                    kind="directory",
                    mode=stat.S_IMODE(metadata.st_mode),
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                    links=metadata.st_nlink,
                    size_bytes=metadata.st_size,
                    modified_ns=metadata.st_mtime_ns,
                    changed_ns=metadata.st_ctime_ns,
                    sha256=None,
                )
            )
            for child in sorted(path.iterdir(), key=lambda value: value.name):
                child_relative = f"{relative}/{child.name}" if relative else child.name
                visit(child, child_relative)
            return
        _require(
            stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
            "parent checkpoint scope contains a special or aliased file",
        )
        _require(
            relative in expected_files, "parent checkpoint scope contains an extra file"
        )
        observed.append(
            _FileIdentity(
                relative_path=relative,
                kind="file",
                mode=stat.S_IMODE(metadata.st_mode),
                device=metadata.st_dev,
                inode=metadata.st_ino,
                links=metadata.st_nlink,
                size_bytes=metadata.st_size,
                modified_ns=metadata.st_mtime_ns,
                changed_ns=metadata.st_ctime_ns,
                sha256=_sha256_file(path),
            )
        )

    for top in ("months", "receipts"):
        path = root / top
        _require(
            path.exists() and not path.is_symlink(),
            f"parent checkpoint {top} root is absent",
        )
        visit(path, top)
    observed_files = {
        record.relative_path for record in observed if record.kind == "file"
    }
    observed_directories = {
        record.relative_path for record in observed if record.kind == "directory"
    }
    _require(
        observed_files == expected_files
        and observed_directories == expected_directories,
        "parent checkpoint scope is not the exact closed set",
    )
    return tuple(sorted(observed, key=lambda value: value.relative_path))


def _snapshot_sha256(snapshot: Sequence[_FileIdentity]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes([record.mapping() for record in snapshot])
    )


def _scan_replacement_topology(
    root: Path,
    *,
    pairs: Sequence[_Pair],
    completion_exists: bool,
) -> None:
    expected_files, expected_directories = _expected_scope_paths(pairs)
    for top in ("months", "receipts"):
        top_path = root / top
        _require(
            top_path.is_dir() and not top_path.is_symlink(),
            "replacement checkpoint root is unsafe",
        )
        for directory, directory_names, file_names in os.walk(
            top_path, followlinks=False
        ):
            current = Path(directory)
            current_metadata = current.stat(follow_symlinks=False)
            _require(
                stat.S_ISDIR(current_metadata.st_mode)
                and not stat.S_ISLNK(current_metadata.st_mode),
                "replacement checkpoint directory is unsafe",
            )
            for name in tuple(directory_names):
                child = current / name
                metadata = child.stat(follow_symlinks=False)
                _require(
                    stat.S_ISDIR(metadata.st_mode)
                    and not stat.S_ISLNK(metadata.st_mode),
                    "replacement checkpoint scope contains an unsafe directory",
                )
                relative = child.relative_to(root).as_posix()
                if not completion_exists:
                    _require(
                        relative in expected_directories,
                        "replacement checkpoint scope contains an extra directory",
                    )
            for name in file_names:
                child = current / name
                metadata = child.stat(follow_symlinks=False)
                _require(
                    stat.S_ISREG(metadata.st_mode)
                    and not stat.S_ISLNK(metadata.st_mode)
                    and metadata.st_nlink == 1,
                    "replacement checkpoint scope contains a special or aliased file",
                )
                relative = child.relative_to(root).as_posix()
                if not completion_exists:
                    _require(
                        relative in expected_files,
                        "replacement checkpoint scope contains an extra file",
                    )


def _rebind_partial_paths(
    root: Path,
    *,
    pairs: Sequence[_Pair],
) -> tuple[Path, ...]:
    result = [root / f".{COMPLETION_RECEIPT}.rebind-partial"]
    for pair in pairs:
        payload = root / pair.payload_relative
        receipt = root / pair.receipt_relative
        result.extend(
            (
                payload.with_name(f".{payload.name}.rebind-partial"),
                receipt.with_name(f".{receipt.name}.rebind-partial"),
            )
        )
    return tuple(result)


def _recover_or_reject_stale_partials(
    root: Path,
    *,
    pairs: Sequence[_Pair],
    completion_exists: bool,
) -> None:
    for path in _rebind_partial_paths(root, pairs=pairs):
        if not path.exists() and not path.is_symlink():
            continue
        metadata = path.stat(follow_symlinks=False)
        _require(
            not completion_exists
            and stat.S_ISREG(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and metadata.st_nlink == 1,
            "checkpoint rebind temporary state is unsafe",
        )
        path.unlink()
        _fsync_directory(path.parent)


def _context_document(path: Path, *, label: str) -> dict[str, object]:
    metadata = path.stat(follow_symlinks=False)
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_nlink == 1,
        f"{label} is unsafe",
    )
    document = _json(path, label=label)
    _require(
        path.read_bytes() == _canonical_json_bytes(document),
        f"{label} is not canonical JSON",
    )
    asserted = document.get("context_sha256")
    body = dict(document)
    body.pop("context_sha256", None)
    _require(
        _HEX64.fullmatch(str(asserted)) is not None
        and _sha256_bytes(_canonical_json_bytes(body)) == asserted,
        f"{label} self-identity differs",
    )
    return document


def _authenticate_contexts(
    *,
    namespace_root: Path,
    replacement_context_document: Mapping[str, object],
    authority: _Authority,
) -> tuple[Path, Path, Mapping[str, object], Mapping[str, object]]:
    parent_pin = authority.parent_context
    parent_sha = str(parent_pin.get("context_sha256"))
    replacement_sha = str(replacement_context_document.get("context_sha256"))
    _require(
        _HEX64.fullmatch(parent_sha) is not None
        and _HEX64.fullmatch(replacement_sha) is not None
        and replacement_sha != parent_sha,
        "checkpoint rebind context identity differs",
    )
    parent_root = _safe_absolute_child(
        namespace_root / parent_sha,
        root=namespace_root,
        label="parent replay context",
    )
    replacement_root = _safe_absolute_child(
        namespace_root / replacement_sha,
        root=namespace_root,
        label="replacement replay context",
    )
    _require(
        parent_root.is_dir() and not parent_root.is_symlink(),
        "parent replay context is unsafe",
    )
    _require(
        replacement_root.is_dir() and not replacement_root.is_symlink(),
        "replacement replay context is unsafe",
    )
    parent_path = parent_root / str(parent_pin.get("context_json_relative_path"))
    parent_document = _context_document(
        parent_path, label="parent replay context document"
    )
    replacement_path = replacement_root / "context.json"
    replacement_document = _context_document(
        replacement_path,
        label="replacement replay context document",
    )
    _require(
        _sha256_file(parent_path) == parent_pin.get("context_json_sha256")
        and parent_document.get("context_sha256") == parent_sha
        and all(
            parent_document.get(key) == parent_pin.get(key)
            for key in (
                "raw_snapshot_id",
                "raw_file_identities_sha256",
                "stage12_binding_sha256",
                "approval_payload_sha256",
            )
        ),
        "parent replay context document differs",
    )
    supplied = dict(replacement_context_document)
    _require(
        replacement_document == supplied
        and replacement_document.get("context_sha256") == replacement_sha
        and replacement_document.get("raw_snapshot_id")
        == parent_document.get("raw_snapshot_id")
        and replacement_document.get("raw_file_identities_sha256")
        == parent_document.get("raw_file_identities_sha256")
        and replacement_document.get("stage12_binding_sha256")
        != parent_document.get("stage12_binding_sha256")
        and replacement_document.get("approval_payload_sha256")
        != parent_document.get("approval_payload_sha256"),
        "replacement replay context is not an authorized fresh transition",
    )
    return (
        parent_root,
        replacement_root,
        MappingProxyType(parent_document),
        MappingProxyType(replacement_document),
    )


def _validate_receipt(
    path: Path,
    *,
    pair: _Pair,
    context_sha256: str,
    checkpoint_id: str,
) -> dict[str, object]:
    metadata = path.stat(follow_symlinks=False)
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_nlink == 1,
        f"checkpoint receipt is unsafe: {pair.family.name}/{pair.month}",
    )
    receipt = _json(path, label=f"checkpoint receipt {pair.family.name}/{pair.month}")
    _require(
        path.read_bytes() == _canonical_json_bytes(receipt),
        f"checkpoint receipt is not canonical: {pair.family.name}/{pair.month}",
    )
    _require(set(receipt) == _RECEIPT_KEYS, "checkpoint receipt fields differ")
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("status") == "FINAL_FORMULA_MONTH_CHECKPOINT_COMPLETE"
        and receipt.get("checkpoint_id") == checkpoint_id
        and receipt.get("context_sha256") == context_sha256
        and receipt.get("table_name") == pair.family.name
        and receipt.get("holding_month") == pair.month
        and receipt.get("relative_path") == pair.payload_relative
        and type(receipt.get("rows")) is int
        and int(receipt["rows"]) > 0
        and receipt.get("arrow_schema_sha256") == pair.family.arrow_schema_sha256
        and all(
            _HEX64.fullmatch(str(receipt.get(key))) is not None
            for key in (
                "parquet_sha256",
                "logical_content_sha256",
                "arrow_schema_sha256",
            )
        ),
        f"checkpoint receipt identity differs: {pair.family.name}/{pair.month}",
    )
    return receipt


def _validate_payload(path: Path, *, pair: _Pair, receipt: Mapping[str, object]) -> int:
    metadata = path.stat(follow_symlinks=False)
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_nlink == 1,
        f"checkpoint payload is unsafe: {pair.family.name}/{pair.month}",
    )
    _require(
        _sha256_file(path) == receipt.get("parquet_sha256"),
        f"checkpoint Parquet hash differs: {pair.family.name}/{pair.month}",
    )
    try:
        table = pq.read_table(path, use_threads=False).combine_chunks()
    except Exception as exc:
        raise SourceCheckpointRebindError(
            f"checkpoint Parquet is invalid: {pair.family.name}/{pair.month}"
        ) from exc
    schema_sha = _sha256_bytes(table.schema.serialize().to_pybytes())
    _require(
        table.num_rows == receipt.get("rows")
        and schema_sha == receipt.get("arrow_schema_sha256")
        and _table_fingerprint(table) == receipt.get("logical_content_sha256"),
        f"checkpoint logical identity differs: {pair.family.name}/{pair.month}",
    )
    month_name = (
        "source_holding_month"
        if "source_holding_month" in table.column_names
        else "holding_month" if "holding_month" in table.column_names else None
    )
    _require(month_name is not None, "checkpoint payload lacks its holding-month key")
    months = table[month_name]
    _require(
        months.null_count == 0
        and pc.all(pc.equal(months, pa.scalar(pair.month, months.type))).as_py()
        is True,
        f"checkpoint month domain differs: {pair.family.name}/{pair.month}",
    )
    return metadata.st_size


def _authenticate_parent_pairs(
    root: Path,
    *,
    authority: _Authority,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    receipts: list[dict[str, object]] = []
    receipt_records: list[tuple[str, str]] = []
    payload_records: list[tuple[str, str]] = []
    logical_records: list[tuple[str, str]] = []
    rows_by_family = {family.name: 0 for family in authority.families}
    total_payload_bytes = 0
    total_receipt_bytes = 0
    context_sha = str(authority.parent_context["context_sha256"])
    for pair in authority.pairs:
        receipt_path = root / pair.receipt_relative
        payload_path = root / pair.payload_relative
        receipt = _validate_receipt(
            receipt_path,
            pair=pair,
            context_sha256=context_sha,
            checkpoint_id=authority.checkpoint_id,
        )
        total_payload_bytes += _validate_payload(
            payload_path,
            pair=pair,
            receipt=receipt,
        )
        total_receipt_bytes += receipt_path.stat(follow_symlinks=False).st_size
        rows_by_family[pair.family.name] += int(receipt["rows"])
        receipt_records.append((pair.receipt_relative, _sha256_file(receipt_path)))
        payload_records.append((pair.payload_relative, str(receipt["parquet_sha256"])))
        logical_records.append(
            (pair.payload_relative, str(receipt["logical_content_sha256"]))
        )
        receipts.append(receipt)
    _require(
        all(
            rows_by_family[family.name] == family.rows for family in authority.families
        ),
        "checkpoint family row totals differ",
    )
    observed = {
        "checkpoint_count": len(receipts),
        "receipt_count": len(receipts),
        "total_rows": sum(rows_by_family.values()),
        "total_payload_bytes": total_payload_bytes,
        "total_receipt_bytes": total_receipt_bytes,
        "receipt_file_set_sha256": _aggregate_digest(receipt_records),
        "payload_content_set_sha256": _aggregate_digest(payload_records),
        "logical_content_set_sha256": _aggregate_digest(logical_records),
        "aggregate_method": "sha256_sorted_relative_path_nul_binary_digest_v1",
    }
    _require(observed == dict(authority.aggregates), "checkpoint aggregate pins differ")
    return tuple(receipts), observed


def _copy_authenticated_payload(
    source: Path,
    destination: Path,
    *,
    pair: _Pair,
    receipt: Mapping[str, object],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.rebind-partial")
    if temporary.exists() or temporary.is_symlink():
        metadata = temporary.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 1
            and not destination.exists()
            and not destination.is_symlink(),
            "checkpoint rebind temporary payload is unsafe",
        )
        temporary.unlink()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(source, flags)
    try:
        source_metadata = os.fstat(source_descriptor)
        _require(
            stat.S_ISREG(source_metadata.st_mode) and source_metadata.st_nlink == 1,
            "checkpoint source payload changed topology",
        )
        try:
            with temporary.open("xb") as output, os.fdopen(
                os.dup(source_descriptor), "rb"
            ) as input_handle:
                digest = hashlib.sha256()
                while block := input_handle.read(8 << 20):
                    output.write(block)
                    digest.update(block)
                output.flush()
                os.fsync(output.fileno())
            _require(
                digest.hexdigest() == receipt.get("parquet_sha256"),
                f"checkpoint source changed during copy: {pair.family.name}/{pair.month}",
            )
            _validate_payload(temporary, pair=pair, receipt=receipt)
            _require(
                not destination.exists() and not destination.is_symlink(),
                "replacement checkpoint payload appeared concurrently",
            )
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
    finally:
        os.close(source_descriptor)


def _fresh_receipt(
    parent_receipt: Mapping[str, object],
    *,
    replacement_context_sha256: str,
) -> dict[str, object]:
    fresh = dict(parent_receipt)
    fresh["context_sha256"] = replacement_context_sha256
    changed = {key for key in fresh if fresh.get(key) != parent_receipt.get(key)}
    _require(
        changed == {"context_sha256"}, "checkpoint rebind changed a scientific field"
    )
    return fresh


def _authenticate_or_publish_pair(
    *,
    parent_root: Path,
    replacement_root: Path,
    pair: _Pair,
    parent_receipt: Mapping[str, object],
    replacement_context_sha256: str,
    checkpoint_id: str,
    publication_permitted: bool,
) -> dict[str, object]:
    source_payload = parent_root / pair.payload_relative
    target_payload = replacement_root / pair.payload_relative
    target_receipt_path = replacement_root / pair.receipt_relative
    payload_present = target_payload.exists() or target_payload.is_symlink()
    receipt_present = target_receipt_path.exists() or target_receipt_path.is_symlink()
    _require(
        payload_present or not receipt_present,
        f"replacement receipt exists without payload: {pair.family.name}/{pair.month}",
    )
    if not publication_permitted:
        _require(
            payload_present and receipt_present,
            (
                "completed checkpoint rebind pair is incomplete: "
                f"{pair.family.name}/{pair.month}"
            ),
        )
    fresh = _fresh_receipt(
        parent_receipt,
        replacement_context_sha256=replacement_context_sha256,
    )
    if payload_present:
        _validate_payload(target_payload, pair=pair, receipt=fresh)
        source_metadata = source_payload.stat(follow_symlinks=False)
        target_metadata = target_payload.stat(follow_symlinks=False)
        _require(
            source_metadata.st_nlink == target_metadata.st_nlink == 1
            and (source_metadata.st_dev, source_metadata.st_ino)
            != (target_metadata.st_dev, target_metadata.st_ino),
            "replacement checkpoint payload aliases its parent",
        )
    else:
        _copy_authenticated_payload(
            source_payload,
            target_payload,
            pair=pair,
            receipt=fresh,
        )
    if receipt_present:
        observed = _validate_receipt(
            target_receipt_path,
            pair=pair,
            context_sha256=replacement_context_sha256,
            checkpoint_id=checkpoint_id,
        )
        _require(observed == fresh, "replacement checkpoint receipt differs")
    else:
        target_receipt_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(target_receipt_path, _canonical_json_bytes(fresh))
        observed = _validate_receipt(
            target_receipt_path,
            pair=pair,
            context_sha256=replacement_context_sha256,
            checkpoint_id=checkpoint_id,
        )
        _require(
            observed == fresh, "replacement checkpoint receipt publication differs"
        )
    return fresh


def rebind_stage13_decision_checkpoints(
    *,
    namespace_root: Path,
    replacement_context_document: Mapping[str, object],
    authority_path: Path,
    expected_authority_sha256: str,
) -> Mapping[str, object]:
    """Authenticate and rebind the one authority-pinned checkpoint set.

    Integration point: call after the replacement replay store has created and
    authenticated ``context.json`` plus empty ``months``/``receipts`` roots,
    and before asking that store for pending source months.  A retry may call
    this again: ``payload/no receipt`` is completed, ``payload+receipt`` is
    reauthenticated, and ``receipt/no payload`` fails closed.  Once the
    completion receipt exists, later ordinary Stage 13 checkpoints may coexist
    with the rebound subset.
    """

    namespace = Path(os.path.abspath(namespace_root.expanduser()))
    namespace_metadata = namespace.stat(follow_symlinks=False)
    _require(
        stat.S_ISDIR(namespace_metadata.st_mode)
        and not stat.S_ISLNK(namespace_metadata.st_mode),
        "Stage 13 replay namespace is unsafe",
    )
    authority = _load_authority(authority_path, expected_authority_sha256)
    (
        parent_root,
        replacement_root,
        parent_context,
        replacement_context,
    ) = _authenticate_contexts(
        namespace_root=namespace,
        replacement_context_document=replacement_context_document,
        authority=authority,
    )
    completion_path = replacement_root / COMPLETION_RECEIPT
    _require(
        not completion_path.is_symlink(),
        "checkpoint rebind completion receipt is symlinked",
    )
    completion_exists = completion_path.exists()
    if completion_exists:
        completion_metadata = completion_path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(completion_metadata.st_mode)
            and completion_metadata.st_nlink == 1,
            "checkpoint rebind completion receipt is unsafe",
        )
        early_completion = _json(
            completion_path,
            label="checkpoint rebind completion receipt",
        )
        _require(
            completion_path.read_bytes() == _canonical_json_bytes(early_completion)
            and early_completion.get("schema_version") == 1
            and early_completion.get("status") == COMPLETION_STATUS
            and early_completion.get("rebind_id") == REBIND_ID
            and early_completion.get("authority_sha256") == authority.sha256
            and early_completion.get("parent_context_sha256")
            == authority.parent_context.get("context_sha256")
            and early_completion.get("stage13_relaunch_authorized") is False
            and early_completion.get("completion_receipt_published_last") is True,
            "checkpoint rebind completion receipt differs",
        )
    _require(
        not (parent_root / "DECISION_SOURCES_COMMITTED.json").exists()
        and not (parent_root / "DECISION_SOURCES_COMMITTED.json").is_symlink(),
        "parent decision-source barrier exists",
    )
    _require(
        completion_exists
        or (
            not (replacement_root / "DECISION_SOURCES_COMMITTED.json").exists()
            and not (replacement_root / "DECISION_SOURCES_COMMITTED.json").is_symlink()
        ),
        "replacement decision-source barrier predates checkpoint rebind",
    )
    before = _closed_parent_snapshot(parent_root, authority.pairs)
    parent_receipts, aggregates = _authenticate_parent_pairs(
        parent_root,
        authority=authority,
    )
    _recover_or_reject_stale_partials(
        replacement_root,
        pairs=authority.pairs,
        completion_exists=completion_exists,
    )
    _scan_replacement_topology(
        replacement_root,
        pairs=authority.pairs,
        completion_exists=completion_exists,
    )
    if completion_exists:
        for pair in authority.pairs:
            _require(
                (replacement_root / pair.payload_relative).is_file()
                and not (replacement_root / pair.payload_relative).is_symlink()
                and (replacement_root / pair.receipt_relative).is_file()
                and not (replacement_root / pair.receipt_relative).is_symlink(),
                (
                    "completed checkpoint rebind pair is incomplete: "
                    f"{pair.family.name}/{pair.month}"
                ),
            )
    fresh_receipts: list[dict[str, object]] = []
    for pair, parent_receipt in zip(authority.pairs, parent_receipts, strict=True):
        fresh_receipts.append(
            _authenticate_or_publish_pair(
                parent_root=parent_root,
                replacement_root=replacement_root,
                pair=pair,
                parent_receipt=parent_receipt,
                replacement_context_sha256=str(replacement_context["context_sha256"]),
                checkpoint_id=authority.checkpoint_id,
                publication_permitted=not completion_exists,
            )
        )
    _scan_replacement_topology(
        replacement_root,
        pairs=authority.pairs,
        completion_exists=True,
    )
    after = _closed_parent_snapshot(parent_root, authority.pairs)
    _require(before == after, "parent checkpoint tree changed during rebind")
    destination_receipt_records = [
        (
            pair.receipt_relative,
            _sha256_file(replacement_root / pair.receipt_relative),
        )
        for pair in authority.pairs
    ]
    family_summary = {
        family.name: {
            "checkpoint_count": len(family.months),
            "first_holding_month": family.months[0],
            "last_holding_month": family.months[-1],
            "total_rows": family.rows,
            "arrow_schema_sha256": family.arrow_schema_sha256,
        }
        for family in authority.families
    }
    completion: dict[str, object] = {
        "schema_version": 1,
        "status": COMPLETION_STATUS,
        "rebind_id": REBIND_ID,
        "authority_relative_path": authority.document["authority_path"],
        "authority_sha256": authority.sha256,
        "parent_context_sha256": parent_context["context_sha256"],
        "replacement_context_sha256": replacement_context["context_sha256"],
        "checkpoint_id": authority.checkpoint_id,
        "source_aggregates": aggregates,
        "destination_receipt_file_set_sha256": _aggregate_digest(
            destination_receipt_records
        ),
        "families": family_summary,
        "parent_decision_market_cache_authority": dict(
            authority.parent_decision_market_cache
        ),
        "parent_tree_snapshot_sha256": _snapshot_sha256(before),
        "parent_tree_identical_before_after": True,
        "payload_bytes_copied_exactly": True,
        "receipt_change_fields": ["context_sha256"],
        "hardlinks_symlinks_aliases_created": False,
        "completion_receipt_published_last": True,
        "stage13_relaunch_authorized": False,
    }
    if completion_exists:
        observed = _json(completion_path, label="checkpoint rebind completion receipt")
        _require(
            completion_path.read_bytes() == _canonical_json_bytes(observed)
            and observed == completion,
            "checkpoint rebind completion receipt differs",
        )
    else:
        _atomic_bytes(completion_path, _canonical_json_bytes(completion))
        observed = _json(completion_path, label="checkpoint rebind completion receipt")
        _require(
            observed == completion, "checkpoint rebind completion publication differs"
        )
    return MappingProxyType(completion)


__all__ = [
    "COMPLETION_RECEIPT",
    "COMPLETION_STATUS",
    "REBIND_ID",
    "REBIND_STATUS",
    "SourceCheckpointRebindError",
    "rebind_stage13_decision_checkpoints",
]
