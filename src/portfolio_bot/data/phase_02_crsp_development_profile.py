"""Profile CRSP development-slice duplicates and categorical domains.

The scanner is deliberately narrow.  It reads the complete daily and monthly
CSV streams because rows are interleaved by security, but it evaluates the
date column first and never exposes an out-of-scope non-date value to Python.
Within the retained 2000--2014 slice, key adjacency and categorical counts are
computed with Arrow kernels.  Full 36-field Python tuples are materialized only
for duplicate edges and monthly effective-interval candidate groups.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import platform
import shutil
import tempfile
import time
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_CONFIG = (
    PROJECT_ROOT / "configs" / "phase_02_canonicalization" / "crsp_development_v1.toml"
)
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.toml"
DEFAULT_PHASE_CONFIG = PROJECT_ROOT / "configs" / "phase_02_canonical_panel.toml"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "phase_02_canonical_panel"
BYTE_CHUNK_SIZE = 8 << 20
LOCKED_PHASE_CONFIG_SHA256 = (
    "fb727b10375d5eaf5a8e72629934daeb2f5790c7b09a46287df08c972db00adb"
)
LOCKED_PHASE_01_FREEZE_MANIFEST_SHA256 = (
    "fc2478d15eba609731c8374b761ba77a4399d346befcc596abddd5d1ac1ff7ac"
)
LOCKED_PHASE_01_FREEZE_RECHECK_SHA256 = (
    "ee6128b6e3ee256c60cb38de50e7bf093d227b2070a545613fcfac543c91f59b"
)


class ProfileBuildError(RuntimeError):
    """Raised when an authenticated input, scan, gate, or publication fails."""


@dataclasses.dataclass(frozen=True)
class ProfileSettings:
    project_root: Path
    profile_config_path: Path
    paths_config_path: Path
    phase_config_path: Path
    config: dict[str, Any]
    database_root: Path
    raw_root: Path
    quality_root: Path
    artifact_root: Path
    run_root: Path
    freeze_manifest_path: Path
    freeze_recheck_path: Path
    block_size_bytes: int = 16 << 20


def canonical_json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json_bytes(value: Any) -> bytes:
    return (canonical_json_text(value) + "\n").encode("utf-8")


def sha256_file(path: Path, chunk_size: int = BYTE_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def lexical_absolute(path: str | Path) -> Path:
    """Normalize an absolute path without erasing symlink ancestry evidence."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def project_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileBuildError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProfileBuildError(f"Expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _wire(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return canonical_json_text(value)
    return str(value)


def _write_csv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            delimiter=",",
            quotechar='"',
            doublequote=True,
            escapechar=None,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _wire(row.get(field)) for field in fields})


def operational_stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
    }


def load_settings(
    profile_config_path: Path = DEFAULT_PROFILE_CONFIG,
    paths_config_path: Path = DEFAULT_PATHS_CONFIG,
    phase_config_path: Path = DEFAULT_PHASE_CONFIG,
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    run_root: Path = DEFAULT_RUN_ROOT,
    block_size_bytes: int = 16 << 20,
) -> ProfileSettings:
    profile_config_path = lexical_absolute(profile_config_path)
    paths_config_path = lexical_absolute(paths_config_path)
    phase_config_path = lexical_absolute(phase_config_path)
    with profile_config_path.open("rb") as handle:
        config = tomllib.load(handle)
    with paths_config_path.open("rb") as handle:
        paths = tomllib.load(handle)
    with phase_config_path.open("rb") as handle:
        phase = tomllib.load(handle)
    configured = Path(paths["database"]["root"]).expanduser()
    database_root = lexical_absolute(
        Path(os.environ.get("PORTFOLIO_DATABASE_ROOT", configured))
    )
    namespace = phase["storage"]["quality_namespace"]
    freeze_manifest = project_path(phase["phase_01_freeze_manifest"])
    return ProfileSettings(
        project_root=PROJECT_ROOT,
        profile_config_path=profile_config_path,
        paths_config_path=paths_config_path,
        phase_config_path=phase_config_path,
        config=config,
        database_root=database_root,
        raw_root=lexical_absolute(database_root / paths["database"]["raw"]),
        quality_root=lexical_absolute(
            database_root / paths["database"]["quality"] / namespace
        ),
        artifact_root=lexical_absolute(artifact_root),
        run_root=lexical_absolute(run_root),
        freeze_manifest_path=freeze_manifest,
        freeze_recheck_path=freeze_manifest.parent / "raw_freeze_recheck_metadata.json",
        block_size_bytes=block_size_bytes,
    )


def _assert_exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    actual = sorted(value)
    wanted = sorted(expected)
    if actual != wanted:
        raise ProfileBuildError(f"{label} keys differ: expected {wanted}, actual {actual}")


def _read_hash_inventory(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, name = line.split("  ", 1)
        except ValueError as exc:
            raise ProfileBuildError(f"Malformed SHA-256 inventory at {path}:{line_number}") from exc
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ProfileBuildError(f"Malformed SHA-256 digest at {path}:{line_number}")
        if name in result:
            raise ProfileBuildError(f"Duplicate inventory path {name!r} in {path}")
        result[name] = digest
    return result


def _authenticate_contract(settings: ProfileSettings) -> dict[str, Any]:
    locked = settings.config["upstream"]["contract"]
    pointer_path = project_path(locked["verified_pointer"])
    if sha256_file(pointer_path) != locked["verified_pointer_sha256"]:
        raise ProfileBuildError("Verified contract pointer SHA-256 differs from locked config")
    pointer = _read_json(pointer_path)
    if pointer.get("status") != "verified_passed":
        raise ProfileBuildError("Verified contract pointer is not verified_passed")
    if pointer.get("contract_id") != locked["contract_id"]:
        raise ProfileBuildError("Verified contract ID differs from locked config")
    if pointer.get("verification_id") != locked["verification_id"]:
        raise ProfileBuildError("Verified contract verification ID differs from locked config")
    contract_dir = project_path(pointer["contract_path"])
    verification_dir = project_path(pointer["verification_path"])
    fixed = {
        contract_dir / "contract.json": pointer["contract_json_sha256"],
        contract_dir / "artifact_hashes.sha256": pointer["artifact_hashes_sha256"],
        verification_dir / "independent_check.json": pointer["independent_check_sha256"],
        verification_dir / "differences.csv": pointer["differences_sha256"],
    }
    for path, expected in fixed.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ProfileBuildError(f"Authenticated contract artifact differs: {path}")
    inventory = _read_hash_inventory(contract_dir / "artifact_hashes.sha256")
    if not inventory:
        raise ProfileBuildError("Verified contract artifact inventory is empty")
    for name, digest in inventory.items():
        path = contract_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ProfileBuildError(f"Contract inventory mismatch: {path}")
    independent = _read_json(verification_dir / "independent_check.json")
    if independent.get("contract_id") != locked["contract_id"]:
        raise ProfileBuildError("Contract independent-check ID mismatch")
    if independent.get("verification_id") != locked["verification_id"]:
        raise ProfileBuildError("Contract independent-check verification ID mismatch")
    if independent.get("status") != "verified_passed":
        raise ProfileBuildError("Contract independent check is not verified_passed")
    summary = independent.get("summary", {})
    if summary.get("fail_count") != 0 or summary.get("difference_count") != 0:
        raise ProfileBuildError("Contract independent check contains failures or differences")
    if independent.get("independence", {}).get("raw_csv_rows_read") != 0:
        raise ProfileBuildError("Contract-only verifier unexpectedly read raw rows")
    source_rows = {row["filename"]: row for row in _read_csv(contract_dir / "source_files.csv")}
    for source_name in ("daily", "monthly"):
        source = settings.config["sources"][source_name]
        row = source_rows.get(source["filename"])
        expected = {
            "sha256": source["sha256"],
            "size_bytes": str(source["size_bytes"]),
            "parsed_data_rows": str(source["full_file_rows"]),
            "observed_column_count": str(len(source["header"])),
        }
        if row is None or any(row.get(key) != value for key, value in expected.items()):
            raise ProfileBuildError(f"{source_name} source identity differs from verified contract")
    return {
        "pointer_path": pointer_path,
        "pointer_sha256": locked["verified_pointer_sha256"],
        "pointer": pointer,
        "contract_dir": contract_dir,
        "source_rows": source_rows,
    }


def _authenticate_calendar(settings: ProfileSettings) -> dict[str, Any]:
    locked = settings.config["upstream"]["calendar"]
    pointer_path = project_path(locked["verified_pointer"])
    if sha256_file(pointer_path) != locked["verified_pointer_sha256"]:
        raise ProfileBuildError("Verified calendar pointer SHA-256 differs from locked config")
    pointer = _read_json(pointer_path)
    for key in ("build_id", "verification_id", "status"):
        expected = locked["required_status"] if key == "status" else locked[key]
        if pointer.get(key) != expected:
            raise ProfileBuildError(f"Verified calendar {key} differs from locked config")
    if pointer.get("contract_id") != settings.config["upstream"]["contract"]["contract_id"]:
        raise ProfileBuildError("Calendar is bound to a different contract")
    if pointer.get("independent_check_sha256") != locked["independent_check_sha256"]:
        raise ProfileBuildError("Calendar independent-check hash differs from locked config")
    manifest_path = project_path(pointer["primary_artifact_path"]) / "manifest.json"
    verification_dir = project_path(pointer["verification_path"])
    fixed = {
        manifest_path: pointer["primary_manifest_sha256"],
        verification_dir / "independent_check.json": pointer["independent_check_sha256"],
        verification_dir / "differences.csv": pointer["differences_sha256"],
    }
    for path, expected in fixed.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ProfileBuildError(f"Authenticated calendar artifact differs: {path}")
    independent = _read_json(verification_dir / "independent_check.json")
    if independent.get("build_id") != locked["build_id"]:
        raise ProfileBuildError("Calendar independent-check build ID mismatch")
    if independent.get("verification_id") != locked["verification_id"]:
        raise ProfileBuildError("Calendar independent-check verification ID mismatch")
    if independent.get("status") != locked["required_status"]:
        raise ProfileBuildError("Calendar independent check is not verified_passed")
    summary = independent.get("summary", {})
    if summary.get("fail_count") != 0 or summary.get("difference_count") != 0:
        raise ProfileBuildError("Calendar independent check contains failures or differences")
    independence = independent.get("independence", {})
    if independence.get("raw_columns_requested") != ["DlyCalDt"]:
        raise ProfileBuildError("Calendar independent check read unexpected raw columns")
    manifest = _read_json(manifest_path)
    if manifest.get("build_id") != locked["build_id"]:
        raise ProfileBuildError("Calendar primary manifest build ID mismatch")
    for item in manifest.get("outputs", []):
        path = Path(item["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(item["size_bytes"])
            or sha256_file(path) != item["sha256"]
        ):
            raise ProfileBuildError(f"Calendar primary output mismatch: {path}")
    return {
        "pointer_path": pointer_path,
        "pointer_sha256": locked["verified_pointer_sha256"],
        "pointer": pointer,
        "manifest": manifest,
        "independent": independent,
    }


def authenticate_upstreams(settings: ProfileSettings) -> dict[str, Any]:
    """Authenticate both immutable upstream pointers and their sealed evidence."""
    return {
        "contract": _authenticate_contract(settings),
        "calendar": _authenticate_calendar(settings),
    }


def _locked_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return lexical_absolute(path if path.is_absolute() else project_root / path)


def _read_authenticated_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    expected_size_bytes: int | None = None,
) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProfileBuildError(f"Cannot read authenticated {label}: {path}") from exc
    if expected_size_bytes is not None and len(payload) != expected_size_bytes:
        raise ProfileBuildError(f"Authenticated {label} size differs from its sealed evidence")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ProfileBuildError(f"Authenticated {label} bytes differ from their sealed SHA-256")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProfileBuildError(f"Authenticated {label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProfileBuildError(f"Authenticated {label} is not a JSON object")
    return value


def frozen_raw_identities(settings: ProfileSettings) -> dict[str, dict[str, int]]:
    try:
        phase_payload = settings.phase_config_path.read_bytes()
    except OSError as exc:
        raise ProfileBuildError("Cannot read the locked Phase 2 configuration") from exc
    if hashlib.sha256(phase_payload).hexdigest() != LOCKED_PHASE_CONFIG_SHA256:
        raise ProfileBuildError("Phase 2 configuration bytes differ from the implementation lock")
    try:
        phase = tomllib.loads(phase_payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProfileBuildError("Locked Phase 2 configuration is not valid UTF-8 TOML") from exc
    if (
        phase.get("phase_01_freeze_manifest_sha256")
        != LOCKED_PHASE_01_FREEZE_MANIFEST_SHA256
    ):
        raise ProfileBuildError("Phase 1 freeze-manifest hash differs from the Phase 2 lock")
    expected_manifest_path = _locked_project_path(
        settings.project_root, phase["phase_01_freeze_manifest"]
    )
    if lexical_absolute(settings.freeze_manifest_path) != expected_manifest_path:
        raise ProfileBuildError("Phase 1 freeze-manifest path differs from the locked Phase 2 path")
    manifest = _read_authenticated_json(
        expected_manifest_path,
        expected_sha256=LOCKED_PHASE_01_FREEZE_MANIFEST_SHA256,
        label="Phase 1 freeze manifest",
    )
    identity = manifest.get("identity")
    if (
        not isinstance(identity, dict)
        or manifest.get("status") != "phase_01_frozen"
        or manifest.get("freeze_id") != phase.get("phase_01_freeze_id")
        or manifest.get("identity_sha256") != sha256_json(identity)
        or identity.get("status") != "phase_01_frozen"
        or identity.get("phase_01_build_id") != phase.get("phase_01_build_id")
        or identity.get("raw_snapshot_id") != phase.get("raw_snapshot_id")
        or identity.get("verification_id") != phase.get("verification_id")
    ):
        raise ProfileBuildError("Phase 1 freeze manifest identity differs from the Phase 2 lock")
    evidence_groups = identity.get("evidence_groups")
    operational_evidence = (
        evidence_groups.get("operational_evidence")
        if isinstance(evidence_groups, dict)
        else None
    )
    if (
        not isinstance(operational_evidence, list)
        or len(operational_evidence) != 1
        or not isinstance(operational_evidence[0], dict)
    ):
        raise ProfileBuildError("Phase 1 freeze manifest has no unique operational evidence")
    evidence = operational_evidence[0]
    if evidence.get("sha256") != LOCKED_PHASE_01_FREEZE_RECHECK_SHA256:
        raise ProfileBuildError("Phase 1 raw-recheck hash differs from the sealed manifest")
    if type(evidence.get("size_bytes")) is not int or evidence["size_bytes"] < 0:
        raise ProfileBuildError("Phase 1 raw-recheck evidence size is invalid")
    expected_recheck_path = _locked_project_path(settings.project_root, evidence.get("path", ""))
    if lexical_absolute(settings.freeze_recheck_path) != expected_recheck_path:
        raise ProfileBuildError("Phase 1 raw-recheck path differs from the sealed manifest")
    recheck = _read_authenticated_json(
        expected_recheck_path,
        expected_sha256=evidence["sha256"],
        expected_size_bytes=evidence["size_bytes"],
        label="Phase 1 raw-recheck metadata",
    )
    if (
        recheck.get("status") != "passed"
        or recheck.get("all_hashes_match_primary") is not True
        or recheck.get("all_files_stable_during_own_hash") is not True
        or recheck.get("all_files_stable_through_entire_recheck") is not True
    ):
        raise ProfileBuildError("Phase 1 final raw recheck is not a clean pass")
    result: dict[str, dict[str, int]] = {}
    for source_name in ("daily", "monthly"):
        filename = settings.config["sources"][source_name]["filename"]
        try:
            metadata = recheck["per_file_operational_metadata"][filename]
            raw = metadata["end_of_complete_recheck_stat"]
            before = metadata["stat_before"]
            after_hash = metadata["stat_after_hash"]
        except (KeyError, TypeError) as exc:
            raise ProfileBuildError(f"Frozen raw identity missing for {filename}") from exc
        if (
            metadata.get("stable_during_own_hash") is not True
            or metadata.get("stable_through_entire_recheck") is not True
            or before != after_hash
            or before != raw
        ):
            raise ProfileBuildError(f"Frozen raw identity is not stable for {filename}")
        fields = ("device", "inode", "mode", "size_bytes", "mtime_ns")
        if any(type(raw.get(field)) is not int for field in fields):
            raise ProfileBuildError(f"Frozen raw identity fields are invalid for {filename}")
        result[source_name] = {
            field: raw[field]
            for field in fields
        }
    return result


def _strict_date_array(values: pa.Array, *, label: str) -> tuple[pa.Array, np.ndarray]:
    """Return date32 values and a NumPy validity mask for strict YYYY-MM-DD."""
    lexical = pc.match_substring_regex(values, r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    parsed = pc.strptime(values, format="%Y-%m-%d", unit="s", error_is_null=True)
    parsed_valid = pc.is_valid(parsed)
    # Arrow normalizes some impossible dates; canonical round-trip equality is
    # therefore part of strict Gregorian validation, as is exclusion of year 0000.
    round_trip = pc.strftime(parsed, format="%Y-%m-%d")
    canonical = pc.fill_null(pc.equal(round_trip, values), False)
    valid = pc.and_(
        pc.and_(lexical, parsed_valid),
        pc.and_(canonical, pc.greater_equal(values, "0001-01-01")),
    )
    valid_numpy = np.asarray(valid.to_numpy(zero_copy_only=False), dtype=bool)
    if not valid_numpy.all():
        blank = int(pc.sum(pc.cast(pc.equal(values, ""), pa.int64())).as_py() or 0)
        invalid = len(values) - int(valid_numpy.sum()) - blank
        raise ProfileBuildError(
            f"{label} contains blank or invalid dates (blank={blank}, invalid={invalid})"
        )
    safe_parsed = pc.if_else(valid, parsed, None)
    return pc.cast(safe_parsed, pa.date32()), valid_numpy


def _positive_permno_array(values: pa.Array, *, label: str) -> pa.Array:
    lexical = np.asarray(
        pc.match_substring_regex(values, r"^[0-9]+$").to_numpy(zero_copy_only=False),
        dtype=bool,
    )
    if not lexical.all():
        blank = int(pc.sum(pc.cast(pc.equal(values, ""), pa.int64())).as_py() or 0)
        raise ProfileBuildError(
            f"{label} contains blank or non-decimal PERMNO values "
            f"(blank={blank}, invalid={len(values) - int(lexical.sum()) - blank})"
        )
    try:
        parsed = pc.cast(values, pa.int64(), safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise ProfileBuildError(f"{label} contains PERMNO outside positive int64") from exc
    nonpositive = int(pc.sum(pc.cast(pc.less_equal(parsed, 0), pa.int64())).as_py() or 0)
    if nonpositive:
        raise ProfileBuildError(f"{label} contains {nonpositive} nonpositive PERMNO values")
    return parsed


def _arrow_bool(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=bool)


def _adjacent_typed_edges(
    key_arrays: Sequence[pa.Array], previous_key: tuple[Any, ...] | None
) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
    """Vectorize equality and lexicographic-order checks for adjacent typed keys."""
    if not key_arrays or len(key_arrays[0]) == 0:
        raise ProfileBuildError("Cannot compute typed edges for an empty retained batch")
    length = len(key_arrays[0])
    internal_length = max(0, length - 1)
    if internal_length:
        equal = np.ones(internal_length, dtype=bool)
        less = np.zeros(internal_length, dtype=bool)
        prefix_equal = np.ones(internal_length, dtype=bool)
        for array in key_arrays:
            left = array.slice(0, internal_length)
            right = array.slice(1, internal_length)
            component_equal = _arrow_bool(pc.equal(right, left))
            component_less = _arrow_bool(pc.less(right, left))
            less |= prefix_equal & component_less
            prefix_equal &= component_equal
            equal &= component_equal
    else:
        equal = np.empty(0, dtype=bool)
        less = np.empty(0, dtype=bool)

    first = tuple(array[0].as_py() for array in key_arrays)
    last = tuple(array[length - 1].as_py() for array in key_arrays)
    if previous_key is not None:
        boundary_same = previous_key == first
        boundary_less = first < previous_key
        equal = np.concatenate((np.array([boundary_same], dtype=bool), equal))
        less = np.concatenate((np.array([boundary_less], dtype=bool), less))
    return equal, less, last


def _candidate_row_tuples(
    batch: pa.RecordBatch,
    same_edges: np.ndarray,
    *,
    has_previous: bool,
) -> dict[int, tuple[str, ...]]:
    """Materialize only batch rows participating in same-key edges.

    Returned keys are current-batch row indexes.  A boundary predecessor is
    intentionally absent; the caller supplies its one-row sentinel.
    """
    right_edges = np.flatnonzero(same_edges)
    indexes: set[int] = set()
    for edge_index in right_edges.tolist():
        right = edge_index if has_previous else edge_index + 1
        indexes.add(right)
        if right > 0:
            indexes.add(right - 1)
    if not indexes:
        return {}
    ordered = sorted(indexes)
    selected = batch.take(pa.array(ordered, type=pa.int64()))
    columns = [column.to_pylist() for column in selected.columns]
    return {
        original: tuple(column[position] for column in columns)
        for position, original in enumerate(ordered)
    }


def _edge_exactness(
    batch: pa.RecordBatch,
    same_edges: np.ndarray,
    *,
    previous_row: tuple[str, ...] | None,
) -> tuple[np.ndarray, dict[int, tuple[str, ...]]]:
    exact = np.ones(len(same_edges), dtype=bool)
    rows = _candidate_row_tuples(batch, same_edges, has_previous=previous_row is not None)
    for edge_index in np.flatnonzero(same_edges).tolist():
        right = edge_index if previous_row is not None else edge_index + 1
        left_row = previous_row if right == 0 else rows[right - 1]
        exact[edge_index] = left_row == rows[right]
    return exact, rows


class OrderedMultiplicityAccumulator:
    """Hash typed keys and multiplicities while retaining one boundary group.

    Complete groups are serialized by Arrow in chunks.  This keeps the
    26-million-row daily path out of Python row loops while implementing the
    locked fingerprint payload exactly.
    """

    FINGERPRINT_DOMAIN = b"portfolio-bot/crsp-key-multiplicity/v1\0"

    def __init__(self) -> None:
        self._pending_key: tuple[Any, ...] | None = None
        self._pending_size = 0
        self._pending_all_exact = True
        self._digest = hashlib.sha256()
        self._digest.update(self.FINGERPRINT_DOMAIN)
        self.histogram: Counter[int] = Counter()
        self.distinct_groups = 0
        self.duplicate_groups = 0
        self.affected_rows = 0
        self.surplus_rows = 0
        self.all_field_exact_groups = 0
        self.conflicting_groups = 0
        self.maximum_group_size = 0
        self._duplicate_metadata: list[dict[str, Any]] = []

    @staticmethod
    def _key_payload_arrays(
        key_arrays: Sequence[pa.Array], sizes: np.ndarray
    ) -> pa.Array:
        if not key_arrays:
            raise ProfileBuildError("Fingerprint key arrays are empty")
        arguments: list[pa.Array | pa.Scalar] = [pa.scalar("[")]
        for index, array in enumerate(key_arrays):
            if index:
                arguments.append(pa.scalar(","))
            rendered = pc.cast(array, pa.string())
            if pa.types.is_date(array.type):
                arguments.extend((pa.scalar('"'), rendered, pa.scalar('"')))
            elif pa.types.is_integer(array.type):
                arguments.append(rendered)
            else:
                raise ProfileBuildError(f"Unsupported typed fingerprint key: {array.type}")
        arguments.extend(
            (
                pa.scalar(","),
                pc.cast(pa.array(sizes, type=pa.int64()), pa.string()),
                pa.scalar("]\n"),
                pa.scalar(""),
            )
        )
        return pc.binary_join_element_wise(*arguments)

    def _record_stats(
        self,
        sizes: np.ndarray,
        all_exact: np.ndarray,
    ) -> None:
        if len(sizes) != len(all_exact) or np.any(sizes <= 0):
            raise ProfileBuildError("Invalid group multiplicity vectors")
        base_ordinal = self.distinct_groups
        unique, counts = np.unique(sizes, return_counts=True)
        for size, count in zip(unique.tolist(), counts.tolist()):
            self.histogram[int(size)] += int(count)
        self.distinct_groups += len(sizes)
        if len(sizes):
            self.maximum_group_size = max(self.maximum_group_size, int(sizes.max()))
        duplicate_indexes = np.flatnonzero(sizes > 1)
        self.duplicate_groups += len(duplicate_indexes)
        if len(duplicate_indexes):
            duplicate_sizes = sizes[duplicate_indexes]
            self.affected_rows += int(duplicate_sizes.sum())
            self.surplus_rows += int((duplicate_sizes - 1).sum())
            duplicate_exact = all_exact[duplicate_indexes]
            self.all_field_exact_groups += int(duplicate_exact.sum())
            self.conflicting_groups += int(len(duplicate_exact) - duplicate_exact.sum())
            for index in duplicate_indexes.tolist():
                item = {
                    "group_ordinal": base_ordinal + index,
                    "size": int(sizes[index]),
                    "all_exact": bool(all_exact[index]),
                }
                self._duplicate_metadata.append(item)

    def _emit_arrow(
        self,
        key_arrays: Sequence[pa.Array],
        sizes: np.ndarray,
        all_exact: np.ndarray,
    ) -> None:
        if not len(sizes):
            return
        payloads = self._key_payload_arrays(key_arrays, sizes)
        self._digest.update("".join(payloads.to_pylist()).encode("utf-8"))
        self._record_stats(sizes, all_exact)

    def _emit_one(self, key: tuple[Any, ...], size: int, all_exact: bool) -> None:
        arrays: list[pa.Array] = []
        for value in key:
            if isinstance(value, int):
                arrays.append(pa.array([value], type=pa.int64()))
            elif isinstance(value, dt.date):
                arrays.append(pa.array([value], type=pa.date32()))
            else:
                raise ProfileBuildError(f"Unsupported typed fingerprint key value: {value!r}")
        self._emit_arrow(
            arrays,
            np.array([size], dtype=np.int64),
            np.array([all_exact], dtype=bool),
        )

    def consume_batch(
        self,
        key_arrays: Sequence[pa.Array],
        internal_same_edges: np.ndarray,
        internal_exact_edges: np.ndarray,
        *,
        boundary_exact: bool = True,
    ) -> None:
        length = len(key_arrays[0])
        if length == 0:
            return
        if len(internal_same_edges) != max(0, length - 1):
            raise ProfileBuildError("Internal same-key edges have the wrong length")
        if len(internal_exact_edges) != len(internal_same_edges):
            raise ProfileBuildError("Internal exact edges have the wrong length")
        group_starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.flatnonzero(~internal_same_edges) + 1)
        )
        group_ends = np.concatenate((group_starts[1:], np.array([length], dtype=np.int64)))
        sizes = group_ends - group_starts
        exact = np.ones(len(sizes), dtype=bool)
        for index in np.flatnonzero(sizes > 1).tolist():
            start = int(group_starts[index])
            end = int(group_ends[index])
            exact[index] = bool(internal_exact_edges[start : end - 1].all())
        first_key = tuple(array[0].as_py() for array in key_arrays)
        last_key = tuple(array[length - 1].as_py() for array in key_arrays)
        if self._pending_key is not None:
            if self._pending_key == first_key:
                sizes[0] += self._pending_size
                exact[0] = exact[0] and self._pending_all_exact and boundary_exact
            else:
                self._emit_one(
                    self._pending_key, self._pending_size, self._pending_all_exact
                )
        emit_count = len(sizes) - 1
        if emit_count:
            selected_keys = [
                pc.take(array, pa.array(group_starts[:emit_count], type=pa.int64()))
                for array in key_arrays
            ]
            self._emit_arrow(selected_keys, sizes[:emit_count], exact[:emit_count])
        self._pending_key = last_key
        self._pending_size = int(sizes[-1])
        self._pending_all_exact = bool(exact[-1])

    def finish(self) -> dict[str, Any]:
        if self._pending_key is not None:
            self._emit_one(
                self._pending_key, self._pending_size, self._pending_all_exact
            )
            self._pending_key = None
        return {
            "distinct_groups": self.distinct_groups,
            "duplicate_groups": self.duplicate_groups,
            "affected_rows": self.affected_rows,
            "surplus_rows": self.surplus_rows,
            "all_field_exact_groups": self.all_field_exact_groups,
            "conflicting_groups": self.conflicting_groups,
            "maximum_group_size": self.maximum_group_size,
            "multiplicity_fingerprint_sha256": self._digest.hexdigest(),
            "histogram": dict(sorted(self.histogram.items())),
            "duplicate_metadata": self._duplicate_metadata,
        }


def _count_categoricals(
    batch: pa.RecordBatch,
    header_index: Mapping[str, int],
    columns: Sequence[str],
    counters: Mapping[str, Counter[str]],
) -> None:
    for column in columns:
        counts = pc.value_counts(batch.column(header_index[column]))
        values = counts.field("values").to_pylist()
        frequencies = counts.field("counts").to_pylist()
        counters[column].update(dict(zip(values, frequencies)))


def _last_row_tuple(batch: pa.RecordBatch) -> tuple[str, ...]:
    index = len(batch) - 1
    return tuple(column[index].as_py() for column in batch.columns)


def _scope_filter_batch(
    batch: pa.RecordBatch,
    date_index: int,
    *,
    start_date: str,
    end_date: str,
    source_name: str,
) -> tuple[pa.RecordBatch, pa.Array]:
    """Gate on the sole permitted out-of-scope value before other processing."""
    raw_dates = batch.column(date_index)
    parsed, _ = _strict_date_array(raw_dates, label=f"{source_name} scope date")
    mask = pc.and_(pc.greater_equal(raw_dates, start_date), pc.less_equal(raw_dates, end_date))
    retained = batch.filter(mask)
    retained_dates = pc.filter(parsed, mask)
    return retained, retained_dates


class SparseDuplicateGroupCollector:
    """Collect only rows belonging to adjacent duplicate typed-key groups."""

    def __init__(self) -> None:
        self.groups: list[list[tuple[str, ...]]] = []
        self._current: list[tuple[str, ...]] = []
        self._last_global_position: int | None = None

    def _finish_current(self) -> None:
        if self._current:
            self.groups.append(self._current)
            self._current = []
            self._last_global_position = None

    def consume(
        self,
        same_edges: np.ndarray,
        row_tuples: Mapping[int, tuple[str, ...]],
        *,
        previous_row: tuple[str, ...] | None,
        global_batch_start: int,
    ) -> None:
        has_previous = previous_row is not None
        for edge_index in np.flatnonzero(same_edges).tolist():
            right_local = edge_index if has_previous else edge_index + 1
            right_global = global_batch_start + right_local
            left_global = right_global - 1
            left_row = previous_row if right_local == 0 else row_tuples[right_local - 1]
            right_row = row_tuples[right_local]
            if self._last_global_position != left_global:
                self._finish_current()
                self._current = [left_row, right_row]
            else:
                self._current.append(right_row)
            self._last_global_position = right_global

    def finish(self) -> list[list[tuple[str, ...]]]:
        self._finish_current()
        return self.groups


def _optional_strict_date_diagnostics(values: pa.Array, *, label: str) -> dict[str, int]:
    blank_mask = pc.equal(values, "")
    blank = int(pc.sum(pc.cast(blank_mask, pa.int64())).as_py() or 0)
    if blank == len(values):
        return {"blank": blank, "invalid_nonblank": 0}
    nonblank = pc.filter(values, pc.invert(blank_mask))
    try:
        _strict_date_array(nonblank, label=label)
    except ProfileBuildError as exc:
        raise ProfileBuildError(f"{label} contains an invalid nonblank value") from exc
    return {"blank": blank, "invalid_nonblank": 0}


def _open_string_csv(
    path: Path, header: Sequence[str], *, block_size_bytes: int
) -> pacsv.CSVStreamingReader:
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(
            # Arrow's first block must hold the wide 36-field header.
            block_size=max(block_size_bytes, 4096),
            use_threads=True,
            encoding="utf8",
        ),
        parse_options=pacsv.ParseOptions(
            delimiter=",",
            quote_char='"',
            double_quote=True,
            escape_char=False,
            newlines_in_values=True,
            ignore_empty_lines=False,
        ),
        convert_options=pacsv.ConvertOptions(
            column_types={column: pa.string() for column in header},
            strings_can_be_null=False,
            quoted_strings_can_be_null=False,
            null_values=[],
            check_utf8=True,
        ),
    )
    if reader.schema.names != list(header):
        raise ProfileBuildError(
            f"{path.name} header differs: expected {list(header)!r}, "
            f"actual {reader.schema.names!r}"
        )
    if any(not pa.types.is_string(field.type) for field in reader.schema):
        raise ProfileBuildError(f"{path.name} was not parsed as 36 decoded strings")
    return reader


def scan_daily_source(
    path: Path,
    config: Mapping[str, Any],
    *,
    block_size_bytes: int = 16 << 20,
) -> dict[str, Any]:
    """Stream the daily source with a strict date-first development-scope gate."""
    source = config["sources"]["daily"]
    scope = config["scope"]
    header = list(source["header"])
    if len(header) != int(config["comparison"]["raw_field_count_per_source"]):
        raise ProfileBuildError("Daily configured header width is not the locked field count")
    header_index = {column: index for index, column in enumerate(header)}
    date_index = header_index[source["date_column"]]
    categorical_columns = list(config["profiled_categorical_columns"]["daily"])
    if any(column not in header_index for column in categorical_columns):
        raise ProfileBuildError("Configured daily categorical column is absent from header")

    reader = _open_string_csv(path, header, block_size_bytes=block_size_bytes)
    accumulator = OrderedMultiplicityAccumulator()
    category_counts: dict[str, Counter[str]] = {
        column: Counter() for column in categorical_columns
    }
    full_rows = 0
    retained_rows = 0
    sort_violations = 0
    inverted_intervals = 0
    previous_key: tuple[Any, ...] | None = None
    previous_row: tuple[str, ...] | None = None

    for raw_batch in reader:
        full_rows += len(raw_batch)
        retained, retained_dates = _scope_filter_batch(
            raw_batch,
            date_index,
            start_date=scope["start_date"],
            end_date=scope["end_date"],
            source_name="daily",
        )
        if len(retained) == 0:
            continue
        permno = _positive_permno_array(
            retained.column(header_index["PERMNO"]), label="daily retained rows"
        )
        interval_start, _ = _strict_date_array(
            retained.column(header_index["SecInfoStartDt"]),
            label="daily interval start",
        )
        interval_end, _ = _strict_date_array(
            retained.column(header_index["SecInfoEndDt"]),
            label="daily interval end",
        )
        inverted = int(
            pc.sum(pc.cast(pc.greater(interval_start, interval_end), pa.int64())).as_py()
            or 0
        )
        inverted_intervals += inverted
        if inverted:
            raise ProfileBuildError(f"Daily retained rows contain {inverted} inverted intervals")
        key_arrays = [permno, retained_dates]
        combined_same, combined_less, last_key = _adjacent_typed_edges(
            key_arrays, previous_key
        )
        batch_violations = int(combined_less.sum())
        sort_violations += batch_violations
        if batch_violations:
            raise ProfileBuildError(
                "Daily retained typed raw key is not nondecreasing; "
                f"observed {sort_violations} adjacent violation(s)"
            )
        combined_exact, _ = _edge_exactness(
            retained, combined_same, previous_row=previous_row
        )
        offset = 1 if previous_key is not None else 0
        internal_same = combined_same[offset:]
        internal_exact = combined_exact[offset:]
        boundary_exact = (
            bool(combined_exact[0])
            if previous_key is not None and bool(combined_same[0])
            else True
        )
        accumulator.consume_batch(
            key_arrays,
            internal_same,
            internal_exact,
            boundary_exact=boundary_exact,
        )
        _count_categoricals(
            retained, header_index, categorical_columns, category_counts
        )
        retained_rows += len(retained)
        previous_key = last_key
        # A single boundary sentinel prevents materializing ordinary 36-field rows.
        previous_row = _last_row_tuple(retained)

    metrics = accumulator.finish()
    metrics.update(
        {
            "source": source["filename"],
            "stage": "daily_raw_key",
            "key_fields": list(source["raw_key"]),
            "input_rows": retained_rows,
            "output_rows": metrics["distinct_groups"],
            "sort_violations": sort_violations,
            "effective_interval_candidate_groups": 0,
        }
    )
    expected = config["expected"]["daily_raw_key"]
    actual_gate = {
        key: metrics[key]
        for key in (
            "input_rows",
            "distinct_groups",
            "duplicate_groups",
            "affected_rows",
            "surplus_rows",
            "all_field_exact_groups",
            "conflicting_groups",
            "output_rows",
            "sort_violations",
        )
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual_gate[key]}
        for key in actual_gate
        if actual_gate[key] != expected[key]
    }
    if full_rows != int(source["full_file_rows"]):
        mismatches["full_file_rows"] = {
            "expected": int(source["full_file_rows"]),
            "actual": full_rows,
        }
    for column, values in config["expected"]["flag_domains"]["daily"].items():
        observed = sorted(
            (value for value in category_counts[column] if value != ""),
            key=lambda value: value.encode("utf-8"),
        )
        wanted = list(values)
        if observed != wanted:
            mismatches.setdefault("hard_gate_domains", {})[column] = {
                "expected": wanted,
                "actual": observed,
            }
    if mismatches:
        raise ProfileBuildError(f"Daily configured regression gates failed: {mismatches}")
    metrics.pop("duplicate_metadata", None)
    return {
        "source": "daily",
        "filename": source["filename"],
        "header": header,
        "full_file_rows_scanned": full_rows,
        "retained_rows": retained_rows,
        "stage": metrics,
        "categorical_counts": {
            column: dict(sorted(counter.items(), key=lambda item: item[0].encode("utf-8")))
            for column, counter in category_counts.items()
        },
        "additional_diagnostics": {"inverted_interval_rows": inverted_intervals},
        "scope_gate": {
            "date_column": source["date_column"],
            "start_date": scope["start_date"],
            "end_date": scope["end_date"],
            "excluded_rows": full_rows - retained_rows,
            "excluded_non_date_values_compared_or_aggregated": 0,
        },
    }


def _typed_monthly_key(
    row: tuple[str, ...], indexes: Mapping[str, int], *, raw: bool
) -> tuple[Any, ...]:
    key: tuple[Any, ...] = (
        int(row[indexes["PERMNO"]]),
        dt.date.fromisoformat(row[indexes["MthCalDt"]]),
    )
    if raw:
        key += (
            dt.date.fromisoformat(row[indexes["SecInfoStartDt"]]),
            dt.date.fromisoformat(row[indexes["SecInfoEndDt"]]),
        )
    return key


def _analyze_monthly_duplicate_groups(
    pre_groups: Sequence[Sequence[tuple[str, ...]]],
    post_groups: Sequence[Sequence[tuple[str, ...]]],
    header: Sequence[str],
    expected_pattern: Mapping[str, Any],
) -> dict[str, Any]:
    indexes = {column: index for index, column in enumerate(header)}
    embedded_raw_exact_groups = 0
    pre_effective_candidates = 0
    for group in pre_groups:
        raw_runs: list[list[tuple[str, ...]]] = []
        for row in group:
            key = _typed_monthly_key(row, indexes, raw=True)
            if not raw_runs or _typed_monthly_key(raw_runs[-1][-1], indexes, raw=True) != key:
                raw_runs.append([row])
            else:
                raw_runs[-1].append(row)
        collapsed = [run[0] for run in raw_runs]
        if len(collapsed) > 1:
            pre_effective_candidates += 1
            embedded_raw_exact_groups += sum(len(run) > 1 for run in raw_runs)

    difference_counts = {column: 0 for column in header}
    match_counts = {
        "groups_with_exactly_one_price_date_match": 0,
        "groups_with_zero_price_date_matches": 0,
        "groups_with_multiple_price_date_matches": 0,
        "groups_with_exactly_one_calendar_date_match": 0,
        "groups_with_zero_calendar_date_matches": 0,
        "groups_with_multiple_calendar_date_matches": 0,
    }
    candidate_sizes: Counter[int] = Counter()
    missing_price_date_rows_inside_candidates = 0
    price_and_calendar_selected_different_rows = 0
    observed_pattern = 0
    selected_conditional: set[str] = set()
    selected_status: set[str] = set()
    rejected_conditional: set[str] = set()
    rejected_status: set[str] = set()

    for group in post_groups:
        candidate_sizes[len(group)] += 1
        for column, index in indexes.items():
            if len({row[index] for row in group}) > 1:
                difference_counts[column] += 1
        price_matches: list[int] = []
        calendar_matches: list[int] = []
        for position, row in enumerate(group):
            start = dt.date.fromisoformat(row[indexes["SecInfoStartDt"]])
            end = dt.date.fromisoformat(row[indexes["SecInfoEndDt"]])
            calendar_date = dt.date.fromisoformat(row[indexes["MthCalDt"]])
            price_text = row[indexes["MthPrcDt"]]
            if price_text == "":
                missing_price_date_rows_inside_candidates += 1
            else:
                price_date = dt.date.fromisoformat(price_text)
                if start <= price_date <= end:
                    price_matches.append(position)
            if start <= calendar_date <= end:
                calendar_matches.append(position)
        if len(price_matches) == 1:
            match_counts["groups_with_exactly_one_price_date_match"] += 1
        elif not price_matches:
            match_counts["groups_with_zero_price_date_matches"] += 1
        else:
            match_counts["groups_with_multiple_price_date_matches"] += 1
        if len(calendar_matches) == 1:
            match_counts["groups_with_exactly_one_calendar_date_match"] += 1
        elif not calendar_matches:
            match_counts["groups_with_zero_calendar_date_matches"] += 1
        else:
            match_counts["groups_with_multiple_calendar_date_matches"] += 1
        if len(price_matches) != 1 or len(calendar_matches) != 1:
            continue
        if price_matches[0] != calendar_matches[0]:
            price_and_calendar_selected_different_rows += 1
            continue
        selected_index = price_matches[0]
        selected = group[selected_index]
        rejected = [row for index, row in enumerate(group) if index != selected_index]
        selected_conditional.add(selected[indexes["ConditionalType"]])
        selected_status.add(selected[indexes["TradingStatusFlg"]])
        rejected_conditional.update(row[indexes["ConditionalType"]] for row in rejected)
        rejected_status.update(row[indexes["TradingStatusFlg"]] for row in rejected)
        if (
            selected[indexes["ConditionalType"]]
            == expected_pattern["selected_conditional_type"]
            and selected[indexes["TradingStatusFlg"]]
            == expected_pattern["selected_trading_status"]
            and all(
                row[indexes["ConditionalType"]]
                == expected_pattern["rejected_conditional_type"]
                and row[indexes["TradingStatusFlg"]]
                == expected_pattern["rejected_trading_status"]
                for row in rejected
            )
        ):
            observed_pattern += 1

    def one_or_values(values: set[str]) -> Any:
        ordered = sorted(values, key=lambda value: value.encode("utf-8"))
        return ordered[0] if len(ordered) == 1 else ordered

    return {
        "pre_effective_interval_candidate_groups": pre_effective_candidates,
        "raw_exact_duplicate_groups_embedded_in_effective_interval_candidates": (
            embedded_raw_exact_groups
        ),
        "post_candidate_groups": len(post_groups),
        "candidate_size_histogram": dict(sorted(candidate_sizes.items())),
        "candidate_group_size": (
            next(iter(candidate_sizes)) if len(candidate_sizes) == 1 else sorted(candidate_sizes)
        ),
        **match_counts,
        "missing_price_date_rows_inside_candidates": missing_price_date_rows_inside_candidates,
        "price_and_calendar_selected_different_rows": (
            price_and_calendar_selected_different_rows
        ),
        "selected_conditional_type": one_or_values(selected_conditional),
        "selected_trading_status": one_or_values(selected_status),
        "rejected_conditional_type": one_or_values(rejected_conditional),
        "rejected_trading_status": one_or_values(rejected_status),
        "groups_with_observed_pattern": observed_pattern,
        "column_difference_groups": difference_counts,
    }


def _enforce_monthly_regression_gates(
    config: Mapping[str, Any],
    *,
    full_rows: int,
    raw_metrics: Mapping[str, Any],
    pre_metrics: Mapping[str, Any],
    post_metrics: Mapping[str, Any],
    candidates: Mapping[str, Any],
    category_counts: Mapping[str, Counter[str]],
) -> None:
    expected = config["expected"]
    failures: dict[str, Any] = {}

    def compare_map(
        label: str,
        actual: Mapping[str, Any],
        wanted: Mapping[str, Any],
        keys: Sequence[str],
    ) -> None:
        differences = {
            key: {"expected": wanted[key], "actual": actual[key]}
            for key in keys
            if actual[key] != wanted[key]
        }
        if differences:
            failures[label] = differences

    if full_rows != int(config["sources"]["monthly"]["full_file_rows"]):
        failures["full_file_rows"] = {
            "expected": int(config["sources"]["monthly"]["full_file_rows"]),
            "actual": full_rows,
        }
    common = (
        "input_rows",
        "distinct_groups",
        "duplicate_groups",
        "affected_rows",
        "surplus_rows",
        "output_rows",
    )
    compare_map(
        "monthly_raw_key",
        raw_metrics,
        expected["monthly_raw_key"],
        (*common, "all_field_exact_groups", "conflicting_groups", "sort_violations"),
    )
    compare_map(
        "monthly_canonical_key_before_raw_exact_collapse",
        pre_metrics,
        expected["monthly_canonical_key_before_raw_exact_collapse"],
        (
            "input_rows",
            "distinct_groups",
            "duplicate_groups",
            "affected_rows",
            "surplus_rows",
            "all_field_exact_groups",
            "effective_interval_candidate_groups",
        ),
    )
    embedded_expected = expected[
        "monthly_canonical_key_before_raw_exact_collapse"
    ]["raw_exact_duplicate_groups_embedded_in_effective_interval_candidates"]
    embedded_actual = candidates[
        "raw_exact_duplicate_groups_embedded_in_effective_interval_candidates"
    ]
    if embedded_actual != embedded_expected:
        failures["embedded_raw_exact_duplicate_groups"] = {
            "expected": embedded_expected,
            "actual": embedded_actual,
        }
    compare_map(
        "monthly_canonical_key_after_raw_exact_collapse",
        post_metrics,
        expected["monthly_canonical_key_after_raw_exact_collapse"],
        (
            "input_rows",
            "distinct_groups",
            "duplicate_groups",
            "affected_rows",
            "surplus_rows",
            "effective_interval_candidate_groups",
            "output_rows",
        ),
    )
    after_expected = expected["monthly_canonical_key_after_raw_exact_collapse"]
    candidate_actual = {
        key: candidates[key]
        for key in (
            "candidate_group_size",
            "groups_with_exactly_one_price_date_match",
            "groups_with_zero_price_date_matches",
            "groups_with_multiple_price_date_matches",
            "groups_with_exactly_one_calendar_date_match",
            "groups_with_zero_calendar_date_matches",
            "groups_with_multiple_calendar_date_matches",
        )
    }
    compare_map(
        "monthly_candidate_matches",
        candidate_actual,
        after_expected,
        tuple(candidate_actual),
    )
    diagnostic_expected = expected["monthly_candidate_diagnostics"]
    compare_map(
        "monthly_candidate_pattern",
        candidates,
        diagnostic_expected,
        (
            "selected_conditional_type",
            "selected_trading_status",
            "rejected_conditional_type",
            "rejected_trading_status",
            "groups_with_observed_pattern",
        ),
    )

    listed_differences = expected["monthly_candidate_column_difference_groups"]
    for column in config["sources"]["monthly"]["header"]:
        wanted = int(listed_differences.get(column, 0))
        actual = int(candidates["column_difference_groups"][column])
        if actual != wanted:
            failures.setdefault("monthly_candidate_column_differences", {})[column] = {
                "expected": wanted,
                "actual": actual,
            }

    for column, values in expected["flag_domains"]["monthly"].items():
        observed = sorted(
            (value for value in category_counts[column] if value != ""),
            key=lambda value: value.encode("utf-8"),
        )
        wanted = list(values)
        if observed != wanted:
            failures.setdefault("monthly_hard_gate_domains", {})[column] = {
                "expected": wanted,
                "actual": observed,
            }
    if failures:
        raise ProfileBuildError(f"Monthly configured regression gates failed: {failures}")


def scan_monthly_source(
    path: Path,
    config: Mapping[str, Any],
    *,
    block_size_bytes: int = 16 << 20,
) -> dict[str, Any]:
    """Stream, exactly collapse, and resolve monthly effective-interval candidates."""
    source = config["sources"]["monthly"]
    scope = config["scope"]
    header = list(source["header"])
    if len(header) != int(config["comparison"]["raw_field_count_per_source"]):
        raise ProfileBuildError("Monthly configured header width is not the locked field count")
    indexes = {column: index for index, column in enumerate(header)}
    categoricals = list(config["profiled_categorical_columns"]["monthly"])
    reader = _open_string_csv(path, header, block_size_bytes=block_size_bytes)

    raw_accumulator = OrderedMultiplicityAccumulator()
    pre_accumulator = OrderedMultiplicityAccumulator()
    post_accumulator = OrderedMultiplicityAccumulator()
    pre_collector = SparseDuplicateGroupCollector()
    post_collector = SparseDuplicateGroupCollector()
    category_counts = {column: Counter() for column in categoricals}
    full_rows = retained_rows = collapsed_rows = 0
    raw_sort_violations = pre_sort_violations = post_sort_violations = 0
    inverted_intervals = 0
    missing_price_date_rows_overall = 0
    previous_raw_key: tuple[Any, ...] | None = None
    previous_pre_key: tuple[Any, ...] | None = None
    previous_post_key: tuple[Any, ...] | None = None
    previous_raw_row: tuple[str, ...] | None = None
    previous_post_row: tuple[str, ...] | None = None

    for raw_batch in reader:
        full_rows += len(raw_batch)
        retained, calendar_dates = _scope_filter_batch(
            raw_batch,
            indexes[source["date_column"]],
            start_date=scope["start_date"],
            end_date=scope["end_date"],
            source_name="monthly",
        )
        if len(retained) == 0:
            continue
        batch_global_start = retained_rows
        permno = _positive_permno_array(
            retained.column(indexes["PERMNO"]), label="monthly retained rows"
        )
        interval_start, _ = _strict_date_array(
            retained.column(indexes[source["interval_start_column"]]),
            label="monthly interval start",
        )
        interval_end, _ = _strict_date_array(
            retained.column(indexes[source["interval_end_column"]]),
            label="monthly interval end",
        )
        inverted = int(
            pc.sum(pc.cast(pc.greater(interval_start, interval_end), pa.int64())).as_py()
            or 0
        )
        inverted_intervals += inverted
        if inverted:
            raise ProfileBuildError(f"Monthly retained rows contain {inverted} inverted intervals")
        price_diag = _optional_strict_date_diagnostics(
            retained.column(indexes[source["price_date_column"]]),
            label="monthly price date",
        )
        missing_price_date_rows_overall += price_diag["blank"]

        raw_keys = [permno, calendar_dates, interval_start, interval_end]
        pre_keys = [permno, calendar_dates]
        raw_same, raw_less, raw_last_key = _adjacent_typed_edges(
            raw_keys, previous_raw_key
        )
        pre_same, pre_less, pre_last_key = _adjacent_typed_edges(
            pre_keys, previous_pre_key
        )
        raw_sort_violations += int(raw_less.sum())
        pre_sort_violations += int(pre_less.sum())
        if raw_sort_violations or pre_sort_violations:
            raise ProfileBuildError(
                "Monthly retained typed raw/canonical keys are not nondecreasing"
            )
        pre_exact, pre_rows = _edge_exactness(
            retained, pre_same, previous_row=previous_raw_row
        )
        raw_exact = np.ones(len(raw_same), dtype=bool)
        raw_exact[raw_same] = pre_exact[raw_same]
        raw_offset = 1 if previous_raw_key is not None else 0
        pre_offset = 1 if previous_pre_key is not None else 0
        raw_accumulator.consume_batch(
            raw_keys,
            raw_same[raw_offset:],
            raw_exact[raw_offset:],
            boundary_exact=(
                bool(raw_exact[0])
                if previous_raw_key is not None and bool(raw_same[0])
                else True
            ),
        )
        pre_accumulator.consume_batch(
            pre_keys,
            pre_same[pre_offset:],
            pre_exact[pre_offset:],
            boundary_exact=(
                bool(pre_exact[0])
                if previous_pre_key is not None and bool(pre_same[0])
                else True
            ),
        )
        pre_collector.consume(
            pre_same,
            pre_rows,
            previous_row=previous_raw_row,
            global_batch_start=batch_global_start,
        )

        internal_raw_same = raw_same[raw_offset:]
        keep = np.ones(len(retained), dtype=bool)
        if previous_raw_key is not None and bool(raw_same[0]):
            keep[0] = False
        if len(retained) > 1:
            keep[1:] = ~internal_raw_same
        keep_indexes = np.flatnonzero(keep)
        if len(keep_indexes):
            collapsed = retained.take(pa.array(keep_indexes, type=pa.int64()))
            post_keys = [
                pc.take(permno, pa.array(keep_indexes, type=pa.int64())),
                pc.take(calendar_dates, pa.array(keep_indexes, type=pa.int64())),
            ]
            post_same, post_less, post_last_key = _adjacent_typed_edges(
                post_keys, previous_post_key
            )
            post_sort_violations += int(post_less.sum())
            if post_sort_violations:
                raise ProfileBuildError("Monthly post-collapse canonical key is not sorted")
            post_exact, post_rows = _edge_exactness(
                collapsed, post_same, previous_row=previous_post_row
            )
            post_offset = 1 if previous_post_key is not None else 0
            post_accumulator.consume_batch(
                post_keys,
                post_same[post_offset:],
                post_exact[post_offset:],
                boundary_exact=(
                    bool(post_exact[0])
                    if previous_post_key is not None and bool(post_same[0])
                    else True
                ),
            )
            post_collector.consume(
                post_same,
                post_rows,
                previous_row=previous_post_row,
                global_batch_start=collapsed_rows,
            )
            collapsed_rows += len(collapsed)
            previous_post_key = post_last_key
            previous_post_row = _last_row_tuple(collapsed)

        _count_categoricals(retained, indexes, categoricals, category_counts)
        retained_rows += len(retained)
        previous_raw_key = raw_last_key
        previous_pre_key = pre_last_key
        previous_raw_row = _last_row_tuple(retained)

    raw_metrics = raw_accumulator.finish()
    pre_metrics = pre_accumulator.finish()
    post_metrics = post_accumulator.finish()
    pre_groups = pre_collector.finish()
    post_groups = post_collector.finish()
    candidates = _analyze_monthly_duplicate_groups(
        pre_groups,
        post_groups,
        header,
        config["expected"]["monthly_candidate_diagnostics"],
    )
    if raw_metrics["conflicting_groups"]:
        raise ProfileBuildError(
            "Monthly four-part raw key contains nonexact duplicate rows"
        )
    if len(pre_groups) != pre_metrics["duplicate_groups"]:
        raise ProfileBuildError("Monthly precanonical sparse group reconstruction differs")
    if len(post_groups) != post_metrics["duplicate_groups"]:
        raise ProfileBuildError("Monthly post-collapse sparse group reconstruction differs")
    invalid_selection = {
        key: candidates[key]
        for key in (
            "groups_with_zero_price_date_matches",
            "groups_with_multiple_price_date_matches",
            "groups_with_zero_calendar_date_matches",
            "groups_with_multiple_calendar_date_matches",
            "price_and_calendar_selected_different_rows",
        )
        if candidates[key] != 0
    }
    if invalid_selection:
        raise ProfileBuildError(
            f"Monthly effective-interval candidate selection is not unique: {invalid_selection}"
        )

    raw_metrics.update(
        {
            "source": source["filename"],
            "stage": "monthly_raw_key",
            "key_fields": list(source["raw_key"]),
            "input_rows": retained_rows,
            "output_rows": raw_metrics["distinct_groups"],
            "sort_violations": raw_sort_violations,
            "effective_interval_candidate_groups": 0,
        }
    )
    pre_metrics.update(
        {
            "source": source["filename"],
            "stage": "monthly_canonical_key_before_raw_exact_collapse",
            "key_fields": list(source["canonical_key"]),
            "input_rows": retained_rows,
            "output_rows": pre_metrics["distinct_groups"],
            "sort_violations": pre_sort_violations,
            "conflicting_groups": 0,
            "effective_interval_candidate_groups": candidates[
                "pre_effective_interval_candidate_groups"
            ],
        }
    )
    post_metrics.update(
        {
            "source": source["filename"],
            "stage": "monthly_canonical_key_after_raw_exact_collapse",
            "key_fields": list(source["canonical_key"]),
            "input_rows": collapsed_rows,
            "output_rows": post_metrics["distinct_groups"],
            "sort_violations": post_sort_violations,
            "all_field_exact_groups": 0,
            "conflicting_groups": 0,
            "effective_interval_candidate_groups": len(post_groups),
        }
    )
    _enforce_monthly_regression_gates(
        config,
        full_rows=full_rows,
        raw_metrics=raw_metrics,
        pre_metrics=pre_metrics,
        post_metrics=post_metrics,
        candidates=candidates,
        category_counts=category_counts,
    )
    for metrics in (raw_metrics, pre_metrics, post_metrics):
        # Ordinals are useful only during the private scan.  Publishable scan
        # results contain aggregates/final digests and never row-level keys.
        metrics.pop("duplicate_metadata", None)
    return {
        "source": "monthly",
        "filename": source["filename"],
        "header": header,
        "full_file_rows_scanned": full_rows,
        "retained_rows": retained_rows,
        "collapsed_rows": collapsed_rows,
        "stages": {
            "monthly_raw_key": raw_metrics,
            "monthly_canonical_key_before_raw_exact_collapse": pre_metrics,
            "monthly_canonical_key_after_raw_exact_collapse": post_metrics,
        },
        "categorical_counts": {
            column: dict(sorted(counter.items(), key=lambda item: item[0].encode("utf-8")))
            for column, counter in category_counts.items()
        },
        "candidate_diagnostics": candidates,
        "additional_diagnostics": {
            "inverted_interval_rows": inverted_intervals,
            "missing_price_date_rows_overall": missing_price_date_rows_overall,
        },
        "scope_gate": {
            "date_column": source["date_column"],
            "start_date": scope["start_date"],
            "end_date": scope["end_date"],
            "excluded_rows": full_rows - retained_rows,
            "excluded_non_date_values_compared_or_aggregated": 0,
        },
    }


def _check_row(
    check_id: str,
    *,
    expected: Any,
    actual: Any,
    evidence: str,
    message: str,
    severity: str = "hard",
    passed: bool | None = None,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "severity": severity,
        "status": "PASS" if (actual == expected if passed is None else passed) else "FAIL",
        "expected": canonical_json_text(expected),
        "actual": canonical_json_text(actual),
        "evidence": evidence,
        "message": message,
    }


def _duplicate_summary_rows(
    config: Mapping[str, Any],
    daily: Mapping[str, Any],
    monthly: Mapping[str, Any],
) -> list[dict[str, Any]]:
    stages = {"daily_raw_key": daily["stage"], **monthly["stages"]}
    fields = config["outputs"]["schemas"]["duplicate_summary"]["columns"]
    rows = []
    for stage_name in config["outputs"]["population_registry"]["duplicate_stage_order"]:
        stage = stages[stage_name]
        rows.append({field: stage[field] for field in fields})
    return rows


def _multiplicity_rows(
    config: Mapping[str, Any], duplicate_rows: Sequence[Mapping[str, Any]], daily: Mapping[str, Any], monthly: Mapping[str, Any]
) -> list[dict[str, Any]]:
    stages = {"daily_raw_key": daily["stage"], **monthly["stages"]}
    result: list[dict[str, Any]] = []
    for summary in duplicate_rows:
        stage_name = summary["stage"]
        for multiplicity, count in sorted(stages[stage_name]["histogram"].items()):
            if int(count) <= 0:
                continue
            result.append(
                {
                    "source": summary["source"],
                    "stage": stage_name,
                    "multiplicity": int(multiplicity),
                    "group_count": int(count),
                    "affected_rows": int(multiplicity) * int(count),
                }
            )
    return result


def _column_difference_rows(
    config: Mapping[str, Any], monthly: Mapping[str, Any]
) -> list[dict[str, Any]]:
    source = config["sources"]["monthly"]
    schema = config["outputs"]["schemas"]["duplicate_column_differences"]
    listed = config["expected"]["monthly_candidate_column_difference_groups"]
    actual = monthly["candidate_diagnostics"]["column_difference_groups"]
    return [
        {
            "source": schema["source_literal"],
            "stage": schema["stage_literal"],
            "column": column,
            "groups_differing": int(actual[column]),
            "expected_groups_differing": int(listed.get(column, 0)),
            "status": "PASS" if int(actual[column]) == int(listed.get(column, 0)) else "FAIL",
        }
        for column in source["header"]
    ]


def _flag_domain_rows(
    config: Mapping[str, Any],
    daily: Mapping[str, Any],
    monthly: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source_name, scan, stage in (
        ("daily", daily, "daily_retained_raw_rows"),
        ("monthly", monthly, "monthly_retained_raw_rows"),
    ):
        filename = config["sources"][source_name]["filename"]
        hard_domains = config["expected"]["flag_domains"][source_name]
        for column in config["profiled_categorical_columns"][source_name]:
            policy = "hard_gate" if column in hard_domains else "discovery"
            expected = set(hard_domains.get(column, []))
            values = scan["categorical_counts"][column]
            for value in sorted(values, key=lambda item: (item != "", item.encode("utf-8"))):
                if value == "":
                    classification = "blank"
                elif policy == "discovery":
                    classification = "discovered_nonblank"
                elif value in expected:
                    classification = "expected_nonblank"
                else:
                    classification = "unexpected_nonblank"
                result.append(
                    {
                        "source": filename,
                        "stage": stage,
                        "column": column,
                        "raw_value": value,
                        "row_count": int(values[value]),
                        "is_blank": value == "",
                        "domain_policy": policy,
                        "classification": classification,
                    }
                )
    return result


def _compute_profile_build_id(compiled: Mapping[str, Any]) -> str:
    """Reproduce the binding-addressed primary build identifier."""
    return f"crsp-profile-{sha256_json(compiled['binding'])[:16]}"


def _expected_check_metadata(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Return the closed, deterministic audit-check vocabulary."""
    result: dict[str, dict[str, str]] = {}

    def add(
        check_id: str, evidence: str, message: str, severity: str = "hard"
    ) -> None:
        result[check_id] = {
            "severity": severity,
            "evidence": evidence,
            "message": message,
        }

    add(
        "UPSTREAM_CALENDAR_VERIFIED",
        project_label(project_path(config["upstream"]["calendar"]["verified_pointer"])),
        "The profile consumes the independently verified observed calendar.",
    )
    add(
        "UPSTREAM_CONTRACT_VERIFIED",
        project_label(project_path(config["upstream"]["contract"]["verified_pointer"])),
        "The profile consumes the independently verified CRSP contract.",
    )
    for source_name in ("daily", "monthly"):
        upper = source_name.upper()
        add(
            f"HEADER_{upper}_EXACT",
            f"{source_name} Arrow schema",
            "All 36 decoded string columns match the locked header in order.",
        )
        add(
            f"RAW_{upper}_IDENTITY_BEFORE",
            config["sources"][source_name]["filename"],
            "The raw operational identity matched the frozen Phase 1 recheck before scanning.",
        )
        add(
            f"RAW_{upper}_STABLE_DURING_SCAN",
            config["sources"][source_name]["filename"],
            "The raw file did not change during the complete scan.",
        )
        add(
            f"SCOPE_{upper}_NONDATE_EXCLUSION",
            f"{source_name} date-first scope receipt",
            "No out-of-scope non-date value entered a statistic, comparison, fingerprint, or output.",
        )
        add(
            f"DOMAIN_{upper}_HARD_GATES",
            "flag_domains.csv source population",
            "Every locked nonblank categorical domain is reproduced exactly.",
        )
    fixed = {
        "REGRESSION_DAILY_RAW_KEY": (
            "daily typed-key accumulator",
            "All locked daily duplicate metrics reproduce exactly.",
        ),
        "REGRESSION_MONTHLY_RAW_KEY": (
            "monthly typed four-part raw-key accumulator",
            "All locked monthly raw-key duplicate metrics reproduce exactly.",
        ),
        "REGRESSION_MONTHLY_CANONICAL_PRECOLLAPSE": (
            "monthly precanonical typed-key accumulator",
            "All locked precollapse canonical metrics reproduce exactly.",
        ),
        "REGRESSION_MONTHLY_CANONICAL_POSTCOLLAPSE": (
            "monthly postcollapse typed-key accumulator",
            "All locked postcollapse canonical and interval-match metrics reproduce exactly.",
        ),
        "FULL_FILE_ROW_COUNTS": (
            "complete monolithic CSV scan receipts",
            "Both full files were read completely while only retained rows fed profile statistics.",
        ),
        "INTERVAL_ENDPOINT_DIAGNOSTICS": (
            "strict retained-row interval parser",
            "Every retained effective interval is valid and non-inverted.",
        ),
        "MONTHLY_POSTCOLLAPSE_CANDIDATE_SIZE_HISTOGRAM": (
            "post-collapse monthly candidate collector",
            "Every effective-interval candidate group has the locked multiplicity.",
        ),
        "MULTIPLICITY_HISTOGRAM_MAXIMUM_AND_FINGERPRINTS": (
            "duplicate_multiplicity.csv and duplicate_summary.csv",
            "Ordered typed-key histograms, maxima, and domain-separated aggregate fingerprints are internally valid.",
        ),
        "ROW_CONSERVATION_ALL_STAGES": (
            "duplicate_summary.csv",
            "Every exact-collapse and canonical-resolution row equation conserves rows.",
        ),
        "MONTHLY_CANDIDATE_PATTERN_DIAGNOSTIC": (
            "monthly selected and rejected interval diagnostics",
            "The locked selected/rejected diagnostic pattern reproduces but never selects a row.",
        ),
        "MONTHLY_CANDIDATE_SAME_ROW_SELECTION": (
            "monthly post-collapse candidate diagnostics",
            "Price date and calendar date selected the same unique interval row.",
        ),
        "MONTHLY_COLUMN_DIFFERENCE_COMPLEMENT_ZERO": (
            "duplicate_column_differences.csv",
            "Only locked candidate columns differ, at their exact expected group counts.",
        ),
        "NO_ROW_LEVEL_KEY_METADATA_RETURNED": (
            "publishable scanner return objects",
            "No raw security key or row-level duplicate metadata reaches immutable outputs.",
        ),
        "NO_DOWNSTREAM_ANALYTICS_BUILT": (
            "locked profile scope",
            "This quality profile creates no features, targets, portfolios, P&L, or performance result.",
        ),
    }
    for check_id, (evidence, message) in fixed.items():
        add(check_id, evidence, message)
    add(
        "MONTHLY_MISSING_PRICE_DATE_DIAGNOSTICS",
        "retained monthly rows and post-collapse candidate groups",
        "Blank monthly price dates are counted; only singleton keys may pass through with one.",
        severity="info",
    )
    return result


def assemble_profile_outputs(
    settings: ProfileSettings,
    upstream: Mapping[str, Any],
    frozen_raw_stats: Mapping[str, Mapping[str, int]],
    raw_stats_before: Mapping[str, Mapping[str, int]],
    raw_stats_after: Mapping[str, Mapping[str, int]],
    daily: Mapping[str, Any],
    monthly: Mapping[str, Any],
    *,
    sealed_config_sha256: str | None = None,
    sealed_implementation_sha256: str | None = None,
) -> dict[str, Any]:
    """Compile deterministic, schema-locked output objects from passing scans."""
    config = settings.config
    current_config_sha256 = sha256_file(settings.profile_config_path)
    current_implementation_sha256 = sha256_file(Path(__file__).resolve())
    config_sha256 = sealed_config_sha256 or current_config_sha256
    implementation_sha256 = (
        sealed_implementation_sha256 or current_implementation_sha256
    )
    if current_config_sha256 != config_sha256:
        raise ProfileBuildError("Profile config bytes changed after the sealed scan start")
    if current_implementation_sha256 != implementation_sha256:
        raise ProfileBuildError("Primary implementation bytes changed after the sealed scan start")
    contract_pointer = upstream["contract"]["pointer"]
    calendar_pointer = upstream["calendar"]["pointer"]
    binding = {
        "calendar_build_id": calendar_pointer["build_id"],
        "calendar_pointer_sha256": upstream["calendar"]["pointer_sha256"],
        "calendar_verification_id": calendar_pointer["verification_id"],
        "config_sha256": config_sha256,
        "contract_id": contract_pointer["contract_id"],
        "contract_pointer_sha256": upstream["contract"]["pointer_sha256"],
        "contract_verification_id": contract_pointer["verification_id"],
        "daily_raw_sha256": config["sources"]["daily"]["sha256"],
        "implementation_sha256": implementation_sha256,
        "monthly_raw_sha256": config["sources"]["monthly"]["sha256"],
    }
    build_id = ""
    duplicate_rows = _duplicate_summary_rows(config, daily, monthly)
    multiplicity_rows = _multiplicity_rows(config, duplicate_rows, daily, monthly)
    difference_rows = _column_difference_rows(config, monthly)
    flag_rows = _flag_domain_rows(config, daily, monthly)

    checks = [
        _check_row(
            "UPSTREAM_CALENDAR_VERIFIED",
            expected="verified_passed",
            actual=calendar_pointer["status"],
            evidence=project_label(upstream["calendar"]["pointer_path"]),
            message="The profile consumes the independently verified observed calendar.",
        ),
        _check_row(
            "UPSTREAM_CONTRACT_VERIFIED",
            expected="verified_passed",
            actual=contract_pointer["status"],
            evidence=project_label(upstream["contract"]["pointer_path"]),
            message="The profile consumes the independently verified CRSP contract.",
        ),
    ]
    for source_name, scan in (("daily", daily), ("monthly", monthly)):
        checks.extend(
            (
                _check_row(
                    f"HEADER_{source_name.upper()}_EXACT",
                    expected=list(config["sources"][source_name]["header"]),
                    actual=list(scan["header"]),
                    evidence=f"{source_name} Arrow schema",
                    message="All 36 decoded string columns match the locked header in order.",
                ),
                _check_row(
                    f"RAW_{source_name.upper()}_IDENTITY_BEFORE",
                    expected=frozen_raw_stats[source_name],
                    actual=raw_stats_before[source_name],
                    evidence=config["sources"][source_name]["filename"],
                    message="The raw operational identity matched the frozen Phase 1 recheck before scanning.",
                ),
                _check_row(
                    f"RAW_{source_name.upper()}_STABLE_DURING_SCAN",
                    expected=raw_stats_before[source_name],
                    actual=raw_stats_after[source_name],
                    evidence=config["sources"][source_name]["filename"],
                    message="The raw file did not change during the complete scan.",
                ),
                _check_row(
                    f"SCOPE_{source_name.upper()}_NONDATE_EXCLUSION",
                    expected=0,
                    actual=scan["scope_gate"]["excluded_non_date_values_compared_or_aggregated"],
                    evidence=f"{source_name} date-first scope receipt",
                    message="No out-of-scope non-date value entered a statistic, comparison, fingerprint, or output.",
                ),
            )
        )
        expected_domains = config["expected"]["flag_domains"][source_name]
        actual_domains = {
            column: sorted(
                (value for value in scan["categorical_counts"][column] if value != ""),
                key=lambda value: value.encode("utf-8"),
            )
            for column in expected_domains
        }
        checks.append(
            _check_row(
                f"DOMAIN_{source_name.upper()}_HARD_GATES",
                expected={column: list(values) for column, values in expected_domains.items()},
                actual=actual_domains,
                evidence="flag_domains.csv source population",
                message="Every locked nonblank categorical domain is reproduced exactly.",
            )
        )
    expected_daily = config["expected"]["daily_raw_key"]
    daily_actual = {key: daily["stage"][key] for key in expected_daily}
    expected_monthly_raw = config["expected"]["monthly_raw_key"]
    monthly_raw_actual = {
        key: monthly["stages"]["monthly_raw_key"][key] for key in expected_monthly_raw
    }
    expected_pre = config["expected"]["monthly_canonical_key_before_raw_exact_collapse"]
    pre_stage = monthly["stages"]["monthly_canonical_key_before_raw_exact_collapse"]
    pre_actual = {
        key: (
            monthly["candidate_diagnostics"][key]
            if key == "raw_exact_duplicate_groups_embedded_in_effective_interval_candidates"
            else pre_stage[key]
        )
        for key in expected_pre
    }
    expected_post = config["expected"]["monthly_canonical_key_after_raw_exact_collapse"]
    post_stage = monthly["stages"]["monthly_canonical_key_after_raw_exact_collapse"]
    post_actual = {
        key: (
            monthly["candidate_diagnostics"][key]
            if key.startswith("groups_with_") or key == "candidate_group_size"
            else post_stage[key]
        )
        for key in expected_post
    }
    checks.extend(
        (
            _check_row(
                "REGRESSION_DAILY_RAW_KEY",
                expected=expected_daily,
                actual=daily_actual,
                evidence="daily typed-key accumulator",
                message="All locked daily duplicate metrics reproduce exactly.",
            ),
            _check_row(
                "REGRESSION_MONTHLY_RAW_KEY",
                expected=expected_monthly_raw,
                actual=monthly_raw_actual,
                evidence="monthly typed four-part raw-key accumulator",
                message="All locked monthly raw-key duplicate metrics reproduce exactly.",
            ),
            _check_row(
                "REGRESSION_MONTHLY_CANONICAL_PRECOLLAPSE",
                expected=expected_pre,
                actual=pre_actual,
                evidence="monthly precanonical typed-key accumulator",
                message="All locked precollapse canonical metrics reproduce exactly.",
            ),
            _check_row(
                "REGRESSION_MONTHLY_CANONICAL_POSTCOLLAPSE",
                expected=expected_post,
                actual=post_actual,
                evidence="monthly postcollapse typed-key accumulator",
                message="All locked postcollapse canonical and interval-match metrics reproduce exactly.",
            ),
        )
    )
    row_conservation = {
        "daily": daily["stage"]["input_rows"] - daily["stage"]["surplus_rows"]
        == daily["stage"]["output_rows"],
        "monthly_raw": monthly["stages"]["monthly_raw_key"]["input_rows"]
        - monthly["stages"]["monthly_raw_key"]["surplus_rows"]
        == monthly["stages"]["monthly_raw_key"]["output_rows"],
        "monthly_pre": pre_stage["input_rows"] - pre_stage["surplus_rows"]
        == pre_stage["output_rows"],
        "monthly_post": post_stage["input_rows"] - post_stage["surplus_rows"]
        == post_stage["output_rows"],
    }
    expected_pattern = config["expected"]["monthly_candidate_diagnostics"]
    actual_pattern = {
        key: monthly["candidate_diagnostics"][key]
        for key in (
            "selected_conditional_type",
            "selected_trading_status",
            "rejected_conditional_type",
            "rejected_trading_status",
            "groups_with_observed_pattern",
        )
    }
    locked_pattern = {key: expected_pattern[key] for key in actual_pattern}
    stage_map = {"daily_raw_key": daily["stage"], **monthly["stages"]}
    multiplicity_health = {}
    for stage_name, stage in stage_map.items():
        histogram = {int(key): int(value) for key, value in stage["histogram"].items()}
        multiplicity_health[stage_name] = {
            "fingerprint_is_lowercase_sha256": (
                len(stage["multiplicity_fingerprint_sha256"]) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in stage["multiplicity_fingerprint_sha256"]
                )
            ),
            "histogram_group_count_matches": sum(histogram.values())
            == stage["distinct_groups"],
            "histogram_row_count_matches": sum(
                multiplicity * count for multiplicity, count in histogram.items()
            )
            == stage["input_rows"],
            "maximum_group_size_matches": max(histogram, default=0)
            == stage["maximum_group_size"],
        }
    checks.extend(
        (
            _check_row(
                "FULL_FILE_ROW_COUNTS",
                expected={
                    "daily": int(config["sources"]["daily"]["full_file_rows"]),
                    "monthly": int(config["sources"]["monthly"]["full_file_rows"]),
                },
                actual={
                    "daily": int(daily["full_file_rows_scanned"]),
                    "monthly": int(monthly["full_file_rows_scanned"]),
                },
                evidence="complete monolithic CSV scan receipts",
                message="Both full files were read completely while only retained rows fed profile statistics.",
            ),
            _check_row(
                "INTERVAL_ENDPOINT_DIAGNOSTICS",
                expected={"daily_inverted": 0, "monthly_inverted": 0},
                actual={
                    "daily_inverted": daily["additional_diagnostics"]["inverted_interval_rows"],
                    "monthly_inverted": monthly["additional_diagnostics"]["inverted_interval_rows"],
                },
                evidence="strict retained-row interval parser",
                message="Every retained effective interval is valid and non-inverted.",
            ),
            _check_row(
                "MONTHLY_MISSING_PRICE_DATE_DIAGNOSTICS",
                expected="diagnostic_count_reported",
                actual={
                    "inside_candidate_rows": monthly["candidate_diagnostics"][
                        "missing_price_date_rows_inside_candidates"
                    ],
                    "overall_retained_rows": monthly["additional_diagnostics"][
                        "missing_price_date_rows_overall"
                    ],
                },
                evidence="retained monthly rows and post-collapse candidate groups",
                message="Blank monthly price dates are counted; only singleton keys may pass through with one.",
                severity="info",
                passed=True,
            ),
            _check_row(
                "MONTHLY_POSTCOLLAPSE_CANDIDATE_SIZE_HISTOGRAM",
                expected={
                    str(expected_post["candidate_group_size"]): expected_post[
                        "effective_interval_candidate_groups"
                    ]
                },
                actual={
                    str(key): value
                    for key, value in monthly["candidate_diagnostics"][
                        "candidate_size_histogram"
                    ].items()
                },
                evidence="post-collapse monthly candidate collector",
                message="Every effective-interval candidate group has the locked multiplicity.",
            ),
            _check_row(
                "MULTIPLICITY_HISTOGRAM_MAXIMUM_AND_FINGERPRINTS",
                expected={
                    stage_name: {key: True for key in health}
                    for stage_name, health in multiplicity_health.items()
                },
                actual=multiplicity_health,
                evidence="duplicate_multiplicity.csv and duplicate_summary.csv",
                message="Ordered typed-key histograms, maxima, and domain-separated aggregate fingerprints are internally valid.",
            ),
            _check_row(
                "ROW_CONSERVATION_ALL_STAGES",
                expected={key: True for key in row_conservation},
                actual=row_conservation,
                evidence="duplicate_summary.csv",
                message="Every exact-collapse and canonical-resolution row equation conserves rows.",
            ),
            _check_row(
                "MONTHLY_CANDIDATE_PATTERN_DIAGNOSTIC",
                expected=locked_pattern,
                actual=actual_pattern,
                evidence="monthly selected and rejected interval diagnostics",
                message="The locked selected/rejected diagnostic pattern reproduces but never selects a row.",
            ),
            _check_row(
                "MONTHLY_CANDIDATE_SAME_ROW_SELECTION",
                expected=0,
                actual=monthly["candidate_diagnostics"]["price_and_calendar_selected_different_rows"],
                evidence="monthly post-collapse candidate diagnostics",
                message="Price date and calendar date selected the same unique interval row.",
            ),
            _check_row(
                "MONTHLY_COLUMN_DIFFERENCE_COMPLEMENT_ZERO",
                expected=0,
                actual=sum(row["status"] != "PASS" for row in difference_rows),
                evidence="duplicate_column_differences.csv",
                message="Only locked candidate columns differ, at their exact expected group counts.",
            ),
            _check_row(
                "NO_ROW_LEVEL_KEY_METADATA_RETURNED",
                expected=False,
                actual=(
                    "duplicate_metadata" in daily["stage"]
                    or any(
                        "duplicate_metadata" in stage
                        for stage in monthly["stages"].values()
                    )
                ),
                evidence="publishable scanner return objects",
                message="No raw security key or row-level duplicate metadata reaches immutable outputs.",
            ),
            _check_row(
                "NO_DOWNSTREAM_ANALYTICS_BUILT",
                expected={"features": False, "targets": False, "portfolios": False, "pnl": False},
                actual={"features": False, "targets": False, "portfolios": False, "pnl": False},
                evidence="locked profile scope",
                message="This quality profile creates no features, targets, portfolios, P&L, or performance result.",
            ),
        )
    )
    checks = sorted(checks, key=lambda row: row["check_id"])
    failures = [row for row in checks if row["severity"] == "hard" and row["status"] != "PASS"]
    if failures:
        raise ProfileBuildError(f"Compiled hard checks failed: {[row['check_id'] for row in failures]}")

    status = config["outputs"]["serialization"]["published_primary_status"]
    summary = {
        "build_id": build_id,
        "check_count": len(checks),
        "daily_full_file_rows_scanned": int(daily["full_file_rows_scanned"]),
        "daily_retained_rows": int(daily["retained_rows"]),
        "daily_unique_security_days": int(daily["stage"]["output_rows"]),
        "fail_count": 0,
        "features_built": False,
        "monthly_full_file_rows_scanned": int(monthly["full_file_rows_scanned"]),
        "monthly_retained_rows": int(monthly["retained_rows"]),
        "monthly_unique_security_months": int(monthly["stages"]["monthly_canonical_key_after_raw_exact_collapse"]["output_rows"]),
        "pass_count": len(checks),
        "performance_results_seen": False,
        "pnl_built": False,
        "portfolios_built": False,
        "profile_name": config["profile_name"],
        "schema_version": 1,
        "status": status,
        "targets_built": False,
    }
    canonicalization_rules = {
        "comparison": copy.deepcopy(config["comparison"]),
        "config_sha256": config_sha256,
        "key_and_date_semantics": copy.deepcopy(config["key_and_date_semantics"]),
        "ordering_and_fingerprints": copy.deepcopy(config["ordering_and_fingerprints"]),
        "profile_name": config["profile_name"],
        "profile_version": config["profile_version"],
        "rules": copy.deepcopy(config["rules"]),
        "scope": copy.deepcopy(config["scope"]),
        "sources": copy.deepcopy(config["sources"]),
    }
    raw_inputs = {}
    for source_name, scan in (("daily", daily), ("monthly", monthly)):
        source = config["sources"][source_name]
        raw_inputs[source_name] = {
            "expected_full_file_rows": int(source["full_file_rows"]),
            "expected_sha256": source["sha256"],
            "expected_size_bytes": int(source["size_bytes"]),
            "filename": source["filename"],
            "full_file_rows_scanned": int(scan["full_file_rows_scanned"]),
            "operational_stat_after": dict(raw_stats_after[source_name]),
            "operational_stat_before": dict(raw_stats_before[source_name]),
            "retained_rows": int(scan["retained_rows"]),
        }
    input_receipt = {
        "build_id": build_id,
        "columns_read": {
            "daily": list(config["sources"]["daily"]["header"]),
            "monthly": list(config["sources"]["monthly"]["header"]),
        },
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "profile_name": config["profile_name"],
        "raw_inputs": raw_inputs,
        "schema_version": 1,
        "scope": {"end_date": config["scope"]["end_date"], "start_date": config["scope"]["start_date"]},
        "upstream": {
            "calendar_build_id": calendar_pointer["build_id"],
            "calendar_pointer_sha256": upstream["calendar"]["pointer_sha256"],
            "calendar_verification_id": calendar_pointer["verification_id"],
            "contract_id": contract_pointer["contract_id"],
            "contract_pointer_sha256": upstream["contract"]["pointer_sha256"],
            "contract_verification_id": contract_pointer["verification_id"],
        },
    }
    compiled = {
        "binding": binding,
        "build_id": build_id,
        "canonicalization_rules": canonicalization_rules,
        "checks": checks,
        "duplicate_column_differences": difference_rows,
        "duplicate_multiplicity": multiplicity_rows,
        "duplicate_summary": duplicate_rows,
        "flag_domains": flag_rows,
        "input_receipt": input_receipt,
        "summary": summary,
    }
    build_id = _compute_profile_build_id(compiled)
    compiled["build_id"] = build_id
    compiled["input_receipt"]["build_id"] = build_id
    compiled["summary"]["build_id"] = build_id
    validate_compiled_profile(
        config,
        compiled,
        config_path=settings.profile_config_path,
        settings=settings,
    )
    return compiled


def validate_compiled_profile(
    config: Mapping[str, Any],
    compiled: Mapping[str, Any],
    *,
    config_path: Path | None = None,
    settings: ProfileSettings | None = None,
) -> None:
    """Validate exact immutable schemas and deterministic row ordering."""
    expected_compiled_keys = {
        "binding",
        "build_id",
        "canonicalization_rules",
        "checks",
        "duplicate_column_differences",
        "duplicate_multiplicity",
        "duplicate_summary",
        "flag_domains",
        "input_receipt",
        "summary",
    }
    if set(compiled) != expected_compiled_keys:
        raise ProfileBuildError("Compiled profile object contains unexpected keys")
    preliminary_binding = compiled.get("binding")
    if not isinstance(preliminary_binding, Mapping):
        raise ProfileBuildError("Compiled binding is not an object")
    bound_config_sha = preliminary_binding.get("config_sha256")
    if config_path is None:
        default_hash = (
            sha256_file(DEFAULT_PROFILE_CONFIG) if DEFAULT_PROFILE_CONFIG.is_file() else None
        )
        if bound_config_sha == default_hash:
            config_path = DEFAULT_PROFILE_CONFIG
        else:
            raise ProfileBuildError(
                "A config_path is required to authenticate non-default compiled config bytes"
            )
    config_path = config_path.resolve()
    if not config_path.is_file() or sha256_file(config_path) != bound_config_sha:
        raise ProfileBuildError("Locked profile TOML bytes differ from binding config SHA-256")
    with config_path.open("rb") as handle:
        fresh_config = tomllib.load(handle)
    if fresh_config != config:
        raise ProfileBuildError("Caller config differs from freshly parsed locked TOML bytes")
    # All remaining validation is anchored to a fresh parse, never a mutable
    # object that could be aliased by a compiled subtree.
    config = fresh_config
    if settings is None:
        validation_settings = load_settings(profile_config_path=config_path)
    else:
        if settings.profile_config_path.resolve() != config_path:
            raise ProfileBuildError("Validation settings select a different profile config path")
        validation_settings = dataclasses.replace(
            settings,
            profile_config_path=config_path,
            config=copy.deepcopy(fresh_config),
        )
    authenticated_frozen_stats = frozen_raw_identities(validation_settings)
    schemas = config["outputs"]["schemas"]
    for name in ("canonicalization_rules", "input_receipt", "summary"):
        expected = set(schemas[name]["top_level_keys"])
        actual = set(compiled[name])
        if actual != expected:
            raise ProfileBuildError(
                f"{name} top-level schema differs: expected {sorted(expected)}, actual {sorted(actual)}"
            )
    if set(compiled["binding"]) != set(schemas["manifest"]["binding_keys"]):
        raise ProfileBuildError("Compiled binding schema differs from manifest contract")
    binding = compiled["binding"]

    def is_sha256(value: Any) -> bool:
        return (
            type(value) is str
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    for key, value in binding.items():
        if type(value) is not str:
            raise ProfileBuildError(f"Compiled binding {key} is not a string")
        if key.endswith("sha256") and not is_sha256(value):
            raise ProfileBuildError(f"Compiled binding {key} is not lowercase SHA-256")
    expected_binding_literals = {
        "calendar_build_id": config["upstream"]["calendar"]["build_id"],
        "calendar_pointer_sha256": config["upstream"]["calendar"][
            "verified_pointer_sha256"
        ],
        "calendar_verification_id": config["upstream"]["calendar"]["verification_id"],
        "contract_id": config["upstream"]["contract"]["contract_id"],
        "contract_pointer_sha256": config["upstream"]["contract"][
            "verified_pointer_sha256"
        ],
        "contract_verification_id": config["upstream"]["contract"]["verification_id"],
        "daily_raw_sha256": config["sources"]["daily"]["sha256"],
        "monthly_raw_sha256": config["sources"]["monthly"]["sha256"],
    }
    for key, expected_value in expected_binding_literals.items():
        if binding[key] != expected_value:
            raise ProfileBuildError(f"Compiled binding {key} differs from locked config")
    if not is_sha256(binding["config_sha256"]) or not is_sha256(
        binding["implementation_sha256"]
    ):
        raise ProfileBuildError("Compiled config/implementation hash is not lowercase SHA-256")
    if sha256_file(Path(__file__).resolve()) != binding["implementation_sha256"]:
        raise ProfileBuildError("Primary implementation bytes differ from bound implementation SHA-256")
    expected_build_id = _compute_profile_build_id(compiled)
    if type(compiled["build_id"]) is not str or compiled["build_id"] != expected_build_id:
        raise ProfileBuildError("Compiled build ID does not reproduce from the exact binding")
    rules = compiled["canonicalization_rules"]
    for key in (
        "comparison",
        "key_and_date_semantics",
        "ordering_and_fingerprints",
        "rules",
        "scope",
        "sources",
    ):
        if rules[key] != config[key]:
            raise ProfileBuildError(f"canonicalization_rules.{key} differs from locked config")
    if (
        rules["config_sha256"] != binding["config_sha256"]
        or not is_sha256(rules["config_sha256"])
    ):
        raise ProfileBuildError("canonicalization_rules config hash differs from binding")
    if rules["profile_name"] != config["profile_name"] or rules["profile_version"] != config[
        "profile_version"
    ]:
        raise ProfileBuildError("canonicalization_rules profile identity differs")
    receipt = compiled["input_receipt"]
    receipt_schema = schemas["input_receipt"]
    _assert_exact_keys(receipt["columns_read"], receipt_schema["columns_read_keys"], "columns_read")
    _assert_exact_keys(receipt["scope"], receipt_schema["scope_keys"], "input scope")
    _assert_exact_keys(receipt["upstream"], receipt_schema["upstream_keys"], "input upstream")
    _assert_exact_keys(receipt["raw_inputs"], receipt_schema["raw_input_keys"], "raw inputs")
    if (
        receipt["build_id"] != expected_build_id
        or receipt["profile_name"] != config["profile_name"]
        or type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != int(config["schema_version"])
    ):
        raise ProfileBuildError("Input receipt build/profile/schema identity differs")
    if receipt["config_sha256"] != binding["config_sha256"] or not is_sha256(
        receipt["config_sha256"]
    ):
        raise ProfileBuildError("Input receipt config hash differs from binding")
    if receipt["implementation_sha256"] != binding["implementation_sha256"] or not is_sha256(
        receipt["implementation_sha256"]
    ):
        raise ProfileBuildError("Input receipt implementation hash differs from binding")
    expected_columns = {
        source_name: list(config["sources"][source_name]["header"])
        for source_name in ("daily", "monthly")
    }
    if receipt["columns_read"] != expected_columns:
        raise ProfileBuildError("Input receipt columns differ from exact locked headers")
    expected_scope = {
        "end_date": config["scope"]["end_date"],
        "start_date": config["scope"]["start_date"],
    }
    if receipt["scope"] != expected_scope:
        raise ProfileBuildError("Input receipt scope differs from locked profile scope")
    expected_upstream = {
        "calendar_build_id": binding["calendar_build_id"],
        "calendar_pointer_sha256": binding["calendar_pointer_sha256"],
        "calendar_verification_id": binding["calendar_verification_id"],
        "contract_id": binding["contract_id"],
        "contract_pointer_sha256": binding["contract_pointer_sha256"],
        "contract_verification_id": binding["contract_verification_id"],
    }
    if receipt["upstream"] != expected_upstream:
        raise ProfileBuildError("Input receipt upstream binding differs")
    for source_name in receipt_schema["raw_input_keys"]:
        entry = receipt["raw_inputs"][source_name]
        _assert_exact_keys(entry, receipt_schema["raw_input_entry_keys"], f"raw_inputs.{source_name}")
        for stat_name in ("operational_stat_before", "operational_stat_after"):
            _assert_exact_keys(
                entry[stat_name], receipt_schema["operational_stat_keys"], f"{source_name}.{stat_name}"
            )
            if any(type(value) is not int for value in entry[stat_name].values()):
                raise ProfileBuildError(f"{source_name}.{stat_name} contains a non-int64 value")
        source = config["sources"][source_name]
        expected_entry_literals = {
            "expected_full_file_rows": int(source["full_file_rows"]),
            "expected_sha256": source["sha256"],
            "expected_size_bytes": int(source["size_bytes"]),
            "filename": source["filename"],
            "full_file_rows_scanned": int(source["full_file_rows"]),
            "retained_rows": int(
                config["expected"][
                    "daily_raw_key" if source_name == "daily" else "monthly_raw_key"
                ]["input_rows"]
            ),
        }
        for key, expected_value in expected_entry_literals.items():
            if entry[key] != expected_value or type(entry[key]) is not type(expected_value):
                raise ProfileBuildError(f"raw_inputs.{source_name}.{key} differs from lock")
        if not is_sha256(entry["expected_sha256"]):
            raise ProfileBuildError(f"raw_inputs.{source_name}.expected_sha256 is invalid")
        if entry["operational_stat_before"] != entry["operational_stat_after"]:
            raise ProfileBuildError(f"raw_inputs.{source_name} changed during scan")
        if entry["operational_stat_before"] != authenticated_frozen_stats[source_name]:
            raise ProfileBuildError(
                f"raw_inputs.{source_name} operational identity differs from authenticated Phase 1 freeze"
            )
        if entry["operational_stat_before"]["size_bytes"] != int(source["size_bytes"]):
            raise ProfileBuildError(f"raw_inputs.{source_name} operational size differs")
    table_map = {
        "duplicate_summary": "duplicate_summary",
        "duplicate_column_differences": "duplicate_column_differences",
        "duplicate_multiplicity": "duplicate_multiplicity",
        "flag_domains": "flag_domains",
        "checks": "checks",
    }
    for compiled_name, schema_name in table_map.items():
        fields = list(schemas[schema_name]["columns"])
        for index, row in enumerate(compiled[compiled_name]):
            if list(row) != fields:
                raise ProfileBuildError(
                    f"{compiled_name} row {index} fields differ from exact schema"
                )
    summary_rows = compiled["duplicate_summary"]
    expected_stage_order = schemas["duplicate_summary"]["stage_order"]
    stage_specs = {
        "daily_raw_key": (
            "daily",
            config["sources"]["daily"]["raw_key"],
            config["expected"]["daily_raw_key"],
        ),
        "monthly_raw_key": (
            "monthly",
            config["sources"]["monthly"]["raw_key"],
            config["expected"]["monthly_raw_key"],
        ),
        "monthly_canonical_key_before_raw_exact_collapse": (
            "monthly",
            config["sources"]["monthly"]["canonical_key"],
            config["expected"]["monthly_canonical_key_before_raw_exact_collapse"],
        ),
        "monthly_canonical_key_after_raw_exact_collapse": (
            "monthly",
            config["sources"]["monthly"]["canonical_key"],
            config["expected"]["monthly_canonical_key_after_raw_exact_collapse"],
        ),
    }
    for row in summary_rows:
        if type(row["source"]) is not str or type(row["stage"]) is not str:
            raise ProfileBuildError("Duplicate-summary source/stage scalar type differs")
        if not isinstance(row["key_fields"], list) or not all(
            type(value) is str for value in row["key_fields"]
        ):
            raise ProfileBuildError("Duplicate-summary key_fields is not a string array")
        for field in (
            "input_rows",
            "distinct_groups",
            "duplicate_groups",
            "affected_rows",
            "surplus_rows",
            "all_field_exact_groups",
            "conflicting_groups",
            "effective_interval_candidate_groups",
            "output_rows",
            "sort_violations",
            "maximum_group_size",
        ):
            if type(row[field]) is not int or row[field] < 0:
                raise ProfileBuildError(f"Duplicate-summary {field} is not a nonnegative int64")
        digest = row["multiplicity_fingerprint_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ProfileBuildError("Duplicate-summary fingerprint is not lowercase SHA-256")
    if [row["stage"] for row in summary_rows] != expected_stage_order:
        raise ProfileBuildError("Duplicate summary stage order differs from locked order")
    if len(summary_rows) != int(schemas["duplicate_summary"]["row_count"]):
        raise ProfileBuildError("Duplicate summary row count differs")
    for row in summary_rows:
        source_name, expected_key_fields, expected_metrics = stage_specs[row["stage"]]
        if row["source"] != config["sources"][source_name]["filename"]:
            raise ProfileBuildError(f"Duplicate-summary {row['stage']} source differs")
        if row["key_fields"] != list(expected_key_fields):
            raise ProfileBuildError(f"Duplicate-summary {row['stage']} key fields differ")
        metric_keys = (
            "input_rows",
            "distinct_groups",
            "duplicate_groups",
            "affected_rows",
            "surplus_rows",
            "output_rows",
        )
        for key in metric_keys:
            locked_value = expected_metrics.get(
                key,
                expected_metrics["distinct_groups"] if key == "output_rows" else None,
            )
            if row[key] != int(locked_value):
                raise ProfileBuildError(f"Duplicate-summary {row['stage']} {key} differs")
        if row["sort_violations"] != 0 or row["conflicting_groups"] != 0:
            raise ProfileBuildError(f"Duplicate-summary {row['stage']} has a forbidden failure")
        expected_exact = int(expected_metrics.get("all_field_exact_groups", 0))
        expected_candidates = int(expected_metrics.get("effective_interval_candidate_groups", 0))
        if row["all_field_exact_groups"] != expected_exact:
            raise ProfileBuildError(f"Duplicate-summary {row['stage']} exact-group count differs")
        if row["effective_interval_candidate_groups"] != expected_candidates:
            raise ProfileBuildError(
                f"Duplicate-summary {row['stage']} candidate-group count differs"
            )
        if row["input_rows"] - row["surplus_rows"] != row["output_rows"]:
            raise ProfileBuildError(f"Duplicate-summary {row['stage']} row conservation differs")
    differences = compiled["duplicate_column_differences"]
    if len(differences) != int(schemas["duplicate_column_differences"]["row_count"]):
        raise ProfileBuildError("Duplicate-column difference row count differs")
    monthly_header = config["sources"]["monthly"]["header"]
    if [row["column"] for row in differences] != monthly_header:
        raise ProfileBuildError("Duplicate-column differences are not in monthly header order")
    if any(row["status"] != "PASS" for row in differences):
        raise ProfileBuildError("A candidate column-difference row failed")
    multiplicity_order = {
        stage: index for index, stage in enumerate(schemas["duplicate_multiplicity"]["stage_order"])
    }
    stage_sources = {row["stage"]: row["source"] for row in summary_rows}
    for row in compiled["duplicate_multiplicity"]:
        if (
            type(row["source"]) is not str
            or type(row["stage"]) is not str
            or row["stage"] not in multiplicity_order
            or any(
                type(row[field]) is not int
                for field in ("multiplicity", "group_count", "affected_rows")
            )
        ):
            raise ProfileBuildError("Duplicate-multiplicity scalar type or stage differs")
        if row["source"] != stage_sources[row["stage"]]:
            raise ProfileBuildError("Duplicate-multiplicity source differs from stage")
        if row["multiplicity"] <= 0 or row["group_count"] <= 0:
            raise ProfileBuildError("Duplicate-multiplicity values must be positive")
    observed_order = [
        (multiplicity_order[row["stage"]], row["multiplicity"])
        for row in compiled["duplicate_multiplicity"]
    ]
    if observed_order != sorted(observed_order):
        raise ProfileBuildError("Duplicate multiplicity rows are not deterministically ordered")
    if any(
        row["group_count"] <= 0
        or row["affected_rows"] != row["multiplicity"] * row["group_count"]
        for row in compiled["duplicate_multiplicity"]
    ):
        raise ProfileBuildError("Duplicate multiplicity row conservation failed")
    for stage_name, stage_source in stage_sources.items():
        rows = [row for row in compiled["duplicate_multiplicity"] if row["stage"] == stage_name]
        summary_row = next(row for row in summary_rows if row["stage"] == stage_name)
        if not rows:
            raise ProfileBuildError(f"Duplicate-multiplicity stage {stage_name} is absent")
        if len({row["multiplicity"] for row in rows}) != len(rows):
            raise ProfileBuildError(f"Duplicate-multiplicity stage {stage_name} repeats a size")
        if sum(row["group_count"] for row in rows) != summary_row["distinct_groups"]:
            raise ProfileBuildError(f"Duplicate-multiplicity stage {stage_name} group total differs")
        if sum(row["affected_rows"] for row in rows) != summary_row["input_rows"]:
            raise ProfileBuildError(f"Duplicate-multiplicity stage {stage_name} row total differs")
        if max(row["multiplicity"] for row in rows) != summary_row["maximum_group_size"]:
            raise ProfileBuildError(f"Duplicate-multiplicity stage {stage_name} maximum differs")
    for row in differences:
        if (
            type(row["source"]) is not str
            or type(row["stage"]) is not str
            or type(row["column"]) is not str
            or type(row["groups_differing"]) is not int
            or type(row["expected_groups_differing"]) is not int
            or row["status"] not in {"PASS", "FAIL"}
        ):
            raise ProfileBuildError("Duplicate-column difference scalar type differs")
        if (
            row["source"] != schemas["duplicate_column_differences"]["source_literal"]
            or row["stage"] != schemas["duplicate_column_differences"]["stage_literal"]
        ):
            raise ProfileBuildError("Duplicate-column difference source/stage differs")
        expected_groups = int(
            config["expected"]["monthly_candidate_column_difference_groups"].get(
                row["column"], 0
            )
        )
        if (
            row["groups_differing"] != expected_groups
            or row["expected_groups_differing"] != expected_groups
            or row["status"] != "PASS"
        ):
            raise ProfileBuildError("Duplicate-column difference count differs from lock")
    expected_pattern_config = config["expected"]["monthly_candidate_diagnostics"]
    pattern_keys = (
        "selected_conditional_type",
        "selected_trading_status",
        "rejected_conditional_type",
        "rejected_trading_status",
        "groups_with_observed_pattern",
    )
    expected_check_values: dict[str, Any] = {
        "UPSTREAM_CALENDAR_VERIFIED": "verified_passed",
        "UPSTREAM_CONTRACT_VERIFIED": "verified_passed",
        "HEADER_DAILY_EXACT": list(config["sources"]["daily"]["header"]),
        "HEADER_MONTHLY_EXACT": list(config["sources"]["monthly"]["header"]),
        "RAW_DAILY_IDENTITY_BEFORE": authenticated_frozen_stats["daily"],
        "RAW_MONTHLY_IDENTITY_BEFORE": authenticated_frozen_stats["monthly"],
        "RAW_DAILY_STABLE_DURING_SCAN": authenticated_frozen_stats["daily"],
        "RAW_MONTHLY_STABLE_DURING_SCAN": authenticated_frozen_stats["monthly"],
        "SCOPE_DAILY_NONDATE_EXCLUSION": 0,
        "SCOPE_MONTHLY_NONDATE_EXCLUSION": 0,
        "DOMAIN_DAILY_HARD_GATES": {
            column: list(values)
            for column, values in config["expected"]["flag_domains"]["daily"].items()
        },
        "DOMAIN_MONTHLY_HARD_GATES": {
            column: list(values)
            for column, values in config["expected"]["flag_domains"]["monthly"].items()
        },
        "REGRESSION_DAILY_RAW_KEY": config["expected"]["daily_raw_key"],
        "REGRESSION_MONTHLY_RAW_KEY": config["expected"]["monthly_raw_key"],
        "REGRESSION_MONTHLY_CANONICAL_PRECOLLAPSE": config["expected"][
            "monthly_canonical_key_before_raw_exact_collapse"
        ],
        "REGRESSION_MONTHLY_CANONICAL_POSTCOLLAPSE": config["expected"][
            "monthly_canonical_key_after_raw_exact_collapse"
        ],
        "FULL_FILE_ROW_COUNTS": {
            "daily": int(config["sources"]["daily"]["full_file_rows"]),
            "monthly": int(config["sources"]["monthly"]["full_file_rows"]),
        },
        "INTERVAL_ENDPOINT_DIAGNOSTICS": {
            "daily_inverted": 0,
            "monthly_inverted": 0,
        },
        "MONTHLY_MISSING_PRICE_DATE_DIAGNOSTICS": "diagnostic_count_reported",
        "MONTHLY_POSTCOLLAPSE_CANDIDATE_SIZE_HISTOGRAM": {
            str(
                config["expected"]["monthly_canonical_key_after_raw_exact_collapse"][
                    "candidate_group_size"
                ]
            ): int(
                config["expected"]["monthly_canonical_key_after_raw_exact_collapse"][
                    "effective_interval_candidate_groups"
                ]
            )
        },
        "MULTIPLICITY_HISTOGRAM_MAXIMUM_AND_FINGERPRINTS": {
            stage_name: {
                "fingerprint_is_lowercase_sha256": True,
                "histogram_group_count_matches": True,
                "histogram_row_count_matches": True,
                "maximum_group_size_matches": True,
            }
            for stage_name in schemas["duplicate_summary"]["stage_order"]
        },
        "ROW_CONSERVATION_ALL_STAGES": {
            "daily": True,
            "monthly_raw": True,
            "monthly_pre": True,
            "monthly_post": True,
        },
        "MONTHLY_CANDIDATE_PATTERN_DIAGNOSTIC": {
            key: expected_pattern_config[key] for key in pattern_keys
        },
        "MONTHLY_CANDIDATE_SAME_ROW_SELECTION": 0,
        "MONTHLY_COLUMN_DIFFERENCE_COMPLEMENT_ZERO": 0,
        "NO_ROW_LEVEL_KEY_METADATA_RETURNED": False,
        "NO_DOWNSTREAM_ANALYTICS_BUILT": {
            "features": False,
            "targets": False,
            "portfolios": False,
            "pnl": False,
        },
    }
    checks = compiled["checks"]
    expected_check_metadata = _expected_check_metadata(config)
    for row in checks:
        if (
            type(row["check_id"]) is not str
            or row["severity"] not in {"hard", "info"}
            or row["status"] not in {"PASS", "FAIL"}
            or type(row["expected"]) is not str
            or type(row["actual"]) is not str
            or type(row["evidence"]) is not str
            or type(row["message"]) is not str
        ):
            raise ProfileBuildError("Check row scalar type or enum differs")
        metadata = expected_check_metadata.get(row["check_id"])
        if metadata is None:
            raise ProfileBuildError("Check ID differs from the closed audit vocabulary")
        for field in ("severity", "evidence", "message"):
            if row[field] != metadata[field]:
                raise ProfileBuildError(f"Check {row['check_id']} {field} differs from contract")
    check_ids = [row["check_id"] for row in checks]
    if check_ids != sorted(check_ids) or len(check_ids) != len(set(check_ids)):
        raise ProfileBuildError("Checks are not unique and lexicographically ordered")
    if set(check_ids) != set(expected_check_metadata):
        raise ProfileBuildError("Checks do not equal the closed 27-check audit registry")
    if any(row["severity"] == "hard" and row["status"] != "PASS" for row in checks):
        raise ProfileBuildError("Compiled profile contains a failing hard check")
    if any(row["status"] != "PASS" for row in checks):
        raise ProfileBuildError("Primary profile may not publish a failing informational check")
    for row in checks:
        if row["severity"] == "hard" and row["expected"] != row["actual"]:
            raise ProfileBuildError(f"Hard check {row['check_id']} actual differs from expected")
        parsed_values: dict[str, Any] = {}
        for field in ("expected", "actual"):
            try:
                parsed = json.loads(row[field])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ProfileBuildError(f"Check {field} is not a JSON value") from exc
            if canonical_json_text(parsed) != row[field]:
                raise ProfileBuildError(f"Check {field} is not canonically serialized JSON")
            parsed_values[field] = parsed
        if parsed_values["expected"] != expected_check_values[row["check_id"]]:
            raise ProfileBuildError(f"Check {row['check_id']} expected contract differs")
        if row["check_id"] == "MONTHLY_MISSING_PRICE_DATE_DIAGNOSTICS":
            actual_diagnostic = parsed_values["actual"]
            if (
                type(actual_diagnostic) is not dict
                or set(actual_diagnostic)
                != {"inside_candidate_rows", "overall_retained_rows"}
                or any(type(value) is not int or value < 0 for value in actual_diagnostic.values())
                or actual_diagnostic["inside_candidate_rows"]
                > actual_diagnostic["overall_retained_rows"]
            ):
                raise ProfileBuildError("Missing-price diagnostic actual shape differs")
    if compiled["summary"]["check_count"] != len(checks):
        raise ProfileBuildError("Summary check count differs from checks.csv")
    if compiled["summary"]["pass_count"] != len(checks) or compiled["summary"]["fail_count"] != 0:
        raise ProfileBuildError("Summary pass/fail counts differ from checks.csv")
    summary_type_map = {
        "build_id": str,
        "check_count": int,
        "daily_full_file_rows_scanned": int,
        "daily_retained_rows": int,
        "daily_unique_security_days": int,
        "fail_count": int,
        "features_built": bool,
        "monthly_full_file_rows_scanned": int,
        "monthly_retained_rows": int,
        "monthly_unique_security_months": int,
        "pass_count": int,
        "performance_results_seen": bool,
        "pnl_built": bool,
        "portfolios_built": bool,
        "profile_name": str,
        "schema_version": int,
        "status": str,
        "targets_built": bool,
    }
    for key, expected_type in summary_type_map.items():
        if type(compiled["summary"][key]) is not expected_type:
            raise ProfileBuildError(f"Summary {key} scalar type differs")
    summary = compiled["summary"]
    summary_expected_literals = {
        "build_id": expected_build_id,
        "daily_full_file_rows_scanned": int(config["sources"]["daily"]["full_file_rows"]),
        "daily_retained_rows": int(config["expected"]["daily_raw_key"]["input_rows"]),
        "daily_unique_security_days": int(config["expected"]["daily_raw_key"]["output_rows"]),
        "features_built": False,
        "monthly_full_file_rows_scanned": int(
            config["sources"]["monthly"]["full_file_rows"]
        ),
        "monthly_retained_rows": int(config["expected"]["monthly_raw_key"]["input_rows"]),
        "monthly_unique_security_months": int(
            config["expected"]["monthly_canonical_key_after_raw_exact_collapse"][
                "output_rows"
            ]
        ),
        "performance_results_seen": False,
        "pnl_built": False,
        "portfolios_built": False,
        "profile_name": config["profile_name"],
        "schema_version": int(config["schema_version"]),
        "status": config["outputs"]["serialization"]["published_primary_status"],
        "targets_built": False,
    }
    for key, expected_value in summary_expected_literals.items():
        if summary[key] != expected_value or type(summary[key]) is not type(expected_value):
            raise ProfileBuildError(f"Summary {key} differs from locked primary semantics")
    flag_rows = compiled["flag_domains"]
    source_names = ("daily", "monthly")
    source_index = {
        config["sources"][source_name]["filename"]: index
        for index, source_name in enumerate(source_names)
    }
    source_by_filename = {
        config["sources"][source_name]["filename"]: source_name
        for source_name in source_names
    }
    expected_flag_stage = {
        "daily": config["outputs"]["population_registry"]["flag_daily_stage"],
        "monthly": config["outputs"]["population_registry"]["flag_monthly_stage"],
    }
    flag_order: list[tuple[int, int, int, bytes]] = []
    observed_flag_values: dict[tuple[str, str], set[str]] = defaultdict(set)
    observed_flag_counts: Counter[tuple[str, str]] = Counter()
    for row in flag_rows:
        if (
            type(row["source"]) is not str
            or type(row["stage"]) is not str
            or type(row["column"]) is not str
            or type(row["raw_value"]) is not str
            or type(row["row_count"]) is not int
            or type(row["is_blank"]) is not bool
            or type(row["domain_policy"]) is not str
            or type(row["classification"]) is not str
        ):
            raise ProfileBuildError("Flag-domain scalar type differs")
        if row["source"] not in source_by_filename:
            raise ProfileBuildError("Flag-domain source differs from locked source vocabulary")
        source_name = source_by_filename[row["source"]]
        columns = config["profiled_categorical_columns"][source_name]
        if row["column"] not in columns:
            raise ProfileBuildError("Flag-domain column differs from profiled vocabulary")
        if row["stage"] != expected_flag_stage[source_name]:
            raise ProfileBuildError("Flag-domain stage differs from population registry")
        hard_domains = config["expected"]["flag_domains"][source_name]
        expected_policy = "hard_gate" if row["column"] in hard_domains else "discovery"
        if row["domain_policy"] != expected_policy:
            raise ProfileBuildError("Flag-domain policy differs from locked domain policy")
        if row["row_count"] <= 0:
            raise ProfileBuildError("Flag-domain row_count is not a positive int64")
        if row["is_blank"] != (row["raw_value"] == ""):
            raise ProfileBuildError("Flag-domain blank classification is inconsistent")
        if row["raw_value"] == "":
            expected_classification = "blank"
        elif expected_policy == "discovery":
            expected_classification = "discovered_nonblank"
        elif row["raw_value"] in hard_domains[row["column"]]:
            expected_classification = "expected_nonblank"
        else:
            expected_classification = "unexpected_nonblank"
        if row["classification"] != expected_classification:
            raise ProfileBuildError("Flag-domain classification differs")
        if expected_classification == "unexpected_nonblank":
            raise ProfileBuildError("Unexpected hard-gate categorical value reached output")
        value_key = (source_name, row["column"])
        if row["raw_value"] in observed_flag_values[value_key]:
            raise ProfileBuildError("Flag-domain source/column/raw_value row is duplicated")
        observed_flag_values[value_key].add(row["raw_value"])
        observed_flag_counts[value_key] += row["row_count"]
        flag_order.append(
            (
                source_index[row["source"]],
                columns.index(row["column"]),
                0 if row["raw_value"] == "" else 1,
                row["raw_value"].encode("utf-8"),
            )
        )
    if flag_order != sorted(flag_order):
        raise ProfileBuildError("Flag-domain rows differ from locked deterministic order")
    for source_name in source_names:
        retained_rows = int(
            config["expected"][
                "daily_raw_key" if source_name == "daily" else "monthly_raw_key"
            ]["input_rows"]
        )
        hard_domains = config["expected"]["flag_domains"][source_name]
        for column in config["profiled_categorical_columns"][source_name]:
            key = (source_name, column)
            if key not in observed_flag_values:
                raise ProfileBuildError("Flag-domain profiled column is absent")
            if observed_flag_counts[key] != retained_rows:
                raise ProfileBuildError("Flag-domain frequencies do not conserve retained rows")
            if column in hard_domains:
                actual_nonblank = sorted(
                    (value for value in observed_flag_values[key] if value != ""),
                    key=lambda value: value.encode("utf-8"),
                )
                if actual_nonblank != list(hard_domains[column]):
                    raise ProfileBuildError("Flag-domain hard-gate literals differ")


def _csv_bytes(fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=list(fields),
        delimiter=",",
        quotechar='"',
        doublequote=True,
        escapechar=None,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _wire(row.get(field)) for field in fields})
    return handle.getvalue().encode("utf-8")


def render_quality_payloads(
    config: Mapping[str, Any],
    compiled: Mapping[str, Any],
    *,
    settings: ProfileSettings | None = None,
) -> dict[str, bytes]:
    """Render the exact nine-file immutable quality payload without writing."""
    validate_compiled_profile(
        config,
        compiled,
        config_path=settings.profile_config_path if settings is not None else None,
        settings=settings,
    )
    outputs = config["outputs"]
    schemas = outputs["schemas"]
    payloads = {
        outputs["success"]: canonical_json_bytes(compiled["summary"]),
        outputs["canonicalization_rules"]: canonical_json_bytes(
            compiled["canonicalization_rules"]
        ),
        outputs["checks"]: _csv_bytes(schemas["checks"]["columns"], compiled["checks"]),
        outputs["duplicate_column_differences"]: _csv_bytes(
            schemas["duplicate_column_differences"]["columns"],
            compiled["duplicate_column_differences"],
        ),
        outputs["duplicate_multiplicity"]: _csv_bytes(
            schemas["duplicate_multiplicity"]["columns"],
            compiled["duplicate_multiplicity"],
        ),
        outputs["duplicate_summary"]: _csv_bytes(
            schemas["duplicate_summary"]["columns"], compiled["duplicate_summary"]
        ),
        outputs["flag_domains"]: _csv_bytes(
            schemas["flag_domains"]["columns"], compiled["flag_domains"]
        ),
        outputs["input_receipt"]: canonical_json_bytes(compiled["input_receipt"]),
        outputs["summary"]: canonical_json_bytes(compiled["summary"]),
    }
    expected = set(outputs["closed_quality_file_set"])
    if set(payloads) != expected:
        raise ProfileBuildError(
            f"Rendered quality file set differs: expected {sorted(expected)}, actual {sorted(payloads)}"
        )
    return dict(sorted(payloads.items()))


def render_artifact_documents(
    settings: ProfileSettings,
    compiled: Mapping[str, Any],
    quality_directory: Path,
    quality_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    """Render manifest, inventory, and pointer objects without filesystem writes."""
    validate_compiled_profile(
        settings.config,
        compiled,
        config_path=settings.profile_config_path,
        settings=settings,
    )
    expected_names = set(settings.config["outputs"]["closed_quality_file_set"])
    if set(quality_payloads) != expected_names:
        raise ProfileBuildError("Artifact renderer received a non-closed quality file set")
    expected_quality_directory = (
        settings.quality_root
        / compiled["binding"]["contract_id"]
        / settings.config["outputs"]["quality_subdirectory"]
        / compiled["build_id"]
    ).resolve()
    if quality_directory.resolve() != expected_quality_directory:
        raise ProfileBuildError("Quality directory differs from exact locked build destination")
    try:
        quality_relative = quality_directory.resolve().relative_to(settings.database_root)
    except ValueError as exc:
        raise ProfileBuildError("Quality directory is outside configured database root") from exc
    if not quality_relative.parts or quality_relative.parts[0] != "quality":
        raise ProfileBuildError("Quality inventory paths must begin with quality/")
    outputs = []
    for filename in sorted(quality_payloads):
        path = (quality_relative / filename).as_posix()
        payload = quality_payloads[filename]
        outputs.append(
            {"path": path, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        )
    manifest = {
        "binding": compiled["binding"],
        "build_id": compiled["build_id"],
        "calendar_build_id": compiled["binding"]["calendar_build_id"],
        "contract_id": compiled["binding"]["contract_id"],
        "outputs": outputs,
        "profile_name": settings.config["profile_name"],
        "schema_version": 1,
        "status": compiled["summary"]["status"],
        "summary": compiled["summary"],
    }
    manifest_schema = settings.config["outputs"]["schemas"]["manifest"]
    _assert_exact_keys(manifest, manifest_schema["top_level_keys"], "manifest")
    _assert_exact_keys(manifest["binding"], manifest_schema["binding_keys"], "manifest.binding")
    manifest_bytes = canonical_json_bytes(manifest)
    inventory = "".join(
        f"{item['sha256']}  {item['path']}\n" for item in outputs
    ).encode("utf-8")
    artifact_relative = (
        Path(settings.config["outputs"]["artifact_subdirectory"]) / compiled["build_id"]
    )
    artifact_directory = settings.artifact_root / artifact_relative
    try:
        artifact_path = artifact_directory.resolve().relative_to(
            settings.project_root.resolve()
        ).as_posix()
    except ValueError as exc:
        raise ProfileBuildError(
            "Artifact root is outside the configured project root; pointer path cannot be relative"
        ) from exc
    pointer = {
        "artifact_path": artifact_path,
        "build_id": compiled["build_id"],
        "calendar_build_id": compiled["binding"]["calendar_build_id"],
        "contract_id": compiled["binding"]["contract_id"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "schema_version": 1,
        "status": compiled["summary"]["status"],
    }
    _assert_exact_keys(
        pointer,
        settings.config["outputs"]["schemas"]["primary_pointer"]["top_level_keys"],
        "primary pointer",
    )
    return {
        "artifact_directory": artifact_directory,
        "artifact_payloads": {
            "manifest.json": manifest_bytes,
            "output_hashes.sha256": inventory,
        },
        "manifest": manifest,
        "pointer": pointer,
        "pointer_bytes": canonical_json_bytes(pointer),
    }


def compile_profile_build(settings: ProfileSettings | None = None) -> dict[str, Any]:
    """Authenticate, scan, and compile the profile without publishing files."""
    settings = settings or load_settings()
    sealed_config_sha256 = sha256_file(settings.profile_config_path)
    sealed_implementation_sha256 = sha256_file(Path(__file__).resolve())
    with settings.profile_config_path.open("rb") as handle:
        sealed_config = tomllib.load(handle)
    if sealed_config != settings.config:
        raise ProfileBuildError("Loaded settings differ from sealed profile TOML before scan")
    upstream = authenticate_upstreams(settings)
    frozen = frozen_raw_identities(settings)
    paths = {
        source_name: settings.raw_root / settings.config["sources"][source_name]["filename"]
        for source_name in ("daily", "monthly")
    }
    before: dict[str, dict[str, int]] = {}
    for source_name, path in paths.items():
        if not path.is_file():
            raise ProfileBuildError(f"Missing raw source: {path}")
        before[source_name] = operational_stat(path)
        if before[source_name] != frozen[source_name]:
            raise ProfileBuildError(
                f"{source_name} raw identity differs from the Phase 1 final recheck before scan"
            )
        if before[source_name]["size_bytes"] != int(
            settings.config["sources"][source_name]["size_bytes"]
        ):
            raise ProfileBuildError(f"{source_name} raw size differs from locked config")
    daily = scan_daily_source(
        paths["daily"], settings.config, block_size_bytes=settings.block_size_bytes
    )
    monthly = scan_monthly_source(
        paths["monthly"], settings.config, block_size_bytes=settings.block_size_bytes
    )
    after = {source_name: operational_stat(path) for source_name, path in paths.items()}
    for source_name in ("daily", "monthly"):
        if after[source_name] != before[source_name]:
            raise ProfileBuildError(f"{source_name} raw identity changed during profile scan")
    if sha256_file(settings.profile_config_path) != sealed_config_sha256:
        raise ProfileBuildError("Profile config bytes changed during the raw scans")
    if sha256_file(Path(__file__).resolve()) != sealed_implementation_sha256:
        raise ProfileBuildError("Primary implementation bytes changed during the raw scans")
    return assemble_profile_outputs(
        settings,
        upstream,
        frozen,
        before,
        after,
        daily,
        monthly,
        sealed_config_sha256=sealed_config_sha256,
        sealed_implementation_sha256=sealed_implementation_sha256,
    )
