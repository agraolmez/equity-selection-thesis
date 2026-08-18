"""Prepare the immutable, development-only matrices consumed by Stage 9 fits.

This module performs preprocessing only.  It has no model-framework import and
no interface for portfolio weights, trades, P&L, validation data, or final-test
data.  Every cache is bound to authenticated Stage 6--8 inputs, frozen Stage 9
configuration, and the exact preprocessing code that produced it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import platform
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.selectors.cache import atomic_write_json, canonical_json_bytes
from portfolio_bot.selectors.contract import (
    BASE_MATRIX_WIDTH,
    CONTRACT_ID,
    DEFAULT_POLICY,
    ELASTIC_NET_INTERACTION_COUNT,
    ELASTIC_NET_TOTAL_WIDTH,
    PROJECT_ROOT,
    REQUIRED_INPUT_NAMES,
    SelectorContractError,
    project_path,
    require,
    sha256_file,
    validate_contract,
    validate_runtime_versions,
)
from portfolio_bot.selectors.inputs import (
    LABEL_COLUMNS,
    SPLIT_COLUMNS,
    Stage9Inputs,
    reauthenticate_stage9_inputs,
    resolve_stage9_inputs,
)
from portfolio_bot.selectors.preprocessing import (
    SelectorMatrixSchema,
    build_elastic_net_interaction_registry,
    build_selector_matrix,
    build_selector_schema,
)
from portfolio_bot.selectors.splits import FoldIndices, build_fold_indices


EXPECTED_ROWS = 286_628
EXPECTED_SPLIT_ROWS = 712
EXPECTED_AGGREGATE_HISTORY_ROWS = 178 * 23
EXPECTED_FOLDS = ("dev_fold_1", "dev_fold_2", "dev_fold_3", "dev_fold_4")
PARQUET_ROW_GROUP_SIZE = 65_536
INTERACTION_ROW_CHUNK = 4_096

_CODE_PATHS = (
    "src/portfolio_bot/selectors/contract.py",
    "src/portfolio_bot/selectors/inputs.py",
    "src/portfolio_bot/selectors/preprocessing.py",
    "src/portfolio_bot/selectors/splits.py",
    "src/portfolio_bot/selectors/weights.py",
    "src/portfolio_bot/selectors/cache.py",
    "src/portfolio_bot/selectors/materialization.py",
    "src/portfolio_bot/selectors/prepared_data.py",
    "src/portfolio_bot/features/feature_math.py",
)


@dataclass(frozen=True)
class PreparedBuildIdentity:
    """External authority used to authenticate a prepared-matrix cache."""

    contract_id: str
    build_id: str
    content_fingerprint: str
    configuration_hashes: tuple[tuple[str, str], ...]
    code_hashes: tuple[tuple[str, str], ...]
    runtime_versions: tuple[tuple[str, str], ...]
    input_receipts: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "build_id": self.build_id,
            "content_fingerprint": self.content_fingerprint,
            "configuration_hashes": dict(self.configuration_hashes),
            "code_hashes": dict(self.code_hashes),
            "runtime_versions": dict(self.runtime_versions),
            "input_receipts": [dict(value) for value in self.input_receipts],
        }


@dataclass(frozen=True)
class PreparedMaterialization:
    identity: PreparedBuildIdentity
    root: Path
    metadata: Mapping[str, Any]
    resumed_complete_build: bool


def _relative_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _configuration_paths(policy: Mapping[str, Any], policy_path: Path) -> tuple[Path, ...]:
    values = {
        policy_path.resolve(),
        project_path(policy["physical_rules"]["path"]),
        project_path(policy["methodology"]["source"]),
        project_path(policy["methodology"]["freeze_receipt"]),
        project_path(policy["authority"]["verified_stage08_pointer"]),
        project_path(policy["verified_feature_store"]["verified_pointer"]),
        project_path(policy["verified_feature_store"]["primary_manifest"]),
        project_path(policy["verified_label_store"]["verified_pointer"]),
        project_path(policy["verified_label_store"]["primary_manifest"]),
        project_path(policy["verified_stage08_key_audit"]["primary_manifest"]),
        project_path(policy["database"]["paths_config"]),
        project_path(policy["database"]["relocation_receipt"]),
    }
    result = tuple(sorted(values, key=lambda value: value.as_posix()))
    for path in result:
        require(path.is_file() and not path.is_symlink(), f"Unsafe Stage 9 identity file: {path}")
    return result


def compute_prepared_build_identity(
    inputs: Stage9Inputs,
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> PreparedBuildIdentity:
    """Return a deterministic identity for the exact prepared-matrix build."""

    policy_path = policy_path.expanduser().resolve()
    policy, _ = validate_contract(policy_path)
    require(inputs.hashes_authenticated, "Prepared Stage 9 identity requires authenticated inputs")
    require(
        tuple(inputs.tables) == REQUIRED_INPUT_NAMES,
        "Prepared Stage 9 identity authenticated table set differs",
    )
    require(
        tuple(str(value.get("name")) for value in inputs.input_receipts)
        == REQUIRED_INPUT_NAMES,
        "Prepared Stage 9 identity input receipt set differs",
    )
    require(
        tuple(dict(inputs.tables[name].receipt()) for name in REQUIRED_INPUT_NAMES)
        == tuple(dict(value) for value in inputs.input_receipts),
        "Prepared Stage 9 table and input receipts disagree",
    )
    configuration_hashes = tuple(
        (_relative_name(path), sha256_file(path))
        for path in _configuration_paths(policy, policy_path)
    )
    code_hashes: list[tuple[str, str]] = []
    for relative in _CODE_PATHS:
        path = PROJECT_ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"Missing Stage 9 code dependency: {relative}")
        code_hashes.append((relative, sha256_file(path)))
    runtime_versions = tuple(
        sorted(
            {
                "python": platform.python_version(),
                "numpy": importlib.metadata.version("numpy"),
                "pandas": importlib.metadata.version("pandas"),
                "pyarrow": importlib.metadata.version("pyarrow"),
                "scipy": importlib.metadata.version("scipy"),
                "scikit-learn": importlib.metadata.version("scikit-learn"),
            }.items()
        )
    )
    payload = {
        "schema_version": 2,
        "purpose": "stage09_prepared_selector_matrices_only",
        "contract_id": CONTRACT_ID,
        "configuration_hashes": dict(configuration_hashes),
        "code_hashes": dict(code_hashes),
        "runtime_versions": dict(runtime_versions),
        "input_receipts": [dict(value) for value in inputs.input_receipts],
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return PreparedBuildIdentity(
        contract_id=CONTRACT_ID,
        build_id=f"phase04-selector-prepared-{fingerprint[:24]}",
        content_fingerprint=fingerprint,
        configuration_hashes=configuration_hashes,
        code_hashes=tuple(code_hashes),
        runtime_versions=runtime_versions,
        input_receipts=tuple(inputs.input_receipts),
    )


def prepared_cache_root(
    inputs: Stage9Inputs,
    identity: PreparedBuildIdentity,
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> Path:
    """Resolve the content-addressed production cache location."""

    policy, _ = validate_contract(policy_path.expanduser().resolve())
    relative = Path(str(policy["database"]["cache_root"]))
    require(relative.parts and not relative.is_absolute() and ".." not in relative.parts, "Unsafe Stage 9 cache root")
    return inputs.database_root / relative / identity.build_id / "prepared"


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _receipt_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.receipt.json")


def _all_finite_npy(values: np.ndarray, *, chunk_rows: int = 32_768) -> bool:
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        return False
    for start in range(0, values.shape[0], chunk_rows):
        if not np.isfinite(values[start : start + chunk_rows]).all():
            return False
    return True


def _receipt_matches_payload(
    path: Path,
    *,
    kind: str,
    receipt: Mapping[str, Any],
) -> bool:
    """Independently re-read a payload instead of trusting its sidecar claims."""

    try:
        if path.stat().st_size <= 0:
            return False
        if kind == "npy":
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            return (
                receipt.get("shape") == list(values.shape)
                and receipt.get("dtype") == values.dtype.name
                and receipt.get("all_finite") is True
                and _all_finite_npy(values)
            )
        if kind == "parquet":
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            return (
                parquet.metadata is not None
                and parquet.metadata.num_rows > 0
                and parquet.metadata.num_columns > 0
                and receipt.get("rows") == parquet.metadata.num_rows
                and receipt.get("columns") == parquet.metadata.num_columns
                and receipt.get("schema_sha256") == _schema_sha256(schema)
            )
        if kind == "npz":
            with np.load(path, allow_pickle=False) as archive:
                names = sorted(archive.files)
                if not names:
                    return False
                shapes = {name: list(archive[name].shape) for name in names}
                dtypes = {name: archive[name].dtype.name for name in names}
            return receipt.get("arrays") == shapes and receipt.get("dtypes") == dtypes
    except Exception:
        return False
    return False


def _safe_directory(path: Path) -> None:
    candidate = Path(os.path.abspath(path.expanduser()))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise SelectorContractError(
                    f"Unsafe Stage 9 prepared-cache path component: {current}"
                )
        else:
            current.mkdir(mode=0o700)


def _load_valid_receipt(
    path: Path,
    *,
    identity: PreparedBuildIdentity,
    kind: str,
) -> dict[str, Any] | None:
    receipt_path = _receipt_path(path)
    if not path.is_file() or path.is_symlink() or not receipt_path.is_file() or receipt_path.is_symlink():
        return None
    try:
        receipt = json.loads(receipt_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        size_bytes = path.stat().st_size
        digest = sha256_file(path)
    except OSError:
        return None
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "COMPLETE"
        or receipt.get("name") != path.name
        or receipt.get("kind") != kind
        or receipt.get("build_id") != identity.build_id
        or receipt.get("content_fingerprint") != identity.content_fingerprint
        or not isinstance(receipt.get("size_bytes"), int)
        or receipt.get("size_bytes") <= 0
        or receipt.get("size_bytes") != size_bytes
        or receipt.get("sha256") != digest
        or not _receipt_matches_payload(path, kind=kind, receipt=receipt)
    ):
        return None
    return receipt


def _publish_generated_file(
    path: Path,
    *,
    identity: PreparedBuildIdentity,
    kind: str,
    producer: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    """Generate once, atomically publish, and sidecar-authenticate one file."""

    valid = _load_valid_receipt(path, identity=identity, kind=kind)
    if valid is not None:
        return valid
    _safe_directory(path.parent)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        details = dict(producer(temporary))
        if not temporary.is_file() or temporary.is_symlink():
            raise RuntimeError(f"Stage 9 producer did not create a safe file: {path.name}")
        digest = sha256_file(temporary)
        if path.exists():
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise RuntimeError(f"Existing content-addressed Stage 9 file differs: {path}")
            temporary.unlink()
        else:
            os.replace(temporary, path)
        receipt = {
            "schema_version": 1,
            "status": "COMPLETE",
            "name": path.name,
            "kind": kind,
            "build_id": identity.build_id,
            "content_fingerprint": identity.content_fingerprint,
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            **details,
        }
        atomic_write_json(
            _receipt_path(path),
            receipt,
            trusted_root=path.parent,
        )
        return receipt
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_parquet(path: Path, value: pa.Table | pd.DataFrame) -> dict[str, Any]:
    table = value if isinstance(value, pa.Table) else pa.Table.from_pandas(value, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
        write_statistics=True,
        version="2.6",
    )
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "rows": table.num_rows,
        "columns": table.num_columns,
        "schema_sha256": _schema_sha256(table.schema),
    }


def _write_npy(path: Path, values: np.ndarray) -> dict[str, Any]:
    source = np.asarray(values)
    target = np.lib.format.open_memmap(path, mode="w+", dtype=source.dtype, shape=source.shape)
    target[...] = source
    target.flush()
    del target
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"shape": list(source.shape), "dtype": source.dtype.name, "all_finite": bool(np.isfinite(source).all())}


def _write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Write an NPZ whose bytes do not depend on wall-clock ZIP timestamps."""

    names = sorted(arrays)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        strict_timestamps=True,
    ) as archive:
        for name in names:
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"Unsafe prepared support-array name: {name!r}")
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asarray(arrays[name]),
                allow_pickle=False,
            )
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "arrays": {name: list(np.asarray(arrays[name]).shape) for name in names},
        "dtypes": {name: np.asarray(arrays[name]).dtype.name for name in names},
    }


def _interaction_positions(
    schema: SelectorMatrixSchema,
    registry: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    positions = {name: index for index, name in enumerate(schema.column_names)}
    firm = registry["firm_column"].map(positions)
    aggregate = registry["aggregate_column"].map(positions)
    if firm.isna().any() or aggregate.isna().any():
        raise ValueError("Elastic Net interaction registry escaped the base schema")
    return firm.to_numpy(np.int64), aggregate.to_numpy(np.int64)


def _interaction_support(
    base_matrix: np.ndarray,
    valid_training_mask: np.ndarray,
    firm_positions: np.ndarray,
    aggregate_positions: np.ndarray,
    *,
    chunk_rows: int = INTERACTION_ROW_CHUNK,
) -> np.ndarray:
    minimum = np.full(firm_positions.size, np.inf, dtype=np.float64)
    maximum = np.full(firm_positions.size, -np.inf, dtype=np.float64)
    training_rows = np.flatnonzero(valid_training_mask)
    if training_rows.size == 0:
        raise ValueError("Cannot compute interaction support without valid training rows")
    for start in range(0, training_rows.size, chunk_rows):
        indices = training_rows[start : start + chunk_rows]
        block = np.asarray(base_matrix[indices], dtype=np.float32)
        values = block[:, firm_positions] * block[:, aggregate_positions]
        minimum = np.minimum(minimum, values.min(axis=0).astype(np.float64))
        maximum = np.maximum(maximum, values.max(axis=0).astype(np.float64))
    return maximum > minimum


def _write_elastic_matrix(
    path: Path,
    base_matrix: np.ndarray,
    firm_positions: np.ndarray,
    aggregate_positions: np.ndarray,
    *,
    chunk_rows: int = INTERACTION_ROW_CHUNK,
) -> dict[str, Any]:
    """Write base then continuous-major/aggregate-minor products in chunks."""

    rows, base_width = base_matrix.shape
    interaction_count = firm_positions.size
    target = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(rows, base_width + interaction_count),
    )
    all_finite = True
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        base = np.asarray(base_matrix[start:stop], dtype=np.float32)
        interactions = base[:, firm_positions] * base[:, aggregate_positions]
        target[start:stop, :base_width] = base
        target[start:stop, base_width:] = interactions
        all_finite = all_finite and bool(np.isfinite(interactions).all())
    target.flush()
    del target
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "shape": [rows, base_width + interaction_count],
        "dtype": "float32",
        "all_finite": all_finite and bool(np.isfinite(base_matrix).all()),
        "base_prefix_columns": base_width,
        "interaction_columns": interaction_count,
    }


def _exact_keys(left: pd.DataFrame, right: pd.DataFrame, *, label: str) -> None:
    if len(left) != len(right):
        raise SelectorContractError(f"{label} row counts differ")
    for name in ("signal_date", "permno"):
        if not np.array_equal(left[name].to_numpy(), right[name].to_numpy()):
            raise SelectorContractError(f"{label} keys differ at {name}")


def _index_mask(indices: np.ndarray, row_count: int) -> np.ndarray:
    result = np.zeros(row_count, dtype=bool)
    result[np.asarray(indices, dtype=np.int64)] = True
    return result


def _elastic_support_receipt(
    schema: SelectorMatrixSchema,
    interaction_registry: pd.DataFrame,
    base_support: np.ndarray,
    interaction_support: np.ndarray,
) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "elastic_column_ordinal": np.arange(1, schema.width + 1, dtype=np.int64),
            "column_name": schema.column_names,
            "component": "base",
            "active": np.asarray(base_support, dtype=bool),
        }
    )
    interactions = pd.DataFrame(
        {
            "elastic_column_ordinal": schema.width
            + interaction_registry["interaction_ordinal"].to_numpy(np.int64),
            "column_name": interaction_registry["interaction_column"].astype(str),
            "component": "continuous_by_aggregate_interaction",
            "active": np.asarray(interaction_support, dtype=bool),
        }
    )
    return pd.concat([base, interactions], ignore_index=True)


def _expected_prepared_files() -> set[str]:
    files = {
        "rows.parquet",
        "split_membership.parquet",
        "base_column_dictionary.parquet",
        "elastic_interaction_dictionary.parquet",
        "aggregate_history.parquet",
        "support_masks.npz",
    }
    for fold in EXPECTED_FOLDS:
        files.update(
            {
                f"base_matrix_inner_{fold}.npy",
                f"base_matrix_full_{fold}.npy",
                f"elastic_matrix_full_{fold}.npy",
                f"category_mapping_inner_{fold}.parquet",
                f"category_mapping_full_{fold}.parquet",
                f"base_support_inner_{fold}.parquet",
                f"base_support_full_{fold}.parquet",
                f"elastic_support_full_{fold}.parquet",
            }
        )
    return files


def _prepared_file_kind(name: str) -> str:
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".npy"):
        return "npy"
    if name == "support_masks.npz":
        return "npz"
    raise ValueError(f"Unexpected Stage 9 prepared file: {name}")


def _expected_parquet_shape(name: str) -> tuple[int, int] | None:
    if name == "rows.parquet":
        return EXPECTED_ROWS, len(LABEL_COLUMNS)
    if name == "split_membership.parquet":
        return EXPECTED_SPLIT_ROWS, len(SPLIT_COLUMNS)
    if name == "base_column_dictionary.parquet":
        return BASE_MATRIX_WIDTH, 8
    if name == "elastic_interaction_dictionary.parquet":
        return ELASTIC_NET_INTERACTION_COUNT, 6
    if name == "aggregate_history.parquet":
        return EXPECTED_AGGREGATE_HISTORY_ROWS, 9
    if name.startswith("category_mapping_"):
        return 245, 10
    if name.startswith("base_support_"):
        return BASE_MATRIX_WIDTH, 12
    if name.startswith("elastic_support_full_"):
        return ELASTIC_NET_TOTAL_WIDTH, 4
    return None


def _prepared_payload_contract_is_valid(
    root: Path,
    files: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Validate actual payload shapes and nested preprocessing roles."""

    expected_support_names = {
        *(f"{fold}__base_inner" for fold in EXPECTED_FOLDS),
        *(f"{fold}__base_full" for fold in EXPECTED_FOLDS),
        *(f"{fold}__elastic_full" for fold in EXPECTED_FOLDS),
    }
    try:
        for name, receipt in files.items():
            if name.endswith(".parquet"):
                expected_shape = _expected_parquet_shape(name)
                if expected_shape is None or (
                    receipt.get("rows"), receipt.get("columns")
                ) != expected_shape:
                    return False
            elif name.endswith(".npy"):
                expected_shape = (
                    [EXPECTED_ROWS, ELASTIC_NET_TOTAL_WIDTH]
                    if name.startswith("elastic_matrix_full_")
                    else [EXPECTED_ROWS, BASE_MATRIX_WIDTH]
                )
                if (
                    receipt.get("shape") != expected_shape
                    or receipt.get("dtype") != "float32"
                    or receipt.get("all_finite") is not True
                ):
                    return False
                if name.startswith("elastic_matrix_full_") and (
                    receipt.get("base_prefix_columns") != BASE_MATRIX_WIDTH
                    or receipt.get("interaction_columns")
                    != ELASTIC_NET_INTERACTION_COUNT
                ):
                    return False
            elif name == "support_masks.npz":
                arrays = receipt.get("arrays")
                dtypes = receipt.get("dtypes")
                if (
                    not isinstance(arrays, dict)
                    or not isinstance(dtypes, dict)
                    or set(arrays) != expected_support_names
                    or set(dtypes) != expected_support_names
                ):
                    return False
                for key in expected_support_names:
                    expected_width = (
                        ELASTIC_NET_TOTAL_WIDTH
                        if key.endswith("__elastic_full")
                        else BASE_MATRIX_WIDTH
                    )
                    if arrays[key] != [expected_width] or dtypes[key] != "bool":
                        return False

        # The full-training base matrix must be the exact prefix of the
        # full-training Elastic Net design, not merely asserted in metadata.
        for fold in EXPECTED_FOLDS:
            base = np.load(
                root / f"base_matrix_full_{fold}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            elastic = np.load(
                root / f"elastic_matrix_full_{fold}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for start in range(0, EXPECTED_ROWS, 32_768):
                stop = min(start + 32_768, EXPECTED_ROWS)
                if not np.array_equal(
                    base[start:stop], elastic[start:stop, :BASE_MATRIX_WIDTH]
                ):
                    return False
    except (OSError, ValueError, TypeError):
        return False
    return True


def _metadata_is_complete(
    root: Path,
    identity: PreparedBuildIdentity,
) -> dict[str, Any] | None:
    path = root / "metadata.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        return None
    try:
        metadata = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_files = _expected_prepared_files()
    expected_metadata_keys = {
        "schema_version",
        "status",
        "contract_id",
        "build_id",
        "content_fingerprint",
        "configuration_hashes",
        "code_hashes",
        "runtime_versions",
        "input_receipts",
        "scope",
        "rows",
        "fold_ids",
        "preprocessing_fit_scopes",
        "inner_base_matrix_shape",
        "full_base_matrix_shape",
        "elastic_full_matrix_shape",
        "matrix_dtype",
        "elastic_full_base_prefix_source",
        "base_prefix_identical_in_elastic_full_matrix",
        "elastic_interaction_order",
        "required_files",
        "files",
    }
    if (
        set(metadata) != expected_metadata_keys
        or metadata.get("schema_version") != 2
        or metadata.get("status") != "PREPARED_COMPLETE"
        or metadata.get("build_id") != identity.build_id
        or metadata.get("content_fingerprint") != identity.content_fingerprint
        or metadata.get("contract_id") != identity.contract_id
        or metadata.get("configuration_hashes") != dict(identity.configuration_hashes)
        or metadata.get("code_hashes") != dict(identity.code_hashes)
        or metadata.get("runtime_versions") != dict(identity.runtime_versions)
        or metadata.get("input_receipts")
        != [dict(value) for value in identity.input_receipts]
        or metadata.get("rows") != EXPECTED_ROWS
        or metadata.get("fold_ids") != list(EXPECTED_FOLDS)
        or metadata.get("inner_base_matrix_shape")
        != [EXPECTED_ROWS, BASE_MATRIX_WIDTH]
        or metadata.get("full_base_matrix_shape")
        != [EXPECTED_ROWS, BASE_MATRIX_WIDTH]
        or metadata.get("elastic_full_matrix_shape")
        != [EXPECTED_ROWS, ELASTIC_NET_TOTAL_WIDTH]
        or metadata.get("matrix_dtype") != "float32"
        or metadata.get("elastic_full_base_prefix_source") != "full_outer_train"
        or metadata.get("base_prefix_identical_in_elastic_full_matrix") is not True
        or metadata.get("elastic_interaction_order")
        != "continuous_firm_registry_ordinal_major_then_aggregate_state_registry_ordinal_minor"
    ):
        return None
    if metadata.get("scope") != {
        "development_only": True,
        "model_fits_performed": False,
        "stage10_or_later_permitted": False,
        "portfolio_or_backtest_permitted": False,
        "validation_or_final_payload_accessed": False,
    }:
        return None
    if metadata.get("preprocessing_fit_scopes") != {
        "inner": {
            "fitted_on": "inner_train_only",
            "excluded": ["inner_purge_month", "inner_validation", "outer_validation"],
            "consumer": "neural_network_and_xgboost_early_stopping_only",
        },
        "full": {
            "fitted_on": "full_outer_training_only",
            "excluded": ["outer_validation"],
            "consumer": "neural_network_and_xgboost_refit_and_oof_plus_elastic_net",
        },
    }:
        return None
    files = metadata.get("files")
    if (
        not isinstance(files, dict)
        or set(files) != expected_files
        or set(metadata.get("required_files", [])) != expected_files
    ):
        return None
    for name in expected_files:
        if not isinstance(files[name], dict):
            return None
        observed = _load_valid_receipt(
            root / name,
            identity=identity,
            kind=_prepared_file_kind(name),
        )
        if observed is None or observed != files[name]:
            return None
    if not _prepared_payload_contract_is_valid(root, files):
        return None
    expected_names = {
        "metadata.json",
        *expected_files,
        *(_receipt_path(root / name).name for name in expected_files),
    }
    if {path.name for path in root.iterdir()} != expected_names:
        return None
    return metadata


def prepare_stage9_selector_matrices(
    *,
    inputs: Stage9Inputs | None = None,
    policy_path: Path = DEFAULT_POLICY,
    output_root: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> PreparedMaterialization:
    """Materialize four fold-local base and Elastic Net design matrices.

    ``output_root`` exists for isolated tests.  Production callers should omit
    it so the cache is written beneath the frozen database cache namespace.
    Metadata is the final file published; a complete cache is never inferred
    from the presence of matrix files alone.
    """

    policy_path = policy_path.expanduser().resolve()
    policy, _ = validate_contract(policy_path)
    validate_runtime_versions()
    catalog = resolve_stage9_inputs(policy_path) if inputs is None else inputs
    reauthenticate_stage9_inputs(catalog)
    identity = compute_prepared_build_identity(catalog, policy_path=policy_path)
    root = (
        prepared_cache_root(catalog, identity, policy_path=policy_path)
        if output_root is None
        else Path(os.path.abspath(output_root.expanduser()))
    )
    _safe_directory(root)
    existing = _metadata_is_complete(root, identity)
    if existing is not None:
        return PreparedMaterialization(identity, root, existing, True)

    emit = progress if progress is not None else (lambda _message: None)
    emit("authenticated Stage 6--8 inputs and computed prepared-build identity")
    dictionary_table = catalog.read_feature_dictionary()
    catalog.read_feature_set_membership()
    vocabulary_table = catalog.read_category_vocabulary()
    aggregate_history_table = catalog.read_aggregate_signal_history()
    feature_table = catalog.read_development_feature_panel()
    label_table = catalog.read_development_labels()
    split_table = catalog.read_split_membership()
    stage8_keys = catalog.read_stage8_key_audit()

    features = feature_table.to_pandas()
    labels = label_table.to_pandas()
    splits = split_table.to_pandas()
    _exact_keys(features, labels, label="Stage 6 feature/Stage 7 label")
    _exact_keys(labels, stage8_keys.to_pandas(), label="Stage 7 label/Stage 8 key-audit")
    valid_target = (
        labels["primary_label_status"].eq("PRESENT").to_numpy(bool)
        & np.isfinite(pd.to_numeric(labels["residual_return_rank"], errors="coerce").to_numpy(np.float64))
    )
    require(int(valid_target.sum()) == int(policy["scope"]["required_present_primary_labels"]), "Prepared target-valid count differs")
    folds = build_fold_indices(labels, splits)
    require(tuple(fold.fold_id for fold in folds) == EXPECTED_FOLDS, "Prepared fold order differs")

    schema = build_selector_schema(dictionary_table, vocabulary_table)
    interaction_registry = build_elastic_net_interaction_registry(schema).drop(columns="active")
    require(schema.width == BASE_MATRIX_WIDTH, "Prepared base width differs")
    require(len(interaction_registry) == ELASTIC_NET_INTERACTION_COUNT, "Prepared interaction width differs")
    firm_positions, aggregate_positions = _interaction_positions(schema, interaction_registry)

    receipts: dict[str, dict[str, Any]] = {}

    def parquet(name: str, value: pa.Table | pd.DataFrame) -> None:
        receipts[name] = _publish_generated_file(
            root / name,
            identity=identity,
            kind="parquet",
            producer=lambda path, value=value: _write_parquet(path, value),
        )

    parquet("rows.parquet", label_table)
    parquet("split_membership.parquet", split_table)
    parquet("base_column_dictionary.parquet", schema.receipt())
    parquet("elastic_interaction_dictionary.parquet", interaction_registry)

    inner_base_masks: dict[str, np.ndarray] = {}
    full_base_masks: dict[str, np.ndarray] = {}
    elastic_full_masks: dict[str, np.ndarray] = {}
    aggregate_name = "aggregate_history.parquet"
    aggregate_cached = _load_valid_receipt(
        root / aggregate_name,
        identity=identity,
        kind="parquet",
    )
    aggregate_receipt_written = aggregate_cached is not None
    if aggregate_cached is not None:
        receipts[aggregate_name] = aggregate_cached
    for fold_number, fold in enumerate(folds, start=1):
        fold_kinds = {
            f"base_matrix_inner_{fold.fold_id}.npy": "npy",
            f"base_matrix_full_{fold.fold_id}.npy": "npy",
            f"elastic_matrix_full_{fold.fold_id}.npy": "npy",
            f"category_mapping_inner_{fold.fold_id}.parquet": "parquet",
            f"category_mapping_full_{fold.fold_id}.parquet": "parquet",
            f"base_support_inner_{fold.fold_id}.parquet": "parquet",
            f"base_support_full_{fold.fold_id}.parquet": "parquet",
            f"elastic_support_full_{fold.fold_id}.parquet": "parquet",
        }
        cached = {
            name: _load_valid_receipt(root / name, identity=identity, kind=kind)
            for name, kind in fold_kinds.items()
        }
        if aggregate_receipt_written and all(value is not None for value in cached.values()):
            receipts.update({name: value for name, value in cached.items() if value is not None})
            for scope in ("inner", "full"):
                base_receipt = cached[f"base_matrix_{scope}_{fold.fold_id}.npy"]
                if (
                    base_receipt.get("shape") != [EXPECTED_ROWS, BASE_MATRIX_WIDTH]
                    or base_receipt.get("dtype") != "float32"
                ):
                    raise RuntimeError(
                        f"Cached {scope} prepared base receipt differs for {fold.fold_id}"
                    )
            elastic_receipt = cached[f"elastic_matrix_full_{fold.fold_id}.npy"]
            if (
                elastic_receipt.get("shape") != [EXPECTED_ROWS, ELASTIC_NET_TOTAL_WIDTH]
                or elastic_receipt.get("dtype") != "float32"
            ):
                raise RuntimeError(f"Cached prepared Elastic receipt differs for {fold.fold_id}")
            inner_base_masks[fold.fold_id] = pq.read_table(
                root / f"base_support_inner_{fold.fold_id}.parquet",
                columns=["active"],
                use_threads=False,
            ).to_pandas()["active"].to_numpy(bool)
            full_base_masks[fold.fold_id] = pq.read_table(
                root / f"base_support_full_{fold.fold_id}.parquet",
                columns=["active"],
                use_threads=False,
            ).to_pandas()["active"].to_numpy(bool)
            elastic_full_masks[fold.fold_id] = pq.read_table(
                root / f"elastic_support_full_{fold.fold_id}.parquet",
                columns=["active"],
                use_threads=False,
            ).to_pandas()["active"].to_numpy(bool)
            require(inner_base_masks[fold.fold_id].shape == (BASE_MATRIX_WIDTH,), f"{fold.fold_id} cached inner base support differs")
            require(full_base_masks[fold.fold_id].shape == (BASE_MATRIX_WIDTH,), f"{fold.fold_id} cached full base support differs")
            require(elastic_full_masks[fold.fold_id].shape == (ELASTIC_NET_TOTAL_WIDTH,), f"{fold.fold_id} cached Elastic support differs")
            emit(f"complete {fold.fold_id} (authenticated resume)")
            continue
        emit(f"preprocessing {fold.fold_id} inner-train fit ({fold_number}/4)")
        inner_result = build_selector_matrix(
            features,
            dictionary_table,
            vocabulary_table,
            outer_training_mask=_index_mask(fold.inner_train, len(features)),
            valid_target_mask=valid_target,
            aggregate_history_table=aggregate_history_table,
        )
        _exact_keys(inner_result.row_keys, labels, label=f"{fold.fold_id} inner prepared/label")
        require(inner_result.frozen_matrix.shape == (EXPECTED_ROWS, BASE_MATRIX_WIDTH), f"{fold.fold_id} inner base shape differs")
        require(inner_result.frozen_matrix.dtype == np.float32 and np.isfinite(inner_result.frozen_matrix).all(), f"{fold.fold_id} inner base dtype or finiteness differs")
        inner_base_support = np.asarray(inner_result.support_mask, dtype=bool)
        require(
            int(inner_result.valid_outer_training_scoring_mask.sum()) == fold.inner_train.size,
            f"{fold.fold_id} inner preprocessing fit domain differs",
        )
        inner_base_masks[fold.fold_id] = inner_base_support

        inner_base_name = f"base_matrix_inner_{fold.fold_id}.npy"
        receipts[inner_base_name] = _publish_generated_file(
            root / inner_base_name,
            identity=identity,
            kind="npy",
            producer=lambda path, matrix=inner_result.frozen_matrix: _write_npy(path, matrix),
        )
        parquet(
            f"category_mapping_inner_{fold.fold_id}.parquet",
            inner_result.category_mapping_receipt,
        )
        parquet(
            f"base_support_inner_{fold.fold_id}.parquet",
            inner_result.support_receipt,
        )
        if not aggregate_receipt_written:
            parquet(aggregate_name, inner_result.aggregate_history_receipt)
            aggregate_receipt_written = True
        del inner_result

        emit(f"preprocessing {fold.fold_id} full-outer-train fit ({fold_number}/4)")
        full_result = build_selector_matrix(
            features,
            dictionary_table,
            vocabulary_table,
            outer_training_mask=_index_mask(fold.full_train, len(features)),
            valid_target_mask=valid_target,
            aggregate_history_table=aggregate_history_table,
        )
        _exact_keys(full_result.row_keys, labels, label=f"{fold.fold_id} full prepared/label")
        require(full_result.frozen_matrix.shape == (EXPECTED_ROWS, BASE_MATRIX_WIDTH), f"{fold.fold_id} full base shape differs")
        require(full_result.frozen_matrix.dtype == np.float32 and np.isfinite(full_result.frozen_matrix).all(), f"{fold.fold_id} full base dtype or finiteness differs")
        require(
            int(full_result.valid_outer_training_scoring_mask.sum()) == fold.full_train.size,
            f"{fold.fold_id} full preprocessing fit domain differs",
        )

        interaction_support = _interaction_support(
            full_result.frozen_matrix,
            full_result.valid_outer_training_scoring_mask,
            firm_positions,
            aggregate_positions,
        )
        full_base_support = np.asarray(full_result.support_mask, dtype=bool)
        interaction_support &= (
            full_base_support[firm_positions] & full_base_support[aggregate_positions]
        )
        elastic_full_support = np.concatenate([full_base_support, interaction_support])
        require(elastic_full_support.shape == (ELASTIC_NET_TOTAL_WIDTH,), f"{fold.fold_id} Elastic Net support shape differs")
        full_base_masks[fold.fold_id] = full_base_support
        elastic_full_masks[fold.fold_id] = elastic_full_support

        full_base_name = f"base_matrix_full_{fold.fold_id}.npy"
        receipts[full_base_name] = _publish_generated_file(
            root / full_base_name,
            identity=identity,
            kind="npy",
            producer=lambda path, matrix=full_result.frozen_matrix: _write_npy(path, matrix),
        )
        elastic_name = f"elastic_matrix_full_{fold.fold_id}.npy"
        receipts[elastic_name] = _publish_generated_file(
            root / elastic_name,
            identity=identity,
            kind="npy",
            producer=lambda path, matrix=full_result.frozen_matrix: _write_elastic_matrix(
                path, matrix, firm_positions, aggregate_positions
            ),
        )
        require(receipts[inner_base_name]["shape"] == [EXPECTED_ROWS, BASE_MATRIX_WIDTH], f"{fold.fold_id} inner base receipt differs")
        require(receipts[full_base_name]["shape"] == [EXPECTED_ROWS, BASE_MATRIX_WIDTH], f"{fold.fold_id} full base receipt differs")
        require(receipts[elastic_name]["shape"] == [EXPECTED_ROWS, ELASTIC_NET_TOTAL_WIDTH], f"{fold.fold_id} Elastic Net receipt differs")

        parquet(f"category_mapping_full_{fold.fold_id}.parquet", full_result.category_mapping_receipt)
        parquet(f"base_support_full_{fold.fold_id}.parquet", full_result.support_receipt)
        parquet(
            f"elastic_support_full_{fold.fold_id}.parquet",
            _elastic_support_receipt(
                schema,
                interaction_registry,
                full_base_support,
                interaction_support,
            ),
        )
        del full_result
        emit(f"complete {fold.fold_id}")

    support_name = "support_masks.npz"
    support_arrays = {
        **{f"{fold}__base_inner": mask for fold, mask in inner_base_masks.items()},
        **{f"{fold}__base_full": mask for fold, mask in full_base_masks.items()},
        **{f"{fold}__elastic_full": mask for fold, mask in elastic_full_masks.items()},
    }
    receipts[support_name] = _publish_generated_file(
        root / support_name,
        identity=identity,
        kind="npz",
        producer=lambda path: _write_deterministic_npz(path, support_arrays),
    )
    require(
        set(receipts) == _expected_prepared_files(),
        "Prepared Stage 9 publication set is not closed",
    )

    reauthenticate_stage9_inputs(catalog)
    required_files = sorted(receipts)
    metadata = {
        "schema_version": 2,
        "status": "PREPARED_COMPLETE",
        **identity.as_dict(),
        "scope": {
            "development_only": True,
            "model_fits_performed": False,
            "stage10_or_later_permitted": False,
            "portfolio_or_backtest_permitted": False,
            "validation_or_final_payload_accessed": False,
        },
        "rows": EXPECTED_ROWS,
        "fold_ids": list(EXPECTED_FOLDS),
        "preprocessing_fit_scopes": {
            "inner": {
                "fitted_on": "inner_train_only",
                "excluded": ["inner_purge_month", "inner_validation", "outer_validation"],
                "consumer": "neural_network_and_xgboost_early_stopping_only",
            },
            "full": {
                "fitted_on": "full_outer_training_only",
                "excluded": ["outer_validation"],
                "consumer": "neural_network_and_xgboost_refit_and_oof_plus_elastic_net",
            },
        },
        "inner_base_matrix_shape": [EXPECTED_ROWS, BASE_MATRIX_WIDTH],
        "full_base_matrix_shape": [EXPECTED_ROWS, BASE_MATRIX_WIDTH],
        "elastic_full_matrix_shape": [EXPECTED_ROWS, ELASTIC_NET_TOTAL_WIDTH],
        "matrix_dtype": "float32",
        "elastic_full_base_prefix_source": "full_outer_train",
        "base_prefix_identical_in_elastic_full_matrix": True,
        "elastic_interaction_order": "continuous_firm_registry_ordinal_major_then_aggregate_state_registry_ordinal_minor",
        "required_files": required_files,
        "files": {name: receipts[name] for name in required_files},
    }
    atomic_write_json(root / "metadata.json", metadata, trusted_root=root)
    emit("complete prepared-matrix cache")
    return PreparedMaterialization(identity, root, metadata, False)


__all__ = [
    "PreparedBuildIdentity",
    "PreparedMaterialization",
    "compute_prepared_build_identity",
    "prepare_stage9_selector_matrices",
    "prepared_cache_root",
]
