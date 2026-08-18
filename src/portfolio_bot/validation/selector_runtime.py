"""Locked, native-runtime-free selector contracts for Stage 11.

This module is the validation counterpart to the Stage 9 development search.
It deliberately imports no PyTorch, XGBoost, or Elastic Net runtime.  Native
fitting lives in :mod:`portfolio_bot.validation.selector_native_runtime` and
must run in a family-isolated ``spawn`` child.

The responsibilities here are intentionally narrow:

* admit exactly the nine frozen Stage 9 shortlist configurations;
* admit the five frozen stochastic seeds (and no seed for Elastic Net);
* construct leakage-safe quarterly-refit identities from already authorized
  historical labels;
* persist model and preprocessing state in a content-authenticated checkpoint;
* publish raw forecasts in chronological, outcome-free batches.

No function in this file resolves a validation outcome, computes IC, opens a
portfolio, or reads 2020+ data.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from portfolio_bot.selectors.model_specs import (
    canonical_xgboost_configuration,
    configuration_payload_sha256,
    elastic_net_configurations,
    neural_configurations,
)
from portfolio_bot.selectors.weights import (
    EARLY_STOPPING_MONTHS,
    HALF_LIFE_MONTHS,
    latest_month_mask,
    month_balanced_memory_weights,
    month_ordinal,
)
from portfolio_bot.validation.locked_selection import (
    LOCKED_SHORTLIST,
    VALIDATION_STOCHASTIC_SEEDS,
)


SELECTOR_RUNTIME_ID = "stage11-locked-selector-runtime-v1"
CHECKPOINT_STATUS = "STAGE11_SELECTOR_CHECKPOINT_COMPLETE"
FORECAST_BATCH_STATUS = "STAGE11_RAW_FORECAST_BATCH_COMPLETE"
FORECAST_SET_STATUS = "STAGE11_RAW_FORECAST_SET_COMPLETE"
PRIMARY_REFIT_MONTHS = frozenset({3, 6, 9, 12})
MATRIX_KIND_BY_FAMILY = {
    "neural_network": "base",
    "xgboost": "base",
    "elastic_net": "elastic",
}
MODEL_FORMAT_BY_FAMILY = {
    "neural_network": "stage11_deterministic_tensor_bundle_v1",
    "xgboost": "xgboost_ubj_v1",
    "elastic_net": "stage11_deterministic_tensor_bundle_v1",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_HOLDING_MONTH = re.compile(r"^[0-9]{4}-(?:0[1-9]|1[0-2])$")
_FORBIDDEN_RECEIPT_KEYS = frozenset(
    {
        "actual_return",
        "annualized_return",
        "benchmark_return",
        "forward_return",
        "net_return",
        "outcome_value",
        "pnl",
        "portfolio_return",
        "rank_ic",
        "sharpe",
    }
)
_BUNDLE_MAGIC = b"S11ARRAYS1\n"


class SelectorRuntimeError(RuntimeError):
    """Raised when a Stage 11 selector artifact escapes the locked contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value with one canonical representation."""

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


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SelectorRuntimeError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _date(value: object, *, name: str) -> dt.date:
    if type(value) is not dt.date:
        raise SelectorRuntimeError(f"{name} must be a datetime.date")
    return value


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SelectorRuntimeError(f"{name} is not a canonical identifier")
    return value


def _ordinary_root(path: Path, *, create: bool) -> Path:
    root = Path(os.path.abspath(Path(path).expanduser()))
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise SelectorRuntimeError("selector artifact root is absent or unsafe")
    return root


def _ordinary_child(root: Path, name: str) -> Path:
    if "/" in name or name in {"", ".", ".."}:
        raise SelectorRuntimeError("selector artifact name is unsafe")
    path = root / name
    if path.exists() and path.is_symlink():
        raise SelectorRuntimeError("symlinked selector artifact is prohibited")
    return path


def _file_identity(
    value: os.stat_result,
    *,
    include_ctime: bool,
) -> tuple[int, ...]:
    fields = [
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
    ]
    if include_ctime:
        fields.append("st_ctime_ns")
    return tuple(int(getattr(value, field)) for field in fields)


def _authenticate_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    permit_authenticated_ctime_refresh: bool = False,
    permit_fresh_ctime_stabilization: bool = False,
    capture_payload: bool = False,
) -> tuple[str, int, bytes | None]:
    """Authenticate a file, allowing one narrowly proven ctime-only refresh.

    Existing artifacts require a previously authenticated digest.  The fresh
    mode is private to newly written NPY files in an exclusive directory and
    still requires two identical complete hashes.  Content, size, mtime,
    inode, mode, ownership, or link-count changes always fail closed.
    """

    if expected_sha256 is not None:
        _require_sha256(expected_sha256, name="expected selector artifact SHA-256")
    if permit_authenticated_ctime_refresh and expected_sha256 is None:
        raise SelectorRuntimeError(
            "selector artifact ctime refresh requires an authenticated digest"
        )
    if permit_fresh_ctime_stabilization and (
        expected_sha256 is not None or permit_authenticated_ctime_refresh
    ):
        raise SelectorRuntimeError("fresh selector artifact mode is not exclusive")

    def hash_once() -> tuple[
        str,
        bytes | None,
        tuple[os.stat_result, os.stat_result, os.stat_result, os.stat_result],
    ]:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SelectorRuntimeError(
                f"selector artifact is not an ordinary single-link file: {path}"
            )
        digest = hashlib.sha256()
        blocks: list[bytes] | None = [] if capture_payload else None
        with path.open("rb") as stream:
            opened_before = os.fstat(stream.fileno())
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
                if blocks is not None:
                    blocks.append(block)
            opened_after = os.fstat(stream.fileno())
        after = path.stat(follow_symlinks=False)
        identities = (before, opened_before, opened_after, after)
        if any(
            not stat.S_ISREG(value.st_mode) or value.st_nlink != 1
            for value in identities
        ):
            raise SelectorRuntimeError(
                f"selector artifact is not an ordinary single-link file: {path}"
            )
        payload = None if blocks is None else b"".join(blocks)
        return digest.hexdigest(), payload, identities

    digest, payload, identities = hash_once()
    full_identities = tuple(
        _file_identity(value, include_ctime=True) for value in identities
    )
    if len(set(full_identities)) == 1:
        if expected_sha256 is not None and digest != expected_sha256:
            raise SelectorRuntimeError(f"selector artifact bytes differ: {path}")
        return digest, int(identities[-1].st_size), payload
    non_ctime_identities = tuple(
        _file_identity(value, include_ctime=False) for value in identities
    )
    ctime_only = (
        len(set(non_ctime_identities)) == 1
        and len({int(value.st_ctime_ns) for value in identities}) > 1
    )
    permitted = permit_fresh_ctime_stabilization or (
        permit_authenticated_ctime_refresh and digest == expected_sha256
    )
    if not permitted or not ctime_only:
        raise SelectorRuntimeError(f"selector artifact changed while hashing: {path}")

    refreshed_digest, refreshed_payload, refreshed_identities = hash_once()
    refreshed_full_identities = tuple(
        _file_identity(value, include_ctime=True) for value in refreshed_identities
    )
    if (
        full_identities[-1] != refreshed_full_identities[0]
        or len(set(refreshed_full_identities)) != 1
        or refreshed_digest != digest
        or (
            expected_sha256 is not None
            and refreshed_digest != expected_sha256
        )
    ):
        mode = (
            "authenticated ctime refresh"
            if permit_authenticated_ctime_refresh
            else "fresh ctime stabilization"
        )
        raise SelectorRuntimeError(
            f"selector artifact changed during {mode}: {path}"
        )
    return refreshed_digest, int(refreshed_identities[-1].st_size), refreshed_payload


def _stable_file_sha256(
    path: Path,
    *,
    expected_sha256: str | None = None,
    permit_authenticated_ctime_refresh: bool = False,
) -> tuple[str, int]:
    digest, size, _payload = _authenticate_file(
        path,
        expected_sha256=expected_sha256,
        permit_authenticated_ctime_refresh=permit_authenticated_ctime_refresh,
    )
    return digest, size


def _stable_fresh_file_sha256(path: Path) -> tuple[str, int]:
    digest, size, _payload = _authenticate_file(
        path,
        permit_fresh_ctime_stabilization=True,
    )
    return digest, size


def _authenticated_file_bytes(path: Path, *, expected_sha256: str) -> tuple[bytes, int]:
    digest, size, payload = _authenticate_file(
        path,
        expected_sha256=expected_sha256,
        permit_authenticated_ctime_refresh=True,
        capture_payload=True,
    )
    if digest != expected_sha256 or payload is None:
        raise SelectorRuntimeError(f"selector artifact bytes differ: {path}")
    return payload, size


def _atomic_file(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def array_sha256(values: np.ndarray) -> str:
    """Bind dtype, shape, and canonical C-order bytes of a finite array."""

    array = np.asarray(values)
    if array.dtype.hasobject:
        raise SelectorRuntimeError("object arrays cannot enter selector identities")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise SelectorRuntimeError("nonfinite arrays cannot enter selector identities")
    contiguous = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    )
    digest = hashlib.sha256()
    digest.update(header)
    byte_view = memoryview(contiguous).cast("B")
    for start in range(0, len(byte_view), 8 * 1024 * 1024):
        digest.update(byte_view[start : start + 8 * 1024 * 1024])
    return digest.hexdigest()


_REFIT_ARRAY_NAMES = (
    "X_inner_train",
    "y_inner_train",
    "w_inner_train",
    "X_inner_validation",
    "y_inner_validation",
    "w_inner_validation",
    "X_full_train",
    "y_full_train",
    "w_full_train",
)


@dataclass(frozen=True, slots=True)
class FileBackedArrayBinding:
    name: str
    filename: str
    file_sha256: str
    logical_sha256: str
    size_bytes: int
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        _identifier(self.name, name="file-backed array name")
        if self.filename != f"{self.name}.npy" or "/" in self.filename:
            raise SelectorRuntimeError("file-backed array filename differs")
        _require_sha256(self.file_sha256, name="file-backed array file_sha256")
        _require_sha256(self.logical_sha256, name="file-backed array logical_sha256")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise SelectorRuntimeError("file-backed array size must be positive")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise SelectorRuntimeError("file-backed array dtype is invalid") from exc
        if dtype.hasobject or not self.shape or any(
            type(value) is not int or value <= 0 for value in self.shape
        ):
            raise SelectorRuntimeError("file-backed array shape/dtype differs")


@dataclass(frozen=True, slots=True)
class FileBackedArrayRequest:
    root: Path
    binding: FileBackedArrayBinding


@dataclass(frozen=True, slots=True)
class ValidationRefitPathRequest:
    root: Path
    arrays: Mapping[str, FileBackedArrayBinding]
    training_data: Mapping[str, Any]

    def __post_init__(self) -> None:
        if tuple(sorted(self.arrays)) != tuple(sorted(_REFIT_ARRAY_NAMES)):
            raise SelectorRuntimeError("file-backed refit array set differs")
        if any(name != binding.name for name, binding in self.arrays.items()):
            raise SelectorRuntimeError("file-backed refit key/name differs")


@dataclass(frozen=True, slots=True)
class ReadOnlyValidationRefitArrays:
    X_inner_train: np.ndarray
    y_inner_train: np.ndarray
    w_inner_train: np.ndarray
    X_inner_validation: np.ndarray
    y_inner_validation: np.ndarray
    w_inner_validation: np.ndarray
    X_full_train: np.ndarray
    y_full_train: np.ndarray
    w_full_train: np.ndarray
    training_data: Mapping[str, Any]

    @property
    def full_rows(self) -> int:
        return int(self.X_full_train.shape[0])

    @property
    def inner_train_rows(self) -> int:
        return int(self.X_inner_train.shape[0])

    @property
    def inner_validation_rows(self) -> int:
        return int(self.X_inner_validation.shape[0])

    def identity_payload(self) -> dict[str, Any]:
        return dict(self.training_data)


def _write_npy_bounded(path: Path, values: np.ndarray) -> None:
    array = np.asarray(values)
    if (
        array.dtype.hasobject
        or array.ndim == 0
        or array.shape[0] == 0
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise SelectorRuntimeError("file-backed selector arrays must be finite numeric arrays")
    target = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=array.dtype,
        shape=array.shape,
        fortran_order=False,
    )
    try:
        rows_per_chunk = max(1, (64 * 1024 * 1024) // max(1, array[0].nbytes))
        for start in range(0, array.shape[0], rows_per_chunk):
            target[start : start + rows_per_chunk] = array[start : start + rows_per_chunk]
        target.flush()
    finally:
        mmap = getattr(target, "_mmap", None)
        if mmap is not None and not mmap.closed:
            mmap.close()
        del target
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
            raise SelectorRuntimeError("fresh NPY is not an ordinary single-link file")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_empty_array_root(destination: Path, *, kind: str) -> Path:
    root = Path(os.path.abspath(Path(destination).expanduser()))
    if root.exists():
        if not root.is_dir() or root.is_symlink() or any(root.iterdir()):
            raise SelectorRuntimeError(f"file-backed {kind} destination is not empty/safe")
    else:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    identity = root.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(identity.st_mode)
        or identity.st_uid != os.getuid()
        or stat.S_IMODE(identity.st_mode) & 0o077
    ):
        raise SelectorRuntimeError(f"file-backed {kind} destination is not exclusive")
    return root


def _bind_fresh_npy(path: Path, source: np.ndarray) -> FileBackedArrayBinding:
    """Bind a fresh private NPY by physical and source-derived logical hashes."""

    array = np.asarray(source)
    logical_sha = array_sha256(array)
    digest, size = _stable_fresh_file_sha256(path)
    try:
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SelectorRuntimeError("fresh file-backed array is not safe NPY") from exc
    try:
        if (
            mapped.dtype.str != array.dtype.str
            or tuple(mapped.shape) != tuple(array.shape)
            or mapped.flags.writeable
            or array_sha256(mapped) != logical_sha
        ):
            raise SelectorRuntimeError("fresh file-backed array semantics differ")
    finally:
        mmap = getattr(mapped, "_mmap", None)
        if mmap is not None and not mmap.closed:
            mmap.close()
        del mapped
    confirmed_digest, confirmed_size = _stable_file_sha256(
        path,
        expected_sha256=digest,
        permit_authenticated_ctime_refresh=True,
    )
    if confirmed_digest != digest or confirmed_size != size:
        raise SelectorRuntimeError("fresh file-backed array binding changed")
    return FileBackedArrayBinding(
        name=path.stem,
        filename=path.name,
        file_sha256=digest,
        logical_sha256=logical_sha,
        size_bytes=size,
        dtype=array.dtype.str,
        shape=tuple(int(value) for value in array.shape),
    )


def persist_array_for_spawn(
    values: np.ndarray,
    destination: Path,
    *,
    name: str,
) -> FileBackedArrayRequest:
    """Persist one authenticated ``.npy`` array for read-only child mmap."""

    _identifier(name, name="file-backed array name")
    root = _exclusive_empty_array_root(destination, kind="array")
    array = np.asarray(values)
    path = root / f"{name}.npy"
    _write_npy_bounded(path, array)
    return FileBackedArrayRequest(
        root=root,
        binding=_bind_fresh_npy(path, array),
    )


def persist_validation_refit_arrays_for_spawn(
    arrays: "ValidationRefitArrays",
    destination: Path,
) -> ValidationRefitPathRequest:
    """Persist nine refit arrays once so spawn never pickles matrix bytes."""

    root = _exclusive_empty_array_root(destination, kind="refit")
    bindings: dict[str, FileBackedArrayBinding] = {}
    for name in _REFIT_ARRAY_NAMES:
        values = np.asarray(getattr(arrays, name))
        path = root / f"{name}.npy"
        _write_npy_bounded(path, values)
        bindings[name] = _bind_fresh_npy(path, values)
    return ValidationRefitPathRequest(
        root=root,
        arrays=dict(bindings),
        training_data=arrays.identity_payload(),
    )


def load_array_for_spawn(request: FileBackedArrayRequest) -> np.ndarray:
    """Authenticate and mmap one array read-only in a spawn child."""

    root = _ordinary_root(request.root, create=False)
    binding = request.binding
    path = _ordinary_child(root, binding.filename)
    before = path.stat(follow_symlinks=False)
    if before.st_nlink != 1 or int(before.st_size) != binding.size_bytes:
        raise SelectorRuntimeError("file-backed array identity differs")
    digest, _size = _stable_file_sha256(
        path,
        expected_sha256=binding.file_sha256,
        permit_authenticated_ctime_refresh=True,
    )
    if digest != binding.file_sha256:
        raise SelectorRuntimeError("file-backed array bytes differ")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SelectorRuntimeError("file-backed array is not safe NPY") from exc
    if (
        array.dtype.str != binding.dtype
        or tuple(array.shape) != binding.shape
        or array.flags.writeable
        or array_sha256(array) != binding.logical_sha256
    ):
        raise SelectorRuntimeError("file-backed array semantics differ")
    confirmed_digest, confirmed_size = _stable_file_sha256(
        path,
        expected_sha256=binding.file_sha256,
        permit_authenticated_ctime_refresh=True,
    )
    if confirmed_digest != binding.file_sha256 or confirmed_size != binding.size_bytes:
        raise SelectorRuntimeError("file-backed array changed after semantic verification")
    return array


def load_validation_refit_arrays_from_paths(
    request: ValidationRefitPathRequest,
) -> ReadOnlyValidationRefitArrays:
    """Authenticate the closed nine-file set and mmap it read-only."""

    root = _ordinary_root(request.root, create=False)
    expected_names = {binding.filename for binding in request.arrays.values()}
    if {path.name for path in root.iterdir()} != expected_names:
        raise SelectorRuntimeError("file-backed refit directory is not closed")
    loaded = {
        name: load_array_for_spawn(
            FileBackedArrayRequest(root=root, binding=request.arrays[name])
        )
        for name in _REFIT_ARRAY_NAMES
    }
    for name, values in loaded.items():
        expected = request.training_data[f"{name}_sha256"]
        if request.arrays[name].logical_sha256 != expected:
            raise SelectorRuntimeError("file-backed refit differs from training identity")
    result = ReadOnlyValidationRefitArrays(
        **loaded,
        training_data=dict(request.training_data),
    )
    if (
        result.X_full_train.ndim != 2
        or result.y_full_train.shape != (result.full_rows,)
        or result.w_full_train.shape != (result.full_rows,)
        or result.full_rows != request.training_data.get("full_rows")
        or result.X_inner_train.ndim != 2
        or result.y_inner_train.shape != (result.inner_train_rows,)
        or result.w_inner_train.shape != (result.inner_train_rows,)
        or result.inner_train_rows != request.training_data.get("inner_train_rows")
        or result.X_inner_validation.ndim != 2
        or result.y_inner_validation.shape != (result.inner_validation_rows,)
        or result.w_inner_validation.shape != (result.inner_validation_rows,)
        or result.inner_validation_rows
        != request.training_data.get("inner_validation_rows")
    ):
        raise SelectorRuntimeError("file-backed refit dimensions differ")
    if result.identity_payload() != dict(request.training_data):
        raise SelectorRuntimeError("file-backed refit identity differs")
    return result


def pack_numeric_arrays(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Create a deterministic, pickle-free tensor bundle.

    Unlike ``np.savez``, this small format has no ZIP timestamps.  It is used
    for NN state dictionaries and Elastic Net coefficients so a repeated fit
    can be compared byte for byte.
    """

    if not arrays:
        raise SelectorRuntimeError("a model tensor bundle cannot be empty")
    names = sorted(arrays)
    if len(names) != len(set(names)) or any(not name for name in names):
        raise SelectorRuntimeError("model tensor names must be unique and nonempty")
    normalized: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    offset = 0
    for name in names:
        if not isinstance(name, str):
            raise SelectorRuntimeError("model tensor names must be strings")
        array = np.asarray(arrays[name])
        if array.dtype.hasobject or not (
            np.issubdtype(array.dtype, np.number) or array.dtype == np.dtype("bool")
        ):
            raise SelectorRuntimeError("model tensors must be numeric and pickle-free")
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise SelectorRuntimeError("model tensors must be finite")
        value = np.ascontiguousarray(array)
        normalized[name] = value
        size = int(value.nbytes)
        entries.append(
            {
                "name": name,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "offset": offset,
                "size": size,
            }
        )
        offset += size
    header = canonical_json_bytes({"schema_version": 1, "arrays": entries})
    return b"".join(
        (
            _BUNDLE_MAGIC,
            struct.pack(">Q", len(header)),
            header,
            *(normalized[name].tobytes(order="C") for name in names),
        )
    )


def unpack_numeric_arrays(payload: bytes) -> dict[str, np.ndarray]:
    if not isinstance(payload, bytes) or not payload.startswith(_BUNDLE_MAGIC):
        raise SelectorRuntimeError("model tensor bundle has an invalid magic header")
    prefix = len(_BUNDLE_MAGIC)
    if len(payload) < prefix + 8:
        raise SelectorRuntimeError("model tensor bundle is truncated")
    header_size = struct.unpack(">Q", payload[prefix : prefix + 8])[0]
    header_start = prefix + 8
    header_stop = header_start + header_size
    try:
        header = json.loads(payload[header_start:header_stop].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectorRuntimeError("model tensor bundle has invalid metadata") from exc
    if not isinstance(header, dict) or set(header) != {"schema_version", "arrays"}:
        raise SelectorRuntimeError("model tensor metadata has unexpected keys")
    if header["schema_version"] != 1 or not isinstance(header["arrays"], list):
        raise SelectorRuntimeError("model tensor metadata version is invalid")
    data = memoryview(payload)[header_stop:]
    result: dict[str, np.ndarray] = {}
    expected_offset = 0
    for entry in header["arrays"]:
        if not isinstance(entry, dict) or set(entry) != {
            "name",
            "dtype",
            "shape",
            "offset",
            "size",
        }:
            raise SelectorRuntimeError("model tensor entry has unexpected keys")
        name = entry["name"]
        if not isinstance(name, str) or not name or name in result:
            raise SelectorRuntimeError("model tensor names are invalid")
        dtype = np.dtype(entry["dtype"])
        shape = tuple(int(value) for value in entry["shape"])
        offset = int(entry["offset"])
        size = int(entry["size"])
        if dtype.hasobject or any(value < 0 for value in shape):
            raise SelectorRuntimeError("model tensor schema is unsafe")
        expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if offset != expected_offset or size != expected_size or offset + size > len(data):
            raise SelectorRuntimeError("model tensor offsets are invalid")
        array = np.frombuffer(data[offset : offset + size], dtype=dtype).reshape(shape).copy()
        if np.issubdtype(dtype, np.number) and not np.isfinite(array).all():
            raise SelectorRuntimeError("model tensor contains nonfinite values")
        result[name] = array
        expected_offset += size
    if expected_offset != len(data) or not result:
        raise SelectorRuntimeError("model tensor payload has trailing or absent bytes")
    return result


def _config_payload(family: str, config_id: str) -> tuple[dict[str, Any], str]:
    shortlist = LOCKED_SHORTLIST.get(family)
    if shortlist is None or config_id not in shortlist:
        raise SelectorRuntimeError("selector config is outside the frozen nine-item shortlist")
    if family == "neural_network":
        config = {item.config_id: item for item in neural_configurations()}[config_id]
    elif family == "xgboost":
        config = canonical_xgboost_configuration(config_id)
    else:
        config = {item.config_id: item for item in elastic_net_configurations()}[config_id]
    return asdict(config), configuration_payload_sha256(config)


def validate_locked_selector(family: str, config_id: str, seed: int | None) -> None:
    """Reject every config/seed combination outside Methodology v1."""

    _config_payload(family, config_id)
    if family == "elastic_net":
        if seed is not None:
            raise SelectorRuntimeError("Elastic Net is deterministic and cannot carry a seed")
    elif type(seed) is not int or seed not in VALIDATION_STOCHASTIC_SEEDS:
        raise SelectorRuntimeError(
            "Stage 11 stochastic selectors require one of seeds 11, 23, 47, 71, 101"
        )


@dataclass(frozen=True, slots=True)
class AggregateMomentState:
    feature_name: str
    finite_count: int
    mean: float
    m2: float
    last_observation_date: dt.date

    def __post_init__(self) -> None:
        _identifier(self.feature_name, name="aggregate feature_name")
        if type(self.finite_count) is not int or self.finite_count < 0:
            raise SelectorRuntimeError("aggregate finite_count must be nonnegative")
        if not math.isfinite(float(self.mean)) or not math.isfinite(float(self.m2)):
            raise SelectorRuntimeError("aggregate moments must be finite")
        if float(self.m2) < 0.0:
            raise SelectorRuntimeError("aggregate m2 must be nonnegative")
        _date(self.last_observation_date, name="aggregate last_observation_date")

    def payload(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "finite_count": self.finite_count,
            "mean_hex": float(self.mean).hex(),
            "m2_hex": float(self.m2).hex(),
            "last_observation_date": self.last_observation_date.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class PersistedPreprocessorState:
    """Reconstructible state fitted only on rows available at one refit."""

    scope: str
    matrix_kind: str
    refit_signal_date: dt.date
    fitted_signal_date_max: dt.date
    predictor_availability_date_max: dt.date
    frozen_column_names: tuple[str, ...]
    active_column_indices: tuple[int, ...]
    feature_registry_sha256: str
    category_vocabulary_sha256: str
    aggregate_history_sha256: str
    interaction_registry_sha256: str | None = None
    category_mapping: tuple[tuple[str, str, str], ...] = ()
    aggregate_moments: tuple[AggregateMomentState, ...] = ()

    def __post_init__(self) -> None:
        if self.scope not in {"inner", "full"}:
            raise SelectorRuntimeError("preprocessor scope must be inner or full")
        if self.matrix_kind not in {"base", "elastic"}:
            raise SelectorRuntimeError("preprocessor matrix_kind must be base or elastic")
        refit = _date(self.refit_signal_date, name="refit_signal_date")
        fitted = _date(self.fitted_signal_date_max, name="fitted_signal_date_max")
        available = _date(
            self.predictor_availability_date_max,
            name="predictor_availability_date_max",
        )
        if fitted > refit or available > refit:
            raise SelectorRuntimeError("preprocessor state uses predictors after its refit")
        if not self.frozen_column_names or len(set(self.frozen_column_names)) != len(
            self.frozen_column_names
        ):
            raise SelectorRuntimeError("preprocessor frozen columns must be unique and nonempty")
        if any(not isinstance(name, str) or not name for name in self.frozen_column_names):
            raise SelectorRuntimeError("preprocessor column names must be nonempty strings")
        active = self.active_column_indices
        if (
            not active
            or tuple(sorted(set(active))) != active
            or active[0] < 0
            or active[-1] >= len(self.frozen_column_names)
        ):
            raise SelectorRuntimeError("preprocessor active-column indices are invalid")
        for name, value in (
            ("feature_registry_sha256", self.feature_registry_sha256),
            ("category_vocabulary_sha256", self.category_vocabulary_sha256),
            ("aggregate_history_sha256", self.aggregate_history_sha256),
        ):
            _require_sha256(value, name=name)
        if self.matrix_kind == "elastic":
            _require_sha256(
                self.interaction_registry_sha256,
                name="interaction_registry_sha256",
            )
        elif self.interaction_registry_sha256 is not None:
            raise SelectorRuntimeError("base preprocessing cannot carry interactions")
        if tuple(sorted(self.category_mapping)) != self.category_mapping:
            raise SelectorRuntimeError("category mappings must be in canonical sorted order")
        if any(len(item) != 3 or any(not str(value) for value in item) for item in self.category_mapping):
            raise SelectorRuntimeError("category mapping entries are invalid")
        names = tuple(item.feature_name for item in self.aggregate_moments)
        if names != tuple(sorted(set(names))):
            raise SelectorRuntimeError("aggregate moment states must be unique and sorted")
        if any(item.last_observation_date > refit for item in self.aggregate_moments):
            raise SelectorRuntimeError("aggregate state includes a post-refit observation")

    @property
    def model_width(self) -> int:
        return len(self.active_column_indices)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "runtime_id": SELECTOR_RUNTIME_ID,
            "scope": self.scope,
            "matrix_kind": self.matrix_kind,
            "refit_signal_date": self.refit_signal_date.isoformat(),
            "fitted_signal_date_max": self.fitted_signal_date_max.isoformat(),
            "predictor_availability_date_max": self.predictor_availability_date_max.isoformat(),
            "frozen_column_names": list(self.frozen_column_names),
            "active_column_indices": list(self.active_column_indices),
            "feature_registry_sha256": self.feature_registry_sha256,
            "category_vocabulary_sha256": self.category_vocabulary_sha256,
            "aggregate_history_sha256": self.aggregate_history_sha256,
            "interaction_registry_sha256": self.interaction_registry_sha256,
            "category_mapping": [list(item) for item in self.category_mapping],
            "aggregate_moments": [item.payload() for item in self.aggregate_moments],
        }

    @property
    def sha256(self) -> str:
        return _sha256_bytes(canonical_json_bytes(self.payload()))

    def apply_support(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.frozen_column_names)
            or not np.isfinite(values).all()
        ):
            raise SelectorRuntimeError("preprocessor input matrix has an invalid schema")
        result = values[:, np.asarray(self.active_column_indices, dtype=np.int64)]
        return np.ascontiguousarray(result, dtype=np.float32)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PersistedPreprocessorState":
        expected = {
            "schema_version",
            "runtime_id",
            "scope",
            "matrix_kind",
            "refit_signal_date",
            "fitted_signal_date_max",
            "predictor_availability_date_max",
            "frozen_column_names",
            "active_column_indices",
            "feature_registry_sha256",
            "category_vocabulary_sha256",
            "aggregate_history_sha256",
            "interaction_registry_sha256",
            "category_mapping",
            "aggregate_moments",
        }
        if set(payload) != expected or payload.get("schema_version") != 1:
            raise SelectorRuntimeError("preprocessor payload keys or version differ")
        if payload.get("runtime_id") != SELECTOR_RUNTIME_ID:
            raise SelectorRuntimeError("preprocessor runtime binding differs")
        moments = tuple(
            AggregateMomentState(
                feature_name=str(item["feature_name"]),
                finite_count=int(item["finite_count"]),
                mean=float.fromhex(str(item["mean_hex"])),
                m2=float.fromhex(str(item["m2_hex"])),
                last_observation_date=dt.date.fromisoformat(
                    str(item["last_observation_date"])
                ),
            )
            for item in payload["aggregate_moments"]
        )
        return cls(
            scope=str(payload["scope"]),
            matrix_kind=str(payload["matrix_kind"]),
            refit_signal_date=dt.date.fromisoformat(str(payload["refit_signal_date"])),
            fitted_signal_date_max=dt.date.fromisoformat(
                str(payload["fitted_signal_date_max"])
            ),
            predictor_availability_date_max=dt.date.fromisoformat(
                str(payload["predictor_availability_date_max"])
            ),
            frozen_column_names=tuple(str(value) for value in payload["frozen_column_names"]),
            active_column_indices=tuple(int(value) for value in payload["active_column_indices"]),
            feature_registry_sha256=str(payload["feature_registry_sha256"]),
            category_vocabulary_sha256=str(payload["category_vocabulary_sha256"]),
            aggregate_history_sha256=str(payload["aggregate_history_sha256"]),
            interaction_registry_sha256=(
                None
                if payload["interaction_registry_sha256"] is None
                else str(payload["interaction_registry_sha256"])
            ),
            category_mapping=tuple(
                tuple(str(value) for value in item)  # type: ignore[arg-type]
                for item in payload["category_mapping"]
            ),
            aggregate_moments=moments,
        )


@dataclass(frozen=True, slots=True)
class ValidationRefitArrays:
    """Prepared numerical partitions and their chronology-only evidence."""

    X_inner_train: np.ndarray
    y_inner_train: np.ndarray
    w_inner_train: np.ndarray
    X_inner_validation: np.ndarray
    y_inner_validation: np.ndarray
    w_inner_validation: np.ndarray
    X_full_train: np.ndarray
    y_full_train: np.ndarray
    w_full_train: np.ndarray
    full_holding_months: tuple[str, ...]
    full_exit_dates: tuple[dt.date, ...]
    full_availability_dates: tuple[dt.date, ...]
    full_row_keys_sha256: str

    def __post_init__(self) -> None:
        full_rows = int(np.asarray(self.X_full_train).shape[0])
        if not (
            len(self.full_holding_months)
            == len(self.full_exit_dates)
            == len(self.full_availability_dates)
            == full_rows
        ):
            raise SelectorRuntimeError("full refit chronology lengths differ")
        _require_sha256(self.full_row_keys_sha256, name="full_row_keys_sha256")

    @property
    def full_rows(self) -> int:
        return int(np.asarray(self.X_full_train).shape[0])

    @property
    def inner_train_rows(self) -> int:
        return int(np.asarray(self.X_inner_train).shape[0])

    @property
    def inner_validation_rows(self) -> int:
        return int(np.asarray(self.X_inner_validation).shape[0])

    def identity_payload(self) -> dict[str, Any]:
        return {
            "X_inner_train_sha256": array_sha256(self.X_inner_train),
            "y_inner_train_sha256": array_sha256(self.y_inner_train),
            "w_inner_train_sha256": array_sha256(self.w_inner_train),
            "X_inner_validation_sha256": array_sha256(self.X_inner_validation),
            "y_inner_validation_sha256": array_sha256(self.y_inner_validation),
            "w_inner_validation_sha256": array_sha256(self.w_inner_validation),
            "X_full_train_sha256": array_sha256(self.X_full_train),
            "y_full_train_sha256": array_sha256(self.y_full_train),
            "w_full_train_sha256": array_sha256(self.w_full_train),
            "full_row_keys_sha256": self.full_row_keys_sha256,
            "full_rows": self.full_rows,
            "inner_train_rows": self.inner_train_rows,
            "inner_validation_rows": self.inner_validation_rows,
            "first_holding_month": min(self.full_holding_months, key=month_ordinal),
            "last_holding_month": max(self.full_holding_months, key=month_ordinal),
            "max_exit_date": max(self.full_exit_dates).isoformat(),
            "max_outcome_availability_date": max(self.full_availability_dates).isoformat(),
        }


def prepare_validation_refit_arrays(
    *,
    refit_signal_date: dt.date,
    full_base_matrix: np.ndarray,
    inner_base_matrix: np.ndarray,
    targets: Sequence[float],
    holding_months: Sequence[str],
    exit_dates: Sequence[dt.date],
    outcome_availability_dates: Sequence[dt.date],
    row_keys_sha256: str,
    full_preprocessor: PersistedPreprocessorState,
    inner_preprocessor: PersistedPreprocessorState,
) -> ValidationRefitArrays:
    """Create the frozen latest-12-month early-stop/full-refit partitions.

    Callers must supply only labels already authorized for this refit.  The
    function independently rejects any row whose exit or availability date is
    later than the refit, recomputes all weights, and derives the inner block;
    callers cannot pass a favorable split or observation weights.
    """

    refit = _date(refit_signal_date, name="refit_signal_date")
    if refit.month not in PRIMARY_REFIT_MONTHS:
        raise SelectorRuntimeError("primary selector parameters may refit only quarterly")
    if full_preprocessor.scope != "full" or inner_preprocessor.scope != "inner":
        raise SelectorRuntimeError("refit preprocessing scopes differ from inner/full")
    if full_preprocessor.refit_signal_date != refit or inner_preprocessor.refit_signal_date != refit:
        raise SelectorRuntimeError("preprocessor state is bound to a different refit")
    months = tuple(str(value) for value in holding_months)
    exits = tuple(exit_dates)
    availability = tuple(outcome_availability_dates)
    y = np.asarray(targets, dtype=np.float32)
    full_raw = np.asarray(full_base_matrix, dtype=np.float32)
    inner_raw = np.asarray(inner_base_matrix, dtype=np.float32)
    rows = len(months)
    if (
        rows < EARLY_STOPPING_MONTHS + 1
        or y.shape != (rows,)
        or full_raw.ndim != 2
        or inner_raw.ndim != 2
        or full_raw.shape[0] != rows
        or inner_raw.shape[0] != rows
        or len(exits) != rows
        or len(availability) != rows
    ):
        raise SelectorRuntimeError("validation refit arrays have inconsistent dimensions")
    if not np.isfinite(y).all() or np.any(y < -0.5) or np.any(y > 0.5):
        raise SelectorRuntimeError("selector targets must be finite residual-return ranks")
    for month in months:
        if _HOLDING_MONTH.fullmatch(month) is None:
            raise SelectorRuntimeError("training holding months must use YYYY-MM")
        if month_ordinal(month) >= month_ordinal(refit):
            raise SelectorRuntimeError("current or future holding month entered a refit")
    for exit_date, available in zip(exits, availability, strict=True):
        if type(exit_date) is not dt.date or type(available) is not dt.date:
            raise SelectorRuntimeError("label chronology values must be dates")
        if exit_date > refit or available > refit or available < exit_date:
            raise SelectorRuntimeError("a refit used an incomplete or unavailable label")
    inner_validation_mask = latest_month_mask(months, count=EARLY_STOPPING_MONTHS)
    inner_train_mask = ~inner_validation_mask
    if not inner_train_mask.any() or not inner_validation_mask.any():
        raise SelectorRuntimeError("validation refit lacks an inner training partition")
    selected_inner_months = tuple(
        sorted(set(np.asarray(months, dtype=object)[inner_validation_mask]), key=month_ordinal)
    )
    if len(selected_inner_months) != EARLY_STOPPING_MONTHS:
        raise SelectorRuntimeError("inner early-stopping block is not exactly 12 months")
    full_matrix = full_preprocessor.apply_support(full_raw)
    inner_matrix = inner_preprocessor.apply_support(inner_raw)
    full_weights = month_balanced_memory_weights(months, fit_signal_date=refit)
    inner_train_months = tuple(np.asarray(months, dtype=object)[inner_train_mask].tolist())
    inner_validation_months = tuple(
        np.asarray(months, dtype=object)[inner_validation_mask].tolist()
    )
    inner_train_weights = month_balanced_memory_weights(
        inner_train_months, fit_signal_date=refit
    )
    inner_validation_weights = month_balanced_memory_weights(
        inner_validation_months, fit_signal_date=refit
    )
    return ValidationRefitArrays(
        X_inner_train=np.ascontiguousarray(inner_matrix[inner_train_mask], dtype=np.float32),
        y_inner_train=np.ascontiguousarray(y[inner_train_mask], dtype=np.float32),
        w_inner_train=np.ascontiguousarray(inner_train_weights, dtype=np.float64),
        X_inner_validation=np.ascontiguousarray(
            inner_matrix[inner_validation_mask], dtype=np.float32
        ),
        y_inner_validation=np.ascontiguousarray(y[inner_validation_mask], dtype=np.float32),
        w_inner_validation=np.ascontiguousarray(
            inner_validation_weights, dtype=np.float64
        ),
        X_full_train=np.ascontiguousarray(full_matrix, dtype=np.float32),
        y_full_train=np.ascontiguousarray(y, dtype=np.float32),
        w_full_train=np.ascontiguousarray(full_weights, dtype=np.float64),
        full_holding_months=months,
        full_exit_dates=exits,
        full_availability_dates=availability,
        full_row_keys_sha256=_require_sha256(
            row_keys_sha256, name="row_keys_sha256"
        ),
    )


@dataclass(frozen=True, slots=True)
class ValidationRefitIdentity:
    family: str
    config_id: str
    seed: int | None
    refit_signal_date: dt.date
    validation_calendar_sha256: str
    training_data: Mapping[str, Any]
    inner_preprocessor_sha256: str | None
    full_preprocessor_sha256: str
    model_code_sha256: str
    runtime_versions_sha256: str
    stage9_shortlist_sha256: str
    target_name: str = "residual_return_rank"
    loss_name: str = "month_weighted_mse"
    memory_half_life_months: float = HALF_LIFE_MONTHS
    inner_early_stopping_months: int = EARLY_STOPPING_MONTHS

    def __post_init__(self) -> None:
        validate_locked_selector(self.family, self.config_id, self.seed)
        refit = _date(self.refit_signal_date, name="refit_signal_date")
        if refit.month not in PRIMARY_REFIT_MONTHS:
            raise SelectorRuntimeError("refit identity is not quarterly")
        for name, value in (
            ("validation_calendar_sha256", self.validation_calendar_sha256),
            ("full_preprocessor_sha256", self.full_preprocessor_sha256),
            ("model_code_sha256", self.model_code_sha256),
            ("runtime_versions_sha256", self.runtime_versions_sha256),
            ("stage9_shortlist_sha256", self.stage9_shortlist_sha256),
        ):
            _require_sha256(value, name=name)
        if self.family == "elastic_net":
            if self.inner_preprocessor_sha256 is not None:
                raise SelectorRuntimeError("Elastic Net cannot carry an unused inner preprocessor")
        else:
            _require_sha256(
                self.inner_preprocessor_sha256,
                name="inner_preprocessor_sha256",
            )
        if self.target_name != "residual_return_rank" or self.loss_name != "month_weighted_mse":
            raise SelectorRuntimeError("selector target or loss differs from Methodology v1")
        if (
            float(self.memory_half_life_months) != HALF_LIFE_MONTHS
            or self.inner_early_stopping_months != EARLY_STOPPING_MONTHS
        ):
            raise SelectorRuntimeError("selector memory or inner block differs")
        required_training = {
            "X_inner_train_sha256",
            "y_inner_train_sha256",
            "w_inner_train_sha256",
            "X_inner_validation_sha256",
            "y_inner_validation_sha256",
            "w_inner_validation_sha256",
            "X_full_train_sha256",
            "y_full_train_sha256",
            "w_full_train_sha256",
            "full_row_keys_sha256",
            "full_rows",
            "inner_train_rows",
            "inner_validation_rows",
            "first_holding_month",
            "last_holding_month",
            "max_exit_date",
            "max_outcome_availability_date",
        }
        if set(self.training_data) != required_training:
            raise SelectorRuntimeError("refit training identity has unexpected keys")
        for name in required_training:
            if name.endswith("_sha256"):
                _require_sha256(self.training_data[name], name=name)
        if dt.date.fromisoformat(str(self.training_data["max_exit_date"])) > refit:
            raise SelectorRuntimeError("refit identity includes an uncompleted label")
        if (
            dt.date.fromisoformat(str(self.training_data["max_outcome_availability_date"]))
            > refit
        ):
            raise SelectorRuntimeError("refit identity includes an unavailable label")

    def payload(self) -> dict[str, Any]:
        config, config_sha = _config_payload(self.family, self.config_id)
        return {
            "schema_version": 1,
            "runtime_id": SELECTOR_RUNTIME_ID,
            "family": self.family,
            "config_id": self.config_id,
            "config": config,
            "config_payload_sha256": config_sha,
            "seed": self.seed,
            "refit_signal_date": self.refit_signal_date.isoformat(),
            "validation_calendar_sha256": self.validation_calendar_sha256,
            "training_data": dict(self.training_data),
            "inner_preprocessor_sha256": self.inner_preprocessor_sha256,
            "full_preprocessor_sha256": self.full_preprocessor_sha256,
            "model_code_sha256": self.model_code_sha256,
            "runtime_versions_sha256": self.runtime_versions_sha256,
            "stage9_shortlist_sha256": self.stage9_shortlist_sha256,
            "target_name": self.target_name,
            "loss_name": self.loss_name,
            "memory_half_life_months_hex": float(self.memory_half_life_months).hex(),
            "inner_early_stopping_months": self.inner_early_stopping_months,
            "training_outcomes_scope": "completed_and_available_before_or_on_refit_only",
            "validation_selection_metrics_read": False,
            "portfolio_outcomes_read": False,
            "final_2020_plus_data_read": False,
        }

    @property
    def sha256(self) -> str:
        return _sha256_bytes(canonical_json_bytes(self.payload()))

    @property
    def checkpoint_id(self) -> str:
        return f"stage11-selector-{self.sha256[:32]}"


def build_refit_identity(
    *,
    family: str,
    config_id: str,
    seed: int | None,
    refit_signal_date: dt.date,
    arrays: ValidationRefitArrays,
    full_preprocessor: PersistedPreprocessorState,
    inner_preprocessor: PersistedPreprocessorState | None,
    validation_calendar_sha256: str,
    model_code_sha256: str,
    runtime_versions_sha256: str,
    stage9_shortlist_sha256: str,
) -> ValidationRefitIdentity:
    validate_locked_selector(family, config_id, seed)
    expected_kind = MATRIX_KIND_BY_FAMILY[family]
    if full_preprocessor.scope != "full" or full_preprocessor.matrix_kind != expected_kind:
        raise SelectorRuntimeError("full preprocessor kind differs from model family")
    if full_preprocessor.model_width != np.asarray(arrays.X_full_train).shape[1]:
        raise SelectorRuntimeError("full preprocessor/model training widths differ")
    if family == "elastic_net":
        if inner_preprocessor is not None:
            raise SelectorRuntimeError("Elastic Net does not use early stopping")
        inner_sha = None
    else:
        if (
            inner_preprocessor is None
            or inner_preprocessor.scope != "inner"
            or inner_preprocessor.matrix_kind != "base"
            or inner_preprocessor.model_width != np.asarray(arrays.X_inner_train).shape[1]
            or inner_preprocessor.model_width
            != np.asarray(arrays.X_inner_validation).shape[1]
        ):
            raise SelectorRuntimeError("inner preprocessor/model training widths differ")
        inner_sha = inner_preprocessor.sha256
    return ValidationRefitIdentity(
        family=family,
        config_id=config_id,
        seed=seed,
        refit_signal_date=refit_signal_date,
        validation_calendar_sha256=validation_calendar_sha256,
        training_data=arrays.identity_payload(),
        inner_preprocessor_sha256=inner_sha,
        full_preprocessor_sha256=full_preprocessor.sha256,
        model_code_sha256=model_code_sha256,
        runtime_versions_sha256=runtime_versions_sha256,
        stage9_shortlist_sha256=stage9_shortlist_sha256,
    )


@dataclass(frozen=True, slots=True)
class NativeModelArtifact:
    family: str
    model_format: str
    model_payload: bytes
    fit_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.family not in LOCKED_SHORTLIST:
            raise SelectorRuntimeError("native model artifact family is not locked")
        if self.model_format != MODEL_FORMAT_BY_FAMILY[self.family]:
            raise SelectorRuntimeError("native model artifact format differs")
        if not isinstance(self.model_payload, bytes) or not self.model_payload:
            raise SelectorRuntimeError("native model artifact payload is empty")
        _reject_empirical_receipt_fields(self.fit_receipt)


def _reject_empirical_receipt_fields(value: object, *, path: str = "fit_receipt") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_RECEIPT_KEYS:
                raise SelectorRuntimeError(f"{path} contains forbidden empirical field {key!r}")
            _reject_empirical_receipt_fields(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_empirical_receipt_fields(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise SelectorRuntimeError(f"{path} contains a nonfinite value")


@dataclass(frozen=True, slots=True)
class LoadedSelectorCheckpoint:
    root: Path
    manifest: Mapping[str, Any]
    identity: Mapping[str, Any]
    model_payload: bytes
    inner_preprocessor: PersistedPreprocessorState | None
    full_preprocessor: PersistedPreprocessorState

    @property
    def checkpoint_id(self) -> str:
        return str(self.manifest["checkpoint_id"])

    @property
    def manifest_sha256(self) -> str:
        return _sha256_bytes(canonical_json_bytes(dict(self.manifest)))


def publish_selector_checkpoint(
    *,
    checkpoint_root: Path,
    identity: ValidationRefitIdentity,
    artifact: NativeModelArtifact,
    full_preprocessor: PersistedPreprocessorState,
    inner_preprocessor: PersistedPreprocessorState | None,
) -> LoadedSelectorCheckpoint:
    """Atomically publish and immediately reauthenticate one fitted model."""

    if artifact.family != identity.family:
        raise SelectorRuntimeError("model artifact family differs from refit identity")
    fit_receipt = dict(artifact.fit_receipt)
    if (
        fit_receipt.get("status") != "AUTHORITATIVE_STAGE11_MODEL_FIT_COMPLETE"
        or fit_receipt.get("family") != identity.family
        or fit_receipt.get("config_id") != identity.config_id
        or fit_receipt.get("seed") != identity.seed
        or fit_receipt.get("refit_signal_date")
        != identity.refit_signal_date.isoformat()
    ):
        raise SelectorRuntimeError(
            "only an identity-matched authoritative Stage 11 fit may be published"
        )
    if full_preprocessor.sha256 != identity.full_preprocessor_sha256:
        raise SelectorRuntimeError("full preprocessor differs from refit identity")
    if identity.inner_preprocessor_sha256 is None:
        if inner_preprocessor is not None:
            raise SelectorRuntimeError("unexpected inner preprocessor for deterministic comparator")
    elif inner_preprocessor is None or inner_preprocessor.sha256 != identity.inner_preprocessor_sha256:
        raise SelectorRuntimeError("inner preprocessor differs from refit identity")
    root = _ordinary_root(checkpoint_root, create=True)
    destination = _ordinary_child(root, identity.checkpoint_id)
    if destination.exists():
        loaded = load_selector_checkpoint(destination, expected_identity=identity)
        if loaded.model_payload != artifact.model_payload:
            raise SelectorRuntimeError("repeated selector fit changed model bytes")
        return loaded
    temporary = root / f".{identity.checkpoint_id}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    try:
        model_name = "model.ubj" if identity.family == "xgboost" else "model.arrays"
        model_path = temporary / model_name
        _atomic_file(model_path, artifact.model_payload)
        model_expected_sha = _sha256_bytes(artifact.model_payload)
        model_sha, model_size = _stable_file_sha256(
            model_path,
            expected_sha256=model_expected_sha,
            permit_authenticated_ctime_refresh=True,
        )
        full_payload = canonical_json_bytes(full_preprocessor.payload())
        _atomic_file(temporary / "full_preprocessor.json", full_payload)
        full_sha, full_size = _stable_file_sha256(
            temporary / "full_preprocessor.json",
            expected_sha256=full_preprocessor.sha256,
            permit_authenticated_ctime_refresh=True,
        )
        if full_sha != full_preprocessor.sha256:
            raise SelectorRuntimeError("serialized full preprocessor digest changed")
        inner_record: dict[str, Any] | None = None
        if inner_preprocessor is not None:
            inner_payload = canonical_json_bytes(inner_preprocessor.payload())
            _atomic_file(temporary / "inner_preprocessor.json", inner_payload)
            inner_sha, inner_size = _stable_file_sha256(
                temporary / "inner_preprocessor.json",
                expected_sha256=inner_preprocessor.sha256,
                permit_authenticated_ctime_refresh=True,
            )
            if inner_sha != inner_preprocessor.sha256:
                raise SelectorRuntimeError("serialized inner preprocessor digest changed")
            inner_record = {
                "name": "inner_preprocessor.json",
                "sha256": inner_sha,
                "size_bytes": inner_size,
            }
        manifest = {
            "schema_version": 1,
            "runtime_id": SELECTOR_RUNTIME_ID,
            "status": CHECKPOINT_STATUS,
            "checkpoint_id": identity.checkpoint_id,
            "job_identity": identity.payload(),
            "job_identity_sha256": identity.sha256,
            "model": {
                "name": model_name,
                "format": artifact.model_format,
                "sha256": model_sha,
                "size_bytes": model_size,
            },
            "full_preprocessor": {
                "name": "full_preprocessor.json",
                "sha256": full_sha,
                "size_bytes": full_size,
            },
            "inner_preprocessor": inner_record,
            "fit_receipt": fit_receipt,
            "fit_receipt_sha256": _sha256_bytes(canonical_json_bytes(fit_receipt)),
            "scientific_scope": {
                "validation_walk_forward": True,
                "completed_training_labels_only": True,
                "selection_metrics_read": False,
                "portfolio_weights_or_pnl_read": False,
                "final_2020_plus_data_read": False,
            },
        }
        _atomic_file(temporary / "manifest.json", canonical_json_bytes(manifest))
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            shutil.rmtree(temporary)
    return load_selector_checkpoint(destination, expected_identity=identity)


def _load_json_file(path: Path) -> dict[str, Any]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise SelectorRuntimeError(f"selector JSON is not an ordinary file: {path}")
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorRuntimeError(f"selector JSON is invalid: {path}") from exc
    after = path.stat(follow_symlinks=False)
    if (
        before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or not isinstance(value, dict)
    ):
        raise SelectorRuntimeError(f"selector JSON changed or is not an object: {path}")
    return value


def load_selector_checkpoint(
    checkpoint_directory: Path,
    *,
    expected_identity: ValidationRefitIdentity | None = None,
) -> LoadedSelectorCheckpoint:
    root = Path(os.path.abspath(Path(checkpoint_directory).expanduser()))
    if not root.is_dir() or root.is_symlink():
        raise SelectorRuntimeError("selector checkpoint directory is absent or unsafe")
    manifest = _load_json_file(root / "manifest.json")
    expected_keys = {
        "schema_version",
        "runtime_id",
        "status",
        "checkpoint_id",
        "job_identity",
        "job_identity_sha256",
        "model",
        "full_preprocessor",
        "inner_preprocessor",
        "fit_receipt",
        "fit_receipt_sha256",
        "scientific_scope",
    }
    if (
        set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("runtime_id") != SELECTOR_RUNTIME_ID
        or manifest.get("status") != CHECKPOINT_STATUS
    ):
        raise SelectorRuntimeError("selector checkpoint manifest contract differs")
    identity_payload = manifest["job_identity"]
    if not isinstance(identity_payload, dict):
        raise SelectorRuntimeError("selector checkpoint identity is not an object")
    identity_sha = _sha256_bytes(canonical_json_bytes(identity_payload))
    if manifest["job_identity_sha256"] != identity_sha:
        raise SelectorRuntimeError("selector checkpoint identity digest differs")
    if manifest["checkpoint_id"] != f"stage11-selector-{identity_sha[:32]}":
        raise SelectorRuntimeError("selector checkpoint ID is not content addressed")
    if expected_identity is not None and canonical_json_bytes(
        identity_payload
    ) != canonical_json_bytes(expected_identity.payload()):
        raise SelectorRuntimeError("selector checkpoint identity differs from expectation")
    if root.name != manifest["checkpoint_id"]:
        raise SelectorRuntimeError("selector checkpoint directory name differs")
    family = str(identity_payload.get("family"))
    model_record = manifest["model"]
    if not isinstance(model_record, dict) or set(model_record) != {
        "name",
        "format",
        "sha256",
        "size_bytes",
    }:
        raise SelectorRuntimeError("selector model file receipt differs")
    if model_record["format"] != MODEL_FORMAT_BY_FAMILY.get(family):
        raise SelectorRuntimeError("selector model format/family binding differs")
    model_path = _ordinary_child(root, str(model_record["name"]))
    try:
        model_payload, model_size = _authenticated_file_bytes(
            model_path,
            expected_sha256=str(model_record["sha256"]),
        )
    except SelectorRuntimeError as exc:
        raise SelectorRuntimeError("selector model bytes differ from manifest") from exc
    if model_size != model_record["size_bytes"]:
        raise SelectorRuntimeError("selector model bytes differ from manifest")

    def load_preprocessor(record: object) -> PersistedPreprocessorState:
        if not isinstance(record, dict) or set(record) != {"name", "sha256", "size_bytes"}:
            raise SelectorRuntimeError("selector preprocessor file receipt differs")
        path = _ordinary_child(root, str(record["name"]))
        try:
            payload, size = _authenticated_file_bytes(
                path,
                expected_sha256=str(record["sha256"]),
            )
        except SelectorRuntimeError as exc:
            raise SelectorRuntimeError(
                "selector preprocessor bytes differ from manifest"
            ) from exc
        if size != record["size_bytes"]:
            raise SelectorRuntimeError("selector preprocessor bytes differ from manifest")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SelectorRuntimeError("selector preprocessor JSON is invalid") from exc
        if not isinstance(value, dict):
            raise SelectorRuntimeError("selector preprocessor JSON is not an object")
        return PersistedPreprocessorState.from_payload(value)

    full = load_preprocessor(manifest["full_preprocessor"])
    inner = (
        None
        if manifest["inner_preprocessor"] is None
        else load_preprocessor(manifest["inner_preprocessor"])
    )
    if full.sha256 != identity_payload.get("full_preprocessor_sha256"):
        raise SelectorRuntimeError("full preprocessor identity binding differs")
    if (None if inner is None else inner.sha256) != identity_payload.get(
        "inner_preprocessor_sha256"
    ):
        raise SelectorRuntimeError("inner preprocessor identity binding differs")
    fit_receipt = manifest["fit_receipt"]
    if not isinstance(fit_receipt, dict):
        raise SelectorRuntimeError("selector fit receipt is not an object")
    _reject_empirical_receipt_fields(fit_receipt)
    if manifest["fit_receipt_sha256"] != _sha256_bytes(
        canonical_json_bytes(fit_receipt)
    ):
        raise SelectorRuntimeError("selector fit receipt digest differs")
    return LoadedSelectorCheckpoint(
        root=root,
        manifest=manifest,
        identity=identity_payload,
        model_payload=model_payload,
        inner_preprocessor=inner,
        full_preprocessor=full,
    )


@dataclass(frozen=True, slots=True)
class ForecastBatchReceipt:
    forecast_batch_id: str
    checkpoint_id: str
    checkpoint_manifest_sha256: str
    family: str
    config_id: str
    seed: int | None
    refit_signal_date: dt.date
    signal_date: dt.date
    entry_date: dt.date
    exit_date: dt.date
    predictor_availability_date_max: dt.date
    batch_index: int
    batch_count: int
    rows: int
    first_permno: int
    last_permno: int
    permnos_sha256: str
    raw_forecasts_sha256: str
    payload_sha256: str
    payload_path: str
    outcome_rows_read: int = 0
    forward_label_rows_read: int = 0
    portfolio_rows_read: int = 0
    final_period_rows_read: int = 0
    status: str = FORECAST_BATCH_STATUS

    def __post_init__(self) -> None:
        _identifier(self.forecast_batch_id, name="forecast_batch_id")
        _identifier(self.checkpoint_id, name="checkpoint_id")
        validate_locked_selector(self.family, self.config_id, self.seed)
        for name, value in (
            ("checkpoint_manifest_sha256", self.checkpoint_manifest_sha256),
            ("permnos_sha256", self.permnos_sha256),
            ("raw_forecasts_sha256", self.raw_forecasts_sha256),
            ("payload_sha256", self.payload_sha256),
        ):
            _require_sha256(value, name=name)
        refit = _date(self.refit_signal_date, name="refit_signal_date")
        signal = _date(self.signal_date, name="signal_date")
        entry = _date(self.entry_date, name="entry_date")
        exit_date = _date(self.exit_date, name="exit_date")
        available = _date(
            self.predictor_availability_date_max,
            name="predictor_availability_date_max",
        )
        if not refit <= signal < entry < exit_date or available > signal:
            raise SelectorRuntimeError("raw forecast chronology is invalid")
        if (
            type(self.batch_index) is not int
            or type(self.batch_count) is not int
            or not 0 <= self.batch_index < self.batch_count
            or self.rows <= 0
            or self.first_permno <= 0
            or self.last_permno < self.first_permno
        ):
            raise SelectorRuntimeError("raw forecast batch dimensions are invalid")
        counters = (
            self.outcome_rows_read,
            self.forward_label_rows_read,
            self.portfolio_rows_read,
            self.final_period_rows_read,
        )
        if any(type(value) is not int or value != 0 for value in counters):
            raise SelectorRuntimeError("forecast scoring read a prohibited empirical payload")
        if self.status != FORECAST_BATCH_STATUS:
            raise SelectorRuntimeError("raw forecast batch status differs")

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "refit_signal_date",
            "signal_date",
            "entry_date",
            "exit_date",
            "predictor_availability_date_max",
        ):
            result[name] = getattr(self, name).isoformat()
        return result


def publish_forecast_batch(
    *,
    forecast_root: Path,
    checkpoint: LoadedSelectorCheckpoint,
    signal_date: dt.date,
    entry_date: dt.date,
    exit_date: dt.date,
    predictor_availability_date_max: dt.date,
    batch_index: int,
    batch_count: int,
    permnos: Sequence[int],
    raw_forecasts: Sequence[float],
) -> ForecastBatchReceipt:
    """Publish one sorted raw-forecast shard without accepting any outcome."""

    signal = _date(signal_date, name="signal_date")
    permno_array = np.asarray(tuple(permnos), dtype=np.int64)
    forecasts = np.asarray(tuple(raw_forecasts), dtype=np.float32)
    if (
        permno_array.ndim != 1
        or forecasts.shape != permno_array.shape
        or permno_array.size == 0
        or np.any(permno_array <= 0)
        or not np.all(np.diff(permno_array) > 0)
        or not np.isfinite(forecasts).all()
    ):
        raise SelectorRuntimeError("raw forecast batch must have sorted unique finite rows")
    identity = checkpoint.identity
    refit = dt.date.fromisoformat(str(identity["refit_signal_date"]))
    if refit > signal:
        raise SelectorRuntimeError("forecast uses a model fitted after its signal date")
    checkpoint_manifest_sha = checkpoint.manifest_sha256
    batch_identity = {
        "runtime_id": SELECTOR_RUNTIME_ID,
        "checkpoint_id": checkpoint.checkpoint_id,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "family": identity["family"],
        "config_id": identity["config_id"],
        "seed": identity["seed"],
        "refit_signal_date": refit.isoformat(),
        "signal_date": signal.isoformat(),
        "entry_date": _date(entry_date, name="entry_date").isoformat(),
        "exit_date": _date(exit_date, name="exit_date").isoformat(),
        "predictor_availability_date_max": _date(
            predictor_availability_date_max,
            name="predictor_availability_date_max",
        ).isoformat(),
        "batch_index": batch_index,
        "batch_count": batch_count,
        "permnos_sha256": array_sha256(permno_array),
        "raw_forecasts_sha256": array_sha256(forecasts),
    }
    forecast_batch_id = f"stage11-forecast-{_sha256_bytes(canonical_json_bytes(batch_identity))[:32]}"
    root = _ordinary_root(forecast_root, create=True)
    payload_name = f"{forecast_batch_id}.arrays"
    payload_path = _ordinary_child(root, payload_name)
    payload = pack_numeric_arrays({"permnos": permno_array, "raw_forecasts": forecasts})
    expected_payload_sha = _sha256_bytes(payload)
    if not payload_path.exists():
        _atomic_file(payload_path, payload)
    try:
        authenticated_payload, _payload_size = _authenticated_file_bytes(
            payload_path,
            expected_sha256=expected_payload_sha,
        )
    except SelectorRuntimeError as exc:
        raise SelectorRuntimeError(
            "forecast batch ID collision changed payload bytes"
        ) from exc
    if authenticated_payload != payload:
        raise SelectorRuntimeError("forecast batch ID collision changed payload bytes")
    payload_sha = expected_payload_sha
    receipt = ForecastBatchReceipt(
        forecast_batch_id=forecast_batch_id,
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_manifest_sha256=checkpoint_manifest_sha,
        family=str(identity["family"]),
        config_id=str(identity["config_id"]),
        seed=identity["seed"],
        refit_signal_date=refit,
        signal_date=signal,
        entry_date=entry_date,
        exit_date=exit_date,
        predictor_availability_date_max=predictor_availability_date_max,
        batch_index=batch_index,
        batch_count=batch_count,
        rows=int(permno_array.size),
        first_permno=int(permno_array[0]),
        last_permno=int(permno_array[-1]),
        permnos_sha256=batch_identity["permnos_sha256"],
        raw_forecasts_sha256=batch_identity["raw_forecasts_sha256"],
        payload_sha256=payload_sha,
        payload_path=str(payload_path),
    )
    _atomic_file(
        _ordinary_child(root, f"{forecast_batch_id}.json"),
        canonical_json_bytes(receipt.payload()),
    )
    return receipt


@dataclass(frozen=True, slots=True)
class ForecastSetReceipt:
    forecast_set_id: str
    checkpoint_id: str
    checkpoint_manifest_sha256: str
    signal_date: dt.date
    entry_date: dt.date
    exit_date: dt.date
    batch_ids: tuple[str, ...]
    rows: int
    permnos_sha256: str
    raw_forecasts_sha256: str
    status: str = FORECAST_SET_STATUS


def validate_complete_forecast_batches(
    receipts: Iterable[ForecastBatchReceipt],
) -> ForecastSetReceipt:
    """Authenticate a complete, disjoint set of chronological forecast shards."""

    rows = tuple(receipts)
    if not rows:
        raise SelectorRuntimeError("forecast set cannot be empty")
    ordered = tuple(sorted(rows, key=lambda item: item.batch_index))
    expected_count = ordered[0].batch_count
    if (
        len(ordered) != expected_count
        or tuple(item.batch_index for item in ordered) != tuple(range(expected_count))
    ):
        raise SelectorRuntimeError("forecast batches are incomplete or duplicated")
    invariant = (
        "checkpoint_id",
        "checkpoint_manifest_sha256",
        "family",
        "config_id",
        "seed",
        "refit_signal_date",
        "signal_date",
        "entry_date",
        "exit_date",
        "predictor_availability_date_max",
    )
    for name in invariant:
        if len({getattr(item, name) for item in ordered}) != 1:
            raise SelectorRuntimeError(f"forecast batch invariant differs: {name}")
    all_permnos: list[np.ndarray] = []
    all_forecasts: list[np.ndarray] = []
    prior_last = 0
    for receipt in ordered:
        path = Path(receipt.payload_path)
        try:
            payload, _size = _authenticated_file_bytes(
                path,
                expected_sha256=receipt.payload_sha256,
            )
        except SelectorRuntimeError as exc:
            raise SelectorRuntimeError("forecast payload digest differs") from exc
        arrays = unpack_numeric_arrays(payload)
        if set(arrays) != {"permnos", "raw_forecasts"}:
            raise SelectorRuntimeError("forecast payload arrays differ")
        permnos = np.asarray(arrays["permnos"], dtype=np.int64)
        forecasts = np.asarray(arrays["raw_forecasts"], dtype=np.float32)
        if (
            array_sha256(permnos) != receipt.permnos_sha256
            or array_sha256(forecasts) != receipt.raw_forecasts_sha256
            or permnos.size != receipt.rows
            or int(permnos[0]) != receipt.first_permno
            or int(permnos[-1]) != receipt.last_permno
            or int(permnos[0]) <= prior_last
        ):
            raise SelectorRuntimeError("forecast payload semantics differ from receipt")
        prior_last = int(permnos[-1])
        all_permnos.append(permnos)
        all_forecasts.append(forecasts)
    combined_permnos = np.concatenate(all_permnos)
    combined_forecasts = np.concatenate(all_forecasts)
    set_identity = {
        "checkpoint_id": ordered[0].checkpoint_id,
        "checkpoint_manifest_sha256": ordered[0].checkpoint_manifest_sha256,
        "signal_date": ordered[0].signal_date.isoformat(),
        "entry_date": ordered[0].entry_date.isoformat(),
        "exit_date": ordered[0].exit_date.isoformat(),
        "batch_ids": [item.forecast_batch_id for item in ordered],
        "rows": int(combined_permnos.size),
        "permnos_sha256": array_sha256(combined_permnos),
        "raw_forecasts_sha256": array_sha256(combined_forecasts),
    }
    return ForecastSetReceipt(
        forecast_set_id=f"stage11-forecast-set-{_sha256_bytes(canonical_json_bytes(set_identity))[:32]}",
        checkpoint_id=ordered[0].checkpoint_id,
        checkpoint_manifest_sha256=ordered[0].checkpoint_manifest_sha256,
        signal_date=ordered[0].signal_date,
        entry_date=ordered[0].entry_date,
        exit_date=ordered[0].exit_date,
        batch_ids=tuple(item.forecast_batch_id for item in ordered),
        rows=int(combined_permnos.size),
        permnos_sha256=set_identity["permnos_sha256"],
        raw_forecasts_sha256=set_identity["raw_forecasts_sha256"],
    )


__all__ = [
    "AggregateMomentState",
    "CHECKPOINT_STATUS",
    "FORECAST_BATCH_STATUS",
    "FORECAST_SET_STATUS",
    "ForecastBatchReceipt",
    "ForecastSetReceipt",
    "FileBackedArrayBinding",
    "FileBackedArrayRequest",
    "LoadedSelectorCheckpoint",
    "MODEL_FORMAT_BY_FAMILY",
    "MATRIX_KIND_BY_FAMILY",
    "NativeModelArtifact",
    "PersistedPreprocessorState",
    "ReadOnlyValidationRefitArrays",
    "SELECTOR_RUNTIME_ID",
    "SelectorRuntimeError",
    "ValidationRefitArrays",
    "ValidationRefitIdentity",
    "ValidationRefitPathRequest",
    "array_sha256",
    "build_refit_identity",
    "canonical_json_bytes",
    "load_selector_checkpoint",
    "load_array_for_spawn",
    "load_validation_refit_arrays_from_paths",
    "pack_numeric_arrays",
    "prepare_validation_refit_arrays",
    "persist_array_for_spawn",
    "persist_validation_refit_arrays_for_spawn",
    "publish_forecast_batch",
    "publish_selector_checkpoint",
    "unpack_numeric_arrays",
    "validate_complete_forecast_batches",
    "validate_locked_selector",
]
