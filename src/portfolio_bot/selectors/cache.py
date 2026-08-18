"""Authenticated, resumable cache helpers for the long Stage 9 search."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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


def _anchored_cache_path(
    path: Path,
    *,
    trusted_root: Path,
    create_parent: bool,
    require_file: bool,
) -> Path:
    root = Path(os.path.abspath(trusted_root.expanduser()))
    candidate = Path(os.path.abspath(path.expanduser()))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Stage 9 cache path escaped its trusted root") from exc
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("Stage 9 cache root is absent, non-directory, or symlinked")
    if create_parent:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise RuntimeError(f"Symlinked Stage 9 cache component is prohibited: {current}")
    try:
        candidate.parent.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Stage 9 cache parent escaped its trusted root") from exc
    if require_file:
        try:
            status = candidate.stat(follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(f"Stage 9 cache file is absent: {candidate}") from exc
        if not stat.S_ISREG(status.st_mode):
            raise RuntimeError(f"Stage 9 cache object is not a regular file: {candidate}")
    elif candidate.exists() and candidate.is_symlink():
        raise RuntimeError(f"Symlinked Stage 9 cache output is prohibited: {candidate}")
    return candidate


def _stable_sha256(path: Path) -> tuple[str, tuple[int, ...]]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"Stage 9 cache object is not a regular file: {path}")
    digest = sha256_file(path)
    after = path.stat(follow_symlinks=False)
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"Stage 9 cache file changed during hashing: {path}")
    return digest, _stat_identity(after)


def _stable_read_json(path: Path) -> dict[str, Any]:
    before = path.stat(follow_symlinks=False)
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid Stage 9 cache receipt: {path}") from exc
    after = path.stat(follow_symlinks=False)
    if _stat_identity(before) != _stat_identity(after) or not isinstance(value, dict):
        raise RuntimeError(f"Stage 9 cache receipt changed or is not an object: {path}")
    return value


def job_id_from_identity(identity: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(dict(identity))
    return f"stage09-fit-{hashlib.sha256(payload).hexdigest()}"


def _normalized_array_schema(schema: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for name, value in schema.items():
        if not isinstance(name, str) or not name or not isinstance(value, Mapping):
            raise ValueError("Invalid Stage 9 expected array schema")
        dtype = np.dtype(str(value.get("dtype")))
        if dtype.hasobject:
            raise ValueError("Object arrays are prohibited in the Stage 9 cache")
        shape_value = value.get("shape")
        if not isinstance(shape_value, (list, tuple)):
            raise ValueError("Stage 9 expected array shape is missing")
        shape = tuple(int(item) for item in shape_value)
        if any(item < 0 for item in shape):
            raise ValueError("Stage 9 expected array shape is negative")
        normalized[name] = {"dtype": dtype.str, "shape": list(shape)}
    if "predictions" not in normalized:
        raise ValueError("Stage 9 cache schema must include predictions")
    return normalized


def _validated_arrays(
    arrays: Mapping[str, np.ndarray],
    expected_schema: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    schema = _normalized_array_schema(expected_schema)
    if set(arrays) != set(schema):
        raise ValueError("Stage 9 cache array names differ from the external schema")
    result: dict[str, np.ndarray] = {}
    for name in schema:
        array = np.asarray(arrays[name])
        expected = schema[name]
        if array.dtype.str != expected["dtype"] or list(array.shape) != expected["shape"]:
            raise ValueError(f"Stage 9 cache array schema differs: {name}")
        if array.dtype.hasobject:
            raise ValueError(f"Object Stage 9 cache array is prohibited: {name}")
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"Nonfinite Stage 9 cache array is prohibited: {name}")
        result[name] = array
    return result


def atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    trusted_root: Path,
) -> str:
    path = _anchored_cache_path(
        path,
        trusted_root=trusted_root,
        create_parent=True,
        require_file=False,
    )
    payload = canonical_json_bytes(dict(value))
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
    return hashlib.sha256(payload).hexdigest()


def atomic_write_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    trusted_root: Path,
) -> str:
    path = _anchored_cache_path(
        path,
        trusted_root=trusted_root,
        create_parent=True,
        require_file=False,
    )
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        digest, _ = _stable_sha256(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def load_completed_job(
    path: Path,
    *,
    trusted_root: Path,
    expected_identity: Mapping[str, Any],
    expected_array_schema: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return a cache hit only when identity, bytes, and array schema all match."""

    try:
        path = _anchored_cache_path(
            path,
            trusted_root=trusted_root,
            create_parent=False,
            require_file=True,
        )
        receipt_path = _anchored_cache_path(
            path.with_suffix(".json"),
            trusted_root=trusted_root,
            create_parent=False,
            require_file=True,
        )
        receipt = _stable_read_json(receipt_path)
        identity = dict(expected_identity)
        identity_sha = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        expected_job_id = job_id_from_identity(identity)
        prediction_sha, archive_identity = _stable_sha256(path)
        normalized_schema = _normalized_array_schema(expected_array_schema)
        if (
            receipt.get("schema_version") != 2
            or receipt.get("status") != "COMPLETE"
            or receipt.get("job_id") != expected_job_id
            or receipt.get("job_identity") != identity
            or receipt.get("job_identity_sha256") != identity_sha
            or receipt.get("prediction_sha256") != prediction_sha
            or receipt.get("array_schema") != normalized_schema
        ):
            return None
        fit_receipt = receipt.get("fit_receipt")
        if not isinstance(fit_receipt, Mapping) or receipt.get("fit_receipt_sha256") != hashlib.sha256(
            canonical_json_bytes(dict(fit_receipt))
        ).hexdigest():
            return None
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(normalized_schema):
                return None
            loaded = {name: archive[name] for name in normalized_schema}
            _validated_arrays(loaded, normalized_schema)
        if _stat_identity(path.stat(follow_symlinks=False)) != archive_identity:
            return None
        return receipt
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return None


def publish_completed_job(
    path: Path,
    *,
    trusted_root: Path,
    job_identity: Mapping[str, Any],
    predictions: np.ndarray,
    fit_receipt: Mapping[str, Any],
    expected_array_schema: Mapping[str, Mapping[str, Any]],
    extra_arrays: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    identity = dict(job_identity)
    arrays = {"predictions": np.asarray(predictions)}
    arrays.update({} if extra_arrays is None else extra_arrays)
    schema = _normalized_array_schema(expected_array_schema)
    arrays = _validated_arrays(arrays, schema)
    prediction_sha = atomic_write_npz(path, arrays, trusted_root=trusted_root)
    fit = dict(fit_receipt)
    receipt = {
        "schema_version": 2,
        "job_id": job_id_from_identity(identity),
        "job_identity": identity,
        "job_identity_sha256": hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
        "status": "COMPLETE",
        "prediction_sha256": prediction_sha,
        "array_schema": schema,
        "fit_receipt": fit,
        "fit_receipt_sha256": hashlib.sha256(canonical_json_bytes(fit)).hexdigest(),
    }
    atomic_write_json(path.with_suffix(".json"), receipt, trusted_root=trusted_root)
    return receipt


__all__ = [
    "atomic_write_json",
    "atomic_write_npz",
    "canonical_json_bytes",
    "job_id_from_identity",
    "load_completed_job",
    "publish_completed_job",
    "sha256_file",
]
