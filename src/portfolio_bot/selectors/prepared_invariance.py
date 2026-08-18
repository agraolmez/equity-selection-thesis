"""Scientific old-vs-corrected audit for Stage 9 prepared matrices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SPLIT_SENSITIVE_FEATURES = (
    "price_to_ma_21d",
    "price_to_ma_63d",
    "price_to_ma_126d",
    "price_to_ma_252d",
    "distance_to_high_252d",
    "log_price_trend_slope_63d",
    "log_price_trend_slope_126d",
    "log_price_trend_slope_252d",
    "log_price_trend_r2_63d",
    "log_price_trend_r2_126d",
    "log_price_trend_r2_252d",
)


class PreparedInvarianceError(RuntimeError):
    """Raised when the corrected prepared cache changes an unapproved value."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparedInvarianceError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
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


def _validate_cache(root: Path, expected_build_id: str) -> dict[str, Any]:
    _require(root.is_dir() and not root.is_symlink(), f"Unsafe prepared root: {root}")
    metadata = _json(root / "metadata.json")
    _require(metadata.get("build_id") == expected_build_id, "Prepared build ID differs")
    _require(metadata.get("status") == "PREPARED_COMPLETE", "Prepared cache is incomplete")
    scope = metadata.get("scope")
    _require(isinstance(scope, Mapping), "Prepared scope is missing")
    _require(scope.get("development_only") is True, "Prepared cache is not development-only")
    _require(
        scope.get("validation_or_final_payload_accessed") is False,
        "Prepared cache reports validation/final access",
    )
    _require(scope.get("model_fits_performed") is False, "Prepared cache contains model fits")
    files = metadata.get("files")
    _require(isinstance(files, Mapping), "Prepared file registry is missing")
    for name, registered in files.items():
        _require(isinstance(name, str) and isinstance(registered, Mapping), "Bad file receipt")
        path = root / name
        receipt_path = root / f"{name}.receipt.json"
        _require(path.is_file() and not path.is_symlink(), f"Missing prepared file: {name}")
        _require(receipt_path.is_file(), f"Missing prepared receipt: {name}")
        receipt = _json(receipt_path)
        _require(receipt == dict(registered), f"Metadata/receipt mismatch: {name}")
        _require(path.stat().st_size == int(receipt["size_bytes"]), f"Size differs: {name}")
    return metadata


def _read_exact_table(old_root: Path, new_root: Path, name: str) -> pa.Table:
    old = pq.read_table(old_root / name, use_threads=False)
    new = pq.read_table(new_root / name, use_threads=False)
    _require(old.schema.equals(new.schema, check_metadata=True), f"Schema differs: {name}")
    _require(old.equals(new, check_metadata=True), f"Scientific table differs: {name}")
    return new


def _take(table: pa.Table, indices: np.ndarray) -> pa.Table:
    return table.take(pa.array(indices.astype(np.int64, copy=False)))


def _compare_support_table(
    old_root: Path,
    new_root: Path,
    name: str,
    allowed_indices: np.ndarray,
) -> dict[str, Any]:
    old = pq.read_table(old_root / name, use_threads=False)
    new = pq.read_table(new_root / name, use_threads=False)
    _require(old.schema.equals(new.schema, check_metadata=True), f"Schema differs: {name}")
    _require(old.num_rows == new.num_rows, f"Rows differ: {name}")
    forbidden = np.setdiff1d(
        np.arange(old.num_rows, dtype=np.int64),
        allowed_indices,
        assume_unique=True,
    )
    _require(
        _take(old, forbidden).equals(_take(new, forbidden), check_metadata=True),
        f"Unapproved support rows changed: {name}",
    )
    allowed_changed = not _take(old, allowed_indices).equals(
        _take(new, allowed_indices), check_metadata=True
    )
    return {
        "allowed_rows": int(allowed_indices.size),
        "allowed_rows_changed": allowed_changed,
        "forbidden_rows": int(forbidden.size),
        "forbidden_rows_changed": 0,
    }


def compare_matrix(
    old_path: Path,
    new_path: Path,
    *,
    allowed_indices: np.ndarray,
    column_names: Sequence[str],
    batch_rows: int,
) -> dict[str, Any]:
    old = np.load(old_path, mmap_mode="r", allow_pickle=False)
    new = np.load(new_path, mmap_mode="r", allow_pickle=False)
    _require(old.shape == new.shape, f"Matrix shape differs: {old_path.name}")
    _require(old.dtype == new.dtype == np.dtype("float32"), "Matrix dtype differs")
    _require(old.ndim == 2 and old.shape[1] == len(column_names), "Column registry differs")
    allowed = np.asarray(sorted(set(int(value) for value in allowed_indices)), dtype=np.int64)
    forbidden = np.setdiff1d(
        np.arange(old.shape[1], dtype=np.int64), allowed, assume_unique=True
    )
    changed_by_column = np.zeros(old.shape[1], dtype=np.int64)
    maximum_absolute_change = 0.0
    for start in range(0, old.shape[0], batch_rows):
        stop = min(start + batch_rows, old.shape[0])
        left = np.asarray(old[start:stop])
        right = np.asarray(new[start:stop])
        _require(np.isfinite(left).all() and np.isfinite(right).all(), "Nonfinite matrix value")
        changed = left.view(np.uint32) != right.view(np.uint32)
        changed_by_column += changed.sum(axis=0, dtype=np.int64)
        if allowed.size:
            maximum_absolute_change = max(
                maximum_absolute_change,
                float(
                    np.max(
                        np.abs(
                            left[:, allowed].astype(np.float64)
                            - right[:, allowed].astype(np.float64)
                        )
                    )
                ),
            )
    forbidden_changed = int(changed_by_column[forbidden].sum())
    _require(forbidden_changed == 0, f"Unapproved matrix cells changed: {old_path.name}")
    changed_allowed = {
        str(column_names[index]): int(changed_by_column[index])
        for index in allowed
        if changed_by_column[index] > 0
    }
    _require(changed_allowed, f"No corrected values changed: {old_path.name}")
    return {
        "allowed_column_count": int(allowed.size),
        "allowed_columns_changed": len(changed_allowed),
        "allowed_changed_cells": int(changed_by_column[allowed].sum()),
        "changed_allowed_columns": changed_allowed,
        "forbidden_column_count": int(forbidden.size),
        "forbidden_changed_cells": forbidden_changed,
        "maximum_absolute_change": maximum_absolute_change,
        "shape": [int(value) for value in old.shape],
    }


def compare_prepared_caches(
    old_root: Path,
    new_root: Path,
    *,
    old_build_id: str,
    new_build_id: str,
    batch_rows: int = 4096,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    _require(batch_rows > 0, "batch_rows must be positive")
    report = progress or (lambda _message: None)
    old_metadata = _validate_cache(old_root, old_build_id)
    new_metadata = _validate_cache(new_root, new_build_id)
    report("authenticated both prepared-cache registries")

    exact_tables = (
        "rows.parquet",
        "split_membership.parquet",
        "aggregate_history.parquet",
        "base_column_dictionary.parquet",
        "elastic_interaction_dictionary.parquet",
        *(f"category_mapping_{scope}_dev_fold_{fold}.parquet" for fold in range(1, 5) for scope in ("full", "inner")),
    )
    loaded: dict[str, pa.Table] = {}
    for name in exact_tables:
        loaded[name] = _read_exact_table(old_root, new_root, name)
    report(f"proved {len(exact_tables)} non-matrix scientific tables exact")

    rows = loaded["rows.parquet"]
    max_signal_date = pc.max(rows["signal_date"]).as_py()
    _require(str(max_signal_date) <= "2014-10-31", "Prepared rows extend beyond development")
    base_dictionary = loaded["base_column_dictionary.parquet"].to_pylist()
    _require(len(base_dictionary) == 551, "Base width differs")
    base_names = [str(row["column_name"]) for row in base_dictionary]
    allowed_base = np.asarray(
        [
            index
            for index, row in enumerate(base_dictionary)
            if str(row["source_feature_name"]) in SPLIT_SENSITIVE_FEATURES
        ],
        dtype=np.int64,
    )
    _require(allowed_base.size == 22, "Expected 22 split-sensitive base columns")

    interaction_dictionary = loaded["elastic_interaction_dictionary.parquet"].to_pylist()
    _require(len(interaction_dictionary) == 2806, "Interaction width differs")
    interaction_names = [str(row["interaction_column"]) for row in interaction_dictionary]
    allowed_interactions = np.asarray(
        [
            551 + index
            for index, row in enumerate(interaction_dictionary)
            if str(row["firm_column"]) in SPLIT_SENSITIVE_FEATURES
        ],
        dtype=np.int64,
    )
    allowed_elastic = np.sort(np.concatenate((allowed_base, allowed_interactions)))
    _require(allowed_interactions.size == 253, "Expected 253 split-sensitive interactions")
    _require(allowed_elastic.size == 275, "Expected 275 split-sensitive Elastic-Net columns")
    elastic_names = [*base_names, *interaction_names]

    support_results: dict[str, Any] = {}
    matrix_results: dict[str, Any] = {}
    for fold in range(1, 5):
        for scope in ("full", "inner"):
            support_name = f"base_support_{scope}_dev_fold_{fold}.parquet"
            support_results[support_name] = _compare_support_table(
                old_root, new_root, support_name, allowed_base
            )
            matrix_name = f"base_matrix_{scope}_dev_fold_{fold}.npy"
            matrix_results[matrix_name] = compare_matrix(
                old_root / matrix_name,
                new_root / matrix_name,
                allowed_indices=allowed_base,
                column_names=base_names,
                batch_rows=batch_rows,
            )
            report(f"compared {matrix_name}")
        support_name = f"elastic_support_full_dev_fold_{fold}.parquet"
        support_results[support_name] = _compare_support_table(
            old_root, new_root, support_name, allowed_elastic
        )
        matrix_name = f"elastic_matrix_full_dev_fold_{fold}.npy"
        matrix_results[matrix_name] = compare_matrix(
            old_root / matrix_name,
            new_root / matrix_name,
            allowed_indices=allowed_elastic,
            column_names=elastic_names,
            batch_rows=batch_rows,
        )
        report(f"compared {matrix_name}")

    old_masks = np.load(old_root / "support_masks.npz", allow_pickle=False)
    new_masks = np.load(new_root / "support_masks.npz", allow_pickle=False)
    _require(old_masks.files == new_masks.files, "Support-mask registry differs")
    mask_results: dict[str, Any] = {}
    for name in old_masks.files:
        old_mask = old_masks[name]
        new_mask = new_masks[name]
        _require(old_mask.shape == new_mask.shape, f"Support-mask shape differs: {name}")
        allowed = allowed_elastic if "elastic" in name else allowed_base
        forbidden = np.setdiff1d(
            np.arange(old_mask.size, dtype=np.int64), allowed, assume_unique=True
        )
        _require(
            np.array_equal(old_mask[forbidden], new_mask[forbidden]),
            f"Unapproved support mask changed: {name}",
        )
        mask_results[name] = {
            "allowed_bits_changed": int(np.count_nonzero(old_mask[allowed] != new_mask[allowed])),
            "forbidden_bits_changed": 0,
        }

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed_changes_confined_to_split_sensitive_prepared_columns",
        "old_build_id": old_build_id,
        "new_build_id": new_build_id,
        "old_content_fingerprint": old_metadata["content_fingerprint"],
        "new_content_fingerprint": new_metadata["content_fingerprint"],
        "scope": {
            "accessed_2015_or_later_payload": False,
            "maximum_signal_date": str(max_signal_date),
            "model_fits_performed": False,
            "rows": rows.num_rows,
        },
        "allowed_change_set": {
            "split_sensitive_features": list(SPLIT_SENSITIVE_FEATURES),
            "base_columns": [base_names[index] for index in allowed_base],
            "base_column_count": int(allowed_base.size),
            "elastic_interaction_count": int(allowed_interactions.size),
            "elastic_total_column_count": int(allowed_elastic.size),
        },
        "exact_tables": list(exact_tables),
        "support_tables": support_results,
        "support_masks": mask_results,
        "matrices": matrix_results,
    }
    identity = {
        "old_build_id": old_build_id,
        "new_build_id": new_build_id,
        "status": receipt["status"],
        "matrix_results": matrix_results,
    }
    receipt["comparison_id"] = (
        "stage09-prepared-comparison-"
        + _sha256_bytes(_canonical_json_bytes(identity))[:16]
    )
    report("proved all matrix changes are confined to the approved columns")
    return receipt


def write_receipt(receipt: Mapping[str, Any], path: Path) -> str:
    payload = _canonical_json_bytes(dict(receipt))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


__all__ = [
    "PreparedInvarianceError",
    "SPLIT_SENSITIVE_FEATURES",
    "compare_matrix",
    "compare_prepared_caches",
    "write_receipt",
]
