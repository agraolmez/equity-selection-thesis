"""Authenticated loader for the immutable Stage 9 prepared-matrix cache."""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from portfolio_bot.selectors.cache import canonical_json_bytes, sha256_file
from portfolio_bot.selectors.contract import (
    BASE_MATRIX_WIDTH,
    CONTRACT_ID,
    ELASTIC_NET_INTERACTION_COUNT,
    ELASTIC_NET_TOTAL_WIDTH,
)
from portfolio_bot.selectors.inputs import LABEL_COLUMNS, SPLIT_COLUMNS
from portfolio_bot.selectors.splits import FoldIndices, build_fold_indices


EXPECTED_ROWS = 286_628
EXPECTED_SPLIT_ROWS = 712
EXPECTED_PRESENT_TARGETS = 286_322
EXPECTED_OOF_ROWS = 172_969
EXPECTED_OOF_PRESENT = 172_799
EXPECTED_FOLDS = ("dev_fold_1", "dev_fold_2", "dev_fold_3", "dev_fold_4")


@dataclass(frozen=True)
class PreparedSelectorData:
    root: Path
    metadata: dict
    rows: pd.DataFrame
    split_membership: pd.DataFrame
    folds: tuple[FoldIndices, ...]
    inner_base_matrices: dict[str, np.ndarray]
    full_base_matrices: dict[str, np.ndarray]
    elastic_matrices: dict[str, np.ndarray]
    inner_base_support: dict[str, np.ndarray]
    full_base_support: dict[str, np.ndarray]
    elastic_support: dict[str, np.ndarray]


@dataclass(frozen=True)
class PreparedSelectorFoldData:
    """One authenticated fold with only the requested native-family matrices."""

    root: Path
    metadata: dict
    rows: pd.DataFrame
    split_membership: pd.DataFrame
    fold: FoldIndices
    family: Literal["neural_network", "xgboost", "elastic_net"]
    inner_base_matrix: np.ndarray | None
    full_base_matrix: np.ndarray | None
    elastic_matrix: np.ndarray | None
    inner_base_support: np.ndarray | None
    full_base_support: np.ndarray | None
    elastic_support: np.ndarray | None


def _expected_files() -> set[str]:
    shared = {
        "rows.parquet",
        "split_membership.parquet",
        "base_column_dictionary.parquet",
        "elastic_interaction_dictionary.parquet",
        "aggregate_history.parquet",
        "support_masks.npz",
    }
    for fold in EXPECTED_FOLDS:
        shared.update(
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
    return shared


def _authenticated_path(root: Path, name: str, metadata: dict) -> Path:
    path = root / name
    receipt = metadata["files"][name]
    sidecar = path.with_name(f"{path.name}.receipt.json")
    try:
        sidecar_receipt = json.loads(sidecar.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Prepared Stage 9 cache receipt failed authentication: {name}"
        ) from exc
    if (
        receipt.get("schema_version") != 1
        or
        receipt.get("name") != name
        or receipt.get("status") != "COMPLETE"
        or receipt.get("kind")
        != ("parquet" if name.endswith(".parquet") else "npy" if name.endswith(".npy") else "npz")
        or receipt.get("build_id") != metadata["build_id"]
        or receipt.get("content_fingerprint") != metadata["content_fingerprint"]
        or not path.is_file()
        or path.is_symlink()
        or not sidecar.is_file()
        or sidecar.is_symlink()
        or sidecar_receipt != receipt
        or not isinstance(receipt.get("size_bytes"), int)
        or receipt.get("size_bytes") <= 0
        or path.stat().st_size != receipt.get("size_bytes")
        or sha256_file(path) != receipt.get("sha256")
    ):
        raise RuntimeError(f"Prepared Stage 9 cache file failed authentication: {name}")
    return path


def _strict_sorted_keys(rows: pd.DataFrame) -> None:
    if rows[["signal_date", "permno"]].isna().any().any():
        raise RuntimeError("Prepared Stage 9 rows contain null keys")
    if rows.duplicated(["signal_date", "permno"]).any():
        raise RuntimeError("Prepared Stage 9 rows contain duplicate keys")
    dates = pd.to_datetime(rows["signal_date"], errors="raise").to_numpy()
    permnos = pd.to_numeric(rows["permno"], errors="raise").to_numpy(np.int64)
    if len(rows) > 1:
        bad = (dates[1:] < dates[:-1]) | ((dates[1:] == dates[:-1]) & (permnos[1:] <= permnos[:-1]))
        if bool(np.any(bad)):
            raise RuntimeError("Prepared Stage 9 row keys are not strict sorted")


def _validate_dictionaries(root: Path) -> None:
    base = pq.read_table(root / "base_column_dictionary.parquet", use_threads=False).to_pandas()
    interactions = pq.read_table(
        root / "elastic_interaction_dictionary.parquet", use_threads=False
    ).to_pandas()
    if (
        len(base) != BASE_MATRIX_WIDTH
        or base["column_ordinal"].tolist() != list(range(1, BASE_MATRIX_WIDTH + 1))
        or base["column_name"].duplicated().any()
    ):
        raise RuntimeError("Prepared Stage 9 base-column dictionary differs")
    if (
        len(interactions) != ELASTIC_NET_INTERACTION_COUNT
        or interactions["interaction_ordinal"].tolist()
        != list(range(1, ELASTIC_NET_INTERACTION_COUNT + 1))
        or interactions["interaction_column"].duplicated().any()
    ):
        raise RuntimeError("Prepared Stage 9 interaction dictionary differs")
    continuous = base.loc[
        base["source_kind"].eq("continuous_firm") & base["encoding"].eq("numeric_value"),
        "column_name",
    ].astype(str).tolist()
    aggregate = base.loc[
        base["source_kind"].eq("aggregate_state") & base["encoding"].eq("numeric_value"),
        "column_name",
    ].astype(str).tolist()
    expected_firm = np.repeat(np.asarray(continuous, dtype=object), len(aggregate))
    expected_aggregate = np.tile(np.asarray(aggregate, dtype=object), len(continuous))
    if (
        len(continuous) != 122
        or len(aggregate) != 23
        or not np.array_equal(interactions["firm_column"].astype(str).to_numpy(), expected_firm)
        or not np.array_equal(
            interactions["aggregate_column"].astype(str).to_numpy(), expected_aggregate
        )
    ):
        raise RuntimeError("Prepared Stage 9 interaction order differs")


def _safe_root(root: Path) -> Path:
    root = Path(os.path.abspath(root.expanduser()))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise RuntimeError("Unsafe or missing Stage 9 prepared-cache root")
    if not root.is_dir():
        raise RuntimeError("Unsafe or missing Stage 9 prepared-cache root")
    return root


def _load_authenticated_metadata(
    root: Path,
    *,
    expected_build_id: str,
    expected_content_fingerprint: str,
) -> dict:
    root = _safe_root(root)
    metadata_path = root / "metadata.json"
    if (
        not metadata_path.is_file()
        or metadata_path.is_symlink()
        or metadata_path.stat().st_size <= 0
    ):
        raise RuntimeError("Missing safe Stage 9 prepared-cache metadata")
    try:
        metadata = json.loads(metadata_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Invalid Stage 9 prepared-cache metadata") from exc

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
        or metadata.get("contract_id") != CONTRACT_ID
        or metadata.get("build_id") != expected_build_id
        or metadata.get("content_fingerprint") != expected_content_fingerprint
        or root.name != "prepared"
        or root.parent.name != expected_build_id
    ):
        raise RuntimeError("Stage 9 prepared cache is not bound to the trusted build identity")

    runtime = metadata.get("runtime_versions")
    code_hashes = metadata.get("code_hashes")
    configuration_hashes = metadata.get("configuration_hashes")
    input_receipts = metadata.get("input_receipts")
    required_runtime_names = {
        "python",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
    }
    if (
        not isinstance(runtime, dict)
        or set(runtime) != required_runtime_names
        or any(not isinstance(value, str) or not value for value in runtime.values())
        or not isinstance(code_hashes, dict)
        or "src/portfolio_bot/features/feature_math.py" not in code_hashes
        or not isinstance(configuration_hashes, dict)
        or not configuration_hashes
        or not isinstance(input_receipts, list)
        or not input_receipts
    ):
        raise RuntimeError("Prepared Stage 9 build dependency binding differs")
    identity_payload = {
        "schema_version": 2,
        "purpose": "stage09_prepared_selector_matrices_only",
        "contract_id": CONTRACT_ID,
        "configuration_hashes": configuration_hashes,
        "code_hashes": code_hashes,
        "runtime_versions": runtime,
        "input_receipts": input_receipts,
    }
    recomputed = hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
    if (
        recomputed != expected_content_fingerprint
        or expected_build_id != f"phase04-selector-prepared-{recomputed[:24]}"
    ):
        raise RuntimeError("Prepared Stage 9 metadata does not reproduce its trusted identity")

    expected_scope = {
        "development_only": True,
        "model_fits_performed": False,
        "stage10_or_later_permitted": False,
        "portfolio_or_backtest_permitted": False,
        "validation_or_final_payload_accessed": False,
    }
    expected_fit_scopes = {
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
    }
    if metadata.get("scope") != expected_scope:
        raise RuntimeError("Prepared Stage 9 cache scope differs")
    if metadata.get("preprocessing_fit_scopes") != expected_fit_scopes:
        raise RuntimeError("Prepared Stage 9 nested preprocessing scope differs")
    if (
        metadata.get("rows") != EXPECTED_ROWS
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
        raise RuntimeError("Prepared Stage 9 matrix contract differs")
    expected_files = _expected_files()
    if (
        set(metadata.get("required_files", [])) != expected_files
        or set(metadata.get("files", {})) != expected_files
        or any(not isinstance(value, dict) for value in metadata["files"].values())
    ):
        raise RuntimeError("Prepared Stage 9 closed file set differs")
    expected_directory_names = {
        "metadata.json",
        *expected_files,
        *(f"{name}.receipt.json" for name in expected_files),
    }
    if {path.name for path in root.iterdir()} != expected_directory_names:
        raise RuntimeError("Prepared Stage 9 cache directory is not a closed file set")
    return metadata


def _load_rows_and_folds(
    root: Path,
    metadata: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[FoldIndices, ...]]:
    rows_path = _authenticated_path(root, "rows.parquet", metadata)
    splits_path = _authenticated_path(root, "split_membership.parquet", metadata)
    rows = pq.read_table(rows_path, use_threads=False).to_pandas()
    splits = pq.read_table(splits_path, use_threads=False).to_pandas()
    if len(rows) != EXPECTED_ROWS or tuple(rows.columns) != LABEL_COLUMNS:
        raise RuntimeError("Prepared Stage 9 row table differs")
    if len(splits) != EXPECTED_SPLIT_ROWS or tuple(splits.columns) != SPLIT_COLUMNS:
        raise RuntimeError("Prepared Stage 9 split table differs")
    _strict_sorted_keys(rows)
    present = rows["primary_label_status"].eq("PRESENT") & np.isfinite(
        pd.to_numeric(rows["residual_return_rank"], errors="coerce")
    )
    if int(present.sum()) != EXPECTED_PRESENT_TARGETS:
        raise RuntimeError("Prepared Stage 9 PRESENT target domain differs")
    folds = build_fold_indices(rows, splits)
    if tuple(fold.fold_id for fold in folds) != EXPECTED_FOLDS:
        raise RuntimeError("Prepared Stage 9 fold identities differ")
    if (
        sum(fold.outer_validation.size for fold in folds) != EXPECTED_OOF_ROWS
        or sum(fold.outer_evaluation.size for fold in folds) != EXPECTED_OOF_PRESENT
    ):
        raise RuntimeError("Prepared Stage 9 OOF domain differs")
    return rows, splits, folds


def _support_archive(path: Path) -> np.lib.npyio.NpzFile:
    archive = np.load(path, allow_pickle=False)
    expected = {
        *(f"{fold}__base_inner" for fold in EXPECTED_FOLDS),
        *(f"{fold}__base_full" for fold in EXPECTED_FOLDS),
        *(f"{fold}__elastic_full" for fold in EXPECTED_FOLDS),
    }
    if set(archive.files) != expected:
        archive.close()
        raise RuntimeError("Prepared Stage 9 support archive differs")
    return archive


def _validate_support_receipt(
    path: Path,
    values: np.ndarray,
    *,
    expected_fit_rows: int | None,
) -> None:
    frame = pq.read_table(path, use_threads=False).to_pandas()
    if "active" not in frame or not np.array_equal(values, frame["active"].to_numpy(bool)):
        raise RuntimeError(f"Prepared support receipt disagrees: {path.name}")
    if expected_fit_rows is not None:
        counts = pd.to_numeric(
            frame.get("valid_outer_training_rows"), errors="coerce"
        ).dropna().unique()
        if counts.tolist() != [expected_fit_rows]:
            raise RuntimeError(f"Prepared support fit domain differs: {path.name}")


def load_prepared_selector_data(
    root: Path,
    *,
    expected_build_id: str,
    expected_content_fingerprint: str,
) -> PreparedSelectorData:
    """Load a cache only when bound to a caller-supplied trusted identity.

    The expected values must come from
    :func:`selectors.materialization.compute_prepared_build_identity` after
    re-authenticating the frozen Stage 6--8 inputs.  Metadata is never allowed
    to appoint itself as the trusted build.
    """

    root = _safe_root(root)
    metadata = _load_authenticated_metadata(
        root,
        expected_build_id=expected_build_id,
        expected_content_fingerprint=expected_content_fingerprint,
    )
    paths = {
        name: _authenticated_path(root, name, metadata)
        for name in sorted(_expected_files())
    }
    rows, splits, folds = _load_rows_and_folds(root, metadata)
    _validate_dictionaries(root)
    archive = _support_archive(paths["support_masks.npz"])

    inner_base_matrices: dict[str, np.ndarray] = {}
    full_base_matrices: dict[str, np.ndarray] = {}
    elastic_matrices: dict[str, np.ndarray] = {}
    inner_base_support: dict[str, np.ndarray] = {}
    full_base_support: dict[str, np.ndarray] = {}
    elastic_support: dict[str, np.ndarray] = {}
    for fold in folds:
        inner_base = np.load(
            paths[f"base_matrix_inner_{fold.fold_id}.npy"],
            mmap_mode="r",
            allow_pickle=False,
        )
        full_base = np.load(
            paths[f"base_matrix_full_{fold.fold_id}.npy"],
            mmap_mode="r",
            allow_pickle=False,
        )
        elastic = np.load(
            paths[f"elastic_matrix_full_{fold.fold_id}.npy"],
            mmap_mode="r",
            allow_pickle=False,
        )
        for scope, matrix in (("inner", inner_base), ("full", full_base)):
            if matrix.shape != (EXPECTED_ROWS, BASE_MATRIX_WIDTH) or matrix.dtype != np.dtype("float32"):
                raise RuntimeError(f"Prepared {scope} base matrix differs for {fold.fold_id}")
        if elastic.shape != (EXPECTED_ROWS, ELASTIC_NET_TOTAL_WIDTH) or elastic.dtype != np.dtype("float32"):
            raise RuntimeError(f"Prepared Elastic Net matrix differs for {fold.fold_id}")
        for start in range(0, EXPECTED_ROWS, 32_768):
            stop = min(start + 32_768, EXPECTED_ROWS)
            if not np.isfinite(inner_base[start:stop]).all():
                raise RuntimeError(f"Prepared inner base matrix is nonfinite for {fold.fold_id}")
            if not np.array_equal(full_base[start:stop], elastic[start:stop, :BASE_MATRIX_WIDTH]):
                raise RuntimeError(f"Elastic Net base prefix differs for {fold.fold_id}")
            if not np.isfinite(elastic[start:stop]).all():
                raise RuntimeError(f"Prepared Elastic matrix is nonfinite for {fold.fold_id}")
        inner_base_matrices[fold.fold_id] = inner_base
        full_base_matrices[fold.fold_id] = full_base
        elastic_matrices[fold.fold_id] = elastic
        inner_base_support[fold.fold_id] = np.asarray(
            archive[f"{fold.fold_id}__base_inner"], dtype=bool
        )
        full_base_support[fold.fold_id] = np.asarray(
            archive[f"{fold.fold_id}__base_full"], dtype=bool
        )
        elastic_support[fold.fold_id] = np.asarray(
            archive[f"{fold.fold_id}__elastic_full"], dtype=bool
        )
        if inner_base_support[fold.fold_id].shape != (BASE_MATRIX_WIDTH,):
            raise RuntimeError(f"Invalid inner base support mask for {fold.fold_id}")
        if full_base_support[fold.fold_id].shape != (BASE_MATRIX_WIDTH,):
            raise RuntimeError(f"Invalid full base support mask for {fold.fold_id}")
        if elastic_support[fold.fold_id].shape != (ELASTIC_NET_TOTAL_WIDTH,):
            raise RuntimeError(f"Invalid Elastic Net support mask for {fold.fold_id}")
        _validate_support_receipt(
            paths[f"base_support_inner_{fold.fold_id}.parquet"],
            inner_base_support[fold.fold_id],
            expected_fit_rows=fold.inner_train.size,
        )
        _validate_support_receipt(
            paths[f"base_support_full_{fold.fold_id}.parquet"],
            full_base_support[fold.fold_id],
            expected_fit_rows=fold.full_train.size,
        )
        _validate_support_receipt(
            paths[f"elastic_support_full_{fold.fold_id}.parquet"],
            elastic_support[fold.fold_id],
            expected_fit_rows=None,
        )
    archive.close()

    return PreparedSelectorData(
        root=root,
        metadata=metadata,
        rows=rows,
        split_membership=splits,
        folds=folds,
        inner_base_matrices=inner_base_matrices,
        full_base_matrices=full_base_matrices,
        elastic_matrices=elastic_matrices,
        inner_base_support=inner_base_support,
        full_base_support=full_base_support,
        elastic_support=elastic_support,
    )


def load_prepared_selector_fold(
    root: Path,
    *,
    expected_build_id: str,
    expected_content_fingerprint: str,
    fold_id: str,
    family: Literal["neural_network", "xgboost", "elastic_net"],
) -> PreparedSelectorFoldData:
    """Authenticate and mmap only one fold's matrices for one native family.

    NN and XGBoost receive the independently fitted inner and full base
    matrices.  Elastic Net receives only the full-training Elastic design.
    Opposing-family matrices are deliberately not opened or hashed here.
    """

    if fold_id not in EXPECTED_FOLDS:
        raise ValueError("Prepared Stage 9 fold ID is not frozen")
    if family not in {"neural_network", "xgboost", "elastic_net"}:
        raise ValueError("Prepared Stage 9 native family is not frozen")
    root = _safe_root(root)
    metadata = _load_authenticated_metadata(
        root,
        expected_build_id=expected_build_id,
        expected_content_fingerprint=expected_content_fingerprint,
    )
    rows, splits, folds = _load_rows_and_folds(root, metadata)
    fold = next(value for value in folds if value.fold_id == fold_id)
    support_path = _authenticated_path(root, "support_masks.npz", metadata)
    archive = _support_archive(support_path)

    inner_base_matrix: np.ndarray | None = None
    full_base_matrix: np.ndarray | None = None
    elastic_matrix: np.ndarray | None = None
    inner_base_support: np.ndarray | None = None
    full_base_support: np.ndarray | None = None
    elastic_support: np.ndarray | None = None
    if family in {"neural_network", "xgboost"}:
        _authenticated_path(root, "base_column_dictionary.parquet", metadata)
        inner_path = _authenticated_path(
            root, f"base_matrix_inner_{fold_id}.npy", metadata
        )
        full_path = _authenticated_path(
            root, f"base_matrix_full_{fold_id}.npy", metadata
        )
        inner_receipt_path = _authenticated_path(
            root, f"base_support_inner_{fold_id}.parquet", metadata
        )
        full_receipt_path = _authenticated_path(
            root, f"base_support_full_{fold_id}.parquet", metadata
        )
        inner_base_matrix = np.load(inner_path, mmap_mode="r", allow_pickle=False)
        full_base_matrix = np.load(full_path, mmap_mode="r", allow_pickle=False)
        for scope, matrix in (
            ("inner", inner_base_matrix),
            ("full", full_base_matrix),
        ):
            if (
                matrix.shape != (EXPECTED_ROWS, BASE_MATRIX_WIDTH)
                or matrix.dtype != np.dtype("float32")
            ):
                archive.close()
                raise RuntimeError(
                    f"Prepared {scope} base matrix differs for {fold_id}"
                )
        inner_base_support = np.asarray(
            archive[f"{fold_id}__base_inner"], dtype=bool
        )
        full_base_support = np.asarray(
            archive[f"{fold_id}__base_full"], dtype=bool
        )
        if (
            inner_base_support.shape != (BASE_MATRIX_WIDTH,)
            or full_base_support.shape != (BASE_MATRIX_WIDTH,)
        ):
            archive.close()
            raise RuntimeError(f"Prepared base support shape differs for {fold_id}")
        _validate_support_receipt(
            inner_receipt_path,
            inner_base_support,
            expected_fit_rows=fold.inner_train.size,
        )
        _validate_support_receipt(
            full_receipt_path,
            full_base_support,
            expected_fit_rows=fold.full_train.size,
        )
    else:
        _authenticated_path(root, "base_column_dictionary.parquet", metadata)
        _authenticated_path(root, "elastic_interaction_dictionary.parquet", metadata)
        elastic_path = _authenticated_path(
            root, f"elastic_matrix_full_{fold_id}.npy", metadata
        )
        elastic_receipt_path = _authenticated_path(
            root, f"elastic_support_full_{fold_id}.parquet", metadata
        )
        elastic_matrix = np.load(elastic_path, mmap_mode="r", allow_pickle=False)
        if (
            elastic_matrix.shape != (EXPECTED_ROWS, ELASTIC_NET_TOTAL_WIDTH)
            or elastic_matrix.dtype != np.dtype("float32")
        ):
            archive.close()
            raise RuntimeError(f"Prepared Elastic matrix differs for {fold_id}")
        elastic_support = np.asarray(
            archive[f"{fold_id}__elastic_full"], dtype=bool
        )
        if elastic_support.shape != (ELASTIC_NET_TOTAL_WIDTH,):
            archive.close()
            raise RuntimeError(f"Prepared Elastic support shape differs for {fold_id}")
        _validate_support_receipt(
            elastic_receipt_path,
            elastic_support,
            expected_fit_rows=None,
        )
    archive.close()
    return PreparedSelectorFoldData(
        root=root,
        metadata=metadata,
        rows=rows,
        split_membership=splits,
        fold=fold,
        family=family,
        inner_base_matrix=inner_base_matrix,
        full_base_matrix=full_base_matrix,
        elastic_matrix=elastic_matrix,
        inner_base_support=inner_base_support,
        full_base_support=full_base_support,
        elastic_support=elastic_support,
    )


__all__ = [
    "PreparedSelectorData",
    "PreparedSelectorFoldData",
    "load_prepared_selector_data",
    "load_prepared_selector_fold",
]
