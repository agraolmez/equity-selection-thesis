"""Concrete, authority-bound selector backend for Stage 11.

The backend deliberately has no caller-supplied data paths.  It receives the
single authenticated execution-input object, reads validation predictors
through its broker, reads development features/labels through the exact Stage
9 catalog, and exposes outcomes only month-by-month after their recorded
availability date.  Native model fitting and prediction run in separate
``spawn`` children claimed by exactly one model family.

This module does not perform family selection, portfolio optimization, or P&L
calculation.  Those remain controlled by :mod:`execution_orchestrator` after
the complete pseudo-live forecast/outcome ledger has been produced.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import stat
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.selectors.contract import canonical_json_bytes as selector_json
from portfolio_bot.selectors.inputs import Stage9Inputs
from portfolio_bot.selectors.native_worker import RUNTIME_VERSIONS_SHA256
from portfolio_bot.selectors.preprocessing import (
    SelectorMatrixSchema,
    SelectorPreprocessingResult,
    build_elastic_net_interaction_registry,
    build_selector_matrix,
)
from portfolio_bot.validation.access import PayloadRole
from portfolio_bot.validation.execution_input_manifest import (
    AuthenticatedStage11ExecutionInputs,
)
from portfolio_bot.validation.execution_orchestrator import (
    FORECAST_COLUMNS,
    OUTCOME_COLUMNS,
    AuthorityBoundStage11WalkForwardBackend,
    MonthlyExecutionStep,
    SelectorFitJob,
    Stage11ExecutionPlan,
)
from portfolio_bot.validation.locked_selection import (
    LOCKED_SHORTLIST,
    LOCKED_STAGE9_SHORTLIST_SHA256,
    VALIDATION_STOCHASTIC_SEEDS,
)
from portfolio_bot.validation.outcome_materialization import (
    DEFAULT_INPUT_NAMES,
    Stage11OutcomePublication,
)
from portfolio_bot.validation.selector_runtime import (
    AggregateMomentState,
    FileBackedArrayRequest,
    ForecastBatchReceipt,
    LoadedSelectorCheckpoint,
    PersistedPreprocessorState,
    ValidationRefitArrays,
    ValidationRefitIdentity,
    build_refit_identity,
    canonical_json_bytes,
    load_selector_checkpoint,
    pack_numeric_arrays,
    persist_array_for_spawn,
    persist_validation_refit_arrays_for_spawn,
    prepare_validation_refit_arrays,
    publish_forecast_batch,
    publish_selector_checkpoint,
    unpack_numeric_arrays,
    validate_complete_forecast_batches,
)
from portfolio_bot.validation.walk_forward import (
    EXPECTED_REBALANCE_CALENDAR_SHA256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PRESENT = "PRESENT"
_MODEL_CODE_COMMON = tuple(
    sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "src/portfolio_bot").rglob("*.py")
    )
)
_MODEL_CODE_FAMILY = {
    "neural_network": "src/portfolio_bot/selectors/neural_model.py",
    "xgboost": "src/portfolio_bot/selectors/xgboost_model.py",
    "elastic_net": "src/portfolio_bot/selectors/elastic_net.py",
}
_KEY_COLUMNS = ("signal_date", "permno")
_DEVELOPMENT_LABEL_COLUMNS = (
    "signal_date",
    "permno",
    "holding_month",
    "entry_date",
    "exit_date",
    "month_label_available_date",
    "residual_return_rank",
    "primary_label_status",
)
_OUTCOME_SOURCE_COLUMNS = (
    "signal_date",
    "holding_month",
    "entry_date",
    "exit_date",
    "month_label_available_date",
    "permno",
    "unranked_residual_return",
    "residual_return_rank",
    "primary_label_status",
)


class EmpiricalSelectorBackendError(RuntimeError):
    """Raised when an empirical selector input, worker, or chronology differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EmpiricalSelectorBackendError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _stable_identity(path: Path) -> tuple[int, ...]:
    value = path.stat(follow_symlinks=False)
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _ordinary_directory(path: Path, *, create: bool) -> Path:
    selected = Path(os.path.abspath(path.expanduser()))
    if create:
        selected.mkdir(parents=True, exist_ok=True)
    _require(
        selected.is_dir() and not selected.is_symlink(), f"unsafe directory: {selected}"
    )
    return selected


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA or bool(pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return value.isoformat()
    if type(value) is dt.date:
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    rows = [
        {
            name: _canonical_value(value)
            for name, value in zip(columns, row, strict=True)
        }
        for row in frame.loc[:, list(columns)].itertuples(index=False, name=None)
    ]
    return hashlib.sha256(selector_json(rows)).hexdigest()


def _normalized_dates(values: pd.Series, *, label: str) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="raise").dt.date
    except (TypeError, ValueError) as exc:
        raise EmpiricalSelectorBackendError(
            f"{label} contains an invalid date"
        ) from exc


def _normalize_feature_frame(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    result = frame.copy()
    _require(set(_KEY_COLUMNS).issubset(result), f"{label} omitted feature keys")
    result["signal_date"] = _normalized_dates(result["signal_date"], label=label)
    raw_permno = pd.to_numeric(result["permno"], errors="coerce").to_numpy(float)
    _require(
        np.isfinite(raw_permno).all()
        and np.all(raw_permno > 0.0)
        and np.all(raw_permno == np.floor(raw_permno)),
        f"{label} has invalid PERMNO values",
    )
    result["permno"] = raw_permno.astype(np.int64)
    _require(
        not result.duplicated(list(_KEY_COLUMNS)).any(), f"{label} keys are duplicated"
    )
    return result.sort_values(list(_KEY_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )


def _category_mapping(
    result: SelectorPreprocessingResult,
) -> tuple[tuple[str, str, str], ...]:
    receipt = result.category_mapping_receipt
    required = ("feature_name", "category_value", "mapped_to_category_value")
    _require(set(required).issubset(receipt), "category mapping receipt differs")
    values = tuple(
        sorted(
            (
                str(row.feature_name),
                str(row.category_value),
                str(row.mapped_to_category_value),
            )
            for row in receipt.loc[:, list(required)].itertuples(index=False)
        )
    )
    _require(len(values) == len(set(values)), "category mapping is duplicated")
    return values


def _aggregate_history_sha256(
    history: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    cutoff: dt.date,
) -> str:
    selected = history.loc[
        history["signal_date"].le(cutoff), ["signal_date", *feature_names]
    ]
    _require(not selected.empty, "aggregate history is empty at refit")
    return _frame_sha256(selected, ("signal_date", *feature_names))


def _aggregate_moments(
    history: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    cutoff: dt.date,
) -> tuple[AggregateMomentState, ...]:
    selected = history.loc[history["signal_date"].le(cutoff)].copy()
    _require(not selected.empty, "aggregate moment history is empty")
    result: list[AggregateMomentState] = []
    for name in sorted(str(value) for value in feature_names):
        numeric = pd.to_numeric(selected[name], errors="coerce").to_numpy(float)
        finite = np.isfinite(numeric)
        values = numeric[finite]
        dates = selected.loc[finite, "signal_date"]
        if values.size:
            mean = float(values.mean(dtype=np.float64))
            m2 = float(np.square(values - mean, dtype=np.float64).sum(dtype=np.float64))
            last = max(dates)
        else:
            mean, m2, last = 0.0, 0.0, min(selected["signal_date"])
        result.append(
            AggregateMomentState(
                feature_name=name,
                finite_count=int(values.size),
                mean=mean,
                m2=max(0.0, m2),
                last_observation_date=last,
            )
        )
    return tuple(result)


def _interaction_registry(
    schema: SelectorMatrixSchema,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    registry = build_elastic_net_interaction_registry(schema).drop(columns="active")
    positions = {name: index for index, name in enumerate(schema.column_names)}
    firm = registry["firm_column"].map(positions)
    aggregate = registry["aggregate_column"].map(positions)
    _require(
        not firm.isna().any() and not aggregate.isna().any(),
        "interaction registry escaped base schema",
    )
    columns = tuple(str(value) for value in registry.columns)
    digest = _frame_sha256(registry, columns)
    return (
        registry,
        firm.to_numpy(np.int64),
        aggregate.to_numpy(np.int64),
        digest,
    )


def _interaction_support(
    base_training: np.ndarray,
    firm: np.ndarray,
    aggregate: np.ndarray,
    base_support: np.ndarray,
    *,
    chunk_rows: int = 16_384,
) -> np.ndarray:
    minimum = np.full(firm.size, np.inf, dtype=np.float64)
    maximum = np.full(firm.size, -np.inf, dtype=np.float64)
    for start in range(0, len(base_training), chunk_rows):
        block = np.asarray(base_training[start : start + chunk_rows], dtype=np.float32)
        values = block[:, firm] * block[:, aggregate]
        minimum = np.minimum(minimum, values.min(axis=0).astype(np.float64))
        maximum = np.maximum(maximum, values.max(axis=0).astype(np.float64))
    active_components = (
        np.asarray(base_support, dtype=bool)[firm]
        & np.asarray(base_support, dtype=bool)[aggregate]
    )
    return active_components & (maximum > minimum)


def _elastic_matrix(
    base: np.ndarray,
    firm: np.ndarray,
    aggregate: np.ndarray,
    *,
    destination: Path | None,
    chunk_rows: int = 8_192,
) -> np.ndarray:
    rows, base_width = base.shape
    shape = (rows, base_width + firm.size)
    if destination is None:
        target: np.ndarray = np.empty(shape, dtype=np.float32)
    else:
        target = np.lib.format.open_memmap(
            destination,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        block = np.asarray(base[start:stop], dtype=np.float32)
        target[start:stop, :base_width] = block
        target[start:stop, base_width:] = block[:, firm] * block[:, aggregate]
    if isinstance(target, np.memmap):
        target.flush()
    _require(np.isfinite(target).all(), "Elastic Net interaction matrix is not finite")
    return target


@dataclass(frozen=True, slots=True)
class PreparedRefit:
    refit_signal_date: dt.date
    base_arrays: ValidationRefitArrays
    elastic_arrays: ValidationRefitArrays
    inner_preprocessor: PersistedPreprocessorState
    full_base_preprocessor: PersistedPreprocessorState
    full_elastic_preprocessor: PersistedPreprocessorState
    score_permnos: np.ndarray
    score_base_frozen: np.ndarray
    score_elastic_frozen: np.ndarray


def _preprocessor_state(
    *,
    scope: str,
    matrix_kind: str,
    refit_signal_date: dt.date,
    fitted_signal_date_max: dt.date,
    column_names: tuple[str, ...],
    active_mask: np.ndarray,
    result: SelectorPreprocessingResult,
    stage9_inputs: Stage9Inputs,
    aggregate_history: pd.DataFrame,
    aggregate_names: Sequence[str],
    interaction_registry_sha256: str | None,
) -> PersistedPreprocessorState:
    active = tuple(
        int(value) for value in np.flatnonzero(np.asarray(active_mask, dtype=bool))
    )
    return PersistedPreprocessorState(
        scope=scope,
        matrix_kind=matrix_kind,
        refit_signal_date=refit_signal_date,
        fitted_signal_date_max=fitted_signal_date_max,
        predictor_availability_date_max=refit_signal_date,
        frozen_column_names=column_names,
        active_column_indices=active,
        feature_registry_sha256=stage9_inputs.tables[
            "feature_dictionary.parquet"
        ].sha256,
        category_vocabulary_sha256=stage9_inputs.tables[
            "category_vocabulary.parquet"
        ].sha256,
        aggregate_history_sha256=_aggregate_history_sha256(
            aggregate_history,
            feature_names=aggregate_names,
            cutoff=refit_signal_date,
        ),
        interaction_registry_sha256=interaction_registry_sha256,
        category_mapping=_category_mapping(result),
        aggregate_moments=_aggregate_moments(
            aggregate_history,
            feature_names=aggregate_names,
            cutoff=refit_signal_date,
        ),
    )


def _model_code_sha256(family: str) -> str:
    names = tuple(dict.fromkeys((*_MODEL_CODE_COMMON, _MODEL_CODE_FAMILY[family])))
    hashes: dict[str, str] = {}
    for name in sorted(names):
        path = PROJECT_ROOT / name
        _require(
            path.is_file() and not path.is_symlink(),
            f"model code dependency is unsafe: {name}",
        )
        hashes[name] = _sha256_file(path)
    return hashlib.sha256(
        canonical_json_bytes({"model_family": family, "code_hashes": hashes})
    ).hexdigest()


def prepare_authenticated_refit(
    *,
    refit_signal_date: dt.date,
    scoring_signal_date: dt.date,
    combined_features: pd.DataFrame,
    aggregate_history: pd.DataFrame,
    training_labels: pd.DataFrame,
    feature_dictionary: pa.Table,
    category_vocabulary: pa.Table,
    stage9_inputs: Stage9Inputs,
    scratch_root: Path,
    permit_projected_global_ordinals: bool = False,
) -> PreparedRefit:
    """Reuse the frozen Stage 9 preprocessing for one refit and score month."""

    _require(refit_signal_date <= scoring_signal_date, "scoring precedes active refit")
    labels = training_labels.copy()
    for name in ("signal_date", "entry_date", "exit_date", "outcome_availability_date"):
        labels[name] = _normalized_dates(labels[name], label=f"training {name}")
    labels["holding_month"] = labels["holding_month"].astype(str)
    labels["primary_label_status"] = labels["primary_label_status"].astype(str)
    targets = pd.to_numeric(labels["residual_return_rank"], errors="coerce").to_numpy(
        float
    )
    valid = (
        labels["primary_label_status"].eq(_PRESENT).to_numpy(bool)
        & np.isfinite(targets)
        & labels["exit_date"].le(refit_signal_date).to_numpy(bool)
        & labels["outcome_availability_date"].le(refit_signal_date).to_numpy(bool)
    )
    labels = labels.loc[valid].copy()
    labels["residual_return_rank"] = targets[valid]
    _require(not labels.empty, "refit has no completed, available labels")
    _require(
        not labels.duplicated(list(_KEY_COLUMNS)).any(),
        "training label keys are duplicated",
    )
    labels = labels.sort_values(
        ["holding_month", "permno"], kind="mergesort"
    ).reset_index(drop=True)

    ordered_months = tuple(sorted(labels["holding_month"].unique()))
    _require(len(ordered_months) >= 13, "refit lacks a 12-month inner block")
    inner_validation_months = frozenset(ordered_months[-12:])
    inner_labels = labels.loc[~labels["holding_month"].isin(inner_validation_months)]
    _require(not inner_labels.empty, "inner refit training set is empty")

    training_keys = pd.MultiIndex.from_frame(labels.loc[:, list(_KEY_COLUMNS)])
    inner_keys = pd.MultiIndex.from_frame(inner_labels.loc[:, list(_KEY_COLUMNS)])
    features = combined_features.loc[
        combined_features["signal_date"].eq(scoring_signal_date)
        | pd.MultiIndex.from_frame(combined_features.loc[:, list(_KEY_COLUMNS)]).isin(
            training_keys
        )
    ].copy()
    features = _normalize_feature_frame(features, label="refit feature slice")
    feature_keys = pd.MultiIndex.from_frame(features.loc[:, list(_KEY_COLUMNS)])
    full_outer = feature_keys.isin(training_keys)
    inner_outer = feature_keys.isin(inner_keys)
    target_valid = full_outer.copy()
    _require(
        int(full_outer.sum()) == len(labels), "training features and labels differ"
    )
    _require(
        int(features["signal_date"].eq(scoring_signal_date).sum()) > 0,
        "scoring month has no predictor rows",
    )

    aggregate = aggregate_history.loc[
        aggregate_history["signal_date"].le(scoring_signal_date)
    ].copy()
    _require(not aggregate.empty, "aggregate history is empty for scoring")
    full_result = build_selector_matrix(
        features,
        feature_dictionary,
        category_vocabulary,
        outer_training_mask=full_outer,
        valid_target_mask=target_valid,
        aggregate_history_table=aggregate,
        enforce_frozen_contract=not permit_projected_global_ordinals,
        permit_projected_global_ordinals=permit_projected_global_ordinals,
    )
    inner_result = build_selector_matrix(
        features,
        feature_dictionary,
        category_vocabulary,
        outer_training_mask=inner_outer,
        valid_target_mask=target_valid,
        aggregate_history_table=aggregate,
        enforce_frozen_contract=not permit_projected_global_ordinals,
        permit_projected_global_ordinals=permit_projected_global_ordinals,
    )
    _require(
        full_result.schema.column_names == inner_result.schema.column_names
        and np.array_equal(
            full_result.row_keys[list(_KEY_COLUMNS)],
            inner_result.row_keys[list(_KEY_COLUMNS)],
        ),
        "inner/full preprocessing row or schema identity differs",
    )
    result_keys = pd.MultiIndex.from_frame(
        full_result.row_keys.loc[:, list(_KEY_COLUMNS)]
    )
    training_positions = result_keys.get_indexer(training_keys)
    _require(
        np.all(training_positions >= 0), "a training key is absent after preprocessing"
    )
    score_mask = (
        full_result.row_keys["signal_date"].eq(scoring_signal_date).to_numpy(bool)
    )
    _require(score_mask.any(), "scoring rows disappeared during preprocessing")

    full_base_training = np.ascontiguousarray(
        full_result.frozen_matrix[training_positions], dtype=np.float32
    )
    inner_base_training = np.ascontiguousarray(
        inner_result.frozen_matrix[training_positions], dtype=np.float32
    )
    score_base = np.ascontiguousarray(
        full_result.frozen_matrix[score_mask], dtype=np.float32
    )
    score_permnos = full_result.row_keys.loc[score_mask, "permno"].to_numpy(np.int64)
    order = np.argsort(score_permnos, kind="mergesort")
    score_permnos = np.ascontiguousarray(score_permnos[order], dtype=np.int64)
    score_base = np.ascontiguousarray(score_base[order], dtype=np.float32)
    _require(
        np.all(score_permnos > 0) and np.all(np.diff(score_permnos) > 0),
        "scoring PERMNO domain is not sorted unique",
    )

    registry_frame = feature_dictionary.to_pandas()
    aggregate_names = tuple(
        registry_frame.loc[
            registry_frame["kind"].eq("aggregate_state"), "feature_name"
        ].astype(str)
    )
    fitted_signal_date_max = max(labels["signal_date"])
    full_base_state = _preprocessor_state(
        scope="full",
        matrix_kind="base",
        refit_signal_date=refit_signal_date,
        fitted_signal_date_max=fitted_signal_date_max,
        column_names=full_result.schema.column_names,
        active_mask=full_result.support_mask,
        result=full_result,
        stage9_inputs=stage9_inputs,
        aggregate_history=aggregate,
        aggregate_names=aggregate_names,
        interaction_registry_sha256=None,
    )
    inner_state = _preprocessor_state(
        scope="inner",
        matrix_kind="base",
        refit_signal_date=refit_signal_date,
        fitted_signal_date_max=fitted_signal_date_max,
        column_names=inner_result.schema.column_names,
        active_mask=inner_result.support_mask,
        result=inner_result,
        stage9_inputs=stage9_inputs,
        aggregate_history=aggregate,
        aggregate_names=aggregate_names,
        interaction_registry_sha256=None,
    )

    interactions, firm, aggregate_positions, interaction_sha = _interaction_registry(
        full_result.schema
    )
    interaction_active = _interaction_support(
        full_base_training,
        firm,
        aggregate_positions,
        full_result.support_mask,
    )
    elastic_active = np.concatenate(
        (np.asarray(full_result.support_mask, dtype=bool), interaction_active)
    )
    elastic_columns = (
        *full_result.schema.column_names,
        *tuple(interactions["interaction_column"].astype(str)),
    )
    full_elastic_state = _preprocessor_state(
        scope="full",
        matrix_kind="elastic",
        refit_signal_date=refit_signal_date,
        fitted_signal_date_max=fitted_signal_date_max,
        column_names=tuple(elastic_columns),
        active_mask=elastic_active,
        result=full_result,
        stage9_inputs=stage9_inputs,
        aggregate_history=aggregate,
        aggregate_names=aggregate_names,
        interaction_registry_sha256=interaction_sha,
    )
    root = _ordinary_directory(scratch_root, create=True)
    elastic_path = root / "full_elastic_training.npy"
    _require(
        not elastic_path.exists() and not elastic_path.is_symlink(),
        "stale Elastic matrix exists",
    )
    elastic_training = _elastic_matrix(
        full_base_training,
        firm,
        aggregate_positions,
        destination=elastic_path,
    )
    score_elastic = _elastic_matrix(
        score_base,
        firm,
        aggregate_positions,
        destination=None,
    )
    row_key_frame = labels.loc[
        :,
        [
            "signal_date",
            "permno",
            "holding_month",
            "exit_date",
            "outcome_availability_date",
        ],
    ]
    row_keys_sha256 = _frame_sha256(row_key_frame, tuple(row_key_frame.columns))
    common = {
        "refit_signal_date": refit_signal_date,
        "targets": labels["residual_return_rank"].to_numpy(np.float32),
        "holding_months": tuple(labels["holding_month"]),
        "exit_dates": tuple(labels["exit_date"]),
        "outcome_availability_dates": tuple(labels["outcome_availability_date"]),
        "row_keys_sha256": row_keys_sha256,
        "inner_preprocessor": inner_state,
    }
    base_arrays = prepare_validation_refit_arrays(
        full_base_matrix=full_base_training,
        inner_base_matrix=inner_base_training,
        full_preprocessor=full_base_state,
        **common,
    )
    elastic_arrays = prepare_validation_refit_arrays(
        full_base_matrix=elastic_training,
        inner_base_matrix=inner_base_training,
        full_preprocessor=full_elastic_state,
        **common,
    )
    return PreparedRefit(
        refit_signal_date=refit_signal_date,
        base_arrays=base_arrays,
        elastic_arrays=elastic_arrays,
        inner_preprocessor=inner_state,
        full_base_preprocessor=full_base_state,
        full_elastic_preprocessor=full_elastic_state,
        score_permnos=score_permnos,
        score_base_frozen=score_base,
        score_elastic_frozen=score_elastic,
    )


def prepare_authenticated_score(
    *,
    refit_signal_date: dt.date,
    scoring_signal_date: dt.date,
    combined_features: pd.DataFrame,
    aggregate_history: pd.DataFrame,
    training_labels: pd.DataFrame,
    feature_dictionary: pa.Table,
    category_vocabulary: pa.Table,
    stage9_inputs: Stage9Inputs,
    full_base_preprocessor: PersistedPreprocessorState,
    full_elastic_preprocessor: PersistedPreprocessorState,
    permit_projected_global_ordinals: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a later month with the exact state fitted at its active refit."""

    labels = training_labels.copy()
    for name in ("signal_date", "exit_date", "outcome_availability_date"):
        labels[name] = _normalized_dates(labels[name], label=f"score training {name}")
    targets = pd.to_numeric(labels["residual_return_rank"], errors="coerce").to_numpy(
        float
    )
    valid = (
        labels["primary_label_status"].astype(str).eq(_PRESENT).to_numpy(bool)
        & np.isfinite(targets)
        & labels["exit_date"].le(refit_signal_date).to_numpy(bool)
        & labels["outcome_availability_date"].le(refit_signal_date).to_numpy(bool)
    )
    labels = labels.loc[valid].copy()
    _require(not labels.empty, "active refit training labels are empty")
    training_keys = pd.MultiIndex.from_frame(labels.loc[:, list(_KEY_COLUMNS)])
    all_keys = pd.MultiIndex.from_frame(combined_features.loc[:, list(_KEY_COLUMNS)])
    features = combined_features.loc[
        combined_features["signal_date"].eq(scoring_signal_date)
        | all_keys.isin(training_keys)
    ].copy()
    features = _normalize_feature_frame(features, label="scoring feature slice")
    feature_keys = pd.MultiIndex.from_frame(features.loc[:, list(_KEY_COLUMNS)])
    outer = feature_keys.isin(training_keys)
    _require(int(outer.sum()) == len(labels), "active scoring training domain differs")
    aggregate = aggregate_history.loc[
        aggregate_history["signal_date"].le(scoring_signal_date)
    ].copy()
    result = build_selector_matrix(
        features,
        feature_dictionary,
        category_vocabulary,
        outer_training_mask=outer,
        valid_target_mask=outer,
        aggregate_history_table=aggregate,
        enforce_frozen_contract=not permit_projected_global_ordinals,
        permit_projected_global_ordinals=permit_projected_global_ordinals,
    )
    registry_frame = feature_dictionary.to_pandas()
    aggregate_names = tuple(
        registry_frame.loc[
            registry_frame["kind"].eq("aggregate_state"), "feature_name"
        ].astype(str)
    )
    fitted_signal_date_max = max(labels["signal_date"])
    candidate_base = _preprocessor_state(
        scope="full",
        matrix_kind="base",
        refit_signal_date=refit_signal_date,
        fitted_signal_date_max=fitted_signal_date_max,
        column_names=result.schema.column_names,
        active_mask=result.support_mask,
        result=result,
        stage9_inputs=stage9_inputs,
        aggregate_history=aggregate,
        aggregate_names=aggregate_names,
        interaction_registry_sha256=None,
    )
    _require(
        candidate_base.sha256 == full_base_preprocessor.sha256,
        "later scoring rebuilt a different full base preprocessor",
    )
    interactions, firm, aggregate_positions, interaction_sha = _interaction_registry(
        result.schema
    )
    elastic_mask = np.zeros(
        len(full_elastic_preprocessor.frozen_column_names), dtype=bool
    )
    elastic_mask[
        np.asarray(full_elastic_preprocessor.active_column_indices, dtype=np.int64)
    ] = True
    candidate_elastic = _preprocessor_state(
        scope="full",
        matrix_kind="elastic",
        refit_signal_date=refit_signal_date,
        fitted_signal_date_max=fitted_signal_date_max,
        column_names=(
            *result.schema.column_names,
            *tuple(interactions["interaction_column"].astype(str)),
        ),
        active_mask=elastic_mask,
        result=result,
        stage9_inputs=stage9_inputs,
        aggregate_history=aggregate,
        aggregate_names=aggregate_names,
        interaction_registry_sha256=interaction_sha,
    )
    _require(
        candidate_elastic.sha256 == full_elastic_preprocessor.sha256,
        "later scoring rebuilt a different Elastic Net preprocessor",
    )
    score_mask = result.row_keys["signal_date"].eq(scoring_signal_date).to_numpy(bool)
    _require(score_mask.any(), "scoring rows disappeared during transformation")
    permnos = result.row_keys.loc[score_mask, "permno"].to_numpy(np.int64)
    base = np.ascontiguousarray(result.frozen_matrix[score_mask], dtype=np.float32)
    order = np.argsort(permnos, kind="mergesort")
    permnos = np.ascontiguousarray(permnos[order], dtype=np.int64)
    base = np.ascontiguousarray(base[order], dtype=np.float32)
    _require(np.all(np.diff(permnos) > 0), "scoring PERMNO domain is duplicated")
    elastic = _elastic_matrix(
        base,
        firm,
        aggregate_positions,
        destination=None,
    )
    return permnos, base, elastic


def _fit_family_worker(
    family: str,
    identities: tuple[ValidationRefitIdentity, ...],
    request: object,
    full_preprocessor: PersistedPreprocessorState,
    inner_preprocessor: PersistedPreprocessorState | None,
    checkpoint_root: str,
    elastic_scratch_path: str | None,
    result_queue: Any,
) -> None:
    """Fit and publish one complete family inside a spawn child."""

    try:
        from portfolio_bot.validation.selector_native_runtime import (
            fit_locked_elastic_net_group_from_paths,
            fit_locked_selector_model_from_paths,
        )

        if family == "elastic_net":
            artifacts = fit_locked_elastic_net_group_from_paths(
                identities=identities,
                request=request,
                elastic_scratch_path=(
                    None if elastic_scratch_path is None else Path(elastic_scratch_path)
                ),
            )
        else:
            artifacts = tuple(
                fit_locked_selector_model_from_paths(identity=identity, request=request)
                for identity in identities
            )
        records: list[dict[str, object]] = []
        for identity, artifact in zip(identities, artifacts, strict=True):
            checkpoint = publish_selector_checkpoint(
                checkpoint_root=Path(checkpoint_root),
                identity=identity,
                artifact=artifact,
                full_preprocessor=full_preprocessor,
                inner_preprocessor=inner_preprocessor,
            )
            records.append(
                {
                    "checkpoint_path": str(checkpoint.root),
                    "manifest_sha256": checkpoint.manifest_sha256,
                    "identity_sha256": identity.sha256,
                }
            )
        result_queue.put({"ok": True, "records": records})
    except BaseException:
        result_queue.put({"ok": False, "traceback": traceback.format_exc()})


def _score_family_worker(
    family: str,
    checkpoint_paths: tuple[str, ...],
    scoring_request: FileBackedArrayRequest,
    signal_date: dt.date,
    entry_date: dt.date,
    exit_date: dt.date,
    permnos: tuple[int, ...],
    forecast_root: str,
    result_queue: Any,
) -> None:
    """Score and publish a complete family inside one spawn child."""

    try:
        from portfolio_bot.validation.selector_native_runtime import (
            predict_locked_selector_model_from_path,
        )

        receipts: list[ForecastBatchReceipt] = []
        for path in checkpoint_paths:
            checkpoint = load_selector_checkpoint(Path(path))
            _require(
                str(checkpoint.identity["family"]) == family,
                "checkpoint family differs",
            )
            predictions = predict_locked_selector_model_from_path(
                checkpoint=checkpoint,
                request=scoring_request,
            )
            receipts.append(
                publish_forecast_batch(
                    forecast_root=Path(forecast_root),
                    checkpoint=checkpoint,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    predictor_availability_date_max=signal_date,
                    batch_index=0,
                    batch_count=1,
                    permnos=permnos,
                    raw_forecasts=predictions,
                )
            )
        result_queue.put({"ok": True, "receipts": receipts})
    except BaseException:
        result_queue.put({"ok": False, "traceback": traceback.format_exc()})


def _run_spawn_worker(
    *,
    target: Callable[..., None],
    args: tuple[object, ...],
    name: str,
    liveness_gate: Callable[[], None],
) -> Mapping[str, Any]:
    return _run_spawn_workers(
        target=target,
        args_by_worker=(args,),
        names=(name,),
        liveness_gate=liveness_gate,
    )[0]


def _run_spawn_workers(
    *,
    target: Callable[..., None],
    args_by_worker: tuple[tuple[object, ...], ...],
    names: tuple[str, ...],
    liveness_gate: Callable[[], None],
) -> tuple[Mapping[str, Any], ...]:
    """Run a small ordered batch of isolated workers with one live gate."""

    _require(
        bool(args_by_worker)
        and len(args_by_worker) == len(names)
        and len(names) == len(set(names))
        and len(names) <= len(LOCKED_SHORTLIST),
        "native worker batch identity differs",
    )
    context = mp.get_context("spawn")
    queues: list[Any] = []
    processes: list[Any] = []
    messages: list[Mapping[str, Any] | None] = [None] * len(names)
    try:
        for worker_args, worker_name in zip(args_by_worker, names, strict=True):
            queue = context.Queue()
            process = context.Process(
                target=target,
                args=(*worker_args, queue),
                name=worker_name,
            )
            queues.append(queue)
            processes.append(process)
            process.start()
        while any(message is None for message in messages):
            for index, (process, queue, worker_name) in enumerate(
                zip(processes, queues, names, strict=True)
            ):
                if messages[index] is not None:
                    continue
                try:
                    # Drain before joining: a large receipt can fill the
                    # Queue feeder pipe and keep the child alive until the
                    # parent consumes it.
                    message = queue.get(timeout=0.05)
                except Empty:
                    if process.is_alive():
                        continue
                    process.join()
                    try:
                        message = queue.get(timeout=10.0)
                    except Empty as exc:
                        raise EmpiricalSelectorBackendError(
                            f"native worker {worker_name} exited without a receipt "
                            f"(exit={process.exitcode})"
                        ) from exc
                process.join(timeout=10.0)
                _require(
                    not process.is_alive(),
                    f"native worker {worker_name} did not exit after receipt drain",
                )
                _require(
                    isinstance(message, Mapping),
                    f"native worker {worker_name} receipt is invalid",
                )
                _require(
                    process.exitcode == 0 and message.get("ok") is True,
                    f"native worker {worker_name} failed: "
                    f"{message.get('traceback', 'no traceback')}",
                )
                messages[index] = MappingProxyType(dict(message))
            liveness_gate()
        _require(
            all(message is not None for message in messages), "worker result missing"
        )
        return tuple(message for message in messages if message is not None)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10.0)
            if process.is_alive():
                process.kill()
                process.join()
        for queue in queues:
            queue.close()


def _fit_identity_groups(
    family: str,
    identities: tuple[ValidationRefitIdentity, ...],
) -> tuple[tuple[ValidationRefitIdentity, ...], ...]:
    """Keep one deterministic worker per locked stochastic configuration."""

    _require(family in LOCKED_SHORTLIST and identities, "fit family grid differs")
    if family == "elastic_net":
        return (identities,)
    groups = tuple(
        tuple(identity for identity in identities if identity.config_id == config_id)
        for config_id in LOCKED_SHORTLIST[family]
    )
    _require(
        all(
            tuple(identity.seed for identity in group) == VALIDATION_STOCHASTIC_SEEDS
            for group in groups
        )
        and tuple(identity for group in groups for identity in group) == identities,
        f"{family} parallel fit grouping differs from the locked grid",
    )
    return groups


@dataclass(frozen=True, slots=True)
class _ActiveRefit:
    refit_signal_date: dt.date
    training_labels: pd.DataFrame
    completed_outcomes_sha256: str
    inner_preprocessor: PersistedPreprocessorState
    full_base_preprocessor: PersistedPreprocessorState
    full_elastic_preprocessor: PersistedPreprocessorState
    initial_score_signal_date: dt.date
    initial_score_permnos: np.ndarray
    initial_score_base: np.ndarray
    initial_score_elastic: np.ndarray


class EmpiricalStage11SelectorBackend(AuthorityBoundStage11WalkForwardBackend):
    """Production backend admitted by ``run_empirical_locked_walk_forward``."""

    def __init__(
        self,
        authority: object,
        execution_inputs: AuthenticatedStage11ExecutionInputs,
        outcome_publication: Stage11OutcomePublication,
        plan: Stage11ExecutionPlan,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        _require(
            type(execution_inputs) is AuthenticatedStage11ExecutionInputs,
            "selector backend requires authenticated execution inputs",
        )
        execution_inputs.gate()
        _require(type(plan) is Stage11ExecutionPlan, "selector backend plan differs")
        super().__init__(authority, execution_inputs.broker)
        self._execution_inputs = execution_inputs
        self._plan = plan
        self._progress = progress if progress is not None else (lambda _message: None)
        self._stage9_inputs = execution_inputs.development_oof.stage9_inputs
        self._checkpoint_root = _ordinary_directory(
            execution_inputs.output_roots["selector_checkpoints"], create=True
        )
        self._forecast_root = _ordinary_directory(
            execution_inputs.output_roots["raw_forecasts"], create=True
        )
        self._work_root = _ordinary_directory(
            self._checkpoint_root / "_runtime_work", create=True
        )
        self._feature_dictionary = self._stage9_inputs.read_feature_dictionary()
        self._category_vocabulary = self._stage9_inputs.read_category_vocabulary()
        development_features = _normalize_feature_frame(
            self._stage9_inputs.read_development_feature_panel().to_pandas(),
            label="authenticated development predictors",
        )
        development_labels = (
            self._stage9_inputs.read_development_labels()
            .select(list(_DEVELOPMENT_LABEL_COLUMNS))
            .to_pandas()
        )
        self._development_labels = self._normalize_development_labels(
            development_labels
        )
        development_feature_keys = pd.MultiIndex.from_frame(
            development_features.loc[:, list(_KEY_COLUMNS)]
        )
        development_label_keys = pd.MultiIndex.from_frame(
            self._development_labels.loc[:, list(_KEY_COLUMNS)]
        )
        _require(
            development_feature_keys.equals(development_label_keys),
            "authenticated development feature/label keys differ",
        )
        development_predictor_columns = tuple(development_features.columns)
        validation_features = self._read_validation_predictors(
            columns=development_predictor_columns
        )
        _require(
            tuple(validation_features.columns) == tuple(development_features.columns),
            "development/validation predictor schema or order differs",
        )
        self._combined_features = _normalize_feature_frame(
            pd.concat((development_features, validation_features), ignore_index=True),
            label="combined authenticated predictors",
        )
        self._aggregate_history = self._build_aggregate_history(validation_features)
        self._outcome_path, self._outcome_identity = self._authenticate_outcome_store(
            outcome_publication
        )
        self._active_refit: _ActiveRefit | None = None
        self._active_checkpoints: dict[
            tuple[str, str, int | None], LoadedSelectorCheckpoint
        ] = {}
        self._checkpoint_digests: dict[tuple[str, str, int | None], str] = {}
        self._model_code_sha256 = {
            family: _model_code_sha256(family) for family in LOCKED_SHORTLIST
        }
        self.gate()

    def _normalize_development_labels(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.rename(
            columns={"month_label_available_date": "outcome_availability_date"}
        ).copy()
        for name in (
            "signal_date",
            "entry_date",
            "exit_date",
            "outcome_availability_date",
        ):
            result[name] = _normalized_dates(result[name], label=f"development {name}")
        result["permno"] = pd.to_numeric(result["permno"], errors="raise").astype(
            np.int64
        )
        result["holding_month"] = result["holding_month"].astype(str)
        result["primary_label_status"] = result["primary_label_status"].astype(str)
        _require(
            not result.duplicated(list(_KEY_COLUMNS)).any(),
            "development label keys are duplicated",
        )
        return result.sort_values(list(_KEY_COLUMNS), kind="mergesort").reset_index(
            drop=True
        )

    def _read_validation_predictors(
        self,
        *,
        columns: tuple[str, ...],
    ) -> pd.DataFrame:
        self.gate()
        name = DEFAULT_INPUT_NAMES.predictor
        record = self._execution_inputs.registrations[name]
        _require(
            type(columns) is tuple
            and columns
            and len(columns) == len(set(columns))
            and "long_eligible" in columns
            and set(columns).issubset(record.allowed_columns),
            "development predictor projection escaped validation allowlist",
        )
        request = self._stage11_access_broker.authorize_table_request(
            name,
            role=PayloadRole.VALIDATION_PREDICTOR,
            columns=columns,
            minimum_date=record.minimum_date,
            maximum_date=record.maximum_date,
        )
        table = self._stage11_access_broker.read_guarded_parquet(request)
        self.gate()
        table = table.filter(pc.fill_null(table["long_eligible"], False))
        _require(
            table.num_rows > 0,
            "authenticated long-eligible validation predictor is empty",
        )
        return _normalize_feature_frame(
            table.to_pandas(), label="authenticated validation predictors"
        )

    def _build_aggregate_history(
        self,
        validation_features: pd.DataFrame,
    ) -> pd.DataFrame:
        development = self._stage9_inputs.read_aggregate_signal_history().to_pandas()
        development["signal_date"] = _normalized_dates(
            development["signal_date"], label="development aggregate history"
        )
        dictionary = self._feature_dictionary.to_pandas()
        names = tuple(
            dictionary.loc[
                dictionary["kind"].eq("aggregate_state"), "feature_name"
            ].astype(str)
        )
        grouped = validation_features.groupby("signal_date", sort=True, observed=True)
        for name in names:
            _require(
                bool((grouped[name].nunique(dropna=False) <= 1).all()),
                f"validation aggregate predictor varies within month: {name}",
            )
        validation = grouped[list(names)].first().reset_index()
        combined = (
            pd.concat(
                (development.loc[:, ["signal_date", *names]], validation),
                ignore_index=True,
            )
            .sort_values("signal_date", kind="mergesort")
            .reset_index(drop=True)
        )
        _require(
            not combined["signal_date"].duplicated().any(),
            "development/validation aggregate histories overlap",
        )
        return combined

    def _authenticate_outcome_store(
        self,
        publication: Stage11OutcomePublication,
    ) -> tuple[Path, tuple[int, ...]]:
        self.gate()
        _require(
            type(publication) is Stage11OutcomePublication
            and publication.output_root
            == self._execution_inputs.output_roots["outcome_tables"],
            "outcome publication is not bound to execution inputs",
        )
        tables = publication.receipt.get("tables")
        _require(
            isinstance(tables, Mapping), "outcome publication table receipts are absent"
        )
        receipt = tables.get("validation_labels.parquet")
        _require(isinstance(receipt, Mapping), "validation label receipt is absent")
        path = publication.output_root / "validation_labels.parquet"
        before = _stable_identity(path)
        metadata = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and metadata.st_nlink == 1
            and metadata.st_size == receipt.get("size_bytes"),
            "validation label store is unsafe or changed",
        )
        _require(
            _sha256_file(path) == receipt.get("sha256"),
            "validation label store digest differs",
        )
        parquet = pq.ParquetFile(path)
        _require(
            parquet.metadata.num_rows == receipt.get("rows")
            and set(_OUTCOME_SOURCE_COLUMNS).issubset(parquet.schema_arrow.names),
            "validation label store footer differs",
        )
        _require(
            _stable_identity(path) == before,
            "validation label store changed while authenticated",
        )
        self.gate()
        return path, before

    def _completed_outcome_sha256(self, completed: pd.DataFrame) -> str:
        if completed.empty:
            return hashlib.sha256(canonical_json_bytes([])).hexdigest()
        return _frame_sha256(
            completed.sort_values(list(_KEY_COLUMNS), kind="mergesort"),
            OUTCOME_COLUMNS,
        )

    def _training_labels(self, completed: pd.DataFrame) -> pd.DataFrame:
        if completed.empty:
            validation = pd.DataFrame(columns=self._development_labels.columns)
        else:
            _require(
                set(completed.columns) == set(OUTCOME_COLUMNS),
                "completed validation outcome columns differ",
            )
            validation = completed.loc[
                :,
                [
                    "signal_date",
                    "permno",
                    "holding_month",
                    "entry_date",
                    "exit_date",
                    "outcome_availability_date",
                    "residual_return_rank",
                    "primary_label_status",
                ],
            ].copy()
        combined = pd.concat((self._development_labels, validation), ignore_index=True)
        _require(
            not combined.duplicated(list(_KEY_COLUMNS)).any(),
            "development and validation training labels overlap",
        )
        return combined.sort_values(list(_KEY_COLUMNS), kind="mergesort").reset_index(
            drop=True
        )

    def _identities(
        self,
        prepared: PreparedRefit,
    ) -> Mapping[str, tuple[ValidationRefitIdentity, ...]]:
        result: dict[str, tuple[ValidationRefitIdentity, ...]] = {}
        jobs = tuple(
            job
            for job in self._plan.fit_jobs
            if job.refit_signal_date == prepared.refit_signal_date
        )
        for family in LOCKED_SHORTLIST:
            selected = tuple(job for job in jobs if job.model_family == family)
            arrays = (
                prepared.elastic_arrays
                if family == "elastic_net"
                else prepared.base_arrays
            )
            full = (
                prepared.full_elastic_preprocessor
                if family == "elastic_net"
                else prepared.full_base_preprocessor
            )
            result[family] = tuple(
                build_refit_identity(
                    family=job.model_family,
                    config_id=job.config_id,
                    seed=job.seed,
                    refit_signal_date=job.refit_signal_date,
                    arrays=arrays,
                    full_preprocessor=full,
                    inner_preprocessor=(
                        None if family == "elastic_net" else prepared.inner_preprocessor
                    ),
                    validation_calendar_sha256=EXPECTED_REBALANCE_CALENDAR_SHA256,
                    model_code_sha256=self._model_code_sha256[family],
                    runtime_versions_sha256=RUNTIME_VERSIONS_SHA256,
                    stage9_shortlist_sha256=LOCKED_STAGE9_SHORTLIST_SHA256,
                )
                for job in selected
            )
        return MappingProxyType(result)

    def _fit_refit(
        self,
        job: SelectorFitJob,
        completed_outcomes: pd.DataFrame,
    ) -> None:
        self.gate()
        training_labels = self._training_labels(completed_outcomes)
        completed_sha = self._completed_outcome_sha256(completed_outcomes)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f"refit-{job.refit_index:02d}-",
                dir=self._work_root,
            )
        )
        try:
            self._progress(
                f"Stage 11 refit {job.refit_index + 1}/20: preprocessing authenticated history"
            )
            prepared = prepare_authenticated_refit(
                refit_signal_date=job.refit_signal_date,
                scoring_signal_date=job.refit_signal_date,
                combined_features=self._combined_features,
                aggregate_history=self._aggregate_history,
                training_labels=training_labels,
                feature_dictionary=self._feature_dictionary,
                category_vocabulary=self._category_vocabulary,
                stage9_inputs=self._stage9_inputs,
                scratch_root=temporary / "preprocessing",
            )
            base_request = persist_validation_refit_arrays_for_spawn(
                prepared.base_arrays,
                temporary / "base_refit_arrays",
            )
            elastic_request = persist_validation_refit_arrays_for_spawn(
                prepared.elastic_arrays,
                temporary / "elastic_refit_arrays",
            )
            identities = self._identities(prepared)
            loaded: dict[tuple[str, str, int | None], LoadedSelectorCheckpoint] = {}
            digests: dict[tuple[str, str, int | None], str] = {}
            for family in ("neural_network", "xgboost", "elastic_net"):
                self._progress(
                    f"Stage 11 refit {job.refit_index + 1}/20: fitting {family} family"
                )
                full = (
                    prepared.full_elastic_preprocessor
                    if family == "elastic_net"
                    else prepared.full_base_preprocessor
                )
                inner = None if family == "elastic_net" else prepared.inner_preprocessor
                identity_groups = _fit_identity_groups(family, identities[family])
                messages = _run_spawn_workers(
                    target=_fit_family_worker,
                    args_by_worker=tuple(
                        (
                            family,
                            group,
                            (
                                elastic_request
                                if family == "elastic_net"
                                else base_request
                            ),
                            full,
                            inner,
                            str(self._checkpoint_root),
                            (
                                str(temporary / "elastic.weighted-f64.memmap")
                                if family == "elastic_net"
                                else None
                            ),
                        )
                        for group in identity_groups
                    ),
                    names=tuple(
                        f"stage11-refit-{job.refit_index:02d}-{family}-{index:02d}"
                        for index in range(len(identity_groups))
                    ),
                    liveness_gate=self.gate,
                )
                records: list[object] = []
                for group, message in zip(identity_groups, messages, strict=True):
                    group_records = message.get("records")
                    _require(
                        isinstance(group_records, list)
                        and len(group_records) == len(group),
                        f"{family} worker checkpoint count differs",
                    )
                    records.extend(group_records)
                _require(
                    len(records) == len(identities[family]),
                    f"{family} worker checkpoint count differs",
                )
                for identity, record in zip(identities[family], records, strict=True):
                    _require(
                        isinstance(record, Mapping), "checkpoint worker receipt differs"
                    )
                    checkpoint = load_selector_checkpoint(
                        Path(str(record["checkpoint_path"])),
                        expected_identity=identity,
                    )
                    _require(
                        checkpoint.manifest_sha256 == record.get("manifest_sha256")
                        and identity.sha256 == record.get("identity_sha256"),
                        "checkpoint worker identity or digest differs",
                    )
                    key = (identity.family, identity.config_id, identity.seed)
                    loaded[key] = checkpoint
                    digests[key] = checkpoint.manifest_sha256
            expected = tuple(
                (value.model_family, value.config_id, value.seed)
                for value in self._plan.fit_jobs
                if value.refit_signal_date == job.refit_signal_date
            )
            _require(tuple(loaded) == expected, "refit checkpoint grid/order differs")
            self._active_checkpoints = loaded
            self._checkpoint_digests = digests
            self._active_refit = _ActiveRefit(
                refit_signal_date=job.refit_signal_date,
                training_labels=training_labels.copy(deep=True),
                completed_outcomes_sha256=completed_sha,
                inner_preprocessor=prepared.inner_preprocessor,
                full_base_preprocessor=prepared.full_base_preprocessor,
                full_elastic_preprocessor=prepared.full_elastic_preprocessor,
                initial_score_signal_date=job.refit_signal_date,
                initial_score_permnos=prepared.score_permnos.copy(),
                initial_score_base=prepared.score_base_frozen.copy(),
                initial_score_elastic=prepared.score_elastic_frozen.copy(),
            )
        finally:
            shutil.rmtree(temporary, ignore_errors=False)
        self.gate()

    def fit_selector(
        self,
        job: SelectorFitJob,
        completed_outcomes: pd.DataFrame,
    ) -> str:
        self.gate()
        _require(
            isinstance(job, SelectorFitJob)
            and isinstance(completed_outcomes, pd.DataFrame),
            "selector fit inputs differ",
        )
        if (
            self._active_refit is None
            or self._active_refit.refit_signal_date != job.refit_signal_date
        ):
            self._fit_refit(job, completed_outcomes)
        assert self._active_refit is not None
        _require(
            self._active_refit.completed_outcomes_sha256
            == self._completed_outcome_sha256(completed_outcomes),
            "completed outcome set changed within one refit",
        )
        key = (job.model_family, job.config_id, job.seed)
        _require(key in self._checkpoint_digests, "requested fit checkpoint is absent")
        self.gate()
        return self._checkpoint_digests[key]

    def _score_arrays(
        self,
        step: MonthlyExecutionStep,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        active = self._active_refit
        _require(active is not None, "monthly scoring has no active refit")
        if step.calendar_row.signal_date == active.initial_score_signal_date:
            return (
                active.initial_score_permnos.copy(),
                active.initial_score_base.copy(),
                active.initial_score_elastic.copy(),
            )
        return prepare_authenticated_score(
            refit_signal_date=active.refit_signal_date,
            scoring_signal_date=step.calendar_row.signal_date,
            combined_features=self._combined_features,
            aggregate_history=self._aggregate_history,
            training_labels=active.training_labels,
            feature_dictionary=self._feature_dictionary,
            category_vocabulary=self._category_vocabulary,
            stage9_inputs=self._stage9_inputs,
            full_base_preprocessor=active.full_base_preprocessor,
            full_elastic_preprocessor=active.full_elastic_preprocessor,
        )

    def score_month(
        self,
        step: MonthlyExecutionStep,
        checkpoints: Mapping[tuple[str, str, int | None], str],
    ) -> pd.DataFrame:
        self.gate()
        _require(
            isinstance(step, MonthlyExecutionStep)
            and self._active_refit is not None
            and step.active_refit_signal_date == self._active_refit.refit_signal_date,
            "monthly scoring active refit differs",
        )
        _require(
            dict(checkpoints) == self._checkpoint_digests,
            "monthly scoring checkpoint grid/digests differ",
        )
        permnos, base, elastic = self._score_arrays(step)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f"score-{step.month_index:02d}-", dir=self._work_root
            )
        )
        try:
            base_request = persist_array_for_spawn(
                base,
                temporary / "base_scoring",
                name="full_frozen_matrix",
            )
            elastic_request = persist_array_for_spawn(
                elastic,
                temporary / "elastic_scoring",
                name="full_frozen_matrix",
            )
            rows: list[pd.DataFrame] = []
            for family in ("neural_network", "xgboost", "elastic_net"):
                keys = tuple(
                    key for key in self._active_checkpoints if key[0] == family
                )
                paths = tuple(str(self._active_checkpoints[key].root) for key in keys)
                message = _run_spawn_worker(
                    target=_score_family_worker,
                    args=(
                        family,
                        paths,
                        elastic_request if family == "elastic_net" else base_request,
                        step.calendar_row.signal_date,
                        step.calendar_row.entry_date,
                        step.calendar_row.exit_date,
                        tuple(int(value) for value in permnos),
                        str(self._forecast_root),
                    ),
                    name=f"stage11-score-{step.month_index:02d}-{family}",
                    liveness_gate=self.gate,
                )
                receipts = message.get("receipts")
                _require(
                    isinstance(receipts, list) and len(receipts) == len(keys),
                    f"{family} forecast receipt count differs",
                )
                for key, receipt in zip(keys, receipts, strict=True):
                    _require(
                        type(receipt) is ForecastBatchReceipt,
                        "forecast receipt type differs",
                    )
                    completed = validate_complete_forecast_batches((receipt,))
                    _require(
                        receipt.checkpoint_manifest_sha256 == checkpoints[key]
                        and completed.rows == len(permnos),
                        "forecast receipt checkpoint/domain differs",
                    )
                    arrays = unpack_numeric_arrays(
                        Path(receipt.payload_path).read_bytes()
                    )
                    observed_permnos = np.asarray(arrays["permnos"], dtype=np.int64)
                    forecasts = np.asarray(arrays["raw_forecasts"], dtype=np.float32)
                    _require(
                        np.array_equal(observed_permnos, permnos)
                        and forecasts.shape == permnos.shape,
                        "published forecast payload domain differs",
                    )
                    rows.append(
                        pd.DataFrame(
                            {
                                "model_family": key[0],
                                "config_id": key[1],
                                "seed": key[2],
                                "signal_date": step.calendar_row.signal_date,
                                "holding_month": step.calendar_row.holding_month,
                                "permno": permnos,
                                "raw_forecast": forecasts,
                            }
                        )
                    )
            result = pd.concat(rows, ignore_index=True).loc[:, list(FORECAST_COLUMNS)]
        finally:
            shutil.rmtree(temporary, ignore_errors=False)
        self.gate()
        return result

    def read_outcome_month(
        self,
        row: object,
        knowledge_date: dt.date,
    ) -> pd.DataFrame | None:
        from portfolio_bot.validation.calendar import ValidationCalendarRow

        self.gate()
        _require(
            type(row) is ValidationCalendarRow and type(knowledge_date) is dt.date,
            "outcome request chronology differs",
        )
        _require(
            _stable_identity(self._outcome_path) == self._outcome_identity,
            "validation label store changed during execution",
        )
        availability = pq.read_table(
            self._outcome_path,
            columns=["holding_month", "month_label_available_date"],
            filters=[("holding_month", "=", row.holding_month)],
            use_threads=False,
        )
        _require(availability.num_rows > 0, "validation outcome month is absent")
        dates = availability["month_label_available_date"].combine_chunks().to_pylist()
        _require(
            all(type(value) is dt.date and value >= row.exit_date for value in dates),
            "validation outcome availability metadata differs",
        )
        if max(dates) > knowledge_date:
            self.gate()
            return None
        table = pq.read_table(
            self._outcome_path,
            columns=list(_OUTCOME_SOURCE_COLUMNS),
            filters=[("holding_month", "=", row.holding_month)],
            use_threads=False,
        )
        _require(
            table.num_rows == availability.num_rows
            and _stable_identity(self._outcome_path) == self._outcome_identity,
            "validation outcome changed between availability and value reads",
        )
        frame = table.to_pandas().rename(
            columns={"month_label_available_date": "outcome_availability_date"}
        )
        result = frame.loc[:, list(OUTCOME_COLUMNS)].copy()
        self.gate()
        return result


__all__ = [
    "EmpiricalSelectorBackendError",
    "EmpiricalStage11SelectorBackend",
    "PreparedRefit",
    "prepare_authenticated_refit",
    "prepare_authenticated_score",
]
