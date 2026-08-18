"""Assemble and immutably seal the completed Stage 9 selector search.

The module is deliberately native-runtime free: it never imports PyTorch,
XGBoost, or scikit-learn model implementations and it never fits a model.  It
accepts complete development OOF cubes and fit receipts, constructs the exact
six payloads consumed by :mod:`portfolio_bot.selectors.publication`, validates
the complete scientific ledger in memory, and then publishes one flat,
content-addressed cache directory.  The canonical search receipt is written
last and closes the other six files by hash, size, row count, row-group count,
schema hash, and column registry.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import SecureStagingDirectory
from portfolio_bot.selectors.contract import (
    CONTRACT_ID,
    DEFAULT_POLICY,
    PROJECT_ROOT,
    canonical_json_bytes,
    sha256_file,
    validate_contract,
    validate_runtime_versions,
)
from portfolio_bot.selectors.evaluation import FamilyEvaluation
from portfolio_bot.selectors.metrics import (
    BOOTSTRAP_BLOCK_MONTHS,
    BOOTSTRAP_MASTER_SEED,
    BOOTSTRAP_REPLICATES,
    DEVELOPMENT_SEEDS,
)
from portfolio_bot.selectors.model_specs import (
    ElasticNetConfig,
    NeuralConfig,
    XGBoostConfig,
    configuration_payload_sha256,
    core_xgboost_configurations,
    elastic_net_configurations,
    neural_configurations,
)
from portfolio_bot.selectors.publication import (
    CONFIG_NAME,
    FAMILY_CONFIG_COUNTS,
    FAMILY_ORDER,
    FIT_RECEIPTS_NAME,
    FOLD_NAME,
    MONTHLY_NAME,
    OOF_NAME,
    SEARCH_BUNDLE_FILES,
    SEARCH_PAYLOAD_FILES,
    SEARCH_RECEIPT_NAME,
    SHORTLIST_NAME,
    expected_oof_schema,
    open_completed_search_bundle,
    published_schemas,
    validate_completed_search_tables,
    validate_search_bundle,
)
from portfolio_bot.selectors.results_bundle import (
    OOFDomain,
    build_wide_oof_table,
    raw_column_name,
    score_column_name,
)
from portfolio_bot.selectors.search_plan import (
    derive_gamma_challengers,
    select_gamma_parent_ids,
)
from portfolio_bot.selectors.shortlist import (
    SHORTLIST_COLUMNS,
    shortlist_all_families,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PREPARED_ID = re.compile(r"phase04-selector-prepared-[0-9a-f]{24}\Z")
_SEARCH_ID = re.compile(r"phase04-selector-search-[0-9a-f]{24}\Z")
_HOLDING_MONTH = re.compile(r"\d{4}-(0[1-9]|1[0-2])\Z")
_PARQUET_ROW_GROUP_SIZE = 2_048

_CODE_PATHS = (
    "src/portfolio_bot/data/secure_publication.py",
    "src/portfolio_bot/selectors/cache.py",
    "src/portfolio_bot/selectors/contract.py",
    "src/portfolio_bot/selectors/model_specs.py",
    "src/portfolio_bot/selectors/native_process.py",
    "src/portfolio_bot/selectors/native_worker.py",
    "src/portfolio_bot/selectors/fit_handlers.py",
    "src/portfolio_bot/selectors/prepared_data.py",
    "src/portfolio_bot/selectors/neural_model.py",
    "src/portfolio_bot/selectors/xgboost_model.py",
    "src/portfolio_bot/selectors/elastic_net.py",
    "src/portfolio_bot/selectors/search_plan.py",
    "src/portfolio_bot/selectors/results_bundle.py",
    "src/portfolio_bot/selectors/metrics.py",
    "src/portfolio_bot/selectors/evaluation.py",
    "src/portfolio_bot/selectors/complexity.py",
    "src/portfolio_bot/selectors/shortlist.py",
    "src/portfolio_bot/selectors/publication.py",
    "src/portfolio_bot/selectors/search_bundle.py",
    "src/portfolio_bot/selectors/search_execution.py",
)


class CompletedSearchError(RuntimeError):
    """Raised when completed Stage 9 results cannot be proven publishable."""


@dataclass(frozen=True)
class CompletedSearchInputs:
    """All completed, development-only inputs to the search bundle writer.

    ``fit_receipts`` is the already normalized one-row-per-published-fit table
    matching ``published_schemas()[FIT_RECEIPTS_NAME]``.  ``complexities`` is
    keyed by family; each value supplies exactly ``config_id`` and
    ``complexity`` (additional audit columns are permitted).  A caller may
    supply its independently constructed shortlist; the writer recomputes it
    from the frozen ledger and requires exact agreement before publication.
    """

    prepared_metadata: Mapping[str, Any]
    domain: OOFDomain
    raw_predictions: Mapping[str, np.ndarray]
    evaluations: Mapping[str, FamilyEvaluation]
    fit_receipts: pd.DataFrame | Sequence[Mapping[str, Any]]
    complexities: Mapping[str, pd.DataFrame]
    shortlist: pd.DataFrame | None
    sealed_xgboost_core_receipt_sha256: str
    gamma_parent_config_ids: Sequence[str]
    extra_code_paths: Sequence[str] = ()


@dataclass(frozen=True)
class CompletedSearchIdentity:
    search_id: str
    content_fingerprint: str
    prepared_build_id: str
    prepared_content_fingerprint: str
    prepared_metadata_sha256: str
    rules_sha256: str
    runtime_versions: Mapping[str, str | None]
    runtime_versions_sha256: str
    code_hashes: Mapping[str, str]
    bootstrap_plan_sha256: str
    raw_prediction_sha256s: Mapping[str, str]


@dataclass(frozen=True)
class CompletedSearchPublication:
    search_id: str
    path: Path
    receipt: Mapping[str, Any]
    receipt_sha256: str
    reused_existing: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompletedSearchError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _schema_sha256(schema: pa.Schema) -> str:
    return _sha256_bytes(schema.serialize().to_pybytes())


def _table_content_sha256(table: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return _sha256_bytes(sink.getvalue().to_pybytes())


def _array_sha256(values: np.ndarray, *, chunk_rows: int = 16_384) -> str:
    array = np.asarray(values)
    _require(not array.dtype.hasobject, "Object prediction arrays are prohibited")
    _require(array.ndim >= 1, "Prediction arrays must have a row axis")
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {"dtype": array.dtype.str, "shape": list(array.shape)}
        )
    )
    for start in range(0, array.shape[0], chunk_rows):
        block = np.ascontiguousarray(array[start : start + chunk_rows])
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _normalized_code_path(value: str) -> str:
    path = Path(value)
    _require(
        value and not path.is_absolute() and ".." not in path.parts,
        f"Unsafe Stage 9 code dependency: {value!r}",
    )
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:  # pragma: no cover - guarded above
        raise CompletedSearchError("Stage 9 code dependency escaped project") from exc
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        f"Missing safe Stage 9 code dependency: {value}",
    )
    return path.as_posix()


def current_search_code_hashes(
    extra_code_paths: Sequence[str] = (),
) -> dict[str, str]:
    """Hash every code dependency that can alter the assembled search."""

    paths = tuple(dict.fromkeys((*_CODE_PATHS, *(str(v) for v in extra_code_paths))))
    normalized = tuple(_normalized_code_path(value) for value in paths)
    return {value: sha256_file(PROJECT_ROOT / value) for value in normalized}


def current_search_runtime_versions() -> dict[str, str | None]:
    """Validate the frozen model runtime and record the full table runtime."""

    required = validate_runtime_versions()
    observed: dict[str, str | None] = {
        "python": platform.python_version(),
        **required,
    }
    for distribution in ("pandas", "pyarrow", "skglm", "numba", "llvmlite"):
        try:
            observed[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            observed[distribution] = None
    return dict(sorted(observed.items()))


def build_column_registry(
    evaluations: Mapping[str, FamilyEvaluation],
) -> dict[str, list[dict[str, Any]]]:
    """Return the sole mapping from opaque OOF columns to model identities."""

    raw: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        _require(family in evaluations, f"Missing family evaluation: {family}")
        config_ids = tuple(str(value) for value in evaluations[family].config_ids)
        _require(
            len(config_ids) == FAMILY_CONFIG_COUNTS[family]
            and len(set(config_ids)) == len(config_ids),
            f"{family} configuration registry differs",
        )
        seeds: tuple[int | None, ...] = (
            DEVELOPMENT_SEEDS if family != "elastic_net" else (None,)
        )
        for ordinal, config_id in enumerate(config_ids):
            for seed in seeds:
                raw.append(
                    {
                        "column_name": raw_column_name(family, ordinal, seed),
                        "model_family": family,
                        "config_id": config_id,
                        "seed": seed,
                    }
                )
            scores.append(
                {
                    "column_name": score_column_name(family, ordinal),
                    "model_family": family,
                    "config_id": config_id,
                }
            )
    _require(len(raw) == 538 and len(scores) == 346, "OOF column registry totals differ")
    return {"raw_forecasts": raw, "monthly_scores": scores}


def _coerce_table(frame: pd.DataFrame, schema: pa.Schema, *, name: str) -> pa.Table:
    missing = [column for column in schema.names if column not in frame.columns]
    _require(not missing, f"{name} is missing columns: {missing}")
    selected = frame.loc[:, schema.names].copy()
    if "holding_month" in selected:
        values = selected["holding_month"]
        _require(
            values.map(lambda value: isinstance(value, str) and _HOLDING_MONTH.fullmatch(value) is not None).all(),
            f"{name} holding_month must remain an exact YYYY-MM string",
        )
    if "signal_date" in selected:
        selected["signal_date"] = pd.to_datetime(
            selected["signal_date"], errors="raise"
        ).dt.date
    try:
        table = pa.Table.from_pandas(
            selected,
            schema=schema,
            preserve_index=False,
            safe=True,
        ).combine_chunks()
        # ``Table.from_pandas`` attaches a pandas interchange metadata block
        # even when an explicit Arrow schema is supplied.  That metadata is
        # runtime-generated rather than part of the frozen publication
        # contract, so normalize it back to the contract schema before the
        # exact equality check and receipt hashing below.
        table = table.replace_schema_metadata(schema.metadata)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise CompletedSearchError(f"{name} cannot be cast to its frozen schema") from exc
    _require(
        table.schema.equals(schema, check_metadata=True),
        f"{name} schema differs after casting",
    )
    return table


def _validate_family_inputs(inputs: CompletedSearchInputs) -> None:
    _require(
        set(inputs.raw_predictions) == set(FAMILY_ORDER),
        "Raw prediction family set differs",
    )
    _require(
        set(inputs.evaluations) == set(FAMILY_ORDER),
        "Evaluation family set differs",
    )
    _require(
        set(inputs.complexities) == set(FAMILY_ORDER),
        "Complexity family set differs",
    )
    for family in FAMILY_ORDER:
        evaluation = inputs.evaluations[family]
        _require(
            evaluation.model_family == family,
            f"Evaluation is labelled with the wrong family: {family}",
        )
        raw = np.asarray(inputs.raw_predictions[family])
        _require(
            raw.dtype == np.dtype("float32") and np.isfinite(raw).all(),
            f"{family} OOF cube must be finite float32",
        )
        expected_mean = (
            raw
            if family == "elastic_net"
            else np.mean(raw, axis=2, dtype=np.float64).astype(np.float32)
        )
        observed_mean = np.asarray(evaluation.raw_forecast_mean, dtype=np.float32)
        _require(
            expected_mean.shape == observed_mean.shape
            and np.array_equal(expected_mean, observed_mean),
            f"{family} evaluation is not bound to its raw OOF cube",
        )


def _bootstrap_binding(
    evaluations: Mapping[str, FamilyEvaluation],
) -> tuple[dict[str, Any], str]:
    first = evaluations[FAMILY_ORDER[0]]
    months = tuple(str(value) for value in first.bootstrap_months)
    indices = np.asarray(first.bootstrap_indices)
    _require(
        len(months) == 95
        and all(_HOLDING_MONTH.fullmatch(value) is not None for value in months)
        and indices.shape == (BOOTSTRAP_REPLICATES, 95)
        and indices.dtype == np.dtype("int32"),
        "Frozen development bootstrap plan dimensions differ",
    )
    for family in FAMILY_ORDER[1:]:
        evaluation = evaluations[family]
        _require(
            tuple(str(value) for value in evaluation.bootstrap_months) == months
            and np.array_equal(np.asarray(evaluation.bootstrap_indices), indices),
            "Families did not use one shared bootstrap plan",
        )
    plan_payload = canonical_json_bytes(
        {
            "months": list(months),
            "dtype": indices.dtype.str,
            "shape": list(indices.shape),
        }
    ) + memoryview(np.ascontiguousarray(indices)).cast("B").tobytes()
    plan_sha = _sha256_bytes(plan_payload)
    binding = {
        "method": "non_circular_overlapping_moving_block",
        "statistical_unit": "calendar_month",
        "replicates": BOOTSTRAP_REPLICATES,
        "block_length_months": BOOTSTRAP_BLOCK_MONTHS,
        "rng": "numpy_random_PCG64",
        "master_seed": BOOTSTRAP_MASTER_SEED,
        "shared_resample_plan_across_configurations": True,
        "months": list(months),
        "indices_dtype": indices.dtype.str,
        "indices_shape": list(indices.shape),
        "plan_sha256": plan_sha,
    }
    return binding, plan_sha


def _complexity_by_family(
    inputs: CompletedSearchInputs,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for family in FAMILY_ORDER:
        frame = inputs.complexities[family].copy()
        _require(
            {"config_id", "complexity"} <= set(frame),
            f"{family} complexity table is incomplete",
        )
        ids = tuple(str(value) for value in inputs.evaluations[family].config_ids)
        frame["config_id"] = frame["config_id"].astype(str)
        _require(
            len(frame) == len(ids)
            and not frame["config_id"].duplicated().any()
            and set(frame["config_id"]) == set(ids),
            f"{family} complexity grid differs",
        )
        values = pd.to_numeric(frame["complexity"], errors="coerce")
        _require(
            np.isfinite(values.to_numpy(np.float64)).all() and bool((values >= 0.0).all()),
            f"{family} complexity values differ",
        )
        frame["complexity"] = values.astype(np.float64)
        result[family] = frame.set_index("config_id").loc[list(ids)].reset_index()
    return result


def _config_json(config: NeuralConfig | XGBoostConfig | ElasticNetConfig) -> str:
    return _canonical_text(asdict(config))


def _regularization_key(
    family: str,
    config: NeuralConfig | XGBoostConfig | ElasticNetConfig,
    complexity: float,
) -> list[float]:
    # The first field implements the frozen lower-complexity tie.  XGBoost
    # then applies its explicitly preregistered stronger-regularization tuple.
    if family != "xgboost":
        return [float(complexity)]
    assert isinstance(config, XGBoostConfig)
    return [
        float(complexity),
        -float(config.reg_lambda),
        -float(config.reg_alpha),
        -float(config.child_share),
        float(config.subsample),
        float(config.colsample_bytree),
        float(config.eta),
    ]


def _assemble_configuration_ledger(
    inputs: CompletedSearchInputs,
    complexities: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, tuple[XGBoostConfig, ...], tuple[str, ...]]:
    nn = neural_configurations()
    enet = elastic_net_configurations()
    _require(
        tuple(inputs.evaluations["neural_network"].config_ids)
        == tuple(value.config_id for value in nn),
        "Neural evaluation grid/order differs from the frozen grid",
    )
    _require(
        tuple(inputs.evaluations["elastic_net"].config_ids)
        == tuple(value.config_id for value in enet),
        "Elastic Net evaluation grid/order differs from the frozen grid",
    )

    core = core_xgboost_configurations()
    supplied_parents = tuple(str(value) for value in inputs.gamma_parent_config_ids)
    _require(
        _SHA256.fullmatch(inputs.sealed_xgboost_core_receipt_sha256) is not None,
        "A sealed XGBoost core receipt SHA-256 is required",
    )
    gamma = derive_gamma_challengers(
        supplied_parents,
        sealed_core_receipt_sha256=inputs.sealed_xgboost_core_receipt_sha256,
    )
    xgb = (*core, *gamma)
    _require(
        tuple(inputs.evaluations["xgboost"].config_ids)
        == tuple(value.config_id for value in xgb),
        "XGBoost evaluation grid/order differs from its sealed gamma parents",
    )

    metrics: dict[str, pd.DataFrame] = {}
    for family in FAMILY_ORDER:
        evaluation = inputs.evaluations[family]
        frame = evaluation.config_metrics.copy()
        frame["config_id"] = frame["config_id"].astype(str)
        ids = list(evaluation.config_ids)
        _require(
            len(frame) == len(ids)
            and not frame["config_id"].duplicated().any()
            and set(frame["config_id"]) == set(ids),
            f"{family} configuration metrics grid differs",
        )
        frame = frame.set_index("config_id").loc[ids].reset_index()
        complexity = complexities[family][["config_id", "complexity"]]
        frame = frame.merge(complexity, on="config_id", validate="one_to_one")
        metrics[family] = frame

    recomputed_parents = select_gamma_parent_ids(
        metrics["xgboost"].iloc[:40][
            ["config_id", "mean_rank_ic", "complexity"]
        ]
    )
    _require(
        recomputed_parents == supplied_parents,
        "Gamma parents differ from the sealed top-four core development IC rule",
    )

    config_objects: Mapping[str, Sequence[NeuralConfig | XGBoostConfig | ElasticNetConfig]] = {
        "neural_network": nn,
        "xgboost": xgb,
        "elastic_net": enet,
    }
    outputs: list[pd.DataFrame] = []
    for family in FAMILY_ORDER:
        frame = metrics[family].copy()
        objects = tuple(config_objects[family])
        by_id = {value.config_id: value for value in objects}
        frame["trial_phase"] = (
            ["core_grid"] * 40 + ["gamma_challenger"] * 8
            if family == "xgboost"
            else "full_grid"
        )
        frame["parent_config_id"] = [
            (
                by_id[config_id].core_config_id
                if family == "xgboost"
                and isinstance(by_id[config_id], XGBoostConfig)
                and by_id[config_id].challenger
                else None
            )
            for config_id in frame["config_id"]
        ]
        frame["configuration_json"] = [
            _config_json(by_id[config_id]) for config_id in frame["config_id"]
        ]
        keys = [
            _regularization_key(
                family,
                by_id[row.config_id],
                float(row.complexity),
            )
            for row in frame[["config_id", "complexity"]].itertuples(index=False)
        ]
        frame["_regularization_key"] = [tuple(value) for value in keys]
        frame["regularization_key_json"] = [_canonical_text(value) for value in keys]
        outputs.append(frame)
    ledger = pd.concat(outputs, ignore_index=True)
    return ledger, tuple(xgb), recomputed_parents


def _shortlist(
    ledger: pd.DataFrame,
    supplied: pd.DataFrame | None,
) -> pd.DataFrame:
    computed = shortlist_all_families(
        ledger,
        regularization_columns_by_family={
            family: ("_regularization_key",) for family in FAMILY_ORDER
        },
    )
    family_order = {value: index for index, value in enumerate(FAMILY_ORDER)}
    computed["_family_order"] = computed["model_family"].map(family_order)
    computed = computed.sort_values(
        ["_family_order", "shortlist_order", "config_id"], kind="mergesort"
    ).drop(columns="_family_order").reset_index(drop=True)
    computed = computed.loc[:, list(SHORTLIST_COLUMNS)]
    if supplied is not None:
        missing = sorted(set(SHORTLIST_COLUMNS).difference(supplied.columns))
        _require(not missing, f"Supplied shortlist is missing fields: {missing}")
        observed = supplied.loc[:, list(SHORTLIST_COLUMNS)].copy()
        observed["_family_order"] = observed["model_family"].map(family_order)
        observed = observed.sort_values(
            ["_family_order", "shortlist_order", "config_id"], kind="mergesort"
        ).drop(columns="_family_order").reset_index(drop=True)
        _require(
            len(observed) == len(computed),
            "Supplied shortlist row count differs from frozen recomputation",
        )
        float_columns = {
            "mean_rank_ic",
            "rank_ic_bootstrap_se",
            "fold_dispersion",
            "complexity",
            "family_best_mean_rank_ic",
            "family_best_bootstrap_se",
            "one_se_threshold",
        }
        for column in SHORTLIST_COLUMNS:
            if column in float_columns:
                matches = np.allclose(
                    pd.to_numeric(observed[column]).to_numpy(np.float64),
                    pd.to_numeric(computed[column]).to_numpy(np.float64),
                    rtol=0.0,
                    atol=1e-12,
                )
            else:
                matches = observed[column].astype(str).equals(
                    computed[column].astype(str)
                )
            _require(matches, f"Supplied shortlist differs in {column}")
    return computed


def _normalize_fit_receipts(
    value: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    schema = published_schemas()[FIT_RECEIPTS_NAME]
    missing = sorted(set(schema.names).difference(frame.columns))
    _require(not missing, f"Fit receipts are missing fields: {missing}")
    frame = frame.loc[:, schema.names].copy()
    family_order = {value: index for index, value in enumerate(FAMILY_ORDER)}
    config_order: dict[tuple[str, str], int] = {}
    # Lexical config IDs already follow every canonical family order, including
    # parent-major gamma IDs, but the explicit map makes that invariant clear.
    for family in FAMILY_ORDER:
        ids = frame.loc[frame["model_family"].eq(family), "config_id"].astype(str).unique()
        for ordinal, config_id in enumerate(sorted(ids)):
            config_order[(family, config_id)] = ordinal
    frame["_family_order"] = frame["model_family"].map(family_order)
    frame["_config_order"] = [
        config_order.get((str(family), str(config_id)), -1)
        for family, config_id in zip(frame["model_family"], frame["config_id"], strict=True)
    ]
    frame["_seed_order"] = pd.to_numeric(frame["seed"], errors="coerce").fillna(-1)
    frame = frame.sort_values(
        ["_family_order", "_config_order", "fold_id", "_seed_order"],
        kind="mergesort",
    ).drop(columns=["_family_order", "_config_order", "_seed_order"])
    return frame.reset_index(drop=True)


def _validate_fit_prediction_bindings(
    inputs: CompletedSearchInputs,
    fit_receipts: pd.DataFrame,
) -> None:
    """Bind every normalized fit row to its exact raw OOF forecast bytes.

    Native cache archives are authenticated before assembly, but the immutable
    search must independently prove that the forecast columns it publishes are
    the same columns named by those receipts.  The hash convention is exactly
    the handler convention: contiguous little-endian float32 C-order bytes,
    with no shape or metadata prefix.
    """

    required = {
        "model_family",
        "config_id",
        "fold_id",
        "seed",
        "prediction_sha256",
        "prediction_rows",
        "outer_validation_rows",
    }
    _require(required <= set(fit_receipts), "Fit binding fields are incomplete")
    _require(fit_receipts["job_id"].is_unique, "Fit receipt job_id is duplicated")
    config_positions = {
        family: {
            str(config_id): ordinal
            for ordinal, config_id in enumerate(inputs.evaluations[family].config_ids)
        }
        for family in FAMILY_ORDER
    }
    expected: set[tuple[str, str, str, int | None]] = set()
    for family in FAMILY_ORDER:
        seeds: tuple[int | None, ...] = (
            DEVELOPMENT_SEEDS if family != "elastic_net" else (None,)
        )
        expected.update(
            (family, str(config_id), fold_id, seed)
            for config_id in inputs.evaluations[family].config_ids
            for fold_id in inputs.domain.fold_output_positions
            for seed in seeds
        )

    observed: set[tuple[str, str, str, int | None]] = set()
    seed_position = {seed: position for position, seed in enumerate(DEVELOPMENT_SEEDS)}
    for row in fit_receipts.itertuples(index=False):
        family = str(row.model_family)
        config_id = str(row.config_id)
        fold_id = str(row.fold_id)
        seed = None if pd.isna(row.seed) else int(row.seed)
        identity = (family, config_id, fold_id, seed)
        _require(identity not in observed, f"Duplicate normalized fit identity: {identity}")
        _require(identity in expected, f"Unknown normalized fit identity: {identity}")
        observed.add(identity)
        positions = np.asarray(
            inputs.domain.fold_output_positions[fold_id], dtype=np.int64
        )
        config_position = config_positions[family][config_id]
        raw = np.asarray(inputs.raw_predictions[family])
        if family == "elastic_net":
            _require(seed is None, "Elastic Net fit receipt unexpectedly carries a seed")
            prediction = raw[positions, config_position]
        else:
            _require(seed in seed_position, "Stochastic fit receipt seed differs")
            prediction = raw[positions, config_position, seed_position[int(seed)]]
        prediction = np.ascontiguousarray(prediction, dtype=np.float32)
        expected_hash = hashlib.sha256(prediction.tobytes(order="C")).hexdigest()
        _require(
            int(row.prediction_rows) == len(positions)
            and int(row.outer_validation_rows) == len(positions),
            f"Fit prediction row binding differs: {identity}",
        )
        _require(
            str(row.prediction_sha256) == expected_hash,
            f"Fit prediction SHA-256 differs from raw OOF bytes: {identity}",
        )
    _require(observed == expected, "Normalized fit identity grid is incomplete")


def assemble_completed_search_tables(
    inputs: CompletedSearchInputs,
) -> tuple[dict[str, pa.Table], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Assemble all six exact search payloads without writing a file."""

    _validate_family_inputs(inputs)
    domain_rows = inputs.domain.rows
    _require(
        len(domain_rows) == 172_969
        and "outcome_valid" in domain_rows
        and int(domain_rows["outcome_valid"].astype(bool).sum()) == 172_799,
        "Completed search OOF development domain counts differ",
    )
    signal_dates = pd.to_datetime(domain_rows["signal_date"], errors="raise")
    holding_months = domain_rows["holding_month"]
    _require(
        signal_dates.max() <= pd.Timestamp("2014-10-31")
        and holding_months.map(
            lambda value: isinstance(value, str)
            and _HOLDING_MONTH.fullmatch(value) is not None
            and value < "2015-01"
        ).all()
        and holding_months.nunique() == 95,
        "Completed search contains a 2015+ or malformed OOF month",
    )
    registry = build_column_registry(inputs.evaluations)
    bootstrap, _ = _bootstrap_binding(inputs.evaluations)
    complexities = _complexity_by_family(inputs)
    ledger, xgb_configs, gamma_parents = _assemble_configuration_ledger(
        inputs, complexities
    )
    shortlist = _shortlist(ledger, inputs.shortlist)

    oof_frame = build_wide_oof_table(
        inputs.domain,
        raw_predictions=inputs.raw_predictions,
        evaluations=inputs.evaluations,
    )
    _require(
        not any(
            value in oof_frame.columns
            for value in (
                "unranked_residual_return",
                "residual_return_rank",
                "outcome_valid",
                "primary_label_status",
            )
        ),
        "OOF handoff contains a target column",
    )
    provisional_receipt = {"column_registry": registry}
    normalized_fit_receipts = _normalize_fit_receipts(inputs.fit_receipts)
    _validate_fit_prediction_bindings(inputs, normalized_fit_receipts)
    tables: dict[str, pa.Table] = {
        OOF_NAME: _coerce_table(
            oof_frame,
            expected_oof_schema(provisional_receipt),
            name=OOF_NAME,
        ),
        FIT_RECEIPTS_NAME: _coerce_table(
            normalized_fit_receipts,
            published_schemas()[FIT_RECEIPTS_NAME],
            name=FIT_RECEIPTS_NAME,
        ),
    }

    monthly = pd.concat(
        [inputs.evaluations[family].monthly_metrics for family in FAMILY_ORDER],
        ignore_index=True,
    ).sort_values(["model_family", "config_id", "holding_month"], kind="mergesort")
    fold = pd.concat(
        [inputs.evaluations[family].fold_metrics for family in FAMILY_ORDER],
        ignore_index=True,
    ).sort_values(["model_family", "config_id", "fold_id"], kind="mergesort")
    configuration = ledger.loc[:, published_schemas()[CONFIG_NAME].names]
    tables[MONTHLY_NAME] = _coerce_table(
        monthly.reset_index(drop=True),
        published_schemas()[MONTHLY_NAME],
        name=MONTHLY_NAME,
    )
    tables[CONFIG_NAME] = _coerce_table(
        configuration.reset_index(drop=True),
        published_schemas()[CONFIG_NAME],
        name=CONFIG_NAME,
    )
    tables[FOLD_NAME] = _coerce_table(
        fold.reset_index(drop=True),
        published_schemas()[FOLD_NAME],
        name=FOLD_NAME,
    )
    tables[SHORTLIST_NAME] = _coerce_table(
        shortlist.reset_index(drop=True),
        published_schemas()[SHORTLIST_NAME],
        name=SHORTLIST_NAME,
    )
    gamma_binding = {
        "sealed_core_receipt_sha256": inputs.sealed_xgboost_core_receipt_sha256,
        "gamma_parent_config_ids": list(gamma_parents),
        "gamma_challenger_config_ids": [value.config_id for value in xgb_configs[40:]],
        "gamma_challenger_payload_sha256s": {
            value.config_id: configuration_payload_sha256(value)
            for value in xgb_configs[40:]
        },
    }
    return tables, registry, {"bootstrap": bootstrap, "xgboost": gamma_binding}


def _prepared_binding(metadata: Mapping[str, Any]) -> dict[str, str]:
    value = dict(metadata)
    build_id = str(value.get("build_id", ""))
    fingerprint = str(value.get("content_fingerprint", ""))
    _require(_PREPARED_ID.fullmatch(build_id) is not None, "Prepared build ID is malformed")
    _require(_SHA256.fullmatch(fingerprint) is not None, "Prepared fingerprint is malformed")
    _require(
        value.get("contract_id") == CONTRACT_ID
        and value.get("status") == "PREPARED_COMPLETE",
        "Prepared metadata is not a completed Stage 9 cache",
    )
    scope = value.get("scope")
    _require(
        scope
        == {
            "development_only": True,
            "model_fits_performed": False,
            "stage10_or_later_permitted": False,
            "portfolio_or_backtest_permitted": False,
            "validation_or_final_payload_accessed": False,
        },
        "Prepared metadata scope escaped Stage 9",
    )
    return {
        "build_id": build_id,
        "content_fingerprint": fingerprint,
        "metadata_sha256": _sha256_bytes(canonical_json_bytes(value)),
    }


def compute_completed_search_identity(
    inputs: CompletedSearchInputs,
    tables: Mapping[str, pa.Table],
    bindings: Mapping[str, Any],
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> CompletedSearchIdentity:
    """Return the deterministic identity of the exact completed search."""

    policy_path = policy_path.expanduser().resolve()
    policy, rules = validate_contract(policy_path)
    prepared = _prepared_binding(inputs.prepared_metadata)
    runtime = current_search_runtime_versions()
    runtime_sha = _sha256_bytes(canonical_json_bytes(runtime))
    code_hashes = current_search_code_hashes(inputs.extra_code_paths)
    raw_hashes = {
        family: _array_sha256(np.asarray(inputs.raw_predictions[family]))
        for family in FAMILY_ORDER
    }
    compact_hashes = {
        name: _table_content_sha256(tables[name])
        for name in SEARCH_PAYLOAD_FILES
        if name != OOF_NAME
    }
    oof_key_hash = _table_content_sha256(
        tables[OOF_NAME].select(("fold_id", "signal_date", "holding_month", "permno"))
    )
    rules_path = (PROJECT_ROOT / str(policy["physical_rules"]["path"])).resolve()
    rules_sha = sha256_file(rules_path)
    _require(
        rules_sha == str(policy["physical_rules"]["sha256"]),
        "Current physical rules differ from the frozen policy",
    )
    scope = {
        "development_only": True,
        "first_development_signal_date": "2000-01-31",
        "last_development_signal_date": "2014-10-31",
        "validation_2015_2019_rows_read": 0,
        "final_2020_2025_rows_read": 0,
        "stage10_or_later_permitted": False,
        "portfolio_weights_created": 0,
        "trades_or_pnl_created": 0,
        "sharpe_or_portfolio_performance_calculated": False,
    }
    payload = {
        "schema_version": 1,
        "purpose": "stage09_completed_development_selector_search_only",
        "contract_id": CONTRACT_ID,
        "policy_id": policy["policy_id"],
        "policy_sha256": sha256_file(policy_path),
        "rules_id": rules["rules_id"],
        "rules_sha256": rules_sha,
        "prepared": prepared,
        "runtime_versions": runtime,
        "runtime_versions_sha256": runtime_sha,
        "code_hashes": code_hashes,
        "raw_prediction_sha256s": raw_hashes,
        "compact_table_content_sha256s": compact_hashes,
        "oof_key_content_sha256": oof_key_hash,
        "bootstrap": bindings["bootstrap"],
        "xgboost_sealed_core": bindings["xgboost"],
        "scope": scope,
    }
    fingerprint = _sha256_bytes(canonical_json_bytes(payload))
    return CompletedSearchIdentity(
        search_id=f"phase04-selector-search-{fingerprint[:24]}",
        content_fingerprint=fingerprint,
        prepared_build_id=prepared["build_id"],
        prepared_content_fingerprint=prepared["content_fingerprint"],
        prepared_metadata_sha256=prepared["metadata_sha256"],
        rules_sha256=rules_sha,
        runtime_versions=runtime,
        runtime_versions_sha256=runtime_sha,
        code_hashes=code_hashes,
        bootstrap_plan_sha256=str(bindings["bootstrap"]["plan_sha256"]),
        raw_prediction_sha256s=raw_hashes,
    )


class _DigestingWriter(io.RawIOBase):
    """A sequential binary adapter that hashes exactly the bytes accepted."""

    def __init__(self, stream: Any) -> None:
        super().__init__()
        self._stream = stream
        self._digest = hashlib.sha256()
        self._size = 0

    def writable(self) -> bool:  # pragma: no cover - queried by PyArrow
        return True

    def seekable(self) -> bool:  # pragma: no cover - queried by PyArrow
        return False

    def tell(self) -> int:
        return self._size

    def write(self, value: Any) -> int:
        block = memoryview(value)
        written = self._stream.write(block)
        if written is None:
            written = len(block)
        _require(written == len(block), "Short Parquet staging write")
        self._digest.update(block[:written])
        self._size += int(written)
        return int(written)

    def flush(self) -> None:
        self._stream.flush()

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    @property
    def size_bytes(self) -> int:
        return self._size

    def close(self) -> None:
        # PyArrow may close its Python wrapper.  The secure staging context owns
        # the underlying descriptor and performs its authenticated fsync.
        if not self.closed:
            self.flush()
            super().close()


def _write_parquet_member(
    stage: SecureStagingDirectory,
    name: str,
    table: pa.Table,
) -> dict[str, Any]:
    with stage.open_exclusive_binary(name) as stream:
        writer = _DigestingWriter(stream)
        pq.write_table(
            table,
            writer,
            compression="zstd",
            use_dictionary=True,
            row_group_size=_PARQUET_ROW_GROUP_SIZE,
            write_statistics=True,
            version="2.6",
        )
        writer.flush()
        digest = writer.sha256
        size = writer.size_bytes
        writer.close()
    return {
        "sha256": digest,
        "size_bytes": size,
        "rows": table.num_rows,
        "columns": table.num_columns,
        "row_groups": int(math.ceil(table.num_rows / _PARQUET_ROW_GROUP_SIZE)),
        "schema_sha256": _schema_sha256(table.schema),
    }


def _safe_relative_cache_root(policy: Mapping[str, Any]) -> Path:
    relative = Path(str(policy["database"]["cache_root"]))
    _require(
        relative.parts and not relative.is_absolute() and ".." not in relative.parts,
        "Unsafe Stage 9 database cache root",
    )
    return relative


def _receipt(
    identity: CompletedSearchIdentity,
    files: Mapping[str, Mapping[str, Any]],
    registry: Mapping[str, Any],
    bindings: Mapping[str, Any],
    *,
    policy_path: Path,
) -> dict[str, Any]:
    policy, rules = validate_contract(policy_path)
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "policy_id": policy["policy_id"],
        "search_id": identity.search_id,
        "content_fingerprint": identity.content_fingerprint,
        "status": "COMPLETE",
        "prepared": {
            "build_id": identity.prepared_build_id,
            "content_fingerprint": identity.prepared_content_fingerprint,
            "metadata_sha256": identity.prepared_metadata_sha256,
        },
        "rules": {
            "rules_id": rules["rules_id"],
            "sha256": identity.rules_sha256,
        },
        "runtime": {
            "versions": dict(identity.runtime_versions),
            "versions_sha256": identity.runtime_versions_sha256,
            "threads_per_fit": 1,
            "authoritative_process_isolation": True,
        },
        "code_hashes": dict(sorted(identity.code_hashes.items())),
        "raw_prediction_sha256s": dict(identity.raw_prediction_sha256s),
        "xgboost_sealed_core": bindings["xgboost"],
        "bootstrap": bindings["bootstrap"],
        "column_registry": registry,
        "files": {name: dict(files[name]) for name in SEARCH_PAYLOAD_FILES},
        "scope": {
            "development_only": True,
            "first_development_signal_date": "2000-01-31",
            "last_development_signal_date": "2014-10-31",
            "validation_2015_2019_rows_read": 0,
            "final_2020_2025_rows_read": 0,
            "stage10_or_later_permitted": False,
            "portfolio_weights_created": 0,
            "trades_or_pnl_created": 0,
            "sharpe_or_portfolio_performance_calculated": False,
        },
        "authorizations": {
            "primary_publication_authorized": True,
            "independent_verification_completed": False,
            "stage_10_allocation_architectures_authorized": False,
            "stage_11_or_later_authorized": False,
            "portfolios_pnl_or_backtests_authorized": False,
            "validation_or_final_test_access_authorized": False,
        },
    }


def _load_identical_existing(
    path: Path,
    *,
    database_root: Path,
    identity: CompletedSearchIdentity,
) -> CompletedSearchPublication | None:
    if not path.exists():
        return None
    with open_completed_search_bundle(path, database_root=database_root) as (
        observed,
        guard,
    ):
        validate_search_bundle(observed, guard)
        _require(observed.search_id == identity.search_id, "Existing search ID differs")
        _require(
            observed.receipt.get("content_fingerprint")
            == identity.content_fingerprint,
            "Existing search fingerprint differs",
        )
        guard.gate()
        return CompletedSearchPublication(
            search_id=identity.search_id,
            path=path,
            receipt=dict(observed.receipt),
            receipt_sha256=observed.receipt_sha256,
            reused_existing=True,
        )


def publish_completed_search_bundle(
    inputs: CompletedSearchInputs,
    *,
    database_root: Path,
    output_parent: Path | None = None,
    policy_path: Path = DEFAULT_POLICY,
    progress: Callable[[str], None] | None = None,
) -> CompletedSearchPublication:
    """Validate, atomically close, and return one completed search bundle.

    This function publishes only the database-cache search bundle.  It never
    publishes the Stage 9 primary pointer and it grants no Stage 10 authority.
    Calling it again with identical inputs reauthenticates and reuses the same
    content-addressed directory.
    """

    emit = progress if progress is not None else (lambda _message: None)
    policy_path = policy_path.expanduser().resolve()
    policy, _ = validate_contract(policy_path)
    root = Path(os.path.abspath(database_root.expanduser()))
    _require(root.is_dir() and not root.is_symlink(), "Database root is unsafe or absent")
    _require(
        str(policy["database"]["root"]) == str(root),
        "Database root differs from the frozen Stage 9 policy",
    )

    emit("assembling six exact development-only selector search payloads")
    tables, registry, bindings = assemble_completed_search_tables(inputs)
    identity = compute_completed_search_identity(
        inputs, tables, bindings, policy_path=policy_path
    )
    _require(_SEARCH_ID.fullmatch(identity.search_id) is not None, "Search ID is malformed")
    parent = (
        root
        / _safe_relative_cache_root(policy)
        / identity.prepared_build_id
        / "searches"
        if output_parent is None
        else Path(os.path.abspath(output_parent.expanduser()))
    )
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise CompletedSearchError("Search output parent escaped database root") from exc
    destination = parent / identity.search_id

    existing = _load_identical_existing(
        destination,
        database_root=root,
        identity=identity,
    )
    if existing is not None:
        emit("reauthenticated and reused the identical completed search bundle")
        return existing

    # The in-memory gate deliberately uses the same validation implementation
    # as the later primary publisher.  A malformed immutable directory is never
    # created and left behind.
    validation_receipt = {"column_registry": registry}
    validate_completed_search_tables(tables, validation_receipt)
    emit("scientific ledger and exact publication schemas passed in-memory gates")

    with SecureStagingDirectory.create(
        trusted_root=root,
        destination=destination,
        expected_files=SEARCH_BUNDLE_FILES,
        label="Stage 9 completed selector search",
    ) as stage:
        file_receipts: dict[str, dict[str, Any]] = {}
        hashes: dict[str, str] = {}
        for name in SEARCH_PAYLOAD_FILES:
            record = _write_parquet_member(stage, name, tables[name])
            file_receipts[name] = record
            hashes[name] = str(record["sha256"])
            emit(f"sealed search payload {name}")
        receipt = _receipt(
            identity,
            file_receipts,
            registry,
            bindings,
            policy_path=policy_path,
        )
        payload = canonical_json_bytes(receipt)
        with stage.open_exclusive_binary(SEARCH_RECEIPT_NAME) as stream:
            written = stream.write(payload)
            _require(written == len(payload), "Short completed-search receipt write")
        hashes[SEARCH_RECEIPT_NAME] = _sha256_bytes(payload)
        snapshot = stage.snapshot(expected_hashes=hashes)
        publication = stage.publish(expected_hashes=snapshot.hashes)

    emit("published canonical receipt last and closed the immutable search directory")
    completed = _load_identical_existing(
        publication.path,
        database_root=root,
        identity=identity,
    )
    _require(completed is not None, "Published search bundle could not be reauthenticated")
    return CompletedSearchPublication(
        search_id=completed.search_id,
        path=completed.path,
        receipt=completed.receipt,
        receipt_sha256=completed.receipt_sha256,
        reused_existing=publication.reused_existing,
    )


__all__ = [
    "CompletedSearchError",
    "CompletedSearchIdentity",
    "CompletedSearchInputs",
    "CompletedSearchPublication",
    "assemble_completed_search_tables",
    "build_column_registry",
    "compute_completed_search_identity",
    "current_search_code_hashes",
    "current_search_runtime_versions",
    "publish_completed_search_bundle",
]
