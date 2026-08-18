"""Immutable Stage 9 selector-results publication.

This module consumes a *completed* development-only selector search bundle.  It
does not fit models, choose portfolio weights, calculate P&L, or read the
2015--2025 validation/final periods.  The input bundle is held with
``AuthenticatedBundleGuard`` while three content-addressed database bundles
and the artifact manifest are published.  The mutable primary pointer is
replaced last and authorizes independent verification only.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    AuthenticatedBundleGuard,
    AuthenticatedPointerGuard,
    PublishedBundle,
    SecureStagingDirectory,
    replace_pointer_last,
)
from portfolio_bot.selectors.contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
    canonical_json_bytes,
    project_path,
    sha256_file,
    validate_contract,
)
from portfolio_bot.selectors.inputs import Stage9Inputs, resolve_stage9_inputs


SUCCESS_NAME = "_SUCCESS.json"
MANIFEST_NAME = "manifest.json"
SEARCH_RECEIPT_NAME = "selector_search_receipt.json"
OOF_NAME = "selector_oof_forecasts.parquet"
FIT_RECEIPTS_NAME = "selector_fit_receipts.parquet"
MONTHLY_NAME = "selector_monthly_metrics.parquet"
CONFIG_NAME = "selector_configuration_metrics.parquet"
FOLD_NAME = "selector_fold_metrics.parquet"
SHORTLIST_NAME = "selector_shortlist.parquet"
CHECKS_NAME = "selector_publication_checks.parquet"
SUMMARY_NAME = "selector_publication_summary.json"

SEARCH_PAYLOAD_FILES = (
    OOF_NAME,
    FIT_RECEIPTS_NAME,
    MONTHLY_NAME,
    CONFIG_NAME,
    FOLD_NAME,
    SHORTLIST_NAME,
)
SEARCH_BUNDLE_FILES = (*SEARCH_PAYLOAD_FILES, SEARCH_RECEIPT_NAME)

INTERIM_FILES = (OOF_NAME, FIT_RECEIPTS_NAME, SUCCESS_NAME)
PROCESSED_FILES = (
    MONTHLY_NAME,
    CONFIG_NAME,
    FOLD_NAME,
    SHORTLIST_NAME,
    SEARCH_RECEIPT_NAME,
    SUCCESS_NAME,
)
QUALITY_FILES = (CHECKS_NAME, SUMMARY_NAME, SUCCESS_NAME)
MANIFEST_FILES = (MANIFEST_NAME, SUCCESS_NAME)

FAMILY_ORDER = ("neural_network", "xgboost", "elastic_net")
FAMILY_CODES = {
    "neural_network": "nn",
    "xgboost": "xgb",
    "elastic_net": "enet",
}
FAMILY_CONFIG_COUNTS = {
    "neural_network": 48,
    "xgboost": 48,
    "elastic_net": 250,
}
STOCHASTIC_SEEDS = (11, 23, 47)
FOLD_IDS = ("dev_fold_1", "dev_fold_2", "dev_fold_3", "dev_fold_4")

FROZEN_COUNTS = {
    OOF_NAME: 172_969,
    FIT_RECEIPTS_NAME: 2_152,
    MONTHLY_NAME: 32_870,
    CONFIG_NAME: 346,
    FOLD_NAME: 1_384,
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BUILD_ID = re.compile(r"phase04-selector-results-[0-9a-f]{24}\Z")
_SAFE_COLUMN = re.compile(r"[a-z0-9_]{1,80}\Z")


class SelectorPublicationError(RuntimeError):
    """Raised when a Stage 9 search or publication fails closed."""


@dataclass(frozen=True)
class SearchBundleIdentity:
    search_id: str
    path: Path
    receipt: Mapping[str, Any]
    receipt_sha256: str
    file_hashes: Mapping[str, str]
    file_sizes: Mapping[str, int]


@dataclass(frozen=True)
class SelectorBuildIdentity:
    build_id: str
    content_fingerprint: str
    code_hashes: Mapping[str, str]
    configuration_hashes: Mapping[str, str]
    input_receipts_sha256: str
    search_receipt_sha256: str


@dataclass(frozen=True)
class FileReceipt:
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
class OutputPublication:
    path: Path
    bundle: PublishedBundle
    files: Mapping[str, FileReceipt]


@dataclass(frozen=True)
class PrimarySelectorPublication:
    build_id: str
    interim_path: Path
    processed_path: Path
    quality_path: Path
    manifest_path: Path
    pointer_path: Path
    manifest: Mapping[str, Any]
    pointer: Mapping[str, Any]
    reused_existing: bool


def _fail(message: str) -> None:
    raise SelectorPublicationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _schema_hash(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(value))


def published_schemas() -> dict[str, pa.Schema]:
    """Return every fixed Stage 9 publication schema.

    The OOF table is deliberately excluded: its 884 forecast/score fields are
    generated from the authenticated column registry in the search receipt.
    """

    return {
        FIT_RECEIPTS_NAME: pa.schema(
            [
                pa.field("model_family", pa.string(), nullable=False),
                pa.field("config_id", pa.string(), nullable=False),
                pa.field("fold_id", pa.string(), nullable=False),
                pa.field("seed", pa.int32()),
                pa.field("job_id", pa.string(), nullable=False),
                pa.field("job_status", pa.string(), nullable=False),
                pa.field("prediction_sha256", pa.string(), nullable=False),
                pa.field("prediction_rows", pa.int32(), nullable=False),
                pa.field("input_feature_count", pa.int32(), nullable=False),
                pa.field("input_rows", pa.int32(), nullable=False),
                pa.field("training_rows", pa.int32(), nullable=False),
                pa.field("inner_training_rows", pa.int32()),
                pa.field("inner_validation_rows", pa.int32()),
                pa.field("outer_validation_rows", pa.int32(), nullable=False),
                pa.field("parameter_count", pa.int64()),
                pa.field("best_tree_count", pa.int32()),
                pa.field("nonzero_coefficients", pa.int32()),
                pa.field("dual_gap", pa.float64()),
                pa.field("dual_gap_limit", pa.float64()),
                pa.field("kkt_max_violation", pa.float64()),
                pa.field("kkt_limit", pa.float64()),
                pa.field("iterations_or_epochs", pa.int32(), nullable=False),
                pa.field("elapsed_seconds", pa.float64(), nullable=False),
                pa.field("inner_best_iteration", pa.int32()),
                pa.field("refit_iteration_count", pa.int32()),
                pa.field("fit_converged", pa.bool_(), nullable=False),
                pa.field("fit_status", pa.string(), nullable=False),
            ]
        ),
        MONTHLY_NAME: pa.schema(
            [
                pa.field("model_family", pa.string(), nullable=False),
                pa.field("config_id", pa.string(), nullable=False),
                pa.field("fold_id", pa.string(), nullable=False),
                pa.field("signal_date", pa.date32(), nullable=False),
                pa.field("holding_month", pa.string(), nullable=False),
                pa.field("scored_rows", pa.int32(), nullable=False),
                pa.field("evaluation_rows", pa.int32(), nullable=False),
                pa.field("excluded_outcome_rows", pa.int32(), nullable=False),
                pa.field("rank_ic", pa.float64(), nullable=False),
                pa.field("rank_ic_status", pa.string(), nullable=False),
                pa.field("linear_ic", pa.float64(), nullable=False),
                pa.field("linear_ic_status", pa.string(), nullable=False),
            ]
        ),
        FOLD_NAME: pa.schema(
            [
                pa.field("model_family", pa.string(), nullable=False),
                pa.field("config_id", pa.string(), nullable=False),
                pa.field("fold_id", pa.string(), nullable=False),
                pa.field("months", pa.int16(), nullable=False),
                pa.field("mean_rank_ic", pa.float64(), nullable=False),
                pa.field("mean_linear_ic", pa.float64(), nullable=False),
            ]
        ),
        CONFIG_NAME: pa.schema(
            [
                pa.field("model_family", pa.string(), nullable=False),
                pa.field("config_id", pa.string(), nullable=False),
                pa.field("months", pa.int16(), nullable=False),
                pa.field("folds", pa.int8(), nullable=False),
                pa.field("mean_rank_ic", pa.float64(), nullable=False),
                pa.field("mean_linear_ic", pa.float64(), nullable=False),
                pa.field("rank_ic_bootstrap_se", pa.float64(), nullable=False),
                pa.field("linear_ic_bootstrap_se", pa.float64(), nullable=False),
                pa.field("positive_fold_count", pa.int8(), nullable=False),
                pa.field("fold_dispersion", pa.float64(), nullable=False),
                pa.field("gate_pass", pa.bool_(), nullable=False),
                pa.field("complexity", pa.float64(), nullable=False),
                pa.field("trial_phase", pa.string(), nullable=False),
                pa.field("parent_config_id", pa.string()),
                pa.field("configuration_json", pa.string(), nullable=False),
                pa.field("regularization_key_json", pa.string(), nullable=False),
            ]
        ),
        SHORTLIST_NAME: pa.schema(
            [
                pa.field("model_family", pa.string(), nullable=False),
                pa.field("config_id", pa.string(), nullable=False),
                pa.field("shortlist_order", pa.int8(), nullable=False),
                pa.field("selection_reasons", pa.string(), nullable=False),
                pa.field("benchmark_only_fallback", pa.bool_(), nullable=False),
                pa.field("gate_pass", pa.bool_(), nullable=False),
                pa.field("mean_rank_ic", pa.float64(), nullable=False),
                pa.field("rank_ic_bootstrap_se", pa.float64(), nullable=False),
                pa.field("positive_fold_count", pa.int8(), nullable=False),
                pa.field("fold_dispersion", pa.float64(), nullable=False),
                pa.field("complexity", pa.float64(), nullable=False),
                pa.field("family_best_mean_rank_ic", pa.float64(), nullable=False),
                pa.field("family_best_bootstrap_se", pa.float64(), nullable=False),
                pa.field("one_se_threshold", pa.float64(), nullable=False),
            ]
        ),
        CHECKS_NAME: pa.schema(
            [
                pa.field("check_name", pa.string(), nullable=False),
                pa.field("status", pa.string(), nullable=False),
                pa.field("observed_value", pa.string(), nullable=False),
                pa.field("expected_value", pa.string(), nullable=False),
                pa.field("detail", pa.string(), nullable=False),
            ]
        ),
    }


def raw_column_name(family: str, ordinal: int, seed: int | None) -> str:
    code = FAMILY_CODES[family]
    suffix = "" if seed is None else f"__s{seed:03d}"
    return f"raw__{code}__{ordinal:03d}{suffix}"


def score_column_name(family: str, ordinal: int) -> str:
    return f"score__{FAMILY_CODES[family]}__{ordinal:03d}"


def expected_oof_schema(receipt: Mapping[str, Any]) -> pa.Schema:
    registry = receipt.get("column_registry")
    _require(isinstance(registry, Mapping), "Search receipt omitted column_registry")
    raw = registry.get("raw_forecasts")
    scores = registry.get("monthly_scores")
    _require(isinstance(raw, list) and len(raw) == 538, "Raw forecast registry must contain 538 columns")
    _require(isinstance(scores, list) and len(scores) == 346, "Score registry must contain 346 columns")
    fields = [
        pa.field("fold_id", pa.string(), nullable=False),
        pa.field("signal_date", pa.date32(), nullable=False),
        pa.field("holding_month", pa.string(), nullable=False),
        pa.field("permno", pa.int64(), nullable=False),
    ]
    for entry in (*raw, *scores):
        _require(isinstance(entry, Mapping), "Column registry entry is not an object")
        name = str(entry.get("column_name", ""))
        _require(_SAFE_COLUMN.fullmatch(name) is not None, f"Unsafe OOF column name: {name!r}")
        fields.append(pa.field(name, pa.float32(), nullable=False))
    schema = pa.schema(fields)
    _validate_column_registry(receipt, schema.names[4:])
    return schema


def _validate_column_registry(receipt: Mapping[str, Any], actual_columns: Sequence[str]) -> None:
    registry = receipt["column_registry"]
    raw = list(registry["raw_forecasts"])
    scores = list(registry["monthly_scores"])
    observed = [str(item["column_name"]) for item in (*raw, *scores)]
    _require(observed == list(actual_columns), "OOF column order differs from its registry")
    _require(len(observed) == len(set(observed)), "OOF forecast column is duplicated")

    offset = 0
    score_offset = 0
    for family in FAMILY_ORDER:
        count = FAMILY_CONFIG_COUNTS[family]
        seeds: tuple[int | None, ...] = (
            STOCHASTIC_SEEDS if family != "elastic_net" else (None,)
        )
        family_raw = raw[offset : offset + count * len(seeds)]
        offset += len(family_raw)
        family_scores = scores[score_offset : score_offset + count]
        score_offset += count
        config_ids: list[str] = []
        for ordinal in range(count):
            score_entry = family_scores[ordinal]
            config_id = str(score_entry.get("config_id", ""))
            _require(config_id and config_id not in config_ids, f"Invalid {family} config registry")
            config_ids.append(config_id)
            _require(
                score_entry.get("model_family") == family
                and score_entry.get("column_name") == score_column_name(family, ordinal),
                f"{family} score registry order differs at {ordinal}",
            )
            for seed_index, seed in enumerate(seeds):
                raw_entry = family_raw[ordinal * len(seeds) + seed_index]
                _require(
                    raw_entry.get("model_family") == family
                    and str(raw_entry.get("config_id", "")) == config_id
                    and raw_entry.get("seed") == seed
                    and raw_entry.get("column_name") == raw_column_name(family, ordinal, seed),
                    f"{family} raw registry order differs at config {ordinal}",
                )
    _require(offset == 538 and score_offset == 346, "Column registry family totals differ")


def _validate_json_text(value: str, *, expect: type, label: str) -> Any:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SelectorPublicationError(f"Invalid {label} JSON") from exc
    _require(isinstance(decoded, expect), f"{label} JSON has wrong type")
    _require(
        json.dumps(decoded, sort_keys=True, separators=(",", ":"), allow_nan=False) == value,
        f"{label} JSON is not canonical",
    )
    return decoded


def _parquet_metadata(stream: BinaryIO) -> tuple[int, int, pa.Schema]:
    parquet = pq.ParquetFile(stream)
    return (
        int(parquet.metadata.num_rows),
        int(parquet.metadata.num_row_groups),
        parquet.schema_arrow,
    )


def _validate_compact_table(name: str, table: pa.Table) -> None:
    schema = published_schemas()[name]
    _require(table.schema.equals(schema, check_metadata=True), f"{name} schema differs")
    if name in FROZEN_COUNTS:
        _require(table.num_rows == FROZEN_COUNTS[name], f"{name} row count differs")
    nulls = sum(column.null_count for column in table.columns)
    _require(nulls == 0 or name in {FIT_RECEIPTS_NAME, CONFIG_NAME}, f"{name} has prohibited nulls")


def _table_null_count(table: pa.Table) -> int:
    """Return the total Arrow null count across every table column."""

    return sum(column.null_count for column in table.columns)


def _validate_configuration_ledger(config: pa.Table, receipt: Mapping[str, Any]) -> None:
    frame = config.to_pandas()
    _require(not frame.duplicated(["model_family", "config_id"]).any(), "Configuration key is duplicated")
    registry = receipt["column_registry"]["monthly_scores"]
    expected = [(str(item["model_family"]), str(item["config_id"])) for item in registry]
    observed = list(zip(frame["model_family"].astype(str), frame["config_id"].astype(str), strict=True))
    _require(observed == expected, "Configuration ledger order differs from OOF registry")
    for row in frame.itertuples(index=False):
        _validate_json_text(str(row.configuration_json), expect=dict, label="configuration")
        key = _validate_json_text(str(row.regularization_key_json), expect=list, label="regularization key")
        _require(len(key) > 0, "Regularization key may not be empty")
        _require(math.isfinite(float(row.complexity)) and float(row.complexity) >= 0.0, "Invalid complexity")
        _require(bool(row.gate_pass) == (float(row.mean_rank_ic) > 0.0 and int(row.positive_fold_count) >= 3), "Gate differs")
    phases = frame.groupby(["model_family", "trial_phase"], observed=True).size().to_dict()
    _require(phases.get(("neural_network", "full_grid")) == 48, "Neural trial ledger differs")
    _require(phases.get(("elastic_net", "full_grid")) == 250, "Elastic Net trial ledger differs")
    _require(phases.get(("xgboost", "core_grid")) == 40, "XGBoost core ledger differs")
    _require(phases.get(("xgboost", "gamma_challenger")) == 8, "XGBoost gamma ledger differs")
    gamma = frame.loc[(frame["model_family"] == "xgboost") & (frame["trial_phase"] == "gamma_challenger")]
    _require(gamma["parent_config_id"].notna().all() and gamma["parent_config_id"].nunique() == 4, "XGBoost gamma parents differ")
    non_gamma = frame.loc[frame["trial_phase"] != "gamma_challenger"]
    _require(non_gamma["parent_config_id"].isna().all(), "Only gamma challengers may have parents")


def _validate_fit_receipts(table: pa.Table, receipt: Mapping[str, Any]) -> None:
    frame = table.to_pandas()
    _require((frame["job_status"] == "COMPLETE").all(), "Incomplete fit receipt")
    _require(frame["fit_converged"].astype(bool).all(), "Authoritative fit did not converge")
    _require(frame["job_id"].is_unique, "Fit job_id is duplicated")
    _require(frame["prediction_sha256"].astype(str).map(lambda x: _SHA256.fullmatch(x) is not None).all(), "Malformed prediction hash")
    for column in (
        "prediction_rows",
        "input_feature_count",
        "input_rows",
        "training_rows",
        "outer_validation_rows",
        "iterations_or_epochs",
        "elapsed_seconds",
    ):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        _require(np.isfinite(values).all() and bool((values > 0.0).all()), f"Invalid fit receipt field: {column}")
    _require(
        (frame["prediction_rows"] == frame["outer_validation_rows"]).all(),
        "Fit prediction and outer-validation row counts differ",
    )
    neural = frame["model_family"].eq("neural_network")
    xgboost = frame["model_family"].eq("xgboost")
    elastic = frame["model_family"].eq("elastic_net")
    _require(frame.loc[neural, "parameter_count"].notna().all(), "Neural parameter counts are missing")
    _require(frame.loc[xgboost, "best_tree_count"].notna().all(), "XGBoost tree counts are missing")
    _require(
        frame.loc[
            elastic,
            [
                "nonzero_coefficients",
                "dual_gap",
                "dual_gap_limit",
                "kkt_max_violation",
                "kkt_limit",
            ],
        ]
        .notna()
        .all()
        .all(),
        "Elastic Net convergence/complexity evidence is missing",
    )
    dual = frame.loc[elastic, "dual_gap"].to_numpy(dtype=np.float64)
    dual_limit = frame.loc[elastic, "dual_gap_limit"].to_numpy(dtype=np.float64)
    violation = frame.loc[elastic, "kkt_max_violation"].to_numpy(dtype=np.float64)
    limit = frame.loc[elastic, "kkt_limit"].to_numpy(dtype=np.float64)
    _require(
        np.isfinite(dual).all()
        and np.isfinite(dual_limit).all()
        and np.isfinite(violation).all()
        and np.isfinite(limit).all()
        and bool((dual >= 0.0).all())
        and bool((dual_limit >= 0.0).all())
        and bool((dual <= dual_limit).all())
        and bool((violation >= 0.0).all())
        and bool((limit >= 0.0).all())
        and bool((violation <= limit).all()),
        "Elastic Net convergence certificate failed",
    )
    _require(
        frame.loc[neural | xgboost, ["inner_training_rows", "inner_validation_rows"]].notna().all().all(),
        "Early-stopped fit receipt omitted inner split sizes",
    )
    keys = frame[["model_family", "config_id", "fold_id", "seed"]].copy()
    _require(not keys.astype(str).duplicated().any(), "Fit identity is duplicated")
    config_by_family = {
        family: [
            str(item["config_id"])
            for item in receipt["column_registry"]["monthly_scores"]
            if item["model_family"] == family
        ]
        for family in FAMILY_ORDER
    }
    expected: set[tuple[str, str, str, int | None]] = set()
    for family, configs in config_by_family.items():
        seeds: tuple[int | None, ...] = STOCHASTIC_SEEDS if family != "elastic_net" else (None,)
        expected.update((family, config, fold, seed) for config in configs for fold in FOLD_IDS for seed in seeds)
    observed = {
        (
            str(row.model_family),
            str(row.config_id),
            str(row.fold_id),
            None if row.seed is None or (isinstance(row.seed, float) and math.isnan(row.seed)) else int(row.seed),
        )
        for row in frame.itertuples(index=False)
    }
    _require(observed == expected, "Fit-receipt grid is incomplete or out of scope")


def _validate_cross_tables(tables: Mapping[str, pa.Table], receipt: Mapping[str, Any]) -> None:
    config = tables[CONFIG_NAME]
    _validate_configuration_ledger(config, receipt)
    _validate_fit_receipts(tables[FIT_RECEIPTS_NAME], receipt)
    config_frame = config.to_pandas()
    fit_frame = tables[FIT_RECEIPTS_NAME].to_pandas()
    for row in config_frame.itertuples(index=False):
        fits = fit_frame.loc[
            fit_frame["model_family"].eq(row.model_family)
            & fit_frame["config_id"].eq(row.config_id)
        ]
        if row.model_family == "neural_network":
            counts = fits["parameter_count"].to_numpy(dtype=np.float64)
            observed_complexity = float(np.median(counts))
        elif row.model_family == "xgboost":
            configuration = json.loads(str(row.configuration_json))
            depth = int(configuration["max_depth"])
            observed_complexity = depth * float(
                np.median(fits["best_tree_count"].to_numpy(dtype=np.float64))
            )
        else:
            observed_complexity = float(
                np.median(
                    fits["nonzero_coefficients"].to_numpy(dtype=np.float64)
                )
            )
        _require(
            math.isclose(
                observed_complexity,
                float(row.complexity),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"Configuration complexity differs from fit receipts: {row.config_id}",
        )
    config_keys = set(zip(config["model_family"].to_pylist(), config["config_id"].to_pylist(), strict=True))
    for name in (MONTHLY_NAME, FOLD_NAME, SHORTLIST_NAME):
        table = tables[name]
        keys = set(zip(table["model_family"].to_pylist(), table["config_id"].to_pylist(), strict=True))
        _require(keys.issubset(config_keys), f"{name} references unknown configuration")
    monthly = tables[MONTHLY_NAME]
    _require(len(set(monthly["holding_month"].to_pylist())) == 95, "Monthly metric domain differs")
    _require(set(monthly["fold_id"].to_pylist()) == set(FOLD_IDS), "Monthly fold domain differs")
    _require(not monthly.to_pandas().duplicated(["model_family", "config_id", "holding_month"]).any(), "Monthly metric key is duplicated")
    fold = tables[FOLD_NAME]
    _require(not fold.to_pandas().duplicated(["model_family", "config_id", "fold_id"]).any(), "Fold metric key is duplicated")
    shortlist = tables[SHORTLIST_NAME]
    _require(shortlist.num_rows <= 9, "Shortlist exceeds three configurations per family")
    if shortlist.num_rows:
        _require(not shortlist.to_pandas().duplicated(["model_family", "config_id"]).any(), "Shortlist key is duplicated")
        counts = shortlist.to_pandas().groupby("model_family", observed=True).size()
        _require(bool((counts <= 3).all()), "Family shortlist exceeds three")


def _parse_search_receipt(payload: bytes) -> dict[str, Any]:
    try:
        receipt = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectorPublicationError("Search receipt is not valid JSON") from exc
    _require(isinstance(receipt, dict), "Search receipt is not an object")
    _require(
        canonical_json_bytes(receipt) == payload,
        "Search receipt is not canonical JSON with one trailing newline",
    )
    _require(receipt.get("schema_version") == 1 and receipt.get("status") == "COMPLETE", "Search receipt is not complete")
    _require(receipt.get("contract_id") == "phase04-selector-v1", "Search receipt contract differs")
    search_id = str(receipt.get("search_id", ""))
    _require(re.fullmatch(r"phase04-selector-search-[0-9a-f]{24}", search_id) is not None, "Search ID is malformed")
    files = receipt.get("files")
    _require(isinstance(files, Mapping) and set(files) == set(SEARCH_PAYLOAD_FILES), "Search receipt closed file set differs")
    for name, record in files.items():
        _require(isinstance(record, Mapping), f"Search receipt file record differs: {name}")
        _require(_SHA256.fullmatch(str(record.get("sha256", ""))) is not None, f"Malformed search hash: {name}")
        _require(int(record.get("size_bytes", -1)) >= 0, f"Malformed search size: {name}")
        _require(int(record.get("rows", -1)) >= 0, f"Malformed search rows: {name}")
        _require(int(record.get("row_groups", 0)) > 0, f"Malformed search row groups: {name}")
        _require(
            _SHA256.fullmatch(str(record.get("schema_sha256", "")))
            is not None,
            f"Malformed search schema hash: {name}",
        )
        if name in FROZEN_COUNTS:
            _require(int(record.get("rows", -1)) == FROZEN_COUNTS[name], f"Search row receipt differs: {name}")
        if name == SHORTLIST_NAME:
            _require(int(record.get("rows", -1)) <= 9, "Search shortlist row receipt exceeds nine")
    expected_oof_schema(receipt)
    return receipt


@contextlib.contextmanager
def open_completed_search_bundle(
    path: Path,
    *,
    database_root: Path,
) -> Iterator[tuple[SearchBundleIdentity, AuthenticatedBundleGuard]]:
    """Hold one closed search result directory against symlink/race attacks."""

    bundle_path = Path(os.path.abspath(path.expanduser()))
    root = Path(os.path.abspath(database_root.expanduser()))
    try:
        bundle_path.relative_to(root)
    except ValueError as exc:
        raise SelectorPublicationError("Search bundle escaped the database root") from exc
    receipt_path = bundle_path / SEARCH_RECEIPT_NAME
    with AuthenticatedPointerGuard.open(
        trusted_root=root,
        path=receipt_path,
        payload_validator=_parse_search_receipt,
        label="Stage 9 completed search receipt",
    ) as receipt_guard:
        receipt = _parse_search_receipt(receipt_guard.payload)
        hashes = {
            name: str(receipt["files"][name]["sha256"])
            for name in SEARCH_PAYLOAD_FILES
        }
        hashes[SEARCH_RECEIPT_NAME] = receipt_guard.payload_sha256
        sizes = {
            name: int(receipt["files"][name]["size_bytes"])
            for name in SEARCH_PAYLOAD_FILES
        }
        sizes[SEARCH_RECEIPT_NAME] = receipt_guard.record.size_bytes
        with AuthenticatedBundleGuard.open(
            trusted_root=root,
            path=bundle_path,
            expected_files=SEARCH_BUNDLE_FILES,
            expected_hashes=hashes,
            label="Stage 9 completed search bundle",
        ) as guard:
            receipt_guard.gate()
            identity = SearchBundleIdentity(
                search_id=str(receipt["search_id"]),
                path=bundle_path,
                receipt=receipt,
                receipt_sha256=receipt_guard.payload_sha256,
                file_hashes=hashes,
                file_sizes=sizes,
            )
            yield identity, guard
            guard.gate()
            receipt_guard.gate()


def _read_table(guard: AuthenticatedBundleGuard, name: str, columns: Sequence[str] | None = None) -> pa.Table:
    with guard.open_binary(name) as stream:
        return pq.read_table(stream, columns=None if columns is None else list(columns), use_threads=False).combine_chunks()


def validate_search_bundle(
    identity: SearchBundleIdentity,
    guard: AuthenticatedBundleGuard,
) -> tuple[dict[str, pa.Table], list[dict[str, str]]]:
    """Validate fixed schemas, shapes, ledgers, and the OOF key spine."""

    for name in SEARCH_PAYLOAD_FILES:
        with guard.open_binary(name) as stream:
            rows, row_groups, schema = _parquet_metadata(stream)
        record = identity.receipt["files"][name]
        _require(rows == int(record["rows"]), f"Search Parquet row receipt differs: {name}")
        _require(
            row_groups == int(record["row_groups"]),
            f"Search Parquet row-group receipt differs: {name}",
        )
        _require(
            _schema_hash(schema) == str(record["schema_sha256"]),
            f"Search Parquet schema receipt differs: {name}",
        )
    tables = {name: _read_table(guard, name) for name in SEARCH_PAYLOAD_FILES if name != OOF_NAME}
    for name, table in tables.items():
        _validate_compact_table(name, table)
    _validate_cross_tables(tables, identity.receipt)

    with guard.open_binary(OOF_NAME) as stream:
        rows, _groups, schema = _parquet_metadata(stream)
    expected = expected_oof_schema(identity.receipt)
    _require(rows == FROZEN_COUNTS[OOF_NAME], "OOF row count differs")
    _require(schema.equals(expected, check_metadata=True), "OOF schema or column order differs")
    oof_keys = _read_table(guard, OOF_NAME, ("fold_id", "signal_date", "holding_month", "permno"))
    _require(_table_null_count(oof_keys) == 0, "OOF key has nulls")
    key_frame = oof_keys.to_pandas()
    _require(not key_frame.duplicated().any(), "OOF key is duplicated")
    _require(set(key_frame["fold_id"]) == set(FOLD_IDS), "OOF fold domain differs")
    _require(len(set(key_frame["holding_month"])) == 95, "OOF month domain differs")

    checks = [
        {"check_name": "closed_search_file_set", "status": "PASS", "observed_value": "7", "expected_value": "7", "detail": "Receipt plus six Parquet payloads authenticated."},
        {"check_name": "oof_rows", "status": "PASS", "observed_value": str(rows), "expected_value": "172969", "detail": "Genuine development OOF scoring rows."},
        {"check_name": "oof_raw_columns", "status": "PASS", "observed_value": "538", "expected_value": "538", "detail": "Three seeds for NN/XGBoost and deterministic Elastic Net."},
        {"check_name": "oof_score_columns", "status": "PASS", "observed_value": "346", "expected_value": "346", "detail": "One seed-averaged monthly score per configuration."},
        {"check_name": "fit_receipts", "status": "PASS", "observed_value": str(tables[FIT_RECEIPTS_NAME].num_rows), "expected_value": "2152", "detail": "Complete fold/configuration/seed fit grid."},
        {"check_name": "configuration_trials", "status": "PASS", "observed_value": str(tables[CONFIG_NAME].num_rows), "expected_value": "346", "detail": "48 NN, 48 XGBoost, and 250 Elastic Net trials."},
        {"check_name": "monthly_metric_rows", "status": "PASS", "observed_value": str(tables[MONTHLY_NAME].num_rows), "expected_value": "32870", "detail": "346 configurations by 95 OOF months."},
        {"check_name": "fold_metric_rows", "status": "PASS", "observed_value": str(tables[FOLD_NAME].num_rows), "expected_value": "1384", "detail": "346 configurations by four folds."},
        {"check_name": "shortlist_cap", "status": "PASS", "observed_value": str(tables[SHORTLIST_NAME].num_rows), "expected_value": "<=9", "detail": "At most three distinct configurations per family."},
    ]
    return tables, checks


def validate_completed_search_tables(
    tables: Mapping[str, pa.Table],
    receipt: Mapping[str, Any],
) -> None:
    """Validate an assembled search payload before immutable publication.

    The completed-search writer uses this in-memory gate before any directory
    becomes authoritative.  Keeping the gate here guarantees that the writer
    and the later publisher apply exactly the same schemas, frozen counts,
    ledgers, and cross-table rules.
    """

    _require(
        set(tables) == set(SEARCH_PAYLOAD_FILES),
        "Completed search table set differs",
    )
    compact = {
        name: table
        for name, table in tables.items()
        if name != OOF_NAME
    }
    for name, table in compact.items():
        _validate_compact_table(name, table)
    _validate_cross_tables(compact, receipt)

    oof = tables[OOF_NAME]
    expected = expected_oof_schema(receipt)
    _require(
        oof.num_rows == FROZEN_COUNTS[OOF_NAME],
        "OOF row count differs",
    )
    _require(
        oof.schema.equals(expected, check_metadata=True),
        "OOF schema or column order differs",
    )
    keys = oof.select(("fold_id", "signal_date", "holding_month", "permno"))
    _require(_table_null_count(keys) == 0, "OOF key has nulls")
    frame = keys.to_pandas()
    _require(not frame.duplicated().any(), "OOF key is duplicated")
    _require(set(frame["fold_id"]) == set(FOLD_IDS), "OOF fold domain differs")
    _require(
        len(set(frame["holding_month"])) == 95,
        "OOF month domain differs",
    )


def compute_build_identity(
    search: SearchBundleIdentity,
    inputs: Stage9Inputs,
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> SelectorBuildIdentity:
    policy, _ = validate_contract(policy_path)
    code_paths = (
        "src/portfolio_bot/data/secure_publication.py",
        "src/portfolio_bot/selectors/contract.py",
        "src/portfolio_bot/selectors/inputs.py",
        "src/portfolio_bot/selectors/publication.py",
        "scripts/publish_stage_09_selectors.py",
    )
    config_paths = (
        str(Path(policy_path).resolve().relative_to(PROJECT_ROOT)),
        str(policy["physical_rules"]["path"]),
        str(policy["methodology"]["source"]),
        str(policy["methodology"]["freeze_receipt"]),
    )
    code_hashes = {path: sha256_file(PROJECT_ROOT / path) for path in code_paths}
    configuration_hashes = {path: sha256_file(project_path(path)) for path in config_paths}
    input_receipts_sha256 = _sha256_bytes(canonical_json_bytes(list(inputs.input_receipts)))
    fingerprint_payload = {
        "schema_version": 1,
        "purpose": "stage09_primary_selector_results_only",
        "contract_id": policy["contract_id"],
        "policy_id": policy["policy_id"],
        "search_id": search.search_id,
        "search_receipt_sha256": search.receipt_sha256,
        "search_file_hashes": dict(sorted(search.file_hashes.items())),
        "stage08_authority_build_id": inputs.authority["build_id"],
        "stage08_authority_verification_id": inputs.authority["verification_id"],
        "input_receipts_sha256": input_receipts_sha256,
        "code_hashes": code_hashes,
        "configuration_hashes": configuration_hashes,
    }
    fingerprint = _sha256_bytes(canonical_json_bytes(fingerprint_payload))
    return SelectorBuildIdentity(
        build_id=f"phase04-selector-results-{fingerprint[:24]}",
        content_fingerprint=fingerprint,
        code_hashes=code_hashes,
        configuration_hashes=configuration_hashes,
        input_receipts_sha256=input_receipts_sha256,
        search_receipt_sha256=search.receipt_sha256,
    )


def _write_bytes(stage: SecureStagingDirectory, name: str, payload: bytes) -> FileReceipt:
    with stage.open_exclusive_binary(name) as stream:
        written = stream.write(payload)
        _require(written == len(payload), f"Short publication write: {name}")
    return FileReceipt(name, _sha256_bytes(payload), len(payload), None, None, None)


def _copy_member(
    source: AuthenticatedBundleGuard,
    stage: SecureStagingDirectory,
    name: str,
    *,
    expected_hash: str,
    expected_size: int,
    expected_rows: int | None,
    expected_row_groups: int | None,
    expected_schema: pa.Schema | None,
) -> FileReceipt:
    digest = hashlib.sha256()
    size = 0
    with source.open_binary(name) as input_stream, stage.open_exclusive_binary(name) as output_stream:
        while block := input_stream.read(8 << 20):
            written = output_stream.write(block)
            _require(written == len(block), f"Short publication copy: {name}")
            digest.update(block)
            size += written
    observed = digest.hexdigest()
    _require(observed == expected_hash and size == expected_size, f"Copied payload differs: {name}")
    return FileReceipt(
        name,
        observed,
        size,
        expected_rows,
        expected_row_groups,
        None if expected_schema is None else _schema_hash(expected_schema),
    )


def _write_parquet(stage: SecureStagingDirectory, name: str, table: pa.Table) -> FileReceipt:
    expected = published_schemas()[name]
    _require(table.schema.equals(expected, check_metadata=True), f"Generated {name} schema differs")
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=1024, compression="zstd", use_dictionary=True, version="2.6", write_statistics=True)
    payload = sink.getvalue().to_pybytes()
    receipt = _write_bytes(stage, name, payload)
    return FileReceipt(name, receipt.sha256, receipt.size_bytes, table.num_rows, max(1, math.ceil(table.num_rows / 1024)), _schema_hash(expected))


def _success_payload(identity: SelectorBuildIdentity, location: str, receipts: Mapping[str, FileReceipt]) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "contract_id": "phase04-selector-v1",
            "build_id": identity.build_id,
            "content_fingerprint": identity.content_fingerprint,
            "location": location,
            "status": f"PRIMARY_{location.upper()}_COMPLETE",
            "files": {name: value.as_dict() for name, value in sorted(receipts.items())},
        }
    )


def _publish_payload_set(
    *,
    trusted_root: Path,
    destination: Path,
    expected_files: Sequence[str],
    copy_names: Sequence[str],
    generated_tables: Mapping[str, pa.Table],
    generated_json: Mapping[str, Mapping[str, Any]],
    source: AuthenticatedBundleGuard,
    search: SearchBundleIdentity,
    identity: SelectorBuildIdentity,
    location: str,
) -> OutputPublication:
    receipts: dict[str, FileReceipt] = {}
    with SecureStagingDirectory.create(trusted_root=trusted_root, destination=destination, expected_files=expected_files, label=f"Stage 9 selector {location}") as stage:
        for name in copy_names:
            expected_rows = (
                int(search.receipt["files"][name]["rows"])
                if name in SEARCH_PAYLOAD_FILES
                else None
            )
            expected_row_groups = (
                int(search.receipt["files"][name]["row_groups"])
                if name in SEARCH_PAYLOAD_FILES
                else None
            )
            expected_schema = expected_oof_schema(search.receipt) if name == OOF_NAME else published_schemas().get(name)
            if expected_schema is not None and name in SEARCH_PAYLOAD_FILES:
                _require(
                    search.receipt["files"][name]["schema_sha256"]
                    == _schema_hash(expected_schema),
                    f"Search schema receipt differs: {name}",
                )
            receipts[name] = _copy_member(
                source,
                stage,
                name,
                expected_hash=search.file_hashes[name],
                expected_size=search.file_sizes[name],
                expected_rows=expected_rows,
                expected_row_groups=expected_row_groups,
                expected_schema=expected_schema,
            )
        for name, table in generated_tables.items():
            receipts[name] = _write_parquet(stage, name, table)
        for name, value in generated_json.items():
            receipts[name] = _write_bytes(stage, name, _json_bytes(value))
        success = _write_bytes(stage, SUCCESS_NAME, _success_payload(identity, location, receipts))
        hashes = {SUCCESS_NAME: success.sha256, **{name: value.sha256 for name, value in receipts.items()}}
        snapshot = stage.snapshot(expected_hashes=hashes)
        bundle = stage.publish(expected_hashes=snapshot.hashes, pre_publish_gate=source.gate, post_publish_gate=source.gate)
    return OutputPublication(destination, bundle, receipts)


def _manifest_output(output: OutputPublication, database_root: Path) -> dict[str, Any]:
    return {
        "path": output.path.relative_to(database_root).as_posix(),
        "success_sha256": output.bundle.snapshot.hashes[SUCCESS_NAME],
        "files": {name: receipt.as_dict() for name, receipt in sorted(output.files.items())},
    }


def _publish_manifest(
    destination: Path,
    manifest: Mapping[str, Any],
    *,
    identity: SelectorBuildIdentity,
    gate: Any,
) -> PublishedBundle:
    manifest_payload = canonical_json_bytes(manifest)
    success_payload = canonical_json_bytes(
        {
            "schema_version": 1,
            "contract_id": "phase04-selector-v1",
            "build_id": identity.build_id,
            "status": "PRIMARY_MANIFEST_COMPLETE",
            "manifest_sha256": _sha256_bytes(manifest_payload),
        }
    )
    with SecureStagingDirectory.create(trusted_root=PROJECT_ROOT, destination=destination, expected_files=MANIFEST_FILES, label="Stage 9 primary selector manifest") as stage:
        manifest_receipt = _write_bytes(stage, MANIFEST_NAME, manifest_payload)
        success_receipt = _write_bytes(stage, SUCCESS_NAME, success_payload)
        snapshot = stage.snapshot(expected_hashes={MANIFEST_NAME: manifest_receipt.sha256, SUCCESS_NAME: success_receipt.sha256})
        return stage.publish(expected_hashes=snapshot.hashes, pre_publish_gate=gate, post_publish_gate=gate)


def publish_primary_selector_results(
    search_bundle_path: Path,
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> PrimarySelectorPublication:
    """Publish an authenticated completed search and pointer-last primary gate."""

    policy, _ = validate_contract(policy_path)
    inputs = resolve_stage9_inputs(policy_path)
    database_root = inputs.database_root
    with open_completed_search_bundle(search_bundle_path, database_root=database_root) as (search, guard):
        tables, check_rows = validate_search_bundle(search, guard)
        identity = compute_build_identity(search, inputs, policy_path=policy_path)
        _require(_BUILD_ID.fullmatch(identity.build_id) is not None, "Selector build ID is malformed")
        summary = {
            "schema_version": 1,
            "build_id": identity.build_id,
            "status": "PRIMARY_CHECKS_PASSED",
            "hard_gates": {row["check_name"]: True for row in check_rows},
            "scope": {
                "development_only": True,
                "validation_or_final_rows_read": 0,
                "portfolio_weights_created": 0,
                "pnl_or_backtests_created": 0,
            },
        }
        checks = pa.Table.from_pylist(check_rows, schema=published_schemas()[CHECKS_NAME])
        interim_path = database_root / policy["database"]["interim_build_root"] / identity.build_id
        processed_path = database_root / policy["database"]["processed_build_root"] / identity.build_id
        quality_path = database_root / policy["database"]["quality_build_root"] / identity.build_id
        interim = _publish_payload_set(
            trusted_root=database_root,
            destination=interim_path,
            expected_files=INTERIM_FILES,
            copy_names=(OOF_NAME, FIT_RECEIPTS_NAME),
            generated_tables={},
            generated_json={},
            source=guard,
            search=search,
            identity=identity,
            location="interim",
        )
        processed = _publish_payload_set(
            trusted_root=database_root,
            destination=processed_path,
            expected_files=PROCESSED_FILES,
            copy_names=(MONTHLY_NAME, CONFIG_NAME, FOLD_NAME, SHORTLIST_NAME, SEARCH_RECEIPT_NAME),
            generated_tables={},
            generated_json={},
            source=guard,
            search=search,
            identity=identity,
            location="processed",
        )
        quality = _publish_payload_set(
            trusted_root=database_root,
            destination=quality_path,
            expected_files=QUALITY_FILES,
            copy_names=(),
            generated_tables={CHECKS_NAME: checks},
            generated_json={SUMMARY_NAME: summary},
            source=guard,
            search=search,
            identity=identity,
            location="quality",
        )
        manifest = {
            "schema_version": 1,
            "contract_id": policy["contract_id"],
            "policy_id": policy["policy_id"],
            "build_id": identity.build_id,
            "status": "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION",
            "authorizations": {
                "independent_verification_authorized": True,
                "independent_verification_completed": False,
                "stage_10_allocation_architectures_authorized": False,
                "stage_11_or_later_authorized": False,
                "portfolios_pnl_or_backtests_authorized": False,
                "validation_or_final_test_access_authorized": False,
            },
            "bindings": {
                "content_fingerprint": identity.content_fingerprint,
                "code_hashes": dict(identity.code_hashes),
                "configuration_hashes": dict(identity.configuration_hashes),
                "input_receipts_sha256": identity.input_receipts_sha256,
                "search_receipt_sha256": identity.search_receipt_sha256,
                "search_file_hashes": dict(sorted(search.file_hashes.items())),
            },
            "inputs": {
                "stage08_authority_build_id": inputs.authority["build_id"],
                "stage08_authority_verification_id": inputs.authority["verification_id"],
                "table_receipts": list(inputs.input_receipts),
                "search_id": search.search_id,
                "validation_or_final_rows_read": 0,
            },
            "outputs": {
                "interim": _manifest_output(interim, database_root),
                "processed": _manifest_output(processed, database_root),
                "quality": _manifest_output(quality, database_root),
            },
            "summary": {
                "oof_rows": FROZEN_COUNTS[OOF_NAME],
                "present_oof_outcomes": 172_799,
                "configurations": 346,
                "shortlisted_configurations": tables[SHORTLIST_NAME].num_rows,
            },
            "hard_gates": summary["hard_gates"],
        }

        def reauthenticate() -> None:
            guard.gate()
            current = resolve_stage9_inputs(policy_path)
            _require(current.input_receipts == inputs.input_receipts, "Stage 9 inputs changed")
            _require(current.authority == inputs.authority, "Stage 9 authority changed")
            _require(compute_build_identity(search, current, policy_path=policy_path) == identity, "Stage 9 build identity changed")

        artifact_root = project_path(policy["publication"]["artifact_root"])
        manifest_directory = artifact_root / policy["publication"]["primary_build_namespace"] / identity.build_id
        artifact_bundle = _publish_manifest(manifest_directory, manifest, identity=identity, gate=reauthenticate)
        manifest_sha = artifact_bundle.snapshot.hashes[MANIFEST_NAME]
        pointer = {
            "schema_version": 1,
            "contract_id": policy["contract_id"],
            "policy_id": policy["policy_id"],
            "build_id": identity.build_id,
            "status": "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION",
            "primary_manifest_sha256": manifest_sha,
            "authorizations": manifest["authorizations"],
        }
        pointer_path = project_path(policy["publication"]["latest_primary_pointer"])
        replaced = replace_pointer_last(
            pointer_path,
            canonical_json_bytes(pointer),
            trusted_root=PROJECT_ROOT,
            pre_replace_gate=reauthenticate,
            post_replace_gate=reauthenticate,
            label="Stage 9 latest selector primary pointer",
        )
        reused = interim.bundle.reused_existing and processed.bundle.reused_existing and quality.bundle.reused_existing and artifact_bundle.reused_existing and not replaced
        return PrimarySelectorPublication(
            build_id=identity.build_id,
            interim_path=interim_path,
            processed_path=processed_path,
            quality_path=quality_path,
            manifest_path=manifest_directory / MANIFEST_NAME,
            pointer_path=pointer_path,
            manifest=manifest,
            pointer=pointer,
            reused_existing=reused,
        )


__all__ = [
    "CHECKS_NAME",
    "CONFIG_NAME",
    "FAMILY_CODES",
    "FAMILY_CONFIG_COUNTS",
    "FAMILY_ORDER",
    "FIT_RECEIPTS_NAME",
    "FOLD_IDS",
    "FOLD_NAME",
    "INTERIM_FILES",
    "MANIFEST_FILES",
    "MONTHLY_NAME",
    "OOF_NAME",
    "PROCESSED_FILES",
    "PrimarySelectorPublication",
    "QUALITY_FILES",
    "SEARCH_BUNDLE_FILES",
    "SEARCH_PAYLOAD_FILES",
    "SEARCH_RECEIPT_NAME",
    "SHORTLIST_NAME",
    "STOCHASTIC_SEEDS",
    "SUMMARY_NAME",
    "SearchBundleIdentity",
    "SelectorBuildIdentity",
    "SelectorPublicationError",
    "compute_build_identity",
    "expected_oof_schema",
    "open_completed_search_bundle",
    "publish_primary_selector_results",
    "published_schemas",
    "raw_column_name",
    "score_column_name",
    "validate_search_bundle",
    "validate_completed_search_tables",
]
