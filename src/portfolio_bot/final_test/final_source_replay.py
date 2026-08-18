"""Checkpointed repository-native formula replay for final-period sources.

This module is imported only after the Stage 12 binding, the separate Stage 13
approval, and the Phase 1 raw snapshot have all been authenticated by
``final_source_builder``.  It has no public path/date/provider arguments.  The
only production callable uses repository-fixed paths and the exact final
calendar.

The replay publishes source roles in decision-then-outcome order.  Predictor,
risk/cost, and entry-market source products are committed first; realized
accounting, factor, benchmark, label, and outcome products are published only
after a durable source-table barrier proves that every decision-side table is
present.  This is a publication-order barrier, not a claim that final-period
return payloads are globally unopened: later predictor and risk rows correctly
depend on returns available before their own signal dates.  The production
runtime enforces the stronger per-month decision-before-accounting chronology.
Each table is stored as one content-addressed Parquet fragment per holding
month, so a retry reauthenticates and reuses completed months.

The empirical adapter at the bottom of this file reuses the frozen Stage 2,
Stage 6, Stage 7, Stage 8, and Stage 11 implementations.  Its process-local
scope changes only the authenticated date/cardinality domain; every economic
formula remains the frozen implementation.  Scope mutations are serialized,
restored in ``finally`` blocks, and included in the replay identity.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.final_test.calendar import (
    FINAL_COMPLETE_HOLDING_MONTHS,
    FINAL_HOLDING_MONTHS,
    FROZEN_HOLDING_INTERVAL_ANCHORS,
    PARTIAL_DECEMBER_HOLDING_MONTH,
    final_holding_month_descriptor,
    require_frozen_holding_interval_anchor,
)
from portfolio_bot.final_test.final_input_construction import (
    PROJECT_ROOT,
    _SOURCE_MONTHS,
    _SOURCE_NAMES,
    _source_schemas,
)
from portfolio_bot.final_test.input_materialization import _configured_database_root
from portfolio_bot.risk_cost.cost_math import PRIMARY_BORROW_RATE


REPLAY_ID = "stage13-repository-native-final-replay-v1"
CHECKPOINT_ID = "stage13-final-month-checkpoint-v1"
DECISION_BARRIER_ID = "stage13-final-decision-source-barrier-v1"
WORK_NAMESPACE = Path("cache/phase_06_final_test/stage13-final-source-replay-v1")
SOURCE_CONTEXT_MIGRATION_CONFIG = Path(
    "configs/phase_06_final_test/"
    "stage_13_source_checkpoint_context_migration_v2.json"
)
EXPECTED_SOURCE_CONTEXT_MIGRATION_CONFIG_SHA256 = (
    "d504e2dc75c58645b458d255db4f140bd0c9d8a9fa20022b021b4098f586613e"
)
SOURCE_CONTEXT_MIGRATION_RECEIPT = (
    "stage13_source_checkpoint_context_migration_receipt.json"
)
DECISION_CHECKPOINT_REBIND_CONFIG = Path(
    "configs/phase_06_final_test/stage_13_decision_checkpoint_rebind_v2.json"
)
EXPECTED_DECISION_CHECKPOINT_REBIND_CONFIG_SHA256 = (
    "1453e467c622913660a9097fc260c5632bbd963474f5b99c167478212ee70908"
)

STAGE11_FEATURE_RECEIPT = Path(
    "artifacts/phase_05_validation/stage_11_validation_data_v1/"
    "empirical_candidate_build_receipt.json"
)
STAGE11_CANDIDATE_MANIFEST_GLOB = (
    "artifacts/phase_05_validation/stage_11_validation_data_v1/"
    "candidate_manifest_*.json"
)
STAGE11_FEATURE_INPUT_ROOT = Path(
    "interim/phase_05_validation/stage11-validation-data-v1/" "feature_input_bundle"
)
STAGE11_FULL_CONTEXT_ROOT = Path(
    "interim/phase_05_validation/stage11-validation-data-v1/" "full_predictor_context"
)
FROZEN_STAGE11_FEATURE_INPUT_ADAPTER_ID = "stage11-verified-feature-input-adapter-v1"
FROZEN_STAGE11_FEATURE_INPUT_ADAPTER_CODE_SHA256 = (
    "f02c0429a126a6a576197d2175030bc8a0018969ba01732b254b4a8318cfff31"
)
FROZEN_STAGE11_FEATURE_INPUT_RECEIPT_SHA256 = (
    "84598a2fb4841331b377399eb414844c2896f3b8d71ede3f3588997ebf446b0e"
)
FROZEN_STAGE11_FEATURE_INPUT_BUNDLE_SHA256 = (
    "4684230ea43d804c272e92f438784db15a279c0529af987ef5b5276a583592ae"
)
CURRENT_FEATURE_INPUT_ADAPTER_ID = "stage11-verified-feature-input-adapter-v3"

FINAL_EXTENSION_START = dt.date(2019, 11, 1)
FINAL_CONTEXT_SIGNAL = dt.date(2019, 11, 29)
FINAL_FIRST_SIGNAL = dt.date(2019, 12, 31)
FINAL_LAST_SIGNAL = dt.date(2025, 11, 28)
FINAL_PARTIAL_ENTRY_DATE = dt.date(2025, 12, 1)
FINAL_LAST_EXIT = dt.date(2025, 12, 31)
FIRST_POST_FINAL_DATE = dt.date(2026, 1, 1)

DECISION_TABLES = (
    "final_calendar",
    "final_predictors",
    "final_risk_cost",
    "final_factor_covariance",
    "final_risk_cost_month_summary",
    "final_market_inputs",
)
OUTCOME_TABLES = (
    "final_market_accounting",
    "final_factors",
    "final_benchmarks",
    "final_completed_labels",
    "final_outcomes",
    "partial_descriptive_outcome",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REPLAY_LOCK = threading.RLock()
_SYNTHETIC_ENGINE_TOKEN = object()
_ACCOUNTING_NAN_TRANSPORT_FIELDS = (
    "entry_close_usd_per_share",
    "full_spread_20d",
    "adv20_usd",
    "impact_volatility_daily",
)
_PRIMARY_ACCOUNTING_BORROW_RATE_ANNUAL = PRIMARY_BORROW_RATE.annual

_LAST_TRADABLE_CLOSE_POLICY_PATH = PROJECT_ROOT / (
    "configs/phase_06_final_test/"
    "stage_13_public_interest_last_tradable_close_liquidation_v1.json"
)
_LAST_TRADABLE_CLOSE_HOLDING_MONTHS = frozenset({"2021-01", "2021-03"})

_CROSS_SCOPE_TERMINAL_STITCH_ID = (
    "stage13-2025-11-permno-91416-cross-scope-terminal-stitch-v1"
)
_CROSS_SCOPE_TERMINAL_STITCH_MONTH = "2025-11"
_CROSS_SCOPE_TERMINAL_STITCH_PERMNO = 91_416
_CROSS_SCOPE_TERMINAL_STITCH_DATE = dt.date(2025, 11, 28)
_CROSS_SCOPE_TERMINAL_POSTING_DATE = dt.date(2025, 12, 1)
_CROSS_SCOPE_RAW_DELISTING_SHA256 = (
    "bdee1560043d059c7b75518a31a4637398234168a087d9f37ba1fcb8fc90d4cb"
)
_CROSS_SCOPE_RAW_DELISTING_ROW_SHA256 = (
    "b00ea07c1c22ac5e4ca174d8af9551c32354a70aa9e7428328de41bf890bf6b4"
)
_CROSS_SCOPE_DECISION_RETURN_ROW_SHA256 = (
    "90139b616a70bb8bd88b86d73b9bd28619103ac9a2ec416b85341417cb85da7f"
)
_CROSS_SCOPE_OUTCOME_AUDIT_ROW_SHA256 = (
    "041c7a9623b1e2ccb8e1f61709746a46c12f9d6b48cd238cba6dd474e0e80d0b"
)
_CROSS_SCOPE_STITCHED_RETURN_ROW_SHA256 = (
    "68e7045975bb4b4ea64f8680bf5ba35d268552789e0fad3e6e2b1b2e8bea0187"
)
_CROSS_SCOPE_STITCHED_ACCOUNTING_PAYLOAD_SHA256 = (
    "963ea6252b35c21bcde8b877cbd8f9b7788a009ff1e58a45eb33e371e89be3bf"
)
_CROSS_SCOPE_OUTCOME_ACCOUNTING_PAYLOAD_SHA256 = (
    "0c552d1d5017cc06d71407f9eafc848229628cafa3dee4a568ac162afcb81acb"
)
_CROSS_SCOPE_DELISTING_EVENT_ROW_SHA256 = (
    "3681d9525e0391a72427cc611d005d70ad01ccbc1d14698b33af97cf0a8fa20a"
)
_CROSS_SCOPE_CORPORATE_ACTION_ID = "stage11-action-2862ec25c18db28f324099b563a34c3a"
_CROSS_SCOPE_RAW_DELISTING_COLUMNS = (
    "PERMNO",
    "DelistingDt",
    "DelDtPrc",
    "DelDtPrcFlg",
    "DelActionType",
    "DelStatusType",
    "DelReasonType",
    "DelPaymentType",
    "DelPERMNO",
    "DelPERMCO",
    "DelRet",
    "DelRetMissType",
    "DelNextDt",
    "DelNextPrc",
    "DelNextPrcFlg",
    "DelAmtDt",
    "DelDivAmt",
    "DelDisType",
    "DelDlyDt",
    "PrimaryExch",
    "SICCD",
    "NASDIssuno",
)
_CROSS_SCOPE_DELISTING_EVENT_COLUMNS = (
    "permno",
    "delisting_date",
    "posting_date",
    "delisting_price",
    "delisting_price_flag",
    "action_type",
    "status_type",
    "reason_type",
    "payment_type",
    "successor_permno",
    "successor_permco",
    "observed_delisting_return",
    "return_missing_type",
    "next_date",
    "next_price",
    "next_price_flag",
    "amount_date",
    "delisting_dividend_amount",
    "distribution_type",
    "primary_exchange",
    "performance_related",
    "embedding_status",
    "treatment_status",
    "effective_delisting_return",
    "imputation_applied",
    "assigned_return_date",
    "outcome_availability_date",
    "assigned_exactly_once",
)


class FinalSourceReplayError(RuntimeError):
    """Raised when final formula replay/checkpoint evidence differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalSourceReplayError(message)


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
        raise FinalSourceReplayError(
            "final replay metadata is not finite canonical JSON"
        ) from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _table_fingerprint(table: pa.Table) -> str:
    source = table.combine_chunks()
    digest = hashlib.sha256(source.schema.serialize().to_pybytes())
    for batch in source.to_batches(max_chunksize=131_072):
        digest.update(batch.serialize().to_pybytes())
    return digest.hexdigest()


def _tables_logically_equal_with_nan_transport(
    left: pa.Table,
    right: pa.Table,
) -> bool:
    """Compare exact Arrow values without hashing undefined null-buffer bytes."""

    first = left.combine_chunks()
    second = right.combine_chunks()
    if (
        first.num_rows != second.num_rows
        or first.num_columns != second.num_columns
        or not first.schema.equals(second.schema, check_metadata=True)
    ):
        return False
    for field in first.schema:
        left_array = first[field.name].combine_chunks()
        right_array = second[field.name].combine_chunks()
        if not pa.types.is_floating(field.type):
            if not left_array.equals(right_array):
                return False
            continue
        left_null = pc.is_null(left_array).to_numpy(zero_copy_only=False)
        right_null = pc.is_null(right_array).to_numpy(zero_copy_only=False)
        if not np.array_equal(left_null, right_null):
            return False
        left_values = np.asarray(
            pc.fill_null(
                left_array,
                pa.scalar(0.0, type=field.type),
            ).to_numpy(zero_copy_only=False)
        )
        right_values = np.asarray(
            pc.fill_null(
                right_array,
                pa.scalar(0.0, type=field.type),
            ).to_numpy(zero_copy_only=False)
        )
        left_nan = np.isnan(left_values)
        right_nan = np.isnan(right_values)
        if not np.array_equal(left_nan, right_nan):
            return False
        unsigned_type = {
            16: np.uint16,
            32: np.uint32,
            64: np.uint64,
        }.get(field.type.bit_width)
        if unsigned_type is None:  # pragma: no cover - Arrow float widths are closed
            return False
        left_bits = left_values.view(unsigned_type).copy()
        right_bits = right_values.view(unsigned_type).copy()
        left_bits[left_nan] = 0
        right_bits[right_nan] = 0
        if not np.array_equal(left_bits, right_bits):
            return False
    return True


def _ordinary_directory(path: Path, *, root: Path, label: str) -> Path:
    trusted = Path(os.path.abspath(root.expanduser()))
    selected = Path(os.path.abspath(path.expanduser()))
    try:
        relative = selected.relative_to(trusted)
    except ValueError as exc:
        raise FinalSourceReplayError(f"{label} escaped its root") from exc
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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.partial-{os.getpid()}-{hashlib.sha256(payload).hexdigest()[:12]}"
    )
    _require(
        not temporary.exists() and not temporary.is_symlink(),
        "final replay temporary path already exists",
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalSourceReplayError(f"{label} is absent or invalid") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _native_schema(name: str, project_root: Path = PROJECT_ROOT) -> pa.Schema:
    target = _source_schemas(project_root)[name]
    return pa.schema(
        [
            pa.field("source_holding_month", pa.string(), nullable=False),
            *list(target)[1:],
        ],
        metadata=target.metadata,
    )


def _month_column(table: pa.Table) -> pa.ChunkedArray:
    if "source_holding_month" in table.column_names:
        return table["source_holding_month"]
    if "holding_month" in table.column_names:
        return table["holding_month"]
    raise FinalSourceReplayError("formula table lacks a holding-month column")


def _as_native_table(name: str, table: pa.Table) -> pa.Table:
    _require(isinstance(table, pa.Table), f"{name} is not an Arrow table")
    source = table.combine_chunks()
    expected = _native_schema(name)
    if "holding_month" in source.column_names:
        source = source.rename_columns(expected.names)
    _require(
        source.schema.equals(expected, check_metadata=True),
        f"{name} native formula schema differs",
    )
    return source


def _require_no_nonnullable_nulls(table: pa.Table, *, label: str) -> None:
    """Audit every non-nullable column before any Parquet write is attempted."""

    violations = {
        field.name: table[field.name].null_count
        for field in table.schema
        if not field.nullable and table[field.name].null_count
    }
    _require(
        not violations,
        f"{label} non-nullable pre-write audit failed: "
        + ", ".join(f"{name}={count}" for name, count in sorted(violations.items())),
    )


def _filter_month(table: pa.Table, month: str) -> pa.Table:
    values = _month_column(table)
    return table.filter(
        pc.equal(values, pa.scalar(month, pa.string()))
    ).combine_chunks()


def _expected_months(name: str) -> tuple[str, ...]:
    return tuple(_SOURCE_MONTHS[name])


@dataclass(frozen=True, slots=True)
class FinalReplayContext:
    """Canonical identity selecting one immutable checkpoint namespace."""

    raw_snapshot_id: str
    raw_file_identities_sha256: str
    stage12_binding_sha256: str
    approval_payload_sha256: str

    def __post_init__(self) -> None:
        _require(bool(self.raw_snapshot_id), "raw snapshot ID is absent")
        for label, value in (
            ("raw file identity", self.raw_file_identities_sha256),
            ("Stage 12 binding", self.stage12_binding_sha256),
            ("Stage 13 approval", self.approval_payload_sha256),
        ):
            _require(
                _HEX64.fullmatch(value) is not None, f"{label} SHA-256 is malformed"
            )

    def mapping(self) -> dict[str, object]:
        return {
            "replay_id": REPLAY_ID,
            "raw_snapshot_id": self.raw_snapshot_id,
            "raw_file_identities_sha256": self.raw_file_identities_sha256,
            "stage12_binding_sha256": self.stage12_binding_sha256,
            "approval_payload_sha256": self.approval_payload_sha256,
            "holding_months": list(FINAL_HOLDING_MONTHS),
            "complete_holding_months": list(FINAL_COMPLETE_HOLDING_MONTHS),
            "partial_holding_month": PARTIAL_DECEMBER_HOLDING_MONTH,
        }

    @property
    def context_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.mapping()))


@dataclass(frozen=True, slots=True)
class MonthCheckpointReceipt:
    table_name: str
    holding_month: str
    rows: int
    parquet_sha256: str
    logical_content_sha256: str
    arrow_schema_sha256: str
    relative_path: str

    def mapping(self) -> dict[str, object]:
        return dataclasses.asdict(self)


class FinalReplayCheckpointStore:
    """Content-addressed, per-table/per-month memoization ledger."""

    def __init__(
        self,
        root: Path,
        *,
        context: FinalReplayContext,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self.project_root = project_root.absolute()
        self.context = context
        self.namespace_root = _ordinary_directory(
            root,
            root=root.parent,
            label="final replay checkpoint root",
        )
        self.run_root = self.namespace_root / context.context_sha256
        self.fragments_root = self.run_root / "months"
        self.receipts_root = self.run_root / "receipts"
        self.context_path = self.run_root / "context.json"

    def initialize(self) -> None:
        self.fragments_root.mkdir(parents=True, exist_ok=True)
        self.receipts_root.mkdir(parents=True, exist_ok=True)
        expected = {
            **self.context.mapping(),
            "context_sha256": self.context.context_sha256,
        }
        if self.context_path.exists():
            observed = _json(self.context_path, label="final replay context")
            _require(observed == expected, "final replay context changed")
            _require(
                self.context_path.read_bytes() == _canonical_json_bytes(observed),
                "final replay context is not canonical JSON",
            )
        else:
            _atomic_bytes(self.context_path, _canonical_json_bytes(expected))

    def _paths(self, name: str, month: str) -> tuple[Path, Path]:
        _require(name in _SOURCE_NAMES, "unknown final replay table")
        _require(
            month in _expected_months(name), f"{name} holding month is outside scope"
        )
        payload = self.fragments_root / name / f"{month}.parquet"
        receipt = self.receipts_root / name / f"{month}.json"
        return payload, receipt

    def _inspect(self, name: str, month: str) -> MonthCheckpointReceipt | None:
        payload, receipt_path = self._paths(name, month)
        if not receipt_path.exists():
            _require(
                not payload.exists(),
                f"unreceipted final replay fragment exists: {name}/{month}",
            )
            return None
        _require(
            payload.is_file() and not payload.is_symlink(),
            "checkpoint payload is absent",
        )
        value = _json(receipt_path, label=f"checkpoint receipt {name}/{month}")
        _require(
            receipt_path.read_bytes() == _canonical_json_bytes(value),
            f"checkpoint receipt is not canonical: {name}/{month}",
        )
        expected_keys = {
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
        _require(set(value) == expected_keys, "checkpoint receipt fields differ")
        table = pq.read_table(payload, use_threads=False).combine_chunks()
        table = _as_native_table(name, table)
        _require(
            table.num_rows > 0 and set(_month_column(table).to_pylist()) == {month},
            f"checkpoint month domain differs: {name}/{month}",
        )
        observed = MonthCheckpointReceipt(
            table_name=name,
            holding_month=month,
            rows=table.num_rows,
            parquet_sha256=_sha256_file(payload),
            logical_content_sha256=_table_fingerprint(table),
            arrow_schema_sha256=_sha256_bytes(table.schema.serialize().to_pybytes()),
            relative_path=payload.relative_to(self.run_root).as_posix(),
        )
        _require(
            value["schema_version"] == 1
            and value["status"] == "FINAL_FORMULA_MONTH_CHECKPOINT_COMPLETE"
            and value["checkpoint_id"] == CHECKPOINT_ID
            and value["context_sha256"] == self.context.context_sha256
            and all(
                value[key] == observed.mapping()[key] for key in observed.mapping()
            ),
            f"checkpoint content identity differs: {name}/{month}",
        )
        return observed

    def completed_months(self, name: str) -> tuple[str, ...]:
        self.initialize()
        return tuple(
            month for month in _expected_months(name) if self._inspect(name, month)
        )

    def pending_months(self, name: str) -> tuple[str, ...]:
        completed = frozenset(self.completed_months(name))
        return tuple(
            month for month in _expected_months(name) if month not in completed
        )

    def commit_table(
        self, name: str, table: pa.Table
    ) -> tuple[MonthCheckpointReceipt, ...]:
        """Commit only absent months; authenticated completed months are immutable."""

        self.initialize()
        source = _as_native_table(name, table)
        _require_no_nonnullable_nulls(source, label=name)
        observed_months = tuple(sorted(set(_month_column(source).to_pylist())))
        _require(bool(observed_months), f"{name} formula output is empty")
        _require(
            set(observed_months).issubset(_expected_months(name)),
            f"{name} formula output escaped its month domain",
        )
        receipts: list[MonthCheckpointReceipt] = []
        for month in observed_months:
            fragment = _filter_month(source, month)
            _require(fragment.num_rows > 0, f"{name}/{month} formula output is empty")
            payload, receipt_path = self._paths(name, month)
            existing = self._inspect(name, month)
            if existing is not None:
                _require(
                    existing.logical_content_sha256 == _table_fingerprint(fragment),
                    f"completed final month changed: {name}/{month}",
                )
                receipts.append(existing)
                continue
            payload.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            stage_descriptor, stage_name = tempfile.mkstemp(
                prefix=f".{month}-",
                suffix=".parquet.partial",
                dir=payload.parent,
            )
            os.close(stage_descriptor)
            stage = Path(stage_name)
            try:
                pq.write_table(
                    fragment,
                    stage,
                    compression="zstd",
                    row_group_size=131_072,
                    use_dictionary=True,
                    write_statistics=True,
                )
                reopened = _as_native_table(
                    name, pq.read_table(stage, use_threads=False).combine_chunks()
                )
                _require_no_nonnullable_nulls(reopened, label=f"staged {name}/{month}")
                _require(
                    _tables_logically_equal_with_nan_transport(reopened, fragment),
                    f"staged final fragment differs: {name}/{month}",
                )
                receipt = MonthCheckpointReceipt(
                    table_name=name,
                    holding_month=month,
                    rows=reopened.num_rows,
                    parquet_sha256=_sha256_file(stage),
                    logical_content_sha256=_table_fingerprint(reopened),
                    arrow_schema_sha256=_sha256_bytes(
                        reopened.schema.serialize().to_pybytes()
                    ),
                    relative_path=payload.relative_to(self.run_root).as_posix(),
                )
                document = {
                    "schema_version": 1,
                    "status": "FINAL_FORMULA_MONTH_CHECKPOINT_COMPLETE",
                    "checkpoint_id": CHECKPOINT_ID,
                    "context_sha256": self.context.context_sha256,
                    **receipt.mapping(),
                }
                pending = receipt_path.with_suffix(".pending.json")
                _atomic_bytes(pending, _canonical_json_bytes(document))
                os.replace(stage, payload)
                os.replace(pending, receipt_path)
                receipts.append(self._inspect(name, month) or receipt)
            finally:
                if stage.exists() and not stage.is_symlink():
                    stage.unlink()
        return tuple(receipts)

    @property
    def barrier_path(self) -> Path:
        return self.run_root / "DECISION_SOURCES_COMMITTED.json"

    def commit_decision_barrier(self) -> Mapping[str, object]:
        self.initialize()
        identities: dict[str, list[dict[str, object]]] = {}
        for name in DECISION_TABLES:
            _require(
                not self.pending_months(name), f"decision table is incomplete: {name}"
            )
            identities[name] = [
                (self._inspect(name, month) or _missing_receipt()).mapping()
                for month in _expected_months(name)
            ]
        body: dict[str, object] = {
            "schema_version": 1,
            "status": "FINAL_DECISION_SOURCES_COMMITTED",
            "barrier_id": DECISION_BARRIER_ID,
            "context_sha256": self.context.context_sha256,
            "decision_tables": identities,
            "decision_source_tables_committed_before_outcome_source_publication": True,
        }
        body["decision_identity_sha256"] = _sha256_bytes(_canonical_json_bytes(body))
        if self.barrier_path.exists():
            observed = _json(self.barrier_path, label="final decision barrier")
            _require(observed == body, "final decision barrier changed")
        else:
            _atomic_bytes(self.barrier_path, _canonical_json_bytes(body))
        return MappingProxyType(body)

    def require_decision_barrier(self) -> Mapping[str, object]:
        _require(
            self.barrier_path.is_file(), "final outcome opened before decision commit"
        )
        expected = dict(self.commit_decision_barrier())
        observed = _json(self.barrier_path, label="final decision barrier")
        _require(observed == expected, "final decision barrier differs")
        return MappingProxyType(observed)

    def commit_outcome_table(
        self, name: str, table: pa.Table
    ) -> tuple[MonthCheckpointReceipt, ...]:
        _require(name in OUTCOME_TABLES, "non-outcome table used outcome commit")
        self.require_decision_barrier()
        return self.commit_table(name, table)

    def load_table(self, name: str) -> pa.Table:
        self.initialize()
        _require(
            not self.pending_months(name), f"final replay table is incomplete: {name}"
        )
        pieces: list[pa.Table] = []
        for month in _expected_months(name):
            payload, _receipt = self._paths(name, month)
            pieces.append(
                _as_native_table(name, pq.read_table(payload, use_threads=False))
            )
        result = pa.concat_tables(pieces).combine_chunks()
        _require(
            tuple(sorted(set(_month_column(result).to_pylist())))
            == _expected_months(name),
            f"assembled final replay months differ: {name}",
        )
        return result

    def completion_identity(self) -> dict[str, object]:
        self.require_decision_barrier()
        tables: dict[str, object] = {}
        for name in _SOURCE_NAMES:
            _require(
                not self.pending_months(name),
                f"final replay table is incomplete: {name}",
            )
            receipts = [
                (self._inspect(name, month) or _missing_receipt()).mapping()
                for month in _expected_months(name)
            ]
            tables[name] = {
                "months": len(receipts),
                "rows": sum(int(item["rows"]) for item in receipts),
                "month_receipts_sha256": _sha256_bytes(_canonical_json_bytes(receipts)),
            }
        identity = {
            "replay_id": REPLAY_ID,
            "checkpoint_id": CHECKPOINT_ID,
            "context_sha256": self.context.context_sha256,
            "decision_barrier_sha256": _sha256_file(self.barrier_path),
            "tables": tables,
            "completed_holding_months_memoized": True,
            "passed_months_recomputed_on_resume": 0,
        }
        identity["completion_sha256"] = _sha256_bytes(_canonical_json_bytes(identity))
        return identity


def _missing_receipt() -> MonthCheckpointReceipt:
    raise FinalSourceReplayError("authenticated month receipt disappeared")


def _require_exact_built_months(
    name: str,
    table: pa.Table,
    pending_months: tuple[str, ...],
) -> None:
    """Reject a partial family before any of its fragments can be committed."""

    _require(isinstance(table, pa.Table), f"{name} builder did not return a table")
    months = _month_column(table)
    _require(months.null_count == 0, f"{name} emitted a null holding month")
    observed = tuple(sorted(set(str(value) for value in months.to_pylist())))
    expected = tuple(sorted(pending_months))
    _require(
        observed == expected,
        f"{name} built month domain differs; expected={expected}, observed={observed}",
    )
    counts = Counter(str(value) for value in months.to_pylist())
    _require(
        all(counts[month] > 0 for month in expected),
        f"{name} emitted an empty pending month",
    )


class _FinalFormulaEngine(Protocol):
    """Internal fixed engine; never accepted as a public caller argument."""

    def preflight_decision_sources(self) -> Mapping[str, object]: ...

    def prepare_decision_sources(self) -> Mapping[str, object]: ...

    def prepare_checkpoint_store(
        self,
        *,
        store: FinalReplayCheckpointStore,
    ) -> Mapping[str, object]: ...

    def build_decision_table(
        self,
        name: str,
        pending_months: tuple[str, ...],
        *,
        store: FinalReplayCheckpointStore,
    ) -> pa.Table: ...

    def open_outcome_sources(
        self, *, decision_barrier: Mapping[str, object]
    ) -> None: ...

    def build_outcome_table(
        self,
        name: str,
        pending_months: tuple[str, ...],
        *,
        store: FinalReplayCheckpointStore,
    ) -> pa.Table: ...

    def identity(self) -> Mapping[str, object]: ...


def _execute_with_engine(
    *,
    context: FinalReplayContext,
    raw_identity: Mapping[str, object],
    engine: _FinalFormulaEngine,
    checkpoint_root: Path,
):
    """Private state machine used by production and synthetic certification."""

    from portfolio_bot.final_test.final_source_builder import (
        AuthenticatedFinalFormulaReplay,
        EXPECTED_SOURCE_PUBLICATION_SEQUENCE,
        _REPLAY_SEAL,
    )

    # Authenticate the complete provider/date graph before the checkpoint
    # store is constructed.  A static contradiction therefore cannot publish
    # even the first source receipt in a fresh context.
    preflight = engine.preflight_decision_sources()
    _require(
        preflight.get("status") == "FINAL_DECISION_SOURCE_PREFLIGHT_PASSED"
        and preflight.get("outcome_sources_opened") is False,
        "final decision-source preflight is not authoritative",
    )
    decision_source_preparation = engine.prepare_decision_sources()
    _require(
        decision_source_preparation.get("status")
        == "FINAL_DECISION_SOURCE_PREPARATION_COMPLETE"
        and decision_source_preparation.get("outcome_sources_opened") is False,
        "final decision-source preparation is not authoritative",
    )
    store = FinalReplayCheckpointStore(checkpoint_root, context=context)
    store.initialize()
    checkpoint_preparation = engine.prepare_checkpoint_store(store=store)
    _require(
        checkpoint_preparation.get("status")
        == "FINAL_DECISION_CHECKPOINT_STORE_PREPARED"
        and checkpoint_preparation.get("outcome_sources_opened") is False,
        "final decision checkpoint preparation is not authoritative",
    )
    for name in DECISION_TABLES:
        pending = store.pending_months(name)
        if pending:
            built = engine.build_decision_table(name, pending, store=store)
            _require_exact_built_months(name, built, pending)
            store.commit_table(name, built)
        _require(
            not store.pending_months(name),
            f"decision source remains incomplete: {name}",
        )
    barrier = store.commit_decision_barrier()
    engine.open_outcome_sources(decision_barrier=barrier)
    for name in OUTCOME_TABLES:
        pending = store.pending_months(name)
        if pending:
            built = engine.build_outcome_table(name, pending, store=store)
            _require_exact_built_months(name, built, pending)
            store.commit_outcome_table(name, built)
        _require(
            not store.pending_months(name), f"outcome source remains incomplete: {name}"
        )
    completion = store.completion_identity()
    tables = {name: store.load_table(name) for name in _SOURCE_NAMES}
    formula_identity = {
        **dict(engine.identity()),
        "decision_source_preflight": dict(preflight),
        "decision_source_preparation": dict(decision_source_preparation),
        "decision_checkpoint_preparation": dict(checkpoint_preparation),
        "checkpoint_completion": completion,
        "stage6_formulas_reused": True,
        "stage7_formulas_reused": True,
        "stage8_formulas_reused": True,
        "stage11_source_formulas_reused": True,
        "decision_source_tables_committed_before_outcome_source_open": True,
    }
    return AuthenticatedFinalFormulaReplay(
        tables=MappingProxyType(tables),
        raw_source_identity=MappingProxyType(dict(raw_identity)),
        source_publication_sequence=EXPECTED_SOURCE_PUBLICATION_SEQUENCE,
        formula_identity=MappingProxyType(formula_identity),
        _seal=_REPLAY_SEAL,
    )


def _issue_synthetic_engine_replay(
    *,
    context: FinalReplayContext,
    raw_identity: Mapping[str, object],
    engine: _FinalFormulaEngine,
    checkpoint_root: Path,
    _token: object = _SYNTHETIC_ENGINE_TOKEN,
):
    _require(_token is _SYNTHETIC_ENGINE_TOKEN, "synthetic engine token differs")
    return _execute_with_engine(
        context=context,
        raw_identity=raw_identity,
        engine=engine,
        checkpoint_root=checkpoint_root,
    )


@contextlib.contextmanager
def _temporary_assignments(
    assignments: Sequence[tuple[object, str, object]],
) -> Iterator[None]:
    """Serialize, install, and restore a closed set of module assignments."""

    keys = tuple((id(module), name) for module, name, _value in assignments)
    _require(
        bool(keys) and len(keys) == len(set(keys)), "final scope assignments differ"
    )
    acquired = _REPLAY_LOCK.acquire(blocking=False)
    _require(acquired, "another final formula replay scope is active")
    originals = [
        (module, name, getattr(module, name)) for module, name, _ in assignments
    ]
    try:
        for module, name, value in assignments:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)
        _REPLAY_LOCK.release()


def _final_signal_rows(calendar: pa.Table) -> tuple[object, ...]:
    from portfolio_bot.validation.validation_data_contract import ValidationSignal

    result: list[ValidationSignal] = []
    for row in calendar.to_pylist():
        result.append(
            ValidationSignal(
                holding_month=str(row["source_holding_month"]),
                signal_date=row["signal_date"],
                entry_date=row["entry_date"],
                exit_date=row["exit_date"],
            )
        )
    _require(
        len(result) == 72
        and result[0].signal_date == FINAL_FIRST_SIGNAL
        and result[-1].signal_date == FINAL_LAST_SIGNAL,
        "final signal calendar differs",
    )
    return tuple(result)


def _stable_stage11_history_identity(project_root: Path) -> dict[str, object]:
    """Authenticate the memoized Stage 11 logical feature history metadata."""

    execution_path = (
        project_root / "artifacts/phase_05_validation/STAGE_11_EXECUTION_INPUTS.json"
    )
    verified_path = (
        project_root
        / "artifacts/phase_05_validation/STAGE_11_EXECUTION_INPUTS_VERIFIED.json"
    )
    execution = _json(execution_path, label="Stage 11 execution-input manifest")
    verified = _json(verified_path, label="Stage 11 execution-input verification")
    execution_payload = execution_path.read_bytes()
    _require(
        execution.get("status") == "SEALED_STAGE11_EMPIRICAL_EXECUTION_INPUTS"
        and verified.get("execution_manifest_sha256") == _sha256_file(execution_path)
        and verified.get("execution_manifest_payload_sha256")
        == execution.get("payload_sha256"),
        "Stage 11 execution-input authority differs",
    )
    predictor = execution.get("predictor_authority")
    _require(isinstance(predictor, Mapping), "Stage 11 predictor authority is absent")
    predictor_manifest = Path(str(predictor.get("manifest_path")))
    _require(
        predictor_manifest.is_absolute()
        and predictor_manifest.is_file()
        and not predictor_manifest.is_symlink()
        and _sha256_file(predictor_manifest) == predictor.get("manifest_sha256"),
        "Stage 11 predictor manifest differs",
    )
    predictor_document = _json(predictor_manifest, label="Stage 11 predictor manifest")
    validation_binding = predictor_document.get("validation_data_verification")
    _require(
        isinstance(validation_binding, Mapping),
        "Stage 11 validation-data binding is absent",
    )
    candidate_relative = validation_binding.get("candidate_manifest_path")
    candidate_sha = validation_binding.get("candidate_manifest_sha256")
    _require(
        isinstance(candidate_relative, str)
        and isinstance(candidate_sha, str)
        and _HEX64.fullmatch(candidate_sha) is not None,
        "Stage 11 candidate binding is absent",
    )
    candidate_path = project_root / candidate_relative
    _require(
        candidate_path.is_file()
        and not candidate_path.is_symlink()
        and _sha256_file(candidate_path) == candidate_sha
        and candidate_sha == predictor.get("candidate_manifest_sha256"),
        "Stage 11 candidate manifest differs",
    )
    candidate = _json(candidate_path, label="Stage 11 candidate manifest")
    empirical = candidate.get("empirical_build")
    _require(isinstance(empirical, Mapping), "Stage 11 empirical binding is absent")
    empirical_path = project_root / str(empirical.get("receipt_path"))
    _require(
        empirical_path.is_file()
        and _sha256_file(empirical_path) == empirical.get("receipt_sha256"),
        "Stage 11 empirical receipt differs",
    )
    empirical_document = _json(empirical_path, label="Stage 11 empirical receipt")
    feature = empirical_document.get("feature_input")
    _require(isinstance(feature, Mapping), "Stage 11 feature-input binding is absent")
    feature_receipt = Path(str(feature.get("receipt_path")))
    expected_feature_sha = feature.get("receipt_sha256")
    expected_bundle_sha = feature.get("input_bundle_sha256")
    _require(
        expected_feature_sha == empirical.get("feature_input_receipt_sha256")
        and expected_bundle_sha == empirical.get("feature_input_bundle_sha256"),
        "Stage 11 feature-input receipt/candidate binding differs",
    )
    _require(
        feature_receipt.is_absolute()
        and feature_receipt.is_file()
        and not feature_receipt.is_symlink()
        and _sha256_file(feature_receipt) == expected_feature_sha,
        "Stage 11 feature-input receipt differs",
    )
    feature_document = _json(feature_receipt, label="Stage 11 feature-input receipt")
    feature_content = {
        key: value
        for key, value in feature_document.items()
        if key not in {"input_bundle_sha256", "input_bundle_id"}
    }
    _require(
        expected_feature_sha == FROZEN_STAGE11_FEATURE_INPUT_RECEIPT_SHA256
        and expected_bundle_sha == FROZEN_STAGE11_FEATURE_INPUT_BUNDLE_SHA256
        and feature.get("adapter_code_sha256")
        == FROZEN_STAGE11_FEATURE_INPUT_ADAPTER_CODE_SHA256
        and empirical.get("feature_input_adapter_code_sha256")
        == FROZEN_STAGE11_FEATURE_INPUT_ADAPTER_CODE_SHA256
        and feature_document.get("status")
        == "stage11_verified_logical_feature_history_complete"
        and feature_document.get("adapter_id")
        == FROZEN_STAGE11_FEATURE_INPUT_ADAPTER_ID
        and feature_document.get("input_bundle_sha256") == expected_bundle_sha
        and feature_document.get("input_bundle_id")
        == f"stage11-feature-inputs-{str(expected_bundle_sha)[:24]}"
        and _sha256_bytes(_canonical_json_bytes(feature_content))
        == expected_bundle_sha,
        "Stage 11 feature-input bundle identity differs",
    )
    return {
        "stage11_execution_input_manifest_sha256": _sha256_bytes(execution_payload),
        "stage11_execution_input_verification_sha256": _sha256_file(verified_path),
        "stage11_predictor_manifest_sha256": _sha256_file(predictor_manifest),
        "stage11_candidate_manifest_sha256": candidate_sha,
        "stage11_feature_input_receipt_sha256": expected_feature_sha,
        "stage11_feature_input_bundle_sha256": expected_bundle_sha,
        "stage11_feature_input_adapter_id": (FROZEN_STAGE11_FEATURE_INPUT_ADAPTER_ID),
        "stage11_feature_input_adapter_code_sha256": (
            FROZEN_STAGE11_FEATURE_INPUT_ADAPTER_CODE_SHA256
        ),
        "stage11_feature_input_root": str(feature_receipt.parent),
    }


def _load_stage11_development_inputs(
    database_root: Path,
    *,
    stage11_identity: Mapping[str, object],
):
    """Load the authenticated Stage 11 catalog and restore the full calendar."""

    from portfolio_bot.features.feature_store_inputs import (
        FeatureStoreInputs,
        resolve_input_catalog,
    )
    from portfolio_bot.validation.validation_data_contract import (
        validate_validation_data_contract,
    )
    from portfolio_bot.validation import validation_feature_input_adapter as adapter

    stage11_root = database_root / STAGE11_FEATURE_INPUT_ROOT
    _require(
        stage11_identity.get("stage11_feature_input_root") == str(stage11_root)
        and stage11_identity.get("stage11_feature_input_receipt_sha256")
        == FROZEN_STAGE11_FEATURE_INPUT_RECEIPT_SHA256
        and stage11_identity.get("stage11_feature_input_bundle_sha256")
        == FROZEN_STAGE11_FEATURE_INPUT_BUNDLE_SHA256
        and stage11_identity.get("stage11_feature_input_adapter_id")
        == FROZEN_STAGE11_FEATURE_INPUT_ADAPTER_ID
        and stage11_identity.get("stage11_feature_input_adapter_code_sha256")
        == FROZEN_STAGE11_FEATURE_INPUT_ADAPTER_CODE_SHA256
        and adapter.INPUT_ADAPTER_ID == CURRENT_FEATURE_INPUT_ADAPTER_ID,
        "frozen Stage 11 feature-input compatibility binding differs",
    )
    authority = validate_validation_data_contract()
    # Stage 11's immutable, independently bound receipt predates the narrow
    # nullability-transport and zero-row-provenance amendments.  Authenticate
    # that exact historical receipt with its writer ID, then restore the
    # current adapter before any new Stage 13 feature bundle can be composed.
    with _temporary_assignments(
        (
            (
                adapter,
                "INPUT_ADAPTER_ID",
                FROZEN_STAGE11_FEATURE_INPUT_ADAPTER_ID,
            ),
        )
    ):
        stage11 = adapter.load_validation_feature_input_bundle(
            stage11_root,
            predictor_authority=authority,
        ).scored_inputs
    _require(
        adapter.INPUT_ADAPTER_ID == CURRENT_FEATURE_INPUT_ADAPTER_ID,
        "current feature-input adapter was not restored",
    )
    stage6 = resolve_input_catalog(authenticate_hashes=True)
    tables = dict(stage11.tables)
    for name in ("trading_calendar.parquet", "rebalance_calendar.parquet"):
        tables[name] = stage6.tables[name]
    return FeatureStoreInputs(stage6.database_root, MappingProxyType(tables)), authority


def _build_final_calendar(development_inputs: object) -> pa.Table:
    source_path = development_inputs.path("rebalance_calendar.parquet")
    table = pq.read_table(
        source_path,
        columns=["holding_month", "signal_date", "entry_date", "exit_date"],
        filters=[
            ("holding_month", ">=", FINAL_HOLDING_MONTHS[0]),
            ("holding_month", "<=", FINAL_HOLDING_MONTHS[-1]),
        ],
        use_threads=False,
    ).combine_chunks()
    rows_by_month = {str(row["holding_month"]): row for row in table.to_pylist()}
    _require(
        tuple(sorted(rows_by_month)) == FINAL_HOLDING_MONTHS,
        "verified rebalance calendar lacks the final domain",
    )
    rows: list[dict[str, object]] = []
    for month in FINAL_HOLDING_MONTHS:
        source = rows_by_month[month]
        descriptor = final_holding_month_descriptor(month)
        exit_date = source["exit_date"]
        if month == PARTIAL_DECEMBER_HOLDING_MONTH:
            exit_date = FINAL_LAST_EXIT
        rows.append(
            {
                "source_holding_month": month,
                "signal_date": source["signal_date"],
                "entry_date": source["entry_date"],
                "exit_date": exit_date,
                "outcome_availability_date": exit_date,
                "segment": descriptor.segment.value,
                "complete_holding_interval": descriptor.complete_holding_interval,
                "confirmatory_inference_eligible": (
                    descriptor.confirmatory_inference_eligible
                ),
            }
        )
    result = pa.Table.from_pylist(rows, schema=_native_schema("final_calendar"))
    _require(
        result["signal_date"][0].as_py() == FINAL_FIRST_SIGNAL
        and result["signal_date"][-1].as_py() == FINAL_LAST_SIGNAL,
        "final calendar signal boundaries differ",
    )
    return result


@contextlib.contextmanager
def _market_formula_scope(
    *,
    extension_start: dt.date,
    last_signal: dt.date,
    last_exit: dt.date,
    first_context_signal: dt.date,
) -> Iterator[object]:
    """Authorize one closed date adapter around the frozen CRSP pipeline."""

    from portfolio_bot.validation import validation_market_extension as market

    reserve_start = last_signal + dt.timedelta(days=1)
    assignments = (
        (market, "EXTENSION_START", extension_start),
        (market, "LAST_VALIDATION_SIGNAL", last_signal),
        (market, "LAST_VALIDATION_EXIT", last_exit),
        (market, "OUTCOME_RESERVE_START", reserve_start),
        (market, "FIRST_CONTEXT_SIGNAL", first_context_signal),
    )
    with _temporary_assignments(assignments):
        scope = market.ValidationMarketScope(
            extension_start=extension_start,
            last_signal=last_signal,
            last_exit=last_exit,
            first_context_signal=first_context_signal,
        )
        with _temporary_assignments(((market, "DEFAULT_SCOPE", scope),)):
            yield scope


def _quarantine_incomplete(path: Path) -> None:
    """Preserve incomplete work while clearing the fixed resumable target."""

    if not path.exists() and not path.is_symlink():
        return
    _require(
        path.is_dir() and not path.is_symlink(), "incomplete replay path is unsafe"
    )
    digest = hashlib.sha256(
        f"{path}:{path.stat().st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    destination = path.with_name(f"{path.name}.interrupted-{digest}")
    _require(not destination.exists(), "incomplete replay quarantine already exists")
    os.replace(path, destination)


def _return_root_from_receipt(
    receipt: Mapping[str, object], *, database_root: Path
) -> Path:
    pointer = receipt.get("rehearsal_pointer")
    _require(isinstance(pointer, Mapping), "return replay pointer is absent")
    relative = Path(str(pointer.get("interim_path")))
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        "return replay path is unsafe",
    )
    result = database_root / relative
    _require(
        result.is_dir() and not result.is_symlink(), "return replay output is absent"
    )
    return result


def _stable_typed_market_extension(
    root: Path,
    *,
    extension_start: dt.date,
    last_signal: dt.date,
    last_exit: dt.date,
    first_context_signal: dt.date,
    expected_receipt_sha256: str | None = None,
) -> Mapping[str, object]:
    """Reauthenticate every typed-market payload before memoized reuse."""

    from portfolio_bot.data.phase_02_crsp_typed_backbone import (
        load_production_settings,
    )
    from portfolio_bot.validation.validation_market_extension import (
        CONTRACT_ID,
        DEVELOPMENT_CONTEXT_START,
        _receipt as market_receipt,
        validation_table_schema,
    )

    table_names = {
        "crsp_identity_interval",
        "crsp_shares_interval",
        "crsp_security_day",
        "crsp_security_day_return_staging",
        "crsp_security_month_crosscheck",
        "crsp_security_month_return_staging",
    }
    allowed = {f"{name}.parquet" for name in table_names} | {
        "typed_extension_receipt.json"
    }
    _require(root.is_dir() and not root.is_symlink(), "typed market root is absent")
    _require(
        {path.name for path in root.iterdir()} == allowed,
        "typed market artifact set differs",
    )
    receipt_path = root / "typed_extension_receipt.json"
    receipt_metadata = receipt_path.stat(follow_symlinks=False)
    _require(
        stat.S_ISREG(receipt_metadata.st_mode)
        and not receipt_path.is_symlink()
        and receipt_metadata.st_nlink == 1,
        "typed market receipt is unsafe",
    )
    if expected_receipt_sha256 is not None:
        _require(
            _HEX64.fullmatch(expected_receipt_sha256) is not None
            and _sha256_file(receipt_path) == expected_receipt_sha256,
            "typed market receipt pin differs",
        )
    receipt = _json(receipt_path, label="typed market receipt")
    _require(
        receipt_path.read_bytes() == _canonical_json_bytes(receipt)
        and set(receipt)
        == {
            "schema_version",
            "contract_id",
            "status",
            "materialization_role",
            "scope",
            "source_bindings",
            "tables",
            "interval_structure",
            "authorizations",
        },
        "typed market receipt differs",
    )
    expected_scope = {
        "extension_start": extension_start.isoformat(),
        "last_signal": last_signal.isoformat(),
        "last_exit": last_exit.isoformat(),
        "first_context_signal": first_context_signal.isoformat(),
        "maximum_payload_date": last_signal.isoformat(),
        "post_2019_10_31_market_observation_or_interval_start_rows_retained": 0,
        "future_interval_endpoints_may_be_retained_for_asof_join_audit": True,
        "reserved_outcome_interval": {
            "start": (last_signal + dt.timedelta(days=1)).isoformat(),
            "end": last_exit.isoformat(),
            "materialized": False,
            "separate_unblind_authority_required": True,
        },
    }
    expected_authorizations = {
        "predictor_context_materialization_only": True,
        "validation_labels_or_selection_authorized": False,
        "portfolio_weights_trades_pnl_or_performance_authorized": False,
        "final_2020_2025_access_authorized": False,
    }
    settings = load_production_settings()
    expected_sources = {
        name: {
            "filename": specification.filename,
            "sha256": specification.sha256,
            "size_bytes": specification.size_bytes,
            "full_file_rows": specification.full_file_rows,
        }
        for name, specification in sorted(settings.sources.items())
    }
    tables = receipt.get("tables")
    interval_structure = receipt.get("interval_structure")
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("contract_id") == CONTRACT_ID
        and receipt.get("status")
        == "TYPED_VALIDATION_PREDICTOR_MARKET_EXTENSION_BUILT_PENDING_VERIFICATION"
        and receipt.get("materialization_role")
        == "point_in_time_predictor_context_only"
        and receipt.get("scope") == expected_scope
        and receipt.get("source_bindings") == expected_sources
        and receipt.get("authorizations") == expected_authorizations
        and isinstance(tables, Mapping)
        and set(tables) == table_names
        and isinstance(interval_structure, Mapping)
        and set(interval_structure) == {"identity", "shares"},
        "typed market authority differs",
    )
    date_columns = {
        "crsp_identity_interval": "start_date",
        "crsp_shares_interval": "start_date",
        "crsp_security_day": "calendar_date",
        "crsp_security_day_return_staging": "calendar_date",
        "crsp_security_month_crosscheck": "calendar_month",
        "crsp_security_month_return_staging": "calendar_month",
    }
    for name in sorted(table_names):
        record = tables[name]
        _require(isinstance(record, Mapping), f"typed market receipt differs: {name}")
        path = root / f"{name}.parquet"
        metadata = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and not path.is_symlink()
            and metadata.st_nlink == 1,
            f"typed market payload is unsafe: {name}",
        )
        observed = market_receipt(
            path,
            name=name,
            date_column=date_columns[name],
        ).as_dict()
        minimum_date = (
            DEVELOPMENT_CONTEXT_START
            if name in {"crsp_identity_interval", "crsp_shares_interval"}
            else extension_start
        )
        expected_schema = validation_table_schema(
            settings.registry.tables[name],
            minimum_date=minimum_date,
            maximum_date=last_signal,
        )
        _require(
            dict(record) == observed
            and pq.ParquetFile(path).schema_arrow.equals(
                expected_schema,
                check_metadata=True,
            ),
            f"typed market payload identity differs: {name}",
        )
    for name in ("identity", "shares"):
        structure = interval_structure[name]
        _require(
            isinstance(structure, Mapping)
            and set(structure)
            == {
                "rows",
                "unique_permnos",
                "duplicate_key_groups",
                "contiguous_edges",
                "gap_edges",
                "overlap_edges",
            }
            and int(structure.get("rows", -1))
            == int(tables[f"crsp_{name}_interval"]["rows"])
            and int(structure.get("unique_permnos", 0)) > 0
            and int(structure.get("duplicate_key_groups", -1)) == 0
            and int(structure.get("gap_edges", -1)) == 0
            and int(structure.get("overlap_edges", -1)) == 0,
            f"typed market interval structure differs: {name}",
        )
    _require(
        tables["crsp_security_day"]["rows"]
        == tables["crsp_security_day_return_staging"]["rows"]
        and tables["crsp_security_month_crosscheck"]["rows"]
        == tables["crsp_security_month_return_staging"]["rows"],
        "typed market consumer and return row counts differ",
    )
    return MappingProxyType(dict(receipt))


def _stable_return_market_extension(
    return_base: Path,
    *,
    typed_root: Path,
    database_root: Path,
    extension_start: dt.date,
    last_signal: dt.date,
    last_exit: dt.date,
    expected_receipt_sha256: str | None = None,
    expected_market_extension_source_sha256: str | None = None,
) -> tuple[Path, Mapping[str, object]]:
    """Reauthenticate the complete return-replay bundle before reuse."""

    from portfolio_bot.data import phase_02_crsp_return_reconciliation as reconciliation
    from portfolio_bot.validation import validation_return_adapter
    from portfolio_bot.validation.validation_market_extension import (
        CONTRACT_ID,
        _receipt as market_receipt,
    )

    _require(
        return_base.is_dir() and not return_base.is_symlink(),
        "return market root is absent",
    )
    _require(
        {path.name for path in return_base.iterdir()}
        == {"artifact", "interim", "quality", "return_extension_receipt.json"},
        "return market artifact set differs",
    )
    receipt_path = return_base / "return_extension_receipt.json"
    receipt_metadata = receipt_path.stat(follow_symlinks=False)
    _require(
        stat.S_ISREG(receipt_metadata.st_mode)
        and not receipt_path.is_symlink()
        and receipt_metadata.st_nlink == 1,
        "return market receipt is unsafe",
    )
    if expected_receipt_sha256 is not None:
        _require(
            _HEX64.fullmatch(expected_receipt_sha256) is not None
            and _sha256_file(receipt_path) == expected_receipt_sha256,
            "return market receipt pin differs",
        )
    receipt = _json(receipt_path, label="return market receipt")
    _require(
        receipt_path.read_bytes() == _canonical_json_bytes(receipt)
        and set(receipt)
        == {
            "schema_version",
            "contract_id",
            "status",
            "build_id",
            "materialization_role",
            "scope",
            "bindings",
            "pointer_publication_attempted",
            "production_return_pointer_sha256_after",
            "rehearsal_pointer",
            "monthly_transport_audit",
            "authoritative_daily_return",
            "authorizations",
        },
        "return market receipt differs",
    )
    bindings = receipt.get("bindings")
    pointer = receipt.get("rehearsal_pointer")
    _require(
        isinstance(bindings, Mapping) and isinstance(pointer, Mapping),
        "return market bindings are absent",
    )
    production_pointer = reconciliation.PROJECT_ROOT / (
        "artifacts/phase_02_canonical_panel/"
        "LATEST_CRSP_RETURN_RECONCILIATION_PRIMARY.json"
    )
    production_pointer_sha = (
        _sha256_file(production_pointer) if production_pointer.is_file() else None
    )
    current_market_extension_source_sha256 = _sha256_file(
        Path(__file__).resolve().parents[1]
        / "validation/validation_market_extension.py"
    )
    bound_market_extension_source_sha256 = (
        current_market_extension_source_sha256
        if expected_market_extension_source_sha256 is None
        else expected_market_extension_source_sha256
    )
    _require(
        _HEX64.fullmatch(bound_market_extension_source_sha256) is not None,
        "return market extension source pin is malformed",
    )
    expected_binding = {
        "contract_id": CONTRACT_ID,
        "materialization_role": "point_in_time_predictor_context_only",
        "scope_start": extension_start.isoformat(),
        "scope_end": last_signal.isoformat(),
        "reserved_outcome_start": (last_signal + dt.timedelta(days=1)).isoformat(),
        "reserved_outcome_end": last_exit.isoformat(),
        "typed_daily_staging_sha256": _sha256_file(
            typed_root / "crsp_security_day_return_staging.parquet"
        ),
        "typed_monthly_staging_sha256": _sha256_file(
            typed_root / "crsp_security_month_return_staging.parquet"
        ),
        "reused_reconciliation_source_sha256": _sha256_file(
            Path(reconciliation.__file__).resolve()
        ),
        "stage11_return_adapter_source_sha256": _sha256_file(
            Path(validation_return_adapter.__file__).resolve()
        ),
        "stage11_market_extension_source_sha256": (
            bound_market_extension_source_sha256
        ),
        "expected_rows": bindings.get("expected_rows"),
        "production_return_pointer_sha256_before": production_pointer_sha,
    }
    expected_build_id = (
        "stage11-return-extension-"
        + _sha256_bytes(_canonical_json_bytes(expected_binding))[:24]
    )
    expected_scope = {
        "start": extension_start.isoformat(),
        "end": last_signal.isoformat(),
        "reserved_outcome_interval": {
            "start": (last_signal + dt.timedelta(days=1)).isoformat(),
            "end": last_exit.isoformat(),
            "materialized": False,
            "separate_unblind_authority_required": True,
        },
        "first_prohibited_final_entry": "2020-01-02",
    }
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("contract_id") == CONTRACT_ID
        and receipt.get("status")
        == "VALIDATION_PREDICTOR_RETURN_EXTENSION_BUILT_PENDING_VERIFICATION"
        and receipt.get("build_id") == expected_build_id
        and receipt.get("materialization_role")
        == "point_in_time_predictor_context_only"
        and receipt.get("scope") == expected_scope
        and dict(bindings) == expected_binding
        and receipt.get("pointer_publication_attempted") is False
        and receipt.get("production_return_pointer_sha256_after")
        == production_pointer_sha
        and receipt.get("authorizations")
        == {
            "point_in_time_predictor_context_only": True,
            "validation_label_or_model_selection_access_authorized": False,
            "portfolio_or_performance_authorized": False,
            "final_2020_2025_access_authorized": False,
        },
        "return market authority differs",
    )
    base_relative = return_base.relative_to(database_root).as_posix()
    expected_paths = {
        "interim_path": f"{base_relative}/interim/{expected_build_id}",
        "quality_path": f"{base_relative}/quality/{expected_build_id}",
        "artifact_path": f"{base_relative}/artifact/{expected_build_id}",
    }
    _require(
        pointer.get("schema_version") == 1
        and pointer.get("build_id") == expected_build_id
        and pointer.get("status")
        == "return_reconciliation_primary_passed_pending_independent_verification"
        and pointer.get("return_history_authorized") is False
        and pointer.get("adjusted_price_path_feature_authorized") is False
        and all(pointer.get(key) == value for key, value in expected_paths.items()),
        "return market pointer differs",
    )
    interim_root = _return_root_from_receipt(
        receipt,
        database_root=database_root,
    )
    artifact_root = database_root / expected_paths["artifact_path"]
    quality_root = database_root / expected_paths["quality_path"]
    _require(
        all(
            path.is_dir() and not path.is_symlink()
            for path in (interim_root, artifact_root, quality_root)
        ),
        "return market bundle path is unsafe",
    )
    manifest_path = artifact_root / "manifest.json"
    success_path = interim_root / "_SUCCESS.json"
    for path in (manifest_path, success_path):
        metadata = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and not path.is_symlink()
            and metadata.st_nlink == 1,
            "return market manifest or success is unsafe",
        )
    manifest = _json(manifest_path, label="return market manifest")
    success = _json(success_path, label="return market success")
    _require(
        manifest_path.read_bytes() == _canonical_json_bytes(manifest)
        and success_path.read_bytes() == _canonical_json_bytes(success)
        and _sha256_file(manifest_path) == pointer.get("manifest_sha256")
        and {
            key: value
            for key, value in manifest.items()
            if key
            not in {
                "interim_path",
                "quality_path",
                "primary_manifest_sha256",
            }
        }
        == success,
        "return market manifest or success differs",
    )
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("build_id") == expected_build_id
        and manifest.get("status")
        == "return_reconciliation_primary_passed_pending_independent_verification"
        and manifest.get("scope")
        == {
            "start": extension_start.isoformat(),
            "end": last_signal.isoformat(),
        }
        and pointer.get("contract_id") == manifest.get("contract_id")
        and manifest.get("authorizations")
        == {
            "adjusted_price_path_feature_authorized": False,
            "beta_estimation_authorized": False,
            (
                "features_labels_models_portfolios_pnl_performance_or_"
                "backtests_authorized"
            ): False,
            "independent_verification_required": True,
            "return_history_authorized": False,
            "validation_or_final_test_access_authorized": False,
        },
        "return market manifest semantics differ",
    )
    outputs = manifest.get("outputs")
    expected_output_names = {
        "crsp_distribution_event.parquet",
        "crsp_delisting_event.parquet",
        "crsp_security_day_return.parquet",
        "crsp_security_month_return_reconciliation.parquet",
    }
    _require(
        isinstance(outputs, list)
        and {str(item.get("path")) for item in outputs if isinstance(item, Mapping)}
        == expected_output_names
        and {path.name for path in interim_root.iterdir()}
        == expected_output_names | {"_SUCCESS.json"}
        and {path.name for path in artifact_root.iterdir()} == {"manifest.json"}
        and {path.name for path in quality_root.iterdir()}
        == {"checks.csv", "summary.json"},
        "return market closed artifact set differs",
    )
    output_by_name: dict[str, Mapping[str, object]] = {}
    for raw in outputs:
        _require(
            isinstance(raw, Mapping)
            and set(raw) == {"path", "rows", "sha256", "size_bytes"},
            "return market output receipt differs",
        )
        name = str(raw["path"])
        path = interim_root / name
        metadata = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and not path.is_symlink()
            and metadata.st_nlink == 1
            and int(raw.get("rows", -1)) == pq.ParquetFile(path).metadata.num_rows
            and int(raw.get("size_bytes", -1)) == metadata.st_size
            and raw.get("sha256") == _sha256_file(path),
            f"return market output identity differs: {name}",
        )
        output_by_name[name] = raw
    expected_rows = {
        "crsp_distribution_event": int(
            output_by_name["crsp_distribution_event.parquet"]["rows"]
        ),
        "crsp_delisting_event": int(
            output_by_name["crsp_delisting_event.parquet"]["rows"]
        ),
        "crsp_security_day_return": int(
            output_by_name["crsp_security_day_return.parquet"]["rows"]
        ),
        "crsp_security_month_return_reconciliation": int(
            output_by_name["crsp_security_month_return_reconciliation.parquet"]["rows"]
        ),
    }
    daily_receipt = receipt.get("authoritative_daily_return")
    observed_daily_receipt = market_receipt(
        interim_root / "crsp_security_day_return.parquet",
        name="crsp_security_day_return",
        date_column="calendar_date",
    ).as_dict()
    monthly_audit = receipt.get("monthly_transport_audit")
    _require(
        bindings.get("expected_rows") == expected_rows
        and isinstance(daily_receipt, Mapping)
        and dict(daily_receipt) == observed_daily_receipt
        and daily_receipt.get("rows") == expected_rows["crsp_security_day_return"],
        "return market row or daily authority differs",
    )
    _require(
        isinstance(monthly_audit, Mapping)
        and monthly_audit.get("schema_version") == 1
        and monthly_audit.get("diagnostics") == []
        and monthly_audit.get("out_of_tolerance_diagnostic_rows") == 0
        and monthly_audit.get("within_frozen_tolerance_rows")
        == monthly_audit.get("monthly_ns_rows")
        and monthly_audit.get("rule")
        == "NS_new_security_partial_start_is_diagnostic_not_complete_month",
        "return market monthly transport audit differs",
    )
    for path in (
        manifest_path,
        success_path,
        quality_root / "checks.csv",
        quality_root / "summary.json",
    ):
        metadata = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and not path.is_symlink()
            and metadata.st_nlink == 1,
            "return market metadata file is unsafe",
        )
    return interim_root, MappingProxyType(dict(receipt))


def _closed_readonly_cache_snapshot(
    roots: Mapping[str, Path],
) -> tuple[dict[str, object], ...]:
    """Hash a closed set of selected cache subtrees without following aliases."""

    records: list[dict[str, object]] = []
    seen_inodes: set[tuple[int, int]] = set()

    def visit(label: str, root: Path, path: Path) -> None:
        metadata = path.stat(follow_symlinks=False)
        relative = path.relative_to(root).as_posix()
        identity = f"{label}/{relative}" if relative != "." else label
        _require(
            not stat.S_ISLNK(metadata.st_mode),
            "parent decision-market cache contains a symlink",
        )
        if stat.S_ISDIR(metadata.st_mode):
            records.append(
                {
                    "path": identity,
                    "kind": "directory",
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "links": metadata.st_nlink,
                    "size_bytes": metadata.st_size,
                    "modified_ns": metadata.st_mtime_ns,
                    "changed_ns": metadata.st_ctime_ns,
                    "sha256": None,
                }
            )
            for child in sorted(path.iterdir(), key=lambda value: value.name):
                visit(label, root, child)
            return
        _require(
            stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
            "parent decision-market cache contains a special or aliased file",
        )
        inode = (metadata.st_dev, metadata.st_ino)
        _require(
            inode not in seen_inodes, "parent decision-market cache aliases a file"
        )
        seen_inodes.add(inode)
        records.append(
            {
                "path": identity,
                "kind": "file",
                "mode": stat.S_IMODE(metadata.st_mode),
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "links": metadata.st_nlink,
                "size_bytes": metadata.st_size,
                "modified_ns": metadata.st_mtime_ns,
                "changed_ns": metadata.st_ctime_ns,
                "sha256": _sha256_file(path),
            }
        )

    _require(
        set(roots) == {"typed", "returns"},
        "parent decision-market selected cache roots differ",
    )
    for label, root in sorted(roots.items()):
        _require(
            root.is_dir() and not root.is_symlink(),
            f"parent decision-market {label} cache is unsafe",
        )
        visit(label, root, root)
    return tuple(sorted(records, key=lambda record: str(record["path"])))


def _authenticate_parent_decision_market_cache(
    *,
    namespace_root: Path,
    database_root: Path,
    rebind_receipt: Mapping[str, object],
) -> tuple[Path, Path, Mapping[str, object]]:
    """Reuse only the authority-selected v4 decision cache, directly and read-only."""

    from portfolio_bot.data import phase_02_crsp_return_reconciliation as reconciliation
    from portfolio_bot.validation import validation_return_adapter

    parent_sha = str(rebind_receipt.get("parent_context_sha256"))
    replacement_sha = str(rebind_receipt.get("replacement_context_sha256"))
    cache = rebind_receipt.get("parent_decision_market_cache_authority")
    _require(
        rebind_receipt.get("status")
        == "STAGE13_DECISION_CHECKPOINT_V4_TO_V5_REBIND_COMPLETE"
        and _HEX64.fullmatch(parent_sha) is not None
        and _HEX64.fullmatch(replacement_sha) is not None
        and parent_sha != replacement_sha
        and isinstance(cache, Mapping)
        and cache.get("reuse_mode") == "authenticated_read_only_direct_parent_paths",
        "parent decision-market cache rebind authority differs",
    )
    scope = cache.get("scope")
    safety = cache.get("safety")
    _require(
        scope
        == {
            "extension_start": FINAL_EXTENSION_START.isoformat(),
            "last_signal": FINAL_LAST_SIGNAL.isoformat(),
            "last_exit": FINAL_LAST_EXIT.isoformat(),
            "first_context_signal": "2019-10-31",
        }
        and isinstance(safety, Mapping)
        and safety.get("selected_subtrees_read_only") is True
        and safety.get("selected_subtrees_reauthenticated_before_reuse") is True
        and safety.get("selected_subtrees_identical_before_after") is True
        and safety.get("unselected_siblings_not_opened") is True
        and safety.get("hardlinks_symlinks_special_nodes_forbidden") is True
        and safety.get("typed_and_return_builder_semantics_unchanged") is True
        and safety.get("entry_projection_change_semantically_disjoint") is True
        and safety.get("replacement_cache_copy_authorized") is False
        and safety.get("fallback_rebuild_on_authentication_failure") is False
        and safety.get("interrupted_siblings_may_not_be_opened_or_quarantined") is True,
        "parent decision-market cache scope or safety differs",
    )
    selected_namespace = Path(os.path.abspath(namespace_root.expanduser()))
    selected_database = Path(os.path.abspath(database_root.expanduser()))
    _require(
        selected_namespace.is_dir()
        and not selected_namespace.is_symlink()
        and selected_namespace.is_relative_to(selected_database),
        "parent decision-market namespace is unsafe",
    )
    parent_context_root = _ordinary_directory(
        selected_namespace / parent_sha,
        root=selected_namespace,
        label="parent decision-market context",
    )
    replacement_context_root = _ordinary_directory(
        selected_namespace / replacement_sha,
        root=selected_namespace,
        label="replacement decision-market context",
    )
    _require(
        parent_context_root.is_dir()
        and not parent_context_root.is_symlink()
        and replacement_context_root.is_dir()
        and not replacement_context_root.is_symlink(),
        "parent or replacement decision-market context is unsafe",
    )
    relative = Path(str(cache.get("relative_path")))
    typed_relative = Path(str(cache.get("typed_relative_path")))
    returns_relative = Path(str(cache.get("returns_relative_path")))
    _require(
        all(
            not path.is_absolute() and bool(path.parts) and ".." not in path.parts
            for path in (relative, typed_relative, returns_relative)
        ),
        "parent decision-market cache path is unsafe",
    )
    decision_root = _ordinary_directory(
        parent_context_root / relative,
        root=parent_context_root,
        label="parent decision-market cache root",
    )
    typed_root = _ordinary_directory(
        decision_root / typed_relative,
        root=decision_root,
        label="parent typed decision-market cache",
    )
    returns_root = _ordinary_directory(
        decision_root / returns_relative,
        root=decision_root,
        label="parent return decision-market cache",
    )
    replacement_cache = replacement_context_root / relative
    _require(
        not replacement_cache.exists() and not replacement_cache.is_symlink(),
        "replacement decision-market cache exists instead of direct parent reuse",
    )
    expected_pins = {
        name: str(cache.get(name))
        for name in (
            "typed_receipt_sha256",
            "return_receipt_sha256",
            "legacy_market_extension_source_sha256",
            "reconciliation_source_sha256",
            "return_adapter_source_sha256",
        )
    }
    _require(
        all(_HEX64.fullmatch(value) is not None for value in expected_pins.values())
        and _sha256_file(Path(reconciliation.__file__).resolve())
        == expected_pins["reconciliation_source_sha256"]
        and _sha256_file(Path(validation_return_adapter.__file__).resolve())
        == expected_pins["return_adapter_source_sha256"],
        "parent decision-market non-overridable source identity differs",
    )
    before = _closed_readonly_cache_snapshot(
        {"typed": typed_root, "returns": returns_root}
    )
    typed_receipt = _stable_typed_market_extension(
        typed_root,
        extension_start=FINAL_EXTENSION_START,
        last_signal=FINAL_LAST_SIGNAL,
        last_exit=FINAL_LAST_EXIT,
        first_context_signal=dt.date(2019, 10, 31),
        expected_receipt_sha256=expected_pins["typed_receipt_sha256"],
    )
    return_root, return_receipt = _stable_return_market_extension(
        returns_root,
        typed_root=typed_root,
        database_root=selected_database,
        extension_start=FINAL_EXTENSION_START,
        last_signal=FINAL_LAST_SIGNAL,
        last_exit=FINAL_LAST_EXIT,
        expected_receipt_sha256=expected_pins["return_receipt_sha256"],
        expected_market_extension_source_sha256=expected_pins[
            "legacy_market_extension_source_sha256"
        ],
    )
    after = _closed_readonly_cache_snapshot(
        {"typed": typed_root, "returns": returns_root}
    )
    _require(before == after, "parent decision-market cache changed during reuse")
    _require(
        not replacement_cache.exists() and not replacement_cache.is_symlink(),
        "parent decision-market reuse created a replacement cache",
    )
    tree_sha256 = _sha256_bytes(_canonical_json_bytes(before))
    result = {
        "status": "PARENT_DECISION_MARKET_CACHE_AUTHENTICATED_READ_ONLY",
        "parent_context_sha256": parent_sha,
        "replacement_context_sha256": replacement_sha,
        "typed_receipt_sha256": expected_pins["typed_receipt_sha256"],
        "return_receipt_sha256": expected_pins["return_receipt_sha256"],
        "legacy_market_extension_source_sha256": expected_pins[
            "legacy_market_extension_source_sha256"
        ],
        "reconciliation_source_sha256": expected_pins["reconciliation_source_sha256"],
        "return_adapter_source_sha256": expected_pins["return_adapter_source_sha256"],
        "selected_cache_tree_sha256": tree_sha256,
        "selected_cache_record_count": len(before),
        "typed_payload_rows": sum(
            int(record["rows"]) for record in typed_receipt["tables"].values()
        ),
        "return_daily_rows": int(return_receipt["authoritative_daily_return"]["rows"]),
        "direct_parent_paths_reused": True,
        "replacement_cache_created": False,
        "fallback_rebuild_permitted": False,
        "unselected_parent_siblings_opened": False,
        "parent_selected_subtrees_identical_before_after": True,
    }
    return typed_root, return_root, MappingProxyType(result)


def _ensure_market_replay(
    *,
    root: Path,
    database_root: Path,
    extension_start: dt.date,
    last_signal: dt.date,
    last_exit: dt.date,
    first_context_signal: dt.date,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> tuple[Path, Path, Mapping[str, object]]:
    """Build/reuse an authenticated CRSP typed+return formula checkpoint."""

    from portfolio_bot.validation.validation_market_extension import (
        build_return_extension,
        build_typed_market_extension,
    )

    typed_root = root / "typed"
    return_base = root / "returns"
    typed_receipt_path = typed_root / "typed_extension_receipt.json"
    return_receipt_path = return_base / "return_extension_receipt.json"
    with _market_formula_scope(
        extension_start=extension_start,
        last_signal=last_signal,
        last_exit=last_exit,
        first_context_signal=first_context_signal,
    ) as scope:
        if typed_receipt_path.is_file():
            typed_receipt = _stable_typed_market_extension(
                typed_root,
                extension_start=extension_start,
                last_signal=last_signal,
                last_exit=last_exit,
                first_context_signal=first_context_signal,
            )
        else:
            if typed_root.exists():
                _quarantine_incomplete(typed_root)
            build_typed_market_extension(
                typed_root,
                scope=scope,
                progress=progress,
            )
            typed_receipt = _stable_typed_market_extension(
                typed_root,
                extension_start=extension_start,
                last_signal=last_signal,
                last_exit=last_exit,
                first_context_signal=first_context_signal,
            )
        if return_receipt_path.is_file():
            return_root, return_receipt = _stable_return_market_extension(
                return_base,
                typed_root=typed_root,
                database_root=database_root,
                extension_start=extension_start,
                last_signal=last_signal,
                last_exit=last_exit,
            )
        else:
            if return_base.exists():
                _quarantine_incomplete(return_base)
            build_return_extension(
                typed_root,
                return_base,
                scope=scope,
                progress=progress,
            )
            return_root, return_receipt = _stable_return_market_extension(
                return_base,
                typed_root=typed_root,
                database_root=database_root,
                extension_start=extension_start,
                last_signal=last_signal,
                last_exit=last_exit,
            )
    return (
        typed_root,
        return_root,
        MappingProxyType(
            {
                "typed_receipt_sha256": _sha256_file(typed_receipt_path),
                "return_receipt_sha256": _sha256_file(return_receipt_path),
                "typed_scope": [extension_start.isoformat(), last_signal.isoformat()],
            }
        ),
    )


def _stable_entry_market_projection(
    root: Path,
    *,
    expected_entry_date: dt.date,
) -> tuple[Path, Mapping[str, object]]:
    """Reauthenticate the exact-entry consumer projection on every reuse."""

    from portfolio_bot.data.phase_02_crsp_typed_backbone import (
        load_production_settings,
    )
    from portfolio_bot.validation.validation_market_extension import (
        ENTRY_MARKET_PROJECTION_CONTRACT_ID,
        ValidationMarketExtensionError,
        _ENTRY_MARKET_EXPECTED_DAILY_CANONICAL_ROWS,
        _ENTRY_MARKET_EXPECTED_DAILY_CONSUMER_CONFLICTING_DUPLICATE_GROUPS,
        _ENTRY_MARKET_EXPECTED_DAILY_CONSUMER_EXACT_DUPLICATE_KEY_GROUPS,
        _ENTRY_MARKET_EXPECTED_DAILY_CONSUMER_EXACT_DUPLICATE_SURPLUS_ROWS,
        _ENTRY_MARKET_EXPECTED_DAILY_OUTPUT_DUPLICATE_KEYS,
        _ENTRY_MARKET_EXPECTED_DAILY_PROJECTION_RETAINED_ROWS,
        _ENTRY_MARKET_RAW_COLUMNS,
        _RAW_RETURN_COLUMNS,
        _entry_market_share_coverage,
        _receipt as market_receipt,
        entry_market_projection_schema,
    )

    _require(root.is_dir() and not root.is_symlink(), "entry-market root is absent")
    allowed = {
        "crsp_security_day.parquet",
        "crsp_shares_interval.parquet",
        "entry_market_projection_receipt.json",
    }
    _require(
        {path.name for path in root.iterdir()} == allowed,
        "entry-market projection artifact set differs",
    )
    receipt_path = root / "entry_market_projection_receipt.json"
    receipt_metadata = receipt_path.stat(follow_symlinks=False)
    _require(
        stat.S_ISREG(receipt_metadata.st_mode)
        and not receipt_path.is_symlink()
        and receipt_metadata.st_nlink == 1,
        "entry-market projection receipt is unsafe",
    )
    receipt = _json(receipt_path, label="entry-market projection receipt")
    _require(
        receipt_path.read_bytes() == _canonical_json_bytes(receipt),
        "entry-market projection receipt is not canonical",
    )
    _require(
        set(receipt)
        == {
            "schema_version",
            "contract_id",
            "status",
            "materialization_role",
            "scope",
            "raw_access",
            "source_bindings",
            "tables",
            "entry_rows_by_date",
            "canonicalization",
            "share_coverage",
            "authorizations",
        },
        "entry-market projection receipt fields differ",
    )
    expected_date = expected_entry_date.isoformat()
    expected_dates_sha256 = _sha256_bytes(_canonical_json_bytes([expected_date]))
    scope = receipt.get("scope")
    raw_access = receipt.get("raw_access")
    authorizations = receipt.get("authorizations")
    source_bindings = receipt.get("source_bindings")
    canonicalization = receipt.get("canonicalization")
    share_coverage = receipt.get("share_coverage")
    tables = receipt.get("tables")
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("contract_id") == ENTRY_MARKET_PROJECTION_CONTRACT_ID
        and receipt.get("status") == "EXACT_ENTRY_MARKET_CONSUMER_PROJECTION_COMPLETE"
        and receipt.get("materialization_role")
        == "prebarrier_exact_entry_market_consumer_only"
        and isinstance(scope, Mapping)
        and set(scope)
        == {
            "exact_entry_dates",
            "exact_entry_dates_sha256",
            "minimum_entry_date",
            "maximum_entry_date",
            "decision_return_cutoff_unchanged",
            "reserved_outcome_end",
        }
        and scope.get("exact_entry_dates") == [expected_date]
        and scope.get("exact_entry_dates_sha256") == expected_dates_sha256
        and scope.get("minimum_entry_date") == expected_date
        and scope.get("maximum_entry_date") == expected_date
        and scope.get("decision_return_cutoff_unchanged")
        == FINAL_LAST_SIGNAL.isoformat()
        and scope.get("reserved_outcome_end") == FINAL_LAST_EXIT.isoformat()
        and isinstance(raw_access, Mapping)
        and set(raw_access)
        == {
            "daily_requested_columns",
            "raw_return_columns_requested",
            "monthly_sources_opened",
            "return_reconciliation_invoked",
            "labels_outcomes_or_accounting_opened",
        }
        and raw_access.get("raw_return_columns_requested") == []
        and raw_access.get("monthly_sources_opened") is False
        and raw_access.get("return_reconciliation_invoked") is False
        and raw_access.get("labels_outcomes_or_accounting_opened") is False
        and isinstance(authorizations, Mapping)
        and set(authorizations)
        == {
            "exact_entry_market_consumer_facts_only",
            "returns_labels_outcomes_accounting_authorized",
            "portfolio_weights_trades_pnl_or_performance_authorized",
        }
        and authorizations.get("exact_entry_market_consumer_facts_only") is True
        and authorizations.get("returns_labels_outcomes_accounting_authorized") is False
        and authorizations.get("portfolio_weights_trades_pnl_or_performance_authorized")
        is False
        and isinstance(source_bindings, Mapping)
        and set(source_bindings) == {"daily", "shares"}
        and isinstance(canonicalization, Mapping)
        and set(canonicalization)
        == {
            "daily_projection_retained_rows",
            "daily_canonical_rows",
            "daily_consumer_exact_duplicate_key_groups",
            "daily_consumer_exact_duplicate_surplus_rows",
            "daily_consumer_conflicting_duplicate_groups",
            "daily_output_duplicate_keys",
            "shares_duplicate_key_groups",
            "shares_overlap_edges",
            "shares_gap_edges",
            "shares_non_entry_interval_rows",
        }
        and isinstance(share_coverage, Mapping)
        and set(share_coverage)
        == {
            "join_rule",
            "population_rows",
            "one_match_rows",
            "zero_match_rows",
            "multiple_match_rows",
            "active_share_rows",
            "active_share_rows_without_daily",
            "zero_match_policy",
            "tombstone_predicate_id",
            "tombstone_rows",
            "tombstone_key_set_sha256",
            "final_market_inputs_join_semantics",
        }
        and isinstance(tables, Mapping)
        and set(tables) == {"crsp_security_day", "crsp_shares_interval"},
        "entry-market projection authority differs",
    )
    settings = load_production_settings()
    expected_requested_columns = [
        name
        for name in settings.sources["daily"].header
        if name in _ENTRY_MARKET_RAW_COLUMNS
    ]
    requested_columns = raw_access.get("daily_requested_columns")
    _require(
        requested_columns == expected_requested_columns
        and not _RAW_RETURN_COLUMNS.intersection(
            str(value) for value in requested_columns
        ),
        "entry-market projection requested a raw return field",
    )
    _require(
        all(
            source_bindings[name]
            == {
                "filename": settings.sources[name].filename,
                "sha256": settings.sources[name].sha256,
                "size_bytes": settings.sources[name].size_bytes,
                "full_file_rows": settings.sources[name].full_file_rows,
            }
            for name in ("daily", "shares")
        ),
        "entry-market projection source binding differs",
    )

    paths: dict[str, Path] = {}
    for name in ("crsp_security_day", "crsp_shares_interval"):
        record = tables[name]
        _require(
            isinstance(record, Mapping)
            and set(record)
            == {
                "name",
                "path",
                "rows",
                "row_groups",
                "size_bytes",
                "sha256",
                "schema_sha256",
                "minimum_date",
                "maximum_date",
            },
            f"entry-market table receipt differs: {name}",
        )
        path = root / f"{name}.parquet"
        metadata = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and not path.is_symlink()
            and metadata.st_nlink == 1
            and record.get("name") == name
            and record.get("path") == path.name
            and record.get("size_bytes") == metadata.st_size
            and record.get("sha256") == _sha256_file(path),
            f"entry-market table identity differs: {name}",
        )
        parquet = pq.ParquetFile(path)
        expected_schema = entry_market_projection_schema(
            settings.registry.tables[name],
            minimum_date=expected_entry_date,
            maximum_date=expected_entry_date,
            exact_entry_dates_sha256=expected_dates_sha256,
        )
        observed_receipt = market_receipt(
            path,
            name=name,
            date_column=(
                "start_date" if name == "crsp_shares_interval" else "calendar_date"
            ),
        ).as_dict()
        _require(
            dict(record) == observed_receipt
            and record.get("rows") == parquet.metadata.num_rows
            and record.get("row_groups") == parquet.metadata.num_row_groups
            and record.get("schema_sha256")
            == _sha256_bytes(parquet.schema_arrow.serialize().to_pybytes())
            and parquet.schema_arrow.equals(expected_schema, check_metadata=True),
            f"entry-market table schema or cardinality differs: {name}",
        )
        paths[name] = path

    daily = pq.read_table(
        paths["crsp_security_day"], use_threads=False
    ).combine_chunks()
    _require(
        daily.num_rows > 0
        and set(daily["calendar_date"].to_pylist()) == {expected_entry_date}
        and receipt.get("entry_rows_by_date") == {expected_date: daily.num_rows},
        "entry-market daily date domain differs",
    )
    daily_keys = (
        daily.select(["permno", "calendar_date"])
        .group_by(["permno", "calendar_date"])
        .aggregate([])
    )
    daily_output_duplicate_keys = daily.num_rows - daily_keys.num_rows
    _require(daily_output_duplicate_keys == 0, "entry-market daily keys duplicate")
    canonical_counts = tuple(canonicalization.values())
    duplicate_groups = canonicalization["daily_consumer_exact_duplicate_key_groups"]
    duplicate_surplus = canonicalization["daily_consumer_exact_duplicate_surplus_rows"]
    canonical_rows = canonicalization["daily_canonical_rows"]
    retained_rows = canonicalization["daily_projection_retained_rows"]
    _require(
        all(type(value) is int and value >= 0 for value in canonical_counts)
        and canonical_rows == daily.num_rows
        and canonical_rows == sum(receipt["entry_rows_by_date"].values())
        and retained_rows == canonical_rows + duplicate_surplus
        and duplicate_groups <= duplicate_surplus
        and (duplicate_groups == 0) == (duplicate_surplus == 0)
        and canonicalization["daily_consumer_conflicting_duplicate_groups"] == 0
        and canonicalization["daily_output_duplicate_keys"]
        == daily_output_duplicate_keys
        and retained_rows == _ENTRY_MARKET_EXPECTED_DAILY_PROJECTION_RETAINED_ROWS
        and canonical_rows == _ENTRY_MARKET_EXPECTED_DAILY_CANONICAL_ROWS
        and duplicate_groups
        == _ENTRY_MARKET_EXPECTED_DAILY_CONSUMER_EXACT_DUPLICATE_KEY_GROUPS
        and duplicate_surplus
        == _ENTRY_MARKET_EXPECTED_DAILY_CONSUMER_EXACT_DUPLICATE_SURPLUS_ROWS
        and canonicalization["daily_consumer_conflicting_duplicate_groups"]
        == _ENTRY_MARKET_EXPECTED_DAILY_CONSUMER_CONFLICTING_DUPLICATE_GROUPS
        and canonicalization["daily_output_duplicate_keys"]
        == _ENTRY_MARKET_EXPECTED_DAILY_OUTPUT_DUPLICATE_KEYS
        and canonicalization["shares_duplicate_key_groups"] == 0
        and canonicalization["shares_overlap_edges"] == 0
        and canonicalization["shares_gap_edges"] == 0
        and canonicalization["shares_non_entry_interval_rows"] == 0,
        "entry-market canonicalization receipt differs",
    )
    shares = pq.read_table(
        paths["crsp_shares_interval"], use_threads=False
    ).combine_chunks()
    _require(
        shares.num_rows > 0
        and shares["permno"].null_count == 0
        and shares["start_date"].null_count == 0
        and shares["end_date"].null_count == 0
        and all(
            start <= expected_entry_date <= end
            for start, end in zip(
                shares["start_date"].to_pylist(),
                shares["end_date"].to_pylist(),
                strict=True,
            )
        ),
        "entry-market shares contain a non-entry interval",
    )
    active_permnos = shares["permno"].to_pylist()
    _require(
        len(active_permnos) == len(set(active_permnos)),
        "entry-market active shares intervals are empty or duplicate",
    )
    try:
        observed_share_coverage = _entry_market_share_coverage(
            daily,
            shares,
            entry_date=expected_entry_date,
        )
    except ValidationMarketExtensionError as exc:
        raise FinalSourceReplayError(str(exc)) from exc
    _require(
        dict(share_coverage) == observed_share_coverage,
        "entry-market share coverage receipt differs",
    )
    return root, MappingProxyType(dict(receipt))


def _final_decision_boundary_plan(calendar: pa.Table) -> Mapping[str, object]:
    """Prove every frozen decision date has a non-outcome provider."""

    rows = calendar.to_pylist()
    months = tuple(str(row["source_holding_month"]) for row in rows)
    _require(
        months == tuple(FINAL_HOLDING_MONTHS), "final calendar month order differs"
    )
    _require(
        all(row["signal_date"] < row["entry_date"] < row["exit_date"] for row in rows),
        "final calendar chronology differs",
    )
    _require(
        max(row["signal_date"] for row in rows) == FINAL_LAST_SIGNAL
        and max(row["entry_date"] for row in rows) == FINAL_PARTIAL_ENTRY_DATE
        and max(row["exit_date"] for row in rows) == FINAL_LAST_EXIT,
        "final signal, entry, or exit boundary differs",
    )
    rows_by_month = {str(row["source_holding_month"]): row for row in rows}
    for month in FROZEN_HOLDING_INTERVAL_ANCHORS:
        row = rows_by_month[month]
        require_frozen_holding_interval_anchor(
            holding_month=month,
            signal_date=row["signal_date"],
            entry_date=row["entry_date"],
            exit_date=row["exit_date"],
        )
    partial = rows[-1]
    _require(
        partial["source_holding_month"] == PARTIAL_DECEMBER_HOLDING_MONTH
        and partial["signal_date"] == FINAL_LAST_SIGNAL
        and partial["entry_date"] == FINAL_PARTIAL_ENTRY_DATE
        and partial["exit_date"] == FINAL_LAST_EXIT
        and all(row["entry_date"] <= FINAL_LAST_SIGNAL for row in rows[:-1]),
        "partial-December provider boundary differs",
    )
    plan = {
        "schema_version": 1,
        "decision_return_maximum_date": FINAL_LAST_SIGNAL.isoformat(),
        "ordinary_entry_market_maximum_date": FINAL_LAST_SIGNAL.isoformat(),
        "partial_entry_market_exact_dates": [FINAL_PARTIAL_ENTRY_DATE.isoformat()],
        "outcome_market_minimum_date": (
            FINAL_LAST_SIGNAL + dt.timedelta(days=1)
        ).isoformat(),
        "outcome_return_minimum_date": (
            FINAL_LAST_SIGNAL + dt.timedelta(days=1)
        ).isoformat(),
        "final_exit_date": FINAL_LAST_EXIT.isoformat(),
        "holding_month_count": len(rows),
        "decision_dependency_on_outcome_provider": False,
        "partial_entry_projection_contains_returns": False,
    }
    return MappingProxyType(
        {
            **plan,
            "boundary_plan_sha256": _sha256_bytes(_canonical_json_bytes(plan)),
        }
    )


class RepositoryNativeFinalFormulaEngine:
    """Fixed Stage 13 adapter around the repository's frozen formula modules."""

    def __init__(
        self,
        *,
        database_root: Path,
        work_root: Path,
        project_root: Path,
        replay_context: FinalReplayContext | None = None,
    ) -> None:
        self.database_root = database_root.absolute()
        self.work_root = work_root.absolute()
        self.project_root = project_root.absolute()
        self.replay_context = replay_context
        self._stage11_identity = _stable_stage11_history_identity(self.project_root)
        self._development_inputs, self._stage11_contract = (
            _load_stage11_development_inputs(
                self.database_root,
                stage11_identity=self._stage11_identity,
            )
        )
        self._calendar = _build_final_calendar(self._development_inputs)
        self._calendar_by_month = {
            str(row["source_holding_month"]): row for row in self._calendar.to_pylist()
        }
        self._decision_market: tuple[Path, Path] | None = None
        self._decision_return_daily_sha256: str | None = None
        self._decision_entry_market: Path | None = None
        self._outcome_market: tuple[Path, Path] | None = None
        self._outcome_return_daily_sha256: str | None = None
        self._feature_bundle: object | None = None
        self._risk_cache: dict[
            tuple[str, ...], tuple[Mapping[str, pa.Table], Mapping[str, object]]
        ] = {}
        self._factor_table: pa.Table | None = None
        self._factor_normalization_receipt: Mapping[str, object] | None = None
        self._formula_receipts: dict[str, object] = {}
        self._outcome_opened = False

    def _ensure_decision_market(self) -> tuple[Path, Path]:
        if self._decision_market is None:
            _require(
                self.replay_context is None,
                (
                    "production decision market was not authenticated from the "
                    "read-only parent cache"
                ),
            )
            typed, returns, receipt = _ensure_market_replay(
                root=self.work_root / "decision_market",
                database_root=self.database_root,
                extension_start=FINAL_EXTENSION_START,
                last_signal=FINAL_LAST_SIGNAL,
                last_exit=FINAL_LAST_EXIT,
                first_context_signal=dt.date(2019, 10, 31),
            )
            self._decision_market = (typed, returns)
            self._decision_return_daily_sha256 = _sha256_file(
                returns / "crsp_security_day_return.parquet"
            )
            self._formula_receipts["decision_market"] = dict(receipt)
        return self._decision_market

    def _ensure_decision_entry_market(self) -> Path:
        if self._decision_entry_market is None:
            from portfolio_bot.validation.validation_market_extension import (
                build_entry_market_projection,
            )

            root = self.work_root / "decision_entry_market"
            receipt_path = root / "entry_market_projection_receipt.json"
            with _market_formula_scope(
                extension_start=FINAL_EXTENSION_START,
                last_signal=FINAL_LAST_SIGNAL,
                last_exit=FINAL_LAST_EXIT,
                first_context_signal=dt.date(2019, 10, 31),
            ) as scope:
                if not receipt_path.is_file():
                    if root.exists() or root.is_symlink():
                        _quarantine_incomplete(root)
                    build_entry_market_projection(
                        root,
                        entry_dates=(FINAL_PARTIAL_ENTRY_DATE,),
                        scope=scope,
                    )
            stable_root, receipt = _stable_entry_market_projection(
                root,
                expected_entry_date=FINAL_PARTIAL_ENTRY_DATE,
            )
            self._decision_entry_market = stable_root
            self._formula_receipts["decision_entry_market"] = {
                "receipt_sha256": _sha256_file(receipt_path),
                "contract_id": receipt["contract_id"],
                "exact_entry_dates": list(receipt["scope"]["exact_entry_dates"]),
                "raw_return_columns_requested": [],
                "outcome_or_accounting_sources_opened": False,
            }
        return self._decision_entry_market

    def _ensure_outcome_market(self) -> tuple[Path, Path]:
        _require(self._outcome_opened, "outcome market opened before decision barrier")
        if self._outcome_market is None:
            typed, returns, receipt = _ensure_market_replay(
                root=self.work_root / "outcome_market",
                database_root=self.database_root,
                extension_start=FINAL_LAST_SIGNAL + dt.timedelta(days=1),
                last_signal=FINAL_LAST_EXIT,
                last_exit=dt.date(2026, 1, 2),
                first_context_signal=FINAL_LAST_SIGNAL,
            )
            self._outcome_market = (typed, returns)
            self._outcome_return_daily_sha256 = _sha256_file(
                returns / "crsp_security_day_return.parquet"
            )
            self._formula_receipts["outcome_market"] = dict(receipt)
        return self._outcome_market

    def open_outcome_sources(self, *, decision_barrier: Mapping[str, object]) -> None:
        _require(
            decision_barrier.get("status") == "FINAL_DECISION_SOURCES_COMMITTED"
            and decision_barrier.get(
                "decision_source_tables_committed_before_outcome_source_publication"
            )
            is True,
            "decision barrier is not authoritative",
        )
        self._outcome_opened = True

    def preflight_decision_sources(self) -> Mapping[str, object]:
        plan = _final_decision_boundary_plan(self._calendar)
        _require(
            not self._outcome_opened and self._outcome_market is None,
            "outcome source opened during decision preflight",
        )
        result = {
            "status": "FINAL_DECISION_SOURCE_PREFLIGHT_PASSED",
            "boundary_plan_sha256": plan["boundary_plan_sha256"],
            "holding_month_count": len(FINAL_HOLDING_MONTHS),
            "decision_return_cutoff": FINAL_LAST_SIGNAL.isoformat(),
            "partial_entry_date": FINAL_PARTIAL_ENTRY_DATE.isoformat(),
            "entry_market_projection_contains_returns": False,
            "all_decision_dependencies_have_prebarrier_providers": True,
            "write_free_static_preflight": True,
            "outcome_sources_opened": False,
        }
        return MappingProxyType(
            {
                **result,
                "preflight_sha256": _sha256_bytes(_canonical_json_bytes(result)),
            }
        )

    def prepare_decision_sources(self) -> Mapping[str, object]:
        """Materialize and authenticate the sole post-cutoff decision provider."""

        entry_root = self._ensure_decision_entry_market()
        _require(
            not self._outcome_opened and self._outcome_market is None,
            "outcome source opened during decision-source preparation",
        )
        result = {
            "status": "FINAL_DECISION_SOURCE_PREPARATION_COMPLETE",
            "entry_market_projection_receipt_sha256": _sha256_file(
                entry_root / "entry_market_projection_receipt.json"
            ),
            "partial_entry_date": FINAL_PARTIAL_ENTRY_DATE.isoformat(),
            "entry_market_projection_contains_returns": False,
            "checkpoint_store_created": False,
            "outcome_sources_opened": False,
        }
        return MappingProxyType(
            {
                **result,
                "preparation_sha256": _sha256_bytes(_canonical_json_bytes(result)),
            }
        )

    def prepare_checkpoint_store(
        self,
        *,
        store: FinalReplayCheckpointStore,
    ) -> Mapping[str, object]:
        """Rebind the authenticated v4 decision prefix before any build."""

        from portfolio_bot.final_test.source_checkpoint_rebind import (
            COMPLETION_STATUS,
            rebind_stage13_decision_checkpoints,
        )

        _require(
            self.replay_context is not None and store.context == self.replay_context,
            "production checkpoint rebind lacks its replay context",
        )
        receipt = rebind_stage13_decision_checkpoints(
            namespace_root=store.namespace_root,
            replacement_context_document={
                **store.context.mapping(),
                "context_sha256": store.context.context_sha256,
            },
            authority_path=self.project_root / DECISION_CHECKPOINT_REBIND_CONFIG,
            expected_authority_sha256=(
                EXPECTED_DECISION_CHECKPOINT_REBIND_CONFIG_SHA256
            ),
        )
        _require(
            receipt.get("status") == COMPLETION_STATUS
            and receipt.get("source_aggregates", {}).get("checkpoint_count") == 431
            and receipt.get("stage13_relaunch_authorized") is False,
            "decision checkpoint rebind receipt differs",
        )
        pending = {name: store.pending_months(name) for name in DECISION_TABLES}
        barrier_present = store.barrier_path.is_file()
        _require(
            all(not pending[name] for name in DECISION_TABLES[:-1])
            and pending["final_market_inputs"] in {(), ("2025-12",)}
            and (not barrier_present or all(not value for value in pending.values())),
            "rebound decision checkpoint state differs",
        )
        outcome_completed = {
            name: len(_expected_months(name)) - len(store.pending_months(name))
            for name in OUTCOME_TABLES
        }
        _require(
            barrier_present or all(value == 0 for value in outcome_completed.values()),
            "outcome checkpoint predates the decision barrier",
        )
        typed_root, return_root, cache_receipt = (
            _authenticate_parent_decision_market_cache(
                namespace_root=store.namespace_root,
                database_root=self.database_root,
                rebind_receipt=receipt,
            )
        )
        self._decision_market = (typed_root, return_root)
        self._decision_return_daily_sha256 = _sha256_file(
            return_root / "crsp_security_day_return.parquet"
        )
        self._formula_receipts["parent_decision_market_cache"] = dict(cache_receipt)
        self._formula_receipts["decision_checkpoint_rebind"] = dict(receipt)
        result = {
            "status": "FINAL_DECISION_CHECKPOINT_STORE_PREPARED",
            "rebind_status": receipt["status"],
            "rebound_checkpoint_count": 431,
            "current_pending_decision_months": {
                name: list(months) for name, months in pending.items()
            },
            "decision_barrier_present_before_execution": barrier_present,
            "outcome_checkpoint_count_before_execution": sum(
                outcome_completed.values()
            ),
            "parent_decision_market_cache_status": cache_receipt["status"],
            "parent_decision_market_cache_reused_read_only": True,
            "replacement_decision_market_cache_created": False,
            "decision_market_fallback_rebuild_permitted": False,
            "only_missing_initial_decision_checkpoint": ("final_market_inputs/2025-12"),
            "parent_context_mutated": False,
            "outcome_sources_opened": False,
        }
        return MappingProxyType(
            {
                **result,
                "checkpoint_preparation_sha256": _sha256_bytes(
                    _canonical_json_bytes(result)
                ),
            }
        )

    def identity(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "native_engine_id": REPLAY_ID,
                "stage11_memoized_history": dict(self._stage11_identity),
                "formula_receipts": dict(self._formula_receipts),
                "final_scope": {
                    "extension_start": FINAL_EXTENSION_START.isoformat(),
                    "context_signal": FINAL_CONTEXT_SIGNAL.isoformat(),
                    "first_signal": FINAL_FIRST_SIGNAL.isoformat(),
                    "last_signal": FINAL_LAST_SIGNAL.isoformat(),
                    "partial_entry_date": FINAL_PARTIAL_ENTRY_DATE.isoformat(),
                    "last_exit": FINAL_LAST_EXIT.isoformat(),
                    "holding_month_count": len(FINAL_HOLDING_MONTHS),
                },
                "stage11_artifacts_recomputed": False,
            }
        )

    def _final_contract(self):
        from portfolio_bot.validation.validation_data_contract import (
            ValidationDataContract,
        )

        _require(
            type(self._stage11_contract) is ValidationDataContract,
            "Stage 11 feature contract type differs",
        )
        return dataclasses.replace(
            self._stage11_contract,
            validation_signals=_final_signal_rows(self._calendar),
        )

    def _ensure_feature_bundle(self):
        if self._feature_bundle is not None:
            return self._feature_bundle
        if self.replay_context is not None:
            bundle, feature_receipt, migration_receipt = (
                _prepare_parent_context_migrated_feature_inputs(
                    database_root=self.database_root,
                    work_root=self.work_root / "feature_sources",
                    development_inputs=self._development_inputs,
                    final_contract=self._final_contract(),
                    replay_context=self.replay_context,
                    project_root=self.project_root,
                )
            )
            self._feature_bundle = bundle
            self._formula_receipts["feature_sources"] = dict(feature_receipt)
            self._formula_receipts["source_checkpoint_context_migration"] = dict(
                migration_receipt
            )
            return self._feature_bundle
        typed_root, return_root = self._ensure_decision_market()
        paths = _prepare_final_feature_inputs(
            database_root=self.database_root,
            work_root=self.work_root / "feature_sources",
            development_inputs=self._development_inputs,
            stage11_contract=self._stage11_contract,
            final_contract=self._final_contract(),
            final_calendar=self._calendar,
            final_market_root=typed_root,
            final_return_root=return_root,
            stage11_identity=self._stage11_identity,
        )
        self._feature_bundle = paths[0]
        self._formula_receipts["feature_sources"] = dict(paths[1])
        return self._feature_bundle

    def _ensure_final_factors(self) -> pa.Table:
        if self._factor_table is None:
            path, receipt = _ensure_final_factor_normalization(
                database_root=self.database_root,
                work_root=self.work_root / "outcome_factors",
                trading_calendar_path=self._development_inputs.path(
                    "trading_calendar.parquet"
                ),
            )
            self._factor_table = _final_daily_factors(path)
            self._factor_normalization_receipt = receipt
            self._formula_receipts["outcome_factor_normalization"] = {
                "receipt_sha256": _sha256_file(
                    self.work_root / "outcome_factors/factor_normalization_receipt.json"
                ),
                "rows": self._factor_table.num_rows,
            }
        return self._factor_table

    def build_decision_table(
        self,
        name: str,
        pending_months: tuple[str, ...],
        *,
        store: FinalReplayCheckpointStore,
    ) -> pa.Table:
        _require(name in DECISION_TABLES, "unknown decision source table")
        if name == "final_calendar":
            return self._calendar.filter(
                pc.is_in(
                    self._calendar["source_holding_month"],
                    value_set=pa.array(pending_months, pa.string()),
                )
            )
        if name == "final_predictors":
            table, receipt = _materialize_final_predictors(
                bundle=self._ensure_feature_bundle(),
                final_contract=self._final_contract(),
                final_calendar=self._calendar,
                pending_months=pending_months,
                work_root=self.work_root / "predictor_formula",
            )
            self._formula_receipts.setdefault("predictor_formula_runs", []).append(
                dict(receipt)
            )
            return table
        if name in {
            "final_risk_cost",
            "final_factor_covariance",
            "final_risk_cost_month_summary",
        }:
            cache = self._risk_cache.get(pending_months)
            if cache is None:
                cache = _materialize_final_risk_tables(
                    predictors=store.load_table("final_predictors"),
                    bundle=self._ensure_feature_bundle(),
                    final_contract=self._final_contract(),
                    final_calendar=self._calendar,
                    pending_months=pending_months,
                )
                self._risk_cache[pending_months] = cache
                self._formula_receipts.setdefault("risk_formula_runs", []).append(
                    dict(cache[1])
                )
            return cache[0][name]
        if name == "final_market_inputs":
            typed_root, _return_root = self._ensure_decision_market()
            entry_market_root = self._ensure_decision_entry_market()
            table, receipt = _materialize_final_market_inputs(
                final_calendar=self._calendar,
                predictors=store.load_table("final_predictors"),
                risk_cost=store.load_table("final_risk_cost"),
                typed_market_root=typed_root,
                entry_market_root=entry_market_root,
                pending_months=pending_months,
            )
            self._formula_receipts.setdefault("market_input_formula_runs", []).append(
                dict(receipt)
            )
            return table
        raise FinalSourceReplayError(f"unimplemented decision source: {name}")

    def build_outcome_table(
        self,
        name: str,
        pending_months: tuple[str, ...],
        *,
        store: FinalReplayCheckpointStore,
    ) -> pa.Table:
        _require(
            self._outcome_opened, "outcome source publication opened before barrier"
        )
        _require(name in OUTCOME_TABLES, "unknown outcome source table")
        decision_market, decision_returns = self._ensure_decision_market()
        outcome_market, outcome_returns = self._ensure_outcome_market()
        factors = self._ensure_final_factors()
        if name == "final_market_accounting":
            table, receipt = _materialize_final_market_accounting(
                final_calendar=self._calendar,
                predictors=store.load_table("final_predictors"),
                risk_cost=store.load_table("final_risk_cost"),
                decision_market_root=decision_market,
                outcome_market_root=outcome_market,
                decision_return_root=decision_returns,
                outcome_return_root=outcome_returns,
                factor_table=factors,
                trading_calendar_path=self._development_inputs.path(
                    "trading_calendar.parquet"
                ),
                pending_months=pending_months,
                raw_delisting_path=(self.database_root / "raw/crsp_delisting_ciz.csv"),
                expected_decision_return_sha256=(self._decision_return_daily_sha256),
                expected_outcome_return_sha256=(self._outcome_return_daily_sha256),
            )
        elif name == "final_factors":
            table, receipt = _materialize_final_factors(
                accounting=store.load_table("final_market_accounting"),
                pending_months=pending_months,
            )
        elif name == "final_benchmarks":
            table, receipt = _materialize_final_benchmarks(
                final_calendar=self._calendar,
                factor_table=factors,
                decision_market_root=decision_market,
                outcome_market_root=outcome_market,
                decision_return_root=decision_returns,
                outcome_return_root=outcome_returns,
                pending_months=pending_months,
            )
        elif name == "final_completed_labels":
            table, receipt = _materialize_final_completed_labels(
                final_calendar=self._calendar,
                predictors=store.load_table("final_predictors"),
                accounting=store.load_table("final_market_accounting"),
                factor_table=factors,
                trading_calendar_path=self._development_inputs.path(
                    "trading_calendar.parquet"
                ),
                final_contract=self._final_contract(),
                pending_months=pending_months,
            )
        elif name == "final_outcomes":
            table, receipt = _materialize_security_outcomes(
                name=name,
                accounting=store.load_table("final_market_accounting"),
                predictor=store.load_table("final_predictors"),
                completed_labels=store.load_table("final_completed_labels"),
                pending_months=pending_months,
                calendar=self._calendar,
            )
        elif name == "partial_descriptive_outcome":
            table, receipt = _materialize_security_outcomes(
                name=name,
                accounting=store.load_table("final_market_accounting"),
                predictor=store.load_table("final_predictors"),
                completed_labels=None,
                pending_months=pending_months,
                calendar=self._calendar,
            )
        else:  # pragma: no cover - tuple membership above closes this branch.
            raise FinalSourceReplayError(f"unimplemented outcome source: {name}")
        self._formula_receipts.setdefault("outcome_formula_runs", []).append(
            {"table": name, **dict(receipt)}
        )
        return table


@contextlib.contextmanager
def _universe_auxiliary_scope() -> Iterator[object]:
    from portfolio_bot.validation import validation_universe_auxiliary as universe

    assignments = (
        (universe, "AUXILIARY_EXTENSION_START", FINAL_EXTENSION_START),
        (universe, "INTERNAL_CONTEXT_SIGNAL", FINAL_CONTEXT_SIGNAL),
        (universe, "FIRST_SCORED_SIGNAL", FINAL_FIRST_SIGNAL),
        (universe, "MARKET_EXTENSION_START", FINAL_EXTENSION_START),
        (universe, "LAST_SCORED_SIGNAL", FINAL_LAST_SIGNAL),
        (universe, "FIRST_PROHIBITED_SIGNAL", FINAL_LAST_EXIT),
        (universe, "LAST_PERMITTED_EXIT", FINAL_LAST_EXIT),
        (universe, "FIRST_FINAL_ENTRY", dt.date(2026, 1, 2)),
        (universe, "REQUIRED_SCORED_SIGNALS", len(FINAL_HOLDING_MONTHS)),
        (
            universe,
            "load_policy",
            lambda *_args, **_kwargs: MappingProxyType(
                {
                    "policy_id": REPLAY_ID,
                    "stage13_authorized": True,
                    "formula_change": False,
                }
            ),
        ),
    )
    with _temporary_assignments(assignments):
        scope = universe.ValidationUniverseAuxiliaryScope(
            context_signal=FINAL_CONTEXT_SIGNAL,
            first_scored_signal=FINAL_FIRST_SIGNAL,
            last_scored_signal=FINAL_LAST_SIGNAL,
            last_exit=FINAL_LAST_EXIT,
        )
        with _temporary_assignments(((universe, "DEFAULT_SCOPE", scope),)):
            yield scope


def _consolidated_development_catalog(
    inputs: object,
    *,
    destination: Path,
) -> object:
    """Hard-link the authenticated auxiliary history into one final work root."""

    from portfolio_bot.features.feature_store_inputs import FeatureStoreInputs

    destination.mkdir(parents=True, exist_ok=True)
    tables = dict(inputs.tables)
    for name, table in tuple(tables.items()):
        if table.gate != "auxiliary":
            continue
        target = destination / name
        if target.exists():
            _require(
                target.is_file()
                and not target.is_symlink()
                and target.stat().st_size == table.size_bytes
                and _sha256_file(target) == table.sha256,
                f"memoized Stage 11 auxiliary link differs: {name}",
            )
        else:
            try:
                os.link(table.path, target)
            except OSError:
                # Some macOS/APFS permission profiles prohibit hard links even
                # within the same trusted data root.  A verified byte copy has
                # identical catalog semantics and remains content-addressed.
                shutil.copyfile(table.path, target)
            _require(
                target.is_file()
                and not target.is_symlink()
                and target.stat().st_size == table.size_bytes
                and _sha256_file(target) == table.sha256,
                f"Stage 11 auxiliary catalog copy differs: {name}",
            )
        tables[name] = dataclasses.replace(table, path=target)
    return FeatureStoreInputs(inputs.database_root, MappingProxyType(tables))


def _final_universe_receipt_validator(root: Path) -> dict[str, Any]:
    from portfolio_bot.validation import validation_feature_input_adapter as adapter
    from portfolio_bot.validation.validation_universe_auxiliary import (
        CONTEXT_UNIVERSE_FILE,
        RECEIPT_FILE,
        SCORED_UNIVERSE_FILE,
    )

    receipt = _json(root / RECEIPT_FILE, label="final universe receipt")
    for key, filename, count, first, last in (
        (
            "scored",
            SCORED_UNIVERSE_FILE,
            len(FINAL_HOLDING_MONTHS),
            FINAL_FIRST_SIGNAL,
            FINAL_LAST_SIGNAL,
        ),
        (
            "internal_context",
            CONTEXT_UNIVERSE_FILE,
            1,
            FINAL_CONTEXT_SIGNAL,
            FINAL_CONTEXT_SIGNAL,
        ),
    ):
        value = receipt.get(key)
        _require(isinstance(value, Mapping), f"final universe {key} receipt is absent")
        adapter._verify_parquet_receipt(
            root / filename,
            value,
            label=f"final universe {key}",
        )
        _require(
            value.get("signal_dates") == count
            and value.get("minimum_signal_date") == first.isoformat()
            and value.get("maximum_signal_date") == last.isoformat(),
            f"final universe {key} scope differs",
        )
    return receipt


def _final_development_auxiliary_authenticator(
    path: Path,
    *,
    expected: Path,
    stage11_identity: Mapping[str, object],
) -> dict[str, str]:
    _require(path.resolve() == expected.resolve(), "Stage 11 auxiliary history differs")
    return {
        "build_id": "stage11-memoized-feature-input-bundle",
        "verification_id": "stage11-execution-inputs-verified",
        "manifest_sha256": str(
            stage11_identity["stage11_feature_input_receipt_sha256"]
        ),
    }


def _load_composed_final_feature_bundle(
    root: Path,
    *,
    final_contract: object,
) -> object:
    """Reauthenticate the composed 26-table catalog without Stage 11 literals."""

    from portfolio_bot.features.feature_store_inputs import (
        FeatureStoreInputs,
        InputTable,
    )
    from portfolio_bot.validation import validation_feature_input_adapter as adapter

    receipt_path = root / "validation_feature_input_bundle_receipt.json"
    receipt = _json(receipt_path, label="final feature-input receipt")
    fingerprint = str(receipt.get("input_bundle_sha256"))
    content = {
        key: value
        for key, value in receipt.items()
        if key not in {"input_bundle_sha256", "input_bundle_id"}
    }
    _require(
        _HEX64.fullmatch(fingerprint) is not None
        and _sha256_bytes(adapter._canonical_json(content)) == fingerprint,
        "final feature-input fingerprint differs",
    )
    catalog = receipt.get("scored_catalog")
    _require(
        isinstance(catalog, Mapping) and len(catalog) == 26,
        "final feature catalog differs",
    )
    scored: dict[str, InputTable] = {}
    for name, raw in catalog.items():
        _require(isinstance(raw, Mapping), f"final feature receipt differs: {name}")
        table = InputTable(
            gate=str(raw.get("gate")),
            name=str(name),
            path=Path(str(raw.get("path"))).resolve(),
            sha256=str(raw.get("sha256")),
            size_bytes=int(raw.get("size_bytes", -1)),
            rows=int(raw.get("rows", -1)),
            row_groups=int(raw.get("row_groups", -1)),
        )
        adapter._verify_parquet_receipt(
            table.path,
            {
                "sha256": table.sha256,
                "size_bytes": table.size_bytes,
                "rows": table.rows,
                "row_groups": table.row_groups,
            },
            label=f"final feature catalog {name}",
        )
        scored[str(name)] = table
    expected_signals = tuple(final_contract.scored_signal_dates)
    spine = pq.read_table(
        scored["crsp_signal_date_universe.parquet"].path,
        columns=["signal_date"],
        use_threads=False,
    )
    _require(
        tuple(sorted(set(spine["signal_date"].to_pylist()))) == expected_signals,
        "final scored feature spine differs",
    )
    scored_inputs = FeatureStoreInputs(
        adapter._common_real_root([value.path for value in scored.values()]),
        MappingProxyType(scored),
    )
    context = receipt.get("internal_optionmetrics_context")
    _require(
        isinstance(context, Mapping), "final OptionMetrics context receipt is absent"
    )
    working = dict(scored)
    for name, filename, hash_key in (
        (
            "crsp_signal_date_universe.parquet",
            "optionmetrics_working_signal_date_universe.parquet",
            "working_spine_sha256",
        ),
        (
            "auxiliary_signal_linkage.parquet",
            "optionmetrics_working_auxiliary_signal_linkage.parquet",
            "working_linkage_sha256",
        ),
    ):
        table = adapter._table_receipt(
            root / "tables" / filename,
            gate=scored[name].gate,
            name=name,
        )
        _require(
            table.sha256 == context.get(hash_key),
            f"final working table differs: {name}",
        )
        working[name] = table
    working_spine = pq.read_table(
        working["crsp_signal_date_universe.parquet"].path,
        columns=["signal_date"],
        use_threads=False,
    )
    _require(
        tuple(sorted(set(working_spine["signal_date"].to_pylist())))
        == (FINAL_CONTEXT_SIGNAL, *expected_signals),
        "final OptionMetrics working scope differs",
    )
    working_inputs = FeatureStoreInputs(
        adapter._common_real_root([value.path for value in working.values()]),
        MappingProxyType(working),
    )
    from portfolio_bot.validation.validation_feature_input_adapter import (
        ValidationFeatureInputBundle,
    )

    return ValidationFeatureInputBundle(
        scored_inputs=scored_inputs,
        optionmetrics_working_inputs=working_inputs,
        receipt_path=receipt_path,
        logical_history_root=root,
        input_bundle_sha256=fingerprint,
        source_receipts=MappingProxyType(
            {
                str(k): str(v)
                for k, v in dict(receipt.get("source_receipts", {})).items()
            }
        ),
    )


def _rewrite_final_feature_bundle_receipt(root: Path) -> None:
    """Replace Stage 11 receipt-only scope literals with truthful final scope.

    The adapter's table construction is reused unchanged, but its receipt
    template contains literal ``59``/``60`` and no-final-access statements.
    Those statements are metadata, not formulas, and would be false for the
    gated final replay.  Reissue the receipt over the same table identities
    with the actual 72/73 domain before it can become authority.
    """

    from portfolio_bot.validation import validation_feature_input_adapter as adapter

    path = root / "validation_feature_input_bundle_receipt.json"
    receipt = _json(path, label="final feature-input receipt template")
    if receipt.get("status") == "FINAL_VERIFIED_LOGICAL_FEATURE_HISTORY_COMPLETE":
        return
    receipt["status"] = "FINAL_VERIFIED_LOGICAL_FEATURE_HISTORY_COMPLETE"
    receipt["scope"] = {
        "internal_context_signal": FINAL_CONTEXT_SIGNAL.isoformat(),
        "first_scored_signal": FINAL_FIRST_SIGNAL.isoformat(),
        "last_scored_signal": FINAL_LAST_SIGNAL.isoformat(),
        "scored_signal_count": len(FINAL_HOLDING_MONTHS),
        "final_period_payload_accessed": True,
        "per_row_information_cutoff_enforced": True,
        "labels_forward_returns_weights_trades_pnl_rows_read": 0,
    }
    context = receipt.get("internal_optionmetrics_context")
    _require(isinstance(context, dict), "final OptionMetrics receipt is absent")
    context["signal_date"] = FINAL_CONTEXT_SIGNAL.isoformat()
    context["working_signal_count"] = len(FINAL_HOLDING_MONTHS) + 1
    receipt["source_authorizations"] = {
        "stage13_final_period_source_access": True,
        "model_fitting": False,
        "labels_or_portfolio_outcomes_in_predictor_branch": False,
    }
    receipt.pop("source_authorizations_remain_false", None)
    receipt.pop("input_bundle_sha256", None)
    receipt.pop("input_bundle_id", None)
    fingerprint = _sha256_bytes(adapter._canonical_json(receipt))
    receipt["input_bundle_sha256"] = fingerprint
    receipt["input_bundle_id"] = f"stage13-feature-inputs-{fingerprint[:24]}"
    _atomic_bytes(path, _canonical_json_bytes(receipt))


def _verify_composition_receipt(
    root: Path,
    *,
    receipt_name: str,
    table_directory: str | None = None,
) -> Mapping[str, object]:
    """Reauthenticate every receipted Parquet output of a reusable composition."""

    receipt_path = root / receipt_name
    receipt = _json(receipt_path, label=receipt_name)
    candidates: list[tuple[Path, Mapping[str, object]]] = []
    for container_name in ("outputs", "tables"):
        container = receipt.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for name, raw in container.items():
            if not isinstance(raw, Mapping) or "sha256" not in raw:
                continue
            relative = raw.get("path")
            if isinstance(relative, str):
                path = root / relative
            elif table_directory is not None:
                path = root / table_directory / str(name)
            else:
                continue
            candidates.append((path, raw))
    _require(bool(candidates), f"{receipt_name} has no reusable table identities")
    for path, raw in candidates:
        _require(
            path.is_file() and not path.is_symlink(),
            f"memoized table is absent: {path.name}",
        )
        parquet = pq.ParquetFile(path)
        _require(
            raw.get("sha256") == _sha256_file(path)
            and int(raw.get("size_bytes", -1)) == path.stat().st_size
            and (
                "rows" not in raw
                or int(raw.get("rows", -1)) == parquet.metadata.num_rows
            ),
            f"memoized composition table differs: {path.name}",
        )
    return MappingProxyType(receipt)


_SOURCE_TREE_DIGEST_METHOD = (
    "sha256_of_canonical_newline_JSON_sorted_relative_path_sha256_size_records"
)


def _stable_runtime_source_tree(
    root: Path,
    *,
    subroots: Sequence[str],
    ignored_directory_topologies: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Hash one closed prerequisite tree without following any symlink.

    The sealed migration document preserves the independently captured parent
    identity.  This separately named runtime digest proves that the same
    authenticated parent tree remains byte-identical before and after the
    fresh child-bundle composition; it is never presented as the historical
    capture algorithm.
    """

    selected = Path(os.path.abspath(root))
    metadata = selected.stat(follow_symlinks=False)
    _require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        "parent feature-source root is absent or unsafe",
    )
    expected_subroots = tuple(sorted(subroots))
    top_level = tuple(sorted(path.name for path in selected.iterdir()))
    _require(top_level == expected_subroots, "parent feature-source root is not closed")
    ignored = {
        str(relative): tuple(str(value) for value in descendants)
        for relative, descendants in ignored_directory_topologies.items()
    }
    _require(
        all(
            relative
            and not Path(relative).is_absolute()
            and Path(relative).as_posix() == relative
            and ".." not in Path(relative).parts
            and tuple(sorted(set(descendants))) == descendants
            and all(
                descendant
                and not Path(descendant).is_absolute()
                and Path(descendant).as_posix() == descendant
                and ".." not in Path(descendant).parts
                for descendant in descendants
            )
            for relative, descendants in ignored.items()
        ),
        "ignored parent scratch topology specification is unsafe",
    )
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )

    def directory_identity(path: Path, *, label: str) -> tuple[int, ...]:
        item = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISDIR(item.st_mode) and not stat.S_ISLNK(item.st_mode),
            f"ignored parent scratch entry is unsafe: {label}",
        )
        return tuple(int(getattr(item, field)) for field in identity_fields)

    def scan_ignored_topology(
        path: Path,
        *,
        relative: str,
        expected_descendants: Sequence[str],
    ) -> dict[str, tuple[int, ...]]:
        expected = tuple(expected_descendants)

        def scan_once() -> dict[str, tuple[int, ...]]:
            pending: list[tuple[str, Path]] = [("", path)]
            identities: dict[str, tuple[int, ...]] = {}
            observed: list[str] = []
            while pending:
                local, current = pending.pop()
                public = relative if not local else f"{relative}/{local}"
                identities[public] = directory_identity(current, label=public)
                try:
                    with os.scandir(current) as stream:
                        entries = sorted(stream, key=lambda item: item.name)
                except OSError as exc:
                    raise FinalSourceReplayError(
                        f"cannot scan ignored parent scratch: {public}"
                    ) from exc
                for entry in entries:
                    child_local = entry.name if not local else f"{local}/{entry.name}"
                    child_public = f"{relative}/{child_local}"
                    try:
                        item = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise FinalSourceReplayError(
                            f"cannot stat ignored parent scratch entry: {child_public}"
                        ) from exc
                    _require(
                        stat.S_ISDIR(item.st_mode) and not stat.S_ISLNK(item.st_mode),
                        f"ignored parent scratch contains a non-directory entry: {child_public}",
                    )
                    observed.append(child_local)
                    pending.append((child_local, Path(entry.path)))
            _require(
                tuple(sorted(observed)) == expected,
                f"ignored parent scratch topology differs: {relative}",
            )
            return identities

        first = scan_once()
        second = scan_once()
        _require(
            first == second,
            f"ignored parent scratch changed while scanning: {relative}",
        )
        return first

    seen_ignored: dict[str, tuple[int, ...]] = {}
    ignored_topology_receipts: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    file_identities: list[tuple[int, ...]] = []
    for current_text, directory_names, file_names in os.walk(
        selected, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        relative_current = current.relative_to(selected)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            path = current / name
            relative = (relative_current / name).as_posix()
            item = path.stat(follow_symlinks=False)
            _require(
                stat.S_ISDIR(item.st_mode) and not stat.S_ISLNK(item.st_mode),
                f"parent feature-source directory is unsafe: {relative}",
            )
            if relative in ignored:
                identities = scan_ignored_topology(
                    path,
                    relative=relative,
                    expected_descendants=ignored[relative],
                )
                seen_ignored.update(identities)
                topology_paths = [relative] + [
                    f"{relative}/{value}" for value in ignored[relative]
                ]
                ignored_topology_receipts[relative] = {
                    "exact_descendant_relative_paths": list(ignored[relative]),
                    "root_and_descendant_directory_count": len(topology_paths),
                    "regular_file_count": 0,
                    "payload_bytes": 0,
                    "symlink_count": 0,
                    "special_node_count": 0,
                    "topology_sha256": _sha256_bytes(
                        _canonical_json_bytes(topology_paths)
                    ),
                }
            else:
                kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            path = current / name
            relative = (relative_current / name).as_posix()
            before = path.stat(follow_symlinks=False)
            _require(
                stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
                f"parent feature-source file is unsafe: {relative}",
            )
            flags = (
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path, flags)
            digest = hashlib.sha256()
            try:
                opened = os.fstat(descriptor)
                while block := os.read(descriptor, 8 << 20):
                    digest.update(block)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            path_after = path.stat(follow_symlinks=False)
            _require(
                all(
                    getattr(before, field)
                    == getattr(opened, field)
                    == getattr(after, field)
                    == getattr(path_after, field)
                    for field in identity_fields
                ),
                f"parent feature-source file changed while hashing: {relative}",
            )
            records.append(
                {
                    "path": relative,
                    "sha256": digest.hexdigest(),
                    "size_bytes": int(after.st_size),
                }
            )
            file_identities.append(
                tuple(int(getattr(after, field)) for field in identity_fields)
            )
    expected_ignored_paths = {
        relative
        for root_relative, descendants in ignored.items()
        for relative in (
            root_relative,
            *(f"{root_relative}/{value}" for value in descendants),
        )
    }
    _require(
        set(seen_ignored) == expected_ignored_paths
        and set(ignored_topology_receipts) == set(ignored),
        "required parent scratch topology differs",
    )
    records.sort(key=lambda value: str(value["path"]))

    def summary(
        items: Sequence[Mapping[str, object]], *, prefix: str = ""
    ) -> dict[str, object]:
        canonical_records = [
            {
                "path": (
                    str(item["path"])[len(prefix) :] if prefix else str(item["path"])
                ),
                "sha256": str(item["sha256"]),
                "size_bytes": int(item["size_bytes"]),
            }
            for item in items
        ]
        return {
            "file_count": len(canonical_records),
            "total_size_bytes": sum(
                int(item["size_bytes"]) for item in canonical_records
            ),
            "runtime_tree_sha256": _sha256_bytes(
                _canonical_json_bytes(canonical_records)
            ),
            "runtime_tree_digest_method": _SOURCE_TREE_DIGEST_METHOD,
        }

    subroot_summaries: dict[str, object] = {}
    for name in expected_subroots:
        prefix = f"{name}/"
        members = [item for item in records if str(item["path"]).startswith(prefix)]
        subroot_summaries[name] = summary(members, prefix=prefix)
    return {
        "summary": summary(records),
        "subroots": subroot_summaries,
        "records": records,
        "file_identities": file_identities,
        "ignored_directory_identities": seen_ignored,
        "ignored_directory_identity_sha256": _sha256_bytes(
            _canonical_json_bytes(
                {key: list(value) for key, value in sorted(seen_ignored.items())}
            )
        ),
        "ignored_scratch_topologies": ignored_topology_receipts,
    }


def _source_tree_public_identity(snapshot: Mapping[str, object]) -> dict[str, object]:
    return {
        "summary": dict(snapshot["summary"]),
        "subroots": {
            str(key): dict(value) for key, value in dict(snapshot["subroots"]).items()
        },
        "ignored_directory_identity_sha256": str(
            snapshot["ignored_directory_identity_sha256"]
        ),
        "ignored_scratch_topologies": {
            str(key): dict(value)
            for key, value in dict(snapshot["ignored_scratch_topologies"]).items()
        },
    }


def _checkpoint_family_state(
    context_root: Path, *, context_sha256: str
) -> dict[str, int]:
    result: dict[str, int] = {}
    receipt_stems: dict[str, frozenset[str]] = {}
    receipts_root = context_root / "receipts"
    months_root = context_root / "months"
    if not receipts_root.exists() and not months_root.exists():
        return result
    _require(
        receipts_root.is_dir()
        and not receipts_root.is_symlink()
        and months_root.is_dir()
        and not months_root.is_symlink(),
        "checkpoint receipt/payload roots are unsafe or incomplete",
    )
    for family in sorted(receipts_root.iterdir(), key=lambda path: path.name):
        _require(
            family.is_dir() and not family.is_symlink(),
            "checkpoint receipt family is unsafe",
        )
        receipts = sorted(family.iterdir(), key=lambda path: path.name)
        for path in receipts:
            metadata = path.stat(follow_symlinks=False)
            _require(
                path.suffix == ".json"
                and stat.S_ISREG(metadata.st_mode)
                and not stat.S_ISLNK(metadata.st_mode)
                and metadata.st_nlink == 1,
                "checkpoint receipt is unsafe",
            )
            value = _json(path, label=f"checkpoint receipt {family.name}")
            _require(
                value.get("context_sha256") == context_sha256,
                f"checkpoint receipt context differs: {family.name}",
            )
        receipt_stems[family.name] = frozenset(path.stem for path in receipts)
        result[family.name] = len(receipts)
    payload_counts: dict[str, int] = {}
    payload_stems: dict[str, frozenset[str]] = {}
    for family in sorted(months_root.iterdir(), key=lambda path: path.name):
        _require(
            family.is_dir() and not family.is_symlink(),
            "checkpoint payload family is unsafe",
        )
        payloads = sorted(family.iterdir(), key=lambda path: path.name)
        _require(
            all(
                path.suffix == ".parquet"
                and stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
                and not stat.S_ISLNK(path.stat(follow_symlinks=False).st_mode)
                and path.stat(follow_symlinks=False).st_nlink == 1
                for path in payloads
            ),
            f"checkpoint payload family is unsafe: {family.name}",
        )
        payload_stems[family.name] = frozenset(path.stem for path in payloads)
        payload_counts[family.name] = len(payloads)
    _require(
        payload_counts == result and payload_stems == receipt_stems,
        "checkpoint receipt and payload families differ",
    )
    return result


def _expected_development_catalog_gates() -> dict[str, str]:
    """Return the exact frozen 26-table Stage 6 name-to-gate topology."""

    from portfolio_bot.validation import validation_feature_input_adapter as adapter

    result = {
        name: "auxiliary"
        for name in (
            set(adapter._auxiliary_specs())
            | set(adapter._FACTOR_TABLES)
            | {adapter._LINKAGE_TABLE}
        )
    }
    result.update(
        {
            adapter._CALENDAR_TABLE: "calendar",
            adapter._REBALANCE_TABLE: "calendar",
            "crsp_security_day.parquet": "typed_backbone",
            "crsp_identity_interval.parquet": "typed_backbone",
            "crsp_shares_interval.parquet": "typed_backbone",
            "crsp_security_day_return.parquet": "return_reconciliation",
            "crsp_delisting_event.parquet": "return_reconciliation",
            adapter._SPINE_TABLE: "signal_universe",
        }
    )
    _require(
        len(result) == 26
        and sum(gate == "auxiliary" for gate in result.values()) == 18,
        "frozen development feature catalog topology differs",
    )
    return result


def _parent_development_catalog(
    development_inputs: object,
    *,
    parent_root: Path,
) -> object:
    from portfolio_bot.features.feature_store_inputs import FeatureStoreInputs

    tables = dict(development_inputs.tables)
    expected_gates = _expected_development_catalog_gates()
    _require(
        set(tables) == set(expected_gates)
        and all(tables[name].gate == gate for name, gate in expected_gates.items()),
        "frozen development feature catalog topology differs",
    )
    rebound = 0
    for name, gate in expected_gates.items():
        if gate != "auxiliary":
            continue
        table = tables[name]
        target = parent_root / name
        _require(
            target.is_file()
            and not target.is_symlink()
            and target.stat().st_size == table.size_bytes
            and _sha256_file(target) == table.sha256,
            f"parent development auxiliary identity differs: {name}",
        )
        tables[name] = dataclasses.replace(table, path=target)
        rebound += 1
    _require(rebound == 18, "parent development auxiliary catalog differs")
    return FeatureStoreInputs(
        development_inputs.database_root, MappingProxyType(tables)
    )


def _migrated_parquet(path: Path, *, label: str) -> pq.ParquetFile:
    """Open one ordinary migrated Parquet while normalizing parser failures."""

    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise FinalSourceReplayError(f"{label} is absent or unsafe") from exc
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_nlink == 1,
        f"{label} is absent or unsafe",
    )
    try:
        return pq.ParquetFile(path)
    except Exception as exc:
        raise FinalSourceReplayError(f"{label} is not valid Parquet") from exc


def _validate_migrated_auxiliary_merge_provenance(
    *,
    adapter: object,
    bundle_root: Path,
    parent_root: Path,
    merges: Mapping[str, object],
) -> None:
    """Authenticate the exact 0+0 provenance marker and forbid scope drift."""

    metadata_key = adapter._ZERO_ROW_AUXILIARY_MERGE_METADATA_KEY
    provenance_version = adapter._ZERO_ROW_AUXILIARY_MERGE_PROVENANCE_VERSION
    for name, value in merges.items():
        _require(isinstance(value, Mapping), f"migrated merge receipt differs: {name}")
        raw = value
        output_path = bundle_root / "tables" / str(name)
        output = _migrated_parquet(
            output_path,
            label=f"migrated auxiliary merge output {name}",
        )
        output_metadata = dict(output.schema_arrow.metadata or {})
        development_rows = int(raw.get("development_rows", -1))
        extension_rows = int(raw.get("extension_rows", -1))
        is_zero_plus_zero = development_rows == 0 and extension_rows == 0
        provenance = raw.get("zero_row_materialization_provenance")
        if not is_zero_plus_zero:
            _require(
                "zero_row_materialization_provenance" not in raw
                and metadata_key not in output_metadata,
                f"migrated zero-row provenance escaped its exact scope: {name}",
            )
            continue

        development_path = parent_root / "development_auxiliary" / str(name)
        extension_path = parent_root / "auxiliary/tables" / str(name)
        development = _migrated_parquet(
            development_path,
            label=f"parent development auxiliary source {name}",
        )
        extension = _migrated_parquet(
            extension_path,
            label=f"parent extension auxiliary source {name}",
        )
        development_sha256 = _sha256_file(development_path)
        extension_sha256 = _sha256_file(extension_path)
        expected = {
            "schema_version": provenance_version,
            "materialization": "independent_zero_row_auxiliary_merge",
            "adapter_id": adapter.INPUT_ADAPTER_ID,
            "destination_table": str(name),
            "development_rows": 0,
            "extension_rows": 0,
            "development_sha256": development_sha256,
            "extension_sha256": extension_sha256,
            "merge_operator": "UNION ALL",
        }
        expected_bytes = adapter._canonical_json(expected)
        output_sha256 = _sha256_file(output_path)
        _require(
            development.metadata.num_rows == 0
            and extension.metadata.num_rows == 0
            and raw.get("merge_operator") == "UNION ALL"
            and raw.get("conflicting_primary_keys") == 0
            and int(raw.get("rows", -1)) == output.metadata.num_rows == 0
            and int(raw.get("row_groups", -1)) == output.metadata.num_row_groups == 0
            and raw.get("sha256") == output_sha256
            and int(raw.get("size_bytes", -1)) == output_path.stat().st_size
            and provenance == expected
            and output_metadata == {metadata_key: expected_bytes}
            and output.schema_arrow.remove_metadata().equals(
                development.schema_arrow.remove_metadata(),
                check_metadata=True,
            )
            and output_sha256 not in {development_sha256, extension_sha256},
            f"migrated zero-row provenance differs: {name}",
        )


def _validate_migrated_feature_bundle_receipt(
    *,
    bundle: object,
    bundle_root: Path,
    parent_root: Path,
    development_inputs: object,
    receipt_hashes: Mapping[str, str],
) -> dict[str, object]:
    from portfolio_bot.validation import validation_feature_input_adapter as adapter

    receipt = _json(bundle.receipt_path, label="migrated feature-input receipt")
    scope = receipt.get("scope")
    merges = receipt.get("auxiliary_history_merges")
    specifications = adapter._auxiliary_specs()
    _require(
        adapter.INPUT_ADAPTER_ID == "stage11-verified-feature-input-adapter-v3"
        and receipt.get("status") == "FINAL_VERIFIED_LOGICAL_FEATURE_HISTORY_COMPLETE"
        and receipt.get("adapter_id") == adapter.INPUT_ADAPTER_ID
        and scope
        == {
            "internal_context_signal": FINAL_CONTEXT_SIGNAL.isoformat(),
            "first_scored_signal": FINAL_FIRST_SIGNAL.isoformat(),
            "last_scored_signal": FINAL_LAST_SIGNAL.isoformat(),
            "scored_signal_count": len(FINAL_HOLDING_MONTHS),
            "final_period_payload_accessed": True,
            "per_row_information_cutoff_enforced": True,
            "labels_forward_returns_weights_trades_pnl_rows_read": 0,
        }
        and receipt.get("development_auxiliary_root")
        == (parent_root / "development_auxiliary").as_posix()
        and isinstance(merges, Mapping)
        and set(merges) == set(specifications),
        "migrated feature-input receipt scope or authority differs",
    )
    _validate_migrated_auxiliary_merge_provenance(
        adapter=adapter,
        bundle_root=bundle_root,
        parent_root=parent_root,
        merges=merges,
    )
    widened = 0
    development_rows = 0
    for name, raw in merges.items():
        _require(isinstance(raw, Mapping), f"migrated merge receipt differs: {name}")
        columns = raw.get("development_physical_nullability_widened_columns")
        per_column = raw.get("development_required_column_null_rows_by_column")
        _require(
            isinstance(columns, list)
            and all(isinstance(value, str) and value for value in columns)
            and len(columns) == len(set(columns))
            and columns
            == [
                field.name
                for field in specifications[str(name)][2]
                if not field.nullable
            ]
            and isinstance(per_column, Mapping)
            and set(per_column) == set(columns)
            and raw.get("development_required_column_null_rows") == 0
            and not any(int(value) for value in per_column.values()),
            f"migrated zero-null proof differs: {name}",
        )
        widened += len(columns)
        development_rows += int(raw.get("development_rows", -1))
    _require(
        widened == 106 and development_rows == 317047702,
        "migrated nullability diagnostic coverage differs",
    )
    source_receipts = receipt.get("source_receipts")
    _require(
        source_receipts
        == {
            "validation_universe_receipt_sha256": receipt_hashes["universe"],
            "validation_auxiliary_receipt_sha256": receipt_hashes["auxiliary"],
            "full_predictor_context_receipt_sha256": receipt_hashes[
                "full_predictor_context"
            ],
        },
        "migrated feature-input source receipts differ",
    )
    expected_gates = _expected_development_catalog_gates()
    expected_paths = {name: bundle_root / "tables" / name for name in specifications}
    expected_paths.update(
        {
            name: parent_root / "full_predictor_context/tables" / name
            for name in adapter._FULL_CONTEXT_TABLES
        }
    )
    expected_paths.update(
        {
            adapter._SPINE_TABLE: (
                parent_root / "universe" / adapter.SCORED_UNIVERSE_FILE
            ),
            adapter._LINKAGE_TABLE: (
                parent_root / "auxiliary/tables" / adapter.SCORED_LINKAGE_FILE
            ),
            adapter._CALENDAR_TABLE: (bundle_root / "tables" / adapter._CALENDAR_TABLE),
            adapter._REBALANCE_TABLE: (
                bundle_root / "tables" / adapter._REBALANCE_TABLE
            ),
            "crsp_delisting_event.parquet": development_inputs.path(
                "crsp_delisting_event.parquet"
            ),
        }
    )
    scored = bundle.scored_inputs.tables
    _require(
        set(expected_paths) == set(expected_gates) == set(scored),
        "migrated feature catalog table set differs",
    )
    for name, expected_path in expected_paths.items():
        observed_path = Path(scored[name].path)
        _require(
            not observed_path.is_symlink()
            and observed_path.resolve() == expected_path.resolve()
            and scored[name].gate == expected_gates[name],
            f"migrated feature catalog role/path differs: {name}",
        )
    return receipt


def _prepare_parent_context_migrated_feature_inputs(
    *,
    database_root: Path,
    work_root: Path,
    development_inputs: object,
    final_contract: object,
    replay_context: FinalReplayContext,
    project_root: Path,
) -> tuple[object, Mapping[str, object], Mapping[str, object]]:
    """Compose one fresh bundle from an authenticated, read-only parent tree."""

    from portfolio_bot.validation import validation_feature_input_adapter as adapter

    config_path = project_root / SOURCE_CONTEXT_MIGRATION_CONFIG
    _require(
        config_path.is_file()
        and not config_path.is_symlink()
        and _sha256_file(config_path)
        == EXPECTED_SOURCE_CONTEXT_MIGRATION_CONFIG_SHA256,
        "source checkpoint migration authority differs",
    )
    contract = _json(config_path, label="source checkpoint migration authority")
    parent = contract.get("parent_context")
    sources = contract.get("parent_feature_sources")
    receipts = contract.get("receipts")
    excluded = contract.get("excluded_parent_state")
    safety = contract.get("safety")
    _require(
        contract.get("status")
        == "STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_V2_AUTHORIZED"
        and contract.get("migration_id")
        == "stage13-source-checkpoint-context-migration-v2"
        and isinstance(parent, Mapping)
        and isinstance(sources, Mapping)
        and isinstance(receipts, Mapping)
        and isinstance(excluded, Mapping)
        and isinstance(safety, Mapping)
        and safety.get("parent_access_read_only") is True
        and safety.get("parent_tree_must_be_identical_before_after") is True
        and safety.get("migration_created_hardlinks_copies_symlinks_forbidden") is True
        and safety.get("stage12_tests_synthetic_only") is True,
        "source checkpoint migration contract differs",
    )
    _require(
        replay_context.raw_snapshot_id == parent.get("raw_snapshot_id")
        and replay_context.raw_file_identities_sha256
        == parent.get("raw_file_identities_sha256")
        and replay_context.stage12_binding_sha256
        != parent.get("stage12_binding_sha256")
        and replay_context.approval_payload_sha256
        != parent.get("approval_payload_sha256"),
        "replacement replay context is not an authorized parent transition",
    )
    selected_database_root = Path(os.path.abspath(database_root))
    database_metadata = selected_database_root.stat(follow_symlinks=False)
    _require(
        stat.S_ISDIR(database_metadata.st_mode)
        and not stat.S_ISLNK(database_metadata.st_mode),
        "final replay database root is unsafe",
    )
    namespace = database_root / WORK_NAMESPACE
    _ordinary_directory(namespace, root=database_root, label="final replay namespace")
    _require(
        namespace.is_dir() and not namespace.is_symlink(),
        "final replay namespace is absent or unsafe",
    )
    parent_context_root = _ordinary_directory(
        namespace / str(parent.get("context_sha256")),
        root=namespace,
        label="parent replay context root",
    )
    _require(
        parent_context_root.is_dir() and not parent_context_root.is_symlink(),
        "parent replay context root is absent or unsafe",
    )
    expected_work_root = (
        namespace / replay_context.context_sha256 / "formula_work/feature_sources"
    )
    selected_work_root = _ordinary_directory(
        work_root,
        root=namespace,
        label="replacement feature-source work root",
    )
    _require(
        selected_work_root == Path(os.path.abspath(expected_work_root)),
        "replacement feature-source work root differs",
    )
    parent_context_path = parent_context_root / str(
        parent.get("context_json_relative_path")
    )
    parent_context_document = _json(parent_context_path, label="parent replay context")
    _require(
        _sha256_file(parent_context_path) == parent.get("context_json_sha256")
        and parent_context_path.read_bytes()
        == _canonical_json_bytes(parent_context_document)
        and parent_context_document.get("context_sha256")
        == parent.get("context_sha256")
        and all(
            parent_context_document.get(key) == parent.get(key)
            for key in (
                "raw_snapshot_id",
                "raw_file_identities_sha256",
                "stage12_binding_sha256",
                "approval_payload_sha256",
            )
        ),
        "parent replay context differs",
    )
    parent_feature_root = _ordinary_directory(
        parent_context_root / str(sources.get("relative_path")),
        root=parent_context_root,
        label="parent feature-source root",
    )
    _require(
        parent_feature_root.is_dir() and not parent_feature_root.is_symlink(),
        "parent feature-source root is absent or unsafe",
    )
    subroots = tuple(
        str(value) for value in sources.get("allowed_prerequisite_directories", [])
    )
    raw_ignored = safety.get("ignored_recursively_payload_empty_scratch_topologies")
    _require(
        isinstance(raw_ignored, list)
        and safety.get("ignored_scratch_topology_must_match_exactly") is True
        and safety.get("ignored_scratch_regular_file_count_required") == 0
        and safety.get("ignored_scratch_payload_bytes_required") == 0
        and safety.get("ignored_scratch_symlink_count_required") == 0
        and safety.get("ignored_scratch_special_node_count_required") == 0
        and safety.get("ignored_scratch_directory_identities_before_after_must_match")
        is True
        and safety.get("ignored_scratch_entries_must_not_be_removed_or_mutated")
        is True,
        "source checkpoint scratch-topology contract differs",
    )
    ignored: dict[str, tuple[str, ...]] = {}
    for raw_topology in raw_ignored:
        _require(
            isinstance(raw_topology, Mapping)
            and set(raw_topology)
            == {"relative_path", "exact_descendant_relative_paths"}
            and isinstance(raw_topology.get("exact_descendant_relative_paths"), list),
            "source checkpoint scratch-topology specification differs",
        )
        relative = str(raw_topology["relative_path"])
        descendants = tuple(
            str(value) for value in raw_topology["exact_descendant_relative_paths"]
        )
        _require(
            relative not in ignored and tuple(sorted(set(descendants))) == descendants,
            "source checkpoint scratch-topology paths differ",
        )
        ignored[relative] = descendants
    parent_families = _checkpoint_family_state(
        parent_context_root, context_sha256=str(parent.get("context_sha256"))
    )
    _require(
        parent_families
        == {
            str(excluded.get("only_parent_month_receipt_family")): int(
                excluded.get("final_calendar_month_receipt_count", -1)
            )
        }
        and not (parent_context_root / "DECISION_SOURCES_COMMITTED.json").exists(),
        "excluded parent checkpoint state differs",
    )
    migration_path = work_root / SOURCE_CONTEXT_MIGRATION_RECEIPT
    _require(
        not migration_path.is_symlink(),
        "source checkpoint migration receipt is symlinked",
    )
    if migration_path.exists():
        migration_metadata = migration_path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(migration_metadata.st_mode)
            and migration_metadata.st_nlink == 1,
            "source checkpoint migration receipt is unsafe",
        )
    existing_migration = migration_path.is_file()
    replacement_families = _checkpoint_family_state(
        namespace / replay_context.context_sha256,
        context_sha256=replay_context.context_sha256,
    )
    if not existing_migration:
        _require(
            replacement_families
            == {
                "final_calendar": int(
                    excluded.get("final_calendar_month_receipt_count", -1)
                )
            }
            and not (
                namespace
                / replay_context.context_sha256
                / "DECISION_SOURCES_COMMITTED.json"
            ).exists(),
            "fresh replacement context checkpoint state differs",
        )
    before = _stable_runtime_source_tree(
        parent_feature_root,
        subroots=subroots,
        ignored_directory_topologies=ignored,
    )
    before_public = _source_tree_public_identity(before)
    _require(
        before_public["summary"]["file_count"] == sources.get("file_count")
        and before_public["summary"]["total_size_bytes"]
        == sources.get("total_size_bytes"),
        "parent feature-source tree size differs",
    )
    for name, expected in dict(sources.get("subroots", {})).items():
        observed = before_public["subroots"].get(name)
        _require(
            isinstance(observed, Mapping)
            and observed.get("file_count") == expected.get("file_count")
            and observed.get("total_size_bytes") == expected.get("total_size_bytes"),
            f"parent feature-source subroot differs: {name}",
        )
    receipt_hashes: dict[str, str] = {}
    for name, raw in receipts.items():
        _require(isinstance(raw, Mapping), f"parent receipt binding differs: {name}")
        path = parent_feature_root / str(raw.get("relative_path"))
        _require(
            path.is_file()
            and not path.is_symlink()
            and _sha256_file(path) == raw.get("sha256"),
            f"parent receipt differs: {name}",
        )
        receipt_hashes[str(name)] = str(raw.get("sha256"))
    work_root.mkdir(parents=True, exist_ok=True)
    allowed_child_entries = {"feature_input_bundle", SOURCE_CONTEXT_MIGRATION_RECEIPT}
    _require(
        {path.name for path in work_root.iterdir()} <= allowed_child_entries,
        "replacement feature-source work root has an unauthorized entry",
    )
    bundle_root = work_root / "feature_input_bundle"
    _require(not bundle_root.is_symlink(), "migrated feature bundle root is symlinked")
    bundle_receipt_path = bundle_root / "validation_feature_input_bundle_receipt.json"
    _require(
        not bundle_receipt_path.is_symlink(),
        "migrated feature bundle receipt is symlinked",
    )
    if bundle_root.exists():
        _require(bundle_root.is_dir(), "migrated feature bundle root is unsafe")
        if not bundle_receipt_path.is_file() and any(bundle_root.iterdir()):
            shutil.rmtree(bundle_root)
    if bundle_receipt_path.exists():
        bundle_receipt_metadata = bundle_receipt_path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(bundle_receipt_metadata.st_mode)
            and bundle_receipt_metadata.st_nlink == 1,
            "migrated feature bundle receipt is unsafe",
        )
    bundle: object | None = None
    post: dict[str, object] | None = None
    try:
        parent_development = _parent_development_catalog(
            development_inputs,
            parent_root=parent_feature_root / "development_auxiliary",
        )
        assignments = (
            (adapter, "FIRST_EXTENSION_AVAILABILITY", FINAL_EXTENSION_START),
            (adapter, "INTERNAL_CONTEXT_SIGNAL", FINAL_CONTEXT_SIGNAL),
            (adapter, "LAST_SIGNAL", FINAL_LAST_SIGNAL),
            (adapter, "LAST_PERMITTED_EXIT", FINAL_LAST_EXIT),
            (adapter, "_require_predictor_authority", lambda authority: authority),
            (adapter, "_authenticate_universe_root", _final_universe_receipt_validator),
        )
        with _temporary_assignments(assignments):
            if not bundle_receipt_path.is_file():
                adapter.compose_verified_validation_feature_inputs(
                    predictor_authority=final_contract,
                    validation_universe_root=parent_feature_root / "universe",
                    validation_auxiliary_root=parent_feature_root / "auxiliary",
                    full_predictor_context_root=parent_feature_root
                    / "full_predictor_context",
                    output_root=bundle_root,
                    development_inputs=parent_development,
                )
                _rewrite_final_feature_bundle_receipt(bundle_root)
            bundle = _load_composed_final_feature_bundle(
                bundle_root, final_contract=final_contract
            )
            feature_document = _validate_migrated_feature_bundle_receipt(
                bundle=bundle,
                bundle_root=bundle_root,
                parent_root=parent_feature_root,
                development_inputs=development_inputs,
                receipt_hashes=receipt_hashes,
            )
    finally:
        post = _stable_runtime_source_tree(
            parent_feature_root,
            subroots=subroots,
            ignored_directory_topologies=ignored,
        )
        _require(
            before == post,
            "parent feature-source tree or empty scratch changed during migration",
        )
    assert bundle is not None and post is not None
    parent_inodes = {
        (identity[0], identity[1]) for identity in before["file_identities"]
    }
    parent_hashes = {str(record["sha256"]) for record in before["records"]}
    for path in bundle_root.rglob("*"):
        item = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISDIR(item.st_mode) or stat.S_ISREG(item.st_mode),
            "migrated bundle contains a nonordinary entry",
        )
        if stat.S_ISREG(item.st_mode):
            _require(
                item.st_nlink == 1
                and (item.st_dev, item.st_ino) not in parent_inodes
                and _sha256_file(path) not in parent_hashes,
                "migrated bundle aliases or copies a parent prerequisite",
            )
    feature_receipt = MappingProxyType(
        {
            "stage11_history_identity": dict(
                _stable_stage11_history_identity(project_root)
            ),
            "universe_context_receipt_sha256": receipt_hashes["universe_context"],
            "universe_receipt_sha256": receipt_hashes["universe"],
            "auxiliary_receipt_sha256": receipt_hashes["auxiliary"],
            "full_predictor_context_receipt_sha256": receipt_hashes[
                "full_predictor_context"
            ],
            "feature_input_bundle_receipt_sha256": _sha256_file(bundle.receipt_path),
            "feature_input_bundle_sha256": bundle.input_bundle_sha256,
            "final_signal_count": len(final_contract.validation_signals),
            "stage11_artifacts_recomputed": False,
            "parent_prerequisites_reused_read_only": True,
        }
    )
    migration_receipt = {
        "schema_version": 1,
        "status": "STAGE13_SOURCE_CHECKPOINT_CONTEXT_MIGRATION_COMPLETE",
        "migration_id": contract["migration_id"],
        "migration_authority_sha256": EXPECTED_SOURCE_CONTEXT_MIGRATION_CONFIG_SHA256,
        "parent_context_sha256": parent["context_sha256"],
        "replacement_context_sha256": replay_context.context_sha256,
        "parent_captured_tree_sha256": sources["canonical_tree_sha256"],
        "runtime_parent_tree_before": before_public,
        "runtime_parent_tree_after": _source_tree_public_identity(post),
        "parent_receipts": receipt_hashes,
        "feature_input_bundle_receipt_sha256": _sha256_file(bundle.receipt_path),
        "feature_input_bundle_sha256": bundle.input_bundle_sha256,
        "adapter_id": adapter.INPUT_ADAPTER_ID,
        "nullability_transport_proof": {
            "merge_table_count": len(feature_document["auxiliary_history_merges"]),
            "widened_required_field_count": 106,
            "development_rows_covered": 317047702,
            "required_field_null_rows": 0,
        },
        "reuse_boundary": {
            "parent_access_read_only": True,
            "parent_final_calendar_receipts_reused": False,
            "new_context_final_calendar_recomputed": True,
            "predictor_outcome_or_economic_months_reused": 0,
            "migration_created_hardlinks_copies_or_symlinks": False,
        },
    }
    if existing_migration:
        observed = _json(migration_path, label="source checkpoint migration receipt")
        _require(
            migration_path.read_bytes() == _canonical_json_bytes(observed)
            and observed == migration_receipt,
            "source checkpoint migration receipt differs",
        )
    else:
        _atomic_bytes(migration_path, _canonical_json_bytes(migration_receipt))
    return bundle, feature_receipt, MappingProxyType(migration_receipt)


def _prepare_final_feature_inputs(
    *,
    database_root: Path,
    work_root: Path,
    development_inputs: object,
    stage11_contract: object,
    final_contract: object,
    final_calendar: pa.Table,
    final_market_root: Path,
    final_return_root: Path,
    stage11_identity: Mapping[str, object],
) -> tuple[object, Mapping[str, object]]:
    """Compose/reuse the exact logical history consumed by Stage 6 formulas.

    This adapter extends the authenticated Stage 11 input graph without
    changing any economic formula.  Expensive auxiliary family checkpoints
    are owned by ``build_validation_auxiliary_normalization`` and survive a
    retry; the outer composition receipts are reauthenticated before reuse.
    """

    from portfolio_bot.validation import validation_feature_input_adapter as adapter
    from portfolio_bot.validation import validation_universe_auxiliary as universe

    work_root.mkdir(parents=True, exist_ok=True)
    calendar_path = development_inputs.path("trading_calendar.parquet")
    stage11_context = database_root / STAGE11_FULL_CONTEXT_ROOT / "tables"
    _require(
        stage11_context.is_dir() and not stage11_context.is_symlink(),
        "Stage 11 full context is absent",
    )
    consolidated = _consolidated_development_catalog(
        development_inputs,
        destination=work_root / "development_auxiliary",
    )
    development_auxiliary_root = work_root / "development_auxiliary"
    context_root = work_root / "universe_context"
    universe_root = work_root / "universe"
    auxiliary_root = work_root / "auxiliary"
    full_context_root = work_root / "full_predictor_context"
    bundle_root = work_root / "feature_input_bundle"

    with _universe_auxiliary_scope() as scope:
        assignments = (
            (
                universe,
                "_authenticate_development_auxiliary_root",
                lambda path: _final_development_auxiliary_authenticator(
                    path,
                    expected=development_auxiliary_root,
                    stage11_identity=stage11_identity,
                ),
            ),
            (adapter, "FIRST_EXTENSION_AVAILABILITY", FINAL_EXTENSION_START),
            (adapter, "INTERNAL_CONTEXT_SIGNAL", FINAL_CONTEXT_SIGNAL),
            (adapter, "LAST_SIGNAL", FINAL_LAST_SIGNAL),
            (adapter, "LAST_PERMITTED_EXIT", FINAL_LAST_EXIT),
            (adapter, "_require_predictor_authority", lambda authority: authority),
            (adapter, "_authenticate_universe_root", _final_universe_receipt_validator),
        )
        with _temporary_assignments(assignments):
            context_receipt_path = context_root / "context_receipt.json"
            if context_receipt_path.is_file():
                context_receipt = _verify_composition_receipt(
                    context_root,
                    receipt_name="context_receipt.json",
                )
                universe_inputs = universe.ValidationUniverseInputs(
                    context_root / "typed",
                    context_root / "returns",
                    calendar_path,
                )
            else:
                if context_root.exists():
                    _quarantine_incomplete(context_root)
                universe_inputs, context_receipt = (
                    universe.compose_validation_universe_inputs(
                        development_typed_root=stage11_context,
                        development_return_root=stage11_context,
                        validation_market_root=final_market_root,
                        validation_return_root=final_return_root,
                        calendar_path=calendar_path,
                        output_root=context_root,
                    )
                )

            if (universe_root / universe.RECEIPT_FILE).is_file():
                universe_receipt = _final_universe_receipt_validator(universe_root)
            else:
                if universe_root.exists():
                    _quarantine_incomplete(universe_root)
                result = universe.compute_validation_universe(
                    universe_inputs,
                    scope=scope,
                )
                universe_receipt = universe.write_validation_universe(
                    result,
                    universe_root,
                )

            auxiliary_receipt = universe.build_validation_auxiliary_normalization(
                raw_root=database_root / "raw",
                development_auxiliary_root=development_auxiliary_root,
                trading_calendar_path=calendar_path,
                validation_identity_path=final_market_root
                / "crsp_identity_interval.parquet",
                scored_universe_path=universe_root / universe.SCORED_UNIVERSE_FILE,
                internal_context_universe_path=universe_root
                / universe.CONTEXT_UNIVERSE_FILE,
                output_root=auxiliary_root,
                scope=scope,
            )

            full_receipt_path = (
                full_context_root / "full_predictor_context_receipt.json"
            )
            if full_receipt_path.is_file():
                full_receipt = _verify_composition_receipt(
                    full_context_root,
                    receipt_name="full_predictor_context_receipt.json",
                    table_directory="tables",
                )
                adapter._authenticate_full_context(full_context_root)
            else:
                if full_context_root.exists():
                    _quarantine_incomplete(full_context_root)
                full_receipt = universe.compose_full_validation_predictor_context(
                    universe.FullPredictorContextInputs(
                        development_typed_root=stage11_context,
                        development_return_root=stage11_context,
                        development_auxiliary_root=development_auxiliary_root,
                        validation_market_root=final_market_root,
                        validation_return_root=final_return_root,
                        validation_auxiliary_root=auxiliary_root / "tables",
                        calendar_path=calendar_path,
                    ),
                    full_context_root,
                )

            if (bundle_root / "validation_feature_input_bundle_receipt.json").is_file():
                _rewrite_final_feature_bundle_receipt(bundle_root)
                bundle = _load_composed_final_feature_bundle(
                    bundle_root,
                    final_contract=final_contract,
                )
            else:
                if bundle_root.exists():
                    _quarantine_incomplete(bundle_root)
                adapter.compose_verified_validation_feature_inputs(
                    predictor_authority=final_contract,
                    validation_universe_root=universe_root,
                    validation_auxiliary_root=auxiliary_root,
                    full_predictor_context_root=full_context_root,
                    output_root=bundle_root,
                    development_inputs=consolidated,
                )
                _rewrite_final_feature_bundle_receipt(bundle_root)
                bundle = _load_composed_final_feature_bundle(
                    bundle_root,
                    final_contract=final_contract,
                )

    receipt = MappingProxyType(
        {
            "stage11_history_identity": dict(stage11_identity),
            "universe_context_receipt_sha256": _sha256_file(context_receipt_path),
            "universe_receipt_sha256": _sha256_file(
                universe_root / universe.RECEIPT_FILE
            ),
            "auxiliary_receipt_sha256": _sha256_file(
                auxiliary_root / universe.RECEIPT_FILE
            ),
            "full_predictor_context_receipt_sha256": _sha256_file(full_receipt_path),
            "feature_input_bundle_receipt_sha256": _sha256_file(bundle.receipt_path),
            "feature_input_bundle_sha256": bundle.input_bundle_sha256,
            "final_signal_count": len(final_contract.validation_signals),
            "stage11_artifacts_recomputed": False,
            "memoized_auxiliary_family_checkpoints": bool(
                auxiliary_receipt.get("checkpoint_receipts")
            ),
        }
    )
    return bundle, receipt


def _month_signal_map(calendar: pa.Table) -> Mapping[str, dt.date]:
    return MappingProxyType(
        {
            str(row["source_holding_month"]): row["signal_date"]
            for row in calendar.select(
                ["source_holding_month", "signal_date"]
            ).to_pylist()
        }
    )


def _signal_month_map(calendar: pa.Table) -> Mapping[dt.date, str]:
    return MappingProxyType(
        {value: key for key, value in _month_signal_map(calendar).items()}
    )


def _signal_lag(
    value: dt.date, months: int, available: Sequence[dt.date]
) -> dt.date | None:
    ordinal = value.year * 12 + value.month - 1 - months
    target = (ordinal // 12, ordinal % 12 + 1)
    return next(
        (item for item in available if (item.year, item.month) == target),
        None,
    )


def _filtered_input_table(
    source: Path,
    destination: Path,
    *,
    signal_dates: Sequence[dt.date],
) -> None:
    values = pa.array(tuple(signal_dates), type=pa.date32())
    table = pq.read_table(source, use_threads=False)
    _require(
        "signal_date" in table.column_names,
        f"filtered input lacks signal_date: {source.name}",
    )
    result = table.filter(
        pc.is_in(table["signal_date"], value_set=values)
    ).combine_chunks()
    observed = tuple(sorted(set(result["signal_date"].to_pylist())))
    _require(
        observed == tuple(sorted(signal_dates)),
        f"filtered input signal scope differs: {source.name}",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    _require(
        not destination.exists() and not temporary.exists(),
        "filtered feature input already exists",
    )
    try:
        pq.write_table(
            result,
            temporary,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
            row_group_size=131_072,
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _subset_final_feature_bundle(
    bundle: object,
    *,
    pending_months: tuple[str, ...],
    final_calendar: pa.Table,
    work_root: Path,
) -> tuple[object, tuple[dt.date, ...], frozenset[dt.date], Mapping[str, object]]:
    """Create a content-addressed spine containing only unfinished months.

    The OptionMetrics working spine additionally retains only the one- and
    three-month lag signals needed by the frozen change formulas.  Those rows
    remain internal to that family and are filtered before cross-family
    alignment, so completed predictor months are never rematerialized.
    """

    from portfolio_bot.features.feature_store_inputs import FeatureStoreInputs
    from portfolio_bot.validation import validation_feature_input_adapter as adapter
    from portfolio_bot.validation.validation_feature_input_adapter import (
        ValidationFeatureInputBundle,
    )

    _require(
        type(bundle) is ValidationFeatureInputBundle,
        "final feature bundle type differs",
    )
    month_to_signal = _month_signal_map(final_calendar)
    pending_signals = tuple(month_to_signal[month] for month in pending_months)
    pending_set = frozenset(pending_signals)
    _require(bool(pending_signals), "pending predictor scope is empty")
    full_working_spine = pq.read_table(
        bundle.optionmetrics_working_inputs.path("crsp_signal_date_universe.parquet"),
        columns=["signal_date"],
        use_threads=False,
    )
    all_working_signals = tuple(
        sorted(set(full_working_spine["signal_date"].to_pylist()))
    )
    option_signals = set(pending_signals)
    for signal in pending_signals:
        for lag in (1, 3):
            prior = _signal_lag(signal, lag, all_working_signals)
            if prior is not None:
                option_signals.add(prior)
    ordered_option_signals = tuple(sorted(option_signals))
    identity = _sha256_bytes(
        _canonical_json_bytes(
            {
                "input_bundle_sha256": bundle.input_bundle_sha256,
                "pending_months": list(pending_months),
                "option_signals": [
                    value.isoformat() for value in ordered_option_signals
                ],
            }
        )
    )
    root = work_root / identity
    receipt_path = root / "subset_receipt.json"
    specifications = (
        (
            "scored_spine",
            bundle.scored_inputs.path("crsp_signal_date_universe.parquet"),
            root / "scored" / "crsp_signal_date_universe.parquet",
            pending_signals,
        ),
        (
            "scored_linkage",
            bundle.scored_inputs.path("auxiliary_signal_linkage.parquet"),
            root / "scored" / "auxiliary_signal_linkage.parquet",
            pending_signals,
        ),
        (
            "working_spine",
            bundle.optionmetrics_working_inputs.path(
                "crsp_signal_date_universe.parquet"
            ),
            root / "working" / "crsp_signal_date_universe.parquet",
            ordered_option_signals,
        ),
        (
            "working_linkage",
            bundle.optionmetrics_working_inputs.path(
                "auxiliary_signal_linkage.parquet"
            ),
            root / "working" / "auxiliary_signal_linkage.parquet",
            ordered_option_signals,
        ),
    )
    if receipt_path.is_file():
        receipt = _json(receipt_path, label="feature subset receipt")
        _require(
            receipt.get("subset_sha256") == identity, "feature subset identity differs"
        )
        files = receipt.get("files")
        _require(isinstance(files, Mapping), "feature subset files are absent")
        for label, _source, destination, signals in specifications:
            raw = files.get(label)
            _require(
                isinstance(raw, Mapping)
                and destination.is_file()
                and not destination.is_symlink()
                and _sha256_file(destination) == raw.get("sha256")
                and tuple(
                    sorted(
                        set(
                            pq.read_table(
                                destination,
                                columns=["signal_date"],
                                use_threads=False,
                            )["signal_date"].to_pylist()
                        )
                    )
                )
                == tuple(sorted(signals)),
                f"memoized feature subset differs: {label}",
            )
    else:
        if root.exists():
            _quarantine_incomplete(root)
        root.mkdir(parents=True)
        files: dict[str, object] = {}
        for label, source, destination, signals in specifications:
            _filtered_input_table(source, destination, signal_dates=signals)
            parquet = pq.ParquetFile(destination)
            files[label] = {
                "path": destination.relative_to(root).as_posix(),
                "sha256": _sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.metadata.num_row_groups,
                "signal_count": len(signals),
            }
        receipt = {
            "schema_version": 1,
            "status": "FINAL_FEATURE_SUBSET_COMPLETE",
            "subset_sha256": identity,
            "input_bundle_sha256": bundle.input_bundle_sha256,
            "pending_months": list(pending_months),
            "option_signals": [value.isoformat() for value in ordered_option_signals],
            "files": files,
        }
        _atomic_bytes(receipt_path, _canonical_json_bytes(receipt))

    scored_tables = dict(bundle.scored_inputs.tables)
    working_tables = dict(bundle.optionmetrics_working_inputs.tables)
    for name in (
        "crsp_signal_date_universe.parquet",
        "auxiliary_signal_linkage.parquet",
    ):
        scored_tables[name] = adapter._table_receipt(
            root / "scored" / name,
            gate=scored_tables[name].gate,
            name=name,
        )
        working_tables[name] = adapter._table_receipt(
            root / "working" / name,
            gate=working_tables[name].gate,
            name=name,
        )
    scored_inputs = FeatureStoreInputs(
        bundle.scored_inputs.database_root,
        MappingProxyType(scored_tables),
    )
    working_inputs = FeatureStoreInputs(
        bundle.optionmetrics_working_inputs.database_root,
        MappingProxyType(working_tables),
    )
    subset = ValidationFeatureInputBundle(
        scored_inputs=scored_inputs,
        optionmetrics_working_inputs=working_inputs,
        receipt_path=receipt_path,
        logical_history_root=root,
        input_bundle_sha256=identity,
        source_receipts=MappingProxyType(
            {"parent_input_bundle_sha256": bundle.input_bundle_sha256}
        ),
    )
    return subset, pending_signals, pending_set, MappingProxyType(receipt)


def _materialize_final_predictors(
    *,
    bundle: object,
    final_contract: object,
    final_calendar: pa.Table,
    pending_months: tuple[str, ...],
    work_root: Path,
) -> tuple[pa.Table, Mapping[str, object]]:
    """Replay the seven frozen Stage 6 family implementations for pending rows."""

    from portfolio_bot.features.feature_store_materialization import (
        compute_feature_store_build_identity,
    )
    from portfolio_bot.validation import validation_data_candidate_builder as candidate
    from portfolio_bot.validation import validation_empirical_feature_stream as stream
    from portfolio_bot.validation.validation_fundamental_adapter import (
        select_validation_fundamental_bundle,
        validate_validation_fundamental_source_gates,
    )

    subset, pending_signals, pending_set, subset_receipt = _subset_final_feature_bundle(
        bundle,
        pending_months=pending_months,
        final_calendar=final_calendar,
        work_root=work_root / "input_subsets",
    )
    scored = subset.scored_inputs
    working = subset.optionmetrics_working_inputs
    scored_rows = scored.tables["crsp_signal_date_universe.parquet"].rows
    working_rows = working.tables["crsp_signal_date_universe.parquet"].rows
    option_dates = tuple(
        sorted(
            set(
                pq.read_table(
                    working.path("crsp_signal_date_universe.parquet"),
                    columns=["signal_date"],
                    use_threads=False,
                )["signal_date"].to_pylist()
            )
        )
    )
    first_signal = min(pending_signals)
    last_signal = max(pending_signals)
    assignments = (
        (stream.crsp_adapter, "DEVELOPMENT_LAST_SIGNAL", last_signal),
        (stream.crsp_adapter, "MAXIMUM_CONTEXT_DATE", last_signal),
        (stream.crsp_streaming, "DEVELOPMENT_LAST_SIGNAL", last_signal),
        (stream.crsp_streaming, "MAXIMUM_CONTEXT_DATE", last_signal),
        (stream.fundamental_adapter, "FIRST_SIGNAL_DATE", first_signal),
        (stream.fundamental_adapter, "LAST_SIGNAL_DATE", last_signal),
        (stream.fundamental_adapter, "MAXIMUM_CONTEXT_DATE", last_signal),
        (stream.fundamental_adapter, "EXPECTED_SPINE_ROWS", scored_rows),
        (stream.fundamental_adapter, "EXPECTED_SIGNAL_DATES", len(pending_signals)),
        (
            stream.fundamental_adapter,
            "_validate_source_gates",
            validate_validation_fundamental_source_gates,
        ),
        (
            stream.fundamental_adapter,
            "select_fundamental_bundle",
            select_validation_fundamental_bundle,
        ),
        (
            stream.ownership_adapter,
            "select_fundamental_bundle",
            select_validation_fundamental_bundle,
        ),
        (stream.ownership_adapter, "FIRST_SIGNAL_DATE", first_signal),
        (stream.ownership_adapter, "LAST_SIGNAL_DATE", last_signal),
        (stream.optionmetrics_adapter, "FIRST_SIGNAL_DATE", min(option_dates)),
        (stream.optionmetrics_adapter, "LAST_SIGNAL_DATE", last_signal),
        (stream.optionmetrics_adapter, "EXPECTED_SPINE_ROWS", working_rows),
        (stream.optionmetrics_adapter, "EXPECTED_SIGNAL_DATES", len(option_dates)),
        (stream.aggregate_adapter, "PREDICTOR_END_DATE", last_signal),
        (stream.aggregate_adapter, "CONTEXT_END_DATE", last_signal),
    )
    with stream._formula_scope(assignments):
        scored_identity = compute_feature_store_build_identity(scored)
        option_identity = stream._derived_identity(scored_identity, working)

        def crsp_source(checkpoint: tuple[int, dt.date] | None):
            return stream.iter_unfinalized_crsp_rows(
                scored,
                start_after_key=checkpoint,
            )

        router = stream.spool_crsp_signal_router(
            crsp_source,
            spool_path=work_root / "crsp_router" / subset.input_bundle_sha256,
            source_id=f"{REPLAY_ID}:{scored_identity.content_fingerprint}",
            expected_rows=scored_rows,
            checkpoint_rows=4_096,
        )
        fundamentals = stream._flatten(
            stream.iter_fundamental_feature_batches(
                scored,
                batch_size=2_048,
                build_identity=scored_identity,
            )
        )
        ownership = stream._flatten(
            stream.iter_ownership_event_feature_batches(
                scored,
                batch_size=2_048,
                build_identity=scored_identity,
            )
        )
        option_all = stream._flatten(
            stream.iter_optionmetrics_feature_batches(
                working,
                batch_size=2_048,
                build_identity=option_identity,
            )
        )
        option_scored = (row for row in option_all if row.signal_date in pending_set)
        aggregate = stream._flatten(
            stream.materialize_aggregate_state(
                scored,
                batch_size=2_048,
            ).iter_materialization_row_batches()
        )
        chunks: list[pa.Table] = []
        for rows in stream._aligned_scored_batches(
            router=router,
            fundamental_rows=fundamentals,
            ownership_rows=ownership,
            optionmetrics_rows=option_scored,
            aggregate_rows=aggregate,
            batch_size=2_048,
        ):
            predictor, _lineage, _dates, _long, _segment = (
                candidate._materialize_feature_chunk(
                    rows,
                    final_contract,
                    allowed_signal_dates=pending_set,
                )
            )
            chunks.append(predictor)
    _require(bool(chunks), "final predictor formula emitted no rows")
    predictor = pa.concat_tables(chunks).combine_chunks()
    signal_to_month = _signal_month_map(final_calendar)
    month_values = pa.array(
        [signal_to_month[value] for value in predictor["signal_date"].to_pylist()],
        type=pa.string(),
    )
    native = predictor.add_column(0, "source_holding_month", month_values)
    native = native.cast(_native_schema("final_predictors"), safe=True)
    _require(
        tuple(sorted(set(native["source_holding_month"].to_pylist())))
        == tuple(sorted(pending_months)),
        "final predictor month scope differs",
    )
    return native, MappingProxyType(
        {
            "subset_sha256": subset.input_bundle_sha256,
            "subset_receipt_sha256": _sha256_file(subset.receipt_path),
            "scored_build_identity": scored_identity.content_fingerprint,
            "optionmetrics_build_identity": option_identity.content_fingerprint,
            "crsp_router_reused": router.reused_existing,
            "pending_month_count": len(pending_months),
            "passed_months_recomputed": 0,
            "rows": native.num_rows,
        }
    )


def _native_month_axis(
    name: str,
    table: pa.Table,
    *,
    final_calendar: pa.Table,
) -> pa.Table:
    signal_to_month = _signal_month_map(final_calendar)
    _require("signal_date" in table.column_names, f"{name} lacks signal_date")
    months = pa.array(
        [signal_to_month[value] for value in table["signal_date"].to_pylist()],
        type=pa.string(),
    )
    return table.add_column(0, "source_holding_month", months).cast(
        _native_schema(name),
        safe=True,
    )


def _final_risk_history_inputs(
    *,
    bundle: object,
    cutoff: dt.date,
) -> tuple[object, Mapping[str, int]]:
    from portfolio_bot.validation import validation_data_candidate_builder as candidate

    calendar_source = pq.read_table(
        bundle.scored_inputs.path("trading_calendar.parquet"),
        columns=["calendar_date"],
        filters=[("calendar_date", "<=", cutoff)],
        use_threads=False,
    ).combine_chunks()
    factor_source = pq.read_table(
        bundle.scored_inputs.path("ff_factor_daily_observation.parquet"),
        columns=list(candidate.FACTOR_COLUMNS),
        use_threads=False,
    ).combine_chunks()
    factor_mask = pc.and_(
        pc.less_equal(
            factor_source["observation_date"], pa.scalar(cutoff, pa.date32())
        ),
        pc.less_equal(
            factor_source["public_availability_date"],
            pa.scalar(cutoff, pa.date32()),
        ),
    )
    factors = factor_source.filter(pc.fill_null(factor_mask, False)).combine_chunks()
    return_path = bundle.scored_inputs.path("crsp_security_day_return.parquet")
    connection = duckdb.connect(":memory:")
    try:
        delivered_rows, final_rows = connection.execute(
            f"""
            SELECT count(*), count(*) FILTER (
                WHERE calendar_date >= DATE '2020-01-01'
            )
            FROM read_parquet('{return_path.as_posix().replace("'", "''")}')
            WHERE calendar_date <= DATE '{cutoff.isoformat()}'
              AND return_span_end_date <= DATE '{cutoff.isoformat()}'
              AND outcome_availability_date <= DATE '{cutoff.isoformat()}'
            """
        ).fetchone()
    finally:
        connection.close()
    delivered_rows = int(delivered_rows)
    final_rows = int(final_rows)

    def factory(batch_rows: int) -> Iterable[pa.RecordBatch]:
        parquet = pq.ParquetFile(return_path)
        for batch in parquet.iter_batches(
            batch_size=batch_rows,
            columns=list(candidate.RETURN_COLUMNS),
            use_threads=False,
        ):
            table = pa.Table.from_batches([batch])
            mask = pc.and_kleene(
                pc.and_kleene(
                    pc.less_equal(
                        table["calendar_date"], pa.scalar(cutoff, pa.date32())
                    ),
                    pc.less_equal(
                        table["return_span_end_date"], pa.scalar(cutoff, pa.date32())
                    ),
                ),
                pc.less_equal(
                    table["outcome_availability_date"],
                    pa.scalar(cutoff, pa.date32()),
                ),
            )
            selected = table.filter(pc.fill_null(mask, False)).combine_chunks()
            if selected.num_rows:
                yield from selected.to_batches(max_chunksize=batch_rows)

    inputs = candidate.ValidationRiskHistoryInputs(
        trading_calendar=calendar_source,
        factors=factors,
        return_batch_factory=factory,
        label_rows_read=0,
        outcome_rows_read=0,
        final_period_rows_read=final_rows,
    )
    counters: dict[str, int] = {
        "delivered_return_rows": delivered_rows,
        "final_period_lagged_return_rows": final_rows,
        "factor_rows": factors.num_rows,
    }
    return inputs, counters


def _materialize_final_risk_tables(
    *,
    predictors: pa.Table,
    bundle: object,
    final_contract: object,
    final_calendar: pa.Table,
    pending_months: tuple[str, ...],
) -> tuple[Mapping[str, pa.Table], Mapping[str, object]]:
    """Run the frozen Stage 8 numerics with a dynamic, cutoff-safe scope."""

    from portfolio_bot.validation import validation_data_candidate_builder as candidate

    pending_set = frozenset(pending_months)
    predictor = predictors.filter(
        pc.is_in(
            predictors["source_holding_month"],
            value_set=pa.array(tuple(pending_set), pa.string()),
        )
    ).drop(["source_holding_month"])
    feature_spine = predictor.select(list(candidate.RISK_SPINE_COLUMNS))
    order = pc.sort_indices(
        feature_spine,
        sort_keys=[("signal_date", "ascending"), ("permno", "ascending")],
    )
    spine = feature_spine.take(order).combine_chunks()
    signals = candidate.frozen_stage8._date_days(spine["signal_date"])
    permnos = candidate.frozen_stage8._int_values(spine["permno"], np.int64)
    _require(
        len(signals) > 0
        and not np.any((signals[1:] == signals[:-1]) & (permnos[1:] == permnos[:-1])),
        "final risk spine keys are empty or duplicate",
    )
    unique_signals = np.unique(signals)
    month_to_signal = _month_signal_map(final_calendar)
    expected_days = np.asarray(
        [
            (month_to_signal[month] - dt.date(1970, 1, 1)).days
            for month in pending_months
        ],
        dtype=np.int32,
    )
    _require(
        np.array_equal(unique_signals, np.sort(expected_days)),
        "final risk signal scope differs",
    )
    cutoff = max(month_to_signal[month] for month in pending_months)
    inputs, risk_source_counters = _final_risk_history_inputs(
        bundle=bundle,
        cutoff=cutoff,
    )
    long_mask = pc.fill_null(spine["long_eligible"], False).to_numpy(
        zero_copy_only=False
    )
    eligible_indices = np.flatnonzero(long_mask)
    _require(eligible_indices.size > 0, "final risk scope has no long-eligible row")
    eligible_signals = signals[eligible_indices]
    eligible_permnos = permnos[eligible_indices]
    state = candidate.frozen_stage8._new_state(
        eligible_signals,
        eligible_permnos,
    )
    factor_context = candidate.frozen_stage8._factor_context(inputs, unique_signals)
    targets = candidate.frozen_stage8._target_map(eligible_permnos)
    with _temporary_assignments(((candidate, "LAST_SIGNAL", cutoff),)):
        delivered, used = candidate._scope_aware_return_scan(
            inputs,
            targets,
            state,
            factor_context,
            batch_rows=262_144,
        )
    eligible_spine = spine.take(pa.array(eligible_indices, type=pa.int64()))
    month_rows = candidate.frozen_stage8._finalize_cross_sections(
        state,
        sectors=eligible_spine["ff12_sector_category"].to_pylist(),
        short_eligible=pc.fill_null(eligible_spine["short_eligible"], False).to_numpy(
            zero_copy_only=False
        ),
        spreads=candidate.frozen_stage8._float_values(
            eligible_spine["median_estimated_full_spread_20d"]
        ),
        adv=candidate.frozen_stage8._float_values(
            eligible_spine["median_dollar_volume_20d"]
        ),
        factors=factor_context,
    )
    audit = candidate.frozen_stage8._build_audit_table(
        spine,
        eligible_indices,
        state,
    )
    covariance = candidate.frozen_stage8._factor_covariance_table(
        unique_signals,
        factor_context,
    )
    completed = candidate.frozen_stage8._complete_month_rows(
        signals,
        month_rows,
        factor_context,
    )
    summary = pa.Table.from_pylist(
        completed,
        schema=final_contract.payload_specs[
            "validation_risk_cost_month_summary"
        ].schema,
    )
    result = MappingProxyType(
        {
            "final_risk_cost": _native_month_axis(
                "final_risk_cost",
                audit,
                final_calendar=final_calendar,
            ),
            "final_factor_covariance": _native_month_axis(
                "final_factor_covariance",
                covariance,
                final_calendar=final_calendar,
            ),
            "final_risk_cost_month_summary": _native_month_axis(
                "final_risk_cost_month_summary",
                summary,
                final_calendar=final_calendar,
            ),
        }
    )
    _require(
        result["final_risk_cost"].num_rows == spine.num_rows
        and result["final_factor_covariance"].num_rows == len(pending_months) * 36
        and result["final_risk_cost_month_summary"].num_rows == len(pending_months),
        "final Stage 8 output cardinality differs",
    )
    return result, MappingProxyType(
        {
            "pending_month_count": len(pending_months),
            "risk_audit_rows": audit.num_rows,
            "factor_covariance_rows": covariance.num_rows,
            "month_summary_rows": summary.num_rows,
            "return_rows_delivered": delivered,
            "return_rows_precounted": risk_source_counters["delivered_return_rows"],
            "final_period_lagged_return_rows": risk_source_counters[
                "final_period_lagged_return_rows"
            ],
            "eligible_security_history_rows_processed": used,
            "per_signal_information_cutoff_enforced": True,
            "passed_months_recomputed": 0,
            "frozen_stage8_numerics_reused": True,
        }
    )


def _materialize_final_market_inputs(
    *,
    final_calendar: pa.Table,
    predictors: pa.Table,
    risk_cost: pa.Table,
    typed_market_root: Path,
    entry_market_root: Path,
    pending_months: tuple[str, ...],
) -> tuple[pa.Table, Mapping[str, object]]:
    """Join frozen Stage 2 entry-day facts to predictor/risk decision rows."""

    pending = pa.array(pending_months, pa.string())
    calendar = final_calendar.filter(
        pc.is_in(final_calendar["source_holding_month"], value_set=pending)
    )
    # Predictor/risk history intentionally remains full.  On resume the first
    # pending decision month must still recover a last-known row from a
    # memoized passed month for an inherited holding.  The SQL cutoff below is
    # the only membership boundary and excludes every future signal.
    predictor = predictors
    risk = risk_cost
    market_path = typed_market_root / "crsp_security_day.parquet"
    shares_path = typed_market_root / "crsp_shares_interval.parquet"
    entry_market_path = entry_market_root / "crsp_security_day.parquet"
    entry_shares_path = entry_market_root / "crsp_shares_interval.parquet"
    _require(
        market_path.is_file()
        and shares_path.is_file()
        and entry_market_path.is_file()
        and entry_shares_path.is_file()
        and not market_path.is_symlink()
        and not shares_path.is_symlink()
        and not entry_market_path.is_symlink()
        and not entry_shares_path.is_symlink(),
        "final entry-market sources are absent",
    )
    connection = duckdb.connect(":memory:")
    try:
        connection.register("calendar", calendar)
        connection.register("predictor", predictor)
        connection.register("risk", risk)
        query = f"""
            WITH market AS (
                SELECT *, 'signal_history' AS entry_market_scope
                FROM read_parquet('{market_path.as_posix().replace("'", "''")}')
                UNION ALL
                SELECT *, 'partial_entry' AS entry_market_scope
                FROM read_parquet('{entry_market_path.as_posix().replace("'", "''")}')
            ),
            shares AS (
                SELECT *, 'signal_history' AS entry_market_scope
                FROM read_parquet('{shares_path.as_posix().replace("'", "''")}')
                UNION ALL
                SELECT *, 'partial_entry' AS entry_market_scope
                FROM read_parquet('{entry_shares_path.as_posix().replace("'", "''")}')
            )
            SELECT
                c.source_holding_month,
                c.signal_date,
                c.entry_date,
                c.exit_date,
                m.permno,
                m.source_permco,
                m.source_security_info_start_date,
                m.source_security_info_end_date,
                m.source_cusip,
                m.source_primary_exchange,
                m.conditional_type,
                m.source_trading_status_flag,
                m.source_us_incorporation_flag,
                m.source_issuer_type,
                m.source_security_type,
                m.source_security_subtype,
                m.source_share_type,
                m.source_ticker,
                m.source_sic_code,
                m.source_naics_code,
                m.source_icb_industry,
                m.calendar_date,
                m.diagnostic_price_usd_per_share,
                m.price_flag,
                m.market_cap_thousand_usd,
                m.market_cap_usd,
                m.volume_shares,
                m.close_usd_per_share,
                m.low_usd_per_share,
                m.high_usd_per_share,
                m.bid_usd_per_share,
                m.ask_usd_per_share,
                m.open_usd_per_share,
                m.vendor_price_volume_usd,
                m.screening_dollar_volume_usd,
                m.screening_price_valid,
                m.market_cap_valid,
                m.liquidity_session_valid,
                s.start_date AS shares_interval_start_date,
                s.end_date AS shares_interval_end_date,
                s.shares_outstanding_thousand,
                s.shares_outstanding,
                s.shares_source,
                s.shares_factor_type,
                s.shares_adr_flag,
                p.permco,
                p.long_eligible,
                p.short_eligible,
                r.information_cutoff_date,
                r.full_spread_20d,
                r.adv20_usd,
                r.impact_volatility_daily,
                r.impact_volatility_source,
                r.impact_volatility_observations,
                r.primary_borrow_rate_annual,
                r.stress_borrow_rate_annual,
                r.new_long_trade_allowed,
                r.new_short_trade_allowed,
                r.cost_input_status,
                CASE
                    WHEN m.close_usd_per_share IS NULL
                      OR m.close_usd_per_share <= 0
                    THEN 'MISSING_VALID_ENTRY_CLOSE'
                    ELSE 'PRESENT'
                END AS entry_close_status
            FROM calendar c
            INNER JOIN predictor p
              ON p.signal_date <= c.signal_date
             AND p.long_eligible IS TRUE
            INNER JOIN risk r
              ON r.signal_date = p.signal_date
             AND r.permno = p.permno
            INNER JOIN market m
              ON m.permno = p.permno
             AND m.calendar_date = c.entry_date
             AND m.entry_market_scope = CASE
                    WHEN c.source_holding_month = '{PARTIAL_DECEMBER_HOLDING_MONTH}'
                    THEN 'partial_entry'
                    ELSE 'signal_history'
                 END
            INNER JOIN shares s
              ON s.permno = p.permno
             AND c.entry_date BETWEEN s.start_date AND s.end_date
             AND s.entry_market_scope = CASE
                    WHEN c.source_holding_month = '{PARTIAL_DECEMBER_HOLDING_MONTH}'
                    THEN 'partial_entry'
                    ELSE 'signal_history'
                 END
            QUALIFY row_number() OVER (
                PARTITION BY c.source_holding_month, p.permno
                ORDER BY p.signal_date DESC
            ) = 1
            ORDER BY c.signal_date, m.permno
        """
        table = connection.execute(query).fetch_arrow_table().combine_chunks()
    finally:
        connection.close()
    native = table.cast(_native_schema("final_market_inputs"), safe=True)
    _require(native.num_rows > 0, "final market-input formula emitted no rows")
    observed_months = tuple(
        sorted(set(str(value) for value in native["source_holding_month"].to_pylist()))
    )
    _require(
        observed_months == tuple(sorted(pending_months)),
        "final market-input formula omitted a pending month",
    )
    _require(
        pc.all(pc.equal(native["calendar_date"], native["entry_date"])).as_py() is True,
        "final market-input formula used a non-entry market date",
    )
    if PARTIAL_DECEMBER_HOLDING_MONTH in pending_months:
        partial = native.filter(
            pc.equal(
                native["source_holding_month"],
                pa.scalar(PARTIAL_DECEMBER_HOLDING_MONTH, pa.string()),
            )
        )
        _require(
            partial.num_rows > 0
            and set(partial["entry_date"].to_pylist()) == {FINAL_PARTIAL_ENTRY_DATE}
            and set(partial["calendar_date"].to_pylist()) == {FINAL_PARTIAL_ENTRY_DATE},
            "partial-December entry-market domain differs",
        )
    keys = (
        native.select(["source_holding_month", "calendar_date", "permno"])
        .group_by(["source_holding_month", "calendar_date", "permno"])
        .aggregate([])
    )
    _require(keys.num_rows == native.num_rows, "final market-input keys duplicate")
    return native, MappingProxyType(
        {
            "pending_month_count": len(pending_months),
            "rows": native.num_rows,
            "entry_day_only": True,
            "inherited_holding_decision_inputs_included": True,
            "inherited_holding_rule": "latest_authenticated_long_eligible_row_at_or_before_signal",
            "full_memoized_predictor_risk_history_consulted_on_resume": True,
            "future_predictor_risk_rows_permitted": False,
            "outcome_or_accounting_sources_opened": False,
            "current_month_calendar_identity_projected": True,
            "exact_current_entry_market_and_shares_joined": True,
            "ordinary_months_use_signal_history_market": True,
            "partial_december_uses_exact_entry_consumer_projection": True,
            "partial_december_entry_date": FINAL_PARTIAL_ENTRY_DATE.isoformat(),
            "entry_projection_return_columns_requested": 0,
            "duplicate_output_keys": 0,
            "frozen_stage2_typed_market_reused": True,
            "passed_months_recomputed": 0,
        }
    )


def _jsonable_source_value(value: object) -> object:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_source_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_source_value(item) for item in value]
    return value


def _source_row_sha256(row: Mapping[str, object]) -> str:
    return _sha256_bytes(_canonical_json_bytes(_jsonable_source_value(row)))


def _exact_security_day_row(
    table: pa.Table,
    *,
    permno: int,
    calendar_date: dt.date,
    label: str,
) -> dict[str, object]:
    _require(
        {"permno", "calendar_date"}.issubset(table.column_names),
        f"{label} key columns are absent",
    )
    mask = pc.and_(
        pc.equal(
            table["permno"],
            pa.scalar(permno, type=table.schema.field("permno").type),
        ),
        pc.equal(
            table["calendar_date"],
            pa.scalar(
                calendar_date,
                type=table.schema.field("calendar_date").type,
            ),
        ),
    )
    selected = table.filter(mask)
    _require(selected.num_rows == 1, f"{label} must contain exactly one row")
    return dict(selected.to_pylist()[0])


def _security_day_row_count(
    table: pa.Table,
    *,
    permno: int,
    calendar_date: dt.date,
) -> int:
    mask = pc.and_(
        pc.equal(
            table["permno"],
            pa.scalar(permno, type=table.schema.field("permno").type),
        ),
        pc.equal(
            table["calendar_date"],
            pa.scalar(
                calendar_date,
                type=table.schema.field("calendar_date").type,
            ),
        ),
    )
    return table.filter(mask).num_rows


def _authenticated_file_sha256(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> str:
    _require(
        _HEX64.fullmatch(expected_sha256) is not None,
        f"{label} expected SHA-256 is malformed",
    )
    metadata = path.stat(follow_symlinks=False)
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not path.is_symlink()
        and metadata.st_nlink == 1,
        f"{label} is unsafe",
    )
    observed = _sha256_file(path)
    _require(observed == expected_sha256, f"{label} bytes differ")
    return observed


def _read_authenticated_cross_scope_delisting_row(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, str], str]:
    """Read the one hash-frozen raw event authorized for the boundary stitch."""

    source_sha = _authenticated_file_sha256(
        path,
        expected_sha256=expected_sha256,
        label="cross-scope raw delisting source",
    )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(
            tuple(reader.fieldnames or ()) == _CROSS_SCOPE_RAW_DELISTING_COLUMNS,
            "cross-scope raw delisting columns differ",
        )
        matches = [
            dict(row)
            for row in reader
            if row.get("PERMNO") == str(_CROSS_SCOPE_TERMINAL_STITCH_PERMNO)
            and row.get("DelistingDt") == _CROSS_SCOPE_TERMINAL_STITCH_DATE.isoformat()
        ]
    _require(
        len(matches) == 1 and None not in matches[0],
        "cross-scope raw delisting identity is not unique",
    )
    row = matches[0]
    _require(
        tuple(row) == _CROSS_SCOPE_RAW_DELISTING_COLUMNS
        and _source_row_sha256(row) == _CROSS_SCOPE_RAW_DELISTING_ROW_SHA256,
        "cross-scope raw delisting row differs",
    )
    expected_literals = {
        "PERMNO": "91416",
        "DelistingDt": "2025-11-28",
        "DelDtPrc": "6.470000",
        "DelDtPrcFlg": "TR",
        "DelActionType": "MER",
        "DelStatusType": "FPAY",
        "DelReasonType": "UNAV",
        "DelPaymentType": "CST",
        "DelPERMNO": "86121",
        "DelPERMCO": "34782",
        "DelRet": "0.031716",
        "DelRetMissType": "NA",
        "DelNextDt": "",
        "DelNextPrc": "",
        "DelNextPrcFlg": "DA",
        "DelAmtDt": "2025-12-01",
        "DelDivAmt": "6.675200",
        "DelDisType": "D2",
        "DelDlyDt": "2025-12-01",
        "PrimaryExch": "N",
        "SICCD": "2253",
        "NASDIssuno": "0",
    }
    _require(
        row == expected_literals,
        "cross-scope raw delisting economics differ",
    )
    return row, source_sha


def _authenticate_2025_11_91416_boundary_stitch(
    *,
    decision_return_path: Path,
    outcome_return_path: Path,
    delisting_event_schema: pa.Schema,
    raw_delisting_path: Path,
    expected_decision_return_sha256: str,
    expected_outcome_return_sha256: str,
    expected_raw_delisting_sha256: str = _CROSS_SCOPE_RAW_DELISTING_SHA256,
) -> tuple[pa.Table, pa.Table, Mapping[str, object]]:
    """Stitch one terminal event split by the Nov-28/Dec-1 source boundary.

    The decision row remains an immutable source record.  A derived copy gets
    the already-observed terminal adjustment after the outcome barrier opens;
    the Dec-1 posting row remains unchanged as suppressed audit evidence.
    """

    from portfolio_bot.data.phase_02_crsp_return_policy import (
        combine_nonoverlapping_returns,
        is_performance_related_delisting,
    )

    decision_sha_before = _authenticated_file_sha256(
        decision_return_path,
        expected_sha256=expected_decision_return_sha256,
        label="cross-scope decision return source",
    )
    outcome_sha_before = _authenticated_file_sha256(
        outcome_return_path,
        expected_sha256=expected_outcome_return_sha256,
        label="cross-scope outcome return source",
    )
    decision = pq.read_table(
        decision_return_path,
        use_threads=False,
    ).replace_schema_metadata(None)
    outcome = pq.read_table(
        outcome_return_path,
        use_threads=False,
    ).replace_schema_metadata(None)
    _require(
        decision.schema.equals(outcome.schema, check_metadata=True),
        "cross-scope return source schemas differ",
    )
    _require(
        tuple(delisting_event_schema.names) == _CROSS_SCOPE_DELISTING_EVENT_COLUMNS,
        "cross-scope delisting event schema differs",
    )
    _require(
        _security_day_row_count(
            outcome,
            permno=_CROSS_SCOPE_TERMINAL_STITCH_PERMNO,
            calendar_date=_CROSS_SCOPE_TERMINAL_STITCH_DATE,
        )
        == 0
        and _security_day_row_count(
            decision,
            permno=_CROSS_SCOPE_TERMINAL_STITCH_PERMNO,
            calendar_date=_CROSS_SCOPE_TERMINAL_POSTING_DATE,
        )
        == 0,
        "cross-scope return rows overlap their source partitions",
    )
    decision_row = _exact_security_day_row(
        decision,
        permno=_CROSS_SCOPE_TERMINAL_STITCH_PERMNO,
        calendar_date=_CROSS_SCOPE_TERMINAL_STITCH_DATE,
        label="cross-scope decision boundary row",
    )
    outcome_audit_row = _exact_security_day_row(
        outcome,
        permno=_CROSS_SCOPE_TERMINAL_STITCH_PERMNO,
        calendar_date=_CROSS_SCOPE_TERMINAL_POSTING_DATE,
        label="cross-scope outcome posting audit row",
    )
    _require(
        _source_row_sha256(decision_row) == _CROSS_SCOPE_DECISION_RETURN_ROW_SHA256,
        "cross-scope decision boundary row differs",
    )
    _require(
        _source_row_sha256(outcome_audit_row) == _CROSS_SCOPE_OUTCOME_AUDIT_ROW_SHA256,
        "cross-scope outcome posting audit row differs",
    )
    raw_row, raw_source_sha = _read_authenticated_cross_scope_delisting_row(
        raw_delisting_path,
        expected_sha256=expected_raw_delisting_sha256,
    )

    ordinary = decision_row.get("authoritative_total_return")
    adjustment = Decimal(raw_row["DelRet"]).quantize(Decimal("0.000000000001"))
    _require(
        ordinary == Decimal("-0.018209000000")
        and decision_row.get("raw_total_return") == Decimal("-0.018209")
        and decision_row.get("raw_return_ex_distributions") == Decimal("-0.018209")
        and decision_row.get("raw_return_missing_flag") == "NA"
        and decision_row.get("raw_return_duration_flag") == "D2"
        and decision_row.get("return_leg_class") == "ordinary_one_session"
        and decision_row.get("return_span_start_date") == dt.date(2025, 11, 26)
        and decision_row.get("return_span_end_date")
        == _CROSS_SCOPE_TERMINAL_STITCH_DATE
        and decision_row.get("covered_trading_sessions") == 1
        and decision_row.get("span_alignment_status") == "verified_one_session_span"
        and decision_row.get("delisting_event_count") == 0
        and decision_row.get("delisting_treatment_status") == "none"
        and decision_row.get("delisting_adjustment_return") is None
        and decision_row.get("authoritative_return_state") == "observed"
        and decision_row.get("reconciliation_status") == "no_event_return_difference"
        and decision_row.get("return_history_usable") is True
        and decision_row.get("return_history_authorized") is False,
        "cross-scope decision boundary semantics differ",
    )
    _require(
        outcome_audit_row.get("raw_total_return") == Decimal("0.031716")
        and outcome_audit_row.get("raw_return_ex_distributions") == Decimal("0.031716")
        and outcome_audit_row.get("raw_return_missing_flag") == "NA"
        and outcome_audit_row.get("raw_return_duration_flag") == "DD"
        and outcome_audit_row.get("mapped_trading_periods") is None
        and outcome_audit_row.get("return_leg_class")
        == "pre_scope_terminal_delisting_context_suppressed"
        and outcome_audit_row.get("return_span_start_date")
        == _CROSS_SCOPE_TERMINAL_STITCH_DATE
        and outcome_audit_row.get("return_span_end_date")
        == _CROSS_SCOPE_TERMINAL_POSTING_DATE
        and outcome_audit_row.get("covered_trading_sessions") == 0
        and outcome_audit_row.get("span_alignment_status")
        == "pre_scope_event_context_suppressed_no_factor_window"
        and outcome_audit_row.get("distribution_event_count") == 2
        and outcome_audit_row.get("delisting_event_count") == 0
        and outcome_audit_row.get("delisting_treatment_status")
        == "pre_scope_delisting_context_suppressed"
        and outcome_audit_row.get("delisting_adjustment_return") is None
        and outcome_audit_row.get("authoritative_total_return") is None
        and outcome_audit_row.get("authoritative_return_state")
        == "pre_scope_delisting_posting_context_suppressed"
        and outcome_audit_row.get("return_history_usable") is False
        and outcome_audit_row.get("return_history_authorized") is False
        and adjustment == Decimal("0.031716000000"),
        "cross-scope outcome posting audit semantics differ",
    )

    event_row = {
        "permno": _CROSS_SCOPE_TERMINAL_STITCH_PERMNO,
        "delisting_date": _CROSS_SCOPE_TERMINAL_STITCH_DATE,
        "posting_date": _CROSS_SCOPE_TERMINAL_POSTING_DATE,
        "delisting_price": Decimal(raw_row["DelDtPrc"]),
        "delisting_price_flag": raw_row["DelDtPrcFlg"],
        "action_type": raw_row["DelActionType"],
        "status_type": raw_row["DelStatusType"],
        "reason_type": raw_row["DelReasonType"],
        "payment_type": raw_row["DelPaymentType"],
        "successor_permno": int(raw_row["DelPERMNO"]),
        "successor_permco": int(raw_row["DelPERMCO"]),
        "observed_delisting_return": Decimal(raw_row["DelRet"]),
        "return_missing_type": raw_row["DelRetMissType"],
        "next_date": None,
        "next_price": None,
        "next_price_flag": raw_row["DelNextPrcFlg"],
        "amount_date": dt.date.fromisoformat(raw_row["DelAmtDt"]),
        "delisting_dividend_amount": Decimal(raw_row["DelDivAmt"]),
        "distribution_type": raw_row["DelDisType"],
        "primary_exchange": raw_row["PrimaryExch"],
        "performance_related": is_performance_related_delisting(
            action_type=raw_row["DelActionType"],
            payment_type=raw_row["DelPaymentType"],
            status_type=raw_row["DelStatusType"],
            reason_type=raw_row["DelReasonType"],
        ),
        "embedding_status": (
            "embedded_in_posting_daily_total_used_as_identity_evidence"
        ),
        "treatment_status": ("observed_event_return_attached_to_delisting_date_once"),
        "effective_delisting_return": adjustment,
        "imputation_applied": False,
        "assigned_return_date": _CROSS_SCOPE_TERMINAL_STITCH_DATE,
        "outcome_availability_date": _CROSS_SCOPE_TERMINAL_POSTING_DATE,
        "assigned_exactly_once": True,
    }
    event_table = pa.Table.from_pylist(
        [event_row],
        schema=delisting_event_schema.remove_metadata(),
    ).combine_chunks()
    canonical_event_row = event_table.to_pylist()[0]
    event_row_sha = _source_row_sha256(canonical_event_row)
    _require(
        event_row_sha == _CROSS_SCOPE_DELISTING_EVENT_ROW_SHA256,
        "cross-scope canonical delisting event differs",
    )

    assert isinstance(ordinary, Decimal)
    combined = combine_nonoverlapping_returns(ordinary, adjustment)
    _require(
        combined == Decimal("0.012929483356"),
        "cross-scope terminal compound differs",
    )
    stitched_row = dict(decision_row)
    stitched_row.update(
        {
            "return_leg_class": ("ordinary_plus_terminal_delisting_adjustment"),
            "span_alignment_status": (
                "ordinary_span_plus_separate_terminal_adjustment"
            ),
            "delisting_event_count": 1,
            "delisting_treatment_status": (
                "observed_event_return_attached_to_delisting_date_once"
            ),
            "delisting_adjustment_return": adjustment,
            "authoritative_total_return": combined,
            "outcome_availability_date": _CROSS_SCOPE_TERMINAL_POSTING_DATE,
            "authoritative_return_state": ("observed_plus_delisting_adjustment"),
            "reconciliation_status": "delisting_treatment_accounted_once",
            "return_history_usable": True,
            "return_history_authorized": False,
        }
    )
    replacement = pa.Table.from_pylist([stitched_row], schema=decision.schema)
    canonical_stitched_row = replacement.to_pylist()[0]
    stitched_row_sha = _source_row_sha256(canonical_stitched_row)
    _require(
        stitched_row_sha == _CROSS_SCOPE_STITCHED_RETURN_ROW_SHA256,
        "cross-scope stitched return row differs",
    )
    target_mask = pc.and_(
        pc.equal(
            decision["permno"],
            pa.scalar(
                _CROSS_SCOPE_TERMINAL_STITCH_PERMNO,
                type=decision.schema.field("permno").type,
            ),
        ),
        pc.equal(
            decision["calendar_date"],
            pa.scalar(
                _CROSS_SCOPE_TERMINAL_STITCH_DATE,
                type=decision.schema.field("calendar_date").type,
            ),
        ),
    )
    stitched_decision = pa.concat_tables(
        [decision.filter(pc.invert(target_mask)), replacement]
    ).combine_chunks()
    stitched = pa.concat_tables([stitched_decision, outcome]).combine_chunks()
    _require(
        stitched.num_rows == decision.num_rows + outcome.num_rows
        and _source_row_sha256(
            _exact_security_day_row(
                stitched,
                permno=_CROSS_SCOPE_TERMINAL_STITCH_PERMNO,
                calendar_date=_CROSS_SCOPE_TERMINAL_STITCH_DATE,
                label="cross-scope stitched return row",
            )
        )
        == _CROSS_SCOPE_STITCHED_RETURN_ROW_SHA256
        and _source_row_sha256(
            _exact_security_day_row(
                stitched,
                permno=_CROSS_SCOPE_TERMINAL_STITCH_PERMNO,
                calendar_date=_CROSS_SCOPE_TERMINAL_POSTING_DATE,
                label="cross-scope preserved posting audit row",
            )
        )
        == _CROSS_SCOPE_OUTCOME_AUDIT_ROW_SHA256,
        "cross-scope stitch row preservation differs",
    )

    decision_sha_after = _sha256_file(decision_return_path)
    outcome_sha_after = _sha256_file(outcome_return_path)
    raw_sha_after = _sha256_file(raw_delisting_path)
    _require(
        decision_sha_after == decision_sha_before
        and outcome_sha_after == outcome_sha_before
        and raw_sha_after == raw_source_sha,
        "cross-scope stitch mutated an authenticated source",
    )
    receipt = {
        "stitch_id": _CROSS_SCOPE_TERMINAL_STITCH_ID,
        "status": "AUTHENTICATED_CROSS_SCOPE_TERMINAL_STITCH_APPLIED",
        "holding_month": _CROSS_SCOPE_TERMINAL_STITCH_MONTH,
        "permno": _CROSS_SCOPE_TERMINAL_STITCH_PERMNO,
        "economic_event_date": _CROSS_SCOPE_TERMINAL_STITCH_DATE.isoformat(),
        "posting_date": _CROSS_SCOPE_TERMINAL_POSTING_DATE.isoformat(),
        "decision_return_source_sha256_before": decision_sha_before,
        "decision_return_source_sha256_after": decision_sha_after,
        "decision_return_source_bytes_preserved": True,
        "decision_source_row_sha256": _CROSS_SCOPE_DECISION_RETURN_ROW_SHA256,
        "outcome_return_source_sha256_before": outcome_sha_before,
        "outcome_return_source_sha256_after": outcome_sha_after,
        "outcome_posting_audit_row_sha256": (_CROSS_SCOPE_OUTCOME_AUDIT_ROW_SHA256),
        "outcome_posting_audit_row_preserved": True,
        "raw_delisting_source_sha256": raw_source_sha,
        "raw_delisting_row_sha256": _CROSS_SCOPE_RAW_DELISTING_ROW_SHA256,
        "canonical_delisting_event_row_sha256": event_row_sha,
        "ordinary_return": format(ordinary, "f"),
        "terminal_delisting_adjustment_return": format(adjustment, "f"),
        "combined_authoritative_total_return": format(combined, "f"),
        "compound_rule": "frozen_combine_nonoverlapping_returns_once",
        "stitched_return_row_sha256": stitched_row_sha,
        "decision_source_row_replaced_in_place": False,
        "derived_copy_only": True,
        "outcome_barrier_required": True,
        "rows_added": 0,
        "rows_removed": 0,
    }
    return stitched, event_table, MappingProxyType(receipt)


def _ensure_final_factor_normalization(
    *,
    database_root: Path,
    work_root: Path,
    trading_calendar_path: Path,
) -> tuple[Path, Mapping[str, object]]:
    """Build/reuse the frozen Stage 2 factor parser through final close."""

    from portfolio_bot.data import phase_02_auxiliary_compustat_factors as factors
    from portfolio_bot.data.phase_02_auxiliary_core import (
        TradingCalendar,
        load_schema_document,
    )

    receipt_path = work_root / "factor_normalization_receipt.json"
    daily_path = work_root / "ff_factor_daily_observation.parquet"
    if receipt_path.is_file():
        receipt = _json(receipt_path, label="final factor normalization receipt")
        outputs = receipt.get("outputs")
        _require(isinstance(outputs, list), "final factor outputs are absent")
        for raw in outputs:
            _require(isinstance(raw, Mapping), "final factor output receipt differs")
            path = work_root / str(raw.get("path"))
            _require(
                path.is_file()
                and not path.is_symlink()
                and _sha256_file(path) == raw.get("sha256")
                and path.stat().st_size == int(raw.get("size_bytes", -1)),
                f"memoized final factor table differs: {path.name}",
            )
        _require(daily_path.is_file(), "final daily factor table is absent")
        return daily_path, MappingProxyType(receipt)
    if work_root.exists():
        _quarantine_incomplete(work_root)
    work_root.mkdir(parents=True)
    calendar_table = pq.read_table(
        trading_calendar_path,
        columns=["calendar_date"],
        filters=[("calendar_date", "<=", FINAL_LAST_EXIT)],
        use_threads=False,
    )
    calendar = TradingCalendar(calendar_table["calendar_date"].to_pylist())
    schema_document = load_schema_document(
        PROJECT_ROOT
        / "configs/phase_02_canonicalization/auxiliary_source_schema_v1.json"
    )
    with _temporary_assignments(
        (
            (factors, "LAST_SIGNAL", FINAL_LAST_SIGNAL),
            (factors, "LAST_EXIT", FINAL_LAST_EXIT),
        )
    ):
        outputs = factors.build_factor_tables(
            raw_root=database_root / "raw",
            output_dir=work_root,
            schema_document=schema_document,
            calendar=calendar,
        )
    receipt = {
        "schema_version": 1,
        "status": "FINAL_FROZEN_FACTOR_NORMALIZATION_COMPLETE",
        "formula_module": "phase_02_auxiliary_compustat_factors.build_factor_tables",
        "scope": {
            "last_signal": FINAL_LAST_SIGNAL.isoformat(),
            "last_exit": FINAL_LAST_EXIT.isoformat(),
        },
        "outputs": outputs,
    }
    _atomic_bytes(receipt_path, _canonical_json_bytes(receipt))
    return daily_path, MappingProxyType(receipt)


def _parquet_union(
    paths: Sequence[Path],
    *,
    columns: Sequence[str] | None = None,
) -> pa.Table:
    _require(bool(paths), "Parquet union source set is empty")
    tables: list[pa.Table] = []
    for path in paths:
        _require(
            path.is_file() and not path.is_symlink(),
            f"Parquet source is absent: {path}",
        )
        table = pq.read_table(
            path,
            columns=None if columns is None else list(columns),
            use_threads=False,
        ).replace_schema_metadata(None)
        tables.append(table)
    fields = tables[0].schema
    _require(
        all(table.schema.equals(fields, check_metadata=True) for table in tables[1:]),
        "Parquet union field schemas differ",
    )
    return pa.concat_tables(tables).combine_chunks()


def _final_daily_factors(path: Path) -> pa.Table:
    from portfolio_bot.validation.validation_source_extraction import FACTOR_SCHEMA

    source = pq.read_table(
        path,
        columns=list(FACTOR_SCHEMA.names),
        filters=[
            ("observation_date", ">=", dt.date(2020, 1, 1)),
            ("observation_date", "<=", FINAL_LAST_EXIT),
        ],
        use_threads=False,
    ).combine_chunks()
    return source.cast(FACTOR_SCHEMA, safe=True)


def _event_identity_maps(
    *,
    return_roots: Sequence[Path],
    additional_delisting_rows: pa.Table | None = None,
    terminal_close_overlay: object | None = None,
) -> tuple[
    Mapping[tuple[int, dt.date], tuple[int, str, bool]],
    Mapping[tuple[int, dt.date], Mapping[str, object]],
    Mapping[str, object],
]:
    """Authenticate distribution sets and apply the frozen cash-action rule."""

    from portfolio_bot.validation import corporate_action_adapter as action_adapter

    distribution_paths = tuple(
        root / "crsp_distribution_event.parquet" for root in return_roots
    )
    delisting_paths = tuple(
        root / "crsp_delisting_event.parquet" for root in return_roots
    )
    distributions = _parquet_union(distribution_paths)
    delistings = _parquet_union(delisting_paths)
    published_delisting_rows = delistings.num_rows
    added_delisting_rows = (
        0 if additional_delisting_rows is None else additional_delisting_rows.num_rows
    )
    if additional_delisting_rows is not None:
        additional = additional_delisting_rows.combine_chunks()
        _require(
            additional.num_rows == 1
            and additional.schema.equals(delistings.schema, check_metadata=True),
            "additional cross-scope delisting event differs",
        )
        delistings = pa.concat_tables([delistings, additional]).combine_chunks()

    by_distribution: dict[tuple[int, dt.date], list[str]] = defaultdict(list)
    for row in distributions.to_pylist():
        mapped = row.get("mapped_return_date")
        if type(mapped) is not dt.date:
            continue
        _require(
            row.get("assigned_exactly_once") is True,
            "distribution event is not assigned exactly once",
        )
        by_distribution[(int(row["permno"]), mapped)].append(_source_row_sha256(row))
    distribution_map: dict[tuple[int, dt.date], tuple[int, str, bool]] = {}
    for key, values in by_distribution.items():
        ordered = sorted(values)
        distribution_map[key] = (
            len(ordered),
            _sha256_bytes(_canonical_json_bytes({"event_sha256": ordered})),
            True,
        )

    manifest_path = action_adapter.FROZEN_MAPPING_MANIFEST_PATH
    _require(
        manifest_path.is_file()
        and not manifest_path.is_symlink()
        and _sha256_file(manifest_path)
        == action_adapter.FROZEN_MAPPING_MANIFEST_SHA256,
        "frozen corporate-action mapping manifest differs",
    )
    manifest = _json(manifest_path, label="frozen corporate-action mapping manifest")
    rules = manifest.get("rules")
    _require(
        isinstance(rules, list)
        and len(rules) == 1
        and isinstance(rules[0], Mapping)
        and rules[0].get("disposition") == "cash"
        and rules[0].get("successor_policy") == "ignored"
        and rules[0].get("terminal_total_return_already_applied") is True,
        "frozen corporate-action economic rule differs",
    )
    action_map: dict[tuple[int, dt.date], Mapping[str, object]] = {}
    sequence_by_date: dict[dt.date, int] = defaultdict(int)
    candidates: list[tuple[dict[str, object], str, str, str | None]] = []
    for row in delistings.to_pylist():
        assigned = row.get("assigned_return_date")
        treatment = row.get("treatment_status")
        if (
            type(assigned) is not dt.date
            or treatment not in action_adapter.RESOLVED_TERMINAL_TREATMENTS
        ):
            continue
        _require(
            row.get("assigned_exactly_once") is True,
            "corporate action is not assigned exactly once",
        )
        candidates.append((row, _source_row_sha256(row), "stage11", None))
    terminal_close_count = 0
    terminal_close_policy_sha256: str | None = None
    if terminal_close_overlay is not None:
        from portfolio_bot.final_test.terminal_event_overlay import (
            LastTradableCloseOverlay,
        )

        _require(
            isinstance(terminal_close_overlay, LastTradableCloseOverlay),
            "last-tradable-close overlay type differs",
        )
        terminal_close_policy_sha256 = terminal_close_overlay.policy_sha256
        terminal_close_count = len(terminal_close_overlay.events)
        _require(
            terminal_close_count > 0,
            "last-tradable-close overlay contains no events",
        )
        for event in terminal_close_overlay.events:
            _require(
                _HEX64.fullmatch(event.source_event_sha256) is not None
                and _HEX64.fullmatch(event.evidence_sha256) is not None,
                "last-tradable-close event evidence differs",
            )
            candidates.append(
                (
                    {
                        "assigned_return_date": event.delisting_date,
                        "permno": event.permno,
                        "delisting_date": event.delisting_date,
                        "outcome_availability_date": (event.outcome_availability_date),
                        "terminal_sensitivity_policy_sha256": (
                            terminal_close_overlay.policy_sha256
                        ),
                        "terminal_sensitivity_evidence_sha256": (event.evidence_sha256),
                        "event_day_quoted_full_spread": (
                            event.event_day_quoted_full_spread
                        ),
                    },
                    event.source_event_sha256,
                    "stage13_last_tradable_close",
                    event.evidence_sha256,
                )
            )
    candidates.sort(
        key=lambda value: (
            value[0]["assigned_return_date"],
            int(value[0]["permno"]),
            value[0]["delisting_date"],
            value[1],
        )
    )
    for row, source_sha, action_source, evidence_sha in candidates:
        assigned = row["assigned_return_date"]
        assert isinstance(assigned, dt.date)
        key = (int(row["permno"]), assigned)
        _require(
            key not in action_map, "multiple corporate actions occupy one security-day"
        )
        sequence_by_date[assigned] += 1
        if action_source == "stage11":
            action_id = (
                "stage11-action-"
                + _sha256_bytes(
                    _canonical_json_bytes(
                        {
                            "source_record_sha256": source_sha,
                            "mapping_manifest_sha256": action_adapter.FROZEN_MAPPING_MANIFEST_SHA256,
                            "rules_sha256": manifest.get("rules_sha256"),
                            "disposition": "cash",
                        }
                    )
                )[:32]
            )
        else:
            _require(
                action_source == "stage13_last_tradable_close"
                and evidence_sha is not None
                and terminal_close_policy_sha256 is not None,
                "last-tradable-close action source differs",
            )
            action_id = (
                "stage13-final-close-action-"
                + _sha256_bytes(
                    _canonical_json_bytes(
                        {
                            "source_record_sha256": source_sha,
                            "evidence_sha256": evidence_sha,
                            "policy_sha256": terminal_close_policy_sha256,
                            "treatment": (
                                "public_interest_last_tradable_close_liquidation"
                            ),
                            "disposition": "cash",
                        }
                    )
                )[:32]
            )
        action_map[key] = MappingProxyType(
            {
                "corporate_action_id": action_id,
                "corporate_economic_event_date": row["delisting_date"],
                "corporate_assigned_return_date": assigned,
                "corporate_public_availability_date": row["outcome_availability_date"],
                "corporate_action_sequence": sequence_by_date[assigned],
                "corporate_source_record_sha256": source_sha,
                "corporate_source_permno": int(row["permno"]),
                "corporate_settlement_kind": "cash",
                "corporate_successor_permno": None,
                "corporate_settlement_value_multiplier": 1.0,
                "corporate_terminal_total_return_already_applied": True,
                "corporate_terminal_sensitivity_policy_sha256": (
                    row.get("terminal_sensitivity_policy_sha256")
                ),
                "corporate_terminal_sensitivity_evidence_sha256": (
                    row.get("terminal_sensitivity_evidence_sha256")
                ),
                "corporate_event_day_quoted_full_spread": (
                    None
                    if row.get("event_day_quoted_full_spread") is None
                    else float(row["event_day_quoted_full_spread"])
                ),
            }
        )
    identity = MappingProxyType(
        {
            "distribution_event_rows": distributions.num_rows,
            "published_delisting_event_rows": published_delisting_rows,
            "cross_scope_boundary_delisting_event_rows": added_delisting_rows,
            "last_tradable_close_liquidation_action_rows": terminal_close_count,
            "last_tradable_close_policy_sha256": terminal_close_policy_sha256,
            "resolved_corporate_action_rows": len(action_map),
            "corporate_action_mapping_manifest_sha256": action_adapter.FROZEN_MAPPING_MANIFEST_SHA256,
            "corporate_action_rules_sha256": manifest.get("rules_sha256"),
        }
    )
    return MappingProxyType(distribution_map), MappingProxyType(action_map), identity


def _accounting_transport_finite(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalSourceReplayError(f"{label} is not numeric") from exc
    _require(math.isfinite(result), f"{label} is nonfinite")
    return result


def _normalize_final_market_accounting_transport_rows(
    rows: list[dict[str, object]],
) -> Mapping[str, object]:
    """Encode authenticated missing states without changing economic meaning.

    The v1 registry deliberately keeps these numeric transport columns
    non-nullable.  IEEE NaN carries an upstream missing execution/cost state to
    the frozen fail-closed accounting kernel, where an exact hold is permitted
    and any nonzero trade remains prohibited.  Accounting is strategy
    independent, so a missing source borrow rate is filled with the frozen
    primary 1% rate: longs incur no borrow charge, while an inherited short
    remains charged even after it loses current short eligibility.
    """

    counts = Counter({name: 0 for name in _ACCOUNTING_NAN_TRANSPORT_FIELDS})
    borrow_defaults = 0
    for row_index, row in enumerate(rows):
        status = row.get("entry_close_status")
        raw_entry = row.get("entry_close_usd_per_share")
        if status == "PRESENT":
            entry = _accounting_transport_finite(
                raw_entry,
                label=f"accounting entry close row {row_index}",
            )
            _require(entry > 0.0, "present accounting entry close is not positive")
            row["entry_close_usd_per_share"] = entry
        elif status == "MISSING_VALID_ENTRY_CLOSE":
            if raw_entry is not None:
                invalid_entry = _accounting_transport_finite(
                    raw_entry,
                    label=f"missing accounting entry close row {row_index}",
                )
                _require(
                    invalid_entry <= 0.0,
                    "missing accounting entry-close status contradicts its value",
                )
            row["entry_close_usd_per_share"] = math.nan
            counts["entry_close_usd_per_share"] += 1
        else:
            raise FinalSourceReplayError("accounting entry-close status differs")

        for name, minimum, strictly_positive in (
            ("full_spread_20d", 0.0, False),
            ("adv20_usd", 0.0, True),
            ("impact_volatility_daily", 0.0, False),
        ):
            raw = row.get(name)
            if raw is None:
                row[name] = math.nan
                counts[name] += 1
                continue
            value = _accounting_transport_finite(
                raw,
                label=f"accounting {name} row {row_index}",
            )
            _require(
                value > minimum if strictly_positive else value >= minimum,
                f"accounting {name} is outside its frozen domain",
            )
            row[name] = value

        raw_borrow = row.get("borrow_rate_annual")
        if raw_borrow is None:
            row["borrow_rate_annual"] = _PRIMARY_ACCOUNTING_BORROW_RATE_ANNUAL
            borrow_defaults += 1
        else:
            borrow = _accounting_transport_finite(
                raw_borrow,
                label=f"accounting borrow rate row {row_index}",
            )
            _require(
                borrow == _PRIMARY_ACCOUNTING_BORROW_RATE_ANNUAL,
                "accounting source borrow rate differs from the frozen primary rate",
            )
            row["borrow_rate_annual"] = borrow

    return MappingProxyType(
        {
            "transport_rule": "stage13-accounting-missing-numeric-transport-v1",
            "transport_row_count": len(rows),
            "nan_encoded_row_counts": {
                name: int(counts[name]) for name in _ACCOUNTING_NAN_TRANSPORT_FIELDS
            },
            "source_null_borrow_default_row_count": borrow_defaults,
            "source_null_borrow_default_annual_rate": (
                _PRIMARY_ACCOUNTING_BORROW_RATE_ANNUAL
            ),
            "rows_dropped": 0,
            "positive_adv_imputations": 0,
            "missing_execution_or_cost_remains_nontradable": True,
        }
    )


def _materialize_final_market_accounting(
    *,
    final_calendar: pa.Table,
    predictors: pa.Table,
    risk_cost: pa.Table,
    decision_market_root: Path,
    outcome_market_root: Path,
    decision_return_root: Path,
    outcome_return_root: Path,
    factor_table: pa.Table,
    trading_calendar_path: Path,
    pending_months: tuple[str, ...],
    raw_delisting_path: Path,
    expected_decision_return_sha256: str | None,
    expected_outcome_return_sha256: str | None,
) -> tuple[pa.Table, Mapping[str, object]]:
    """Materialize authenticated daily accounting rows for pending months."""

    return_paths = (
        decision_return_root / "crsp_security_day_return.parquet",
        outcome_return_root / "crsp_security_day_return.parquet",
    )
    market = _parquet_union(
        (
            decision_market_root / "crsp_security_day.parquet",
            outcome_market_root / "crsp_security_day.parquet",
        )
    )
    # Accounting authenticates the complete Stage 2 reconciliation evidence,
    # not only the seven-column Stage 8 risk projection.
    terminal_close_overlay: object | None = None
    terminal_close_receipt: Mapping[str, object] = MappingProxyType(
        {
            "policy_id": ("stage13-public-interest-last-tradable-close-liquidation-v1"),
            "status": "NOT_IN_PENDING_HOLDING_MONTH_SCOPE",
            "affected_holding_months": sorted(_LAST_TRADABLE_CLOSE_HOLDING_MONTHS),
            "economic_or_outcome_payload_opened": False,
        }
    )
    terminal_source_returns: pa.Table | None = None
    if _LAST_TRADABLE_CLOSE_HOLDING_MONTHS.intersection(pending_months):
        from portfolio_bot.final_test.terminal_event_overlay import (
            TerminalEventOverlayError,
            identify_last_tradable_close_liquidations,
        )

        policy_metadata = _LAST_TRADABLE_CLOSE_POLICY_PATH.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(policy_metadata.st_mode)
            and not _LAST_TRADABLE_CLOSE_POLICY_PATH.is_symlink()
            and policy_metadata.st_nlink == 1,
            "last-tradable-close policy path is unsafe",
        )
        policy = _json(
            _LAST_TRADABLE_CLOSE_POLICY_PATH,
            label="last-tradable-close policy",
        )
        terminal_source_returns = _parquet_union(return_paths)
        terminal_delistings = _parquet_union(
            tuple(
                root / "crsp_delisting_event.parquet"
                for root in (decision_return_root, outcome_return_root)
            )
        )
        terminal_monthly = _parquet_union(
            tuple(
                root / "crsp_security_month_return_reconciliation.parquet"
                for root in (decision_return_root, outcome_return_root)
            )
        )
        try:
            terminal_close_overlay = identify_last_tradable_close_liquidations(
                delistings=terminal_delistings,
                returns=terminal_source_returns,
                market=market,
                monthly_reconciliation=terminal_monthly,
                policy=policy,
                policy_sha256=_sha256_file(_LAST_TRADABLE_CLOSE_POLICY_PATH),
            )
        except TerminalEventOverlayError as exc:
            raise FinalSourceReplayError(
                "last-tradable-close liquidation evidence failed authentication"
            ) from exc
        terminal_close_receipt = terminal_close_overlay.receipt  # type: ignore[union-attr]
    boundary_event: pa.Table | None = None
    boundary_receipt: Mapping[str, object] = MappingProxyType(
        {
            "stitch_id": _CROSS_SCOPE_TERMINAL_STITCH_ID,
            "status": "NOT_IN_PENDING_HOLDING_MONTH_SCOPE",
            "holding_month": _CROSS_SCOPE_TERMINAL_STITCH_MONTH,
            "raw_or_outcome_boundary_payload_opened": False,
        }
    )
    if _CROSS_SCOPE_TERMINAL_STITCH_MONTH in pending_months:
        # The boundary authenticator reads and hashes both immutable return
        # sources independently.  Release the ordinary union first so the
        # large tables are never duplicated in memory.
        terminal_source_returns = None
        _require(
            isinstance(expected_decision_return_sha256, str)
            and isinstance(expected_outcome_return_sha256, str),
            "cross-scope authenticated return source pins are absent",
        )
        event_paths = (
            decision_return_root / "crsp_delisting_event.parquet",
            outcome_return_root / "crsp_delisting_event.parquet",
        )
        event_schemas = tuple(
            pq.read_schema(path).remove_metadata() for path in event_paths
        )
        _require(
            event_schemas[0].equals(event_schemas[1], check_metadata=True),
            "cross-scope delisting source schemas differ",
        )
        returns, boundary_event, boundary_receipt = (
            _authenticate_2025_11_91416_boundary_stitch(
                decision_return_path=return_paths[0],
                outcome_return_path=return_paths[1],
                delisting_event_schema=event_schemas[0],
                raw_delisting_path=raw_delisting_path,
                expected_decision_return_sha256=(expected_decision_return_sha256),
                expected_outcome_return_sha256=(expected_outcome_return_sha256),
            )
        )
    else:
        returns = (
            terminal_source_returns
            if terminal_source_returns is not None
            else _parquet_union(return_paths)
        )
    sessions = pq.read_table(
        trading_calendar_path,
        columns=["calendar_date"],
        filters=[
            ("calendar_date", ">=", dt.date(2020, 1, 1)),
            ("calendar_date", "<=", FINAL_LAST_EXIT),
        ],
        use_threads=False,
    )
    distribution_map, action_map, event_identity = _event_identity_maps(
        return_roots=(decision_return_root, outcome_return_root),
        additional_delisting_rows=boundary_event,
        terminal_close_overlay=terminal_close_overlay,
    )
    if terminal_close_overlay is not None:
        from portfolio_bot.final_test.terminal_event_overlay import (
            LastTradableCloseOverlay,
        )

        assert isinstance(terminal_close_overlay, LastTradableCloseOverlay)
        canonical_actions: list[dict[str, object]] = []
        for event in terminal_close_overlay.events:
            action = action_map.get((event.permno, event.delisting_date))
            _require(
                action is not None
                and str(action.get("corporate_action_id", "")).startswith(
                    "stage13-final-close-action-"
                )
                and action.get("corporate_economic_event_date") == event.delisting_date
                and action.get("corporate_assigned_return_date") == event.delisting_date
                and action.get("corporate_public_availability_date")
                == event.outcome_availability_date
                and action.get("corporate_source_record_sha256")
                == event.source_event_sha256
                and action.get("corporate_source_permno") == event.permno
                and action.get("corporate_settlement_kind") == "cash"
                and action.get("corporate_successor_permno") is None
                and action.get("corporate_settlement_value_multiplier") == 1.0
                and action.get("corporate_terminal_total_return_already_applied")
                is True
                and action.get("corporate_terminal_sensitivity_policy_sha256")
                == terminal_close_overlay.policy_sha256
                and action.get("corporate_terminal_sensitivity_evidence_sha256")
                == event.evidence_sha256
                and action.get("corporate_event_day_quoted_full_spread")
                == float(event.event_day_quoted_full_spread)
                and action_map.get((event.permno, event.posting_date)) is None,
                "last-tradable-close canonical cash action differs",
            )
            canonical_actions.append(
                {
                    "permno": event.permno,
                    "economic_event_date": event.delisting_date.isoformat(),
                    "posting_date": event.posting_date.isoformat(),
                    "corporate_action_id": action["corporate_action_id"],
                    "action_sequence": int(action["corporate_action_sequence"]),
                    "source_record_sha256": event.source_event_sha256,
                    "evidence_sha256": event.evidence_sha256,
                    "settlement_kind": "cash",
                    "settlement_value_multiplier": 1.0,
                    "terminal_total_return_already_applied": True,
                    "terminal_sensitivity_policy_sha256": (
                        terminal_close_overlay.policy_sha256
                    ),
                    "event_day_quoted_full_spread": format(
                        event.event_day_quoted_full_spread, "f"
                    ),
                }
            )
        terminal_close_receipt = MappingProxyType(
            {
                **dict(terminal_close_receipt),
                "canonical_cash_actions": canonical_actions,
            }
        )
    if boundary_event is not None:
        boundary_action = action_map.get(
            (
                _CROSS_SCOPE_TERMINAL_STITCH_PERMNO,
                _CROSS_SCOPE_TERMINAL_STITCH_DATE,
            )
        )
        _require(
            boundary_action is not None
            and boundary_action.get("corporate_action_id")
            == _CROSS_SCOPE_CORPORATE_ACTION_ID
            and boundary_action.get("corporate_economic_event_date")
            == _CROSS_SCOPE_TERMINAL_STITCH_DATE
            and boundary_action.get("corporate_assigned_return_date")
            == _CROSS_SCOPE_TERMINAL_STITCH_DATE
            and boundary_action.get("corporate_public_availability_date")
            == _CROSS_SCOPE_TERMINAL_POSTING_DATE
            and boundary_action.get("corporate_source_record_sha256")
            == _CROSS_SCOPE_DELISTING_EVENT_ROW_SHA256
            and boundary_action.get("corporate_source_permno")
            == _CROSS_SCOPE_TERMINAL_STITCH_PERMNO
            and boundary_action.get("corporate_settlement_kind") == "cash"
            and boundary_action.get("corporate_successor_permno") is None
            and boundary_action.get("corporate_settlement_value_multiplier") == 1.0
            and boundary_action.get("corporate_terminal_total_return_already_applied")
            is True
            and action_map.get(
                (
                    _CROSS_SCOPE_TERMINAL_STITCH_PERMNO,
                    _CROSS_SCOPE_TERMINAL_POSTING_DATE,
                )
            )
            is None,
            "cross-scope canonical cash action differs",
        )
        boundary_receipt = MappingProxyType(
            {
                **dict(boundary_receipt),
                "canonical_cash_action": {
                    "corporate_action_id": _CROSS_SCOPE_CORPORATE_ACTION_ID,
                    "source_record_sha256": (_CROSS_SCOPE_DELISTING_EVENT_ROW_SHA256),
                    "economic_event_date": (
                        _CROSS_SCOPE_TERMINAL_STITCH_DATE.isoformat()
                    ),
                    "assigned_return_date": (
                        _CROSS_SCOPE_TERMINAL_STITCH_DATE.isoformat()
                    ),
                    "public_availability_date": (
                        _CROSS_SCOPE_TERMINAL_POSTING_DATE.isoformat()
                    ),
                    "action_sequence": int(
                        boundary_action["corporate_action_sequence"]
                    ),
                    "settlement_kind": "cash",
                    "successor_permno": None,
                    "settlement_value_multiplier": 1.0,
                    "terminal_total_return_already_applied": True,
                },
            }
        )
    pending = pa.array(pending_months, pa.string())
    calendar = final_calendar.filter(
        pc.is_in(final_calendar["source_holding_month"], value_set=pending)
    )
    connection = duckdb.connect(":memory:")
    try:
        for name, table in (
            ("calendar", calendar),
            ("predictors", predictors),
            ("risk", risk_cost),
            ("returns", returns),
            ("market", market),
            ("factors", factor_table),
            ("sessions", sessions),
        ):
            connection.register(name, table)
        base = (
            connection.execute(
                """
            WITH first_seen AS (
                SELECT permno, min(signal_date) AS first_signal_date
                FROM predictors
                WHERE long_eligible IS TRUE
                GROUP BY permno
            ), active AS (
                SELECT c.*, f.permno
                FROM calendar c
                JOIN first_seen f ON f.first_signal_date <= c.signal_date
            ), active_cost AS (
                SELECT a.*, r.full_spread_20d, r.adv20_usd,
                       r.impact_volatility_daily,
                       r.primary_borrow_rate_annual,
                       row_number() OVER (
                           PARTITION BY a.source_holding_month, a.permno
                           ORDER BY r.signal_date DESC
                       ) AS cost_rank
                FROM active a
                JOIN risk r
                  ON r.permno = a.permno
                 AND r.signal_date <= a.signal_date
                 AND r.long_eligible IS TRUE
            ), scoped AS (
                SELECT * EXCLUDE(cost_rank) FROM active_cost WHERE cost_rank = 1
            )
            SELECT
                a.source_holding_month,
                a.signal_date,
                a.entry_date,
                a.exit_date,
                a.permno,
                ret.calendar_date,
                ret.return_leg_class,
                ret.return_span_start_date,
                ret.return_span_end_date,
                ret.covered_trading_sessions,
                ret.span_alignment_status,
                ret.delisting_event_count,
                ret.delisting_treatment_status,
                ret.delisting_adjustment_return,
                ret.authoritative_total_return,
                ret.outcome_availability_date,
                ret.authoritative_return_state,
                ret.reconciliation_status,
                ret.return_history_usable,
                ret.return_history_authorized,
                TRUE AS authenticated_trading_session,
                m.close_usd_per_share,
                CAST(entry_market.close_usd_per_share AS DOUBLE)
                    AS entry_close_usd_per_share,
                CASE
                    WHEN entry_market.close_usd_per_share IS NULL
                      OR entry_market.close_usd_per_share <= 0
                    THEN 'MISSING_VALID_ENTRY_CLOSE'
                    ELSE 'PRESENT'
                END AS entry_close_status,
                a.full_spread_20d,
                a.adv20_usd,
                a.impact_volatility_daily,
                a.primary_borrow_rate_annual AS borrow_rate_annual,
                CAST(0.005 AS DOUBLE) AS financing_rate_annual,
                CAST(NULL AS VARCHAR) AS source_payload_sha256,
                CAST(NULL AS VARCHAR) AS no_mark_source_record_sha256,
                f.observation_date AS factor_observation_date,
                f.public_availability_date AS factor_public_availability_date,
                f.mktrf, f.smb, f.hml, f.rmw, f.cma, f.rf, f.umd,
                CAST(0 AS INTEGER) AS distribution_event_count,
                CAST(NULL AS VARCHAR) AS distribution_event_set_sha256,
                TRUE AS distribution_events_assigned_exactly_once,
                CAST(NULL AS VARCHAR) AS corporate_action_id,
                CAST(NULL AS DATE) AS corporate_economic_event_date,
                CAST(NULL AS DATE) AS corporate_assigned_return_date,
                CAST(NULL AS DATE) AS corporate_public_availability_date,
                CAST(NULL AS INTEGER) AS corporate_action_sequence,
                CAST(NULL AS VARCHAR) AS corporate_source_record_sha256,
                CAST(NULL AS BIGINT) AS corporate_source_permno,
                CAST(NULL AS VARCHAR) AS corporate_settlement_kind,
                CAST(NULL AS BIGINT) AS corporate_successor_permno,
                CAST(NULL AS DOUBLE) AS corporate_settlement_value_multiplier,
                CAST(NULL AS BOOLEAN)
                    AS corporate_terminal_total_return_already_applied,
                CAST(NULL AS VARCHAR)
                    AS corporate_terminal_sensitivity_policy_sha256,
                CAST(NULL AS VARCHAR)
                    AS corporate_terminal_sensitivity_evidence_sha256,
                CAST(NULL AS DOUBLE)
                    AS corporate_event_day_quoted_full_spread
            FROM scoped a
            JOIN returns ret
              ON ret.permno = a.permno
             AND ret.calendar_date > a.entry_date
             AND ret.calendar_date <= a.exit_date
            JOIN sessions s ON s.calendar_date = ret.calendar_date
            JOIN factors f ON f.observation_date = ret.calendar_date
            LEFT JOIN market m
              ON m.permno = a.permno
             AND m.calendar_date = ret.calendar_date
            LEFT JOIN market entry_market
              ON entry_market.permno = a.permno
             AND entry_market.calendar_date = a.entry_date
            ORDER BY a.signal_date, ret.calendar_date, a.permno
            """
            )
            .fetch_arrow_table()
            .combine_chunks()
        )
    finally:
        connection.close()
    terminal_events_by_key = (
        {}
        if terminal_close_overlay is None
        else dict(terminal_close_overlay.events_by_return_key)  # type: ignore[union-attr]
    )
    terminal_projected_keys: set[tuple[int, dt.date]] = set()
    rows: list[dict[str, object]] = []
    empty_distribution_sha = _sha256_bytes(_canonical_json_bytes({"event_sha256": []}))
    for raw in base.to_pylist():
        row = dict(raw)
        key = (int(row["permno"]), row["calendar_date"])
        terminal_event = terminal_events_by_key.get(key)
        if terminal_event is not None:
            _require(
                row["source_holding_month"]
                == terminal_event.delisting_date.strftime("%Y-%m")
                and row["return_leg_class"]
                == "ordinary_with_unresolved_terminal_delisting"
                and row["covered_trading_sessions"] == 1
                and row["span_alignment_status"]
                == "ordinary_span_terminal_outcome_unresolved"
                and row["delisting_event_count"] == 1
                and row["delisting_treatment_status"]
                == "missing_nonperformance_terminal_leg_unresolved"
                and row["delisting_adjustment_return"] is None
                and row["authoritative_total_return"] is None
                and row["outcome_availability_date"]
                == terminal_event.outcome_availability_date
                and row["authoritative_return_state"]
                == "unresolved_nonperformance_delisting"
                and row["return_history_usable"] is False
                and row["return_history_authorized"] is False,
                "last-tradable-close accounting source row differs",
            )
            row.update(
                {
                    "return_leg_class": (
                        "ordinary_then_public_interest_last_tradable_close_liquidation"
                    ),
                    "span_alignment_status": (
                        "ordinary_span_then_last_tradable_close_liquidation"
                    ),
                    "delisting_treatment_status": terminal_close_overlay.treatment_status,  # type: ignore[union-attr]
                    "delisting_adjustment_return": Decimal("0.000000000000"),
                    "authoritative_total_return": (
                        terminal_event.ordinary_total_return
                    ),
                    "authoritative_return_state": (
                        "observed_ordinary_return_then_final_close_liquidation"
                    ),
                    "reconciliation_status": (
                        "post_unblind_last_tradable_close_liquidation_accounted_once"
                    ),
                    "return_history_usable": True,
                    "return_history_authorized": False,
                }
            )
            terminal_projected_keys.add(key)
        distribution = distribution_map.get(
            key,
            (0, empty_distribution_sha, True),
        )
        _require(
            int(row["delisting_event_count"]) >= 0,
            "negative delisting count escaped",
        )
        row["distribution_event_count"] = distribution[0]
        row["distribution_event_set_sha256"] = distribution[1]
        row["distribution_events_assigned_exactly_once"] = distribution[2]
        action = action_map.get(key)
        if action is not None:
            row.update(action)
        source_payload = {
            name: row[name]
            for name in (
                "permno",
                "calendar_date",
                "return_leg_class",
                "return_span_start_date",
                "return_span_end_date",
                "covered_trading_sessions",
                "authoritative_total_return",
                "authoritative_return_state",
                "reconciliation_status",
            )
        }
        row["source_payload_sha256"] = _source_row_sha256(source_payload)
        exact_no_mark = (
            row["authoritative_total_return"] is None
            and row["authoritative_return_state"] == "missing"
            and row["return_leg_class"] == "missing_return"
            and row["return_span_start_date"] is None
            and row["return_span_end_date"] == row["calendar_date"]
            and row["covered_trading_sessions"] == 0
            and row["span_alignment_status"] == "no_return_span"
            and row["delisting_event_count"] == 0
            and row["delisting_treatment_status"] == "none"
            and row["delisting_adjustment_return"] is None
            and row["return_history_usable"] is False
            and row["return_history_authorized"] is False
        )
        row["no_mark_source_record_sha256"] = (
            row["source_payload_sha256"] if exact_no_mark else None
        )
        rows.append(row)
    if terminal_close_overlay is not None:
        expected_terminal_keys = {
            (event.permno, event.delisting_date)
            for event in terminal_close_overlay.events  # type: ignore[union-attr]
            if event.delisting_date.strftime("%Y-%m") in pending_months
        }
        _require(
            terminal_projected_keys == expected_terminal_keys,
            "last-tradable-close accounting projection population differs",
        )
        for event in terminal_close_overlay.events:  # type: ignore[union-attr]
            if event.delisting_date.strftime("%Y-%m") not in pending_months:
                continue
            projected = [
                row
                for row in rows
                if row["source_holding_month"] == event.delisting_date.strftime("%Y-%m")
                and row["permno"] == event.permno
                and row["calendar_date"] == event.delisting_date
            ]
            posting_audit = [
                row
                for row in rows
                if row["source_holding_month"] == event.delisting_date.strftime("%Y-%m")
                and row["permno"] == event.permno
                and row["calendar_date"] == event.posting_date
            ]
            _require(
                len(projected) == 1
                and len(posting_audit) == 1
                and projected[0]["authoritative_total_return"]
                == event.ordinary_total_return
                and projected[0]["delisting_adjustment_return"]
                == Decimal("0.000000000000")
                and str(projected[0]["corporate_action_id"]).startswith(
                    "stage13-final-close-action-"
                )
                and projected[0]["corporate_terminal_total_return_already_applied"]
                is True
                and projected[0]["corporate_terminal_sensitivity_policy_sha256"]
                == terminal_close_overlay.policy_sha256
                and projected[0]["corporate_terminal_sensitivity_evidence_sha256"]
                == event.evidence_sha256
                and projected[0]["corporate_event_day_quoted_full_spread"]
                == float(event.event_day_quoted_full_spread)
                and posting_audit[0]["authoritative_total_return"] is None
                and posting_audit[0]["return_history_usable"] is False
                and posting_audit[0]["return_history_authorized"] is False
                and posting_audit[0]["corporate_action_id"] is None,
                "last-tradable-close accounting rows differ",
            )
    if boundary_event is not None:
        stitched_rows = [
            row
            for row in rows
            if row["source_holding_month"] == _CROSS_SCOPE_TERMINAL_STITCH_MONTH
            and row["permno"] == _CROSS_SCOPE_TERMINAL_STITCH_PERMNO
            and row["calendar_date"] == _CROSS_SCOPE_TERMINAL_STITCH_DATE
        ]
        posting_audit_rows = [
            row
            for row in rows
            if row["source_holding_month"] == _CROSS_SCOPE_TERMINAL_STITCH_MONTH
            and row["permno"] == _CROSS_SCOPE_TERMINAL_STITCH_PERMNO
            and row["calendar_date"] == _CROSS_SCOPE_TERMINAL_POSTING_DATE
        ]
        _require(
            len(stitched_rows) == 1 and len(posting_audit_rows) == 1,
            "cross-scope accounting rows are not unique",
        )
        stitched_accounting = stitched_rows[0]
        posting_accounting = posting_audit_rows[0]
        _require(
            stitched_accounting["source_payload_sha256"]
            == _CROSS_SCOPE_STITCHED_ACCOUNTING_PAYLOAD_SHA256
            and stitched_accounting["authoritative_total_return"]
            == Decimal("0.012929483356")
            and stitched_accounting["delisting_adjustment_return"]
            == Decimal("0.031716000000")
            and stitched_accounting["corporate_action_id"]
            == _CROSS_SCOPE_CORPORATE_ACTION_ID
            and stitched_accounting["corporate_terminal_total_return_already_applied"]
            is True
            and posting_accounting["source_payload_sha256"]
            == _CROSS_SCOPE_OUTCOME_ACCOUNTING_PAYLOAD_SHA256
            and posting_accounting["return_leg_class"]
            == "pre_scope_terminal_delisting_context_suppressed"
            and posting_accounting["authoritative_total_return"] is None
            and posting_accounting["corporate_action_id"] is None,
            "cross-scope accounting projection differs",
        )
    transport_audit = _normalize_final_market_accounting_transport_rows(rows)
    native = pa.Table.from_pylist(
        rows,
        schema=_native_schema("final_market_accounting"),
    )
    _require_no_nonnullable_nulls(native, label="final_market_accounting")
    _require(native.num_rows > 0, "final accounting formula emitted no rows")
    return native, MappingProxyType(
        {
            **dict(event_identity),
            "cross_scope_terminal_boundary_stitch": dict(boundary_receipt),
            "last_tradable_close_liquidation_overlay": dict(terminal_close_receipt),
            "accounting_numeric_transport": dict(transport_audit),
            "pending_month_count": len(pending_months),
            "rows": native.num_rows,
            "stage2_authoritative_return_reconciliation_reused": True,
            "post_unblind_terminal_return_overlay_applied": (
                terminal_close_overlay is not None
            ),
            "stage11_corporate_action_mapping_reused": True,
            "financing_rate_annual": 0.005,
            "per_row_holding_interval_cutoff_enforced": True,
            "passed_months_recomputed": 0,
        }
    )


def _compound_finite(values: Sequence[object]) -> float:
    result = 1.0
    _require(bool(values), "return path is empty")
    for raw in values:
        _require(raw is not None, "return path contains a missing value")
        value = float(raw)
        _require(math.isfinite(value) and value >= -1.0, "return path is invalid")
        result *= 1.0 + value
    output = result - 1.0
    _require(math.isfinite(output), "compounded return is nonfinite")
    return 0.0 if output == 0.0 else float(output)


def _materialize_final_factors(
    *,
    accounting: pa.Table,
    pending_months: tuple[str, ...],
) -> tuple[pa.Table, Mapping[str, object]]:
    rows: list[dict[str, object]] = []
    for month in pending_months:
        month_table = accounting.filter(
            pc.equal(
                accounting["source_holding_month"],
                pa.scalar(month, pa.string()),
            )
        )
        unique = (
            month_table.select(
                ["calendar_date", "mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]
            )
            .group_by("calendar_date")
            .aggregate(
                [
                    (name, "min")
                    for name in ("mktrf", "smb", "hml", "rmw", "cma", "umd", "rf")
                ]
            )
        )
        _require(unique.num_rows > 0, f"daily factor path is absent: {month}")
        rows.append(
            {
                "source_holding_month": month,
                **{
                    name: _compound_finite(unique[f"{name}_min"].to_pylist())
                    for name in ("mktrf", "smb", "hml", "rmw", "cma", "umd", "rf")
                },
            }
        )
    table = pa.Table.from_pylist(rows, schema=_native_schema("final_factors"))
    return table, MappingProxyType(
        {
            "rows": table.num_rows,
            "daily_frozen_factor_compounding_reused": True,
            "passed_months_recomputed": 0,
        }
    )


def _materialize_final_benchmarks(
    *,
    final_calendar: pa.Table,
    factor_table: pa.Table,
    decision_market_root: Path,
    outcome_market_root: Path,
    decision_return_root: Path,
    outcome_return_root: Path,
    pending_months: tuple[str, ...],
) -> tuple[pa.Table, Mapping[str, object]]:
    from portfolio_bot.validation.validation_data_candidate_builder import (
        RETURN_COLUMNS,
    )

    market = _parquet_union(
        (
            decision_market_root / "crsp_security_day.parquet",
            outcome_market_root / "crsp_security_day.parquet",
        )
    )
    returns = _parquet_union(
        (
            decision_return_root / "crsp_security_day_return.parquet",
            outcome_return_root / "crsp_security_day_return.parquet",
        ),
        columns=RETURN_COLUMNS,
    )
    pending = pa.array(pending_months, pa.string())
    calendar = final_calendar.filter(
        pc.is_in(final_calendar["source_holding_month"], value_set=pending)
    )
    connection = duckdb.connect(":memory:")
    try:
        for name, table in (
            ("calendar", calendar),
            ("market", market),
            ("returns", returns),
            ("factors", factor_table),
        ):
            connection.register(name, table)
        passive = connection.execute(
            """
            WITH tickers AS (
                SELECT c.source_holding_month, c.entry_date, c.exit_date,
                       m.source_ticker, m.permno,
                       row_number() OVER (
                           PARTITION BY c.source_holding_month, m.source_ticker
                           ORDER BY m.permno
                       ) AS ticker_rank
                FROM calendar c
                JOIN market m ON m.calendar_date = c.entry_date
                WHERE m.source_ticker IN ('SPY', 'QQQ')
            )
            SELECT t.source_holding_month, t.source_ticker,
                   r.calendar_date, r.authoritative_total_return
            FROM tickers t
            JOIN returns r
              ON r.permno = t.permno
             AND r.calendar_date > t.entry_date
             AND r.calendar_date <= t.exit_date
            WHERE t.ticker_rank = 1
            ORDER BY t.source_holding_month, t.source_ticker, r.calendar_date
            """
        ).fetch_arrow_table()
        cash = connection.execute(
            """
            SELECT c.source_holding_month, f.observation_date, f.rf
            FROM calendar c
            JOIN factors f
              ON f.observation_date > c.entry_date
             AND f.observation_date <= c.exit_date
            ORDER BY c.source_holding_month, f.observation_date
            """
        ).fetch_arrow_table()
    finally:
        connection.close()
    passive_rows = passive.to_pylist()
    cash_rows = cash.to_pylist()
    rows: list[dict[str, object]] = []
    for month in pending_months:
        rows.append(
            {
                "source_holding_month": month,
                "benchmark_id": "cash_or_one_month_tbill",
                "total_return": _compound_finite(
                    [
                        row["rf"]
                        for row in cash_rows
                        if row["source_holding_month"] == month
                    ]
                ),
            }
        )
        for ticker, benchmark_id in (
            ("SPY", "spy_total_return"),
            ("QQQ", "qqq_total_return"),
        ):
            values = [
                row["authoritative_total_return"]
                for row in passive_rows
                if row["source_holding_month"] == month
                and row["source_ticker"] == ticker
            ]
            rows.append(
                {
                    "source_holding_month": month,
                    "benchmark_id": benchmark_id,
                    "total_return": _compound_finite(values),
                }
            )
    table = pa.Table.from_pylist(rows, schema=_native_schema("final_benchmarks"))
    return table, MappingProxyType(
        {
            "rows": table.num_rows,
            "passive_benchmark_ids": [
                "cash_or_one_month_tbill",
                "spy_total_return",
                "qqq_total_return",
            ],
            "authoritative_total_returns_used": True,
            "passed_months_recomputed": 0,
        }
    )


def _label_factor_map(factors: pa.Table) -> Mapping[dt.date, object]:
    from portfolio_bot.features.label_math import FactorObservation

    result: dict[dt.date, FactorObservation] = {}
    for row in factors.to_pylist():
        observed = row["observation_date"]
        result[observed] = FactorObservation(
            calendar_date=observed,
            rf=float(row["rf"]),
            mktrf=float(row["mktrf"]),
            smb=float(row["smb"]),
            hml=float(row["hml"]),
            rmw=float(row["rmw"]),
            cma=float(row["cma"]),
            umd=float(row["umd"]),
            public_availability_date=row["public_availability_date"],
        )
    return MappingProxyType(result)


def _materialize_final_completed_labels(
    *,
    final_calendar: pa.Table,
    predictors: pa.Table,
    accounting: pa.Table,
    factor_table: pa.Table,
    trading_calendar_path: Path,
    final_contract: object,
    pending_months: tuple[str, ...],
) -> tuple[pa.Table, Mapping[str, object]]:
    from portfolio_bot.features.label_store_materialization import (
        DevelopmentCalendarRow,
        materialize_label_signal_month,
        project_development_labels,
    )

    rules = _json(
        PROJECT_ROOT / "configs/phase_03_features/label_build_rules_v1.json",
        label="frozen Stage 7 label rules",
    )
    sessions = tuple(
        pq.read_table(
            trading_calendar_path,
            columns=["calendar_date"],
            filters=[("calendar_date", "<=", FINAL_LAST_EXIT)],
            use_threads=False,
        )["calendar_date"].to_pylist()
    )
    factor_map = _label_factor_map(factor_table)
    calendar_by_month = {
        str(row["source_holding_month"]): row for row in final_calendar.to_pylist()
    }
    output_rows: list[dict[str, object]] = []
    for month in pending_months:
        period = calendar_by_month[month]
        feature = predictors.filter(
            pc.equal(predictors["source_holding_month"], pa.scalar(month, pa.string()))
        ).drop(["source_holding_month"])
        feature = feature.cast(final_contract.raw_feature_schema, safe=True)
        long_permnos = pa.array(
            [
                int(row["permno"])
                for row in feature.select(["permno", "long_eligible"]).to_pylist()
                if row["long_eligible"] is True
            ],
            pa.int64(),
        )
        routed = accounting.filter(
            pc.and_(
                pc.equal(
                    accounting["source_holding_month"],
                    pa.scalar(month, pa.string()),
                ),
                pc.is_in(accounting["permno"], value_set=long_permnos),
            )
        ).select(
            [
                "signal_date",
                *(
                    name
                    for name in (
                        "permno",
                        "calendar_date",
                        "return_leg_class",
                        "return_span_start_date",
                        "return_span_end_date",
                        "covered_trading_sessions",
                        "delisting_event_count",
                        "delisting_treatment_status",
                        "delisting_adjustment_return",
                        "authoritative_total_return",
                        "outcome_availability_date",
                        "authoritative_return_state",
                        "reconciliation_status",
                        "return_history_usable",
                        "return_history_authorized",
                    )
                ),
            ]
        )
        materialized = materialize_label_signal_month(
            feature_month=feature,
            routed_returns=routed,
            calendar=DevelopmentCalendarRow(
                signal_date=period["signal_date"],
                holding_month=month,
                entry_date=period["entry_date"],
                exit_date=period["exit_date"],
            ),
            sessions=sessions,
            factors=factor_map,
            rules=rules,
        )
        labels = project_development_labels(materialized.audit, rules=rules)
        for row in labels.to_pylist():
            if row["primary_label_status"] != "PRESENT":
                continue
            output_rows.append(
                {
                    "source_holding_month": month,
                    "signal_date": row["signal_date"],
                    "entry_date": row["entry_date"],
                    "exit_date": row["exit_date"],
                    "outcome_availability_date": period["outcome_availability_date"],
                    "permno": row["permno"],
                    "unranked_residual_return": row["unranked_residual_return"],
                    "residual_return_rank": row["residual_return_rank"],
                    "primary_label_status": row["primary_label_status"],
                }
            )
    table = pa.Table.from_pylist(
        output_rows,
        schema=_native_schema("final_completed_labels"),
    )
    _require(table.num_rows > 0, "final completed-label formula emitted no rows")
    return table, MappingProxyType(
        {
            "rows": table.num_rows,
            "pending_month_count": len(pending_months),
            "stage7_label_formula_reused": True,
            "only_primary_present_labels_published": True,
            "passed_months_recomputed": 0,
        }
    )


def _materialize_security_outcomes(
    *,
    name: str,
    accounting: pa.Table,
    predictor: pa.Table,
    pending_months: tuple[str, ...],
    completed_labels: pa.Table | None = None,
    calendar: pa.Table,
) -> tuple[pa.Table, Mapping[str, object]]:
    calendar_by_month = {
        str(row["source_holding_month"]): row for row in calendar.to_pylist()
    }
    predictor_rows = predictor.select(
        ["source_holding_month", "permno", "long_eligible"]
    ).to_pylist()
    eligible = {
        (str(row["source_holding_month"]), int(row["permno"]))
        for row in predictor_rows
        if row["long_eligible"] is True
    }
    if completed_labels is not None:
        eligible &= {
            (str(row["source_holding_month"]), int(row["permno"]))
            for row in completed_labels.select(
                ["source_holding_month", "permno"]
            ).to_pylist()
        }
    grouped: dict[tuple[str, int], list[tuple[dt.date, object]]] = defaultdict(list)
    for row in accounting.select(
        [
            "source_holding_month",
            "calendar_date",
            "permno",
            "authoritative_total_return",
        ]
    ).to_pylist():
        key = (str(row["source_holding_month"]), int(row["permno"]))
        if key in eligible and key[0] in pending_months:
            grouped[key].append(
                (row["calendar_date"], row["authoritative_total_return"])
            )
    rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        month, permno = key
        path = sorted(grouped[key])
        if any(value is None for _date, value in path):
            continue
        period = calendar_by_month[month]
        if name == "final_outcomes":
            rows.append(
                {
                    "source_holding_month": month,
                    "entry_date": period["entry_date"],
                    "exit_date": period["exit_date"],
                    "permno": permno,
                    "total_return": _compound_finite([value for _date, value in path]),
                }
            )
        else:
            rows.append(
                {
                    "source_holding_month": month,
                    "entry_date": period["entry_date"],
                    "administrative_close_date": period["exit_date"],
                    "permno": permno,
                    "partial_total_return": _compound_finite(
                        [value for _date, value in path]
                    ),
                }
            )
    table = pa.Table.from_pylist(rows, schema=_native_schema(name))
    _require(table.num_rows > 0, f"{name} formula emitted no rows")
    return table, MappingProxyType(
        {
            "rows": table.num_rows,
            "pending_month_count": len(pending_months),
            "authoritative_total_return_compounding_reused": True,
            "passed_months_recomputed": 0,
        }
    )


def execute_repository_native_final_replay(
    *,
    raw_authority: object,
    stage12_binding_sha256: str,
    approval_payload_sha256: str,
):
    """Execute the one repository-fixed, checkpointed final source replay.

    This is intentionally not a general builder API.  It accepts only sealed
    authority identities already issued by ``final_source_builder``; all
    paths, dates, tables, formulas, and checkpoint locations are fixed here.
    """

    from portfolio_bot.final_test.final_source_builder import (
        AuthenticatedFinalRawAuthority,
    )

    _require(
        type(raw_authority) is AuthenticatedFinalRawAuthority,
        "final replay raw authority type differs",
    )
    _require(
        _HEX64.fullmatch(stage12_binding_sha256) is not None,
        "Stage 12 binding SHA-256 is malformed",
    )
    _require(
        _HEX64.fullmatch(approval_payload_sha256) is not None,
        "Stage 13 approval payload SHA-256 is malformed",
    )
    raw_identity = dict(raw_authority.identity)
    snapshot_id = raw_identity.get("raw_snapshot_id")
    raw_files_sha = raw_identity.get("raw_file_identities_sha256")
    _require(
        isinstance(snapshot_id, str)
        and bool(snapshot_id)
        and isinstance(raw_files_sha, str)
        and _HEX64.fullmatch(raw_files_sha) is not None
        and len(raw_authority.raw_files) == 25,
        "final replay raw identity differs",
    )
    context = FinalReplayContext(
        raw_snapshot_id=snapshot_id,
        raw_file_identities_sha256=raw_files_sha,
        stage12_binding_sha256=stage12_binding_sha256,
        approval_payload_sha256=approval_payload_sha256,
    )
    database_root = _configured_database_root(PROJECT_ROOT)
    checkpoint_root = database_root / WORK_NAMESPACE
    formula_work = checkpoint_root / context.context_sha256 / "formula_work"
    # The metadata-only authority begins with zero reads.  Do not carry that
    # pre-replay counter into the empirical result: lagged final-period inputs
    # are intentionally opened under the Stage 13 gate and each formula run is
    # separately receipted by the engine.
    raw_identity.pop("raw_payload_files_opened", None)
    raw_identity.pop("final_period_rows_read", None)
    raw_identity.update(
        {
            "final_period_payload_accessed": True,
            "raw_payload_access_is_formula_receipted": True,
            "raw_payload_file_authority_count": len(raw_authority.raw_files),
            "per_row_information_cutoff_enforced": True,
        }
    )
    with _REPLAY_LOCK:
        engine = RepositoryNativeFinalFormulaEngine(
            database_root=database_root,
            work_root=formula_work,
            project_root=PROJECT_ROOT,
            replay_context=context,
        )
        return _execute_with_engine(
            context=context,
            raw_identity=MappingProxyType(raw_identity),
            engine=engine,
            checkpoint_root=checkpoint_root,
        )


__all__ = [
    "FinalReplayCheckpointStore",
    "FinalReplayContext",
    "FinalSourceReplayError",
    "execute_repository_native_final_replay",
]
