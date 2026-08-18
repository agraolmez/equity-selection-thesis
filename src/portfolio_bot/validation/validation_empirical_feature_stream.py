"""Empirical Stage 11 feature stream built from frozen Stage 6 formulas.

This adapter changes only the permitted date/cardinality scope around the
unchanged Stage 6 formula implementations.  It authenticates the 26-table
path catalog once, checkpoints the single-scan CRSP router, aligns all seven
family streams by exact ``(signal_date, permno)`` key, and sends bounded
batches to the Stage 11 candidate writer.  The sole 2014-11-28 OptionMetrics
row set is consumed internally and never enters a scored payload.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import stat
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

import portfolio_bot.features.materialize_aggregate_state as aggregate_adapter
import portfolio_bot.features.materialize_crsp as crsp_adapter
import portfolio_bot.features.materialize_crsp_streaming as crsp_streaming
import portfolio_bot.features.materialize_fundamentals as fundamental_adapter
import portfolio_bot.features.materialize_optionmetrics as optionmetrics_adapter
import portfolio_bot.features.materialize_ownership_events as ownership_adapter
from portfolio_bot.features.feature_store_materialization import (
    AuthenticatedInputReceipt,
    FeatureMaterializationRow,
    FeatureStoreBuildIdentity,
    compute_feature_store_build_identity,
)
from portfolio_bot.features.materialize_aggregate_state import (
    materialize_aggregate_state,
)
from portfolio_bot.features.materialize_crsp import CRSP_FAMILIES
from portfolio_bot.features.materialize_crsp_routing import (
    CrspSignalRouter,
    spool_crsp_signal_router,
)
from portfolio_bot.features.materialize_crsp_streaming import (
    iter_unfinalized_crsp_rows,
)
from portfolio_bot.features.materialize_fundamentals import (
    iter_fundamental_feature_batches,
)
from portfolio_bot.features.materialize_optionmetrics import (
    iter_optionmetrics_feature_batches,
)
from portfolio_bot.features.materialize_ownership_events import (
    iter_ownership_event_feature_batches,
)
from portfolio_bot.features.phase_03_contract import canonical_json_bytes
from portfolio_bot.validation.validation_data_candidate_builder import (
    FACTOR_COLUMNS,
    RETURN_COLUMNS,
    OptionMetricsContextRow,
    StreamingFeatureWriteReceipt,
    ValidationFeatureRow,
    ValidationRiskCostMaterialization,
    ValidationRiskHistoryInputs,
    write_validation_candidate_streaming,
)
from portfolio_bot.validation.validation_build_code_identity import (
    materialization_code_hashes,
    materialization_code_hashes_sha256,
)
from portfolio_bot.validation.validation_data_contract import (
    FIRST_SIGNAL,
    INTERNAL_CONTEXT_SIGNAL,
    LAST_SIGNAL,
    PROJECT_ROOT,
    ValidationDataContract,
    schema_sha256,
    sha256_file,
    validate_validation_data_contract,
)
from portfolio_bot.validation.validation_feature_input_adapter import (
    ValidationFeatureInputBundle,
    load_validation_feature_input_bundle,
)
from portfolio_bot.validation.validation_fundamental_adapter import (
    FISCAL_TRANSITION_POLICY,
    select_validation_fundamental_bundle,
    validate_validation_fundamental_source_gates,
    validation_fiscal_transition_audit,
)


EMPIRICAL_STREAM_ID = "stage11-empirical-frozen-formula-stream-v1"
EMPIRICAL_BUILD_STATUS = "STAGE11_EMPIRICAL_CANDIDATE_BUILT_NO_OUTCOMES"
_EMPIRICAL_BUILD_SCHEMA_VERSION = 1
_EMPIRICAL_BUILD_RECEIPT_SEAL = object()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FEATURE_INPUT_ADAPTER_CODE = (
    "src/portfolio_bot/validation/validation_feature_input_adapter.py"
)
_FORMULA_SCOPE_LOCK = threading.Lock()
_FAMILY_ORDER = (
    "price_trends_path",
    "risk_return_distribution",
    "size_liquidity_trading_frictions",
    "valuation_fundamentals",
    "ownership_filings_events_segments",
    "optionmetrics",
    "aggregate_state",
)


class ValidationEmpiricalFeatureStreamError(RuntimeError):
    """Raised when empirical frozen-formula streams cannot be aligned safely."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationEmpiricalFeatureStreamError(message)


def _require_predictor_authority(authority: object) -> ValidationDataContract:
    _check(
        type(authority) is ValidationDataContract,
        "Empirical feature access requires the exact no-outcome predictor authority",
    )
    assert isinstance(authority, ValidationDataContract)
    current = validate_validation_data_contract()
    _check(
        authority.scored_signal_dates[0] == FIRST_SIGNAL
        and authority.scored_signal_dates[-1] == LAST_SIGNAL
        and len(authority.scored_signal_dates) == 59,
        "No-outcome predictor authority scope differs",
    )
    _check(
        authority.policy_sha256 == current.policy_sha256
        and authority.raw_feature_schema.equals(
            current.raw_feature_schema, check_metadata=True
        ),
        "No-outcome predictor authority is stale or differs",
    )
    return authority


def _derived_identity(
    source: FeatureStoreBuildIdentity,
    inputs: Any,
) -> FeatureStoreBuildIdentity:
    """Replace only changed catalog receipts without rehashing shared GB files."""

    receipts = tuple(
        AuthenticatedInputReceipt(
            gate=table.gate,
            name=name,
            sha256=table.sha256,
            size_bytes=table.size_bytes,
            rows=table.rows,
            row_groups=table.row_groups,
        )
        for name, table in sorted(inputs.tables.items())
    )
    content = {
        "schema_version": 1,
        "contract_id": source.contract_id,
        "policy_id": source.policy_id,
        "configuration_hashes": source.configuration_hash_map,
        "code_hashes": source.code_hash_map,
        "input_receipts": [item.as_dict() for item in receipts],
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    return FeatureStoreBuildIdentity(
        contract_id=source.contract_id,
        policy_id=source.policy_id,
        build_id=f"phase03-features-{fingerprint[:24]}",
        content_fingerprint=fingerprint,
        configuration_hashes=source.configuration_hashes,
        code_hashes=source.code_hashes,
        input_receipts=receipts,
    )


@contextmanager
def _formula_scope(
    assignments: Sequence[tuple[Any, str, Any]],
) -> Iterator[None]:
    """Install one closed scope assignment set and always restore it."""

    _check(bool(assignments), "Formula scope assignments are empty")
    keys = tuple((module.__name__, name) for module, name, _value in assignments)
    _check(len(keys) == len(set(keys)), "Formula scope assignment repeats")
    acquired = _FORMULA_SCOPE_LOCK.acquire(blocking=False)
    _check(acquired, "Another Stage 11 formula scope is active")
    originals = [
        (module, name, getattr(module, name)) for module, name, _value in assignments
    ]
    try:
        for module, name, value in assignments:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)
        _FORMULA_SCOPE_LOCK.release()


@contextmanager
def _validation_formula_scope(
    *,
    scored_rows: int,
    optionmetrics_working_rows: int,
) -> Iterator[None]:
    """Temporarily install the exact Stage 11 scope; restore on every exit."""

    _check(
        scored_rows > 0 and optionmetrics_working_rows > scored_rows,
        "Formula scope cardinality differs",
    )
    assignments = (
        (crsp_adapter, "DEVELOPMENT_LAST_SIGNAL", LAST_SIGNAL),
        (crsp_adapter, "MAXIMUM_CONTEXT_DATE", LAST_SIGNAL),
        (crsp_streaming, "DEVELOPMENT_LAST_SIGNAL", LAST_SIGNAL),
        (crsp_streaming, "MAXIMUM_CONTEXT_DATE", LAST_SIGNAL),
        (fundamental_adapter, "FIRST_SIGNAL_DATE", FIRST_SIGNAL),
        (fundamental_adapter, "LAST_SIGNAL_DATE", LAST_SIGNAL),
        (fundamental_adapter, "MAXIMUM_CONTEXT_DATE", LAST_SIGNAL),
        (fundamental_adapter, "EXPECTED_SPINE_ROWS", scored_rows),
        (fundamental_adapter, "EXPECTED_SIGNAL_DATES", 59),
        (
            fundamental_adapter,
            "_validate_source_gates",
            validate_validation_fundamental_source_gates,
        ),
        (
            fundamental_adapter,
            "select_fundamental_bundle",
            select_validation_fundamental_bundle,
        ),
        (
            ownership_adapter,
            "select_fundamental_bundle",
            select_validation_fundamental_bundle,
        ),
        (ownership_adapter, "FIRST_SIGNAL_DATE", FIRST_SIGNAL),
        (ownership_adapter, "LAST_SIGNAL_DATE", LAST_SIGNAL),
        (optionmetrics_adapter, "FIRST_SIGNAL_DATE", INTERNAL_CONTEXT_SIGNAL),
        (optionmetrics_adapter, "LAST_SIGNAL_DATE", LAST_SIGNAL),
        (optionmetrics_adapter, "EXPECTED_SPINE_ROWS", optionmetrics_working_rows),
        (optionmetrics_adapter, "EXPECTED_SIGNAL_DATES", 60),
        (aggregate_adapter, "PREDICTOR_END_DATE", LAST_SIGNAL),
        (aggregate_adapter, "CONTEXT_END_DATE", LAST_SIGNAL),
    )
    with _formula_scope(assignments):
        yield


@contextmanager
def _development_formula_scope_identity() -> Iterator[None]:
    """Execute the same scope mechanism with the exact Stage 6 values.

    This is used only by the byte-authenticated development-equivalence gate;
    it reads no market row and must be an observable no-op.
    """

    assignments = (
        (crsp_adapter, "DEVELOPMENT_LAST_SIGNAL", dt.date(2014, 10, 31)),
        (crsp_adapter, "MAXIMUM_CONTEXT_DATE", dt.date(2014, 12, 1)),
        (crsp_streaming, "DEVELOPMENT_LAST_SIGNAL", dt.date(2014, 10, 31)),
        (crsp_streaming, "MAXIMUM_CONTEXT_DATE", dt.date(2014, 12, 1)),
        (fundamental_adapter, "FIRST_SIGNAL_DATE", dt.date(2000, 1, 31)),
        (fundamental_adapter, "LAST_SIGNAL_DATE", dt.date(2014, 10, 31)),
        (fundamental_adapter, "MAXIMUM_CONTEXT_DATE", dt.date(2014, 12, 1)),
        (fundamental_adapter, "EXPECTED_SPINE_ROWS", 1_263_048),
        (fundamental_adapter, "EXPECTED_SIGNAL_DATES", 178),
        (ownership_adapter, "FIRST_SIGNAL_DATE", dt.date(2000, 1, 31)),
        (ownership_adapter, "LAST_SIGNAL_DATE", dt.date(2014, 10, 31)),
        (optionmetrics_adapter, "FIRST_SIGNAL_DATE", dt.date(2000, 1, 31)),
        (optionmetrics_adapter, "LAST_SIGNAL_DATE", dt.date(2014, 10, 31)),
        (optionmetrics_adapter, "EXPECTED_SPINE_ROWS", 1_263_048),
        (optionmetrics_adapter, "EXPECTED_SIGNAL_DATES", 178),
        (aggregate_adapter, "PREDICTOR_END_DATE", dt.date(2014, 10, 31)),
        (aggregate_adapter, "CONTEXT_END_DATE", dt.date(2014, 12, 1)),
    )
    originals = tuple(getattr(module, name) for module, name, _value in assignments)
    expected = tuple(value for _module, _name, value in assignments)
    _check(
        originals == expected,
        "Development scope is not already the frozen Stage 6 scope",
    )
    with _formula_scope(assignments):
        _check(
            tuple(getattr(module, name) for module, name, _value in assignments)
            == expected,
            "Development formula scope is not an exact no-op",
        )
        yield
    _check(
        tuple(getattr(module, name) for module, name, _value in assignments)
        == originals,
        "Development formula scope was not restored",
    )


def _flatten(
    batches: Iterable[Sequence[FeatureMaterializationRow]],
) -> Iterator[FeatureMaterializationRow]:
    for batch in batches:
        _check(bool(batch), "Frozen family emitted an empty batch")
        for row in batch:
            _check(
                isinstance(row, FeatureMaterializationRow),
                "Frozen family emitted an invalid row",
            )
            yield row


def _row_key(row: FeatureMaterializationRow) -> tuple[dt.date, int]:
    return row.signal_date, int(row.permno)


def _same_audit_row(
    first: FeatureMaterializationRow,
    second: FeatureMaterializationRow,
) -> bool:
    return (
        first.signal_date == second.signal_date
        and first.permno == second.permno
        and first.permco == second.permco
        and first.long_eligible is second.long_eligible
        and first.short_eligible is second.short_eligible
        and first.primary_exchange == second.primary_exchange
        and first.share_type == second.share_type
    )


def _combined_crsp_rows(
    router: CrspSignalRouter,
    *,
    batch_size: int,
) -> Iterator[tuple[FeatureMaterializationRow, Mapping[str, Any]]]:
    active_signal: dt.date | None = None
    by_family: dict[str, list[FeatureMaterializationRow]] = {}

    def finish() -> Iterator[tuple[FeatureMaterializationRow, Mapping[str, Any]]]:
        _check(
            set(by_family) == set(CRSP_FAMILIES),
            "CRSP signal does not contain three families",
        )
        ordered = [by_family[name] for name in CRSP_FAMILIES]
        lengths = {len(values) for values in ordered}
        _check(
            len(lengths) == 1 and next(iter(lengths)) > 0,
            "CRSP family cardinalities differ",
        )
        for rows in zip(*ordered, strict=True):
            first = rows[0]
            _check(
                all(_same_audit_row(first, row) for row in rows[1:]),
                "CRSP family keys/audit fields differ",
            )
            yield first, MappingProxyType(
                {
                    family: row.result
                    for family, row in zip(CRSP_FAMILIES, rows, strict=True)
                }
            )

    for routed in router.iter_family_row_batches(batch_size=batch_size):
        if active_signal is None:
            active_signal = routed.signal_date
        elif routed.signal_date != active_signal:
            yield from finish()
            _check(
                routed.signal_date > active_signal, "CRSP routed signal order regressed"
            )
            active_signal = routed.signal_date
            by_family = {}
        by_family.setdefault(routed.family, []).extend(routed.rows)
    if active_signal is not None:
        yield from finish()


def _split_optionmetrics_context(
    rows: Iterator[FeatureMaterializationRow],
) -> tuple[tuple[OptionMetricsContextRow, ...], Iterator[FeatureMaterializationRow]]:
    context: list[OptionMetricsContextRow] = []
    first_scored: FeatureMaterializationRow | None = None
    for row in rows:
        if row.signal_date == INTERNAL_CONTEXT_SIGNAL:
            context.append(
                OptionMetricsContextRow(
                    signal_date=row.signal_date,
                    permno=int(row.permno),
                    result=row.result,
                )
            )
            continue
        first_scored = row
        break
    _check(bool(context), "OptionMetrics internal lag context is absent")
    _check(
        first_scored is not None and first_scored.signal_date == FIRST_SIGNAL,
        "OptionMetrics scored stream starts incorrectly",
    )
    return tuple(context), chain((first_scored,), rows)


def _aligned_scored_batches(
    *,
    router: CrspSignalRouter,
    fundamental_rows: Iterator[FeatureMaterializationRow],
    ownership_rows: Iterator[FeatureMaterializationRow],
    optionmetrics_rows: Iterator[FeatureMaterializationRow],
    aggregate_rows: Iterator[FeatureMaterializationRow],
    batch_size: int,
) -> Iterator[tuple[ValidationFeatureRow, ...]]:
    streams = {
        "valuation_fundamentals": fundamental_rows,
        "ownership_filings_events_segments": ownership_rows,
        "optionmetrics": optionmetrics_rows,
        "aggregate_state": aggregate_rows,
    }
    pending: list[ValidationFeatureRow] = []
    prior: tuple[dt.date, int] | None = None
    count = 0
    for crsp_row, crsp_results in _combined_crsp_rows(router, batch_size=batch_size):
        key = _row_key(crsp_row)
        _check(prior is None or key > prior, "Empirical feature key order regressed")
        prior = key
        results = dict(crsp_results)
        for family, stream in streams.items():
            try:
                row = next(stream)
            except StopIteration as exc:
                raise ValidationEmpiricalFeatureStreamError(
                    f"Family stream ended early: {family}"
                ) from exc
            _check(
                _same_audit_row(crsp_row, row),
                f"Family stream key/audit differs: {family}",
            )
            results[family] = row.result
        _check(tuple(results) == _FAMILY_ORDER, "Empirical family order differs")
        pending.append(
            ValidationFeatureRow(
                signal_date=crsp_row.signal_date,
                permno=int(crsp_row.permno),
                permco=int(crsp_row.permco),
                long_eligible=bool(crsp_row.long_eligible),
                short_eligible=bool(crsp_row.short_eligible),
                primary_exchange=crsp_row.primary_exchange,
                share_type=crsp_row.share_type,
                family_results=MappingProxyType(results),
            )
        )
        count += 1
        if len(pending) >= batch_size:
            yield tuple(pending)
            pending.clear()
    for family, stream in streams.items():
        try:
            extra = next(stream)
        except StopIteration:
            continue
        raise ValidationEmpiricalFeatureStreamError(
            f"Family stream contains an extra key: {family}/{_row_key(extra)}"
        )
    _check(count == router.row_count, "Empirical aligned row count differs")
    if pending:
        yield tuple(pending)


def validation_risk_history_inputs(
    bundle: ValidationFeatureInputBundle,
    *,
    predictor_authority: ValidationDataContract,
) -> ValidationRiskHistoryInputs:
    """Create the exact projected, pre-cutoff risk-history readers."""

    _require_predictor_authority(predictor_authority)
    inputs = bundle.scored_inputs
    calendar = pq.read_table(
        inputs.path("trading_calendar.parquet"),
        columns=["calendar_date"],
        filters=[("calendar_date", "<=", LAST_SIGNAL)],
        use_threads=False,
    )
    factors = pq.read_table(
        inputs.path("ff_factor_daily_observation.parquet"),
        columns=list(FACTOR_COLUMNS),
        filters=[
            ("observation_date", "<=", LAST_SIGNAL),
            ("public_availability_date", "<=", LAST_SIGNAL),
        ],
        use_threads=False,
    )
    return_path = inputs.path("crsp_security_day_return.parquet")

    def batches(batch_rows: int) -> Iterable[pa.RecordBatch]:
        yield from pq.ParquetFile(return_path).iter_batches(
            batch_size=batch_rows,
            columns=list(RETURN_COLUMNS),
            use_threads=False,
        )

    return ValidationRiskHistoryInputs(
        trading_calendar=calendar,
        factors=factors,
        return_batch_factory=batches,
        label_rows_read=0,
        outcome_rows_read=0,
        final_period_rows_read=0,
    )


def _absolute_path(value: Path) -> Path:
    return Path(os.path.abspath(value.expanduser()))


def _stable_regular_file_sha256(path: Path, *, label: str) -> tuple[str, int]:
    lexical = _absolute_path(path)
    _check(lexical.is_file() and not lexical.is_symlink(), f"Missing or unsafe {label}")
    before = lexical.stat(follow_symlinks=False)
    _check(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, f"Unsafe {label}")
    digest = sha256_file(lexical)
    after = lexical.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    _check(identity(before) == identity(after), f"{label} changed while hashed")
    return digest, int(before.st_size)


def _json_document(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    lexical = _absolute_path(path)
    _check(lexical.is_file() and not lexical.is_symlink(), f"Missing or unsafe {label}")
    before = lexical.stat(follow_symlinks=False)
    _check(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, f"Unsafe {label}")
    payload = lexical.read_bytes()
    after = lexical.stat(follow_symlinks=False)
    _check(
        (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        f"{label} changed while read",
    )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationEmpiricalFeatureStreamError(f"Invalid {label} JSON") from exc
    _check(isinstance(value, dict), f"{label} is not an object")
    _check(payload == canonical_json_bytes(value), f"{label} is not canonical JSON")
    return value, payload


def _empirical_receipt_target(receipt_path: Path, candidate_root: Path) -> Path:
    path = _absolute_path(receipt_path)
    parent = path.parent
    _check(
        parent.is_dir() and not parent.is_symlink(),
        "Unsafe empirical-build receipt directory",
    )
    _check(
        not path.exists() and not path.is_symlink(),
        "Empirical-build receipt destination already exists",
    )
    root = _absolute_path(candidate_root)
    _check(
        path != root and root not in path.parents,
        "Empirical-build receipt must be outside the candidate root",
    )
    project = PROJECT_ROOT.resolve()
    try:
        relative = path.relative_to(project)
    except ValueError as exc:
        raise ValidationEmpiricalFeatureStreamError(
            "Empirical-build receipt escaped the project artifact root"
        ) from exc
    _check(
        relative.parts[:2] == ("artifacts", "phase_05_validation"),
        "Empirical-build receipt escaped the Stage 11 artifact namespace",
    )
    _check(
        path.resolve(strict=False) == path,
        "Empirical-build receipt uses a symlinked path",
    )
    return path


def _candidate_output_receipts(
    root: Path,
    contract: ValidationDataContract,
) -> Mapping[str, Mapping[str, Any]]:
    candidate = _absolute_path(root)
    _check(
        candidate.is_dir() and not candidate.is_symlink(),
        "Unsafe empirical candidate root",
    )
    expected_names = {item.filename for item in contract.payload_specs.values()}
    _check(
        {item.name for item in candidate.iterdir()} == expected_names,
        "Empirical candidate root is not the exact twelve-payload set",
    )
    receipts: dict[str, Mapping[str, Any]] = {}
    for role, specification in contract.payload_specs.items():
        path = candidate / specification.filename
        digest, size = _stable_regular_file_sha256(
            path, label=f"candidate output {role}"
        )
        parquet = pq.ParquetFile(path)
        _check(
            parquet.schema_arrow.equals(specification.schema, check_metadata=True),
            f"Empirical output schema differs: {role}",
        )
        receipts[role] = MappingProxyType(
            {
                "role": role,
                "path": specification.filename,
                "sha256": digest,
                "size_bytes": size,
                "rows": int(parquet.metadata.num_rows),
                "row_groups": int(parquet.metadata.num_row_groups),
                "schema_sha256": schema_sha256(parquet.schema_arrow),
                "minimum_signal_date": contract.scored_signal_dates[0].isoformat(),
                "maximum_signal_date": contract.scored_signal_dates[-1].isoformat(),
            }
        )
    return MappingProxyType(receipts)


def _receipt_content_fingerprint(document: Mapping[str, Any]) -> str:
    content = {
        key: value
        for key, value in document.items()
        if key not in {"build_id", "content_fingerprint"}
    }
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def _empirical_build_document(
    *,
    candidate_root: Path,
    bundle: ValidationFeatureInputBundle,
    feature_input_receipt_sha256: str,
    scored_identity: FeatureStoreBuildIdentity,
    option_identity: FeatureStoreBuildIdentity,
    router: CrspSignalRouter,
    router_database_sha256: str,
    router_database_size: int,
    outputs: Mapping[str, Mapping[str, Any]],
    fundamental_fiscal_transition: Mapping[str, Any],
    start_code_hashes: Mapping[str, str],
    end_code_hashes: Mapping[str, str],
) -> dict[str, Any]:
    start_digest = materialization_code_hashes_sha256(start_code_hashes)
    end_digest = materialization_code_hashes_sha256(end_code_hashes)
    _check(
        dict(start_code_hashes) == dict(end_code_hashes) and start_digest == end_digest,
        "Materialization code changed while the empirical candidate was built",
    )
    output_values = {role: dict(value) for role, value in sorted(outputs.items())}
    output_digest = hashlib.sha256(canonical_json_bytes(output_values)).hexdigest()
    document: dict[str, Any] = {
        "schema_version": _EMPIRICAL_BUILD_SCHEMA_VERSION,
        "status": EMPIRICAL_BUILD_STATUS,
        "empirical_stream_id": EMPIRICAL_STREAM_ID,
        "candidate_root": _absolute_path(candidate_root).as_posix(),
        "feature_input": {
            "receipt_path": _absolute_path(bundle.receipt_path).as_posix(),
            "receipt_sha256": feature_input_receipt_sha256,
            "input_bundle_sha256": bundle.input_bundle_sha256,
            "input_bundle_id": f"stage11-feature-inputs-{bundle.input_bundle_sha256[:24]}",
            "logical_history_root": _absolute_path(
                bundle.logical_history_root
            ).as_posix(),
            "adapter_code_path": _FEATURE_INPUT_ADAPTER_CODE,
            "adapter_code_sha256": start_code_hashes[_FEATURE_INPUT_ADAPTER_CODE],
        },
        "feature_identities": {
            "scored_build_id": scored_identity.build_id,
            "scored_content_fingerprint": scored_identity.content_fingerprint,
            "optionmetrics_working_build_id": option_identity.build_id,
            "optionmetrics_working_content_fingerprint": option_identity.content_fingerprint,
        },
        "fundamental_fiscal_transition": dict(fundamental_fiscal_transition),
        "crsp_router": {
            "spool_path": _absolute_path(router.spool_path).as_posix(),
            "database_path": _absolute_path(router.database_path).as_posix(),
            "database_sha256": router_database_sha256,
            "database_size_bytes": router_database_size,
            "source_id": router.source_id,
            "row_count": router.row_count,
            "signal_date_count": len(router.signal_dates),
            "first_signal_date": router.signal_dates[0].isoformat(),
            "last_signal_date": router.signal_dates[-1].isoformat(),
            "reused_existing_at_build": router.reused_existing,
        },
        "outputs": output_values,
        "output_receipts_sha256": output_digest,
        "code_identity": {
            "start_hashes": dict(start_code_hashes),
            "start_hashes_sha256": start_digest,
            "end_hashes": dict(end_code_hashes),
            "end_hashes_sha256": end_digest,
            "unchanged_during_build": True,
        },
        "access_counters": {
            "validation_label_rows_read": 0,
            "validation_outcome_rows_read": 0,
            "validation_forward_return_rows_read": 0,
            "final_period_rows_read": 0,
            "weights_trades_or_pnl_rows_created": 0,
        },
    }
    fingerprint = _receipt_content_fingerprint(document)
    document["content_fingerprint"] = fingerprint
    document["build_id"] = f"stage11-empirical-candidate-{fingerprint[:24]}"
    return document


@dataclass(frozen=True, slots=True)
class EmpiricalBuildReceipt:
    """Immutable canonical empirical-build receipt payload."""

    canonical_bytes: bytes = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        value = json.loads(self.canonical_bytes)
        _check(isinstance(value, dict), "Empirical-build receipt is not an object")
        return value

    @property
    def build_id(self) -> str:
        return str(self.as_dict()["build_id"])

    @property
    def content_fingerprint(self) -> str:
        return str(self.as_dict()["content_fingerprint"])


@dataclass(frozen=True, slots=True)
class ValidatedEmpiricalCandidateBuild:
    """Opaque authority issued only after complete empirical reauthentication."""

    receipt: EmpiricalBuildReceipt
    receipt_path: Path
    receipt_sha256: str
    candidate_root: Path
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _check(
            self._seal is _EMPIRICAL_BUILD_RECEIPT_SEAL,
            "Empirical-build authority is not authenticated",
        )


@dataclass(frozen=True, slots=True)
class EmpiricalCandidateReceipt:
    feature_receipt: StreamingFeatureWriteReceipt
    risk_cost: ValidationRiskCostMaterialization
    scored_build_id: str
    optionmetrics_working_build_id: str
    crsp_router_reused: bool
    input_bundle_sha256: str
    empirical_build: ValidatedEmpiricalCandidateBuild


def _exact_keys(value: object, expected: set[str], *, label: str) -> Mapping[str, Any]:
    _check(isinstance(value, dict), f"{label} is not an object")
    _check(set(value) == expected, f"{label} keys differ")
    return value


def _safe_existing_empirical_receipt(path: Path) -> Path:
    lexical = _absolute_path(path)
    project = PROJECT_ROOT.resolve()
    try:
        relative = lexical.relative_to(project)
    except ValueError as exc:
        raise ValidationEmpiricalFeatureStreamError(
            "Empirical-build receipt escaped the project root"
        ) from exc
    _check(
        relative.parts[:2] == ("artifacts", "phase_05_validation"),
        "Empirical-build receipt escaped the Stage 11 artifact namespace",
    )
    _check(
        lexical.resolve(strict=False) == lexical
        and lexical.is_file()
        and not lexical.is_symlink(),
        "Unsafe empirical-build receipt",
    )
    return lexical


def _validate_empirical_document_shape(
    document: Mapping[str, Any],
    contract: ValidationDataContract,
) -> None:
    _exact_keys(
        document,
        {
            "schema_version",
            "status",
            "empirical_stream_id",
            "candidate_root",
            "feature_input",
            "feature_identities",
            "fundamental_fiscal_transition",
            "crsp_router",
            "outputs",
            "output_receipts_sha256",
            "code_identity",
            "access_counters",
            "content_fingerprint",
            "build_id",
        },
        label="empirical-build receipt",
    )
    _check(
        document["schema_version"] == _EMPIRICAL_BUILD_SCHEMA_VERSION
        and document["status"] == EMPIRICAL_BUILD_STATUS
        and document["empirical_stream_id"] == EMPIRICAL_STREAM_ID,
        "Empirical-build receipt version/status/stream differs",
    )
    feature_input = _exact_keys(
        document["feature_input"],
        {
            "receipt_path",
            "receipt_sha256",
            "input_bundle_sha256",
            "input_bundle_id",
            "logical_history_root",
            "adapter_code_path",
            "adapter_code_sha256",
        },
        label="empirical-build feature input",
    )
    _exact_keys(
        document["feature_identities"],
        {
            "scored_build_id",
            "scored_content_fingerprint",
            "optionmetrics_working_build_id",
            "optionmetrics_working_content_fingerprint",
        },
        label="empirical-build feature identities",
    )
    transition = _exact_keys(
        document["fundamental_fiscal_transition"],
        {
            "resolution_policy",
            "duplicate_group_count",
            "duplicate_row_count",
            "affected_security_count",
            "affected_signal_row_count",
            "pair_signal_instance_count",
            "availability_tie_group_count",
        },
        label="empirical-build fundamental fiscal-transition audit",
    )
    _check(
        transition["resolution_policy"] == FISCAL_TRANSITION_POLICY
        and all(
            type(transition[name]) is int and transition[name] >= 0
            for name in set(transition) - {"resolution_policy"}
        )
        and transition["duplicate_group_count"] > 0
        and transition["duplicate_row_count"] >= transition["duplicate_group_count"] * 2
        and transition["availability_tie_group_count"] == 0,
        "Empirical-build fiscal-transition audit differs",
    )
    _exact_keys(
        document["crsp_router"],
        {
            "spool_path",
            "database_path",
            "database_sha256",
            "database_size_bytes",
            "source_id",
            "row_count",
            "signal_date_count",
            "first_signal_date",
            "last_signal_date",
            "reused_existing_at_build",
        },
        label="empirical-build CRSP router",
    )
    code = _exact_keys(
        document["code_identity"],
        {
            "start_hashes",
            "start_hashes_sha256",
            "end_hashes",
            "end_hashes_sha256",
            "unchanged_during_build",
        },
        label="empirical-build code identity",
    )
    _exact_keys(
        document["access_counters"],
        {
            "validation_label_rows_read",
            "validation_outcome_rows_read",
            "validation_forward_return_rows_read",
            "final_period_rows_read",
            "weights_trades_or_pnl_rows_created",
        },
        label="empirical-build access counters",
    )
    outputs = document["outputs"]
    _check(
        isinstance(outputs, dict) and set(outputs) == set(contract.payload_specs),
        "Empirical-build output role set differs",
    )
    output_keys = {
        "role",
        "path",
        "sha256",
        "size_bytes",
        "rows",
        "row_groups",
        "schema_sha256",
        "minimum_signal_date",
        "maximum_signal_date",
    }
    for role, value in outputs.items():
        receipt = _exact_keys(value, output_keys, label=f"empirical output {role}")
        _check(receipt["role"] == role, f"Empirical output role differs: {role}")
    for digest in (
        feature_input["receipt_sha256"],
        feature_input["input_bundle_sha256"],
        feature_input["adapter_code_sha256"],
        document["output_receipts_sha256"],
        document["content_fingerprint"],
        code["start_hashes_sha256"],
        code["end_hashes_sha256"],
    ):
        _check(
            isinstance(digest, str) and _SHA256.fullmatch(digest),
            "Malformed empirical-build digest",
        )
    _check(
        all(value == 0 for value in document["access_counters"].values()),
        "Empirical-build receipt reports a prohibited read/output",
    )


def _write_empirical_build_receipt(
    document: Mapping[str, Any],
    destination: Path,
) -> tuple[Path, bytes, str]:
    payload = canonical_json_bytes(dict(document))
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValidationEmpiricalFeatureStreamError(
            "Empirical-build receipt destination already exists"
        ) from exc
    digest, size = _stable_regular_file_sha256(
        destination,
        label="written empirical-build receipt",
    )
    _check(
        size == len(payload) and destination.read_bytes() == payload,
        "Written empirical-build receipt differs",
    )
    return destination, payload, digest


def load_and_reauthenticate_empirical_candidate_build(
    receipt_path: Path,
    contract: ValidationDataContract | None = None,
) -> ValidatedEmpiricalCandidateBuild:
    """Reauthenticate every source, code, spool, and output bound at build time."""

    authority = _require_predictor_authority(
        validate_validation_data_contract() if contract is None else contract
    )
    path = _safe_existing_empirical_receipt(receipt_path)
    document, receipt_bytes = _json_document(path, label="empirical-build receipt")
    _validate_empirical_document_shape(document, authority)
    fingerprint = _receipt_content_fingerprint(document)
    _check(
        document["content_fingerprint"] == fingerprint
        and document["build_id"] == f"stage11-empirical-candidate-{fingerprint[:24]}",
        "Empirical-build fingerprint/build ID differs",
    )

    code = document["code_identity"]
    current_code = dict(materialization_code_hashes())
    current_digest = materialization_code_hashes_sha256(current_code)
    _check(
        code["unchanged_during_build"] is True
        and code["start_hashes"] == code["end_hashes"] == current_code
        and code["start_hashes_sha256"] == code["end_hashes_sha256"] == current_digest,
        "Current materialization code differs from exact build-time code",
    )
    feature_input = document["feature_input"]
    _check(
        feature_input["adapter_code_path"] == _FEATURE_INPUT_ADAPTER_CODE
        and feature_input["adapter_code_sha256"]
        == current_code[_FEATURE_INPUT_ADAPTER_CODE],
        "Feature-input adapter code binding differs",
    )
    input_receipt_path = _absolute_path(Path(str(feature_input["receipt_path"])))
    input_document, input_bytes = _json_document(
        input_receipt_path,
        label="feature-input bundle receipt",
    )
    _check(
        hashlib.sha256(input_bytes).hexdigest() == feature_input["receipt_sha256"]
        and input_document.get("input_bundle_sha256")
        == feature_input["input_bundle_sha256"]
        and input_document.get("input_bundle_id") == feature_input["input_bundle_id"],
        "Feature-input receipt binding differs",
    )
    logical_root = _absolute_path(Path(str(feature_input["logical_history_root"])))
    bundle = load_validation_feature_input_bundle(
        logical_root,
        predictor_authority=authority,
    )
    _check(
        bundle.receipt_path == input_receipt_path
        and bundle.input_bundle_sha256 == feature_input["input_bundle_sha256"],
        "Reauthenticated feature-input bundle differs",
    )
    current_transition = dict(validation_fiscal_transition_audit(bundle.scored_inputs))
    _check(
        current_transition == document["fundamental_fiscal_transition"],
        "Reauthenticated fundamental fiscal-transition audit differs",
    )
    scored_identity = compute_feature_store_build_identity(bundle.scored_inputs)
    option_identity = _derived_identity(
        scored_identity,
        bundle.optionmetrics_working_inputs,
    )
    identities = document["feature_identities"]
    _check(
        identities["scored_build_id"] == scored_identity.build_id
        and identities["scored_content_fingerprint"]
        == scored_identity.content_fingerprint
        and identities["optionmetrics_working_build_id"] == option_identity.build_id
        and identities["optionmetrics_working_content_fingerprint"]
        == option_identity.content_fingerprint,
        "Empirical-build feature identities differ",
    )

    router_receipt = document["crsp_router"]
    spool_path = _absolute_path(Path(str(router_receipt["spool_path"])))
    router = spool_crsp_signal_router(
        (),
        spool_path=spool_path,
        source_id=str(router_receipt["source_id"]),
        expected_rows=int(router_receipt["row_count"]),
    )
    _check(
        router.database_path
        == _absolute_path(Path(str(router_receipt["database_path"])))
        and router.source_id == router_receipt["source_id"]
        and router.row_count == router_receipt["row_count"]
        and len(router.signal_dates) == router_receipt["signal_date_count"]
        and router.signal_dates[0].isoformat() == router_receipt["first_signal_date"]
        and router.signal_dates[-1].isoformat() == router_receipt["last_signal_date"]
        and type(router_receipt["reused_existing_at_build"]) is bool,
        "Empirical-build CRSP router identity differs",
    )
    database_sha, database_size = _stable_regular_file_sha256(
        router.database_path,
        label="CRSP router database",
    )
    _check(
        database_sha == router_receipt["database_sha256"]
        and database_size == router_receipt["database_size_bytes"],
        "Empirical-build CRSP router database differs",
    )

    candidate_root = _absolute_path(Path(str(document["candidate_root"])))
    _check(
        path != candidate_root and candidate_root not in path.parents,
        "Empirical receipt entered candidate root",
    )
    outputs = _candidate_output_receipts(candidate_root, authority)
    output_values = {role: dict(value) for role, value in sorted(outputs.items())}
    _check(
        output_values == document["outputs"]
        and hashlib.sha256(canonical_json_bytes(output_values)).hexdigest()
        == document["output_receipts_sha256"],
        "Empirical candidate outputs differ from build-time receipts",
    )
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    return ValidatedEmpiricalCandidateBuild(
        receipt=EmpiricalBuildReceipt(receipt_bytes),
        receipt_path=path,
        receipt_sha256=receipt_sha,
        candidate_root=candidate_root,
        _seal=_EMPIRICAL_BUILD_RECEIPT_SEAL,
    )


def write_empirical_validation_candidate(
    *,
    bundle: ValidationFeatureInputBundle,
    destination: Path,
    crsp_spool_path: Path,
    empirical_build_receipt_path: Path,
    contract: ValidationDataContract,
    batch_size: int = 2_048,
    crsp_checkpoint_rows: int = 4_096,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> EmpiricalCandidateReceipt:
    """Run the full path-backed seven-family stream and write 12 candidates."""

    _require_predictor_authority(contract)
    _check(
        batch_size > 0 and crsp_checkpoint_rows > 0,
        "Empirical batch controls must be positive",
    )
    receipt_target = _empirical_receipt_target(
        empirical_build_receipt_path,
        destination,
    )
    start_code_hashes = dict(materialization_code_hashes())
    _check(
        _FEATURE_INPUT_ADAPTER_CODE in start_code_hashes,
        "Feature-input adapter is absent from the build-time code identity",
    )
    input_receipt, input_receipt_bytes = _json_document(
        bundle.receipt_path,
        label="feature-input bundle receipt",
    )
    input_receipt_sha = hashlib.sha256(input_receipt_bytes).hexdigest()
    _check(
        input_receipt.get("input_bundle_sha256") == bundle.input_bundle_sha256
        and input_receipt.get("input_bundle_id")
        == f"stage11-feature-inputs-{bundle.input_bundle_sha256[:24]}"
        and _absolute_path(bundle.receipt_path)
        == _absolute_path(bundle.logical_history_root)
        / "validation_feature_input_bundle_receipt.json",
        "Feature-input bundle/receipt identity differs before empirical build",
    )
    scored_inputs = bundle.scored_inputs
    option_inputs = bundle.optionmetrics_working_inputs
    scored_rows = scored_inputs.tables["crsp_signal_date_universe.parquet"].rows
    option_rows = option_inputs.tables["crsp_signal_date_universe.parquet"].rows
    with _validation_formula_scope(
        scored_rows=scored_rows,
        optionmetrics_working_rows=option_rows,
    ):
        _require_predictor_authority(contract)
        if progress is not None:
            progress("stage11_feature_catalog_authentication_started", {"tables": 26})
        scored_identity = compute_feature_store_build_identity(scored_inputs)
        option_identity = _derived_identity(scored_identity, option_inputs)
        fundamental_transition = validation_fiscal_transition_audit(scored_inputs)
        if progress is not None:
            progress(
                "stage11_feature_catalog_authentication_complete",
                {"build_id": scored_identity.build_id},
            )
            progress(
                "stage11_fundamental_fiscal_transition_audit_complete",
                dict(fundamental_transition),
            )

        def crsp_source(checkpoint: tuple[int, dt.date] | None):
            return iter_unfinalized_crsp_rows(
                scored_inputs,
                start_after_key=checkpoint,
                progress=(
                    None
                    if progress is None
                    else lambda value: progress(
                        "stage11_crsp_feature_scan_progress", dict(value)
                    )
                ),
            )

        router = spool_crsp_signal_router(
            crsp_source,
            spool_path=crsp_spool_path,
            source_id=f"{EMPIRICAL_STREAM_ID}:{scored_identity.content_fingerprint}",
            expected_rows=scored_rows,
            checkpoint_rows=crsp_checkpoint_rows,
        )
        fundamentals = _flatten(
            iter_fundamental_feature_batches(
                scored_inputs,
                batch_size=batch_size,
                build_identity=scored_identity,
            )
        )
        ownership = _flatten(
            iter_ownership_event_feature_batches(
                scored_inputs,
                batch_size=batch_size,
                build_identity=scored_identity,
            )
        )
        option_all = _flatten(
            iter_optionmetrics_feature_batches(
                option_inputs,
                batch_size=batch_size,
                build_identity=option_identity,
            )
        )
        context_rows, option_scored = _split_optionmetrics_context(option_all)
        aggregate = _flatten(
            materialize_aggregate_state(
                scored_inputs,
                batch_size=batch_size,
            ).iter_materialization_row_batches()
        )
        scored_batches = _aligned_scored_batches(
            router=router,
            fundamental_rows=fundamentals,
            ownership_rows=ownership,
            optionmetrics_rows=option_scored,
            aggregate_rows=aggregate,
            batch_size=batch_size,
        )
        feature_receipt, risk = write_validation_candidate_streaming(
            scored_batches,
            context_rows,
            validation_risk_history_inputs(
                bundle,
                predictor_authority=contract,
            ),
            destination,
            contract,
            maximum_feature_rows_per_chunk=batch_size,
        )
    outputs = _candidate_output_receipts(feature_receipt.destination, contract)
    router_database_sha, router_database_size = _stable_regular_file_sha256(
        router.database_path,
        label="completed CRSP router database",
    )
    end_code_hashes = dict(materialization_code_hashes())
    document = _empirical_build_document(
        candidate_root=feature_receipt.destination,
        bundle=bundle,
        feature_input_receipt_sha256=input_receipt_sha,
        scored_identity=scored_identity,
        option_identity=option_identity,
        router=router,
        router_database_sha256=router_database_sha,
        router_database_size=router_database_size,
        outputs=outputs,
        fundamental_fiscal_transition=fundamental_transition,
        start_code_hashes=start_code_hashes,
        end_code_hashes=end_code_hashes,
    )
    _validate_empirical_document_shape(document, contract)
    written_path, receipt_bytes, receipt_sha = _write_empirical_build_receipt(
        document,
        receipt_target,
    )
    empirical_build = ValidatedEmpiricalCandidateBuild(
        receipt=EmpiricalBuildReceipt(receipt_bytes),
        receipt_path=written_path,
        receipt_sha256=receipt_sha,
        candidate_root=feature_receipt.destination,
        _seal=_EMPIRICAL_BUILD_RECEIPT_SEAL,
    )
    return EmpiricalCandidateReceipt(
        feature_receipt=feature_receipt,
        risk_cost=risk,
        scored_build_id=scored_identity.build_id,
        optionmetrics_working_build_id=option_identity.build_id,
        crsp_router_reused=router.reused_existing,
        input_bundle_sha256=bundle.input_bundle_sha256,
        empirical_build=empirical_build,
    )


__all__ = [
    "EMPIRICAL_STREAM_ID",
    "EmpiricalCandidateReceipt",
    "EmpiricalBuildReceipt",
    "ValidatedEmpiricalCandidateBuild",
    "ValidationEmpiricalFeatureStreamError",
    "load_and_reauthenticate_empirical_candidate_build",
    "validation_risk_history_inputs",
    "write_empirical_validation_candidate",
]
