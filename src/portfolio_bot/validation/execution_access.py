"""Authority-gated data and output broker for empirical Stage 11 validation.

This module is deliberately separate from :mod:`portfolio_bot.validation.access`.
The preflight broker remains permanently incapable of reading outcomes or
creating empirical outputs; this broker can be constructed only inside a live,
descriptor-anchored ``Stage11ValidationUnblindAuthority`` scope.

Every table is registered by exact file identity, role, projection, temporal
axis, and date bounds.  The scan applies its date predicate before materializing
rows.  Signal-date payloads stop at 31 October 2019, while daily
market/accounting/session payloads stop at the terminal validation exit on
2 December 2019.  The December-2019 holding month and every 2020+ observation
therefore remain inaccessible even after validation is unblinded.
"""

from __future__ import annotations

import hashlib
import stat
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from portfolio_bot.validation.access import (
    AccessMode,
    FROZEN_STAGE6_RAW_SCHEMA_SHA256,
    PayloadRole,
    Stage11AccessError,
    _lexical_absolute,
    _require_predictor_safe_columns,
    _stable_identity,
)
from portfolio_bot.validation.calendar import VALIDATION_HOLDING_MONTHS


FIRST_VALIDATION_SIGNAL_DATE = date(2014, 12, 31)
LAST_VALIDATION_SIGNAL_DATE = date(2019, 10, 31)
FIRST_VALIDATION_ENTRY_DATE = date(2015, 1, 2)
FIRST_VALIDATION_EXIT_DATE = date(2015, 2, 2)
LAST_VALIDATION_ENTRY_DATE = date(2019, 11, 1)
LAST_VALIDATION_EXIT_DATE = date(2019, 12, 2)
EMBARGOED_HOLDING_MONTH = "2019-12"
FIRST_PROHIBITED_FINAL_DATE = date(2020, 1, 2)


class Stage11ExecutionAccessSealedError(Stage11AccessError):
    """Raised before path lookup when no live hash-bound authority exists."""


class ExecutionDateAxis(str, Enum):
    """Meaning of the bounded date column exposed by one table."""

    SIGNAL = "signal_date"
    OBSERVATION = "observation_or_exit_date"


_SIGNAL_ONLY_INPUT_ROLES = frozenset(
    {
        PayloadRole.VALIDATION_CALENDAR_METADATA,
        PayloadRole.VALIDATION_PREDICTOR,
        PayloadRole.VALIDATION_RISK_INPUT,
        PayloadRole.VALIDATION_COST_INPUT,
    }
)
_OBSERVATION_ONLY_INPUT_ROLES = frozenset(
    {
        PayloadRole.VALIDATION_TRADING_SESSION_METADATA,
        PayloadRole.VALIDATION_MARKET_INPUT,
        PayloadRole.VALIDATION_ACCOUNTING_INPUT,
    }
)
_OUTCOME_INPUT_ROLES = frozenset(
    {
        PayloadRole.VALIDATION_LABEL,
        PayloadRole.VALIDATION_OUTCOME,
        PayloadRole.VALIDATION_FORWARD_RETURN,
    }
)
EXECUTION_INPUT_ROLES = frozenset(
    _SIGNAL_ONLY_INPUT_ROLES
    | _OBSERVATION_ONLY_INPUT_ROLES
    | _OUTCOME_INPUT_ROLES
)

EXECUTION_EMPIRICAL_OUTPUT_ROLES = frozenset(
    {
        PayloadRole.REAL_VALIDATION_FORECAST,
        PayloadRole.REAL_VALIDATION_SELECTION,
        PayloadRole.REAL_VALIDATION_WEIGHT,
        PayloadRole.REAL_VALIDATION_TRADE,
        PayloadRole.REAL_VALIDATION_PNL,
    }
)

_REQUEST_ISSUANCE_TOKEN = object()


def _normalize_role(role: PayloadRole | str) -> PayloadRole:
    try:
        return role if isinstance(role, PayloadRole) else PayloadRole(role)
    except (TypeError, ValueError) as exc:
        raise Stage11AccessError("unregistered Stage 11 execution role") from exc


def _allowed_axes(role: PayloadRole) -> frozenset[ExecutionDateAxis]:
    if role in _SIGNAL_ONLY_INPUT_ROLES:
        return frozenset({ExecutionDateAxis.SIGNAL})
    if role in _OBSERVATION_ONLY_INPUT_ROLES:
        return frozenset({ExecutionDateAxis.OBSERVATION})
    if role in _OUTCOME_INPUT_ROLES:
        # A label store may be keyed by its signal date or by its completed
        # exit date.  The selected axis is immutable in the registration.
        return frozenset(
            {ExecutionDateAxis.SIGNAL, ExecutionDateAxis.OBSERVATION}
        )
    return frozenset()


def _registration_bounds(
    role: PayloadRole,
    axis: ExecutionDateAxis,
) -> tuple[date, date]:
    if axis is ExecutionDateAxis.SIGNAL:
        return FIRST_VALIDATION_SIGNAL_DATE, LAST_VALIDATION_SIGNAL_DATE
    if role is PayloadRole.VALIDATION_FORWARD_RETURN:
        # The frozen Stage 7 router receives the entry-close return row as
        # terminal-event context even though ordinary signal-to-entry return
        # is never credited.  The first credited return remains strictly after
        # entry.  Denying this one boundary row would falsely classify a
        # security delisted on entry as an ordinary missing post-entry path.
        return FIRST_VALIDATION_ENTRY_DATE, LAST_VALIDATION_EXIT_DATE
    if role in _OUTCOME_INPUT_ROLES:
        return FIRST_VALIDATION_EXIT_DATE, LAST_VALIDATION_EXIT_DATE
    return FIRST_VALIDATION_ENTRY_DATE, LAST_VALIDATION_EXIT_DATE


def _output_bounds(role: PayloadRole) -> tuple[date, date]:
    if role is PayloadRole.REAL_VALIDATION_FORECAST:
        return FIRST_VALIDATION_SIGNAL_DATE, LAST_VALIDATION_SIGNAL_DATE
    if role in {PayloadRole.REAL_VALIDATION_WEIGHT, PayloadRole.REAL_VALIDATION_TRADE}:
        return FIRST_VALIDATION_ENTRY_DATE, LAST_VALIDATION_ENTRY_DATE
    if role in {PayloadRole.REAL_VALIDATION_SELECTION, PayloadRole.REAL_VALIDATION_PNL}:
        return FIRST_VALIDATION_ENTRY_DATE, LAST_VALIDATION_EXIT_DATE
    raise Stage11AccessError("role is not a controlled empirical output")


@dataclass(frozen=True, slots=True)
class RegisteredExecutionPayload:
    """Exact identity and projection of one empirical input table."""

    role: PayloadRole
    path: Path
    sha256: str
    size_bytes: int
    allowed_columns: tuple[str, ...]
    date_column: str
    date_axis: ExecutionDateAxis
    minimum_date: date
    maximum_date: date
    holding_month_column: str | None = None
    trusted_schema_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.role not in EXECUTION_INPUT_ROLES:
            raise Stage11AccessError("role is not a Stage 11 empirical input")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise Stage11AccessError("execution payload path must be absolute")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.sha256)
        ):
            raise Stage11AccessError("execution payload SHA-256 is malformed")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise Stage11AccessError("execution payload size must be nonnegative")
        if (
            type(self.allowed_columns) is not tuple
            or not self.allowed_columns
            or len(self.allowed_columns) != len(set(self.allowed_columns))
            or any(not isinstance(value, str) or not value for value in self.allowed_columns)
        ):
            raise Stage11AccessError(
                "execution payload requires an exact nonempty column allowlist"
            )
        if not isinstance(self.date_column, str) or self.date_column not in self.allowed_columns:
            raise Stage11AccessError(
                "execution payload date column must be in its exact allowlist"
            )
        if type(self.date_axis) is not ExecutionDateAxis:
            raise Stage11AccessError("execution payload date axis is invalid")
        if self.date_axis not in _allowed_axes(self.role):
            raise Stage11AccessError("execution payload role/date axis combination is prohibited")
        if type(self.minimum_date) is not date or type(self.maximum_date) is not date:
            raise Stage11AccessError("execution payload requires exact date bounds")
        if self.minimum_date > self.maximum_date:
            raise Stage11AccessError("execution payload date bounds are reversed")
        allowed_minimum, allowed_maximum = _registration_bounds(
            self.role, self.date_axis
        )
        if self.minimum_date < allowed_minimum or self.maximum_date > allowed_maximum:
            raise Stage11AccessError(
                "execution payload crosses the frozen validation date boundary"
            )
        if self.holding_month_column is not None and (
            not isinstance(self.holding_month_column, str)
            or self.holding_month_column not in self.allowed_columns
        ):
            raise Stage11AccessError(
                "execution holding-month column must be in the exact allowlist"
            )
        if self.role is PayloadRole.VALIDATION_PREDICTOR:
            if self.trusted_schema_sha256 not in {
                None,
                FROZEN_STAGE6_RAW_SCHEMA_SHA256,
            }:
                raise Stage11AccessError(
                    "unrecognized trusted predictor schema digest"
                )
            _require_predictor_safe_columns(
                self.allowed_columns,
                trusted_schema_sha256=self.trusted_schema_sha256,
            )
        elif self.trusted_schema_sha256 is not None:
            raise Stage11AccessError(
                "only the validation predictor role may carry a trusted predictor schema"
            )


@dataclass(frozen=True, slots=True, init=False)
class AuthorizedExecutionTableRequest:
    """Privately issued exact request consumed by the guarded scanner."""

    registration_name: str
    record: RegisteredExecutionPayload
    columns: tuple[str, ...]
    minimum_date: date
    maximum_date: date

    def __init__(
        self,
        *,
        registration_name: str,
        record: RegisteredExecutionPayload,
        columns: tuple[str, ...],
        minimum_date: date,
        maximum_date: date,
        _issuance_token: object,
    ) -> None:
        if _issuance_token is not _REQUEST_ISSUANCE_TOKEN:
            raise Stage11AccessError(
                "execution table request must be issued by its live access broker"
            )
        object.__setattr__(self, "registration_name", registration_name)
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "minimum_date", minimum_date)
        object.__setattr__(self, "maximum_date", maximum_date)


@dataclass(frozen=True, slots=True)
class ExecutionAccessLedger:
    """Immutable, authority-bound accounting for empirical Stage 11 access."""

    mode: str
    authority_build_id: str
    authority_verification_id: str
    approval_contract_file_sha256: str
    approval_contract_payload_sha256: str
    execution_input_manifest_sha256: str
    permitted_file_reads: int
    permitted_bytes_read: int
    validation_predictor_rows_read: int
    validation_label_rows_read: int
    validation_outcome_rows_read: int
    validation_forward_return_rows_read: int
    validation_risk_input_rows_read: int
    validation_cost_input_rows_read: int
    validation_market_input_rows_read: int
    validation_accounting_input_rows_read: int
    validation_calendar_rows_read: int
    validation_trading_session_rows_read: int
    maximum_signal_date_read: str | None
    maximum_observation_or_exit_date_read: str | None
    embargoed_holding_month_rows_read: int
    final_period_rows_read: int
    stage12_or_later_rows_read: int
    real_validation_forecasts_created: int
    real_validation_selections_created: int
    real_validation_weights_created: int
    real_validation_trades_created: int
    real_validation_pnl_or_performance_rows_created: int
    maximum_empirical_output_date: str | None

    @property
    def prohibited_counters_are_zero(self) -> bool:
        return all(
            value == 0
            for value in (
                self.embargoed_holding_month_rows_read,
                self.final_period_rows_read,
                self.stage12_or_later_rows_read,
            )
        )

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


class ValidationExecutionAccessBroker:
    """Outcome-enabled broker that exists only inside one live unblind scope."""

    def __init__(
        self,
        authority: object,
        registered: Mapping[str, RegisteredExecutionPayload] | None = None,
    ) -> None:
        # Authentication deliberately precedes copying registrations or
        # examining any supplied path/name.
        self._authority = self._require_authority(authority)
        records = dict(registered or {})
        if any(not isinstance(name, str) or not name for name in records):
            raise Stage11AccessError("execution payload names must be nonempty strings")
        if any(type(record) is not RegisteredExecutionPayload for record in records.values()):
            raise Stage11AccessError("execution registration has an invalid record type")
        paths = [record.path for record in records.values()]
        if len(paths) != len(set(paths)):
            raise Stage11AccessError(
                "one empirical input file cannot be registered under multiple names"
            )
        self._registered = MappingProxyType(records)
        self._recorded_requests: set[AuthorizedExecutionTableRequest] = set()
        self._permitted_file_reads = 0
        self._permitted_bytes_read = 0
        self._input_rows = {role: 0 for role in EXECUTION_INPUT_ROLES}
        self._maximum_signal_date: date | None = None
        self._maximum_observation_date: date | None = None
        self._output_rows = {role: 0 for role in EXECUTION_EMPIRICAL_OUTPUT_ROLES}
        self._maximum_output_date: date | None = None

    @staticmethod
    def _require_authority(authority: object) -> object:
        # Local import avoids a module-import cycle: validation_runner binds
        # publication.py, whose no-outcome publisher imports access.py.
        from portfolio_bot.validation.validation_runner import (
            Stage11ValidationSealedError,
            require_live_stage11_validation_unblind_authority,
        )

        try:
            return require_live_stage11_validation_unblind_authority(authority)
        except Stage11ValidationSealedError as exc:
            raise Stage11ExecutionAccessSealedError(str(exc)) from exc

    def _gate(self) -> None:
        self._require_authority(self._authority)

    @property
    def mode(self) -> AccessMode:
        return AccessMode.VALIDATION_EXECUTION

    @property
    def registered(self) -> Mapping[str, RegisteredExecutionPayload]:
        self._gate()
        return self._registered

    def require_input_role(self, role: PayloadRole | str) -> PayloadRole:
        self._gate()
        normalized = _normalize_role(role)
        if normalized not in EXECUTION_INPUT_ROLES:
            raise Stage11AccessError(
                f"{normalized.value} is prohibited as a Stage 11 empirical input"
            )
        return normalized

    def authorize_table_request(
        self,
        name: str,
        *,
        role: PayloadRole | str,
        columns: tuple[str, ...],
        minimum_date: date,
        maximum_date: date,
    ) -> AuthorizedExecutionTableRequest:
        normalized = self.require_input_role(role)
        # Name lookup occurs only after the authority and role gates.
        try:
            record = self._registered[name]
        except KeyError as exc:
            raise Stage11AccessError("unregistered Stage 11 execution payload name") from exc
        if record.role is not normalized:
            raise Stage11AccessError("execution payload role mismatch")
        if (
            type(columns) is not tuple
            or not columns
            or len(columns) != len(set(columns))
            or not set(columns).issubset(record.allowed_columns)
            or record.date_column not in columns
            or (
                record.holding_month_column is not None
                and record.holding_month_column not in columns
            )
        ):
            raise Stage11AccessError(
                "execution table projection escaped its registered allowlist"
            )
        if record.role is PayloadRole.VALIDATION_PREDICTOR:
            _require_predictor_safe_columns(
                columns,
                trusted_schema_sha256=record.trusted_schema_sha256,
            )
        if type(minimum_date) is not date or type(maximum_date) is not date:
            raise Stage11AccessError("execution table request requires date bounds")
        if (
            minimum_date > maximum_date
            or minimum_date < record.minimum_date
            or maximum_date > record.maximum_date
        ):
            raise Stage11AccessError(
                "execution table date request escaped its registered boundary"
            )
        return AuthorizedExecutionTableRequest(
            registration_name=name,
            record=record,
            columns=columns,
            minimum_date=minimum_date,
            maximum_date=maximum_date,
            _issuance_token=_REQUEST_ISSUANCE_TOKEN,
        )

    def _validate_request(self, request: AuthorizedExecutionTableRequest) -> None:
        self._gate()
        if type(request) is not AuthorizedExecutionTableRequest:
            raise Stage11AccessError("guarded execution read requires an authorized request")
        if (
            request.registration_name not in self._registered
            or self._registered[request.registration_name] != request.record
        ):
            raise Stage11AccessError("execution request is not registered in this broker")
        if request in self._recorded_requests:
            raise Stage11AccessError("execution request was already recorded")

    def record_guarded_table_rows(
        self,
        request: AuthorizedExecutionTableRequest,
        *,
        rows: int,
        observed_minimum_date: date | None,
        observed_maximum_date: date | None,
        exposed_columns: tuple[str, ...],
        observed_holding_months: tuple[str, ...] = (),
    ) -> None:
        self._validate_request(request)
        if type(rows) is not int or rows < 0:
            raise Stage11AccessError("execution input row count must be nonnegative")
        if exposed_columns != request.columns:
            raise Stage11AccessError("execution reader projection differs from authorization")
        if request.record.role is PayloadRole.VALIDATION_PREDICTOR:
            _require_predictor_safe_columns(
                exposed_columns,
                trusted_schema_sha256=request.record.trusted_schema_sha256,
            )
        if rows == 0:
            if observed_minimum_date is not None or observed_maximum_date is not None:
                raise Stage11AccessError("empty execution read must report null date bounds")
            if observed_holding_months:
                raise Stage11AccessError("empty execution read cannot report holding months")
        else:
            if type(observed_minimum_date) is not date or type(observed_maximum_date) is not date:
                raise Stage11AccessError("nonempty execution read requires observed date bounds")
            if not (
                request.minimum_date
                <= observed_minimum_date
                <= observed_maximum_date
                <= request.maximum_date
            ):
                raise Stage11AccessError("observed execution dates escaped authorization")
        if request.record.holding_month_column is None:
            if observed_holding_months:
                raise Stage11AccessError(
                    "holding months were reported for a table without that column"
                )
        else:
            if rows > 0 and not observed_holding_months:
                raise Stage11AccessError(
                    "holding-month table must report its observed months"
                )
            if (
                len(observed_holding_months) != len(set(observed_holding_months))
                or any(month not in VALIDATION_HOLDING_MONTHS for month in observed_holding_months)
            ):
                raise Stage11AccessError(
                    "execution input contains the December-2019 embargo or another out-of-scope month"
                )
        self._recorded_requests.add(request)
        self._input_rows[request.record.role] += rows
        if observed_maximum_date is not None:
            if request.record.date_axis is ExecutionDateAxis.SIGNAL:
                if (
                    self._maximum_signal_date is None
                    or observed_maximum_date > self._maximum_signal_date
                ):
                    self._maximum_signal_date = observed_maximum_date
            elif (
                self._maximum_observation_date is None
                or observed_maximum_date > self._maximum_observation_date
            ):
                self._maximum_observation_date = observed_maximum_date

    def read_guarded_parquet(
        self,
        request: AuthorizedExecutionTableRequest,
    ) -> pa.Table:
        """Hash-authenticate, push down, project, and ledger one empirical table."""

        self._validate_request(request)
        record = request.record
        lexical = _lexical_absolute(record.path)
        try:
            before = lexical.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise Stage11AccessError("registered execution table is missing") from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise Stage11AccessError("registered execution table is not a regular file")
        if before.st_nlink != 1:
            raise Stage11AccessError("registered execution table has unsafe hard links")
        if before.st_size != record.size_bytes:
            raise Stage11AccessError("registered execution table size changed")
        digest = hashlib.sha256()
        with lexical.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        after_hash = lexical.stat(follow_symlinks=False)
        if _stable_identity(before) != _stable_identity(after_hash):
            raise Stage11AccessError("registered execution table changed while hashing")
        if digest.hexdigest() != record.sha256:
            raise Stage11AccessError("registered execution table hash changed")

        # The registered empirical payload itself must be physically scoped to
        # Stage 11.  Date pushdown may narrow a request within that payload,
        # but it may never be used to hide final-period rows in a combined
        # validation/final file.  Row-group statistics provide this proof
        # without materializing an out-of-scope value column.
        try:
            parquet = pq.ParquetFile(lexical)
            date_index = parquet.schema_arrow.names.index(record.date_column)
            holding_index = (
                None
                if record.holding_month_column is None
                else parquet.schema_arrow.names.index(record.holding_month_column)
            )
        except (ValueError, pa.ArrowException, OSError) as exc:
            raise Stage11AccessError(
                "registered execution table footer is invalid"
            ) from exc
        for row_group_index in range(parquet.metadata.num_row_groups):
            row_group = parquet.metadata.row_group(row_group_index)
            # A physically empty, schema-bearing shard has no date value that
            # can cross a boundary.  This is legitimate for sparse inputs
            # such as corporate actions in a short reserve interval.
            if row_group.num_rows == 0:
                continue
            date_statistics = row_group.column(date_index).statistics
            if (
                date_statistics is None
                or not date_statistics.has_min_max
                or date_statistics.null_count not in {0, None}
                or type(date_statistics.min) is not date
                or type(date_statistics.max) is not date
                or date_statistics.min < record.minimum_date
                or date_statistics.max > record.maximum_date
            ):
                raise Stage11AccessError(
                    "execution payload is not physically bounded to its registered dates"
                )
            if holding_index is not None:
                holding_statistics = row_group.column(holding_index).statistics
                if (
                    holding_statistics is None
                    or not holding_statistics.has_min_max
                    or holding_statistics.null_count not in {0, None}
                    or not isinstance(holding_statistics.min, str)
                    or not isinstance(holding_statistics.max, str)
                    or holding_statistics.min not in VALIDATION_HOLDING_MONTHS
                    or holding_statistics.max not in VALIDATION_HOLDING_MONTHS
                ):
                    raise Stage11AccessError(
                        "execution payload physically contains an embargo/final holding month"
                    )
        after_footer = lexical.stat(follow_symlinks=False)
        if _stable_identity(before) != _stable_identity(after_footer):
            raise Stage11AccessError(
                "registered execution table changed during footer validation"
            )

        predicate = (
            (ds.field(record.date_column) >= pa.scalar(request.minimum_date))
            & (ds.field(record.date_column) <= pa.scalar(request.maximum_date))
        )
        try:
            table = ds.dataset(lexical, format="parquet").scanner(
                columns=list(request.columns),
                filter=predicate,
                use_threads=False,
            ).to_table()
        except (pa.ArrowException, OSError, ValueError) as exc:
            raise Stage11AccessError("guarded execution Parquet scan failed") from exc
        after_scan = lexical.stat(follow_symlinks=False)
        if _stable_identity(before) != _stable_identity(after_scan):
            raise Stage11AccessError("registered execution table changed during scan")
        if tuple(table.column_names) != request.columns:
            raise Stage11AccessError("guarded execution projection changed")
        dates = tuple(table[record.date_column].combine_chunks().to_pylist())
        if any(type(value) is not date for value in dates):
            raise Stage11AccessError("guarded execution dates are null or malformed")
        if any(value < request.minimum_date or value > request.maximum_date for value in dates):
            raise Stage11AccessError("guarded execution scan returned an escaped date")

        observed_months: tuple[str, ...] = ()
        if record.holding_month_column is not None:
            raw_months = tuple(
                table[record.holding_month_column].combine_chunks().to_pylist()
            )
            if any(not isinstance(value, str) for value in raw_months):
                raise Stage11AccessError("guarded execution holding months are malformed")
            observed_months = tuple(sorted(set(raw_months)))
            if any(month not in VALIDATION_HOLDING_MONTHS for month in observed_months):
                raise Stage11AccessError(
                    "guarded execution scan exposed the December-2019 embargo or final period"
                )
        self.record_guarded_table_rows(
            request,
            rows=table.num_rows,
            observed_minimum_date=min(dates) if dates else None,
            observed_maximum_date=max(dates) if dates else None,
            exposed_columns=tuple(table.column_names),
            observed_holding_months=observed_months,
        )
        self._permitted_file_reads += 1
        self._permitted_bytes_read += record.size_bytes
        return table

    def record_created_output(
        self,
        role: PayloadRole | str,
        *,
        rows: int,
        minimum_date: date | None,
        maximum_date: date | None,
        holding_months: tuple[str, ...],
    ) -> None:
        """Authorize and ledger an empirical output before publication.

        A caller must report the complete set of holding months represented by
        a nonempty output.  This makes the December-2019 embargo an explicit
        hard failure rather than an informal convention.
        """

        self._gate()
        normalized = _normalize_role(role)
        if normalized not in EXECUTION_EMPIRICAL_OUTPUT_ROLES:
            raise Stage11AccessError(
                f"{normalized.value} is prohibited as an empirical Stage 11 output"
            )
        if type(rows) is not int or rows < 0:
            raise Stage11AccessError("empirical output row count must be nonnegative")
        if type(holding_months) is not tuple:
            raise Stage11AccessError("empirical output holding months must be a tuple")
        if rows == 0:
            if minimum_date is not None or maximum_date is not None or holding_months:
                raise Stage11AccessError("empty empirical output must report empty scope")
            return
        if type(minimum_date) is not date or type(maximum_date) is not date:
            raise Stage11AccessError("nonempty empirical output requires exact date bounds")
        allowed_minimum, allowed_maximum = _output_bounds(normalized)
        if not (
            allowed_minimum <= minimum_date <= maximum_date <= allowed_maximum
        ):
            raise Stage11AccessError(
                "empirical output crosses the frozen Stage 11 validation boundary"
            )
        if (
            not holding_months
            or len(holding_months) != len(set(holding_months))
            or any(month not in VALIDATION_HOLDING_MONTHS for month in holding_months)
        ):
            raise Stage11AccessError(
                "empirical output contains the December-2019 embargo or another out-of-scope month"
            )
        self._output_rows[normalized] += rows
        if self._maximum_output_date is None or maximum_date > self._maximum_output_date:
            self._maximum_output_date = maximum_date

    def ledger(self) -> ExecutionAccessLedger:
        self._gate()
        authority = self._authority
        return ExecutionAccessLedger(
            mode=self.mode.value,
            authority_build_id=authority.build_id,
            authority_verification_id=authority.verification_id,
            approval_contract_file_sha256=authority.approval_contract_file_sha256,
            approval_contract_payload_sha256=authority.approval_contract_payload_sha256,
            execution_input_manifest_sha256=authority.execution_input_manifest_sha256,
            permitted_file_reads=self._permitted_file_reads,
            permitted_bytes_read=self._permitted_bytes_read,
            validation_predictor_rows_read=self._input_rows[PayloadRole.VALIDATION_PREDICTOR],
            validation_label_rows_read=self._input_rows[PayloadRole.VALIDATION_LABEL],
            validation_outcome_rows_read=self._input_rows[PayloadRole.VALIDATION_OUTCOME],
            validation_forward_return_rows_read=self._input_rows[
                PayloadRole.VALIDATION_FORWARD_RETURN
            ],
            validation_risk_input_rows_read=self._input_rows[
                PayloadRole.VALIDATION_RISK_INPUT
            ],
            validation_cost_input_rows_read=self._input_rows[
                PayloadRole.VALIDATION_COST_INPUT
            ],
            validation_market_input_rows_read=self._input_rows[
                PayloadRole.VALIDATION_MARKET_INPUT
            ],
            validation_accounting_input_rows_read=self._input_rows[
                PayloadRole.VALIDATION_ACCOUNTING_INPUT
            ],
            validation_calendar_rows_read=self._input_rows[
                PayloadRole.VALIDATION_CALENDAR_METADATA
            ],
            validation_trading_session_rows_read=self._input_rows[
                PayloadRole.VALIDATION_TRADING_SESSION_METADATA
            ],
            maximum_signal_date_read=(
                None
                if self._maximum_signal_date is None
                else self._maximum_signal_date.isoformat()
            ),
            maximum_observation_or_exit_date_read=(
                None
                if self._maximum_observation_date is None
                else self._maximum_observation_date.isoformat()
            ),
            embargoed_holding_month_rows_read=0,
            final_period_rows_read=0,
            stage12_or_later_rows_read=0,
            real_validation_forecasts_created=self._output_rows[
                PayloadRole.REAL_VALIDATION_FORECAST
            ],
            real_validation_selections_created=self._output_rows[
                PayloadRole.REAL_VALIDATION_SELECTION
            ],
            real_validation_weights_created=self._output_rows[
                PayloadRole.REAL_VALIDATION_WEIGHT
            ],
            real_validation_trades_created=self._output_rows[
                PayloadRole.REAL_VALIDATION_TRADE
            ],
            real_validation_pnl_or_performance_rows_created=self._output_rows[
                PayloadRole.REAL_VALIDATION_PNL
            ],
            maximum_empirical_output_date=(
                None
                if self._maximum_output_date is None
                else self._maximum_output_date.isoformat()
            ),
        )


__all__ = [
    "AuthorizedExecutionTableRequest",
    "EMBARGOED_HOLDING_MONTH",
    "EXECUTION_EMPIRICAL_OUTPUT_ROLES",
    "EXECUTION_INPUT_ROLES",
    "ExecutionAccessLedger",
    "ExecutionDateAxis",
    "FIRST_PROHIBITED_FINAL_DATE",
    "FIRST_VALIDATION_ENTRY_DATE",
    "FIRST_VALIDATION_EXIT_DATE",
    "FIRST_VALIDATION_SIGNAL_DATE",
    "LAST_VALIDATION_ENTRY_DATE",
    "LAST_VALIDATION_EXIT_DATE",
    "LAST_VALIDATION_SIGNAL_DATE",
    "RegisteredExecutionPayload",
    "Stage11ExecutionAccessSealedError",
    "ValidationExecutionAccessBroker",
]
