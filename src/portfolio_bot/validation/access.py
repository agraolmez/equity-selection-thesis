"""Fail-closed data-access broker for the Stage 11 preflight boundary.

The narrow preflight receipt permits validation predictors and calendar
metadata only through exact column/date requests.  It never permits validation
outcomes, forward returns, real portfolio results, or the final period.  Role
authorization is decided *before* a path is normalized, statted, opened, or
handed to a tabular reader.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import pyarrow as pa
import pyarrow.dataset as ds


class Stage11AccessError(RuntimeError):
    """Raised before any unauthorized filesystem interaction can occur."""


class AccessMode(str, Enum):
    PREFLIGHT_NO_OUTCOMES = "preflight_no_outcomes"
    VALIDATION_EXECUTION = "validation_execution"


class PayloadRole(str, Enum):
    REPOSITORY_METADATA = "repository_metadata"
    AUTHENTICATED_STAGE10_METADATA = "authenticated_stage10_metadata"
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    DEVELOPMENT_PARITY_FIXTURE = "development_parity_fixture"
    VALIDATION_CALENDAR_METADATA = "validation_calendar_metadata"
    VALIDATION_TRADING_SESSION_METADATA = "validation_trading_session_metadata"
    VALIDATION_PREDICTOR = "validation_predictor_payload"
    VALIDATION_LABEL = "validation_label_payload"
    VALIDATION_OUTCOME = "validation_outcome_payload"
    VALIDATION_FORWARD_RETURN = "validation_forward_return"
    VALIDATION_RISK_INPUT = "validation_risk_input"
    VALIDATION_COST_INPUT = "validation_cost_input"
    VALIDATION_MARKET_INPUT = "validation_market_input"
    VALIDATION_ACCOUNTING_INPUT = "validation_accounting_input"
    REAL_VALIDATION_FORECAST = "real_validation_forecast"
    REAL_VALIDATION_SELECTION = "real_validation_model_or_ensemble_selection"
    REAL_VALIDATION_WEIGHT = "real_validation_weight"
    REAL_VALIDATION_TRADE = "real_validation_trade"
    REAL_VALIDATION_PNL = "real_validation_pnl_or_performance"
    FINAL_PERIOD = "final_period_predictor_or_outcome"
    STAGE12_OR_LATER = "stage12_or_later_payload"


PREFLIGHT_PERMITTED_ROLES = frozenset(
    {
        PayloadRole.REPOSITORY_METADATA,
        PayloadRole.AUTHENTICATED_STAGE10_METADATA,
        PayloadRole.SYNTHETIC_FIXTURE,
        PayloadRole.DEVELOPMENT_PARITY_FIXTURE,
        PayloadRole.VALIDATION_CALENDAR_METADATA,
        PayloadRole.VALIDATION_TRADING_SESSION_METADATA,
        PayloadRole.VALIDATION_PREDICTOR,
    }
)

PREFLIGHT_PROHIBITED_ROLES = frozenset(PayloadRole) - PREFLIGHT_PERMITTED_ROLES

# The registration object is supplied by orchestration code, so its boolean
# content assertion and column allowlist are not themselves semantic
# authority.  These names are therefore rejected independently at
# registration, authorization, and post-scan audit boundaries.  Historical
# predictors such as ``return_lag_1m`` remain valid; only an explicitly
# forward/target/outcome-labelled column is prohibited.
FROZEN_STAGE6_RAW_SCHEMA_SHA256 = (
    "e67436adbe6e2adb2f445a799f07002a272a5cdf780662aca639f0eb3c0969c3"
)
_FORWARD_RETURN_TOKENS = frozenset({"forward", "future", "fwd", "lead", "next"})


def _normalized_column_tokens(value: str) -> tuple[str, ...]:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return tuple(token for token in re.split(r"[^a-z0-9]+", snake.lower()) if token)


def _is_prohibited_validation_column(
    value: object,
    *,
    trusted_schema_sha256: str | None,
) -> bool:
    if not isinstance(value, str) or not value:
        return True
    tokens = set(_normalized_column_tokens(value))
    ordered = _normalized_column_tokens(value)
    trusted_stage6_schema = (
        trusted_schema_sha256 == FROZEN_STAGE6_RAW_SCHEMA_SHA256
    )
    if "outcome" in tokens or "target" in tokens:
        return True
    if "return" in tokens and (
        bool(tokens & _FORWARD_RETURN_TOKENS)
        or "residual" in tokens
        or "realized" in tokens
    ):
        return True
    if "label" in tokens:
        # Stage 6 contains legitimate industry beta label predictors.  They
        # are admitted only as members of its frozen physical schema, never
        # through a generic caller-declared allowlist.
        return not (
            trusted_stage6_schema
            and len(ordered) >= 2
            and ordered[0:2] == ("beta", "label")
        )
    return False


def _require_predictor_safe_columns(
    columns: tuple[str, ...],
    *,
    trusted_schema_sha256: str | None,
) -> None:
    prohibited = sorted(
        value
        for value in columns
        if _is_prohibited_validation_column(
            value, trusted_schema_sha256=trusted_schema_sha256
        )
    )
    if prohibited:
        raise Stage11AccessError(
            "validation predictor columns contain a prohibited "
            f"forward/outcome/target/label semantic: {prohibited}"
        )


@dataclass(frozen=True, slots=True)
class RegisteredPayload:
    """Exact identity of one file that the broker may read."""

    role: PayloadRole
    path: Path
    sha256: str
    size_bytes: int
    allowed_columns: tuple[str, ...] = ()
    minimum_signal_date: date | None = None
    maximum_signal_date: date | None = None
    contains_outcome_or_forward_return: bool = False
    trusted_schema_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.role not in PREFLIGHT_PERMITTED_ROLES - {PayloadRole.SYNTHETIC_FIXTURE}:
            raise Stage11AccessError("only permitted file-backed roles may be registered")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise Stage11AccessError("registered payload path must be absolute")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.sha256)
        ):
            raise Stage11AccessError("registered payload SHA-256 is malformed")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise Stage11AccessError("registered payload size must be nonnegative")
        if type(self.contains_outcome_or_forward_return) is not bool:
            raise Stage11AccessError("outcome-content flag must be boolean")
        if self.contains_outcome_or_forward_return:
            raise Stage11AccessError("outcome or forward-return content is prohibited")
        if self.trusted_schema_sha256 is not None:
            if (
                not isinstance(self.trusted_schema_sha256, str)
                or self.role is not PayloadRole.VALIDATION_PREDICTOR
                or self.trusted_schema_sha256 != FROZEN_STAGE6_RAW_SCHEMA_SHA256
            ):
                raise Stage11AccessError("unrecognized trusted predictor schema digest")
        if self.role in {
            PayloadRole.VALIDATION_CALENDAR_METADATA,
            PayloadRole.VALIDATION_TRADING_SESSION_METADATA,
            PayloadRole.VALIDATION_PREDICTOR,
        }:
            if (
                type(self.allowed_columns) is not tuple
                or not self.allowed_columns
                or len(self.allowed_columns) != len(set(self.allowed_columns))
                or any(not isinstance(value, str) or not value for value in self.allowed_columns)
            ):
                raise Stage11AccessError(
                    "validation metadata/predictor registration requires an exact column allowlist"
                )
            if type(self.minimum_signal_date) is not date or type(
                self.maximum_signal_date
            ) is not date:
                raise Stage11AccessError(
                    "validation metadata/predictor registration requires exact date bounds"
                )
            if self.minimum_signal_date > self.maximum_signal_date:
                raise Stage11AccessError("registered validation date bounds are reversed")
            maximum_permitted_date = (
                date(2019, 12, 2)
                if self.role is PayloadRole.VALIDATION_TRADING_SESSION_METADATA
                else date(2019, 10, 31)
            )
            if self.maximum_signal_date > maximum_permitted_date:
                raise Stage11AccessError(
                    "registered validation table crosses its permitted cutoff"
                )
            _require_predictor_safe_columns(
                self.allowed_columns,
                trusted_schema_sha256=self.trusted_schema_sha256,
            )


@dataclass(frozen=True, slots=True)
class AccessLedger:
    mode: str
    permitted_file_reads: int
    permitted_bytes_read: int
    development_parity_rows_read: int
    validation_predictor_rows_read: int
    validation_predictor_maximum_signal_date: str | None
    validation_predictor_forward_return_or_label_columns_read: int
    validation_outcome_rows_read: int
    final_period_rows_read: int
    real_validation_weights_created: int
    real_validation_trades_created: int
    real_validation_pnl_or_performance_rows_created: int

    @property
    def forbidden_counters_are_zero(self) -> bool:
        return all(
            value == 0
            for value in (
                self.validation_predictor_forward_return_or_label_columns_read,
                self.validation_outcome_rows_read,
                self.final_period_rows_read,
                self.real_validation_weights_created,
                self.real_validation_trades_created,
                self.real_validation_pnl_or_performance_rows_created,
            )
        )


@dataclass(frozen=True, slots=True)
class AuthorizedValidationTableRequest:
    """Exact bounded request that a pushdown-capable reader may execute."""

    registration_name: str
    record: RegisteredPayload
    columns: tuple[str, ...]
    minimum_signal_date: date
    maximum_signal_date: date

    def __post_init__(self) -> None:
        if not isinstance(self.registration_name, str) or not self.registration_name:
            raise Stage11AccessError("validation request registration name is invalid")
        if self.record.role not in {
            PayloadRole.VALIDATION_CALENDAR_METADATA,
            PayloadRole.VALIDATION_TRADING_SESSION_METADATA,
            PayloadRole.VALIDATION_PREDICTOR,
        }:
            raise Stage11AccessError("validation request record has a non-table role")
        if (
            type(self.columns) is not tuple
            or not self.columns
            or len(self.columns) != len(set(self.columns))
            or not set(self.columns).issubset(self.record.allowed_columns)
        ):
            raise Stage11AccessError("validation request columns escaped registration")
        _require_predictor_safe_columns(
            self.columns,
            trusted_schema_sha256=self.record.trusted_schema_sha256,
        )
        if (
            type(self.minimum_signal_date) is not date
            or type(self.maximum_signal_date) is not date
            or self.minimum_signal_date > self.maximum_signal_date
            or self.record.minimum_signal_date is None
            or self.record.maximum_signal_date is None
            or self.minimum_signal_date < self.record.minimum_signal_date
            or self.maximum_signal_date > self.record.maximum_signal_date
        ):
            raise Stage11AccessError("validation request date scope escaped registration")


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


class PreflightAccessBroker:
    """One-purpose broker that can never be upgraded to outcome access."""

    def __init__(
        self,
        registered: Mapping[str, RegisteredPayload] | None = None,
        *,
        byte_reader: Callable[[Path], bytes] | None = None,
    ) -> None:
        records = dict(registered or {})
        if any(not isinstance(key, str) or not key for key in records):
            raise Stage11AccessError("registered payload names must be nonempty strings")
        if len(records) != len(set(records)):
            raise Stage11AccessError("registered payload names must be unique")
        paths = [record.path for record in records.values()]
        if len(paths) != len(set(paths)):
            raise Stage11AccessError("one file cannot be registered under multiple names")
        self._registered = MappingProxyType(records)
        self._byte_reader = byte_reader or self._read_regular_file
        self._permitted_file_reads = 0
        self._permitted_bytes_read = 0
        self._development_parity_rows_read = 0
        self._validation_predictor_rows_read = 0
        self._validation_predictor_maximum_signal_date: date | None = None
        self._validation_predictor_forward_return_or_label_columns_read = 0
        self._recorded_validation_requests: set[AuthorizedValidationTableRequest] = set()

    @property
    def mode(self) -> AccessMode:
        return AccessMode.PREFLIGHT_NO_OUTCOMES

    @property
    def registered(self) -> Mapping[str, RegisteredPayload]:
        return self._registered

    @staticmethod
    def require_role(role: PayloadRole | str) -> PayloadRole:
        """Authorize by role before accepting or examining any path."""

        try:
            normalized = role if isinstance(role, PayloadRole) else PayloadRole(role)
        except (TypeError, ValueError) as exc:
            raise Stage11AccessError("unregistered Stage 11 payload role") from exc
        if normalized not in PREFLIGHT_PERMITTED_ROLES:
            raise Stage11AccessError(
                f"{normalized.value} is prohibited during no-outcome preflight"
            )
        return normalized

    @staticmethod
    def synthetic_fixture(value: object, *, role: PayloadRole | str) -> object:
        normalized = PreflightAccessBroker.require_role(role)
        if normalized is not PayloadRole.SYNTHETIC_FIXTURE:
            raise Stage11AccessError("in-memory fixture must use synthetic_fixture role")
        return value

    def read_registered_bytes(
        self,
        name: str,
        *,
        role: PayloadRole | str,
    ) -> bytes:
        normalized = self.require_role(role)
        if normalized is PayloadRole.SYNTHETIC_FIXTURE:
            raise Stage11AccessError("synthetic fixtures are not file-backed")
        if normalized in {
            PayloadRole.VALIDATION_CALENDAR_METADATA,
            PayloadRole.VALIDATION_TRADING_SESSION_METADATA,
            PayloadRole.VALIDATION_PREDICTOR,
        }:
            raise Stage11AccessError(
                "validation tables require the guarded column/date reader"
            )
        # Lookup happens only after role authorization.  A prohibited request
        # therefore cannot reveal whether a path/name exists.
        try:
            record = self._registered[name]
        except KeyError as exc:
            raise Stage11AccessError("unregistered Stage 11 payload name") from exc
        if record.role is not normalized:
            raise Stage11AccessError("registered payload role mismatch")
        payload = self._byte_reader(record.path)
        if len(payload) != record.size_bytes or _sha256_bytes(payload) != record.sha256:
            raise Stage11AccessError("registered Stage 11 payload identity changed")
        self._permitted_file_reads += 1
        self._permitted_bytes_read += len(payload)
        return payload

    def authorize_validation_table_request(
        self,
        name: str,
        *,
        role: PayloadRole | str,
        columns: tuple[str, ...],
        minimum_signal_date: date,
        maximum_signal_date: date,
    ) -> AuthorizedValidationTableRequest:
        """Authorize a bounded reader request without opening the file.

        The eventual Arrow/DuckDB adapter must receive this returned identity
        and apply the filter inside its scan.  Reading a combined table and
        filtering afterward is outside the contract.
        """

        normalized = self.require_role(role)
        if normalized not in {
            PayloadRole.VALIDATION_CALENDAR_METADATA,
            PayloadRole.VALIDATION_TRADING_SESSION_METADATA,
            PayloadRole.VALIDATION_PREDICTOR,
        }:
            raise Stage11AccessError("role is not a validation table role")
        try:
            record = self._registered[name]
        except KeyError as exc:
            raise Stage11AccessError("unregistered Stage 11 payload name") from exc
        if record.role is not normalized:
            raise Stage11AccessError("registered payload role mismatch")
        if (
            type(columns) is not tuple
            or not columns
            or len(columns) != len(set(columns))
            or not set(columns).issubset(record.allowed_columns)
        ):
            raise Stage11AccessError("validation table column request escaped its allowlist")
        _require_predictor_safe_columns(
            columns,
            trusted_schema_sha256=record.trusted_schema_sha256,
        )
        if type(minimum_signal_date) is not date or type(maximum_signal_date) is not date:
            raise Stage11AccessError("validation table request requires date bounds")
        if minimum_signal_date > maximum_signal_date:
            raise Stage11AccessError("validation table request date bounds are reversed")
        if (
            minimum_signal_date < record.minimum_signal_date  # type: ignore[operator]
            or maximum_signal_date > record.maximum_signal_date  # type: ignore[operator]
        ):
            raise Stage11AccessError("validation table date request escaped its allowlist")
        return AuthorizedValidationTableRequest(
            registration_name=name,
            record=record,
            columns=columns,
            minimum_signal_date=minimum_signal_date,
            maximum_signal_date=maximum_signal_date,
        )

    def record_guarded_validation_table_rows(
        self,
        request: AuthorizedValidationTableRequest,
        *,
        rows: int,
        observed_maximum_signal_date: date | None,
        exposed_columns: tuple[str, ...],
    ) -> None:
        """Commit the audit receipt after one bounded reader request finishes.

        This method does not read the table.  It ensures the reader reports the
        exact authorized projection, a bounded observed date, and a nonnegative
        row count.  A request cannot be recorded twice.
        """

        if not isinstance(request, AuthorizedValidationTableRequest):
            raise Stage11AccessError("validation row audit requires an authorized request")
        if request.registration_name not in self._registered or self._registered[
            request.registration_name
        ] != request.record:
            raise Stage11AccessError("validation row audit request is not registered")
        if request in self._recorded_validation_requests:
            raise Stage11AccessError("validation row audit request was already recorded")
        if type(rows) is not int or rows < 0:
            raise Stage11AccessError("validation row count must be nonnegative")
        if tuple(exposed_columns) != request.columns:
            raise Stage11AccessError("reader projection differs from its authorized columns")
        _require_predictor_safe_columns(
            tuple(exposed_columns),
            trusted_schema_sha256=request.record.trusted_schema_sha256,
        )
        if rows == 0:
            if observed_maximum_signal_date is not None:
                raise Stage11AccessError("empty validation read must report no maximum date")
        else:
            if type(observed_maximum_signal_date) is not date:
                raise Stage11AccessError("nonempty validation read requires a maximum date")
            if not (
                request.minimum_signal_date
                <= observed_maximum_signal_date
                <= request.maximum_signal_date
            ):
                raise Stage11AccessError("observed validation date escaped its request")
        self._recorded_validation_requests.add(request)
        if request.record.role is PayloadRole.VALIDATION_PREDICTOR:
            self._validation_predictor_rows_read += rows
            if observed_maximum_signal_date is not None and (
                self._validation_predictor_maximum_signal_date is None
                or observed_maximum_signal_date
                > self._validation_predictor_maximum_signal_date
            ):
                self._validation_predictor_maximum_signal_date = (
                    observed_maximum_signal_date
                )

    def read_guarded_parquet(
        self,
        request: AuthorizedValidationTableRequest,
        *,
        signal_date_column: str = "signal_date",
    ) -> pa.Table:
        """Execute one registered Parquet projection with date pushdown.

        The file identity is streamed and authenticated before Arrow receives
        the path.  Arrow applies the date predicate and projection inside its
        scanner; no unfiltered table is materialized and filtered afterward.
        A stable filesystem identity is required through completion.
        """

        if not isinstance(request, AuthorizedValidationTableRequest):
            raise Stage11AccessError("guarded Parquet read requires an authorized request")
        if request.registration_name not in self._registered or self._registered[
            request.registration_name
        ] != request.record:
            raise Stage11AccessError("guarded Parquet request is not registered")
        if request in self._recorded_validation_requests:
            raise Stage11AccessError("validation row audit request was already recorded")
        if signal_date_column not in request.columns:
            raise Stage11AccessError("guarded Parquet projection must expose signal_date")
        lexical = _lexical_absolute(request.record.path)
        try:
            before = lexical.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise Stage11AccessError("registered validation table is missing") from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise Stage11AccessError("registered validation table is not a regular file")
        if before.st_nlink != 1:
            raise Stage11AccessError("registered validation table has unsafe hard links")
        if before.st_size != request.record.size_bytes:
            raise Stage11AccessError("registered validation table size changed")
        digest = hashlib.sha256()
        with lexical.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        after_hash = lexical.stat(follow_symlinks=False)
        if _stable_identity(before) != _stable_identity(after_hash):
            raise Stage11AccessError("registered validation table changed while hashing")
        if digest.hexdigest() != request.record.sha256:
            raise Stage11AccessError("registered validation table hash changed")

        predicate = (
            (ds.field(signal_date_column) >= pa.scalar(request.minimum_signal_date))
            & (ds.field(signal_date_column) <= pa.scalar(request.maximum_signal_date))
        )
        try:
            dataset = ds.dataset(lexical, format="parquet")
            table = dataset.scanner(
                columns=list(request.columns),
                filter=predicate,
                use_threads=False,
            ).to_table()
        except (pa.ArrowException, OSError, ValueError) as exc:
            raise Stage11AccessError("guarded validation Parquet scan failed") from exc
        after_scan = lexical.stat(follow_symlinks=False)
        if _stable_identity(before) != _stable_identity(after_scan):
            raise Stage11AccessError("registered validation table changed during scan")
        if tuple(table.column_names) != request.columns:
            raise Stage11AccessError("guarded validation reader projection changed")
        _require_predictor_safe_columns(
            tuple(table.column_names),
            trusted_schema_sha256=request.record.trusted_schema_sha256,
        )
        dates = table[signal_date_column].combine_chunks().to_pylist()
        if any(type(value) is not date for value in dates):
            raise Stage11AccessError("guarded validation signal dates are null or malformed")
        if any(
            value < request.minimum_signal_date or value > request.maximum_signal_date
            for value in dates
        ):
            raise Stage11AccessError("guarded validation scanner returned an escaped date")
        observed_maximum = max(dates) if dates else None
        self.record_guarded_validation_table_rows(
            request,
            rows=table.num_rows,
            observed_maximum_signal_date=observed_maximum,
            exposed_columns=tuple(table.column_names),
        )
        self._permitted_file_reads += 1
        self._permitted_bytes_read += request.record.size_bytes
        return table

    def record_development_parity_rows(self, rows: int) -> None:
        if type(rows) is not int or rows < 0:
            raise Stage11AccessError("development parity row count must be nonnegative")
        self._development_parity_rows_read += rows

    def record_created_output(self, role: PayloadRole | str, *, rows: int) -> None:
        # Any empirical output is prohibited in this immutable preflight mode.
        try:
            normalized = role if isinstance(role, PayloadRole) else PayloadRole(role)
        except (TypeError, ValueError) as exc:
            raise Stage11AccessError("unregistered Stage 11 output role") from exc
        if normalized in {
            PayloadRole.REAL_VALIDATION_WEIGHT,
            PayloadRole.REAL_VALIDATION_TRADE,
            PayloadRole.REAL_VALIDATION_PNL,
        }:
            raise Stage11AccessError(
                f"{normalized.value} cannot be created during no-outcome preflight"
            )
        raise Stage11AccessError("role is not a creatable Stage 11 output")

    def ledger(self) -> AccessLedger:
        return AccessLedger(
            mode=self.mode.value,
            permitted_file_reads=self._permitted_file_reads,
            permitted_bytes_read=self._permitted_bytes_read,
            development_parity_rows_read=self._development_parity_rows_read,
            validation_predictor_rows_read=self._validation_predictor_rows_read,
            validation_predictor_maximum_signal_date=(
                None
                if self._validation_predictor_maximum_signal_date is None
                else self._validation_predictor_maximum_signal_date.isoformat()
            ),
            validation_predictor_forward_return_or_label_columns_read=(
                self._validation_predictor_forward_return_or_label_columns_read
            ),
            validation_outcome_rows_read=0,
            final_period_rows_read=0,
            real_validation_weights_created=0,
            real_validation_trades_created=0,
            real_validation_pnl_or_performance_rows_created=0,
        )

    @staticmethod
    def _read_regular_file(path: Path) -> bytes:
        lexical = _lexical_absolute(path)
        try:
            observed = lexical.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise Stage11AccessError("registered Stage 11 payload is missing") from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
            raise Stage11AccessError("registered Stage 11 payload is not a regular file")
        if observed.st_nlink != 1:
            raise Stage11AccessError("registered Stage 11 payload has unsafe hard links")
        with lexical.open("rb") as handle:
            payload = handle.read()
        after = lexical.stat(follow_symlinks=False)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(observed, field) != getattr(after, field) for field in stable_fields):
            raise Stage11AccessError("registered Stage 11 payload changed during read")
        return payload


__all__ = [
    "AccessLedger",
    "AccessMode",
    "AuthorizedValidationTableRequest",
    "FROZEN_STAGE6_RAW_SCHEMA_SHA256",
    "PREFLIGHT_PERMITTED_ROLES",
    "PREFLIGHT_PROHIBITED_ROLES",
    "PayloadRole",
    "PreflightAccessBroker",
    "RegisteredPayload",
    "Stage11AccessError",
]
