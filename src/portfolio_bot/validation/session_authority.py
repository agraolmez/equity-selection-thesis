"""Authenticated holding-period session boundaries for Stage 11.

The daily engine must not infer its holding interval from the rows that happen
to be supplied.  The sole public issuer below starts from the frozen Phase 2
rebalance-calendar authority and performs a hash-guarded, date-pushdown scan
of the frozen Phase 2 trading calendar.  It then seals the exact ``(entry,
exit]`` sessions into an immutable, content-addressed authority.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.validation.access import (
    AuthorizedValidationTableRequest,
    PayloadRole,
    PreflightAccessBroker,
)
from portfolio_bot.validation.calendar import ValidationCalendarRow
from portfolio_bot.validation.walk_forward import (
    AuthenticatedValidationCalendar,
    EXPECTED_REBALANCE_CALENDAR_SHA256,
)


EXPECTED_TRADING_CALENDAR_PATH = Path(
    "/Users/agraemirolmez/Desktop/Thesis/interim/phase_02_canonical_panel/"
    "phase02-crsp-b03263a970e52fbd/calendar/calendar-ae2e4ba4a280d66e/"
    "trading_calendar.parquet"
)
EXPECTED_TRADING_CALENDAR_SHA256 = (
    "b53c9fb689cea035019180405f7c2b3f90fae76f1815dadc8d2b30a1bfc7ea03"
)
EXPECTED_TRADING_CALENDAR_SIZE_BYTES = 28_382
EXPECTED_REBALANCE_CALENDAR_PATH = EXPECTED_TRADING_CALENDAR_PATH.with_name(
    "rebalance_calendar.parquet"
)
EXPECTED_REBALANCE_CALENDAR_SIZE_BYTES = 6_648
_SESSION_COLUMNS = ("calendar_date",)
_ISSUANCE_TOKEN = object()
_DEVELOPMENT_CALENDAR_ISSUANCE_TOKEN = object()
FINAL_RETURN_AUTHORIZATION_OVERLAY_ID = (
    "stage13-session-authoritative-return-overlay-v1"
)
DEVELOPMENT_ALPHA_HOLDING_MONTHS = tuple(
    f"{year:04d}-{month:02d}"
    for year in range(2007, 2015)
    for month in range(1, 13)
    if year < 2014 or month <= 11
)
DEVELOPMENT_ALPHA_LAST_EXIT_DATE = dt.date(2014, 12, 1)


class HoldingPeriodSessionAuthorityError(ValueError):
    """Raised when Stage 11 session authority cannot be authenticated."""


def _date(value: object, *, name: str) -> dt.date:
    if type(value) is not dt.date:
        raise HoldingPeriodSessionAuthorityError(
            f"{name} must contain only datetime.date values"
        )
    return value


@dataclass(frozen=True, slots=True, init=False)
class AuthenticatedDevelopmentSessionCalendar:
    """Privately issued bounded calendar/session source for the alpha gate."""

    rows: tuple[ValidationCalendarRow, ...]
    trading_dates: tuple[dt.date, ...]
    calendar_payload_sha256: str
    trading_payload_sha256: str

    def __init__(
        self,
        *,
        rows: tuple[ValidationCalendarRow, ...],
        trading_dates: tuple[dt.date, ...],
        calendar_payload_sha256: str,
        trading_payload_sha256: str,
        _issuance_token: object,
    ) -> None:
        if _issuance_token is not _DEVELOPMENT_CALENDAR_ISSUANCE_TOKEN:
            raise HoldingPeriodSessionAuthorityError(
                "development session calendar must be issued by its authenticated factory"
            )
        if (
            tuple(row.holding_month for row in rows)
            != DEVELOPMENT_ALPHA_HOLDING_MONTHS
            or not trading_dates
            or tuple(sorted(set(trading_dates))) != trading_dates
            or trading_dates[-1] != DEVELOPMENT_ALPHA_LAST_EXIT_DATE
            or calendar_payload_sha256 != EXPECTED_REBALANCE_CALENDAR_SHA256
            or trading_payload_sha256 != EXPECTED_TRADING_CALENDAR_SHA256
        ):
            raise HoldingPeriodSessionAuthorityError(
                "development session calendar identity or domain differs"
            )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "trading_dates", trading_dates)
        object.__setattr__(self, "calendar_payload_sha256", calendar_payload_sha256)
        object.__setattr__(self, "trading_payload_sha256", trading_payload_sha256)


@dataclass(frozen=True, slots=True, init=False)
class HoldingPeriodSessionAuthority:
    """Privately issued exact return sessions for one validation interval."""

    calendar_row: ValidationCalendarRow
    return_dates: tuple[dt.date, ...]
    source_payload_sha256: str

    def __init__(
        self,
        *,
        calendar_row: ValidationCalendarRow,
        return_dates: tuple[dt.date, ...],
        source_payload_sha256: str,
        _issuance_token: object,
    ) -> None:
        if _issuance_token is not _ISSUANCE_TOKEN:
            raise HoldingPeriodSessionAuthorityError(
                "holding-period session authority must be issued by its authenticated factory"
            )
        object.__setattr__(self, "calendar_row", calendar_row)
        object.__setattr__(self, "return_dates", return_dates)
        object.__setattr__(self, "source_payload_sha256", source_payload_sha256)

    @property
    def authority_sha256(self) -> str:
        """Content address of the source, interval, and exact session sequence."""

        row = self.calendar_row
        digest = hashlib.sha256(b"stage11-holding-period-session-authority-v1\0")
        for value in (
            self.source_payload_sha256,
            row.holding_month,
            row.signal_date.isoformat(),
            row.entry_date.isoformat(),
            row.exit_date.isoformat(),
        ):
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        for value in self.return_dates:
            digest.update(value.isoformat().encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def require_exact_return_dates(
        self,
        observed_dates: Sequence[dt.date],
    ) -> None:
        """Reject any reordered, missing, extra, or substituted session date."""

        if isinstance(observed_dates, (str, bytes)) or not isinstance(
            observed_dates, Sequence
        ):
            raise HoldingPeriodSessionAuthorityError(
                "observed return dates must be a sequence"
            )
        observed = tuple(
            _date(value, name="observed return dates") for value in observed_dates
        )
        if observed != self.return_dates:
            raise HoldingPeriodSessionAuthorityError(
                "daily inputs do not match the complete authenticated return-session sequence"
            )


@dataclass(frozen=True, slots=True)
class FinalCalendarIssuance:
    """One authenticated final calendar row opened before any outcome payload."""

    calendar_row: ValidationCalendarRow
    approval_payload_sha256: str
    stage12_binding_sha256: str
    calendar_payload_sha256: str
    calendar_columns: tuple[str, ...]
    access_receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.calendar_row) is not ValidationCalendarRow:
            raise HoldingPeriodSessionAuthorityError(
                "final calendar issuance row type differs"
            )
        for name in (
            "approval_payload_sha256",
            "stage12_binding_sha256",
            "calendar_payload_sha256",
            "access_receipt_sha256",
        ):
            _final_sha256(getattr(self, name), name=name)
        if not self.calendar_columns or any(
            not isinstance(value, str) for value in self.calendar_columns
        ):
            raise HoldingPeriodSessionAuthorityError(
                "final calendar issuance columns differ"
            )


@dataclass(frozen=True, slots=True)
class FinalHoldingPeriodSessionIssuance:
    """Broker-authenticated final session authority and its opened input table."""

    authority: HoldingPeriodSessionAuthority
    calendar_row: ValidationCalendarRow
    accounting_table: pa.Table
    access_receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.authority) is not HoldingPeriodSessionAuthority:
            raise HoldingPeriodSessionAuthorityError(
                "final issuance authority type differs"
            )
        if type(self.calendar_row) is not ValidationCalendarRow:
            raise HoldingPeriodSessionAuthorityError(
                "final issuance calendar row type differs"
            )
        if self.authority.calendar_row != self.calendar_row:
            raise HoldingPeriodSessionAuthorityError(
                "final issuance calendar and authority differ"
            )
        if not isinstance(self.accounting_table, pa.Table):
            raise HoldingPeriodSessionAuthorityError(
                "final issuance accounting table differs"
            )
        _final_sha256(self.access_receipt_sha256, name="access_receipt_sha256")


def _final_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HoldingPeriodSessionAuthorityError(f"{name} must be a lowercase SHA-256")
    return value


def _final_canonical_sha256(value: object) -> str:
    try:
        payload = (
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
        raise HoldingPeriodSessionAuthorityError(
            "final session receipt cannot be serialized"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _apply_final_return_authorization_overlay(
    accounting_table: pa.Table,
) -> tuple[pa.Table, Mapping[str, object]]:
    """Authorize only authenticated numeric, usable returns in memory.

    Frozen final accounting rows deliberately retain the Stage 2 source flag
    ``return_history_authorized=False``.  That flag records that the source
    table itself cannot authorize final-period use.  Only the post-decision
    broker/session issuance may derive an execution authorization, and it may
    do so solely for a non-null authoritative return whose source usability
    flag is true.  Nullable no-mark, suppressed-posting, unresolved-terminal,
    and every other nonnumeric row therefore remain unauthorized.

    The source Arrow table is immutable and is never rewritten.  The returned
    table carries the in-memory overlay consumed by the typed accounting
    projection, while the exact derived vector is content-bound into the
    session access receipt.
    """

    if not isinstance(accounting_table, pa.Table):
        raise HoldingPeriodSessionAuthorityError(
            "final return authorization overlay requires an Arrow table"
        )
    required = {
        "permno",
        "calendar_date",
        "source_payload_sha256",
        "authoritative_total_return",
        "return_history_usable",
        "return_history_authorized",
    }
    missing = sorted(required - set(accounting_table.column_names))
    if missing:
        raise HoldingPeriodSessionAuthorityError(
            "final return authorization overlay omits columns: " + ", ".join(missing)
        )
    source = accounting_table.combine_chunks()
    if source.num_rows == 0:
        raise HoldingPeriodSessionAuthorityError(
            "final return authorization overlay cannot authorize an empty table"
        )
    authorized_field = source.schema.field("return_history_authorized")
    usable_field = source.schema.field("return_history_usable")
    if not pa.types.is_boolean(authorized_field.type) or not pa.types.is_boolean(
        usable_field.type
    ):
        raise HoldingPeriodSessionAuthorityError(
            "final return authorization source flags are not boolean"
        )
    source_authorized = source["return_history_authorized"]
    source_usable = source["return_history_usable"]
    if source_authorized.null_count or source_usable.null_count:
        raise HoldingPeriodSessionAuthorityError(
            "final return authorization source flags contain nulls"
        )
    if pc.any(source_authorized).as_py() is not False:
        raise HoldingPeriodSessionAuthorityError(
            "final accounting source arrived self-authorized"
        )

    numeric = pc.is_valid(source["authoritative_total_return"])
    derived_authorized = pc.and_(numeric, source_usable)
    authorized_count = int(pc.sum(pc.cast(derived_authorized, pa.int64())).as_py())
    numeric_count = int(pc.sum(pc.cast(numeric, pa.int64())).as_py())
    usable_count = int(pc.sum(pc.cast(source_usable, pa.int64())).as_py())

    vector_digest = hashlib.sha256(b"stage13-session-authoritative-return-overlay-v1\0")
    for position, (raw_permno, raw_date, raw_source_sha, permitted) in enumerate(
        zip(
            source["permno"].to_pylist(),
            source["calendar_date"].to_pylist(),
            source["source_payload_sha256"].to_pylist(),
            derived_authorized.to_pylist(),
            strict=True,
        )
    ):
        if isinstance(raw_permno, bool) or not isinstance(raw_permno, int):
            raise HoldingPeriodSessionAuthorityError(
                "final return authorization PERMNO identity differs"
            )
        date_value = _date(raw_date, name="final return authorization date")
        source_sha = _final_sha256(
            raw_source_sha,
            name="final return authorization source_payload_sha256",
        )
        for value in (
            str(position),
            str(raw_permno),
            date_value.isoformat(),
            source_sha,
        ):
            vector_digest.update(value.encode("ascii"))
            vector_digest.update(b"\0")
        vector_digest.update(b"\1" if permitted else b"\0")

    authorized_index = source.schema.get_field_index("return_history_authorized")
    overlaid = source.set_column(
        authorized_index,
        authorized_field,
        derived_authorized,
    )
    receipt = {
        "overlay_id": FINAL_RETURN_AUTHORIZATION_OVERLAY_ID,
        "formula": (
            "return_history_authorized := "
            "is_valid(authoritative_total_return) AND return_history_usable"
        ),
        "source_rows_required_unauthorized": True,
        "source_row_count": source.num_rows,
        "source_authorized_row_count": 0,
        "numeric_authoritative_return_row_count": numeric_count,
        "source_usable_row_count": usable_count,
        "derived_authorized_row_count": authorized_count,
        "derived_unauthorized_row_count": source.num_rows - authorized_count,
        "authorization_vector_sha256": vector_digest.hexdigest(),
        "source_table_mutated": False,
    }
    return overlaid, receipt


def authenticate_final_calendar_row(
    *,
    access_broker: object,
    calendar_request: object,
) -> FinalCalendarIssuance:
    """Open and authenticate exactly one final calendar row pre-decision."""

    from portfolio_bot.final_test.access import (
        AuthorizedFinalTableRequest,
        FinalAccessBroker,
        FinalPayloadRole,
    )
    from portfolio_bot.final_test.calendar import (
        final_holding_month_descriptor,
        shift_month,
    )

    if type(access_broker) is not FinalAccessBroker:
        raise HoldingPeriodSessionAuthorityError(
            "final calendar issuance requires the exact live FinalAccessBroker"
        )
    if type(calendar_request) is not AuthorizedFinalTableRequest:
        raise HoldingPeriodSessionAuthorityError(
            "final calendar issuance requires a broker-issued table request"
        )
    if (
        calendar_request.registration_name != "final_calendar"
        or calendar_request.record.role is not FinalPayloadRole.CALENDAR_METADATA
        or calendar_request.columns != calendar_request.record.allowed_columns
        or len(calendar_request.holding_months) != 1
    ):
        raise HoldingPeriodSessionAuthorityError(
            "final calendar request differs from its full registered projection"
        )
    holding_month = calendar_request.holding_months[0]
    descriptor = final_holding_month_descriptor(holding_month)
    calendar_table = access_broker.read_guarded_parquet(calendar_request)
    if calendar_table.num_rows != 1:
        raise HoldingPeriodSessionAuthorityError(
            "final calendar request must return exactly one row"
        )
    raw_calendar = calendar_table.to_pylist()[0]
    try:
        calendar_row = ValidationCalendarRow(
            holding_month=raw_calendar["holding_month"],
            signal_date=raw_calendar["signal_date"],
            entry_date=raw_calendar["entry_date"],
            exit_date=raw_calendar["exit_date"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HoldingPeriodSessionAuthorityError(
            "final calendar row cannot be projected"
        ) from exc
    outcome_availability = raw_calendar.get("outcome_availability_date")
    if (
        calendar_row.holding_month != holding_month
        or not calendar_row.signal_date
        < calendar_row.entry_date
        < calendar_row.exit_date
        or calendar_row.signal_date.strftime("%Y-%m") != shift_month(holding_month, -1)
        or calendar_row.entry_date.strftime("%Y-%m") != holding_month
        or raw_calendar.get("segment") != descriptor.segment.value
        or raw_calendar.get("complete_holding_interval")
        is not descriptor.complete_holding_interval
        or raw_calendar.get("confirmatory_inference_eligible")
        is not descriptor.confirmatory_inference_eligible
        or type(outcome_availability) is not dt.date
        or outcome_availability < calendar_row.exit_date
    ):
        raise HoldingPeriodSessionAuthorityError(
            "final calendar row differs from the frozen segment contract"
        )
    receipt_payload = {
        "issuer": "stage13-final-calendar-issuance-v1",
        "holding_month": holding_month,
        "approval_payload_sha256": calendar_request.approval_payload_sha256,
        "stage12_binding_sha256": calendar_request.stage12_binding_sha256,
        "calendar_registration_name": calendar_request.registration_name,
        "calendar_payload_sha256": calendar_request.record.sha256,
        "calendar_columns": list(calendar_request.columns),
        "calendar_row": {
            "signal_date": calendar_row.signal_date.isoformat(),
            "entry_date": calendar_row.entry_date.isoformat(),
            "exit_date": calendar_row.exit_date.isoformat(),
            "segment": descriptor.segment.value,
        },
    }
    return FinalCalendarIssuance(
        calendar_row=calendar_row,
        approval_payload_sha256=calendar_request.approval_payload_sha256,
        stage12_binding_sha256=calendar_request.stage12_binding_sha256,
        calendar_payload_sha256=calendar_request.record.sha256,
        calendar_columns=tuple(calendar_request.columns),
        access_receipt_sha256=_final_canonical_sha256(receipt_payload),
    )


def authenticate_final_holding_period_session_authority(
    *,
    access_broker: object,
    calendar_issuance: object,
    accounting_request: object,
) -> FinalHoldingPeriodSessionIssuance:
    """Open final accounting only after a pre-decision calendar issuance.

    The calendar bytes are not reopened.  The accounting table is returned so
    callers can project the exact same authenticated bytes into the dated
    engine without a second read.
    """

    from portfolio_bot.final_test.access import (
        AuthorizedFinalTableRequest,
        FinalAccessBroker,
        FinalPayloadRole,
    )
    from portfolio_bot.final_test.calendar import (
        FinalPeriodSegment,
        final_holding_month_descriptor,
    )
    from portfolio_bot.final_test.input_materialization import (
        project_final_daily_accounting_inputs,
    )

    if type(access_broker) is not FinalAccessBroker:
        raise HoldingPeriodSessionAuthorityError(
            "final session issuance requires the exact live FinalAccessBroker"
        )
    if type(calendar_issuance) is not FinalCalendarIssuance:
        raise HoldingPeriodSessionAuthorityError(
            "final session issuance requires FinalCalendarIssuance"
        )
    if type(accounting_request) is not AuthorizedFinalTableRequest:
        raise HoldingPeriodSessionAuthorityError(
            "final session issuance requires a broker-issued accounting request"
        )
    calendar_row = calendar_issuance.calendar_row
    holding_month = calendar_row.holding_month
    descriptor = final_holding_month_descriptor(holding_month)
    accounting_name = {
        FinalPeriodSegment.PRIMARY_FULL: "primary_market_accounting",
        FinalPeriodSegment.CORE_COMPLETE: "core_market_accounting",
        FinalPeriodSegment.PARTIAL_DECEMBER_DESCRIPTIVE: ("partial_market_accounting"),
    }[descriptor.segment]
    if (
        accounting_request.registration_name != accounting_name
        or accounting_request.record.role is not FinalPayloadRole.ACCOUNTING_INPUT
        or accounting_request.columns != accounting_request.record.allowed_columns
        or accounting_request.holding_months != (holding_month,)
        or accounting_request.approval_payload_sha256
        != calendar_issuance.approval_payload_sha256
        or accounting_request.stage12_binding_sha256
        != calendar_issuance.stage12_binding_sha256
    ):
        raise HoldingPeriodSessionAuthorityError(
            "final accounting request differs from the registered segment/month"
        )

    source_accounting_table = access_broker.read_guarded_parquet(accounting_request)
    accounting_table, return_authorization_overlay = (
        _apply_final_return_authorization_overlay(source_accounting_table)
    )
    try:
        projection = project_final_daily_accounting_inputs(
            accounting_table,
            holding_month=holding_month,
        )
    except Exception as exc:
        raise HoldingPeriodSessionAuthorityError(
            "final accounting table failed its exact typed projection"
        ) from exc
    if (
        projection.signal_date != calendar_row.signal_date
        or projection.entry_date != calendar_row.entry_date
        or projection.exit_date != calendar_row.exit_date
    ):
        raise HoldingPeriodSessionAuthorityError(
            "final calendar and accounting intervals differ"
        )
    return_dates = tuple(value.trading_date for value in projection.daily_inputs)
    if (
        not return_dates
        or return_dates != tuple(sorted(set(return_dates)))
        or return_dates[0] <= calendar_row.entry_date
        or return_dates[-1] != calendar_row.exit_date
    ):
        raise HoldingPeriodSessionAuthorityError(
            "final accounting sessions do not form the exact (entry, exit] path"
        )
    receipt_payload = {
        "issuer": "stage13-final-broker-session-authority-v1",
        "holding_month": holding_month,
        "approval_payload_sha256": calendar_issuance.approval_payload_sha256,
        "stage12_binding_sha256": calendar_issuance.stage12_binding_sha256,
        "calendar_registration_name": "final_calendar",
        "calendar_payload_sha256": calendar_issuance.calendar_payload_sha256,
        "calendar_access_receipt_sha256": (calendar_issuance.access_receipt_sha256),
        "accounting_registration_name": accounting_request.registration_name,
        "accounting_payload_sha256": accounting_request.record.sha256,
        "calendar_columns": list(calendar_issuance.calendar_columns),
        "accounting_columns": list(accounting_request.columns),
        "return_dates": [value.isoformat() for value in return_dates],
        "daily_projection_sha256": [
            value.projected_payload_sha256 for value in projection.daily_inputs
        ],
        "return_authorization_overlay": dict(return_authorization_overlay),
    }
    access_receipt_sha256 = _final_canonical_sha256(receipt_payload)
    authority = HoldingPeriodSessionAuthority(
        calendar_row=calendar_row,
        return_dates=return_dates,
        source_payload_sha256=access_receipt_sha256,
        _issuance_token=_ISSUANCE_TOKEN,
    )
    return FinalHoldingPeriodSessionIssuance(
        authority=authority,
        calendar_row=calendar_row,
        accounting_table=accounting_table,
        access_receipt_sha256=access_receipt_sha256,
    )


def authenticate_holding_period_session_authority(
    *,
    calendar_authority: AuthenticatedValidationCalendar,
    calendar_row: ValidationCalendarRow,
    access_broker: PreflightAccessBroker,
    source_request: AuthorizedValidationTableRequest,
) -> HoldingPeriodSessionAuthority:
    """Read and seal the exact sessions for one authenticated calendar row.

    The source request must be an exact ``calendar_date`` projection over the
    row's inclusive entry/exit bounds.  The broker verifies the full frozen
    Parquet payload hash before applying that projection.  No caller-supplied
    date list or source digest is accepted.
    """

    if type(calendar_authority) is not AuthenticatedValidationCalendar:
        raise HoldingPeriodSessionAuthorityError(
            "session issuance requires AuthenticatedValidationCalendar"
        )
    if calendar_authority.source_payload_sha256 != EXPECTED_REBALANCE_CALENDAR_SHA256:
        raise HoldingPeriodSessionAuthorityError(
            "rebalance-calendar authority is not bound to the frozen Phase 2 payload"
        )
    if type(calendar_row) is not ValidationCalendarRow:
        raise HoldingPeriodSessionAuthorityError(
            "calendar_row must be a ValidationCalendarRow"
        )
    matches = tuple(row for row in calendar_authority.rows if row == calendar_row)
    if len(matches) != 1:
        raise HoldingPeriodSessionAuthorityError(
            "calendar row is not an exact member of the authenticated calendar"
        )
    if type(access_broker) is not PreflightAccessBroker:
        raise HoldingPeriodSessionAuthorityError(
            "session issuance requires the fail-closed Stage 11 access broker"
        )
    if type(source_request) is not AuthorizedValidationTableRequest:
        raise HoldingPeriodSessionAuthorityError(
            "session issuance requires an authorized source request"
        )
    record = source_request.record
    if (
        record.role is not PayloadRole.VALIDATION_TRADING_SESSION_METADATA
        or record.path != EXPECTED_TRADING_CALENDAR_PATH
        or record.sha256 != EXPECTED_TRADING_CALENDAR_SHA256
        or record.size_bytes != EXPECTED_TRADING_CALENDAR_SIZE_BYTES
        or record.allowed_columns != _SESSION_COLUMNS
        or source_request.columns != _SESSION_COLUMNS
        or source_request.minimum_signal_date != calendar_row.entry_date
        or source_request.maximum_signal_date != calendar_row.exit_date
    ):
        raise HoldingPeriodSessionAuthorityError(
            "session request differs from the frozen trading-calendar identity or interval"
        )

    table = access_broker.read_guarded_parquet(
        source_request,
        signal_date_column="calendar_date",
    )
    source_dates = tuple(table["calendar_date"].combine_chunks().to_pylist())
    if (
        not source_dates
        or any(type(value) is not dt.date for value in source_dates)
        or any(left >= right for left, right in zip(source_dates, source_dates[1:]))
        or source_dates[0] != calendar_row.entry_date
        or source_dates[-1] != calendar_row.exit_date
    ):
        raise HoldingPeriodSessionAuthorityError(
            "authenticated trading-calendar projection is incomplete or unordered"
        )
    return_dates = source_dates[1:]
    if not return_dates or return_dates[0] <= calendar_row.entry_date:
        raise HoldingPeriodSessionAuthorityError(
            "holding interval has no session strictly after the entry close"
        )
    return HoldingPeriodSessionAuthority(
        calendar_row=calendar_row,
        return_dates=return_dates,
        source_payload_sha256=record.sha256,
        _issuance_token=_ISSUANCE_TOKEN,
    )


def _stable_development_calendar_payload(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
) -> None:
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise HoldingPeriodSessionAuthorityError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != expected_size_bytes
    ):
        raise HoldingPeriodSessionAuthorityError(f"{label} is unsafe")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    after = path.stat(follow_symlinks=False)
    identity = lambda value: (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )
    if digest.hexdigest() != expected_sha256 or identity(before) != identity(after):
        raise HoldingPeriodSessionAuthorityError(
            f"{label} digest or stable identity differs"
        )


def authenticate_development_session_calendar(
) -> AuthenticatedDevelopmentSessionCalendar:
    """Authenticate only the bounded calendar/session metadata, not outcomes."""

    _stable_development_calendar_payload(
        EXPECTED_REBALANCE_CALENDAR_PATH,
        expected_sha256=EXPECTED_REBALANCE_CALENDAR_SHA256,
        expected_size_bytes=EXPECTED_REBALANCE_CALENDAR_SIZE_BYTES,
        label="development rebalance calendar",
    )
    _stable_development_calendar_payload(
        EXPECTED_TRADING_CALENDAR_PATH,
        expected_sha256=EXPECTED_TRADING_CALENDAR_SHA256,
        expected_size_bytes=EXPECTED_TRADING_CALENDAR_SIZE_BYTES,
        label="development trading calendar",
    )
    calendar = pq.read_table(
        EXPECTED_REBALANCE_CALENDAR_PATH,
        columns=[
            "holding_month",
            "stage",
            "signal_date",
            "entry_date",
            "exit_date",
            "is_embargo",
            "is_partial",
        ],
        filters=[
            ("stage", "=", "development"),
            ("holding_month", ">=", DEVELOPMENT_ALPHA_HOLDING_MONTHS[0]),
            ("holding_month", "<=", DEVELOPMENT_ALPHA_HOLDING_MONTHS[-1]),
        ],
        use_threads=False,
    )
    raw_rows = sorted(calendar.to_pylist(), key=lambda value: value["holding_month"])
    rows: list[ValidationCalendarRow] = []
    for expected_month, raw in zip(
        DEVELOPMENT_ALPHA_HOLDING_MONTHS, raw_rows, strict=True
    ):
        if (
            raw.get("holding_month") != expected_month
            or raw.get("stage") != "development"
            or raw.get("is_embargo") is not False
            or raw.get("is_partial") is not False
        ):
            raise HoldingPeriodSessionAuthorityError(
                "bounded development rebalance calendar differs"
            )
        try:
            row = ValidationCalendarRow(
                holding_month=raw["holding_month"],
                signal_date=raw["signal_date"],
                entry_date=raw["entry_date"],
                exit_date=raw["exit_date"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HoldingPeriodSessionAuthorityError(
                "bounded development calendar row is malformed"
            ) from exc
        if (
            not row.signal_date < row.entry_date < row.exit_date
            or row.exit_date > DEVELOPMENT_ALPHA_LAST_EXIT_DATE
        ):
            raise HoldingPeriodSessionAuthorityError(
                "bounded development calendar chronology differs"
            )
        rows.append(row)
    if len(rows) != len(DEVELOPMENT_ALPHA_HOLDING_MONTHS):
        raise HoldingPeriodSessionAuthorityError(
            "bounded development rebalance calendar does not have 95 rows"
        )
    first_entry = rows[0].entry_date
    trading = pq.read_table(
        EXPECTED_TRADING_CALENDAR_PATH,
        columns=list(_SESSION_COLUMNS),
        filters=[
            ("calendar_date", ">=", first_entry),
            ("calendar_date", "<=", DEVELOPMENT_ALPHA_LAST_EXIT_DATE),
        ],
        use_threads=False,
    )
    trading_dates = tuple(trading["calendar_date"].combine_chunks().to_pylist())
    if (
        not trading_dates
        or any(type(value) is not dt.date for value in trading_dates)
        or tuple(sorted(set(trading_dates))) != trading_dates
        or trading_dates[0] != first_entry
        or trading_dates[-1] != DEVELOPMENT_ALPHA_LAST_EXIT_DATE
        or any(
            row.entry_date not in trading_dates or row.exit_date not in trading_dates
            for row in rows
        )
    ):
        raise HoldingPeriodSessionAuthorityError(
            "bounded development trading calendar differs"
        )
    return AuthenticatedDevelopmentSessionCalendar(
        rows=tuple(rows),
        trading_dates=trading_dates,
        calendar_payload_sha256=EXPECTED_REBALANCE_CALENDAR_SHA256,
        trading_payload_sha256=EXPECTED_TRADING_CALENDAR_SHA256,
        _issuance_token=_DEVELOPMENT_CALENDAR_ISSUANCE_TOKEN,
    )


def authenticate_development_holding_period_session_authority(
    *,
    calendar_authority: AuthenticatedDevelopmentSessionCalendar,
    calendar_row: ValidationCalendarRow,
) -> HoldingPeriodSessionAuthority:
    """Issue one strict 2007--2014 authority from bounded metadata only."""

    if type(calendar_authority) is not AuthenticatedDevelopmentSessionCalendar:
        raise HoldingPeriodSessionAuthorityError(
            "development session issuance requires its exact calendar authority"
        )
    if type(calendar_row) is not ValidationCalendarRow or calendar_row not in calendar_authority.rows:
        raise HoldingPeriodSessionAuthorityError(
            "development calendar row is not an exact authenticated member"
        )
    interval = tuple(
        value
        for value in calendar_authority.trading_dates
        if calendar_row.entry_date <= value <= calendar_row.exit_date
    )
    if len(interval) < 2 or interval[0] != calendar_row.entry_date or interval[-1] != calendar_row.exit_date:
        raise HoldingPeriodSessionAuthorityError(
            "development sessions do not include the exact entry/exit boundary"
        )
    return_dates = interval[1:]
    digest = hashlib.sha256(b"research-v2-development-session-authority-v1\0")
    for value in (
        calendar_authority.calendar_payload_sha256,
        calendar_authority.trading_payload_sha256,
        calendar_row.holding_month,
        calendar_row.signal_date.isoformat(),
        calendar_row.entry_date.isoformat(),
        calendar_row.exit_date.isoformat(),
    ):
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return HoldingPeriodSessionAuthority(
        calendar_row=calendar_row,
        return_dates=return_dates,
        source_payload_sha256=digest.hexdigest(),
        _issuance_token=_ISSUANCE_TOKEN,
    )


def authenticate_execution_holding_period_session_authority(
    *,
    calendar_authority: AuthenticatedValidationCalendar,
    calendar_row: ValidationCalendarRow,
    access_broker: object,
    source_requests: Sequence[object],
) -> HoldingPeriodSessionAuthority:
    """Issue an empirical session authority from guarded main/reserve shards.

    Materialized Stage 11 sessions are physically split at the outcome-reserve
    boundary.  This issuer accepts one or two requests from the *same* live
    execution broker, authenticates every shard through that broker, and seals
    the exact ``(entry, exit]`` sequence.  It never accepts a caller-supplied
    session list or digest.
    """

    from portfolio_bot.validation.execution_access import (
        AuthorizedExecutionTableRequest,
        ValidationExecutionAccessBroker,
    )

    if type(calendar_authority) is not AuthenticatedValidationCalendar:
        raise HoldingPeriodSessionAuthorityError(
            "execution session issuance requires AuthenticatedValidationCalendar"
        )
    if calendar_authority.source_payload_sha256 != EXPECTED_REBALANCE_CALENDAR_SHA256:
        raise HoldingPeriodSessionAuthorityError(
            "execution calendar is not bound to the frozen rebalance payload"
        )
    if (
        type(calendar_row) is not ValidationCalendarRow
        or calendar_row not in calendar_authority.rows
    ):
        raise HoldingPeriodSessionAuthorityError(
            "execution calendar row is not an exact authenticated member"
        )
    if type(access_broker) is not ValidationExecutionAccessBroker:
        raise HoldingPeriodSessionAuthorityError(
            "execution session issuance requires the exact live broker"
        )
    if isinstance(source_requests, (str, bytes)) or not isinstance(
        source_requests, Sequence
    ):
        raise HoldingPeriodSessionAuthorityError(
            "execution session requests must be a sequence"
        )
    requests = tuple(source_requests)
    if (
        not requests
        or len(requests) > 2
        or any(type(value) is not AuthorizedExecutionTableRequest for value in requests)
    ):
        raise HoldingPeriodSessionAuthorityError(
            "execution session issuance requires one or two authorized shards"
        )
    dates: list[dt.date] = []
    source_hashes: list[str] = []
    for request in requests:
        record = request.record
        if (
            record.role is not PayloadRole.VALIDATION_TRADING_SESSION_METADATA
            or request.columns
            not in {
                ("calendar_date",),
                (
                    "signal_date",
                    "holding_month",
                    "entry_date",
                    "exit_date",
                    "calendar_date",
                ),
            }
            or request.minimum_date < calendar_row.entry_date
            or request.maximum_date > calendar_row.exit_date
        ):
            raise HoldingPeriodSessionAuthorityError(
                "execution session request differs from the frozen interval contract"
            )
        table = access_broker.read_guarded_parquet(request)
        shard_dates = tuple(table["calendar_date"].combine_chunks().to_pylist())
        if any(type(value) is not dt.date for value in shard_dates):
            raise HoldingPeriodSessionAuthorityError(
                "execution session shard contains malformed dates"
            )
        dates.extend(shard_dates)
        source_hashes.append(record.sha256)
    observed = tuple(sorted(set(dates)))
    if (
        not observed
        or len(observed) != len(dates)
        or any(left >= right for left, right in zip(observed, observed[1:]))
        or observed[0] <= calendar_row.entry_date
        or observed[-1] != calendar_row.exit_date
    ):
        raise HoldingPeriodSessionAuthorityError(
            "execution session shards do not form the exact (entry, exit] path"
        )
    # Bind the set of authenticated shard payloads, not their mutable names.
    digest = hashlib.sha256(b"stage11-execution-session-shards-v1\0")
    for value in sorted(source_hashes):
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return HoldingPeriodSessionAuthority(
        calendar_row=calendar_row,
        return_dates=observed,
        source_payload_sha256=digest.hexdigest(),
        _issuance_token=_ISSUANCE_TOKEN,
    )


def authenticate_materialized_holding_period_session_authority(
    *,
    calendar_authority: AuthenticatedValidationCalendar,
    calendar_row: ValidationCalendarRow,
    outcome_publication: object,
) -> HoldingPeriodSessionAuthority:
    """Issue session authority from exact outcome-materialization receipts."""

    from portfolio_bot.validation.outcome_materialization import (
        Stage11OutcomePublication,
    )

    if type(calendar_authority) is not AuthenticatedValidationCalendar:
        raise HoldingPeriodSessionAuthorityError(
            "materialized session issuance requires authenticated calendar authority"
        )
    if (
        type(calendar_row) is not ValidationCalendarRow
        or calendar_row not in calendar_authority.rows
    ):
        raise HoldingPeriodSessionAuthorityError(
            "materialized session calendar row is not an authenticated member"
        )
    if type(outcome_publication) is not Stage11OutcomePublication:
        raise HoldingPeriodSessionAuthorityError(
            "materialized session issuance requires the exact outcome publication"
        )
    names = (
        "daily_sessions.parquet",
        "outcome_reserve_daily_sessions.parquet",
    )
    table_receipts = outcome_publication.receipt.get("tables")
    if not isinstance(table_receipts, Mapping):
        raise HoldingPeriodSessionAuthorityError(
            "materialized session receipts are absent"
        )
    dates: list[dt.date] = []
    source_hashes: list[str] = []
    for name in names:
        receipt = table_receipts.get(name)
        if not isinstance(receipt, Mapping) or receipt.get("path") != name:
            raise HoldingPeriodSessionAuthorityError(
                f"materialized session receipt differs: {name}"
            )
        path = outcome_publication.output_root / name
        try:
            before = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise HoldingPeriodSessionAuthorityError(
                f"materialized session table is unavailable: {name}"
            ) from exc
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != receipt.get("size_bytes")
        ):
            raise HoldingPeriodSessionAuthorityError(
                f"materialized session table is unsafe: {name}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 << 20):
                digest.update(block)
        if digest.hexdigest() != receipt.get("sha256"):
            raise HoldingPeriodSessionAuthorityError(
                f"materialized session table digest differs: {name}"
            )
        table = pq.read_table(
            path,
            columns=[
                "signal_date",
                "holding_month",
                "entry_date",
                "exit_date",
                "calendar_date",
            ],
            filters=[("signal_date", "=", calendar_row.signal_date)],
            use_threads=False,
        )
        after = path.stat(follow_symlinks=False)
        identity = lambda value: (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_mode),
            int(value.st_nlink),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )
        if identity(before) != identity(after):
            raise HoldingPeriodSessionAuthorityError(
                f"materialized session table changed while read: {name}"
            )
        for value in table.to_pylist():
            if (
                value["signal_date"] != calendar_row.signal_date
                or value["holding_month"] != calendar_row.holding_month
                or value["entry_date"] != calendar_row.entry_date
                or value["exit_date"] != calendar_row.exit_date
                or type(value["calendar_date"]) is not dt.date
            ):
                raise HoldingPeriodSessionAuthorityError(
                    "materialized session interval identity differs"
                )
            dates.append(value["calendar_date"])
        source_hashes.append(str(receipt["sha256"]))
    observed = tuple(sorted(dates))
    if (
        not observed
        or len(observed) != len(set(observed))
        or observed[0] <= calendar_row.entry_date
        or observed[-1] != calendar_row.exit_date
        or any(left >= right for left, right in zip(observed, observed[1:]))
    ):
        raise HoldingPeriodSessionAuthorityError(
            "materialized sessions do not form the exact (entry, exit] path"
        )
    digest = hashlib.sha256(b"stage11-materialized-session-shards-v1\0")
    for value in sorted(source_hashes):
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return HoldingPeriodSessionAuthority(
        calendar_row=calendar_row,
        return_dates=observed,
        source_payload_sha256=digest.hexdigest(),
        _issuance_token=_ISSUANCE_TOKEN,
    )


__all__ = [
    "AuthenticatedDevelopmentSessionCalendar",
    "DEVELOPMENT_ALPHA_HOLDING_MONTHS",
    "DEVELOPMENT_ALPHA_LAST_EXIT_DATE",
    "FINAL_RETURN_AUTHORIZATION_OVERLAY_ID",
    "EXPECTED_TRADING_CALENDAR_PATH",
    "EXPECTED_TRADING_CALENDAR_SHA256",
    "EXPECTED_TRADING_CALENDAR_SIZE_BYTES",
    "EXPECTED_REBALANCE_CALENDAR_PATH",
    "EXPECTED_REBALANCE_CALENDAR_SIZE_BYTES",
    "FinalCalendarIssuance",
    "FinalHoldingPeriodSessionIssuance",
    "HoldingPeriodSessionAuthority",
    "HoldingPeriodSessionAuthorityError",
    "authenticate_development_holding_period_session_authority",
    "authenticate_development_session_calendar",
    "authenticate_execution_holding_period_session_authority",
    "authenticate_final_calendar_row",
    "authenticate_final_holding_period_session_authority",
    "authenticate_holding_period_session_authority",
    "authenticate_materialized_holding_period_session_authority",
]
