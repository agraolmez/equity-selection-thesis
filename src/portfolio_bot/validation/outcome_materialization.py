"""Authority-gated Stage 11 label and daily-market materialization.

Nothing in this module opens an empirical file without a live
``Stage11ValidationUnblindAuthority``.  The caller supplies exact, hash-bound
registrations to :class:`ValidationExecutionAccessBroker`; this module then
constructs the 59 January-2015--November-2019 validation label months with the
unchanged Stage 7 formulas and exposes the inputs needed by fixed-share daily
accounting.

The final 2019-11-01--2019-12-02 observations are deliberately supplied and
published as a separate reserve.  They may complete the October and November
2019 holding outcomes, but they can never enter predictor history.  The
December-2019 holding month and every 2020+ row are structurally absent.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import stat
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.features.label_math import FACTOR_NAMES
from portfolio_bot.features.label_store_contract import published_schemas
from portfolio_bot.features.label_store_materialization import (
    DevelopmentCalendarRow,
    materialize_label_signal_month,
    project_development_labels,
)
from portfolio_bot.validation.access import (
    FROZEN_STAGE6_RAW_SCHEMA_SHA256,
    PayloadRole,
)
from portfolio_bot.validation.calendar import (
    FIRST_VALIDATION_ENTRY_DATE,
    FIRST_VALIDATION_SIGNAL_DATE,
    LAST_VALIDATION_ENTRY_DATE,
    LAST_VALIDATION_EXIT_DATE,
    LAST_VALIDATION_SIGNAL_DATE,
    VALIDATION_HOLDING_MONTH_COUNT,
    VALIDATION_HOLDING_MONTHS,
    ValidationCalendarRow,
    validate_validation_calendar,
)
from portfolio_bot.validation.execution_access import (
    ExecutionAccessLedger,
    ExecutionDateAxis,
    RegisteredExecutionPayload,
    ValidationExecutionAccessBroker,
)
from portfolio_bot.validation.execution_risk_inputs import (
    RISK_COST_REGISTRATION_NAMES,
    validate_risk_cost_registrations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGE7_RULES_PATH = PROJECT_ROOT / "configs/phase_03_features/label_build_rules_v1.json"
STAGE7_LABEL_MATH_PATH = PROJECT_ROOT / "src/portfolio_bot/features/label_math.py"
STAGE7_MATERIALIZER_PATH = (
    PROJECT_ROOT / "src/portfolio_bot/features/label_store_materialization.py"
)
FROZEN_STAGE7_RULES_SHA256 = (
    "f93f006062ba441f61c9cdda3a5d4fc1cb7b8b5ffdf422f1bebe7ea3057d2947"
)
FROZEN_STAGE7_LABEL_MATH_SHA256 = (
    "fd745e8ba7457d426f62e287a292cdf8e9985e4e6c0eb1927f838b526f9cec07"
)
FROZEN_STAGE7_MATERIALIZER_SHA256 = (
    "868e06badd148ba2ae409a04f483e75fd9a5110f24a23700215866e741009261"
)
FROZEN_INPUT_MANIFEST_PATH = (
    PROJECT_ROOT / "artifacts/phase_05_validation/STAGE_11_OUTCOME_INPUTS.json"
)

OUTCOME_RESERVE_START = dt.date(2019, 11, 1)
MAIN_OBSERVATION_END = dt.date(2019, 10, 31)
FIRST_LABEL_EXIT_DATE = dt.date(2015, 2, 2)
FIRST_PROHIBITED_DATE = dt.date(2020, 1, 2)
CONTRACT_ID = "phase05-stage11-outcome-materialization-v1"
STATUS_COMPLETE = "STAGE11_OUTCOME_MATERIALIZATION_COMPLETE"

CALENDAR_COLUMNS = (
    "signal_date",
    "holding_month",
    "entry_date",
    "exit_date",
)
PREDICTOR_COLUMNS = (
    "signal_date",
    "permno",
    "permco",
    "long_eligible",
    "short_eligible",
    *(f"beta_label_{name}" for name in FACTOR_NAMES),
)
SESSION_COLUMNS = ("calendar_date",)
MARKET_COLUMNS = ("permno", "calendar_date", "close_usd_per_share")
RETURN_COLUMNS = (
    "permno",
    "calendar_date",
    "return_leg_class",
    "return_span_start_date",
    "return_span_end_date",
    "covered_trading_sessions",
    "span_alignment_status",
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
FACTOR_COLUMNS = (
    "observation_date",
    "public_availability_date",
    "mktrf",
    "smb",
    "hml",
    "rmw",
    "cma",
    "rf",
    "umd",
)
DELISTING_COLUMNS = (
    "permno",
    "delisting_date",
    "posting_date",
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


class Stage11OutcomeMaterializationError(RuntimeError):
    """Raised when a gated input, chronology, formula, or receipt fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11OutcomeMaterializationError(message)


def _canonical_json(value: Any) -> bytes:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _rules() -> Mapping[str, Any]:
    _require(
        _sha256(STAGE7_RULES_PATH) == FROZEN_STAGE7_RULES_SHA256
        and _sha256(STAGE7_LABEL_MATH_PATH) == FROZEN_STAGE7_LABEL_MATH_SHA256
        and _sha256(STAGE7_MATERIALIZER_PATH) == FROZEN_STAGE7_MATERIALIZER_SHA256,
        "frozen Stage 7 rules or formula implementation changed",
    )
    try:
        value = json.loads(STAGE7_RULES_PATH.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11OutcomeMaterializationError(
            "frozen Stage 7 label rules are unavailable"
        ) from exc
    _require(isinstance(value, dict), "frozen Stage 7 label rules are malformed")
    _require(
        value.get("rules_id") == "phase03-label-build-rules-v1",
        "frozen Stage 7 label rules identity differs",
    )
    return MappingProxyType(value)


@dataclass(frozen=True, slots=True)
class Stage11OutcomeInputNames:
    """Fixed logical registrations; reserve inputs can never be predictors."""

    calendar: str = "validation_calendar"
    predictor: str = "validation_predictor"
    sessions_main: str = "validation_sessions_main"
    sessions_reserve: str = "validation_sessions_reserve"
    market_main: str = "validation_market_main"
    market_reserve: str = "validation_market_reserve"
    returns_main: str = "validation_returns_main"
    returns_reserve: str = "validation_returns_reserve"
    factors_main: str = "validation_factors_main"
    factors_reserve: str = "validation_factors_reserve"
    delistings_main: str = "validation_delistings_main"
    delistings_reserve: str = "validation_delistings_reserve"

    def values(self) -> tuple[str, ...]:
        return tuple(getattr(self, field.name) for field in fields(self))


DEFAULT_INPUT_NAMES = Stage11OutcomeInputNames()


@dataclass(frozen=True, slots=True)
class Stage11OutcomeTables:
    """In-memory result that must remain inside the live authority scope."""

    validation_labels: pa.Table
    validation_label_audit: pa.Table
    validation_label_leg_lineage: pa.Table
    validation_label_month_summary: pa.Table
    entry_closes: pa.Table
    daily_total_returns: pa.Table
    daily_factors: pa.Table
    daily_sessions: pa.Table
    delisting_inputs: pa.Table
    reserve_entry_closes: pa.Table
    reserve_daily_total_returns: pa.Table
    reserve_daily_factors: pa.Table
    reserve_daily_sessions: pa.Table
    reserve_delisting_inputs: pa.Table

    def as_mapping(self) -> Mapping[str, pa.Table]:
        return MappingProxyType(
            {
                "validation_labels.parquet": self.validation_labels,
                "validation_label_audit.parquet": self.validation_label_audit,
                "validation_label_leg_lineage.parquet": self.validation_label_leg_lineage,
                "validation_label_month_summary.parquet": self.validation_label_month_summary,
                "entry_closes.parquet": self.entry_closes,
                "daily_total_returns.parquet": self.daily_total_returns,
                "daily_factors.parquet": self.daily_factors,
                "daily_sessions.parquet": self.daily_sessions,
                "delisting_inputs.parquet": self.delisting_inputs,
                "outcome_reserve_entry_closes.parquet": self.reserve_entry_closes,
                "outcome_reserve_daily_total_returns.parquet": self.reserve_daily_total_returns,
                "outcome_reserve_daily_factors.parquet": self.reserve_daily_factors,
                "outcome_reserve_daily_sessions.parquet": self.reserve_daily_sessions,
                "outcome_reserve_delisting_inputs.parquet": self.reserve_delisting_inputs,
            }
        )


@dataclass(frozen=True, slots=True)
class Stage11OutcomeBuild:
    tables: Stage11OutcomeTables
    calendar_rows: tuple[ValidationCalendarRow, ...]
    ledger: ExecutionAccessLedger
    input_fingerprint: str


@dataclass(frozen=True, slots=True)
class Stage11OutcomePublication:
    output_root: Path
    build_id: str
    receipt: Mapping[str, Any]
    reused_complete_receipt: bool
    calendar_rows: tuple[ValidationCalendarRow, ...]


def _specifications(
    names: Stage11OutcomeInputNames,
) -> Mapping[
    str, tuple[PayloadRole, tuple[str, ...], dt.date, dt.date, ExecutionDateAxis]
]:
    main = (FIRST_VALIDATION_ENTRY_DATE, MAIN_OBSERVATION_END)
    reserve = (OUTCOME_RESERVE_START, LAST_VALIDATION_EXIT_DATE)
    return MappingProxyType(
        {
            names.calendar: (
                PayloadRole.VALIDATION_CALENDAR_METADATA,
                CALENDAR_COLUMNS,
                FIRST_VALIDATION_SIGNAL_DATE,
                LAST_VALIDATION_SIGNAL_DATE,
                ExecutionDateAxis.SIGNAL,
            ),
            names.predictor: (
                PayloadRole.VALIDATION_PREDICTOR,
                PREDICTOR_COLUMNS,
                FIRST_VALIDATION_SIGNAL_DATE,
                LAST_VALIDATION_SIGNAL_DATE,
                ExecutionDateAxis.SIGNAL,
            ),
            names.sessions_main: (
                PayloadRole.VALIDATION_TRADING_SESSION_METADATA,
                SESSION_COLUMNS,
                *main,
                ExecutionDateAxis.OBSERVATION,
            ),
            names.sessions_reserve: (
                PayloadRole.VALIDATION_TRADING_SESSION_METADATA,
                SESSION_COLUMNS,
                *reserve,
                ExecutionDateAxis.OBSERVATION,
            ),
            names.market_main: (
                PayloadRole.VALIDATION_MARKET_INPUT,
                MARKET_COLUMNS,
                *main,
                ExecutionDateAxis.OBSERVATION,
            ),
            names.market_reserve: (
                PayloadRole.VALIDATION_MARKET_INPUT,
                MARKET_COLUMNS,
                *reserve,
                ExecutionDateAxis.OBSERVATION,
            ),
            names.returns_main: (
                PayloadRole.VALIDATION_FORWARD_RETURN,
                RETURN_COLUMNS,
                *main,
                ExecutionDateAxis.OBSERVATION,
            ),
            names.returns_reserve: (
                PayloadRole.VALIDATION_FORWARD_RETURN,
                RETURN_COLUMNS,
                *reserve,
                ExecutionDateAxis.OBSERVATION,
            ),
            names.factors_main: (
                PayloadRole.VALIDATION_ACCOUNTING_INPUT,
                FACTOR_COLUMNS,
                *main,
                ExecutionDateAxis.OBSERVATION,
            ),
            names.factors_reserve: (
                PayloadRole.VALIDATION_ACCOUNTING_INPUT,
                FACTOR_COLUMNS,
                *reserve,
                ExecutionDateAxis.OBSERVATION,
            ),
            names.delistings_main: (
                PayloadRole.VALIDATION_ACCOUNTING_INPUT,
                DELISTING_COLUMNS,
                *main,
                ExecutionDateAxis.OBSERVATION,
            ),
            names.delistings_reserve: (
                PayloadRole.VALIDATION_ACCOUNTING_INPUT,
                DELISTING_COLUMNS,
                *reserve,
                ExecutionDateAxis.OBSERVATION,
            ),
        }
    )


def _validate_registration_contract(
    registered: Mapping[str, RegisteredExecutionPayload],
    names: Stage11OutcomeInputNames,
) -> None:
    expected = _specifications(names)
    observed = set(registered)
    expected_names = set(expected)
    extras = observed.difference(expected_names)
    _require(
        expected_names.issubset(observed)
        and frozenset(extras) in {frozenset(), frozenset(RISK_COST_REGISTRATION_NAMES)},
        "outcome/execution input registration set differs",
    )
    if extras:
        validate_risk_cost_registrations(
            {name: registered[name] for name in RISK_COST_REGISTRATION_NAMES}
        )
    _require(
        len(names.values()) == len(set(names.values())), "outcome input names collide"
    )
    for name, (role, columns, minimum, maximum, axis) in expected.items():
        record = registered[name]
        _require(record.role is role, f"outcome input role differs: {name}")
        if name == names.predictor:
            # The empirical execution front door replaces the narrow label-
            # materialization registration with the independently verified
            # 170-column predictor registration.  The outcome materializer
            # still requests only ``PREDICTOR_COLUMNS`` below, so this does
            # not widen what label construction reads; it merely lets model
            # fitting and label construction share one authority-bound broker
            # without registering the same physical payload twice.
            _require(
                set(columns).issubset(record.allowed_columns),
                f"outcome input projection differs: {name}",
            )
        else:
            _require(
                record.allowed_columns == columns,
                f"outcome input projection differs: {name}",
            )
        _require(record.date_axis is axis, f"outcome input date axis differs: {name}")
        _require(
            record.minimum_date == minimum and record.maximum_date == maximum,
            f"outcome input boundary differs: {name}",
        )
        expected_date_column = {
            names.calendar: "signal_date",
            names.predictor: "signal_date",
            names.sessions_main: "calendar_date",
            names.sessions_reserve: "calendar_date",
            names.market_main: "calendar_date",
            names.market_reserve: "calendar_date",
            names.returns_main: "calendar_date",
            names.returns_reserve: "calendar_date",
            names.factors_main: "observation_date",
            names.factors_reserve: "observation_date",
            names.delistings_main: "assigned_return_date",
            names.delistings_reserve: "assigned_return_date",
        }[name]
        _require(
            record.date_column == expected_date_column, f"date column differs: {name}"
        )
    predictor = registered[names.predictor]
    _require(
        predictor.trusted_schema_sha256 == FROZEN_STAGE6_RAW_SCHEMA_SHA256,
        "validation beta predictors require the frozen Stage 6 schema authority",
    )
    _require(
        registered[names.calendar].holding_month_column == "holding_month",
        "validation calendar must bind its holding-month column",
    )
    for name in set(expected) - {names.calendar}:
        _require(
            registered[name].holding_month_column is None,
            f"source table may not carry a caller-declared holding month: {name}",
        )


def _read_inputs(
    broker: ValidationExecutionAccessBroker,
    names: Stage11OutcomeInputNames,
) -> dict[str, pa.Table]:
    result: dict[str, pa.Table] = {}
    for name, (role, columns, minimum, maximum, _axis) in _specifications(
        names
    ).items():
        request = broker.authorize_table_request(
            name,
            role=role,
            columns=columns,
            minimum_date=minimum,
            maximum_date=maximum,
        )
        result[name] = broker.read_guarded_parquet(request)
    return result


def _concat(left: pa.Table, right: pa.Table, *, label: str) -> pa.Table:
    _require(
        left.schema.remove_metadata().equals(
            right.schema.remove_metadata(), check_metadata=True
        ),
        f"{label} main/reserve schema differs",
    )
    # Scope metadata is expected to differ: the reserve is independently
    # receipted as 2019-11-01--2019-12-02.  Field names, types, nullability,
    # and field-level semantics must remain identical.
    return pa.concat_tables(
        (
            left.replace_schema_metadata(None),
            right.replace_schema_metadata(None),
        )
    ).combine_chunks()


def _sort(table: pa.Table, keys: Sequence[str]) -> pa.Table:
    if table.num_rows == 0:
        return table.combine_chunks()
    indices = pc.sort_indices(table, sort_keys=[(key, "ascending") for key in keys])
    return pc.take(table, indices).combine_chunks()


def _require_unique(table: pa.Table, keys: Sequence[str], label: str) -> None:
    if table.num_rows == 0:
        return
    connection = duckdb.connect(":memory:")
    try:
        connection.register("candidate", table)
        columns = ", ".join(f'"{value}"' for value in keys)
        duplicate = connection.execute(
            f"SELECT COUNT(*) FROM (SELECT {columns}, COUNT(*) n FROM candidate "
            f"GROUP BY {columns} HAVING COUNT(*) > 1)"
        ).fetchone()[0]
    finally:
        connection.close()
    _require(int(duplicate) == 0, f"{label} contains duplicate keys")


def _validate_calendar(table: pa.Table) -> tuple[ValidationCalendarRow, ...]:
    _require(
        tuple(table.column_names) == CALENDAR_COLUMNS, "calendar projection differs"
    )
    rows = tuple(ValidationCalendarRow.from_mapping(row) for row in table.to_pylist())
    return validate_validation_calendar(rows)


def _validate_predictor(
    table: pa.Table,
    calendar: Sequence[ValidationCalendarRow],
) -> pa.Table:
    _require(
        tuple(table.column_names) == PREDICTOR_COLUMNS, "predictor projection differs"
    )
    _require(table.num_rows > 0, "validation predictor is empty")
    for name in ("signal_date", "permno", "permco", "long_eligible", "short_eligible"):
        _require(
            table[name].null_count == 0, f"predictor key/eligibility is null: {name}"
        )
    ordered = _sort(table, ("signal_date", "permno"))
    _require_unique(ordered, ("signal_date", "permno"), "validation predictor")
    observed = tuple(sorted(set(ordered["signal_date"].to_pylist())))
    expected = tuple(row.signal_date for row in calendar)
    _require(
        observed == expected, "predictor does not contain the exact 59 signal dates"
    )
    _require(
        all(int(value) > 0 for value in ordered["permno"].combine_chunks().to_pylist()),
        "predictor contains an invalid PERMNO",
    )
    return ordered


def _validate_sessions(
    main: pa.Table,
    reserve: pa.Table,
    calendar: Sequence[ValidationCalendarRow],
) -> tuple[pa.Table, tuple[dt.date, ...]]:
    combined = _sort(_concat(main, reserve, label="session"), ("calendar_date",))
    _require(combined["calendar_date"].null_count == 0, "sessions contain null dates")
    sessions = tuple(combined["calendar_date"].to_pylist())
    _require(sessions == tuple(sorted(set(sessions))), "sessions are not sorted unique")
    _require(
        sessions
        and sessions[0] == FIRST_VALIDATION_ENTRY_DATE
        and sessions[-1] == LAST_VALIDATION_EXIT_DATE,
        "session boundary differs",
    )
    session_set = set(sessions)
    for row in calendar:
        required = tuple(
            value for value in sessions if row.entry_date < value <= row.exit_date
        )
        _require(
            row.entry_date in session_set
            and required
            and required[-1] == row.exit_date,
            f"calendar interval lacks exact entry/exit sessions: {row.holding_month}",
        )
    return combined, sessions


def _validate_daily_sources(
    market: pa.Table,
    returns: pa.Table,
    factors: pa.Table,
    delistings: pa.Table,
    sessions: Sequence[dt.date],
) -> None:
    _require_unique(market, ("permno", "calendar_date"), "market source")
    _require_unique(returns, ("permno", "calendar_date"), "return source")
    _require_unique(factors, ("observation_date",), "factor source")
    _require_unique(
        delistings,
        ("permno", "delisting_date"),
        "delisting source",
    )
    session_set = set(sessions)
    factor_dates = tuple(factors["observation_date"].to_pylist())
    _require(
        factor_dates == tuple(sorted(factor_dates))
        and set(factor_dates) == session_set,
        "daily factors do not exactly cover verified sessions",
    )
    _require(
        all(
            value is not None and value <= date_value
            for date_value, value in zip(
                factor_dates,
                factors["public_availability_date"].to_pylist(),
                strict=True,
            )
        ),
        "factor publication date exceeds its observation date",
    )
    if delistings.num_rows:
        _require(
            all(
                bool(value) for value in delistings["assigned_exactly_once"].to_pylist()
            ),
            "delisting input is not assigned exactly once",
        )
        _require(
            all(
                value in session_set
                for value in delistings["assigned_return_date"].to_pylist()
            ),
            "delisting input is not assigned to a verified session",
        )


def _interval_table(
    predictor: pa.Table,
    calendar: Sequence[ValidationCalendarRow],
) -> pa.Table:
    by_signal = {row.signal_date: row for row in calendar}
    rows: list[dict[str, Any]] = []
    for row in predictor.select(
        ("signal_date", "permno", "long_eligible", "short_eligible")
    ).to_pylist():
        period = by_signal[row["signal_date"]]
        rows.append(
            {
                **row,
                "holding_month": period.holding_month,
                "entry_date": period.entry_date,
                "exit_date": period.exit_date,
            }
        )
    schema = pa.schema(
        [
            pa.field("signal_date", pa.date32(), nullable=False),
            pa.field("permno", pa.int64(), nullable=False),
            pa.field("long_eligible", pa.bool_(), nullable=False),
            pa.field("short_eligible", pa.bool_(), nullable=False),
            pa.field("holding_month", pa.string(), nullable=False),
            pa.field("entry_date", pa.date32(), nullable=False),
            pa.field("exit_date", pa.date32(), nullable=False),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _query_tables(
    intervals: pa.Table,
    market: pa.Table,
    returns: pa.Table,
    factors: pa.Table,
    sessions_table: pa.Table,
    delistings: pa.Table,
) -> tuple[pa.Table, pa.Table, pa.Table, pa.Table, pa.Table]:
    connection = duckdb.connect(":memory:")
    try:
        for name, table in (
            ("intervals", intervals),
            ("market", market),
            ("returns", returns),
            ("factors", factors),
            ("sessions", sessions_table),
            ("delistings", delistings),
        ):
            connection.register(name, table)
        # Portfolio holdings are stateful: a name selected in an earlier month
        # can remain economically open after it disappears from the current
        # predictor spine.  Preserve an authenticated return/action path for
        # every security observed on or before each signal.  This table is used
        # only by accounting; label and selector domains remain the exact
        # contemporaneous predictor population.
        connection.execute(
            """
            CREATE TEMP TABLE accounting_paths AS
            WITH periods AS (
                SELECT DISTINCT signal_date, holding_month, entry_date, exit_date
                FROM intervals
            ), first_seen AS (
                SELECT permno, MIN(signal_date) AS first_signal_date
                FROM intervals
                GROUP BY permno
            )
            SELECT p.signal_date, p.holding_month, p.entry_date, p.exit_date,
                   f.permno, p.entry_date AS active_after_date
            FROM periods p
            JOIN first_seen f ON f.first_signal_date <= p.signal_date
            """
        )
        entry = connection.execute(
            """
            SELECT i.signal_date, i.holding_month, i.entry_date, i.exit_date,
                   i.permno, CAST(m.close_usd_per_share AS DOUBLE) close_usd_per_share,
                   CASE
                     WHEN m.permno IS NULL THEN 'MISSING_ENTRY_MARKET_ROW'
                     WHEN m.close_usd_per_share IS NULL OR m.close_usd_per_share <= 0
                       THEN 'MISSING_VALID_ENTRY_CLOSE'
                     ELSE 'PRESENT'
                   END entry_close_status
            FROM intervals i
            LEFT JOIN market m
              ON m.permno = i.permno AND m.calendar_date = i.entry_date
            ORDER BY i.signal_date, i.permno
            """
        ).fetch_arrow_table()
        # Never infer accounting settlement from raw successor nullability.
        # The frozen Stage 7 label contract prohibits successor-security
        # returns and ends the source path at its terminal payoff.  A future
        # reviewed stock-consideration extension must start from the explicit
        # corporate-action mapping authority; raw non-null successor fields
        # are not sufficient authorization to follow another security.
        daily = connection.execute(
            """
            SELECT DISTINCT i.signal_date, i.holding_month, i.entry_date, i.exit_date,
                   r.*
            FROM accounting_paths i
            JOIN returns r ON r.permno = i.permno
              AND r.calendar_date > i.active_after_date
              AND r.calendar_date <= i.exit_date
            ORDER BY i.signal_date, r.calendar_date, r.permno
            """
        ).fetch_arrow_table()
        daily_factors = connection.execute(
            """
            SELECT DISTINCT i.signal_date, i.holding_month, i.entry_date, i.exit_date,
                   f.*
            FROM intervals i
            JOIN factors f
              ON f.observation_date > i.entry_date
             AND f.observation_date <= i.exit_date
            ORDER BY i.signal_date, f.observation_date
            """
        ).fetch_arrow_table()
        daily_sessions = connection.execute(
            """
            SELECT DISTINCT i.signal_date, i.holding_month, i.entry_date, i.exit_date,
                   s.calendar_date
            FROM intervals i
            JOIN sessions s
              ON s.calendar_date > i.entry_date
             AND s.calendar_date <= i.exit_date
            ORDER BY i.signal_date, s.calendar_date
            """
        ).fetch_arrow_table()
        action = connection.execute(
            """
            SELECT DISTINCT i.signal_date, i.holding_month, i.entry_date, i.exit_date,
                   d.*
            FROM accounting_paths i
            JOIN delistings d ON d.permno = i.permno
              AND d.assigned_return_date > i.active_after_date
              AND d.assigned_return_date <= i.exit_date
            ORDER BY i.signal_date, d.assigned_return_date, d.permno, d.delisting_date
            """
        ).fetch_arrow_table()
    finally:
        connection.close()
    return entry, daily, daily_factors, daily_sessions, action


def _routed_label_returns(
    returns: pa.Table,
    intervals: pa.Table,
) -> pa.Table:
    """Reproduce the Stage 7 router's normal and exit-straddle delivery."""

    long_intervals = intervals.filter(intervals["long_eligible"])
    connection = duckdb.connect(":memory:")
    try:
        connection.register("returns", returns)
        connection.register("intervals", long_intervals)
        routed = connection.execute(
            """
            SELECT i.signal_date, r.*
            FROM intervals i
            JOIN returns r ON r.permno = i.permno
             AND r.calendar_date >= i.signal_date
             AND r.calendar_date <= LEAST(
                   i.exit_date + INTERVAL 45 DAY,
                   DATE '2019-12-02'
                 )
            WHERE r.calendar_date <= i.exit_date
               OR (
                    r.calendar_date > i.exit_date
                AND r.return_span_start_date < i.exit_date
                AND r.return_span_end_date > i.exit_date
               )
            ORDER BY i.signal_date, r.permno, r.calendar_date
            """
        ).fetch_arrow_table()
    finally:
        connection.close()
    return routed


def _factor_mapping(factors: pa.Table) -> dict[dt.date, Any]:
    from portfolio_bot.features.label_math import FactorObservation

    result: dict[dt.date, FactorObservation] = {}
    for row in factors.to_pylist():
        date_value = row["observation_date"]
        result[date_value] = FactorObservation(
            calendar_date=date_value,
            rf=None if row["rf"] is None else float(row["rf"]),
            mktrf=None if row["mktrf"] is None else float(row["mktrf"]),
            smb=None if row["smb"] is None else float(row["smb"]),
            hml=None if row["hml"] is None else float(row["hml"]),
            rmw=None if row["rmw"] is None else float(row["rmw"]),
            cma=None if row["cma"] is None else float(row["cma"]),
            umd=None if row["umd"] is None else float(row["umd"]),
            public_availability_date=row["public_availability_date"],
        )
    return result


def _slice_date(table: pa.Table, column: str, reserve: bool) -> pa.Table:
    values = table[column]
    boundary = pa.scalar(OUTCOME_RESERVE_START, type=pa.date32())
    mask = pc.greater_equal(values, boundary) if reserve else pc.less(values, boundary)
    return table.filter(pc.fill_null(mask, False)).combine_chunks()


def _materialize_labels(
    predictor: pa.Table,
    routed: pa.Table,
    calendar: Sequence[ValidationCalendarRow],
    sessions: Sequence[dt.date],
    factors: pa.Table,
    rules: Mapping[str, Any],
) -> tuple[pa.Table, pa.Table, pa.Table, pa.Table]:
    factor_map = _factor_mapping(factors)
    audit: list[pa.Table] = []
    lineage: list[pa.Table] = []
    summaries: list[pa.Table] = []
    labels: list[pa.Table] = []
    for row in calendar:
        feature_month = predictor.filter(
            pc.equal(
                predictor["signal_date"], pa.scalar(row.signal_date, type=pa.date32())
            )
        )
        routed_month = routed.filter(
            pc.equal(
                routed["signal_date"], pa.scalar(row.signal_date, type=pa.date32())
            )
        )
        month = materialize_label_signal_month(
            feature_month=feature_month,
            routed_returns=routed_month,
            calendar=DevelopmentCalendarRow(
                signal_date=row.signal_date,
                holding_month=row.holding_month,
                entry_date=row.entry_date,
                exit_date=row.exit_date,
            ),
            sessions=sessions,
            factors=factor_map,
            rules=rules,
        )
        audit.append(month.audit)
        lineage.append(month.lineage)
        summaries.append(month.summary)
        labels.append(project_development_labels(month.audit, rules=rules))
    schemas = published_schemas(rules)
    result = (
        pa.concat_tables(labels).cast(schemas["development_labels.parquet"]),
        pa.concat_tables(audit).cast(schemas["development_label_audit.parquet"]),
        pa.concat_tables(lineage).cast(
            schemas["development_label_leg_lineage.parquet"]
        ),
        pa.concat_tables(summaries).cast(
            schemas["development_label_month_summary.parquet"]
        ),
    )
    summary = result[3]
    _require(
        summary.num_rows == VALIDATION_HOLDING_MONTH_COUNT,
        "validation label materialization did not create exactly 59 months",
    )
    _require(
        tuple(summary["holding_month"].to_pylist()) == VALIDATION_HOLDING_MONTHS,
        "validation label months differ from the frozen sequence",
    )
    _require(
        max(result[0]["exit_date"].to_pylist(), default=FIRST_LABEL_EXIT_DATE)
        <= LAST_VALIDATION_EXIT_DATE,
        "validation labels crossed the final exit",
    )
    return result


def _input_fingerprint(
    authority: object,
    registered: Mapping[str, RegisteredExecutionPayload],
    rules_sha256: str,
) -> str:
    payload = {
        "contract_id": CONTRACT_ID,
        "authority": {
            "build_id": getattr(authority, "build_id"),
            "verification_id": getattr(authority, "verification_id"),
            "approval_contract_file_sha256": getattr(
                authority, "approval_contract_file_sha256"
            ),
            "approval_contract_payload_sha256": getattr(
                authority, "approval_contract_payload_sha256"
            ),
            "execution_input_manifest_sha256": getattr(
                authority, "execution_input_manifest_sha256"
            ),
        },
        "rules_sha256": rules_sha256,
        "stage7_label_math_sha256": _sha256(STAGE7_LABEL_MATH_PATH),
        "stage7_materializer_sha256": _sha256(STAGE7_MATERIALIZER_PATH),
        "code_sha256": _sha256(Path(__file__)),
        "inputs": {
            name: {
                "role": record.role.value,
                "path": record.path.as_posix(),
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
                "allowed_columns": record.allowed_columns,
                "date_column": record.date_column,
                "date_axis": record.date_axis.value,
                "minimum_date": record.minimum_date.isoformat(),
                "maximum_date": record.maximum_date.isoformat(),
                "holding_month_column": record.holding_month_column,
                "trusted_schema_sha256": record.trusted_schema_sha256,
            }
            for name, record in sorted(registered.items())
        },
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _build_stage11_outcome_tables_with_broker(
    authority: object,
    registered: Mapping[str, RegisteredExecutionPayload],
    broker: ValidationExecutionAccessBroker,
    *,
    names: Stage11OutcomeInputNames = DEFAULT_INPUT_NAMES,
) -> Stage11OutcomeBuild:
    # Authenticate through the broker before inspecting either mapping.  The
    # authenticated execution front door passes one broker through outcome
    # materialization and the later walk-forward so the final ledger is a
    # single continuous record rather than a collection of partial ledgers.
    broker.ledger()
    _require(
        dict(broker.registered) == dict(registered),
        "outcome materializer broker registrations differ",
    )
    _validate_registration_contract(broker.registered, names)
    rules = _rules()
    source = _read_inputs(broker, names)

    calendar = _validate_calendar(source[names.calendar])
    predictor = _validate_predictor(source[names.predictor], calendar)
    sessions_table, sessions = _validate_sessions(
        source[names.sessions_main], source[names.sessions_reserve], calendar
    )
    market = _sort(
        _concat(
            source[names.market_main], source[names.market_reserve], label="market"
        ),
        ("permno", "calendar_date"),
    )
    returns = _sort(
        _concat(
            source[names.returns_main], source[names.returns_reserve], label="return"
        ),
        ("permno", "calendar_date"),
    )
    factors = _sort(
        _concat(
            source[names.factors_main], source[names.factors_reserve], label="factor"
        ),
        ("observation_date",),
    )
    delistings = _sort(
        _concat(
            source[names.delistings_main],
            source[names.delistings_reserve],
            label="delisting",
        ),
        ("assigned_return_date", "permno", "delisting_date"),
    )
    _validate_daily_sources(market, returns, factors, delistings, sessions)

    intervals = _interval_table(predictor, calendar)
    routed = _routed_label_returns(returns, intervals)
    labels, audit, lineage, summary = _materialize_labels(
        predictor, routed, calendar, sessions, factors, rules
    )
    entry, daily, daily_factors, daily_sessions, actions = _query_tables(
        intervals,
        market,
        returns,
        factors,
        sessions_table,
        delistings,
    )

    tables = Stage11OutcomeTables(
        validation_labels=labels,
        validation_label_audit=audit,
        validation_label_leg_lineage=lineage,
        validation_label_month_summary=summary,
        entry_closes=_slice_date(entry, "entry_date", False),
        daily_total_returns=_slice_date(daily, "calendar_date", False),
        daily_factors=_slice_date(daily_factors, "observation_date", False),
        daily_sessions=_slice_date(daily_sessions, "calendar_date", False),
        delisting_inputs=_slice_date(actions, "assigned_return_date", False),
        reserve_entry_closes=_slice_date(entry, "entry_date", True),
        reserve_daily_total_returns=_slice_date(daily, "calendar_date", True),
        reserve_daily_factors=_slice_date(daily_factors, "observation_date", True),
        reserve_daily_sessions=_slice_date(daily_sessions, "calendar_date", True),
        reserve_delisting_inputs=_slice_date(actions, "assigned_return_date", True),
    )
    _require(
        tables.reserve_entry_closes.num_rows > 0
        and min(tables.reserve_entry_closes["entry_date"].to_pylist())
        == LAST_VALIDATION_ENTRY_DATE,
        "reserved entry-close payload does not begin at 2019-11-01",
    )
    for table_name, table in tables.as_mapping().items():
        for column in ("signal_date", "entry_date", "exit_date"):
            if column in table.column_names and table.num_rows:
                values = table[column].to_pylist()
                _require(
                    max(values) < FIRST_PROHIBITED_DATE, f"{table_name} crossed 2020"
                )
        if "holding_month" in table.column_names and table.num_rows:
            _require(
                set(table["holding_month"].to_pylist()).issubset(
                    VALIDATION_HOLDING_MONTHS
                ),
                f"{table_name} contains an embargo/final holding month",
            )

    ledger = broker.ledger()
    _require(
        ledger.prohibited_counters_are_zero, "execution access ledger is contaminated"
    )
    fingerprint = _input_fingerprint(
        authority, broker.registered, _sha256(STAGE7_RULES_PATH)
    )
    return Stage11OutcomeBuild(
        tables=tables,
        calendar_rows=tuple(calendar),
        ledger=ledger,
        input_fingerprint=fingerprint,
    )


def build_stage11_outcome_tables(
    authority: object,
    registered: Mapping[str, RegisteredExecutionPayload],
    *,
    names: Stage11OutcomeInputNames = DEFAULT_INPUT_NAMES,
) -> Stage11OutcomeBuild:
    """Read and materialize only while ``authority`` is live.

    Authority authentication is intentionally the first operation.  Even the
    registration names and paths are not inspected before the gate succeeds.
    """

    broker = ValidationExecutionAccessBroker(authority, registered)
    return _build_stage11_outcome_tables_with_broker(
        authority,
        registered,
        broker,
        names=names,
    )


def _safe_output_root(path: Path) -> Path:
    _require(
        isinstance(path, Path) and path.is_absolute(), "output root must be absolute"
    )
    candidate = Path(os.path.abspath(path.expanduser()))
    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        cursor /= part
        if cursor.exists():
            observed = os.lstat(cursor)
            _require(
                not stat.S_ISLNK(observed.st_mode),
                "symlinked output path is prohibited",
            )
    candidate.mkdir(parents=True, exist_ok=True)
    _require(candidate.is_dir() and not candidate.is_symlink(), "unsafe output root")
    return candidate


def _table_receipt(path: Path) -> dict[str, Any]:
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and observed.st_nlink == 1,
        f"unsafe materialized table: {path.name}",
    )
    parquet = pq.ParquetFile(path)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "schema_sha256": _schema_sha256(parquet.schema_arrow),
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _require(not temporary.exists(), "stale temporary receipt exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_or_reuse_table(
    root: Path,
    name: str,
    table: pa.Table,
    fingerprint: str,
) -> dict[str, Any]:
    path = root / name
    checkpoint = root / f"checkpoint_{Path(name).stem}.json"
    if checkpoint.exists():
        try:
            value = json.loads(checkpoint.read_text("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage11OutcomeMaterializationError(
                f"invalid checkpoint receipt: {checkpoint.name}"
            ) from exc
        _require(
            value.get("input_fingerprint") == fingerprint,
            "checkpoint fingerprint differs",
        )
        _require(path.is_file(), f"checkpoint table is missing: {name}")
        observed = _table_receipt(path)
        _require(observed == value.get("table"), f"checkpoint table changed: {name}")
        _require(
            observed["rows"] == table.num_rows
            and observed["schema_sha256"] == _schema_sha256(table.schema),
            f"checkpoint table shape differs: {name}",
        )
        return observed
    _require(not path.exists(), f"unreceipted output exists: {name}")
    temporary = root / f".{name}.{os.getpid()}.tmp"
    _require(not temporary.exists(), f"stale temporary table exists: {name}")
    try:
        pq.write_table(
            table,
            temporary,
            row_group_size=100_000,
            compression="zstd",
            compression_level=9,
            use_dictionary=True,
            version="2.6",
            write_statistics=True,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt = _table_receipt(path)
    _atomic_json(
        checkpoint,
        {
            "schema_version": 1,
            "status": "STAGE11_OUTCOME_TABLE_COMPLETE",
            "contract_id": CONTRACT_ID,
            "input_fingerprint": fingerprint,
            "table": receipt,
        },
    )
    return receipt


def _publish_stage11_outcome_tables_with_broker(
    authority: object,
    registered: Mapping[str, RegisteredExecutionPayload],
    output_root: Path,
    broker: ValidationExecutionAccessBroker,
    *,
    names: Stage11OutcomeInputNames = DEFAULT_INPUT_NAMES,
) -> Stage11OutcomePublication:
    # Authenticate before touching either the output path or registrations.
    broker.ledger()
    root = _safe_output_root(output_root)
    build = _build_stage11_outcome_tables_with_broker(
        authority,
        registered,
        broker,
        names=names,
    )
    build_id = f"stage11-outcomes-{build.input_fingerprint[:24]}"
    final_path = root / "outcome_materialization_receipt.json"
    table_names = tuple(build.tables.as_mapping())
    permitted_names = {
        *table_names,
        *(f"checkpoint_{Path(name).stem}.json" for name in table_names),
        final_path.name,
    }
    unexpected = sorted(
        path.name for path in root.iterdir() if path.name not in permitted_names
    )
    _require(
        not unexpected, f"outcome output root contains unexpected files: {unexpected}"
    )
    if final_path.exists():
        value = json.loads(final_path.read_text("utf-8"))
        _require(
            value.get("input_fingerprint") == build.input_fingerprint,
            "final receipt differs",
        )
        for receipt in value.get("tables", {}).values():
            _require(
                _table_receipt(root / receipt["path"]) == receipt,
                "final materialized output changed",
            )
        return Stage11OutcomePublication(
            root, build_id, value, True, build.calendar_rows
        )

    receipts: dict[str, Any] = {}
    for name, table in build.tables.as_mapping().items():
        receipts[name] = _write_or_reuse_table(
            root, name, table, build.input_fingerprint
        )
    receipt = {
        "schema_version": 1,
        "status": STATUS_COMPLETE,
        "contract_id": CONTRACT_ID,
        "build_id": build_id,
        "input_fingerprint": build.input_fingerprint,
        "authority_build_id": build.ledger.authority_build_id,
        "authority_verification_id": build.ledger.authority_verification_id,
        "label_months": VALIDATION_HOLDING_MONTH_COUNT,
        "first_holding_month": VALIDATION_HOLDING_MONTHS[0],
        "last_holding_month": VALIDATION_HOLDING_MONTHS[-1],
        "outcome_reserve": {
            "minimum_date": OUTCOME_RESERVE_START.isoformat(),
            "maximum_date": LAST_VALIDATION_EXIT_DATE.isoformat(),
            "predictor_use_permitted": False,
        },
        "stage7_formula_reuse": {
            "rules_path": STAGE7_RULES_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "rules_sha256": _sha256(STAGE7_RULES_PATH),
            "label_math_sha256": _sha256(STAGE7_LABEL_MATH_PATH),
            "materializer_sha256": _sha256(STAGE7_MATERIALIZER_PATH),
            "formula_module": "portfolio_bot.features.label_math",
            "materializer_module": "portfolio_bot.features.label_store_materialization",
        },
        "access_ledger": build.ledger.to_dict(),
        "tables": receipts,
        "final_period_rows_read": 0,
        "december_2019_holding_month_rows": 0,
    }
    _atomic_json(final_path, receipt)
    return Stage11OutcomePublication(
        root, build_id, receipt, False, build.calendar_rows
    )


def publish_stage11_outcome_tables(
    authority: object,
    registered: Mapping[str, RegisteredExecutionPayload],
    output_root: Path,
    *,
    names: Stage11OutcomeInputNames = DEFAULT_INPUT_NAMES,
) -> Stage11OutcomePublication:
    """Materialize and checkpoint every authority-gated Stage 11 table."""

    broker = ValidationExecutionAccessBroker(authority, registered)
    return _publish_stage11_outcome_tables_with_broker(
        authority,
        registered,
        output_root,
        broker,
        names=names,
    )


def publish_from_authenticated_execution_inputs(
    authority: object,
    execution_inputs: object,
) -> Stage11OutcomePublication:
    """Publish through the single execution-manifest authority and broker."""

    # Local import avoids the module cycle: the execution manifest uses this
    # module to parse its hash-bound outcome-source registrations.
    from portfolio_bot.validation.execution_input_manifest import (
        AuthenticatedStage11ExecutionInputs,
    )

    _require(
        type(execution_inputs) is AuthenticatedStage11ExecutionInputs,
        "outcome publication requires authenticated Stage 11 execution inputs",
    )
    execution_inputs.gate()
    _require(
        execution_inputs._stage11_authority is authority,
        "outcome publication authority differs from execution inputs",
    )
    return _publish_stage11_outcome_tables_with_broker(
        authority,
        execution_inputs.registrations,
        execution_inputs.output_roots["outcome_tables"],
        execution_inputs.broker,
    )


def materialize_from_frozen_manifest(authority: object) -> Stage11OutcomePublication:
    """Compatibility entrypoint routed through the sole execution front door."""

    from portfolio_bot.validation.execution_input_manifest import (
        authenticate_stage11_execution_inputs,
    )

    execution_inputs = authenticate_stage11_execution_inputs(authority)
    return publish_from_authenticated_execution_inputs(authority, execution_inputs)


__all__ = [
    "CALENDAR_COLUMNS",
    "CONTRACT_ID",
    "DEFAULT_INPUT_NAMES",
    "DELISTING_COLUMNS",
    "FACTOR_COLUMNS",
    "FROZEN_INPUT_MANIFEST_PATH",
    "MAIN_OBSERVATION_END",
    "MARKET_COLUMNS",
    "OUTCOME_RESERVE_START",
    "PREDICTOR_COLUMNS",
    "RETURN_COLUMNS",
    "SESSION_COLUMNS",
    "STATUS_COMPLETE",
    "Stage11OutcomeBuild",
    "Stage11OutcomeInputNames",
    "Stage11OutcomeMaterializationError",
    "Stage11OutcomePublication",
    "Stage11OutcomeTables",
    "build_stage11_outcome_tables",
    "materialize_from_frozen_manifest",
    "publish_from_authenticated_execution_inputs",
    "publish_stage11_outcome_tables",
]
