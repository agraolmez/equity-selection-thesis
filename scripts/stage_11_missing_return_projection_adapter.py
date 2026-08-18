"""Stage 11-only projection of nullable CRSP rows into daily accounting.

The canonical outcome table deliberately retains structural missing-return and
suppressed posting rows as nulls.  They are audit records, not zero returns.
Exactly authenticated ordinary missing-return rows are supplied to the dated
engine as explicit no-mark records.  Their identity, borrow authority, and
source-record digest remain separate from observed numeric returns.  The dated
engine alone decides whether a later observed return span exactly resolves a
pending mark.  Every other nullable return class remains a sparse omission, so
an exposed identity still fails closed.

The matching unresolved nonperformance delisting rows contain no known
accounting instruction.  The canonical corporate-action mapper rejects them
before strategy exposure is available, even for an unheld security.  This
adapter retains those rows as audit evidence, requires an exact nullable
daily-return twin, and maps every resolved action through the unchanged
canonical mapper.  The unchanged dated engine remains the economic gate: an
exact nonzero long or short position without a return still fails closed.

No null is changed to zero, no price is reconstructed, no settlement is
invented, and both canonical call-site bindings are restored on exit.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa

import portfolio_bot.validation.empirical_portfolio_backend as portfolio_backend_module
from portfolio_bot.validation.corporate_action_adapter import (
    CorporateActionMappingAuthority,
    UNRESOLVED_TERMINAL_TREATMENT,
    map_delisting_rows_to_corporate_actions as _CANONICAL_ACTION_MAPPER,
)
from portfolio_bot.validation.corporate_actions import CorporateAction
from portfolio_bot.validation.daily_engine import DatedMarketCloseInput
from portfolio_bot.validation.empirical_portfolio_backend import (
    _daily_inputs as _CANONICAL_DAILY_INPUTS,
)
from portfolio_bot.validation.problem_builder import MonthlyProblemBuild


_NULL_STATES = frozenset(
    {
        "missing",
        "quarantined_MV",
        "quarantined_unresolved_span",
        "suppressed_delisting_posting_payload",
        "unresolved_nonperformance_delisting",
    }
)
_NULL_LEG_CLASSES = frozenset(
    {
        "missing_return",
        "ordinary_one_session",
        "ordinary_multi_session",
        "ordinary_with_unresolved_terminal_delisting",
        "terminal_delisting_posting_audit_suppressed",
    }
)
_UNRESOLVED_RETURN_STATE = "unresolved_nonperformance_delisting"
_UNRESOLVED_RETURN_LEG = "ordinary_with_unresolved_terminal_delisting"
_UNRESOLVED_EMBEDDING_STATUS = "posting_daily_total_missing"
_NO_MARK_RETURN_STATE = "missing"
_NO_MARK_RETURN_LEG = "missing_return"
_NO_MARK_SPAN_ALIGNMENT = "no_return_span"
_NO_MARK_TREATMENT = "none"
_EXPECTED_UNRESOLVED_COUNT = 20
_EXPECTED_UNRESOLVED_KEY_SHA256 = (
    "3888ef6b7e50a3eed374aeb1d53af00df8172d0cf1b62bff86efcdec92473bec"
)
_EXPECTED_NO_MARK_SOURCE_COUNT = 65_157
_EXPECTED_NO_MARK_KEY_SHA256 = (
    "7390f43681c866d4f81ad614662d2908ece14a239b81eaaae25504fc686eb2b0"
)


class Stage11MissingReturnProjectionError(RuntimeError):
    """Raised when a row is not an authenticated nullable audit record."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11MissingReturnProjectionError(message)


def _as_date(value: object, *, label: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    raise Stage11MissingReturnProjectionError(f"{label} is not a date")


def _canonical_value(value: object) -> object:
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    return value


def _canonical_json(value: object) -> bytes:
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


def _key_sha256(keys: set[tuple[dt.date, str, int, dt.date]]) -> str:
    payload = [
        {
            "assigned_return_date": assigned.isoformat(),
            "holding_month": holding_month,
            "permno": permno,
            "signal_date": signal_date.isoformat(),
        }
        for signal_date, holding_month, permno, assigned in sorted(keys)
    ]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _no_mark_key_sha256(
    keys: set[tuple[dt.date, str, int, dt.date]],
) -> str:
    payload = [
        {
            "calendar_date": calendar_date.isoformat(),
            "holding_month": holding_month,
            "permno": permno,
            "signal_date": signal_date.isoformat(),
        }
        for signal_date, holding_month, permno, calendar_date in sorted(keys)
    ]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _normalize_return_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    """Normalize only identity/span types needed by the projection contract."""

    result = rows.copy()
    try:
        for name in (
            "signal_date",
            "calendar_date",
            "return_span_start_date",
            "return_span_end_date",
        ):
            result[name] = pd.to_datetime(result[name], errors="raise").dt.date
        numeric_permno = pd.to_numeric(result["permno"], errors="raise")
    except (TypeError, ValueError, OverflowError) as exc:
        raise Stage11MissingReturnProjectionError(
            "daily return projection identity/span metadata is invalid"
        ) from exc
    permno = numeric_permno.to_numpy(np.float64)
    _require(
        np.isfinite(permno).all()
        and np.equal(permno, np.floor(permno)).all()
        and np.all(permno > 0.0),
        "daily return projection PERMNO is invalid",
    )
    result["permno"] = permno.astype(np.int64)
    return result


def _literal_false(value: object) -> bool:
    return isinstance(value, (bool, np.bool_)) and not bool(value)


def _exact_no_mark_mask(rows: pd.DataFrame) -> np.ndarray:
    """Identify the sole nullable structure authorized as an explicit no-mark."""

    covered = pd.to_numeric(rows["covered_trading_sessions"], errors="coerce")
    events = pd.to_numeric(rows["delisting_event_count"], errors="coerce")
    return np.asarray(
        rows["authoritative_total_return"].isna()
        & rows["authoritative_return_state"].eq(_NO_MARK_RETURN_STATE)
        & rows["return_leg_class"].eq(_NO_MARK_RETURN_LEG)
        & rows["return_span_start_date"].isna()
        & rows["return_span_end_date"].notna()
        & rows["return_span_end_date"].eq(rows["calendar_date"])
        & covered.eq(0)
        & rows["span_alignment_status"].eq(_NO_MARK_SPAN_ALIGNMENT)
        & events.eq(0)
        & rows["delisting_treatment_status"].eq(_NO_MARK_TREATMENT)
        & rows["delisting_adjustment_return"].isna()
        & rows["return_history_usable"].map(_literal_false)
        & rows["return_history_authorized"].map(_literal_false),
        dtype=np.bool_,
    )


def _no_mark_identity(
    row: object,
) -> tuple[dt.date, str, int, dt.date]:
    signal_date = _as_date(
        getattr(row, "signal_date"),
        label="no-mark signal date",
    )
    calendar_date = _as_date(
        getattr(row, "calendar_date"),
        label="no-mark calendar date",
    )
    holding_month = getattr(row, "holding_month")
    permno = getattr(row, "permno")
    _require(
        isinstance(holding_month, str)
        and bool(holding_month)
        and isinstance(permno, (int, np.integer))
        and int(permno) > 0,
        "no-mark source identity is invalid",
    )
    return signal_date, holding_month, int(permno), calendar_date


def _no_mark_source_record_sha256(row: object) -> str:
    signal_date, holding_month, permno, calendar_date = _no_mark_identity(row)
    payload = {
        "authoritative_return_state": _NO_MARK_RETURN_STATE,
        "authoritative_total_return": None,
        "calendar_date": calendar_date.isoformat(),
        "covered_trading_sessions": 0,
        "delisting_adjustment_return": None,
        "delisting_event_count": 0,
        "delisting_treatment_status": _NO_MARK_TREATMENT,
        "holding_month": holding_month,
        "permno": permno,
        "return_history_authorized": False,
        "return_history_usable": False,
        "return_leg_class": _NO_MARK_RETURN_LEG,
        "return_span_end_date": calendar_date.isoformat(),
        "return_span_start_date": None,
        "signal_date": signal_date.isoformat(),
        "span_alignment_status": _NO_MARK_SPAN_ALIGNMENT,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(slots=True)
class MissingReturnProjectionAudit:
    """Deterministic aggregate evidence for the sparse projection."""

    calls: int = 0
    action_projection_calls: int = 0
    resolved_action_rows_mapped: int = 0
    unresolved_actions_mapped: int = 0
    zero_imputations: int = 0
    _row_keys: set[tuple[object, ...]] = field(default_factory=set)
    _unresolved_action_keys: set[
        tuple[dt.date, str, int, dt.date]
    ] = field(default_factory=set)
    _unresolved_return_keys: set[
        tuple[dt.date, str, int, dt.date]
    ] = field(default_factory=set)
    _unresolved_action_source_hashes: set[str] = field(default_factory=set)
    _no_mark_source_by_key: dict[
        tuple[dt.date, str, int, dt.date], str
    ] = field(default_factory=dict)
    projected_no_mark_rows: int = 0
    _state_counts: dict[str, int] = field(default_factory=dict)
    _leg_counts: dict[str, int] = field(default_factory=dict)

    def record(self, rows: pd.DataFrame) -> None:
        self.calls += 1
        unresolved_keys_in_call: set[
            tuple[dt.date, str, int, dt.date]
        ] = set()
        for row in rows.itertuples(index=False):
            key = (
                getattr(row, "signal_date", None),
                getattr(row, "holding_month", None),
                int(row.permno),
                row.calendar_date,
                str(row.authoritative_return_state),
                str(row.return_leg_class),
            )
            state = str(row.authoritative_return_state)
            leg = str(row.return_leg_class)
            treatment = str(row.delisting_treatment_status)
            unresolved_state = state == _UNRESOLVED_RETURN_STATE
            unresolved_leg = leg == _UNRESOLVED_RETURN_LEG
            unresolved_treatment = (
                treatment == UNRESOLVED_TERMINAL_TREATMENT
            )
            _require(
                unresolved_state
                == unresolved_leg
                == unresolved_treatment,
                "unresolved terminal return state/leg/treatment pairing differs",
            )
            if unresolved_state:
                signal_date = _as_date(
                    getattr(row, "signal_date", None),
                    label="unresolved return signal date",
                )
                assigned_date = _as_date(
                    row.calendar_date,
                    label="unresolved return assigned date",
                )
                unresolved_key = (
                    signal_date,
                    str(getattr(row, "holding_month", "")),
                    int(row.permno),
                    assigned_date,
                )
                _require(
                    unresolved_key not in unresolved_keys_in_call,
                    "unresolved terminal nullable-return twin is duplicated",
                )
                unresolved_keys_in_call.add(unresolved_key)
                self._unresolved_return_keys.add(unresolved_key)
            if key in self._row_keys:
                continue
            self._row_keys.add(key)
            self._state_counts[state] = self._state_counts.get(state, 0) + 1
            self._leg_counts[leg] = self._leg_counts.get(leg, 0) + 1

    def record_no_mark_sources(self, rows: pd.DataFrame) -> None:
        """Bind every exact no-mark source row once across strategy projections."""

        for row in rows.itertuples(index=False):
            key = _no_mark_identity(row)
            digest = _no_mark_source_record_sha256(row)
            prior = self._no_mark_source_by_key.get(key)
            _require(
                prior is None or prior == digest,
                "no-mark source identity changed across projections",
            )
            self._no_mark_source_by_key[key] = digest

    def record_no_mark_projection(self, count: int) -> None:
        _require(
            isinstance(count, int) and count >= 0,
            "projected no-mark row count is invalid",
        )
        self.projected_no_mark_rows += count

    def record_unresolved_actions(
        self,
        rows: tuple[dict[str, Any], ...],
        *,
        observed_columns: tuple[str, ...],
    ) -> None:
        """Authenticate unresolved rows retained as audit evidence only."""

        for raw in rows:
            _require(
                raw.get("treatment_status") == UNRESOLVED_TERMINAL_TREATMENT
                and raw.get("assigned_exactly_once") is True
                and raw.get("performance_related") is False
                and raw.get("embedding_status") == _UNRESOLVED_EMBEDDING_STATUS
                and raw.get("observed_delisting_return") is None
                and raw.get("effective_delisting_return") is None
                and raw.get("imputation_applied") is False
                and raw.get("successor_permno") is None
                and raw.get("successor_permco") is None
                and raw.get("next_date") is None
                and raw.get("next_price") is None,
                "unresolved terminal action escaped its audit-only state",
            )
            signal_date = _as_date(
                raw.get("signal_date"),
                label="unresolved action signal date",
            )
            assigned_date = _as_date(
                raw.get("assigned_return_date"),
                label="unresolved action assigned date",
            )
            delisting_date = _as_date(
                raw.get("delisting_date"),
                label="unresolved action delisting date",
            )
            holding_month = raw.get("holding_month")
            permno = raw.get("permno")
            _require(
                assigned_date == delisting_date
                and isinstance(holding_month, str)
                and bool(holding_month)
                and isinstance(permno, int),
                "unresolved terminal action identity differs",
            )
            key = (
                signal_date,
                holding_month,
                int(permno),
                assigned_date,
            )
            _require(
                key not in self._unresolved_action_keys,
                "unresolved terminal action was projected more than once",
            )
            self._unresolved_action_keys.add(key)
            payload = {
                name: _canonical_value(raw.get(name))
                for name in observed_columns
            }
            self._unresolved_action_source_hashes.add(
                hashlib.sha256(_canonical_json(payload)).hexdigest()
            )

    def record_action_projection(
        self,
        *,
        input_rows: int,
        unresolved_rows: int,
        mapped_actions: int,
    ) -> None:
        self.action_projection_calls += 1
        _require(
            mapped_actions == input_rows - unresolved_rows,
            "resolved corporate-action projection changed row cardinality",
        )
        self.resolved_action_rows_mapped += mapped_actions

    def require_unresolved_terminal_pairing(
        self,
        *,
        expected_count: int,
        expected_key_sha256: str | None,
    ) -> None:
        """Require one exact nullable-return twin for every audit-only action."""

        _require(
            expected_count >= 0
            and len(self._unresolved_action_keys) == expected_count
            and self._unresolved_action_keys == self._unresolved_return_keys
            and len(self._unresolved_action_source_hashes) == expected_count
            and self.unresolved_actions_mapped == 0,
            "unresolved terminal action/return audit pairing differs",
        )
        if expected_key_sha256 is not None:
            _require(
                _key_sha256(self._unresolved_action_keys)
                == expected_key_sha256,
                "unresolved terminal audit key digest differs",
            )

    def require_no_mark_sources(
        self,
        *,
        expected_count: int,
        expected_key_sha256: str | None,
    ) -> None:
        """Authenticate the frozen unique no-mark source domain."""

        source_hashes = set(self._no_mark_source_by_key.values())
        _require(
            expected_count >= 0
            and len(self._no_mark_source_by_key) == expected_count
            and len(source_hashes) == expected_count,
            "no-mark source count or source-record uniqueness differs",
        )
        if expected_key_sha256 is not None:
            _require(
                _no_mark_key_sha256(set(self._no_mark_source_by_key))
                == expected_key_sha256,
                "no-mark source key digest differs",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "projection_calls": self.calls,
            "action_projection_calls": self.action_projection_calls,
            "resolved_action_rows_mapped_canonically": (
                self.resolved_action_rows_mapped
            ),
            "unique_null_audit_rows_omitted_from_sparse_projection": len(
                self._row_keys
            ),
            "unique_unresolved_terminal_actions_retained_as_audit_only": len(
                self._unresolved_action_keys
            ),
            "unique_unresolved_terminal_return_twins": len(
                self._unresolved_return_keys
            ),
            "unresolved_terminal_key_sha256": _key_sha256(
                self._unresolved_action_keys
            ),
            "unresolved_terminal_source_row_count": len(
                self._unresolved_action_source_hashes
            ),
            "unresolved_terminal_actions_mapped_or_settled": (
                self.unresolved_actions_mapped
            ),
            "unique_exact_no_mark_source_rows": len(
                self._no_mark_source_by_key
            ),
            "unique_exact_no_mark_source_record_sha256": len(
                set(self._no_mark_source_by_key.values())
            ),
            "exact_no_mark_source_key_sha256": _no_mark_key_sha256(
                set(self._no_mark_source_by_key)
            ),
            "projected_exact_no_mark_rows_including_strategy_repeats": (
                self.projected_no_mark_rows
            ),
            "authoritative_nulls_converted_to_zero": self.zero_imputations,
            "by_authoritative_return_state": dict(sorted(self._state_counts.items())),
            "by_return_leg_class": dict(sorted(self._leg_counts.items())),
            "open_nonzero_position_without_return_policy": (
                "exact_missing_return_is_explicit_no_mark; "
                "all_other_nulls_fail_closed_by_dated_daily_engine"
            ),
        }


def project_resolved_corporate_actions(
    table: object,
    *,
    authority: CorporateActionMappingAuthority,
    audit: MissingReturnProjectionAudit,
) -> tuple[CorporateAction, ...]:
    """Map resolved rows unchanged and retain unresolved rows as audit-only."""

    _require(
        isinstance(table, pa.Table)
        and isinstance(audit, MissingReturnProjectionAudit),
        "corporate-action projection inputs differ",
    )
    source = table.combine_chunks()
    rows = tuple(source.to_pylist())
    unresolved_mask = tuple(
        raw.get("treatment_status") == UNRESOLVED_TERMINAL_TREATMENT
        for raw in rows
    )
    unresolved = tuple(
        raw
        for raw, is_unresolved in zip(rows, unresolved_mask, strict=True)
        if is_unresolved
    )
    if unresolved:
        audit.record_unresolved_actions(
            unresolved,
            observed_columns=tuple(source.column_names),
        )
    resolved = source.filter(
        pa.array(
            [not is_unresolved for is_unresolved in unresolved_mask],
            type=pa.bool_(),
        )
    )
    actions = _CANONICAL_ACTION_MAPPER(resolved, authority=authority)
    audit.record_action_projection(
        input_rows=source.num_rows,
        unresolved_rows=len(unresolved),
        mapped_actions=len(actions),
    )
    return actions


def project_nullable_daily_inputs(
    *,
    problem: MonthlyProblemBuild,
    target_weights: np.ndarray,
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    return_dates: tuple[dt.date, ...],
    source_sha256: str,
    audit: MissingReturnProjectionAudit,
) -> tuple[DatedMarketCloseInput, ...]:
    """Project exact missing rows as no-marks and keep other nulls sparse."""

    _require(
        isinstance(returns, pd.DataFrame)
        and isinstance(factors, pd.DataFrame)
        and isinstance(audit, MissingReturnProjectionAudit),
        "daily projection inputs differ",
    )
    required = {
        "signal_date",
        "holding_month",
        "permno",
        "calendar_date",
        "authoritative_total_return",
        "authoritative_return_state",
        "return_leg_class",
        "return_span_start_date",
        "return_span_end_date",
        "covered_trading_sessions",
        "span_alignment_status",
        "delisting_event_count",
        "delisting_treatment_status",
        "delisting_adjustment_return",
        "return_history_usable",
        "return_history_authorized",
    }
    _require(
        required.issubset(returns.columns),
        "daily return projection metadata is incomplete",
    )
    normalized = _normalize_return_metadata(returns)
    _require(
        not normalized.duplicated(
            ["signal_date", "holding_month", "permno", "calendar_date"],
            keep=False,
        ).any(),
        "daily return projection identity is duplicated",
    )
    raw = normalized["authoritative_total_return"]
    numeric = pd.to_numeric(raw, errors="coerce").to_numpy(float)
    raw_present = raw.notna().to_numpy(bool)
    finite = np.isfinite(numeric)
    usable = normalized["return_history_usable"].astype(bool).to_numpy()
    _require(
        not np.any(raw_present & (~finite | (numeric < -1.0))),
        "daily authoritative return contains an invalid nonnull numeric value",
    )
    _require(
        np.array_equal(raw_present, usable),
        "authoritative return presence differs from frozen usability",
    )
    null_mask = ~raw_present
    null_rows = normalized.loc[null_mask].copy()
    no_mark_rows = normalized.loc[_exact_no_mark_mask(normalized)].copy()
    if not null_rows.empty:
        states = set(null_rows["authoritative_return_state"].astype(str))
        legs = set(null_rows["return_leg_class"].astype(str))
        _require(
            states.issubset(_NULL_STATES)
            and legs.issubset(_NULL_LEG_CLASSES)
            and not null_rows["return_history_usable"].astype(bool).any()
            and not null_rows["return_history_authorized"].astype(bool).any(),
            "nullable daily row escaped its authenticated audit-only state",
        )
        audit.record(null_rows)
    if not no_mark_rows.empty:
        audit.record_no_mark_sources(no_mark_rows)
    projected = normalized.loc[~null_mask].copy()
    _require(
        len(projected) + len(null_rows) == len(normalized)
        and projected["authoritative_total_return"].notna().all()
        and all(
            math.isfinite(value) and value >= -1.0
            for value in pd.to_numeric(
                projected["authoritative_total_return"], errors="raise"
            ).to_numpy(float)
        ),
        "sparse daily projection changed a numeric return",
    )

    try:
        problem_permno = np.asarray(problem.problem.permno, dtype=np.int64)
        problem_borrow = np.asarray(
            problem.problem.borrow_rate_annual,
            dtype=np.float64,
        )
        weights = np.asarray(target_weights, dtype=np.float64)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise Stage11MissingReturnProjectionError(
            "daily no-mark problem vectors are invalid"
        ) from exc
    _require(
        problem_permno.ndim == problem_borrow.ndim == weights.ndim == 1
        and len(problem_permno) == len(problem_borrow) == len(weights)
        and np.all(problem_permno > 0)
        and len(np.unique(problem_permno)) == len(problem_permno)
        and np.isfinite(problem_borrow).all()
        and np.all(problem_borrow >= 0.0)
        and np.isfinite(weights).all(),
        "daily no-mark problem vectors differ",
    )
    economic = {
        int(permno)
        for permno, weight in zip(problem_permno, weights, strict=True)
        if float(weight) != 0.0
    }
    borrow_by_permno = {
        int(permno): float(rate)
        for permno, rate in zip(
            problem_permno,
            problem_borrow,
            strict=True,
        )
    }

    observed_span_start: dict[tuple[dt.date, int], dt.date] = {}
    observed_economic = projected.loc[projected["permno"].isin(economic)]
    for row in observed_economic.itertuples(index=False):
        calendar_date = _as_date(
            row.calendar_date,
            label="observed return calendar date",
        )
        span_start = _as_date(
            row.return_span_start_date,
            label="observed return span start date",
        )
        _require(
            span_start < calendar_date,
            "observed return span start must precede its calendar date",
        )
        observed_span_start[(calendar_date, int(row.permno))] = span_start

    no_mark_by_date: dict[dt.date, list[object]] = {}
    economic_no_mark = no_mark_rows.loc[no_mark_rows["permno"].isin(economic)]
    for row in economic_no_mark.itertuples(index=False):
        calendar_date = _as_date(
            row.calendar_date,
            label="no-mark calendar date",
        )
        no_mark_by_date.setdefault(calendar_date, []).append(row)
    audit.record_no_mark_projection(len(economic_no_mark))

    canonical_days = _CANONICAL_DAILY_INPUTS(
        problem=problem,
        target_weights=weights,
        returns=projected,
        factors=factors,
        return_dates=return_dates,
        source_sha256=source_sha256,
    )
    result: list[DatedMarketCloseInput] = []
    for day in canonical_days:
        observed_starts = tuple(
            observed_span_start[(day.trading_date, int(permno))]
            for permno in day.permno
        )
        no_mark_source = sorted(
            no_mark_by_date.get(day.trading_date, ()),
            key=lambda row: int(getattr(row, "permno")),
        )
        no_mark_permno = np.asarray(
            [int(getattr(row, "permno")) for row in no_mark_source],
            dtype=np.int64,
        )
        _require(
            set(int(value) for value in day.permno).isdisjoint(
                int(value) for value in no_mark_permno
            ),
            "observed and no-mark daily identities overlap",
        )
        result.append(
            DatedMarketCloseInput(
                trading_date=day.trading_date,
                permno=day.permno,
                total_return=day.total_return,
                borrow_rate_annual=day.borrow_rate_annual,
                daily_risk_free_rate=day.daily_risk_free_rate,
                source_payload_sha256=day.source_payload_sha256,
                observed_return_span_start_date=observed_starts,
                no_mark_permno=no_mark_permno,
                no_mark_borrow_rate_annual=np.asarray(
                    [
                        borrow_by_permno[int(value)]
                        for value in no_mark_permno
                    ],
                    dtype=np.float64,
                ),
                no_mark_source_record_sha256=tuple(
                    _no_mark_source_record_sha256(row)
                    for row in no_mark_source
                ),
            )
        )
    return tuple(result)


@contextmanager
def stage11_missing_return_projection_scope(
    *,
    expected_unresolved_count: int = _EXPECTED_UNRESOLVED_COUNT,
    expected_unresolved_key_sha256: str | None = (
        _EXPECTED_UNRESOLVED_KEY_SHA256
    ),
    expected_no_mark_source_count: int = _EXPECTED_NO_MARK_SOURCE_COUNT,
    expected_no_mark_key_sha256: str | None = (
        _EXPECTED_NO_MARK_KEY_SHA256
    ),
) -> Iterator[
    MissingReturnProjectionAudit
]:
    """Install, pair, and restore the narrow Stage 11 projection adapters."""

    original_daily = portfolio_backend_module._daily_inputs
    original_actions = (
        portfolio_backend_module.map_delisting_rows_to_corporate_actions
    )
    _require(
        original_daily is _CANONICAL_DAILY_INPUTS
        and original_actions is _CANONICAL_ACTION_MAPPER,
        "portfolio projection binding changed before Stage 11 projection",
    )
    audit = MissingReturnProjectionAudit()

    def daily_adapter(**kwargs: object) -> tuple[DatedMarketCloseInput, ...]:
        return project_nullable_daily_inputs(
            problem=kwargs["problem"],
            target_weights=kwargs["target_weights"],
            returns=kwargs["returns"],
            factors=kwargs["factors"],
            return_dates=kwargs["return_dates"],
            source_sha256=kwargs["source_sha256"],
            audit=audit,
        )

    def action_adapter(
        table: object,
        *,
        authority: CorporateActionMappingAuthority,
    ) -> tuple[CorporateAction, ...]:
        return project_resolved_corporate_actions(
            table,
            authority=authority,
            audit=audit,
        )

    portfolio_backend_module._daily_inputs = daily_adapter
    portfolio_backend_module.map_delisting_rows_to_corporate_actions = (
        action_adapter
    )
    try:
        yield audit
        audit.require_unresolved_terminal_pairing(
            expected_count=expected_unresolved_count,
            expected_key_sha256=expected_unresolved_key_sha256,
        )
        audit.require_no_mark_sources(
            expected_count=expected_no_mark_source_count,
            expected_key_sha256=expected_no_mark_key_sha256,
        )
    finally:
        changed = (
            portfolio_backend_module._daily_inputs is not daily_adapter
            or portfolio_backend_module.map_delisting_rows_to_corporate_actions
            is not action_adapter
        )
        portfolio_backend_module._daily_inputs = _CANONICAL_DAILY_INPUTS
        portfolio_backend_module.map_delisting_rows_to_corporate_actions = (
            _CANONICAL_ACTION_MAPPER
        )
        _require(
            not changed,
            "portfolio projection binding changed during Stage 11 projection",
        )


__all__ = [
    "MissingReturnProjectionAudit",
    "Stage11MissingReturnProjectionError",
    "project_nullable_daily_inputs",
    "project_resolved_corporate_actions",
    "stage11_missing_return_projection_scope",
]
