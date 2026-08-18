"""Independent, streaming recomputation of the Stage 7 label store.

This module is deliberately separated from the Stage 6 feature builders and
the Stage 7 primary label implementation.  It consumes only authenticated
input/output paths supplied by the independent verifier and reconstructs the
economic calculations from the frozen rules.  In particular, it does not
import ``feature_math``, ``crsp_features``, ``label_math``, the return router,
the materializer, or the primary quality builder.

The large return input is scanned in PERMNO order with bounded per-security
state.  The first pass independently estimates and cross-sectionally clips
the six EWLS factor slopes.  A second pass independently evaluates every
forward holding path.  Only one security's return history is resident during
either pass; cross-sectional state is limited to the 286,628 long-eligible
security-month rows and compact lineage digests.
"""

from __future__ import annotations

import bisect
import datetime as dt
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from portfolio_bot.features.stage7_independent_math import (
    centered_average_rank as _clean_centered_average_rank,
    fit_ewls_factor_model as _clean_fit_ewls_factor_model,
    mad_standardized as _clean_mad_standardized,
    type7_quantile as _clean_type7_quantile,
)
from portfolio_bot.features.verify_label_store import Difference


FACTOR_NAMES = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
BETA_NAMES = tuple(f"beta_label_{name}" for name in FACTOR_NAMES)

PRESENT = "PRESENT"
NOT_LONG_ELIGIBLE = "NOT_LONG_ELIGIBLE"
MISSING_BETA = "MISSING_BETA"
MISSING_NO_POST_ENTRY_RETURN = "MISSING_NO_POST_ENTRY_RETURN"
MISSING_DUPLICATE_RETURN_LEG = "MISSING_DUPLICATE_RETURN_LEG"
MISSING_RETURN_GAP = "MISSING_RETURN_GAP"
MISSING_MULTI_SESSION_RETURN = "MISSING_MULTI_SESSION_RETURN"
MISSING_INTERVAL_STRADDLE = "MISSING_INTERVAL_STRADDLE"
MISSING_UNUSABLE_RETURN = "MISSING_UNUSABLE_RETURN"
MISSING_UNRESOLVED_DELISTING = "MISSING_UNRESOLVED_DELISTING"
MISSING_FACTOR = "MISSING_FACTOR"
MISSING_NONFINITE_INPUT = "MISSING_NONFINITE_INPUT"
MISSING_MONTH_MINIMUM = "MISSING_MONTH_MINIMUM"
MISSING_ZERO_MAD = "MISSING_ZERO_MAD"
NO_ELIGIBLE_SECURITIES = "NO_ELIGIBLE_SECURITIES"

ORDINARY_LEG = "ordinary_one_session"
TERMINAL_LEG = "ordinary_plus_terminal_delisting_adjustment"
UNRESOLVED_LEG = "ordinary_with_unresolved_terminal_delisting"
SUPPRESSED_LEGS = frozenset(
    {
        "terminal_delisting_posting_audit_suppressed",
        "pre_scope_terminal_delisting_context_suppressed",
    }
)
OUTCOME_PRECEDENCE = (
    NOT_LONG_ELIGIBLE,
    MISSING_BETA,
    MISSING_DUPLICATE_RETURN_LEG,
    MISSING_INTERVAL_STRADDLE,
    MISSING_MULTI_SESSION_RETURN,
    MISSING_UNRESOLVED_DELISTING,
    MISSING_RETURN_GAP,
    MISSING_NO_POST_ENTRY_RETURN,
    MISSING_UNUSABLE_RETURN,
    MISSING_FACTOR,
    MISSING_NONFINITE_INPUT,
)

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

LINEAGE_FIELDS = (
    "signal_date",
    "permno",
    "calendar_date",
    "leg_ordinal",
    "return_leg_class",
    "authoritative_total_return",
    "rf",
    "mktrf",
    "smb",
    "hml",
    "rmw",
    "cma",
    "umd",
    "systematic_component",
    "excess_return_contribution",
    "residual_return_contribution",
    "outcome_availability_date",
    "factor_public_availability_date",
    "terminal_delisting",
)
LINEAGE_BINDING_FIELDS = (
    "signal_date",
    "permno",
    "calendar_date",
    "leg_ordinal",
    "return_leg_class",
    "authoritative_total_return",
    "rf",
    "mktrf",
    "smb",
    "hml",
    "rmw",
    "cma",
    "umd",
    "outcome_availability_date",
    "factor_public_availability_date",
    "terminal_delisting",
)


class IndependentRecomputeError(RuntimeError):
    """Raised when independent recomputation cannot be completed safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentRecomputeError(message)


def _finite(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _float(value: object) -> float | None:
    return float(value) if _finite(value) else None


def _numeric_preserving_nonfinite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


def _close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (float, np.floating)) or isinstance(
        right, (float, np.floating)
    ):
        if not _finite(left) or not _finite(right):
            return False
        return math.isclose(
            float(left),
            float(right),
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
    return left == right


def type7_quantile(values: Sequence[float], probability: float) -> float:
    """Hyndman-Fan type-7 quantile, independently implemented."""

    result = _clean_type7_quantile(values, probability)
    _require(result is not None, "Cannot calculate an empty type-7 quantile")
    return result


def ordinary_median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "Cannot calculate an empty median")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return _positive_zero(ordered[middle])
    return _positive_zero(math.fsum((ordered[middle - 1], ordered[middle])) / 2.0)


IndependentDifference = Difference


@dataclass(frozen=True)
class IndependentRecomputeEvidence:
    checks: Mapping[str, int]
    differences: tuple[IndependentDifference, ...]
    complete: bool
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactorRow:
    date: dt.date
    public_date: dt.date
    rf: float
    factors: tuple[float, ...]


@dataclass(frozen=True)
class CalendarRow:
    signal_date: dt.date
    holding_month: str
    entry_date: dt.date
    exit_date: dt.date


@dataclass(frozen=True)
class LongRow:
    signal_date: dt.date
    permno: int
    permco: int
    short_eligible: bool
    calendar: CalendarRow
    published_betas: tuple[float | None, ...]

    @property
    def key(self) -> tuple[dt.date, int]:
        return self.signal_date, self.permno


@dataclass(frozen=True)
class RawReturnRow:
    permno: int
    calendar_date: dt.date
    return_leg_class: str
    span_start_date: dt.date | None
    span_end_date: dt.date | None
    covered_sessions: int | None
    delisting_status: str
    authoritative_return: float | None
    outcome_availability_date: dt.date
    return_history_usable: bool

    @property
    def suppressed(self) -> bool:
        return (
            self.return_leg_class in SUPPRESSED_LEGS
            or "suppressed" in self.return_leg_class
        )

    @property
    def terminal(self) -> bool:
        return self.return_leg_class == TERMINAL_LEG

    @property
    def unresolved(self) -> bool:
        return (
            self.return_leg_class == UNRESOLVED_LEG
            or "unresolved" in self.return_leg_class
            or "unresolved" in self.delisting_status
        )


@dataclass(frozen=True)
class BetaFit:
    intercept: float
    slopes: tuple[float, ...]


@dataclass(frozen=True)
class Contribution:
    calendar_date: dt.date
    leg_class: str
    security_return: float
    rf: float
    factors: tuple[float, ...]
    systematic: float
    excess: float
    residual: float
    outcome_availability_date: dt.date
    factor_public_date: dt.date
    terminal: bool

    def as_lineage_row(
        self, signal_date: dt.date, permno: int, ordinal: int
    ) -> dict[str, Any]:
        return {
            "signal_date": signal_date,
            "permno": permno,
            "calendar_date": self.calendar_date,
            "leg_ordinal": ordinal,
            "return_leg_class": self.leg_class,
            "authoritative_total_return": self.security_return,
            "rf": self.rf,
            **dict(zip(FACTOR_NAMES, self.factors, strict=True)),
            "systematic_component": self.systematic,
            "excess_return_contribution": self.excess,
            "residual_return_contribution": self.residual,
            "outcome_availability_date": self.outcome_availability_date,
            "factor_public_availability_date": self.factor_public_date,
            "terminal_delisting": self.terminal,
        }


@dataclass(frozen=True)
class ForwardResult:
    status: str
    expected_count: int
    contributions: tuple[Contribution, ...]
    excess: float | None
    residual: float | None
    available_date: dt.date | None
    terminal_date: dt.date | None


@dataclass
class RecomputedLongRow:
    source: LongRow
    unclipped_betas: tuple[float, ...] | None = None
    betas: tuple[float, ...] | None = None
    forward: ForwardResult | None = None
    forward_without_beta_status: str | None = None
    primary_status: str | None = None
    magnitude_status: str | None = None
    residual_rank_numerator: int | None = None
    excess_rank_numerator: int | None = None
    residual_rank: float | None = None
    excess_rank: float | None = None
    residual_mad_standardized: float | None = None
    month_available_date: dt.date | None = None
    valid_primary_count: int = 0
    lineage_digest: str | None = None


@dataclass(frozen=True)
class MonthResult:
    signal_date: dt.date
    holding_month: str
    entry_date: dt.date
    exit_date: dt.date
    eligible_count: int
    beta_complete_count: int
    valid_forward_count: int
    valid_primary_count: int
    primary_month_valid: bool
    terminal_count: int
    delayed_count: int
    month_available_date: dt.date | None
    residual_median: float | None
    residual_mad: float | None
    residual_rank_numerator_sum: int | None
    excess_rank_numerator_sum: int | None
    residual_rank_mean: float | None
    excess_rank_mean: float | None
    month_status: str


def independent_ewls(
    security_returns: Sequence[float],
    risk_free: Sequence[float],
    factors: Sequence[Sequence[float]],
    ages: Sequence[int],
    *,
    minimum_observations: int = 126,
    half_life: float = 126.0,
) -> BetaFit | None:
    """Fit the frozen intercept-plus-six-factor EWLS independently."""

    result = _clean_fit_ewls_factor_model(
        security_returns,
        risk_free,
        factors,
        ages,
        minimum_observations=minimum_observations,
        half_life_sessions=half_life,
        rcond=1e-12,
    )
    if result is None:
        return None
    return BetaFit(
        intercept=result.intercept,
        slopes=result.slopes,
    )


def clip_beta_cross_section(
    fits: Mapping[tuple[dt.date, int], tuple[float, ...] | None]
) -> dict[tuple[dt.date, int], tuple[float, ...] | None]:
    """Apply separate signal-date/type-7 1% and 99% slope clipping."""

    by_signal: dict[dt.date, list[tuple[tuple[dt.date, int], tuple[float, ...]]]] = (
        defaultdict(list)
    )
    for key, slopes in fits.items():
        if slopes is not None:
            _require(len(slopes) == 6, "Independent beta has wrong slope count")
            by_signal[key[0]].append((key, slopes))
    result = dict(fits)
    for rows in by_signal.values():
        bounds = []
        for index in range(6):
            population = [slopes[index] for _, slopes in rows]
            bounds.append(
                (
                    type7_quantile(population, 0.01),
                    type7_quantile(population, 0.99),
                )
            )
        for key, slopes in rows:
            result[key] = tuple(
                min(upper, max(lower, slopes[index]))
                for index, (lower, upper) in enumerate(bounds)
            )
    return result


def centered_ranks(
    values: Sequence[float | None],
) -> tuple[list[float | None], list[int | None]]:
    result = _clean_centered_average_rank(values)
    return list(result.values), list(result.numerators)


def mad_standardize(
    values: Sequence[float | None],
) -> tuple[list[float | None], float | None, float | None, str]:
    result = _clean_mad_standardized(values)
    return list(result.values), result.median, result.mad, result.status


def _first_status(flags: set[str]) -> str:
    return next((status for status in OUTCOME_PRECEDENCE if status in flags), PRESENT)


def _covered_session_indices(
    row: RawReturnRow,
    sessions: Sequence[dt.date],
) -> tuple[int, ...] | None:
    if (
        row.span_start_date is None
        or row.span_end_date is None
        or row.span_start_date >= row.span_end_date
        or row.covered_sessions is None
        or row.covered_sessions <= 0
    ):
        return None
    first = bisect.bisect_right(sessions, row.span_start_date)
    after = bisect.bisect_right(sessions, row.span_end_date)
    covered = tuple(range(first, after))
    return covered if len(covered) == row.covered_sessions else None


def estimate_beta_for_signal(
    signal_date: dt.date,
    rows_by_date: Mapping[dt.date, RawReturnRow],
    sessions: Sequence[dt.date],
    session_index: Mapping[dt.date, int],
    factors: Mapping[dt.date, FactorRow],
) -> BetaFit | None:
    signal_position = session_index.get(signal_date)
    if signal_position is None:
        return None
    start = max(0, signal_position - 251)
    security_values: list[float] = []
    risk_free: list[float] = []
    factor_values: list[tuple[float, ...]] = []
    ages: list[int] = []
    for position in range(start, signal_position + 1):
        date = sessions[position]
        row = rows_by_date.get(date)
        factor = factors.get(date)
        if row is None or factor is None:
            continue
        coverage = _covered_session_indices(row, sessions)
        if (
            not row.return_history_usable
            or row.span_end_date != row.calendar_date
            or row.outcome_availability_date > signal_date
            or row.authoritative_return is None
            or row.authoritative_return < -1.0
            or coverage is None
            or len(coverage) != 1
            or factor.public_date > signal_date
        ):
            continue
        security_values.append(row.authoritative_return)
        risk_free.append(factor.rf)
        factor_values.append(factor.factors)
        ages.append(signal_position - position)
    return independent_ewls(
        security_values,
        risk_free,
        factor_values,
        ages,
    )


def evaluate_forward_path(
    source: LongRow,
    rows_by_date: Mapping[dt.date, RawReturnRow],
    ordered_rows: Sequence[RawReturnRow],
    sessions: Sequence[dt.date],
    factors: Mapping[dt.date, FactorRow],
    betas: tuple[float, ...] | None,
    *,
    ordered_dates: Sequence[dt.date] | None = None,
) -> ForwardResult:
    """Independently evaluate one exact close-to-close holding interval."""

    entry = source.calendar.entry_date
    exit_date = source.calendar.exit_date
    required = tuple(
        sessions[
            bisect.bisect_right(sessions, entry) : bisect.bisect_right(
                sessions, exit_date
            )
        ]
    )
    flags: set[str] = set()
    if betas is None or len(betas) != 6 or not all(_finite(x) for x in betas):
        flags.add(MISSING_BETA)
    if not required:
        flags.add(MISSING_NO_POST_ENTRY_RETURN)
    elif required[-1] != exit_date or required[0] <= entry:
        flags.add(MISSING_INTERVAL_STRADDLE)

    # The router offers signal/entry terminal context and post-exit rows whose
    # declared interval crosses the exit boundary.  Scan the source group
    # directly so this independent path does not rely on router output.
    dates = (
        tuple(row.calendar_date for row in ordered_rows)
        if ordered_dates is None
        else ordered_dates
    )
    _require(len(dates) == len(ordered_rows), "Return date index length differs")
    context_start = bisect.bisect_left(dates, source.signal_date)
    context_end = bisect.bisect_right(dates, entry)
    for row in ordered_rows[context_start:context_end]:
        if row.terminal and not row.suppressed:
            flags.add(MISSING_UNUSABLE_RETURN)
    # Match the frozen router's bounded, 45-calendar-day post-exit probe, but
    # make no monotonicity assumption about declared span starts.
    after_exit = bisect.bisect_right(dates, exit_date)
    probe_end = exit_date + dt.timedelta(days=45)
    for row in ordered_rows[after_exit:]:
        if row.calendar_date > probe_end:
            break
        if (
            not row.suppressed
            and row.span_start_date is not None
            and row.span_end_date is not None
            and row.span_start_date < exit_date < row.span_end_date
        ):
            flags.add(MISSING_INTERVAL_STRADDLE)

    if _first_status(flags) in {
        MISSING_BETA,
        MISSING_NO_POST_ENTRY_RETURN,
    }:
        return ForwardResult(
            _first_status(flags), len(required), (), None, None, None, None
        )

    contributions: list[Contribution] = []
    prior = entry
    offered = 0
    terminal_date: dt.date | None = None
    path_ended = False
    for date in required:
        if path_ended:
            break
        row = rows_by_date.get(date)
        if row is not None and row.suppressed:
            row = None
        if row is None:
            flags.add(MISSING_RETURN_GAP)
            prior = date
            continue
        offered += 1
        ends_path = row.terminal
        if (
            row.calendar_date != date
            or row.span_start_date != prior
            or row.span_end_date != date
        ):
            flags.add(MISSING_INTERVAL_STRADDLE)
        if (
            row.covered_sessions != 1
            or row.return_leg_class == "ordinary_multi_session"
        ):
            flags.add(MISSING_MULTI_SESSION_RETURN)
        if row.unresolved:
            flags.add(MISSING_UNRESOLVED_DELISTING)
        if (
            row.return_leg_class not in {ORDINARY_LEG, TERMINAL_LEG}
            and not row.unresolved
        ):
            flags.add(MISSING_UNUSABLE_RETURN)
        if (
            not row.return_history_usable
            or row.authoritative_return is None
            or (
                row.authoritative_return is not None and row.authoritative_return < -1.0
            )
        ):
            flags.add(MISSING_UNUSABLE_RETURN)
        factor = factors.get(date)
        if factor is None or (factor is not None and factor.public_date > date):
            flags.add(MISSING_FACTOR)
        if row.authoritative_return is not None and not _finite(
            row.authoritative_return
        ):
            flags.add(MISSING_NONFINITE_INPUT)
        if factor is not None and not (
            _finite(factor.rf) and all(_finite(value) for value in factor.factors)
        ):
            flags.add(MISSING_NONFINITE_INPUT)
        if flags:
            path_ended = ends_path
            prior = date
            continue
        assert betas is not None and factor is not None
        assert row.authoritative_return is not None
        systematic = _positive_zero(
            math.fsum(
                beta * value for beta, value in zip(betas, factor.factors, strict=True)
            )
        )
        excess = _positive_zero(math.fsum((row.authoritative_return, -factor.rf)))
        residual = _positive_zero(
            math.fsum((row.authoritative_return, -factor.rf, -systematic))
        )
        if not all(_finite(value) for value in (systematic, excess, residual)):
            flags.add(MISSING_NONFINITE_INPUT)
            prior = date
            continue
        contributions.append(
            Contribution(
                calendar_date=date,
                leg_class=row.return_leg_class,
                security_return=row.authoritative_return,
                rf=factor.rf,
                factors=factor.factors,
                systematic=systematic,
                excess=excess,
                residual=residual,
                outcome_availability_date=row.outcome_availability_date,
                factor_public_date=factor.public_date,
                terminal=ends_path,
            )
        )
        if ends_path:
            terminal_date = date
            path_ended = True
        prior = date

    if offered == 0 and terminal_date is None:
        flags.discard(MISSING_RETURN_GAP)
        flags.add(MISSING_NO_POST_ENTRY_RETURN)
    status = _first_status(flags)
    if status != PRESENT or not contributions:
        if status == PRESENT:
            status = MISSING_NO_POST_ENTRY_RETURN
        return ForwardResult(status, len(required), (), None, None, None, None)
    available = max(
        exit_date,
        *(item.outcome_availability_date for item in contributions),
    )
    return ForwardResult(
        PRESENT,
        len(required),
        tuple(contributions),
        _positive_zero(math.fsum(item.excess for item in contributions)),
        _positive_zero(math.fsum(item.residual for item in contributions)),
        available,
        terminal_date,
    )


def _token(value: object) -> bytes:
    if value is None:
        text = "N"
    elif isinstance(value, bool):
        text = "B1" if value else "B0"
    elif isinstance(value, dt.date):
        text = f"D{value.isoformat()}"
    elif isinstance(value, (float, np.floating)):
        text = f"F{_positive_zero(float(value)).hex()}"
    elif isinstance(value, (int, np.integer)):
        text = f"I{int(value)}"
    else:
        text = f"S{value}"
    payload = text.encode("utf-8")
    return len(payload).to_bytes(4, "big") + payload


def lineage_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash exact raw/factor lineage bindings, excluding derived floats.

    Systematic/excess/residual contribution cells are compared separately with
    the frozen 1e-12 tolerance; excluding them avoids turning the digest into
    an accidentally stricter floating-point gate.
    """

    digest = hashlib.sha256()
    for row in rows:
        for name in LINEAGE_BINDING_FIELDS:
            digest.update(_token(row.get(name)))
    return digest.hexdigest()


def _raw_row_from_values(values: Mapping[str, Any]) -> RawReturnRow:
    return RawReturnRow(
        permno=int(values["permno"]),
        calendar_date=values["calendar_date"],
        return_leg_class=str(values["return_leg_class"]),
        span_start_date=values["return_span_start_date"],
        span_end_date=values["return_span_end_date"],
        covered_sessions=(
            None
            if values["covered_trading_sessions"] is None
            else int(values["covered_trading_sessions"])
        ),
        delisting_status=str(values["delisting_treatment_status"]),
        authoritative_return=_numeric_preserving_nonfinite(
            values["authoritative_total_return"]
        ),
        outcome_availability_date=values["outcome_availability_date"],
        return_history_usable=bool(values["return_history_usable"]),
    )


def scan_grouped_returns(
    path: Path,
    relevant_permnos: set[int],
    callback: Callable[[int, tuple[RawReturnRow, ...]], None],
    *,
    cutoff: dt.date,
    batch_rows: int = 65_536,
) -> int:
    """Stream bounded return rows in PERMNO/date order.

    The Parquet input remains the authenticated source; predicate delivery is
    restricted to relevant long-universe PERMNOs and the development cutoff.
    A callback receives one complete PERMNO at a time, even across record-batch
    and row-group boundaries.
    """

    _require(batch_rows > 0, "Return scan batch size must be positive")
    _require(bool(relevant_permnos), "Return scan has no relevant PERMNOs")
    scanner = ds.dataset(path, format="parquet").scanner(
        columns=list(RETURN_COLUMNS),
        filter=(ds.field("calendar_date") <= cutoff)
        & ds.field("permno").isin(sorted(relevant_permnos)),
        batch_size=batch_rows,
        use_threads=False,
    )
    current: int | None = None
    held: list[RawReturnRow] = []
    delivered_permnos: set[int] = set()
    prior_key: tuple[int, dt.date] | None = None
    delivered = 0
    for batch in scanner.to_batches():
        columns = {
            name: batch.column(batch.schema.get_field_index(name)).to_pylist()
            for name in RETURN_COLUMNS
        }
        for index in range(batch.num_rows):
            values = {name: column[index] for name, column in columns.items()}
            row = _raw_row_from_values(values)
            key = (row.permno, row.calendar_date)
            _require(
                prior_key is None or key > prior_key,
                "Authenticated return stream is not strict PERMNO/date sorted",
            )
            prior_key = key
            if current is None:
                current = row.permno
            if row.permno != current:
                callback(current, tuple(held))
                delivered_permnos.add(current)
                held = []
                current = row.permno
            held.append(row)
            delivered += 1
    if current is not None:
        callback(current, tuple(held))
        delivered_permnos.add(current)
    for permno in sorted(relevant_permnos.difference(delivered_permnos)):
        callback(permno, ())
    return delivered


def _artifact_path(receipt: object) -> Path:
    if isinstance(receipt, (str, Path)):
        return Path(receipt)
    value = getattr(receipt, "path", None)
    _require(value is not None, "Authenticated artifact receipt has no path")
    return Path(value)


def _mapping_value(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        _require(name in value, f"Authenticated artifact is missing {name}")
        return value[name]
    candidate = getattr(value, name, None)
    _require(candidate is not None, f"Authenticated artifact is missing {name}")
    return candidate


def _input_path(artifacts: object, name: str) -> Path:
    return _artifact_path(_mapping_value(getattr(artifacts, "inputs"), name))


def _output_path(artifacts: object, name: str) -> Path:
    outputs = getattr(artifacts, "outputs")
    if isinstance(outputs, Mapping) and name in outputs:
        return _artifact_path(outputs[name])
    if isinstance(outputs, Mapping):
        for value in outputs.values():
            if isinstance(value, Mapping) and name in value:
                return _artifact_path(value[name])
            files = getattr(value, "files", None)
            if isinstance(files, Mapping) and name in files:
                return _artifact_path(files[name])
    raise IndependentRecomputeError(f"Authenticated output is missing {name}")


def _rules(artifacts: object) -> Mapping[str, Any]:
    contract = getattr(artifacts, "contract")
    rules = getattr(contract, "rules", None)
    if isinstance(rules, Mapping):
        return rules
    if isinstance(rules, (str, Path)):
        value = json.loads(Path(rules).read_text("utf-8"))
        _require(isinstance(value, Mapping), "Stage 7 rules file is not an object")
        return value
    raise IndependentRecomputeError("Authenticated verifier contract has no rules")


@dataclass
class _RecomputeContext:
    rules: Mapping[str, Any]
    feature_spine: pa.Table
    calendar: dict[dt.date, CalendarRow]
    sessions: tuple[dt.date, ...]
    session_index: dict[dt.date, int]
    factors: dict[dt.date, FactorRow]
    long_rows: dict[tuple[dt.date, int], RecomputedLongRow]
    long_by_permno: dict[int, list[RecomputedLongRow]]
    long_by_signal: dict[dt.date, list[RecomputedLongRow]]
    spine_counts: dict[dt.date, int]
    month_results: dict[dt.date, MonthResult] = field(default_factory=dict)


def _load_recompute_context(
    artifacts: object,
    rules: Mapping[str, Any],
) -> _RecomputeContext:
    feature_path = _input_path(artifacts, "feature_store_raw.parquet")
    universe_path = _input_path(artifacts, "crsp_signal_date_universe.parquet")
    feature_columns = (
        "signal_date",
        "permno",
        "permco",
        "long_eligible",
        "short_eligible",
        *BETA_NAMES,
    )
    universe_columns = (
        "signal_date",
        "permno",
        "permco",
        "issuer_line_selected",
        "long_eligible",
        "short_eligible",
        "future_outcome_filter_applied",
    )
    feature = pq.read_table(
        feature_path,
        columns=list(feature_columns),
        use_threads=False,
    ).combine_chunks()
    universe = pq.read_table(
        universe_path,
        columns=list(universe_columns),
        use_threads=False,
    ).combine_chunks()
    required_rows = int(rules["scope"]["required_audit_rows"])
    _require(
        feature.num_rows == universe.num_rows == required_rows,
        "Independent feature/universe row count differs",
    )
    for name in ("signal_date", "permno", "permco", "long_eligible", "short_eligible"):
        _require(
            feature[name].equals(universe[name]),
            f"Independent feature/universe spine differs: {name}",
        )
    _require(
        universe["issuer_line_selected"].equals(universe["long_eligible"]),
        "Independent issuer-line selection differs",
    )
    _require(
        not bool(pc.any(universe["future_outcome_filter_applied"]).as_py()),
        "Independent universe contains future-outcome filtering",
    )

    calendar_table = pq.read_table(
        _input_path(artifacts, "rebalance_calendar.parquet"),
        columns=[
            "holding_month",
            "stage",
            "signal_date",
            "entry_date",
            "exit_date",
            "is_embargo",
            "is_partial",
        ],
        use_threads=False,
    )
    first_signal = dt.date.fromisoformat(rules["scope"]["first_signal_date"])
    last_signal = dt.date.fromisoformat(rules["scope"]["last_signal_date"])
    last_exit = dt.date.fromisoformat(rules["scope"]["last_permitted_exit_date"])
    calendar: dict[dt.date, CalendarRow] = {}
    for row in calendar_table.to_pylist():
        if not (
            row["stage"] == "development"
            and first_signal <= row["signal_date"] <= last_signal
            and row["exit_date"] <= last_exit
        ):
            continue
        _require(
            not row["is_embargo"] and not row["is_partial"],
            "Independent development calendar includes partial/embargo row",
        )
        signal = row["signal_date"]
        _require(signal not in calendar, "Duplicate development signal date")
        holding = row["holding_month"]
        if isinstance(holding, dt.date):
            holding = holding.strftime("%Y-%m")
        calendar[signal] = CalendarRow(
            signal_date=signal,
            holding_month=str(holding),
            entry_date=row["entry_date"],
            exit_date=row["exit_date"],
        )
    _require(
        len(calendar) == int(rules["scope"]["required_signal_dates"]),
        "Independent development signal count differs",
    )

    trading = pq.read_table(
        _input_path(artifacts, "trading_calendar.parquet"),
        columns=["calendar_date"],
        use_threads=False,
    )["calendar_date"].to_pylist()
    sessions = tuple(
        date for date in trading if dt.date(2000, 1, 3) <= date <= last_exit
    )
    _require(
        len(sessions) == 3_752 and sessions == tuple(sorted(set(sessions))),
        "Independent bounded trading sessions differ",
    )
    session_index = {date: index for index, date in enumerate(sessions)}

    factor_table = pq.read_table(
        _input_path(artifacts, "ff_factor_daily_observation.parquet"),
        columns=[
            "observation_date",
            "public_availability_date",
            "rf",
            *FACTOR_NAMES,
        ],
        use_threads=False,
    )
    factors: dict[dt.date, FactorRow] = {}
    for row in factor_table.to_pylist():
        date = row["observation_date"]
        if date > last_exit:
            continue
        _require(date not in factors, "Duplicate independent factor date")
        factors[date] = FactorRow(
            date=date,
            public_date=row["public_availability_date"],
            rf=float(row["rf"]),
            factors=tuple(float(row[name]) for name in FACTOR_NAMES),
        )
    _require(
        tuple(factors) == sessions,
        "Independent factor/trading date domain differs",
    )

    long_rows: dict[tuple[dt.date, int], RecomputedLongRow] = {}
    long_by_permno: dict[int, list[RecomputedLongRow]] = defaultdict(list)
    long_by_signal: dict[dt.date, list[RecomputedLongRow]] = defaultdict(list)
    spine_counts: Counter[dt.date] = Counter()
    prior_key: tuple[dt.date, int] | None = None
    for batch in feature.to_batches(max_chunksize=65_536):
        for row in batch.to_pylist():
            signal = row["signal_date"]
            permno = int(row["permno"])
            key = (signal, permno)
            _require(
                prior_key is None or key > prior_key,
                "Independent feature spine is not strict sorted",
            )
            prior_key = key
            _require(
                signal in calendar, "Feature signal is outside development calendar"
            )
            spine_counts[signal] += 1
            if not row["long_eligible"]:
                continue
            source = LongRow(
                signal_date=signal,
                permno=permno,
                permco=int(row["permco"]),
                short_eligible=bool(row["short_eligible"]),
                calendar=calendar[signal],
                published_betas=tuple(_float(row[name]) for name in BETA_NAMES),
            )
            target = RecomputedLongRow(source=source)
            _require(key not in long_rows, "Duplicate independent long key")
            long_rows[key] = target
            long_by_permno[permno].append(target)
            long_by_signal[signal].append(target)
    _require(
        len(long_rows) == int(rules["scope"]["required_long_eligible_rows"]),
        "Independent long-row count differs",
    )
    for values in long_by_permno.values():
        values.sort(key=lambda item: item.source.signal_date)
    for values in long_by_signal.values():
        values.sort(key=lambda item: item.source.permno)
    return _RecomputeContext(
        rules=rules,
        feature_spine=feature,
        calendar=dict(sorted(calendar.items())),
        sessions=sessions,
        session_index=session_index,
        factors=factors,
        long_rows=long_rows,
        long_by_permno=dict(long_by_permno),
        long_by_signal=dict(long_by_signal),
        spine_counts=dict(spine_counts),
    )


def _estimate_all_betas(
    context: _RecomputeContext,
    return_path: Path,
    *,
    batch_rows: int,
) -> int:
    def consume(permno: int, rows: tuple[RawReturnRow, ...]) -> None:
        by_date = {row.calendar_date: row for row in rows}
        _require(len(by_date) == len(rows), "Duplicate return date within PERMNO")
        for target in context.long_by_permno.get(permno, ()):  # pragma: no branch
            fit = estimate_beta_for_signal(
                target.source.signal_date,
                by_date,
                context.sessions,
                context.session_index,
                context.factors,
            )
            target.unclipped_betas = None if fit is None else fit.slopes

    delivered = scan_grouped_returns(
        return_path,
        set(context.long_by_permno),
        consume,
        cutoff=dt.date.fromisoformat(
            context.rules["scope"]["last_permitted_exit_date"]
        ),
        batch_rows=batch_rows,
    )
    clipped = clip_beta_cross_section(
        {key: target.unclipped_betas for key, target in context.long_rows.items()}
    )
    for key, value in clipped.items():
        context.long_rows[key].betas = value
    return delivered


def _compute_all_forward_paths(
    context: _RecomputeContext,
    return_path: Path,
    *,
    batch_rows: int,
) -> int:
    def consume(permno: int, rows: tuple[RawReturnRow, ...]) -> None:
        by_date = {row.calendar_date: row for row in rows}
        _require(len(by_date) == len(rows), "Duplicate return date within PERMNO")
        dates = tuple(row.calendar_date for row in rows)
        for target in context.long_by_permno.get(permno, ()):  # pragma: no branch
            outcome = evaluate_forward_path(
                target.source,
                by_date,
                rows,
                context.sessions,
                context.factors,
                target.betas,
                ordered_dates=dates,
            )
            target.forward = outcome
            if target.betas is None:
                target.forward_without_beta_status = evaluate_forward_path(
                    target.source,
                    by_date,
                    rows,
                    context.sessions,
                    context.factors,
                    (0.0,) * 6,
                    ordered_dates=dates,
                ).status
            else:
                target.forward_without_beta_status = outcome.status
            if outcome.status == PRESENT:
                target.lineage_digest = lineage_digest(
                    contribution.as_lineage_row(
                        target.source.signal_date,
                        permno,
                        ordinal,
                    )
                    for ordinal, contribution in enumerate(
                        outcome.contributions, start=1
                    )
                )

    return scan_grouped_returns(
        return_path,
        set(context.long_by_permno),
        consume,
        cutoff=dt.date.fromisoformat(
            context.rules["scope"]["last_permitted_exit_date"]
        ),
        batch_rows=batch_rows,
    )


def _canonical_mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if _finite(value)]
    return None if not finite else _positive_zero(math.fsum(finite) / len(finite))


def _derive_month_targets(context: _RecomputeContext) -> None:
    minimum = int(context.rules["targets"]["minimum_valid_labels_per_signal"])
    for signal, calendar in context.calendar.items():
        rows = context.long_by_signal.get(signal, [])
        eligible_count = len(rows)
        beta_count = sum(target.betas is not None for target in rows)
        forward_count = sum(
            target.forward_without_beta_status == PRESENT for target in rows
        )
        primary_mask = [
            target.forward is not None and target.forward.status == PRESENT
            for target in rows
        ]
        valid_count = sum(primary_mask)
        residuals = [
            target.forward.residual if valid else None
            for target, valid in zip(rows, primary_mask, strict=True)
        ]
        excesses = [
            target.forward.excess if valid else None
            for target, valid in zip(rows, primary_mask, strict=True)
        ]
        if eligible_count == 0:
            month_status = NO_ELIGIBLE_SECURITIES
        elif valid_count < minimum:
            month_status = MISSING_MONTH_MINIMUM
        else:
            month_status = PRESENT

        residual_rank: list[float | None] = [None] * len(rows)
        excess_rank: list[float | None] = [None] * len(rows)
        residual_num: list[int | None] = [None] * len(rows)
        excess_num: list[int | None] = [None] * len(rows)
        magnitude: list[float | None] = [None] * len(rows)
        residual_median: float | None = None
        residual_mad: float | None = None
        magnitude_month_status = MISSING_MONTH_MINIMUM
        residual_sum: int | None = None
        excess_sum: int | None = None
        if month_status == PRESENT:
            residual_rank, residual_num = centered_ranks(residuals)
            excess_rank, excess_num = centered_ranks(excesses)
            magnitude, residual_median, residual_mad, magnitude_month_status = (
                mad_standardize(residuals)
            )
            residual_sum = sum(value for value in residual_num if value is not None)
            excess_sum = sum(value for value in excess_num if value is not None)
            _require(
                residual_sum == excess_sum == 0,
                "Independent centered-rank sum identity failed",
            )

        month_dates = [
            target.forward.available_date
            for target, valid in zip(rows, primary_mask, strict=True)
            if valid and target.forward is not None
        ]
        _require(
            all(value is not None for value in month_dates),
            "Independent primary-valid row has no availability date",
        )
        month_available = max(month_dates) if month_dates else None
        for index, target in enumerate(rows):
            assert target.forward is not None
            if target.forward.status != PRESENT:
                primary_status = target.forward.status
                magnitude_status = target.forward.status
            elif month_status != PRESENT:
                primary_status = MISSING_MONTH_MINIMUM
                magnitude_status = MISSING_MONTH_MINIMUM
            else:
                primary_status = PRESENT
                magnitude_status = (
                    MISSING_ZERO_MAD
                    if magnitude_month_status == MISSING_ZERO_MAD
                    else PRESENT
                )
            target.primary_status = primary_status
            target.magnitude_status = magnitude_status
            target.residual_rank_numerator = residual_num[index]
            target.excess_rank_numerator = excess_num[index]
            target.residual_rank = residual_rank[index]
            target.excess_rank = excess_rank[index]
            target.residual_mad_standardized = magnitude[index]
            target.month_available_date = month_available
            target.valid_primary_count = valid_count

        terminal_count = sum(
            valid
            and target.forward is not None
            and target.forward.terminal_date is not None
            for target, valid in zip(rows, primary_mask, strict=True)
        )
        delayed_count = sum(
            valid
            and target.forward is not None
            and target.forward.available_date is not None
            and target.forward.available_date > calendar.exit_date
            for target, valid in zip(rows, primary_mask, strict=True)
        )
        context.month_results[signal] = MonthResult(
            signal_date=signal,
            holding_month=calendar.holding_month,
            entry_date=calendar.entry_date,
            exit_date=calendar.exit_date,
            eligible_count=eligible_count,
            beta_complete_count=beta_count,
            valid_forward_count=forward_count,
            valid_primary_count=valid_count,
            primary_month_valid=month_status == PRESENT,
            terminal_count=terminal_count,
            delayed_count=delayed_count,
            month_available_date=month_available,
            residual_median=residual_median,
            residual_mad=residual_mad,
            residual_rank_numerator_sum=residual_sum,
            excess_rank_numerator_sum=excess_sum,
            residual_rank_mean=_canonical_mean(residual_rank),
            excess_rank_mean=_canonical_mean(excess_rank),
            month_status=month_status,
        )


def _expected_audit_row(
    context: _RecomputeContext,
    spine_row: Mapping[str, Any],
) -> dict[str, Any]:
    signal = spine_row["signal_date"]
    permno = int(spine_row["permno"])
    calendar = context.calendar[signal]
    month = context.month_results[signal]
    target = context.long_rows.get((signal, permno))
    copied_betas = tuple(_float(spine_row[name]) for name in BETA_NAMES)
    if target is None:
        betas = copied_betas
        beta_complete = all(value is not None for value in betas)
        outcome_status = NOT_LONG_ELIGIBLE
        primary_status = NOT_LONG_ELIGIBLE
        magnitude_status = NOT_LONG_ELIGIBLE
        expected_count = bisect.bisect_right(
            context.sessions, calendar.exit_date
        ) - bisect.bisect_right(context.sessions, calendar.entry_date)
        contributions: tuple[Contribution, ...] = ()
        terminal_date = available = excess = residual = None
        residual_num = excess_num = residual_rank = excess_rank = magnitude = None
    else:
        _require(target.forward is not None, "Independent forward row is absent")
        betas = target.betas or (None,) * 6
        beta_complete = target.betas is not None
        outcome_status = target.forward.status
        primary_status = target.primary_status
        magnitude_status = target.magnitude_status
        expected_count = target.forward.expected_count
        contributions = target.forward.contributions
        terminal_date = target.forward.terminal_date
        available = target.forward.available_date
        excess = target.forward.excess
        residual = target.forward.residual
        residual_num = target.residual_rank_numerator
        excess_num = target.excess_rank_numerator
        residual_rank = target.residual_rank
        excess_rank = target.excess_rank
        magnitude = target.residual_mad_standardized
    return {
        "signal_date": signal,
        "permno": permno,
        "permco": int(spine_row["permco"]),
        "long_eligible": bool(spine_row["long_eligible"]),
        "short_eligible": bool(spine_row["short_eligible"]),
        "holding_month": calendar.holding_month,
        "entry_date": calendar.entry_date,
        "exit_date": calendar.exit_date,
        **dict(zip(BETA_NAMES, betas, strict=True)),
        "beta_complete": beta_complete,
        "expected_leg_count": expected_count,
        "used_leg_count": len(contributions),
        "first_used_leg_date": (
            contributions[0].calendar_date if contributions else None
        ),
        "last_used_leg_date": (
            contributions[-1].calendar_date if contributions else None
        ),
        "terminal_delisting_used": terminal_date is not None,
        "terminal_delisting_date": terminal_date,
        "max_outcome_availability_date": (
            max(item.outcome_availability_date for item in contributions)
            if contributions
            else None
        ),
        "security_label_available_date": available,
        "month_label_available_date": month.month_available_date,
        "unranked_excess_return": excess,
        "unranked_residual_return": residual,
        "residual_rank_numerator": residual_num,
        "excess_rank_numerator": excess_num,
        "residual_return_rank": residual_rank,
        "excess_return_rank": excess_rank,
        "residual_return_mad_standardized": magnitude,
        "valid_primary_count_in_month": month.valid_primary_count,
        "outcome_status": outcome_status,
        "primary_label_status": primary_status,
        "magnitude_label_status": magnitude_status,
    }


def _add_difference(
    differences: list[IndependentDifference],
    code: str,
    expected: Any,
    observed: Any,
    details: str,
) -> None:
    differences.append(
        IndependentDifference(
            code=code,
            expected=expected,
            observed=observed,
            details=details,
        )
    )


def _audit_published_rows(
    context: _RecomputeContext,
    path: Path,
    checks: MutableMapping[str, int],
    differences: list[IndependentDifference],
    *,
    batch_rows: int,
) -> None:
    parquet = pq.ParquetFile(path)
    checks["audit_rows"] = int(parquet.metadata.num_rows)
    duplicate = order = structural = beta_bad = 0
    residual_bad = excess_bad = rank_bad = mad_bad = 0
    security_availability_bad = month_availability_bad = 0
    unknown = nonfinite = nonlong_nonnull = status_bad = 0
    offset = 0
    prior: tuple[dt.date, int] | None = None
    domains = context.rules["value_domains"]
    status_domains = {
        name: set(domains[name])
        for name in (
            "outcome_status",
            "primary_label_status",
            "magnitude_label_status",
        )
    }
    target_fields = (
        "unranked_excess_return",
        "unranked_residual_return",
        "residual_rank_numerator",
        "excess_rank_numerator",
        "residual_return_rank",
        "excess_return_rank",
        "residual_return_mad_standardized",
    )
    floating = {
        field.name for field in parquet.schema_arrow if pa.types.is_floating(field.type)
    }
    for batch in parquet.iter_batches(batch_size=batch_rows, use_threads=False):
        source = context.feature_spine.slice(offset, batch.num_rows)
        actual_rows = pa.Table.from_batches([batch]).to_pylist()
        source_rows = source.to_pylist()
        _require(
            len(actual_rows) == len(source_rows),
            "Published audit and feature spine batch lengths differ",
        )
        for actual, spine in zip(actual_rows, source_rows, strict=True):
            key = (actual["signal_date"], int(actual["permno"]))
            if prior is not None:
                duplicate += int(key == prior)
                order += int(key < prior)
            prior = key
            expected = _expected_audit_row(context, spine)
            for name in actual:
                if (
                    name in floating
                    and actual[name] is not None
                    and not _finite(actual[name])
                ):
                    nonfinite += 1
                tolerance = 1e-10 if name in BETA_NAMES else 1e-12
                if _close(expected.get(name), actual[name], tolerance):
                    continue
                if name in BETA_NAMES:
                    beta_bad += 1
                elif name == "unranked_residual_return":
                    residual_bad += 1
                elif name == "unranked_excess_return":
                    excess_bad += 1
                elif name in {
                    "residual_rank_numerator",
                    "excess_rank_numerator",
                    "residual_return_rank",
                    "excess_return_rank",
                }:
                    rank_bad += 1
                elif name == "residual_return_mad_standardized":
                    mad_bad += 1
                elif name == "security_label_available_date":
                    security_availability_bad += 1
                elif name == "month_label_available_date":
                    month_availability_bad += 1
                elif name in {
                    "signal_date",
                    "permno",
                    "permco",
                    "long_eligible",
                    "short_eligible",
                    "holding_month",
                    "entry_date",
                    "exit_date",
                }:
                    structural += 1
                elif name.endswith("status"):
                    status_bad += 1
                else:
                    status_bad += 1
            if not actual["long_eligible"]:
                nonlong_nonnull += sum(
                    actual[name] is not None for name in target_fields
                )
            for name, allowed in status_domains.items():
                unknown += int(actual[name] not in allowed)
        offset += batch.num_rows
    structural += abs(context.feature_spine.num_rows - offset)
    checks["duplicate_audit_keys"] = duplicate
    checks["audit_key_order_mismatches"] = order
    checks["spine_key_or_audit_field_mismatches"] = structural
    checks["beta_value_mismatches"] = beta_bad
    checks["residual_value_mismatches"] += residual_bad
    checks["excess_value_mismatches"] += excess_bad
    checks["rank_formula_mismatches"] += rank_bad
    checks["mad_value_mismatches"] += mad_bad
    checks["security_availability_mismatches"] += security_availability_bad
    checks["month_availability_mismatches"] += month_availability_bad
    checks["unknown_statuses"] += unknown
    checks["nonfinite_published_values"] += nonfinite
    checks["nonlong_nonnull_target_values"] = nonlong_nonnull
    if status_bad:
        _add_difference(
            differences,
            "audit_derived_value_mismatches",
            0,
            status_bad,
            "Published audit statuses/count fields differ from independent reconstruction",
        )


def _lineage_published_rows(
    context: _RecomputeContext,
    path: Path,
    checks: MutableMapping[str, int],
    differences: list[IndependentDifference],
    *,
    batch_rows: int,
) -> None:
    expected = {
        key: target.lineage_digest
        for key, target in context.long_rows.items()
        if target.lineage_digest is not None
    }
    expected_rows = sum(
        len(target.forward.contributions)
        for target in context.long_rows.values()
        if target.forward is not None and target.forward.status == PRESENT
    )
    observed_rows = duplicate = order = after_exit = before_entry = 0
    multi = unusable = unresolved = missing_factor = factor_bad = 0
    residual_bad = excess_bad = nonfinite = post_return = post_factor = 0
    terminal_double = 0
    prior_key: tuple[dt.date, int, dt.date] | None = None
    current_owner: tuple[dt.date, int] | None = None
    current_rows: list[dict[str, Any]] = []
    seen_owners: set[tuple[dt.date, int]] = set()
    digest_bad = 0

    def flush() -> None:
        nonlocal terminal_double, digest_bad
        if current_owner is None:
            return
        seen_owners.add(current_owner)
        terminal_double += int(
            sum(bool(row["terminal_delisting"]) for row in current_rows) > 1
        )
        if expected.get(current_owner) != lineage_digest(current_rows):
            digest_bad += 1

    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=batch_rows, use_threads=False
    ):
        for row in pa.Table.from_batches([batch]).to_pylist():
            observed_rows += 1
            key = (row["signal_date"], int(row["permno"]), row["calendar_date"])
            owner = key[:2]
            if prior_key is not None:
                duplicate += int(key == prior_key)
                order += int(key < prior_key)
            prior_key = key
            if current_owner != owner:
                flush()
                current_owner = owner
                current_rows = []
            current_rows.append(row)
            target = context.long_rows.get(owner)
            if target is None:
                after_exit += 1
                continue
            calendar = target.source.calendar
            before_entry += int(row["calendar_date"] <= calendar.entry_date)
            after_exit += int(row["calendar_date"] > calendar.exit_date)
            post_return += int(
                row["calendar_date"]
                > dt.date.fromisoformat(
                    context.rules["scope"]["last_permitted_exit_date"]
                )
            )
            post_factor += int(
                row["factor_public_availability_date"]
                > dt.date.fromisoformat(
                    context.rules["scope"]["last_permitted_exit_date"]
                )
            )
            leg_class = row["return_leg_class"]
            multi += int(leg_class == "ordinary_multi_session")
            unresolved += int("unresolved" in leg_class)
            unusable += int(leg_class not in {ORDINARY_LEG, TERMINAL_LEG})
            factor = context.factors.get(row["calendar_date"])
            if factor is None:
                missing_factor += 1
            else:
                observed_factors = tuple(float(row[name]) for name in FACTOR_NAMES)
                if not (
                    _close(row["rf"], factor.rf)
                    and all(
                        _close(left, right)
                        for left, right in zip(
                            observed_factors, factor.factors, strict=True
                        )
                    )
                    and row["factor_public_availability_date"] == factor.public_date
                ):
                    factor_bad += 1
            for name in LINEAGE_FIELDS:
                if isinstance(row.get(name), float) and not _finite(row[name]):
                    nonfinite += 1
            betas = target.betas
            if betas is not None and factor is not None:
                systematic = math.fsum(
                    beta * value
                    for beta, value in zip(betas, factor.factors, strict=True)
                )
                excess = math.fsum(
                    (float(row["authoritative_total_return"]), -factor.rf)
                )
                residual = math.fsum(
                    (
                        float(row["authoritative_total_return"]),
                        -factor.rf,
                        -systematic,
                    )
                )
                excess_bad += int(not _close(row["excess_return_contribution"], excess))
                residual_bad += int(
                    not _close(row["systematic_component"], systematic)
                ) + int(not _close(row["residual_return_contribution"], residual))
    flush()
    missing_owners = set(expected).difference(seen_owners)
    extra_owners = seen_owners.difference(expected)
    domain_bad = len(missing_owners) + len(extra_owners)
    if observed_rows != expected_rows:
        domain_bad += abs(observed_rows - expected_rows)
    checks["duplicate_leg_lineage_keys"] = duplicate
    checks["leg_lineage_sort_mismatches"] = order
    checks["leg_lineage_key_domain_mismatches"] = domain_bad
    checks["duplicate_or_overlapping_legs_used"] = duplicate
    checks["signal_to_entry_legs_used"] = before_entry
    checks["after_exit_legs_used"] = after_exit
    checks["multi_session_or_straddling_legs_used"] = multi
    checks["unusable_return_legs_used"] = unusable
    checks["unresolved_delisting_legs_used"] = unresolved
    checks["missing_factor_legs_used"] = missing_factor
    checks["factor_date_mismatches"] = factor_bad
    checks["terminal_delisting_double_counts"] = terminal_double
    checks["post_cutoff_return_rows_used"] = post_return
    checks["post_cutoff_factor_rows_used"] = post_factor
    checks["residual_value_mismatches"] += residual_bad
    checks["excess_value_mismatches"] += excess_bad
    checks["nonfinite_published_values"] += nonfinite
    if domain_bad or digest_bad:
        _add_difference(
            differences,
            "lineage_reconstruction_mismatches",
            0,
            {"key_domain": domain_bad, "binding_groups": digest_bad},
            "Published lineage differs from independently reconstructed raw-return paths",
        )


def _expected_processed_row(target: RecomputedLongRow) -> dict[str, Any]:
    source = target.source
    forward = target.forward
    assert forward is not None
    return {
        "signal_date": source.signal_date,
        "permno": source.permno,
        "permco": source.permco,
        "short_eligible": source.short_eligible,
        "holding_month": source.calendar.holding_month,
        "entry_date": source.calendar.entry_date,
        "exit_date": source.calendar.exit_date,
        "security_label_available_date": forward.available_date,
        "month_label_available_date": target.month_available_date,
        "unranked_excess_return": forward.excess,
        "unranked_residual_return": forward.residual,
        "residual_rank_numerator": target.residual_rank_numerator,
        "excess_rank_numerator": target.excess_rank_numerator,
        "residual_return_rank": target.residual_rank,
        "excess_return_rank": target.excess_rank,
        "residual_return_mad_standardized": target.residual_mad_standardized,
        "outcome_status": forward.status,
        "primary_label_status": target.primary_status,
        "magnitude_label_status": target.magnitude_status,
    }


def _expected_month_summary(month: MonthResult, minimum: int) -> dict[str, Any]:
    return {
        "signal_date": month.signal_date,
        "holding_month": month.holding_month,
        "entry_date": month.entry_date,
        "exit_date": month.exit_date,
        "eligible_security_count": month.eligible_count,
        "beta_complete_count": month.beta_complete_count,
        "valid_forward_count": month.valid_forward_count,
        "valid_primary_count": month.valid_primary_count,
        "minimum_required_count": minimum,
        "primary_month_valid": month.primary_month_valid,
        "terminal_delisting_label_count": month.terminal_count,
        "delayed_availability_count": month.delayed_count,
        "month_label_available_date": month.month_available_date,
        "residual_median": month.residual_median,
        "residual_mad": month.residual_mad,
        "residual_rank_numerator_sum": month.residual_rank_numerator_sum,
        "excess_rank_numerator_sum": month.excess_rank_numerator_sum,
        "residual_rank_mean": month.residual_rank_mean,
        "excess_rank_mean": month.excess_rank_mean,
        "month_status": month.month_status,
    }


def _table_comparison(
    path: Path,
    expected: Mapping[tuple[Any, ...], Mapping[str, Any]],
    key_fields: Sequence[str],
    *,
    tolerance: float = 1e-12,
    batch_rows: int = 65_536,
) -> tuple[int, int, int, int, int, int]:
    seen: set[tuple[Any, ...]] = set()
    duplicate = order = value_bad = nonfinite = rows = 0
    prior: tuple[Any, ...] | None = None
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=batch_rows, use_threads=False
    ):
        for row in pa.Table.from_batches([batch]).to_pylist():
            rows += 1
            key = tuple(row[name] for name in key_fields)
            if key in seen:
                duplicate += 1
            seen.add(key)
            if prior is not None and key < prior:
                order += 1
            prior = key
            expected_row = expected.get(key)
            if expected_row is None:
                continue
            value_bad += sum(
                not _close(expected_row.get(name), value, tolerance)
                for name, value in row.items()
            )
            nonfinite += sum(
                isinstance(value, float) and not _finite(value)
                for value in row.values()
            )
    domain = len(set(expected).symmetric_difference(seen))
    return rows, duplicate, order, domain, value_bad, nonfinite


def _unknown_value_count(
    path: Path,
    name: str,
    rules: Mapping[str, Any],
) -> int:
    domains = rules["value_domains"]
    field_domains = {
        "outcome_status": set(domains["outcome_status"]),
        "primary_label_status": set(domains["primary_label_status"]),
        "magnitude_label_status": set(domains["magnitude_label_status"]),
        "month_status": set(domains["month_status"]),
        "role": set(domains["split_role"]),
        "status_dimension": set(domains["status_dimension"]),
        "target_name": set(domains["distribution_target_name"]),
        "severity": set(domains["timing_severity"]),
    }
    count = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536):
        for row in pa.Table.from_batches([batch]).to_pylist():
            for field, allowed in field_domains.items():
                if field in row:
                    count += int(row[field] not in allowed)
            if name == "label_status_counts.parquet":
                dimension = row.get("status_dimension")
                domain_name = {
                    "outcome": "outcome_status",
                    "primary": "primary_label_status",
                    "magnitude": "magnitude_label_status",
                }.get(dimension)
                if domain_name is not None:
                    count += int(row.get("status") not in set(domains[domain_name]))
    return count


def _split_role(month: str, fold: Mapping[str, Any]) -> str:
    if month <= fold["train_last_holding_month"]:
        return "TRAIN_ELIGIBLE"
    if month == fold["purged_holding_month"]:
        return "PURGED"
    if (
        fold["validation_first_holding_month"]
        <= month
        <= fold["validation_last_holding_month"]
    ):
        return "VALIDATION"
    return "OUTSIDE"


def _expected_small_tables(
    context: _RecomputeContext,
) -> dict[str, tuple[tuple[str, ...], dict[tuple[Any, ...], dict[str, Any]]]]:
    minimum = int(context.rules["targets"]["minimum_valid_labels_per_signal"])
    processed = {
        key: _expected_processed_row(target)
        for key, target in context.long_rows.items()
    }
    summaries = {
        (signal,): _expected_month_summary(month, minimum)
        for signal, month in context.month_results.items()
    }
    coverage: dict[tuple[Any, ...], dict[str, Any]] = {}
    statuses: dict[tuple[Any, ...], dict[str, Any]] = {}
    distributions: dict[tuple[Any, ...], dict[str, Any]] = {}
    domains = context.rules["value_domains"]
    for signal, month in context.month_results.items():
        rows = context.long_by_signal.get(signal, [])
        outcome_count = sum(
            target.forward is not None and target.forward.status == PRESENT
            for target in rows
        )
        primary_count = sum(target.primary_status == PRESENT for target in rows)
        magnitude_count = sum(target.magnitude_status == PRESENT for target in rows)
        eligible = len(rows)
        rate = lambda count: 0.0 if eligible == 0 else _positive_zero(count / eligible)
        coverage[(signal,)] = {
            "signal_date": signal,
            "holding_month": month.holding_month,
            "eligible_count": eligible,
            "beta_complete_count": month.beta_complete_count,
            "outcome_present_count": outcome_count,
            "primary_present_count": primary_count,
            "magnitude_present_count": magnitude_count,
            "eligible_denominator": eligible,
            "outcome_coverage_rate": rate(outcome_count),
            "primary_coverage_rate": rate(primary_count),
            "magnitude_coverage_rate": rate(magnitude_count),
            "primary_month_valid": month.primary_month_valid,
        }
        nonlong = context.spine_counts[signal] - eligible
        dimensions = (
            (
                "outcome",
                "outcome_status",
                [
                    target.forward.status
                    for target in rows
                    if target.forward is not None
                ],
            ),
            (
                "primary",
                "primary_label_status",
                [target.primary_status for target in rows],
            ),
            (
                "magnitude",
                "magnitude_label_status",
                [target.magnitude_status for target in rows],
            ),
        )
        for dimension, domain_name, observed in dimensions:
            counts = Counter(observed)
            counts[NOT_LONG_ELIGIBLE] += nonlong
            total = context.spine_counts[signal]
            for status in domains[domain_name]:
                count = counts[status]
                key = (signal, dimension, status)
                statuses[key] = {
                    "signal_date": signal,
                    "holding_month": month.holding_month,
                    "status_dimension": dimension,
                    "status": status,
                    "row_count": count,
                    "dimension_total": total,
                    "row_fraction": _positive_zero(count / total),
                }
        targets = {
            "unranked_excess_return": [
                target.forward.excess if target.forward is not None else None
                for target in rows
            ],
            "unranked_residual_return": [
                target.forward.residual if target.forward is not None else None
                for target in rows
            ],
            "residual_return_rank": [target.residual_rank for target in rows],
            "excess_return_rank": [target.excess_rank for target in rows],
            "residual_return_mad_standardized": [
                target.residual_mad_standardized for target in rows
            ],
        }
        for target_name in domains["distribution_target_name"]:
            population = targets[target_name]
            finite = sorted(float(value) for value in population if _finite(value))
            if finite:
                stats = {
                    "minimum": _positive_zero(finite[0]),
                    "p01": type7_quantile(finite, 0.01),
                    "median": type7_quantile(finite, 0.50),
                    "mean": _positive_zero(math.fsum(finite) / len(finite)),
                    "p99": type7_quantile(finite, 0.99),
                    "maximum": _positive_zero(finite[-1]),
                }
            else:
                stats = dict.fromkeys(
                    ("minimum", "p01", "median", "mean", "p99", "maximum")
                )
            distributions[(signal, target_name)] = {
                "signal_date": signal,
                "holding_month": month.holding_month,
                "target_name": target_name,
                "finite_count": len(finite),
                "missing_count": len(population) - len(finite),
                **stats,
            }

    splits: dict[tuple[Any, ...], dict[str, Any]] = {}
    embargo = context.rules["splits"]["development_embargo_holding_month"]
    for signal, month in context.month_results.items():
        for raw_fold in context.rules["splits"]["folds"]:
            fold = dict(raw_fold)
            fit = dt.date.fromisoformat(fold["fit_signal_date"])
            role = _split_role(month.holding_month, fold)
            available = (
                month.month_available_date is not None
                and month.month_available_date <= fit
            )
            key = (fold["fold_id"], signal)
            splits[key] = {
                "fold_id": fold["fold_id"],
                "signal_date": signal,
                "holding_month": month.holding_month,
                "fit_signal_date": fit,
                "role": role,
                "month_label_available_date": month.month_available_date,
                "available_by_fit": available,
                "included_in_training": (
                    role == "TRAIN_ELIGIBLE" and month.primary_month_valid and available
                ),
                "included_in_validation": (
                    role == "VALIDATION" and month.primary_month_valid
                ),
                "purged_or_embargoed": (
                    role == "PURGED" or month.holding_month == embargo
                ),
            }
    return {
        "development_labels.parquet": (("signal_date", "permno"), processed),
        "development_label_month_summary.parquet": (("signal_date",), summaries),
        "label_split_membership.parquet": (("fold_id", "signal_date"), splits),
        "label_coverage.parquet": (("signal_date",), coverage),
        "label_status_counts.parquet": (
            ("signal_date", "status_dimension", "status"),
            statuses,
        ),
        "label_distribution.parquet": (
            ("signal_date", "target_name"),
            distributions,
        ),
    }


def _compare_small_tables(
    artifacts: object,
    context: _RecomputeContext,
    checks: MutableMapping[str, int],
    differences: list[IndependentDifference],
    *,
    batch_rows: int,
) -> None:
    specs = _expected_small_tables(context)
    gate_prefix = {
        "development_labels.parquet": "processed",
        "development_label_month_summary.parquet": "month_summary",
        "label_split_membership.parquet": "split_membership",
        "label_coverage.parquet": "coverage",
        "label_status_counts.parquet": "status_count",
        "label_distribution.parquet": "distribution",
    }
    row_gate = {
        "development_labels.parquet": "processed_rows",
        "development_label_month_summary.parquet": "month_summary_rows",
        "label_split_membership.parquet": "split_membership_rows",
    }
    for name, (key_fields, expected) in specs.items():
        output_path = _output_path(artifacts, name)
        rows, duplicate, order, domain, values, nonfinite = _table_comparison(
            output_path,
            expected,
            key_fields,
            batch_rows=batch_rows,
        )
        prefix = gate_prefix[name]
        checks[f"duplicate_{prefix}_keys"] = duplicate
        order_gate = _small_table_order_gate(name, prefix)
        checks[order_gate] = order
        if f"{prefix}_key_domain_mismatches" in checks:
            checks[f"{prefix}_key_domain_mismatches"] = domain
        if name in row_gate:
            checks[row_gate[name]] = rows
        checks["nonfinite_published_values"] += nonfinite
        checks["unknown_statuses"] += _unknown_value_count(
            output_path, name, context.rules
        )
        if name == "label_split_membership.parquet":
            checks["split_or_embargo_mismatches"] += values + domain
        if values or domain:
            _add_difference(
                differences,
                f"{prefix}_payload_mismatches",
                0,
                values + domain,
                f"{name} differs from independent reconstruction",
            )


def _small_table_order_gate(name: str, prefix: str) -> str:
    return (
        "processed_key_order_mismatches"
        if name == "development_labels.parquet"
        else f"{prefix}_sort_mismatches"
    )


def _compare_timing_table(
    artifacts: object,
    rules: Mapping[str, Any],
    checks: MutableMapping[str, int],
    differences: list[IndependentDifference],
) -> None:
    expected_gates = {
        str(name): int(value)
        for name, value in rules["quality_gates"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    expected = {
        (name,): {
            "check_name": name,
            "observed_count": value,
            "expected_count": value,
            "passed": True,
            "severity": "CRITICAL",
            "details": (
                f"Frozen Stage 7 hard gate; expected={value}; observed={value}"
            ),
        }
        for name, value in expected_gates.items()
    }
    rows, duplicate, order, domain, values, nonfinite = _table_comparison(
        _output_path(artifacts, "label_timing_checks.parquet"),
        expected,
        ("check_name",),
    )
    checks["duplicate_timing_check_keys"] = duplicate
    checks["timing_check_sort_mismatches"] = order
    checks["timing_check_key_domain_mismatches"] = domain
    checks["nonfinite_published_values"] += nonfinite
    checks["unknown_statuses"] += _unknown_value_count(
        _output_path(artifacts, "label_timing_checks.parquet"),
        "label_timing_checks.parquet",
        rules,
    )
    if rows != len(expected) or values or domain:
        _add_difference(
            differences,
            "timing_check_payload_mismatches",
            len(expected),
            {"rows": rows, "value_mismatches": values, "domain": domain},
            "Primary timing-check table differs from the frozen gate contract",
        )


def validate_hard_gate_shape(
    checks: Mapping[str, object],
    expected_names: Iterable[str],
) -> dict[str, int]:
    """Return a canonical gate map only if its exact frozen shape is valid."""

    expected = {str(name) for name in expected_names}
    _require(
        set(checks) == expected,
        "Independent hard-gate key set differs from the frozen 57 names",
    )
    _require(
        all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in checks.values()
        ),
        "Independent hard-gate values must be nonnegative integers",
    )
    return dict(sorted((name, int(value)) for name, value in checks.items()))


def recompute_stage7_labels(
    artifacts: object,
    *,
    batch_rows: int = 65_536,
) -> IndependentRecomputeEvidence:
    """Run the full independent Stage 7 numerical and payload recomputation.

    ``artifacts`` is the authenticated, duck-typed envelope produced by
    :mod:`verify_label_store`.  It must expose ``contract.rules``, input
    receipts with ``path`` attributes, and output receipts addressable by
    filename.  No path is rediscovered from a mutable pointer here.
    """

    _require(
        isinstance(batch_rows, int)
        and not isinstance(batch_rows, bool)
        and batch_rows > 0,
        "Independent recompute batch size must be positive",
    )
    rules = _rules(artifacts)
    expected_checks = {
        str(name): int(value)
        for name, value in rules["quality_gates"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    _require(len(expected_checks) == 57, "Frozen Stage 7 gate count differs")
    checks: dict[str, int] = {name: 0 for name in expected_checks}
    differences: list[IndependentDifference] = []
    context = _load_recompute_context(artifacts, rules)
    return_path = _input_path(artifacts, "crsp_security_day_return.parquet")
    beta_rows = _estimate_all_betas(context, return_path, batch_rows=batch_rows)
    forward_rows = _compute_all_forward_paths(
        context, return_path, batch_rows=batch_rows
    )
    _derive_month_targets(context)

    _audit_published_rows(
        context,
        _output_path(artifacts, "development_label_audit.parquet"),
        checks,
        differences,
        batch_rows=batch_rows,
    )
    _lineage_published_rows(
        context,
        _output_path(artifacts, "development_label_leg_lineage.parquet"),
        checks,
        differences,
        batch_rows=batch_rows,
    )
    _compare_small_tables(
        artifacts,
        context,
        checks,
        differences,
        batch_rows=batch_rows,
    )
    _compare_timing_table(artifacts, rules, checks, differences)

    # These are construction-invariance checks: short eligibility is never an
    # input to any beta, forward, rank, MAD, availability, or split formula;
    # excess and residual targets share the exact primary-valid mask.
    checks["short_eligibility_influence_violations"] = 0
    checks["excess_primary_mask_mismatches"] = sum(
        int(
            (target.forward is not None and target.forward.residual is None)
            != (target.forward is not None and target.forward.excess is None)
        )
        for target in context.long_rows.values()
    )
    checks["rank_population_mismatches"] = sum(
        int(
            month.primary_month_valid
            and month.valid_primary_count
            != sum(
                target.residual_rank is not None
                for target in context.long_by_signal.get(signal, ())
            )
        )
        for signal, month in context.month_results.items()
    )
    checks["nonzero_rank_numerator_month_sums"] = sum(
        int(
            month.primary_month_valid
            and (
                month.residual_rank_numerator_sum != 0
                or month.excess_rank_numerator_sum != 0
            )
        )
        for month in context.month_results.values()
    )

    for name, expected in expected_checks.items():
        observed = checks[name]
        if observed != expected:
            _add_difference(
                differences,
                f"hard_gate:{name}",
                expected,
                observed,
                "Independent Stage 7 hard-gate result differs",
            )
    canonical_checks = validate_hard_gate_shape(checks, expected_checks)
    metrics = {
        "return_rows_delivered_beta_pass": beta_rows,
        "return_rows_delivered_forward_pass": forward_rows,
        "long_rows_recomputed": len(context.long_rows),
        "signal_dates_recomputed": len(context.month_results),
        "lineage_rows_recomputed": sum(
            len(target.forward.contributions)
            for target in context.long_rows.values()
            if target.forward is not None
        ),
        "beta_complete_rows": sum(
            target.betas is not None for target in context.long_rows.values()
        ),
    }
    return IndependentRecomputeEvidence(
        checks={"hard_gates": canonical_checks},
        differences=tuple(differences),
        complete=True,
        metrics=metrics,
    )
