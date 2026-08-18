"""Dated dynamic-PERMNO accounting engine for Stage 11.

Each session is processed in one enforced order: opening-notional charges,
authoritative total returns, then corporate actions assigned to that close.
Daily inputs are keyed by PERMNO, so a cash settlement can retire a security
and a stock successor can begin receiving returns and borrow charges on the
next session without changing or padding a fixed return matrix.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from portfolio_bot.allocation.costs import PRIMARY_FINANCING_ANNUAL_RATE
from portfolio_bot.validation.accounting import (
    AccountingError,
    PortfolioState,
    apply_daily_accounting,
    rebalance_portfolio,
)
from portfolio_bot.validation.corporate_actions import (
    STOCK_SUCCESSOR_SETTLEMENT,
    CorporateAction,
    CorporateActionLedger,
    CorporateActionReceipt,
    DailyReturnCloseReceipt,
    apply_corporate_actions,
    portfolio_state_sha256,
)
from portfolio_bot.validation.session_authority import (
    HoldingPeriodSessionAuthority,
    HoldingPeriodSessionAuthorityError,
)


class DailyEngineError(ValueError):
    """Raised when a dated holding-period path is incomplete or ambiguous."""


_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _date(value: object, *, name: str) -> dt.date:
    if type(value) is not dt.date:
        raise DailyEngineError(f"{name} must be a datetime.date")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DailyEngineError(f"{name} must be a lowercase SHA-256")
    return value


def _vector(value: object, *, dtype: np.dtype | type, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=dtype).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise DailyEngineError(f"{name} cannot be converted") from exc
    if result.ndim != 1:
        raise DailyEngineError(f"{name} must be one-dimensional")
    result.setflags(write=False)
    return result


def _date_tuple(value: object, *, name: str) -> tuple[dt.date, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DailyEngineError(f"{name} must be a sequence of dates")
    result = tuple(value)
    if any(type(item) is not dt.date for item in result):
        raise DailyEngineError(f"{name} must contain only datetime.date values")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class MarkContinuityLedger:
    """Last authoritative mark and unresolved-gap state for live identities."""

    permno: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    last_authoritative_mark_date: tuple[dt.date, ...] = ()
    pending_no_mark_count: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    pending_evidence_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        permno = _vector(self.permno, dtype=np.int64, name="mark ledger PERMNO")
        dates = _date_tuple(
            self.last_authoritative_mark_date,
            name="last_authoritative_mark_date",
        )
        pending = _vector(
            self.pending_no_mark_count,
            dtype=np.int64,
            name="pending_no_mark_count",
        )
        evidence = tuple(self.pending_evidence_sha256)
        if not (
            len(permno) == len(dates) == len(pending) == len(evidence)
        ):
            raise DailyEngineError("mark-continuity ledger lengths differ")
        if (
            np.any(permno <= 0)
            or len(np.unique(permno)) != len(permno)
            or (len(permno) and not np.all(permno[:-1] < permno[1:]))
        ):
            raise DailyEngineError(
                "mark-continuity PERMNO values must be positive, unique, and sorted"
            )
        if np.any(pending < 0):
            raise DailyEngineError("pending no-mark counts must be nonnegative")
        for count, digest in zip(pending, evidence, strict=True):
            if int(count) == 0:
                if digest != _EMPTY_SHA256:
                    raise DailyEngineError(
                        "fully marked identity has pending evidence"
                    )
            else:
                _sha256(digest, name="pending_evidence_sha256")
                if digest == _EMPTY_SHA256:
                    raise DailyEngineError(
                        "pending identity lacks no-mark evidence"
                    )
        object.__setattr__(self, "permno", permno)
        object.__setattr__(self, "last_authoritative_mark_date", dates)
        object.__setattr__(self, "pending_no_mark_count", pending)
        object.__setattr__(self, "pending_evidence_sha256", evidence)

    @classmethod
    def empty(cls) -> "MarkContinuityLedger":
        return cls()

    @property
    def pending_permnos(self) -> tuple[int, ...]:
        return tuple(
            int(permno)
            for permno, count in zip(
                self.permno, self.pending_no_mark_count, strict=True
            )
            if int(count) > 0
        )

    @property
    def ledger_sha256(self) -> str:
        digest = hashlib.sha256(b"stage11-mark-continuity-ledger-v1\0")
        for permno, mark_date, pending, evidence in zip(
            self.permno,
            self.last_authoritative_mark_date,
            self.pending_no_mark_count,
            self.pending_evidence_sha256,
            strict=True,
        ):
            digest.update(struct.pack("<q", int(permno)))
            digest.update(mark_date.isoformat().encode("ascii"))
            digest.update(struct.pack("<q", int(pending)))
            digest.update(evidence.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class DatedMarketCloseInput:
    """One authenticated daily return/rate projection keyed by PERMNO."""

    trading_date: dt.date
    permno: np.ndarray
    total_return: np.ndarray
    borrow_rate_annual: np.ndarray
    daily_risk_free_rate: float
    source_payload_sha256: str
    observed_return_span_start_date: tuple[dt.date, ...] = ()
    no_mark_permno: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    no_mark_borrow_rate_annual: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    no_mark_source_record_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _date(self.trading_date, name="trading_date")
        permno = _vector(self.permno, dtype=np.int64, name="permno")
        returns = _vector(
            self.total_return, dtype=np.float64, name="total_return"
        )
        borrow = _vector(
            self.borrow_rate_annual,
            dtype=np.float64,
            name="borrow_rate_annual",
        )
        if not (len(permno) == len(returns) == len(borrow)):
            raise DailyEngineError("daily PERMNO, return, and borrow lengths differ")
        if np.any(permno <= 0) or len(np.unique(permno)) != len(permno):
            raise DailyEngineError("daily PERMNO values must be positive and unique")
        if len(permno) and not np.all(permno[:-1] < permno[1:]):
            raise DailyEngineError("daily PERMNO values must be strictly increasing")
        if not np.isfinite(returns).all() or np.any(returns < -1.0):
            raise DailyEngineError("daily total returns must be finite and >= -1")
        if not np.isfinite(borrow).all() or np.any(borrow < 0.0):
            raise DailyEngineError("daily borrow rates must be finite and nonnegative")
        span_starts = _date_tuple(
            self.observed_return_span_start_date,
            name="observed_return_span_start_date",
        )
        if span_starts and len(span_starts) != len(permno):
            raise DailyEngineError(
                "observed return-span starts must align to observed PERMNO"
            )
        if any(value >= self.trading_date for value in span_starts):
            raise DailyEngineError(
                "observed return-span starts must precede their trading date"
            )
        no_mark_permno = _vector(
            self.no_mark_permno,
            dtype=np.int64,
            name="no_mark_permno",
        )
        no_mark_borrow = _vector(
            self.no_mark_borrow_rate_annual,
            dtype=np.float64,
            name="no_mark_borrow_rate_annual",
        )
        no_mark_hashes = tuple(self.no_mark_source_record_sha256)
        if not (
            len(no_mark_permno) == len(no_mark_borrow) == len(no_mark_hashes)
        ):
            raise DailyEngineError("no-mark PERMNO, borrow, and evidence lengths differ")
        if (
            np.any(no_mark_permno <= 0)
            or len(np.unique(no_mark_permno)) != len(no_mark_permno)
            or (
                len(no_mark_permno)
                and not np.all(no_mark_permno[:-1] < no_mark_permno[1:])
            )
        ):
            raise DailyEngineError(
                "no-mark PERMNO values must be positive, unique, and sorted"
            )
        if set(int(value) for value in permno).intersection(
            int(value) for value in no_mark_permno
        ):
            raise DailyEngineError("observed and no-mark channels overlap")
        if not np.isfinite(no_mark_borrow).all() or np.any(no_mark_borrow < 0.0):
            raise DailyEngineError(
                "no-mark borrow rates must be finite and nonnegative"
            )
        for value in no_mark_hashes:
            _sha256(value, name="no_mark_source_record_sha256")
        try:
            risk_free = float(self.daily_risk_free_rate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DailyEngineError("daily_risk_free_rate must be numeric") from exc
        if not math.isfinite(risk_free) or risk_free <= -1.0:
            raise DailyEngineError("daily_risk_free_rate must be finite and exceed -1")
        _sha256(self.source_payload_sha256, name="source_payload_sha256")
        object.__setattr__(self, "permno", permno)
        object.__setattr__(self, "total_return", returns)
        object.__setattr__(self, "borrow_rate_annual", borrow)
        object.__setattr__(self, "daily_risk_free_rate", risk_free)
        object.__setattr__(
            self, "observed_return_span_start_date", span_starts
        )
        object.__setattr__(self, "no_mark_permno", no_mark_permno)
        object.__setattr__(
            self, "no_mark_borrow_rate_annual", no_mark_borrow
        )
        object.__setattr__(
            self, "no_mark_source_record_sha256", no_mark_hashes
        )

    @property
    def projected_payload_sha256(self) -> str:
        digest = hashlib.sha256(b"stage11-dated-market-close-v2\0")
        digest.update(self.trading_date.isoformat().encode("ascii"))
        digest.update(self.source_payload_sha256.encode("ascii"))
        digest.update(np.asarray(self.permno, dtype="<i8").tobytes(order="C"))
        digest.update(np.asarray(self.total_return, dtype="<f8").tobytes(order="C"))
        digest.update(
            np.asarray(self.borrow_rate_annual, dtype="<f8").tobytes(order="C")
        )
        digest.update(
            b"explicit-spans\0"
            if self.observed_return_span_start_date
            else b"legacy-spans\0"
        )
        for value in self.observed_return_span_start_date:
            digest.update(value.isoformat().encode("ascii"))
            digest.update(b"\0")
        digest.update(
            np.asarray(self.no_mark_permno, dtype="<i8").tobytes(order="C")
        )
        digest.update(
            np.asarray(
                self.no_mark_borrow_rate_annual, dtype="<f8"
            ).tobytes(order="C")
        )
        for value in self.no_mark_source_record_sha256:
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        digest.update(struct.pack("<d", self.daily_risk_free_rate))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DailyEngineSessionReceipt:
    trading_date: dt.date
    session_authority_sha256: str
    opening_state_sha256: str
    post_return_state_sha256: str
    closing_state_sha256: str
    projected_payload_sha256: str
    opening_mark_continuity_sha256: str
    closing_mark_continuity_sha256: str
    no_mark_permnos: tuple[int, ...]
    resolved_pending_permnos: tuple[int, ...]
    observed_zero_return_permnos: tuple[int, ...]
    corporate_action_ids: tuple[str, ...]
    unheld_corporate_action_ids: tuple[str, ...]
    net_total_return_before_actions: float
    borrow_cost_dollars: float
    financing_cost_dollars: float


@dataclass(frozen=True, slots=True, eq=False)
class DatedHoldingPeriodAccounting:
    state: PortfolioState
    action_ledger: CorporateActionLedger
    mark_continuity_ledger: MarkContinuityLedger
    session_receipts: tuple[DailyEngineSessionReceipt, ...]
    corporate_action_receipts: tuple[CorporateActionReceipt, ...]
    session_authority_sha256: str
    pre_trade_nav_dollars: float
    daily_nav_dollars: np.ndarray
    transaction_cost_dollars: float
    borrow_cost_dollars: float
    financing_cost_dollars: float
    net_total_return: float
    risk_free_total_return: float
    net_excess_return: float

    def __post_init__(self) -> None:
        authority_sha256 = _sha256(
            self.session_authority_sha256,
            name="session_authority_sha256",
        )
        path = _vector(
            self.daily_nav_dollars,
            dtype=np.float64,
            name="daily_nav_dollars",
        )
        if len(path) != len(self.session_receipts) or not np.isfinite(path).all():
            raise DailyEngineError("daily NAV path differs from session receipts")
        if np.any(path <= 0.0):
            raise DailyEngineError("daily NAV must remain positive")
        if any(
            receipt.session_authority_sha256 != authority_sha256
            for receipt in self.session_receipts
        ):
            raise DailyEngineError(
                "session receipts differ from the holding-period authority"
            )
        live = {
            int(permno)
            for permno, value in zip(
                self.state.permno,
                self.state.position_value_dollars,
                strict=True,
            )
            if float(value) != 0.0
        }
        if set(int(value) for value in self.mark_continuity_ledger.permno) != live:
            raise DailyEngineError(
                "closing mark-continuity ledger differs from live positions"
            )
        object.__setattr__(self, "daily_nav_dollars", path)


def _ledger_records(
    ledger: MarkContinuityLedger,
) -> dict[int, tuple[dt.date, int, str]]:
    return {
        int(permno): (mark_date, int(pending), evidence)
        for permno, mark_date, pending, evidence in zip(
            ledger.permno,
            ledger.last_authoritative_mark_date,
            ledger.pending_no_mark_count,
            ledger.pending_evidence_sha256,
            strict=True,
        )
    }


def _ledger_from_records(
    records: dict[int, tuple[dt.date, int, str]],
) -> MarkContinuityLedger:
    ordered = sorted(records)
    return MarkContinuityLedger(
        permno=np.asarray(ordered, dtype=np.int64),
        last_authoritative_mark_date=tuple(
            records[value][0] for value in ordered
        ),
        pending_no_mark_count=np.asarray(
            [records[value][1] for value in ordered], dtype=np.int64
        ),
        pending_evidence_sha256=tuple(
            records[value][2] for value in ordered
        ),
    )


def _advance_pending_evidence(
    *,
    prior_sha256: str,
    source_sha256: str,
    trading_date: dt.date,
    permno: int,
) -> str:
    digest = hashlib.sha256(b"stage11-pending-no-mark-evidence-v1\0")
    for value in (
        prior_sha256,
        source_sha256,
        trading_date.isoformat(),
        str(permno),
    ):
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _rebalance_mark_continuity(
    state: PortfolioState,
    *,
    target_permno: np.ndarray,
    target_weights: np.ndarray,
    execution_price: np.ndarray,
    entry_date: dt.date,
    initial: MarkContinuityLedger,
) -> MarkContinuityLedger:
    opening_live = {
        int(permno): float(value)
        for permno, value in zip(
            state.permno, state.position_value_dollars, strict=True
        )
        if float(value) != 0.0
    }
    prior = _ledger_records(initial)
    if set(prior) != set(opening_live):
        raise DailyEngineError(
            "opening mark-continuity ledger differs from live positions"
        )
    opening_weights = {
        permno: value / state.nav_dollars
        for permno, value in opening_live.items()
    }
    target_values = np.asarray(target_weights, dtype=np.float64) * state.nav_dollars
    target_by_permno = {
        int(permno): (float(weight), float(value), float(price))
        for permno, weight, value, price in zip(
            target_permno,
            target_weights,
            target_values,
            execution_price,
            strict=True,
        )
    }
    for permno, (_mark_date, pending, _evidence) in prior.items():
        if pending <= 0:
            continue
        target_weight = target_by_permno.get(
            permno, (0.0, 0.0, math.nan)
        )[0]
        if target_weight != opening_weights[permno]:
            raise DailyEngineError(
                "pending identity cannot be traded at rebalance: "
                + str(permno)
            )
    records: dict[int, tuple[dt.date, int, str]] = {}
    for permno, (target_weight, target_value, price) in target_by_permno.items():
        if target_value == 0.0:
            continue
        previous = prior.get(permno)
        if previous is not None and previous[1] > 0:
            records[permno] = previous
            continue
        if math.isfinite(price) and price > 0.0:
            records[permno] = (entry_date, 0, _EMPTY_SHA256)
            continue
        if (
            previous is not None
            and opening_weights[permno] == target_weight
        ):
            records[permno] = previous
            continue
        raise DailyEngineError(
            "live identity lacks an authoritative entry mark: " + str(permno)
        )
    return _ledger_from_records(records)


def _aligned_daily_values(
    state: PortfolioState,
    daily: DatedMarketCloseInput,
    ledger: MarkContinuityLedger,
) -> tuple[
    np.ndarray,
    np.ndarray,
    MarkContinuityLedger,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    by_permno = {int(value): index for index, value in enumerate(daily.permno)}
    no_mark_by_permno = {
        int(value): index for index, value in enumerate(daily.no_mark_permno)
    }
    continuity = _ledger_records(ledger)
    returns = np.zeros(len(state.permno), dtype=np.float64)
    borrow = np.zeros(len(state.permno), dtype=np.float64)
    missing_nonzero: list[int] = []
    no_mark_held: list[int] = []
    resolved_pending: list[int] = []
    observed_zero: list[int] = []
    for position, (permno, value) in enumerate(
        zip(state.permno, state.position_value_dollars, strict=True)
    ):
        identity = int(permno)
        index = by_permno.get(identity)
        no_mark_index = no_mark_by_permno.get(identity)
        if float(value) == 0.0:
            continue
        if identity not in continuity:
            raise DailyEngineError(
                "live identity escaped its mark-continuity ledger: "
                + str(identity)
            )
        if index is None:
            if no_mark_index is None:
                missing_nonzero.append(identity)
                continue
            mark_date, pending, evidence = continuity[identity]
            source_sha = daily.no_mark_source_record_sha256[no_mark_index]
            continuity[identity] = (
                mark_date,
                pending + 1,
                _advance_pending_evidence(
                    prior_sha256=evidence,
                    source_sha256=source_sha,
                    trading_date=daily.trading_date,
                    permno=identity,
                ),
            )
            borrow[position] = daily.no_mark_borrow_rate_annual[no_mark_index]
            no_mark_held.append(identity)
            continue
        mark_date, pending, _evidence = continuity[identity]
        if daily.observed_return_span_start_date:
            span_start = daily.observed_return_span_start_date[index]
            if span_start != mark_date:
                raise DailyEngineError(
                    "authoritative return span does not reconnect to the last "
                    f"mark for PERMNO {identity}: {span_start} != {mark_date}"
                )
        # Legacy synthetic/preflight inputs have no explicit span metadata.
        # They are accepted only as a one-step engine fixture and anchor the
        # observed close directly; production Stage 11 supplies exact spans.
        returns[position] = daily.total_return[index]
        borrow[position] = daily.borrow_rate_annual[index]
        if pending > 0:
            resolved_pending.append(identity)
        if float(daily.total_return[index]) == 0.0:
            observed_zero.append(identity)
        continuity[identity] = (
            daily.trading_date,
            0,
            _EMPTY_SHA256,
        )
    if missing_nonzero:
        raise DailyEngineError(
            "open identities lack an authoritative daily return/rate: "
            + ", ".join(str(value) for value in missing_nonzero)
        )
    returns.setflags(write=False)
    borrow.setflags(write=False)
    return (
        returns,
        borrow,
        _ledger_from_records(continuity),
        tuple(sorted(no_mark_held)),
        tuple(sorted(resolved_pending)),
        tuple(sorted(observed_zero)),
    )


def run_dated_holding_period(
    state: PortfolioState,
    *,
    session_authority: HoldingPeriodSessionAuthority,
    target_permno: Sequence[int] | np.ndarray,
    target_weights: Sequence[float] | np.ndarray,
    execution_price: Sequence[float] | np.ndarray,
    full_spread: Sequence[float] | np.ndarray,
    adv_dollars: Sequence[float] | np.ndarray,
    impact_volatility_daily: Sequence[float] | np.ndarray,
    daily_inputs: Sequence[DatedMarketCloseInput],
    corporate_actions: Iterable[CorporateAction] = (),
    initial_action_ledger: CorporateActionLedger = CorporateActionLedger(),
    initial_mark_continuity_ledger: MarkContinuityLedger | None = None,
    financing_rate_annual: float = PRIMARY_FINANCING_ANNUAL_RATE,
) -> DatedHoldingPeriodAccounting:
    """Trade at entry close and run the exact authenticated ``(entry, exit]`` path."""

    if type(session_authority) is not HoldingPeriodSessionAuthority:
        raise DailyEngineError(
            "daily accounting requires HoldingPeriodSessionAuthority"
        )

    if isinstance(daily_inputs, (str, bytes)) or not isinstance(
        daily_inputs, Sequence
    ):
        raise DailyEngineError("daily_inputs must be a sequence")
    days = tuple(daily_inputs)
    if not days or any(not isinstance(day, DatedMarketCloseInput) for day in days):
        raise DailyEngineError("holding period requires dated market-close inputs")
    dates = tuple(day.trading_date for day in days)
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise DailyEngineError("daily input dates must be strictly increasing")
    try:
        session_authority.require_exact_return_dates(dates)
    except HoldingPeriodSessionAuthorityError as exc:
        raise DailyEngineError(
            "daily input dates escaped their authenticated holding-period authority"
        ) from exc
    actions = tuple(corporate_actions)
    if any(not isinstance(action, CorporateAction) for action in actions):
        raise TypeError("corporate_actions must contain CorporateAction records")
    expected_actions = tuple(
        sorted(
            actions,
            key=lambda action: (
                action.assigned_return_date,
                action.action_sequence,
                action.action_id,
            ),
        )
    )
    if actions != expected_actions:
        raise DailyEngineError("corporate actions are not in deterministic date order")
    if any(action.assigned_return_date not in set(dates) for action in actions):
        raise DailyEngineError("corporate action falls outside the holding-period dates")
    actions_by_date: dict[dt.date, list[CorporateAction]] = {}
    for action in actions:
        actions_by_date.setdefault(action.assigned_return_date, []).append(action)

    if initial_mark_continuity_ledger is None:
        if np.any(np.asarray(state.position_value_dollars) != 0.0):
            raise DailyEngineError(
                "non-cash opening state requires a mark-continuity ledger"
            )
        initial_marks = MarkContinuityLedger.empty()
    elif type(initial_mark_continuity_ledger) is MarkContinuityLedger:
        initial_marks = initial_mark_continuity_ledger
    else:
        raise DailyEngineError(
            "initial_mark_continuity_ledger must be MarkContinuityLedger"
        )

    target_permno_array = np.asarray(target_permno, dtype=np.int64)
    target_weight_array = np.asarray(target_weights, dtype=np.float64)
    execution_price_array = np.asarray(execution_price, dtype=np.float64)
    mark_ledger = _rebalance_mark_continuity(
        state,
        target_permno=target_permno_array,
        target_weights=target_weight_array,
        execution_price=execution_price_array,
        entry_date=session_authority.calendar_row.entry_date,
        initial=initial_marks,
    )

    rebalance = rebalance_portfolio(
        state,
        target_permno=target_permno_array,
        target_weights=target_weight_array,
        execution_price=execution_price_array,
        full_spread=full_spread,
        adv_dollars=adv_dollars,
        impact_volatility_daily=impact_volatility_daily,
    )
    current = rebalance.state
    ledger = initial_action_ledger
    nav_path: list[float] = []
    session_receipts: list[DailyEngineSessionReceipt] = []
    action_receipts: list[CorporateActionReceipt] = []
    borrow_total = 0.0
    financing_total = 0.0
    risk_free_rates: list[float] = []
    for day in days:
        opening_hash = portfolio_state_sha256(current)
        opening_mark_hash = mark_ledger.ledger_sha256
        opening_held_permnos = {
            int(permno)
            for permno, value in zip(
                current.permno,
                current.position_value_dollars,
                strict=True,
            )
            if float(value) != 0.0
        }
        (
            returns,
            borrow,
            post_return_marks,
            no_mark_permnos,
            resolved_pending_permnos,
            observed_zero_permnos,
        ) = _aligned_daily_values(current, day, mark_ledger)
        accounted = apply_daily_accounting(
            current,
            asset_total_returns=returns,
            daily_risk_free_rate=day.daily_risk_free_rate,
            borrow_rate_annual=borrow,
            financing_rate_annual=financing_rate_annual,
        )
        post_return = accounted.state
        post_return_hash = portfolio_state_sha256(post_return)
        close_receipt = DailyReturnCloseReceipt(
            assigned_return_date=day.trading_date,
            closing_state_sha256=post_return_hash,
            total_return_payload_sha256=day.projected_payload_sha256,
        )
        today_all = tuple(actions_by_date.get(day.trading_date, ()))
        today = tuple(
            action
            for action in today_all
            if action.source_permno in opening_held_permnos
        )
        unheld_today = tuple(
            action
            for action in today_all
            if action.source_permno not in opening_held_permnos
        )
        pending_after_return = set(post_return_marks.pending_permnos)
        pending_actions = tuple(
            action.source_permno
            for action in today
            if action.source_permno in pending_after_return
        )
        if pending_actions:
            raise DailyEngineError(
                "corporate action cannot alter a pending no-mark identity: "
                + ", ".join(str(value) for value in sorted(pending_actions))
            )
        if today:
            applied = apply_corporate_actions(
                post_return,
                today,
                close_receipt=close_receipt,
                ledger=ledger,
            )
            current = applied.state
            ledger = applied.ledger
            action_receipts.extend(applied.receipts)
        else:
            current = post_return
        mark_records = _ledger_records(post_return_marks)
        successor_sources = {
            int(action.successor_permno): int(action.source_permno)
            for action in today
            if action.settlement_kind == STOCK_SUCCESSOR_SETTLEMENT
            and action.successor_permno is not None
        }
        closing_live = {
            int(permno)
            for permno, value in zip(
                current.permno,
                current.position_value_dollars,
                strict=True,
            )
            if float(value) != 0.0
        }
        closing_records: dict[int, tuple[dt.date, int, str]] = {}
        for identity in sorted(closing_live):
            if identity in mark_records:
                if identity in successor_sources:
                    source = successor_sources[identity]
                    source_record = mark_records.get(source)
                    target_record = mark_records[identity]
                    if (
                        source_record is None
                        or source_record[0] != day.trading_date
                        or source_record[1] != 0
                        or target_record[0] != day.trading_date
                        or target_record[1] != 0
                    ):
                        raise DailyEngineError(
                            "stock-successor merge has incompatible mark anchors"
                        )
                closing_records[identity] = mark_records[identity]
                continue
            source = successor_sources.get(identity)
            source_record = None if source is None else mark_records.get(source)
            if (
                source_record is None
                or source_record[0] != day.trading_date
                or source_record[1] != 0
            ):
                raise DailyEngineError(
                    "new live identity lacks a resolved stock-successor mark"
                )
            closing_records[identity] = (
                day.trading_date,
                0,
                _EMPTY_SHA256,
            )
        mark_ledger = _ledger_from_records(closing_records)
        borrow_total += accounted.borrow_cost_dollars
        financing_total += accounted.financing_cost_dollars
        risk_free_rates.append(day.daily_risk_free_rate)
        nav_path.append(current.nav_dollars)
        session_receipts.append(
            DailyEngineSessionReceipt(
                trading_date=day.trading_date,
                session_authority_sha256=session_authority.authority_sha256,
                opening_state_sha256=opening_hash,
                post_return_state_sha256=post_return_hash,
                closing_state_sha256=portfolio_state_sha256(current),
                projected_payload_sha256=day.projected_payload_sha256,
                opening_mark_continuity_sha256=opening_mark_hash,
                closing_mark_continuity_sha256=mark_ledger.ledger_sha256,
                no_mark_permnos=no_mark_permnos,
                resolved_pending_permnos=resolved_pending_permnos,
                observed_zero_return_permnos=observed_zero_permnos,
                corporate_action_ids=tuple(action.action_id for action in today),
                unheld_corporate_action_ids=tuple(
                    action.action_id for action in unheld_today
                ),
                net_total_return_before_actions=accounted.net_total_return,
                borrow_cost_dollars=accounted.borrow_cost_dollars,
                financing_cost_dollars=accounted.financing_cost_dollars,
            )
        )

    risk_free_total = float(
        np.prod(1.0 + np.asarray(risk_free_rates), dtype=np.float64) - 1.0
    )
    net_total = current.nav_dollars / rebalance.pre_trade_nav_dollars - 1.0
    return DatedHoldingPeriodAccounting(
        state=current,
        action_ledger=ledger,
        mark_continuity_ledger=mark_ledger,
        session_receipts=tuple(session_receipts),
        corporate_action_receipts=tuple(action_receipts),
        session_authority_sha256=session_authority.authority_sha256,
        pre_trade_nav_dollars=rebalance.pre_trade_nav_dollars,
        daily_nav_dollars=np.asarray(nav_path, dtype=np.float64),
        transaction_cost_dollars=rebalance.transaction_cost_dollars,
        borrow_cost_dollars=borrow_total,
        financing_cost_dollars=financing_total,
        net_total_return=net_total,
        risk_free_total_return=risk_free_total,
        net_excess_return=net_total - risk_free_total,
    )


__all__ = [
    "DailyEngineError",
    "DailyEngineSessionReceipt",
    "DatedHoldingPeriodAccounting",
    "DatedMarketCloseInput",
    "HoldingPeriodSessionAuthority",
    "MarkContinuityLedger",
    "run_dated_holding_period",
]
