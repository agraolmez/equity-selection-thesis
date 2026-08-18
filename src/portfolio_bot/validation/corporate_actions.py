"""Dated, post-return corporate-action accounting for Stage 11.

An action can be applied only against a receipt binding the exact portfolio
state after the authoritative return assigned to that session has run.  This
prevents an already represented delisting or distribution from being applied
before the return or counted twice.  Cash settlements retire the source row;
stock consideration replaces it with the authenticated successor identity.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import struct
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from portfolio_bot.validation.accounting import PortfolioState


CASH_SETTLEMENT = "cash"
STOCK_SUCCESSOR_SETTLEMENT = "stock_successor"
SettlementKind = Literal["cash", "stock_successor"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CorporateActionError(ValueError):
    """Raised when a corporate action cannot be accounted unambiguously."""


class UnresolvedCorporateActionError(CorporateActionError):
    """Raised when stock consideration has no authenticated successor mapping."""


def _date(value: object, *, name: str) -> dt.date:
    if type(value) is not dt.date:
        raise CorporateActionError(f"{name} must be a datetime.date")
    return value


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CorporateActionError(f"{name} must be a lowercase SHA-256")
    return value


def portfolio_state_sha256(state: PortfolioState) -> str:
    """Hash one exact ordered portfolio state for post-return action binding."""

    if not isinstance(state, PortfolioState):
        raise TypeError("state must be PortfolioState")
    digest = hashlib.sha256(b"stage11-portfolio-state-v1\0")
    digest.update(np.asarray(state.permno, dtype="<i8").tobytes(order="C"))
    digest.update(
        np.asarray(state.position_value_dollars, dtype="<f8").tobytes(order="C")
    )
    digest.update(struct.pack("<d", float(state.cash_dollars)))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DailyReturnCloseReceipt:
    """Close identity proving the assigned daily return was applied first."""

    assigned_return_date: dt.date
    closing_state_sha256: str
    total_return_payload_sha256: str

    def __post_init__(self) -> None:
        _date(self.assigned_return_date, name="assigned_return_date")
        _digest(self.closing_state_sha256, name="closing_state_sha256")
        _digest(
            self.total_return_payload_sha256,
            name="total_return_payload_sha256",
        )


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """One authenticated action assigned to one authoritative return date."""

    action_id: str
    economic_event_date: dt.date
    assigned_return_date: dt.date
    public_availability_date: dt.date
    action_sequence: int
    source_record_sha256: str
    source_permno: int
    settlement_kind: SettlementKind
    successor_permno: int | None = None
    settlement_value_multiplier: float = 1.0
    terminal_total_return_already_applied: bool = True
    terminal_sensitivity_policy_sha256: str | None = None
    terminal_sensitivity_evidence_sha256: str | None = None
    event_day_quoted_full_spread: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id.strip():
            raise CorporateActionError("action_id must be a nonempty string")
        object.__setattr__(self, "action_id", self.action_id.strip())
        economic = _date(self.economic_event_date, name="economic_event_date")
        assigned = _date(self.assigned_return_date, name="assigned_return_date")
        available = _date(
            self.public_availability_date,
            name="public_availability_date",
        )
        if economic > assigned:
            raise CorporateActionError(
                "economic_event_date cannot follow assigned_return_date"
            )
        # ``assigned_return_date`` is the economic effective session used by
        # the retrospective fixed-share accounting engine.  CRSP can publish
        # or revise delisting mechanics after that session.  Preserve that
        # later knowledge date instead of rewriting economic lineage; the
        # walk-forward outcome gate (not this accounting primitive) must keep
        # the action/return out of every fit until ``public_availability_date``.
        if type(self.action_sequence) is not int or self.action_sequence < 1:
            raise CorporateActionError("action_sequence must be a positive integer")
        _digest(self.source_record_sha256, name="source_record_sha256")
        if isinstance(self.source_permno, bool) or not isinstance(
            self.source_permno, (int, np.integer)
        ):
            raise CorporateActionError("source_permno must be an integer")
        source = int(self.source_permno)
        if source <= 0:
            raise CorporateActionError("source_permno must be positive")
        object.__setattr__(self, "source_permno", source)
        if self.settlement_kind not in {
            CASH_SETTLEMENT,
            STOCK_SUCCESSOR_SETTLEMENT,
        }:
            raise CorporateActionError("unknown settlement_kind")
        try:
            multiplier = float(self.settlement_value_multiplier)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CorporateActionError(
                "settlement_value_multiplier must be numeric"
            ) from exc
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise CorporateActionError(
                "settlement_value_multiplier must be finite and nonnegative"
            )
        if type(self.terminal_total_return_already_applied) is not bool:
            raise CorporateActionError(
                "terminal_total_return_already_applied must be boolean"
            )
        if self.terminal_total_return_already_applied and multiplier != 1.0:
            raise CorporateActionError(
                "an already-applied terminal total return requires multiplier 1"
            )
        object.__setattr__(self, "settlement_value_multiplier", multiplier)

        terminal_fields = (
            self.terminal_sensitivity_policy_sha256,
            self.terminal_sensitivity_evidence_sha256,
            self.event_day_quoted_full_spread,
        )
        if any(value is not None for value in terminal_fields):
            if not all(value is not None for value in terminal_fields):
                raise CorporateActionError(
                    "terminal sensitivity metadata must be complete or absent"
                )
            _digest(
                self.terminal_sensitivity_policy_sha256,
                name="terminal_sensitivity_policy_sha256",
            )
            _digest(
                self.terminal_sensitivity_evidence_sha256,
                name="terminal_sensitivity_evidence_sha256",
            )
            try:
                quoted_spread = float(self.event_day_quoted_full_spread)
            except (TypeError, ValueError, OverflowError) as exc:
                raise CorporateActionError(
                    "event_day_quoted_full_spread must be numeric"
                ) from exc
            if not np.isfinite(quoted_spread) or not 0.0 <= quoted_spread <= 0.20:
                raise CorporateActionError(
                    "event_day_quoted_full_spread is outside the authenticated range"
                )
            if (
                self.settlement_kind != CASH_SETTLEMENT
                or multiplier != 1.0
                or self.terminal_total_return_already_applied is not True
            ):
                raise CorporateActionError(
                    "terminal sensitivity metadata requires final-close cash retirement"
                )
            object.__setattr__(self, "event_day_quoted_full_spread", quoted_spread)

        if self.settlement_kind == CASH_SETTLEMENT:
            if self.successor_permno is not None:
                raise CorporateActionError(
                    "cash settlement cannot specify a successor_permno"
                )
        else:
            if self.successor_permno is None:
                # Fail closed later with the action identity fully retained.
                return
            if isinstance(self.successor_permno, bool) or not isinstance(
                self.successor_permno, (int, np.integer)
            ):
                raise CorporateActionError("successor_permno must be an integer")
            successor = int(self.successor_permno)
            if successor <= 0 or successor == source:
                raise CorporateActionError(
                    "successor_permno must be positive and differ from source"
                )
            object.__setattr__(self, "successor_permno", successor)


@dataclass(frozen=True, slots=True)
class CorporateActionLedger:
    """Immutable append-only action identity and chronology ledger."""

    applied_action_ids: tuple[str, ...] = ()
    last_assigned_return_date: dt.date | None = None
    last_action_sequence: int = 0

    def __post_init__(self) -> None:
        normalized = tuple(str(value).strip() for value in self.applied_action_ids)
        if any(not value for value in normalized):
            raise CorporateActionError("applied action IDs must be nonempty")
        if len(set(normalized)) != len(normalized):
            raise CorporateActionError("applied action IDs must be unique")
        object.__setattr__(self, "applied_action_ids", normalized)
        if self.last_assigned_return_date is None:
            if self.last_action_sequence != 0:
                raise CorporateActionError(
                    "empty corporate-action chronology must have sequence zero"
                )
        else:
            _date(
                self.last_assigned_return_date,
                name="last_assigned_return_date",
            )
            if (
                type(self.last_action_sequence) is not int
                or self.last_action_sequence < 1
            ):
                raise CorporateActionError(
                    "nonempty action chronology requires a positive sequence"
                )


@dataclass(frozen=True, slots=True)
class CorporateActionReceipt:
    action_id: str
    economic_event_date: dt.date
    assigned_return_date: dt.date
    action_sequence: int
    source_record_sha256: str
    daily_close_state_sha256: str
    total_return_payload_sha256: str
    source_permno: int
    settlement_kind: SettlementKind
    successor_permno: int | None
    opening_source_value_dollars: float
    settled_value_dollars: float
    terminal_sensitivity_policy_sha256: str | None
    terminal_sensitivity_evidence_sha256: str | None
    event_day_quoted_full_spread: float | None


@dataclass(frozen=True, slots=True)
class CorporateActionResult:
    state: PortfolioState
    ledger: CorporateActionLedger
    receipts: tuple[CorporateActionReceipt, ...]


def _retired_state(
    state: PortfolioState,
    *,
    source_index: int,
    successor_permno: int | None,
    settled_value: float,
    cash_settlement: bool,
) -> PortfolioState:
    # Retire the source row instead of retaining an unreturnable post-delisting
    # zero.  If the successor already exists, merge signed value into it.
    keep = np.ones(len(state.permno), dtype=bool)
    keep[source_index] = False
    permno = state.permno[keep].copy()
    values = state.position_value_dollars[keep].copy()
    cash = state.cash_dollars
    if cash_settlement:
        cash += settled_value
    else:
        assert successor_permno is not None
        successor = np.flatnonzero(permno == successor_permno)
        if successor.size:
            values[int(successor[0])] += settled_value
        else:
            permno = np.append(permno, np.int64(successor_permno))
            values = np.append(values, np.float64(settled_value))
    return PortfolioState(
        permno=permno,
        position_value_dollars=values,
        cash_dollars=cash,
    )


def _apply_one(
    state: PortfolioState,
    action: CorporateAction,
    close_receipt: DailyReturnCloseReceipt,
) -> tuple[PortfolioState, CorporateActionReceipt]:
    if (
        action.settlement_kind == STOCK_SUCCESSOR_SETTLEMENT
        and action.successor_permno is None
    ):
        raise UnresolvedCorporateActionError(
            f"stock-successor action {action.action_id!r} has no successor mapping"
        )
    matches = np.flatnonzero(state.permno == action.source_permno)
    if matches.size != 1:
        raise CorporateActionError(
            f"action source PERMNO {action.source_permno} is not in the portfolio state"
        )
    source_index = int(matches[0])
    opening_value = float(state.position_value_dollars[source_index])
    # A -100% terminal return legitimately leaves signed zero before closure.
    settled_value = opening_value * action.settlement_value_multiplier
    result = _retired_state(
        state,
        source_index=source_index,
        successor_permno=action.successor_permno,
        settled_value=settled_value,
        cash_settlement=action.settlement_kind == CASH_SETTLEMENT,
    )
    receipt = CorporateActionReceipt(
        action_id=action.action_id,
        economic_event_date=action.economic_event_date,
        assigned_return_date=action.assigned_return_date,
        action_sequence=action.action_sequence,
        source_record_sha256=action.source_record_sha256,
        daily_close_state_sha256=close_receipt.closing_state_sha256,
        total_return_payload_sha256=close_receipt.total_return_payload_sha256,
        source_permno=action.source_permno,
        settlement_kind=action.settlement_kind,
        successor_permno=action.successor_permno,
        opening_source_value_dollars=opening_value,
        settled_value_dollars=settled_value,
        terminal_sensitivity_policy_sha256=(action.terminal_sensitivity_policy_sha256),
        terminal_sensitivity_evidence_sha256=(
            action.terminal_sensitivity_evidence_sha256
        ),
        event_day_quoted_full_spread=action.event_day_quoted_full_spread,
    )
    return result, receipt


def apply_corporate_actions(
    state: PortfolioState,
    actions: Iterable[CorporateAction],
    *,
    close_receipt: DailyReturnCloseReceipt,
    ledger: CorporateActionLedger = CorporateActionLedger(),
) -> CorporateActionResult:
    """Apply an ordered action batch bound to one post-return close receipt."""

    if not isinstance(state, PortfolioState):
        raise TypeError("state must be PortfolioState")
    if not isinstance(ledger, CorporateActionLedger):
        raise TypeError("ledger must be CorporateActionLedger")
    if not isinstance(close_receipt, DailyReturnCloseReceipt):
        raise TypeError("close_receipt must be DailyReturnCloseReceipt")
    if close_receipt.closing_state_sha256 != portfolio_state_sha256(state):
        raise CorporateActionError(
            "daily close receipt does not bind the supplied post-return state"
        )
    normalized = tuple(actions)
    if any(not isinstance(action, CorporateAction) for action in normalized):
        raise TypeError("actions must contain only CorporateAction objects")
    identifiers = tuple(action.action_id for action in normalized)
    if len(set(identifiers)) != len(identifiers):
        raise CorporateActionError("one action batch contains duplicate action IDs")
    repeated = sorted(set(ledger.applied_action_ids).intersection(identifiers))
    if repeated:
        raise CorporateActionError(
            "corporate action has already been applied: " + ", ".join(repeated)
        )
    expected_order = tuple(
        sorted(
            normalized,
            key=lambda action: (
                action.assigned_return_date,
                action.action_sequence,
                action.action_id,
            ),
        )
    )
    if normalized != expected_order:
        raise CorporateActionError(
            "corporate actions must be ordered by assigned date, sequence, and ID"
        )
    if any(
        action.assigned_return_date != close_receipt.assigned_return_date
        for action in normalized
    ):
        raise CorporateActionError(
            "corporate action is not assigned to the authenticated daily close"
        )
    keys = tuple(
        (action.assigned_return_date, action.action_sequence) for action in normalized
    )
    if len(set(keys)) != len(keys):
        raise CorporateActionError("action date/sequence keys must be unique")
    prior_key = (
        ledger.last_assigned_return_date or dt.date.min,
        ledger.last_action_sequence,
    )
    if keys and keys[0] <= prior_key:
        raise CorporateActionError("corporate action chronology is backdated")

    current = state
    receipts: list[CorporateActionReceipt] = []
    for action in normalized:
        current, receipt = _apply_one(current, action, close_receipt)
        receipts.append(receipt)
    last_date = ledger.last_assigned_return_date
    last_sequence = ledger.last_action_sequence
    if normalized:
        last_date = normalized[-1].assigned_return_date
        last_sequence = normalized[-1].action_sequence
    return CorporateActionResult(
        state=current,
        ledger=CorporateActionLedger(
            applied_action_ids=ledger.applied_action_ids + identifiers,
            last_assigned_return_date=last_date,
            last_action_sequence=last_sequence,
        ),
        receipts=tuple(receipts),
    )


__all__ = [
    "CASH_SETTLEMENT",
    "STOCK_SUCCESSOR_SETTLEMENT",
    "CorporateAction",
    "CorporateActionError",
    "CorporateActionLedger",
    "CorporateActionReceipt",
    "CorporateActionResult",
    "DailyReturnCloseReceipt",
    "UnresolvedCorporateActionError",
    "apply_corporate_actions",
    "portfolio_state_sha256",
]
