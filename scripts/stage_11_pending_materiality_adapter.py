"""Stage 11-only pending-dust materiality and checkpoint-tip adapter.

The protected accounting implementation and the authenticated diagnostic
checkpoint chain remain unchanged.  This adapter broadens only the existing
pending/no-mark dust retirement policy from a strict one-cent bound to a
strict five-cent bound.  The bound corresponds to at most the existing
``5e-8`` normalized required-exit feasibility scale at the Stage 11 initial
NAV.

Checkpoint 045 predates the broadened bound and contains one authenticated
pending numerical residue.  A narrowly scoped restored-state hook therefore
normalizes that exact tip in memory before month 046 is constructed.  It does
not rewrite the checkpoint, advance the month, fabricate a return, or loosen
return-span continuity.
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import portfolio_bot.validation.empirical_portfolio_backend as portfolio_backend_module
from portfolio_bot.numerical_materiality import (
    SOLVER_FEASIBILITY_ATOL,
    STRICT_DOLLAR_MATERIALITY,
)


PENDING_MATERIALITY_POLICY_ID = (
    "stage11-pending-five-cent-dust-retirement-v2"
)
PENDING_MATERIALITY_THRESHOLD_DOLLARS = STRICT_DOLLAR_MATERIALITY
PENDING_MATERIALITY_NORMALIZED_REQUIRED_EXIT_LIMIT = SOLVER_FEASIBILITY_ATOL

MIGRATION_SOURCE_CHECKPOINT_SHA256 = (
    "194b05c1bdcc29ef5c771de9388e10dcb9e6a17c1a6a64613574dfc4ad4ad7ea"
)
MIGRATION_SOURCE_COMPLETED_MONTH = "2018-09"
MIGRATION_NEXT_MONTH_INDEX = 45
MIGRATION_NEXT_HOLDING_MONTH = "2018-10"
MIGRATION_BOUNDARY_DATE = dt.date(2018, 10, 1)
MIGRATION_STRATEGY_ID = "elastic_net_hybrid_challenger"
MIGRATION_PERMNO = 91_366
MIGRATION_POSITION_VALUE_DOLLARS_HEX = "0x1.68a9c50792833p-6"
MIGRATION_OPENING_NAV_DOLLARS_HEX = "0x1.806a243fe0569p+26"
MIGRATION_LAST_AUTHORITATIVE_MARK_DATE = dt.date(2018, 5, 11)
MIGRATION_PENDING_NO_MARK_COUNT = 98
MIGRATION_PENDING_EVIDENCE_SHA256 = (
    "7898097a3f8a7aa003978f99d77bf900cb5c10edec5debb1798c57a853d7906e"
)
MIGRATION_EVIDENCE_SHA256 = (
    "c46f4cc4e29e7576c023a310b368c7d32c7ab9e35db4844d6376f112cbe554cc"
)

_EXPECTED_STRATEGY_IDS = (
    "elastic_net_hybrid_challenger",
    "ensemble_hybrid_primary",
    "ensemble_rank_challenger",
)
_CANONICAL_POLICY_ID = (
    portfolio_backend_module.PENDING_DUST_RETIREMENT_POLICY_ID
)
_CANONICAL_THRESHOLD_DOLLARS = (
    portfolio_backend_module.PENDING_DUST_RETIREMENT_THRESHOLD_DOLLARS
)
_CANONICAL_RESTORE = (
    portfolio_backend_module._restore_portfolio_diagnostic_state
)


class Stage11PendingMaterialityError(RuntimeError):
    """Raised before an unauthenticated materiality transition can apply."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11PendingMaterialityError(message)


@dataclass(slots=True)
class Stage11PendingMaterialityAudit:
    """Runtime evidence for the compatibility-preserving tip transition."""

    diagnostic_session: Any | None
    opening_checkpoint_sha256: str | None = None
    transition_application_count: int = 0
    transition_evidence: tuple[Any, ...] = ()

    def gate(self) -> None:
        expected = (
            self.opening_checkpoint_sha256
            == MIGRATION_SOURCE_CHECKPOINT_SHA256
        )
        _require(
            self.transition_application_count == (1 if expected else 0),
            "pending-materiality checkpoint transition count differs",
        )
        _require(
            (len(self.transition_evidence) == 1) is expected,
            "pending-materiality checkpoint evidence count differs",
        )


def _migrate_restored_states(
    states: dict[str, Any],
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    _require(
        tuple(sorted(states)) == _EXPECTED_STRATEGY_IDS,
        "pending-materiality migration strategy state differs",
    )

    migrated_states = dict(states)
    evidence: list[Any] = []
    for strategy_id in _EXPECTED_STRATEGY_IDS:
        migrated, items = (
            portfolio_backend_module
            ._retire_pending_subcent_dust_at_month_boundary(
                strategy_id=strategy_id,
                holding_month=MIGRATION_SOURCE_COMPLETED_MONTH,
                boundary_date=MIGRATION_BOUNDARY_DATE,
                state=migrated_states[strategy_id],
            )
        )
        migrated_states[strategy_id] = migrated
        evidence.extend(items)

    _require(
        len(evidence) == 1,
        "pending-materiality migration candidate set differs",
    )
    item = evidence[0]
    normalized_required_exit_violation = (
        abs(item.position_value_dollars) / item.opening_nav_dollars * 100.0
    )
    _require(
        item.policy_id == PENDING_MATERIALITY_POLICY_ID
        and item.strategy_id == MIGRATION_STRATEGY_ID
        and item.holding_month == MIGRATION_SOURCE_COMPLETED_MONTH
        and item.boundary_date == MIGRATION_BOUNDARY_DATE
        and item.permno == MIGRATION_PERMNO
        and item.position_value_dollars.hex()
        == MIGRATION_POSITION_VALUE_DOLLARS_HEX
        and item.threshold_dollars
        == PENDING_MATERIALITY_THRESHOLD_DOLLARS
        and item.opening_nav_dollars.hex()
        == MIGRATION_OPENING_NAV_DOLLARS_HEX
        and item.opening_nav_dollars == item.closing_nav_dollars
        and item.last_authoritative_mark_date
        == MIGRATION_LAST_AUTHORITATIVE_MARK_DATE
        and item.pending_no_mark_count == MIGRATION_PENDING_NO_MARK_COUNT
        and item.pending_evidence_sha256
        == MIGRATION_PENDING_EVIDENCE_SHA256
        and item.evidence_sha256 == MIGRATION_EVIDENCE_SHA256
        and normalized_required_exit_violation
        < PENDING_MATERIALITY_NORMALIZED_REQUIRED_EXIT_LIMIT
        and not item.execution_price_read
        and not item.return_imputed
        and not item.trade_emitted
        and not item.transaction_cost_imputed
        and not item.corporate_action_applied
        and not item.identity_terminally_retired,
        "pending-materiality migration evidence differs",
    )
    return migrated_states, tuple(evidence)


@contextmanager
def stage11_pending_materiality_scope(
    *,
    diagnostic_session: Any | None = None,
) -> Iterator[Stage11PendingMaterialityAudit]:
    """Install the reviewed five-cent policy and optional tip migration."""

    _require(
        portfolio_backend_module.PENDING_DUST_RETIREMENT_POLICY_ID
        == _CANONICAL_POLICY_ID
        and portfolio_backend_module.PENDING_DUST_RETIREMENT_THRESHOLD_DOLLARS
        == _CANONICAL_THRESHOLD_DOLLARS
        and portfolio_backend_module._restore_portfolio_diagnostic_state
        is _CANONICAL_RESTORE
        and _CANONICAL_POLICY_ID
        == "stage11-pending-subcent-dust-retirement-v1"
        and _CANONICAL_THRESHOLD_DOLLARS == 0.01,
        "pending-materiality backend binding changed before amendment",
    )
    opening_checkpoint_sha256 = None
    if diagnostic_session is not None:
        opening = diagnostic_session.load_latest_checkpoint()
        if opening.completed_checkpoint is not None:
            opening_checkpoint_sha256 = (
                opening.completed_checkpoint.checkpoint_sha256
            )
        if (
            opening_checkpoint_sha256
            == MIGRATION_SOURCE_CHECKPOINT_SHA256
        ):
            checkpoint = opening.completed_checkpoint
            assert checkpoint is not None
            _require(
                checkpoint.ordinal == MIGRATION_NEXT_MONTH_INDEX
                and checkpoint.holding_month
                == MIGRATION_SOURCE_COMPLETED_MONTH
                and opening.next_month_index == MIGRATION_NEXT_MONTH_INDEX
                and opening.next_holding_month
                == MIGRATION_NEXT_HOLDING_MONTH
                and set(opening.state_metadata)
                == {
                    "completed_month",
                    "next_month_index",
                    "active_strategy_ids",
                    "qualifies",
                }
                and opening.state_metadata["completed_month"]
                == MIGRATION_SOURCE_COMPLETED_MONTH
                and opening.state_metadata["next_month_index"]
                == MIGRATION_NEXT_MONTH_INDEX
                and tuple(opening.state_metadata["active_strategy_ids"])
                == _EXPECTED_STRATEGY_IDS
                and opening.state_metadata["qualifies"] is True,
                "pending-materiality migration source checkpoint differs",
            )
    audit = Stage11PendingMaterialityAudit(
        diagnostic_session=diagnostic_session,
        opening_checkpoint_sha256=opening_checkpoint_sha256,
    )

    def restore_adapter(payload: Any):
        states, risk_memory, retired_on = _CANONICAL_RESTORE(payload)
        if (
            audit.opening_checkpoint_sha256
            != MIGRATION_SOURCE_CHECKPOINT_SHA256
        ):
            return states, risk_memory, retired_on
        _require(
            audit.transition_application_count == 0,
            "pending-materiality checkpoint transition was reapplied",
        )
        migrated_states, evidence = _migrate_restored_states(states)
        audit.transition_application_count = 1
        audit.transition_evidence = evidence
        return migrated_states, risk_memory, retired_on

    portfolio_backend_module.PENDING_DUST_RETIREMENT_POLICY_ID = (
        PENDING_MATERIALITY_POLICY_ID
    )
    portfolio_backend_module.PENDING_DUST_RETIREMENT_THRESHOLD_DOLLARS = (
        PENDING_MATERIALITY_THRESHOLD_DOLLARS
    )
    portfolio_backend_module._restore_portfolio_diagnostic_state = (
        restore_adapter
    )
    try:
        yield audit
        audit.gate()
    finally:
        changed = (
            portfolio_backend_module.PENDING_DUST_RETIREMENT_POLICY_ID
            != PENDING_MATERIALITY_POLICY_ID
            or portfolio_backend_module
            .PENDING_DUST_RETIREMENT_THRESHOLD_DOLLARS
            != PENDING_MATERIALITY_THRESHOLD_DOLLARS
            or portfolio_backend_module._restore_portfolio_diagnostic_state
            is not restore_adapter
        )
        portfolio_backend_module.PENDING_DUST_RETIREMENT_POLICY_ID = (
            _CANONICAL_POLICY_ID
        )
        portfolio_backend_module.PENDING_DUST_RETIREMENT_THRESHOLD_DOLLARS = (
            _CANONICAL_THRESHOLD_DOLLARS
        )
        portfolio_backend_module._restore_portfolio_diagnostic_state = (
            _CANONICAL_RESTORE
        )
        _require(
            not changed,
            "pending-materiality backend binding changed during amendment",
        )


__all__ = [
    "MIGRATION_BOUNDARY_DATE",
    "MIGRATION_EVIDENCE_SHA256",
    "MIGRATION_LAST_AUTHORITATIVE_MARK_DATE",
    "MIGRATION_NEXT_HOLDING_MONTH",
    "MIGRATION_NEXT_MONTH_INDEX",
    "MIGRATION_OPENING_NAV_DOLLARS_HEX",
    "MIGRATION_PENDING_EVIDENCE_SHA256",
    "MIGRATION_PENDING_NO_MARK_COUNT",
    "MIGRATION_PERMNO",
    "MIGRATION_POSITION_VALUE_DOLLARS_HEX",
    "MIGRATION_SOURCE_CHECKPOINT_SHA256",
    "MIGRATION_SOURCE_COMPLETED_MONTH",
    "MIGRATION_STRATEGY_ID",
    "PENDING_MATERIALITY_NORMALIZED_REQUIRED_EXIT_LIMIT",
    "PENDING_MATERIALITY_POLICY_ID",
    "PENDING_MATERIALITY_THRESHOLD_DOLLARS",
    "Stage11PendingMaterialityAudit",
    "Stage11PendingMaterialityError",
    "stage11_pending_materiality_scope",
]
