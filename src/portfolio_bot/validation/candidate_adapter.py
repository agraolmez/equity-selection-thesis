"""Stage 11 entry gating around the hash-frozen Stage 10 candidate domain.

The frozen Stage 10 universe deliberately uses ``long_eligible`` for the full
monthly scoring cross-section.  Stage 11 can discover a later execution/risk
reason that prevents opening a position without changing that cross-section.
This adapter therefore preserves the frozen score order and quintile
boundaries, while applying separate entry-allowed masks only to candidate
assignment.
"""

from __future__ import annotations

import math

import numpy as np

from portfolio_bot.allocation.candidates import (
    DEFAULT_MINIMUM_CANDIDATES,
    DEFAULT_TAIL_FRACTION,
    _joint_disjoint_tail_assignment,
    _ordered_indices,
    _tail_then_scan,
    rerank_short_borrow_liquidity,
)
from portfolio_bot.allocation.domain import (
    ACTIVE,
    INSUFFICIENT_DISJOINT_CANDIDATES,
    LONG_SIDE,
    SHORT_SIDE,
    CandidateSelection,
    CandidateUniverse,
)


def _readonly(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _entry_mask(value: object, *, name: str, size: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) != size or raw.dtype.kind not in {"b", "i", "u"}:
        raise ValueError(f"{name} must be a length-{size} boolean vector")
    if raw.dtype.kind != "b" and not np.isin(raw, (0, 1)).all():
        raise ValueError(f"{name} must contain only boolean values")
    return np.asarray(raw, dtype=np.bool_)


def form_stage11_monthly_candidates(
    universe: CandidateUniverse,
    *,
    long_entry_allowed: np.ndarray,
    short_entry_allowed: np.ndarray,
    minimum_candidates: int = DEFAULT_MINIMUM_CANDIDATES,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
) -> CandidateSelection:
    """Form candidates without changing the authenticated score cross-section."""

    if not isinstance(universe, CandidateUniverse):
        raise TypeError("universe must be CandidateUniverse")
    if isinstance(minimum_candidates, bool) or not isinstance(minimum_candidates, int):
        raise ValueError("minimum_candidates must be an integer")
    if minimum_candidates <= 0:
        raise ValueError("minimum_candidates must be positive")
    if not np.isfinite(tail_fraction) or not 0.0 < float(tail_fraction) < 0.5:
        raise ValueError("tail_fraction must be finite and within (0, 0.5)")

    long_entry = _entry_mask(
        long_entry_allowed,
        name="long_entry_allowed",
        size=universe.n_assets,
    )
    short_entry = _entry_mask(
        short_entry_allowed,
        name="short_entry_allowed",
        size=universe.n_assets,
    )
    if np.any(long_entry & ~universe.long_eligible):
        raise ValueError("long_entry_allowed must be a subset of long_eligible")
    if np.any(short_entry & ~universe.short_eligible):
        raise ValueError("short_entry_allowed must be a subset of short_eligible")
    if np.any(short_entry & ~long_entry):
        raise ValueError("short_entry_allowed must be a subset of long_entry_allowed")

    scoring_count = int(universe.long_eligible.sum())
    tail_size = max(1, int(math.ceil(scoring_count * float(tail_fraction))))
    long_order = _ordered_indices(universe, descending=True)
    short_order = _ordered_indices(universe, descending=False)
    long_boundary = (
        float(universe.score[long_order[tail_size - 1]]) if scoring_count else None
    )
    short_boundary = (
        float(universe.score[short_order[tail_size - 1]]) if scoring_count else None
    )

    long_allowed = (
        long_entry
        & ~universe.required_exit
        & (universe.previous_weight >= 0.0)
    )
    short_allowed = (
        short_entry
        & ~universe.required_exit
        & (universe.previous_weight <= 0.0)
    )
    proposed_long = _tail_then_scan(
        long_order,
        tail_size=tail_size,
        allowed=long_allowed,
        minimum=minimum_candidates,
    )
    proposed_short = _tail_then_scan(
        short_order,
        tail_size=tail_size,
        allowed=short_allowed,
        minimum=minimum_candidates,
    )
    joint_long, joint_short, joint_feasible = _joint_disjoint_tail_assignment(
        long_order=long_order,
        short_order=short_order,
        tail_size=tail_size,
        long_allowed=long_allowed,
        short_allowed=short_allowed,
        minimum=minimum_candidates,
    )
    diagnostic_available_long = int(proposed_long.sum())
    diagnostic_available_short = int(proposed_short.sum())
    diagnostic_overlap_count = int(np.sum(proposed_long & proposed_short))
    active = (
        joint_feasible
        and int(joint_long.sum()) >= minimum_candidates
        and int(joint_short.sum()) >= minimum_candidates
    )
    if active:
        long_candidate = joint_long
        short_candidate = joint_short
        available_long = int(long_candidate.sum())
        available_short = int(short_candidate.sum())
        overlap_count = 0
        status = ACTIVE
    else:
        long_candidate = np.zeros(universe.n_assets, dtype=np.bool_)
        short_candidate = np.zeros(universe.n_assets, dtype=np.bool_)
        available_long = diagnostic_available_long
        available_short = diagnostic_available_short
        overlap_count = diagnostic_overlap_count
        status = INSUFFICIENT_DISJOINT_CANDIDATES

    held = universe.previous_weight != 0.0
    selected = long_candidate | short_candidate | held
    side = np.zeros(universe.n_assets, dtype=np.int8)
    side[long_candidate] = LONG_SIDE
    side[short_candidate] = SHORT_SIDE
    side[universe.previous_weight > 0.0] = LONG_SIDE
    side[universe.previous_weight < 0.0] = SHORT_SIDE
    if active:
        same_side_candidate = ((universe.previous_weight > 0.0) & long_candidate) | (
            (universe.previous_weight < 0.0) & short_candidate
        )
        reduction_only = held & ~same_side_candidate
        reduction_only |= universe.required_exit
    else:
        reduction_only = held.copy()

    short_union = short_candidate | (universe.previous_weight < 0.0)
    borrow_quintile = rerank_short_borrow_liquidity(
        universe.permno,
        universe.adv20_usd,
        short_union,
    )
    return CandidateSelection(
        universe=universe,
        long_candidate=_readonly(long_candidate, np.bool_),
        short_candidate=_readonly(short_candidate, np.bool_),
        selected=_readonly(selected, np.bool_),
        side=_readonly(side, np.int8),
        reduction_only=_readonly(reduction_only, np.bool_),
        borrow_liquidity_quintile=borrow_quintile,
        long_entry_boundary=long_boundary,
        short_entry_boundary=short_boundary,
        tail_size=tail_size,
        minimum_candidates=minimum_candidates,
        available_long_candidates=available_long,
        available_short_candidates=available_short,
        overlap_count=overlap_count,
        status=status,
    )


__all__ = ["form_stage11_monthly_candidates"]
