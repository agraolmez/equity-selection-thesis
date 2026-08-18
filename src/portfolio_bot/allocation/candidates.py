"""Frozen deterministic monthly candidate formation for roadmap Stage 10."""

from __future__ import annotations

import math

import numpy as np

from portfolio_bot.allocation.domain import (
    ACTIVE,
    INSUFFICIENT_DISJOINT_CANDIDATES,
    LONG_SIDE,
    SHORT_SIDE,
    CandidateSelection,
    CandidateUniverse,
)


DEFAULT_MINIMUM_CANDIDATES = 50
DEFAULT_TAIL_FRACTION = 0.20


def _readonly(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _ordered_indices(
    universe: CandidateUniverse,
    *,
    descending: bool,
) -> np.ndarray:
    scoring = np.flatnonzero(universe.long_eligible)
    score = universe.score[scoring]
    primary = -score if descending else score
    order = np.lexsort((universe.permno[scoring], primary))
    return scoring[order]


def _tail_then_scan(
    order: np.ndarray,
    *,
    tail_size: int,
    allowed: np.ndarray,
    minimum: int,
) -> np.ndarray:
    selected = np.zeros(len(allowed), dtype=np.bool_)
    initial = order[:tail_size]
    selected[initial[allowed[initial]]] = True
    if int(selected.sum()) < minimum:
        for index in order:
            if allowed[index] and not selected[index]:
                selected[index] = True
                if int(selected.sum()) >= minimum:
                    break
    return selected


def _joint_disjoint_tail_assignment(
    *,
    long_order: np.ndarray,
    short_order: np.ndarray,
    tail_size: int,
    long_allowed: np.ndarray,
    short_allowed: np.ndarray,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Find the deterministic extreme-first disjoint side assignment.

    Valid positional-tail names are fixed to their natural side first.  The
    remaining minimum-count slots alternate long then short.  Each slot scans
    its side's score/PERMNO order and, on collision, follows a deterministic
    augmenting path that moves an earlier *extension* slot to its next-best
    feasible name.  Fixed tail assignments are never displaced.

    This is ordinary bipartite maximum matching with stable vertex and edge
    order.  It therefore finds a disjoint assignment whenever one exists,
    unlike two independent greedy scans followed by overlap suppression.
    """

    n = len(long_allowed)
    fixed_long = np.zeros(n, dtype=np.bool_)
    fixed_short = np.zeros(n, dtype=np.bool_)
    initial_long = long_order[:tail_size]
    initial_short = short_order[:tail_size]
    fixed_long[initial_long[long_allowed[initial_long]]] = True
    fixed_short[initial_short[short_allowed[initial_short]]] = True
    if np.any(fixed_long & fixed_short):
        return fixed_long, fixed_short, False

    long_deficit = max(0, minimum - int(fixed_long.sum()))
    short_deficit = max(0, minimum - int(fixed_short.sum()))
    slot_side: list[int] = []
    for slot_number in range(max(long_deficit, short_deficit)):
        if slot_number < long_deficit:
            slot_side.append(int(LONG_SIDE))
        if slot_number < short_deficit:
            slot_side.append(int(SHORT_SIDE))

    fixed_any = fixed_long | fixed_short
    preferences = {
        int(LONG_SIDE): [
            int(index)
            for index in long_order
            if long_allowed[index] and not fixed_any[index]
        ],
        int(SHORT_SIDE): [
            int(index)
            for index in short_order
            if short_allowed[index] and not fixed_any[index]
        ],
    }
    slot_to_asset: dict[int, int] = {}
    asset_to_slot: dict[int, int] = {}

    def augment(slot: int, seen_assets: set[int]) -> bool:
        side = slot_side[slot]
        for asset in preferences[side]:
            if asset in seen_assets:
                continue
            seen_assets.add(asset)
            incumbent = asset_to_slot.get(asset)
            if incumbent is None or augment(incumbent, seen_assets):
                slot_to_asset[slot] = asset
                asset_to_slot[asset] = slot
                return True
        return False

    for slot in range(len(slot_side)):
        augment(slot, set())

    if len(slot_to_asset) != len(slot_side):
        return fixed_long, fixed_short, False

    long_candidate = fixed_long.copy()
    short_candidate = fixed_short.copy()
    for slot, asset in slot_to_asset.items():
        if slot_side[slot] == int(LONG_SIDE):
            long_candidate[asset] = True
        else:
            short_candidate[asset] = True
    if np.any(long_candidate & short_candidate):  # defensive matching invariant
        raise RuntimeError("augmenting-path assignment produced overlapping sides")
    return long_candidate, short_candidate, True


def rerank_short_borrow_liquidity(
    permno: np.ndarray,
    adv20_usd: np.ndarray,
    current_short_union: np.ndarray,
) -> np.ndarray:
    """Rerank candidates plus inherited shorts into frozen ADV quintiles.

    Missing and nonpositive ADV are least liquid and tie-break by PERMNO.
    Positive ADV then sorts ascending by ADV and PERMNO.  Quintile one is the
    least-liquid group and non-short rows receive zero.
    """

    permnos = np.asarray(permno)
    adv = np.asarray(adv20_usd, dtype=np.float64)
    mask = np.asarray(current_short_union)
    if permnos.ndim != 1 or adv.ndim != 1 or mask.ndim != 1:
        raise ValueError("borrow-rank inputs must be one-dimensional")
    if len(permnos) != len(adv) or len(permnos) != len(mask):
        raise ValueError("borrow-rank input lengths differ")
    if mask.dtype.kind not in {"b", "i", "u"} or (
        mask.dtype.kind != "b" and not np.isin(mask, (0, 1)).all()
    ):
        raise ValueError("current_short_union must be boolean")
    try:
        normalized_permno = np.asarray(permnos, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("permno must contain integers") from exc
    if np.any(normalized_permno <= 0) or len(np.unique(normalized_permno)) != len(
        normalized_permno
    ):
        raise ValueError("permno must be positive and unique")
    if np.isinf(adv).any():
        raise ValueError("adv20_usd may be finite or NaN, but not infinite")
    selected = np.flatnonzero(mask.astype(np.bool_))
    result = np.zeros(len(mask), dtype=np.int8)
    if selected.size == 0:
        return _readonly(result, np.int8)

    def key(index: int) -> tuple[int, float, int]:
        value = float(adv[index])
        if not math.isfinite(value) or value <= 0.0:
            return (0, 0.0, int(normalized_permno[index]))
        return (1, value, int(normalized_permno[index]))

    ordered = sorted((int(value) for value in selected), key=key)
    count = len(ordered)
    for rank, index in enumerate(ordered):
        result[index] = math.floor(5 * rank / count) + 1
    return _readonly(result, np.int8)


def form_monthly_candidates(
    universe: CandidateUniverse,
    *,
    minimum_candidates: int = DEFAULT_MINIMUM_CANDIDATES,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
) -> CandidateSelection:
    """Form fixed-sign long/short candidates and inherited holding rows.

    Quintile membership is positional within the complete monthly scoring
    universe.  Scores sort first and PERMNO breaks every tie.  When a tail has
    fewer than ``minimum_candidates`` after eligibility/sign exclusions, the
    scan continues toward the middle.  If both sides cannot supply the minimum
    as disjoint sets, all new candidates are suppressed and inherited holdings
    become reduction-only.
    """

    if not isinstance(universe, CandidateUniverse):
        raise TypeError("universe must be CandidateUniverse")
    if isinstance(minimum_candidates, bool) or not isinstance(minimum_candidates, int):
        raise ValueError("minimum_candidates must be an integer")
    if minimum_candidates <= 0:
        raise ValueError("minimum_candidates must be positive")
    if not np.isfinite(tail_fraction) or not 0.0 < float(tail_fraction) < 0.5:
        raise ValueError("tail_fraction must be finite and within (0, 0.5)")

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

    # No same-rebalance sign reversal.  Required exits can only move to zero.
    long_allowed = (
        universe.long_eligible
        & ~universe.required_exit
        & (universe.previous_weight >= 0.0)
    )
    short_allowed = (
        universe.short_eligible
        & ~universe.required_exit
        & (universe.previous_weight <= 0.0)
    )
    # Retain the independent scans only as failure diagnostics.  Actual
    # assignment is joint: deterministic augmenting paths repair collisions.
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


__all__ = [
    "DEFAULT_MINIMUM_CANDIDATES",
    "DEFAULT_TAIL_FRACTION",
    "form_monthly_candidates",
    "rerank_short_borrow_liquidity",
]
