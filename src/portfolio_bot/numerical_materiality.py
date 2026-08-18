"""Shared economic-materiality policy for numerical portfolio residue.

This policy is deliberately narrow.  It classifies only portfolio weights and
other already-dollar-denominated residuals; it does not relax schemas, keys,
dates, hashes, missing-input checks, risk constraints, or solver certificates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


NUMERICAL_MATERIALITY_POLICY_ID = (
    "portfolio-bot-strict-five-cent-materiality-v2"
)
STRICT_DOLLAR_MATERIALITY = 0.05
AGGREGATE_DOLLAR_MATERIALITY_LIMIT = 1.00
SOLVER_FEASIBILITY_ATOL = 5e-8


class NumericalMaterialityError(ValueError):
    """Raised when a materiality classification has invalid numeric inputs."""


@dataclass(frozen=True, slots=True)
class NumericalMaterialityAudit:
    """Closed audit of residue presented by one strategy-month solve.

    The one-dollar aggregate ceiling is fixed in dollars rather than inferred
    from observed returns.  At the thesis portfolio's $100 million initial
    NAV it is one hundred-millionth of NAV (0.0001 basis points), while still
    allowing several independent floating-point residues below five cents.
    """

    policy_id: str
    residue_count: int
    aggregate_absolute_dollars: float
    maximum_absolute_dollars: float
    strict_position_limit_dollars: float
    aggregate_limit_dollars: float

    def to_dict(self) -> dict[str, object]:
        """Return a stable machine-readable representation."""

        return {
            "policy_id": self.policy_id,
            "residue_count": self.residue_count,
            "aggregate_absolute_dollars": self.aggregate_absolute_dollars,
            "maximum_absolute_dollars": self.maximum_absolute_dollars,
            "strict_position_limit_dollars": (
                self.strict_position_limit_dollars
            ),
            "aggregate_limit_dollars": self.aggregate_limit_dollars,
        }


def _finite_vector(
    values: Sequence[float] | np.ndarray | float,
    *,
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NumericalMaterialityError(f"{name} must be numeric") from exc
    if not np.isfinite(result).all():
        raise NumericalMaterialityError(f"{name} must be finite")
    return result


def absolute_value_dollars(
    weight: Sequence[float] | np.ndarray | float,
    nav_dollars: Sequence[float] | np.ndarray | float,
) -> np.ndarray:
    """Return finite absolute position value for matching weights and NAV."""

    weights = _finite_vector(weight, name="weight")
    nav = _finite_vector(nav_dollars, name="nav_dollars")
    if nav.ndim != 0 and nav.shape != weights.shape:
        raise NumericalMaterialityError(
            "nav_dollars must be scalar or match weight shape"
        )
    if np.any(nav <= 0.0):
        raise NumericalMaterialityError("nav_dollars must be positive")
    return np.abs(weights) * nav


def is_immaterial_weight(
    weight: Sequence[float] | np.ndarray | float,
    nav_dollars: Sequence[float] | np.ndarray | float,
) -> np.ndarray:
    """Classify weights worth strictly less than five cents as immaterial."""

    return absolute_value_dollars(weight, nav_dollars) < (
        STRICT_DOLLAR_MATERIALITY
    )


def is_immaterial_dollar_amount(
    value_dollars: Sequence[float] | np.ndarray | float,
) -> np.ndarray:
    """Classify finite dollar residuals under the same strict threshold."""

    values = _finite_vector(value_dollars, name="value_dollars")
    return np.abs(values) < STRICT_DOLLAR_MATERIALITY


def audit_strategy_month_residue_dollars(
    value_dollars: Sequence[float] | np.ndarray | float,
) -> NumericalMaterialityAudit:
    """Audit the complete negligible-residue set for one strategy-month.

    Every supplied residue must be strictly below five cents in absolute
    value, and their aggregate absolute value must be at most one dollar.
    Empty sets are valid.  Callers must invoke this once on the complete set
    they intend to normalize; splitting a strategy-month into multiple calls
    would violate the policy contract.
    """

    values = _finite_vector(value_dollars, name="value_dollars")
    if values.ndim > 1:
        raise NumericalMaterialityError(
            "strategy-month residue must be scalar or one-dimensional"
        )
    absolute = np.abs(values).reshape(-1)
    if np.any(absolute >= STRICT_DOLLAR_MATERIALITY):
        raise NumericalMaterialityError(
            "each strategy-month residue must be strictly below five cents"
        )
    aggregate = math.fsum(float(value) for value in absolute)
    if aggregate > AGGREGATE_DOLLAR_MATERIALITY_LIMIT:
        raise NumericalMaterialityError(
            "strategy-month aggregate residue exceeds one dollar"
        )
    return NumericalMaterialityAudit(
        policy_id=NUMERICAL_MATERIALITY_POLICY_ID,
        residue_count=int(absolute.size),
        aggregate_absolute_dollars=float(aggregate),
        maximum_absolute_dollars=(
            float(np.max(absolute)) if absolute.size else 0.0
        ),
        strict_position_limit_dollars=STRICT_DOLLAR_MATERIALITY,
        aggregate_limit_dollars=AGGREGATE_DOLLAR_MATERIALITY_LIMIT,
    )


def audit_strategy_month_residue_weights(
    weight: Sequence[float] | np.ndarray | float,
    nav_dollars: Sequence[float] | np.ndarray | float,
) -> NumericalMaterialityAudit:
    """Convert one strategy-month residue set to dollars and audit it."""

    return audit_strategy_month_residue_dollars(
        absolute_value_dollars(weight, nav_dollars)
    )


def side_compatible(
    weight: Sequence[float] | np.ndarray | float,
    side: Sequence[int] | np.ndarray | int,
    nav_dollars: Sequence[float] | np.ndarray | float,
) -> np.ndarray:
    """Return whether each categorical side matches every material weight."""

    weights = _finite_vector(weight, name="weight")
    sides = _finite_vector(side, name="side")
    if sides.shape != weights.shape:
        raise NumericalMaterialityError("side must match weight shape")
    valid_side = np.isin(sides, (-1.0, 0.0, 1.0))
    compatible_sign = (
        (weights == 0.0)
        | ((weights > 0.0) & (sides == 1.0))
        | ((weights < 0.0) & (sides == -1.0))
    )
    return valid_side & (
        is_immaterial_weight(weights, nav_dollars) | compatible_sign
    )


__all__ = [
    "AGGREGATE_DOLLAR_MATERIALITY_LIMIT",
    "NUMERICAL_MATERIALITY_POLICY_ID",
    "NumericalMaterialityAudit",
    "NumericalMaterialityError",
    "SOLVER_FEASIBILITY_ATOL",
    "STRICT_DOLLAR_MATERIALITY",
    "absolute_value_dollars",
    "audit_strategy_month_residue_dollars",
    "audit_strategy_month_residue_weights",
    "is_immaterial_dollar_amount",
    "is_immaterial_weight",
    "side_compatible",
]
