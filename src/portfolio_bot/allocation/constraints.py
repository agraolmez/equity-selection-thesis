"""Common hard constraints and independent diagnostics for Stage 10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cvxpy as cp
import numpy as np

from portfolio_bot.allocation.domain import AllocationProblemData


GROSS_LIMIT = 2.0
NET_NAV_LIMIT = 0.10
NET_GROSS_FRACTION = 0.05
LONG_CAP = 0.0100
SHORT_CAP = 0.0075
ANNUAL_VARIANCE_LIMIT = 0.10**2
FACTOR_EXPOSURE_LIMIT = 0.10
SECTOR_NET_LIMIT = 0.05
SECTOR_GROSS_LIMIT = 0.30
COUNTRY_NET_LIMIT = 0.05
FOREIGN_NET_LIMIT = 0.10
ONE_WAY_TURNOVER_LIMIT = 0.50


@dataclass(frozen=True)
class ConstraintSpec:
    """One convex upper-bound constraint eligible for inherited slack."""

    name: str
    expression: cp.Expression
    limit: float
    normalizer: float
    pretrade_left: float

    @property
    def pretrade_normalized_violation(self) -> float:
        return max((self.pretrade_left - self.limit) / self.normalizer, 0.0)


@dataclass(frozen=True)
class ConstraintDiagnostics:
    """Optimizer-independent recomputation of every frozen constraint family."""

    long_gross: float
    short_gross: float
    gross: float
    net: float
    one_way_turnover: float
    augmented_annual_variance: float
    augmented_annual_volatility: float
    factor_exposures: tuple[float, ...]
    violations: dict[str, float]
    normalized_violations: dict[str, float]

    @property
    def maximum_violation(self) -> float:
        return max((max(value, 0.0) for value in self.violations.values()), default=0.0)

    @property
    def maximum_normalized_violation(self) -> float:
        return max(
            (max(value, 0.0) for value in self.normalized_violations.values()),
            default=0.0,
        )


def _indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def _sum(expression: cp.Expression, indices: np.ndarray) -> cp.Expression:
    if indices.size == 0:
        return cp.Constant(0.0)
    return cp.sum(expression[indices])


def _numeric_sum(values: np.ndarray, indices: np.ndarray) -> float:
    return float(np.sum(values[indices])) if indices.size else 0.0


def _group_masks(values: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    for label in sorted(set(str(value) for value in values.tolist())):
        yield label, np.asarray(values == label, dtype=np.bool_)


def hard_trade_and_sign_constraints(
    data: AllocationProblemData,
    weights: cp.Variable,
) -> list[cp.Constraint]:
    """Constraints that are never relaxed, including in emergency mode."""

    previous = data.previous_weight
    delta = weights - previous
    long_indices = _indices(data.long_mask)
    short_indices = _indices(data.short_mask)
    constraints: list[cp.Constraint] = []
    if long_indices.size:
        constraints.append(weights[long_indices] >= 0.0)
    if short_indices.size:
        constraints.append(weights[short_indices] <= 0.0)
    for index in _indices(data.reduction_only):
        if data.side[index] > 0:
            constraints.append(weights[index] <= previous[index])
        else:
            constraints.append(weights[index] >= previous[index])
    valid_adv = np.isfinite(data.adv20_usd) & (data.adv20_usd > 0.0)
    adv_capacity = np.where(
        valid_adv,
        0.10 * data.adv20_usd / float(data.nav_dollars),
        0.0,
    )
    constraints.extend(
        [
            cp.abs(delta) <= data.trade_capacity_weight,
            # Recompute the frozen ten-percent-ADV ceiling here rather than
            # trusting a supplied aggregate capacity field alone.
            cp.abs(delta) <= adv_capacity,
            0.5 * cp.norm1(delta) <= ONE_WAY_TURNOVER_LIMIT,
        ]
    )
    return constraints


def slackable_constraint_specs(
    data: AllocationProblemData,
    weights: cp.Variable,
) -> list[ConstraintSpec]:
    """Build the exact frozen target constraints and their pretrade values."""

    previous = data.previous_weight
    long_indices = _indices(data.long_mask)
    short_indices = _indices(data.short_mask)
    long_gross = _sum(weights, long_indices)
    short_gross = -_sum(weights, short_indices)
    gross = long_gross + short_gross
    net = long_gross - short_gross
    pre_long = _numeric_sum(previous, long_indices)
    pre_short = -_numeric_sum(previous, short_indices)
    pre_gross = pre_long + pre_short
    pre_net = pre_long - pre_short

    specs: list[ConstraintSpec] = [
        ConstraintSpec("gross", gross, GROSS_LIMIT, GROSS_LIMIT, pre_gross),
        ConstraintSpec("net_positive", net, NET_NAV_LIMIT, NET_NAV_LIMIT, pre_net),
        ConstraintSpec("net_negative", -net, NET_NAV_LIMIT, NET_NAV_LIMIT, -pre_net),
        # These are exactly N <= .05G and -N <= .05G.  The zero RHS after
        # rearrangement has no usable scale, so inherited excess is measured
        # in NAV units against the separate 10% net safety band.
        ConstraintSpec(
            "dynamic_net_positive",
            0.95 * long_gross - 1.05 * short_gross,
            0.0,
            NET_NAV_LIMIT,
            0.95 * pre_long - 1.05 * pre_short,
        ),
        ConstraintSpec(
            "dynamic_net_negative",
            0.95 * short_gross - 1.05 * long_gross,
            0.0,
            NET_NAV_LIMIT,
            0.95 * pre_short - 1.05 * pre_long,
        ),
    ]

    for index in long_indices:
        specs.append(
            ConstraintSpec(
                f"position_long:{int(data.permno[index])}",
                weights[index],
                LONG_CAP,
                LONG_CAP,
                float(previous[index]),
            )
        )
        if data.required_exit[index]:
            # A tradable ineligible holding has a zero target.  Keeping this
            # as a separate position-cap-style constraint makes zero exact in
            # ordinary mode while allowing only its inherited amount to be
            # grandfathered by the emergency hierarchy.
            specs.append(
                ConstraintSpec(
                    f"required_exit_long:{int(data.permno[index])}",
                    weights[index],
                    0.0,
                    LONG_CAP,
                    float(previous[index]),
                )
            )
    for index in short_indices:
        specs.append(
            ConstraintSpec(
                f"position_short:{int(data.permno[index])}",
                -weights[index],
                SHORT_CAP,
                SHORT_CAP,
                float(-previous[index]),
            )
        )
        if data.required_exit[index]:
            specs.append(
                ConstraintSpec(
                    f"required_exit_short:{int(data.permno[index])}",
                    -weights[index],
                    0.0,
                    SHORT_CAP,
                    float(-previous[index]),
                )
            )

    for issuer in sorted(set(int(value) for value in data.permco.tolist())):
        issuer_mask = data.permco == issuer
        issuer_long = _indices(issuer_mask & data.long_mask)
        issuer_short = _indices(issuer_mask & data.short_mask)
        if issuer_long.size:
            specs.append(
                ConstraintSpec(
                    f"issuer_long:{issuer}",
                    _sum(weights, issuer_long),
                    LONG_CAP,
                    LONG_CAP,
                    _numeric_sum(previous, issuer_long),
                )
            )
        if issuer_short.size:
            specs.append(
                ConstraintSpec(
                    f"issuer_short:{issuer}",
                    -_sum(weights, issuer_short),
                    SHORT_CAP,
                    SHORT_CAP,
                    -_numeric_sum(previous, issuer_short),
                )
            )

    annual_variance = 12.0 * cp.sum_squares(data.risk_root @ weights)
    pre_annual_variance = 12.0 * float(
        np.dot(data.risk_root @ previous, data.risk_root @ previous)
    )
    specs.append(
        ConstraintSpec(
            "augmented_risk",
            annual_variance,
            ANNUAL_VARIANCE_LIMIT,
            ANNUAL_VARIANCE_LIMIT,
            pre_annual_variance,
        )
    )

    factor_exposures = data.factor_betas.T @ weights
    pre_factor = data.factor_betas.T @ previous
    factor_names = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
    for ordinal, name in enumerate(factor_names):
        specs.extend(
            [
                ConstraintSpec(
                    f"factor_positive:{name}",
                    factor_exposures[ordinal],
                    FACTOR_EXPOSURE_LIMIT,
                    FACTOR_EXPOSURE_LIMIT,
                    float(pre_factor[ordinal]),
                ),
                ConstraintSpec(
                    f"factor_negative:{name}",
                    -factor_exposures[ordinal],
                    FACTOR_EXPOSURE_LIMIT,
                    FACTOR_EXPOSURE_LIMIT,
                    float(-pre_factor[ordinal]),
                ),
            ]
        )

    for sector, mask in _group_masks(data.sector):
        indices = _indices(mask)
        sector_net = _sum(weights, indices)
        sector_gross = _sum(weights, _indices(mask & data.long_mask)) - _sum(
            weights,
            _indices(mask & data.short_mask),
        )
        pre_net = _numeric_sum(previous, indices)
        pre_gross_value = _numeric_sum(
            previous, _indices(mask & data.long_mask)
        ) - _numeric_sum(
            previous,
            _indices(mask & data.short_mask),
        )
        specs.extend(
            [
                ConstraintSpec(
                    f"sector_net_positive:{sector}",
                    sector_net,
                    SECTOR_NET_LIMIT,
                    SECTOR_NET_LIMIT,
                    pre_net,
                ),
                ConstraintSpec(
                    f"sector_net_negative:{sector}",
                    -sector_net,
                    SECTOR_NET_LIMIT,
                    SECTOR_NET_LIMIT,
                    -pre_net,
                ),
                ConstraintSpec(
                    f"sector_gross:{sector}",
                    sector_gross,
                    SECTOR_GROSS_LIMIT,
                    SECTOR_GROSS_LIMIT,
                    pre_gross_value,
                ),
            ]
        )

    for country, mask in _group_masks(data.country):
        if country == "USA":
            continue
        indices = _indices(mask)
        country_net = _sum(weights, indices)
        pre_country = _numeric_sum(previous, indices)
        specs.extend(
            [
                ConstraintSpec(
                    f"country_positive:{country}",
                    country_net,
                    COUNTRY_NET_LIMIT,
                    COUNTRY_NET_LIMIT,
                    pre_country,
                ),
                ConstraintSpec(
                    f"country_negative:{country}",
                    -country_net,
                    COUNTRY_NET_LIMIT,
                    COUNTRY_NET_LIMIT,
                    -pre_country,
                ),
            ]
        )
    foreign_indices = _indices(data.foreign_mask)
    if foreign_indices.size:
        foreign_net = _sum(weights, foreign_indices)
        pre_foreign = _numeric_sum(previous, foreign_indices)
        specs.extend(
            [
                ConstraintSpec(
                    "foreign_positive",
                    foreign_net,
                    FOREIGN_NET_LIMIT,
                    FOREIGN_NET_LIMIT,
                    pre_foreign,
                ),
                ConstraintSpec(
                    "foreign_negative",
                    -foreign_net,
                    FOREIGN_NET_LIMIT,
                    FOREIGN_NET_LIMIT,
                    -pre_foreign,
                ),
            ]
        )
    return specs


def ordinary_constraints(
    data: AllocationProblemData,
    weights: cp.Variable,
) -> list[cp.Constraint]:
    constraints = hard_trade_and_sign_constraints(data, weights)
    constraints.extend(
        spec.expression <= spec.limit
        for spec in slackable_constraint_specs(data, weights)
    )
    return constraints


def independent_constraint_diagnostics(
    data: AllocationProblemData,
    weights: object,
) -> ConstraintDiagnostics:
    """Recompute constraints directly from NumPy arrays, never solver duals."""

    vector = np.asarray(weights, dtype=np.float64)
    if vector.shape != (data.n_assets,) or not np.isfinite(vector).all():
        raise ValueError("weights must be a finite vector with n_assets entries")
    previous = data.previous_weight
    long_mask = data.long_mask
    short_mask = data.short_mask
    long_gross = float(np.sum(vector[long_mask]))
    short_gross = float(-np.sum(vector[short_mask]))
    gross = long_gross + short_gross
    net = long_gross - short_gross
    turnover = 0.5 * float(np.sum(np.abs(vector - previous)))
    risk_values = data.risk_root @ vector
    annual_variance = 12.0 * float(np.dot(risk_values, risk_values))
    factor = data.factor_betas.T @ vector

    violations: dict[str, float] = {
        "gross": gross - GROSS_LIMIT,
        "net_positive": net - NET_NAV_LIMIT,
        "net_negative": -net - NET_NAV_LIMIT,
        "dynamic_net_positive": 0.95 * long_gross - 1.05 * short_gross,
        "dynamic_net_negative": 0.95 * short_gross - 1.05 * long_gross,
        "augmented_risk": annual_variance - ANNUAL_VARIANCE_LIMIT,
        "turnover": turnover - ONE_WAY_TURNOVER_LIMIT,
    }
    normalizers: dict[str, float] = {
        "gross": GROSS_LIMIT,
        "net_positive": NET_NAV_LIMIT,
        "net_negative": NET_NAV_LIMIT,
        "dynamic_net_positive": NET_NAV_LIMIT,
        "dynamic_net_negative": NET_NAV_LIMIT,
        "augmented_risk": ANNUAL_VARIANCE_LIMIT,
        "turnover": ONE_WAY_TURNOVER_LIMIT,
    }
    delta = vector - previous
    for index in range(data.n_assets):
        permno = int(data.permno[index])
        if data.side[index] > 0:
            name = f"position_long:{permno}"
            violations[name] = float(vector[index] - LONG_CAP)
            normalizers[name] = LONG_CAP
            sign_violation = float(-vector[index])
            if data.required_exit[index]:
                exit_name = f"required_exit_long:{permno}"
                violations[exit_name] = float(vector[index])
                normalizers[exit_name] = LONG_CAP
        else:
            name = f"position_short:{permno}"
            violations[name] = float(-vector[index] - SHORT_CAP)
            normalizers[name] = SHORT_CAP
            sign_violation = float(vector[index])
            if data.required_exit[index]:
                exit_name = f"required_exit_short:{permno}"
                violations[exit_name] = float(-vector[index])
                normalizers[exit_name] = SHORT_CAP
        violations[f"sign:{permno}"] = sign_violation
        normalizers[f"sign:{permno}"] = LONG_CAP if data.side[index] > 0 else SHORT_CAP
        violations[f"trade_capacity:{permno}"] = float(
            abs(delta[index]) - data.trade_capacity_weight[index]
        )
        normalizers[f"trade_capacity:{permno}"] = max(
            float(data.trade_capacity_weight[index]),
            LONG_CAP if data.side[index] > 0 else SHORT_CAP,
        )
        adv = float(data.adv20_usd[index])
        adv_capacity = (
            0.10 * adv / float(data.nav_dollars)
            if np.isfinite(adv) and adv > 0.0
            else 0.0
        )
        violations[f"adv_capacity:{permno}"] = float(abs(delta[index]) - adv_capacity)
        normalizers[f"adv_capacity:{permno}"] = max(
            adv_capacity,
            LONG_CAP if data.side[index] > 0 else SHORT_CAP,
        )
        if data.reduction_only[index]:
            violations[f"reduction_only:{permno}"] = float(
                abs(vector[index]) - abs(previous[index])
            )
            normalizers[f"reduction_only:{permno}"] = max(
                abs(float(previous[index])),
                LONG_CAP if data.side[index] > 0 else SHORT_CAP,
            )

    for issuer in sorted(set(int(value) for value in data.permco.tolist())):
        issuer_mask = data.permco == issuer
        long_value = float(np.sum(vector[issuer_mask & long_mask]))
        short_value = float(-np.sum(vector[issuer_mask & short_mask]))
        if np.any(issuer_mask & long_mask):
            violations[f"issuer_long:{issuer}"] = long_value - LONG_CAP
            normalizers[f"issuer_long:{issuer}"] = LONG_CAP
        if np.any(issuer_mask & short_mask):
            violations[f"issuer_short:{issuer}"] = short_value - SHORT_CAP
            normalizers[f"issuer_short:{issuer}"] = SHORT_CAP

    factor_names = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
    for index, name in enumerate(factor_names):
        violations[f"factor_positive:{name}"] = float(
            factor[index] - FACTOR_EXPOSURE_LIMIT
        )
        violations[f"factor_negative:{name}"] = float(
            -factor[index] - FACTOR_EXPOSURE_LIMIT
        )
        normalizers[f"factor_positive:{name}"] = FACTOR_EXPOSURE_LIMIT
        normalizers[f"factor_negative:{name}"] = FACTOR_EXPOSURE_LIMIT

    for sector, mask in _group_masks(data.sector):
        sector_net = float(np.sum(vector[mask]))
        sector_gross = float(np.sum(np.abs(vector[mask])))
        violations[f"sector_net_positive:{sector}"] = sector_net - SECTOR_NET_LIMIT
        violations[f"sector_net_negative:{sector}"] = -sector_net - SECTOR_NET_LIMIT
        violations[f"sector_gross:{sector}"] = sector_gross - SECTOR_GROSS_LIMIT
        normalizers[f"sector_net_positive:{sector}"] = SECTOR_NET_LIMIT
        normalizers[f"sector_net_negative:{sector}"] = SECTOR_NET_LIMIT
        normalizers[f"sector_gross:{sector}"] = SECTOR_GROSS_LIMIT

    for country, mask in _group_masks(data.country):
        if country == "USA":
            continue
        country_net = float(np.sum(vector[mask]))
        violations[f"country_positive:{country}"] = country_net - COUNTRY_NET_LIMIT
        violations[f"country_negative:{country}"] = -country_net - COUNTRY_NET_LIMIT
        normalizers[f"country_positive:{country}"] = COUNTRY_NET_LIMIT
        normalizers[f"country_negative:{country}"] = COUNTRY_NET_LIMIT
    if np.any(data.foreign_mask):
        foreign_net = float(np.sum(vector[data.foreign_mask]))
        violations["foreign_positive"] = foreign_net - FOREIGN_NET_LIMIT
        violations["foreign_negative"] = -foreign_net - FOREIGN_NET_LIMIT
        normalizers["foreign_positive"] = FOREIGN_NET_LIMIT
        normalizers["foreign_negative"] = FOREIGN_NET_LIMIT

    normalized = {name: value / normalizers[name] for name, value in violations.items()}
    return ConstraintDiagnostics(
        long_gross=long_gross,
        short_gross=short_gross,
        gross=gross,
        net=net,
        one_way_turnover=turnover,
        augmented_annual_variance=annual_variance,
        augmented_annual_volatility=float(np.sqrt(max(annual_variance, 0.0))),
        factor_exposures=tuple(float(value) for value in factor),
        violations=violations,
        normalized_violations=normalized,
    )


__all__ = [
    "ANNUAL_VARIANCE_LIMIT",
    "COUNTRY_NET_LIMIT",
    "ConstraintDiagnostics",
    "ConstraintSpec",
    "FACTOR_EXPOSURE_LIMIT",
    "FOREIGN_NET_LIMIT",
    "GROSS_LIMIT",
    "LONG_CAP",
    "NET_GROSS_FRACTION",
    "NET_NAV_LIMIT",
    "ONE_WAY_TURNOVER_LIMIT",
    "SECTOR_GROSS_LIMIT",
    "SECTOR_NET_LIMIT",
    "SHORT_CAP",
    "hard_trade_and_sign_constraints",
    "independent_constraint_diagnostics",
    "ordinary_constraints",
    "slackable_constraint_specs",
]
