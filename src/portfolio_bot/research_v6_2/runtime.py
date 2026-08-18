"""Training runtime for the V6.2 portfolio-level policy objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from portfolio_bot.allocation.costs import torch_total_cost
from portfolio_bot.research_v6_2.policy import (
    LinearPortfolioPolicy,
    approximate_monthly_variance,
)


@dataclass(frozen=True)
class PolicyMonth:
    holding_month: str
    permno: np.ndarray
    features: np.ndarray
    excess_return: np.ndarray
    long_eligible: np.ndarray
    short_eligible: np.ndarray
    full_spread: np.ndarray
    adv20_usd: np.ndarray
    impact_volatility_daily: np.ndarray
    borrow_rate_annual: np.ndarray
    factor_betas: np.ndarray
    specific_variance_monthly: np.ndarray


def build_policy(config: dict[str, Any], feature_count: int, *, seed_offset: int = 0) -> LinearPortfolioPolicy:
    model = config["model"]
    return LinearPortfolioPolicy(
        feature_count,
        maximum_long_gross=float(model["maximum_long_gross"]),
        maximum_short_gross=float(model["maximum_short_gross"]),
        initial_long_gross_logit=float(model["initial_long_gross_logit"]),
        initial_short_gross_logit=float(model["initial_short_gross_logit"]),
        temperature=float(model["temperature"]),
        seed=int(model["random_seed"]) + int(seed_offset),
    )


def _aligned_previous(month: PolicyMonth, previous: dict[int, float]) -> np.ndarray:
    return np.asarray([previous.get(int(permno), 0.0) for permno in month.permno], dtype=np.float32)


def fit_policy(
    model: LinearPortfolioPolicy,
    months: Iterable[PolicyMonth],
    *,
    profile: dict[str, Any],
    nav_dollars: float,
    maximum_position_weight: float,
    maximum_one_way_adv_participation: float,
    financing_rate_annual: float,
    spread_multiplier: float,
    impact_coefficient: float,
    gradient_clip: float,
) -> list[dict[str, float | int]]:
    """Optimize sequential monthly net utility with causal turnover state."""

    ordered = list(months)
    if not ordered:
        raise ValueError("direct-policy fit received no months")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(profile["learning_rate"])
    )
    diagnostics: list[dict[str, float | int]] = []
    for epoch in range(int(profile["epochs"])):
        previous: dict[int, float] = {}
        epoch_utility = 0.0
        epoch_return = 0.0
        epoch_cost = 0.0
        epoch_risk = 0.0
        for month in ordered:
            optimizer.zero_grad(set_to_none=True)
            features = torch.as_tensor(month.features, dtype=torch.float32)
            long_eligible = torch.as_tensor(month.long_eligible, dtype=torch.bool)
            short_eligible = torch.as_tensor(month.short_eligible, dtype=torch.bool)
            capacity = torch.minimum(
                torch.full(
                    (len(month.permno),),
                    float(maximum_position_weight),
                    dtype=torch.float32,
                ),
                torch.as_tensor(month.adv20_usd, dtype=torch.float32)
                * float(maximum_one_way_adv_participation)
                / float(nav_dollars),
            )
            weights, _, _ = model(
                features,
                long_eligible=long_eligible,
                short_eligible=short_eligible,
                capacity=capacity,
            )
            previous_array = _aligned_previous(month, previous)
            cost = torch_total_cost(
                weights,
                previous_array,
                month.full_spread,
                month.adv20_usd,
                month.impact_volatility_daily,
                month.borrow_rate_annual,
                nav_dollars,
                impact_coefficient=float(impact_coefficient),
                spread_multiplier=float(spread_multiplier),
                financing_rate_annual=float(financing_rate_annual),
            )
            realized = weights @ torch.as_tensor(month.excess_return, dtype=torch.float32)
            variance = approximate_monthly_variance(
                weights,
                torch.as_tensor(month.factor_betas, dtype=torch.float32),
                torch.as_tensor(month.specific_variance_monthly, dtype=torch.float32),
            )
            l2 = (
                torch.sum(torch.square(model.stock_score))
                + torch.sum(torch.square(model.long_context))
                + torch.sum(torch.square(model.short_context))
            )
            utility = (
                realized
                - cost.total
                - float(profile["risk_penalty"]) * 12.0 * variance
                - float(profile["l2_penalty"]) * l2 / len(ordered)
            )
            (-utility).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()
            numeric = weights.detach().cpu().numpy().astype(float)
            previous = {
                int(permno): float(weight)
                for permno, weight in zip(month.permno, numeric)
                if abs(float(weight)) > 1e-12
            }
            epoch_utility += float(utility.detach().cpu().item())
            epoch_return += float(realized.detach().cpu().item())
            epoch_cost += float(cost.total.detach().cpu().item())
            epoch_risk += float(variance.detach().cpu().item())
        diagnostics.append(
            {
                "epoch": epoch + 1,
                "mean_utility": epoch_utility / len(ordered),
                "mean_gross_excess_return": epoch_return / len(ordered),
                "mean_cost": epoch_cost / len(ordered),
                "mean_monthly_variance_proxy": epoch_risk / len(ordered),
            }
        )
    return diagnostics


def predict_policy(
    model: LinearPortfolioPolicy,
    month: PolicyMonth,
    *,
    nav_dollars: float,
    maximum_position_weight: float,
    maximum_one_way_adv_participation: float,
    short_gross_zero_threshold: float,
) -> tuple[np.ndarray, float, float]:
    model.eval()
    with torch.no_grad():
        capacity = torch.minimum(
            torch.full(
                (len(month.permno),),
                float(maximum_position_weight),
                dtype=torch.float32,
            ),
            torch.as_tensor(month.adv20_usd, dtype=torch.float32)
            * float(maximum_one_way_adv_participation)
            / float(nav_dollars),
        )
        weights, long_gross, short_gross = model(
            torch.as_tensor(month.features, dtype=torch.float32),
            long_eligible=torch.as_tensor(month.long_eligible, dtype=torch.bool),
            short_eligible=torch.as_tensor(month.short_eligible, dtype=torch.bool),
            capacity=capacity,
        )
        result = weights.cpu().numpy().astype(float)
        short_value = float(short_gross.cpu().item())
        if short_value < float(short_gross_zero_threshold):
            result[result < 0.0] = 0.0
            short_value = 0.0
        return result, float(long_gross.cpu().item()), short_value


__all__ = [
    "PolicyMonth",
    "build_policy",
    "fit_policy",
    "predict_policy",
]
