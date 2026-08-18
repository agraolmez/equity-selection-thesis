"""Portfolio-aware linear policy and deterministic implementation repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from portfolio_bot.research_v6_1.portfolio import portfolio_risk


class LinearPortfolioPolicy(nn.Module):
    """Map stock features directly to a cross-sectional portfolio target."""

    def __init__(
        self,
        feature_count: int,
        *,
        maximum_long_gross: float,
        maximum_short_gross: float,
        initial_long_gross_logit: float,
        initial_short_gross_logit: float,
        temperature: float,
        seed: int,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        scale = 0.02 / max(float(feature_count) ** 0.5, 1.0)
        self.stock_score = nn.Parameter(
            torch.randn(feature_count, generator=generator) * scale
        )
        self.long_context = nn.Parameter(torch.zeros(feature_count))
        self.short_context = nn.Parameter(torch.zeros(feature_count))
        self.long_bias = nn.Parameter(torch.tensor(float(initial_long_gross_logit)))
        self.short_bias = nn.Parameter(torch.tensor(float(initial_short_gross_logit)))
        self.maximum_long_gross = float(maximum_long_gross)
        self.maximum_short_gross = float(maximum_short_gross)
        self.temperature = float(temperature)

    @staticmethod
    def _capacity_clip(
        desired: torch.Tensor,
        capacity: torch.Tensor,
    ) -> torch.Tensor:
        return torch.minimum(torch.clamp(desired, min=0.0), torch.clamp(capacity, min=0.0))

    def forward(
        self,
        features: torch.Tensor,
        *,
        long_eligible: torch.Tensor,
        short_eligible: torch.Tensor,
        capacity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.stock_score.numel():
            raise ValueError("policy feature matrix has an unexpected shape")
        score = features @ self.stock_score
        eligible = long_eligible | short_eligible
        if bool(eligible.any().detach().cpu().item()):
            score = score - score[eligible].mean()
        direction = torch.tanh(score / self.temperature)
        long_strength = torch.relu(direction) * long_eligible.to(features.dtype)
        short_strength = torch.relu(-direction) * short_eligible.to(features.dtype)
        context = features.mean(dim=0)
        long_gross = self.maximum_long_gross * torch.sigmoid(
            self.long_bias + context @ self.long_context
        )
        short_gross = self.maximum_short_gross * torch.sigmoid(
            self.short_bias + context @ self.short_context
        )
        long_desired = long_gross * long_strength / torch.clamp(long_strength.sum(), min=1e-8)
        short_desired = short_gross * short_strength / torch.clamp(short_strength.sum(), min=1e-8)
        long_weight = self._capacity_clip(long_desired, capacity)
        short_weight = self._capacity_clip(short_desired, capacity)
        return long_weight - short_weight, long_gross, short_gross


def approximate_monthly_variance(
    weights: torch.Tensor,
    factor_betas: torch.Tensor,
    specific_variance_monthly: torch.Tensor,
    *,
    factor_variance_scale: float = 0.0004,
) -> torch.Tensor:
    """Differentiable PIT factor-plus-specific variance proxy for training."""

    exposure = factor_betas.T @ weights
    factor = float(factor_variance_scale) * torch.sum(torch.square(exposure))
    specific = torch.sum(
        torch.clamp(specific_variance_monthly, min=0.0) * torch.square(weights)
    )
    return factor + specific


@dataclass(frozen=True)
class RepairResult:
    weights: np.ndarray
    scale: float
    forecast_volatility: float
    factor_exposures: np.ndarray
    turnover: float


def _constraint_scale(
    weights: np.ndarray,
    *,
    factor_betas: np.ndarray,
    covariance_root: np.ndarray,
    specific_variance_monthly: np.ndarray,
    industries: np.ndarray,
    maximum_gross: float,
    minimum_net: float,
    maximum_net: float,
    forecast_volatility_target: float,
    market_beta_absolute_limit: float,
    other_factor_absolute_limit: float,
    sic2_industry_net_absolute_limit: float,
    sic2_industry_gross_limit: float,
) -> float:
    scale = 1.0
    gross = float(np.abs(weights).sum())
    net = float(weights.sum())
    if gross > maximum_gross:
        scale = min(scale, maximum_gross / gross)
    if net > maximum_net:
        scale = min(scale, maximum_net / net)
    if net < minimum_net:
        scale = min(scale, minimum_net / net)
    risk, exposures = portfolio_risk(
        weights, factor_betas, covariance_root, specific_variance_monthly
    )
    if risk > forecast_volatility_target:
        scale = min(scale, forecast_volatility_target / risk)
    if abs(float(exposures[0])) > market_beta_absolute_limit:
        scale = min(scale, market_beta_absolute_limit / abs(float(exposures[0])))
    for value in exposures[1:]:
        if abs(float(value)) > other_factor_absolute_limit:
            scale = min(scale, other_factor_absolute_limit / abs(float(value)))
    industry_values = np.asarray(industries, dtype=str)
    for industry in np.unique(industry_values):
        member = weights[industry_values == industry]
        industry_net = abs(float(member.sum()))
        industry_gross = float(np.abs(member).sum())
        if industry_net > sic2_industry_net_absolute_limit:
            scale = min(scale, sic2_industry_net_absolute_limit / industry_net)
        if industry_gross > sic2_industry_gross_limit:
            scale = min(scale, sic2_industry_gross_limit / industry_gross)
    return float(np.clip(scale, 0.0, 1.0))


def repair_policy_weights(
    desired_weights: np.ndarray,
    previous_weights: np.ndarray,
    *,
    factor_betas: np.ndarray,
    covariance_root: np.ndarray,
    specific_variance_monthly: np.ndarray,
    industries: np.ndarray,
    adv20_usd: np.ndarray,
    nav_dollars: float,
    maximum_position_weight: float,
    maximum_one_way_adv_participation: float,
    maximum_one_way_turnover: float,
    maximum_gross: float,
    minimum_net: float,
    maximum_net: float,
    forecast_volatility_target: float,
    market_beta_absolute_limit: float,
    other_factor_absolute_limit: float,
    sic2_industry_net_absolute_limit: float,
    sic2_industry_gross_limit: float,
) -> RepairResult:
    """Shrink a learned target to all hard limits without re-ranking names."""

    desired = np.asarray(desired_weights, dtype=np.float64)
    previous = np.asarray(previous_weights, dtype=np.float64)
    capacity = np.minimum(
        float(maximum_position_weight),
        float(maximum_one_way_adv_participation)
        * np.asarray(adv20_usd, dtype=np.float64)
        / float(nav_dollars),
    )
    target = np.clip(desired, -capacity, capacity)
    scale = _constraint_scale(
        target,
        factor_betas=factor_betas,
        covariance_root=covariance_root,
        specific_variance_monthly=specific_variance_monthly,
        industries=industries,
        maximum_gross=maximum_gross,
        minimum_net=minimum_net,
        maximum_net=maximum_net,
        forecast_volatility_target=forecast_volatility_target,
        market_beta_absolute_limit=market_beta_absolute_limit,
        other_factor_absolute_limit=other_factor_absolute_limit,
        sic2_industry_net_absolute_limit=sic2_industry_net_absolute_limit,
        sic2_industry_gross_limit=sic2_industry_gross_limit,
    )
    target *= scale
    delta = np.clip(target - previous, -capacity, capacity)
    turnover = 0.5 * float(np.abs(delta).sum())
    if turnover > maximum_one_way_turnover:
        delta *= float(maximum_one_way_turnover) / turnover
    weights = previous + delta
    final_scale = _constraint_scale(
        weights,
        factor_betas=factor_betas,
        covariance_root=covariance_root,
        specific_variance_monthly=specific_variance_monthly,
        industries=industries,
        maximum_gross=maximum_gross,
        minimum_net=minimum_net,
        maximum_net=maximum_net,
        forecast_volatility_target=forecast_volatility_target,
        market_beta_absolute_limit=market_beta_absolute_limit,
        other_factor_absolute_limit=other_factor_absolute_limit,
        sic2_industry_net_absolute_limit=sic2_industry_net_absolute_limit,
        sic2_industry_gross_limit=sic2_industry_gross_limit,
    )
    if final_scale < 1.0:
        weights *= final_scale
    risk, exposures = portfolio_risk(
        weights, factor_betas, covariance_root, specific_variance_monthly
    )
    return RepairResult(
        weights=weights,
        scale=scale * final_scale,
        forecast_volatility=risk,
        factor_exposures=exposures,
        turnover=0.5 * float(np.abs(weights - previous).sum()),
    )


__all__ = [
    "LinearPortfolioPolicy",
    "RepairResult",
    "approximate_monthly_variance",
    "repair_policy_weights",
]
