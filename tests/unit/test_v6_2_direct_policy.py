from __future__ import annotations

import numpy as np
import torch

from portfolio_bot.research_v6_2.policy import (
    LinearPortfolioPolicy,
    repair_policy_weights,
)


def test_direct_policy_respects_direction_eligibility_and_capacity() -> None:
    model = LinearPortfolioPolicy(
        3,
        maximum_long_gross=1.0,
        maximum_short_gross=0.5,
        initial_long_gross_logit=0.0,
        initial_short_gross_logit=-2.2,
        temperature=0.5,
        seed=7,
    )
    features = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    capacity = torch.full((4,), 0.10)
    weights, long_gross, short_gross = model(
        features,
        long_eligible=torch.tensor([True, True, False, False]),
        short_eligible=torch.tensor([False, False, True, True]),
        capacity=capacity,
    )
    assert torch.all(weights[:2] >= 0.0)
    assert torch.all(weights[2:] <= 0.0)
    assert torch.max(torch.abs(weights)) <= 0.10 + 1e-7
    assert 0.0 < float(long_gross.detach()) < 1.0
    assert 0.0 < float(short_gross.detach()) < 0.5


def test_direct_policy_repair_scales_to_risk_factor_and_industry_limits() -> None:
    desired = np.asarray([0.10, 0.10, -0.02, -0.02])
    betas = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), (4, 1))
    result = repair_policy_weights(
        desired,
        np.zeros(4),
        factor_betas=betas,
        covariance_root=np.zeros((6, 6)),
        specific_variance_monthly=np.full(4, 0.0001),
        industries=np.asarray(["A", "A", "B", "B"]),
        adv20_usd=np.full(4, 1e9),
        nav_dollars=100_000_000.0,
        maximum_position_weight=0.10,
        maximum_one_way_adv_participation=0.10,
        maximum_one_way_turnover=0.50,
        maximum_gross=0.50,
        minimum_net=-0.10,
        maximum_net=1.00,
        forecast_volatility_target=0.50,
        market_beta_absolute_limit=0.05,
        other_factor_absolute_limit=0.10,
        sic2_industry_net_absolute_limit=0.05,
        sic2_industry_gross_limit=0.30,
    )
    assert abs(float(result.factor_exposures[0])) <= 0.05 + 1e-8
    assert abs(float(result.weights[:2].sum())) <= 0.05 + 1e-8
    assert result.turnover <= 0.50 + 1e-8
    assert result.scale < 1.0
