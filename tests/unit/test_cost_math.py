from __future__ import annotations

import math

import numpy as np
import pytest

from portfolio_bot.risk_cost.cost_math import (
    ILLIQUID_STRESS_BORROW_RATE,
    NAV_SENSITIVITY_DOLLARS,
    PRIMARY_COST_SCENARIO,
    REFERENCE_NAV_DOLLARS,
    REGULAR_STRESS_BORROW_RATE,
    SEVERE_IMPACT_COST_SCENARIO,
    STANDARD_STRESS_COST_SCENARIO,
    assign_adv_liquidity_quintiles,
    borrow_rate_schedule,
    estimate_impact_volatility,
    financing_cost_schedule,
    required_exit_volatility_p95,
    transaction_cost_return,
)


def test_impact_volatility_uses_sample_std_and_frozen_thresholds() -> None:
    latest_20 = [None] * 5 + [value / 10_000 for value in range(15)]
    latest_60 = [value / 10_000 for value in range(60)]
    estimate = estimate_impact_volatility(latest_20, latest_60)
    assert estimate is not None
    assert estimate.source_window_sessions == 20
    assert estimate.valid_observations == 15
    assert math.isclose(
        estimate.daily_volatility,
        float(np.std(np.arange(15) / 10_000, ddof=1)),
        rel_tol=1e-15,
    )

    twenty_insufficient = [None] * 6 + [value / 10_000 for value in range(14)]
    sixty_exact = [None] * 20 + [value / 10_000 for value in range(40)]
    fallback = estimate_impact_volatility(twenty_insufficient, sixty_exact)
    assert fallback is not None
    assert fallback.source_window_sessions == 60
    assert fallback.valid_observations == 40
    assert math.isclose(
        fallback.daily_volatility,
        float(np.std(np.arange(40) / 10_000, ddof=1)),
        rel_tol=1e-15,
    )

    assert estimate_impact_volatility(
        [None] * 6 + [0.01] * 14,
        [None] * 21 + [0.01] * 39,
    ) is None
    # Valid-count thresholds do not replace the global-session window itself.
    assert estimate_impact_volatility([0.01] * 15, [0.01] * 40) is None


def test_impact_volatility_windows_are_the_latest_global_sessions() -> None:
    # Values before the frozen windows must not influence the estimates.
    estimate = estimate_impact_volatility(
        [99.0] * 10 + [0.01] * 20,
        [99.0] * 10 + [0.01] * 60,
    )
    assert estimate is not None
    assert estimate.source_window_sessions == 20
    assert estimate.daily_volatility < 1e-15


def test_required_exit_fallback_is_type7_p95_over_valid_nonnegative_values() -> None:
    assert required_exit_volatility_p95([0.0, 0.10, None, float("nan")]) == 0.095
    assert required_exit_volatility_p95([None, -0.01, float("inf")]) is None


def test_transaction_cost_formula_and_frozen_scenarios() -> None:
    delta_weight = -0.02
    full_spread = 0.004
    adv = 50_000_000.0
    sigma = 0.025
    nav = 100_000_000.0
    participation_root = math.sqrt(abs(delta_weight) * nav / adv)

    for scenario in (
        PRIMARY_COST_SCENARIO,
        STANDARD_STRESS_COST_SCENARIO,
        SEVERE_IMPACT_COST_SCENARIO,
    ):
        expected = abs(delta_weight) * (
            scenario.spread_multiplier * full_spread / 2.0
            + 0.0001
            + scenario.impact_coefficient * sigma * participation_root
        )
        actual = transaction_cost_return(
            delta_weight,
            full_spread=full_spread,
            adv_dollars=adv,
            impact_volatility=sigma,
            nav_dollars=nav,
            scenario=scenario.name,
        )
        assert actual is not None
        assert math.isclose(actual, expected, rel_tol=1e-15)

    assert PRIMARY_COST_SCENARIO.spread_multiplier == 1.0
    assert PRIMARY_COST_SCENARIO.impact_coefficient == 0.10
    assert STANDARD_STRESS_COST_SCENARIO.spread_multiplier == 2.0
    assert STANDARD_STRESS_COST_SCENARIO.impact_coefficient == 0.20
    assert SEVERE_IMPACT_COST_SCENARIO.spread_multiplier == 1.0
    assert SEVERE_IMPACT_COST_SCENARIO.impact_coefficient == 0.50


@pytest.mark.parametrize(
    ("spread", "adv", "sigma"),
    [
        (None, 1_000_000.0, 0.02),
        (-0.001, 1_000_000.0, 0.02),
        (0.001, None, 0.02),
        (0.001, 0.0, 0.02),
        (0.001, 1_000_000.0, None),
        (0.001, 1_000_000.0, -0.02),
    ],
)
def test_nonzero_new_trade_fails_closed(
    spread: float | None, adv: float | None, sigma: float | None
) -> None:
    assert transaction_cost_return(
        0.01,
        full_spread=spread,
        adv_dollars=adv,
        impact_volatility=sigma,
        required_exit=False,
        required_exit_volatility_fallback=0.05,
    ) is None


def test_required_exit_may_use_supplied_p95_but_other_inputs_still_fail_closed() -> None:
    expected = transaction_cost_return(
        0.01,
        full_spread=0.002,
        adv_dollars=10_000_000.0,
        impact_volatility=0.05,
    )
    actual = transaction_cost_return(
        0.01,
        full_spread=0.002,
        adv_dollars=10_000_000.0,
        impact_volatility=None,
        required_exit=True,
        required_exit_volatility_fallback=0.05,
    )
    assert actual == expected
    assert transaction_cost_return(
        0.01,
        full_spread=None,
        adv_dollars=10_000_000.0,
        impact_volatility=None,
        required_exit=True,
        required_exit_volatility_fallback=0.05,
    ) is None
    assert transaction_cost_return(
        0.01,
        full_spread=0.002,
        adv_dollars=10_000_000.0,
        impact_volatility=None,
        required_exit=True,
        required_exit_volatility_fallback=-0.05,
    ) is None


def test_zero_trade_returns_zero_without_requiring_market_inputs() -> None:
    assert transaction_cost_return(
        0.0,
        full_spread=None,
        adv_dollars=None,
        impact_volatility=None,
        nav_dollars=float("nan"),
    ) == 0.0
    assert transaction_cost_return(
        -0.0,
        full_spread=-1.0,
        adv_dollars=0.0,
        impact_volatility=-1.0,
    ) == 0.0


def test_borrow_rates_use_total_stress_rates_and_simple_period_division() -> None:
    regular = borrow_rate_schedule(lowest_liquidity_quintile=False)
    illiquid = borrow_rate_schedule(lowest_liquidity_quintile=True)
    assert regular.primary.annual == 0.01
    assert regular.stress is REGULAR_STRESS_BORROW_RATE
    assert regular.stress.annual == 0.03
    assert math.isclose(regular.stress.optimizer_monthly, 0.03 / 12.0)
    assert math.isclose(regular.stress.daily, 0.03 / 252.0)
    assert illiquid.stress is ILLIQUID_STRESS_BORROW_RATE
    assert illiquid.stress.annual == 0.10
    assert math.isclose(illiquid.stress.optimizer_monthly, 0.10 / 12.0)
    assert math.isclose(illiquid.stress.daily, 0.10 / 252.0)


def test_financing_is_charged_only_on_gross_exposure_above_one() -> None:
    unlevered = financing_cost_schedule(0.80)
    boundary = financing_cost_schedule(1.0)
    leveraged = financing_cost_schedule(1.40)
    assert unlevered is not None and unlevered.primary.annual == 0.0
    assert boundary is not None and boundary.stress.annual == 0.0
    assert leveraged is not None
    assert math.isclose(leveraged.excess_gross_exposure, 0.40)
    assert math.isclose(leveraged.primary.annual, 0.40 * 0.005)
    assert math.isclose(leveraged.primary.optimizer_monthly, 0.40 * 0.005 / 12.0)
    assert math.isclose(leveraged.primary.daily, 0.40 * 0.005 / 252.0)
    assert math.isclose(leveraged.stress.annual, 0.40 * 0.015)
    assert financing_cost_schedule(-0.01) is None
    assert financing_cost_schedule(float("nan")) is None


def test_adv_liquidity_quintiles_are_deterministic_and_preserve_input_order() -> None:
    permnos = [30, 10, 20, 40, 50, 60, 70, 80, 90, 100]
    adv = [5.0, None, 0.0, 5.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0]
    actual = assign_adv_liquidity_quintiles(permnos, adv)

    # Sorted worst-to-best order is PERMNO 10, 20, then positive ADV ascending
    # with the ADV=5 tie resolved by PERMNO 30 before PERMNO 40.
    expected_by_permno = {
        10: 1,
        20: 1,
        50: 2,
        60: 2,
        70: 3,
        80: 3,
        30: 4,
        40: 4,
        90: 5,
        100: 5,
    }
    assert actual.dtype == np.int8
    assert actual.tolist() == [expected_by_permno[permno] for permno in permnos]


def test_adv_quintiles_use_literal_floor_rank_rule_and_validate_keys() -> None:
    permnos = [7, 6, 5, 4, 3, 2, 1]
    adv = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    actual = assign_adv_liquidity_quintiles(permnos, adv)
    expected_by_rank = [math.floor(5 * rank / 7) + 1 for rank in range(7)]
    # Reverse because caller order is most-liquid to least-liquid.
    assert actual.tolist() == list(reversed(expected_by_rank))
    assert assign_adv_liquidity_quintiles([], []).tolist() == []
    with pytest.raises(ValueError, match="equal length"):
        assign_adv_liquidity_quintiles([1], [])
    with pytest.raises(ValueError, match="unique"):
        assign_adv_liquidity_quintiles([1, 1], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite integers"):
        assign_adv_liquidity_quintiles([1, 2.5], [1.0, 2.0])


def test_reference_nav_and_sensitivities_are_frozen() -> None:
    assert REFERENCE_NAV_DOLLARS == 100_000_000.0
    assert NAV_SENSITIVITY_DOLLARS == (
        25_000_000.0,
        100_000_000.0,
        500_000_000.0,
    )
