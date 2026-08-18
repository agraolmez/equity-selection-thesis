from __future__ import annotations

import inspect

import cvxpy as cp
import numpy as np
import pytest
import torch

from portfolio_bot.allocation.costs import (
    COMMISSION_RATE,
    PRIMARY_BORROW_ANNUAL_RATE,
    PRIMARY_FINANCING_ANNUAL_RATE,
    CostInputError,
    borrow_cost_cvxpy,
    borrow_cost_numpy,
    borrow_cost_torch,
    cvxpy_total_cost,
    financing_cost_cvxpy,
    financing_cost_numpy,
    financing_cost_torch,
    numpy_total_cost,
    torch_total_cost,
    transaction_cost_cvxpy,
    transaction_cost_numpy,
    transaction_cost_torch,
)
from portfolio_bot.risk_cost.cost_math import (
    COMMISSION_AND_OPERATIONAL_SLIPPAGE,
    PRIMARY_BORROW_RATE,
    PRIMARY_COST_SCENARIO,
    PRIMARY_FINANCING_RATE,
    REFERENCE_NAV_DOLLARS,
    transaction_cost_return,
)


def _transaction_fixture() -> dict[str, np.ndarray | float]:
    return {
        "delta_weight": np.asarray([0.02, -0.03, 0.0], dtype=np.float64),
        "full_spread": np.asarray([0.002, 0.004, 0.006], dtype=np.float64),
        "adv_dollars": np.asarray([40e6, 75e6, 20e6], dtype=np.float64),
        "impact_volatility": np.asarray([0.020, 0.035, 0.050], dtype=np.float64),
        "nav_dollars": REFERENCE_NAV_DOLLARS,
    }


def test_frozen_constants_are_exact_stage8_constants() -> None:
    assert COMMISSION_RATE == COMMISSION_AND_OPERATIONAL_SLIPPAGE == 0.0001
    assert PRIMARY_BORROW_ANNUAL_RATE == PRIMARY_BORROW_RATE.annual == 0.01
    assert PRIMARY_FINANCING_ANNUAL_RATE == PRIMARY_FINANCING_RATE.annual == 0.005


def test_numpy_scalar_and_vector_transaction_audit_math_matches_stage8() -> None:
    fixture = _transaction_fixture()
    delta = np.asarray(fixture["delta_weight"])
    spread = np.asarray(fixture["full_spread"])
    adv = np.asarray(fixture["adv_dollars"])
    sigma = np.asarray(fixture["impact_volatility"])
    nav = float(fixture["nav_dollars"])
    expected = np.abs(delta) * (spread / 2.0 + 0.0001) + (
        PRIMARY_COST_SCENARIO.impact_coefficient
        * sigma
        * np.sqrt(nav / adv)
        * np.abs(delta) ** 1.5
    )

    actual = transaction_cost_numpy(**fixture, reduction="none")
    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0.0)
    assert transaction_cost_numpy(**fixture) == pytest.approx(float(expected.sum()))

    for index in range(delta.size):
        stage8 = transaction_cost_return(
            float(delta[index]),
            full_spread=float(spread[index]),
            adv_dollars=float(adv[index]),
            impact_volatility=float(sigma[index]),
            nav_dollars=nav,
        )
        assert stage8 is not None
        assert actual[index] == pytest.approx(stage8, rel=2e-15)

    scalar = transaction_cost_numpy(
        -0.02,
        full_spread=0.003,
        adv_dollars=50e6,
        impact_volatility=0.025,
    )
    scalar_stage8 = transaction_cost_return(
        -0.02,
        full_spread=0.003,
        adv_dollars=50e6,
        impact_volatility=0.025,
    )
    assert scalar_stage8 is not None
    assert scalar == pytest.approx(scalar_stage8, rel=2e-15)


def test_numpy_cvxpy_and_torch_transaction_costs_are_exactly_equivalent() -> None:
    fixture = _transaction_fixture()
    expected = float(transaction_cost_numpy(**fixture))

    delta_parameter = cp.Parameter(3, value=fixture["delta_weight"])
    cvx_term = transaction_cost_cvxpy(
        delta_parameter,
        full_spread=fixture["full_spread"],
        adv_dollars=fixture["adv_dollars"],
        impact_volatility=fixture["impact_volatility"],
        nav_dollars=float(fixture["nav_dollars"]),
    )
    assert cvx_term.total.is_convex()
    assert cvx_term.constraints == ()
    assert cvx_term.trade_permitted.tolist() == [True, True, True]
    assert float(cvx_term.total.value) == pytest.approx(expected, rel=2e-14)

    torch_cost = transaction_cost_torch(
        torch.as_tensor(fixture["delta_weight"], dtype=torch.float64),
        full_spread=fixture["full_spread"],
        adv_dollars=fixture["adv_dollars"],
        impact_volatility=fixture["impact_volatility"],
        nav_dollars=float(fixture["nav_dollars"]),
    )
    assert float(torch_cost.detach()) == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize(
    ("spread", "adv", "volatility", "nav"),
    [
        (None, 10e6, 0.02, 100e6),
        (-0.001, 10e6, 0.02, 100e6),
        (0.001, None, 0.02, 100e6),
        (0.001, 0.0, 0.02, 100e6),
        (0.001, 10e6, None, 100e6),
        (0.001, 10e6, -0.02, 100e6),
        (0.001, 10e6, 0.02, float("nan")),
    ],
)
def test_invalid_inputs_prohibit_nonzero_trade_but_allow_exact_zero(
    spread: float | None,
    adv: float | None,
    volatility: float | None,
    nav: float,
) -> None:
    with pytest.raises(CostInputError, match="nonzero trade"):
        transaction_cost_numpy(
            0.01,
            full_spread=spread,
            adv_dollars=adv,
            impact_volatility=volatility,
            nav_dollars=nav,
        )
    assert (
        transaction_cost_numpy(
            0.0,
            full_spread=spread,
            adv_dollars=adv,
            impact_volatility=volatility,
            nav_dollars=nav,
        )
        == 0.0
    )

    with pytest.raises(CostInputError, match="nonzero trade"):
        transaction_cost_torch(
            torch.tensor(0.01, dtype=torch.float64),
            full_spread=spread,
            adv_dollars=adv,
            impact_volatility=volatility,
            nav_dollars=nav,
        )
    torch_zero = transaction_cost_torch(
        torch.tensor(0.0, dtype=torch.float64),
        full_spread=spread,
        adv_dollars=adv,
        impact_volatility=volatility,
        nav_dollars=nav,
    )
    assert float(torch_zero) == 0.0


def test_cvxpy_invalid_input_adds_hard_zero_trade_constraint() -> None:
    delta = cp.Variable(2)
    term = transaction_cost_cvxpy(
        delta,
        full_spread=[0.002, np.nan],
        adv_dollars=[20e6, 20e6],
        impact_volatility=[0.02, 0.02],
    )
    assert term.trade_permitted.tolist() == [True, False]
    assert len(term.constraints) == 1

    allowed_zero = cp.Problem(
        cp.Minimize(term.total),
        [delta == np.asarray([0.01, 0.0]), *term.constraints],
    )
    allowed_zero.solve(solver="CLARABEL")
    assert allowed_zero.status == cp.OPTIMAL

    prohibited_nonzero = cp.Problem(
        cp.Minimize(term.total),
        [delta == np.asarray([0.01, 0.01]), *term.constraints],
    )
    prohibited_nonzero.solve(solver="CLARABEL")
    assert prohibited_nonzero.status == cp.INFEASIBLE


def test_required_exit_volatility_is_already_resolved_by_the_caller() -> None:
    assert "required_exit" not in inspect.signature(transaction_cost_numpy).parameters
    resolved = transaction_cost_numpy(
        -0.01,
        full_spread=0.002,
        adv_dollars=10e6,
        impact_volatility=0.05,
    )
    assert resolved > 0.0
    with pytest.raises(CostInputError, match="nonzero trade"):
        transaction_cost_numpy(
            -0.01,
            full_spread=0.002,
            adv_dollars=10e6,
            impact_volatility=None,
        )


def test_transaction_cost_is_sign_symmetric_and_monotone_in_every_friction() -> None:
    kwargs = {
        "full_spread": 0.002,
        "adv_dollars": 50e6,
        "impact_volatility": 0.02,
        "nav_dollars": 100e6,
    }
    base = float(transaction_cost_numpy(0.01, **kwargs))
    assert transaction_cost_numpy(-0.01, **kwargs) == pytest.approx(base)
    assert transaction_cost_numpy(0.02, **kwargs) > base
    assert transaction_cost_numpy(0.01, **{**kwargs, "full_spread": 0.004}) > base
    assert transaction_cost_numpy(0.01, **{**kwargs, "impact_volatility": 0.04}) > base
    assert transaction_cost_numpy(0.01, **{**kwargs, "nav_dollars": 200e6}) > base
    assert transaction_cost_numpy(0.01, **{**kwargs, "adv_dollars": 100e6}) < base
    assert transaction_cost_numpy(0.0, **kwargs) == 0.0


def test_torch_transaction_gradient_matches_closed_form() -> None:
    delta = torch.tensor([0.04, -0.01], dtype=torch.float64, requires_grad=True)
    spread = np.asarray([0.002, 0.004])
    adv = np.asarray([50e6, 80e6])
    sigma = np.asarray([0.02, 0.03])
    nav = 100e6
    cost = transaction_cost_torch(
        delta,
        full_spread=spread,
        adv_dollars=adv,
        impact_volatility=sigma,
        nav_dollars=nav,
    )
    cost.backward()
    linear = spread / 2.0 + 0.0001
    impact_scale = 0.10 * sigma * np.sqrt(nav / adv)
    expected_gradient = np.sign(delta.detach().numpy()) * (
        linear + 1.5 * impact_scale * np.sqrt(np.abs(delta.detach().numpy()))
    )
    assert delta.grad is not None
    np.testing.assert_allclose(
        delta.grad.detach().numpy(), expected_gradient, rtol=2e-14, atol=2e-16
    )


def test_borrow_cost_equivalence_units_signs_zero_and_gradient() -> None:
    weights = np.asarray([-0.40, 0.30, -0.10, 0.0])
    annual_rates = np.asarray([0.01, 0.03, 0.10, 0.10])
    expected_per_asset = np.asarray([0.40 * 0.01 / 12.0, 0.0, 0.10 * 0.10 / 12.0, 0.0])
    actual_per_asset = borrow_cost_numpy(
        weights, annual_rate=annual_rates, reduction="none"
    )
    np.testing.assert_allclose(actual_per_asset, expected_per_asset, rtol=0.0, atol=0.0)
    expected = float(expected_per_asset.sum())
    assert borrow_cost_numpy(weights, annual_rate=annual_rates) == expected
    assert borrow_cost_numpy([0.0, 0.5]) == 0.0

    parameter = cp.Parameter(4, value=weights)
    cvx_cost = borrow_cost_cvxpy(parameter, annual_rate=annual_rates)
    assert cvx_cost.is_convex()
    assert float(cvx_cost.value) == pytest.approx(expected, rel=2e-15)

    tensor = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    torch_cost = borrow_cost_torch(tensor, annual_rate=annual_rates)
    assert float(torch_cost.detach()) == pytest.approx(expected, rel=2e-15)
    torch_cost.backward()
    assert tensor.grad is not None
    np.testing.assert_allclose(
        tensor.grad.detach().numpy(),
        np.asarray([-0.01 / 12.0, 0.0, -0.10 / 12.0, 0.0]),
        rtol=2e-15,
        atol=0.0,
    )


def test_financing_cost_equivalence_units_cash_threshold_and_gradient() -> None:
    assert financing_cost_numpy([0.0, 0.0]) == 0.0
    assert financing_cost_numpy([0.60, -0.40]) == 0.0
    weights = np.asarray([0.80, -0.40])
    expected = 0.20 * 0.005 / 12.0
    assert financing_cost_numpy(weights) == pytest.approx(expected, rel=2e-15)

    parameter = cp.Parameter(2, value=weights)
    cvx_cost = financing_cost_cvxpy(parameter)
    assert cvx_cost.is_convex()
    assert float(cvx_cost.value) == pytest.approx(expected, rel=2e-15)

    tensor = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    torch_cost = financing_cost_torch(tensor)
    assert float(torch_cost.detach()) == pytest.approx(expected, rel=2e-15)
    torch_cost.backward()
    assert tensor.grad is not None
    np.testing.assert_allclose(
        tensor.grad.detach().numpy(),
        np.asarray([0.005 / 12.0, -0.005 / 12.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_total_cost_wrappers_match_components_across_all_backends() -> None:
    weights = np.asarray([0.70, -0.45], dtype=np.float64)
    previous = np.asarray([0.65, -0.40], dtype=np.float64)
    spread = np.asarray([0.002, 0.004], dtype=np.float64)
    adv = np.asarray([50e6, 80e6], dtype=np.float64)
    volatility = np.asarray([0.02, 0.03], dtype=np.float64)
    borrow_rates = np.asarray([0.01, 0.10], dtype=np.float64)
    nav = 100e6

    numpy_cost = numpy_total_cost(
        weights,
        previous,
        spread,
        adv,
        volatility,
        borrow_rates,
        nav,
    )
    expected_transaction = float(
        transaction_cost_numpy(
            weights - previous,
            full_spread=spread,
            adv_dollars=adv,
            impact_volatility=volatility,
            nav_dollars=nav,
        )
    )
    expected_borrow = float(borrow_cost_numpy(weights, annual_rate=borrow_rates))
    expected_financing = financing_cost_numpy(weights)
    assert numpy_cost.transaction == expected_transaction
    assert numpy_cost.borrow == expected_borrow
    assert numpy_cost.financing == expected_financing
    assert (
        numpy_cost.total == expected_transaction + expected_borrow + expected_financing
    )

    parameter = cp.Parameter(2, value=weights)
    cvx_cost = cvxpy_total_cost(
        parameter,
        previous,
        spread,
        adv,
        volatility,
        borrow_rates,
        nav,
    )
    assert cvx_cost.total.is_convex()
    assert cvx_cost.constraints == ()
    assert cvx_cost.trade_permitted.tolist() == [True, True]
    assert float(cvx_cost.transaction.value) == pytest.approx(expected_transaction)
    assert float(cvx_cost.borrow.value) == pytest.approx(expected_borrow)
    assert float(cvx_cost.financing.value) == pytest.approx(expected_financing)
    assert float(cvx_cost.total.value) == pytest.approx(numpy_cost.total)

    tensor = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    torch_cost = torch_total_cost(
        tensor,
        previous,
        spread,
        adv,
        volatility,
        borrow_rates,
        nav,
    )
    assert float(torch_cost.transaction.detach()) == pytest.approx(expected_transaction)
    assert float(torch_cost.borrow.detach()) == pytest.approx(expected_borrow)
    assert float(torch_cost.financing.detach()) == pytest.approx(expected_financing)
    assert float(torch_cost.total.detach()) == pytest.approx(numpy_cost.total)
    torch_cost.total.backward()
    assert tensor.grad is not None
    assert bool(torch.isfinite(tensor.grad).all())


def test_total_cost_propagates_fail_closed_invalid_trade_rule() -> None:
    weights = np.asarray([0.10, -0.10])
    previous = np.asarray([0.10, -0.10])
    kwargs = {
        "full_spread": [0.002, np.nan],
        "adv20_usd": [20e6, 20e6],
        "impact_volatility_daily": [0.02, 0.02],
        "borrow_rate_annual": [0.01, 0.01],
        "nav_dollars": 100e6,
    }
    # The invalid second cost row is harmless only because its trade is zero.
    assert numpy_total_cost(weights, previous, **kwargs).transaction == 0.0
    with pytest.raises(CostInputError, match="nonzero trade"):
        numpy_total_cost(np.asarray([0.10, -0.11]), previous, **kwargs)

    variable = cp.Variable(2)
    cvx_cost = cvxpy_total_cost(variable, previous, **kwargs)
    assert cvx_cost.trade_permitted.tolist() == [True, False]
    assert len(cvx_cost.constraints) == 1


@pytest.mark.parametrize("bad_rate", [-0.01, float("nan"), float("inf")])
def test_borrow_and_financing_rates_fail_closed(bad_rate: float) -> None:
    with pytest.raises(CostInputError, match="annual_rate"):
        borrow_cost_numpy([-0.1], annual_rate=bad_rate)
    with pytest.raises(CostInputError, match="annual_rate"):
        borrow_cost_cvxpy(cp.Variable(1), annual_rate=bad_rate)
    with pytest.raises(CostInputError, match="annual_rate"):
        borrow_cost_torch(torch.tensor([-0.1]), annual_rate=bad_rate)
    with pytest.raises(CostInputError, match="annual_rate"):
        financing_cost_numpy([-0.1], annual_rate=bad_rate)
    with pytest.raises(CostInputError, match="annual_rate"):
        financing_cost_cvxpy(cp.Variable(1), annual_rate=bad_rate)
    with pytest.raises(CostInputError, match="annual_rate"):
        financing_cost_torch(torch.tensor([-0.1]), annual_rate=bad_rate)


def test_invalid_reduction_and_nonfinite_decisions_fail_closed() -> None:
    with pytest.raises(ValueError, match="reduction"):
        transaction_cost_numpy(
            0.0,
            full_spread=0.0,
            adv_dollars=1.0,
            impact_volatility=0.0,
            reduction="mean",  # type: ignore[arg-type]
        )
    with pytest.raises(CostInputError, match="delta_weight"):
        transaction_cost_numpy(
            float("nan"),
            full_spread=0.0,
            adv_dollars=1.0,
            impact_volatility=0.0,
        )
    with pytest.raises(CostInputError, match="weight"):
        financing_cost_numpy([float("inf")])
    with pytest.raises(CostInputError, match="weight"):
        borrow_cost_numpy([float("nan")])
