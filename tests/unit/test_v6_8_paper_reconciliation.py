from __future__ import annotations

from portfolio_bot.research_v6_8.paper_reconciliation import reconcile_plan


def test_reconciliation_passes_only_when_orders_and_positions_match() -> None:
    plan = {
        "schema": "research_v6_8_alpaca_order_plan_v1",
        "signal_id": "2026-09-monthly-test",
        "execution_session": "2026-09-01",
        "target_quantities": {"AAA": 10, "BBB": -4, "OLD": 0},
        "orders": [
            {"client_order_id": "one", "symbol": "AAA"},
            {"client_order_id": "two", "symbol": "BBB"},
            {"client_order_id": "three", "symbol": "OLD"},
        ],
    }
    result = reconcile_plan(
        plan=plan,
        positions=[{"symbol": "AAA", "qty": "10"}, {"symbol": "BBB", "qty": "-4"}],
        orders={
            "one": {"status": "filled", "filled_qty": "10"},
            "two": {"status": "filled", "filled_qty": "4"},
            "three": {"status": "filled", "filled_qty": "2"},
        },
    )
    assert result["status"] == "PASS_FILLED_AND_RECONCILED"
    assert result["position_difference_count"] == 0


def test_reconciliation_reports_pending_partial_fill() -> None:
    plan = {
        "schema": "research_v6_8_alpaca_order_plan_v1",
        "signal_id": "x",
        "execution_session": "2026-09-01",
        "target_quantities": {"AAA": 10},
        "orders": [{"client_order_id": "one", "symbol": "AAA"}],
    }
    result = reconcile_plan(
        plan=plan,
        positions=[{"symbol": "AAA", "qty": "5"}],
        orders={"one": {"status": "partially_filled", "filled_qty": "5"}},
    )
    assert result["status"] == "PENDING_BROKER_COMPLETION"
    assert result["position_difference_count"] == 1


def test_reconciliation_fails_filled_order_with_position_mismatch() -> None:
    plan = {
        "schema": "research_v6_8_alpaca_order_plan_v1",
        "signal_id": "x",
        "execution_session": "2026-09-01",
        "target_quantities": {"AAA": 10},
        "orders": [{"client_order_id": "one", "symbol": "AAA"}],
    }
    result = reconcile_plan(
        plan=plan,
        positions=[{"symbol": "AAA", "qty": "9"}],
        orders={"one": {"status": "filled", "filled_qty": "10"}},
    )
    assert result["status"] == "FAIL_ORDER_OR_POSITION_MISMATCH"
