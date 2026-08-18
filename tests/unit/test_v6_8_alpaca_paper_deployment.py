from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from portfolio_bot.research_v6_8.alpaca_paper import (
    AlpacaPaperClient,
    DeploymentError,
    build_order_plan,
    local_readiness,
    submit_order_plan,
    validate_signal,
)


def _config() -> dict:
    return {
        "strategy_id": "research_v6_8_final_paper",
        "frozen_model": {"result_sha256": "model-hash"},
        "signal_contract": {
            "schema": "research_v6_8_paper_signal_v1",
            "maximum_signal_age_hours": 96,
            "allowed_policy_actions": ["monthly_rebalance", "daily_warning_reoptimization", "safety_deleveraging"],
        },
        "portfolio_limits": {
            "maximum_starting_gross": 1.8,
            "operational_safety_gross": 1.95,
            "maximum_absolute_position_weight": 0.3,
            "minimum_net": -0.2,
            "maximum_net": 0.6,
        },
        "account_limits": {"minimum_margin_equity": 2000.0},
        "asset_checks": {"accepted_borrow_statuses": ["easy_to_borrow"]},
        "orders": {
            "type": "market",
            "time_in_force": "cls",
            "minimum_order_notional_dollars": 5.0,
            "latest_submission_time_new_york": "15:45",
        },
        "authorization": {"submission_acknowledgement": "SUBMIT_V6_8_PAPER_ORDERS"},
    }


def _signal(**overrides: object) -> dict:
    result = {
        "schema": "research_v6_8_paper_signal_v1",
        "strategy_id": "research_v6_8_final_paper",
        "model_result_sha256": "model-hash",
        "policy_action": "monthly_rebalance",
        "input_manifest_sha256": "a" * 64,
        "source_max_signal_date": "2026-08-31",
        "signal_id": "2026-09-monthly",
        "generated_at": "2026-08-31T21:05:00+00:00",
        "signal_cutoff_at": "2026-08-31T20:00:00+00:00",
        "execution_session": "2026-09-01",
        "execution_authorized": True,
        "targets": [
            {"symbol": "AAA", "target_weight": 0.2},
            {"symbol": "BBB", "target_weight": -0.1},
        ],
    }
    result.update(overrides)
    return result


def _asset(**overrides: object) -> dict:
    result = {
        "status": "active",
        "tradable": True,
        "marginable": True,
        "shortable": True,
        "borrow_status": "easy_to_borrow",
    }
    result.update(overrides)
    return result


def _account(**overrides: object) -> dict:
    result = {
        "status": "ACTIVE",
        "trading_blocked": False,
        "equity": "100000",
        "multiplier": "2",
        "shorting_enabled": True,
    }
    result.update(overrides)
    return result


def test_stale_signal_is_rejected() -> None:
    with pytest.raises(DeploymentError, match="stale"):
        validate_signal(
            _signal(),
            _config(),
            now=datetime(2026, 9, 5, tzinfo=timezone.utc),
        )


def test_order_plan_reduces_before_increasing_and_uses_deterministic_ids() -> None:
    config = _config()
    signal = _signal()
    targets = validate_signal(
        signal,
        config,
        now=datetime(2026, 9, 1, 14, tzinfo=timezone.utc),
        require_execution_today=True,
    )
    positions = [{"symbol": "AAA", "qty": "250"}, {"symbol": "CCC", "qty": "20"}]
    assets = {symbol: _asset() for symbol in ("AAA", "BBB", "CCC")}
    prices = {"AAA": 100.0, "BBB": 50.0, "CCC": 25.0}
    plan = build_order_plan(
        signal=signal,
        targets=targets,
        account=_account(),
        positions=positions,
        assets=assets,
        prices=prices,
        config=config,
    )
    assert [row["phase"] for row in plan["orders"][:2]] == ["reduce", "reduce"]
    assert all(row["client_order_id"].startswith("v68-") for row in plan["orders"])
    assert len({row["client_order_id"] for row in plan["orders"]}) == len(plan["orders"])
    assert plan["target_gross_after_rounding"] <= 1.8


def test_new_short_requires_current_easy_borrow_status() -> None:
    config = _config()
    signal = _signal()
    targets = validate_signal(
        signal,
        config,
        now=datetime(2026, 9, 1, 14, tzinfo=timezone.utc),
    )
    with pytest.raises(DeploymentError, match="easy to borrow"):
        build_order_plan(
            signal=signal,
            targets=targets,
            account=_account(),
            positions=[],
            assets={"AAA": _asset(), "BBB": _asset(borrow_status="hard_to_borrow")},
            prices={"AAA": 100.0, "BBB": 50.0},
            config=config,
        )


def test_short_book_requires_account_shorting_permission() -> None:
    with pytest.raises(DeploymentError, match="permit shorting"):
        build_order_plan(
            signal=_signal(),
            targets={"AAA": 0.2, "BBB": -0.1},
            account=_account(shorting_enabled=False),
            positions=[],
            assets={"AAA": _asset(), "BBB": _asset()},
            prices={"AAA": 100.0, "BBB": 50.0},
            config=_config(),
        )


def test_suppressed_small_delta_keeps_current_quantity_as_realized_target() -> None:
    plan = build_order_plan(
        signal=_signal(targets=[{"symbol": "AAA", "target_weight": 0.00004}]),
        targets={"AAA": 0.00004},
        account=_account(),
        positions=[{"symbol": "AAA", "qty": "3"}],
        assets={"AAA": _asset()},
        prices={"AAA": 1.0},
        config=_config(),
    )
    assert plan["orders"] == []
    assert plan["target_quantities"] == {"AAA": 3}


def test_live_endpoint_is_impossible() -> None:
    with pytest.raises(DeploymentError, match="paper"):
        AlpacaPaperClient("key", "secret", base_url="https://api.alpaca.markets")


def test_safety_breach_allows_only_reductions() -> None:
    config = _config()
    signal = _signal(targets=[{"symbol": "AAA", "target_weight": 0.3}, {"symbol": "BBB", "target_weight": 0.3}])
    targets = validate_signal(signal, config, now=datetime(2026, 9, 1, 14, tzinfo=timezone.utc))
    with pytest.raises(DeploymentError, match="safety breach"):
        build_order_plan(
            signal=signal,
            targets=targets,
            account=_account(),
            positions=[{"symbol": "AAA", "qty": "2000"}],
            assets={"AAA": _asset(), "BBB": _asset()},
            prices={"AAA": 100.0, "BBB": 100.0},
            config=config,
        )


def test_submission_is_idempotent_for_existing_client_order_id() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.created: list[dict] = []

        def get_clock(self) -> dict:
            return {"is_open": True, "timestamp": "2026-09-01T15:00:00-04:00"}

        def get_open_orders(self) -> list:
            return []

        def get_order_by_client_id(self, client_order_id: str) -> dict | None:
            return {"status": "filled"} if client_order_id == "existing" else None

        def submit_order(self, order: dict) -> dict:
            self.created.append(order)
            return {"id": "new-id", **order, "status": "accepted"}

    plan = {
        "execution_session": "2026-09-01",
        "signal_id": "signal",
        "orders": [
            {"client_order_id": "existing", "symbol": "AAA", "side": "sell", "qty": 1},
            {"client_order_id": "new", "symbol": "BBB", "side": "buy", "qty": 1},
        ],
    }
    client = FakeClient()
    receipt = submit_order_plan(
        client,  # type: ignore[arg-type]
        plan,
        _config(),
        acknowledgement="SUBMIT_V6_8_PAPER_ORDERS",
        now=datetime(2026, 9, 1, 19, tzinfo=timezone.utc),
    )
    assert [row["client_order_id"] for row in client.created] == ["new"]
    assert receipt["idempotent_skips"] == [{"client_order_id": "existing", "status": "filled"}]


def test_production_data_gate_blocks_before_account_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path = tmp_path / "source-contract.json"
    contract_path.write_text("{}\n", "utf-8")
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(
        json.dumps({"status": "BLOCKED_FOR_PAPER_ORDERS"}), "utf-8"
    )
    months = tmp_path / "months"
    months.mkdir()
    (months / "2025-08.parquet").touch()
    config = {
        "broker": {"base_url": "https://paper-api.alpaca.markets"},
        "credentials": {
            "key_environment": "TEST_APCA_KEY",
            "secret_environment": "TEST_APCA_SECRET",
        },
        "local_signal_source": {
            "model_feature_months_directory": str(months),
            "authenticated_latest_holding_month": "2025-08",
            "authenticated_latest_signal_date": "2025-07-31",
        },
        "production_data_gate": {
            "source_contract_path": str(contract_path),
            "source_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            "readiness_receipt_path": str(readiness_path),
            "required_status": "READY_FOR_SIGNAL_MATERIALIZATION",
        },
    }
    monkeypatch.setenv("TEST_APCA_KEY", "configured")
    monkeypatch.setenv("TEST_APCA_SECRET", "configured")
    receipt = local_readiness(config, signal_path=tmp_path / "missing-signal.json")
    assert receipt["status"] == "BLOCKED"
    assert receipt["production_data_gate_status"] == "BLOCKED_FOR_PAPER_ORDERS"
    assert any("production-data gate" in blocker for blocker in receipt["blockers"])
