"""Post-close reconciliation for a submitted V6.8 Alpaca paper order plan."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from portfolio_bot.research_v6_8.alpaca_paper import AlpacaPaperClient


class PaperReconciliationError(RuntimeError):
    """A submitted paper plan or its broker reconciliation is invalid."""


class ReconciliationReader(Protocol):
    def get_positions(self) -> Sequence[Mapping[str, Any]]: ...
    def get_order_by_client_id(self, client_order_id: str) -> Mapping[str, Any] | None: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def reconcile_plan(
    *,
    plan: Mapping[str, Any],
    positions: Sequence[Mapping[str, Any]],
    orders: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Compare broker positions and order states with the exact rounded plan."""

    if plan.get("schema") != "research_v6_8_alpaca_order_plan_v1":
        raise PaperReconciliationError("order-plan schema differs")
    targets = {str(key).upper(): int(value) for key, value in plan.get("target_quantities", {}).items()}
    if not targets:
        raise PaperReconciliationError("order plan omits full target quantities")
    actual: dict[str, int] = {}
    for row in positions:
        symbol = str(row.get("symbol", "")).upper()
        quantity = float(row.get("qty", 0.0))
        if not quantity.is_integer():
            raise PaperReconciliationError("broker returned an unsupported fractional position")
        if symbol in actual:
            raise PaperReconciliationError(f"broker duplicated position {symbol}")
        actual[symbol] = int(quantity)
    symbols = sorted(set(targets) | set(actual))
    differences = [
        {
            "symbol": symbol,
            "target_qty": int(targets.get(symbol, 0)),
            "actual_qty": int(actual.get(symbol, 0)),
            "difference_qty": int(actual.get(symbol, 0) - targets.get(symbol, 0)),
        }
        for symbol in symbols
        if int(actual.get(symbol, 0)) != int(targets.get(symbol, 0))
    ]
    order_rows: list[dict[str, Any]] = []
    pending_statuses = {"new", "accepted", "pending_new", "accepted_for_bidding", "partially_filled"}
    failed_statuses = {"canceled", "expired", "rejected", "replaced", "stopped", "suspended"}
    pending = 0
    failed = 0
    for row in plan.get("orders", []):
        client_id = str(row["client_order_id"])
        broker = orders.get(client_id)
        status = "missing" if broker is None else str(broker.get("status", "unknown")).lower()
        pending += int(status in pending_statuses)
        failed += int(status in failed_statuses or status in {"missing", "unknown"})
        order_rows.append(
            {
                "client_order_id": client_id,
                "symbol": str(row["symbol"]).upper(),
                "status": status,
                "filled_qty": None if broker is None else broker.get("filled_qty"),
                "filled_avg_price": None if broker is None else broker.get("filled_avg_price"),
            }
        )
    if not differences and pending == 0 and failed == 0:
        status = "PASS_FILLED_AND_RECONCILED"
    elif failed or (differences and pending == 0):
        status = "FAIL_ORDER_OR_POSITION_MISMATCH"
    else:
        status = "PENDING_BROKER_COMPLETION"
    return {
        "schema": "research_v6_8_paper_reconciliation_v1",
        "status": status,
        "signal_id": plan.get("signal_id"),
        "execution_session": plan.get("execution_session"),
        "target_position_count": sum(value != 0 for value in targets.values()),
        "actual_position_count": sum(value != 0 for value in actual.values()),
        "position_difference_count": len(differences),
        "position_differences": differences,
        "order_count": len(order_rows),
        "pending_order_count": pending,
        "failed_order_count": failed,
        "orders": order_rows,
    }


def materialize_paper_reconciliation(
    *,
    deployment_config_path: Path,
    deployment_receipt_path: Path,
    output: Path,
    client: ReconciliationReader | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    deployment = json.loads(deployment_receipt_path.read_text(encoding="utf-8"))
    if deployment.get("status") != "PAPER_ORDERS_SUBMITTED" or "plan" not in deployment:
        raise PaperReconciliationError("deployment receipt does not bind a submitted order plan")
    config = json.loads(deployment_config_path.read_text(encoding="utf-8"))
    reader = client or AlpacaPaperClient.from_environment(config)
    plan = deployment["plan"]
    order_states = {
        str(row["client_order_id"]): reader.get_order_by_client_id(
            str(row["client_order_id"])
        )
        for row in plan.get("orders", [])
    }
    receipt = reconcile_plan(
        plan=plan,
        positions=list(reader.get_positions()),
        orders=order_states,
    )
    receipt.update(
        {
            "checked_at": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
            "paper_endpoint_locked": config["broker"]["base_url"]
            == "https://paper-api.alpaca.markets",
            "deployment_receipt_path": str(deployment_receipt_path),
            "deployment_receipt_sha256": _sha256(deployment_receipt_path),
            "credentials_persisted": False,
        }
    )
    _atomic_json(output / "receipt.json", receipt)
    return receipt


__all__ = [
    "PaperReconciliationError",
    "materialize_paper_reconciliation",
    "reconcile_plan",
]
