#!/usr/bin/env python3
"""Preflight, plan, or submit the frozen V6.8 policy to Alpaca paper trading."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.research_v6_8.alpaca_paper import (  # noqa: E402
    AlpacaPaperClient,
    DeploymentError,
    atomic_json,
    build_order_plan,
    local_readiness,
    read_json,
    submit_order_plan,
    validate_signal,
)

DEFAULT_CONFIG = ROOT / "configs/research_v6_8/alpaca_paper_deployment.json"
DEFAULT_OUTPUT = ROOT / "paper_trading/research_v6_8/runs"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_account(account: dict[str, Any]) -> dict[str, Any]:
    fields = ["status", "currency", "cash", "equity", "buying_power", "regt_buying_power", "multiplier", "trading_blocked", "account_blocked", "shorting_enabled"]
    return {field: account.get(field) for field in fields}


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config_path = args.config.resolve()
    config = read_json(config_path)
    now = datetime.now(timezone.utc)
    signal_path = args.signal.resolve() if args.signal else None
    readiness = local_readiness(config, now=now, signal_path=signal_path)
    receipt: dict[str, Any] = {
        "schema": "research_v6_8_alpaca_deployment_run_v1",
        "mode": args.mode,
        "created_at": now.isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "local_readiness": readiness,
    }
    if args.mode == "local-preflight":
        return receipt, 0 if readiness["status"] != "BLOCKED" else 2
    if readiness["status"] == "BLOCKED":
        raise DeploymentError(
            "local deployment readiness is blocked; no Alpaca account or market-data calls were made"
        )
    if signal_path is None:
        raise DeploymentError("--signal is required for account-preflight, plan, and submit")

    signal = read_json(signal_path)
    targets = validate_signal(
        signal,
        config,
        now=now,
        require_execution_today=args.mode == "submit",
    )
    client = AlpacaPaperClient.from_environment(config)
    account = dict(client.get_account())
    clock = dict(client.get_clock())
    positions = list(client.get_positions())
    open_orders = list(client.get_open_orders())
    receipt["account"] = _safe_account(account)
    receipt["clock"] = clock
    receipt["position_count"] = len(positions)
    receipt["open_order_count"] = len(open_orders)
    if args.mode == "account-preflight":
        receipt["status"] = "PASS" if not open_orders else "BLOCKED_OPEN_ORDERS"
        return receipt, 0 if not open_orders else 2
    if open_orders:
        raise DeploymentError("open broker orders exist; order planning is blocked")

    symbols = sorted(set(targets) | {str(row["symbol"]).upper() for row in positions})
    active_assets = {
        str(row.get("symbol", "")).upper(): dict(row)
        for row in client.get_all_assets()
    }
    assets = {symbol: active_assets[symbol] for symbol in symbols if symbol in active_assets}
    prices = client.get_latest_trades(symbols)
    plan = build_order_plan(
        signal=signal,
        targets=targets,
        account=account,
        positions=positions,
        assets=assets,
        prices=prices,
        config=config,
    )
    receipt["plan"] = plan
    receipt["status"] = "PLANNED_NOT_SUBMITTED"
    if args.mode == "submit":
        receipt["submission"] = submit_order_plan(
            client,
            plan,
            config,
            acknowledgement=args.acknowledge,
            now=now,
        )
        receipt["status"] = "PAPER_ORDERS_SUBMITTED"
    return receipt, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--signal", type=Path)
    parser.add_argument(
        "--mode",
        choices=["local-preflight", "account-preflight", "plan", "submit"],
        default="local-preflight",
    )
    parser.add_argument("--acknowledge", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output.resolve() / f"{timestamp}-{args.mode}"
    try:
        receipt, exit_code = _run(args)
    except Exception as exc:
        receipt = {
            "schema": "research_v6_8_alpaca_deployment_run_v1",
            "mode": args.mode,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        exit_code = 2
    atomic_json(output / "receipt.json", receipt)
    print(json.dumps({"output": str(output), **receipt}, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
