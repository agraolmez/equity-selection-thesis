"""Read-only, secret-free Alpaca paper-account snapshot for V6.8."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import pandas as pd

from portfolio_bot.research_v6_8.alpaca_paper import AlpacaPaperClient, DeploymentError


class BrokerSnapshotError(RuntimeError):
    """The paper broker snapshot failed an identity or account check."""


class BrokerReader(Protocol):
    def get_account(self) -> Mapping[str, Any]: ...
    def get_positions(self) -> Sequence[Mapping[str, Any]]: ...
    def get_open_orders(self) -> Sequence[Mapping[str, Any]]: ...
    def get_all_assets(self) -> Sequence[Mapping[str, Any]]: ...
    def get_calendar(self, start: date, end: date) -> Sequence[Mapping[str, Any]]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _authenticated_universe(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    receipt_path = path.parent / "receipt.json"
    if not path.is_file() or not receipt_path.is_file():
        raise BrokerSnapshotError("current universe or its receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    digest = _sha256(path)
    if receipt.get("status") != "PASS" or digest not in json.dumps(receipt, sort_keys=True):
        raise BrokerSnapshotError("current universe is not authenticated by a passing receipt")
    universe = pd.read_parquet(path)
    required = {"live_security_id", "TickerSymbol", "holding_month", "signal_date"}
    if required - set(universe) or universe["live_security_id"].duplicated().any():
        raise BrokerSnapshotError("current universe has an invalid security grain")
    return universe, {
        "path": str(path),
        "sha256": digest,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
    }


def _safe_account(account: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "status", "currency", "cash", "equity", "buying_power",
        "regt_buying_power", "multiplier", "trading_blocked",
        "account_blocked", "shorting_enabled", "pattern_day_trader",
    )
    return {field: account.get(field) for field in fields}


def _position_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    columns = (
        "symbol", "asset_id", "asset_class", "exchange", "qty", "side",
        "market_value", "cost_basis", "avg_entry_price", "current_price",
        "unrealized_pl", "unrealized_plpc",
    )
    return pd.DataFrame([{column: row.get(column) for column in columns} for row in rows], columns=columns)


def _asset_frame(rows: Sequence[Mapping[str, Any]], symbols: set[str]) -> pd.DataFrame:
    columns = (
        "id", "class", "exchange", "symbol", "name", "status", "tradable",
        "marginable", "maintenance_margin_requirement", "shortable",
        "easy_to_borrow", "fractionable", "attributes",
    )
    selected = []
    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        if symbol in symbols:
            selected.append({column: row.get(column) for column in columns})
    return pd.DataFrame(selected, columns=columns).sort_values("symbol", kind="mergesort")


def materialize_live_broker_snapshot(
    *,
    deployment_config_path: Path,
    universe_path: Path,
    output: Path,
    client: BrokerReader | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    config = json.loads(deployment_config_path.read_text(encoding="utf-8"))
    if config.get("broker", {}).get("base_url") != "https://paper-api.alpaca.markets":
        raise BrokerSnapshotError("broker snapshot is restricted to Alpaca paper")
    universe, universe_binding = _authenticated_universe(universe_path)
    reader: BrokerReader
    if client is None:
        try:
            reader = AlpacaPaperClient.from_environment(config)
        except DeploymentError as error:
            raise BrokerSnapshotError(str(error)) from error
    else:
        reader = client
    account = dict(reader.get_account())
    if str(account.get("status", "")).upper() != "ACTIVE":
        raise BrokerSnapshotError("paper account is not active")
    if bool(account.get("account_blocked", False)) or bool(account.get("trading_blocked", False)):
        raise BrokerSnapshotError("paper account is blocked")
    if float(account.get("equity", 0.0)) <= 0.0:
        raise BrokerSnapshotError("paper account equity is not positive")
    positions = list(reader.get_positions())
    open_orders = list(reader.get_open_orders())
    assets = list(reader.get_all_assets())
    universe_symbols = set(universe["TickerSymbol"].astype(str).str.upper())
    position_symbols = {str(row.get("symbol", "")).upper() for row in positions}
    assets_frame = _asset_frame(assets, universe_symbols | position_symbols)
    positions_frame = _position_frame(positions)
    current = now or datetime.now(timezone.utc)
    calendar_start = current.date() - timedelta(days=40)
    calendar_end = current.date() + timedelta(days=45)
    calendar_frame = pd.DataFrame(list(reader.get_calendar(calendar_start, calendar_end)))
    if calendar_frame.empty or "date" not in calendar_frame:
        raise BrokerSnapshotError("Alpaca trading calendar snapshot is empty")
    paths = {
        "account": output / "account.json",
        "positions": output / "positions.parquet",
        "assets": output / "assets.parquet",
        "calendar": output / "calendar.parquet",
    }
    _atomic_json(paths["account"], _safe_account(account))
    _atomic_parquet(paths["positions"], positions_frame)
    _atomic_parquet(paths["assets"], assets_frame)
    _atomic_parquet(paths["calendar"], calendar_frame)
    missing_asset_symbols = sorted(universe_symbols - set(assets_frame["symbol"].astype(str)))
    receipt = {
        "schema": "research_v6_8_live_broker_snapshot_receipt_v1",
        "status": "PASS" if not open_orders else "BLOCKED_OPEN_ORDERS",
        "captured_at": current.astimezone(timezone.utc).isoformat(),
        "paper_endpoint_locked": True,
        "account_equity": float(account["equity"]),
        "position_count": len(positions_frame),
        "open_order_count": len(open_orders),
        "universe_rows": len(universe),
        "matched_active_asset_rows": len(assets_frame),
        "missing_active_asset_count": len(missing_asset_symbols),
        "missing_active_asset_symbols": missing_asset_symbols,
        "universe": universe_binding,
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "credential_source": str(
            getattr(reader, "credential_source", "injected_test_client")
        ),
        "credentials_in_artifact": False,
    }
    _atomic_json(output / "receipt.json", receipt)
    return receipt


__all__ = ["BrokerSnapshotError", "materialize_live_broker_snapshot"]
