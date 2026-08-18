"""Fail-closed Alpaca paper adapter for the frozen V6.8 policy.

Credentials are resolved from the environment or dedicated macOS Keychain
entries. This module refuses any API domain other than Alpaca's paper endpoint
and never logs authentication values.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import requests

from portfolio_bot.research_v6_8.credentials import (
    PaperCredentialError,
    load_paper_credentials,
    paper_credential_presence,
)


PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"
NEW_YORK = ZoneInfo("America/New_York")


class DeploymentError(RuntimeError):
    """A paper deployment safety or readiness check failed."""


def _as_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise DeploymentError(f"timestamp must include a timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _latest_month(months_directory: Path) -> str | None:
    months = sorted(path.stem for path in months_directory.glob("????-??.parquet"))
    return months[-1] if months else None


def local_readiness(
    config: Mapping[str, Any],
    *,
    now: datetime | None = None,
    signal_path: Path | None = None,
) -> dict[str, Any]:
    """Return a secret-free, local-only deployment readiness receipt."""

    current = (now or datetime.now(timezone.utc)).astimezone(NEW_YORK)
    source = config.get("local_signal_source")
    feature_directory = (
        Path(source["model_feature_months_directory"]) if source else None
    )
    latest_month = _latest_month(feature_directory) if feature_directory else None
    credentials = config["credentials"]
    blockers: list[str] = []
    data_gate_status: str | None = None
    data_gate = config.get("production_data_gate")
    if data_gate:
        contract_path = Path(data_gate["source_contract_path"])
        if not contract_path.is_file():
            blockers.append("production feature-source contract is missing")
        elif _sha256(contract_path) != data_gate["source_contract_sha256"]:
            blockers.append("production feature-source contract identity mismatch")
        readiness_value = data_gate.get("readiness_receipt_path")
        if readiness_value:
            readiness_path = Path(readiness_value)
            if not readiness_path.is_file():
                blockers.append("production-data readiness receipt is missing")
            else:
                try:
                    data_receipt = json.loads(readiness_path.read_text("utf-8"))
                    data_gate_status = str(data_receipt.get("status"))
                except (OSError, ValueError, TypeError):
                    blockers.append("production-data readiness receipt is unreadable")
                else:
                    if data_gate_status != str(data_gate["required_status"]):
                        blockers.append(
                            f"production-data gate is {data_gate_status}; expected {data_gate['required_status']}"
                        )
    signal_payload: Mapping[str, Any] | None = None
    if signal_path is not None and signal_path.is_file():
        try:
            signal_payload = json.loads(signal_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            blockers.append("signal package is unreadable")
    signal_gate = config.get("signal_source_gate")
    if signal_gate:
        portfolio_config = Path(signal_gate["prospective_portfolio_config_path"])
        if not portfolio_config.is_file():
            blockers.append("prospective portfolio config is missing")
        elif _sha256(portfolio_config) != signal_gate["prospective_portfolio_config_sha256"]:
            blockers.append("prospective portfolio config identity mismatch")
        if signal_path is not None and signal_path.is_file():
            producer_path = signal_path.parent / str(signal_gate["producer_receipt_name"])
            if not producer_path.is_file():
                blockers.append("signal producer receipt is missing")
            else:
                try:
                    producer = json.loads(producer_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    blockers.append("signal producer receipt is unreadable")
                else:
                    if producer.get("status") != signal_gate["required_producer_status"]:
                        blockers.append("signal producer is not executable")
                    if producer.get("signal_sha256") != _sha256(signal_path):
                        blockers.append("signal differs from its producer receipt")
            if signal_gate.get("require_execution_authorized") and not bool(
                (signal_payload or {}).get("execution_authorized", False)
            ):
                blockers.append("signal is not authorized for execution")
    if config["broker"]["base_url"] != PAPER_BASE_URL:
        blockers.append("broker endpoint is not the immutable Alpaca paper endpoint")
    if source and latest_month != source["authenticated_latest_holding_month"]:
        blockers.append("local feature-source identity differs from the frozen readiness record")
    credential_presence = paper_credential_presence(
        key_environment=credentials["key_environment"],
        secret_environment=credentials["secret_environment"],
    )
    if not credential_presence["key"]:
        blockers.append(f"missing environment variable {credentials['key_environment']}")
    if not credential_presence["secret"]:
        blockers.append(f"missing environment variable {credentials['secret_environment']}")
    if signal_path is None:
        blockers.append("no current V6.8 signal package was supplied")
    elif not signal_path.is_file():
        blockers.append(f"signal package does not exist: {signal_path}")
    return {
        "schema": "research_v6_8_alpaca_local_readiness_v1",
        "status": "READY_FOR_ACCOUNT_PREFLIGHT" if not blockers else "BLOCKED",
        "checked_at": current.isoformat(),
        "paper_endpoint_locked": config["broker"]["base_url"] == PAPER_BASE_URL,
        "credentials_present": credential_presence,
        "latest_authenticated_holding_month": latest_month,
        "latest_authenticated_signal_date": (
            source["authenticated_latest_signal_date"] if source else
            (signal_payload or {}).get("source_max_signal_date")
        ),
        "production_data_gate_status": data_gate_status,
        "signal_package": str(signal_path) if signal_path else None,
        "blockers": blockers,
    }


def validate_signal(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    now: datetime,
    require_execution_today: bool = False,
) -> dict[str, float]:
    """Validate identity, freshness, schedule, and portfolio-level constraints."""

    expected = config["signal_contract"]
    if payload.get("schema") != expected["schema"]:
        raise DeploymentError("signal schema is not authorized")
    if payload.get("strategy_id") != config["strategy_id"]:
        raise DeploymentError("signal strategy identity mismatch")
    if payload.get("model_result_sha256") != config["frozen_model"]["result_sha256"]:
        raise DeploymentError("signal model identity mismatch")
    if payload.get("policy_action") not in expected["allowed_policy_actions"]:
        raise DeploymentError("signal policy action is not authorized")
    if payload.get("execution_authorized") is not True:
        raise DeploymentError("signal is not authorized for execution")
    manifest_sha256 = str(payload.get("input_manifest_sha256", ""))
    if len(manifest_sha256) != 64 or any(value not in "0123456789abcdef" for value in manifest_sha256.lower()):
        raise DeploymentError("signal input-manifest identity is missing or invalid")
    generated = _as_utc(str(payload["generated_at"]))
    cutoff = _as_utc(str(payload["signal_cutoff_at"]))
    current = now.astimezone(timezone.utc)
    if generated > current:
        raise DeploymentError("signal generation timestamp is in the future")
    age_hours = (current - generated).total_seconds() / 3600.0
    if age_hours > float(expected["maximum_signal_age_hours"]):
        raise DeploymentError("signal is stale")
    if cutoff > generated:
        raise DeploymentError("signal cutoff occurs after signal generation")
    if date.fromisoformat(str(payload["source_max_signal_date"])) != cutoff.astimezone(NEW_YORK).date():
        raise DeploymentError("signal source date does not match the point-in-time cutoff date")
    execution_session = date.fromisoformat(str(payload["execution_session"]))
    if execution_session <= cutoff.astimezone(NEW_YORK).date():
        raise DeploymentError("execution session must follow the signal cutoff date")
    if require_execution_today and execution_session != now.astimezone(NEW_YORK).date():
        raise DeploymentError("submission is allowed only on the signal's execution session")

    targets: dict[str, float] = {}
    for row in payload.get("targets", []):
        symbol = str(row["symbol"]).strip().upper()
        if not symbol or symbol in targets:
            raise DeploymentError(f"invalid or duplicate signal symbol: {symbol!r}")
        weight = float(row["target_weight"])
        if not math.isfinite(weight):
            raise DeploymentError(f"non-finite target weight for {symbol}")
        targets[symbol] = weight
    if not targets:
        raise DeploymentError("signal contains no target positions")

    limits = config["portfolio_limits"]
    maximum_position = float(limits["maximum_absolute_position_weight"])
    if max(abs(value) for value in targets.values()) > maximum_position + 1e-9:
        raise DeploymentError("signal exceeds the position-weight limit")
    gross = sum(abs(value) for value in targets.values())
    net = sum(targets.values())
    if gross > float(limits["maximum_starting_gross"]) + 1e-9:
        raise DeploymentError("signal exceeds the 1.80x starting-gross limit")
    if not float(limits["minimum_net"]) - 1e-9 <= net <= float(limits["maximum_net"]) + 1e-9:
        raise DeploymentError("signal exceeds the frozen net-exposure interval")
    return targets


def _position_quantities(positions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in positions:
        quantity = float(row["qty"])
        if not quantity.is_integer():
            raise DeploymentError("existing fractional position is unsupported by the whole-share policy")
        result[str(row["symbol"]).upper()] = int(quantity)
    return result


def _borrow_is_easy(asset: Mapping[str, Any], accepted: set[str]) -> bool:
    status = str(asset.get("borrow_status") or "").lower()
    if status:
        return status in accepted
    return bool(asset.get("easy_to_borrow", False))


def _client_order_id(strategy_id: str, signal_id: str, symbol: str, target_qty: int) -> str:
    digest = hashlib.sha256(
        f"{strategy_id}|{signal_id}|{symbol}|{target_qty}".encode("utf-8")
    ).hexdigest()[:20]
    return f"v68-{digest}"


def build_order_plan(
    *,
    signal: Mapping[str, Any],
    targets: Mapping[str, float],
    account: Mapping[str, Any],
    positions: Sequence[Mapping[str, Any]],
    assets: Mapping[str, Mapping[str, Any]],
    prices: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert full-book target weights into idempotent whole-share MOC orders."""

    if str(account.get("status", "")).upper() != "ACTIVE":
        raise DeploymentError("paper account is not active")
    if bool(account.get("trading_blocked", False)):
        raise DeploymentError("paper account is trading-blocked")
    equity = float(account["equity"])
    if not math.isfinite(equity) or equity <= 0:
        raise DeploymentError("paper account equity is invalid")

    current = _position_quantities(positions)
    symbols = sorted(set(targets) | set(current))
    missing_prices = [symbol for symbol in symbols if not math.isfinite(float(prices.get(symbol, math.nan))) or float(prices.get(symbol, 0.0)) <= 0]
    if missing_prices:
        raise DeploymentError(f"missing executable prices for: {', '.join(missing_prices)}")
    missing_assets = [symbol for symbol in symbols if symbol not in assets]
    if missing_assets:
        raise DeploymentError(f"missing asset status for: {', '.join(missing_assets)}")

    target_qty: dict[str, int] = {
        symbol: math.trunc(float(targets.get(symbol, 0.0)) * equity / float(prices[symbol]))
        for symbol in symbols
    }
    has_short = any(quantity < 0 for quantity in target_qty.values())
    requires_margin = has_short or sum(abs(value) for value in targets.values()) > 1.0
    if requires_margin:
        if equity < float(config["account_limits"]["minimum_margin_equity"]):
            raise DeploymentError("account equity is below the margin/short minimum")
        if float(account.get("multiplier", 0.0)) < 2.0:
            raise DeploymentError("paper account does not currently have 2x margin enabled")
    if has_short and not bool(account.get("shorting_enabled", False)):
        raise DeploymentError("paper account does not currently permit shorting")

    current_gross = sum(abs(current.get(symbol, 0)) * float(prices[symbol]) for symbol in symbols) / equity

    accepted_borrow = {str(value).lower() for value in config["asset_checks"]["accepted_borrow_statuses"]}
    orders: list[dict[str, Any]] = []
    for symbol in symbols:
        asset = assets[symbol]
        old = int(current.get(symbol, 0))
        target = int(target_qty[symbol])
        delta = target - old
        if delta == 0:
            continue
        if not bool(asset.get("tradable", False)) or str(asset.get("status", "active")).lower() != "active":
            raise DeploymentError(f"{symbol} is not active and tradable")
        opening_short = target < min(old, 0)
        if opening_short:
            if not bool(asset.get("shortable", False)):
                raise DeploymentError(f"{symbol} is not currently shortable")
            if not _borrow_is_easy(asset, accepted_borrow):
                raise DeploymentError(f"{symbol} is not currently easy to borrow")
        increasing_absolute_risk = abs(target) > abs(old) or (old != 0 and target * old < 0)
        if increasing_absolute_risk and not bool(asset.get("marginable", False)):
            raise DeploymentError(f"{symbol} is not marginable for an exposure increase")
        phase = "reduce" if target == 0 or (target * old >= 0 and abs(target) < abs(old)) else "increase_or_reverse"
        notional = abs(delta) * float(prices[symbol])
        if notional < float(config["orders"]["minimum_order_notional_dollars"]):
            # The broker plan is authoritative at whole-share grain.  If an
            # immaterial delta is deliberately suppressed, retain the current
            # quantity as the realized target so post-close reconciliation and
            # gross/net accounting describe the book we actually asked the
            # broker to hold.
            target_qty[symbol] = old
            continue
        orders.append(
            {
                "symbol": symbol,
                "side": "buy" if delta > 0 else "sell",
                "qty": abs(delta),
                "type": config["orders"]["type"],
                "time_in_force": config["orders"]["time_in_force"],
                "extended_hours": False,
                "client_order_id": _client_order_id(
                    config["strategy_id"], str(signal["signal_id"]), symbol, target
                ),
                "phase": phase,
                "current_qty": old,
                "target_qty": target,
                "reference_price": float(prices[symbol]),
                "estimated_notional": notional,
            }
        )
    orders.sort(key=lambda row: (row["phase"] != "reduce", row["symbol"]))
    realized_target_gross = sum(abs(target_qty[s]) * float(prices[s]) for s in symbols) / equity
    realized_target_net = sum(target_qty[s] * float(prices[s]) for s in symbols) / equity
    if realized_target_gross > float(config["portfolio_limits"]["maximum_starting_gross"]) + 1e-6:
        raise DeploymentError("rounded target portfolio exceeds the starting-gross limit")
    safety_limit = float(config["portfolio_limits"]["operational_safety_gross"])
    if current_gross > safety_limit + 1e-6 and any(row["phase"] != "reduce" for row in orders):
        raise DeploymentError("gross safety breach permits only reduction orders")
    return {
        "schema": "research_v6_8_alpaca_order_plan_v1",
        "strategy_id": config["strategy_id"],
        "signal_id": signal["signal_id"],
        "execution_session": signal["execution_session"],
        "account_equity": equity,
        "current_gross": current_gross,
        "target_gross_after_rounding": realized_target_gross,
        "target_net_after_rounding": realized_target_net,
        "target_quantities": {symbol: int(target_qty[symbol]) for symbol in symbols},
        "order_count": len(orders),
        "estimated_one_way_notional": sum(row["estimated_notional"] for row in orders),
        "orders": orders,
    }


@dataclass
class AlpacaPaperClient:
    key: str
    secret: str
    base_url: str = PAPER_BASE_URL
    data_url: str = DATA_BASE_URL
    timeout_seconds: float = 20.0
    credential_source: str = "direct"

    def __post_init__(self) -> None:
        if self.base_url.rstrip("/") != PAPER_BASE_URL:
            raise DeploymentError("only Alpaca's paper trading endpoint is permitted")
        self.base_url = self.base_url.rstrip("/")
        self.data_url = self.data_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}
        )

    @classmethod
    def from_environment(cls, config: Mapping[str, Any]) -> "AlpacaPaperClient":
        credentials = config["credentials"]
        try:
            resolved = load_paper_credentials(
                key_environment=credentials["key_environment"],
                secret_environment=credentials["secret_environment"],
            )
        except PaperCredentialError as error:
            raise DeploymentError(str(error)) from error
        return cls(
            key=resolved.key_id,
            secret=resolved.secret_key,
            base_url=config["broker"]["base_url"],
            data_url=config["broker"]["data_url"],
            timeout_seconds=float(config["broker"]["timeout_seconds"]),
            credential_source=resolved.source,
        )

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        response = self._session.request(method, url, timeout=self.timeout_seconds, **kwargs)
        if response.status_code >= 400:
            message = response.text[:500]
            raise DeploymentError(f"Alpaca paper API {method} failed ({response.status_code}): {message}")
        return response.json() if response.content else None

    def get_account(self) -> Mapping[str, Any]:
        return self._request("GET", f"{self.base_url}/v2/account")

    def get_clock(self) -> Mapping[str, Any]:
        return self._request("GET", f"{self.base_url}/v2/clock")

    def get_positions(self) -> Sequence[Mapping[str, Any]]:
        return self._request("GET", f"{self.base_url}/v2/positions")

    def get_all_assets(self) -> Sequence[Mapping[str, Any]]:
        """Return the active US-equity master without per-symbol API calls."""

        return self._request(
            "GET",
            f"{self.base_url}/v2/assets",
            params={"status": "active", "asset_class": "us_equity"},
        )

    def get_calendar(self, start: date, end: date) -> Sequence[Mapping[str, Any]]:
        return self._request(
            "GET",
            f"{self.base_url}/v2/calendar",
            params={"start": start.isoformat(), "end": end.isoformat()},
        )

    def get_open_orders(self) -> Sequence[Mapping[str, Any]]:
        return self._request("GET", f"{self.base_url}/v2/orders", params={"status": "open", "limit": 500})

    def get_asset(self, symbol: str) -> Mapping[str, Any]:
        return self._request("GET", f"{self.base_url}/v2/assets/{symbol}")

    def get_latest_trades(self, symbols: Sequence[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        for start in range(0, len(symbols), 200):
            batch = list(symbols[start : start + 200])
            payload = self._request(
                "GET",
                f"{self.data_url}/v2/stocks/trades/latest",
                params={"symbols": ",".join(batch), "feed": "iex"},
            )
            for symbol, row in payload.get("trades", {}).items():
                prices[str(symbol).upper()] = float(row["p"])
        return prices

    def get_order_by_client_id(self, client_order_id: str) -> Mapping[str, Any] | None:
        response = self._session.get(
            f"{self.base_url}/v2/orders:by_client_order_id",
            params={"client_order_id": client_order_id},
            timeout=self.timeout_seconds,
        )
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise DeploymentError(f"Alpaca paper order lookup failed ({response.status_code})")
        return response.json()

    def submit_order(self, order: Mapping[str, Any]) -> Mapping[str, Any]:
        body = {key: value for key, value in order.items() if key in {
            "symbol", "side", "qty", "type", "time_in_force", "extended_hours", "client_order_id"
        }}
        return self._request("POST", f"{self.base_url}/v2/orders", json=body)


def submit_order_plan(
    client: AlpacaPaperClient,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    acknowledgement: str,
    now: datetime,
) -> dict[str, Any]:
    """Submit an idempotent paper-only order plan after final clock checks."""

    if acknowledgement != config["authorization"]["submission_acknowledgement"]:
        raise DeploymentError("paper-order acknowledgement is missing or incorrect")
    clock = client.get_clock()
    if not bool(clock.get("is_open", False)):
        raise DeploymentError("market is closed; paper MOC orders were not submitted")
    clock_time = _as_utc(str(clock["timestamp"])).astimezone(NEW_YORK)
    if clock_time.date().isoformat() != str(plan["execution_session"]):
        raise DeploymentError("market clock date does not match the signal execution session")
    deadline = config["orders"]["latest_submission_time_new_york"]
    hours, minutes = (int(value) for value in deadline.split(":"))
    if (clock_time.hour, clock_time.minute) >= (hours, minutes):
        raise DeploymentError("the market-on-close submission deadline has passed")
    open_orders = list(client.get_open_orders())
    if open_orders:
        raise DeploymentError("open broker orders exist; refusing to create an overlapping rebalance")

    submitted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for order in plan["orders"]:
        existing = client.get_order_by_client_id(str(order["client_order_id"]))
        if existing is not None:
            skipped.append({"client_order_id": order["client_order_id"], "status": existing.get("status")})
            continue
        response = client.submit_order(order)
        submitted.append(
            {
                "id": response.get("id"),
                "client_order_id": response.get("client_order_id"),
                "symbol": response.get("symbol"),
                "side": response.get("side"),
                "qty": response.get("qty"),
                "status": response.get("status"),
            }
        )
    return {
        "schema": "research_v6_8_alpaca_submission_receipt_v1",
        "submitted_at": now.astimezone(timezone.utc).isoformat(),
        "paper_endpoint": PAPER_BASE_URL,
        "signal_id": plan["signal_id"],
        "submitted": submitted,
        "idempotent_skips": skipped,
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "AlpacaPaperClient",
    "DeploymentError",
    "PAPER_BASE_URL",
    "atomic_json",
    "build_order_plan",
    "local_readiness",
    "read_json",
    "submit_order_plan",
    "validate_signal",
]
