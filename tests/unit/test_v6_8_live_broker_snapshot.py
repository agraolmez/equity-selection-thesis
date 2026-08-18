from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio_bot.research_v6_8.live_broker_snapshot import materialize_live_broker_snapshot


class FakeBroker:
    def get_account(self):
        return {"status": "ACTIVE", "equity": "100000", "cash": "100000", "trading_blocked": False, "account_blocked": False}
    def get_positions(self):
        return [{"symbol": "AAA", "qty": "10", "market_value": "1000"}]
    def get_open_orders(self):
        return []
    def get_all_assets(self):
        return [{"symbol": "AAA", "class": "us_equity", "status": "active", "tradable": True, "marginable": True, "shortable": True, "easy_to_borrow": True}]
    def get_calendar(self, start: date, end: date):
        return [{"date": "2026-08-31", "open": "09:30", "close": "16:00"}, {"date": "2026-09-01", "open": "09:30", "close": "16:00"}]


def test_materializes_secret_free_broker_snapshot(tmp_path: Path) -> None:
    config = tmp_path / "deployment.json"
    config.write_text(json.dumps({"broker": {"base_url": "https://paper-api.alpaca.markets"}}))
    source = tmp_path / "universe" / "universe.parquet"
    source.parent.mkdir()
    pd.DataFrame({"live_security_id": ["1"], "TickerSymbol": ["AAA"], "holding_month": ["2026-09"], "signal_date": [pd.Timestamp("2026-08-31")]}).to_parquet(source, index=False)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (source.parent / "receipt.json").write_text(json.dumps({"status": "PASS", "universe_sha256": digest}))
    receipt = materialize_live_broker_snapshot(
        deployment_config_path=config, universe_path=source,
        output=tmp_path / "output", client=FakeBroker(),
        now=datetime(2026, 8, 31, 21, tzinfo=timezone.utc),
    )
    assert receipt["status"] == "PASS"
    assert receipt["matched_active_asset_rows"] == 1
    assert "key" not in (tmp_path / "output" / "receipt.json").read_text().lower()
