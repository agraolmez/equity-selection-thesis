from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_bot.research_v6_1.portfolio import FACTOR_NAMES
from portfolio_bot.research_v6_8.prospective_portfolio import materialize_prospective_portfolio


ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_current_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    n = 200
    ids = [f"s{index:03d}" for index in range(n)]
    symbols = [f"S{index:03d}" for index in range(n)]
    # High long ranks agree with high FF5 ranks; high short ranks agree with low FF5 ranks.
    ff5 = np.linspace(0.0, 1.0, n)
    long_score = np.linspace(0.0, 1.0, n)
    short_score = np.linspace(1.0, 0.0, n)
    scores = pd.DataFrame({
        "live_security_id": ids, "symbol": symbols, "permno": np.arange(n) + 1,
        "holding_month": "2026-09", "signal_date": pd.Timestamp("2026-08-31"),
        "ff5_momentum_score": ff5, "long_net_score": long_score,
        "short_net_score": short_score,
    })
    selector = tmp_path / "selector"
    selector.mkdir()
    scores_path = selector / "selector_scores.parquet"
    scores.to_parquet(scores_path, index=False)
    (selector / "receipt.json").write_text(json.dumps({
        "status": "PASS", "forecast_month": "2026-09", "scores_sha256": _sha(scores_path)
    }))

    risk = pd.DataFrame({
        "live_security_id": ids, "symbol": symbols, "long_eligible": True,
        "short_eligible": True, "new_long_trade_allowed": True,
        "new_short_trade_allowed": True, "risk_status": "PRESENT",
        "cost_input_status": "PRESENT", "specific_variance_monthly": 0.0001,
        "full_spread_20d": 0.001, "adv20_usd": 100_000_000.0,
        "impact_volatility_daily": 0.01, "primary_borrow_rate_annual": 0.01,
        "stress_borrow_rate_annual": 0.03,
        "sic2_industry_category": [str(index % 20) for index in range(n)],
        **{f"beta_risk_{name}": 0.0 for name in FACTOR_NAMES},
    })
    covariance = pd.DataFrame([
        {"factor_row": left, "factor_col": right,
         "covariance_monthly": 0.001 if left == right else 0.0}
        for left in FACTOR_NAMES for right in FACTOR_NAMES
    ])
    risk_root = tmp_path / "risk"
    risk_root.mkdir()
    risk_path = risk_root / "risk_cost.parquet"
    covariance_path = risk_root / "factor_covariance.parquet"
    risk.to_parquet(risk_path, index=False)
    covariance.to_parquet(covariance_path, index=False)
    (risk_root / "receipt.json").write_text(json.dumps({
        "status": "PASS", "risk_cost_sha256": _sha(risk_path),
        "factor_covariance_sha256": _sha(covariance_path),
    }))

    broker = tmp_path / "broker"
    broker.mkdir()
    account = broker / "account.json"
    positions = broker / "positions.parquet"
    assets = broker / "assets.parquet"
    calendar = broker / "calendar.parquet"
    account.write_text(json.dumps({"status": "ACTIVE", "equity": "100000", "cash": "100000"}))
    pd.DataFrame(columns=["symbol", "market_value"]).to_parquet(positions, index=False)
    pd.DataFrame({
        "symbol": symbols, "status": "active", "tradable": True,
        "marginable": True, "shortable": True, "easy_to_borrow": True,
    }).to_parquet(assets, index=False)
    pd.DataFrame({
        "date": ["2026-08-31", "2026-09-01"],
        "open": ["09:30", "09:30"], "close": ["16:00", "16:00"],
    }).to_parquet(calendar, index=False)
    outputs = {name: {"path": str(path), "sha256": _sha(path)} for name, path in {
        "account": account, "positions": positions, "assets": assets, "calendar": calendar,
    }.items()}
    (broker / "receipt.json").write_text(json.dumps({"status": "PASS", "outputs": outputs}))
    return scores_path, risk_path, covariance_path, broker


def test_current_portfolio_builds_executable_signal(tmp_path: Path) -> None:
    config = json.loads(
        (ROOT / "configs/research_v6_8/prospective_portfolio.json").read_text()
    )
    frozen = config["frozen_bindings"]
    required = []
    for name in ("historical_rows", "historical_factors"):
        path = Path(frozen[name]["path"])
        required.append(path if path.is_absolute() else ROOT / path)
    selector_root = ROOT / frozen["historical_selector_root"]
    required.extend(
        selector_root / target / "official_predictions.parquet"
        for target in ("ff5_momentum", "long_net", "short_net")
    )
    if any(not path.is_file() for path in required):
        pytest.skip("requires the authenticated frozen V6 historical artifacts")

    scores, risk, covariance, broker = _write_current_inputs(tmp_path)
    receipt = materialize_prospective_portfolio(
        config_path=ROOT / "configs/research_v6_8/prospective_portfolio.json",
        selector_scores_path=scores, risk_cost_path=risk,
        factor_covariance_path=covariance, broker_snapshot=broker,
        output=tmp_path / "portfolio",
        generated_at=datetime(2026, 8, 31, 21, 0, tzinfo=timezone.utc),
    )
    assert receipt["status"] == "PASS_EXECUTABLE"
    assert receipt["long_candidates"] >= 40
    assert receipt["short_candidates"] >= 40
    assert receipt["translation"]["paper_gross"] <= 1.8 + 1e-8
    signal = json.loads((tmp_path / "portfolio" / "signal.json").read_text())
    assert signal["execution_authorized"] is True
    assert signal["execution_session"] == "2026-09-01"
