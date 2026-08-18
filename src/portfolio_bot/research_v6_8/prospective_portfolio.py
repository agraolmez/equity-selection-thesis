"""Prospective monthly V6.8 target-book construction.

The implementation reuses the frozen V6.1 calibration and optimizer, applies
only the validation-selected 1.5% position-cap override, then translates the
relative book to the frozen 1.80x Alpaca paper gross cap.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from portfolio_bot.research_v6_1.portfolio import (
    FACTOR_NAMES,
    causal_monotonic_calibration,
    factor_covariance_matrix,
    percentile_rank,
    solve_expected_net_return_portfolio,
)


NEW_YORK = ZoneInfo("America/New_York")


class ProspectivePortfolioError(RuntimeError):
    """A frozen target-book input, schedule, or optimization check failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


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


def _verify_file(root: Path, binding: Mapping[str, Any]) -> Path:
    path = _resolve(root, str(binding["path"]))
    if not path.is_file() or _sha256(path) != str(binding["sha256"]):
        raise ProspectivePortfolioError(f"frozen input identity mismatch: {path}")
    return path


def _verified_output(path: Path, hash_field: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    receipt_path = path.parent / "receipt.json"
    if not path.is_file() or not receipt_path.is_file():
        raise ProspectivePortfolioError(f"current input or receipt is missing: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS" or receipt.get(hash_field) != _sha256(path):
        raise ProspectivePortfolioError(f"current input is not authenticated: {path}")
    return pd.read_parquet(path), receipt


def _historical_calibration_frame(
    *, root: Path, frozen: Mapping[str, Any], rows_path: Path, factors_path: Path,
    portfolio_config: Mapping[str, Any],
) -> pd.DataFrame:
    selector_root = _resolve(root, str(frozen["historical_selector_root"]))
    selected = str(frozen["selected_score_column"])
    selected_short = str(frozen["selected_short_score_column"])
    specifications = {
        "ff5_momentum": (selected, "alpha_ff5_momentum"),
        "long_net": (selected, "long_net_opportunity"),
        "short_net": (selected_short, "short_net_opportunity"),
    }
    pieces: list[pd.DataFrame] = []
    keys = ["holding_month", "signal_date", "permno", "permco"]
    source_bindings = {
        str(binding["id"]): binding for binding in portfolio_config["source_bindings"]
    }
    for name, (score, outcome) in specifications.items():
        path = selector_root / name / "official_predictions.parquet"
        binding = source_bindings[f"{name}_predictions"]
        if _sha256(path) != str(binding["sha256"]):
            raise ProspectivePortfolioError(
                f"frozen historical selector identity mismatch: {name}"
            )
        frame = pd.read_parquet(path, columns=[*keys, score, outcome]).rename(
            columns={score: f"score_{name}", outcome: f"actual_{name}"}
        )
        pieces.append(frame)
    history = pieces[0]
    for piece in pieces[1:]:
        history = history.merge(piece, on=keys, how="inner", validate="one_to_one")
    targets = pd.read_parquet(
        rows_path,
        columns=[*keys, "label_available_date", "unranked_excess_return"],
    )
    targets["holding_month"] = targets["holding_month"].astype(str)
    targets["signal_date"] = pd.to_datetime(targets["signal_date"], errors="raise")
    targets["label_available_date"] = pd.to_datetime(
        targets["label_available_date"], errors="coerce"
    )
    history["holding_month"] = history["holding_month"].astype(str)
    history["signal_date"] = pd.to_datetime(history["signal_date"], errors="raise")
    history = history.merge(targets, on=keys, how="inner", validate="one_to_one")
    factors = pd.read_parquet(factors_path, columns=["holding_month", "rf"])
    factors["holding_month"] = factors["holding_month"].astype(str)
    history = history.merge(factors, on="holding_month", how="left", validate="many_to_one")
    stock_excess = (
        pd.to_numeric(history["unranked_excess_return"], errors="coerce")
        - pd.to_numeric(history["rf"], errors="coerce")
    )
    history["payoff_long"] = stock_excess.where(
        pd.to_numeric(history["actual_long_net"], errors="coerce").notna()
    )
    history["payoff_short"] = (-stock_excess).where(
        pd.to_numeric(history["actual_short_net"], errors="coerce").notna()
    )
    for target in ("ff5_momentum", "long_net", "short_net"):
        history[f"rank_{target}"] = history.groupby(
            "holding_month", observed=True, sort=False
        )[f"score_{target}"].transform(percentile_rank)
    return history


def _calibrate(
    history: pd.DataFrame,
    current: pd.DataFrame,
    *,
    signal_date: pd.Timestamp,
    portfolio_config: Mapping[str, Any],
) -> pd.DataFrame:
    result = current.copy()
    specifications = {
        "ff5_momentum": ("rank_ff5_momentum", "actual_ff5_momentum"),
        "long": ("rank_long_net", "payoff_long"),
        "short": ("rank_short_net", "payoff_short"),
    }
    calibration = portfolio_config["calibration"]
    for name, (score, outcome) in specifications.items():
        fitted = causal_monotonic_calibration(
            history,
            result[score].to_numpy(float),
            score_column=score,
            outcome_column=outcome,
            month_column="holding_month",
            availability_column="label_available_date",
            signal_date=signal_date,
            bin_edges=calibration["bin_edges"],
            minimum_history_months=int(calibration["minimum_history_months"]),
            shrinkage_prior_months=float(calibration["shrinkage_prior_months"]),
        )
        result[f"expected_{name}"] = fitted.values
        result[f"calibration_months_{name}"] = fitted.history_months
    return result


def _calendar_schedule(
    calendar: pd.DataFrame, holding_month: str, signal_date: pd.Timestamp
) -> tuple[pd.Timestamp, datetime, bool]:
    frame = calendar.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.sort_values("date", kind="mergesort")
    holding = pd.Period(holding_month, freq="M")
    prior = holding - 1
    prior_rows = frame.loc[frame["date"].dt.to_period("M").eq(prior)]
    holding_rows = frame.loc[frame["date"].dt.to_period("M").eq(holding)]
    if prior_rows.empty or holding_rows.empty:
        raise ProspectivePortfolioError("broker calendar does not cover signal and entry months")
    expected_signal = pd.Timestamp(prior_rows["date"].max()).normalize()
    execution = pd.Timestamp(holding_rows["date"].min()).normalize()
    row = prior_rows.loc[prior_rows["date"].eq(expected_signal)].iloc[-1]
    close_text = str(row.get("close", "16:00"))
    hour, minute = (int(value) for value in close_text[:5].split(":"))
    cutoff = datetime.combine(expected_signal.date(), time(hour, minute), tzinfo=NEW_YORK)
    return execution, cutoff, pd.Timestamp(signal_date).normalize() == expected_signal


def _broker_frames(snapshot: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    receipt_path = snapshot / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise ProspectivePortfolioError("paper broker snapshot has not passed")
    outputs = receipt["outputs"]
    frames: dict[str, pd.DataFrame] = {}
    account = json.loads(Path(outputs["account"]["path"]).read_text(encoding="utf-8"))
    if _sha256(Path(outputs["account"]["path"])) != outputs["account"]["sha256"]:
        raise ProspectivePortfolioError("broker account snapshot identity changed")
    for name in ("positions", "assets", "calendar"):
        path = Path(outputs[name]["path"])
        if _sha256(path) != outputs[name]["sha256"]:
            raise ProspectivePortfolioError(f"broker {name} snapshot identity changed")
        frames[name] = pd.read_parquet(path)
    return account, frames["positions"], frames["assets"], frames["calendar"], receipt


def _previous_base_weights(
    frame: pd.DataFrame,
    positions: pd.DataFrame,
    account: Mapping[str, Any],
    prior_signal: Mapping[str, Any] | None,
) -> np.ndarray:
    if positions.empty:
        return np.zeros(len(frame), dtype=float)
    if prior_signal is None:
        raise ProspectivePortfolioError("a nonempty account requires the prior V6.8 signal state")
    if (
        prior_signal.get("schema") != "research_v6_8_paper_signal_v1"
        or prior_signal.get("execution_authorized") is not True
    ):
        raise ProspectivePortfolioError("prior signal state is not an authorized V6.8 signal")
    multiplier = float(prior_signal.get("translation", {}).get("effective_multiplier", 0.0))
    if multiplier <= 0.0:
        raise ProspectivePortfolioError("prior signal omits its effective paper multiplier")
    equity = float(account["equity"])
    weights: dict[str, float] = {}
    for row in positions.itertuples(index=False):
        symbol = str(row.symbol).upper()
        market_value = float(row.market_value)
        weights[symbol] = market_value / equity / multiplier
    prior_symbols = {
        str(row["symbol"]).upper() for row in prior_signal.get("targets", [])
    }
    unmanaged_prior = sorted(set(weights) - prior_symbols)
    if unmanaged_prior:
        raise ProspectivePortfolioError(
            "current account contains positions not owned by the prior V6.8 target: "
            + ", ".join(unmanaged_prior)
        )
    known = set(frame["symbol"].astype(str).str.upper())
    unmanaged = sorted(set(weights) - known)
    if unmanaged:
        raise ProspectivePortfolioError(
            "current account contains positions outside the current risk universe: "
            + ", ".join(unmanaged)
        )
    return frame["symbol"].astype(str).str.upper().map(weights).fillna(0.0).to_numpy(float)


def materialize_prospective_portfolio(
    *,
    config_path: Path,
    selector_scores_path: Path,
    risk_cost_path: Path,
    factor_covariance_path: Path,
    broker_snapshot: Path,
    output: Path,
    prior_signal_path: Path | None = None,
    technical_smoke: bool = False,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "research_v6_8_prospective_portfolio_v1":
        raise ProspectivePortfolioError("prospective portfolio config schema differs")
    frozen = config["frozen_bindings"]
    portfolio_path = _verify_file(project_root, frozen["portfolio_config"])
    _verify_file(project_root, frozen["translation_config"])
    rows_path = _verify_file(project_root, frozen["historical_rows"])
    factors_path = _verify_file(project_root, frozen["historical_factors"])
    portfolio_config = json.loads(portfolio_path.read_text(encoding="utf-8"))
    scores, score_receipt = _verified_output(selector_scores_path, "scores_sha256")
    risk, risk_receipt = _verified_output(risk_cost_path, "risk_cost_sha256")
    covariance, covariance_receipt = _verified_output(
        factor_covariance_path, "factor_covariance_sha256"
    )
    if risk_receipt != covariance_receipt:
        raise ProspectivePortfolioError("risk and factor covariance receipts differ")
    account, positions, assets, calendar, broker_receipt = _broker_frames(broker_snapshot)
    scores["holding_month"] = scores["holding_month"].astype(str)
    scores["signal_date"] = pd.to_datetime(scores["signal_date"], errors="raise")
    if scores["holding_month"].nunique() != 1 or scores["signal_date"].nunique() != 1:
        raise ProspectivePortfolioError("selector scores do not describe one decision month")
    holding_month = str(scores["holding_month"].iloc[0])
    signal_date = pd.Timestamp(scores["signal_date"].iloc[0]).normalize()
    execution, cutoff, official_schedule = _calendar_schedule(calendar, holding_month, signal_date)
    now = generated_at or datetime.now(timezone.utc)
    if not technical_smoke and (not official_schedule or now.astimezone(NEW_YORK) < cutoff):
        raise ProspectivePortfolioError("current package is not an executable month-end close signal")

    score_columns = [
        "live_security_id", "symbol", "permno", "holding_month", "signal_date",
        "ff5_momentum_score", "long_net_score", "short_net_score",
    ]
    current_scores = scores[score_columns].copy()
    for name in ("ff5_momentum", "long_net", "short_net"):
        current_scores[f"rank_{name}"] = percentile_rank(
            current_scores[f"{name}_score"].to_numpy(float)
        )
    history = _historical_calibration_frame(
        root=project_root, frozen=frozen, rows_path=rows_path,
        factors_path=factors_path, portfolio_config=portfolio_config,
    )
    current_scores = _calibrate(
        history, current_scores, signal_date=signal_date, portfolio_config=portfolio_config
    )
    frame = risk.merge(
        current_scores.drop(columns=["holding_month", "signal_date", "permno", "symbol"]),
        on="live_security_id", how="left", validate="one_to_one",
    )
    frame["permno"] = frame["live_security_id"].map(
        scores.set_index("live_security_id")["permno"]
    )
    frame["symbol"] = frame["symbol"].astype("string").str.upper()
    frame = frame.loc[frame["symbol"].notna()].copy()
    asset = assets.copy()
    asset["symbol"] = asset["symbol"].astype(str).str.upper()
    asset = asset.drop_duplicates("symbol").set_index("symbol")
    for column in ("tradable", "marginable", "shortable", "easy_to_borrow"):
        frame[f"broker_{column}"] = frame["symbol"].map(asset[column]).fillna(False).astype(bool)
    frame["broker_active"] = frame["symbol"].map(asset["status"]).fillna("").astype(str).str.lower().eq("active")
    current_symbols = set(positions["symbol"].astype(str).str.upper()) if not positions.empty else set()
    blocked_existing = sorted(
        symbol for symbol in current_symbols
        if symbol not in asset.index or not bool(asset.loc[symbol, "tradable"])
    )
    if blocked_existing:
        raise ProspectivePortfolioError("existing position is not currently tradable: " + ", ".join(blocked_existing))
    prior_signal: Mapping[str, Any] | None = None
    if prior_signal_path is not None:
        prior_receipt_path = prior_signal_path.parent / "receipt.json"
        if not prior_receipt_path.is_file():
            raise ProspectivePortfolioError("prior signal producer receipt is missing")
        prior_receipt = json.loads(prior_receipt_path.read_text(encoding="utf-8"))
        if prior_receipt.get("signal_sha256") != _sha256(prior_signal_path):
            raise ProspectivePortfolioError("prior signal differs from its producer receipt")
        prior_signal = json.loads(prior_signal_path.read_text(encoding="utf-8"))
    previous = _previous_base_weights(frame, positions, account, prior_signal)
    risk_ready = (
        frame["risk_status"].astype(str).eq("PRESENT").to_numpy()
        & frame["cost_input_status"].astype(str).eq("PRESENT").to_numpy()
    )
    # SIC2 is not merely a model category here: it is the grouping key for a
    # binding portfolio constraint.  Missing model categories are legitimate
    # encoder inputs, but opening a position with an unknown constraint group
    # would make the live optimizer economically different from the frozen
    # historical policy.  Existing positions remain reduction-only.
    known_industry = (
        frame["sic2_industry_category"]
        .fillna("UNKNOWN_INDUSTRY")
        .astype(str)
        .str.strip()
        .ne("")
        .to_numpy()
        & ~frame["sic2_industry_category"]
        .fillna("UNKNOWN_INDUSTRY")
        .astype(str)
        .isin({"UNKNOWN_INDUSTRY", "UNKNOWN_SIC2", "<NA>"})
        .to_numpy()
    )
    threshold = 1.0 - float(config["formation"]["candidate_fraction_each_side"])
    long_candidate = (
        frame["long_eligible"].fillna(False).to_numpy(bool)
        & frame["new_long_trade_allowed"].fillna(False).to_numpy(bool)
        & frame["rank_long_net"].fillna(-np.inf).to_numpy(float).__ge__(threshold)
        & frame["expected_ff5_momentum"].fillna(-np.inf).to_numpy(float).__gt__(0.0)
        & frame["broker_active"].to_numpy(bool)
        & frame["broker_tradable"].to_numpy(bool)
        & frame["broker_marginable"].to_numpy(bool)
        & known_industry
        & risk_ready
    )
    short_candidate = (
        frame["short_eligible"].fillna(False).to_numpy(bool)
        & frame["new_short_trade_allowed"].fillna(False).to_numpy(bool)
        & frame["rank_short_net"].fillna(-np.inf).to_numpy(float).__ge__(threshold)
        & frame["expected_ff5_momentum"].fillna(np.inf).to_numpy(float).__lt__(0.0)
        & frame["broker_active"].to_numpy(bool)
        & frame["broker_tradable"].to_numpy(bool)
        & frame["broker_marginable"].to_numpy(bool)
        & frame["broker_shortable"].to_numpy(bool)
        & frame["broker_easy_to_borrow"].to_numpy(bool)
        & known_industry
        & risk_ready
    )
    overlap = long_candidate & short_candidate
    prefer_long = (
        frame["expected_long"].fillna(-np.inf).to_numpy(float)
        >= frame["expected_short"].fillna(-np.inf).to_numpy(float)
    )
    short_candidate[overlap & prefer_long] = False
    long_candidate[overlap & ~prefer_long] = False
    reduction_only = (np.abs(previous) > 1e-12) & ~(long_candidate | short_candidate)
    if np.any(reduction_only & ~risk_ready):
        symbols = frame.loc[reduction_only & ~risk_ready, "symbol"].astype(str).tolist()
        raise ProspectivePortfolioError("inherited position lacks current risk/cost inputs: " + ", ".join(symbols))
    active = long_candidate | short_candidate | reduction_only
    frame = frame.loc[active].reset_index(drop=True)
    long_candidate = long_candidate[active]
    short_candidate = short_candidate[active]
    reduction_only = reduction_only[active]
    previous = previous[active]
    if int(long_candidate.sum()) < 40 or int(short_candidate.sum()) < 40:
        raise ProspectivePortfolioError("broker-feasible gated candidate breadth is below 40 per side")
    mu = np.zeros(len(frame), dtype=float)
    mu[long_candidate] = frame.loc[long_candidate, "expected_long"].to_numpy(float)
    mu[short_candidate] = -frame.loc[short_candidate, "expected_short"].to_numpy(float)
    optimizer = dict(portfolio_config["optimizer"])
    optimizer.update(config["portfolio_overrides"])
    costs = portfolio_config["costs"]["primary"]
    solution = solve_expected_net_return_portfolio(
        expected_stock_excess_return=mu,
        long_candidate=long_candidate,
        short_candidate=short_candidate,
        reduction_only=reduction_only,
        previous_weights=previous,
        factor_betas=frame[[f"beta_risk_{name}" for name in FACTOR_NAMES]].to_numpy(float),
        factor_covariance_root=factor_covariance_matrix(covariance),
        specific_variance_monthly=frame["specific_variance_monthly"].to_numpy(float),
        full_spread=frame["full_spread_20d"].to_numpy(float),
        adv20_usd=frame["adv20_usd"].to_numpy(float),
        impact_volatility_daily=frame["impact_volatility_daily"].to_numpy(float),
        borrow_rate_annual=frame[str(costs["borrow_column"])].to_numpy(float),
        sic2_industry=frame["sic2_industry_category"].astype(str).to_numpy(),
        nav_dollars=float(account["equity"]),
        forecast_volatility_target=float(optimizer["primary_forecast_volatility"]),
        maximum_gross=float(optimizer["maximum_gross"]),
        minimum_long_gross=float(optimizer["minimum_long_gross"]),
        minimum_short_gross=float(optimizer["minimum_short_gross"]),
        minimum_net=float(optimizer["minimum_net"]),
        maximum_net=float(optimizer["maximum_net"]),
        maximum_position_weight=float(optimizer["maximum_position_weight"]),
        maximum_one_way_turnover=float(optimizer["maximum_one_way_turnover"]),
        maximum_one_way_adv_participation=float(optimizer["maximum_one_way_adv_participation"]),
        market_beta_absolute_limit=float(optimizer["market_beta_absolute_limit"]),
        other_factor_absolute_limit=float(optimizer["other_factor_absolute_limit"]),
        sic2_industry_net_absolute_limit=float(optimizer["sic2_industry_net_absolute_limit"]),
        sic2_industry_gross_limit=float(optimizer["sic2_industry_gross_limit"]),
        spread_multiplier=float(costs["spread_multiplier"]),
        impact_coefficient=float(costs["impact_coefficient"]),
        financing_rate_annual=float(costs["financing_rate_annual"]),
        solver=str(optimizer["solver"]),
    )
    raw = solution.weights * float(config["paper_translation"]["raw_multiplier"])
    raw_gross = float(np.abs(raw).sum())
    scale = min(1.0, float(config["paper_translation"]["maximum_starting_gross"]) / max(raw_gross, 1e-12))
    translated = raw * scale
    effective_multiplier = float(config["paper_translation"]["raw_multiplier"]) * scale
    if float(np.abs(translated).max(initial=0.0)) > float(
        config["paper_translation"]["maximum_absolute_position_weight"]
    ):
        raise ProspectivePortfolioError(
            "translated target exceeds the frozen month-start position limit"
        )
    book_columns = [
        "live_security_id", "symbol", "permno", "sic2_industry_category",
        "rank_ff5_momentum", "rank_long_net", "rank_short_net",
        "expected_ff5_momentum", "expected_long", "expected_short",
        *[f"beta_risk_{name}" for name in FACTOR_NAMES],
        "specific_variance_monthly", "full_spread_20d", "adv20_usd",
        "impact_volatility_daily", "primary_borrow_rate_annual",
    ]
    book = frame[book_columns].copy()
    book["monthly_expected_stock_excess_return"] = mu
    book["base_weight"] = solution.weights
    book["target_weight"] = translated
    book["side"] = np.where(translated > 0, "long", np.where(translated < 0, "short", "cash"))
    book = book.loc[book["target_weight"].abs().gt(1e-10)].sort_values("symbol", kind="mergesort")
    book_path = output / "target_book.parquet"
    _atomic_parquet(book_path, book)
    manifest = {
        "config_sha256": _sha256(config_path),
        "selector_receipt_sha256": _sha256(selector_scores_path.parent / "receipt.json"),
        "risk_receipt_sha256": _sha256(risk_cost_path.parent / "receipt.json"),
        "broker_receipt_sha256": _sha256(broker_snapshot / "receipt.json"),
        "historical_rows_sha256": _sha256(rows_path),
        "historical_factors_sha256": _sha256(factors_path),
        "target_book_sha256": _sha256(book_path),
    }
    manifest_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    executable = bool(official_schedule and now.astimezone(NEW_YORK) >= cutoff and not technical_smoke)
    signal = {
        "schema": "research_v6_8_paper_signal_v1" if executable else "research_v6_8_paper_signal_smoke_v1",
        "strategy_id": config["strategy_id"],
        "model_result_sha256": frozen["v6_8_result_sha256"],
        "policy_action": "monthly_rebalance",
        "input_manifest_sha256": manifest_digest,
        "source_max_signal_date": signal_date.date().isoformat(),
        "signal_id": f"{holding_month}-monthly-{manifest_digest[:12]}",
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "signal_cutoff_at": cutoff.astimezone(timezone.utc).isoformat(),
        "execution_session": execution.date().isoformat(),
        "execution_authorized": executable,
        "targets": [
            {"symbol": row.symbol, "target_weight": float(row.target_weight), "base_weight": float(row.base_weight)}
            for row in book.itertuples(index=False)
        ],
        "translation": {
            "raw_multiplier": float(config["paper_translation"]["raw_multiplier"]),
            "gross_cap_scalar": scale,
            "effective_multiplier": effective_multiplier,
            "base_gross": solution.gross,
            "paper_gross": float(np.abs(translated).sum()),
            "paper_net": float(translated.sum()),
        },
    }
    signal_path = output / ("signal.json" if executable else "technical_smoke_signal.json")
    _atomic_json(signal_path, signal)
    receipt = {
        "schema": "research_v6_8_prospective_portfolio_receipt_v1",
        "status": "PASS_EXECUTABLE" if executable else "PASS_TECHNICAL_SMOKE_NON_EXECUTABLE",
        "holding_month": holding_month,
        "signal_date": signal_date.date().isoformat(),
        "execution_session": execution.date().isoformat(),
        "official_month_end_schedule": official_schedule,
        "long_candidates": int(long_candidate.sum()),
        "short_candidates": int(short_candidate.sum()),
        "positions": int(len(book)),
        "base_solution": {
            "status": solution.status, "gross": solution.gross, "net": solution.net,
            "forecast_volatility": solution.forecast_volatility, "turnover": solution.turnover,
            "expected_net_objective": solution.objective_value,
        },
        "translation": signal["translation"],
        "manifest": manifest,
        "signal_path": str(signal_path),
        "signal_sha256": _sha256(signal_path),
        "target_book_sha256": _sha256(book_path),
        "broker_snapshot_status": broker_receipt["status"],
        "selector_forecast_month": score_receipt["forecast_month"],
    }
    _atomic_json(output / "receipt.json", receipt)
    return receipt


__all__ = ["ProspectivePortfolioError", "materialize_prospective_portfolio"]
