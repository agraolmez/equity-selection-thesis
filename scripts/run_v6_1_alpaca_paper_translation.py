#!/usr/bin/env python3
"""Replay the frozen V6.1 portfolio under an Alpaca paper-account translation.

The study does not refit a selector, rerun an optimizer, or change any stock's
relative weight.  It applies a common monthly scalar to the authenticated
V6.1 position_1p5 holdings, recomputes stateful trading/borrow/financing costs
at a $100,000 starting NAV, and reports both the frozen financing contract and
a broker-current financing sensitivity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_v6_1_return_scaling_overlay as replay  # noqa: E402
from scripts import run_v6_1_maximum_return_scaling_frontier as frontier  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/research_v6_1/alpaca_paper_translation.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_1/alpaca_paper_translation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _apply_uniform_gross_cap(
    raw_weights: np.ndarray, target_gross: float | None
) -> tuple[np.ndarray, float]:
    weights = np.asarray(raw_weights, dtype=float)
    gross = float(np.abs(weights).sum())
    if target_gross is None or gross <= float(target_gross):
        return weights.copy(), 1.0
    if target_gross <= 0.0 or gross <= 0.0:
        raise ValueError("gross target and raw gross must be positive")
    scalar = float(target_gross) / gross
    return weights * scalar, scalar


def _max_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return float(np.min(wealth / peak - 1.0, initial=0.0))


def _verify(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], Path]:
    config = json.loads(config_path.read_text())
    if config.get("status") != "AUTHORIZED_POST_DIAGNOSTIC_DEPLOYMENT_TRANSLATION":
        raise RuntimeError("Alpaca deployment translation is not authorized")
    source = config["sources"]["maximum_return_frontier_config"]
    source_path = ROOT / str(source["path"])
    if _sha256(source_path) != str(source["sha256"]):
        raise RuntimeError("maximum-return frontier config identity changed")
    frontier_config, frontier_identity, source_root = frontier._verify(source_path)
    code_digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), Path(replay.__file__).resolve(), Path(frontier.__file__).resolve()):
        code_digest.update(str(path).encode())
        code_digest.update(bytes.fromhex(_sha256(path)))
    identity = {
        "config_sha256": _sha256(config_path),
        "frontier_config_sha256": str(source["sha256"]),
        "source_manifest_sha256": frontier_identity["source_manifest_sha256"],
        "risk_cost_sha256": frontier_identity["risk_cost_sha256"],
        "benchmarks_sha256": frontier_identity["benchmarks_sha256"],
        "code_identity": code_digest.hexdigest(),
    }
    return config, identity, frontier_config, source_root


def _summary(policy: str, cost_case: str, frame: pd.DataFrame) -> dict[str, Any]:
    total = frame["net_return"].to_numpy(float)
    excess = frame["net_excess_return"].to_numpy(float)
    wealth = np.cumprod(1.0 + total)
    volatility = float(excess.std(ddof=1) * math.sqrt(12.0))
    months = len(frame)
    return {
        "policy": policy,
        "cost_case": cost_case,
        "months": int(months),
        "starting_nav": float(frame["nav_start"].iloc[0]),
        "ending_nav": float(frame["nav_end"].iloc[-1]),
        "cumulative_total_return": float(wealth[-1] - 1.0),
        "total_return_cagr": float(wealth[-1] ** (12.0 / months) - 1.0),
        "annualized_net_excess_return": float(excess.mean() * 12.0),
        "annualized_realized_volatility": volatility,
        "net_sharpe": float(excess.mean() / excess.std(ddof=1) * math.sqrt(12.0)),
        "maximum_drawdown": _max_drawdown(total),
        "positive_month_fraction": float((excess > 0.0).mean()),
        "annualized_cost": float(frame["total_cost"].mean() * 12.0),
        "annualized_transaction_cost": float(frame["transaction_cost"].mean() * 12.0),
        "annualized_borrow_cost": float(frame["borrow_cost"].mean() * 12.0),
        "annualized_financing_cost": float(frame["financing_cost"].mean() * 12.0),
        "mean_rebalance_gross": float(frame["rebalance_gross"].mean()),
        "maximum_rebalance_gross": float(frame["rebalance_gross"].max()),
        "mean_rebalance_net": float(frame["rebalance_net"].mean()),
        "maximum_absolute_rebalance_net": float(frame["rebalance_net"].abs().max()),
        "mean_effective_multiplier": float(frame["effective_multiplier"].mean()),
        "minimum_effective_multiplier": float(frame["effective_multiplier"].min()),
        "maximum_effective_multiplier": float(frame["effective_multiplier"].max()),
        "maximum_end_drift_gross": float(frame["end_drift_gross"].max()),
        "month_end_safety_breaches": int(frame["month_end_safety_breach"].sum()),
        "maximum_safety_overage": float(frame["safety_overage"].max()),
        "maximum_dollar_safety_overage": float(frame["dollar_safety_overage"].max()),
        "maximum_one_way_adv_participation": float(frame["maximum_adv_participation"].max()),
        "maximum_absolute_position": float(frame["maximum_absolute_position"].max()),
        "maximum_absolute_factor_exposure": float(frame["maximum_absolute_factor_exposure"].max()),
        "maximum_sic2_net": float(frame["maximum_sic2_net"].max()),
        "maximum_sic2_gross": float(frame["maximum_sic2_gross"].max()),
        "best_month": float(frame["net_return"].max()),
        "worst_month": float(frame["net_return"].min()),
    }


def _run_phase(
    phase: str,
    policy: dict[str, Any],
    cost_case: str,
    config: dict[str, Any],
    replay_config: dict[str, Any],
    risk: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = replay_config["sources"][phase]
    month_root = ROOT / str(source["month_root"])
    months = sorted(path.stem for path in month_root.glob("*.parquet"))
    expected = pd.period_range(source["start"], source["end"], freq="M").astype(str).tolist()
    if months != expected:
        raise RuntimeError(f"{phase} month sequence is incomplete")
    nav = float(config["account"]["starting_nav_dollars"])
    safety = float(config["account"]["operational_safety_gross"])
    previous: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    specification = dict(config["cost_cases"][cost_case])
    for month in months:
        receipt = json.loads((month_root / f"{month}.json").read_text())["summary"]
        positions = replay._read_parquet(month_root / f"{month}.parquet")
        month_risk = risk.loc[risk["holding_month"].eq(month)].drop(columns="holding_month")
        positions = positions.merge(month_risk, on="permno", how="left", validate="one_to_one")
        active = positions["weight"].abs().gt(1e-14) | positions["permno"].isin(previous)
        positions = positions.loc[active].copy()
        raw_multiplier = float(policy["raw_multiplier"])
        raw_weights = raw_multiplier * positions["weight"].to_numpy(float)
        weights, cap_scalar = _apply_uniform_gross_cap(
            raw_weights, policy["rebalance_gross_target"]
        )
        previous_weights = positions["permno"].map(previous).fillna(0.0).to_numpy(float)
        excess = positions["stock_excess_return"].to_numpy(float)
        transaction, borrow, financing, total_cost = replay._cost(
            weights,
            previous_weights,
            positions,
            nav,
            specification,
            float(config["commission_rate"]),
        )
        gross_excess = float(weights @ excess)
        net_excess = gross_excess - total_cost
        rf = float(receipt["rf"])
        net_return = rf + net_excess
        if 1.0 + net_return <= 0.0:
            raise RuntimeError(f"NAV became nonpositive in {phase} {month}")
        drift = weights * (1.0 + excess + rf) / (1.0 + net_return)
        end_drift_gross = float(np.abs(drift).sum())
        safety_overage = max(end_drift_gross - safety, 0.0)
        previous = {
            int(permno): float(weight)
            for permno, weight in zip(positions["permno"], drift)
            if abs(float(weight)) > 1e-12
        }
        trade = np.abs(weights - previous_weights)
        grouped = pd.DataFrame(
            {"industry": positions["sic2_industry_category"].astype(str), "weight": weights}
        ).groupby("industry", observed=True, sort=False)["weight"]
        sic_net = float(grouped.sum().abs().max()) if len(grouped) else 0.0
        sic_gross = float(grouped.apply(lambda values: values.abs().sum()).max()) if len(grouped) else 0.0
        effective_multiplier = raw_multiplier * cap_scalar
        factor_max = max(
            abs(effective_multiplier * float(receipt[f"factor_exposure_{name}"]))
            for name in replay.FACTOR_NAMES
        )
        row = {
            "phase": phase,
            "policy": str(policy["id"]),
            "cost_case": cost_case,
            "holding_month": month,
            "nav_start": nav,
            "nav_end": nav * (1.0 + net_return),
            "rf": rf,
            "gross_excess_return": gross_excess,
            "net_excess_return": net_excess,
            "net_return": net_return,
            "transaction_cost": transaction,
            "borrow_cost": borrow,
            "financing_cost": financing,
            "total_cost": total_cost,
            "raw_gross": float(np.abs(raw_weights).sum()),
            "rebalance_gross": float(np.abs(weights).sum()),
            "rebalance_net": float(weights.sum()),
            "cap_scalar": cap_scalar,
            "effective_multiplier": effective_multiplier,
            "end_drift_gross": end_drift_gross,
            "month_end_safety_breach": bool(safety_overage > 1e-12),
            "safety_overage": safety_overage,
            "dollar_safety_overage": nav * safety_overage,
            "one_way_turnover": float(0.5 * trade.sum()),
            "maximum_adv_participation": float(
                np.max(trade * nav / positions["adv20_usd"].to_numpy(float), initial=0.0)
            ),
            "maximum_absolute_position": float(np.max(np.abs(weights), initial=0.0)),
            "maximum_absolute_factor_exposure": factor_max,
            "maximum_sic2_net": sic_net,
            "maximum_sic2_gross": sic_gross,
        }
        rows.append(row)
        nav = row["nav_end"]
    frame = pd.DataFrame(rows)
    return frame, _summary(str(policy["id"]), cost_case, frame)


def _markdown(result: dict[str, Any]) -> str:
    phase = result["historical_test"]
    frozen = phase["alpaca_target_1p80"]["frozen_primary"]
    broker = phase["alpaca_target_1p80"]["broker_current"]
    benchmarks = result["benchmarks"]["historical_test"]
    return f"""# Alpaca paper-account deployment translation

The authenticated V6.1 1.5% position-cap holdings are replayed from a
$100,000 starting NAV. Security selection, model ranks, alpha gate, and every
relative monthly stock weight remain frozen. The deployment policy applies a
common scalar so rebalance gross does not exceed 1.80x.

| Historical test view | CAGR | Sharpe | Volatility | Max drawdown | Ending $100k |
|---|---:|---:|---:|---:|---:|
| 1.80x, frozen financing | {frozen['total_return_cagr']:.2%} | {frozen['net_sharpe']:.2f} | {frozen['annualized_realized_volatility']:.2%} | {frozen['maximum_drawdown']:.2%} | ${frozen['ending_nav']:,.0f} |
| 1.80x, broker-current financing | {broker['total_return_cagr']:.2%} | {broker['net_sharpe']:.2f} | {broker['annualized_realized_volatility']:.2%} | {broker['maximum_drawdown']:.2%} | ${broker['ending_nav']:,.0f} |
| SPY | {benchmarks['spy_total_return']['cagr']:.2%} | - | {benchmarks['spy_total_return']['volatility']:.2%} | {benchmarks['spy_total_return']['maximum_drawdown']:.2%} | ${100000*(1+benchmarks['spy_total_return']['cumulative_return']):,.0f} |
| QQQ | {benchmarks['qqq_total_return']['cagr']:.2%} | - | {benchmarks['qqq_total_return']['volatility']:.2%} | {benchmarks['qqq_total_return']['maximum_drawdown']:.2%} | ${100000*(1+benchmarks['qqq_total_return']['cumulative_return']):,.0f} |

The frozen-cost translation is the historical research comparison. The
broker-current view is a financing sensitivity, not a statement that Alpaca
paper accounts reproduce every live fee. Alpaca states that paper trading does
not simulate market impact, latency slippage, order queue position, regulatory
fees, dividends, or borrow fees.

The 1.80x rebalance target is respected in every month. Month-end market drift
crosses the 1.95x operating threshold in
{int(frozen['month_end_safety_breaches'])} historical-test month(s), with a
maximum end-drift gross of {frozen['maximum_end_drift_gross']:.4f}x. Exact
intramonth enforcement cannot be reconstructed from monthly returns; the live
paper trader therefore requires daily exposure monitoring and proportional
deleveraging before the broker's 2x overnight boundary.
"""


def run(config_path: Path, output_root: Path) -> Path:
    config, identity, frontier_config, source_root = _verify(config_path)
    output = output_root / (
        f"{identity['config_sha256'][:12]}-"
        f"{identity['source_manifest_sha256'][:12]}-"
        f"{identity['code_identity'][:12]}"
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run_manifest.json"
    _atomic_json(
        manifest,
        {
            "schema": "research_v6_1_alpaca_paper_translation_run_v1",
            "status": "RUNNING",
            "identity": identity,
        },
    )
    replay_config = frontier._replay_config(frontier_config, source_root)
    risk = replay._load_risk(replay_config)
    result: dict[str, Any] = {
        "schema": "research_v6_1_alpaca_paper_translation_result_v1",
        "status": "COMPLETE",
        "identity": identity,
        "account": config["account"],
        "validation": {},
        "historical_test": {},
        "benchmarks": {},
        "interpretation": config["interpretation"],
    }
    summary_rows: list[dict[str, Any]] = []
    for phase in ("validation", "historical_test"):
        result[phase] = {}
        for policy in config["policies"]:
            policy_id = str(policy["id"])
            result[phase][policy_id] = {}
            for cost_case in config["cost_cases"]:
                monthly, summary = _run_phase(
                    phase, policy, str(cost_case), config, replay_config, risk
                )
                _atomic_csv(output / phase / policy_id / f"{cost_case}_monthly.csv", monthly)
                result[phase][policy_id][str(cost_case)] = summary
                summary_rows.append({"phase": phase, **summary})
        benchmark_frame = frontier._benchmark_frame(
            frontier_config,
            replay_config["sources"][phase]["start"],
            replay_config["sources"][phase]["end"],
        )
        result["benchmarks"][phase] = frontier._benchmark_summary(benchmark_frame)
    _atomic_csv(output / "summary.csv", pd.DataFrame(summary_rows))
    _atomic_json(output / "result.json", result)
    _atomic_text(output / "RESULT.md", _markdown(result))
    _atomic_json(
        manifest,
        {
            "schema": "research_v6_1_alpaca_paper_translation_run_v1",
            "status": "COMPLETE",
            "identity": identity,
            "result": "result.json",
            "summary": "summary.csv",
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args.config.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
