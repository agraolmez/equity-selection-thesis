#!/usr/bin/env python3
"""Run the checkpointed V6.1 Step 04 short-sleeve/hedge comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_v6_1_portfolio_step_02 as base  # noqa: E402
from portfolio_bot.allocation.costs import numpy_total_cost  # noqa: E402
from portfolio_bot.research_v6_1.portfolio import FACTOR_NAMES  # noqa: E402
from portfolio_bot.research_v6_1.portfolio_step_03 import stressed_round_trip_hurdle  # noqa: E402
from portfolio_bot.research_v6_1.portfolio_step_04 import (  # noqa: E402
    Step04Solution,
    solve_optional_short_portfolio,
)


DEFAULT_CONFIG = ROOT / "configs/research_v6_1/portfolio_step_04.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_1/portfolio_step_04"


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


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _code_identity() -> str:
    files = [
        Path(__file__).resolve(),
        ROOT / "scripts/run_v6_1_portfolio_step_02.py",
        ROOT / "src/portfolio_bot/research_v6_1/portfolio.py",
        ROOT / "src/portfolio_bot/research_v6_1/portfolio_step_03.py",
        ROOT / "src/portfolio_bot/research_v6_1/portfolio_step_04.py",
        ROOT / "src/portfolio_bot/allocation/costs.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def verify_contract(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str, dict[str, Path]]:
    config = json.loads(config_path.read_text())
    if config.get("status") != "FROZEN_READY_TO_RUN":
        raise RuntimeError("V6.1 Step 04 contract is not frozen and ready")
    base_path = ROOT / str(config["base_contract"]["path"])
    if _sha256(base_path) != str(config["base_contract"]["sha256"]):
        raise RuntimeError("bound V6.1 Step 02 contract identity changed")
    base_config, _, input_identity, paths = base.verify_contract(base_path)
    result_root = ROOT / str(config["base_result"]["path"])
    manifest_path = result_root / "run_manifest.json"
    if _sha256(manifest_path) != str(config["base_result"]["manifest_sha256"]):
        raise RuntimeError("bound V6.1 Step 02 result identity changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != config["base_result"]["required_status"]:
        raise RuntimeError("bound V6.1 Step 02 result is not complete")
    if manifest.get("config_sha256") != config["base_contract"]["sha256"]:
        raise RuntimeError("bound V6.1 Step 02 result used another contract")
    return config, base_config, _sha256(config_path), input_identity, paths


def _expanded_loader_config(base_config: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base_config)
    result["evidence"]["portfolio_start"] = config["evidence"]["calibration_start"]
    result["evidence"]["portfolio_end"] = config["evidence"]["portfolio_end"]
    return result


def _spy_history(inputs: base.PreparedInputs) -> pd.DataFrame:
    spy = inputs.benchmarks.loc[
        inputs.benchmarks["benchmark_id"].astype(str).eq("spy_total_return"),
        ["holding_month", "total_return"],
    ].rename(columns={"total_return": "spy_total_return"})
    factors = inputs.factors[["holding_month", "mktrf", "rf"]].copy()
    history = spy.merge(factors, on="holding_month", how="inner", validate="one_to_one")
    for column in ("spy_total_return", "mktrf", "rf"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history["spy_excess_return"] = history["spy_total_return"] - history["rf"]
    history["spy_residual"] = history["spy_excess_return"] - history["mktrf"]
    return history.sort_values("holding_month", kind="mergesort").reset_index(drop=True)


def _spy_inputs(
    history: pd.DataFrame, month: str, config: dict[str, Any]
) -> tuple[float, float, float]:
    prior = history.loc[history["holding_month"].lt(month)].dropna(
        subset=["spy_excess_return", "spy_residual"]
    )
    if len(prior) < 24:
        raise RuntimeError(f"SPY causal history has {len(prior)} months before {month}; requires 24")
    shrink = len(prior) / (len(prior) + 24.0)
    expected = max(float(prior["spy_excess_return"].mean()) * shrink, 0.0)
    observed_variance = float(prior["spy_residual"].var(ddof=1))
    residual_variance = max(
        observed_variance * shrink + 0.0001 * (1.0 - shrink), 1e-8
    )
    current = history.loc[history["holding_month"].eq(month), "spy_excess_return"]
    if len(current) != 1 or not np.isfinite(float(current.iloc[0])):
        raise RuntimeError(f"SPY realized excess return is unavailable for {month}")
    return expected, residual_variance, float(current.iloc[0])


def _candidate_masks(
    frame: pd.DataFrame,
    base_config: dict[str, Any],
    config: dict[str, Any],
    policy: dict[str, Any],
    previous: np.ndarray,
    nav_dollars: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    long_mask, raw_short = base._policy_masks(
        frame, base_config, str(config["formation"]["policy"])
    )
    stress = base_config["costs"]["standard_stress"]
    hurdle_cfg = config["cost_hurdle"]
    short_hurdle = stressed_round_trip_hurdle(
        full_spread=frame["full_spread_20d"].to_numpy(float),
        adv20_usd=frame["adv20_usd"].to_numpy(float),
        impact_volatility_daily=frame["impact_volatility_daily"].to_numpy(float),
        stress_borrow_rate_annual=frame["stress_borrow_rate_annual"].to_numpy(float),
        nav_dollars=nav_dollars,
        representative_position_weight=float(hurdle_cfg["representative_position_weight"]),
        spread_multiplier=float(stress["spread_multiplier"]),
        impact_coefficient=float(stress["impact_coefficient"]),
        commission_each_way=float(hurdle_cfg["commission_each_way"]),
        include_borrow=True,
    )
    alpha_short = raw_short & (
        frame["expected_short"].to_numpy(float) > short_hurdle
    )
    hedge_short = np.zeros(len(frame), dtype=bool)
    if not bool(policy["spy_allowed"]):
        pool = config["formation"]["stock_hedge_pool"]
        adv = frame["adv20_usd"].to_numpy(float)
        borrow = frame["stress_borrow_rate_annual"].to_numpy(float)
        adv_cut = float(np.nanquantile(adv, float(pool["minimum_adv_cross_section_percentile"])))
        borrow_cut = float(
            np.nanquantile(borrow, float(pool["maximum_stress_borrow_cross_section_percentile"]))
        )
        hedge_short = (
            frame["short_eligible"].fillna(False).to_numpy(bool)
            & frame["new_short_trade_allowed"].fillna(False).to_numpy(bool)
            & (frame["rank_short_net"].to_numpy(float) >= float(pool["minimum_short_rank"]))
            & (adv >= adv_cut)
            & (borrow <= borrow_cut)
            & ~long_mask
            & ~alpha_short
        )
    long_mask[previous < 0.0] = False
    alpha_short[previous > 0.0] = False
    hedge_short[previous > 0.0] = False
    short_mask = alpha_short | hedge_short
    reduction_only = (previous != 0.0) & ~(long_mask | short_mask)
    return long_mask, alpha_short, hedge_short, short_mask, reduction_only


def _stock_costs(
    weights: np.ndarray,
    previous: np.ndarray,
    frame: pd.DataFrame,
    base_config: dict[str, Any],
    nav_dollars: float,
    scenario: str,
) -> Any:
    cost = base_config["costs"][scenario]
    return numpy_total_cost(
        weights,
        previous,
        frame["full_spread_20d"].to_numpy(float),
        frame["adv20_usd"].to_numpy(float),
        frame["impact_volatility_daily"].to_numpy(float),
        frame[str(cost["borrow_column"])].to_numpy(float),
        nav_dollars,
        impact_coefficient=float(cost["impact_coefficient"]),
        spread_multiplier=float(cost["spread_multiplier"]),
        financing_rate_annual=0.0,
    )


def _solve_month(
    frame: pd.DataFrame,
    covariance_root: np.ndarray,
    spy_history: pd.DataFrame,
    config: dict[str, Any],
    base_config: dict[str, Any],
    policy: dict[str, Any],
    risk_aversion: float,
    previous_map: dict[int, float],
    previous_spy: float,
    nav_dollars: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    previous = base._aligned_previous(frame, previous_map)
    long_mask, alpha_short, hedge_short, short_mask, reduction_only = _candidate_masks(
        frame, base_config, config, policy, previous, nav_dollars
    )
    mu = np.zeros(len(frame), dtype=np.float64)
    mu[long_mask] = frame.loc[long_mask, "expected_long"].to_numpy(float)
    mu[short_mask] = -frame.loc[short_mask, "expected_short"].to_numpy(float)
    month = str(frame["holding_month"].iloc[0])
    spy_mu, spy_specific_variance, spy_realized_excess = _spy_inputs(
        spy_history, month, config
    )
    optimizer = config["optimizer"]
    primary = base_config["costs"]["primary"]
    spy_primary = config["spy_hedge"]["primary"]
    solution = solve_optional_short_portfolio(
        expected_stock_excess_return=mu,
        long_candidate=long_mask,
        short_candidate=short_mask,
        alpha_short_candidate=alpha_short,
        reduction_only=reduction_only,
        previous_stock_weights=previous,
        factor_betas=frame[[f"beta_risk_{name}" for name in FACTOR_NAMES]].to_numpy(float),
        factor_covariance_root=covariance_root,
        specific_variance_monthly=frame["specific_variance_monthly"].to_numpy(float),
        full_spread=frame["full_spread_20d"].to_numpy(float),
        adv20_usd=frame["adv20_usd"].to_numpy(float),
        impact_volatility_daily=frame["impact_volatility_daily"].to_numpy(float),
        borrow_rate_annual=frame[str(primary["borrow_column"])].to_numpy(float),
        sic2_industry=frame["sic2_industry_category"].to_numpy(str),
        nav_dollars=nav_dollars,
        risk_aversion=risk_aversion,
        forecast_volatility_limit=float(optimizer["forecast_volatility_limit"]),
        maximum_gross=float(optimizer["maximum_gross"]),
        minimum_net=float(optimizer["minimum_net"]),
        maximum_net=float(optimizer["maximum_net"]),
        maximum_position_weight=float(optimizer["maximum_position_weight"]),
        maximum_one_way_turnover=float(optimizer["maximum_one_way_turnover"]),
        maximum_one_way_adv_participation=float(optimizer["maximum_one_way_adv_participation"]),
        market_beta_absolute_limit=float(policy["market_beta_absolute_limit"]),
        other_factor_absolute_limit=float(optimizer["other_factor_absolute_limit"]),
        sic2_industry_net_absolute_limit=float(optimizer["sic2_industry_net_absolute_limit"]),
        sic2_industry_gross_limit=float(optimizer["sic2_industry_gross_limit"]),
        spread_multiplier=float(primary["spread_multiplier"]),
        impact_coefficient=float(primary["impact_coefficient"]),
        financing_rate_annual=float(primary["financing_rate_annual"]),
        previous_spy_weight=previous_spy,
        spy_allowed=bool(policy["spy_allowed"]),
        spy_expected_excess_return=spy_mu,
        spy_factor_exposures=np.asarray(config["spy_hedge"]["factor_exposures"], dtype=float),
        spy_specific_variance_monthly=spy_specific_variance,
        maximum_spy_short_gross=float(config["spy_hedge"]["maximum_short_gross"]),
        spy_transaction_cost_rate=float(spy_primary["one_way_transaction_cost_rate"]),
        spy_borrow_rate_annual=float(spy_primary["borrow_rate_annual"]),
        require_nonnegative_expected_alpha_short_net=bool(
            policy["require_nonnegative_expected_alpha_short_net"]
        ),
        solver=str(optimizer["solver"]),
    )
    weights = solution.stock_weights
    primary_stock = _stock_costs(weights, previous, frame, base_config, nav_dollars, "primary")
    stress_stock = _stock_costs(
        weights, previous, frame, base_config, nav_dollars, "standard_stress"
    )
    long_gross = float(np.maximum(weights, 0.0).sum())
    stock_short_gross = float(np.maximum(-weights, 0.0).sum())
    total_gross = long_gross + stock_short_gross + abs(solution.spy_weight)
    primary_financing = max(total_gross - 1.0, 0.0) * float(primary["financing_rate_annual"]) / 12.0
    stress_cfg = base_config["costs"]["standard_stress"]
    stress_financing = max(total_gross - 1.0, 0.0) * float(stress_cfg["financing_rate_annual"]) / 12.0
    spy_stress_cfg = config["spy_hedge"]["standard_stress"]
    spy_primary_transaction = float(spy_primary["one_way_transaction_cost_rate"]) * abs(
        solution.spy_weight - previous_spy
    )
    spy_stress_transaction = float(spy_stress_cfg["one_way_transaction_cost_rate"]) * abs(
        solution.spy_weight - previous_spy
    )
    spy_primary_borrow = float(spy_primary["borrow_rate_annual"]) / 12.0 * max(
        -solution.spy_weight, 0.0
    )
    spy_stress_borrow = float(spy_stress_cfg["borrow_rate_annual"]) / 12.0 * max(
        -solution.spy_weight, 0.0
    )
    primary_total_cost = (
        primary_stock.transaction
        + primary_stock.borrow
        + spy_primary_transaction
        + spy_primary_borrow
        + primary_financing
    )
    stress_total_cost = (
        stress_stock.transaction
        + stress_stock.borrow
        + spy_stress_transaction
        + spy_stress_borrow
        + stress_financing
    )
    if not np.isclose(primary_total_cost, solution.total_cost, atol=2e-8, rtol=0.0):
        raise RuntimeError("Step 04 optimizer and accounting primary costs do not tie")

    excess = frame["stock_excess_return"].to_numpy(float)
    if np.any((weights != 0.0) & ~np.isfinite(excess)):
        raise RuntimeError("a nonzero Step 04 stock holding lacks an authenticated outcome")
    excess = np.where(np.isfinite(excess), excess, 0.0)
    stock_gross_excess = float(weights @ excess)
    spy_contribution = float(solution.spy_weight * spy_realized_excess)
    gross_excess = stock_gross_excess + spy_contribution
    rf = float(frame["rf"].iloc[0])
    net_return = rf + gross_excess - primary_total_cost
    denominator = 1.0 + net_return
    if denominator <= 0.0:
        raise RuntimeError("Step 04 portfolio NAV became nonpositive")
    end_drift = weights * (1.0 + excess + rf) / denominator
    spy_total_return = spy_realized_excess + rf
    end_spy = solution.spy_weight * (1.0 + spy_total_return) / denominator
    alpha_weight = np.where(alpha_short, weights, 0.0)
    hedge_weight = np.where(hedge_short, weights, 0.0)

    positions = frame.loc[:, [*base.KEYS, "sic2_industry_category", "return_source"]].copy()
    positions["weight"] = weights
    positions["previous_weight"] = previous
    positions["end_drift_weight"] = end_drift
    positions["stock_excess_return"] = excess
    positions["long_candidate"] = long_mask
    positions["alpha_short_candidate"] = alpha_short
    positions["hedge_short_candidate"] = hedge_short
    positions["expected_stock_excess_return"] = mu
    positions = positions.loc[
        (positions["weight"] != 0.0)
        | (positions["previous_weight"] != 0.0)
        | positions["long_candidate"]
        | positions["alpha_short_candidate"]
        | positions["hedge_short_candidate"]
    ].reset_index(drop=True)
    summary = {
        "holding_month": month,
        "signal_date": str(pd.Timestamp(frame["signal_date"].iloc[0]).date()),
        "variant": str(policy["id"]),
        "policy": str(policy["id"]),
        "weighting": "mean_variance_utility",
        "volatility_target": float(optimizer["forecast_volatility_limit"]),
        "risk_aversion": float(risk_aversion),
        "solver_status": solution.status,
        "nav_start": float(nav_dollars),
        "nav_end_primary": float(nav_dollars * (1.0 + net_return)),
        "rf": rf,
        "gross_return": rf + gross_excess,
        "gross_excess_return": gross_excess,
        "net_return_primary": net_return,
        "net_excess_return_primary": gross_excess - primary_total_cost,
        "net_return_stress": rf + gross_excess - stress_total_cost,
        "net_excess_return_stress": gross_excess - stress_total_cost,
        "long_contribution": float(np.maximum(weights, 0.0) @ excess),
        "short_contribution": float(np.minimum(weights, 0.0) @ excess) + spy_contribution,
        "alpha_short_contribution": float(alpha_weight @ excess),
        "hedge_short_contribution": float(hedge_weight @ excess),
        "spy_hedge_contribution": spy_contribution,
        "stock_transaction_cost_primary": primary_stock.transaction,
        "stock_borrow_cost_primary": primary_stock.borrow,
        "spy_transaction_cost_primary": spy_primary_transaction,
        "spy_borrow_cost_primary": spy_primary_borrow,
        "financing_cost_primary": primary_financing,
        "transaction_cost_primary": primary_stock.transaction + spy_primary_transaction,
        "borrow_cost_primary": primary_stock.borrow + spy_primary_borrow,
        "total_cost_primary": primary_total_cost,
        "total_cost_stress": stress_total_cost,
        "forecast_volatility": solution.forecast_volatility,
        "long_gross": long_gross,
        "short_gross": stock_short_gross + abs(solution.spy_weight),
        "stock_short_gross": stock_short_gross,
        "alpha_short_gross": float(-alpha_weight.sum()),
        "hedge_short_gross": float(-hedge_weight.sum()),
        "spy_hedge_gross": abs(solution.spy_weight),
        "spy_weight": solution.spy_weight,
        "end_spy_weight": end_spy,
        "gross": total_gross,
        "net": float(weights.sum() + solution.spy_weight),
        "turnover": solution.turnover,
        "active_long_holdings": int((weights > 1e-8).sum()),
        "active_short_holdings": int((weights < -1e-8).sum()) + int(solution.spy_weight < -1e-8),
        "active_alpha_shorts": int((alpha_weight < -1e-8).sum()),
        "active_hedge_shorts": int((hedge_weight < -1e-8).sum()),
        "eligible_long_candidates": int(long_mask.sum()),
        "eligible_short_candidates": int(short_mask.sum()),
        "expected_spy_excess_return": spy_mu,
        "spy_specific_variance_monthly": spy_specific_variance,
        "maximum_weight": float(np.max(weights, initial=0.0)),
        "minimum_weight": float(np.min(weights, initial=0.0)),
        **{
            f"factor_exposure_{name}": float(solution.factor_exposures[index])
            for index, name in enumerate(FACTOR_NAMES)
        },
    }
    return positions, summary


def _checkpoint_paths(output: Path, phase: str, profile: str, month: str) -> tuple[Path, Path]:
    root = output / phase / profile / "months"
    return root / f"{month}.parquet", root / f"{month}.json"


def _load_resume(
    output: Path,
    phase: str,
    profile: str,
    months: list[str],
    identity: dict[str, str],
) -> tuple[int, dict[int, float], float, float, list[dict[str, Any]]]:
    previous: dict[int, float] = {}
    previous_spy = 0.0
    nav = 100_000_000.0
    summaries: list[dict[str, Any]] = []
    for index, month in enumerate(months):
        parquet_path, receipt_path = _checkpoint_paths(output, phase, profile, month)
        if not parquet_path.is_file() or not receipt_path.is_file():
            return index, previous, previous_spy, nav, summaries
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"Step 04 checkpoint identity mismatch for {phase}/{profile}/{month}")
        positions = pd.read_parquet(parquet_path, columns=["permno", "end_drift_weight"])
        previous = {
            int(row.permno): float(row.end_drift_weight)
            for row in positions.itertuples(index=False)
            if abs(float(row.end_drift_weight)) > 1e-12
        }
        summary = dict(receipt["summary"])
        previous_spy = float(summary["end_spy_weight"])
        nav = float(summary["nav_end_primary"])
        summaries.append(summary)
    return len(months), previous, previous_spy, nav, summaries


def _aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    result = base._aggregate_variant(frame)
    result.update(
        {
            "risk_aversion": float(frame["risk_aversion"].iloc[0]),
            "annualized_long_contribution": float(frame["long_contribution"].mean() * 12.0),
            "annualized_alpha_short_contribution": float(frame["alpha_short_contribution"].mean() * 12.0),
            "annualized_hedge_short_contribution": float(frame["hedge_short_contribution"].mean() * 12.0),
            "annualized_spy_hedge_contribution": float(frame["spy_hedge_contribution"].mean() * 12.0),
            "mean_long_gross": float(frame["long_gross"].mean()),
            "mean_stock_short_gross": float(frame["stock_short_gross"].mean()),
            "mean_alpha_short_gross": float(frame["alpha_short_gross"].mean()),
            "mean_hedge_short_gross": float(frame["hedge_short_gross"].mean()),
            "mean_spy_hedge_gross": float(frame["spy_hedge_gross"].mean()),
            "mean_market_beta": float(frame["factor_exposure_mktrf"].mean()),
            "maximum_absolute_market_beta": float(frame["factor_exposure_mktrf"].abs().max()),
        }
    )
    return result


def _profile_id(policy_id: str, risk_aversion: float) -> str:
    return f"{policy_id}__gamma_{risk_aversion:g}".replace(".", "p")


def _run_profile(
    *,
    output: Path,
    phase: str,
    profile: str,
    months: list[str],
    identity: dict[str, str],
    inputs: base.PreparedInputs,
    spy_history: pd.DataFrame,
    config: dict[str, Any],
    base_config: dict[str, Any],
    policy: dict[str, Any],
    risk_aversion: float,
    month_cache: dict[str, tuple[pd.DataFrame, np.ndarray]],
) -> list[dict[str, Any]]:
    completed, previous, previous_spy, nav, summaries = _load_resume(
        output, phase, profile, months, identity
    )
    for month in months[completed:]:
        print(f"START {phase} {profile} {month}", flush=True)
        try:
            if month not in month_cache:
                month_cache[month] = base._month_frame(inputs, base_config, month)
            frame, covariance_root = month_cache[month]
            positions, summary = _solve_month(
                frame.copy(),
                covariance_root,
                spy_history,
                config,
                base_config,
                policy,
                risk_aversion,
                previous,
                previous_spy,
                nav,
            )
            summary["profile"] = profile
            summary["phase"] = phase
            parquet_path, receipt_path = _checkpoint_paths(output, phase, profile, month)
            _atomic_parquet(parquet_path, positions)
            _atomic_json(
                receipt_path,
                {**identity, "profile": profile, "holding_month": month, "summary": summary},
            )
            previous = {
                int(row.permno): float(row.end_drift_weight)
                for row in positions.itertuples(index=False)
                if abs(float(row.end_drift_weight)) > 1e-12
            }
            previous_spy = float(summary["end_spy_weight"])
            nav = float(summary["nav_end_primary"])
            summaries.append(summary)
            print(
                f"DONE {phase} {profile} {month} net={summary['net_excess_return_primary']:.6f} "
                f"L={summary['long_gross']:.3f} AS={summary['alpha_short_gross']:.3f} "
                f"HS={summary['hedge_short_gross']:.3f} SPY={summary['spy_hedge_gross']:.3f}",
                flush=True,
            )
        except Exception as exc:
            _atomic_json(
                output / "error.json",
                {
                    **identity,
                    "phase": phase,
                    "profile": profile,
                    "holding_month": month,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    return summaries


def run(
    config_path: Path,
    output_root: Path,
    *,
    check_only: bool,
    smoke_month: str | None,
    max_months: int | None,
) -> Path | None:
    config, frozen_base_config, config_sha, input_identity, paths = verify_contract(config_path)
    code_identity = _code_identity()
    identity = {
        "config_sha256": config_sha,
        "input_identity": input_identity,
        "code_identity": code_identity,
    }
    if check_only:
        print(
            f"READY: 2 Step 04 policies, 4 causal validation profiles each, no side floors; "
            f"config={config_sha[:12]}",
            flush=True,
        )
        return None
    output = output_root / f"{config_sha[:12]}-{input_identity[:12]}-{code_identity[:12]}"
    if smoke_month:
        output = output / "smoke"
    _atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_1_portfolio_step_04_run_manifest_v1",
            "status": "RUNNING",
            **identity,
            "config": str(config_path.resolve()),
            "smoke_month": smoke_month,
        },
    )
    loader_config = _expanded_loader_config(frozen_base_config, config)
    inputs = base.load_inputs(loader_config, paths)
    spy_history = _spy_history(inputs)
    month_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    policies = list(config["policies"])
    candidates = [float(value) for value in config["optimizer"]["risk_aversion_candidates"]]

    if smoke_month:
        smoke_summaries: list[dict[str, Any]] = []
        for policy in policies:
            risk_aversion = candidates[0]
            profile = _profile_id(str(policy["id"]), risk_aversion)
            smoke_summaries.extend(
                _run_profile(
                    output=output,
                    phase="smoke",
                    profile=profile,
                    months=[smoke_month],
                    identity=identity,
                    inputs=inputs,
                    spy_history=spy_history,
                    config=config,
                    base_config=loader_config,
                    policy=policy,
                    risk_aversion=risk_aversion,
                    month_cache=month_cache,
                )
            )
        _atomic_parquet(output / "monthly_results.parquet", pd.DataFrame(smoke_summaries))
        _atomic_json(
            output / "run_manifest.json",
            {
                "schema": "research_v6_1_portfolio_step_04_run_manifest_v1",
                "status": "SMOKE_COMPLETE",
                **identity,
                "completed_policy_months": len(smoke_summaries),
            },
        )
        print(f"COMPLETE {output}", flush=True)
        return output

    validation_months = pd.period_range(
        config["evidence"]["risk_aversion_validation_start"],
        config["evidence"]["risk_aversion_validation_end"],
        freq="M",
    ).astype(str).tolist()
    test_months = pd.period_range(
        config["evidence"]["portfolio_start"], config["evidence"]["portfolio_end"], freq="M"
    ).astype(str).tolist()
    if max_months is not None:
        validation_months = validation_months[: int(max_months)]
        test_months = test_months[: int(max_months)]

    validation_rows: list[dict[str, Any]] = []
    validation_summaries: list[dict[str, Any]] = []
    for policy in policies:
        for risk_aversion in candidates:
            profile = _profile_id(str(policy["id"]), risk_aversion)
            rows = _run_profile(
                output=output,
                phase="validation",
                profile=profile,
                months=validation_months,
                identity=identity,
                inputs=inputs,
                spy_history=spy_history,
                config=config,
                base_config=loader_config,
                policy=policy,
                risk_aversion=risk_aversion,
                month_cache=month_cache,
            )
            frame = pd.DataFrame(rows)
            aggregate = _aggregate(frame)
            aggregate["profile"] = profile
            validation_rows.extend(rows)
            validation_summaries.append(aggregate)
    validation_monthly = pd.DataFrame(validation_rows).drop_duplicates(
        ["profile", "holding_month"], keep="last"
    )
    validation_summary = pd.DataFrame(validation_summaries)
    _atomic_parquet(output / "validation_monthly.parquet", validation_monthly)
    _atomic_parquet(output / "validation_summary.parquet", validation_summary)

    selections: list[dict[str, Any]] = []
    for policy in policies:
        eligible = validation_summary.loc[
            validation_summary["policy"].eq(policy["id"])
        ].copy()
        eligible = eligible.sort_values(
            ["net_sharpe_primary", "net_sharpe_stress", "mean_one_way_turnover", "risk_aversion"],
            ascending=[False, False, True, False],
            kind="mergesort",
        )
        winner = eligible.iloc[0]
        selections.append(
            {
                "policy": str(policy["id"]),
                "profile": str(winner["profile"]),
                "risk_aversion": float(winner["risk_aversion"]),
                "validation_net_sharpe_primary": float(winner["net_sharpe_primary"]),
                "validation_net_sharpe_stress": float(winner["net_sharpe_stress"]),
            }
        )
    _atomic_json(output / "selected_profiles.json", {"selections": selections, **identity})

    test_rows: list[dict[str, Any]] = []
    for selection in selections:
        policy = next(item for item in policies if item["id"] == selection["policy"])
        test_rows.extend(
            _run_profile(
                output=output,
                phase="test",
                profile=str(policy["id"]),
                months=test_months,
                identity=identity,
                inputs=inputs,
                spy_history=spy_history,
                config=config,
                base_config=loader_config,
                policy=policy,
                risk_aversion=float(selection["risk_aversion"]),
                month_cache=month_cache,
            )
        )
    monthly = pd.DataFrame(test_rows).drop_duplicates(
        ["policy", "holding_month"], keep="last"
    ).sort_values(["policy", "holding_month"], kind="mergesort").reset_index(drop=True)
    summary = pd.DataFrame(
        [_aggregate(group) for _, group in monthly.groupby("policy", observed=True, sort=True)]
    ).sort_values("net_sharpe_primary", ascending=False, kind="mergesort").reset_index(drop=True)
    _atomic_parquet(output / "monthly_results.parquet", monthly)
    _atomic_parquet(output / "variant_summary.parquet", summary)

    control = pd.read_parquet(
        ROOT / str(config["base_result"]["path"]) / "variant_summary.parquet"
    )
    control = control.loc[
        control["variant"].eq(config["base_result"]["primary_control"])
    ].copy()
    control["lineage"] = "v6_1_step_02_frozen_control"
    challenger = summary.copy()
    challenger["lineage"] = "v6_1_step_04"
    shared = sorted(set(control.columns) & set(challenger.columns))
    _atomic_parquet(
        output / "comparison_summary.parquet",
        pd.concat([control[shared], challenger[shared]], ignore_index=True),
    )
    _atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_1_portfolio_step_04_run_manifest_v1",
            "status": "COMPLETE",
            **identity,
            "config": str(config_path.resolve()),
            "selected_profiles": selections,
            "completed_validation_months": int(len(validation_monthly)),
            "completed_test_months": int(len(monthly)),
            "selector_retrained": False,
            "minimum_long_gross": 0.0,
            "minimum_short_gross": 0.0,
            "realized_short_return_constraint": False,
        },
    )
    print(f"COMPLETE {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-month")
    parser.add_argument("--max-months", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.config.resolve(),
        arguments.output_root.resolve(),
        check_only=arguments.check_only,
        smoke_month=arguments.smoke_month,
        max_months=arguments.max_months,
    )
