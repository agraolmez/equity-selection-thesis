#!/usr/bin/env python3
"""Run the frozen V6.1 Step 03 optional-short portfolio comparison."""

from __future__ import annotations

import argparse
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
from portfolio_bot.research_v6_1.portfolio import (  # noqa: E402
    FACTOR_NAMES,
    causal_monotonic_calibration,
    portfolio_risk,
    solve_expected_net_return_portfolio,
)
from portfolio_bot.research_v6_1.portfolio_step_03 import (  # noqa: E402
    causal_direct_rank_payoff,
    causal_shrunk_linear_calibration,
    stressed_round_trip_hurdle,
)


DEFAULT_CONFIG = ROOT / "configs/research_v6_1/portfolio_step_03.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_1/portfolio_step_03"


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
        raise RuntimeError("V6.1 Step 03 contract is not frozen and ready")
    base_path = ROOT / str(config["base_contract"]["path"])
    if _sha256(base_path) != str(config["base_contract"]["sha256"]):
        raise RuntimeError("bound V6.1 Step 02 contract identity changed")
    base_config, _, input_identity, paths = base.verify_contract(base_path)
    result_root = ROOT / str(config["base_result"]["path"])
    manifest = json.loads((result_root / "run_manifest.json").read_text())
    if manifest.get("status") != config["base_result"]["required_status"]:
        raise RuntimeError("bound V6.1 Step 02 result is not complete")
    if manifest.get("config_sha256") != config["base_contract"]["sha256"]:
        raise RuntimeError("bound V6.1 Step 02 result used another contract")
    for key in ("portfolio_start", "portfolio_end"):
        if config["evidence"][key] != base_config["evidence"][key]:
            raise RuntimeError(f"Step 03 {key} differs from its bound Step 02 window")
    return config, base_config, _sha256(config_path), input_identity, paths


def _mapping_values(
    inputs: base.PreparedInputs,
    base_config: dict[str, Any],
    config: dict[str, Any],
    month: str,
    method: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    frame, covariance_root = base._month_frame(inputs, base_config, month)
    history = inputs.forecasts.loc[inputs.forecasts["holding_month"].lt(month)].copy()
    signal_date = pd.Timestamp(frame["signal_date"].iloc[0])
    policy = config["rank_to_payoff"]
    specifications = {
        "long": ("rank_long_net", "payoff_long"),
        "short": ("rank_short_net", "payoff_short"),
    }
    for side, (score_column, outcome_column) in specifications.items():
        common = {
            "score_column": score_column,
            "outcome_column": outcome_column,
            "month_column": "holding_month",
            "availability_column": "label_available_date",
            "signal_date": signal_date,
            "minimum_history_months": int(policy["minimum_history_months"]),
            "shrinkage_prior_months": float(policy["shrinkage_prior_months"]),
        }
        if method == "direct_rank":
            result = causal_direct_rank_payoff(
                history,
                frame[score_column].to_numpy(float),
                threshold=float(config["formation"]["entry_percentile"]),
                **common,
            )
        elif method == "shrunk_linear":
            result = causal_shrunk_linear_calibration(
                history,
                frame[score_column].to_numpy(float),
                **common,
            )
        elif method == "broad_bucket":
            result = causal_monotonic_calibration(
                history,
                frame[score_column].to_numpy(float),
                score_column=score_column,
                outcome_column=outcome_column,
                month_column="holding_month",
                availability_column="label_available_date",
                signal_date=signal_date,
                bin_edges=policy["broad_bucket"]["bin_edges"],
                minimum_history_months=int(policy["minimum_history_months"]),
                shrinkage_prior_months=float(policy["shrinkage_prior_months"]),
            )
        else:
            raise ValueError(f"unknown Step 03 rank mapping: {method}")
        frame[f"expected_{side}"] = result.values
        frame[f"mapping_months_{side}"] = result.history_months
    return frame, covariance_root


def _alpha_gate(frame: pd.DataFrame, policy: str, *, long: bool) -> np.ndarray:
    gate = np.ones(len(frame), dtype=bool)
    comparator = np.greater if long else np.less
    if policy in {"return_plus_capm", "return_plus_both"}:
        gate &= comparator(frame["expected_capm"].to_numpy(float), 0.0)
    if policy in {"return_plus_ff5_momentum", "return_plus_both"}:
        gate &= comparator(frame["expected_ff5_momentum"].to_numpy(float), 0.0)
    return gate


def _candidate_masks(
    frame: pd.DataFrame,
    config: dict[str, Any],
    policy: str,
    previous: np.ndarray,
    nav_dollars: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    formation = config["formation"]
    entry = float(formation["entry_percentile"])
    holding = float(formation["holding_percentile"])
    long_cutoff = np.where(previous > 0.0, holding, entry)
    short_cutoff = np.where(previous < 0.0, holding, entry)
    rank_long = frame["rank_long_net"].to_numpy(float)
    rank_short = frame["rank_short_net"].to_numpy(float)
    long_mask = (
        frame["long_eligible"].fillna(False).to_numpy(bool)
        & frame["new_long_trade_allowed"].fillna(False).to_numpy(bool)
        & (rank_long >= long_cutoff)
        & _alpha_gate(frame, policy, long=True)
    )
    alpha_short = (
        frame["short_eligible"].fillna(False).to_numpy(bool)
        & frame["new_short_trade_allowed"].fillna(False).to_numpy(bool)
        & (rank_short >= short_cutoff)
        & _alpha_gate(frame, policy, long=False)
    )
    cost_config = config["cost_hurdle"]
    stress = frame.attrs["base_config"]["costs"]["standard_stress"]
    common = {
        "full_spread": frame["full_spread_20d"].to_numpy(float),
        "adv20_usd": frame["adv20_usd"].to_numpy(float),
        "impact_volatility_daily": frame["impact_volatility_daily"].to_numpy(float),
        "stress_borrow_rate_annual": frame["stress_borrow_rate_annual"].to_numpy(float),
        "nav_dollars": nav_dollars,
        "representative_position_weight": float(cost_config["representative_position_weight"]),
        "spread_multiplier": float(stress["spread_multiplier"]),
        "impact_coefficient": float(stress["impact_coefficient"]),
        "commission_each_way": float(cost_config["commission_each_way"]),
    }
    long_hurdle = stressed_round_trip_hurdle(include_borrow=False, **common)
    short_hurdle = stressed_round_trip_hurdle(include_borrow=True, **common)
    long_mask &= frame["expected_long"].to_numpy(float) > long_hurdle
    alpha_short &= frame["expected_short"].to_numpy(float) > short_hurdle

    overlap = long_mask & alpha_short
    if overlap.any():
        prefer_long = frame["expected_long"].to_numpy(float) >= frame["expected_short"].to_numpy(float)
        alpha_short[overlap & prefer_long] = False
        long_mask[overlap & ~prefer_long] = False
    long_mask[previous < 0.0] = False
    alpha_short[previous > 0.0] = False

    hedge = formation["hedge_short"]
    adv = frame["adv20_usd"].to_numpy(float)
    borrow = frame["stress_borrow_rate_annual"].to_numpy(float)
    adv_cut = float(np.nanquantile(adv, float(hedge["minimum_adv_cross_section_percentile"])))
    borrow_cut = float(np.nanquantile(borrow, float(hedge["maximum_stress_borrow_cross_section_percentile"])))
    hedge_short = (
        frame["short_eligible"].fillna(False).to_numpy(bool)
        & frame["new_short_trade_allowed"].fillna(False).to_numpy(bool)
        & (adv >= adv_cut)
        & (borrow <= borrow_cut)
        & ~alpha_short
        & ~long_mask
        & (previous <= 0.0)
    )
    short_mask = alpha_short | hedge_short
    reduction_only = (previous != 0.0) & ~(long_mask | short_mask)
    return long_mask, alpha_short, hedge_short, short_mask, reduction_only


def _solve_month(
    frame: pd.DataFrame,
    covariance_root: np.ndarray,
    config: dict[str, Any],
    base_config: dict[str, Any],
    variant: dict[str, str],
    previous_map: dict[int, float],
    nav_dollars: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame.attrs["base_config"] = base_config
    previous = base._aligned_previous(frame, previous_map)
    long_mask, alpha_short, hedge_short, short_mask, reduction_only = _candidate_masks(
        frame, config, variant["policy"], previous, nav_dollars
    )
    mu = np.zeros(len(frame), dtype=np.float64)
    mu[long_mask] = frame.loc[long_mask, "expected_long"].to_numpy(float)
    mu[alpha_short] = -frame.loc[alpha_short, "expected_short"].to_numpy(float)
    betas = frame[[f"beta_risk_{name}" for name in FACTOR_NAMES]].to_numpy(float)
    optimizer = config["optimizer"]
    primary = base_config["costs"]["primary"]
    solution = solve_expected_net_return_portfolio(
        expected_stock_excess_return=mu,
        long_candidate=long_mask,
        short_candidate=short_mask,
        hedge_short_candidate=hedge_short,
        maximum_hedge_short_gross=float(config["formation"]["hedge_short"]["maximum_gross"]),
        reduction_only=reduction_only,
        previous_weights=previous,
        factor_betas=betas,
        factor_covariance_root=covariance_root,
        specific_variance_monthly=frame["specific_variance_monthly"].to_numpy(float),
        full_spread=frame["full_spread_20d"].to_numpy(float),
        adv20_usd=frame["adv20_usd"].to_numpy(float),
        impact_volatility_daily=frame["impact_volatility_daily"].to_numpy(float),
        borrow_rate_annual=frame[str(primary["borrow_column"])].to_numpy(float),
        sic2_industry=frame["sic2_industry_category"].to_numpy(str),
        nav_dollars=nav_dollars,
        forecast_volatility_target=float(optimizer["forecast_volatility"]),
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
        spread_multiplier=float(primary["spread_multiplier"]),
        impact_coefficient=float(primary["impact_coefficient"]),
        financing_rate_annual=float(primary["financing_rate_annual"]),
        solver=str(optimizer["solver"]),
    )
    weights = solution.weights
    primary_cost, stress_cost = base._cost_breakdowns(weights, previous, frame, base_config, nav_dollars)
    excess = frame["stock_excess_return"].to_numpy(float)
    if np.any((weights != 0.0) & ~np.isfinite(excess)):
        raise RuntimeError("a nonzero Step 03 holding lacks an authenticated outcome")
    excess = np.where(np.isfinite(excess), excess, 0.0)
    gross_excess = float(weights @ excess)
    rf = float(frame["rf"].iloc[0])
    net_return = rf + gross_excess - primary_cost.total
    denominator = 1.0 + net_return
    if denominator <= 0.0:
        raise RuntimeError("Step 03 portfolio NAV became nonpositive")
    end_drift = weights * (1.0 + excess + rf) / denominator
    alpha_short_weight = np.where(alpha_short, weights, 0.0)
    hedge_short_weight = np.where(hedge_short, weights, 0.0)

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
        "holding_month": str(frame["holding_month"].iloc[0]),
        "signal_date": str(pd.Timestamp(frame["signal_date"].iloc[0]).date()),
        "variant": variant["id"],
        "policy": variant["policy"],
        "weighting": variant["mapping"],
        "volatility_target": float(optimizer["forecast_volatility"]),
        "solver_status": solution.status,
        "nav_start": nav_dollars,
        "nav_end_primary": nav_dollars * (1.0 + net_return),
        "rf": rf,
        "gross_return": rf + gross_excess,
        "gross_excess_return": gross_excess,
        "net_return_primary": net_return,
        "net_excess_return_primary": gross_excess - primary_cost.total,
        "net_return_stress": rf + gross_excess - stress_cost.total,
        "net_excess_return_stress": gross_excess - stress_cost.total,
        "long_contribution": float(np.maximum(weights, 0.0) @ excess),
        "short_contribution": float(np.minimum(weights, 0.0) @ excess),
        "alpha_short_contribution": float(alpha_short_weight @ excess),
        "hedge_short_contribution": float(hedge_short_weight @ excess),
        "transaction_cost_primary": primary_cost.transaction,
        "borrow_cost_primary": primary_cost.borrow,
        "financing_cost_primary": primary_cost.financing,
        "total_cost_primary": primary_cost.total,
        "total_cost_stress": stress_cost.total,
        "forecast_volatility": solution.forecast_volatility,
        "long_gross": solution.long_gross,
        "short_gross": solution.short_gross,
        "alpha_short_gross": float(-alpha_short_weight.sum()),
        "hedge_short_gross": float(-hedge_short_weight.sum()),
        "gross": solution.gross,
        "net": solution.net,
        "turnover": solution.turnover,
        "active_long_holdings": int((weights > 1e-8).sum()),
        "active_short_holdings": int((weights < -1e-8).sum()),
        "active_alpha_shorts": int((alpha_short_weight < -1e-8).sum()),
        "active_hedge_shorts": int((hedge_short_weight < -1e-8).sum()),
        "eligible_long_candidates": int(long_mask.sum()),
        "eligible_short_candidates": int(short_mask.sum()),
        "maximum_weight": float(np.max(weights, initial=0.0)),
        "minimum_weight": float(np.min(weights, initial=0.0)),
        **{f"factor_exposure_{name}": float(solution.factor_exposures[i]) for i, name in enumerate(FACTOR_NAMES)},
    }
    return positions, summary


def _variants(config: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"id": f"{policy}__{mapping}", "policy": policy, "mapping": mapping}
        for policy in config["formation"]["policies"]
        for mapping in config["rank_to_payoff"]["methods"]
    ]


def _checkpoint_paths(output: Path, variant: str, month: str) -> tuple[Path, Path]:
    root = output / "variants" / variant / "months"
    return root / f"{month}.parquet", root / f"{month}.json"


def _load_resume(
    output: Path,
    variant: str,
    months: list[str],
    identity: dict[str, str],
) -> tuple[int, dict[int, float], float, list[dict[str, Any]]]:
    previous: dict[int, float] = {}
    nav = 100_000_000.0
    summaries: list[dict[str, Any]] = []
    for index, month in enumerate(months):
        parquet_path, receipt_path = _checkpoint_paths(output, variant, month)
        if not parquet_path.is_file() or not receipt_path.is_file():
            return index, previous, nav, summaries
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"Step 03 checkpoint identity mismatch for {variant} {month}")
        positions = pd.read_parquet(parquet_path, columns=["permno", "end_drift_weight"])
        previous = {
            int(row.permno): float(row.end_drift_weight)
            for row in positions.itertuples(index=False)
            if abs(float(row.end_drift_weight)) > 1e-12
        }
        summary = dict(receipt["summary"])
        nav = float(summary["nav_end_primary"])
        summaries.append(summary)
    return len(months), previous, nav, summaries


def _aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    result = base._aggregate_variant(frame)
    result.update(
        {
            "annualized_alpha_short_contribution": float(frame["alpha_short_contribution"].mean() * 12.0),
            "annualized_hedge_short_contribution": float(frame["hedge_short_contribution"].mean() * 12.0),
            "mean_alpha_short_gross": float(frame["alpha_short_gross"].mean()),
            "mean_hedge_short_gross": float(frame["hedge_short_gross"].mean()),
            "mean_active_alpha_shorts": float(frame["active_alpha_shorts"].mean()),
            "mean_active_hedge_shorts": float(frame["active_hedge_shorts"].mean()),
        }
    )
    return result


def run(
    config_path: Path,
    output_root: Path,
    *,
    check_only: bool,
    selected_variants: set[str] | None,
    smoke_month: str | None,
    max_months: int | None,
) -> Path | None:
    config, base_config, config_sha, input_identity, paths = verify_contract(config_path)
    variants = _variants(config)
    if selected_variants:
        variants = [item for item in variants if item["id"] in selected_variants]
        missing = selected_variants - {item["id"] for item in variants}
        if missing:
            raise ValueError(f"unknown Step 03 variants: {sorted(missing)}")
    if check_only:
        print(
            f"READY: {len(variants)} Step 03 paths; frozen selectors; no side floor; "
            f"config={config_sha[:12]}",
            flush=True,
        )
        return None
    code_identity = _code_identity()
    output = output_root / f"{config_sha[:12]}-{input_identity[:12]}-{code_identity[:12]}"
    if smoke_month:
        output = output / "smoke"
    identity = {
        "config_sha256": config_sha,
        "input_identity": input_identity,
        "code_identity": code_identity,
    }
    _atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_1_portfolio_step_03_run_manifest_v1",
            "status": "RUNNING",
            **identity,
            "config": str(config_path.resolve()),
            "variants": [item["id"] for item in variants],
            "smoke_month": smoke_month,
        },
    )
    inputs = base.load_inputs(base_config, paths)
    months = pd.period_range(
        config["evidence"]["portfolio_start"], config["evidence"]["portfolio_end"], freq="M"
    ).astype(str).tolist()
    if smoke_month:
        if smoke_month not in months:
            raise ValueError("smoke month is outside the Step 03 window")
        months = [smoke_month]
    if max_months is not None:
        months = months[: int(max_months)]
    all_summaries: list[dict[str, Any]] = []
    month_cache: dict[tuple[str, str], tuple[pd.DataFrame, np.ndarray]] = {}
    for variant in variants:
        completed, previous, nav, summaries = _load_resume(
            output, variant["id"], months, identity
        )
        all_summaries.extend(summaries)
        for month in months[completed:]:
            print(f"START {variant['id']} {month}", flush=True)
            try:
                cache_key = (month, variant["mapping"])
                if cache_key not in month_cache:
                    month_cache[cache_key] = _mapping_values(
                        inputs, base_config, config, month, variant["mapping"]
                    )
                frame, covariance_root = month_cache[cache_key]
                positions, summary = _solve_month(
                    frame.copy(), covariance_root, config, base_config, variant, previous, nav
                )
                parquet_path, receipt_path = _checkpoint_paths(output, variant["id"], month)
                _atomic_parquet(parquet_path, positions)
                _atomic_json(
                    receipt_path,
                    {**identity, "variant": variant["id"], "holding_month": month, "summary": summary},
                )
                previous = {
                    int(row.permno): float(row.end_drift_weight)
                    for row in positions.itertuples(index=False)
                    if abs(float(row.end_drift_weight)) > 1e-12
                }
                nav = float(summary["nav_end_primary"])
                all_summaries.append(summary)
                print(
                    f"DONE {variant['id']} {month} net={summary['net_excess_return_primary']:.6f} "
                    f"L={summary['long_gross']:.3f} AS={summary['alpha_short_gross']:.3f} "
                    f"HS={summary['hedge_short_gross']:.3f}",
                    flush=True,
                )
            except Exception as exc:
                _atomic_json(
                    output / "error.json",
                    {
                        **identity,
                        "variant": variant["id"],
                        "holding_month": month,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
    monthly = pd.DataFrame(all_summaries).drop_duplicates(
        ["variant", "holding_month"], keep="last"
    ).sort_values(["variant", "holding_month"], kind="mergesort").reset_index(drop=True)
    _atomic_parquet(output / "monthly_results.parquet", monthly)
    summary = pd.DataFrame(
        [_aggregate(group) for _, group in monthly.groupby("variant", observed=True, sort=True)]
    ).sort_values("net_sharpe_primary", ascending=False, kind="mergesort").reset_index(drop=True)
    _atomic_parquet(output / "variant_summary.parquet", summary)
    controls = pd.read_parquet(
        ROOT / str(config["base_result"]["path"]) / "variant_summary.parquet"
    )
    controls = controls.loc[
        controls["variant"].isin(
            [
                "return_plus_both__optimized_10pct_vol",
                "return_plus_ff5_momentum__optimized_10pct_vol",
                "return_only__squared_tail",
            ]
        )
    ].copy()
    controls["lineage"] = "v6_1_step_02_control"
    challenger = summary.copy()
    challenger["lineage"] = "v6_1_step_03"
    shared = sorted(set(controls.columns) & set(challenger.columns))
    _atomic_parquet(
        output / "comparison_summary.parquet",
        pd.concat([controls[shared], challenger[shared]], ignore_index=True),
    )
    _atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_1_portfolio_step_03_run_manifest_v1",
            "status": "SMOKE_COMPLETE" if smoke_month else "COMPLETE",
            **identity,
            "config": str(config_path.resolve()),
            "variants": [item["id"] for item in variants],
            "completed_variant_months": int(len(monthly)),
            "expected_variant_months": int(len(variants) * len(months)),
            "selector_retrained": False,
            "minimum_short_gross": 0.0,
        },
    )
    print(f"COMPLETE {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--smoke-month")
    parser.add_argument("--max-months", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.config.resolve(),
        arguments.output_root.resolve(),
        check_only=arguments.check_only,
        selected_variants=set(arguments.variant) or None,
        smoke_month=arguments.smoke_month,
        max_months=arguments.max_months,
    )
