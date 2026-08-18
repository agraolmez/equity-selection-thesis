#!/usr/bin/env python3
"""Run the bounded final V6.1 optimizer-regularization challenge."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_v6_1_portfolio_step_02 as base  # noqa: E402
from portfolio_bot.research_v6_1.portfolio import (  # noqa: E402
    FACTOR_NAMES,
    solve_expected_net_return_portfolio,
)


DEFAULT_CONFIG = ROOT / "configs/research_v6_1/final_optimizer_challenge.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_1/final_optimizer_challenge"
POLICY = "return_plus_ff5_momentum"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _code_identity() -> str:
    files = [
        Path(__file__).resolve(),
        ROOT / "src/portfolio_bot/research_v6_1/portfolio.py",
        ROOT / "scripts/run_v6_1_portfolio_step_02.py",
        ROOT / "src/portfolio_bot/allocation/costs.py",
        ROOT / "src/portfolio_bot/risk_cost/cost_math.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _verify(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], str, str]:
    challenge = json.loads(config_path.read_text())
    if challenge.get("status") != "FROZEN_READY_TO_RUN":
        raise RuntimeError("final optimizer challenge is not frozen")
    base_path = ROOT / str(challenge["base_portfolio_contract"]["path"])
    if _sha256(base_path) != challenge["base_portfolio_contract"]["sha256"]:
        raise RuntimeError("base portfolio contract identity changed")
    base_config, _, input_identity, paths = base.verify_contract(base_path)
    control_root = ROOT / str(challenge["frozen_control"]["path"])
    manifest = control_root / "run_manifest.json"
    if not manifest.is_file() or json.loads(manifest.read_text()).get("status") != "COMPLETE":
        raise RuntimeError("frozen control portfolio is not complete")
    config_sha = _sha256(config_path)
    identity = hashlib.sha256(
        _json_bytes(
            {
                "challenge": config_sha,
                "base": challenge["base_portfolio_contract"]["sha256"],
                "inputs": input_identity,
                "control_manifest": _sha256(manifest),
            }
        )
    ).hexdigest()
    return challenge, base_config, paths, config_sha, identity


def _regularized_mu(
    frame: pd.DataFrame,
    long_mask: np.ndarray,
    short_mask: np.ndarray,
    dispersion_scale: float,
) -> np.ndarray:
    scale = float(dispersion_scale)
    if scale < 0.0 or scale > 1.0:
        raise ValueError("dispersion scale must lie in [0, 1]")
    mu = np.zeros(len(frame), dtype=np.float64)
    mu[long_mask] = frame.loc[long_mask, "expected_long"].to_numpy(float)
    mu[short_mask] = -frame.loc[short_mask, "expected_short"].to_numpy(float)
    if scale == 1.0:
        return mu
    for mask in (long_mask, short_mask):
        if mask.any():
            anchor = float(np.median(mu[mask]))
            mu[mask] = anchor + scale * (mu[mask] - anchor)
    return mu


def _solve_month(
    frame: pd.DataFrame,
    covariance_root: np.ndarray,
    config: dict[str, Any],
    profile: dict[str, Any],
    previous_map: dict[int, float],
    nav_dollars: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if str(profile["id"]) == "control":
        variant = {
            "id": "return_plus_ff5_momentum__optimized_10pct_vol",
            "policy": POLICY,
            "kind": "optimized",
            "method": "optimized",
            "volatility": float(config["optimizer"]["primary_forecast_volatility"]),
        }
        solution, positions, summary = base._solve_variant_month(
            frame, covariance_root, config, variant, previous_map, nav_dollars
        )
        previous = base._aligned_previous(frame, previous_map)
        participation = np.divide(
            np.abs(solution.weights - previous) * nav_dollars,
            frame["adv20_usd"].to_numpy(float),
            out=np.zeros(len(frame)),
            where=frame["adv20_usd"].to_numpy(float) > 0.0,
        )
        participation_by_permno = dict(
            zip(frame["permno"].astype(int), participation, strict=True)
        )
        positions["regularized_expected_stock_excess_return"] = positions[
            "expected_stock_excess_return"
        ]
        positions["adv_participation"] = (
            positions["permno"].astype(int).map(participation_by_permno).to_numpy(float)
        )
        summary.update(
            {
                "variant": "control",
                "profile": "control",
                "family": "control",
                "weighting": "regularized_expected_net_optimizer",
                "weight_l2_squared": float(solution.weights @ solution.weights),
                "maximum_adv_participation": float(np.max(participation, initial=0.0)),
            }
        )
        return positions, summary

    long_mask, short_mask = base._policy_masks(frame, config, POLICY)
    previous = base._aligned_previous(frame, previous_map)
    opposing = ((previous > 0.0) & short_mask) | ((previous < 0.0) & long_mask)
    long_mask[opposing] = False
    short_mask[opposing] = False
    reduction_only = (previous != 0.0) & ~(long_mask | short_mask)
    betas = frame[[f"beta_risk_{name}" for name in FACTOR_NAMES]].to_numpy(float)
    specific = frame["specific_variance_monthly"].to_numpy(float)
    optimizer = config["optimizer"]
    primary = config["costs"]["primary"]
    mu = _regularized_mu(
        frame, long_mask, short_mask, float(profile["dispersion_scale"])
    )
    solution = solve_expected_net_return_portfolio(
        expected_stock_excess_return=mu,
        long_candidate=long_mask,
        short_candidate=short_mask,
        reduction_only=reduction_only,
        previous_weights=previous,
        factor_betas=betas,
        factor_covariance_root=covariance_root,
        specific_variance_monthly=specific,
        full_spread=frame["full_spread_20d"].to_numpy(float),
        adv20_usd=frame["adv20_usd"].to_numpy(float),
        impact_volatility_daily=frame["impact_volatility_daily"].to_numpy(float),
        borrow_rate_annual=frame[str(primary["borrow_column"])].to_numpy(float),
        sic2_industry=frame["sic2_industry_category"].to_numpy(str),
        nav_dollars=nav_dollars,
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
        spread_multiplier=float(primary["spread_multiplier"]),
        impact_coefficient=float(primary["impact_coefficient"]),
        financing_rate_annual=float(primary["financing_rate_annual"]),
        risk_aversion=float(profile["risk_aversion"]),
        concentration_aversion=float(profile["concentration_aversion"]),
        objective_cost_multiplier=float(profile["objective_cost_multiplier"]),
        solver=str(optimizer["solver"]),
    )
    weights = solution.weights
    primary_cost, stress_cost = base._cost_breakdowns(weights, previous, frame, config, nav_dollars)
    excess = frame["stock_excess_return"].to_numpy(float)
    if np.any((weights != 0.0) & ~np.isfinite(excess)):
        raise RuntimeError("a held stock lacks an authenticated outcome")
    excess = np.where(np.isfinite(excess), excess, 0.0)
    gross_excess = float(weights @ excess)
    rf = float(frame["rf"].iloc[0])
    net_return = rf + gross_excess - primary_cost.total
    stress_return = rf + gross_excess - stress_cost.total
    if 1.0 + net_return <= 0.0:
        raise RuntimeError("portfolio NAV became nonpositive")
    end_drift = weights * (1.0 + excess + rf) / (1.0 + net_return)
    participation = np.divide(
        np.abs(weights - previous) * nav_dollars,
        frame["adv20_usd"].to_numpy(float),
        out=np.zeros(len(frame)),
        where=frame["adv20_usd"].to_numpy(float) > 0.0,
    )

    positions = frame.loc[:, [*base.KEYS, "sic2_industry_category", "return_source"]].copy()
    positions["weight"] = weights
    positions["previous_weight"] = previous
    positions["end_drift_weight"] = end_drift
    positions["stock_excess_return"] = excess
    positions["long_candidate"] = long_mask
    positions["short_candidate"] = short_mask
    positions["regularized_expected_stock_excess_return"] = mu
    positions["adv_participation"] = participation
    positions = positions.loc[
        (positions["weight"] != 0.0)
        | (positions["previous_weight"] != 0.0)
        | positions["long_candidate"]
        | positions["short_candidate"]
    ].reset_index(drop=True)

    summary = {
        "holding_month": str(frame["holding_month"].iloc[0]),
        "signal_date": str(pd.Timestamp(frame["signal_date"].iloc[0]).date()),
        "variant": str(profile["id"]),
        "profile": str(profile["id"]),
        "family": str(profile["family"]),
        "policy": POLICY,
        "weighting": "regularized_expected_net_optimizer",
        "volatility_target": float(optimizer["primary_forecast_volatility"]),
        "solver_status": solution.status,
        "nav_start": float(nav_dollars),
        "nav_end_primary": float(nav_dollars * (1.0 + net_return)),
        "rf": rf,
        "gross_return": rf + gross_excess,
        "gross_excess_return": gross_excess,
        "net_return_primary": net_return,
        "net_excess_return_primary": gross_excess - primary_cost.total,
        "net_return_stress": stress_return,
        "net_excess_return_stress": gross_excess - stress_cost.total,
        "long_contribution": float(np.maximum(weights, 0.0) @ excess),
        "short_contribution": float(np.minimum(weights, 0.0) @ excess),
        "transaction_cost_primary": primary_cost.transaction,
        "borrow_cost_primary": primary_cost.borrow,
        "financing_cost_primary": primary_cost.financing,
        "total_cost_primary": primary_cost.total,
        "total_cost_stress": stress_cost.total,
        "forecast_volatility": solution.forecast_volatility,
        "long_gross": solution.long_gross,
        "short_gross": solution.short_gross,
        "gross": solution.gross,
        "net": solution.net,
        "turnover": solution.turnover,
        "weight_l2_squared": float(weights @ weights),
        "maximum_adv_participation": float(np.max(participation, initial=0.0)),
        "active_long_holdings": int((weights > 1e-8).sum()),
        "active_short_holdings": int((weights < -1e-8).sum()),
        "eligible_long_candidates": int(long_mask.sum()),
        "eligible_short_candidates": int(short_mask.sum()),
        "maximum_weight": float(np.max(weights, initial=0.0)),
        "minimum_weight": float(np.min(weights, initial=0.0)),
        **{
            f"factor_exposure_{name}": float(solution.factor_exposures[index])
            for index, name in enumerate(FACTOR_NAMES)
        },
    }
    return positions, summary


def _checkpoint(output: Path, phase: str, profile: str, month: str) -> tuple[Path, Path]:
    root = output / phase / profile / "months"
    return root / f"{month}.parquet", root / f"{month}.json"


def _run_profile(
    *,
    output: Path,
    phase: str,
    months: list[str],
    profile: dict[str, Any],
    identity: dict[str, str],
    inputs: base.PreparedInputs,
    loader_config: dict[str, Any],
    month_cache: dict[str, tuple[pd.DataFrame, np.ndarray]],
) -> pd.DataFrame:
    previous: dict[int, float] = {}
    nav = float(loader_config["nav_dollars"])
    summaries: list[dict[str, Any]] = []
    completed = 0
    for month in months:
        parquet_path, receipt_path = _checkpoint(output, phase, str(profile["id"]), month)
        if parquet_path.is_file() and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if any(receipt.get(key) != value for key, value in identity.items()):
                raise RuntimeError(f"checkpoint identity mismatch for {phase} {profile['id']} {month}")
            positions = pd.read_parquet(parquet_path)
            summary = dict(receipt["summary"])
            previous = {
                int(row.permno): float(row.end_drift_weight)
                for row in positions.itertuples(index=False)
                if abs(float(row.end_drift_weight)) > 1e-12
            }
            nav = float(summary["nav_end_primary"])
            summaries.append(summary)
            completed += 1
            continue
        if completed != len(summaries):
            raise RuntimeError("noncontiguous optimizer checkpoint state")
        if month not in month_cache:
            month_cache[month] = base._month_frame(inputs, loader_config, month)
        frame, covariance_root = month_cache[month]
        positions, summary = _solve_month(
            frame, covariance_root, loader_config, profile, previous, nav
        )
        base._atomic_parquet(parquet_path, positions)
        receipt = {
            **identity,
            "phase": phase,
            "profile": profile["id"],
            "holding_month": month,
            "summary": summary,
        }
        base._atomic_json(receipt_path, receipt)
        previous = {
            int(row.permno): float(row.end_drift_weight)
            for row in positions.itertuples(index=False)
            if abs(float(row.end_drift_weight)) > 1e-12
        }
        nav = float(summary["nav_end_primary"])
        summaries.append(summary)
        completed += 1
        print(
            f"DONE {phase} {profile['id']} {month} "
            f"net={summary['net_excess_return_primary']:.6f}",
            flush=True,
        )
    return pd.DataFrame(summaries).sort_values("holding_month", kind="mergesort").reset_index(drop=True)


def _aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    result = base._aggregate_variant(frame)
    net = frame["net_excess_return_primary"].to_numpy(float)
    stress = frame["net_excess_return_stress"].to_numpy(float)
    without_best_two = np.sort(net)[:-2] if len(net) > 2 else np.asarray([], dtype=float)
    result.update(
        {
            "profile": str(frame["profile"].iloc[0]),
            "family": str(frame["family"].iloc[0]),
            "annualized_long_contribution": float(frame["long_contribution"].mean() * 12.0),
            "annualized_short_contribution": float(frame["short_contribution"].mean() * 12.0),
            "mean_weight_l2_squared": float(frame["weight_l2_squared"].mean()),
            "maximum_adv_participation": float(frame["maximum_adv_participation"].max()),
            "annualized_net_excess_without_best_two_months": (
                float(without_best_two.mean() * 12.0) if len(without_best_two) else float("nan")
            ),
            "positive_stress_month_fraction": float((stress > 0.0).mean()),
        }
    )
    return result


def _hac_t(values: np.ndarray, lags: int) -> float:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    centered = values - mean
    n = len(values)
    long_run = float(centered @ centered / n)
    for lag in range(1, min(int(lags), n - 1) + 1):
        weight = 1.0 - lag / (int(lags) + 1.0)
        long_run += 2.0 * weight * float(centered[lag:] @ centered[:-lag] / n)
    standard_error = np.sqrt(max(long_run, 0.0) / n)
    return mean / standard_error if standard_error > 0.0 else float("nan")


def _paired_inference(
    challenger: pd.DataFrame,
    control: pd.DataFrame,
    specification: dict[str, Any],
) -> dict[str, Any]:
    joined = challenger[["holding_month", "net_excess_return_primary"]].merge(
        control[["holding_month", "net_excess_return_primary"]],
        on="holding_month",
        suffixes=("_challenger", "_control"),
        validate="one_to_one",
    ).sort_values("holding_month", kind="mergesort")
    challenger_return = joined["net_excess_return_primary_challenger"].to_numpy(float)
    control_return = joined["net_excess_return_primary_control"].to_numpy(float)
    difference = challenger_return - control_return
    n = len(joined)
    block = int(specification["circular_block_length_months"])
    resamples = int(specification["bootstrap_resamples"])
    rng = np.random.default_rng(int(specification["bootstrap_seed"]))
    starts = rng.integers(0, n, size=(resamples, int(np.ceil(n / block))))
    indices = (starts[:, :, None] + np.arange(block)) % n
    indices = indices.reshape(resamples, -1)[:, :n]
    candidate_samples = challenger_return[indices]
    control_samples = control_return[indices]
    candidate_sharpe = (
        candidate_samples.mean(axis=1)
        / candidate_samples.std(axis=1, ddof=1)
        * np.sqrt(12.0)
    )
    control_sharpe = (
        control_samples.mean(axis=1)
        / control_samples.std(axis=1, ddof=1)
        * np.sqrt(12.0)
    )
    sharpe_difference = candidate_sharpe - control_sharpe
    mean_difference = (candidate_samples - control_samples).mean(axis=1) * 12.0
    candidate_point = float(
        challenger_return.mean() / challenger_return.std(ddof=1) * np.sqrt(12.0)
    )
    control_point = float(control_return.mean() / control_return.std(ddof=1) * np.sqrt(12.0))
    return {
        "months": n,
        "challenger_sharpe": candidate_point,
        "control_sharpe": control_point,
        "sharpe_difference": candidate_point - control_point,
        "paired_hac_t_mean_net_return_difference": _hac_t(
            difference, int(specification["hac_lags"])
        ),
        "annualized_mean_net_return_difference": float(difference.mean() * 12.0),
        "bootstrap_sharpe_difference_ci_95": np.quantile(
            sharpe_difference, [0.025, 0.975]
        ).tolist(),
        "bootstrap_annualized_mean_difference_ci_95": np.quantile(
            mean_difference, [0.025, 0.975]
        ).tolist(),
        "bootstrap_probability_sharpe_difference_positive": float(
            np.mean(sharpe_difference > 0.0)
        ),
        "challenger_hac_t_mean_net_return": _hac_t(
            challenger_return, int(specification["hac_lags"])
        ),
    }


def _constraint_audit(frame: pd.DataFrame, config: dict[str, Any], output: Path, profile: str) -> dict[str, Any]:
    optimizer = config["optimizer"]
    tolerance = 5e-7
    positions_root = output / "historical_test" / profile / "months"
    max_sic_net = 0.0
    max_sic_gross = 0.0
    for path in positions_root.glob("*.parquet"):
        positions = pd.read_parquet(path)
        active = positions.loc[positions["weight"].ne(0.0)]
        for _, group in active.groupby("sic2_industry_category", observed=True):
            weights = group["weight"].to_numpy(float)
            max_sic_net = max(max_sic_net, abs(float(weights.sum())))
            max_sic_gross = max(max_sic_gross, float(np.abs(weights).sum()))
    other_factors = [f"factor_exposure_{name}" for name in FACTOR_NAMES[1:]]
    breaches = {
        "maximum_gross": float(frame["gross"].max() - float(optimizer["maximum_gross"])),
        "minimum_long_gross": float(float(optimizer["minimum_long_gross"]) - frame["long_gross"].min()),
        "minimum_short_gross": float(float(optimizer["minimum_short_gross"]) - frame["short_gross"].min()),
        "minimum_net": float(float(optimizer["minimum_net"]) - frame["net"].min()),
        "maximum_net": float(frame["net"].max() - float(optimizer["maximum_net"])),
        "position_long": float(frame["maximum_weight"].max() - float(optimizer["maximum_position_weight"])),
        "position_short": float(-frame["minimum_weight"].min() - float(optimizer["maximum_position_weight"])),
        "turnover": float(frame["turnover"].max() - float(optimizer["maximum_one_way_turnover"])),
        "adv": float(frame["maximum_adv_participation"].max() - float(optimizer["maximum_one_way_adv_participation"])),
        "market_beta": float(frame["factor_exposure_mktrf"].abs().max() - float(optimizer["market_beta_absolute_limit"])),
        "other_factor": float(frame[other_factors].abs().to_numpy().max() - float(optimizer["other_factor_absolute_limit"])),
        "sic2_net": float(max_sic_net - float(optimizer["sic2_industry_net_absolute_limit"])),
        "sic2_gross": float(max_sic_gross - float(optimizer["sic2_industry_gross_limit"])),
        "forecast_volatility": float(frame["forecast_volatility"].max() - float(optimizer["primary_forecast_volatility"])),
    }
    return {
        "breach_residuals": breaches,
        "maximum_positive_breach": max(max(breaches.values()), 0.0),
        "passes_practical_tolerance": max(breaches.values()) <= tolerance,
        "practical_tolerance": tolerance,
    }


def run(config_path: Path, output_root: Path, *, smoke: bool = False) -> Path:
    challenge, frozen_config, paths, config_sha, input_identity = _verify(config_path)
    code_identity = _code_identity()
    output = output_root / f"{config_sha[:12]}-{input_identity[:12]}-{code_identity[:12]}"
    if smoke:
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": config_sha,
        "input_identity": input_identity,
        "code_identity": code_identity,
    }
    base._atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_1_final_optimizer_challenge_run_v1",
            "status": "RUNNING",
            **identity,
            "profiles": [profile["id"] for profile in challenge["profiles"]],
        },
    )
    loader_config = copy.deepcopy(frozen_config)
    loader_config["evidence"]["portfolio_start"] = challenge["evidence"]["validation_start"]
    loader_config["evidence"]["portfolio_end"] = challenge["evidence"]["historical_test_end"]
    inputs = base.load_inputs(loader_config, paths)
    validation_months = pd.period_range(
        challenge["evidence"]["validation_start"],
        challenge["evidence"]["validation_end"],
        freq="M",
    ).astype(str).tolist()
    test_months = pd.period_range(
        challenge["evidence"]["historical_test_start"],
        challenge["evidence"]["historical_test_end"],
        freq="M",
    ).astype(str).tolist()
    if smoke:
        validation_months = validation_months[:1]
        test_months = test_months[:1]
    month_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    validation_frames: list[pd.DataFrame] = []
    validation_summaries: list[dict[str, Any]] = []
    for profile in challenge["profiles"]:
        frame = _run_profile(
            output=output,
            phase="validation",
            months=validation_months,
            profile=profile,
            identity=identity,
            inputs=inputs,
            loader_config=loader_config,
            month_cache=month_cache,
        )
        validation_frames.append(frame)
        validation_summaries.append(_aggregate(frame))
        print(
            f"PROFILE {profile['id']} validation_sharpe={validation_summaries[-1]['net_sharpe_primary']:.6f}",
            flush=True,
        )
    validation_monthly = pd.concat(validation_frames, ignore_index=True)
    validation_summary = pd.DataFrame(validation_summaries)
    profile_order = {profile["id"]: index for index, profile in enumerate(challenge["profiles"])}
    validation_summary["profile_order"] = validation_summary["profile"].map(profile_order)
    validation_summary = validation_summary.sort_values(
        ["net_sharpe_primary", "profile_order"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected_id = str(validation_summary.iloc[0]["profile"])
    selected = next(profile for profile in challenge["profiles"] if profile["id"] == selected_id)
    base._atomic_parquet(output / "validation_monthly.parquet", validation_monthly)
    base._atomic_parquet(output / "validation_summary.parquet", validation_summary)
    base._atomic_json(output / "selected_profile.json", selected)
    print(f"SELECTED {selected_id}", flush=True)

    test = _run_profile(
        output=output,
        phase="historical_test",
        months=test_months,
        profile=selected,
        identity=identity,
        inputs=inputs,
        loader_config=loader_config,
        month_cache=month_cache,
    )
    test_summary = _aggregate(test)
    base._atomic_parquet(output / "historical_test_monthly.parquet", test)
    base._atomic_parquet(output / "historical_test_summary.parquet", pd.DataFrame([test_summary]))
    if smoke:
        base._atomic_json(
            output / "run_manifest.json",
            {
                "schema": "research_v6_1_final_optimizer_challenge_run_v1",
                "status": "SMOKE_COMPLETE",
                **identity,
                "selected_profile": selected_id,
            },
        )
        return output

    control_root = ROOT / str(challenge["frozen_control"]["path"])
    control = pd.read_parquet(control_root / "monthly_results.parquet")
    control = control.loc[
        control["variant"].eq(challenge["frozen_control"]["variant"])
    ].sort_values("holding_month", kind="mergesort").reset_index(drop=True)
    inference = _paired_inference(test, control, challenge["inference"])
    audit = _constraint_audit(test, loader_config, output, selected_id)
    control_without_best_two = float(
        np.sort(control["net_excess_return_primary"].to_numpy(float))[:-2].mean() * 12.0
    )
    selected_validation = validation_summary.loc[validation_summary["profile"].eq(selected_id)].iloc[0]
    control_validation = validation_summary.loc[validation_summary["profile"].eq("control")].iloc[0]
    statistical_pass = bool(
        inference["sharpe_difference"] > 0.0
        and inference["bootstrap_sharpe_difference_ci_95"][0] > 0.0
        and inference["paired_hac_t_mean_net_return_difference"] > 1.96
    )
    economic_pass = bool(
        selected_id != "control"
        and float(selected_validation["net_sharpe_primary"]) > float(control_validation["net_sharpe_primary"])
        and float(test_summary["net_sharpe_stress"]) > 0.0
        and bool(audit["passes_practical_tolerance"])
        and float(test_summary["annualized_net_excess_without_best_two_months"])
        >= control_without_best_two
    )
    promoted = statistical_pass and economic_pass
    decision = (
        challenge["decision"]["pass"] if promoted else challenge["decision"]["fail"]
    )
    result = {
        "schema": "research_v6_1_final_optimizer_challenge_result_v1",
        "status": "COMPLETE",
        "selected_profile": selected,
        "validation_selected_sharpe": float(selected_validation["net_sharpe_primary"]),
        "validation_control_sharpe": float(control_validation["net_sharpe_primary"]),
        "historical_test_summary": test_summary,
        "frozen_control_sharpe": float(challenge["frozen_control"]["net_sharpe"]),
        "paired_inference": inference,
        "constraint_audit": audit,
        "control_annualized_net_without_best_two_months": control_without_best_two,
        "statistical_superiority_pass": statistical_pass,
        "economic_mechanism_pass": economic_pass,
        "decision": decision,
        "post_diagnostic_historical_evidence": True,
    }
    base._atomic_json(output / "result.json", result)
    base._atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_1_final_optimizer_challenge_run_v1",
            "status": "COMPLETE",
            **identity,
            "selected_profile": selected_id,
            "validation_profile_months": len(challenge["profiles"]) * len(validation_months),
            "historical_test_months": len(test_months),
            "decision": decision,
        },
    )
    print(json.dumps(result, indent=2, default=str), flush=True)
    print(f"COMPLETE {output}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        run(args.config.resolve(), args.output_root.resolve(), smoke=args.smoke)
    except Exception as exc:
        error_root = args.output_root.resolve() / "latest_error"
        error_root.mkdir(parents=True, exist_ok=True)
        base._atomic_json(
            error_root / "error.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
