#!/usr/bin/env python3
"""Run the frozen, checkpointed Research V6.1 portfolio study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from portfolio_bot.allocation.costs import numpy_total_cost  # noqa: E402
from portfolio_bot.research_v6_1.portfolio import (  # noqa: E402
    FACTOR_NAMES,
    PortfolioSolution,
    apply_stateful_trade_limits,
    causal_monotonic_calibration,
    deterministic_rank_weights,
    factor_covariance_matrix,
    percentile_rank,
    portfolio_risk,
    solve_expected_net_return_portfolio,
)


EXTERNAL_ROOT = Path("/Users/agraemirolmez/Desktop/Thesis")
DEFAULT_CONFIG = ROOT / "configs/research_v6_1/portfolio_step_02.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_1/portfolio_step_02"
KEYS = ["holding_month", "signal_date", "permno", "permco"]
ECONOMIC_COLUMNS = {
    "capm": "alpha_capm",
    "ff5_momentum": "alpha_ff5_momentum",
    "long_net": "long_net_opportunity",
    "short_net": "short_net_opportunity",
}


@dataclass
class PreparedInputs:
    forecasts: pd.DataFrame
    risk: pd.DataFrame
    covariance: pd.DataFrame
    factors: pd.DataFrame
    benchmarks: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


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


def _resolve_source(item: dict[str, Any]) -> Path:
    root = ROOT if item["root"] == "workspace" else EXTERNAL_ROOT
    return root / str(item["path"])


def verify_contract(config_path: Path) -> tuple[dict[str, Any], str, str, dict[str, Path]]:
    config = json.loads(config_path.read_text())
    if config.get("status") != "FROZEN_READY_TO_RUN":
        raise RuntimeError("V6.1 portfolio contract is not frozen and ready")
    selector_root = ROOT / str(config["selector_run"]["path"])
    selector_manifest = selector_root / "run_manifest.json"
    if _sha256(selector_manifest) != config["selector_run"]["manifest_sha256"]:
        raise RuntimeError("frozen V6.1 selector manifest identity changed")
    if json.loads(selector_manifest.read_text()).get("status") != config["selector_run"]["required_status"]:
        raise RuntimeError("frozen V6.1 selector run is not complete")
    paths: dict[str, Path] = {}
    for item in config["source_bindings"]:
        path = _resolve_source(item)
        if not path.is_file():
            raise FileNotFoundError(f"missing bound portfolio source: {path}")
        actual = _sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"bound source identity changed: {item['id']}")
        paths[str(item["id"])] = path
    config_sha = _sha256(config_path)
    input_identity = hashlib.sha256(_json_bytes({item["id"]: item["sha256"] for item in config["source_bindings"]})).hexdigest()
    return config, config_sha, input_identity, paths


def _code_identity() -> str:
    files = [
        Path(__file__).resolve(),
        ROOT / "src/portfolio_bot/research_v6_1/portfolio.py",
        ROOT / "src/portfolio_bot/allocation/costs.py",
        ROOT / "src/portfolio_bot/risk_cost/cost_math.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def load_inputs(config: dict[str, Any], paths: dict[str, Path]) -> PreparedInputs:
    forecast_parts: list[pd.DataFrame] = []
    for target in ("capm", "ff5_momentum", "long_net", "short_net"):
        signal_column = str(config["selector_run"]["signals"][target]["column"])
        economic = ECONOMIC_COLUMNS[target]
        frame = pd.read_parquet(
            paths[f"{target}_predictions"],
            columns=[*KEYS, signal_column, economic, "period"],
        ).rename(
            columns={signal_column: f"score_{target}", economic: f"actual_{target}"}
        )
        frame["holding_month"] = frame["holding_month"].astype(str)
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise")
        forecast_parts.append(frame)
    forecasts = forecast_parts[0]
    for frame in forecast_parts[1:]:
        forecasts = forecasts.merge(
            frame.drop(columns="period"), on=KEYS, how="inner", validate="one_to_one"
        )
    if forecasts.duplicated(["holding_month", "permno"]).any():
        raise RuntimeError("frozen selector predictions contain duplicate stock-months")

    availability = pd.read_parquet(
        paths["v6_target_rows"], columns=[*KEYS, "label_available_date"]
    )
    availability["holding_month"] = availability["holding_month"].astype(str)
    availability["signal_date"] = pd.to_datetime(availability["signal_date"], errors="raise")
    availability["label_available_date"] = pd.to_datetime(
        availability["label_available_date"], errors="coerce"
    )
    forecasts = forecasts.merge(availability, on=KEYS, how="left", validate="one_to_one")

    executable = pd.read_parquet(
        paths["market_inputs"],
        columns=[*KEYS, "entry_date", "exit_date", "entry_close_status"],
        filters=[
            ("holding_month", ">=", config["evidence"]["portfolio_start"]),
            ("holding_month", "<=", config["evidence"]["portfolio_end"]),
        ],
    )
    executable["holding_month"] = executable["holding_month"].astype(str)
    executable["signal_date"] = pd.to_datetime(executable["signal_date"], errors="raise")
    executable["entry_date"] = pd.to_datetime(executable["entry_date"], errors="raise")
    executable["exit_date"] = pd.to_datetime(executable["exit_date"], errors="raise")
    executable = executable.loc[
        executable["entry_close_status"].astype(str).eq("PRESENT")
    ].drop(columns="entry_close_status")
    if executable.duplicated(KEYS).any():
        raise RuntimeError("market inputs contain duplicate executable entry rows")

    outcomes = pd.read_parquet(paths["outcomes"], columns=["holding_month", "permno", "total_return"])
    outcomes["holding_month"] = outcomes["holding_month"].astype(str)
    accounting = pd.read_parquet(
        paths["market_accounting"],
        columns=[
            "holding_month",
            "permno",
            "authoritative_total_return",
            "return_history_usable",
        ],
    )
    accounting["holding_month"] = accounting["holding_month"].astype(str)
    accounting["_return"] = pd.to_numeric(
        accounting["authoritative_total_return"], errors="coerce"
    )
    accounting = accounting.loc[
        accounting["return_history_usable"].fillna(False).astype(bool)
        & np.isfinite(accounting["_return"].to_numpy(float))
    ]
    accounting_outcomes = (
        accounting.groupby(["holding_month", "permno"], observed=True, sort=True)["_return"]
        .agg(lambda values: float(np.prod(1.0 + values.to_numpy(float)) - 1.0))
        .rename("accounting_total_return")
        .reset_index()
    )
    delistings = pd.read_parquet(
        paths["delisting_events"],
        columns=[
            "permno",
            "delisting_date",
            "effective_delisting_return",
            "assigned_exactly_once",
        ],
    )
    delistings["delisting_date"] = pd.to_datetime(
        delistings["delisting_date"], errors="raise"
    )
    delistings["terminal_total_return"] = pd.to_numeric(
        delistings["effective_delisting_return"], errors="coerce"
    )
    terminal_outcomes = executable.merge(
        delistings, on="permno", how="inner", validate="many_to_many"
    )
    terminal_outcomes = terminal_outcomes.loc[
        terminal_outcomes["assigned_exactly_once"].fillna(False).astype(bool)
        & np.isfinite(terminal_outcomes["terminal_total_return"].to_numpy(float))
        & terminal_outcomes["delisting_date"].ge(terminal_outcomes["entry_date"])
        & terminal_outcomes["delisting_date"].le(terminal_outcomes["exit_date"])
    ]
    if terminal_outcomes.duplicated(KEYS).any():
        raise RuntimeError("multiple terminal-event outcomes map to one executable stock-month")
    terminal_outcomes = terminal_outcomes[[*KEYS, "terminal_total_return"]]
    factors = pd.read_parquet(paths["factors_and_cash"])
    factors["holding_month"] = factors["holding_month"].astype(str)
    forecasts = forecasts.merge(outcomes, on=["holding_month", "permno"], how="left", validate="one_to_one")
    forecasts = forecasts.merge(
        accounting_outcomes,
        on=["holding_month", "permno"],
        how="left",
        validate="one_to_one",
    )
    forecasts = forecasts.merge(
        terminal_outcomes, on=KEYS, how="left", validate="one_to_one"
    )
    compact_return = pd.to_numeric(forecasts["total_return"], errors="coerce")
    accounting_return = pd.to_numeric(
        forecasts["accounting_total_return"], errors="coerce"
    )
    terminal_return = pd.to_numeric(
        forecasts["terminal_total_return"], errors="coerce"
    )
    forecasts["return_source"] = np.select(
        [compact_return.notna(), accounting_return.notna(), terminal_return.notna()],
        ["primary_outcomes", "market_accounting", "terminal_event"],
        default="MISSING",
    )
    forecasts["total_return"] = compact_return.fillna(accounting_return).fillna(
        terminal_return
    )
    forecasts = forecasts.merge(
        factors[["holding_month", "rf"]], on="holding_month", how="left", validate="many_to_one"
    )
    forecasts["stock_excess_return"] = (
        pd.to_numeric(forecasts["total_return"], errors="coerce")
        - pd.to_numeric(forecasts["rf"], errors="coerce")
    )
    forecasts["payoff_long"] = forecasts["stock_excess_return"].where(
        pd.to_numeric(forecasts["actual_long_net"], errors="coerce").notna()
    )
    forecasts["payoff_short"] = (-forecasts["stock_excess_return"]).where(
        pd.to_numeric(forecasts["actual_short_net"], errors="coerce").notna()
    )
    for target in ("capm", "ff5_momentum", "long_net", "short_net"):
        score = f"score_{target}"
        forecasts[f"rank_{target}"] = forecasts.groupby("holding_month", observed=True, sort=False)[score].transform(
            lambda values: percentile_rank(values)
        )

    start, end = config["evidence"]["portfolio_start"], config["evidence"]["portfolio_end"]
    risk_columns = [
        *KEYS,
        "long_eligible",
        "short_eligible",
        "risk_status",
        *(f"beta_risk_{name}" for name in FACTOR_NAMES),
        "specific_variance_monthly",
        "full_spread_20d",
        "adv20_usd",
        "impact_volatility_daily",
        "primary_borrow_rate_annual",
        "stress_borrow_rate_annual",
        "new_long_trade_allowed",
        "new_short_trade_allowed",
        "cost_input_status",
    ]
    risk = pd.read_parquet(
        paths["risk_cost"],
        columns=risk_columns,
        filters=[("holding_month", ">=", start), ("holding_month", "<=", end)],
    )
    risk["holding_month"] = risk["holding_month"].astype(str)
    risk["signal_date"] = pd.to_datetime(risk["signal_date"], errors="raise")
    controls = pd.read_parquet(
        paths["predictor_controls"],
        columns=[*KEYS, "sic2_industry_category"],
        filters=[("holding_month", ">=", start), ("holding_month", "<=", end)],
    )
    controls["holding_month"] = controls["holding_month"].astype(str)
    controls["signal_date"] = pd.to_datetime(controls["signal_date"], errors="raise")
    risk = risk.merge(controls, on=KEYS, how="left", validate="one_to_one")
    # Entry-close availability is an execution feasibility requirement, not a
    # future-return screen.  Securities without an authenticated entry price
    # cannot be placed in a historical portfolio.
    risk = risk.merge(executable[KEYS], on=KEYS, how="inner", validate="one_to_one")
    covariance = pd.read_parquet(
        paths["factor_covariance"],
        filters=[("holding_month", ">=", start), ("holding_month", "<=", end)],
    )
    covariance["holding_month"] = covariance["holding_month"].astype(str)
    benchmarks = pd.read_parquet(paths["benchmarks"])
    benchmarks["holding_month"] = benchmarks["holding_month"].astype(str)
    return PreparedInputs(
        forecasts=forecasts.sort_values(["holding_month", "permno"], kind="mergesort").reset_index(drop=True),
        risk=risk.sort_values(["holding_month", "permno"], kind="mergesort").reset_index(drop=True),
        covariance=covariance,
        factors=factors,
        benchmarks=benchmarks,
    )


def _calibrate_month(
    inputs: PreparedInputs, config: dict[str, Any], month: str
) -> pd.DataFrame:
    current = inputs.forecasts.loc[inputs.forecasts["holding_month"].eq(month)].copy()
    if current.empty:
        raise RuntimeError(f"no frozen predictions for portfolio month {month}")
    signal_date = pd.Timestamp(current["signal_date"].iloc[0])
    history = inputs.forecasts.loc[inputs.forecasts["holding_month"].lt(month)].copy()
    calibration = config["calibration"]
    specifications = {
        "capm": ("rank_capm", "actual_capm"),
        "ff5_momentum": ("rank_ff5_momentum", "actual_ff5_momentum"),
        "long": ("rank_long_net", "payoff_long"),
        "short": ("rank_short_net", "payoff_short"),
    }
    for name, (score_column, outcome_column) in specifications.items():
        result = causal_monotonic_calibration(
            history,
            current[score_column].to_numpy(float),
            score_column=score_column,
            outcome_column=outcome_column,
            month_column="holding_month",
            availability_column="label_available_date",
            signal_date=signal_date,
            bin_edges=calibration["bin_edges"],
            minimum_history_months=int(calibration["minimum_history_months"]),
            shrinkage_prior_months=float(calibration["shrinkage_prior_months"]),
        )
        current[f"expected_{name}"] = result.values
        current[f"calibration_months_{name}"] = result.history_months
    return current


def _policy_masks(frame: pd.DataFrame, config: dict[str, Any], policy: str) -> tuple[np.ndarray, np.ndarray]:
    threshold = 1.0 - float(config["formation"]["candidate_fraction_each_side"])
    long_mask = (
        frame["long_eligible"].fillna(False).to_numpy(bool)
        & frame["new_long_trade_allowed"].fillna(False).to_numpy(bool)
        & (frame["rank_long_net"].to_numpy(float) >= threshold)
    )
    short_mask = (
        frame["short_eligible"].fillna(False).to_numpy(bool)
        & frame["new_short_trade_allowed"].fillna(False).to_numpy(bool)
        & (frame["rank_short_net"].to_numpy(float) >= threshold)
    )
    if policy in {"return_plus_capm", "return_plus_both"}:
        long_mask &= frame["expected_capm"].to_numpy(float) > 0.0
        short_mask &= frame["expected_capm"].to_numpy(float) < 0.0
    if policy in {"return_plus_ff5_momentum", "return_plus_both"}:
        long_mask &= frame["expected_ff5_momentum"].to_numpy(float) > 0.0
        short_mask &= frame["expected_ff5_momentum"].to_numpy(float) < 0.0

    overlap = long_mask & short_mask
    if overlap.any():
        long_edge = frame["expected_long"].to_numpy(float)
        short_edge = frame["expected_short"].to_numpy(float)
        long_rank = frame["rank_long_net"].to_numpy(float)
        short_rank = frame["rank_short_net"].to_numpy(float)
        prefer_long = (long_edge > short_edge) | (
            np.isclose(long_edge, short_edge, atol=1e-14, rtol=0.0)
            & (long_rank >= short_rank)
        )
        short_mask[overlap & prefer_long] = False
        long_mask[overlap & ~prefer_long] = False
    return long_mask, short_mask


def _month_frame(inputs: PreparedInputs, config: dict[str, Any], month: str) -> tuple[pd.DataFrame, np.ndarray]:
    calibrated = _calibrate_month(inputs, config, month)
    risk = inputs.risk.loc[inputs.risk["holding_month"].eq(month)].copy()
    frame = calibrated.merge(risk, on=KEYS, how="inner", validate="one_to_one")
    numeric = [
        *(f"beta_risk_{name}" for name in FACTOR_NAMES),
        "specific_variance_monthly",
        "full_spread_20d",
        "adv20_usd",
        "impact_volatility_daily",
        "primary_borrow_rate_annual",
        "stress_borrow_rate_annual",
    ]
    good = np.ones(len(frame), dtype=bool)
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        good &= np.isfinite(frame[column].to_numpy(float))
    good &= frame["risk_status"].astype(str).eq("PRESENT").to_numpy(bool)
    good &= frame["cost_input_status"].astype(str).eq("PRESENT").to_numpy(bool)
    frame = frame.loc[good].sort_values("permno", kind="mergesort").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"no risk/cost-ready portfolio rows for {month}")
    frame["sic2_industry_category"] = frame["sic2_industry_category"].fillna("UNKNOWN_SIC2").astype(str)
    covariance = inputs.covariance.loc[inputs.covariance["holding_month"].eq(month)]
    return frame, factor_covariance_matrix(covariance)


def _variant_definitions(config: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    methods = ["equal_weight", *config["rank_weighting"]["methods"]]
    for policy in config["formation"]["alpha_policies"]:
        for method in methods:
            variants.append({"id": f"{policy}__{method}", "policy": policy, "kind": "rank", "method": method, "volatility": 0.10})
        variants.append({"id": f"{policy}__optimized_10pct_vol", "policy": policy, "kind": "optimized", "method": "optimized", "volatility": 0.10})
    primary = str(config["formation"]["primary_policy"])
    for volatility in config["optimizer"]["robustness_forecast_volatility"]:
        percentage = int(round(float(volatility) * 100))
        variants.append({"id": f"{primary}__optimized_{percentage}pct_vol", "policy": primary, "kind": "optimized", "method": "optimized", "volatility": float(volatility)})
    return variants


def _aligned_previous(frame: pd.DataFrame, previous: dict[int, float]) -> np.ndarray:
    return frame["permno"].map(previous).fillna(0.0).to_numpy(float)


def _cost_breakdowns(
    weights: np.ndarray,
    previous: np.ndarray,
    frame: pd.DataFrame,
    config: dict[str, Any],
    nav_dollars: float,
) -> tuple[Any, Any]:
    primary = config["costs"]["primary"]
    stress = config["costs"]["standard_stress"]
    common = dict(
        weights=weights,
        previous_weights=previous,
        full_spread=frame["full_spread_20d"].to_numpy(float),
        adv20_usd=frame["adv20_usd"].to_numpy(float),
        impact_volatility_daily=frame["impact_volatility_daily"].to_numpy(float),
        nav_dollars=nav_dollars,
    )
    primary_result = numpy_total_cost(
        **common,
        borrow_rate_annual=frame[str(primary["borrow_column"])].to_numpy(float),
        impact_coefficient=float(primary["impact_coefficient"]),
        spread_multiplier=float(primary["spread_multiplier"]),
        financing_rate_annual=float(primary["financing_rate_annual"]),
    )
    stress_result = numpy_total_cost(
        **common,
        borrow_rate_annual=frame[str(stress["borrow_column"])].to_numpy(float),
        impact_coefficient=float(stress["impact_coefficient"]),
        spread_multiplier=float(stress["spread_multiplier"]),
        financing_rate_annual=float(stress["financing_rate_annual"]),
    )
    return primary_result, stress_result


def _solve_variant_month(
    frame: pd.DataFrame,
    covariance_root: np.ndarray,
    config: dict[str, Any],
    variant: dict[str, Any],
    previous_map: dict[int, float],
    nav_dollars: float,
) -> tuple[PortfolioSolution, pd.DataFrame, dict[str, Any]]:
    long_mask, short_mask = _policy_masks(frame, config, str(variant["policy"]))
    previous = _aligned_previous(frame, previous_map)
    opposing = ((previous > 0.0) & short_mask) | ((previous < 0.0) & long_mask)
    long_mask[opposing] = False
    short_mask[opposing] = False
    reduction_only = (previous != 0.0) & ~(long_mask | short_mask)
    betas = frame[[f"beta_risk_{name}" for name in FACTOR_NAMES]].to_numpy(float)
    specific = frame["specific_variance_monthly"].to_numpy(float)
    optimizer = config["optimizer"]

    if variant["kind"] == "optimized":
        mu = np.zeros(len(frame), dtype=np.float64)
        mu[long_mask] = frame.loc[long_mask, "expected_long"].to_numpy(float)
        mu[short_mask] = -frame.loc[short_mask, "expected_short"].to_numpy(float)
        primary = config["costs"]["primary"]
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
            forecast_volatility_target=float(variant["volatility"]),
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
    else:
        baseline = config["rank_weighting"]["baseline_book"]
        target = deterministic_rank_weights(
            long_mask=long_mask,
            short_mask=short_mask,
            long_rank=frame["rank_long_net"].to_numpy(float),
            short_rank=frame["rank_short_net"].to_numpy(float),
            method=str(variant["method"]),
            factor_betas=betas,
            factor_covariance_root=covariance_root,
            specific_variance_monthly=specific,
            adv20_usd=frame["adv20_usd"].to_numpy(float),
            nav_dollars=nav_dollars,
            forecast_volatility_target=float(baseline["forecast_volatility_target"]),
            maximum_gross=float(baseline["maximum_gross"]),
            maximum_position_weight=float(baseline["maximum_position_weight"]),
            maximum_one_way_adv_participation=float(baseline["maximum_one_way_adv_participation"]),
        )
        target[reduction_only] = 0.0
        weights = apply_stateful_trade_limits(
            target,
            previous,
            adv20_usd=frame["adv20_usd"].to_numpy(float),
            nav_dollars=nav_dollars,
            maximum_one_way_adv_participation=float(optimizer["maximum_one_way_adv_participation"]),
            maximum_one_way_turnover=float(optimizer["maximum_one_way_turnover"]),
        )
        primary_cost, _ = _cost_breakdowns(weights, previous, frame, config, nav_dollars)
        risk, exposures = portfolio_risk(weights, betas, covariance_root, specific)
        long_value = float(np.maximum(weights, 0.0).sum())
        short_value = float(np.maximum(-weights, 0.0).sum())
        solution = PortfolioSolution(
            weights=weights,
            status="DETERMINISTIC",
            objective_value=float("nan"),
            forecast_volatility=risk,
            long_gross=long_value,
            short_gross=short_value,
            gross=long_value + short_value,
            net=float(weights.sum()),
            turnover=0.5 * float(np.abs(weights - previous).sum()),
            factor_exposures=exposures,
            transaction_cost=primary_cost.transaction,
            borrow_cost=primary_cost.borrow,
            financing_cost=primary_cost.financing,
            total_cost=primary_cost.total,
        )

    primary_cost, stress_cost = _cost_breakdowns(weights, previous, frame, config, nav_dollars)
    excess = frame["stock_excess_return"].to_numpy(float)
    if np.any((weights != 0.0) & ~np.isfinite(excess)):
        raise RuntimeError("a nonzero portfolio holding lacks an authenticated outcome")
    excess = np.where(np.isfinite(excess), excess, 0.0)
    gross_excess = float(weights @ excess)
    rf = float(frame["rf"].iloc[0])
    gross_return = rf + gross_excess
    net_return = gross_return - primary_cost.total
    stress_return = gross_return - stress_cost.total
    denominator = 1.0 + net_return
    if denominator <= 0.0:
        raise RuntimeError("portfolio NAV became nonpositive")
    end_drift = weights * (1.0 + excess + rf) / denominator

    positions = frame.loc[:, [*KEYS, "sic2_industry_category", "return_source"]].copy()
    positions["weight"] = weights
    positions["previous_weight"] = previous
    positions["end_drift_weight"] = end_drift
    positions["stock_excess_return"] = excess
    positions["long_candidate"] = long_mask
    positions["short_candidate"] = short_mask
    positions["expected_stock_excess_return"] = np.where(
        long_mask,
        frame["expected_long"].to_numpy(float),
        np.where(short_mask, -frame["expected_short"].to_numpy(float), 0.0),
    )
    positions = positions.loc[
        (positions["weight"] != 0.0)
        | (positions["previous_weight"] != 0.0)
        | positions["long_candidate"]
        | positions["short_candidate"]
    ].reset_index(drop=True)

    summary = {
        "holding_month": str(frame["holding_month"].iloc[0]),
        "signal_date": str(pd.Timestamp(frame["signal_date"].iloc[0]).date()),
        "variant": variant["id"],
        "policy": variant["policy"],
        "weighting": variant["method"],
        "volatility_target": float(variant["volatility"]),
        "solver_status": solution.status,
        "nav_start": float(nav_dollars),
        "nav_end_primary": float(nav_dollars * (1.0 + net_return)),
        "rf": rf,
        "gross_return": gross_return,
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
        "active_long_holdings": int((weights > 1e-8).sum()),
        "active_short_holdings": int((weights < -1e-8).sum()),
        "eligible_long_candidates": int(long_mask.sum()),
        "eligible_short_candidates": int(short_mask.sum()),
        "maximum_weight": float(np.max(weights, initial=0.0)),
        "minimum_weight": float(np.min(weights, initial=0.0)),
        **{f"factor_exposure_{name}": float(solution.factor_exposures[i]) for i, name in enumerate(FACTOR_NAMES)},
    }
    return solution, positions, summary


def _checkpoint_paths(output: Path, variant_id: str, month: str) -> tuple[Path, Path]:
    root = output / "variants" / variant_id / "months"
    return root / f"{month}.parquet", root / f"{month}.json"


def _load_resume_state(
    output: Path,
    variant_id: str,
    months: list[str],
    identity: dict[str, str],
) -> tuple[int, dict[int, float], float, list[dict[str, Any]]]:
    previous: dict[int, float] = {}
    nav = 100_000_000.0
    summaries: list[dict[str, Any]] = []
    completed = 0
    for month in months:
        parquet_path, receipt_path = _checkpoint_paths(output, variant_id, month)
        if not parquet_path.is_file() or not receipt_path.is_file():
            break
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"checkpoint identity mismatch for {variant_id} {month}")
        positions = pd.read_parquet(parquet_path, columns=["permno", "end_drift_weight"])
        previous = {
            int(row.permno): float(row.end_drift_weight)
            for row in positions.itertuples(index=False)
            if abs(float(row.end_drift_weight)) > 1e-12
        }
        summary = dict(receipt["summary"])
        nav = float(summary["nav_end_primary"])
        summaries.append(summary)
        completed += 1
    return completed, previous, nav, summaries


def _max_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return float(np.min(wealth / peak - 1.0, initial=0.0))


def _aggregate_variant(frame: pd.DataFrame) -> dict[str, Any]:
    net = frame["net_excess_return_primary"].to_numpy(float)
    stress = frame["net_excess_return_stress"].to_numpy(float)
    total = frame["net_return_primary"].to_numpy(float)
    volatility = float(np.std(net, ddof=1) * np.sqrt(12.0)) if len(net) > 1 else float("nan")
    stress_vol = float(np.std(stress, ddof=1) * np.sqrt(12.0)) if len(stress) > 1 else float("nan")
    order = np.argsort(net)[::-1]
    without_best_two = np.delete(net, order[: min(2, len(net))])
    return {
        "variant": str(frame["variant"].iloc[0]),
        "policy": str(frame["policy"].iloc[0]),
        "weighting": str(frame["weighting"].iloc[0]),
        "volatility_target": float(frame["volatility_target"].iloc[0]),
        "months": int(len(frame)),
        "annualized_gross_excess_return": float(frame["gross_excess_return"].mean() * 12.0),
        "annualized_net_excess_return_primary": float(net.mean() * 12.0),
        "annualized_net_excess_return_stress": float(stress.mean() * 12.0),
        "annualized_realized_volatility_primary": volatility,
        "annualized_realized_volatility_stress": stress_vol,
        "net_sharpe_primary": float(net.mean() / np.std(net, ddof=1) * np.sqrt(12.0)) if volatility > 0.0 else float("nan"),
        "net_sharpe_stress": float(stress.mean() / np.std(stress, ddof=1) * np.sqrt(12.0)) if stress_vol > 0.0 else float("nan"),
        "maximum_drawdown_primary": _max_drawdown(total),
        "mean_one_way_turnover": float(frame["turnover"].mean()),
        "annualized_primary_cost": float(frame["total_cost_primary"].mean() * 12.0),
        "annualized_stress_cost": float(frame["total_cost_stress"].mean() * 12.0),
        "mean_gross": float(frame["gross"].mean()),
        "mean_net": float(frame["net"].mean()),
        "mean_long_holdings": float(frame["active_long_holdings"].mean()),
        "mean_short_holdings": float(frame["active_short_holdings"].mean()),
        "positive_net_month_fraction": float((net > 0.0).mean()),
        "annualized_net_excess_without_best_two_months": float(without_best_two.mean() * 12.0) if len(without_best_two) else float("nan"),
    }


def run(
    config_path: Path,
    output_root: Path,
    *,
    check_only: bool,
    selected_variants: set[str] | None,
    smoke_month: str | None,
    max_months: int | None,
) -> Path | None:
    config, config_sha, input_identity, paths = verify_contract(config_path)
    code_identity = _code_identity()
    variants = _variant_definitions(config)
    if selected_variants:
        variants = [variant for variant in variants if variant["id"] in selected_variants]
        missing = selected_variants - {variant["id"] for variant in variants}
        if missing:
            raise ValueError(f"unknown requested variants: {sorted(missing)}")
    if check_only:
        print(
            f"READY: {len(variants)} portfolio paths; selector frozen; sources authenticated; "
            f"objective=max expected net return at fixed risk; config={config_sha[:12]}"
        )
        return None

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
            "schema": "research_v6_1_portfolio_run_manifest_v1",
            "status": "RUNNING",
            **identity,
            "config": str(config_path.resolve()),
            "variants": [variant["id"] for variant in variants],
            "smoke_month": smoke_month,
        },
    )
    inputs = load_inputs(config, paths)
    months = pd.period_range(
        config["evidence"]["portfolio_start"], config["evidence"]["portfolio_end"], freq="M"
    ).astype(str).tolist()
    if smoke_month:
        if smoke_month not in months:
            raise ValueError("smoke month lies outside the frozen portfolio window")
        months = [smoke_month]
    if max_months is not None:
        months = months[: int(max_months)]

    all_summaries: list[dict[str, Any]] = []
    month_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for variant in variants:
        completed, previous, nav, summaries = _load_resume_state(
            output, str(variant["id"]), months, identity
        )
        all_summaries.extend(summaries)
        for month in months[completed:]:
            print(f"START {variant['id']} {month}", flush=True)
            try:
                if month not in month_cache:
                    month_cache[month] = _month_frame(inputs, config, month)
                frame, covariance_root = month_cache[month]
                _, positions, summary = _solve_variant_month(
                    frame, covariance_root, config, variant, previous, nav
                )
                parquet_path, receipt_path = _checkpoint_paths(output, str(variant["id"]), month)
                _atomic_parquet(parquet_path, positions)
                receipt = {**identity, "variant": variant["id"], "holding_month": month, "summary": summary}
                _atomic_json(receipt_path, receipt)
                previous = {
                    int(row.permno): float(row.end_drift_weight)
                    for row in positions.itertuples(index=False)
                    if abs(float(row.end_drift_weight)) > 1e-12
                }
                nav = float(summary["nav_end_primary"])
                all_summaries.append(summary)
                print(
                    f"DONE {variant['id']} {month} net={summary['net_excess_return_primary']:.6f} "
                    f"gross={summary['gross']:.3f} vol={summary['forecast_volatility']:.3f}",
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

    monthly = pd.DataFrame(all_summaries).drop_duplicates(["variant", "holding_month"], keep="last")
    monthly = monthly.sort_values(["variant", "holding_month"], kind="mergesort").reset_index(drop=True)
    _atomic_parquet(output / "monthly_results.parquet", monthly)
    summary = pd.DataFrame(
        [_aggregate_variant(group) for _, group in monthly.groupby("variant", observed=True, sort=True)]
    ).sort_values("net_sharpe_primary", ascending=False, kind="mergesort")
    _atomic_parquet(output / "variant_summary.parquet", summary.reset_index(drop=True))
    _atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_1_portfolio_run_manifest_v1",
            "status": "SMOKE_COMPLETE" if smoke_month else "COMPLETE",
            **identity,
            "config": str(config_path.resolve()),
            "variants": [variant["id"] for variant in variants],
            "completed_variant_months": int(len(monthly)),
            "expected_variant_months": int(len(variants) * len(months)),
            "portfolio_constructed": True,
            "optimizer_run": any(variant["kind"] == "optimized" for variant in variants),
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
