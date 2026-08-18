#!/usr/bin/env python3
"""Run the V6.8 survivor-only intramonth re-optimization test.

The V6.7 warning dates, thresholds, monthly portfolio, and execution convention
are immutable inputs.  After a saved V6.7 trim/exit executes at the close, this
runner may refill only surviving positions that were already nonzero at the
monthly rebalance.  It never opens a new stock, reverses a side, re-enters an
exit, or enlarges a warned name.  Every completed replay month checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import cvxpy as cp
import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from portfolio_bot.allocation.costs import cvxpy_total_cost  # noqa: E402
from portfolio_bot.research_v6_1.portfolio import (  # noqa: E402
    FACTOR_NAMES,
    factor_covariance_matrix,
    portfolio_risk,
)
from scripts import run_v6_6_intramonth_interventions as base  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/research_v6_8/daily_reoptimization.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_8/daily_reoptimization"
KEYS = ("holding_month", "permno")
RISK_COLUMNS = (
    *tuple(f"beta_risk_{name}" for name in FACTOR_NAMES),
    "specific_variance_monthly",
)
COST_COLUMNS = (
    "full_spread_20d",
    "adv20_usd",
    "impact_volatility_daily",
    "primary_borrow_rate_annual",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _duck(path: Path) -> str:
    return str(path).replace("'", "''")


def _read_parquet(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"SELECT * FROM read_parquet('{_duck(path)}')").df()


def _months(start: str, end: str) -> list[str]:
    return pd.period_range(start, end, freq="M").astype(str).tolist()


def _verify(config_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config = json.loads(config_path.read_text())
    if config.get("status") != "AUTHORIZED_POST_DIAGNOSTIC_STUDY":
        raise RuntimeError("V6.8 daily re-optimization is not authorized")
    checks: list[tuple[Path, str, str]] = []
    for name, item in config["sources"].items():
        checks.append((_resolve(item["path"]), str(item["sha256"]), name))
    for path, expected, label in checks:
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(f"{label} identity changed: {path}")
    code_identity = hashlib.sha256()
    for path in (Path(__file__).resolve(), Path(base.__file__).resolve()):
        code_identity.update(path.name.encode())
        code_identity.update(bytes.fromhex(_sha256(path)))
    identity = {
        "config_sha256": _sha256(config_path),
        "code_identity": code_identity.hexdigest(),
        "prepared_position_days_sha256": str(
            config["sources"]["prepared_position_days"]["sha256"]
        ),
        "v6_7_result_sha256": str(config["sources"]["v6_7_result"]["sha256"]),
    }
    identity["identity_sha256"] = _canonical_sha(identity)
    return config, identity


def _output_path(output_root: Path, identity: Mapping[str, str]) -> Path:
    return output_root / (
        f"{identity['config_sha256'][:12]}-"
        f"{identity['prepared_position_days_sha256'][:12]}-"
        f"{identity['code_identity'][:12]}"
    )


def _load_inputs(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    days = _read_parquet(_resolve(config["sources"]["prepared_position_days"]["path"]))
    days["holding_month"] = days["holding_month"].astype(str)
    days["calendar_date"] = pd.to_datetime(days["calendar_date"], errors="raise")
    if days.duplicated(["holding_month", "calendar_date", "permno"]).any():
        raise RuntimeError("prepared position-day panel contains duplicate keys")
    start = config["chronology"]["initial_replay_start"]
    end = config["chronology"]["historical_test_end"]
    risk_path = _resolve(config["sources"]["risk_cost"]["path"])
    risk_columns = ", ".join(["CAST(holding_month AS VARCHAR) holding_month", "permno", *RISK_COLUMNS])
    risk = duckdb.sql(
        f"""
        SELECT {risk_columns}
        FROM read_parquet('{_duck(risk_path)}')
        WHERE holding_month BETWEEN '{start}' AND '{end}'
        """
    ).df()
    if risk.duplicated(list(KEYS)).any():
        raise RuntimeError("risk source contains duplicate stock-month keys")
    days = days.merge(risk, on=list(KEYS), how="left", validate="many_to_one")
    if days[list(RISK_COLUMNS)].isna().any().any():
        missing = days.loc[
            days[list(RISK_COLUMNS)].isna().any(axis=1),
            ["holding_month", "permno", *RISK_COLUMNS],
        ]
        raise RuntimeError(f"active position-days lack risk inputs:\n{missing.head(20)}")
    covariance_path = _resolve(config["sources"]["factor_covariance"]["path"])
    covariance = duckdb.sql(
        f"""
        SELECT CAST(holding_month AS VARCHAR) holding_month, factor_row,
               factor_col, covariance_monthly
        FROM read_parquet('{_duck(covariance_path)}')
        WHERE holding_month BETWEEN '{start}' AND '{end}'
        """
    ).df()
    roots = {
        str(month): factor_covariance_matrix(frame)
        for month, frame in covariance.groupby("holding_month", sort=True)
    }
    required_months = set(days["holding_month"].unique())
    if required_months.difference(roots):
        raise RuntimeError("factor covariance is missing replay months")
    return days, roots


def _constraint_diagnostics(
    weights: np.ndarray,
    frame: pd.DataFrame,
    covariance_root: np.ndarray,
) -> dict[str, Any]:
    betas = frame[[f"beta_risk_{name}" for name in FACTOR_NAMES]].to_numpy(float)
    specific = frame["specific_variance_monthly"].to_numpy(float)
    risk, exposures = portfolio_risk(weights, betas, covariance_root, specific)
    industry = frame["sic2_industry_category"].fillna("UNKNOWN_SIC2").astype(str).to_numpy()
    maximum_net = 0.0
    maximum_gross = 0.0
    for name in sorted(set(industry.tolist())):
        values = weights[industry == name]
        maximum_net = max(maximum_net, abs(float(values.sum())))
        maximum_gross = max(maximum_gross, float(np.abs(values).sum()))
    return {
        "gross": float(np.abs(weights).sum()),
        "net": float(weights.sum()),
        "forecast_volatility": risk,
        "factor_exposures": exposures,
        "maximum_industry_net": maximum_net,
        "maximum_industry_gross": maximum_gross,
        "maximum_position": float(np.abs(weights).max(initial=0.0)),
    }


def _solve_survivor_refill(
    *,
    current_weights: np.ndarray,
    warned: np.ndarray,
    frame: pd.DataFrame,
    covariance_root: np.ndarray,
    nav_dollars: float,
    refill_budget: float,
    pre_action_gross: float,
    remaining_fraction: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Maximize frozen expected net payoff using only monotone survivor adds."""

    current = np.asarray(current_weights, dtype=np.float64)
    warned = np.asarray(warned, dtype=bool)
    if len(current) != len(frame) or len(warned) != len(frame):
        raise ValueError("refill inputs have different lengths")
    rules = config["reoptimization"]
    tolerance = float(rules["solver_tolerance"])
    current_diag = _constraint_diagnostics(current, frame, covariance_root)
    gross_limit = max(
        current_diag["gross"],
        min(float(rules["maximum_gross"]), float(pre_action_gross)),
    )
    effective_budget = min(
        max(float(refill_budget), 0.0),
        max(gross_limit - current_diag["gross"], 0.0),
    )
    survivor = (np.abs(current) > tolerance) & ~warned
    if effective_budget <= tolerance or not survivor.any() or remaining_fraction <= 0.0:
        return current.copy(), {
            "solver_status": "NO_REFILL_CAPACITY",
            "refill_budget": float(refill_budget),
            "effective_refill_budget": effective_budget,
            "refill_gross": 0.0,
            **{f"pre_{key}": value for key, value in current_diag.items() if key != "factor_exposures"},
            **{f"post_{key}": value for key, value in current_diag.items() if key != "factor_exposures"},
        }

    n = len(current)
    variable = cp.Variable(n)
    delta = variable - current
    constraints: list[cp.Constraint] = []
    position_cap = np.maximum(
        float(rules["maximum_position_weight"]), np.abs(current)
    )
    for index in range(n):
        if warned[index] or abs(current[index]) <= tolerance:
            constraints.append(variable[index] == current[index])
        elif current[index] > 0.0:
            constraints.extend(
                [variable[index] >= current[index], variable[index] <= position_cap[index]]
            )
        else:
            constraints.extend(
                [variable[index] <= current[index], variable[index] >= -position_cap[index]]
            )

    adv = frame["adv20_usd"].to_numpy(float)
    trade_cap = float(rules["maximum_one_way_adv_participation"]) * adv / float(nav_dollars)
    constraints.extend([cp.abs(delta) <= trade_cap, cp.norm1(delta) <= effective_budget])

    long_index = np.flatnonzero(current > tolerance)
    short_index = np.flatnonzero(current < -tolerance)
    long_gross = cp.sum(variable[long_index]) if len(long_index) else cp.Constant(0.0)
    short_gross = -cp.sum(variable[short_index]) if len(short_index) else cp.Constant(0.0)
    gross = long_gross + short_gross
    net = cp.sum(variable)
    constraints.append(gross <= gross_limit + tolerance)
    constraints.append(
        net
        >= min(float(rules["minimum_net"]), current_diag["net"]) - tolerance
    )
    constraints.append(
        net
        <= max(float(rules["maximum_net"]), current_diag["net"]) + tolerance
    )

    betas = frame[[f"beta_risk_{name}" for name in FACTOR_NAMES]].to_numpy(float)
    exposures = betas.T @ variable
    current_exposures = np.asarray(current_diag["factor_exposures"], dtype=float)
    factor_limits = np.maximum(
        np.asarray(
            [
                float(rules["market_beta_absolute_limit"]),
                *([float(rules["other_factor_absolute_limit"])] * (len(FACTOR_NAMES) - 1)),
            ]
        ),
        np.abs(current_exposures),
    )
    constraints.append(cp.abs(exposures) <= factor_limits + tolerance)
    factor_component = covariance_root @ exposures
    specific = frame["specific_variance_monthly"].to_numpy(float)
    specific_component = cp.multiply(np.sqrt(np.maximum(specific, 0.0)), variable)
    annual_variance = 12.0 * (
        cp.sum_squares(factor_component) + cp.sum_squares(specific_component)
    )
    risk_limit = max(
        float(rules["forecast_volatility_limit"]),
        float(current_diag["forecast_volatility"]),
    )
    constraints.append(annual_variance <= risk_limit**2 + tolerance)

    industries = frame["sic2_industry_category"].fillna("UNKNOWN_SIC2").astype(str).to_numpy()
    for industry in sorted(set(industries.tolist())):
        index = np.flatnonzero(industries == industry)
        current_values = current[index]
        net_limit = max(
            float(rules["sic2_industry_net_absolute_limit"]),
            abs(float(current_values.sum())),
        )
        industry_gross_limit = max(
            float(rules["sic2_industry_gross_limit"]),
            float(np.abs(current_values).sum()),
        )
        constraints.extend(
            [
                cp.abs(cp.sum(variable[index])) <= net_limit + tolerance,
                cp.sum(cp.abs(variable[index])) <= industry_gross_limit + tolerance,
            ]
        )

    costs = config["account_and_costs"]
    cost = cvxpy_total_cost(
        variable,
        current,
        frame["full_spread_20d"].to_numpy(float),
        adv,
        frame["impact_volatility_daily"].to_numpy(float),
        frame["primary_borrow_rate_annual"].to_numpy(float) * float(remaining_fraction),
        float(nav_dollars),
        impact_coefficient=float(costs["impact_coefficient"]),
        spread_multiplier=float(costs["spread_multiplier"]),
        financing_rate_annual=float(costs["financing_rate_annual"])
        * float(remaining_fraction),
    )
    constraints.extend(cost.constraints)
    mu = (
        frame["monthly_expected_stock_excess_return"].to_numpy(float)
        * float(remaining_fraction)
    )
    problem = cp.Problem(cp.Maximize(10_000.0 * (mu @ variable - cost.total)), constraints)
    problem.solve(
        solver=str(rules["solver"]),
        max_iter=1000,
        tol_gap_abs=tolerance,
        tol_gap_rel=tolerance,
        tol_feas=tolerance,
        equilibrate_enable=True,
    )
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or variable.value is None:
        raise RuntimeError(f"survivor refill failed with status {problem.status}")
    result = np.asarray(variable.value, dtype=np.float64).reshape(-1)
    result[np.abs(result - current) <= 5e-8] = current[np.abs(result - current) <= 5e-8]
    violation = np.zeros(n, dtype=bool)
    violation |= warned & (np.abs(result - current) > 2e-6)
    violation |= (current > tolerance) & (result < current - 2e-6)
    violation |= (current < -tolerance) & (result > current + 2e-6)
    violation |= (np.abs(current) <= tolerance) & (np.abs(result) > 2e-6)
    if violation.any():
        raise RuntimeError("refill solution reduced, re-entered, or reversed a position")
    refill_gross = float(np.abs(result - current).sum())
    if refill_gross > effective_budget + 2e-6:
        raise RuntimeError("refill exceeded the contemporaneous released-gross budget")
    post_diag = _constraint_diagnostics(result, frame, covariance_root)
    if post_diag["gross"] > gross_limit + 2e-6:
        raise RuntimeError("refill exceeded the gross limit")
    return result, {
        "solver_status": str(problem.status),
        "objective_value": float(problem.value),
        "refill_budget": float(refill_budget),
        "effective_refill_budget": effective_budget,
        "refill_gross": refill_gross,
        "remaining_fraction": float(remaining_fraction),
        **{f"pre_{key}": value for key, value in current_diag.items() if key != "factor_exposures"},
        **{f"post_{key}": value for key, value in post_diag.items() if key != "factor_exposures"},
    }


def _state_from_receipt(path: Path, identity: Mapping[str, str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if value.get("identity_sha256") != identity["identity_sha256"]:
        raise RuntimeError(f"checkpoint identity mismatch: {path}")
    if value.get("status") != "COMPLETE":
        return None
    return value["state"]


def _month_replay(
    *,
    month: str,
    month_frame: pd.DataFrame,
    action_schedule: pd.DataFrame,
    covariance_root: np.ndarray,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    policy_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    month_frame = month_frame.sort_values(["calendar_date", "permno"], kind="mergesort")
    first = month_frame.sort_values("calendar_date", kind="mergesort").drop_duplicates("permno")
    first = first.sort_values("permno", kind="mergesort").reset_index(drop=True)
    permnos = first["permno"].astype(int).to_numpy()
    target = dict(zip(permnos, first["target_weight"].astype(float), strict=True))
    nav = float(state["nav"])
    exposure = {int(key): float(value) for key, value in state.get("exposure", {}).items()}
    risk_cache = {
        int(key): {name: float(item[name]) for name in COST_COLUMNS}
        for key, item in state.get("risk_cache", {}).items()
    }
    for row in first.itertuples(index=False):
        risk_cache[int(row.permno)] = {
            "full_spread_20d": float(row.full_spread_20d),
            "adv20_usd": float(row.adv20_usd),
            "impact_volatility_daily": float(row.impact_volatility_daily),
            "primary_borrow_rate_annual": float(row.primary_borrow_rate_annual),
        }
    month_nav_start = nav
    union = sorted(set(exposure) | set(target))
    transition_risk = pd.DataFrame([risk_cache[value] for value in union])
    previous_weight = np.asarray([exposure.get(value, 0.0) / nav for value in union])
    target_weight = np.asarray([target.get(value, 0.0) for value in union])
    entry_cost_rate = base._transaction_cost(  # noqa: SLF001
        target_weight - previous_weight,
        transition_risk,
        nav,
        config["account_and_costs"],
    )
    entry_cost_dollars = nav * entry_cost_rate
    exposure = {value: target[value] * nav for value in target}
    nav -= entry_cost_dollars
    fraction = {value: 1.0 for value in target}
    warned: set[int] = set()
    costs_month = {"transaction": entry_cost_dollars, "borrow": 0.0, "financing": 0.0}
    daily_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    schedule = action_schedule.copy()
    if len(schedule):
        schedule["calendar_date"] = pd.to_datetime(schedule["calendar_date"], errors="raise")
    processed_actions = 0

    for calendar_date, day in month_frame.groupby("calendar_date", sort=True):
        day = day.sort_values("permno", kind="mergesort")
        nav_start = nav
        rf = float(day["rf"].iloc[0])
        day_permnos = day["permno"].astype(int).to_numpy()
        exposed = np.asarray([exposure.get(value, 0.0) for value in day_permnos], dtype=float)
        stock_return = day["stock_total_return"].to_numpy(float)
        stock_pnl = float(np.sum(exposed * (stock_return - rf)))
        weights = exposed / nav
        sessions = int(day["sessions_in_month"].iloc[0])
        borrow_rate = day["primary_borrow_rate_annual"].to_numpy(float)
        borrow_rate_day = float(
            np.sum(np.maximum(-weights, 0.0) * borrow_rate / 12.0 / sessions)
        )
        gross = float(np.abs(weights).sum())
        financing_rate_day = (
            max(gross - 1.0, 0.0)
            * float(config["account_and_costs"]["financing_rate_annual"])
            / 12.0
            / sessions
        )
        borrow_dollars = nav * borrow_rate_day
        financing_dollars = nav * financing_rate_day
        nav = nav * (1.0 + rf) + stock_pnl - borrow_dollars - financing_dollars
        costs_month["borrow"] += borrow_dollars
        costs_month["financing"] += financing_dollars
        for value, result in zip(day_permnos, exposed * (1.0 + stock_return), strict=True):
            exposure[int(value)] = float(result)

        date_actions = schedule.loc[schedule["calendar_date"].eq(pd.Timestamp(calendar_date))]
        action_cost_dollars = 0.0
        refill_gross = 0.0
        if len(date_actions):
            processed_actions += len(date_actions)
            meta = first.set_index("permno").loc[permnos].reset_index()
            pre_exposure = np.asarray([exposure.get(int(value), 0.0) for value in permnos])
            pre_weights = pre_exposure / nav
            post_action_exposure = pre_exposure.copy()
            warning_delta = np.zeros(len(permnos), dtype=float)
            index_lookup = {int(value): index for index, value in enumerate(permnos)}
            for action in date_actions.sort_values("permno").itertuples(index=False):
                value = int(action.permno)
                index = index_lookup[value]
                old_fraction = float(fraction.get(value, 0.0))
                if abs(old_fraction - float(action.old_fraction)) > 1e-8:
                    raise RuntimeError(
                        f"warning fraction mismatch for {month}/{calendar_date}/{value}"
                    )
                new_fraction = float(action.new_fraction)
                if new_fraction >= old_fraction - 1e-12 or old_fraction <= 0.0:
                    raise RuntimeError("saved V6.7 warning is not a strict reduction")
                old_amount = post_action_exposure[index]
                new_amount = old_amount * new_fraction / old_fraction
                post_action_exposure[index] = new_amount
                warning_delta[index] = new_amount - old_amount
                fraction[value] = new_fraction
                warned.add(value)
                action_rows.append(
                    {
                        "policy": policy_id,
                        "holding_month": month,
                        "calendar_date": pd.Timestamp(calendar_date),
                        "permno": value,
                        "action_type": str(action.action_type),
                        "old_fraction": old_fraction,
                        "new_fraction": new_fraction,
                        "warning_trade_weight": abs(new_amount - old_amount) / nav,
                    }
                )
            post_action_weights = post_action_exposure / nav
            pre_action_gross = float(np.abs(pre_weights).sum())
            post_action_gross = float(np.abs(post_action_weights).sum())
            removed_gross = max(pre_action_gross - post_action_gross, 0.0)
            remaining_sessions = max(sessions - int(day["day_index"].iloc[0]) - 1, 0)
            warned_mask = np.asarray([int(value) in warned for value in permnos], dtype=bool)
            final_weights, solve = _solve_survivor_refill(
                current_weights=post_action_weights,
                warned=warned_mask,
                frame=meta,
                covariance_root=covariance_root,
                nav_dollars=nav,
                refill_budget=removed_gross,
                pre_action_gross=pre_action_gross,
                remaining_fraction=remaining_sessions / sessions,
                config=config,
            )
            refill_gross = float(solve["refill_gross"])
            combined_delta = final_weights - pre_weights
            event_cost_rate = base._transaction_cost(  # noqa: SLF001
                combined_delta,
                meta,
                nav,
                config["account_and_costs"],
            )
            action_cost_dollars = nav * event_cost_rate
            final_exposure = final_weights * nav
            for value, amount in zip(permnos, final_exposure, strict=True):
                exposure[int(value)] = float(amount)
            nav -= action_cost_dollars
            costs_month["transaction"] += action_cost_dollars
            event_id = f"{month}-{pd.Timestamp(calendar_date).date()}"
            event_rows.append(
                {
                    "policy": policy_id,
                    "event_id": event_id,
                    "holding_month": month,
                    "calendar_date": pd.Timestamp(calendar_date),
                    "warning_count": int(len(date_actions)),
                    "pre_action_gross": pre_action_gross,
                    "post_action_gross": post_action_gross,
                    "removed_gross": removed_gross,
                    "refill_gross": refill_gross,
                    "final_gross_before_cost": float(np.abs(final_weights).sum()),
                    "combined_gross_traded_weight": float(np.abs(combined_delta).sum()),
                    "combined_one_way_turnover": 0.5 * float(np.abs(combined_delta).sum()),
                    "transaction_cost_dollars": action_cost_dollars,
                    **solve,
                }
            )
            for index, value in enumerate(permnos):
                if abs(combined_delta[index]) <= 1e-10:
                    continue
                trade_rows.append(
                    {
                        "policy": policy_id,
                        "event_id": event_id,
                        "holding_month": month,
                        "calendar_date": pd.Timestamp(calendar_date),
                        "permno": int(value),
                        "warned": bool(warning_delta[index] != 0.0),
                        "pre_weight": float(pre_weights[index]),
                        "post_warning_weight": float(post_action_weights[index]),
                        "final_weight": float(final_weights[index]),
                        "warning_delta": float(warning_delta[index] / nav),
                        "refill_delta": float(final_weights[index] - post_action_weights[index]),
                        "combined_delta": float(combined_delta[index]),
                    }
                )

        gross_end = sum(abs(value) for value in exposure.values()) / nav
        daily_rows.append(
            {
                "policy": policy_id,
                "holding_month": month,
                "calendar_date": pd.Timestamp(calendar_date),
                "nav_start": nav_start,
                "nav_end": nav,
                "rf": rf,
                "stock_pnl_dollars": stock_pnl,
                "borrow_cost_dollars": borrow_dollars,
                "financing_cost_dollars": financing_dollars,
                "action_cost_dollars": action_cost_dollars,
                "gross_end": gross_end,
                "warning_actions": int(len(date_actions)),
                "refill_gross": refill_gross,
            }
        )

    if processed_actions != len(schedule):
        raise RuntimeError(f"not every saved warning executed in {month}")
    month_days = month_frame.drop_duplicates("calendar_date").sort_values("calendar_date")
    rf_month = float(np.prod(1.0 + month_days["rf"].to_numpy(float)) - 1.0)
    net_return = nav / month_nav_start - 1.0
    total_cost = sum(costs_month.values()) / month_nav_start
    monthly = pd.DataFrame(
        [
            {
                "policy": policy_id,
                "holding_month": month,
                "nav_start": month_nav_start,
                "nav_end": nav,
                "net_return": net_return,
                "rf": rf_month,
                "net_excess_return": net_return - rf_month,
                "transaction_cost": costs_month["transaction"] / month_nav_start,
                "borrow_cost": costs_month["borrow"] / month_nav_start,
                "financing_cost": costs_month["financing"] / month_nav_start,
                "total_cost": total_cost,
                "warning_count": int(len(action_rows)),
                "reoptimization_count": int(len(event_rows)),
            }
        ]
    )
    daily = pd.DataFrame(daily_rows)
    actions = pd.DataFrame(action_rows)
    events = pd.DataFrame(event_rows)
    trades = pd.DataFrame(trade_rows)
    next_state = {
        "nav": nav,
        "exposure": {
            str(key): float(value)
            for key, value in exposure.items()
            if abs(float(value)) > 1e-10
        },
        "risk_cache": {
            str(key): risk_cache[key]
            for key in exposure
            if key in risk_cache and abs(float(exposure[key])) > 1e-10
        },
    }
    return monthly, daily, actions, events, trades, next_state


def _phase_replay(
    *,
    phase: str,
    days: pd.DataFrame,
    covariance_roots: Mapping[str, np.ndarray],
    actions: pd.DataFrame,
    evaluation_start: str,
    evaluation_end: str,
    config: Mapping[str, Any],
    identity: Mapping[str, str],
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    phase_root = output / phase
    state: dict[str, Any] = {
        "nav": float(config["account_and_costs"]["starting_nav_dollars"]),
        "exposure": {},
        "risk_cache": {},
    }
    month_list = _months(config["chronology"]["initial_replay_start"], evaluation_end)
    for month in month_list:
        month_root = phase_root / "checkpoints" / month
        receipt_path = month_root / "receipt.json"
        saved_state = _state_from_receipt(receipt_path, identity)
        if saved_state is not None:
            state = saved_state
            continue
        month_frame = days.loc[days["holding_month"].eq(month)].copy()
        if month_frame.empty:
            raise RuntimeError(f"prepared panel lacks replay month {month}")
        month_actions = actions.loc[actions["holding_month"].astype(str).eq(month)].copy()
        if month < evaluation_start and len(month_actions):
            raise RuntimeError("warning action appears before the evaluation start")
        monthly, daily, warning_rows, events, trades, state = _month_replay(
            month=month,
            month_frame=month_frame,
            action_schedule=month_actions,
            covariance_root=covariance_roots[month],
            state=state,
            config=config,
            policy_id="v6_8_survivor_reoptimization",
        )
        _atomic_parquet(monthly, month_root / "monthly.parquet")
        _atomic_parquet(daily, month_root / "daily.parquet")
        _atomic_parquet(warning_rows, month_root / "actions.parquet")
        _atomic_parquet(events, month_root / "events.parquet")
        _atomic_parquet(trades, month_root / "trades.parquet")
        _atomic_json(
            receipt_path,
            {
                "schema": "research_v6_8_month_checkpoint_v1",
                "status": "COMPLETE",
                "phase": phase,
                "holding_month": month,
                "identity_sha256": identity["identity_sha256"],
                "warning_count": int(len(warning_rows)),
                "reoptimization_count": int(len(events)),
                "state": state,
            },
        )

    def collect(name: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for month in month_list:
            if month < evaluation_start:
                continue
            path = phase_root / "checkpoints" / month / f"{name}.parquet"
            value = _read_parquet(path)
            if len(value):
                frames.append(value)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    monthly = collect("monthly")
    daily = collect("daily")
    warnings = collect("actions")
    events = collect("events")
    trades = collect("trades")
    summary = _summary("v6_8_survivor_reoptimization", monthly, daily, warnings, events)
    _atomic_parquet(monthly, phase_root / "monthly.parquet")
    _atomic_parquet(daily, phase_root / "daily.parquet")
    _atomic_parquet(warnings, phase_root / "actions.parquet")
    _atomic_parquet(events, phase_root / "events.parquet")
    _atomic_parquet(trades, phase_root / "trades.parquet")
    _atomic_json(phase_root / "summary.json", summary)
    return monthly, daily, warnings, events, trades, summary


def _summary(
    policy_id: str,
    monthly: pd.DataFrame,
    daily: pd.DataFrame,
    actions: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, Any]:
    excess = monthly["net_excess_return"].to_numpy(float)
    total = monthly["net_return"].to_numpy(float)
    volatility = float(np.std(excess, ddof=1) * math.sqrt(12.0))
    wealth = float(np.prod(1.0 + total))
    daily_nav = daily["nav_end"].to_numpy(float)
    peaks = np.maximum.accumulate(np.r_[daily["nav_start"].iloc[0], daily_nav])[1:]
    gross_traded = float(events["combined_gross_traded_weight"].sum()) if len(events) else 0.0
    refill = float(events["refill_gross"].sum()) if len(events) else 0.0
    removed = float(events["removed_gross"].sum()) if len(events) else 0.0
    return {
        "policy": policy_id,
        "months": int(len(monthly)),
        "starting_nav": float(monthly["nav_start"].iloc[0]),
        "ending_nav": float(monthly["nav_end"].iloc[-1]),
        "cumulative_total_return": wealth - 1.0,
        "total_return_cagr": wealth ** (12.0 / len(monthly)) - 1.0,
        "annualized_net_excess_return": float(np.mean(excess) * 12.0),
        "annualized_realized_volatility": volatility,
        "net_sharpe": float(np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(12.0)),
        "maximum_drawdown": float(np.min(daily_nav / peaks - 1.0, initial=0.0)),
        "positive_month_fraction": float(np.mean(excess > 0.0)),
        "annualized_total_cost": float(monthly["total_cost"].mean() * 12.0),
        "annualized_transaction_cost": float(monthly["transaction_cost"].mean() * 12.0),
        "annualized_borrow_cost": float(monthly["borrow_cost"].mean() * 12.0),
        "annualized_financing_cost": float(monthly["financing_cost"].mean() * 12.0),
        "maximum_close_gross": float(daily["gross_end"].max()),
        "intervention_count": int(len(actions)),
        "exit_count": int(actions["new_fraction"].eq(0.0).sum()) if len(actions) else 0,
        "trim_count": int(actions["new_fraction"].between(0.0, 1.0, inclusive="neither").sum()) if len(actions) else 0,
        "reoptimization_count": int(len(events)),
        "removed_gross": removed,
        "refilled_gross": refill,
        "refill_fraction": refill / removed if removed > 0.0 else 0.0,
        "intervention_gross_traded_weight": gross_traded,
        "intervention_one_way_turnover": 0.5 * gross_traded,
    }


def _hac_t(values: Iterable[float], lags: int = 6) -> float:
    series = np.asarray(list(values), dtype=float)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 3:
        return math.nan
    centered = series - float(series.mean())
    gamma0 = float(centered @ centered) / n
    variance = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = float(centered[lag:] @ centered[:-lag]) / n
        variance += 2.0 * weight * gamma
    standard_error = math.sqrt(max(variance, 0.0) / n)
    return float(series.mean() / standard_error) if standard_error > 0.0 else math.nan


def _paired(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    paired = left[["holding_month", "net_excess_return"]].merge(
        right[["holding_month", "net_excess_return"]],
        on="holding_month",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    difference = paired["net_excess_return_right"] - paired["net_excess_return_left"]
    return {
        "months": int(len(paired)),
        "annualized_difference": float(difference.mean() * 12.0),
        "paired_hac_t": _hac_t(difference, lags=6),
        "positive_difference_fraction": float((difference > 0.0).mean()),
    }


def _result_markdown(result: Mapping[str, Any]) -> str:
    validation = result["validation"]
    historical = result["historical_test"]
    return f"""# Research V6.8 daily re-optimization result

## Decision

`{result['decision']['terminal_decision']}`

{result['decision']['interpretation']}

## What changed

The saved V6.7 warnings, monthly stocks, initial weights, and next-close execution
are unchanged.  After each executed warning batch, V6.8 maximizes the remaining
frozen monthly expected net payoff using only monotone increases in surviving,
never-warned holdings.  It cannot open a new stock, reverse a side, re-enter an
exit, or enlarge a warned name.  Refill is capped by the gross removed in that
same batch, and cash remains feasible.

## Validation: January--December 2023

- Monthly control: {validation['monthly_control']['annualized_net_excess_return']:.2%} annualized net excess return, {validation['monthly_control']['net_sharpe']:.3f} Sharpe.
- V6.7 cash-holding warning overlay: {validation['v6_7_cash_overlay']['annualized_net_excess_return']:.2%}, {validation['v6_7_cash_overlay']['net_sharpe']:.3f} Sharpe.
- V6.8 survivor re-optimization: {validation['v6_8_reoptimization']['annualized_net_excess_return']:.2%}, {validation['v6_8_reoptimization']['net_sharpe']:.3f} Sharpe.

## Historical evidence: January 2024--August 2025

- Monthly control: {historical['monthly_control']['annualized_net_excess_return']:.2%} annualized net excess return, {historical['monthly_control']['annualized_realized_volatility']:.2%} volatility, {historical['monthly_control']['net_sharpe']:.3f} Sharpe, {historical['monthly_control']['maximum_drawdown']:.2%} maximum drawdown.
- V6.7 cash overlay: {historical['v6_7_cash_overlay']['annualized_net_excess_return']:.2%}, {historical['v6_7_cash_overlay']['annualized_realized_volatility']:.2%}, {historical['v6_7_cash_overlay']['net_sharpe']:.3f}, {historical['v6_7_cash_overlay']['maximum_drawdown']:.2%}.
- V6.8 re-optimization: {historical['v6_8_reoptimization']['annualized_net_excess_return']:.2%}, {historical['v6_8_reoptimization']['annualized_realized_volatility']:.2%}, {historical['v6_8_reoptimization']['net_sharpe']:.3f}, {historical['v6_8_reoptimization']['maximum_drawdown']:.2%}.
- Refilled {historical['v6_8_reoptimization']['refill_fraction']:.1%} of contemporaneously released gross across {historical['v6_8_reoptimization']['reoptimization_count']} action batches.
- Paired annualized difference versus V6.7: {result['paired_differences']['historical_versus_v6_7']['annualized_difference']:.2%}, HAC t {result['paired_differences']['historical_versus_v6_7']['paired_hac_t']:.2f}.

## Evidence boundary

This is a user-authorized post-diagnostic mechanism study on already observed
2024--August 2025 outcomes.  Even if the economic gates pass, it can only become
a prospective paper-trading shadow candidate.  The monthly strategy remains the
trading primary until clean evidence beginning no earlier than September 2026.
"""


def run(config_path: Path, output_root: Path) -> Path:
    config, identity = _verify(config_path)
    output = _output_path(output_root, identity)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run_manifest.json"
    _atomic_json(
        manifest,
        {
            "schema": "research_v6_8_daily_reoptimization_run_v1",
            "status": "RUNNING",
            "identity": identity,
        },
    )
    days, roots = _load_inputs(config)
    validation_actions = _read_parquet(
        _resolve(config["sources"]["validation_warning_actions"]["path"])
    )
    historical_actions = _read_parquet(
        _resolve(config["sources"]["historical_warning_actions"]["path"])
    )
    for actions in (validation_actions, historical_actions):
        actions["holding_month"] = actions["holding_month"].astype(str)
        actions["calendar_date"] = pd.to_datetime(actions["calendar_date"], errors="raise")
        if actions.duplicated(["holding_month", "calendar_date", "permno"]).any():
            raise RuntimeError("saved V6.7 warning schedule contains duplicate action keys")

    validation_monthly, _, validation_warnings, validation_events, _, validation_summary = _phase_replay(
        phase="validation",
        days=days,
        covariance_roots=roots,
        actions=validation_actions,
        evaluation_start=config["chronology"]["validation_start"],
        evaluation_end=config["chronology"]["validation_end"],
        config=config,
        identity=identity,
        output=output,
    )
    historical_monthly, _, historical_warnings, historical_events, _, historical_summary = _phase_replay(
        phase="historical_test",
        days=days,
        covariance_roots=roots,
        actions=historical_actions,
        evaluation_start=config["chronology"]["historical_test_start"],
        evaluation_end=config["chronology"]["historical_test_end"],
        config=config,
        identity=identity,
        output=output,
    )
    if len(validation_warnings) != len(validation_actions):
        raise RuntimeError("validation warning count does not reproduce V6.7")
    if len(historical_warnings) != len(historical_actions):
        raise RuntimeError("historical warning count does not reproduce V6.7")

    selection = json.loads(_resolve(config["sources"]["v6_7_selection"]["path"]).read_text())
    predecessor = json.loads(_resolve(config["sources"]["v6_7_result"]["path"]).read_text())
    validation_control = _read_parquet(
        _resolve(config["sources"]["validation_monthly_control"]["path"])
    )
    validation_v6_7 = _read_parquet(
        _resolve(config["sources"]["validation_v6_7_monthly"]["path"])
    )
    historical_control = _read_parquet(
        _resolve(config["sources"]["historical_monthly_control"]["path"])
    )
    historical_v6_7 = _read_parquet(
        _resolve(config["sources"]["historical_v6_7_monthly"]["path"])
    )
    paired = {
        "validation_versus_monthly_control": _paired(validation_control, validation_monthly),
        "validation_versus_v6_7": _paired(validation_v6_7, validation_monthly),
        "historical_versus_monthly_control": _paired(historical_control, historical_monthly),
        "historical_versus_v6_7": _paired(historical_v6_7, historical_monthly),
    }
    control_historical = predecessor["historical_test"]["monthly_control"]
    v6_7_historical = predecessor["historical_test"]["selected_v6_7_challenger"]
    gates = {
        "exact_v6_7_warning_schedule_reproduced": len(validation_warnings) == len(validation_actions)
        and len(historical_warnings) == len(historical_actions),
        "validation_sharpe_higher_than_v6_7": validation_summary["net_sharpe"]
        > selection["selected_v6_7_challenger"]["summary"]["net_sharpe"],
        "historical_return_higher_than_v6_7": historical_summary[
            "annualized_net_excess_return"
        ]
        > v6_7_historical["annualized_net_excess_return"],
        "historical_sharpe_higher_than_v6_7": historical_summary["net_sharpe"]
        > v6_7_historical["net_sharpe"],
        "historical_drawdown_within_monthly_control_tolerance": historical_summary[
            "maximum_drawdown"
        ]
        >= control_historical["maximum_drawdown"]
        - float(
            config["evaluation"][
                "maximum_drawdown_not_worse_than_monthly_control_by_more_than"
            ]
        ),
        "historical_paired_hac_t_versus_v6_7": paired["historical_versus_v6_7"][
            "paired_hac_t"
        ]
        >= float(config["evaluation"]["paired_monthly_difference_hac_t_minimum"]),
    }
    qualifies = all(gates.values())
    terminal = (
        "V6_8_REOPTIMIZATION_FORWARD_SHADOW_CANDIDATE_ONLY"
        if qualifies
        else "RETAIN_MONTHLY_PRIMARY_AND_V6_7_CASH_OVERLAY_AS_SHADOW_ONLY"
    )
    interpretation = (
        "The survivor re-optimizer improves the frozen economic comparisons and paired inference, but the evidence is post-diagnostic. It may be logged prospectively in paper-trading shadow mode only."
        if qualifies
        else "The survivor re-optimizer does not clear every frozen economic and paired-inference gate. It therefore does not replace the monthly primary or promote the V6.7 overlay beyond shadow monitoring."
    )
    result = {
        "schema": "research_v6_8_daily_reoptimization_result_v1",
        "status": "COMPLETE",
        "identity": identity,
        "data_quality": {
            "position_days": int(len(days)),
            "months": int(days["holding_month"].nunique()),
            "validation_warning_actions": int(len(validation_warnings)),
            "validation_reoptimization_events": int(len(validation_events)),
            "historical_warning_actions": int(len(historical_warnings)),
            "historical_reoptimization_events": int(len(historical_events)),
        },
        "validation": {
            "monthly_control": selection["monthly_control"],
            "v6_7_cash_overlay": selection["selected_v6_7_challenger"]["summary"],
            "v6_8_reoptimization": validation_summary,
        },
        "historical_test": {
            "monthly_control": control_historical,
            "v6_7_cash_overlay": v6_7_historical,
            "v6_8_reoptimization": historical_summary,
        },
        "paired_differences": paired,
        "promotion_gates": gates,
        "decision": {
            "terminal_decision": terminal,
            "all_gates_pass": qualifies,
            "permitted_status_if_all_gates_pass": config["evaluation"][
                "promotion_status_even_if_all_gates_pass"
            ],
            "interpretation": interpretation,
        },
    }
    _atomic_json(output / "result.json", result)
    _atomic_text(output / "RESULT.md", _result_markdown(result))
    _atomic_json(
        manifest,
        {
            "schema": "research_v6_8_daily_reoptimization_run_v1",
            "status": "COMPLETE",
            "identity": identity,
            "result": "result.json",
            "report": "RESULT.md",
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(run(arguments.config.resolve(), arguments.output_root.resolve()))


if __name__ == "__main__":
    main()
