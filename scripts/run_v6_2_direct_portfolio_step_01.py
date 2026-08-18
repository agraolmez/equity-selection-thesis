#!/usr/bin/env python3
"""Run Research V6.2 portfolio-level learning with frozen V6 information."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path("/Users/agraemirolmez/Desktop/Thesis")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_v6_1_portfolio_step_02 as predecessor  # noqa: E402
from portfolio_bot.research_v6_1.portfolio import (  # noqa: E402
    FACTOR_NAMES,
    causal_monotonic_calibration,
    factor_covariance_matrix,
    percentile_rank,
)
from portfolio_bot.research_v6_2.policy import repair_policy_weights  # noqa: E402
from portfolio_bot.research_v6_2.runtime import (  # noqa: E402
    PolicyMonth,
    build_policy,
    fit_policy,
    predict_policy,
)


DEFAULT_CONFIG = ROOT / "configs/research_v6_2/direct_portfolio_step_01.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_2/direct_portfolio_step_01"
RISK_COLUMNS = [
    "holding_month",
    "signal_date",
    "permno",
    "permco",
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


@dataclass
class Dataset:
    master: pd.DataFrame
    matrix: np.ndarray
    views: dict[str, np.ndarray]


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


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _resolve(item: dict[str, Any]) -> Path:
    root = ROOT if item["root"] == "workspace" else EXTERNAL_ROOT
    return root / str(item["path"])


def _code_identity() -> str:
    files = [
        Path(__file__).resolve(),
        ROOT / "src/portfolio_bot/research_v6_2/policy.py",
        ROOT / "src/portfolio_bot/research_v6_2/runtime.py",
        ROOT / "src/portfolio_bot/research_v6_1/portfolio.py",
        ROOT / "src/portfolio_bot/allocation/costs.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def verify_contract(
    config_path: Path,
) -> tuple[dict[str, Any], str, str, dict[str, Path], dict[str, Any], dict[str, Path]]:
    config = json.loads(config_path.read_text())
    if config.get("status") != "FROZEN_READY_TO_RUN":
        raise RuntimeError("V6.2 contract is not frozen and ready")
    predecessor_path = ROOT / str(config["bound_predecessor"]["portfolio_contract"])
    if _sha256(predecessor_path) != config["bound_predecessor"]["portfolio_contract_sha256"]:
        raise RuntimeError("bound V6.1 portfolio contract identity changed")
    predecessor_config, _, predecessor_identity, predecessor_paths = predecessor.verify_contract(
        predecessor_path
    )
    step03_manifest = (
        ROOT / str(config["bound_predecessor"]["step03_result"]) / "run_manifest.json"
    )
    if json.loads(step03_manifest.read_text()).get("status") != "COMPLETE":
        raise RuntimeError("bound V6.1 Step 03 result is not complete")
    paths: dict[str, Path] = {}
    for item in config["sources"]:
        path = _resolve(item)
        if not path.is_file():
            raise FileNotFoundError(f"missing V6.2 source: {path}")
        if _sha256(path) != item["sha256"]:
            raise RuntimeError(f"V6.2 source identity changed: {item['id']}")
        paths[str(item["id"])] = path
    input_identity = hashlib.sha256(
        json.dumps(
            {
                **{item["id"]: item["sha256"] for item in config["sources"]},
                "predecessor": predecessor_identity,
                "step03": _sha256(step03_manifest),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return (
        config,
        _sha256(config_path),
        input_identity,
        paths,
        predecessor_config,
        predecessor_paths,
    )


def _read_risk(path: Path, signal_map: pd.DataFrame | None = None) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [column for column in RISK_COLUMNS if column in available]
    frame = pd.read_parquet(path, columns=columns)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise")
    if "holding_month" not in frame.columns:
        if signal_map is None:
            raise RuntimeError(f"risk source lacks holding_month: {path}")
        frame = frame.merge(signal_map, on="signal_date", how="left", validate="many_to_one")
    frame["holding_month"] = frame["holding_month"].astype(str)
    return frame[RISK_COLUMNS]


def _ff5_calibration(rows: pd.DataFrame, path: Path, config: dict[str, Any]) -> pd.DataFrame:
    column = str(config["ff5_gate"]["prediction_column"])
    keys = ["holding_month", "signal_date", "permno", "permco"]
    frame = pd.read_parquet(
        path, columns=[*keys, column, "alpha_ff5_momentum"]
    ).rename(columns={column: "ff5_score", "alpha_ff5_momentum": "ff5_actual"})
    frame["holding_month"] = frame["holding_month"].astype(str)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise")
    availability = rows[[*keys, "label_available_date"]]
    frame = frame.merge(availability, on=keys, how="left", validate="one_to_one")
    frame["ff5_rank"] = frame.groupby("holding_month", observed=True, sort=False)["ff5_score"].transform(
        lambda value: percentile_rank(value)
    )
    outputs: list[pd.DataFrame] = []
    months = sorted(frame["holding_month"].unique())
    gate = config["ff5_gate"]
    for month in months:
        current = frame.loc[frame["holding_month"].eq(month)].copy()
        signal_date = pd.Timestamp(current["signal_date"].iloc[0])
        history = frame.loc[frame["holding_month"].lt(month)]
        available_months = history.loc[
            pd.to_datetime(history["label_available_date"], errors="coerce").le(signal_date)
        ]["holding_month"].nunique()
        if available_months < int(gate["minimum_history_months"]):
            current["expected_ff5"] = np.nan
        else:
            result = causal_monotonic_calibration(
                history,
                current["ff5_rank"].to_numpy(float),
                score_column="ff5_rank",
                outcome_column="ff5_actual",
                month_column="holding_month",
                availability_column="label_available_date",
                signal_date=signal_date,
                bin_edges=gate["calibration_bin_edges"],
                minimum_history_months=int(gate["minimum_history_months"]),
                shrinkage_prior_months=float(gate["shrinkage_prior_months"]),
            )
            current["expected_ff5"] = result.values
        outputs.append(current[keys + ["expected_ff5"]])
    return pd.concat(outputs, ignore_index=True)


def load_dataset(config: dict[str, Any], paths: dict[str, Path]) -> Dataset:
    rows = pd.read_parquet(paths["rows"]).reset_index(drop=True)
    rows["row_id"] = np.arange(len(rows), dtype=np.int64)
    rows["holding_month"] = rows["holding_month"].astype(str)
    rows["signal_date"] = pd.to_datetime(rows["signal_date"], errors="raise")
    rows["label_available_date"] = pd.to_datetime(
        rows["label_available_date"], errors="coerce"
    )
    signal_map = rows[["signal_date", "holding_month"]].drop_duplicates()
    if signal_map.duplicated("signal_date").any():
        raise RuntimeError("one signal date maps to multiple holding months")
    risk = pd.concat(
        [
            _read_risk(paths["validation_risk_cost"], signal_map),
            _read_risk(paths["december_2019_risk_cost"]),
            _read_risk(paths["final_risk_cost"]),
        ],
        ignore_index=True,
    ).drop_duplicates(["holding_month", "signal_date", "permno", "permco"], keep="last")
    keys = ["holding_month", "signal_date", "permno", "permco"]
    master = rows.merge(risk, on=keys, how="inner", validate="one_to_one")
    master = master.loc[
        master["risk_status"].astype(str).eq("PRESENT")
        & master["cost_input_status"].astype(str).eq("PRESENT")
        & np.isfinite(pd.to_numeric(master["unranked_excess_return"], errors="coerce"))
    ].copy()
    numeric = [
        *(f"beta_risk_{name}" for name in FACTOR_NAMES),
        "specific_variance_monthly",
        "full_spread_20d",
        "adv20_usd",
        "impact_volatility_daily",
        "primary_borrow_rate_annual",
    ]
    valid = np.ones(len(master), dtype=bool)
    for column in numeric:
        master[column] = pd.to_numeric(master[column], errors="coerce")
        valid &= np.isfinite(master[column].to_numpy(float))
    master = master.loc[valid].copy()
    master = master.merge(
        _ff5_calibration(rows, paths["ff5_predictions"], config),
        on=keys,
        how="left",
        validate="one_to_one",
    ).sort_values(["holding_month", "permno"], kind="mergesort").reset_index(drop=True)
    matrix = np.load(paths["encoded_values"], mmap_mode="r")
    if matrix.shape[0] != len(rows):
        raise RuntimeError("V6.2 feature matrix row count differs from bound rows")
    manifest = json.loads(paths["feature_views"].read_text())
    views = {
        name: np.asarray(manifest["views"][name]["encoded_indices"], dtype=np.int64)
        for name in config["feature_views"]
    }
    return Dataset(master=master, matrix=matrix, views=views)


def _policy_month(
    dataset: Dataset,
    view: str,
    policy: str,
    month: str,
    *,
    eligible_outcomes_only_as_of: pd.Timestamp | None = None,
    apply_ff5_gate: bool = True,
) -> PolicyMonth:
    frame = dataset.master.loc[dataset.master["holding_month"].eq(month)].copy()
    if eligible_outcomes_only_as_of is not None:
        frame = frame.loc[frame["label_available_date"].le(eligible_outcomes_only_as_of)]
    if frame.empty:
        raise RuntimeError(f"V6.2 has no usable rows for {month}")
    long_eligible = (
        frame["long_eligible"].fillna(False).to_numpy(bool)
        & frame["new_long_trade_allowed"].fillna(False).to_numpy(bool)
    )
    short_eligible = (
        frame["short_eligible"].fillna(False).to_numpy(bool)
        & frame["new_short_trade_allowed"].fillna(False).to_numpy(bool)
    )
    if policy == "direct_weight_plus_ff5" and apply_ff5_gate:
        expected = frame["expected_ff5"].to_numpy(float)
        mature = np.isfinite(expected)
        long_eligible &= (~mature) | (expected > 0.0)
        short_eligible &= (~mature) | (expected < 0.0)
    elif policy not in {"direct_weight", "direct_weight_plus_ff5"}:
        raise ValueError(f"unknown V6.2 policy: {policy}")
    row_index = frame["row_id"].to_numpy(np.int64)
    feature_index = dataset.views[view]
    features = np.asarray(
        dataset.matrix[np.ix_(row_index, feature_index)], dtype=np.float32
    )
    return PolicyMonth(
        holding_month=month,
        permno=frame["permno"].to_numpy(np.int64),
        features=features,
        excess_return=frame["unranked_excess_return"].to_numpy(np.float32),
        long_eligible=long_eligible,
        short_eligible=short_eligible,
        full_spread=frame["full_spread_20d"].to_numpy(np.float32),
        adv20_usd=frame["adv20_usd"].to_numpy(np.float32),
        impact_volatility_daily=frame["impact_volatility_daily"].to_numpy(np.float32),
        borrow_rate_annual=frame["primary_borrow_rate_annual"].to_numpy(np.float32),
        factor_betas=frame[[f"beta_risk_{name}" for name in FACTOR_NAMES]].to_numpy(np.float32),
        specific_variance_monthly=frame["specific_variance_monthly"].to_numpy(np.float32),
    )


def _fit_month_list(
    dataset: Dataset,
    config: dict[str, Any],
    view: str,
    policy: str,
    *,
    start: str,
    end: str,
    cutoff: pd.Timestamp,
) -> list[PolicyMonth]:
    months = [
        month
        for month in sorted(dataset.master["holding_month"].unique())
        if start <= month <= end
        and bool(
            dataset.master.loc[
                dataset.master["holding_month"].eq(month), "label_available_date"
            ].le(cutoff).any()
        )
    ]
    return [
        _policy_month(
            dataset,
            view,
            policy,
            month,
            eligible_outcomes_only_as_of=cutoff,
            apply_ff5_gate=False,
        )
        for month in months
    ]


def _fit_settings(config: dict[str, Any]) -> dict[str, float]:
    portfolio = config["portfolio"]
    model = config["model"]
    return {
        "nav_dollars": float(portfolio["nav_dollars"]),
        "maximum_position_weight": float(portfolio["maximum_position_weight"]),
        "maximum_one_way_adv_participation": float(portfolio["maximum_one_way_adv_participation"]),
        "financing_rate_annual": 0.005,
        "spread_multiplier": 1.0,
        "impact_coefficient": 0.10,
        "gradient_clip": float(model["gradient_clip"]),
    }


def _raw_base_frame(
    inputs: predecessor.PreparedInputs,
    month: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    current = inputs.forecasts.loc[inputs.forecasts["holding_month"].eq(month)].copy()
    risk = inputs.risk.loc[inputs.risk["holding_month"].eq(month)].copy()
    frame = current.merge(risk, on=predecessor.KEYS, how="inner", validate="one_to_one")
    numeric = [
        *(f"beta_risk_{name}" for name in FACTOR_NAMES),
        "specific_variance_monthly",
        "full_spread_20d",
        "adv20_usd",
        "impact_volatility_daily",
        "primary_borrow_rate_annual",
        "stress_borrow_rate_annual",
    ]
    valid = np.ones(len(frame), dtype=bool)
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        valid &= np.isfinite(frame[column].to_numpy(float))
    valid &= frame["risk_status"].astype(str).eq("PRESENT").to_numpy(bool)
    valid &= frame["cost_input_status"].astype(str).eq("PRESENT").to_numpy(bool)
    frame = frame.loc[valid].sort_values("permno", kind="mergesort").reset_index(drop=True)
    frame["sic2_industry_category"] = frame["sic2_industry_category"].fillna("UNKNOWN_SIC2").astype(str)
    covariance = inputs.covariance.loc[inputs.covariance["holding_month"].eq(month)]
    return frame, factor_covariance_matrix(covariance)


def _model_from_state(
    config: dict[str, Any], feature_count: int, state: dict[str, torch.Tensor], seed_offset: int
) -> torch.nn.Module:
    model = build_policy(config, feature_count, seed_offset=seed_offset)
    model.load_state_dict(state)
    model.eval()
    return model


def _evaluate_block(
    model: torch.nn.Module,
    dataset: Dataset,
    config: dict[str, Any],
    base_config: dict[str, Any],
    base_frames: dict[str, tuple[pd.DataFrame, np.ndarray]],
    view: str,
    policy: str,
    profile_id: str,
    months: list[str],
    *,
    initial_previous: dict[int, float] | None = None,
    initial_nav: float | None = None,
    checkpoint_root: Path | None = None,
    identity: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, float], float]:
    previous_map = dict(initial_previous or {})
    nav = float(initial_nav or config["portfolio"]["nav_dollars"])
    summaries: list[dict[str, Any]] = []
    portfolio = config["portfolio"]
    for month in months:
        receipt_path = checkpoint_root / f"{month}.json" if checkpoint_root else None
        position_path = checkpoint_root / f"{month}.parquet" if checkpoint_root else None
        if receipt_path and position_path and receipt_path.is_file() and position_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if identity and any(receipt.get(key) != value for key, value in identity.items()):
                raise RuntimeError(f"V6.2 month checkpoint identity mismatch for {month}")
            positions = pd.read_parquet(position_path, columns=["permno", "end_drift_weight"])
            previous_map = {
                int(row.permno): float(row.end_drift_weight)
                for row in positions.itertuples(index=False)
                if abs(float(row.end_drift_weight)) > 1e-12
            }
            summary = dict(receipt["summary"])
            nav = float(summary["nav_end_primary"])
            summaries.append(summary)
            continue
        policy_month = _policy_month(dataset, view, policy, month)
        desired, learned_long, learned_short = predict_policy(
            model,
            policy_month,
            nav_dollars=nav,
            maximum_position_weight=float(portfolio["maximum_position_weight"]),
            maximum_one_way_adv_participation=float(portfolio["maximum_one_way_adv_participation"]),
            short_gross_zero_threshold=float(config["model"]["short_gross_zero_threshold"]),
        )
        frame, covariance_root = base_frames[month]
        desired_map = {int(p): float(w) for p, w in zip(policy_month.permno, desired)}
        desired_aligned = frame["permno"].map(desired_map).fillna(0.0).to_numpy(float)
        previous = predecessor._aligned_previous(frame, previous_map)
        betas = frame[[f"beta_risk_{name}" for name in FACTOR_NAMES]].to_numpy(float)
        repair = repair_policy_weights(
            desired_aligned,
            previous,
            factor_betas=betas,
            covariance_root=covariance_root,
            specific_variance_monthly=frame["specific_variance_monthly"].to_numpy(float),
            industries=frame["sic2_industry_category"].to_numpy(str),
            adv20_usd=frame["adv20_usd"].to_numpy(float),
            nav_dollars=nav,
            maximum_position_weight=float(portfolio["maximum_position_weight"]),
            maximum_one_way_adv_participation=float(portfolio["maximum_one_way_adv_participation"]),
            maximum_one_way_turnover=float(portfolio["maximum_one_way_turnover"]),
            maximum_gross=float(portfolio["maximum_gross"]),
            minimum_net=float(portfolio["minimum_net"]),
            maximum_net=float(portfolio["maximum_net"]),
            forecast_volatility_target=float(portfolio["forecast_volatility_target"]),
            market_beta_absolute_limit=float(portfolio["market_beta_absolute_limit"]),
            other_factor_absolute_limit=float(portfolio["other_factor_absolute_limit"]),
            sic2_industry_net_absolute_limit=float(portfolio["sic2_industry_net_absolute_limit"]),
            sic2_industry_gross_limit=float(portfolio["sic2_industry_gross_limit"]),
        )
        weights = repair.weights
        primary_cost, stress_cost = predecessor._cost_breakdowns(
            weights, previous, frame, base_config, nav
        )
        excess = frame["stock_excess_return"].to_numpy(float)
        if np.any((weights != 0.0) & ~np.isfinite(excess)):
            raise RuntimeError(f"V6.2 holding lacks an authenticated outcome in {month}")
        excess = np.where(np.isfinite(excess), excess, 0.0)
        gross_excess = float(weights @ excess)
        rf = float(frame["rf"].iloc[0])
        net_return = rf + gross_excess - primary_cost.total
        denominator = 1.0 + net_return
        if denominator <= 0.0:
            raise RuntimeError("V6.2 portfolio NAV became nonpositive")
        end_drift = weights * (1.0 + excess + rf) / denominator
        long_gross = float(np.maximum(weights, 0.0).sum())
        short_gross = float(np.maximum(-weights, 0.0).sum())
        variant = f"{view}__{policy}"
        positions = frame[[*predecessor.KEYS, "sic2_industry_category", "return_source"]].copy()
        positions["weight"] = weights
        positions["previous_weight"] = previous
        positions["end_drift_weight"] = end_drift
        positions["desired_policy_weight"] = desired_aligned
        positions["stock_excess_return"] = excess
        positions = positions.loc[
            (positions["weight"] != 0.0)
            | (positions["previous_weight"] != 0.0)
            | (positions["desired_policy_weight"] != 0.0)
        ].reset_index(drop=True)
        summary = {
            "holding_month": month,
            "signal_date": str(pd.Timestamp(frame["signal_date"].iloc[0]).date()),
            "variant": variant,
            "policy": policy,
            "weighting": "direct_portfolio_policy",
            "feature_view": view,
            "selected_profile": profile_id,
            "volatility_target": float(portfolio["forecast_volatility_target"]),
            "nav_start": nav,
            "nav_end_primary": nav * (1.0 + net_return),
            "rf": rf,
            "gross_return": rf + gross_excess,
            "gross_excess_return": gross_excess,
            "net_return_primary": net_return,
            "net_excess_return_primary": gross_excess - primary_cost.total,
            "net_return_stress": rf + gross_excess - stress_cost.total,
            "net_excess_return_stress": gross_excess - stress_cost.total,
            "long_contribution": float(np.maximum(weights, 0.0) @ excess),
            "short_contribution": float(np.minimum(weights, 0.0) @ excess),
            "transaction_cost_primary": primary_cost.transaction,
            "borrow_cost_primary": primary_cost.borrow,
            "financing_cost_primary": primary_cost.financing,
            "total_cost_primary": primary_cost.total,
            "total_cost_stress": stress_cost.total,
            "forecast_volatility": repair.forecast_volatility,
            "long_gross": long_gross,
            "short_gross": short_gross,
            "gross": long_gross + short_gross,
            "net": float(weights.sum()),
            "turnover": repair.turnover,
            "active_long_holdings": int((weights > 1e-8).sum()),
            "active_short_holdings": int((weights < -1e-8).sum()),
            "eligible_long_candidates": int(policy_month.long_eligible.sum()),
            "eligible_short_candidates": int(policy_month.short_eligible.sum()),
            "learned_long_gross": learned_long,
            "learned_short_gross": learned_short,
            "implementation_scale": repair.scale,
            "maximum_weight": float(np.max(weights, initial=0.0)),
            "minimum_weight": float(np.min(weights, initial=0.0)),
            **{f"factor_exposure_{name}": float(repair.factor_exposures[i]) for i, name in enumerate(FACTOR_NAMES)},
        }
        nav = float(summary["nav_end_primary"])
        previous_map = {
            int(row.permno): float(row.end_drift_weight)
            for row in positions.itertuples(index=False)
            if abs(float(row.end_drift_weight)) > 1e-12
        }
        summaries.append(summary)
        if receipt_path and position_path:
            _atomic_parquet(position_path, positions)
            _atomic_json(receipt_path, {**(identity or {}), "summary": summary})
        print(
            f"DONE {variant} {month} net={summary['net_excess_return_primary']:.6f} "
            f"L={long_gross:.3f} S={short_gross:.3f} scale={repair.scale:.3f}",
            flush=True,
        )
    return summaries, previous_map, nav


def _profile_summary(frame: pd.DataFrame) -> dict[str, Any]:
    net = frame["net_excess_return_primary"].to_numpy(float)
    volatility = float(np.std(net, ddof=1) * np.sqrt(12.0))
    return {
        "feature_view": str(frame["feature_view"].iloc[0]),
        "policy": str(frame["policy"].iloc[0]),
        "profile_id": str(frame["selected_profile"].iloc[0]),
        "months": int(len(frame)),
        "annualized_net_excess_return": float(net.mean() * 12.0),
        "annualized_volatility": volatility,
        "net_sharpe": float(net.mean() / np.std(net, ddof=1) * np.sqrt(12.0)) if volatility > 0.0 else float("nan"),
        "annualized_cost": float(frame["total_cost_primary"].mean() * 12.0),
        "mean_turnover": float(frame["turnover"].mean()),
    }


def _run_validation(
    output: Path,
    identity: dict[str, str],
    dataset: Dataset,
    config: dict[str, Any],
    base_config: dict[str, Any],
    base_frames: dict[str, tuple[pd.DataFrame, np.ndarray]],
) -> pd.DataFrame:
    chronology = config["chronology"]
    validation_months = pd.period_range(
        chronology["validation_start"], chronology["validation_end"], freq="M"
    ).astype(str).tolist()
    cutoff = pd.Timestamp(base_frames[validation_months[0]][0]["signal_date"].iloc[0])
    summaries: list[dict[str, Any]] = []
    for view in config["feature_views"]:
        for policy in config["policies"]:
            for profile_offset, profile in enumerate(config["model"]["profiles"]):
                root = output / "validation" / view / policy / str(profile["id"])
                receipt_path = root / "receipt.json"
                model_path = root / "model.pt"
                monthly_path = root / "monthly.parquet"
                if receipt_path.is_file() and model_path.is_file() and monthly_path.is_file():
                    receipt = json.loads(receipt_path.read_text())
                    if any(receipt.get(key) != value for key, value in identity.items()):
                        raise RuntimeError("V6.2 validation checkpoint identity mismatch")
                    summaries.append(dict(receipt["summary"]))
                    continue
                print(f"FIT validation {view} {policy} {profile['id']}", flush=True)
                train_months = _fit_month_list(
                    dataset,
                    config,
                    view,
                    policy,
                    start=chronology["profile_fit_start"],
                    end=chronology["profile_fit_end"],
                    cutoff=cutoff,
                )
                model = build_policy(
                    config, len(dataset.views[view]), seed_offset=profile_offset
                )
                diagnostics = fit_policy(
                    model, train_months, profile=profile, **_fit_settings(config)
                )
                monthly, _, _ = _evaluate_block(
                    model,
                    dataset,
                    config,
                    base_config,
                    base_frames,
                    view,
                    policy,
                    str(profile["id"]),
                    validation_months,
                )
                monthly_frame = pd.DataFrame(monthly)
                summary = _profile_summary(monthly_frame)
                _atomic_torch(model_path, model.state_dict())
                _atomic_parquet(root / "training_diagnostics.parquet", pd.DataFrame(diagnostics))
                _atomic_parquet(monthly_path, monthly_frame)
                _atomic_json(receipt_path, {**identity, "summary": summary})
                summaries.append(summary)
                del model, train_months
                gc.collect()
    frame = pd.DataFrame(summaries).sort_values(
        ["feature_view", "policy", "net_sharpe"], ascending=[True, True, False]
    ).reset_index(drop=True)
    _atomic_parquet(output / "validation_profile_summary.parquet", frame)
    return frame


def _selected_profiles(frame: pd.DataFrame) -> dict[tuple[str, str], str]:
    selected: dict[tuple[str, str], str] = {}
    for (view, policy), group in frame.groupby(["feature_view", "policy"], observed=True):
        ordered = group.sort_values(
            ["net_sharpe", "profile_id"], ascending=[False, True], kind="mergesort"
        )
        selected[(str(view), str(policy))] = str(ordered.iloc[0]["profile_id"])
    return selected


def _profile(config: dict[str, Any], profile_id: str) -> dict[str, Any]:
    for item in config["model"]["profiles"]:
        if item["id"] == profile_id:
            return item
    raise KeyError(profile_id)


def _run_test(
    output: Path,
    identity: dict[str, str],
    dataset: Dataset,
    config: dict[str, Any],
    base_config: dict[str, Any],
    base_frames: dict[str, tuple[pd.DataFrame, np.ndarray]],
    selected: dict[tuple[str, str], str],
) -> pd.DataFrame:
    chronology = config["chronology"]
    all_rows: list[dict[str, Any]] = []
    for view_index, view in enumerate(config["feature_views"]):
        for policy_index, policy in enumerate(config["policies"]):
            profile_id = selected[(view, policy)]
            profile = _profile(config, profile_id)
            previous: dict[int, float] = {}
            nav = float(config["portfolio"]["nav_dollars"])
            for year, block_end in ((2024, "2024-12"), (2025, chronology["historical_test_end"])):
                block_start = f"{year}-01"
                months = pd.period_range(block_start, block_end, freq="M").astype(str).tolist()
                cutoff = pd.Timestamp(base_frames[months[0]][0]["signal_date"].iloc[0])
                root = output / "test" / view / policy / f"refit_{year}"
                model_path = root / "model.pt"
                receipt_path = root / "fit_receipt.json"
                if model_path.is_file() and receipt_path.is_file():
                    receipt = json.loads(receipt_path.read_text())
                    if any(receipt.get(key) != value for key, value in identity.items()):
                        raise RuntimeError("V6.2 refit checkpoint identity mismatch")
                    state = torch.load(model_path, map_location="cpu", weights_only=True)
                    model = _model_from_state(
                        config,
                        len(dataset.views[view]),
                        state,
                        seed_offset=100 + view_index * 10 + year,
                    )
                else:
                    print(f"FIT test {view} {policy} {profile_id} refit={year}", flush=True)
                    last_training_month = str(pd.Period(block_start, freq="M") - 1)
                    train_months = _fit_month_list(
                        dataset,
                        config,
                        view,
                        policy,
                        start=chronology["profile_fit_start"],
                        end=last_training_month,
                        cutoff=cutoff,
                    )
                    model = build_policy(
                        config,
                        len(dataset.views[view]),
                        seed_offset=100 + view_index * 10 + year,
                    )
                    diagnostics = fit_policy(
                        model, train_months, profile=profile, **_fit_settings(config)
                    )
                    _atomic_torch(model_path, model.state_dict())
                    _atomic_parquet(root / "training_diagnostics.parquet", pd.DataFrame(diagnostics))
                    _atomic_json(
                        receipt_path,
                        {
                            **identity,
                            "feature_view": view,
                            "policy": policy,
                            "profile_id": profile_id,
                            "refit_year": year,
                            "cutoff": str(cutoff.date()),
                            "training_months": len(train_months),
                        },
                    )
                    del train_months
                rows, previous, nav = _evaluate_block(
                    model,
                    dataset,
                    config,
                    base_config,
                    base_frames,
                    view,
                    policy,
                    profile_id,
                    months,
                    initial_previous=previous,
                    initial_nav=nav,
                    checkpoint_root=root / "months",
                    identity=identity,
                )
                all_rows.extend(rows)
                del model
                gc.collect()
    monthly = pd.DataFrame(all_rows).drop_duplicates(
        ["variant", "holding_month"], keep="last"
    ).sort_values(["variant", "holding_month"], kind="mergesort").reset_index(drop=True)
    _atomic_parquet(output / "monthly_results.parquet", monthly)
    summary = pd.DataFrame(
        [
            predecessor._aggregate_variant(group)
            | {
                "feature_view": str(group["feature_view"].iloc[0]),
                "selected_profile": str(group["selected_profile"].iloc[0]),
                "mean_learned_long_gross": float(group["learned_long_gross"].mean()),
                "mean_learned_short_gross": float(group["learned_short_gross"].mean()),
                "mean_implementation_scale": float(group["implementation_scale"].mean()),
            }
            for _, group in monthly.groupby("variant", observed=True, sort=True)
        ]
    ).sort_values("net_sharpe_primary", ascending=False, kind="mergesort").reset_index(drop=True)
    _atomic_parquet(output / "variant_summary.parquet", summary)
    return summary


def _comparison(output: Path, config: dict[str, Any], summary: pd.DataFrame) -> None:
    step02 = pd.read_parquet(
        ROOT
        / "artifacts/research_v6_1/portfolio_step_02/48f429eb0d7f-53a6425a3d92-7256b91cc393/variant_summary.parquet"
    )
    step02 = step02.loc[
        step02["variant"].eq("return_plus_both__optimized_10pct_vol")
    ].copy()
    step02["lineage"] = "v6_1_step_02_primary"
    step03_path = ROOT / str(config["bound_predecessor"]["step03_result"])
    step03 = pd.read_parquet(step03_path / "variant_summary.parquet").head(1).copy()
    step03["lineage"] = "v6_1_step_03_best"
    primary_names = {
        config["evaluation"]["primary_direct_weight"],
        config["evaluation"]["primary_direct_weight_plus_ff5"],
    }
    direct = summary.loc[summary["variant"].isin(primary_names)].copy()
    direct["lineage"] = "v6_2_direct_policy"
    shared = sorted(set(step02.columns) & set(step03.columns) & set(direct.columns))
    _atomic_parquet(
        output / "four_way_comparison.parquet",
        pd.concat([step02[shared], step03[shared], direct[shared]], ignore_index=True),
    )


def run(config_path: Path, output_root: Path, *, check_only: bool, smoke: bool) -> Path | None:
    (
        config,
        config_sha,
        input_identity,
        paths,
        base_config,
        base_paths,
    ) = verify_contract(config_path)
    code_identity = _code_identity()
    if check_only:
        print(
            f"READY: 3 views x 2 policies x 2 validation profiles; "
            f"portfolio-level objective; config={config_sha[:12]}",
            flush=True,
        )
        return None
    output = output_root / f"{config_sha[:12]}-{input_identity[:12]}-{code_identity[:12]}"
    if smoke:
        output = output / "smoke"
    identity = {
        "config_sha256": config_sha,
        "input_identity": input_identity,
        "code_identity": code_identity,
    }
    _atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_2_direct_portfolio_run_manifest_v1",
            "status": "RUNNING",
            **identity,
            "config": str(config_path.resolve()),
        },
    )
    try:
        dataset = load_dataset(config, paths)
        evaluation_base_config = json.loads(json.dumps(base_config))
        evaluation_base_config["evidence"]["portfolio_start"] = config["chronology"]["validation_start"]
        base_inputs = predecessor.load_inputs(evaluation_base_config, base_paths)
        evaluation_months = pd.period_range(
            config["chronology"]["validation_start"],
            config["chronology"]["historical_test_end"],
            freq="M",
        ).astype(str).tolist()
        if smoke:
            evaluation_months = ["2020-01", "2024-01"]
        base_frames = {
            month: _raw_base_frame(base_inputs, month)
            for month in evaluation_months
        }
        if smoke:
            profile = config["model"]["profiles"][0]
            cutoff = pd.Timestamp(base_frames["2020-01"][0]["signal_date"].iloc[0])
            months = _fit_month_list(
                dataset,
                config,
                "correlation_reduced",
                "direct_weight",
                start="2019-01",
                end="2019-03",
                cutoff=cutoff,
            )
            smoke_profile = {**profile, "epochs": 1}
            model = build_policy(config, len(dataset.views["correlation_reduced"]))
            diagnostics = fit_policy(
                model, months, profile=smoke_profile, **_fit_settings(config)
            )
            rows, _, _ = _evaluate_block(
                model,
                dataset,
                config,
                base_config,
                base_frames,
                "correlation_reduced",
                "direct_weight",
                str(profile["id"]),
                ["2024-01"],
            )
            _atomic_parquet(output / "training_diagnostics.parquet", pd.DataFrame(diagnostics))
            _atomic_parquet(output / "monthly_results.parquet", pd.DataFrame(rows))
            _atomic_json(output / "run_manifest.json", {"schema": "research_v6_2_direct_portfolio_run_manifest_v1", "status": "SMOKE_COMPLETE", **identity})
            print(f"SMOKE_COMPLETE {output}", flush=True)
            return output
        validation = _run_validation(
            output, identity, dataset, config, base_config, base_frames
        )
        selected = _selected_profiles(validation)
        _atomic_json(
            output / "selected_profiles.json",
            {
                f"{view}__{policy}": profile
                for (view, policy), profile in sorted(selected.items())
            },
        )
        summary = _run_test(
            output,
            identity,
            dataset,
            config,
            base_config,
            base_frames,
            selected,
        )
        _comparison(output, config, summary)
        _atomic_json(
            output / "run_manifest.json",
            {
                "schema": "research_v6_2_direct_portfolio_run_manifest_v1",
                "status": "COMPLETE",
                **identity,
                "config": str(config_path.resolve()),
                "validation_candidates": int(len(validation)),
                "completed_variants": int(len(summary)),
                "completed_variant_months": int(len(summary) * 20),
                "models_reused_from_v6_or_v6_1": False,
                "portfolio_level_objective": True,
            },
        )
        print(f"COMPLETE {output}", flush=True)
        return output
    except Exception as exc:
        _atomic_json(
            output / "error.json",
            {**identity, "error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.config.resolve(),
        arguments.output_root.resolve(),
        check_only=arguments.check_only,
        smoke=arguments.smoke,
    )
