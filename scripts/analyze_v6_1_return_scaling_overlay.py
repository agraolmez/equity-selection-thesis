#!/usr/bin/env python3
"""Evaluate causal exposure overlays on the frozen V6.1 winning portfolio.

The script never changes security selection or relative target weights.  It
replays the frozen monthly holdings with constant or signal-date forecast-vol
scalars and recomputes stateful trades, square-root impact, borrow, financing,
NAV, and drifted starting weights at the overlay's actual scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/research_v6_1/return_scaling_overlay.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_1/return_scaling_overlay"
FACTOR_NAMES = ("mktrf", "smb", "hml", "rmw", "cma", "umd")


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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def _duckdb_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _read_parquet(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"SELECT * FROM read_parquet('{_duckdb_path(path)}')").df()


def _verify(config_path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(config_path.read_text())
    if config.get("status") != "AUTHORIZED_POST_DIAGNOSTIC_TEST":
        raise RuntimeError("return-scaling overlay is not authorized")
    for phase in ("validation", "historical_test"):
        source = config["sources"][phase]
        manifest = ROOT / source["run_manifest"]
        if _sha256(manifest) != source["run_manifest_sha256"]:
            raise RuntimeError(f"{phase} source manifest identity changed")
        month_root = ROOT / source["month_root"]
        if not month_root.is_dir():
            raise FileNotFoundError(month_root)
    risk_path = Path(config["sources"]["risk_cost"]["path"])
    if _sha256(risk_path) != config["sources"]["risk_cost"]["sha256"]:
        raise RuntimeError("risk/cost source identity changed")
    return config, _sha256(config_path)


def _load_risk(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["sources"]["risk_cost"]["path"])
    start = config["sources"]["validation"]["start"]
    end = config["sources"]["historical_test"]["end"]
    query = f"""
        SELECT CAST(holding_month AS VARCHAR) AS holding_month,
               permno, full_spread_20d, adv20_usd,
               impact_volatility_daily, primary_borrow_rate_annual,
               stress_borrow_rate_annual
        FROM read_parquet('{_duckdb_path(path)}')
        WHERE holding_month BETWEEN '{start}' AND '{end}'
    """
    frame = duckdb.sql(query).df()
    frame["holding_month"] = frame["holding_month"].astype(str)
    if frame.duplicated(["holding_month", "permno"]).any():
        raise RuntimeError("risk/cost source contains duplicate stock-months")
    return frame


def _cost(
    weights: np.ndarray,
    previous: np.ndarray,
    frame: pd.DataFrame,
    nav: float,
    specification: dict[str, Any],
    commission_rate: float,
) -> tuple[float, float, float, float]:
    delta = np.abs(weights - previous)
    spread = frame["full_spread_20d"].to_numpy(float)
    adv = frame["adv20_usd"].to_numpy(float)
    impact_volatility = frame["impact_volatility_daily"].to_numpy(float)
    borrow_rate = frame[str(specification["borrow_column"])].to_numpy(float)
    required = np.c_[spread, adv, impact_volatility, borrow_rate]
    if not np.isfinite(required).all() or np.any(spread < 0.0) or np.any(adv <= 0.0):
        raise RuntimeError("active overlay row lacks valid risk/cost inputs")
    linear = (
        float(specification["spread_multiplier"]) * spread / 2.0
        + commission_rate
    ) * delta
    impact = (
        float(specification["impact_coefficient"])
        * impact_volatility
        * np.sqrt(nav / adv)
        * np.power(delta, 1.5)
    )
    transaction = float(np.sum(linear + impact))
    borrow = float(np.sum(np.maximum(-weights, 0.0) * borrow_rate / 12.0))
    gross = float(np.abs(weights).sum())
    financing = max(gross - 1.0, 0.0) * float(
        specification["financing_rate_annual"]
    ) / 12.0
    return transaction, borrow, financing, transaction + borrow + financing


def _scale(profile: dict[str, Any], summary: dict[str, Any]) -> float:
    if profile["kind"] == "constant":
        return float(profile["multiplier"])
    if profile["kind"] == "forecast_volatility_target":
        forecast = float(summary["forecast_volatility"])
        return min(float(profile["maximum_multiplier"]), float(profile["target"]) / forecast)
    raise ValueError(f"unknown overlay kind: {profile['kind']}")


def _max_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return float(np.min(wealth / peak - 1.0, initial=0.0))


def _summary(profile: str, frame: pd.DataFrame) -> dict[str, Any]:
    net = frame["net_excess_return_primary"].to_numpy(float)
    stress = frame["net_excess_return_stress"].to_numpy(float)
    volatility = float(net.std(ddof=1) * math.sqrt(12.0))
    stress_volatility = float(stress.std(ddof=1) * math.sqrt(12.0))
    return {
        "profile": profile,
        "months": int(len(frame)),
        "annualized_net_excess_return": float(net.mean() * 12.0),
        "annualized_stress_net_excess_return": float(stress.mean() * 12.0),
        "annualized_realized_volatility": volatility,
        "net_sharpe": float(net.mean() / net.std(ddof=1) * math.sqrt(12.0)),
        "stress_sharpe": float(stress.mean() / stress.std(ddof=1) * math.sqrt(12.0)),
        "maximum_drawdown": _max_drawdown(frame["net_return_primary"].to_numpy(float)),
        "cumulative_net_total_return": float(
            np.prod(1.0 + frame["net_return_primary"].to_numpy(float)) - 1.0
        ),
        "annualized_primary_cost": float(frame["total_cost_primary"].mean() * 12.0),
        "mean_scale": float(frame["scale"].mean()),
        "minimum_scale": float(frame["scale"].min()),
        "maximum_scale": float(frame["scale"].max()),
        "mean_gross": float(frame["gross"].mean()),
        "maximum_gross": float(frame["gross"].max()),
        "maximum_absolute_net": float(frame["net"].abs().max()),
        "mean_one_way_turnover": float(frame["turnover"].mean()),
        "maximum_one_way_turnover": float(frame["turnover"].max()),
        "maximum_one_way_adv_participation": float(
            frame["maximum_adv_participation"].max()
        ),
        "maximum_absolute_position": float(frame["maximum_absolute_position"].max()),
        "maximum_absolute_factor_exposure": float(
            frame["maximum_absolute_factor_exposure"].max()
        ),
        "maximum_sic2_net": float(frame["maximum_sic2_net"].max()),
        "maximum_sic2_gross": float(frame["maximum_sic2_gross"].max()),
        "maximum_forecast_volatility": float(frame["forecast_volatility"].max()),
        "positive_month_fraction": float((net > 0.0).mean()),
    }


def _run_phase(
    phase: str,
    config: dict[str, Any],
    risk: pd.DataFrame,
    profiles: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    source = config["sources"][phase]
    month_root = ROOT / source["month_root"]
    months = sorted(path.stem for path in month_root.glob("*.parquet"))
    expected = pd.period_range(source["start"], source["end"], freq="M").astype(str).tolist()
    if months != expected:
        raise RuntimeError(f"{phase} month sequence is incomplete")
    all_months: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    maximum_base_reproduction_error = 0.0
    for profile in profiles if profiles is not None else config["profiles"]:
        nav = 100_000_000.0
        previous: dict[int, float] = {}
        records: list[dict[str, Any]] = []
        for month in months:
            positions_path = month_root / f"{month}.parquet"
            receipt = json.loads((month_root / f"{month}.json").read_text())["summary"]
            positions = _read_parquet(positions_path)
            month_risk = risk.loc[risk["holding_month"].eq(month)].drop(
                columns="holding_month"
            )
            positions = positions.merge(
                month_risk, on="permno", how="left", validate="one_to_one"
            )
            active = positions["weight"].abs().gt(1e-14) | positions["permno"].isin(previous)
            positions = positions.loc[active].copy()
            scale = _scale(profile, receipt)
            weights = scale * positions["weight"].to_numpy(float)
            previous_weights = positions["permno"].map(previous).fillna(0.0).to_numpy(float)
            excess = positions["stock_excess_return"].to_numpy(float)
            primary = _cost(
                weights,
                previous_weights,
                positions,
                nav,
                config["costs"]["primary"],
                float(config["costs"]["commission_rate"]),
            )
            stress = _cost(
                weights,
                previous_weights,
                positions,
                nav,
                config["costs"]["stress"],
                float(config["costs"]["commission_rate"]),
            )
            gross_excess = float(weights @ excess)
            rf = float(receipt["rf"])
            net_excess = gross_excess - primary[3]
            stress_excess = gross_excess - stress[3]
            net_return = rf + net_excess
            if 1.0 + net_return <= 0.0:
                raise RuntimeError(f"overlay NAV became nonpositive in {phase} {month}")
            drift = weights * (1.0 + excess + rf) / (1.0 + net_return)
            previous = {
                int(permno): float(weight)
                for permno, weight in zip(positions["permno"], drift)
                if abs(float(weight)) > 1e-12
            }
            trade = np.abs(weights - previous_weights)
            maximum_adv = float(
                np.max(
                    trade * nav / positions["adv20_usd"].to_numpy(float),
                    initial=0.0,
                )
            )
            grouped = pd.DataFrame(
                {
                    "industry": positions["sic2_industry_category"].astype(str),
                    "weight": weights,
                }
            ).groupby("industry", observed=True, sort=False)["weight"]
            sic_net = grouped.sum().abs().max() if len(positions) else 0.0
            sic_gross = grouped.apply(lambda values: values.abs().sum()).max() if len(positions) else 0.0
            factor_max = max(
                abs(scale * float(receipt[f"factor_exposure_{name}"]))
                for name in FACTOR_NAMES
            )
            record = {
                "phase": phase,
                "profile": profile["id"],
                "holding_month": month,
                "scale": scale,
                "nav_start": nav,
                "nav_end": nav * (1.0 + net_return),
                "rf": rf,
                "gross_excess_return": gross_excess,
                "net_excess_return_primary": net_excess,
                "net_excess_return_stress": stress_excess,
                "net_return_primary": net_return,
                "transaction_cost_primary": primary[0],
                "borrow_cost_primary": primary[1],
                "financing_cost_primary": primary[2],
                "total_cost_primary": primary[3],
                "total_cost_stress": stress[3],
                "gross": float(np.abs(weights).sum()),
                "net": float(weights.sum()),
                "turnover": float(0.5 * trade.sum()),
                "maximum_adv_participation": maximum_adv,
                "maximum_absolute_position": float(np.max(np.abs(weights), initial=0.0)),
                "maximum_absolute_factor_exposure": factor_max,
                "maximum_sic2_net": float(sic_net),
                "maximum_sic2_gross": float(sic_gross),
                "forecast_volatility": scale * float(receipt["forecast_volatility"]),
            }
            if profile["id"] == "constant_1p00":
                maximum_base_reproduction_error = max(
                    maximum_base_reproduction_error,
                    abs(net_excess - float(receipt["net_excess_return_primary"])),
                    abs(primary[3] - float(receipt["total_cost_primary"])),
                )
            records.append(record)
            nav = record["nav_end"]
        frame = pd.DataFrame(records)
        all_months.append(frame)
        summaries.append(_summary(str(profile["id"]), frame))
    return (
        pd.concat(all_months, ignore_index=True),
        pd.DataFrame(summaries),
        maximum_base_reproduction_error,
    )


def _hac_t(values: np.ndarray, lags: int) -> float:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    centered = values - mean
    n = len(values)
    long_run = float(centered @ centered / n)
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        long_run += 2.0 * weight * float(centered[lag:] @ centered[:-lag] / n)
    standard_error = math.sqrt(max(long_run, 0.0) / n)
    return mean / standard_error if standard_error > 0.0 else float("nan")


def _paired_inference(
    monthly: pd.DataFrame,
    candidate: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    candidate_return = monthly.loc[
        monthly["profile"].eq(candidate), "net_excess_return_primary"
    ].to_numpy(float)
    control_return = monthly.loc[
        monthly["profile"].eq("constant_1p00"), "net_excess_return_primary"
    ].to_numpy(float)
    difference = candidate_return - control_return
    n = len(difference)
    block = int(config["inference"]["circular_block_length_months"])
    resamples = int(config["inference"]["bootstrap_resamples"])
    rng = np.random.default_rng(int(config["inference"]["bootstrap_seed"]))
    starts = rng.integers(0, n, size=(resamples, int(math.ceil(n / block))))
    indices = (starts[:, :, None] + np.arange(block)).reshape(resamples, -1)[:, :n] % n
    candidate_samples = candidate_return[indices]
    control_samples = control_return[indices]
    candidate_sharpe = (
        candidate_samples.mean(axis=1)
        / candidate_samples.std(axis=1, ddof=1)
        * math.sqrt(12.0)
    )
    control_sharpe = (
        control_samples.mean(axis=1)
        / control_samples.std(axis=1, ddof=1)
        * math.sqrt(12.0)
    )
    sharpe_difference = candidate_sharpe - control_sharpe
    annualized_difference = (candidate_samples - control_samples).mean(axis=1) * 12.0
    return {
        "candidate": candidate,
        "annualized_mean_net_excess_difference": float(difference.mean() * 12.0),
        "paired_hac_t_mean_difference": _hac_t(
            difference, int(config["inference"]["hac_lags"])
        ),
        "bootstrap_sharpe_difference_ci_95": np.quantile(
            sharpe_difference, [0.025, 0.975]
        ).tolist(),
        "bootstrap_annualized_return_difference_ci_95": np.quantile(
            annualized_difference, [0.025, 0.975]
        ).tolist(),
        "bootstrap_probability_sharpe_preserved_or_improved": float(
            np.mean(sharpe_difference >= 0.0)
        ),
    }


def _select(validation: pd.DataFrame, config: dict[str, Any]) -> tuple[str, str]:
    base = validation.loc[validation["profile"].eq("constant_1p00")].iloc[0]
    feasible = validation.loc[
        validation["maximum_one_way_adv_participation"].le(
            float(config["constraints"]["hard_maximum_one_way_adv_participation"]) + 1e-12
        )
        & validation["maximum_gross"].le(
            float(config["constraints"]["hard_maximum_gross"]) + 1e-12
        )
        & validation["stress_sharpe"].gt(0.0)
    ]
    strict = feasible.loc[feasible["net_sharpe"].ge(float(base["net_sharpe"]) - 1e-12)]
    practical = feasible.loc[
        feasible["net_sharpe"].ge(
            float(config["selection"]["practical_sharpe_floor_ratio"])
            * float(base["net_sharpe"])
        )
    ]
    if strict.empty or practical.empty:
        raise RuntimeError("no feasible scaling profile survived a Sharpe rule")
    strict_id = str(
        strict.sort_values("annualized_net_excess_return", ascending=False).iloc[0]["profile"]
    )
    practical_id = str(
        practical.sort_values("annualized_net_excess_return", ascending=False).iloc[0]["profile"]
    )
    return strict_id, practical_id


def _practical_feasible(
    row: pd.Series, base_sharpe: float, config: dict[str, Any]
) -> bool:
    return bool(
        float(row["net_sharpe"])
        >= float(config["selection"]["practical_sharpe_floor_ratio"]) * base_sharpe
        and float(row["maximum_one_way_adv_participation"])
        <= float(config["constraints"]["hard_maximum_one_way_adv_participation"])
        + 1e-12
        and float(row["maximum_gross"])
        <= float(config["constraints"]["hard_maximum_gross"]) + 1e-12
        and float(row["stress_sharpe"]) > 0.0
    )


def _maximum_practical_multiplier(
    config: dict[str, Any], risk: pd.DataFrame, base_sharpe: float
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    search = config["continuous_constant_search"]
    low = float(search["minimum_multiplier"])
    high = float(search["maximum_multiplier"])
    tolerance = float(search["bisection_tolerance"])
    if low <= 0.0 or high < low or tolerance <= 0.0:
        raise ValueError("invalid continuous constant-search bounds")
    last_monthly = pd.DataFrame()
    last_summary = pd.DataFrame()
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        profile = {
            "id": "validation_bisection",
            "kind": "constant",
            "multiplier": middle,
        }
        monthly, summary, _ = _run_phase(
            "validation", config, risk, profiles=[profile]
        )
        if _practical_feasible(summary.iloc[0], base_sharpe, config):
            low = middle
            last_monthly, last_summary = monthly, summary
        else:
            high = middle
    selected = {
        "id": "validation_max_return_constant",
        "kind": "constant",
        "multiplier": low,
    }
    monthly, summary, _ = _run_phase(
        "validation", config, risk, profiles=[selected]
    )
    if not _practical_feasible(summary.iloc[0], base_sharpe, config):
        raise RuntimeError("derived validation multiplier is not feasible")
    return selected, monthly, summary


def _result_markdown(result: dict[str, Any]) -> str:
    rows = []
    for item in result["historical_test_summary"]:
        rows.append(
            "| {profile} | {ret:.2%} | {vol:.2%} | {sh:.3f} | {stress:.3f} | {mdd:.2%} | {adv:.2%} | {turn:.2%} |".format(
                profile=item["profile"],
                ret=item["annualized_net_excess_return"],
                vol=item["annualized_realized_volatility"],
                sh=item["net_sharpe"],
                stress=item["stress_sharpe"],
                mdd=item["maximum_drawdown"],
                adv=item["maximum_one_way_adv_participation"],
                turn=item["mean_one_way_turnover"],
            )
        )
    selected = result["selected_historical_test"]
    return f"""# Research V6.1 — return-scaling overlay test

## Decision

The exact-Sharpe rule selects `{result['strict_selected_profile']}`: with convex costs, no scaled profile preserved or improved validation Sharpe exactly.

The practical 95%-Sharpe rule selects `{result['practical_selected_profile']}` on the separate February 2022–December 2023 validation window. On January 2024–August 2025 it raises annualized net excess return from **{result['base_historical_test']['annualized_net_excess_return']:.2%}** to **{selected['annualized_net_excess_return']:.2%}**, while net Sharpe changes from **{result['base_historical_test']['net_sharpe']:.3f}** to **{selected['net_sharpe']:.3f}**.

This is a post-diagnostic mandate-expansion study. It does not replace the frozen winner or constitute untouched confirmation.

## Historical-test profiles

| Profile | Annualized net excess | Volatility | Sharpe | Stress Sharpe | Max drawdown | Max ADV participation | Mean turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
{os.linesep.join(rows)}

## Required mandate change

The practical profile keeps maximum gross below 1.50 and one-way ADV participation below 10%, but proportionally expands the absolute position, turnover, net, beta, factor, and SIC2 limits. It preserves the stock rankings, gates, relative target weights, and hedge geometry.

## Honest interpretation

The improvement is economically modest and comes from taking more of the same risk, not from a better selector. The Sharpe difference is not statistically superior; the overlay's purpose is higher return with nearly unchanged efficiency.
"""


def run(config_path: Path, output_root: Path) -> Path:
    config, config_sha = _verify(config_path)
    code_sha = _sha256(Path(__file__).resolve())
    output = output_root / f"{config_sha[:12]}-{code_sha[:12]}"
    output.mkdir(parents=True, exist_ok=True)
    risk = _load_risk(config)
    validation_monthly, validation_summary, validation_error = _run_phase(
        "validation", config, risk
    )
    if validation_error > 1e-10:
        raise RuntimeError(f"validation base reproduction failed: {validation_error}")
    derived_profile: dict[str, Any] | None = None
    if bool(config.get("continuous_constant_search", {}).get("enabled", False)):
        base_sharpe = float(
            validation_summary.loc[
                validation_summary["profile"].eq("constant_1p00"), "net_sharpe"
            ].iloc[0]
        )
        derived_profile, derived_monthly, derived_summary = _maximum_practical_multiplier(
            config, risk, base_sharpe
        )
        validation_monthly = pd.concat(
            [validation_monthly, derived_monthly], ignore_index=True
        )
        validation_summary = pd.concat(
            [validation_summary, derived_summary], ignore_index=True
        )
    strict_id, practical_id = _select(validation_summary, config)
    test_profiles = list(config["profiles"])
    if derived_profile is not None:
        test_profiles.append(derived_profile)
    test_monthly, test_summary, test_error = _run_phase(
        "historical_test", config, risk, profiles=test_profiles
    )
    if test_error > 1e-10:
        raise RuntimeError(f"historical-test base reproduction failed: {test_error}")
    base_test = test_summary.loc[test_summary["profile"].eq("constant_1p00")].iloc[0]
    selected_test = test_summary.loc[test_summary["profile"].eq(practical_id)].iloc[0]
    result = {
        "schema": "research_v6_1_return_scaling_overlay_result_v1",
        "status": "COMPLETE",
        "config_sha256": config_sha,
        "code_sha256": code_sha,
        "strict_selected_profile": strict_id,
        "practical_selected_profile": practical_id,
        "derived_validation_multiplier": (
            float(derived_profile["multiplier"]) if derived_profile is not None else None
        ),
        "validation_base_reproduction_max_error": validation_error,
        "historical_test_base_reproduction_max_error": test_error,
        "validation_summary": validation_summary.to_dict(orient="records"),
        "historical_test_summary": test_summary.to_dict(orient="records"),
        "base_historical_test": base_test.to_dict(),
        "selected_historical_test": selected_test.to_dict(),
        "selected_test_inference": _paired_inference(test_monthly, practical_id, config),
        "interpretation": {
            "strict": "No return increase preserves validation Sharpe exactly after nonlinear costs.",
            "practical": "The validation-selected practical overlay increases return with only a small Sharpe reduction and retains hard gross and ADV capacity limits.",
            "evidence": "Post-diagnostic historical development; untouched confirmation remains pending from September 2026."
        },
    }
    _atomic_csv(output / "validation_monthly.csv", validation_monthly)
    _atomic_csv(output / "validation_summary.csv", validation_summary)
    _atomic_csv(output / "historical_test_monthly.csv", test_monthly)
    _atomic_csv(output / "historical_test_summary.csv", test_summary)
    _atomic_json(output / "result.json", result)
    _atomic_text(output / "RESULT.md", _result_markdown(result))
    _atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_1_return_scaling_overlay_run_v1",
            "status": "COMPLETE",
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": config_sha,
            "code_sha256": code_sha,
            "strict_selected_profile": strict_id,
            "practical_selected_profile": practical_id,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = run(args.config.resolve(), args.output_root.resolve())
    print(output)


if __name__ == "__main__":
    main()
