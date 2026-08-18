#!/usr/bin/env python3
"""Build the Stage 13 alpha-deterioration diagnostic and executed notebook.

This is a post-unblind diagnostic. It does not alter the frozen Stage 12 system,
the verified Stage 13 result, or the prespecified Stage 14 registry.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
S11 = Path(
    "artifacts/phase_05_validation/stage_11_execution_reports/"
    "stage11-report-87029cb260f808d825a33394"
)
S13 = Path(
    "artifacts/phase_06_final_test/stage_13_results/"
    "stage13-result-76f8a56ef41c0aab2a41173a"
)
VERIFIED_POINTER = Path("artifacts/phase_06_final_test/LATEST_STAGE_13_VERIFIED.json")
FROZEN_SYSTEM = Path("configs/phase_06_final_test/stage_12_final_system_v1.json")
STAGE14_REGISTRY = S13 / "robustness_registry_results.json"
OUT = ROOT / "reports/stage_13_alpha_diagnostic"
NOTEBOOK = ROOT / "notebooks/13_stage_13_alpha_diagnostic.ipynb"


def read_json(relative_path: Path) -> dict[str, Any]:
    with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256(relative_path: Path) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def annualized_sharpe(monthly_excess: pd.Series) -> float:
    values = pd.to_numeric(monthly_excess, errors="coerce").dropna()
    return float(np.sqrt(12.0) * values.mean() / values.std(ddof=1))


def fit_factor_model(frame: pd.DataFrame, y_column: str, factors: list[str]) -> dict[str, Any]:
    x = sm.add_constant(frame[factors].astype(float), has_constant="add")
    y = frame[y_column].astype(float)
    ols = sm.OLS(y, x).fit()
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=6)
    names = list(x.columns)
    params = dict(zip(names, ols.params, strict=True))
    hac_pvalues = dict(zip(names, hac.pvalues, strict=True))
    hac_tvalues = dict(zip(names, hac.tvalues, strict=True))
    hac_ci = dict(zip(names, hac.conf_int(alpha=0.05), strict=True))
    return {
        "monthly_alpha": float(params["const"]),
        "annualized_alpha": float(params["const"] * 12.0),
        "ols_alpha_p_value": float(ols.pvalues["const"]),
        "hac6_alpha_p_value": float(hac_pvalues["const"]),
        "hac6_alpha_t_stat": float(hac_tvalues["const"]),
        "hac6_alpha_95pct_ci_annualized": [
            float(hac_ci["const"][0] * 12.0),
            float(hac_ci["const"][1] * 12.0),
        ],
        "betas": {factor: float(params[factor]) for factor in factors},
        "annual_factor_means": {
            factor: float(frame[factor].mean() * 12.0) for factor in factors
        },
        "annual_factor_contributions": {
            factor: float(params[factor] * frame[factor].mean() * 12.0)
            for factor in factors
        },
        "annual_factor_contribution_total": float(
            sum(params[factor] * frame[factor].mean() * 12.0 for factor in factors)
        ),
        "r_squared": float(ols.rsquared),
        "observations": int(ols.nobs),
    }


def extract_monthly_ic() -> list[dict[str, Any]]:
    """Reduce the 948k-row calibration file to one authenticated row per month."""
    jq_program = r"""
      reduce (
        .records[]
        | select(
            .segment == "primary_full"
            and .strategy_id == "ensemble_hybrid_primary"
          )
      ) as $r (
        {};
        .[$r.holding_month] = {
          holding_month: $r.holding_month,
          monthly_spearman_ic: $r.monthly_spearman_ic,
          calibration_slope: $r.calibration_slope,
          refit_id: $r.refit_id,
          training_cutoff_month: $r.training_cutoff_month
        }
      )
      | [.[]]
      | sort_by(.holding_month)
    """
    result = subprocess.run(
        [
            "jq",
            jq_program,
            str(ROOT / S13 / "model_refit_forecast_calibration.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads(result.stdout)
    if len(rows) != 68:
        raise RuntimeError(f"Expected 68 monthly IC rows, found {len(rows)}")
    return rows


def source(
    source_id: str,
    label: str,
    relative_path: Path,
    description: str,
    metric_definitions: list[str] | None = None,
) -> dict[str, Any]:
    path = relative_path.as_posix()
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "description": description,
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_json_auto('{path}')",
            "tables_used": [path],
            "metric_definitions": metric_definitions or [],
        },
    }


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)

    s11_summary = read_json(S11 / "summary.json")
    required = read_json(S13 / "required_metrics.json")
    monthly_payload = read_json(S13 / "monthly_accounting.json")
    factor_payload = read_json(S13 / "factors_benchmarks.json")
    cost_payload = read_json(S13 / "costs_constraints_materiality.json")
    inference = read_json(S13 / "inference_results.json")
    verified = read_json(VERIFIED_POINTER)
    stage14 = read_json(STAGE14_REGISTRY)
    monthly_ic_rows = extract_monthly_ic()

    old_metric = next(
        row
        for row in s11_summary["strategy_metrics"]
        if row["strategy_id"] == "ensemble_hybrid_primary"
    )
    primary_required = next(
        row
        for row in required["metric_rows"]
        if row["segment"] == "primary_full"
        and row["strategy_id"] == "ensemble_hybrid_primary"
    )

    monthly = pd.DataFrame(
        [
            row
            for row in monthly_payload["records"]
            if row["segment"] == "primary_full"
            and row["strategy_id"] == "ensemble_hybrid_primary"
        ]
    ).sort_values("holding_month")
    factors = pd.DataFrame(
        [
            row
            for row in factor_payload["factor_rows"]
            if row["segment"] == "primary_full"
        ]
    ).sort_values("holding_month")
    costs = pd.DataFrame(
        [
            row
            for row in cost_payload["records"]
            if row["segment"] == "primary_full"
            and row["strategy_id"] == "ensemble_hybrid_primary"
        ]
    ).sort_values("holding_month")
    ic = pd.DataFrame(monthly_ic_rows).sort_values("holding_month")

    if not (len(monthly) == len(factors) == len(costs) == len(ic) == 68):
        raise RuntimeError(
            "Stage 13 primary monthly evidence did not reconcile to 68 rows: "
            f"monthly={len(monthly)}, factors={len(factors)}, costs={len(costs)}, ic={len(ic)}"
        )

    merged = (
        monthly.merge(
            factors[
                ["holding_month", "mktrf", "smb", "hml", "rmw", "cma", "umd"]
            ],
            on="holding_month",
            validate="one_to_one",
        )
        .merge(
            costs[
                [
                    "holding_month",
                    "commission_usd",
                    "spread_usd",
                    "impact_usd",
                    "borrow_usd",
                    "financing_usd",
                    "maximum_adv_trade_fraction",
                ]
            ],
            on="holding_month",
            validate="one_to_one",
        )
        .merge(
            ic[["holding_month", "monthly_spearman_ic", "calibration_slope", "refit_id"]],
            on="holding_month",
            validate="one_to_one",
        )
        .sort_values("holding_month")
        .reset_index(drop=True)
    )

    merged["gross_excess_return"] = merged["gross_return"] - merged["risk_free_return"]
    merged["net_excess_return"] = merged["net_return"] - merged["risk_free_return"]
    merged["trading_cost_return"] = (
        merged["commission_usd"] + merged["spread_usd"] + merged["impact_usd"]
    ) / merged["start_nav_usd"]
    merged["borrow_cost_return"] = merged["borrow_usd"] / merged["start_nav_usd"]
    merged["financing_cost_return"] = merged["financing_usd"] / merged["start_nav_usd"]
    merged["total_cost_return"] = (
        merged["trading_cost_return"]
        + merged["borrow_cost_return"]
        + merged["financing_cost_return"]
    )
    cost_reconciliation_error = (
        merged["gross_return"] - merged["net_return"] - merged["total_cost_return"]
    ).abs()
    if float(cost_reconciliation_error.max()) > 1e-12:
        raise RuntimeError(
            "Gross-to-net cost reconciliation failed: "
            f"{float(cost_reconciliation_error.max()):.3e}"
        )

    factor_names = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
    gross_factor = fit_factor_model(merged, "gross_excess_return", factor_names)
    net_factor = fit_factor_model(merged, "net_excess_return", factor_names)
    capm = fit_factor_model(merged, "net_excess_return", ["mktrf"])

    ic_fit = sm.OLS(
        merged["monthly_spearman_ic"].astype(float),
        np.ones((len(merged), 1)),
    ).fit()
    ic_hac = ic_fit.get_robustcov_results(cov_type="HAC", maxlags=6)
    ic_q = pd.qcut(
        merged["monthly_spearman_ic"],
        q=4,
        labels=["Q1 low", "Q2", "Q3", "Q4 high"],
    )

    cost_drag = {
        "trading": float(12.0 * merged["trading_cost_return"].mean()),
        "borrow": float(12.0 * merged["borrow_cost_return"].mean()),
        "financing": float(12.0 * merged["financing_cost_return"].mean()),
        "total": float(12.0 * merged["total_cost_return"].mean()),
    }
    cost_counterfactuals = []
    scenarios = {
        "Reported net": merged["net_excess_return"],
        "Remove trading only": merged["net_excess_return"] + merged["trading_cost_return"],
        "Remove borrow only": merged["net_excess_return"] + merged["borrow_cost_return"],
        "Remove financing only": merged["net_excess_return"]
        + merged["financing_cost_return"],
        "Halve all explicit costs": merged["net_excess_return"]
        + 0.5 * merged["total_cost_return"],
        "Remove all explicit costs": merged["gross_excess_return"],
    }
    for scenario, returns in scenarios.items():
        cost_counterfactuals.append(
            {
                "scenario": scenario,
                "annualized_excess_return": float(12.0 * returns.mean()),
                "annualized_volatility": float(np.sqrt(12.0) * returns.std(ddof=1)),
                "sharpe": annualized_sharpe(returns),
                "classification": "diagnostic_post_hoc",
            }
        )

    old_total_return = float(old_metric["mean_monthly_net_total_return"] * 12.0)
    old_rf = float(
        old_total_return - old_metric["annualized_arithmetic_net_excess_return"]
    )
    new_net_total = float(12.0 * merged["net_return"].mean())
    new_rf = float(12.0 * merged["risk_free_return"].mean())
    old_new = [
        {
            "period": "Stage 11 validation",
            "holding_window": "2015-01 to 2019-11",
            "months": int(old_metric["holding_months"]),
            "status": "selection diagnostic; not confirmatory",
            "sharpe": float(old_metric["fully_net_sharpe"]),
            "annualized_net_excess_return": float(
                old_metric["annualized_arithmetic_net_excess_return"]
            ),
            "annualized_volatility": float(
                old_metric["annualized_net_excess_volatility"]
            ),
            "annualized_net_total_return": old_total_return,
            "annualized_risk_free_return": old_rf,
            "mean_rank_ic": float(s11_summary["selection"]["ensemble"]["mean_rank_ic"]),
            "mean_gross": float(old_metric["mean_gross_exposure"]),
            "mean_turnover": float(old_metric["mean_monthly_turnover"]),
            "total_explicit_cost_usd": float(
                old_metric["total_borrow_cost_dollars"]
                + old_metric["total_financing_cost_dollars"]
                + old_metric["total_transaction_cost_dollars"]
            ),
        },
        {
            "period": "Stage 13 final",
            "holding_window": "2020-01 to 2025-08",
            "months": int(len(merged)),
            "status": "verified untouched final-period result",
            "sharpe": float(primary_required["metrics"]["sharpe"]),
            "annualized_net_excess_return": float(
                primary_required["metrics"]["arithmetic_annualized_net_excess_return"]
            ),
            "annualized_volatility": float(
                primary_required["metrics"]["annualized_volatility"]
            ),
            "annualized_net_total_return": new_net_total,
            "annualized_risk_free_return": new_rf,
            "mean_rank_ic": float(primary_required["metrics"]["mean_monthly_spearman_ic"]),
            "mean_gross": float(
                primary_required["metrics"]["exposures_and_concentration"]["mean_gross"]
            ),
            "mean_turnover": float(
                primary_required["metrics"]["liquidity_and_capacity"][
                    "mean_one_way_turnover"
                ]
            ),
            "total_explicit_cost_usd": float(
                primary_required["metrics"]["gross_to_net_drag"]["total_usd"]
            ),
        },
    ]

    alpha_bridge = [
        {
            "component": "Gross FF6 alpha",
            "annualized_return": gross_factor["annualized_alpha"],
            "kind": "positive",
            "order": 1,
        },
        {
            "component": "Factor contribution",
            "annualized_return": gross_factor["annual_factor_contribution_total"],
            "kind": "negative",
            "order": 2,
        },
        {
            "component": "Gross excess return",
            "annualized_return": float(12.0 * merged["gross_excess_return"].mean()),
            "kind": "subtotal",
            "order": 3,
        },
        {
            "component": "Explicit costs",
            "annualized_return": -cost_drag["total"],
            "kind": "negative",
            "order": 4,
        },
        {
            "component": "Net excess return",
            "annualized_return": float(12.0 * merged["net_excess_return"].mean()),
            "kind": "result",
            "order": 5,
        },
    ]

    cumulative = []
    index_levels = {"Gross before explicit costs": 100.0, "Net": 100.0, "Cash": 100.0}
    for row in merged.itertuples(index=False):
        month_date = f"{row.holding_month}-01"
        index_levels["Gross before explicit costs"] *= 1.0 + float(row.gross_return)
        index_levels["Net"] *= 1.0 + float(row.net_return)
        index_levels["Cash"] *= 1.0 + float(row.risk_free_return)
        for series, value in index_levels.items():
            cumulative.append(
                {"holding_month": month_date, "series": series, "wealth_index": value}
            )

    constraint_rows = [
        {
            "constraint": "Turnover at 50% cap",
            "binding_months": int((merged["turnover"] >= 0.5 - 5e-7).sum()),
            "total_months": 68,
        },
        {
            "constraint": "Gross at 2.0x cap",
            "binding_months": int((merged["gross_exposure"] >= 2.0 - 1e-6).sum()),
            "total_months": 68,
        },
        {
            "constraint": "Ex-ante beta at -0.10 bound",
            "binding_months": int((merged["market_beta"] <= -0.1 + 1e-6).sum()),
            "total_months": 68,
        },
        {
            "constraint": "ADV ceiling hit",
            "binding_months": int(
                (merged["maximum_adv_trade_fraction"] >= 0.1 - 1e-6).sum()
            ),
            "total_months": 68,
        },
    ]
    for row in constraint_rows:
        row["binding_rate"] = row["binding_months"] / row["total_months"]

    annual_rows = []
    annual = merged.assign(year=merged["holding_month"].str[:4]).groupby("year")
    for year, frame in annual:
        annual_rows.append(
            {
                "year": int(year),
                "months": int(len(frame)),
                "gross_excess_return": float(12.0 * frame["gross_excess_return"].mean()),
                "net_excess_return": float(12.0 * frame["net_excess_return"].mean()),
                "explicit_cost_drag": float(12.0 * frame["total_cost_return"].mean()),
                "net_sharpe": annualized_sharpe(frame["net_excess_return"]),
                "mean_ic": float(frame["monthly_spearman_ic"].mean()),
                "mean_ex_ante_beta": float(frame["market_beta"].mean()),
                "market_factor_return": float(12.0 * frame["mktrf"].mean()),
            }
        )

    signal_periods = [
        {
            "period": "2015–2019 validation",
            "mean_ic": float(s11_summary["selection"]["ensemble"]["mean_rank_ic"]),
            "period_order": 0,
        }
    ]
    for row in annual_rows:
        signal_periods.append(
            {
                "period": str(row["year"]),
                "mean_ic": row["mean_ic"],
                "period_order": row["year"] - 2019,
            }
        )

    strategy_labels = {
        "ensemble_hybrid_primary": "Hybrid primary",
        "elastic_net_hybrid_challenger": "Elastic Net",
        "ensemble_rank_challenger": "Rank challenger",
        "trailing_residual_mean_markowitz": "Residual Markowitz",
        "fixed_value_quality_momentum": "Fixed VQM",
        "equal_weight_long_only_scoring_universe": "Equal weight",
    }
    strategy_rows = []
    for row in required["metric_rows"]:
        if row["segment"] != "primary_full":
            continue
        metrics = row["metrics"]
        alpha_payload = metrics.get("ff5_plus_momentum_alpha_and_betas", {})
        strategy_rows.append(
            {
                "strategy_id": row["strategy_id"],
                "strategy": strategy_labels.get(row["strategy_id"], row["strategy_id"]),
                "sharpe": float(metrics["sharpe"]),
                "annualized_net_excess_return": float(
                    metrics["arithmetic_annualized_net_excess_return"]
                ),
                "ff6_alpha": (
                    float(alpha_payload["annualized_alpha"])
                    if "annualized_alpha" in alpha_payload
                    else None
                ),
                "factor_information_ratio": (
                    float(metrics["factor_adjusted_information_ratio"])
                    if metrics.get("factor_adjusted_information_ratio") is not None
                    else None
                ),
                "total_explicit_cost_usd": float(
                    metrics["gross_to_net_drag"]["total_usd"]
                ),
            }
        )

    alpha_evidence = [
        {
            "specification": "Gross FF5 + momentum",
            "annualized_alpha": gross_factor["annualized_alpha"],
            "hac6_p_value": gross_factor["hac6_alpha_p_value"],
            "interpretation": "Suggestive residual return before explicit costs; not significant.",
            "status": "post-hoc diagnostic",
        },
        {
            "specification": "Net FF5 + momentum",
            "annualized_alpha": net_factor["annualized_alpha"],
            "hac6_p_value": net_factor["hac6_alpha_p_value"],
            "interpretation": "Positive point estimate, wide uncertainty; not confirmed.",
            "status": "post-hoc p-value; official point estimate",
        },
        {
            "specification": "Net CAPM",
            "annualized_alpha": capm["annualized_alpha"],
            "hac6_p_value": capm["hac6_alpha_p_value"],
            "interpretation": "Near zero; alpha is factor-specification dependent.",
            "status": "post-hoc diagnostic",
        },
    ]

    diagnostic_summary = {
        "classification": "SHARE_WITH_CAVEATS",
        "stage13_primary_status": verified.get("status", "STAGE13_VERIFIED"),
        "stage13_difference_count": int(verified.get("difference_count", 0)),
        "stage14_status": "NOT_STARTED",
        "alpha_conclusion": (
            "Suggestive gross residual stock-selection alpha exists, but neither gross "
            "nor net alpha is statistically confirmed. The selected primary failed to "
            "monetize the signal after factor tilts and explicit costs."
        ),
        "old_validation_sharpe": float(old_metric["fully_net_sharpe"]),
        "final_sharpe": float(primary_required["metrics"]["sharpe"]),
        "ic_decline_fraction": float(
            1.0
            - primary_required["metrics"]["mean_monthly_spearman_ic"]
            / s11_summary["selection"]["ensemble"]["mean_rank_ic"]
        ),
        "gross_excess_return": float(12.0 * merged["gross_excess_return"].mean()),
        "gross_sharpe": annualized_sharpe(merged["gross_excess_return"]),
        "net_excess_return": float(12.0 * merged["net_excess_return"].mean()),
        "net_sharpe": annualized_sharpe(merged["net_excess_return"]),
        "cost_drag": cost_drag,
        "factor_models": {
            "gross_ff6": gross_factor,
            "net_ff6": net_factor,
            "net_capm": capm,
        },
        "signal_diagnostics": {
            "mean_ic": float(merged["monthly_spearman_ic"].mean()),
            "median_ic": float(merged["monthly_spearman_ic"].median()),
            "positive_month_share": float((merged["monthly_spearman_ic"] > 0).mean()),
            "monthly_ic_std": float(merged["monthly_spearman_ic"].std(ddof=1)),
            "ols_mean_ic_t_stat": float(ic_fit.tvalues.iloc[0]),
            "ols_mean_ic_p_value": float(ic_fit.pvalues.iloc[0]),
            "hac6_mean_ic_t_stat": float(ic_hac.tvalues[0]),
            "hac6_mean_ic_p_value": float(ic_hac.pvalues[0]),
            "annualized_ic_information_ratio": float(
                np.sqrt(12.0)
                * merged["monthly_spearman_ic"].mean()
                / merged["monthly_spearman_ic"].std(ddof=1)
            ),
            "ic_gross_excess_correlation": float(
                merged["monthly_spearman_ic"].corr(merged["gross_excess_return"])
            ),
            "low_ic_quartile_mean_monthly_gross_excess": float(
                merged.loc[ic_q == "Q1 low", "gross_excess_return"].mean()
            ),
            "high_ic_quartile_mean_monthly_gross_excess": float(
                merged.loc[ic_q == "Q4 high", "gross_excess_return"].mean()
            ),
            "mean_calibration_slope": float(merged["calibration_slope"].mean()),
            "calibration_slope_ic_correlation": float(
                merged["calibration_slope"].corr(merged["monthly_spearman_ic"])
            ),
            "calibration_slope_gross_return_correlation": float(
                merged["calibration_slope"].corr(merged["gross_excess_return"])
            ),
            "refit_count": int(merged["refit_id"].nunique()),
        },
        "allocator_diagnostics": {
            "mean_ex_ante_beta": float(merged["market_beta"].mean()),
            "realized_ff6_market_beta": float(net_factor["betas"]["mktrf"]),
            "negative_beta_months": int((merged["market_beta"] < 0).sum()),
            "constraint_bindings": constraint_rows,
        },
        "selection_bias": {
            "raw_dsr_trial_count": int(inference["raw_dsr_trial_count"]),
            "old_validation_months": int(old_metric["holding_months"]),
            "interpretation": (
                "The 1.537 Sharpe was the winner from a broad development/validation "
                "search. Stage 14 multiple-testing correction is required before any "
                "alpha claim."
            ),
        },
        "cost_counterfactuals": cost_counterfactuals,
    }

    analysis_metadata = [
        {
            "item": "Evidence status",
            "value": "Stage 13 verified with zero differences; Stage 14 not started",
        },
        {
            "item": "Primary final window",
            "value": "2020-01 through 2025-08 (68 months)",
        },
        {
            "item": "Inference warning",
            "value": "Regression p-values in this report are post-hoc diagnostics",
        },
        {
            "item": "Decision",
            "value": "Do not claim confirmed alpha; run the frozen Stage 14 registry",
        },
    ]

    datasets = {
        "analysis_metadata": analysis_metadata,
        "old_vs_final": old_new,
        "alpha_bridge": alpha_bridge,
        "cumulative_performance": cumulative,
        "constraint_bindings": constraint_rows,
        "annual_diagnostics": annual_rows,
        "signal_by_period": signal_periods,
        "strategy_comparison": strategy_rows,
        "alpha_evidence": alpha_evidence,
        "cost_counterfactuals": cost_counterfactuals,
        "monthly_diagnostics": clean(
            merged[
                [
                    "holding_month",
                    "gross_return",
                    "net_return",
                    "risk_free_return",
                    "gross_excess_return",
                    "net_excess_return",
                    "total_cost_return",
                    "monthly_spearman_ic",
                    "calibration_slope",
                    "gross_exposure",
                    "turnover",
                    "market_beta",
                    "maximum_adv_trade_fraction",
                    *factor_names,
                ]
            ].to_dict(orient="records")
        ),
        "summary_metrics": [
            {"metric": "Old validation Sharpe", "value": diagnostic_summary["old_validation_sharpe"]},
            {"metric": "Final-period Sharpe", "value": diagnostic_summary["final_sharpe"]},
            {"metric": "Gross annualized excess return", "value": diagnostic_summary["gross_excess_return"]},
            {"metric": "Net annualized excess return", "value": diagnostic_summary["net_excess_return"]},
            {"metric": "Annualized explicit cost drag", "value": cost_drag["total"]},
            {"metric": "Gross FF6 alpha", "value": gross_factor["annualized_alpha"]},
            {"metric": "Net FF6 alpha", "value": net_factor["annualized_alpha"]},
            {"metric": "Raw DSR trial count", "value": int(inference["raw_dsr_trial_count"])},
        ],
    }
    snapshot = {"version": 1, "status": "ready", "datasets": clean(datasets)}

    sources = [
        source(
            "stage11_verified_summary",
            "Verified Stage 11 validation-selection summary",
            S11 / "summary.json",
            "Supplies the 2015-01 to 2019-11 selected validation metrics and their non-confirmatory classification.",
            [
                "Fully net Sharpe = annualized arithmetic net excess return / annualized net excess volatility."
            ],
        ),
        source(
            "stage13_verified_pointer",
            "Verified Stage 13 authority pointer",
            VERIFIED_POINTER,
            "Binds the final-period primary result to its independent zero-difference verification.",
        ),
        source(
            "stage13_required_metrics",
            "Stage 13 required metrics",
            S13 / "required_metrics.json",
            "Supplies official Stage 13 point estimates for the primary and comparator strategies.",
        ),
        source(
            "stage13_monthly_accounting",
            "Stage 13 monthly accounting",
            S13 / "monthly_accounting.json",
            "Supplies the 68 primary monthly gross, net, cash, exposure, turnover, and beta observations.",
            [
                "Gross excess return = gross return - risk-free return.",
                "Net excess return = net return - risk-free return.",
            ],
        ),
        source(
            "stage13_factors",
            "Stage 13 factor and benchmark returns",
            S13 / "factors_benchmarks.json",
            "Supplies monthly FF5 plus momentum returns used in the post-hoc diagnostic regressions.",
        ),
        source(
            "stage13_costs",
            "Stage 13 cost, constraint, and materiality records",
            S13 / "costs_constraints_materiality.json",
            "Supplies monthly commission, spread, impact, borrow, financing, and constraint observations.",
            [
                "Annualized cost drag = 12 times the mean monthly component cost divided by start-of-month NAV."
            ],
        ),
        source(
            "stage13_calibration",
            "Stage 13 model refit, forecast, and calibration records",
            S13 / "model_refit_forecast_calibration.json",
            "Supplies monthly Spearman IC and calibration slopes; reduced to one repeated-value record per holding month.",
        ),
        source(
            "stage13_inference",
            "Stage 13 inference inputs and trial count",
            S13 / "inference_results.json",
            "Supplies the frozen raw DSR trial count used to frame selection-bias risk.",
        ),
        source(
            "stage12_frozen_system",
            "Frozen Stage 12 final system",
            FROZEN_SYSTEM,
            "Supplies the immutable primary model, allocator, factor-bound, refit, and cost settings.",
        ),
        source(
            "stage14_registry",
            "Prespecified Stage 14 robustness registry",
            STAGE14_REGISTRY,
            "Supplies the still-pending prespecified robustness and sensitivity tests.",
        ),
        source(
            "analysis_snapshot",
            "Bounded Stage 13 alpha diagnostic snapshot",
            Path("reports/stage_13_alpha_diagnostic/analysis_snapshot.json"),
            "Contains the bounded datasets and post-hoc calculations rendered in this report.",
        ),
    ]

    pct = lambda x: f"{100.0 * x:.2f}%"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Stage 13 Alpha Diagnostic: Why the 1.54 Sharpe Did Not Hold",
        "description": (
            "A verified old-versus-final reconciliation, gross-to-net alpha bridge, "
            "signal and allocator diagnosis, and prespecified improvement path."
        ),
        "filters": [],
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "layout": "full",
                "body": "# Stage 13 Alpha Diagnostic\n\n## Why the 1.54 validation Sharpe did not survive 2020–2025",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "layout": "full",
                "sourceId": "stage13_verified_pointer",
                "body": (
                    "## Technical Summary\n\n"
                    "**Bottom line:** the final system did not produce confirmed investable alpha. "
                    "There is a **suggestive gross FF5+momentum residual alpha of "
                    f"{pct(gross_factor['annualized_alpha'])}**, but its HAC(6) p-value is "
                    f"{gross_factor['hac6_alpha_p_value']:.3f}. Persistent factor tilts reduce "
                    f"gross excess return to {pct(12.0 * merged['gross_excess_return'].mean())}; "
                    f"{pct(cost_drag['total'])} of annual explicit cost drag then leaves "
                    f"{pct(12.0 * merged['net_excess_return'].mean())} net excess return and "
                    f"a Sharpe of {annualized_sharpe(merged['net_excess_return']):.3f}. "
                    "Stage 13 is verified with zero differences; Stage 14 inference remains pending."
                ),
            },
            {
                "id": "old_new_finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "stage11_verified_summary",
                "body": (
                    "## The apparent collapse is real, but the two numbers had different evidentiary status\n\n"
                    "The remembered 1.58 is the verified Stage 11 estimate of 1.537 over 59 "
                    "selected validation months. It was explicitly labeled a selection diagnostic, "
                    "not confirmatory out-of-sample evidence. The Stage 13 result is the untouched "
                    "68-month final-period test. Mean IC fell by "
                    f"{100.0 * diagnostic_summary['ic_decline_fraction']:.1f}%, while volatility "
                    "rose and the risk-free hurdle increased."
                ),
            },
            {
                "id": "old_new_table_block",
                "type": "table",
                "layout": "full",
                "tableId": "old_new_table",
            },
            {
                "id": "bridge_finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "analysis_snapshot",
                "body": (
                    "## Where the economics went\n\n"
                    "The clean bridge is: gross FF6 residual alpha, minus adverse factor "
                    "contributions, minus explicit trading/borrow/financing costs. Even a "
                    "zero-cost counterfactual reaches only a 0.20 Sharpe, so bypassing cost "
                    "validators or deleting cost assumptions cannot restore the old result."
                ),
            },
            {
                "id": "alpha_bridge_chart_block",
                "type": "chart",
                "layout": "full",
                "chartId": "alpha_bridge_chart",
            },
            {
                "id": "cost_table_block",
                "type": "table",
                "layout": "full",
                "tableId": "cost_counterfactual_table",
            },
            {
                "id": "signal_finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "stage13_calibration",
                "body": (
                    "## The signal weakened and became unstable\n\n"
                    f"Mean monthly rank IC fell from {s11_summary['selection']['ensemble']['mean_rank_ic']:.4f} "
                    f"to {merged['monthly_spearman_ic'].mean():.4f}. Only "
                    f"{100.0 * (merged['monthly_spearman_ic'] > 0).mean():.1f}% of final months "
                    f"had positive IC, and the HAC(6) mean-IC p-value is {float(ic_hac.pvalues[0]):.3f}. "
                    f"IC still correlates {merged['monthly_spearman_ic'].corr(merged['gross_excess_return']):.2f} "
                    "with gross portfolio excess return, so the signal is not pure noise; it is "
                    "too weak and inconsistent for the frozen exposure/cost structure."
                ),
            },
            {
                "id": "signal_chart_block",
                "type": "chart",
                "layout": "full",
                "chartId": "signal_chart",
            },
            {
                "id": "allocator_finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "stage13_costs",
                "body": (
                    "## The allocator behaved like a corner solution\n\n"
                    f"Turnover hit its 50% cap in {constraint_rows[0]['binding_months']}/68 months, "
                    f"gross exposure hit 2.0x in {constraint_rows[1]['binding_months']}/68, and "
                    f"ex-ante beta sat at the -0.10 bound in {constraint_rows[2]['binding_months']}/68. "
                    f"The realized FF6 market beta was {net_factor['betas']['mktrf']:.3f}, more "
                    "negative than the mean ex-ante estimate. The system therefore spent heavily "
                    "and maintained maximum exposure even when expected edge was weak."
                ),
            },
            {
                "id": "constraint_chart_block",
                "type": "chart",
                "layout": "full",
                "chartId": "constraint_chart",
            },
            {
                "id": "performance_finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "stage13_monthly_accounting",
                "body": (
                    "## The failure concentrated in specific regimes\n\n"
                    "2021 and 2024 were the main losses. The negative-beta posture sometimes "
                    "helped when the market fell, as in 2022, but hurt during strong market years. "
                    "IC alone did not translate cleanly into portfolio economics because factor "
                    "tilts, binding constraints, and costs dominated the conversion layer."
                ),
            },
            {
                "id": "cumulative_chart_block",
                "type": "chart",
                "layout": "full",
                "chartId": "cumulative_chart",
            },
            {
                "id": "annual_table_block",
                "type": "table",
                "layout": "full",
                "tableId": "annual_table",
            },
            {
                "id": "alpha_finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "stage13_factors",
                "body": (
                    "## Did we find alpha?\n\n"
                    "**A hint, yes; confirmation, no.** Gross FF6 residual alpha is positive but "
                    "not statistically significant. Net FF6 alpha is also positive, but CAPM "
                    "alpha is near zero, showing specification sensitivity. With 386 recorded "
                    "raw trials and only 59 old validation months, multiple-testing correction "
                    "is essential before making a thesis-level alpha claim."
                ),
            },
            {
                "id": "alpha_table_block",
                "type": "table",
                "layout": "full",
                "tableId": "alpha_table",
            },
            {
                "id": "strategy_finding",
                "type": "markdown",
                "layout": "full",
                "sourceId": "stage13_required_metrics",
                "body": (
                    "## A simpler baseline converted signal more efficiently\n\n"
                    "The fixed value-quality-momentum comparator had the strongest final-period "
                    "Sharpe and much lower explicit costs. It should be an anchor for an exploratory "
                    "v2, with machine learning required to prove incremental residual value."
                ),
            },
            {
                "id": "strategy_chart_block",
                "type": "chart",
                "layout": "full",
                "chartId": "strategy_chart",
            },
            {
                "id": "improvement_plan",
                "type": "markdown",
                "layout": "full",
                "sourceId": "stage14_registry",
                "body": (
                    "## Improvement plan\n\n"
                    "1. **Run Stage 14 unchanged first.** Execute the prespecified inference, "
                    "robustness, cost, capacity, delay, target/loss, refit-memory, feature-ablation, "
                    "and regime tests. Do not rewrite the primary after seeing Stage 13.\n\n"
                    "2. **For an explicitly exploratory v2, fix conversion before adding model "
                    "complexity.** Target beta near zero with an estimation-error buffer; shrink "
                    "other factor exposures; throttle gross exposure by forecast edge net of costs; "
                    "and use no-trade bands, rank buffers, position persistence, or staggered cohorts.\n\n"
                    "3. **Make the learner earn its complexity.** Test shorter/recency-weighted "
                    "memory, trailing-OOS-IC ensemble weights, pairwise ranking or Huber targets, "
                    "and core-feature ablations. Use fixed VQM as the low-turnover anchor and add "
                    "ML only for incremental residual alpha.\n\n"
                    "4. **Restore validation discipline.** Use fewer preregistered trials, purged "
                    "nested walk-forward selection, deflated-Sharpe/multiple-testing correction, "
                    "and a genuinely untouched future holdout. Any tuning informed by 2020–2025 "
                    "is post hoc and cannot rehabilitate the Stage 13 primary."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## Limitations and robustness status\n\n"
                    "Official Stage 13 point estimates and the zero-difference verification are "
                    "authoritative. The regression p-values, cost counterfactuals, IC quartiles, "
                    "and causal interpretation in this report are post-unblind diagnostics. They "
                    "do not replace the frozen Stage 14 procedures. The current defensible thesis "
                    "statement is: **suggestive gross residual alpha, unconfirmed net alpha, and a "
                    "failed primary implementation.**"
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## Next step\n\n"
                    "Complete Stage 14 against the verified Stage 13 bundle. Then classify which "
                    "failure channels survive prespecified robustness tests before designing or "
                    "tuning an exploratory successor."
                ),
            },
        ],
        "cards": [],
        "charts": [
            {
                "id": "alpha_bridge_chart",
                "type": "bar",
                "layout": "full",
                "dataset": "alpha_bridge",
                "sourceId": "analysis_snapshot",
                "title": "Annualized gross-to-net alpha bridge",
                "subtitle": "FF5 + momentum regression; factor contribution and explicit costs shown as drags",
                "intent": "comparison",
                "question": "How did a positive gross residual alpha become a negative net excess return?",
                "rationale": "The ordered bridge separates stock-selection residual, factor contribution, and explicit costs.",
                "showDescription": True,
                "maxRows": 5,
                "valueFormat": "percent",
                "encodings": {
                    "x": {
                        "field": "component",
                        "label": "Bridge component",
                        "type": "ordinal",
                        "sort": {"field": "order", "direction": "asc"},
                    },
                    "y": {
                        "field": "annualized_return",
                        "label": "Annualized return contribution",
                        "type": "quantitative",
                        "format": "percent",
                    },
                    "color": {
                        "field": "kind",
                        "label": "Component type",
                        "type": "nominal",
                    },
                },
            },
            {
                "id": "signal_chart",
                "type": "line",
                "layout": "full",
                "dataset": "signal_by_period",
                "sourceId": "stage13_calibration",
                "title": "Rank IC fell sharply after selection",
                "subtitle": "Stage 11 selected-validation mean versus annual Stage 13 monthly Spearman IC",
                "intent": "trend",
                "question": "Did cross-sectional forecast skill persist in the untouched final period?",
                "rationale": "The compact sequence exposes the level shift and the 2025 sign reversal.",
                "showDescription": True,
                "maxRows": 7,
                "valueFormat": "number",
                "encodings": {
                    "x": {
                        "field": "period",
                        "label": "Evaluation period",
                        "type": "ordinal",
                        "sort": {"field": "period_order", "direction": "asc"},
                    },
                    "y": {
                        "field": "mean_ic",
                        "label": "Mean monthly Spearman IC",
                        "type": "quantitative",
                        "format": "number",
                    },
                },
            },
            {
                "id": "constraint_chart",
                "type": "bar",
                "layout": "full",
                "dataset": "constraint_bindings",
                "sourceId": "stage13_costs",
                "title": "The primary allocator repeatedly hit its limits",
                "subtitle": "Binding months divided by 68 final-period months",
                "intent": "comparison",
                "question": "Was the portfolio adapting exposure to weak signal conditions?",
                "rationale": "Binding rates make the corner-solution behavior directly comparable.",
                "showDescription": True,
                "maxRows": 4,
                "valueFormat": "percent",
                "encodings": {
                    "x": {
                        "field": "constraint",
                        "label": "Constraint",
                        "type": "ordinal",
                    },
                    "y": {
                        "field": "binding_rate",
                        "label": "Binding share",
                        "type": "quantitative",
                        "format": "percent",
                    },
                },
            },
            {
                "id": "cumulative_chart",
                "type": "line",
                "layout": "full",
                "dataset": "cumulative_performance",
                "sourceId": "stage13_monthly_accounting",
                "title": "Stage 13 wealth path before costs, after costs, and in cash",
                "subtitle": "Indexed to 100 before the January 2020 holding month",
                "intent": "trend",
                "question": "How much did costs and the risk-free hurdle matter through time?",
                "rationale": "Three indexed paths separate portfolio economics from the cash benchmark.",
                "showDescription": True,
                "maxRows": 204,
                "valueFormat": "number",
                "encodings": {
                    "x": {
                        "field": "holding_month",
                        "label": "Holding month",
                        "type": "temporal",
                    },
                    "y": {
                        "field": "wealth_index",
                        "label": "Wealth index",
                        "type": "quantitative",
                        "format": "number",
                    },
                    "color": {
                        "field": "series",
                        "label": "Series",
                        "type": "nominal",
                    },
                },
            },
            {
                "id": "strategy_chart",
                "type": "bar",
                "layout": "full",
                "dataset": "strategy_comparison",
                "sourceId": "stage13_required_metrics",
                "title": "Final-period Sharpe by frozen strategy",
                "subtitle": "Fully net arithmetic excess-return Sharpe, 2020-01 through 2025-08",
                "intent": "comparison",
                "question": "Which frozen architecture converted final-period opportunities most efficiently?",
                "rationale": "A common-window Sharpe comparison highlights the low-turnover VQM baseline.",
                "showDescription": True,
                "maxRows": 6,
                "valueFormat": "number",
                "encodings": {
                    "x": {
                        "field": "strategy",
                        "label": "Strategy",
                        "type": "ordinal",
                    },
                    "y": {
                        "field": "sharpe",
                        "label": "Sharpe",
                        "type": "quantitative",
                        "format": "number",
                    },
                },
            },
        ],
        "tables": [
            {
                "id": "old_new_table",
                "dataset": "old_vs_final",
                "sourceId": "analysis_snapshot",
                "layout": "full",
                "density": "comfortable",
                "showDescription": True,
                "title": "Selected validation versus untouched final period",
                "subtitle": "Same frozen primary architecture; different windows and evidentiary status",
                "columns": [
                    {"field": "period", "label": "Period", "type": "text"},
                    {"field": "holding_window", "label": "Holding window", "type": "text"},
                    {"field": "months", "label": "Months", "type": "number", "format": "number"},
                    {"field": "status", "label": "Status", "type": "text"},
                    {"field": "sharpe", "label": "Sharpe", "type": "number", "format": "number"},
                    {
                        "field": "annualized_net_excess_return",
                        "label": "Net excess",
                        "type": "number",
                        "format": "percent",
                    },
                    {
                        "field": "annualized_volatility",
                        "label": "Volatility",
                        "type": "number",
                        "format": "percent",
                    },
                    {
                        "field": "mean_rank_ic",
                        "label": "Mean IC",
                        "type": "number",
                        "format": "number",
                    },
                    {
                        "field": "mean_gross",
                        "label": "Mean gross",
                        "type": "number",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "cost_counterfactual_table",
                "dataset": "cost_counterfactuals",
                "sourceId": "analysis_snapshot",
                "layout": "full",
                "density": "comfortable",
                "showDescription": True,
                "title": "Cost counterfactuals cannot recreate the old Sharpe",
                "subtitle": "Post-hoc diagnostics only; all scenarios preserve the realized gross portfolio",
                "columns": [
                    {"field": "scenario", "label": "Scenario", "type": "text"},
                    {
                        "field": "annualized_excess_return",
                        "label": "Annualized excess",
                        "type": "number",
                        "format": "percent",
                    },
                    {
                        "field": "annualized_volatility",
                        "label": "Volatility",
                        "type": "number",
                        "format": "percent",
                    },
                    {"field": "sharpe", "label": "Sharpe", "type": "number", "format": "number"},
                ],
            },
            {
                "id": "annual_table",
                "dataset": "annual_diagnostics",
                "sourceId": "analysis_snapshot",
                "layout": "full",
                "density": "comfortable",
                "showDescription": True,
                "title": "Annual final-period diagnostics",
                "subtitle": "2025 contains January through August only; returns remain arithmetic annualizations",
                "columns": [
                    {"field": "year", "label": "Year", "type": "number", "format": "number"},
                    {"field": "months", "label": "Months", "type": "number", "format": "number"},
                    {
                        "field": "gross_excess_return",
                        "label": "Gross excess",
                        "type": "number",
                        "format": "percent",
                    },
                    {
                        "field": "net_excess_return",
                        "label": "Net excess",
                        "type": "number",
                        "format": "percent",
                    },
                    {
                        "field": "explicit_cost_drag",
                        "label": "Cost drag",
                        "type": "number",
                        "format": "percent",
                    },
                    {"field": "net_sharpe", "label": "Net Sharpe", "type": "number", "format": "number"},
                    {"field": "mean_ic", "label": "Mean IC", "type": "number", "format": "number"},
                    {
                        "field": "mean_ex_ante_beta",
                        "label": "Ex-ante beta",
                        "type": "number",
                        "format": "number",
                    },
                    {
                        "field": "market_factor_return",
                        "label": "MKT-RF",
                        "type": "number",
                        "format": "percent",
                    },
                ],
            },
            {
                "id": "alpha_table",
                "dataset": "alpha_evidence",
                "sourceId": "analysis_snapshot",
                "layout": "full",
                "density": "comfortable",
                "showDescription": True,
                "title": "Alpha evidence depends on specification and remains unconfirmed",
                "subtitle": "Monthly OLS with Newey-West HAC(6) inference; 68 observations",
                "columns": [
                    {"field": "specification", "label": "Specification", "type": "text"},
                    {
                        "field": "annualized_alpha",
                        "label": "Annualized alpha",
                        "type": "number",
                        "format": "percent",
                    },
                    {
                        "field": "hac6_p_value",
                        "label": "HAC(6) p-value",
                        "type": "number",
                        "format": "number",
                    },
                    {"field": "interpretation", "label": "Interpretation", "type": "text"},
                    {"field": "status", "label": "Evidence status", "type": "text"},
                ],
            },
        ],
        "sources": sources,
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }

    dump_json(OUT / "analysis_snapshot.json", snapshot)
    dump_json(OUT / "artifact.json", artifact)
    source_notes = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": diagnostic_summary["classification"],
        "evidence_boundary": (
            "Post-unblind diagnostic only. No Stage 12 or Stage 13 artifact was modified, "
            "and no pending Stage 14 result was fabricated or bypassed."
        ),
        "methodology": {
            "returns": "Arithmetic monthly mean times 12; sample volatility times sqrt(12).",
            "factor_models": "Monthly OLS; alpha inference also reported with Newey-West HAC(6).",
            "costs": "Component USD divided by start-of-month NAV, then monthly mean times 12.",
            "signal": "One repeated monthly IC/calibration value retained per holding month.",
            "constraints": "Binding counts use numerical tolerances of 1e-6 or tighter.",
        },
        "source_hashes": {
            (S11 / "summary.json").as_posix(): sha256(S11 / "summary.json"),
            VERIFIED_POINTER.as_posix(): sha256(VERIFIED_POINTER),
            (S13 / "required_metrics.json").as_posix(): sha256(
                S13 / "required_metrics.json"
            ),
            (S13 / "monthly_accounting.json").as_posix(): sha256(
                S13 / "monthly_accounting.json"
            ),
            (S13 / "factors_benchmarks.json").as_posix(): sha256(
                S13 / "factors_benchmarks.json"
            ),
            (S13 / "costs_constraints_materiality.json").as_posix(): sha256(
                S13 / "costs_constraints_materiality.json"
            ),
            (S13 / "inference_results.json").as_posix(): sha256(
                S13 / "inference_results.json"
            ),
            FROZEN_SYSTEM.as_posix(): sha256(FROZEN_SYSTEM),
            STAGE14_REGISTRY.as_posix(): sha256(STAGE14_REGISTRY),
        },
        "chart_map": {
            "alpha_bridge_chart": "Explains gross residual alpha -> factor drag -> costs -> net excess.",
            "signal_chart": "Shows the selected-validation IC level and final-period annual IC.",
            "constraint_chart": "Shows persistent corner-solution behavior.",
            "cumulative_chart": "Separates gross, net, and cash wealth paths.",
            "strategy_chart": "Compares all frozen strategies on the same final window.",
        },
        "qa_expectation": (
            "The portable HTML should be structurally validated. Browser screenshot validation "
            "may be unavailable when Chromium headless-shell is not installed."
        ),
    }
    dump_json(OUT / "source_notes.json", source_notes)

    build_notebook(snapshot)
    return {
        "artifact": str(OUT / "artifact.json"),
        "snapshot": str(OUT / "analysis_snapshot.json"),
        "source_notes": str(OUT / "source_notes.json"),
        "notebook": str(NOTEBOOK),
        "summary": clean(diagnostic_summary),
    }


def build_notebook(snapshot: dict[str, Any]) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Stage 13 alpha deterioration diagnostic\n\n"
            "**TL;DR:** Stage 13 is a verified failed primary result. There is a positive "
            "gross FF5+momentum residual-alpha point estimate, but it is not statistically "
            "confirmed; factor drag and explicit costs turn weak gross economics negative.\n\n"
            "This notebook is post-unblind diagnostic evidence. It does not modify the frozen "
            "system or replace Stage 14 inference."
        ),
        nbf.v4.new_markdown_cell(
            "## Context and data\n\n"
            "The notebook reads the bounded diagnostic snapshot plus the official Stage 13 "
            "monthly accounting and factor files. The generator reduced the 948k-row model "
            "calibration file to one repeated-value IC/calibration record per holding month."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import statsmodels.api as sm\n"
            "import matplotlib.pyplot as plt\n\n"
            "ROOT = Path.cwd()\n"
            "SNAPSHOT_PATH = ROOT / 'reports/stage_13_alpha_diagnostic/analysis_snapshot.json'\n"
            "with SNAPSHOT_PATH.open() as fh:\n"
            "    snapshot = json.load(fh)\n"
            "ds = snapshot['datasets']\n"
            "pd.DataFrame(ds['analysis_metadata'])"
        ),
        nbf.v4.new_markdown_cell("## Old validation versus final result"),
        nbf.v4.new_code_cell(
            "old_final = pd.DataFrame(ds['old_vs_final'])\n"
            "old_final[['period', 'holding_window', 'status', 'sharpe', "
            "'annualized_net_excess_return', 'annualized_volatility', 'mean_rank_ic']]"
        ),
        nbf.v4.new_markdown_cell(
            "## Recompute the final gross-to-net and factor model\n\n"
            "The arithmetic is rerun from the bounded monthly diagnostic rows. HAC(6) "
            "p-values are post-hoc diagnostics."
        ),
        nbf.v4.new_code_cell(
            "m = pd.DataFrame(ds['monthly_diagnostics']).sort_values('holding_month')\n"
            "factors = ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd']\n"
            "def sharpe(x):\n"
            "    return np.sqrt(12) * x.mean() / x.std(ddof=1)\n"
            "gross_excess = 12 * m['gross_excess_return'].mean()\n"
            "net_excess = 12 * m['net_excess_return'].mean()\n"
            "cost_drag = 12 * m['total_cost_return'].mean()\n"
            "X = sm.add_constant(m[factors], has_constant='add')\n"
            "gross_fit = sm.OLS(m['gross_excess_return'], X).fit()\n"
            "gross_hac = gross_fit.get_robustcov_results(cov_type='HAC', maxlags=6)\n"
            "net_fit = sm.OLS(m['net_excess_return'], X).fit()\n"
            "net_hac = net_fit.get_robustcov_results(cov_type='HAC', maxlags=6)\n"
            "pd.DataFrame({\n"
            "  'metric': ['Gross excess', 'Explicit cost drag', 'Net excess', "
            "'Gross Sharpe', 'Net Sharpe', 'Gross FF6 alpha', 'Gross alpha HAC p', "
            "'Net FF6 alpha', 'Net alpha HAC p'],\n"
            "  'value': [gross_excess, cost_drag, net_excess, "
            "sharpe(m['gross_excess_return']), sharpe(m['net_excess_return']), "
            "12*gross_fit.params['const'], gross_hac.pvalues[0], "
            "12*net_fit.params['const'], net_hac.pvalues[0]]\n"
            "})"
        ),
        nbf.v4.new_markdown_cell("## Return bridge"),
        nbf.v4.new_code_cell(
            "bridge = pd.DataFrame(ds['alpha_bridge']).sort_values('order')\n"
            "colors = ['#3b82f6' if x >= 0 else '#ef4444' for x in bridge['annualized_return']]\n"
            "ax = bridge.plot.bar(x='component', y='annualized_return', color=colors, "
            "legend=False, figsize=(10, 4))\n"
            "ax.axhline(0, color='black', linewidth=0.8)\n"
            "ax.set_ylabel('Annualized return contribution')\n"
            "ax.set_title('Gross FF6 residual alpha to realized net excess return')\n"
            "plt.xticks(rotation=25, ha='right')\n"
            "plt.tight_layout();"
        ),
        nbf.v4.new_markdown_cell("## Annual regime diagnostics"),
        nbf.v4.new_code_cell(
            "annual = pd.DataFrame(ds['annual_diagnostics'])\n"
            "annual[['year', 'months', 'gross_excess_return', 'net_excess_return', "
            "'explicit_cost_drag', 'net_sharpe', 'mean_ic', 'mean_ex_ante_beta', "
            "'market_factor_return']]"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "- The old 1.537 Sharpe was a selected validation diagnostic; the final "
            "2020–2025 Sharpe is -0.091.\n"
            "- Mean rank IC declined by roughly two thirds.\n"
            "- The primary ran at its turnover, gross, and beta bounds in most months.\n"
            "- Explicit cost reduction alone cannot recover the old Sharpe; zero-cost "
            "final-period Sharpe is only about 0.20.\n"
            "- Alpha is suggestive before costs, but not statistically confirmed and "
            "sensitive to factor specification.\n"
            "- Run Stage 14 unchanged before any exploratory v2 redesign."
        ),
    ]
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    executed = client.execute()
    with NOTEBOOK.open("w", encoding="utf-8") as handle:
        nbf.write(executed, handle)


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
