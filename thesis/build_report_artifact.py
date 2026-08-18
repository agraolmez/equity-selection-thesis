"""Package the frozen thesis evidence into the canonical report artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "thesis" / "report"
FRONTIER = (
    ROOT
    / "artifacts/research_v6_1/net_return_frontier/"
    "880c4b16de97-6e6c91767c46-eac64c8cf6db"
)
SELECTOR = (
    ROOT
    / "artifacts/research_v6_1/selector_step_01/"
    "481e306553c2-16046ee58521-d5dadd9cd9c3"
)
V6_3 = (
    ROOT
    / "artifacts/research_v6_3/signal_weight_research/"
    "69841fc46fbd-c1dc1cd6be2e-ec470fe0dfb4"
)
BENCHMARKS = Path(
    "/Users/agraemirolmez/Desktop/Thesis/processed/phase_06_final_test/"
    "stage13-final-inputs-v2/primary_benchmarks.parquet"
)
GENERATED_AT = "2026-08-14T16:30:00+03:00"


def _selected_selector_rows() -> list[dict]:
    chosen = {
        "capm": ("CAPM alpha", "ensemble__cam_free_mlp_plus_cam__equal", "CAM-free continuous MLP + direct CAM"),
        "ff5_momentum": ("FF5 + momentum alpha", "ensemble__cam_free_mlp_plus_cam__equal", "CAM-free pairwise MLP + direct CAM"),
        "long_net": ("Long net payoff", "ensemble__cam_free_mlp_plus_cam__equal", "CAM-free continuous MLP + direct CAM"),
        "short_net": ("Short net payoff", "ensemble__correlation_reduced_xgb_mlp__equal", "Correlation-reduced pairwise XGBoost + MLP"),
    }
    rows: list[dict] = []
    for target, (label, model_id, model) in chosen.items():
        frame = pd.read_parquet(SELECTOR / target / "official_summary.parquet")
        for period in ["validation", "historical_test"]:
            row = frame.loc[frame["model_id"].eq(model_id) & frame["period"].eq(period)]
            if len(row) != 1:
                raise RuntimeError(f"missing selected selector row: {target}/{period}")
            value = row.iloc[0]
            rows.append(
                {
                    "target": label,
                    "model": model,
                    "period": "Validation" if period == "validation" else "Historical test",
                    "months": int(value["months"]),
                    "mean_ic": round(float(value["mean_rank_ic"]), 6),
                    "hac_t": round(float(value["rank_ic_hac_t"]), 4),
                    "positive_fraction": round(float(value["positive_ic_fraction"]), 4),
                    "top_minus_bottom": round(float(value["mean_economic_top_minus_bottom"]), 6),
                }
            )
    return rows


def _portfolio_data() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    result = json.loads((FRONTIER / "result.json").read_text())
    win = result["historical_test"]["position_1p5"]
    control = result["historical_test"]["control"]
    headline = [
        {
            "net_excess": win["annualized_net_excess_return_primary"],
            "stress_net_excess": win["annualized_net_excess_return_stress"],
            "total_cagr": win["net_total_return_cagr"],
            "sharpe": win["net_sharpe_primary"],
            "stress_sharpe": win["net_sharpe_stress"],
            "volatility": win["annualized_realized_volatility_primary"],
            "drawdown": win["maximum_drawdown_primary"],
            "positive_months": win["positive_net_month_fraction"],
            "hac_t": result["paired_inference_vs_frozen_control"]["position_1p5"]["challenger_hac_t_mean_net_return"],
            "months": win["months"],
            "mean_gross": win["mean_gross"],
            "mean_net": win["mean_net"],
            "primary_cost": win["annualized_primary_cost"],
            "maximum_adv": win["maximum_adv_participation"],
        }
    ]
    comparison = []
    for label, row in [("Frozen 1.0% control", control), ("Validation-selected 1.5% cap", win)]:
        comparison.append(
            {
                "portfolio": label,
                "net_excess": row["annualized_net_excess_return_primary"],
                "stress_net_excess": row["annualized_net_excess_return_stress"],
                "total_cagr": row["net_total_return_cagr"],
                "volatility": row["annualized_realized_volatility_primary"],
                "sharpe": row["net_sharpe_primary"],
                "stress_sharpe": row["net_sharpe_stress"],
                "drawdown": row["maximum_drawdown_primary"],
                "primary_cost": row["annualized_primary_cost"],
                "gross": row["mean_gross"],
                "net": row["mean_net"],
            }
        )

    monthly = pd.read_parquet(FRONTIER / "historical_test_monthly.parquet")
    monthly = monthly.loc[monthly["profile"].eq("position_1p5")].copy()
    monthly = monthly.sort_values("holding_month", kind="mergesort")
    spy = pd.read_parquet(BENCHMARKS)
    spy = spy.loc[
        spy["benchmark_id"].astype(str).eq("spy_total_return"),
        ["holding_month", "total_return"],
    ].rename(columns={"total_return": "spy_return"})
    monthly = monthly.merge(spy, on="holding_month", how="left", validate="one_to_one")
    monthly["strategy_wealth"] = (1.0 + monthly["net_return_primary"]).cumprod()
    monthly["spy_wealth"] = (1.0 + monthly["spy_return"]).cumprod()
    path = []
    returns = []
    for _, row in monthly.iterrows():
        month = str(row["holding_month"])
        path.extend(
            [
                {"month": month, "series": "Frozen ML portfolio", "wealth": round(float(row["strategy_wealth"]), 6)},
                {"month": month, "series": "SPY total return", "wealth": round(float(row["spy_wealth"]), 6)},
            ]
        )
        net = float(row["net_excess_return_primary"])
        returns.append(
            {
                "month": month,
                "net_excess_return": round(net, 8),
                "direction": "Positive" if net >= 0 else "Negative",
                "gross": round(float(row["gross"]), 6),
                "net": round(float(row["net"]), 6),
                "turnover": round(float(row["turnover"]), 6),
                "cost": round(float(row["total_cost_primary"]), 8),
            }
        )
    return headline, comparison, path, returns


def _validation_frontier() -> list[dict]:
    keep = ["position_1p5", "factor_15", "position_2p0", "control", "short_free", "turnover_75", "high_return"]
    labels = {
        "position_1p5": "1.5% position cap",
        "factor_15": "15% factor limits",
        "position_2p0": "2.0% position cap",
        "control": "Frozen 1.0% control",
        "short_free": "No short floor",
        "turnover_75": "75% turnover cap",
        "high_return": "Joint aggressive",
    }
    frame = pd.read_parquet(FRONTIER / "validation_summary.parquet").set_index("profile")
    rows = []
    for profile in keep:
        row = frame.loc[profile]
        rows.append(
            {
                "profile": labels[profile],
                "annualized_net_excess": round(float(row["annualized_net_excess_return_primary"]), 8),
                "stress_net_excess": round(float(row["annualized_net_excess_return_stress"]), 8),
                "sharpe": round(float(row["net_sharpe_primary"]), 6),
                "drawdown": round(float(row["maximum_drawdown_primary"]), 8),
            }
        )
    return rows


def _sources() -> list[dict]:
    return [
        {
            "id": "selector_results",
            "label": "Frozen V6.1 selector summaries",
            "path": "artifacts/research_v6_1/selector_step_01/481e306553c2-16046ee58521-d5dadd9cd9c3",
            "query": {
                "engine": "pandas/Apache Arrow",
                "language": "python",
                "sql": "SELECT model_id, period, months, mean_rank_ic, rank_ic_hac_t, positive_ic_fraction, mean_economic_top_minus_bottom FROM read_parquet('artifacts/research_v6_1/selector_step_01/481e306553c2-16046ee58521-d5dadd9cd9c3/*/official_summary.parquet') WHERE (target, model_id) IN (('capm','ensemble__cam_free_mlp_plus_cam__equal'),('ff5_momentum','ensemble__cam_free_mlp_plus_cam__equal'),('long_net','ensemble__cam_free_mlp_plus_cam__equal'),('short_net','ensemble__correlation_reduced_xgb_mlp__equal')) AND period IN ('validation','historical_test');",
                "description": "Selected the four frozen V6.1 score columns and their validation and historical-test rank statistics.",
                "executed_at": GENERATED_AT,
                "tables_used": ["artifacts.research_v6_1.selector_step_01.target.official_summary"],
                "filters": ["selected score columns from the frozen portfolio contract", "validation 2020-01 to 2023-12", "historical test 2024-01 to 2025-08"],
                "metric_definitions": {"mean_ic": "Equal-month mean cross-sectional Spearman rank correlation.", "hac_t": "Six-lag HAC t-statistic of monthly rank IC."},
            },
        },
        {
            "id": "frontier_results",
            "label": "Frozen V6.1 net-return frontier and SPY benchmark",
            "path": "artifacts/research_v6_1/net_return_frontier/880c4b16de97-6e6c91767c46-eac64c8cf6db",
            "query": {
                "engine": "pandas/Apache Arrow",
                "language": "python/sql",
                "sql": "SELECT * FROM read_parquet('artifacts/research_v6_1/net_return_frontier/880c4b16de97-6e6c91767c46-eac64c8cf6db/historical_test_monthly.parquet') WHERE profile IN ('control','position_1p5'); SELECT * FROM read_parquet('artifacts/research_v6_1/net_return_frontier/880c4b16de97-6e6c91767c46-eac64c8cf6db/validation_summary.parquet'); SELECT holding_month, total_return FROM read_parquet('processed/phase_06_final_test/stage13-final-inputs-v2/primary_benchmarks.parquet') WHERE benchmark_id = 'spy_total_return';",
                "description": "Read the complete validation constraint frontier, the two frozen historical-test paths, and the authenticated SPY total-return benchmark.",
                "executed_at": GENERATED_AT,
                "tables_used": ["artifacts.research_v6_1.net_return_frontier.validation_summary", "artifacts.research_v6_1.net_return_frontier.historical_test_monthly", "phase_06_final_test.stage13_final_inputs_v2.primary_benchmarks"],
                "filters": ["portfolio validation 2022-02 to 2023-12", "historical test 2024-01 to 2025-08", "no outcomes after 2025-08"],
                "metric_definitions": {"net_excess": "Annualized arithmetic mean monthly return in excess of cash after modeled costs.", "sharpe": "Annualized mean monthly net excess return divided by realized annualized volatility.", "total_cagr": "Geometric annualized net total return including cash on net capital."},
            },
        },
        {
            "id": "final_challenge",
            "label": "Frozen V6.3 signal-to-weight challenge",
            "path": "artifacts/research_v6_3/signal_weight_research/69841fc46fbd-c1dc1cd6be2e-ec470fe0dfb4",
            "query": {
                "engine": "pandas/Apache Arrow",
                "language": "python/sql",
                "sql": "SELECT profile, annualized_net_excess_return_primary, net_sharpe_primary FROM read_parquet('artifacts/research_v6_3/signal_weight_research/69841fc46fbd-c1dc1cd6be2e-ec470fe0dfb4/historical_test_summary.parquet');",
                "description": "Compared the exactly reproduced incumbent with the only validation-surviving latent-risk challenger.",
                "executed_at": GENERATED_AT,
                "tables_used": ["artifacts.research_v6_3.signal_weight_research.historical_test_summary"],
                "filters": ["historical test 2024-01 to 2025-08", "validation-selected challenger only"],
            },
        },
    ]


def main() -> None:
    selector = _selected_selector_rows()
    headline, comparison, path, monthly = _portfolio_data()
    validation = _validation_frontier()
    sources = _sources()
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Machine Learning in Portfolio Construction: Stock Selection and Dynamic Equity Allocation",
            "description": "Technical thesis reading copy of the frozen V6.1 selector and portfolio evidence, with V6.3 closure and forward-validation limits.",
            "generatedAt": GENERATED_AT,
            "cards": [
                {"id": "net_return", "description": "Annualized arithmetic mean monthly return in excess of cash after primary modeled costs.", "dataset": "headline", "sourceId": "frontier_results", "metrics": [{"label": "Annualized net excess", "field": "net_excess", "format": "percent"}, {"label": "Stressed costs", "field": "stress_net_excess", "format": "percent"}]},
                {"id": "sharpe", "description": "Annualized net excess return divided by realized annualized volatility.", "dataset": "headline", "sourceId": "frontier_results", "metrics": [{"label": "Net Sharpe", "field": "sharpe", "format": "number"}, {"label": "Stressed", "field": "stress_sharpe", "format": "number"}]},
                {"id": "risk", "description": "Realized risk of the frozen 20-month total-return path.", "dataset": "headline", "sourceId": "frontier_results", "metrics": [{"label": "Realized volatility", "field": "volatility", "format": "percent"}, {"label": "Maximum drawdown", "field": "drawdown", "format": "percent", "signed": True}]},
                {"id": "evidence", "description": "Short post-diagnostic historical-development sample; not untouched production confirmation.", "dataset": "headline", "sourceId": "frontier_results", "metrics": [{"label": "Historical-test months", "field": "months", "format": "number"}, {"label": "Positive months", "field": "positive_months", "format": "percent"}]},
                {"id": "capacity", "description": "$100 million research scale; maximum one-way ADV participation reaches the frozen limit.", "dataset": "headline", "sourceId": "frontier_results", "metrics": [{"label": "Maximum ADV", "field": "maximum_adv", "format": "percent"}, {"label": "Annualized primary cost", "field": "primary_cost", "format": "percent"}]},
            ],
            "charts": [
                {"id": "selector_ic", "title": "Mean monthly rank IC by selected target", "subtitle": "All four frozen scores rank positively; short-payoff validation is directionally positive but weak.", "type": "bar", "intent": "comparison", "dataset": "selector", "sourceId": "selector_results", "encodings": {"x": {"field": "target", "type": "nominal", "label": "Prediction target"}, "y": {"field": "mean_ic", "type": "quantitative", "label": "Mean rank IC", "format": "number"}, "color": {"field": "period", "type": "nominal", "label": "Evidence period"}}, "valueFormat": "number", "layout": "full"},
                {"id": "validation_frontier", "title": "Selected validation constraint profiles", "subtitle": "The 1.5% cap is selected on February 2022–December 2023 net return, before the historical-test outcomes.", "type": "bar", "intent": "comparison", "dataset": "validation", "sourceId": "frontier_results", "encodings": {"x": {"field": "profile", "type": "nominal", "label": "Portfolio profile"}, "y": {"field": "annualized_net_excess", "type": "quantitative", "label": "Annualized net excess return", "format": "percent"}}, "valueFormat": "percent", "layout": "full"},
                {"id": "wealth_path", "title": "Compounded total-return paths", "subtitle": "The portfolio and SPY finish at nearly identical wealth; the portfolio path is smoother in this sample.", "type": "line", "intent": "trend", "dataset": "path", "sourceId": "frontier_results", "encodings": {"x": {"field": "month", "type": "temporal", "label": "Holding month"}, "y": {"field": "wealth", "type": "quantitative", "label": "Growth of $1", "format": "number"}, "color": {"field": "series", "type": "nominal", "label": "Series"}}, "valueFormat": "number", "layout": "full"},
                {"id": "monthly_return", "title": "Monthly net excess returns", "subtitle": "Primary modeled trading, borrow, and financing costs; 15 of 20 months are positive.", "type": "bar", "intent": "trend", "dataset": "monthly", "sourceId": "frontier_results", "encodings": {"x": {"field": "month", "type": "temporal", "label": "Holding month"}, "y": {"field": "net_excess_return", "type": "quantitative", "label": "Net excess return", "format": "percent"}, "color": {"field": "direction", "type": "nominal", "label": "Direction"}}, "valueFormat": "percent", "layout": "full"},
            ],
            "tables": [
                {"id": "selector_table", "title": "Frozen selected models", "subtitle": "Validation and historical-test rank statistics for the exact scores used or retained by the final portfolio lineage.", "dataset": "selector", "sourceId": "selector_results", "defaultSort": {"field": "target", "direction": "asc"}, "density": "dense", "layout": "full", "columns": [{"field": "target", "label": "Target", "type": "text"}, {"field": "model", "label": "Model", "type": "text"}, {"field": "period", "label": "Period", "type": "text"}, {"field": "months", "label": "Months", "format": "number"}, {"field": "mean_ic", "label": "Mean IC", "format": "number"}, {"field": "hac_t", "label": "HAC t", "format": "number"}, {"field": "positive_fraction", "label": "Positive IC", "format": "percent"}]},
                {"id": "portfolio_table", "title": "Frozen portfolio comparison", "subtitle": "The 1.5% cap is the validation-selected confirmation candidate; the 1.0% row is its frozen comparator.", "dataset": "comparison", "sourceId": "frontier_results", "defaultSort": {"field": "net_excess", "direction": "desc"}, "density": "dense", "layout": "full", "columns": [{"field": "portfolio", "label": "Portfolio", "type": "text"}, {"field": "net_excess", "label": "Net excess", "format": "percent"}, {"field": "stress_net_excess", "label": "Stress net", "format": "percent"}, {"field": "total_cagr", "label": "Total CAGR", "format": "percent"}, {"field": "volatility", "label": "Volatility", "format": "percent"}, {"field": "sharpe", "label": "Sharpe", "format": "number"}, {"field": "drawdown", "label": "Max drawdown", "format": "percent"}, {"field": "primary_cost", "label": "Cost", "format": "percent"}]},
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Machine Learning in Portfolio Construction: Stock Selection and Dynamic Equity Allocation"},
                {"id": "summary", "type": "markdown", "sourceId": "frontier_results", "body": "## Technical Summary\n\nThe completed research supports a qualified positive conclusion. Frozen machine-learning ensembles rank CAPM alpha, FF5-plus-momentum alpha, long payoff, and short payoff positively. A causal rank-to-payoff calibration and cost-aware optimizer translate those rankings into a $100 million portfolio. The validation-selected 1.5% position-cap policy earns **14.67% annualized net excess return**, **6.32% realized volatility**, a **2.320 Sharpe ratio**, and a **−1.34% maximum drawdown** over January 2024–August 2025. Under stressed costs, Sharpe remains **2.054**.\n\nThe result is post-diagnostic historical development, not production proof. It spans 20 months, reaches the 10% ADV limit, and has no untouched confirmation."},
                {"id": "metrics", "type": "metric-strip", "cardIds": ["net_return", "sharpe", "risk", "evidence", "capacity"]},
                {"id": "selectors_heading", "type": "markdown", "body": "## The Models Rank Stocks, Not Exact Returns\n\nThe final design asks models to order stocks cross-sectionally. CAPM, FF5-plus-momentum, and long payoff use equal-rank blends of a CAM-free neural network and direct LSEG CAM. Short payoff uses a pairwise, correlation-reduced XGBoost–MLP ensemble. CAM is an analyst-consensus rank, not a separately trained model."},
                {"id": "selector_chart", "type": "chart", "chartId": "selector_ic", "layout": "full"},
                {"id": "selector_table_block", "type": "table", "tableId": "selector_table", "layout": "full"},
                {"id": "portfolio_heading", "type": "markdown", "sourceId": "frontier_results", "body": "## Validation Selected a Simple Concentration Change\n\nThe frozen selectors, FF5-plus-momentum sign gate, candidate tails, causal calibration, cost model, and risk model remain unchanged. The frontier changes transparent portfolio constraints only. A **1.5% candidate cap** wins on February 2022–December 2023 validation net return. More aggressive turnover, leverage, and broad freedom generally reduce net performance after costs."},
                {"id": "validation_chart_block", "type": "chart", "chartId": "validation_frontier", "layout": "full"},
                {"id": "results_heading", "type": "markdown", "sourceId": "frontier_results", "body": "## Historical-Test Result\n\nThe 1.5% candidate improves annualized net excess return by **4.73 percentage points** and Sharpe by **0.597** relative to the frozen 1.0% control. Its paired HAC t-statistic is **5.135** and its block-bootstrap annualized return-difference interval is **+2.984% to +6.524%**. Those figures are strong within the observed comparison but do not correct for the complete post-diagnostic research history."},
                {"id": "portfolio_table_block", "type": "table", "tableId": "portfolio_table", "layout": "full"},
                {"id": "wealth_chart_block", "type": "chart", "chartId": "wealth_path", "layout": "full"},
                {"id": "monthly_chart_block", "type": "chart", "chartId": "monthly_return", "layout": "full"},
                {"id": "mechanism", "type": "markdown", "sourceId": "frontier_results", "body": "## Economic Mechanism\n\nThe result is long-driven. Annualized long contribution is **+16.28%**, short contribution is **−0.54%**, and primary costs consume **1.07%**. Mean gross exposure is **1.013**, mean net exposure is **0.279**, and the portfolio holds about 60 longs and 62 shorts. The short sleeve is best interpreted as a joint alpha-and-covariance instrument rather than a separately proven return engine."},
                {"id": "robustness", "type": "markdown", "sourceId": "final_challenge", "body": "## Final Robustness Challenge\n\nV6.3 tested six additional signal-to-weight ideas. Only a three-factor latent-risk challenger survived validation; it earned 14.55% net excess return and a 2.282 Sharpe in the historical test, slightly below the unchanged incumbent. V6.3 reproduced every incumbent monthly return exactly. The final decision is to retain the simpler frozen policy."},
                {"id": "limitations", "type": "markdown", "body": "## Limitations and Uncertainty\n\nThe test contains only 20 months and follows multiple earlier research decisions. The short-payoff selector's validation t-statistic is only 0.97. All monthly optimizations carry an `optimal_inaccurate` solver label, although realized constraints and accounting reconcile to immaterial residue. The 10% ADV limit binds at $100 million. The active local dependency environment also differs from the pinned project contract. None of these facts changes the frozen historical arithmetic, but together they prevent a production-quality claim."},
                {"id": "next_steps", "type": "markdown", "body": "## Next Step: Frozen Paper Trading\n\nDo not tune further on the historical window. Recreate the pinned environment, freeze the repository and source manifest, and begin prospective paper trading no earlier than September 2026. Preserve each signal-time data snapshot, feature vector, score, calibrated payoff, target weight, constraint audit, order, fill, borrow status, and realized cost. Evaluate the untouched policy only after a predeclared forward horizon."},
                {"id": "questions", "type": "markdown", "body": "## Further Questions\n\n1. Do selector IC and calibration slopes remain positive prospectively?\n2. Does the short sleeve continue to reduce covariance without excessive drag?\n3. How large is broker-paper slippage relative to modeled cost?\n4. Does the 10% ADV constraint remain feasible as the universe and capital change?\n5. Does the strategy retain its low-volatility benchmark-matching profile outside the 2024–2025 regime?"},
            ],
        },
        "snapshot": {"version": 1, "generatedAt": GENERATED_AT, "status": "ready", "datasets": {"headline": headline, "selector": selector, "comparison": comparison, "validation": validation, "path": path, "monthly": monthly}, "accessIssues": []},
        "sources": sources,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
