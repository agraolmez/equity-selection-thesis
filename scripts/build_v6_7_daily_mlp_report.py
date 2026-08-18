#!/usr/bin/env python3
"""Build the executed audit notebook and canonical V6.7 report artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / (
    "artifacts/research_v6_7/daily_mlp_overlay/"
    "4108d1f64761-16046ee58521-22f799195507"
)
V6_6_ROOT = ROOT / (
    "artifacts/research_v6_6/intrmonth_interventions/"
    "566f3749ee9b-16046ee58521-a82fbba1983b"
)


def _rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.replace([np.inf, -np.inf], np.nan)
    return json.loads(clean.to_json(orient="records", date_format="iso"))


def _sql_values(frame: pd.DataFrame, columns: list[str], table: str) -> str:
    values: list[str] = []
    for row in frame[columns].itertuples(index=False, name=None):
        encoded: list[str] = []
        for value in row:
            if pd.isna(value):
                encoded.append("NULL")
            elif isinstance(value, str):
                encoded.append("'" + value.replace("'", "''") + "'")
            else:
                encoded.append(
                    repr(float(value))
                    if isinstance(value, (float, np.floating))
                    else str(int(value))
                )
        values.append("(" + ",".join(encoded) + ")")
    return f"SELECT * FROM (VALUES {','.join(values)}) AS {table}({','.join(columns)})"


def _build_notebook(result_root: Path, output: Path) -> None:
    result = json.loads((result_root / "result.json").read_text())
    historical = result["historical_test"]
    challenger = historical["selected_v6_7_challenger"]
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "## tl;dr\n\n"
            f"The validation-selected five-day MLP–XGBoost ensemble reached "
            f"{challenger['annualized_net_excess_return']:.2%} annualized net excess return "
            f"and {challenger['net_sharpe']:.3f} Sharpe in January 2024–August 2025. "
            "It beat the monthly control economically, but its paired HAC t-statistic was "
            f"{result['paired_differences']['versus_monthly_control']['paired_hac_t']:.2f}; "
            "the monthly primary is retained."
        ),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "V6.7 is a separate post-diagnostic comparison. It preserves the V6.6 prepared panel, "
            "monthly portfolio, 1.80× gross translation, costs, and next-session-close execution. "
            "Two compact neural objectives (signed-return regression and adverse-return classification) "
            "were tested at one- and five-day horizons as three-seed averages."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json, numpy as np, pandas as pd\n"
            f"root = Path({str(result_root)!r})\n"
            f"v6_6_root = Path({str(V6_6_ROOT)!r})\n"
            "result = json.loads((root/'result.json').read_text())\n"
            "candidates = pd.read_parquet(root/'selection/candidate_summary.parquet')\n"
            "result['decision']"
        ),
        nbformat.v4.new_markdown_cell("## Data"),
        nbformat.v4.new_code_cell(
            "pd.Series(result['data_quality'], name='value').to_frame()"
        ),
        nbformat.v4.new_markdown_cell("## Results"),
        nbformat.v4.new_code_cell(
            "pd.DataFrame([{'policy': k, **v} for k,v in result['historical_test'].items()])"
            "[['policy','annualized_net_excess_return','annualized_realized_volatility','net_sharpe','maximum_drawdown','intervention_count']]"
        ),
        nbformat.v4.new_code_cell(
            "candidates[['policy_id','annualized_net_excess_return','net_sharpe','maximum_drawdown']]"
            ".sort_values('net_sharpe', ascending=False).head(13)"
        ),
        nbformat.v4.new_markdown_cell("## Validation checks"),
        nbformat.v4.new_code_cell(
            "issues=[]\n"
            "for summary_path in list((root/'selection').glob('*/summary.json')) + list((root/'historical_test').glob('*/summary.json')):\n"
            "    p=summary_path.parent\n"
            "    monthly=pd.read_parquet(p/'monthly.parquet')\n"
            "    daily=pd.read_parquet(p/'daily.parquet')\n"
            "    actions=pd.read_parquet(p/'actions.parquet')\n"
            "    assert np.allclose(monthly.nav_end.to_numpy()[:-1], monthly.nav_start.to_numpy()[1:], atol=1e-7, rtol=0)\n"
            "    assert np.isfinite(daily[['nav_start','nav_end','gross_end']].to_numpy()).all()\n"
            "    if len(actions): assert (actions.new_fraction <= actions.old_fraction+1e-12).all() and (actions.new_fraction >= 0).all()\n"
            "receipts=[]\n"
            "for p in (root/'models').glob('**/*.json'):\n"
            "    receipt=json.loads(p.read_text())\n"
            "    if receipt.get('schema')=='research_v6_7_mlp_refit_v1':\n"
            "        receipts.append(receipt); assert receipt['train_last_month'] < receipt['refit_month']\n"
            "{'mechanical_issues':issues, 'causal_model_receipts':len(receipts), 'promotion_gates':result['promotion_gates']}"
        ),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "The daily MLP was worth testing: it improved the frozen XGBoost overlay when combined with it, "
            "and the selected ensemble improved return, Sharpe, and drawdown versus the monthly control. "
            "The improvement is nevertheless too concentrated for promotion on 20 historical months. "
            "Log the ensemble in shadow mode and retain monthly trading until prospective evidence is available."
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    executed = nbformat.read(output, as_version=4)
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        executed, {"metadata": {"path": str(ROOT)}}
    )
    nbformat.write(executed, output)


def _candidate_label(policy_id: str) -> str:
    label = policy_id.removeprefix("predictive__")
    replacements = {
        "ensemble_mlp_signed_return_xgb_depth_2__h5__selective": "Ensemble regression · 5d · selective",
        "ensemble_mlp_signed_return_xgb_depth_2__h5__broad": "Ensemble regression · 5d · broad",
        "ensemble_mlp_adverse_classifier_xgb_depth_2__h5__selective": "Ensemble classifier · 5d · selective",
        "ensemble_mlp_adverse_classifier_xgb_depth_2__h5__broad": "Ensemble classifier · 5d · broad",
        "mlp_signed_return__h1__selective": "MLP regression · 1d · selective",
        "mlp_signed_return__h1__broad": "MLP regression · 1d · broad",
        "mlp_signed_return__h5__selective": "MLP regression · 5d · selective",
        "mlp_signed_return__h5__broad": "MLP regression · 5d · broad",
        "mlp_adverse_classifier__h1__selective": "MLP classifier · 1d · selective",
        "mlp_adverse_classifier__h1__broad": "MLP classifier · 1d · broad",
        "mlp_adverse_classifier__h5__selective": "MLP classifier · 5d · selective",
        "mlp_adverse_classifier__h5__broad": "MLP classifier · 5d · broad",
        "xgb_depth_2__h5__selective": "Frozen XGBoost · 5d · selective",
    }
    return replacements.get(label, label)


def build(result_root: Path) -> None:
    result = json.loads((result_root / "result.json").read_text())
    predecessor_result = json.loads((V6_6_ROOT / "result.json").read_text())
    labels = {
        "monthly_control": "Monthly control",
        "frozen_xgboost": "Frozen XGBoost overlay",
        "selected_v6_7_challenger": "MLP–XGBoost ensemble",
    }
    performance = pd.DataFrame(
        [
            {
                "policy": policy,
                "policy_label": labels[policy],
                **summary,
                "paired_hac_t_vs_control": (
                    None
                    if policy == "monthly_control"
                    else result["paired_differences"]["versus_monthly_control"]["paired_hac_t"]
                    if policy == "selected_v6_7_challenger"
                    else predecessor_result["paired_differences"]["selected_predictive"]["paired_hac_t"]
                ),
            }
            for policy, summary in result["historical_test"].items()
        ]
    )

    wealth_parts: list[pd.DataFrame] = []
    daily_paths = {
        "Monthly control": V6_6_ROOT / "historical_test/monthly_control/daily.parquet",
        "Frozen XGBoost overlay": V6_6_ROOT / "historical_test/selected_predictive/daily.parquet",
        "MLP–XGBoost ensemble": result_root / "historical_test/selected_v6_7_challenger/daily.parquet",
    }
    for label, path in daily_paths.items():
        daily = pd.read_parquet(path).sort_values("calendar_date")
        base_nav = float(daily["nav_start"].iloc[0])
        subset = daily[["calendar_date", "nav_end"]].copy()
        subset["policy"] = label
        subset["wealth_100k"] = 100_000.0 * subset["nav_end"] / base_nav
        wealth_parts.append(subset[["calendar_date", "policy", "wealth_100k"]])
    wealth = pd.concat(wealth_parts, ignore_index=True)

    stability_rows: list[dict[str, object]] = []
    for policy, summary in (
        ("Monthly control", result["selection"]["monthly_control"]),
        ("Frozen XGBoost overlay", result["selection"]["frozen_xgboost"]),
        (
            "MLP–XGBoost ensemble",
            result["selection"]["selected_v6_7_challenger"]["summary"],
        ),
    ):
        stability_rows.append(
            {"policy": policy, "period": "2023 selection", "net_sharpe": summary["net_sharpe"]}
        )
    for key, policy in labels.items():
        stability_rows.append(
            {
                "policy": policy,
                "period": "Jan. 2024–Aug. 2025",
                "net_sharpe": result["historical_test"][key]["net_sharpe"],
            }
        )
    stability = pd.DataFrame(stability_rows)

    candidate_summary = pd.read_parquet(result_root / "selection/candidate_summary.parquet")
    candidates = candidate_summary[
        ["policy_id", "kind", "annualized_net_excess_return", "net_sharpe", "intervention_count"]
    ].copy()
    candidates["candidate"] = candidates["policy_id"].map(_candidate_label)
    candidates = candidates.sort_values("net_sharpe", ascending=True).reset_index(drop=True)

    control_monthly = pd.read_parquet(
        V6_6_ROOT / "historical_test/monthly_control/monthly.parquet"
    )[["holding_month", "net_excess_return"]]
    xgb_monthly = pd.read_parquet(
        V6_6_ROOT / "historical_test/selected_predictive/monthly.parquet"
    )[["holding_month", "net_excess_return"]]
    challenger_monthly = pd.read_parquet(
        result_root / "historical_test/selected_v6_7_challenger/monthly.parquet"
    )[["holding_month", "net_excess_return"]]
    deltas = control_monthly.merge(
        challenger_monthly, on="holding_month", suffixes=("_control", "_challenger")
    ).merge(
        xgb_monthly.rename(columns={"net_excess_return": "net_excess_return_xgb"}),
        on="holding_month",
        validate="one_to_one",
    )
    deltas["difference_vs_control"] = (
        deltas["net_excess_return_challenger"] - deltas["net_excess_return_control"]
    )
    deltas["difference_vs_xgboost"] = (
        deltas["net_excess_return_challenger"] - deltas["net_excess_return_xgb"]
    )
    monthly_deltas = deltas.melt(
        id_vars="holding_month",
        value_vars=["difference_vs_control", "difference_vs_xgboost"],
        var_name="comparison",
        value_name="monthly_difference",
    )
    monthly_deltas["comparison"] = monthly_deltas["comparison"].replace(
        {
            "difference_vs_control": "Versus monthly control",
            "difference_vs_xgboost": "Versus frozen XGBoost",
        }
    )

    perf_columns = [
        "policy_label",
        "annualized_net_excess_return",
        "annualized_realized_volatility",
        "net_sharpe",
        "maximum_drawdown",
        "intervention_count",
        "paired_hac_t_vs_control",
    ]
    performance_table = performance[perf_columns].copy()
    sources = [
        {
            "id": "historical_performance",
            "label": "V6.7 frozen historical comparison",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": _sql_values(performance_table, perf_columns, "t"),
                "description": "Compared the frozen monthly control, frozen V6.6 XGBoost overlay, and validation-selected V6.7 neural ensemble.",
                "tables_used": [
                    "research_v6_6/historical_test/monthly_control/monthly.parquet",
                    "research_v6_6/historical_test/selected_predictive/monthly.parquet",
                    "research_v6_7/historical_test/selected_v6_7_challenger/monthly.parquet",
                ],
                "filters": [
                    "Holding months January 2024 through August 2025",
                    "Starting gross exposure capped at 1.80x",
                    "Broker-current financing and drift-aware daily costs",
                ],
                "metric_definitions": [
                    "Annualized net excess return is 12 times the equal-month mean net return minus compounded monthly risk-free return.",
                    "Net Sharpe is annualized mean monthly net excess return divided by annualized monthly volatility.",
                    "Maximum drawdown is measured on the daily after-cost NAV path.",
                ],
            },
        },
        {
            "id": "daily_wealth",
            "label": "V6.7 daily after-cost wealth paths",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": "SELECT calendar_date, policy, 100000 * nav_end / FIRST_VALUE(nav_start) OVER (PARTITION BY policy ORDER BY calendar_date) AS wealth_100k FROM frozen_daily_replays ORDER BY calendar_date, policy",
                "description": "Normalized each saved daily replay to a $100,000 starting value.",
                "tables_used": [
                    "research_v6_6/historical_test/monthly_control/daily.parquet",
                    "research_v6_6/historical_test/selected_predictive/daily.parquet",
                    "research_v6_7/historical_test/selected_v6_7_challenger/daily.parquet",
                ],
            },
        },
        {
            "id": "validation_candidates",
            "label": "V6.7 validation candidate comparison",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": _sql_values(
                    candidates,
                    ["candidate", "annualized_net_excess_return", "net_sharpe", "intervention_count"],
                    "t",
                ),
                "description": "Ranked all 12 MLP/ensemble policies and the frozen XGBoost comparator on 2023 monthly net Sharpe.",
                "tables_used": ["research_v6_7/selection/candidate_summary.parquet"],
                "filters": ["Holding months January–December 2023"],
            },
        },
        {
            "id": "selection_stability",
            "label": "V6.7 selection-to-historical stability",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": _sql_values(stability, ["policy", "period", "net_sharpe"], "t"),
                "description": "Compared 2023 selection Sharpe with January 2024–August 2025 historical Sharpe for the three decision-relevant policies.",
                "tables_used": [
                    "research_v6_6/selection/selection.json",
                    "research_v6_7/selection/selection.json",
                    "research_v6_7/result.json",
                ],
            },
        },
        {
            "id": "monthly_differences",
            "label": "V6.7 paired monthly differences",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": _sql_values(
                    monthly_deltas,
                    ["holding_month", "comparison", "monthly_difference"],
                    "t",
                ),
                "description": "Subtracted each incumbent monthly net excess return from the V6.7 ensemble month by month.",
                "tables_used": [
                    "research_v6_6/historical_test/monthly_control/monthly.parquet",
                    "research_v6_6/historical_test/selected_predictive/monthly.parquet",
                    "research_v6_7/historical_test/selected_v6_7_challenger/monthly.parquet",
                ],
            },
        },
    ]
    charts = [
        {
            "id": "wealth_path",
            "title": "$100,000 historical wealth paths",
            "subtitle": "Daily after-cost NAV from January 2024 through August 2025; all policies begin from the same normalized value.",
            "type": "line",
            "dataset": "wealth_path",
            "sourceId": "daily_wealth",
            "valueFormat": "currency",
            "unit": "$",
            "encodings": {
                "x": {"field": "calendar_date", "type": "temporal", "label": "Date"},
                "y": {"field": "wealth_100k", "type": "quantitative", "label": "Portfolio value"},
                "color": {"field": "policy", "type": "nominal", "label": "Policy"},
                "tooltip": [
                    {"field": "calendar_date", "type": "temporal", "label": "Date"},
                    {"field": "wealth_100k", "type": "quantitative", "label": "Value", "format": "currency"},
                ],
            },
            "compatibleTypes": ["line"],
            "maxRows": len(wealth),
        },
        {
            "id": "candidate_sharpe",
            "title": "2023 validation Sharpe by daily model",
            "subtitle": "Twelve bounded neural policies plus the frozen XGBoost comparator; higher values rank better, although every value is negative.",
            "type": "bar",
            "dataset": "validation_candidates",
            "sourceId": "validation_candidates",
            "valueFormat": "number",
            "unit": "Sharpe",
            "encodings": {
                "x": {"field": "candidate", "type": "nominal", "label": "Candidate"},
                "y": {"field": "net_sharpe", "type": "quantitative", "label": "Net Sharpe"},
                "tooltip": [
                    {"field": "net_sharpe", "type": "quantitative", "label": "Sharpe", "format": "number"},
                    {"field": "annualized_net_excess_return", "type": "quantitative", "label": "Net excess return", "format": "percent"},
                ],
            },
            "compatibleTypes": ["bar"],
            "maxRows": len(candidates),
        },
        {
            "id": "stability",
            "title": "Net Sharpe by evidence period",
            "subtitle": "2023 selected one neural challenger; January 2024–August 2025 was then evaluated once with frozen thresholds.",
            "type": "bar",
            "dataset": "stability",
            "sourceId": "selection_stability",
            "valueFormat": "number",
            "unit": "Sharpe",
            "encodings": {
                "x": {"field": "policy", "type": "nominal", "label": "Policy"},
                "y": {"field": "net_sharpe", "type": "quantitative", "label": "Net Sharpe"},
                "color": {"field": "period", "type": "nominal", "label": "Evidence period"},
                "tooltip": [
                    {"field": "net_sharpe", "type": "quantitative", "label": "Sharpe", "format": "number"}
                ],
            },
            "compatibleTypes": ["bar"],
            "maxRows": len(stability),
        },
        {
            "id": "monthly_deltas",
            "title": "Monthly return difference of the neural ensemble",
            "subtitle": "Challenger minus incumbent monthly net excess return; January 2024–August 2025.",
            "type": "bar",
            "dataset": "monthly_deltas",
            "sourceId": "monthly_differences",
            "valueFormat": "percent",
            "unit": "%",
            "encodings": {
                "x": {"field": "holding_month", "type": "nominal", "label": "Holding month"},
                "y": {"field": "monthly_difference", "type": "quantitative", "label": "Monthly difference"},
                "color": {"field": "comparison", "type": "nominal", "label": "Comparison"},
                "tooltip": [
                    {"field": "monthly_difference", "type": "quantitative", "label": "Difference", "format": "percent"}
                ],
            },
            "compatibleTypes": ["bar"],
            "maxRows": len(monthly_deltas),
        },
    ]
    table = {
        "id": "performance_table",
        "title": "Historical daily-overlay metrics",
        "subtitle": "Exact January 2024–August 2025 after-cost results; paired HAC t-statistics use monthly differences versus the control.",
        "dataset": "performance",
        "sourceId": "historical_performance",
        "defaultSort": {"field": "net_sharpe", "direction": "desc"},
        "density": "spacious",
        "columns": [
            {"field": "policy_label", "label": "Policy", "type": "text"},
            {"field": "annualized_net_excess_return", "label": "Ann. net excess", "format": "percent"},
            {"field": "annualized_realized_volatility", "label": "Volatility", "format": "percent"},
            {"field": "net_sharpe", "label": "Sharpe", "format": "number"},
            {"field": "maximum_drawdown", "label": "Max drawdown", "format": "percent"},
            {"field": "intervention_count", "label": "Actions", "format": "number"},
            {"field": "paired_hac_t_vs_control", "label": "Paired HAC t", "format": "number"},
        ],
    }
    generated = dt.datetime.now(dt.timezone(dt.timedelta(hours=3))).isoformat(timespec="seconds")
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Did the Daily MLP Improve Portfolio Management?",
            "description": "Technical decision report comparing compact daily neural overlays with the frozen V6.6 XGBoost intervention and monthly-control portfolio.",
            "generatedAt": generated,
            "sources": sources,
            "charts": charts,
            "tables": [table],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Did the Daily MLP Improve Portfolio Management?"},
                {"id": "summary", "type": "markdown", "sourceId": "historical_performance", "body": "## Technical summary\n\n- **The neural model improved the tested daily overlay.** The validation-selected five-day equal MLP–XGBoost ensemble reached 25.18% annualized net excess return and 2.132 Sharpe, versus 23.00% and 1.913 for frozen XGBoost alone.\n- **It also beat the monthly control economically.** The control earned 21.63% at 1.961 Sharpe; the ensemble added 3.55 annualized percentage points, reduced daily maximum drawdown from −8.21% to −6.47%, and ended a normalized $100,000 path near $162.7k.\n- **The improvement is not reliable enough for promotion.** The paired monthly HAC t-statistic is 0.96 versus the control and 1.19 versus XGBoost. The monthly policy remains primary; the ensemble is suitable only for prospective shadow logging."},
                {"id": "wealth_text", "type": "markdown", "sourceId": "daily_wealth", "body": "## The neural ensemble compounded faster without increasing realized volatility\n\nAll three paths use identical monthly targets, financing assumptions, trading costs, and next-session-close execution. The selected ensemble trims or exits only existing positions. Its higher ending wealth came with 11.81% realized volatility, below XGBoost's 12.03% and only moderately above the control's 11.03%."},
                {"id": "wealth_chart", "type": "chart", "chartId": "wealth_path", "layout": "full"},
                {"id": "candidate_text", "type": "markdown", "sourceId": "validation_candidates", "body": "## MLP helped as a complement, not as a standalone replacement\n\nThe pure one-day and five-day MLP policies did not beat frozen XGBoost in 2023. The two best neural candidates were both five-day equal ensembles. Signed-return regression narrowly won the frozen selection rule, so only that challenger entered the historical period. This preserves the distinction between testing MLP and selecting a test-period winner."},
                {"id": "candidate_chart", "type": "chart", "chartId": "candidate_sharpe", "layout": "full"},
                {"id": "scope", "type": "markdown", "body": "## Scope, data, and metric definitions\n\nThe unit of prediction is an existing stock-position day. Each MLP receives the same 850 encoded monthly variables and 38 causal daily price, liquidity, position-state, market, and related-news variables used in V6.6. Training uses label-available observations only: 2022 is the initial fit, 2023 selects one policy, and seven quarterly expanding refits cover January 2024–August 2025. Net excess return includes modeled transaction costs, security borrow, and 6.25% annual financing above 1× gross. Sharpe is the annualized equal-month mean net excess return divided by monthly volatility."},
                {"id": "method", "type": "markdown", "body": "## Model and experimental design\n\nTwo fixed 128–64 ReLU networks were tested: a winsorized signed-return regression and a probability model for adverse signed return. Both used 10% dropout, AdamW regularization, five epochs, and a deterministic three-seed average. One- and five-day horizons were crossed with selective and broad action thresholds. Equal MLP–XGBoost ensembles were allowed only at the frozen XGBoost model's five-day horizon. Validation scaling and action thresholds were frozen before the historical replay; no stock could be added or re-entered intramonth."},
                {"id": "stability_text", "type": "markdown", "sourceId": "selection_stability", "body": "## The historical uplift followed a weak validation year\n\nThe ensemble improved the 2023 XGBoost Sharpe from −0.336 to −0.232, but both remained negative. Performance then rose sharply for all policies in 2024–2025. This regime shift does not invalidate the chronology, but it limits how confidently the 20-month historical uplift can be attributed to a stable neural edge."},
                {"id": "stability_chart", "type": "chart", "chartId": "stability", "layout": "full"},
                {"id": "uncertainty", "type": "markdown", "sourceId": "monthly_differences", "body": "## The gain is economically meaningful but statistically concentrated\n\nThe ensemble beat the monthly control in 60% of historical months and frozen XGBoost in 70%. Several months contributed disproportionately, leaving paired HAC t-statistics below the 1.96 promotion gate. With only 20 monthly observations and a design opened after V6.6 outcomes were visible, the correct status is post-diagnostic mechanism evidence—not untouched confirmation."},
                {"id": "delta_chart", "type": "chart", "chartId": "monthly_deltas", "layout": "full"},
                {"id": "exact", "type": "markdown", "body": "## Exact historical results"},
                {"id": "performance_table_block", "type": "table", "tableId": "performance_table", "layout": "full"},
                {"id": "next", "type": "markdown", "body": "## Recommended next steps\n\n1. Keep the frozen monthly 1.80× portfolio as the paper-trading primary.\n2. Log the five-day MLP–XGBoost ensemble in shadow mode without allowing it to place intramonth trades.\n3. Retain the exposure-safety guardrail as an operational control.\n4. Compare proposed and realized shadow actions prospectively from September 2026 before reconsidering the trading policy."},
                {"id": "questions", "type": "markdown", "body": "## Further questions\n\n- Does the ensemble's uplift persist prospectively across both long and short sleeves?\n- Are benefits broad across ordinary months or concentrated in a few market states?\n- Do real-time news updates, paper fills, and borrow availability preserve the backtest action timing?"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "wealth_path": _rows(wealth),
                "validation_candidates": _rows(candidates),
                "stability": _rows(stability),
                "monthly_deltas": _rows(monthly_deltas),
                "performance": _rows(performance_table),
            },
        },
        "sources": sources,
    }
    (result_root / "artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n"
    )
    (result_root / "chart_map.json").write_text(
        json.dumps(
            [
                {"section": "Compounding", "question": "How did $100k compound?", "family": "trend", "chart": "wealth_path", "palette": "relaxed three-category", "source": "daily after-cost replays"},
                {"section": "Candidate comparison", "question": "Which daily model won validation?", "family": "comparison and ranking", "chart": "candidate_sharpe", "palette": "single-root preferred", "source": "2023 candidate summaries"},
                {"section": "Stability", "question": "Did the ordering generalize?", "family": "comparison", "chart": "stability", "palette": "hard two-root cap", "source": "selection and historical summaries"},
                {"section": "Uncertainty", "question": "Were gains broad by month?", "family": "comparison", "chart": "monthly_deltas", "palette": "hard two-root cap", "source": "paired monthly replays"},
            ],
            indent=2,
        )
        + "\n"
    )
    _build_notebook(result_root, result_root / "v6_7_daily_mlp_overlay_audit.ipynb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    arguments = parser.parse_args()
    build(arguments.result_root.resolve())


if __name__ == "__main__":
    main()
