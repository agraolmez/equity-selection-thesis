#!/usr/bin/env python3
"""Build the reproducible notebook and portable report for V6.6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import nbformat
import numpy as np
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / (
    "artifacts/research_v6_6/intrmonth_interventions/"
    "566f3749ee9b-16046ee58521-a82fbba1983b"
)


def _read(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"SELECT * FROM read_parquet('{str(path).replace(chr(39), chr(39)*2)}')").df()


def _rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.replace([np.inf, -np.inf], np.nan)
    return json.loads(clean.to_json(orient="records", date_format="iso"))


def _sql_values(frame: pd.DataFrame, columns: list[str], table: str) -> str:
    values = []
    for row in frame[columns].itertuples(index=False, name=None):
        encoded = []
        for value in row:
            if pd.isna(value):
                encoded.append("NULL")
            elif isinstance(value, str):
                encoded.append("'" + value.replace("'", "''") + "'")
            else:
                encoded.append(repr(float(value)) if isinstance(value, (float, np.floating)) else str(int(value)))
        values.append("(" + ",".join(encoded) + ")")
    return f"SELECT * FROM (VALUES {','.join(values)}) AS {table}({','.join(columns)})"


def _build_notebook(result_root: Path, output: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Research V6.6 intramonth intervention audit\n\n"
            "This notebook reproduces the saved performance comparison and the key accounting, action, and causality checks."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json, duckdb, numpy as np, pandas as pd\n"
            f"root = Path({str(result_root)!r})\n"
            "result = json.loads((root/'result.json').read_text())\n"
            "result['decision']"
        ),
        nbformat.v4.new_markdown_cell("## Data quality and replay reconciliation"),
        nbformat.v4.new_code_cell(
            "pd.DataFrame([result['data_quality'], result['control_reconciliation']['historical_test']]).T"
        ),
        nbformat.v4.new_markdown_cell("## Validation-selected policies"),
        nbformat.v4.new_code_cell(
            "sel = result['selection']\n"
            "pd.DataFrame([\n"
            " {'policy':'control', **sel['control']},\n"
            " {'policy':sel['selected_rule']['policy_id'], **sel['selected_rule']['summary']},\n"
            " {'policy':sel['selected_predictive']['policy_id'], **sel['selected_predictive']['summary']},\n"
            "])[['policy','annualized_net_excess_return','annualized_realized_volatility','net_sharpe','maximum_drawdown','intervention_count']]"
        ),
        nbformat.v4.new_markdown_cell("## Historical evaluation"),
        nbformat.v4.new_code_cell(
            "historical = pd.DataFrame([{'policy':k, **v, **result['paired_differences'].get(k,{})} for k,v in result['historical_test'].items()])\n"
            "historical[['policy','annualized_net_excess_return','annualized_realized_volatility','net_sharpe','maximum_drawdown','intervention_count','paired_hac_t']]"
        ),
        nbformat.v4.new_markdown_cell("## Mechanical audit"),
        nbformat.v4.new_code_cell(
            "issues=[]\n"
            "for phase in ['selection','historical_test']:\n"
            "  for summary_path in (root/phase).glob('*/summary.json'):\n"
            "    p=summary_path.parent\n"
            "    m=duckdb.sql(f\"select * from read_parquet('{p/'monthly.parquet'}') order by holding_month\").df()\n"
            "    d=duckdb.sql(f\"select * from read_parquet('{p/'daily.parquet'}') order by calendar_date\").df()\n"
            "    a=duckdb.sql(f\"select * from read_parquet('{p/'actions.parquet'}')\").df()\n"
            "    assert np.allclose(m.nav_end.to_numpy()[:-1],m.nav_start.to_numpy()[1:],atol=1e-7,rtol=0)\n"
            "    assert np.isfinite(d[['nav_start','nav_end','gross_end']].to_numpy()).all()\n"
            "    if len(a): assert (a.new_fraction <= a.old_fraction+1e-12).all() and (a.new_fraction>=0).all()\n"
            "receipts=0\n"
            "for p in root.glob('models/**/*.json'):\n"
            "  x=json.loads(p.read_text())\n"
            "  if x.get('schema')=='research_v6_6_predictive_refit_v1':\n"
            "    receipts+=1; assert x['train_last_month'] < x['refit_month']\n"
            "{'mechanical_issues':issues,'causal_model_receipts':receipts}"
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    executed = nbformat.read(output, as_version=4)
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(executed, {"metadata": {"path": str(ROOT)}})
    nbformat.write(executed, output)


def build(result_root: Path) -> None:
    result = json.loads((result_root / "result.json").read_text())
    policies = list(result["historical_test"])
    performance = pd.DataFrame(
        [
            {
                "policy": policy,
                **summary,
                "paired_hac_t": result["paired_differences"].get(policy, {}).get("paired_hac_t"),
            }
            for policy, summary in result["historical_test"].items()
        ]
    )
    performance["policy_label"] = performance["policy"].replace(
        {
            "monthly_control": "Monthly control",
            "safety_only": "Safety only",
            "selected_rule": "15% stop-loss",
            "selected_predictive": "Predictive XGBoost",
            "selected_predictive_plus_safety": "Predictive + safety",
        }
    )
    wealth_parts = []
    wealth_policies = [
        policy for policy in policies if policy != "selected_predictive_plus_safety"
    ]
    for policy in wealth_policies:
        daily = _read(result_root / "historical_test" / policy / "daily.parquet").sort_values("calendar_date")
        base = float(daily["nav_start"].iloc[0])
        daily = daily[["calendar_date", "nav_end"]].copy()
        daily["policy"] = performance.set_index("policy").at[policy, "policy_label"]
        daily["wealth_100k"] = 100_000.0 * daily["nav_end"] / base
        wealth_parts.append(daily[["calendar_date", "policy", "wealth_100k"]])
    wealth = pd.concat(wealth_parts, ignore_index=True)

    selection_rows = []
    selection = result["selection"]
    for label, summary in (
        ("Monthly control", selection["control"]),
        ("15% stop-loss", selection["selected_rule"]["summary"]),
        ("Predictive XGBoost", selection["selected_predictive"]["summary"]),
    ):
        selection_rows.append({"policy": label, "period": "2023 selection", "net_sharpe": summary["net_sharpe"]})
    for policy, label in (("monthly_control", "Monthly control"), ("selected_rule", "15% stop-loss"), ("selected_predictive", "Predictive XGBoost")):
        selection_rows.append({"policy": label, "period": "2024–Aug. 2025", "net_sharpe": result["historical_test"][policy]["net_sharpe"]})
    stability = pd.DataFrame(selection_rows)

    base = _read(result_root / "historical_test" / "monthly_control" / "monthly.parquet")[["holding_month", "net_excess_return"]]
    deltas = []
    for policy, label in (("selected_rule", "15% stop-loss"), ("selected_predictive", "Predictive XGBoost")):
        challenger = _read(result_root / "historical_test" / policy / "monthly.parquet")[["holding_month", "net_excess_return"]]
        joined = base.merge(challenger, on="holding_month", suffixes=("_control", "_policy"), validate="one_to_one")
        joined["policy"] = label
        joined["monthly_difference"] = joined["net_excess_return_policy"] - joined["net_excess_return_control"]
        deltas.append(joined[["holding_month", "policy", "monthly_difference"]])
    monthly_deltas = pd.concat(deltas, ignore_index=True)

    perf_columns = [
        "policy_label", "annualized_net_excess_return", "annualized_realized_volatility",
        "net_sharpe", "maximum_drawdown", "intervention_count", "paired_hac_t",
    ]
    performance_table = performance[perf_columns].copy()
    sources = [
        {
            "id": "historical_performance",
            "label": "V6.6 historical intervention performance",
            "query": {
                "engine": "DuckDB", "language": "sql",
                "sql": _sql_values(performance_table, perf_columns, "t"),
                "description": "Loaded the five validation-frozen historical policies and recomputed their monthly after-cost performance summaries.",
                "tables_used": ["historical_test/*/monthly.parquet", "historical_test/*/daily.parquet"],
                "filters": ["Holding months 2024-01 through 2025-08", "Broker-current financing", "Starting monthly gross capped at 1.80x"],
                "metric_definitions": [
                    "Annualized net excess return is 12 times the equal-month mean net return minus compounded monthly risk-free return.",
                    "Net Sharpe is annualized mean monthly net excess return divided by annualized monthly volatility.",
                    "Maximum drawdown is measured from the reconstructed daily NAV path.",
                ],
            },
        },
        {
            "id": "daily_wealth",
            "label": "V6.6 daily wealth paths",
            "query": {
                "engine": "DuckDB", "language": "sql",
                "sql": "SELECT calendar_date, policy, 100000 * nav_end / FIRST_VALUE(nav_start) OVER (PARTITION BY policy ORDER BY calendar_date) AS wealth_100k FROM historical_daily_replays ORDER BY calendar_date, policy",
                "description": "Normalized each saved daily policy NAV path to a $100,000 starting value.",
                "tables_used": ["historical_test/*/daily.parquet"],
                "filters": ["Calendar dates in holding months 2024-01 through 2025-08"],
            },
        },
        {
            "id": "selection_stability",
            "label": "V6.6 selection-to-test stability",
            "query": {
                "engine": "DuckDB", "language": "sql",
                "sql": _sql_values(stability, ["policy", "period", "net_sharpe"], "t"),
                "description": "Compared the 2023 selection Sharpe with the frozen January 2024–August 2025 evaluation Sharpe.",
                "tables_used": ["selection/*/monthly.parquet", "historical_test/*/monthly.parquet"],
            },
        },
        {
            "id": "monthly_differences",
            "label": "V6.6 paired monthly intervention differences",
            "query": {
                "engine": "DuckDB", "language": "sql",
                "sql": _sql_values(monthly_deltas, ["holding_month", "policy", "monthly_difference"], "t"),
                "description": "Subtracted the monthly control net excess return from each selected intervention month by month.",
                "tables_used": ["historical_test/monthly_control/monthly.parquet", "historical_test/selected_rule/monthly.parquet", "historical_test/selected_predictive/monthly.parquet"],
            },
        },
    ]
    charts = [
        {
            "id": "wealth_path", "title": "$100,000 historical wealth paths",
            "subtitle": "Daily after-cost NAV, normalized at the start of January 2024; evaluation ends August 2025.",
            "type": "line", "dataset": "wealth_path", "sourceId": "daily_wealth",
            "valueFormat": "currency", "unit": "$",
            "encodings": {
                "x": {"field": "calendar_date", "type": "temporal", "label": "Date"},
                "y": {"field": "wealth_100k", "type": "quantitative", "label": "Portfolio value"},
                "color": {"field": "policy", "type": "nominal", "label": "Policy"},
                "tooltip": [
                    {"field": "calendar_date", "type": "temporal", "label": "Date"},
                    {"field": "wealth_100k", "type": "quantitative", "label": "Value", "format": "currency"},
                ],
            }, "compatibleTypes": ["line"], "maxRows": len(wealth),
        },
        {
            "id": "stability", "title": "Net Sharpe by evidence period",
            "subtitle": "2023 selects the rule and predictive overlay; January 2024–August 2025 is evaluated once with frozen decisions.",
            "type": "bar", "dataset": "stability", "sourceId": "selection_stability",
            "valueFormat": "number", "unit": "Sharpe",
            "encodings": {
                "x": {"field": "policy", "type": "nominal", "label": "Policy"},
                "y": {"field": "net_sharpe", "type": "quantitative", "label": "Net Sharpe"},
                "color": {"field": "period", "type": "nominal", "label": "Evidence period"},
                "tooltip": [{"field": "net_sharpe", "type": "quantitative", "label": "Sharpe", "format": "number"}],
            }, "compatibleTypes": ["bar"], "maxRows": len(stability),
        },
        {
            "id": "return_comparison", "title": "Annualized net excess return",
            "subtitle": "Broker-current financing and daily drift-aware costs; January 2024–August 2025.",
            "type": "bar", "dataset": "performance", "sourceId": "historical_performance",
            "valueFormat": "percent", "unit": "%",
            "encodings": {
                "x": {"field": "policy_label", "type": "nominal", "label": "Policy"},
                "y": {"field": "annualized_net_excess_return", "type": "quantitative", "label": "Annualized net excess return"},
                "tooltip": [
                    {"field": "annualized_net_excess_return", "type": "quantitative", "label": "Net excess return", "format": "percent"},
                    {"field": "net_sharpe", "type": "quantitative", "label": "Sharpe", "format": "number"},
                    {"field": "maximum_drawdown", "type": "quantitative", "label": "Max drawdown", "format": "percent"},
                ],
            }, "compatibleTypes": ["bar"], "maxRows": len(performance),
        },
        {
            "id": "monthly_deltas", "title": "Monthly return difference versus control",
            "subtitle": "Intervention net excess return minus monthly-control net excess return; percentage-point scale.",
            "type": "bar", "dataset": "monthly_deltas", "sourceId": "monthly_differences",
            "valueFormat": "percent", "unit": "%",
            "encodings": {
                "x": {"field": "holding_month", "type": "nominal", "label": "Holding month"},
                "y": {"field": "monthly_difference", "type": "quantitative", "label": "Monthly difference"},
                "color": {"field": "policy", "type": "nominal", "label": "Policy"},
                "tooltip": [{"field": "monthly_difference", "type": "quantitative", "label": "Difference", "format": "percent"}],
            }, "compatibleTypes": ["bar"], "maxRows": len(monthly_deltas),
        },
    ]
    table = {
        "id": "performance_table", "title": "Historical intervention metrics",
        "subtitle": "Exact January 2024–August 2025 results; paired HAC t-statistics compare monthly net excess returns with the control.",
        "dataset": "performance", "sourceId": "historical_performance",
        "defaultSort": {"field": "net_sharpe", "direction": "desc"}, "density": "spacious",
        "columns": [
            {"field": "policy_label", "label": "Policy", "type": "text"},
            {"field": "annualized_net_excess_return", "label": "Ann. net excess", "format": "percent"},
            {"field": "annualized_realized_volatility", "label": "Volatility", "format": "percent"},
            {"field": "net_sharpe", "label": "Sharpe", "format": "number"},
            {"field": "maximum_drawdown", "label": "Max drawdown", "format": "percent"},
            {"field": "intervention_count", "label": "Actions", "format": "number"},
            {"field": "paired_hac_t", "label": "Paired HAC t", "format": "number"},
        ],
    }
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report",
            "title": "Should We Intervene Between Monthly Rebalances?",
            "description": "Decision report on daily safety, stop-loss, take-profit, trailing-stop, and adaptive predictive overlays for the frozen monthly V6.1 strategy.",
            "generatedAt": "2026-08-16T09:30:00+03:00",
            "sources": sources, "charts": charts, "tables": [table],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Should We Intervene Between Monthly Rebalances?"},
                {"id": "executive", "type": "markdown", "sourceId": "historical_performance", "body": "## Executive Summary\n\n- **Keep the monthly rebalance as the investment policy.** No validation-frozen intervention passed every return, Sharpe, drawdown, cost, and paired-inference gate.\n- **The predictive overlay raised return but not risk-adjusted performance.** Annualized net excess return increased from 21.63% to 23.00%, while volatility rose from 11.03% to 12.03% and Sharpe fell from 1.961 to 1.913. The paired improvement was not reliable (HAC t = 0.62).\n- **The 15% stop-loss did not generalize.** It was strong in the difficult 2023 selection year but cut profitable recoveries later, reducing historical annualized net excess return to 15.78%.\n- **Retain daily monitoring only as an operational guardrail.** The 1.95x gross-exposure safety rule fired once and was economically neutral; it is useful for broker compliance, not a new source of alpha."},
                {"id": "wealth_text", "type": "markdown", "sourceId": "daily_wealth", "body": "## Monthly discipline captured the strongest compounding\n\nThe predictive overlay occasionally avoided meaningful losses and ended with more return, but its path was more variable. The stop-loss rule repeatedly removed positions that subsequently recovered. The wealth paths show why a mechanically smaller drawdown does not automatically imply a higher Sharpe or a better long-run allocation rule."},
                {"id": "wealth_chart", "type": "chart", "chartId": "wealth_path", "layout": "full"},
                {"id": "stability_text", "type": "markdown", "sourceId": "selection_stability", "body": "## The stop-loss result was regime-specific\n\nThe 15% signed-position stop was selected because it strongly improved a weak 2023 control. Its ordering reversed in 2024–2025. The predictive XGBoost overlay generalized better on return, but not on Sharpe or paired statistical evidence. This validation-to-test reversal is the central reason not to replace monthly rebalancing."},
                {"id": "stability_chart", "type": "chart", "chartId": "stability", "layout": "full"},
                {"id": "return_text", "type": "markdown", "sourceId": "historical_performance", "body": "## Higher predictive return came with higher volatility\n\nThe causal depth-2 XGBoost overlay refitted quarterly on expanding, label-available history and used all 850 encoded monthly features plus 38 daily state variables. It lifted annualized net excess return by 1.37 percentage points and improved maximum drawdown from −8.21% to −6.58%, but neither the Sharpe change nor the paired monthly return difference supported promotion."},
                {"id": "return_chart", "type": "chart", "chartId": "return_comparison", "layout": "full"},
                {"id": "delta_text", "type": "markdown", "sourceId": "monthly_differences", "body": "## Intervention gains were concentrated in a few months\n\nThe predictive overlay's advantage came mainly from several isolated months and was offset by missed rebounds elsewhere. The stop-loss rule had even larger reversals. A deployable daily layer should improve many independent months or show a clear crash-state mechanism, not depend on a small number of exits."},
                {"id": "delta_chart", "type": "chart", "chartId": "monthly_deltas", "layout": "full"},
                {"id": "table_text", "type": "markdown", "body": "## Exact results"},
                {"id": "performance_table_block", "type": "table", "tableId": "performance_table", "layout": "full"},
                {"id": "next", "type": "markdown", "body": "## Recommended next step\n\n1. Deploy the frozen monthly 1.80x strategy as the paper-trading primary.\n2. Run daily gross-exposure monitoring and proportional safety deleveraging as an operational control.\n3. Log the predictive XGBoost overlay in shadow mode without allowing it to trade.\n4. Reconsider intramonth execution only after prospective evidence beginning no earlier than September 2026.\n\nThe shadow log should record proposed actions, realized post-action returns, simulated costs, and whether improvements remain broad across months and both sleeves."},
                {"id": "questions", "type": "markdown", "body": "## Further questions\n\n- Does the predictive overlay help specifically during broad market shocks rather than ordinary months?\n- Are its benefits concentrated in long positions, short positions, or news-heavy names?\n- Do paper fills and real-time article updates preserve the historical action timing?"},
                {"id": "caveats", "type": "markdown", "sourceId": "historical_performance", "body": "## Caveats and assumptions\n\nThis is post-diagnostic historical development evidence, not untouched or live confirmation. The historical evaluation contains 20 monthly observations. Daily return coverage is 99.53% before explicit exit-date accounting fallbacks. Close information executes at the following session's close; no same-close fill is assumed. The replay uses broker-current financing sensitivity and security-level borrow estimates, while real paper trading may omit or simplify some charges."},
            ],
        },
        "snapshot": {
            "version": 1, "generatedAt": "2026-08-16T09:30:00+03:00", "status": "ready",
            "datasets": {
                "wealth_path": _rows(wealth),
                "stability": _rows(stability),
                "performance": _rows(performance_table),
                "monthly_deltas": _rows(monthly_deltas),
            },
        },
        "sources": sources,
    }
    (result_root / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n")
    (result_root / "chart_map.json").write_text(
        json.dumps(
            [
                {"section": "Monthly discipline", "question": "How did $100k compound?", "family": "trend", "chart": "wealth_path", "palette": "hard two-root cap plus neutral extensions", "source": "historical daily replays"},
                {"section": "Stability", "question": "Did validation generalize?", "family": "comparison", "chart": "stability", "palette": "hard two-root cap", "source": "selection and historical summaries"},
                {"section": "Return", "question": "Which policy earned more?", "family": "comparison", "chart": "return_comparison", "palette": "single-root preferred", "source": "historical summaries"},
                {"section": "Monthly deltas", "question": "Were gains broad?", "family": "comparison", "chart": "monthly_deltas", "palette": "hard two-root cap", "source": "paired monthly replays"},
            ], indent=2,
        ) + "\n"
    )
    _build_notebook(result_root, result_root / "v6_6_intramonth_intervention_audit.ipynb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    build(args.result_root.resolve())


if __name__ == "__main__":
    main()
