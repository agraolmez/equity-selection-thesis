#!/usr/bin/env python3
"""Build the canonical portable technical report for V6.8."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / (
    "artifacts/research_v6_8/daily_reoptimization/"
    "d55a1db689f9-2ef39b0755c5-2e422abd519f"
)
V6_6_ROOT = ROOT / (
    "artifacts/research_v6_6/intrmonth_interventions/"
    "566f3749ee9b-16046ee58521-a82fbba1983b"
)
V6_7_ROOT = ROOT / (
    "artifacts/research_v6_7/daily_mlp_overlay/"
    "4108d1f64761-16046ee58521-22f799195507"
)


def _duck(path: Path) -> str:
    return str(path).replace("'", "''")


def _read(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"SELECT * FROM read_parquet('{_duck(path)}')").df()


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


def build(result_root: Path) -> None:
    result = json.loads((result_root / "result.json").read_text())
    labels = {
        "monthly_control": "Monthly control",
        "v6_7_cash_overlay": "V6.7 warnings · cash",
        "v6_8_reoptimization": "V6.8 warnings · re-optimize",
    }
    performance_rows: list[dict[str, object]] = []
    for key, summary in result["historical_test"].items():
        performance_rows.append(
            {
                "policy": labels[key],
                "annualized_net_excess_return": summary["annualized_net_excess_return"],
                "annualized_realized_volatility": summary["annualized_realized_volatility"],
                "net_sharpe": summary["net_sharpe"],
                "maximum_drawdown": summary["maximum_drawdown"],
                "annualized_total_cost": summary["annualized_total_cost"],
                "actions": summary["intervention_count"],
                "refill_fraction": summary.get("refill_fraction"),
                "paired_hac_t_vs_v6_7": (
                    result["paired_differences"]["historical_versus_v6_7"]["paired_hac_t"]
                    if key == "v6_8_reoptimization"
                    else None
                ),
            }
        )
    performance = pd.DataFrame(performance_rows)

    stability_rows: list[dict[str, object]] = []
    for period, values in (
        ("2023 validation", result["validation"]),
        ("Jan. 2024–Aug. 2025", result["historical_test"]),
    ):
        for key, summary in values.items():
            stability_rows.append(
                {
                    "period": period,
                    "policy": labels[key],
                    "net_sharpe": summary["net_sharpe"],
                    "annualized_net_excess_return": summary["annualized_net_excess_return"],
                }
            )
    stability = pd.DataFrame(stability_rows)

    wealth_parts: list[pd.DataFrame] = []
    daily_paths = {
        "Monthly control": V6_6_ROOT / "historical_test/monthly_control/daily.parquet",
        "V6.7 warnings · cash": V6_7_ROOT / "historical_test/selected_v6_7_challenger/daily.parquet",
        "V6.8 warnings · re-optimize": result_root / "historical_test/daily.parquet",
    }
    for label, path in daily_paths.items():
        daily = _read(path).sort_values("calendar_date", kind="mergesort")
        start = float(daily["nav_start"].iloc[0])
        frame = daily[["calendar_date", "nav_end"]].copy()
        frame["policy"] = label
        frame["wealth_100k"] = 100_000.0 * frame["nav_end"] / start
        wealth_parts.append(frame[["calendar_date", "policy", "wealth_100k"]])
    wealth = pd.concat(wealth_parts, ignore_index=True)

    v6_7_monthly = _read(V6_7_ROOT / "historical_test/selected_v6_7_challenger/monthly.parquet")
    v6_8_monthly = _read(result_root / "historical_test/monthly.parquet")
    monthly_delta = v6_7_monthly[["holding_month", "net_excess_return"]].merge(
        v6_8_monthly[["holding_month", "net_excess_return"]],
        on="holding_month",
        suffixes=("_v6_7", "_v6_8"),
        validate="one_to_one",
    )
    monthly_delta["difference"] = (
        monthly_delta["net_excess_return_v6_8"]
        - monthly_delta["net_excess_return_v6_7"]
    )

    events = _read(result_root / "historical_test/events.parquet")
    refill = (
        events.groupby("holding_month", as_index=False)
        .agg(
            removed_gross=("removed_gross", "sum"),
            refilled_gross=("refill_gross", "sum"),
            event_count=("event_id", "nunique"),
        )
        .sort_values("holding_month")
    )
    refill_long = refill.melt(
        id_vars=["holding_month", "event_count"],
        value_vars=["removed_gross", "refilled_gross"],
        var_name="gross_type",
        value_name="gross_exposure",
    )
    refill_long["gross_type"] = refill_long["gross_type"].replace(
        {"removed_gross": "Removed", "refilled_gross": "Refilled"}
    )

    sources = [
        {
            "id": "historical_performance",
            "label": "V6.8 frozen historical performance comparison",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": _sql_values(performance, list(performance.columns), "performance"),
                "description": "Compared the untouched monthly portfolio, frozen V6.7 cash-holding warning overlay, and V6.8 survivor re-optimizer.",
                "tables_used": [
                    "research_v6_6.historical_test.monthly_control",
                    "research_v6_7.historical_test.selected_v6_7_challenger",
                    "research_v6_8.historical_test.v6_8_survivor_reoptimization",
                ],
                "filters": [
                    "Holding months January 2024 through August 2025",
                    "Identical frozen V6.7 warning execution dates",
                    "Starting gross exposure capped at 1.80x",
                ],
                "metric_definitions": [
                    "Annualized net excess return is 12 times the equal-month mean after-cost return above cash.",
                    "Net Sharpe is annualized mean monthly net excess return divided by annualized monthly volatility.",
                    "Maximum drawdown is measured on the daily after-cost NAV path.",
                ],
            },
        },
        {
            "id": "daily_wealth",
            "label": "Daily after-cost wealth paths",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": "SELECT calendar_date, policy, 100000 * nav_end / FIRST_VALUE(nav_start) OVER (PARTITION BY policy ORDER BY calendar_date) AS wealth_100k FROM frozen_daily_replays ORDER BY calendar_date, policy",
                "description": "Normalized the three saved daily replay paths to a common $100,000 starting value.",
                "tables_used": [
                    "research_v6_6.historical_test.monthly_control.daily",
                    "research_v6_7.historical_test.selected_v6_7_challenger.daily",
                    "research_v6_8.historical_test.daily",
                ],
            },
        },
        {
            "id": "period_stability",
            "label": "Validation and historical policy comparison",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": _sql_values(stability, list(stability.columns), "stability"),
                "description": "Compared annualized return and Sharpe for all three policies in 2023 validation and January 2024–August 2025 historical evidence.",
                "tables_used": ["research_v6_8.result"],
            },
        },
        {
            "id": "paired_monthly_difference",
            "label": "Paired V6.8 minus V6.7 monthly differences",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": _sql_values(
                    monthly_delta,
                    ["holding_month", "net_excess_return_v6_7", "net_excess_return_v6_8", "difference"],
                    "monthly_delta",
                ),
                "description": "Subtracted the V6.7 cash-overlay monthly net excess return from V6.8 month by month.",
                "tables_used": [
                    "research_v6_7.historical_test.selected_v6_7_challenger.monthly",
                    "research_v6_8.historical_test.monthly",
                ],
            },
        },
        {
            "id": "refill_mechanics",
            "label": "V6.8 released and refilled gross by month",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": _sql_values(refill, list(refill.columns), "refill"),
                "description": "Aggregated executed same-close warning batches by holding month.",
                "tables_used": ["research_v6_8.historical_test.events"],
                "metric_definitions": [
                    "Removed gross is the absolute exposure eliminated by the contemporaneous V6.7 warning batch.",
                    "Refilled gross is the same-close monotone exposure added to surviving, never-warned monthly holdings.",
                ],
            },
        },
        {
            "id": "independent_audit",
            "label": "V6.8 independent accounting and constraint audit",
            "query": {
                "engine": "Python",
                "language": "python",
                "description": "Recomputed headline metrics and checked NAV continuity, warning counts, refill budgets, no re-entry, same-side additions, and warned-name freezes.",
                "tables_used": ["research_v6_8.independent_audit"],
            },
        },
    ]

    charts = [
        {
            "id": "wealth_path",
            "title": "$100,000 historical wealth paths",
            "subtitle": "Daily after-cost NAV, January 2024–August 2025; all paths are normalized to the same starting value.",
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
            "id": "period_sharpe",
            "title": "Net Sharpe by evidence period",
            "subtitle": "The re-optimizer won 2023 validation but finished slightly below V6.7 on the 20 historical months.",
            "type": "bar",
            "dataset": "period_stability",
            "sourceId": "period_stability",
            "valueFormat": "number",
            "unit": "Sharpe",
            "encodings": {
                "x": {"field": "policy", "type": "nominal", "label": "Policy"},
                "y": {"field": "net_sharpe", "type": "quantitative", "label": "Net Sharpe"},
                "color": {"field": "period", "type": "nominal", "label": "Evidence period"},
                "tooltip": [
                    {"field": "net_sharpe", "type": "quantitative", "label": "Sharpe", "format": "number"},
                    {"field": "annualized_net_excess_return", "type": "quantitative", "label": "Ann. net excess", "format": "percent"},
                ],
            },
            "compatibleTypes": ["bar"],
            "maxRows": len(stability),
        },
        {
            "id": "monthly_delta",
            "title": "Monthly net-return difference versus V6.7",
            "subtitle": "V6.8 minus V6.7 monthly net excess return, January 2024–August 2025; the mean annualized difference is +0.49 percentage points.",
            "type": "bar",
            "dataset": "monthly_delta",
            "sourceId": "paired_monthly_difference",
            "valueFormat": "percent",
            "unit": "%",
            "encodings": {
                "x": {"field": "holding_month", "type": "nominal", "label": "Holding month"},
                "y": {"field": "difference", "type": "quantitative", "label": "Monthly difference"},
                "tooltip": [
                    {"field": "difference", "type": "quantitative", "label": "Difference", "format": "percent"}
                ],
            },
            "compatibleTypes": ["bar"],
            "maxRows": len(monthly_delta),
        },
        {
            "id": "refill_by_month",
            "title": "Released and refilled gross exposure",
            "subtitle": "The optimizer restored 41.7% of released gross across 157 historical warning batches.",
            "type": "bar",
            "dataset": "refill_by_month",
            "sourceId": "refill_mechanics",
            "valueFormat": "number",
            "unit": "× NAV",
            "encodings": {
                "x": {"field": "holding_month", "type": "nominal", "label": "Holding month"},
                "y": {"field": "gross_exposure", "type": "quantitative", "label": "Summed gross exposure"},
                "color": {"field": "gross_type", "type": "nominal", "label": "Exposure change"},
                "tooltip": [
                    {"field": "gross_exposure", "type": "quantitative", "label": "Gross", "format": "number"},
                    {"field": "event_count", "type": "quantitative", "label": "Action batches", "format": "number"},
                ],
            },
            "compatibleTypes": ["bar"],
            "maxRows": len(refill_long),
        },
    ]
    table = {
        "id": "performance_table",
        "title": "Historical policy metrics",
        "subtitle": "Exact January 2024–August 2025 after-cost results; the paired HAC statistic for V6.8 is measured versus V6.7.",
        "dataset": "performance",
        "sourceId": "historical_performance",
        "defaultSort": {"field": "net_sharpe", "direction": "desc"},
        "density": "spacious",
        "columns": [
            {"field": "policy", "label": "Policy", "type": "text"},
            {"field": "annualized_net_excess_return", "label": "Ann. net excess", "format": "percent"},
            {"field": "annualized_realized_volatility", "label": "Volatility", "format": "percent"},
            {"field": "net_sharpe", "label": "Sharpe", "format": "number"},
            {"field": "maximum_drawdown", "label": "Max drawdown", "format": "percent"},
            {"field": "annualized_total_cost", "label": "Ann. cost", "format": "percent"},
            {"field": "actions", "label": "Warnings", "format": "number"},
            {"field": "refill_fraction", "label": "Refill share", "format": "percent"},
            {"field": "paired_hac_t_vs_v6_7", "label": "Paired HAC t", "format": "number"},
        ],
    }

    generated = dt.datetime.now(dt.timezone(dt.timedelta(hours=3))).isoformat(timespec="seconds")
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Should Daily Exits Be Re-Optimized?",
            "description": "Technical comparison of leaving V6.7 exit proceeds in cash versus causally refilling surviving monthly holdings.",
            "generatedAt": generated,
            "sources": sources,
            "charts": charts,
            "tables": [table],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Should Daily Exits Be Re-Optimized?"},
                {"id": "summary", "type": "markdown", "sourceId": "historical_performance", "body": "## Technical summary\n\n- **Re-optimization increased return but not Sharpe.** V6.8 reached 25.67% annualized net excess return versus 25.18% for the V6.7 cash overlay, while volatility rose from 11.81% to 12.14% and Sharpe slipped from 2.132 to 2.114.\n- **The incremental gain is not statistically reliable.** The paired annualized return difference was +0.49 percentage points with HAC t = 0.64 and positive performance in 50% of the 20 historical months.\n- **The mechanism is operationally valid but does not qualify for promotion.** It restored 41.7% of released gross without adding stocks, re-entering exits, reversing sides, or enlarging warned names. The frozen monthly strategy remains primary; V6.7 and V6.8 remain shadow diagnostics."},
                {"id": "wealth_text", "type": "markdown", "sourceId": "daily_wealth", "body": "## Reinvestment modestly raised terminal wealth\n\nThe V6.8 path usually remains close to V6.7 because both use the same warnings and execution dates. Reinvestment retains more market exposure after exits, which slightly lifts the ending path but also restores some of the volatility that V6.7 removed by holding cash."},
                {"id": "wealth_chart", "type": "chart", "chartId": "wealth_path", "layout": "full"},
                {"id": "stability_text", "type": "markdown", "sourceId": "period_stability", "body": "## Strong validation did not translate into a higher historical Sharpe\n\nV6.8 improved 2023 validation Sharpe from −0.232 to 0.560. On January 2024–August 2025, however, the cash overlay retained the slightly higher Sharpe. This is why the design cannot be promoted solely because validation looked substantially better."},
                {"id": "stability_chart", "type": "chart", "chartId": "period_sharpe", "layout": "full"},
                {"id": "scope", "type": "markdown", "body": "## Scope, data, and metric definitions\n\nThe intervention unit is an executed same-close batch of V6.7 warnings. Validation covers January–December 2023; historical evidence covers January 2024–August 2025. Net excess return includes trading costs, stock borrow, and 6.25% annual financing on gross exposure above 1× NAV. Sharpe uses equal-weighted monthly net excess returns. Refill share is total gross added to survivors divided by gross removed by the same warning batches."},
                {"id": "method", "type": "markdown", "body": "## The refill solver changes allocation, not prediction\n\nV6.8 reuses the exact saved V6.7 warning dates. After a trim or exit executes, it maximizes the remaining-horizon frozen monthly expected stock excess return net of trading, borrow, and financing costs. Expected monthly returns are prorated by remaining sessions. Only existing, never-warned holdings may be enlarged in their original direction; warned names are frozen at their post-action exposure. Added gross cannot exceed contemporaneously removed gross, and the translated 1.80× gross, 3% position, factor, industry, net, liquidity, and forecast-risk limits remain in force. If no admissible refill has positive expected value, cash is allowed."},
                {"id": "refill_text", "type": "markdown", "sourceId": "refill_mechanics", "body": "## Constraints prevented mechanical full reinvestment\n\nThe optimizer restored 1.134× cumulative gross out of 2.717× removed across the historical replay. Thirty-three of 157 batches had no admissible refill capacity. This partial response is intentional: a full pro-rata refill could undo the warning, breach the position cap, or worsen factor and industry concentration."},
                {"id": "refill_chart", "type": "chart", "chartId": "refill_by_month", "layout": "full"},
                {"id": "uncertainty", "type": "markdown", "sourceId": "paired_monthly_difference", "body": "## The incremental result is too small and inconsistent to justify trading\n\nV6.8 beat V6.7 in exactly half of the historical months. The +0.49 percentage-point annualized difference is economically modest and its HAC t-statistic of 0.64 is far below the frozen 1.96 gate. The evidence supports the mechanism's feasibility, not a claim that reinvestment adds a dependable edge."},
                {"id": "delta_chart", "type": "chart", "chartId": "monthly_delta", "layout": "full"},
                {"id": "audit", "type": "markdown", "sourceId": "independent_audit", "body": "## Accounting and constraint checks passed\n\nAn independent audit reproduced all headline metrics exactly and found zero within-month NAV continuity error. All 450 validation and 370 historical warning actions matched V6.7. Every warned name remained frozen, every survivor change preserved its side, and no event refilled more gross than it removed."},
                {"id": "exact", "type": "markdown", "body": "## Exact historical results"},
                {"id": "performance_table_block", "type": "table", "tableId": "performance_table", "layout": "full"},
                {"id": "limitations", "type": "markdown", "body": "## Limitations and evidence boundary\n\nThis is a user-authorized post-diagnostic study on outcomes that were already observed when V6.8 was opened. The 20-month historical window is short, and the same monthly expected-return estimates are reused intramonth rather than replaced by a separately calibrated residual-horizon forecast. These limitations do not invalidate the replay, but they prevent production promotion from this evidence."},
                {"id": "next", "type": "markdown", "body": "## Recommended next steps\n\n1. Keep the frozen monthly 1.80× portfolio as the paper-trading primary.\n2. Log both V6.7 cash actions and V6.8 proposed refills in shadow mode.\n3. Compare their realized incremental returns, volatility, and fill costs prospectively from September 2026.\n4. Promote re-optimization only if it improves both return and Sharpe with a reliable paired advantage."},
                {"id": "questions", "type": "markdown", "body": "## Further questions\n\n- Does refill performance differ between long exits and short covers?\n- Are prospective benefits concentrated in particular market or volatility regimes?\n- Do paper fills and live borrow availability reduce the already modest incremental return?"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "performance": _rows(performance),
                "period_stability": _rows(stability),
                "wealth_path": _rows(wealth),
                "monthly_delta": _rows(monthly_delta),
                "refill_by_month": _rows(refill_long),
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
                {"section": "Stability", "question": "Did validation Sharpe generalize?", "family": "comparison", "chart": "period_sharpe", "palette": "hard two-root cap", "source": "frozen period summaries"},
                {"section": "Mechanism", "question": "How much released gross was restored?", "family": "comparison", "chart": "refill_by_month", "palette": "hard two-root cap", "source": "executed V6.8 events"},
                {"section": "Uncertainty", "question": "Was incremental return broad by month?", "family": "comparison", "chart": "monthly_delta", "palette": "single-root preferred", "source": "paired monthly replays"},
            ],
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    arguments = parser.parse_args()
    build(arguments.result_root.resolve())


if __name__ == "__main__":
    main()
