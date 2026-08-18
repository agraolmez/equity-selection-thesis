#!/usr/bin/env python3
"""Build the portable Sharpe-regime diagnostic for the V6.1 2.05x profile."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_v6_1_return_scaling_overlay as replay  # noqa: E402
from scripts import run_v6_1_maximum_return_scaling_frontier as frontier  # noqa: E402


def _clean(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(key): _clean(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _exclusion_rows(window: str, values: np.ndarray) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    order = np.argsort(values)
    for remove in (0, 1, 3, 5):
        keep = np.ones(len(values), dtype=bool)
        if remove:
            keep[order[-remove:]] = False
        retained = values[keep]
        result.append(
            {
                "window": window,
                "removed_best_months": remove,
                "remaining_months": int(len(retained)),
                "annualized_mean_net_excess": float(retained.mean() * 12.0),
                "annualized_volatility": float(retained.std(ddof=1) * np.sqrt(12.0)),
                "sharpe": float(
                    retained.mean() / retained.std(ddof=1) * np.sqrt(12.0)
                ),
            }
        )
    return result


def build(config_path: Path, output_dir: Path, target_nav: float) -> Path:
    config, identity, source_root = frontier._verify(config_path)
    replay_config = frontier._replay_config(config, source_root)
    risk = replay._load_risk(replay_config).copy()
    risk["adv20_usd"] *= float(config["nav_dollars"]) / target_nav
    profile = frontier._profile(2.05)

    monthly_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    for phase, window in (
        ("validation", "Validation"),
        ("historical_test", "Historical test"),
    ):
        monthly, summary, _ = replay._run_phase(
            phase, replay_config, risk, profiles=[profile]
        )
        monthly = monthly.copy()
        monthly["window"] = window
        monthly["month"] = monthly["holding_month"].astype(str)
        monthly["net_excess_return"] = monthly["net_excess_return_primary"]
        monthly["gross_excess_return_display"] = monthly["gross_excess_return"]
        monthly["primary_cost"] = monthly["total_cost_primary"]
        monthly_frames.append(
            monthly[
                [
                    "window",
                    "month",
                    "net_excess_return",
                    "gross_excess_return_display",
                    "primary_cost",
                    "net_return_primary",
                ]
            ]
        )
        row = summary.iloc[0]
        summary_rows.append(
            {
                "window": window,
                "months": int(row["months"]),
                "annualized_net_excess_return": float(
                    row["annualized_net_excess_return"]
                ),
                "annualized_volatility": float(row["annualized_realized_volatility"]),
                "net_sharpe": float(row["net_sharpe"]),
                "annualized_primary_cost": float(row["annualized_primary_cost"]),
                "positive_month_fraction": float(row["positive_month_fraction"]),
                "maximum_drawdown": float(row["maximum_drawdown"]),
                "cumulative_net_total_return": float(
                    row["cumulative_net_total_return"]
                ),
                "mean_monthly_net_excess": float(
                    monthly["net_excess_return_primary"].mean()
                ),
                "monthly_net_excess_std": float(
                    monthly["net_excess_return_primary"].std(ddof=1)
                ),
            }
        )
        concentration_rows.extend(
            _exclusion_rows(
                window, monthly["net_excess_return_primary"].to_numpy(float)
            )
        )

    monthly_data = pd.concat(monthly_frames, ignore_index=True)
    summary_data = pd.DataFrame(summary_rows)
    concentration_data = pd.DataFrame(concentration_rows)
    pooled = monthly_data["net_excess_return"].to_numpy(float)
    pooled_sharpe = float(pooled.mean() / pooled.std(ddof=1) * np.sqrt(12.0))

    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = output_dir / "sharpe_diagnostic_monthly.csv"
    summary_path = output_dir / "sharpe_diagnostic_summary.csv"
    concentration_path = output_dir / "sharpe_diagnostic_concentration.csv"
    monthly_data.to_csv(monthly_path, index=False)
    summary_data.to_csv(summary_path, index=False)
    concentration_data.to_csv(concentration_path, index=False)

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    monthly_source = {
        "id": "monthly_replay",
        "label": "$100,000 frozen-portfolio monthly replay",
        "path": str(monthly_path.relative_to(ROOT)),
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": (
                "SELECT window, month, net_excess_return, "
                "gross_excess_return_display, primary_cost, net_return_primary "
                "FROM read_csv_auto('"
                + str(monthly_path.relative_to(ROOT))
                + "') ORDER BY month"
            ),
            "description": "Reviewed monthly $100,000 replay rows used in the diagnostic chart.",
            "executed_at": generated_at,
            "tables_used": [str(monthly_path.relative_to(ROOT))],
            "filters": [
                "Frozen 2.05x profile",
                "Validation 2022-02 through 2023-12",
                "Historical test 2024-01 through 2025-08",
            ],
        },
    }
    summary_source = {
        "id": "sharpe_summary",
        "label": "Validation and historical-test Sharpe decomposition",
        "path": str(summary_path.relative_to(ROOT)),
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": (
                "SELECT * FROM read_csv_auto('"
                + str(summary_path.relative_to(ROOT))
                + "') ORDER BY net_sharpe"
            ),
            "description": "Window-level return, volatility, cost, and Sharpe decomposition.",
            "executed_at": generated_at,
            "tables_used": [str(summary_path.relative_to(ROOT))],
            "metric_definitions": {
                "net_sharpe": (
                    "Mean monthly net excess return divided by its sample standard "
                    "deviation, multiplied by square root of 12."
                )
            },
        },
    }
    concentration_source = {
        "id": "return_concentration",
        "label": "Best-month exclusion sensitivity",
        "path": str(concentration_path.relative_to(ROOT)),
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": (
                "SELECT * FROM read_csv_auto('"
                + str(concentration_path.relative_to(ROOT))
                + "') ORDER BY window, remaining_months DESC"
            ),
            "description": "Descriptive Sharpe sensitivity after removing the strongest months.",
            "executed_at": generated_at,
            "tables_used": [str(concentration_path.relative_to(ROOT))],
        },
    }
    selection_source = {
        "id": "selection_contract",
        "label": "Maximum-return scaling selection contract",
        "path": str(config_path.relative_to(ROOT)),
    }
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Why validation Sharpe was lower",
            "description": (
                "Technical diagnostic of the frozen V6.1 2.05x portfolio at a "
                "$100,000 starting NAV."
            ),
            "generatedAt": generated_at,
            "cards": [],
            "charts": [
                {
                    "id": "monthly_net_excess",
                    "title": "Monthly net excess returns by evaluation window",
                    "subtitle": (
                        "Validation had a lower average payoff relative to its monthly "
                        "dispersion; the historical-test distribution shifted upward."
                    ),
                    "type": "line",
                    "dataset": "monthly",
                    "sourceId": "monthly_replay",
                    "encodings": {
                        "x": {
                            "field": "month",
                            "type": "temporal",
                            "label": "Holding month",
                        },
                        "y": {
                            "field": "net_excess_return",
                            "type": "quantitative",
                            "label": "Monthly net excess return",
                            "format": "percent",
                        },
                        "color": {
                            "field": "window",
                            "type": "nominal",
                            "label": "Evaluation window",
                        },
                    },
                    "yAxisTitle": "Monthly net excess return",
                    "valueFormat": "percent",
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "window_summary",
                    "title": "Sharpe decomposition by window",
                    "subtitle": (
                        "Annualized net excess return, volatility, costs, and downside "
                        "for the same frozen portfolio."
                    ),
                    "dataset": "summary",
                    "sourceId": "sharpe_summary",
                    "defaultSort": {"field": "net_sharpe", "direction": "asc"},
                    "columns": [
                        {"field": "window", "label": "Window", "type": "text"},
                        {
                            "field": "months",
                            "label": "Months",
                            "format": "number",
                        },
                        {
                            "field": "annualized_net_excess_return",
                            "label": "Annualized net excess",
                            "format": "percent",
                        },
                        {
                            "field": "annualized_volatility",
                            "label": "Annualized volatility",
                            "format": "percent",
                        },
                        {
                            "field": "net_sharpe",
                            "label": "Net Sharpe",
                            "format": "number",
                        },
                        {
                            "field": "annualized_primary_cost",
                            "label": "Annualized modeled cost",
                            "format": "percent",
                        },
                        {
                            "field": "positive_month_fraction",
                            "label": "Positive months",
                            "format": "percent",
                        },
                        {
                            "field": "maximum_drawdown",
                            "label": "Maximum drawdown",
                            "format": "percent",
                        },
                    ],
                },
                {
                    "id": "concentration_sensitivity",
                    "title": "Sensitivity to the strongest months",
                    "subtitle": (
                        "Descriptive Sharpe after removing each window's best one, "
                        "three, or five monthly observations."
                    ),
                    "dataset": "concentration",
                    "sourceId": "return_concentration",
                    "defaultSort": {"field": "remaining_months", "direction": "desc"},
                    "columns": [
                        {"field": "window", "label": "Window", "type": "text"},
                        {
                            "field": "removed_best_months",
                            "label": "Best months removed",
                            "format": "number",
                        },
                        {
                            "field": "remaining_months",
                            "label": "Remaining months",
                            "format": "number",
                        },
                        {
                            "field": "annualized_mean_net_excess",
                            "label": "Annualized mean net excess",
                            "format": "percent",
                        },
                        {
                            "field": "annualized_volatility",
                            "label": "Annualized volatility",
                            "format": "percent",
                        },
                        {"field": "sharpe", "label": "Sharpe", "format": "number"},
                    ],
                },
            ],
            "sources": [
                monthly_source,
                summary_source,
                concentration_source,
                selection_source,
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# Why validation Sharpe was lower",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "sharpe_summary",
                    "body": (
                        "## The mean payoff changed far more than risk or costs\n\n"
                        "The validation Sharpe of **0.85** is correctly calculated. "
                        "Validation produced **8.68% annualized net excess return** "
                        "against **10.16% volatility**. In the historical test, net "
                        "excess return rose to **30.24%** while volatility increased only "
                        "to **12.96%**, producing a **2.33 Sharpe**. Modeled annual costs "
                        "were approximately **2.0% in both windows**, so costs do not "
                        "explain the gap."
                    ),
                },
                {
                    "id": "selection_rule",
                    "type": "markdown",
                    "sourceId": "selection_contract",
                    "body": (
                        "## The portfolio was selected for return, not maximum Sharpe\n\n"
                        "The balanced high-return tier maximized validation net total-"
                        "return CAGR subject to a **0.75 minimum validation Sharpe**. The "
                        "2.05× profile passed that floor at 0.85; it was never expected to "
                        "be the highest-Sharpe validation portfolio. Its moderate "
                        "validation Sharpe is therefore consistent with the frozen "
                        "selection rule, not evidence of a calculation error."
                    ),
                },
                {
                    "id": "key_finding",
                    "type": "markdown",
                    "sourceId": "monthly_replay",
                    "body": (
                        "## Validation returns were positive but concentrated\n\n"
                        "Validation earned money, but its average monthly excess return "
                        "was only **0.72%** and one month—May 2022—contributed **11.21%**. "
                        "July 2023 then lost **6.51%** in net excess terms. The chart shows "
                        "why a respectable terminal wealth gain can coexist with a "
                        "moderate Sharpe: the path was uneven and the average payoff was "
                        "small relative to month-to-month dispersion."
                    ),
                },
                {
                    "id": "monthly_chart",
                    "type": "chart",
                    "chartId": "monthly_net_excess",
                    "layout": "full",
                },
                {
                    "id": "metric_definition",
                    "type": "markdown",
                    "body": (
                        "## What the reported Sharpe measures\n\n"
                        "Sharpe is the mean monthly **net excess return over cash**, "
                        "divided by its monthly standard deviation and annualized by the "
                        "square root of 12. It is not calculated from terminal wealth or "
                        "total return. Consequently, the validation path can grow from "
                        "$100k to roughly $125k while still reporting a 0.85 Sharpe."
                    ),
                },
                {
                    "id": "cost_finding",
                    "type": "markdown",
                    "sourceId": "sharpe_summary",
                    "body": (
                        "## Similar costs rule out implementation drag as the driver\n\n"
                        "The two windows used the same frozen selectors, weights, cost "
                        "logic, and $100k NAV. Annualized modeled costs were **1.96%** in "
                        "validation and **2.02%** in the historical test. The difference "
                        "therefore comes primarily from realized stock-selection payoff "
                        "and regime, not from a preprocessing or accounting discrepancy."
                    ),
                },
                {
                    "id": "summary_table",
                    "type": "table",
                    "tableId": "window_summary",
                    "layout": "full",
                },
                {
                    "id": "concentration_finding",
                    "type": "markdown",
                    "sourceId": "return_concentration",
                    "body": (
                        "## The test was stronger and broader, but still regime-sensitive\n\n"
                        "Removing validation's best month reduces its Sharpe from **0.85 "
                        "to 0.45**; removing its best three makes the remaining mean excess "
                        "return slightly negative. The historical test remains at **2.25** "
                        "after removing its best month and **1.30** after removing its best "
                        "five. This supports a real window difference, while also warning "
                        "that the 2.33 test Sharpe should not be treated as a stable "
                        "production expectation."
                    ),
                },
                {
                    "id": "concentration_table",
                    "type": "table",
                    "tableId": "concentration_sensitivity",
                    "layout": "full",
                },
                {
                    "id": "scope_method",
                    "type": "markdown",
                    "body": (
                        "## Scope and methodology\n\n"
                        "The comparison uses the frozen 2.05× portfolio with a $100k "
                        "starting NAV. Validation covers February 2022–December 2023; the "
                        "historical test covers January 2024–August 2025. Spread, "
                        "square-root impact, financing, borrow, commission, NAV drift, and "
                        "cash returns are recomputed monthly. Features, selectors, alpha "
                        "gate, stock membership, and relative weights are unchanged."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## What this comparison does not prove\n\n"
                        "Both windows are short, and the historical test is development "
                        "evidence rather than untouched prospective confirmation. The "
                        "portfolio also reaches 2.76× gross in validation and 2.62× in "
                        "the test, so this is not yet the 2.0×-capped Alpaca paper policy. "
                        "The descriptive pooled 43-month Sharpe is "
                        f"**{pooled_sharpe:.2f}**, but pooling does not convert these "
                        "post-diagnostic windows into independent confirmation."
                    ),
                },
                {
                    "id": "next_step",
                    "type": "markdown",
                    "body": (
                        "## Recommended next step\n\n"
                        "Freeze and simulate the actual Alpaca policy with a hard 2.0× "
                        "overnight gross cap, easy-to-borrow checks, and the same monthly "
                        "signals. Then monitor rolling 12-month Sharpe and selector IC in "
                        "paper trading rather than assuming the 2.33 historical-test "
                        "Sharpe will persist."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further question\n\n"
                        "How much return and Sharpe are lost when the uncapped research "
                        "portfolio is translated into Alpaca's 2.0× overnight buying-power "
                        "limit? That result remains unrun."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "monthly": _rows(monthly_data),
                "summary": _rows(summary_data),
                "concentration": _rows(concentration_data),
            },
        },
        "sources": [
            monthly_source,
            summary_source,
            concentration_source,
            selection_source,
        ],
        "package_info": {
            "root": "research_v6_1_sharpe_diagnostic",
            "manifestPath": "artifact.json",
            "snapshotPath": "artifact.json",
        },
    }
    artifact_path = output_dir / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research_v6_1/maximum_return_scaling_frontier.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-nav", type=float, default=100_000.0)
    args = parser.parse_args()
    artifact_path = build(
        args.config.resolve(), args.output_dir.resolve(), args.target_nav
    )
    print(artifact_path)


if __name__ == "__main__":
    main()
