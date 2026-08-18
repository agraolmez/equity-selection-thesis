#!/usr/bin/env python3
"""Render the frozen V6.1 maximum-return scaling frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


INK = "#20252b"
GRID = "#d9dee3"
BLUE = "#2b6cb0"
BLUE_DARK = "#164e7a"
BLUE_LIGHT = "#a8c7e6"
GOLD = "#c58b19"
NEUTRAL = "#7b8794"
OPEN = "#eef3f7"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def render(run_dir: Path, output: Path) -> None:
    validation = pd.read_csv(run_dir / "validation_summary.csv").sort_values("multiplier")
    test = pd.read_csv(run_dir / "historical_test_summary.csv").set_index("profile")
    result = json.loads((run_dir / "result.json").read_text())

    tier_order = [
        ("realistic_100m", "Original mandate"),
        ("expanded_institutional_100m", "Expanded mandate"),
        ("balanced_high_return", "Balanced high return"),
        ("stress_positive_ceiling", "Stress-positive ceiling"),
        ("absolute_return_ceiling", "Absolute ceiling"),
    ]
    selected = []
    for tier, label in tier_order:
        profile = result["tier_winners"][tier]
        row = test.loc[profile]
        selected.append(
            {
                "label": label,
                "profile": profile,
                "cagr": float(row["net_total_return_cagr"]),
                "sharpe": float(row["net_sharpe"]),
                "adv": float(row["maximum_one_way_adv_participation"]),
                "gross": float(row["maximum_gross"]),
            }
        )

    spy_val = float(result["validation_benchmarks"]["spy_total_return"]["cagr"])
    qqq_val = float(result["validation_benchmarks"]["qqq_total_return"]["cagr"])
    spy_test = float(result["historical_benchmarks"]["spy_total_return"]["cagr"])
    qqq_test = float(result["historical_benchmarks"]["qqq_total_return"]["cagr"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    fig = plt.figure(figsize=(12.0, 7.1), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.33)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])

    ax1.plot(
        validation["multiplier"],
        validation["net_total_return_cagr"],
        color=BLUE,
        linewidth=2.2,
        label="Frozen portfolio scaling frontier",
    )
    ax1.axhline(spy_val, color=NEUTRAL, linewidth=1.2, linestyle="--")
    ax1.axhline(qqq_val, color=INK, linewidth=1.2, linestyle=":")
    ax1.text(19.8, spy_val + 0.004, f"SPY {_pct(spy_val)}", ha="right", color=NEUTRAL)
    ax1.text(19.8, qqq_val + 0.004, f"QQQ {_pct(qqq_val)}", ha="right", color=INK)

    tier_colors = [BLUE_DARK, BLUE, GOLD, BLUE_LIGHT, INK]
    tier_markers = ["o", "s", "D", "^", "X"]
    for item, color, marker in zip(selected, tier_colors, tier_markers):
        scale = float(item["profile"].split("_")[1])
        val_row = validation.loc[validation["profile"] == item["profile"]].iloc[0]
        val_cagr = float(val_row["net_total_return_cagr"])
        ax1.scatter(scale, val_cagr, s=50, marker=marker, color=color, zorder=5)
        if item["label"] in {"Original mandate", "Balanced high return", "Absolute ceiling"}:
            offset = (6, 8) if item["label"] != "Absolute ceiling" else (7, -18)
            ax1.annotate(
                f"{scale:.2f}x\n{_pct(val_cagr)}",
                (scale, val_cagr),
                xytext=offset,
                textcoords="offset points",
                color=INK,
                fontsize=8.2,
            )

    ax1.set_title(
        "Validation return frontier", loc="left", color=INK, fontweight="bold", pad=29
    )
    ax1.text(
        0.0,
        1.012,
        "Net total-return CAGR by constant scaling of frozen weights, Feb 2022–Dec 2023",
        transform=ax1.transAxes,
        color=NEUTRAL,
        fontsize=8.5,
        va="bottom",
    )
    ax1.set_xlabel("Portfolio scale relative to the frozen winner")
    ax1.set_ylabel("Net total-return CAGR")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax1.set_xlim(0.25, 20.25)
    ax1.grid(axis="y", color=GRID, linewidth=0.7)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines[["left", "bottom"]].set_color(INK)

    labels = ["SPY", "QQQ"] + [x["label"] for x in selected]
    values = [spy_test, qqq_test] + [x["cagr"] for x in selected]
    colors = [NEUTRAL, INK, BLUE_DARK, BLUE, GOLD, BLUE_LIGHT, OPEN]
    edges = [NEUTRAL, INK, BLUE_DARK, BLUE_DARK, GOLD, BLUE_DARK, INK]
    hatches = ["", "", "", "", "", "///", "xxx"]
    y = list(range(len(labels)))
    bars = ax2.barh(y, values, color=colors, edgecolor=edges, linewidth=1.0)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for idx, (bar, value) in enumerate(zip(bars, values)):
        ax2.text(
            value + 0.018,
            bar.get_y() + bar.get_height() / 2,
            _pct(value),
            va="center",
            color=INK,
            fontsize=8.5,
            fontweight="bold" if idx in {3, 4} else "normal",
        )

    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()
    ax2.set_xlim(0.0, max(values) * 1.19)
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.set_xlabel("Net total-return CAGR")
    ax2.set_title(
        "Historical selected-tier comparison",
        loc="left",
        color=INK,
        fontweight="bold",
        pad=29,
    )
    ax2.text(
        0.0,
        1.012,
        "January 2024–August 2025; hatched profiles are research ceilings, not deployable mandates",
        transform=ax2.transAxes,
        color=NEUTRAL,
        fontsize=8.5,
        va="bottom",
    )
    ax2.grid(axis="x", color=GRID, linewidth=0.7)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color(INK)

    details = (
        "Selected profile details (historical):  "
        + "  |  ".join(
            f"{x['profile'].replace('constant_', '')}x: Sharpe {x['sharpe']:.2f}, "
            f"max ADV {100*x['adv']:.1f}%, max gross {x['gross']:.2f}x"
            for x in selected
        )
    )
    fig.suptitle(
        "Research V6.1 maximum-return scaling frontier",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(0.06, 0.035, details, ha="left", va="bottom", fontsize=7.3, color=NEUTRAL)
    fig.text(
        0.06,
        0.012,
        "Source: authenticated V6.1 frozen portfolio replay; net of modeled trading, impact, financing, and borrow costs. "
        "Historical outcomes are exploratory development evidence, not untouched confirmation.",
        ha="left",
        va="bottom",
        fontsize=7.3,
        color=NEUTRAL,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.run_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
