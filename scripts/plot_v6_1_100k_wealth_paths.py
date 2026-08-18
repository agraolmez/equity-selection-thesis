#!/usr/bin/env python3
"""Plot $100,000 wealth paths for the frozen V6.1 2.05x research profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_v6_1_return_scaling_overlay as replay  # noqa: E402
from scripts import run_v6_1_maximum_return_scaling_frontier as frontier  # noqa: E402


INK = "#20252b"
GRID = "#d9dee3"
BLUE = "#2b6cb0"
NEUTRAL = "#7b8794"
GOLD = "#c58b19"


def _money(value: float, _position: int | None = None) -> str:
    return f"${value / 1_000:.0f}k"


def _phase_frame(
    *,
    phase: str,
    monthly: pd.DataFrame,
    benchmarks: pd.DataFrame,
    initial_nav: float,
) -> pd.DataFrame:
    strategy = monthly[["holding_month", "net_return_primary"]].copy()
    pivot = benchmarks.pivot(
        index="holding_month", columns="benchmark_id", values="total_return"
    ).reset_index()
    frame = strategy.merge(pivot, on="holding_month", how="left", validate="one_to_one")
    frame["date"] = pd.PeriodIndex(frame["holding_month"], freq="M").to_timestamp("M")
    frame["strategy_value"] = initial_nav * np.cumprod(
        1.0 + frame["net_return_primary"].to_numpy(float)
    )
    frame["spy_value"] = initial_nav * np.cumprod(
        1.0 + frame["spy_total_return"].to_numpy(float)
    )
    frame["qqq_value"] = initial_nav * np.cumprod(
        1.0 + frame["qqq_total_return"].to_numpy(float)
    )
    first_period = pd.Period(str(frame.iloc[0]["holding_month"]), freq="M") - 1
    start = pd.DataFrame(
        [
            {
                "holding_month": str(first_period),
                "net_return_primary": 0.0,
                "spy_total_return": 0.0,
                "qqq_total_return": 0.0,
                "date": first_period.to_timestamp("M"),
                "strategy_value": initial_nav,
                "spy_value": initial_nav,
                "qqq_value": initial_nav,
            }
        ]
    )
    result = pd.concat([start, frame], ignore_index=True)
    result.insert(0, "phase", phase)
    return result


def _cagr(final_value: float, initial_nav: float, months: int) -> float:
    return float((final_value / initial_nav) ** (12.0 / months) - 1.0)


def run(config_path: Path, output_dir: Path, initial_nav: float) -> tuple[Path, Path, Path]:
    config, identity, source_root = frontier._verify(config_path)
    source_nav = float(config["nav_dollars"])
    if initial_nav <= 0.0 or source_nav <= 0.0:
        raise ValueError("NAV must be positive")
    replay_config = frontier._replay_config(config, source_root)
    risk = replay._load_risk(replay_config)

    # The replay engine is frozen at the source study's $100m NAV. Scaling ADV
    # by source_nav / target_nav is algebraically equivalent to evaluating its
    # square-root impact and participation equations at the requested NAV.
    risk = risk.copy()
    risk["adv20_usd"] *= source_nav / initial_nav

    profile = frontier._profile(2.05)
    frames: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, float | int]] = {}
    for phase in ("validation", "historical_test"):
        monthly, summary, _ = replay._run_phase(
            phase, replay_config, risk, profiles=[profile]
        )
        evidence = config["evidence"]
        start = str(evidence[f"{phase}_start"])
        end = str(evidence[f"{phase}_end"])
        benchmarks = frontier._benchmark_frame(config, start, end)
        frame = _phase_frame(
            phase=phase,
            monthly=monthly,
            benchmarks=benchmarks,
            initial_nav=initial_nav,
        )
        frames.append(frame)
        row = summary.iloc[0]
        months = int(row["months"])
        summaries[phase] = {
            "months": months,
            "strategy_final_value": float(frame.iloc[-1]["strategy_value"]),
            "spy_final_value": float(frame.iloc[-1]["spy_value"]),
            "qqq_final_value": float(frame.iloc[-1]["qqq_value"]),
            "strategy_cagr": _cagr(
                float(frame.iloc[-1]["strategy_value"]), initial_nav, months
            ),
            "spy_cagr": _cagr(float(frame.iloc[-1]["spy_value"]), initial_nav, months),
            "qqq_cagr": _cagr(float(frame.iloc[-1]["qqq_value"]), initial_nav, months),
            "net_sharpe": float(row["net_sharpe"]),
            "maximum_drawdown": float(row["maximum_drawdown"]),
            "annualized_primary_cost": float(row["annualized_primary_cost"]),
            "maximum_gross": float(row["maximum_gross"]),
            "maximum_one_way_adv_participation": float(
                row["maximum_one_way_adv_participation"]
            ),
            "maximum_absolute_position_dollars_at_start": float(
                row["maximum_absolute_position"] * initial_nav
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "wealth_paths_100k.csv"
    json_path = output_dir / "wealth_paths_100k_summary.json"
    png_path = output_dir / "wealth_paths_100k.png"
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "schema": "research_v6_1_100k_wealth_visualization_v1",
                "status": "COMPLETE",
                "profile": profile,
                "initial_nav": initial_nav,
                "costs_recomputed_at_target_nav": True,
                "frozen_identity": identity,
                "interpretation": (
                    "The 2.05x research profile is uncapped and reached more than 2x "
                    "overnight gross exposure; it is not the still-unrun 2.0x-capped "
                    "Alpaca paper implementation."
                ),
                "phases": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

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
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.5), facecolor="white")
    labels = {
        "strategy_value": "Frozen 2.05× portfolio",
        "qqq_value": "QQQ",
        "spy_value": "SPY",
    }
    styles = {
        "strategy_value": (BLUE, "-", 2.6),
        "qqq_value": (INK, "--", 1.7),
        "spy_value": (NEUTRAL, ":", 1.9),
    }
    phase_titles = {
        "validation": "Validation wealth path",
        "historical_test": "Historical-test wealth path",
    }
    phase_subtitles = {
        "validation": "February 2022–December 2023",
        "historical_test": "January 2024–August 2025",
    }
    for ax, phase in zip(axes, ("validation", "historical_test")):
        frame = combined.loc[combined["phase"].eq(phase)].copy()
        for column in ("strategy_value", "qqq_value", "spy_value"):
            color, linestyle, width = styles[column]
            ax.plot(
                frame["date"],
                frame[column],
                color=color,
                linestyle=linestyle,
                linewidth=width,
                label=labels[column],
            )
            final = float(frame.iloc[-1][column])
            ax.scatter(frame.iloc[-1]["date"], final, s=24, color=color, zorder=5)
            y_offset = {
                "strategy_value": 8,
                "qqq_value": 10 if phase == "validation" else 7,
                "spy_value": -16 if phase == "validation" else -13,
            }[column]
            ax.annotate(
                f"{labels[column]}  {_money(final)}",
                (frame.iloc[-1]["date"], final),
                xytext=(-4, y_offset),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=8.4,
                fontweight="bold" if column == "strategy_value" else "normal",
            )
        stats = summaries[phase]
        ax.set_title(phase_titles[phase], loc="left", color=INK, fontweight="bold", pad=26)
        ax.text(
            0.0,
            1.01,
            phase_subtitles[phase]
            + f"  |  strategy CAGR {stats['strategy_cagr']:.1%}, Sharpe {stats['net_sharpe']:.2f}",
            transform=ax.transAxes,
            color=NEUTRAL,
            fontsize=8.6,
            va="bottom",
        )
        ax.yaxis.set_major_formatter(FuncFormatter(_money))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(INK)
        low = min(frame[["strategy_value", "spy_value", "qqq_value"]].min())
        high = max(frame[["strategy_value", "spy_value", "qqq_value"]].max())
        padding = 0.10 * (high - low)
        ax.set_ylim(low - padding, high + 1.35 * padding)
        ax.set_ylabel("Portfolio value")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.965, 0.955),
        ncol=3,
        frameon=False,
        fontsize=8.8,
    )
    fig.suptitle(
        "Growth of \\$100,000: frozen V6.1 2.05× return portfolio",
        x=0.065,
        y=0.965,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.065,
        0.045,
        "Each panel independently resets to \\$100,000. Trading impact is recomputed for \\$100,000; "
        "returns are net of modeled spread, impact, financing, borrow, and commission costs.",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=NEUTRAL,
    )
    fig.text(
        0.065,
        0.020,
        "Important: this is the uncapped 2.05× research profile (validation/test max gross 2.76×/2.62×), not a "
        "2.0×-capped Alpaca paper portfolio. Historical evidence is exploratory, not prospective confirmation.",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=GOLD,
    )
    fig.subplots_adjust(left=0.08, right=0.96, top=0.82, bottom=0.17, wspace=0.24)
    fig.savefig(png_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return png_path, csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research_v6_1/maximum_return_scaling_frontier.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--initial-nav", type=float, default=100_000.0)
    args = parser.parse_args()
    paths = run(args.config.resolve(), args.output_dir.resolve(), args.initial_nav)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
