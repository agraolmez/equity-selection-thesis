"""Build source-backed figures for the Bocconi final paper.

The script is read-only with respect to research artifacts. It writes only
presentation files under ``thesis/figures`` and uses frozen selector,
portfolio, and benchmark authorities.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "thesis" / "figures"
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
MAX_RETURN = (
    ROOT
    / "artifacts/research_v6_1/maximum_return_scaling_frontier/"
    "b50c39dfbd62-ca8bc83ff816-e5468ec50cbf"
)
BENCHMARKS = Path(
    "/Users/agraemirolmez/Desktop/Thesis/processed/phase_06_final_test/"
    "stage13-final-inputs-v2/primary_benchmarks.parquet"
)

BLUE = "#2563EB"
NAVY = "#0F172A"
GOLD = "#D4A72C"
ORANGE = "#EA580C"
GRAY = "#64748B"
LIGHT = "#E2E8F0"
PALE_BLUE = "#DBEAFE"
PALE_GOLD = "#FEF3C7"
PALE_GRAY = "#F1F5F9"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "font.family": "Arial",
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": LIGHT,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _portfolio_and_spy() -> pd.DataFrame:
    monthly = pd.read_parquet(FRONTIER / "historical_test_monthly.parquet")
    monthly = monthly.loc[monthly["profile"].eq("position_1p5")].copy()
    monthly = monthly.sort_values("holding_month", kind="mergesort")
    benchmark = pd.read_parquet(BENCHMARKS)
    spy = benchmark.loc[
        benchmark["benchmark_id"].astype(str).eq("spy_total_return"),
        ["holding_month", "total_return"],
    ].rename(columns={"total_return": "spy_total_return"})
    data = monthly.merge(spy, on="holding_month", how="left", validate="one_to_one")
    if data["spy_total_return"].isna().any():
        raise RuntimeError("SPY benchmark is incomplete")
    data["strategy_wealth"] = 100.0 * (1.0 + data["net_return_primary"]).cumprod()
    data["spy_wealth"] = 100.0 * (1.0 + data["spy_total_return"]).cumprod()
    return data


def _benchmark_returns(months: pd.Series) -> pd.DataFrame:
    benchmark = pd.read_parquet(BENCHMARKS)
    wide = (
        benchmark.loc[
            benchmark["benchmark_id"].isin(["spy_total_return", "qqq_total_return"]),
            ["holding_month", "benchmark_id", "total_return"],
        ]
        .pivot(index="holding_month", columns="benchmark_id", values="total_return")
        .reset_index()
    )
    frame = pd.DataFrame({"holding_month": months.astype(str)}).merge(
        wide, on="holding_month", how="left", validate="one_to_one"
    )
    if frame[["spy_total_return", "qqq_total_return"]].isna().any().any():
        raise RuntimeError("SPY/QQQ benchmark coverage is incomplete")
    return frame


def _scaled_monthly(phase: str, multiplier: float = 2.05) -> pd.DataFrame:
    path = MAX_RETURN / phase / "profiles" / f"constant_{multiplier:.2f}.csv"
    frame = pd.read_csv(path, dtype={"holding_month": str})
    frame = frame.sort_values("holding_month", kind="mergesort").reset_index(drop=True)
    return frame.merge(
        _benchmark_returns(frame["holding_month"]),
        on="holding_month",
        how="left",
        validate="one_to_one",
    )


def benchmark_profile() -> None:
    data = _portfolio_and_spy()
    high_return = _scaled_monthly("historical_test")
    x = pd.to_datetime(data["holding_month"].astype(str) + "-01")
    x_high_return = pd.to_datetime(high_return["holding_month"].astype(str) + "-01")
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.9))

    left.plot(x, data["strategy_wealth"], color=BLUE, lw=2.4, label="ML portfolio")
    left.plot(x, data["spy_wealth"], color=GOLD, lw=2.1, label="SPY")
    left.axhline(100.0, color=GRAY, lw=0.8)
    left.set_title("Risk-controlled specification", loc="left")
    left.set_ylabel("Portfolio value ($)")
    left.legend(loc="upper left")
    left.text(
        x.iloc[-1], data["strategy_wealth"].iloc[-1] + 1.0,
        f"${data['strategy_wealth'].iloc[-1]:.2f}", color=BLUE,
        ha="right", va="bottom", fontweight="bold",
    )
    left.text(
        x.iloc[-1], data["spy_wealth"].iloc[-1] - 1.0,
        f"${data['spy_wealth'].iloc[-1]:.2f}", color=GOLD,
        ha="right", va="top", fontweight="bold",
    )

    high_return["strategy_wealth"] = 100.0 * (1.0 + high_return["net_return_primary"]).cumprod()
    high_return["spy_wealth"] = 100.0 * (1.0 + high_return["spy_total_return"]).cumprod()
    high_return["qqq_wealth"] = 100.0 * (1.0 + high_return["qqq_total_return"]).cumprod()
    right.plot(x_high_return, high_return["strategy_wealth"], color=BLUE, lw=2.4,
               label="2.05x exposure specification")
    right.plot(x_high_return, high_return["spy_wealth"], color=GOLD, lw=2.0, label="SPY")
    right.plot(x_high_return, high_return["qqq_wealth"], color=GRAY, lw=1.8, label="QQQ")
    right.axhline(100.0, color=GRAY, lw=0.8)
    right.set_title("Validation-selected 2.05x specification", loc="left")
    right.set_ylabel("Normalized portfolio value ($)")
    right.legend(loc="upper left")
    for column, color, offset in [
        ("strategy_wealth", BLUE, 1.0), ("spy_wealth", GOLD, -1.1),
        ("qqq_wealth", GRAY, 0.2),
    ]:
        right.text(
            x_high_return.iloc[-1], high_return[column].iloc[-1] + offset,
            f"${high_return[column].iloc[-1]:.1f}", color=color,
            ha="right", va="bottom" if offset >= 0 else "top",
            fontsize=8.2, fontweight="bold",
        )

    fig.suptitle(
        "Risk-controlled and scaled-exposure specifications",
        x=0.04, y=1.02, ha="left", fontsize=13.5, fontweight="bold", color=NAVY,
    )
    fig.text(
        0.04, -0.02,
        "January 2024-August 2025. Both panels use the same fixed research cost assumptions. "
        "The 2.05x specification preserves every stock sign and relative weight while scaling the "
        "validation-selected portfolio uniformly.",
        color=GRAY, fontsize=8.7,
    )
    _save(fig, "figure_1_benchmark_profile.png")


def feature_families() -> None:
    data = pd.DataFrame(
        {
            "family": [
                "LSEG analyst information", "Benzinga news measures",
                "Valuation and fundamentals", "Ownership, filings and identity",
                "OptionMetrics", "Aggregate market state", "Price and trend",
                "Trailing risk and return", "Size, liquidity and frictions",
            ],
            "sources": [97, 56, 33, 27, 26, 23, 22, 20, 14],
            "kind": ["Alternative information", "Alternative information"]
            + ["Market and accounting"] * 7,
        }
    ).sort_values("sources")
    colors = data["kind"].map(
        {"Alternative information": GOLD, "Market and accounting": BLUE}
    )
    fig, ax = plt.subplots(figsize=(9.7, 5.4))
    bars = ax.barh(data["family"], data["sources"], color=colors)
    ax.set_xlabel("Raw feature sources")
    ax.set_title("Composition of the 318-source point-in-time feature panel", loc="left")
    for bar, value in zip(bars, data["sources"]):
        ax.text(value + 1.3, bar.get_y() + bar.get_height() / 2, str(value), va="center")
    handles = [
        Rectangle((0, 0), 1, 1, color=BLUE, label="Market and accounting"),
        Rectangle((0, 0), 1, 1, color=GOLD, label="Alternative information"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_xlim(0, 108)
    ax.text(
        0.0, -0.14,
        "The 318 sources expand to 850 finite model columns after ranking, missing indicators, "
        "categorical encoding, and aggregate-state transforms.",
        transform=ax.transAxes, color=GRAY, fontsize=8.7,
    )
    _save(fig, "figure_2_feature_families.png")


def research_chronology() -> None:
    rows = [
        ("Model history", 2015.0, 2025.67, PALE_GRAY, NAVY),
        ("Predictive validation", 2020.0, 2024.0, PALE_BLUE, BLUE),
        ("Portfolio validation", 2022.08, 2024.0, PALE_GOLD, GOLD),
        ("Historical test", 2024.0, 2025.67, BLUE, "white"),
        ("Quarantine", 2025.67, 2026.67, ORANGE, "white"),
        ("Prospective confirmation", 2026.67, 2028.75, NAVY, "white"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y_positions = np.arange(len(rows))[::-1]
    for y, (label, start, end, color, text_color) in zip(y_positions, rows):
        ax.barh(y, end - start, left=start, height=0.58, color=color, edgecolor="none")
        ax.text((start + end) / 2, y, label, ha="center", va="center", color=text_color,
                fontsize=8.8, fontweight="bold")
    ax.axvline(2024.0, color=GRAY, lw=0.8, ls="--")
    ax.axvline(2025.67, color=GRAY, lw=0.8, ls="--")
    ax.axvline(2026.67, color=GRAY, lw=0.8, ls="--")
    ax.set_xlim(2014.8, 2028.9)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_yticks([])
    ax.set_xticks(np.arange(2015, 2029))
    ax.set_xticklabels([str(year) for year in range(2015, 2029)], rotation=45, ha="right")
    ax.set_title("Research and validation chronology", loc="left")
    ax.text(
        0.0, -0.22,
        "The reported result is a leakage-controlled historical evaluation separated from validation. "
        "Prospective evaluation is required for evidence beyond the historical test.",
        transform=ax.transAxes, color=GRAY, fontsize=8.7,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _save(fig, "figure_3_research_chronology.png")


def selector_ic() -> None:
    selected = {
        "capm": "ensemble__cam_free_mlp_plus_cam__equal",
        "ff5_momentum": "ensemble__cam_free_mlp_plus_cam__equal",
        "long_net": "ensemble__cam_free_mlp_plus_cam__equal",
        "short_net": "ensemble__correlation_reduced_xgb_mlp__equal",
    }
    labels = {
        "capm": "CAPM alpha", "ff5_momentum": "FF5 + momentum alpha",
        "long_net": "Long-payoff rank", "short_net": "Short-payoff rank",
    }
    rows = []
    for target, model_id in selected.items():
        summary = pd.read_parquet(SELECTOR / target / "official_summary.parquet")
        for period in ["validation", "historical_test"]:
            row = summary.loc[
                summary["model_id"].eq(model_id) & summary["period"].eq(period)
            ]
            if len(row) != 1:
                raise RuntimeError(f"missing selected selector row: {target}/{period}")
            value = row.iloc[0]
            mean = float(value["mean_rank_ic"])
            se = float(value["rank_ic_hac_standard_error"])
            rows.append({"target": labels[target], "period": period, "mean": mean,
                         "low": mean - 1.96 * se, "high": mean + 1.96 * se})
    data = pd.DataFrame(rows)
    targets = list(labels.values())
    y = np.arange(len(targets))
    fig, ax = plt.subplots(figsize=(9.7, 5.3))
    for offset, period, color, title in [
        (-0.11, "validation", GOLD, "Validation (2020-2023)"),
        (0.11, "historical_test", BLUE, "Historical test (2024-Aug. 2025)"),
    ]:
        frame = data.loc[data["period"].eq(period)].set_index("target").loc[targets]
        means = frame["mean"].to_numpy(float)
        low = means - frame["low"].to_numpy(float)
        high = frame["high"].to_numpy(float) - means
        ax.errorbar(means, y + offset, xerr=np.vstack([low, high]), fmt="o", color=color,
                    ecolor=color, elinewidth=1.4, capsize=3, ms=6, label=title)
        for x_value, y_value in zip(means, y + offset):
            ax.text(x_value + 0.0025, y_value, f"{x_value:.3f}", va="center", fontsize=8.2)
    ax.axvline(0.0, color=NAVY, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(targets)
    ax.invert_yaxis()
    ax.set_xlim(-0.04, 0.13)
    ax.set_xlabel("Mean monthly Spearman rank IC")
    ax.set_title("Selected model rank information coefficients", loc="left")
    fig.legend(loc="upper right", bbox_to_anchor=(0.96, 0.96), ncol=2)
    ax.text(
        0.0, -0.16,
        "Points show equal-month mean IC; whiskers show HAC-based 95% intervals. "
        "Positive IC indicates useful cross-sectional ordering.",
        transform=ax.transAxes, color=GRAY, fontsize=8.7,
    )
    _save(fig, "figure_4_selector_ic.png")


def portfolio_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, "Fixed predictors", "Long and short\npayoff ranks", PALE_BLUE, BLUE),
        (0.22, "Alpha sign gate", "FF5 + momentum\nagreement", PALE_GOLD, GOLD),
        (0.42, "Causal calibration", "Rank percentiles to\nexpected payoff", PALE_BLUE, BLUE),
        (0.62, "Net-return optimizer", "Costs, risk, liquidity\nand constraints", PALE_GOLD, GOLD),
        (0.82, "$100m portfolio", "Executable monthly\ntarget weights", PALE_BLUE, BLUE),
    ]
    for idx, (x, title, subtitle, face, edge) in enumerate(boxes):
        patch = FancyBboxPatch(
            (x, 0.32), 0.16, 0.40,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.3, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + 0.08, 0.59, title, ha="center", va="center",
                fontsize=9.2, fontweight="bold", color=NAVY)
        ax.text(x + 0.08, 0.43, subtitle, ha="center", va="center",
                fontsize=8.3, color=GRAY)
        if idx < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.165, 0.52), (x + 0.215, 0.52),
                                         arrowstyle="-|>", mutation_scale=13,
                                         linewidth=1.2, color=GRAY))
    ax.set_title("From cross-sectional rankings to portfolio weights", loc="left",
                 fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(
        0.02, 0.13,
        "The models determine ordering; calibration supplies cardinal payoff estimates; "
        "the optimizer determines capital allocation after implementation frictions.",
        color=GRAY, fontsize=8.8,
    )
    _save(fig, "figure_5_portfolio_pipeline.png")


def frontier_validation() -> None:
    summary = pd.read_parquet(FRONTIER / "validation_summary.parquet")
    keep = [
        "position_1p5", "factor_15", "position_2p0", "control",
        "short_free", "turnover_75", "high_return",
    ]
    label = {
        "position_1p5": "1.5% position cap", "factor_15": "15% factor limits",
        "position_2p0": "2.0% position limit", "control": "1.0% position-limit baseline",
        "short_free": "No short floor", "turnover_75": "75% turnover cap",
        "high_return": "Joint aggressive",
    }
    data = summary.set_index("profile").loc[keep].copy()
    data["label"] = [label[item] for item in keep]
    data = data.sort_values("annualized_net_excess_return_primary", ascending=True)
    vals = data["annualized_net_excess_return_primary"].to_numpy(float) * 100.0
    colors = [BLUE if item == "position_1p5" else GRAY for item in data.index]
    fig, ax = plt.subplots(figsize=(9.7, 5.2))
    ax.barh(data["label"], vals, color=colors)
    ax.axvline(0.0, color=NAVY, lw=0.9)
    ax.set_xlabel("Annualized validation net excess return (%)")
    ax.set_title("Validation performance across portfolio configurations", loc="left")
    for i, value in enumerate(vals):
        ax.text(value + (0.08 if value >= 0 else -0.08), i, f"{value:.2f}%",
                va="center", ha="left" if value >= 0 else "right", fontsize=8.6)
    ax.text(
        0.0, -0.14,
        "February 2022-December 2023. The 1.5% position cap was selected before the "
        "January 2024-August 2025 historical test.",
        transform=ax.transAxes, color=GRAY, fontsize=8.7,
    )
    _save(fig, "figure_6_validation_frontier.png")


def monthly_returns() -> None:
    monthly = pd.read_parquet(FRONTIER / "historical_test_monthly.parquet")
    data = monthly.loc[monthly["profile"].eq("position_1p5")].copy()
    data = data.sort_values("holding_month", kind="mergesort")
    x = np.arange(len(data))
    vals = data["net_excess_return_primary"].to_numpy(float) * 100.0
    colors = np.where(vals >= 0.0, BLUE, ORANGE)
    labels = data["holding_month"].astype(str).str.replace("-", "\n", regex=False)
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.bar(x, vals, color=colors, width=0.72)
    ax.axhline(0.0, color=NAVY, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("Net excess return (%)")
    ax.set_title("Monthly net excess returns of the final portfolio", loc="left")
    ax.text(
        0.0, -0.20,
        "Primary modeled trading, borrow, and financing costs; 15 of 20 months were positive.",
        transform=ax.transAxes, color=GRAY, fontsize=8.7,
    )
    _save(fig, "figure_7_monthly_returns.png")


def contribution_and_cost() -> None:
    result = json.loads((FRONTIER / "result.json").read_text())
    row = result["historical_test"]["position_1p5"]
    long = row["annualized_long_contribution"] * 100.0
    short = row["annualized_short_contribution"] * 100.0
    cost = -row["annualized_primary_cost"] * 100.0
    net = row["annualized_net_excess_return_primary"] * 100.0
    values = [long, short, cost]
    starts = [0.0, long, long + short]
    fig, ax = plt.subplots(figsize=(9.5, 5.1))
    labels = ["Long sleeve", "Short sleeve", "Primary costs", "Net excess"]
    colors = [BLUE, GOLD, ORANGE]
    for idx, (start, value, color) in enumerate(zip(starts, values, colors)):
        bottom = min(start, start + value)
        ax.bar(idx, abs(value), bottom=bottom, color=color, width=0.62)
        ax.text(idx, start + value + (0.45 if value >= 0 else -0.45), f"{value:+.2f}%",
                ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
        if idx < 2:
            ax.plot([idx + 0.31, idx + 0.69], [start + value, start + value],
                    color=GRAY, lw=0.8, ls="--")
    ax.bar(3, net, color=NAVY, width=0.62)
    ax.text(3, net + 0.45, f"{net:.2f}%", ha="center", va="bottom", fontsize=9,
            fontweight="bold")
    ax.axhline(0.0, color=NAVY, lw=0.9)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Annualized contribution (%)")
    ax.set_title("Bridge from sleeve contributions to net excess return", loc="left")
    ax.text(
        0.0, -0.14,
        "The long sleeve generated the return; the short sleeve was slightly negative on a "
        "standalone basis and remained part of the joint risk-aware allocation.",
        transform=ax.transAxes, color=GRAY, fontsize=8.7,
    )
    _save(fig, "figure_8_return_attribution.png")


def risk_return_map() -> None:
    frontier_validation = pd.read_parquet(FRONTIER / "validation_summary.parquet")
    frontier_test = pd.read_parquet(FRONTIER / "historical_test_summary.parquet")
    scaled_validation = pd.read_csv(MAX_RETURN / "validation_summary.csv")
    scaled_test = pd.read_csv(MAX_RETURN / "historical_test_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), sharex=False, sharey=False)
    for ax, phase, title, frontier_frame, scaled_frame in [
        (axes[0], "validation", "Validation: Feb. 2022-Dec. 2023", frontier_validation, scaled_validation),
        (axes[1], "historical_test", "Historical test: Jan. 2024-Aug. 2025", frontier_test, scaled_test),
    ]:
        institutional = frontier_frame.loc[frontier_frame["profile"].eq("position_1p5")].iloc[0]
        high_return = scaled_frame.loc[scaled_frame["profile"].eq("constant_2.05")].iloc[0]
        monthly = _scaled_monthly(phase)
        benchmarks = {}
        for benchmark_id in ["spy_total_return", "qqq_total_return"]:
            returns = monthly[benchmark_id].to_numpy(float)
            benchmarks[benchmark_id] = {
                "cagr": float(np.prod(1.0 + returns) ** (12.0 / len(returns)) - 1.0),
                "volatility": float(np.std(returns, ddof=1) * np.sqrt(12.0)),
            }
        points = [
            ("1.00x reference", institutional["net_total_return_volatility"],
             institutional["net_total_return_cagr"], BLUE, "o"),
            ("2.05x exposure", high_return["annualized_realized_volatility"],
             high_return["net_total_return_cagr"], ORANGE, "D"),
            ("SPY", benchmarks["spy_total_return"]["volatility"],
             benchmarks["spy_total_return"]["cagr"], GOLD, "^"),
            ("QQQ", benchmarks["qqq_total_return"]["volatility"],
             benchmarks["qqq_total_return"]["cagr"], GRAY, "v"),
        ]
        for label, volatility, cagr, color, marker in points:
            ax.scatter(100 * volatility, 100 * cagr, s=62, color=color, marker=marker,
                       edgecolor="white", linewidth=0.7, zorder=3)
            offset = {
                "1.00x reference": (5, 13),
                "2.05x exposure": (5, -18),
            }.get(label, (5, 5))
            ax.annotate(label, (100 * volatility, 100 * cagr), xytext=offset,
                        textcoords="offset points", fontsize=7.8, color=color)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Realized annualized volatility (%)")
        ax.set_ylabel("Total-return CAGR (%)")
    fig.suptitle(
        "Return-risk comparison across validation and historical test",
        x=0.04, y=1.02, ha="left", fontsize=13.5, fontweight="bold", color=NAVY,
    )
    fig.text(
        0.04, -0.02,
        "The 2.05x exposure specification was selected on validation with a 0.75 Sharpe floor. It subsequently "
        "earned a 39.17% test CAGR with a 2.26 Sharpe under the fixed research cost assumptions.",
        color=GRAY, fontsize=8.7,
    )
    _save(fig, "figure_9_risk_return_map.png")


def exposure_safety() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7), sharey=True)
    for ax, phase, title in [
        (axes[0], "validation", "Validation"),
        (axes[1], "historical_test", "Historical test"),
    ]:
        data = _scaled_monthly(phase)
        x = pd.to_datetime(data["holding_month"].astype(str) + "-01")
        ax.plot(x, data["gross"], color=ORANGE, lw=2.0,
                label="2.05x realized gross")
        ax.axhline(2.00, color=NAVY, lw=1.1, ls=":", label="2.00x exposure reference")
        ax.axhline(1.50, color=GOLD, lw=1.1, ls="--", label="Research 1.50x limit")
        ax.fill_between(x, 2.00, 2.90, color=PALE_GOLD, alpha=0.55)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Gross exposure (x NAV)")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(1.35, 2.90)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=4)
    fig.suptitle(
        "Exposure path of the 2.05x specification relative to implementation limits",
        x=0.04, y=1.04, ha="left", fontsize=13.5, fontweight="bold", color=NAVY,
    )
    fig.text(
        0.04, -0.02,
        "Observed monthly gross reaches 2.76x in validation and 2.62x in the historical test. "
        "The scaled result is therefore an exposure sensitivity rather than a "
        "capacity-constrained implementation.",
        color=GRAY, fontsize=8.7,
    )
    _save(fig, "figure_10_exposure_safety.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    _style()
    benchmark_profile()
    feature_families()
    research_chronology()
    selector_ic()
    portfolio_pipeline()
    frontier_validation()
    monthly_returns()
    contribution_and_cost()
    risk_return_map()
    exposure_safety()


if __name__ == "__main__":
    main()
