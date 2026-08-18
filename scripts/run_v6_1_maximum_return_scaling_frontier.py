#!/usr/bin/env python3
"""Map the validation-selected absolute-return ceiling of the frozen V6.1 winner.

The study scales the already frozen 1.5% position-cap portfolio.  Security
selection, alpha gating, relative monthly weights, and signal timing never
change.  Every multiplier is replayed with stateful NAV, drift, spread,
square-root impact, financing, and borrow costs.  Validation profiles are
checkpointed individually; only validation-selected tier winners are run on
the historical-test window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from decimal import Decimal
from pathlib import Path
import sys
import traceback
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_v6_1_return_scaling_overlay as replay  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/research_v6_1/maximum_return_scaling_frontier.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_1/maximum_return_scaling_frontier"


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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _duckdb_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _multiplier_grid(config: dict[str, Any]) -> list[float]:
    values: set[Decimal] = set()
    for segment in config["multiplier_grid"]:
        start = Decimal(str(segment["start"]))
        end = Decimal(str(segment["end"]))
        step = Decimal(str(segment["step"]))
        if start <= 0 or end < start or step <= 0:
            raise ValueError("invalid multiplier-grid segment")
        current = start
        while current <= end:
            values.add(current)
            current += step
    values.add(Decimal("1.0"))
    return [float(value) for value in sorted(values)]


def _verify(config_path: Path) -> tuple[dict[str, Any], dict[str, str], Path]:
    config = json.loads(config_path.read_text())
    if config.get("status") != "AUTHORIZED_RETURN_CEILING_STUDY":
        raise RuntimeError("maximum-return scaling frontier is not authorized")
    source = config["sources"]["portfolio"]
    manifest_path = ROOT / str(source["run_manifest"])
    if _sha256(manifest_path) != str(source["run_manifest_sha256"]):
        raise RuntimeError("frozen portfolio source identity changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("frozen portfolio source is incomplete")
    source_root = ROOT / str(source["root"])
    profile = str(source["profile"])
    for phase in ("validation", "historical_test"):
        if not (source_root / phase / profile / "months").is_dir():
            raise FileNotFoundError(source_root / phase / profile / "months")
    for key in ("risk_cost", "benchmarks"):
        item = config["sources"][key]
        path = Path(str(item["path"]))
        if _sha256(path) != str(item["sha256"]):
            raise RuntimeError(f"{key} source identity changed")
    code_paths = [Path(__file__).resolve(), Path(replay.__file__).resolve()]
    code_digest = hashlib.sha256()
    for path in code_paths:
        code_digest.update(str(path).encode())
        code_digest.update(bytes.fromhex(_sha256(path)))
    identity = {
        "config_sha256": _sha256(config_path),
        "source_manifest_sha256": str(source["run_manifest_sha256"]),
        "risk_cost_sha256": str(config["sources"]["risk_cost"]["sha256"]),
        "benchmarks_sha256": str(config["sources"]["benchmarks"]["sha256"]),
        "code_identity": code_digest.hexdigest(),
    }
    return config, identity, source_root


def _replay_config(config: dict[str, Any], source_root: Path) -> dict[str, Any]:
    profile = str(config["sources"]["portfolio"]["profile"])
    evidence = config["evidence"]
    return {
        "sources": {
            "validation": {
                "month_root": str(
                    (source_root / "validation" / profile / "months").relative_to(ROOT)
                ),
                "start": str(evidence["validation_start"]),
                "end": str(evidence["validation_end"]),
            },
            "historical_test": {
                "month_root": str(
                    (source_root / "historical_test" / profile / "months").relative_to(ROOT)
                ),
                "start": str(evidence["historical_test_start"]),
                "end": str(evidence["historical_test_end"]),
            },
            "risk_cost": config["sources"]["risk_cost"],
        },
        "costs": config["costs"],
    }


def _profile(multiplier: float) -> dict[str, Any]:
    return {
        "id": f"constant_{multiplier:.2f}",
        "kind": "constant",
        "multiplier": float(multiplier),
    }


def _add_summary_fields(summary: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    result = summary.copy()
    result["multiplier"] = float(multiplier)
    result["net_total_return_cagr"] = np.power(
        1.0 + result["cumulative_net_total_return"], 12.0 / result["months"]
    ) - 1.0
    return result


def _checkpoint_paths(output: Path, phase: str, profile_id: str) -> tuple[Path, Path]:
    root = output / phase / "profiles"
    return root / f"{profile_id}.csv", root / f"{profile_id}.json"


def _run_checkpointed_profile(
    *,
    output: Path,
    phase: str,
    profile: dict[str, Any],
    config: dict[str, Any],
    risk: pd.DataFrame,
    identity: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_path, receipt_path = _checkpoint_paths(output, phase, str(profile["id"]))
    if monthly_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("identity") != identity:
            raise RuntimeError(f"checkpoint identity mismatch: {phase} {profile['id']}")
        return pd.read_csv(monthly_path), pd.DataFrame([receipt["summary"]])
    monthly, summary, reproduction_error = replay._run_phase(
        phase, config, risk, profiles=[profile]
    )
    summary = _add_summary_fields(summary, float(profile["multiplier"]))
    if abs(float(profile["multiplier"]) - 1.0) < 1e-12 and reproduction_error > 1e-10:
        raise RuntimeError(
            f"{phase} frozen-weight reproduction failed: {reproduction_error}"
        )
    _atomic_csv(monthly_path, monthly)
    _atomic_json(
        receipt_path,
        {
            "schema": "research_v6_1_maximum_return_scaling_profile_v1",
            "status": "COMPLETE",
            "identity": identity,
            "phase": phase,
            "profile": profile,
            "summary": summary.iloc[0].to_dict(),
            "base_reproduction_max_error": reproduction_error,
        },
    )
    return monthly, summary


def _benchmark_frame(config: dict[str, Any], start: str, end: str) -> pd.DataFrame:
    path = Path(str(config["sources"]["benchmarks"]["path"]))
    ids = [str(value) for value in config["sources"]["benchmarks"]["ids"]]
    quoted = ",".join("'" + value.replace("'", "''") + "'" for value in ids)
    query = f"""
        SELECT CAST(holding_month AS VARCHAR) AS holding_month,
               benchmark_id, total_return
        FROM read_parquet('{_duckdb_path(path)}')
        WHERE holding_month BETWEEN '{start}' AND '{end}'
          AND benchmark_id IN ({quoted})
        ORDER BY holding_month, benchmark_id
    """
    frame = duckdb.sql(query).df()
    expected = pd.period_range(start, end, freq="M").astype(str).tolist()
    for benchmark_id in ids:
        months = frame.loc[
            frame["benchmark_id"].eq(benchmark_id), "holding_month"
        ].astype(str).tolist()
        if months != expected:
            raise RuntimeError(f"benchmark {benchmark_id} is incomplete")
    return frame


def _benchmark_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for benchmark_id, group in frame.groupby("benchmark_id", sort=True):
        values = group["total_return"].to_numpy(float)
        wealth = np.cumprod(1.0 + values)
        peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
        result[str(benchmark_id)] = {
            "months": int(len(values)),
            "cumulative_return": float(wealth[-1] - 1.0),
            "cagr": float(wealth[-1] ** (12.0 / len(values)) - 1.0),
            "volatility": float(values.std(ddof=1) * math.sqrt(12.0)),
            "maximum_drawdown": float(np.min(wealth / peak - 1.0, initial=0.0)),
        }
    return result


def _feasible(row: pd.Series, rule: dict[str, Any]) -> bool:
    mappings = {
        "maximum_gross": "maximum_gross",
        "maximum_one_way_adv_participation": "maximum_one_way_adv_participation",
        "maximum_absolute_position": "maximum_absolute_position",
        "maximum_one_way_turnover": "maximum_one_way_turnover",
    }
    for rule_name, column in mappings.items():
        if rule_name in rule and float(row[column]) > float(rule[rule_name]) + 1e-9:
            return False
    if "minimum_net_sharpe" in rule:
        if float(row["net_sharpe"]) + 1e-12 < float(rule["minimum_net_sharpe"]):
            return False
    if bool(rule.get("require_positive_stress_excess_return", False)):
        if float(row["annualized_stress_net_excess_return"]) <= 0.0:
            return False
    return True


def _select_tiers(validation: pd.DataFrame, config: dict[str, Any]) -> dict[str, str]:
    metric = str(config["selection"]["metric"])
    winners: dict[str, str] = {}
    for tier, rule in config["selection"]["tiers"].items():
        eligible = validation.loc[
            [_feasible(row, rule) for _, row in validation.iterrows()]
        ].copy()
        if eligible.empty:
            raise RuntimeError(f"selection tier {tier} has no feasible profile")
        eligible = eligible.sort_values(
            [metric, "multiplier"], ascending=[False, True], kind="mergesort"
        )
        winners[str(tier)] = str(eligible.iloc[0]["profile"])
    return winners


def _active_inference(
    monthly: pd.DataFrame,
    benchmark: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    joined = monthly[["holding_month", "net_return_primary"]].merge(
        benchmark[["holding_month", "total_return"]],
        on="holding_month",
        how="left",
        validate="one_to_one",
    )
    if joined["total_return"].isna().any():
        raise RuntimeError("benchmark join is incomplete")
    portfolio = joined["net_return_primary"].to_numpy(float)
    market = joined["total_return"].to_numpy(float)
    active = portfolio - market
    n = len(active)
    block = int(config["inference"]["circular_block_length_months"])
    resamples = int(config["inference"]["bootstrap_resamples"])
    rng = np.random.default_rng(int(config["inference"]["bootstrap_seed"]))
    starts = rng.integers(0, n, size=(resamples, int(math.ceil(n / block))))
    indices = (starts[:, :, None] + np.arange(block)).reshape(resamples, -1)[:, :n] % n
    annualized = active[indices].mean(axis=1) * 12.0
    return {
        "annualized_active_arithmetic_return": float(active.mean() * 12.0),
        "geometric_relative_cagr": float(
            (np.prod(1.0 + portfolio) / np.prod(1.0 + market)) ** (12.0 / n)
            - 1.0
        ),
        "tracking_error": float(active.std(ddof=1) * math.sqrt(12.0)),
        "information_ratio": float(
            active.mean() / active.std(ddof=1) * math.sqrt(12.0)
        ),
        "correlation": float(np.corrcoef(portfolio, market)[0, 1]),
        "beta_total_return": float(np.cov(portfolio, market, ddof=1)[0, 1] / np.var(market, ddof=1)),
        "bootstrap_active_return_ci_95": np.quantile(annualized, [0.025, 0.975]).tolist(),
        "bootstrap_probability_active_return_positive": float(np.mean(annualized > 0.0)),
    }


def _profile_comparison(
    monthly: pd.DataFrame,
    summary: pd.Series,
    benchmarks: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "profile": str(summary["profile"]),
        "multiplier": float(summary["multiplier"]),
        "portfolio_cagr": float(summary["net_total_return_cagr"]),
        "benchmarks": {},
    }
    for benchmark_id in config["sources"]["benchmarks"]["ids"]:
        group = benchmarks.loc[benchmarks["benchmark_id"].eq(benchmark_id)].copy()
        values = group["total_return"].to_numpy(float)
        benchmark_cagr = float(np.prod(1.0 + values) ** (12.0 / len(values)) - 1.0)
        result["benchmarks"][str(benchmark_id)] = {
            "cagr": benchmark_cagr,
            "portfolio_minus_benchmark_cagr": float(
                summary["net_total_return_cagr"] - benchmark_cagr
            ),
            "portfolio_beats_benchmark": bool(
                float(summary["net_total_return_cagr"]) > benchmark_cagr
            ),
            **_active_inference(monthly, group, config),
        }
    result["beats_all_benchmarks"] = all(
        item["portfolio_beats_benchmark"]
        for item in result["benchmarks"].values()
    )
    return result


def _result_markdown(result: dict[str, Any]) -> str:
    rows = []
    for tier, profile_id in result["tier_winners"].items():
        summary = result["historical_test"][profile_id]
        comparison = result["historical_benchmark_comparison"][profile_id]
        rows.append(
            "| {tier} | {scale:.2f}x | {cagr:.2%} | {spy:.2%} | {qqq:.2%} | "
            "{sharpe:.2f} | {stress:.2f} | {dd:.2%} | {gross:.2f} | {adv:.1%} | {all_benchmarks} |".format(
                tier=tier,
                scale=float(summary["multiplier"]),
                cagr=float(summary["net_total_return_cagr"]),
                spy=float(comparison["benchmarks"]["spy_total_return"]["cagr"]),
                qqq=float(comparison["benchmarks"]["qqq_total_return"]["cagr"]),
                sharpe=float(summary["net_sharpe"]),
                stress=float(summary["stress_sharpe"]),
                dd=float(summary["maximum_drawdown"]),
                gross=float(summary["maximum_gross"]),
                adv=float(summary["maximum_one_way_adv_participation"]),
                all_benchmarks="yes" if comparison["beats_all_benchmarks"] else "no",
            )
        )
    validation_ceiling = result["validation"][
        result["tier_winners"]["absolute_return_ceiling"]
    ]
    return f"""# Frozen-selector maximum-return scaling frontier

The frozen 1.5% position-cap portfolio is replayed at constant exposure
multipliers.  Stock selection, ranks, FF5+momentum gate, relative holdings, and
signal dates do not change.  Spread, square-root impact, financing, borrow,
drift, and NAV are recomputed at every scale.

The search does not stop when SPY or QQQ is crossed.  The absolute validation
ceiling is **{validation_ceiling['multiplier']:.2f}x**, where validation net
total-return CAGR reaches **{validation_ceiling['net_total_return_cagr']:.2%}**.

| Validation-selected tier | Scale | Historical CAGR | SPY CAGR | QQQ CAGR | Sharpe | Stress Sharpe | Max drawdown | Max gross | Max ADV | Beats both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{os.linesep.join(rows)}

`realistic_100m` and `expanded_institutional_100m` are constrained capacity
views.  `balanced_high_return` retains a validation Sharpe floor.
`stress_positive_ceiling` is the largest return selected while stressed excess
return remains positive.  `absolute_return_ceiling` is a research limit, not a
deployable mandate: its leverage, turnover, ADV participation, and drawdown
must be read together with its return.

All tier choices use February 2022--December 2023 only.  January 2024--August
2025 is evaluated after the tier winners are frozen.  Prospective confirmation
remains separate.
"""


def run(config_path: Path, output_root: Path) -> Path:
    config, identity, source_root = _verify(config_path)
    output = output_root / (
        f"{identity['config_sha256'][:12]}-"
        f"{identity['source_manifest_sha256'][:12]}-"
        f"{identity['code_identity'][:12]}"
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    multipliers = _multiplier_grid(config)
    _atomic_json(
        manifest_path,
        {
            "schema": "research_v6_1_maximum_return_scaling_frontier_run_v1",
            "status": "RUNNING",
            "identity": identity,
            "validation_profiles_total": len(multipliers),
        },
    )
    replay_config = _replay_config(config, source_root)
    risk = replay._load_risk(replay_config)
    validation_summaries: list[pd.DataFrame] = []
    for index, multiplier in enumerate(multipliers, start=1):
        profile = _profile(multiplier)
        _, summary = _run_checkpointed_profile(
            output=output,
            phase="validation",
            profile=profile,
            config=replay_config,
            risk=risk,
            identity=identity,
        )
        validation_summaries.append(summary)
        print(
            f"VALIDATION {index}/{len(multipliers)} scale={multiplier:.2f} "
            f"cagr={float(summary.iloc[0]['net_total_return_cagr']):.6f} "
            f"sharpe={float(summary.iloc[0]['net_sharpe']):.4f}",
            flush=True,
        )
    validation = pd.concat(validation_summaries, ignore_index=True).sort_values(
        "multiplier", kind="mergesort"
    )
    _atomic_csv(output / "validation_summary.csv", validation)
    tier_winners = _select_tiers(validation, config)
    _atomic_json(output / "tier_winners.json", tier_winners)
    print(f"TIER_WINNERS {json.dumps(tier_winners, sort_keys=True)}", flush=True)

    profile_by_id = {
        str(_profile(multiplier)["id"]): _profile(multiplier)
        for multiplier in multipliers
    }
    selected_ids = list(dict.fromkeys(["constant_1.00", *tier_winners.values()]))
    test_monthly: dict[str, pd.DataFrame] = {}
    test_summary_rows: list[pd.DataFrame] = []
    for index, profile_id in enumerate(selected_ids, start=1):
        monthly, summary = _run_checkpointed_profile(
            output=output,
            phase="historical_test",
            profile=profile_by_id[profile_id],
            config=replay_config,
            risk=risk,
            identity=identity,
        )
        test_monthly[profile_id] = monthly
        test_summary_rows.append(summary)
        print(
            f"HISTORICAL_TEST {index}/{len(selected_ids)} {profile_id} "
            f"cagr={float(summary.iloc[0]['net_total_return_cagr']):.6f}",
            flush=True,
        )
    historical = pd.concat(test_summary_rows, ignore_index=True)
    _atomic_csv(output / "historical_test_summary.csv", historical)

    evidence = config["evidence"]
    validation_benchmarks = _benchmark_frame(
        config,
        str(evidence["validation_start"]),
        str(evidence["validation_end"]),
    )
    historical_benchmarks = _benchmark_frame(
        config,
        str(evidence["historical_test_start"]),
        str(evidence["historical_test_end"]),
    )
    validation_records = {
        str(row.profile): row._asdict() for row in validation.itertuples(index=False)
    }
    historical_records = {
        str(row.profile): row._asdict() for row in historical.itertuples(index=False)
    }
    historical_comparison = {
        profile_id: _profile_comparison(
            test_monthly[profile_id],
            historical.loc[historical["profile"].eq(profile_id)].iloc[0],
            historical_benchmarks,
            config,
        )
        for profile_id in selected_ids
    }
    validation_benchmark_summary = _benchmark_summary(validation_benchmarks)
    historical_benchmark_summary = _benchmark_summary(historical_benchmarks)
    absolute_id = tier_winners["absolute_return_ceiling"]
    absolute_index = validation.index[validation["profile"].eq(absolute_id)][0]
    absolute_position = validation.index.get_loc(absolute_index)
    interior_peak = 0 < absolute_position < len(validation) - 1
    if interior_peak:
        current = float(validation.iloc[absolute_position]["net_total_return_cagr"])
        interior_peak = bool(
            current > float(validation.iloc[absolute_position - 1]["net_total_return_cagr"])
            and current > float(validation.iloc[absolute_position + 1]["net_total_return_cagr"])
        )
    result = {
        "schema": "research_v6_1_maximum_return_scaling_frontier_result_v1",
        "status": "COMPLETE",
        "identity": identity,
        "tier_winners": tier_winners,
        "validation": validation_records,
        "historical_test": historical_records,
        "validation_benchmarks": validation_benchmark_summary,
        "historical_benchmarks": historical_benchmark_summary,
        "historical_benchmark_comparison": historical_comparison,
        "absolute_validation_peak_is_interior": interior_peak,
        "selection_used_historical_test_outcomes": False,
        "frozen_selector_signal_and_relative_weights_unchanged": True,
        "reused_prior_optimizer_frontier_without_rerun": True,
        "interpretation": config["interpretation"],
    }
    _atomic_json(output / "result.json", result)
    _atomic_text(output / "RESULT.md", _result_markdown(result))
    _atomic_json(
        manifest_path,
        {
            "schema": "research_v6_1_maximum_return_scaling_frontier_run_v1",
            "status": "COMPLETE",
            "identity": identity,
            "validation_profiles_total": len(multipliers),
            "historical_test_profiles": selected_ids,
            "tier_winners": tier_winners,
        },
    )
    print(f"COMPLETE {output}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        run(args.config.resolve(), args.output_root.resolve())
    except Exception as exc:
        error_root = args.output_root.resolve() / "latest_error"
        error_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            error_root / "error.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
