#!/usr/bin/env python3
"""Run a validation-selected net-return frontier on frozen V6.1 selectors."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_v6_1_final_optimizer_challenge as prior  # noqa: E402
from scripts import run_v6_1_portfolio_step_02 as base  # noqa: E402
from portfolio_bot.research_v6_1.portfolio import FACTOR_NAMES  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/research_v6_1/net_return_frontier.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_1/net_return_frontier"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _code_identity() -> str:
    files = [
        Path(__file__).resolve(),
        ROOT / "scripts/run_v6_1_final_optimizer_challenge.py",
        ROOT / "scripts/run_v6_1_portfolio_step_02.py",
        ROOT / "src/portfolio_bot/research_v6_1/portfolio.py",
        ROOT / "src/portfolio_bot/allocation/costs.py",
        ROOT / "src/portfolio_bot/risk_cost/cost_math.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _verify(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], str, str]:
    study = json.loads(config_path.read_text())
    if study.get("status") != "AUTHORIZED_POST_DIAGNOSTIC_TEST":
        raise RuntimeError("net-return frontier is not authorized")
    base_path = ROOT / str(study["base_portfolio_contract"]["path"])
    if _sha256(base_path) != str(study["base_portfolio_contract"]["sha256"]):
        raise RuntimeError("bound base portfolio contract identity changed")
    frozen_config, _, input_identity, paths = base.verify_contract(base_path)
    control_root = ROOT / str(study["frozen_control"]["path"])
    manifest_path = control_root / "run_manifest.json"
    if _sha256(manifest_path) != str(study["frozen_control"]["manifest_sha256"]):
        raise RuntimeError("frozen control manifest identity changed")
    if json.loads(manifest_path.read_text()).get("status") != "COMPLETE":
        raise RuntimeError("frozen control result is incomplete")
    comparator = study.get("paired_comparator")
    if comparator is not None:
        comparator_root = ROOT / str(comparator["path"])
        comparator_manifest = comparator_root / "run_manifest.json"
        if _sha256(comparator_manifest) != str(comparator["manifest_sha256"]):
            raise RuntimeError("paired comparator manifest identity changed")
        if json.loads(comparator_manifest.read_text()).get("status") != "COMPLETE":
            raise RuntimeError("paired comparator result is incomplete")
    config_sha = _sha256(config_path)
    identity = hashlib.sha256(
        _json_bytes(
            {
                "config": config_sha,
                "base": study["base_portfolio_contract"]["sha256"],
                "inputs": input_identity,
                "control": study["frozen_control"]["manifest_sha256"],
            }
        )
    ).hexdigest()
    return study, frozen_config, paths, config_sha, identity


def _profile_config(frozen: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(frozen)
    allowed = set(result["optimizer"])
    unknown = set(profile.get("overrides", {})) - allowed
    if unknown:
        raise ValueError(f"profile {profile['id']} has unknown optimizer overrides: {sorted(unknown)}")
    result["optimizer"].update(profile.get("overrides", {}))
    return result


def _solver_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(profile["id"]),
        "family": str(profile["family"]),
        "dispersion_scale": 1.0,
        "risk_aversion": 0.0,
        "concentration_aversion": 0.0,
        "objective_cost_multiplier": 1.0,
    }


def _maximum_drawdown(values: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + np.asarray(values, dtype=float))
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return float(np.min(wealth / peak - 1.0, initial=0.0))


def _position_exposures(root: Path, frame: pd.DataFrame) -> dict[str, float]:
    max_sic_net = 0.0
    max_sic_gross = 0.0
    max_position = 0.0
    max_candidate_position = 0.0
    max_reduction_only_position = 0.0
    for path in sorted((root / "months").glob("*.parquet")):
        positions = pd.read_parquet(path)
        active = positions.loc[positions["weight"].abs().gt(1e-14)].copy()
        if active.empty:
            continue
        max_position = max(max_position, float(active["weight"].abs().max()))
        candidate = active["long_candidate"].astype(bool) | active[
            "short_candidate"
        ].astype(bool)
        if candidate.any():
            max_candidate_position = max(
                max_candidate_position,
                float(active.loc[candidate, "weight"].abs().max()),
            )
        reduction_only = ~candidate & active["previous_weight"].abs().gt(1e-12)
        if reduction_only.any():
            max_reduction_only_position = max(
                max_reduction_only_position,
                float(active.loc[reduction_only, "weight"].abs().max()),
            )
        grouped = active.groupby("sic2_industry_category", observed=True)["weight"]
        max_sic_net = max(max_sic_net, float(grouped.sum().abs().max()))
        max_sic_gross = max(
            max_sic_gross,
            float(grouped.apply(lambda values: values.abs().sum()).max()),
        )
    factor_columns = [f"factor_exposure_{name}" for name in FACTOR_NAMES]
    return {
        "maximum_absolute_position": max_position,
        "maximum_candidate_absolute_position": max_candidate_position,
        "maximum_reduction_only_absolute_position": max_reduction_only_position,
        "maximum_absolute_factor_exposure": float(frame[factor_columns].abs().to_numpy().max()),
        "maximum_sic2_net": max_sic_net,
        "maximum_sic2_gross": max_sic_gross,
    }


def _aggregate(output: Path, phase: str, profile: str, frame: pd.DataFrame) -> dict[str, Any]:
    result = prior._aggregate(frame)
    result.update(_position_exposures(output / phase / profile, frame))
    total = frame["net_return_primary"].to_numpy(float)
    result.update(
        {
            "maximum_gross": float(frame["gross"].max()),
            "maximum_one_way_turnover": float(frame["turnover"].max()),
            "cumulative_net_total_return": float(np.prod(1.0 + total) - 1.0),
            "net_total_return_cagr": float(np.prod(1.0 + total) ** (12.0 / len(total)) - 1.0),
            "net_total_return_volatility": float(total.std(ddof=1) * math.sqrt(12.0)),
        }
    )
    return result


def _feasible(summary: pd.Series, rule: dict[str, Any], profile: str) -> bool:
    allowed = rule.get("allowed_profiles")
    if allowed is not None and profile not in set(map(str, allowed)):
        return False
    mappings = {
        "maximum_gross": "maximum_gross",
        "maximum_one_way_adv_participation": "maximum_adv_participation",
        "maximum_absolute_position": "maximum_absolute_position",
        "maximum_candidate_absolute_position": "maximum_candidate_absolute_position",
        "maximum_one_way_turnover": "maximum_one_way_turnover",
        "maximum_absolute_factor_exposure": "maximum_absolute_factor_exposure",
        "maximum_sic2_net": "maximum_sic2_net",
        "maximum_sic2_gross": "maximum_sic2_gross",
    }
    for rule_name, summary_name in mappings.items():
        if rule_name in rule and float(summary[summary_name]) > float(rule[rule_name]) + 5e-7:
            return False
    if bool(rule.get("require_positive_stressed_net_return", False)):
        if float(summary["annualized_net_excess_return_stress"]) <= 0.0:
            return False
    return True


def _select_tiers(
    validation: pd.DataFrame,
    study: dict[str, Any],
) -> dict[str, str]:
    order = {str(profile["id"]): index for index, profile in enumerate(study["profiles"])}
    result: dict[str, str] = {}
    for tier, rule in study["selection_tiers"].items():
        eligible = validation.loc[
            [
                _feasible(row, rule, str(row["profile"]))
                for _, row in validation.iterrows()
            ]
        ].copy()
        if eligible.empty:
            raise RuntimeError(f"selection tier {tier} has no feasible validation profile")
        eligible["profile_order"] = eligible["profile"].map(order)
        selection_metric = str(
            rule.get("selection_metric", study["selection"]["metric"])
        )
        if selection_metric not in eligible.columns:
            raise RuntimeError(
                f"selection tier {tier} has unknown metric {selection_metric}"
            )
        eligible = eligible.sort_values(
            [selection_metric, "profile_order"],
            ascending=[False, True],
            kind="mergesort",
        )
        result[str(tier)] = str(eligible.iloc[0]["profile"])
    return result


def _benchmark_summary(
    frame: pd.DataFrame,
    benchmarks: pd.DataFrame,
) -> dict[str, Any]:
    spy = benchmarks.loc[
        benchmarks["benchmark_id"].astype(str).eq("spy_total_return"),
        ["holding_month", "total_return"],
    ].rename(columns={"total_return": "spy_total_return"})
    joined = frame[["holding_month", "net_return_primary"]].merge(
        spy, on="holding_month", how="left", validate="one_to_one"
    )
    if joined["spy_total_return"].isna().any():
        raise RuntimeError("SPY benchmark is incomplete for historical test")
    portfolio = joined["net_return_primary"].to_numpy(float)
    market = joined["spy_total_return"].to_numpy(float)
    active = portfolio - market
    return {
        "portfolio_cumulative_return": float(np.prod(1.0 + portfolio) - 1.0),
        "portfolio_cagr": float(np.prod(1.0 + portfolio) ** (12.0 / len(portfolio)) - 1.0),
        "portfolio_total_volatility": float(portfolio.std(ddof=1) * math.sqrt(12.0)),
        "portfolio_maximum_drawdown": _maximum_drawdown(portfolio),
        "spy_cumulative_return": float(np.prod(1.0 + market) - 1.0),
        "spy_cagr": float(np.prod(1.0 + market) ** (12.0 / len(market)) - 1.0),
        "spy_volatility": float(market.std(ddof=1) * math.sqrt(12.0)),
        "spy_maximum_drawdown": _maximum_drawdown(market),
        "annualized_active_arithmetic_return": float(active.mean() * 12.0),
        "annualized_geometric_relative_return": float(
            (np.prod(1.0 + portfolio) / np.prod(1.0 + market)) ** (12.0 / len(portfolio)) - 1.0
        ),
        "tracking_error": float(active.std(ddof=1) * math.sqrt(12.0)),
        "information_ratio": float(active.mean() / active.std(ddof=1) * math.sqrt(12.0)),
        "correlation_with_spy": float(np.corrcoef(portfolio, market)[0, 1]),
    }


def _write_result_markdown(path: Path, result: dict[str, Any]) -> None:
    rows = []
    for tier, profile in result["tier_winners"].items():
        item = result["historical_test"][profile]
        benchmark = result["benchmark_comparison"][profile]
        rows.append(
            "| {tier} | {profile} | {ret:.2%} | {cagr:.2%} | {sharpe:.3f} | "
            "{stress:.3f} | {dd:.2%} | {gross:.2f} | {adv:.2%} | {spy:.2%} |".format(
                tier=tier,
                profile=profile,
                ret=item["annualized_net_excess_return_primary"],
                cagr=benchmark["portfolio_cagr"],
                sharpe=item["net_sharpe_primary"],
                stress=item["net_sharpe_stress"],
                dd=item["maximum_drawdown_primary"],
                gross=item["maximum_gross"],
                adv=item["maximum_adv_participation"],
                spy=benchmark["spy_cagr"],
            )
        )
    content = """# V6.1 frozen-selector net-return frontier

This is a user-authorized post-diagnostic portfolio study. The V6.1 selectors,
features, ranks, FF5+momentum alpha gate, candidate tails, causal rank-to-payoff
calibration, cost model, and monthly schedule are unchanged. Profiles are
selected only on February 2022--December 2023 annualized net excess return;
January 2024--August 2025 outcomes never select a profile.

| Selection tier | Validation-selected profile | Annualized net excess | Total-return CAGR | Net Sharpe | Stress Sharpe | Max drawdown | Max gross | Max ADV | SPY CAGR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

The unrestricted tier is diagnostic rather than deployable. The practical and
expanded-institutional tiers are defined by validation-period realized gross,
liquidity, position, turnover, factor, and SIC2 limits. All reported costs are
stateful and include spread, square-root impact, financing, and borrow.

September 2025--August 2026 remains quarantined. None of these historical
results is untouched confirmation or a production forecast.
""".format(rows="\n".join(rows))
    _atomic_text(path, content)


def run(
    config_path: Path,
    output_root: Path,
    *,
    smoke: bool = False,
    resume_output: Path | None = None,
) -> Path:
    study, frozen, paths, config_sha, input_identity = _verify(config_path)
    code_identity = _code_identity()
    output = (
        resume_output.resolve()
        if resume_output is not None
        else output_root / f"{config_sha[:12]}-{input_identity[:12]}-{code_identity[:12]}"
    )
    if smoke:
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": config_sha,
        "input_identity": input_identity,
        "code_identity": code_identity,
    }
    reporting_repair: dict[str, Any] | None = None
    manifest_path = output / "run_manifest.json"
    if resume_output is not None:
        if not manifest_path.is_file():
            raise RuntimeError("resume output has no run manifest")
        prior_manifest = json.loads(manifest_path.read_text())
        for key in ("config_sha256", "input_identity"):
            if prior_manifest.get(key) != identity[key]:
                raise RuntimeError(f"resume output {key} mismatch")
        # The initial run completed every optimizer checkpoint and then failed
        # only because two reporting-field names were absent. Preserve the
        # authenticated optimizer checkpoint identity while recording the new
        # reporting-driver identity separately.
        identity["code_identity"] = str(prior_manifest["code_identity"])
        reporting_repair = {
            "reason": "reporting_field_name_repair_after_complete_validation",
            "execution_code_identity": identity["code_identity"],
            "reporting_code_identity": code_identity,
        }
    base._atomic_json(
        manifest_path,
        {
            "schema": "research_v6_1_net_return_frontier_run_v1",
            "status": "RUNNING",
            **identity,
            "reporting_repair": reporting_repair,
            "profiles": [str(profile["id"]) for profile in study["profiles"]],
        },
    )
    loader = copy.deepcopy(frozen)
    loader["evidence"]["portfolio_start"] = study["evidence"]["validation_start"]
    loader["evidence"]["portfolio_end"] = study["evidence"]["historical_test_end"]
    inputs = base.load_inputs(loader, paths)
    validation_months = pd.period_range(
        study["evidence"]["validation_start"],
        study["evidence"]["validation_end"],
        freq="M",
    ).astype(str).tolist()
    test_months = pd.period_range(
        study["evidence"]["historical_test_start"],
        study["evidence"]["historical_test_end"],
        freq="M",
    ).astype(str).tolist()
    profiles = study["profiles"]
    if smoke:
        validation_months = validation_months[:1]
        test_months = test_months[:1]
        profiles = profiles[:2]
    month_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    validation_frames: list[pd.DataFrame] = []
    validation_summaries: list[dict[str, Any]] = []
    for profile in profiles:
        profile_id = str(profile["id"])
        frame = prior._run_profile(
            output=output,
            phase="validation",
            months=validation_months,
            profile=_solver_profile(profile),
            identity=identity,
            inputs=inputs,
            loader_config=_profile_config(loader, profile),
            month_cache=month_cache,
        )
        validation_frames.append(frame)
        summary = _aggregate(output, "validation", profile_id, frame)
        validation_summaries.append(summary)
        print(
            f"PROFILE {profile_id} validation_net="
            f"{summary['annualized_net_excess_return_primary']:.6f} "
            f"sharpe={summary['net_sharpe_primary']:.6f}",
            flush=True,
        )
    validation_monthly = pd.concat(validation_frames, ignore_index=True)
    validation_summary = pd.DataFrame(validation_summaries)
    base._atomic_parquet(output / "validation_monthly.parquet", validation_monthly)
    base._atomic_parquet(output / "validation_summary.parquet", validation_summary)
    if smoke:
        base._atomic_json(
            output / "run_manifest.json",
            {
                "schema": "research_v6_1_net_return_frontier_run_v1",
                "status": "SMOKE_COMPLETE",
                **identity,
            },
        )
        return output

    tier_winners = _select_tiers(validation_summary, study)
    base._atomic_json(output / "tier_winners.json", tier_winners)
    print(f"TIER WINNERS {json.dumps(tier_winners, sort_keys=True)}", flush=True)
    profile_lookup = {str(profile["id"]): profile for profile in study["profiles"]}
    selected_profiles = list(dict.fromkeys(["control", *tier_winners.values()]))
    test_frames: dict[str, pd.DataFrame] = {}
    test_summaries: dict[str, dict[str, Any]] = {}
    benchmark_summaries: dict[str, dict[str, Any]] = {}
    for profile_id in selected_profiles:
        profile = profile_lookup[profile_id]
        frame = prior._run_profile(
            output=output,
            phase="historical_test",
            months=test_months,
            profile=_solver_profile(profile),
            identity=identity,
            inputs=inputs,
            loader_config=_profile_config(loader, profile),
            month_cache=month_cache,
        )
        test_frames[profile_id] = frame
        test_summaries[profile_id] = _aggregate(
            output, "historical_test", profile_id, frame
        )
        benchmark_summaries[profile_id] = _benchmark_summary(frame, inputs.benchmarks)
        print(
            f"TEST {profile_id} net="
            f"{test_summaries[profile_id]['annualized_net_excess_return_primary']:.6f} "
            f"sharpe={test_summaries[profile_id]['net_sharpe_primary']:.6f}",
            flush=True,
        )
    test_monthly = pd.concat(test_frames.values(), ignore_index=True)
    base._atomic_parquet(output / "historical_test_monthly.parquet", test_monthly)
    base._atomic_parquet(
        output / "historical_test_summary.parquet",
        pd.DataFrame(test_summaries.values()),
    )
    comparator = study.get("paired_comparator")
    if comparator is None:
        comparator = {
            "label": "frozen_control",
            "path": study["frozen_control"]["path"],
            "monthly_file": "monthly_results.parquet",
            "filter_column": "variant",
            "filter_value": study["frozen_control"]["variant"],
        }
    paired_control = pd.read_parquet(
        ROOT / comparator["path"] / comparator["monthly_file"]
    )
    paired_control = paired_control.loc[
        paired_control[str(comparator["filter_column"])].eq(
            comparator["filter_value"]
        )
    ].sort_values("holding_month", kind="mergesort").reset_index(drop=True)
    inference = {
        profile_id: prior._paired_inference(
            frame, paired_control, study["inference"]
        )
        for profile_id, frame in test_frames.items()
    }
    result = {
        "schema": "research_v6_1_net_return_frontier_result_v1",
        "status": "COMPLETE",
        "tier_winners": tier_winners,
        "validation_profiles": validation_summaries,
        "historical_test": test_summaries,
        "benchmark_comparison": benchmark_summaries,
        "paired_comparator": str(comparator["label"]),
        "paired_inference_vs_comparator": inference,
        "frozen_selector_and_gate_unchanged": True,
        "selection_used_historical_test_outcomes": False,
        "post_diagnostic_historical_development_only": True,
        "quarantined_start": study["evidence"]["quarantined_start"],
        "untouched_confirmation_earliest_start": study["evidence"][
            "untouched_confirmation_earliest_start"
        ],
    }
    base._atomic_json(output / "result.json", result)
    _write_result_markdown(output / "RESULT.md", result)
    base._atomic_json(
        output / "run_manifest.json",
        {
            "schema": "research_v6_1_net_return_frontier_run_v1",
            "status": "COMPLETE",
            **identity,
            "reporting_repair": reporting_repair,
            "validation_profile_months": len(profiles) * len(validation_months),
            "historical_test_profiles": selected_profiles,
            "historical_test_months": len(selected_profiles) * len(test_months),
            "tier_winners": tier_winners,
        },
    )
    print(json.dumps(result, indent=2, default=str), flush=True)
    print(f"COMPLETE {output}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume-output",
        type=Path,
        help="Resume an authenticated lineage after a reporting-only failure.",
    )
    args = parser.parse_args()
    try:
        run(
            args.config.resolve(),
            args.output_root.resolve(),
            smoke=args.smoke,
            resume_output=args.resume_output,
        )
    except Exception as exc:
        error_root = args.output_root.resolve() / "latest_error"
        error_root.mkdir(parents=True, exist_ok=True)
        base._atomic_json(
            error_root / "error.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
