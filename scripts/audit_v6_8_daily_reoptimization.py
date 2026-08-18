#!/usr/bin/env python3
"""Independent, focused accounting and constraint audit for V6.8."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / (
    "artifacts/research_v6_8/daily_reoptimization/"
    "d55a1db689f9-2ef39b0755c5-2e422abd519f"
)


def _duck(path: Path) -> str:
    return str(path).replace("'", "''")


def _read(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"SELECT * FROM read_parquet('{_duck(path)}')").df()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _recompute(monthly: pd.DataFrame, daily: pd.DataFrame) -> dict[str, float]:
    excess = monthly["net_excess_return"].to_numpy(float)
    total = monthly["net_return"].to_numpy(float)
    nav = daily["nav_end"].to_numpy(float)
    peak = np.maximum.accumulate(np.r_[daily["nav_start"].iloc[0], nav])[1:]
    wealth = float(np.prod(1.0 + total))
    return {
        "annualized_net_excess_return": float(excess.mean() * 12.0),
        "annualized_realized_volatility": float(excess.std(ddof=1) * math.sqrt(12.0)),
        "net_sharpe": float(excess.mean() / excess.std(ddof=1) * math.sqrt(12.0)),
        "maximum_drawdown": float(np.min(nav / peak - 1.0, initial=0.0)),
        "total_return_cagr": float(wealth ** (12.0 / len(monthly)) - 1.0),
    }


def audit(run_root: Path) -> dict[str, Any]:
    result = json.loads((run_root / "result.json").read_text())
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    tolerance = 2e-8
    for phase, result_key in (("validation", "validation"), ("historical_test", "historical_test")):
        root = run_root / phase
        monthly = _read(root / "monthly.parquet")
        daily = _read(root / "daily.parquet")
        actions = _read(root / "actions.parquet")
        events = _read(root / "events.parquet")
        trades = _read(root / "trades.parquet")
        expected = result[result_key]["v6_8_reoptimization"]
        recomputed = _recompute(monthly, daily)
        metric_delta = {
            key: float(recomputed[key] - float(expected[key])) for key in recomputed
        }
        checks[f"{phase}_headline_metrics_tie"] = max(
            abs(value) for value in metric_delta.values()
        ) <= tolerance
        monthly_nav_error = float(
            np.max(
                np.abs(
                    monthly["nav_end"].to_numpy(float)
                    / monthly["nav_start"].to_numpy(float)
                    - 1.0
                    - monthly["net_return"].to_numpy(float)
                ),
                initial=0.0,
            )
        )
        checks[f"{phase}_monthly_nav_tie"] = monthly_nav_error <= tolerance
        continuity_error = 0.0
        for _, frame in daily.groupby("holding_month", sort=False):
            if len(frame) > 1:
                continuity_error = max(
                    continuity_error,
                    float(
                        np.max(
                            np.abs(
                                frame["nav_start"].to_numpy(float)[1:]
                                - frame["nav_end"].to_numpy(float)[:-1]
                            )
                        )
                    ),
                )
        checks[f"{phase}_daily_nav_continuity"] = continuity_error <= 1e-7
        checks[f"{phase}_finite_accounting"] = bool(
            np.isfinite(
                monthly.select_dtypes(include=[np.number]).to_numpy(float)
            ).all()
            and np.isfinite(daily.select_dtypes(include=[np.number]).to_numpy(float)).all()
        )
        checks[f"{phase}_refill_not_above_removed"] = bool(
            (events["refill_gross"] <= events["removed_gross"] + 2e-6).all()
        )
        checks[f"{phase}_final_gross_not_above_pre_action"] = bool(
            (events["final_gross_before_cost"] <= events["pre_action_gross"] + 2e-6).all()
        )
        warned_trades = trades.loc[trades["warned"].astype(bool)]
        checks[f"{phase}_warned_names_not_refilled"] = bool(
            (warned_trades["refill_delta"].abs() <= 2e-6).all()
        )
        survivor = trades.loc[~trades["warned"].astype(bool)]
        monotone = (
            ((survivor["pre_weight"] > 0.0) & (survivor["refill_delta"] >= -2e-6))
            | ((survivor["pre_weight"] < 0.0) & (survivor["refill_delta"] <= 2e-6))
        )
        checks[f"{phase}_survivor_adds_are_same_side"] = bool(monotone.all())
        checks[f"{phase}_warning_count_ties"] = int(len(actions)) == int(
            result["data_quality"][f"{'validation' if phase == 'validation' else 'historical'}_warning_actions"]
        )
        details[phase] = {
            "metric_delta": metric_delta,
            "monthly_nav_error": monthly_nav_error,
            "daily_continuity_error_dollars": continuity_error,
            "warnings": int(len(actions)),
            "events": int(len(events)),
            "trades": int(len(trades)),
            "solver_status_counts": {
                str(key): int(value)
                for key, value in events["solver_status"].value_counts().items()
            },
            "removed_gross": float(events["removed_gross"].sum()),
            "refilled_gross": float(events["refill_gross"].sum()),
        }
    passed = all(checks.values())
    audit_result = {
        "schema": "research_v6_8_independent_audit_v1",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "details": details,
    }
    _atomic_json(run_root / "independent_audit.json", audit_result)
    if not passed:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"V6.8 independent audit failed: {failed}")
    return audit_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.run_root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
