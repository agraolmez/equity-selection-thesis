#!/usr/bin/env python3
"""Audit the completed V6.1 final optimizer challenge under frozen position semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v6_1_portfolio_step_02 import _atomic_json  # noqa: E402


DEFAULT_RUN = ROOT / (
    "artifacts/research_v6_1/final_optimizer_challenge/"
    "72e9dc82f89e-67cac227c15c-a8f3e5acf95a"
)
BASE_CONFIG = ROOT / "configs/research_v6_1/portfolio_step_02.json"
FROZEN_CONTROL = ROOT / (
    "artifacts/research_v6_1/portfolio_step_02/"
    "48f429eb0d7f-53a6425a3d92-7256b91cc393"
)
FROZEN_VARIANT = "return_plus_ff5_momentum__optimized_10pct_vol"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(run_root: Path) -> dict[str, object]:
    result_path = run_root / "result.json"
    manifest_path = run_root / "run_manifest.json"
    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if result.get("status") != "COMPLETE" or manifest.get("status") != "COMPLETE":
        raise RuntimeError("optimizer challenge is not complete")
    profile = str(result["selected_profile"]["id"])
    config = json.loads(BASE_CONFIG.read_text())
    optimizer = config["optimizer"]
    monthly = pd.read_parquet(run_root / "historical_test_monthly.parquet")
    positions_root = run_root / "historical_test" / profile / "months"

    maximum_candidate_long = -np.inf
    maximum_candidate_short = -np.inf
    maximum_inherited_increase = -np.inf
    maximum_reduction_only_sign_breach = -np.inf
    maximum_sic2_net = 0.0
    maximum_sic2_gross = 0.0
    maximum_control_weight_difference = 0.0
    exact_control_position_months = 0
    position_months = 0
    largest_inherited_short: dict[str, object] | None = None
    cap = float(optimizer["maximum_position_weight"])
    for path in sorted(positions_root.glob("*.parquet")):
        positions = pd.read_parquet(path)
        frozen_path = FROZEN_CONTROL / "variants" / FROZEN_VARIANT / "months" / path.name
        frozen = pd.read_parquet(frozen_path).loc[:, ["permno", "weight"]]
        comparison = frozen.merge(
            positions.loc[:, ["permno", "weight"]],
            on="permno",
            how="outer",
            suffixes=("_frozen", "_challenge"),
            validate="one_to_one",
        ).fillna(0.0)
        weight_difference = (
            comparison["weight_frozen"] - comparison["weight_challenge"]
        ).abs()
        month_maximum = float(weight_difference.max())
        maximum_control_weight_difference = max(
            maximum_control_weight_difference, month_maximum
        )
        exact_control_position_months += int(month_maximum == 0.0)
        position_months += 1
        long_candidate = positions["long_candidate"].astype(bool)
        short_candidate = positions["short_candidate"].astype(bool)
        reduction_only = (
            ~long_candidate
            & ~short_candidate
            & positions["previous_weight"].ne(0.0)
        )
        if long_candidate.any():
            maximum_candidate_long = max(
                maximum_candidate_long,
                float(positions.loc[long_candidate, "weight"].max() - cap),
            )
        if short_candidate.any():
            maximum_candidate_short = max(
                maximum_candidate_short,
                float(-positions.loc[short_candidate, "weight"].min() - cap),
            )
        if reduction_only.any():
            inherited = positions.loc[reduction_only]
            maximum_inherited_increase = max(
                maximum_inherited_increase,
                float(
                    (inherited["weight"].abs() - inherited["previous_weight"].abs()).max()
                ),
            )
            sign_breach = np.where(
                inherited["previous_weight"].to_numpy(float) > 0.0,
                -inherited["weight"].to_numpy(float),
                inherited["weight"].to_numpy(float),
            )
            maximum_reduction_only_sign_breach = max(
                maximum_reduction_only_sign_breach, float(sign_breach.max(initial=-np.inf))
            )
            short = inherited.loc[inherited["weight"].lt(0.0)]
            if not short.empty:
                row = short.loc[short["weight"].idxmin()]
                if largest_inherited_short is None or float(row["weight"]) < float(
                    largest_inherited_short["weight"]
                ):
                    largest_inherited_short = {
                        "holding_month": path.stem,
                        "permno": int(row["permno"]),
                        "weight": float(row["weight"]),
                        "previous_weight": float(row["previous_weight"]),
                    }
        active = positions.loc[positions["weight"].ne(0.0)]
        for _, group in active.groupby("sic2_industry_category", observed=True):
            weights = group["weight"].to_numpy(float)
            maximum_sic2_net = max(maximum_sic2_net, abs(float(weights.sum())))
            maximum_sic2_gross = max(maximum_sic2_gross, float(np.abs(weights).sum()))

    other_factor_columns = [
        column
        for column in monthly.columns
        if column.startswith("factor_exposure_") and column != "factor_exposure_mktrf"
    ]
    breaches = {
        "maximum_gross": float(monthly["gross"].max() - optimizer["maximum_gross"]),
        "minimum_long_gross": float(
            optimizer["minimum_long_gross"] - monthly["long_gross"].min()
        ),
        "minimum_short_gross": float(
            optimizer["minimum_short_gross"] - monthly["short_gross"].min()
        ),
        "minimum_net": float(optimizer["minimum_net"] - monthly["net"].min()),
        "maximum_net": float(monthly["net"].max() - optimizer["maximum_net"]),
        "candidate_position_long": float(maximum_candidate_long),
        "candidate_position_short": float(maximum_candidate_short),
        "reduction_only_magnitude_increase": float(maximum_inherited_increase),
        "reduction_only_sign_flip": float(maximum_reduction_only_sign_breach),
        "turnover": float(
            monthly["turnover"].max() - optimizer["maximum_one_way_turnover"]
        ),
        "adv": float(
            monthly["maximum_adv_participation"].max()
            - optimizer["maximum_one_way_adv_participation"]
        ),
        "market_beta": float(
            monthly["factor_exposure_mktrf"].abs().max()
            - optimizer["market_beta_absolute_limit"]
        ),
        "other_factor": float(
            monthly[other_factor_columns].abs().to_numpy().max()
            - optimizer["other_factor_absolute_limit"]
        ),
        "sic2_net": float(maximum_sic2_net - optimizer["sic2_industry_net_absolute_limit"]),
        "sic2_gross": float(maximum_sic2_gross - optimizer["sic2_industry_gross_limit"]),
        "forecast_volatility": float(
            monthly["forecast_volatility"].max()
            - optimizer["primary_forecast_volatility"]
        ),
    }
    tolerance = 5e-7
    return {
        "schema": "research_v6_1_final_optimizer_constraint_audit_addendum_v1",
        "source_result_sha256": _sha256(result_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "selected_profile": profile,
        "position_semantics": (
            "The 1% cap applies to active long/short candidates. A holding that is no "
            "longer a candidate is reduction-only: it may remain beyond 1% after return "
            "drift but cannot increase in absolute weight or change sign."
        ),
        "generic_audit_correction": (
            "The result.json position_short residual compared every inherited holding "
            "with the candidate cap and therefore reported a false breach."
        ),
        "largest_inherited_short": largest_inherited_short,
        "exact_control_reproduction": {
            "position_months": position_months,
            "exact_weight_months": exact_control_position_months,
            "maximum_absolute_weight_difference": maximum_control_weight_difference,
            "net_sharpe_difference": float(
                result["historical_test_summary"]["net_sharpe_primary"]
                - result["frozen_control_sharpe"]
            ),
        },
        "breach_residuals": breaches,
        "maximum_positive_breach": float(max(max(breaches.values()), 0.0)),
        "practical_tolerance": tolerance,
        "passes_practical_tolerance": bool(max(breaches.values()) <= tolerance),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    arguments = parser.parse_args()
    run_root = arguments.run_root.resolve()
    output = audit(run_root)
    _atomic_json(run_root / "constraint_audit_addendum.json", output)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
