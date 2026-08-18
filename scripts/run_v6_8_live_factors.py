#!/usr/bin/env python3
"""Snapshot official factors and optionally materialize live V6.8 factor features."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from portfolio_bot.research_v6_8.live_factor_features import (
    materialize_live_factor_features,
    snapshot_official_factors,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research_v6_8/live_factor_policy.json"),
    )
    parser.add_argument("--snapshot-output", type=Path, required=True)
    parser.add_argument("--signal-date", type=date.fromisoformat)
    parser.add_argument("--market-history", type=Path)
    parser.add_argument("--market-features", type=Path)
    parser.add_argument("--feature-output", type=Path)
    args = parser.parse_args()
    snapshot = snapshot_official_factors(
        config_path=args.config, output=args.snapshot_output
    )
    payload: dict[str, object] = {"snapshot": snapshot}
    feature_inputs = [
        args.signal_date,
        args.market_history,
        args.market_features,
        args.feature_output,
    ]
    if any(value is not None for value in feature_inputs):
        if not all(value is not None for value in feature_inputs):
            parser.error("all feature-materialization arguments must be supplied together")
        payload["features"] = materialize_live_factor_features(
            config_path=args.config,
            factor_snapshot=args.snapshot_output,
            market_history_path=args.market_history,
            market_features_path=args.market_features,
            output=args.feature_output,
            signal_date=args.signal_date,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
