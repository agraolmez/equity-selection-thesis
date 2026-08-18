#!/usr/bin/env python3
"""Run the checkpointed LSEG production capability and replay probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_bot.research_v6_8.lseg_production_probe import (
    run_lseg_production_probe,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research_v6_8/lseg_production_probe/20260816",
    )
    parser.add_argument(
        "--archive-month",
        type=Path,
        default=Path(
            "/Users/agraemirolmez/Desktop/Thesis/raw/research_v6/lseg/months/2025-07.parquet"
        ),
    )
    parser.add_argument(
        "--archive-contract",
        type=Path,
        default=Path(
            "/Users/agraemirolmez/Desktop/Thesis/raw/research_v6/lseg/contract.json"
        ),
    )
    parser.add_argument(
        "--model-feature-month",
        type=Path,
        default=Path(
            "/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/model_features_v2/months/2025-07.parquet"
        ),
    )
    parser.add_argument("--sample-size", type=int, default=12)
    args = parser.parse_args()
    receipt = run_lseg_production_probe(
        output=args.output,
        archive_month=args.archive_month,
        archive_contract=args.archive_contract,
        model_feature_month=args.model_feature_month,
        sample_size=args.sample_size,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
