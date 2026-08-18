#!/usr/bin/env python3
"""Verify that the prospective encoder reproduces the frozen last history month."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_bot.research_v6.data import FEATURE_ROOT, load_model_feature_registry
from portfolio_bot.research_v6_8.live_preprocessing import encode_live_panel


def main() -> int:
    model_root = Path("/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/model_features_v2")
    cache = ROOT / "artifacts/research_v6/selector_step_01/841b5f23ca9d-01c2e95ca6b9-dab475636cfc/input_cache"
    month = "2025-08"
    registry = load_model_feature_registry(root=model_root)
    panel = pd.read_parquet(model_root / "months" / f"{month}.parquet")
    live = encode_live_panel(
        panel,
        registry,
        category_vocabulary_path=FEATURE_ROOT / "category_vocabulary.parquet",
        historical_model_root=model_root,
    )
    rows = pd.read_parquet(cache / "rows.parquet", columns=["holding_month", "permno"])
    indices = np.flatnonzero(rows["holding_month"].astype(str).eq(month).to_numpy())
    frozen = np.load(cache / "encoded_values.npy", mmap_mode="r")[indices]
    metadata = json.loads((cache / "feature_metadata.json").read_text(encoding="utf-8"))
    receipt = {
        "schema": "research_v6_8_live_encoder_parity_v1",
        "status": "PASS" if np.array_equal(live.values, frozen) else "FAIL",
        "month": month,
        "rows": int(len(live.values)),
        "features": int(live.values.shape[1]),
        "feature_names_exact": list(live.feature_names) == metadata["feature_names"],
        "row_keys_exact": live.keys["permno"].tolist() == rows.iloc[indices]["permno"].tolist(),
        "maximum_absolute_difference": float(np.max(np.abs(live.values - frozen))),
        "exact_matrix_match": bool(np.array_equal(live.values, frozen)),
    }
    output = ROOT / "artifacts/research_v6_8/live_encoder_parity/20260817/receipt.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
