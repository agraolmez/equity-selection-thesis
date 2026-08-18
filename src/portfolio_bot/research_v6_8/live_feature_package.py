"""Assemble and encode one identity-bound V6.8 prospective feature month."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from portfolio_bot.research_v6.data import FEATURE_ROOT
from portfolio_bot.research_v6_8.production_data import build_feature_source_matrix
from portfolio_bot.research_v6_8.live_preprocessing import encode_live_panel


STRUCTURAL_GROUP = "pastor_stambaugh_structural_missing"
REQUIRED_SOURCE_GROUPS = frozenset(
    {
        "market_price_liquidity",
        "valuation_fundamentals",
        "institutional_ownership",
        "sec_filings_events",
        "security_identity",
        "optionmetrics",
        "factor_dependent",
        "lseg_analyst_cam",
        "alpaca_related_news",
    }
)


class LiveFeaturePackageError(RuntimeError):
    """The current raw or encoded feature package is structurally invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _receipt_strings(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {item for child in value.values() for item in _receipt_strings(child)}
    if isinstance(value, (list, tuple)):
        return {item for child in value for item in _receipt_strings(child)}
    return {str(value)} if isinstance(value, str) else set()


def _authenticated_input(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise LiveFeaturePackageError(f"{label} file is missing: {path}")
    receipt_path = path.parent / "receipt.json"
    if not receipt_path.is_file():
        raise LiveFeaturePackageError(f"{label} has no producer receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise LiveFeaturePackageError(f"{label} producer receipt did not pass")
    digest = _sha256(path)
    if digest not in _receipt_strings(receipt):
        raise LiveFeaturePackageError(f"{label} file hash is not bound by its receipt")
    return {
        "path": str(path),
        "sha256": digest,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
    }


def _stable_int(value: Any) -> int:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if text.isdigit() and int(text) <= np.iinfo(np.int64).max:
        return int(text)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & np.iinfo(np.int64).max


def _base_panel(universe: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    required = {
        "live_security_id",
        "holding_month",
        "signal_date",
        "TickerSymbol",
        "RIC",
        "PermID",
        "issuer_key",
    }
    missing = sorted(required - set(universe.columns))
    if missing:
        raise LiveFeaturePackageError(f"universe omits package keys: {missing}")
    if universe["live_security_id"].duplicated().any():
        raise LiveFeaturePackageError("universe duplicates live security identities")
    result = pd.DataFrame(
        {
            "live_security_id": universe["live_security_id"].astype(str),
            "holding_month": universe["holding_month"].astype(str),
            "signal_date": pd.to_datetime(universe["signal_date"], errors="raise"),
            "permno": universe["PermID"].map(_stable_int).astype("int64"),
            "permco": universe["issuer_key"].map(_stable_int).astype("int64"),
            "symbol": universe["TickerSymbol"].astype(str),
            "ric": universe["RIC"].astype(str),
            "issuer_key": universe["issuer_key"].astype(str),
        }
    )
    if result["permno"].duplicated().any():
        raise LiveFeaturePackageError("numeric live model security keys collide")
    for row in registry.itertuples(index=False):
        result[str(row.feature_name)] = None if str(row.kind).startswith("categorical") else np.nan
    return result


def _merge_group(
    panel: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    feature_names: set[str],
    group: str,
) -> tuple[pd.DataFrame, set[str]]:
    if "live_security_id" not in frame or frame["live_security_id"].duplicated().any():
        raise LiveFeaturePackageError(f"{group} has invalid live-security grain")
    source = frame.copy()
    source["live_security_id"] = source["live_security_id"].astype(str)
    columns = sorted(feature_names.intersection(source.columns))
    if not columns:
        raise LiveFeaturePackageError(f"{group} supplies no registered features")
    indexed = source.set_index("live_security_id")
    result = panel.copy()
    for feature in columns:
        result[feature] = result["live_security_id"].map(indexed[feature])
    # Operational controls travel with the market source but never enter the model.
    if group == "market_price_liquidity":
        for control in (
            "long_eligible",
            "issue_market_cap_usd",
            "signal_price_usd",
            "median_dollar_volume_60d",
            "latest_market_session",
        ):
            if control in indexed:
                result[control] = result["live_security_id"].map(indexed[control])
    return result, set(columns)


def _derive_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    momentum_6 = pd.to_numeric(result["momentum_6_1"], errors="coerce").to_numpy(float)
    momentum_12 = pd.to_numeric(result["momentum_12_1"], errors="coerce").to_numpy(float)
    older = np.full(len(result), np.nan)
    valid = (
        np.isfinite(momentum_6)
        & np.isfinite(momentum_12)
        & (1.0 + momentum_6 > 0.0)
        & (1.0 + momentum_12 > 0.0)
    )
    older[valid] = (1.0 + momentum_12[valid]) / (1.0 + momentum_6[valid]) - 1.0
    result["momentum_acceleration_6m"] = momentum_6 - older

    log_cap = pd.to_numeric(result["log_market_cap"], errors="coerce").to_numpy(float)
    cap = np.full(len(result), np.nan)
    finite_cap = np.isfinite(log_cap)
    cap[finite_cap] = np.exp(log_cap[finite_cap])
    industry = result["sic2_industry_category"].astype("string")
    contributor = (
        np.isfinite(momentum_12)
        & np.isfinite(cap)
        & (cap > 0.0)
        & industry.notna().to_numpy()
        & industry.ne("UNKNOWN_INDUSTRY").fillna(False).to_numpy()
    )
    work = pd.DataFrame(
        {
            "industry": industry,
            "weighted": np.where(contributor, momentum_12 * cap, 0.0),
            "weight": np.where(contributor, cap, 0.0),
            "contributor": contributor.astype(int),
        }
    )
    grouped = work.groupby("industry", observed=True, sort=False, dropna=False)
    numerator = grouped["weighted"].transform("sum").to_numpy(float)
    denominator = grouped["weight"].transform("sum").to_numpy(float)
    count = grouped["contributor"].transform("sum").to_numpy(int)
    industry_momentum = np.full(len(result), np.nan)
    valid_group = (denominator > 0.0) & (count >= 2)
    industry_momentum[valid_group] = numerator[valid_group] / denominator[valid_group]
    result["industry_momentum_12_1_vw"] = industry_momentum
    return result


def assemble_live_feature_panel(
    *,
    universe: pd.DataFrame,
    registry: pd.DataFrame,
    source_matrix: pd.DataFrame,
    group_frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = _base_panel(universe, registry)
    populated: dict[str, str] = {}
    for group, frame in group_frames.items():
        expected = set(
            source_matrix.loc[
                source_matrix["production_group"].eq(group), "feature_name"
            ].astype(str)
        )
        if not expected:
            raise LiveFeaturePackageError(f"unknown production feature group: {group}")
        overlap = sorted(set(populated).intersection(expected))
        if overlap:
            raise LiveFeaturePackageError(f"feature supplied by multiple groups: {overlap}")
        panel, supplied = _merge_group(
            panel, frame, feature_names=expected, group=group
        )
        populated.update({feature: group for feature in supplied})
    panel = _derive_cross_sectional_features(panel)
    if "long_eligible" not in panel:
        panel["long_eligible"] = False
    panel["long_eligible"] = panel["long_eligible"].fillna(False).astype(bool)
    missing_groups = sorted(REQUIRED_SOURCE_GROUPS - set(group_frames))
    coverage = []
    for row in registry.itertuples(index=False):
        feature = str(row.feature_name)
        series = panel[feature]
        if str(row.kind).startswith("categorical"):
            observed = series.notna() & series.astype("string").str.len().gt(0)
        else:
            observed = pd.to_numeric(series, errors="coerce").notna()
        coverage.append(
            {
                "feature_name": feature,
                "model_ordinal": int(row.model_ordinal),
                "production_group": source_matrix.loc[
                    source_matrix["feature_name"].eq(feature), "production_group"
                ].iloc[0],
                "observed_share": float(observed.mean()),
            }
        )
    audit = {
        "status": "PASS" if not missing_groups else "BLOCKED_MISSING_SOURCE_GROUPS",
        "rows": int(len(panel)),
        "eligible_rows": int(panel["long_eligible"].sum()),
        "missing_source_groups": missing_groups,
        "supplied_source_groups": sorted(group_frames),
        "coverage": coverage,
    }
    return panel, audit


def materialize_live_feature_package(
    *,
    contract_path: Path,
    universe_path: Path,
    group_paths: Mapping[str, Path],
    output: Path,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    registry_path = Path(contract["frozen_feature_registry"]["path"])
    registry = pd.read_parquet(registry_path)
    source_matrix = build_feature_source_matrix(contract, registry)
    authenticated_inputs = {
        "universe": _authenticated_input(universe_path, label="live universe")
    }
    for group, path in group_paths.items():
        authenticated_inputs[group] = _authenticated_input(
            path, label=f"feature group {group}"
        )
    universe = pd.read_parquet(universe_path)
    frames = {group: pd.read_parquet(path) for group, path in group_paths.items()}
    panel, audit = assemble_live_feature_panel(
        universe=universe,
        registry=registry,
        source_matrix=source_matrix,
        group_frames=frames,
    )
    raw_path = output / "raw_features.parquet"
    coverage_path = output / "feature_coverage.parquet"
    _atomic_parquet(raw_path, panel)
    _atomic_parquet(coverage_path, pd.DataFrame(audit.pop("coverage")))
    encoded_path: Path | None = None
    encoded_rows_path: Path | None = None
    encoded_features = 0
    eligible = panel.loc[panel["long_eligible"]].copy()
    if not eligible.empty:
        encoded = encode_live_panel(
            eligible,
            registry,
            category_vocabulary_path=FEATURE_ROOT / "category_vocabulary.parquet",
            historical_model_root=registry_path.parent,
        )
        project_root = Path(__file__).resolve().parents[3]
        expected_names = json.loads(
            (
                project_root
                / "artifacts/research_v6/selector_step_01/841b5f23ca9d-01c2e95ca6b9-dab475636cfc/"
                "input_cache/feature_metadata.json"
            ).read_text(encoding="utf-8")
        )["feature_names"]
        if list(encoded.feature_names) != expected_names:
            raise LiveFeaturePackageError("live encoded feature identities differ from frozen V6")
        encoded_path = output / "encoded_values.npy"
        encoded_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = encoded_path.with_suffix(".npy.tmp")
        with temporary.open("wb") as destination:
            np.save(destination, encoded.values)
        os.replace(temporary, encoded_path)
        encoded_rows_path = output / "encoded_rows.parquet"
        rows = eligible[
            ["live_security_id", "symbol", "ric", "issuer_key", "holding_month", "signal_date", "permno", "permco"]
        ].sort_values(["holding_month", "permno"], kind="mergesort")
        _atomic_parquet(encoded_rows_path, rows)
        encoded_features = encoded.values.shape[1]
    if audit["status"] == "PASS" and eligible.empty:
        audit["status"] = "BLOCKED_NO_ELIGIBLE_SECURITIES"
    receipt = {
        "schema": "research_v6_8_live_feature_package_receipt_v1",
        **audit,
        "raw_feature_count": int(len(registry)),
        "encoded_feature_count": int(encoded_features),
        "raw_features_sha256": _sha256(raw_path),
        "feature_coverage_sha256": _sha256(coverage_path),
        "encoded_values_sha256": None if encoded_path is None else _sha256(encoded_path),
        "encoded_rows_sha256": None if encoded_rows_path is None else _sha256(encoded_rows_path),
        "contract_sha256": _sha256(contract_path),
        "universe_sha256": _sha256(universe_path),
        "authenticated_inputs": authenticated_inputs,
    }
    _atomic_json(output / "receipt.json", receipt)
    return receipt


__all__ = [
    "LiveFeaturePackageError",
    "REQUIRED_SOURCE_GROUPS",
    "assemble_live_feature_panel",
    "materialize_live_feature_package",
]
