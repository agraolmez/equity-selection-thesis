#!/usr/bin/env python3
"""Publish the audited V6 model-ready feature panel and exact registry.

This is an outcome-blind data build.  It keeps the complete eligible universe,
preserves genuine missing values for the common model encoder, replaces LSEG
currency-bearing levels with the separately authenticated USD lineage, converts
vendor dates into causal information ages, and joins the point-in-time related
news block.  It does not fit, tune, evaluate, or backtest a selector.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DATABASE = Path("/Users/agraemirolmez/Desktop/Thesis")
PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = DATABASE / "processed/research_v6/model_features_v1"
PREDECESSOR_ROOT = DATABASE / (
    "processed/phase_03_features/phase03-feature-label-split-v1/"
    "feature_store_builds/phase03-features-fda21f0e51b8276c10246f9f"
)
VALIDATION_FEATURES = DATABASE / (
    "interim/phase_05_validation/stage11-validation-data-v1/"
    "candidate_payloads/validation_feature_store_raw.parquet"
)
FINAL_FEATURES = DATABASE / (
    "processed/phase_06_final_test/stage13-final-inputs-v2/primary_predictors.parquet"
)
DECEMBER_FEATURES = DATABASE / "processed/research_v6/december_2019_v1/predictors.parquet"
UNIVERSE = DATABASE / "raw/research_v6/lseg/input_universe.parquet"
LSEG = DATABASE / "raw/research_v6/lseg"
LSEG_USD = DATABASE / "raw/research_v6/lseg_usd_v1"
NEWS = DATABASE / "processed/research_v6/news_features_v2"

KEYS = ["holding_month", "signal_date", "permno", "permco"]
STRUCTURALLY_EMPTY = {
    "business_segment_count",
    "geographic_segment_count",
    "business_segment_revenue_hhi",
    "business_segment_revenue_hhi_change_1y",
}
DIAGNOSTIC_ONLY = {"ff12_sector_category"}
DATE_COLUMNS = {
    "fy1__tr_epsmean_period_fy1_calcdate": "lseg_fy1_information_age_days",
    "fy2__tr_epsmean_period_fy2_calcdate": "lseg_fy2_information_age_days",
    "fq1__tr_epsmean_period_fq1_calcdate": "lseg_fq1_information_age_days",
    "signals__tr_recmean_date": "lseg_recommendation_information_age_days",
    "signals__tr_pricetargetmean_date": "lseg_price_target_information_age_days",
    "signals__tr_combinedalphacountryrank_date": "lseg_cam_information_age_days",
}
CAM_RAW = "signals__tr_combinedalphacountryrank"
INT64_NULL = np.iinfo(np.int64).min


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _holding_month(signal: pd.Series) -> pd.Series:
    return (pd.to_datetime(signal, errors="raise").dt.to_period("M") + 1).astype(str)


def _predecessor_registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_parquet(PREDECESSOR_ROOT / "feature_dictionary.parquet")
    source["source"] = "predecessor"
    source["model_status"] = "INCLUDED"
    source["exclusion_reason"] = pd.NA
    source.loc[source["feature_name"].isin(STRUCTURALLY_EMPTY), "model_status"] = "EXCLUDED"
    source.loc[source["feature_name"].isin(STRUCTURALLY_EMPTY), "exclusion_reason"] = (
        "source table has zero observations; not imputed"
    )
    source.loc[source["feature_name"].isin(DIAGNOSTIC_ONLY), "model_status"] = "DIAGNOSTIC_ONLY"
    source.loc[source["feature_name"].isin(DIAGNOSTIC_ONLY), "exclusion_reason"] = (
        "FF12 is diagnostic-only under the research governance contract"
    )
    included = source.loc[source["model_status"].eq("INCLUDED")].copy()
    if len(included) != 158:
        raise RuntimeError(f"expected 158 included predecessor features, got {len(included)}")
    return included, source


def _load_predecessor_month(month: str, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = ["signal_date", "permno", "permco", "ff12_sector_category"]
    if month <= "2019-11":
        source = VALIDATION_FEATURES
        frame = pd.read_parquet(source, columns=[*controls, *features])
        frame.insert(0, "holding_month", _holding_month(frame["signal_date"]))
        frame = frame.loc[frame["holding_month"].eq(month)]
    elif month == "2019-12":
        if not DECEMBER_FEATURES.is_file():
            raise FileNotFoundError("exact December 2019 predecessor predictors are not complete")
        frame = pd.read_parquet(
            DECEMBER_FEATURES,
            columns=["holding_month", *controls, *features],
        )
    else:
        frame = pd.read_parquet(
            FINAL_FEATURES,
            filters=[("holding_month", "=", month)],
            columns=["holding_month", *controls, *features],
        )
    frame["holding_month"] = frame["holding_month"].astype(str)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    diagnostic = frame[KEYS + ["ff12_sector_category"]].copy()
    return frame.drop(columns="ff12_sector_category"), diagnostic


def _safe_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"^usd__tr_", "", value)
    value = re.sub(r"^(fy1|fy2|fq1)__tr_", r"\1_", value)
    value = re.sub(r"^signals__tr_", "", value)
    value = re.sub(r"_period_(fy1|fy2|fq1)$", r"_\1", value)
    return "lseg_" + re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _lseg_registry(sample_raw: pd.DataFrame, sample_usd: pd.DataFrame) -> pd.DataFrame:
    control = set(pd.read_parquet(UNIVERSE).columns)
    raw_features = [column for column in sample_raw if column not in control]
    rows: list[dict[str, Any]] = []
    usd_semantics = {column.removeprefix("usd__tr_"): column for column in sample_usd if column.startswith("usd__")}
    for column in raw_features:
        if column in DATE_COLUMNS:
            name = DATE_COLUMNS[column]
            source_column = column
            transformation = "signal_date_minus_vendor_date_days"
            unit = "days"
        else:
            semantic = re.sub(r"^(fy1|fy2|fq1)__tr_", "", column)
            semantic = re.sub(r"^signals__tr_", "", semantic)
            usd_column = usd_semantics.get(semantic)
            if column == CAM_RAW:
                name = "lseg_cam_country_rank"
                source_column = column
                transformation = "int64_null_sentinel_to_missing"
                unit = "vendor_country_rank"
            elif usd_column is not None:
                name = _safe_name(usd_column)
                source_column = usd_column
                transformation = "LSEG_Curn_USD_replacement"
                unit = "USD_or_USD_per_share"
            else:
                name = _safe_name(column)
                source_column = column
                transformation = "int64_null_sentinel_to_missing"
                unit = "vendor_native_noncurrency"
        rows.append(
            {
                "feature_name": name,
                "family": "lseg_analyst_expectations",
                "kind": "continuous_firm",
                "source": "lseg_usd_v1" if source_column.startswith("usd__") else "lseg",
                "source_column": source_column,
                "unit": unit,
                "transformation": transformation,
                "model_status": "INCLUDED",
                "exclusion_reason": pd.NA,
                "primary_full": True,
                "cam_free": "cam" not in name,
                "cam_only": "cam" in name,
            }
        )
    for lag in (1, 3, 6):
        rows.append(
            {
                "feature_name": f"lseg_cam_rank_change_{lag}m",
                "family": "lseg_analyst_expectations",
                "kind": "continuous_firm",
                "source": "lseg",
                "source_column": CAM_RAW,
                "unit": "vendor_rank_change",
                "transformation": f"current_minus_{lag}m_lag_same_permno",
                "model_status": "INCLUDED",
                "exclusion_reason": pd.NA,
                "primary_full": True,
                "cam_free": False,
                "cam_only": True,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 89 or result["feature_name"].duplicated().any():
        raise RuntimeError("LSEG model registry identity differs")
    return result


def _news_registry(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in sample.columns:
        if column in KEYS:
            continue
        binary = column in {"news_source_available", "news_related"}
        rows.append(
            {
                "feature_name": column,
                "family": "alpaca_related_news",
                "kind": "binary_firm" if binary else "continuous_firm",
                "source": "alpaca_news_features_v2",
                "source_column": column,
                "unit": "binary" if binary else "article_aggregate",
                "transformation": "point_in_time_related_news_aggregation",
                "model_status": "INCLUDED",
                "exclusion_reason": pd.NA,
                "primary_full": True,
                "cam_free": True,
                "cam_only": False,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 56 or result["feature_name"].duplicated().any():
        raise RuntimeError("news model registry identity differs")
    return result


def _assemble_lseg(month: str, registry: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(LSEG / "months" / f"{month}.parquet")
    usd = pd.read_parquet(LSEG_USD / "months" / f"{month}.parquet")
    raw["signal_date"] = pd.to_datetime(raw["signal_date"])
    usd["signal_date"] = pd.to_datetime(usd["signal_date"])
    if raw.duplicated(KEYS).any() or usd.duplicated(KEYS).any():
        raise RuntimeError(f"{month}: duplicate LSEG keys")
    if set(map(tuple, raw[KEYS].itertuples(index=False, name=None))) != set(
        map(tuple, usd[KEYS].itertuples(index=False, name=None))
    ):
        raise RuntimeError(f"{month}: USD repair keys differ")
    joined = raw.merge(usd, on=[*KEYS, "cusip9"], how="left", validate="one_to_one")
    output = joined[KEYS].copy()
    for row in registry.itertuples(index=False):
        if row.feature_name.startswith("lseg_cam_rank_change_"):
            continue
        values = joined[row.source_column]
        if row.transformation == "signal_date_minus_vendor_date_days":
            dates = pd.to_datetime(values, errors="coerce")
            age = (joined["signal_date"] - dates).dt.total_seconds() / 86_400.0
            if age.dropna().lt(0).any():
                raise RuntimeError(f"{month}: future LSEG date escaped")
            output[row.feature_name] = age.astype("float64")
        else:
            numeric = pd.to_numeric(values, errors="coerce").astype("float64")
            numeric = numeric.mask(numeric.eq(float(INT64_NULL)))
            output[row.feature_name] = numeric
    cam_name = "lseg_cam_country_rank"
    current = output[["permno", cam_name]].copy()
    for lag in (1, 3, 6):
        lag_period = str(pd.Period(month, freq="M") - lag)
        prior = history.loc[history["holding_month"].eq(lag_period), ["permno", "cam_rank"]]
        mapped = current[["permno"]].merge(prior, on="permno", how="left", validate="one_to_one")
        output[f"lseg_cam_rank_change_{lag}m"] = current[cam_name].to_numpy() - mapped["cam_rank"].to_numpy()
    next_history = pd.DataFrame(
        {"holding_month": month, "permno": output["permno"], "cam_rank": output[cam_name]}
    )
    return output, next_history


def _registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    predecessor, predecessor_audit = _predecessor_registry()
    raw_sample = pd.read_parquet(LSEG / "months/2015-01.parquet")
    usd_sample = pd.read_parquet(LSEG_USD / "months/2015-01.parquet")
    news_sample = pd.read_parquet(NEWS / "months/2015-01.parquet")
    lseg = _lseg_registry(raw_sample, usd_sample)
    news = _news_registry(news_sample)
    base_columns = [
        "feature_name", "family", "kind", "source", "source_column", "unit",
        "transformation", "model_status", "exclusion_reason", "primary_full",
        "cam_free", "cam_only",
    ]
    p = predecessor.copy()
    p["source_column"] = p["feature_name"]
    p["transformation"] = "predecessor_outcome_blind_encoder"
    p["primary_full"] = True
    p["cam_free"] = True
    p["cam_only"] = False
    p = p.reindex(columns=base_columns)
    registry = pd.concat([p, lseg[base_columns], news[base_columns]], ignore_index=True)
    registry.insert(0, "model_ordinal", np.arange(1, len(registry) + 1, dtype=int))
    if len(registry) != 303 or registry["feature_name"].duplicated().any():
        raise RuntimeError("unified model registry identity differs")

    audit = predecessor_audit.copy()
    audit["source_column"] = audit["feature_name"]
    audit["transformation"] = "predecessor_outcome_blind_encoder"
    audit["primary_full"] = audit["model_status"].eq("INCLUDED")
    audit["cam_free"] = audit["primary_full"]
    audit["cam_only"] = False
    audit = audit.reindex(columns=base_columns)
    audit = pd.concat([audit, lseg[base_columns], news[base_columns]], ignore_index=True)
    return registry, audit


def main() -> int:
    registry, audit_registry = _registry()
    months = sorted(pd.read_parquet(UNIVERSE, columns=["holding_month"])["holding_month"].astype(str).unique())
    source_identity = {
        "script_sha256": _sha256(Path(__file__).resolve()),
        "universe_sha256": _sha256(UNIVERSE),
        "predecessor_dictionary_sha256": _sha256(PREDECESSOR_ROOT / "feature_dictionary.parquet"),
        "validation_features_sha256": _sha256(VALIDATION_FEATURES),
        "final_features_sha256": _sha256(FINAL_FEATURES),
        "december_features_sha256": _sha256(DECEMBER_FEATURES),
        "lseg_contract_sha256": json.loads((LSEG / "contract.json").read_text())["contract_sha256"],
        "lseg_usd_contract_sha256": json.loads((LSEG_USD / "contract.json").read_text())["contract_sha256"],
        "news_manifest_sha256": _sha256(NEWS / "manifest.json"),
    }
    build_identity = hashlib.sha256(
        json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_parquet(registry, OUTPUT / "feature_registry.parquet")
    _atomic_parquet(audit_registry, OUTPUT / "feature_audit_registry.parquet")
    registry.to_csv(OUTPUT / "feature_registry.csv", index=False)

    predecessor_names = registry.loc[registry["source"].eq("predecessor"), "feature_name"].tolist()
    lseg_registry = registry.loc[registry["family"].eq("lseg_analyst_expectations")]
    history = pd.DataFrame(columns=["holding_month", "permno", "cam_rank"])
    coverage_parts: list[pd.DataFrame] = []
    rows_total = 0
    for ordinal, month in enumerate(months, start=1):
        manifest_path = OUTPUT / "manifests" / f"{month}.json"
        destination = OUTPUT / "months" / f"{month}.parquet"
        diagnostic_path = OUTPUT / "diagnostics" / f"{month}.parquet"
        predecessor, diagnostic = _load_predecessor_month(month, predecessor_names)
        lseg, history_row = _assemble_lseg(month, lseg_registry, history)
        history = pd.concat([history, history_row], ignore_index=True)
        news = pd.read_parquet(NEWS / "months" / f"{month}.parquet")
        news["signal_date"] = pd.to_datetime(news["signal_date"])
        universe = pd.read_parquet(UNIVERSE, filters=[("holding_month", "=", month)], columns=KEYS)
        universe["signal_date"] = pd.to_datetime(universe["signal_date"])
        result = universe.merge(predecessor, on=KEYS, how="left", validate="one_to_one")
        result = result.merge(lseg, on=KEYS, how="left", validate="one_to_one")
        result = result.merge(news, on=KEYS, how="left", validate="one_to_one")
        expected = registry["feature_name"].tolist()
        if len(result) != len(universe) or result[KEYS].isna().any().any():
            raise RuntimeError(f"{month}: unified feature keys differ")
        if set(expected).difference(result.columns):
            raise RuntimeError(f"{month}: unified features are missing")
        result = result[[*KEYS, *expected]].sort_values("permno", kind="mergesort")
        numeric = result.select_dtypes(include=[np.number])
        if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            raise RuntimeError(f"{month}: nonfinite infinity escaped")
        receipt_reusable = False
        if destination.is_file() and manifest_path.is_file() and diagnostic_path.is_file():
            receipt = json.loads(manifest_path.read_text())
            receipt_reusable = (
                receipt.get("build_identity") == build_identity
                and receipt.get("sha256") == _sha256(destination)
                and receipt.get("rows") == len(result)
            )
        if not receipt_reusable:
            _atomic_parquet(result, destination)
            diagnostic = universe.merge(diagnostic, on=KEYS, how="left", validate="one_to_one")
            _atomic_parquet(diagnostic, diagnostic_path)
            _atomic_json(
                manifest_path,
                {
                    "status": "COMPLETE",
                    "holding_month": month,
                    "build_identity": build_identity,
                    "rows": len(result),
                    "features": len(expected),
                    "sha256": _sha256(destination),
                    "diagnostic_sha256": _sha256(diagnostic_path),
                },
            )
        coverage = result[expected].notna().mean().rename("coverage").reset_index()
        coverage = coverage.rename(columns={"index": "feature_name"})
        coverage.insert(0, "holding_month", month)
        coverage_parts.append(coverage)
        rows_total += len(result)
        print(f"{ordinal}/{len(months)} {month}: {len(result):,} rows", flush=True)

    coverage = pd.concat(coverage_parts, ignore_index=True).merge(
        registry[["feature_name", "family"]], on="feature_name", how="left", validate="many_to_one"
    )
    coverage["year"] = coverage["holding_month"].str[:4]
    _atomic_parquet(coverage, OUTPUT / "coverage_by_month_feature.parquet")
    year_family = coverage.groupby(["year", "family"], observed=True)["coverage"].agg(
        mean_coverage="mean", minimum_feature_coverage="min", maximum_feature_coverage="max"
    ).reset_index()
    _atomic_parquet(year_family, OUTPUT / "coverage_by_year_family.parquet")
    _atomic_json(
        OUTPUT / "manifest.json",
        {
            "schema": "research_v6_model_features_v1",
            "status": "BACKTEST_READY",
            "build_identity": build_identity,
            "first_holding_month": months[0],
            "last_holding_month": months[-1],
            "months": len(months),
            "rows": rows_total,
            "model_features_primary_full": int(registry["primary_full"].sum()),
            "model_features_cam_free": int(registry["cam_free"].sum()),
            "model_features_cam_only": int(registry["cam_only"].sum()),
            "predecessor_features": int(registry["source"].eq("predecessor").sum()),
            "lseg_features": int(registry["family"].eq("lseg_analyst_expectations").sum()),
            "news_features": int(registry["family"].eq("alpaca_related_news").sum()),
            "complete_case_filter_used": False,
            "missing_policy": "retain_full_universe_then_common_encoder_zero_plus_missing_indicator",
            "source_identity": source_identity,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
