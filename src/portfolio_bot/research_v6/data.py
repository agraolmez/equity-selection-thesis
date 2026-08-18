"""Lean point-in-time panel and target construction for V6 selectors."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXTERNAL_ROOT = Path("/Users/agraemirolmez/Desktop/Thesis")
FEATURE_ROOT = EXTERNAL_ROOT / (
    "processed/phase_03_features/phase03-feature-label-split-v1/"
    "feature_store_builds/phase03-features-fda21f0e51b8276c10246f9f"
)
VALIDATION_ROOT = EXTERNAL_ROOT / (
    "interim/phase_05_validation/stage11-validation-data-v1"
)
VALIDATION_EXECUTION_ROOT = EXTERNAL_ROOT / (
    "interim/phase_05_validation/stage_11_execution_v1"
)
FINAL_ROOT = EXTERNAL_ROOT / (
    "processed/phase_06_final_test/stage13-final-inputs-v2"
)
DECEMBER_2019_ROOT = EXTERNAL_ROOT / "processed/research_v6/december_2019_v1"
MODEL_FEATURE_ROOT = EXTERNAL_ROOT / "processed/research_v6/model_features_v2"

STRUCTURALLY_EMPTY_FEATURES = frozenset(
    {
        "business_segment_count",
        "geographic_segment_count",
        "business_segment_revenue_hhi",
        "business_segment_revenue_hhi_change_1y",
    }
)
FACTOR_NAMES = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
TARGET_COLUMNS = (
    "target_rank_capm",
    "target_rank_ff5_momentum",
    "target_rank_long_net",
    "target_rank_short_net",
)
REFERENCE_NAV_DOLLARS = 100_000_000.0
SIDE_GROSS_DOLLARS = 100_000_000.0
PRIMARY_IMPACT_COEFFICIENT = 0.10
COMMISSION_RATE = 0.0001
MAXIMUM_ADV_PARTICIPATION = 0.10


@dataclasses.dataclass(frozen=True, slots=True)
class V6Authorities:
    feature_dictionary: Path = FEATURE_ROOT / "feature_dictionary.parquet"
    category_vocabulary: Path = FEATURE_ROOT / "category_vocabulary.parquet"
    validation_features: Path = (
        VALIDATION_ROOT / "candidate_payloads/validation_feature_store_raw.parquet"
    )
    validation_label_audit: Path = (
        VALIDATION_EXECUTION_ROOT / "outcome_tables/validation_label_audit.parquet"
    )
    validation_factors: Path = (
        VALIDATION_EXECUTION_ROOT / "outcome_tables/daily_factors.parquet"
    )
    validation_reserve_factors: Path = (
        VALIDATION_EXECUTION_ROOT
        / "outcome_tables/outcome_reserve_daily_factors.parquet"
    )
    validation_costs: Path = (
        VALIDATION_ROOT / "candidate_payloads/validation_risk_cost_inputs.parquet"
    )
    final_features: Path = FINAL_ROOT / "primary_predictors.parquet"
    final_labels: Path = FINAL_ROOT / "primary_completed_labels.parquet"
    final_outcomes: Path = FINAL_ROOT / "primary_outcomes.parquet"
    final_accounting: Path = FINAL_ROOT / "primary_market_accounting.parquet"
    final_costs: Path = FINAL_ROOT / "primary_cost_inputs.parquet"
    december_2019_features: Path = DECEMBER_2019_ROOT / "predictors.parquet"
    december_2019_targets: Path = DECEMBER_2019_ROOT / "targets.parquet"
    december_2019_costs: Path = DECEMBER_2019_ROOT / "cost_inputs.parquet"


def load_feature_registry(
    authorities: V6Authorities = V6Authorities(),
) -> pd.DataFrame:
    registry = pd.read_parquet(authorities.feature_dictionary)
    required = {"feature_ordinal", "feature_name", "family", "kind"}
    if not required.issubset(registry.columns):
        raise ValueError("feature dictionary has the wrong schema")
    if registry["feature_name"].duplicated().any():
        raise ValueError("feature dictionary contains duplicate names")
    registry = registry.loc[
        ~registry["feature_name"].isin(STRUCTURALLY_EMPTY_FEATURES)
    ].copy()
    if len(registry) != 159:
        raise ValueError(f"V6 expected 159 usable predecessor features, found {len(registry)}")
    return registry.sort_values("feature_ordinal", kind="mergesort").reset_index(
        drop=True
    )


def load_model_feature_registry(
    *,
    feature_view: str = "primary_full",
    root: Path = MODEL_FEATURE_ROOT,
) -> pd.DataFrame:
    """Load the audited V6 registry for one prespecified information view."""

    if feature_view not in {"primary_full", "cam_free", "cam_only"}:
        raise ValueError(f"unknown V6 feature view: {feature_view}")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "BACKTEST_READY":
        raise ValueError("V6 model feature panel is not backtest-ready")
    registry = pd.read_parquet(root / "feature_registry.parquet")
    required = {"model_ordinal", "feature_name", "family", "kind", feature_view}
    if not required.issubset(registry.columns):
        raise ValueError("V6 model feature registry has the wrong schema")
    registry = registry.loc[registry[feature_view].fillna(False).astype(bool)].copy()
    if registry.empty or registry["feature_name"].duplicated().any():
        raise ValueError("V6 model feature registry identity is invalid")
    return registry.sort_values("model_ordinal", kind="mergesort").reset_index(drop=True)


def load_model_feature_panel(
    registry: pd.DataFrame,
    *,
    start_month: str = "2015-01",
    end_month: str = "2025-08",
    root: Path = MODEL_FEATURE_ROOT,
) -> pd.DataFrame:
    """Load monthly audited V6 inputs without consulting any outcomes."""

    features = registry["feature_name"].astype(str).tolist()
    months = pd.period_range(start_month, end_month, freq="M").astype(str).tolist()
    parts: list[pd.DataFrame] = []
    for month in months:
        path = root / "months" / f"{month}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing V6 model feature month: {month}")
        parts.append(pd.read_parquet(path, columns=["holding_month", "signal_date", "permno", "permco", *features]))
    panel = pd.concat(parts, ignore_index=True)
    panel["holding_month"] = panel["holding_month"].astype(str)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"], errors="raise")
    for column in ("permno", "permco"):
        panel[column] = pd.to_numeric(panel[column], errors="raise").astype("int64")
    if panel.duplicated(["holding_month", "permno"]).any():
        raise ValueError("V6 model feature panel contains duplicate stock-months")
    return panel.sort_values(["holding_month", "permno"], kind="mergesort").reset_index(drop=True)


def _holding_month_from_signal(signal: pd.Series) -> pd.Series:
    return (pd.to_datetime(signal, errors="raise").dt.to_period("M") + 1).astype(str)


def _normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["holding_month"] = result["holding_month"].astype(str)
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="raise")
    for column in ("permno", "permco"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    for column in ("long_eligible", "short_eligible"):
        result[column] = result[column].astype("boolean").fillna(False).astype(bool)
    if result.duplicated(["holding_month", "permno"]).any():
        raise ValueError("predictor panel contains duplicate stock-months")
    return result.sort_values(["holding_month", "permno"], kind="mergesort").reset_index(
        drop=True
    )


def load_predictor_panel(
    registry: pd.DataFrame,
    *,
    authorities: V6Authorities = V6Authorities(),
    start_month: str = "2015-01",
    end_month: str = "2025-08",
) -> pd.DataFrame:
    features = registry["feature_name"].astype(str).tolist()
    controls = [
        "signal_date",
        "permno",
        "permco",
        "long_eligible",
        "short_eligible",
        "primary_exchange",
        "share_type",
    ]
    validation = pd.read_parquet(
        authorities.validation_features, columns=[*controls, *features]
    )
    validation.insert(0, "holding_month", _holding_month_from_signal(validation["signal_date"]))
    final = pd.read_parquet(
        authorities.final_features,
        columns=["holding_month", *controls, *features],
    )
    parts = [validation, final]
    if authorities.december_2019_features.is_file():
        parts.append(
            pd.read_parquet(
                authorities.december_2019_features,
                columns=["holding_month", *controls, *features],
            )
        )
    panel = _normalize_panel(pd.concat(parts, ignore_index=True))
    panel = panel.loc[
        panel["holding_month"].between(start_month, end_month)
    ].reset_index(drop=True)
    panel["v6_december_2019_pending"] = False
    return panel


def _factor_sums(paths: Iterable[Path], *, date_column: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        if path.is_file():
            columns = ["holding_month", date_column, *FACTOR_NAMES]
            parts.append(pd.read_parquet(path, columns=columns))
    if not parts:
        raise FileNotFoundError("no factor source was available")
    frame = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["holding_month", date_column]
    )
    numeric = frame[list(FACTOR_NAMES)].apply(pd.to_numeric, errors="coerce")
    frame[list(FACTOR_NAMES)] = numeric
    return (
        frame.groupby("holding_month", observed=True, sort=True)[list(FACTOR_NAMES)]
        .sum(min_count=1)
        .add_prefix("factor_")
        .reset_index()
    )


def _final_factor_sums(path: Path) -> pd.DataFrame:
    columns = ["holding_month", "factor_observation_date", *FACTOR_NAMES]
    frame = pd.read_parquet(path, columns=columns).drop_duplicates(
        ["holding_month", "factor_observation_date"]
    )
    return (
        frame.groupby("holding_month", observed=True, sort=True)[list(FACTOR_NAMES)]
        .sum(min_count=1)
        .add_prefix("factor_")
        .reset_index()
    )


def _centered_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=np.float64)
    valid = np.isfinite(numeric.to_numpy(float))
    if valid.any():
        ranks = numeric.loc[valid].rank(method="average")
        result.loc[valid] = (ranks - 0.5) / int(valid.sum()) - 0.5
    return result


def _load_validation_targets(authorities: V6Authorities) -> pd.DataFrame:
    columns = [
        "holding_month",
        "signal_date",
        "permno",
        "permco",
        "long_eligible",
        "short_eligible",
        "month_label_available_date",
        "unranked_excess_return",
        "unranked_residual_return",
        "primary_label_status",
        *(f"beta_label_{name}" for name in FACTOR_NAMES),
    ]
    frame = pd.read_parquet(authorities.validation_label_audit, columns=columns)
    frame = frame.loc[frame["primary_label_status"].eq("PRESENT")].copy()
    frame = frame.rename(columns={"month_label_available_date": "label_available_date"})
    factors = _factor_sums(
        (authorities.validation_factors, authorities.validation_reserve_factors),
        date_column="observation_date",
    )
    return frame.merge(factors, on="holding_month", how="left", validate="many_to_one")


def _load_final_targets(
    panel: pd.DataFrame, authorities: V6Authorities
) -> pd.DataFrame:
    keys = ["holding_month", "signal_date", "permno", "permco", "long_eligible", "short_eligible"]
    beta_columns = [f"beta_label_{name}" for name in FACTOR_NAMES]
    base = panel.loc[panel["holding_month"].ge("2020-01"), [*keys, *beta_columns]].copy()
    labels = pd.read_parquet(
        authorities.final_labels,
        columns=[
            "holding_month",
            "signal_date",
            "permno",
            "outcome_availability_date",
            "unranked_residual_return",
            "primary_label_status",
        ],
    ).rename(columns={"outcome_availability_date": "label_available_date"})
    labels["holding_month"] = labels["holding_month"].astype(str)
    labels["signal_date"] = pd.to_datetime(labels["signal_date"], errors="raise")
    outcomes = pd.read_parquet(authorities.final_outcomes).loc[
        :, ["holding_month", "permno", "total_return"]
    ]
    outcomes["holding_month"] = outcomes["holding_month"].astype(str)
    frame = base.merge(
        labels, on=["holding_month", "signal_date", "permno"], how="inner", validate="one_to_one"
    ).merge(outcomes, on=["holding_month", "permno"], how="inner", validate="one_to_one")
    frame = frame.loc[frame["primary_label_status"].eq("PRESENT")].copy()
    frame["unranked_excess_return"] = pd.to_numeric(frame["total_return"], errors="coerce")
    return frame.merge(
        _final_factor_sums(authorities.final_accounting),
        on="holding_month",
        how="left",
        validate="many_to_one",
    )


def _load_costs(authorities: V6Authorities) -> pd.DataFrame:
    columns = [
        "signal_date",
        "permno",
        "permco",
        "full_spread_20d",
        "adv20_usd",
        "impact_volatility_daily",
        "primary_borrow_rate_annual",
        "new_long_trade_allowed",
        "new_short_trade_allowed",
    ]
    validation = pd.read_parquet(authorities.validation_costs, columns=columns)
    validation.insert(0, "holding_month", _holding_month_from_signal(validation["signal_date"]))
    final = pd.read_parquet(authorities.final_costs, columns=["holding_month", *columns])
    parts = [validation, final]
    if authorities.december_2019_costs.is_file():
        parts.append(
            pd.read_parquet(
                authorities.december_2019_costs,
                columns=["holding_month", *columns],
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    frame["holding_month"] = frame["holding_month"].astype(str)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise")
    if frame.duplicated(["holding_month", "permno"]).any():
        raise ValueError("cost inputs contain duplicate stock-months")
    return frame


def _attach_alpha_values(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric_columns = [
        "unranked_residual_return",
        *(f"beta_label_{name}" for name in FACTOR_NAMES),
        *(f"factor_{name}" for name in FACTOR_NAMES),
    ]
    numeric = result[numeric_columns].apply(pd.to_numeric, errors="coerce")
    residual = numeric["unranked_residual_return"].to_numpy(float)
    contribution = {
        name: numeric[f"beta_label_{name}"].to_numpy(float)
        * numeric[f"factor_{name}"].to_numpy(float)
        for name in FACTOR_NAMES
    }
    # The requested FF5+momentum alpha is the residual after all six factors,
    # including UMD.  Adding UMD back would instead create an FF5-only alpha.
    result["alpha_ff5_momentum"] = residual
    result["alpha_capm"] = residual + sum(
        contribution[name] for name in ("smb", "hml", "rmw", "cma", "umd")
    )
    return result


def _attach_net_values(frame: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    keys = ["holding_month", "signal_date", "permno", "permco"]
    outcomes = frame.copy()
    normalized_costs = costs.copy()
    for value in (outcomes, normalized_costs):
        value["holding_month"] = value["holding_month"].astype(str)
        value["signal_date"] = pd.to_datetime(value["signal_date"], errors="raise")
        value["permno"] = pd.to_numeric(value["permno"], errors="raise").astype("int64")
        value["permco"] = pd.to_numeric(value["permco"], errors="raise").astype("int64")
    overlapping_controls = sorted(
        (set(outcomes.columns) & set(normalized_costs.columns)) - set(keys)
    )
    if overlapping_controls:
        normalized_costs = normalized_costs.drop(columns=overlapping_controls)
    result = outcomes.merge(
        normalized_costs, on=keys, how="left", validate="one_to_one"
    )
    for column in (
        "full_spread_20d",
        "adv20_usd",
        "impact_volatility_daily",
        "primary_borrow_rate_annual",
        "unranked_excess_return",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    long_flag = result["long_eligible"].fillna(False).astype(bool)
    short_flag = result["short_eligible"].fillna(False).astype(bool)
    long_count = long_flag.groupby(result["holding_month"], observed=True).transform("sum")
    short_count = short_flag.groupby(result["holding_month"], observed=True).transform("sum")

    def side_cost(side_count: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tail_count = np.maximum(np.ceil(side_count.to_numpy(float) * 0.10), 1.0)
        position_weight = 1.0 / tail_count
        participation = (
            position_weight
            * SIDE_GROSS_DOLLARS
            / result["adv20_usd"].to_numpy(float)
        )
        per_leg_cost = (
            result["full_spread_20d"].to_numpy(float) / 2.0
            + COMMISSION_RATE
            + PRIMARY_IMPACT_COEFFICIENT
            * result["impact_volatility_daily"].to_numpy(float)
            * np.sqrt(participation)
        )
        valid = (
            np.isfinite(per_leg_cost)
            & (per_leg_cost >= 0.0)
            & np.isfinite(participation)
            & (participation >= 0.0)
            & (participation <= MAXIMUM_ADV_PARTICIPATION)
        )
        return participation, np.where(valid, 2.0 * per_leg_cost, np.nan), valid

    long_participation, long_roundtrip, long_cost_valid = side_cost(long_count)
    short_participation, short_roundtrip, short_cost_valid = side_cost(short_count)
    gross = result["unranked_excess_return"].to_numpy(float)
    long_allowed = result["long_eligible"].fillna(False).to_numpy(bool) & result[
        "new_long_trade_allowed"
    ].fillna(False).to_numpy(bool)
    short_allowed = result["short_eligible"].fillna(False).to_numpy(bool) & result[
        "new_short_trade_allowed"
    ].fillna(False).to_numpy(bool)
    borrow = result["primary_borrow_rate_annual"].to_numpy(float) / 12.0
    result["long_net_opportunity"] = np.where(
        long_allowed & long_cost_valid & np.isfinite(gross),
        gross - long_roundtrip,
        np.nan,
    )
    result["short_net_opportunity"] = np.where(
        short_allowed & short_cost_valid & np.isfinite(gross) & np.isfinite(borrow),
        -gross - short_roundtrip - borrow,
        np.nan,
    )
    result["long_shadow_adv_participation"] = long_participation
    result["long_shadow_roundtrip_cost_return"] = long_roundtrip
    result["short_shadow_adv_participation"] = short_participation
    result["short_shadow_roundtrip_cost_return"] = short_roundtrip
    return result


def build_target_panel(
    predictor_panel: pd.DataFrame,
    *,
    authorities: V6Authorities = V6Authorities(),
) -> pd.DataFrame:
    """Build four separate within-month ranks; higher always means more attractive."""

    validation = _load_validation_targets(authorities)
    final = _load_final_targets(predictor_panel, authorities)
    targets = _attach_alpha_values(pd.concat([validation, final], ignore_index=True))
    targets["holding_month"] = targets["holding_month"].astype(str)
    targets["signal_date"] = pd.to_datetime(targets["signal_date"], errors="raise")
    targets["label_available_date"] = pd.to_datetime(
        targets["label_available_date"], errors="coerce"
    )
    targets = _attach_net_values(targets, _load_costs(authorities))
    for source, output in (
        ("alpha_capm", "target_rank_capm"),
        ("alpha_ff5_momentum", "target_rank_ff5_momentum"),
        ("long_net_opportunity", "target_rank_long_net"),
        ("short_net_opportunity", "target_rank_short_net"),
    ):
        targets[output] = targets.groupby("holding_month", observed=True, sort=False)[
            source
        ].transform(_centered_rank)
    columns = [
        "holding_month",
        "signal_date",
        "permno",
        "permco",
        "label_available_date",
        "unranked_excess_return",
        "alpha_capm",
        "alpha_ff5_momentum",
        "long_net_opportunity",
        "short_net_opportunity",
        "long_shadow_adv_participation",
        "long_shadow_roundtrip_cost_return",
        "short_shadow_adv_participation",
        "short_shadow_roundtrip_cost_return",
        *TARGET_COLUMNS,
    ]
    output = targets.loc[:, columns]
    if authorities.december_2019_targets.is_file():
        december = pd.read_parquet(authorities.december_2019_targets, columns=columns)
        output = pd.concat([output, december], ignore_index=True)
    output = output.sort_values(
        ["holding_month", "permno"], kind="mergesort"
    )
    if output.duplicated(["holding_month", "permno"]).any():
        raise ValueError("V6 target panel contains duplicate stock-months")
    return output.reset_index(drop=True)


__all__ = [
    "FACTOR_NAMES",
    "REFERENCE_NAV_DOLLARS",
    "STRUCTURALLY_EMPTY_FEATURES",
    "TARGET_COLUMNS",
    "V6Authorities",
    "build_target_panel",
    "load_feature_registry",
    "load_model_feature_panel",
    "load_model_feature_registry",
    "load_predictor_panel",
]
