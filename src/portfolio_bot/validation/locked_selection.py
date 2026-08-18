"""Locked Stage 11 family and ensemble selection primitives.

The functions in this module are deliberately table-pure.  They do not read
validation data, run a portfolio, or calculate a performance statistic.  They
accept only the prespecified monthly rank-IC and signal-turnover fields so a
Sharpe ratio, P&L series, or allocator result cannot accidentally become a
selection input.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from portfolio_bot.selectors.metrics import (
    BOOTSTRAP_BLOCK_MONTHS,
    BOOTSTRAP_MASTER_SEED,
    BOOTSTRAP_REPLICATES,
    bootstrap_equal_month_mean_se,
    centered_average_rank,
    make_moving_block_bootstrap_plan,
)


TIE_TOLERANCE = 1e-12
VALIDATION_MONTHS = tuple(pd.period_range("2015-01", "2019-11", freq="M"))
VALIDATION_YEARS = (2015, 2016, 2017, 2018, 2019)
VALIDATION_STOCHASTIC_SEEDS = (11, 23, 47, 71, 101)

LOCKED_STAGE9_SELECTOR_BUILD_ID = "phase04-selector-results-09a59d61ed58e77d80448a97"
LOCKED_STAGE9_SHORTLIST_SHA256 = (
    "8136101bed9a66cc97faea66afbf30d6dafcf3b38a0d740c5ffc931862912e64"
)

LOCKED_SHORTLIST: Mapping[str, tuple[str, ...]] = {
    "neural_network": ("nn_003", "nn_001", "nn_041"),
    "xgboost": ("xgb_core_023", "xgb_core_038_g10", "xgb_core_038"),
    "elastic_net": ("enet_r000_l001", "enet_r004_l005", "enet_r003_l006"),
}
LOCKED_ENSEMBLE_NN_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)

# These are copied from the independently verified Stage 9 shortlist named by
# the exact build and payload digest above.  Validation callers cannot replace
# model complexity with a favorable value.
LOCKED_STAGE9_COMPLEXITY: Mapping[tuple[str, str], float] = {
    ("neural_network", "nn_003"): 7_881.0,
    ("neural_network", "nn_001"): 7_881.0,
    ("neural_network", "nn_041"): 16_449.0,
    ("xgboost", "xgb_core_023"): 72.0,
    ("xgboost", "xgb_core_038_g10"): 19.0,
    ("xgboost", "xgb_core_038"): 29.0,
    ("elastic_net", "enet_r000_l001"): 3_087.5,
    ("elastic_net", "enet_r004_l005"): 15.5,
    ("elastic_net", "enet_r003_l006"): 27.0,
}

LOCKED_FORECAST_MODE = "locked_walk_forward_out_of_sample"
LOCKED_SEED_AGGREGATION = "mean_raw_forecasts_before_monthly_score"
LOCKED_COMPONENT_SCORE_TRANSFORM = "centered_average_percentile_rank"
LOCKED_ENSEMBLE_COMBINATION = "weighted_sum_of_component_scores"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_MONTHLY_IC_COLUMNS = frozenset(
    {"model_family", "config_id", "holding_month", "rank_ic"}
)
_CANDIDATE_METADATA_COLUMNS = frozenset(
    {"model_family", "config_id", "signal_turnover"}
)
_ENSEMBLE_MONTHLY_COLUMNS = frozenset(
    {"nn_weight", "holding_month", "rank_ic"}
)
_ENSEMBLE_METADATA_COLUMNS = frozenset({"nn_weight", "signal_turnover"})
_RAW_COMPONENT_FORECAST_COLUMNS = frozenset(
    {
        "model_family",
        "config_id",
        "signal_date",
        "holding_month",
        "permno",
        "seed",
        "raw_forecast",
    }
)


@dataclass(frozen=True)
class ValidationForecastProvenance:
    """Closed, hash-bound description of the permitted validation forecasts."""

    source_receipt_sha256: str
    stage9_selector_build_id: str = LOCKED_STAGE9_SELECTOR_BUILD_ID
    stage9_shortlist_sha256: str = LOCKED_STAGE9_SHORTLIST_SHA256
    stochastic_seed_order: tuple[int, ...] = VALIDATION_STOCHASTIC_SEEDS
    forecast_mode: str = LOCKED_FORECAST_MODE
    seed_aggregation: str = LOCKED_SEED_AGGREGATION
    component_score_transform: str = LOCKED_COMPONENT_SCORE_TRANSFORM
    ensemble_combination: str = LOCKED_ENSEMBLE_COMBINATION
    ensemble_reranked: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source_receipt_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.source_receipt_sha256
        ):
            raise ValueError("source_receipt_sha256 must be a lowercase SHA-256 digest")
        expected = {
            "stage9_selector_build_id": LOCKED_STAGE9_SELECTOR_BUILD_ID,
            "stage9_shortlist_sha256": LOCKED_STAGE9_SHORTLIST_SHA256,
            "stochastic_seed_order": VALIDATION_STOCHASTIC_SEEDS,
            "forecast_mode": LOCKED_FORECAST_MODE,
            "seed_aggregation": LOCKED_SEED_AGGREGATION,
            "component_score_transform": LOCKED_COMPONENT_SCORE_TRANSFORM,
            "ensemble_combination": LOCKED_ENSEMBLE_COMBINATION,
            "ensemble_reranked": False,
        }
        for field, required in expected.items():
            if getattr(self, field) != required:
                raise ValueError(f"Validation forecast provenance differs at {field}")


@dataclass(frozen=True)
class LockedEnsembleScores:
    """Five-seed component scores and their direct, non-reranked composite."""

    scores: pd.DataFrame
    raw_forecast_payload_sha256: str
    ensemble_score_payload_sha256: str
    provenance: ValidationForecastProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.scores, pd.DataFrame):
            raise ValueError("scores must be a pandas DataFrame")
        if not isinstance(self.provenance, ValidationForecastProvenance):
            raise ValueError("validated forecast provenance is required")
        for field in (
            "raw_forecast_payload_sha256",
            "ensemble_score_payload_sha256",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        scores = self.scores.copy(deep=True)
        if selection_frame_sha256(scores) != self.ensemble_score_payload_sha256:
            raise ValueError("ensemble score payload digest differs")
        object.__setattr__(self, "scores", scores)


@dataclass(frozen=True)
class FamilySelection:
    """One prespecified family choice and its validation diagnostics."""

    model_family: str
    config_id: str
    mean_rank_ic: float
    rank_ic_bootstrap_se: float
    family_best_mean_rank_ic: float
    family_best_bootstrap_se: float
    one_se_threshold: float
    complexity: float
    signal_turnover: float
    annual_rank_ic: tuple[tuple[int, float], ...]
    positive_annual_ic_years: int
    annual_ic_gate_pass: bool
    mandatory_comparator_retained: bool
    provenance: ValidationForecastProvenance


@dataclass(frozen=True)
class EnsembleSelection:
    """Diagnostic and effective fixed NN weight after annual family gates."""

    diagnostic_nn_weight: float
    effective_nn_weight: float | None
    effective_xgboost_weight: float | None
    mean_rank_ic: float
    rank_ic_bootstrap_se: float
    best_mean_rank_ic: float
    best_bootstrap_se: float
    one_se_threshold: float
    signal_turnover: float
    forced_by_annual_gate: bool
    active_v1_ml_strategy_qualifies: bool
    gate_reason: str
    provenance: ValidationForecastProvenance


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = frozenset(str(column) for column in frame.columns)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        raise ValueError(
            f"{label} columns must be exact; missing={missing}, "
            f"unexpected={unexpected}"
        )


def _finite_number(value: object, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return 0.0 if result == 0.0 else result


def _rank_ic(value: object) -> float:
    result = _finite_number(value, label="rank_ic")
    if result < -1.0 or result > 1.0:
        raise ValueError("rank_ic must lie within [-1, 1]")
    return result


def _signal_turnover(value: object) -> float:
    result = _finite_number(value, label="signal_turnover", nonnegative=True)
    if result > 1.0:
        raise ValueError("signal_turnover must lie within [0, 1]")
    return result


def _require_provenance(value: object) -> ValidationForecastProvenance:
    if not isinstance(value, ValidationForecastProvenance):
        raise ValueError("validated forecast provenance is required")
    return value


def _canonical_scalar(value: object) -> str:
    if isinstance(value, pd.Period):
        return f"period:{value.freqstr}:{value}"
    if isinstance(value, pd.Timestamp):
        return f"timestamp:{value.isoformat()}"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("payload hash input may not contain nonfinite values")
        return f"float:{value.hex()}"
    if isinstance(value, bool):
        return f"bool:{int(value)}"
    if isinstance(value, int):
        return f"int:{value}"
    if value is None or bool(pd.isna(value)):
        raise ValueError("payload hash input may not contain null values")
    return f"text:{value}"


def selection_frame_sha256(frame: pd.DataFrame) -> str:
    """Return an order-independent canonical digest for a closed table."""

    if any(not isinstance(column, str) for column in frame.columns):
        raise ValueError("payload hash input columns must be strings")
    columns = tuple(sorted(frame.columns))
    if len(set(columns)) != len(frame.columns):
        raise ValueError("payload hash input columns must be unique")
    records = [
        "\x1f".join(_canonical_scalar(row[column]) for column in columns)
        for _, row in frame.loc[:, columns].iterrows()
    ]
    payload = "\x1e".join(("\x1f".join(columns), *sorted(records))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_holding_months(values: Sequence[object]) -> pd.PeriodIndex:
    try:
        months = pd.PeriodIndex(values, freq="M")
    except (TypeError, ValueError) as exc:
        raise ValueError("holding_month must contain valid calendar months") from exc
    if months.hasnans:
        raise ValueError("holding_month may not be null")
    return months


def _bootstrap_plan():
    return make_moving_block_bootstrap_plan(
        VALIDATION_MONTHS,
        block_length=BOOTSTRAP_BLOCK_MONTHS,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_MASTER_SEED,
    )


def _validated_family_inputs(
    monthly_ic: pd.DataFrame,
    candidate_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_exact_columns(monthly_ic, _MONTHLY_IC_COLUMNS, label="monthly IC")
    _require_exact_columns(
        candidate_metadata,
        _CANDIDATE_METADATA_COLUMNS,
        label="candidate metadata",
    )
    monthly = monthly_ic.copy()
    metadata = candidate_metadata.copy()
    if monthly.empty or metadata.empty:
        raise ValueError("Locked family selection inputs may not be empty")
    monthly["model_family"] = monthly["model_family"].astype(str)
    monthly["config_id"] = monthly["config_id"].astype(str)
    metadata["model_family"] = metadata["model_family"].astype(str)
    metadata["config_id"] = metadata["config_id"].astype(str)
    monthly["holding_month"] = _canonical_holding_months(monthly["holding_month"])
    monthly["rank_ic"] = [
        _rank_ic(value) for value in monthly["rank_ic"]
    ]
    metadata["signal_turnover"] = [
        _signal_turnover(value) for value in metadata["signal_turnover"]
    ]
    key = ["model_family", "config_id"]
    if metadata.duplicated(key).any():
        raise ValueError("Candidate metadata keys must be unique")
    if monthly.duplicated([*key, "holding_month"]).any():
        raise ValueError("Each configuration must have one IC per validation month")

    expected_keys = {
        (family, config_id)
        for family, config_ids in LOCKED_SHORTLIST.items()
        for config_id in config_ids
    }
    monthly_keys = set(map(tuple, monthly[key].drop_duplicates().to_numpy()))
    metadata_keys = set(map(tuple, metadata[key].to_numpy()))
    if monthly_keys != expected_keys or metadata_keys != expected_keys:
        raise ValueError("Inputs must contain exactly the nine locked configurations")
    for family, config_ids in LOCKED_SHORTLIST.items():
        for config_id in config_ids:
            group = monthly.loc[
                monthly["model_family"].eq(family)
                & monthly["config_id"].eq(config_id)
            ]
            if tuple(sorted(group["holding_month"].tolist())) != VALIDATION_MONTHS:
                raise ValueError(
                    f"{family}/{config_id} must contain exactly the 59 locked "
                    "validation holding months"
                )
    return monthly, metadata


def select_family_specifications(
    monthly_ic: pd.DataFrame,
    candidate_metadata: pd.DataFrame,
    *,
    provenance: ValidationForecastProvenance,
) -> dict[str, FamilySelection]:
    """Select one locked specification per family using IC and no performance.

    The best configuration supplies the one-standard-error threshold.  Within
    that set, lower development-locked complexity wins, followed by lower
    validation signal turnover and then the configuration identifier.
    """

    _require_provenance(provenance)
    tolerance = TIE_TOLERANCE
    monthly, metadata = _validated_family_inputs(monthly_ic, candidate_metadata)
    plan = _bootstrap_plan()
    selections: dict[str, FamilySelection] = {}
    for family, locked_ids in LOCKED_SHORTLIST.items():
        rows: list[dict[str, object]] = []
        for config_id in locked_ids:
            values = monthly.loc[
                monthly["model_family"].eq(family)
                & monthly["config_id"].eq(config_id)
            ].sort_values("holding_month", kind="mergesort")
            mean_ic = float(values["rank_ic"].mean())
            bootstrap_se = bootstrap_equal_month_mean_se(
                values[["holding_month", "rank_ic"]],
                plan,
            )
            details = metadata.loc[
                metadata["model_family"].eq(family)
                & metadata["config_id"].eq(config_id)
            ].iloc[0]
            rows.append(
                {
                    "config_id": config_id,
                    "mean_rank_ic": mean_ic,
                    "bootstrap_se": bootstrap_se,
                    "complexity": LOCKED_STAGE9_COMPLEXITY[(family, config_id)],
                    "signal_turnover": float(details["signal_turnover"]),
                }
            )
        aggregate = pd.DataFrame(rows)
        maximum = float(aggregate["mean_rank_ic"].max())
        best_candidates = aggregate.loc[
            aggregate["mean_rank_ic"] >= maximum - tolerance
        ].sort_values(
            ["complexity", "signal_turnover", "config_id"], kind="mergesort"
        )
        best = best_candidates.iloc[0]
        threshold = float(best["mean_rank_ic"] - best["bootstrap_se"])
        qualifying = aggregate.loc[
            aggregate["mean_rank_ic"] >= threshold - tolerance
        ].sort_values(
            ["complexity", "signal_turnover", "config_id"], kind="mergesort"
        )
        selected = qualifying.iloc[0]
        selected_monthly = monthly.loc[
            monthly["model_family"].eq(family)
            & monthly["config_id"].eq(str(selected["config_id"]))
        ].copy()
        selected_monthly["year"] = selected_monthly["holding_month"].map(
            lambda month: int(month.year)
        )
        annual = tuple(
            (year, float(selected_monthly.loc[selected_monthly["year"].eq(year), "rank_ic"].mean()))
            for year in VALIDATION_YEARS
        )
        positive_years = sum(value > 0.0 for _, value in annual)
        annual_gate = positive_years >= 3
        selections[family] = FamilySelection(
            model_family=family,
            config_id=str(selected["config_id"]),
            mean_rank_ic=float(selected["mean_rank_ic"]),
            rank_ic_bootstrap_se=float(selected["bootstrap_se"]),
            family_best_mean_rank_ic=float(best["mean_rank_ic"]),
            family_best_bootstrap_se=float(best["bootstrap_se"]),
            one_se_threshold=threshold,
            complexity=float(selected["complexity"]),
            signal_turnover=float(selected["signal_turnover"]),
            annual_rank_ic=annual,
            positive_annual_ic_years=positive_years,
            annual_ic_gate_pass=annual_gate,
            mandatory_comparator_retained=(family == "elastic_net"),
            provenance=provenance,
        )
    return selections


def _validated_ensemble_inputs(
    monthly_ic: pd.DataFrame,
    candidate_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_exact_columns(
        monthly_ic, _ENSEMBLE_MONTHLY_COLUMNS, label="ensemble monthly IC"
    )
    _require_exact_columns(
        candidate_metadata,
        _ENSEMBLE_METADATA_COLUMNS,
        label="ensemble metadata",
    )
    monthly = monthly_ic.copy()
    metadata = candidate_metadata.copy()
    monthly["nn_weight"] = [
        _finite_number(value, label="nn_weight") for value in monthly["nn_weight"]
    ]
    metadata["nn_weight"] = [
        _finite_number(value, label="nn_weight") for value in metadata["nn_weight"]
    ]
    monthly["holding_month"] = _canonical_holding_months(monthly["holding_month"])
    monthly["rank_ic"] = [
        _rank_ic(value) for value in monthly["rank_ic"]
    ]
    metadata["signal_turnover"] = [
        _signal_turnover(value) for value in metadata["signal_turnover"]
    ]
    if monthly.duplicated(["nn_weight", "holding_month"]).any():
        raise ValueError("Each ensemble weight must have one IC per validation month")
    if metadata["nn_weight"].duplicated().any():
        raise ValueError("Ensemble metadata weights must be unique")
    actual_monthly_weights = tuple(sorted(monthly["nn_weight"].unique()))
    actual_metadata_weights = tuple(sorted(metadata["nn_weight"].unique()))
    if (
        actual_monthly_weights != LOCKED_ENSEMBLE_NN_WEIGHTS
        or actual_metadata_weights != LOCKED_ENSEMBLE_NN_WEIGHTS
    ):
        raise ValueError("Inputs must contain exactly the five locked NN weights")
    for weight in LOCKED_ENSEMBLE_NN_WEIGHTS:
        group = monthly.loc[np.isclose(monthly["nn_weight"], weight, atol=0.0)]
        if tuple(sorted(group["holding_month"].tolist())) != VALIDATION_MONTHS:
            raise ValueError(
                f"NN weight {weight} must contain exactly the 59 locked months"
            )
    return monthly, metadata


def select_ensemble_weight(
    monthly_ic: pd.DataFrame,
    candidate_metadata: pd.DataFrame,
    *,
    nn_annual_gate_pass: bool,
    xgboost_annual_gate_pass: bool,
    provenance: ValidationForecastProvenance,
) -> EnsembleSelection:
    """Apply the fixed-weight IC rule and the two nonlinear annual gates."""

    if not isinstance(nn_annual_gate_pass, (bool, np.bool_)) or not isinstance(
        xgboost_annual_gate_pass, (bool, np.bool_)
    ):
        raise ValueError("Annual gate inputs must be booleans")
    _require_provenance(provenance)
    tolerance = TIE_TOLERANCE
    monthly, metadata = _validated_ensemble_inputs(monthly_ic, candidate_metadata)
    plan = _bootstrap_plan()
    rows: list[dict[str, float]] = []
    for weight in LOCKED_ENSEMBLE_NN_WEIGHTS:
        values = monthly.loc[
            np.isclose(monthly["nn_weight"], weight, atol=0.0)
        ].sort_values("holding_month", kind="mergesort")
        details = metadata.loc[
            np.isclose(metadata["nn_weight"], weight, atol=0.0)
        ].iloc[0]
        rows.append(
            {
                "nn_weight": weight,
                "mean_rank_ic": float(values["rank_ic"].mean()),
                "bootstrap_se": bootstrap_equal_month_mean_se(
                    values[["holding_month", "rank_ic"]], plan
                ),
                "signal_turnover": float(details["signal_turnover"]),
            }
        )
    aggregate = pd.DataFrame(rows)
    maximum = float(aggregate["mean_rank_ic"].max())
    best_candidates = aggregate.loc[
        aggregate["mean_rank_ic"] >= maximum - tolerance
    ].copy()
    best_candidates["distance_from_half"] = (
        best_candidates["nn_weight"] - 0.5
    ).abs()
    best = best_candidates.sort_values(
        ["distance_from_half", "signal_turnover", "nn_weight"],
        kind="mergesort",
    ).iloc[0]
    threshold = float(best["mean_rank_ic"] - best["bootstrap_se"])
    qualifying = aggregate.loc[
        aggregate["mean_rank_ic"] >= threshold - tolerance
    ].copy()
    qualifying["distance_from_half"] = (qualifying["nn_weight"] - 0.5).abs()
    diagnostic = qualifying.sort_values(
        ["distance_from_half", "signal_turnover", "nn_weight"],
        kind="mergesort",
    ).iloc[0]

    if bool(nn_annual_gate_pass) and bool(xgboost_annual_gate_pass):
        effective = float(diagnostic["nn_weight"])
        active = True
        forced = False
        reason = "both_nonlinear_families_pass"
    elif bool(nn_annual_gate_pass):
        effective = 1.0
        active = True
        forced = True
        reason = "xgboost_failed_annual_ic_gate"
    elif bool(xgboost_annual_gate_pass):
        effective = 0.0
        active = True
        forced = True
        reason = "neural_network_failed_annual_ic_gate"
    else:
        effective = None
        active = False
        forced = True
        reason = "both_nonlinear_families_failed_no_active_v1_ml_strategy"

    diagnostic_weight = float(diagnostic["nn_weight"])
    diagnostic_row = aggregate.loc[
        np.isclose(aggregate["nn_weight"], diagnostic_weight, atol=0.0)
    ].iloc[0]
    return EnsembleSelection(
        diagnostic_nn_weight=diagnostic_weight,
        effective_nn_weight=effective,
        effective_xgboost_weight=None if effective is None else 1.0 - effective,
        mean_rank_ic=float(diagnostic_row["mean_rank_ic"]),
        rank_ic_bootstrap_se=float(diagnostic_row["bootstrap_se"]),
        best_mean_rank_ic=float(best["mean_rank_ic"]),
        best_bootstrap_se=float(best["bootstrap_se"]),
        one_se_threshold=threshold,
        signal_turnover=float(diagnostic_row["signal_turnover"]),
        forced_by_annual_gate=forced,
        active_v1_ml_strategy_qualifies=active,
        gate_reason=reason,
        provenance=provenance,
    )


def compose_locked_ensemble_scores(
    raw_component_forecasts: pd.DataFrame,
    *,
    selected_nn_config_id: str,
    selected_xgboost_config_id: str,
    nn_weight: float,
    provenance: ValidationForecastProvenance,
) -> LockedEnsembleScores:
    """Build the five-seed component scores and direct fixed-weight composite.

    The function accepts raw forecasts only—never outcomes.  It enforces the
    exact five validation seeds for every security/month/component, averages
    raw forecasts before ranking, and returns the weighted component-score sum
    without a second rank transform.
    """

    provenance = _require_provenance(provenance)
    _require_exact_columns(
        raw_component_forecasts,
        _RAW_COMPONENT_FORECAST_COLUMNS,
        label="raw component forecast",
    )
    if selected_nn_config_id not in LOCKED_SHORTLIST["neural_network"]:
        raise ValueError("selected NN configuration is outside the locked shortlist")
    if selected_xgboost_config_id not in LOCKED_SHORTLIST["xgboost"]:
        raise ValueError("selected XGBoost configuration is outside the locked shortlist")
    weight = _finite_number(nn_weight, label="nn_weight")
    if weight not in LOCKED_ENSEMBLE_NN_WEIGHTS:
        raise ValueError("nn_weight is outside the five locked ensemble weights")

    frame = raw_component_forecasts.copy()
    if frame.empty:
        raise ValueError("raw component forecasts may not be empty")
    frame["model_family"] = frame["model_family"].astype(str)
    frame["config_id"] = frame["config_id"].astype(str)
    expected_configs = {
        ("neural_network", selected_nn_config_id),
        ("xgboost", selected_xgboost_config_id),
    }
    actual_configs = set(
        map(
            tuple,
            frame[["model_family", "config_id"]].drop_duplicates().to_numpy(),
        )
    )
    if actual_configs != expected_configs:
        raise ValueError("raw forecasts must contain exactly the two selected components")
    try:
        signal_dates = pd.to_datetime(frame["signal_date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("signal_date must contain valid dates") from exc
    if signal_dates.isna().any() or signal_dates.dt.tz is not None:
        raise ValueError("signal_date must be timezone-naive and non-null")
    frame["signal_date"] = signal_dates.dt.date
    frame["holding_month"] = _canonical_holding_months(frame["holding_month"])
    if not set(frame["holding_month"]).issubset(set(VALIDATION_MONTHS)):
        raise ValueError("raw forecasts contain a month outside locked validation")

    raw_permno = pd.to_numeric(frame["permno"], errors="coerce").to_numpy(float)
    raw_seed = pd.to_numeric(frame["seed"], errors="coerce").to_numpy(float)
    raw_forecast = pd.to_numeric(
        frame["raw_forecast"], errors="coerce"
    ).to_numpy(float)
    if (
        not np.isfinite(raw_permno).all()
        or np.any(raw_permno <= 0.0)
        or np.any(raw_permno != np.floor(raw_permno))
    ):
        raise ValueError("permno must contain positive integers")
    if not np.isfinite(raw_seed).all() or np.any(raw_seed != np.floor(raw_seed)):
        raise ValueError("seed must contain integers")
    if not np.isfinite(raw_forecast).all():
        raise ValueError("raw_forecast must be finite")
    frame["permno"] = raw_permno.astype(np.int64)
    frame["seed"] = raw_seed.astype(np.int64)
    frame["raw_forecast"] = raw_forecast

    month_signal = frame[["holding_month", "signal_date"]].drop_duplicates()
    if month_signal["holding_month"].duplicated().any():
        raise ValueError("each holding month must have exactly one signal date")
    for row in month_signal.itertuples(index=False):
        if row.signal_date >= row.holding_month.start_time.date():
            raise ValueError("signal_date must precede its holding month")

    key = [
        "model_family",
        "config_id",
        "signal_date",
        "holding_month",
        "permno",
    ]
    if frame.duplicated([*key, "seed"]).any():
        raise ValueError("raw component forecast keys must be unique")
    frame = frame.sort_values([*key, "seed"], kind="mergesort").reset_index(drop=True)
    records: list[dict[str, object]] = []
    for values, group in frame.groupby(key, sort=True, observed=True):
        seeds = tuple(sorted(int(seed) for seed in group["seed"]))
        if seeds != VALIDATION_STOCHASTIC_SEEDS:
            raise ValueError(
                f"forecast key {values!r} has seeds {seeds}; "
                f"expected {VALIDATION_STOCHASTIC_SEEDS}"
            )
        record = dict(zip(key, values, strict=True))
        record["raw_forecast_mean"] = float(group["raw_forecast"].mean())
        records.append(record)
    averaged = pd.DataFrame.from_records(records)
    averaged["component_score"] = np.nan
    group_columns = ["model_family", "config_id", "signal_date", "holding_month"]
    for _, positions in averaged.groupby(
        group_columns,
        sort=True,
        observed=True,
    ).indices.items():
        indices = np.asarray(positions, dtype=np.int64)
        averaged.iloc[
            indices,
            averaged.columns.get_loc("component_score"),
        ] = centered_average_rank(averaged.iloc[indices]["raw_forecast_mean"])

    join_key = ["signal_date", "holding_month", "permno"]
    nn = averaged.loc[
        averaged["model_family"].eq("neural_network"),
        [*join_key, "component_score"],
    ].rename(columns={"component_score": "nn_score"})
    xgboost = averaged.loc[
        averaged["model_family"].eq("xgboost"),
        [*join_key, "component_score"],
    ].rename(columns={"component_score": "xgboost_score"})
    scores = nn.merge(
        xgboost,
        on=join_key,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not scores["_merge"].eq("both").all():
        raise ValueError("NN and XGBoost score domains must match exactly")
    scores = scores.drop(columns="_merge")
    scores["ensemble_score"] = (
        weight * scores["nn_score"]
        + (1.0 - weight) * scores["xgboost_score"]
    )
    scores["nn_weight"] = weight
    scores["xgboost_weight"] = 1.0 - weight
    scores = scores.sort_values(join_key, kind="mergesort").reset_index(drop=True)
    return LockedEnsembleScores(
        scores=scores,
        raw_forecast_payload_sha256=selection_frame_sha256(frame),
        ensemble_score_payload_sha256=selection_frame_sha256(scores),
        provenance=provenance,
    )


def mean_adjacent_signal_turnover(
    scores: pd.DataFrame,
    *,
    candidate_columns: Sequence[str],
    security_column: str = "permno",
) -> pd.DataFrame:
    """Calculate mean absolute score change for adjacent-month survivors.

    Every common security-month transition receives one observation.  Months
    that are not consecutive are never bridged.  The output is suitable only
    as the locked IC tie-break input; it is not portfolio turnover.
    """

    candidates = tuple(str(column) for column in candidate_columns)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidate_columns must be a nonempty unique sequence")
    expected = frozenset((*candidates, "holding_month", security_column, "score"))
    _require_exact_columns(scores, expected, label="signal score")
    frame = scores.copy()
    frame["holding_month"] = _canonical_holding_months(frame["holding_month"])
    frame["score"] = [
        _finite_number(value, label="score") for value in frame["score"]
    ]
    if (frame["score"].abs() > 0.5 + TIE_TOLERANCE).any():
        raise ValueError("signal scores must lie within [-0.5, 0.5]")
    if frame.duplicated([*candidates, "holding_month", security_column]).any():
        raise ValueError("Signal score keys must be unique")
    rows: list[dict[str, object]] = []
    group_key: str | list[str] = candidates[0] if len(candidates) == 1 else list(candidates)
    for key, candidate in frame.groupby(group_key, sort=True, observed=True):
        key_values = (key,) if len(candidates) == 1 else tuple(key)
        changes: list[np.ndarray] = []
        pair_count = 0
        ordered_months = sorted(candidate["holding_month"].unique())
        for prior_month, current_month in zip(ordered_months, ordered_months[1:]):
            if current_month.ordinal - prior_month.ordinal != 1:
                continue
            prior = candidate.loc[
                candidate["holding_month"].eq(prior_month),
                [security_column, "score"],
            ].rename(columns={"score": "prior_score"})
            current = candidate.loc[
                candidate["holding_month"].eq(current_month),
                [security_column, "score"],
            ].rename(columns={"score": "current_score"})
            common = prior.merge(
                current, on=security_column, how="inner", validate="one_to_one"
            )
            if not common.empty:
                changes.append(
                    np.abs(
                        common["current_score"].to_numpy(dtype=float)
                        - common["prior_score"].to_numpy(dtype=float)
                    )
                )
                pair_count += 1
        if not changes:
            raise ValueError(f"Candidate {key_values!r} has no adjacent-month survivors")
        record = dict(zip(candidates, key_values, strict=True))
        record["signal_turnover"] = float(np.mean(np.concatenate(changes)))
        record["adjacent_month_pairs"] = pair_count
        record["security_transitions"] = int(sum(len(values) for values in changes))
        rows.append(record)
    return pd.DataFrame(rows).sort_values(list(candidates), kind="mergesort").reset_index(
        drop=True
    )
