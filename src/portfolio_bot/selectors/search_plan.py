"""Native-free planning primitives for the frozen Stage 9 selector search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from portfolio_bot.selectors.model_specs import (
    DEVELOPMENT_SEEDS,
    XGBoostConfig,
    core_xgboost_configurations,
    neural_configurations,
)


FOLD_IDS = ("dev_fold_1", "dev_fold_2", "dev_fold_3", "dev_fold_4")
TIE_TOLERANCE = 1e-12


def elastic_net_config_ids() -> tuple[str, ...]:
    """Return the native-free mirror of the frozen 5 by 50 path IDs."""

    return tuple(
        f"enet_r{rho_index:03d}_l{lambda_index:03d}"
        for rho_index in range(5)
        for lambda_index in range(1, 51)
    )


@dataclass(frozen=True)
class SearchTrial:
    model_family: str
    config_id: str
    stage09_execution: str
    seed_count: int
    fold_count: int
    fit_receipt_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SearchPlanCounts:
    neural_configurations: int
    xgboost_core_configurations: int
    xgboost_gamma_configurations: int
    elastic_net_configurations: int
    stochastic_fit_jobs: int
    elastic_path_jobs: int
    fit_receipts: int
    evaluated_configurations: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def frozen_search_plan_counts() -> SearchPlanCounts:
    neural = len(neural_configurations())
    xgb_core = len(core_xgboost_configurations())
    xgb_gamma = 8
    elastic = len(elastic_net_config_ids())
    stochastic_jobs = (neural + xgb_core + xgb_gamma) * len(DEVELOPMENT_SEEDS) * len(FOLD_IDS)
    receipts = stochastic_jobs + elastic * len(FOLD_IDS)
    return SearchPlanCounts(
        neural_configurations=neural,
        xgboost_core_configurations=xgb_core,
        xgboost_gamma_configurations=xgb_gamma,
        elastic_net_configurations=elastic,
        stochastic_fit_jobs=stochastic_jobs,
        elastic_path_jobs=len(FOLD_IDS),
        fit_receipts=receipts,
        evaluated_configurations=neural + xgb_core + xgb_gamma + elastic,
    )


def primary_trial_ledger() -> pd.DataFrame:
    """Return one row per Stage 9 selector decision alternative.

    Seed repetitions and chronological folds are mandatory replications of one
    configuration, not separate research alternatives.
    """

    records: list[dict[str, object]] = []
    for config in neural_configurations():
        records.append(
            SearchTrial(
                "neural_network",
                config.config_id,
                "EXECUTED_PRIMARY_RESIDUAL_RANK_MSE",
                len(DEVELOPMENT_SEEDS),
                len(FOLD_IDS),
                len(DEVELOPMENT_SEEDS) * len(FOLD_IDS),
            ).as_dict()
        )
    for config in core_xgboost_configurations():
        records.append(
            SearchTrial(
                "xgboost",
                config.config_id,
                "EXECUTED_CORE_PRIMARY_RESIDUAL_RANK_MSE",
                len(DEVELOPMENT_SEEDS),
                len(FOLD_IDS),
                len(DEVELOPMENT_SEEDS) * len(FOLD_IDS),
            ).as_dict()
        )
    # Parent IDs are filled only after the sealed core search.  The challenger
    # alternatives themselves are nevertheless preregistered as eight slots.
    for ordinal in range(1, 9):
        records.append(
            SearchTrial(
                "xgboost",
                f"xgb_gamma_slot_{ordinal:02d}",
                "REGISTERED_PENDING_SEALED_CORE_PARENT",
                len(DEVELOPMENT_SEEDS),
                len(FOLD_IDS),
                len(DEVELOPMENT_SEEDS) * len(FOLD_IDS),
            ).as_dict()
        )
    for config_id in elastic_net_config_ids():
        records.append(
            SearchTrial(
                "elastic_net",
                config_id,
                "EXECUTED_PRIMARY_RESIDUAL_RANK_MSE",
                0,
                len(FOLD_IDS),
                len(FOLD_IDS),
            ).as_dict()
        )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != 346:
        raise AssertionError("Frozen Stage 9 trial ledger must have 346 rows")
    return frame


def _xgboost_strength_key(config: XGBoostConfig) -> tuple[float, ...]:
    """Ascending deterministic key; stronger shrinkage is preferred."""

    return (
        -float(config.reg_lambda),
        -float(config.reg_alpha),
        -float(config.child_share),
        float(config.subsample),
        float(config.colsample_bytree),
        float(config.eta),
    )


def select_gamma_parent_ids(
    core_metrics: pd.DataFrame,
    *,
    tolerance: float = TIE_TOLERANCE,
) -> tuple[str, ...]:
    """Select the four highest three-seed core development IC configs.

    ``core_metrics`` must already contain the IC of raw-seed-averaged OOF
    forecasts and XGBoost's frozen complexity measure.  Metric ties within
    1e-12 prefer lower complexity, then stronger regularization, then ID.
    """

    required = {"config_id", "mean_rank_ic", "complexity"}
    if not required <= set(core_metrics):
        raise ValueError("Core XGBoost metrics are missing gamma-parent fields")
    canonical = {value.config_id: value for value in core_xgboost_configurations()}
    frame = core_metrics.copy()
    if set(frame["config_id"].astype(str)) != set(canonical) or len(frame) != 40:
        raise ValueError("Gamma-parent selection requires exactly all 40 core configs")
    mean_ic = pd.to_numeric(frame["mean_rank_ic"], errors="coerce").to_numpy(float)
    complexity = pd.to_numeric(frame["complexity"], errors="coerce").to_numpy(float)
    if not np.isfinite(mean_ic).all() or not np.isfinite(complexity).all():
        raise ValueError("Gamma-parent metrics must be finite")
    records = [
        (
            str(row.config_id),
            float(row.mean_rank_ic),
            float(row.complexity),
            _xgboost_strength_key(canonical[str(row.config_id)]),
        )
        for row in frame[["config_id", "mean_rank_ic", "complexity"]].itertuples(index=False)
    ]

    def compare_key(value: tuple[str, float, float, tuple[float, ...]]) -> tuple[object, ...]:
        config_id, metric, model_complexity, strength = value
        # Quantizing only for tie ordering would be unstable around bin edges;
        # selection below therefore groups exact tolerance ties iteratively.
        return (-metric, model_complexity, strength, config_id)

    remaining = records[:]
    selected: list[str] = []
    while len(selected) < 4:
        best_metric = max(value[1] for value in remaining)
        tied = [value for value in remaining if value[1] >= best_metric - tolerance]
        winner = min(tied, key=lambda value: compare_key((value[0], best_metric, value[2], value[3])))
        selected.append(winner[0])
        remaining = [value for value in remaining if value[0] != winner[0]]
    return tuple(selected)


def derive_gamma_challengers(
    parent_ids: Sequence[str],
    *,
    sealed_core_receipt_sha256: str,
) -> tuple[XGBoostConfig, ...]:
    """Derive the eight frozen gamma challengers from sealed core parents."""

    if len(sealed_core_receipt_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sealed_core_receipt_sha256
    ):
        raise ValueError("A lowercase SHA-256 sealed core receipt is required")
    ids = tuple(str(value) for value in parent_ids)
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError("Exactly four distinct gamma parents are required")
    canonical: Mapping[str, XGBoostConfig] = {
        value.config_id: value for value in core_xgboost_configurations()
    }
    if any(value not in canonical for value in ids):
        raise ValueError("Gamma parent escaped the frozen core grid")
    result: list[XGBoostConfig] = []
    for parent_id in ids:
        base = canonical[parent_id]
        for gamma, suffix in ((0.1, "g01"), (1.0, "g10")):
            result.append(
                replace(
                    base,
                    config_id=f"{base.config_id}_{suffix}",
                    gamma=gamma,
                    challenger=True,
                )
            )
    return tuple(result)


def finalize_trial_ledger(
    gamma_challengers: Sequence[XGBoostConfig],
) -> pd.DataFrame:
    """Replace the eight sealed-parent slots with their executed config IDs."""

    challengers = tuple(gamma_challengers)
    if len(challengers) != 8 or len({value.config_id for value in challengers}) != 8:
        raise ValueError("Exactly eight distinct gamma challengers are required")
    if any(not value.challenger or value.gamma not in (0.1, 1.0) for value in challengers):
        raise ValueError("Trial ledger gamma configurations are not frozen challengers")
    frame = primary_trial_ledger()
    slots = frame.index[
        frame["stage09_execution"].eq("REGISTERED_PENDING_SEALED_CORE_PARENT")
    ].tolist()
    if len(slots) != len(challengers):  # pragma: no cover - internal invariant
        raise AssertionError("Gamma slot count differs")
    for position, config in zip(slots, challengers, strict=True):
        frame.loc[position, "config_id"] = config.config_id
        frame.loc[position, "stage09_execution"] = (
            "EXECUTED_GAMMA_PRIMARY_RESIDUAL_RANK_MSE"
        )
    if frame["config_id"].duplicated().any():
        raise ValueError("Final Stage 9 trial ledger contains duplicate alternatives")
    return frame


__all__ = [
    "FOLD_IDS",
    "SearchPlanCounts",
    "SearchTrial",
    "derive_gamma_challengers",
    "elastic_net_config_ids",
    "finalize_trial_ledger",
    "frozen_search_plan_counts",
    "primary_trial_ledger",
    "select_gamma_parent_ids",
]
