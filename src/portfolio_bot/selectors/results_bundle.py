"""Assemble compact Stage 9 OOF result tables from authenticated job outputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_bot.selectors.evaluation import FamilyEvaluation
from portfolio_bot.selectors.metrics import DEVELOPMENT_SEEDS
from portfolio_bot.selectors.splits import FoldIndices


EXPECTED_OOF_ROWS = 172_969
EXPECTED_OOF_PRESENT = 172_799
FAMILY_CODES = {
    "neural_network": "nn",
    "xgboost": "xgb",
    "elastic_net": "enet",
}


@dataclass(frozen=True)
class OOFDomain:
    source_positions: np.ndarray
    rows: pd.DataFrame
    fold_output_positions: Mapping[str, np.ndarray]


def build_oof_domain(
    rows: pd.DataFrame,
    folds: Sequence[FoldIndices],
    *,
    enforce_frozen_counts: bool = True,
) -> OOFDomain:
    """Build the unique canonical union of the four outer score domains."""

    required = {
        "signal_date",
        "holding_month",
        "permno",
        "unranked_residual_return",
        "primary_label_status",
    }
    if not required <= set(rows):
        raise ValueError("Stage 9 rows are missing OOF-domain columns")
    fold_values = tuple(folds)
    positions = np.concatenate(
        [np.asarray(fold.outer_validation, dtype=np.int64) for fold in fold_values]
    )
    if len(np.unique(positions)) != len(positions):
        raise ValueError("Outer-fold scoring rows overlap")
    order = np.argsort(positions, kind="stable")
    canonical = positions[order]
    output_by_source = {int(source): index for index, source in enumerate(canonical)}
    fold_output: dict[str, np.ndarray] = {}
    for fold in fold_values:
        fold_output[fold.fold_id] = np.asarray(
            [output_by_source[int(source)] for source in fold.outer_validation],
            dtype=np.int64,
        )
    frame = rows.iloc[canonical][
        [
            "signal_date",
            "holding_month",
            "permno",
            "unranked_residual_return",
            "primary_label_status",
        ]
    ].reset_index(drop=True)
    fold_id = np.empty(len(frame), dtype=object)
    for fold in fold_values:
        fold_id[fold_output[fold.fold_id]] = fold.fold_id
    if pd.isna(fold_id).any():
        raise AssertionError("An OOF row was not assigned to a fold")
    frame.insert(0, "fold_id", fold_id.astype(str))
    residual = pd.to_numeric(
        frame["unranked_residual_return"], errors="coerce"
    ).to_numpy(np.float64)
    outcome_valid = (
        frame["primary_label_status"].eq("PRESENT").to_numpy(bool)
        & np.isfinite(residual)
    )
    frame["unranked_residual_return"] = residual
    frame["outcome_valid"] = outcome_valid
    frame = frame.drop(columns="primary_label_status")
    if enforce_frozen_counts and (
        len(frame) != EXPECTED_OOF_ROWS
        or int(outcome_valid.sum()) != EXPECTED_OOF_PRESENT
    ):
        raise ValueError("Stage 9 OOF domain counts differ")
    return OOFDomain(
        source_positions=canonical,
        rows=frame,
        fold_output_positions=fold_output,
    )


def assemble_stochastic_cube(
    domain: OOFDomain,
    folds: Sequence[FoldIndices],
    *,
    config_ids: Sequence[str],
    prediction_loader: Callable[[str, str, int], np.ndarray],
    seeds: Sequence[int] = DEVELOPMENT_SEEDS,
) -> np.ndarray:
    """Assemble one finite ``OOF x config x seed`` raw-forecast cube."""

    configs = tuple(str(value) for value in config_ids)
    seed_tuple = tuple(int(value) for value in seeds)
    if not configs or len(configs) != len(set(configs)):
        raise ValueError("Stochastic config IDs must be nonempty and unique")
    if seed_tuple != DEVELOPMENT_SEEDS:
        raise ValueError("Stage 9 stochastic cube requires seeds 11, 23, 47")
    cube = np.empty((len(domain.rows), len(configs), len(seed_tuple)), dtype=np.float32)
    for fold in folds:
        destination = domain.fold_output_positions[fold.fold_id]
        expected_rows = len(destination)
        for config_position, config_id in enumerate(configs):
            for seed_position, seed in enumerate(seed_tuple):
                prediction = np.asarray(
                    prediction_loader(fold.fold_id, config_id, seed), dtype=np.float32
                )
                if prediction.shape != (expected_rows,) or not np.isfinite(prediction).all():
                    raise ValueError(
                        f"Invalid OOF prediction for {fold.fold_id}/{config_id}/{seed}"
                    )
                cube[destination, config_position, seed_position] = prediction
    if not np.isfinite(cube).all():
        raise FloatingPointError("Stochastic OOF cube is incomplete")
    return cube


def assemble_elastic_cube(
    domain: OOFDomain,
    folds: Sequence[FoldIndices],
    *,
    config_ids: Sequence[str],
    fold_prediction_loader: Callable[[str], np.ndarray],
) -> np.ndarray:
    """Assemble deterministic Elastic Net paths from four fold matrices."""

    configs = tuple(str(value) for value in config_ids)
    if not configs or len(configs) != len(set(configs)):
        raise ValueError("Elastic Net config IDs must be nonempty and unique")
    cube = np.empty((len(domain.rows), len(configs)), dtype=np.float32)
    for fold in folds:
        destination = domain.fold_output_positions[fold.fold_id]
        prediction = np.asarray(fold_prediction_loader(fold.fold_id), dtype=np.float32)
        expected = (len(destination), len(configs))
        if prediction.shape != expected or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid Elastic Net OOF path for {fold.fold_id}")
        cube[destination] = prediction
    if not np.isfinite(cube).all():
        raise FloatingPointError("Elastic Net OOF cube is incomplete")
    return cube


def raw_column_name(family: str, ordinal: int, seed: int | None) -> str:
    suffix = "" if seed is None else f"__s{seed:03d}"
    return f"raw__{FAMILY_CODES[family]}__{ordinal:03d}{suffix}"


def score_column_name(family: str, ordinal: int) -> str:
    return f"score__{FAMILY_CODES[family]}__{ordinal:03d}"


def build_wide_oof_table(
    domain: OOFDomain,
    *,
    raw_predictions: Mapping[str, np.ndarray],
    evaluations: Mapping[str, FamilyEvaluation],
) -> pd.DataFrame:
    """Return the frozen wide forecast handoff without copying target payloads."""

    keys = domain.rows[["fold_id", "signal_date", "holding_month", "permno"]].copy()
    columns: dict[str, np.ndarray] = {}
    for family in ("neural_network", "xgboost", "elastic_net"):
        if family not in raw_predictions or family not in evaluations:
            raise ValueError(f"Missing Stage 9 OOF family: {family}")
        evaluation = evaluations[family]
        raw = np.asarray(raw_predictions[family], dtype=np.float32)
        if family == "elastic_net":
            expected = (len(keys), len(evaluation.config_ids))
            if raw.shape != expected or evaluation.seeds:
                raise ValueError("Elastic Net wide OOF dimensions differ")
            for config_position, _config_id in enumerate(evaluation.config_ids):
                columns[raw_column_name(family, config_position, None)] = raw[:, config_position]
        else:
            expected = (len(keys), len(evaluation.config_ids), 3)
            if raw.shape != expected or evaluation.seeds != DEVELOPMENT_SEEDS:
                raise ValueError(f"{family} wide OOF dimensions differ")
            for config_position, _config_id in enumerate(evaluation.config_ids):
                for seed_position, seed in enumerate(DEVELOPMENT_SEEDS):
                    columns[raw_column_name(family, config_position, seed)] = raw[
                        :, config_position, seed_position
                    ]
        scores = np.asarray(evaluation.monthly_scores, dtype=np.float32)
        if scores.shape != (len(keys), len(evaluation.config_ids)):
            raise ValueError(f"{family} monthly score dimensions differ")
        for config_position, _config_id in enumerate(evaluation.config_ids):
            columns[score_column_name(family, config_position)] = scores[:, config_position]
    forecast_frame = pd.DataFrame(columns, index=keys.index)
    if forecast_frame.shape[1] != 884:
        raise ValueError(
            f"Frozen Stage 9 forecast width differs: found {forecast_frame.shape[1]}"
        )
    return pd.concat([keys, forecast_frame], axis=1)


__all__ = [
    "OOFDomain",
    "assemble_elastic_cube",
    "assemble_stochastic_cube",
    "build_oof_domain",
    "build_wide_oof_table",
    "raw_column_name",
    "score_column_name",
]
