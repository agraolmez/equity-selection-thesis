"""Authenticated Stage 9 OOF adapter for Stage 11 score calibration.

The adapter reads only development-era OOF score columns selected by the
locked Stage 11 validation decision.  Model IDs are resolved through the
independently verified Stage 9 column registry retained by the execution-input
authority; numeric column ordinals are never guessed by a caller.
"""

from __future__ import annotations

import datetime as dt
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from portfolio_bot.validation.calibration import CalibrationObservation
from portfolio_bot.validation.execution_input_manifest import (
    AuthenticatedStage11ExecutionInputs,
)
from portfolio_bot.validation.execution_orchestrator import (
    ELASTIC_NET_CHALLENGER_ID,
    PRIMARY_STRATEGY_ID,
    LockedValidationSelection,
)
from portfolio_bot.validation.locked_selection import selection_frame_sha256


_PRESENT = "PRESENT"
_KEY_COLUMNS = ("signal_date", "holding_month", "permno")
_LABEL_COLUMNS = (
    "signal_date",
    "holding_month",
    "permno",
    "entry_date",
    "exit_date",
    "month_label_available_date",
    "unranked_residual_return",
    "primary_label_status",
)


class DevelopmentCalibrationError(RuntimeError):
    """Raised when authenticated OOF scores cannot form frozen calibration."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentCalibrationError(message)


def _date_series(values: pd.Series, *, label: str) -> pd.Series:
    try:
        result = pd.to_datetime(values, errors="raise").dt.date
    except (TypeError, ValueError) as exc:
        raise DevelopmentCalibrationError(f"{label} contains an invalid date") from exc
    _require(result.notna().all(), f"{label} contains a missing date")
    return result


def _selected_score_columns(
    execution_inputs: AuthenticatedStage11ExecutionInputs,
    selection: LockedValidationSelection,
) -> tuple[str, str, str, float]:
    families = selection.family_selections
    _require(
        set(families) == {"neural_network", "xgboost", "elastic_net"},
        "locked family selection set differs",
    )
    registry = execution_inputs.development_oof.score_column_by_model

    def resolve(family: str) -> str:
        key = (family, families[family].config_id)
        _require(key in registry, f"selected {family} is absent from the Stage 9 OOF registry")
        return registry[key]

    effective = selection.ensemble_selection.effective_nn_weight
    nn_weight = (
        selection.ensemble_selection.diagnostic_nn_weight
        if effective is None
        else effective
    )
    _require(
        math.isfinite(float(nn_weight)) and 0.0 <= float(nn_weight) <= 1.0,
        "selected ensemble weight is invalid",
    )
    return (
        resolve("neural_network"),
        resolve("xgboost"),
        resolve("elastic_net"),
        float(nn_weight),
    )


def _observations(
    frame: pd.DataFrame,
    *,
    strategy_id: str,
    score_column: str,
) -> tuple[CalibrationObservation, ...]:
    result: list[CalibrationObservation] = []
    for holding_month, month in frame.groupby(
        "holding_month", sort=True, observed=True
    ):
        payload = month.loc[
            :, ["signal_date", "holding_month", "permno", score_column]
        ].rename(columns={score_column: "score"})
        payload_sha256 = selection_frame_sha256(payload)
        for row in month.itertuples(index=False):
            score = float(getattr(row, score_column))
            residual = float(row.unranked_residual_return)
            _require(
                math.isfinite(score) and -0.5 <= score <= 0.5,
                "development calibration score is invalid",
            )
            _require(
                math.isfinite(residual),
                "development calibration residual return is invalid",
            )
            result.append(
                CalibrationObservation(
                    forecast_id=(
                        f"{strategy_id}:development_oof:{holding_month}:"
                        f"{int(row.permno)}"
                    ),
                    holding_month=str(holding_month),
                    permno=int(row.permno),
                    score=score,
                    residual_return=residual,
                    signal_date=row.signal_date,
                    entry_date=row.entry_date,
                    exit_date=row.exit_date,
                    forecast_payload_sha256=payload_sha256,
                    forecast_committed_at=dt.datetime.combine(
                        row.signal_date,
                        dt.time(12, 0),
                        tzinfo=dt.timezone.utc,
                    ),
                    outcome_available_at=dt.datetime.combine(
                        row.month_label_available_date,
                        dt.time(23, 59, 59),
                        tzinfo=dt.timezone.utc,
                    ),
                    source="development_oof",
                )
            )
    _require(bool(result), f"{strategy_id} development OOF calibration is empty")
    return tuple(result)


def build_authenticated_development_calibration_observations(
    execution_inputs: AuthenticatedStage11ExecutionInputs,
    selection: LockedValidationSelection,
) -> Mapping[str, Sequence[CalibrationObservation]]:
    """Build primary and Elastic-Net calibration histories from exact OOF rows."""

    _require(
        type(execution_inputs) is AuthenticatedStage11ExecutionInputs,
        "development calibration requires authenticated execution inputs",
    )
    _require(
        type(selection) is LockedValidationSelection,
        "development calibration requires the exact locked selection",
    )
    execution_inputs.gate()
    nn_column, xgb_column, enet_column, nn_weight = _selected_score_columns(
        execution_inputs, selection
    )
    oof_columns = [*_KEY_COLUMNS, nn_column, xgb_column, enet_column]
    oof = pq.read_table(
        execution_inputs.development_oof.path,
        columns=oof_columns,
        use_threads=False,
    ).to_pandas()
    execution_inputs.gate()
    _require(
        len(oof) == execution_inputs.development_oof.rows
        and not oof.duplicated(list(_KEY_COLUMNS)).any(),
        "development OOF row count or key uniqueness differs",
    )
    oof["signal_date"] = _date_series(oof["signal_date"], label="development OOF")
    oof["holding_month"] = oof["holding_month"].astype(str)
    raw_permno = pd.to_numeric(oof["permno"], errors="coerce").to_numpy(float)
    _require(
        np.isfinite(raw_permno).all()
        and np.all(raw_permno > 0.0)
        and np.all(raw_permno == np.floor(raw_permno)),
        "development OOF PERMNO domain differs",
    )
    oof["permno"] = raw_permno.astype(np.int64)
    for column in (nn_column, xgb_column, enet_column):
        values = pd.to_numeric(oof[column], errors="coerce").to_numpy(float)
        _require(
            np.isfinite(values).all() and np.all(np.abs(values) <= 0.5),
            f"development OOF score differs: {column}",
        )
        oof[column] = values
    oof["ensemble_score"] = (
        nn_weight * oof[nn_column] + (1.0 - nn_weight) * oof[xgb_column]
    )
    oof["elastic_net_score"] = oof[enet_column]

    labels = (
        execution_inputs.development_oof.stage9_inputs.read_development_labels()
        .select(list(_LABEL_COLUMNS))
        .to_pandas()
    )
    for column in (
        "signal_date",
        "entry_date",
        "exit_date",
        "month_label_available_date",
    ):
        labels[column] = _date_series(labels[column], label=f"development {column}")
    labels["holding_month"] = labels["holding_month"].astype(str)
    labels["permno"] = pd.to_numeric(labels["permno"], errors="raise").astype(np.int64)
    _require(
        len(labels) == len(oof)
        and not labels.duplicated(list(_KEY_COLUMNS)).any(),
        "development label row count or key uniqueness differs from OOF",
    )
    merged = oof.loc[
        :, [*_KEY_COLUMNS, "ensemble_score", "elastic_net_score"]
    ].merge(
        labels,
        on=list(_KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    _require(len(merged) == len(oof), "development OOF and label key domains differ")
    residual = pd.to_numeric(
        merged["unranked_residual_return"], errors="coerce"
    ).to_numpy(float)
    valid = merged["primary_label_status"].astype(str).eq(_PRESENT).to_numpy(bool)
    valid &= np.isfinite(residual)
    merged = merged.loc[valid].copy()
    merged["unranked_residual_return"] = residual[valid]
    merged = merged.sort_values(
        ["holding_month", "permno"], kind="mergesort"
    ).reset_index(drop=True)
    _require(
        not merged.duplicated(["holding_month", "permno"]).any(),
        "development calibration security-month domain is duplicated",
    )
    result = {
        PRIMARY_STRATEGY_ID: _observations(
            merged,
            strategy_id=PRIMARY_STRATEGY_ID,
            score_column="ensemble_score",
        ),
        ELASTIC_NET_CHALLENGER_ID: _observations(
            merged,
            strategy_id=ELASTIC_NET_CHALLENGER_ID,
            score_column="elastic_net_score",
        ),
    }
    execution_inputs.gate()
    return MappingProxyType(result)


__all__ = [
    "DevelopmentCalibrationError",
    "build_authenticated_development_calibration_observations",
]
