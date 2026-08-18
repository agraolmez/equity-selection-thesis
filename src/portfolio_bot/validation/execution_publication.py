"""Deterministic, resumable publication for a completed Stage 11 execution.

The empirical wrappers require a live unblind authority for their full scope.
The byte-level core is intentionally pure so the same closed bundle can be
exercised with synthetic fixtures before any validation outcome is opened.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.allocation.costs import transaction_cost_numpy
from portfolio_bot.numerical_materiality import (
    NumericalMaterialityError,
    is_immaterial_weight,
    side_compatible,
)
from portfolio_bot.validation.calendar import VALIDATION_HOLDING_MONTHS
from portfolio_bot.validation.execution_orchestrator import (
    DECISION_FOCUSED_CHALLENGER_ID,
    ELASTIC_NET_CHALLENGER_ID,
    LockedValidationSelection,
    PRIMARY_STRATEGY_ID,
    RANK_CHALLENGER_ID,
    Stage11ExecutionPlan,
    Stage11PseudoLiveLedger,
    StrategyCalibrationSchedule,
    canonical_frame_sha256,
)
from portfolio_bot.validation.locked_selection import (
    LOCKED_ENSEMBLE_NN_WEIGHTS,
    LOCKED_SHORTLIST,
    VALIDATION_STOCHASTIC_SEEDS,
)


PUBLICATION_ID = "stage11-validation-execution-publication-v1"
PRIMARY_STATUS = "STAGE11_VALIDATION_EXECUTION_COMPLETE"
VERIFIED_STATUS = "STAGE11_VALIDATION_EXECUTION_VERIFIED"

RAW_FORECASTS_NAME = "raw_forecasts.parquet"
OUTCOMES_NAME = "validation_residual_outcomes.parquet"
FAMILY_IC_NAME = "family_monthly_ic.parquet"
ENSEMBLE_IC_NAME = "ensemble_monthly_ic.parquet"
SELECTED_SCORES_NAME = "selected_scores.parquet"
CALIBRATION_NAME = "calibration_schedule.parquet"
EXECUTION_SESSIONS_NAME = "execution_sessions.parquet"
WEIGHTS_NAME = "portfolio_weights.parquet"
TRADES_NAME = "portfolio_trades.parquet"
DAILY_ACCOUNTING_NAME = "daily_accounting.parquet"
MONTHLY_METRICS_NAME = "monthly_metrics.parquet"
PLAN_NAME = "execution_plan.json"
SELECTION_NAME = "selection.json"
ACCESS_LEDGER_NAME = "access_ledger.json"
MANIFEST_NAME = "manifest.json"
SUCCESS_NAME = "_SUCCESS.json"

TABLE_NAMES = (
    RAW_FORECASTS_NAME,
    OUTCOMES_NAME,
    FAMILY_IC_NAME,
    ENSEMBLE_IC_NAME,
    SELECTED_SCORES_NAME,
    CALIBRATION_NAME,
    EXECUTION_SESSIONS_NAME,
    WEIGHTS_NAME,
    TRADES_NAME,
    DAILY_ACCOUNTING_NAME,
    MONTHLY_METRICS_NAME,
)
PRIMARY_MEMBERS = (
    *TABLE_NAMES,
    PLAN_NAME,
    SELECTION_NAME,
    ACCESS_LEDGER_NAME,
    MANIFEST_NAME,
    SUCCESS_NAME,
)
VERIFICATION_MEMBERS = ("verification_report.json", "differences.json", SUCCESS_NAME)

TABLE_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        RAW_FORECASTS_NAME: (
            "model_family",
            "config_id",
            "seed",
            "signal_date",
            "holding_month",
            "permno",
            "raw_forecast",
        ),
        OUTCOMES_NAME: (
            "signal_date",
            "holding_month",
            "entry_date",
            "exit_date",
            "outcome_availability_date",
            "permno",
            "unranked_residual_return",
            "residual_return_rank",
            "primary_label_status",
        ),
        FAMILY_IC_NAME: (
            "model_family",
            "config_id",
            "holding_month",
            "rank_ic",
        ),
        ENSEMBLE_IC_NAME: ("nn_weight", "holding_month", "rank_ic"),
        SELECTED_SCORES_NAME: (
            "signal_date",
            "holding_month",
            "permno",
            "ensemble_score",
            "elastic_net_score",
        ),
        CALIBRATION_NAME: (
            "fit_signal_date",
            "kappa",
            "unconstrained_slope",
            "observations",
            "distinct_months",
            "first_holding_month",
            "last_holding_month",
            "development_oof_observations",
            "validation_prior_live_observations",
            "equal_month_mass",
            "half_life_months",
            "huber_epsilon",
            "nonpositive_slope_truncated_to_zero",
            "strategy_id",
            "schedule_sha256",
        ),
        EXECUTION_SESSIONS_NAME: (
            "holding_month",
            "signal_date",
            "entry_date",
            "exit_date",
            "date",
        ),
        WEIGHTS_NAME: (
            "strategy_id",
            "holding_month",
            "signal_date",
            "entry_date",
            "exit_date",
            "permno",
            "previous_weight",
            "weight",
            "score",
            "expected_return",
            "side",
            "required_exit",
            "solver_status",
            "solver_mode",
        ),
        TRADES_NAME: (
            "strategy_id",
            "holding_month",
            "signal_date",
            "entry_date",
            "exit_date",
            "permno",
            "previous_weight",
            "target_weight",
            "trade_weight",
            "trade_dollars",
            "execution_price",
            "full_spread",
            "adv20_usd",
            "impact_volatility_daily",
            "trade_capacity_weight",
            "binding_trade_limit",
        ),
        DAILY_ACCOUNTING_NAME: (
            "strategy_id",
            "holding_month",
            "signal_date",
            "entry_date",
            "exit_date",
            "date",
            "opening_nav_dollars",
            "nav_dollars",
            "net_total_return",
            "risk_free_return",
            "net_excess_return",
            "transaction_cost_dollars",
            "borrow_cost_dollars",
            "financing_cost_dollars",
            "opening_state_sha256",
            "closing_state_sha256",
            "corporate_action_ids",
        ),
        MONTHLY_METRICS_NAME: (
            "strategy_id",
            "holding_month",
            "signal_date",
            "entry_date",
            "exit_date",
            "pre_trade_nav_dollars",
            "closing_nav_dollars",
            "net_total_return",
            "risk_free_total_return",
            "net_excess_return",
            "transaction_cost_dollars",
            "borrow_cost_dollars",
            "financing_cost_dollars",
            "turnover",
            "gross_exposure",
            "net_exposure",
            "cash_weight",
            "position_count",
            "solver_status",
            "solver_mode",
            "objective_value",
        ),
    }
)

_MAX_SIGNAL_DATE = pd.Timestamp("2019-10-31")
_MAX_OBSERVATION_DATE = pd.Timestamp("2019-12-02")


class Stage11ExecutionPublicationError(RuntimeError):
    """Raised when an execution bundle is unsafe, changed, or incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11ExecutionPublicationError(message)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_root(path: Path) -> Path:
    _require(
        isinstance(path, Path) and path.is_absolute(),
        "publication root must be absolute",
    )
    lexical = Path(os.path.abspath(path.expanduser()))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor /= part
        if cursor.exists():
            metadata = os.lstat(cursor)
            _require(
                not stat.S_ISLNK(metadata.st_mode),
                "symlinked publication path is prohibited",
            )
    lexical.mkdir(parents=True, exist_ok=True)
    _require(
        lexical.is_dir() and not lexical.is_symlink(), "publication root is unsafe"
    )
    return lexical


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        row_group_size=100_000,
        data_page_version="1.0",
    )
    return sink.getvalue().to_pybytes()


def _atomic_file(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _require(not temporary.exists(), "stale publication temporary file exists")
    try:
        with temporary.open("xb") as stream:
            written = stream.write(payload)
            _require(written == len(payload), "short publication write")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _normalize_date_scope(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    result = frame.copy(deep=True)
    if "holding_month" in result.columns:
        months = tuple(sorted(set(result["holding_month"].dropna().astype(str))))
        _require(
            all(value in VALIDATION_HOLDING_MONTHS for value in months),
            f"{label} contains an embargoed or out-of-scope holding month",
        )
    for column in result.columns:
        if column == "signal_date" or column.endswith("_date") or column == "date":
            if result.empty or result[column].dropna().empty:
                continue
            try:
                values = pd.to_datetime(result[column].dropna(), errors="raise")
            except (TypeError, ValueError) as exc:
                raise Stage11ExecutionPublicationError(
                    f"{label} has an invalid date column: {column}"
                ) from exc
            maximum = (
                _MAX_SIGNAL_DATE if column == "signal_date" else _MAX_OBSERVATION_DATE
            )
            _require(
                values.max() <= maximum,
                f"{label} date scope crossed Stage 11: {column}",
            )
    return result


def _calibration_frame(
    schedules: Mapping[str, StrategyCalibrationSchedule],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for strategy_id, schedule in sorted(schedules.items()):
        _require(
            strategy_id == schedule.strategy_id, "calibration schedule key differs"
        )
        for receipt in schedule.receipts:
            record = asdict(receipt)
            record["strategy_id"] = strategy_id
            record["schedule_sha256"] = schedule.schedule_sha256
            records.append(record)
    result = pd.DataFrame.from_records(records)
    _require(not result.empty, "calibration schedule table cannot be empty")
    return result.sort_values(
        ["strategy_id", "fit_signal_date"], kind="mergesort"
    ).reset_index(drop=True)


def _selection_payload(selection: LockedValidationSelection) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selection_sha256": selection.selection_sha256,
        "provenance": asdict(selection.provenance),
        "families": {
            family: {
                **asdict(value),
                "provenance": asdict(value.provenance),
            }
            for family, value in sorted(selection.family_selections.items())
        },
        "ensemble": {
            **asdict(selection.ensemble_selection),
            "provenance": asdict(selection.ensemble_selection.provenance),
        },
    }


def _semantic_require(condition: bool, code: str) -> None:
    _require(condition, f"execution semantic verification failed: {code}")


def _semantic_dates(frame: pd.DataFrame, column: str, *, code: str) -> pd.Series:
    try:
        values = pd.to_datetime(frame[column], errors="raise").dt.date
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage11ExecutionPublicationError(
            f"execution semantic verification failed: {code}"
        ) from exc
    _semantic_require(not values.isna().any(), code)
    return values


def _semantic_numeric(
    frame: pd.DataFrame,
    column: str,
    *,
    code: str,
    finite: bool = True,
) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage11ExecutionPublicationError(
            f"execution semantic verification failed: {code}"
        ) from exc
    if finite:
        _semantic_require(np.isfinite(values).all(), code)
    else:
        _semantic_require(not np.isinf(values).any(), code)
    return values


def _semantic_integers(
    frame: pd.DataFrame,
    column: str,
    *,
    code: str,
    positive: bool = False,
) -> np.ndarray:
    values = _semantic_numeric(frame, column, code=code)
    _semantic_require(np.equal(values, np.floor(values)).all(), code)
    if positive:
        _semantic_require(np.all(values > 0.0), code)
    return values.astype(np.int64)


def _semantic_close(
    left: Sequence[object] | np.ndarray,
    right: Sequence[object] | np.ndarray,
    *,
    code: str,
    equal_nan: bool = False,
) -> None:
    try:
        left_values = np.asarray(left, dtype=np.float64)
        right_values = np.asarray(right, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Stage11ExecutionPublicationError(
            f"execution semantic verification failed: {code}"
        ) from exc
    _semantic_require(
        left_values.shape == right_values.shape
        and np.allclose(
            left_values,
            right_values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=equal_nan,
        ),
        code,
    )


def _calendar_contract(
    plan_payload: Mapping[str, Any],
) -> tuple[Mapping[str, Mapping[str, dt.date]], tuple[dt.date, ...]]:
    rows = plan_payload.get("calendar_rows")
    _semantic_require(isinstance(rows, list), "plan_calendar_schema")
    _semantic_require(
        len(rows) == len(VALIDATION_HOLDING_MONTHS), "plan_calendar_count"
    )
    parsed: dict[str, Mapping[str, dt.date]] = {}
    for expected_month, raw in zip(VALIDATION_HOLDING_MONTHS, rows, strict=True):
        _semantic_require(
            isinstance(raw, Mapping)
            and set(raw) == {"holding_month", "signal_date", "entry_date", "exit_date"}
            and raw.get("holding_month") == expected_month,
            "plan_calendar_identity",
        )
        try:
            signal = dt.date.fromisoformat(str(raw["signal_date"]))
            entry = dt.date.fromisoformat(str(raw["entry_date"]))
            exit_date = dt.date.fromisoformat(str(raw["exit_date"]))
        except (TypeError, ValueError) as exc:
            raise Stage11ExecutionPublicationError(
                "execution semantic verification failed: plan_calendar_dates"
            ) from exc
        _semantic_require(signal < entry < exit_date, "plan_calendar_order")
        parsed[expected_month] = {
            "signal_date": signal,
            "entry_date": entry,
            "exit_date": exit_date,
        }
    ordered = tuple(parsed[value] for value in VALIDATION_HOLDING_MONTHS)
    _semantic_require(
        all(
            left["exit_date"] == right["entry_date"]
            for left, right in zip(ordered, ordered[1:])
        ),
        "plan_calendar_continuity",
    )
    raw_refits = plan_payload.get("refit_signal_dates")
    _semantic_require(
        isinstance(raw_refits, list) and len(raw_refits) == 20, "plan_refit_count"
    )
    try:
        refits = tuple(dt.date.fromisoformat(str(value)) for value in raw_refits)
    except ValueError as exc:
        raise Stage11ExecutionPublicationError(
            "execution semantic verification failed: plan_refit_dates"
        ) from exc
    _semantic_require(len(set(refits)) == 20, "plan_refit_dates")
    return MappingProxyType(parsed), refits


def _active_strategy_ids(selection_payload: Mapping[str, Any]) -> tuple[str, ...]:
    ensemble = selection_payload.get("ensemble")
    _semantic_require(isinstance(ensemble, Mapping), "selection_ensemble_schema")
    qualifies = ensemble.get("active_v1_ml_strategy_qualifies")
    _semantic_require(type(qualifies) is bool, "selection_active_gate")
    return (
        (ELASTIC_NET_CHALLENGER_ID, PRIMARY_STRATEGY_ID, RANK_CHALLENGER_ID)
        if qualifies
        else (ELASTIC_NET_CHALLENGER_ID,)
    )


def _architecture_status(selection_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the closed four-architecture execution status record."""

    ensemble = selection_payload.get("ensemble")
    _semantic_require(isinstance(ensemble, Mapping), "selection_ensemble_schema")
    qualifies = ensemble.get("active_v1_ml_strategy_qualifies")
    _semantic_require(type(qualifies) is bool, "selection_active_gate")
    inactive = None if qualifies else "both_nonlinear_families_failed"
    return {
        "schema_version": 1,
        "nonlinear_annual_gate_qualifies": qualifies,
        "decision_focused_fit_performed": False,
        "architectures": [
            {
                "architecture_id": PRIMARY_STRATEGY_ID,
                "active": qualifies,
                "inactive_reason": inactive,
            },
            {
                "architecture_id": ELASTIC_NET_CHALLENGER_ID,
                "active": True,
                "inactive_reason": None,
            },
            {
                "architecture_id": RANK_CHALLENGER_ID,
                "active": qualifies,
                "inactive_reason": inactive,
            },
            {
                "architecture_id": DECISION_FOCUSED_CHALLENGER_ID,
                "active": False,
                "inactive_reason": (
                    "production_training_runtime_not_certified"
                    if qualifies
                    else "nonlinear_family_gate_failed"
                ),
            },
        ],
    }


def _validate_calendar_columns(
    frame: pd.DataFrame,
    calendar: Mapping[str, Mapping[str, dt.date]],
    *,
    label: str,
    columns: tuple[str, ...],
) -> None:
    months = frame["holding_month"].astype(str)
    _semantic_require(months.notna().all(), f"{label}_holding_month_null")
    _semantic_require(set(months).issubset(calendar), f"{label}_holding_month_scope")
    for column in columns:
        observed = _semantic_dates(frame, column, code=f"{label}_{column}_date")
        expected = months.map(lambda value: calendar[value][column])
        _semantic_require(
            np.array_equal(observed.to_numpy(object), expected.to_numpy(object)),
            f"{label}_{column}_calendar",
        )


def _selected_component_scores(
    raw: pd.DataFrame,
    *,
    family: str,
    config_id: str,
    score_name: str,
) -> pd.DataFrame:
    selected = raw.loc[
        raw["model_family"].astype(str).eq(family)
        & raw["config_id"].astype(str).eq(config_id),
        ["signal_date", "holding_month", "permno", "raw_forecast"],
    ].copy()
    _semantic_require(not selected.empty, f"selected_{family}_forecast_absent")
    grouped = selected.groupby(
        ["signal_date", "holding_month", "permno"],
        sort=False,
        observed=True,
        as_index=False,
    )["raw_forecast"].mean()
    ranks = grouped.groupby("holding_month", sort=False, observed=True)[
        "raw_forecast"
    ].rank(method="average", ascending=True)
    counts = grouped.groupby("holding_month", sort=False, observed=True)[
        "raw_forecast"
    ].transform("size")
    grouped[score_name] = ((ranks - 0.5) / counts.astype(float)) - 0.5
    return grouped.drop(columns="raw_forecast")


def _calibration_schedule_digest(frame: pd.DataFrame) -> str:
    rows: list[dict[str, object]] = []
    ordered = frame.sort_values("fit_signal_date", kind="mergesort")
    for row in ordered.itertuples(index=False):
        rows.append(
            {
                "fit_signal_date": row.fit_signal_date.isoformat(),
                "kappa": float(row.kappa).hex(),
                "unconstrained_slope": float(row.unconstrained_slope).hex(),
                "observations": int(row.observations),
                "distinct_months": int(row.distinct_months),
                "first_holding_month": str(row.first_holding_month),
                "last_holding_month": str(row.last_holding_month),
                "development_oof_observations": int(row.development_oof_observations),
                "validation_prior_live_observations": int(
                    row.validation_prior_live_observations
                ),
                "equal_month_mass": bool(row.equal_month_mass),
                "half_life_months": float(row.half_life_months).hex(),
                "huber_epsilon": float(row.huber_epsilon).hex(),
                "nonpositive_slope_truncated_to_zero": bool(
                    row.nonpositive_slope_truncated_to_zero
                ),
            }
        )
    return _sha_bytes(_canonical_json(rows))


def _session_authority_digest(
    *,
    holding_month: str,
    calendar: Mapping[str, dt.date],
    return_dates: Sequence[dt.date],
    source_payload_sha256: str,
) -> str:
    digest = hashlib.sha256(b"stage11-holding-period-session-authority-v1\0")
    for value in (
        source_payload_sha256,
        holding_month,
        calendar["signal_date"].isoformat(),
        calendar["entry_date"].isoformat(),
        calendar["exit_date"].isoformat(),
    ):
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    for value in return_dates:
        digest.update(value.isoformat().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_execution_semantics(
    tables: Mapping[str, pd.DataFrame],
    *,
    plan_payload: Mapping[str, Any],
    selection_payload: Mapping[str, Any],
    expected_calibration_hashes: Mapping[str, str],
    expected_session_receipts: Mapping[str, Mapping[str, str]],
    expected_architecture_status: Mapping[str, Any],
) -> None:
    """Independently reconcile the logical execution tables.

    This deliberately does not trust the producer's in-memory accounting
    receipts.  It operates on the same logical frames that are serialized, and
    the verifier invokes it again after reopening the Parquet members.
    """

    _semantic_require(set(tables) == set(TABLE_NAMES), "table_set")
    for name in TABLE_NAMES:
        frame = tables[name]
        _semantic_require(isinstance(frame, pd.DataFrame), f"{name}_type")
        _semantic_require(
            tuple(str(value) for value in frame.columns) == TABLE_COLUMNS[name],
            f"{name}_schema",
        )
        _semantic_require(not frame.empty, f"{name}_empty")

    calendar, refit_dates = _calendar_contract(plan_payload)
    active_ids = _active_strategy_ids(selection_payload)
    _semantic_require(
        dict(expected_architecture_status) == _architecture_status(selection_payload),
        "architecture_status",
    )
    active_set = set(active_ids)
    expected_pairs = {
        (strategy_id, holding_month)
        for strategy_id in active_ids
        for holding_month in VALIDATION_HOLDING_MONTHS
    }

    sessions = tables[EXECUTION_SESSIONS_NAME].copy()
    sessions["holding_month"] = sessions["holding_month"].astype(str)
    _validate_calendar_columns(
        sessions,
        calendar,
        label="execution_session",
        columns=("signal_date", "entry_date", "exit_date"),
    )
    sessions["date"] = _semantic_dates(sessions, "date", code="execution_session_date")
    _semantic_require(
        not sessions.duplicated(["holding_month", "date"], keep=False).any(),
        "execution_session_key",
    )
    _semantic_require(
        tuple(sorted(sessions["holding_month"].unique())) == VALIDATION_HOLDING_MONTHS,
        "execution_session_month_coverage",
    )
    _semantic_require(
        set(expected_session_receipts) == set(VALIDATION_HOLDING_MONTHS),
        "execution_session_receipt_set",
    )
    expected_sessions: dict[str, tuple[dt.date, ...]] = {}
    for holding_month in VALIDATION_HOLDING_MONTHS:
        group = sessions.loc[sessions["holding_month"].eq(holding_month)].sort_values(
            "date", kind="mergesort"
        )
        dates = tuple(group["date"])
        period = calendar[holding_month]
        _semantic_require(
            bool(dates)
            and dates[0] > period["entry_date"]
            and dates[-1] == period["exit_date"]
            and all(left < right for left, right in zip(dates, dates[1:]))
            and all(value.weekday() < 5 for value in dates),
            "execution_session_scope",
        )
        receipt = expected_session_receipts[holding_month]
        _semantic_require(
            isinstance(receipt, Mapping)
            and set(receipt) == {"source_payload_sha256", "authority_sha256"},
            "execution_session_receipt_schema",
        )
        source_sha = str(receipt.get("source_payload_sha256"))
        authority_sha = str(receipt.get("authority_sha256"))
        _semantic_require(
            len(source_sha) == len(authority_sha) == 64
            and all(
                value in "0123456789abcdef" for value in source_sha + authority_sha
            ),
            "execution_session_receipt_digest",
        )
        _semantic_require(
            _session_authority_digest(
                holding_month=holding_month,
                calendar=period,
                return_dates=dates,
                source_payload_sha256=source_sha,
            )
            == authority_sha,
            "execution_session_authority",
        )
        expected_sessions[holding_month] = dates

    raw = tables[RAW_FORECASTS_NAME].copy()
    raw["holding_month"] = raw["holding_month"].astype(str)
    raw["signal_date"] = _semantic_dates(raw, "signal_date", code="raw_signal_date")
    raw["permno"] = _semantic_integers(raw, "permno", code="raw_permno", positive=True)
    raw["raw_forecast"] = _semantic_numeric(raw, "raw_forecast", code="raw_forecast")
    _semantic_require(
        not raw.duplicated(
            ["model_family", "config_id", "seed", "holding_month", "permno"],
            keep=False,
        ).any(),
        "raw_forecast_key",
    )
    _semantic_require(
        tuple(sorted(raw["holding_month"].unique())) == VALIDATION_HOLDING_MONTHS,
        "raw_forecast_month_coverage",
    )
    _validate_calendar_columns(
        raw,
        calendar,
        label="raw_forecast",
        columns=("signal_date",),
    )
    allowed_grid = {
        (family, config_id, seed)
        for family, config_ids in LOCKED_SHORTLIST.items()
        for config_id in config_ids
        for seed in (
            (None,) if family == "elastic_net" else VALIDATION_STOCHASTIC_SEEDS
        )
    }
    observed_grid: set[tuple[str, str, int | None]] = set()
    for row in (
        raw[["model_family", "config_id", "seed"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        if pd.isna(row.seed):
            seed: int | None = None
        else:
            numeric_seed = float(row.seed)
            _semantic_require(numeric_seed.is_integer(), "raw_forecast_seed")
            seed = int(numeric_seed)
        observed_grid.add((str(row.model_family), str(row.config_id), seed))
    _semantic_require(observed_grid == allowed_grid, "raw_forecast_model_grid")
    grid_counts = raw.groupby(
        ["holding_month", "permno"], sort=False, observed=True
    ).size()
    _semantic_require(
        bool(len(grid_counts)) and bool(grid_counts.eq(len(allowed_grid)).all()),
        "raw_forecast_security_grid",
    )
    raw_domain = raw[["signal_date", "holding_month", "permno"]].drop_duplicates()

    outcomes = tables[OUTCOMES_NAME].copy()
    outcomes["holding_month"] = outcomes["holding_month"].astype(str)
    for column in (
        "signal_date",
        "entry_date",
        "exit_date",
        "outcome_availability_date",
    ):
        outcomes[column] = _semantic_dates(outcomes, column, code=f"outcome_{column}")
    outcomes["permno"] = _semantic_integers(
        outcomes, "permno", code="outcome_permno", positive=True
    )
    _semantic_require(
        not outcomes.duplicated(["holding_month", "permno"], keep=False).any(),
        "outcome_key",
    )
    _validate_calendar_columns(
        outcomes,
        calendar,
        label="outcome",
        columns=("signal_date", "entry_date", "exit_date"),
    )
    _semantic_require(
        bool((outcomes["outcome_availability_date"] >= outcomes["exit_date"]).all()),
        "outcome_availability",
    )
    outcome_domain = outcomes[["signal_date", "holding_month", "permno"]]
    _semantic_require(
        raw_domain.sort_values(["holding_month", "permno"], kind="mergesort")
        .reset_index(drop=True)
        .equals(
            outcome_domain.sort_values(
                ["holding_month", "permno"], kind="mergesort"
            ).reset_index(drop=True)
        ),
        "forecast_outcome_domain",
    )
    outcome_value = _semantic_numeric(
        outcomes,
        "unranked_residual_return",
        code="outcome_unranked_return",
        finite=False,
    )
    outcome_rank = _semantic_numeric(
        outcomes,
        "residual_return_rank",
        code="outcome_residual_rank",
        finite=False,
    )
    present = outcomes["primary_label_status"].astype(str).eq("PRESENT").to_numpy(bool)
    _semantic_require(
        np.isfinite(outcome_value[present]).all()
        and np.isfinite(outcome_rank[present]).all()
        and np.all(np.abs(outcome_rank[present]) <= 0.5 + 1e-12),
        "outcome_present_values",
    )

    family_ic = tables[FAMILY_IC_NAME].copy()
    family_ic["holding_month"] = family_ic["holding_month"].astype(str)
    expected_family_keys = {
        (family, config_id, month)
        for family, configs in LOCKED_SHORTLIST.items()
        for config_id in configs
        for month in VALIDATION_HOLDING_MONTHS
    }
    observed_family_keys = set(
        family_ic[["model_family", "config_id", "holding_month"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    _semantic_require(
        len(family_ic) == len(expected_family_keys)
        and not family_ic.duplicated(
            ["model_family", "config_id", "holding_month"], keep=False
        ).any()
        and observed_family_keys == expected_family_keys,
        "family_ic_coverage",
    )
    _semantic_numeric(family_ic, "rank_ic", code="family_ic_value", finite=False)

    ensemble_ic = tables[ENSEMBLE_IC_NAME].copy()
    ensemble_ic["holding_month"] = ensemble_ic["holding_month"].astype(str)
    ensemble_weights = _semantic_numeric(
        ensemble_ic, "nn_weight", code="ensemble_ic_weight"
    )
    ensemble_ic["nn_weight"] = ensemble_weights
    expected_ensemble_keys = {
        (float(weight), month)
        for weight in LOCKED_ENSEMBLE_NN_WEIGHTS
        for month in VALIDATION_HOLDING_MONTHS
    }
    observed_ensemble_keys = set(
        ensemble_ic[["nn_weight", "holding_month"]].itertuples(index=False, name=None)
    )
    _semantic_require(
        len(ensemble_ic) == len(expected_ensemble_keys)
        and not ensemble_ic.duplicated(["nn_weight", "holding_month"], keep=False).any()
        and observed_ensemble_keys == expected_ensemble_keys,
        "ensemble_ic_coverage",
    )
    _semantic_numeric(ensemble_ic, "rank_ic", code="ensemble_ic_value", finite=False)

    selected = tables[SELECTED_SCORES_NAME].copy()
    selected["holding_month"] = selected["holding_month"].astype(str)
    selected["signal_date"] = _semantic_dates(
        selected, "signal_date", code="selected_signal_date"
    )
    selected["permno"] = _semantic_integers(
        selected, "permno", code="selected_permno", positive=True
    )
    _semantic_require(
        not selected.duplicated(["holding_month", "permno"], keep=False).any(),
        "selected_score_key",
    )
    _validate_calendar_columns(
        selected,
        calendar,
        label="selected_score",
        columns=("signal_date",),
    )
    _semantic_require(
        selected[["signal_date", "holding_month", "permno"]]
        .sort_values(["holding_month", "permno"], kind="mergesort")
        .reset_index(drop=True)
        .equals(
            outcome_domain.sort_values(
                ["holding_month", "permno"], kind="mergesort"
            ).reset_index(drop=True)
        ),
        "selected_score_domain",
    )
    families = selection_payload.get("families")
    _semantic_require(
        isinstance(families, Mapping) and set(families) == set(LOCKED_SHORTLIST),
        "selection_family_schema",
    )
    selected_config_ids: dict[str, str] = {}
    for family in LOCKED_SHORTLIST:
        record = families[family]
        _semantic_require(isinstance(record, Mapping), "selection_family_schema")
        config_id = str(record.get("config_id"))
        _semantic_require(
            config_id in LOCKED_SHORTLIST[family], "selection_family_config"
        )
        selected_config_ids[family] = config_id
    enet_scores = _selected_component_scores(
        raw,
        family="elastic_net",
        config_id=selected_config_ids["elastic_net"],
        score_name="expected_elastic_net_score",
    )
    score_check = selected.merge(
        enet_scores,
        on=["signal_date", "holding_month", "permno"],
        how="left",
        validate="one_to_one",
    )
    _semantic_close(
        score_check["elastic_net_score"],
        score_check["expected_elastic_net_score"],
        code="selected_elastic_net_score",
    )
    qualifies = len(active_ids) == 3
    if qualifies:
        nn_scores = _selected_component_scores(
            raw,
            family="neural_network",
            config_id=selected_config_ids["neural_network"],
            score_name="nn_score",
        )
        xgb_scores = _selected_component_scores(
            raw,
            family="xgboost",
            config_id=selected_config_ids["xgboost"],
            score_name="xgb_score",
        )
        score_check = score_check.merge(
            nn_scores,
            on=["signal_date", "holding_month", "permno"],
            how="left",
            validate="one_to_one",
        ).merge(
            xgb_scores,
            on=["signal_date", "holding_month", "permno"],
            how="left",
            validate="one_to_one",
        )
        ensemble = selection_payload["ensemble"]
        weight = float(ensemble.get("effective_nn_weight"))
        _semantic_require(
            weight in LOCKED_ENSEMBLE_NN_WEIGHTS, "selection_ensemble_weight"
        )
        expected_ensemble = weight * score_check["nn_score"].to_numpy(float) + (
            1.0 - weight
        ) * score_check["xgb_score"].to_numpy(float)
        _semantic_close(
            score_check["ensemble_score"],
            expected_ensemble,
            code="selected_ensemble_score",
        )
    else:
        values = _semantic_numeric(
            score_check,
            "ensemble_score",
            code="inactive_ensemble_score",
            finite=False,
        )
        _semantic_require(np.isnan(values).all(), "inactive_ensemble_score")

    calibration = tables[CALIBRATION_NAME].copy()
    calibration["fit_signal_date"] = _semantic_dates(
        calibration, "fit_signal_date", code="calibration_fit_date"
    )
    _semantic_require(
        set(calibration["strategy_id"].astype(str))
        == {PRIMARY_STRATEGY_ID, ELASTIC_NET_CHALLENGER_ID},
        "calibration_strategy_set",
    )
    _semantic_require(
        not calibration.duplicated(
            ["strategy_id", "fit_signal_date"], keep=False
        ).any(),
        "calibration_key",
    )
    _semantic_require(
        set(expected_calibration_hashes)
        == {PRIMARY_STRATEGY_ID, ELASTIC_NET_CHALLENGER_ID},
        "calibration_manifest_set",
    )
    for strategy_id, group in calibration.groupby(
        "strategy_id", sort=True, observed=True
    ):
        group = group.copy()
        _semantic_require(
            tuple(sorted(group["fit_signal_date"])) == tuple(sorted(refit_dates)),
            "calibration_refit_coverage",
        )
        kappa = _semantic_numeric(group, "kappa", code="calibration_kappa")
        _semantic_require(np.all(kappa >= 0.0), "calibration_kappa")
        unconstrained = _semantic_numeric(
            group,
            "unconstrained_slope",
            code="calibration_unconstrained_slope",
        )
        half_life = _semantic_numeric(
            group, "half_life_months", code="calibration_half_life"
        )
        epsilon = _semantic_numeric(
            group, "huber_epsilon", code="calibration_huber_epsilon"
        )
        _semantic_require(
            np.all(half_life > 0.0)
            and np.all(epsilon > 1.0)
            and np.isfinite(unconstrained).all(),
            "calibration_numeric_bounds",
        )
        _semantic_require(
            group["equal_month_mass"].map(type).isin((bool, np.bool_)).all()
            and group["equal_month_mass"].astype(bool).all()
            and group["nonpositive_slope_truncated_to_zero"]
            .map(type)
            .isin((bool, np.bool_))
            .all(),
            "calibration_boolean_fields",
        )
        _semantic_require(
            group["first_holding_month"].astype(str).str.fullmatch(r"\d{4}-\d{2}").all()
            and group["last_holding_month"]
            .astype(str)
            .str.fullmatch(r"\d{4}-\d{2}")
            .all(),
            "calibration_holding_months",
        )
        for column in (
            "observations",
            "distinct_months",
            "development_oof_observations",
            "validation_prior_live_observations",
        ):
            values = _semantic_integers(group, column, code=f"calibration_{column}")
            _semantic_require(np.all(values >= 0), f"calibration_{column}")
        hashes = set(group["schedule_sha256"].astype(str))
        expected_hash = str(expected_calibration_hashes[str(strategy_id)])
        _semantic_require(
            hashes == {expected_hash}
            and len(expected_hash) == 64
            and _calibration_schedule_digest(group) == expected_hash,
            "calibration_schedule_digest",
        )

    weights = tables[WEIGHTS_NAME].copy()
    trades = tables[TRADES_NAME].copy()
    daily = tables[DAILY_ACCOUNTING_NAME].copy()
    monthly = tables[MONTHLY_METRICS_NAME].copy()
    for label, frame in (
        ("weight", weights),
        ("trade", trades),
        ("daily", daily),
        ("monthly", monthly),
    ):
        frame["holding_month"] = frame["holding_month"].astype(str)
        frame["strategy_id"] = frame["strategy_id"].astype(str)
        _semantic_require(
            set(frame["strategy_id"]) == active_set,
            f"{label}_strategy_set",
        )
        _validate_calendar_columns(
            frame,
            calendar,
            label=label,
            columns=("signal_date", "entry_date", "exit_date"),
        )
        observed_pairs = set(
            frame[["strategy_id", "holding_month"]].itertuples(index=False, name=None)
        )
        _semantic_require(observed_pairs == expected_pairs, f"{label}_month_coverage")

    weights["permno"] = _semantic_integers(
        weights, "permno", code="weight_permno", positive=True
    )
    trades["permno"] = _semantic_integers(
        trades, "permno", code="trade_permno", positive=True
    )
    position_key = ["strategy_id", "holding_month", "permno"]
    _semantic_require(
        not weights.duplicated(position_key, keep=False).any(), "weight_key"
    )
    _semantic_require(
        not trades.duplicated(position_key, keep=False).any(), "trade_key"
    )
    _semantic_require(
        set(weights[position_key].itertuples(index=False, name=None))
        == set(trades[position_key].itertuples(index=False, name=None)),
        "weight_trade_keys",
    )
    for column in ("previous_weight", "weight"):
        weights[column] = _semantic_numeric(weights, column, code=f"weight_{column}")
    weights["score"] = _semantic_numeric(
        weights, "score", code="weight_score", finite=False
    )
    weights["expected_return"] = _semantic_numeric(
        weights, "expected_return", code="weight_expected_return", finite=False
    )
    weights["side"] = _semantic_integers(weights, "side", code="weight_side")
    _semantic_require(np.isin(weights["side"], (-1, 0, 1)).all(), "weight_side")
    _semantic_require(
        weights["required_exit"].map(type).isin((bool, np.bool_)).all(),
        "weight_required_exit",
    )
    _semantic_require(
        weights["solver_status"].notna().all()
        and weights["solver_status"].astype(str).str.len().gt(0).all()
        and weights["solver_mode"].notna().all()
        and weights["solver_mode"].astype(str).str.len().gt(0).all(),
        "weight_solver_identity",
    )

    for column in (
        "previous_weight",
        "target_weight",
        "trade_weight",
        "trade_dollars",
        "execution_price",
        "full_spread",
        "adv20_usd",
        "impact_volatility_daily",
        "trade_capacity_weight",
    ):
        trades[column] = _semantic_numeric(
            trades,
            column,
            code=f"trade_{column}",
            finite=column
            in {
                "previous_weight",
                "target_weight",
                "trade_weight",
                "trade_dollars",
                "trade_capacity_weight",
            },
        )
    _semantic_require(
        trades["binding_trade_limit"].map(type).isin((bool, np.bool_)).all(),
        "trade_binding_limit",
    )

    monthly_key = ["strategy_id", "holding_month"]
    _semantic_require(
        len(monthly) == len(expected_pairs)
        and not monthly.duplicated(monthly_key, keep=False).any(),
        "monthly_key",
    )
    monthly_numeric = (
        "pre_trade_nav_dollars",
        "closing_nav_dollars",
        "net_total_return",
        "risk_free_total_return",
        "net_excess_return",
        "transaction_cost_dollars",
        "borrow_cost_dollars",
        "financing_cost_dollars",
        "turnover",
        "gross_exposure",
        "net_exposure",
        "cash_weight",
        "objective_value",
    )
    for column in monthly_numeric:
        monthly[column] = _semantic_numeric(monthly, column, code=f"monthly_{column}")
    monthly["position_count"] = _semantic_integers(
        monthly, "position_count", code="monthly_position_count"
    )
    _semantic_require(
        np.all(monthly["pre_trade_nav_dollars"] > 0.0)
        and np.all(monthly["closing_nav_dollars"] > 0.0)
        and np.all(monthly["transaction_cost_dollars"] >= 0.0)
        and np.all(monthly["borrow_cost_dollars"] >= 0.0)
        and np.all(monthly["financing_cost_dollars"] >= 0.0)
        and np.all(monthly["turnover"] >= 0.0)
        and np.all(monthly["gross_exposure"] >= 0.0)
        and np.all(monthly["position_count"] >= 0),
        "monthly_numeric_bounds",
    )
    _semantic_close(
        monthly["net_total_return"],
        monthly["closing_nav_dollars"] / monthly["pre_trade_nav_dollars"] - 1.0,
        code="monthly_nav_return",
    )
    _semantic_close(
        monthly["net_excess_return"],
        monthly["net_total_return"] - monthly["risk_free_total_return"],
        code="monthly_excess_return",
    )
    _semantic_close(
        monthly["cash_weight"] + monthly["net_exposure"],
        np.ones(len(monthly)),
        code="monthly_cash_net_identity",
    )
    _semantic_require(
        np.all(monthly["gross_exposure"] + 1e-10 >= np.abs(monthly["net_exposure"])),
        "monthly_gross_net_identity",
    )
    _semantic_require(
        monthly["solver_status"].notna().all() and monthly["solver_mode"].notna().all(),
        "monthly_solver_identity",
    )

    joined = weights.merge(
        trades,
        on=position_key,
        how="inner",
        validate="one_to_one",
        suffixes=("_weight", "_trade"),
    ).merge(
        monthly[
            [
                "strategy_id",
                "holding_month",
                "pre_trade_nav_dollars",
                "turnover",
                "solver_status",
                "solver_mode",
            ]
        ],
        on=monthly_key,
        how="left",
        validate="many_to_one",
    )
    _semantic_require(len(joined) == len(weights), "weight_trade_join")
    try:
        weight_side_compatible = side_compatible(
            joined["weight"].to_numpy(np.float64),
            joined["side"].to_numpy(np.int64),
            joined["pre_trade_nav_dollars"].to_numpy(np.float64),
        )
    except NumericalMaterialityError as exc:
        raise Stage11ExecutionPublicationError(
            "execution semantic verification failed: weight_side_sign"
        ) from exc
    _semantic_require(
        np.all(weight_side_compatible),
        "weight_side_sign",
    )
    for column in ("signal_date", "entry_date", "exit_date"):
        _semantic_require(
            np.array_equal(
                _semantic_dates(joined, f"{column}_weight", code="weight_trade_dates"),
                _semantic_dates(joined, f"{column}_trade", code="weight_trade_dates"),
            ),
            "weight_trade_dates",
        )
    _semantic_close(
        joined["previous_weight_weight"],
        joined["previous_weight_trade"],
        code="weight_trade_previous",
    )
    _semantic_close(
        joined["weight"], joined["target_weight"], code="weight_trade_target"
    )
    _semantic_close(
        joined["trade_weight"],
        joined["target_weight"] - joined["previous_weight_trade"],
        code="trade_weight_identity",
    )
    _semantic_close(
        joined["trade_dollars"],
        joined["trade_weight"] * joined["pre_trade_nav_dollars"],
        code="trade_dollar_identity",
    )
    nonzero_trade = np.abs(joined["trade_weight"].to_numpy(float)) > 1e-14
    _semantic_require(
        np.isfinite(joined.loc[nonzero_trade, "execution_price"]).all()
        and (joined.loc[nonzero_trade, "execution_price"] > 0.0).all()
        and np.isfinite(joined.loc[nonzero_trade, "full_spread"]).all()
        and (joined.loc[nonzero_trade, "full_spread"] >= 0.0).all()
        and np.isfinite(joined.loc[nonzero_trade, "adv20_usd"]).all()
        and (joined.loc[nonzero_trade, "adv20_usd"] > 0.0).all()
        and np.isfinite(joined.loc[nonzero_trade, "impact_volatility_daily"]).all()
        and (joined.loc[nonzero_trade, "impact_volatility_daily"] >= 0.0).all(),
        "nonzero_trade_execution_inputs",
    )
    _semantic_require(
        np.all(
            np.abs(joined["trade_weight"].to_numpy(float))
            <= joined["trade_capacity_weight"].to_numpy(float) + 1e-8
        ),
        "trade_capacity",
    )
    turnover = (
        joined.groupby(monthly_key, sort=False, observed=True)["trade_weight"]
        .apply(lambda value: 0.5 * float(np.abs(value.to_numpy(float)).sum()))
        .rename("recomputed_turnover")
        .reset_index()
    )
    turnover = monthly.merge(
        turnover, on=monthly_key, how="left", validate="one_to_one"
    )
    _semantic_close(
        turnover["turnover"],
        turnover["recomputed_turnover"],
        code="monthly_turnover",
    )
    recomputed_cost_rows: list[dict[str, object]] = []
    for (strategy_id, holding_month), group in joined.groupby(
        monthly_key, sort=False, observed=True
    ):
        nav = float(group["pre_trade_nav_dollars"].iloc[0])
        try:
            transaction_return = float(
                transaction_cost_numpy(
                    group["trade_weight"].to_numpy(float),
                    full_spread=group["full_spread"].to_numpy(float),
                    adv_dollars=group["adv20_usd"].to_numpy(float),
                    impact_volatility=group["impact_volatility_daily"].to_numpy(float),
                    nav_dollars=nav,
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise Stage11ExecutionPublicationError(
                "execution semantic verification failed: transaction_cost_inputs"
            ) from exc
        recomputed_cost_rows.append(
            {
                "strategy_id": strategy_id,
                "holding_month": holding_month,
                "recomputed_transaction_cost_dollars": transaction_return * nav,
            }
        )
    recomputed_costs = pd.DataFrame.from_records(recomputed_cost_rows)
    recomputed_costs = monthly.merge(
        recomputed_costs,
        on=monthly_key,
        how="left",
        validate="one_to_one",
    )
    _semantic_close(
        recomputed_costs["transaction_cost_dollars"],
        recomputed_costs["recomputed_transaction_cost_dollars"],
        code="transaction_cost_model",
    )
    _semantic_require(
        (
            joined["solver_status_x"].astype(str)
            == joined["solver_status_y"].astype(str)
        ).all()
        and (
            joined["solver_mode_x"].astype(str) == joined["solver_mode_y"].astype(str)
        ).all(),
        "weight_monthly_solver_identity",
    )

    first_month = VALIDATION_HOLDING_MONTHS[0]
    first_month_rows = joined.loc[
        joined["holding_month"].eq(first_month)
    ]
    try:
        initial_weights_are_immaterial = is_immaterial_weight(
            first_month_rows["previous_weight_weight"].to_numpy(np.float64),
            first_month_rows["pre_trade_nav_dollars"].to_numpy(np.float64),
        )
    except NumericalMaterialityError as exc:
        raise Stage11ExecutionPublicationError(
            "execution semantic verification failed: initial_portfolio_not_cash"
        ) from exc
    _semantic_require(
        np.all(initial_weights_are_immaterial),
        "initial_portfolio_not_cash",
    )

    daily["date"] = _semantic_dates(daily, "date", code="daily_date")
    daily_key = ["strategy_id", "holding_month", "date"]
    _semantic_require(not daily.duplicated(daily_key, keep=False).any(), "daily_key")
    daily_numeric = (
        "opening_nav_dollars",
        "nav_dollars",
        "net_total_return",
        "risk_free_return",
        "net_excess_return",
        "transaction_cost_dollars",
        "borrow_cost_dollars",
        "financing_cost_dollars",
    )
    for column in daily_numeric:
        daily[column] = _semantic_numeric(daily, column, code=f"daily_{column}")
    _semantic_require(
        np.all(daily["opening_nav_dollars"] > 0.0)
        and np.all(daily["nav_dollars"] > 0.0)
        and np.all(daily["transaction_cost_dollars"] >= 0.0)
        and np.all(daily["borrow_cost_dollars"] >= 0.0)
        and np.all(daily["financing_cost_dollars"] >= 0.0)
        and np.all(daily["risk_free_return"] > -1.0),
        "daily_numeric_bounds",
    )
    _semantic_close(
        daily["net_total_return"],
        daily["nav_dollars"] / daily["opening_nav_dollars"] - 1.0,
        code="daily_nav_return",
    )
    _semantic_close(
        daily["net_excess_return"],
        daily["net_total_return"] - daily["risk_free_return"],
        code="daily_excess_return",
    )
    sha_columns = ("opening_state_sha256", "closing_state_sha256")
    for column in sha_columns:
        values = daily[column].astype(str)
        _semantic_require(
            values.str.fullmatch(r"[0-9a-f]{64}").all(),
            f"daily_{column}",
        )
    _semantic_require(
        daily["corporate_action_ids"].notna().all(), "daily_corporate_action_ids"
    )

    session_reference: dict[str, tuple[tuple[dt.date, float], ...]] = {}
    monthly_by_key = monthly.set_index(monthly_key, verify_integrity=True)
    for (strategy_id, holding_month), group in daily.groupby(
        monthly_key, sort=False, observed=True
    ):
        group = group.sort_values("date", kind="mergesort").reset_index(drop=True)
        period = calendar[str(holding_month)]
        _semantic_require(
            bool(len(group))
            and tuple(group["date"]) == expected_sessions[str(holding_month)]
            and bool((group["date"] > period["entry_date"]).all())
            and bool((group["date"] <= period["exit_date"]).all())
            and group.iloc[-1]["date"] == period["exit_date"]
            and all(value.weekday() < 5 for value in group["date"]),
            "daily_session_scope",
        )
        reference = tuple(
            (date_value, float(rf))
            for date_value, rf in group[["date", "risk_free_return"]].itertuples(
                index=False, name=None
            )
        )
        prior_reference = session_reference.setdefault(str(holding_month), reference)
        _semantic_require(reference == prior_reference, "cross_strategy_sessions")
        if len(group) > 1:
            _semantic_close(
                group["opening_nav_dollars"].iloc[1:],
                group["nav_dollars"].iloc[:-1],
                code="daily_nav_chain",
            )
            _semantic_require(
                tuple(group["opening_state_sha256"].iloc[1:])
                == tuple(group["closing_state_sha256"].iloc[:-1]),
                "daily_state_hash_chain",
            )
        month_row = monthly_by_key.loc[(strategy_id, holding_month)]
        _semantic_close(
            [group.iloc[0]["opening_nav_dollars"]],
            [month_row["pre_trade_nav_dollars"]],
            code="daily_monthly_opening_nav",
        )
        _semantic_close(
            [group.iloc[-1]["nav_dollars"]],
            [month_row["closing_nav_dollars"]],
            code="daily_monthly_closing_nav",
        )
        compounded_total = float(
            np.prod(1.0 + group["net_total_return"].to_numpy(float), dtype=np.float64)
            - 1.0
        )
        compounded_rf = float(
            np.prod(1.0 + group["risk_free_return"].to_numpy(float), dtype=np.float64)
            - 1.0
        )
        _semantic_close(
            [compounded_total],
            [month_row["net_total_return"]],
            code="daily_monthly_total_return",
        )
        _semantic_close(
            [compounded_rf],
            [month_row["risk_free_total_return"]],
            code="daily_monthly_risk_free_return",
        )
        for column in (
            "transaction_cost_dollars",
            "borrow_cost_dollars",
            "financing_cost_dollars",
        ):
            _semantic_close(
                [float(group[column].sum())],
                [month_row[column]],
                code=f"daily_monthly_{column}",
            )
        _semantic_require(
            bool((group["transaction_cost_dollars"].iloc[1:] == 0.0).all()),
            "transaction_cost_timing",
        )

    month_order = {
        value: index for index, value in enumerate(VALIDATION_HOLDING_MONTHS)
    }
    for strategy_id, group in monthly.groupby("strategy_id", sort=False, observed=True):
        group = group.assign(
            _month_order=group["holding_month"].map(month_order)
        ).sort_values("_month_order", kind="mergesort")
        _semantic_require(
            tuple(group["holding_month"]) == VALIDATION_HOLDING_MONTHS, "monthly_order"
        )
        _semantic_close(
            group["pre_trade_nav_dollars"].iloc[1:],
            group["closing_nav_dollars"].iloc[:-1],
            code="monthly_nav_chain",
        )


@dataclass(frozen=True)
class Stage11ExecutionArtifactSet:
    plan: Stage11ExecutionPlan
    ledger: Stage11PseudoLiveLedger
    selection: LockedValidationSelection
    calibration_schedules: Mapping[str, StrategyCalibrationSchedule]
    execution_sessions: pd.DataFrame
    execution_session_receipts: Mapping[str, Mapping[str, str]]
    portfolio_weights: pd.DataFrame
    portfolio_trades: pd.DataFrame
    daily_accounting: pd.DataFrame
    monthly_metrics: pd.DataFrame
    access_ledger: Mapping[str, Any]

    def tables(self) -> Mapping[str, pd.DataFrame]:
        values = {
            RAW_FORECASTS_NAME: self.ledger.combined_forecasts(),
            OUTCOMES_NAME: self.ledger.combined_outcomes(),
            FAMILY_IC_NAME: self.selection.family_monthly_ic,
            ENSEMBLE_IC_NAME: self.selection.ensemble_monthly_ic,
            SELECTED_SCORES_NAME: self.selection.selected_scores,
            CALIBRATION_NAME: _calibration_frame(self.calibration_schedules),
            EXECUTION_SESSIONS_NAME: self.execution_sessions,
            WEIGHTS_NAME: self.portfolio_weights,
            TRADES_NAME: self.portfolio_trades,
            DAILY_ACCOUNTING_NAME: self.daily_accounting,
            MONTHLY_METRICS_NAME: self.monthly_metrics,
        }
        normalized: dict[str, pd.DataFrame] = {}
        for name in TABLE_NAMES:
            frame = _normalize_date_scope(values[name], label=name)
            expected = TABLE_COLUMNS[name]
            _require(
                len(frame.columns) == len(set(frame.columns))
                and set(frame.columns) == set(expected),
                f"{name} schema differs from the closed execution contract",
            )
            normalized[name] = frame.loc[:, list(expected)].copy(deep=True)
        return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class Stage11ExecutionPublication:
    build_id: str
    bundle_path: Path
    manifest_sha256: str
    reused: bool


@dataclass(frozen=True, slots=True)
class Stage11ExecutionVerification:
    verification_id: str
    bundle_path: Path
    difference_count: int
    report_sha256: str
    reused: bool


def _build_payloads(
    artifacts: Stage11ExecutionArtifactSet,
    *,
    implementation_code_hashes: Mapping[str, str],
    synthetic_fixture: bool,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    _require(
        type(artifacts.plan) is Stage11ExecutionPlan, "execution artifact plan differs"
    )
    _require(
        type(artifacts.ledger) is Stage11PseudoLiveLedger
        and artifacts.ledger.complete
        and artifacts.ledger.plan.plan_sha256 == artifacts.plan.plan_sha256,
        "execution artifact ledger is incomplete or bound to another plan",
    )
    _require(
        type(artifacts.selection) is LockedValidationSelection,
        "execution artifact selection differs",
    )
    _require(
        all(
            isinstance(key, str) and isinstance(value, str) and len(value) == 64
            for key, value in implementation_code_hashes.items()
        ),
        "implementation code hashes are malformed",
    )
    tables = artifacts.tables()
    selection_payload = _selection_payload(artifacts.selection)
    calibration_hashes = {
        name: value.schedule_sha256
        for name, value in sorted(artifacts.calibration_schedules.items())
    }
    session_receipts = {
        str(month): {str(key): str(value) for key, value in sorted(receipt.items())}
        for month, receipt in sorted(artifacts.execution_session_receipts.items())
    }
    architecture_status = _architecture_status(selection_payload)
    _validate_execution_semantics(
        tables,
        plan_payload=artifacts.plan.payload(),
        selection_payload=selection_payload,
        expected_calibration_hashes=calibration_hashes,
        expected_session_receipts=session_receipts,
        expected_architecture_status=architecture_status,
    )
    payloads = {name: _parquet_bytes(tables[name]) for name in TABLE_NAMES}
    payloads[PLAN_NAME] = _canonical_json(artifacts.plan.payload())
    payloads[SELECTION_NAME] = _canonical_json(selection_payload)
    payloads[ACCESS_LEDGER_NAME] = _canonical_json(dict(artifacts.access_ledger))
    # Bind the logical representation reconstructed from the exact Parquet
    # bytes, not pandas' pre-serialization dtype choices (notably the nullable
    # Elastic-Net seed column).
    logical_hashes = {
        name: canonical_frame_sha256(
            pq.read_table(pa.BufferReader(payloads[name])).to_pandas()
        )
        for name in TABLE_NAMES
    }
    identity = {
        "schema_version": 1,
        "publication_id": PUBLICATION_ID,
        "synthetic_fixture": synthetic_fixture,
        "plan_sha256": artifacts.plan.plan_sha256,
        "pseudo_live_ledger_sha256": artifacts.ledger.ledger_sha256,
        "selection_sha256": artifacts.selection.selection_sha256,
        "calibration_schedule_sha256": calibration_hashes,
        "execution_session_receipts": session_receipts,
        "architecture_status": architecture_status,
        "logical_table_sha256": logical_hashes,
        "implementation_code_hashes": dict(sorted(implementation_code_hashes.items())),
        "access_ledger_sha256": _sha_bytes(payloads[ACCESS_LEDGER_NAME]),
    }
    fingerprint = _sha_bytes(_canonical_json(identity))
    build_id = f"stage11-validation-{fingerprint[:24]}"
    manifest = {
        **identity,
        "status": PRIMARY_STATUS,
        "build_id": build_id,
        "content_fingerprint": fingerprint,
        "members": {
            name: {"sha256": _sha_bytes(value), "size_bytes": len(value)}
            for name, value in sorted(payloads.items())
        },
        "scope": {
            "first_holding_month": VALIDATION_HOLDING_MONTHS[0],
            "last_holding_month": VALIDATION_HOLDING_MONTHS[-1],
            "holding_months": len(VALIDATION_HOLDING_MONTHS),
            "december_2019_holding_month_included": False,
            "final_2020_plus_included": False,
        },
    }
    payloads[MANIFEST_NAME] = _canonical_json(manifest)
    success = {
        "schema_version": 1,
        "status": PRIMARY_STATUS,
        "build_id": build_id,
        "manifest_sha256": _sha_bytes(payloads[MANIFEST_NAME]),
        "closed_members": list(PRIMARY_MEMBERS),
    }
    payloads[SUCCESS_NAME] = _canonical_json(success)
    _require(
        set(payloads) == set(PRIMARY_MEMBERS), "primary publication member set differs"
    )
    return payloads, manifest


def _reuse_or_publish(
    root: Path, directory: Path, payloads: Mapping[str, bytes]
) -> bool:
    _require(directory.parent == root, "publication destination escaped its root")
    if directory.exists():
        _require(
            directory.is_dir() and not directory.is_symlink(),
            "existing bundle is unsafe",
        )
        _require(
            set(path.name for path in directory.iterdir()) == set(payloads),
            "existing bundle members differ",
        )
        for name, payload in payloads.items():
            path = directory / name
            _require(
                path.is_file() and not path.is_symlink(),
                f"unsafe existing member: {name}",
            )
            _require(
                _sha_file(path) == _sha_bytes(payload),
                f"existing bundle changed: {name}",
            )
        return True
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=root))
    try:
        for name in (
            PRIMARY_MEMBERS
            if set(payloads) == set(PRIMARY_MEMBERS)
            else VERIFICATION_MEMBERS
        ):
            _atomic_file(temporary / name, payloads[name])
        os.replace(temporary, directory)
    finally:
        if temporary.exists() or temporary.is_symlink():
            shutil.rmtree(temporary)
    return False


def publish_stage11_execution_core(
    artifacts: Stage11ExecutionArtifactSet,
    *,
    output_root: Path,
    implementation_code_hashes: Mapping[str, str],
    synthetic_fixture: bool,
) -> Stage11ExecutionPublication:
    """Byte-identical resumable publisher; empirical callers gate it externally."""

    root = _ordinary_root(output_root)
    payloads, manifest = _build_payloads(
        artifacts,
        implementation_code_hashes=implementation_code_hashes,
        synthetic_fixture=synthetic_fixture,
    )
    bundle = root / str(manifest["build_id"])
    reused = _reuse_or_publish(root, bundle, payloads)
    return Stage11ExecutionPublication(
        build_id=str(manifest["build_id"]),
        bundle_path=bundle,
        manifest_sha256=_sha_bytes(payloads[MANIFEST_NAME]),
        reused=reused,
    )


def publish_stage11_execution(
    authority: object,
    access_broker: object,
    artifacts: Stage11ExecutionArtifactSet,
    *,
    output_root: Path,
    implementation_code_hashes: Mapping[str, str],
) -> Stage11ExecutionPublication:
    """Publish an empirical bundle while retaining live unblind authority."""

    from portfolio_bot.validation.access import PayloadRole
    from portfolio_bot.validation.execution_access import (
        ValidationExecutionAccessBroker,
    )
    from portfolio_bot.validation.validation_runner import (
        require_live_stage11_validation_unblind_authority,
    )

    require_live_stage11_validation_unblind_authority(authority)
    _require(
        type(access_broker) is ValidationExecutionAccessBroker,
        "empirical publication requires the exact execution access broker",
    )
    access_broker.ledger()

    def record(
        role: PayloadRole,
        frame: pd.DataFrame,
        *,
        date_column: str,
    ) -> None:
        require_live_stage11_validation_unblind_authority(authority)
        _require(
            isinstance(frame, pd.DataFrame), "empirical output must be a DataFrame"
        )
        if frame.empty:
            access_broker.record_created_output(
                role,
                rows=0,
                minimum_date=None,
                maximum_date=None,
                holding_months=(),
            )
            return
        _require(
            date_column in frame.columns and "holding_month" in frame.columns,
            "empirical output omitted its exact date/month scope",
        )
        try:
            dates = pd.to_datetime(frame[date_column], errors="raise").dt.date
        except (TypeError, ValueError) as exc:
            raise Stage11ExecutionPublicationError(
                f"empirical output has invalid {date_column}"
            ) from exc
        months = tuple(
            value
            for value in VALIDATION_HOLDING_MONTHS
            if value in set(frame["holding_month"].astype(str))
        )
        _require(
            len(months) == len(set(frame["holding_month"].astype(str))),
            "empirical output holding-month scope differs",
        )
        access_broker.record_created_output(
            role,
            rows=len(frame),
            minimum_date=min(dates),
            maximum_date=max(dates),
            holding_months=months,
        )

    record(
        PayloadRole.REAL_VALIDATION_WEIGHT,
        artifacts.portfolio_weights,
        date_column="entry_date",
    )
    record(
        PayloadRole.REAL_VALIDATION_TRADE,
        artifacts.portfolio_trades,
        date_column="entry_date",
    )
    record(
        PayloadRole.REAL_VALIDATION_PNL,
        artifacts.daily_accounting,
        date_column="date",
    )
    record(
        PayloadRole.REAL_VALIDATION_PNL,
        artifacts.monthly_metrics,
        date_column="exit_date",
    )
    require_live_stage11_validation_unblind_authority(authority)
    bound = replace(artifacts, access_ledger=access_broker.ledger().to_dict())
    result = publish_stage11_execution_core(
        bound,
        output_root=output_root,
        implementation_code_hashes=implementation_code_hashes,
        synthetic_fixture=False,
    )
    require_live_stage11_validation_unblind_authority(authority)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11ExecutionPublicationError(
            f"invalid JSON member: {path.name}"
        ) from exc
    _require(isinstance(value, dict), f"JSON member is not an object: {path.name}")
    return value


def verify_stage11_execution_core(
    publication: Stage11ExecutionPublication,
    *,
    verification_root: Path,
    expected_synthetic_fixture: bool,
) -> Stage11ExecutionVerification:
    """Independently reopen every member and recompute physical/logical hashes."""

    bundle = publication.bundle_path
    _require(
        bundle.is_dir() and not bundle.is_symlink(),
        "primary execution bundle is unsafe",
    )
    _require(
        set(path.name for path in bundle.iterdir()) == set(PRIMARY_MEMBERS),
        "primary closed member set differs",
    )
    manifest_path = bundle / MANIFEST_NAME
    manifest = _read_json(manifest_path)
    differences: list[str] = []
    if _sha_file(manifest_path) != publication.manifest_sha256:
        differences.append("manifest_sha256")
    if manifest.get("build_id") != publication.build_id:
        differences.append("build_id")
    if manifest.get("synthetic_fixture") is not expected_synthetic_fixture:
        differences.append("synthetic_fixture")
    if manifest.get("status") != PRIMARY_STATUS:
        differences.append("status")
    members = manifest.get("members")
    if not isinstance(members, dict) or set(members) != set(PRIMARY_MEMBERS) - {
        MANIFEST_NAME,
        SUCCESS_NAME,
    }:
        differences.append("manifest_member_set")
        members = {}
    for name in set(PRIMARY_MEMBERS) - {MANIFEST_NAME, SUCCESS_NAME}:
        path = bundle / name
        record = members.get(name, {}) if isinstance(members, dict) else {}
        if (
            not path.is_file()
            or path.is_symlink()
            or record.get("sha256") != _sha_file(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            differences.append(f"member:{name}")
    logical = manifest.get("logical_table_sha256")
    reopened_tables: dict[str, pd.DataFrame] = {}
    if not isinstance(logical, dict) or set(logical) != set(TABLE_NAMES):
        differences.append("logical_table_set")
        logical = {}
    else:
        for name in TABLE_NAMES:
            try:
                frame = pq.read_table(bundle / name).to_pandas()
                frame = _normalize_date_scope(frame, label=f"verify:{name}")
                reopened_tables[name] = frame
                if canonical_frame_sha256(frame) != logical[name]:
                    differences.append(f"logical:{name}")
            except (OSError, pa.ArrowException, Stage11ExecutionPublicationError):
                differences.append(f"table:{name}")
    if set(reopened_tables) == set(TABLE_NAMES):
        try:
            plan_payload = _read_json(bundle / PLAN_NAME)
            selection_payload = _read_json(bundle / SELECTION_NAME)
            calibration_hashes = manifest.get("calibration_schedule_sha256")
            session_receipts = manifest.get("execution_session_receipts")
            architecture_status = manifest.get("architecture_status")
            _semantic_require(
                isinstance(calibration_hashes, Mapping),
                "calibration_manifest_schema",
            )
            _semantic_require(
                isinstance(session_receipts, Mapping),
                "execution_session_manifest_schema",
            )
            _semantic_require(
                isinstance(architecture_status, Mapping),
                "architecture_status_manifest_schema",
            )
            _validate_execution_semantics(
                reopened_tables,
                plan_payload=plan_payload,
                selection_payload=selection_payload,
                expected_calibration_hashes={
                    str(key): str(value) for key, value in calibration_hashes.items()
                },
                expected_session_receipts={
                    str(month): {str(key): str(value) for key, value in receipt.items()}
                    for month, receipt in session_receipts.items()
                    if isinstance(receipt, Mapping)
                },
                expected_architecture_status={
                    str(key): value for key, value in architecture_status.items()
                },
            )
        except (OSError, Stage11ExecutionPublicationError, TypeError, ValueError):
            differences.append("semantic:execution_tables")
    success = _read_json(bundle / SUCCESS_NAME)
    if (
        success.get("status") != PRIMARY_STATUS
        or success.get("build_id") != publication.build_id
        or success.get("manifest_sha256") != publication.manifest_sha256
        or success.get("closed_members") != list(PRIMARY_MEMBERS)
    ):
        differences.append("success_marker")

    report_identity = {
        "schema_version": 1,
        "publication_id": PUBLICATION_ID,
        "primary_build_id": publication.build_id,
        "primary_manifest_sha256": publication.manifest_sha256,
        "difference_count": len(differences),
        "differences": sorted(set(differences)),
        "status": (
            VERIFIED_STATUS
            if not differences
            else "STAGE11_VALIDATION_EXECUTION_VERIFICATION_FAILED"
        ),
        "independent_runner": "stage11-execution-independent-verifier-v1",
    }
    verification_id = f"verify-{_sha_bytes(_canonical_json(report_identity))[:16]}"
    report = {**report_identity, "verification_id": verification_id}
    difference_payload = {
        "schema_version": 1,
        "difference_count": len(report_identity["differences"]),
        "differences": report_identity["differences"],
    }
    payloads = {
        "verification_report.json": _canonical_json(report),
        "differences.json": _canonical_json(difference_payload),
    }
    payloads[SUCCESS_NAME] = _canonical_json(
        {
            "schema_version": 1,
            "status": report["status"],
            "verification_id": verification_id,
            "primary_manifest_sha256": publication.manifest_sha256,
            "report_sha256": _sha_bytes(payloads["verification_report.json"]),
            "differences_sha256": _sha_bytes(payloads["differences.json"]),
            "closed_members": list(VERIFICATION_MEMBERS),
        }
    )
    root = _ordinary_root(verification_root)
    target = root / verification_id
    reused = _reuse_or_publish(root, target, payloads)
    return Stage11ExecutionVerification(
        verification_id=verification_id,
        bundle_path=target,
        difference_count=len(report_identity["differences"]),
        report_sha256=_sha_bytes(payloads["verification_report.json"]),
        reused=reused,
    )


def verify_stage11_execution(
    authority: object,
    publication: Stage11ExecutionPublication,
    *,
    verification_root: Path,
) -> Stage11ExecutionVerification:
    from portfolio_bot.validation.validation_runner import (
        require_live_stage11_validation_unblind_authority,
    )

    require_live_stage11_validation_unblind_authority(authority)
    result = verify_stage11_execution_core(
        publication,
        verification_root=verification_root,
        expected_synthetic_fixture=False,
    )
    require_live_stage11_validation_unblind_authority(authority)
    return result


__all__ = [
    "ACCESS_LEDGER_NAME",
    "CALIBRATION_NAME",
    "DAILY_ACCOUNTING_NAME",
    "ENSEMBLE_IC_NAME",
    "FAMILY_IC_NAME",
    "MANIFEST_NAME",
    "MONTHLY_METRICS_NAME",
    "OUTCOMES_NAME",
    "PLAN_NAME",
    "PRIMARY_MEMBERS",
    "PRIMARY_STATUS",
    "PUBLICATION_ID",
    "RAW_FORECASTS_NAME",
    "SELECTED_SCORES_NAME",
    "SELECTION_NAME",
    "SUCCESS_NAME",
    "Stage11ExecutionArtifactSet",
    "Stage11ExecutionPublication",
    "Stage11ExecutionPublicationError",
    "Stage11ExecutionVerification",
    "TABLE_NAMES",
    "TRADES_NAME",
    "VERIFICATION_MEMBERS",
    "VERIFIED_STATUS",
    "WEIGHTS_NAME",
    "publish_stage11_execution",
    "publish_stage11_execution_core",
    "verify_stage11_execution",
    "verify_stage11_execution_core",
]
