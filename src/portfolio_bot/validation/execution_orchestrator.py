"""Locked orchestration primitives for the Stage 11 validation execution.

This module is deliberately split from the repository authority gate.  It
contains the deterministic state machine that the future empirical entrypoint
will call *inside* a live :class:`Stage11ValidationUnblindAuthority` scope.
The state machine is also executable against synthetic fixtures, which lets us
verify chronology and scientific choices without opening validation outcomes.

The important ordering is enforced here rather than left to a runbook:

1. completed outcomes are admitted only after their forecasts were committed;
2. 20 quarterly refits execute the exact nine-configuration shortlist (660
   model fits in total), while all 59 months are scored;
3. family and NN/XGBoost weight selection uses only monthly rank IC and the
   frozen turnover tie break;
4. calibration is recomputed quarterly from development OOF plus prior-live
   validation rows only; and
5. portfolio construction keeps the hybrid primary, Elastic-Net/hybrid and
   rank challengers distinct.  Decision-focused execution remains explicitly
   inactive unless the Stage 10 production certification constant is true.

No function in this file resolves a path, creates an approval, or knows the
location of validation/final data.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from portfolio_bot.allocation.decision_focused import (
    PRODUCTION_TRAINING_RUNTIME_CERTIFIED,
)
from portfolio_bot.allocation.rank import allocate_rank
from portfolio_bot.allocation.solver import (
    AllocationResult,
    AllocationSolveError,
    solve_hybrid,
)
from portfolio_bot.validation.accounting import PortfolioState
from portfolio_bot.selectors.metrics import (
    centered_average_rank,
    correlation_with_status,
)
from portfolio_bot.validation.calibration import (
    CalibrationObservation,
    CalibrationReceipt,
    fit_nonnegative_huber_calibration,
)
from portfolio_bot.validation.corporate_actions import (
    CorporateAction,
    CorporateActionLedger,
)
from portfolio_bot.validation.daily_engine import (
    DatedHoldingPeriodAccounting,
    DatedMarketCloseInput,
    MarkContinuityLedger,
    run_dated_holding_period,
)
from portfolio_bot.validation.calendar import (
    VALIDATION_HOLDING_MONTHS,
    ValidationCalendarRow,
    validate_validation_calendar,
)
from portfolio_bot.validation.locked_selection import (
    LOCKED_ENSEMBLE_NN_WEIGHTS,
    LOCKED_SHORTLIST,
    VALIDATION_STOCHASTIC_SEEDS,
    EnsembleSelection,
    FamilySelection,
    ValidationForecastProvenance,
    compose_locked_ensemble_scores,
    mean_adjacent_signal_turnover,
    select_ensemble_weight,
    select_family_specifications,
    selection_frame_sha256,
)
from portfolio_bot.validation.problem_builder import MonthlyProblemBuild
from portfolio_bot.validation.walk_forward import (
    AuthenticatedValidationCalendar,
    PRIMARY_REFIT_MONTHS,
)
from portfolio_bot.validation.session_authority import HoldingPeriodSessionAuthority


ORCHESTRATOR_ID = "stage11-locked-execution-orchestrator-v1"
PRIMARY_STRATEGY_ID = "ensemble_hybrid_primary"
ELASTIC_NET_CHALLENGER_ID = "elastic_net_hybrid_challenger"
RANK_CHALLENGER_ID = "ensemble_rank_challenger"
DECISION_FOCUSED_CHALLENGER_ID = "decision_focused_neural_projected"

FAMILY_ORDER = ("neural_network", "xgboost", "elastic_net")
EXPECTED_REFIT_COUNT = 20
EXPECTED_FITS_PER_REFIT = 33
EXPECTED_TOTAL_FITS = EXPECTED_REFIT_COUNT * EXPECTED_FITS_PER_REFIT

FORECAST_COLUMNS = (
    "model_family",
    "config_id",
    "seed",
    "signal_date",
    "holding_month",
    "permno",
    "raw_forecast",
)
OUTCOME_COLUMNS = (
    "signal_date",
    "holding_month",
    "entry_date",
    "exit_date",
    "outcome_availability_date",
    "permno",
    "unranked_residual_return",
    "residual_return_rank",
    "primary_label_status",
)

_PRESENT = "PRESENT"
_SHA256 = frozenset("0123456789abcdef")


class Stage11ExecutionOrchestratorError(RuntimeError):
    """Raised when execution order or a frozen Stage 11 choice differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11ExecutionOrchestratorError(message)


def _canonical_json_bytes(value: Any) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_SHA256)


def _canonical_scalar(value: object) -> str:
    if value is None or value is pd.NA or bool(pd.isna(value)):
        return "null"
    if isinstance(value, pd.Period):
        return f"period:{value.freqstr}:{value}"
    if isinstance(value, pd.Timestamp):
        return f"timestamp:{value.isoformat()}"
    if isinstance(value, dt.date):
        return f"date:{value.isoformat()}"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return f"bool:{int(value)}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        _require(math.isfinite(value), "canonical table contains a nonfinite float")
        return f"float:{value.hex()}"
    return f"text:{value}"


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    """Order-independent digest that also supports the null Elastic-Net seed."""

    _require(isinstance(frame, pd.DataFrame), "canonical payload must be a DataFrame")
    _require(
        all(isinstance(column, str) for column in frame.columns)
        and len(set(frame.columns)) == len(frame.columns),
        "canonical payload columns must be unique strings",
    )
    columns = tuple(sorted(frame.columns))
    rows = [
        "\x1f".join(_canonical_scalar(row[column]) for column in columns)
        for _, row in frame.loc[:, columns].iterrows()
    ]
    payload = "\x1e".join(("\x1f".join(columns), *sorted(rows))).encode("utf-8")
    return _sha256_bytes(payload)


@dataclass(frozen=True, slots=True)
class SelectorFitJob:
    """One exact model/configuration/seed fit at one quarterly refit."""

    refit_index: int
    refit_signal_date: dt.date
    model_family: str
    config_id: str
    seed: int | None

    def __post_init__(self) -> None:
        _require(
            type(self.refit_index) is int and self.refit_index >= 0,
            "invalid refit index",
        )
        _require(type(self.refit_signal_date) is dt.date, "refit date must be a date")
        _require(
            self.refit_signal_date.month in PRIMARY_REFIT_MONTHS,
            "selector fit is not on a quarterly refit signal",
        )
        _require(
            self.model_family in LOCKED_SHORTLIST
            and self.config_id in LOCKED_SHORTLIST[self.model_family],
            "selector fit escaped the frozen shortlist",
        )
        if self.model_family == "elastic_net":
            _require(self.seed is None, "Elastic Net fit cannot carry a seed")
        else:
            _require(
                type(self.seed) is int and self.seed in VALIDATION_STOCHASTIC_SEEDS,
                "stochastic selector fit escaped the five frozen seeds",
            )

    @property
    def job_id(self) -> str:
        seed = "deterministic" if self.seed is None else str(self.seed)
        return (
            f"refit-{self.refit_index:02d}:{self.refit_signal_date.isoformat()}:"
            f"{self.model_family}:{self.config_id}:{seed}"
        )


@dataclass(frozen=True, slots=True)
class MonthlyExecutionStep:
    """One monthly score/optimization decision and its active model refit."""

    month_index: int
    calendar_row: ValidationCalendarRow
    active_refit_index: int
    active_refit_signal_date: dt.date
    refit_now: bool


@dataclass(frozen=True, slots=True)
class Stage11ExecutionPlan:
    calendar_rows: tuple[ValidationCalendarRow, ...]
    refit_signal_dates: tuple[dt.date, ...]
    fit_jobs: tuple[SelectorFitJob, ...]
    monthly_steps: tuple[MonthlyExecutionStep, ...]
    plan_sha256: str

    def __post_init__(self) -> None:
        rows = validate_validation_calendar(self.calendar_rows)
        _require(rows == self.calendar_rows, "execution calendar order changed")
        _require(
            len(self.refit_signal_dates) == EXPECTED_REFIT_COUNT
            and len(set(self.refit_signal_dates)) == EXPECTED_REFIT_COUNT,
            "execution plan must contain exactly 20 quarterly refits",
        )
        _require(
            len(self.fit_jobs) == EXPECTED_TOTAL_FITS,
            "execution plan must contain exactly 660 selector fits",
        )
        _require(
            len(self.monthly_steps) == len(VALIDATION_HOLDING_MONTHS) == 59,
            "execution plan must contain exactly 59 monthly score steps",
        )
        for refit_index, refit_date in enumerate(self.refit_signal_dates):
            jobs = tuple(job for job in self.fit_jobs if job.refit_index == refit_index)
            _require(
                len(jobs) == EXPECTED_FITS_PER_REFIT
                and all(job.refit_signal_date == refit_date for job in jobs),
                "quarterly refit does not contain exactly 33 bound fits",
            )
            expected = _fit_job_keys()
            actual = tuple((job.model_family, job.config_id, job.seed) for job in jobs)
            _require(
                actual == expected, "quarterly selector fit order/configuration differs"
            )
        _require(
            self.plan_sha256
            == _sha256_bytes(_canonical_json_bytes(self.payload(include_sha=False))),
            "execution plan digest differs",
        )

    def payload(self, *, include_sha: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": 1,
            "orchestrator_id": ORCHESTRATOR_ID,
            "calendar_rows": [
                {
                    "holding_month": row.holding_month,
                    "signal_date": row.signal_date.isoformat(),
                    "entry_date": row.entry_date.isoformat(),
                    "exit_date": row.exit_date.isoformat(),
                }
                for row in self.calendar_rows
            ],
            "refit_signal_dates": [
                value.isoformat() for value in self.refit_signal_dates
            ],
            "fit_jobs": [
                {
                    "job_id": job.job_id,
                    "refit_index": job.refit_index,
                    "refit_signal_date": job.refit_signal_date.isoformat(),
                    "model_family": job.model_family,
                    "config_id": job.config_id,
                    "seed": job.seed,
                }
                for job in self.fit_jobs
            ],
            "monthly_steps": [
                {
                    "month_index": step.month_index,
                    "holding_month": step.calendar_row.holding_month,
                    "signal_date": step.calendar_row.signal_date.isoformat(),
                    "entry_date": step.calendar_row.entry_date.isoformat(),
                    "exit_date": step.calendar_row.exit_date.isoformat(),
                    "active_refit_index": step.active_refit_index,
                    "active_refit_signal_date": step.active_refit_signal_date.isoformat(),
                    "refit_now": step.refit_now,
                }
                for step in self.monthly_steps
            ],
            "counts": {
                "holding_months": len(self.calendar_rows),
                "quarterly_refits": len(self.refit_signal_dates),
                "fits_per_refit": EXPECTED_FITS_PER_REFIT,
                "total_selector_fits": len(self.fit_jobs),
            },
        }
        if include_sha:
            result["plan_sha256"] = self.plan_sha256
        return result


def _fit_job_keys() -> tuple[tuple[str, str, int | None], ...]:
    rows: list[tuple[str, str, int | None]] = []
    for family in FAMILY_ORDER:
        for config_id in LOCKED_SHORTLIST[family]:
            seeds: tuple[int | None, ...] = (
                (None,) if family == "elastic_net" else VALIDATION_STOCHASTIC_SEEDS
            )
            rows.extend((family, config_id, seed) for seed in seeds)
    result = tuple(rows)
    _require(len(result) == EXPECTED_FITS_PER_REFIT, "internal fit-key count differs")
    return result


def build_stage11_execution_plan(
    calendar_authority: AuthenticatedValidationCalendar,
) -> Stage11ExecutionPlan:
    """Build the only permitted 59-month/20-refit/660-fit execution plan."""

    _require(
        type(calendar_authority) is AuthenticatedValidationCalendar,
        "execution planning requires authenticated calendar authority",
    )
    rows = validate_validation_calendar(calendar_authority.rows)
    refits = tuple(
        row.signal_date for row in rows if row.signal_date.month in PRIMARY_REFIT_MONTHS
    )
    _require(
        len(refits) == EXPECTED_REFIT_COUNT
        and refits[0] == rows[0].signal_date
        and refits[-1] <= rows[-1].signal_date,
        "authenticated calendar does not produce the exact 20 refits",
    )
    jobs = tuple(
        SelectorFitJob(index, refit_date, family, config_id, seed)
        for index, refit_date in enumerate(refits)
        for family, config_id, seed in _fit_job_keys()
    )
    steps: list[MonthlyExecutionStep] = []
    active_index = -1
    for month_index, row in enumerate(rows):
        if row.signal_date in refits:
            active_index = refits.index(row.signal_date)
        _require(active_index >= 0, "monthly scoring precedes the first model refit")
        steps.append(
            MonthlyExecutionStep(
                month_index=month_index,
                calendar_row=row,
                active_refit_index=active_index,
                active_refit_signal_date=refits[active_index],
                refit_now=row.signal_date == refits[active_index],
            )
        )
    provisional = Stage11ExecutionPlan.__new__(Stage11ExecutionPlan)
    object.__setattr__(provisional, "calendar_rows", rows)
    object.__setattr__(provisional, "refit_signal_dates", refits)
    object.__setattr__(provisional, "fit_jobs", jobs)
    object.__setattr__(provisional, "monthly_steps", tuple(steps))
    object.__setattr__(provisional, "plan_sha256", "")
    digest = _sha256_bytes(
        _canonical_json_bytes(provisional.payload(include_sha=False))
    )
    return Stage11ExecutionPlan(rows, refits, jobs, tuple(steps), digest)


@dataclass(frozen=True, slots=True)
class MonthlyForecastCommit:
    holding_month: str
    signal_date: dt.date
    entry_date: dt.date
    exit_date: dt.date
    rows: int
    securities: int
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class MonthlyOutcomeCommit:
    holding_month: str
    exit_date: dt.date
    maximum_outcome_availability_date: dt.date
    committed_at: dt.date
    rows: int
    valid_rows: int
    payload_sha256: str


def _exact_columns(frame: pd.DataFrame, expected: Sequence[str], label: str) -> None:
    actual = tuple(str(column) for column in frame.columns)
    _require(
        len(actual) == len(set(actual)) and set(actual) == set(expected),
        f"{label} columns differ; expected={sorted(expected)}, actual={sorted(actual)}",
    )


def _normalize_forecast_month(
    frame: pd.DataFrame,
    row: ValidationCalendarRow,
) -> pd.DataFrame:
    _exact_columns(frame, FORECAST_COLUMNS, "monthly raw forecast")
    result = frame.loc[:, FORECAST_COLUMNS].copy()
    _require(not result.empty, "monthly raw forecast cannot be empty")
    try:
        result["signal_date"] = pd.to_datetime(
            result["signal_date"], errors="raise"
        ).dt.date
    except (TypeError, ValueError) as exc:
        raise Stage11ExecutionOrchestratorError(
            "forecast signal dates are invalid"
        ) from exc
    _require(
        set(result["signal_date"]) == {row.signal_date}
        and set(result["holding_month"].astype(str)) == {row.holding_month},
        "forecast month/signal differs from the execution plan",
    )
    raw_permno = pd.to_numeric(result["permno"], errors="coerce").to_numpy(float)
    raw_forecast = pd.to_numeric(result["raw_forecast"], errors="coerce").to_numpy(
        float
    )
    _require(
        np.isfinite(raw_permno).all()
        and np.all(raw_permno > 0.0)
        and np.all(raw_permno == np.floor(raw_permno)),
        "forecast PERMNO values must be positive integers",
    )
    _require(np.isfinite(raw_forecast).all(), "raw forecasts must be finite")
    result["permno"] = raw_permno.astype(np.int64)
    result["raw_forecast"] = raw_forecast
    result["model_family"] = result["model_family"].astype(str)
    result["config_id"] = result["config_id"].astype(str)
    result["holding_month"] = result["holding_month"].astype(str)
    keys = [
        "model_family",
        "config_id",
        "seed",
        "signal_date",
        "holding_month",
        "permno",
    ]
    _require(not result.duplicated(keys).any(), "monthly forecast keys are duplicated")

    domains: list[tuple[int, ...]] = []
    expected_keys = _fit_job_keys()
    actual_keys: list[tuple[str, str, int | None]] = []
    for family in FAMILY_ORDER:
        for config_id in LOCKED_SHORTLIST[family]:
            group = result.loc[
                result["model_family"].eq(family) & result["config_id"].eq(config_id)
            ]
            _require(not group.empty, "monthly forecast omits a locked configuration")
            if family == "elastic_net":
                _require(
                    group["seed"].isna().all(), "Elastic Net forecast seed must be null"
                )
                seeds: tuple[int | None, ...] = (None,)
            else:
                _require(
                    not group["seed"].isna().any(), "stochastic forecast seed is null"
                )
                numeric = pd.to_numeric(group["seed"], errors="coerce").to_numpy(float)
                _require(
                    np.isfinite(numeric).all() and np.all(numeric == np.floor(numeric)),
                    "stochastic forecast seeds must be integers",
                )
                group = group.copy()
                group["seed"] = numeric.astype(np.int64)
                result.loc[group.index, "seed"] = group["seed"]
                seeds = tuple(sorted(int(value) for value in group["seed"].unique()))
                _require(
                    seeds == VALIDATION_STOCHASTIC_SEEDS,
                    "monthly forecast escaped the five frozen stochastic seeds",
                )
            for seed in seeds:
                seed_group = (
                    group.loc[group["seed"].isna()]
                    if seed is None
                    else group.loc[pd.to_numeric(group["seed"]).eq(seed)]
                )
                domain = tuple(sorted(int(value) for value in seed_group["permno"]))
                _require(
                    len(domain) == len(set(domain)) and bool(domain),
                    "forecast security domain differs",
                )
                domains.append(domain)
                actual_keys.append((family, config_id, seed))
    _require(tuple(actual_keys) == expected_keys, "monthly forecast model grid differs")
    _require(
        len(set(domains)) == 1, "all locked models must score the same monthly domain"
    )
    return result.sort_values(
        ["model_family", "config_id", "seed", "permno"],
        kind="mergesort",
        na_position="first",
    ).reset_index(drop=True)


def _normalize_outcome_month(
    frame: pd.DataFrame,
    row: ValidationCalendarRow,
    forecast_permnos: tuple[int, ...],
    *,
    knowledge_date: dt.date,
) -> pd.DataFrame:
    _exact_columns(frame, OUTCOME_COLUMNS, "monthly residual outcome")
    result = frame.loc[:, OUTCOME_COLUMNS].copy()
    _require(not result.empty, "monthly outcome cannot be empty")
    for column, expected in (
        ("signal_date", row.signal_date),
        ("entry_date", row.entry_date),
        ("exit_date", row.exit_date),
    ):
        try:
            result[column] = pd.to_datetime(result[column], errors="raise").dt.date
        except (TypeError, ValueError) as exc:
            raise Stage11ExecutionOrchestratorError(
                f"outcome {column} is invalid"
            ) from exc
        _require(
            set(result[column]) == {expected}, f"outcome {column} differs from calendar"
        )
    try:
        result["outcome_availability_date"] = pd.to_datetime(
            result["outcome_availability_date"], errors="raise"
        ).dt.date
    except (TypeError, ValueError) as exc:
        raise Stage11ExecutionOrchestratorError(
            "outcome availability dates are invalid"
        ) from exc
    _require(
        type(knowledge_date) is dt.date
        and not result["outcome_availability_date"].isna().any()
        and max(result["outcome_availability_date"]) <= knowledge_date,
        "outcome was exposed before its public availability date",
    )
    result["holding_month"] = result["holding_month"].astype(str)
    _require(
        set(result["holding_month"]) == {row.holding_month},
        "outcome holding month differs",
    )
    raw_permno = pd.to_numeric(result["permno"], errors="coerce").to_numpy(float)
    _require(
        np.isfinite(raw_permno).all()
        and np.all(raw_permno > 0.0)
        and np.all(raw_permno == np.floor(raw_permno)),
        "outcome PERMNO values must be positive integers",
    )
    result["permno"] = raw_permno.astype(np.int64)
    _require(
        not result["permno"].duplicated().any(), "monthly outcome PERMNO is duplicated"
    )
    _require(
        tuple(sorted(int(value) for value in result["permno"])) == forecast_permnos,
        "monthly outcome domain differs from the committed forecast domain",
    )
    result["primary_label_status"] = result["primary_label_status"].astype(str)
    valid = result["primary_label_status"].eq(_PRESENT)
    for column in ("unranked_residual_return", "residual_return_rank"):
        numeric = pd.to_numeric(result[column], errors="coerce").to_numpy(float)
        _require(
            np.isfinite(numeric[valid]).all(), f"present outcome {column} is not finite"
        )
        _require(
            np.isnan(numeric[~valid]).all(), f"invalid outcome {column} must be null"
        )
        result[column] = numeric
    ranks = result.loc[valid, "residual_return_rank"].to_numpy(float)
    _require(
        np.all((ranks >= -0.5) & (ranks <= 0.5)),
        "residual target rank escaped [-0.5, 0.5]",
    )
    return result.sort_values("permno", kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class Stage11PseudoLiveLedger:
    """Append-only, hash-bound monthly forecast/outcome chronology."""

    plan: Stage11ExecutionPlan
    forecast_commits: tuple[MonthlyForecastCommit, ...] = ()
    outcome_commits: tuple[MonthlyOutcomeCommit, ...] = ()
    forecast_frames: tuple[pd.DataFrame, ...] = field(default=(), repr=False)
    outcome_frames: tuple[pd.DataFrame, ...] = field(default=(), repr=False)
    event_dates: tuple[dt.date, ...] = ()

    def __post_init__(self) -> None:
        _require(
            isinstance(self.plan, Stage11ExecutionPlan),
            "ledger requires an execution plan",
        )
        _require(
            len(self.forecast_commits) == len(self.forecast_frames)
            and len(self.outcome_commits) == len(self.outcome_frames),
            "ledger receipt/payload lengths differ",
        )
        _require(
            tuple(sorted(self.event_dates)) == self.event_dates,
            "pseudo-live ledger events are backdated",
        )
        forecast_months = tuple(value.holding_month for value in self.forecast_commits)
        outcome_months = tuple(value.holding_month for value in self.outcome_commits)
        _require(
            forecast_months == VALIDATION_HOLDING_MONTHS[: len(forecast_months)],
            "forecast commits are not the ordered validation prefix",
        )
        _require(
            len(set(outcome_months)) == len(outcome_months),
            "outcome month is duplicated",
        )
        _require(
            set(outcome_months).issubset(forecast_months),
            "outcome preceded its forecast",
        )

    def _row(self, holding_month: str) -> ValidationCalendarRow:
        rows = tuple(
            row for row in self.plan.calendar_rows if row.holding_month == holding_month
        )
        _require(len(rows) == 1, "holding month is outside the execution plan")
        return rows[0]

    def _forecast_domain(self, holding_month: str) -> tuple[int, ...]:
        index = tuple(value.holding_month for value in self.forecast_commits).index(
            holding_month
        )
        frame = self.forecast_frames[index]
        first = frame.loc[
            frame["model_family"].eq(FAMILY_ORDER[0])
            & frame["config_id"].eq(LOCKED_SHORTLIST[FAMILY_ORDER[0]][0])
            & pd.to_numeric(frame["seed"]).eq(VALIDATION_STOCHASTIC_SEEDS[0])
        ]
        return tuple(sorted(int(value) for value in first["permno"]))

    def commit_forecast(self, frame: pd.DataFrame) -> "Stage11PseudoLiveLedger":
        index = len(self.forecast_commits)
        _require(
            index < len(self.plan.calendar_rows),
            "all 59 forecasts are already committed",
        )
        row = self.plan.calendar_rows[index]
        _require(
            not self.event_dates or row.signal_date >= self.event_dates[-1],
            "forecast append is backdated",
        )
        normalized = _normalize_forecast_month(frame, row)
        domain = tuple(sorted(int(value) for value in normalized["permno"].unique()))
        receipt = MonthlyForecastCommit(
            holding_month=row.holding_month,
            signal_date=row.signal_date,
            entry_date=row.entry_date,
            exit_date=row.exit_date,
            rows=len(normalized),
            securities=len(domain),
            payload_sha256=canonical_frame_sha256(normalized),
        )
        return Stage11PseudoLiveLedger(
            plan=self.plan,
            forecast_commits=self.forecast_commits + (receipt,),
            outcome_commits=self.outcome_commits,
            forecast_frames=self.forecast_frames + (normalized,),
            outcome_frames=self.outcome_frames,
            event_dates=self.event_dates + (row.signal_date,),
        )

    def commit_outcome(
        self,
        holding_month: str,
        frame: pd.DataFrame,
        *,
        committed_at: dt.date | None = None,
    ) -> "Stage11PseudoLiveLedger":
        _require(
            holding_month in {value.holding_month for value in self.forecast_commits},
            "outcome arrived before its forecast commitment",
        )
        _require(
            holding_month
            not in {value.holding_month for value in self.outcome_commits},
            "outcome month is already committed",
        )
        row = self._row(holding_month)
        if committed_at is None:
            # Synthetic/direct callers may omit the event clock; the payload's
            # own authenticated availability then supplies the earliest legal
            # commit.  The empirical walk-forward always passes its current
            # pseudo-live knowledge date explicitly.
            _exact_columns(frame, OUTCOME_COLUMNS, "monthly residual outcome")
            try:
                availability = pd.to_datetime(
                    frame["outcome_availability_date"], errors="raise"
                ).dt.date
            except (TypeError, ValueError) as exc:
                raise Stage11ExecutionOrchestratorError(
                    "outcome availability dates are invalid"
                ) from exc
            _require(not availability.isna().any(), "outcome availability date is null")
            committed_at = max(row.exit_date, max(availability))
        _require(type(committed_at) is dt.date, "outcome commit date must be a date")
        _require(
            committed_at >= row.exit_date
            and (not self.event_dates or committed_at >= self.event_dates[-1]),
            "outcome append is backdated",
        )
        normalized = _normalize_outcome_month(
            frame,
            row,
            self._forecast_domain(holding_month),
            knowledge_date=committed_at,
        )
        maximum_availability = max(normalized["outcome_availability_date"])
        receipt = MonthlyOutcomeCommit(
            holding_month=holding_month,
            exit_date=row.exit_date,
            maximum_outcome_availability_date=maximum_availability,
            committed_at=committed_at,
            rows=len(normalized),
            valid_rows=int(normalized["primary_label_status"].eq(_PRESENT).sum()),
            payload_sha256=canonical_frame_sha256(normalized),
        )
        return Stage11PseudoLiveLedger(
            plan=self.plan,
            forecast_commits=self.forecast_commits,
            outcome_commits=self.outcome_commits + (receipt,),
            forecast_frames=self.forecast_frames,
            outcome_frames=self.outcome_frames + (normalized,),
            event_dates=self.event_dates + (committed_at,),
        )

    @property
    def complete(self) -> bool:
        return len(self.forecast_commits) == len(self.outcome_commits) == 59

    def gate(self) -> None:
        """Reopen every logical payload against its append-only receipt."""

        for commit, frame in zip(
            self.forecast_commits, self.forecast_frames, strict=True
        ):
            row = self._row(commit.holding_month)
            normalized = _normalize_forecast_month(frame, row)
            securities = int(normalized["permno"].nunique())
            _require(
                commit.signal_date == row.signal_date
                and commit.entry_date == row.entry_date
                and commit.exit_date == row.exit_date
                and commit.rows == len(normalized)
                and commit.securities == securities
                and commit.payload_sha256 == canonical_frame_sha256(normalized),
                f"forecast payload changed after commitment: {commit.holding_month}",
            )
        for commit, frame in zip(
            self.outcome_commits, self.outcome_frames, strict=True
        ):
            row = self._row(commit.holding_month)
            normalized = _normalize_outcome_month(
                frame,
                row,
                self._forecast_domain(commit.holding_month),
                knowledge_date=commit.committed_at,
            )
            valid = normalized["primary_label_status"].eq(_PRESENT)
            maximum_availability = max(normalized["outcome_availability_date"])
            _require(
                commit.exit_date == row.exit_date
                and commit.rows == len(normalized)
                and commit.valid_rows == int(valid.sum())
                and commit.maximum_outcome_availability_date == maximum_availability
                and commit.payload_sha256 == canonical_frame_sha256(normalized),
                f"outcome payload changed after commitment: {commit.holding_month}",
            )

    def combined_forecasts(self) -> pd.DataFrame:
        _require(bool(self.forecast_frames), "ledger has no forecast payload")
        return pd.concat(self.forecast_frames, ignore_index=True)

    def combined_outcomes(self) -> pd.DataFrame:
        _require(bool(self.outcome_frames), "ledger has no outcome payload")
        return pd.concat(self.outcome_frames, ignore_index=True)

    @property
    def ledger_sha256(self) -> str:
        self.gate()
        payload = {
            "plan_sha256": self.plan.plan_sha256,
            "forecasts": [
                {
                    **asdict(value),
                    "signal_date": value.signal_date.isoformat(),
                    "entry_date": value.entry_date.isoformat(),
                    "exit_date": value.exit_date.isoformat(),
                }
                for value in self.forecast_commits
            ],
            "outcomes": [
                {
                    **asdict(value),
                    "exit_date": value.exit_date.isoformat(),
                    "maximum_outcome_availability_date": (
                        value.maximum_outcome_availability_date.isoformat()
                    ),
                    "committed_at": value.committed_at.isoformat(),
                }
                for value in self.outcome_commits
            ],
        }
        return _sha256_bytes(_canonical_json_bytes(payload))


class Stage11WalkForwardBackend(Protocol):
    """Narrow port controlled by :func:`run_locked_walk_forward`."""

    def fit_selector(
        self,
        job: SelectorFitJob,
        completed_outcomes: pd.DataFrame,
    ) -> str:
        """Fit/reuse one job and return its authenticated checkpoint SHA-256."""

    def score_month(
        self,
        step: MonthlyExecutionStep,
        checkpoints: Mapping[tuple[str, str, int | None], str],
    ) -> pd.DataFrame:
        """Return all 33 raw forecast streams for the current month."""

    def read_outcome_month(
        self,
        row: ValidationCalendarRow,
        knowledge_date: dt.date,
    ) -> pd.DataFrame | None:
        """Return a complete, available label month or ``None`` if not ready.

        The backend must inspect availability metadata before materializing
        any outcome value and may return the values only when every row is
        public on or before ``knowledge_date``.
        """


class AuthorityBoundStage11WalkForwardBackend:
    """Base class for the only backend admitted to empirical execution.

    The pure protocol above intentionally remains available to synthetic tests.
    An empirical run is stricter: its backend must retain the same live
    descriptor-anchored authority and the same access broker for its complete
    lifetime.  Subclasses implement the three protocol methods, and should
    call :meth:`gate` immediately before and after every filesystem or native
    worker boundary.
    """

    def __init__(self, authority: object, access_broker: object) -> None:
        from portfolio_bot.validation.execution_access import (
            ValidationExecutionAccessBroker,
        )
        from portfolio_bot.validation.validation_runner import (
            require_live_stage11_validation_unblind_authority,
        )

        authenticated = require_live_stage11_validation_unblind_authority(authority)
        _require(
            type(access_broker) is ValidationExecutionAccessBroker,
            "empirical walk-forward requires the exact execution access broker",
        )
        ledger = access_broker.ledger()
        _require(
            ledger.authority_build_id == authenticated.build_id
            and ledger.authority_verification_id == authenticated.verification_id
            and ledger.approval_contract_file_sha256
            == authenticated.approval_contract_file_sha256
            and ledger.approval_contract_payload_sha256
            == authenticated.approval_contract_payload_sha256
            and ledger.execution_input_manifest_sha256
            == authenticated.execution_input_manifest_sha256,
            "empirical backend broker is bound to another authority",
        )
        self._stage11_authority = authenticated
        self._stage11_access_broker = access_broker

    @property
    def access_broker(self) -> object:
        self.gate()
        return self._stage11_access_broker

    def gate(self) -> None:
        from portfolio_bot.validation.validation_runner import (
            require_live_stage11_validation_unblind_authority,
        )

        current = require_live_stage11_validation_unblind_authority(
            self._stage11_authority
        )
        ledger = self._stage11_access_broker.ledger()
        _require(
            ledger.authority_build_id == current.build_id
            and ledger.authority_verification_id == current.verification_id
            and ledger.approval_contract_file_sha256
            == current.approval_contract_file_sha256
            and ledger.approval_contract_payload_sha256
            == current.approval_contract_payload_sha256
            and ledger.execution_input_manifest_sha256
            == current.execution_input_manifest_sha256,
            "empirical backend authority changed during execution",
        )

    def fit_selector(
        self,
        job: SelectorFitJob,
        completed_outcomes: pd.DataFrame,
    ) -> str:
        raise NotImplementedError

    def score_month(
        self,
        step: MonthlyExecutionStep,
        checkpoints: Mapping[tuple[str, str, int | None], str],
    ) -> pd.DataFrame:
        raise NotImplementedError

    def read_outcome_month(
        self,
        row: ValidationCalendarRow,
        knowledge_date: dt.date,
    ) -> pd.DataFrame | None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class WalkForwardRunReceipt:
    ledger: Stage11PseudoLiveLedger
    checkpoint_sha256_by_refit_and_model: Mapping[
        tuple[dt.date, str, str, int | None], str
    ]
    refit_count: int
    selector_fit_count: int
    monthly_score_count: int


def run_locked_walk_forward(
    plan: Stage11ExecutionPlan,
    backend: Stage11WalkForwardBackend,
    *,
    liveness_gate: Callable[[], None] = lambda: None,
) -> WalkForwardRunReceipt:
    """Execute the frozen chronology through a caller-supplied, gated backend.

    The empirical entrypoint supplies a liveness gate that reauthenticates the
    descriptor-anchored unblind authority before and after every backend call.
    Synthetic tests use the default no-op gate and an in-memory backend.
    """

    _require(isinstance(plan, Stage11ExecutionPlan), "walk-forward plan is invalid")
    _require(callable(liveness_gate), "walk-forward liveness gate must be callable")
    ledger = Stage11PseudoLiveLedger(plan)
    checkpoints: dict[tuple[str, str, int | None], str] = {}
    checkpoint_history: dict[tuple[dt.date, str, str, int | None], str] = {}
    fit_count = 0
    refit_count = 0

    def attach_completed(as_of: dt.date) -> None:
        nonlocal ledger
        pending = tuple(
            row
            for row in plan.calendar_rows
            if row.holding_month
            in {value.holding_month for value in ledger.forecast_commits}
            and row.holding_month
            not in {value.holding_month for value in ledger.outcome_commits}
            and row.exit_date <= as_of
        )
        for row in pending:
            liveness_gate()
            outcome = backend.read_outcome_month(row, as_of)
            liveness_gate()
            if outcome is None:
                continue
            ledger = ledger.commit_outcome(
                row.holding_month,
                outcome,
                committed_at=as_of,
            )

    for step in plan.monthly_steps:
        attach_completed(step.calendar_row.signal_date)
        if step.refit_now:
            refit_count += 1
            completed = (
                ledger.combined_outcomes()
                if ledger.outcome_frames
                else pd.DataFrame(columns=OUTCOME_COLUMNS)
            )
            jobs = tuple(
                job
                for job in plan.fit_jobs
                if job.refit_index == step.active_refit_index
            )
            _require(
                len(jobs) == EXPECTED_FITS_PER_REFIT, "active refit job count differs"
            )
            new_checkpoints: dict[tuple[str, str, int | None], str] = {}
            for job in jobs:
                liveness_gate()
                digest = backend.fit_selector(job, completed.copy(deep=True))
                liveness_gate()
                _require(
                    _is_sha256(digest),
                    "selector backend returned a malformed checkpoint digest",
                )
                key = (job.model_family, job.config_id, job.seed)
                new_checkpoints[key] = digest
                checkpoint_history[(job.refit_signal_date, *key)] = digest
                fit_count += 1
            _require(
                tuple(new_checkpoints) == _fit_job_keys(),
                "active checkpoint grid differs",
            )
            checkpoints = new_checkpoints
        _require(
            len(checkpoints) == EXPECTED_FITS_PER_REFIT,
            "monthly scoring lacks 33 checkpoints",
        )
        liveness_gate()
        forecasts = backend.score_month(step, MappingProxyType(dict(checkpoints)))
        liveness_gate()
        ledger = ledger.commit_forecast(forecasts)

    # The last exit occurs after the last signal.  No selection or portfolio
    # operation is permitted until every one of the 59 committed forecasts has
    # received its completed outcome.
    attach_completed(plan.calendar_rows[-1].exit_date)
    _require(ledger.complete, "walk-forward ended without all 59 completed outcomes")
    _require(
        refit_count == EXPECTED_REFIT_COUNT and fit_count == EXPECTED_TOTAL_FITS,
        "walk-forward refit/fit counts differ from the frozen plan",
    )
    ledger.gate()
    return WalkForwardRunReceipt(
        ledger=ledger,
        checkpoint_sha256_by_refit_and_model=MappingProxyType(checkpoint_history),
        refit_count=refit_count,
        selector_fit_count=fit_count,
        monthly_score_count=len(plan.monthly_steps),
    )


def run_empirical_locked_walk_forward(
    authority: object,
    access_broker: object,
    plan: Stage11ExecutionPlan,
    backend: AuthorityBoundStage11WalkForwardBackend,
) -> WalkForwardRunReceipt:
    """Run the chronology engine with one live authority-bound backend."""

    from portfolio_bot.validation.access import PayloadRole
    from portfolio_bot.validation.execution_access import (
        ValidationExecutionAccessBroker,
    )
    from portfolio_bot.validation.validation_runner import (
        require_live_stage11_validation_unblind_authority,
    )

    # Authenticate before examining the plan or backend.
    require_live_stage11_validation_unblind_authority(authority)
    _require(
        type(access_broker) is ValidationExecutionAccessBroker,
        "empirical walk-forward requires the exact execution access broker",
    )
    _require(
        isinstance(backend, AuthorityBoundStage11WalkForwardBackend)
        and backend._stage11_authority is authority
        and backend._stage11_access_broker is access_broker,
        "empirical backend is not bound to this authority and broker",
    )

    def gate() -> None:
        require_live_stage11_validation_unblind_authority(authority)
        access_broker.ledger()
        backend.gate()

    gate()
    receipt = run_locked_walk_forward(plan, backend, liveness_gate=gate)
    gate()
    access_broker.record_created_output(
        PayloadRole.REAL_VALIDATION_FORECAST,
        rows=sum(value.rows for value in receipt.ledger.forecast_commits),
        minimum_date=plan.calendar_rows[0].signal_date,
        maximum_date=plan.calendar_rows[-1].signal_date,
        holding_months=tuple(row.holding_month for row in plan.calendar_rows),
    )
    gate()
    return receipt


@dataclass(frozen=True)
class LockedValidationSelection:
    family_selections: Mapping[str, FamilySelection]
    ensemble_selection: EnsembleSelection
    configuration_scores: pd.DataFrame
    selected_scores: pd.DataFrame
    family_monthly_ic: pd.DataFrame
    family_metadata: pd.DataFrame
    ensemble_monthly_ic: pd.DataFrame
    ensemble_metadata: pd.DataFrame
    provenance: ValidationForecastProvenance
    selection_sha256: str


def _configuration_scores(forecasts: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    key = ["model_family", "config_id", "signal_date", "holding_month", "permno"]
    for values, group in forecasts.groupby(key, sort=True, observed=True, dropna=False):
        family = str(values[0])
        if family == "elastic_net":
            _require(
                group["seed"].isna().all() and len(group) == 1,
                "Elastic Net forecast multiplicity differs",
            )
        else:
            seeds = tuple(sorted(int(value) for value in pd.to_numeric(group["seed"])))
            _require(
                seeds == VALIDATION_STOCHASTIC_SEEDS, "configuration seed set differs"
            )
        record = dict(zip(key, values, strict=True))
        record["raw_forecast_mean"] = float(group["raw_forecast"].mean())
        records.append(record)
    result = pd.DataFrame.from_records(records)
    result["score"] = np.nan
    grouping = ["model_family", "config_id", "signal_date", "holding_month"]
    for positions in result.groupby(
        grouping, sort=True, observed=True
    ).indices.values():
        indices = np.asarray(positions, dtype=np.int64)
        result.iloc[indices, result.columns.get_loc("score")] = centered_average_rank(
            result.iloc[indices]["raw_forecast_mean"]
        )
    return result.sort_values([*grouping, "permno"], kind="mergesort").reset_index(
        drop=True
    )


def _monthly_rank_ic(
    scores: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    candidate_columns: tuple[str, ...],
    score_column: str,
) -> pd.DataFrame:
    left = scores.copy()
    right = outcomes[
        [
            "holding_month",
            "permno",
            "unranked_residual_return",
            "primary_label_status",
        ]
    ].copy()
    # ``compose_locked_ensemble_scores`` intentionally canonicalizes months to
    # ``Period[M]`` while the outcome store uses its frozen YYYY-MM strings.
    # Normalize only the join key; the selection functions recanonicalize it.
    left["holding_month"] = left["holding_month"].astype(str)
    right["holding_month"] = right["holding_month"].astype(str)
    joined = left.merge(
        right,
        on=["holding_month", "permno"],
        how="left",
        validate="many_to_one",
    )
    _require(
        not joined["primary_label_status"].isna().any(), "score/outcome domain differs"
    )
    records: list[dict[str, object]] = []
    grouping = [*candidate_columns, "holding_month"]
    for values, group in joined.groupby(
        grouping, sort=True, observed=True, dropna=False
    ):
        key_values = values if isinstance(values, tuple) else (values,)
        valid = group["primary_label_status"].eq(_PRESENT)
        result = correlation_with_status(
            group.loc[valid, score_column],
            group.loc[valid, "unranked_residual_return"],
            method="spearman",
        )
        record = dict(zip(grouping, key_values, strict=True))
        record["rank_ic"] = result.value
        records.append(record)
    return pd.DataFrame.from_records(records)


def derive_locked_validation_selection(
    ledger: Stage11PseudoLiveLedger,
) -> LockedValidationSelection:
    """Apply the prespecified IC-only family and ensemble selection once."""

    _require(
        type(ledger) is Stage11PseudoLiveLedger and ledger.complete,
        "selection requires a complete pseudo-live ledger",
    )
    forecasts = ledger.combined_forecasts()
    outcomes = ledger.combined_outcomes()
    provenance = ValidationForecastProvenance(
        source_receipt_sha256=ledger.ledger_sha256
    )
    scores = _configuration_scores(forecasts)
    family_ic = _monthly_rank_ic(
        scores,
        outcomes,
        candidate_columns=("model_family", "config_id"),
        score_column="score",
    )[["model_family", "config_id", "holding_month", "rank_ic"]]
    turnover_input = scores[
        ["model_family", "config_id", "holding_month", "permno", "score"]
    ]
    family_metadata = mean_adjacent_signal_turnover(
        turnover_input,
        candidate_columns=("model_family", "config_id"),
    )[["model_family", "config_id", "signal_turnover"]]
    family = select_family_specifications(
        family_ic, family_metadata, provenance=provenance
    )

    nonlinear_raw = forecasts.loc[
        forecasts["model_family"].isin(("neural_network", "xgboost"))
        & (
            (
                forecasts["model_family"].eq("neural_network")
                & forecasts["config_id"].eq(family["neural_network"].config_id)
            )
            | (
                forecasts["model_family"].eq("xgboost")
                & forecasts["config_id"].eq(family["xgboost"].config_id)
            )
        )
    ].copy()
    ensemble_score_rows: list[pd.DataFrame] = []
    ensemble_ic_rows: list[pd.DataFrame] = []
    ensemble_turnover_rows: list[pd.DataFrame] = []
    for weight in LOCKED_ENSEMBLE_NN_WEIGHTS:
        composed = compose_locked_ensemble_scores(
            nonlinear_raw,
            selected_nn_config_id=family["neural_network"].config_id,
            selected_xgboost_config_id=family["xgboost"].config_id,
            nn_weight=weight,
            provenance=provenance,
        ).scores.copy()
        composed["nn_weight"] = weight
        ensemble_score_rows.append(composed)
        monthly = _monthly_rank_ic(
            composed.rename(columns={"ensemble_score": "score"}),
            outcomes,
            candidate_columns=("nn_weight",),
            score_column="score",
        )[["nn_weight", "holding_month", "rank_ic"]]
        ensemble_ic_rows.append(monthly)
        turnover = mean_adjacent_signal_turnover(
            composed[["nn_weight", "holding_month", "permno", "ensemble_score"]].rename(
                columns={"ensemble_score": "score"}
            ),
            candidate_columns=("nn_weight",),
        )[["nn_weight", "signal_turnover"]]
        ensemble_turnover_rows.append(turnover)
    ensemble_ic = pd.concat(ensemble_ic_rows, ignore_index=True)
    ensemble_metadata = pd.concat(ensemble_turnover_rows, ignore_index=True)
    ensemble = select_ensemble_weight(
        ensemble_ic,
        ensemble_metadata,
        nn_annual_gate_pass=family["neural_network"].annual_ic_gate_pass,
        xgboost_annual_gate_pass=family["xgboost"].annual_ic_gate_pass,
        provenance=provenance,
    )

    if ensemble.effective_nn_weight is None:
        effective_scores = pd.DataFrame(
            columns=["signal_date", "holding_month", "permno", "ensemble_score"]
        )
    else:
        effective_scores = pd.concat(ensemble_score_rows, ignore_index=True)
        effective_scores = effective_scores.loc[
            np.isclose(
                effective_scores["nn_weight"].to_numpy(float),
                ensemble.effective_nn_weight,
                atol=0.0,
            ),
            ["signal_date", "holding_month", "permno", "ensemble_score"],
        ]
    enet = scores.loc[
        scores["model_family"].eq("elastic_net")
        & scores["config_id"].eq(family["elastic_net"].config_id),
        ["signal_date", "holding_month", "permno", "score"],
    ].rename(columns={"score": "elastic_net_score"})
    enet = enet.copy()
    enet["holding_month"] = enet["holding_month"].astype(str)
    effective_scores = effective_scores.copy()
    if not effective_scores.empty:
        effective_scores["holding_month"] = effective_scores["holding_month"].astype(
            str
        )
    if effective_scores.empty:
        selected = enet.copy()
        selected["ensemble_score"] = np.nan
    else:
        selected = effective_scores.merge(
            enet,
            on=["signal_date", "holding_month", "permno"],
            how="outer",
            validate="one_to_one",
        )
    selected = selected.sort_values(
        ["holding_month", "permno"], kind="mergesort"
    ).reset_index(drop=True)
    selection_payload = {
        "ledger_sha256": ledger.ledger_sha256,
        "families": {
            name: {
                "config_id": value.config_id,
                "mean_rank_ic": value.mean_rank_ic.hex(),
                "annual_gate": value.annual_ic_gate_pass,
            }
            for name, value in sorted(family.items())
        },
        "ensemble": {
            "diagnostic_nn_weight": ensemble.diagnostic_nn_weight.hex(),
            "effective_nn_weight": (
                None
                if ensemble.effective_nn_weight is None
                else ensemble.effective_nn_weight.hex()
            ),
            "gate_reason": ensemble.gate_reason,
        },
        "selected_scores_sha256": canonical_frame_sha256(selected),
    }
    return LockedValidationSelection(
        family_selections=MappingProxyType(dict(family)),
        ensemble_selection=ensemble,
        configuration_scores=scores.copy(deep=True),
        selected_scores=selected.copy(deep=True),
        family_monthly_ic=family_ic.copy(deep=True),
        family_metadata=family_metadata.copy(deep=True),
        ensemble_monthly_ic=ensemble_ic.copy(deep=True),
        ensemble_metadata=ensemble_metadata.copy(deep=True),
        provenance=provenance,
        selection_sha256=_sha256_bytes(_canonical_json_bytes(selection_payload)),
    )


def derive_empirical_locked_validation_selection(
    authority: object,
    access_broker: object,
    ledger: Stage11PseudoLiveLedger,
) -> LockedValidationSelection:
    """Derive and ledger the frozen IC-only empirical selection."""

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
        "empirical selection requires the exact execution access broker",
    )
    access_broker.ledger()
    selection = derive_locked_validation_selection(ledger)
    require_live_stage11_validation_unblind_authority(authority)
    access_broker.record_created_output(
        PayloadRole.REAL_VALIDATION_SELECTION,
        rows=len(selection.family_selections) + 1,
        minimum_date=ledger.plan.calendar_rows[0].entry_date,
        maximum_date=ledger.plan.calendar_rows[-1].exit_date,
        holding_months=tuple(row.holding_month for row in ledger.plan.calendar_rows),
    )
    require_live_stage11_validation_unblind_authority(authority)
    return selection


@dataclass(frozen=True, slots=True)
class StrategyCalibrationSchedule:
    strategy_id: str
    receipts: tuple[CalibrationReceipt, ...]
    schedule_sha256: str

    def kappa_for_signal(self, signal_date: dt.date) -> float:
        eligible = tuple(
            value for value in self.receipts if value.fit_signal_date <= signal_date
        )
        _require(bool(eligible), "no calibration fit precedes the scoring signal")
        return eligible[-1].kappa


def _validation_calibration_observations(
    *,
    strategy_id: str,
    score_column: str,
    selected_scores: pd.DataFrame,
    ledger: Stage11PseudoLiveLedger,
) -> tuple[CalibrationObservation, ...]:
    outcomes = ledger.combined_outcomes()
    merged = selected_scores[
        ["signal_date", "holding_month", "permno", score_column]
    ].merge(
        outcomes,
        on=["signal_date", "holding_month", "permno"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.loc[
        merged["primary_label_status"].eq(_PRESENT) & merged[score_column].notna()
    ].copy()
    observations: list[CalibrationObservation] = []
    for holding_month, month in merged.groupby(
        "holding_month", sort=True, observed=True
    ):
        payload_hash = selection_frame_sha256(
            month[["signal_date", "holding_month", "permno", score_column]].rename(
                columns={score_column: "score"}
            )
        )
        for row in month.itertuples(index=False):
            observations.append(
                CalibrationObservation(
                    forecast_id=f"{strategy_id}:{holding_month}:{int(row.permno)}",
                    holding_month=str(holding_month),
                    permno=int(row.permno),
                    score=float(getattr(row, score_column)),
                    residual_return=float(row.unranked_residual_return),
                    signal_date=row.signal_date,
                    entry_date=row.entry_date,
                    exit_date=row.exit_date,
                    forecast_payload_sha256=payload_hash,
                    forecast_committed_at=dt.datetime.combine(
                        row.signal_date,
                        dt.time(12, 0),
                        tzinfo=dt.timezone.utc,
                    ),
                    outcome_available_at=dt.datetime.combine(
                        row.outcome_availability_date,
                        dt.time(23, 59, 59),
                        tzinfo=dt.timezone.utc,
                    ),
                    source="validation_prior_live",
                )
            )
    return tuple(observations)


def build_strategy_calibration_schedules(
    selection: LockedValidationSelection,
    ledger: Stage11PseudoLiveLedger,
    *,
    development_oof: Mapping[str, Sequence[CalibrationObservation]],
) -> Mapping[str, StrategyCalibrationSchedule]:
    """Fit exact quarterly, past-only calibrations for primary and comparator."""

    _require(
        set(development_oof) == {PRIMARY_STRATEGY_ID, ELASTIC_NET_CHALLENGER_ID},
        "development OOF calibration inputs must contain primary and Elastic Net",
    )
    strategy_columns = {
        PRIMARY_STRATEGY_ID: "ensemble_score",
        ELASTIC_NET_CHALLENGER_ID: "elastic_net_score",
    }
    result: dict[str, StrategyCalibrationSchedule] = {}
    for strategy_id, score_column in strategy_columns.items():
        development = tuple(development_oof[strategy_id])
        _require(
            bool(development), f"{strategy_id} development OOF calibration is empty"
        )
        _require(
            all(
                isinstance(value, CalibrationObservation)
                and value.source == "development_oof"
                for value in development
            ),
            "development calibration contains a non-OOF row",
        )
        validation = _validation_calibration_observations(
            strategy_id=strategy_id,
            score_column=score_column,
            selected_scores=selection.selected_scores,
            ledger=ledger,
        )
        receipts: list[CalibrationReceipt] = []
        for refit_signal_date in ledger.plan.refit_signal_dates:
            prior_validation = tuple(
                value
                for value in validation
                if value.exit_date <= refit_signal_date
                and value.outcome_available_at.date() <= refit_signal_date
            )
            receipt = fit_nonnegative_huber_calibration(
                (*development, *prior_validation),
                fit_signal_date=refit_signal_date,
            )
            _require(
                receipt.validation_prior_live_observations == len(prior_validation),
                "calibration prior-live count differs",
            )
            receipts.append(receipt)
        _require(
            len(receipts) == EXPECTED_REFIT_COUNT,
            "calibration schedule must have 20 refits",
        )
        payload = [
            {
                **asdict(value),
                "fit_signal_date": value.fit_signal_date.isoformat(),
                "kappa": value.kappa.hex(),
                "unconstrained_slope": value.unconstrained_slope.hex(),
                "half_life_months": value.half_life_months.hex(),
                "huber_epsilon": value.huber_epsilon.hex(),
            }
            for value in receipts
        ]
        result[strategy_id] = StrategyCalibrationSchedule(
            strategy_id=strategy_id,
            receipts=tuple(receipts),
            schedule_sha256=_sha256_bytes(_canonical_json_bytes(payload)),
        )
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class StrategyAllocation:
    strategy_id: str
    signal_source: str
    allocator_architecture: str
    active: bool
    inactive_reason: str | None
    problem: MonthlyProblemBuild | None
    result: AllocationResult | None


@dataclass(frozen=True, slots=True)
class StrategyAllocationFailure:
    """One diagnostic solver failure with no portfolio-performance fields."""

    strategy_id: str
    allocator_architecture: str
    error_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class DiagnosticMonthlyAllocation:
    """All successful allocations, or every active failure for one month."""

    allocations: Mapping[str, StrategyAllocation] | None
    failures: tuple[StrategyAllocationFailure, ...]

    def __post_init__(self) -> None:
        _require(
            (self.allocations is None) == bool(self.failures),
            "diagnostic allocation must contain either allocations or failures",
        )
        _require(
            tuple(value.strategy_id for value in self.failures)
            == tuple(
                sorted(
                    value.strategy_id for value in self.failures
                )
            ),
            "diagnostic allocation failures must be strategy-sorted",
        )


def _assemble_locked_monthly_allocations(
    *,
    primary_problem: MonthlyProblemBuild | None,
    primary_result: AllocationResult | None,
    rank_problem: MonthlyProblemBuild | None,
    rank_result: AllocationResult | None,
    elastic_net_problem: MonthlyProblemBuild,
    elastic_net_result: AllocationResult,
    active_v1_ml_strategy_qualifies: bool,
) -> Mapping[str, StrategyAllocation]:
    """Assemble the frozen strategy map after every active solve succeeds."""

    decision_active = bool(PRODUCTION_TRAINING_RUNTIME_CERTIFIED) and bool(
        active_v1_ml_strategy_qualifies
    )
    decision_reason = (
        None
        if decision_active
        else (
            "nonlinear_family_gate_failed"
            if not active_v1_ml_strategy_qualifies
            else "production_training_runtime_not_certified"
        )
    )
    _require(
        not decision_active,
        "decision-focused certification is active but no reviewed empirical adapter is connected",
    )
    return MappingProxyType(
        {
            PRIMARY_STRATEGY_ID: StrategyAllocation(
                PRIMARY_STRATEGY_ID,
                "selected_nn_xgboost_ensemble",
                "hybrid_alpha_risk_cost_convex",
                bool(active_v1_ml_strategy_qualifies),
                (
                    None
                    if active_v1_ml_strategy_qualifies
                    else "both_nonlinear_families_failed"
                ),
                primary_problem,
                primary_result,
            ),
            ELASTIC_NET_CHALLENGER_ID: StrategyAllocation(
                ELASTIC_NET_CHALLENGER_ID,
                "selected_elastic_net",
                "hybrid_alpha_risk_cost_convex",
                True,
                None,
                elastic_net_problem,
                elastic_net_result,
            ),
            RANK_CHALLENGER_ID: StrategyAllocation(
                RANK_CHALLENGER_ID,
                "selected_nn_xgboost_ensemble",
                "rank_volatility_projected",
                bool(active_v1_ml_strategy_qualifies),
                (
                    None
                    if active_v1_ml_strategy_qualifies
                    else "both_nonlinear_families_failed"
                ),
                rank_problem,
                rank_result,
            ),
            DECISION_FOCUSED_CHALLENGER_ID: StrategyAllocation(
                DECISION_FOCUSED_CHALLENGER_ID,
                "selected_nn_xgboost_ensemble",
                "decision_focused_neural_projected",
                False,
                decision_reason,
                None,
                None,
            ),
        }
    )


def allocate_locked_monthly_strategies(
    *,
    primary_problem: MonthlyProblemBuild | None,
    rank_problem: MonthlyProblemBuild | None,
    elastic_net_problem: MonthlyProblemBuild,
    active_v1_ml_strategy_qualifies: bool,
) -> Mapping[str, StrategyAllocation]:
    """Call the frozen Stage 10 allocators with no validation architecture choice."""

    _require(
        isinstance(elastic_net_problem, MonthlyProblemBuild),
        "monthly allocation requires an authenticated Elastic-Net problem",
    )
    if active_v1_ml_strategy_qualifies:
        _require(
            isinstance(primary_problem, MonthlyProblemBuild)
            and isinstance(rank_problem, MonthlyProblemBuild)
            and len(
                {
                    id(primary_problem),
                    id(rank_problem),
                    id(elastic_net_problem),
                }
            )
            == 3
            and primary_problem.identity
            == rank_problem.identity
            == elastic_net_problem.identity,
            "active primary, rank, and Elastic-Net problems must be distinct and bind the same monthly snapshot",
        )
    else:
        _require(
            primary_problem is None and rank_problem is None,
            "inactive nonlinear strategies must not receive allocation problems",
        )
    try:
        primary_result = (
            solve_hybrid(primary_problem.problem)  # type: ignore[union-attr]
            if active_v1_ml_strategy_qualifies
            else None
        )
    except AllocationSolveError as exc:
        raise AllocationSolveError(
            f"{PRIMARY_STRATEGY_ID} allocation failed: {exc}"
        ) from exc
    rank_result = None
    if active_v1_ml_strategy_qualifies:
        try:
            _, rank_result = allocate_rank(
                rank_problem.problem,  # type: ignore[union-attr]
                long_entry_boundary=rank_problem.selection.long_entry_boundary,  # type: ignore[union-attr]
                short_entry_boundary=rank_problem.selection.short_entry_boundary,  # type: ignore[union-attr]
            )
        except AllocationSolveError as exc:
            raise AllocationSolveError(
                f"{RANK_CHALLENGER_ID} allocation failed: {exc}"
            ) from exc
    try:
        enet_result = solve_hybrid(elastic_net_problem.problem)
    except AllocationSolveError as exc:
        raise AllocationSolveError(
            f"{ELASTIC_NET_CHALLENGER_ID} allocation failed: {exc}"
        ) from exc
    return _assemble_locked_monthly_allocations(
        primary_problem=primary_problem,
        primary_result=primary_result,
        rank_problem=rank_problem,
        rank_result=rank_result,
        elastic_net_problem=elastic_net_problem,
        elastic_net_result=enet_result,
        active_v1_ml_strategy_qualifies=active_v1_ml_strategy_qualifies,
    )


def diagnose_locked_monthly_strategies(
    *,
    primary_problem: MonthlyProblemBuild | None,
    rank_problem: MonthlyProblemBuild | None,
    elastic_net_problem: MonthlyProblemBuild,
    active_v1_ml_strategy_qualifies: bool,
) -> DiagnosticMonthlyAllocation:
    """Attempt every active allocation before returning any monthly failure.

    This diagnostic-only wrapper preserves the frozen allocators and order but
    does not short-circuit after the first exception. Accounting remains
    prohibited unless all active strategies succeed.
    """

    _require(
        isinstance(elastic_net_problem, MonthlyProblemBuild),
        "monthly allocation requires an authenticated Elastic-Net problem",
    )
    if active_v1_ml_strategy_qualifies:
        _require(
            isinstance(primary_problem, MonthlyProblemBuild)
            and isinstance(rank_problem, MonthlyProblemBuild)
            and len(
                {
                    id(primary_problem),
                    id(rank_problem),
                    id(elastic_net_problem),
                }
            )
            == 3
            and primary_problem.identity
            == rank_problem.identity
            == elastic_net_problem.identity,
            "active primary, rank, and Elastic-Net problems must be distinct and bind the same monthly snapshot",
        )
    else:
        _require(
            primary_problem is None and rank_problem is None,
            "inactive nonlinear strategies must not receive allocation problems",
        )

    results: dict[str, AllocationResult] = {}
    failures: list[StrategyAllocationFailure] = []
    attempts: list[tuple[str, str, Callable[[], AllocationResult]]] = []
    if active_v1_ml_strategy_qualifies:
        assert primary_problem is not None and rank_problem is not None
        attempts.extend(
            (
                (
                    PRIMARY_STRATEGY_ID,
                    "hybrid_alpha_risk_cost_convex",
                    lambda: solve_hybrid(primary_problem.problem),
                ),
                (
                    RANK_CHALLENGER_ID,
                    "rank_volatility_projected",
                    lambda: allocate_rank(
                        rank_problem.problem,
                        long_entry_boundary=(
                            rank_problem.selection.long_entry_boundary
                        ),
                        short_entry_boundary=(
                            rank_problem.selection.short_entry_boundary
                        ),
                    )[1],
                ),
            )
        )
    attempts.append(
        (
            ELASTIC_NET_CHALLENGER_ID,
            "hybrid_alpha_risk_cost_convex",
            lambda: solve_hybrid(elastic_net_problem.problem),
        )
    )
    for strategy_id, architecture, solve in attempts:
        try:
            results[strategy_id] = solve()
        except Exception as exc:
            failures.append(
                StrategyAllocationFailure(
                    strategy_id=strategy_id,
                    allocator_architecture=architecture,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
    if failures:
        return DiagnosticMonthlyAllocation(
            allocations=None,
            failures=tuple(
                sorted(failures, key=lambda value: value.strategy_id)
            ),
        )
    allocations = _assemble_locked_monthly_allocations(
        primary_problem=primary_problem,
        primary_result=results.get(PRIMARY_STRATEGY_ID),
        rank_problem=rank_problem,
        rank_result=results.get(RANK_CHALLENGER_ID),
        elastic_net_problem=elastic_net_problem,
        elastic_net_result=results[ELASTIC_NET_CHALLENGER_ID],
        active_v1_ml_strategy_qualifies=active_v1_ml_strategy_qualifies,
    )
    return DiagnosticMonthlyAllocation(allocations=allocations, failures=())


@dataclass(frozen=True, slots=True)
class StrategyAccountingInput:
    """Opening state and entry-close price aligned to one solved problem."""

    strategy_id: str
    problem: MonthlyProblemBuild
    opening_state: PortfolioState
    execution_price: np.ndarray
    action_ledger: CorporateActionLedger = field(default_factory=CorporateActionLedger)
    mark_continuity_ledger: MarkContinuityLedger = field(
        default_factory=MarkContinuityLedger.empty
    )

    def __post_init__(self) -> None:
        _require(
            isinstance(self.problem, MonthlyProblemBuild), "accounting problem differs"
        )
        _require(
            isinstance(self.opening_state, PortfolioState),
            "opening portfolio state differs",
        )
        _require(
            isinstance(self.action_ledger, CorporateActionLedger),
            "corporate-action ledger differs",
        )
        _require(
            isinstance(self.mark_continuity_ledger, MarkContinuityLedger),
            "mark-continuity ledger differs",
        )
        prices = np.asarray(self.execution_price, dtype=np.float64).copy()
        _require(
            prices.shape == (self.problem.problem.n_assets,),
            "entry execution prices do not align to the solved problem",
        )
        _require(
            not np.isinf(prices).any(), "entry execution prices may be finite or NaN"
        )
        prices.setflags(write=False)
        object.__setattr__(self, "execution_price", prices)


@dataclass(frozen=True, slots=True)
class StrategyAccountingFailure:
    """One performance-blind diagnostic failure from monthly accounting."""

    strategy_id: str
    phase: str
    failure_code: str
    exception_class: str
    private_detail: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require(bool(self.strategy_id), "accounting failure strategy is empty")
        _require(
            self.phase in {"accounting_preflight", "holding_period_accounting"},
            "accounting failure phase differs",
        )
        _require(
            self.failure_code.startswith(f"{self.phase}."),
            "accounting failure code differs from its phase",
        )
        _require(
            bool(self.exception_class),
            "accounting failure exception class is empty",
        )
        _require(
            type(self.private_detail) is bytes and bool(self.private_detail),
            "accounting failure private detail must be nonempty bytes",
        )


@dataclass(frozen=True, slots=True)
class DiagnosticMonthlyAccounting:
    """All active accounting results, or every failure observed for one month."""

    accounting: Mapping[str, DatedHoldingPeriodAccounting] | None
    failures: tuple[StrategyAccountingFailure, ...]

    def __post_init__(self) -> None:
        _require(
            (self.accounting is None) == bool(self.failures),
            "diagnostic accounting must contain either results or failures",
        )
        strategy_ids = tuple(value.strategy_id for value in self.failures)
        _require(
            strategy_ids == tuple(sorted(strategy_ids))
            and len(strategy_ids) == len(set(strategy_ids)),
            "diagnostic accounting failures must be unique and strategy-sorted",
        )


def _accounting_failure(
    *,
    strategy_id: str,
    phase: str,
    error: Exception,
) -> StrategyAccountingFailure:
    exception_class = type(error).__name__
    code_token = "".join(
        character.casefold()
        if character.isascii() and (character.isalnum() or character == "_")
        else "_"
        for character in exception_class
    ).strip("_")
    if not code_token:
        code_token = "exception"
    private_detail = _canonical_json_bytes(
        {
            "exception_class": exception_class,
            "exception_message": str(error),
            "phase": phase,
            "strategy_id": strategy_id,
        }
    )
    return StrategyAccountingFailure(
        strategy_id=strategy_id,
        phase=phase,
        failure_code=f"{phase}.{code_token}",
        exception_class=exception_class,
        private_detail=private_detail,
    )


def _validate_locked_monthly_accounting_inputs(
    *,
    allocations: Mapping[str, StrategyAllocation],
    inputs: Mapping[str, StrategyAccountingInput],
    session_authority: HoldingPeriodSessionAuthority,
    daily_inputs_by_strategy: Mapping[str, Sequence[DatedMarketCloseInput]],
    corporate_actions: Iterable[CorporateAction] = (),
) -> tuple[tuple[str, ...], tuple[CorporateAction, ...]]:
    """Validate shared monthly inputs without running a strategy."""

    _require(
        type(session_authority) is HoldingPeriodSessionAuthority,
        "monthly accounting requires authenticated holding-period sessions",
    )
    expected_ids = {
        PRIMARY_STRATEGY_ID,
        ELASTIC_NET_CHALLENGER_ID,
        RANK_CHALLENGER_ID,
        DECISION_FOCUSED_CHALLENGER_ID,
    }
    _require(
        set(allocations) == expected_ids, "monthly allocation strategy set differs"
    )
    active_ids = {name for name, value in allocations.items() if value.active}
    _require(
        set(inputs) == active_ids,
        "accounting inputs must match exactly the active strategies",
    )
    _require(
        set(daily_inputs_by_strategy) == active_ids,
        "daily accounting inputs must match exactly the active strategies",
    )
    actions = tuple(corporate_actions)
    _require(
        len({id(value.opening_state) for value in inputs.values()}) == len(inputs)
        and len({id(value.action_ledger) for value in inputs.values()}) == len(inputs),
        "active strategies must own distinct portfolio, action-ledger, and "
        "mark-continuity states",
    )
    _require(
        len({id(value.mark_continuity_ledger) for value in inputs.values()})
        == len(inputs),
        "active strategies must own distinct portfolio, action-ledger, and "
        "mark-continuity states",
    )
    ordered_ids = tuple(sorted(active_ids))
    for left_index, left_id in enumerate(ordered_ids):
        left_days = tuple(daily_inputs_by_strategy[left_id])
        for right_id in ordered_ids[left_index + 1 :]:
            right_days = tuple(daily_inputs_by_strategy[right_id])
            _require(
                len(left_days) == len(right_days),
                "active strategy daily paths have different lengths",
            )
            for left_day, right_day in zip(left_days, right_days, strict=True):
                _require(
                    left_day.trading_date == right_day.trading_date
                    and left_day.daily_risk_free_rate == right_day.daily_risk_free_rate,
                    "active strategies received different dates or risk-free returns",
                )
                left_returns = {
                    int(permno): float(value)
                    for permno, value in zip(
                        left_day.permno, left_day.total_return, strict=True
                    )
                }
                right_returns = {
                    int(permno): float(value)
                    for permno, value in zip(
                        right_day.permno, right_day.total_return, strict=True
                    )
                }
                left_classification = {
                    int(permno): "observed" for permno in left_day.permno
                }
                left_classification.update(
                    {
                        int(permno): "no_mark"
                        for permno in left_day.no_mark_permno
                    }
                )
                right_classification = {
                    int(permno): "observed" for permno in right_day.permno
                }
                right_classification.update(
                    {
                        int(permno): "no_mark"
                        for permno in right_day.no_mark_permno
                    }
                )
                shared = set(left_classification).intersection(
                    right_classification
                )
                _require(
                    all(
                        left_classification[value]
                        == right_classification[value]
                        for value in shared
                    ),
                    "active strategies received different observed/no-mark "
                    "classifications",
                )
                shared_observed = shared.intersection(left_returns)
                _require(
                    all(
                        left_returns[value] == right_returns[value]
                        for value in shared_observed
                    ),
                    "active strategies received different realized security returns",
                )
                left_span_starts = (
                    {
                        int(permno): span_start
                        for permno, span_start in zip(
                            left_day.permno,
                            left_day.observed_return_span_start_date,
                            strict=True,
                        )
                    }
                    if left_day.observed_return_span_start_date
                    else {int(permno): None for permno in left_day.permno}
                )
                right_span_starts = (
                    {
                        int(permno): span_start
                        for permno, span_start in zip(
                            right_day.permno,
                            right_day.observed_return_span_start_date,
                            strict=True,
                        )
                    }
                    if right_day.observed_return_span_start_date
                    else {int(permno): None for permno in right_day.permno}
                )
                _require(
                    all(
                        left_span_starts.get(value)
                        == right_span_starts.get(value)
                        for value in shared_observed
                    ),
                    "active strategies received different observed return-span "
                    "starts",
                )
    return ordered_ids, actions


def _run_locked_strategy_accounting(
    *,
    strategy_id: str,
    allocations: Mapping[str, StrategyAllocation],
    inputs: Mapping[str, StrategyAccountingInput],
    session_authority: HoldingPeriodSessionAuthority,
    daily_inputs_by_strategy: Mapping[str, Sequence[DatedMarketCloseInput]],
    corporate_actions: tuple[CorporateAction, ...],
) -> DatedHoldingPeriodAccounting:
    """Run the unchanged locked accounting transition for one active strategy."""

    identity = session_authority.calendar_row.signal_date
    allocation = allocations[strategy_id]
    accounting = inputs[strategy_id]
    days = tuple(daily_inputs_by_strategy[strategy_id])
    _require(
        allocation.result is not None, "active strategy lacks an allocation result"
    )
    _require(
        allocation.problem is accounting.problem,
        "accounting problem differs from the exact solved strategy problem",
    )
    _require(
        accounting.strategy_id == strategy_id,
        "accounting strategy identity differs",
    )
    _require(
        accounting.problem.identity.signal_date == identity,
        "accounting problem signal differs from session authority",
    )
    problem = accounting.problem.problem
    _require(
        allocation.result.weights.shape == (problem.n_assets,),
        "allocation weights do not align to accounting problem",
    )
    return run_dated_holding_period(
        accounting.opening_state,
        session_authority=session_authority,
        target_permno=problem.permno,
        target_weights=allocation.result.weights,
        execution_price=accounting.execution_price,
        full_spread=problem.full_spread,
        adv_dollars=problem.adv20_usd,
        impact_volatility_daily=problem.impact_volatility_daily,
        daily_inputs=days,
        corporate_actions=corporate_actions,
        initial_action_ledger=accounting.action_ledger,
        initial_mark_continuity_ledger=accounting.mark_continuity_ledger,
    )


def run_locked_monthly_accounting(
    *,
    allocations: Mapping[str, StrategyAllocation],
    inputs: Mapping[str, StrategyAccountingInput],
    session_authority: HoldingPeriodSessionAuthority,
    daily_inputs_by_strategy: Mapping[str, Sequence[DatedMarketCloseInput]],
    corporate_actions: Iterable[CorporateAction] = (),
) -> Mapping[str, DatedHoldingPeriodAccounting]:
    """Run fixed-share daily accounting for every active locked strategy.

    The common session authority ensures identical dates and corporate-action
    inputs. Realized returns and RF must agree wherever strategy domains
    overlap; borrow vectors remain strategy-specific because side assignments
    and monthly liquidity quintiles can differ. No liquidation is added at
    the terminal month.
    """

    ordered_ids, actions = _validate_locked_monthly_accounting_inputs(
        allocations=allocations,
        inputs=inputs,
        session_authority=session_authority,
        daily_inputs_by_strategy=daily_inputs_by_strategy,
        corporate_actions=corporate_actions,
    )
    result: dict[str, DatedHoldingPeriodAccounting] = {}
    for strategy_id in ordered_ids:
        result[strategy_id] = _run_locked_strategy_accounting(
            strategy_id=strategy_id,
            allocations=allocations,
            inputs=inputs,
            session_authority=session_authority,
            daily_inputs_by_strategy=daily_inputs_by_strategy,
            corporate_actions=actions,
        )
    return MappingProxyType(result)


def diagnose_locked_monthly_accounting(
    *,
    allocations: Mapping[str, StrategyAllocation],
    inputs: Mapping[str, StrategyAccountingInput],
    session_authority: HoldingPeriodSessionAuthority,
    daily_inputs_by_strategy: Mapping[str, Sequence[DatedMarketCloseInput]],
    corporate_actions: Iterable[CorporateAction] = (),
) -> DiagnosticMonthlyAccounting:
    """Attempt every active accounting transition and aggregate all failures.

    This diagnostic-only wrapper never returns a partial result. Shared-input
    failures are recorded against every active strategy because no independent
    accounting transition is then admissible. Once shared preflight passes,
    every active strategy is attempted even if an earlier strategy fails.
    ``BaseException`` subclasses remain interrupt signals and are not caught.
    """

    expected_ids = {
        PRIMARY_STRATEGY_ID,
        ELASTIC_NET_CHALLENGER_ID,
        RANK_CHALLENGER_ID,
        DECISION_FOCUSED_CHALLENGER_ID,
    }
    _require(
        set(allocations) == expected_ids,
        "monthly allocation strategy set differs",
    )
    active_ids = tuple(
        sorted(name for name, value in allocations.items() if value.active)
    )
    try:
        ordered_ids, actions = _validate_locked_monthly_accounting_inputs(
            allocations=allocations,
            inputs=inputs,
            session_authority=session_authority,
            daily_inputs_by_strategy=daily_inputs_by_strategy,
            corporate_actions=corporate_actions,
        )
    except Exception as error:
        return DiagnosticMonthlyAccounting(
            accounting=None,
            failures=tuple(
                _accounting_failure(
                    strategy_id=strategy_id,
                    phase="accounting_preflight",
                    error=error,
                )
                for strategy_id in active_ids
            ),
        )

    result: dict[str, DatedHoldingPeriodAccounting] = {}
    failures: list[StrategyAccountingFailure] = []
    for strategy_id in ordered_ids:
        try:
            result[strategy_id] = _run_locked_strategy_accounting(
                strategy_id=strategy_id,
                allocations=allocations,
                inputs=inputs,
                session_authority=session_authority,
                daily_inputs_by_strategy=daily_inputs_by_strategy,
                corporate_actions=actions,
            )
        except Exception as error:
            failures.append(
                _accounting_failure(
                    strategy_id=strategy_id,
                    phase="holding_period_accounting",
                    error=error,
                )
            )
    if failures:
        return DiagnosticMonthlyAccounting(
            accounting=None,
            failures=tuple(failures),
        )
    return DiagnosticMonthlyAccounting(
        accounting=MappingProxyType(result),
        failures=(),
    )


__all__ = [
    "AuthorityBoundStage11WalkForwardBackend",
    "DECISION_FOCUSED_CHALLENGER_ID",
    "ELASTIC_NET_CHALLENGER_ID",
    "EXPECTED_FITS_PER_REFIT",
    "EXPECTED_REFIT_COUNT",
    "EXPECTED_TOTAL_FITS",
    "FAMILY_ORDER",
    "FORECAST_COLUMNS",
    "LockedValidationSelection",
    "MonthlyExecutionStep",
    "MonthlyForecastCommit",
    "MonthlyOutcomeCommit",
    "ORCHESTRATOR_ID",
    "OUTCOME_COLUMNS",
    "PRIMARY_STRATEGY_ID",
    "RANK_CHALLENGER_ID",
    "SelectorFitJob",
    "Stage11ExecutionOrchestratorError",
    "Stage11ExecutionPlan",
    "Stage11PseudoLiveLedger",
    "Stage11WalkForwardBackend",
    "DiagnosticMonthlyAccounting",
    "DiagnosticMonthlyAllocation",
    "StrategyAllocation",
    "StrategyAllocationFailure",
    "StrategyAccountingFailure",
    "StrategyAccountingInput",
    "StrategyCalibrationSchedule",
    "WalkForwardRunReceipt",
    "allocate_locked_monthly_strategies",
    "build_stage11_execution_plan",
    "build_strategy_calibration_schedules",
    "canonical_frame_sha256",
    "derive_empirical_locked_validation_selection",
    "derive_locked_validation_selection",
    "diagnose_locked_monthly_accounting",
    "diagnose_locked_monthly_strategies",
    "run_empirical_locked_walk_forward",
    "run_locked_walk_forward",
    "run_locked_monthly_accounting",
]
