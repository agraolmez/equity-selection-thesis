"""Pure reporting-only sensitivity for Stage 13 terminal cash retirements.

The primary accounting applies each authenticated ordinary final-session
return and then retires the resulting signed position value to cash at a
multiplier of one.  This module does not alter that accounting.  It computes
four one-at-a-time reporting counterfactuals from retained event evidence:

* cash-recovery multipliers of 0.70, 0.45, and 0.00; and
* one additional full-spread terminal execution charge.

The event value is the signed, post-return and pre-settlement position value.
All adjustments include the risk-free growth that the resulting cash change
would have earned over sessions strictly after the event session through the
end of the holding month.  No model fit, score, selection, allocation, or
primary estimate is changed.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np


TERMINAL_EVENT_SENSITIVITY_ID = (
    "stage13-post-unblind-terminal-liquidation-sensitivity-v1"
)
RECOVERY_MULTIPLIERS = (0.70, 0.45, 0.00)
RECOVERY_SCENARIO_IDS = (
    "terminal_recovery_multiplier_0_70",
    "terminal_recovery_multiplier_0_45",
    "terminal_recovery_multiplier_0_00",
)
FULL_SPREAD_SCENARIO_ID = "terminal_execution_full_spread_stress"
ANNUALIZATION_PERIODS = 12

_MONTH = re.compile(r"^[0-9]{4}-(?:0[1-9]|1[0-2])$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TerminalEventSensitivityError(ValueError):
    """Raised when sensitivity inputs or a counterfactual escape the contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalEventSensitivityError(message)


def _identifier(value: object, *, field: str) -> str:
    _require(
        isinstance(value, str) and bool(value) and value == value.strip(),
        f"{field} must be a nonempty stripped string",
    )
    return value


def _holding_month(value: object) -> str:
    _require(
        isinstance(value, str) and _MONTH.fullmatch(value) is not None,
        "holding_month must use exact YYYY-MM format",
    )
    return value


def _sha256(value: object, *, field: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{field} must be an exact lowercase SHA-256",
    )
    return value


def _finite(value: object, *, field: str) -> float:
    _require(not isinstance(value, bool), f"{field} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TerminalEventSensitivityError(f"{field} must be finite numeric") from exc
    _require(math.isfinite(result), f"{field} must be finite numeric")
    return result


@dataclass(frozen=True, slots=True)
class TerminalEventSource:
    """Strategy-independent evidence for one authenticated terminal event.

    ``remaining_cash_growth`` compounds risk-free returns only over sessions
    strictly after ``event_date`` through the holding-period exit.
    """

    terminal_action_id: str
    permno: int
    event_date: dt.date
    holding_month: str
    source_record_sha256: str
    terminal_policy_sha256: str
    terminal_evidence_sha256: str
    event_day_quoted_full_spread: float
    frozen_full_spread_20d: float
    remaining_cash_growth: float

    def __post_init__(self) -> None:
        _identifier(self.terminal_action_id, field="terminal_action_id")
        _require(
            type(self.permno) is int and self.permno > 0,
            "permno must be a positive integer",
        )
        _require(type(self.event_date) is dt.date, "event_date must be a date")
        month = _holding_month(self.holding_month)
        _require(
            self.event_date.strftime("%Y-%m") == month,
            "event_date must fall in holding_month",
        )
        _sha256(self.source_record_sha256, field="source_record_sha256")
        _sha256(self.terminal_policy_sha256, field="terminal_policy_sha256")
        _sha256(self.terminal_evidence_sha256, field="terminal_evidence_sha256")
        quoted = _finite(
            self.event_day_quoted_full_spread,
            field="event_day_quoted_full_spread",
        )
        trailing = _finite(
            self.frozen_full_spread_20d,
            field="frozen_full_spread_20d",
        )
        growth = _finite(self.remaining_cash_growth, field="remaining_cash_growth")
        _require(quoted >= 0.0, "event_day_quoted_full_spread cannot be negative")
        _require(trailing >= 0.0, "frozen_full_spread_20d cannot be negative")
        _require(growth > 0.0, "remaining_cash_growth must be positive")

    @property
    def selected_full_spread_stress_rate(self) -> float:
        """Return the stricter of event-day quoted and frozen 20-day spread."""

        return max(
            float(self.event_day_quoted_full_spread),
            float(self.frozen_full_spread_20d),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "terminal_action_id": self.terminal_action_id,
            "permno": self.permno,
            "event_date": self.event_date.isoformat(),
            "holding_month": self.holding_month,
            "source_record_sha256": self.source_record_sha256,
            "terminal_policy_sha256": self.terminal_policy_sha256,
            "terminal_evidence_sha256": self.terminal_evidence_sha256,
            "event_day_quoted_full_spread": float(self.event_day_quoted_full_spread),
            "frozen_full_spread_20d": float(self.frozen_full_spread_20d),
            "selected_full_spread_stress_rate": (self.selected_full_spread_stress_rate),
            "remaining_cash_growth": float(self.remaining_cash_growth),
        }

    def payload(self) -> dict[str, object]:
        """Return exact float-hex evidence for enclosing content digests."""

        return {
            "terminal_action_id": self.terminal_action_id,
            "permno": self.permno,
            "event_date": self.event_date.isoformat(),
            "holding_month": self.holding_month,
            "source_record_sha256": self.source_record_sha256,
            "terminal_policy_sha256": self.terminal_policy_sha256,
            "terminal_evidence_sha256": self.terminal_evidence_sha256,
            "event_day_quoted_full_spread": float(
                self.event_day_quoted_full_spread
            ).hex(),
            "frozen_full_spread_20d": float(self.frozen_full_spread_20d).hex(),
            "selected_full_spread_stress_rate": (
                self.selected_full_spread_stress_rate.hex()
            ),
            "remaining_cash_growth": float(self.remaining_cash_growth).hex(),
        }


@dataclass(frozen=True, slots=True)
class TerminalEventStrategyObservation:
    """One explicit event-by-strategy exposure, including unheld zero rows."""

    terminal_action_id: str
    strategy_id: str
    position_held: bool
    target_weight: float
    month_start_nav_dollars: float
    event_nav_dollars: float
    signed_event_value_dollars: float

    def __post_init__(self) -> None:
        _identifier(self.terminal_action_id, field="terminal_action_id")
        _identifier(self.strategy_id, field="strategy_id")
        _require(type(self.position_held) is bool, "position_held must be boolean")
        _finite(self.target_weight, field="target_weight")
        start_nav = _finite(
            self.month_start_nav_dollars,
            field="month_start_nav_dollars",
        )
        event_nav = _finite(self.event_nav_dollars, field="event_nav_dollars")
        value = _finite(
            self.signed_event_value_dollars,
            field="signed_event_value_dollars",
        )
        _require(start_nav > 0.0, "month_start_nav_dollars must be positive")
        _require(event_nav > 0.0, "event_nav_dollars must be positive")
        if not self.position_held:
            _require(
                value == 0.0,
                "an unheld terminal position must be an explicit zero row",
            )

    @property
    def event_day_portfolio_weight(self) -> float:
        return float(self.signed_event_value_dollars) / float(self.event_nav_dollars)

    def as_dict(self) -> dict[str, object]:
        return {
            "terminal_action_id": self.terminal_action_id,
            "strategy_id": self.strategy_id,
            "position_held": self.position_held,
            "target_weight": float(self.target_weight),
            "month_start_nav_dollars": float(self.month_start_nav_dollars),
            "event_nav_dollars": float(self.event_nav_dollars),
            "signed_event_value_dollars": float(self.signed_event_value_dollars),
            "event_day_portfolio_weight": self.event_day_portfolio_weight,
        }

    def payload(self) -> dict[str, object]:
        """Return exact float-hex evidence for enclosing content digests."""

        return {
            "terminal_action_id": self.terminal_action_id,
            "strategy_id": self.strategy_id,
            "position_held": self.position_held,
            "target_weight": float(self.target_weight).hex(),
            "month_start_nav_dollars": float(self.month_start_nav_dollars).hex(),
            "event_nav_dollars": float(self.event_nav_dollars).hex(),
            "signed_event_value_dollars": float(self.signed_event_value_dollars).hex(),
            "event_day_portfolio_weight": self.event_day_portfolio_weight.hex(),
        }


@dataclass(frozen=True, slots=True)
class StrategyMonthlyReturn:
    """One primary monthly return row used only for reporting recomputation."""

    strategy_id: str
    holding_month: str
    net_return: float
    excess_return: float

    def __post_init__(self) -> None:
        _identifier(self.strategy_id, field="strategy_id")
        _holding_month(self.holding_month)
        net = _finite(self.net_return, field="net_return")
        _finite(self.excess_return, field="excess_return")
        _require(net >= -1.0, "net_return cannot be below -100%")


@dataclass(frozen=True, slots=True)
class TerminalSensitivityMetrics:
    """Frozen headline metrics recomputed on one complete monthly series."""

    net_cagr: float
    annualized_excess_sharpe: float
    maximum_drawdown: float

    def as_dict(self) -> dict[str, float]:
        return {
            "net_cagr": self.net_cagr,
            "annualized_excess_sharpe": self.annualized_excess_sharpe,
            "maximum_drawdown": self.maximum_drawdown,
        }


@dataclass(frozen=True, slots=True)
class TerminalSensitivityMonthEffect:
    """Aggregated adjustment for one strategy and affected holding month."""

    holding_month: str
    primary_net_return: float
    primary_excess_return: float
    return_adjustment: float
    counterfactual_net_return: float
    counterfactual_excess_return: float

    def as_dict(self) -> dict[str, object]:
        return {
            "holding_month": self.holding_month,
            "primary_net_return": self.primary_net_return,
            "primary_excess_return": self.primary_excess_return,
            "return_adjustment": self.return_adjustment,
            "counterfactual_net_return": self.counterfactual_net_return,
            "counterfactual_excess_return": self.counterfactual_excess_return,
        }


@dataclass(frozen=True, slots=True)
class TerminalSensitivityStrategyResult:
    """One strategy's metrics and affected-month returns in one scenario."""

    strategy_id: str
    affected_months: tuple[TerminalSensitivityMonthEffect, ...]
    metrics: TerminalSensitivityMetrics
    rank: int

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "affected_months": [value.as_dict() for value in self.affected_months],
            "metrics": self.metrics.as_dict(),
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class TerminalEventSensitivityScenario:
    """One reporting-only, one-at-a-time terminal-event counterfactual."""

    scenario_id: str
    scenario_kind: str
    recovery_multiplier: float | None
    strategy_results: tuple[TerminalSensitivityStrategyResult, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_kind": self.scenario_kind,
            "recovery_multiplier": self.recovery_multiplier,
            "one_at_a_time": True,
            "primary_estimates_modified": False,
            "model_refit_or_portfolio_rerun": False,
            "promoted_to_primary": False,
            "strategy_results": [value.as_dict() for value in self.strategy_results],
        }


@dataclass(frozen=True, slots=True)
class TerminalEventSensitivityEvaluation:
    """Baseline ranking plus all four ordered reporting counterfactuals."""

    baseline_strategy_results: tuple[TerminalSensitivityStrategyResult, ...]
    scenarios: tuple[TerminalEventSensitivityScenario, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "sensitivity_id": TERMINAL_EVENT_SENSITIVITY_ID,
            "one_at_a_time_only": True,
            "cartesian_product_executed": False,
            "primary_estimates_modified": False,
            "baseline_strategy_results": [
                value.as_dict() for value in self.baseline_strategy_results
            ],
            "scenarios": [value.as_dict() for value in self.scenarios],
        }


def recovery_return_adjustment(
    observation: TerminalEventStrategyObservation,
    source: TerminalEventSource,
    recovery_multiplier: object,
) -> float:
    """Return the exact holding-month adjustment for one recovery scenario."""

    _require(
        isinstance(observation, TerminalEventStrategyObservation),
        "observation must be a TerminalEventStrategyObservation",
    )
    _require(
        isinstance(source, TerminalEventSource),
        "source must be a TerminalEventSource",
    )
    _require(
        observation.terminal_action_id == source.terminal_action_id,
        "observation and source terminal action identities differ",
    )
    multiplier = _finite(recovery_multiplier, field="recovery_multiplier")
    _require(0.0 <= multiplier <= 1.0, "recovery_multiplier must be in [0, 1]")
    return float(
        (multiplier - 1.0)
        * float(observation.signed_event_value_dollars)
        * float(source.remaining_cash_growth)
        / float(observation.month_start_nav_dollars)
    )


def full_spread_return_adjustment(
    observation: TerminalEventStrategyObservation,
    source: TerminalEventSource,
) -> float:
    """Return one additional full-spread charge as a holding-month return."""

    _require(
        isinstance(observation, TerminalEventStrategyObservation),
        "observation must be a TerminalEventStrategyObservation",
    )
    _require(
        isinstance(source, TerminalEventSource),
        "source must be a TerminalEventSource",
    )
    _require(
        observation.terminal_action_id == source.terminal_action_id,
        "observation and source terminal action identities differ",
    )
    return float(
        -abs(float(observation.signed_event_value_dollars))
        * source.selected_full_spread_stress_rate
        * float(source.remaining_cash_growth)
        / float(observation.month_start_nav_dollars)
    )


def compute_terminal_sensitivity_metrics(
    monthly_net_returns: Sequence[object],
    monthly_excess_returns: Sequence[object],
) -> TerminalSensitivityMetrics:
    """Recompute Stage 13 CAGR, sample-vol Sharpe, and NAV drawdown."""

    _require(
        not isinstance(monthly_net_returns, (str, bytes))
        and not isinstance(monthly_excess_returns, (str, bytes)),
        "monthly returns must be numeric sequences",
    )
    try:
        net = np.asarray(monthly_net_returns, dtype=np.float64)
        excess = np.asarray(monthly_excess_returns, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TerminalEventSensitivityError(
            "monthly returns must be numeric sequences"
        ) from exc
    _require(
        net.ndim == 1 and excess.ndim == 1 and net.size == excess.size,
        "monthly net and excess returns must be aligned vectors",
    )
    _require(net.size >= 2, "terminal sensitivity metrics need at least two months")
    _require(
        bool(np.isfinite(net).all()) and bool(np.isfinite(excess).all()),
        "monthly returns must be finite",
    )
    _require(bool(np.all(net >= -1.0)), "monthly net return cannot be below -100%")

    monthly_volatility = float(np.std(excess, ddof=1))
    _require(
        math.isfinite(monthly_volatility) and monthly_volatility > 0.0,
        "monthly excess returns have zero variance",
    )
    sharpe = float(
        math.sqrt(ANNUALIZATION_PERIODS) * float(np.mean(excess)) / monthly_volatility
    )
    cumulative = float(np.prod(1.0 + net))
    _require(
        math.isfinite(cumulative) and cumulative >= 0.0,
        "compounded terminal sensitivity return is invalid",
    )
    cagr = float(cumulative ** (ANNUALIZATION_PERIODS / net.size) - 1.0)

    wealth = np.concatenate((np.ones(1), np.cumprod(1.0 + net)))
    peaks = np.maximum.accumulate(wealth)
    drawdowns = 1.0 - np.divide(
        wealth,
        peaks,
        out=np.ones_like(wealth),
        where=peaks > 0.0,
    )
    maximum_drawdown = float(np.max(drawdowns))
    _require(
        math.isfinite(maximum_drawdown) and 0.0 <= maximum_drawdown <= 1.0,
        "maximum drawdown is outside [0, 1]",
    )
    return TerminalSensitivityMetrics(
        net_cagr=0.0 if cagr == 0.0 else cagr,
        annualized_excess_sharpe=0.0 if sharpe == 0.0 else sharpe,
        maximum_drawdown=(0.0 if maximum_drawdown == 0.0 else maximum_drawdown),
    )


def _ordered_monthly_rows(
    monthly_returns: Sequence[StrategyMonthlyReturn],
) -> tuple[tuple[str, ...], Mapping[str, tuple[StrategyMonthlyReturn, ...]]]:
    _require(bool(monthly_returns), "monthly return rows are empty")
    rows_by_strategy: dict[str, dict[str, StrategyMonthlyReturn]] = {}
    for row in monthly_returns:
        _require(
            isinstance(row, StrategyMonthlyReturn),
            "monthly return row has the wrong type",
        )
        strategy_rows = rows_by_strategy.setdefault(row.strategy_id, {})
        _require(
            row.holding_month not in strategy_rows,
            "monthly return rows contain a duplicate strategy-month",
        )
        strategy_rows[row.holding_month] = row
    strategy_ids = tuple(sorted(rows_by_strategy))
    reference_months = tuple(sorted(rows_by_strategy[strategy_ids[0]]))
    _require(len(reference_months) >= 2, "monthly return panel is too short")
    ordered: dict[str, tuple[StrategyMonthlyReturn, ...]] = {}
    for strategy_id in strategy_ids:
        months = tuple(sorted(rows_by_strategy[strategy_id]))
        _require(
            months == reference_months,
            "strategies do not share one complete monthly return panel",
        )
        ordered[strategy_id] = tuple(
            rows_by_strategy[strategy_id][month] for month in reference_months
        )
    return strategy_ids, ordered


def _validated_event_grid(
    *,
    strategy_ids: Sequence[str],
    holding_months: frozenset[str],
    event_sources: Sequence[TerminalEventSource],
    observations: Sequence[TerminalEventStrategyObservation],
) -> tuple[
    tuple[TerminalEventSource, ...],
    Mapping[tuple[str, str], TerminalEventStrategyObservation],
]:
    _require(bool(event_sources), "terminal event sources are empty")
    source_by_id: dict[str, TerminalEventSource] = {}
    source_identity: set[tuple[int, dt.date]] = set()
    for source in event_sources:
        _require(
            isinstance(source, TerminalEventSource),
            "terminal event source has the wrong type",
        )
        _require(
            source.terminal_action_id not in source_by_id,
            "terminal event sources contain a duplicate action id",
        )
        identity = (source.permno, source.event_date)
        _require(
            identity not in source_identity,
            "terminal event sources contain a duplicate security-date",
        )
        _require(
            source.holding_month in holding_months,
            "terminal event falls outside the monthly return panel",
        )
        source_by_id[source.terminal_action_id] = source
        source_identity.add(identity)

    observation_by_key: dict[tuple[str, str], TerminalEventStrategyObservation] = {}
    for observation in observations:
        _require(
            isinstance(observation, TerminalEventStrategyObservation),
            "terminal event observation has the wrong type",
        )
        key = (observation.terminal_action_id, observation.strategy_id)
        _require(
            key not in observation_by_key,
            "terminal event observations contain a duplicate action-strategy",
        )
        _require(
            observation.terminal_action_id in source_by_id,
            "terminal event observation names an unknown action",
        )
        _require(
            observation.strategy_id in strategy_ids,
            "terminal event observation names an unknown strategy",
        )
        observation_by_key[key] = observation

    expected_keys = {
        (action_id, strategy_id)
        for action_id in source_by_id
        for strategy_id in strategy_ids
    }
    _require(
        set(observation_by_key) == expected_keys,
        "terminal event observations must form a complete action-strategy grid",
    )

    for strategy_id in strategy_ids:
        for holding_month in {source.holding_month for source in event_sources}:
            start_navs = {
                observation_by_key[
                    (source.terminal_action_id, strategy_id)
                ].month_start_nav_dollars
                for source in event_sources
                if source.holding_month == holding_month
            }
            _require(
                len(start_navs) == 1,
                "same-month terminal observations disagree on month-start NAV",
            )

    ordered_sources = tuple(
        sorted(
            source_by_id.values(),
            key=lambda value: (value.event_date, value.terminal_action_id),
        )
    )
    return ordered_sources, observation_by_key


def _ranked_results(
    values: Sequence[TerminalSensitivityStrategyResult],
) -> tuple[TerminalSensitivityStrategyResult, ...]:
    ordered = sorted(
        values,
        key=lambda value: (
            -value.metrics.annualized_excess_sharpe,
            -value.metrics.net_cagr,
            value.strategy_id,
        ),
    )
    return tuple(
        replace(value, rank=ordinal) for ordinal, value in enumerate(ordered, start=1)
    )


def _scenario_results(
    *,
    strategy_ids: Sequence[str],
    monthly_by_strategy: Mapping[str, tuple[StrategyMonthlyReturn, ...]],
    event_sources: Sequence[TerminalEventSource],
    observation_by_key: Mapping[tuple[str, str], TerminalEventStrategyObservation],
    recovery_multiplier: float | None,
) -> tuple[TerminalSensitivityStrategyResult, ...]:
    affected_holding_months = tuple(
        sorted({source.holding_month for source in event_sources})
    )
    result_rows: list[TerminalSensitivityStrategyResult] = []
    for strategy_id in strategy_ids:
        adjustment_by_month = {month: 0.0 for month in affected_holding_months}
        for source in event_sources:
            observation = observation_by_key[(source.terminal_action_id, strategy_id)]
            if recovery_multiplier is None:
                adjustment = full_spread_return_adjustment(observation, source)
            else:
                adjustment = recovery_return_adjustment(
                    observation,
                    source,
                    recovery_multiplier,
                )
            adjustment_by_month[source.holding_month] += adjustment

        adjusted_net: list[float] = []
        adjusted_excess: list[float] = []
        affected_rows: list[TerminalSensitivityMonthEffect] = []
        for row in monthly_by_strategy[strategy_id]:
            adjustment = adjustment_by_month.get(row.holding_month, 0.0)
            net_return = float(row.net_return) + adjustment
            excess_return = float(row.excess_return) + adjustment
            _require(
                math.isfinite(net_return) and net_return >= -1.0,
                "terminal sensitivity produces a monthly return below -100%",
            )
            _require(
                math.isfinite(excess_return),
                "terminal sensitivity produces a nonfinite excess return",
            )
            adjusted_net.append(net_return)
            adjusted_excess.append(excess_return)
            if row.holding_month in adjustment_by_month:
                affected_rows.append(
                    TerminalSensitivityMonthEffect(
                        holding_month=row.holding_month,
                        primary_net_return=float(row.net_return),
                        primary_excess_return=float(row.excess_return),
                        return_adjustment=adjustment,
                        counterfactual_net_return=net_return,
                        counterfactual_excess_return=excess_return,
                    )
                )
        result_rows.append(
            TerminalSensitivityStrategyResult(
                strategy_id=strategy_id,
                affected_months=tuple(affected_rows),
                metrics=compute_terminal_sensitivity_metrics(
                    adjusted_net,
                    adjusted_excess,
                ),
                rank=0,
            )
        )
    return _ranked_results(result_rows)


def evaluate_terminal_event_sensitivities(
    monthly_returns: Sequence[StrategyMonthlyReturn],
    event_sources: Sequence[TerminalEventSource],
    observations: Sequence[TerminalEventStrategyObservation],
) -> TerminalEventSensitivityEvaluation:
    """Evaluate the frozen OAT scenarios without changing primary accounting.

    Every source event must have exactly one observation for every strategy in
    the monthly panel.  An unheld event is represented by ``position_held=False``
    and an exact zero signed value; omitting that row is an error.
    """

    strategy_ids, monthly_by_strategy = _ordered_monthly_rows(monthly_returns)
    holding_months = frozenset(
        row.holding_month for row in monthly_by_strategy[strategy_ids[0]]
    )
    ordered_sources, observation_by_key = _validated_event_grid(
        strategy_ids=strategy_ids,
        holding_months=holding_months,
        event_sources=event_sources,
        observations=observations,
    )

    baseline_rows = [
        TerminalSensitivityStrategyResult(
            strategy_id=strategy_id,
            affected_months=(),
            metrics=compute_terminal_sensitivity_metrics(
                [row.net_return for row in monthly_by_strategy[strategy_id]],
                [row.excess_return for row in monthly_by_strategy[strategy_id]],
            ),
            rank=0,
        )
        for strategy_id in strategy_ids
    ]
    baseline = _ranked_results(baseline_rows)

    scenarios = tuple(
        TerminalEventSensitivityScenario(
            scenario_id=scenario_id,
            scenario_kind="post_delisting_cash_recovery_multiplier",
            recovery_multiplier=multiplier,
            strategy_results=_scenario_results(
                strategy_ids=strategy_ids,
                monthly_by_strategy=monthly_by_strategy,
                event_sources=ordered_sources,
                observation_by_key=observation_by_key,
                recovery_multiplier=multiplier,
            ),
        )
        for scenario_id, multiplier in zip(
            RECOVERY_SCENARIO_IDS,
            RECOVERY_MULTIPLIERS,
            strict=True,
        )
    ) + (
        TerminalEventSensitivityScenario(
            scenario_id=FULL_SPREAD_SCENARIO_ID,
            scenario_kind="additional_terminal_full_spread_execution_cost",
            recovery_multiplier=None,
            strategy_results=_scenario_results(
                strategy_ids=strategy_ids,
                monthly_by_strategy=monthly_by_strategy,
                event_sources=ordered_sources,
                observation_by_key=observation_by_key,
                recovery_multiplier=None,
            ),
        ),
    )
    return TerminalEventSensitivityEvaluation(
        baseline_strategy_results=baseline,
        scenarios=scenarios,
    )


__all__ = [
    "ANNUALIZATION_PERIODS",
    "FULL_SPREAD_SCENARIO_ID",
    "RECOVERY_MULTIPLIERS",
    "RECOVERY_SCENARIO_IDS",
    "TERMINAL_EVENT_SENSITIVITY_ID",
    "StrategyMonthlyReturn",
    "TerminalEventSensitivityError",
    "TerminalEventSensitivityEvaluation",
    "TerminalEventSensitivityScenario",
    "TerminalEventSource",
    "TerminalEventStrategyObservation",
    "TerminalSensitivityMetrics",
    "TerminalSensitivityMonthEffect",
    "TerminalSensitivityStrategyResult",
    "compute_terminal_sensitivity_metrics",
    "evaluate_terminal_event_sensitivities",
    "full_spread_return_adjustment",
    "recovery_return_adjustment",
]
