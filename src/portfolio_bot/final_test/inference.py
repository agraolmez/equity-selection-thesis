"""Frozen, data-independent inference mechanics for the final portfolio test.

The functions in this module are numerical primitives only.  They never open a
file or inspect a date.  A caller must pass a complete, chronologically ordered
matrix whose rows are calendar months.  Stationary-bootstrap indices always
resample those rows jointly, preserving pairing across strategies and factors.

The production bootstrap constants implement Methodology v1 exactly.  Smaller
replicate counts remain configurable solely so the algorithms can be exercised
quickly with synthetic unit tests before the final-period data are unblinded.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import norm
from scipy.stats import skew as scipy_skew

from portfolio_bot.final_test.calendar import (
    CORE_COMPLETE_HOLDING_MONTHS,
    PARTIAL_DECEMBER_HOLDING_MONTH,
    PRIMARY_FULL_HOLDING_MONTHS,
    FinalCalendarError,
    require_exact_core_extension_months,
    require_exact_partial_december_month,
    require_exact_primary_inference_months,
)


ANNUALIZATION_PERIODS = 12
STATIONARY_BOOTSTRAP_EXPECTED_BLOCK = 6
STATIONARY_BOOTSTRAP_MASTER_SEED = 20_260_714
STUDENTIZED_OUTER_REPLICATES = 10_000
STUDENTIZED_INNER_REPLICATES = 1_000
MAX_INVALID_OUTER_FRACTION = 0.01
NEWEY_WEST_LAGS = 6
FROZEN_PRIMARY_OBSERVATIONS = 68
FROZEN_CONFIDENCE_LEVEL = 0.95
FROZEN_RAW_TRIAL_COUNT = 386
FROZEN_TRIAL_LEDGER_ID = "stage12-complete-frozen-trial-ledger-v1"
AUTOMATIC_HAC_BANDWIDTH_RULE_ID = "floor_4_times_T_over_100_power_2_over_9"
SECONDARY_AUTOMATIC_HAC_LABEL = "secondary_automatic_bandwidth_hac_robustness"
SECONDARY_PERCENTILE_LABEL = "secondary_stationary_bootstrap_percentile_not_studentized"
SECONDARY_ALPHA_STATISTIC_ID = "ff5_umd_annualized_alpha_secondary_percentile"
SECONDARY_MAXIMUM_DRAWDOWN_STATISTIC_ID = "maximum_drawdown_secondary_percentile"
FROZEN_JOINT_OUTER_INDICES_SHA256 = (
    "5c925d9acce36c6452f23239115adfa7a873a3594537daf53d5fa31f5c8fe41d"
)

FF5_UMD_FACTOR_NAMES = ("mktrf", "smb", "hml", "rmw", "cma", "umd")

ROMANO_WOLF_FAMILY = (
    "hybrid_minus_elastic_net",
    "hybrid_minus_trailing_residual_mean_markowitz",
    "hybrid_minus_value_quality_momentum",
    "hybrid_minus_rank_allocator",
    "hybrid_minus_decision_focused_allocator",
)
FROZEN_ACTIVE_ROMANO_WOLF_FAMILY = ROMANO_WOLF_FAMILY[:4]
FROZEN_INACTIVE_ROMANO_WOLF_COMPARISON = ROMANO_WOLF_FAMILY[4]

if len(PRIMARY_FULL_HOLDING_MONTHS) != FROZEN_PRIMARY_OBSERVATIONS:
    raise RuntimeError("internal frozen primary inference calendar differs")


class FinalInferenceError(ValueError):
    """Base class for fail-closed final-inference errors."""


class BootstrapInferenceFailure(FinalInferenceError):
    """Raised when a confirmatory studentized result cannot be issued."""

    def __init__(
        self,
        message: str,
        *,
        outer_replicates: int,
        invalid_outer_replicates: int,
        attempted_inner_replicates: int,
        invalid_inner_replicates: int,
    ) -> None:
        super().__init__(message)
        self.outer_replicates = int(outer_replicates)
        self.invalid_outer_replicates = int(invalid_outer_replicates)
        self.attempted_inner_replicates = int(attempted_inner_replicates)
        self.invalid_inner_replicates = int(invalid_inner_replicates)

    @property
    def invalid_outer_fraction(self) -> float:
        if self.outer_replicates == 0:
            return 0.0
        return self.invalid_outer_replicates / self.outer_replicates


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FinalInferenceError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _positive_int(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise FinalInferenceError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise FinalInferenceError(f"{name} must be at least {minimum}")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise FinalInferenceError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise FinalInferenceError(f"{name} must be nonnegative")
    return result


def _finite_vector(
    values: Sequence[object] | np.ndarray,
    *,
    name: str,
    minimum_size: int = 1,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalInferenceError(f"{name} must be numeric") from exc
    if result.ndim != 1:
        raise FinalInferenceError(f"{name} must be one-dimensional")
    if result.size < minimum_size:
        raise FinalInferenceError(
            f"{name} must contain at least {minimum_size} observations"
        )
    if not np.isfinite(result).all():
        raise FinalInferenceError(f"{name} must be finite")
    return np.array(result, dtype=np.float64, copy=True, order="C")


def _finite_matrix(
    values: Sequence[Sequence[object]] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalInferenceError(f"{name} must be numeric") from exc
    if result.ndim != 2:
        raise FinalInferenceError(f"{name} must be two-dimensional")
    if not np.isfinite(result).all():
        raise FinalInferenceError(f"{name} must be finite")
    return np.array(result, dtype=np.float64, copy=True, order="C")


def _readonly_float_array(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_confidence(confidence_level: object) -> float:
    try:
        result = float(confidence_level)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalInferenceError("confidence_level must be numeric") from exc
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise FinalInferenceError("confidence_level must lie strictly between 0 and 1")
    return result


def _validate_invalid_fraction(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalInferenceError("max_invalid_outer_fraction must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise FinalInferenceError("max_invalid_outer_fraction must lie between 0 and 1")
    return result


def _validate_expected_block_length(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise FinalInferenceError("expected_block_length must be positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalInferenceError("expected_block_length must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise FinalInferenceError("expected_block_length must be positive")
    return result


def automatic_newey_west_lags(observations: int) -> int:
    """Return the frozen Newey--West rule-of-thumb bandwidth.

    The pre-unblind implementation fixes the commonly used monthly HAC rule

    ``floor(4 * (T / 100) ** (2 / 9))``

    capped at ``T - 1``.  It depends only on the prespecified observation count,
    never on realized returns, so it cannot adapt to a favorable final result.
    """

    count = _positive_int(observations, name="observations", minimum=1)
    bandwidth = math.floor(4.0 * (count / 100.0) ** (2.0 / 9.0))
    return min(max(0, bandwidth), count - 1)


def _annualized_ff5_umd_alpha_or_nan(matrix: np.ndarray) -> float:
    if (
        matrix.ndim != 2
        or matrix.shape[1] != 1 + len(FF5_UMD_FACTOR_NAMES)
        or matrix.shape[0] <= 1 + len(FF5_UMD_FACTOR_NAMES)
        or not np.isfinite(matrix).all()
    ):
        return math.nan
    design = np.column_stack(
        (np.ones(matrix.shape[0], dtype=np.float64), matrix[:, 1:])
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, matrix[:, 0], rcond=None)
    if rank != design.shape[1] or not np.isfinite(coefficients).all():
        return math.nan
    annualized = ANNUALIZATION_PERIODS * float(coefficients[0])
    return annualized if math.isfinite(annualized) else math.nan


def ff5_umd_annualized_alpha(
    monthly_excess_returns: Sequence[object] | np.ndarray,
    monthly_factor_returns: Sequence[Sequence[object]] | np.ndarray,
) -> float:
    """Recompute annualized FF5-plus-UMD OLS alpha on joint month rows."""

    dependent = _finite_vector(
        monthly_excess_returns,
        name="monthly_excess_returns",
        minimum_size=2 + len(FF5_UMD_FACTOR_NAMES),
    )
    factors = _finite_matrix(
        monthly_factor_returns,
        name="monthly_factor_returns",
    )
    if factors.shape != (dependent.size, len(FF5_UMD_FACTOR_NAMES)):
        raise FinalInferenceError(
            "monthly_factor_returns must have one joint row and exactly six "
            "FF5-plus-UMD columns"
        )
    result = _annualized_ff5_umd_alpha_or_nan(np.column_stack((dependent, factors)))
    if not math.isfinite(result):
        raise FinalInferenceError(
            "annualized FF5-plus-UMD alpha cannot be estimated from a "
            "rank-deficient joint panel"
        )
    return result


def maximum_drawdown(
    monthly_total_returns: Sequence[object] | np.ndarray,
) -> float:
    """Return maximum peak-to-trough NAV loss as a nonnegative fraction.

    Wealth starts at one immediately before the first holding month.  A total
    return of exactly ``-1`` is a complete loss and produces drawdown one;
    values below ``-1`` are invalid total returns.
    """

    returns = _finite_vector(
        monthly_total_returns,
        name="monthly_total_returns",
        minimum_size=1,
    )
    if np.any(returns < -1.0):
        raise FinalInferenceError("monthly total returns cannot be below -100%")
    wealth = np.concatenate((np.ones(1, dtype=np.float64), np.cumprod(1.0 + returns)))
    peaks = np.maximum.accumulate(wealth)
    drawdowns = 1.0 - np.divide(
        wealth,
        peaks,
        out=np.ones_like(wealth),
        where=peaks > 0.0,
    )
    result = float(np.max(drawdowns))
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise FinalInferenceError("maximum drawdown is outside [0, 1]")
    return 0.0 if result == 0.0 else result


def _maximum_drawdown_or_nan(matrix: np.ndarray) -> float:
    if matrix.ndim != 2 or matrix.shape[1] != 1 or not np.isfinite(matrix).all():
        return math.nan
    try:
        return maximum_drawdown(matrix[:, 0])
    except FinalInferenceError:
        return math.nan


def annualized_sharpe_ratio(
    monthly_excess_returns: Sequence[object] | np.ndarray,
    *,
    annualization_periods: int = ANNUALIZATION_PERIODS,
) -> float:
    """Return arithmetic annualized Sharpe using monthly sample volatility.

    Inputs are already excess returns.  The denominator uses ``ddof=1`` so the
    result matches the Stage 11 reporting definition.
    """

    values = _finite_vector(
        monthly_excess_returns,
        name="monthly_excess_returns",
        minimum_size=2,
    )
    periods = _positive_int(
        annualization_periods, name="annualization_periods", minimum=1
    )
    standard_deviation = float(np.std(values, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise FinalInferenceError("monthly_excess_returns have zero variance")
    result = math.sqrt(periods) * float(np.mean(values)) / standard_deviation
    if not math.isfinite(result):
        raise FinalInferenceError("annualized Sharpe ratio is nonfinite")
    return 0.0 if result == 0.0 else result


def paired_sharpe_difference(
    strategy_monthly_excess_returns: Sequence[object] | np.ndarray,
    benchmark_monthly_excess_returns: Sequence[object] | np.ndarray,
    *,
    annualization_periods: int = ANNUALIZATION_PERIODS,
) -> float:
    """Return strategy Sharpe minus benchmark Sharpe on the same monthly rows."""

    strategy = _finite_vector(
        strategy_monthly_excess_returns,
        name="strategy_monthly_excess_returns",
        minimum_size=2,
    )
    benchmark = _finite_vector(
        benchmark_monthly_excess_returns,
        name="benchmark_monthly_excess_returns",
        minimum_size=2,
    )
    if strategy.shape != benchmark.shape:
        raise FinalInferenceError("paired Sharpe inputs must have equal length")
    return annualized_sharpe_ratio(
        strategy, annualization_periods=annualization_periods
    ) - annualized_sharpe_ratio(benchmark, annualization_periods=annualization_periods)


def stationary_bootstrap_indices(
    observations: int,
    *,
    replicates: int = STUDENTIZED_OUTER_REPLICATES,
    expected_block_length: float = STATIONARY_BOOTSTRAP_EXPECTED_BLOCK,
    seed: int = STATIONARY_BOOTSTRAP_MASTER_SEED,
) -> np.ndarray:
    """Return circular Politis--Romano stationary-bootstrap row indices.

    A fresh block starts with probability ``1 / expected_block_length`` at
    every position after the first.  Otherwise the previous source index is
    advanced by one with circular wraparound.  All rows are generated by a
    local PCG64 stream; no process-global random state is touched.
    """

    count = _positive_int(observations, name="observations", minimum=1)
    repetitions = _positive_int(replicates, name="replicates", minimum=1)
    block_length = _validate_expected_block_length(expected_block_length)
    master_seed = _nonnegative_int(seed, name="seed")

    rng = np.random.Generator(np.random.PCG64(master_seed))
    indices = np.empty((repetitions, count), dtype=np.int32)
    indices[:, 0] = rng.integers(0, count, size=repetitions, dtype=np.int32)
    if count > 1:
        restart = rng.random((repetitions, count - 1)) < (1.0 / block_length)
        fresh = rng.integers(
            0,
            count,
            size=(repetitions, count - 1),
            dtype=np.int32,
        )
        for position in range(1, count):
            continuation = (indices[:, position - 1] + 1) % count
            indices[:, position] = np.where(
                restart[:, position - 1],
                fresh[:, position - 1],
                continuation,
            )
    indices.setflags(write=False)
    return indices


def _outer_indices_sha256(indices: np.ndarray) -> str:
    canonical = np.asarray(indices).astype("<i4", copy=False).tobytes(order="C")
    return hashlib.sha256(canonical).hexdigest()


def _derived_inner_seed(
    *,
    master_seed: int,
    outer_index: int,
    statistic_id: str,
    inner_index: int,
) -> int:
    if not isinstance(statistic_id, str) or not statistic_id or "\x00" in statistic_id:
        raise FinalInferenceError("statistic_id must be nonempty text without NUL")
    payload = (
        f"{master_seed}\x00{outer_index}\x00{statistic_id}\x00{inner_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _stationary_bootstrap_row(
    observations: int,
    *,
    expected_block_length: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    result = np.empty(observations, dtype=np.int32)
    result[0] = rng.integers(0, observations, dtype=np.int32)
    restart_probability = 1.0 / expected_block_length
    for position in range(1, observations):
        if rng.random() < restart_probability:
            result[position] = rng.integers(0, observations, dtype=np.int32)
        else:
            result[position] = (result[position - 1] + 1) % observations
    return result


def _annualized_sharpe_or_nan(values: np.ndarray, annualization_periods: int) -> float:
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        return math.nan
    standard_deviation = float(np.std(values, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        return math.nan
    result = (
        math.sqrt(annualization_periods) * float(np.mean(values)) / standard_deviation
    )
    return result if math.isfinite(result) else math.nan


def _inner_standard_error(
    sample: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    inner_replicates: int,
    expected_block_length: float,
    master_seed: int,
    outer_index: int,
    statistic_id: str,
) -> tuple[float, int]:
    estimates = np.full(inner_replicates, np.nan, dtype=np.float64)
    for inner_index in range(inner_replicates):
        seed = _derived_inner_seed(
            master_seed=master_seed,
            outer_index=outer_index,
            statistic_id=statistic_id,
            inner_index=inner_index,
        )
        indices = _stationary_bootstrap_row(
            sample.shape[0],
            expected_block_length=expected_block_length,
            seed=seed,
        )
        estimates[inner_index] = statistic(sample[indices])
    valid = np.isfinite(estimates)
    invalid_count = int(inner_replicates - np.count_nonzero(valid))
    if np.count_nonzero(valid) < 2:
        return math.nan, invalid_count
    standard_error = float(np.std(estimates[valid], ddof=1))
    if not math.isfinite(standard_error) or standard_error <= 0.0:
        return math.nan, invalid_count
    return standard_error, invalid_count


@dataclass(frozen=True)
class StudentizedBootstrapResult:
    """Successful studentized stationary-bootstrap confidence interval."""

    statistic_id: str
    estimate: float
    standard_error: float
    confidence_level: float
    confidence_lower: float
    confidence_upper: float
    outer_replicates: int
    valid_outer_replicates: int
    invalid_outer_replicates: int
    inner_replicates_per_standard_error: int
    attempted_inner_replicates: int
    invalid_inner_replicates: int
    expected_block_length: float
    master_seed: int
    outer_estimates: np.ndarray
    outer_standard_errors: np.ndarray
    studentized_statistics: np.ndarray

    def __post_init__(self) -> None:
        for field_name in (
            "outer_estimates",
            "outer_standard_errors",
            "studentized_statistics",
        ):
            value = _readonly_float_array(getattr(self, field_name))
            if value.ndim != 1 or value.size != self.outer_replicates:
                raise FinalInferenceError(f"{field_name} must match outer_replicates")
            object.__setattr__(self, field_name, value)

    @property
    def invalid_outer_fraction(self) -> float:
        return self.invalid_outer_replicates / self.outer_replicates


@dataclass(frozen=True, slots=True)
class SecondaryPercentileBootstrapResult:
    """Secondary stationary-bootstrap percentile interval.

    This result intentionally carries an explicit nonstudentized marker.  It
    must not be presented as a Ledoit--Wolf studentized Sharpe interval.
    """

    label: str
    statistic_id: str
    estimate: float
    confidence_level: float
    confidence_lower: float
    confidence_upper: float
    outer_replicates: int
    valid_outer_replicates: int
    invalid_outer_replicates: int
    expected_block_length: float
    master_seed: int
    joint_outer_indices_sha256: str
    bootstrap_estimates: np.ndarray
    studentized: bool = False

    def __post_init__(self) -> None:
        if self.label != SECONDARY_PERCENTILE_LABEL:
            raise FinalInferenceError("percentile interval is not secondary-labelled")
        if self.statistic_id not in {
            SECONDARY_ALPHA_STATISTIC_ID,
            SECONDARY_MAXIMUM_DRAWDOWN_STATISTIC_ID,
        }:
            raise FinalInferenceError("secondary percentile statistic identity differs")
        if self.studentized is not False:
            raise FinalInferenceError(
                "secondary percentile interval cannot be studentized"
            )
        try:
            estimate = float(self.estimate)
            lower = float(self.confidence_lower)
            upper = float(self.confidence_upper)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FinalInferenceError(
                "secondary percentile estimates must be numeric"
            ) from exc
        if not all(math.isfinite(value) for value in (estimate, lower, upper)):
            raise FinalInferenceError("secondary percentile interval must be finite")
        if lower > upper:
            raise FinalInferenceError(
                "secondary percentile interval bounds are reversed"
            )
        confidence = _validate_confidence(self.confidence_level)
        outer_count = _positive_int(
            self.outer_replicates,
            name="outer_replicates",
            minimum=2,
        )
        valid_count = _positive_int(
            self.valid_outer_replicates,
            name="valid_outer_replicates",
            minimum=2,
        )
        invalid_count = _nonnegative_int(
            self.invalid_outer_replicates,
            name="invalid_outer_replicates",
        )
        if valid_count + invalid_count != outer_count:
            raise FinalInferenceError("secondary percentile replicate counts differ")
        block_length = _validate_expected_block_length(self.expected_block_length)
        seed = _nonnegative_int(self.master_seed, name="master_seed")
        _require_sha256(
            self.joint_outer_indices_sha256,
            name="joint_outer_indices_sha256",
        )
        values = _readonly_float_array(self.bootstrap_estimates)
        if values.ndim != 1 or values.size != outer_count:
            raise FinalInferenceError(
                "bootstrap_estimates must match secondary outer_replicates"
            )
        if np.isinf(values).any():
            raise FinalInferenceError("invalid bootstrap estimates must be NaN")
        if int(np.count_nonzero(np.isfinite(values))) != valid_count:
            raise FinalInferenceError("secondary valid replicate count differs")
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "confidence_lower", lower)
        object.__setattr__(self, "confidence_upper", upper)
        object.__setattr__(self, "outer_replicates", outer_count)
        object.__setattr__(self, "valid_outer_replicates", valid_count)
        object.__setattr__(self, "invalid_outer_replicates", invalid_count)
        object.__setattr__(self, "expected_block_length", block_length)
        object.__setattr__(self, "master_seed", seed)
        object.__setattr__(self, "bootstrap_estimates", values)

    @property
    def invalid_outer_fraction(self) -> float:
        return self.invalid_outer_replicates / self.outer_replicates


def _statistic_or_nan(
    statistic: Callable[[np.ndarray], float],
    sample: np.ndarray,
) -> float:
    try:
        result = float(statistic(sample))
    except (ArithmeticError, TypeError, ValueError, np.linalg.LinAlgError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _stationary_bootstrap_percentile_interval(
    sample: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    statistic_id: str,
    confidence_level: float,
    outer_replicates: int,
    expected_block_length: float,
    master_seed: int,
    max_invalid_outer_fraction: float,
    outer_indices: np.ndarray | None = None,
) -> SecondaryPercentileBootstrapResult:
    """Return an equal-tailed plain percentile interval, never studentized."""

    matrix = _finite_matrix(sample, name="sample")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise FinalInferenceError("sample must have at least one row and one column")
    if statistic_id not in {
        SECONDARY_ALPHA_STATISTIC_ID,
        SECONDARY_MAXIMUM_DRAWDOWN_STATISTIC_ID,
    }:
        raise FinalInferenceError("secondary percentile statistic identity differs")
    confidence = _validate_confidence(confidence_level)
    outer_count = _positive_int(outer_replicates, name="outer_replicates", minimum=2)
    block_length = _validate_expected_block_length(expected_block_length)
    seed = _nonnegative_int(master_seed, name="master_seed")
    invalid_limit = _validate_invalid_fraction(max_invalid_outer_fraction)

    estimate = _statistic_or_nan(statistic, matrix)
    if not math.isfinite(estimate):
        raise BootstrapInferenceFailure(
            "the original secondary percentile statistic is invalid",
            outer_replicates=outer_count,
            invalid_outer_replicates=outer_count,
            attempted_inner_replicates=0,
            invalid_inner_replicates=0,
        )

    if outer_indices is None:
        resolved_outer_indices = stationary_bootstrap_indices(
            matrix.shape[0],
            replicates=outer_count,
            expected_block_length=block_length,
            seed=seed,
        )
    else:
        candidate_indices = np.asarray(outer_indices)
        if (
            candidate_indices.ndim != 2
            or candidate_indices.shape != (outer_count, matrix.shape[0])
            or candidate_indices.dtype.kind not in {"i", "u"}
            or np.any(candidate_indices < 0)
            or np.any(candidate_indices >= matrix.shape[0])
        ):
            raise FinalInferenceError(
                "outer_indices must be an in-range integer matrix matching "
                "replicates and observations"
            )
        resolved_outer_indices = candidate_indices

    bootstrap_estimates = np.full(outer_count, np.nan, dtype=np.float64)
    for outer_index, indices in enumerate(resolved_outer_indices):
        bootstrap_estimates[outer_index] = _statistic_or_nan(
            statistic,
            matrix[indices],
        )
    valid = np.isfinite(bootstrap_estimates)
    valid_count = int(np.count_nonzero(valid))
    invalid_count = outer_count - valid_count
    if invalid_count / outer_count > invalid_limit:
        raise BootstrapInferenceFailure(
            "invalid outer-resample fraction exceeds the frozen limit",
            outer_replicates=outer_count,
            invalid_outer_replicates=invalid_count,
            attempted_inner_replicates=0,
            invalid_inner_replicates=0,
        )
    if valid_count < 2:
        raise BootstrapInferenceFailure(
            "fewer than two valid secondary percentile resamples remain",
            outer_replicates=outer_count,
            invalid_outer_replicates=invalid_count,
            attempted_inner_replicates=0,
            invalid_inner_replicates=0,
        )

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(
        bootstrap_estimates[valid],
        (tail, 1.0 - tail),
        method="linear",
    )
    if not math.isfinite(float(lower)) or not math.isfinite(float(upper)):
        raise BootstrapInferenceFailure(
            "secondary percentile confidence interval is nonfinite",
            outer_replicates=outer_count,
            invalid_outer_replicates=invalid_count,
            attempted_inner_replicates=0,
            invalid_inner_replicates=0,
        )
    return SecondaryPercentileBootstrapResult(
        label=SECONDARY_PERCENTILE_LABEL,
        statistic_id=statistic_id,
        estimate=estimate,
        confidence_level=confidence,
        confidence_lower=float(lower),
        confidence_upper=float(upper),
        outer_replicates=outer_count,
        valid_outer_replicates=valid_count,
        invalid_outer_replicates=invalid_count,
        expected_block_length=block_length,
        master_seed=seed,
        joint_outer_indices_sha256=_outer_indices_sha256(resolved_outer_indices),
        bootstrap_estimates=bootstrap_estimates,
        studentized=False,
    )


def stationary_bootstrap_alpha_percentile_interval(
    monthly_excess_returns: Sequence[object] | np.ndarray,
    monthly_factor_returns: Sequence[Sequence[object]] | np.ndarray,
    *,
    confidence_level: float = FROZEN_CONFIDENCE_LEVEL,
    outer_replicates: int = STUDENTIZED_OUTER_REPLICATES,
    expected_block_length: float = STATIONARY_BOOTSTRAP_EXPECTED_BLOCK,
    master_seed: int = STATIONARY_BOOTSTRAP_MASTER_SEED,
    max_invalid_outer_fraction: float = MAX_INVALID_OUTER_FRACTION,
    outer_indices: np.ndarray | None = None,
) -> SecondaryPercentileBootstrapResult:
    """Return the secondary joint return/factor percentile alpha interval."""

    dependent = _finite_vector(
        monthly_excess_returns,
        name="monthly_excess_returns",
        minimum_size=2 + len(FF5_UMD_FACTOR_NAMES),
    )
    factors = _finite_matrix(
        monthly_factor_returns,
        name="monthly_factor_returns",
    )
    if factors.shape != (dependent.size, len(FF5_UMD_FACTOR_NAMES)):
        raise FinalInferenceError(
            "monthly_factor_returns must have one joint row and exactly six "
            "FF5-plus-UMD columns"
        )
    return _stationary_bootstrap_percentile_interval(
        np.column_stack((dependent, factors)),
        _annualized_ff5_umd_alpha_or_nan,
        statistic_id=SECONDARY_ALPHA_STATISTIC_ID,
        confidence_level=confidence_level,
        outer_replicates=outer_replicates,
        expected_block_length=expected_block_length,
        master_seed=master_seed,
        max_invalid_outer_fraction=max_invalid_outer_fraction,
        outer_indices=outer_indices,
    )


def stationary_bootstrap_maximum_drawdown_percentile_interval(
    monthly_total_returns: Sequence[object] | np.ndarray,
    *,
    confidence_level: float = FROZEN_CONFIDENCE_LEVEL,
    outer_replicates: int = STUDENTIZED_OUTER_REPLICATES,
    expected_block_length: float = STATIONARY_BOOTSTRAP_EXPECTED_BLOCK,
    master_seed: int = STATIONARY_BOOTSTRAP_MASTER_SEED,
    max_invalid_outer_fraction: float = MAX_INVALID_OUTER_FRACTION,
    outer_indices: np.ndarray | None = None,
) -> SecondaryPercentileBootstrapResult:
    """Return the secondary nonstudentized maximum-drawdown percentile interval."""

    total_returns = _finite_vector(
        monthly_total_returns,
        name="monthly_total_returns",
        minimum_size=1,
    )
    if np.any(total_returns < -1.0):
        raise FinalInferenceError("monthly total returns cannot be below -100%")
    return _stationary_bootstrap_percentile_interval(
        total_returns[:, None],
        _maximum_drawdown_or_nan,
        statistic_id=SECONDARY_MAXIMUM_DRAWDOWN_STATISTIC_ID,
        confidence_level=confidence_level,
        outer_replicates=outer_replicates,
        expected_block_length=expected_block_length,
        master_seed=master_seed,
        max_invalid_outer_fraction=max_invalid_outer_fraction,
        outer_indices=outer_indices,
    )


def _studentized_interval(
    sample: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    statistic_id: str,
    confidence_level: float,
    outer_replicates: int,
    inner_replicates: int,
    expected_block_length: float,
    master_seed: int,
    max_invalid_outer_fraction: float,
    outer_indices: np.ndarray | None = None,
) -> StudentizedBootstrapResult:
    confidence = _validate_confidence(confidence_level)
    outer_count = _positive_int(outer_replicates, name="outer_replicates", minimum=2)
    inner_count = _positive_int(inner_replicates, name="inner_replicates", minimum=2)
    block_length = _validate_expected_block_length(expected_block_length)
    seed = _nonnegative_int(master_seed, name="master_seed")
    invalid_limit = _validate_invalid_fraction(max_invalid_outer_fraction)
    _derived_inner_seed(
        master_seed=seed,
        outer_index=-1,
        statistic_id=statistic_id,
        inner_index=0,
    )

    estimate = statistic(sample)
    if not math.isfinite(estimate):
        raise BootstrapInferenceFailure(
            "the original statistic is invalid",
            outer_replicates=outer_count,
            invalid_outer_replicates=outer_count,
            attempted_inner_replicates=0,
            invalid_inner_replicates=0,
        )

    original_standard_error, original_invalid_inner = _inner_standard_error(
        sample,
        statistic,
        inner_replicates=inner_count,
        expected_block_length=block_length,
        master_seed=seed,
        outer_index=-1,
        statistic_id=statistic_id,
    )
    attempted_inner = inner_count
    invalid_inner = original_invalid_inner
    if not math.isfinite(original_standard_error) or original_standard_error <= 0.0:
        raise BootstrapInferenceFailure(
            "the original inner-bootstrap standard error is invalid",
            outer_replicates=outer_count,
            invalid_outer_replicates=outer_count,
            attempted_inner_replicates=attempted_inner,
            invalid_inner_replicates=invalid_inner,
        )

    if outer_indices is None:
        resolved_outer_indices = stationary_bootstrap_indices(
            sample.shape[0],
            replicates=outer_count,
            expected_block_length=block_length,
            seed=seed,
        )
    else:
        candidate_indices = np.asarray(outer_indices)
        if (
            candidate_indices.ndim != 2
            or candidate_indices.shape != (outer_count, sample.shape[0])
            or candidate_indices.dtype.kind not in {"i", "u"}
            or np.any(candidate_indices < 0)
            or np.any(candidate_indices >= sample.shape[0])
        ):
            raise FinalInferenceError(
                "outer_indices must be an in-range integer matrix matching "
                "replicates and observations"
            )
        resolved_outer_indices = candidate_indices
    outer_estimates = np.full(outer_count, np.nan, dtype=np.float64)
    outer_standard_errors = np.full(outer_count, np.nan, dtype=np.float64)
    studentized = np.full(outer_count, np.nan, dtype=np.float64)

    for outer_index, indices in enumerate(resolved_outer_indices):
        outer_sample = sample[indices]
        outer_estimate = statistic(outer_sample)
        outer_estimates[outer_index] = outer_estimate
        if not math.isfinite(outer_estimate):
            continue
        outer_standard_error, outer_invalid_inner = _inner_standard_error(
            outer_sample,
            statistic,
            inner_replicates=inner_count,
            expected_block_length=block_length,
            master_seed=seed,
            outer_index=outer_index,
            statistic_id=statistic_id,
        )
        attempted_inner += inner_count
        invalid_inner += outer_invalid_inner
        outer_standard_errors[outer_index] = outer_standard_error
        if not math.isfinite(outer_standard_error) or outer_standard_error <= 0.0:
            continue
        value = (outer_estimate - estimate) / outer_standard_error
        if math.isfinite(value):
            studentized[outer_index] = value

    valid = np.isfinite(studentized)
    valid_count = int(np.count_nonzero(valid))
    invalid_outer = outer_count - valid_count
    if invalid_outer / outer_count > invalid_limit:
        raise BootstrapInferenceFailure(
            "invalid outer-resample fraction exceeds the frozen limit",
            outer_replicates=outer_count,
            invalid_outer_replicates=invalid_outer,
            attempted_inner_replicates=attempted_inner,
            invalid_inner_replicates=invalid_inner,
        )
    if valid_count < 2:
        raise BootstrapInferenceFailure(
            "fewer than two valid studentized outer resamples remain",
            outer_replicates=outer_count,
            invalid_outer_replicates=invalid_outer,
            attempted_inner_replicates=attempted_inner,
            invalid_inner_replicates=invalid_inner,
        )

    tail = (1.0 - confidence) / 2.0
    lower_quantile, upper_quantile = np.quantile(
        studentized[valid],
        (tail, 1.0 - tail),
        method="linear",
    )
    confidence_lower = estimate - float(upper_quantile) * original_standard_error
    confidence_upper = estimate - float(lower_quantile) * original_standard_error
    if not math.isfinite(confidence_lower) or not math.isfinite(confidence_upper):
        raise BootstrapInferenceFailure(
            "studentized confidence interval is nonfinite",
            outer_replicates=outer_count,
            invalid_outer_replicates=invalid_outer,
            attempted_inner_replicates=attempted_inner,
            invalid_inner_replicates=invalid_inner,
        )

    return StudentizedBootstrapResult(
        statistic_id=statistic_id,
        estimate=estimate,
        standard_error=original_standard_error,
        confidence_level=confidence,
        confidence_lower=confidence_lower,
        confidence_upper=confidence_upper,
        outer_replicates=outer_count,
        valid_outer_replicates=valid_count,
        invalid_outer_replicates=invalid_outer,
        inner_replicates_per_standard_error=inner_count,
        attempted_inner_replicates=attempted_inner,
        invalid_inner_replicates=invalid_inner,
        expected_block_length=block_length,
        master_seed=seed,
        outer_estimates=outer_estimates,
        outer_standard_errors=outer_standard_errors,
        studentized_statistics=studentized,
    )


def studentized_sharpe_interval(
    monthly_excess_returns: Sequence[object] | np.ndarray,
    *,
    statistic_id: str = "sharpe_level",
    confidence_level: float = 0.95,
    outer_replicates: int = STUDENTIZED_OUTER_REPLICATES,
    inner_replicates: int = STUDENTIZED_INNER_REPLICATES,
    expected_block_length: float = STATIONARY_BOOTSTRAP_EXPECTED_BLOCK,
    master_seed: int = STATIONARY_BOOTSTRAP_MASTER_SEED,
    max_invalid_outer_fraction: float = MAX_INVALID_OUTER_FRACTION,
    annualization_periods: int = ANNUALIZATION_PERIODS,
) -> StudentizedBootstrapResult:
    """Return a nested studentized stationary-bootstrap Sharpe interval."""

    values = _finite_vector(
        monthly_excess_returns,
        name="monthly_excess_returns",
        minimum_size=2,
    )
    periods = _positive_int(
        annualization_periods, name="annualization_periods", minimum=1
    )
    sample = values[:, None]

    def statistic(matrix: np.ndarray) -> float:
        return _annualized_sharpe_or_nan(matrix[:, 0], periods)

    return _studentized_interval(
        sample,
        statistic,
        statistic_id=statistic_id,
        confidence_level=confidence_level,
        outer_replicates=outer_replicates,
        inner_replicates=inner_replicates,
        expected_block_length=expected_block_length,
        master_seed=master_seed,
        max_invalid_outer_fraction=max_invalid_outer_fraction,
    )


def studentized_paired_sharpe_interval(
    strategy_monthly_excess_returns: Sequence[object] | np.ndarray,
    benchmark_monthly_excess_returns: Sequence[object] | np.ndarray,
    *,
    statistic_id: str,
    confidence_level: float = 0.95,
    outer_replicates: int = STUDENTIZED_OUTER_REPLICATES,
    inner_replicates: int = STUDENTIZED_INNER_REPLICATES,
    expected_block_length: float = STATIONARY_BOOTSTRAP_EXPECTED_BLOCK,
    master_seed: int = STATIONARY_BOOTSTRAP_MASTER_SEED,
    max_invalid_outer_fraction: float = MAX_INVALID_OUTER_FRACTION,
    annualization_periods: int = ANNUALIZATION_PERIODS,
) -> StudentizedBootstrapResult:
    """Return a paired interval, recomputing both Sharpes on each joint row draw."""

    strategy = _finite_vector(
        strategy_monthly_excess_returns,
        name="strategy_monthly_excess_returns",
        minimum_size=2,
    )
    benchmark = _finite_vector(
        benchmark_monthly_excess_returns,
        name="benchmark_monthly_excess_returns",
        minimum_size=2,
    )
    if strategy.shape != benchmark.shape:
        raise FinalInferenceError("paired Sharpe inputs must have equal length")
    periods = _positive_int(
        annualization_periods, name="annualization_periods", minimum=1
    )
    sample = np.column_stack((strategy, benchmark))

    def statistic(matrix: np.ndarray) -> float:
        left = _annualized_sharpe_or_nan(matrix[:, 0], periods)
        right = _annualized_sharpe_or_nan(matrix[:, 1], periods)
        if not math.isfinite(left) or not math.isfinite(right):
            return math.nan
        result = left - right
        return result if math.isfinite(result) else math.nan

    return _studentized_interval(
        sample,
        statistic,
        statistic_id=statistic_id,
        confidence_level=confidence_level,
        outer_replicates=outer_replicates,
        inner_replicates=inner_replicates,
        expected_block_length=expected_block_length,
        master_seed=master_seed,
        max_invalid_outer_fraction=max_invalid_outer_fraction,
    )


@dataclass(frozen=True)
class HACCoefficient:
    name: str
    estimate: float
    standard_error: float
    statistic: float
    p_value: float
    confidence_lower: float
    confidence_upper: float


@dataclass(frozen=True)
class FF5UMDHACResult:
    """OLS FF5-plus-UMD estimates with Bartlett Newey--West covariance."""

    observations: int
    lags: int
    confidence_level: float
    coefficient_names: tuple[str, ...]
    coefficients: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    statistics: np.ndarray
    p_values: np.ndarray
    confidence_intervals: np.ndarray
    residuals: np.ndarray
    r_squared: float
    residual_volatility_annualized: float
    factor_adjusted_information_ratio: float

    def __post_init__(self) -> None:
        expected = len(self.coefficient_names)
        arrays_and_shapes = (
            ("coefficients", (expected,)),
            ("covariance", (expected, expected)),
            ("standard_errors", (expected,)),
            ("statistics", (expected,)),
            ("p_values", (expected,)),
            ("confidence_intervals", (expected, 2)),
            ("residuals", (self.observations,)),
        )
        for field_name, shape in arrays_and_shapes:
            value = _readonly_float_array(getattr(self, field_name))
            if value.shape != shape:
                raise FinalInferenceError(f"{field_name} has an invalid shape")
            object.__setattr__(self, field_name, value)

    def coefficient(self, name: str) -> HACCoefficient:
        try:
            index = self.coefficient_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return HACCoefficient(
            name=name,
            estimate=float(self.coefficients[index]),
            standard_error=float(self.standard_errors[index]),
            statistic=float(self.statistics[index]),
            p_value=float(self.p_values[index]),
            confidence_lower=float(self.confidence_intervals[index, 0]),
            confidence_upper=float(self.confidence_intervals[index, 1]),
        )

    @property
    def monthly_alpha(self) -> HACCoefficient:
        return self.coefficient("alpha")

    @property
    def annualized_alpha(self) -> float:
        return ANNUALIZATION_PERIODS * self.monthly_alpha.estimate

    @property
    def annualized_alpha_confidence_interval(self) -> tuple[float, float]:
        alpha = self.monthly_alpha
        return (
            ANNUALIZATION_PERIODS * alpha.confidence_lower,
            ANNUALIZATION_PERIODS * alpha.confidence_upper,
        )

    @property
    def market_beta(self) -> HACCoefficient:
        return self.coefficient("mktrf")


def ff5_umd_hac_regression(
    monthly_excess_returns: Sequence[object] | np.ndarray,
    monthly_factor_returns: Sequence[Sequence[object]] | np.ndarray,
    *,
    lags: int = NEWEY_WEST_LAGS,
    confidence_level: float = 0.95,
) -> FF5UMDHACResult:
    """Fit monthly OLS with intercept and Newey--West HAC standard errors.

    Factor columns must be ordered ``mktrf, smb, hml, rmw, cma, umd``.  The
    Bartlett weights are ``1 - lag / (lags + 1)``.  The conventional
    ``T / (T - K)`` finite-sample covariance correction is frozen on.
    """

    dependent = _finite_vector(
        monthly_excess_returns,
        name="monthly_excess_returns",
        minimum_size=2,
    )
    factors = _finite_matrix(monthly_factor_returns, name="monthly_factor_returns")
    if factors.shape != (dependent.size, len(FF5_UMD_FACTOR_NAMES)):
        raise FinalInferenceError(
            "monthly_factor_returns must have one paired row and exactly six "
            "FF5-plus-UMD columns"
        )
    lag_count = _nonnegative_int(lags, name="lags")
    if lag_count >= dependent.size:
        raise FinalInferenceError("lags must be smaller than the observation count")
    confidence = _validate_confidence(confidence_level)

    design = np.column_stack((np.ones(dependent.size, dtype=np.float64), factors))
    parameter_count = design.shape[1]
    if dependent.size <= parameter_count:
        raise FinalInferenceError(
            "OLS requires more observations than FF5-plus-UMD coefficients"
        )
    coefficients, _, rank, _ = np.linalg.lstsq(design, dependent, rcond=None)
    if rank != parameter_count:
        raise FinalInferenceError("FF5-plus-UMD design matrix is rank deficient")
    residuals = dependent - design @ coefficients

    score = design * residuals[:, None]
    meat = score.T @ score
    for lag in range(1, lag_count + 1):
        weight = 1.0 - lag / (lag_count + 1.0)
        cross = score[lag:].T @ score[:-lag]
        meat += weight * (cross + cross.T)

    inverse_cross_product = np.linalg.inv(design.T @ design)
    covariance = inverse_cross_product @ meat @ inverse_cross_product
    covariance *= dependent.size / (dependent.size - parameter_count)
    covariance = (covariance + covariance.T) * 0.5
    diagonal = np.diag(covariance).copy()
    scale = max(1.0, float(np.max(np.abs(covariance))))
    tolerance = 100.0 * np.finfo(np.float64).eps * scale
    if np.any(diagonal < -tolerance) or not np.isfinite(covariance).all():
        raise FinalInferenceError(
            "Newey-West covariance is not finite positive semidefinite"
        )
    diagonal = np.maximum(diagonal, 0.0)
    standard_errors = np.sqrt(diagonal)
    statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.full(parameter_count, np.nan, dtype=np.float64),
        where=standard_errors > 0.0,
    )
    p_values = 2.0 * norm.sf(np.abs(statistics))
    critical_value = float(norm.ppf(0.5 + confidence / 2.0))
    confidence_intervals = np.column_stack(
        (
            coefficients - critical_value * standard_errors,
            coefficients + critical_value * standard_errors,
        )
    )
    if not (
        np.isfinite(coefficients).all()
        and np.isfinite(standard_errors).all()
        and np.isfinite(confidence_intervals).all()
    ):
        raise FinalInferenceError("FF5-plus-UMD coefficient inference is nonfinite")

    centered = dependent - float(np.mean(dependent))
    total_sum_squares = float(np.dot(centered, centered))
    residual_sum_squares = float(np.dot(residuals, residuals))
    r_squared = (
        math.nan
        if total_sum_squares <= 0.0
        else 1.0 - residual_sum_squares / total_sum_squares
    )
    residual_volatility = float(
        np.std(residuals, ddof=1) * math.sqrt(ANNUALIZATION_PERIODS)
    )
    annualized_alpha = ANNUALIZATION_PERIODS * float(coefficients[0])
    information_ratio = (
        math.nan
        if residual_volatility <= 0.0
        else annualized_alpha / residual_volatility
    )
    return FF5UMDHACResult(
        observations=dependent.size,
        lags=lag_count,
        confidence_level=confidence,
        coefficient_names=("alpha", *FF5_UMD_FACTOR_NAMES),
        coefficients=coefficients,
        covariance=covariance,
        standard_errors=standard_errors,
        statistics=statistics,
        p_values=p_values,
        confidence_intervals=confidence_intervals,
        residuals=residuals,
        r_squared=r_squared,
        residual_volatility_annualized=residual_volatility,
        factor_adjusted_information_ratio=information_ratio,
    )


@dataclass(frozen=True, slots=True)
class AutomaticBandwidthHACRobustnessResult:
    """Secondary HAC robustness under the frozen automatic bandwidth rule."""

    label: str
    bandwidth_rule_id: str
    selected_lags: int
    regression: FF5UMDHACResult

    def __post_init__(self) -> None:
        if self.label != SECONDARY_AUTOMATIC_HAC_LABEL:
            raise FinalInferenceError("automatic-HAC result is not secondary-labelled")
        if self.bandwidth_rule_id != AUTOMATIC_HAC_BANDWIDTH_RULE_ID:
            raise FinalInferenceError("automatic-HAC bandwidth rule differs")
        if not isinstance(self.regression, FF5UMDHACResult):
            raise FinalInferenceError("automatic-HAC regression is missing")
        expected = automatic_newey_west_lags(self.regression.observations)
        if self.selected_lags != expected or self.regression.lags != expected:
            raise FinalInferenceError("automatic-HAC selected bandwidth differs")


def ff5_umd_automatic_bandwidth_hac_robustness(
    monthly_excess_returns: Sequence[object] | np.ndarray,
    monthly_factor_returns: Sequence[Sequence[object]] | np.ndarray,
    *,
    confidence_level: float = FROZEN_CONFIDENCE_LEVEL,
) -> AutomaticBandwidthHACRobustnessResult:
    """Fit the explicitly secondary automatic-bandwidth HAC robustness."""

    dependent = _finite_vector(
        monthly_excess_returns,
        name="monthly_excess_returns",
        minimum_size=2,
    )
    lags = automatic_newey_west_lags(int(dependent.size))
    regression = ff5_umd_hac_regression(
        dependent,
        monthly_factor_returns,
        lags=lags,
        confidence_level=confidence_level,
    )
    return AutomaticBandwidthHACRobustnessResult(
        label=SECONDARY_AUTOMATIC_HAC_LABEL,
        bandwidth_rule_id=AUTOMATIC_HAC_BANDWIDTH_RULE_ID,
        selected_lags=lags,
        regression=regression,
    )


def _require_exact_family_mapping(value: Mapping[str, object], *, name: str) -> None:
    if not isinstance(value, Mapping):
        raise FinalInferenceError(f"{name} must be a mapping")
    actual = frozenset(value)
    expected = frozenset(ROMANO_WOLF_FAMILY)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise FinalInferenceError(
            f"{name} must contain the exact five-family keys; "
            f"missing={missing}, extra={extra}"
        )


@dataclass(frozen=True)
class RomanoWolfComparison:
    comparison_id: str
    active: bool
    observed_difference: float | None
    standard_error: float | None
    observed_statistic: float | None
    unadjusted_p_value: float
    adjusted_p_value: float
    step: int | None


@dataclass(frozen=True)
class RomanoWolfResult:
    family: tuple[str, ...]
    comparisons: tuple[RomanoWolfComparison, ...]
    outer_replicates: int
    valid_joint_outer_replicates: int
    invalid_joint_outer_replicates: int

    def comparison(self, comparison_id: str) -> RomanoWolfComparison:
        for comparison in self.comparisons:
            if comparison.comparison_id == comparison_id:
                return comparison
        raise KeyError(comparison_id)

    @property
    def invalid_joint_outer_fraction(self) -> float:
        if self.outer_replicates == 0:
            return 0.0
        return self.invalid_joint_outer_replicates / self.outer_replicates


def romano_wolf_stepdown(
    *,
    observed_differences: Mapping[str, float | None],
    standard_errors: Mapping[str, float | None],
    outer_differences: Mapping[str, Sequence[object] | np.ndarray | None],
    outer_standard_errors: Mapping[str, Sequence[object] | np.ndarray | None],
    active: Mapping[str, bool],
    max_invalid_outer_fraction: float = MAX_INVALID_OUTER_FRACTION,
) -> RomanoWolfResult:
    """Apply joint recentered studentized Romano--Wolf max-T stepdown.

    Every mapping must retain the exact frozen five-comparison family.  An
    inactive challenger is reported with both p-values equal to one and never
    disappears from the output.  Active comparisons must share aligned outer
    resamples; a row invalid for any active comparison is excluded jointly.
    """

    for name, mapping in (
        ("observed_differences", observed_differences),
        ("standard_errors", standard_errors),
        ("outer_differences", outer_differences),
        ("outer_standard_errors", outer_standard_errors),
        ("active", active),
    ):
        _require_exact_family_mapping(mapping, name=name)
    invalid_limit = _validate_invalid_fraction(max_invalid_outer_fraction)

    active_indices: list[int] = []
    observed = np.full(len(ROMANO_WOLF_FAMILY), np.nan, dtype=np.float64)
    standard_error = np.full(len(ROMANO_WOLF_FAMILY), np.nan, dtype=np.float64)
    active_flags: list[bool] = []
    outer_count: int | None = None
    bootstrap_columns: list[np.ndarray] = []

    for index, comparison_id in enumerate(ROMANO_WOLF_FAMILY):
        flag_value = active[comparison_id]
        if not isinstance(flag_value, (bool, np.bool_)):
            raise FinalInferenceError("active values must be boolean")
        flag = bool(flag_value)
        active_flags.append(flag)
        if not flag:
            continue
        try:
            difference = float(observed_differences[comparison_id])
            error = float(standard_errors[comparison_id])
        except (TypeError, ValueError, OverflowError) as exc:
            raise FinalInferenceError(
                f"active comparison {comparison_id} lacks numeric estimates"
            ) from exc
        if not math.isfinite(difference):
            raise FinalInferenceError(
                f"active comparison {comparison_id} has a nonfinite difference"
            )
        if not math.isfinite(error) or error <= 0.0:
            raise FinalInferenceError(
                f"active comparison {comparison_id} has an invalid standard error"
            )
        differences_value = outer_differences[comparison_id]
        errors_value = outer_standard_errors[comparison_id]
        if differences_value is None or errors_value is None:
            raise FinalInferenceError(
                f"active comparison {comparison_id} lacks outer resamples"
            )
        try:
            differences = np.asarray(differences_value, dtype=np.float64)
            errors = np.asarray(errors_value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FinalInferenceError(
                f"active comparison {comparison_id} outer resamples must be numeric"
            ) from exc
        if (
            differences.ndim != 1
            or errors.ndim != 1
            or differences.shape != errors.shape
        ):
            raise FinalInferenceError(
                f"active comparison {comparison_id} outer arrays must be equal-length vectors"
            )
        if differences.size < 2:
            raise FinalInferenceError(
                "Romano-Wolf requires at least two outer resamples"
            )
        if outer_count is None:
            outer_count = int(differences.size)
        elif differences.size != outer_count:
            raise FinalInferenceError(
                "all active comparisons must use the same aligned outer resamples"
            )
        observed[index] = difference
        standard_error[index] = error
        with np.errstate(divide="ignore", invalid="ignore"):
            recentered = (differences - difference) / errors
        recentered[(~np.isfinite(errors)) | (errors <= 0.0)] = np.nan
        bootstrap_columns.append(recentered)
        active_indices.append(index)

    if not active_indices:
        comparisons = tuple(
            RomanoWolfComparison(
                comparison_id=comparison_id,
                active=False,
                observed_difference=None,
                standard_error=None,
                observed_statistic=None,
                unadjusted_p_value=1.0,
                adjusted_p_value=1.0,
                step=None,
            )
            for comparison_id in ROMANO_WOLF_FAMILY
        )
        return RomanoWolfResult(
            family=ROMANO_WOLF_FAMILY,
            comparisons=comparisons,
            outer_replicates=0,
            valid_joint_outer_replicates=0,
            invalid_joint_outer_replicates=0,
        )

    assert outer_count is not None
    bootstrap_statistics = np.column_stack(bootstrap_columns)
    valid_joint = np.isfinite(bootstrap_statistics).all(axis=1)
    valid_count = int(np.count_nonzero(valid_joint))
    invalid_count = outer_count - valid_count
    if invalid_count / outer_count > invalid_limit:
        raise BootstrapInferenceFailure(
            "Romano-Wolf joint invalid outer-resample fraction exceeds the frozen limit",
            outer_replicates=outer_count,
            invalid_outer_replicates=invalid_count,
            attempted_inner_replicates=0,
            invalid_inner_replicates=0,
        )
    if valid_count < 2:
        raise BootstrapInferenceFailure(
            "Romano-Wolf has fewer than two valid joint outer resamples",
            outer_replicates=outer_count,
            invalid_outer_replicates=invalid_count,
            attempted_inner_replicates=0,
            invalid_inner_replicates=0,
        )
    bootstrap_statistics = bootstrap_statistics[valid_joint]
    observed_statistics = observed[active_indices] / standard_error[active_indices]

    order = sorted(
        range(len(active_indices)),
        key=lambda active_position: (
            -abs(float(observed_statistics[active_position])),
            active_indices[active_position],
        ),
    )
    unadjusted = np.ones(len(active_indices), dtype=np.float64)
    adjusted = np.ones(len(active_indices), dtype=np.float64)
    steps = np.zeros(len(active_indices), dtype=np.int64)
    denominator = valid_count + 1.0
    running_adjusted = 0.0
    for step_index, ordered_position in enumerate(order):
        observed_absolute = abs(float(observed_statistics[ordered_position]))
        own_bootstrap = np.abs(bootstrap_statistics[:, ordered_position])
        unadjusted[ordered_position] = (
            1.0 + np.count_nonzero(own_bootstrap >= observed_absolute)
        ) / denominator
        remaining = order[step_index:]
        maximum_remaining = np.max(np.abs(bootstrap_statistics[:, remaining]), axis=1)
        raw_adjusted = (
            1.0 + np.count_nonzero(maximum_remaining >= observed_absolute)
        ) / denominator
        running_adjusted = max(running_adjusted, float(raw_adjusted))
        adjusted[ordered_position] = min(1.0, running_adjusted)
        steps[ordered_position] = step_index + 1

    active_position_by_index = {
        family_index: active_position
        for active_position, family_index in enumerate(active_indices)
    }
    comparison_results: list[RomanoWolfComparison] = []
    for family_index, comparison_id in enumerate(ROMANO_WOLF_FAMILY):
        if not active_flags[family_index]:
            comparison_results.append(
                RomanoWolfComparison(
                    comparison_id=comparison_id,
                    active=False,
                    observed_difference=None,
                    standard_error=None,
                    observed_statistic=None,
                    unadjusted_p_value=1.0,
                    adjusted_p_value=1.0,
                    step=None,
                )
            )
            continue
        active_position = active_position_by_index[family_index]
        comparison_results.append(
            RomanoWolfComparison(
                comparison_id=comparison_id,
                active=True,
                observed_difference=float(observed[family_index]),
                standard_error=float(standard_error[family_index]),
                observed_statistic=float(observed_statistics[active_position]),
                unadjusted_p_value=float(unadjusted[active_position]),
                adjusted_p_value=float(adjusted[active_position]),
                step=int(steps[active_position]),
            )
        )
    return RomanoWolfResult(
        family=ROMANO_WOLF_FAMILY,
        comparisons=tuple(comparison_results),
        outer_replicates=outer_count,
        valid_joint_outer_replicates=valid_count,
        invalid_joint_outer_replicates=invalid_count,
    )


EULER_MASCHERONI = 0.577_215_664_901_532_9


def expected_maximum_sharpe(
    trial_sharpe_standard_deviation: float,
    *,
    raw_trial_count: int,
) -> float:
    """Return the Bailey--Lopez de Prado zero-mean expected maximum Sharpe."""

    try:
        dispersion = float(trial_sharpe_standard_deviation)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalInferenceError(
            "trial_sharpe_standard_deviation must be numeric"
        ) from exc
    if not math.isfinite(dispersion) or dispersion < 0.0:
        raise FinalInferenceError(
            "trial_sharpe_standard_deviation must be finite and nonnegative"
        )
    trials = _positive_int(raw_trial_count, name="raw_trial_count", minimum=1)
    if trials == 1 or dispersion == 0.0:
        return 0.0
    maximum_standard_normal = (1.0 - EULER_MASCHERONI) * norm.ppf(
        1.0 - 1.0 / trials
    ) + EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (trials * math.e))
    result = dispersion * float(maximum_standard_normal)
    if not math.isfinite(result):
        raise FinalInferenceError("expected maximum Sharpe ratio is nonfinite")
    return result


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Probabilistic Deflated Sharpe result in per-observation Sharpe units."""

    observed_sharpe: float
    expected_maximum_sharpe: float
    trial_sharpe_standard_deviation: float
    observations: int
    skewness: float
    kurtosis: float
    raw_trial_count: int
    z_statistic: float
    probability: float
    annualization_periods: int

    @property
    def deflated_sharpe_ratio(self) -> float:
        return self.probability

    @property
    def annualized_observed_sharpe(self) -> float:
        return math.sqrt(self.annualization_periods) * self.observed_sharpe

    @property
    def annualized_expected_maximum_sharpe(self) -> float:
        return math.sqrt(self.annualization_periods) * self.expected_maximum_sharpe


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    observations: int,
    skewness: float,
    kurtosis: float,
    raw_trial_count: int,
    trial_sharpe_standard_deviation: float,
    annualization_periods: int = ANNUALIZATION_PERIODS,
) -> DeflatedSharpeResult:
    """Compute DSR from per-observation Sharpe and raw frozen-ledger count.

    ``kurtosis`` is Pearson kurtosis (three for a normal distribution), not
    excess kurtosis.  The observed Sharpe and cross-trial Sharpe dispersion
    must use the same per-observation scale.  Annualized values are exposed as
    reporting properties only.
    """

    try:
        sharpe = float(observed_sharpe)
        sample_skewness = float(skewness)
        sample_kurtosis = float(kurtosis)
        dispersion = float(trial_sharpe_standard_deviation)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalInferenceError("Deflated Sharpe inputs must be numeric") from exc
    if not all(
        math.isfinite(value)
        for value in (sharpe, sample_skewness, sample_kurtosis, dispersion)
    ):
        raise FinalInferenceError("Deflated Sharpe inputs must be finite")
    if sample_kurtosis <= 0.0:
        raise FinalInferenceError("kurtosis must be positive Pearson kurtosis")
    if dispersion < 0.0:
        raise FinalInferenceError("trial_sharpe_standard_deviation must be nonnegative")
    sample_size = _positive_int(observations, name="observations", minimum=2)
    trials = _positive_int(raw_trial_count, name="raw_trial_count", minimum=1)
    periods = _positive_int(
        annualization_periods, name="annualization_periods", minimum=1
    )
    benchmark = expected_maximum_sharpe(dispersion, raw_trial_count=trials)
    denominator_squared = (
        1.0
        - sample_skewness * sharpe
        + ((sample_kurtosis - 1.0) / 4.0) * sharpe * sharpe
    )
    if not math.isfinite(denominator_squared) or denominator_squared <= 0.0:
        raise FinalInferenceError("Deflated Sharpe sampling variance is nonpositive")
    z_statistic = (
        (sharpe - benchmark)
        * math.sqrt(sample_size - 1.0)
        / math.sqrt(denominator_squared)
    )
    probability = float(norm.cdf(z_statistic))
    if not math.isfinite(z_statistic) or not math.isfinite(probability):
        raise FinalInferenceError("Deflated Sharpe result is nonfinite")
    return DeflatedSharpeResult(
        observed_sharpe=sharpe,
        expected_maximum_sharpe=benchmark,
        trial_sharpe_standard_deviation=dispersion,
        observations=sample_size,
        skewness=sample_skewness,
        kurtosis=sample_kurtosis,
        raw_trial_count=trials,
        z_statistic=z_statistic,
        probability=probability,
        annualization_periods=periods,
    )


def deflated_sharpe_from_returns(
    monthly_excess_returns: Sequence[object] | np.ndarray,
    *,
    trial_sharpes: Sequence[object] | np.ndarray,
    raw_trial_count: int,
    trial_sharpes_are_annualized: bool = True,
    annualization_periods: int = ANNUALIZATION_PERIODS,
) -> DeflatedSharpeResult:
    """Estimate moments from returns and apply raw-count DSR.

    ``trial_sharpes`` contains the usable trial Sharpe estimates needed to
    estimate cross-trial dispersion.  ``raw_trial_count`` is nevertheless the
    complete frozen-ledger count, including registered failed/inactive trials.
    Consequently it may exceed the number of usable trial Sharpes, but it may
    never be smaller.
    """

    values = _finite_vector(
        monthly_excess_returns,
        name="monthly_excess_returns",
        minimum_size=4,
    )
    trials = _finite_vector(
        trial_sharpes,
        name="trial_sharpes",
        minimum_size=1,
    )
    raw_count = _positive_int(raw_trial_count, name="raw_trial_count", minimum=1)
    if raw_count < trials.size:
        raise FinalInferenceError(
            "raw_trial_count cannot be smaller than the usable trial Sharpe count"
        )
    periods = _positive_int(
        annualization_periods, name="annualization_periods", minimum=1
    )
    if not isinstance(trial_sharpes_are_annualized, (bool, np.bool_)):
        raise FinalInferenceError("trial_sharpes_are_annualized must be boolean")

    monthly_standard_deviation = float(np.std(values, ddof=1))
    if monthly_standard_deviation <= 0.0 or not math.isfinite(
        monthly_standard_deviation
    ):
        raise FinalInferenceError("monthly_excess_returns have zero variance")
    observed_monthly_sharpe = float(np.mean(values)) / monthly_standard_deviation
    trial_values = (
        trials / math.sqrt(periods) if trial_sharpes_are_annualized else trials
    )
    if trial_values.size == 1:
        if raw_count != 1:
            raise FinalInferenceError(
                "at least two usable trial Sharpes are required to estimate dispersion"
            )
        trial_dispersion = 0.0
    else:
        trial_dispersion = float(np.std(trial_values, ddof=1))

    sample_skewness = float(scipy_skew(values, bias=False))
    sample_kurtosis = float(scipy_kurtosis(values, fisher=False, bias=False))
    return deflated_sharpe_ratio(
        observed_monthly_sharpe,
        observations=values.size,
        skewness=sample_skewness,
        kurtosis=sample_kurtosis,
        raw_trial_count=raw_count,
        trial_sharpe_standard_deviation=trial_dispersion,
        annualization_periods=periods,
    )


@dataclass(frozen=True, slots=True)
class FrozenTrialLedgerBinding:
    """Exact Stage-12 trial-ledger identity required by primary DSR."""

    ledger_sha256: str
    raw_trial_count: int = FROZEN_RAW_TRIAL_COUNT
    ledger_id: str = FROZEN_TRIAL_LEDGER_ID

    def __post_init__(self) -> None:
        _require_sha256(self.ledger_sha256, name="ledger_sha256")
        if self.raw_trial_count != FROZEN_RAW_TRIAL_COUNT:
            raise FinalInferenceError(
                f"raw_trial_count must equal frozen ledger count "
                f"{FROZEN_RAW_TRIAL_COUNT}"
            )
        if self.ledger_id != FROZEN_TRIAL_LEDGER_ID:
            raise FinalInferenceError("trial ledger identity differs from Stage 12")


def _joint_outer_receipt_payload(*, indices_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "observations": FROZEN_PRIMARY_OBSERVATIONS,
        "outer_replicates": STUDENTIZED_OUTER_REPLICATES,
        "expected_block_months": STATIONARY_BOOTSTRAP_EXPECTED_BLOCK,
        "master_seed": STATIONARY_BOOTSTRAP_MASTER_SEED,
        "indices_dtype": "little_endian_int32",
        "indices_sha256": indices_sha256,
    }


@dataclass(frozen=True, slots=True)
class FrozenJointOuterIndexReceipt:
    """Content receipt for the single joint 10,000-by-68 row resample."""

    schema_version: int
    observations: int
    outer_replicates: int
    expected_block_months: int
    master_seed: int
    indices_dtype: str
    indices_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise FinalInferenceError("joint outer-index schema version differs")
        if (
            type(self.observations) is not int
            or self.observations != FROZEN_PRIMARY_OBSERVATIONS
        ):
            raise FinalInferenceError("joint outer-index observation count differs")
        if (
            type(self.outer_replicates) is not int
            or self.outer_replicates != STUDENTIZED_OUTER_REPLICATES
        ):
            raise FinalInferenceError("joint outer-index replicate count differs")
        if (
            type(self.expected_block_months) is not int
            or self.expected_block_months != STATIONARY_BOOTSTRAP_EXPECTED_BLOCK
        ):
            raise FinalInferenceError("joint outer-index block length differs")
        if (
            type(self.master_seed) is not int
            or self.master_seed != STATIONARY_BOOTSTRAP_MASTER_SEED
        ):
            raise FinalInferenceError("joint outer-index master seed differs")
        if self.indices_dtype != "little_endian_int32":
            raise FinalInferenceError("joint outer-index dtype differs")
        _require_sha256(self.indices_sha256, name="indices_sha256")
        if self.indices_sha256 != FROZEN_JOINT_OUTER_INDICES_SHA256:
            raise FinalInferenceError("joint outer-index deterministic hash differs")
        _require_sha256(self.receipt_sha256, name="receipt_sha256")
        expected = hashlib.sha256(
            _canonical_json_bytes(
                _joint_outer_receipt_payload(indices_sha256=self.indices_sha256)
            )
        ).hexdigest()
        if self.receipt_sha256 != expected:
            raise FinalInferenceError("joint outer-index receipt hash differs")


def _frozen_joint_outer_indices() -> tuple[np.ndarray, FrozenJointOuterIndexReceipt]:
    indices = stationary_bootstrap_indices(
        FROZEN_PRIMARY_OBSERVATIONS,
        replicates=STUDENTIZED_OUTER_REPLICATES,
        expected_block_length=STATIONARY_BOOTSTRAP_EXPECTED_BLOCK,
        seed=STATIONARY_BOOTSTRAP_MASTER_SEED,
    )
    indices_sha256 = _outer_indices_sha256(indices)
    if indices_sha256 != FROZEN_JOINT_OUTER_INDICES_SHA256:
        raise FinalInferenceError(
            "the production joint outer indices differ from the frozen receipt"
        )
    payload = _joint_outer_receipt_payload(indices_sha256=indices_sha256)
    receipt = FrozenJointOuterIndexReceipt(
        schema_version=1,
        observations=FROZEN_PRIMARY_OBSERVATIONS,
        outer_replicates=STUDENTIZED_OUTER_REPLICATES,
        expected_block_months=STATIONARY_BOOTSTRAP_EXPECTED_BLOCK,
        master_seed=STATIONARY_BOOTSTRAP_MASTER_SEED,
        indices_dtype="little_endian_int32",
        indices_sha256=indices_sha256,
        receipt_sha256=hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
    )
    return indices, receipt


def frozen_joint_outer_index_receipt() -> FrozenJointOuterIndexReceipt:
    """Reproduce the immutable production joint-resample receipt."""

    _, receipt = _frozen_joint_outer_indices()
    return receipt


@dataclass(frozen=True, slots=True)
class FrozenPrimaryInferencePanel:
    """Exact joint 68-month input panel for confirmatory inference."""

    holding_months: tuple[str, ...]
    hybrid_monthly_excess_returns: np.ndarray
    hybrid_monthly_total_returns: np.ndarray
    comparison_monthly_excess_returns: Mapping[str, np.ndarray]
    monthly_factor_returns: np.ndarray

    def __post_init__(self) -> None:
        try:
            months = require_exact_primary_inference_months(self.holding_months)
        except FinalCalendarError as exc:
            raise FinalInferenceError(str(exc)) from exc
        hybrid = _finite_vector(
            self.hybrid_monthly_excess_returns,
            name="hybrid_monthly_excess_returns",
            minimum_size=FROZEN_PRIMARY_OBSERVATIONS,
        )
        if hybrid.size != FROZEN_PRIMARY_OBSERVATIONS:
            raise FinalInferenceError("hybrid returns must have exactly 68 rows")
        total = _finite_vector(
            self.hybrid_monthly_total_returns,
            name="hybrid_monthly_total_returns",
            minimum_size=FROZEN_PRIMARY_OBSERVATIONS,
        )
        if total.size != FROZEN_PRIMARY_OBSERVATIONS:
            raise FinalInferenceError("hybrid total returns must have exactly 68 rows")
        if np.any(total < -1.0):
            raise FinalInferenceError("hybrid total returns cannot be below -100%")
        if not isinstance(self.comparison_monthly_excess_returns, Mapping):
            raise FinalInferenceError(
                "comparison_monthly_excess_returns must be a mapping"
            )
        actual = frozenset(self.comparison_monthly_excess_returns)
        expected = frozenset(FROZEN_ACTIVE_ROMANO_WOLF_FAMILY)
        if actual != expected:
            raise FinalInferenceError(
                "comparison returns must contain exactly the four active "
                "Romano-Wolf comparisons"
            )
        comparisons: dict[str, np.ndarray] = {}
        for comparison_id in FROZEN_ACTIVE_ROMANO_WOLF_FAMILY:
            values = _finite_vector(
                self.comparison_monthly_excess_returns[comparison_id],
                name=f"comparison_monthly_excess_returns[{comparison_id}]",
                minimum_size=FROZEN_PRIMARY_OBSERVATIONS,
            )
            if values.size != FROZEN_PRIMARY_OBSERVATIONS:
                raise FinalInferenceError(
                    f"comparison {comparison_id} must have exactly 68 rows"
                )
            comparisons[comparison_id] = _readonly_float_array(values)
        factors = _finite_matrix(
            self.monthly_factor_returns,
            name="monthly_factor_returns",
        )
        if factors.shape != (
            FROZEN_PRIMARY_OBSERVATIONS,
            len(FF5_UMD_FACTOR_NAMES),
        ):
            raise FinalInferenceError(
                "monthly_factor_returns must be the joint 68-by-6 FF5-UMD panel"
            )
        object.__setattr__(self, "holding_months", months)
        object.__setattr__(
            self,
            "hybrid_monthly_excess_returns",
            _readonly_float_array(hybrid),
        )
        object.__setattr__(
            self,
            "hybrid_monthly_total_returns",
            _readonly_float_array(total),
        )
        object.__setattr__(
            self,
            "comparison_monthly_excess_returns",
            MappingProxyType(comparisons),
        )
        object.__setattr__(
            self,
            "monthly_factor_returns",
            _readonly_float_array(factors),
        )


@dataclass(frozen=True, slots=True)
class FrozenPrimaryInferenceResult:
    """Complete production inference output with immutable provenance."""

    holding_months: tuple[str, ...]
    joint_outer_index_receipt: FrozenJointOuterIndexReceipt
    trial_ledger_binding: FrozenTrialLedgerBinding
    primary_sharpe_interval: StudentizedBootstrapResult
    paired_sharpe_intervals: Mapping[str, StudentizedBootstrapResult]
    hac_result: FF5UMDHACResult
    automatic_bandwidth_hac_robustness: AutomaticBandwidthHACRobustnessResult
    secondary_alpha_percentile_interval: SecondaryPercentileBootstrapResult
    secondary_maximum_drawdown_percentile_interval: SecondaryPercentileBootstrapResult
    romano_wolf_result: RomanoWolfResult
    deflated_sharpe_result: DeflatedSharpeResult

    def __post_init__(self) -> None:
        try:
            months = require_exact_primary_inference_months(self.holding_months)
        except FinalCalendarError as exc:
            raise FinalInferenceError(str(exc)) from exc
        if not isinstance(self.joint_outer_index_receipt, FrozenJointOuterIndexReceipt):
            raise FinalInferenceError("joint outer-index receipt is missing")
        if not isinstance(self.trial_ledger_binding, FrozenTrialLedgerBinding):
            raise FinalInferenceError("trial-ledger binding is missing")
        if not isinstance(self.primary_sharpe_interval, StudentizedBootstrapResult):
            raise FinalInferenceError("primary Sharpe interval is missing")
        if (
            not isinstance(self.paired_sharpe_intervals, Mapping)
            or frozenset(self.paired_sharpe_intervals)
            != frozenset(FROZEN_ACTIVE_ROMANO_WOLF_FAMILY)
            or any(
                not isinstance(value, StudentizedBootstrapResult)
                for value in self.paired_sharpe_intervals.values()
            )
        ):
            raise FinalInferenceError("paired interval family differs")
        intervals = (
            self.primary_sharpe_interval,
            *tuple(self.paired_sharpe_intervals.values()),
        )
        if any(
            interval.outer_replicates != STUDENTIZED_OUTER_REPLICATES
            or interval.inner_replicates_per_standard_error
            != STUDENTIZED_INNER_REPLICATES
            or interval.expected_block_length != STATIONARY_BOOTSTRAP_EXPECTED_BLOCK
            or interval.master_seed != STATIONARY_BOOTSTRAP_MASTER_SEED
            or interval.confidence_level != FROZEN_CONFIDENCE_LEVEL
            for interval in intervals
        ):
            raise FinalInferenceError("a studentized interval differs from production")
        if self.primary_sharpe_interval.statistic_id != "hybrid_primary_sharpe_level":
            raise FinalInferenceError("primary Sharpe statistic identity differs")
        if any(
            self.paired_sharpe_intervals[comparison_id].statistic_id != comparison_id
            for comparison_id in FROZEN_ACTIVE_ROMANO_WOLF_FAMILY
        ):
            raise FinalInferenceError("a paired Sharpe statistic identity differs")
        if not isinstance(self.hac_result, FF5UMDHACResult):
            raise FinalInferenceError("HAC result is missing")
        if (
            self.hac_result.observations != FROZEN_PRIMARY_OBSERVATIONS
            or self.hac_result.lags != NEWEY_WEST_LAGS
            or self.hac_result.confidence_level != FROZEN_CONFIDENCE_LEVEL
        ):
            raise FinalInferenceError("HAC result differs from production")
        if not isinstance(
            self.automatic_bandwidth_hac_robustness,
            AutomaticBandwidthHACRobustnessResult,
        ):
            raise FinalInferenceError("automatic-bandwidth HAC robustness is missing")
        automatic = self.automatic_bandwidth_hac_robustness
        if (
            automatic.label != SECONDARY_AUTOMATIC_HAC_LABEL
            or automatic.bandwidth_rule_id != AUTOMATIC_HAC_BANDWIDTH_RULE_ID
            or automatic.regression.observations != FROZEN_PRIMARY_OBSERVATIONS
            or automatic.selected_lags
            != automatic_newey_west_lags(FROZEN_PRIMARY_OBSERVATIONS)
            or automatic.regression.confidence_level != FROZEN_CONFIDENCE_LEVEL
        ):
            raise FinalInferenceError(
                "automatic-bandwidth HAC robustness differs from production"
            )
        secondary_intervals = (
            self.secondary_alpha_percentile_interval,
            self.secondary_maximum_drawdown_percentile_interval,
        )
        if any(
            not isinstance(interval, SecondaryPercentileBootstrapResult)
            for interval in secondary_intervals
        ):
            raise FinalInferenceError("a secondary percentile interval is missing")
        if (
            self.secondary_alpha_percentile_interval.statistic_id
            != SECONDARY_ALPHA_STATISTIC_ID
            or self.secondary_maximum_drawdown_percentile_interval.statistic_id
            != SECONDARY_MAXIMUM_DRAWDOWN_STATISTIC_ID
        ):
            raise FinalInferenceError("a secondary percentile identity differs")
        if any(
            interval.label != SECONDARY_PERCENTILE_LABEL
            or interval.studentized is not False
            or interval.outer_replicates != STUDENTIZED_OUTER_REPLICATES
            or interval.expected_block_length != STATIONARY_BOOTSTRAP_EXPECTED_BLOCK
            or interval.master_seed != STATIONARY_BOOTSTRAP_MASTER_SEED
            or interval.confidence_level != FROZEN_CONFIDENCE_LEVEL
            or interval.invalid_outer_fraction > MAX_INVALID_OUTER_FRACTION
            or interval.joint_outer_indices_sha256
            != self.joint_outer_index_receipt.indices_sha256
            for interval in secondary_intervals
        ):
            raise FinalInferenceError(
                "a secondary percentile interval differs from production"
            )
        if not math.isclose(
            self.secondary_alpha_percentile_interval.estimate,
            self.hac_result.annualized_alpha,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise FinalInferenceError(
                "secondary alpha was not recomputed from the primary joint panel"
            )
        if not isinstance(self.romano_wolf_result, RomanoWolfResult):
            raise FinalInferenceError("Romano-Wolf result is missing")
        if self.romano_wolf_result.family != ROMANO_WOLF_FAMILY:
            raise FinalInferenceError("Romano-Wolf result family differs")
        if self.romano_wolf_result.outer_replicates != STUDENTIZED_OUTER_REPLICATES:
            raise FinalInferenceError("Romano-Wolf outer replicate count differs")
        for comparison_id in FROZEN_ACTIVE_ROMANO_WOLF_FAMILY:
            if not self.romano_wolf_result.comparison(comparison_id).active:
                raise FinalInferenceError("a frozen active comparison is inactive")
        inactive = self.romano_wolf_result.comparison(
            FROZEN_INACTIVE_ROMANO_WOLF_COMPARISON
        )
        if (
            inactive.active
            or inactive.unadjusted_p_value != 1.0
            or inactive.adjusted_p_value != 1.0
        ):
            raise FinalInferenceError(
                "decision-focused comparison must remain inactive with p=1"
            )
        if not isinstance(self.deflated_sharpe_result, DeflatedSharpeResult):
            raise FinalInferenceError("Deflated Sharpe result is missing")
        if (
            self.deflated_sharpe_result.observations != FROZEN_PRIMARY_OBSERVATIONS
            or self.deflated_sharpe_result.raw_trial_count != FROZEN_RAW_TRIAL_COUNT
            or self.deflated_sharpe_result.annualization_periods
            != ANNUALIZATION_PERIODS
        ):
            raise FinalInferenceError("Deflated Sharpe result differs from production")
        object.__setattr__(self, "holding_months", months)
        object.__setattr__(
            self,
            "paired_sharpe_intervals",
            MappingProxyType(dict(self.paired_sharpe_intervals)),
        )


def _frozen_sharpe_interval(
    values: np.ndarray,
    *,
    statistic_id: str,
    outer_indices: np.ndarray,
) -> StudentizedBootstrapResult:
    sample = np.asarray(values, dtype=np.float64)[:, None]

    def statistic(matrix: np.ndarray) -> float:
        return _annualized_sharpe_or_nan(matrix[:, 0], ANNUALIZATION_PERIODS)

    return _studentized_interval(
        sample,
        statistic,
        statistic_id=statistic_id,
        confidence_level=FROZEN_CONFIDENCE_LEVEL,
        outer_replicates=STUDENTIZED_OUTER_REPLICATES,
        inner_replicates=STUDENTIZED_INNER_REPLICATES,
        expected_block_length=STATIONARY_BOOTSTRAP_EXPECTED_BLOCK,
        master_seed=STATIONARY_BOOTSTRAP_MASTER_SEED,
        max_invalid_outer_fraction=MAX_INVALID_OUTER_FRACTION,
        outer_indices=outer_indices,
    )


def _frozen_paired_sharpe_interval(
    hybrid: np.ndarray,
    comparison: np.ndarray,
    *,
    statistic_id: str,
    outer_indices: np.ndarray,
) -> StudentizedBootstrapResult:
    sample = np.column_stack((hybrid, comparison))

    def statistic(matrix: np.ndarray) -> float:
        left = _annualized_sharpe_or_nan(matrix[:, 0], ANNUALIZATION_PERIODS)
        right = _annualized_sharpe_or_nan(matrix[:, 1], ANNUALIZATION_PERIODS)
        if not math.isfinite(left) or not math.isfinite(right):
            return math.nan
        result = left - right
        return result if math.isfinite(result) else math.nan

    return _studentized_interval(
        sample,
        statistic,
        statistic_id=statistic_id,
        confidence_level=FROZEN_CONFIDENCE_LEVEL,
        outer_replicates=STUDENTIZED_OUTER_REPLICATES,
        inner_replicates=STUDENTIZED_INNER_REPLICATES,
        expected_block_length=STATIONARY_BOOTSTRAP_EXPECTED_BLOCK,
        master_seed=STATIONARY_BOOTSTRAP_MASTER_SEED,
        max_invalid_outer_fraction=MAX_INVALID_OUTER_FRACTION,
        outer_indices=outer_indices,
    )


def run_frozen_primary_inference(
    panel: FrozenPrimaryInferencePanel,
    *,
    trial_ledger_binding: FrozenTrialLedgerBinding,
    usable_trial_annualized_sharpes: Sequence[object] | np.ndarray,
) -> FrozenPrimaryInferenceResult:
    """Run the immutable Methodology-v1 confirmatory inference protocol.

    Unlike the numerical primitives above, this production entry point exposes
    no configurable dates, seeds, block lengths, replicate counts, HAC lags,
    family activity, annualization, or invalid-resample threshold.
    """

    if not isinstance(panel, FrozenPrimaryInferencePanel):
        raise FinalInferenceError("panel must be a FrozenPrimaryInferencePanel")
    if not isinstance(trial_ledger_binding, FrozenTrialLedgerBinding):
        raise FinalInferenceError(
            "trial_ledger_binding must be an authenticated frozen binding"
        )
    joint_indices, joint_receipt = _frozen_joint_outer_indices()
    primary_interval = _frozen_sharpe_interval(
        panel.hybrid_monthly_excess_returns,
        statistic_id="hybrid_primary_sharpe_level",
        outer_indices=joint_indices,
    )
    paired: dict[str, StudentizedBootstrapResult] = {}
    for comparison_id in FROZEN_ACTIVE_ROMANO_WOLF_FAMILY:
        paired[comparison_id] = _frozen_paired_sharpe_interval(
            panel.hybrid_monthly_excess_returns,
            panel.comparison_monthly_excess_returns[comparison_id],
            statistic_id=comparison_id,
            outer_indices=joint_indices,
        )
    hac = ff5_umd_hac_regression(
        panel.hybrid_monthly_excess_returns,
        panel.monthly_factor_returns,
        lags=NEWEY_WEST_LAGS,
        confidence_level=FROZEN_CONFIDENCE_LEVEL,
    )
    automatic_hac = ff5_umd_automatic_bandwidth_hac_robustness(
        panel.hybrid_monthly_excess_returns,
        panel.monthly_factor_returns,
        confidence_level=FROZEN_CONFIDENCE_LEVEL,
    )
    secondary_alpha = stationary_bootstrap_alpha_percentile_interval(
        panel.hybrid_monthly_excess_returns,
        panel.monthly_factor_returns,
        confidence_level=FROZEN_CONFIDENCE_LEVEL,
        outer_replicates=STUDENTIZED_OUTER_REPLICATES,
        expected_block_length=STATIONARY_BOOTSTRAP_EXPECTED_BLOCK,
        master_seed=STATIONARY_BOOTSTRAP_MASTER_SEED,
        max_invalid_outer_fraction=MAX_INVALID_OUTER_FRACTION,
        outer_indices=joint_indices,
    )
    secondary_maximum_drawdown = (
        stationary_bootstrap_maximum_drawdown_percentile_interval(
            panel.hybrid_monthly_total_returns,
            confidence_level=FROZEN_CONFIDENCE_LEVEL,
            outer_replicates=STUDENTIZED_OUTER_REPLICATES,
            expected_block_length=STATIONARY_BOOTSTRAP_EXPECTED_BLOCK,
            master_seed=STATIONARY_BOOTSTRAP_MASTER_SEED,
            max_invalid_outer_fraction=MAX_INVALID_OUTER_FRACTION,
            outer_indices=joint_indices,
        )
    )
    observed: dict[str, float | None] = {}
    errors: dict[str, float | None] = {}
    outer_differences: dict[str, np.ndarray | None] = {}
    outer_errors: dict[str, np.ndarray | None] = {}
    active: dict[str, bool] = {}
    for comparison_id in ROMANO_WOLF_FAMILY:
        is_active = comparison_id in FROZEN_ACTIVE_ROMANO_WOLF_FAMILY
        active[comparison_id] = is_active
        if is_active:
            interval = paired[comparison_id]
            observed[comparison_id] = interval.estimate
            errors[comparison_id] = interval.standard_error
            outer_differences[comparison_id] = interval.outer_estimates
            outer_errors[comparison_id] = interval.outer_standard_errors
        else:
            observed[comparison_id] = None
            errors[comparison_id] = None
            outer_differences[comparison_id] = None
            outer_errors[comparison_id] = None
    romano = romano_wolf_stepdown(
        observed_differences=observed,
        standard_errors=errors,
        outer_differences=outer_differences,
        outer_standard_errors=outer_errors,
        active=active,
        max_invalid_outer_fraction=MAX_INVALID_OUTER_FRACTION,
    )
    dsr = deflated_sharpe_from_returns(
        panel.hybrid_monthly_excess_returns,
        trial_sharpes=usable_trial_annualized_sharpes,
        raw_trial_count=trial_ledger_binding.raw_trial_count,
        trial_sharpes_are_annualized=True,
        annualization_periods=ANNUALIZATION_PERIODS,
    )
    return FrozenPrimaryInferenceResult(
        holding_months=panel.holding_months,
        joint_outer_index_receipt=joint_receipt,
        trial_ledger_binding=trial_ledger_binding,
        primary_sharpe_interval=primary_interval,
        paired_sharpe_intervals=paired,
        hac_result=hac,
        automatic_bandwidth_hac_robustness=automatic_hac,
        secondary_alpha_percentile_interval=secondary_alpha,
        secondary_maximum_drawdown_percentile_interval=(secondary_maximum_drawdown),
        romano_wolf_result=romano,
        deflated_sharpe_result=dsr,
    )


@dataclass(frozen=True, slots=True)
class CoreExtensionDescriptiveResult:
    """Statistics for the three complete core months; never confirmatory."""

    holding_months: tuple[str, ...]
    monthly_total_returns: np.ndarray
    monthly_risk_free_returns: np.ndarray
    monthly_excess_returns: np.ndarray
    arithmetic_annualized_excess_return: float
    net_cagr: float
    annualized_volatility: float
    sharpe: float
    formal_inference_permitted: bool = False

    def __post_init__(self) -> None:
        try:
            months = require_exact_core_extension_months(self.holding_months)
        except FinalCalendarError as exc:
            raise FinalInferenceError(str(exc)) from exc
        if self.formal_inference_permitted is not False:
            raise FinalInferenceError("core extension cannot enter formal inference")
        for field_name in (
            "monthly_total_returns",
            "monthly_risk_free_returns",
            "monthly_excess_returns",
        ):
            values = _finite_vector(
                getattr(self, field_name),
                name=field_name,
                minimum_size=len(CORE_COMPLETE_HOLDING_MONTHS),
            )
            if values.size != len(CORE_COMPLETE_HOLDING_MONTHS):
                raise FinalInferenceError(f"{field_name} must contain three rows")
            object.__setattr__(self, field_name, _readonly_float_array(values))
        object.__setattr__(self, "holding_months", months)


def describe_frozen_core_extension(
    *,
    holding_months: Sequence[str],
    monthly_total_returns: Sequence[object] | np.ndarray,
    monthly_risk_free_returns: Sequence[object] | np.ndarray,
) -> CoreExtensionDescriptiveResult:
    """Describe exactly Sep--Nov 2025 without confirmatory inference."""

    try:
        months = require_exact_core_extension_months(holding_months)
    except FinalCalendarError as exc:
        raise FinalInferenceError(str(exc)) from exc
    total = _finite_vector(
        monthly_total_returns,
        name="monthly_total_returns",
        minimum_size=len(months),
    )
    risk_free = _finite_vector(
        monthly_risk_free_returns,
        name="monthly_risk_free_returns",
        minimum_size=len(months),
    )
    if total.size != len(months) or risk_free.size != len(months):
        raise FinalInferenceError("core return vectors must contain exactly three rows")
    if np.any(total < -1.0):
        raise FinalInferenceError("core monthly total return cannot be below -100%")
    excess = total - risk_free
    monthly_volatility = float(np.std(excess, ddof=1))
    if not math.isfinite(monthly_volatility) or monthly_volatility <= 0.0:
        raise FinalInferenceError("core monthly excess returns have zero variance")
    annualized_volatility = math.sqrt(ANNUALIZATION_PERIODS) * monthly_volatility
    annualized_excess = ANNUALIZATION_PERIODS * float(np.mean(excess))
    cagr = float(np.prod(1.0 + total) ** (ANNUALIZATION_PERIODS / len(total)) - 1.0)
    sharpe = annualized_excess / annualized_volatility
    return CoreExtensionDescriptiveResult(
        holding_months=months,
        monthly_total_returns=total,
        monthly_risk_free_returns=risk_free,
        monthly_excess_returns=excess,
        arithmetic_annualized_excess_return=annualized_excess,
        net_cagr=cagr,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        formal_inference_permitted=False,
    )


@dataclass(frozen=True, slots=True)
class PartialDecemberDescriptiveResult:
    """The single unannualized, noninferential partial-December return."""

    holding_month: str
    partial_total_return: float
    annualized_statistics_permitted: bool = False
    formal_inference_permitted: bool = False

    def __post_init__(self) -> None:
        try:
            require_exact_partial_december_month((self.holding_month,))
        except FinalCalendarError as exc:
            raise FinalInferenceError(str(exc)) from exc
        try:
            value = float(self.partial_total_return)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FinalInferenceError("partial_total_return must be numeric") from exc
        if not math.isfinite(value) or value < -1.0:
            raise FinalInferenceError(
                "partial_total_return must be finite and no less than -100%"
            )
        if self.annualized_statistics_permitted is not False:
            raise FinalInferenceError(
                "partial December cannot enter annualized statistics"
            )
        if self.formal_inference_permitted is not False:
            raise FinalInferenceError("partial December cannot enter formal inference")
        object.__setattr__(self, "partial_total_return", value)


def describe_frozen_partial_december(
    *,
    holding_months: Sequence[str],
    partial_total_return: object,
) -> PartialDecemberDescriptiveResult:
    """Return only the exact partial-December 2025 descriptive observation."""

    try:
        months = require_exact_partial_december_month(holding_months)
    except FinalCalendarError as exc:
        raise FinalInferenceError(str(exc)) from exc
    return PartialDecemberDescriptiveResult(
        holding_month=months[0],
        partial_total_return=partial_total_return,  # type: ignore[arg-type]
        annualized_statistics_permitted=False,
        formal_inference_permitted=False,
    )


__all__ = [
    "ANNUALIZATION_PERIODS",
    "AUTOMATIC_HAC_BANDWIDTH_RULE_ID",
    "AutomaticBandwidthHACRobustnessResult",
    "BootstrapInferenceFailure",
    "CoreExtensionDescriptiveResult",
    "DeflatedSharpeResult",
    "EULER_MASCHERONI",
    "FF5UMDHACResult",
    "FF5_UMD_FACTOR_NAMES",
    "FROZEN_ACTIVE_ROMANO_WOLF_FAMILY",
    "FROZEN_CONFIDENCE_LEVEL",
    "FROZEN_INACTIVE_ROMANO_WOLF_COMPARISON",
    "FROZEN_JOINT_OUTER_INDICES_SHA256",
    "FROZEN_PRIMARY_OBSERVATIONS",
    "FROZEN_RAW_TRIAL_COUNT",
    "FROZEN_TRIAL_LEDGER_ID",
    "FinalInferenceError",
    "FrozenJointOuterIndexReceipt",
    "FrozenPrimaryInferencePanel",
    "FrozenPrimaryInferenceResult",
    "FrozenTrialLedgerBinding",
    "HACCoefficient",
    "MAX_INVALID_OUTER_FRACTION",
    "NEWEY_WEST_LAGS",
    "PartialDecemberDescriptiveResult",
    "ROMANO_WOLF_FAMILY",
    "RomanoWolfComparison",
    "RomanoWolfResult",
    "SECONDARY_ALPHA_STATISTIC_ID",
    "SECONDARY_AUTOMATIC_HAC_LABEL",
    "SECONDARY_MAXIMUM_DRAWDOWN_STATISTIC_ID",
    "SECONDARY_PERCENTILE_LABEL",
    "STATIONARY_BOOTSTRAP_EXPECTED_BLOCK",
    "STATIONARY_BOOTSTRAP_MASTER_SEED",
    "STUDENTIZED_INNER_REPLICATES",
    "STUDENTIZED_OUTER_REPLICATES",
    "SecondaryPercentileBootstrapResult",
    "StudentizedBootstrapResult",
    "annualized_sharpe_ratio",
    "automatic_newey_west_lags",
    "deflated_sharpe_from_returns",
    "deflated_sharpe_ratio",
    "describe_frozen_core_extension",
    "describe_frozen_partial_december",
    "expected_maximum_sharpe",
    "ff5_umd_annualized_alpha",
    "ff5_umd_automatic_bandwidth_hac_robustness",
    "ff5_umd_hac_regression",
    "frozen_joint_outer_index_receipt",
    "maximum_drawdown",
    "paired_sharpe_difference",
    "romano_wolf_stepdown",
    "run_frozen_primary_inference",
    "stationary_bootstrap_indices",
    "stationary_bootstrap_alpha_percentile_interval",
    "stationary_bootstrap_maximum_drawdown_percentile_interval",
    "studentized_paired_sharpe_interval",
    "studentized_sharpe_interval",
]
