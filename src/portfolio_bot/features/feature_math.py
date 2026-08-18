"""Deterministic numerical primitives for roadmap Stage 6 features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _finite_vector(values: Iterable[float]) -> np.ndarray | None:
    result = np.asarray(list(values), dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        return None
    return result


def type7_quantile(values: Iterable[float], quantile: float) -> float | None:
    vector = _finite_vector(values)
    if vector is None or vector.size == 0 or not 0.0 <= quantile <= 1.0:
        return None
    result = float(np.quantile(vector, quantile, method="linear"))
    return result if math.isfinite(result) else None


def ordinary_median(values: Iterable[float]) -> float | None:
    vector = _finite_vector(values)
    if vector is None or vector.size == 0:
        return None
    result = float(np.median(vector))
    return result if math.isfinite(result) else None


def compounded_return(values: Iterable[float]) -> float | None:
    vector = _finite_vector(values)
    if vector is None or vector.size == 0 or np.any(vector < -1.0):
        return None
    if np.any(vector == -1.0):
        return -1.0
    result = float(np.expm1(np.log1p(vector).sum(dtype=np.float64)))
    return result if math.isfinite(result) else None


def price_to_moving_average(closes: Iterable[float]) -> float | None:
    vector = _finite_vector(closes)
    if vector is None or vector.size == 0 or np.any(vector <= 0.0):
        return None
    denominator = float(vector.mean(dtype=np.float64))
    if denominator <= 0.0:
        return None
    result = float(vector[-1] / denominator - 1.0)
    return result if math.isfinite(result) else None


def distance_to_high(closes: Iterable[float]) -> float | None:
    vector = _finite_vector(closes)
    if vector is None or vector.size == 0 or np.any(vector <= 0.0):
        return None
    high = float(vector.max())
    result = float(vector[-1] / high - 1.0)
    return result if math.isfinite(result) else None


def log_price_trend(closes: Iterable[float]) -> tuple[float | None, float | None]:
    vector = _finite_vector(closes)
    if vector is None or vector.size < 2 or np.any(vector <= 0.0):
        return None, None
    y = np.log(vector)
    x = np.arange(vector.size, dtype=np.float64)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(x_centered @ x_centered)
    total_sum_squares = float(y_centered @ y_centered)
    if denominator <= 0.0 or total_sum_squares <= 0.0:
        return None, None
    slope = float((x_centered @ y_centered) / denominator)
    residual = y_centered - slope * x_centered
    r_squared = float(1.0 - (residual @ residual) / total_sum_squares)
    if not math.isfinite(slope) or not math.isfinite(r_squared):
        return None, None
    return slope, min(1.0, max(0.0, r_squared))


def maximum_drawdown(returns: Iterable[float]) -> float | None:
    vector = _finite_vector(returns)
    if vector is None or vector.size == 0 or np.any(vector < -1.0):
        return None
    wealth = 1.0
    running_max = 1.0
    minimum = 0.0
    for value in vector:
        wealth *= 1.0 + float(value)
        if not math.isfinite(wealth):
            return None
        running_max = max(running_max, wealth)
        minimum = min(minimum, wealth / running_max - 1.0)
    return float(minimum)


def realized_volatility(returns: Iterable[float]) -> float | None:
    vector = _finite_vector(returns)
    if vector is None or vector.size < 2:
        return None
    result = float(math.sqrt(252.0) * np.std(vector, ddof=1))
    return result if math.isfinite(result) else None


def downside_semivolatility(returns: Iterable[float]) -> float | None:
    vector = _finite_vector(returns)
    if vector is None or vector.size == 0:
        return None
    downside = np.minimum(vector, 0.0)
    result = float(math.sqrt(252.0 * np.mean(downside * downside)))
    return result if math.isfinite(result) else None


def sample_skewness(returns: Iterable[float]) -> float | None:
    vector = _finite_vector(returns)
    if vector is None or vector.size < 3:
        return None
    centered = vector - vector.mean()
    m2 = float(np.mean(centered**2))
    if m2 <= 0.0:
        return None
    m3 = float(np.mean(centered**3))
    n = float(vector.size)
    result = math.sqrt(n * (n - 1.0)) / (n - 2.0) * m3 / (m2**1.5)
    return float(result) if math.isfinite(result) else None


def sample_excess_kurtosis(returns: Iterable[float]) -> float | None:
    vector = _finite_vector(returns)
    if vector is None or vector.size < 4:
        return None
    centered = vector - vector.mean()
    m2 = float(np.mean(centered**2))
    if m2 <= 0.0:
        return None
    m4 = float(np.mean(centered**4))
    biased_excess = m4 / (m2 * m2) - 3.0
    n = float(vector.size)
    result = (n - 1.0) / ((n - 2.0) * (n - 3.0)) * (
        (n + 1.0) * biased_excess + 6.0
    )
    return float(result) if math.isfinite(result) else None


@dataclass(frozen=True)
class EwlsResult:
    intercept: float
    slopes: tuple[float, ...]
    annualized_idiosyncratic_volatility: float


def ewls_factor_fit(
    security_returns: Iterable[float],
    risk_free_rates: Iterable[float],
    factors: np.ndarray,
    ages: Iterable[int],
    *,
    half_life_sessions: float = 126.0,
    minimum_observations: int = 126,
) -> EwlsResult | None:
    returns = _finite_vector(security_returns)
    rf = _finite_vector(risk_free_rates)
    age = np.asarray(list(ages), dtype=np.float64)
    design_factors = np.asarray(factors, dtype=np.float64)
    if (
        returns is None
        or rf is None
        or returns.size != rf.size
        or age.ndim != 1
        or age.size != returns.size
        or design_factors.ndim != 2
        or design_factors.shape[0] != returns.size
        or returns.size < minimum_observations
        or half_life_sessions <= 0.0
        or not np.isfinite(age).all()
        or not np.isfinite(design_factors).all()
        or np.any(age < 0.0)
    ):
        return None
    x = np.column_stack([np.ones(returns.size, dtype=np.float64), design_factors])
    y = returns - rf
    weights = np.exp2(-age / half_life_sessions)
    root = np.sqrt(weights)
    weighted_x = x * root[:, None]
    weighted_y = y * root
    coefficients, _, rank, _ = np.linalg.lstsq(weighted_x, weighted_y, rcond=1e-12)
    if rank != x.shape[1] or not np.isfinite(coefficients).all():
        return None
    residual = y - x @ coefficients
    variance = float(np.sum(weights * residual * residual) / np.sum(weights))
    if variance < 0.0 or not math.isfinite(variance):
        return None
    idiosyncratic = math.sqrt(252.0 * variance)
    if not math.isfinite(idiosyncratic):
        return None
    return EwlsResult(
        intercept=float(coefficients[0]),
        slopes=tuple(float(value) for value in coefficients[1:]),
        annualized_idiosyncratic_volatility=float(idiosyncratic),
    )


def corwin_schultz_pair(
    prior_high: float,
    prior_low: float,
    prior_close: float,
    next_high: float,
    next_low: float,
) -> float | None:
    values = np.asarray(
        [prior_high, prior_low, prior_close, next_high, next_low], dtype=np.float64
    )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        return None
    adjusted_high = float(next_high)
    adjusted_low = float(next_low)
    if next_low > prior_close:
        gap = next_low - prior_close
        adjusted_high -= gap
        adjusted_low -= gap
    elif next_high < prior_close:
        gap = prior_close - next_high
        adjusted_high += gap
        adjusted_low += gap
    if prior_high < prior_low or adjusted_high < adjusted_low:
        return None
    beta = math.log(prior_high / prior_low) ** 2 + math.log(
        adjusted_high / adjusted_low
    ) ** 2
    gamma = math.log(
        max(prior_high, adjusted_high) / min(prior_low, adjusted_low)
    ) ** 2
    denominator = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = (
        (math.sqrt(2.0 * beta) - math.sqrt(beta)) / denominator
        - math.sqrt(gamma / denominator)
    )
    exponent = math.exp(alpha)
    spread = 2.0 * (exponent - 1.0) / (1.0 + exponent)
    result = max(0.0, spread)
    return float(result) if math.isfinite(result) else None


def centered_average_ranks(values: Iterable[float | None]) -> np.ndarray:
    vector = np.asarray(
        [np.nan if value is None else float(value) for value in values], dtype=np.float64
    )
    result = np.full(vector.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(vector))
    if finite_indices.size == 0:
        return result
    order = finite_indices[np.argsort(vector[finite_indices], kind="mergesort")]
    position = 0
    while position < order.size:
        end = position + 1
        while end < order.size and vector[order[end]] == vector[order[position]]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        result[order[position:end]] = (
            (average_rank - 0.5) / finite_indices.size - 0.5
        )
        position = end
    return result


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if (
        not math.isfinite(numerator_value)
        or not math.isfinite(denominator_value)
        or denominator_value == 0.0
    ):
        return None
    result = numerator_value / denominator_value
    return float(result) if math.isfinite(result) else None


def signed_growth(current: float | None, prior: float | None, *, positive: bool) -> float | None:
    if current is None or prior is None:
        return None
    current_value = float(current)
    prior_value = float(prior)
    if (
        not math.isfinite(current_value)
        or not math.isfinite(prior_value)
        or prior_value == 0.0
        or (positive and (current_value <= 0.0 or prior_value <= 0.0))
    ):
        return None
    result = (current_value - prior_value) / abs(prior_value)
    return float(result) if math.isfinite(result) else None
