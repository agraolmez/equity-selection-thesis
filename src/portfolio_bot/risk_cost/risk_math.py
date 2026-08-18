"""Deterministic numerical primitives for the frozen Stage 8 risk model.

The functions in this module are intentionally independent of data access and
publication code.  Callers must first select observations that were available
at the signal cutoff.  A missing or malformed numerical input fails closed
instead of being silently imputed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


FACTOR_NAMES = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
FACTOR_COUNT = len(FACTOR_NAMES)

RISK_LOOKBACK_SESSIONS = 252
RISK_MINIMUM_OBSERVATIONS = 126
RISK_HALF_LIFE_SESSIONS = 63.0
RISK_EWLS_RCOND = 1e-12
RISK_EWLS_REQUIRED_RANK = FACTOR_COUNT + 1

SPECIFIC_VARIANCE_PRIOR_OBSERVATIONS = 60.0
SECTOR_MEDIAN_MINIMUM_SECURITIES = 10
RIDGE_ABSOLUTE_FLOOR_DAILY = 1e-8
RIDGE_DIAGONAL_MULTIPLIER = 1e-6
MONTHLY_SESSION_SCALE = 21.0


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def type7_quantile(
    values: Sequence[float | None],
    probability: float,
) -> float | None:
    """Return the Hyndman--Fan type-7 quantile of the finite observations.

    Missing and nonfinite observations are not members of the estimable
    population.  An invalid probability is a programming error and raises.
    """

    if not _finite(probability) or not 0.0 <= float(probability) <= 1.0:
        raise ValueError("Type-7 probability must be finite and within [0, 1]")
    ordered = sorted(
        float(value) for value in values if value is not None and _finite(value)
    )
    if not ordered:
        return None
    if len(ordered) == 1:
        return _positive_zero(ordered[0])
    position = (len(ordered) - 1) * float(probability)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return _positive_zero(ordered[lower_index])
    fraction = position - lower_index
    result = math.fsum(
        (
            ordered[lower_index],
            fraction * (ordered[upper_index] - ordered[lower_index]),
        )
    )
    return _positive_zero(result) if math.isfinite(result) else None


def _valid_estimation_inputs(
    ages: Sequence[int | float],
    observation_count: int,
    *,
    minimum_observations: int,
    half_life_sessions: float,
) -> np.ndarray | None:
    if (
        isinstance(minimum_observations, bool)
        or not isinstance(minimum_observations, int)
        or minimum_observations <= 0
        or minimum_observations > RISK_LOOKBACK_SESSIONS
        or not _finite(half_life_sessions)
        or float(half_life_sessions) <= 0.0
        or observation_count < minimum_observations
        or observation_count > RISK_LOOKBACK_SESSIONS
    ):
        return None
    try:
        age_vector = np.asarray(ages, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        age_vector.ndim != 1
        or age_vector.size != observation_count
        or not np.isfinite(age_vector).all()
        or np.any(age_vector < 0.0)
        or np.any(age_vector > RISK_LOOKBACK_SESSIONS - 1)
        or np.any(age_vector != np.floor(age_vector))
        or np.unique(age_vector).size != age_vector.size
    ):
        return None
    return age_vector


@dataclass(frozen=True)
class RiskEwlsResult:
    """One security's daily six-factor EWLS risk estimate."""

    intercept: float
    betas: tuple[float, ...]
    residual_variance_daily: float
    observation_count: int
    weight_sum: float


def fit_security_risk_ewls(
    security_returns: Sequence[float],
    risk_free_rates: Sequence[float],
    factors: Sequence[Sequence[float]] | np.ndarray,
    ages: Sequence[int | float],
    *,
    half_life_sessions: float = RISK_HALF_LIFE_SESSIONS,
    minimum_observations: int = RISK_MINIMUM_OBSERVATIONS,
    rcond: float = RISK_EWLS_RCOND,
) -> RiskEwlsResult | None:
    """Fit excess security returns on an intercept and the six factors.

    ``ages`` are global-session ages, so an observation on the signal session
    has age zero even if the security has gaps elsewhere in its return history.
    Slopes are returned exactly as estimated; Stage 8 does not beta-clip.
    """

    if not _finite(rcond) or float(rcond) < 0.0:
        return None
    try:
        returns = np.asarray(security_returns, dtype=np.float64)
        rf = np.asarray(risk_free_rates, dtype=np.float64)
        factor_matrix = np.asarray(factors, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        returns.ndim != 1
        or rf.ndim != 1
        or factor_matrix.ndim != 2
        or factor_matrix.shape != (returns.size, FACTOR_COUNT)
        or rf.size != returns.size
        or not np.isfinite(returns).all()
        or not np.isfinite(rf).all()
        or not np.isfinite(factor_matrix).all()
    ):
        return None
    age_vector = _valid_estimation_inputs(
        ages,
        int(returns.size),
        minimum_observations=minimum_observations,
        half_life_sessions=half_life_sessions,
    )
    if age_vector is None:
        return None

    dependent = returns - rf
    design = np.column_stack(
        (np.ones(returns.size, dtype=np.float64), factor_matrix)
    )
    weights = np.exp2(-age_vector / float(half_life_sessions))
    roots = np.sqrt(weights)
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(
            design * roots[:, None],
            dependent * roots,
            rcond=float(rcond),
        )
    except np.linalg.LinAlgError:
        return None
    if rank != RISK_EWLS_REQUIRED_RANK or not np.isfinite(coefficients).all():
        return None

    residual = dependent - design @ coefficients
    weight_sum = float(np.sum(weights, dtype=np.float64))
    residual_variance = float(
        np.sum(weights * residual * residual, dtype=np.float64) / weight_sum
    )
    if (
        not math.isfinite(weight_sum)
        or weight_sum <= 0.0
        or not math.isfinite(residual_variance)
        or residual_variance < 0.0
    ):
        return None
    return RiskEwlsResult(
        intercept=_positive_zero(float(coefficients[0])),
        betas=tuple(_positive_zero(float(value)) for value in coefficients[1:]),
        residual_variance_daily=_positive_zero(residual_variance),
        observation_count=int(returns.size),
        weight_sum=weight_sum,
    )


@dataclass(frozen=True)
class FactorCovarianceResult:
    """Exponentially weighted daily factor moments."""

    factor_means_daily: tuple[float, ...]
    covariance_daily: tuple[tuple[float, ...], ...]
    observation_count: int
    weight_sum: float


def estimate_factor_covariance(
    factors: Sequence[Sequence[float]] | np.ndarray,
    ages: Sequence[int | float],
    *,
    half_life_sessions: float = RISK_HALF_LIFE_SESSIONS,
    minimum_observations: int = RISK_MINIMUM_OBSERVATIONS,
) -> FactorCovarianceResult | None:
    """Estimate the frozen EW population covariance of the six factors."""

    try:
        factor_matrix = np.asarray(factors, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        factor_matrix.ndim != 2
        or factor_matrix.shape[1:] != (FACTOR_COUNT,)
        or not np.isfinite(factor_matrix).all()
    ):
        return None
    age_vector = _valid_estimation_inputs(
        ages,
        int(factor_matrix.shape[0]),
        minimum_observations=minimum_observations,
        half_life_sessions=half_life_sessions,
    )
    if age_vector is None:
        return None

    weights = np.exp2(-age_vector / float(half_life_sessions))
    weight_sum = float(np.sum(weights, dtype=np.float64))
    means = np.sum(factor_matrix * weights[:, None], axis=0) / weight_sum
    centered = factor_matrix - means
    covariance = (centered.T @ (centered * weights[:, None])) / weight_sum
    covariance = (covariance + covariance.T) * 0.5
    if (
        not math.isfinite(weight_sum)
        or weight_sum <= 0.0
        or not np.isfinite(means).all()
        or not np.isfinite(covariance).all()
    ):
        return None
    try:
        minimum_eigenvalue = float(np.linalg.eigvalsh(covariance)[0])
    except np.linalg.LinAlgError:
        return None
    if minimum_eigenvalue < -1e-12:
        return None
    return FactorCovarianceResult(
        factor_means_daily=tuple(_positive_zero(float(value)) for value in means),
        covariance_daily=tuple(
            tuple(_positive_zero(float(value)) for value in row)
            for row in covariance
        ),
        observation_count=int(factor_matrix.shape[0]),
        weight_sum=weight_sum,
    )


def shrink_residual_variance(
    raw_variance_daily: float,
    observation_count: int,
    reference_median_daily: float,
    *,
    prior_observations: float = SPECIFIC_VARIANCE_PRIOR_OBSERVATIONS,
) -> float | None:
    """Shrink one raw residual variance toward its contemporaneous median."""

    if (
        isinstance(observation_count, bool)
        or not isinstance(observation_count, (int, np.integer))
        or int(observation_count) <= 0
        or not _finite(raw_variance_daily)
        or float(raw_variance_daily) < 0.0
        or not _finite(reference_median_daily)
        or float(reference_median_daily) < 0.0
        or not _finite(prior_observations)
        or float(prior_observations) <= 0.0
    ):
        return None
    count = float(observation_count)
    prior = float(prior_observations)
    result = (
        count / (count + prior) * float(raw_variance_daily)
        + prior / (count + prior) * float(reference_median_daily)
    )
    return _positive_zero(result) if math.isfinite(result) and result >= 0.0 else None


def _normalized_sector(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.upper() == "UNKNOWN":
        return None
    return normalized


def _valid_raw_variance(
    value: object,
    observation_count: object,
    *,
    minimum_observations: int,
) -> bool:
    return (
        _finite(value)
        and float(value) >= 0.0
        and not isinstance(observation_count, bool)
        and isinstance(observation_count, (int, np.integer))
        and int(observation_count) >= minimum_observations
        and int(observation_count) <= RISK_LOOKBACK_SESSIONS
    )


@dataclass(frozen=True)
class ShrunkResidualVariance:
    """One cross-sectionally shrunk daily specific-variance estimate."""

    raw_variance_daily: float
    reference_median_daily: float
    shrunk_variance_daily: float
    observation_count: int
    reference_scope: str
    reference_security_count: int


def shrink_residual_variances(
    raw_variances_daily: Sequence[float | None],
    observation_counts: Sequence[int | None],
    sectors: Sequence[str | None],
    scoring_rows: Sequence[bool],
    *,
    minimum_observations: int = RISK_MINIMUM_OBSERVATIONS,
    sector_minimum_securities: int = SECTOR_MEDIAN_MINIMUM_SECURITIES,
    prior_observations: float = SPECIFIC_VARIANCE_PRIOR_OBSERVATIONS,
) -> tuple[ShrunkResidualVariance | None, ...] | None:
    """Apply the frozen sector/cross-sectional residual-variance shrinkage.

    Median populations contain only valid rows in the contemporaneous scoring
    universe.  Known sectors use their sector median only when at least ten
    valid scoring securities are available; all other cases use the full
    scoring-universe median.
    """

    row_count = len(raw_variances_daily)
    if (
        len(observation_counts) != row_count
        or len(sectors) != row_count
        or len(scoring_rows) != row_count
        or isinstance(minimum_observations, bool)
        or not isinstance(minimum_observations, int)
        or minimum_observations <= 0
        or minimum_observations > RISK_LOOKBACK_SESSIONS
        or isinstance(sector_minimum_securities, bool)
        or not isinstance(sector_minimum_securities, int)
        or sector_minimum_securities <= 0
        or any(
            not isinstance(value, (bool, np.bool_)) for value in scoring_rows
        )
    ):
        return None

    normalized_sectors = tuple(_normalized_sector(value) for value in sectors)
    valid = tuple(
        _valid_raw_variance(
            value,
            count,
            minimum_observations=minimum_observations,
        )
        for value, count in zip(
            raw_variances_daily, observation_counts, strict=True
        )
    )
    population_indices = tuple(
        index
        for index in range(row_count)
        if bool(scoring_rows[index]) and valid[index]
    )
    cross_values = [float(raw_variances_daily[index]) for index in population_indices]
    cross_median = type7_quantile(cross_values, 0.5)
    if cross_median is None:
        return tuple(None for _ in range(row_count))

    sector_values: dict[str, list[float]] = {}
    for index in population_indices:
        sector = normalized_sectors[index]
        if sector is not None:
            sector_values.setdefault(sector, []).append(
                float(raw_variances_daily[index])
            )

    results: list[ShrunkResidualVariance | None] = []
    for index in range(row_count):
        if not valid[index]:
            results.append(None)
            continue
        sector = normalized_sectors[index]
        values = sector_values.get(sector, []) if sector is not None else []
        if sector is not None and len(values) >= sector_minimum_securities:
            reference = type7_quantile(values, 0.5)
            reference_scope = "sector"
            reference_count = len(values)
        else:
            reference = cross_median
            reference_scope = "cross_section"
            reference_count = len(cross_values)
        if reference is None:
            results.append(None)
            continue
        shrunk = shrink_residual_variance(
            float(raw_variances_daily[index]),
            int(observation_counts[index]),
            reference,
            prior_observations=prior_observations,
        )
        if shrunk is None:
            results.append(None)
            continue
        results.append(
            ShrunkResidualVariance(
                raw_variance_daily=_positive_zero(
                    float(raw_variances_daily[index])
                ),
                reference_median_daily=_positive_zero(reference),
                shrunk_variance_daily=shrunk,
                observation_count=int(observation_counts[index]),
                reference_scope=reference_scope,
                reference_security_count=reference_count,
            )
        )
    return tuple(results)


def _validated_factor_model_inputs(
    betas: Sequence[Sequence[float]] | np.ndarray,
    factor_covariance_daily: Sequence[Sequence[float]] | np.ndarray,
    specific_variances_daily: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        beta_matrix = np.asarray(betas, dtype=np.float64)
        factor_covariance = np.asarray(factor_covariance_daily, dtype=np.float64)
        specific = np.asarray(specific_variances_daily, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        beta_matrix.ndim != 2
        or beta_matrix.shape[1:] != (FACTOR_COUNT,)
        or beta_matrix.shape[0] == 0
        or factor_covariance.shape != (FACTOR_COUNT, FACTOR_COUNT)
        or specific.shape != (beta_matrix.shape[0],)
        or not np.isfinite(beta_matrix).all()
        or not np.isfinite(factor_covariance).all()
        or not np.isfinite(specific).all()
        or np.any(specific < 0.0)
        or not np.allclose(
            factor_covariance,
            factor_covariance.T,
            rtol=0.0,
            atol=1e-14,
        )
    ):
        return None
    factor_covariance = (factor_covariance + factor_covariance.T) * 0.5
    try:
        minimum_eigenvalue = float(np.linalg.eigvalsh(factor_covariance)[0])
    except np.linalg.LinAlgError:
        return None
    if minimum_eigenvalue < -1e-12:
        return None
    return beta_matrix, factor_covariance, specific


@dataclass(frozen=True)
class RiskCovarianceRegularization:
    """Factor-model covariance components before and after frozen ridge."""

    factor_covariance_daily: tuple[tuple[float, ...], ...]
    factor_covariance_monthly: tuple[tuple[float, ...], ...]
    specific_variances_daily_pre_ridge: tuple[float, ...]
    specific_variances_daily_post_ridge: tuple[float, ...]
    specific_variances_monthly_post_ridge: tuple[float, ...]
    security_diagonal_daily_pre_ridge: tuple[float, ...]
    security_diagonal_daily_post_ridge: tuple[float, ...]
    security_diagonal_monthly_post_ridge: tuple[float, ...]
    ridge_daily: float
    ridge_monthly: float


def regularize_factor_model_covariance(
    betas: Sequence[Sequence[float]] | np.ndarray,
    factor_covariance_daily: Sequence[Sequence[float]] | np.ndarray,
    specific_variances_daily: Sequence[float],
    *,
    ridge_floor_daily: float = RIDGE_ABSOLUTE_FLOOR_DAILY,
    ridge_diagonal_multiplier: float = RIDGE_DIAGONAL_MULTIPLIER,
    monthly_session_scale: float = MONTHLY_SESSION_SCALE,
) -> RiskCovarianceRegularization | None:
    """Compute the frozen diagonal ridge and monthly factor representation."""

    validated = _validated_factor_model_inputs(
        betas,
        factor_covariance_daily,
        specific_variances_daily,
    )
    if (
        validated is None
        or not _finite(ridge_floor_daily)
        or float(ridge_floor_daily) < 0.0
        or not _finite(ridge_diagonal_multiplier)
        or float(ridge_diagonal_multiplier) < 0.0
        or not _finite(monthly_session_scale)
        or float(monthly_session_scale) <= 0.0
    ):
        return None
    beta_matrix, factor_covariance, specific = validated
    systematic_diagonal = np.einsum(
        "ij,jk,ik->i", beta_matrix, factor_covariance, beta_matrix
    )
    pre_ridge_diagonal = systematic_diagonal + specific
    if not np.isfinite(pre_ridge_diagonal).all() or np.any(pre_ridge_diagonal < 0.0):
        return None
    median_diagonal = type7_quantile(
        tuple(float(value) for value in pre_ridge_diagonal), 0.5
    )
    if median_diagonal is None or median_diagonal < 0.0:
        return None
    ridge = max(
        float(ridge_floor_daily),
        float(ridge_diagonal_multiplier) * median_diagonal,
    )
    post_specific = specific + ridge
    post_diagonal = pre_ridge_diagonal + ridge
    monthly_scale = float(monthly_session_scale)
    factor_monthly = factor_covariance * monthly_scale

    def vector_tuple(vector: np.ndarray) -> tuple[float, ...]:
        return tuple(_positive_zero(float(value)) for value in vector)

    def matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(_positive_zero(float(value)) for value in row) for row in matrix
        )

    return RiskCovarianceRegularization(
        factor_covariance_daily=matrix_tuple(factor_covariance),
        factor_covariance_monthly=matrix_tuple(factor_monthly),
        specific_variances_daily_pre_ridge=vector_tuple(specific),
        specific_variances_daily_post_ridge=vector_tuple(post_specific),
        specific_variances_monthly_post_ridge=vector_tuple(
            post_specific * monthly_scale
        ),
        security_diagonal_daily_pre_ridge=vector_tuple(pre_ridge_diagonal),
        security_diagonal_daily_post_ridge=vector_tuple(post_diagonal),
        security_diagonal_monthly_post_ridge=vector_tuple(
            post_diagonal * monthly_scale
        ),
        ridge_daily=_positive_zero(ridge),
        ridge_monthly=_positive_zero(ridge * monthly_scale),
    )


def materialize_security_covariance(
    betas: Sequence[Sequence[float]] | np.ndarray,
    factor_covariance_daily: Sequence[Sequence[float]] | np.ndarray,
    specific_variances_daily: Sequence[float],
    *,
    ridge_daily: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray | None:
    """Materialize a small factor-model covariance matrix for checks/tests.

    Production stores should retain ``B``, factor covariance, and the specific
    diagonal instead of writing a dense security-by-security matrix.
    """

    validated = _validated_factor_model_inputs(
        betas,
        factor_covariance_daily,
        specific_variances_daily,
    )
    if (
        validated is None
        or not _finite(ridge_daily)
        or float(ridge_daily) < 0.0
        or not _finite(scale)
        or float(scale) <= 0.0
    ):
        return None
    beta_matrix, factor_covariance, specific = validated
    covariance = beta_matrix @ factor_covariance @ beta_matrix.T
    covariance += np.diag(specific + float(ridge_daily))
    covariance = (covariance + covariance.T) * (0.5 * float(scale))
    if not np.isfinite(covariance).all():
        return None
    try:
        minimum_eigenvalue = float(np.linalg.eigvalsh(covariance)[0])
    except np.linalg.LinAlgError:
        return None
    if minimum_eigenvalue < -1e-12:
        return None
    return covariance
