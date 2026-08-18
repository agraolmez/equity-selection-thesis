"""Risk primitives shared by the three frozen Stage 10 allocators.

The functions are deliberately independent of CVXPY and portfolio state.  The
optimizer consumes the returned square-root matrix, while independent
verification recomputes the same economic quantities from the primitive
factor inputs.  All inputs and outputs use monthly decimal-return units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FACTOR_ORDER = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
ANNUAL_PERIODS = 12.0
ANNUAL_VARIANCE_BUDGET = 0.10**2
MARKET_BETA_VARIANCE_INCREMENT = 4.0
PSD_EIGENVALUE_FLOOR = -1e-10


@dataclass(frozen=True)
class RiskDiagnostics:
    """Independently recomputed risk and factor exposure diagnostics."""

    monthly_variance: float
    augmented_monthly_variance: float
    augmented_annual_variance: float
    augmented_annual_volatility: float
    factor_exposures: np.ndarray


def _finite_array(value: object, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def covariance_square_root(
    covariance: object,
    *,
    minimum_eigenvalue: float = PSD_EIGENVALUE_FLOOR,
) -> np.ndarray:
    """Return ``C`` such that ``C @ C.T`` reproduces a symmetric PSD matrix.

    Tiny negative eigenvalues caused by floating-point serialization are
    clipped to zero.  A materially negative eigenvalue fails closed.
    """

    matrix = _finite_array(covariance, ndim=2, name="factor covariance")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("factor covariance must be square")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("factor covariance must be symmetric")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(np.min(eigenvalues)) < float(minimum_eigenvalue):
        raise ValueError("factor covariance is not positive semidefinite")
    clipped = np.maximum(eigenvalues, 0.0)
    return eigenvectors @ np.diag(np.sqrt(clipped))


def build_augmented_risk_root(
    factor_betas: object,
    factor_covariance_monthly: object,
    specific_variance_monthly: object,
    *,
    market_variance_monthly: float | None = None,
) -> np.ndarray:
    """Construct ``A`` for the frozen constraint ``12 * ||A w||^2 <= .10^2``.

    The ordinary covariance is ``B F B' + D``.  The final row adds four more
    units of market-factor variance around zero beta without removing the
    ordinary market component already present in ``F``.
    """

    betas = _finite_array(factor_betas, ndim=2, name="factor betas")
    if betas.shape[1] != len(FACTOR_ORDER):
        raise ValueError("factor betas must follow the six-factor frozen order")
    covariance = _finite_array(
        factor_covariance_monthly,
        ndim=2,
        name="factor covariance",
    )
    if covariance.shape != (len(FACTOR_ORDER), len(FACTOR_ORDER)):
        raise ValueError("factor covariance must be 6 by 6")
    specific = _finite_array(
        specific_variance_monthly,
        ndim=1,
        name="specific variance",
    )
    if specific.shape[0] != betas.shape[0]:
        raise ValueError("specific variance and beta row counts differ")
    if np.any(specific <= 0.0):
        raise ValueError("specific variances must be strictly positive")

    if market_variance_monthly is None:
        market_variance = float(covariance[0, 0])
    else:
        market_variance = float(market_variance_monthly)
    if not np.isfinite(market_variance) or market_variance < 0.0:
        raise ValueError("market variance must be finite and nonnegative")

    factor_root = covariance_square_root(covariance)
    # F = C C'; hence B F B' = (B C)(B C)'.  Rows of A are risk sources.
    systematic_rows = (betas @ factor_root).T
    specific_rows = np.diag(np.sqrt(specific))
    soft_market_row = (
        np.sqrt(MARKET_BETA_VARIANCE_INCREMENT * market_variance) * betas[:, 0]
    )[None, :]
    result = np.vstack((systematic_rows, specific_rows, soft_market_row))
    if not np.all(np.isfinite(result)):
        raise ValueError("augmented risk root contains nonfinite values")
    return result


def augmented_risk_diagnostics(
    weights: object,
    factor_betas: object,
    factor_covariance_monthly: object,
    specific_variance_monthly: object,
    *,
    market_variance_monthly: float | None = None,
) -> RiskDiagnostics:
    """Recompute ordinary and augmented risk without optimizer residuals."""

    vector = _finite_array(weights, ndim=1, name="weights")
    betas = _finite_array(factor_betas, ndim=2, name="factor betas")
    if vector.shape[0] != betas.shape[0]:
        raise ValueError("weight and beta row counts differ")
    covariance = _finite_array(
        factor_covariance_monthly,
        ndim=2,
        name="factor covariance",
    )
    specific = _finite_array(
        specific_variance_monthly,
        ndim=1,
        name="specific variance",
    )
    if specific.shape[0] != vector.shape[0]:
        raise ValueError("weight and specific-variance row counts differ")
    if np.any(specific <= 0.0):
        raise ValueError("specific variances must be strictly positive")
    exposures = betas.T @ vector
    ordinary = float(exposures @ covariance @ exposures + np.dot(specific, vector**2))
    market_variance = (
        float(covariance[0, 0])
        if market_variance_monthly is None
        else float(market_variance_monthly)
    )
    augmented = ordinary + (
        MARKET_BETA_VARIANCE_INCREMENT * market_variance * float(exposures[0]) ** 2
    )
    annual_variance = ANNUAL_PERIODS * augmented
    # Negative values no larger than roundoff are clipped only for reporting.
    annual_volatility = float(np.sqrt(max(annual_variance, 0.0)))
    return RiskDiagnostics(
        monthly_variance=ordinary,
        augmented_monthly_variance=augmented,
        augmented_annual_variance=annual_variance,
        augmented_annual_volatility=annual_volatility,
        factor_exposures=np.asarray(exposures, dtype=np.float64),
    )


def augmented_annual_volatility_from_root(
    weights: object,
    risk_root: object,
) -> float:
    vector = _finite_array(weights, ndim=1, name="weights")
    root = _finite_array(risk_root, ndim=2, name="risk root")
    if root.shape[1] != vector.shape[0]:
        raise ValueError("risk root and weights have incompatible shapes")
    return float(np.sqrt(ANNUAL_PERIODS) * np.linalg.norm(root @ vector, ord=2))


__all__ = [
    "ANNUAL_PERIODS",
    "ANNUAL_VARIANCE_BUDGET",
    "FACTOR_ORDER",
    "MARKET_BETA_VARIANCE_INCREMENT",
    "PSD_EIGENVALUE_FLOOR",
    "RiskDiagnostics",
    "augmented_annual_volatility_from_root",
    "augmented_risk_diagnostics",
    "build_augmented_risk_root",
    "covariance_square_root",
]
