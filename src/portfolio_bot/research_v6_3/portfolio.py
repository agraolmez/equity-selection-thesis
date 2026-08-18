"""Portfolio-side transformations for V6.3 signal-to-weight research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.linalg import block_diag


@dataclass(frozen=True, slots=True)
class LatentRiskResult:
    factor_betas: np.ndarray
    covariance_root: np.ndarray
    specific_variance_monthly: np.ndarray
    observations: int
    factors: int
    explained_residual_variance: float


def causal_opportunity_percentiles(
    monthly_scores: pd.Series, *, minimum_history_months: int = 12
) -> pd.Series:
    """Expanding percentile of a signal-only monthly opportunity statistic."""

    scores = pd.to_numeric(monthly_scores, errors="coerce")
    result = pd.Series(0.50, index=scores.index, dtype=float)
    for position in range(len(scores)):
        history = scores.iloc[:position].dropna()
        current = scores.iloc[position]
        if len(history) < int(minimum_history_months) or not np.isfinite(current):
            continue
        result.iloc[position] = float(
            (history.lt(current).sum() + 0.5 * history.eq(current).sum()) / len(history)
        )
    return result


def opportunity_multiplier(
    percentile: float,
    *,
    strength: float,
    minimum: float,
    maximum: float,
) -> float:
    """Map a causal opportunity percentile to a bounded expected-return scale."""

    base = 0.75 + 0.50 * float(percentile)
    return float(np.clip(1.0 + float(strength) * (base - 1.0), minimum, maximum))


def latent_residual_risk(
    current: pd.DataFrame,
    history: pd.DataFrame,
    base_covariance_root: np.ndarray,
    *,
    factor_columns: Sequence[str],
    latent_factors: int,
    latent_variance_share: float,
    trailing_months: int,
    minimum_months: int,
    specific_variance_floor_fraction: float,
) -> LatentRiskResult:
    """Add causal PCA factors extracted from factor-model residual returns.

    This is deliberately described as an IPCA-style latent risk layer rather
    than a full characteristic-instrumented IPCA estimator. It preserves the
    frozen economic factor block and adds only common covariance remaining in
    trailing stock residuals.
    """

    if latent_factors < 1:
        raise ValueError("latent factor count must be positive")
    if not 0.0 < latent_variance_share <= 1.0:
        raise ValueError("latent variance share must lie in (0, 1]")
    month = str(current["holding_month"].iloc[0])
    signal_date = pd.Timestamp(current["signal_date"].iloc[0])
    usable = history.loc[
        history["holding_month"].astype(str).lt(month)
        & pd.to_datetime(history["label_available_date"], errors="coerce").le(signal_date)
    ].copy()
    observed_months = sorted(usable["holding_month"].astype(str).unique())
    observed_months = observed_months[-int(trailing_months) :]
    usable = usable.loc[usable["holding_month"].astype(str).isin(observed_months)]
    if len(observed_months) < int(minimum_months):
        raise ValueError(
            f"latent risk has {len(observed_months)} completed months, needs {minimum_months}"
        )
    current_permnos = current["permno"].astype(int).to_numpy()
    usable = usable.loc[usable["permno"].astype(int).isin(set(current_permnos.tolist()))]
    if usable.empty:
        raise ValueError("latent risk has no trailing returns for the current universe")
    excess = pd.to_numeric(usable["stock_excess_return"], errors="coerce").to_numpy(float)
    betas = usable.loc[:, list(factor_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    realized = usable.loc[:, [column.removeprefix("beta_risk_") for column in factor_columns]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    residual = excess - np.sum(betas * realized, axis=1)
    usable["_residual"] = residual
    panel = usable.pivot_table(
        index="holding_month", columns="permno", values="_residual", aggfunc="first"
    ).reindex(index=observed_months, columns=current_permnos)
    values = panel.to_numpy(float)
    observed = np.isfinite(values)
    column_count = observed.sum(axis=0)
    column_mean = np.divide(
        np.nansum(values, axis=0),
        column_count,
        out=np.zeros(values.shape[1], dtype=float),
        where=column_count > 0,
    )
    centered = np.where(observed, values - column_mean, 0.0)
    # Months are also centered cross-sectionally so the added layer represents
    # residual co-movement rather than re-estimating the existing market block.
    row_count = observed.sum(axis=1)
    row_mean = np.divide(
        centered.sum(axis=1),
        row_count,
        out=np.zeros(values.shape[0], dtype=float),
        where=row_count > 0,
    )
    centered = np.where(observed, centered - row_mean[:, None], 0.0)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    factors = min(int(latent_factors), len(singular), max(len(observed_months) - 1, 1))
    scale = np.sqrt(float(latent_variance_share))
    latent_betas = (
        right[:factors].T
        * (singular[:factors] / np.sqrt(max(len(observed_months) - 1, 1)))
        * scale
    )
    base_betas = current.loc[:, list(factor_columns)].to_numpy(float)
    combined_betas = np.column_stack([base_betas, latent_betas])
    combined_root = block_diag(
        np.asarray(base_covariance_root, dtype=float), np.eye(factors, dtype=float)
    )
    original_specific = pd.to_numeric(
        current["specific_variance_monthly"], errors="coerce"
    ).to_numpy(float)
    explained = np.sum(np.square(latent_betas), axis=1)
    floor = float(specific_variance_floor_fraction) * original_specific
    adjusted_specific = np.maximum(original_specific - explained, floor)
    total_residual_variance = float(np.sum(np.var(centered, axis=0, ddof=1)))
    explained_share = (
        float(np.sum(explained) / total_residual_variance)
        if total_residual_variance > 0.0
        else 0.0
    )
    return LatentRiskResult(
        factor_betas=combined_betas,
        covariance_root=combined_root,
        specific_variance_monthly=adjusted_specific,
        observations=len(observed_months),
        factors=factors,
        explained_residual_variance=explained_share,
    )


__all__ = [
    "LatentRiskResult",
    "causal_opportunity_percentiles",
    "latent_residual_risk",
    "opportunity_multiplier",
]
