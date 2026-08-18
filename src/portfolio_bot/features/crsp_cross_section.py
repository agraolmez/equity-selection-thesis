"""Cross-sectional finalization for materialized CRSP Stage 6 features."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from portfolio_bot.features.crsp_features import (
    BETA_FEATURE_NAMES,
    CRSP_FEATURE_NAMES,
)
from portfolio_bot.features.feature_lineage import (
    FeatureFamilyResult,
    FeatureLineage,
    make_lineage,
)
from portfolio_bot.features.feature_math import ordinary_median, type7_quantile


@dataclass(frozen=True)
class CrspCrossSectionRow:
    signal_date: dt.date
    permno: int
    long_eligible: bool
    primary_exchange: str
    result: FeatureFamilyResult


class CrspCrossSectionError(RuntimeError):
    """Raised when an unfinalized signal cross-section is not canonical."""


def _lineage_max(
    contributors: Iterable[FeatureLineage],
) -> tuple[dt.date | None, dt.date | None, dt.date | None]:
    items = [item for item in contributors if item.value_status == "PRESENT"]
    observations = [
        item.max_observation_date
        for item in items
        if item.max_observation_date is not None
    ]
    availability = [
        item.max_public_availability_date
        for item in items
        if item.max_public_availability_date is not None
    ]
    stale = [
        item.stale_after_date
        for item in items
        if item.stale_after_date is not None
    ]
    return (
        max(observations) if observations else None,
        max(availability) if availability else None,
        min(stale) if stale else None,
    )


def _type7_bound_rows(
    rows: Sequence[CrspCrossSectionRow],
    feature: str,
    quantile: float,
) -> tuple[CrspCrossSectionRow, ...]:
    population = [
        row
        for row in rows
        if row.long_eligible and row.result.values[feature] is not None
    ]
    population.sort(
        key=lambda row: (float(row.result.values[feature]), row.permno)
    )
    if not population:
        return ()
    position = (len(population) - 1) * quantile
    return tuple(
        population[index]
        for index in sorted({math.floor(position), math.ceil(position)})
    )


def _assign_exchange_quintiles(
    rows: Sequence[CrspCrossSectionRow],
    feature: str,
) -> dict[int, int]:
    by_exchange: dict[str, list[CrspCrossSectionRow]] = {}
    for row in rows:
        value = row.result.values[feature]
        if (
            row.long_eligible
            and row.primary_exchange in {"N", "A", "Q"}
            and value is not None
            and math.isfinite(float(value))
            and float(value) > 0.0
        ):
            by_exchange.setdefault(row.primary_exchange, []).append(row)
    result: dict[int, int] = {}
    for group in by_exchange.values():
        group.sort(
            key=lambda row: (float(row.result.values[feature]), row.permno)
        )
        count = len(group)
        for rank, row in enumerate(group):
            result[row.permno] = math.floor(5 * rank / count) + 1
    return result


def _finalize_betas(
    rows: Sequence[CrspCrossSectionRow],
    values: Mapping[int, dict[str, object]],
    lineage: Mapping[int, dict[str, FeatureLineage]],
) -> None:
    for feature in BETA_FEATURE_NAMES:
        population = [
            float(row.result.values[feature])
            for row in rows
            if row.long_eligible and row.result.values[feature] is not None
        ]
        lower = type7_quantile(population, 0.01)
        upper = type7_quantile(population, 0.99)
        for row in rows:
            raw = row.result.values[feature]
            if raw is None:
                continue
            if lower is None or upper is None:
                values[row.permno][feature] = None
                lineage[row.permno][feature] = make_lineage(
                    None,
                    None,
                    None,
                    missing_status="MISSING_INSUFFICIENT_HISTORY",
                )
                continue
            raw_float = float(raw)
            final = min(upper, max(lower, raw_float))
            values[row.permno][feature] = final
            bound_rows: tuple[CrspCrossSectionRow, ...] = ()
            if raw_float < lower:
                bound_rows = _type7_bound_rows(rows, feature, 0.01)
            elif raw_float > upper:
                bound_rows = _type7_bound_rows(rows, feature, 0.99)
            contributors = [
                row.result.lineage[feature],
                *(item.result.lineage[feature] for item in bound_rows),
            ]
            observation, public, stale = _lineage_max(contributors)
            lineage[row.permno][feature] = make_lineage(
                final, observation, public, stale
            )


def _finalize_spreads(
    rows: Sequence[CrspCrossSectionRow],
    values: Mapping[int, dict[str, object]],
    lineage: Mapping[int, dict[str, FeatureLineage]],
) -> None:
    feature = "median_estimated_full_spread_20d"
    size_bins = _assign_exchange_quintiles(rows, "log_market_cap")
    adv_bins = _assign_exchange_quintiles(rows, "median_dollar_volume_20d")
    contributors = [
        row
        for row in rows
        if row.long_eligible
        and row.result.values[feature] is not None
        and math.isfinite(float(row.result.values[feature]))
    ]

    def peer_result(
        candidates: Iterable[CrspCrossSectionRow],
    ) -> tuple[float | None, tuple[CrspCrossSectionRow, ...]]:
        selected = tuple(candidates)
        peer_values = [float(row.result.values[feature]) for row in selected]
        return (
            ordinary_median(peer_values) if len(peer_values) >= 20 else None,
            selected,
        )

    for row in rows:
        if row.result.values[feature] is not None:
            continue
        if (
            not row.long_eligible
            or row.permno not in size_bins
            or row.permno not in adv_bins
        ):
            continue
        size = size_bins[row.permno]
        adv = adv_bins[row.permno]
        exchange = row.primary_exchange
        candidate_groups = (
            [
                item
                for item in contributors
                if item.primary_exchange == exchange
                and size_bins.get(item.permno) == size
                and adv_bins.get(item.permno) == adv
            ],
            [
                item
                for item in contributors
                if item.primary_exchange == exchange
                and size_bins.get(item.permno) == size
                and abs(adv_bins.get(item.permno, -99) - adv) <= 1
            ],
            [
                item
                for item in contributors
                if item.primary_exchange == exchange
                and size_bins.get(item.permno) == size
                and abs(adv_bins.get(item.permno, -99) - adv) <= 2
            ],
            [
                item
                for item in contributors
                if item.primary_exchange == exchange
                and abs(size_bins.get(item.permno, -99) - size) <= 1
            ],
            [
                item
                for item in contributors
                if item.primary_exchange == exchange
                and abs(size_bins.get(item.permno, -99) - size) <= 2
            ],
            [item for item in contributors if item.primary_exchange == exchange],
            contributors,
        )
        resolved = None
        selected: tuple[CrspCrossSectionRow, ...] = ()
        for candidates in candidate_groups:
            resolved, selected = peer_result(candidates)
            if resolved is not None:
                break
        if resolved is None:
            continue
        values[row.permno][feature] = resolved
        assignment = (
            row.result.lineage["log_market_cap"],
            row.result.lineage["median_dollar_volume_20d"],
        )
        observation, public, stale = _lineage_max(
            (
                *assignment,
                *(item.result.lineage[feature] for item in selected),
            )
        )
        lineage[row.permno][feature] = make_lineage(
            resolved, observation, public, stale
        )


def finalize_crsp_signal_cross_section(
    rows: Sequence[CrspCrossSectionRow],
) -> dict[int, FeatureFamilyResult]:
    """Apply frozen beta clipping and spread-peer imputation at one signal."""

    if not rows:
        return {}
    signal_dates = {row.signal_date for row in rows}
    permnos = [row.permno for row in rows]
    if len(signal_dates) != 1 or len(permnos) != len(set(permnos)):
        raise CrspCrossSectionError(
            "CRSP finalization requires one unique signal cross-section"
        )
    for row in rows:
        if tuple(row.result.values) != CRSP_FEATURE_NAMES:
            raise CrspCrossSectionError("Unfinalized CRSP feature order differs")
    values: dict[int, dict[str, object]] = {
        row.permno: dict(row.result.values) for row in rows
    }
    lineage: dict[int, dict[str, FeatureLineage]] = {
        row.permno: dict(row.result.lineage) for row in rows
    }
    _finalize_betas(rows, values, lineage)
    _finalize_spreads(rows, values, lineage)
    return {
        row.permno: FeatureFamilyResult(
            values={
                name: values[row.permno][name] for name in CRSP_FEATURE_NAMES
            },
            lineage={
                name: lineage[row.permno][name] for name in CRSP_FEATURE_NAMES
            },
        )
        for row in rows
    }
