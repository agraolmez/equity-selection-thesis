"""One-pass CRSP family adapters for Stage 6.3 materialization."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa

from portfolio_bot.features.crsp_cross_section import (
    CrspCrossSectionRow,
    finalize_crsp_signal_cross_section,
)
from portfolio_bot.features.crsp_features import (
    PRICE_TREND_FEATURE_NAMES,
    RISK_DISTRIBUTION_FEATURE_NAMES,
    SIZE_LIQUIDITY_FEATURE_NAMES,
)
from portfolio_bot.features.feature_lineage import FeatureFamilyResult
from portfolio_bot.features.fast_crsp_features import (
    CachedCrspSecurityEngine,
)


DEVELOPMENT_LAST_SIGNAL = dt.date(2014, 10, 31)
MAXIMUM_CONTEXT_DATE = dt.date(2014, 12, 1)

CRSP_FAMILIES = {
    "price_trends_path": PRICE_TREND_FEATURE_NAMES,
    "risk_return_distribution": RISK_DISTRIBUTION_FEATURE_NAMES,
    "size_liquidity_trading_frictions": SIZE_LIQUIDITY_FEATURE_NAMES,
}


class CrspMaterializationError(RuntimeError):
    """Raised when a Stage 6.3 CRSP batch is not canonical."""


@dataclass(frozen=True)
class CrspMaterializedRow:
    signal_date: dt.date
    permno: int
    permco: int
    long_eligible: bool
    short_eligible: bool
    primary_exchange: str
    share_type: str
    result: FeatureFamilyResult


def _validated_sessions(sessions: Sequence[dt.date]) -> tuple[dt.date, ...]:
    result = tuple(sessions)
    if (
        not result
        or result != tuple(sorted(set(result)))
        or any(
            not isinstance(day, dt.date)
            or isinstance(day, dt.datetime)
            or day > MAXIMUM_CONTEXT_DATE
            for day in result
        )
    ):
        raise CrspMaterializationError(
            "Trading sessions are not unique sorted development context"
        )
    return result


def compute_unfinalized_crsp_security_rows(
    *,
    signal_rows: Iterable[Mapping[str, Any]],
    sessions: Sequence[dt.date],
    market_rows: Sequence[Mapping[str, Any]],
    return_rows: Sequence[Mapping[str, Any]],
    factor_rows: Sequence[Mapping[str, Any]],
    shares_rows: Sequence[Mapping[str, Any]],
) -> list[CrspMaterializedRow]:
    """Compute every signal for one PERMNO before cross-sectional finalization."""

    spine = sorted(
        (dict(row) for row in signal_rows),
        key=lambda row: (row["signal_date"], int(row["permno"])),
    )
    if not spine:
        return []
    permnos = {int(row["permno"]) for row in spine}
    keys = [(row["signal_date"], int(row["permno"])) for row in spine]
    if len(permnos) != 1 or len(keys) != len(set(keys)):
        raise CrspMaterializationError(
            "Security materialization requires one PERMNO and unique signals"
        )
    session_clock = _validated_sessions(sessions)
    engine = CachedCrspSecurityEngine(
        sessions=session_clock,
        market_rows=market_rows,
        return_rows=return_rows,
        factor_rows=factor_rows,
        shares_rows=shares_rows,
        listing_history_rows=market_rows,
    )
    output: list[CrspMaterializedRow] = []
    for row in spine:
        signal_date = row["signal_date"]
        if (
            not isinstance(signal_date, dt.date)
            or isinstance(signal_date, dt.datetime)
            or signal_date > DEVELOPMENT_LAST_SIGNAL
            or signal_date not in session_clock
        ):
            raise CrspMaterializationError(
                "Signal escaped the frozen development session scope"
            )
        output.append(
            CrspMaterializedRow(
                signal_date=signal_date,
                permno=int(row["permno"]),
                permco=int(row["permco"]),
                long_eligible=bool(row["long_eligible"]),
                short_eligible=bool(row["short_eligible"]),
                primary_exchange=str(row["primary_exchange"]),
                share_type=str(row["share_type"]),
                result=engine.build_unfinalized_result(
                    permno=int(row["permno"]),
                    signal_date=signal_date,
                    long_eligible=bool(row["long_eligible"]),
                    primary_exchange=str(row["primary_exchange"]),
                ),
            )
        )
    return output


def finalize_crsp_materialized_signal(
    rows: Sequence[CrspMaterializedRow],
) -> list[CrspMaterializedRow]:
    """Finalize one complete signal cross-section and preserve spine order."""

    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: (row.signal_date, row.permno))
    if len({row.signal_date for row in ordered}) != 1:
        raise CrspMaterializationError(
            "Cross-sectional finalization received multiple signal dates"
        )
    finalized = finalize_crsp_signal_cross_section(
        [
            CrspCrossSectionRow(
                signal_date=row.signal_date,
                permno=row.permno,
                long_eligible=row.long_eligible,
                primary_exchange=row.primary_exchange,
                result=row.result,
            )
            for row in ordered
        ]
    )
    return [
        replace(row, result=finalized[row.permno])
        for row in ordered
    ]


def crsp_rows_to_arrow(
    rows: Sequence[CrspMaterializedRow],
) -> tuple[pa.Table, Mapping[str, pa.Table]]:
    """Convert a finalized signal chunk into wide values and three lineage shards."""

    ordered = sorted(rows, key=lambda row: (row.signal_date, row.permno))
    keys = [(row.signal_date, row.permno) for row in ordered]
    if len(keys) != len(set(keys)):
        raise CrspMaterializationError("Duplicate CRSP materialized key")
    audit: dict[str, pa.Array] = {
        "signal_date": pa.array(
            [row.signal_date for row in ordered], type=pa.date32()
        ),
        "permno": pa.array([row.permno for row in ordered], type=pa.int64()),
        "permco": pa.array([row.permco for row in ordered], type=pa.int64()),
        "long_eligible": pa.array(
            [row.long_eligible for row in ordered], type=pa.bool_()
        ),
        "short_eligible": pa.array(
            [row.short_eligible for row in ordered], type=pa.bool_()
        ),
        "primary_exchange": pa.array(
            [row.primary_exchange for row in ordered], type=pa.string()
        ),
        "share_type": pa.array(
            [row.share_type for row in ordered], type=pa.string()
        ),
    }
    feature_names = tuple(
        name for family in CRSP_FAMILIES.values() for name in family
    )
    wide = pa.table(
        {
            **audit,
            **{
                name: pa.array(
                    [row.result.values[name] for row in ordered],
                    type=pa.float64(),
                    from_pandas=True,
                )
                for name in feature_names
            },
        }
    )
    lineage_tables: dict[str, pa.Table] = {}
    for family, names in CRSP_FAMILIES.items():
        long_rows = [
            (
                row.signal_date,
                row.permno,
                name,
                row.result.lineage[name],
            )
            for row in ordered
            for name in names
        ]
        lineage_tables[family] = pa.table(
            {
                "signal_date": pa.array(
                    [item[0] for item in long_rows], type=pa.date32()
                ),
                "permno": pa.array(
                    [item[1] for item in long_rows], type=pa.int64()
                ),
                "feature_name": pa.array(
                    [item[2] for item in long_rows], type=pa.string()
                ),
                "max_observation_date": pa.array(
                    [
                        item[3].max_observation_date
                        for item in long_rows
                    ],
                    type=pa.date32(),
                ),
                "max_public_availability_date": pa.array(
                    [
                        item[3].max_public_availability_date
                        for item in long_rows
                    ],
                    type=pa.date32(),
                ),
                "stale_after_date": pa.array(
                    [item[3].stale_after_date for item in long_rows],
                    type=pa.date32(),
                ),
                "value_status": pa.array(
                    [item[3].value_status for item in long_rows],
                    type=pa.string(),
                ),
            }
        )
    return wide, lineage_tables
