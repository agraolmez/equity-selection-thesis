"""Point-in-time fiscal-transition adapter for Stage 11 fundamentals.

The frozen Stage 6 quarterly formula indexes observations by
``(gvkey, permno, fyearq, fqtr)``.  Compustat can legitimately reuse that
key when a company changes its fiscal year end.  Development contains no
such reuse, while the fixed Stage 11 input contains a small number of
strictly time-ordered fiscal-calendar transitions.

This module changes neither the frozen feature formulas nor their ordinary
development path.  During the Stage 11 validation scope it:

* admits only uniquely time-resolvable fiscal-calendar transitions;
* selects the latest version that was publicly available at each signal;
* fails closed on an availability tie or any other duplicate shape; and
* passes the resolved rows to the unchanged Stage 6 bundle selector.

Consequently, a later fiscal-calendar observation can never affect an
earlier signal, and development continues through the exact original
callables because it has no duplicate fiscal keys.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import duckdb

import portfolio_bot.features.materialize_fundamentals as frozen_adapter
from portfolio_bot.features.feature_store_inputs import FeatureStoreInputs
from portfolio_bot.features.fundamental_features import (
    FundamentalBundle,
    FundamentalFeatureError,
)


FISCAL_TRANSITION_POLICY = "LATEST_UNIQUELY_AVAILABLE_VERSION_AS_OF_SIGNAL"

# Capture the exact frozen objects before the Stage 11 scope temporarily
# installs the two adapters below.
FROZEN_SELECT_FUNDAMENTAL_BUNDLE = frozen_adapter.select_fundamental_bundle
FROZEN_VALIDATE_SOURCE_GATES = frozen_adapter._validate_source_gates


def _known_by_signal(row: Mapping[str, Any], signal_date: dt.date) -> bool:
    observation = row.get("observation_date")
    availability = row.get("public_availability_date")
    return (
        row.get("standard_format_eligible") is True
        and isinstance(observation, dt.date)
        and not isinstance(observation, dt.datetime)
        and isinstance(availability, dt.date)
        and not isinstance(availability, dt.datetime)
        and observation <= signal_date
        and availability <= signal_date
    )


def _fiscal_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    try:
        return (
            str(row["gvkey"]),
            int(row["permno"]),
            int(row["fyearq"]),
            int(row["fqtr"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise FundamentalFeatureError(
            "Quarterly fiscal-transition row has an invalid key"
        ) from exc


def resolve_quarterly_versions_as_of_signal(
    quarterly_rows: Iterable[Mapping[str, Any]],
    signal_date: dt.date,
) -> tuple[Mapping[str, Any], ...]:
    """Return one uniquely latest, publicly known row per frozen fiscal key."""

    if not isinstance(signal_date, dt.date) or isinstance(signal_date, dt.datetime):
        raise FundamentalFeatureError("Fiscal-transition signal date is invalid")
    selected: dict[tuple[str, int, int, int], Mapping[str, Any]] = {}
    for row in quarterly_rows:
        if not _known_by_signal(row, signal_date):
            continue
        key = _fiscal_key(row)
        availability = row["public_availability_date"]
        previous = selected.get(key)
        if previous is None:
            selected[key] = row
            continue
        prior_availability = previous["public_availability_date"]
        if availability == prior_availability:
            raise FundamentalFeatureError(
                f"Quarterly fiscal versions tie at public availability: {key}"
            )
        if availability > prior_availability:
            selected[key] = row
    return tuple(
        selected[key]
        for key in sorted(
            selected,
            key=lambda value: (value[0], value[1], value[2], value[3]),
        )
    )


def select_validation_fundamental_bundle(
    quarterly_rows: Iterable[Mapping[str, Any]],
    annual_rows: Iterable[Mapping[str, Any]],
    signal_date: dt.date,
) -> FundamentalBundle | None:
    """Resolve quarterly versions as-of signal, then call the frozen selector."""

    resolved = resolve_quarterly_versions_as_of_signal(
        quarterly_rows,
        signal_date,
    )
    return FROZEN_SELECT_FUNDAMENTAL_BUNDLE(
        resolved,
        annual_rows,
        signal_date,
    )


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _transition_summary(
    connection: duckdb.DuckDBPyConnection,
) -> Mapping[str, int | str]:
    row = connection.execute(
        """
        WITH eligible AS (
            SELECT *
            FROM quarterly
            WHERE standard_format_eligible IS TRUE
        ),
        duplicate_groups AS (
            SELECT
                gvkey,
                permno,
                fyearq,
                fqtr,
                count(*) AS version_count,
                max(public_availability_date) AS ambiguity_known_date
            FROM eligible
            GROUP BY gvkey, permno, fyearq, fqtr
            HAVING count(*) > 1
        ),
        first_ambiguity AS (
            SELECT permno, min(ambiguity_known_date) AS ambiguity_known_date
            FROM duplicate_groups
            GROUP BY permno
        ),
        affected_signals AS (
            SELECT spine.signal_date, spine.permno
            FROM spine
            INNER JOIN first_ambiguity USING (permno)
            WHERE spine.signal_date >= first_ambiguity.ambiguity_known_date
        ),
        pair_signal_instances AS (
            SELECT spine.signal_date, duplicate_groups.*
            FROM spine
            INNER JOIN duplicate_groups USING (permno)
            WHERE spine.signal_date >= duplicate_groups.ambiguity_known_date
        )
        SELECT
            count(*) FROM duplicate_groups
        """
    ).fetchone()
    if row is None:
        raise frozen_adapter.FundamentalMaterializationError(
            "Fiscal-transition audit did not return a group count"
        )
    duplicate_groups = int(row[0])
    values = connection.execute(
        """
        WITH eligible AS (
            SELECT *
            FROM quarterly
            WHERE standard_format_eligible IS TRUE
        ),
        duplicate_groups AS (
            SELECT
                gvkey,
                permno,
                fyearq,
                fqtr,
                count(*) AS version_count,
                max(public_availability_date) AS ambiguity_known_date
            FROM eligible
            GROUP BY gvkey, permno, fyearq, fqtr
            HAVING count(*) > 1
        ),
        first_ambiguity AS (
            SELECT permno, min(ambiguity_known_date) AS ambiguity_known_date
            FROM duplicate_groups
            GROUP BY permno
        ),
        affected_signals AS (
            SELECT spine.signal_date, spine.permno
            FROM spine
            INNER JOIN first_ambiguity USING (permno)
            WHERE spine.signal_date >= first_ambiguity.ambiguity_known_date
        ),
        pair_signal_instances AS (
            SELECT spine.signal_date, duplicate_groups.permno,
                   duplicate_groups.fyearq, duplicate_groups.fqtr
            FROM spine
            INNER JOIN duplicate_groups USING (permno)
            WHERE spine.signal_date >= duplicate_groups.ambiguity_known_date
        )
        SELECT
            coalesce(sum(version_count), 0),
            count(DISTINCT permno),
            (SELECT count(*) FROM affected_signals),
            (SELECT count(*) FROM pair_signal_instances)
        FROM duplicate_groups
        """
    ).fetchone()
    if values is None:
        raise frozen_adapter.FundamentalMaterializationError(
            "Fiscal-transition audit did not return its summary"
        )
    return MappingProxyType(
        {
            "resolution_policy": FISCAL_TRANSITION_POLICY,
            "duplicate_group_count": duplicate_groups,
            "duplicate_row_count": int(values[0]),
            "affected_security_count": int(values[1]),
            "affected_signal_row_count": int(values[2]),
            "pair_signal_instance_count": int(values[3]),
            "availability_tie_group_count": 0,
        }
    )


def _validate_transition_shapes(
    connection: duckdb.DuckDBPyConnection,
) -> Mapping[str, int | str]:
    bad = connection.execute(
        """
        WITH eligible AS (
            SELECT *
            FROM quarterly
            WHERE standard_format_eligible IS TRUE
        ),
        duplicate_groups AS (
            SELECT
                gvkey,
                permno,
                fyearq,
                fqtr,
                count(*) AS version_count,
                count(DISTINCT source_record_id) AS source_count,
                count(DISTINCT datadate) AS datadate_count,
                count(DISTINCT observation_date) AS observation_count,
                count(DISTINCT public_availability_date) AS availability_count,
                count(DISTINCT fyr) AS fiscal_year_end_count,
                count(DISTINCT datafqtr) AS datafqtr_count,
                count(*) FILTER (
                    WHERE observation_date IS DISTINCT FROM datadate
                ) AS observation_mismatch_count,
                arg_min(public_availability_date, datadate)
                    AS earlier_public_availability,
                arg_max(public_availability_date, datadate)
                    AS later_public_availability
            FROM eligible
            GROUP BY gvkey, permno, fyearq, fqtr
            HAVING count(*) > 1
        )
        SELECT count(*)
        FROM duplicate_groups
        WHERE version_count <> 2
           OR source_count <> 2
           OR datadate_count <> 2
           OR observation_count <> 2
           OR availability_count <> 2
           OR fiscal_year_end_count <> 2
           OR datafqtr_count <> 1
           OR observation_mismatch_count <> 0
           OR earlier_public_availability >= later_public_availability
        """
    ).fetchone()
    if bad is None or int(bad[0]) != 0:
        raise frozen_adapter.FundamentalMaterializationError(
            "Quarterly duplicate is not a uniquely time-resolvable fiscal transition"
        )
    summary = _transition_summary(connection)
    if int(summary["duplicate_group_count"]) <= 0:
        raise frozen_adapter.FundamentalMaterializationError(
            "Fiscal-transition gate was entered without a duplicate group"
        )
    return summary


def validate_validation_fundamental_source_gates(
    connection: duckdb.DuckDBPyConnection,
    inputs: FeatureStoreInputs,
) -> None:
    """Run frozen gates, admitting only audited fiscal-calendar transitions."""

    try:
        FROZEN_VALIDATE_SOURCE_GATES(connection, inputs)
        return
    except frozen_adapter.FundamentalMaterializationError as exc:
        if str(exc) != "Duplicate quarterly gvkey+permno fiscal key":
            raise

    _validate_transition_shapes(connection)
    quarterly_path = inputs.path(frozen_adapter.QUARTERLY_TABLE)
    quoted_path = _sql_path(quarterly_path)
    connection.execute("DROP VIEW quarterly")
    connection.execute(
        f"""
        CREATE TEMP VIEW quarterly AS
        WITH ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY gvkey, permno, fyearq, fqtr
                    ORDER BY
                        public_availability_date DESC,
                        datadate DESC,
                        source_record_id DESC
                ) AS _validation_version_rank
            FROM read_parquet('{quoted_path}')
        )
        SELECT * EXCLUDE (_validation_version_rank)
        FROM ranked
        WHERE standard_format_eligible IS DISTINCT FROM TRUE
           OR _validation_version_rank = 1
        """
    )
    try:
        FROZEN_VALIDATE_SOURCE_GATES(connection, inputs)
    finally:
        connection.execute("DROP VIEW quarterly")
        connection.read_parquet(str(quarterly_path)).create_view("quarterly")


def validation_fiscal_transition_audit(
    inputs: FeatureStoreInputs,
) -> Mapping[str, int | str]:
    """Return the immutable empirical audit summary for receipt publication."""

    connection = duckdb.connect(database=":memory:")
    try:
        connection.read_parquet(
            str(inputs.path(frozen_adapter.SPINE_TABLE))
        ).create_view("spine")
        connection.read_parquet(
            str(inputs.path(frozen_adapter.QUARTERLY_TABLE))
        ).create_view("quarterly")
        return _validate_transition_shapes(connection)
    finally:
        connection.close()


__all__ = [
    "FISCAL_TRANSITION_POLICY",
    "FROZEN_SELECT_FUNDAMENTAL_BUNDLE",
    "FROZEN_VALIDATE_SOURCE_GATES",
    "resolve_quarterly_versions_as_of_signal",
    "select_validation_fundamental_bundle",
    "validate_validation_fundamental_source_gates",
    "validation_fiscal_transition_audit",
]
