"""Authenticated, bounded-memory OptionMetrics materialization for Stage 6.

The verified option-flow and historical-volatility sources contain roughly
181 million rows.  This adapter therefore reduces them set-wise in DuckDB and
streams only one compact row per signal-spine key into Python.  It never loads
either vendor table into a Python collection and never writes a feature-store
artifact.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import duckdb
import pyarrow.parquet as pq

from portfolio_bot.features.feature_lineage import (
    FeatureFamilyResult,
    FeatureLineage,
)
from portfolio_bot.features.feature_store_contract import DEFAULT_POLICY
from portfolio_bot.features.feature_store_inputs import FeatureStoreInputs
from portfolio_bot.features.feature_store_materialization import (
    FeatureMaterializationRow,
    FeatureStoreBuildIdentity,
    authenticate_feature_store_inputs,
    validate_precomputed_feature_store_build_identity,
)
from portfolio_bot.features.optionmetrics_features import FEATURE_NAMES


SPINE_TABLE = "crsp_signal_date_universe.parquet"
LINKAGE_TABLE = "auxiliary_signal_linkage.parquet"
OPTION_LINK_TABLE = "optionmetrics_crsp_link_interval.parquet"
FLOW_TABLE = "optionmetrics_option_volume_observation.parquet"
HISTORICAL_VOLATILITY_TABLE = (
    "optionmetrics_historical_volatility_observation.parquet"
)
MARKET_TABLE = "crsp_security_day.parquet"
CALENDAR_TABLE = "trading_calendar.parquet"

FIRST_SIGNAL_DATE = dt.date(2000, 1, 31)
LAST_SIGNAL_DATE = dt.date(2014, 10, 31)
EXPECTED_SPINE_ROWS = 1_263_048
EXPECTED_SIGNAL_DATES = 178

_SPINE_COLUMNS = (
    "signal_date",
    "permno",
    "permco",
    "long_eligible",
    "short_eligible",
    "primary_exchange",
    "share_type",
)
_LINKAGE_COLUMNS = (
    "signal_date",
    "permno",
    "optionmetrics_match_status",
    "optionmetrics_secid",
    "universe_row_preserved",
)
_OPTION_LINK_COLUMNS = (
    "permno",
    "secid",
    "link_start_date",
    "link_end_date",
    "link_status",
    "candidate_count",
)
_FLOW_COLUMNS = (
    "secid",
    "observation_date",
    "public_availability_date",
    "row_role",
    "authoritative_flow_component",
    "volume",
    "open_interest",
    "genuine_zero_volume",
    "genuine_zero_open_interest",
)
_HV_COLUMNS = (
    "secid",
    "observation_date",
    "public_availability_date",
    "days",
    "volatility",
    "supported_horizon",
)
_MARKET_COLUMNS = ("permno", "calendar_date", "volume_shares")
_CALENDAR_COLUMNS = ("calendar_date",)

_LAG_FEATURES = (
    "option_total_volume_change_1m",
    "option_total_volume_change_3m",
    "option_total_open_interest_change_1m",
    "option_total_open_interest_change_3m",
)


class OptionMetricsMaterializationError(RuntimeError):
    """Raised when verified OptionMetrics inputs cannot be joined exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OptionMetricsMaterializationError(message)


def _date(value: Any, label: str) -> dt.date:
    _require(
        isinstance(value, dt.date) and not isinstance(value, dt.datetime),
        f"{label} must be a date",
    )
    return value


def _integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return int(value)


def _number(value: Any, *, nonnegative: bool = True) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        return None
    return result


def _required_columns(
    path: Path,
    columns: Sequence[str],
    table: str,
) -> None:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(columns) - available
    _require(not missing, f"{table} is missing columns: {sorted(missing)}")


def _create_views(
    connection: duckdb.DuckDBPyConnection,
    inputs: FeatureStoreInputs,
) -> None:
    for view, table in (
        ("spine", SPINE_TABLE),
        ("linkage", LINKAGE_TABLE),
        ("option_links", OPTION_LINK_TABLE),
        ("option_flow", FLOW_TABLE),
        ("option_hv", HISTORICAL_VOLATILITY_TABLE),
        ("market", MARKET_TABLE),
        ("calendar_source", CALENDAR_TABLE),
    ):
        connection.read_parquet(str(inputs.path(table))).create_view(view)


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    row = connection.execute(query).fetchone()
    _require(row is not None and len(row) == 1, "DuckDB scalar gate failed")
    return int(row[0])


def _validate_source_gates(
    connection: duckdb.DuckDBPyConnection,
    inputs: FeatureStoreInputs,
) -> tuple[dt.date, ...]:
    _require(
        inputs.tables[SPINE_TABLE].rows
        == inputs.tables[LINKAGE_TABLE].rows
        == EXPECTED_SPINE_ROWS,
        "Spine/linkage receipts do not equal the frozen Stage 6 scope",
    )
    scope = connection.execute(
        """
        SELECT
            count(*),
            count(DISTINCT signal_date),
            min(signal_date),
            max(signal_date)
        FROM spine
        """
    ).fetchone()
    _require(
        scope
        == (
            EXPECTED_SPINE_ROWS,
            EXPECTED_SIGNAL_DATES,
            FIRST_SIGNAL_DATE,
            LAST_SIGNAL_DATE,
        ),
        "Signal spine differs from the frozen row/date scope",
    )
    bad_spine = _scalar(
        connection,
        f"""
        SELECT count(*)
        FROM spine
        WHERE signal_date IS NULL
           OR permno IS NULL
           OR permco IS NULL
           OR long_eligible IS NULL
           OR short_eligible IS NULL
           OR primary_exchange IS NULL
           OR share_type IS NULL
           OR signal_date < DATE '{FIRST_SIGNAL_DATE.isoformat()}'
           OR signal_date > DATE '{LAST_SIGNAL_DATE.isoformat()}'
           OR (short_eligible IS TRUE AND long_eligible IS NOT TRUE)
        """,
    )
    _require(bad_spine == 0, "Signal spine escaped the development contract")
    bad_linkage = _scalar(
        connection,
        f"""
        SELECT count(*)
        FROM linkage
        WHERE signal_date IS NULL
           OR permno IS NULL
           OR signal_date < DATE '{FIRST_SIGNAL_DATE.isoformat()}'
           OR signal_date > DATE '{LAST_SIGNAL_DATE.isoformat()}'
           OR universe_row_preserved IS DISTINCT FROM TRUE
           OR optionmetrics_match_status
              NOT IN ('MATCHED', 'NOT_COVERED', 'AMBIGUOUS')
           OR (
                optionmetrics_match_status = 'MATCHED'
                AND optionmetrics_secid IS NULL
              )
           OR (
                optionmetrics_match_status <> 'MATCHED'
                AND optionmetrics_secid IS NOT NULL
              )
        """,
    )
    _require(bad_linkage == 0, "OptionMetrics linkage status is invalid")
    key_failures = _scalar(
        connection,
        """
        SELECT count(*)
        FROM (
            SELECT 'spine' AS source, signal_date, permno
            FROM spine
            GROUP BY signal_date, permno
            HAVING count(*) <> 1
            UNION ALL
            SELECT 'linkage' AS source, signal_date, permno
            FROM linkage
            GROUP BY signal_date, permno
            HAVING count(*) <> 1
        )
        """,
    )
    _require(key_failures == 0, "Spine or linkage signal key is duplicated")
    unmatched_keys = _scalar(
        connection,
        """
        SELECT count(*)
        FROM spine
        FULL OUTER JOIN linkage USING (signal_date, permno)
        WHERE spine.permno IS NULL OR linkage.permno IS NULL
        """,
    )
    _require(
        unmatched_keys == 0,
        "OptionMetrics linkage does not preserve the exact signal-spine key set",
    )
    bad_intervals = _scalar(
        connection,
        """
        SELECT count(*)
        FROM option_links
        WHERE permno IS NULL
           OR secid IS NULL
           OR link_start_date IS NULL
           OR link_end_date IS NULL
           OR link_start_date > link_end_date
           OR link_status <> 'MATCHED'
           OR candidate_count <> 1
        """,
    )
    _require(bad_intervals == 0, "OptionMetrics link interval is invalid")
    duplicate_intervals = _scalar(
        connection,
        """
        SELECT count(*)
        FROM (
            SELECT permno, secid, link_start_date, link_end_date
            FROM option_links
            GROUP BY permno, secid, link_start_date, link_end_date
            HAVING count(*) <> 1
        )
        """,
    )
    _require(
        duplicate_intervals == 0,
        "OptionMetrics link interval key is duplicated",
    )
    matched_link_failures = _scalar(
        connection,
        """
        SELECT count(*)
        FROM (
            SELECT
                linkage.signal_date,
                linkage.permno,
                count(option_links.secid) AS matching_links
            FROM linkage
            LEFT JOIN option_links
              ON linkage.optionmetrics_match_status = 'MATCHED'
             AND linkage.permno = option_links.permno
             AND linkage.optionmetrics_secid = option_links.secid
             AND linkage.signal_date
                 BETWEEN option_links.link_start_date
                     AND option_links.link_end_date
            WHERE linkage.optionmetrics_match_status = 'MATCHED'
            GROUP BY linkage.signal_date, linkage.permno
            HAVING matching_links <> 1
        )
        """,
    )
    _require(
        matched_link_failures == 0,
        "Matched OptionMetrics signal does not resolve to one exact active link",
    )
    calendar_failures = _scalar(
        connection,
        f"""
        SELECT count(*)
        FROM (
            SELECT calendar_date
            FROM calendar_source
            WHERE calendar_date <= DATE '{LAST_SIGNAL_DATE.isoformat()}'
            GROUP BY calendar_date
            HAVING count(*) <> 1
        )
        """,
    )
    _require(calendar_failures == 0, "Development trading calendar is duplicated")
    missing_signal_sessions = _scalar(
        connection,
        f"""
        SELECT count(*)
        FROM (SELECT DISTINCT signal_date FROM spine) AS signals
        LEFT JOIN calendar_source
          ON signals.signal_date = calendar_source.calendar_date
        WHERE calendar_source.calendar_date IS NULL
        """
    )
    _require(
        missing_signal_sessions == 0,
        "A signal date is absent from the verified trading calendar",
    )
    signal_dates = tuple(
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT signal_date FROM spine ORDER BY signal_date"
        ).fetchall()
    )
    _require(
        len(signal_dates) == EXPECTED_SIGNAL_DATES
        and all(isinstance(value, dt.date) for value in signal_dates),
        "Signal-date sequence is invalid",
    )
    return signal_dates


def _prepare_signal_links(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TEMP TABLE option_signal AS
        SELECT
            spine.signal_date,
            spine.permno,
            spine.permco,
            spine.long_eligible,
            spine.short_eligible,
            spine.primary_exchange,
            spine.share_type,
            linkage.optionmetrics_match_status AS match_status,
            linkage.optionmetrics_secid AS secid,
            option_links.link_start_date,
            option_links.link_end_date
        FROM spine
        INNER JOIN linkage USING (signal_date, permno)
        LEFT JOIN option_links
          ON linkage.optionmetrics_match_status = 'MATCHED'
         AND linkage.permno = option_links.permno
         AND linkage.optionmetrics_secid = option_links.secid
         AND linkage.signal_date
             BETWEEN option_links.link_start_date
                 AND option_links.link_end_date
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE matched_option_signal AS
        SELECT *
        FROM option_signal
        WHERE match_status = 'MATCHED'
        """
    )


def _prepare_calendar(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TEMP TABLE development_calendar AS
        SELECT
            calendar_date,
            row_number() OVER (ORDER BY calendar_date) - 1
                AS session_ordinal
        FROM calendar_source
        WHERE calendar_date <= DATE '{LAST_SIGNAL_DATE.isoformat()}'
        """
    )


def _prepare_flow_reductions(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TEMP TABLE complete_option_flow_day AS
        SELECT
            flow.secid,
            flow.observation_date,
            calendar.session_ordinal,
            max(flow.public_availability_date) AS public_availability_date,
            max(
                CASE
                    WHEN flow.row_role = 'CALL'
                     AND flow.volume IS NOT NULL
                     AND flow.volume >= 0
                     AND (
                          flow.volume <> 0
                          OR flow.genuine_zero_volume IS TRUE
                         )
                    THEN CAST(flow.volume AS DOUBLE)
                END
            ) AS call_volume,
            max(
                CASE
                    WHEN flow.row_role = 'PUT'
                     AND flow.volume IS NOT NULL
                     AND flow.volume >= 0
                     AND (
                          flow.volume <> 0
                          OR flow.genuine_zero_volume IS TRUE
                         )
                    THEN CAST(flow.volume AS DOUBLE)
                END
            ) AS put_volume,
            max(
                CASE
                    WHEN flow.row_role = 'CALL'
                     AND flow.open_interest IS NOT NULL
                     AND flow.open_interest >= 0
                     AND (
                          flow.open_interest <> 0
                          OR flow.genuine_zero_open_interest IS TRUE
                         )
                    THEN CAST(flow.open_interest AS DOUBLE)
                END
            ) AS call_open_interest,
            max(
                CASE
                    WHEN flow.row_role = 'PUT'
                     AND flow.open_interest IS NOT NULL
                     AND flow.open_interest >= 0
                     AND (
                          flow.open_interest <> 0
                          OR flow.genuine_zero_open_interest IS TRUE
                         )
                    THEN CAST(flow.open_interest AS DOUBLE)
                END
            ) AS put_open_interest
        FROM option_flow AS flow
        INNER JOIN development_calendar AS calendar
          ON flow.observation_date = calendar.calendar_date
        WHERE flow.authoritative_flow_component IS TRUE
          AND flow.row_role IN ('CALL', 'PUT')
          AND flow.observation_date <= DATE '{LAST_SIGNAL_DATE.isoformat()}'
          AND flow.public_availability_date <= DATE '{LAST_SIGNAL_DATE.isoformat()}'
        GROUP BY
            flow.secid,
            flow.observation_date,
            calendar.session_ordinal
        HAVING count(*) FILTER (WHERE flow.row_role = 'CALL') = 1
           AND count(*) FILTER (WHERE flow.row_role = 'PUT') = 1
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE option_link_first_coverage AS
        WITH active_links AS (
            SELECT DISTINCT
                permno,
                secid,
                link_start_date,
                link_end_date
            FROM matched_option_signal
        )
        SELECT
            links.permno,
            links.secid,
            links.link_start_date,
            links.link_end_date,
            min(flow.observation_date) AS first_complete_date
        FROM active_links AS links
        LEFT JOIN complete_option_flow_day AS flow
          ON links.secid = flow.secid
         AND flow.observation_date
             BETWEEN links.link_start_date AND links.link_end_date
        GROUP BY
            links.permno,
            links.secid,
            links.link_start_date,
            links.link_end_date
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE option_flow_at_signal AS
        WITH rolled AS (
            SELECT
                *,
                count(*) OVER w21 AS pair_count_21,
                min(session_ordinal) OVER w21 AS first_ordinal_21,
                min(observation_date) OVER w21 AS first_date_21,
                count(call_volume) OVER w21 AS call_count_21,
                count(put_volume) OVER w21 AS put_count_21,
                sum(call_volume) OVER w21 AS call_volume_21,
                sum(put_volume) OVER w21 AS put_volume_21,
                max(public_availability_date) OVER w21
                    AS public_availability_21,
                count(*) OVER w63 AS pair_count_63,
                min(session_ordinal) OVER w63 AS first_ordinal_63,
                min(observation_date) OVER w63 AS first_date_63,
                count(call_volume) OVER w63 AS call_count_63,
                count(put_volume) OVER w63 AS put_count_63,
                sum(call_volume) OVER w63 AS call_volume_63,
                sum(put_volume) OVER w63 AS put_volume_63,
                max(public_availability_date) OVER w63
                    AS public_availability_63
            FROM complete_option_flow_day
            WINDOW
                w21 AS (
                    PARTITION BY secid
                    ORDER BY session_ordinal
                    ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                ),
                w63 AS (
                    PARTITION BY secid
                    ORDER BY session_ordinal
                    ROWS BETWEEN 62 PRECEDING AND CURRENT ROW
                )
        )
        SELECT
            signals.signal_date,
            signals.permno,
            rolled.session_ordinal,
            rolled.public_availability_date AS signal_public_availability,
            rolled.call_open_interest,
            rolled.put_open_interest,
            rolled.pair_count_21,
            rolled.first_ordinal_21,
            rolled.first_date_21,
            rolled.call_count_21,
            rolled.put_count_21,
            rolled.call_volume_21,
            rolled.put_volume_21,
            rolled.public_availability_21,
            rolled.pair_count_63,
            rolled.first_ordinal_63,
            rolled.first_date_63,
            rolled.call_count_63,
            rolled.put_count_63,
            rolled.call_volume_63,
            rolled.put_volume_63,
            rolled.public_availability_63
        FROM matched_option_signal AS signals
        LEFT JOIN rolled
          ON signals.secid = rolled.secid
         AND signals.signal_date = rolled.observation_date
        """
    )
    connection.execute("DROP TABLE complete_option_flow_day")


def _prepare_underlying_reduction(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    connection.execute(
        f"""
        CREATE TEMP TABLE option_underlying_at_signal AS
        WITH relevant_permnos AS (
            SELECT DISTINCT permno
            FROM matched_option_signal
        ),
        market_sessions AS (
            SELECT
                market.permno,
                market.calendar_date,
                calendar.session_ordinal,
                CASE
                    WHEN market.volume_shares IS NOT NULL
                     AND market.volume_shares >= 0
                    THEN CAST(market.volume_shares AS DOUBLE)
                END AS volume_shares
            FROM market
            INNER JOIN relevant_permnos USING (permno)
            INNER JOIN development_calendar AS calendar
              ON market.calendar_date = calendar.calendar_date
            WHERE market.calendar_date
                  <= DATE '{LAST_SIGNAL_DATE.isoformat()}'
        ),
        rolled AS (
            SELECT
                *,
                count(*) OVER window_21 AS row_count_21,
                count(volume_shares) OVER window_21 AS volume_count_21,
                min(session_ordinal) OVER window_21 AS first_ordinal_21,
                sum(volume_shares) OVER window_21 AS volume_sum_21
            FROM market_sessions
            WINDOW window_21 AS (
                PARTITION BY permno
                ORDER BY session_ordinal
                ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
            )
        )
        SELECT
            signals.signal_date,
            signals.permno,
            rolled.row_count_21,
            rolled.volume_count_21,
            rolled.first_ordinal_21,
            rolled.session_ordinal,
            rolled.volume_sum_21
        FROM matched_option_signal AS signals
        LEFT JOIN rolled
          ON signals.permno = rolled.permno
         AND signals.signal_date = rolled.calendar_date
        """
    )


def _prepare_historical_volatility_reduction(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    connection.execute(
        f"""
        CREATE TEMP TABLE option_hv_at_signal AS
        SELECT
            signals.signal_date,
            signals.permno,
            max(
                CASE WHEN hv.days = 30 AND hv.volatility >= 0
                     THEN CAST(hv.volatility AS DOUBLE) END
            ) AS hv_30,
            max(
                CASE WHEN hv.days = 60 AND hv.volatility >= 0
                     THEN CAST(hv.volatility AS DOUBLE) END
            ) AS hv_60,
            max(
                CASE WHEN hv.days = 91 AND hv.volatility >= 0
                     THEN CAST(hv.volatility AS DOUBLE) END
            ) AS hv_91,
            max(
                CASE WHEN hv.days = 182 AND hv.volatility >= 0
                     THEN CAST(hv.volatility AS DOUBLE) END
            ) AS hv_182,
            max(
                CASE WHEN hv.days = 365 AND hv.volatility >= 0
                     THEN CAST(hv.volatility AS DOUBLE) END
            ) AS hv_365,
            max(
                CASE WHEN hv.days = 30 AND hv.volatility >= 0
                     THEN hv.public_availability_date END
            ) AS hv_public_30,
            max(
                CASE WHEN hv.days = 60 AND hv.volatility >= 0
                     THEN hv.public_availability_date END
            ) AS hv_public_60,
            max(
                CASE WHEN hv.days = 91 AND hv.volatility >= 0
                     THEN hv.public_availability_date END
            ) AS hv_public_91,
            max(
                CASE WHEN hv.days = 182 AND hv.volatility >= 0
                     THEN hv.public_availability_date END
            ) AS hv_public_182,
            max(
                CASE WHEN hv.days = 365 AND hv.volatility >= 0
                     THEN hv.public_availability_date END
            ) AS hv_public_365
        FROM matched_option_signal AS signals
        LEFT JOIN option_hv AS hv
          ON signals.secid = hv.secid
         AND signals.signal_date = hv.observation_date
         AND hv.supported_horizon IS TRUE
         AND hv.days IN (30, 60, 91, 182, 365)
         AND hv.observation_date <= DATE '{LAST_SIGNAL_DATE.isoformat()}'
         AND hv.public_availability_date <= signals.signal_date
        GROUP BY signals.signal_date, signals.permno
        """
    )


def _compact_reader(
    connection: duckdb.DuckDBPyConnection,
    batch_size: int,
):
    return connection.execute(
        """
        SELECT
            signals.*,
            flow.session_ordinal AS flow_session_ordinal,
            flow.signal_public_availability,
            flow.call_open_interest,
            flow.put_open_interest,
            flow.pair_count_21,
            flow.first_ordinal_21 AS flow_first_ordinal_21,
            flow.first_date_21 AS flow_first_date_21,
            flow.call_count_21,
            flow.put_count_21,
            flow.call_volume_21,
            flow.put_volume_21,
            flow.public_availability_21,
            flow.pair_count_63,
            flow.first_ordinal_63 AS flow_first_ordinal_63,
            flow.first_date_63 AS flow_first_date_63,
            flow.call_count_63,
            flow.put_count_63,
            flow.call_volume_63,
            flow.put_volume_63,
            flow.public_availability_63,
            coverage.first_complete_date,
            underlying.row_count_21 AS underlying_row_count_21,
            underlying.volume_count_21 AS underlying_volume_count_21,
            underlying.first_ordinal_21 AS underlying_first_ordinal_21,
            underlying.session_ordinal AS underlying_session_ordinal,
            underlying.volume_sum_21,
            hv.hv_30,
            hv.hv_60,
            hv.hv_91,
            hv.hv_182,
            hv.hv_365,
            hv.hv_public_30,
            hv.hv_public_60,
            hv.hv_public_91,
            hv.hv_public_182,
            hv.hv_public_365
        FROM option_signal AS signals
        LEFT JOIN option_flow_at_signal AS flow
          USING (signal_date, permno)
        LEFT JOIN option_link_first_coverage AS coverage
          ON signals.permno = coverage.permno
         AND signals.secid = coverage.secid
         AND signals.link_start_date = coverage.link_start_date
         AND signals.link_end_date = coverage.link_end_date
        LEFT JOIN option_underlying_at_signal AS underlying
          USING (signal_date, permno)
        LEFT JOIN option_hv_at_signal AS hv
          USING (signal_date, permno)
        ORDER BY signals.signal_date, signals.permno
        """
    ).fetch_record_batch(rows_per_batch=batch_size)


@dataclass(frozen=True)
class _Endpoint:
    combined_volume: float | None
    volume_observation: dt.date | None
    volume_availability: dt.date | None
    combined_open_interest: float | None
    open_interest_observation: dt.date | None
    open_interest_availability: dt.date | None


def _lineage(
    value: float | int | None,
    observation: dt.date | None,
    availability: dt.date | None,
    stale_after: dt.date | None,
    missing_status: str,
) -> FeatureLineage:
    if value is None:
        return FeatureLineage(None, None, None, missing_status)
    _require(
        isinstance(observation, dt.date)
        and isinstance(availability, dt.date),
        "Present OptionMetrics value lacks lineage dates",
    )
    return FeatureLineage(
        observation,
        availability,
        stale_after,
        "PRESENT",
    )


def _empty_result(status: str) -> tuple[FeatureFamilyResult, _Endpoint]:
    values = {name: None for name in FEATURE_NAMES}
    lineage = {
        name: FeatureLineage(None, None, None, status)
        for name in FEATURE_NAMES
    }
    return (
        FeatureFamilyResult(values=values, lineage=lineage),
        _Endpoint(None, None, None, None, None, None),
    )


def _completed_months(start: dt.date, end: dt.date) -> int:
    return max(0, (end.year - start.year) * 12 + end.month - start.month)


def _build_result(
    row: Mapping[str, Any],
    prior_endpoints: Mapping[int, tuple[dt.date, _Endpoint] | None],
) -> tuple[FeatureFamilyResult, _Endpoint]:
    status = str(row["match_status"])
    if status != "MATCHED":
        return _empty_result(
            "MISSING_AMBIGUOUS_LINK"
            if status == "AMBIGUOUS"
            else "MISSING_NOT_COVERED"
        )

    signal = _date(row["signal_date"], "OptionMetrics signal_date")
    link_start = _date(row["link_start_date"], "OptionMetrics link start")
    link_end = _date(row["link_end_date"], "OptionMetrics link end")
    _require(
        link_start <= signal <= link_end,
        "Matched OptionMetrics signal escaped its active link",
    )
    values: dict[str, float | int | None] = {
        name: None for name in FEATURE_NAMES
    }
    lineage: dict[str, FeatureLineage] = {
        name: FeatureLineage(
            None,
            None,
            None,
            "MISSING_INSUFFICIENT_HISTORY",
        )
        for name in FEATURE_NAMES
    }

    flow_ordinal = row.get("flow_session_ordinal")
    complete_21 = (
        flow_ordinal is not None
        and row.get("pair_count_21") == 21
        and row.get("call_count_21") == 21
        and row.get("put_count_21") == 21
        and row.get("flow_first_ordinal_21") == flow_ordinal - 20
        and row.get("flow_first_date_21") >= link_start
    )
    complete_63 = (
        flow_ordinal is not None
        and row.get("pair_count_63") == 63
        and row.get("call_count_63") == 63
        and row.get("put_count_63") == 63
        and row.get("flow_first_ordinal_63") == flow_ordinal - 62
        and row.get("flow_first_date_63") >= link_start
    )
    for length, complete in ((21, complete_21), (63, complete_63)):
        public = row.get(f"public_availability_{length}")
        call = (
            _number(row.get(f"call_volume_{length}"))
            if complete
            else None
        )
        put = (
            _number(row.get(f"put_volume_{length}"))
            if complete
            else None
        )
        if call is None or put is None:
            call = put = None
        for name, value in (
            (f"option_call_volume_{length}d", call),
            (f"option_put_volume_{length}d", put),
            (
                f"option_combined_volume_{length}d",
                None if call is None else call + put,
            ),
        ):
            values[name] = value
            lineage[name] = _lineage(
                value,
                signal,
                public,
                link_end,
                "MISSING_INSUFFICIENT_HISTORY",
            )

    signal_pair_present = flow_ordinal is not None
    call_oi = (
        _number(row.get("call_open_interest"))
        if signal_pair_present
        else None
    )
    put_oi = (
        _number(row.get("put_open_interest"))
        if signal_pair_present
        else None
    )
    if call_oi is None or put_oi is None:
        call_oi = put_oi = None
    signal_public = row.get("signal_public_availability")
    for name, value in (
        ("option_call_open_interest_eom", call_oi),
        ("option_put_open_interest_eom", put_oi),
        (
            "option_combined_open_interest_eom",
            None if call_oi is None else call_oi + put_oi,
        ),
    ):
        values[name] = value
        lineage[name] = _lineage(
            value,
            signal,
            signal_public,
            link_end,
            "MISSING_INVALID_INPUT",
        )

    call_21 = values["option_call_volume_21d"]
    put_21 = values["option_put_volume_21d"]
    if call_21 is not None and put_21 is not None:
        combined_21 = float(call_21) + float(put_21)
        public_21 = row.get("public_availability_21")
        volume_ratio = math.log(
            (float(put_21) + 1.0) / (float(call_21) + 1.0)
        )
        call_share = (
            0.0
            if combined_21 == 0.0
            else (float(call_21) - float(put_21)) / combined_21
        )
        for name, value in (
            ("log_option_put_call_volume_ratio_21d", volume_ratio),
            ("option_call_minus_put_volume_share_21d", call_share),
        ):
            values[name] = value
            lineage[name] = _lineage(
                value,
                signal,
                public_21,
                link_end,
                "MISSING_INVALID_INPUT",
            )
        underlying_complete = (
            row.get("underlying_row_count_21") == 21
            and row.get("underlying_volume_count_21") == 21
            and row.get("underlying_session_ordinal") is not None
            and row.get("underlying_first_ordinal_21")
            == row.get("underlying_session_ordinal") - 20
        )
        underlying = (
            _number(row.get("volume_sum_21"))
            if underlying_complete
            else None
        )
        option_underlying_ratio = (
            None
            if underlying is None or underlying <= 0.0
            else combined_21 / underlying
        )
        name = "option_volume_to_underlying_volume_21d"
        values[name] = option_underlying_ratio
        lineage[name] = _lineage(
            option_underlying_ratio,
            signal,
            public_21,
            link_end,
            "MISSING_INVALID_INPUT",
        )

    if call_oi is not None and put_oi is not None:
        name = "log_option_put_call_open_interest_ratio_eom"
        value = math.log((put_oi + 1.0) / (call_oi + 1.0))
        values[name] = value
        lineage[name] = _lineage(
            value,
            signal,
            signal_public,
            link_end,
            "MISSING_INVALID_INPUT",
        )

    coverage = 1 if signal_pair_present else None
    values["optionmetrics_covered_indicator"] = coverage
    lineage["optionmetrics_covered_indicator"] = _lineage(
        coverage,
        signal,
        signal_public,
        link_end,
        "MISSING_INSUFFICIENT_HISTORY",
    )
    first_complete = row.get("first_complete_date")
    age = (
        _completed_months(first_complete, signal)
        if coverage is not None
        and isinstance(first_complete, dt.date)
        and link_start <= first_complete <= signal
        else None
    )
    values["optionmetrics_coverage_age_months"] = age
    lineage["optionmetrics_coverage_age_months"] = _lineage(
        age,
        signal,
        signal_public,
        link_end,
        "MISSING_INSUFFICIENT_HISTORY",
    )

    endpoint = _Endpoint(
        combined_volume=(
            None
            if values["option_combined_volume_21d"] is None
            else float(values["option_combined_volume_21d"])
        ),
        volume_observation=lineage[
            "option_combined_volume_21d"
        ].max_observation_date,
        volume_availability=lineage[
            "option_combined_volume_21d"
        ].max_public_availability_date,
        combined_open_interest=(
            None
            if values["option_combined_open_interest_eom"] is None
            else float(values["option_combined_open_interest_eom"])
        ),
        open_interest_observation=lineage[
            "option_combined_open_interest_eom"
        ].max_observation_date,
        open_interest_availability=lineage[
            "option_combined_open_interest_eom"
        ].max_public_availability_date,
    )
    for lag in (1, 3):
        prior_item = prior_endpoints.get(lag)
        for source, feature in (
            (
                "volume",
                f"option_total_volume_change_{lag}m",
            ),
            (
                "open_interest",
                f"option_total_open_interest_change_{lag}m",
            ),
        ):
            current_value = getattr(endpoint, f"combined_{source}")
            current_observation = getattr(
                endpoint, f"{source}_observation"
            )
            current_availability = getattr(
                endpoint, f"{source}_availability"
            )
            prior_signal: dt.date | None = None
            prior_value: float | None = None
            prior_availability: dt.date | None = None
            if prior_item is not None:
                prior_signal, prior = prior_item
                prior_value = getattr(prior, f"combined_{source}")
                prior_availability = getattr(
                    prior, f"{source}_availability"
                )
            value = (
                None
                if current_value is None
                or prior_value is None
                or prior_availability is None
                or prior_availability > signal
                else math.log1p(current_value) - math.log1p(prior_value)
            )
            observation = (
                None
                if value is None
                else max(current_observation, prior_signal)
            )
            availability = (
                None
                if value is None
                else max(current_availability, prior_availability)
            )
            values[feature] = value
            lineage[feature] = _lineage(
                value,
                observation,
                availability,
                link_end,
                "MISSING_INSUFFICIENT_HISTORY",
            )

    for nominal, vendor in (
        (30, 30),
        (60, 60),
        (90, 91),
        (180, 182),
        (360, 365),
    ):
        name = f"option_historical_volatility_{nominal}d"
        value = _number(row.get(f"hv_{vendor}"))
        public = row.get(f"hv_public_{vendor}")
        values[name] = value
        lineage[name] = _lineage(
            value,
            signal,
            public,
            link_end,
            "MISSING_INVALID_INPUT",
        )
    for name, left, right in (
        (
            "option_historical_volatility_30d_minus_90d",
            "option_historical_volatility_30d",
            "option_historical_volatility_90d",
        ),
        (
            "option_historical_volatility_60d_minus_180d",
            "option_historical_volatility_60d",
            "option_historical_volatility_180d",
        ),
    ):
        value = (
            None
            if values[left] is None or values[right] is None
            else float(values[left]) - float(values[right])
        )
        observations = (
            lineage[left].max_observation_date,
            lineage[right].max_observation_date,
        )
        availability = (
            lineage[left].max_public_availability_date,
            lineage[right].max_public_availability_date,
        )
        values[name] = value
        lineage[name] = _lineage(
            value,
            None if value is None else max(observations),
            None if value is None else max(availability),
            link_end,
            "MISSING_INVALID_INPUT",
        )

    ordered_values = {name: values[name] for name in FEATURE_NAMES}
    ordered_lineage = {name: lineage[name] for name in FEATURE_NAMES}
    _require(
        not any(
            isinstance(value, float) and not math.isfinite(value)
            for value in ordered_values.values()
        ),
        "Nonfinite OptionMetrics value escaped",
    )
    for name, item in ordered_lineage.items():
        if item.value_status == "PRESENT":
            _require(
                item.max_observation_date <= signal
                and item.max_public_availability_date <= signal
                and (
                    item.stale_after_date is None
                    or signal <= item.stale_after_date
                ),
                f"Future or stale OptionMetrics lineage escaped: {name}",
            )
    return (
        FeatureFamilyResult(
            values=ordered_values,
            lineage=ordered_lineage,
        ),
        endpoint,
    )


def _prior_signal_map(
    signal_dates: Sequence[dt.date],
) -> Mapping[dt.date, Mapping[int, dt.date | None]]:
    by_month: dict[tuple[int, int], dt.date] = {}
    for value in signal_dates:
        key = (value.year, value.month)
        _require(key not in by_month, "Multiple signal dates occupy one month")
        by_month[key] = value
    result: dict[dt.date, dict[int, dt.date | None]] = {}
    for value in signal_dates:
        ordinal = value.year * 12 + value.month - 1
        result[value] = {}
        for lag in (1, 3):
            prior_ordinal = ordinal - lag
            result[value][lag] = by_month.get(
                (prior_ordinal // 12, prior_ordinal % 12 + 1)
            )
    return result


def iter_optionmetrics_feature_batches(
    inputs: FeatureStoreInputs,
    *,
    batch_size: int = 20_000,
    policy_path: Path = DEFAULT_POLICY,
    build_identity: FeatureStoreBuildIdentity | None = None,
) -> Iterator[tuple[FeatureMaterializationRow, ...]]:
    """Yield all 26 OptionMetrics features in exact signal-spine order."""

    _require(
        isinstance(batch_size, int)
        and not isinstance(batch_size, bool)
        and batch_size > 0,
        "batch_size must be a positive integer",
    )
    if build_identity is None:
        authenticate_feature_store_inputs(inputs, policy_path)
    else:
        validate_precomputed_feature_store_build_identity(
            inputs,
            build_identity,
            policy_path,
        )
    for table, columns in (
        (SPINE_TABLE, _SPINE_COLUMNS),
        (LINKAGE_TABLE, _LINKAGE_COLUMNS),
        (OPTION_LINK_TABLE, _OPTION_LINK_COLUMNS),
        (FLOW_TABLE, _FLOW_COLUMNS),
        (HISTORICAL_VOLATILITY_TABLE, _HV_COLUMNS),
        (MARKET_TABLE, _MARKET_COLUMNS),
        (CALENDAR_TABLE, _CALENDAR_COLUMNS),
    ):
        _required_columns(inputs.path(table), columns, table)

    def generate() -> Iterator[tuple[FeatureMaterializationRow, ...]]:
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute("PRAGMA threads=1")
            connection.execute("SET preserve_insertion_order=false")
            _create_views(connection, inputs)
            signal_dates = _validate_source_gates(connection, inputs)
            priors = _prior_signal_map(signal_dates)
            _prepare_calendar(connection)
            _prepare_signal_links(connection)
            _prepare_flow_reductions(connection)
            _prepare_underlying_reduction(connection)
            _prepare_historical_volatility_reduction(connection)

            endpoint_months: dict[
                dt.date, dict[int, _Endpoint]
            ] = {}
            active_signal: dt.date | None = None
            active_endpoints: dict[int, _Endpoint] = {}
            previous_key: tuple[dt.date, int] | None = None
            produced = 0
            pending: list[FeatureMaterializationRow] = []

            for record_batch in _compact_reader(connection, batch_size):
                for compact in record_batch.to_pylist():
                    signal = _date(
                        compact["signal_date"],
                        "OptionMetrics signal_date",
                    )
                    permno = _integer(
                        compact["permno"],
                        "OptionMetrics permno",
                    )
                    key = (signal, permno)
                    _require(
                        previous_key is None or key > previous_key,
                        "Compact OptionMetrics rows are not sorted and unique",
                    )
                    if active_signal is None:
                        active_signal = signal
                    elif signal != active_signal:
                        endpoint_months[active_signal] = active_endpoints
                        active_endpoints = {}
                        active_signal = signal
                        signal_ordinal = signal.year * 12 + signal.month - 1
                        endpoint_months = {
                            month: values
                            for month, values in endpoint_months.items()
                            if 0
                            < signal_ordinal
                            - (month.year * 12 + month.month - 1)
                            <= 3
                        }
                    prior_endpoints = {
                        lag: (
                            None
                            if (prior_signal := priors[signal][lag]) is None
                            else (
                                prior_signal,
                                endpoint_months.get(
                                    prior_signal, {}
                                ).get(permno),
                            )
                        )
                        for lag in (1, 3)
                    }
                    prior_endpoints = {
                        lag: (
                            None
                            if item is None or item[1] is None
                            else (item[0], item[1])
                        )
                        for lag, item in prior_endpoints.items()
                    }
                    result, endpoint = _build_result(
                        compact,
                        prior_endpoints,
                    )
                    active_endpoints[permno] = endpoint
                    _require(
                        isinstance(compact["long_eligible"], bool)
                        and isinstance(compact["short_eligible"], bool),
                        "Spine eligibility flags must be Boolean",
                    )
                    pending.append(
                        FeatureMaterializationRow(
                            signal_date=signal,
                            permno=permno,
                            permco=_integer(
                                compact["permco"],
                                "OptionMetrics permco",
                            ),
                            long_eligible=compact["long_eligible"],
                            short_eligible=compact["short_eligible"],
                            primary_exchange=str(
                                compact["primary_exchange"]
                            ),
                            share_type=str(compact["share_type"]),
                            result=result,
                        )
                    )
                    previous_key = key
                    if len(pending) == batch_size:
                        produced += len(pending)
                        yield tuple(pending)
                        pending = []
            if pending:
                produced += len(pending)
                yield tuple(pending)
            _require(
                produced == inputs.tables[SPINE_TABLE].rows,
                "OptionMetrics adapter did not preserve every spine row",
            )
        finally:
            connection.close()

    return generate()


# Concise orchestration alias.
materialize_optionmetrics = iter_optionmetrics_feature_batches
