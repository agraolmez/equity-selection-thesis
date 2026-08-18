#!/usr/bin/env python3
"""Publish the approved 15-feature V6 engineering extension.

This outcome-blind build reads the immutable 303-feature V6 panel and its
authenticated CRSP/Compustat inputs, then writes a separate 318-feature V2
lineage.  It never reads labels, predictions, model results, or portfolios.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from portfolio_bot.research_v6.engineered_features import (  # noqa: E402
    ENGINEERED_FEATURE_NAMES,
    ENGINEERED_FEATURE_SPECS,
    PANEL_INPUT_COLUMNS,
    derive_panel_engineered_features,
)


DATABASE = Path("/Users/agraemirolmez/Desktop/Thesis")
BASE = DATABASE / "processed/research_v6/model_features_v1"
OUTPUT = DATABASE / "processed/research_v6/model_features_v2"
UNIVERSE = DATABASE / "raw/research_v6/lseg/input_universe.parquet"

REPLAY = DATABASE / (
    "cache/phase_06_final_test/stage13-final-source-replay-v1/"
    "dbccbf9cd220d890d8c025bb6e92854bf61098f1eb42517da934c62b98e7adf5/"
    "formula_work/feature_sources"
)
BUNDLE = DATABASE / (
    "cache/phase_06_final_test/stage13-final-source-replay-v1/"
    "082c42dd57e8077f47cf409a12aefd8ec2ee98b732839496a320f8f84abeb66e/"
    "formula_work/feature_sources/feature_input_bundle"
)
CRSP_MARKET = REPLAY / "full_predictor_context/tables/crsp_security_day.parquet"
CRSP_SHARES = REPLAY / "full_predictor_context/tables/crsp_shares_interval.parquet"
CRSP_RECEIPT = REPLAY / "full_predictor_context/full_predictor_context_receipt.json"
TRADING_CALENDAR = BUNDLE / "tables/trading_calendar.parquet"
QUARTERLY = BUNDLE / "tables/compustat_quarterly_observation.parquet"
EARLY_LINKAGE = DATABASE / (
    "interim/phase_05_validation/stage11-validation-data-v1/feature_input_bundle/"
    "tables/optionmetrics_working_auxiliary_signal_linkage.parquet"
)
LATE_LINKAGE = BUNDLE / "tables/optionmetrics_working_auxiliary_signal_linkage.parquet"

KEYS = ["holding_month", "signal_date", "permno", "permco"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _read_parquet(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Read one physical Parquet file with a fallback for legacy page metadata."""

    try:
        return pq.ParquetFile(path).read(columns=columns).to_pandas()
    except OSError as exc:
        if "Repetition level histogram size mismatch" not in str(exc):
            raise
        selected = "*" if columns is None else ", ".join(
            f'"{column}"' for column in columns
        )
        connection = duckdb.connect(database=":memory:")
        try:
            return connection.execute(
                f"SELECT {selected} FROM read_parquet('{_sql_path(path)}')"
            ).fetch_df()
        finally:
            connection.close()


def _load_panel_inputs(months: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in months:
        source = BASE / "months" / f"{month}.parquet"
        parts.append(_read_parquet(source, columns=list(PANEL_INPUT_COLUMNS)))
    panel = pd.concat(parts, ignore_index=True)
    return panel.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def _turnover_features(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = f"""
        WITH dates AS (
            SELECT
                calendar_date,
                row_number() OVER (ORDER BY calendar_date) AS session_id
            FROM read_parquet('{_sql_path(TRADING_CALENDAR)}')
        ),
        signals AS (
            SELECT
                universe.holding_month,
                universe.signal_date,
                universe.permno,
                universe.permco,
                dates.session_id
            FROM read_parquet('{_sql_path(UNIVERSE)}') AS universe
            INNER JOIN dates USING (calendar_date)
        ),
        contributors AS (
            SELECT
                signals.holding_month,
                signals.signal_date,
                signals.permno,
                signals.permco,
                dates.calendar_date,
                market.volume_shares,
                shares.shares_outstanding,
                CASE
                    WHEN market.volume_shares >= 0
                     AND shares.shares_outstanding > 0
                    THEN market.volume_shares::DOUBLE
                         / shares.shares_outstanding::DOUBLE
                END AS daily_turnover
            FROM signals
            INNER JOIN dates
              ON dates.session_id BETWEEN signals.session_id - 62
                                      AND signals.session_id
            LEFT JOIN read_parquet('{_sql_path(CRSP_MARKET)}') AS market
              ON market.permno = signals.permno
             AND market.calendar_date = dates.calendar_date
            LEFT JOIN read_parquet('{_sql_path(CRSP_SHARES)}') AS shares
              ON shares.permno = signals.permno
             AND dates.calendar_date BETWEEN shares.start_date AND shares.end_date
        )
        SELECT
            holding_month,
            signal_date,
            permno,
            permco,
            CASE
                WHEN count(*) = 63
                 AND count(daily_turnover) = 63
                THEN stddev_samp(daily_turnover)
            END AS share_turnover_volatility_63d
        FROM contributors
        GROUP BY holding_month, signal_date, permno, permco
        ORDER BY holding_month, permno
    """
    # DuckDB uses ``calendar_date`` as the join name.  The universe calls the
    # same date ``signal_date``, so make that equivalence explicit here.
    query = query.replace(
        "INNER JOIN dates USING (calendar_date)",
        "INNER JOIN dates ON universe.signal_date = dates.calendar_date",
    )
    return connection.execute(query).fetch_df()


def _fundamental_features(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = f"""
        WITH universe AS (
            SELECT
                holding_month,
                signal_date,
                permno,
                permco,
                market_cap_usd::DOUBLE AS market_cap_usd
            FROM read_parquet('{_sql_path(UNIVERSE)}')
        ),
        linkage AS (
            SELECT signal_date, permno, ccm_match_status, ccm_gvkey
            FROM read_parquet('{_sql_path(EARLY_LINKAGE)}')
            UNION ALL
            SELECT signal_date, permno, ccm_match_status, ccm_gvkey
            FROM read_parquet('{_sql_path(LATE_LINKAGE)}')
        ),
        matched AS (
            SELECT universe.*, linkage.ccm_gvkey
            FROM universe
            INNER JOIN linkage USING (signal_date, permno)
            WHERE linkage.ccm_match_status = 'MATCHED'
        ),
        known_versions AS (
            SELECT
                matched.*,
                quarterly.source_record_id,
                quarterly.fyearq::INTEGER AS fyearq,
                quarterly.fqtr::INTEGER AS fqtr,
                quarterly.datadate,
                quarterly.public_availability_date,
                quarterly.stale_after_date,
                quarterly.currency_code,
                quarterly.ibq::DOUBLE AS ibq,
                quarterly.saleq::DOUBLE AS saleq,
                row_number() OVER (
                    PARTITION BY matched.signal_date, matched.permno,
                                 quarterly.fyearq, quarterly.fqtr
                    ORDER BY quarterly.public_availability_date DESC,
                             quarterly.datadate DESC,
                             quarterly.source_record_id DESC
                ) AS version_rank
            FROM matched
            INNER JOIN read_parquet('{_sql_path(QUARTERLY)}') AS quarterly
              ON quarterly.gvkey = matched.ccm_gvkey
             AND quarterly.permno = matched.permno
            WHERE quarterly.standard_format_eligible IS TRUE
              AND quarterly.observation_date <= matched.signal_date
              AND quarterly.public_availability_date <= matched.signal_date
        ),
        resolved AS (
            SELECT
                * EXCLUDE (version_rank),
                fyearq * 4 + fqtr - 1 AS fiscal_ordinal
            FROM known_versions
            WHERE version_rank = 1
        ),
        anchors AS (
            SELECT * EXCLUDE (anchor_rank)
            FROM (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY signal_date, permno
                        ORDER BY fiscal_ordinal DESC,
                                 public_availability_date DESC,
                                 datadate DESC,
                                 source_record_id DESC
                    ) AS anchor_rank
                FROM resolved
                WHERE signal_date <= stale_after_date
            )
            WHERE anchor_rank = 1
        ),
        base AS (
            SELECT
                anchor.holding_month,
                anchor.signal_date,
                anchor.permno,
                anchor.permco,
                anchor.fiscal_ordinal AS anchor_ordinal,
                CASE
                    WHEN anchor.currency_code = 'USD'
                     AND prior.currency_code = 'USD'
                     AND anchor.market_cap_usd > 0
                     AND anchor.ibq IS NOT NULL
                     AND prior.ibq IS NOT NULL
                    THEN (anchor.ibq - prior.ibq) * 1000000.0
                         / anchor.market_cap_usd
                END AS quarterly_earnings_change_to_market_cap,
                CASE
                    WHEN anchor.currency_code = 'USD'
                     AND prior.currency_code = 'USD'
                     AND anchor.market_cap_usd > 0
                     AND anchor.saleq IS NOT NULL
                     AND prior.saleq IS NOT NULL
                    THEN (anchor.saleq - prior.saleq) * 1000000.0
                         / anchor.market_cap_usd
                END AS quarterly_sales_change_to_market_cap,
                date_diff(
                    'day', anchor.public_availability_date, anchor.signal_date
                )::DOUBLE AS days_since_latest_quarterly_report
            FROM anchors AS anchor
            LEFT JOIN resolved AS prior
              ON prior.signal_date = anchor.signal_date
             AND prior.permno = anchor.permno
             AND prior.fiscal_ordinal = anchor.fiscal_ordinal - 4
        ),
        comparison_slots AS (
            SELECT
                base.*,
                slot.k,
                CASE
                    WHEN current.currency_code = 'USD'
                     AND prior.currency_code = 'USD'
                     AND current.ibq IS NOT NULL
                     AND prior.ibq IS NOT NULL
                    THEN current.ibq > prior.ibq
                END AS increased
            FROM base
            CROSS JOIN range(8) AS slot(k)
            LEFT JOIN resolved AS current
              ON current.signal_date = base.signal_date
             AND current.permno = base.permno
             AND current.fiscal_ordinal = base.anchor_ordinal - slot.k
            LEFT JOIN resolved AS prior
              ON prior.signal_date = base.signal_date
             AND prior.permno = base.permno
             AND prior.fiscal_ordinal = base.anchor_ordinal - slot.k - 4
        )
        SELECT
            holding_month,
            signal_date,
            permno,
            permco,
            any_value(quarterly_earnings_change_to_market_cap)
                AS quarterly_earnings_change_to_market_cap,
            any_value(quarterly_sales_change_to_market_cap)
                AS quarterly_sales_change_to_market_cap,
            CASE
                WHEN max(
                    CASE WHEN k = 0 AND increased IS NOT NULL THEN 1 ELSE 0 END
                ) = 0
                THEN NULL
                ELSE coalesce(
                    min(k) FILTER (WHERE increased IS DISTINCT FROM TRUE), 8
                )::DOUBLE
            END AS consecutive_quarterly_earnings_increases,
            any_value(days_since_latest_quarterly_report)
                AS days_since_latest_quarterly_report
        FROM comparison_slots
        GROUP BY holding_month, signal_date, permno, permco
        ORDER BY holding_month, permno
    """
    return connection.execute(query).fetch_df()


def _engineered_registry(base_registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for specification in ENGINEERED_FEATURE_SPECS:
        rows.append(
            {
                "model_ordinal": len(base_registry) + len(rows) + 1,
                "feature_name": specification["feature_name"],
                "family": specification["family"],
                "kind": "continuous_firm",
                "source": "v6_engineered_features_v1",
                "source_column": specification["feature_name"],
                "unit": specification["unit"],
                "transformation": specification["transformation"],
                "model_status": "INCLUDED",
                "exclusion_reason": pd.NA,
                "primary_full": True,
                "cam_free": True,
                "cam_only": False,
            }
        )
    output = pd.concat([base_registry, pd.DataFrame(rows)], ignore_index=True)
    if len(output) != 318 or output["feature_name"].duplicated().any():
        raise RuntimeError("engineered registry identity differs")
    return output


def main() -> int:
    base_manifest = json.loads((BASE / "manifest.json").read_text())
    if (
        base_manifest.get("status") != "BACKTEST_READY"
        or int(base_manifest.get("model_features_primary_full", -1)) != 303
        or int(base_manifest.get("months", -1)) != 128
    ):
        raise RuntimeError("base V6 feature authority differs")
    months = sorted(
        _read_parquet(UNIVERSE, columns=["holding_month"])["holding_month"]
        .astype(str)
        .unique()
    )
    if len(months) != 128 or months[0] != "2015-01" or months[-1] != "2025-08":
        raise RuntimeError("V6 universe month scope differs")

    replay_receipt = json.loads(CRSP_RECEIPT.read_text())
    source_identity = {
        "base_build_identity": base_manifest["build_identity"],
        "script_sha256": _sha256(Path(__file__).resolve()),
        "transformation_module_sha256": _sha256(
            SOURCE / "portfolio_bot/research_v6/engineered_features.py"
        ),
        "universe_sha256": _sha256(UNIVERSE),
        "crsp_market_sha256": replay_receipt["tables"]["crsp_security_day.parquet"][
            "sha256"
        ],
        "crsp_shares_sha256": replay_receipt["tables"]["crsp_shares_interval.parquet"][
            "sha256"
        ],
        "trading_calendar_sha256": _sha256(TRADING_CALENDAR),
        "quarterly_sha256": _sha256(QUARTERLY),
        "early_linkage_sha256": _sha256(EARLY_LINKAGE),
        "late_linkage_sha256": _sha256(LATE_LINKAGE),
    }
    build_identity = hashlib.sha256(
        json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    panel_inputs = _load_panel_inputs(months)
    derived_panel = derive_panel_engineered_features(panel_inputs)
    _atomic_parquet(derived_panel, OUTPUT / "work/panel_derived_features.parquet")

    connection = duckdb.connect(database=":memory:")
    connection.execute("SET threads=6")
    connection.execute("SET memory_limit='8GB'")
    try:
        turnover = _turnover_features(connection)
        fundamentals = _fundamental_features(connection)
    finally:
        connection.close()
    _atomic_parquet(turnover, OUTPUT / "work/turnover_features.parquet")
    _atomic_parquet(fundamentals, OUTPUT / "work/fundamental_momentum_features.parquet")

    universe = _read_parquet(UNIVERSE, columns=KEYS)
    universe["holding_month"] = universe["holding_month"].astype(str)
    universe["signal_date"] = pd.to_datetime(universe["signal_date"], errors="raise")
    for frame in (derived_panel, turnover, fundamentals):
        frame["holding_month"] = frame["holding_month"].astype(str)
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise")
        if frame.duplicated(KEYS).any():
            raise RuntimeError("engineered source has duplicate stock-month keys")
    engineered = universe.merge(derived_panel, on=KEYS, how="left", validate="one_to_one")
    engineered = engineered.merge(turnover, on=KEYS, how="left", validate="one_to_one")
    engineered = engineered.merge(fundamentals, on=KEYS, how="left", validate="one_to_one")
    if len(engineered) != len(universe):
        raise RuntimeError("engineered block does not preserve the V6 universe")
    engineered = engineered[[*KEYS, *ENGINEERED_FEATURE_NAMES]]
    numeric = engineered[list(ENGINEERED_FEATURE_NAMES)].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise RuntimeError("engineered block contains infinity")
    if engineered["days_since_latest_quarterly_report"].dropna().lt(0).any():
        raise RuntimeError("future quarterly report escaped the point-in-time gate")
    _atomic_parquet(engineered, OUTPUT / "engineered_feature_block.parquet")

    base_registry = pd.read_csv(BASE / "feature_registry.csv")
    registry = _engineered_registry(base_registry)
    base_audit = _read_parquet(BASE / "feature_audit_registry.parquet")
    new_audit = registry.loc[
        registry["source"].eq("v6_engineered_features_v1"),
        [column for column in base_audit.columns if column in registry.columns],
    ]
    audit_registry = pd.concat([base_audit, new_audit], ignore_index=True)
    _atomic_parquet(registry, OUTPUT / "feature_registry.parquet")
    _atomic_parquet(audit_registry, OUTPUT / "feature_audit_registry.parquet")
    _atomic_csv(registry, OUTPUT / "feature_registry.csv")

    coverage_parts: list[pd.DataFrame] = []
    for ordinal, month in enumerate(months, start=1):
        base_path = BASE / "months" / f"{month}.parquet"
        base_month = _read_parquet(base_path)
        extra_month = engineered.loc[engineered["holding_month"].eq(month)]
        result = base_month.merge(extra_month, on=KEYS, how="left", validate="one_to_one")
        expected = registry["feature_name"].astype(str).tolist()
        if len(result) != len(base_month) or set(expected).difference(result.columns):
            raise RuntimeError(f"{month}: engineered month identity differs")
        result = result[[*KEYS, *expected]].sort_values("permno", kind="mergesort")
        if np.isinf(
            result.select_dtypes(include=[np.number]).to_numpy(
                dtype=float, na_value=np.nan
            )
        ).any():
            raise RuntimeError(f"{month}: infinity escaped")

        destination = OUTPUT / "months" / f"{month}.parquet"
        receipt_path = OUTPUT / "manifests" / f"{month}.json"
        reusable = False
        if destination.is_file() and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            reusable = (
                receipt.get("build_identity") == build_identity
                and receipt.get("sha256") == _sha256(destination)
                and int(receipt.get("rows", -1)) == len(result)
                and int(receipt.get("features", -1)) == len(expected)
            )
        if not reusable:
            _atomic_parquet(result, destination)
            _atomic_json(
                receipt_path,
                {
                    "status": "COMPLETE",
                    "holding_month": month,
                    "build_identity": build_identity,
                    "base_month_sha256": _sha256(base_path),
                    "rows": len(result),
                    "features": len(expected),
                    "sha256": _sha256(destination),
                },
            )
        month_coverage = result[expected].notna().mean().rename("coverage").reset_index()
        month_coverage = month_coverage.rename(columns={"index": "feature_name"})
        month_coverage.insert(0, "holding_month", month)
        coverage_parts.append(month_coverage)
        print(f"{ordinal}/{len(months)} {month}: {len(result):,} rows", flush=True)

    coverage = pd.concat(coverage_parts, ignore_index=True).merge(
        registry[["feature_name", "family"]],
        on="feature_name",
        how="left",
        validate="many_to_one",
    )
    coverage["year"] = coverage["holding_month"].str[:4]
    _atomic_parquet(coverage, OUTPUT / "coverage_by_month_feature.parquet")
    year_family = (
        coverage.groupby(["year", "family"], observed=True)["coverage"]
        .agg(
            mean_coverage="mean",
            minimum_feature_coverage="min",
            maximum_feature_coverage="max",
        )
        .reset_index()
    )
    _atomic_parquet(year_family, OUTPUT / "coverage_by_year_family.parquet")

    engineered_summary = (
        coverage.loc[coverage["feature_name"].isin(ENGINEERED_FEATURE_NAMES)]
        .groupby(["feature_name", "family"], observed=True)["coverage"]
        .agg(mean_coverage="mean", minimum_month_coverage="min")
        .reset_index()
        .sort_values("feature_name", kind="mergesort")
    )
    _atomic_parquet(engineered_summary, OUTPUT / "engineered_feature_coverage.parquet")
    _atomic_csv(engineered_summary, OUTPUT / "engineered_feature_coverage.csv")

    _atomic_json(
        OUTPUT / "manifest.json",
        {
            "schema": "research_v6_model_features_v2",
            "status": "BACKTEST_READY",
            "selector_backtest_run_by_this_build": False,
            "build_identity": build_identity,
            "base_build_identity": base_manifest["build_identity"],
            "first_holding_month": months[0],
            "last_holding_month": months[-1],
            "months": len(months),
            "rows": len(engineered),
            "model_features_primary_full": int(registry["primary_full"].sum()),
            "model_features_cam_free": int(registry["cam_free"].sum()),
            "model_features_cam_only": int(registry["cam_only"].sum()),
            "base_features": 303,
            "engineered_features": len(ENGINEERED_FEATURE_NAMES),
            "complete_case_filter_used": False,
            "missing_policy": (
                "retain_full_universe_then_common_encoder_zero_plus_missing_indicator"
            ),
            "source_identity": source_identity,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
