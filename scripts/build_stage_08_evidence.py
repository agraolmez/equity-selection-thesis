#!/usr/bin/env python3
"""Build the bounded Stage 8 report artifact and executable QA notebook."""

from __future__ import annotations

import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import duckdb
import nbformat
import pyarrow.parquet as pq
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports/stage_08_risk_cost"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/08_stage_08_risk_cost_quality.ipynb"
POINTER_PATH = PROJECT_ROOT / "artifacts/phase_04_models/LATEST_RISK_COST_VERIFIED.json"
DATABASE_ROOT = Path("/Users/agraemirolmez/Desktop/Thesis")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _authenticate_file(path: Path, receipt: Mapping[str, Any]) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Unsafe/missing Stage 8 report source: {path}")
    if path.stat().st_size != int(receipt["size_bytes"]):
        raise RuntimeError(f"Size differs: {path}")
    if _sha256(path) != str(receipt["sha256"]):
        raise RuntimeError(f"Hash differs: {path}")
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != int(receipt["rows"]):
            raise RuntimeError(f"Row count differs: {path}")


def _source(
    source_id: str,
    label: str,
    description: str,
    sql: str,
    tables: list[str],
    definitions: list[str],
    *,
    build_id: str,
    verification_id: str,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "query": {
            "description": description,
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "tables_used": tables,
            "filters": [
                f"build_id = {build_id}",
                f"verification_id = {verification_id}",
                "development signals only: 2000-01-31 through 2014-10-31",
                "no Stage 7 label payload, model, portfolio, P&L, validation, or final-period data",
            ],
            "metric_definitions": definitions,
        },
    }


def _rows(connection: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    table = connection.execute(sql).fetch_arrow_table()
    result: list[dict[str, Any]] = []
    for row in table.to_pylist():
        result.append(
            {
                key: (
                    value.isoformat()
                    if hasattr(value, "isoformat")
                    else float(value)
                    if isinstance(value, Decimal)
                    else value
                )
                for key, value in row.items()
            }
        )
    return result


def _load_evidence() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    pointer = _json(POINTER_PATH)
    if pointer.get("status") != "VERIFIED_PASSED":
        raise RuntimeError("Stage 8 is not independently verified")
    build_id = str(pointer["build_id"])
    verification_id = str(pointer["verification_id"])
    manifest_path = (
        PROJECT_ROOT
        / "artifacts/phase_04_models/risk_cost_builds"
        / build_id
        / "manifest.json"
    )
    if _sha256(manifest_path) != pointer["primary_manifest_sha256"]:
        raise RuntimeError("Stage 8 manifest hash differs from verified pointer")
    manifest = _json(manifest_path)
    verification_dir = (
        PROJECT_ROOT
        / "artifacts/phase_04_models/risk_cost_verifications"
        / build_id
        / verification_id
    )
    check_path = verification_dir / "independent_check.json"
    differences_path = verification_dir / "differences.parquet"
    if _sha256(check_path) != pointer["independent_check_sha256"]:
        raise RuntimeError("Stage 8 independent-check hash differs")
    if _sha256(differences_path) != pointer["differences_sha256"]:
        raise RuntimeError("Stage 8 differences hash differs")
    check = _json(check_path)
    if check.get("status") != "VERIFIED_PASSED":
        raise RuntimeError("Stage 8 independent check did not pass")
    if pq.ParquetFile(differences_path).metadata.num_rows != 0:
        raise RuntimeError("Stage 8 differences payload is not empty")

    output_paths: dict[str, Path] = {}
    for location, bundle in manifest["outputs"].items():
        directory = DATABASE_ROOT / bundle["path"]
        for name, receipt in bundle["files"].items():
            path = directory / name
            _authenticate_file(path, receipt)
            output_paths[name] = path

    coverage_path = output_paths["risk_cost_coverage.parquet"]
    factor_path = output_paths["development_factor_covariance.parquet"]
    processed_path = output_paths["development_risk_cost_inputs.parquet"]
    summary_path = output_paths["development_risk_cost_month_summary.parquet"]
    timing_path = output_paths["risk_cost_timing_checks.parquet"]
    parameters_path = output_paths["model_parameters.json"]
    parameters = _json(parameters_path)

    connection = duckdb.connect()
    coverage_sql = f"""
WITH annual AS (
  SELECT year(signal_date) AS signal_year,
         count(*) AS signal_months,
         sum(long_eligible_rows) AS long_eligible_rows,
         sum(risk_present_rows) AS risk_present_rows,
         sum(cost_complete_rows) AS cost_complete_rows,
         sum(short_eligible_rows) AS short_eligible_rows,
         sum(short_cost_complete_rows) AS short_cost_complete_rows
  FROM read_parquet('{coverage_path.as_posix()}')
  GROUP BY 1
), long_form AS (
  SELECT signal_year, signal_months, 'Risk present' AS coverage_type,
         risk_present_rows AS present_rows, long_eligible_rows AS denominator
  FROM annual
  UNION ALL
  SELECT signal_year, signal_months, 'Cost inputs complete',
         cost_complete_rows, long_eligible_rows FROM annual
  UNION ALL
  SELECT signal_year, signal_months, 'Short cost inputs complete',
         short_cost_complete_rows, short_eligible_rows FROM annual
)
SELECT signal_year, signal_months, coverage_type, present_rows, denominator,
       present_rows::DOUBLE / nullif(denominator, 0) AS coverage_rate,
       denominator - present_rows AS missing_rows
FROM long_form
ORDER BY signal_year, coverage_type
"""
    market_risk_sql = f"""
SELECT signal_date,
       factor_observation_count,
       sqrt(greatest(252.0 * covariance_daily, 0.0)) AS annualized_market_factor_volatility,
       covariance_daily AS market_factor_variance_daily,
       covariance_monthly AS market_factor_variance_monthly
FROM read_parquet('{factor_path.as_posix()}')
WHERE factor_row = 'mktrf' AND factor_col = 'mktrf'
ORDER BY signal_date
"""
    impact_sql = f"""
SELECT impact_volatility_source,
       count(*) AS row_count,
       count(*)::DOUBLE / sum(count(*)) over () AS share_of_long_eligible
FROM read_parquet('{processed_path.as_posix()}')
GROUP BY 1
ORDER BY row_count DESC, impact_volatility_source
"""
    gates_sql = f"""
SELECT check_name, observed_value, expected_value, status, detail
FROM read_parquet('{timing_path.as_posix()}')
ORDER BY check_name
"""
    annual_coverage = _rows(connection, coverage_sql)
    market_risk = _rows(connection, market_risk_sql)
    impact_mix = _rows(connection, impact_sql)
    gates = _rows(connection, gates_sql)

    risk_metrics = [
        ("Market beta", "beta_risk_mktrf"),
        ("SMB beta", "beta_risk_smb"),
        ("HML beta", "beta_risk_hml"),
        ("RMW beta", "beta_risk_rmw"),
        ("CMA beta", "beta_risk_cma"),
        ("Momentum beta", "beta_risk_umd"),
        ("Annualized specific volatility", "sqrt(252.0 * specific_variance_daily)"),
        ("Annualized total volatility", "sqrt(252.0 * security_variance_daily)"),
    ]
    risk_distribution: list[dict[str, Any]] = []
    for label, expression in risk_metrics:
        row = connection.execute(
            f"""SELECT count({expression}) AS finite_rows,
                        quantile_cont({expression}, 0.01) AS p01,
                        quantile_cont({expression}, 0.50) AS median,
                        quantile_cont({expression}, 0.99) AS p99,
                        min({expression}) AS minimum,
                        max({expression}) AS maximum
                 FROM read_parquet('{processed_path.as_posix()}')"""
        ).fetchone()
        risk_distribution.append(
            {
                "metric": label,
                "finite_rows": int(row[0]),
                "p01": float(row[1]),
                "median": float(row[2]),
                "p99": float(row[3]),
                "minimum": float(row[4]),
                "maximum": float(row[5]),
            }
        )

    costs = parameters["cost_model"]
    borrow = parameters["borrow_model"]
    financing = parameters["financing_model"]
    parameter_rows = [
        {
            "component": "Transaction cost",
            "scenario": name,
            "spread_multiplier": values["spread_multiplier"],
            "impact_coefficient": values["impact_coefficient"],
            "annual_rate": None,
            "monthly_rate": None,
            "daily_rate": None,
            "note": "Commission/operational slippage remains 1 bp per traded dollar.",
        }
        for name, values in costs["scenarios"].items()
    ]
    for scenario, rate, note in (
        ("primary", borrow["primary_annual_rate"], "All short notional."),
        ("stress_regular", borrow["stress_regular_annual_rate"], "All ordinary shorts."),
        (
            "stress_lowest_liquidity_quintile",
            borrow["stress_lowest_liquidity_quintile_annual_rate"],
            "Total rate, not additive to 3%.",
        ),
        ("primary", financing["primary_annual_spread"], "Gross exposure above 100% NAV."),
        ("stress", financing["stress_annual_spread"], "Gross exposure above 100% NAV."),
    ):
        component = "Financing" if "Gross" in note else "Borrow"
        parameter_rows.append(
            {
                "component": component,
                "scenario": scenario,
                "spread_multiplier": None,
                "impact_coefficient": None,
                "annual_rate": rate,
                "monthly_rate": rate / 12.0,
                "daily_rate": rate / 252.0,
                "note": note,
            }
        )

    verification_rows = [
        {
            "build_id": build_id,
            "verification_id": verification_id,
            "status": pointer["status"],
            "difference_rows": 0,
            "audit_rows": manifest["summary"]["audit_rows"],
            "processed_rows": manifest["summary"]["processed_rows"],
            "risk_present_rows": manifest["summary"]["risk_present_rows"],
            "cost_complete_rows": manifest["summary"]["cost_complete_rows"],
            "hard_gate_count": len(manifest["hard_gates"]),
            "independent_check_count": check.get("checks", {}).get("check_count"),
            "independent_pass_count": check.get("checks", {}).get("pass_count"),
        }
    ]
    datasets = {
        "verification_evidence": verification_rows,
        "coverage_by_year": annual_coverage,
        "market_factor_risk": market_risk,
        "impact_volatility_source_mix": impact_mix,
        "risk_distribution": risk_distribution,
        "model_parameters": parameter_rows,
        "exact_quality_gates": gates,
    }
    relative = lambda path: path.relative_to(DATABASE_ROOT).as_posix()
    artifact_paths = {
        "coverage": relative(coverage_path),
        "factor": relative(factor_path),
        "processed": relative(processed_path),
        "summary": relative(summary_path),
        "timing": relative(timing_path),
        "parameters": relative(parameters_path),
        "verified_pointer": POINTER_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "independent_check": check_path.relative_to(PROJECT_ROOT).as_posix(),
        "differences": differences_path.relative_to(PROJECT_ROOT).as_posix(),
        "manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    context = {
        "pointer": pointer,
        "check": check,
        "manifest": manifest,
        "paths": artifact_paths,
        "sql": {
            # Source metadata must remain portable.  The same statements were
            # executed above with absolute paths; the report records their
            # database-root-relative equivalents.
            "coverage": coverage_sql.replace(f"{DATABASE_ROOT.as_posix()}/", ""),
            "market_risk": market_risk_sql.replace(f"{DATABASE_ROOT.as_posix()}/", ""),
            "impact": impact_sql.replace(f"{DATABASE_ROOT.as_posix()}/", ""),
            "gates": gates_sql.replace(f"{DATABASE_ROOT.as_posix()}/", ""),
        },
    }
    return context, datasets, parameter_rows


def _artifact(
    context: Mapping[str, Any],
    datasets: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    pointer = context["pointer"]
    manifest = context["manifest"]
    build_id = pointer["build_id"]
    verification_id = pointer["verification_id"]
    summary = manifest["summary"]
    paths = context["paths"]
    sql = context["sql"]
    verification_sql = f"""
WITH pointer AS (
  SELECT * FROM read_json_auto('{paths["verified_pointer"]}')
), independent AS (
  SELECT * FROM read_json_auto('{paths["independent_check"]}')
), differences AS (
  SELECT count(*) AS difference_count
  FROM read_parquet('{paths["differences"]}')
), primary_manifest AS (
  SELECT * FROM read_json_auto('{paths["manifest"]}')
)
SELECT pointer.build_id, pointer.verification_id, pointer.status,
       differences.difference_count,
       primary_manifest.summary.audit_rows AS audit_rows,
       primary_manifest.summary.processed_rows AS processed_rows,
       independent.checks.scientific_check_difference_count AS scientific_check_difference_count
FROM pointer
CROSS JOIN independent
CROSS JOIN differences
CROSS JOIN primary_manifest
"""
    risk_specs = [
        ("Market beta", "beta_risk_mktrf"),
        ("SMB beta", "beta_risk_smb"),
        ("HML beta", "beta_risk_hml"),
        ("RMW beta", "beta_risk_rmw"),
        ("CMA beta", "beta_risk_cma"),
        ("Momentum beta", "beta_risk_umd"),
        ("Annualized specific volatility", "sqrt(252.0 * specific_variance_daily)"),
        ("Annualized total volatility", "sqrt(252.0 * security_variance_daily)"),
    ]
    risk_distribution_sql = "\nUNION ALL\n".join(
        f"""SELECT '{label}' AS metric, count({expression}) AS finite_rows,
       quantile_cont({expression}, 0.01) AS p01,
       quantile_cont({expression}, 0.50) AS median,
       quantile_cont({expression}, 0.99) AS p99,
       min({expression}) AS minimum,
       max({expression}) AS maximum
FROM read_parquet('{paths["processed"]}')"""
        for label, expression in risk_specs
    )
    parameters_sql = (
        "SELECT cost_model, borrow_model, financing_model\n"
        f"FROM read_json_auto('{paths['parameters']}')"
    )
    source_specs = [
        _source(
            "verification_evidence",
            "Verified Stage 8 pointer, independent check, differences, and primary manifest",
            "Binds this report to the immutable primary build and independent zero-difference verification.",
            verification_sql,
            [paths["verified_pointer"], paths["independent_check"], paths["differences"], paths["manifest"]],
            ["difference_rows is the authenticated Parquet row count and must equal zero."],
            build_id=build_id,
            verification_id=verification_id,
        ),
        _source(
            "coverage_by_year",
            "Authenticated annual risk/cost coverage",
            "Aggregates monthly numerators and denominators before calculating annual coverage.",
            sql["coverage"],
            [paths["coverage"]],
            [
                "Risk coverage = SUM(risk_present_rows) / SUM(long_eligible_rows).",
                "Cost coverage = SUM(cost_complete_rows) / SUM(long_eligible_rows).",
                "Short cost coverage = SUM(short_cost_complete_rows) / SUM(short_eligible_rows).",
            ],
            build_id=build_id,
            verification_id=verification_id,
        ),
        _source(
            "market_factor_risk",
            "Authenticated monthly market-factor covariance",
            "Selects the MKT-RF diagonal from the verified six-factor covariance store.",
            sql["market_risk"],
            [paths["factor"]],
            ["Annualized market-factor volatility = sqrt(252 * daily MKT-RF variance)."],
            build_id=build_id,
            verification_id=verification_id,
        ),
        _source(
            "impact_volatility_source_mix",
            "Authenticated impact-volatility fallback mix",
            "Counts the frozen 20-session, 60-session, and missing volatility sources.",
            sql["impact"],
            [paths["processed"]],
            ["Share divides each source count by all 286,628 long-eligible development rows."],
            build_id=build_id,
            verification_id=verification_id,
        ),
        _source(
            "risk_distribution",
            "Authenticated beta-risk and volatility distribution",
            "Calculates bounded descriptive quantiles from verified Stage 8 processed rows.",
            risk_distribution_sql,
            [paths["processed"]],
            ["Quantiles are descriptive QA and are not returns, alpha, skill, or performance evidence."],
            build_id=build_id,
            verification_id=verification_id,
        ),
        _source(
            "model_parameters",
            "Frozen Stage 8 model parameters",
            "Reads the authenticated normalized model-parameter payload.",
            parameters_sql,
            [paths["parameters"]],
            ["Annual/12 is the optimizer-month conversion; annual/252 is the later daily-accounting conversion."],
            build_id=build_id,
            verification_id=verification_id,
        ),
        _source(
            "exact_quality_gates",
            "Authenticated exact Stage 8 hard gates",
            "Reads every observed and expected frozen Stage 8 gate.",
            sql["gates"],
            [paths["timing"]],
            ["A gate passes only when observed_value equals expected_value and status is PASS."],
            build_id=build_id,
            verification_id=verification_id,
        ),
    ]
    sources = {item["id"]: item for item in source_specs}

    blocks = [
        {"id": "title", "type": "markdown", "layout": "full", "body": "# Stage 8 Risk and Cost Models: Verified Quality Audit"},
        {
            "id": "executive_summary",
            "type": "markdown",
            "layout": "full",
            "sourceId": "verification_evidence",
            "body": (
                "## Executive Summary\n\n"
                f"Build `{build_id}` is bound to independent verification `{verification_id}` with an authenticated empty differences table. "
                f"All **{len(manifest['hard_gates'])}** primary hard gates passed. Point-in-time risk estimates are present for **{summary['risk_present_rows']:,} of {summary['processed_rows']:,}** long-eligible development rows, while **{summary['cost_complete_rows']:,}** rows have all inputs required for a new trade. This is implementation evidence only—not a model forecast, portfolio, return, alpha, Sharpe ratio, or backtest result."
            ),
        },
        {
            "id": "boundary",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## The Stage 8 boundary remained sealed\n\n"
                "The build uses development signal cutoffs through 31 October 2014, reads no Stage 7 label payload, and opens neither validation nor final-period data. The primary publication authorized nothing; only the independent verified pointer authorizes Stage 9 selector training. Stage 10 optimization and every portfolio/P&L action remain unauthorized."
            ),
        },
        {
            "id": "coverage_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "coverage_by_year",
            "body": (
                "## Risk coverage is complete; cost missingness is explicit\n\n"
                "Annual rates use summed numerators and denominators, not averages of monthly percentages. A cost row is complete only when the verified full spread is nonnegative, ADV is positive, and impact volatility is present; missing inputs never make a trade artificially cheap."
            ),
        },
        {"id": "coverage_chart_block", "type": "chart", "layout": "full", "chartId": "coverage_chart"},
        {
            "id": "market_risk_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "market_factor_risk",
            "body": (
                "## The covariance model updates with the observed risk regime\n\n"
                "The chart shows the annualized volatility implied by the MKT-RF diagonal of each point-in-time covariance estimate. It is a risk input updated at signal cutoffs, not a realized portfolio-volatility or performance series."
            ),
        },
        {"id": "market_risk_chart_block", "type": "chart", "layout": "full", "chartId": "market_risk_chart"},
        {
            "id": "impact_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "impact_volatility_source_mix",
            "body": (
                "## The primary 20-session impact-volatility window dominates\n\n"
                "The 60-session fallback is used only when the 20-session window lacks 15 valid returns, and a new trade remains prohibited when neither window is available. Exact counts are retained in the chart dataset."
            ),
        },
        {"id": "impact_chart_block", "type": "chart", "layout": "full", "chartId": "impact_chart"},
        {
            "id": "risk_distribution_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "risk_distribution",
            "body": (
                "## Unclipped risk betas and variance inputs remain inspectable\n\n"
                "The distribution table preserves tails rather than clipping `beta_risk`. Specific and total volatility rows are annualized from daily variance solely for interpretation; the optimizer consumes the stored monthly factor-plus-specific representation."
            ),
        },
        {"id": "risk_distribution_table_block", "type": "table", "layout": "full", "tableId": "risk_distribution_table"},
        {
            "id": "parameter_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "model_parameters",
            "body": (
                "## Conservative cost assumptions are normalized and immutable\n\n"
                "Primary, standard-cost, and severe-impact-only transaction scenarios remain separate. Borrow stress is 10% total for the lowest-liquidity quintile—not 3% plus 10%—and financing applies only to gross exposure above 100% NAV."
            ),
        },
        {"id": "parameter_table_block", "type": "table", "layout": "full", "tableId": "parameter_table"},
        {
            "id": "gate_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "exact_quality_gates",
            "body": f"## All {len(manifest['hard_gates'])} frozen Stage 8 gates are visible\n\nEach row retains its observed and expected value; no aggregate pass flag hides a failed condition.",
        },
        {"id": "gate_table_block", "type": "table", "layout": "full", "tableId": "gate_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Limitations\n\n"
                "Verification establishes implementation conformance, timing, deterministic formulas, and publication integrity. It does not validate predictive alpha, portfolio feasibility, capacity, realized transaction costs, securities-lending availability, or Sharpe ratio. The 1%/3%/10% borrow rates and square-root impact coefficients remain transparent modeled assumptions because the v1 snapshot has no historical lending feed or local impact calibration."
            ),
        },
        {
            "id": "next_step",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Recommended next step\n\n"
                "Proceed only to roadmap Stage 9: train and development-evaluate the frozen conditional Elastic Net, XGBoost, and shallow neural-network selectors using the verified Stage 6 features, Stage 7 residual-return labels, and Stage 8 risk/cost inputs. Do not begin allocation, optimization, P&L, validation, or final testing."
            ),
        },
    ]
    charts = [
        {
            "id": "coverage_chart",
            "dataset": "coverage_by_year",
            "type": "line",
            "intent": "trend",
            "layout": "full",
            "title": "Risk and cost-input coverage by signal year",
            "subtitle": "Present rows divided by the applicable long-eligible or short-eligible denominator; development only",
            "question": "Did point-in-time risk and cost-input coverage remain complete and stable through development?",
            "rationale": "A three-series line chart shows the full annual path while retaining exact numerators and denominators in the bounded source dataset.",
            "sourceId": "coverage_by_year",
            "maxRows": len(datasets["coverage_by_year"]),
            "showDescription": True,
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "signal_year", "label": "Signal year", "type": "ordinal"},
                "y": {"field": "coverage_rate", "label": "Coverage rate", "type": "quantitative", "format": "percent"},
                "color": {"field": "coverage_type", "label": "Coverage measure", "type": "nominal"},
                "tooltip": [
                    {"field": "present_rows", "label": "Present rows", "type": "quantitative", "format": "number"},
                    {"field": "missing_rows", "label": "Missing rows", "type": "quantitative", "format": "number"},
                    {"field": "denominator", "label": "Denominator", "type": "quantitative", "format": "number"},
                ],
            },
        },
        {
            "id": "market_risk_chart",
            "dataset": "market_factor_risk",
            "type": "line",
            "intent": "trend",
            "layout": "full",
            "title": "Point-in-time market-factor volatility",
            "subtitle": "Annualized from the daily MKT-RF covariance diagonal; 63-session half-life, up to 252 sessions",
            "question": "Does the monthly risk input reflect changes in the observed market-risk regime?",
            "rationale": "The 178-point monthly line preserves regime shape without implying a portfolio return or forecast-skill result.",
            "sourceId": "market_factor_risk",
            "maxRows": len(datasets["market_factor_risk"]),
            "showDescription": True,
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "signal_date", "label": "Signal date", "type": "temporal"},
                "y": {"field": "annualized_market_factor_volatility", "label": "Annualized volatility", "type": "quantitative", "format": "percent"},
                "tooltip": [
                    {"field": "factor_observation_count", "label": "Factor observations", "type": "quantitative", "format": "number"},
                    {"field": "market_factor_variance_daily", "label": "Daily variance", "type": "quantitative", "format": "number"},
                ],
            },
        },
        {
            "id": "impact_chart",
            "dataset": "impact_volatility_source_mix",
            "type": "bar",
            "intent": "comparison",
            "layout": "full",
            "title": "Impact-volatility source mix",
            "subtitle": "All long-eligible development security-signal rows; missing remains a prohibited-new-trade state",
            "question": "How often does the cost model require its 60-session fallback or remain missing?",
            "rationale": "A sorted single-measure bar chart makes the fallback counts directly comparable without a redundant legend.",
            "sourceId": "impact_volatility_source_mix",
            "maxRows": len(datasets["impact_volatility_source_mix"]),
            "showDescription": True,
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "impact_volatility_source", "label": "Source", "type": "nominal"},
                "y": {"field": "row_count", "label": "Security-signal rows", "type": "quantitative", "format": "number"},
                "tooltip": [
                    {"field": "share_of_long_eligible", "label": "Share of long-eligible", "type": "quantitative", "format": "percent"}
                ],
            },
        },
    ]
    tables = [
        {
            "id": "risk_distribution_table",
            "dataset": "risk_distribution",
            "title": "Risk-input distribution diagnostics",
            "subtitle": "Unclipped beta-risk exposures and annualized volatility descriptors across development rows",
            "layout": "full",
            "density": "comfortable",
            "showDescription": True,
            "sourceId": "risk_distribution",
            "defaultSort": {"field": "metric", "direction": "asc"},
            "columns": [
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "finite_rows", "label": "Finite rows", "type": "number", "format": "number"},
                {"field": "p01", "label": "P01", "type": "number", "format": "number"},
                {"field": "median", "label": "Median", "type": "number", "format": "number"},
                {"field": "p99", "label": "P99", "type": "number", "format": "number"},
                {"field": "minimum", "label": "Minimum", "type": "number", "format": "number"},
                {"field": "maximum", "label": "Maximum", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "parameter_table",
            "dataset": "model_parameters",
            "title": "Frozen cost, borrow, and financing parameters",
            "subtitle": "Primary assumptions and prespecified stresses; null fields are not applicable to that component",
            "layout": "full",
            "density": "comfortable",
            "showDescription": True,
            "sourceId": "model_parameters",
            "defaultSort": {"field": "component", "direction": "asc"},
            "columns": [
                {"field": "component", "label": "Component", "type": "text"},
                {"field": "scenario", "label": "Scenario", "type": "text"},
                {"field": "spread_multiplier", "label": "Spread multiplier", "type": "number", "format": "number"},
                {"field": "impact_coefficient", "label": "Impact coefficient", "type": "number", "format": "number"},
                {"field": "annual_rate", "label": "Annual rate", "type": "number", "format": "percent"},
                {"field": "monthly_rate", "label": "Monthly rate", "type": "number", "format": "percent"},
                {"field": "daily_rate", "label": "Daily rate", "type": "number", "format": "percent"},
                {"field": "note", "label": "Interpretation", "type": "text"},
            ],
        },
        {
            "id": "gate_table",
            "dataset": "exact_quality_gates",
            "title": f"Exact {len(manifest['hard_gates'])} Stage 8 hard gates",
            "subtitle": "Authenticated observed and expected values from primary publication",
            "layout": "full",
            "density": "dense",
            "showDescription": True,
            "sourceId": "exact_quality_gates",
            "defaultSort": {"field": "check_name", "direction": "asc"},
            "columns": [
                {"field": "check_name", "label": "Check", "type": "text"},
                {"field": "observed_value", "label": "Observed", "type": "number", "format": "number"},
                {"field": "expected_value", "label": "Expected", "type": "number", "format": "number"},
                {"field": "status", "label": "Result", "type": "text"},
                {"field": "detail", "label": "Evidence note", "type": "text"},
            ],
        },
    ]
    manifest_payload = {
        "version": 1,
        "surface": "report",
        "title": "Stage 8 Risk and Cost Models: Verified Quality Audit",
        "description": "Development-only point-in-time risk and conservative-cost implementation evidence.",
        "blocks": blocks,
        "cards": [],
        "charts": charts,
        "tables": tables,
        "filters": [],
        "sources": source_specs,
    }
    return {
        "surface": "report",
        "manifest": manifest_payload,
        "snapshot": {"version": 1, "status": "ready", "datasets": datasets},
        "sources": source_specs,
    }


def _write_notebook(context: Mapping[str, Any], datasets: Mapping[str, list[dict[str, Any]]]) -> None:
    pointer = context["pointer"]
    manifest = context["manifest"]
    build_id = pointer["build_id"]
    verification_id = pointer["verification_id"]
    gate_count = len(manifest["hard_gates"])
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# Stage 8 Risk and Cost Model Quality Audit\n\n"
            "This notebook reads only the bounded, verification-bound snapshot in `reports/stage_08_risk_cost/artifact.json`. It contains no label reading, model training, allocation, portfolio, P&L, or backtest logic."
        ),
        nbformat.v4.new_markdown_cell(
            "## tl;dr\n\n"
            f"Build `{build_id}` is bound to verification `{verification_id}`. All **{gate_count}** primary hard gates passed and the independent differences table is empty. Risk estimates cover all 286,628 long-eligible development rows; 286,543 have complete new-trade cost inputs."
        ),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "The evidence generator authenticates the verified pointer, primary manifest, independent check, empty differences table, and every source-file receipt before aggregating bounded development-only QA views. Coverage uses summed numerators and denominators."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display\n\n"
            "RELATIVE = Path('reports/stage_08_risk_cost/artifact.json')\n"
            "root = Path.cwd().resolve()\n"
            "while root != root.parent and not (root / RELATIVE).is_file():\n    root = root.parent\n"
            "path = root / RELATIVE\n"
            "if not path.is_file():\n    raise FileNotFoundError(path)\n"
            "artifact = json.loads(path.read_text(encoding='utf-8'))\n"
            "assert artifact['surface'] == 'report'\nassert artifact['snapshot']['status'] == 'ready'\n"
            "datasets = artifact['snapshot']['datasets']\n"
            f"BUILD_ID = {build_id!r}\nVERIFICATION_ID = {verification_id!r}\n"
            "print({'build_id': BUILD_ID, 'verification_id': VERIFICATION_ID, 'artifact': RELATIVE.as_posix()})"
        ),
        nbformat.v4.new_markdown_cell("## Data\n\nThe notebook uses seven bounded reviewed datasets embedded in the verified report artifact."),
        nbformat.v4.new_code_cell(
            "expected = {'verification_evidence': 1, 'coverage_by_year': 45, "
            f"'market_factor_risk': {len(datasets['market_factor_risk'])}, "
            f"'impact_volatility_source_mix': {len(datasets['impact_volatility_source_mix'])}, "
            f"'risk_distribution': {len(datasets['risk_distribution'])}, "
            f"'model_parameters': {len(datasets['model_parameters'])}, "
            f"'exact_quality_gates': {len(datasets['exact_quality_gates'])}}}\n"
            "observed = {name: len(rows) for name, rows in datasets.items()}\n"
            "assert observed == expected\ndisplay(pd.DataFrame([observed]))"
        ),
        nbformat.v4.new_markdown_cell("## Results\n\nThe following cells reproduce the bounded report evidence."),
        nbformat.v4.new_code_cell(
            "coverage = pd.DataFrame(datasets['coverage_by_year'])\n"
            "assert coverage['coverage_rate'].between(0, 1).all()\n"
            "display(coverage.sort_values(['signal_year', 'coverage_type']))"
        ),
        nbformat.v4.new_code_cell(
            "market_risk = pd.DataFrame(datasets['market_factor_risk'])\n"
            "assert (market_risk['annualized_market_factor_volatility'].dropna() >= 0).all()\n"
            "display(market_risk.tail(24))"
        ),
        nbformat.v4.new_code_cell(
            "impact = pd.DataFrame(datasets['impact_volatility_source_mix'])\n"
            "assert impact['row_count'].sum() == 286_628\n"
            "display(impact)"
        ),
        nbformat.v4.new_code_cell(
            "risk = pd.DataFrame(datasets['risk_distribution'])\n"
            "assert (risk['finite_rows'] == 286_628).all()\n"
            "display(risk)\n"
            "display(pd.DataFrame(datasets['model_parameters']))"
        ),
        nbformat.v4.new_code_cell(
            "gates = pd.DataFrame(datasets['exact_quality_gates'])\n"
            f"assert len(gates) == {gate_count}\nassert (gates['status'] == 'PASS').all()\n"
            "assert (gates['observed_value'] == gates['expected_value']).all()\n"
            "display(gates.sort_values('check_name'))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "1. The verified factor-plus-specific risk representation is available for every long-eligible development row.\n"
            "2. Cost-input missingness remains explicit and blocks new trades rather than reducing modeled costs.\n"
            "3. Independent verification reports an empty differences table and does not authorize portfolios or later-period access.\n"
            "4. The next permitted execution stage is Stage 9 selector training only."
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NOTEBOOK_PATH)
    executed = NotebookClient(nb, timeout=300, kernel_name="python3").execute(cwd=str(PROJECT_ROOT))
    nbformat.write(executed, NOTEBOOK_PATH)


def main() -> int:
    context, datasets, _ = _load_evidence()
    artifact = _artifact(context, datasets)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "artifact.json").write_text(
        json.dumps(artifact, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    notes = {
        "schema_version": 1,
        "build_id": context["pointer"]["build_id"],
        "verification_id": context["pointer"]["verification_id"],
        "chart_map": [
            {
                "section": "coverage",
                "question": "Did risk and cost coverage remain stable?",
                "family": "Trend",
                "type": "line",
                "fields": ["signal_year", "coverage_type", "coverage_rate"],
                "claim": "Risk is complete and cost missingness remains bounded and explicit.",
                "palette": "hard two-root cap plus neutral third series",
            },
            {
                "section": "market factor risk",
                "question": "Does risk estimation respond to the observed regime?",
                "family": "Trend",
                "type": "line",
                "fields": ["signal_date", "annualized_market_factor_volatility"],
                "claim": "The monthly covariance input changes with observed factor risk.",
                "palette": "single-root preferred",
            },
            {
                "section": "impact volatility",
                "question": "How often is the 60-session fallback needed?",
                "family": "Comparison",
                "type": "bar",
                "fields": ["impact_volatility_source", "row_count"],
                "claim": "The 20-session estimator supplies almost every eligible row.",
                "palette": "single-root preferred",
            },
        ],
        "omissions": [
            "No return, alpha, IC, Sharpe, drawdown, portfolio, or P&L visual is permitted in Stage 8."
        ],
    }
    (REPORT_DIR / "source_notes.json").write_text(
        json.dumps(notes, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    _write_notebook(context, datasets)
    print(
        json.dumps(
            {
                "artifact": (REPORT_DIR / "artifact.json").as_posix(),
                "notebook": NOTEBOOK_PATH.as_posix(),
                "build_id": context["pointer"]["build_id"],
                "verification_id": context["pointer"]["verification_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
