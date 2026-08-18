"""Frozen physical contract and schemas for roadmap Stage 8.

This module owns no empirical calculation.  It authenticates the Stage 8
policy/rules pair and exposes the closed schemas used by primary publication
and independent recomputation.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = (
    PROJECT_ROOT / "configs/phase_04_models/risk_cost_policy_v1.toml"
)

FACTOR_NAMES = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
PRESENT = "PRESENT"
NOT_LONG_ELIGIBLE = "NOT_LONG_ELIGIBLE"
MISSING_FACTOR_WINDOW = "MISSING_FACTOR_WINDOW"
MISSING_RETURN_OBSERVATIONS = "MISSING_RETURN_OBSERVATIONS"
MISSING_REGRESSION = "MISSING_REGRESSION"
MISSING_SHRINKAGE_TARGET = "MISSING_SHRINKAGE_TARGET"
MISSING_RIDGE = "MISSING_RIDGE"


class RiskCostContractError(RuntimeError):
    """Raised when the Stage 8 contract or a bound artifact is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RiskCostContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open("rb") as handle:
        policy = tomllib.load(handle)
    require(isinstance(policy, dict), "Stage 8 policy is not a TOML table")
    return policy


def load_rules(policy: Mapping[str, Any]) -> dict[str, Any]:
    path = project_path(str(policy["physical_rules"]["path"]))
    require(path.is_file() and not path.is_symlink(), "Unsafe Stage 8 rules path")
    require(
        sha256_file(path) == str(policy["physical_rules"]["sha256"]),
        "Stage 8 physical-rules hash differs",
    )
    value = json.loads(path.read_text("utf-8"))
    require(isinstance(value, dict), "Stage 8 physical rules are not an object")
    return value


def _audit_schema() -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("signal_date", pa.date32(), nullable=False),
        pa.field("permno", pa.int64(), nullable=False),
        pa.field("permco", pa.int64(), nullable=False),
        pa.field("long_eligible", pa.bool_(), nullable=False),
        pa.field("short_eligible", pa.bool_(), nullable=False),
        pa.field("primary_exchange", pa.string()),
        pa.field("ff12_sector_category", pa.string()),
        pa.field("domicile_country_category", pa.string()),
        pa.field("information_cutoff_date", pa.date32(), nullable=False),
        pa.field("risk_status", pa.string(), nullable=False),
        pa.field("risk_observation_count", pa.int16(), nullable=False),
        pa.field("risk_window_first_date", pa.date32()),
        pa.field("risk_window_last_date", pa.date32()),
        pa.field("beta_risk_intercept", pa.float64()),
    ]
    fields.extend(
        pa.field(f"beta_risk_{name}", pa.float64()) for name in FACTOR_NAMES
    )
    fields.extend(
        [
            pa.field("specific_variance_raw_daily", pa.float64()),
            pa.field("shrinkage_target_daily", pa.float64()),
            pa.field("shrinkage_target_source", pa.string()),
            pa.field("shrinkage_population_count", pa.int32(), nullable=False),
            pa.field("specific_variance_shrunk_daily", pa.float64()),
            pa.field("risk_ridge_daily", pa.float64()),
            pa.field("specific_variance_daily", pa.float64()),
            pa.field("specific_variance_monthly", pa.float64()),
            pa.field("security_variance_pre_ridge_daily", pa.float64()),
            pa.field("security_variance_daily", pa.float64()),
            pa.field("security_variance_monthly", pa.float64()),
            pa.field("full_spread_20d", pa.float64()),
            pa.field("adv20_usd", pa.float64()),
            pa.field("impact_volatility_daily", pa.float64()),
            pa.field("impact_volatility_source", pa.string()),
            pa.field("impact_volatility_observations", pa.int16(), nullable=False),
            pa.field("exit_impact_volatility_daily", pa.float64()),
            pa.field("exit_volatility_source", pa.string()),
            pa.field("short_liquidity_quintile", pa.int8()),
            pa.field("primary_borrow_rate_annual", pa.float64()),
            pa.field("stress_borrow_rate_annual", pa.float64()),
            pa.field("new_long_trade_allowed", pa.bool_(), nullable=False),
            pa.field("new_short_trade_allowed", pa.bool_(), nullable=False),
            pa.field("cost_input_status", pa.string(), nullable=False),
            pa.field("post_cutoff_return_rows_used", pa.int32(), nullable=False),
            pa.field("post_cutoff_factor_rows_used", pa.int32(), nullable=False),
        ]
    )
    return pa.schema(fields)


def published_schemas() -> dict[str, pa.Schema]:
    audit = _audit_schema()
    factor_covariance = pa.schema(
        [
            pa.field("signal_date", pa.date32(), nullable=False),
            pa.field("factor_row", pa.string(), nullable=False),
            pa.field("factor_col", pa.string(), nullable=False),
            pa.field("factor_row_ordinal", pa.int8(), nullable=False),
            pa.field("factor_col_ordinal", pa.int8(), nullable=False),
            pa.field("factor_observation_count", pa.int16(), nullable=False),
            pa.field("factor_window_first_date", pa.date32()),
            pa.field("factor_window_last_date", pa.date32()),
            pa.field("half_life_sessions", pa.float64(), nullable=False),
            pa.field("covariance_daily", pa.float64()),
            pa.field("covariance_monthly", pa.float64()),
            pa.field("factor_covariance_status", pa.string(), nullable=False),
        ]
    )
    month_summary = pa.schema(
        [
            pa.field("signal_date", pa.date32(), nullable=False),
            pa.field("audit_rows", pa.int32(), nullable=False),
            pa.field("long_eligible_rows", pa.int32(), nullable=False),
            pa.field("short_eligible_rows", pa.int32(), nullable=False),
            pa.field("risk_present_rows", pa.int32(), nullable=False),
            pa.field("factor_covariance_present", pa.bool_(), nullable=False),
            pa.field("factor_observation_count", pa.int16(), nullable=False),
            pa.field("factor_covariance_min_eigenvalue", pa.float64()),
            pa.field("ridge_daily", pa.float64()),
            pa.field("cross_sectional_specific_variance_median", pa.float64()),
            pa.field("impact_volatility_20d_rows", pa.int32(), nullable=False),
            pa.field("impact_volatility_60d_rows", pa.int32(), nullable=False),
            pa.field("impact_volatility_missing_rows", pa.int32(), nullable=False),
            pa.field("exit_volatility_p95_daily", pa.float64()),
            pa.field("spread_missing_rows", pa.int32(), nullable=False),
            pa.field("adv_missing_or_nonpositive_rows", pa.int32(), nullable=False),
            pa.field("new_long_trade_allowed_rows", pa.int32(), nullable=False),
            pa.field("new_short_trade_allowed_rows", pa.int32(), nullable=False),
            pa.field("post_cutoff_return_rows_used", pa.int32(), nullable=False),
            pa.field("post_cutoff_factor_rows_used", pa.int32(), nullable=False),
        ]
    )
    timing = pa.schema(
        [
            pa.field("check_name", pa.string(), nullable=False),
            pa.field("observed_value", pa.float64(), nullable=False),
            pa.field("expected_value", pa.float64(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("detail", pa.string(), nullable=False),
        ]
    )
    coverage = pa.schema(
        [
            pa.field("signal_date", pa.date32(), nullable=False),
            pa.field("long_eligible_rows", pa.int32(), nullable=False),
            pa.field("risk_present_rows", pa.int32(), nullable=False),
            pa.field("risk_coverage_fraction", pa.float64()),
            pa.field("cost_complete_rows", pa.int32(), nullable=False),
            pa.field("cost_coverage_fraction", pa.float64()),
            pa.field("short_eligible_rows", pa.int32(), nullable=False),
            pa.field("short_cost_complete_rows", pa.int32(), nullable=False),
            pa.field("short_cost_coverage_fraction", pa.float64()),
        ]
    )
    return {
        "development_risk_cost_audit.parquet": audit,
        "development_risk_cost_inputs.parquet": audit,
        "development_factor_covariance.parquet": factor_covariance,
        "development_risk_cost_month_summary.parquet": month_summary,
        "risk_cost_timing_checks.parquet": timing,
        "risk_cost_coverage.parquet": coverage,
    }


def validate_contract(policy_path: Path = DEFAULT_POLICY) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = load_policy(policy_path)
    rules = load_rules(policy)
    require(policy.get("schema_version") == 1, "Unsupported Stage 8 policy schema")
    require(policy.get("stage") == 8, "Stage 8 policy has wrong stage")
    require(
        policy.get("policy_id") == "phase04-risk-cost-policy-v1"
        and policy.get("contract_id") == "phase04-risk-cost-v1",
        "Stage 8 policy identity differs",
    )
    require(
        rules.get("rules_id") == policy["physical_rules"]["required_rules_id"],
        "Stage 8 rules identity differs",
    )
    scope = policy["scope"]
    require(
        scope["required_signal_dates"] == 178
        and scope["required_audit_rows"] == 1_263_048
        and scope["required_long_eligible_rows"] == 286_628,
        "Stage 8 frozen scope counts differ",
    )
    for key in (
        "model_fitting_permitted",
        "portfolio_or_backtest_permitted",
        "pnl_or_performance_permitted",
        "validation_2015_2019_payload_access_permitted",
        "final_2020_2025_payload_access_permitted",
    ):
        require(scope[key] is False, f"Stage 8 scope unexpectedly permits {key}")
    risk = rules["risk_model"]
    require(tuple(risk["factor_order"]) == FACTOR_NAMES, "Stage 8 factor order differs")
    require(
        risk["window_sessions"] == 252
        and risk["minimum_valid_security_observations"] == 126
        and risk["half_life_sessions"] == 63.0
        and risk["beta_clipping_permitted"] is False,
        "Stage 8 primary beta-risk rules differ",
    )
    costs = rules["cost_model"]
    require(
        costs["commission_operational_slippage_rate"] == 0.0001
        and costs["scenarios"]["primary"]["impact_coefficient"] == 0.10
        and costs["scenarios"]["standard_cost_stress"]["impact_coefficient"] == 0.20
        and costs["scenarios"]["severe_impact_only"]["impact_coefficient"] == 0.50,
        "Stage 8 cost constants differ",
    )
    expected_factor_rows = 178 * len(FACTOR_NAMES) ** 2
    require(
        policy["quality_gates"]["factor_covariance_rows"] == expected_factor_rows,
        "Stage 8 factor-covariance row gate differs",
    )
    schemas = published_schemas()
    require(len(schemas) == 6, "Stage 8 closed schema set differs")
    return policy, rules
