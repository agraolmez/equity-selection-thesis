"""Validate the deterministic roadmap Stage 7 label-store build contract.

This module reads only small configuration, pointer, and manifest evidence.
It does not open return/factor payloads, construct labels, fit models, or
access validation/final-period outcomes.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import pyarrow as pa

from portfolio_bot.features.feature_store_contract import (
    validate_feature_store_contract,
)
from portfolio_bot.features.phase_03_contract import validate_contract


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "configs/phase_03_features/label_store_policy_v1.toml"
)
EXPECTED_POLICY_SHA256 = (
    "1f6f5649cfd1959a707c6e04465b70a334b1fbc4b32e9b81158baf521c2abcad"
)


class LabelStoreContractError(RuntimeError):
    """Raised when the Stage 7 physical label contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LabelStoreContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LabelStoreContractError(
            f"Cannot read Stage 7 JSON evidence: {path}"
        ) from exc
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LabelStoreContractError(
            f"Cannot read Stage 7 TOML evidence: {path}"
        ) from exc
    _require(isinstance(value, dict), f"Expected TOML table: {path}")
    return value


def _safe_relative_namespace(value: str, prefix: str) -> None:
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and ".." not in path.parts
        and bool(path.parts)
        and path.parts[0] == prefix,
        f"Unsafe Stage 7 lifecycle namespace: {value}",
    )


def _safe_closed_filenames(
    values: object,
    label: str,
) -> tuple[str, ...]:
    _require(
        isinstance(values, list) and bool(values),
        f"Empty Stage 7 output set: {label}",
    )
    names = tuple(str(value) for value in values)
    _require(
        len(names) == len(set(names)),
        f"Duplicate Stage 7 output filename: {label}",
    )
    for name in names:
        _require(
            Path(name).name == name and name not in {".", ".."},
            f"Unsafe Stage 7 output filename: {name}",
        )
    return names


_ARROW_TYPES: Mapping[str, pa.DataType] = {
    "date32": pa.date32(),
    "int64": pa.int64(),
    "int32": pa.int32(),
    "int16": pa.int16(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
    "utf8": pa.string(),
}


def _parse_field(value: object) -> pa.Field:
    parts = str(value).split(":")
    _require(
        len(parts) == 3,
        f"Malformed Stage 7 column contract: {value}",
    )
    name, type_name, nullability = parts
    _require(
        name
        and type_name in _ARROW_TYPES
        and nullability in {"not_null", "nullable"},
        f"Invalid Stage 7 column contract: {value}",
    )
    return pa.field(
        name,
        _ARROW_TYPES[type_name],
        nullable=nullability == "nullable",
    )


def published_schemas(
    rules: Mapping[str, Any],
) -> dict[str, pa.Schema]:
    specs = rules.get("published_table_schemas")
    _require(
        isinstance(specs, dict) and bool(specs),
        "Stage 7 published schemas are missing",
    )
    result: dict[str, pa.Schema] = {}
    for table_name, spec in specs.items():
        _require(
            isinstance(spec, dict),
            f"Invalid Stage 7 table contract: {table_name}",
        )
        columns = spec.get("columns")
        primary_key = spec.get("primary_key")
        sort_key = spec.get("sort")
        _require(
            isinstance(columns, list) and bool(columns),
            f"Missing Stage 7 columns: {table_name}",
        )
        _require(
            isinstance(primary_key, list) and bool(primary_key),
            f"Missing Stage 7 primary key: {table_name}",
        )
        _require(
            isinstance(sort_key, list) and bool(sort_key),
            f"Missing Stage 7 sort key: {table_name}",
        )
        schema = pa.schema([_parse_field(value) for value in columns])
        _require(
            len(schema.names) == len(set(schema.names)),
            f"Duplicate Stage 7 schema column: {table_name}",
        )
        _require(
            set(primary_key).issubset(schema.names),
            f"Stage 7 primary-key column absent: {table_name}",
        )
        _require(
            set(sort_key).issubset(schema.names),
            f"Stage 7 sort-key column absent: {table_name}",
        )
        result[str(table_name)] = schema
    return result


def _validate_hash_bound_file(
    specification: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    label: str,
) -> Path:
    path = _project_path(str(specification[path_key]))
    _require(
        path.is_file() and not path.is_symlink(),
        f"Missing or symlinked {label}",
    )
    _require(
        sha256_file(path) == specification[hash_key],
        f"{label} hash mismatch",
    )
    return path


def _validate_pointer(
    path: Path,
    *,
    expected_status: str,
    label: str,
) -> dict[str, Any]:
    pointer = _json(path)
    _require(
        pointer.get("status") == expected_status,
        f"{label} is not verified",
    )
    authorizations = pointer.get("authorizations")
    _require(
        isinstance(authorizations, dict),
        f"{label} authorizations are missing",
    )
    return pointer


def validate_label_store_contract(
    policy_path: Path = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Validate and summarize the frozen Stage 7 physical contract."""

    policy_path = policy_path.resolve()
    _require(
        sha256_file(policy_path) == EXPECTED_POLICY_SHA256,
        "Stage 7 policy is not the reviewed byte artifact",
    )
    policy = _toml(policy_path)
    _require(
        policy.get("schema_version") == 1
        and policy.get("stage") == 7,
        "Wrong Stage 7 policy schema",
    )
    _require(
        policy.get("status")
        == "frozen_development_only_before_primary_label_build",
        "Stage 7 policy is not frozen pre-build",
    )
    _require(
        policy.get("contract_id")
        == "phase03-feature-label-split-v1",
        "Wrong Stage 7 contract ID",
    )

    semantic_spec = policy["semantic_contract"]
    semantic_path = _validate_hash_bound_file(
        semantic_spec,
        path_key="policy_path",
        hash_key="policy_sha256",
        label="Stage 7 semantic contract",
    )
    semantic_summary = validate_contract(semantic_path)
    contract_pointer_path = _validate_hash_bound_file(
        semantic_spec,
        path_key="verified_pointer",
        hash_key="verified_pointer_sha256",
        label="Stage 7 verified semantic pointer",
    )
    contract_pointer = _validate_pointer(
        contract_pointer_path,
        expected_status="verified_passed",
        label="Stage 7 semantic pointer",
    )
    _require(
        contract_pointer.get("build_id")
        == semantic_spec["required_contract_build_id"]
        and contract_pointer.get("verification_id")
        == semantic_spec["required_contract_verification_id"],
        "Stage 7 semantic authority identity differs",
    )
    contract_authorizations = contract_pointer["authorizations"]
    _require(
        contract_authorizations.get(
            semantic_spec["required_authorization"]
        )
        is True,
        "Stage 7 semantic label authorization is closed",
    )
    _require(
        contract_authorizations.get(
            "models_portfolios_pnl_performance_or_backtests_authorized"
        )
        is False
        and contract_authorizations.get(
            "validation_or_final_test_access_authorized"
        )
        is False,
        "Stage 7 semantic downstream seal reopened",
    )

    feature_spec = policy["verified_feature_store"]
    feature_policy_path = _validate_hash_bound_file(
        feature_spec,
        path_key="policy_path",
        hash_key="policy_sha256",
        label="Stage 6 feature policy",
    )
    validate_feature_store_contract(feature_policy_path)
    feature_pointer_path = _validate_hash_bound_file(
        feature_spec,
        path_key="verified_pointer",
        hash_key="verified_pointer_sha256",
        label="Stage 6 verified feature pointer",
    )
    feature_pointer = _validate_pointer(
        feature_pointer_path,
        expected_status="VERIFIED_PASSED",
        label="Stage 6 feature pointer",
    )
    _require(
        feature_pointer.get("build_id")
        == feature_spec["required_build_id"]
        and feature_pointer.get("verification_id")
        == feature_spec["required_verification_id"],
        "Wrong verified Stage 6 feature-store identity",
    )
    _require(
        feature_pointer["authorizations"].get(
            feature_spec["required_authorization"]
        )
        is True,
        "Verified Stage 6 pointer does not authorize labels",
    )
    _require(
        feature_pointer["authorizations"].get(
            "models_portfolios_pnl_or_backtests_authorized"
        )
        is False
        and feature_pointer["authorizations"].get(
            "validation_or_final_test_access_authorized"
        )
        is False,
        "Stage 6 downstream seal reopened",
    )
    feature_manifest_path = _validate_hash_bound_file(
        feature_spec,
        path_key="primary_manifest",
        hash_key="primary_manifest_sha256",
        label="Stage 6 primary feature manifest",
    )
    feature_manifest = _json(feature_manifest_path)
    _require(
        feature_manifest.get("build_id")
        == feature_spec["required_build_id"],
        "Stage 6 primary manifest identity differs",
    )
    _require(
        feature_pointer.get("primary_manifest_sha256")
        == feature_spec["primary_manifest_sha256"],
        "Stage 6 pointer-to-manifest binding differs",
    )

    rules_spec = policy["physical_rules"]
    rules_path = _validate_hash_bound_file(
        rules_spec,
        path_key="path",
        hash_key="sha256",
        label="Stage 7 physical rules",
    )
    rules = _json(rules_path)
    _require(
        rules.get("rules_id") == rules_spec["required_rules_id"]
        and rules.get("stage") == 7,
        "Wrong Stage 7 physical-rules identity",
    )
    _require(
        rules.get("status")
        == "frozen_stage_07_development_only_before_materialization",
        "Stage 7 physical rules are not frozen",
    )

    scope = policy["scope"]
    rule_scope = rules["scope"]
    for policy_key, rule_key in (
        ("first_signal_date", "first_signal_date"),
        ("last_signal_date", "last_signal_date"),
        ("last_permitted_exit_date", "last_permitted_exit_date"),
        ("required_audit_rows", "required_audit_rows"),
        ("required_long_eligible_rows", "required_long_eligible_rows"),
        ("required_signal_dates", "required_signal_dates"),
        ("required_folds", "required_folds"),
        (
            "required_split_membership_rows",
            "required_split_membership_rows",
        ),
    ):
        _require(
            scope[policy_key] == rule_scope[rule_key],
            f"Stage 7 scope differs: {policy_key}",
        )
    _require(
        scope["minimum_valid_labels_per_signal"]
        == rules["targets"]["minimum_valid_labels_per_signal"]
        == 100,
        "Stage 7 minimum monthly label count changed",
    )
    for prohibited in (
        "model_fitting_permitted",
        "portfolio_or_backtest_permitted",
        "pnl_or_performance_permitted",
        "validation_2015_2019_payload_access_permitted",
        "final_2020_2025_payload_access_permitted",
    ):
        _require(
            scope[prohibited] is False,
            f"Stage 7 prohibited scope reopened: {prohibited}",
        )

    allowlist = rules["input_allowlist"]
    expected_inputs = tuple(policy["inputs"]["required_table_names"])
    _require(
        tuple(allowlist) == expected_inputs
        and len(expected_inputs) == len(set(expected_inputs)) == 6,
        "Stage 7 input allowlist differs",
    )
    for name, specification in allowlist.items():
        _require(
            isinstance(specification, dict)
            and isinstance(specification.get("rows"), int)
            and specification["rows"] >= 0,
            f"Stage 7 input row contract differs: {name}",
        )
        for hash_name in ("sha256", "schema_sha256"):
            value = specification.get(hash_name)
            _require(
                isinstance(value, str) and len(value) == 64,
                f"Stage 7 input {hash_name} missing: {name}",
            )
        projection = specification.get("projected_columns")
        _require(
            isinstance(projection, list)
            and bool(projection)
            and len(projection) == len(set(projection)),
            f"Stage 7 input projection differs: {name}",
        )
    _require(
        allowlist["crsp_security_day_return.parquet"][
            "semantic_max_date"
        ]
        == scope["last_permitted_exit_date"]
        and allowlist["crsp_security_day_return.parquet"][
            "physical_max_date"
        ]
        == "2014-12-31",
        "Stage 7 return cutoff contract differs",
    )
    _require(
        allowlist["ff_factor_daily_observation.parquet"]["last_date"]
        == scope["last_permitted_exit_date"]
        and allowlist["ff_factor_daily_observation.parquet"]["units"]
        == "decimal_ratio",
        "Stage 7 factor boundary or units differ",
    )
    authentication = rules["input_authentication"]
    _require(
        authentication["production_hash_bypass_permitted"] is False
        and authentication[
            "symlinked_root_or_path_component_permitted"
        ]
        is False
        and authentication[
            "lexical_or_resolved_root_escape_permitted"
        ]
        is False
        and authentication["initial_full_sha256_required"] is True
        and authentication["bounded_temporal_delivery_required"] is True
        and authentication[
            "identity_check_before_each_payload_read_required"
        ]
        is True
        and authentication[
            "stable_stat_and_full_rehash_after_materialization_required"
        ]
        is True
        and authentication[
            "stable_stat_and_full_rehash_before_publication_required"
        ]
        is True,
        "Stage 7 input-authentication rules differ",
    )
    input_policy = policy["inputs"]
    _require(
        input_policy["production_hash_bypass_permitted"] is False
        and input_policy[
            "symlinked_root_or_path_component_permitted"
        ]
        is False
        and input_policy[
            "identity_check_before_each_payload_read_required"
        ]
        is True
        and input_policy[
            "return_rows_must_be_predicate_filtered_before_delivery"
        ]
        is True
        and input_policy[
            "rehash_inputs_after_build_before_publication"
        ]
        is True
        and input_policy[
            "stable_stat_and_full_rehash_before_publication_required"
        ]
        is True,
        "Stage 7 input-authentication policy differs",
    )
    upstream_pointers = policy["inputs"]["verified_pointers"]
    verified_upstream: dict[str, dict[str, Any]] = {}
    for name in ("calendar", "returns", "signal_universe", "auxiliary"):
        pointer_path = _validate_hash_bound_file(
            upstream_pointers,
            path_key=name,
            hash_key=f"{name}_sha256",
            label=f"Stage 7 verified {name} pointer",
        )
        pointer = _json(pointer_path)
        verified_upstream[name] = pointer
        _require(
            pointer.get("status") == "verified_passed",
            f"Stage 7 {name} pointer is not verified",
        )
    _require(
        verified_upstream["returns"].get("authorizations", {}).get(
            "return_history_authorized"
        )
        is True,
        "Stage 7 return-history pointer authorization is closed",
    )

    beta = rules["beta_binding"]
    _require(
        beta["primary_builder_action"]
        == (
            "consume_verified_stage_06_clipped_slopes_without_"
            "reestimation_or_reclipping"
        )
        and beta["lookback_global_sessions"] == 252
        and beta["minimum_valid_observations"] == 126
        and beta["half_life_global_sessions"] == 126
        and beta["factor_order"]
        == ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
        and beta["rcond"] == 1e-12
        and beta["required_rank"] == 7,
        "Stage 7 beta binding differs",
    )
    _require(
        beta["intercept"] is True
        and beta["intercept_subtracted_from_forward_label"] is False,
        "Stage 7 beta-intercept rule differs",
    )

    forward = rules["forward_interval"]
    _require(
        forward["signal_to_entry_return_credited"] is False
        and forward["compounding_permitted"] is False
        and forward["required_covered_trading_sessions"] == 1
        and forward["successor_security_returns_permitted"] is False,
        "Stage 7 forward-interval rule differs",
    )
    _require(
        forward["return_authorization_source"]
        == (
            "authenticated verified return-reconciliation pointer with "
            "return_history_authorized true"
        )
        and forward["row_level_return_history_authorized_rule"]
        == "constant-false provenance guard; never a row eligibility filter",
        "Stage 7 return authorization source differs",
    )
    targets = rules["targets"]
    _require(
        targets["common_population"]
        == "primary-valid long-eligible rows within one signal date"
        and targets["short_eligibility_influences_target"] is False
        and targets["rank_numerator_month_sum"] == 0
        and targets["rank_recenter_after_formula_permitted"] is False
        and targets["mad_scale"] == 1.4826
        and targets["mad_clip"] == [-5.0, 5.0]
        and targets["mad_epsilon_permitted"] is False,
        "Stage 7 target rule differs",
    )
    splits = rules["splits"]
    _require(
        splits["axis"] == "holding_month"
        and len(splits["folds"]) == scope["required_folds"]
        and splits["rows_per_signal_date"] == scope["required_folds"]
        and splits["development_embargo_holding_month"] == "2014-12",
        "Stage 7 split or embargo contract differs",
    )

    database = policy["database"]
    _validate_hash_bound_file(
        database,
        path_key="paths_config",
        hash_key="paths_config_sha256",
        label="Stage 7 path configuration",
    )
    _validate_hash_bound_file(
        database,
        path_key="relocation_receipt",
        hash_key="relocation_receipt_sha256",
        label="Stage 7 relocation receipt",
    )
    _safe_relative_namespace(
        database["interim_build_root"], "interim"
    )
    _safe_relative_namespace(
        database["processed_build_root"], "processed"
    )
    _safe_relative_namespace(
        database["quality_build_root"], "quality"
    )
    _safe_relative_namespace(database["cache_root"], "cache")
    _require(
        database["raw_is_read_only"] is True,
        "Stage 7 raw-data mutability reopened",
    )

    output_sets = {
        name: _safe_closed_filenames(
            policy["outputs"][name]["required_files"], name
        )
        for name in ("interim", "processed", "quality")
    }
    schemas = published_schemas(rules)
    expected_schema_names = {
        name
        for output_set in output_sets.values()
        for name in output_set
        if name.endswith(".parquet")
    }
    _require(
        set(schemas) == expected_schema_names,
        "Stage 7 published-schema set differs from closed outputs",
    )
    _require(
        rules["published_table_schemas"][
            "development_label_audit.parquet"
        ]["rows"]
        == scope["required_audit_rows"]
        and rules["published_table_schemas"][
            "development_labels.parquet"
        ]["rows"]
        == scope["required_long_eligible_rows"]
        and rules["published_table_schemas"][
            "development_label_month_summary.parquet"
        ]["rows"]
        == scope["required_signal_dates"]
        and rules["published_table_schemas"][
            "label_split_membership.parquet"
        ]["rows"]
        == scope["required_split_membership_rows"],
        "Stage 7 output row contracts differ",
    )
    _require(
        policy["outputs"]["processed"]["model_matrix_permitted"] is False,
        "Stage 7 model-matrix gate reopened",
    )

    statuses = rules["status_vocabulary"]
    ordered_statuses = tuple(statuses["ordered"])
    _require(
        ordered_statuses[0] == "PRESENT"
        and len(ordered_statuses) == len(set(ordered_statuses))
        and set(statuses["outcome_precedence"]).issubset(
            ordered_statuses
        )
        and statuses["primary_additional_status"]
        == "MISSING_MONTH_MINIMUM"
        and statuses["magnitude_additional_status"]
        == "MISSING_ZERO_MAD",
        "Stage 7 status vocabulary differs",
    )
    domains = rules["value_domains"]
    _require(
        domains["outcome_status"] == list(ordered_statuses[:-2])
        and domains["primary_label_status"]
        == list(ordered_statuses[:-1])
        and domains["magnitude_label_status"]
        == list(ordered_statuses),
        "Stage 7 derived status domains differ",
    )
    _require(
        domains["month_status"]
        == [
            "PRESENT",
            "NO_ELIGIBLE_SECURITIES",
            "MISSING_MONTH_MINIMUM",
        ]
        and domains["status_dimension"]
        == ["outcome", "primary", "magnitude"]
        and domains["split_role"] == splits["roles"]
        and domains["distribution_target_name"]
        == [
            "unranked_excess_return",
            "unranked_residual_return",
            "residual_return_rank",
            "excess_return_rank",
            "residual_return_mad_standardized",
        ]
        and domains["timing_severity"] == ["CRITICAL"]
        and domains["timing_check_name"]
        == {"source": "exact sorted key set of integer-valued quality_gates"},
        "Stage 7 closed output value domains differ",
    )
    derived_status = rules["derived_status_rules"]
    _require(
        derived_status["magnitude_month_minimum_precedes_zero_mad"]
        is True
        and derived_status["primary_present_outcome_invalid_month"]
        == "MISSING_MONTH_MINIMUM"
        and derived_status["magnitude_present_primary_zero_mad"]
        == "MISSING_ZERO_MAD"
        and derived_status["month_no_eligible"]
        == "NO_ELIGIBLE_SECURITIES"
        and derived_status["month_eligible_below_minimum"]
        == "MISSING_MONTH_MINIMUM"
        and derived_status["month_meets_minimum"] == "PRESENT",
        "Stage 7 derived-status precedence differs",
    )
    conventions = rules["implementation_conventions"]
    _require(
        conventions["lineage_leg_ordinal"]
        == "one_based_chronological_within_signal_date_permno"
        and conventions["lineage_scope"]
        == "accepted contributions from PRESENT security outcomes only"
        and conventions["decimal_to_float64_conversion"]
        == (
            "Python float conversion of authenticated decimal128(38,12) "
            "values before chronological math.fsum"
        )
        and conventions["nonpresent_security_label_available_date"]
        == "null"
        and conventions["audit_month_availability_copy_scope"]
        == (
            "copy the signal-month value to every audit-spine row "
            "including nonlong and invalid rows"
        ),
        "Stage 7 lineage or availability convention differs",
    )
    _require(
        conventions["status_counts"]["density"]
        == (
            "dense_zero_filled_signal_date_by_dimension_by_allowed_"
            "status_domain"
        )
        and conventions["status_counts"]["denominator"]
        == "all audit-spine rows in the signal date"
        and conventions["coverage"]["denominator"]
        == "long-eligible audit rows"
        and conventions["coverage"]["zero_denominator_rate"] == 0.0,
        "Stage 7 status-count or coverage convention differs",
    )
    distribution = conventions["distribution"]
    _require(
        distribution["population"]
        == "long-eligible processed rows within signal date"
        and distribution["p01_p99_algorithm"]
        == "Hyndman-Fan type 7 linear interpolation"
        and distribution["mean_primitive"]
        == "math.fsum divided by finite_count"
        and distribution["zero_finite_statistics"] == "null",
        "Stage 7 distribution convention differs",
    )
    split_flags = conventions["split_flags"]
    _require(
        split_flags["included_in_training"]
        == (
            "role TRAIN_ELIGIBLE and primary month valid and month "
            "available by fit"
        )
        and split_flags["included_in_validation"]
        == "role VALIDATION and primary month valid"
        and split_flags["embargo_boundary_checked_outside_membership"]
        is True,
        "Stage 7 split-flag convention differs",
    )
    comparison = conventions["float_comparison"]
    _require(
        comparison["beta_absolute_tolerance"]
        == comparison["beta_relative_tolerance"]
        == 1e-10
        and comparison["label_absolute_tolerance"]
        == comparison["label_relative_tolerance"]
        == 1e-12
        and comparison["integer_and_status_comparison"] == "exact",
        "Stage 7 comparison convention differs",
    )

    hard_gates = {
        key: value
        for key, value in policy["quality_gates"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    rule_gates = {
        key: value
        for key, value in rules["quality_gates"].items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    _require(
        hard_gates == rule_gates,
        "Stage 7 numeric hard gates differ",
    )
    _require(
        policy["quality_gates"][
            "label_coverage_is_descriptive_not_pass_fail"
        ]
        is True
        and policy["quality_gates"][
            "missing_outcomes_are_never_zero_filled"
        ]
        is True,
        "Stage 7 missingness/coverage policy differs",
    )

    publication = policy["publication"]
    _require(
        publication["primary_pointer_authorizes_downstream_use"]
        is False
        and publication["independent_verification_required"] is True
        and publication[
            "verified_pointer_authorizes_stage_08_risk_and_cost_models"
        ]
        is True
        and publication[
            "verified_pointer_authorizes_stage_09_model_training"
        ]
        is False
        and publication[
            "verified_pointer_authorizes_portfolios_pnl_or_backtests"
        ]
        is False
        and publication[
            "verified_pointer_authorizes_validation_or_final_access"
        ]
        is False,
        "Stage 7 publication authority differs",
    )
    for pointer_key in (
        "latest_primary_pointer",
        "latest_verified_pointer",
    ):
        pointer_path = PurePosixPath(publication[pointer_key])
        _require(
            not pointer_path.is_absolute()
            and ".." not in pointer_path.parts
            and pointer_path.parts[0] == "artifacts",
            f"Unsafe Stage 7 pointer path: {pointer_key}",
        )

    payloads = rules["publication_payloads"]
    for key in (
        "success_required_keys",
        "primary_manifest_required_keys",
        "primary_pointer_required_keys",
        "independent_check_required_keys",
        "verified_pointer_required_keys",
        "quality_summary_exact_keys",
    ):
        values = payloads.get(key)
        _require(
            isinstance(values, list)
            and bool(values)
            and len(values) == len(set(values)),
            f"Stage 7 publication key contract differs: {key}",
        )

    return {
        "policy_id": policy["policy_id"],
        "contract_id": policy["contract_id"],
        "rules_id": rules["rules_id"],
        "semantic_contract_build_id": contract_pointer["build_id"],
        "feature_store_build_id": feature_pointer["build_id"],
        "feature_store_verification_id": feature_pointer[
            "verification_id"
        ],
        "required_audit_rows": scope["required_audit_rows"],
        "required_long_eligible_rows": scope[
            "required_long_eligible_rows"
        ],
        "required_signal_dates": scope["required_signal_dates"],
        "required_split_membership_rows": scope[
            "required_split_membership_rows"
        ],
        "input_names": expected_inputs,
        "output_sets": output_sets,
        "published_schemas": schemas,
        "status_vocabulary": ordered_statuses,
        "semantic_summary": semantic_summary,
        "model_training_authorized": False,
        "portfolio_or_backtest_authorized": False,
        "validation_or_final_access": False,
    }
