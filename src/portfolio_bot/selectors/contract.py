"""Frozen physical contract for roadmap Stage 9.

The contract permits development-only selector fitting and genuine
out-of-fold forecast evaluation.  It deliberately defines no portfolio,
optimizer, P&L, validation-period, or final-period interface.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
import tomllib
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = PROJECT_ROOT / "configs/phase_04_models/selector_policy_v1.toml"

POLICY_ID = "phase04-selector-policy-v1"
CONTRACT_ID = "phase04-selector-v1"
RULES_ID = "phase04-selector-build-rules-v1"

REQUIRED_INPUT_NAMES = (
    "feature_store_raw.parquet",
    "feature_dictionary.parquet",
    "feature_set_membership.parquet",
    "category_vocabulary.parquet",
    "development_labels.parquet",
    "label_split_membership.parquet",
    "development_risk_cost_inputs.parquet",
)
PREDICTOR_INPUT_NAMES = ("feature_store_raw.parquet",)
KEY_AUDIT_ONLY_NAMES = ("development_risk_cost_inputs.parquet",)

REQUIRED_RUNTIME_VERSIONS = {
    "numpy": "2.1.3",
    "scipy": "1.15.3",
    "scikit-learn": "1.6.1",
    "xgboost": "3.2.0",
    "torch": "2.7.1",
}

BASE_MATRIX_WIDTH = 551
ELASTIC_NET_INTERACTION_COUNT = 2_806
ELASTIC_NET_TOTAL_WIDTH = 3_357


class SelectorContractError(RuntimeError):
    """Raised when Stage 9 authority or a frozen convention differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectorContractError(message)


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


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open("rb") as handle:
        value = tomllib.load(handle)
    require(isinstance(value, dict), "Stage 9 policy is not a TOML table")
    return value


def load_rules(policy: Mapping[str, Any]) -> dict[str, Any]:
    specification = policy.get("physical_rules")
    require(isinstance(specification, Mapping), "Stage 9 rules binding is missing")
    path = project_path(str(specification["path"]))
    require(path.is_file() and not path.is_symlink(), "Unsafe Stage 9 rules path")
    require(
        sha256_file(path) == str(specification["sha256"]),
        "Stage 9 physical-rules hash differs",
    )
    value = json.loads(path.read_text("utf-8"))
    require(isinstance(value, dict), "Stage 9 physical rules are not an object")
    return value


def inspect_runtime_versions() -> dict[str, str | None]:
    """Report versions without importing numerical/model frameworks.

    Contract and input authentication do not require every training framework
    to be imported.  The future training entry point must call
    :func:`validate_runtime_versions` before creating any fit artifact.
    """

    observed: dict[str, str | None] = {}
    for distribution in REQUIRED_RUNTIME_VERSIONS:
        try:
            observed[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            observed[distribution] = None
    return observed


def validate_runtime_versions() -> dict[str, str | None]:
    observed = inspect_runtime_versions()
    require(
        sys.version_info >= (3, 11),
        "Stage 9 requires Python 3.11 or later",
    )
    for name, expected in REQUIRED_RUNTIME_VERSIONS.items():
        require(
            observed[name] == expected,
            f"Stage 9 runtime version differs for {name}: "
            f"expected {expected}, observed {observed[name]}",
        )
    return observed


def _validate_scope(policy: Mapping[str, Any], rules: Mapping[str, Any]) -> None:
    scope = policy["scope"]
    require(
        scope["required_signal_dates"] == 178
        and scope["required_feature_audit_rows"] == 1_263_048
        and scope["required_long_eligible_rows"] == 286_628
        and scope["required_present_primary_labels"] == 286_322
        and scope["required_development_folds"] == 4
        and scope["required_development_oof_months"] == 95
        and scope["required_development_oof_rows"] == 172_969
        and scope["required_development_oof_present_labels"] == 172_799,
        "Stage 9 frozen development scope differs",
    )
    require(scope["selector_development_training_permitted"] is True, "Selector fitting is not authorized")
    for key in (
        "stage10_or_later_permitted",
        "portfolio_or_backtest_permitted",
        "pnl_or_performance_permitted",
        "sharpe_calculation_permitted",
        "validation_2015_2019_payload_access_permitted",
        "final_2020_2025_payload_access_permitted",
    ):
        require(scope[key] is False, f"Stage 9 scope unexpectedly permits {key}")
    rule_scope = rules["scope"]
    require(
        rule_scope["first_signal_date"] == "2000-01-31"
        and rule_scope["last_signal_date"] == "2014-10-31"
        and rule_scope["long_eligible_rows"] == 286_628,
        "Stage 9 rule scope differs",
    )
    for key in (
        "validation_2015_2019_permitted",
        "final_2020_2025_permitted",
        "stage10_or_later_permitted",
        "portfolio_weights_permitted",
        "trades_or_pnl_permitted",
        "sharpe_or_portfolio_performance_permitted",
    ):
        require(rule_scope[key] is False, f"Stage 9 rules unexpectedly permit {key}")


def _validate_inputs(policy: Mapping[str, Any], rules: Mapping[str, Any]) -> None:
    inputs = policy["inputs"]
    require(
        tuple(inputs["required_table_names"]) == REQUIRED_INPUT_NAMES,
        "Stage 9 closed input allowlist differs",
    )
    require(
        tuple(inputs["predictor_payload_names"]) == PREDICTOR_INPUT_NAMES
        and tuple(inputs["key_audit_only_names"]) == KEY_AUDIT_ONLY_NAMES,
        "Stage 9 predictor or key-audit input set differs",
    )
    require(
        inputs["only_closed_allowlist_tables_permitted"] is True
        and inputs["validation_or_final_tables_permitted"] is False
        and inputs["stage08_numeric_predictor_or_filter_permitted"] is False
        and inputs["production_hash_bypass_permitted"] is False
        and inputs["symlinked_root_or_path_component_permitted"] is False
        and inputs["stable_stat_and_full_rehash_required"] is True,
        "Stage 9 input safety boundary differs",
    )
    table_specs = inputs["tables"]
    require(
        isinstance(table_specs, list)
        and tuple(str(item["name"]) for item in table_specs) == REQUIRED_INPUT_NAMES,
        "Stage 9 input-table specification order differs",
    )
    key_audit = table_specs[-1]
    require(
        key_audit["key_audit_only"] is True
        and key_audit["predictor_permitted"] is False
        and key_audit["filter_permitted"] is False,
        "Stage 8 numerical payload escaped key-audit-only scope",
    )
    boundary = rules["input_boundary"]
    require(
        boundary["authority"] == "verified_stage08_pointer_only"
        and boundary["stage08_numeric_payload_role"] == "key_audit_only"
        and boundary["stage08_numeric_payload_predictor_permitted"] is False
        and boundary["stage08_numeric_payload_filter_permitted"] is False
        and boundary["validation_or_final_payload_permitted"] is False,
        "Stage 9 rules input boundary differs",
    )


def _validate_matrix_and_runtime(policy: Mapping[str, Any], rules: Mapping[str, Any]) -> None:
    matrix = policy["matrix_contract"]
    preprocessing = rules["preprocessing"]
    elastic = rules["elastic_net_design"]
    require(
        matrix["registry_features"] == 163
        and matrix["numeric_features"] == 153
        and matrix["numeric_columns_with_missing_flags"] == 306
        and matrix["category_columns"] == 245
        and matrix["base_matrix_width"] == BASE_MATRIX_WIDTH
        and matrix["elastic_net_interactions"] == ELASTIC_NET_INTERACTION_COUNT
        and matrix["elastic_net_total_width"] == ELASTIC_NET_TOTAL_WIDTH,
        "Stage 9 matrix contract differs",
    )
    require(
        preprocessing["numeric_output_order"]
        == "feature_registry_ordinal_then_value_then_double_underscore_missing"
        and preprocessing["categorical_output_order"]
        == "category_vocabulary_table_order"
        and preprocessing["base_matrix_width"] == BASE_MATRIX_WIDTH,
        "Stage 9 encoded-column order differs",
    )
    require(
        elastic["continuous_by_aggregate_interaction_count"]
        == ELASTIC_NET_INTERACTION_COUNT
        and elastic["total_matrix_width"] == ELASTIC_NET_TOTAL_WIDTH
        and elastic["interaction_order"]
        == "continuous_firm_registry_ordinal_major_then_aggregate_state_registry_ordinal_minor",
        "Stage 9 Elastic Net interaction order differs",
    )
    runtime = rules["runtime"]
    require(
        runtime["required_versions"] == REQUIRED_RUNTIME_VERSIONS
        and runtime["model_array_dtype"] == "float32"
        and runtime["forecast_dtype"] == "float32"
        and runtime["preprocessing_dtype"] == "float64"
        and runtime["metric_dtype"] == "float64"
        and runtime["threads_per_fit"] == 1
        and runtime["macos_process_start_method"] == "spawn"
        and runtime["authoritative_model_process_families"]
        == ["elastic_net", "xgboost", "neural_network"]
        and runtime["pytorch_and_xgboost_same_worker_permitted"] is False,
        "Stage 9 runtime binding differs",
    )


def _validate_search_and_evaluation(rules: Mapping[str, Any]) -> None:
    neural = rules["neural_network"]
    xgboost = rules["xgboost"]
    elastic = rules["elastic_net_design"]
    training = rules["training"]
    require(
        neural["configuration_count"] == 48
        and len(neural["hidden_layers"]) == 6
        and len(neural["learning_rates"]) == 2
        and len(neural["kernel_l1_values"]) == 2
        and len(neural["dropout_values"]) == 2,
        "Stage 9 neural-network grid differs",
    )
    require(
        xgboost["core_configuration_count"] == 40
        and xgboost["gamma_challenger_count"] == 8
        and xgboost["total_configuration_count"] == 48
        and len(xgboost["core_rows"]) == 10
        and len(xgboost["maximum_depths"]) == 4
        and xgboost["gamma_phase_requires_sealed_core_receipt"] is True
        and xgboost["child_share_recomputed_for_inner_and_full_refit"] is True,
        "Stage 9 XGBoost grid or Hessian rule differs",
    )
    require(
        elastic["configuration_count"] == 250
        and elastic["rho_values"] == [0.0, 0.25, 0.5, 0.75, 1.0]
        and elastic["lambda_grid_size_per_rho"] == 50,
        "Stage 9 Elastic Net grid differs",
    )
    require(
        training["development_seeds"] == [11, 23, 47]
        and training["five_seed_models_permitted_in_stage09"] is False
        and training["fit_weight_normalization"]
        == "mean_one_sum_equals_fit_row_count"
        and training["memory_half_life_months"] == 120.0
        and training["inner_early_stopping_months"] == 12,
        "Stage 9 training seeds, weights, memory, or stopping rule differs",
    )
    aggregation = rules["forecast_aggregation"]
    require(
        aggregation["within_family_order"]
        == "average_three_seed_raw_scalar_forecasts_then_monthly_centered_percentile_score_then_IC"
        and aggregation["seed_specific_IC_averaging_permitted"] is False
        and aggregation["composite_NN_XGBoost_ensemble_weight_selection_permitted_in_stage09"] is False
        and aggregation["full_development_refit_permitted_in_stage09"] is False,
        "Stage 9 forecast handoff differs",
    )
    metrics = rules["metrics"]
    bootstrap = rules["development_bootstrap"]
    shortlist = rules["shortlisting"]
    require(
        metrics["authoritative_monthly_metric"] == "Spearman_rank_IC"
        and metrics["constant_monthly_forecast_IC"] == 0.0
        and metrics["descriptive_metric"] == "Pearson_IC"
        and metrics["fold_dispersion"] == "sample_standard_deviation_ddof_1",
        "Stage 9 IC metric rules differ",
    )
    require(
        bootstrap["method"] == "non_circular_overlapping_moving_block"
        and bootstrap["replicates"] == 10_000
        and bootstrap["block_length_months"] == 6
        and bootstrap["shared_resample_plan_across_configurations"] is True
        and bootstrap["rng"] == "numpy_random_PCG64"
        and bootstrap["master_seed"] == 20_260_714,
        "Stage 9 development bootstrap differs",
    )
    require(
        shortlist["one_standard_error_reference"]
        == "standard_error_of_best_mean_IC_configuration"
        and shortlist["numeric_tie_tolerance"] == 1e-12
        and shortlist["neural_complexity_summary"]
        == "median_trainable_parameter_count_across_four_folds_and_three_seeds"
        and shortlist["tie_order"]
        == "within_tolerance_then_model_specific_complexity_then_deterministic_configuration_id",
        "Stage 9 one-SE or deterministic tie rule differs",
    )


def validate_contract(
    policy_path: Path = DEFAULT_POLICY,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = load_policy(policy_path)
    rules = load_rules(policy)
    require(policy.get("schema_version") == 1, "Unsupported Stage 9 policy schema")
    require(policy.get("stage") == 9, "Stage 9 policy has wrong stage")
    require(
        policy.get("policy_id") == POLICY_ID and policy.get("contract_id") == CONTRACT_ID,
        "Stage 9 policy identity differs",
    )
    require(
        rules.get("schema_version") == 1
        and rules.get("rules_id") == RULES_ID
        and rules.get("rules_id") == policy["physical_rules"]["required_rules_id"],
        "Stage 9 rules identity differs",
    )
    _validate_scope(policy, rules)
    _validate_inputs(policy, rules)
    _validate_matrix_and_runtime(policy, rules)
    _validate_search_and_evaluation(rules)
    return policy, rules


__all__ = [
    "BASE_MATRIX_WIDTH",
    "CONTRACT_ID",
    "DEFAULT_POLICY",
    "ELASTIC_NET_INTERACTION_COUNT",
    "ELASTIC_NET_TOTAL_WIDTH",
    "KEY_AUDIT_ONLY_NAMES",
    "POLICY_ID",
    "PREDICTOR_INPUT_NAMES",
    "PROJECT_ROOT",
    "REQUIRED_INPUT_NAMES",
    "REQUIRED_RUNTIME_VERSIONS",
    "RULES_ID",
    "SelectorContractError",
    "canonical_json_bytes",
    "inspect_runtime_versions",
    "load_policy",
    "load_rules",
    "project_path",
    "require",
    "sha256_file",
    "validate_contract",
    "validate_runtime_versions",
]
