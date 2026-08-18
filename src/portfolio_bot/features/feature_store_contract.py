"""Validate the deterministic roadmap Stage 6 feature-store build contract.

The validator reads only small configuration and verified evidence files.  It
does not open market payloads, construct labels, fit models, or access a
validation/final-period outcome.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import pyarrow as pa

from portfolio_bot.features.phase_03_contract import validate_contract


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = PROJECT_ROOT / "configs/phase_03_features/feature_store_policy_v1.toml"
EXPECTED_POLICY_SHA256 = "1a66bfa4ad537e06829f585e080c356750c7360d486414a40a895de6f6b5380f"
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

AUDIT_FIELDS = (
    pa.field("signal_date", pa.date32(), nullable=False),
    pa.field("permno", pa.int64(), nullable=False),
    pa.field("permco", pa.int64(), nullable=False),
    pa.field("long_eligible", pa.bool_(), nullable=False),
    pa.field("short_eligible", pa.bool_(), nullable=False),
    pa.field("primary_exchange", pa.string(), nullable=True),
    pa.field("share_type", pa.string(), nullable=True),
)
LINEAGE_SCHEMA = pa.schema(
    [
        pa.field("signal_date", pa.date32(), nullable=False),
        pa.field("permno", pa.int64(), nullable=False),
        pa.field("feature_name", pa.string(), nullable=False),
        pa.field("max_observation_date", pa.date32(), nullable=True),
        pa.field("max_public_availability_date", pa.date32(), nullable=True),
        pa.field("stale_after_date", pa.date32(), nullable=True),
        pa.field("value_status", pa.string(), nullable=False),
    ]
)

FORMULA_MEMBER_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "compounded_total_return": ("start_offset", "end_offset"),
    "negative_compounded_total_return": ("start_offset", "end_offset"),
    "price_to_moving_average": ("window_sessions",),
    "distance_to_high": ("window_sessions",),
    "log_price_trend": ("window_sessions", "output"),
    "maximum_drawdown": ("window_sessions",),
    "realized_volatility": ("window_sessions",),
    "downside_semivolatility": ("window_sessions",),
    "beta_label": ("factor",),
    "beta_label_idio_vol": (),
    "sample_skewness": ("window_sessions",),
    "sample_excess_kurtosis": ("window_sessions",),
    "window_extreme": ("window_sessions", "output"),
    "volatility_change": (),
    "natural_log": ("input",),
    "window_median": ("input", "window_sessions"),
    "share_turnover": ("window_sessions",),
    "amihud": ("window_sessions",),
    "spread_hierarchy": (),
    "zero_share": ("input", "window_sessions"),
    "dollar_volume_shock": (),
    "calendar_month_age": (),
    "fundamental_ratio": ("numerator", "denominator"),
    "signed_growth": ("input", "positive_levels_required"),
    "fundamental_indicator": ("condition",),
    "fundamental_source_basis": ("categories",),
    "institutional_ownership": ("output",),
    "days_since_filing": ("forms",),
    "filing_form": ("categories",),
    "event_count": ("form", "window_days"),
    "event_recency": ("form", "window_days"),
    "event_indicator": ("window_days",),
    "segment_statistic": ("output",),
    "latest_8k_item_families": ("window_days",),
    "identity_category": (),
    "option_flow_sum": ("side", "window_sessions"),
    "option_open_interest": ("side",),
    "log_put_call_ratio": ("input",),
    "call_minus_put_share": (),
    "option_log_change": ("input", "lag_months"),
    "option_to_underlying_volume": (),
    "option_coverage": (),
    "option_coverage_age": (),
    "option_historical_volatility": ("nominal_days", "vendor_days"),
    "difference": ("left", "right"),
    "factor_trailing_sum": ("factor", "months"),
    "latest_monthly_value": ("input",),
    "market_realized_volatility": ("window_sessions",),
    "ps_level": (),
    "ps_recursive_innovation": (),
}


class FeatureStoreContractError(RuntimeError):
    """Raised when the Stage 6 physical feature contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeatureStoreContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def expanded_features(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    inherited = ("family", "kind", "source_set", "unit", "formula_id")
    for block in registry.get("blocks", []):
        _require(isinstance(block, dict), "Feature block is not an object")
        _require(all(key in block for key in (*inherited, "members")), "Feature block is incomplete")
        members = block["members"]
        _require(isinstance(members, list) and members, "Feature block is empty")
        common = {key: block[key] for key in inherited}
        formula_id = str(block["formula_id"])
        _require(formula_id in FORMULA_MEMBER_REQUIRED_KEYS, f"Unregistered formula: {formula_id}")
        for member in members:
            _require(isinstance(member, dict), "Feature member is not an object")
            _require(SNAKE_CASE.fullmatch(str(member.get("name", ""))) is not None, "Invalid feature name")
            for key in FORMULA_MEMBER_REQUIRED_KEYS[formula_id]:
                _require(key in member, f"Feature {member['name']} is missing parameter {key}")
            if "window_sessions" in member:
                _require(isinstance(member["window_sessions"], int) and member["window_sessions"] > 0, "Invalid session window")
            if "window_days" in member:
                _require(isinstance(member["window_days"], int) and member["window_days"] > 0, "Invalid calendar window")
            if "start_offset" in member or "end_offset" in member:
                _require(
                    isinstance(member.get("start_offset"), int)
                    and isinstance(member.get("end_offset"), int)
                    and member["start_offset"] <= member["end_offset"] <= 0,
                    "Invalid return offset window",
                )
            features.append({**common, **member})
    names = [feature["name"] for feature in features]
    _require(len(names) == len(set(names)), "Duplicate economic feature name")
    return features


def resolve_feature_sets(
    features: list[Mapping[str, Any]], rules: Mapping[str, Any]
) -> dict[str, tuple[str, ...]]:
    ordered_names = [str(feature["name"]) for feature in features]
    by_name = {str(feature["name"]): feature for feature in features}
    result: dict[str, tuple[str, ...]] = {}
    specs = rules.get("feature_sets")
    _require(isinstance(specs, dict), "Feature-set rules are missing")
    for set_name, spec in specs.items():
        _require(isinstance(spec, dict), f"Invalid feature-set rule: {set_name}")
        if spec.get("include_all_registry_features") is True:
            selected = set(ordered_names)
        else:
            families = set(spec.get("include_families", []))
            selected = {
                name for name, feature in by_name.items() if feature["family"] in families
            }
            explicit = set(spec.get("include_feature_names", []))
            _require(explicit.issubset(by_name), f"Unknown explicit feature in {set_name}")
            selected.update(explicit)
        excluded = set(spec.get("exclude_feature_names", []))
        _require(excluded.issubset(by_name), f"Unknown excluded feature in {set_name}")
        selected.difference_update(excluded)
        ordered = tuple(name for name in ordered_names if name in selected)
        _require(len(ordered) == spec.get("expected_count"), f"Wrong count for {set_name}")
        result[str(set_name)] = ordered
    _require(set(result) == set(specs), "Feature-set resolution changed")
    _require(set(result["market_price_liquidity_core"]).issubset(result["reduced_auxiliary_core"]), "Market core is not a reduced-core subset")
    _require(set(result["reduced_auxiliary_core"]).issubset(result["full_feature_primary"]), "Reduced core is not a full-set subset")
    return result


def validate_ff12_mapping(mapping: Mapping[str, Any]) -> tuple[str, ...]:
    _require(mapping.get("schema_version") == 1, "Unsupported FF12 map schema")
    _require(mapping.get("mapping_id") == "fama-french-12-sic-ranges-v1", "Wrong FF12 map ID")
    sectors = mapping.get("sectors")
    _require(isinstance(sectors, list) and len(sectors) == 12, "FF12 map must have twelve sectors")
    occupied: set[int] = set()
    codes: list[str] = []
    for index, sector in enumerate(sectors, start=1):
        _require(isinstance(sector, dict), "FF12 sector is not an object")
        code = str(sector.get("code", ""))
        _require(code.startswith(f"{index:02d}_"), "FF12 sector order/code mismatch")
        codes.append(code)
        if index == 12:
            _require(sector.get("residual_valid_sic") is True, "FF12 residual sector missing")
            _require("ranges" not in sector, "Residual FF12 sector may not have ranges")
            continue
        ranges = sector.get("ranges")
        _require(isinstance(ranges, list) and ranges, f"FF12 sector {code} has no ranges")
        for bounds in ranges:
            _require(
                isinstance(bounds, list)
                and len(bounds) == 2
                and all(isinstance(value, int) for value in bounds),
                f"Invalid FF12 range in {code}",
            )
            start, end = bounds
            _require(0 <= start <= end <= 9999, f"Out-of-domain FF12 range in {code}")
            values = set(range(start, end + 1))
            _require(not occupied.intersection(values), f"Overlapping FF12 ranges in {code}")
            occupied.update(values)
    _require(mapping.get("unknown_category") == "UNKNOWN_SECTOR", "Wrong FF12 unknown category")
    return tuple(codes) + ("UNKNOWN_SECTOR",)


def classify_ff12(sic: object, mapping: Mapping[str, Any]) -> str:
    """Return the frozen FF12 category for a SIC-like value."""

    text = "" if sic is None else str(sic).strip()
    if not text.isascii() or not text.isdigit() or not (1 <= len(text) <= 4):
        return str(mapping["unknown_category"])
    value = int(text)
    for sector in mapping["sectors"][:-1]:
        if any(start <= value <= end for start, end in sector["ranges"]):
            return str(sector["code"])
    return str(mapping["sectors"][-1]["code"])


def raw_feature_schema(features: list[Mapping[str, Any]]) -> pa.Schema:
    kind_type = {
        "continuous_firm": pa.float64(),
        "aggregate_state": pa.float64(),
        "binary_firm": pa.int8(),
        "categorical_firm": pa.string(),
        "categorical_multihot_firm": pa.list_(pa.string()),
    }
    fields = list(AUDIT_FIELDS)
    for feature in features:
        kind = str(feature["kind"])
        _require(kind in kind_type, f"Unknown feature kind: {kind}")
        nullable = kind not in {"categorical_firm", "categorical_multihot_firm"}
        fields.append(pa.field(str(feature["name"]), kind_type[kind], nullable=nullable))
    return pa.schema(fields)


def _safe_relative_namespace(value: str, prefix: str) -> None:
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts, f"Unsafe namespace: {value}")
    _require(path.parts and path.parts[0] == prefix, f"Wrong lifecycle namespace: {value}")


def _safe_closed_filenames(values: object, label: str) -> tuple[str, ...]:
    _require(isinstance(values, list) and values, f"Empty output file set: {label}")
    names = tuple(str(value) for value in values)
    _require(len(names) == len(set(names)), f"Duplicate output filename: {label}")
    for name in names:
        _require(Path(name).name == name and name not in {".", ".."}, f"Unsafe output filename: {name}")
    return names


def validate_feature_store_contract(
    policy_path: Path = DEFAULT_POLICY, *, enforce_reviewed_hash: bool = True
) -> dict[str, Any]:
    policy_path = policy_path.resolve()
    if enforce_reviewed_hash:
        _require(sha256_file(policy_path) == EXPECTED_POLICY_SHA256, "Stage 6 policy is not the reviewed byte artifact")
    policy = _toml(policy_path)
    _require(policy.get("schema_version") == 1 and policy.get("stage") == 6, "Wrong Stage 6 policy schema")
    _require(policy.get("status") == "frozen_development_only_before_primary_feature_build", "Stage 6 policy is not frozen pre-build")

    semantic_spec = policy["semantic_contract"]
    semantic_path = _project_path(semantic_spec["policy_path"])
    registry_path = _project_path(semantic_spec["registry_path"])
    pointer_path = _project_path(semantic_spec["verified_pointer"])
    _require(sha256_file(semantic_path) == semantic_spec["policy_sha256"], "Semantic policy hash mismatch")
    _require(sha256_file(registry_path) == semantic_spec["registry_sha256"], "Registry hash mismatch")
    _require(sha256_file(pointer_path) == semantic_spec["verified_pointer_sha256"], "Verified contract pointer hash mismatch")
    semantic_summary = validate_contract(semantic_path)
    pointer = _json(pointer_path)
    _require(pointer.get("status") == "verified_passed", "Semantic contract is not verified")
    _require(pointer.get("build_id") == semantic_spec["required_contract_build_id"], "Wrong semantic build")
    _require(pointer.get("verification_id") == semantic_spec["required_contract_verification_id"], "Wrong semantic verification")
    authorizations = pointer.get("authorizations", {})
    _require(authorizations.get(semantic_spec["required_authorization"]) is True, "Feature build authorization is closed")
    _require(authorizations.get("models_portfolios_pnl_performance_or_backtests_authorized") is False, "Model/backtest gate reopened")
    _require(authorizations.get("validation_or_final_test_access_authorized") is False, "Sealed-period gate reopened")

    rules_spec = policy["physical_rules"]
    rules_path = _project_path(rules_spec["path"])
    _require(sha256_file(rules_path) == rules_spec["sha256"], "Physical-rules hash mismatch")
    rules = _json(rules_path)
    _require(rules.get("rules_id") == rules_spec["required_rules_id"], "Wrong physical-rules ID")
    _require(rules.get("status") == "frozen_stage_06_development_only_before_materialization", "Physical rules are not frozen")

    mapping_spec = policy["ff12_mapping"]
    mapping_path = _project_path(mapping_spec["path"])
    _require(sha256_file(mapping_path) == mapping_spec["sha256"], "FF12 mapping hash mismatch")
    mapping = _json(mapping_path)
    ff12_categories = validate_ff12_mapping(mapping)

    registry = _json(registry_path)
    features = expanded_features(registry)
    _require(len(features) == 163 == registry.get("economic_feature_count"), "Economic feature count changed")
    family_order = list(registry.get("family_order", []))
    observed_family_order: list[str] = []
    closed_families: set[str] = set()
    for block in registry["blocks"]:
        family = str(block["family"])
        if not observed_family_order or observed_family_order[-1] != family:
            _require(family not in closed_families, "Feature family is not contiguous")
            if observed_family_order:
                closed_families.add(observed_family_order[-1])
            observed_family_order.append(family)
    _require(observed_family_order == family_order, "Registry block order differs from family order")
    _require(policy["build"]["family_order"] == family_order, "Build family order differs from registry")

    source_sets = rules.get("source_sets")
    _require(isinstance(source_sets, dict), "Source-set map missing")
    registry_sources = {str(feature["source_set"]) for feature in features}
    _require(set(source_sets) == registry_sources, "Source-set map differs from registry")
    for source_set, spec in source_sets.items():
        _require(isinstance(spec, dict), f"Invalid source-set spec: {source_set}")
        _require(isinstance(spec.get("tables"), list) and spec["tables"], f"Source set {source_set} has no table")
        _require(isinstance(spec.get("join"), str) and spec["join"], f"Source set {source_set} has no join rule")

    feature_sets = resolve_feature_sets(features, rules)
    raw_schema = raw_feature_schema(features)
    physical = rules["physical_store"]
    scope = policy["scope"]
    _require(scope["required_spine_rows"] == physical.get("lineage_required_rows") // len(features), "Lineage/spine arithmetic differs")
    _require(scope["required_lineage_rows"] == scope["required_spine_rows"] * len(features), "Policy lineage count differs")
    _require(scope["required_lineage_rows"] == physical["lineage_required_rows"], "Physical lineage count differs")
    _require(physical.get("no_model_matrix_in_stage_06") is True, "Stage 6 model matrix gate reopened")
    statuses = tuple(physical.get("value_status_vocabulary", []))
    precedence = tuple(physical.get("value_status_precedence", []))
    _require(statuses and statuses[0] == "PRESENT", "PRESENT status is not registered first")
    _require(set(precedence) == set(statuses) - {"PRESENT"}, "Missing-status precedence is incomplete")
    _require(len(precedence) == len(set(precedence)), "Missing-status precedence has duplicates")

    fundamentals = rules.get("fundamentals", {})
    row_rules = fundamentals.get("row_filter_and_keys", {})
    _require(row_rules.get("required_filter") == "standard_format_eligible==true", "Compustat format gate missing")
    _require(row_rules.get("quarterly_unique_key") == ["gvkey", "permno", "fyearq", "fqtr"], "Quarterly fiscal key changed")
    _require(row_rules.get("annual_unique_key") == ["gvkey", "permno", "fyear"], "Annual fiscal key changed")
    _require("fatal" in str(row_rules.get("duplicate_key_rule", "")), "Compustat duplicate rule is not fail-closed")
    _require(rules.get("beta_label_numerics", {}).get("rcond") == 1e-12, "Beta solver tolerance changed")
    _require("rcond=1e-12" in rules.get("aggregate_state_numerics", {}).get("pastor_stambaugh_solver", ""), "PS solver tolerance changed")
    _require(rules.get("crsp_market_and_return_rules", {}).get("volatility_change_earliest_offset") == -314, "Volatility-change history boundary changed")

    database = policy["database"]
    _safe_relative_namespace(database["interim_build_root"], "interim")
    _safe_relative_namespace(database["processed_build_root"], "processed")
    _safe_relative_namespace(database["quality_build_root"], "quality")
    _safe_relative_namespace(database["cache_root"], "cache")
    _require(database.get("raw_is_read_only") is True, "Raw-data mutability reopened")

    output_sets = {
        name: _safe_closed_filenames(policy["outputs"][name]["required_files"], name)
        for name in ("interim", "processed", "quality")
    }
    _require("feature_store_raw.parquet" in output_sets["interim"], "Raw feature table missing")
    lineage_names = {f"feature_lineage_{family}.parquet" for family in family_order}
    _require(lineage_names.issubset(output_sets["interim"]), "Lineage family shard missing")
    _require(policy["outputs"]["processed"].get("model_matrix_permitted") is False, "Processed model matrix reopened")

    published_schemas = rules.get("published_table_schemas")
    _require(isinstance(published_schemas, dict), "Published table schemas are missing")
    expected_schema_names = {
        name for name in output_sets["processed"] + output_sets["quality"] if name.endswith(".parquet")
    }
    _require(set(published_schemas) == expected_schema_names, "Published processed/QA schema set differs from output contract")
    for table_name, table_spec in published_schemas.items():
        _require(isinstance(table_spec, dict), f"Invalid schema contract: {table_name}")
        primary_key = table_spec.get("primary_key")
        sort = table_spec.get("sort")
        columns = table_spec.get("columns")
        _require(isinstance(primary_key, list) and primary_key, f"Missing primary key: {table_name}")
        _require(isinstance(sort, list) and sort, f"Missing sort key: {table_name}")
        _require(isinstance(columns, list) and columns, f"Missing column contract: {table_name}")
        column_names = [str(column).split(":", 1)[0] for column in columns]
        _require(len(column_names) == len(set(column_names)), f"Duplicate schema column: {table_name}")
        _require(set(primary_key).issubset(column_names), f"Primary-key column absent: {table_name}")
        _require(set(sort).issubset(column_names), f"Sort-key column absent: {table_name}")
    _require(published_schemas["feature_dictionary.parquet"].get("rows") == 163, "Feature dictionary row contract changed")
    _require(published_schemas["feature_set_membership.parquet"].get("rows") == 489, "Membership row contract changed")

    publication_payloads = rules.get("publication_payloads")
    _require(isinstance(publication_payloads, dict), "Publication payload contracts are missing")
    for name in (
        "success_required_keys",
        "primary_manifest_required_keys",
        "primary_pointer_required_keys",
        "verification_manifest_required_keys",
        "verified_pointer_required_keys",
    ):
        keys = publication_payloads.get(name)
        _require(isinstance(keys, list) and keys and len(keys) == len(set(keys)), f"Invalid publication key contract: {name}")
    quality_summary_keys = publication_payloads.get("quality_summary_exact_keys")
    _require(
        isinstance(quality_summary_keys, list)
        and quality_summary_keys
        and len(quality_summary_keys) == len(set(quality_summary_keys)),
        "Invalid exact quality-summary key contract",
    )
    quality_summary_contract = publication_payloads.get("quality_summary_value_contract")
    _require(isinstance(quality_summary_contract, dict), "Quality-summary value contract is missing")
    _require(
        set(quality_summary_contract) == set(quality_summary_keys),
        "Quality-summary keys and value contracts differ",
    )
    _require(
        quality_summary_contract.get("schema_version") == {"type": "integer", "equals": 1},
        "Quality-summary schema version changed",
    )
    _require(
        quality_summary_contract.get("contract_id", {}).get("equals") == policy["contract_id"],
        "Quality-summary contract ID differs from policy",
    )
    _require(
        quality_summary_contract.get("policy_id", {}).get("equals") == policy["policy_id"],
        "Quality-summary policy ID differs from policy",
    )
    expected_scope_summary = {
        "signal_start_date": policy["scope"]["first_signal_date"],
        "last_signal_date": policy["scope"]["last_signal_date"],
        "raw_rows": policy["scope"]["required_spine_rows"],
        "economic_feature_count": policy["scope"]["required_economic_features"],
        "feature_family_count": len(policy["build"]["family_order"]),
        "logical_lineage_rows": policy["scope"]["required_lineage_rows"],
    }
    _require(
        quality_summary_contract.get("scope", {}).get("exact_values") == expected_scope_summary,
        "Quality-summary scope differs from frozen policy",
    )
    expected_input_builds = {
        "calendar": policy["inputs"]["calendar_build_id"],
        "typed_backbone": policy["inputs"]["typed_backbone_build_id"],
        "return_reconciliation": policy["inputs"]["return_build_id"],
        "signal_universe": policy["inputs"]["signal_universe_build_id"],
        "auxiliary": policy["inputs"]["auxiliary_build_id"],
    }
    _require(
        quality_summary_contract.get("input_builds", {}).get("exact_values") == expected_input_builds,
        "Quality-summary input builds differ from frozen policy",
    )
    _require(
        quality_summary_contract.get("status", {}).get("enum")
        == ["PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION", "PRIMARY_FAILED"],
        "Quality-summary status contract changed",
    )
    numeric_quality_keys = {
        key for key, value in policy["quality_gates"].items() if isinstance(value, int) and not isinstance(value, bool)
    }
    _require(
        set(quality_summary_contract.get("hard_gates", {}).get("exact_keys", [])) == numeric_quality_keys,
        "Quality-summary numeric hard gates differ from policy",
    )
    boolean_quality_flags = {
        key: value for key, value in policy["quality_gates"].items() if isinstance(value, bool)
    }
    _require(
        quality_summary_contract.get("quality_policy_flags", {}).get("exact_values") == boolean_quality_flags,
        "Quality-summary policy flags differ from policy",
    )
    ownership = rules.get("ownership_13f", {})
    _require("union of comparable managers" in str(ownership.get("change_1q_comparison_universe", "")), "13F change universe is not frozen")
    _require("authoritative zero position" in str(ownership.get("change_1q_comparison_universe", "")), "13F full exits are not frozen")
    _require("exclude the manager" in str(ownership.get("change_1q_manager_without_prior_filing", "")), "13F no-prior-filing rule is not frozen")
    _require(str(publication_payloads.get("build_id_rule", "")).startswith("phase03-features-"), "Build-ID rule changed")
    _require(str(publication_payloads.get("verification_id_rule", "")).startswith("verify-"), "Verification-ID rule changed")

    quality = policy["quality_gates"]
    for key in (
        "duplicate_raw_keys",
        "duplicate_lineage_keys",
        "missing_lineage_keys",
        "observation_date_after_signal",
        "public_availability_date_after_signal",
        "stale_feature_uses",
        "future_predictor_rows",
        "nonfinite_published_numeric_values",
        "unknown_feature_names",
        "feature_order_mismatch",
        "optionmetrics_total_rows_used",
        "optionmetrics_noncoverage_encoded_as_zero",
        "raw_pastor_stambaugh_innovation_used",
    ):
        _require(quality.get(key) == 0, f"Nonzero Stage 6 hard gate: {key}")
    _require(quality.get("raw_rows") == scope["required_spine_rows"], "Raw row gate differs")
    _require(quality.get("raw_feature_columns") == len(features), "Raw feature-column gate differs")
    _require(quality.get("lineage_rows") == scope["required_lineage_rows"], "Lineage row gate differs")

    for key in (
        "labels_permitted",
        "model_fitting_permitted",
        "portfolio_or_backtest_permitted",
        "pnl_or_performance_permitted",
        "validation_2015_2019_payload_access_permitted",
        "final_2020_2025_payload_access_permitted",
    ):
        _require(scope.get(key) is False, f"Prohibited Stage 6 scope gate reopened: {key}")

    return {
        "policy_id": policy["policy_id"],
        "policy_sha256": sha256_file(policy_path),
        "semantic_contract_build_id": pointer["build_id"],
        "semantic_contract_verification_id": pointer["verification_id"],
        "semantic_summary": semantic_summary,
        "feature_count": len(features),
        "family_order": tuple(family_order),
        "source_sets": tuple(sorted(source_sets)),
        "feature_sets": feature_sets,
        "ff12_categories": ff12_categories,
        "raw_schema": raw_schema,
        "lineage_schema": LINEAGE_SCHEMA,
        "required_raw_rows": scope["required_spine_rows"],
        "required_lineage_rows": scope["required_lineage_rows"],
        "output_sets": output_sets,
        "rules_sha256": rules_spec["sha256"],
        "ff12_sha256": mapping_spec["sha256"],
        "validation_or_final_access": False,
        "labels_models_portfolios_or_backtests": False,
    }
