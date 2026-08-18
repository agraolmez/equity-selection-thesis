"""Validate and publish the development-only Phase 3 semantic contract.

This module reads only small frozen configuration and evidence files.  It does
not open any market, feature, label, validation, or final-period payload.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import tempfile
import tomllib
from contextlib import contextmanager
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = PROJECT_ROOT / "configs/phase_03_features/phase_03_contract_v1.toml"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/phase_03_features"
PRIMARY_IMPLEMENTATION_PATH = "src/portfolio_bot/features/phase_03_contract.py"
EXPECTED_POLICY_SHA256 = "85cff2869ee4a02352f6a2ba944668c642963bfd3f82e8d9b07199a66b65c21e"
EXPECTED_REGISTRY_SHA256 = "7bac81cd8083c9a5c44ab2482223a342ed6ce2ce265d73cd8cc1bd843e33805a"
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
ALLOWED_KINDS = {
    "continuous_firm",
    "binary_firm",
    "categorical_firm",
    "categorical_multihot_firm",
    "aggregate_state",
}


class Phase3ContractError(RuntimeError):
    """Raised when the frozen Phase 3 contract fails closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(label: str | Path) -> Path:
    path = Path(label).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def project_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise Phase3ContractError(f"Expected JSON object: {path}")
    return value


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase3ContractError(message)


def _parquet_output_receipts(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    raw_outputs = manifest.get("outputs")
    entries: list[Mapping[str, Any]] = []
    if isinstance(raw_outputs, list):
        entries.extend(item for item in raw_outputs if isinstance(item, dict))
    elif isinstance(raw_outputs, dict):
        entries.extend(item for item in raw_outputs.values() if isinstance(item, dict))
    output = manifest.get("output")
    if isinstance(output, dict):
        entries.append(output)
    for entry in entries:
        path = entry.get("path")
        if not isinstance(path, str) or not path.endswith(".parquet"):
            continue
        name = Path(path).name
        _require(name not in receipts, f"Duplicate Parquet output basename: {name}")
        receipts[name] = {
            key: entry[key]
            for key in ("path", "rows", "sha256", "size_bytes")
            if key in entry
        }
    return receipts


def _validate_database_binding(database: Mapping[str, Any]) -> dict[str, Any]:
    paths_path = project_path(database["paths_config"])
    receipt_path = project_path(database["relocation_receipt"])
    _require(sha256_file(paths_path) == database["paths_config_sha256"], "Path configuration hash mismatch")
    _require(sha256_file(receipt_path) == database["relocation_receipt_sha256"], "Relocation receipt hash mismatch")
    paths = _load_toml(paths_path)
    receipt = _load_json(receipt_path)
    db_paths = paths.get("database", {})
    _require(db_paths.get("root") == database["active_root"], "Configured database root is not active_root")
    _require(
        {key: db_paths.get(key) for key in ("raw", "interim", "processed", "quality", "cache")}
        == {"raw": "raw", "interim": "interim", "processed": "processed", "quality": "quality", "cache": "cache"},
        "Database subdirectory contract changed",
    )
    _require(receipt.get("status") == "verified_same_frozen_files_at_new_root", "Relocation is not verified")
    _require(receipt.get("new_database_root") == database["active_root"], "Relocation new root mismatch")
    _require(receipt.get("old_database_root") == database["historical_root"], "Relocation old root mismatch")
    _require(
        receipt.get("paths_configuration", {}).get("current_sha256") == database["paths_config_sha256"],
        "Relocation receipt does not bind current paths configuration",
    )
    historical = receipt.get("historical_evidence_policy", {})
    _require(historical.get("new_runs_must_use_new_database_root") is True, "New-root requirement missing")
    _require(historical.get("rewrite_sealed_artifacts") is False, "Sealed artifacts may not be rewritten")
    _require(
        historical.get("relocation_receipt_required_for_new_runs_that_bind_pre_relocation_evidence") is True,
        "Relocation receipt requirement missing",
    )
    active_root = Path(database["active_root"])
    _require(active_root.is_dir(), "Active database root does not exist")
    for field, prefix in (
        ("phase_03_interim_namespace", "interim"),
        ("phase_03_processed_namespace", "processed"),
        ("phase_03_quality_namespace", "quality"),
    ):
        value = PurePosixPath(database[field])
        _require(not value.is_absolute() and ".." not in value.parts, f"Unsafe namespace: {field}")
        _require(value.parts and value.parts[0] == prefix, f"Wrong lifecycle namespace: {field}")
    return {
        "active_root": database["active_root"],
        "historical_root": database["historical_root"],
        "paths_config": database["paths_config"],
        "paths_config_sha256": database["paths_config_sha256"],
        "relocation_receipt": database["relocation_receipt"],
        "relocation_receipt_sha256": database["relocation_receipt_sha256"],
        "raw_is_read_only": True,
        "new_outputs_use_active_root": True,
    }


def expanded_features(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    inherited = ("family", "kind", "source_set", "unit", "formula_id")
    for block_index, block in enumerate(registry.get("blocks", [])):
        _require(isinstance(block, dict), f"Feature block {block_index} is not an object")
        common = {key: block[key] for key in inherited}
        members = block.get("members")
        _require(isinstance(members, list) and members, f"Feature block {block_index} is empty")
        for member_index, member in enumerate(members):
            _require(
                isinstance(member, dict),
                f"Feature member {block_index}:{member_index} is not an object",
            )
            overlap = set(common).intersection(member)
            _require(not overlap, f"Feature member overrides inherited keys: {sorted(overlap)}")
            features.append({**common, **member})
    return features


def _validate_upstream(name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    pointer_path = project_path(spec["verified_pointer"])
    manifest_path = project_path(spec["primary_manifest"])
    _require(pointer_path.is_file(), f"Missing verified pointer for {name}")
    _require(manifest_path.is_file(), f"Missing primary manifest for {name}")
    _require(
        sha256_file(pointer_path) == spec["verified_pointer_sha256"],
        f"Verified pointer hash mismatch for {name}",
    )
    _require(
        sha256_file(manifest_path) == spec["primary_manifest_sha256"],
        f"Primary manifest hash mismatch for {name}",
    )
    pointer = _load_json(pointer_path)
    manifest = _load_json(manifest_path)
    _require(pointer.get("status") == spec["required_status"], f"Unverified {name}")
    _require(pointer.get("contract_id") == spec["required_contract_id"], f"Wrong {name} pointer contract ID")
    _require(manifest.get("contract_id") == spec["required_contract_id"], f"Wrong {name} manifest contract ID")
    _require(pointer.get("build_id") == spec["required_build_id"], f"Wrong {name} build")
    _require(
        pointer.get("verification_id") == spec["required_verification_id"],
        f"Wrong {name} verification",
    )
    _require(manifest.get("build_id") == spec["required_build_id"], f"Manifest build mismatch for {name}")
    _require(
        pointer.get("primary_manifest_sha256") == spec["primary_manifest_sha256"],
        f"Pointer does not bind the required {name} manifest",
    )
    required_true = set(spec.get("required_authorizations", []))
    required_false = set(spec.get("required_false_authorizations", []))
    if spec.get("legacy_no_authorizations_object"):
        _require(pointer.get("authorizations") is None, f"{name} legacy authorization schema changed")
        summary = manifest.get("summary", {})
        for field in spec.get("required_false_manifest_summary", []):
            _require(summary.get(field) is False, f"{name} manifest summary does not close {field}")
    else:
        authorizations = pointer.get("authorizations")
        _require(isinstance(authorizations, dict), f"{name} authorization object missing")
        for authorization in required_true:
            _require(authorizations.get(authorization) is True, f"{name} does not authorize {authorization}")
        for authorization in required_false:
            _require(authorizations.get(authorization) is False, f"{name} does not explicitly close {authorization}")
        boolean_keys = {key for key, value in authorizations.items() if isinstance(value, bool)}
        _require(boolean_keys == required_true | required_false, f"{name} has an unregistered or missing Boolean authorization")
        _require(all(authorizations[key] is True for key in required_true), f"{name} required true gate changed")
        _require(all(authorizations[key] is False for key in required_false), f"{name} required false gate changed")

    outputs = _parquet_output_receipts(manifest)
    required_tables = list(spec.get("required_tables", []))
    _require(required_tables, f"{name} consumer table allowlist is empty")
    _require(len(required_tables) == len(set(required_tables)), f"{name} consumer table allowlist has duplicates")
    _require(set(required_tables).issubset(outputs), f"{name} is missing a required consumer table")
    prohibited_tables = list(spec.get("prohibited_tables", []))
    _require(not set(required_tables).intersection(prohibited_tables), f"{name} allowlist includes a prohibited table")
    if name == "signal_universe":
        summary = manifest.get("summary", {})
        _require(summary.get("rows") == spec["required_rows"], "Signal-universe row count changed")
        _require(summary.get("signal_dates") == spec["required_signal_dates"], "Signal-date count changed")
        _require(summary.get("first_signal_date") == spec["required_min_signal_date"], "First signal changed")
        _require(summary.get("last_signal_date") == spec["required_max_signal_date"], "Last signal changed")
        _require(summary.get("future_outcome_filter_applied") is False, "Future outcome filter detected")
    if name == "auxiliary":
        _require(spec["primary_or_unverified_pointer_may_authorize"] is False, "Primary auxiliary pointer reopened")
        _require(len(outputs) == spec["required_table_count"], "Auxiliary table count changed")
        _require(sum(int(item.get("rows", 0)) for item in outputs.values()) == spec["required_fully_verified_rows"], "Auxiliary row total changed")
        _require(outputs["auxiliary_signal_linkage.parquet"].get("rows") == spec["required_signal_keys"], "Auxiliary signal key count changed")
    if name == "return_reconciliation":
        _require(spec["primary_return_column"] == "authoritative_total_return", "Return authority column changed")
        _require(spec["primary_return_usable_column"] == "return_history_usable", "Return usability column changed")
        _require(spec["outcome_availability_column"] == "outcome_availability_date", "Outcome availability column changed")
        _require(spec["raw_or_staging_total_return_may_be_consumed"] is False, "Raw/staging total-return access reopened")
        _require(spec["verified_price_return_column"] == "raw_return_ex_distributions", "Verified price-return column changed")
        _require(spec["verified_price_return_authorized_use"] == "adjusted_price_path_features_only", "Price-return use broadened")
        _require(
            set(spec["verified_price_return_prohibited_uses"])
            == {"labels", "targets", "beta_estimation", "risk", "pnl", "performance", "backtests"},
            "Price-return prohibited-use boundary changed",
        )
        _require(spec["monthly_return_crosscheck_may_replace_daily_authority"] is False, "Monthly crosscheck promoted")
    return {
        "name": name,
        "build_id": pointer["build_id"],
        "verification_id": pointer["verification_id"],
        "pointer_path": spec["verified_pointer"],
        "pointer_sha256": spec["verified_pointer_sha256"],
        "manifest_path": spec["primary_manifest"],
        "manifest_sha256": spec["primary_manifest_sha256"],
        "consumer_tables": required_tables,
        "prohibited_tables": prohibited_tables,
    }


def validate_contract(policy_path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    policy_path = policy_path.resolve()
    policy = _load_toml(policy_path)
    _require(sha256_file(policy_path) == EXPECTED_POLICY_SHA256, "Phase 3 policy is not the reviewed frozen byte artifact")
    _require(policy.get("schema_version") == 1, "Unsupported Phase 3 policy schema")
    _require(
        policy.get("status")
        == "frozen_development_only_contract_pending_primary_build_and_independent_verification",
        "Phase 3 policy is not in the frozen pre-build state",
    )

    methodology = policy["methodology"]
    for key, hash_key in (("source", "source_sha256"), ("freeze_manifest", "freeze_manifest_sha256")):
        path = project_path(methodology[key])
        _require(path.is_file(), f"Missing methodology binding: {key}")
        _require(sha256_file(path) == methodology[hash_key], f"Methodology hash mismatch: {key}")

    registry_spec = policy["feature_registry"]
    registry_path = project_path(registry_spec["path"])
    _require(registry_spec["sha256"] == EXPECTED_REGISTRY_SHA256, "Policy does not bind the reviewed registry")
    _require(sha256_file(registry_path) == EXPECTED_REGISTRY_SHA256, "Feature registry is not the reviewed frozen byte artifact")
    _require(sha256_file(registry_path) == registry_spec["sha256"], "Feature registry hash mismatch")
    registry = _load_json(registry_path)
    _require(registry.get("registry_id") == registry_spec["required_registry_id"], "Wrong registry ID")
    features = expanded_features(registry)
    names = [feature["name"] for feature in features]
    _require(len(features) == registry_spec["required_economic_feature_count"], "Wrong feature count")
    _require(len(names) == len(set(names)), "Duplicate feature names")
    _require(all(SNAKE_CASE.fullmatch(name) for name in names), "Non-canonical feature name")
    _require(all(feature["kind"] in ALLOWED_KINDS for feature in features), "Unknown feature kind")
    families = registry["family_order"]
    _require(len(families) == registry_spec["required_family_count"], "Wrong family count")
    _require(set(feature["family"] for feature in features) == set(families), "Feature family mismatch")
    _require(
        all(feature["formula_id"] in registry["formula_catalog"] for feature in features),
        "Feature references an unknown formula",
    )
    _require(registry.get("economic_feature_count") == len(features), "Registry count is stale")

    hv = {
        str(feature["nominal_days"]): feature["vendor_days"]
        for feature in features
        if feature["formula_id"] == "option_historical_volatility"
    }
    _require(hv == {"30": 30, "60": 60, "90": 91, "180": 182, "360": 365}, "Wrong OptionMetrics horizon mapping")

    upstream = [_validate_upstream(name, spec) for name, spec in policy["upstream"].items()]
    _require(len(upstream) == 5, "The Phase 3 contract must bind exactly five direct verified gates")

    database = policy["database"]
    _require(database["new_outputs_may_use_historical_root"] is False, "Historical root reopened")
    _require(database["raw_is_read_only"] is True, "Raw input mutability is not forbidden")
    database_binding = _validate_database_binding(database)

    scope = policy["scope"]
    _require(scope["last_permitted_predictor_signal_date"] == "2014-10-31", "Wrong predictor boundary")
    _require(scope["last_permitted_label_exit_date"] == "2014-12-01", "Wrong label boundary")
    for gate in (
        "validation_2015_2019_payload_or_outcome_access_permitted",
        "final_2020_2025_payload_or_outcome_access_permitted",
        "model_fitting_permitted",
        "portfolio_optimization_permitted",
        "portfolio_or_backtest_permitted",
        "pnl_or_performance_permitted",
    ):
        _require(scope[gate] is False, f"Closed scope gate unexpectedly open: {gate}")

    point_in_time = policy["point_in_time"]
    _require(point_in_time["require_observation_date_not_after_signal"], "Observation-date guard missing")
    _require(point_in_time["require_public_availability_date_not_after_signal"], "Availability guard missing")
    _require(point_in_time["quarterly_rdq_before_datadate_known_rows"] == 43, "RDQ anomaly count changed")
    _require(point_in_time["compustat_is_true_historical_vintage"] is False, "Compustat vintage overstated")

    preprocessing = policy["preprocessing"]
    _require(preprocessing["quantile_algorithm"].startswith("Hyndman-Fan type 7"), "Quantile rule is not pinned")
    _require(preprocessing["centered_rank_formula"] == "((average_rank - 0.5) / N_nonmissing) - 0.5", "Feature rank formula changed")
    _require(preprocessing["dedicated_missing_indicator_for_every_continuous"], "Continuous missingness indicator missing")

    labels = policy["labels"]
    beta = labels["beta_label"]
    _require(beta["lookback_sessions"] == 252 and beta["minimum_valid_observations"] == 126, "beta_label window changed")
    _require(beta["factors"] == ["mktrf", "smb", "hml", "rmw", "cma", "umd"], "beta_label factors changed")
    _require(beta["intercept"] and beta["intercept_is_subtracted_from_forward_label"] is False, "Intercept treatment changed")
    primary = labels["primary"]
    _require(primary["name"] == "residual_return_rank", "Primary target changed")
    _require(primary["minimum_valid_labels_per_signal"] == 100, "Label minimum changed")
    _require(labels["forward_interval"]["signal_to_entry_return_credited"] is False, "Signal-to-entry return credited")

    folds = policy["splits"]["folds"]
    _require([fold["fold_id"] for fold in folds] == [f"dev_fold_{n}" for n in range(1, 5)], "Wrong fold IDs")
    _require([fold["fit_signal_date"] for fold in folds] == ["2006-12-29", "2008-12-31", "2010-12-31", "2012-12-31"], "Wrong fold cutoffs")
    _require([fold["purged_holding_month"] for fold in folds] == ["2006-12", "2008-12", "2010-12", "2012-12"], "Wrong fold purges")
    _require(folds[-1]["validation_last_holding_month"] == "2014-11", "Fold 4 crosses development boundary")

    counts: dict[str, int] = {}
    for kind in ALLOWED_KINDS:
        counts[kind] = sum(feature["kind"] == kind for feature in features)
    return {
        "schema_version": 1,
        "contract_id": policy["contract_id"],
        "policy_path": policy_path.relative_to(PROJECT_ROOT).as_posix(),
        "policy_sha256": sha256_file(policy_path),
        "registry_path": registry_spec["path"],
        "registry_sha256": registry_spec["sha256"],
        "feature_count": len(features),
        "feature_kind_counts": dict(sorted(counts.items())),
        "family_count": len(families),
        "upstream": upstream,
        "database": database_binding,
        "scope": {
            "signal_start_date": scope["signal_start_date"],
            "last_permitted_predictor_signal_date": scope["last_permitted_predictor_signal_date"],
            "last_permitted_label_exit_date": scope["last_permitted_label_exit_date"],
            "validation_or_final_access": False,
            "models_or_backtests": False,
        },
        "fold_count": len(folds),
        "status": "contract_semantics_passed",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_artifact_descendant(root: Path, target: Path) -> bool:
    root = root.resolve()
    target = target.absolute()
    try:
        relative = target.relative_to(root)
    except ValueError:
        return False
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            return False
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError:
        return False
    return True


@contextmanager
def _publication_lock(artifact_root: Path) -> Iterable[None]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_root / ".phase_03_contract.lock"
    _require(not lock_path.is_symlink(), f"Unsafe publication lock symlink: {lock_path}")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _remove_staging(staging: Path) -> None:
    if not staging.exists():
        return
    for child in staging.iterdir():
        _require(child.is_file(), f"Unexpected directory in Phase 3 staging bundle: {child}")
        child.unlink()
    staging.rmdir()


def _closed_regular_file_set(directory: Path, expected_names: set[str]) -> bool:
    try:
        entries = list(directory.iterdir())
    except OSError:
        return False
    return (
        directory.is_dir()
        and not directory.is_symlink()
        and {entry.name for entry in entries} == expected_names
        and all(entry.is_file() and not entry.is_symlink() for entry in entries)
    )


def _bundle_matches_receipts(
    directory: Path, receipts: Mapping[str, Mapping[str, Any]]
) -> bool:
    if not _closed_regular_file_set(directory, set(receipts)):
        return False
    try:
        return all(
            sha256_file(directory / name) == receipt["sha256"]
            and (directory / name).stat().st_size == receipt["size_bytes"]
            for name, receipt in receipts.items()
        )
    except OSError:
        return False


def _install_or_reuse_bundle(staging: Path, final_dir: Path, expected_names: set[str]) -> None:
    if not final_dir.exists():
        os.replace(staging, final_dir)
        _fsync_directory(final_dir.parent)
        return
    _require(
        _closed_regular_file_set(final_dir, expected_names),
        f"Existing deterministic primary bundle has an unexpected file set: {final_dir}",
    )
    for name in sorted(expected_names):
        existing = final_dir / name
        candidate = staging / name
        _require(
            existing.is_file() and not existing.is_symlink(),
            f"Existing deterministic primary output is not a regular file: {existing}",
        )
        _require(
            sha256_file(existing) == sha256_file(candidate)
            and existing.stat().st_size == candidate.stat().st_size,
            f"Existing deterministic primary bundle differs from reconstruction: {final_dir}",
        )
    _remove_staging(staging)


def publish_primary_contract(
    policy_path: Path = DEFAULT_POLICY,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    evidence = validate_contract(policy_path)
    implementation_sha = sha256_file(Path(__file__).resolve())
    build_identity = {
        "contract_id": evidence["contract_id"],
        "policy_sha256": evidence["policy_sha256"],
        "registry_sha256": evidence["registry_sha256"],
        "implementation_path": PRIMARY_IMPLEMENTATION_PATH,
        "implementation_sha256": implementation_sha,
        "upstream": evidence["upstream"],
        "database": evidence["database"],
    }
    build_id = "phase03-contract-" + hashlib.sha256(canonical_json_bytes(build_identity)).hexdigest()[:16]
    artifact_root = artifact_root.resolve()
    final_dir = artifact_root / "contracts" / build_id
    _require(
        _safe_artifact_descendant(artifact_root, final_dir),
        f"Unsafe primary artifact path: {final_dir}",
    )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    _require(
        _safe_artifact_descendant(artifact_root, final_dir),
        f"Unsafe primary artifact path after directory creation: {final_dir}",
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=final_dir.parent))
    try:
        policy = _load_toml(policy_path.resolve())
        registry = _load_json(project_path(policy["feature_registry"]["path"]))
        features = expanded_features(registry)
        feature_rows = []
        for order, feature in enumerate(features, start=1):
            parameters = {
                key: value
                for key, value in feature.items()
                if key not in {"name", "family", "kind", "source_set", "unit", "formula_id"}
            }
            feature_rows.append(
                {
                    "feature_order": order,
                    "feature_name": feature["name"],
                    "family": feature["family"],
                    "kind": feature["kind"],
                    "source_set": feature["source_set"],
                    "unit": feature["unit"],
                    "formula_id": feature["formula_id"],
                    "parameters_json": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
                }
            )
        _write_csv(
            staging / "feature_dictionary.csv",
            ["feature_order", "feature_name", "family", "kind", "source_set", "unit", "formula_id", "parameters_json"],
            feature_rows,
        )
        _write_csv(
            staging / "purged_splits.csv",
            ["fold_id", "fit_signal_date", "train_last_eligible_holding_month", "purged_holding_month", "validation_first_holding_month", "validation_last_holding_month"],
            policy["splits"]["folds"],
        )
        contract = {
            **evidence,
            "build_id": build_id,
            "implementation_sha256": implementation_sha,
            "authorizations": {
                "development_feature_and_label_build_authorized": False,
                "independent_verification_required": True,
                "models_portfolios_pnl_performance_or_backtests_authorized": False,
                "validation_or_final_test_access_authorized": False,
            },
            "status": "primary_passed_pending_independent_verification",
        }
        _write_json(staging / "contract.json", contract)
        outputs = {}
        for name in ("contract.json", "feature_dictionary.csv", "purged_splits.csv"):
            path = staging / name
            outputs[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        manifest = {
            "schema_version": 1,
            "contract_id": evidence["contract_id"],
            "build_id": build_id,
            "status": "primary_passed_pending_independent_verification",
            "identity": build_identity,
            "outputs": outputs,
            "authorizations": contract["authorizations"],
        }
        _write_json(staging / "manifest.json", manifest)
        expected_bundle_receipts = {
            name: {
                "sha256": sha256_file(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for name in ("contract.json", "feature_dictionary.csv", "purged_splits.csv", "manifest.json")
        }
        with _publication_lock(artifact_root):
            _require(
                _safe_artifact_descendant(artifact_root, final_dir),
                f"Unsafe primary artifact path at commit: {final_dir}",
            )
            revalidated = validate_contract(policy_path)
            _require(revalidated == evidence, "Authoritative Phase 3 inputs changed during primary publication")
            _require(
                sha256_file(Path(__file__).resolve()) == implementation_sha,
                "Primary implementation changed during publication",
            )
            _install_or_reuse_bundle(
                staging,
                final_dir,
                {"contract.json", "feature_dictionary.csv", "purged_splits.csv", "manifest.json"},
            )
            _require(
                _bundle_matches_receipts(final_dir, expected_bundle_receipts),
                "Installed primary bundle differs from the deterministic reconstruction",
            )
            # Revalidate immediately before committing the mutable latest pointer.
            _require(validate_contract(policy_path) == evidence, "Authoritative inputs changed before pointer commit")
            _require(
                sha256_file(Path(__file__).resolve()) == implementation_sha,
                "Primary implementation changed before pointer commit",
            )
            _require(_bundle_matches_receipts(final_dir, expected_bundle_receipts), "Primary bundle changed before pointer commit")
            pointer = {
                "schema_version": 1,
                "contract_id": evidence["contract_id"],
                "build_id": build_id,
                "status": "primary_passed_pending_independent_verification",
                "manifest_path": project_label(final_dir / "manifest.json"),
                "manifest_sha256": expected_bundle_receipts["manifest.json"]["sha256"],
                "authorizations": contract["authorizations"],
            }
            _atomic_json(artifact_root / "LATEST_CONTRACT_PRIMARY.json", pointer)
        return final_dir
    except Exception:
        _remove_staging(staging)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    output = publish_primary_contract(args.policy, args.artifact_root)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
