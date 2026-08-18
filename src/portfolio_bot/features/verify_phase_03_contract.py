"""Independently reconstruct and verify the Phase 3 semantic contract bundle.

This module deliberately does not import the primary contract implementation.
It reads only frozen configuration, small manifests, and contract outputs; it
does not open any feature, label, market, validation, or final-period payload.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import re
import tempfile
import tomllib
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = PROJECT_ROOT / "configs/phase_03_features/phase_03_contract_v1.toml"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/phase_03_features"
PRIMARY_IMPLEMENTATION_PATH = "src/portfolio_bot/features/phase_03_contract.py"
EXPECTED_POLICY_SHA256 = "85cff2869ee4a02352f6a2ba944668c642963bfd3f82e8d9b07199a66b65c21e"
EXPECTED_REGISTRY_SHA256 = "7bac81cd8083c9a5c44ab2482223a342ed6ce2ce265d73cd8cc1bd843e33805a"
EXPECTED_KINDS = {
    "aggregate_state": 23,
    "binary_firm": 8,
    "categorical_firm": 8,
    "categorical_multihot_firm": 2,
    "continuous_firm": 122,
}
PRIMARY_AUTHORIZATIONS = {
    "development_feature_and_label_build_authorized": False,
    "independent_verification_required": True,
    "models_portfolios_pnl_performance_or_backtests_authorized": False,
    "validation_or_final_test_access_authorized": False,
}
VERIFIED_AUTHORIZATIONS = {
    "development_feature_and_label_build_authorized": True,
    "models_portfolios_pnl_performance_or_backtests_authorized": False,
    "validation_or_final_test_access_authorized": False,
}
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class Phase3VerificationError(RuntimeError):
    """Raised when independent Phase 3 contract verification fails closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise Phase3VerificationError(f"Expected JSON object: {path}")
    return value


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def observe_bytes(path: Path, observed_inputs: dict[Path, str]) -> bytes:
    resolved = path.resolve()
    value = resolved.read_bytes()
    digest = sha256_bytes(value)
    previous = observed_inputs.get(resolved)
    if previous is not None and previous != digest:
        raise Phase3VerificationError(f"Authoritative input changed between reads: {resolved}")
    observed_inputs[resolved] = digest
    return value


def observe_json(path: Path, observed_inputs: dict[Path, str]) -> dict[str, Any]:
    value = json.loads(observe_bytes(path, observed_inputs).decode("utf-8"))
    if not isinstance(value, dict):
        raise Phase3VerificationError(f"Expected JSON object: {path}")
    return value


def observe_toml(path: Path, observed_inputs: dict[Path, str]) -> dict[str, Any]:
    return tomllib.loads(observe_bytes(path, observed_inputs).decode("utf-8"))


def csv_bytes(fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return stream.getvalue().encode("utf-8")


def differences_csv_bytes(rows: Iterable[Mapping[str, str]]) -> bytes:
    return csv_bytes(["check_id", "expected", "actual"], rows)


def add_check(
    checks: list[dict[str, Any]],
    differences: list[dict[str, str]],
    check_id: str,
    condition: bool,
    expected: Any,
    actual: Any,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "PASS" if condition else "FAIL",
            "expected": expected,
            "actual": actual,
        }
    )
    if not condition:
        differences.append(
            {
                "check_id": check_id,
                "expected": json.dumps(expected, sort_keys=True),
                "actual": json.dumps(actual, sort_keys=True),
            }
        )


def independently_expand(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    inherited = ("family", "kind", "source_set", "unit", "formula_id")
    for block in registry.get("blocks", []):
        base = {key: block[key] for key in inherited}
        for member in block.get("members", []):
            overlap = set(base).intersection(member)
            if overlap:
                raise Phase3VerificationError(
                    f"Registry member overrides inherited keys: {sorted(overlap)}"
                )
            result.append({**base, **member})
    return result


def parquet_receipts(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries: list[Mapping[str, Any]] = []
    outputs = manifest.get("outputs")
    if isinstance(outputs, list):
        entries.extend(item for item in outputs if isinstance(item, dict))
    elif isinstance(outputs, dict):
        entries.extend(item for item in outputs.values() if isinstance(item, dict))
    output = manifest.get("output")
    if isinstance(output, dict):
        entries.append(output)
    receipts: dict[str, dict[str, Any]] = {}
    for entry in entries:
        path = entry.get("path")
        if not isinstance(path, str) or not path.endswith(".parquet"):
            continue
        name = Path(path).name
        if name in receipts:
            raise Phase3VerificationError(f"Duplicate upstream Parquet basename: {name}")
        receipts[name] = {
            key: entry[key]
            for key in ("path", "rows", "sha256", "size_bytes")
            if key in entry
        }
    return receipts


def reconstruct_database(
    spec: Mapping[str, Any],
    checks: list[dict[str, Any]],
    differences: list[dict[str, str]],
    observed_inputs: dict[Path, str],
) -> dict[str, Any]:
    paths_path = project_path(spec["paths_config"])
    receipt_path = project_path(spec["relocation_receipt"])
    paths = observe_toml(paths_path, observed_inputs)
    receipt = observe_json(receipt_path, observed_inputs)
    observed_paths_hash = observed_inputs[paths_path.resolve()]
    observed_receipt_hash = observed_inputs[receipt_path.resolve()]
    add_check(checks, differences, "DATABASE_PATH_CONFIG_HASH", observed_paths_hash == spec["paths_config_sha256"], spec["paths_config_sha256"], observed_paths_hash)
    add_check(checks, differences, "DATABASE_RELOCATION_HASH", observed_receipt_hash == spec["relocation_receipt_sha256"], spec["relocation_receipt_sha256"], observed_receipt_hash)
    database_paths = paths.get("database", {})
    expected_subdirectories = {
        "raw": "raw",
        "interim": "interim",
        "processed": "processed",
        "quality": "quality",
        "cache": "cache",
    }
    observed_subdirectories = {
        key: database_paths.get(key) for key in expected_subdirectories
    }
    add_check(checks, differences, "DATABASE_ACTIVE_ROOT", database_paths.get("root") == spec["active_root"], spec["active_root"], database_paths.get("root"))
    add_check(checks, differences, "DATABASE_SUBDIRECTORIES", observed_subdirectories == expected_subdirectories, expected_subdirectories, observed_subdirectories)
    relocation_ok = (
        receipt.get("status") == "verified_same_frozen_files_at_new_root"
        and receipt.get("new_database_root") == spec["active_root"]
        and receipt.get("old_database_root") == spec["historical_root"]
        and receipt.get("paths_configuration", {}).get("current_sha256")
        == spec["paths_config_sha256"]
    )
    add_check(checks, differences, "DATABASE_RELOCATION_BINDING", relocation_ok, True, relocation_ok)
    historical = receipt.get("historical_evidence_policy", {})
    historical_policy_ok = (
        historical.get("new_runs_must_use_new_database_root") is True
        and historical.get("rewrite_sealed_artifacts") is False
        and historical.get(
            "relocation_receipt_required_for_new_runs_that_bind_pre_relocation_evidence"
        )
        is True
    )
    add_check(checks, differences, "DATABASE_HISTORICAL_POLICY", historical_policy_ok, True, historical_policy_ok)
    active_root_exists = Path(spec["active_root"]).is_dir()
    add_check(checks, differences, "DATABASE_ACTIVE_ROOT_EXISTS", active_root_exists, True, active_root_exists)
    namespaces_ok = True
    for field, prefix in (
        ("phase_03_interim_namespace", "interim"),
        ("phase_03_processed_namespace", "processed"),
        ("phase_03_quality_namespace", "quality"),
    ):
        value = PurePosixPath(spec[field])
        namespaces_ok &= (
            not value.is_absolute()
            and ".." not in value.parts
            and bool(value.parts)
            and value.parts[0] == prefix
        )
    namespaces_ok &= spec["new_outputs_may_use_historical_root"] is False
    namespaces_ok &= spec["raw_is_read_only"] is True
    add_check(checks, differences, "DATABASE_LIFECYCLE_NAMESPACES", namespaces_ok, True, namespaces_ok)
    return {
        "active_root": spec["active_root"],
        "historical_root": spec["historical_root"],
        "paths_config": spec["paths_config"],
        "paths_config_sha256": spec["paths_config_sha256"],
        "relocation_receipt": spec["relocation_receipt"],
        "relocation_receipt_sha256": spec["relocation_receipt_sha256"],
        "raw_is_read_only": True,
        "new_outputs_use_active_root": True,
    }


def reconstruct_upstream(
    name: str,
    spec: Mapping[str, Any],
    checks: list[dict[str, Any]],
    differences: list[dict[str, str]],
    observed_inputs: dict[Path, str],
) -> dict[str, Any]:
    pointer_path = project_path(spec["verified_pointer"])
    manifest_path = project_path(spec["primary_manifest"])
    pointer = observe_json(pointer_path, observed_inputs)
    manifest = observe_json(manifest_path, observed_inputs)
    pointer_hash = observed_inputs[pointer_path.resolve()]
    manifest_hash = observed_inputs[manifest_path.resolve()]
    add_check(checks, differences, f"UPSTREAM_{name.upper()}_POINTER_HASH", pointer_hash == spec["verified_pointer_sha256"], spec["verified_pointer_sha256"], pointer_hash)
    add_check(checks, differences, f"UPSTREAM_{name.upper()}_MANIFEST_HASH", manifest_hash == spec["primary_manifest_sha256"], spec["primary_manifest_sha256"], manifest_hash)
    identity_ok = (
        pointer.get("status") == spec["required_status"]
        and pointer.get("contract_id") == spec["required_contract_id"]
        and manifest.get("contract_id") == spec["required_contract_id"]
        and pointer.get("build_id") == spec["required_build_id"]
        and pointer.get("verification_id") == spec["required_verification_id"]
        and manifest.get("build_id") == spec["required_build_id"]
        and pointer.get("primary_manifest_sha256") == spec["primary_manifest_sha256"]
    )
    add_check(checks, differences, f"UPSTREAM_{name.upper()}_IDENTITY", identity_ok, True, identity_ok)

    required_true = set(spec.get("required_authorizations", []))
    required_false = set(spec.get("required_false_authorizations", []))
    if spec.get("legacy_no_authorizations_object"):
        summary = manifest.get("summary", {})
        auth_ok = pointer.get("authorizations") is None and all(
            summary.get(field) is False
            for field in spec.get("required_false_manifest_summary", [])
        )
    else:
        authorizations = pointer.get("authorizations")
        boolean_keys = (
            {key for key, value in authorizations.items() if isinstance(value, bool)}
            if isinstance(authorizations, dict)
            else set()
        )
        auth_ok = (
            isinstance(authorizations, dict)
            and boolean_keys == required_true | required_false
            and all(authorizations.get(key) is True for key in required_true)
            and all(authorizations.get(key) is False for key in required_false)
        )
    add_check(checks, differences, f"UPSTREAM_{name.upper()}_AUTHORIZATIONS", auth_ok, True, auth_ok)

    outputs = parquet_receipts(manifest)
    required_tables = list(spec.get("required_tables", []))
    prohibited_tables = list(spec.get("prohibited_tables", []))
    table_contract_ok = (
        bool(required_tables)
        and len(required_tables) == len(set(required_tables))
        and set(required_tables).issubset(outputs)
        and not set(required_tables).intersection(prohibited_tables)
    )
    add_check(checks, differences, f"UPSTREAM_{name.upper()}_TABLE_ALLOWLIST", table_contract_ok, True, table_contract_ok)

    detail_ok = True
    if name == "signal_universe":
        summary = manifest.get("summary", {})
        detail_ok = (
            summary.get("rows") == spec["required_rows"]
            and summary.get("signal_dates") == spec["required_signal_dates"]
            and summary.get("first_signal_date") == spec["required_min_signal_date"]
            and summary.get("last_signal_date") == spec["required_max_signal_date"]
            and summary.get("future_outcome_filter_applied") is False
        )
    elif name == "auxiliary":
        detail_ok = (
            spec["primary_or_unverified_pointer_may_authorize"] is False
            and len(outputs) == spec["required_table_count"]
            and sum(int(item.get("rows", 0)) for item in outputs.values())
            == spec["required_fully_verified_rows"]
            and outputs.get("auxiliary_signal_linkage.parquet", {}).get("rows")
            == spec["required_signal_keys"]
        )
    elif name == "return_reconciliation":
        detail_ok = (
            spec["primary_return_column"] == "authoritative_total_return"
            and spec["primary_return_usable_column"] == "return_history_usable"
            and spec["outcome_availability_column"] == "outcome_availability_date"
            and spec["raw_or_staging_total_return_may_be_consumed"] is False
            and spec["verified_price_return_column"]
            == "raw_return_ex_distributions"
            and spec["verified_price_return_authorized_use"]
            == "adjusted_price_path_features_only"
            and set(spec["verified_price_return_prohibited_uses"])
            == {
                "labels",
                "targets",
                "beta_estimation",
                "risk",
                "pnl",
                "performance",
                "backtests",
            }
            and spec["monthly_return_crosscheck_may_replace_daily_authority"] is False
        )
    add_check(checks, differences, f"UPSTREAM_{name.upper()}_DETAILS", detail_ok, True, detail_ok)
    return {
        "name": name,
        "build_id": pointer.get("build_id"),
        "verification_id": pointer.get("verification_id"),
        "pointer_path": spec["verified_pointer"],
        "pointer_sha256": pointer_hash,
        "manifest_path": spec["primary_manifest"],
        "manifest_sha256": manifest_hash,
        "consumer_tables": required_tables,
        "prohibited_tables": prohibited_tables,
    }


def _feature_dictionary_bytes(features: list[dict[str, Any]]) -> bytes:
    rows: list[dict[str, Any]] = []
    excluded = {"name", "family", "kind", "source_set", "unit", "formula_id"}
    for order, feature in enumerate(features, start=1):
        parameters = {key: value for key, value in feature.items() if key not in excluded}
        rows.append(
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
    return csv_bytes(
        [
            "feature_order",
            "feature_name",
            "family",
            "kind",
            "source_set",
            "unit",
            "formula_id",
            "parameters_json",
        ],
        rows,
    )


def _inputs_unchanged(observed_inputs: Mapping[Path, str]) -> bool:
    try:
        return all(sha256_file(path) == digest for path, digest in observed_inputs.items())
    except OSError:
        return False


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


@contextmanager
def _publication_lock(artifact_root: Path) -> Iterable[None]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_root / ".phase_03_contract.lock"
    if lock_path.is_symlink():
        raise Phase3VerificationError(f"Unsafe publication lock symlink: {lock_path}")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def _remove_staging(staging: Path) -> None:
    if not staging.exists():
        return
    for child in staging.iterdir():
        if not child.is_file():
            raise Phase3VerificationError(f"Unexpected verification staging directory: {child}")
        child.unlink()
    staging.rmdir()


def _install_or_reuse(staging: Path, final_dir: Path) -> None:
    expected_names = {"independent_check.json", "differences.csv"}
    if not final_dir.exists():
        os.replace(staging, final_dir)
        _fsync_directory(final_dir.parent)
        return
    if not _closed_regular_file_set(final_dir, expected_names):
        raise Phase3VerificationError(f"Existing verification bundle has an unexpected file set: {final_dir}")
    for name in sorted(expected_names):
        existing = final_dir / name
        candidate = staging / name
        if (
            not existing.is_file()
            or existing.is_symlink()
            or sha256_file(existing) != sha256_file(candidate)
            or existing.stat().st_size != candidate.stat().st_size
        ):
            raise Phase3VerificationError(
                f"Existing deterministic verification differs from reconstruction: {final_dir}"
            )
    _remove_staging(staging)


def _verification_bundle_matches(
    final_dir: Path, result_bytes: bytes, differences_bytes: bytes
) -> bool:
    expected = {
        "independent_check.json": result_bytes,
        "differences.csv": differences_bytes,
    }
    if not _closed_regular_file_set(final_dir, set(expected)):
        return False
    try:
        return all((final_dir / name).read_bytes() == value for name, value in expected.items())
    except OSError:
        return False


def verify(
    policy_path: Path = DEFAULT_POLICY,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    policy_path = policy_path.resolve()
    artifact_root = artifact_root.resolve()
    checks: list[dict[str, Any]] = []
    differences: list[dict[str, str]] = []
    observed_inputs: dict[Path, str] = {}
    verifier_path = Path(__file__).resolve()
    verifier_sha = sha256_file(verifier_path)

    policy = observe_toml(policy_path, observed_inputs)
    observed_policy_hash = observed_inputs[policy_path]
    add_check(checks, differences, "REVIEWED_POLICY_HASH", observed_policy_hash == EXPECTED_POLICY_SHA256, EXPECTED_POLICY_SHA256, observed_policy_hash)
    add_check(checks, differences, "POLICY_SCHEMA", policy.get("schema_version") == 1, 1, policy.get("schema_version"))
    expected_policy_status = "frozen_development_only_contract_pending_primary_build_and_independent_verification"
    add_check(checks, differences, "POLICY_STATUS", policy.get("status") == expected_policy_status, expected_policy_status, policy.get("status"))

    registry_spec = policy["feature_registry"]
    registry_path = project_path(registry_spec["path"])
    registry = observe_json(registry_path, observed_inputs)
    observed_registry_hash = observed_inputs[registry_path]
    registry_hash_ok = (
        observed_registry_hash == EXPECTED_REGISTRY_SHA256
        and registry_spec.get("sha256") == EXPECTED_REGISTRY_SHA256
    )
    add_check(checks, differences, "REVIEWED_REGISTRY_HASH", registry_hash_ok, EXPECTED_REGISTRY_SHA256, {"policy": registry_spec.get("sha256"), "file": observed_registry_hash})

    methodology = policy["methodology"]
    for name, hash_name in (("source", "source_sha256"), ("freeze_manifest", "freeze_manifest_sha256")):
        path = project_path(methodology[name])
        observe_bytes(path, observed_inputs)
        observed = observed_inputs[path]
        add_check(checks, differences, f"METHODOLOGY_{name.upper()}_HASH", observed == methodology[hash_name], methodology[hash_name], observed)

    upstream = [
        reconstruct_upstream(name, spec, checks, differences, observed_inputs)
        for name, spec in policy["upstream"].items()
    ]
    add_check(checks, differences, "FIVE_DIRECT_VERIFIED_GATES", len(upstream) == 5, 5, len(upstream))
    database = reconstruct_database(policy["database"], checks, differences, observed_inputs)

    features = independently_expand(registry)
    names = [feature["name"] for feature in features]
    kind_counts = {
        kind: sum(feature["kind"] == kind for feature in features)
        for kind in EXPECTED_KINDS
    }
    family_counts: dict[str, int] = {}
    for feature in features:
        family_counts[feature["family"]] = family_counts.get(feature["family"], 0) + 1
    expected_family_counts = {
        "aggregate_state": 23,
        "optionmetrics": 26,
        "ownership_filings_events_segments": 32,
        "price_trends_path": 20,
        "risk_return_distribution": 20,
        "size_liquidity_trading_frictions": 13,
        "valuation_fundamentals": 29,
    }
    add_check(checks, differences, "FEATURE_COUNT", len(features) == 163 == registry_spec["required_economic_feature_count"] == registry.get("economic_feature_count"), 163, len(features))
    add_check(checks, differences, "FEATURE_NAMES_UNIQUE_CANONICAL", len(names) == len(set(names)) and all(SNAKE_CASE.fullmatch(name) for name in names), True, len(names) == len(set(names)) and all(SNAKE_CASE.fullmatch(name) for name in names))
    add_check(checks, differences, "FEATURE_KIND_COUNTS", kind_counts == EXPECTED_KINDS, EXPECTED_KINDS, kind_counts)
    add_check(checks, differences, "FEATURE_FAMILY_COUNTS", family_counts == expected_family_counts and len(registry["family_order"]) == 7, expected_family_counts, family_counts)
    formula_ok = all(feature["formula_id"] in registry["formula_catalog"] for feature in features)
    add_check(checks, differences, "FORMULA_REFERENCES", formula_ok, True, formula_ok)
    horizons = {
        str(feature["nominal_days"]): feature["vendor_days"]
        for feature in features
        if feature["formula_id"] == "option_historical_volatility"
    }
    expected_horizons = {"30": 30, "60": 60, "90": 91, "180": 182, "360": 365}
    add_check(checks, differences, "OPTIONMETRICS_HORIZONS", horizons == expected_horizons, expected_horizons, horizons)

    scope = policy["scope"]
    closed_scope = (
        scope["last_permitted_predictor_signal_date"] == "2014-10-31"
        and scope["last_permitted_label_exit_date"] == "2014-12-01"
        and all(
            scope[key] is False
            for key in (
                "validation_2015_2019_payload_or_outcome_access_permitted",
                "final_2020_2025_payload_or_outcome_access_permitted",
                "model_fitting_permitted",
                "portfolio_optimization_permitted",
                "portfolio_or_backtest_permitted",
                "pnl_or_performance_permitted",
            )
        )
    )
    add_check(checks, differences, "DEVELOPMENT_SCOPE_CLOSED", closed_scope, True, closed_scope)
    point_in_time = policy["point_in_time"]
    point_in_time_ok = (
        point_in_time["require_observation_date_not_after_signal"] is True
        and point_in_time["require_public_availability_date_not_after_signal"] is True
        and point_in_time["quarterly_rdq_before_datadate_known_rows"] == 43
        and point_in_time["compustat_is_true_historical_vintage"] is False
    )
    add_check(checks, differences, "POINT_IN_TIME_GUARDS", point_in_time_ok, True, point_in_time_ok)
    preprocessing = policy["preprocessing"]
    preprocessing_ok = (
        preprocessing["quantile_algorithm"].startswith("Hyndman-Fan type 7")
        and preprocessing["centered_rank_formula"]
        == "((average_rank - 0.5) / N_nonmissing) - 0.5"
        and preprocessing["dedicated_missing_indicator_for_every_continuous"] is True
    )
    add_check(checks, differences, "PREPROCESSING_RULES", preprocessing_ok, True, preprocessing_ok)
    labels = policy["labels"]
    label_ok = (
        labels["primary"]["name"] == "residual_return_rank"
        and labels["primary"]["minimum_valid_labels_per_signal"] == 100
        and labels["beta_label"]["lookback_sessions"] == 252
        and labels["beta_label"]["minimum_valid_observations"] == 126
        and labels["beta_label"]["factors"] == ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
        and labels["beta_label"]["intercept"] is True
        and labels["beta_label"]["intercept_is_subtracted_from_forward_label"] is False
        and labels["forward_interval"]["signal_to_entry_return_credited"] is False
        and labels["forward_interval"]["multi_session_return_use"].startswith("label missing")
        and "covered_trading_sessions==1" in labels["forward_interval"]["valid_forward_return_legs"]
        and labels["primary"]["unranked_formula"].startswith("sum over one-session legs")
    )
    add_check(checks, differences, "RESIDUAL_LABEL_ONE_SESSION_RULES", label_ok, True, label_ok)
    folds = policy["splits"]["folds"]
    folds_ok = (
        [fold["fold_id"] for fold in folds] == [f"dev_fold_{number}" for number in range(1, 5)]
        and [fold["fit_signal_date"] for fold in folds]
        == ["2006-12-29", "2008-12-31", "2010-12-31", "2012-12-31"]
        and [fold["purged_holding_month"] for fold in folds]
        == ["2006-12", "2008-12", "2010-12", "2012-12"]
        and folds[-1]["validation_last_holding_month"] == "2014-11"
    )
    add_check(checks, differences, "FOUR_PURGED_FOLDS", folds_ok, True, folds_ok)

    primary_implementation = project_path(PRIMARY_IMPLEMENTATION_PATH)
    observe_bytes(primary_implementation, observed_inputs)
    primary_implementation_sha = observed_inputs[primary_implementation]
    evidence = {
        "schema_version": 1,
        "contract_id": policy["contract_id"],
        "policy_path": project_label(policy_path),
        "policy_sha256": observed_policy_hash,
        "registry_path": registry_spec["path"],
        "registry_sha256": registry_spec["sha256"],
        "feature_count": len(features),
        "feature_kind_counts": dict(sorted(kind_counts.items())),
        "family_count": len(registry["family_order"]),
        "upstream": upstream,
        "database": database,
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
    build_identity = {
        "contract_id": evidence["contract_id"],
        "policy_sha256": evidence["policy_sha256"],
        "registry_sha256": evidence["registry_sha256"],
        "implementation_path": PRIMARY_IMPLEMENTATION_PATH,
        "implementation_sha256": primary_implementation_sha,
        "upstream": evidence["upstream"],
        "database": evidence["database"],
    }
    build_id = "phase03-contract-" + sha256_bytes(canonical_json_bytes(build_identity))[:16]
    bundle_dir = artifact_root / "contracts" / build_id
    manifest_path = bundle_dir / "manifest.json"
    primary_pointer_path = artifact_root / "LATEST_CONTRACT_PRIMARY.json"

    add_check(checks, differences, "PRIMARY_BUNDLE_LOCATION", bundle_dir.is_dir(), project_label(bundle_dir), project_label(bundle_dir) if bundle_dir.is_dir() else None)
    expected_file_set = {"contract.json", "feature_dictionary.csv", "purged_splits.csv", "manifest.json"}
    actual_file_set = {item.name for item in bundle_dir.iterdir()} if bundle_dir.is_dir() else set()
    closed_file_set = _closed_regular_file_set(bundle_dir, expected_file_set)
    add_check(checks, differences, "PRIMARY_BUNDLE_FILE_SET", closed_file_set, sorted(expected_file_set), sorted(actual_file_set))

    expected_dictionary = _feature_dictionary_bytes(features)
    split_fields = [
        "fold_id",
        "fit_signal_date",
        "train_last_eligible_holding_month",
        "purged_holding_month",
        "validation_first_holding_month",
        "validation_last_holding_month",
    ]
    expected_splits = csv_bytes(split_fields, folds)
    expected_contract = {
        **evidence,
        "build_id": build_id,
        "implementation_sha256": primary_implementation_sha,
        "authorizations": PRIMARY_AUTHORIZATIONS,
        "status": "primary_passed_pending_independent_verification",
    }
    expected_contract_bytes = canonical_json_bytes(expected_contract)
    expected_outputs = {
        "contract.json": {
            "sha256": sha256_bytes(expected_contract_bytes),
            "size_bytes": len(expected_contract_bytes),
        },
        "feature_dictionary.csv": {
            "sha256": sha256_bytes(expected_dictionary),
            "size_bytes": len(expected_dictionary),
        },
        "purged_splits.csv": {
            "sha256": sha256_bytes(expected_splits),
            "size_bytes": len(expected_splits),
        },
    }
    expected_manifest = {
        "schema_version": 1,
        "contract_id": evidence["contract_id"],
        "build_id": build_id,
        "status": "primary_passed_pending_independent_verification",
        "identity": build_identity,
        "outputs": expected_outputs,
        "authorizations": PRIMARY_AUTHORIZATIONS,
    }
    expected_manifest_bytes = canonical_json_bytes(expected_manifest)
    expected_pointer = {
        "schema_version": 1,
        "contract_id": evidence["contract_id"],
        "build_id": build_id,
        "status": "primary_passed_pending_independent_verification",
        "manifest_path": project_label(manifest_path),
        "manifest_sha256": sha256_bytes(expected_manifest_bytes),
        "authorizations": PRIMARY_AUTHORIZATIONS,
    }

    expected_bytes = {
        "contract.json": expected_contract_bytes,
        "feature_dictionary.csv": expected_dictionary,
        "purged_splits.csv": expected_splits,
        "manifest.json": expected_manifest_bytes,
    }
    for name, expected in expected_bytes.items():
        path = bundle_dir / name
        if path.is_file():
            actual = observe_bytes(path, observed_inputs)
            actual_hash: str | None = sha256_bytes(actual)
        else:
            actual = None
            actual_hash = None
        add_check(checks, differences, f"RECONSTRUCTED_{name.upper().replace('.', '_')}_BYTES", actual == expected, sha256_bytes(expected), actual_hash)

    if primary_pointer_path.is_file():
        pointer_bytes = observe_bytes(primary_pointer_path, observed_inputs)
        pointer_hash: str | None = sha256_bytes(pointer_bytes)
    else:
        pointer_bytes = None
        pointer_hash = None
    expected_pointer_bytes = canonical_json_bytes(expected_pointer)
    add_check(checks, differences, "RECONSTRUCTED_PRIMARY_POINTER_BYTES", pointer_bytes == expected_pointer_bytes, sha256_bytes(expected_pointer_bytes), pointer_hash)

    fail_count = sum(check["status"] == "FAIL" for check in checks)
    differences_bytes = differences_csv_bytes(differences)
    verification_identity = {
        "build_id": build_id,
        "primary_manifest_sha256": sha256_bytes(expected_manifest_bytes),
        "verifier_implementation_sha256": verifier_sha,
        "checks_sha256": sha256_bytes(canonical_json_bytes(checks)),
        "differences_sha256": sha256_bytes(differences_bytes),
    }
    verification_id = "verify-" + sha256_bytes(canonical_json_bytes(verification_identity))[:16]
    result = {
        "schema_version": 1,
        "contract_id": evidence["contract_id"],
        "build_id": build_id,
        "verification_id": verification_id,
        "status": "verified_passed" if fail_count == 0 else "verified_failed",
        "primary_manifest_sha256": sha256_bytes(expected_manifest_bytes),
        "verifier_implementation_sha256": verifier_sha,
        "independence": {
            "primary_contract_module_imported": False,
            "payload_tables_opened": False,
            "validation_or_final_data_opened": False,
        },
        "summary": {
            "check_count": len(checks),
            "pass_count": len(checks) - fail_count,
            "fail_count": fail_count,
            "difference_count": len(differences),
        },
        "checks": checks,
        "authorizations": {
            "development_feature_and_label_build_authorized": fail_count == 0,
            "models_portfolios_pnl_performance_or_backtests_authorized": False,
            "validation_or_final_test_access_authorized": False,
        },
    }
    final_dir = artifact_root / "contract_verifications" / build_id / verification_id
    if not _safe_artifact_descendant(artifact_root, final_dir):
        raise Phase3VerificationError(f"Unsafe verification artifact path: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if not _safe_artifact_descendant(artifact_root, final_dir):
        raise Phase3VerificationError(
            f"Unsafe verification artifact path after directory creation: {final_dir}"
        )
    staging = Path(tempfile.mkdtemp(prefix=f".{verification_id}.", dir=final_dir.parent))
    try:
        result_bytes = canonical_json_bytes(result)
        (staging / "independent_check.json").write_bytes(result_bytes)
        (staging / "differences.csv").write_bytes(differences_bytes)
        with _publication_lock(artifact_root):
            if not _safe_artifact_descendant(artifact_root, final_dir):
                raise Phase3VerificationError(
                    f"Unsafe verification artifact path at commit: {final_dir}"
                )
            if not _inputs_unchanged(observed_inputs):
                raise Phase3VerificationError("Authoritative inputs changed during independent verification")
            if not _closed_regular_file_set(bundle_dir, expected_file_set):
                raise Phase3VerificationError("Primary bundle file set changed during verification")
            if sha256_file(verifier_path) != verifier_sha:
                raise Phase3VerificationError("Verifier implementation changed during verification")
            _install_or_reuse(staging, final_dir)
            if not _verification_bundle_matches(final_dir, result_bytes, differences_bytes):
                raise Phase3VerificationError(
                    "Installed verification bundle differs from the independent reconstruction"
                )
            if fail_count:
                raise Phase3VerificationError(
                    f"Phase 3 contract verification failed: {fail_count} checks"
                )
            if not _inputs_unchanged(observed_inputs):
                raise Phase3VerificationError("Authoritative inputs changed before verified pointer commit")
            if not _closed_regular_file_set(bundle_dir, expected_file_set):
                raise Phase3VerificationError("Primary bundle file set changed before verified pointer commit")
            if not Path(policy["database"]["active_root"]).is_dir():
                raise Phase3VerificationError("Active database root disappeared before pointer commit")
            if sha256_file(verifier_path) != verifier_sha:
                raise Phase3VerificationError("Verifier implementation changed before pointer commit")
            if not _verification_bundle_matches(final_dir, result_bytes, differences_bytes):
                raise Phase3VerificationError("Verification bundle changed before pointer commit")
            pointer = {
                "schema_version": 1,
                "contract_id": evidence["contract_id"],
                "build_id": build_id,
                "verification_id": verification_id,
                "status": "verified_passed",
                "primary_manifest_sha256": sha256_bytes(expected_manifest_bytes),
                "independent_check_path": project_label(final_dir / "independent_check.json"),
                "independent_check_sha256": sha256_bytes(result_bytes),
                "differences_sha256": sha256_bytes(differences_bytes),
                "authorizations": VERIFIED_AUTHORIZATIONS,
            }
            _atomic_json(artifact_root / "LATEST_CONTRACT_VERIFIED.json", pointer)
        return final_dir
    except Exception:
        _remove_staging(staging)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    print(verify(args.policy, args.artifact_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
