"""Independent Stage 9 selector-results verification and Stage-10-only gate.

The verifier reauthenticates the pointer/manifest/output graph, rejoins the
verified Stage 7 labels, and independently recomputes monthly scores, ICs,
moving-block bootstrap standard errors, fold summaries, gates, and family
shortlists.  It deliberately does not import the primary evaluation or
shortlisting modules.  A zero-difference verification may authorize roadmap
Stage 10 only; Stage 11+, validation/final access, portfolios, P&L, and
backtests remain explicitly prohibited.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata

from portfolio_bot.data.secure_publication import (
    AuthenticatedBundleGuard,
    AuthenticatedPointerGuard,
    SecureStagingDirectory,
    replace_pointer_last,
)
from portfolio_bot.selectors.contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
    canonical_json_bytes,
    project_path,
    sha256_file,
    validate_contract,
)
from portfolio_bot.selectors.inputs import Stage9Inputs, resolve_stage9_inputs
from portfolio_bot.selectors.publication import (
    CHECKS_NAME,
    CONFIG_NAME,
    FAMILY_CONFIG_COUNTS,
    FAMILY_ORDER,
    FIT_RECEIPTS_NAME,
    FOLD_IDS,
    FOLD_NAME,
    INTERIM_FILES,
    MANIFEST_FILES,
    MANIFEST_NAME,
    MONTHLY_NAME,
    OOF_NAME,
    PROCESSED_FILES,
    QUALITY_FILES,
    SEARCH_RECEIPT_NAME,
    SHORTLIST_NAME,
    STOCHASTIC_SEEDS,
    SUCCESS_NAME,
    expected_oof_schema,
    published_schemas,
)


VERIFICATION_CHECKS_NAME = "verification_checks.parquet"
VERIFICATION_REPORT_NAME = "verification_report.json"
VERIFICATION_FILES = (
    VERIFICATION_CHECKS_NAME,
    VERIFICATION_REPORT_NAME,
    SUCCESS_NAME,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BUILD_ID = re.compile(r"phase04-selector-results-[0-9a-f]{24}\Z")
_VERIFY_ID = re.compile(r"verify-[0-9a-f]{16}\Z")
_FLOAT_TOLERANCE = 1e-12
_VERIFICATION_RUNTIME_DISTRIBUTIONS = ("numpy", "pandas", "pyarrow", "scipy")

PRIMARY_AUTHORIZATIONS = {
    "independent_verification_authorized": True,
    "independent_verification_completed": False,
    "stage_10_allocation_architectures_authorized": False,
    "stage_11_or_later_authorized": False,
    "portfolios_pnl_or_backtests_authorized": False,
    "validation_or_final_test_access_authorized": False,
}
VERIFIED_AUTHORIZATIONS = {
    "independent_verification_completed": True,
    "selector_results_authorized_for_stage_10": True,
    "stage_10_allocation_architectures_authorized": True,
    "stage_11_or_later_authorized": False,
    "portfolios_pnl_or_backtests_authorized": False,
    "validation_or_final_test_access_authorized": False,
}


class SelectorVerificationError(RuntimeError):
    """Raised when independent verification cannot authenticate its graph."""


@dataclass(frozen=True)
class Difference:
    code: str
    expected: Any
    observed: Any
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "expected": _json_safe(self.expected),
            "observed": _json_safe(self.observed),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    build_id: str
    verification_id: str | None
    checks: Mapping[str, Any]
    differences: tuple[Difference, ...]
    verification_path: Path | None
    verified_pointer_path: Path | None


@dataclass
class AuthenticatedPrimary:
    policy_path: Path
    policy: Mapping[str, Any]
    pointer_path: Path
    pointer_payload: bytes
    pointer_sha256: str
    manifest_payload: bytes
    manifest_sha256: str
    manifest: Mapping[str, Any]
    inputs: Stage9Inputs
    output_guards: Mapping[str, AuthenticatedBundleGuard]
    artifact_guard: AuthenticatedBundleGuard
    pointer_guard: AuthenticatedPointerGuard
    manifest_guard: AuthenticatedPointerGuard
    manifest_success_guard: AuthenticatedPointerGuard

    @property
    def build_id(self) -> str:
        return str(self.manifest["build_id"])

    def gate(self) -> None:
        self.pointer_guard.gate()
        self.manifest_guard.gate()
        self.manifest_success_guard.gate()
        self.artifact_guard.gate()
        for guard in self.output_guards.values():
            guard.gate()
        current = resolve_stage9_inputs(self.policy_path)
        _require(current.input_receipts == self.inputs.input_receipts, "Stage 9 input graph changed")
        _require(current.authority == self.inputs.authority, "Stage 8 authority changed")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectorVerificationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectorVerificationError(f"Invalid {label} JSON") from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _manifest_success_validator(payload: bytes) -> None:
    value = _json(payload, "manifest success")
    _require(value.get("status") == "PRIMARY_MANIFEST_COMPLETE", "Manifest success status differs")


def _pointer_validator(payload: bytes) -> None:
    value = _json(payload, "primary pointer")
    _require(value.get("status") == "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION", "Primary selector pointer status differs")
    _require(_BUILD_ID.fullmatch(str(value.get("build_id", ""))) is not None, "Primary selector build ID differs")
    _require(value.get("authorizations") == PRIMARY_AUTHORIZATIONS, "Primary selector pointer authorization escaped verification-only scope")
    _require(_SHA256.fullmatch(str(value.get("primary_manifest_sha256", ""))) is not None, "Primary manifest hash is malformed")


def _output_hashes(manifest: Mapping[str, Any], section: str) -> tuple[Path, tuple[str, ...], dict[str, str]]:
    output = manifest["outputs"][section]
    files = output["files"]
    names = tuple(sorted((*files.keys(), SUCCESS_NAME)))
    hashes = {str(name): str(record["sha256"]) for name, record in files.items()}
    hashes[SUCCESS_NAME] = str(output["success_sha256"])
    _require(all(_SHA256.fullmatch(value) is not None for value in hashes.values()), f"Malformed {section} output hash")
    return Path(str(output["path"])), names, hashes


@contextlib.contextmanager
def authenticate_primary(
    *,
    policy_path: Path = DEFAULT_POLICY,
    primary_pointer_path: Path | None = None,
) -> Iterator[AuthenticatedPrimary]:
    """Hold the exact primary pointer, manifest, and three closed outputs."""

    policy, _ = validate_contract(policy_path)
    pointer_path = primary_pointer_path or project_path(policy["publication"]["latest_primary_pointer"])
    inputs = resolve_stage9_inputs(policy_path)
    with contextlib.ExitStack() as stack:
        pointer_guard = stack.enter_context(
            AuthenticatedPointerGuard.open(
                trusted_root=PROJECT_ROOT,
                path=pointer_path,
                payload_validator=_pointer_validator,
                label="Stage 9 primary selector pointer",
            )
        )
        pointer = _json(pointer_guard.payload, "primary pointer")
        build_id = str(pointer["build_id"])
        artifact_root = project_path(policy["publication"]["artifact_root"])
        manifest_directory = artifact_root / policy["publication"]["primary_build_namespace"] / build_id
        manifest_path = manifest_directory / MANIFEST_NAME
        manifest_guard = stack.enter_context(
            AuthenticatedPointerGuard.open(
                trusted_root=PROJECT_ROOT,
                path=manifest_path,
                expected_sha256=str(pointer["primary_manifest_sha256"]),
                label="Stage 9 primary selector manifest",
            )
        )
        manifest = _json(manifest_guard.payload, "primary manifest")
        _require(manifest.get("build_id") == build_id, "Primary pointer/manifest build differs")
        _require(manifest.get("status") == "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION", "Primary manifest status differs")
        _require(manifest.get("authorizations") == PRIMARY_AUTHORIZATIONS, "Primary manifest authorization escaped verification-only scope")
        manifest_success_guard = stack.enter_context(
            AuthenticatedPointerGuard.open(
                trusted_root=PROJECT_ROOT,
                path=manifest_directory / SUCCESS_NAME,
                payload_validator=_manifest_success_validator,
                label="Stage 9 primary manifest success",
            )
        )
        manifest_success = _json(
            manifest_success_guard.payload, "manifest success"
        )
        _require(
            manifest_success.get("build_id") == build_id
            and manifest_success.get("manifest_sha256")
            == manifest_guard.payload_sha256,
            "Primary manifest success binding differs",
        )
        artifact_guard = stack.enter_context(
            AuthenticatedBundleGuard.open(
                trusted_root=PROJECT_ROOT,
                path=manifest_directory,
                expected_files=MANIFEST_FILES,
                expected_hashes={
                    MANIFEST_NAME: manifest_guard.payload_sha256,
                    SUCCESS_NAME: manifest_success_guard.payload_sha256,
                },
                label="Stage 9 primary manifest bundle",
            )
        )
        database_root = inputs.database_root
        output_guards: dict[str, AuthenticatedBundleGuard] = {}
        expected_sets = {
            "interim": tuple(sorted(INTERIM_FILES)),
            "processed": tuple(sorted(PROCESSED_FILES)),
            "quality": tuple(sorted(QUALITY_FILES)),
        }
        for section in ("interim", "processed", "quality"):
            relative, names, hashes = _output_hashes(manifest, section)
            _require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe primary {section} path")
            _require(names == expected_sets[section], f"Primary {section} closed set differs")
            output_guards[section] = stack.enter_context(
                AuthenticatedBundleGuard.open(
                    trusted_root=database_root,
                    path=database_root / relative,
                    expected_files=names,
                    expected_hashes=hashes,
                    label=f"Stage 9 primary selector {section}",
                )
            )
        _require(manifest["inputs"]["table_receipts"] == list(inputs.input_receipts), "Primary input receipts differ")
        bindings = manifest.get("bindings")
        _require(isinstance(bindings, Mapping), "Primary selector bindings are missing")
        for group in ("code_hashes", "configuration_hashes"):
            records = bindings.get(group)
            _require(isinstance(records, Mapping) and records, f"Primary {group} are missing")
            for path_value, expected_hash in records.items():
                path = project_path(str(path_value))
                _require(
                    path.is_file()
                    and not path.is_symlink()
                    and sha256_file(path) == str(expected_hash),
                    f"Primary bound file changed: {path_value}",
                )
        input_receipts_hash = _sha256_bytes(
            canonical_json_bytes(list(inputs.input_receipts))
        )
        _require(
            bindings.get("input_receipts_sha256") == input_receipts_hash,
            "Primary Stage 9 input-receipt binding differs",
        )
        search_hashes = bindings.get("search_file_hashes")
        _require(isinstance(search_hashes, Mapping), "Primary search-file hashes are missing")
        published_source_hashes: dict[str, str] = {}
        for section in ("interim", "processed"):
            for name, record in manifest["outputs"][section]["files"].items():
                if name in search_hashes:
                    published_source_hashes[str(name)] = str(record["sha256"])
        _require(
            dict(search_hashes) == published_source_hashes,
            "Primary outputs do not preserve every completed-search payload hash",
        )
        artifacts = AuthenticatedPrimary(
            policy_path=Path(policy_path),
            policy=policy,
            pointer_path=pointer_path,
            pointer_payload=pointer_guard.payload,
            pointer_sha256=pointer_guard.payload_sha256,
            manifest_payload=manifest_guard.payload,
            manifest_sha256=manifest_guard.payload_sha256,
            manifest=manifest,
            inputs=inputs,
            output_guards=output_guards,
            artifact_guard=artifact_guard,
            pointer_guard=pointer_guard,
            manifest_guard=manifest_guard,
            manifest_success_guard=manifest_success_guard,
        )
        artifacts.gate()
        yield artifacts
        artifacts.gate()


def _read_table(guard: AuthenticatedBundleGuard, name: str) -> pa.Table:
    with guard.open_binary(name) as stream:
        return pq.read_table(stream, use_threads=False).combine_chunks()


def _read_json_member(guard: AuthenticatedBundleGuard, name: str) -> dict[str, Any]:
    with guard.open_binary(name) as stream:
        payload = stream.read()
    return _json(payload, name)


def _authenticated_tables(artifacts: AuthenticatedPrimary) -> tuple[dict[str, pa.Table], Mapping[str, Any]]:
    interim = artifacts.output_guards["interim"]
    processed = artifacts.output_guards["processed"]
    tables = {
        OOF_NAME: _read_table(interim, OOF_NAME),
        FIT_RECEIPTS_NAME: _read_table(interim, FIT_RECEIPTS_NAME),
        MONTHLY_NAME: _read_table(processed, MONTHLY_NAME),
        CONFIG_NAME: _read_table(processed, CONFIG_NAME),
        FOLD_NAME: _read_table(processed, FOLD_NAME),
        SHORTLIST_NAME: _read_table(processed, SHORTLIST_NAME),
        CHECKS_NAME: _read_table(artifacts.output_guards["quality"], CHECKS_NAME),
    }
    receipt = _read_json_member(processed, SEARCH_RECEIPT_NAME)
    quality_summary = _read_json_member(
        artifacts.output_guards["quality"], "selector_publication_summary.json"
    )
    expected = expected_oof_schema(receipt)
    _require(tables[OOF_NAME].schema.equals(expected, check_metadata=True), "Authenticated OOF schema differs")
    for name in (FIT_RECEIPTS_NAME, MONTHLY_NAME, CONFIG_NAME, FOLD_NAME, SHORTLIST_NAME):
        _require(tables[name].schema.equals(published_schemas()[name], check_metadata=True), f"Authenticated schema differs: {name}")
    _require(
        tables[CHECKS_NAME].schema.equals(
            published_schemas()[CHECKS_NAME], check_metadata=True
        )
        and set(tables[CHECKS_NAME]["status"].to_pylist()) == {"PASS"},
        "Primary publication checks differ",
    )
    _require(
        quality_summary.get("status") == "PRIMARY_CHECKS_PASSED"
        and quality_summary.get("scope")
        == {
            "development_only": True,
            "validation_or_final_rows_read": 0,
            "portfolio_weights_created": 0,
            "pnl_or_backtests_created": 0,
        }
        and quality_summary.get("hard_gates") == artifacts.manifest.get("hard_gates"),
        "Primary publication quality summary differs",
    )
    return tables, receipt


def _independent_fit_grid(
    fits: pd.DataFrame,
    receipt: Mapping[str, Any],
    oof: pd.DataFrame,
    configs: pd.DataFrame,
) -> tuple[dict[str, Any], list[Difference]]:
    differences: list[Difference] = []
    score_registry = list(receipt["column_registry"]["monthly_scores"])
    config_ids_by_family = {
        family: [
            str(item["config_id"])
            for item in score_registry
            if item["model_family"] == family
        ]
        for family in FAMILY_ORDER
    }
    expected: set[tuple[str, str, str, int | None]] = set()
    for family in FAMILY_ORDER:
        seeds: tuple[int | None, ...] = (
            (None,) if family == "elastic_net" else STOCHASTIC_SEEDS
        )
        expected.update(
            (family, config_id, fold_id, seed)
            for config_id in config_ids_by_family[family]
            for fold_id in FOLD_IDS
            for seed in seeds
        )
    observed: set[tuple[str, str, str, int | None]] = set()
    for row in fits.itertuples(index=False):
        seed = (
            None
            if row.seed is None
            or (isinstance(row.seed, float) and math.isnan(row.seed))
            else int(row.seed)
        )
        observed.add(
            (str(row.model_family), str(row.config_id), str(row.fold_id), seed)
        )
    if observed != expected or len(fits) != len(expected):
        differences.append(
            Difference(
                "fit_grid",
                {"rows": len(expected), "identities": len(expected)},
                {"rows": len(fits), "identities": len(observed)},
                "Fold/configuration/seed fit grid is incomplete or duplicated",
            )
        )
    if not fits["job_id"].is_unique or not (fits["job_status"] == "COMPLETE").all():
        differences.append(
            Difference(
                "fit_completion",
                "unique COMPLETE fit receipts",
                "duplicate or incomplete receipts",
                "Fit receipt status differs",
            )
        )
    if not fits["fit_converged"].astype(bool).all():
        differences.append(
            Difference(
                "fit_convergence",
                "all authoritative fits converged",
                int((~fits["fit_converged"].astype(bool)).sum()),
                "One or more completed fit receipts lack convergence evidence",
            )
        )
    valid_hashes = fits["prediction_sha256"].astype(str).map(
        lambda value: _SHA256.fullmatch(value) is not None
    )
    if not bool(valid_hashes.all()):
        differences.append(
            Difference(
                "fit_prediction_hashes",
                "64-character lowercase SHA-256",
                int((~valid_hashes).sum()),
                "Malformed fit-prediction receipts",
            )
        )
    fold_rows = oof.groupby("fold_id", observed=True).size().to_dict()
    mismatched_rows = sum(
        int(row.prediction_rows) != int(fold_rows.get(str(row.fold_id), -1))
        for row in fits.itertuples(index=False)
    )
    if mismatched_rows:
        differences.append(
            Difference(
                "fit_prediction_rows",
                "each fit equals its complete OOF fold row count",
                mismatched_rows,
                "Fit receipt prediction shape differs",
            )
        )
    complexity_mismatches = 0
    kkt_mismatches = 0
    noncanonical_ledger_rows = 0
    for config in configs.itertuples(index=False):
        try:
            configuration = json.loads(str(config.configuration_json))
            regularization_key = json.loads(str(config.regularization_key_json))
            noncanonical_ledger_rows += int(
                not isinstance(configuration, dict)
                or not isinstance(regularization_key, list)
                or not regularization_key
                or json.dumps(
                    configuration,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                != str(config.configuration_json)
                or json.dumps(
                    regularization_key,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                != str(config.regularization_key_json)
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            configuration = {}
            noncanonical_ledger_rows += 1
        selected = fits.loc[
            fits["model_family"].eq(config.model_family)
            & fits["config_id"].eq(config.config_id)
        ]
        if config.model_family == "neural_network":
            counts = selected["parameter_count"].dropna().to_numpy(
                dtype=np.float64
            )
            observed_complexity = (
                float(np.median(counts))
                if len(counts) == len(selected)
                else math.nan
            )
        elif config.model_family == "xgboost":
            try:
                depth = int(configuration["max_depth"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                depth = -1
            trees = selected["best_tree_count"].dropna().to_numpy(dtype=np.float64)
            observed_complexity = (
                depth * float(np.median(trees))
                if depth > 0 and len(trees) == len(selected)
                else math.nan
            )
        else:
            nonzero = selected["nonzero_coefficients"].dropna().to_numpy(dtype=np.float64)
            observed_complexity = (
                float(np.median(nonzero))
                if len(nonzero) == len(selected)
                else math.nan
            )
            dual = selected["dual_gap"].to_numpy(dtype=np.float64)
            dual_limit = selected["dual_gap_limit"].to_numpy(dtype=np.float64)
            violation = selected["kkt_max_violation"].to_numpy(dtype=np.float64)
            limit = selected["kkt_limit"].to_numpy(dtype=np.float64)
            kkt_mismatches += int(
                not (
                    np.isfinite(dual).all()
                    and np.isfinite(dual_limit).all()
                    and np.isfinite(violation).all()
                    and np.isfinite(limit).all()
                    and (dual >= 0.0).all()
                    and (dual_limit >= 0.0).all()
                    and (dual <= dual_limit).all()
                    and (violation >= 0.0).all()
                    and (limit >= 0.0).all()
                    and (violation <= limit).all()
                )
            )
        complexity_mismatches += int(
            not math.isfinite(observed_complexity)
            or not math.isclose(
                observed_complexity,
                float(config.complexity),
                rel_tol=0.0,
                abs_tol=_FLOAT_TOLERANCE,
            )
        )
    if complexity_mismatches:
        differences.append(
            Difference(
                "configuration_complexity",
                "NN median parameter count; XGB depth×median trees; EN median nonzero coefficients",
                complexity_mismatches,
                "Published complexity differs from fit receipts",
            )
        )
    if kkt_mismatches:
        differences.append(
            Difference(
                "elastic_net_kkt",
                "finite nonnegative dual gap <= dual-gap limit and KKT violation <= KKT limit",
                kkt_mismatches,
                "Elastic Net convergence certificate differs",
            )
        )
    phases = (
        configs.groupby(["model_family", "trial_phase"], observed=True)
        .size()
        .to_dict()
    )
    expected_phases = {
        ("neural_network", "full_grid"): 48,
        ("elastic_net", "full_grid"): 250,
        ("xgboost", "core_grid"): 40,
        ("xgboost", "gamma_challenger"): 8,
    }
    if phases != expected_phases or noncanonical_ledger_rows:
        differences.append(
            Difference(
                "configuration_trial_ledger",
                {"phases": expected_phases, "canonical_rows": 346},
                {
                    "phases": phases,
                    "noncanonical_rows": noncanonical_ledger_rows,
                },
                "Configuration trial ledger differs",
            )
        )
    gamma = configs.loc[
        configs["model_family"].eq("xgboost")
        & configs["trial_phase"].eq("gamma_challenger")
    ]
    gamma_layout_ok = (
        gamma["parent_config_id"].notna().all()
        and gamma["parent_config_id"].nunique() == 4
    )
    if gamma_layout_ok:
        for _, group in gamma.groupby("parent_config_id", observed=True):
            try:
                values = sorted(
                    float(json.loads(value)["gamma"])
                    for value in group["configuration_json"]
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                values = []
            gamma_layout_ok = gamma_layout_ok and values == [0.1, 1.0]
    if not gamma_layout_ok:
        differences.append(
            Difference(
                "xgboost_gamma_ledger",
                "four parents, each with gamma 0.1 and 1.0",
                "gamma challenger layout differs",
                "XGBoost gamma trial ledger is incomplete",
            )
        )
    return (
        {
            "fit_receipts_reauthenticated": len(fits),
            "fit_identities_recomputed": len(expected),
            "fit_prediction_row_shapes_recomputed": len(fits),
            "configuration_complexities_recomputed": len(configs),
            "elastic_net_kkt_certificates_recomputed": 1_000,
            "configuration_trial_ledger_rows_recomputed": len(configs),
        },
        differences,
    )


def _expected_oof_labels(inputs: Stage9Inputs) -> pd.DataFrame:
    labels = inputs.read_development_labels().to_pandas()
    split = inputs.read_split_membership().to_pandas()
    validation = split.loc[split["included_in_validation"].astype(bool), ["fold_id", "signal_date", "holding_month"]]
    _require(len(validation) == 95 and not validation.duplicated().any(), "Verified Stage 7 validation membership differs")
    joined = validation.merge(
        labels[["signal_date", "holding_month", "permno", "unranked_residual_return", "primary_label_status"]],
        on=["signal_date", "holding_month"],
        how="left",
        validate="one_to_many",
        sort=False,
    )
    joined["outcome_valid"] = joined["primary_label_status"].eq("PRESENT")
    _require(len(joined) == 172_969 and int(joined["outcome_valid"].sum()) == 172_799, "Verified Stage 7 OOF outcome domain differs")
    _require(
        np.isfinite(
            pd.to_numeric(
                joined.loc[joined["outcome_valid"], "unranked_residual_return"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
        ).all(),
        "Verified Stage 7 PRESENT OOF residual is nonfinite",
    )
    joined.loc[~joined["outcome_valid"], "unranked_residual_return"] = np.nan
    return joined.sort_values(["fold_id", "signal_date", "holding_month", "permno"], kind="mergesort").reset_index(drop=True)


def _column_registry(receipt: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    registry = receipt["column_registry"]
    return list(registry["raw_forecasts"]), list(registry["monthly_scores"])


def _raw_means(oof: pd.DataFrame, receipt: Mapping[str, Any]) -> np.ndarray:
    raw, scores = _column_registry(receipt)
    result = np.empty((len(oof), len(scores)), dtype=np.float32)
    raw_offset = 0
    score_offset = 0
    for family in FAMILY_ORDER:
        count = FAMILY_CONFIG_COUNTS[family]
        seed_count = 1 if family == "elastic_net" else 3
        columns = [str(item["column_name"]) for item in raw[raw_offset : raw_offset + count * seed_count]]
        values = oof[columns].to_numpy(dtype=np.float32, copy=False).reshape(len(oof), count, seed_count)
        if seed_count == 1:
            result[:, score_offset : score_offset + count] = values[:, :, 0]
        else:
            result[:, score_offset : score_offset + count] = np.mean(values, axis=2, dtype=np.float64).astype(np.float32)
        raw_offset += count * seed_count
        score_offset += count
    _require(np.isfinite(result).all(), "OOF raw seed average is nonfinite")
    return result


def _independent_scores(raw_mean: np.ndarray, months: np.ndarray) -> np.ndarray:
    scores = np.empty(raw_mean.shape, dtype=np.float32)
    for month in np.unique(months):
        positions = np.flatnonzero(months == month)
        ranks = rankdata(raw_mean[positions].astype(np.float64), method="average", axis=0)
        scores[positions] = (((ranks - 0.5) / float(len(positions))) - 0.5).astype(np.float32)
    _require(np.isfinite(scores).all(), "Independent monthly score is nonfinite")
    return scores


def _correlations(matrix: np.ndarray, target: np.ndarray, *, spearman: bool) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if spearman:
        x = rankdata(x, method="average", axis=0)
        y = rankdata(y, method="average")
    x -= np.mean(x, axis=0)
    y -= float(np.mean(y))
    x_ss = np.sum(x * x, axis=0)
    y_ss = float(np.dot(y, y))
    constant = x_ss == 0.0
    values = np.zeros(x.shape[1], dtype=np.float64)
    if y_ss != 0.0:
        usable = ~constant
        values[usable] = (x[:, usable].T @ y) / np.sqrt(x_ss[usable] * y_ss)
    statuses = np.full(x.shape[1], "OK", dtype=object)
    if y_ss == 0.0:
        statuses[:] = "CONSTANT_RIGHT"
        statuses[constant] = "CONSTANT_BOTH"
    else:
        statuses[constant] = "CONSTANT_LEFT"
    return np.clip(values, -1.0, 1.0), statuses


def _independent_monthly(
    labels: pd.DataFrame,
    raw_mean: np.ndarray,
    scores: np.ndarray,
    receipt: Mapping[str, Any],
) -> pd.DataFrame:
    _, score_registry = _column_registry(receipt)
    residual = labels["unranked_residual_return"].to_numpy(dtype=np.float64)
    valid = labels["outcome_valid"].to_numpy(dtype=bool)
    records: list[dict[str, Any]] = []
    months = labels["holding_month"].to_numpy()
    for month in np.unique(months):
        positions = np.flatnonzero(months == month)
        evaluation = positions[valid[positions]]
        rank_ic, rank_status = _correlations(scores[evaluation], residual[evaluation], spearman=True)
        linear_ic, linear_status = _correlations(raw_mean[evaluation], residual[evaluation], spearman=False)
        fold = str(labels.iloc[positions[0]]["fold_id"])
        signal = labels.iloc[positions[0]]["signal_date"]
        for index, item in enumerate(score_registry):
            records.append(
                {
                    "model_family": str(item["model_family"]),
                    "config_id": str(item["config_id"]),
                    "fold_id": fold,
                    "signal_date": signal,
                    "holding_month": month,
                    "scored_rows": len(positions),
                    "evaluation_rows": len(evaluation),
                    "excluded_outcome_rows": len(positions) - len(evaluation),
                    "rank_ic": float(rank_ic[index]),
                    "rank_ic_status": str(rank_status[index]),
                    "linear_ic": float(linear_ic[index]),
                    "linear_ic_status": str(linear_status[index]),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(["model_family", "config_id", "holding_month"], kind="mergesort").reset_index(drop=True)


def _bootstrap_plan(month_count: int) -> np.ndarray:
    block = 6
    replicates = 10_000
    blocks = math.ceil(month_count / block)
    rng = np.random.Generator(np.random.PCG64(20_260_714))
    starts = rng.integers(0, month_count - block + 1, size=(replicates, blocks), dtype=np.int32)
    return (starts[:, :, None] + np.arange(block, dtype=np.int32)).reshape(replicates, -1)[:, :month_count]


def _independent_aggregate(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    months = sorted(monthly["holding_month"].unique())
    plan = _bootstrap_plan(len(months))
    for (family, config_id), config in monthly.groupby(["model_family", "config_id"], sort=True, observed=True):
        by_month = config.set_index("holding_month").reindex(months)
        rank_values = by_month["rank_ic"].to_numpy(dtype=np.float64)
        linear_values = by_month["linear_ic"].to_numpy(dtype=np.float64)
        fold_means: list[float] = []
        for fold_id, fold in config.groupby("fold_id", sort=True, observed=True):
            rank_mean = float(np.mean(fold["rank_ic"].to_numpy(dtype=np.float64)))
            fold_means.append(rank_mean)
            fold_rows.append(
                {
                    "model_family": family,
                    "config_id": config_id,
                    "fold_id": fold_id,
                    "months": len(fold),
                    "mean_rank_ic": rank_mean,
                    "mean_linear_ic": float(np.mean(fold["linear_ic"].to_numpy(dtype=np.float64))),
                }
            )
        rank_bootstrap = np.mean(rank_values[plan], axis=1, dtype=np.float64)
        linear_bootstrap = np.mean(linear_values[plan], axis=1, dtype=np.float64)
        mean_rank = float(np.mean(rank_values))
        positive = sum(value > 0.0 for value in fold_means)
        config_rows.append(
            {
                "model_family": family,
                "config_id": config_id,
                "months": len(months),
                "folds": len(fold_means),
                "mean_rank_ic": mean_rank,
                "mean_linear_ic": float(np.mean(linear_values)),
                "rank_ic_bootstrap_se": float(np.std(rank_bootstrap, ddof=1)),
                "linear_ic_bootstrap_se": float(np.std(linear_bootstrap, ddof=1)),
                "positive_fold_count": positive,
                "fold_dispersion": float(np.std(fold_means, ddof=1)),
                "gate_pass": bool(mean_rank > 0.0 and positive >= 3),
            }
        )
    fold = pd.DataFrame(fold_rows).sort_values(["model_family", "config_id", "fold_id"], kind="mergesort").reset_index(drop=True)
    config = pd.DataFrame(config_rows).sort_values(["model_family", "config_id"], kind="mergesort").reset_index(drop=True)
    return fold, config


def _regularization_key(row: pd.Series) -> tuple[Any, ...]:
    value = json.loads(str(row["regularization_key_json"]))
    _require(isinstance(value, list) and value, "Independent regularization key differs")
    return tuple(value)


def _tie(candidates: pd.DataFrame) -> pd.Series:
    return sorted((row for _, row in candidates.iterrows()), key=lambda row: (_regularization_key(row), str(row["config_id"])))[0]


def _choose(candidates: pd.DataFrame, column: str, *, maximum: bool) -> pd.Series:
    values = candidates[column].astype(float)
    optimum = float(values.max() if maximum else values.min())
    tied = candidates.loc[values >= optimum - _FLOAT_TOLERANCE] if maximum else candidates.loc[values <= optimum + _FLOAT_TOLERANCE]
    return _tie(tied)


def _independent_shortlist(config: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        family_frame = config.loc[config["model_family"] == family].copy()
        eligible = family_frame.loc[family_frame["gate_pass"].astype(bool)].copy()
        if eligible.empty:
            if family != "elastic_net":
                continue
            best = _choose(family_frame, "mean_rank_ic", maximum=True)
            selections = [(best, "benchmark_only_fallback")]
            fallback = True
        else:
            best = _choose(eligible, "mean_rank_ic", maximum=True)
            threshold = float(best["mean_rank_ic"]) - float(best["rank_ic_bootstrap_se"])
            one_se = eligible.loc[eligible["mean_rank_ic"].astype(float) >= threshold - _FLOAT_TOLERANCE]
            selections = [
                (best, "highest_mean_rank_ic"),
                (_choose(one_se, "complexity", maximum=False), "simplest_within_one_se"),
                (_choose(one_se, "fold_dispersion", maximum=False), "lowest_fold_dispersion_within_one_se"),
            ]
            fallback = False
        best_mean = float(best["mean_rank_ic"])
        best_se = float(best["rank_ic_bootstrap_se"])
        by_config: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for selected, reason in selections:
            config_id = str(selected["config_id"])
            if config_id in by_config:
                by_config[config_id]["selection_reasons"] += "|" + reason
                continue
            order.append(config_id)
            by_config[config_id] = {
                "model_family": family,
                "config_id": config_id,
                "shortlist_order": len(order),
                "selection_reasons": reason,
                "benchmark_only_fallback": fallback,
                "gate_pass": bool(selected["gate_pass"]),
                "mean_rank_ic": float(selected["mean_rank_ic"]),
                "rank_ic_bootstrap_se": float(selected["rank_ic_bootstrap_se"]),
                "positive_fold_count": int(selected["positive_fold_count"]),
                "fold_dispersion": float(selected["fold_dispersion"]),
                "complexity": float(selected["complexity"]),
                "family_best_mean_rank_ic": best_mean,
                "family_best_bootstrap_se": best_se,
                "one_se_threshold": best_mean - best_se,
            }
        rows.extend(by_config[config_id] for config_id in order)
    return pd.DataFrame.from_records(rows, columns=published_schemas()[SHORTLIST_NAME].names)


def _compare_frames(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    float_columns: Sequence[str],
    code: str,
) -> list[Difference]:
    left = expected.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)
    right = observed.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)
    differences: list[Difference] = []
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return [Difference(code + "_shape", {"rows": len(left), "columns": list(left.columns)}, {"rows": len(right), "columns": list(right.columns)}, "Table shape or schema differs")]
    for column in left.columns:
        if column in float_columns:
            a = left[column].to_numpy(dtype=np.float64)
            b = right[column].to_numpy(dtype=np.float64)
            mismatch = not np.allclose(a, b, rtol=0.0, atol=_FLOAT_TOLERANCE, equal_nan=False)
        else:
            mismatch = not left[column].astype(str).equals(right[column].astype(str))
        if mismatch:
            differences.append(Difference(code + "_" + column, "independent recomputation", "primary value differs", f"Mismatch in {column}"))
    return differences


def _verification_code_hashes() -> dict[str, str]:
    paths = (
        "src/portfolio_bot/data/secure_publication.py",
        "src/portfolio_bot/selectors/contract.py",
        "src/portfolio_bot/selectors/inputs.py",
        "src/portfolio_bot/selectors/publication.py",
        "src/portfolio_bot/selectors/independent_verification.py",
        "scripts/verify_stage_09_selectors.py",
    )
    return {path: sha256_file(PROJECT_ROOT / path) for path in paths}


def _verification_runtime_versions(
    receipt: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    observed = {
        name: importlib.metadata.version(name)
        for name in _VERIFICATION_RUNTIME_DISTRIBUTIONS
    }
    if receipt is not None:
        runtime = receipt.get("runtime")
        _require(isinstance(runtime, Mapping), "Search runtime receipt is missing")
        expected = runtime.get("versions")
        _require(
            isinstance(expected, Mapping),
            "Search runtime-version receipt is missing",
        )
        for name, version in observed.items():
            _require(
                str(expected.get(name)) == version,
                f"Verification runtime differs for {name}: "
                f"expected {expected.get(name)}, observed {version}",
            )
    return observed


def _verify_science(
    artifacts: AuthenticatedPrimary,
    tables: Mapping[str, pa.Table],
    receipt: Mapping[str, Any],
    *,
    progress: Callable[[str], None],
) -> tuple[dict[str, Any], list[Difference]]:
    differences: list[Difference] = []
    oof = tables[OOF_NAME].to_pandas()
    fit_checks, fit_differences = _independent_fit_grid(
        tables[FIT_RECEIPTS_NAME].to_pandas(),
        receipt,
        oof,
        tables[CONFIG_NAME].to_pandas(),
    )
    differences.extend(fit_differences)
    labels = _expected_oof_labels(artifacts.inputs)
    key_columns = ["fold_id", "signal_date", "holding_month", "permno"]
    observed_keys = oof[key_columns].sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    expected_keys = labels[key_columns]
    if not observed_keys.astype(str).equals(expected_keys.astype(str)):
        differences.append(Difference("oof_stage7_key_join", "exact verified Stage 7 OOF keys", "OOF keys differ", "OOF spine did not rejoin one-to-one"))
        return {}, differences
    oof = oof.sort_values(key_columns, kind="mergesort").reset_index(drop=True)

    progress("independent seed averaging and monthly score replay")
    raw_mean = _raw_means(oof, receipt)
    scores = _independent_scores(raw_mean, labels["holding_month"].to_numpy())
    _, score_registry = _column_registry(receipt)
    published_scores = oof[[str(item["column_name"]) for item in score_registry]].to_numpy(dtype=np.float32, copy=False)
    score_max_difference = float(np.max(np.abs(scores.astype(np.float64) - published_scores.astype(np.float64))))
    if score_max_difference > np.finfo(np.float32).eps:
        differences.append(Difference("monthly_scores", "seed-average then centered average rank", score_max_difference, "Published OOF scores differ"))

    progress("independent monthly IC recomputation")
    monthly = _independent_monthly(labels, raw_mean, scores, receipt)
    published_monthly = tables[MONTHLY_NAME].to_pandas()
    differences.extend(
        _compare_frames(
            monthly,
            published_monthly,
            key_columns=("model_family", "config_id", "holding_month"),
            float_columns=("rank_ic", "linear_ic"),
            code="monthly_metrics",
        )
    )

    progress("independent fold and bootstrap metric recomputation")
    fold, config_metrics = _independent_aggregate(monthly)
    published_fold = tables[FOLD_NAME].to_pandas()
    differences.extend(
        _compare_frames(
            fold,
            published_fold,
            key_columns=("model_family", "config_id", "fold_id"),
            float_columns=("mean_rank_ic", "mean_linear_ic"),
            code="fold_metrics",
        )
    )
    published_config = tables[CONFIG_NAME].to_pandas()
    metric_columns = list(config_metrics.columns)
    differences.extend(
        _compare_frames(
            config_metrics,
            published_config[metric_columns],
            key_columns=("model_family", "config_id"),
            float_columns=("mean_rank_ic", "mean_linear_ic", "rank_ic_bootstrap_se", "linear_ic_bootstrap_se", "fold_dispersion"),
            code="configuration_metrics",
        )
    )

    progress("independent one-SE family shortlist recomputation")
    ledger_columns = ["model_family", "config_id", "complexity", "trial_phase", "parent_config_id", "configuration_json", "regularization_key_json"]
    independent_config = config_metrics.merge(published_config[ledger_columns], on=["model_family", "config_id"], validate="one_to_one")
    shortlist = _independent_shortlist(independent_config)
    published_shortlist = tables[SHORTLIST_NAME].to_pandas()
    differences.extend(
        _compare_frames(
            shortlist,
            published_shortlist,
            key_columns=("model_family", "shortlist_order", "config_id"),
            float_columns=("mean_rank_ic", "rank_ic_bootstrap_se", "fold_dispersion", "complexity", "family_best_mean_rank_ic", "family_best_bootstrap_se", "one_se_threshold"),
            code="shortlist",
        )
    )

    core = independent_config.loc[(independent_config["model_family"] == "xgboost") & (independent_config["trial_phase"] == "core_grid")].copy()
    expected_parents: set[str] = set()
    remaining = core.copy()
    for _ in range(4):
        selected_parent = _choose(remaining, "mean_rank_ic", maximum=True)
        parent_id = str(selected_parent["config_id"])
        expected_parents.add(parent_id)
        remaining = remaining.loc[remaining["config_id"].astype(str) != parent_id]
    gamma = independent_config.loc[(independent_config["model_family"] == "xgboost") & (independent_config["trial_phase"] == "gamma_challenger")]
    observed_parents = set(gamma["parent_config_id"].astype(str))
    if observed_parents != expected_parents:
        differences.append(Difference("xgboost_gamma_parents", sorted(expected_parents), sorted(observed_parents), "Gamma challengers were not generated from sealed top-four core trials"))

    checks = {
        **fit_checks,
        "oof_stage7_keys_rejoined": len(labels),
        "oof_present_outcomes_rejoined": int(labels["outcome_valid"].sum()),
        "raw_forecast_columns_replayed": 538,
        "monthly_score_columns_replayed": 346,
        "score_max_absolute_difference": score_max_difference,
        "monthly_metric_rows_recomputed": len(monthly),
        "fold_metric_rows_recomputed": len(fold),
        "configuration_metric_rows_recomputed": len(config_metrics),
        "bootstrap_replicates_recomputed": 10_000,
        "shortlist_rows_recomputed": len(shortlist),
        "xgboost_gamma_parent_count_recomputed": len(expected_parents),
    }
    return checks, differences


def verification_checks_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("check_name", pa.string(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("observed_json", pa.string(), nullable=False),
            pa.field("expected_json", pa.string(), nullable=False),
            pa.field("detail", pa.string(), nullable=False),
        ]
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _checks_table(checks: Mapping[str, Any]) -> pa.Table:
    rows = [
        {
            "check_name": str(name),
            "status": "PASS",
            "observed_json": json.dumps(
                _json_safe(value),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "expected_json": json.dumps(
                _json_safe(value),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "detail": "Independent Stage 9 recomputation passed.",
        }
        for name, value in sorted(checks.items())
    ]
    return pa.Table.from_pylist(rows, schema=verification_checks_schema())


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=1024, compression="zstd", use_dictionary=True, version="2.6", write_statistics=True)
    return sink.getvalue().to_pybytes()


def _publish_verified(
    artifacts: AuthenticatedPrimary,
    checks: Mapping[str, Any],
    differences: Sequence[Difference],
    verification_runtime_versions: Mapping[str, str],
) -> tuple[str, Path, Path]:
    _require(not differences, "Cannot authorize Stage 10 with verification differences")
    code_hashes = _verification_code_hashes()
    checks_payload = _parquet_bytes(_checks_table(checks))
    identity = {
        "build_id": artifacts.build_id,
        "primary_manifest_sha256": artifacts.manifest_sha256,
        "primary_pointer_sha256": artifacts.pointer_sha256,
        "verification_code_hashes": code_hashes,
        "verification_runtime_versions": dict(verification_runtime_versions),
        "checks": _json_safe(checks),
        "verification_checks_sha256": _sha256_bytes(checks_payload),
    }
    verification_id = "verify-" + _sha256_bytes(canonical_json_bytes(identity))[:16]
    _require(_VERIFY_ID.fullmatch(verification_id) is not None, "Verification ID is malformed")
    report = {
        "schema_version": 1,
        "contract_id": artifacts.policy["contract_id"],
        "policy_id": artifacts.policy["policy_id"],
        "build_id": artifacts.build_id,
        "verification_id": verification_id,
        "status": "VERIFIED_PASSED",
        "primary_manifest_sha256": artifacts.manifest_sha256,
        "primary_pointer_sha256": artifacts.pointer_sha256,
        "verification_code_hashes": code_hashes,
        "verification_runtime_versions": dict(verification_runtime_versions),
        "checks": _json_safe(checks),
        "difference_count": 0,
        "verification_checks_sha256": _sha256_bytes(checks_payload),
        "authorizations": VERIFIED_AUTHORIZATIONS,
    }
    report_payload = canonical_json_bytes(report)
    success_payload = canonical_json_bytes(
        {
            "schema_version": 1,
            "build_id": artifacts.build_id,
            "verification_id": verification_id,
            "status": "VERIFICATION_BUNDLE_COMPLETE",
            "verification_report_sha256": _sha256_bytes(report_payload),
            "verification_checks_sha256": _sha256_bytes(checks_payload),
        }
    )
    payloads = {
        VERIFICATION_CHECKS_NAME: checks_payload,
        VERIFICATION_REPORT_NAME: report_payload,
        SUCCESS_NAME: success_payload,
    }
    artifact_root = project_path(artifacts.policy["publication"]["artifact_root"])
    destination = artifact_root / artifacts.policy["publication"]["verification_namespace"] / artifacts.build_id / verification_id

    def gate() -> None:
        artifacts.gate()
        _require(_verification_code_hashes() == code_hashes, "Verification code changed")
        _require(
            _verification_runtime_versions() == dict(verification_runtime_versions),
            "Verification runtime changed",
        )

    gate()
    with SecureStagingDirectory.create(trusted_root=artifact_root, destination=destination, expected_files=VERIFICATION_FILES, label="Stage 9 independent selector verification") as stage:
        for name in VERIFICATION_FILES:
            with stage.open_exclusive_binary(name) as stream:
                written = stream.write(payloads[name])
                _require(written == len(payloads[name]), f"Short verification write: {name}")
        hashes = {name: _sha256_bytes(payloads[name]) for name in VERIFICATION_FILES}
        snapshot = stage.snapshot(expected_hashes=hashes)
        stage.publish(expected_hashes=snapshot.hashes, pre_publish_gate=gate, post_publish_gate=gate)
    pointer = {
        "schema_version": 1,
        "contract_id": artifacts.policy["contract_id"],
        "policy_id": artifacts.policy["policy_id"],
        "build_id": artifacts.build_id,
        "verification_id": verification_id,
        "status": "VERIFIED_PASSED",
        "primary_manifest_sha256": artifacts.manifest_sha256,
        "verification_report_sha256": _sha256_bytes(report_payload),
        "verification_checks_sha256": _sha256_bytes(checks_payload),
        "authorizations": VERIFIED_AUTHORIZATIONS,
    }
    pointer_path = project_path(artifacts.policy["publication"]["latest_verified_pointer"])
    replace_pointer_last(
        pointer_path,
        canonical_json_bytes(pointer),
        trusted_root=PROJECT_ROOT,
        pre_replace_gate=gate,
        post_replace_gate=gate,
        label="Stage 9 latest selector verified pointer",
    )
    return verification_id, destination, pointer_path


def verify_stage9_selector_results(
    *,
    policy_path: Path = DEFAULT_POLICY,
    primary_pointer_path: Path | None = None,
    publish: bool = True,
    progress: Callable[[str], None] | None = None,
) -> VerificationResult:
    """Authenticate and independently replay every Stage 9 selector result."""

    report = progress or (lambda _message: None)
    with authenticate_primary(policy_path=policy_path, primary_pointer_path=primary_pointer_path) as artifacts:
        report("authenticated pointer, manifest, closed outputs, and Stage 7 inputs")
        tables, receipt = _authenticated_tables(artifacts)
        verification_runtime_versions = _verification_runtime_versions(receipt)
        checks: dict[str, Any] = {
            "primary_artifact_graph_authenticated": True,
            "primary_output_bundles_authenticated": 3,
            "primary_output_files_authenticated": 12,
            "stage9_input_tables_authenticated": len(artifacts.inputs.input_receipts),
            "validation_or_final_rows_read": 0,
            "portfolio_weights_or_pnl_created": 0,
            "verification_runtime_versions": verification_runtime_versions,
        }
        scientific_checks, differences = _verify_science(artifacts, tables, receipt, progress=report)
        checks.update(scientific_checks)
        checks["scientific_difference_count"] = len(differences)
        checks["verification_code_hashes"] = _verification_code_hashes()
        verification_id: str | None = None
        verification_path: Path | None = None
        verified_pointer_path: Path | None = None
        if not differences and publish:
            report("publishing immutable verification and Stage-10-only pointer")
            verification_id, verification_path, verified_pointer_path = _publish_verified(
                artifacts,
                checks,
                differences,
                verification_runtime_versions,
            )
        return VerificationResult(
            passed=not differences,
            build_id=artifacts.build_id,
            verification_id=verification_id,
            checks=checks,
            differences=tuple(differences),
            verification_path=verification_path,
            verified_pointer_path=verified_pointer_path,
        )


__all__ = [
    "AuthenticatedPrimary",
    "Difference",
    "PRIMARY_AUTHORIZATIONS",
    "SelectorVerificationError",
    "VERIFICATION_CHECKS_NAME",
    "VERIFICATION_FILES",
    "VERIFICATION_REPORT_NAME",
    "VERIFIED_AUTHORIZATIONS",
    "VerificationResult",
    "authenticate_primary",
    "verification_checks_schema",
    "verify_stage9_selector_results",
]
